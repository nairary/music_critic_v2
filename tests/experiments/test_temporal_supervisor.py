from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from music_critic.experiments.registry import canonical_json, load_record, seal_record
from scripts import supervise_temporal_campaign as s


def record():
    return seal_record(dict(schema_version="1.0.0", experiment_id="EXP-TEMPORAL-FIXTURE-001",
        kind="engineering_gate", phase="9E", title="Source-free supervisor fixture",
        status="planned", question="Does supervision preserve evidence?", hypothesis="It does",
        decision_use="Fixture only", repository={"git_commit": "a"*40}, seeds=[17], budget={},
        data_access={"access_evidence_status": "not_applicable_planning_record",
            **{k: None for k in ("validation_accessed", "test_inputs_accessed", "test_targets_accessed",
                                 "test_metrics_computed", "test_used_for_selection")},
            "declared_policy": {"validation_allowed": True, "test_inputs_allowed": False,
                "test_targets_allowed": False, "test_metrics_allowed": False,
                "test_selection_allowed": False, "source": "source-free fixture"}},
        results={}, validity={}, claims={"not_supported": ["model-quality evidence"]},
        artifacts=[], import_provenance={"mode": "fixture"}))


@pytest.fixture
def config(tmp_path):
    project = tmp_path/"project"
    records = project/"docs/experiments/records"
    records.mkdir(parents=True)
    master = records/"EXP-TEMPORAL-FIXTURE-001.json"
    value = record()
    master.write_text(canonical_json(value)+"\n")
    output = tmp_path/"evidence"
    planned = output/"planned-input"/master.name
    planned.parent.mkdir(parents=True)
    planned.write_bytes(master.read_bytes())
    return s.validate_config(dict(command=["fixture-python", "-m", "fixture"], cwd=str(project),
        env={}, output_root=str(output), run_output=str(output/"run"),
        record_path=str(master), project_root=str(project), planned_record_input=str(planned),
        artifact_locator_prefix="external-evidence/fixture", gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        mode="train", expected_updates=2))


def journal_row(update):
    return dict(update=update, draws=[{"record_id": "a", "shift_pc": 0}, {"record_id": "b", "shift_pc": 2}],
                loss=.5, lr=.001, seconds=.1)


def completed_files(config):
    output = Path(config["run_output"])
    output.mkdir()
    (output/"training_metrics.jsonl").write_text("".join(json.dumps(journal_row(u))+"\n" for u in (1, 2)))
    checkpoint = output/"checkpoint-000002.pt"
    checkpoint.write_bytes(b"fixture checkpoint, not executable")
    s.write_json(output/"latest-checkpoint.json", dict(path=str(checkpoint), update=2, **s.digest(checkpoint)))
    s.write_json(output/"report.json", dict(completed=True, interrupted=False, start_update=0,
        applied_updates=2, final_update=2, cumulative_draws=4, test_accessed=False,
        validation_accessed=False, best=None))


def receipt(status="exited_zero"):
    return dict(process_status=status, reason="child_exit", wall_seconds=1.)


def test_bounds_defaults_and_rejection(config):
    assert config["limits"]["wall_seconds"] == 36*3600
    assert config["limits"]["poll_seconds"] == 5
    for limits in ({"gpu_memory_mib": 16385}, {"host_available_mib": 8191},
                   {"swap_growth_mib": 257}, {"poll_seconds": .5}, {"wall_seconds": 36*3600+1}):
        with pytest.raises(ValueError, match="resource_limits"):
            s.validate_config(dict(config, limits=limits))
    with pytest.raises(ValueError, match="separate_planned"):
        s.validate_config(dict(config, planned_record_input=config["record_path"]))


def test_default_ownership_fixture_is_the_pinned_accepted_implementation():
    project = Path(__file__).resolve().parents[2]
    module = s.load_ownership(project/"tests/fixtures/research_supervision/accepted_context_wrapper.py")
    assert callable(module._observe_child_exit) and callable(module._cleanup_owned_group)


@pytest.mark.parametrize("change,expected", [({"gpu_memory_mib": 17000}, "gpu_memory_ceiling"),
    ({"available_kib": 8000*1024}, "host_memory_floor"),
    ({"swap_used_kib": 257*1024}, "swap_growth"), ({}, None)])
def test_resource_stop_classification(config, change, expected):
    sample = dict(gpu_memory_mib=3000, available_kib=32000*1024, swap_used_kib=0)
    sample.update(change)
    assert s.resource_violation(sample, 0, config["limits"]) == expected


def test_journal_preserves_valid_prefix_on_truncation_and_duplicate(tmp_path):
    p = tmp_path/"training.jsonl"
    p.write_text(json.dumps(journal_row(1))+"\n"+json.dumps(journal_row(2))[:-3])
    result = s.read_training_journal(p)
    assert result["rows"] == 1 and result["draws_lower_bound"] == 2 and result["partial"]
    p.write_text(json.dumps(journal_row(1))+"\n"+json.dumps(journal_row(1))+"\n")
    result = s.read_training_journal(p)
    assert result["rows"] == 1 and result["partial"]
    assert s.read_training_journal(tmp_path/"absent")["partial"]


def test_complete_report_requires_journal_and_checkpoint(config):
    completed_files(config)
    result = s.inspect_result(config, receipt())
    assert result["status"] == "completed"
    assert result["accounting"]["applied_updates"] == 2
    assert result["accounting"]["draws"] == 4
    (Path(config["run_output"])/"checkpoint-000002.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint_binding"):
        s.inspect_result(config, receipt())


def test_report_only_never_establishes_complete_training(config):
    completed_files(config)
    (Path(config["run_output"])/"training_metrics.jsonl").unlink()
    with pytest.raises(ValueError, match="journal_mismatch"):
        s.inspect_result(config, receipt())


def test_absent_report_access_stays_unknown(config):
    Path(config["run_output"]).mkdir()
    result = s.inspect_result(config, receipt("failed"))
    assert result["status"] == "failed"
    assert all(value is None for value in result["access"].values())
    assert result["accounting"]["applied_updates"] is None


def test_finalize_seals_artifacts_preserves_planned_and_updates_ledger(config):
    completed_files(config)
    supervisor = Path(config["output_root"])/"supervisor"
    supervisor.mkdir()
    s.write_json(supervisor/"resource-receipt.json", receipt())
    initial = load_record(config["record_path"])
    planned = Path(config["planned_record_input"]).read_bytes()
    running = s.transition(Path(config["record_path"]), initial["record_fingerprint"], "running")
    final = s.finalize(config, running, receipt())
    assert final["status"] == "completed"
    assert final["budget"]["actual"]["draws"] == 4
    assert len(final["artifacts"]) >= 6
    assert Path(config["planned_record_input"]).read_bytes() == planned
    assert "EXP-TEMPORAL-FIXTURE-001" in (Path(config["project_root"])/"docs/EXPERIMENT_LEDGER.md").read_text()
    assert "completed" in (Path(config["project_root"])/"docs/STATUS.md").read_text()
    with pytest.raises(ValueError, match="immutable"):
        s.transition(Path(config["record_path"]), final["record_fingerprint"], "failed")


def test_failed_evidence_preserves_journal_lower_bound_as_invalid(config):
    completed_files(config)
    (Path(config["run_output"])/"checkpoint-000002.pt").write_bytes(b"bad")
    (Path(config["output_root"])/"supervisor").mkdir()
    final = s.finalize(config, load_record(config["record_path"]), receipt())
    assert final["status"] == "invalid"
    assert final["budget"]["actual"]["applied_updates_lower_bound"] == 2
    assert final["budget"]["actual"]["applied_updates"] is None


def test_concurrent_master_edit_cannot_be_overwritten(config):
    original = load_record(config["record_path"])
    changed = dict(original, question="Changed concurrently")
    Path(config["record_path"]).write_text(canonical_json(seal_record(changed))+"\n")
    with pytest.raises(ValueError, match="concurrently"):
        s.transition(Path(config["record_path"]), original["record_fingerprint"], "running")


class FakeOwnership:
    GPU_UUID = re.compile(r"GPU-[0-9a-f-]+")
    class ResourceStop(Exception):
        def __init__(self, reason, message):
            self.reason = reason
            super().__init__(message)
    def _require_nonreaping_runtime(self):
        pass
    def query_gpu_memory(self, *args):
        return 1000
    def _observe_child_exit(self, child):
        return 0
    def _cleanup_owned_group(self, child, lifecycle):
        lifecycle.update(ownership_confirmed_before_cleanup=True, leader_reaped_after_cleanup=True)
        return 0


def test_supervisor_full_lifecycle_fixture_and_initial_record_copy(config, monkeypatch):
    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == config["gpu_uuid"]
        assert load_record(config["planned_record_input"])["status"] == "planned"
        completed_files(config)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(s.subprocess, "Popen", launch)
    monkeypatch.setattr(s, "host_memory", lambda: dict(available_kib=32000*1024, swap_used_kib=0))
    assert s.supervise(config, ownership=FakeOwnership()) == 0
    assert load_record(config["record_path"])["status"] == "completed"
    assert load_record(config["planned_record_input"])["status"] == "planned"


def test_spawn_failure_still_terminalizes_record(config, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("fixture startup failure")
    monkeypatch.setattr(s.subprocess, "Popen", failed)
    monkeypatch.setattr(s, "host_memory", lambda: dict(available_kib=32000*1024, swap_used_kib=0))
    assert s.supervise(config, ownership=FakeOwnership()) == 1
    assert load_record(config["record_path"])["status"] == "failed"


def test_graceful_stop_signals_only_owned_trainer_before_group_cleanup(monkeypatch):
    calls = []
    owner = FakeOwnership()
    observations = iter((None, 0))
    owner._observe_child_exit = lambda child: next(observations)
    owner._cleanup_owned_group = lambda child, lifecycle: calls.append(("cleanup", child.pid)) or 0
    monkeypatch.setattr(s.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(s.os, "kill", lambda pid, sig: calls.append(("signal", pid, sig)))
    lifecycle = {}
    assert s.stop_gracefully(SimpleNamespace(pid=123), owner, lifecycle, 30) == 0
    assert calls == [("signal", 123, s.signal.SIGTERM), ("cleanup", 123)]
    assert lifecycle["trainer_sigterm_sent"]
