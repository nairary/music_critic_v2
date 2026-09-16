from importlib.util import module_from_spec, spec_from_file_location
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.experiments.analysisgnn import harmonic_fusion_run as gate
from music_critic.experiments.analysisgnn.harmonic_fusion_training import FusionTrainingSession
from music_critic.experiments.analysisgnn.harmonic_logit_fusion import HarmonicLogitFusion


@pytest.fixture(scope="module")
def completed_gate(tmp_path_factory):
    path = tmp_path_factory.mktemp("fusion-run") / "completed"
    return path, gate.run_synthetic_gate(path)


def read_json(path):
    return json.loads(path.read_text())


def verify_manifest(path, filename="artifact-manifest.json"):
    manifest = read_json(path / filename)
    assert set(manifest) == {item.name for item in path.iterdir() if item.name != filename}
    for name, binding in manifest.items():
        data = (path / name).read_bytes()
        assert binding == {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)}


def _cli():
    path = Path(__file__).resolve().parents[2] / "scripts/run_harmonic_fusion_pilot.py"
    spec = spec_from_file_location("fusion_cli_under_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_requires_registration_before_execution(monkeypatch, tmp_path):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    with pytest.raises(SystemExit):
        _cli().main(["--smoke", "--device", "cuda", "--output-root", str(tmp_path / "uncreated")])
    assert not (tmp_path / "uncreated").exists()


def test_source_free_gate_requires_fresh_output_root(tmp_path):
    from music_critic.experiments.analysisgnn.harmonic_fusion_run import run_synthetic_gate
    with pytest.raises(FileExistsError):
        run_synthetic_gate(tmp_path)


def test_cli_has_no_unregistered_production_mode(tmp_path):
    with pytest.raises(SystemExit):
        _cli().main(["--request", "invented.json", "--output-root", str(tmp_path)])


def test_real_gate_requires_active_fusion_and_reports_all_heads(completed_gate):
    path, result = completed_gate
    assert result["passed"] is True
    assert result["actual_updates"] == 24 and result["actual_draws"] == 48
    assert result["progress"]["attempted_updates"] == result["progress"]["applied_updates"] == 24
    assert result["progress"]["attempted_draws"] == result["progress"]["applied_draws"] == 48
    for arm in result["arms"]:
        diagnostics = arm["learning_diagnostics"]
        assert diagnostics["model_state_changed"] is True
        assert len(diagnostics["per_head"]) == 18
        assert {"root", "quality", "cadence", "metrical_strength"} <= diagnostics["per_head"].keys()
        assert arm["validation_updates"] == [0, 3, 6]
        assert arm["actual_updates"] == 12 and arm["actual_draws"] == 24
        assert arm["interrupted_matches_uninterrupted"] is True
        for row in diagnostics["per_head"].values():
            assert {"initial_support", "final_support", "initial_masked_cross_entropy",
                    "final_masked_cross_entropy", "actual_fusion_residual_max_abs"} <= row.keys()
        if arm["arm"] == "fusion":
            change = diagnostics["fusion_weight_change"]
            assert change["changed_parameters"] > 0 and change["delta_l2"] > 0
            assert change["nonzero_output_head_count"] > 0
            assert any(change["optimizer_steps"].values())
            assert change["initial_state_fingerprint"] != change["final_state_fingerprint"]
        else:
            assert diagnostics["fusion_weight_change"] is None
            assert all(row["actual_fusion_residual_max_abs"] == 0 for row in diagnostics["per_head"].values())
    assert not (path / "failure-receipt.json").exists()
    assert result == read_json(path / "report.json")
    verify_manifest(path)


def test_per_head_improvement_is_diagnostic_not_an_added_acceptance_rule(completed_gate):
    _, result = completed_gate
    diagnostics = {**result["arms"][1]["learning_diagnostics"], "changed_supervised_heads": [],
                   "improved_supervised_heads": []}
    gate._require_learning(diagnostics, enabled=True)


def test_completed_progress_is_durable_and_separates_recomputed_trajectories(completed_gate):
    path, result = completed_gate
    events = [json.loads(line) for line in (path / "progress.jsonl").read_text().splitlines()]
    assert [row["event_index"] for row in events] == list(range(len(events)))
    starts = [row for row in events if row["stage"] == "update" and row["status"] == "started"]
    ends = [row for row in events if row["stage"] == "update" and row["status"] == "completed"]
    assert len(starts) == len(ends) == 24
    assert all(len(row["draws"]) == 2 for row in starts)
    assert [row["applied_updates"] for row in result["progress"]["trajectories"]] == [6, 3, 3, 6, 3, 3]
    assert [row["start_update"] for row in result["progress"]["trajectories"]] == [0, 0, 3, 0, 0, 3]
    assert all(row["status"] == "completed" for row in result["progress"]["trajectories"])
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    with pytest.raises(FileExistsError):
        gate.run_synthetic_gate(path)
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}


def test_actual_identity_fusion_mutation_fails_and_preserves_completed_control(monkeypatch, tmp_path):
    # Exercise actual model/optimizer trajectories with the fusion operation
    # removed. Resume identity alone would incorrectly accept this mutation.
    monkeypatch.setattr(HarmonicLogitFusion, "forward", lambda self, predictions: predictions)
    path = tmp_path / "identity-fusion"
    with pytest.raises(ValueError, match="missing_fusion_learning_signal"):
        gate.run_synthetic_gate(path)
    report = read_json(path / "failure-receipt.json")
    assert report["passed"] is False
    assert report["failure"]["stage"] == "learning_check"
    assert report["failure"]["arm"] == "fusion"
    assert report["failure"]["trajectory"] == "uninterrupted"
    assert report["actual_updates"] == report["progress"]["attempted_updates"] == 18
    assert report["actual_draws"] == 36
    diagnostics = read_json(path / "fusion-learning-diagnostics.json")
    assert diagnostics["model_state_changed"] is True
    assert diagnostics["fusion_weight_change"]["changed_parameters"] == 0
    assert diagnostics["fusion_weight_change"]["nonzero_output_head_count"] == 0
    assert not any(diagnostics["fusion_weight_change"]["optimizer_steps"].values())
    assert (path / "control-update-000006.pt").is_file()
    assert not (path / "report.json").exists()
    verify_manifest(path)


@pytest.mark.parametrize("failure_stage,expected_applied,expected_attempted,expected_draws", [
    ("update", 6, 7, 12), ("validation", 6, 6, 12), ("restore", 9, 9, 18),
])
def test_later_failure_receipt_keeps_completed_full6_and_stage(monkeypatch, tmp_path,
    failure_stage, expected_applied, expected_attempted, expected_draws):
    if failure_stage == "update":
        original = FusionTrainingSession.step
        calls = 0

        def step(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 7:
                raise RuntimeError("injected later update failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(FusionTrainingSession, "step", step)
    elif failure_stage == "validation":
        original = gate.evaluate_validation
        calls = 0

        def validate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("injected later validation failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(gate, "evaluate_validation", validate)
    else:
        def restore(*args, **kwargs):
            raise RuntimeError("injected later restore failure")

        monkeypatch.setattr(gate, "load_training_checkpoint", restore)
    path = tmp_path / failure_stage
    with pytest.raises(RuntimeError, match=f"injected later {failure_stage} failure"):
        gate.run_synthetic_gate(path)
    report = read_json(path / "failure-receipt.json")
    assert report["passed"] is False and report["failure"]["stage"] == failure_stage
    assert report["failure"]["arm"] == "control"
    assert report["failure"]["trajectory"] == ("restored" if failure_stage == "restore" else "interrupted")
    assert report["progress"]["applied_updates"] == report["actual_updates"] == expected_applied
    assert report["progress"]["attempted_updates"] == expected_attempted
    assert report["progress"]["attempted_draws"] == expected_attempted * 2
    assert report["progress"]["applied_draws"] == expected_applied * 2
    assert report["progress"]["actual_draws"] == report["actual_draws"] == expected_draws
    full = report["progress"]["trajectories"][0]
    assert full["trajectory"] == "uninterrupted" and full["status"] == "completed"
    assert full["applied_updates"] == 6 and full["applied_draws"] == 12
    assert not (path / "report.json").exists()
    verify_manifest(path)
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    with pytest.raises(FileExistsError):
        gate.run_synthetic_gate(path)
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}


@pytest.mark.parametrize("stage", ["manifest", "report_fsync"])
def test_finalization_failure_cannot_publish_a_success_report(monkeypatch, tmp_path, stage):
    if stage == "manifest":
        original = gate._manifest

        def manifest(*args, **kwargs):
            if kwargs.get("pending_files"):
                raise OSError("injected manifest failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(gate, "_manifest", manifest)
    else:
        original = gate._write_bytes_once

        def fail_fsync(*args):
            raise OSError("injected report_fsync failure")

        def write(path, content):
            if Path(path).name == "report.json":
                with monkeypatch.context() as context:
                    context.setattr(gate.os, "fsync", fail_fsync)
                    return original(path, content)
            return original(path, content)

        monkeypatch.setattr(gate, "_write_bytes_once", write)
    path = tmp_path / stage
    with pytest.raises(OSError, match=f"injected {stage} failure"):
        gate.run_synthetic_gate(path)
    assert not (path / "report.json").exists()
    failure = read_json(path / "failure-receipt.json")
    assert failure["passed"] is False and failure["failure"]["stage"] == "report"
    assert failure["actual_updates"] == 24 and failure["actual_draws"] == 48
    assert not list(path.glob(".report.json.*"))
    verify_manifest(path, "failure-artifact-manifest.json" if stage == "report_fsync" else "artifact-manifest.json")


def test_primary_exception_survives_receipt_write_failure(monkeypatch, tmp_path):
    error = ValueError("primary update failure")

    def fail_update(*args, **kwargs):
        raise error

    original = gate._write_json_once

    def write(path, payload):
        if Path(path).name == "failure-receipt.json":
            raise OSError("injected receipt storage failure")
        return original(path, payload)

    monkeypatch.setattr(FusionTrainingSession, "step", fail_update)
    monkeypatch.setattr(gate, "_write_json_once", write)
    path = tmp_path / "receipt-failure"
    with pytest.raises(ValueError, match="primary update failure") as raised:
        gate.run_synthetic_gate(path)
    assert raised.value is error
    assert isinstance(raised.value.__cause__, OSError)
    assert not (path / "report.json").exists()
    assert not (path / "failure-receipt.json").exists()
    events = [json.loads(line) for line in (path / "progress.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "failed"
    assert events[-1]["progress"]["attempted_updates"] == 1
    assert events[-1]["progress"]["applied_updates"] == 0
