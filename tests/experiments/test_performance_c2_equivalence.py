"""Source-free C2 tests; CUDA unit smoke is skipped on the local CPU runtime."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch

from music_critic.experiments.analysisgnn import performance_c2_equivalence as gate
from music_critic.experiments.analysisgnn import performance_prefetch_replay as prefetch
from music_critic.experiments.analysisgnn import performance_replay as existing
from music_critic.experiments.analysisgnn import performance_training as optimized
from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
from music_critic.experiments.registry import load_record


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_performance_c2_fixture_gate.py"
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA runtime unavailable"))]


def _load_script():
    spec = importlib.util.spec_from_file_location("c2_gate_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbid_production(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic C2 gate touched production data/cache")
    monkeypatch.setattr(existing.reference, "load_production_record", forbidden)
    monkeypatch.setattr(existing.reference, "production_component_records", forbidden)
    monkeypatch.setattr(prefetch, "WarmTrainBatchPreparer", forbidden)
    monkeypatch.setattr(existing, "PreparedGraphCache", forbidden)


def test_fixed_protocol_is_separate_c2_plan_and_never_constructs_model(monkeypatch):
    from music_critic.experiments.analysisgnn import corrected_model
    before = gate._rng_fingerprint("cpu")
    def forbidden(*args, **kwargs):
        pytest.fail("model created while planning")
    monkeypatch.setattr(corrected_model, "CorrectedAnalysisGNNModel", forbidden)
    protocol = gate.build_protocol("cpu")
    jobs = gate._jobs(protocol)
    assert len(jobs) == 12 and protocol["planned_arm_count"] == 9
    assert protocol["planned_updates"] == 108 and protocol["planned_record_draws"] == 216
    assert protocol["full_protocol_exercised"] is True
    assert protocol["comparison_kind"] == "historical_c2_equivalence"
    assert [job.applied_shifts for job in jobs] == [(s, (s+5) % 12) for s in range(12)]
    assert all(job.applied_shifts == job.schedule_shifts for job in jobs)
    assert jobs[6].record_ids[0] == jobs[6].record_ids[1]
    assert protocol["historical_profile_id"] == "music-critic-v2-corrected-full-orbit-transposition-v1"
    assert gate._rng_fingerprint("cpu") == before
    with pytest.raises(ValueError, match="descriptor_binding_mismatch"):
        prefetch.jobs_from_prefetch_descriptor(protocol["schedule"])
    corrupt = deepcopy(protocol)
    corrupt["schedule"]["jobs"][0]["applied_shifts"][0] = 11
    with pytest.raises(gate.C2EquivalenceError, match="schedule_seal_mismatch"):
        gate._jobs(corrupt)


def test_fixture_has_closed_supported_labels_for_every_head_and_shift(monkeypatch):
    _forbid_production(monkeypatch)
    observed = gate.fixture_metadata_preflight()
    assert observed["task_shift_checks"] == 216 and observed["dialect"] == "dlc"
    assert {(row["task"], row["shift"]) for row in observed["rows"]} == {
        (task, shift) for task in existing.reference.ACTIVE_HEADS for shift in range(12)}


def test_independent_oracle_covers_all_shifts_and_rejects_feature_complete_track_updates(monkeypatch):
    _forbid_production(monkeypatch)
    for job in gate._jobs(gate.build_protocol("cpu")):
        source, _ = gate._unshifted_fixture(job)
        before = gate.raw_fingerprint(source)
        shifted = existing.reference.transpose_raw_graph_batch(source, job.applied_shifts)
        details = gate.assert_historical_raw_view(source, shifted, job.applied_shifts)
        assert details["historical_track_statistics_preserved"]
        assert details["nonzero_pitched_notes_checked"] > 0
        assert gate.raw_fingerprint(source) == before
    job = gate._jobs(gate.build_protocol("cpu"))[6]
    source, _ = gate._unshifted_fixture(job)
    shifted = existing.reference.transpose_raw_graph_batch(source, job.applied_shifts)
    first = shifted.to_data_list()[0]
    pitch_index = first["note"].cat_feature_names.index("pitch")
    assert first["note"].x_cat[:, pitch_index].tolist() == [66, 70]
    # Recomputing the historical stale track statistics must fail even if both
    # candidate and reference accidentally switched to a new transform.
    changed = deepcopy(shifted)
    changed["track"].x_cont[0, changed["track"].cont_feature_names[0].index("mean_pitch")] += 6
    with pytest.raises(gate.C2EquivalenceError, match="historical_track_statistics_changed"):
        gate.assert_historical_raw_view(source, changed, job.applied_shifts)
    changed = deepcopy(shifted)
    changed._inc_dict["note"]["x_cat"] = changed._inc_dict["note"]["x_cat"] + 1
    with pytest.raises((gate.C2EquivalenceError, ValueError)):
        gate.assert_historical_raw_view(source, changed, job.applied_shifts)


def test_oracle_preserves_identity_early_return_and_drum_relative_rule():
    from torch_geometric.data import Batch
    job = gate._jobs(gate.build_protocol("cpu"))[0]
    source, _ = gate._unshifted_fixture(job)
    graphs = source.to_data_list()
    # Deliberately distinguish identity deepcopy from a fresh recomputation.
    first = graphs[0]["note"]
    relative = first.cont_feature_names.index("track_relative_pitch")
    first.x_cont[:, relative] = 7
    first.x_cont_available[:, relative] = True
    drum = graphs[1]["note"].cat_feature_names.index("is_percussion")
    graphs[1]["note"].x_cat[0, drum] = 1
    source = Batch.from_data_list(graphs)
    actual = existing.reference.transpose_raw_graph_batch(source, (0, 6))
    gate.assert_historical_raw_view(source, actual, (0, 6))
    assert actual.to_data_list()[0]["note"].x_cont[:, relative].tolist() == [7, 7]
    # For the mixed synthetic owner, the drum pitch stays60 and other pitch
    # becomes70. The frozen relative path includes both, giving[-1,+1].
    assert actual.to_data_list()[1]["note"].x_cont[:, relative].tolist() == [-1, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_small_c2_update_prefetch_resume_has_exact_full_state_and_native_alignment_once(tmp_path, monkeypatch, device):
    _forbid_production(monkeypatch)
    for name in gate.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    if device == "cpu":
        monkeypatch.setattr(existing, "_assert_idle_gpu", lambda: pytest.fail("CPU touched GPU"))
        monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU read CUDA RNG"))
    protocol = gate._plan(device, shifts=(0, 6, 11), worker_counts=(0, 1), resume_after=1)
    before = gate._rng_fingerprint(device)
    result = gate._execute(tmp_path / "smoke", experiment_id="UNIT-C2", protocol=protocol)
    assert gate._rng_fingerprint(device) == before
    assert result["equivalence_passed"] and result["full_protocol_exercised"] is False
    assert result["actual_attempted_updates"] == result["actual_applied_updates"] == 15
    assert result["actual_measured_draws"] == 30
    assert len({row["trace_fingerprint"] for row in result["arms"]}) == 1
    for arm in result["arms"]:
        rows = [json.loads(line) for line in (tmp_path / "smoke" / arm["arm_id"] / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        assert all(row["forward_calls"] == row["native_alignment_calls"] == 1 for row in rows)
        assert all(len(row["alignment_rows"]) == 18 for row in rows)
        assert all(row["historical_track_statistics_preserved"] for row in rows)
        if arm["resume"]:
            assert (tmp_path / "smoke" / arm["arm_id"] / "own-checkpoint.pt").is_file()
            assert arm["fresh_state_differed_before_restore"] is True
            assert arm["checkpoint_training_state_fingerprint"] == arm["restored_training_state_fingerprint"]
            assert arm["fresh_training_state_fingerprint"] != arm["restored_training_state_fingerprint"]
            if arm["worker_count"]:
                assert arm["pending_at_resume"] > 0


def test_full_twelve_shift_cpu_reference_optimized_direct_and_resume(tmp_path, monkeypatch):
    _forbid_production(monkeypatch)
    protocol = gate._plan("cpu", worker_counts=(0,))
    result = gate._execute(tmp_path / "all_shifts", experiment_id="UNIT-C2-12", protocol=protocol)
    assert result["actual_applied_updates"] == 36 and result["actual_measured_draws"] == 72
    assert len({row["trace_fingerprint"] for row in result["arms"]}) == 1
    assert result["full_protocol_exercised"] is False  # Not a substitute for all-worker CUDA acceptance.


def test_numerical_mismatch_stops_and_keeps_partial_negative_evidence(tmp_path, monkeypatch):
    protocol = gate._plan("cpu", shifts=(0, 6), worker_counts=(0,), resume_after=1)
    update = optimized.attempt_performance_update
    def mismatch(**kwargs):
        outcome = update(**kwargs)
        return replace(outcome, gradient_norm=outcome.gradient_norm + 1)
    monkeypatch.setattr(optimized, "attempt_performance_update", mismatch)
    with pytest.raises(gate.C2EquivalenceError, match="exact_mismatch:optimized-w0-uninterrupted:job0:outcome"):
        gate._execute(tmp_path / "failed", experiment_id="UNIT-C2-FAIL", protocol=protocol)
    failure = json.loads((tmp_path / "failed/failure.json").read_text())
    assert failure["completed"] is False and failure["equivalence_passed"] is False
    assert failure["actual_attempted_updates"] == failure["actual_applied_updates"] == 3
    assert len(failure["completed_arms"]) == 1
    assert (tmp_path / "failed/optimized-w0-uninterrupted/trace.jsonl").is_file()


def test_second_native_alignment_is_rejected_without_an_extra_forward(tmp_path, monkeypatch):
    protocol = gate._plan("cpu", shifts=(0, 6), worker_counts=(0,), resume_after=1)
    align = optimized.align_performance_target_sidecars
    def duplicated(*args, **kwargs):
        align(*args, **kwargs)
        return align(*args, **kwargs)
    monkeypatch.setattr(optimized, "align_performance_target_sidecars", duplicated)
    with pytest.raises(gate.C2EquivalenceError, match="single_native_forward_alignment_required"):
        gate._execute(tmp_path / "double", experiment_id="UNIT-C2-DOUBLE", protocol=protocol)
    failure = json.loads((tmp_path / "double/failure.json").read_text())
    assert failure["actual_applied_updates"] == 3


@pytest.mark.parametrize("domain", ("all", "optimizer", "rng"))
def test_resume_requires_restoring_fresh_model_adam_and_rng_domains(tmp_path, monkeypatch, domain):
    protocol = gate._plan("cpu", shifts=(0, 6), worker_counts=(0,), resume_after=1)
    restore = existing._restore
    def incomplete(snapshot, model, optimizer, scheduler, scaler):
        before = existing._snapshot(model, optimizer, scheduler, scaler)
        if domain == "all":
            return
        restore(snapshot, model, optimizer, scheduler, scaler)
        if domain == "optimizer":
            optimizer.load_state_dict(before["optimizer"])
        else:
            existing._restore_rng(before["rng"])
    monkeypatch.setattr(existing, "_restore", incomplete)
    with pytest.raises(gate.C2EquivalenceError, match="restored_full_training_state_mismatch"):
        gate._execute(tmp_path / "incomplete", experiment_id="UNIT-RESTORE", protocol=protocol)
    failure = json.loads((tmp_path / "incomplete/failure.json").read_text())
    assert failure["actual_applied_updates"] == 5
    assert (tmp_path / "incomplete/optimized-w0-resumed/own-checkpoint.pt").is_file()


@pytest.mark.parametrize("failure", ("scaler_check", "hook_registration"))
def test_observation_hooks_are_removed_when_setup_fails(monkeypatch, failure):
    protocol = gate._plan("cpu", shifts=(0, 6), worker_counts=(0,), resume_after=1)
    progress = {"attempted_updates": 0, "completed_applied_updates": 0,
                "skipped_updates": 0, "applied_count_exact": True}
    with existing._preserve_runtime(False):
        components = prefetch._components("cpu")
        model, _, _, scaler, _, _ = components
        stream = prefetch.BatchStream(gate._jobs(protocol), prefetch.FixturePreparer(),
            source_fingerprint="unit-cleanup", config=prefetch.PrefetchConfig(), synthetic=True)
        if failure == "scaler_check":
            monkeypatch.setattr(scaler, "get_scale", lambda: 2.0)
            error = "disabled_fp32_scaler_required"
        else:
            def broken_hook(*args, **kwargs):
                raise ValueError("injected hook registration failure")
            monkeypatch.setattr(next(model.parameters()), "register_hook", broken_hook)
            error = "injected hook registration failure"
        try:
            with pytest.raises(ValueError, match=error):
                gate._observed_step(stream, components, strategy="original", device="cpu", progress=progress)
            assert not model._forward_hooks
            assert all(not parameter._backward_hooks for parameter in model.parameters())
            assert progress["attempted_updates"] == 0
            assert stream.state_dict()["committed_batch_cursor"] == 0
        finally:
            gate._close_stream(stream, "cpu")


def test_masked_missing_and_unsupported_rows_agree_after_nonzero_transform():
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    job = gate._jobs(gate.build_protocol("cpu"))[6]
    prepared = prefetch.FixturePreparer()(job)
    sidecars = deepcopy(prepared.sidecars)
    sidecars[0]["entities"][0]["targets"]["root"]["masked"] = True
    sidecars[1]["relations"][0]["target_entity_id"] = "missing:synthetic"
    for sidecar in sidecars:
        sidecar.pop("fingerprint")
        sidecar["fingerprint"] = existing.fingerprint(sidecar)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel().eval()
        with torch.no_grad():
            output = model(prepared.raw_graph_batch)
        reference = existing.reference.align_target_sidecars_after_prediction(
            output, prepared.raw_graph_batch, sidecars, shifts=job.applied_shifts)
        candidate = optimized.align_performance_target_sidecars(
            output, prepared.raw_graph_batch, sidecars, shifts=job.applied_shifts)
        assert gate._tensor_fingerprint(reference) == gate._tensor_fingerprint(candidate)
        assert not reference.heads["root"].valid_mask.any()
        assert reference.diagnostics
        weights = existing.reference.load_frozen_class_weights()
        task = "note_degree"
        weights.supported[task][reference.heads[task].values[0]] = False
        first = existing.reference.corrected_supervised_loss(output, reference, weights)
        second = existing.reference.corrected_supervised_loss(output, candidate,
            optimized.prepare_performance_class_weights(weights, "cpu"))
        assert gate._tensor_fingerprint(first) == gate._tensor_fingerprint(second)
        assert first.heads[task].unsupported_row_count > 0


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"), ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# fixture\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    run_registered_replay(experiment_id="EXP-PARENT", output_root=root / "outputs/parent",
                          protocol={"mode": "fixture"}, execute=lambda: {"equivalence_passed": True}, repository_root=root)
    return root


def _fake_result(protocol):
    arms = []
    for arm in protocol["arm_manifest"]:
        arms.append({**arm, "repeat": 0, "batch_size": 2, "exact": True,
            "window": {"records": 24}, "applied_updates": 12,
            **({"checkpoint_sha256": "f"*64, "resume_prefix_rows": 5,
                "fresh_state_differed_before_restore": True,
                "restored_training_state_fingerprint": "a"*64,
                "checkpoint_training_state_fingerprint": "a"*64,
                "pending_at_resume": 3 if arm["worker_count"] else 0} if arm["resume"] else {})})
    return {**protocol, "equivalence_passed": True, "arms": arms, "actual_applied_updates": 108,
            "actual_attempted_updates": 108, "actual_measured_draws": 216, "skipped_updates": 0}


def test_registered_wrapper_plans_before_work_binds_nine_truthful_arms_and_hashes_checkpoint(repo, monkeypatch):
    record_path = repo / "docs/experiments/records/EXP-C2.json"
    def fake(output_root, *, experiment_id, device, expected_protocol):
        assert load_record(record_path)["status"] == "running"
        output_root.mkdir(parents=True)
        (output_root / "own.pt").write_bytes(b"synthetic own checkpoint")
        return _fake_result(expected_protocol)
    monkeypatch.setattr(gate, "run_c2_equivalence", fake)
    result = gate.run_registered_c2_gate(experiment_id="EXP-C2", parent_experiment_id="EXP-PARENT",
        output_root=repo / "outputs/c2", device="cpu", repository_root=repo)
    record = load_record(record_path)
    assert record["status"] == "completed" and record["budget"]["planned"] == record["budget"]["actual"] == 216
    assert result["actual_applied_updates"] == 108
    assert "own.pt" in {row["role"] for row in record["artifacts"]}
    with pytest.raises(ValueError, match="fresh_experiment"):
        gate.run_registered_c2_gate(experiment_id="EXP-C2", parent_experiment_id="EXP-PARENT",
            output_root=repo / "outputs/c2", device="cpu", repository_root=repo)


@pytest.mark.parametrize("problem", ("failure", "missing_arm", "fake_resume", "reduced_scope", "different_weights"))
def test_registered_failures_or_incomplete_coverage_cannot_complete(repo, monkeypatch, problem):
    def fake(output_root, **kwargs):
        assert load_record(repo / "docs/experiments/records/EXP-C2.json")["status"] == "running"
        output_root.mkdir(parents=True)
        (output_root / "partial.pt").write_bytes(b"partial own checkpoint")
        if problem == "failure":
            raise RuntimeError("partial resume failed")
        result = _fake_result(kwargs["expected_protocol"])
        if problem == "missing_arm":
            result["arms"].pop()
        elif problem == "fake_resume":
            result["arms"][-1]["pending_at_resume"] = 0
        elif problem == "reduced_scope":
            result["full_protocol_exercised"] = False
        else:
            result["class_weight_fingerprint"] = "changed"
        return result
    monkeypatch.setattr(gate, "run_c2_equivalence", fake)
    with pytest.raises((RuntimeError, gate.C2EquivalenceError)):
        gate.run_registered_c2_gate(experiment_id="EXP-C2", parent_experiment_id="EXP-PARENT",
            output_root=repo / "outputs/c2", device="cpu", repository_root=repo)
    record = load_record(repo / "docs/experiments/records/EXP-C2.json")
    assert record["status"] == "failed" and record["validity"]["valid"] is False
    assert "partial.pt" in {row["role"] for row in record["artifacts"]}


@pytest.mark.parametrize("path", (SCRIPT, Path(gate.__file__)))
def test_module_import_as_spawn_main_does_not_import_torch_or_execute(monkeypatch, path):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "numpy", "music_critic")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("__mp_main__", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize("name", gate.THREAD_ENVIRONMENT)
def test_cuda_cli_env_guard_precedes_analysisgnn_package_and_torch_import(monkeypatch, name):
    import builtins
    cli = _load_script()
    for variable in gate.THREAD_ENVIRONMENT:
        monkeypatch.setenv(variable, "1")
    monkeypatch.delenv(name)
    original = builtins.__import__
    def guarded(module, *args, **kwargs):
        assert module.split(".")[0] not in ("torch", "numpy", "music_critic")
        return original(module, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--experiment-id", "X", "--parent-experiment-id", "P", "--output-root", "/unused"])
    assert exc.value.code == 2


def test_cli_has_fixed_full_synthetic_interface(monkeypatch, capsys):
    cli = _load_script()
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    calls = []
    monkeypatch.setattr(gate, "run_registered_c2_gate", lambda **kwargs: calls.append(kwargs) or {"equivalence_passed": True})
    args = ["--experiment-id", "X", "--parent-experiment-id", "P", "--output-root", "/unused", "--device", "cpu"]
    assert cli.main(args) == 0
    assert calls == [{"experiment_id": "X", "parent_experiment_id": "P", "output_root": Path("/unused"), "device": "cpu"}]
    assert json.loads(capsys.readouterr().out)["equivalence_passed"] is True
    for extra in (["--worker-counts", "0"], ["--corpus-root", "/forbidden"], ["--measured-draws", "2"]):
        with pytest.raises(SystemExit):
            cli.main(args + extra)
