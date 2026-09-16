"""Bounded source-free replay/CLI gates; no real corpus or GPU execution."""

from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import weakref

import pytest
import torch

from music_critic.experiments.analysisgnn import performance_prefetch as pool
from music_critic.experiments.analysisgnn import performance_prefetch_replay as replay


def test_descriptor_is_shared_immutable_schedule_and_tampering_fails():
    jobs, descriptor = replay.build_prefetch_schedule(mode="fixture", measured_draws=4, warmup_draws=2, reserved_bytes=2_000_000)
    assert replay.jobs_from_prefetch_descriptor(descriptor) == jobs
    assert [job.draw_ids for job in jobs] == [(0, 1), (2, 3), (4, 5)]
    assert [job.position for job in jobs] == [0, 1, 2]
    assert all(job.applied_shifts == (0, 0) for job in jobs)
    assert jobs[0].record_ids == jobs[1].record_ids
    assert jobs[0].schedule_shifts != jobs[1].schedule_shifts
    changed = json.loads(json.dumps(descriptor))
    changed["jobs"][1]["record_ids"].reverse()
    with pytest.raises(ValueError, match="descriptor_binding_mismatch"):
        replay.jobs_from_prefetch_descriptor(changed)


def test_direct_zero_stream_never_uses_pickle_or_pool_and_checkpoint_is_exact(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("direct worker-zero control invoked serialization or multiprocessing")
    monkeypatch.setattr(replay, "OrderedPrefetch", forbidden)
    monkeypatch.setattr(pool, "_prepare_bytes", forbidden)
    monkeypatch.setattr(pool.pickle, "dump", forbidden)
    monkeypatch.setattr(pool.pickle, "loads", forbidden)
    jobs, _ = replay.build_prefetch_schedule(mode="fixture", measured_draws=4, warmup_draws=0, reserved_bytes=2_000_000)
    config = pool.PrefetchConfig()
    stream = replay.BatchStream(jobs, replay.FixturePreparer(), source_fingerprint="synthetic", config=config, synthetic=True)
    try:
        runtime = stream.worker_runtime_evidence()
        assert runtime["observations"] == [] and runtime["started_workers"] == 0
        assert runtime["native_thread_limits_all_one"] is None
        assert "no_worker_acknowledgments" in runtime["scope"]
        assert stream.diagnostics()["worker_runtime"] == runtime
        first = stream.peek()
        assert first.job == jobs[0]
        assert not stream.commit(0, applied=False)
        assert stream.peek().job == first.job
        stream.commit(0, applied=True)
        state = stream.state_dict()
    finally:
        stream.close()
    resumed = replay.BatchStream(jobs, replay.FixturePreparer(), source_fingerprint="synthetic", config=config,
                                synthetic=True, checkpoint=state)
    assert resumed.peek().job == jobs[1]
    resumed.close()
    with pytest.raises(ValueError, match="direct_resume_binding_mismatch"):
        replay.BatchStream(jobs, replay.FixturePreparer(), source_fingerprint="changed", config=config,
                           synthetic=True, checkpoint=state)


def test_fixture_all_worker_counts_share_full_state_rng_loss_and_original_update_path(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fixture touched production source/cache/reference update or CUDA")
    monkeypatch.setattr(replay, "WarmTrainBatchPreparer", forbidden)
    monkeypatch.setattr(replay.existing.reference, "load_production_record", forbidden)
    monkeypatch.setattr(replay.existing.reference, "attempt_applied_update", forbidden)
    monkeypatch.setattr(replay.existing, "PreparedGraphCache", forbidden)
    monkeypatch.setattr(replay.existing, "_assert_idle_gpu", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    before = replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    result = replay.run_prefetch_replay(tmp_path / "all", experiment_id="SYNTHETIC-PREFETCH",
        measured_draws=4, warmup_draws=2, repeats=1, config=pool.PrefetchConfig(close_timeout_seconds=0.5))
    assert result["completed"] and result["equivalence_passed"]
    assert result["worker_counts"] == [0, 1, 2, 4]
    assert len(result["b2_equivalence"]) == 3
    assert result["actual_measured_draws"] == 16
    assert result["actual_applied_updates"] == 12
    assert before == replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    assert all(value is None for value in result["data_access"].values())
    assert "validation_accessed" not in result and "test_accessed" not in result
    for arm in result["arms"]:
        assert arm["update_strategy"] == "optimized" and arm["data_strategy"] == "frozen"
        assert arm["window"]["records"] == 4 and arm["window"]["updates"] == 2
        assert arm["prefetch_cursor"]["committed_batch_cursor"] == 3
        runtime = json.loads((tmp_path / "all" / arm["arm"] / "worker-runtime.json").read_text())
        assert runtime == arm["worker_runtime"] == arm["queue_final"]["worker_runtime"]
        assert runtime["observations_complete"]
        assert runtime["configured_workers"] == runtime["started_workers"] == arm["worker_count"]
        assert [row["worker_id"] for row in runtime["observations"]] == list(range(arm["worker_count"]))
        for observation in runtime["observations"]:
            assert observation["torch_num_threads"] == observation["torch_num_interop_threads"] == 1
            assert observation["cuda_initialized"] is False
        assert "worker_runtime" not in json.dumps(arm["prefetch_cursor"])
        stages = json.loads((tmp_path / "all" / arm["arm"] / "stages.json").read_text())
        names = {row["stage"] for row in stages}
        assert all(row["phase"] == "measured" for row in stages)
        assert ("direct_prepare" in names) == (arm["worker_count"] == 0)
        assert ("queue_wait_decode_dispatch" in names) == (arm["worker_count"] != 0)
        assert arm["startup_seconds"] >= 0 and arm["cleanup_seconds"] >= 0
        metrics = [json.loads(line) for line in (tmp_path / "all" / arm["arm"] / "metrics.jsonl").read_text().splitlines()]
        assert all((row["counts"]["transport_serialized_payload_bytes"] == 0) == (arm["worker_count"] == 0) for row in metrics)
        assert all("worker_runtime" not in row["queue"] for row in metrics)
    assert result["environment"]["parent_torch_threads"] == torch.get_num_threads()
    assert result["zero_worker_control"] == "direct_original_preparer_without_pickle_or_queue"


def test_full_model_logits_adam_rng_and_serialized_resume_with_pending_are_exact(tmp_path):
    before = replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    result = replay.run_fixture_resume_gate(tmp_path / "checkpoints")
    assert result["passed"] and result["worker_counts"] == [0, 1, 2, 4]
    assert before == replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    for row in result["results"]:
        assert row["resume_exact"]
        if row["worker_count"]:
            assert row["resumed"]["pending_at_resume"] > 0
        assert len(row["uninterrupted"]["worker_runtime_sessions"]) == 1
        assert len(row["resumed"]["worker_runtime_sessions"]) == 2
        for branch in ("uninterrupted", "resumed"):
            for session in row[branch]["worker_runtime_sessions"]:
                assert session["observations_complete"] and session["started_workers"] == row["worker_count"]
        if row["worker_count"]:
            first, second = row["resumed"]["worker_runtime_sessions"]
            assert {obs["pid"] for obs in first["observations"]}.isdisjoint(obs["pid"] for obs in second["observations"])
        assert (tmp_path / "checkpoints" / f"worker-{row['worker_count']}.pt").is_file()


def test_production_missing_inputs_rejected_before_loader_model_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("production missing-gate error reached source/model")
    monkeypatch.setattr(replay, "WarmTrainBatchPreparer", forbidden)
    monkeypatch.setattr(replay, "_components", forbidden)
    with pytest.raises(ValueError, match="explicit_production_inputs_required"):
        replay.run_prefetch_replay(tmp_path / "absent", experiment_id="NO-RUN", mode="production", device="cuda")
    assert not (tmp_path / "absent").exists()


def _sizing_artifacts(tmp_path, *, status="completed", mode="production"):
    from music_critic.experiments.registry import DATA_ACCESS_FIELDS, seal_record, write_record_once
    output = tmp_path / "sizing"
    output.mkdir()
    _, descriptor = replay.build_prefetch_schedule(mode="fixture", measured_draws=4, warmup_draws=0, reserved_bytes=2_000_000)
    values = {"schedule.json": descriptor, "source-bindings.json": {"record_bindings": {}},
              "reservation-manifest.json": {"evidence_id": "EXP-SYNTHETIC-SIZES", "fingerprint": "f" * 64}}
    artifacts = []
    for role, value in values.items():
        payload = (json.dumps(value) + "\n").encode()
        (output / role).write_bytes(payload)
        artifacts.append({"role": role, "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload),
                          "availability": "local_ignored", "locator": f"outputs/synthetic/{role}"})
    record = {"schema_version": "1.0.0", "experiment_id": "EXP-SYNTHETIC-SIZES", "kind": "data_audit",
        "phase": "9E-P1", "title": "Synthetic artifact authentication fixture", "status": status,
        "question": "Unit test?", "hypothesis": "Unit test", "decision_use": "Unit test only",
        "repository": {}, "seeds": [], "budget": {}, "data_access": {
            **{name: None for name in DATA_ACCESS_FIELDS}, "access_evidence_status": "pending_primary_evidence",
            "declared_policy": {"source": "synthetic test", "validation_allowed": False,
                "test_inputs_allowed": False, "test_targets_allowed": False, "test_metrics_allowed": False, "test_selection_allowed": False}},
        "protocol": {"schema_version": "AnalysisGNNPrefetchReservationGate@1.0.0", "mode": mode,
                     "split": "train", "buffer_byte_cap": 64 * 1024**2, "max_pending_batches": 4},
        "results": {"synthetic": True}, "validity": {"valid": status == "completed", "reason": "synthetic fixture"},
        "claims": {"supported": [], "not_supported": ["real data"], "caveats": []},
        "artifacts": artifacts, "import_provenance": {"mode": "synthetic_test"}}
    records = tmp_path / "records"
    write_record_once(records / "EXP-SYNTHETIC-SIZES.json", seal_record(record))
    return {"schedule_path": output / "schedule.json", "binding_path": output / "source-bindings.json",
            "reservation_path": output / "reservation-manifest.json", "records_root": records}


@pytest.mark.parametrize("problem", ("none", "failed", "synthetic", "bytes", "symlink"))
def test_sizing_artifact_authentication_requires_completed_production_record_and_exact_bytes(tmp_path, problem):
    paths = _sizing_artifacts(tmp_path, status="failed" if problem == "failed" else "completed",
                              mode="synthetic" if problem == "synthetic" else "production")
    if problem == "bytes":
        with paths["binding_path"].open("ab") as stream:
            stream.write(b" ")
    if problem == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(paths["schedule_path"])
        paths["schedule_path"] = link
    if problem == "none":
        got = replay.load_production_prefetch_inputs(**paths)
        assert got["reservation_evidence_id"] == "EXP-SYNTHETIC-SIZES"
        assert set(got["input_artifact_sha256"]) == {"schedule.json", "source-bindings.json", "reservation-manifest.json"}
    else:
        with pytest.raises(ValueError, match={"failed": "completed_reservation", "synthetic": "production_train_reservation",
                                            "bytes": "artifact_mismatch", "symlink": "regular_input_artifact"}[problem]):
            replay.load_production_prefetch_inputs(**paths)


def test_rotated_order_releases_previous_full_snapshot_and_registered_budget_is_correct(tmp_path, monkeypatch):
    from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
    from music_critic.experiments.registry import load_record
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"), ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# synthetic\n")
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    snapshots, original = [], replay.existing._snapshot
    def observed(*args):
        value = original(*args)
        # First snapshot is the common initial state. Later snapshot tensors
        # must have died before a new arm's initial-state equality check.
        snapshots.append(weakref.ref(next(iter(value["model"].values()))))
        return value
    monkeypatch.setattr(replay.existing, "_snapshot", observed)
    checks, prewarm = [], replay._prewarm
    def checked(*args):
        checks.append(sum(reference() is not None for reference in snapshots))
        assert sum(reference() is not None for reference in snapshots) <= 1
        return prewarm(*args)
    monkeypatch.setattr(replay, "_prewarm", checked)
    config = pool.PrefetchConfig(close_timeout_seconds=0.5)
    _, descriptor = replay.build_prefetch_schedule(mode="fixture", measured_draws=2, warmup_draws=0, reserved_bytes=2_000_000)
    protocol = {**replay.protocol_bindings(mode="fixture", descriptor=descriptor, reservation_manifest=None,
        config=config, worker_counts=(0, 2)), "mode": "fixture", "device": "cpu", "repeats": 2,
        "measured_draws": 2, "warmup_draws": 0, "batch_sizes": [2], "update_strategy": "optimized"}
    output = root / "outputs/EXP-FIXTURE-PREFETCH"
    result = run_registered_replay(experiment_id="EXP-FIXTURE-PREFETCH", output_root=output,
        repository_root=root, protocol=protocol, execute=lambda: replay.run_prefetch_replay(output,
            experiment_id="EXP-FIXTURE-PREFETCH", measured_draws=2, warmup_draws=0, repeats=2,
            worker_counts=(0, 2), config=config, descriptor=descriptor))
    assert [(row["repeat"], row["worker_count"]) for row in result["arms"]] == [(0, 0), (0, 2), (1, 2), (1, 0)]
    record = load_record(root / "docs/experiments/records/EXP-FIXTURE-PREFETCH.json")
    assert record["status"] == "completed"
    assert record["budget"]["planned"] == record["budget"]["actual"] == 8
    assert all(record["data_access"][name] is None for name in ("test_inputs_accessed", "test_targets_accessed", "validation_accessed"))


def _load_cli():
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_prefetch_replay.py"
    spec = importlib.util.spec_from_file_location("synthetic_prefetch_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_guard_and_declared_evidence_bindings_without_actual_registration(tmp_path, monkeypatch):
    from music_critic.experiments.analysisgnn import performance_evidence as evidence
    captured = {}
    def register(**kwargs):
        captured.update(kwargs)
        return {"synthetic": True}
    monkeypatch.setattr(evidence, "run_registered_replay", register)
    monkeypatch.setattr(replay, "_components", lambda *_: pytest.fail("CLI import initialized model"))
    module = _load_cli()
    assert not captured
    assert module.main(["--experiment-id", "SYNTHETIC", "--parent-experiment-id", "SYNTHETIC-PARENT",
        "--output-root", str(tmp_path / "not-created"), "--mode", "fixture", "--measured-draws", "4",
        "--warmup-draws", "2", "--repeats", "1"]) == 0
    protocol = captured["protocol"]
    assert protocol["comparison_kind"] == "ordered_prefetch"
    assert protocol["planned_arm_count"] == 4 and protocol["worker_counts"] == [0, 1, 2, 4]
    assert protocol["update_strategy"] == "optimized" and protocol["batch_sizes"] == [2]
    assert all(len(protocol[key]) == 64 for key in ("worker_configs_fingerprint", "schedule_descriptor_fingerprint", "reservation_manifest_fingerprint"))
    assert captured["parent_experiment_id"] == "SYNTHETIC-PARENT"
    assert not (tmp_path / "not-created").exists()
    with pytest.raises(SystemExit):
        module.main(["--experiment-id", "INVALID", "--parent-experiment-id", "P", "--output-root", str(tmp_path / "no"), "--mode", "production", "--device", "cuda"])
