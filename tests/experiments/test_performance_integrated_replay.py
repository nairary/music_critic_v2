"""Cold process-local cache integration; all inputs are synthetic."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pickle

import pytest
import torch

from music_critic.experiments.analysisgnn import performance_prefetch as pool
from music_critic.experiments.analysisgnn import performance_prefetch_replay as replay
from music_critic.experiments.analysisgnn import performance_evidence as evidence


def _bindings():
    _, descriptor = replay.build_prefetch_schedule(mode="fixture", measured_draws=8,
        warmup_draws=4, reserved_bytes=2_000_000)
    return {**replay.protocol_bindings(mode="fixture", descriptor=descriptor,
        reservation_manifest=None, config=pool.PrefetchConfig(), worker_counts=(4, 0),
        preparation_strategy="compact_cache", compact_cache_global_max_bytes=64 * 1024**2),
        "repeats": 1, "measured_draws": 8, "warmup_draws": 4, "batch_size": 2}


@pytest.mark.parametrize("strategy,budget", [("bad", None), ("frozen", 1024),
    ("compact_cache", None), ("compact_cache", True), ("compact_cache", 3)])
def test_invalid_cache_configuration_fails_before_model_or_output(tmp_path, monkeypatch, strategy, budget):
    monkeypatch.setattr(replay, "_components", lambda *_: pytest.fail("model before cache validation"))
    with pytest.raises(ValueError):
        replay.run_prefetch_replay(tmp_path / "absent", experiment_id="SYNTHETIC-INVALID",
            worker_counts=(4, 0), preparation_strategy=strategy, compact_cache_global_max_bytes=budget)
    assert not (tmp_path / "absent").exists()


def test_one_budget_is_partitioned_and_exact_policy_is_preregistered():
    protocol = _bindings()
    evidence._validate_protocol(protocol)
    assert protocol["worker_counts"] == [4, 0]
    for workers in (4, 0):
        policy = protocol["cache_policies"][str(workers)]
        assert policy["preparing_processes"] == max(workers, 1)
        assert policy["global_max_bytes"] == 64 * 1024**2
    for field in ("cache_policies_fingerprint", "cache_start_policy", "cache_budget_scope"):
        changed = deepcopy(protocol)
        changed[field] = "wrong"
        with pytest.raises(ValueError, match="compact_cache_binding_mismatch"):
            evidence._validate_protocol(changed)
    changed = deepcopy(protocol)
    del changed["compact_cache_global_max_bytes"]
    with pytest.raises(ValueError, match="compact_cache_binding_incomplete"):
        evidence._validate_protocol(changed)


def test_fixture_pickle_discards_populated_cache_and_direct_stream_has_no_ipc(monkeypatch):
    jobs, _ = replay.build_prefetch_schedule(mode="fixture", measured_draws=4,
        warmup_draws=0, reserved_bytes=2_000_000)
    policy = replay.cache_policies("compact_cache", 64 * 1024**2, (0,))["0"]
    preparer = replay.FixturePreparer(policy)
    preparer(jobs[0])
    assert preparer.cache_diagnostics()["initialized"]
    restored = pickle.loads(pickle.dumps(preparer))
    assert not restored.cache_diagnostics()["initialized"]
    assert restored.cache_policy == policy
    # A disabled/default fixture must also initialize correctly after spawn pickle.
    assert pickle.loads(pickle.dumps(replay.FixturePreparer())).cache_policy is None
    monkeypatch.setattr(replay, "OrderedPrefetch", lambda *a, **kw: pytest.fail("direct invoked pool"))
    stream = replay.BatchStream(jobs, restored, source_fingerprint="synthetic",
        config=pool.PrefetchConfig(), synthetic=True)
    try:
        batch = stream.peek()
        assert batch.job == jobs[0]
        assert stream.diagnostics()["prepared_cache"]["misses"] == 2
        stream.commit(0, applied=True)
    finally:
        stream.close()


def test_actual_three_arms_preserve_full_training_and_reference_baseline(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("integrated fixture accessed corpus/CUDA or population prewarm")
    monkeypatch.setattr(replay.existing.reference, "load_production_record", forbidden)
    monkeypatch.setattr(replay, "_prewarm", forbidden)
    monkeypatch.setattr(replay.existing, "_assert_idle_gpu", forbidden)
    original_update = replay.existing.reference.attempt_applied_update
    reference_calls = []
    def reference_update(**kwargs):
        assert "instrumentation" not in kwargs
        reference_calls.append(kwargs["class_weights"])
        return original_update(**kwargs)
    monkeypatch.setattr(replay.existing.reference, "attempt_applied_update", reference_update)
    before = replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    result = replay.run_prefetch_replay(tmp_path / "integrated", experiment_id="SYNTHETIC-INTEGRATED",
        measured_draws=8, warmup_draws=4, repeats=1, worker_counts=(4, 0),
        preparation_strategy="compact_cache", compact_cache_global_max_bytes=64 * 1024**2,
        include_original_baseline=True,
        config=pool.PrefetchConfig(close_timeout_seconds=1.0))
    assert result["equivalence_passed"]
    assert result["reference_data_strategy"] == "frozen"
    assert "selected_update" in result["timing_scope"]
    assert result["update_strategy"] == "mixed_reference_and_optimized"
    assert result["actual_applied_updates"] == 18
    assert len(reference_calls) == 6 and all(row is reference_calls[0] for row in reference_calls)
    assert result["worker_counts"] == [4, 0]
    assert before == replay.existing.recursive_state_fingerprint(replay.existing._capture_rng(False))
    assert all(value is None for value in result["data_access"].values())
    arms = result["arms"]
    assert [row["arm_key"] for row in arms] == ["integrated-workers4", "integrated-workers0", "original-workers0"]
    assert result["replay_wall_seconds"] >= sum(row["whole_arm_wall_seconds"] for row in arms)
    for arm in arms:
        directory = tmp_path / "integrated" / arm["arm"]
        start = json.loads((directory / "cache-start.json").read_text())
        assert not start["initialized"] and start["accounted_bytes"] == 0
        assert arm["whole_arm_wall_seconds"] > arm["window"]["wall_seconds"]
        assert not arm["cache_after_cleanup"]["initialized"]
        assert arm["cache_after_cleanup"]["accounted_bytes"] == 0
        original = arm["arm_key"] == "original-workers0"
        assert arm["data_strategy"] == ("frozen" if original else "compact_cache")
        assert arm["update_strategy"] == ("reference" if original else "optimized")
        assert arm["prefetch_cursor"]["committed_batch_cursor"] == 6
        assert (directory / "cache-after-warmup.json").is_file()
    stats = arms[0]["cache_before_close"]
    assert len(stats["observations"]) == 4
    assert sum(row["observation"]["accounted_bytes"] for row in stats["observations"]) <= 64 * 1024**2
    assert arms[0]["parent_cache_before_close"]["accounted_bytes"] == 0
    direct = arms[1]["cache_before_close"]
    assert direct["misses"] == 2 and direct["hits"] == 10
    assert arms[2]["cache_before_close"] is None
    assert arms[2]["parent_cache_before_close"]["policy"] is None
    assert all(row["control_arm_key"] == "original-workers0" for row in result["b2_equivalence"])


def test_third_arm_has_distinct_identity_and_cannot_be_another_optimized_zero():
    _, descriptor = replay.build_prefetch_schedule(mode="fixture", measured_draws=8,
        warmup_draws=4, reserved_bytes=2_000_000)
    protocol = {**replay.protocol_bindings(mode="fixture", descriptor=descriptor,
        reservation_manifest=None, config=pool.PrefetchConfig(), worker_counts=(4, 0),
        preparation_strategy="compact_cache", compact_cache_global_max_bytes=64 * 1024**2,
        include_original_baseline=True), "repeats": 1, "measured_draws": 8, "warmup_draws": 4,
        "update_strategy": "mixed_reference_and_optimized"}
    evidence._validate_protocol(protocol)
    result = {"arms": [{"repeat": 0, "arm_key": row["arm_key"], "worker_count": row["worker_count"],
        "batch_size": 2, "kind": ("original_baseline" if row["update_strategy"] == "reference" else
        "baseline" if row["worker_count"] == 0 else "optimized"), "data_strategy": row["preparation_strategy"],
        "update_strategy": row["update_strategy"], "window": {"records": 8}} for row in protocol["arm_strategies"]]}
    evidence._validate_declared_arm_count(protocol, result)
    changed = deepcopy(result)
    changed["arms"][2]["update_strategy"] = "optimized"
    with pytest.raises(ValueError, match="original_arm_strategy_mismatch"):
        evidence._validate_declared_arm_count(protocol, changed)
    changed = deepcopy(result)
    changed["arms"][2]["arm_key"] = "integrated-workers0"
    with pytest.raises(ValueError, match="arm_coverage_mismatch"):
        evidence._validate_declared_arm_count(protocol, changed)


def test_cli_carries_cache_policy_into_declared_protocol_and_execution(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_prefetch_replay.py"
    spec = importlib.util.spec_from_file_location("integrated_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}
    def register(**kwargs):
        captured.update(kwargs)
        return kwargs["execute"]()
    monkeypatch.setattr(evidence, "run_registered_replay", register)
    monkeypatch.setattr(replay, "run_prefetch_replay", lambda *a, **kw: captured.update(executed=kw) or {})
    assert module.main(["--experiment-id", "SYNTHETIC", "--parent-experiment-id", "P",
        "--output-root", str(tmp_path / "none"), "--mode", "fixture", "--measured-draws", "8",
        "--warmup-draws", "4", "--repeats", "1", "--worker-counts", "4", "0",
        "--preparation-strategy", "compact_cache", "--compact-cache-global-max-bytes", "67108864"]) == 0
    assert captured["protocol"]["cache_policies"] == _bindings()["cache_policies"]
    assert captured["executed"]["preparation_strategy"] == "compact_cache"
    assert captured["executed"]["compact_cache_global_max_bytes"] == 67108864
    assert captured["executed"]["worker_counts"] == (4, 0)
    assert not (tmp_path / "none").exists()
