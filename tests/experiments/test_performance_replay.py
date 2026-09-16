"""Tiny source-free replay mechanics and arm-routing tests."""

from collections import Counter
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import performance_replay as replay


def test_recursive_fingerprint_binds_tensor_numpy_and_typed_recursive_state():
    value = {"tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
             "numpy": np.array([1, 2], dtype=np.uint32), "state": (1, True, None, 0.5)}
    reordered = {"state": value["state"], "numpy": value["numpy"].copy(),
                 "tensor": value["tensor"].clone()}
    assert replay.recursive_state_fingerprint(value) == replay.recursive_state_fingerprint(reordered)
    reordered["tensor"][0, 0] += 1
    assert replay.recursive_state_fingerprint(value) != replay.recursive_state_fingerprint(reordered)
    assert replay.recursive_state_fingerprint(1) != replay.recursive_state_fingerprint(True)
    assert replay.recursive_state_fingerprint([1]) != replay.recursive_state_fingerprint((1,))
    assert replay.recursive_state_fingerprint(torch.tensor(1.0)) != replay.recursive_state_fingerprint(torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(TypeError, match="unsupported_state_type"):
        replay.recursive_state_fingerprint(object())


def test_two_update_fixture_runs_both_b2_paths_and_preserves_caller_rng(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fixture reached real-data or CUDA reader")

    monkeypatch.setattr(replay.reference, "load_production_record", forbidden)
    monkeypatch.setattr(replay, "FrozenProductionRecordLoader", forbidden)
    monkeypatch.setattr(replay, "_assert_idle_gpu", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    before = replay.recursive_state_fingerprint(replay._capture_rng(False))
    result = replay.run_replay(
        tmp_path / "tiny", experiment_id="FIXTURE-PERFORMANCE", mode="fixture", device="cpu",
        measured_draws=2, warmup_draws=0, repeats=1, batch_sizes=(2,),
    )
    assert replay.recursive_state_fingerprint(replay._capture_rng(False)) == before
    assert result["completed"] and result["equivalence_passed"]
    assert result["claim_boundary"] == "synthetic_mechanics_only"
    assert not result["test_accessed"] and not result["validation_accessed"]
    assert not result["checkpoint_input_used"] and not result["quality_claims"]
    assert result["reference_data_strategy"] == "frozen"
    assert result["optimized_data_strategy"] == "prepared_cache"
    assert result["cache_policy"] == "global_frozen_prewarm_baseline_no_cache_optimized_fresh_bounded_LRU"
    assert len(result["arms"]) == 2
    for arm in result["arms"]:
        assert arm["window"]["updates"] == 1 and arm["window"]["records"] == 2
        directory = tmp_path / "tiny" / arm["arm"]
        assert len((directory / "metrics.jsonl").read_text().splitlines()) == 1
        stages = json.loads((directory / "stages.json").read_text())
        assert {row["stage"] for row in stages} >= {"host_to_device", "full_update", "metrics_logging"}
        assert all(row["phase"] == "measured" for row in stages)
        expected_data = "frozen" if arm["kind"] == "baseline" else "prepared_cache"
        assert arm["data_strategy"] == expected_data
        assert arm["update_strategy"] == ("reference" if arm["kind"] == "baseline" else "optimized")
        assert ("prepare_batch" in {row["stage"] for row in stages}) == (arm["kind"] == "optimized")
        prewarm = json.loads((directory / "prewarm.json").read_text())
        if arm["kind"] == "optimized":
            assert "prewarm_prepared" in {row["stage"] for row in prewarm["stages"]}
            assert prewarm["cache"]["entries"] > 0
        else:
            assert "prewarm_prepared" not in {row["stage"] for row in prewarm["stages"]}
            assert prewarm["cache"] is arm["cache_before"] is arm["cache_after"] is None
    root_prewarm = json.loads((tmp_path / "tiny/prewarm.json").read_text())
    assert root_prewarm["cache"] is None
    assert "prewarm_prepared" not in {row["stage"] for row in root_prewarm["stages"]}
    assert json.loads((tmp_path / "tiny" / "summary.json").read_text())["equivalence_passed"]
    with pytest.raises(FileExistsError):
        replay.run_replay(tmp_path / "tiny", experiment_id="FIXTURE-REUSE", measured_draws=2,
                          warmup_draws=0, repeats=1, batch_sizes=(2,))


def test_prepared_cache_only_lives_in_optimized_arms_including_rotated_order(tmp_path, monkeypatch):
    references = []
    update_calls = Counter()
    measured = False
    collections = []
    original_cache = replay.PreparedGraphCache
    original_collect = replay.gc.collect
    original_snapshot = replay.resource_snapshot
    original_write = replay._write

    def live_cache_count():
        return sum(reference() is not None for reference in references)

    class ObservedCache(original_cache):
        def __init__(self, *args, **kwargs):
            assert live_cache_count() == 0, "previous arm still owns a cache"
            assert not measured
            super().__init__(*args, **kwargs)
            # Exercise release of cyclic references, not just reference counts.
            self._test_cycle = self
            references.append(weakref.ref(self))

    def collect(*args, **kwargs):
        assert not measured, "explicit collection occurred inside measured window"
        collections.append(True)
        return original_collect(*args, **kwargs)

    def snapshot():
        nonlocal measured
        measured = not measured
        return original_snapshot()

    def write(path, value):
        if path.name == "prewarm.json":
            if path.parent.name == "rotated" or "baseline" in path.parent.name:
                assert live_cache_count() == 0, "baseline/global prewarm owns unused cache"
                assert value["cache"] is None
            else:
                assert live_cache_count() == 1
        return original_write(path, value)

    def observe_update(owner, name, expected_caches):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            update_calls[name] += 1
            assert live_cache_count() == expected_caches
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    monkeypatch.setattr(replay, "PreparedGraphCache", ObservedCache)
    monkeypatch.setattr(replay.gc, "collect", collect)
    monkeypatch.setattr(replay, "resource_snapshot", snapshot)
    monkeypatch.setattr(replay, "_write", write)
    observe_update(replay.reference, "attempt_applied_update", 0)
    observe_update(replay, "attempt_performance_update", 1)
    before = replay.recursive_state_fingerprint(replay._capture_rng(False))
    result = replay.run_replay(
        tmp_path / "rotated", experiment_id="FIXTURE-CACHE-ARM-OWNERSHIP",
        measured_draws=2, warmup_draws=2, repeats=3, batch_sizes=(2,),
    )
    assert replay.recursive_state_fingerprint(replay._capture_rng(False)) == before
    assert result["equivalence_passed"]
    assert [(arm["repeat"], arm["kind"]) for arm in result["arms"]] == [
        (0, "baseline"), (0, "optimized"), (1, "optimized"),
        (1, "baseline"), (2, "baseline"), (2, "optimized"),
    ]
    assert len(references) == 3 and live_cache_count() == 0
    assert len(collections) == 7 and measured is False
    assert update_calls == {"attempt_applied_update": 6, "attempt_performance_update": 6}
    assert len(result["b2_equivalence"]) == 3
    assert all(all(value for key, value in pair.items() if key != "repeat")
               for pair in result["b2_equivalence"])
    population_fingerprints = set()
    for arm in result["arms"]:
        prewarm = json.loads((tmp_path / "rotated" / arm["arm"] / "prewarm.json").read_text())
        population_fingerprints.add(prewarm["ordered_population_fingerprint"])
        if arm["kind"] == "baseline":
            assert arm["cache_before"] is arm["cache_after"] is prewarm["cache"] is None
        else:
            assert prewarm["cache"]["misses"] == prewarm["cache"]["entries"] == 2
            assert prewarm["cache"]["hits"] == prewarm["cache"]["evictions"] == 0
    assert len(population_fingerprints) == 1
    assert result["cache_cleanup_policy"] == "drop_previous_cache_and_collect_before_each_arm_and_after_last_arm"


def test_frozen_data_isolates_update_and_skips_prepared_cache_with_exact_state(tmp_path, monkeypatch):
    calls = Counter()

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen source-free comparison reached prepared cache, corpus or CUDA")

    monkeypatch.setattr(replay, "PreparedGraphCache", forbidden)
    monkeypatch.setattr(replay, "FrozenProductionRecordLoader", forbidden)
    monkeypatch.setattr(replay.reference, "load_production_record", forbidden)
    monkeypatch.setattr(replay, "_assert_idle_gpu", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)

    def observe(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            calls[name] += 1
            if name == "transpose_raw_graph_batch":
                assert args[1] == (0, 0)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    for name in ("combine_single_record_raw_batches", "transpose_raw_graph_batch",
                 "move_raw_graph_batch", "attempt_applied_update"):
        observe(replay.reference, name)
    observe(replay, "attempt_performance_update")
    before = replay.recursive_state_fingerprint(replay._capture_rng(False))
    root = tmp_path / "frozen"
    result = replay.run_replay(
        root, experiment_id="FIXTURE-FROZEN-DATA", mode="fixture", device="cpu",
        measured_draws=4, warmup_draws=2, repeats=1, batch_sizes=(2,),
        optimized_data_strategy="frozen",
    )
    assert replay.recursive_state_fingerprint(replay._capture_rng(False)) == before
    assert calls == {
        "combine_single_record_raw_batches": 6, "transpose_raw_graph_batch": 6,
        "move_raw_graph_batch": 6, "attempt_applied_update": 3, "attempt_performance_update": 3,
    }
    assert result["equivalence_passed"]
    assert result["reference_data_strategy"] == result["optimized_data_strategy"] == "frozen"
    assert result["cache_policy"] == "prewarm_frozen_population_all_arms_no_prepared_graph_cache"
    assert result["actual_attempted_updates"] == result["actual_applied_updates"] == 6
    assert result["actual_measured_draws"] == 8 and result["actual_warmup_draws"] == 4
    assert result["resume_equivalence_checked"] is False
    assert all(value for key, value in result["b2_equivalence"][0].items() if key != "repeat")
    assert all(value is False for value in result["data_access"].values())
    for arm in result["arms"]:
        assert arm["data_strategy"] == "frozen"
        assert arm["cache_before"] is arm["cache_after"] is None
        directory = root / arm["arm"]
        stages = {row["stage"] for row in json.loads((directory / "stages.json").read_text())}
        assert stages >= {"load", "collation", "transposition", "host_to_device", "full_update"}
        assert "prepare_batch" not in stages
        prewarm = json.loads((directory / "prewarm.json").read_text())
        assert prewarm["cache"] is None
        assert "prewarm_prepared" not in {row["stage"] for row in prewarm["stages"]}
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert len(metrics) == 2
        assert all(row["counts"]["cache_diagnostics"] is None for row in metrics)
    assert json.loads((root / "protocol.json").read_text())["optimized_data_strategy"] == "frozen"


@pytest.mark.parametrize("strategy", ["prepared_cache", "frozen"])
def test_production_b2_only_configuration_reaches_execution_with_three_repeats(tmp_path, monkeypatch, strategy):
    observed = []
    monkeypatch.setattr(replay, "_assert_idle_gpu", lambda: [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(replay, "_preserve_runtime", lambda _: nullcontext())

    def execute(root, protocol, plan, paths, context, max_bytes, max_entries):
        observed.append(protocol)
        assert plan.updates_by_batch == {2: 400}
        assert protocol["repeats"] == 3 and protocol["warmup_draws_per_arm"] == 160
        assert protocol["optimized_data_strategy"] == strategy
        assert isinstance(paths, replay.reference.ProductionArtifactPaths)
        return {"synthetic_configuration_check": True}

    monkeypatch.setattr(replay, "_run_arms", execute)
    replay.run_replay(
        tmp_path / "configuration", experiment_id="FIXTURE-PRODUCTION-CONFIG", mode="production",
        device="cuda", paths=replay.reference.ProductionArtifactPaths(), batch_sizes=(2,),
        optimized_data_strategy=strategy,
    )
    assert len(observed) == 1


@pytest.mark.parametrize("kwargs", [
    {"mode": "invalid"}, {"device": "invalid"}, {"experiment_id": ""},
    {"warmup_draws": -1}, {"warmup_draws": 3}, {"repeats": 0}, {"repeats": True},
    {"batch_sizes": (4, 8)}, {"mode": "production", "device": "cpu"},
    {"mode": "production", "device": "cuda"},
    {"optimized_data_strategy": "unknown"}, {"optimized_data_strategy": None},
    {"optimized_data_strategy": "frozen", "cache_max_bytes": 0},
    {"optimized_data_strategy": "frozen", "cache_max_entries": 0},
    {"cache_max_bytes": 0}, {"cache_max_entries": 0},
    {"mode": "production", "device": "cuda", "batch_sizes": (2, 4)},
])
def test_invalid_configuration_rejected_before_output(tmp_path, kwargs):
    args = {"experiment_id": "FIXTURE-INVALID", **kwargs}
    with pytest.raises(ValueError):
        replay.run_replay(tmp_path / "absent", **args)
    assert not (tmp_path / "absent").exists()


def test_existing_gpu_process_prevents_replay_and_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="999999, canonical-training-python\n"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU touched after guard"))
    with pytest.raises(RuntimeError, match="other_gpu_compute_process_active"):
        replay.run_replay(tmp_path / "blocked", experiment_id="FIXTURE-BLOCKED", device="cuda",
                          measured_draws=2, warmup_draws=0, repeats=1, batch_sizes=(2,))
    failure = json.loads((tmp_path / "blocked" / "failure.json").read_text())
    assert failure["completed"] is False
    assert failure["error"] == "performance.other_gpu_compute_process_active"
    assert not (tmp_path / "blocked" / "summary.json").exists()


def test_gpu_guard_fails_closed_if_monitor_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("synthetic")
    monkeypatch.setattr(replay.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="gpu_idle_state_unverified"):
        replay._assert_idle_gpu()


def test_scaler_adapter_uses_legacy_torch_api_when_new_api_absent(monkeypatch):
    monkeypatch.delattr(torch.amp, "GradScaler", raising=False)
    calls = []
    sentinel = object()
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", lambda **kwargs: calls.append(kwargs) or sentinel)
    scaler, api = replay._scaler()
    assert scaler is sentinel and calls == [{"enabled": False}]
    assert api == "torch.cuda.amp.GradScaler(enabled=False)"


@pytest.mark.parametrize("explicit_strategy", [None, "prepared_cache", "frozen"])
def test_cli_registers_before_execute_without_running_training(tmp_path, monkeypatch, explicit_strategy):
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_performance_replay.py"
    spec = importlib.util.spec_from_file_location("performance_replay_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = []
    expected_strategy = explicit_strategy or "prepared_cache"

    def register(**kwargs):
        order.append("planned")
        assert kwargs["experiment_id"] == "FIXTURE-CLI"
        assert all("path" not in key and "root" not in key for key in kwargs["protocol"])
        assert kwargs["protocol"]["reference_data_strategy"] == "frozen"
        assert kwargs["protocol"]["optimized_data_strategy"] == expected_strategy
        return kwargs["execute"]()

    def execute(*args, **kwargs):
        order.append("execute")
        assert kwargs["batch_sizes"] == (2,)
        assert kwargs["optimized_data_strategy"] == expected_strategy
        return {"fixture": True}

    monkeypatch.setattr(module, "run_registered_replay", register)
    monkeypatch.setattr(module, "run_replay", execute)
    argv = ["--experiment-id", "FIXTURE-CLI", "--output-root", str(tmp_path / "cli"),
            "--mode", "fixture", "--measured-draws", "2", "--warmup-draws", "0",
            "--repeats", "1", "--batch-sizes", "2"]
    if explicit_strategy is not None:
        argv.extend(("--optimized-data-strategy", explicit_strategy))
    assert module.main(argv) == 0
    assert order == ["planned", "execute"]
