"""Isolated, equal-draw throughput replay with a frozen B2 reference arm.

No validation, checkpoint input, new corpus conversion, or model-quality
selection is performed. Production execution requires existing verified TRAIN
cache entries and an idle CUDA device. The CLI registers evidence separately.
"""

from __future__ import annotations

from music_critic.cuda_memory import initialize_cuda_memory_statistics
from music_critic.device import resolve_runtime_device

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.full_orbit_training import build_full_orbit_table
from music_critic.experiments.analysisgnn.orbit_matched_control import (
    OrbitMatchedControlSampler, build_orbit_matched_control_optimizer_scheduler,
)
from music_critic.experiments.analysisgnn.performance_data import (
    FrozenProductionRecordLoader, PerformanceDataError, PreparedGraphCache, SyntheticRecordLoader,
)
from music_critic.experiments.analysisgnn.performance_measurement import (
    EqualDrawBenchmarkPlan, StageTimer, raw_graph_counts, resource_snapshot, summarize_window,
)
from music_critic.experiments.analysisgnn.performance_training import (
    attempt_performance_update, prepare_performance_class_weights,
)


REPLAY_SCHEMA = "AnalysisGNNPerformanceReplay@1.0.0"


def recursive_state_fingerprint(value: object) -> str:
    """Hash typed recursive state, including exact tensor/NumPy bytes."""
    digest = sha256()

    def frame(tag: str, payload: bytes = b"") -> None:
        header = tag.encode("utf-8")
        digest.update(len(header).to_bytes(8, "big") + header)
        digest.update(len(payload).to_bytes(8, "big") + payload)

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            frame("torch", str(tensor.dtype).encode())
            visit(tuple(tensor.shape))
            frame("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            frame("numpy", item.dtype.str.encode())
            visit(item.shape)
            frame("bytes", item.tobytes(order="C"))
        elif isinstance(item, np.generic):
            frame("numpy_scalar", item.dtype.str.encode() + item.tobytes())
        elif isinstance(item, Mapping):
            frame("mapping", str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            frame(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            frame("none")
        elif type(item) is bool:
            frame("bool", str(item).encode())
        elif type(item) is int:
            frame("int", str(item).encode())
        elif type(item) is float:
            frame("float", item.hex().encode())
        elif isinstance(item, str):
            frame("str", item.encode("utf-8"))
        else:
            raise TypeError(f"performance.unsupported_state_type:{type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _assert_idle_gpu() -> list[dict[str, object]]:
    """Allow only the observed desktop daemon; never signal another process."""
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("performance.gpu_idle_state_unverified") from exc
    exempt = []
    for line in process.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(",", 1)
        pid = parts[0].strip()
        if not pid.isdigit():
            raise RuntimeError("performance.gpu_process_listing_invalid")
        if int(pid) == os.getpid():
            continue
        name = parts[1].strip() if len(parts) == 2 else ""
        if name == "/usr/libexec/gnome-remote-desktop-daemon":
            exempt.append({"pid": int(pid), "process_name": name})
            continue
        raise RuntimeError("performance.other_gpu_compute_process_active")
    return exempt


def _scaler() -> tuple[object, str]:
    factory = getattr(torch.amp, "GradScaler", None)
    if factory is not None:
        return factory("cuda", enabled=False), "torch.amp.GradScaler(cuda,enabled=False)"
    return torch.cuda.amp.GradScaler(enabled=False), "torch.cuda.amp.GradScaler(enabled=False)"


def _capture_rng(include_cuda: bool) -> dict[str, object]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if include_cuda else None}


def _restore_rng(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return deepcopy(value)


def _snapshot(model, optimizer, scheduler, scaler) -> dict[str, object]:
    return {
        "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "optimizer": _cpu_copy(optimizer.state_dict()), "scheduler": deepcopy(scheduler.state_dict()),
        "scaler": deepcopy(scaler.state_dict()),
        "rng": deepcopy(_capture_rng(next(model.parameters()).device.type == "cuda")),
        "training": model.training,
    }


def _restore(snapshot, model, optimizer, scheduler, scaler) -> None:
    model.load_state_dict(snapshot["model"], strict=True)
    optimizer.load_state_dict(deepcopy(snapshot["optimizer"]))
    scheduler.load_state_dict(deepcopy(snapshot["scheduler"]))
    scaler.load_state_dict(deepcopy(snapshot["scaler"]))
    model.train(snapshot["training"])
    model.zero_grad(set_to_none=True)
    _restore_rng(deepcopy(snapshot["rng"]))


@contextmanager
def _preserve_runtime(include_cuda: bool):
    state = None
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        state = _capture_rng(include_cuda)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        if state is not None:
            _restore_rng(state)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def _fixture_callback(record_id: str):
    batch, sidecar = reference.build_source_free_fixture()
    sidecar["record_id"] = record_id
    sidecar.pop("fingerprint", None)
    sidecar["fingerprint"] = fingerprint(sidecar)
    return batch, sidecar


def _schedule(mode: str, paths, count: int) -> list[dict[str, object]]:
    if mode == "fixture":
        return [{"record_id": f"synthetic:performance:{index % 2}",
                 "schedule_shift_pc": index % 12, "applied_shift_pc": 0}
                for index in range(count)]
    sampler = OrbitMatchedControlSampler(build_full_orbit_table(
        reference.production_component_records(paths), reference.production_valid_shifts(paths),
    ))
    return [asdict(sampler.peek(index)) for index in range(count)]


def run_replay(
    output_root: Path | str, *, experiment_id: str, mode: str = "fixture", device: str = "cpu",
    measured_draws: int = 800, warmup_draws: int = 160, repeats: int = 3,
    paths: reference.ProductionArtifactPaths | None = None, context: Callable | None = None,
    batch_sizes: tuple[int, ...] = (2, 4, 8), cache_max_bytes: int = 1_073_741_824,
    cache_max_entries: int = 1024, diagnostic: bool = False,
    optimized_data_strategy: str = "prepared_cache",
) -> dict[str, object]:
    """Run fresh, isolated arms; errors leave their output directory intact.

    The caller must allocate a planned experiment before production execution.
    Production supports the B2-only comparison or the original B2/B4/B8 pilot.
    ``optimized_data_strategy="frozen"`` isolates the optimized update while
    retaining reference loading, collation, identity transform and H2D in both
    arms. Prepared-cache experiments allocate a cache only for optimized arms;
    this policy differs from historical replays that retained one in baseline.
    No timing result from fixture mode is hardware or corpus-throughput evidence.
    """
    if mode not in {"fixture", "production"} or device not in {"cpu", "cuda"}:
        raise ValueError("performance.invalid_mode_or_device")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("performance.experiment_id_required")
    if type(diagnostic) is not bool:
        raise ValueError("performance.invalid_diagnostic")
    if optimized_data_strategy not in ("prepared_cache", "frozen"):
        raise ValueError("performance.invalid_optimized_data_strategy")
    if cache_max_bytes <= 0 or cache_max_entries <= 0:
        raise PerformanceDataError("performance_data.cache_limit_invalid", "positive limits required")
    plan = EqualDrawBenchmarkPlan(measured_draws, batch_sizes, fixture_smoke=mode == "fixture" or diagnostic)
    if diagnostic and measured_draws > 32:
        raise ValueError("performance.diagnostic_draw_budget_exceeds_32")
    if 2 not in batch_sizes or (mode == "production" and batch_sizes not in ((2,), (2, 4, 8))):
        raise ValueError("performance.reference_b2_and_production_b2_or_248_required")
    if type(warmup_draws) is not int or warmup_draws < 0 or any(warmup_draws % batch for batch in batch_sizes):
        raise ValueError("performance.invalid_warmup_draws")
    if type(repeats) is not int or repeats < 1 or type(diagnostic) is not bool:
        raise ValueError("performance.invalid_repeat_or_diagnostic")
    if mode == "production" and (device != "cuda" or paths is None):
        raise ValueError("performance.production_requires_cuda_and_explicit_paths")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": REPLAY_SCHEMA, "experiment_id": experiment_id, "mode": mode, "device": device,
        "measured_draws_per_arm": measured_draws, "warmup_draws_per_arm": warmup_draws,
        "repeats": repeats, "updates_by_batch": plan.updates_by_batch,
        "initialization_seed": 17, "training_rng_seed": 17001,
        "diagnostic_pass": diagnostic, "quality_claims": False,
        "validation_accessed": False, "test_accessed": False, "checkpoint_input_used": False,
        "warmup_policy": "discard_updates_restore_common_model_optimizer_scheduler_scaler_rng",
        "reference_data_strategy": "frozen", "optimized_data_strategy": optimized_data_strategy,
        "cache_policy": ("global_frozen_prewarm_baseline_no_cache_optimized_fresh_bounded_LRU"
                         if optimized_data_strategy == "prepared_cache" else
                         "prewarm_frozen_population_all_arms_no_prepared_graph_cache"),
        "cache_cleanup_policy": "drop_previous_cache_and_collect_before_each_arm_and_after_last_arm",
        "first_measured_update": "fresh_Adam_state_allocations_remain_after_snapshot_restore",
        "timing_scope": "train_only_including_metrics_logging_excluding_cache_cleanup_prewarm_and_state_hashes",
        "cuda_event_scope": "diagnostic_pass_only_stream_elapsed_not_kernel_busy_time",
        "cache_max_bytes": cache_max_bytes, "cache_max_entries": cache_max_entries,
        "resume_equivalence_checked": False,
        "equivalence_scope": "B2_uninterrupted_final_full_state_and_per_update_loss_gradient_norm",
    }
    _write(root / "protocol.json", protocol)
    try:
        if device == "cuda":
            protocol["gpu_exempt_processes"] = _assert_idle_gpu()
            if not torch.cuda.is_available():
                raise RuntimeError("performance.cuda_unavailable")
            _write(root / "protocol.json", protocol)
        with _preserve_runtime(device == "cuda"):
            result = _run_arms(root, protocol, plan, paths, context, cache_max_bytes, cache_max_entries)
        _write(root / "summary.json", result)
        return result
    except BaseException as exc:
        _write(root / "failure.json", {
            "schema": REPLAY_SCHEMA, "experiment_id": experiment_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "resource": resource_snapshot(), "completed": False,
        })
        raise


def _run_arms(root, protocol, plan, paths, external_context, max_bytes, max_entries):
    mode, device = protocol["mode"], protocol["device"]
    diagnostic = protocol["diagnostic_pass"]
    current: dict[str, object] = {"timer": StageTimer(), "phase": "prewarm", "update": None}

    @contextmanager
    def stage(name):
        with current["timer"].span(name, phase=current["phase"], update=current["update"]):
            with external_context(name) if external_context is not None else nullcontext():
                yield

    loader = (SyntheticRecordLoader(_fixture_callback, source_seal="synthetic-performance-v1", synthetic=True)
              if mode == "fixture" else FrozenProductionRecordLoader(paths=paths, split="train", context=stage))
    use_prepared_cache = protocol["optimized_data_strategy"] == "prepared_cache"
    cache = None
    schedule = _schedule(mode, paths, plan.measured_draws + protocol["warmup_draws_per_arm"])
    warmup = schedule[:protocol["warmup_draws_per_arm"]]
    measured = schedule[protocol["warmup_draws_per_arm"]:]
    _write(root / "schedule.json", {"warmup": warmup, "measured": measured,
                                   "fingerprint": recursive_state_fingerprint(schedule)})

    def forbidden_conversion(*args, **kwargs):
        raise RuntimeError("performance.cold_canonical_conversion_forbidden")

    def baseline_load(record_id):
        if mode == "fixture":
            return _fixture_callback(record_id)
        loader.check_binding(record_id, split="train")
        # This guard is local to the replay process and leaves source files and
        # the separate canonical process untouched. A disappeared warm cache
        # fails before the reference reader can resolve/convert source data.
        with patch.object(reference, "_resolve_selected_corpus_record", forbidden_conversion):
            return reference.load_production_record(record_id, split="train", paths=paths)

    before_prewarm = time.perf_counter()
    population = tuple(dict.fromkeys(row["record_id"] for row in schedule))
    for record_id in population:
        with stage("prewarm_baseline"):
            baseline_load(record_id)
    prewarm_seconds = time.perf_counter() - before_prewarm
    prewarm_stages = current["timer"].resolve()
    binding_manifest = loader.validate_all_bindings()
    _write(root / "source_bindings.json", binding_manifest)
    _write(root / "prewarm.json", {"wall_seconds": prewarm_seconds, "stages": prewarm_stages,
                                  "cache": None})

    random.seed(17)
    np.random.seed(17)
    torch.random.default_generator.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    model = CorrectedAnalysisGNNModel().to(device).train()
    random.seed(17001)
    np.random.seed(17001)
    torch.random.default_generator.manual_seed(17001)
    if device == "cuda":
        torch.cuda.manual_seed_all(17001)
    optimizer, scheduler = build_orbit_matched_control_optimizer_scheduler(model)
    scaler, scaler_api = _scaler()
    weights = reference.load_frozen_class_weights()
    prepared_weights = prepare_performance_class_weights(weights, device)
    initial = _snapshot(model, optimizer, scheduler, scaler)
    initial_hash = recursive_state_fingerprint(initial)
    arms = [("baseline", 2), *(("optimized", batch) for batch in plan.batch_sizes)]
    results = []

    def update(kind, rows):
        ids = tuple(row["record_id"] for row in rows)
        frozen_data = kind == "baseline" or not use_prepared_cache
        if frozen_data:
            with stage("load"):
                loaded = [baseline_load(record_id) for record_id in ids]
            with stage("collation"):
                raw = reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
            sidecars = tuple(row[1] for row in loaded)
            with stage("transposition"):
                raw = reference.transpose_raw_graph_batch(raw, (0,) * len(rows))
        else:
            with stage("prepare_batch"):
                prepared = cache.prepare_batch(ids, split="train")
            raw, sidecars = prepared.raw_graph_batch, prepared.sidecars
        counts = raw_graph_counts(raw)
        counts["cache_diagnostics"] = None if frozen_data else prepared.cache_diagnostics
        with stage("host_to_device"):
            raw = reference.move_raw_graph_batch(raw, device)
        kwargs = dict(model=model, raw_graph_batch=raw, sidecars=sidecars,
                      shifts=(0,) * len(rows), optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        with stage("full_update"):
            outcome = (reference.attempt_applied_update(**kwargs, class_weights=weights)
                       if kind == "baseline" else
                       attempt_performance_update(**kwargs, class_weights=prepared_weights, instrumentation=stage))
        if not outcome.applied:
            raise RuntimeError("performance.unapplied_update_invalidates_replay")
        return outcome, counts

    for repeat in range(protocol["repeats"]):
        order = arms[repeat % len(arms):] + arms[:repeat % len(arms)]
        for position, (kind, batch_size) in enumerate(order):
            # Release the previous arm's cache, including cycles, before any
            # setup/warmup/timing for this arm. Apply the same collection to
            # every arm; an unused candidate cache must never tax baseline.
            cache = None
            gc.collect()
            if device == "cuda":
                _assert_idle_gpu()
            arm_id = f"r{repeat:02d}-{kind}-b{batch_size}"
            directory = root / arm_id
            directory.mkdir()
            # Both paths warm the same fixed source population. Only the arm
            # that actually consumes prepared graphs gets a fresh bounded LRU.
            if use_prepared_cache and kind == "optimized":
                cache = PreparedGraphCache(loader, max_bytes=max_bytes, max_entries=max_entries,
                                           synthetic=mode == "fixture", context=stage)
            current.update(timer=StageTimer(), phase="prewarm", update=None)
            arm_prewarm_started = time.perf_counter()
            for record_id in population:
                with stage("prewarm_baseline"):
                    baseline_load(record_id)
                if cache is not None:
                    with stage("prewarm_prepared"):
                        cache.prepare_batch((record_id,), split="train")
            loader.validate_all_bindings(expected=binding_manifest)
            arm_prewarm_seconds = time.perf_counter() - arm_prewarm_started
            _write(directory / "prewarm.json", {"wall_seconds": arm_prewarm_seconds,
                "stages": current["timer"].resolve(),
                "cache": None if cache is None else cache.diagnostics(),
                "ordered_population_fingerprint": recursive_state_fingerprint(population)})
            _restore(initial, model, optimizer, scheduler, scaler)
            current.update(timer=StageTimer(), phase="warmup", update=None)
            started_warmup = time.perf_counter()
            for offset in range(0, len(warmup), batch_size):
                current["update"] = offset // batch_size
                update(kind, warmup[offset:offset + batch_size])
            if device == "cuda":
                torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - started_warmup
            _write(directory / "warmup.json", {"wall_seconds": warmup_seconds,
                   "draws": len(warmup), "stages": current["timer"].resolve()})
            _restore(initial, model, optimizer, scheduler, scaler)
            if recursive_state_fingerprint(_snapshot(model, optimizer, scheduler, scaler)) != initial_hash:
                raise RuntimeError("performance.initial_snapshot_restore_mismatch")
            current.update(timer=StageTimer(diagnostic=diagnostic,
                           cuda_events=diagnostic and device == "cuda"), phase="measured", update=None)
            if device == "cuda":
                torch.cuda.synchronize()
                initialize_cuda_memory_statistics(resolve_runtime_device("cuda"))
            resources_before = resource_snapshot()
            cache_before = None if cache is None else cache.diagnostics()
            profiler = None
            if diagnostic:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
                profiler.__enter__()
            wall_started = time.perf_counter()
            durations, counts, trace = [], [], []
            try:
                for offset in range(0, len(measured), batch_size):
                    number = offset // batch_size
                    current["update"] = number
                    started_update = time.perf_counter()
                    rows = measured[offset:offset + batch_size]
                    outcome, graph_counts = update(kind, rows)
                    with stage("metrics_logging"):
                        loss = {name: None if value is None else float(value.detach().cpu())
                                for name, value in (("total", outcome.loss.total),
                                                    ("primary", outcome.loss.primary),
                                                    ("auxiliary", outcome.loss.auxiliary))}
                        trace.append({**loss, "gradient_norm": outcome.gradient_norm})
                        _append(directory / "metrics.jsonl", {
                            "arm": arm_id, "phase": "measured", "update": number + 1,
                            "record_ids": [row["record_id"] for row in rows],
                            "schedule_shifts": [row["schedule_shift_pc"] for row in rows],
                            "applied_shifts": [0] * len(rows), "counts": graph_counts,
                            "loss": loss, "gradient_norm": outcome.gradient_norm,
                            "monotonic_ns": time.perf_counter_ns(),
                        })
                    counts.append(graph_counts)
                    durations.append(time.perf_counter() - started_update)
                    if profiler is not None:
                        profiler.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                wall_seconds = time.perf_counter() - wall_started
                resources_after = resource_snapshot()
            finally:
                if profiler is not None:
                    profiler.__exit__(None, None, None)
                    profiler.export_chrome_trace(str(directory / "profile-trace.json"))
            stages = current["timer"].resolve()
            window_summary = summarize_window(wall_seconds=wall_seconds, counts=counts,
                update_seconds=durations, resource_before=resources_before,
                resource_after=resources_after, cuda_device=device if device == "cuda" else None)
            _write(directory / "stages.json", stages)
            final = _snapshot(model, optimizer, scheduler, scaler)
            summary = {
                "arm": arm_id, "kind": kind, "batch_size": batch_size, "repeat": repeat,
                "data_strategy": "frozen" if kind == "baseline" else protocol["optimized_data_strategy"],
                "update_strategy": "reference" if kind == "baseline" else "optimized",
                "position": position, "initial_state_fingerprint": initial_hash,
                "final_state_fingerprint": recursive_state_fingerprint(final),
                "final_rng_fingerprint": recursive_state_fingerprint(final["rng"]),
                "loss_trace_fingerprint": recursive_state_fingerprint(trace),
                "measured_schedule_fingerprint": recursive_state_fingerprint(measured),
                "warmup_seconds": warmup_seconds,
                "prewarm_seconds": arm_prewarm_seconds,
                "attempted_updates": len(trace), "applied_updates": len(trace),
                "warmup_updates": len(warmup) // batch_size,
                "warmup_draws": len(warmup), "measured_draws": len(measured),
                "cache_before": cache_before,
                "cache_after": None if cache is None else cache.diagnostics(),
                "window": window_summary,
            }
            _write(directory / "summary.json", summary)
            results.append(summary)
    cache = None
    gc.collect()
    loader.validate_all_bindings(expected=binding_manifest)
    equivalences = []
    for repeat in range(protocol["repeats"]):
        pair = {row["kind"]: row for row in results if row["repeat"] == repeat and row["batch_size"] == 2}
        keys = ("final_state_fingerprint", "final_rng_fingerprint", "loss_trace_fingerprint", "measured_schedule_fingerprint")
        equivalences.append({"repeat": repeat, **{
            key: pair["baseline"][key] == pair["optimized"][key] for key in keys}})
    passed = all(all(value for key, value in row.items() if key != "repeat") for row in equivalences)
    return {**protocol, "completed": True, "scaler_api": scaler_api,
            "data_access": {name: False for name in (
                "validation_accessed", "test_inputs_accessed", "test_targets_accessed",
                "test_metrics_computed", "test_used_for_selection")},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
            "actual_attempted_updates": sum(row["attempted_updates"] + row["warmup_updates"] for row in results),
            "actual_applied_updates": sum(row["applied_updates"] + row["warmup_updates"] for row in results),
            "actual_measured_draws": sum(row["measured_draws"] for row in results),
            "actual_warmup_draws": sum(row["warmup_draws"] for row in results),
            "source_binding_fingerprint": recursive_state_fingerprint(binding_manifest),
            "initial_state_fingerprint": initial_hash, "prewarm_seconds": prewarm_seconds,
            "equivalence_passed": passed, "b2_equivalence": equivalences, "arms": results,
            "claim_boundary": ("synthetic_mechanics_only" if mode == "fixture" else
                               "isolated_TRAIN_throughput_only_no_accuracy_or_canonical_run_claim"),
            "different_batch_sizes_are_semantically_equivalent": False}
