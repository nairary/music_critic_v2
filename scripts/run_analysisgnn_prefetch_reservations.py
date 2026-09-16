#!/usr/bin/env python3
"""Registered sequential TRAIN job serialization gate for a future prefetch replay.

No pool, model, optimizer, CUDA call or real-memory capacity check is performed.
One explicit provisional reservation is retained unchanged in every final job;
the reservation integer itself is part of each serialized PreparedCPUJob.
The opt-in compact cache sizes the current integrated preparation path once;
its declared process count divides a cache budget, without spawning workers.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import gc
from hashlib import sha256
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Callable

import torch
import torch_geometric

from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.performance_data import FrozenProductionRecordLoader
from music_critic.experiments.analysisgnn import performance_prefetch as pp
from music_critic.experiments.analysisgnn.performance_evidence import (
    _artifact_rows, _git, _now, _transition, _validated_parent_ids, source_manifest,
)
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, file_sha256, render_registry,
)


GATE_SCHEMA = "AnalysisGNNPrefetchReservationGate@1.0.0"
LIMITS = (
    "Measured bytes are exact serialized CPU jobs for this schedule/source/runtime and retained reservation policy.",
    "The global buffer cap counts one serialized payload reservation per uncommitted job; it is not an RSS bound.",
    "Worker preparation, decoded batches, IPC copies and model/GPU memory require separate measured gates.",
    "No speed, universal OOM safety, prefetch RAM readiness, C2 equivalence or musical quality is established.",
    "Changing job reservations, source, implementation or runtime requires remeasurement or an independently verified gate.",
)
COMPACT_SIZE_LIMIT = ("Compact cache sizing reports the declared policy/history, not an upper bound for other "
    "capacities or histories. Equal payload values can have different pickle lengths; every future "
    "serialization enforces actual bytes <= the declared job reservation.")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("prefetch_reservations." + message)


def _cache_policy(strategy: str, global_max_bytes: int | None, preparing_processes: int | None):
    """Validate CLI policy before source access or experiment registration."""
    _require(strategy in ("frozen", "compact_cache"), "invalid_preparation_strategy")
    if strategy == "frozen":
        _require(global_max_bytes is None and preparing_processes is None,
                 "cache_options_require_compact_cache")
        return None
    _require(type(global_max_bytes) is int and global_max_bytes > 0,
             "explicit_positive_compact_cache_budget_required")
    _require(type(preparing_processes) is int and preparing_processes in (1, 2, 4),
             "explicit_compact_cache_process_divisor_required")
    _require(global_max_bytes // preparing_processes > 0, "compact_cache_process_budget_empty")
    return pp.validate_cache_policy({"schema": pp.CACHE_POLICY_SCHEMA,
            "global_max_bytes": global_max_bytes, "preparing_processes": preparing_processes})


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(_jsonable(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class ProductionReservationSource:
    """Bootstrap only with complete scheduled warm-TRAIN bindings; never a pool."""

    def __init__(self, paths: ProductionArtifactPaths, *, split: str = "train") -> None:
        _require(split == "train", "train_only")
        self.paths = paths
        self.loader = None

    def schedule(self, *, measured_draws: int, warmup_draws: int, reserved_bytes: int):
        # Shared EXACT descriptor factory with the opt-in prefetch replay.
        from music_critic.experiments.analysisgnn.performance_prefetch_replay import build_prefetch_schedule
        return build_prefetch_schedule(mode="production", paths=self.paths,
            measured_draws=measured_draws, warmup_draws=warmup_draws, reserved_bytes=reserved_bytes)

    def prebind(self, jobs: tuple[pp.PrefetchJob, ...]) -> dict:
        self.loader = FrozenProductionRecordLoader(paths=self.paths, split="train")
        _require(type(self.loader) is FrozenProductionRecordLoader, "concrete_loader_required")
        record_ids = sorted({record for job in jobs for record in job.record_ids})
        for record_id in record_ids:
            self.loader.check_binding(record_id, split="train")
        manifest = self.loader.validate_all_bindings()
        _require(set(manifest["record_bindings"]) == set(record_ids), "prebind_membership_mismatch")
        return manifest

    def preparer(self, bindings: dict, *, reservation_manifest: dict | None = None,
                 cache_policy: dict | None = None):
        options = {} if cache_policy is None else {"cache_policy": cache_policy}
        return pp.WarmTrainBatchPreparer(self.paths, bindings, identity_only=True,
                                        reservation_manifest=reservation_manifest, **options)

    def finalize(self, bindings: dict) -> dict:
        _require(self.loader is not None, "prebind_required")
        return self.loader.validate_all_bindings(expected=bindings)


def _check_jobs(jobs, *, measured_draws: int, warmup_draws: int, reservation: int):
    jobs = tuple(jobs)
    _require(len(jobs) == (measured_draws + warmup_draws) // 2, "job_count_mismatch")
    for position, job in enumerate(jobs):
        _require(type(job) is pp.PrefetchJob and job.position == position
                 and job.draw_ids == (position * 2, position * 2 + 1), "contiguous_fixed_b2_required")
        _require(job.applied_shifts == (0, 0), "identity_only")
        _require(job.reserved_bytes == reservation, "reservation_changed")
    return jobs


def _measure(source, output: Path, *, experiment_id: str, config: dict) -> dict:
    # The immutable descriptor covers warmup prefix + measured suffix once.
    # Future worker counts and repetitions reuse these exact jobs/IDs.
    jobs, descriptor = source.schedule(measured_draws=config["measured_draws"],
        warmup_draws=config["warmup_draws"], reserved_bytes=config["max_job_payload_bytes"])
    jobs = _check_jobs(jobs, measured_draws=config["measured_draws"], warmup_draws=config["warmup_draws"],
                       reservation=config["max_job_payload_bytes"])
    from music_critic.experiments.analysisgnn.performance_prefetch_replay import jobs_from_prefetch_descriptor
    _require(tuple(jobs_from_prefetch_descriptor(descriptor)) == jobs, "descriptor_job_mismatch")
    _require(descriptor["mode"] == ("fixture" if config["mode"] == "synthetic" else "production")
             and descriptor["measured_draws"] == config["measured_draws"]
             and descriptor["warmup_draws"] == config["warmup_draws"], "descriptor_protocol_mismatch")
    _write_json(output / "schedule.json", descriptor)
    initial = _jsonable(source.prebind(jobs))
    initial_bytes = canonical_json_bytes(initial)
    population = {record for job in jobs for record in job.record_ids}
    _require(isinstance(initial.get("record_bindings"), dict)
             and set(initial["record_bindings"]) == population, "complete_scheduled_bindings_required")
    _write_json(output / "source-bindings.json", initial)
    cache_policy = config.get("cache_policy")
    prepare_options = {} if cache_policy is None else {"cache_policy": _jsonable(cache_policy)}
    prepare = source.preparer(_jsonable(initial), **prepare_options)
    base_fingerprint = prepare.base_source_fingerprint
    measured = {}

    def checked_prepare(job):
        payload = prepare(job)
        _require(type(payload) is pp.PreparedCPUJob and payload.job == job, "prepared_job_mismatch")
        return payload

    cache_diagnostics = None
    try:
        with (output / "payload-sizes.jsonl").open("xb") as stream:
            for job in jobs:
                payload = None
                try:
                    # Exact production serialization/CPU/RNG/reservation guards.
                    payload = pp._prepare_bytes(checked_prepare, job)
                    size = len(payload)
                    _require(0 < size <= job.reserved_bytes, "serialized_size_exceeds_reservation")
                    key = pp.reservation_job_fingerprint(job)
                    _require(key not in measured, "duplicate_job_identity")
                    measured[key] = size
                    row = {"position": job.position, "draw_ids": list(job.draw_ids),
                           "phase": "warmup" if job.position < config["warmup_draws"] // 2 else "measured",
                           "job_fingerprint": key, "reserved_bytes": job.reserved_bytes,
                           "serialized_bytes": size, "payload_sha256": sha256(payload).hexdigest()}
                    stream.write(canonical_json_bytes(row) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    del payload
                    gc.collect()
        if cache_policy is not None:
            cache_diagnostics = _jsonable(prepare.cache_diagnostics())
    finally:
        if cache_policy is not None:
            prepare.reset_cache()
    _require(len(measured) == len(jobs), "incomplete_measured_jobs")
    manifest = pp.make_reservation_manifest(source_fingerprint=base_fingerprint,
        evidence_id=experiment_id, measured_serialized_bytes=measured)
    # No resizing from measured lengths: the same job reservation integers remain
    # in the final descriptor and hence in every replay serialization.
    final_prepare = source.preparer(_jsonable(initial), reservation_manifest=manifest, **prepare_options)
    _require(final_prepare.base_source_fingerprint == base_fingerprint, "source_fingerprint_changed")
    final_prepare.validate_jobs(jobs)
    final = _jsonable(source.finalize(_jsonable(initial)))
    _write_json(output / "source-bindings-final.json", final)
    _require(canonical_json_bytes(final) == initial_bytes, "source_bindings_changed")
    _write_json(output / "reservation-manifest.json", manifest)
    result = {"schema_version": GATE_SCHEMA, "measured_jobs": len(jobs),
            "warmup_jobs": config["warmup_draws"] // 2, "future_measured_jobs": config["measured_draws"] // 2,
            "scheduled_train_records": len(population), "base_source_fingerprint": base_fingerprint,
            "final_preparer_source_fingerprint": final_prepare.source_fingerprint,
            "reservation_manifest_fingerprint": manifest["fingerprint"],
            "schedule_sha256": file_sha256(output / "schedule.json"),
            "source_bindings_sha256": file_sha256(output / "source-bindings.json"),
            "reservation_manifest_sha256": file_sha256(output / "reservation-manifest.json"),
            "payload_sizes_sha256": file_sha256(output / "payload-sizes.jsonl"),
            "max_observed_serialized_bytes": max(measured.values()),
            "total_observed_serialized_bytes": sum(measured.values()),
            "retained_per_job_reserved_bytes": config["max_job_payload_bytes"],
            "planned_global_buffer_byte_cap": config["buffer_byte_cap"],
            "planned_max_pending_batches": config["max_pending_batches"],
            "planned_max_admitted_jobs": min(config["max_pending_batches"],
                                              config["buffer_byte_cap"] // config["max_job_payload_bytes"]),
            "limits": list(LIMITS)}
    if cache_policy is not None:
        result.update(preparation_strategy="compact_cache", cache_policy=_jsonable(cache_policy),
            cache_diagnostics=cache_diagnostics,
            reservation_compatibility={
                "binding": "base_source_fingerprint_and_exact_jobs_runtime_reservations",
                "cache_capacity_in_reservation_base_identity": False,
                "instance_source_fingerprint_includes_cache_policy": True,
                "serialized_size_invariant_across_cache_histories": False,
                "scope": "Cache capacity/topology must still be declared in the consuming replay protocol; "
                         "reported sizes are observations for this policy/history, not an upper bound for others. "
                         "Pickle memoization may change lengths despite equal values. Actual payload size must "
                         "fit the retained reservation at every serialization. This gate has no worker pool."})
        result["limits"].append(COMPACT_SIZE_LIMIT)
    return result


def _environment() -> dict:
    return _jsonable({"python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(), "torch": str(torch.__version__),
        "pyg": str(torch_geometric.__version__), "torch_cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "packages": sorted((item.metadata.get("Name", "unknown"), item.version) for item in distributions()),
        "environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED")}})


def run_registered_reservations(
    *, experiment_id: str, parent_experiment_id: str, output_root: Path,
    measured_draws: int, warmup_draws: int, max_job_payload_bytes: int,
    buffer_byte_cap: int, max_pending_batches: int,
    paths: ProductionArtifactPaths | None = None, split: str = "train",
    repository_root: Path | None = None, command: list[str] | None = None,
    synthetic: bool = False, synthetic_source_factory: Callable | None = None,
    preparation_strategy: str = "frozen", compact_cache_global_max_bytes: int | None = None,
    compact_cache_preparing_processes: int | None = None,
) -> dict:
    """Register before schedule/source startup; preserve negative/partial results."""
    _require(split == "train", "train_only")
    cache_policy = _cache_policy(preparation_strategy, compact_cache_global_max_bytes,
                                 compact_cache_preparing_processes)
    _require(isinstance(experiment_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id)
             is not None, "invalid_experiment_id")
    for name, value, minimum in (("measured_draws", measured_draws, 2), ("warmup_draws", warmup_draws, 0)):
        _require(type(value) is int and value >= minimum and value % 2 == 0, "invalid_" + name)
    _require(type(max_job_payload_bytes) is int and max_job_payload_bytes > 0, "positive_explicit_payload_cap_required")
    pp.PrefetchConfig(workers=0, buffer_byte_cap=buffer_byte_cap, max_pending_batches=max_pending_batches)
    _require(max_job_payload_bytes <= buffer_byte_cap, "reservation_exceeds_global_cap")
    _require(type(synthetic) is bool and ((synthetic and paths is None and synthetic_source_factory is not None)
             or (not synthetic and isinstance(paths, ProductionArtifactPaths) and synthetic_source_factory is None)),
             "explicit_production_or_synthetic_source_required")
    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    output = Path(output_root).resolve()
    records = root / "docs/experiments/records"
    record_path = records / f"{experiment_id}.json"
    _require(not record_path.exists() and not record_path.is_symlink() and not output.exists(),
             "fresh_experiment_and_output_required")
    _require(output != root and not (root in output.parents and not ((root / "outputs") in output.parents
             or (root / "artifacts/experiments") in output.parents)), "output_must_be_ignored_or_external")
    _require(parent_experiment_id is not None, "terminal_parent_required")
    parents = _validated_parent_ids(records, experiment_id, parent_experiment_id)
    manifest, environment = source_manifest(root), _environment()
    config = {"schema_version": GATE_SCHEMA, "mode": "synthetic" if synthetic else "production", "split": split,
        "measured_draws": measured_draws, "warmup_draws": warmup_draws, "batch_size": 2,
        "max_job_payload_bytes": max_job_payload_bytes, "buffer_byte_cap": buffer_byte_cap,
        "max_pending_batches": max_pending_batches, "workers_for_this_gate": 0,
        "reservation_policy": "one_global_provisional_value_retained_in_every_final_job",
        "paths": None if paths is None else {field.name: str(getattr(paths, field.name)) for field in fields(paths)}}
    if cache_policy is not None:
        config.update(preparation_strategy=preparation_strategy, cache_policy=cache_policy)
    protocol = {key: value for key, value in config.items() if key != "paths"}
    protocol.update(config_fingerprint=sha256(canonical_json_bytes(config)).hexdigest(),
        environment_fingerprint=sha256(canonical_json_bytes(environment)).hexdigest(),
        prefetch_implementation_fingerprint=pp._implementation_fingerprint())
    state = {"git_commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "diff_sha256": sha256(_git(root, "diff", "--binary", "HEAD").encode()).hexdigest(),
        "source_manifest_fingerprint": manifest["fingerprint"]}
    policy = {name: False for name in ("validation_allowed", "test_inputs_allowed", "test_targets_allowed",
                                      "test_metrics_allowed", "test_selection_allowed")}
    policy["source"] = "docs/PERFORMANCE_REPLAY.md; TRAIN-only reservation gate"
    record = {"schema_version": "1.0.0", "experiment_id": experiment_id, "parent_ids": parents,
        "kind": "data_audit", "phase": "9E-P1", "status": "planned",
        "title": "Warm TRAIN serialized job reservation gate for B2 prefetch",
        "question": "Do exact scheduled CPU job payloads fit the explicitly bounded retained reservations?",
        "hypothesis": "Sequential serialization provides source-bound sizes for the exact future B2 jobs.",
        "decision_use": "Supply a sealed reservation manifest to a separate prefetch replay; no RAM/GPU acceptance.",
        "repository": state, "seeds": [], "protocol": protocol,
        "budget": {"planned": (measured_draws + warmup_draws) // 2, "actual": None, "unit": "serialized_B2_jobs"},
        "data_access": {**{name: None for name in DATA_ACCESS_FIELDS}, "declared_policy": policy,
                        "access_evidence_status": "pending_primary_evidence"},
        "results": {}, "validity": {"valid": None},
        "claims": {"supported": [], "not_supported": ["RSS bound", "OOM safety", "GPU capacity",
                   "production speedup", "C2 equivalence", "model quality"], "caveats": list(LIMITS)},
        "artifacts": [], "import_provenance": {"mode": "synthetic_test" if synthetic else "local_execution"},
        "created_at": _now(), "started_at": None}
    if cache_policy is not None:
        record["claims"]["caveats"].append(COMPACT_SIZE_LIMIT)
        record["claims"]["not_supported"].append("capacity-independent serialized payload bound")
    lifecycle = [_transition(record_path, record)]
    owned, error, result = False, None, {}
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        record.update(status="running", started_at=_now())
        lifecycle.append(_transition(record_path, record))
        for name, value in (("lifecycle-start.json", lifecycle), ("resolved-config.json", config),
                            ("environment.json", environment), ("source-manifest-initial.json", manifest)):
            _write_json(output / name, value)
        source = synthetic_source_factory() if synthetic else ProductionReservationSource(paths, split=split)
        result = _measure(source, output, experiment_id=experiment_id, config=config)
        final_manifest = source_manifest(root)
        _write_json(output / "source-manifest-final.json", final_manifest)
        _require(final_manifest == manifest and pp._implementation_fingerprint()
                 == protocol["prefetch_implementation_fingerprint"], "source_changed_during_execution")
        result.update(experiment_id=experiment_id, mode=protocol["mode"],
                      config_fingerprint=protocol["config_fingerprint"],
                      environment_fingerprint=protocol["environment_fingerprint"])
        _write_json(output / "summary.json", result)
        record.update(status="completed", results=result, validity={"valid": True})
        record["budget"]["actual"] = result["measured_jobs"]
        record["claims"]["supported"] = ["Exact scheduled serialized payload sizes fit the unchanged declared reservations."]
    except BaseException as exc:
        error = exc
        record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      validity={"valid": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if owned:
                partial = output / "payload-sizes.jsonl"
                if partial.is_file():
                    with partial.open("rb") as stream:
                        record["budget"]["actual"] = sum(1 for line in stream if line.endswith(b"\n") and line.strip())
                _write_json(output / "execution-evidence.json", {"protocol": protocol, "repository": state,
                    "lifecycle": lifecycle, "argv": list(sys.argv if command is None else command),
                    "failure": None if error is None else repr(error),
                    "access_observation_limits": "Guarded scheduled TRAIN bindings only; no independent OS I/O trace."})
                record["artifacts"] = _artifact_rows(root, output, experiment_id)
        except BaseException as exc:
            record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          validity={"valid": False, "reason": f"evidence_finalization_failed:{type(exc).__name__}: {exc}"})
            record["artifacts"], record["claims"]["supported"] = [], []
            if error is None:
                error = exc
        record["ended_at"] = _now()
        record["budget"]["wall_seconds"] = time.perf_counter() - started
        _transition(record_path, record)
        (root / "docs/EXPERIMENT_LEDGER.md").write_text(render_registry(records), encoding="utf-8")
    if error is not None:
        raise error
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-id", "parent-experiment-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--preparation-strategy", choices=("frozen", "compact_cache"), default="frozen")
    parser.add_argument("--compact-cache-global-max-bytes", type=int)
    parser.add_argument("--compact-cache-preparing-processes", type=int, choices=(1, 2, 4))
    for name in ("measured-draws", "warmup-draws", "max-job-payload-bytes", "buffer-byte-cap", "max-pending-batches"):
        parser.add_argument("--" + name, type=int, required=True)
    for field in fields(ProductionArtifactPaths):
        parser.add_argument("--" + field.name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_registered_reservations(experiment_id=args.experiment_id,
        parent_experiment_id=args.parent_experiment_id, output_root=args.output_root, split=args.split,
        measured_draws=args.measured_draws, warmup_draws=args.warmup_draws,
        max_job_payload_bytes=args.max_job_payload_bytes, buffer_byte_cap=args.buffer_byte_cap,
        max_pending_batches=args.max_pending_batches,
        preparation_strategy=args.preparation_strategy,
        compact_cache_global_max_bytes=args.compact_cache_global_max_bytes,
        compact_cache_preparing_processes=args.compact_cache_preparing_processes,
        paths=ProductionArtifactPaths(**{field.name: getattr(args, field.name) for field in fields(ProductionArtifactPaths)}),
        command=[sys.executable, __file__, *(sys.argv[1:] if argv is None else argv)])
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
