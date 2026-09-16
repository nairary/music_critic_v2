"""Isolated B2 replay of original CPU preparation plus the optimized update.

Worker zero calls the original warm preparer directly. Multiprocess workers
pay their actual serialization/IPC/queue costs. No CUDA or model initialization
occurs at import; production launch belongs to the registered guarded CLI.
"""

from __future__ import annotations

from music_critic.cuda_memory import initialize_cuda_memory_statistics
from music_critic.device import resolve_runtime_device

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import re
import time
from unittest.mock import patch

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn import performance_replay as existing
from music_critic.experiments.analysisgnn.performance_measurement import (
    EqualDrawBenchmarkPlan, StageTimer, raw_graph_counts, resource_snapshot, summarize_window,
)
from music_critic.experiments.analysisgnn.performance_prefetch import (
    OrderedPrefetch, PrefetchConfig, PrefetchJob, PreparedCPUJob, WarmTrainBatchPreparer,
    WORKER_RUNTIME_SCHEMA, cache_observation, cache_instance_fingerprint, validate_cache_policy,
)
from music_critic.experiments.analysisgnn.performance_training import (
    attempt_performance_update, prepare_performance_class_weights,
)


SCHEDULE_SCHEMA = "AnalysisGNNPrefetchSchedule@1.0.0"
REPLAY_SCHEMA = "AnalysisGNNPrefetchReplay@1.0.0"


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def load_production_prefetch_inputs(*, schedule_path, binding_path, reservation_path, records_root):
    """Authenticate portable sizing artifacts against their completed record."""
    from music_critic.experiments.registry import load_record
    paths = {"schedule.json": Path(schedule_path), "source-bindings.json": Path(binding_path),
             "reservation-manifest.json": Path(reservation_path)}
    payloads = {}
    for role, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("prefetch_replay.regular_input_artifact_required")
        payloads[role] = path.read_bytes()
    values = {role: json.loads(payload) for role, payload in payloads.items()}
    evidence_id = values["reservation-manifest.json"].get("evidence_id")
    if not isinstance(evidence_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", evidence_id) is None:
        raise ValueError("prefetch_replay.reservation_evidence_id_invalid")
    record_path = Path(records_root) / f"{evidence_id}.json"
    if record_path.is_symlink():
        raise ValueError("prefetch_replay.regular_evidence_record_required")
    record = load_record(record_path)
    if record["status"] != "completed" or record["experiment_id"] != evidence_id:
        raise ValueError("prefetch_replay.completed_reservation_evidence_required")
    gate_protocol = record.get("protocol", {})
    if (record.get("kind") != "data_audit" or gate_protocol.get("mode") != "production"
            or gate_protocol.get("split") != "train"
            or gate_protocol.get("schema_version") != "AnalysisGNNPrefetchReservationGate@1.0.0"):
        raise ValueError("prefetch_replay.production_train_reservation_gate_required")
    for role, payload in payloads.items():
        matches = [row for row in record["artifacts"] if row["role"] == role]
        if len(matches) != 1 or matches[0]["sha256"] != sha256(payload).hexdigest() or matches[0]["size_bytes"] != len(payload):
            raise ValueError("prefetch_replay.reservation_artifact_mismatch:" + role)
    return {"descriptor": values["schedule.json"], "binding_manifest": values["source-bindings.json"],
            "reservation_manifest": values["reservation-manifest.json"], "reservation_evidence_id": evidence_id,
            "reservation_protocol": gate_protocol,
            "input_artifact_sha256": {role: sha256(payload).hexdigest() for role, payload in payloads.items()}}


def build_prefetch_schedule(*, mode: str, paths=None, measured_draws: int,
                            warmup_draws: int, reserved_bytes: int):
    """Return (immutable jobs, sealed JSON descriptor), shared with sizing gate.

    One absolute schedule contains the warmup prefix followed by measured jobs.
    Repeats and worker counts reuse it verbatim. Reservation remains explicit;
    the size gate keeps its provisional value unchanged after measurement.
    """
    if mode not in ("fixture", "production"):
        raise ValueError("prefetch_replay.invalid_mode")
    EqualDrawBenchmarkPlan(measured_draws, (2,), fixture_smoke=mode == "fixture")
    if type(warmup_draws) is not int or warmup_draws < 0 or warmup_draws % 2:
        raise ValueError("prefetch_replay.invalid_warmup_draws")
    if type(reserved_bytes) is not int or reserved_bytes <= 0:
        raise ValueError("prefetch_replay.explicit_reservation_required")
    if mode == "production" and paths is None:
        raise ValueError("prefetch_replay.explicit_production_paths_required")
    schedule = _json(existing._schedule(mode, paths, measured_draws + warmup_draws))
    jobs = tuple(PrefetchJob(i // 2, (i, i + 1),
        tuple(row["record_id"] for row in schedule[i:i+2]),
        tuple(row["schedule_shift_pc"] for row in schedule[i:i+2]),
        tuple(row["applied_shift_pc"] for row in schedule[i:i+2]), reserved_bytes)
        for i in range(0, len(schedule), 2))
    if len(schedule) != measured_draws + warmup_draws or any(any(job.applied_shifts) for job in jobs):
        raise ValueError("prefetch_replay.identity_schedule_required")
    body = _json({"schema": SCHEDULE_SCHEMA, "mode": mode, "batch_size": 2,
        "measured_draws": measured_draws, "warmup_draws": warmup_draws,
        "schedule": schedule, "jobs": [asdict(job) for job in jobs],
        "schedule_fingerprint": existing.recursive_state_fingerprint(schedule)})
    return jobs, {**body, "fingerprint": fingerprint(body)}


def jobs_from_prefetch_descriptor(descriptor: Mapping[str, object]) -> tuple[PrefetchJob, ...]:
    """Validate a sealed descriptor without loading source graphs or targets."""
    body = _json(dict(descriptor))
    seal = body.pop("fingerprint", None)
    if seal != fingerprint(body) or body.get("schema") != SCHEDULE_SCHEMA or body.get("batch_size") != 2:
        raise ValueError("prefetch_replay.descriptor_binding_mismatch")
    mode, measured, warmup = body.get("mode"), body.get("measured_draws"), body.get("warmup_draws")
    if mode not in ("fixture", "production") or type(warmup) is not int or warmup < 0 or warmup % 2:
        raise ValueError("prefetch_replay.invalid_descriptor_plan")
    EqualDrawBenchmarkPlan(measured, (2,), fixture_smoke=mode == "fixture")
    rows, raw_jobs = body.get("schedule"), body.get("jobs")
    if not isinstance(rows, list) or not isinstance(raw_jobs, list) or len(rows) != measured + warmup or len(raw_jobs) * 2 != len(rows):
        raise ValueError("prefetch_replay.invalid_descriptor_schedule")
    if body.get("schedule_fingerprint") != existing.recursive_state_fingerprint(rows):
        raise ValueError("prefetch_replay.descriptor_schedule_mismatch")
    jobs = []
    for i, values in enumerate(raw_jobs):
        try:
            job = PrefetchJob(values["position"], tuple(values["draw_ids"]), tuple(values["record_ids"]),
                tuple(values["schedule_shifts"]), tuple(values["applied_shifts"]), values["reserved_bytes"])
            pair = rows[2*i:2*i+2]
            valid = (job.position == i and job.draw_ids == (2*i, 2*i+1)
                and job.record_ids == tuple(row["record_id"] for row in pair)
                and job.schedule_shifts == tuple(row["schedule_shift_pc"] for row in pair)
                and job.applied_shifts == tuple(row["applied_shift_pc"] for row in pair)
                and job.applied_shifts == (0, 0) and values == _json(asdict(job)))
        except (TypeError, KeyError) as exc:
            raise ValueError("prefetch_replay.invalid_descriptor_job") from exc
        if not valid:
            raise ValueError("prefetch_replay.descriptor_job_mismatch")
        jobs.append(job)
    return tuple(jobs)


class FixturePreparer:
    """Source-free fixture using the same original singleton collation path."""

    def __init__(self, cache_policy=None):
        self.cache_policy = validate_cache_policy(cache_policy)
        self.cache_instance_fingerprint = cache_instance_fingerprint(self.cache_policy, max_entries=2)
        self._cache = None

    def __getstate__(self):
        return (self.cache_policy,)

    def __setstate__(self, state):
        self.__init__(state[0])

    def reset_cache(self):
        self._cache = None

    def cache_diagnostics(self):
        return cache_observation(self._cache, self.cache_policy, max_entries=2)

    def __call__(self, job):
        if self.cache_policy is not None:
            if self._cache is None:
                from music_critic.experiments.analysisgnn.performance_data import PreparedGraphCache, SyntheticRecordLoader
                self._cache = PreparedGraphCache(SyntheticRecordLoader(existing._fixture_callback,
                    source_seal="synthetic-performance-v1", synthetic=True), synthetic=True,
                    max_bytes=self.cache_policy["global_max_bytes"] // self.cache_policy["preparing_processes"],
                    max_entries=2)
            loaded = self._cache.prepare_batch(job.record_ids)
            raw = existing.reference.transpose_raw_graph_batch(loaded.raw_graph_batch, job.applied_shifts)
            return PreparedCPUJob(job, raw, loaded.sidecars)
        loaded = [existing._fixture_callback(record) for record in job.record_ids]
        raw = existing.reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
        raw = existing.reference.transpose_raw_graph_batch(raw, job.applied_shifts)
        return PreparedCPUJob(job, raw, tuple(row[1] for row in loaded))


class BatchStream:
    """Parent direct control or real multiprocessing transport, never fake IPC.

    No pickle encode/decode, cache, or eager graph queue exists in worker zero.
    Checkpoint cursor means applied batches; parent training state is separate.
    """

    def __init__(self, jobs, preparer, *, source_fingerprint, config, synthetic, checkpoint=None,
                 direct_reference=False):
        if type(direct_reference) is not bool or (direct_reference and
                (config.workers != 0 or getattr(preparer, "cache_policy", None) is not None)):
            raise ValueError("prefetch_replay.original_direct_control_required")
        self.jobs, self.preparer, self.config = tuple(jobs), preparer, config
        self.pool, self._cursor, self._delivered, self._closed = None, 0, None, False
        if config.workers:
            self.pool = OrderedPrefetch(self.jobs, preparer, source_fingerprint=source_fingerprint,
                config=config, synthetic=synthetic, checkpoint=checkpoint)
            return
        if type(preparer) is WarmTrainBatchPreparer:
            if synthetic or source_fingerprint != preparer.source_fingerprint:
                raise ValueError("prefetch_replay.direct_source_mismatch")
            (preparer.validate_direct_jobs if direct_reference else preparer.validate_jobs)(self.jobs)
        elif type(preparer) is not FixturePreparer or not synthetic:
            raise ValueError("prefetch_replay.direct_concrete_preparer_required")
        self._bindings = {"schema": "AnalysisGNNDirectBatchCursor@1.0.0", "source_fingerprint": source_fingerprint,
            "schedule_fingerprint": fingerprint([asdict(job) for job in self.jobs]),
            "config_fingerprint": fingerprint(asdict(config)), "source_mode": "synthetic" if synthetic else "frozen_warm_train"}
        if direct_reference:
            self._bindings["direct_reference_without_transport_reservations"] = True
        if checkpoint is not None:
            body = _json(checkpoint)
            seal = body.pop("fingerprint", None)
            cursor = body.pop("committed_batch_cursor", None)
            if type(cursor) is not int or not 0 <= cursor <= len(self.jobs) or body != self._bindings or seal != fingerprint({**body, "committed_batch_cursor": cursor}):
                raise ValueError("prefetch_replay.direct_resume_binding_mismatch")
            self._cursor = cursor

    def peek(self):
        if self._closed:
            raise ValueError("prefetch_replay.stream_closed")
        if self.pool is not None:
            return self.pool.peek()
        if self._cursor == len(self.jobs):
            raise StopIteration
        result = self.preparer(self.jobs[self._cursor])
        self._delivered = self._cursor
        return result

    def commit(self, position, *, applied):
        if self.pool is not None:
            return self.pool.commit(position, applied=applied)
        if self._closed or type(position) is not int or position != self._cursor or self._delivered != position:
            raise ValueError("prefetch_replay.direct_commit_invalid")
        if type(applied) is not bool:
            raise ValueError("prefetch_replay.applied_boolean_required")
        if applied:
            self._cursor += 1
            self._delivered = None
        return applied

    def state_dict(self):
        if self.pool is not None:
            return self.pool.state_dict()
        body = {**self._bindings, "committed_batch_cursor": self._cursor}
        return {**body, "fingerprint": fingerprint(body)}

    def worker_runtime_evidence(self):
        return self.pool.worker_runtime_evidence() if self.pool is not None else {
            "schema": WORKER_RUNTIME_SCHEMA, "configured_workers": 0, "required_workers": 0,
            "started_workers": 0, "observations_complete": True, "observations": [],
            "native_thread_limits_all_one": None, "spawn_parent_native_thread_environment": None,
            "scope": "direct_parent_preparation_no_worker_acknowledgments"}

    def diagnostics(self, *, include_worker_runtime=True):
        if self.pool is not None:
            return self.pool.diagnostics(include_worker_runtime=include_worker_runtime)
        result = {
            "committed_batch_cursor": self._cursor, "pending_batches": 0, "reserved_bytes": 0,
            "worker_pids": (), "transport": "direct_no_serialization", "closed": self._closed}
        if include_worker_runtime:
            result["worker_runtime"] = self.worker_runtime_evidence()
        if getattr(self.preparer, "cache_policy", None) is not None:
            result["prepared_cache"] = self.preparer.cache_diagnostics()
        return result

    def close(self):
        self._closed = True
        if self.pool is not None:
            self.pool.close()


def worker_resources(pids: Sequence[int]) -> dict[str, object]:
    """Sample own worker /proc counters; unavailable is never reported as zero."""
    rows = []
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            values = stat[stat.rfind(")")+2:].split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
            rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
            threads = next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
            ticks = os.sysconf("SC_CLK_TCK")
            rows.append({"pid": pid, "cpu_user_seconds": int(values[11]) / ticks,
                         "cpu_system_seconds": int(values[12]) / ticks, "rss_bytes": rss, "os_threads": threads})
        except (OSError, ValueError, StopIteration, IndexError):
            rows.append({"pid": pid, "cpu_user_seconds": None, "cpu_system_seconds": None, "rss_bytes": None, "os_threads": None})
    return {"workers": rows, "scope": "own_worker_proc_samples_not_peak_RSS_or_total_process_tree",
            "aggregate_sampled_rss_bytes": (sum(row["rss_bytes"] for row in rows)
                if rows and all(row["rss_bytes"] is not None for row in rows) else None)}


def worker_configurations(config: PrefetchConfig, worker_counts: Sequence[int]):
    if not isinstance(config, PrefetchConfig) or config.workers != 0:
        raise ValueError("prefetch_replay.base_config_workers_zero_required")
    counts = tuple(worker_counts)
    if not counts or 0 not in counts or len(set(counts)) != len(counts) or any(type(n) is not int or n not in (0, 1, 2, 4) for n in counts):
        raise ValueError("prefetch_replay.worker_counts_invalid")
    return {n: replace(config, workers=n) for n in counts}


def cache_policies(preparation_strategy, compact_cache_global_max_bytes, worker_counts):
    """One retained-entry budget, partitioned only among preparing processes."""
    if preparation_strategy not in ("frozen", "compact_cache"):
        raise ValueError("prefetch_replay.invalid_preparation_strategy")
    if preparation_strategy == "frozen":
        if compact_cache_global_max_bytes is not None:
            raise ValueError("prefetch_replay.cache_budget_requires_compact_strategy")
        return {}
    if (type(compact_cache_global_max_bytes) is not int or
            compact_cache_global_max_bytes < max(max(worker_counts), 1)):
        raise ValueError("prefetch_replay.explicit_positive_aggregate_cache_budget_required")
    return {str(n): {"schema": "AnalysisGNNProcessLocalCompactCache@1.0.0",
                    "global_max_bytes": compact_cache_global_max_bytes,
                    "preparing_processes": max(n, 1)} for n in worker_counts}


def protocol_bindings(*, mode, descriptor, reservation_manifest, config, worker_counts,
                      preparation_strategy="frozen", compact_cache_global_max_bytes=None,
                      include_original_baseline=False):
    configs = worker_configurations(config, worker_counts)
    jobs = jobs_from_prefetch_descriptor(descriptor)
    if mode != descriptor["mode"] or any(job.reserved_bytes > config.buffer_byte_cap for job in jobs):
        raise ValueError("prefetch_replay.descriptor_config_mismatch")
    if mode == "production" and not isinstance(reservation_manifest, Mapping):
        raise ValueError("prefetch_replay.production_reservation_manifest_required")
    policies = cache_policies(preparation_strategy, compact_cache_global_max_bytes, tuple(configs))
    if type(include_original_baseline) is not bool or (include_original_baseline and not policies):
        raise ValueError("prefetch_replay.original_comparison_requires_integrated_strategy")
    result = {"comparison_kind": "ordered_prefetch", "planned_arm_count": len(configs),
        "worker_counts": list(configs), "worker_configs_fingerprint": fingerprint({str(n): asdict(c) for n, c in configs.items()}),
        "schedule_descriptor_fingerprint": descriptor["fingerprint"],
        "reservation_manifest_fingerprint": (reservation_manifest["fingerprint"] if mode == "production" else
            fingerprint({"synthetic_reservation_policy": "explicit_bytes_per_job", "reservations": [job.reserved_bytes for job in jobs],
                         "schedule_descriptor_fingerprint": descriptor["fingerprint"]})),
        "reference_data_strategy": "frozen", "optimized_data_strategy": "frozen"}
    if policies:
        result.update(reference_data_strategy="compact_cache", optimized_data_strategy="compact_cache",
            preparation_strategy=preparation_strategy, compact_cache_global_max_bytes=compact_cache_global_max_bytes,
            cache_policies=policies, cache_policies_fingerprint=fingerprint(policies),
            cache_start_policy="cold_process_local_per_arm_no_prepared_population_prewarm",
            cache_budget_scope="retained_entries_only_partitioned_across_preparing_processes_not_RSS_or_IPC")
    if include_original_baseline:
        result.update(comparison_kind="ordered_prefetch_with_original", planned_arm_count=len(configs) + 1,
            reference_data_strategy="frozen",
            include_original_baseline=True,
            arm_strategies=[{"arm_key": f"integrated-workers{n}", "worker_count": n,
                "preparation_strategy": "compact_cache", "update_strategy": "optimized"} for n in configs] +
                [{"arm_key": "original-workers0", "worker_count": 0,
                  "preparation_strategy": "frozen", "update_strategy": "reference"}])
    return result


def _prewarm(preparer, population):
    def forbidden(*args, **kwargs):
        raise RuntimeError("prefetch_replay.cold_conversion_forbidden")
    for record in population:
        if type(preparer) is FixturePreparer:
            existing._fixture_callback(record)
        else:
            if preparer._loader is None:
                preparer.start()
            preparer._loader.check_binding(record, split="train")
            with patch.object(existing.reference, "_resolve_selected_corpus_record", forbidden):
                existing.reference.load_production_record(record, split="train", paths=preparer.paths)
            preparer._loader.check_binding(record, split="train")


def _components(device):
    random.seed(17)
    np.random.seed(17)
    torch.random.default_generator.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    model = existing.CorrectedAnalysisGNNModel().to(device).train()
    random.seed(17001)
    np.random.seed(17001)
    torch.random.default_generator.manual_seed(17001)
    if device == "cuda":
        torch.cuda.manual_seed_all(17001)
    optimizer, scheduler = existing.build_orbit_matched_control_optimizer_scheduler(model)
    scaler, api = existing._scaler()
    weights = prepare_performance_class_weights(existing.reference.load_frozen_class_weights(), device)
    return model, optimizer, scheduler, scaler, weights, api


def _step(stream, components, device, stage, *, update_strategy="optimized", reference_weights=None):
    model, optimizer, scheduler, scaler, weights, _ = components
    with stage("direct_prepare" if stream.config.workers == 0 else "queue_wait_decode_dispatch"):
        batch = stream.peek()
    counts = raw_graph_counts(batch.raw_graph_batch)
    counts["transport_serialized_payload_bytes"] = (0 if stream.pool is None else
        len(stream.pool._ready[batch.job.position]))
    with stage("host_to_device"):
        raw = existing.reference.move_raw_graph_batch(batch.raw_graph_batch, device)
    with stage("full_update"):
        kwargs = dict(model=model, raw_graph_batch=raw, sidecars=batch.sidecars,
            shifts=batch.job.applied_shifts, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        if update_strategy == "reference":
            outcome = existing.reference.attempt_applied_update(**kwargs, class_weights=reference_weights)
        else:
            outcome = attempt_performance_update(**kwargs, class_weights=weights, instrumentation=stage)
    if not outcome.applied:
        raise RuntimeError("prefetch_replay.unapplied_update_invalidates_replay")
    stream.commit(batch.job.position, applied=True)
    loss = {name: None if value is None else float(value.detach().cpu()) for name, value in (
        ("total", outcome.loss.total), ("primary", outcome.loss.primary), ("auxiliary", outcome.loss.auxiliary))}
    return {**loss, "gradient_norm": outcome.gradient_norm}, counts, batch.job


def run_prefetch_replay(output_root: Path | str, *, experiment_id: str, mode="fixture", device="cpu",
                        measured_draws=800, warmup_draws=160, repeats=3, worker_counts=(0, 1, 2, 4),
                        config: PrefetchConfig | None = None, paths=None, descriptor=None,
                        binding_manifest=None, reservation_manifest=None, fixture_reserved_bytes=2_000_000,
                        production_input_paths=None, gpu_idle_check=None,
                        preparation_strategy="frozen", compact_cache_global_max_bytes=None,
                        include_original_baseline=False):
    """Fresh registered benchmark; only fixture mode has local size defaults."""
    if gpu_idle_check is not None and not callable(gpu_idle_check):
        raise ValueError("prefetch_replay.gpu_idle_check_must_be_callable")
    if mode not in ("fixture", "production") or device not in ("cpu", "cuda") or type(repeats) is not int or repeats <= 0:
        raise ValueError("prefetch_replay.invalid_configuration")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("prefetch_replay.experiment_id_required")
    if mode == "production" and (device != "cuda" or paths is None or config is None or production_input_paths is None):
        raise ValueError("prefetch_replay.explicit_production_inputs_required")
    input_evidence = None
    if mode == "production":
        input_evidence = load_production_prefetch_inputs(**production_input_paths)
        if (config.buffer_byte_cap != input_evidence["reservation_protocol"].get("buffer_byte_cap")
                or config.max_pending_batches != input_evidence["reservation_protocol"].get("max_pending_batches")):
            raise ValueError("prefetch_replay.sizing_buffer_policy_mismatch")
        for name, supplied in (("descriptor", descriptor), ("binding_manifest", binding_manifest), ("reservation_manifest", reservation_manifest)):
            if supplied is not None and _json(supplied) != input_evidence[name]:
                raise ValueError("prefetch_replay.input_artifact_changed:" + name)
        descriptor, binding_manifest, reservation_manifest = (input_evidence[name] for name in ("descriptor", "binding_manifest", "reservation_manifest"))
    config = config or PrefetchConfig()
    configs = worker_configurations(config, worker_counts)
    policies = cache_policies(preparation_strategy, compact_cache_global_max_bytes, tuple(configs))
    if input_evidence is not None:
        sizing_strategy = input_evidence["reservation_protocol"].get("preparation_strategy", "frozen")
        if sizing_strategy != preparation_strategy:
            raise ValueError("prefetch_replay.sizing_preparation_strategy_mismatch")
    if descriptor is None:
        jobs, descriptor = build_prefetch_schedule(mode=mode, paths=paths, measured_draws=measured_draws,
            warmup_draws=warmup_draws, reserved_bytes=fixture_reserved_bytes)
    else:
        descriptor = _json(descriptor)
        jobs = jobs_from_prefetch_descriptor(descriptor)
        if descriptor["measured_draws"] != measured_draws or descriptor["warmup_draws"] != warmup_draws:
            raise ValueError("prefetch_replay.descriptor_plan_mismatch")
        if len({job.reserved_bytes for job in jobs}) != 1:
            raise ValueError("prefetch_replay.common_explicit_reservation_required")
        _, regenerated = build_prefetch_schedule(mode=mode, paths=paths, measured_draws=measured_draws,
            warmup_draws=warmup_draws, reserved_bytes=jobs[0].reserved_bytes)
        if regenerated != descriptor:
            raise ValueError("prefetch_replay.frozen_schedule_changed")
    bindings = protocol_bindings(mode=mode, descriptor=descriptor, reservation_manifest=reservation_manifest,
                                 config=config, worker_counts=worker_counts,
                                 preparation_strategy=preparation_strategy,
                                 compact_cache_global_max_bytes=compact_cache_global_max_bytes,
                                 include_original_baseline=include_original_baseline)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    protocol = {**bindings, "schema": REPLAY_SCHEMA, "experiment_id": experiment_id, "mode": mode,
        "device": device, "repeats": repeats, "batch_size": 2, "measured_draws_per_arm": measured_draws,
        "warmup_draws_per_arm": warmup_draws, "update_strategy": "optimized", "quality_claims": False,
        "checkpoint_input_used": False, "resume_equivalence_checked": False,
        "validation_allowed": False, "test_allowed": False,
        "warmup_policy": "discard_updates_restore_common_full_snapshot_keep_speculative_CPU_queue",
        "timing_scope": "measured_train_including_prepare_or_queue_decode_H2D_optimized_update_logging_worker_samples",
        "excluded_timing": "fixed_population_prewarm_worker_startup_snapshot_hashes_cleanup",
        "transport_scope": "queue_wait_includes_decode_dispatch_and_blocking; IPC_components_not_separately_attributed",
        "transport_byte_scope": "serialized_CPU_result_payload_only_excludes_outer_pickle_envelope_OS_headers_and_transient_copies",
        "memory_scope": "serialized_reservations_not_RSS; sampled_worker_RSS_excludes_unobserved_peaks",
        "worker_runtime_scope": "per_arm_startup_ready_getters_and_inherited_environment_readback_not_continuous_or_native_pool_telemetry",
        "zero_worker_control": "direct_original_preparer_without_pickle_or_queue",
        "claim_boundary": "synthetic_mechanics_only" if mode == "fixture" else "same_runtime_frozen_B2_throughput_only"}
    if policies:
        protocol.update(zero_worker_control="direct_compact_preparer_without_pickle_or_queue",
            excluded_timing="steady_window_excludes_cold_setup_startup_warmup_restore_hashes_cleanup",
            whole_arm_timing_scope="direct_elapsed_before_fresh_preparer_to_closed_workers_freed_cache_and_CUDA_completion_includes_validation_logging_hashes",
            whole_arm_excludes="shared_model_initialization_initial_snapshot_and_final_pair_evidence_writes",
            comparison_scope="worker_topology_including_cache_partitioning_with_compact_cache_and_optimized_update_in_both_arms")
    if include_original_baseline:
        protocol.update(update_strategy="mixed_reference_and_optimized",
            timing_scope="measured_train_including_prepare_or_queue_decode_H2D_selected_update_logging_worker_samples",
            comparison_scope="integrated_workers4_vs_integrated_workers0_and_each_vs_original_reference_with_same_initial_state_and_draws",
            raw_population_prewarm_policy="none_for_all_arms_existing_sealed_disk_inputs_only_warmup_prefix_is_identical")
    existing._write(root / "protocol.json", protocol)
    existing._write(root / "schedule.json", descriptor)
    existing._write(root / "worker-configs.json", {str(n): asdict(c) for n, c in configs.items()})
    if input_evidence is not None:
        existing._write(root / "input-evidence.json", {key: input_evidence[key] for key in ("reservation_evidence_id", "input_artifact_sha256")})
    replay_started = time.perf_counter()
    try:
        if device == "cuda":
            (gpu_idle_check if gpu_idle_check is not None else existing._assert_idle_gpu)()
            if not torch.cuda.is_available():
                raise RuntimeError("prefetch_replay.cuda_unavailable")
        with existing._preserve_runtime(device == "cuda"):
            preparer = None if policies else (FixturePreparer() if mode == "fixture" else
                WarmTrainBatchPreparer(paths, binding_manifest, reservation_manifest=reservation_manifest))
            source = (None if policies else
                fingerprint({"fixture": "original-source-free-preparer-v1"}) if mode == "fixture" else preparer.source_fingerprint)
            if mode == "production":
                if preparer is not None:
                    preparer.validate_jobs(jobs)
                existing._write(root / "source-bindings.json", binding_manifest)
                existing._write(root / "reservation-manifest.json", reservation_manifest)
            population = tuple(dict.fromkeys(record for job in jobs for record in job.record_ids))
            if not policies:
                _prewarm(preparer, population)
            components = _components(device)
            reference_weights = existing.reference.load_frozen_class_weights() if include_original_baseline else None
            model, optimizer, scheduler, scaler, _, scaler_api = components
            initial = existing._snapshot(model, optimizer, scheduler, scaler)
            initial_hash = existing.recursive_state_fingerprint(initial)
            results = []
            for repeat in range(repeats):
                counts_order = list(configs)[repeat % len(configs):] + list(configs)[:repeat % len(configs)]
                arm_plan = [(n, False) for n in counts_order]
                if include_original_baseline:
                    arm_plan.append((0, True))
                for order, (worker_count, original_control) in enumerate(arm_plan):
                    arm_started = time.perf_counter()
                    gc.collect()
                    if device == "cuda":
                        (gpu_idle_check if gpu_idle_check is not None else existing._assert_idle_gpu)()
                    arm_id = (f"r{repeat:02d}-original-workers0-b2" if original_control else
                              f"r{repeat:02d}-workers{worker_count}-b2")
                    arm_key = "original-workers0" if original_control else f"integrated-workers{worker_count}"
                    update_strategy = "reference" if original_control else "optimized"
                    directory = root / arm_id
                    directory.mkdir()
                    started = time.perf_counter()
                    if policies:
                        policy = None if original_control else policies[str(worker_count)]
                        preparer = (FixturePreparer(policy) if mode == "fixture" else
                            WarmTrainBatchPreparer(paths, binding_manifest,
                                reservation_manifest=None if original_control else reservation_manifest, cache_policy=policy))
                        source = (fingerprint({"fixture": "compact-source-free-preparer-v1", "policy": policy})
                                  if mode == "fixture" else preparer.source_fingerprint)
                        if mode == "production":
                            (preparer.validate_direct_jobs if original_control else preparer.validate_jobs)(jobs)
                            preparer.start()  # Validate source metadata without populating a prepared cache.
                        existing._write(directory / "cache-start.json", preparer.cache_diagnostics())
                    else:
                        _prewarm(preparer, population)
                    prewarm_seconds = time.perf_counter() - started
                    if mode == "production":
                        preparer._loader.validate_all_bindings(expected=binding_manifest)
                    started = time.perf_counter()
                    stream = BatchStream(jobs, preparer, source_fingerprint=source, config=configs[worker_count],
                                         synthetic=mode == "fixture", direct_reference=original_control)
                    startup_seconds = time.perf_counter() - started
                    timer, phase, update_number = StageTimer(), "warmup", None
                    @contextmanager
                    def stage(name):
                        with timer.span(name, phase=phase, update=update_number):
                            yield
                    try:
                        worker_runtime = stream.worker_runtime_evidence()
                        existing._write(directory / "worker-runtime.json", worker_runtime)
                        existing._restore(initial, model, optimizer, scheduler, scaler)
                        started = time.perf_counter()
                        for update_number in range(warmup_draws // 2):
                            _step(stream, components, device, stage, update_strategy=update_strategy,
                                  reference_weights=reference_weights)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        warmup_seconds = time.perf_counter() - started
                        existing._write(directory / "warmup.json", {"seconds": warmup_seconds, "stages": timer.resolve()})
                        if policies:
                            existing._write(directory / "cache-after-warmup.json", stream.diagnostics())
                        existing._restore(initial, model, optimizer, scheduler, scaler)
                        if existing.recursive_state_fingerprint(existing._snapshot(model, optimizer, scheduler, scaler)) != initial_hash:
                            raise RuntimeError("prefetch_replay.initial_snapshot_mismatch")
                        timer, phase = StageTimer(), "measured"
                        if device == "cuda":
                            torch.cuda.synchronize()
                            initialize_cuda_memory_statistics(resolve_runtime_device("cuda"))
                        pids = stream.diagnostics(include_worker_runtime=False)["worker_pids"]
                        before, worker_before = resource_snapshot(), worker_resources(pids)
                        started = time.perf_counter()
                        durations, graph_counts, trace = [], [], []
                        for update_number in range(measured_draws // 2):
                            update_started = time.perf_counter()
                            values, counts, job = _step(stream, components, device, stage,
                                update_strategy=update_strategy, reference_weights=reference_weights)
                            with stage("metrics_logging_and_worker_sample"):
                                existing._append(directory / "metrics.jsonl", {"update": update_number + 1,
                                    "job": asdict(job), "loss": values, "counts": counts,
                                    "queue": stream.diagnostics(include_worker_runtime=False), "workers": worker_resources(pids)})
                            trace.append(values)
                            graph_counts.append(counts)
                            durations.append(time.perf_counter() - update_started)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        after, worker_after = resource_snapshot(), worker_resources(pids)
                        if policies and mode == "production":
                            preparer._loader.validate_all_bindings(expected=binding_manifest)
                        final = existing._snapshot(model, optimizer, scheduler, scaler)
                        summary = {"arm": arm_id, "kind": ("original_baseline" if original_control else
                            "baseline" if worker_count == 0 else "optimized"),
                            "repeat": repeat, "position": order, "worker_count": worker_count, "batch_size": 2,
                            "data_strategy": "frozen" if original_control else preparation_strategy,
                            "update_strategy": update_strategy,
                            "initial_state_fingerprint": initial_hash,
                            "final_state_fingerprint": existing.recursive_state_fingerprint(final),
                            "final_rng_fingerprint": existing.recursive_state_fingerprint(final["rng"]),
                            "loss_trace_fingerprint": existing.recursive_state_fingerprint(trace),
                            "measured_schedule_fingerprint": existing.recursive_state_fingerprint(descriptor["schedule"][warmup_draws:]),
                            "attempted_updates": len(trace), "applied_updates": len(trace),
                            "warmup_updates": warmup_draws // 2, "warmup_seconds": warmup_seconds,
                            "startup_seconds": startup_seconds, "prewarm_seconds": prewarm_seconds,
                            "prefetch_cursor": stream.state_dict(), "queue_final": stream.diagnostics(),
                            "worker_runtime": worker_runtime,
                            "worker_resources_before": worker_before, "worker_resources_after": worker_after,
                            "window": summarize_window(wall_seconds=seconds, counts=graph_counts, update_seconds=durations,
                                resource_before=before, resource_after=after, cuda_device=device if device == "cuda" else None)}
                        if policies:
                            summary.update(arm_key=arm_key, cache_policy=policy, preparer_source_fingerprint=source,
                                cold_setup_seconds=prewarm_seconds,
                                cache_start_policy=bindings["cache_start_policy"],
                                cache_before_close=stream.diagnostics().get("prepared_cache"),
                                parent_cache_before_close=preparer.cache_diagnostics())
                        del final  # Do not tax later arms with an unused full CPU Adam snapshot.
                        existing._write(directory / "stages.json", timer.resolve())
                    finally:
                        started = time.perf_counter()
                        stream.close()
                        if policies:
                            preparer.reset_cache()
                            gc.collect()
                            if device == "cuda":
                                torch.cuda.synchronize()
                        cleanup_seconds = time.perf_counter() - started
                    summary["cleanup_seconds"] = cleanup_seconds
                    summary["whole_arm_wall_seconds"] = time.perf_counter() - arm_started
                    if policies:
                        summary["cache_after_cleanup"] = preparer.cache_diagnostics()
                    existing._write(directory / "summary.json", summary)
                    results.append(summary)
            if mode == "production" and not policies:
                preparer._loader.validate_all_bindings(expected=binding_manifest)
            keys = ("final_state_fingerprint", "final_rng_fingerprint", "loss_trace_fingerprint", "measured_schedule_fingerprint")
            equivalences = []
            for repeat in range(repeats):
                control = next(row for row in results if row["repeat"] == repeat and
                    (row["kind"] == "original_baseline" if include_original_baseline else row["worker_count"] == 0))
                for row in results:
                    if row["repeat"] == repeat and (row is not control if include_original_baseline else row["worker_count"]):
                        equivalences.append({"repeat": repeat, "worker_count": row["worker_count"],
                                             **({"arm_key": row["arm_key"], "control_arm_key": control["arm_key"]} if include_original_baseline else {}),
                                             **{key: control[key] == row[key] for key in keys}})
            passed = bool(equivalences) and all(all(row[key] for key in keys) for row in equivalences)
            result = {**protocol, "completed": True, "equivalence_passed": passed,
                "b2_equivalence": equivalences, "arms": results, "scaler_api": scaler_api,
                "data_access": {key: None for key in ("validation_accessed", "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed", "test_used_for_selection")},
                "data_access_observation_scope": "TRAIN-only concrete code gates; no OS-level read audit",
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
                    "parent_torch_threads": torch.get_num_threads(),
                    "parent_torch_interop_threads": torch.get_num_interop_threads(),
                    "worker_torch_thread_policy": {"intra_op": 1, "inter_op": 1},
                    "worker_runtime_observations": "per_arm_worker-runtime.json_and_arm_summary_from_validated_ready_acknowledgments",
                    "environment_variables": {name: os.environ.get(name) for name in (
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")},
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU"},
                "actual_attempted_updates": sum(row["attempted_updates"] + row["warmup_updates"] for row in results),
                "actual_applied_updates": sum(row["applied_updates"] + row["warmup_updates"] for row in results),
                "actual_measured_draws": measured_draws * len(results)}
        result["replay_wall_seconds"] = time.perf_counter() - replay_started
        result["replay_wall_scope"] = "direct_elapsed_including_all_arms_shared_model_setup_and_runtime_restoration_excludes_argument_checks_initial_protocol_writes_final_summary_write_and_process_imports"
        existing._write(root / "summary.json", result)
        return result
    except BaseException as exc:
        existing._write(root / "failure.json", {"error_type": type(exc).__name__, "error": str(exc), "completed": False})
        raise


def run_fixture_resume_gate(checkpoint_root: Path, *, worker_counts=(0, 1, 2, 4),
                            measured_draws=6, resume_after_updates=1, device="cpu"):
    """Tiny full-model fixture gate, separate from every throughput window.

    Serializes only this function's own synthetic training checkpoint. Every
    resume discards speculative jobs, reconstructs the stream, then restores
    complete parent model/Adam/scheduler/scaler/RNG. Forward hashes are costly
    correctness diagnostics and are never installed by run_prefetch_replay.
    """
    if device not in ("cpu", "cuda") or not 0 < resume_after_updates < measured_draws // 2:
        raise ValueError("prefetch_replay.invalid_fixture_resume_gate")
    jobs, descriptor = build_prefetch_schedule(mode="fixture", measured_draws=measured_draws,
                                              warmup_draws=0, reserved_bytes=2_000_000)
    config = PrefetchConfig(close_timeout_seconds=0.5)
    configs = worker_configurations(config, worker_counts)
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=False)
    source = fingerprint({"fixture_resume_gate": descriptor["fingerprint"]})
    results = []
    with existing._preserve_runtime(device == "cuda"):
        for count, worker_config in configs.items():
            branches = []
            for interrupt in (False, True):
                components = _components(device)
                model, optimizer, scheduler, scaler, _, _ = components
                stream = BatchStream(jobs, FixturePreparer(), source_fingerprint=source,
                                     config=worker_config, synthetic=True)
                traces, logits, pending = [], [], None
                worker_runtime_sessions = [stream.worker_runtime_evidence()]
                def capture(_model, _inputs, output):
                    logits.append(existing.recursive_state_fingerprint(output.logits))
                hook = model.register_forward_hook(capture)
                try:
                    for number in range(len(jobs)):
                        if interrupt and number == resume_after_updates:
                            pending = stream.diagnostics()["pending_batches"]
                            checkpoint = root / f"worker-{count}.pt"
                            torch.save({"training": existing._snapshot(model, optimizer, scheduler, scaler),
                                        "cursor": stream.state_dict()}, checkpoint)
                            stream.close()
                            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                            stream = BatchStream(jobs, FixturePreparer(), source_fingerprint=source,
                                config=worker_config, synthetic=True, checkpoint=state["cursor"])
                            worker_runtime_sessions.append(stream.worker_runtime_evidence())
                            existing._restore(state["training"], model, optimizer, scheduler, scaler)
                            del state
                        values, _, job = _step(stream, components, device, lambda _: existing.nullcontext())
                        if job != jobs[number]:
                            raise RuntimeError("prefetch_replay.resume_job_order_mismatch")
                        traces.append(values)
                    final = existing._snapshot(model, optimizer, scheduler, scaler)
                    branches.append({"state": existing.recursive_state_fingerprint(final),
                        "rng": existing.recursive_state_fingerprint(final["rng"]),
                        "trace": existing.recursive_state_fingerprint(traces),
                        "logits": existing.recursive_state_fingerprint(logits),
                        "worker_runtime_sessions": worker_runtime_sessions,
                        "committed": stream.state_dict()["committed_batch_cursor"], "pending_at_resume": pending})
                finally:
                    hook.remove()
                    stream.close()
                del final, components, model, optimizer, scheduler, scaler
                gc.collect()
            keys = ("state", "rng", "trace", "logits", "committed")
            results.append({"worker_count": count, "uninterrupted": branches[0], "resumed": branches[1],
                            "resume_exact": all(branches[0][key] == branches[1][key] for key in keys)})
    baseline = next(row for row in results if row["worker_count"] == 0)["uninterrupted"]
    passed = all(row["resume_exact"] and all(row["uninterrupted"][key] == baseline[key]
                 for key in ("state", "rng", "trace", "logits", "committed")) for row in results)
    return {"scope": "synthetic_full_model_checkpoint_gate_no_throughput_claim", "passed": passed,
            "device": device, "worker_counts": list(configs), "results": results}
