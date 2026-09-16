"""Opt-in ordered CPU preparation with explicit applied-update commits.

No sampler, CUDA model, optimizer or persistent cache writer lives here. An
explicit policy may enable a bounded process-local prepared graph cache.
Jobs are a fixed B2 schedule, not a mutable sampler iterator.
One reservation per submitted/uncommitted job bounds the sum of serialized
result payloads (in flight, ready, or delivered). It is NOT a bound on RSS:
worker preparation, decoded caller batches and simultaneous IPC copies are
additional memory and require separate measurement before production use.

Callables must be picklable, CPU-only and deterministic from their job/source;
global Python/NumPy/Torch RNG consumption is rejected. Explicit private RNGs
derived from job identity are permitted. Parent seed/thread settings are never
changed during normal execution, including workers=0. No DataLoader is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import queue
import random
import threading
import time
from typing import Any, Callable


SCHEMA = "AnalysisGNNOrderedPrefetch@1.0.0"
WORKER_RUNTIME_SCHEMA = "AnalysisGNNPrefetchWorkerRuntime@1.0.0"
CACHE_POLICY_SCHEMA = "AnalysisGNNProcessLocalCompactCache@1.0.0"
CACHE_OBSERVATION_SCHEMA = "AnalysisGNNProcessLocalCacheObservation@1.0.0"
NATIVE_THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_IMPORTED_IMPLEMENTATION_FINGERPRINT = sha256(Path(__file__).read_bytes()).hexdigest()


class PrefetchError(RuntimeError):
    pass


def _fingerprint(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _implementation_fingerprint() -> str:
    current = sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _IMPORTED_IMPLEMENTATION_FINGERPRINT:
        raise PrefetchError("prefetch.implementation_changed_after_import")
    return current


def validate_cache_policy(policy: Mapping[str, object] | None) -> dict[str, object] | None:
    """Validate an explicit GLOBAL retained-entry budget, not a RAM guarantee.

    The caller counts active preparing processes (one for a direct arm). Each
    process receives floor(global/count); parent validation must not populate
    another cache while a worker arm runs. Transient graphs/sidecars, IPC and
    models are outside this accounting. No default capacity is guessed here.
    """
    if policy is None:
        return None
    if not isinstance(policy, Mapping) or set(policy) != {
        "schema", "global_max_bytes", "preparing_processes"
    } or policy.get("schema") != CACHE_POLICY_SCHEMA:
        raise ValueError("prefetch.invalid_cache_policy")
    total, processes = policy["global_max_bytes"], policy["preparing_processes"]
    if type(total) is not int or type(processes) is not int or processes not in (1, 2, 4) or total < processes:
        raise ValueError("prefetch.invalid_cache_budget")
    return {"schema": CACHE_POLICY_SCHEMA, "global_max_bytes": total,
            "preparing_processes": processes}


def cache_instance_fingerprint(policy: Mapping[str, object] | None, *, max_entries: int) -> str | None:
    policy = validate_cache_policy(policy)
    if policy is None:
        return None
    if type(max_entries) is not int or max_entries <= 0:
        raise ValueError("prefetch.invalid_cache_entry_limit")
    return _fingerprint({"policy": policy,
        "max_bytes": policy["global_max_bytes"] // policy["preparing_processes"], "max_entries": max_entries})


def cache_observation(cache, policy: Mapping[str, object] | None, *, max_entries: int) -> dict[str, object]:
    """Shared numeric observation for concrete production and explicit fixtures."""
    policy = validate_cache_policy(policy)
    maximum = 0 if policy is None else policy["global_max_bytes"] // policy["preparing_processes"]
    stats = {"entries": 0, "accounted_bytes": 0, "max_bytes": maximum,
             "max_entries": 0 if policy is None else max_entries,
             "hits": 0, "misses": 0, "evictions": 0, "oversized_bypasses": 0}
    if cache is not None:
        stats = cache.diagnostics()
        if policy is None or stats["max_bytes"] != maximum or stats["max_entries"] != max_entries:
            raise PrefetchError("prefetch.cache_instance_limit_mismatch")
    return {"schema": CACHE_OBSERVATION_SCHEMA, "policy": policy,
            "policy_fingerprint": cache_instance_fingerprint(policy, max_entries=max_entries),
            "pid": os.getpid(), "initialized": cache is not None, **stats}


def _validate_cache_observation(payload, prepare, *, pid):
    expected = prepare.cache_diagnostics()
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise PrefetchError("prefetch.invalid_cache_observation")
    try:
        validate_cache_policy(payload["policy"])
    except ValueError as exc:
        raise PrefetchError("prefetch.invalid_cache_observation_policy") from exc
    for key in ("schema", "policy", "policy_fingerprint", "max_bytes", "max_entries"):
        if payload[key] != expected[key]:
            raise PrefetchError("prefetch.cache_observation_binding_mismatch:" + key)
    if type(payload["pid"]) is not int or payload["pid"] != pid or payload["initialized"] is not True:
        raise PrefetchError("prefetch.cache_observation_process_mismatch")
    for key in ("entries", "accounted_bytes", "max_bytes", "max_entries", "hits", "misses", "evictions", "oversized_bypasses"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise PrefetchError("prefetch.invalid_cache_observation_counter:" + key)
    if payload["entries"] > payload["max_entries"] or payload["accounted_bytes"] > payload["max_bytes"]:
        raise PrefetchError("prefetch.cache_observation_exceeds_limit")


@dataclass(frozen=True, slots=True)
class PrefetchJob:
    """Position is a BATCH ordinal; draw_ids are absolute canonical draw IDs."""

    position: int
    draw_ids: tuple[int, int]
    record_ids: tuple[str, str]
    schedule_shifts: tuple[int, int]
    applied_shifts: tuple[int, int]
    reserved_bytes: int

    def __post_init__(self):
        if type(self.position) is not int or self.position < 0:
            raise ValueError("prefetch.invalid_batch_position")
        for name in ("draw_ids", "record_ids", "schedule_shifts", "applied_shifts"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != 2:
                raise ValueError("prefetch.fixed_b2_tuple_required:" + name)
        if any(type(value) is not int or value < 0 for value in self.draw_ids) or (
            self.draw_ids[0] % 2 or self.draw_ids[1] != self.draw_ids[0] + 1
        ):
            raise ValueError("prefetch.invalid_draw_ids")
        if any(type(value) is not str or not value or len(value) > 1024 for value in self.record_ids):
            raise ValueError("prefetch.invalid_record_ids")
        if any(type(value) is not int or value not in range(12)
               for value in (*self.schedule_shifts, *self.applied_shifts)):
            raise ValueError("prefetch.invalid_shift")
        if type(self.reserved_bytes) is not int or self.reserved_bytes <= 0:
            raise ValueError("prefetch.explicit_positive_reservation_required")


@dataclass(frozen=True, slots=True)
class PrefetchConfig:
    workers: int = 0
    max_pending_batches: int = 4
    buffer_byte_cap: int = 64 * 1024**2
    seed: int = 17291
    job_timeout_seconds: float = 120.0
    startup_timeout_seconds: float = 60.0
    close_timeout_seconds: float = 3.0

    def __post_init__(self):
        if type(self.workers) is not int or self.workers not in (0, 1, 2, 4):
            raise ValueError("prefetch.workers_must_be_0_1_2_4")
        for name in ("max_pending_batches", "buffer_byte_cap"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("prefetch.invalid_" + name)
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("prefetch.invalid_seed")
        for name in ("job_timeout_seconds", "startup_timeout_seconds", "close_timeout_seconds"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("prefetch.invalid_" + name)


@dataclass(frozen=True)
class PreparedCPUJob:
    job: PrefetchJob
    raw_graph_batch: Any
    sidecars: tuple[dict[str, object], ...]


def reservation_job_fingerprint(job: PrefetchJob) -> str:
    identity = asdict(job)
    identity.pop("reserved_bytes")
    return _fingerprint(identity)


def make_reservation_manifest(*, source_fingerprint: str, evidence_id: str,
                              measured_serialized_bytes: Mapping[str, int]) -> dict[str, object]:
    """Seal externally measured job sizes; does not authenticate measurements.

    A registered warm size gate must supply these observations. This helper
    validates their shape and binds exact jobs/source, not RSS or OOM safety.
    Bootstrap with a bounded provisional reservation, measure the exact warm
    payload, then retain that reservation or remeasure after changing it: the
    returned job metadata also contains the reservation integer. Job identity
    excludes that policy field; the complete schedule/checkpoint still binds
    it, and serialization always checks the actual final reservation.
    """
    if not isinstance(source_fingerprint, str) or not source_fingerprint or not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("prefetch.reservation_evidence_required")
    sizes = dict(measured_serialized_bytes)
    if any(type(key) is not str or len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
           or type(value) is not int or value <= 0 for key, value in sizes.items()):
        raise ValueError("prefetch.invalid_measured_job_size")
    body = {"schema": "AnalysisGNNPrefetchReservations@1.0.0", "source_fingerprint": source_fingerprint,
            "evidence_id": evidence_id, "measured_serialized_bytes": dict(sorted(sizes.items()))}
    return {**body, "fingerprint": _fingerprint(body)}


class WarmTrainBatchPreparer:
    """Frozen TRAIN preparation, optionally with compact private CPU caching.

    An externally sealed manifest must cover EVERY scheduled record. Parent
    validation precedes process creation; each worker revalidates that same
    manifest before accepting jobs. ``source_fingerprint`` binds paths and the
    complete manifest. The default is C0 identity only; nonidentity use needs
    an explicit separate caller protocol and is not validated as C2 here.
    Cache capacity preserves numeric content, but pickle memoization may change
    serialized length when metadata aliases differ on hits and misses. Actual
    serialization always enforces the job reservation; measured lengths are
    observations at the size gate's declared policy, not universal bounds.
    The reservation base identity binds
    the chosen preparation algorithm, while the final source identity also
    binds the exact per-instance capacity policy. A capacity-independent size
    gate therefore need not pretend different worker allocations are equal.
    """

    def __init__(self, paths, binding_manifest: Mapping[str, object], *, identity_only: bool = True,
                 reservation_manifest: Mapping[str, object] | None = None,
                 cache_policy: Mapping[str, object] | None = None,
                 runtime_payload_cap_bytes: int | None = None):
        from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
        if not isinstance(paths, ProductionArtifactPaths) or type(identity_only) is not bool:
            raise ValueError("prefetch.invalid_production_configuration")
        self.paths = paths
        self._manifest_json = json.dumps(binding_manifest, sort_keys=True, allow_nan=False)
        self.identity_only = identity_only
        if runtime_payload_cap_bytes is not None and (
                type(runtime_payload_cap_bytes) is not int or runtime_payload_cap_bytes <= 0
                or reservation_manifest is not None):
            raise ValueError("prefetch.invalid_runtime_payload_cap")
        self.runtime_payload_cap_bytes = runtime_payload_cap_bytes
        self._loader = None
        self._cache = None
        self._owner_pid = None
        policy = validate_cache_policy(cache_policy)
        self._cache_policy_json = None if policy is None else json.dumps(policy, sort_keys=True)
        manifest = json.loads(self._manifest_json)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("record_bindings"), dict):
            raise ValueError("prefetch.sealed_record_bindings_required")
        base = {
            "schema": SCHEMA, "manifest": manifest, "identity_only": identity_only,
            "implementation_fingerprint": _implementation_fingerprint(),
            "paths": {field.name: str(getattr(paths, field.name)) for field in fields(paths)},
        }
        if policy is not None:
            if not manifest["record_bindings"]:
                raise ValueError("prefetch.cache_requires_bound_records")
            base["cache_algorithm"] = {"schema": CACHE_POLICY_SCHEMA,
                "representation": "frozen_cpu_graph_and_original_verified_sidecar_json_bytes",
                "collation": "PreparedGraphCache.prepare_batch_then_frozen_transposition"}
        if runtime_payload_cap_bytes is not None:
            # Explicit admission policy for long runs: NOT measured sizing evidence.
            # The existing bounded serializer rejects an oversized job before IPC.
            base["runtime_payload_policy"] = {"schema": "RuntimeBoundedPayload@1.0.0",
                "max_bytes": runtime_payload_cap_bytes, "premeasured": False}
        self.base_source_fingerprint = _fingerprint(base)
        self.cache_instance_fingerprint = cache_instance_fingerprint(
            policy, max_entries=len(manifest["record_bindings"]))
        self._reservation_json = None
        if reservation_manifest is not None:
            reservation = json.loads(json.dumps(reservation_manifest, allow_nan=False))
            try:
                expected = make_reservation_manifest(source_fingerprint=self.base_source_fingerprint,
                    evidence_id=reservation["evidence_id"], measured_serialized_bytes=reservation["measured_serialized_bytes"])
            except (TypeError, KeyError) as exc:
                raise ValueError("prefetch.invalid_reservation_manifest") from exc
            if reservation != expected:
                raise ValueError("prefetch.reservation_manifest_binding_mismatch")
            self._reservation_json = json.dumps(reservation, sort_keys=True, allow_nan=False)
        source = {"base_source_fingerprint": self.base_source_fingerprint,
                  "reservation_manifest": None if self._reservation_json is None else json.loads(self._reservation_json)}
        if policy is not None:
            source["cache_instance_fingerprint"] = self.cache_instance_fingerprint
        self.source_fingerprint = _fingerprint(source)

    @property
    def cache_policy(self):
        return None if self._cache_policy_json is None else json.loads(self._cache_policy_json)

    def __getstate__(self):
        # Never pickle graphs, loader locks, cache locks or observations. Spawn
        # starts a cold private cache, irrespective of the parent's history.
        state = self.paths, self._manifest_json, self.identity_only, self._reservation_json
        if self.runtime_payload_cap_bytes is not None:
            return (*state, self._cache_policy_json, self.runtime_payload_cap_bytes)
        return state if self._cache_policy_json is None else (*state, self._cache_policy_json)

    def __setstate__(self, state):
        self.__init__(state[0], json.loads(state[1]), identity_only=state[2],
                      reservation_manifest=None if state[3] is None else json.loads(state[3]),
                      cache_policy=None if len(state) == 4 or state[4] is None else json.loads(state[4]),
                      runtime_payload_cap_bytes=state[5] if len(state) == 6 else None)

    def _new_loader(self):
        from music_critic.experiments.analysisgnn.performance_data import FrozenProductionRecordLoader
        loader = FrozenProductionRecordLoader(paths=self.paths, split="train")
        loader.validate_all_bindings(expected=json.loads(self._manifest_json))
        return loader

    def validate_jobs(self, jobs: Sequence[PrefetchJob]) -> None:
        manifest = json.loads(self._manifest_json)
        for job in jobs:
            if self.identity_only and any(job.applied_shifts):
                raise ValueError("prefetch.production_identity_only")
            if any(record not in manifest["record_bindings"] for record in job.record_ids):
                raise ValueError("prefetch.unbound_scheduled_record")
        if self.runtime_payload_cap_bytes is not None:
            if any(job.reserved_bytes != self.runtime_payload_cap_bytes for job in jobs):
                raise ValueError("prefetch.runtime_payload_reservation_mismatch")
            self._new_loader()
            return
        if self._reservation_json is None:
            raise ValueError("prefetch.production_reservation_manifest_required")
        sizes = json.loads(self._reservation_json)["measured_serialized_bytes"]
        for job in jobs:
            measured = sizes.get(reservation_job_fingerprint(job))
            if measured is None or measured > job.reserved_bytes:
                raise ValueError("prefetch.unmeasured_or_insufficient_job_reservation")
        # A fresh empty loader manifest would not bind record bytes. Require
        # explicit coverage above, then rehash all expected record bindings.
        self._new_loader()

    def validate_direct_jobs(self, jobs: Sequence[PrefetchJob]) -> None:
        """Authenticate direct TRAIN inputs without inventing IPC measurements.

        Only an explicit direct caller may use this entry point. The pool
        continues to call validate_jobs and always requires measured budgets.
        """
        manifest = json.loads(self._manifest_json)
        for job in jobs:
            if self.identity_only and any(job.applied_shifts):
                raise ValueError("prefetch.production_identity_only")
            if any(record not in manifest["record_bindings"] for record in job.record_ids):
                raise ValueError("prefetch.unbound_scheduled_record")
        self._new_loader()

    def start(self) -> None:
        self.reset_cache()
        self._loader = self._new_loader()
        self._owner_pid = os.getpid()

    def reset_cache(self) -> dict[str, object]:
        """Drop this process's retained entries/loader and reset counters cold."""
        previous = self.cache_diagnostics()
        self._cache = self._loader = None
        self._owner_pid = None
        return previous

    def cache_diagnostics(self) -> dict[str, object]:
        """Local cumulative counters; no RSS, transient or cache-hit guarantee."""
        return cache_observation(self._cache, self.cache_policy,
                                 max_entries=len(json.loads(self._manifest_json)["record_bindings"]))

    def __call__(self, job: PrefetchJob) -> PreparedCPUJob:
        from unittest.mock import patch
        from music_critic.experiments.analysisgnn import corrected_training as reference
        if self.identity_only and any(job.applied_shifts):
            raise ValueError("prefetch.production_identity_only")
        manifest = json.loads(self._manifest_json)
        if any(record not in manifest["record_bindings"] for record in job.record_ids):
            raise ValueError("prefetch.unbound_scheduled_record")
        if self._loader is None or self._owner_pid != os.getpid():
            self.start()

        if self._cache_policy_json is not None:
            if self._cache is None:
                from music_critic.experiments.analysisgnn.performance_data import PreparedGraphCache
                stats = self.cache_diagnostics()
                self._cache = PreparedGraphCache(self._loader, max_bytes=stats["max_bytes"],
                                                 max_entries=stats["max_entries"])
            prepared = self._cache.prepare_batch(job.record_ids, split="train")
            raw = reference.transpose_raw_graph_batch(prepared.raw_graph_batch, job.applied_shifts)
            return PreparedCPUJob(job, raw, prepared.sidecars)

        def forbidden(*args, **kwargs):
            raise PrefetchError("prefetch.cold_conversion_forbidden")

        loaded = []
        for record_id in job.record_ids:
            self._loader.check_binding(record_id, split="train")
            with patch.object(reference, "_resolve_selected_corpus_record", forbidden):
                loaded.append(reference.load_production_record(record_id, split="train", paths=self.paths))
            self._loader.check_binding(record_id, split="train")
        raw = reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
        raw = reference.transpose_raw_graph_batch(raw, job.applied_shifts)
        return PreparedCPUJob(job, raw, tuple(row[1] for row in loaded))


def _assert_cpu(value: object, seen: set[int] | None = None) -> None:
    import numpy as np
    import torch
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise PrefetchError("prefetch.cpu_tensor_required")
    elif value is None or type(value) in (bool, int, float, str, bytes):
        return
    elif isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            raise PrefetchError("prefetch.numpy_object_payload_forbidden")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_cpu(key, seen)
            _assert_cpu(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_cpu(item, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_cpu(getattr(value, field.name), seen)
    else:
        from torch_geometric.data import HeteroData
        if not isinstance(value, HeteroData):
            raise PrefetchError("prefetch.unsupported_cpu_payload:" + type(value).__name__)
        for store in value.stores:
            _assert_cpu(dict(store.items()), seen)
        for name in ("_slice_dict", "_inc_dict"):
            if hasattr(value, name):
                _assert_cpu(getattr(value, name), seen)


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise PrefetchError("prefetch.result_exceeds_reservation")
        return super().write(data)


def _prepare_bytes(prepare, job):
    import numpy as np
    import torch
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        payload = prepare(job)
        _assert_cpu(payload)
        stream = _LimitedBuffer(job.reserved_bytes)
        pickle.dump(payload, stream, protocol=5)
        return stream.getvalue()
    finally:
        numpy_state = np.random.get_state()
        changed = (before[0] != random.getstate() or before[1][0] != numpy_state[0]
                   or not np.array_equal(before[1][1], numpy_state[1])
                   or before[1][2:] != numpy_state[2:] or not torch.equal(before[2], torch.get_rng_state()))
        if changed:
            # Invalid user callable: fail closed and repair the caller's RNG.
            # Valid preprocessing does not set/reseed any parent RNG at all.
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])
            raise PrefetchError("prefetch.callable_consumed_global_rng")


def _runtime_versions():
    import numpy as np
    import torch
    return {"python": platform.python_version(), "numpy": str(np.__version__),
            "torch": str(torch.__version__), "torch_cuda_runtime": torch.version.cuda}


def _observe_worker_runtime(worker_id, implementation_fingerprint, source_fingerprint, inherited_environment):
    """Read back startup state, not constants or a native thread-pool inventory."""
    import torch
    return {"schema": WORKER_RUNTIME_SCHEMA, "worker_id": worker_id, "pid": os.getpid(),
            "implementation_fingerprint": implementation_fingerprint, "source_fingerprint": source_fingerprint,
            "runtime_versions": _runtime_versions(), "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "inherited_native_thread_environment": inherited_environment,
            "native_thread_environment": {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT},
            "cuda_initialized": torch.cuda.is_initialized()}


def _validate_worker_runtime(payload, *, worker_id, pid, implementation_fingerprint,
                             source_fingerprint, runtime_versions, inherited_environment):
    keys = {"schema", "worker_id", "pid", "implementation_fingerprint", "source_fingerprint",
            "runtime_versions", "torch_num_threads", "torch_num_interop_threads",
            "native_thread_environment", "inherited_native_thread_environment", "cuda_initialized"}
    if type(payload) is not dict or set(payload) != keys:
        raise PrefetchError("prefetch.invalid_worker_runtime_schema")
    if (payload["schema"] != WORKER_RUNTIME_SCHEMA or type(payload["worker_id"]) is not int
            or payload["worker_id"] != worker_id or type(payload["pid"]) is not int
            or payload["pid"] != pid or pid <= 0
            or payload["implementation_fingerprint"] != implementation_fingerprint
            or payload["source_fingerprint"] != source_fingerprint):
        raise PrefetchError("prefetch.worker_runtime_binding_mismatch")
    versions, environment = payload["runtime_versions"], payload["native_thread_environment"]
    inherited = payload["inherited_native_thread_environment"]
    expected_environment = {**inherited_environment, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if (type(versions) is not dict or versions != runtime_versions
            or type(inherited) is not dict or inherited != inherited_environment
            or type(environment) is not dict
            or environment != expected_environment
            or any(type(payload[key]) is not int or payload[key] != 1
                   for key in ("torch_num_threads", "torch_num_interop_threads"))
            or payload["cuda_initialized"] is not False):
        raise PrefetchError("prefetch.worker_runtime_policy_mismatch")


def _worker_main(connection, prepare, worker_id, seed, implementation_fingerprint, source_fingerprint):
    position = None
    try:
        if _implementation_fingerprint() != implementation_fingerprint:
            raise PrefetchError("prefetch.worker_implementation_mismatch")
        inherited_environment = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT}
        # Preserve the historical setters. Production must inherit all four
        # native limits before spawn imports; these assignments cannot prove it.
        os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        import numpy as np
        import torch
        if torch.cuda.is_initialized():
            raise PrefetchError("prefetch.worker_cuda_initialized")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        worker_seed = int.from_bytes(sha256(f"{SCHEMA}:{seed}:{worker_id}".encode()).digest()[:8], "big")
        random.seed(worker_seed)
        np.random.seed(worker_seed % 2**32)
        torch.random.default_generator.manual_seed(worker_seed)
        if type(prepare) is WarmTrainBatchPreparer:
            prepare.start()
        observation = _observe_worker_runtime(worker_id, implementation_fingerprint, source_fingerprint,
                                              inherited_environment)
        connection.send(("ready", worker_id, observation))
        del observation
        while True:
            job = connection.recv()
            if job is None:
                return
            position = job.position
            payload = _prepare_bytes(prepare, job)
            if torch.cuda.is_initialized():
                raise PrefetchError("prefetch.worker_cuda_initialized")
            if getattr(prepare, "cache_policy", None) is None:
                connection.send(("result", position, payload))
            else:
                # Diagnostics are transport metadata, never PreparedCPUJob
                # fields or part of the measured serialized-job reservation.
                connection.send(("cached_result", position, (payload, prepare.cache_diagnostics())))
            del payload, job  # no previous serialized batch retained while idle
    except BaseException as exc:
        try:
            connection.send(("error", position, f"{type(exc).__name__}: {exc}"[:4000]))
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


@dataclass
class _Slot:
    process: Any
    connection: Any
    incoming: Any
    receiver: Any
    stop_receiver: Any
    worker_id: int
    ready: bool = False
    position: int | None = None
    started: float = 0.0


def _receive_frames(connection, incoming, stop, limit):
    """Blocking frame reads live outside the parent control/watchdog thread.

    poll() alone is insufficient: a partial large frame can otherwise keep
    recv_bytes() blocked past a worker deadline. Exactly one submitted job per
    slot and a size-one queue retain the same global job reservation budget.
    """
    while not stop.is_set():
        try:
            item = ("frame", connection.recv_bytes(maxlength=limit))
        except (OSError, EOFError) as exc:
            item = ("channel_error", str(exc))
        while not stop.is_set():
            try:
                incoming.put(item, timeout=0.05)
                break
            except queue.Full:
                pass
        failed = item[0] == "channel_error"
        del item  # the bounded queue now owns the frame; retain no idle copy
        if failed:
            return


class OrderedPrefetch:
    """Peek returns a fresh CPU value; commit requires an applied B2 update.

    Speculation never advances a sampler. The caller advances its real sampler
    only alongside ``commit(position, applied=True)`` and stores both states
    in its existing checkpoint transaction. A failed update can peek again;
    this class never rewinds model/dropout RNG. No iterator auto-commits work.
    """

    def __init__(self, jobs: Sequence[PrefetchJob], prepare: Callable, *,
                 source_fingerprint: str, config: PrefetchConfig = PrefetchConfig(),
                 checkpoint: Mapping[str, object] | None = None, synthetic: bool = False):
        self.jobs, self.config, self.prepare = tuple(jobs), config, prepare
        if not isinstance(config, PrefetchConfig) or not callable(prepare):
            raise ValueError("prefetch.invalid_configuration")
        if type(synthetic) is not bool:
            raise ValueError("prefetch.synthetic_boolean_required")
        if type(prepare) is WarmTrainBatchPreparer:
            if synthetic:
                raise ValueError("prefetch.production_synthetic_mode_mismatch")
        elif not synthetic:
            raise ValueError("prefetch.synthetic_opt_in_required")
        if type(source_fingerprint) is not str or not source_fingerprint:
            raise ValueError("prefetch.source_fingerprint_required")
        for position, job in enumerate(self.jobs):
            if type(job) is not PrefetchJob or job.position != position:
                raise ValueError("prefetch.contiguous_batch_schedule_required")
            if position and job.draw_ids[0] != self.jobs[position - 1].draw_ids[-1] + 1:
                raise ValueError("prefetch.contiguous_draw_schedule_required")
            if job.reserved_bytes > config.buffer_byte_cap:
                raise ValueError("prefetch.reservation_exceeds_global_cap")
        if type(prepare) is WarmTrainBatchPreparer and source_fingerprint != prepare.source_fingerprint:
            raise ValueError("prefetch.production_source_fingerprint_mismatch")
        self._cache_policy = validate_cache_policy(getattr(prepare, "cache_policy", None))
        if self._cache_policy is not None and self._cache_policy["preparing_processes"] != max(config.workers, 1):
            raise ValueError("prefetch.cache_preparing_process_count_mismatch")
        if self._cache_policy is not None and config.workers and prepare.cache_diagnostics()["initialized"]:
            raise ValueError("prefetch.worker_arm_requires_cold_parent_cache")
        self._bindings = {"schema": SCHEMA, "source_fingerprint": source_fingerprint,
                          "implementation_fingerprint": _implementation_fingerprint(),
                          "source_mode": "synthetic" if synthetic else "frozen_warm_train",
                          "schedule_fingerprint": _fingerprint([asdict(job) for job in self.jobs]),
                          "config_fingerprint": _fingerprint(asdict(config)), "batch_size": 2,
                          "start_draw_offset": self.jobs[0].draw_ids[0] if self.jobs else 0}
        if self._cache_policy is not None:
            self._bindings["cache_instance_fingerprint"] = prepare.cache_instance_fingerprint
        self._cursor = 0
        if checkpoint is not None:
            body = json.loads(json.dumps(dict(checkpoint), allow_nan=False))
            seal = body.pop("fingerprint", None)
            if seal != _fingerprint(body) or any(body.get(key) != value for key, value in self._bindings.items()):
                raise ValueError("prefetch.checkpoint_binding_mismatch")
            cursor = body.get("committed_batch_cursor")
            if type(cursor) is not int or not 0 <= cursor <= len(self.jobs) or body.get(
                "committed_draw_cursor") != self._bindings["start_draw_offset"] + 2 * cursor:
                raise ValueError("prefetch.checkpoint_cursor_invalid")
            self._cursor = cursor
        self._next = self._cursor
        self._reserved = 0
        self._ready: dict[int, bytes] = {}
        self._pending: set[int] = set()
        self._slots: list[_Slot] = []
        self._worker_runtime: dict[int, dict[str, object]] = {}
        self._cache_observations: dict[int, dict[str, object]] = {}
        self._required_workers = config.workers if self._cursor < len(self.jobs) else 0
        self._expected_runtime = None
        self._inherited_environment = None
        self._delivered = None
        self._closed = False
        self._error = None
        self._high_jobs = self._high_bytes = 0
        try:
            if self._required_workers:
                self._inherited_environment = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT}
                if not synthetic and self._inherited_environment != {name: "1" for name in NATIVE_THREAD_ENVIRONMENT}:
                    raise PrefetchError("prefetch.production_native_limits_required_before_spawn")
            if type(prepare) is WarmTrainBatchPreparer:
                prepare.validate_jobs(self.jobs)
            if config.workers and self._cursor < len(self.jobs):
                self._expected_runtime = _runtime_versions()
                context = mp.get_context("spawn")
                for worker_id in range(config.workers):
                    parent, child = context.Pipe(duplex=True)
                    process = context.Process(target=_worker_main,
                        args=(child, prepare, worker_id, config.seed,
                              self._bindings["implementation_fingerprint"], source_fingerprint),
                        daemon=True)
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    incoming, stop = queue.Queue(maxsize=1), threading.Event()
                    receiver = threading.Thread(target=_receive_frames,
                        args=(parent, incoming, stop, config.buffer_byte_cap + 65536), daemon=True,
                        name=f"analysisgnn-prefetch-{process.pid}")
                    self._slots.append(_Slot(process, parent, incoming, receiver, stop, worker_id))
                    receiver.start()
                deadline = time.monotonic() + config.startup_timeout_seconds
                while not all(slot.ready for slot in self._slots):
                    self._receive(deadline, startup=True)
                if set(self._worker_runtime) != set(range(config.workers)):
                    self._fail("prefetch.incomplete_worker_runtime_observations")
            self._fill()
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._error is not None:
            raise PrefetchError(self._error)
        if self._closed:
            raise PrefetchError("prefetch.closed")

    def _fail(self, reason):
        self._error = reason
        self.close()
        raise PrefetchError(reason)

    def _fill(self):
        self._ensure_open()
        while self._next < len(self.jobs) and len(self._pending) < self.config.max_pending_batches:
            job = self.jobs[self._next]
            if self._reserved + job.reserved_bytes > self.config.buffer_byte_cap:
                break
            idle = next((slot for slot in self._slots if slot.ready and slot.position is None), None)
            if self.config.workers and idle is None:
                break
            # Zero-worker execution prepares only the requested current batch,
            # never eagerly serializes a sequence in the parent at startup.
            if not self.config.workers:
                break
            self._reserve(job)
            idle.position, idle.started = job.position, time.monotonic()
            try:
                idle.connection.send(job)
            except (OSError, EOFError) as exc:
                self._fail("prefetch.dispatch_failed:" + str(exc))

    def _reserve(self, job):
        self._pending.add(job.position)
        self._reserved += job.reserved_bytes
        self._next += 1
        self._high_jobs = max(self._high_jobs, len(self._pending))
        self._high_bytes = max(self._high_bytes, self._reserved)

    def _receive(self, deadline, *, startup=False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail("prefetch.startup_timeout" if startup else "prefetch.worker_timeout")
        received = False
        for slot in self._slots:
            try:
                transport_kind, frame = slot.incoming.get_nowait()
            except queue.Empty:
                pass
            else:
                received = True
                if transport_kind == "channel_error":
                    self._fail("prefetch.worker_channel_failed:" + frame)
                try:
                    message = pickle.loads(frame)
                except Exception as exc:
                    self._fail("prefetch.invalid_worker_frame:" + str(exc))
                if not isinstance(message, tuple) or len(message) != 3:
                    self._fail("prefetch.invalid_worker_message")
                kind, position, payload = message
                if kind == "error":
                    self._fail(f"prefetch.worker_failed:{position}:{payload}")
                if kind == "ready" and not slot.ready and slot.position is None:
                    if type(position) is not int or position != slot.worker_id or position in self._worker_runtime:
                        self._fail("prefetch.worker_runtime_identity_mismatch")
                    try:
                        _validate_worker_runtime(payload, worker_id=slot.worker_id, pid=slot.process.pid,
                            implementation_fingerprint=self._bindings["implementation_fingerprint"],
                            source_fingerprint=self._bindings["source_fingerprint"],
                            runtime_versions=self._expected_runtime,
                            inherited_environment=self._inherited_environment)
                    except PrefetchError as exc:
                        self._fail(str(exc))
                    self._worker_runtime[position] = payload
                    slot.ready = True
                elif kind in ("result", "cached_result") and type(position) is int and slot.position == position and position in self._pending:
                    if (kind == "cached_result") != (self._cache_policy is not None):
                        self._fail("prefetch.cache_result_policy_mismatch")
                    if kind == "cached_result":
                        if type(payload) is not tuple or len(payload) != 2:
                            self._fail("prefetch.invalid_cached_result")
                        payload, observation = payload
                        try:
                            _validate_cache_observation(observation, self.prepare, pid=slot.process.pid)
                        except PrefetchError as exc:
                            self._fail(str(exc))
                        previous = self._cache_observations.get(slot.worker_id)
                        if previous is not None and any(observation[key] < previous["observation"][key]
                            for key in ("hits", "misses", "evictions", "oversized_bypasses")):
                            self._fail("prefetch.cache_counters_regressed")
                        self._cache_observations[slot.worker_id] = {
                            "worker_id": slot.worker_id, "last_completed_position": position,
                            "observation": observation}
                    if type(payload) is not bytes or len(payload) > self.jobs[position].reserved_bytes:
                        self._fail("prefetch.result_exceeds_reservation")
                    self._ready[position] = payload
                    slot.position = None
                else:
                    self._fail("prefetch.unexpected_worker_message")
            if slot.process.exitcode is not None:
                self._fail(f"prefetch.worker_exited:{slot.process.pid}:{slot.process.exitcode}")
            if slot.position is not None and time.monotonic() - slot.started > self.config.job_timeout_seconds:
                self._fail("prefetch.worker_timeout")
        if not received:
            time.sleep(min(remaining, 0.01))

    def peek(self):
        self._ensure_open()
        if self._cursor == len(self.jobs):
            raise StopIteration
        job = self.jobs[self._cursor]
        if not self.config.workers and self._cursor not in self._ready:
            self._reserve(job)
            try:
                self._ready[self._cursor] = _prepare_bytes(self.prepare, job)
            except BaseException as exc:
                self._fail(f"prefetch.synchronous_prepare_failed:{type(exc).__name__}:{exc}")
        if self.config.workers:
            self._fill()
            deadline = time.monotonic() + self.config.job_timeout_seconds
            while self._cursor not in self._ready:
                self._receive(deadline)
                self._fill()
        try:
            result = pickle.loads(self._ready[self._cursor])
            _assert_cpu(result)
        except BaseException as exc:
            self._fail("prefetch.result_decode_failed:" + str(exc))
        self._delivered = self._cursor
        return result

    def commit(self, position: int, *, applied: bool) -> bool:
        self._ensure_open()
        if type(position) is not int or position != self._cursor or self._delivered != position:
            raise PrefetchError("prefetch.out_of_order_or_unobserved_commit")
        if type(applied) is not bool:
            raise ValueError("prefetch.applied_boolean_required")
        if not applied:
            return False
        # Bookkeeping only: dispatch/errors cannot interrupt an applied commit.
        self._ready.pop(position)
        self._pending.remove(position)
        self._reserved -= self.jobs[position].reserved_bytes
        self._cursor += 1
        self._delivered = None
        return True

    def state_dict(self) -> dict[str, object]:
        body = {**self._bindings, "committed_batch_cursor": self._cursor,
                "committed_draw_cursor": self._bindings["start_draw_offset"] + 2 * self._cursor}
        return {**body, "fingerprint": _fingerprint(body)}

    def worker_runtime_evidence(self) -> dict[str, object]:
        """Detached startup observations; excludes parent state and model traces.

        Empty/exhausted schedules create no workers. Fresh resume pools collect
        new observations, never restore stale PID/runtime metadata from a cursor.
        Environment readback does not measure already-created native pools.
        """
        complete = (set(self._worker_runtime) == set(range(self._required_workers))
                    and len(self._slots) == self._required_workers)
        all_one = {name: "1" for name in NATIVE_THREAD_ENVIRONMENT}
        return {"schema": WORKER_RUNTIME_SCHEMA, "configured_workers": self.config.workers,
                "required_workers": self._required_workers,
                "started_workers": len(self._slots),
                "observations_complete": complete,
                "native_thread_limits_all_one": (None if not self._required_workers else complete and
                    all(row["inherited_native_thread_environment"] == row["native_thread_environment"] == all_one
                        for row in self._worker_runtime.values())),
                "spawn_parent_native_thread_environment": (None if self._inherited_environment is None
                                                           else dict(self._inherited_environment)),
                "observations": json.loads(json.dumps([self._worker_runtime[key] for key in sorted(self._worker_runtime)])),
                "scope": "worker_startup_getters_and_environment_readback_not_continuous_or_native_pool_telemetry"}

    def diagnostics(self, *, include_worker_runtime: bool = True) -> dict[str, object]:
        result = {"committed_batch_cursor": self._cursor, "submitted_batch_cursor": self._next,
                "pending_batches": len(self._pending), "ready_batches": len(self._ready),
                "inflight_batches": sum(slot.position is not None for slot in self._slots),
                "reserved_bytes": self._reserved, "serialized_ready_bytes": sum(map(len, self._ready.values())),
                "high_water_pending_batches": self._high_jobs, "high_water_reserved_bytes": self._high_bytes,
                "worker_pids": tuple(slot.process.pid for slot in self._slots), "closed": self._closed,
                "buffer_scope": "one_serialized_payload_reservation_per_uncommitted_job_not_RSS_or_IPC_copies"}
        if include_worker_runtime:
            result["worker_runtime"] = self.worker_runtime_evidence()
        if self._cache_policy is not None:
            result["prepared_cache"] = self.cache_diagnostics()
        return result

    def cache_diagnostics(self) -> dict[str, object]:
        """Latest received post-job snapshots; pending work may be newer.

        This is bounded by worker count, not an all-job journal. Worker exit
        releases worker-local caches. No simultaneous RSS/peak claim is made.
        """
        rows = ([{"worker_id": None, "last_completed_position": None,
                  "observation": self.prepare.cache_diagnostics()}]
                if self._cache_policy is not None and not self.config.workers else
                [self._cache_observations[key] for key in sorted(self._cache_observations)])
        return {"policy": self._cache_policy, "configured_workers": self.config.workers,
                "observations": json.loads(json.dumps(rows)),
                "scope": "latest_received_post_job_per_process_not_RSS_or_continuous_telemetry"}

    def close(self):
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + self.config.close_timeout_seconds
        for slot in self._slots:
            slot.stop_receiver.set()
            if slot.process.is_alive():
                try:
                    slot.connection.send(None)
                except (OSError, EOFError, BrokenPipeError):
                    pass
        # Reader threads drain an already pending frame so a worker sending a
        # result can reach its stop message. The control thread never reads a
        # possibly partial frame and keeps its common pool deadline.
        while any(slot.process.is_alive() for slot in self._slots) and time.monotonic() < deadline:
            for slot in self._slots:
                slot.process.join(timeout=0)
            time.sleep(0.01)
        for slot in self._slots:
            if slot.process.is_alive():
                slot.process.terminate()
        for slot in self._slots:
            slot.process.join(timeout=max(0.0, deadline - time.monotonic()) + 0.1)
            if slot.process.is_alive():
                slot.process.kill()
                slot.process.join(timeout=0.5)
            slot.connection.close()
            if slot.receiver.ident is not None:
                slot.receiver.join(timeout=0.1)
            while not slot.incoming.empty():
                try:
                    slot.incoming.get_nowait()
                except queue.Empty:
                    break
        self._ready.clear()
        self._pending.clear()
        self._reserved = 0

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()
