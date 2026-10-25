"""Isolated C4 training: admitted C3 architecture, POP supervision and full resume.

Only the dataset and its safely available targets change. Eighteen C3 heads
retain their state layout; unsupported targets never enter a loss. This module
cannot open TEST and every real-data CLI action requires a planned record.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import math
import multiprocessing
import os
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from music_critic.cuda_memory import initialize_cuda_memory_statistics
from music_critic.device import resolve_runtime_device

from music_critic.critic.c3_backbone import (
    C3ThreePassRelationGateModel, C3_SOURCE_ID, validate_c3_source_model,
)
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.full_orbit_training import (
    FullOrbitPair, FullOrbitSampler, build_full_orbit_optimizer_scheduler,
    build_full_orbit_table, full_orbit_table_contract,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.performance_training import (
    _coalesced_host_scalars, performance_gradient_diagnostics,
)
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from music_critic.experiments.analysisgnn.training_policy import _normalized_class_weights
from music_critic.experiments.analysisgnn.training_rng import (
    capture_safe_rng, restore_safe_rng, preserve_caller_rng,
)
from music_critic.experiments.analysisgnn.temporal_slices.cache import (
    accounted_bytes, collate_temporal_graphs, encode_temporal_graph,
    decode_temporal_graph, transpose_temporal_historical,
)
from music_critic.experiments.pop909_cl_frozen.projection import root_pc_from_label

SCHEMA = "C4Pop909Training@1.0.0"
MANIFEST_SCHEMA = "C4Pop909Preparation@1.0.0"
CHECKPOINT_SCHEMA = "C4Pop909TrainingCheckpoint@1.0.0"
HEADS = ("root", "bass", "quality", "inversion", "pitch_class_set")
PRIMARY_LOSS_HEADS = ("root", "bass", "quality", "inversion")
PAYLOAD_CAP = 954204160  # Historical B4 reservation, not a process RSS bound.


def require(condition, reason):
    if not condition:
        raise ValueError("c4_pop909." + reason)


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def emit(output, event, **values):
    row = dict(event=event, time=time.time(), **values)
    encoded = json.dumps(row, sort_keys=True, allow_nan=False)
    with (Path(output) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
    print(encoded, flush=True)


def journal_update(output, value):
    """Commit one inspectable applied-draw row before any checkpoint publication."""
    with (Path(output) / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def file_identity(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), size_bytes=Path(path).stat().st_size)


def implementation_binding():
    """Bind all V2 Python sources so worker/resume cannot silently drift."""
    source_root = Path(__file__).resolve().parents[2]
    files = {str(path.relative_to(source_root)): file_identity(path)["sha256"]
             for path in sorted(source_root.rglob("*.py"))}
    return dict(files=files, fingerprint=fingerprint(files), c3_source_id=C3_SOURCE_ID)


@dataclass(frozen=True)
class TrainingConfig:
    total_updates: int = 80000
    batch_size: int = 4
    workers: int = 8
    max_pending: int = 9
    cache_bytes: int = 8 * 1024**3
    checkpoint_interval: int = 1000
    validation_interval: int = 5000
    progress_interval: int = 25
    max_wall_seconds: float = 252000.
    device: str = "cpu"

    def __post_init__(self):
        require(type(self.total_updates) is int and 0 < self.total_updates <= 80000, "update_budget")
        require(type(self.batch_size) is int and self.batch_size in (1, 2, 4), "batch_size")
        require(type(self.workers) is int and self.workers in (0, 1, 4, 8), "workers")
        require(type(self.max_pending) is int and 1 <= self.max_pending <= 9, "pending_budget")
        require(type(self.cache_bytes) is int and 0 < self.cache_bytes <= 8 * 1024**3, "cache_budget")
        require(all(type(v) is int and v > 0 for v in (self.checkpoint_interval,
            self.validation_interval, self.progress_interval)), "interval")
        require(math.isfinite(self.max_wall_seconds) and self.max_wall_seconds > 0, "wall_bound")
        require(self.device in ("cpu", "cuda"), "device")

    def contract(self):
        return dict(schema_version=SCHEMA, **asdict(self), model_name="C4",
            architecture="C3_threepass_relationgate_cloud_6e901634", from_scratch=True,
            model_seed=17, dropout_seed=17001, precision="float32", runtime_profile="coalesced_reference",
            cpu_transport="packed_tensors_v1", scheduler_total_updates=120000,
            warmup_updates=6000, peak_lr=.005, weight_decay=.0005, clip_norm=1.,
            train_targets=list(HEADS), primary_loss_heads=list(PRIMARY_LOSS_HEADS),
            auxiliary_weight=.25, selection="identity_validation_mean_root_quality_observed_true_macro_f1",
            train_routing="exact_slice_then_containing_beat", validation_routing="containing_beat",
            transposition="historical_C2_stale_track_pitch_summaries", test_locked=True)


@lru_cache(maxsize=None)
def pc_class_mapping(task):
    require(task in ("root", "bass"), "pc_task")
    labels = get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels
    return tuple(root_pc_from_label(label) for label in labels)


def pc_log_probabilities(logits, task):
    """Differentiable log marginalization over the unchanged spelled classes."""
    mapping = pc_class_mapping(task)
    require(logits.ndim == 2 and logits.shape[-1] == len(mapping), "pc_logit_width")
    values = logits.float().log_softmax(-1)
    return torch.stack([torch.logsumexp(values[:, [i for i, pc in enumerate(mapping) if pc == k]], -1)
                        for k in range(12)], -1)


def class_count(task):
    return 12 if task in ("root", "bass") else get_vocabulary(TASK_BY_ID[task].vocabulary_id).class_count


def aligned_rows(output, sample_rows, task, *, validation=False):
    """Join labels only after raw-candidate prediction; no gold topology."""
    candidates = {key: i for i, key in enumerate(output.candidate_keys.get(task, ()))}
    require(len(candidates) == len(output.candidate_keys.get(task, ())), "duplicate_candidate_key")
    indices, values = [], []
    for sample, rows in enumerate(sample_rows):
        for row in rows:
            target = row["targets"].get(task)
            if not target or target["available"] is not True:
                continue
            key = (sample, "beat", row.get("beat_id"))
            if not validation and row.get("slice_id") is not None:
                proposed = (sample, "slice", row["slice_id"])
                if proposed in candidates:
                    key = proposed
            if key not in candidates:
                continue
            value = target["value"]
            require(type(value) is int and 0 <= value < class_count(task), "invalid_available_target:" + task)
            indices.append(candidates[key]); values.append(value)
    return indices, values


def supervised_loss(output, sample_rows, class_weights=None):
    losses, counts = {}, {}
    for task in HEADS:
        indices, values = aligned_rows(output, sample_rows, task)
        counts[task] = len(values)
        if not values:
            losses[task] = None
            continue
        logits = output.logits[task][indices].float()
        target = torch.tensor(values, dtype=torch.long, device=logits.device)
        weights = None if class_weights is None else class_weights[task].to(logits.device)
        if weights is not None:
            supported = weights[target] > 0
            logits, target = logits[supported], target[supported]
            if not target.numel():
                losses[task] = None
                continue
        if task in ("root", "bass"):
            losses[task] = F.nll_loss(pc_log_probabilities(logits, task), target, weight=weights)
        else:
            losses[task] = F.cross_entropy(logits, target, weight=weights)
    primary = [losses[h] for h in PRIMARY_LOSS_HEADS if losses[h] is not None]
    terms = [torch.stack(primary).mean()] if primary else []
    if losses["pitch_class_set"] is not None:
        terms.append(.25 * losses["pitch_class_set"])
    total = sum(terms) if terms else None
    return SimpleNamespace(total=total, heads=losses, counts=counts)


def build_optimizer(model):
    return build_full_orbit_optimizer_scheduler(model)


def components(device="cpu"):
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    model = C3ThreePassRelationGateModel().to(device).train()
    validate_c3_source_model(model)
    initial = model_state_fingerprint(model)
    random.seed(17001); np.random.seed(17001); torch.manual_seed(17001)
    if device == "cuda":
        torch.cuda.manual_seed_all(17001)
    optimizer, scheduler = build_optimizer(model)
    return model, optimizer, scheduler, initial


def forward_validated(model, graph):
    """Historical coalesced path for already CPU-validated private inputs only.

    Public C3 inference retains its graph validation. This internal training
    boundary avoids copying CUDA edges back to Python to repeat the same check.
    No arithmetic, parameter or candidate-routing operation is replaced.
    """
    from music_critic.experiments.analysisgnn.temporal_slices.graph import TemporalGraphBatch, SCHEMA as GRAPH_SCHEMA
    from music_critic.experiments.analysisgnn.temporal_slices.model import TemporalSliceModel
    require(type(model) is C3ThreePassRelationGateModel and type(graph) is TemporalGraphBatch
            and graph.schema == GRAPH_SCHEMA, "validated_forward_types")
    output, _ = TemporalSliceModel.forward_with_base(model, graph)
    return model.output_type(output.logits, output.candidate_keys, output.embeddings, output.graph)


def attempt_update(model, optimizer, scheduler, graph, rows, weights=None, *, validated=False):
    optimizer.zero_grad(set_to_none=True)
    output = forward_validated(model, graph) if validated else model(graph)
    loss = supervised_loss(output, rows, weights)
    require(loss.total is not None and loss.total.dtype == torch.float32, "missing_fp32_loss")
    flags = _coalesced_host_scalars([*(torch.isfinite(v).all() for v in output.logits.values()),
                                    torch.isfinite(loss.total)])
    require(all(flags), "nonfinite_forward")
    loss.total.backward()
    finite, norm = performance_gradient_diagnostics(model)
    require(finite and math.isfinite(norm), "nonfinite_gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step(); scheduler.step()
    return dict(loss=float(loss.total.detach()), gradient_norm=norm, counts=loss.counts)


def save_checkpoint(path, *, model, optimizer, scheduler, sampler, config, bindings, progress):
    path = Path(path)
    require(not path.exists(), "checkpoint_immutable")
    require(sampler.position == config.batch_size * progress["applied_updates"], "checkpoint_cursor")
    payload = dict(schema_version=CHECKPOINT_SCHEMA, config=config.contract(), bindings=bindings,
        progress=deepcopy(progress), model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(), sampler_state=sampler.state_dict(), rng=capture_safe_rng(),
        model_state_fingerprint=model_state_fingerprint(model))
    payload["state_fingerprint"] = recursive_state_fingerprint(payload)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush(); os.fsync(handle.fileno())
    os.rename(temporary, path)
    return dict(path=str(path), **file_identity(path), update=progress["applied_updates"])


def load_checkpoint(path, *, model, optimizer, scheduler, sampler, config, bindings,
                    expected_sha256=None):
    if expected_sha256 is not None:
        require(file_identity(path)["sha256"] == expected_sha256, "checkpoint_sha256")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(state.get("schema_version") == CHECKPOINT_SCHEMA, "checkpoint_schema")
    require(state.get("bindings") == bindings and state.get("config") == config.contract(), "checkpoint_binding")
    require(state.get("state_fingerprint") == recursive_state_fingerprint({
        k: v for k, v in state.items() if k != "state_fingerprint"}), "checkpoint_digest")
    require(type(state["progress"]["applied_updates"]) is int
        and 0 <= state["progress"]["applied_updates"] <= config.total_updates, "checkpoint_progress")
    require(state["sampler_state"]["position"] == config.batch_size * state["progress"]["applied_updates"],
            "checkpoint_cursor")
    require(state["scheduler_state"]["last_epoch"] == state["progress"]["applied_updates"], "checkpoint_scheduler")
    current = model.state_dict()
    require(set(current) == set(state["model_state"]) and all(
        value.shape == current[key].shape and value.dtype == current[key].dtype
        for key, value in state["model_state"].items()), "checkpoint_model_layout")
    scratch = FullOrbitSampler(sampler.table, seed=sampler.seed)
    scratch.load_state_dict(state["sampler_state"])
    before = deepcopy((current, optimizer.state_dict(), scheduler.state_dict(),
                       sampler.state_dict(), capture_safe_rng()))
    try:
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        sampler.load_state_dict(state["sampler_state"])
        restore_safe_rng(state["rng"], device_type=config.device)
    except BaseException:
        model.load_state_dict(before[0]); optimizer.load_state_dict(before[1])
        scheduler.load_state_dict(before[2]); sampler.load_state_dict(before[3])
        restore_safe_rng(before[4], device_type=config.device)
        raise
    return state["progress"]


def registered_record(path, *, manifest=None, config=None):
    from music_critic.experiments.registry import load_record
    record = load_record(Path(path))
    require(record["status"] == "planned", "planned_experiment_required")
    policy = record["data_access"]["declared_policy"]
    require(policy.get("validation_allowed") is True, "validation_policy_required")
    require(all(policy.get(key) is False for key in (
        "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")),
        "test_must_remain_locked")
    if "source_fingerprint" in record["protocol"]:
        require(record["protocol"]["source_fingerprint"] == implementation_binding()["fingerprint"],
                "registered_source_binding")
    if manifest is not None:
        require(record["protocol"].get("c4_manifest_fingerprint") == manifest["fingerprint"],
                "registered_manifest_binding")
    if config is not None:
        require(record["protocol"].get("c4_config_fingerprint") == fingerprint(config.contract()),
                "registered_config_binding")
    return record


def prepare(*, index, split, companion_indices, source_root, output, experiment_record):
    """Authenticate TRAIN targets/eligibility once; never allocate corpus graphs."""
    from .data import (load_population, read_source_record, build_target_rows,
                       eligible_shifts, transpose_target_rows)
    record = registered_record(experiment_record)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    population = load_population(index, split, companion_indices)
    counts = {task: [0] * class_count(task) for task in HEADS}
    metadata, components_by_id, shifts_by_record, excluded = {}, {}, {}, []
    access = dict(train_source_reads=0, train_target_reads=0, validation_source_reads=0,
                  validation_target_reads=0, validation_accessed=False, test_inputs_accessed=False,
                  test_targets_accessed=False, test_metrics_computed=False, test_used_for_selection=False)
    for row in population["records"]:
        if row["split"] != "train":
            continue
        access["train_source_reads"] += 1; access["train_target_reads"] += 1
        emit(output, "prepare_source", record_id=row["record_id"], access=access)
        piece, evidence, binding = read_source_record(population, row["record_id"],
                                                     source_root, split="train")
        rows = build_target_rows(piece, evidence)
        shifts = eligible_shifts(piece)
        require(0 in shifts, "identity_shift_missing")
        metadata[row["record_id"]] = dict(target_fingerprint=fingerprint(rows),
            valid_shifts=list(shifts), source_status=evidence.status,
            raw_note_count=len(piece.notes), raw_beat_count=len(piece.beats))
        supported = [r for r in rows if (r["slice_id"] is not None or r["beat_id"] is not None)
                     and any(t["available"] for t in r["targets"].values())]
        if not supported:
            excluded.append(dict(record_id=row["record_id"], reason="no_aligned_available_targets"))
            continue
        shifts_by_record[row["record_id"]] = shifts
        components_by_id.setdefault(row["component_fingerprint"], []).append(row["record_id"])
        for shift in shifts:
            for target_row in transpose_target_rows(rows, shift):
                if target_row["slice_id"] is None and target_row["beat_id"] is None:
                    continue
                for task, state in target_row["targets"].items():
                    if state["available"]:
                        counts[task][state["value"]] += 1
    table = build_full_orbit_table(components_by_id, shifts_by_record)
    require(table, "no_training_orbit")
    weights = {task: list(_normalized_class_weights(values)) for task, values in counts.items()}
    require(all(sum(counts[h]) > 0 for h in ("root", "quality")), "root_quality_train_support_required")
    payload = dict(schema_version=MANIFEST_SCHEMA, population=population,
        records=metadata, orbit=[row.to_dict() for row in table],
        orbit_contract=full_orbit_table_contract(table), excluded_train_records=excluded,
        validation_ids=[r["record_id"] for r in population["records"] if r["split"] == "validation"],
        class_counts=counts, class_weights=weights,
        class_weight_policy="normalized_clipped_inverse_sqrt_train_eligible_orbit_only",
        implementation=implementation_binding(), preparation_experiment=record["experiment_id"])
    manifest = {**payload, "fingerprint": fingerprint(payload)}
    write_json(output / "manifest.json", manifest)
    report = dict(schema_version=SCHEMA, status="completed", phase="prepare",
        experiment_id=record["experiment_id"], manifest_fingerprint=manifest["fingerprint"],
        manifest=file_identity(output / "manifest.json"), source_fingerprint=payload["implementation"]["fingerprint"],
        applied_updates=0, attempted_updates=0,
        skipped_updates=0, draws=0, access=access, **access,
        wall_seconds=time.monotonic() - started, orbit=payload["orbit_contract"],
        excluded_train_records=excluded, split_counts=population["split_counts"])
    write_json(output / "report.json", report)
    return report


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(payload.get("schema_version") == MANIFEST_SCHEMA and payload.get("fingerprint") == fingerprint({
        k: v for k, v in payload.items() if k != "fingerprint"}), "manifest_binding")
    require(payload["implementation"] == implementation_binding(), "implementation_drift")
    table = tuple(FullOrbitPair(**row) for row in payload["orbit"])
    require(full_orbit_table_contract(table) == payload["orbit_contract"], "orbit_binding")
    train = {r["record_id"] for r in payload["population"]["records"] if r["split"] == "train"}
    validation = {r["record_id"] for r in payload["population"]["records"] if r["split"] == "validation"}
    require(set(row.record_id for row in table) <= train
        and set(payload["validation_ids"]) == validation and bool(validation), "manifest_split_binding")
    return payload


def manifest_weights(manifest, device):
    return {task: torch.tensor([0. if value is None else value for value in manifest["class_weights"][task]],
                              dtype=torch.float32, device=device) for task in HEADS}


class RecordCache:
    """Bound retained identity graphs; cache never contains transposed inputs."""
    def __init__(self, manifest, source_root, max_bytes):
        self.manifest, self.source_root, self.max_bytes = manifest, Path(source_root), max_bytes
        self.entries = OrderedDict(); self.bytes = 0
        self.hits = self.misses = self.evictions = self.oversized = 0
        self.rows = {r["record_id"]: r for r in manifest["population"]["records"]}

    def get(self, record_id):
        from .data import read_record
        row = self.rows[record_id]
        require(row["split"] == "train", "cache_train_only")
        root = self.source_root.resolve()
        path = (root / row["source_relative_path"]).resolve()
        require(path.is_relative_to(root), "cache_source_path")
        # Check retained-entry authenticity on every hit as the historical cache does.
        require(file_identity(path)["sha256"] == row["source_sha256"], "cache_source_changed")
        if record_id in self.entries:
            self.hits += 1
            prepared, size = self.entries.pop(record_id)
            self.entries[record_id] = (prepared, size)
            return prepared
        self.misses += 1
        prepared = read_record(self.manifest["population"], record_id, root, split="train")
        expected = self.manifest["records"][record_id]
        require(prepared.source_binding["target_fingerprint"] == expected["target_fingerprint"]
            and list(prepared.valid_shifts) == expected["valid_shifts"], "prepared_record_binding")
        size = accounted_bytes(prepared)
        if size > self.max_bytes:
            self.oversized += 1
            return prepared
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed; self.evictions += 1
        self.entries[record_id] = (prepared, size); self.bytes += size
        return prepared

    def diagnostics(self):
        return dict(entries=len(self.entries), accounted_bytes=self.bytes, max_bytes=self.max_bytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions, oversized_bypasses=self.oversized)


@dataclass(frozen=True)
class BatchJob:
    update: int
    draw_cursor: int
    record_ids: tuple[str, ...]
    shifts: tuple[int, ...]


def batch_jobs(table, *, start_update, stop_update, batch_size):
    sampler = FullOrbitSampler(table, position=start_update * batch_size)
    for update in range(start_update, stop_update):
        draws = tuple(sampler.peek(i) for i in range(batch_size))
        yield BatchJob(update, sampler.position, tuple(d.record_id for d in draws),
                       tuple(d.shift_pc for d in draws))
        sampler.advance_after_applied_update(batch_size)


def prepare_batch(cache, job):
    from .data import transpose_target_rows
    loaded = [cache.get(rid) for rid in job.record_ids]
    require(all(shift in record.valid_shifts for record, shift in zip(loaded, job.shifts)), "ineligible_shift")
    graph = transpose_temporal_historical(collate_temporal_graphs([r.graph for r in loaded]), job.shifts)
    rows = tuple(transpose_target_rows(record.target_rows, shift) for record, shift in zip(loaded, job.shifts))
    return graph, rows


_worker_cache = None


def _worker_init(manifest, source_root, max_bytes):
    global _worker_cache
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(manifest["implementation"] == implementation_binding(), "worker_implementation_binding")
    _worker_cache = RecordCache(manifest, source_root, max_bytes)


def _worker_prepare(job):
    from .transport import assert_private_cpu_payload, pack_private_cpu_payload
    graph, rows = prepare_batch(_worker_cache, job)
    body = dict(job=asdict(job), manifest=_worker_cache.manifest["fingerprint"],
        graph=encode_temporal_graph(graph), rows=rows, cache=_worker_cache.diagnostics(), pid=os.getpid())
    assert_private_cpu_payload(body)
    return pack_private_cpu_payload(body, cap=PAYLOAD_CAP)


class OrderedPool:
    """Bound submitted/uncommitted batches, preserve sampler order and ownership."""
    def __init__(self, jobs, manifest, source_root, config, *, executor=None, prepare_fn=None):
        self.jobs = iter(jobs); self.pending = deque(); self.delivered = None
        self.manifest, self.config = manifest, config
        self.closed = False; self.committed = 0; self.submitted = 0
        self.worker_observations = {}; self.payload_max = 0
        self.direct = RecordCache(manifest, source_root, config.cache_bytes) if config.workers == 0 else None
        self.executor = executor
        self.prepare_fn = prepare_fn or _worker_prepare
        if config.workers and self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=config.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
                initargs=(manifest, str(source_root), config.cache_bytes // config.workers))
        try:
            self._fill()
        except BaseException:
            self.close(); raise

    def _fill(self):
        while len(self.pending) < self.config.max_pending:
            job = next(self.jobs, None)
            if job is None:
                return
            future = self.executor.submit(self.prepare_fn, job) if self.executor else None
            self.pending.append((job, future)); self.submitted += 1

    def peek(self):
        require(not self.closed and bool(self.pending), "closed_or_empty_pool")
        if self.delivered is not None:
            return self.delivered
        job, future = self.pending[0]
        if future is None:
            graph, rows = prepare_batch(self.direct, job)
        else:
            from .transport import unpack_private_cpu_payload, assert_private_cpu_payload
            payload = future.result(timeout=600.)
            require(type(payload) is bytes and len(payload) <= PAYLOAD_CAP, "worker_payload_cap")
            self.payload_max = max(self.payload_max, len(payload))
            body = unpack_private_cpu_payload(payload, cap=PAYLOAD_CAP)
            assert_private_cpu_payload(body)
            require(body["job"] == asdict(job) and body["manifest"] == self.manifest["fingerprint"],
                    "worker_payload_binding")
            graph, rows = decode_temporal_graph(body["graph"]), body["rows"]
            self.worker_observations[str(body["pid"])] = body["cache"]
        require(len(graph.slots) == len(job.record_ids) == len(rows), "worker_batch_width")
        self.delivered = SimpleNamespace(job=job, graph=graph, rows=rows)
        return self.delivered

    def commit(self, job):
        require(self.delivered is not None and self.delivered.job == job, "pool_commit")
        self.pending.popleft(); self.delivered = None; self.committed += 1
        self._fill()

    def diagnostics(self):
        return dict(submitted=self.submitted, committed=self.committed,
            pending=len(self.pending), max_pending=self.config.max_pending, workers=self.config.workers,
            payload_max_bytes=self.payload_max, reserved_bytes=self.config.max_pending * PAYLOAD_CAP,
            cache={} if self.direct is None else self.direct.diagnostics(), worker_caches=self.worker_observations)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.executor is not None:
            for _, future in self.pending:
                future.cancel()
            processes = tuple((getattr(self.executor, "_processes", None) or {}).values())
            self.executor.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + 5.
            for process in processes:
                process.join(timeout=max(0., deadline - time.monotonic()))
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=.2)
                if process.is_alive():
                    process.kill(); process.join(timeout=.5)


def configure_runtime(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "cublas_workspace_binding")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    require(device != "cuda" or torch.cuda.is_available(), "cuda_unavailable")
    if device == "cuda":
        initialize_cuda_memory_statistics(resolve_runtime_device(device))
    return dict(torch=str(torch.__version__), numpy=str(np.__version__), cuda_build=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), device=device,
        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic=True, cudnn_allow_tf32=True, matmul_allow_tf32=False)


def runtime_bindings(manifest, config, environment):
    return dict(source=manifest["implementation"]["fingerprint"], manifest=manifest["fingerprint"],
                config=fingerprint(config.contract()), environment=environment)


def update_progress(output, progress):
    path = Path(output) / "progress.json"
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(progress, handle, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush()
    os.replace(temporary, path)


def gpu_memory(device):
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)


def validate(model, manifest, source_root, output, update, access, *, device):
    from .data import read_record
    matrices = {task: torch.zeros((class_count(task), class_count(task)), dtype=torch.int64) for task in HEADS}
    masked = {task: 0 for task in HEADS}
    unaligned = {task: 0 for task in HEADS}
    was_training = model.training
    started = time.monotonic()
    try:
        with preserve_caller_rng(), torch.no_grad():
            model.eval()
            for record_id in manifest["validation_ids"]:
                access["validation_accessed"] = True
                access["validation_source_reads"] += 1; access["validation_target_reads"] += 1
                emit(output, "validation_source", update=update, record_id=record_id, access=access)
                record = read_record(manifest["population"], record_id, source_root, split="validation")
                predicted = forward_validated(model, record.graph.to(device))
                access["validation_model_forwards"] += 1
                require(all(_coalesced_host_scalars([torch.isfinite(x).all() for x in predicted.logits.values()])),
                        "nonfinite_validation")
                for task in HEADS:
                    indices, gold = aligned_rows(predicted, (record.target_rows,), task, validation=True)
                    masked[task] += sum(not row["targets"][task]["available"] for row in record.target_rows)
                    unaligned[task] += sum(row["targets"][task]["available"] for row in record.target_rows) - len(gold)
                    if not gold:
                        continue
                    logits = predicted.logits[task][indices].float()
                    guesses = (pc_log_probabilities(logits, task) if task in ("root", "bass") else logits).argmax(-1).cpu()
                    classes = class_count(task)
                    matrices[task] += torch.bincount(torch.tensor(gold) * classes + guesses,
                                                    minlength=classes**2).reshape(classes, classes)
    finally:
        model.train(was_training)
    metrics = {}
    for task, matrix in matrices.items():
        true = matrix.sum(1); predicted = matrix.sum(0); diagonal = matrix.diag()
        observed = true > 0
        f1 = 2. * diagonal.double() / (true + predicted).clamp_min(1).double()
        metrics[task] = dict(eligible_events=int(true.sum()), masked_events=masked[task],
            unaligned_events=unaligned[task], observed_class_ids=observed.nonzero().flatten().tolist(),
            accuracy=float(diagonal.sum().double() / true.sum()) if bool(true.sum()) else None,
            observed_true_class_macro_f1=float(f1[observed].mean()) if bool(observed.any()) else None,
            confusion=matrix.tolist())
    primary_values = [metrics[task]["observed_true_class_macro_f1"] for task in ("root", "quality")]
    require(all(value is not None and math.isfinite(value) for value in primary_values),
            "validation_primary_support_missing")
    row = dict(update=update, primary_score=sum(primary_values) / 2., metrics=metrics,
        record_count=len(manifest["validation_ids"]), identity_only=True, candidate_policy="raw_containing_beat",
        wall_seconds=time.monotonic() - started)
    write_json(Path(output) / f"validation-{update:06d}.json", row)
    emit(output, "validation_complete", update=update, primary_score=row["primary_score"])
    return row


def train(*, manifest_path, source_root, output, experiment_record, config, resume=None,
          resume_sha256=None):
    manifest = load_manifest(manifest_path)
    record = registered_record(experiment_record, manifest=manifest, config=config)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    environment = configure_runtime(config.device)
    bindings = runtime_bindings(manifest, config, environment)
    write_json(output / "bindings.json", bindings); write_json(output / "config.json", config.contract())
    table = tuple(FullOrbitPair(**row) for row in manifest["orbit"])
    sampler = FullOrbitSampler(table)
    model, optimizer, scheduler, initial = components(config.device)
    weights = manifest_weights(manifest, config.device)
    progress = dict(applied_updates=0, attempted_updates=0, skipped_updates=0,
        draw_cursor=0, best_score=None, best_update=None, best_checkpoint_path=None,
        validation_history=[], pending_validation=0, elapsed_seconds=0.)
    if resume:
        require(resume_sha256 is not None, "resume_sha256_required")
        progress = load_checkpoint(resume, expected_sha256=resume_sha256,
            model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
            config=config, bindings=bindings)
    start_update = progress["applied_updates"]
    access = dict(train_draws=0, train_model_forwards=0, validation_source_reads=0,
        validation_target_reads=0, validation_model_forwards=0, validation_accessed=False,
        test_inputs_accessed=False, test_targets_accessed=False, test_metrics_computed=False,
        test_used_for_selection=False)
    pool = None; stopped = {"signal": None}; status = "failed"; failure = None
    checkpoints = {}; best_checkpoint = None; final_checkpoint = None
    update_in_flight = False
    resume_parent = None if resume is None else dict(path=str(resume), **file_identity(resume),
                                                    update=start_update)
    if progress.get("best_checkpoint_path") and Path(progress["best_checkpoint_path"]).is_file():
        best_checkpoint = dict(path=progress["best_checkpoint_path"], update=progress["best_update"],
                               **file_identity(progress["best_checkpoint_path"]))
    handlers = {}
    def stop(signum, frame):
        stopped["signal"] = signum
    def checkpoint():
        update = progress["applied_updates"]
        if update not in checkpoints:
            progress["elapsed_seconds"] = time.monotonic() - started
            checkpoints[update] = save_checkpoint(output / f"checkpoint-{update:06d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
                config=config, bindings=bindings, progress=progress)
        return checkpoints[update]
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, stop)
        if progress.get("pending_validation") is not None:
            update = progress["pending_validation"]
            require(update == progress["applied_updates"], "pending_validation_update")
            result = validate(model, manifest, source_root, output, update, access, device=config.device)
            progress["validation_history"].append(dict(update=update, score=result["primary_score"]))
            progress["pending_validation"] = None
            if progress["best_score"] is None or result["primary_score"] > progress["best_score"]:
                progress["best_score"] = result["primary_score"]; progress["best_update"] = update
                progress["best_checkpoint_path"] = str(output / f"checkpoint-{update:06d}.pt")
                best_checkpoint = checkpoint()
        jobs = batch_jobs(table, start_update=start_update, stop_update=config.total_updates,
                          batch_size=config.batch_size)
        pool = OrderedPool(jobs, manifest, source_root, config)
        while progress["applied_updates"] < config.total_updates:
            if stopped["signal"] is not None or time.monotonic() - started >= config.max_wall_seconds:
                status = "aborted"; break
            batch = pool.peek()
            draws = tuple(sampler.peek(i) for i in range(config.batch_size))
            require(batch.job.update == progress["applied_updates"]
                and batch.job.draw_cursor == sampler.position
                and batch.job.record_ids == tuple(d.record_id for d in draws)
                and batch.job.shifts == tuple(d.shift_pc for d in draws), "sampler_queue_mismatch")
            progress["attempted_updates"] += 1
            update_progress(output, {**progress, "access": access})
            access["train_model_forwards"] += 1; access["train_draws"] += config.batch_size
            update_in_flight = True
            result = attempt_update(model, optimizer, scheduler, batch.graph.to(config.device),
                                    batch.rows, weights, validated=True)
            sampler.advance_after_applied_update(config.batch_size)
            progress["applied_updates"] += 1; progress["draw_cursor"] = sampler.position
            if (progress["applied_updates"] % config.validation_interval == 0
                    or progress["applied_updates"] == config.total_updates):
                progress["pending_validation"] = progress["applied_updates"]
            journal_update(output, dict(update=progress["applied_updates"], draw_cursor=sampler.position,
                record_ids=list(batch.job.record_ids), shifts=list(batch.job.shifts), **result,
                lr=optimizer.param_groups[0]["lr"], wall_seconds=time.monotonic() - started))
            update_in_flight = False
            pool.commit(batch.job)
            update = progress["applied_updates"]
            if update % config.progress_interval == 0 or update == config.total_updates:
                emit(output, "training", update=update, draws=sampler.position,
                    lr=optimizer.param_groups[0]["lr"], **result,
                    wall_seconds=time.monotonic() - started, gpu=gpu_memory(config.device))
                update_progress(output, {**progress, "access": access, "last_update": result})
            if update % config.validation_interval == 0 or update == config.total_updates:
                result = validate(model, manifest, source_root, output, update, access, device=config.device)
                progress["validation_history"].append(dict(update=update, score=result["primary_score"]))
                progress["pending_validation"] = None
                if result["primary_score"] > progress["best_score"]:
                    progress["best_score"] = result["primary_score"]; progress["best_update"] = update
                    progress["best_checkpoint_path"] = str(output / f"checkpoint-{update:06d}.pt")
                    best_checkpoint = checkpoint()
            if update % config.checkpoint_interval == 0 or update == config.total_updates:
                final_checkpoint = checkpoint()
        else:
            status = "completed"
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        emit(output, "failed", error=failure, progress=progress, access=access)
    finally:
        if pool is not None:
            pool.close()
            write_json(output / "pool.json", pool.diagnostics())
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        try:
            if not update_in_flight:
                final_checkpoint = checkpoint()
            else:
                # Failed forwards/backwards/optimizer steps have uncommitted RNG
                # and possibly parameters. Resume only a known committed state.
                final_checkpoint = checkpoints[max(checkpoints)] if checkpoints else resume_parent
        except BaseException as exc:
            status = "failed"
            failure = failure or dict(type=type(exc).__name__, message=str(exc))
        update_progress(output, {**progress, "access": access})
    require(status != "completed" or (progress["applied_updates"] == config.total_updates
        and progress["validation_history"][-1]["update"] == config.total_updates), "terminal_full_budget")
    report = dict(schema_version=SCHEMA, phase="train", status=status,
        experiment_id=record["experiment_id"], manifest_fingerprint=manifest["fingerprint"],
        source_fingerprint=manifest["implementation"]["fingerprint"], config=config.contract(),
        applied_updates=progress["applied_updates"], attempted_updates=progress["attempted_updates"],
        skipped_updates=progress["skipped_updates"], draws=sampler.position,
        this_attempt_updates=progress["applied_updates"] - start_update,
        validation_accessed=access["validation_source_reads"] > 0, access=access,
        test_inputs_accessed=False, test_targets_accessed=False, test_metrics_computed=False,
        best_score=progress["best_score"], best_update=progress["best_update"],
        best_checkpoint=best_checkpoint, final_checkpoint=final_checkpoint,
        best_checkpoint_reference=progress.get("best_checkpoint_path"), resume_parent=resume_parent,
        failed_update_state_discarded=update_in_flight,
        validation_history=progress["validation_history"], initial_model_fingerprint=initial,
        final_model_fingerprint=model_state_fingerprint(model), wall_seconds=time.monotonic() - started,
        gpu=gpu_memory(config.device), error=failure, stopped_signal=stopped["signal"])
    write_json(output / "report.json", report)
    return report


def state_digest(parts):
    model, optimizer, scheduler = parts[:3]
    return recursive_state_fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), gradients={name: p.grad for name, p in model.named_parameters()},
        rng=capture_safe_rng()))


def gate(*, manifest_path, source_root, output, experiment_record, config):
    """32 discarded updates: all shifts/reference-pipeline parity and full resume."""
    manifest = load_manifest(manifest_path)
    record = registered_record(experiment_record, manifest=manifest, config=config)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); environment = configure_runtime(config.device)
    bindings = runtime_bindings(manifest, config, environment)
    write_json(output / "bindings.json", bindings); write_json(output / "config.json", config.contract())
    table = tuple(FullOrbitPair(**row) for row in manifest["orbit"])
    full_ids = sorted({row.record_id for row in table if len(manifest["records"][row.record_id]["valid_shifts"]) == 12},
        key=lambda rid: (-manifest["records"][rid]["raw_note_count"], rid))
    require(len(full_ids) >= config.batch_size, "all_shift_gate_population")
    ids = tuple(full_ids[:config.batch_size])
    coverage_jobs = tuple(BatchJob(i, i * config.batch_size, ids,
                                   tuple((i + 5 * j) % 12 for j in range(config.batch_size))) for i in range(12))
    weights = manifest_weights(manifest, config.device)
    actual = attempted = 0; pool = None; status = "failed"; failure = None; coverage = []
    resume_pin = None; census = []
    def guard():
        require(time.monotonic() - started < config.max_wall_seconds, "gate_wall_bound")
    try:
        a, b = components(config.device), components(config.device)
        rng_a = capture_safe_rng(); rng_b = deepcopy(rng_a)
        # Source/cache preparation is deterministic and cannot consume model RNG.
        serial_cache = RecordCache(manifest, source_root, max_bytes=1)
        pool = OrderedPool(coverage_jobs, manifest, source_root, config)
        for job in coverage_jobs:
            guard()
            before = recursive_state_fingerprint(capture_safe_rng())
            prepared = pool.peek()
            reference, rows = prepare_batch(serial_cache, job)
            require(before == recursive_state_fingerprint(capture_safe_rng()), "preparation_consumed_rng")
            require(reference.fingerprint() == prepared.graph.fingerprint()
                and fingerprint(rows) == fingerprint(prepared.rows), "gate_preparation_parity")
            if not census:
                census = [dict(record_id=rid, raw_note_count=manifest["records"][rid]["raw_note_count"],
                    raw_beat_count=manifest["records"][rid]["raw_beat_count"]) for rid in ids]
            restore_safe_rng(rng_a, device_type=config.device)
            attempted += 1
            first = attempt_update(*a[:3], reference.to(config.device), rows, weights, validated=False)
            actual += 1; rng_a = capture_safe_rng(); digest_a = state_digest(a)
            restore_safe_rng(rng_b, device_type=config.device)
            attempted += 1
            second = attempt_update(*b[:3], prepared.graph.to(config.device), prepared.rows, weights, validated=True)
            actual += 1; rng_b = capture_safe_rng()
            require(first == second and digest_a == state_digest(b), "gate_exact_update_parity")
            pool.commit(job)
            row = dict(position=job.update, shifts=list(job.shifts), **second)
            coverage.append(row); emit(output, "gate_parity", actual_updates=actual, **row)
        pool.close(); write_json(output / "coverage-pool.json", pool.diagnostics()); pool = None
        del a, b, reference, rows, prepared, serial_cache
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
        expected = []
        for phase, start, stop in (("continuous", 0, 4), ("prefix", 0, 2), ("suffix", 2, 4)):
            parts = components(config.device); sampler = FullOrbitSampler(table)
            if start:
                loaded = load_checkpoint(output / "resume-000002.pt", expected_sha256=resume_pin["sha256"],
                    model=parts[0], optimizer=parts[1], scheduler=parts[2], sampler=sampler,
                    config=config, bindings=bindings)
                require(loaded["applied_updates"] == start, "gate_resume_progress")
            pool = OrderedPool(batch_jobs(table, start_update=start, stop_update=stop, batch_size=config.batch_size),
                               manifest, source_root, config)
            for update in range(start, stop):
                guard()
                prepared = pool.peek()
                require(prepared.job.draw_cursor == sampler.position, "gate_resume_cursor")
                attempted += 1
                result = attempt_update(*parts[:3], prepared.graph.to(config.device), prepared.rows,
                                        weights, validated=True)
                actual += 1
                sampler.advance_after_applied_update(config.batch_size); pool.commit(prepared.job)
                digest = state_digest(parts)
                if phase == "continuous":
                    expected.append(digest)
                else:
                    require(digest == expected[update], "gate_exact_resume_parity")
                emit(output, "gate_resume", phase=phase, update=update + 1, actual_updates=actual, **result)
            if phase == "prefix":
                resume_pin = save_checkpoint(output / "resume-000002.pt", model=parts[0], optimizer=parts[1],
                    scheduler=parts[2], sampler=sampler, config=config, bindings=bindings,
                    progress=dict(applied_updates=2))
            pool.close(); write_json(output / f"{phase}-pool.json", pool.diagnostics()); pool = None
            del parts, prepared
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()
        require(actual == 32, "gate_actual_update_count")
        status = "completed"
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
        emit(output, "gate_failed", actual_updates=actual, error=failure)
    finally:
        if pool is not None:
            pool.close()
    report = dict(schema_version=SCHEMA, phase="gate", status=status,
        experiment_id=record["experiment_id"], manifest_fingerprint=manifest["fingerprint"],
        source_fingerprint=manifest["implementation"]["fingerprint"], applied_updates=actual,
        attempted_updates=attempted, skipped_updates=0, draws=actual * config.batch_size,
        validation_accessed=False, test_inputs_accessed=False, test_targets_accessed=False,
        test_metrics_computed=False, wall_seconds=time.monotonic() - started,
        access=dict(validation_accessed=False, test_inputs_accessed=False, test_targets_accessed=False,
                    test_metrics_computed=False, test_used_for_selection=False,
                    train_optimizer_updates=actual, train_draws=actual * config.batch_size),
        gpu=gpu_memory(config.device), shift_coverage=sorted({s for row in coverage for s in row["shifts"]}),
        coverage=coverage, exact_reference_pipeline_parity=status == "completed",
        exact_full_resume_parity=status == "completed", resume_checkpoint=resume_pin,
        sizing_records=census, fresh_training_required=True, error=failure)
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--index", type=Path, required=True)
    prep.add_argument("--split", type=Path, required=True)
    prep.add_argument("--companion-index", type=Path, action="append", default=[])
    for child in (prep, sub.add_parser("train"), sub.add_parser("gate")):
        child.add_argument("--source-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--experiment-record", type=Path, required=True)
        if child is prep:
            continue
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--updates", type=int, default=80000)
        child.add_argument("--batch-size", type=int, default=4)
        child.add_argument("--workers", type=int, default=8)
        child.add_argument("--max-pending", type=int, default=9)
        child.add_argument("--cache-bytes", type=int, default=8 * 1024**3)
        child.add_argument("--checkpoint-interval", type=int, default=1000)
        child.add_argument("--validation-interval", type=int, default=5000)
        child.add_argument("--progress-interval", type=int, default=25)
        child.add_argument("--max-wall-seconds", type=float, default=252000.)
        child.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if child.prog.endswith(" train"):
            child.add_argument("--resume", type=Path)
            child.add_argument("--resume-sha256")
    args = parser.parse_args(argv)
    output_existed = args.output.exists()
    try:
        if args.command == "prepare":
            report = prepare(index=args.index, split=args.split, companion_indices=tuple(args.companion_index),
                source_root=args.source_root, output=args.output, experiment_record=args.experiment_record)
        else:
            config = TrainingConfig(total_updates=args.updates, batch_size=args.batch_size,
                workers=args.workers, max_pending=args.max_pending, cache_bytes=args.cache_bytes,
                checkpoint_interval=args.checkpoint_interval, validation_interval=args.validation_interval,
                progress_interval=args.progress_interval, max_wall_seconds=args.max_wall_seconds, device=args.device)
            kwargs = dict(manifest_path=args.manifest, source_root=args.source_root, output=args.output,
                          experiment_record=args.experiment_record, config=config)
            if args.command == "train":
                report = train(**kwargs, resume=args.resume, resume_sha256=args.resume_sha256)
            else:
                report = gate(**kwargs)
        return 0 if report["status"] == "completed" else 1
    except BaseException as exc:
        # Preserve a terminal receipt even for preflight/registration failures.
        # An existing report/output is never overwritten on an accidental rerun.
        if output_existed:
            raise
        args.output.mkdir(parents=True, exist_ok=True)
        if not (args.output / "report.json").exists():
            record = json.loads(args.experiment_record.read_text(encoding="utf-8"))
            progress_path = args.output / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            access = progress.get("access", {})
            events_path = args.output / "events.jsonl"
            if events_path.is_file():
                for line in events_path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    access = event.get("access", access)
                    if "actual_updates" in event:
                        progress["applied_updates"] = event["actual_updates"]
            write_json(args.output / "report.json", dict(schema_version=SCHEMA, status="failed",
                phase=args.command, experiment_id=record.get("experiment_id"),
                manifest_fingerprint=record.get("protocol", {}).get("c4_manifest_fingerprint"),
                source_fingerprint=record.get("protocol", {}).get("source_fingerprint"),
                applied_updates=progress.get("applied_updates"),
                attempted_updates=progress.get("attempted_updates"),
                skipped_updates=progress.get("skipped_updates"), draws=progress.get("draw_cursor"),
                access=access, validation_accessed=access.get("validation_accessed"),
                test_inputs_accessed=access.get("test_inputs_accessed"),
                test_targets_accessed=access.get("test_targets_accessed"),
                test_metrics_computed=access.get("test_metrics_computed"),
                accounting="last_durable_progress_or_explicitly_unknown",
                error=dict(type=type(exc).__name__, message=str(exc))))
        raise
