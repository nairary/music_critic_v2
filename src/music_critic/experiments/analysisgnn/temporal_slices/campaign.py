"""Isolated C2 temporal-slice campaign with bounded ordered CPU preparation.

Historical campaigns, six-type graph caches and checkpoints are not modified.
The cache is disposable preparation state; only applied draw positions commit.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import math
import multiprocessing
import os
import pickle
import random
import signal
import time

import numpy as np
import torch

from .. import corrected_training as ref
from ..contracts import fingerprint
from ..corrected_model import model_state_fingerprint
from ..full_orbit_training import FullOrbitSampler, build_full_orbit_optimizer_scheduler
from ..frozen_training_data import FrozenTrainingData
from ..training_rng import capture_safe_rng, preserve_caller_rng, restore_safe_rng
from ..performance_prefetch import PrefetchJob, _LimitedBuffer, _assert_cpu
from ..performance_replay import _scaler, recursive_state_fingerprint
from ..performance_training import (
    _coalesced_host_scalars, performance_gradient_diagnostics,
    prepare_performance_class_weights,
)
from .alignment import align_temporal_targets, temporal_supervised_loss
from .model import MODEL_SCHEMA, TemporalSliceModel
from .training import attempt_temporal_update

SCHEMA = "TemporalSliceCampaign@1.0.0"
CAP = 477102080
CACHE_BYTES = 8 * 1024**3
WORKERS = 4
PENDING = 5
TOTAL = 80000
_worker_cache = None
_worker_source = None


def require(condition, reason):
    if not condition:
        raise ValueError("temporal_campaign." + reason)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def emit(out, event, **values):
    row = dict(event=event, time=time.time(), **values)
    encoded = json.dumps(row, sort_keys=True, allow_nan=False)
    with (Path(out) / "events.jsonl").open("a") as handle:
        handle.write(encoded + "\n")
    print(encoded, flush=True)


def training_jobs(table, start_update=0, stop_update=TOTAL, *, cap=CAP):
    require(type(start_update) is int and type(stop_update) is int
            and 0 <= start_update < stop_update <= TOTAL, "budget")
    sampler = FullOrbitSampler(table, seed=17, position=2 * start_update)
    jobs = []
    for position, update in enumerate(range(start_update, stop_update)):
        pair = [sampler.peek(k) for k in range(2)]
        shifts = tuple(p.shift_pc for p in pair)
        jobs.append(PrefetchJob(position, (2 * update, 2 * update + 1),
            tuple(p.record_id for p in pair), shifts, shifts, cap))
        sampler.advance_after_applied_update(2)
    return tuple(jobs)


def gate_jobs(table, *, cap=CAP):
    eligible = {}
    for row in table:
        eligible.setdefault(row.record_id, set()).add(row.shift_pc)
    ids = tuple(sorted(r for r, shifts in eligible.items() if shifts == set(range(12)))[:2])
    require(len(ids) == 2, "all_shift_gate_records_unavailable")
    return tuple(PrefetchJob(i, (2*i, 2*i+1), ids, (i, (i+5) % 12),
                            (i, (i+5) % 12), cap) for i in range(12))


def _cpu_rng():
    return recursive_state_fingerprint((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def _worker_init(manifest, source_fingerprint, per_worker_cache, expected_implementation):
    global _worker_cache, _worker_source
    # Spawn workers never create a CUDA model or a CUDA context.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from .cache import FrozenTemporalRecordLoader, TemporalGraphCache, implementation_fingerprint
    require(expected_implementation == {"cache": implementation_fingerprint(),
                "campaign": sha256(Path(__file__).read_bytes()).hexdigest()}, "worker_implementation_mismatch")
    data = FrozenTrainingData(manifest)
    _worker_cache = TemporalGraphCache(FrozenTemporalRecordLoader(data),
        max_bytes=per_worker_cache, max_entries=1295)
    _worker_source = source_fingerprint


def _worker_prepare(job):
    from .cache import encode_temporal_graph, transpose_temporal_historical
    before = _cpu_rng()
    prepared = _worker_cache.prepare_batch(job.record_ids)
    graph = transpose_temporal_historical(prepared.graph, job.applied_shifts)
    body = dict(job=asdict(job), source_fingerprint=_worker_source,
                graph=encode_temporal_graph(graph), sidecars=prepared.sidecars,
                cache=prepared.cache_diagnostics, pid=os.getpid())
    require(before == _cpu_rng(), "preparation_consumed_global_rng")
    _assert_cpu(body)
    stream = _LimitedBuffer(job.reserved_bytes)
    pickle.dump(body, stream, protocol=5)
    return stream.getvalue()


class OrderedTemporalPool:
    """At most five reserved submitted/uncommitted jobs, delivered in order.

    The five payload reservations bound serialized results, not total RSS;
    decoded batches, worker working sets and IPC copies need separate sizing.
    """
    def __init__(self, jobs, manifest, *, source_fingerprint, workers=WORKERS,
                 cache_bytes=CACHE_BYTES, max_pending=PENDING, timeout=300,
                 _executor=None, _prepare=None):
        self.jobs = tuple(jobs)
        require(bool(self.jobs), "empty_queue")
        require(type(workers) is int and workers == WORKERS, "four_workers_required")
        require(type(max_pending) is int and 1 <= max_pending <= PENDING, "pending_budget")
        require(type(cache_bytes) is int and 0 < cache_bytes <= CACHE_BYTES, "cache_budget")
        require(all(j.position == i for i, j in enumerate(self.jobs)), "ordered_job_positions")
        require(all(j.reserved_bytes == self.jobs[0].reserved_bytes for j in self.jobs), "uniform_reservations")
        require(self.jobs[0].reserved_bytes <= CAP, "payload_cap")
        self.source_fingerprint = source_fingerprint
        self.max_pending, self.timeout = max_pending, timeout
        self.next_submit = self.committed = 0
        self.delivered = None
        self.pending = deque()
        self.closed = False
        self.worker_observations = {}
        self.payload_max = 0
        self._prepare = _prepare or _worker_prepare
        from .cache import implementation_fingerprint
        expected_implementation = {"cache": implementation_fingerprint(),
                                   "campaign": sha256(Path(__file__).read_bytes()).hexdigest()}
        self.executor = _executor or ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
            initargs=(manifest, source_fingerprint, cache_bytes // workers, expected_implementation))
        try:
            self._fill()
        except BaseException:
            self.close()
            raise

    def _fill(self):
        while self.next_submit < len(self.jobs) and len(self.pending) < self.max_pending:
            job = self.jobs[self.next_submit]
            self.pending.append((job, self.executor.submit(self._prepare, job)))
            self.next_submit += 1

    def peek(self):
        require(not self.closed and self.pending, "closed_or_empty_queue")
        if self.delivered is not None:
            return self.delivered
        job, future = self.pending[0]
        payload = future.result(timeout=self.timeout)
        require(type(payload) is bytes and len(payload) <= job.reserved_bytes, "invalid_worker_payload")
        self.payload_max = max(self.payload_max, len(payload))
        # These bytes are produced only by our spawned bound worker, never imported artifacts.
        row = pickle.loads(payload)
        require(row["job"] == asdict(job) and row["source_fingerprint"] == self.source_fingerprint,
                "worker_job_or_source_mismatch")
        from .cache import decode_temporal_graph
        graph = decode_temporal_graph(row["graph"])
        require(len(graph.slots) == 2 and tuple(s["record_id"] for s in row["sidecars"]) == job.record_ids,
                "worker_batch_identity")
        self.worker_observations[str(row["pid"])] = row["cache"]
        self.delivered = SimpleNamespace(job=job, graph=graph, sidecars=tuple(row["sidecars"]))
        return self.delivered

    def commit(self, position, *, applied):
        require(applied is True and self.delivered is not None
                and position == self.committed == self.delivered.job.position, "invalid_commit")
        self.pending.popleft()
        self.delivered = None
        self.committed += 1
        self._fill()

    def diagnostics(self):
        return dict(committed_batches=self.committed, submitted_batches=self.next_submit,
                    pending_batches=len(self.pending), max_pending_batches=self.max_pending,
                    max_serialized_payload_bytes=self.payload_max,
                    reservation_cap_bytes=self.max_pending * self.jobs[0].reserved_bytes,
                    worker_cache_observations=self.worker_observations,
                    cache_not_checkpoint_state=True)

    def close(self):
        if not self.closed:
            self.closed = True
            for _, future in self.pending:
                future.cancel()
            # Only our own spawned processes are eligible for termination.
            # A stalled preparer must not turn a failed admission into an
            # unbounded shutdown. ProcessPool has no public timed shutdown.
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
                    process.kill()
                    process.join(timeout=.5)


def scientific_config():
    return dict(schema_version=SCHEMA, model_kind="temporal_slice_c2", batch_size=2,
                total_updates=TOTAL, scheduler_total_updates=120000, seed=17,
                dropout_rng_seed=17001, workers=WORKERS, cache_bytes=CACHE_BYTES,
                checkpoint_interval=1000, validation_interval=5000,
                precision="float32", transposition="historical_C2_stale_track_pitch_summaries",
                model_selection="common_support_primary_macro_f1", test_locked=True)


def campaign_environment():
    """Normalize the trusted Torch version before strict checkpoint binding.

    ``torch.__version__`` is a TorchVersion str subclass in native runtimes.
    JSON silently accepts it, but the restricted full checkpoint deliberately
    permits only builtin metadata types. Keep that checkpoint boundary strict.
    """
    return {**ref.environment_fingerprint(), "torch": str(torch.__version__),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def components(device="cuda"):
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
    model = TemporalSliceModel().to(device).train()
    initial = model_state_fingerprint(model.base)
    random.seed(17001); np.random.seed(17001); torch.manual_seed(17001)
    if device == "cuda":
        torch.cuda.manual_seed_all(17001)
    optimizer, scheduler = build_full_orbit_optimizer_scheduler(model)
    scaler, _ = _scaler()
    return model, optimizer, scheduler, scaler, initial


def attempt_accelerated_update(*, model, graph, sidecars, shifts, class_weights,
                               optimizer, scheduler, scaler):
    """Same FP32 loss/update; coalesce scalar diagnostics as in accelerated C2."""
    optimizer.zero_grad(set_to_none=True)
    output = model(graph)
    alignment = align_temporal_targets(output, sidecars, shifts=shifts)
    loss = temporal_supervised_loss(output, alignment, class_weights)
    require(loss.total is not None and loss.fp32_boundary, "missing_fp32_loss")
    flags = _coalesced_host_scalars([*(torch.isfinite(v).all() for v in output.logits.values()),
                                   torch.isfinite(loss.total)])
    require(all(flags), "nonfinite_forward")
    before = float(scaler.get_scale())
    scaler.scale(loss.total).backward()
    scaler.unscale_(optimizer)
    finite, norm = performance_gradient_diagnostics(model)
    require(finite and math.isfinite(norm), "nonfinite_gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer); scaler.update()
    after = float(scaler.get_scale())
    applied = after >= before
    if applied:
        scheduler.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return ref.AppliedUpdateResult(applied, loss, norm, True, finite, before, after)


def state_digest(parts, rng):
    model, optimizer, scheduler, scaler = parts[:4]
    return {"model": model_state_fingerprint(model),
            "gradients": recursive_state_fingerprint({n: p.grad for n, p in model.named_parameters()}),
            "optimizer": recursive_state_fingerprint(optimizer.state_dict()),
            "scheduler": recursive_state_fingerprint(scheduler.state_dict()),
            "scaler": recursive_state_fingerprint(scaler.state_dict()),
            "rng": recursive_state_fingerprint(rng)}


def _update(parts, graph, sidecars, shifts, weights, *, accelerated):
    fn = attempt_accelerated_update if accelerated else attempt_temporal_update
    return fn(model=parts[0], optimizer=parts[1], scheduler=parts[2], scaler=parts[3],
              graph=graph, sidecars=sidecars, shifts=shifts, class_weights=weights)


def progress(update=0):
    return dict(applied_updates=update, draw_cursor=2*update, validation_history=[],
                best_score=None, best_update=None, pending_validation=None, elapsed_seconds=0.)


def verify_gate(data, manifest, out, bindings, *, device="cuda"):
    from .cache import read_temporal_record, collate_temporal_graphs, transpose_temporal_historical
    from .checkpoint import save_temporal_checkpoint, load_temporal_checkpoint
    jobs = gate_jobs(data.table)
    data.require_train_records({r for j in jobs for r in j.record_ids})
    pool = OrderedTemporalPool(jobs, manifest, source_fingerprint=bindings["source"])
    a, b = components(device), components(device)
    rng_a = capture_safe_rng(); rng_b = deepcopy(rng_a)
    weights = prepare_performance_class_weights(data.class_weights, device)
    rows = []
    try:
        for job in jobs:
            prepared = pool.peek()
            loaded = [read_temporal_record(data, rid, split="train") for rid in job.record_ids]
            reference = transpose_temporal_historical(collate_temporal_graphs([v[0] for v in loaded]),
                                                      job.applied_shifts)
            require(reference.fingerprint() == prepared.graph.fingerprint(), "gate_graph_mismatch")
            require(fingerprint([v[1] for v in loaded]) == fingerprint(list(prepared.sidecars)),
                    "gate_sidecar_mismatch")
            restore_safe_rng(rng_a, device_type=device)
            r = _update(a, reference.to(device), [v[1] for v in loaded], job.applied_shifts,
                        data.class_weights, accelerated=False)
            rng_a = capture_safe_rng()
            restore_safe_rng(rng_b, device_type=device)
            p = _update(b, prepared.graph.to(device), prepared.sidecars, job.applied_shifts,
                        weights, accelerated=True)
            rng_b = capture_safe_rng()
            require(r.applied and p.applied and float(r.loss.total) == float(p.loss.total)
                    and r.gradient_norm == p.gradient_norm, "gate_loss_or_gradient_mismatch")
            require(state_digest(a, rng_a) == state_digest(b, rng_b), "gate_state_mismatch")
            pool.commit(job.position, applied=True)
            rows.append(dict(update=job.position+1, shifts=job.applied_shifts,
                             loss=float(p.loss.total), gradient_norm=p.gradient_norm))
            emit(out, "equivalence", **rows[-1])
    finally:
        pool.close(); write_json(out / "coverage-queue.json", pool.diagnostics())
    del a, b, reference, prepared
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    # A separate actual orbit prefix checks interrupted continuation; coverage
    # jobs are not mislabeled as historical sampler positions.
    orbit_jobs = training_jobs(data.table, 0, 4)
    data.require_train_records({r for j in orbit_jobs for r in j.record_ids})
    pool = OrderedTemporalPool(orbit_jobs, manifest, source_fingerprint=bindings["source"])
    a, b = components(device), components(device)
    sa, sb = FullOrbitSampler(data.table, seed=17), FullOrbitSampler(data.table, seed=17)
    rng_a = capture_safe_rng(); rng_b = deepcopy(rng_a)
    checkpoint_pin = None
    try:
        for job in orbit_jobs:
            prepared = pool.peek()
            for parts, sampler, name in ((a, sa, "a"), (b, sb, "b")):
                restore_safe_rng(rng_a if name == "a" else rng_b, device_type=device)
                result = _update(parts, prepared.graph.to(device), prepared.sidecars,
                                 job.applied_shifts, weights, accelerated=True)
                require(result.applied, "resume_gate_update_skipped")
                sampler.advance_after_applied_update(2)
                if name == "a":
                    rng_a = capture_safe_rng()
                else:
                    rng_b = capture_safe_rng()
            require(state_digest(a, rng_a) == state_digest(b, rng_b), "resume_gate_state_mismatch")
            pool.commit(job.position, applied=True)
            if job.position == 1:
                checkpoint_path = out / "gate-resume-000002.pt"
                restore_safe_rng(rng_b, device_type=device)
                checkpoint_pin = save_temporal_checkpoint(checkpoint_path, model=b[0], optimizer=b[1],
                    scheduler=b[2], scaler=b[3], config=scientific_config(), bindings=bindings,
                    progress=progress(2), sampler_state=sb.state_dict())
                del b
                gc.collect()
                b = components(device)
                sb = FullOrbitSampler(data.table, seed=17)
                restored = load_temporal_checkpoint(checkpoint_path,
                    expected_sha256=checkpoint_pin["sha256"], expected_size_bytes=checkpoint_pin["size_bytes"],
                    model=b[0], optimizer=b[1], scheduler=b[2], scaler=b[3],
                    config=scientific_config(), bindings=bindings, sampler=sb, restore_rng=True)
                rng_b = capture_safe_rng()
                require(restored["progress"]["applied_updates"] == 2 and sb.position == 4,
                        "resume_gate_restored_position")
            emit(out, "resume_equivalence", update=job.position+1)
        require(sa.state_dict() == sb.state_dict(), "resume_sampler_mismatch")
    finally:
        pool.close(); write_json(out / "resume-queue.json", pool.diagnostics())
    return dict(schema_version=SCHEMA, passed=True, model_kind="temporal_slice_c2",
                bindings=bindings, scientific_config=scientific_config(), shift_coverage=list(range(12)),
                coverage_rows=rows, actual_optimizer_updates=32, actual_draws=64,
                resume_gate={"passed": True, "split_update": 2, "final_update": 4,
                             "checkpoint": checkpoint_pin, "sampler_position": sb.position},
                comparisons="exact graphs, sidecars, loss, gradients, model, optimizer, scheduler, scaler, RNG",
                workers=WORKERS, test_accessed=False, validation_accessed=False,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                scope="native temporal reference versus accelerated pipeline; not historical-runtime equivalence")


def common_alignment(alignment):
    return replace(alignment, heads={h: replace(rows, valid_mask=alignment.common_support_masks[h])
                                    for h, rows in alignment.heads.items()})


def metric_output(output, alignment):
    # Match the loss's empty-bank sentinel without exposing a fake candidate.
    logits = dict(output.logits)
    for task, values in logits.items():
        if values.shape[0] == 0:
            require(not alignment.heads[task].valid_mask.any(), "metric_target_without_candidate")
            logits[task] = values.new_zeros((1, values.shape[1]))
    return SimpleNamespace(logits=logits)


def validate(model, data, out, update, *, device="cuda"):
    from .cache import read_temporal_record
    accumulators = {scope: ref.CorrectedValidationAccumulator(data.class_weights,
        train_seen_tuples=set(data.train_seen_tuples)) for scope in ("common_support", "expanded_support")}
    require(len(data.validation_record_ids) == 162, "validation_population")
    was_training = model.training
    with preserve_caller_rng(), torch.no_grad():
        model.eval()
        try:
            for index, rid in enumerate(data.validation_record_ids, 1):
                graph, sidecar = read_temporal_record(data, rid, split="validation")
                output = model(graph.to(device))
                alignment = align_temporal_targets(output, (sidecar,), shifts=(0,))
                values = metric_output(output, alignment)
                accumulators["expanded_support"].update(values, alignment, sidecars=(sidecar,))
                accumulators["common_support"].update(values, common_alignment(alignment), sidecars=(sidecar,))
                if index % 40 == 0:
                    emit(out, "validation_progress", update=update, records=index)
        finally:
            model.train(was_training)
    result = {scope: {**acc.finalize(), "confusions": {h: v.tolist() for h, v in acc.confusions.items()}}
              for scope, acc in accumulators.items()}
    result.update(schema_version=SCHEMA, update=update, validation_records=162, test_accessed=False,
                  evaluated_model_state_fingerprint=model_state_fingerprint(model),
                  selection_metric="common_support.corrected_primary_macro_score")
    write_json(out / f"validation-{update:06d}.json", result)
    return result


def _checkpoint(out, label, parts, sampler, p, bindings):
    from .checkpoint import save_temporal_checkpoint
    path = out / f"checkpoint-{p['applied_updates']:06d}{label}.pt"
    pin = save_temporal_checkpoint(path, model=parts[0], optimizer=parts[1], scheduler=parts[2],
        scaler=parts[3], config=scientific_config(), bindings=bindings,
        progress=p, sampler_state=sampler.state_dict())
    value = dict(path=str(path.resolve()), **pin, update=p["applied_updates"])
    temporary = out / "latest-checkpoint.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(out / "latest-checkpoint.json")
    return value


def train(data, manifest, out, bindings, config, *, device="cuda"):
    from .checkpoint import load_temporal_checkpoint
    parts = components(device)
    sampler = FullOrbitSampler(data.table, seed=17)
    p = progress()
    if config.get("resume") is not None:
        pin = config["resume"]
        restored = load_temporal_checkpoint(pin["path"], expected_sha256=pin["sha256"],
            expected_size_bytes=pin["size_bytes"], model=parts[0], optimizer=parts[1],
            scheduler=parts[2], scaler=parts[3], config=scientific_config(), bindings=bindings,
            sampler=sampler, restore_rng=True)
        p = restored["progress"]
        write_json(out / "resume-verification.json", dict(restored_update=p["applied_updates"],
                   sampler_position=sampler.position, state_restoration_verified=True))
    start_update = p["applied_updates"]
    require(start_update < TOTAL, "already_complete")
    jobs = training_jobs(data.table, start_update, TOTAL)
    data.require_train_records({r for j in jobs for r in j.record_ids})
    pool = OrderedTemporalPool(jobs, manifest, source_fingerprint=bindings["source"])
    weights = prepare_performance_class_weights(data.class_weights, device)
    write_json(out / "initial.json", dict(base_state_fingerprint=parts[4], from_scratch=start_update == 0,
               full_state_fingerprint=model_state_fingerprint(parts[0]), scientific_config=scientific_config()))
    started, previous_elapsed = time.monotonic(), p["elapsed_seconds"]
    stop = {"requested": False}
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda *_: stop.update(requested=True))

    def validation_step(update):
        result = validate(parts[0], data, out, update, device=device)
        row = dict(update=update,
            primary_score=result["common_support"]["corrected_primary_macro_score"],
            expanded_primary_score=result["expanded_support"]["corrected_primary_macro_score"])
        p["validation_history"].append(row)
        require(row["primary_score"] is not None, "missing_validation_primary")
        improved = p["best_score"] is None or row["primary_score"] > p["best_score"]
        if improved:
            p.update(best_score=row["primary_score"], best_update=update)
        p["pending_validation"] = None
        p["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        pin = _checkpoint(out, "-validated", parts, sampler, p, bindings)
        if improved:
            write_json(out / f"best-{update:06d}.json", dict(checkpoint=pin, **row))
        emit(out, "validation", **row)
        gc.collect()

    try:
        if p["pending_validation"] is not None:
            validation_step(p["pending_validation"])
        for job in jobs:
            if stop["requested"]:
                break
            step_start = time.monotonic()
            prepared = pool.peek()
            draws = [sampler.peek(k) for k in range(2)]
            require(prepared.job == job and tuple(d.record_id for d in draws) == job.record_ids
                    and tuple(d.shift_pc for d in draws) == job.applied_shifts, "sampler_queue_mismatch")
            lr = parts[1].param_groups[0]["lr"]
            outcome = _update(parts, prepared.graph.to(device), prepared.sidecars,
                              job.applied_shifts, weights, accelerated=True)
            require(outcome.applied, "update_not_applied")
            sampler.advance_after_applied_update(2); pool.commit(job.position, applied=True)
            update = start_update + job.position + 1
            p.update(applied_updates=update, draw_cursor=2*update,
                     elapsed_seconds=previous_elapsed + time.monotonic()-started)
            row = dict(update=update, draws=[asdict(d) for d in draws], loss=float(outcome.loss.total),
                       lr=lr, seconds=time.monotonic()-step_start)
            with (out / "training_metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if update <= start_update+2 or update % 100 == 0:
                emit(out, "training", update=update, loss=row["loss"], elapsed_seconds=p["elapsed_seconds"])
            if update % 1000 == 0:
                p["pending_validation"] = update if update % 5000 == 0 else None
                _checkpoint(out, "", parts, sampler, p, bindings)
            if update % 5000 == 0:
                validation_step(update)
        complete = p["applied_updates"] == TOTAL
        if not complete:
            _checkpoint(out, "-interrupted", parts, sampler, p, bindings)
        return dict(schema_version=SCHEMA, completed=complete, interrupted=not complete,
            start_update=start_update, final_update=p["applied_updates"],
            applied_updates=p["applied_updates"]-start_update, cumulative_draws=p["draw_cursor"],
            best={"update": p["best_update"], "primary_score": p["best_score"]},
            validations=p["validation_history"], wall_seconds=time.monotonic()-started, test_accessed=False)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        pool.close(); write_json(out / "queue-final.json", pool.diagnostics())


def main(argv=None):
    import argparse
    from music_critic.experiments.registry import load_record
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    require(config.get("mode") in ("gate", "train"), "mode")
    record = load_record(Path(config["record"]))
    require(record["status"] == "planned" and record["experiment_id"] == config["experiment_id"],
            "planned_record_required")
    source_path = Path(config["source_manifest"])
    source = json.loads(source_path.read_text())
    for row in source["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "source_relative_path")
        require(sha256((Path(config["source_root"]) / relative).read_bytes()).hexdigest() == row["sha256"],
                "source_changed")
    data_path = Path(config["data_manifest"])
    require(sha256(data_path.read_bytes()).hexdigest() == config["data_manifest_sha256"], "data_manifest_changed")
    manifest = json.loads(data_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    device = config.get("device", "cuda")
    require(device == "cuda" and torch.cuda.is_available(), "native_cuda_required")
    environment = campaign_environment()
    write_json(args.output / "environment.json", environment)
    data = FrozenTrainingData(manifest)
    runtime_keys = ("python", "torch", "numpy", "packages", "torch_cuda_build", "cudnn_version",
                    "gpu", "cuda_device_count", "cublas_workspace_config", "cudnn_allow_tf32", "matmul_allow_tf32")
    bindings = dict(source=fingerprint(source), data=data.manifest_fingerprint,
                    environment={k: environment.get(k) for k in runtime_keys},
                    scientific_config=fingerprint(scientific_config()))
    write_json(args.output / "bindings.json", bindings)
    if config["mode"] == "train":
        gate_path = Path(config["gate_report"])
        require(sha256(gate_path.read_bytes()).hexdigest() == config["gate_sha256"], "gate_hash")
        gate = json.loads(gate_path.read_text())
        require(gate.get("passed") is True and gate.get("bindings") == bindings
                and gate.get("resume_gate", {}).get("passed") is True
                and gate.get("shift_coverage") == list(range(12)), "matching_completed_gate_required")
        write_json(args.output / "startup-data-verification.json", data.verify_bindings())
    emit(args.output, "preparing", mode=config["mode"])
    try:
        result = (verify_gate(data, manifest, args.output, bindings, device=device)
                  if config["mode"] == "gate" else train(data, manifest, args.output, bindings, config, device=device))
        write_json(args.output / "report.json", result)
        emit(args.output, "completed" if result.get("completed", True) else "interrupted", mode=config["mode"])
    except BaseException as exc:
        write_json(args.output / "failure.json", dict(exception=type(exc).__name__, reason=str(exc)))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
