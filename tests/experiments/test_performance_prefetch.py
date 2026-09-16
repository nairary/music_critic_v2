"""Synthetic CPU-only contracts; no real corpus, GPU, VALIDATION or TEST reads."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import struct
import time

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import performance_prefetch as pp


def _jobs(count=5, *, reservation=65536, start=24):
    # Repeated record IDs with different schedule shifts straddle a nominal
    # 26-draw orbit boundary. Neither an epoch nor record identity resets IDs.
    return tuple(pp.PrefetchJob(i, (start + 2*i, start + 2*i + 1),
        ("synthetic:a", "synthetic:a" if i % 2 else "synthetic:b"),
        (i % 12, (i + 5) % 12), (0, 0), reservation) for i in range(count))


class _Prepare:
    def __init__(self, *, delays=None, fail=None, abrupt=None, oversized=False, rng=False):
        self.delays, self.fail, self.abrupt = delays or {}, fail, abrupt
        self.oversized, self.rng = oversized, rng

    def __call__(self, job):
        if job.position == self.abrupt:
            os._exit(9)
        if job.position in self.delays:
            time.sleep(self.delays[job.position])
        if job.position == self.fail:
            raise ValueError("deliberate synthetic worker failure")
        if self.rng:
            random.random()
            np.random.random()
            torch.rand(1)
        if self.oversized:
            return b"x" * (job.reserved_bytes + 1)
        return {"job": job, "features": torch.arange(8, dtype=torch.float32).reshape(2, 4) + job.position,
                "numpy": np.arange(3, dtype=np.int64), "private_random": random.Random(job.position).random(),
                "labels": [job.position % 2, (job.position + 1) % 2]}


def _prepare_graph_fixture(job):
    from music_critic.experiments.analysisgnn import corrected_training as reference
    records = [reference.build_source_free_fixture() for _ in job.record_ids]
    raw = reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in records])
    return pp.PreparedCPUJob(job, reference.transpose_raw_graph_batch(raw, job.applied_shifts),
                             tuple(row[1] for row in records))


def _partial_frame_worker(connection, prepare, worker_id, seed, implementation_fingerprint, source_fingerprint):
    """Synthetic transport fault: valid header, incomplete body, then stop/exit."""
    inherited = {name: os.environ.get(name) for name in pp.NATIVE_THREAD_ENVIRONMENT}
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    connection.send(("ready", worker_id, pp._observe_worker_runtime(
        worker_id, implementation_fingerprint, source_fingerprint, inherited)))
    connection.recv()
    os.write(connection.fileno(), struct.pack("!i", 1_000_000))
    os.write(connection.fileno(), b"incomplete")
    if prepare.abrupt == 0:
        os._exit(9)
    time.sleep(10)


class _AcknowledgmentFault(_Prepare):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault


class _FaultConnection:
    def __init__(self, connection, fault):
        self.connection, self.fault = connection, fault

    def send(self, message):
        if message[0] == "ready":
            kind, worker_id, payload = message
            if self.fault == "missing_one_worker" and worker_id == 1:
                time.sleep(10)
            elif self.fault == "missing":
                payload = None
            elif self.fault == "wrong_worker":
                worker_id += 1
            elif self.fault == "wrong_pid":
                payload["pid"] += 1
            elif self.fault == "wrong_source":
                payload["source_fingerprint"] += "changed"
            elif self.fault == "threads":
                payload["torch_num_interop_threads"] = 2
            elif self.fault == "cuda":
                payload["cuda_initialized"] = True
            elif self.fault == "inherited":
                payload["inherited_native_thread_environment"]["NUMEXPR_NUM_THREADS"] = "99"
            elif self.fault == "duplicate":
                self.connection.send(message)
            message = kind, worker_id, payload
        self.connection.send(message)

    def recv(self):
        return self.connection.recv()

    def close(self):
        self.connection.close()


def _acknowledgment_fault_worker(connection, prepare, worker_id, seed, implementation_fingerprint, source_fingerprint):
    # Spawn imports the original module afresh. Only this transport wrapper
    # mutates a real worker's actual observation, without touching job results.
    pp._worker_main(_FaultConnection(connection, prepare.fault), prepare, worker_id, seed,
                    implementation_fingerprint, source_fingerprint)


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def _config(workers, **kwargs):
    return pp.PrefetchConfig(workers=workers, close_timeout_seconds=0.4, **kwargs)


@pytest.mark.parametrize("workers", (0, 1, 2, 4))
def test_ordered_exact_outputs_commits_parent_rng_and_settings(workers, monkeypatch):
    # Includes values that a child setter cannot retroactively prove applied
    # before imports. Synthetic execution reports them instead of hiding them.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    jobs = _jobs()
    expected = [_Prepare()(job) for job in jobs]
    before, threads = _rng(), (torch.get_num_threads(), torch.get_num_interop_threads())
    environment = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", *pp.NATIVE_THREAD_ENVIRONMENT)}
    config = _config(workers, max_pending_batches=3, buffer_byte_cap=3 * jobs[0].reserved_bytes)
    pool = pp.OrderedPrefetch(jobs, _Prepare(delays={0: 0.15}), source_fingerprint="synthetic-source-v1", synthetic=True, config=config)
    try:
        _equal(before, _rng())
        evidence = pool.diagnostics()["worker_runtime"]
        assert evidence["observations_complete"]
        assert evidence["started_workers"] == evidence["required_workers"] == workers
        assert evidence["native_thread_limits_all_one"] == (False if workers else None)
        assert [row["worker_id"] for row in evidence["observations"]] == list(range(workers))
        assert [row["pid"] for row in evidence["observations"]] == list(pool.diagnostics()["worker_pids"])
        for observation in evidence["observations"]:
            assert observation["torch_num_threads"] == observation["torch_num_interop_threads"] == 1
            assert observation["cuda_initialized"] is False
            assert observation["runtime_versions"] == pp._runtime_versions()
            assert observation["source_fingerprint"] == "synthetic-source-v1"
            assert observation["implementation_fingerprint"] == pool.state_dict()["implementation_fingerprint"]
            assert observation["inherited_native_thread_environment"] == {key: environment[key] for key in pp.NATIVE_THREAD_ENVIRONMENT}
            assert observation["native_thread_environment"] == {**observation["inherited_native_thread_environment"], "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        if workers:
            evidence["observations"][0]["native_thread_environment"]["OMP_NUM_THREADS"] = "999"
            evidence["spawn_parent_native_thread_environment"]["OPENBLAS_NUM_THREADS"] = "999"
            assert pool.worker_runtime_evidence()["observations"][0]["native_thread_environment"]["OMP_NUM_THREADS"] == "1"
            assert pool.worker_runtime_evidence()["spawn_parent_native_thread_environment"]["OPENBLAS_NUM_THREADS"] == "2"
        assert "worker_runtime" not in pool.diagnostics(include_worker_runtime=False)
        assert "worker_runtime" not in json.dumps(pool.state_dict())
        with pytest.raises(pp.PrefetchError, match="unobserved_commit"):
            pool.commit(0, applied=True)
        for i, wanted in enumerate(expected):
            output = pool.peek()
            _equal(output, wanted)
            assert pool.state_dict()["committed_batch_cursor"] == i
            assert pool.state_dict()["committed_draw_cursor"] == 24 + 2*i
            with pytest.raises(pp.PrefetchError, match="out_of_order"):
                pool.commit(i + 1, applied=True)
            output["features"].fill_(900)
            output["labels"].clear()
            assert pool.commit(i, applied=False) is False
            _equal(pool.peek(), wanted)
            stats = pool.diagnostics()
            assert stats["pending_batches"] <= 3
            assert stats["reserved_bytes"] <= config.buffer_byte_cap
            assert stats["serialized_ready_bytes"] <= stats["reserved_bytes"]
            assert pool.commit(i, applied=True)
            with pytest.raises(pp.PrefetchError, match="out_of_order"):
                pool.commit(i, applied=True)
        with pytest.raises(StopIteration):
            pool.peek()
        assert pool.diagnostics()["reserved_bytes"] == 0
    finally:
        pool.close()
    assert all(slot.process.exitcode is not None and not slot.receiver.is_alive() for slot in pool._slots)
    _equal(before, _rng())
    assert threads == (torch.get_num_threads(), torch.get_num_interop_threads())
    assert environment == {key: os.environ.get(key) for key in environment}


@pytest.mark.parametrize("fault", ("missing", "wrong_worker", "wrong_pid", "wrong_source", "threads", "cuda", "inherited", "duplicate"))
def test_invalid_ready_acknowledgment_fails_without_commit_and_cleans_own_pool(monkeypatch, fault):
    monkeypatch.setattr(pp, "_worker_main", _acknowledgment_fault_worker)
    closed = []
    original_close = pp.OrderedPrefetch.close
    def capture_close(self):
        original_close(self)
        closed.append(self)
    monkeypatch.setattr(pp.OrderedPrefetch, "close", capture_close)
    before = _rng()
    with pytest.raises(pp.PrefetchError, match="worker_runtime|unexpected_worker_message"):
        with pp.OrderedPrefetch(_jobs(1), _AcknowledgmentFault(fault), source_fingerprint="synthetic-ack-failure",
             synthetic=True, config=_config(2)) as stream:
            stream.peek()  # a duplicate ready may arrive just after constructor startup
    assert closed
    for stream in closed:
        assert stream.state_dict()["committed_batch_cursor"] == 0
        assert stream.diagnostics()["reserved_bytes"] == 0
        assert all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in stream._slots)
    _equal(before, _rng())


def test_every_worker_must_acknowledge_before_any_job_is_dispatched(monkeypatch):
    monkeypatch.setattr(pp, "_worker_main", _acknowledgment_fault_worker)
    closed = []
    original_close = pp.OrderedPrefetch.close
    def capture_close(self):
        original_close(self)
        closed.append(self)
    monkeypatch.setattr(pp.OrderedPrefetch, "close", capture_close)
    before = _rng()
    with pytest.raises(pp.PrefetchError, match="startup_timeout"):
        pp.OrderedPrefetch(_jobs(2), _AcknowledgmentFault("missing_one_worker"),
            source_fingerprint="synthetic-partial-startup", synthetic=True,
            config=_config(2, startup_timeout_seconds=3.0))
    assert closed
    stream = closed[-1]
    assert stream.diagnostics()["submitted_batch_cursor"] == 0
    assert stream.state_dict()["committed_batch_cursor"] == 0
    assert not stream.worker_runtime_evidence()["observations_complete"]
    assert all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in stream._slots)
    _equal(before, _rng())


@pytest.mark.parametrize("field,value", (
    ("worker_id", True), ("pid", True), ("torch_num_threads", True), ("torch_num_interop_threads", 1.0),
    ("cuda_initialized", 0), ("schema", "old"), ("implementation_fingerprint", "changed"),
    ("runtime_versions", {"torch": "other"}), ("native_thread_environment", {}),
    ("inherited_native_thread_environment", {}), ("extra_field", 1),
))
def test_runtime_acknowledgment_rejects_malformed_typed_fields(field, value):
    inherited = {name: "1" for name in pp.NATIVE_THREAD_ENVIRONMENT}
    observation = {"schema": pp.WORKER_RUNTIME_SCHEMA, "worker_id": 0, "pid": 123,
        "implementation_fingerprint": "implementation", "source_fingerprint": "source",
        "runtime_versions": pp._runtime_versions(), "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "cuda_initialized": False, "native_thread_environment": dict(inherited),
        "inherited_native_thread_environment": dict(inherited)}
    args = dict(worker_id=0, pid=123, implementation_fingerprint="implementation", source_fingerprint="source",
                runtime_versions=pp._runtime_versions(), inherited_environment=inherited)
    pp._validate_worker_runtime(observation, **args)
    observation[field] = value
    with pytest.raises(pp.PrefetchError, match="worker_runtime"):
        pp._validate_worker_runtime(observation, **args)


@pytest.mark.parametrize("name", pp.NATIVE_THREAD_ENVIRONMENT)
@pytest.mark.parametrize("value", (None, "2"))
def test_production_requires_native_limits_before_any_source_or_spawn(monkeypatch, name, value):
    for key in pp.NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(key, "1")
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    def forbidden(*args, **kwargs):
        pytest.fail("native environment gate reached source or process startup")
    monkeypatch.setattr(pp.WarmTrainBatchPreparer, "validate_jobs", forbidden)
    monkeypatch.setattr(mp, "get_context", forbidden)
    preparer = object.__new__(pp.WarmTrainBatchPreparer)
    preparer.source_fingerprint = "synthetic-concrete-production-guard"
    with pytest.raises(pp.PrefetchError, match="native_limits_required_before_spawn"):
        pp.OrderedPrefetch(_jobs(1), preparer, source_fingerprint=preparer.source_fingerprint, config=_config(1))


def test_complete_resume_has_no_new_workers_or_fabricated_acknowledgments(monkeypatch):
    for name in pp.NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    jobs, config = _jobs(1), _config(1)
    with pp.OrderedPrefetch(jobs, _Prepare(), source_fingerprint="source", synthetic=True, config=config) as original:
        assert original.worker_runtime_evidence()["native_thread_limits_all_one"] is True
        original.peek()
        original.commit(0, applied=True)
        state = original.state_dict()
    def forbidden(*args, **kwargs):
        pytest.fail("complete resume started a worker")
    monkeypatch.setattr(mp, "get_context", forbidden)
    with pp.OrderedPrefetch(jobs, _Prepare(), source_fingerprint="source", synthetic=True,
                          config=config, checkpoint=state) as complete:
        evidence = complete.worker_runtime_evidence()
        assert evidence["configured_workers"] == 1
        assert evidence["required_workers"] == evidence["started_workers"] == 0
        assert evidence["observations"] == [] and evidence["observations_complete"]
        assert evidence["native_thread_limits_all_one"] is None


def test_global_reservations_include_ready_and_delivered_until_commit():
    jobs = _jobs(4, reservation=8192)
    with pp.OrderedPrefetch(jobs, _Prepare(delays={0: 0.3}), source_fingerprint="synthetic", synthetic=True,
         config=_config(2, max_pending_batches=4, buffer_byte_cap=16384)) as pool:
        assert pool.diagnostics()["submitted_batch_cursor"] == 2
        pool.peek()
        stats = pool.diagnostics()
        assert stats["submitted_batch_cursor"] == 2
        assert stats["pending_batches"] == 2 and stats["reserved_bytes"] == 16384
        assert stats["ready_batches"] == 2  # later job completed while zero was delayed
        assert "not_RSS" in stats["buffer_scope"]
        pool.commit(0, applied=False)
        assert pool.diagnostics()["reserved_bytes"] == 16384
        pool.commit(0, applied=True)
        assert pool.diagnostics()["reserved_bytes"] == 8192
        assert pool.peek()["job"].position == 1
        assert pool.diagnostics()["submitted_batch_cursor"] == 3
        assert pool.diagnostics()["high_water_reserved_bytes"] == 16384


def test_checkpoint_discards_speculation_and_rebuilds_exact_remaining_jobs():
    jobs, config = _jobs(), _config(2)
    pool = pp.OrderedPrefetch(jobs, _Prepare(), source_fingerprint="source", synthetic=True, config=config)
    assert pool.peek()["job"] == jobs[0]
    pool.commit(0, applied=True)
    state = pool.state_dict()
    assert pool.diagnostics()["submitted_batch_cursor"] > state["committed_batch_cursor"]
    pool.close()
    assert "pending" not in json.dumps(state)
    with pp.OrderedPrefetch(jobs, _Prepare(), source_fingerprint="source", synthetic=True, config=config, checkpoint=state) as resumed:
        for job in jobs[1:]:
            assert resumed.peek()["job"] == job
            resumed.commit(job.position, applied=True)
        assert resumed.state_dict()["committed_draw_cursor"] == 34
    for overrides in ({"source_fingerprint": "changed"}, {"config": _config(1)},
                      {"jobs": (replace(jobs[0], schedule_shifts=(1, 2)), *jobs[1:])},
                      {"jobs": (replace(jobs[0], reserved_bytes=65535), *jobs[1:])},
                      {"checkpoint": {**state, "committed_draw_cursor": 27}}):
        args = dict(jobs=jobs, prepare=_Prepare(), source_fingerprint="source", synthetic=True, config=config, checkpoint=state)
        args.update(overrides)
        with pytest.raises(ValueError, match="checkpoint_binding"):
            pp.OrderedPrefetch(**args)


def test_real_sampler_peek_is_pure_across_odd_orbit_boundary_and_failed_update():
    from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair
    from music_critic.experiments.analysisgnn.orbit_matched_control import OrbitMatchedControlSampler
    table = (FullOrbitPair("a", 0, "component-a"), FullOrbitPair("a", 5, "component-a"),
             FullOrbitPair("b", 0, "component-b"))
    sampler = OrbitMatchedControlSampler(table, position=2)
    snapshot, rng = sampler.state_dict(), _rng()
    draws = tuple(sampler.peek(i) for i in range(8))
    assert draws[0].orbit_epoch == 0 and draws[1].orbit_epoch == 1
    jobs = tuple(pp.PrefetchJob(i, (2+2*i, 3+2*i),
        tuple(draw.record_id for draw in draws[2*i:2*i+2]),
        tuple(draw.schedule_shift_pc for draw in draws[2*i:2*i+2]),
        tuple(draw.applied_shift_pc for draw in draws[2*i:2*i+2]), 8192) for i in range(4))
    assert sampler.state_dict() == snapshot
    with pp.OrderedPrefetch(jobs, _Prepare(), source_fingerprint=snapshot["fingerprint"], synthetic=True, config=_config(2)) as pool:
        for job in jobs:
            assert pool.peek()["job"] == job
            # Simulated unapplied optimizer attempt consumes model RNG. Retry
            # must preserve that RNG advancement and retain the same draw.
            torch.rand(1)
            after_attempt = _rng()
            pool.commit(job.position, applied=False)
            assert pool.peek()["job"] == job
            _equal(after_attempt, _rng())
            assert sampler.position == job.draw_ids[0]
            pool.commit(job.position, applied=True)
            sampler.advance_after_applied_update()
            assert pool.state_dict()["committed_draw_cursor"] == sampler.position
    assert sampler.position == 10
    assert not torch.equal(rng[2], _rng()[2])


@pytest.mark.parametrize("workers", (0, 2))
def test_result_failure_or_oversize_never_commits_and_closes(workers):
    for prepare, reason in ((_Prepare(fail=0), "failure"), (_Prepare(oversized=True), "exceeds_reservation")):
        pool = pp.OrderedPrefetch(_jobs(2, reservation=8192), prepare, source_fingerprint="synthetic", synthetic=True, config=_config(workers))
        with pytest.raises(pp.PrefetchError, match=reason):
            pool.peek()
        assert pool.state_dict()["committed_batch_cursor"] == 0
        assert pool.diagnostics()["closed"] and pool.diagnostics()["reserved_bytes"] == 0
        assert all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in pool._slots)


def test_abrupt_exit_and_hung_worker_fail_promptly_without_skipping():
    for prepare, reason in ((_Prepare(abrupt=0), "worker_(exited|channel_failed)"),
                            (_Prepare(delays={0: 10}), "worker_timeout")):
        pool = pp.OrderedPrefetch(_jobs(2), prepare, source_fingerprint="synthetic", synthetic=True,
                                 config=_config(1, job_timeout_seconds=0.2))
        start = time.monotonic()
        with pytest.raises(pp.PrefetchError, match=reason):
            pool.peek()
        assert time.monotonic() - start < 3
        assert pool.state_dict()["committed_batch_cursor"] == 0
        assert all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in pool._slots)


def test_partial_frame_cannot_block_watchdog_or_owned_receiver_cleanup(monkeypatch):
    monkeypatch.setattr(pp, "_worker_main", _partial_frame_worker)
    for prepare, reason in ((_Prepare(), "worker_timeout"),
                            (_Prepare(abrupt=0), "worker_(exited|channel_failed)")):
        pool = pp.OrderedPrefetch(_jobs(1, reservation=2_000_000), prepare,
            source_fingerprint="synthetic-partial-frame", synthetic=True,
            config=_config(1, job_timeout_seconds=0.2))
        start = time.monotonic()
        with pytest.raises(pp.PrefetchError, match=reason):
            pool.peek()
        assert time.monotonic() - start < 3
        assert pool.state_dict()["committed_batch_cursor"] == 0
        assert all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in pool._slots)


def test_impossible_reservation_invalid_jobs_and_unbound_source_fail_before_io(monkeypatch):
    for name in pp.NATIVE_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    def forbidden(*args, **kwargs):
        pytest.fail("pre-dispatch failure touched process or source")
    monkeypatch.setattr(mp, "get_context", forbidden)
    with pytest.raises(ValueError, match="synthetic_opt_in_required"):
        pp.OrderedPrefetch(_jobs(), forbidden, source_fingerprint="untrusted", config=_config(2))
    with pytest.raises(ValueError, match="global_cap"):
        pp.OrderedPrefetch(_jobs(reservation=100), _Prepare(), source_fingerprint="source", synthetic=True,
                          config=_config(2, buffer_byte_cap=99))
    for kwargs in ({"draw_ids": (1, 2)}, {"draw_ids": (2, 4)}, {"record_ids": ["a", "b"]},
                   {"reserved_bytes": 0}, {"schedule_shifts": (-1, 0)}):
        with pytest.raises(ValueError):
            replace(_jobs(1)[0], **kwargs)
    from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
    class UnsealedSubclass(pp.WarmTrainBatchPreparer):
        def validate_jobs(self, jobs):
            forbidden()
    subclass = UnsealedSubclass(ProductionArtifactPaths(), {"record_bindings": {}})
    with pytest.raises(ValueError, match="synthetic_opt_in_required"):
        pp.OrderedPrefetch(_jobs(), subclass, source_fingerprint=subclass.source_fingerprint, config=_config(2))
    preparer = pp.WarmTrainBatchPreparer(ProductionArtifactPaths(), {"record_bindings": {}})
    monkeypatch.setattr(preparer, "_new_loader", forbidden)
    with pytest.raises(ValueError, match="production_synthetic_mode_mismatch"):
        pp.OrderedPrefetch(_jobs(), preparer, source_fingerprint=preparer.source_fingerprint,
                          synthetic=True, config=_config(2))
    with pytest.raises(ValueError, match="unbound_scheduled_record"):
        pp.OrderedPrefetch(_jobs(), preparer, source_fingerprint=preparer.source_fingerprint, config=_config(2))
    with pytest.raises(ValueError, match="identity_only"):
        preparer.validate_jobs((replace(_jobs(1)[0], applied_shifts=(1, 0)),))


def test_global_rng_using_callable_rejected_and_sync_parent_state_repaired():
    before = _rng()
    pool = pp.OrderedPrefetch(_jobs(1), _Prepare(rng=True), source_fingerprint="synthetic", synthetic=True, config=_config(0))
    with pytest.raises(pp.PrefetchError, match="consumed_global_rng"):
        pool.peek()
    _equal(before, _rng())
    assert pool.state_dict()["committed_batch_cursor"] == 0


def test_source_changed_after_import_rejected_before_spawn(monkeypatch):
    monkeypatch.setattr(pp, "_IMPORTED_IMPLEMENTATION_FINGERPRINT", "changed")
    def forbidden(*args, **kwargs):
        pytest.fail("changed code must not spawn")
    monkeypatch.setattr(mp, "get_context", forbidden)
    with pytest.raises(pp.PrefetchError, match="implementation_changed_after_import"):
        pp.OrderedPrefetch(_jobs(), _Prepare(), source_fingerprint="source", synthetic=True, config=_config(2))


def _train_synthetic(workers):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Dropout(0.25), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    losses = []
    with pp.OrderedPrefetch(_jobs(4), _Prepare(), source_fingerprint="synthetic", synthetic=True, config=_config(workers)) as pool:
        for i in range(4):
            payload = pool.peek()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(payload["features"]), torch.tensor(payload["labels"]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pool.commit(i, applied=True)
            losses.append(loss.item())
        cursor = pool.state_dict()["committed_draw_cursor"]
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), scheduler.state_dict(), losses, _rng(), cursor


def test_full_cpu_training_state_loss_and_dropout_rng_are_exact_across_workers():
    _equal(_train_synthetic(0), _train_synthetic(2))


def test_pyg_b2_graph_and_complete_sidecars_survive_spawn_ipc_exactly():
    jobs = _jobs(2, reservation=2_000_000)
    expected = [_prepare_graph_fixture(job) for job in jobs]
    before = _rng()
    with pp.OrderedPrefetch(jobs, _prepare_graph_fixture, source_fingerprint="synthetic-graph", synthetic=True,
                           config=_config(2)) as pool:
        for wanted in expected:
            got = pool.peek()
            assert got.job == wanted.job
            _equal(got.raw_graph_batch.to_dict(), wanted.raw_graph_batch.to_dict())
            _equal(got.raw_graph_batch._slice_dict, wanted.raw_graph_batch._slice_dict)
            _equal(got.raw_graph_batch._inc_dict, wanted.raw_graph_batch._inc_dict)
            _equal(got.sidecars, wanted.sidecars)
            pool.commit(got.job.position, applied=True)
    _equal(before, _rng())


def _warm_fixture(tmp_path, monkeypatch):
    from music_critic.data import dumps_piece, load_piece
    from music_critic.experiments.analysisgnn import corrected_training as reference
    from music_critic.experiments.analysisgnn import performance_data as pd
    root = tmp_path / "splits"
    root.mkdir()
    rows = []
    # Locked IDs exist only as synthetic split metadata; no locked targets.
    for split, count, components in (("train", 1295, 1209), ("validation", 162, 147), ("test", 162, 151)):
        rows.extend({"record_id": f"synthetic:{split}:{i}", "split": split,
                     "source_component_id": f"component:{split}:{i % components}"} for i in range(count))
    data = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (root / "split_assignments.jsonl").write_bytes(data)
    for module in (pd, reference):
        monkeypatch.setattr(module, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", sha256(data).hexdigest())
    piece = load_piece(Path(__file__).resolve().parents[1] / "fixtures/data/canonical_piece_v2.json")
    piece = replace(piece, split="train", annotations=(), targets=())
    record_id = "synthetic:train:0"
    paths = reference.ProductionArtifactPaths(b3_root=root, cache_root=tmp_path / "warm")
    record_root = paths.cache_root / sha256(record_id.encode()).hexdigest()[:24]
    record_root.mkdir(parents=True)
    piece_bytes = dumps_piece(piece).encode()
    sidecar = {"record_id": record_id, "entities": [], "relations": [], "source": "synthetic", "nested": {"keep": [0, None, "C♯"]}}
    sidecar_bytes = json.dumps(sidecar).encode()
    (record_root / "piece.json").write_bytes(piece_bytes)
    (record_root / "target_sidecar.json").write_bytes(sidecar_bytes)
    (record_root / "metadata.json").write_text(json.dumps({"record_id": record_id, "split": "train",
        "piece_sha256": sha256(piece_bytes).hexdigest(), "sidecar_sha256": sha256(sidecar_bytes).hexdigest()}))
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    loader.check_binding(record_id)
    return paths, record_id, record_root, loader.validate_all_bindings()


def test_original_warm_production_route_preserves_graph_sidecar_and_sealed_source(tmp_path, monkeypatch):
    from music_critic.experiments.analysisgnn import corrected_training as reference
    from music_critic.experiments.analysisgnn import performance_data as pd
    paths, record_id, record_root, manifest = _warm_fixture(tmp_path, monkeypatch)
    job = replace(_jobs(1)[0], record_ids=(record_id, record_id), reserved_bytes=2_000_000)
    original = reference.load_production_record(record_id, split="train", paths=paths)
    expected = reference.transpose_raw_graph_batch(reference.combine_single_record_raw_batches(
        [original[0].raw_graph_batch, original[0].raw_graph_batch]), (0, 0))
    calls = []
    original_load = reference.load_production_record
    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(reference, "load_production_record", observed)
    candidate = pp.WarmTrainBatchPreparer(paths, manifest)
    measured_size = len(pp._prepare_bytes(candidate, job))
    calls.clear()
    reservations = pp.make_reservation_manifest(source_fingerprint=candidate.base_source_fingerprint,
        evidence_id="synthetic-fixture-only", measured_serialized_bytes={pp.reservation_job_fingerprint(job): measured_size})
    preparer = pp.WarmTrainBatchPreparer(paths, manifest, reservation_manifest=reservations)
    assert pp.reservation_job_fingerprint(job) == pp.reservation_job_fingerprint(replace(job, reserved_bytes=1))
    with pytest.raises(ValueError, match="insufficient_job_reservation"):
        pp.OrderedPrefetch((replace(job, reserved_bytes=1),), preparer,
                          source_fingerprint=preparer.source_fingerprint)
    with pytest.raises(ValueError, match="reservation_manifest_required"):
        pp.OrderedPrefetch((job,), candidate, source_fingerprint=candidate.source_fingerprint)
    undersized = pp.make_reservation_manifest(source_fingerprint=candidate.base_source_fingerprint,
        evidence_id="synthetic-fixture-only", measured_serialized_bytes={pp.reservation_job_fingerprint(job): job.reserved_bytes + 1})
    invalid = pp.WarmTrainBatchPreparer(paths, manifest, reservation_manifest=undersized)
    with pytest.raises(ValueError, match="insufficient_job_reservation"):
        pp.OrderedPrefetch((job,), invalid, source_fingerprint=invalid.source_fingerprint)
    with pytest.raises(ValueError, match="reservation_manifest_binding_mismatch"):
        pp.WarmTrainBatchPreparer(paths, manifest, reservation_manifest={**reservations, "source_fingerprint": "changed"})
    before = _rng()
    with pp.OrderedPrefetch((job,), preparer, source_fingerprint=preparer.source_fingerprint) as pool:
        got = pool.peek()
        _equal(got.raw_graph_batch.to_dict(), expected.to_dict())
        _equal(got.raw_graph_batch._slice_dict, expected._slice_dict)
        _equal(got.raw_graph_batch._inc_dict, expected._inc_dict)
        _equal(got.sidecars, (original[1], original[1]))
        assert len(calls) == 2 and all(row[1]["split"] == "train" for row in calls)
    _equal(before, _rng())
    changed = json.loads((record_root / "target_sidecar.json").read_text())
    changed["nested"] = {"changed": True}
    payload = json.dumps(changed).encode()
    (record_root / "target_sidecar.json").write_bytes(payload)
    meta = json.loads((record_root / "metadata.json").read_text())
    meta["sidecar_sha256"] = sha256(payload).hexdigest()
    (record_root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(pd.PerformanceDataError, match="resume_binding_mismatch"):
        pp.OrderedPrefetch((job,), pp.WarmTrainBatchPreparer(paths, manifest, reservation_manifest=reservations),
                          source_fingerprint=preparer.source_fingerprint)
