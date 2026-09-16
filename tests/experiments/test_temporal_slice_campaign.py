from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import json
import pickle
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import corrected_training as old
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.training_rng import capture_safe_rng, restore_safe_rng
from music_critic.experiments.analysisgnn.temporal_slices import build_temporal_batch
from music_critic.experiments.analysisgnn.temporal_slices import campaign as c
from music_critic.experiments.analysisgnn.temporal_slices.cache import (
    collate_temporal_graphs, encode_temporal_graph, transpose_temporal_historical,
)
from music_critic.experiments.analysisgnn.temporal_slices.alignment import align_temporal_targets


@pytest.fixture
def fixture(monkeypatch):
    import music_critic.tasks as tasks
    pieces = []
    original = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        pieces.append(piece)
        return original(piece, *args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_multisource_sample", capture)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    _, sidecar = old.build_source_free_fixture()
    yield build_temporal_batch([pieces[0]]), sidecar, old.load_frozen_class_weights()
    torch.set_num_threads(threads)


def table():
    return tuple(FullOrbitPair(r, shift, "component-"+r) for r in ("a", "b") for shift in range(12))


def test_exact_full_orbit_suffix_and_fixed_scientific_configuration():
    jobs = c.training_jobs(table(), 100, 104)
    sampler = FullOrbitSampler(table(), seed=17, position=200)
    for job in jobs:
        draws = [sampler.peek(i) for i in range(2)]
        assert job.record_ids == tuple(d.record_id for d in draws)
        assert job.applied_shifts == tuple(d.shift_pc for d in draws)
        assert job.draw_ids == (sampler.position, sampler.position+1)
        sampler.advance_after_applied_update(2)
    config = c.scientific_config()
    assert config["total_updates"] == 80000
    assert config["scheduler_total_updates"] == 120000
    assert config["batch_size"] == 2 and config["workers"] == 4
    assert config["cache_bytes"] == 8*1024**3
    assert config["test_locked"] is True


@pytest.mark.parametrize("bounds", [(-1, 3), (3, 3), (0, 80001), (0., 3)])
def test_invalid_campaign_budget_fails(bounds):
    with pytest.raises(ValueError, match="budget"):
        c.training_jobs(table(), *bounds)


def test_shift_gate_covers_each_shift_in_each_slot():
    jobs = c.gate_jobs(table())
    assert {j.applied_shifts[0] for j in jobs} == set(range(12))
    assert {j.applied_shifts[1] for j in jobs} == set(range(12))
    with pytest.raises(ValueError, match="all_shift"):
        c.gate_jobs(table()[:12])


def test_accelerated_update_exact_loss_gradient_optimizer_and_rng(fixture):
    graph, sidecar, weights = fixture
    graph = collate_temporal_graphs([graph, graph])
    a, b = c.components("cpu"), c.components("cpu")
    rng = capture_safe_rng()
    r = c._update(a, graph, [sidecar, sidecar], (0, 0), weights, accelerated=False)
    ra = capture_safe_rng()
    restore_safe_rng(rng, device_type="cpu")
    p = c._update(b, graph, [sidecar, sidecar], (0, 0), weights, accelerated=True)
    rb = capture_safe_rng()
    assert r.applied and p.applied
    assert float(r.loss.total) == float(p.loss.total)
    assert r.gradient_norm == p.gradient_norm
    assert c.state_digest(a, ra) == c.state_digest(b, rb)


class ImmediateExecutor:
    def __init__(self):
        self.submissions = []
        self.closed = False
    def submit(self, fn, job):
        self.submissions.append(job)
        future = Future()
        try:
            future.set_result(fn(job))
        except BaseException as exc:
            future.set_exception(exc)
        return future
    def shutdown(self, **kwargs):
        self.closed = True


def payload_maker(fixture):
    graph, sidecar, _ = fixture
    graph = collate_temporal_graphs([graph, graph])
    def prepare(job):
        sidecars = tuple(dict(deepcopy(sidecar), record_id=rid) for rid in job.record_ids)
        return pickle.dumps(dict(job=asdict(job), source_fingerprint="fixture",
            graph=encode_temporal_graph(transpose_temporal_historical(graph, job.applied_shifts)),
            sidecars=sidecars, cache={"entries": 2}, pid=100), protocol=5)
    return prepare


def test_ordered_pool_bounded_commit_repeek_and_cleanup(fixture):
    executor = ImmediateExecutor()
    jobs = c.training_jobs(table(), 0, 8)
    pool = c.OrderedTemporalPool(jobs, {}, source_fingerprint="fixture",
        _executor=executor, _prepare=payload_maker(fixture))
    assert len(executor.submissions) == 5
    try:
        first = pool.peek()
        assert pool.peek() is first
        with pytest.raises(ValueError, match="commit"):
            pool.commit(0, applied=False)
        with pytest.raises(ValueError, match="commit"):
            pool.commit(1, applied=True)
        assert pool.committed == 0 and len(executor.submissions) == 5
        for job in jobs:
            assert pool.peek().job == job
            pool.commit(job.position, applied=True)
            assert len(pool.pending) <= 5
        assert pool.committed == 8 and pool.payload_max > 0
    finally:
        pool.close()
    assert executor.closed


def test_failed_worker_startup_closes_owned_executor():
    executor = ImmediateExecutor()
    def fail(*args):
        raise RuntimeError("worker startup failed")
    executor.submit = fail
    with pytest.raises(RuntimeError, match="startup"):
        c.OrderedTemporalPool(c.training_jobs(table(), 0, 2), {}, source_fingerprint="fixture",
                              _executor=executor)
    assert executor.closed


@pytest.mark.parametrize("change", ["source", "oversize", "record", "exception"])
def test_worker_failure_does_not_commit(fixture, change):
    prepare = payload_maker(fixture)
    def broken(job):
        if change == "exception":
            raise RuntimeError("fixture worker failed")
        if change == "oversize":
            return b"x" * (job.reserved_bytes + 1)
        row = pickle.loads(prepare(job))
        if change == "source":
            row["source_fingerprint"] = "changed"
        else:
            row["sidecars"][0]["record_id"] = "wrong"
        return pickle.dumps(row)
    job = c.training_jobs(table(), 0, 1)[0]
    if change == "oversize":
        job = replace(job, reserved_bytes=100)
    pool = c.OrderedTemporalPool([job], {}, source_fingerprint="fixture",
        _executor=ImmediateExecutor(), _prepare=broken)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            pool.peek()
        assert pool.committed == 0
    finally:
        pool.close()


def test_common_support_preserves_new_rows_in_expanded_metrics(fixture):
    graph, sidecar, weights = fixture
    sidecar = deepcopy(sidecar)
    onset = deepcopy(next(r for r in sidecar["entities"] if r["entity_type"] == "onset"))
    onset["canonical_entity_id"] = "cadence-at-note-release"
    onset["onset_qn"] = {"num": 2, "den": 1}
    sidecar["entities"].append(onset)
    output = c.components("cpu")[0](graph)
    alignment = align_temporal_targets(output, [sidecar])
    common = c.common_alignment(alignment)
    assert alignment.heads["cadence"].valid_mask.tolist() == [True, True]
    assert common.heads["cadence"].valid_mask.tolist() == [True, False]
    for h in old.ACTIVE_HEADS:
        assert torch.equal(common.heads[h].values, alignment.heads[h].values)
        assert torch.equal(common.heads[h].candidate_indices, alignment.heads[h].candidate_indices)
    ea, ca = old.CorrectedValidationAccumulator(weights), old.CorrectedValidationAccumulator(weights)
    ea.update(c.metric_output(output, alignment), alignment, sidecars=[sidecar])
    ca.update(c.metric_output(output, common), common, sidecars=[sidecar])
    assert ea.supports["cadence"] == 2 and ca.supports["cadence"] == 1
    assert all(ea.supports[h] == ca.supports[h] for h in old.ACTIVE_HEADS if h != "cadence")


def test_complete_fixture_gate_exercises_resume_and_all_shifts(fixture, monkeypatch, tmp_path):
    graph, sidecar, weights = fixture
    data = SimpleNamespace(table=table(), class_weights=weights,
                           require_train_records=lambda records: tuple(records))
    original_pool = c.OrderedTemporalPool
    prepare = payload_maker(fixture)
    def pool(jobs, manifest, *, source_fingerprint):
        return original_pool(jobs, manifest, source_fingerprint=source_fingerprint,
                             _executor=ImmediateExecutor(), _prepare=prepare)
    def reader(data, rid, *, split):
        assert split == "train"
        return graph, dict(deepcopy(sidecar), record_id=rid)
    monkeypatch.setattr(c, "OrderedTemporalPool", pool)
    monkeypatch.setattr("music_critic.experiments.analysisgnn.temporal_slices.cache.read_temporal_record", reader)
    result = c.verify_gate(data, {}, tmp_path, {"source": "fixture", "data": "fixture"}, device="cpu")
    assert result["passed"] is True
    assert result["shift_coverage"] == list(range(12))
    assert result["resume_gate"]["passed"] is True
    assert result["resume_gate"]["sampler_position"] == 8
    assert result["actual_optimizer_updates"] == 32
    assert result["test_accessed"] is False and result["validation_accessed"] is False


class SpawnFixtureCache:
    def __init__(self, graph, sidecar):
        self.graph, self.sidecar = graph, sidecar
    def prepare_batch(self, record_ids):
        return SimpleNamespace(graph=collate_temporal_graphs([self.graph, self.graph]),
            sidecars=tuple(dict(deepcopy(self.sidecar), record_id=rid) for rid in record_ids),
            cache_diagnostics={"entries": 1, "accounted_bytes": 0, "max_bytes": 1000000})


def spawn_fixture_init():
    """Source-free worker setup; production preparation/transport runs unchanged."""
    import music_critic.tasks as tasks
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pieces = []
    original = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        pieces.append(piece)
        return original(piece, *args, **kwargs)
    tasks.prepare_multisource_sample = capture
    try:
        _, sidecar = old.build_source_free_fixture()
    finally:
        tasks.prepare_multisource_sample = original
    c._worker_cache = SpawnFixtureCache(build_temporal_batch([pieces[0]]), sidecar)
    c._worker_source = "fixture"


def test_real_spawn_transport_cpu_only_ordered_and_rng_preserving(fixture):
    executor = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=spawn_fixture_init)
    jobs = c.training_jobs(table(), 0, 7)
    before = c._cpu_rng()
    pool = c.OrderedTemporalPool(jobs, {}, source_fingerprint="fixture", _executor=executor)
    try:
        for job in jobs:
            batch = pool.peek()
            assert batch.job == job and batch.graph.slice_features.device.type == "cpu"
            pool.commit(job.position, applied=True)
        assert pool.committed == 7
        assert pool.diagnostics()["max_serialized_payload_bytes"] > 0
    finally:
        pool.close()
    assert c._cpu_rng() == before
