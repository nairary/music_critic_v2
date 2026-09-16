"""Source-free compact/preparation/pool integration; no corpus or GPU reads."""

from collections import Counter
from copy import deepcopy
from dataclasses import fields
import json
import os
import pickle
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn import performance_data as pd
from music_critic.experiments.analysisgnn import performance_prefetch as pp


def _policy(total=10_000_000, processes=1):
    return {"schema": pp.CACHE_POLICY_SCHEMA, "global_max_bytes": total,
            "preparing_processes": processes}


def _job(position=0, ids=("a", "b"), shifts=(0, 0)):
    return pp.PrefetchJob(position, (position * 2, position * 2 + 1), ids,
                          (position % 12, (position + 5) % 12), shifts, 10_000_000)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _equal(first, second)
    else:
        assert left == right


def _equal_job(left, right):
    assert left.job == right.job and left.sidecars == right.sidecars
    for attr in ("to_dict", "_slice_dict", "_inc_dict"):
        a, b = getattr(left.raw_graph_batch, attr), getattr(right.raw_graph_batch, attr)
        _equal(a() if callable(a) else a, b() if callable(b) else b)


@pytest.fixture
def concrete_source(monkeypatch):
    """Inject only bytes/graphs, retaining the exact concrete compact loader."""
    batch, sidecar = reference.build_source_free_fixture()
    raw = batch.raw_graph_batch
    calls, original_calls, binding_calls = Counter(), Counter(), Counter()
    records = {}
    for number, record_id in enumerate(("a", "b", "c")):
        singleton = reference.transpose_raw_graph_batch(raw, (number,))
        target = deepcopy(sidecar)
        target["record_id"] = record_id
        records[record_id] = singleton, target
    manifest = {"record_bindings": {record_id: "seal-" + record_id for record_id in records}}

    def new_loader(self):
        loader = object.__new__(pd.FrozenProductionRecordLoader)
        loader.source_seal = "explicit-source-free-concrete-loader"
        return loader

    def load(self, record_id, *, split):
        assert split == "train" and record_id in records
        calls[record_id] += 1
        singleton, target = records[record_id]
        # The production _get_for_cache retains these exact JSON bytes, not
        # an invented serialization of a previously decoded cache entry.
        return pd.PreparedRecord(singleton.to_data_list()[0], deepcopy(target), "seal-" + record_id), json.dumps(target).encode()

    def binding(self, record_id, *, split="train"):
        assert split == "train" and record_id in records
        binding_calls[record_id] += 1
        return "seal-" + record_id

    def original(record_id, *, split, paths):
        assert split == "train" and record_id in records
        original_calls[record_id] += 1
        singleton, target = records[record_id]
        return SimpleNamespace(raw_graph_batch=deepcopy(singleton)), deepcopy(target)

    monkeypatch.setattr(pp.WarmTrainBatchPreparer, "_new_loader", new_loader)
    monkeypatch.setattr(pd.FrozenProductionRecordLoader, "_load", load)
    monkeypatch.setattr(pd.FrozenProductionRecordLoader, "check_binding", binding)
    monkeypatch.setattr(reference, "load_production_record", original)
    return manifest, calls, original_calls, binding_calls


def _preparer(source, policy=None, **kwargs):
    return pp.WarmTrainBatchPreparer(reference.ProductionArtifactPaths(), source[0],
                                     cache_policy=policy, **kwargs)


@pytest.mark.parametrize("bad", [{}, {"schema": "other"}, _policy(True), _policy(0),
    _policy(1, 4), _policy(10, True), _policy(10, 3), {**_policy(), "extra": 1}])
def test_policy_rejects_invalid_or_implicit_budgets_before_loader(concrete_source, bad):
    with pytest.raises(ValueError, match="invalid_cache"):
        _preparer(concrete_source, bad)
    assert not concrete_source[1]


def test_default_remains_original_and_keeps_old_binding_shape(concrete_source):
    prepare = _preparer(concrete_source)
    prepare(_job()); prepare(_job())
    assert concrete_source[2] == Counter(a=2, b=2) and not concrete_source[1]
    assert prepare._cache is None and prepare.cache_policy is None
    assert len(prepare.__getstate__()) == 4
    expected_base = pp._fingerprint({"schema": pp.SCHEMA, "manifest": concrete_source[0],
        "identity_only": True, "implementation_fingerprint": pp._implementation_fingerprint(),
        "paths": {field.name: str(getattr(prepare.paths, field.name)) for field in fields(prepare.paths)}})
    assert prepare.base_source_fingerprint == expected_base
    assert prepare.source_fingerprint == pp._fingerprint({"base_source_fingerprint": expected_base,
                                                       "reservation_manifest": None})


def test_capacity_is_instance_binding_but_one_algorithm_reservation(concrete_source):
    one, four = _preparer(concrete_source, _policy()), _preparer(concrete_source, _policy(processes=4))
    changed_total = _preparer(concrete_source, _policy(20_000_000))
    assert one.base_source_fingerprint == four.base_source_fingerprint == changed_total.base_source_fingerprint
    assert len({one.source_fingerprint, four.source_fingerprint, changed_total.source_fingerprint}) == 3
    assert one.base_source_fingerprint != _preparer(concrete_source).base_source_fingerprint
    job = _job()
    sizes = {pp.reservation_job_fingerprint(job): len(pp._prepare_bytes(one, job))}
    reservation = pp.make_reservation_manifest(source_fingerprint=one.base_source_fingerprint,
        evidence_id="synthetic-size-evidence", measured_serialized_bytes=sizes)
    for policy in (_policy(), _policy(processes=4), _policy(4, 4)):
        instance = _preparer(concrete_source, policy, reservation_manifest=reservation)
        instance.validate_jobs((job,))
        assert len(pp._prepare_bytes(instance, job)) <= job.reserved_bytes
    with pytest.raises(ValueError, match="reservation_manifest_binding_mismatch"):
        _preparer(concrete_source, reservation_manifest=reservation)


@pytest.mark.parametrize("ids", [("a", "b"), ("b", "a"), ("a", "a")])
def test_cold_hit_bypass_and_capacities_preserve_content_with_actual_reservation_guard(concrete_source, ids):
    job = _job(ids=ids)
    original = _preparer(concrete_source)(job)
    sizes = []
    for policy in (_policy(), _policy(processes=4), _policy(4, 4)):
        prepare = _preparer(concrete_source, policy)
        assert prepare._cache is None
        for _ in range(2):
            encoded = pp._prepare_bytes(prepare, job)  # also asserts global RNG is unchanged
            actual = pickle.loads(encoded)
            _equal_job(actual, original)
            sizes.append(len(encoded))
        stats = prepare.cache_diagnostics()
        assert stats["accounted_bytes"] <= stats["max_bytes"]
        assert stats["hits"] if policy["global_max_bytes"] > 4 else stats["oversized_bypasses"] == 4
        if prepare._cache._entries:
            assert all(type(row.sidecar) is bytes for row, _ in prepare._cache._entries.values())
    # Equal content need not mean equal pickle memoization/length: repeated
    # records can share immutable metadata on hits but not on bypasses.
    assert all(0 < size <= job.reserved_bytes for size in sizes)


def test_direct_collation_transposition_does_not_alias_resident_cache(concrete_source):
    prepare = _preparer(concrete_source, _policy(), identity_only=False)
    baseline = _preparer(concrete_source, identity_only=False)
    job = _job(ids=("a", "a"), shifts=(1, 6))
    actual = prepare(job)
    expected = baseline(job)
    _equal_job(actual, expected)
    actual.raw_graph_batch["note"].x_cat.fill_(0)
    actual.raw_graph_batch["note"].entity_id[0] = ("mutated",)
    actual.sidecars[0]["entities"].clear()
    assert actual.sidecars[1] == expected.sidecars[1]
    _equal_job(prepare(job), expected)
    _equal_job(prepare(_job(ids=("a", "a"))), baseline(_job(ids=("a", "a"))))


def test_reset_pickle_and_process_change_start_cold_without_locks_or_graphs(concrete_source, monkeypatch):
    prepare = _preparer(concrete_source, _policy())
    prepare(_job()); prepare(_job())
    assert prepare.cache_diagnostics()["hits"] == 2
    revived = pickle.loads(pickle.dumps(prepare))
    assert revived._cache is None and revived._loader is None and revived._owner_pid is None
    assert revived.source_fingerprint == prepare.source_fingerprint
    assert revived.cache_diagnostics()["hits"] == 0
    prior = prepare.reset_cache()
    assert prior["hits"] == 2 and prior["entries"] == 2
    assert prepare.cache_diagnostics()["entries"] == 0
    prepare(_job())
    before = dict(concrete_source[1])
    monkeypatch.setattr(pp.os, "getpid", lambda: 123456789)
    prepare(_job())
    assert concrete_source[1] == Counter({key: value + 1 for key, value in before.items()})
    assert prepare.cache_diagnostics()["misses"] == 2


def test_eviction_rebuilds_identical_payload_and_stays_bounded(concrete_source):
    probe = _preparer(concrete_source, _policy())
    probe(_job(ids=("a", "a")))
    size = probe.cache_diagnostics()["accounted_bytes"]
    prepare = _preparer(concrete_source, _policy(size + 16))
    job = _job(ids=("a", "a"))
    before = pp._prepare_bytes(prepare, job)
    prepare(_job(ids=("b", "b")))
    after = pp._prepare_bytes(prepare, job)
    _equal_job(pickle.loads(before), pickle.loads(after))
    assert len(before) <= job.reserved_bytes and len(after) <= job.reserved_bytes
    assert prepare.cache_diagnostics()["evictions"] >= 2
    assert prepare.cache_diagnostics()["accounted_bytes"] <= size + 16


@pytest.mark.parametrize("job", [_job(ids=("not-train", "a")), _job(shifts=(1, 0))])
def test_unbound_or_nonidentity_rejected_before_cache_or_source(concrete_source, job):
    prepare = _preparer(concrete_source, _policy())
    with pytest.raises(ValueError, match="unbound_scheduled_record|production_identity_only"):
        prepare(job)
    assert prepare._cache is None and not concrete_source[1]


def test_actual_serialization_over_reservation_is_rejected_for_cached_payload(concrete_source):
    prepare = _preparer(concrete_source, _policy())
    prepare(_job())  # A hit must still pass the actual serialization guard.
    job = pp.PrefetchJob(0, (0, 1), ("a", "b"), (0, 5), (0, 0), 1)
    with pytest.raises(pp.PrefetchError, match="result_exceeds_reservation"):
        pp._prepare_bytes(prepare, job)


def test_explicit_direct_validation_authenticates_sources_but_pool_still_requires_sizes(concrete_source, monkeypatch):
    prepare = _preparer(concrete_source)
    checked = []
    new_loader = prepare._new_loader
    monkeypatch.setattr(prepare, "_new_loader", lambda: (checked.append(True), new_loader())[1])
    prepare.validate_direct_jobs((_job(),))
    assert checked == [True]
    with pytest.raises(ValueError, match="production_reservation_manifest_required"):
        prepare.validate_jobs((_job(),))
    assert checked == [True]
    with pytest.raises(ValueError, match="unbound_scheduled_record"):
        prepare.validate_direct_jobs((_job(ids=("excluded", "a")),))
    assert checked == [True]


class _SyntheticCompactPrepare:
    """Explicit fixture transport: the real cache with the shared stats schema."""

    def __init__(self, policy, fault=None):
        self.cache_policy = pp.validate_cache_policy(policy)
        self.cache_instance_fingerprint = pp.cache_instance_fingerprint(policy, max_entries=2)
        self._cache = None
        self.fault = fault

    def cache_diagnostics(self):
        result = pp.cache_observation(self._cache, self.cache_policy, max_entries=2)
        if self.fault == "pid":
            result["pid"] += 1
        return result

    def __call__(self, job):
        if self._cache is None:
            self._cache = pd.PreparedGraphCache(pd.SyntheticRecordLoader(
                lambda _: reference.build_source_free_fixture(), source_seal="fixture-cache",
                synthetic=True), synthetic=True, max_entries=2,
                max_bytes=self.cache_policy["global_max_bytes"] // self.cache_policy["preparing_processes"])
        result = self._cache.prepare_batch(job.record_ids)
        return pp.PreparedCPUJob(job, reference.transpose_raw_graph_batch(result.raw_graph_batch, job.applied_shifts),
                                 result.sidecars)


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_real_transport_preserves_payload_and_reports_bounded_process_caches(workers):
    policy = _policy(processes=max(workers, 1))
    prepare = _SyntheticCompactPrepare(policy)
    jobs = tuple(_job(i) for i in range(8))
    with pp.OrderedPrefetch(jobs, prepare, source_fingerprint="explicit-synthetic-cache",
        synthetic=True, config=pp.PrefetchConfig(workers=workers, max_pending_batches=5,
        buffer_byte_cap=50_000_000, startup_timeout_seconds=30)) as pool:
        for job in jobs:
            actual = pool.peek()
            assert actual.job == job
            expected = _SyntheticCompactPrepare(_policy())(job)
            _equal_job(actual, expected)
            pool.commit(job.position, applied=True)
        evidence = pool.diagnostics()["prepared_cache"]
        observations = evidence["observations"]
        assert len(observations) == max(workers, 1)
        assert sum(row["observation"]["accounted_bytes"] for row in observations) <= policy["global_max_bytes"]
        assert sum(row["observation"]["hits"] + row["observation"]["misses"] for row in observations) == 16
        assert len({row["observation"]["pid"] for row in observations}) == max(workers, 1)
        assert all(row["observation"]["initialized"] for row in observations)
        if workers:
            assert prepare._cache is None  # no hidden parent graph copy


def test_wrong_worker_cache_observation_fails_closed_and_closes_owned_pool():
    prepare = _SyntheticCompactPrepare(_policy(), fault="pid")
    pool = pp.OrderedPrefetch((_job(),), prepare, source_fingerprint="fixture-invalid-cache-stats",
        synthetic=True, config=pp.PrefetchConfig(workers=1, buffer_byte_cap=10_000_000))
    with pytest.raises(pp.PrefetchError, match="cache_observation_process_mismatch"):
        pool.peek()
    assert pool.diagnostics()["closed"] and all(not slot.process.is_alive() for slot in pool._slots)


def test_wrong_process_allocation_rejected_before_pool_spawn():
    prepare = _SyntheticCompactPrepare(_policy(processes=4))
    with pytest.raises(ValueError, match="cache_preparing_process_count_mismatch"):
        pp.OrderedPrefetch((_job(),), prepare, source_fingerprint="fixture", synthetic=True,
            config=pp.PrefetchConfig(workers=1, buffer_byte_cap=10_000_000))


def test_worker_pool_rejects_extra_warm_parent_cache_before_spawn():
    prepare = _SyntheticCompactPrepare(_policy())
    prepare(_job())
    with pytest.raises(ValueError, match="worker_arm_requires_cold_parent_cache"):
        pp.OrderedPrefetch((_job(),), prepare, source_fingerprint="fixture", synthetic=True,
            config=pp.PrefetchConfig(workers=1, buffer_byte_cap=10_000_000))


@pytest.mark.parametrize("mutation", [lambda row: row.update(accounted_bytes=row["max_bytes"] + 1),
    lambda row: row.update(hits=True), lambda row: row.update(policy_fingerprint="wrong"),
    lambda row: row.update(initialized=False), lambda row: row.update(unexpected=1),
    lambda row: row["policy"].update(preparing_processes=True)])
def test_numeric_and_policy_diagnostics_reject_malformed_observations(mutation):
    prepare = _SyntheticCompactPrepare(_policy())
    prepare(_job())
    observation = prepare.cache_diagnostics()
    mutation(observation)
    with pytest.raises(pp.PrefetchError, match="cache_observation"):
        pp._validate_cache_observation(observation, prepare, pid=os.getpid())
