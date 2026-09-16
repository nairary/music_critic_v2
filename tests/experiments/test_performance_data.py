"""Source-free equivalence/ownership/binding checks for isolated CPU replay."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from music_critic.data import dumps_piece, load_piece
from music_critic.experiments.analysisgnn import performance_data as pd
from music_critic.experiments.analysisgnn.corrected_training import (
    ProductionArtifactPaths, align_target_sidecars_after_prediction, build_source_free_fixture,
    combine_single_record_raw_batches, transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _equal(first, second)
    else:
        assert left == right


@pytest.fixture
def fixture_record():
    return build_source_free_fixture()


def _cache(fixture_record, *, max_entries=2, max_bytes=10_000_000, binding_getter=None):
    calls = Counter()

    def load(record_id):
        calls[record_id] += 1
        return fixture_record

    loader = pd.SyntheticRecordLoader(load, source_seal="source-free-test-v1", synthetic=True,
                                     binding_getter=binding_getter)
    cache = pd.PreparedGraphCache(loader, max_entries=max_entries, max_bytes=max_bytes, synthetic=True)
    return cache, loader, calls


@pytest.mark.parametrize("ids", [("a", "b"), ("b", "a"), ("a", "a")])
def test_direct_collation_matches_frozen_singleton_combine_identity(fixture_record, ids):
    singleton = fixture_record[0].raw_graph_batch
    distinct = transpose_raw_graph_batch(singleton, (1,))
    calls = Counter()

    def load(record_id):
        calls[record_id] += 1
        sidecar = deepcopy(fixture_record[1])
        sidecar["record_id"] = record_id
        return (singleton if record_id == "a" else distinct), sidecar

    loader = pd.SyntheticRecordLoader(load, source_seal="distinct-synthetic-fixtures", synthetic=True)
    cache = pd.PreparedGraphCache(loader, max_entries=2, max_bytes=10_000_000, synthetic=True)
    result = cache.prepare_batch(ids)
    expected = transpose_raw_graph_batch(combine_single_record_raw_batches(
        [singleton if record_id == "a" else distinct for record_id in ids]), (0, 0))
    _equal(result.raw_graph_batch.to_dict(), expected.to_dict())
    _equal(result.raw_graph_batch._slice_dict, expected._slice_dict)
    _equal(result.raw_graph_batch._inc_dict, expected._inc_dict)
    assert result.record_ids == ids
    assert tuple(sidecar["record_id"] for sidecar in result.sidecars) == ids
    again = cache.prepare_batch(ids)
    _equal(again.raw_graph_batch.to_dict(), expected.to_dict())
    assert sum(calls.values()) == len(set(ids))
    print(f"direct collation exact, order={ids}, builds={sum(calls.values())}")


def test_consumer_and_loader_mutations_do_not_change_cached_graph_or_sidecar(fixture_record):
    cache, _, _ = _cache(fixture_record)
    first = cache.get("a")
    original_graph = deepcopy(first.raw_graph.to_dict())
    original_sidecar = deepcopy(first.sidecar)
    first.raw_graph["note"].x_cat.fill_(0)
    first.sidecar["entities"].clear()
    result = cache.prepare_batch(("a", "a"))
    result.raw_graph_batch["note"].x_cat.fill_(0)
    result.raw_graph_batch["note"].entity_id[0] = ("corrupted",)
    result.sidecars[0]["entities"].clear()
    fixture_record[0].raw_graph_batch["note"].x_cat.fill_(0)
    fixture_record[1]["entities"].clear()
    final = cache.get("a")
    _equal(final.raw_graph.to_dict(), original_graph)
    assert final.sidecar == original_sidecar
    assert result.sidecars[1] == original_sidecar


def test_lru_eviction_and_oversized_bypass_are_bounded(fixture_record):
    cache, _, calls = _cache(fixture_record, max_entries=2)
    cache.get("a")
    cache.get("b")
    cache.get("a")  # b is now least recent.
    cache.get("c")
    cache.get("b")
    assert calls == Counter(a=1, b=2, c=1)
    state = cache.diagnostics()
    assert state["entries"] == 2 and state["evictions"] == 2
    assert 0 < state["accounted_bytes"] <= state["max_bytes"]
    tiny, _, tiny_calls = _cache(fixture_record, max_bytes=1)
    tiny.get("a")
    tiny.get("a")
    assert tiny_calls["a"] == 2
    assert tiny.diagnostics()["entries"] == tiny.diagnostics()["accounted_bytes"] == 0
    assert tiny.diagnostics()["oversized_bypasses"] == 2
    sized, _, _ = _cache(fixture_record, max_entries=10,
                          max_bytes=state["accounted_bytes"] // 2 + 100)
    sized.get("a")
    sized.get("b")
    assert sized.diagnostics()["entries"] == 1
    assert sized.diagnostics()["evictions"] == 1


@pytest.mark.parametrize("split", ["TEST", "test", "validation", "unknown"])
def test_split_gate_precedes_reads_and_injected_loader(fixture_record, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("forbidden split performed a read/callback")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(pd.PerformanceDataError, match="train_only"):
        pd.FrozenProductionRecordLoader(split=split)
    loader = pd.SyntheticRecordLoader(forbidden, source_seal="test", synthetic=True,
                                     binding_getter=forbidden)
    cache = pd.PreparedGraphCache(loader, max_bytes=100, max_entries=1, synthetic=True)
    for action in (loader.get, loader.check_binding, cache.get):
        with pytest.raises(pd.PerformanceDataError, match="train_only"):
            action("a", split=split)
    with pytest.raises(pd.PerformanceDataError, match="train_only"):
        cache.prepare_batch(("a", "b"), split=split)


def test_injection_requires_explicit_synthetic_opt_in(fixture_record):
    with pytest.raises(pd.PerformanceDataError, match="synthetic_opt_in_required"):
        pd.SyntheticRecordLoader(lambda _: fixture_record, source_seal="test")
    _, loader, _ = _cache(fixture_record)
    with pytest.raises(pd.PerformanceDataError, match="synthetic_opt_in_required"):
        pd.PreparedGraphCache(loader, max_bytes=100, max_entries=1)


def test_changed_synthetic_binding_is_rejected_on_hit_and_after_eviction(fixture_record):
    version = {"a": "v1", "b": "v1"}
    cache, _, _ = _cache(fixture_record, max_entries=1, binding_getter=version.__getitem__)
    cache.get("a")
    version["a"] = "v2"
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        cache.get("a")
    cache.get("b")
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        cache.get("a")


def test_synthetic_resume_manifest_rechecks_bindings_without_rebuilding(fixture_record):
    versions = {"a": "v1", "b": "v1"}
    cache, loader, calls = _cache(fixture_record, binding_getter=versions.__getitem__)
    cache.prepare_batch(("a", "b"))
    manifest = loader.validate_all_bindings()
    assert loader.validate_all_bindings(expected=manifest) == manifest
    assert calls == Counter(a=1, b=1)
    _, resumed, resumed_calls = _cache(fixture_record, binding_getter=versions.__getitem__)
    assert resumed.validate_all_bindings(expected=manifest) == manifest
    assert not resumed_calls
    malformed = {**manifest, "source_seal": "changed"}
    with pytest.raises(pd.PerformanceDataError, match="resume_binding_mismatch"):
        resumed.validate_all_bindings(expected=malformed)
    versions["a"] = "v2"
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        loader.validate_all_bindings()
    _, fresh, fresh_calls = _cache(fixture_record, binding_getter=versions.__getitem__)
    with pytest.raises(pd.PerformanceDataError, match="resume_binding_mismatch"):
        fresh.validate_all_bindings(expected=manifest)
    assert not fresh_calls


def _warm_fixture(tmp_path, monkeypatch, *, canonical_split=None):
    """Synthetic split metadata only, including locked IDs; no corpus artifacts."""
    root = tmp_path / "splits"
    root.mkdir()
    rows = []
    for split, count, components in (("train", 1295, 1209), ("validation", 162, 147), ("test", 162, 151)):
        rows.extend({"record_id": f"synthetic:{split}:{i}", "split": split,
                     "source_component_id": f"component:{split}:{i % components}"} for i in range(count))
    split_bytes = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (root / "split_assignments.jsonl").write_bytes(split_bytes)
    monkeypatch.setattr(pd, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", sha256(split_bytes).hexdigest())
    record_id = "synthetic:train:0"
    piece = load_piece(Path(__file__).resolve().parents[1] / "fixtures/data/canonical_piece_v2.json")
    assert piece.dataset_name == "synthetic"
    piece = replace(piece, split=canonical_split, annotations=(), targets=())
    piece_bytes = dumps_piece(piece).encode()
    sidecar = {"record_id": record_id, "entities": [], "relations": [], "source": "synthetic"}
    sidecar_bytes = json.dumps(sidecar, sort_keys=True).encode()
    paths = ProductionArtifactPaths(b3_root=root, cache_root=tmp_path / "warm-cache")
    record_root = paths.cache_root / sha256(record_id.encode()).hexdigest()[:24]
    record_root.mkdir(parents=True)
    (record_root / "piece.json").write_bytes(piece_bytes)
    (record_root / "target_sidecar.json").write_bytes(sidecar_bytes)
    metadata = {"record_id": record_id, "split": "train", "piece_sha256": sha256(piece_bytes).hexdigest(),
                "sidecar_sha256": sha256(sidecar_bytes).hexdigest()}
    (record_root / "metadata.json").write_text(json.dumps(metadata))
    return paths, record_id, record_root, piece, sidecar


def _write_warm_record(paths, record_id, piece, sidecar):
    """Write only synthetic fixture bytes; keep non-canonical JSON formatting."""
    record_root = paths.cache_root / sha256(record_id.encode()).hexdigest()[:24]
    record_root.mkdir(exist_ok=True)
    piece_bytes = dumps_piece(piece).encode()
    # Ordering, Unicode, indentation and trailing whitespace are intentional:
    # the compact cache must retain source bytes, not a reserialization.
    sidecar_bytes = (json.dumps(sidecar, ensure_ascii=False, indent=1) + "\n \t\n").encode()
    (record_root / "piece.json").write_bytes(piece_bytes)
    (record_root / "target_sidecar.json").write_bytes(sidecar_bytes)
    (record_root / "metadata.json").write_text(json.dumps({
        "record_id": record_id, "split": "train", "piece_sha256": sha256(piece_bytes).hexdigest(),
        "sidecar_sha256": sha256(sidecar_bytes).hexdigest(),
    }))
    return record_root, sidecar_bytes


def _rich_sidecar(record_id):
    return {
        "record_id": record_id,
        "entities": [{"canonical_entity_id": "synthetic:entity", "targets": {
            "synthetic:head": {
                "available": False, "masked": True, "canonical_value": None,
                "confidence": 0.5,
                "provenance": {"parents": [{"source": "synthetic:C♯", "ordinal": 2}]},
            },
        }}],
        "relations": [],
        "source": "synthetic",
        "ordered_extra": {"z": None, "a": [-0.0, True, 9007199254740993, "C♯"],
                          "empty": {"array": [], "object": {}}},
    }


def test_production_cache_retains_source_bytes_and_returns_independent_complete_sidecars(
    tmp_path, monkeypatch,
):
    paths, record_id, _, piece, _ = _warm_fixture(tmp_path, monkeypatch)
    sidecar = _rich_sidecar(record_id)
    _, payload = _write_warm_record(paths, record_id, piece, sidecar)
    scopes, loaded_records = [], []

    @contextmanager
    def context(name):
        scopes.append(name)
        yield

    loader = pd.FrozenProductionRecordLoader(paths=paths)
    original_get_for_cache = loader._get_for_cache

    def observed(*args, **kwargs):
        loaded = original_get_for_cache(*args, **kwargs)
        loaded_records.append(loaded)
        return loaded

    monkeypatch.setattr(loader, "_get_for_cache", observed)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2, context=context)
    result = cache.prepare_batch((record_id, record_id))
    stored, accounted = next(iter(cache._entries.values()))
    assert isinstance(stored, pd._EncodedSidecarRecord)
    assert type(stored.sidecar) is bytes and stored.sidecar == payload
    assert accounted == pd._accounted_bytes(stored)
    assert stored.raw_graph is not loaded_records[0].raw_graph
    assert len(loaded_records) == 1  # The second occurrence is a hit.
    assert Counter(scopes) == {
        "cache_entry_copy": 1, "cache_entry_accounting": 1,
        "direct_collation": 1, "sidecar_materialize": 2,
    }
    for returned in result.sidecars:
        _equal(returned, sidecar)
        assert returned["ordered_extra"]["a"][0].hex() == (-0.0).hex()
        assert type(returned["ordered_extra"]["a"][1]) is bool
        assert type(returned["ordered_extra"]["a"][2]) is int
    result.sidecars[0]["entities"][0]["targets"]["synthetic:head"]["provenance"]["parents"][0]["source"] = "changed"
    result.sidecars[0]["ordered_extra"]["empty"]["array"].append("changed")
    result.raw_graph_batch["note"].x_cat.fill_(0)
    loaded_records[0].raw_graph["note"].x_cat.fill_(0)
    _equal(result.sidecars[1], sidecar)
    public = cache.get(record_id)
    assert type(public) is pd.PreparedRecord and isinstance(public.sidecar, dict)
    _equal(public.sidecar, sidecar)
    _equal(public.raw_graph.to_dict(), stored.raw_graph.to_dict())
    public.sidecar["entities"].clear()
    public.raw_graph["note"].x_cat.fill_(0)
    _equal(cache.get(record_id).sidecar, sidecar)
    public_loader_result = loader.get(record_id)
    assert type(public_loader_result) is pd.PreparedRecord
    _equal(public_loader_result.sidecar, sidecar)
    _equal(cache.get(record_id).raw_graph.to_dict(), public_loader_result.raw_graph.to_dict())


def test_compact_cache_byte_accounting_and_exact_capacity_boundary(tmp_path, monkeypatch):
    paths, record_id, _, piece, _ = _warm_fixture(tmp_path, monkeypatch)
    sidecar = _rich_sidecar(record_id)
    sidecar["diagnostics"] = [{"source": "synthetic", "nested": [i, None, True]} for i in range(100)]
    _, payload = _write_warm_record(paths, record_id, piece, sidecar)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2)
    cache.get(record_id)
    stored, size = next(iter(cache._entries.values()))
    assert size == pd._accounted_bytes(stored)
    assert size < pd._accounted_bytes(pd.PreparedRecord(stored.raw_graph, sidecar, stored.binding_seal))
    padded = replace(stored, sidecar=stored.sidecar + b" " * 4096)
    assert pd._accounted_bytes(padded) - size == 4096
    exact = pd.PreparedGraphCache(loader, max_bytes=size, max_entries=2)
    below = pd.PreparedGraphCache(loader, max_bytes=size - 1, max_entries=2)
    _equal(exact.get(record_id).sidecar, sidecar)
    _equal(below.get(record_id).sidecar, sidecar)
    assert exact.diagnostics()["entries"] == 1 and exact.diagnostics()["accounted_bytes"] == size
    assert below.diagnostics()["entries"] == below.diagnostics()["accounted_bytes"] == 0
    assert below.diagnostics()["oversized_bypasses"] == 1
    assert next(iter(exact._entries.values()))[0].sidecar == payload


@pytest.mark.parametrize("max_bytes", [1, 10_000_000])
def test_encoded_sidecars_survive_eviction_reload_and_oversized_bypass(tmp_path, monkeypatch, max_bytes):
    paths, first, _, piece, _ = _warm_fixture(tmp_path, monkeypatch)
    second = "synthetic:train:1"
    sidecars = {record_id: _rich_sidecar(record_id) for record_id in (first, second)}
    for record_id, sidecar in sidecars.items():
        _write_warm_record(paths, record_id, piece, sidecar)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    reads = Counter()
    original = loader._get_for_cache

    def counted(record_id, **kwargs):
        reads[record_id] += 1
        return original(record_id, **kwargs)

    monkeypatch.setattr(loader, "_get_for_cache", counted)
    cache = pd.PreparedGraphCache(loader, max_bytes=max_bytes, max_entries=1)
    for record_id in (first, second, first):
        returned = cache.get(record_id)
        _equal(returned.sidecar, sidecars[record_id])
        returned.sidecar["entities"].clear()
    assert reads == Counter({first: 2, second: 1})
    diagnostics = cache.diagnostics()
    assert diagnostics["misses"] == 3 and diagnostics["hits"] == 0
    assert diagnostics["accounted_bytes"] <= max_bytes
    if max_bytes == 1:
        assert diagnostics["oversized_bypasses"] == 3 and diagnostics["entries"] == 0
        assert diagnostics["accounted_bytes"] == diagnostics["evictions"] == 0
    else:
        assert diagnostics["entries"] == 1 and diagnostics["evictions"] == 2
        assert diagnostics["oversized_bypasses"] == 0
    duplicate = cache.prepare_batch((first, first))
    duplicate.sidecars[0]["entities"].clear()
    _equal(duplicate.sidecars[1], sidecars[first])


def test_production_subclass_keeps_defensive_dictionary_path(tmp_path, monkeypatch):
    paths, record_id, _, piece, _ = _warm_fixture(tmp_path, monkeypatch)
    sidecar = _rich_sidecar(record_id)
    _write_warm_record(paths, record_id, piece, sidecar)
    loaded_records = []

    class DerivedLoader(pd.FrozenProductionRecordLoader):
        def _get_for_cache(self, *args, **kwargs):
            pytest.fail("subclass entered concrete source-byte path")

        def get(self, *args, **kwargs):
            result = super().get(*args, **kwargs)
            loaded_records.append(result)
            return result

    loader = DerivedLoader(paths=paths)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2)
    cache.get(record_id)
    stored, _ = next(iter(cache._entries.values()))
    assert type(stored) is pd.PreparedRecord and isinstance(stored.sidecar, dict)
    loaded_records[0].sidecar["entities"].clear()
    loaded_records[0].raw_graph["note"].x_cat.fill_(0)
    _equal(cache.get(record_id).sidecar, sidecar)
    _equal(cache.get(record_id).raw_graph.to_dict(), loader.get(record_id).raw_graph.to_dict())


def test_source_byte_cache_preserves_all_head_frozen_alignment(tmp_path, monkeypatch):
    paths, record_id, _, _, _ = _warm_fixture(tmp_path, monkeypatch)
    captured = []

    def observe_piece(piece, **kwargs):
        captured.append(piece)
        return prepare_multisource_sample(piece, **kwargs)

    with monkeypatch.context() as local:
        local.setattr("music_critic.tasks.prepare_multisource_sample", observe_piece)
        singleton, sidecar = build_source_free_fixture()
    assert len(captured) == 1 and captured[0].metadata.source_format == "synthetic"
    sidecar["record_id"] = record_id
    sidecar["extra_provenance"] = _rich_sidecar(record_id)["entities"]
    _write_warm_record(paths, record_id, replace(captured[0], split=None), sidecar)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2)
    prepared = cache.prepare_batch((record_id, record_id))
    expected = transpose_raw_graph_batch(combine_single_record_raw_batches(
        [singleton.raw_graph_batch, singleton.raw_graph_batch]), (0, 0))
    _equal(prepared.raw_graph_batch.to_dict(), expected.to_dict())
    _equal(prepared.raw_graph_batch._slice_dict, expected._slice_dict)
    _equal(prepared.raw_graph_batch._inc_dict, expected._inc_dict)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        output = CorrectedAnalysisGNNModel().eval()(expected)
    before = align_target_sidecars_after_prediction(output, expected, (sidecar, sidecar), shifts=(0, 0))
    after = align_target_sidecars_after_prediction(
        output, prepared.raw_graph_batch, prepared.sidecars, shifts=(0, 0),
    )
    assert len(before.heads) == 18 and all(head.valid_mask.any() for head in before.heads.values())
    _equal(asdict(after), asdict(before))


def test_encoded_cache_requires_matching_decoded_record_id(tmp_path, monkeypatch):
    paths, record_id, _, piece, _ = _warm_fixture(tmp_path, monkeypatch)
    _write_warm_record(paths, record_id, piece, _rich_sidecar("synthetic:other-record"))
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2)
    with pytest.raises(pd.PerformanceDataError, match="sidecar_record_mismatch"):
        cache.get(record_id)
    assert cache.diagnostics()["entries"] == cache.diagnostics()["accounted_bytes"] == 0


@pytest.mark.parametrize("canonical_split", [None, "train"])
def test_warm_loader_reads_payload_once_per_miss_hits_only_stat_and_baseline_rebuilds(
    tmp_path, monkeypatch, canonical_split,
):
    paths, record_id, record_root, piece, sidecar = _warm_fixture(
        tmp_path, monkeypatch, canonical_split=canonical_split,
    )
    reads, scopes = Counter(), []
    original_read = Path.read_bytes
    original_prepare = pd.prepare_multisource_sample
    observed_canonical_splits = []

    def observed_prepare(canonical_piece):
        observed_canonical_splits.append(canonical_piece.split)
        return original_prepare(canonical_piece)

    def counted(path):
        reads[str(path)] += 1
        return original_read(path)

    @contextmanager
    def context(name):
        scopes.append(name)
        yield

    monkeypatch.setattr(Path, "read_bytes", counted)
    monkeypatch.setattr(pd, "prepare_multisource_sample", observed_prepare)
    loader = pd.FrozenProductionRecordLoader(paths=paths, context=context)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2, context=context)
    prepared = cache.prepare_batch((record_id, record_id))
    assert reads[str(paths.b3_root / "split_assignments.jsonl")] == 1
    for filename in ("piece.json", "target_sidecar.json", "metadata.json"):
        assert reads[str(record_root / filename)] == 1
    reads_before = dict(reads)
    cache.prepare_batch((record_id, record_id))
    assert dict(reads) == reads_before
    baseline = collate_multisource_samples((prepare_multisource_sample(piece),))
    expected = transpose_raw_graph_batch(combine_single_record_raw_batches(
        [baseline.raw_graph_batch, baseline.raw_graph_batch]), (0, 0))
    _equal(prepared.raw_graph_batch.to_dict(), expected.to_dict())
    _equal(prepared.raw_graph_batch._slice_dict, expected._slice_dict)
    _equal(prepared.raw_graph_batch._inc_dict, expected._inc_dict)
    assert prepared.sidecars == (sidecar, sidecar)
    loader.get(record_id)
    assert observed_canonical_splits == [canonical_split, canonical_split]
    assert reads[str(record_root / "piece.json")] == 2
    assert {"split_read_hash", "read_hash_decode", "original_prepare_graph_and_fingerprints",
            "original_singleton_collation", "direct_collation"} <= set(scopes)


@pytest.mark.parametrize("canonical_split", ["validation", "test", "unknown"])
@pytest.mark.parametrize("through_cache", [False, True])
def test_explicit_nontrain_canonical_split_still_rejected(
    tmp_path, monkeypatch, canonical_split, through_cache,
):
    paths, record_id, _, _, _ = _warm_fixture(tmp_path, monkeypatch, canonical_split=canonical_split)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    consumer = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2) if through_cache else loader
    with pytest.raises(pd.PerformanceDataError, match="piece_not_train"):
        consumer.get(record_id)


@pytest.mark.parametrize("through_cache", [False, True])
def test_neutral_canonical_piece_requires_train_cache_metadata_before_decode(tmp_path, monkeypatch, through_cache):
    paths, record_id, record_root, _, _ = _warm_fixture(tmp_path, monkeypatch)
    metadata_path = record_root / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["split"] = "validation"
    metadata_path.write_text(json.dumps(metadata))
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    consumer = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2) if through_cache else loader
    monkeypatch.setattr(pd, "loads_piece", lambda _: pytest.fail("invalid cache metadata reached canonical decode"))
    with pytest.raises(pd.PerformanceDataError, match="cache_binding_invalid"):
        consumer.get(record_id)


@pytest.mark.parametrize("through_cache", [False, True])
def test_warm_loader_missing_or_nontrain_record_never_reads_or_writes_record(tmp_path, monkeypatch, through_cache):
    paths, record_id, record_root, _, _ = _warm_fixture(tmp_path, monkeypatch)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    consumer = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2) if through_cache else loader
    before = set(tmp_path.rglob("*"))
    original_read = Path.read_bytes
    reads = []

    def counted(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    with pytest.raises(pd.PerformanceDataError, match="record_not_train"):
        consumer.get("synthetic:test:0")
    assert reads == []
    with pytest.raises(pd.PerformanceDataError, match="source_unavailable"):
        consumer.get("synthetic:train:1")
    assert set(tmp_path.rglob("*")) == before


def test_changed_warm_binding_and_resume_reject_resealed_metadata(tmp_path, monkeypatch):
    paths, record_id, record_root, _, sidecar = _warm_fixture(tmp_path, monkeypatch)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    cache = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=1)
    cache.get(record_id)
    manifest = loader.validate_all_bindings()
    # An unchanged file rewrite changes stat identity but preserves content.
    piece_path = record_root / "piece.json"
    piece_path.write_bytes(piece_path.read_bytes())
    cache.get(record_id)
    assert loader.validate_all_bindings() == manifest
    resumed = pd.FrozenProductionRecordLoader(paths=paths)
    assert resumed.validate_all_bindings(expected=manifest) == manifest
    sidecar["source"] = "changed"
    sidecar_bytes = json.dumps(sidecar, sort_keys=True).encode()
    (record_root / "target_sidecar.json").write_bytes(sidecar_bytes)
    metadata_path = record_root / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["sidecar_sha256"] = sha256(sidecar_bytes).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        cache.get(record_id)
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        loader.get(record_id)
    fresh = pd.FrozenProductionRecordLoader(paths=paths)
    with pytest.raises(pd.PerformanceDataError, match="resume_binding_mismatch"):
        fresh.validate_all_bindings(expected=manifest)


@pytest.mark.parametrize("through_cache", [False, True])
def test_invalid_metadata_hash_and_changed_split_fail_closed(tmp_path, monkeypatch, through_cache):
    paths, record_id, record_root, _, _ = _warm_fixture(tmp_path, monkeypatch)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    consumer = pd.PreparedGraphCache(loader, max_bytes=10_000_000, max_entries=2) if through_cache else loader
    (record_root / "metadata.json").write_text("{}")
    with pytest.raises(pd.PerformanceDataError, match="cache_binding_invalid"):
        consumer.get(record_id)
    (paths.b3_root / "split_assignments.jsonl").write_text("{}\n")
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        consumer.get(record_id)


def test_full_rehash_detects_change_even_if_cheap_identity_detector_cannot(tmp_path, monkeypatch):
    paths, record_id, record_root, _, _ = _warm_fixture(tmp_path, monkeypatch)
    loader = pd.FrozenProductionRecordLoader(paths=paths)
    loader.get(record_id)
    path = record_root / "target_sidecar.json"
    old_identity = pd._identity(path)
    real_identity = pd._identity
    path.write_bytes(path.read_bytes() + b" ")
    # Model an identity detector with no observable change. Stat is explicitly
    # not proof; startup/resume integrity must still use the complete bytes.
    monkeypatch.setattr(pd, "_identity", lambda value: old_identity if value == path else real_identity(value))
    loader.check_binding(record_id)
    with pytest.raises(pd.PerformanceDataError, match="source_binding_changed"):
        loader.validate_all_bindings()
