from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn import corrected_training as old
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.performance_prefetch import _assert_cpu
from music_critic.experiments.analysisgnn.temporal_slices import TemporalSliceModel
from music_critic.experiments.analysisgnn.temporal_slices.alignment import align_temporal_targets
from music_critic.experiments.analysisgnn.temporal_slices.graph import EDGES, build_temporal_batch
from music_critic.experiments.analysisgnn.temporal_slices.cache import (
    CACHE_SCHEMA, PreparedTemporalRecord, TemporalGraphCache, accounted_bytes,
    collate_temporal_graphs, encode_temporal_graph, decode_temporal_graph,
    implementation_fingerprint, transpose_temporal_historical,
    FrozenTemporalRecordLoader, read_temporal_record,
)
from music_critic.experiments.analysisgnn.harmonic_fusion_data import FrozenFusionData
from music_critic.data import load_piece
from test_harmonic_fusion_data import source  # Shared synthetic sealed-file fixture.


@pytest.fixture
def fixture(monkeypatch):
    import music_critic.tasks as tasks
    captured = []
    prepare = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        captured.append(piece)
        return prepare(piece, *args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_multisource_sample", capture)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    _, sidecar = old.build_source_free_fixture()
    piece = captured[0]
    other = replace(piece, notes=(
        replace(piece.notes[0], duration_qn=RationalTime(2)),
        replace(piece.notes[1], onset_qn=RationalTime(1, 3), duration_qn=RationalTime(1, 3)),
    ))
    yield piece, other, sidecar
    torch.set_num_threads(threads)


class SyntheticLoader:
    synthetic = True
    source_seal = "fixture-source"
    def __init__(self, pieces, sidecar):
        self.pieces, self.sidecar = pieces, sidecar
        self.calls = 0
        self.revision = "initial"
        self.last = None

    def check_binding(self, record_id, *, split="train"):
        assert split == "train"
        return fingerprint({"record": record_id, "revision": self.revision})

    def get(self, record_id, *, split="train"):
        self.calls += 1
        self.last = PreparedTemporalRecord(build_temporal_batch((self.pieces[record_id],)),
                         json.dumps(self.sidecar).encode(), self.check_binding(record_id, split=split))
        return self.last


def cache_for(fixture, *, max_bytes=10**8, max_entries=3):
    a, b, sidecar = fixture
    loader = SyntheticLoader({"a": a, "b": b}, sidecar)
    return TemporalGraphCache(loader, max_bytes=max_bytes, max_entries=max_entries, synthetic=True), loader


def test_collation_exactly_matches_raw_reference_and_keeps_sample_boundaries(fixture):
    a, b, _ = fixture
    reference = build_temporal_batch((a, b, a))
    result = collate_temporal_graphs([build_temporal_batch((p,)) for p in (a, b, a)])
    assert result.fingerprint() == reference.fingerprint()
    assert result.slots[1].time_slots[1][0] == Fraction(1, 3)
    for edge in EDGES:
        assert torch.equal(result.edges[edge], reference.edges[edge])


def test_payload_roundtrip_is_primitive_cpu_transport_safe(fixture):
    a, b, _ = fixture
    graph = build_temporal_batch((a, b))
    payload = encode_temporal_graph(graph)
    _assert_cpu(payload)
    restored = decode_temporal_graph(pickle.loads(pickle.dumps(payload, protocol=5)))
    assert graph.fingerprint() == restored.fingerprint()
    assert implementation_fingerprint() == implementation_fingerprint()


@pytest.mark.parametrize("mutation", ("schema", "graph_schema", "extra", "fraction", "float", "duplicate_beat"))
def test_payload_rejects_old_cache_schema_and_noncanonical_time(fixture, mutation):
    a, _, _ = fixture
    payload = encode_temporal_graph(build_temporal_batch((a,)))
    if mutation in ("schema", "graph_schema"):
        payload[mutation] = "old-raw-cache"
    elif mutation == "extra":
        payload["unregistered"] = 1
    else:
        row = payload["slots"][0]
        if mutation == "duplicate_beat":
            row["beat_ids"] += row["beat_ids"][:1]
        else:
            time = (2, 2, "rawtime:1_1") if mutation == "fraction" else (1.0, 1, "rawtime:1_1")
            row["times"] = (time,) + row["times"][1:]
    with pytest.raises(ValueError):
        decode_temporal_graph(payload)


def test_cached_entries_and_duplicate_sidecars_are_private(fixture):
    cache, loader = cache_for(fixture)
    prepared = cache.prepare_batch(("a", "a"))
    original = prepared.graph.fingerprint()
    assert loader.calls == 1
    assert prepared.sidecars[0] == prepared.sidecars[1]
    assert prepared.sidecars[0] is not prepared.sidecars[1]
    prepared.sidecars[0]["entities"].clear()
    prepared.graph.slice_features.fill_(-42)
    prepared.graph.base["note"].x_cat.fill_(0)
    prepared.graph.edges[EDGES[0]].fill_(0)
    loader.last.graph.slice_features.fill_(55)
    restored = cache.prepare_batch(("a", "a"))
    assert restored.graph.fingerprint() == original
    assert restored.sidecars[0]["entities"]
    public = cache.get("a")
    public.graph.slice_features.fill_(89)
    assert cache.prepare_batch(("a", "a")).graph.fingerprint() == original
    assert loader.calls == 1


def test_cache_is_bounded_and_oversized_records_bypass(fixture):
    cache, loader = cache_for(fixture, max_entries=1)
    for key in ("a", "b", "a"):
        cache.prepare_batch((key,))
    stats = cache.diagnostics()
    assert stats["entries"] == 1 and stats["evictions"] == 2
    assert stats["accounted_bytes"] <= stats["max_bytes"]
    tiny, _ = cache_for(fixture, max_bytes=1)
    tiny.prepare_batch(("a", "a"))
    assert tiny.diagnostics()["entries"] == 0
    assert tiny.diagnostics()["oversized_bypasses"] == 2
    assert accounted_bytes(loader.last) > 1


def test_changed_binding_is_rejected_before_cached_record_return(fixture):
    cache, loader = cache_for(fixture)
    cache.get("a")
    loader.revision = "changed"
    with pytest.raises(ValueError, match="source_binding_changed"):
        cache.get("a")


def test_train_gate_is_before_loader_calls_and_synthetic_is_explicit(fixture):
    cache, loader = cache_for(fixture)
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="train_only"):
            cache.prepare_batch(("a",), split=split)
    assert loader.calls == 0
    with pytest.raises(ValueError):
        TemporalGraphCache(loader, max_bytes=10000, max_entries=1)
    with pytest.raises(ValueError):
        TemporalGraphCache(loader, max_bytes=True, max_entries=1, synthetic=True)


@pytest.mark.parametrize("shift", range(12))
def test_cached_all_shift_features_predictions_and_targets_equal_reference(fixture, shift):
    a, b, sidecar = fixture
    cache, _ = cache_for(fixture)
    # Exercise warm hits and two different occurrence shifts of the same record.
    cache.prepare_batch(("a", "b"))
    prepared = cache.prepare_batch(("a", "b"))
    shifts = (shift, (shift + 5) % 12)
    optimized = transpose_temporal_historical(prepared.graph, shifts)
    raw_reference = build_temporal_batch((a, b))
    raw_reference = replace(raw_reference, base=old.transpose_raw_graph_batch(raw_reference.base, shifts))
    assert optimized.fingerprint() == raw_reference.fingerprint()
    assert cache.get("a").graph.fingerprint() == build_temporal_batch((a,)).fingerprint()
    model = TemporalSliceModel().eval()
    with torch.no_grad():
        actual, expected = model(optimized), model(raw_reference)
    for task in old.ACTIVE_HEADS:
        assert torch.equal(actual.logits[task], expected.logits[task])
    x = align_temporal_targets(actual, prepared.sidecars, shifts=shifts)
    y = align_temporal_targets(expected, (sidecar, sidecar), shifts=shifts)
    for task in old.ACTIVE_HEADS:
        assert torch.equal(x.heads[task].candidate_indices, y.heads[task].candidate_indices)
        assert torch.equal(x.heads[task].valid_mask, y.heads[task].valid_mask)
        assert torch.equal(x.heads[task].values, y.heads[task].values)
        assert torch.equal(x.common_support_masks[task], y.common_support_masks[task])


def test_preparation_and_cache_do_not_consume_training_rng(fixture):
    cache, _ = cache_for(fixture)
    py, np_state, tr = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    prepared = cache.prepare_batch(("a", "b"))
    restored = decode_temporal_graph(pickle.loads(pickle.dumps(encode_temporal_graph(prepared.graph))))
    transpose_temporal_historical(restored, (5, 7))
    cache.prepare_batch(("a", "b"))
    assert random.getstate() == py
    assert np_state[0] == np.random.get_state()[0]
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert np_state[2:] == np.random.get_state()[2:]
    assert torch.equal(tr, torch.get_rng_state())


@pytest.mark.parametrize("split", ("train", "validation"))
def test_authenticated_reader_reuses_frozen_canonical_bytes(source, split):
    data = FrozenFusionData(source)
    row = next(r for r in source["records"] if r["split"] == split)
    graph, sidecar = read_temporal_record(data, row["record_id"], split=split)
    expected = build_temporal_batch((load_piece(row["piece"]["path"]),))
    assert graph.fingerprint() == expected.fingerprint()
    assert sidecar == json.loads(Path(row["sidecar"]["path"]).read_text())
    if split == "train":
        loader = FrozenTemporalRecordLoader(data)
        cache = TemporalGraphCache(loader, max_bytes=10**8, max_entries=1295)
        first = cache.prepare_batch((row["record_id"],))
        assert first.graph.fingerprint() == graph.fingerprint()
        assert first.sidecars == (sidecar,)
        assert cache.prepare_batch((row["record_id"],)).cache_diagnostics["hits"] == 1


@pytest.mark.parametrize("record_id,split", (("synthetic:test:0000", "test"),
    ("synthetic:test:0000", "train"), ("synthetic:validation:0000", "train"),
    ("synthetic:train:0001", "train"), ("missing", "validation")))
def test_temporal_reader_split_gate_precedes_io(source, monkeypatch, record_id, split):
    import music_critic.experiments.analysisgnn.temporal_slices.cache as module
    data = FrozenFusionData(source)
    def denied(*args, **kwargs):
        pytest.fail("split gate allowed forbidden payload I/O")
    monkeypatch.setattr(module, "read_bound_bytes", denied)
    monkeypatch.setattr(module, "_identity", denied)
    with pytest.raises(ValueError):
        read_temporal_record(data, record_id, split=split)


def test_cached_production_reader_detects_source_file_replacement(source):
    data = FrozenFusionData(source)
    row = next(r for r in source["records"] if r["split"] == "train")
    cache = TemporalGraphCache(FrozenTemporalRecordLoader(data), max_bytes=10**8, max_entries=1295)
    cache.get(row["record_id"])
    file = Path(row["piece"]["path"])
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact_size_mismatch"):
        cache.get(row["record_id"])
