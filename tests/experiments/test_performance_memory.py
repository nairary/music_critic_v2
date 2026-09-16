"""Pure synthetic checks; no corpus traversal, GPU work or memory acceptance."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import itertools
import json

import pytest
import torch
from torch_geometric.data import HeteroData

from music_critic.experiments.analysisgnn.corrected_training import build_source_free_fixture
from music_critic.experiments.analysisgnn import performance_data as pd
from music_critic.experiments.analysisgnn import performance_memory as pm
from music_critic.graph import GraphContractError


@pytest.fixture
def inputs():
    batch, sidecar = build_source_free_fixture()
    sidecar = deepcopy(sidecar)
    sidecar["record_id"] = "record-a"
    # Preserve noncanonical source whitespace, nested target/mask/provenance data.
    payload = json.dumps(sidecar, indent=3, ensure_ascii=False).encode() + b"\n\n"
    metadata = pm.TrainRecordMetadata(
        record_id="record-a", source_component_id="component-a", split="train",
        canonical_sha256="a" * 64, sidecar_sha256=sha256(payload).hexdigest(),
        metadata_sha256="b" * 64, binding_seal="c" * 64, source_seal="d" * 64,
        code_seal="e" * 64, split_sha256="f" * 64,
        source_versions=(("fixture", "synthetic-v1"),),
    )
    prepared = pd.PreparedRecord(batch.raw_graph_batch.to_data_list()[0], sidecar, metadata.binding_seal)
    return prepared, metadata, payload


def _row(inputs, **kwargs):
    prepared, metadata, payload = inputs
    return pm.build_size_index_row(prepared, metadata=metadata, compact_sidecar_bytes=payload,
                                   cache_representation="source_json_bytes", **kwargs)


def _sized(base, record_id, **changes):
    """Explicit supplied-index fixture, not a claim these modified sizes were observed."""
    nodes = dict(base.nodes_by_type)
    for key in tuple(changes):
        if key in nodes:
            nodes[key] = changes.pop(key)
    edges = list(base.edges_by_relation)
    if "edges" in changes:
        edges = [(name, changes.pop("edges") if index == 0 else 0)
                 for index, (name, _) in enumerate(edges)]
    return replace(base, metadata=replace(base.metadata, record_id=record_id),
                   nodes_by_type=tuple(nodes.items()), edges_by_relation=tuple(edges), **changes)


def test_real_raw_fixture_sizes_accounting_and_input_ownership(inputs):
    prepared, metadata, payload = inputs
    original_sidecar = deepcopy(prepared.sidecar)
    graph_dict = deepcopy(prepared.raw_graph.to_dict())
    row = _row(inputs)
    graph = prepared.raw_graph
    assert row.record_id == metadata.record_id
    assert row.nodes == sum(graph[name].num_nodes for name in graph.node_types)
    assert row.edges == sum(graph[name].edge_index.shape[1] for name in graph.edge_types)
    assert row.coarse_length == 1 + graph["bar"].num_nodes + graph["track"].num_nodes
    for singular, plural in (("note", "notes"), ("onset", "onsets"), ("beat", "beats"),
                             ("bar", "bars"), ("track", "tracks")):
        assert row.to_dict()[plural] == graph[singular].num_nodes
    assert row.accounted_decoded_entry_bytes == pd._accounted_bytes(deepcopy(prepared))
    encoded = pd._EncodedSidecarRecord(graph, payload, metadata.binding_seal)
    assert row.accounted_compact_entry_bytes == pd._accounted_bytes(deepcopy(encoded))
    assert row.accounted_cache_entry_bytes == row.accounted_compact_entry_bytes
    assert row.accounted_decoded_entry_bytes != row.accounted_compact_entry_bytes
    assert row.unique_cpu_tensor_storage_bytes > 0
    assert prepared.sidecar == original_sidecar
    for store_name, store in graph_dict.items():
        if isinstance(store, dict):
            for key, value in store.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(graph.to_dict()[store_name][key], value)
    # Row holds only detached immutable scalars/tuples, no graph/sidecar aliases.
    prepared.sidecar.clear()
    prepared.raw_graph["note"].num_nodes = 0
    assert row.to_dict()["notes"] > 0
    assert json.loads(json.dumps(row.to_dict()))["metadata"]["sidecar_sha256"] == metadata.sidecar_sha256


def test_storage_aliases_views_empty_views_and_nested_references_count_backing_once():
    graph = HeteroData()
    backing = torch.arange(128, dtype=torch.int64)
    other = torch.ones(7, dtype=torch.float32)
    graph["note"].a = backing[1:2]
    graph["note"].b = backing[20:40]
    graph["note"].empty = backing[:0]
    repeated = {"x": other, "same": other}
    repeated["cycle"] = repeated
    graph["track"].nested = [repeated, (backing, other)]
    assert pm.unique_cpu_tensor_storage_bytes(graph) == 128 * 8 + 7 * 4
    assert pm.unique_cpu_tensor_storage_bytes(HeteroData()) == 0


@pytest.mark.parametrize("tensor", [torch.empty(2, device="meta"), torch.ones(2, 2).to_sparse(),
                                    torch.quantize_per_tensor(torch.ones(3), 0.1, 0, torch.qint8)])
def test_storage_rejects_non_cpu_or_unsupported_tensor_without_materializing(tensor):
    graph = HeteroData()
    graph["note"].value = tensor
    with pytest.raises(pm.PerformanceMemoryError, match="cpu_required|unsupported_tensor"):
        pm.unique_cpu_tensor_storage_bytes(graph)


def test_decoded_only_explicit_and_compact_requires_actual_bytes(inputs):
    prepared, metadata, _ = inputs
    row = pm.build_size_index_row(prepared, metadata=metadata, cache_representation="decoded_dict")
    assert row.accounted_compact_entry_bytes is None
    assert row.accounted_cache_entry_bytes == row.accounted_decoded_entry_bytes
    with pytest.raises(pm.PerformanceMemoryError, match="compact_bytes_required"):
        pm.build_size_index_row(prepared, metadata=metadata, cache_representation="source_json_bytes")


def test_compact_accounting_uses_verbatim_bytes_not_a_reserialized_projection(inputs):
    prepared, metadata, payload = inputs
    row = _row(inputs)
    padded = payload + b" " * 100_000
    alternate = pm.build_size_index_row(
        prepared, metadata=replace(metadata, sidecar_sha256=sha256(padded).hexdigest()),
        cache_representation="source_json_bytes", compact_sidecar_bytes=padded,
    )
    assert alternate.accounted_compact_entry_bytes == row.accounted_compact_entry_bytes + 100_000
    assert alternate.accounted_decoded_entry_bytes == row.accounted_decoded_entry_bytes


@pytest.mark.parametrize("split", [None, "TRAIN", "test", "validation", "unknown", " train"])
def test_noncanonical_or_nontrain_metadata_rejected_before_record_access(inputs, split):
    _, metadata, _ = inputs
    with pytest.raises(pm.PerformanceMemoryError, match="train_only"):
        replace(metadata, split=split)
    # A caller can bypass frozen dataclasses deliberately; public helpers recheck.
    metadata = deepcopy(metadata)
    object.__setattr__(metadata, "split", split)
    with pytest.raises(pm.PerformanceMemoryError, match="train_only"):
        pm.build_size_index_row(object(), metadata=metadata, cache_representation="decoded_dict")


@pytest.mark.parametrize("field,value", [
    ("record_id", ""), ("source_component_id", "  "), ("canonical_sha256", "bad"),
    ("source_versions", ()), ("source_versions", (("x", "1"), ("x", "2"))),
])
def test_bad_metadata_is_not_silently_filled(inputs, field, value):
    with pytest.raises(pm.PerformanceMemoryError):
        replace(inputs[1], **{field: value})


def test_record_and_source_byte_bindings_fail_closed(inputs):
    prepared, metadata, payload = inputs
    for record, match in ((replace(prepared, binding_seal="0" * 64), "binding_mismatch"),
                          (replace(prepared, sidecar={"record_id": "wrong"}), "record_mismatch")):
        with pytest.raises(pm.PerformanceMemoryError, match=match):
            pm.build_size_index_row(record, metadata=metadata, cache_representation="decoded_dict")
    with pytest.raises(pm.PerformanceMemoryError, match="sidecar_hash_mismatch"):
        pm.build_size_index_row(prepared, metadata=metadata, cache_representation="source_json_bytes",
                               compact_sidecar_bytes=payload + b" ")
    for payload, match in ((b"broken", "invalid_sidecar_json"), (b"[]", "sidecar_mismatch"),
                           (b'{"record_id":"record-a"}', "sidecar_mismatch")):
        with pytest.raises(pm.PerformanceMemoryError, match=match):
            pm.build_size_index_row(prepared, metadata=replace(metadata, sidecar_sha256=sha256(payload).hexdigest()),
                                   cache_representation="source_json_bytes", compact_sidecar_bytes=payload)


def test_bad_raw_graph_cannot_be_promoted_to_index(inputs):
    prepared, metadata, _ = inputs
    prepared.raw_graph["note"].num_nodes = -1
    with pytest.raises((GraphContractError, pm.PerformanceMemoryError)):
        pm.build_size_index_row(prepared, metadata=metadata, cache_representation="decoded_dict")


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_compact_sidecar_cannot_change_integer_into_bool_or_float(inputs, replacement):
    prepared, metadata, _ = inputs
    prepared.sidecar["synthetic_integer"] = 1
    changed = deepcopy(prepared.sidecar)
    changed["synthetic_integer"] = replacement
    payload = json.dumps(changed).encode()
    with pytest.raises(pm.PerformanceMemoryError, match="sidecar_mismatch"):
        pm.build_size_index_row(prepared, metadata=replace(metadata, sidecar_sha256=sha256(payload).hexdigest()),
                               cache_representation="source_json_bytes", compact_sidecar_bytes=payload)


def test_batched_graph_and_wrong_cache_representation_rejected(inputs):
    prepared, metadata, _ = inputs
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([prepared.raw_graph, prepared.raw_graph])
    with pytest.raises(pm.PerformanceMemoryError, match="singleton_required"):
        pm.build_size_index_row(replace(prepared, raw_graph=batch), metadata=metadata,
                               cache_representation="decoded_dict")
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_representation"):
        pm.build_size_index_row(object(), metadata=metadata, cache_representation="unknown")


def test_empty_node_types_are_recorded_as_zero_not_dropped(inputs):
    base = _row(inputs)
    row = _sized(base, "empty", note=0, onset=0, beat=0, bar=0, track=0, edges=0)
    assert row.nodes == row.coarse_length == 1
    result = pm.select_extreme_b2_cases([row], expected_train_ids=["empty"])
    assert result.ordered_b2_cases[0].attention_padding_proxy == 2
    with pytest.raises(pm.PerformanceMemoryError, match="singleton_required"):
        _sized(row, "no-song", song=0)


def test_invalid_relations_and_storage_accounting_rejected(inputs):
    row = _row(inputs)
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_counts"):
        replace(row, edges_by_relation=row.edges_by_relation[:-1])
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_count"):
        replace(row, edges_by_relation=((row.edges_by_relation[0][0], True), *row.edges_by_relation[1:]))
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_accounting"):
        replace(row, accounted_compact_entry_bytes=row.unique_cpu_tensor_storage_bytes - 1)
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_versions"):
        replace(row, graph_versions=(("bad",),))


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_bad_supplied_counts_rejected(inputs, value):
    row = _row(inputs)
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_count"):
        _sized(row, "x", note=value)
    with pytest.raises(pm.PerformanceMemoryError, match="invalid_count"):
        replace(row, accounted_decoded_entry_bytes=value)


def test_every_extreme_axis_is_selected_and_pair_schedule_is_permutation_invariant(inputs):
    base = _row(inputs)
    # Distinct winners avoid an accidental all-axes-use-nodes implementation.
    rows = [
        _sized(base, "note", note=100_000), _sized(base, "onset", onset=100_001),
        _sized(base, "beat", beat=100_002), _sized(base, "track", track=100_003),
        _sized(base, "nodes", note=50_000, onset=50_000, beat=50_000),
        _sized(base, "edges", edges=100_004), _sized(base, "length", bar=100_004),
        _sized(base, "bytes", accounted_compact_entry_bytes=10**10),
        _sized(base, "ordinary"),
    ]
    expected_ids = tuple(row.record_id for row in rows)
    result = pm.select_extreme_b2_cases(rows, expected_train_ids=expected_ids)
    assert {entry.axis: entry.record_id for entry in result.extrema} == dict(zip(pm.EXTREME_AXES,
        ("note", "onset", "beat", "track", "nodes", "edges", "length", "bytes"), strict=True))
    selected = tuple(sorted(set(expected_ids) - {"ordinary"}))
    assert tuple(row.record_id for row in result.selected_rows) == selected
    assert tuple(case.record_ids for case in result.ordered_b2_cases) == tuple(itertools.product(selected, repeat=2))
    assert len(result.ordered_b2_cases) == 64
    assert result.indexed_record_count == 9 and result.indexed_component_count == 1
    for case in result.ordered_b2_cases:
        assert case.attention_padding_proxy == 2 * max(case.coarse_lengths) ** 2
    reversed_result = pm.select_extreme_b2_cases(tuple(reversed(rows)), expected_train_ids=expected_ids[::-1])
    assert asdict(result) == asdict(reversed_result)
    assert result.limits == pm.MEMORY_GATE_LIMITS
    assert any("do not exhaust" in limit for limit in result.limits)
    assert any("OOM" in limit for limit in result.limits)


def test_ties_choose_stable_id_and_deduplicate_to_one_self_pair(inputs):
    base = _row(inputs)
    rows = [_sized(base, "z"), _sized(base, "a"), _sized(base, "m")]
    result = pm.select_extreme_b2_cases(rows, expected_train_ids=("z", "m", "a"))
    assert all(value.record_id == "a" and value.tied_record_ids == ("a", "m", "z") for value in result.extrema)
    assert len(result.selected_rows) == len(result.ordered_b2_cases) == 1
    assert result.ordered_b2_cases[0].record_ids == ("a", "a")


def test_complete_membership_required_and_duplicates_rejected(inputs):
    row = _row(inputs)
    for rows, expected, match in (
        ([], [row.record_id], "empty_index"), ([row], [], "empty_index"),
        ([row], [row.record_id, "missing"], "incomplete_membership"),
        ([row], ["other"], "incomplete_membership"),
        ([row, row], [row.record_id], "duplicate_ids"),
        ([row], [row.record_id, row.record_id], "duplicate_ids"),
    ):
        with pytest.raises(pm.PerformanceMemoryError, match=match):
            pm.select_extreme_b2_cases(rows, expected_train_ids=expected)


def test_index_seal_includes_nonselected_rows_and_preserves_component_count(inputs):
    base = _row(inputs)
    first = _sized(base, "a")
    unselected = _sized(base, "z")
    result = pm.select_extreme_b2_cases([first, unselected], expected_train_ids=("a", "z"))
    changed = replace(unselected, metadata=replace(unselected.metadata, canonical_sha256="0" * 64,
                                                  source_component_id="another-component"))
    amended = pm.select_extreme_b2_cases([first, changed], expected_train_ids=("a", "z"))
    assert result.selected_rows == amended.selected_rows
    assert result.supplied_index_fingerprint != amended.supplied_index_fingerprint
    assert amended.indexed_component_count == 2
    assert result.expected_train_ids_fingerprint == amended.expected_train_ids_fingerprint


def test_selector_rejects_mixed_source_or_cache_policy_and_forged_nontrain(inputs):
    row = _row(inputs)
    other = _sized(row, "other")
    for changed in (replace(other, metadata=replace(other.metadata, source_seal="0" * 64)),
                    replace(other, cache_representation="decoded_dict")):
        with pytest.raises(pm.PerformanceMemoryError, match="mixed_bindings"):
            pm.select_extreme_b2_cases([row, changed], expected_train_ids=(row.record_id, "other"))
    object.__setattr__(other.metadata, "split", "validation")
    with pytest.raises(pm.PerformanceMemoryError, match="train_only"):
        pm.select_extreme_b2_cases([row, other], expected_train_ids=(row.record_id, "other"))
