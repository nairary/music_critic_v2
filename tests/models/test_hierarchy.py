from __future__ import annotations

from dataclasses import replace
import inspect
import re

import pytest
import torch
from torch_geometric.data import HeteroData

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models import (
    COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION,
    ENCODER_OUTPUT_VERSION,
    HIERARCHICAL_ENCODER_OUTPUT_VERSION,
    HIERARCHICAL_MODEL_CONTRACT_VERSION,
    HIERARCHY_POOLING_CONTRACT_VERSION,
    TOP_DOWN_FUSION_CONTRACT_VERSION,
    CoarseMusicTransformer,
    DeterministicHierarchyPool,
    EncoderOutput,
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    HierarchyContractError,
    HierarchyOwnership,
    HierarchyPoolingOutput,
    extract_hierarchy_ownership,
    validate_hierarchy_graph_structure,
)
from music_critic.models import hierarchy as hierarchy_module


def _config(**changes) -> HierarchicalBaselineConfig:
    return HierarchicalBaselineConfig(
        hidden_dim=16,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        **changes,
    )


def _model(**changes) -> HierarchicalHeterogeneousBaseline:
    return HierarchicalHeterogeneousBaseline(_config(**changes))


def test_phase6b_contract_versions_and_configuration() -> None:
    assert HIERARCHY_POOLING_CONTRACT_VERSION == "1.0.0"
    assert COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION == "1.0.0"
    assert HIERARCHICAL_ENCODER_OUTPUT_VERSION == "1.0.0"
    assert TOP_DOWN_FUSION_CONTRACT_VERSION == "1.0.0"
    assert HIERARCHICAL_MODEL_CONTRACT_VERSION == "1.0.0"
    with pytest.raises(ValueError, match="divisible"):
        HierarchicalBaselineConfig(hidden_dim=30, attention_heads=4)
    with pytest.raises(ValueError, match="local_gnn_layers"):
        HierarchicalBaselineConfig(local_gnn_layers=0)
    with pytest.raises(ValueError, match="transformer_layers"):
        HierarchicalBaselineConfig(transformer_layers=0)


def test_deterministic_ownership_matches_all_raw_reverse_relations(
    mixed_batch,
) -> None:
    model = _model()
    local = model.local_baseline.encode(mixed_batch.raw_graph_batch)
    ownership = extract_hierarchy_ownership(
        mixed_batch.raw_graph_batch, local.final_output
    )
    relation_by_name = {
        "beat_to_bar": ("beat", "belongs_to_bar", "bar"),
        "onset_to_bar": ("onset", "belongs_to_bar", "bar"),
        "note_to_bar": ("note", "belongs_to_bar", "bar"),
        "note_to_track": ("note", "belongs_to_track", "track"),
        "bar_to_song": ("bar", "belongs_to_song", "song"),
        "track_to_song": ("track", "belongs_to_song", "song"),
    }
    for name, edge_type in relation_by_name.items():
        edge_index = mixed_batch.raw_graph_batch[edge_type].edge_index
        assert torch.equal(edge_index[0], torch.arange(edge_index.shape[1]))
        assert torch.equal(ownership.owners[name], edge_index[1])
    assert ownership.sample_count == 3


def _graph_store_snapshot(graph: HeteroData) -> tuple[object, ...]:
    node_types = tuple(graph.node_types)
    edge_types = tuple(graph.edge_types)
    return (
        node_types,
        edge_types,
        tuple(
            (node_type, tuple(sorted(graph[node_type].keys())))
            for node_type in node_types
        ),
        tuple(
            (edge_type, tuple(sorted(graph[edge_type].keys())))
            for edge_type in edge_types
        ),
    )


def test_hierarchy_extraction_rejects_invalid_input_type() -> None:
    with pytest.raises(
        HierarchyContractError,
        match="^hierarchy.input_type_invalid$",
    ):
        extract_hierarchy_ownership({})


@pytest.mark.parametrize(
    ("edge_type", "category"),
    (
        (
            ("beat", "belongs_to_bar", "bar"),
            "hierarchy.edge_store_missing:beat_to_bar:ownership",
        ),
        (
            ("bar", "contains_beat", "beat"),
            "hierarchy.edge_store_missing:beat_to_bar:containment",
        ),
    ),
)
def test_missing_ownership_forward_or_reverse_store_is_structured_and_atomic(
    mixed_batch, edge_type, category
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    del graph[edge_type]
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError, match=f"^{re.escape(category)}$"
    ):
        extract_hierarchy_ownership(graph)
    assert _graph_store_snapshot(graph) == snapshot


def test_completely_absent_ownership_relation_pair_is_structured_and_atomic(
    mixed_batch,
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    del graph[("note", "belongs_to_track", "track")]
    del graph[("track", "contains_note", "note")]
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError,
        match=(
            "^hierarchy.edge_store_missing:"
            "note_to_track:ownership$"
        ),
    ):
        validate_hierarchy_graph_structure(graph)
    assert _graph_store_snapshot(graph) == snapshot


def test_missing_node_store_is_structured_and_does_not_create_stores(
    mixed_batch,
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    del graph["bar"]
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError,
        match="^hierarchy.node_store_missing:bar$",
    ):
        extract_hierarchy_ownership(graph)
    assert _graph_store_snapshot(graph) == snapshot


def test_missing_edge_index_is_structured_and_does_not_mutate_attributes(
    mixed_batch,
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    edge_type = ("note", "belongs_to_bar", "bar")
    del graph[edge_type]["edge_index"]
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError,
        match=(
            "^hierarchy.edge_index_missing:"
            "note_to_bar:ownership$"
        ),
    ):
        extract_hierarchy_ownership(graph)
    assert _graph_store_snapshot(graph) == snapshot


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "owner_missing"),
        ("duplicate", "owner_duplicate"),
        ("reordered", "child_reordered"),
        ("cross_sample", "cross_sample_ownership"),
    ),
)
def test_missing_duplicate_and_cross_sample_ownership_rejected(
    mixed_batch, corruption: str, message: str
) -> None:
    model = _model()
    local = model.local_baseline.encode(mixed_batch.raw_graph_batch)
    graph = mixed_batch.raw_graph_batch.clone()
    reverse_type = ("beat", "belongs_to_bar", "bar")
    forward_type = ("bar", "contains_beat", "beat")
    reverse = graph[reverse_type].edge_index.clone()
    if corruption == "missing":
        reverse = reverse[:, 1:]
    elif corruption == "duplicate":
        reverse[0, 1] = reverse[0, 0]
    elif corruption == "reordered":
        reverse = reverse[:, torch.arange(reverse.shape[1] - 1, -1, -1)]
    else:
        beat_membership = local.final_output.batch_membership["beat"]
        bar_membership = local.final_output.batch_membership["bar"]
        beat = int(torch.nonzero(beat_membership == 0)[0])
        other_bar = int(torch.nonzero(bar_membership == 1)[0])
        reverse[1, beat] = other_bar
    graph[reverse_type].edge_index = reverse
    graph[forward_type].edge_index = reverse.flip(0)
    with pytest.raises(HierarchyContractError, match=message):
        extract_hierarchy_ownership(graph, local.final_output)
    with pytest.raises(HierarchyContractError, match=message):
        model.encode(graph)


@pytest.mark.parametrize(
    ("transform", "category"),
    (
        (
            lambda values: values.to(torch.int32),
            "edge_index_dtype_invalid",
        ),
        (
            lambda values: values.flatten(),
            "edge_index_rank_invalid",
        ),
        (
            lambda values: values[:1],
            "edge_index_shape_invalid",
        ),
    ),
)
def test_invalid_raw_owner_tensor_contract_has_stable_category(
    mixed_batch, transform, category
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    edge_type = ("beat", "belongs_to_bar", "bar")
    graph[edge_type].edge_index = transform(graph[edge_type].edge_index)
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError,
        match=(
            f"^hierarchy.{category}:beat_to_bar:ownership$"
        ),
    ):
        extract_hierarchy_ownership(graph)
    assert _graph_store_snapshot(graph) == snapshot


def test_forward_reverse_ownership_mismatch_has_stable_category(
    mixed_batch,
) -> None:
    graph = mixed_batch.raw_graph_batch.clone()
    edge_type = ("bar", "contains_beat", "beat")
    changed = graph[edge_type].edge_index.clone()
    changed[0, 0] = (changed[0, 0] + 1) % graph["bar"].num_nodes
    graph[edge_type].edge_index = changed
    snapshot = _graph_store_snapshot(graph)
    with pytest.raises(
        HierarchyContractError,
        match=(
            "^hierarchy.reverse_containment_mismatch:beat_to_bar$"
        ),
    ):
        extract_hierarchy_ownership(graph)
    assert _graph_store_snapshot(graph) == snapshot


def _replace_owner(
    ownership: HierarchyOwnership,
    name: str,
    values: torch.Tensor,
) -> HierarchyOwnership:
    owners = dict(ownership.owners)
    owners[name] = values
    return replace(ownership, owners=owners)


@pytest.mark.parametrize(
    ("forge", "category"),
    (
        (
            lambda values, _parent_count: values.to(torch.float32),
            "ownership_owner_dtype_invalid",
        ),
        (
            lambda values, _parent_count: values.unsqueeze(0),
            "ownership_owner_rank_invalid",
        ),
        (
            lambda values, _parent_count: values[:-1],
            "ownership_owner_shape_invalid",
        ),
        (
            lambda values, parent_count: torch.full_like(
                values, parent_count
            ),
            "ownership_owner_out_of_range",
        ),
        (
            lambda values, _parent_count: torch.empty_like(
                values, device="meta"
            ),
            "ownership_owner_device_mismatch",
        ),
    ),
)
def test_forged_precomputed_ownership_is_revalidated(
    mixed_batch, forge, category
) -> None:
    model = _model()
    graph = mixed_batch.raw_graph_batch
    local = model.local_baseline.encode(graph)
    expected = extract_hierarchy_ownership(graph, local.final_output)
    name = "beat_to_bar"
    values = expected.owners[name]
    parent_count = local.final_output.embeddings["bar"].shape[0]
    forged = _replace_owner(
        expected, name, forge(values, parent_count)
    )
    with pytest.raises(
        HierarchyContractError,
        match=f"^hierarchy.{category}:{name}$",
    ):
        model.context_encoder(local, graph, ownership=forged)


def test_forged_in_range_owner_must_equal_raw_relation(mixed_batch) -> None:
    model = _model()
    graph = mixed_batch.raw_graph_batch
    local = model.local_baseline.encode(graph)
    expected = extract_hierarchy_ownership(graph, local.final_output)
    name = "note_to_track"
    owners = expected.owners[name].clone()
    note_membership = expected.batch_membership["note"]
    track_membership = expected.batch_membership["track"]
    candidate = torch.nonzero(note_membership == 1, as_tuple=False)[0, 0]
    same_sample_tracks = torch.nonzero(
        track_membership == 1, as_tuple=False
    ).flatten()
    assert same_sample_tracks.numel() >= 2
    owners[candidate] = same_sample_tracks[
        int(owners[candidate] == same_sample_tracks[0])
    ]
    forged = _replace_owner(expected, name, owners)
    with pytest.raises(
        HierarchyContractError,
        match="^hierarchy.ownership_graph_mismatch:note_to_track$",
    ):
        model.context_encoder(local, graph, ownership=forged)


def test_forged_cross_sample_precomputed_owner_is_rejected(
    mixed_batch,
) -> None:
    model = _model()
    graph = mixed_batch.raw_graph_batch
    local = model.local_baseline.encode(graph)
    expected = extract_hierarchy_ownership(graph, local.final_output)
    owners = expected.owners["beat_to_bar"].clone()
    other_sample_bar = torch.nonzero(
        expected.batch_membership["bar"] == 1, as_tuple=False
    )[0, 0]
    owners[0] = other_sample_bar
    forged = _replace_owner(expected, "beat_to_bar", owners)
    with pytest.raises(
        HierarchyContractError,
        match="^hierarchy.ownership_cross_sample:beat_to_bar$",
    ):
        model.context_encoder(local, graph, ownership=forged)


def test_standard_model_path_scans_each_ownership_relation_once(
    mixed_batch, monkeypatch
) -> None:
    calls = 0
    original = hierarchy_module._owner_index

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hierarchy_module, "_owner_index", counted)
    _model().eval().encode(mixed_batch.raw_graph_batch)
    assert calls == 6


def _empty_group_local(hidden_dim: int = 8):
    counts = {
        "song": 1,
        "track": 2,
        "bar": 2,
        "beat": 1,
        "onset": 1,
        "note": 1,
    }
    embeddings = {
        node_type: torch.arange(count * hidden_dim, dtype=torch.float32).reshape(
            count, hidden_dim
        )
        for node_type, count in counts.items()
    }
    membership = {
        node_type: torch.zeros(count, dtype=torch.long)
        for node_type, count in counts.items()
    }
    local = EncoderOutput(
        contract_version=ENCODER_OUTPUT_VERSION,
        embeddings=embeddings,
        batch_membership=membership,
    )
    ownership = HierarchyOwnership(
        contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
        sample_count=1,
        owners={
            "beat_to_bar": torch.tensor([0]),
            "onset_to_bar": torch.tensor([0]),
            "note_to_bar": torch.tensor([0]),
            "note_to_track": torch.tensor([0]),
            "bar_to_song": torch.tensor([0, 0]),
            "track_to_song": torch.tensor([0, 0]),
        },
        batch_membership=membership,
    )
    return local, ownership


def test_pooling_retains_shapes_counts_and_empty_availability() -> None:
    local, ownership = _empty_group_local()
    pooling = DeterministicHierarchyPool(8, 0.0)(local, ownership)
    assert pooling.bar_tokens.shape == (2, 8)
    assert pooling.track_tokens.shape == (2, 8)
    for name in ("bar_beats", "bar_onsets", "bar_notes"):
        assert pooling.child_counts[name].tolist() == [1, 0]
        assert pooling.child_available[name].tolist() == [True, False]
    assert pooling.child_counts["track_notes"].tolist() == [1, 0]
    assert pooling.child_available["track_notes"].tolist() == [True, False]
    assert torch.isfinite(pooling.bar_tokens).all()
    assert torch.isfinite(pooling.track_tokens).all()


def test_pooling_source_has_no_dense_node_group_membership_matrix() -> None:
    source = inspect.getsource(DeterministicHierarchyPool)
    helper_source = inspect.getsource(
        __import__(
            "music_critic.models.hierarchy",
            fromlist=["_scatter_family_statistics"],
        )._scatter_family_statistics
    )
    combined = source + helper_source
    assert "one_hot" not in combined
    assert "to_dense" not in combined
    assert "index_add_" in combined
    assert "scatter_reduce_" in combined


def _uneven_sequence_inputs(
    *,
    device: torch.device | str = "cpu",
    hidden_dim: int = 16,
) -> tuple[EncoderOutput, HierarchyPoolingOutput]:
    memberships = {
        "song": torch.arange(3, dtype=torch.long, device=device),
        "track": torch.repeat_interleave(
            torch.arange(3, device=device),
            torch.tensor([2, 1, 3], device=device),
        ),
        "bar": torch.repeat_interleave(
            torch.arange(3, device=device),
            torch.tensor([1, 3, 2], device=device),
        ),
        "beat": torch.empty(0, dtype=torch.long, device=device),
        "onset": torch.empty(0, dtype=torch.long, device=device),
        "note": torch.empty(0, dtype=torch.long, device=device),
    }
    embeddings = {
        node_type: torch.randn(
            membership.shape[0],
            hidden_dim,
            device=device,
            requires_grad=node_type == "song",
        )
        for node_type, membership in memberships.items()
    }
    local = EncoderOutput(
        contract_version=ENCODER_OUTPUT_VERSION,
        embeddings=embeddings,
        batch_membership=memberships,
    )
    empty = torch.empty(0, dtype=torch.long, device=device)
    ownership = HierarchyOwnership(
        contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
        sample_count=3,
        owners={
            "beat_to_bar": empty,
            "onset_to_bar": empty,
            "note_to_bar": empty,
            "note_to_track": empty,
            "bar_to_song": memberships["bar"],
            "track_to_song": memberships["track"],
        },
        batch_membership=memberships,
    )
    bar_tokens = torch.randn(
        memberships["bar"].shape[0],
        hidden_dim,
        device=device,
        requires_grad=True,
    )
    track_tokens = torch.randn(
        memberships["track"].shape[0],
        hidden_dim,
        device=device,
        requires_grad=True,
    )
    child_counts = {
        "bar_beats": torch.zeros(
            bar_tokens.shape[0], dtype=torch.long, device=device
        ),
        "bar_onsets": torch.zeros(
            bar_tokens.shape[0], dtype=torch.long, device=device
        ),
        "bar_notes": torch.zeros(
            bar_tokens.shape[0], dtype=torch.long, device=device
        ),
        "track_notes": torch.zeros(
            track_tokens.shape[0], dtype=torch.long, device=device
        ),
    }
    return local, HierarchyPoolingOutput(
        contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
        bar_tokens=bar_tokens,
        track_tokens=track_tokens,
        child_counts=child_counts,
        child_available={
            name: values > 0 for name, values in child_counts.items()
        },
        ownership=ownership,
    )


def _reference_sinusoidal(
    ordinals: torch.Tensor, hidden_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    positions = ordinals.to(dtype).unsqueeze(-1)
    exponent = torch.arange(
        (hidden_dim + 1) // 2,
        dtype=dtype,
        device=ordinals.device,
    )
    angles = positions / torch.pow(
        torch.tensor(10_000.0, dtype=dtype, device=ordinals.device),
        (2 * exponent) / hidden_dim,
    )
    encoded = torch.zeros(
        (ordinals.shape[0], hidden_dim),
        dtype=dtype,
        device=ordinals.device,
    )
    encoded[:, 0::2] = torch.sin(angles[:, : encoded[:, 0::2].shape[1]])
    encoded[:, 1::2] = torch.cos(angles[:, : encoded[:, 1::2].shape[1]])
    return encoded


def _reference_sequence(
    transformer: CoarseMusicTransformer,
    local: EncoderOutput,
    pooling: HierarchyPoolingOutput,
) -> tuple[torch.Tensor, ...]:
    bar_membership = pooling.ownership.batch_membership["bar"].tolist()
    track_membership = pooling.ownership.batch_membership["track"].tolist()
    bar_rows = [
        [index for index, sample in enumerate(bar_membership) if sample == row]
        for row in range(pooling.ownership.sample_count)
    ]
    track_rows = [
        [
            index
            for index, sample in enumerate(track_membership)
            if sample == row
        ]
        for row in range(pooling.ownership.sample_count)
    ]
    lengths = torch.tensor(
        [
            1 + len(bar_rows[row]) + len(track_rows[row])
            for row in range(pooling.ownership.sample_count)
        ],
        dtype=torch.long,
        device=pooling.bar_tokens.device,
    )
    maximum = int(lengths.max())
    tokens = pooling.bar_tokens.new_zeros(
        (pooling.ownership.sample_count, maximum, transformer.hidden_dim)
    )
    padding = torch.ones(
        (pooling.ownership.sample_count, maximum),
        dtype=torch.bool,
        device=tokens.device,
    )
    types = torch.full_like(padding, -1, dtype=torch.long)
    ordinals = torch.full_like(types, -1)
    bar_positions = torch.empty(
        pooling.bar_tokens.shape[0], dtype=torch.long, device=tokens.device
    )
    track_positions = torch.empty(
        pooling.track_tokens.shape[0], dtype=torch.long, device=tokens.device
    )
    for sample in range(pooling.ownership.sample_count):
        tokens[sample, 0] = local.embeddings["song"][sample]
        types[sample, 0] = 0
        ordinals[sample, 0] = 0
        cursor = 1
        for ordinal, row in enumerate(bar_rows[sample]):
            tokens[sample, cursor] = pooling.bar_tokens[row]
            types[sample, cursor] = 1
            ordinals[sample, cursor] = ordinal
            bar_positions[row] = cursor
            cursor += 1
        for ordinal, row in enumerate(track_rows[sample]):
            tokens[sample, cursor] = pooling.track_tokens[row]
            types[sample, cursor] = 2
            ordinals[sample, cursor] = ordinal
            track_positions[row] = cursor
            cursor += 1
        padding[sample, :cursor] = False
    tokens = tokens + transformer.type_embedding(types.clamp_min(0))
    positional = _reference_sinusoidal(
        ordinals.clamp_min(0).reshape(-1),
        transformer.hidden_dim,
        tokens.dtype,
    ).reshape_as(tokens)
    tokens = tokens + torch.where(
        (~padding & (types != 0)).unsqueeze(-1),
        positional,
        torch.zeros_like(tokens),
    )
    tokens = torch.where(
        padding.unsqueeze(-1), torch.zeros_like(tokens), tokens
    )
    return (
        tokens,
        padding,
        types,
        ordinals,
        lengths,
        bar_positions,
        track_positions,
    )


def test_tensorized_uneven_sequence_matches_test_only_reference() -> None:
    torch.manual_seed(691)
    transformer = CoarseMusicTransformer(_config()).eval()
    local, pooling = _uneven_sequence_inputs()
    actual = transformer.build_sequence(local, pooling)
    expected = _reference_sequence(transformer, local, pooling)
    assert actual.sequence_lengths.tolist() == [4, 5, 6]
    for observed, reference in zip(
        (
            actual.tokens,
            actual.padding_mask,
            actual.type_codes,
            actual.ordinals,
            actual.sequence_lengths,
            actual.bar_positions,
            actual.track_positions,
        ),
        expected,
    ):
        assert torch.equal(observed, reference)


def test_tensorized_sequence_packing_preserves_backward_gradients() -> None:
    torch.manual_seed(692)
    transformer = CoarseMusicTransformer(_config())
    local, pooling = _uneven_sequence_inputs()
    sequence = transformer.build_sequence(local, pooling)
    sequence.tokens.square().sum().backward()
    assert torch.count_nonzero(local.embeddings["song"].grad)
    assert torch.count_nonzero(pooling.bar_tokens.grad)
    assert torch.count_nonzero(pooling.track_tokens.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tensorized_sequence_cuda_smoke_and_reference_parity() -> None:
    torch.manual_seed(693)
    device = torch.device("cuda")
    transformer = CoarseMusicTransformer(_config()).to(device).eval()
    local, pooling = _uneven_sequence_inputs(device=device)
    actual = transformer.build_sequence(local, pooling)
    expected = _reference_sequence(transformer, local, pooling)
    assert actual.tokens.is_cuda
    assert torch.equal(actual.tokens, expected[0])
    actual.tokens.sum().backward()
    assert pooling.bar_tokens.grad is not None
    assert pooling.track_tokens.grad is not None


def test_production_packing_has_no_per_row_host_materialization() -> None:
    packing_source = (
        inspect.getsource(CoarseMusicTransformer.build_sequence)
        + inspect.getsource(hierarchy_module._coarse_family_layout)
    )
    for forbidden in (".item()", ".tolist()", ".cpu()"):
        assert forbidden not in packing_source
    assert "for sample" not in packing_source
    allocation_source = inspect.getsource(
        hierarchy_module._maximum_padded_length
    )
    assert allocation_source.count(".item()") == 1
    assert ".tolist()" not in allocation_source
    assert ".cpu()" not in allocation_source


def test_context_output_retains_local_rows_and_fusion_cardinality(
    mixed_batch,
) -> None:
    output = _model().eval().encode(
        mixed_batch.raw_graph_batch, return_layers=True
    )
    assert output.local_encoder.layer_outputs
    for node_type in MANDATORY_NODE_TYPES:
        local = output.local_encoder.final_output
        assert output.fused.embeddings[node_type].shape == (
            local.embeddings[node_type].shape
        )
        assert torch.equal(
            output.fused.batch_membership[node_type],
            local.batch_membership[node_type],
        )
        assert output.fused.embeddings[node_type].data_ptr() != (
            local.embeddings[node_type].data_ptr()
        )
    assert output.coarse.song_embeddings.shape == (3, 16)
    assert output.coarse.bar_embeddings.shape[0] == (
        mixed_batch.raw_graph_batch["bar"].num_nodes
    )
    assert output.coarse.track_embeddings.shape[0] == (
        mixed_batch.raw_graph_batch["track"].num_nodes
    )


def test_transformer_has_no_cross_sample_attention(mixed_batch) -> None:
    torch.manual_seed(701)
    model = _model().eval()
    local = model.local_baseline.encode(mixed_batch.raw_graph_batch)
    ownership = extract_hierarchy_ownership(
        mixed_batch.raw_graph_batch, local.final_output
    )
    pooling = model.context_encoder.pooling(local.final_output, ownership)
    baseline = model.context_encoder.transformer(
        local.final_output, pooling
    )
    bar_tokens = pooling.bar_tokens.clone()
    track_tokens = pooling.track_tokens.clone()
    bar_tokens[ownership.batch_membership["bar"] == 1] += 1000
    track_tokens[ownership.batch_membership["track"] == 1] -= 1000
    changed = model.context_encoder.transformer(
        local.final_output,
        replace(
            pooling,
            bar_tokens=bar_tokens,
            track_tokens=track_tokens,
        ),
    )
    assert torch.equal(
        baseline.song_embeddings[0], changed.song_embeddings[0]
    )
    for node_type, before, after in (
        (
            "bar",
            baseline.bar_embeddings,
            changed.bar_embeddings,
        ),
        (
            "track",
            baseline.track_embeddings,
            changed.track_embeddings,
        ),
    ):
        sample_zero = ownership.batch_membership[node_type] == 0
        assert torch.equal(before[sample_zero], after[sample_zero])


def test_padding_mask_makes_valid_transformer_rows_padding_invariant() -> None:
    torch.manual_seed(703)
    transformer = CoarseMusicTransformer(_config()).eval()
    tokens = torch.randn(2, 5, 16)
    padding = torch.tensor(
        [[False, False, True, True, True], [False, False, False, False, False]]
    )
    changed = tokens.clone()
    changed[padding] = torch.randn_like(changed[padding]) * 1000
    before = transformer.encoder(tokens, src_key_padding_mask=padding)
    after = transformer.encoder(changed, src_key_padding_mask=padding)
    assert torch.equal(before[0, :2], after[0, :2])


def test_eval_forward_is_deterministic(mixed_batch) -> None:
    torch.manual_seed(709)
    model = _model().eval()
    first = model(mixed_batch, include_reconstruction=False)
    second = model(mixed_batch, include_reconstruction=False)
    assert torch.equal(
        first.encoder.coarse.song_embeddings,
        second.encoder.coarse.song_embeddings,
    )
    assert all(
        torch.equal(left.logits, right.logits)
        for left, right in zip(first.predictions, second.predictions)
    )
