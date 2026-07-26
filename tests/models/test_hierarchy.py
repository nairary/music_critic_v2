from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

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
    extract_hierarchy_ownership,
)


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


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing", "missing_duplicate_or_unordered"),
        ("duplicate", "missing_duplicate_or_unordered"),
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
