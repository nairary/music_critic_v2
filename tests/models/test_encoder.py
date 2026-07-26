from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    GraphContractError,
)
from music_critic.models import (
    ENCODER_OUTPUT_VERSION,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
)


@pytest.mark.parametrize("variant", ("feature_only", "local_gnn"))
def test_variants_preserve_every_local_row(mixed_batch, variant: str) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant=variant,
            hidden_dim=16,
            gnn_layers=2,
            dropout=0.0,
        )
    )
    output = model.encode(mixed_batch.raw_graph_batch, return_layers=True)
    assert output.contract_version == ENCODER_OUTPUT_VERSION
    assert tuple(output.final_output.embeddings) == MANDATORY_NODE_TYPES
    assert len(output.layer_outputs) == (0 if variant == "feature_only" else 2)
    for node_type in MANDATORY_NODE_TYPES:
        expected = mixed_batch.raw_graph_batch[node_type].num_nodes
        assert output.feature_output.embeddings[node_type].shape == (expected, 16)
        assert output.final_output.embeddings[node_type].shape == (expected, 16)
        assert torch.equal(
            output.final_output.batch_membership[node_type],
            mixed_batch.raw_graph_batch[node_type].batch,
        )


def test_local_gnn_has_one_distinct_projection_for_every_relation() -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=2, dropout=0.0)
    )
    assert len(model.encoder.layers) == 2
    for layer in model.encoder.layers:
        assert len(layer.relation_projections) == len(MANDATORY_EDGE_TYPES)
        assert len({id(module) for module in layer.relation_projections.values()}) == (
            len(MANDATORY_EDGE_TYPES)
        )


def test_availability_is_not_encoded_as_observed_zero(mixed_batch) -> None:
    torch.manual_seed(7)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant="feature_only",
            hidden_dim=16,
            gnn_layers=1,
            dropout=0.0,
        )
    )
    graph = deepcopy(mixed_batch.raw_graph_batch)
    before = model.encode(graph).feature_output.embeddings["song"].clone()
    column = RAW_FEATURE_REGISTRY.names("song", "continuous").index(
        "duration_qn"
    )
    graph["song"].x_cont[0, column] = 0.0
    graph["song"].x_cont_available[0, column] = False
    after = model.encoder.feature_encoder.node_encoders["song"](graph["song"])
    assert not torch.equal(before[0], after[0])


def test_forward_rejects_stale_graph_contract(mixed_batch) -> None:
    graph = deepcopy(mixed_batch.raw_graph_batch)
    graph.feature_registry_version = "stale"
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    with pytest.raises(GraphContractError):
        model.encode(graph)


def test_empty_relation_stores_are_supported(mixed_batch) -> None:
    graph = mixed_batch.raw_graph_batch
    assert any(graph[edge_type].edge_index.shape[1] == 0 for edge_type in MANDATORY_EDGE_TYPES)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    output = model.encode(graph)
    assert all(torch.isfinite(value).all() for value in output.final_output.embeddings.values())
