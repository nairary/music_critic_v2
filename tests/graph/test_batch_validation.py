from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from music_critic.data import CanonicalPiece
from music_critic.graph import (
    BATCH_BASE_NODE_ATTRIBUTES,
    BATCH_CANDIDATE_NODE_ATTRIBUTES,
    BATCH_EDGE_ATTRIBUTES,
    BATCH_GLOBAL_ATTRIBUTES,
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    GraphContractError,
    build_raw_graph,
    validate_raw_graph_batch,
)


def _batch(canonical_piece: CanonicalPiece) -> Batch:
    return Batch.from_data_list(
        [build_raw_graph(canonical_piece), build_raw_graph(canonical_piece)]
    )


def test_real_pyg_batch_matches_exact_raw_allowlists(
    canonical_piece: CanonicalPiece,
) -> None:
    batch = _batch(canonical_piece)
    validate_raw_graph_batch(batch, sample_count=2)
    assert set(batch._global_store.keys()) == BATCH_GLOBAL_ATTRIBUTES
    assert tuple(batch.node_types) == MANDATORY_NODE_TYPES
    assert tuple(batch.edge_types) == MANDATORY_EDGE_TYPES
    for node_type in MANDATORY_NODE_TYPES:
        expected = (
            BATCH_CANDIDATE_NODE_ATTRIBUTES
            if node_type in {"beat", "onset"}
            else BATCH_BASE_NODE_ATTRIBUTES
        )
        assert set(batch[node_type].keys()) == expected
    for edge_type in MANDATORY_EDGE_TYPES:
        assert set(batch[edge_type].keys()) == BATCH_EDGE_ATTRIBUTES
    with pytest.raises(GraphContractError, match="graph count differs"):
        validate_raw_graph_batch(batch, sample_count=1)


@pytest.mark.parametrize(
    "field",
    (
        "target",
        "targets",
        "label",
        "labels",
        "y",
        "gold_chord",
        "theory",
        "split",
        "source_group_id",
        "lineage_group_id",
        "provenance",
        "arbitrary_unknown_field",
    ),
)
def test_batch_rejects_every_unknown_global_field(
    canonical_piece: CanonicalPiece,
    field: str,
) -> None:
    batch = _batch(canonical_piece)
    setattr(batch, field, torch.zeros(2))
    with pytest.raises(GraphContractError, match="batch global attributes differ"):
        validate_raw_graph_batch(batch, sample_count=2)


def test_batch_rejects_unknown_node_and_edge_fields(
    canonical_piece: CanonicalPiece,
) -> None:
    node_batch = _batch(canonical_piece)
    node_batch["note"].gold_chord = torch.zeros(
        node_batch["note"].num_nodes,
        dtype=torch.long,
    )
    with pytest.raises(GraphContractError, match="node store.*attributes differ"):
        validate_raw_graph_batch(node_batch, sample_count=2)

    edge_batch = _batch(canonical_piece)
    edge_batch[MANDATORY_EDGE_TYPES[0]].labels = torch.zeros(
        edge_batch[MANDATORY_EDGE_TYPES[0]].edge_index.shape[1],
        dtype=torch.long,
    )
    with pytest.raises(GraphContractError, match="edge store.*attributes differ"):
        validate_raw_graph_batch(edge_batch, sample_count=2)


def test_batch_rejects_false_raw_only_in_one_source_graph(
    canonical_piece: CanonicalPiece,
) -> None:
    first = build_raw_graph(canonical_piece)
    second = build_raw_graph(canonical_piece)
    second.raw_only = False
    batch = Batch.from_data_list([first, second])
    assert batch.raw_only.tolist() == [True, False]
    with pytest.raises(GraphContractError, match="one True value per source graph"):
        validate_raw_graph_batch(batch, sample_count=2)


@pytest.mark.parametrize(
    "raw_only",
    (
        pytest.param([True, None], id="null"),
        pytest.param(torch.tensor([1, 1], dtype=torch.long), id="wrong-dtype"),
        pytest.param(torch.tensor([True], dtype=torch.bool), id="wrong-length"),
        pytest.param(torch.tensor([[True, True]], dtype=torch.bool), id="wrong-rank"),
    ),
)
def test_batch_rejects_invalid_raw_only_representation(
    canonical_piece: CanonicalPiece,
    raw_only: object,
) -> None:
    batch = _batch(canonical_piece)
    batch.raw_only = raw_only
    with pytest.raises(GraphContractError, match="rank-one bool tensor"):
        validate_raw_graph_batch(batch, sample_count=2)


@pytest.mark.parametrize("attribute", ("batch", "ptr"))
def test_batch_rejects_corrupted_batch_or_ptr(
    canonical_piece: CanonicalPiece,
    attribute: str,
) -> None:
    batch = _batch(canonical_piece)
    if attribute == "batch":
        corrupted = batch["note"].batch.clone()
        corrupted[-1] = 0
    else:
        corrupted = batch["note"].ptr.clone()
        corrupted[-1] -= 1
    setattr(batch["note"], attribute, corrupted)
    with pytest.raises(GraphContractError, match=r"note\.(batch|ptr)"):
        validate_raw_graph_batch(batch, sample_count=2)


@pytest.mark.parametrize(
    ("attribute", "production_value"),
    (
        ("schema_version", "2.0.0"),
        ("graph_schema_version", GRAPH_SCHEMA_VERSION),
        ("feature_registry_version", RAW_FEATURE_REGISTRY.version),
        ("graph_builder_version", GRAPH_BUILDER_VERSION),
    ),
)
def test_batch_rejects_incompatible_version_metadata(
    canonical_piece: CanonicalPiece,
    attribute: str,
    production_value: str,
) -> None:
    batch = _batch(canonical_piece)
    setattr(batch, attribute, [production_value, "incompatible"])
    with pytest.raises(GraphContractError, match=f"metadata {attribute!r}"):
        validate_raw_graph_batch(batch, sample_count=2)


def test_batch_rejects_cross_graph_or_out_of_range_edge(
    canonical_piece: CanonicalPiece,
) -> None:
    edge_type = MANDATORY_EDGE_TYPES[0]
    cross_graph = _batch(canonical_piece)
    edge_index = cross_graph[edge_type].edge_index.clone()
    edge_index[0, 0] = cross_graph[edge_type[0]].ptr[1]
    cross_graph[edge_type].edge_index = edge_index
    with pytest.raises(GraphContractError, match="connects different source graphs"):
        validate_raw_graph_batch(cross_graph, sample_count=2)

    out_of_range = _batch(canonical_piece)
    edge_index = out_of_range[edge_type].edge_index.clone()
    edge_index[0, 0] = out_of_range[edge_type[0]].num_nodes
    out_of_range[edge_type].edge_index = edge_index
    with pytest.raises(GraphContractError, match="source endpoint is out of range"):
        validate_raw_graph_batch(out_of_range, sample_count=2)
