"""Validation for the model-facing Phase 3A heterograph contract."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from music_critic.graph.feature_registry import RAW_FEATURE_REGISTRY, FeatureRegistry
from music_critic.graph.relations import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    REVERSE_EDGE_TYPES,
)


class GraphContractError(ValueError):
    """Raised when a graph does not satisfy the stable raw contract."""


def validate_raw_graph(
    graph: HeteroData,
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> None:
    """Raise ``GraphContractError`` for any schema, feature, or edge violation."""

    if not isinstance(graph, HeteroData):
        raise GraphContractError("graph must be torch_geometric.data.HeteroData")
    if tuple(graph.node_types) != MANDATORY_NODE_TYPES:
        raise GraphContractError(
            f"node types must be exactly {MANDATORY_NODE_TYPES}, got {graph.node_types}"
        )
    if tuple(graph.edge_types) != MANDATORY_EDGE_TYPES:
        raise GraphContractError("edge types or edge ordering differ from the contract")

    expected_metadata = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "feature_registry_version": registry.version,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "raw_only": True,
    }
    for name, expected in expected_metadata.items():
        if getattr(graph, name, None) != expected:
            raise GraphContractError(
                f"graph metadata {name!r} must be {expected!r}"
            )
    schema_version = getattr(graph, "schema_version", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise GraphContractError("graph must store its canonical schema version")

    for node_type in MANDATORY_NODE_TYPES:
        store = graph[node_type]
        count = store.num_nodes
        if not isinstance(count, int) or count < 0:
            raise GraphContractError(f"{node_type}.num_nodes is invalid")
        categorical = registry.for_node(node_type, "categorical")
        continuous = registry.for_node(node_type, "continuous")
        expected_shapes = {
            "x_cat": (count, len(categorical)),
            "x_cat_available": (count, len(categorical)),
            "x_cont": (count, len(continuous)),
            "x_cont_available": (count, len(continuous)),
        }
        for name, shape in expected_shapes.items():
            value = getattr(store, name, None)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise GraphContractError(
                    f"{node_type}.{name} must have shape {shape}"
                )
        if store.x_cat.dtype != torch.long:
            raise GraphContractError(f"{node_type}.x_cat must use torch.long")
        if store.x_cont.dtype != torch.float32:
            raise GraphContractError(f"{node_type}.x_cont must use torch.float32")
        if store.x_cat_available.dtype != torch.bool:
            raise GraphContractError(
                f"{node_type}.x_cat_available must use torch.bool"
            )
        if store.x_cont_available.dtype != torch.bool:
            raise GraphContractError(
                f"{node_type}.x_cont_available must use torch.bool"
            )
        if store.x_cont.numel() and not torch.isfinite(store.x_cont).all():
            raise GraphContractError(f"{node_type}.x_cont contains a non-finite value")
        for column, spec in enumerate(categorical):
            values = store.x_cat[:, column]
            if values.numel() and (
                values.min().item() < 0
                or values.max().item() >= int(spec.vocabulary_size or 0)
            ):
                raise GraphContractError(
                    f"{node_type}.{spec.name} is outside its declared vocabulary"
                )
        if tuple(store.cat_feature_names) != registry.names(
            node_type, "categorical"
        ):
            raise GraphContractError(f"{node_type} categorical columns are reordered")
        if tuple(store.cont_feature_names) != registry.names(node_type, "continuous"):
            raise GraphContractError(f"{node_type} continuous columns are reordered")
        if len(tuple(store.entity_id)) != count:
            raise GraphContractError(f"{node_type}.entity_id length differs from node count")
        if len(set(store.entity_id)) != count:
            raise GraphContractError(f"{node_type}.entity_id values must be unique")

    for candidate_type in ("beat", "onset"):
        slots = getattr(graph[candidate_type], "candidate_slot", None)
        expected_shape = (graph[candidate_type].num_nodes,)
        if (
            not isinstance(slots, torch.Tensor)
            or slots.dtype != torch.bool
            or tuple(slots.shape) != expected_shape
            or (slots.numel() and not slots.all())
        ):
            raise GraphContractError(
                f"{candidate_type}.candidate_slot must select every raw candidate"
            )

    for edge_type in MANDATORY_EDGE_TYPES:
        edge_index = graph[edge_type].edge_index
        if not isinstance(edge_index, torch.Tensor):
            raise GraphContractError(f"{edge_type} has no edge_index tensor")
        if edge_index.dtype != torch.long or edge_index.ndim != 2:
            raise GraphContractError(f"{edge_type}.edge_index must be a rank-2 long tensor")
        if edge_index.shape[0] != 2:
            raise GraphContractError(f"{edge_type}.edge_index must have shape [2, E]")
        source_type, _, destination_type = edge_type
        if edge_index.numel():
            if edge_index[0].min().item() < 0 or edge_index[1].min().item() < 0:
                raise GraphContractError(f"{edge_type} contains a negative endpoint")
            if edge_index[0].max().item() >= graph[source_type].num_nodes:
                raise GraphContractError(f"{edge_type} source endpoint is out of range")
            if edge_index[1].max().item() >= graph[destination_type].num_nodes:
                raise GraphContractError(
                    f"{edge_type} destination endpoint is out of range"
                )

    for forward, reverse in REVERSE_EDGE_TYPES.items():
        expected = graph[forward].edge_index.flip(0)
        if not torch.equal(graph[reverse].edge_index, expected):
            raise GraphContractError(
                f"reverse relation {reverse} is not the exact transpose of {forward}"
            )


__all__ = ["GraphContractError", "validate_raw_graph"]
