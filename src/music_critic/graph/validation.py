"""Validation for the model-facing Phase 3A heterograph contract."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.data import SCHEMA_VERSION
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


ALLOWED_GLOBAL_ATTRIBUTES = frozenset(
    {
        "schema_version",
        "graph_schema_version",
        "feature_registry_version",
        "graph_builder_version",
        "raw_only",
    }
)
BASE_NODE_ATTRIBUTES = frozenset(
    {
        "num_nodes",
        "x_cat",
        "x_cat_available",
        "x_cont",
        "x_cont_available",
        "entity_id",
        "cat_feature_names",
        "cont_feature_names",
    }
)
CANDIDATE_NODE_ATTRIBUTES = BASE_NODE_ATTRIBUTES | {"candidate_slot"}
ALLOWED_EDGE_ATTRIBUTES = frozenset({"edge_index"})
BATCH_GLOBAL_ATTRIBUTES = ALLOWED_GLOBAL_ATTRIBUTES
BATCH_BASE_NODE_ATTRIBUTES = BASE_NODE_ATTRIBUTES | {"batch", "ptr"}
BATCH_CANDIDATE_NODE_ATTRIBUTES = CANDIDATE_NODE_ATTRIBUTES | {"batch", "ptr"}
BATCH_EDGE_ATTRIBUTES = ALLOWED_EDGE_ATTRIBUTES


def _require_exact_attributes(
    *,
    location: str,
    actual: set[str],
    expected: frozenset[str],
) -> None:
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise GraphContractError(
            f"{location} attributes differ from the raw-only contract: "
            f"extra={extra}, missing={missing}"
        )


def validate_raw_graph(
    graph: HeteroData,
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> None:
    """Raise ``GraphContractError`` for any schema, feature, or edge violation."""

    if not isinstance(graph, HeteroData):
        raise GraphContractError("graph must be torch_geometric.data.HeteroData")
    _require_exact_attributes(
        location="global",
        actual=set(graph._global_store.keys()),
        expected=ALLOWED_GLOBAL_ATTRIBUTES,
    )
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
    if getattr(graph, "schema_version", None) != SCHEMA_VERSION:
        raise GraphContractError(
            f"graph metadata 'schema_version' must be {SCHEMA_VERSION!r}"
        )

    for node_type in MANDATORY_NODE_TYPES:
        store = graph[node_type]
        allowed_node_attributes = (
            CANDIDATE_NODE_ATTRIBUTES
            if node_type in {"beat", "onset"}
            else BASE_NODE_ATTRIBUTES
        )
        _require_exact_attributes(
            location=f"node store {node_type!r}",
            actual=set(store.keys()),
            expected=allowed_node_attributes,
        )
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
            available = store.x_cat_available[:, column]
            unavailable = ~available
            if unavailable.any():
                if spec.unknown_id is None:
                    raise GraphContractError(
                        f"{node_type}.{spec.name} has unavailable values but no "
                        "dedicated unknown ID"
                    )
                if not torch.all(values[unavailable] == spec.unknown_id):
                    raise GraphContractError(
                        f"{node_type}.{spec.name} unavailable values must use "
                        f"unknown ID {spec.unknown_id}"
                    )
            if (
                spec.unknown_id is not None
                and available.any()
                and torch.any(values[available] == spec.unknown_id)
            ):
                raise GraphContractError(
                    f"{node_type}.{spec.name} available values cannot use "
                    f"unknown ID {spec.unknown_id}"
                )
        for column, spec in enumerate(continuous):
            unavailable = ~store.x_cont_available[:, column]
            if unavailable.any() and not torch.all(
                store.x_cont[unavailable, column]
                == spec.unavailable_continuous_value
            ):
                raise GraphContractError(
                    f"{node_type}.{spec.name} unavailable values must use "
                    f"placeholder {spec.unavailable_continuous_value}"
                )
        if tuple(store.cat_feature_names) != registry.names(
            node_type, "categorical"
        ):
            raise GraphContractError(f"{node_type} categorical columns are reordered")
        if tuple(store.cont_feature_names) != registry.names(node_type, "continuous"):
            raise GraphContractError(f"{node_type} continuous columns are reordered")
        if not isinstance(store.entity_id, tuple) or not all(
            isinstance(entity_id, str) for entity_id in store.entity_id
        ):
            raise GraphContractError(f"{node_type}.entity_id must be a tuple of strings")
        if len(store.entity_id) != count:
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
        _require_exact_attributes(
            location=f"edge store {edge_type!r}",
            actual=set(graph[edge_type].keys()),
            expected=ALLOWED_EDGE_ATTRIBUTES,
        )
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


def _require_batched_metadata(
    batch: Batch,
    *,
    name: str,
    expected: str,
    sample_count: int,
) -> None:
    values = getattr(batch, name, None)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != sample_count
        or any(not isinstance(value, str) or value != expected for value in values)
    ):
        raise GraphContractError(
            f"batched graph metadata {name!r} must contain exactly "
            f"{sample_count} copies of {expected!r}"
        )


def _require_batch_index(
    *,
    node_type: str,
    batch_index: object,
    ptr: object,
    node_count: int,
    sample_count: int,
) -> tuple[int, ...]:
    if (
        not isinstance(batch_index, torch.Tensor)
        or batch_index.dtype != torch.long
        or batch_index.ndim != 1
        or tuple(batch_index.shape) != (node_count,)
    ):
        raise GraphContractError(
            f"{node_type}.batch must be a rank-one long tensor of node count"
        )
    if (
        not isinstance(ptr, torch.Tensor)
        or ptr.dtype != torch.long
        or ptr.ndim != 1
        or tuple(ptr.shape) != (sample_count + 1,)
    ):
        raise GraphContractError(
            f"{node_type}.ptr must be a rank-one long tensor of sample_count + 1"
        )
    pointers = tuple(int(value) for value in ptr.tolist())
    if (
        pointers[0] != 0
        or pointers[-1] != node_count
        or any(right < left for left, right in zip(pointers, pointers[1:]))
    ):
        raise GraphContractError(
            f"{node_type}.ptr must be monotonic from zero to the node count"
        )
    counts = ptr[1:] - ptr[:-1]
    expected_batch = torch.repeat_interleave(
        torch.arange(sample_count, dtype=torch.long, device=batch_index.device),
        counts.to(batch_index.device),
    )
    if not torch.equal(batch_index, expected_batch):
        raise GraphContractError(
            f"{node_type}.batch is inconsistent with {node_type}.ptr"
        )
    return tuple(
        right - left for left, right in zip(pointers, pointers[1:])
    )


def _require_batched_node_metadata(
    *,
    node_type: str,
    store: object,
    sample_count: int,
    per_graph_counts: tuple[int, ...],
    registry: FeatureRegistry,
) -> None:
    entity_ids = getattr(store, "entity_id", None)
    if (
        not isinstance(entity_ids, Sequence)
        or isinstance(entity_ids, (str, bytes))
        or len(entity_ids) != sample_count
    ):
        raise GraphContractError(
            f"{node_type}.entity_id must contain one tuple per source graph"
        )
    for graph_index, (identifiers, expected_count) in enumerate(
        zip(entity_ids, per_graph_counts)
    ):
        if (
            not isinstance(identifiers, tuple)
            or len(identifiers) != expected_count
            or not all(isinstance(identifier, str) for identifier in identifiers)
            or len(set(identifiers)) != expected_count
        ):
            raise GraphContractError(
                f"{node_type}.entity_id entry {graph_index} differs from its "
                "source-graph node contract"
            )

    for attribute, kind in (
        ("cat_feature_names", "categorical"),
        ("cont_feature_names", "continuous"),
    ):
        values = getattr(store, attribute, None)
        expected = registry.names(node_type, kind)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != sample_count
            or any(not isinstance(value, tuple) or value != expected for value in values)
        ):
            raise GraphContractError(
                f"{node_type}.{attribute} must contain the production column "
                "tuple for every source graph"
            )


def validate_raw_graph_batch(
    batch: Batch,
    *,
    sample_count: int,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> None:
    """Validate the exact Phase 3A contract after normal PyG collation."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise GraphContractError("batch sample_count must be a positive integer")
    if not isinstance(batch, Batch):
        raise GraphContractError(
            "raw graph batch must be torch_geometric.data.Batch"
        )
    if batch.num_graphs != sample_count:
        raise GraphContractError(
            "PyG batch graph count differs from sample metadata count"
        )
    _require_exact_attributes(
        location="batch global",
        actual=set(batch._global_store.keys()),
        expected=BATCH_GLOBAL_ATTRIBUTES,
    )
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "feature_registry_version": registry.version,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
    }
    for name, expected in expected_metadata.items():
        _require_batched_metadata(
            batch,
            name=name,
            expected=expected,
            sample_count=sample_count,
        )
    raw_only = getattr(batch, "raw_only", None)
    if (
        not isinstance(raw_only, torch.Tensor)
        or raw_only.dtype != torch.bool
        or raw_only.ndim != 1
        or tuple(raw_only.shape) != (sample_count,)
        or not bool(torch.all(raw_only).item())
    ):
        raise GraphContractError(
            "batched raw_only must be a rank-one bool tensor containing one "
            "True value per source graph"
        )

    if tuple(batch.node_types) != MANDATORY_NODE_TYPES:
        raise GraphContractError(
            f"batched node types must be exactly {MANDATORY_NODE_TYPES}"
        )
    if tuple(batch.edge_types) != MANDATORY_EDGE_TYPES:
        raise GraphContractError(
            "batched edge types or edge ordering differ from the raw contract"
        )

    for node_type in MANDATORY_NODE_TYPES:
        store = batch[node_type]
        expected_attributes = (
            BATCH_CANDIDATE_NODE_ATTRIBUTES
            if node_type in {"beat", "onset"}
            else BATCH_BASE_NODE_ATTRIBUTES
        )
        _require_exact_attributes(
            location=f"batched node store {node_type!r}",
            actual=set(store.keys()),
            expected=expected_attributes,
        )
        node_count = store.num_nodes
        if not isinstance(node_count, int) or node_count < 0:
            raise GraphContractError(f"{node_type}.num_nodes is invalid")
        categorical = registry.for_node(node_type, "categorical")
        continuous = registry.for_node(node_type, "continuous")
        expected_shapes = {
            "x_cat": (node_count, len(categorical)),
            "x_cat_available": (node_count, len(categorical)),
            "x_cont": (node_count, len(continuous)),
            "x_cont_available": (node_count, len(continuous)),
        }
        expected_dtypes = {
            "x_cat": torch.long,
            "x_cat_available": torch.bool,
            "x_cont": torch.float32,
            "x_cont_available": torch.bool,
        }
        for name, shape in expected_shapes.items():
            value = getattr(store, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.dtype != expected_dtypes[name]
            ):
                raise GraphContractError(
                    f"batched {node_type}.{name} must have shape {shape} and "
                    f"dtype {expected_dtypes[name]}"
                )
        if store.x_cont.numel() and not torch.isfinite(store.x_cont).all():
            raise GraphContractError(
                f"batched {node_type}.x_cont contains a non-finite value"
            )
        per_graph_counts = _require_batch_index(
            node_type=node_type,
            batch_index=getattr(store, "batch", None),
            ptr=getattr(store, "ptr", None),
            node_count=node_count,
            sample_count=sample_count,
        )
        _require_batched_node_metadata(
            node_type=node_type,
            store=store,
            sample_count=sample_count,
            per_graph_counts=per_graph_counts,
            registry=registry,
        )
        if node_type in {"beat", "onset"}:
            slots = getattr(store, "candidate_slot", None)
            if (
                not isinstance(slots, torch.Tensor)
                or slots.dtype != torch.bool
                or tuple(slots.shape) != (node_count,)
                or (slots.numel() and not bool(slots.all().item()))
            ):
                raise GraphContractError(
                    f"batched {node_type}.candidate_slot must select every candidate"
                )

    for edge_type in MANDATORY_EDGE_TYPES:
        store = batch[edge_type]
        _require_exact_attributes(
            location=f"batched edge store {edge_type!r}",
            actual=set(store.keys()),
            expected=BATCH_EDGE_ATTRIBUTES,
        )
        edge_index = getattr(store, "edge_index", None)
        if (
            not isinstance(edge_index, torch.Tensor)
            or edge_index.dtype != torch.long
            or edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise GraphContractError(
                f"batched {edge_type}.edge_index must be a [2, E] long tensor"
            )
        source_type, _, destination_type = edge_type
        if edge_index.numel():
            if edge_index.min().item() < 0:
                raise GraphContractError(
                    f"batched {edge_type} contains a negative endpoint"
                )
            if edge_index[0].max().item() >= batch[source_type].num_nodes:
                raise GraphContractError(
                    f"batched {edge_type} source endpoint is out of range"
                )
            if edge_index[1].max().item() >= batch[destination_type].num_nodes:
                raise GraphContractError(
                    f"batched {edge_type} destination endpoint is out of range"
                )
            source_batch = batch[source_type].batch[edge_index[0]]
            destination_batch = batch[destination_type].batch[edge_index[1]]
            if not torch.equal(source_batch, destination_batch):
                raise GraphContractError(
                    f"batched {edge_type} connects different source graphs"
                )

    for forward, reverse in REVERSE_EDGE_TYPES.items():
        if not torch.equal(
            batch[reverse].edge_index,
            batch[forward].edge_index.flip(0),
        ):
            raise GraphContractError(
                f"batched reverse relation {reverse} is not the exact transpose "
                f"of {forward}"
            )

    try:
        source_graphs = batch.to_data_list()
    except Exception as exc:
        raise GraphContractError(
            "PyG batch cannot be reconstructed into source graphs"
        ) from exc
    if len(source_graphs) != sample_count:
        raise GraphContractError(
            "PyG batch reconstruction count differs from sample metadata count"
        )
    for graph_index, graph in enumerate(source_graphs):
        try:
            validate_raw_graph(graph, registry=registry)
        except GraphContractError as exc:
            raise GraphContractError(
                f"source graph {graph_index} violates the raw graph contract: {exc}"
            ) from exc


__all__ = [
    "ALLOWED_EDGE_ATTRIBUTES",
    "ALLOWED_GLOBAL_ATTRIBUTES",
    "BATCH_BASE_NODE_ATTRIBUTES",
    "BATCH_CANDIDATE_NODE_ATTRIBUTES",
    "BATCH_EDGE_ATTRIBUTES",
    "BATCH_GLOBAL_ATTRIBUTES",
    "BASE_NODE_ATTRIBUTES",
    "CANDIDATE_NODE_ATTRIBUTES",
    "GraphContractError",
    "validate_raw_graph",
    "validate_raw_graph_batch",
]
