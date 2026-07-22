"""Deterministic diagnostic serialization for raw heterographs."""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any

from torch_geometric.data import HeteroData

from music_critic.graph.feature_registry import RAW_FEATURE_REGISTRY, FeatureRegistry
from music_critic.graph.relations import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES
from music_critic.graph.validation import validate_raw_graph


def graph_to_dict(
    graph: HeteroData,
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> dict[str, Any]:
    """Return the stable JSON mapping used for inspection and regression tests."""

    validate_raw_graph(graph, registry=registry)
    nodes: list[dict[str, Any]] = []
    for node_type in MANDATORY_NODE_TYPES:
        store = graph[node_type]
        node = {
            "node_type": node_type,
            "entity_id": list(store.entity_id),
            "cat_feature_names": list(store.cat_feature_names),
            "cont_feature_names": list(store.cont_feature_names),
            "x_cat": store.x_cat.tolist(),
            "x_cat_available": store.x_cat_available.tolist(),
            "x_cont": store.x_cont.tolist(),
            "x_cont_available": store.x_cont_available.tolist(),
        }
        if node_type in {"beat", "onset"}:
            node["candidate_slot"] = store.candidate_slot.tolist()
        nodes.append(node)

    edges = [
        {
            "source_type": source_type,
            "relation": relation,
            "destination_type": destination_type,
            "edge_index": graph[(source_type, relation, destination_type)]
            .edge_index.tolist(),
        }
        for source_type, relation, destination_type in MANDATORY_EDGE_TYPES
    ]
    return {
        "schema_version": graph.schema_version,
        "graph_schema_version": graph.graph_schema_version,
        "feature_registry_version": graph.feature_registry_version,
        "graph_builder_version": graph.graph_builder_version,
        "raw_only": graph.raw_only,
        "nodes": nodes,
        "edges": edges,
    }


def dumps_graph(
    graph: HeteroData,
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
    indent: int | None = None,
) -> str:
    return json.dumps(
        graph_to_dict(graph, registry=registry),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def dump_graph(
    graph: HeteroData,
    path: str | PathLike[str],
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
    indent: int | None = 2,
) -> None:
    Path(path).write_text(
        dumps_graph(graph, registry=registry, indent=indent) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def graph_fingerprint(
    graph: HeteroData,
    *,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> str:
    return hashlib.sha256(
        dumps_graph(graph, registry=registry).encode("utf-8")
    ).hexdigest()


__all__ = ["dump_graph", "dumps_graph", "graph_fingerprint", "graph_to_dict"]
