#!/usr/bin/env python3
"""Report raw graph size and deterministic construction timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from music_critic.data import CanonicalPiece, load_piece
from music_critic.graph import build_raw_graph


def benchmark_piece(piece: CanonicalPiece, *, repeats: int = 5) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    build_raw_graph(piece)
    elapsed_seconds: list[float] = []
    graph = None
    for _ in range(repeats):
        started = perf_counter()
        graph = build_raw_graph(piece)
        elapsed_seconds.append(perf_counter() - started)
    assert graph is not None

    node_counts = {
        node_type: graph[node_type].num_nodes for node_type in graph.node_types
    }
    edge_counts = {
        "|".join(edge_type): int(graph[edge_type].edge_index.shape[1])
        for edge_type in graph.edge_types
    }
    return {
        "schema_version": graph.schema_version,
        "graph_schema_version": graph.graph_schema_version,
        "feature_registry_version": graph.feature_registry_version,
        "graph_builder_version": graph.graph_builder_version,
        "repeats": repeats,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "total_nodes": sum(node_counts.values()),
        "total_edges": sum(edge_counts.values()),
        "construction_seconds": {
            "mean": fmean(elapsed_seconds),
            "min": min(elapsed_seconds),
            "max": max(elapsed_seconds),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_json", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = benchmark_piece(load_piece(args.canonical_json), repeats=args.repeats)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
