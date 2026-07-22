from __future__ import annotations

from music_critic.data import CanonicalPiece
from music_critic.graph import GRAPH_SCHEMA_VERSION, MANDATORY_EDGE_TYPES
from scripts.benchmark_graph_builder import benchmark_piece


def test_benchmark_reports_versions_counts_and_time(
    canonical_piece: CanonicalPiece,
) -> None:
    report = benchmark_piece(canonical_piece, repeats=2)
    assert report["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert report["repeats"] == 2
    assert report["node_counts"]["song"] == 1
    assert len(report["edge_counts"]) == len(MANDATORY_EDGE_TYPES)
    assert report["total_nodes"] == sum(report["node_counts"].values())
    assert report["total_edges"] == sum(report["edge_counts"].values())
    assert report["construction_seconds"]["min"] >= 0
