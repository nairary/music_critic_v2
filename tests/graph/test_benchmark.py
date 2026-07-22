from __future__ import annotations

from music_critic.data import CanonicalPiece
from music_critic.graph import GRAPH_SCHEMA_VERSION, MANDATORY_EDGE_TYPES
from scripts.benchmark_graph_builder import (
    benchmark_piece,
    benchmark_synthetic_suite,
    make_synthetic_piece,
)


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
    assert report["output_tensor_bytes"] > 0
    assert set(report["construction_seconds"]) == {"mean", "min", "max"}


def test_synthetic_benchmark_matrix_reports_output_sensitive_scaling() -> None:
    suite = benchmark_synthetic_suite(
        repeats=1,
        note_counts=(10, 20, 40),
        dense_count=40,
        sustain_note_count=3,
        sustain_beats=16,
    )
    cases = suite["cases"]
    assert set(cases) == {
        "sequential_10",
        "sequential_20",
        "sequential_40",
        "dense_same_onset_40",
        "long_sustained_3x16",
    }
    assert cases["sequential_10"]["node_counts"]["note"] == 10
    assert cases["sequential_20"]["node_counts"]["note"] == 20
    assert cases["sequential_40"]["node_counts"]["note"] == 40
    assert cases["sequential_40"]["total_edges"] > cases["sequential_20"]["total_edges"]
    assert cases["dense_same_onset_40"]["node_counts"]["onset"] == 1
    assert cases["long_sustained_3x16"]["active_at_edges"] == 3 * 16
    assert suite["peak_output_tensor_bytes"] == max(
        report["output_tensor_bytes"] for report in cases.values()
    )
    assert "construction_seconds" in cases["sequential_40"]


def test_synthetic_benchmark_inputs_are_validator_clean() -> None:
    for layout in ("sequential", "dense_same_onset", "long_sustained"):
        piece = make_synthetic_piece(12, layout=layout, sustain_beats=8)
        assert piece.notes
        benchmark_piece(piece, repeats=1)
