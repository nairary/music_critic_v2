from __future__ import annotations

from scripts.benchmark_phase6b import (
    benchmark_controlled_ablation,
    benchmark_uneven_sequence_packing,
)


def test_phase6b_cpu_benchmark_reports_controlled_three_way_ablation(
    mixed_batch,
) -> None:
    reports = benchmark_controlled_ablation(mixed_batch)
    assert [item["variant"] for item in reports] == [
        "phase6a_feature_only",
        "phase6a_local_gnn",
        "local_gnn_hierarchy_transformer_fusion",
    ]
    assert {item["candidate_logit_rows"] for item in reports} == {237}
    assert len({item["supervision_rows"] for item in reports}) == 1
    assert all(item["active_task_count"] == 14 for item in reports)
    hierarchical = reports[2]
    assert hierarchical["bars"] > 0
    assert hierarchical["tracks"] > 0
    assert all(
        length == 1 + bars + tracks
        for length, bars, tracks in zip(
            hierarchical["coarse_sequence_lengths"],
            hierarchical["bar_counts_by_sample"],
            hierarchical["track_counts_by_sample"],
        )
    )
    for name in (
        "pooling_seconds",
        "transformer_seconds",
        "fusion_seconds",
        "complete_forward_backward_seconds",
    ):
        assert hierarchical[name] >= 0
    assert hierarchical["peak_tensor_shapes"]["contextual_song"] == [3, 32]


def test_uneven_sequence_benchmark_reports_packing_and_full_forward(
    mixed_batch,
) -> None:
    report = benchmark_uneven_sequence_packing(
        mixed_batch, repeats=2
    )
    assert report["sequence_lengths"] == [3, 4, 3]
    assert report["padded_shape"] == [3, 4, 32]
    assert report["candidate_logit_rows"] == 237
    assert report["sequence_construction_mean_seconds"] >= 0
    assert report["hierarchical_forward_mean_seconds"] >= 0
    assert "no throughput" in report["scope"]
