from __future__ import annotations

from scripts.benchmark_phase6a import benchmark_variant


def test_cpu_benchmark_reports_both_controlled_variants(mixed_batch) -> None:
    reports = [
        benchmark_variant(mixed_batch, variant)
        for variant in ("feature_only", "local_gnn")
    ]
    assert [report["variant"] for report in reports] == [
        "feature_only",
        "local_gnn",
    ]
    for report in reports:
        assert report["device"] == "cpu"
        assert report["graphs"] == report["samples"] == 3
        assert report["nodes"] > 0
        assert report["edges"] > 0
        assert report["parameter_count"] > 0
        assert report["active_task_count"] == 14
        assert report["prediction_task_count"] == 14
        assert report["candidate_logit_rows"] > report["supervision_rows"] > 0
        assert report["routing_operations"]["prediction_task_visits"] == 14
        assert report["total_step_seconds"] >= 0
    assert reports[1]["parameter_count"] > reports[0]["parameter_count"]
