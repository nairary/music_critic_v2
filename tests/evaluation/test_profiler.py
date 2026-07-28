from __future__ import annotations

from pathlib import Path

import pytest

from music_critic.evaluation.config import ProfilerConfig
from music_critic.evaluation.contracts import EvaluationContractError
from music_critic.evaluation.profiler import run_profiler


def test_profiler_requires_explicit_enable(tmp_path: Path) -> None:
    with pytest.raises(
        EvaluationContractError,
        match="explicit_enable_required",
    ):
        run_profiler(
            ProfilerConfig(output_path=str(tmp_path / "report.json"))
        )
    assert not (tmp_path / "report.json").exists()


def test_tiny_profiler_has_all_bounded_stage_and_rate_evidence(
    tmp_path: Path,
) -> None:
    report = run_profiler(
        ProfilerConfig(
            enabled=True,
            output_path=str(tmp_path / "report.json"),
            max_batches=1,
            dataset_values=["hooktheory"],
            model_values=["feature_only"],
            batch_sizes=[1],
            worker_values=[0],
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=1,
        )
    )
    cell = report["cells"][0]
    assert cell["status"] == "completed"
    assert set(cell["stages"]) == {
        "canonical_artifact_read",
        "graph_construction",
        "target_alignment_tensorization",
        "collation",
        "device_transfer",
        "model_forward",
        "loss_construction",
        "backward",
        "optimizer_step",
        "validation_forward",
    }
    assert all(
        value > 0 for value in cell["throughput"].values()
    )
    assert cell["batch_time"]["observation_count"] == 1
    assert set(cell["fingerprints"]) == {
        "dataset",
        "model",
        "batch",
        "worker",
    }
    assert cell["cpu_peak_rss_kib"] > 0
    assert report["retained_per_batch_history"] is False
