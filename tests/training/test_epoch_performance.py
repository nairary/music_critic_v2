from __future__ import annotations

import json
from pathlib import Path

import torch

from music_critic.training.config import (
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from music_critic.training.engine import run_training


def _config(output: Path) -> TrainingConfig:
    return TrainingConfig(
        seed=42,
        output_dir=str(output),
        model=ModelConfig(
            name="feature_only",
            hidden_dim=8,
            local_gnn_layers=0,
            dropout=0.0,
        ),
        data=DataConfig(
            name="bounded",
            batch_size=3,
            epoch_size=3,
            validation_epoch_size=0,
        ),
        experiment=ExperimentConfig(
            name="smoke",
            epochs=1,
            checkpoint_interval=1,
            validation_interval=1,
        ),
        optimizer=OptimizerConfig(learning_rate=3e-4),
        objective=ObjectiveConfig(
            name="supervised_harmonic",
            harmonic_weight=1.0,
            reconstruction_weight=0.0,
        ),
        scheduler=SchedulerConfig(name="none"),
        device=DeviceConfig(name="cpu"),
    )


def test_epoch_timing_is_bounded_and_not_checkpoint_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    report = run_training(_config(output))
    rows = [
        json.loads(line)
        for line in (output / "epoch_performance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["checkpoint_binding_participation"] is False
    for split in ("train", "validation"):
        assert rows[0][split]["wall_seconds"] > 0
        assert rows[0][split]["samples_per_second"] > 0
        assert rows[0][split]["batches_per_second"] > 0
    checkpoint = torch.load(
        output / "last.pt", map_location="cpu", weights_only=True
    )
    metadata_text = json.dumps(
        checkpoint["metadata"], sort_keys=True
    )
    assert "wall_seconds" not in metadata_text
    assert "samples_per_second" not in metadata_text
    assert "batches_per_second" not in metadata_text
    assert report["detailed_profiler_enabled"] is False
    assert not (output / "performance_report.json").exists()
