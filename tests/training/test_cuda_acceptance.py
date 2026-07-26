from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize
import pytest
import torch

from music_critic.graph import RAW_FEATURE_REGISTRY
from music_critic.training.config import register_training_configs
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import build_baseline_model


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 6C CUDA acceptance requires a CUDA runner",
)


def _small_hierarchical_config():
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=[
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=4",
                "model.dropout=0",
                "device=cuda",
            ],
        )


def test_real_cuda_cli_runner_uses_amp_scaler_and_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cuda-cli"
    environment = dict(os.environ)
    source_root = str(Path.cwd() / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (source_root, environment.get("PYTHONPATH", "")),
        )
    )
    command = [
        sys.executable,
        "-m",
        "music_critic.training.run",
        "experiment=one_batch",
        "model=hierarchical",
        "data=bounded",
        "device=cuda",
        "device.amp=true",
        "experiment.steps=6",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=4",
        "model.dropout=0",
        f"output_dir={output}",
    ]
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    report = json.loads(
        (output / "one_batch_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["amp_enabled"] is True
    assert report["scaler_enabled"] is True
    assert report["optimizer_step_count"] == 6
    assert report["checkpoint_reload_bit_exact"] is True
    assert Path(report["checkpoint"]).is_file()
    assert all(
        math.isfinite(row[name])
        for row in report["curve"]
        for name in (
            "harmonic_loss",
            "reconstruction_loss",
            "total_loss",
        )
    )
    assert report["final"]["harmonic_loss"] < report["initial"][
        "harmonic_loss"
    ]
    assert report["final"]["reconstruction_loss"] < report["initial"][
        "reconstruction_loss"
    ]
    assert report["device"]["resolved_device"] == "cuda"
    assert report["device"]["peak_allocated_bytes"] > 0
    assert report["device"]["peak_reserved_bytes"] > 0


def test_cuda_feature_perturbation_changes_only_its_sample(
    bounded_batch,
) -> None:
    config = _small_hierarchical_config()
    batch = move_multisource_batch(
        bounded_batch, "cuda", non_blocking=True
    )
    model = build_baseline_model(config.model).cuda().eval()
    with torch.no_grad():
        before = model(batch)

    note = batch.raw_graph_batch["note"]
    velocity_column = RAW_FEATURE_REGISTRY.names(
        "note", "continuous"
    ).index("velocity")
    changed_rows = torch.nonzero(
        note.batch == 0, as_tuple=False
    ).flatten()
    values = note.x_cont[changed_rows, velocity_column]
    note.x_cont[changed_rows, velocity_column] = torch.where(
        values < 127,
        values + 1,
        values - 1,
    )
    assert bool(
        note.x_cont[changed_rows, velocity_column].ge(0).all()
    )
    assert bool(
        note.x_cont[changed_rows, velocity_column].le(127).all()
    )
    with torch.no_grad():
        after = model(batch)

    before_note = before.encoder.fused.embeddings["note"]
    after_note = after.encoder.fused.embeddings["note"]
    assert not torch.equal(
        before_note.index_select(0, changed_rows),
        after_note.index_select(0, changed_rows),
    )
    changed_own_logits = False
    for left, right in zip(
        before.predictions, after.predictions, strict=True
    ):
        sample_zero = left.sample_indices == 0
        if sample_zero.any() and not torch.equal(
            left.logits[sample_zero],
            right.logits[sample_zero],
        ):
            changed_own_logits = True
        other_samples = left.sample_indices != 0
        assert torch.equal(
            left.logits[other_samples],
            right.logits[other_samples],
        )
    assert changed_own_logits

    for node_type, embeddings in (
        before.encoder.fused.embeddings.items()
    ):
        membership = before.encoder.fused.batch_membership[node_type]
        other_samples = membership != 0
        assert torch.equal(
            embeddings[other_samples],
            after.encoder.fused.embeddings[node_type][other_samples],
        )
