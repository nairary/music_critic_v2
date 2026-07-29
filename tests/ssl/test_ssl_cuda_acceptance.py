"""Optional CUDA+AMP acceptance for the Phase 7A SSL harness."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize
import pytest
import torch

from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.engine import run_ssl_training


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 7A CUDA+AMP acceptance requires CUDA",
)
def test_bounded_cuda_amp_smoke(tmp_path: Path) -> None:
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="ssl_training",
            overrides=[
                "experiment=one_batch",
                "experiment.steps=2",
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=2",
                "model.ffn_multiplier=2",
                "model.dropout=0",
                "data=bounded",
                "device=cuda",
                "device.amp=true",
                "ssl.decoder_hidden_dim=8",
                "ssl.projector_hidden_dim=8",
                f"output_dir={tmp_path / 'cuda-amp'}",
            ],
        )

    report = run_ssl_training(config)

    assert report["device"]["resolved_device"] == "cuda"
    assert report["device"]["cuda_available"] is True
    assert report["device"]["cuda_device_name"]
    assert report["device"]["deterministic_algorithms"] is True
    assert report["device"]["cudnn_benchmark"] is False
    assert report["device"]["cudnn_deterministic"] is True
    assert report["device"]["cublas_workspace_config"] in {
        ":4096:8",
        ":16:8",
    }
    assert report["device"]["peak_allocated_bytes"] > 0
    assert report["device"]["peak_reserved_bytes"] > 0
    assert report["amp_enabled"] is True
    assert report["scaler_enabled"] is True
    assert report["initial"]["total_ssl_loss"] is not None
    assert report["final"]["total_ssl_loss"] is not None
    assert report["checkpoint_reload"]["bit_exact"] is True
    assert all(report["deterministic_repeat"].values())
    assert report["no_leakage_mutation_evidence"]["passed"] is True
