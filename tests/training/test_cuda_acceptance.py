from __future__ import annotations

from hydra import compose, initialize
import pytest
import torch

from music_critic.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from music_critic.training.config import register_training_configs
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import build_baseline_model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 6C CUDA acceptance requires a CUDA runner",
)
def test_full_mixed_cuda_hierarchical_acceptance(
    bounded_batch,
    tmp_path,
) -> None:
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="training",
            overrides=[
                "model=hierarchical",
                "model.hidden_dim=16",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=4",
                "model.dropout=0",
                "device=cuda",
                "optimizer.learning_rate=0.001",
            ],
        )
    batch = move_multisource_batch(
        bounded_batch, "cuda", non_blocking=True
    )
    model = build_baseline_model(config.model).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    assert output.harmonic_loss.total_loss is not None
    assert output.reconstruction_loss is not None
    loss = (
        output.harmonic_loss.total_loss
        + output.reconstruction_loss
    )
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        and bool(torch.isfinite(parameter.grad).all())
    }
    expected_module_prefixes = (
        "local_baseline.encoder.feature_encoder.",
        "local_baseline.encoder.layers.",
        "local_baseline.task_heads.",
        "local_baseline.reconstruction_heads.",
        "context_encoder.pooling.",
        "context_encoder.transformer.",
        "context_encoder.fusion.",
    )
    assert all(
        any(name.startswith(prefix) for name in gradients)
        for prefix in expected_module_prefixes
    )
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = tuple(
            item.logits.detach().clone()
            for item in model(batch).predictions
        )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    resolved = {"cuda_acceptance": True}
    fingerprints = {"bounded": "cuda"}
    checkpoint = tmp_path / "cuda.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=1,
        best_validation_loss=None,
        resolved_config=resolved,
        data_fingerprints=fingerprints,
    )
    restored = build_baseline_model(config.model).cuda()
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=0.001
    )
    load_training_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        scheduler=None,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        resolved_config=resolved,
        data_fingerprints=fingerprints,
    )
    restored.eval()
    with torch.no_grad():
        actual = tuple(
            item.logits for item in restored(batch).predictions
        )
    assert all(
        torch.equal(left, right)
        for left, right in zip(expected, actual, strict=True)
    )

    before = tuple(item.detach().clone() for item in actual)
    membership = batch.raw_graph_batch["note"].batch
    changed_rows = torch.nonzero(
        membership == 0, as_tuple=False
    ).flatten()
    batch.raw_graph_batch["note"].x[changed_rows, 0] += 1
    with torch.no_grad():
        changed = restored(batch).predictions
    for reference, prediction in zip(before, changed, strict=True):
        sample_one = prediction.sample_indices == 1
        assert torch.equal(
            reference[sample_one],
            prediction.logits[sample_one],
        )
