from __future__ import annotations

from pathlib import Path

import pytest
import torch

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models import (
    ACTIVE_TASK_IDS,
    CheckpointContractError,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    checkpoint_metadata,
    load_baseline_checkpoint,
    save_baseline_checkpoint,
)


def test_reconstruction_is_local_masked_and_reaches_all_node_encoders(
    mixed_batch,
) -> None:
    torch.manual_seed(13)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    output = model(mixed_batch)
    assert {item.node_type for item in output.reconstruction} == set(
        MANDATORY_NODE_TYPES
    )
    assert output.reconstruction_loss is not None
    output.reconstruction_loss.backward()
    for node_type in MANDATORY_NODE_TYPES:
        gradients = [
            parameter.grad
            for parameter in model.encoder.feature_encoder.node_encoders[
                node_type
            ].parameters()
        ]
        assert any(
            gradient is not None and torch.count_nonzero(gradient)
            for gradient in gradients
        )
    for item in output.reconstruction:
        assert item.per_node_loss.shape == (item.logits.shape[0],)
        assert torch.all(item.per_node_loss[~item.availability_mask] == 0)


def test_checkpoint_metadata_and_round_trip_reproduce_logits(
    mixed_batch, tmp_path: Path
) -> None:
    torch.manual_seed(17)
    config = LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    model = LocalHeterogeneousBaseline(config).eval()
    before = {
        item.task_id: item.logits.detach().clone()
        for item in model(mixed_batch, include_reconstruction=False).tasks
    }
    path = tmp_path / "phase6a.pt"
    save_baseline_checkpoint(path, model)
    restored = LocalHeterogeneousBaseline(config).eval()
    metadata = load_baseline_checkpoint(path, restored)
    after = {
        item.task_id: item.logits.detach()
        for item in restored(mixed_batch, include_reconstruction=False).tasks
    }
    assert metadata == checkpoint_metadata(model)
    assert set(after) == set(ACTIVE_TASK_IDS)
    assert all(torch.equal(before[task_id], after[task_id]) for task_id in before)


def test_checkpoint_rejects_incompatible_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase6a.pt"
    save_baseline_checkpoint(
        path,
        LocalHeterogeneousBaseline(
            LocalBaselineConfig(hidden_dim=16, gnn_layers=1)
        ),
    )
    incompatible = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=32, gnn_layers=1)
    )
    with pytest.raises(CheckpointContractError):
        load_baseline_checkpoint(path, incompatible)
