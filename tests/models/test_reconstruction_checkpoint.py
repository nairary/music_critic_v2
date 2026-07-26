from __future__ import annotations

from copy import deepcopy
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


def _recursive_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(_recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _snapshot(model, optimizer):
    return (
        {key: value.detach().clone() for key, value in model.state_dict().items()},
        deepcopy(optimizer.state_dict()),
    )


def _assert_snapshot(model, optimizer, snapshot) -> None:
    model_state, optimizer_state = snapshot
    assert _recursive_equal(model.state_dict(), model_state)
    assert _recursive_equal(optimizer.state_dict(), optimizer_state)


def _trained_pair(mixed_batch):
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    output = model(mixed_batch)
    loss = output.harmonic_loss.total_loss + output.reconstruction_loss
    loss.backward()
    optimizer.step()
    return model, optimizer


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
        for item in model(
            mixed_batch, include_reconstruction=False
        ).predictions
    }
    path = tmp_path / "phase6a.pt"
    save_baseline_checkpoint(path, model)
    assert path.exists()
    assert not tuple(tmp_path.glob(".phase6a.pt.*.tmp"))
    restored = LocalHeterogeneousBaseline(config).eval()
    metadata = load_baseline_checkpoint(path, restored)
    after = {
        item.task_id: item.logits.detach()
        for item in restored(
            mixed_batch, include_reconstruction=False
        ).predictions
    }
    assert metadata == checkpoint_metadata(model)
    assert set(after) == set(ACTIVE_TASK_IDS)
    assert all(torch.equal(before[task_id], after[task_id]) for task_id in before)


def test_checkpoint_save_failure_preserves_destination_and_removes_temp(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "phase6a.pt"
    path.write_bytes(b"previous-checkpoint")
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1)
    )

    def fail_save(*_args, **_kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected save failure"):
        save_baseline_checkpoint(path, model)
    assert path.read_bytes() == b"previous-checkpoint"
    assert not tuple(tmp_path.glob(".phase6a.pt.*.tmp"))


def _assert_rejected_without_mutation(
    path: Path,
    model,
    optimizer,
    *,
    match: str,
) -> None:
    snapshot = _snapshot(model, optimizer)
    with pytest.raises(CheckpointContractError, match=match):
        load_baseline_checkpoint(path, model, optimizer=optimizer)
    _assert_snapshot(model, optimizer, snapshot)


def test_checkpoint_missing_model_state_is_failure_atomic(
    mixed_batch, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(mixed_batch)
    path = tmp_path / "missing-model.pt"
    torch.save(
        {
            "metadata": checkpoint_metadata(model),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    _assert_rejected_without_mutation(
        path, model, optimizer, match="model state is missing"
    )


def test_checkpoint_wrong_model_shape_is_failure_atomic(
    mixed_batch, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(mixed_batch)
    model_state = deepcopy(model.state_dict())
    key = next(
        key
        for key, value in model_state.items()
        if value.ndim > 0 and value.shape[0] > 0
    )
    model_state[key] = model_state[key][:-1]
    path = tmp_path / "wrong-shape.pt"
    torch.save(
        {
            "metadata": checkpoint_metadata(model),
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    _assert_rejected_without_mutation(
        path, model, optimizer, match="shape or dtype"
    )


def test_checkpoint_missing_optimizer_state_is_failure_atomic(
    mixed_batch, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(mixed_batch)
    path = tmp_path / "missing-optimizer.pt"
    save_baseline_checkpoint(path, model)
    _assert_rejected_without_mutation(
        path, model, optimizer, match="optimizer state is missing"
    )


@pytest.mark.parametrize("corruption", ["group", "tensor"])
def test_checkpoint_incompatible_optimizer_is_failure_atomic(
    mixed_batch, tmp_path: Path, corruption: str
) -> None:
    model, optimizer = _trained_pair(mixed_batch)
    optimizer_state = deepcopy(optimizer.state_dict())
    if corruption == "group":
        optimizer_state["param_groups"][0]["params"] = optimizer_state[
            "param_groups"
        ][0]["params"][:-1]
        expected = "parameter group is incompatible"
    else:
        parameter_id = next(iter(optimizer_state["state"]))
        state = optimizer_state["state"][parameter_id]
        tensor_key = next(
            key
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        )
        state[tensor_key] = state[tensor_key].reshape(-1)[:-1]
        expected = "tensor shape or dtype"
    path = tmp_path / f"optimizer-{corruption}.pt"
    torch.save(
        {
            "metadata": checkpoint_metadata(model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
        },
        path,
    )
    _assert_rejected_without_mutation(
        path, model, optimizer, match=expected
    )


def test_checkpoint_metadata_mismatch_is_failure_atomic(
    mixed_batch, tmp_path: Path
) -> None:
    source, source_optimizer = _trained_pair(mixed_batch)
    path = tmp_path / "metadata.pt"
    save_baseline_checkpoint(path, source, optimizer=source_optimizer)
    incompatible = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=32, gnn_layers=1)
    )
    incompatible_optimizer = torch.optim.Adam(incompatible.parameters(), lr=0.01)
    _assert_rejected_without_mutation(
        path,
        incompatible,
        incompatible_optimizer,
        match="metadata is incompatible",
    )


def test_checkpoint_application_failure_rolls_back_model_and_optimizer(
    mixed_batch, tmp_path: Path, monkeypatch
) -> None:
    torch.manual_seed(43)
    source_model, source_optimizer = _trained_pair(mixed_batch)
    path = tmp_path / "application-failure.pt"
    save_baseline_checkpoint(
        path, source_model, optimizer=source_optimizer
    )

    torch.manual_seed(47)
    model, optimizer = _trained_pair(mixed_batch)
    snapshot = _snapshot(model, optimizer)
    assert any(
        not torch.equal(value, snapshot[0][key])
        for key, value in source_model.state_dict().items()
    )
    original_load = optimizer.load_state_dict
    call_count = 0

    def fail_after_partial_mutation(state_dict):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            optimizer.param_groups[0]["lr"] = 123.0
            first_state = next(iter(optimizer.state.values()))
            first_state["exp_avg"].add_(17.0)
            raise RuntimeError("injected application-time optimizer failure")
        return original_load(state_dict)

    monkeypatch.setattr(
        optimizer, "load_state_dict", fail_after_partial_mutation
    )
    with pytest.raises(
        CheckpointContractError,
        match="state application failed atomically",
    ):
        load_baseline_checkpoint(path, model, optimizer=optimizer)
    assert call_count == 2
    _assert_snapshot(model, optimizer, snapshot)
