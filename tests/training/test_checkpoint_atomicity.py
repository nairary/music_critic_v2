from __future__ import annotations

import copy
from pathlib import Path
import random
from unittest.mock import patch

from hydra import compose, initialize
import pytest
import torch

from music_critic.training.checkpoint import (
    TrainingCheckpointError,
    capture_rng_state,
    load_training_checkpoint,
    save_training_checkpoint,
)
from music_critic.training.config import register_training_configs
from music_critic.training.models import build_baseline_model


def _assert_state_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def _training_objects():
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="training",
            overrides=[
                "model=feature_only",
                "model.hidden_dim=8",
                "model.dropout=0",
            ],
        )
    model = build_baseline_model(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    return model, optimizer, scheduler, scaler


def _mutate_live_state(model, optimizer, scheduler, scaler) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler_state = scheduler.state_dict()
    scheduler_state["last_epoch"] = int(
        scheduler_state["last_epoch"]
    ) + 1
    scheduler_state["_step_count"] = int(
        scheduler_state["_step_count"]
    ) + 1
    scheduler.load_state_dict(scheduler_state)
    scaler_state = scaler.state_dict()
    scaler_state["scale"] = float(scaler_state["scale"]) / 2
    scaler_state["_growth_tracker"] = (
        int(scaler_state["_growth_tracker"]) + 1
    )
    scaler.load_state_dict(scaler_state)
    random.random()
    torch.rand(7)


def _snapshot(model, optimizer, scheduler, scaler):
    return {
        "model": {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "rng": capture_rng_state(),
    }


def _save(path: Path, model, optimizer, scheduler, scaler) -> None:
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=1,
        best_validation_loss=0.75,
        committed_metric_rows=1,
        resolved_config={"contract": "atomicity-test"},
        data_fingerprints={"data": "bounded"},
    )


def _load(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    maximum_next_epoch: int = 2,
):
    return load_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        maximum_next_epoch=maximum_next_epoch,
        resolved_config={"contract": "atomicity-test"},
        data_fingerprints={"data": "bounded"},
    )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        (
            "next_epoch",
            -1,
            "training.checkpoint.next_epoch_invalid",
        ),
        (
            "next_epoch",
            True,
            "training.checkpoint.next_epoch_invalid",
        ),
        (
            "best_validation_loss",
            1,
            "training.checkpoint.best_metric_invalid",
        ),
        (
            "best_validation_loss",
            float("nan"),
            "training.checkpoint.best_metric_invalid",
        ),
        (
            "committed_metric_rows",
            0,
            "training.checkpoint.metric_rows_invalid",
        ),
    ),
)
def test_epoch_metadata_is_rejected_before_any_live_mutation(
    tmp_path: Path,
    field: str,
    value: object,
    category: str,
) -> None:
    model, optimizer, scheduler, scaler = _training_objects()
    valid = tmp_path / "valid.pt"
    _save(valid, model, optimizer, scheduler, scaler)
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    payload[field] = value
    malformed = tmp_path / f"{field}.pt"
    torch.save(payload, malformed)
    _mutate_live_state(model, optimizer, scheduler, scaler)
    before = _snapshot(model, optimizer, scheduler, scaler)

    with pytest.raises(TrainingCheckpointError, match=category):
        _load(malformed, model, optimizer, scheduler, scaler)

    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )


def test_next_epoch_beyond_configured_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, scaler = _training_objects()
    valid = tmp_path / "valid.pt"
    _save(valid, model, optimizer, scheduler, scaler)
    payload = torch.load(valid, map_location="cpu", weights_only=True)
    payload["next_epoch"] = 3
    payload["committed_metric_rows"] = 3
    corrupted = tmp_path / "future.pt"
    torch.save(payload, corrupted)
    _mutate_live_state(model, optimizer, scheduler, scaler)
    before = _snapshot(model, optimizer, scheduler, scaler)

    with pytest.raises(
        TrainingCheckpointError,
        match="training.checkpoint.next_epoch_beyond_configured",
    ):
        _load(
            corrupted,
            model,
            optimizer,
            scheduler,
            scaler,
            maximum_next_epoch=2,
        )

    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )


@pytest.mark.parametrize("failing_component", ("optimizer", "scheduler"))
def test_application_failure_rolls_back_all_training_and_rng_state(
    tmp_path: Path,
    failing_component: str,
) -> None:
    model, optimizer, scheduler, scaler = _training_objects()
    checkpoint = tmp_path / "checkpoint.pt"
    _save(checkpoint, model, optimizer, scheduler, scaler)
    _mutate_live_state(model, optimizer, scheduler, scaler)
    before = _snapshot(model, optimizer, scheduler, scaler)
    live_id = id(
        optimizer if failing_component == "optimizer" else scheduler
    )
    target_type = (
        type(optimizer)
        if failing_component == "optimizer"
        else type(scheduler)
    )
    original = target_type.load_state_dict
    live_calls = 0

    def partially_apply_then_fail(self, state):
        nonlocal live_calls
        result = original(self, state)
        if id(self) == live_id:
            live_calls += 1
            if live_calls == 1:
                raise RuntimeError(
                    f"injected {failing_component} application failure"
                )
        return result

    with patch.object(
        target_type,
        "load_state_dict",
        partially_apply_then_fail,
    ):
        with pytest.raises(
            TrainingCheckpointError,
            match="training.checkpoint.application_failed",
        ):
            _load(
                checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
            )

    assert live_calls == 2
    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )
