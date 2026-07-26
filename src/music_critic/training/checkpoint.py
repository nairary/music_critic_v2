"""Epoch-boundary reproducible training checkpoints for Phase 6C."""

from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import torch

from music_critic.models.checkpoint import (
    CheckpointContractError,
    _validate_model_state,
    _validate_optimizer_state,
)
from music_critic.training.models import (
    BaselineModel,
    model_contract_metadata,
)


TRAINING_CHECKPOINT_VERSION = "1.0.0"


class TrainingCheckpointError(ValueError):
    """Strict incompatibility or atomic-application failure."""


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def training_checkpoint_metadata(
    model: BaselineModel,
    *,
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> dict[str, object]:
    return {
        "training_checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "model_contract": model_contract_metadata(model),
        "resolved_config_fingerprint": _canonical_fingerprint(
            resolved_config
        ),
        "data_fingerprints": data_fingerprints,
        "data_fingerprint": _canonical_fingerprint(data_fingerprints),
        "resume_boundary": "epoch_only",
    }


def capture_rng_state() -> dict[str, object]:
    return {
        "python": _lists(random.getstate()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            list(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(_tuples(state["python"]))
    torch.set_rng_state(state["torch_cpu"])
    cuda = state["torch_cuda"]
    if cuda:
        if not torch.cuda.is_available():
            raise TrainingCheckpointError(
                "training.checkpoint.cuda_rng_unavailable"
            )
        torch.cuda.set_rng_state_all(cuda)


def _lists(value: object) -> object:
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value


def _tuples(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def save_training_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any,
    next_epoch: int,
    best_validation_loss: float | None,
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> None:
    if next_epoch < 0:
        raise TrainingCheckpointError(
            "training.checkpoint.next_epoch_invalid"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": training_checkpoint_metadata(
            model,
            resolved_config=resolved_config,
            data_fingerprints=data_fingerprints,
        ),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "scaler_state": scaler.state_dict(),
        "next_epoch": next_epoch,
        "best_validation_loss": best_validation_loss,
        "rng_state": capture_rng_state(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_training_checkpoint(
    path: str | Path,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any,
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> tuple[int, float | None]:
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise TrainingCheckpointError(
            f"training.checkpoint.unreadable:{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingCheckpointError(
            "training.checkpoint.payload_invalid"
        )
    expected = training_checkpoint_metadata(
        model,
        resolved_config=resolved_config,
        data_fingerprints=data_fingerprints,
    )
    if payload.get("metadata") != expected:
        raise TrainingCheckpointError(
            "training.checkpoint.metadata_mismatch"
        )
    next_epoch = payload.get("next_epoch")
    if (
        isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or next_epoch < 0
    ):
        raise TrainingCheckpointError(
            "training.checkpoint.next_epoch_invalid"
        )
    try:
        model_state = _validate_model_state(
            payload.get("model_state"), model
        )
        optimizer_state = _validate_optimizer_state(
            payload.get("optimizer_state"), optimizer
        )
    except CheckpointContractError as exc:
        raise TrainingCheckpointError(str(exc)) from exc
    expected_scheduler = scheduler is not None
    if (payload.get("scheduler_state") is not None) != expected_scheduler:
        raise TrainingCheckpointError(
            "training.checkpoint.scheduler_mismatch"
        )
    if not isinstance(payload.get("scaler_state"), dict):
        raise TrainingCheckpointError(
            "training.checkpoint.scaler_state_invalid"
        )
    if not isinstance(payload.get("rng_state"), dict):
        raise TrainingCheckpointError(
            "training.checkpoint.rng_state_invalid"
        )
    originals = {
        "model": {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": (
            None
            if scheduler is None
            else copy.deepcopy(scheduler.state_dict())
        ),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "rng": capture_rng_state(),
    }
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload["scaler_state"])
        restore_rng_state(payload["rng_state"])
    except Exception as exc:
        model.load_state_dict(originals["model"], strict=True)
        optimizer.load_state_dict(originals["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(originals["scheduler"])
        scaler.load_state_dict(originals["scaler"])
        restore_rng_state(originals["rng"])
        raise TrainingCheckpointError(
            f"training.checkpoint.application_failed:{exc}"
        ) from exc
    best = payload.get("best_validation_loss")
    if best is not None and not isinstance(best, float):
        raise TrainingCheckpointError(
            "training.checkpoint.best_metric_invalid"
        )
    return next_epoch, best


__all__ = [
    "TRAINING_CHECKPOINT_VERSION",
    "TrainingCheckpointError",
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
    "training_checkpoint_metadata",
]
