"""Failure-atomic epoch-boundary checkpoints for Phase 7A SSL."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn

from music_critic.models.checkpoint import (
    CheckpointContractError,
    _validate_model_state,
    _validate_optimizer_state,
)
from music_critic.ssl.field_registry import (
    MASKABLE_FIELD_REGISTRY_FINGERPRINT,
    MASKABLE_FIELD_REGISTRY_VERSION,
)
from music_critic.training.checkpoint import (
    capture_rng_state,
    restore_rng_state,
)


SSL_CHECKPOINT_CONTRACT_VERSION = "1.1.0"
SSL_EPOCH_JOURNAL_CONTRACT_VERSION = "1.1.0"
SSL_METRIC_ROW_VERSION = "1.1.0"


class SSLCheckpointError(ValueError):
    """Strict incompatibility or atomic checkpoint-application failure."""


@dataclass(frozen=True, slots=True)
class SSLResumeState:
    """Validated state needed to continue only at an epoch boundary."""

    next_epoch: int
    best_validation_loss: float | None
    epoch_journal: tuple[dict[str, object], ...]


def _canonical_fingerprint(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SSLCheckpointError(
            "ssl.checkpoint.non_canonical_metadata"
        ) from exc
    return sha256(encoded).hexdigest()


def _model_contract(model: nn.Module) -> dict[str, object]:
    method = getattr(model, "ssl_contract_metadata", None)
    if method is None or not callable(method):
        raise SSLCheckpointError(
            "ssl.checkpoint.model_contract_unavailable"
        )
    value = method()
    if not isinstance(value, dict):
        raise SSLCheckpointError(
            "ssl.checkpoint.model_contract_invalid"
        )
    return value


def _sha256_field(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise SSLCheckpointError(
            f"ssl.checkpoint.{name}_invalid"
        )
    return value


def _validated_data_fingerprints(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SSLCheckpointError(
            "ssl.checkpoint.data_fingerprints_invalid"
        )
    kind = value.get("kind")
    if kind == "bounded":
        expected = {
            "kind",
            "bounded_fixture_fingerprint",
            "split_fingerprint",
            "train_composition_fingerprint",
            "validation_composition_fingerprint",
            "validation_membership_fingerprint",
        }
        if set(value) != expected:
            raise SSLCheckpointError(
                "ssl.checkpoint.bounded_data_fingerprints_invalid"
            )
        for name in expected - {"kind"}:
            _sha256_field(value[name], name=name)
    elif kind == "corpus_cache":
        expected = {
            "kind",
            "index_fingerprints",
            "split_manifest_fingerprint",
            "train_composition_fingerprint",
            "validation_composition_fingerprint",
            "validation_membership_fingerprint",
        }
        if set(value) != expected:
            raise SSLCheckpointError(
                "ssl.checkpoint.corpus_data_fingerprints_invalid"
            )
        indices = value["index_fingerprints"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                for row in indices
            )
        ):
            raise SSLCheckpointError(
                "ssl.checkpoint.index_fingerprints_invalid"
            )
        identities = tuple(row[0] for row in indices)
        if (
            identities != tuple(sorted(identities))
            or len(identities) != len(set(identities))
        ):
            raise SSLCheckpointError(
                "ssl.checkpoint.index_fingerprints_order_invalid"
            )
        for _dataset_id, fingerprint in indices:
            _sha256_field(
                fingerprint,
                name="index_fingerprint",
            )
        for name in expected - {"kind", "index_fingerprints"}:
            _sha256_field(value[name], name=name)
    else:
        raise SSLCheckpointError(
            "ssl.checkpoint.data_fingerprint_kind_invalid"
        )
    return copy.deepcopy(value)


def ssl_checkpoint_metadata(
    model: nn.Module,
    *,
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> dict[str, object]:
    """Bind model, SSL policies, raw fields, data composition, and config."""

    validated_data = _validated_data_fingerprints(
        data_fingerprints
    )
    return {
        "ssl_checkpoint_contract_version": (
            SSL_CHECKPOINT_CONTRACT_VERSION
        ),
        "epoch_journal_contract_version": (
            SSL_EPOCH_JOURNAL_CONTRACT_VERSION
        ),
        "model_contract": _model_contract(model),
        "maskable_field_registry_version": (
            MASKABLE_FIELD_REGISTRY_VERSION
        ),
        "maskable_field_registry_fingerprint": (
            MASKABLE_FIELD_REGISTRY_FINGERPRINT
        ),
        "resolved_config_fingerprint": _canonical_fingerprint(
            resolved_config
        ),
        "data_fingerprints": validated_data,
        "data_fingerprint": _canonical_fingerprint(validated_data),
        "resume_boundary": "epoch_only",
    }


def _validate_epoch_state(
    *,
    next_epoch: object,
    best_validation_loss: object,
    epoch_journal: object,
    maximum_next_epoch: int | None = None,
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or next_epoch < 0
        or (
            maximum_next_epoch is not None
            and next_epoch > maximum_next_epoch
        )
    ):
        raise SSLCheckpointError(
            "ssl.checkpoint.next_epoch_invalid"
        )
    if best_validation_loss is not None and (
        isinstance(best_validation_loss, bool)
        or not isinstance(best_validation_loss, float)
        or not math.isfinite(best_validation_loss)
    ):
        raise SSLCheckpointError(
            "ssl.checkpoint.best_validation_loss_invalid"
        )
    if not isinstance(epoch_journal, (list, tuple)):
        raise SSLCheckpointError(
            "ssl.checkpoint.epoch_journal_invalid"
        )
    rows: list[dict[str, object]] = []
    finite_validation_losses: list[float] = []
    expected_row_fields = {
        "metric_row_version",
        "epoch",
        "next_epoch",
        "learning_rate_used",
        "next_learning_rate",
        "train",
        "validation",
        "gradient_coverage",
    }
    for index, row in enumerate(epoch_journal):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
        ):
            raise SSLCheckpointError(
                "ssl.checkpoint.epoch_journal_row_invalid"
            )
        copied = dict(row)
        if (
            copied["metric_row_version"] != SSL_METRIC_ROW_VERSION
            or isinstance(copied["epoch"], bool)
            or not isinstance(copied["epoch"], int)
            or copied["epoch"] != index
            or isinstance(copied["next_epoch"], bool)
            or not isinstance(copied["next_epoch"], int)
            or copied["next_epoch"] != index + 1
        ):
            raise SSLCheckpointError(
                "ssl.checkpoint.epoch_journal_order_invalid"
            )
        for name in ("learning_rate_used", "next_learning_rate"):
            value = copied[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise SSLCheckpointError(
                    "ssl.checkpoint.epoch_journal_learning_rate_invalid"
                )
        if (
            not isinstance(copied["train"], Mapping)
            or (
                copied["validation"] is not None
                and not isinstance(copied["validation"], Mapping)
            )
            or (
                copied["gradient_coverage"] is not None
                and not isinstance(
                    copied["gradient_coverage"], Mapping
                )
            )
        ):
            raise SSLCheckpointError(
                "ssl.checkpoint.epoch_journal_metric_invalid"
            )
        validation = copied["validation"]
        if validation is not None:
            if "total_ssl_loss" not in validation:
                raise SSLCheckpointError(
                    "ssl.checkpoint."
                    "epoch_journal_validation_loss_invalid"
                )
            validation_loss = validation["total_ssl_loss"]
            if validation_loss is not None:
                if (
                    isinstance(validation_loss, bool)
                    or not isinstance(validation_loss, (int, float))
                    or not math.isfinite(float(validation_loss))
                ):
                    raise SSLCheckpointError(
                        "ssl.checkpoint."
                        "epoch_journal_validation_loss_invalid"
                    )
                finite_validation_losses.append(
                    float(validation_loss)
                )
        _canonical_fingerprint(copied)
        rows.append(copied)
    if len(rows) != next_epoch:
        raise SSLCheckpointError(
            "ssl.checkpoint.epoch_journal_length_invalid"
        )
    expected_best_validation_loss = (
        min(finite_validation_losses)
        if finite_validation_losses
        else None
    )
    if best_validation_loss != expected_best_validation_loss:
        raise SSLCheckpointError(
            "ssl.checkpoint.best_validation_loss_inconsistent"
        )
    return tuple(rows)


def save_ssl_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any,
    next_epoch: int,
    best_validation_loss: float | None,
    epoch_journal: tuple[dict[str, object], ...],
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> None:
    """Atomically replace one complete Phase 7A training checkpoint."""

    journal = _validate_epoch_state(
        next_epoch=next_epoch,
        best_validation_loss=best_validation_loss,
        epoch_journal=epoch_journal,
    )
    if not hasattr(scaler, "state_dict"):
        raise SSLCheckpointError(
            "ssl.checkpoint.scaler_invalid"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": ssl_checkpoint_metadata(
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
        "epoch_journal": list(journal),
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


def _validate_rng_state_by_restore(state: object) -> None:
    if not isinstance(state, dict):
        raise SSLCheckpointError(
            "ssl.checkpoint.rng_state_invalid"
        )
    original = capture_rng_state()
    try:
        restore_rng_state(state)
    except Exception as exc:
        raise SSLCheckpointError(
            f"ssl.checkpoint.rng_state_invalid:{exc}"
        ) from exc
    finally:
        restore_rng_state(original)


def load_ssl_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any,
    scaler: Any,
    maximum_next_epoch: int,
    resolved_config: dict[str, object],
    data_fingerprints: dict[str, object],
) -> SSLResumeState:
    """Validate every field, then apply all mutable state failure-atomically."""

    if (
        isinstance(maximum_next_epoch, bool)
        or not isinstance(maximum_next_epoch, int)
        or maximum_next_epoch < 0
    ):
        raise SSLCheckpointError(
            "ssl.checkpoint.maximum_next_epoch_invalid"
        )
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise SSLCheckpointError(
            f"ssl.checkpoint.unreadable:{exc}"
        ) from exc
    expected_fields = {
        "metadata",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "next_epoch",
        "best_validation_loss",
        "epoch_journal",
        "rng_state",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise SSLCheckpointError(
            "ssl.checkpoint.payload_fields_invalid"
        )
    expected_metadata = ssl_checkpoint_metadata(
        model,
        resolved_config=resolved_config,
        data_fingerprints=data_fingerprints,
    )
    if payload["metadata"] != expected_metadata:
        raise SSLCheckpointError(
            "ssl.checkpoint.metadata_mismatch"
        )
    journal = _validate_epoch_state(
        next_epoch=payload["next_epoch"],
        best_validation_loss=payload["best_validation_loss"],
        epoch_journal=payload["epoch_journal"],
        maximum_next_epoch=maximum_next_epoch,
    )
    try:
        model_state = _validate_model_state(
            payload["model_state"], model
        )
        optimizer_state = _validate_optimizer_state(
            payload["optimizer_state"], optimizer
        )
    except CheckpointContractError as exc:
        raise SSLCheckpointError(str(exc)) from exc
    if (payload["scheduler_state"] is None) != (scheduler is None):
        raise SSLCheckpointError(
            "ssl.checkpoint.scheduler_presence_mismatch"
        )
    if not isinstance(payload["scaler_state"], dict):
        raise SSLCheckpointError(
            "ssl.checkpoint.scaler_state_invalid"
        )
    _validate_rng_state_by_restore(payload["rng_state"])
    try:
        if scheduler is not None:
            scheduler_probe = copy.deepcopy(scheduler)
            scheduler_probe.load_state_dict(payload["scheduler_state"])
        scaler_probe = copy.deepcopy(scaler)
        scaler_probe.load_state_dict(payload["scaler_state"])
    except Exception as exc:
        raise SSLCheckpointError(
            f"ssl.checkpoint.auxiliary_state_invalid:{exc}"
        ) from exc
    originals = {
        "model": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
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
        try:
            model.load_state_dict(originals["model"], strict=True)
            optimizer.load_state_dict(originals["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(originals["scheduler"])
            scaler.load_state_dict(originals["scaler"])
            restore_rng_state(originals["rng"])
        except Exception as rollback_exc:
            raise SSLCheckpointError(
                "ssl.checkpoint.rollback_failed:"
                f"{rollback_exc}"
            ) from exc
        raise SSLCheckpointError(
            f"ssl.checkpoint.application_failed:{exc}"
        ) from exc
    return SSLResumeState(
        next_epoch=payload["next_epoch"],
        best_validation_loss=payload["best_validation_loss"],
        epoch_journal=journal,
    )


__all__ = [
    "SSL_CHECKPOINT_CONTRACT_VERSION",
    "SSL_EPOCH_JOURNAL_CONTRACT_VERSION",
    "SSL_METRIC_ROW_VERSION",
    "SSLCheckpointError",
    "SSLResumeState",
    "load_ssl_checkpoint",
    "save_ssl_checkpoint",
    "ssl_checkpoint_metadata",
]
