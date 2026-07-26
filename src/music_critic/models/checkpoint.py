"""Strict metadata-bound checkpoints for the Phase 6A baseline."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch

from music_critic.data import SCHEMA_VERSION
from music_critic.graph import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    RAW_FEATURE_REGISTRY,
)
from music_critic.models.baseline import LocalHeterogeneousBaseline
from music_critic.models.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    MODEL_CONTRACT_VERSION,
)
from music_critic.tasks import (
    TARGET_ENCODING_REGISTRY_VERSION,
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_fingerprint,
    target_encoding_contract_fingerprint,
)


class CheckpointContractError(ValueError):
    """Raised when checkpoint metadata does not match the current model."""


def feature_registry_fingerprint() -> str:
    """Fingerprint ordered Phase 3A feature specifications."""

    payload = {
        "version": RAW_FEATURE_REGISTRY.version,
        "specs": [asdict(spec) for spec in RAW_FEATURE_REGISTRY.specs],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def checkpoint_metadata(
    model: LocalHeterogeneousBaseline,
) -> dict[str, object]:
    """Build every compatibility field required before loading weights."""

    return {
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "model_contract_version": MODEL_CONTRACT_VERSION,
        "model_config": model.config.to_dict(),
        "canonical_schema_version": SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "feature_registry_version": RAW_FEATURE_REGISTRY.version,
        "feature_registry_fingerprint": feature_registry_fingerprint(),
        "target_ontology_version": TARGET_ONTOLOGY_VERSION,
        "target_ontology_fingerprint": ontology_contract_fingerprint(),
        "target_encoding_registry_version": TARGET_ENCODING_REGISTRY_VERSION,
        "target_encoding_fingerprint": target_encoding_contract_fingerprint(),
        "active_task_heads": [asdict(spec) for spec in model.task_specs],
    }


def save_baseline_checkpoint(
    path: str | Path,
    model: LocalHeterogeneousBaseline,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write a checkpoint; callers own artifact location and retention."""

    payload: dict[str, Any] = {
        "metadata": checkpoint_metadata(model),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_baseline_checkpoint(
    path: str | Path,
    model: LocalHeterogeneousBaseline,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Reject incompatible metadata before mutating model/optimizer state."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("metadata"), dict
    ):
        raise CheckpointContractError("checkpoint payload is malformed")
    expected = checkpoint_metadata(model)
    actual = payload["metadata"]
    if actual != expected:
        keys = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise CheckpointContractError(
            f"checkpoint metadata is incompatible: differing={keys}"
        )
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise CheckpointContractError(
                "checkpoint has no requested optimizer state"
            )
        optimizer.load_state_dict(payload["optimizer_state"])
    return actual


__all__ = [
    "CheckpointContractError",
    "checkpoint_metadata",
    "feature_registry_fingerprint",
    "load_baseline_checkpoint",
    "save_baseline_checkpoint",
]
