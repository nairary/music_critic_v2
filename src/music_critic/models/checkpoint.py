"""Strict metadata-bound checkpoints for the Phase 6A baseline."""

from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
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
    """Atomically write a same-directory checkpoint."""

    destination = Path(path)
    payload: dict[str, Any] = {
        "metadata": checkpoint_metadata(model),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
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


def _validate_model_state(
    payload: object,
    model: LocalHeterogeneousBaseline,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise CheckpointContractError("checkpoint model state is missing or malformed")
    expected = model.state_dict()
    if set(payload) != set(expected):
        missing = sorted(set(expected) - set(payload))
        extra = sorted(set(payload) - set(expected))
        raise CheckpointContractError(
            f"checkpoint model-state keys are incompatible: "
            f"missing={missing}, extra={extra}"
        )
    for key, expected_value in expected.items():
        value = payload[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected_value.shape
            or value.dtype != expected_value.dtype
        ):
            raise CheckpointContractError(
                f"checkpoint model tensor {key!r} has incompatible shape or dtype"
            )
    return payload


def _validate_optimizer_state(
    payload: object,
    optimizer: torch.optim.Optimizer,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise CheckpointContractError(
            "checkpoint optimizer state is missing or malformed"
        )
    state = payload.get("state")
    saved_groups = payload.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(saved_groups, list):
        raise CheckpointContractError("checkpoint optimizer structure is malformed")
    current_groups = optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(current_groups) or len(saved_groups) != len(
        optimizer.param_groups
    ):
        raise CheckpointContractError(
            "checkpoint optimizer parameter-group count is incompatible"
        )
    parameter_by_saved_id: dict[object, torch.Tensor] = {}
    all_saved_ids = set()
    for saved_group, current_group, live_group in zip(
        saved_groups, current_groups, optimizer.param_groups
    ):
        if not isinstance(saved_group, Mapping):
            raise CheckpointContractError(
                "checkpoint optimizer parameter group is malformed"
            )
        saved_ids = saved_group.get("params")
        current_ids = current_group.get("params")
        live_parameters = live_group.get("params")
        if (
            not isinstance(saved_ids, list)
            or not isinstance(current_ids, list)
            or not isinstance(live_parameters, list)
            or len(saved_ids) != len(current_ids)
            or len(saved_ids) != len(live_parameters)
            or set(saved_group) != set(current_group)
        ):
            raise CheckpointContractError(
                "checkpoint optimizer parameter group is incompatible"
            )
        for saved_id, parameter in zip(saved_ids, live_parameters):
            if saved_id in all_saved_ids or not isinstance(
                parameter, torch.Tensor
            ):
                raise CheckpointContractError(
                    "checkpoint optimizer parameter mapping is incompatible"
                )
            all_saved_ids.add(saved_id)
            parameter_by_saved_id[saved_id] = parameter
    if not set(state) <= all_saved_ids:
        raise CheckpointContractError(
            "checkpoint optimizer state references an unknown parameter"
        )
    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, Mapping):
            raise CheckpointContractError(
                "checkpoint optimizer per-parameter state is malformed"
            )
        parameter = parameter_by_saved_id[parameter_id]
        for value in parameter_state.values():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise CheckpointContractError(
                    "checkpoint optimizer tensor shape or dtype is incompatible"
                )
    return payload


def load_baseline_checkpoint(
    path: str | Path,
    model: LocalHeterogeneousBaseline,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Reject incompatible metadata before mutating model/optimizer state."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointContractError(
            f"checkpoint payload cannot be loaded: {exc}"
        ) from exc
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
    try:
        model_state = _validate_model_state(
            payload.get("model_state"), model
        )
        optimizer_state = None
        if optimizer is not None:
            optimizer_state = _validate_optimizer_state(
                payload.get("optimizer_state"), optimizer
            )
    except CheckpointContractError:
        raise
    except Exception as exc:
        raise CheckpointContractError(
            f"checkpoint state structure is malformed: {exc}"
        ) from exc
    original_model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    original_optimizer_state = (
        copy.deepcopy(optimizer.state_dict())
        if optimizer is not None
        else None
    )
    try:
        model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            assert optimizer_state is not None
            optimizer.load_state_dict(optimizer_state)
    except Exception as exc:
        model.load_state_dict(original_model_state, strict=True)
        if optimizer is not None and original_optimizer_state is not None:
            optimizer.load_state_dict(original_optimizer_state)
        raise CheckpointContractError(
            f"checkpoint state application failed atomically: {exc}"
        ) from exc
    return actual


__all__ = [
    "CheckpointContractError",
    "checkpoint_metadata",
    "feature_registry_fingerprint",
    "load_baseline_checkpoint",
    "save_baseline_checkpoint",
]
