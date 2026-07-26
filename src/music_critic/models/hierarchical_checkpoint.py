"""Failure-atomic compatibility-bound Phase 6B checkpoints."""

from __future__ import annotations

import copy
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from music_critic.data import SCHEMA_VERSION
from music_critic.graph import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    RAW_FEATURE_REGISTRY,
)
from music_critic.models.checkpoint import (
    CheckpointContractError,
    _validate_model_state,
    _validate_optimizer_state,
    feature_registry_fingerprint,
)
from music_critic.models.contracts import (
    CHECKPOINT_CONTRACT_VERSION,
    ENCODER_OUTPUT_VERSION,
    MODEL_CONTRACT_VERSION,
)
from music_critic.models.hierarchical_baseline import (
    HierarchicalHeterogeneousBaseline,
)
from music_critic.models.hierarchy_contracts import (
    COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION,
    HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION,
    HIERARCHICAL_ENCODER_OUTPUT_VERSION,
    HIERARCHICAL_MODEL_CONTRACT_VERSION,
    HIERARCHY_POOLING_CONTRACT_VERSION,
    TOP_DOWN_FUSION_CONTRACT_VERSION,
)
from music_critic.tasks import (
    TARGET_ENCODING_REGISTRY_VERSION,
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_fingerprint,
    target_encoding_contract_fingerprint,
)


def hierarchical_checkpoint_metadata(
    model: HierarchicalHeterogeneousBaseline,
) -> dict[str, object]:
    """Bind Phase 6B to every inherited and newly introduced contract."""

    return {
        "hierarchical_checkpoint_contract_version": (
            HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION
        ),
        "hierarchical_model_contract_version": (
            HIERARCHICAL_MODEL_CONTRACT_VERSION
        ),
        "hierarchical_encoder_output_version": (
            HIERARCHICAL_ENCODER_OUTPUT_VERSION
        ),
        "hierarchy_pooling_contract_version": (
            HIERARCHY_POOLING_CONTRACT_VERSION
        ),
        "coarse_token_sequence_contract_version": (
            COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION
        ),
        "top_down_fusion_contract_version": (
            TOP_DOWN_FUSION_CONTRACT_VERSION
        ),
        "phase6a_model_contract_version": MODEL_CONTRACT_VERSION,
        "phase6a_encoder_output_version": ENCODER_OUTPUT_VERSION,
        "phase6a_checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "model_config": model.config.to_dict(),
        "phase6a_local_config": model.config.local_config().to_dict(),
        "canonical_schema_version": SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "feature_registry_version": RAW_FEATURE_REGISTRY.version,
        "feature_registry_fingerprint": feature_registry_fingerprint(),
        "target_ontology_version": TARGET_ONTOLOGY_VERSION,
        "target_ontology_fingerprint": ontology_contract_fingerprint(),
        "target_encoding_registry_version": TARGET_ENCODING_REGISTRY_VERSION,
        "target_encoding_fingerprint": target_encoding_contract_fingerprint(),
        "active_task_heads": [
            asdict(spec) for spec in model.task_specs
        ],
    }


def save_hierarchical_checkpoint(
    path: str | Path,
    model: HierarchicalHeterogeneousBaseline,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Atomically replace one same-directory Phase 6B checkpoint."""

    destination = Path(path)
    payload: dict[str, Any] = {
        "metadata": hierarchical_checkpoint_metadata(model),
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


def load_hierarchical_checkpoint(
    path: str | Path,
    model: HierarchicalHeterogeneousBaseline,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Validate completely before failure-atomic model/optimizer application."""

    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise CheckpointContractError(
            f"checkpoint payload cannot be loaded: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("metadata"), dict
    ):
        raise CheckpointContractError("checkpoint payload is malformed")
    expected = hierarchical_checkpoint_metadata(model)
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
    "hierarchical_checkpoint_metadata",
    "load_hierarchical_checkpoint",
    "save_hierarchical_checkpoint",
]
