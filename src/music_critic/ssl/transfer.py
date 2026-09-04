"""Strict Phase 7A pretrained-encoder export and supervised transfer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import Tensor, nn

from music_critic.models import HierarchicalHeterogeneousBaseline
from music_critic.models.hierarchical_checkpoint import (
    hierarchical_checkpoint_metadata,
)


ENCODER_EXPORT_CONTRACT_VERSION = "1.0.0"

_TRANSFERRED_PREFIXES = (
    "local_baseline.encoder.",
    "context_encoder.pooling.",
    "context_encoder.transformer.",
    "context_encoder.fusion.",
)


class EncoderTransferError(ValueError):
    """Raised when an SSL encoder export is incompatible or malformed."""


@dataclass(frozen=True, slots=True)
class EncoderTransferReport:
    """Exact evidence for transferred and deliberately untouched parameters."""

    contract_version: str
    loaded_parameters: tuple[str, ...]
    untouched_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != ENCODER_EXPORT_CONTRACT_VERSION:
            raise EncoderTransferError(
                "ssl.transfer.report_version_incompatible"
            )
        if (
            self.loaded_parameters != tuple(sorted(self.loaded_parameters))
            or self.untouched_parameters
            != tuple(sorted(self.untouched_parameters))
            or set(self.loaded_parameters) & set(self.untouched_parameters)
        ):
            raise EncoderTransferError(
                "ssl.transfer.report_parameter_partition_invalid"
            )


def _hierarchical_encoder(model: nn.Module) -> HierarchicalHeterogeneousBaseline:
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, HierarchicalHeterogeneousBaseline):
        return encoder
    if isinstance(model, HierarchicalHeterogeneousBaseline):
        return model
    raise EncoderTransferError(
        "ssl.transfer.hierarchical_encoder_required"
    )


def _transfer_names(model: HierarchicalHeterogeneousBaseline) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in model.state_dict()
            if name.startswith(_TRANSFERRED_PREFIXES)
        )
    )


def _encoder_contract(
    model: HierarchicalHeterogeneousBaseline,
) -> dict[str, object]:
    """Remove task-head-only fields from the strict Phase 6B metadata."""

    metadata = copy.deepcopy(hierarchical_checkpoint_metadata(model))
    for name in (
        "active_task_heads",
        "target_ontology_version",
        "target_ontology_fingerprint",
        "target_encoding_registry_version",
        "target_encoding_fingerprint",
    ):
        metadata.pop(name)
    for config_name in ("model_config", "phase6a_local_config"):
        config = metadata[config_name]
        assert isinstance(config, dict)
        config.pop("task_hidden_dim", None)
        config.pop("task_weights", None)
    return metadata


def export_pretrained_encoder_state(model: nn.Module) -> dict[str, object]:
    """Return a versioned CPU export containing no supervised task-head state."""

    encoder = _hierarchical_encoder(model)
    names = _transfer_names(encoder)
    if not names:
        raise EncoderTransferError(
            "ssl.transfer.encoder_parameter_set_empty"
        )
    state = encoder.state_dict()
    return {
        "metadata": {
            "encoder_export_contract_version": (
                ENCODER_EXPORT_CONTRACT_VERSION
            ),
            "hierarchical_encoder_contract": (
                _encoder_contract(encoder)
            ),
            "parameter_names": list(names),
        },
        "encoder_state": {
            name: state[name].detach().cpu().clone() for name in names
        },
    }


def validate_pretrained_encoder_export_structure(
    export: object,
) -> Mapping[str, Tensor]:
    """Validate that an artifact is an encoder-only export envelope.

    The metadata-free envelope is retained for the Phase 9C initial-scratch
    export.  Versioned SSL exports must carry the complete Phase 7A manifest.
    A training checkpoint is never accepted by filtering its ``model_state``.
    """

    if not isinstance(export, Mapping) or set(export) not in (
        {"encoder_state"},
        {"metadata", "encoder_state"},
    ):
        raise EncoderTransferError("ssl.transfer.export_fields_invalid")
    state = export["encoder_state"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(name, str)
            or not name.startswith(_TRANSFERRED_PREFIXES)
            or not isinstance(value, Tensor)
            for name, value in state.items()
        )
    ):
        raise EncoderTransferError("ssl.transfer.encoder_state_invalid")
    metadata = export.get("metadata")
    if metadata is None:
        return state
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "encoder_export_contract_version",
        "hierarchical_encoder_contract",
        "parameter_names",
    }:
        raise EncoderTransferError("ssl.transfer.metadata_fields_invalid")
    if (
        metadata["encoder_export_contract_version"]
        != ENCODER_EXPORT_CONTRACT_VERSION
    ):
        raise EncoderTransferError("ssl.transfer.export_version_incompatible")
    if not isinstance(metadata["hierarchical_encoder_contract"], Mapping):
        raise EncoderTransferError("ssl.transfer.encoder_contract_invalid")
    expected_names = tuple(sorted(state))
    if metadata["parameter_names"] != list(expected_names):
        raise EncoderTransferError(
            "ssl.transfer.parameter_manifest_incompatible"
        )
    return state


def _validate_export(
    export: object,
    model: HierarchicalHeterogeneousBaseline,
) -> Mapping[str, Tensor]:
    if not isinstance(export, Mapping) or set(export) != {
        "metadata",
        "encoder_state",
    }:
        raise EncoderTransferError("ssl.transfer.export_fields_invalid")
    metadata = export["metadata"]
    state = export["encoder_state"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "encoder_export_contract_version",
        "hierarchical_encoder_contract",
        "parameter_names",
    }:
        raise EncoderTransferError("ssl.transfer.metadata_fields_invalid")
    if (
        metadata["encoder_export_contract_version"]
        != ENCODER_EXPORT_CONTRACT_VERSION
    ):
        raise EncoderTransferError(
            "ssl.transfer.export_version_incompatible"
        )
    if (
        metadata["hierarchical_encoder_contract"]
        != _encoder_contract(model)
    ):
        raise EncoderTransferError(
            "ssl.transfer.encoder_contract_incompatible"
        )
    expected_names = _transfer_names(model)
    if metadata["parameter_names"] != list(expected_names):
        raise EncoderTransferError(
            "ssl.transfer.parameter_manifest_incompatible"
        )
    actual_names = set(state) if isinstance(state, Mapping) else set()
    if not isinstance(state, Mapping) or actual_names != set(expected_names):
        missing = sorted(set(expected_names) - actual_names)
        unexpected = sorted(actual_names - set(expected_names))
        raise EncoderTransferError(
            "ssl.transfer.state_keys_incompatible:"
            f"missing={missing},unexpected={unexpected}"
        )
    expected_state = model.state_dict()
    for name in expected_names:
        value = state[name]
        expected = expected_state[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise EncoderTransferError(
                f"ssl.transfer.tensor_incompatible:{name}"
            )
    return state


def load_pretrained_encoder_state(
    supervised_model: HierarchicalHeterogeneousBaseline,
    export: object,
) -> EncoderTransferReport:
    """Failure-atomically load only representation parameters into supervision."""

    if not isinstance(
        supervised_model, HierarchicalHeterogeneousBaseline
    ):
        raise EncoderTransferError(
            "ssl.transfer.supervised_hierarchical_model_required"
        )
    transferred = _validate_export(export, supervised_model)
    original = {
        name: value.detach().clone()
        for name, value in supervised_model.state_dict().items()
    }
    loaded_names = tuple(sorted(transferred))
    untouched_names = tuple(sorted(set(original) - set(loaded_names)))
    merged = {
        name: (
            transferred[name].detach().to(
                device=value.device,
                dtype=value.dtype,
            )
            if name in transferred
            else value
        )
        for name, value in original.items()
    }
    try:
        supervised_model.load_state_dict(merged, strict=True)
        current = supervised_model.state_dict()
        changed_untouched = [
            name
            for name in untouched_names
            if not torch.equal(current[name], original[name])
        ]
        if changed_untouched:
            raise EncoderTransferError(
                "ssl.transfer.untouched_parameter_changed:"
                + ",".join(changed_untouched)
            )
    except Exception as exc:
        try:
            supervised_model.load_state_dict(
                copy.deepcopy(original), strict=True
            )
        except Exception as rollback_exc:
            raise EncoderTransferError(
                "ssl.transfer.rollback_failed:"
                f"{rollback_exc}"
            ) from exc
        if isinstance(exc, EncoderTransferError):
            raise
        raise EncoderTransferError(
            f"ssl.transfer.application_failed:{exc}"
        ) from exc
    return EncoderTransferReport(
        contract_version=ENCODER_EXPORT_CONTRACT_VERSION,
        loaded_parameters=loaded_names,
        untouched_parameters=untouched_names,
    )


def save_pretrained_encoder_export(
    path: str | Path, model: nn.Module
) -> None:
    """Failure-atomically save a tensor-only encoder export."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(export_pretrained_encoder_state(model), temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "ENCODER_EXPORT_CONTRACT_VERSION",
    "EncoderTransferError",
    "EncoderTransferReport",
    "export_pretrained_encoder_state",
    "load_pretrained_encoder_state",
    "save_pretrained_encoder_export",
    "validate_pretrained_encoder_export_structure",
]
