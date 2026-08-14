"""Phase 8B.1 transfer of accepted Phase 7A checkpoints.

New Phase 8B.1 checkpoints continue to use the strict failure-atomic SSL
checkpoint container.  The additive model metadata binds the objective
registry/config fingerprints.  This module owns only the explicit old-to-new
transfer path where new heads remain separately initialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch

from music_critic.ssl.checkpoint import SSLCheckpointError
from music_critic.ssl.multilevel import Phase8BMultilevelSSLModel


PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION = "1.2.0"
PHASE8B_PHASE7A_TRANSFER_REPORT_CONTRACT_VERSION = "1.2.0"
_NEW_HEAD_PREFIX = "phase8b_latent_heads."


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase7AToPhase8BTransferReport:
    """Exact old components loaded and new head tensors left initialized."""

    contract_version: str
    checkpoint_binding_contract_version: str
    source_checkpoint_sha256: str
    loaded_parameter_tensors: tuple[str, ...]
    separately_initialized_parameter_tensors: tuple[str, ...]
    loaded_parameter_count: int
    separately_initialized_parameter_count: int
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != PHASE8B_PHASE7A_TRANSFER_REPORT_CONTRACT_VERSION
            or self.checkpoint_binding_contract_version
            != PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
            or len(self.source_checkpoint_sha256) != 64
            or not self.loaded_parameter_tensors
            or not self.separately_initialized_parameter_tensors
            or not all(
                name.startswith(_NEW_HEAD_PREFIX)
                for name in self.separately_initialized_parameter_tensors
            )
            or self.loaded_parameter_count <= 0
            or self.separately_initialized_parameter_count <= 0
        ):
            raise ValueError("Phase 7A to Phase 8B.1 transfer report is invalid")
        if self.fingerprint != _fingerprint(self._payload()):
            raise ValueError("Phase 8B.1 transfer report fingerprint is invalid")

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "checkpoint_binding_contract_version": (
                self.checkpoint_binding_contract_version
            ),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "loaded_parameter_tensors": list(self.loaded_parameter_tensors),
            "separately_initialized_parameter_tensors": list(
                self.separately_initialized_parameter_tensors
            ),
            "loaded_parameter_count": self.loaded_parameter_count,
            "separately_initialized_parameter_count": (
                self.separately_initialized_parameter_count
            ),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def transfer_phase7a_checkpoint_to_phase8b(
    path: str | Path,
    model: Phase8BMultilevelSSLModel,
) -> Phase7AToPhase8BTransferReport:
    """Load every old model tensor and leave all new heads untouched.

    Validation finishes before model mutation.  Any application failure rolls
    the complete Phase 8B.1 model back bit-exactly.
    """

    if type(model) is not Phase8BMultilevelSSLModel:
        raise TypeError("Phase 8B.1 transfer requires the exact multilevel model")
    source = Path(path)
    try:
        checkpoint_bytes = source.read_bytes()
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SSLCheckpointError(
            f"phase8b.checkpoint.transfer_unreadable:{exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise SSLCheckpointError("phase8b.checkpoint.transfer_payload_invalid")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise SSLCheckpointError("phase8b.checkpoint.transfer_model_state_invalid")
    base_contract = super(
        Phase8BMultilevelSSLModel, model
    ).ssl_contract_metadata()
    if payload["metadata"].get("model_contract") != base_contract:
        raise SSLCheckpointError("phase8b.checkpoint.transfer_contract_mismatch")
    target_state = model.state_dict()
    old_names = tuple(
        name for name in target_state if not name.startswith(_NEW_HEAD_PREFIX)
    )
    new_names = tuple(
        name for name in target_state if name.startswith(_NEW_HEAD_PREFIX)
    )
    if tuple(model_state) != old_names:
        raise SSLCheckpointError("phase8b.checkpoint.transfer_keys_mismatch")
    for name in old_names:
        source_value = model_state[name]
        target_value = target_state[name]
        if (
            not isinstance(source_value, torch.Tensor)
            or source_value.shape != target_value.shape
            or source_value.dtype != target_value.dtype
        ):
            raise SSLCheckpointError(
                f"phase8b.checkpoint.transfer_tensor_mismatch:{name}"
            )
    original = {
        name: value.detach().clone() for name, value in target_state.items()
    }
    try:
        incompatible = model.load_state_dict(model_state, strict=False)
        if tuple(incompatible.missing_keys) != new_names or incompatible.unexpected_keys:
            raise SSLCheckpointError("phase8b.checkpoint.transfer_application_keys")
    except Exception as exc:
        try:
            model.load_state_dict(original, strict=True)
        except Exception as rollback_exc:
            raise SSLCheckpointError(
                f"phase8b.checkpoint.transfer_rollback_failed:{rollback_exc}"
            ) from exc
        if isinstance(exc, SSLCheckpointError):
            raise
        raise SSLCheckpointError(
            f"phase8b.checkpoint.transfer_application_failed:{exc}"
        ) from exc
    payload_report = {
        "contract_version": PHASE8B_PHASE7A_TRANSFER_REPORT_CONTRACT_VERSION,
        "checkpoint_binding_contract_version": (
            PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
        ),
        "source_checkpoint_sha256": sha256(checkpoint_bytes).hexdigest(),
        "loaded_parameter_tensors": list(old_names),
        "separately_initialized_parameter_tensors": list(new_names),
        "loaded_parameter_count": sum(target_state[name].numel() for name in old_names),
        "separately_initialized_parameter_count": sum(
            target_state[name].numel() for name in new_names
        ),
    }
    return Phase7AToPhase8BTransferReport(
        contract_version=PHASE8B_PHASE7A_TRANSFER_REPORT_CONTRACT_VERSION,
        checkpoint_binding_contract_version=(
            PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
        ),
        source_checkpoint_sha256=payload_report["source_checkpoint_sha256"],
        loaded_parameter_tensors=old_names,
        separately_initialized_parameter_tensors=new_names,
        loaded_parameter_count=payload_report["loaded_parameter_count"],
        separately_initialized_parameter_count=(
            payload_report["separately_initialized_parameter_count"]
        ),
        fingerprint=_fingerprint(payload_report),
    )


__all__ = [
    "PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION",
    "PHASE8B_PHASE7A_TRANSFER_REPORT_CONTRACT_VERSION",
    "Phase7AToPhase8BTransferReport",
    "transfer_phase7a_checkpoint_to_phase8b",
]
