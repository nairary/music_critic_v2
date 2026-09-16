"""Strict model-only artifacts for the enabled harmonic-fusion ablation.

The existing corrected/POP loaders and training resume contracts are unchanged.
This small format reuses their checked-byte, contract/state and RNG validators;
it never restores an optimizer, scheduler, sampler, scaler or saved RNG. The
caller owns source/split authorization and checkpoint training provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import PathLike

import torch

from music_critic.experiments.pop909_cl_frozen.checkpoint import (
    FrozenCheckpointError as HarmonicFusionCheckpointError,
    _checked_artifact,
    _preserve_caller_rng,
    _validate_model_contract,
    _validate_model_state,
)

from .corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from .harmonic_logit_fusion import HarmonicFusionModel, fusion_model_contract


FUSION_CHECKPOINT_SCHEMA = "HarmonicFusionModelOnlyCheckpoint@1.0.0"
FUSION_CHECKPOINT_EVIDENCE_SCHEMA = "HarmonicFusionCheckpointEvidence@1.0.0"
_FIELDS = frozenset({
    "schema_version", "model_contract", "model_contract_fingerprint",
    "model_state", "model_state_fingerprint",
})


@dataclass(frozen=True, slots=True)
class LoadedFusionCheckpoint:
    model: HarmonicFusionModel
    evidence: dict[str, object]


def create_model_payload(model: HarmonicFusionModel) -> dict[str, object]:
    """Snapshot enabled base+fusion tensors; perform no file access or RNG draw.

    This is a model snapshot, not evidence of how it was trained. The disabled
    wrapper is outside this first schema; its base has the old model format.
    """
    if not isinstance(model, HarmonicFusionModel) or model.fusion is None:
        raise HarmonicFusionCheckpointError(
            "fusion_checkpoint.enabled_model_required", "enabled HarmonicFusionModel required"
        )
    contract = fusion_model_contract(model)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {
        "schema_version": FUSION_CHECKPOINT_SCHEMA,
        "model_contract": contract,
        "model_contract_fingerprint": contract["fingerprint"],
        "model_state": state,
        "model_state_fingerprint": model_state_fingerprint(state),
    }


def load_fusion_checkpoint(
    path: str | PathLike[str],
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> LoadedFusionCheckpoint:
    """Validate the exact artifact, then freeze a fresh enabled CPU model.

    A bounded read supplies the same bytes to SHA validation and restricted
    deserialization. All envelope, complete contract and tensor validation
    happens before strict state loading. There is no unsafe fallback or
    metadata-driven constructor. Caller RNG survives every failure path.
    """
    with _preserve_caller_rng():
        content = _checked_artifact(path, expected_sha256, expected_size_bytes)
        try:
            payload = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise HarmonicFusionCheckpointError(
                "checkpoint.restricted_load_failed", "restricted model-only load failed; no unsafe fallback"
            ) from exc
        if not isinstance(payload, dict):
            raise HarmonicFusionCheckpointError("checkpoint.payload_invalid", "dictionary envelope required")
        if payload.get("schema_version") != FUSION_CHECKPOINT_SCHEMA:
            raise HarmonicFusionCheckpointError("checkpoint.schema_invalid", "fusion model-only schema required")
        if set(payload) != _FIELDS:
            raise HarmonicFusionCheckpointError("checkpoint.envelope_keys_mismatch", "exact model-only fields required")
        with torch.device("cpu"):
            base = CorrectedAnalysisGNNModel().to(dtype=torch.float32)
            model = HarmonicFusionModel(base, enabled=True).to(dtype=torch.float32)
        contract = fusion_model_contract(model)
        if payload["model_contract_fingerprint"] != contract["fingerprint"]:
            raise HarmonicFusionCheckpointError("checkpoint.model_contract_mismatch", "current enabled fusion contract required")
        _validate_model_contract(payload["model_contract"], contract)
        _validate_model_state(payload["model_state"], model)
        actual_fingerprint = model_state_fingerprint(payload["model_state"])
        if actual_fingerprint != payload["model_state_fingerprint"]:
            raise HarmonicFusionCheckpointError("checkpoint.state_fingerprint_mismatch", "tensor fingerprint mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        if model_state_fingerprint(model) != actual_fingerprint:
            raise HarmonicFusionCheckpointError("checkpoint.loaded_state_mismatch", "loaded tensors changed")
        model.eval().requires_grad_(False)
        evidence = {
            "schema_version": FUSION_CHECKPOINT_EVIDENCE_SCHEMA,
            "checkpoint_schema": FUSION_CHECKPOINT_SCHEMA,
            "artifact_sha256": expected_sha256,
            "artifact_size_bytes": expected_size_bytes,
            "model_contract": contract,
            "model_contract_fingerprint": contract["fingerprint"],
            "model_state_fingerprint": actual_fingerprint,
            "contract_verification": "full_payload_and_current_fingerprint",
            "load_policy": "cpu_weights_only_strict_model_state",
            "optimizer_restored": False,
            "checkpoint_rng_applied": False,
            "caller_rng_preserved": True,
            "model_training": False,
            "parameters_require_grad": False,
        }
        return LoadedFusionCheckpoint(model=model, evidence=evidence)


__all__ = [
    "FUSION_CHECKPOINT_SCHEMA", "FUSION_CHECKPOINT_EVIDENCE_SCHEMA",
    "HarmonicFusionCheckpointError", "LoadedFusionCheckpoint",
    "create_model_payload", "load_fusion_checkpoint",
]
