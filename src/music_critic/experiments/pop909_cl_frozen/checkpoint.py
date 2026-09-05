"""Restricted, CPU, model-only loading for the frozen POP909-CL evaluator.

Supported schemas are ``CorrectedAnalysisGNNModelOnlyCheckpoint@1.0.0``
(created by :func:`create_model_only_payload`) and restricted-load-compatible
``CorrectedAnalysisGNNCheckpoint@1.0.0`` envelopes. The latter binds the full
current corrected model contract by its exact fingerprint; its historical
contract payload is unavailable. Resume metadata is neither restored nor used.

Historical training envelopes include NumPy RNG arrays and environment objects
that may be rejected by ``weights_only=True``. Actual trained checkpoints have
not been inspected or validated for compatibility. Rejection is a documented
gap, never grounds for unrestricted pickle loading or global allowlisting.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import random
import re
from typing import Iterator

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)


MODEL_ONLY_CHECKPOINT_SCHEMA = "CorrectedAnalysisGNNModelOnlyCheckpoint@1.0.0"
CORRECTED_TRAINING_CHECKPOINT_SCHEMA = "CorrectedAnalysisGNNCheckpoint@1.0.0"
CHECKPOINT_EVIDENCE_SCHEMA = "POP909CLFrozenCheckpointEvidence@1.0.0"
_MODEL_ONLY_FIELDS = frozenset(
    {
        "schema_version",
        "model_contract",
        "model_contract_fingerprint",
        "model_state",
        "model_state_fingerprint",
    }
)


class FrozenCheckpointError(ValueError):
    """An artifact is incompatible with the frozen model-only boundary."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(f"{category}: {message}")


@dataclass(frozen=True, slots=True)
class LoadedModelCheckpoint:
    model: CorrectedAnalysisGNNModel
    evidence: dict[str, object]


@contextmanager
def _preserve_caller_rng() -> Iterator[None]:
    """Preserve caller state, without initializing CUDA or using saved RNG."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def create_model_only_payload(model: CorrectedAnalysisGNNModel) -> dict[str, object]:
    """Snapshot a corrected model as primitive metadata and plain CPU tensors.

    This function performs no file access. It does not establish training/data
    provenance: an artifact's run registration remains an external requirement.
    """

    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    contract = corrected_model_contract(model)
    return {
        "schema_version": MODEL_ONLY_CHECKPOINT_SCHEMA,
        "model_contract": contract,
        "model_contract_fingerprint": contract["fingerprint"],
        "model_state": state,
        "model_state_fingerprint": model_state_fingerprint(state),
    }


def _checked_artifact(
    path: str | os.PathLike[str], expected_sha256: str, expected_size_bytes: int
) -> bytes:
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise FrozenCheckpointError("artifact.expected_sha256_invalid", "64 lowercase hex digits required")
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise FrozenCheckpointError("artifact.expected_size_invalid", "positive byte count required")
    # The bounded read and digest refer to the same bytes later deserialized;
    # reopening the path after hashing would introduce an artifact race.
    with Path(path).open("rb") as stream:
        content = stream.read(expected_size_bytes + 1)
    if len(content) != expected_size_bytes:
        raise FrozenCheckpointError("artifact.size_mismatch", f"expected {expected_size_bytes}, read {len(content)}")
    if sha256(content).hexdigest() != expected_sha256:
        raise FrozenCheckpointError("artifact.sha256_mismatch", "artifact does not match expected digest")
    return content


def _validate_model_state(state: object, model: CorrectedAnalysisGNNModel) -> None:
    if not isinstance(state, dict) or any(type(name) is not str for name in state):
        raise FrozenCheckpointError("checkpoint.state_invalid", "string-keyed tensor state required")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise FrozenCheckpointError(
            "checkpoint.state_keys_mismatch",
            f"missing={sorted(set(expected) - set(state))}; unexpected={sorted(set(state) - set(expected))}",
        )
    for name, value in state.items():
        reference = expected[name]
        if type(value) is not torch.Tensor:
            raise FrozenCheckpointError("checkpoint.tensor_type_invalid", name)
        if value.layout != torch.strided or value.device.type != "cpu":
            raise FrozenCheckpointError("checkpoint.tensor_layout_invalid", name)
        if value.shape != reference.shape:
            raise FrozenCheckpointError("checkpoint.tensor_shape_mismatch", name)
        if value.dtype != reference.dtype:
            raise FrozenCheckpointError("checkpoint.tensor_dtype_mismatch", name)
        if not bool(torch.isfinite(value).all()):
            raise FrozenCheckpointError("checkpoint.tensor_nonfinite", name)


def _validate_model_contract(supplied: object, expected: dict[str, object]) -> None:
    if not isinstance(supplied, dict):
        raise FrozenCheckpointError("checkpoint.model_contract_mismatch", "dictionary model contract required")
    try:
        supplied_bytes = canonical_json(supplied).encode("utf-8")
        recomputed_fingerprint = fingerprint(
            {key: value for key, value in supplied.items() if key != "fingerprint"}
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise FrozenCheckpointError(
            "checkpoint.model_contract_mismatch", "model contract is not canonical JSON"
        ) from exc
    if supplied.get("fingerprint") != recomputed_fingerprint:
        raise FrozenCheckpointError(
            "checkpoint.model_contract_mismatch", "embedded model contract fingerprint does not match payload"
        )
    # Python equality merges distinct JSON values such as 18 and 18.0 or
    # True and 1. Compare the exact canonical representation the hash binds.
    if supplied_bytes != canonical_json(expected).encode("utf-8"):
        raise FrozenCheckpointError(
            "checkpoint.model_contract_mismatch", "complete canonical corrected contract differs"
        )


def load_model_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> LoadedModelCheckpoint:
    """Validate artifact and full state, then return a fresh frozen eval model.

    Only CPU ``torch.load(weights_only=True)`` is used. There is no resume
    import, optimizer construction, saved-RNG application, unsafe fallback, or
    metadata-driven constructor. Caller Python/NumPy/Torch RNG is preserved
    on every success/failure path, including model-constructor exceptions.
    Split/access guards belong to the caller and must run before this API.
    """

    with _preserve_caller_rng():
        content = _checked_artifact(path, expected_sha256, expected_size_bytes)
        try:
            payload = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise FrozenCheckpointError(
                "checkpoint.restricted_load_failed",
                "artifact is not supported by restricted loading; no unsafe fallback is allowed",
            ) from exc
        if not isinstance(payload, dict):
            raise FrozenCheckpointError("checkpoint.payload_invalid", "dictionary envelope required")
        schema = payload.get("schema_version")
        if schema not in (MODEL_ONLY_CHECKPOINT_SCHEMA, CORRECTED_TRAINING_CHECKPOINT_SCHEMA):
            raise FrozenCheckpointError("checkpoint.schema_invalid", str(schema))
        if schema == MODEL_ONLY_CHECKPOINT_SCHEMA and set(payload) != _MODEL_ONLY_FIELDS:
            raise FrozenCheckpointError("checkpoint.envelope_keys_mismatch", "exact model-only envelope required")
        # Canonical corrected weights are float32 independently of caller's
        # default dtype/device. Only this local model is converted.
        with torch.device("cpu"):
            model = CorrectedAnalysisGNNModel().to(dtype=torch.float32)
        contract = corrected_model_contract(model)
        if payload.get("model_contract_fingerprint") != contract["fingerprint"]:
            raise FrozenCheckpointError("checkpoint.model_contract_mismatch", "current corrected contract fingerprint required")
        contract_available = "model_contract" in payload
        if contract_available:
            _validate_model_contract(payload["model_contract"], contract)
        _validate_model_state(payload.get("model_state"), model)
        state = payload["model_state"]
        actual_fingerprint = model_state_fingerprint(state)
        if actual_fingerprint != payload.get("model_state_fingerprint"):
            raise FrozenCheckpointError("checkpoint.state_fingerprint_mismatch", "model tensors differ from bound fingerprint")
        model.load_state_dict(state, strict=True)
        if model_state_fingerprint(model) != actual_fingerprint:
            raise FrozenCheckpointError("checkpoint.loaded_state_mismatch", "strictly loaded state changed")
        model.eval()
        model.requires_grad_(False)
        evidence: dict[str, object] = {
            "schema_version": CHECKPOINT_EVIDENCE_SCHEMA,
            "checkpoint_schema": schema,
            "artifact_sha256": expected_sha256,
            "artifact_size_bytes": expected_size_bytes,
            "model_contract": contract,
            "model_contract_fingerprint": contract["fingerprint"],
            "model_state_fingerprint": actual_fingerprint,
            "contract_verification": (
                "full_payload_and_current_fingerprint"
                if contract_available
                else "reconstructed_current_contract_fingerprint_only"
            ),
            "historical_contract_payload_available": contract_available,
            "load_policy": "cpu_weights_only_strict_model_state",
            "optimizer_restored": False,
            "checkpoint_rng_applied": False,
            "caller_rng_preserved": True,
            "model_training": False,
            "parameters_require_grad": False,
        }
        return LoadedModelCheckpoint(model=model, evidence=evidence)
