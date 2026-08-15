"""Target-blind SSL boundary and exact target-mutation evidence."""

from __future__ import annotations

from typing import Mapping

from music_critic.experiments.phase8b2.contracts import (
    Phase8B2ContractError,
    fingerprint,
)


LEAKAGE_EVIDENCE_CONTRACT_VERSION = "1.0.0"
PROHIBITED_MODEL_INPUT_FIELDS = frozenset(
    {
        "targets",
        "target_sidecars",
        "target_provenance",
        "confidence",
        "split",
        "split_label",
        "evaluation_labels",
        "theory_targets",
        "chord_targets",
    }
)
_INVARIANT_FIELDS = (
    "raw_input_fingerprint",
    "ssl_plan_fingerprint",
    "logits_fingerprint",
    "loss_fingerprint",
    "gradient_fingerprint",
    "checkpoint_fingerprint",
    "transferred_encoder_fingerprint",
)


def validate_raw_only_ssl_inputs(model_inputs: Mapping[str, object]) -> None:
    """Reject supervision-shaped fields before a model forward."""

    prohibited = sorted(set(model_inputs) & PROHIBITED_MODEL_INPUT_FIELDS)
    if prohibited:
        raise Phase8B2ContractError(
            "phase8b2.leakage.supervision_in_model_input:"
            + ",".join(prohibited)
        )
    if "raw_graph" not in model_inputs and "raw_graph_batch" not in model_inputs:
        raise Phase8B2ContractError(
            "phase8b2.leakage.raw_graph_input_missing"
        )


def target_mutation_evidence(
    original: Mapping[str, object],
    mutated: Mapping[str, object],
    *,
    mutation_kind: str,
) -> dict[str, object]:
    """Require exact invariance through transfer for a target-only mutation."""

    if mutation_kind not in {"changed", "removed", "replaced"}:
        raise Phase8B2ContractError(
            "phase8b2.leakage.mutation_kind_invalid"
        )
    missing = [
        name
        for name in _INVARIANT_FIELDS
        if name not in original or name not in mutated
    ]
    if missing:
        raise Phase8B2ContractError(
            "phase8b2.leakage.evidence_field_missing:"
            + ",".join(missing)
        )
    mismatches = [
        name for name in _INVARIANT_FIELDS if original[name] != mutated[name]
    ]
    if mismatches:
        raise Phase8B2ContractError(
            "phase8b2.leakage.target_mutation_changed_ssl:"
            + ",".join(mismatches)
        )
    artifact = {
        "leakage_evidence_contract_version": (
            LEAKAGE_EVIDENCE_CONTRACT_VERSION
        ),
        "mutation_kind": mutation_kind,
        "target_sidecar_fingerprint_before": original.get(
            "target_sidecar_fingerprint"
        ),
        "target_sidecar_fingerprint_after": mutated.get(
            "target_sidecar_fingerprint"
        ),
        "target_sidecar_changed": original.get(
            "target_sidecar_fingerprint"
        )
        != mutated.get("target_sidecar_fingerprint"),
        "invariants": {name: original[name] for name in _INVARIANT_FIELDS},
        "plans_equal": True,
        "logits_equal": True,
        "losses_equal": True,
        "gradients_equal": True,
        "checkpoints_equal": True,
        "transferred_encoders_equal": True,
        "passed": True,
    }
    if not artifact["target_sidecar_changed"]:
        raise Phase8B2ContractError(
            "phase8b2.leakage.target_mutation_not_observed"
        )
    artifact["fingerprint"] = fingerprint(artifact)
    return artifact


__all__ = [
    "LEAKAGE_EVIDENCE_CONTRACT_VERSION",
    "PROHIBITED_MODEL_INPUT_FIELDS",
    "target_mutation_evidence",
    "validate_raw_only_ssl_inputs",
]
