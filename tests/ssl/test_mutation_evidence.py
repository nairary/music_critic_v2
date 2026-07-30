"""Focused contracts for Phase 7A mutation evidence semantics."""

from __future__ import annotations

import copy
from hashlib import sha256
import json
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from music_critic.ssl.engine import (
    NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION,
    PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION,
    SSLTrainingError,
    _build_no_leakage_mutation_evidence,
    _build_pitch_sensitive_reconstruction_evidence,
    _fp32_pitch_mutation_diagnostics,
    _pitch_reconstruction_loss_changed,
)


def _mutation_provenance() -> dict[str, object]:
    return {
        "mutation_contract_version": "1.0.0",
        "mutation_policy": "midi_axis_reflection_v1",
        "mutation_policy_fingerprint": "a" * 64,
        "coherent_mutations": [
            {
                "dataset_id": "bounded",
                "piece_id": "piece-0",
                "mutation_instance_fingerprint": "b" * 64,
            }
        ],
    }


def _no_leakage_payload(
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "applicable": True,
        "mutation_applicable": True,
        "unavailable_reason": None,
        "raw_graph_stores_bit_exact_after_view": True,
        "runtime_source_binding": {"passed": True},
        "fixed_mask_plan": True,
        "fixed_prepared_binding_fingerprint": True,
        "online_embeddings_bit_exact_after_masked_mutation": True,
        "online_predictions_bit_exact_after_masked_mutation": True,
        "full_view_target_changed": True,
        "metrics_finite": True,
        **_mutation_provenance(),
    }
    payload.update(overrides)
    return payload


def _pitch_payload(
    diagnostics: dict[str, object],
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "applicable": True,
        "mutation_applicable": True,
        "unavailable_reason": None,
        "full_view_target_changed": True,
        "reconstruction_loss_changed": True,
        **diagnostics,
        **_mutation_provenance(),
    }
    payload.update(overrides)
    return payload


def _negative_margin_diagnostics(
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    prediction = torch.tensor([[0.0, 1.0]], dtype=dtype)
    correct_target = torch.tensor([[1.0, 0.0]])
    mutated_target = torch.tensor([[0.0, 1.0]])
    return _fp32_pitch_mutation_diagnostics(
        (prediction,),
        correct_target,
        mutated_target,
    )


def _canonical_evidence_fingerprint(
    evidence: dict[str, object],
) -> str:
    payload = {
        key: value
        for key, value in evidence.items()
        if key != "fingerprint"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def test_negative_preference_margin_is_not_an_acceptance_gate() -> None:
    diagnostics = _negative_margin_diagnostics()
    leakage = _build_no_leakage_mutation_evidence(
        _no_leakage_payload()
    )
    pitch = _build_pitch_sensitive_reconstruction_evidence(
        _pitch_payload(diagnostics)
    )

    assert diagnostics["correct_minus_mutated_margin"] < 0
    assert diagnostics["correct_target_preference_observed"] is False
    assert diagnostics["preference_status"] == "not_observed"
    assert diagnostics["preference_is_acceptance_criterion"] is False
    assert leakage["passed"] is True
    assert pitch["passed"] is True


@pytest.mark.parametrize(
    "exact_field",
    (
        "online_embeddings_bit_exact_after_masked_mutation",
        "online_predictions_bit_exact_after_masked_mutation",
    ),
)
def test_online_one_ulp_drift_fails_strict_no_leakage(
    exact_field: str,
) -> None:
    baseline = torch.tensor([1.0], dtype=torch.float32)
    drifted = torch.nextafter(
        baseline,
        torch.tensor([2.0], dtype=torch.float32),
    )
    assert torch.allclose(baseline, drifted)
    assert not torch.equal(baseline, drifted)

    evidence = _build_no_leakage_mutation_evidence(
        _no_leakage_payload(
            **{exact_field: torch.equal(baseline, drifted)}
        )
    )

    assert evidence[exact_field] is False
    assert evidence["passed"] is False


@pytest.mark.parametrize(
    "field",
    ("full_view_target_changed", "reconstruction_loss_changed"),
)
def test_ineffective_reconstruction_challenge_fails(
    field: str,
) -> None:
    evidence = _build_pitch_sensitive_reconstruction_evidence(
        _pitch_payload(
            _negative_margin_diagnostics(),
            **{field: False},
        )
    )

    assert evidence[field] is False
    assert evidence["passed"] is False


def test_evidence_objects_are_independent_and_domain_fingerprinted() -> None:
    diagnostics = _negative_margin_diagnostics()
    shared_provenance = _mutation_provenance()
    leakage_payload = _no_leakage_payload(**shared_provenance)
    pitch_payload = _pitch_payload(
        diagnostics,
        **shared_provenance,
    )
    leakage = _build_no_leakage_mutation_evidence(leakage_payload)
    pitch = _build_pitch_sensitive_reconstruction_evidence(
        pitch_payload
    )

    assert leakage is not pitch
    assert leakage["coherent_mutations"] is not pitch[
        "coherent_mutations"
    ]
    assert leakage["evidence_kind"] == "no_leakage_mutation"
    assert pitch["evidence_kind"] == "pitch_sensitive_reconstruction"
    assert leakage["contract_version"] == (
        NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION
    )
    assert pitch["contract_version"] == (
        PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION
    )
    assert leakage["fingerprint"] != pitch["fingerprint"]
    assert leakage["fingerprint"] == _canonical_evidence_fingerprint(
        leakage
    )
    assert pitch["fingerprint"] == _canonical_evidence_fingerprint(
        pitch
    )

    serialized = json.dumps(
        {
            "no_leakage_mutation_evidence": leakage,
            "pitch_sensitive_reconstruction_evidence": pitch,
        },
        allow_nan=False,
        sort_keys=True,
    )
    decoded = json.loads(serialized)
    assert decoded["no_leakage_mutation_evidence"] != decoded[
        "pitch_sensitive_reconstruction_evidence"
    ]
    leakage["coherent_mutations"][0]["piece_id"] = "changed"
    assert pitch["coherent_mutations"][0]["piece_id"] == "piece-0"


@pytest.mark.parametrize("source_dtype", (torch.float16, torch.bfloat16))
def test_low_precision_sources_use_fp32_diagnostics(
    source_dtype: torch.dtype,
) -> None:
    cosine_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
    norm_dtypes: list[torch.dtype] = []
    original_cosine = F.cosine_similarity
    original_norm = torch.linalg.vector_norm

    def cosine_spy(
        left: torch.Tensor,
        right: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        cosine_dtypes.append((left.dtype, right.dtype))
        return original_cosine(left, right, *args, **kwargs)

    def norm_spy(
        value: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        norm_dtypes.append(value.dtype)
        return original_norm(value, *args, **kwargs)

    prediction = torch.tensor(
        [[0.25, 0.75]],
        dtype=source_dtype,
    )
    correct_target = torch.tensor([[1.0, 0.0]])
    mutated_target = torch.tensor([[0.0, 1.0]])
    with (
        patch(
            "music_critic.ssl.engine.F.cosine_similarity",
            side_effect=cosine_spy,
        ),
        patch(
            "music_critic.ssl.engine.torch.linalg.vector_norm",
            side_effect=norm_spy,
        ),
    ):
        diagnostics = _fp32_pitch_mutation_diagnostics(
            (prediction,),
            correct_target,
            mutated_target,
        )

    assert cosine_dtypes == [
        (torch.float32, torch.float32),
        (torch.float32, torch.float32),
        (torch.float32, torch.float32),
    ]
    assert norm_dtypes == [torch.float32]
    assert diagnostics["source_dtype"] == str(
        source_dtype
    ).removeprefix("torch.")
    assert diagnostics["diagnostic_compute_dtype"] == "float32"
    assert diagnostics["margin_floor"] == pytest.approx(
        8.0 * torch.finfo(torch.float32).eps
    )
    assert diagnostics["target_distance_floor"] == pytest.approx(
        torch.finfo(torch.float32).eps
    )


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
)
def test_nonfinite_pitch_diagnostics_fail_before_serialization(
    nonfinite: float,
) -> None:
    prediction = torch.tensor([[nonfinite, 1.0]])
    correct_target = torch.tensor([[1.0, 0.0]])
    mutated_target = torch.tensor([[0.0, 1.0]])

    with pytest.raises(
        SSLTrainingError,
        match=r"^ssl\.training\.pitch_diagnostic_nonfinite$",
    ):
        _fp32_pitch_mutation_diagnostics(
            (prediction,),
            correct_target,
            mutated_target,
        )


@pytest.mark.parametrize(
    ("correct_loss", "mutated_loss"),
    (
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (float("-inf"), 1.0),
    ),
)
def test_nonfinite_reconstruction_losses_fail_before_serialization(
    correct_loss: float,
    mutated_loss: float,
) -> None:
    with pytest.raises(
        SSLTrainingError,
        match=r"^ssl\.training\.pitch_diagnostic_nonfinite$",
    ):
        _pitch_reconstruction_loss_changed(
            torch.tensor(correct_loss),
            torch.tensor(mutated_loss),
        )


def test_cpu_evidence_fingerprints_are_deterministic() -> None:
    diagnostics_left = _negative_margin_diagnostics()
    diagnostics_right = _negative_margin_diagnostics()
    assert diagnostics_left == diagnostics_right

    leakage_left = _build_no_leakage_mutation_evidence(
        _no_leakage_payload()
    )
    leakage_right = _build_no_leakage_mutation_evidence(
        _no_leakage_payload()
    )
    pitch_left = _build_pitch_sensitive_reconstruction_evidence(
        _pitch_payload(diagnostics_left)
    )
    pitch_right = _build_pitch_sensitive_reconstruction_evidence(
        _pitch_payload(diagnostics_right)
    )

    assert leakage_left == leakage_right
    assert pitch_left == pitch_right
    assert leakage_left["fingerprint"] == leakage_right["fingerprint"]
    assert pitch_left["fingerprint"] == pitch_right["fingerprint"]
    assert copy.deepcopy(leakage_left) == leakage_left
    assert copy.deepcopy(pitch_left) == pitch_left
