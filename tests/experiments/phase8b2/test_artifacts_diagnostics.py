from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from music_critic.experiments.phase8b2.accounting import (
    ComputeAccounting,
    validate_compute_matrix,
)
from music_critic.experiments.phase8b2.artifacts import (
    REQUIRED_ARTIFACTS,
    validate_aggregate_bundles,
    write_complete_artifact_bundle,
    write_json_once,
)
from music_critic.experiments.phase8b2.contracts import Phase8B2ContractError
from music_critic.experiments.phase8b2.diagnostics import encoder_diagnostics


def _account(forwards: int = 8) -> ComputeAccounting:
    return ComputeAccounting(
        logical_updates=2,
        policy_views=4,
        encoder_forwards=forwards,
        raw_samples_seen=4,
        nodes_seen=20,
        edges_seen=30,
        eligible_objective_rows=5,
        optimizer_updates_applied=2,
        optimizer_updates_skipped=0,
        wall_seconds=0.5,
        peak_allocated_vram_bytes=None,
        peak_reserved_vram_bytes=None,
    )


def test_compute_accounting_matches_primary_budgets() -> None:
    report = validate_compute_matrix(
        (("phase7a_control", _account()), ("onset_latent", _account())),
        comparison_mode="encoder_forward_matched",
    )
    assert "encoder_forwards" in report["matched_fields"]
    assert not report["cells"]["phase7a_control"][
        "cuda_vram_evidence_available"
    ]


def test_natural_accounting_allows_honest_forward_difference() -> None:
    validate_compute_matrix(
        (("phase7a_control", _account(4)), ("phase8a_mask_only", _account(16))),
        comparison_mode="natural_schedule",
    )
    with pytest.raises(Phase8B2ContractError, match="budget_mismatch"):
        validate_compute_matrix(
            (("phase7a_control", _account(4)), ("phase8a_mask_only", _account(16))),
            comparison_mode="encoder_forward_matched",
        )


def _bundle(cell_id: str, *, seed: int = 17) -> dict[str, object]:
    return {
        "protocol_fingerprint": "protocol",
        "comparison_mode": "encoder_forward_matched",
        "data_binding_fingerprint": "data",
        "initial_encoder_fingerprint": "encoder",
        "seed": seed,
        "cell_id": cell_id,
        "complete": True,
        "artifact_fingerprint": "fresh",
        "recomputed_artifact_fingerprint": "fresh",
        "test_access": False,
    }


def test_aggregate_bundle_validation_and_negative_paths() -> None:
    result = validate_aggregate_bundles((_bundle("a"), _bundle("b")))
    assert result["cell_count"] == 2
    for field, value, category in (
        ("protocol_fingerprint", "other", "protocol_fingerprint_mismatch"),
        ("comparison_mode", "natural_schedule", "comparison_mode_mixed"),
        ("data_binding_fingerprint", "other", "data_binding_mismatch"),
        ("initial_encoder_fingerprint", "other", "initial_encoder_mismatch"),
        ("complete", False, "incomplete_run"),
        ("recomputed_artifact_fingerprint", "stale", "stale_artifact"),
        ("test_access", True, "unauthorized_test_access"),
    ):
        changed = _bundle("b")
        changed[field] = value
        with pytest.raises(Phase8B2ContractError, match=category):
            validate_aggregate_bundles((_bundle("a"), changed))
    different_seed = _bundle("c", seed=29)
    different_seed["initial_encoder_fingerprint"] = "encoder-seed-29"
    result = validate_aggregate_bundles((_bundle("a"), different_seed))
    assert result["initial_encoder_fingerprints_by_seed"] == {
        "17": "encoder",
        "29": "encoder-seed-29",
    }


def test_immutable_artifact_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "comparison_protocol.json"
    write_json_once(path, {"version": "1.0.0"})
    with pytest.raises(Phase8B2ContractError, match="already_exists"):
        write_json_once(path, {"version": "changed"})


def test_complete_artifact_bundle_is_failure_atomic_and_manifested(
    tmp_path: Path,
) -> None:
    json_names = set(REQUIRED_ARTIFACTS) - {
        "run_manifest.json",
        "ssl_training_metrics.jsonl",
    }
    output = tmp_path / "bundle"
    result = write_complete_artifact_bundle(
        output,
        protocol_fingerprint="protocol",
        repository={"git_sha": "a" * 40, "dirty": False},
        environment={"device": "cpu"},
        cells=({"cell_id": "cell"},),
        json_artifacts={name: {"name": name} for name in json_names},
        ssl_metric_rows=({"step": 1},),
    )
    assert result["complete"]
    assert set(path.name for path in output.iterdir()) == set(
        REQUIRED_ARTIFACTS
    )
    with pytest.raises(
        Phase8B2ContractError, match="new_output_directory_required"
    ):
        write_complete_artifact_bundle(
            output,
            protocol_fingerprint="protocol",
            repository={"git_sha": "a" * 40, "dirty": False},
            environment={"device": "cpu"},
            cells=({"cell_id": "cell"},),
            json_artifacts={name: {"name": name} for name in json_names},
            ssl_metric_rows=(),
        )


def test_diagnostics_are_bounded_and_excluded_from_selection() -> None:
    original = {
        "note": torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        "onset": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "beat": torch.tensor([[1.0, 1.0], [0.5, 0.5]]),
        "bar": torch.tensor([[1.0, 0.0]]),
        "song": torch.tensor([[0.0, 0.0]]),
    }
    perturbed = {key: value + 0.1 for key, value in original.items()}
    report = encoder_diagnostics(original, perturbed)
    assert report["diagnostic_only"]
    assert not report["participates_in_primary_selection"]
    assert not report["pairwise_n_by_n_matrix_created"]
    assert report["retained_prediction_tensor_count"] == 0
    assert report["node_types"]["note"]["effective_rank"] > 0
    assert report["node_types"]["song"]["zero_norm_count"] == 1


def test_diagnostic_missing_groups_have_explicit_unavailable_reason() -> None:
    report = encoder_diagnostics(
        {"note": torch.ones(1, 2)},
        {"note": torch.ones(1, 2)},
    )
    assert report["unavailable_group_count"] == 4
    assert report["node_types"]["onset"]["unavailable"]["category"] == (
        "node_type_absent"
    )
