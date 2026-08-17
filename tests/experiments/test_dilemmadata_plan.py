from __future__ import annotations

import copy

import pytest

from music_critic.experiments.dilemmadata import (
    DILEMMADATA_PRIMARY_VARIANTS,
    DILEMMADATA_SEEDS,
    DilemmadataExperimentPlanError,
    build_dilemmadata_experiment_plan,
    dilemmadata_command_matrix,
    dilemmadata_report_bundle_manifest,
    validate_dilemmadata_experiment_plan,
    verify_dilemmadata_report_bundle,
)


def _plan():
    return build_dilemmadata_experiment_plan(
        raw_index_fingerprint="a" * 64,
        target_cache_index_fingerprint="b" * 64,
        split_manifest_fingerprint="c" * 64,
        sample_schedule_fingerprint="d" * 64,
        phase7a_encoder_export_path="artifacts/phase7a-encoder.pt",
        phase7a_encoder_export_sha256="e" * 64,
        phase7a_source_checkpoint_sha256="f" * 64,
        phase8b_encoder_export_path="artifacts/phase8b-encoder.pt",
        phase8b_encoder_export_sha256="1" * 64,
        phase8b_source_checkpoint_sha256="2" * 64,
    )


def test_primary_plan_is_closed_equal_budget_and_not_executed() -> None:
    plan = _plan()
    validate_dilemmadata_experiment_plan(plan)
    assert tuple(
        row["variant_id"] for row in plan["variants"] if row["primary"]
    ) == DILEMMADATA_PRIMARY_VARIANTS
    assert tuple(plan["seeds"]) == DILEMMADATA_SEEDS
    assert plan["execution_state"] == "planned_not_executed"
    assert plan["hardware"]["accelerator"] == "NVIDIA GeForce RTX 3090"
    assert plan["fixed_training"]["reconstruction_weight"] == 0
    assert plan["fixed_training"]["class_weight_policy"] == "unweighted"
    assert plan["comparison"]["equal_sample_schedule"] is True
    assert plan["comparison"]["unit"] == "connected_component"

    matrix = dilemmadata_command_matrix(plan)
    assert matrix["long_training_executed"] is False
    assert len(matrix["commands"]) == 9
    for command in matrix["commands"]:
        assert command["execution_state"] == "not_started"
        assert "experiment=dilemmadata_scratch_vs_ssl" in command["argv"]
        assert "model=hierarchical" in command["argv"]
        assert "data=dilemmadata" in command["argv"]
        assert "device=cuda" in command["argv"]
        assert any(
            value.startswith("transfer.sample_schedule_fingerprint=")
            for value in command["argv"]
        )
    manifest = dilemmadata_report_bundle_manifest(plan, matrix)
    verify_dilemmadata_report_bundle(plan, matrix, manifest)
    assert manifest["long_training_executed"] is False
    assert set(manifest["artifacts"]) == {
        "plan.json",
        "command_matrix.json",
    }


def test_plan_tampering_and_partial_optional_cell_fail_closed() -> None:
    plan = copy.deepcopy(_plan())
    plan["seeds"][0] = 18
    with pytest.raises(DilemmadataExperimentPlanError, match="plan_invalid"):
        validate_dilemmadata_experiment_plan(plan)
    with pytest.raises(
        DilemmadataExperimentPlanError, match="optional_export_incomplete"
    ):
        build_dilemmadata_experiment_plan(
            raw_index_fingerprint="a" * 64,
            target_cache_index_fingerprint="b" * 64,
            split_manifest_fingerprint="c" * 64,
            sample_schedule_fingerprint="d" * 64,
            phase7a_encoder_export_path="phase7a.pt",
            phase7a_encoder_export_sha256="e" * 64,
            phase7a_source_checkpoint_sha256="f" * 64,
            phase8b_encoder_export_path="phase8b.pt",
            phase8b_encoder_export_sha256="1" * 64,
            phase8b_source_checkpoint_sha256="2" * 64,
            optional_equal_encoder_export_path="equal.pt",
        )
    plan = _plan()
    matrix = dilemmadata_command_matrix(plan)
    manifest = dilemmadata_report_bundle_manifest(plan, matrix)
    manifest["long_training_executed"] = True
    with pytest.raises(
        DilemmadataExperimentPlanError, match="report_bundle_invalid"
    ):
        verify_dilemmadata_report_bundle(plan, matrix, manifest)
