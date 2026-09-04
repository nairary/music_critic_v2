from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CORRECTED_CHECKPOINT_SCHEMA,
    build_source_free_fixture,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_UPDATE_BUDGET,
    full_runtime_config,
    full_training_contract,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    semantic_mapping_rows,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    EXPECTED_PAIR_COUNT,
    EXPECTED_TRANSFORMATION_KINDS,
    audit_graph_transform,
    audit_record_observation_transform,
    audit_sidecar_targets,
    check_compact_fixture,
    cross_head_checks,
    independent_target_oracle,
    source_free_runtime_regression,
    transformation_matrix,
    validate_checkpoint_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5f_transposition_correctness.json"


def test_b5a_registry_is_the_only_complete_20_head_matrix() -> None:
    rows = transformation_matrix()
    assert len(rows) == 20
    assert {row["task_id"]: row["transformation_kind"] for row in rows} == (
        EXPECTED_TRANSFORMATION_KINDS
    )
    assert all(row["classification_matches"] for row in rows)
    assert EXPECTED_PAIR_COUNT == (1295 + 162) * 12


def test_independent_semantic_oracle_accepts_every_valid_b5a_mapping() -> None:
    valid = [row for row in semantic_mapping_rows() if row.valid]
    failures = [
        asdict(row)
        for row in valid
        if row.target_semantic_value is None
        or not independent_target_oracle(
            row.source_task_id,
            row.source_semantic_value,
            row.target_semantic_value,
            shift_pc=row.shift_pc,
        )
    ]
    assert valid
    assert failures == []


def test_raw_graph_forward_oracles_pass_but_tritone_round_trip_exposes_defect() -> None:
    batch, _sidecar = build_source_free_fixture()
    graph = batch.raw_graph_batch.to_data_list()[0]
    rows = {shift: audit_graph_transform(graph, shift_pc=shift) for shift in SHIFT_PCS}
    assert all(row["runtime_path_matches_contract"] for row in rows.values())
    assert all(row["identity_exact"] for row in rows.values())
    assert all(row["round_trip_valid"] for shift, row in rows.items() if shift != 6)
    assert rows[6]["round_trip_valid"] is False
    assert rows[6]["round_trip_differences"] == ["note.x_cat"]
    assert rows[6]["invalid_reasons"] == ["graph_round_trip_mismatch"]


def test_sidecar_target_oracles_masks_ids_and_semantic_round_trips_pass() -> None:
    _batch, sidecar = build_source_free_fixture()
    for shift in SHIFT_PCS:
        row = audit_sidecar_targets(sidecar, shift_pc=shift)
        assert row["target_vocabulary_closed"] is True
        assert row["spelling_valid"] is True
        assert row["round_trip_valid"] is True
        assert row["masks_preserved"] is True
        assert row["entity_ids_preserved"] is True
        assert row["invalid_reasons"] == []
    assert audit_sidecar_targets(sidecar, shift_pc=6)["tritone_rows_checked"] > 0


def test_record_observation_api_matches_oracle_for_all_shifts() -> None:
    _batch, sidecar = build_source_free_fixture()
    for shift in SHIFT_PCS:
        row = audit_record_observation_transform(sidecar, shift_pc=shift)
        assert row["mismatch_count"] == 0
        assert row["round_trip_mismatch_count"] == 0
        assert row["masks_preserved"] is True
        assert row["entity_ids_preserved"] is True


def test_runtime_forward_then_alignment_matches_b5a_while_round_trip_fails() -> None:
    row = source_free_runtime_regression()
    assert row["shift_count"] == 12
    assert row["runtime_path_matches_contract"] is True
    assert row["routing_mismatch_count"] == 0
    assert row["identity_exact"] is True
    assert row["round_trip_passed"] is False


def test_cross_head_missing_context_is_not_automatic_success() -> None:
    batch, sidecar = build_source_free_fixture()
    graph = batch.raw_graph_batch.to_data_list()[0]
    row = cross_head_checks(sidecar, graph, shift_pc=1)
    assert row["failed"] is False
    assert row["passed"] is False
    assert "note_degree_with_pitch_key" in row["not_checkable"]


def test_seed17_schedule_reproduces_exact_b5d_evidence() -> None:
    row = check_compact_fixture(FIXTURE)["schedule"]
    assert row["record_draws"] == 20_000
    assert row["shift_draw_counts"] == {
        "0": 1539,
        "1": 1624,
        "2": 1630,
        "3": 1682,
        "4": 1690,
        "5": 1650,
        "6": 1706,
        "7": 1747,
        "8": 1636,
        "9": 1641,
        "10": 1781,
        "11": 1674,
    }
    assert row["record_schedule_fingerprint"] == (
        "67f4401806f2d5419bb849449aef811fd54dfbca62588c5a1543dbbe6c1b63f8"
    )
    assert row["C0_transposition_schedule_fingerprint"] == (
        "af937f0ece2ffc459a093b5d8a19be815c4159653b545059eee723c3bc71bb2b"
    )
    assert row["C1_transposition_schedule_fingerprint"] == (
        "745aef3bf213228635bbd4926a5f9d61f4dc26a425434b3757535eeccae4ef4a"
    )
    assert row["record_schedules_equal"] is True
    assert row["limited_record_count"] == 64


def test_checkpoint_metadata_requires_every_available_b5d_binding() -> None:
    profile = "C0"
    expected_state = "a" * 64
    expected_record_schedule = "b" * 64
    architecture = str(corrected_model_contract(CorrectedAnalysisGNNModel())["fingerprint"])
    payload = {
        "schema_version": CORRECTED_CHECKPOINT_SCHEMA,
        "phase": "9E-B5D",
        "resolved_config": full_runtime_config(profile).to_dict(),
        "applied_update": FULL_UPDATE_BUDGET,
        "full_training_contract_fingerprint": full_training_contract()["fingerprint"],
        "model_contract_fingerprint": architecture,
        "model_state_fingerprint": expected_state,
        "sampler_state": {
            "record_schedule_fingerprint": expected_record_schedule,
        },
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    result = validate_checkpoint_metadata(
        payload,
        profile=profile,
        expected_model_fingerprint=expected_state,
        expected_model_contract_fingerprint=architecture,
        expected_record_schedule_fingerprint=expected_record_schedule,
    )
    assert result["valid"] is True
    broken = dict(payload)
    broken["applied_update"] = FULL_UPDATE_BUDGET - 1
    result = validate_checkpoint_metadata(
        broken,
        profile=profile,
        expected_model_fingerprint=expected_state,
        expected_model_contract_fingerprint=architecture,
        expected_record_schedule_fingerprint=expected_record_schedule,
    )
    assert result["valid"] is False
    assert result["checks"]["applied_update"] is False


def test_committed_fixture_records_defect_without_checkpoint_claim() -> None:
    value = check_compact_fixture(FIXTURE)
    assert value["final_status"] == "implementation_or_contract_defect"
    assert value["status"]["transposition_correctness_passed"] is False
    assert value["status"]["round_trip_passed"] is False
    assert value["status"]["checkpoint_diagnostics_run"] is False
    assert value["status"]["ready_for_soft_augmentation"] is False
    assert value["status"]["test_loader_created"] is False
    assert value["status"]["test_targets_read"] is False
    assert value["status"]["test_metrics_computed"] is False
