from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_phase9eb5a_analysisgnn_transposition as audit_script
from music_critic.experiments.analysisgnn.transposition import (
    CORRECTED_PROFILE_ID,
    OFFICIAL_PROFILE_ID,
    TRANSPOSITION_AUDIT_SCHEMA,
)


def _by_task() -> dict[str, dict[str, object]]:
    fixture = audit_script.check_fixture()
    return {row["task_id"]: row for row in fixture["head_summaries"]}


def test_source_free_fixture_is_deterministic_and_covers_all_heads() -> None:
    first = audit_script.check_fixture()
    second = audit_script.check_fixture()
    assert first == second
    assert first["schema"] == TRANSPOSITION_AUDIT_SCHEMA
    assert first["valid"] is True
    assert first["head_count"] == len(first["head_summaries"]) == 20
    assert first["semantic_fingerprint"] == (
        "b8aba86430fe2c87b250a5d1d1adc7557eed41ac54f24ae6cff32fd8bc815644"
    )


def test_source_free_check_never_calls_production_builder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("source-free check must not open the corpus")

    monkeypatch.setattr(audit_script, "build_audit", forbidden)
    assert audit_script.main(["--check"]) == 0
    assert '"valid": true' in capsys.readouterr().out


def test_test_descriptor_bytes_are_filtered_before_decoder_call() -> None:
    registry = [
        json.dumps({"record_id": "train"}) + "\n",
        json.dumps({"record_id": "validation"}) + "\n",
        json.dumps({"record_id": "test"}) + "\n",
    ]
    descriptors = [
        json.dumps({"record_id": "train"}) + "\n",
        json.dumps({"record_id": "validation"}) + "\n",
        json.dumps({"record_id": "test"}) + "\n",
    ]
    decoded: list[str] = []

    def decoder(line: str) -> object:
        value = json.loads(line)
        if value["record_id"] != "train":
            raise AssertionError("held-out target descriptor decoder must not run")
        decoded.append(value["record_id"])
        return value

    ids, selected = audit_script._selected_train_target_descriptors(
        registry,
        descriptors,
        train_ids={"train"},
        descriptor_decoder=decoder,
    )
    assert ids == {"train", "validation", "test"}
    assert decoded == ["train"]
    assert set(selected) == {"train"}


def test_fixture_proves_strict_test_lock_and_raw_only_access() -> None:
    lock = audit_script.check_fixture()["test_lock"]
    assert lock == {
        "test_assignment_record_count": 162,
        "test_assignments_seen": True,
        "test_evaluated": False,
        "test_raw_records_opened": 162,
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
    }


def test_profiles_registry_mapping_and_composition_are_frozen() -> None:
    fixture = audit_script.check_fixture()
    assert OFFICIAL_PROFILE_ID != CORRECTED_PROFILE_ID
    assert fixture["official_evidence_fingerprint"] != fixture[
        "corrected_policy_fingerprint"
    ]
    mapping = fixture["mapping_summary"]
    assert mapping["rows"] == 6408
    assert mapping["valid"] == 5956
    assert mapping["invalid_reason_counts"] == {
        "non_bijective_mapping": 36,
        "target_oov": 416,
    }
    assert mapping["composition"]["promised_checked"] == 26784
    assert mapping["composition"]["promised_failure_count"] == 0


def test_record_eligibility_identity_and_leakage_results_are_frozen() -> None:
    fixture = audit_script.check_fixture()
    eligibility = fixture["eligibility_summary"]
    assert eligibility["train_record_count"] == 1295
    assert eligibility["record_shift_row_count"] == 1295 * 12
    assert eligibility["records_with_12_valid_shifts"] == 1231
    assert eligibility["records_with_2_to_11_valid_shifts"] == 64
    assert eligibility["identity_only_records"] == 0
    assert eligibility["minimum_valid_shifts"] == 2
    assert eligibility["corrected_valid_variant_count"] == 15389
    assert eligibility["official_valid_variant_count"] == 15540
    assert eligibility["official_requested_variant_count"] == 15540
    assert eligibility["official_materialization_success_attested"] is False
    assert eligibility["invalid_reason_counts"] == {
        "non_bijective_mapping": 170,
        "target_oov": 217,
    }
    leakage = fixture["leakage_summary"]
    assert leakage["corrected_collision_variant_count"] == 0
    assert leakage["official_collision_variant_count"] == 0
    assert leakage["variants_moved_to_other_split"] == 0
    assert leakage["variants_counted_as_new_source_components"] is False


def test_quality_roman_note_degree_and_structural_distributions_are_invariant() -> None:
    proof = audit_script.check_fixture()["invariance_proof"]
    assert proof == {
        "checked_class_rows": 361,
        "mismatch_count": 0,
        "mismatches": [],
        "note_degree_distribution_identical": True,
        "phrase_section_negative_examples_created": False,
        "quality_17_distribution_identical": True,
        "roman_184_distribution_identical": True,
    }
    heads = _by_task()
    assert heads["quality"]["corrected_absent_classes"] == ["augmented sixth"]
    assert len(heads["roman_numeral"]["corrected_absent_classes"]) == 6
    assert heads["note_degree"]["augmentation_effect"] == "unchanged_invariant"
    assert heads["phrase"]["augmentation_effect"] == "unchanged_invariant"
    assert heads["section"]["augmentation_effect"] == "unchanged_invariant"


def test_pitch_dependent_head_results_are_frozen() -> None:
    heads = _by_task()
    assert (heads["local_key"]["raw_observed_class_count"], heads["local_key"]["corrected_observed_class_count"]) == (30, 48)
    assert heads["local_key"]["corrected_absent_classes"] == ["bbb", "cb"]
    assert (heads["tonicized_key"]["raw_observed_class_count"], heads["tonicized_key"]["corrected_observed_class_count"]) == (43, 48)
    assert heads["root"]["corrected_absent_classes"] == ["E###"]
    assert heads["bass"]["augmentation_effect"] == "coverage_recovered"
    assert heads["bass"]["corrected_absent_classes"] == []
    assert heads["pitch_class_set"]["augmentation_effect"] == "balance_improved"
    assert heads["local_key"]["source_component_count"] == 1194
    assert heads["local_key"]["transformed_component_support"] == 1194
    assert heads["local_key"]["component_shift_support"] == 14232


def test_recommendations_are_advisory_and_partition_all_heads() -> None:
    summary = audit_script.check_fixture()["recommendation_summary"]
    assert summary["official_counts"] == {
        "official_semantic_ambiguity": 2,
        "official_sparse": 12,
        "official_trainable_as_pinned": 1,
        "official_unobservable": 5,
    }
    assert summary["corrected_counts"] == {
        "auxiliary_candidate": 7,
        "derived_metric_candidate": 5,
        "primary_candidate": 1,
        "requires_policy_decision": 7,
    }
    assert summary["corrected_by_task"]["inversion"] == "primary_candidate"
    assert summary["corrected_by_task"]["local_key"] == "requires_policy_decision"


def test_fixture_contains_sha256_for_every_required_artifact() -> None:
    hashes = audit_script.check_fixture()["artifact_sha256"]
    assert set(hashes) == set(audit_script.OUTPUT_ARTIFACTS)
    assert all(not Path(name).is_absolute() for name in hashes)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in hashes.values()
    )
