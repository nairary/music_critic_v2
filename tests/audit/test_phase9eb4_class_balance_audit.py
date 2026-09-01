from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_phase9eb4_analysisgnn_class_balance as audit_script
from music_critic.experiments.analysisgnn.class_balance import CLASS_BALANCE_SCHEMA


def test_source_free_fixture_is_self_fingerprinted_and_covers_all_heads() -> None:
    first = audit_script.check_fixture()
    second = audit_script.check_fixture()
    assert first == second
    assert first["schema"] == CLASS_BALANCE_SCHEMA
    assert first["valid"] is True
    assert first["head_count"] == len(first["head_summaries"]) == 20
    assert first["split_counts"] == {"train": 1295, "validation": 162, "test": 162}
    assert first["semantic_fingerprint"] == (
        "4b1edf9f47815bafa5e197be87b9331a19789142c0625ef4aceda1f87649df4d"
    )


def test_source_free_check_does_not_call_production_builder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("source-free check must not open the corpus")

    monkeypatch.setattr(audit_script, "build_audit", forbidden)
    assert audit_script.main(["--check"]) == 0
    assert '"valid": true' in capsys.readouterr().out


def test_test_descriptor_decoder_is_filtered_before_invocation() -> None:
    registry = [
        json.dumps({"record_id": "train"}) + "\n",
        json.dumps({"record_id": "test"}) + "\n",
    ]
    descriptors = [
        json.dumps({"record_id": "train", "targets": [1]}) + "\n",
        json.dumps({"record_id": "test", "targets": [999]}) + "\n",
    ]
    decoded: list[str] = []

    def decoder(line: str) -> object:
        value = json.loads(line)
        if value["record_id"] == "test":
            raise AssertionError("TEST descriptor decoder must not run")
        decoded.append(value["record_id"])
        return value

    registry_ids, selected = audit_script._selected_target_descriptors(
        registry,
        descriptors,
        allowed_ids={"train"},
        descriptor_decoder=decoder,
    )
    assert registry_ids == {"train", "test"}
    assert decoded == ["train"]
    assert set(selected) == {"train"}


def test_fixture_proves_strict_test_lock() -> None:
    lock = audit_script.check_fixture()["test_lock"]
    assert lock == {
        "test_assignment_record_count": 162,
        "test_assignments_seen": True,
        "test_evaluated": False,
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
    }


def test_head_summary_contains_frozen_recommendation_partition() -> None:
    fixture = audit_script.check_fixture()
    assert fixture["recommendation_groups"] == {
        "trainable": ["inversion"],
        "trainable_with_reweighting": [
            "primary_degree",
            "harmonic_rhythm",
            "cadence",
            "pedal",
            "chord_tone",
            "is_root",
            "is_bass",
        ],
        "insufficient_support": [
            "root",
            "bass",
            "secondary_degree",
            "quality",
            "pitch_class_set",
        ],
        "descriptive_only": [
            "local_key",
            "tonicized_key",
            "roman_numeral",
            "phrase",
            "section",
            "metrical_strength",
            "note_degree",
        ],
    }
    by_task = {row["task_id"]: row for row in fixture["head_summaries"]}
    assert by_task["quality"]["vocabulary_size"] == 17
    assert by_task["quality"]["train_observed_class_count"] == 16
    assert by_task["roman_numeral"]["vocabulary_size"] == 184
    assert by_task["inversion"]["recommendation"] == "trainable"


def test_quality_rows_include_augmented_classes_and_projection_support() -> None:
    quality = audit_script.check_fixture()["quality"]
    corrected = quality["corrected_quality_17"]["focus_classes"]
    plus_seven = corrected["augmented seventh chord"]
    plus_major_seven = corrected["augmented major tetrachord"]
    triad = corrected["augmented triad"]
    assert plus_seven["train"]["canonical_target_row_count"] == 245
    assert plus_seven["train"]["component_count"] == 38
    assert plus_seven["validation"]["canonical_target_row_count"] == 77
    assert plus_major_seven["train"]["canonical_target_row_count"] == 145
    assert plus_major_seven["train"]["component_count"] == 24
    assert plus_major_seven["validation"]["validation_tier"] == "fragile_validation"
    assert triad["train"]["canonical_target_row_count"] == 2403
    projected = quality["compatibility_quality_15"]["augmented_triad"]
    assert projected["train"]["canonical_target_row_count"] == 2793
    assert projected["train"]["component_count"] == 246
    assert projected["validation"]["canonical_target_row_count"] == 431
    corrected_balance = quality["corrected_quality_17"]["balance"]
    compatibility_balance = quality["compatibility_quality_15"]["balance"]
    assert corrected_balance == {
        "vocabulary_size": 17,
        "train_observed_class_count": 16,
        "validation_observed_class_count": 16,
        "train_absent_classes": ["augmented sixth"],
        "majority_share": 0.454997613164,
        "max_to_min_nonzero_ratio": 3575.91724137931,
        "normalized_entropy": 0.532869567161,
    }
    assert compatibility_balance["vocabulary_size"] == 15
    assert compatibility_balance["train_observed_class_count"] == 14
    assert compatibility_balance["normalized_entropy"] == 0.557048917368
    assert quality["projection"] == {
        "augmented seventh chord": "augmented triad",
        "augmented major tetrachord": "augmented triad",
        "other_classes": "identity",
        "missing": "mask",
    }


def test_roman_184_long_tail_and_concrete_rows_are_frozen() -> None:
    roman = audit_script.check_fixture()["roman_numeral_184"]
    assert roman["class_count"] == 184
    assert roman["train_absent_class_count"] == 6
    assert roman["validation_absent_class_count"] == 71
    assert roman["component_threshold_counts"] == {"lt_3": 56, "lt_10": 107, "lt_20": 125}
    assert roman["target_row_threshold_counts"] == {"lt_20": 65, "lt_100": 113, "lt_1000": 151}
    assert roman["required_vocabulary_evidence"] == {
        "none": False,
        "#VIIbvio7": False,
        "#VII": True,
        "bvio7": True,
    }
    bvio7 = next(row for row in roman["bottom_20_nonzero"] if row["class_value"] == "bvio7")
    assert bvio7 == {
        "class_id": 132,
        "class_value": "bvio7",
        "canonical_target_row_count": 4,
        "component_count": 1,
    }
    assert roman["validation_classes_absent_in_train"] == ["vii%9", "N+7", "bV+7", "#v7"]


def test_joint_event_and_note_summaries_keep_broadcast_units_separate() -> None:
    joint = audit_script.check_fixture()["joint_tuples"]
    corrected = joint["corrected_harmonic_event"]
    compatibility = joint["compatibility_note"]
    assert corrected["train"]["row_count"] == 98_715
    assert corrected["train"]["canonical_harmonic_target_rows"] == 98_715
    assert corrected["validation"]["row_count"] == 10_507
    assert corrected["train"]["unique_tuple_count"] == 3339
    assert len(corrected["validation_tuples_unseen_in_train"]) == 187
    assert compatibility["train"]["row_count"] == 187_548
    assert compatibility["train"]["canonical_harmonic_target_rows"] == 98_438
    assert compatibility["validation"]["row_count"] == 20_465
    assert compatibility["validation"]["canonical_harmonic_target_rows"] == 10_477
    assert len(compatibility["validation_tuples_unseen_in_train"]) == 185


def test_fixture_contains_all_required_artifact_sha256_values() -> None:
    fixture = audit_script.check_fixture()
    hashes = fixture["artifact_sha256"]
    assert set(hashes) == set(audit_script.OUTPUT_ARTIFACTS)
    assert all(not Path(name).is_absolute() for name in hashes)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in hashes.values()
    )
    weights = fixture["candidate_class_weight_summary"]
    assert weights["methods"] == [
        "inverse_frequency",
        "inverse_sqrt_frequency",
        "effective_number",
    ]
    assert weights["train_only"] is True
    assert weights["validation_counts_used"] is False
    assert weights["weighting_policy_frozen"] is False
    assert weights["policy_counts"] == {
        "class_weighting_candidate": 1,
        "component_balanced_sampling_candidate": 7,
        "head_not_ready": 12,
    }
