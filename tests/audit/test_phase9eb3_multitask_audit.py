from __future__ import annotations

from pathlib import Path

from scripts.audit_phase9eb3_analysisgnn_multitask import (
    DEFAULT_FIXTURE,
    DEFAULT_SCIENTIFIC_FIXTURE,
    check_fixture,
    check_scientific_fixture,
    main,
)


def test_source_free_audit_fixture_is_self_fingerprinted_and_ready() -> None:
    fixture = check_fixture(DEFAULT_FIXTURE)
    assert fixture["valid"] is True
    assert fixture["ready"] is True
    assert fixture["source_component_counts"] == {
        "full_raw": 1507,
        "paper_candidate": 1507,
    }
    assert fixture["split"]["record_counts"] == {
        "train": 1295,
        "validation": 162,
        "test": 162,
    }
    assert fixture["split"]["component_leakage_failure_count"] == 0
    corrected = fixture["corrected_v2_contract"]
    compatibility = fixture["analysisgnn_compatibility_contract"]
    assert corrected["corrected_quality_class_count"] == 17
    assert corrected["corrected_roman_numeral_class_count"] == 184
    assert corrected["entity_type"] == "harmonic_event"
    assert corrected["paper_compatible"] is False
    assert corrected["joint_structural_support"]["train"] > 0
    assert corrected["joint_structural_support"]["validation"] > 0
    assert corrected["joint_structural_support"]["test"] == "not_evaluated"
    assert compatibility["compatibility_quality_class_count"] == 15
    assert compatibility["entity_type"] == "note"
    assert compatibility["metric_evaluated"] is False
    assert fixture["model_implemented"] is False
    assert fixture["training_run"] is False
    assert fixture["validation_inference_run"] is False
    assert fixture["test_evaluated"] is False
    assert fixture["test_targets_used_for_evaluation"] is False
    assert fixture["test_lock"]["test_metrics_computed"] is False


def test_independent_pinned_scientific_fixture_is_self_fingerprinted() -> None:
    fixture = check_scientific_fixture(DEFAULT_SCIENTIFIC_FIXTURE)
    assert fixture["external_commit"] == (
        "e115182fb29b74bdcb6bf3547ed427d967580947"
    )
    assert len(fixture["pinned_code"]["quality"]["semantic_labels"]) == 15
    assert fixture["corrected_v2"]["quality_class_count"] == 17
    assert fixture["corrected_v2"]["roman_numeral_class_count"] == 184
    assert fixture["pinned_code"]["code_only_excluded_heads"] == ["staff"]


def test_check_cli_needs_no_external_corpus(capsys) -> None:
    assert main(["--check"]) == 0
    output = capsys.readouterr().out
    assert '"ready": true' in output
    assert '"valid": true' in output


def test_fixture_contains_only_sha256_artifact_locks() -> None:
    fixture = check_fixture(DEFAULT_FIXTURE)
    assert fixture["artifact_sha256"]
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in fixture["artifact_sha256"].values()
    )
    assert all(not Path(name).is_absolute() for name in fixture["artifact_sha256"])
