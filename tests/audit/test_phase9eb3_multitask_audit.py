from __future__ import annotations

from pathlib import Path

from scripts.audit_phase9eb3_analysisgnn_multitask import (
    DEFAULT_FIXTURE,
    check_fixture,
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
    assert fixture["joint_metric"]["train"] > 0
    assert fixture["joint_metric"]["validation"] > 0
    assert fixture["joint_metric"]["test"] == "not_evaluated"
    assert fixture["test_lock"]["test_metrics_computed"] is False


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
