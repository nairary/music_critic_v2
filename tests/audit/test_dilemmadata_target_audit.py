from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_dilemmadata_targets import (
    DEFAULT_MANIFEST,
    build_target_audit_report,
    dumps_target_audit_report,
    manifest_projection,
    validate_manifest_self_fingerprint,
    validate_report_semantic_fingerprint,
)
from tests.adapters.test_dilemmadata import CORPUS, _fixture_identity


def test_bounded_target_audit_is_deterministic_and_path_free() -> None:
    identity = _fixture_identity(CORPUS)
    first = build_target_audit_report(
        CORPUS,
        run_e2e=False,
        identity=identity,
    )
    second = build_target_audit_report(
        CORPUS,
        run_e2e=False,
        identity=identity,
    )
    assert dumps_target_audit_report(first) == dumps_target_audit_report(second)
    assert validate_report_semantic_fingerprint(first)
    assert str(CORPUS.resolve()) not in dumps_target_audit_report(first)
    assert first["outcomes"]["raw"]["status_counts"] == {"accepted": 3}
    assert first["outcomes"]["target_sidecar"]["status_counts"] == {
        "accepted": 3
    }
    assert first["outcomes"]["target_sidecar"]["fatal_count"] == 0
    assert len(first["families"]) == 22
    assert {row["dialect"] for row in first["families"]} == {
        "an_joint",
        "dlc",
    }
    assert first["grouping_and_split"]["analysis_views_are_separate"] is True
    assert first["end_to_end"] == {
        "performed": False,
        "reason": "disabled for bounded fixture/unit audit",
    }


def test_family_projection_has_explicit_state_alignment_and_encoding_counts() -> None:
    report = build_target_audit_report(
        CORPUS,
        run_e2e=False,
        identity=_fixture_identity(CORPUS),
    )
    required = {
        "source_row_count",
        "available",
        "masked",
        "missing",
        "ambiguous",
        "unsupported",
        "unaligned",
        "conflict",
        "deferred",
        "emitted_rows",
        "model_ready_rows",
        "observed_value_count",
        "observed_distinct_value_count",
        "encoding_kind",
        "alignment_entity_type",
        "provenance_source",
    }
    for row in report["families"]:
        assert required <= set(row)
        assert (
            row["available"]
            + row["masked"]
            + row["missing"]
            + row["unsupported"]
            == row["source_row_count"]
        )
        assert len(row["observed_values_fingerprint"]) == 64
        if row["encoding_kind"] == "open_string_cpu":
            assert row["model_ready"] is False
            assert row["deferred"] == row["available"]
        else:
            assert row["model_ready"] is True
            assert row["deferred"] == 0


def test_manifest_projection_is_self_fingerprinted_and_mutation_sensitive() -> None:
    report = build_target_audit_report(
        CORPUS,
        run_e2e=False,
        identity=_fixture_identity(CORPUS),
    )
    manifest = manifest_projection(report)
    assert validate_manifest_self_fingerprint(manifest)
    mutated = json.loads(json.dumps(manifest))
    mutated["alignment_totals"]["unaligned"] += 1
    assert not validate_manifest_self_fingerprint(mutated)


def test_committed_target_manifest_is_compact_canonical_and_self_fingerprinted() -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert validate_manifest_self_fingerprint(manifest)
    assert len(manifest["families"]) == 22
    assert manifest["outcomes"]["raw"]["status_counts"] == {
        "accepted": 719,
        "quarantined": 914,
    }
    assert manifest["outcomes"]["target_sidecar"]["status_counts"] == {
        "accepted": 719
    }
    assert manifest["readiness"]["phase9b2a_contract_ready"] is True
    assert not any(
        Path(value).is_absolute()
        for value in _all_strings(manifest)
        if value.startswith("/")
    )


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
