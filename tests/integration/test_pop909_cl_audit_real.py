from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.audit_pop909_cl import (
    EXPECTED_SONG_IDS,
    build_report,
    ensure_output_outside_root,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "pop909_cl" / "audit_manifest.json"
RUN_REAL = os.environ.get("MUSIC_CRITIC_RUN_REAL_POP909_CL_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "set MUSIC_CRITIC_RUN_REAL_POP909_CL_TESTS=1 with "
        "MUSIC_CRITIC_POP909_CL_ROOT and MUSIC_CRITIC_POP909_CL_UPSTREAM_ROOT"
    ),
)


def test_complete_recorded_pop909_cl_audit() -> None:
    supplied = os.environ.get("MUSIC_CRITIC_POP909_CL_ROOT")
    upstream_supplied = os.environ.get("MUSIC_CRITIC_POP909_CL_UPSTREAM_ROOT")
    assert supplied, "MUSIC_CRITIC_POP909_CL_ROOT must explicitly identify the installed corpus"
    assert upstream_supplied, "MUSIC_CRITIC_POP909_CL_UPSTREAM_ROOT must identify the pinned checkout"
    root = Path(supplied)
    upstream_root = Path(upstream_supplied)
    assert root.is_dir(), f"explicit POP909-CL root is missing: {root}"
    assert upstream_root.is_dir(), f"explicit pinned upstream root is missing: {upstream_root}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing_report = os.environ.get("MUSIC_CRITIC_POP909_CL_EXISTING_REPORT")
    if existing_report:
        report = json.loads(Path(existing_report).read_text(encoding="utf-8"))
    else:
        report = build_report(root, upstream_root=upstream_root)

    assert report["audit_schema_version"] == manifest["audit_schema_version"]
    identity = report["corpus_identity"]
    for key, expected in manifest["corpus"].items():
        assert identity[key] == expected, key
    assert tuple(row["logical_id"] for row in identity["files"]) == EXPECTED_SONG_IDS
    comparison = report["upstream_comparison"]
    assert comparison["observed_git"]["commit"] == manifest["upstream"]["commit"]
    for key in (
        "exact_content_matches",
        "content_mismatches",
        "local_only_song_ids",
        "upstream_only_song_ids",
        "license",
        "license_sha256",
    ):
        assert comparison[key] == manifest["upstream"][key], key
    assert comparison["provenance_confirmed"] is True
    assert report["instrument_contract"]["failure_counts"] == manifest[
        "instrument_failure_counts"
    ]
    assert report["instrument_contract"]["fatal_failure_counts"] == {}
    assert report["instrument_contract"][
        "expected_masked_target_unavailability_song_ids"
    ] == manifest["expected_masked_target_unavailability_song_ids"]

    crosswalk = report["score_only_crosswalk"]
    for key in ("attempted", "converted", "failed", "quarantined", "fatal_failed"):
        assert crosswalk[key] == manifest["score_crosswalk"][key], key
    assert [row["song_id"] for row in crosswalk["failures"]] == manifest[
        "score_crosswalk"
    ]["failed_song_ids"]
    assert crosswalk["failures_by_category"] == {
        "midi_adapter.meter_change_inside_bar": 1
    }
    assert crosswalk["quarantined_song_ids"] == ["172"]
    assert all(row["equal"] for row in crosswalk["serialization_round_trip_sample"])
    assert report["unsafe_complete_file_generic_diagnostics"]["production_safe"] is False
    assert "midi_parse_failure" not in report["instrument_contract"]["failure_counts"]
    for section, keys in (
        ("score_only_crosswalk", ("warnings_by_code", "files_affected_by_warning_code")),
        ("unsafe_complete_file_generic_diagnostics", ("warnings_by_code",)),
        (
            "chord_annotation_inventory",
            (
                "total_blocks",
                "normalization_status_counts",
                "implicit_n_gap_count",
                "trailing_unannotated_span_count",
                "task_mask_counts",
                "overlap_count",
                "duplicate_block_onset_count",
                "repeated_pitch_at_onset_block_count",
                "mixed_note_end_tick_block_count",
                "pairing_diagnostics",
                "pairing_anomaly_evidence_sha256",
            ),
        ),
    ):
        for key in keys:
            assert report[section][key] == manifest["aggregates"][section][key], (
                section,
                key,
            )

    inventory = report["chord_annotation_inventory"]
    assert inventory["raw_block_provenance"]["source"] == "human"
    assert inventory["raw_block_provenance"]["details"] == [
        "human_corrected",
        "expert_reviewed",
    ]
    assert inventory["normalized_target_provenance"]["source"] == "derived"
    assert inventory["implicit_n_provenance"]["source"] == "derived"
    assert len(inventory["pairing_anomaly_events"]) == 8
    for event in inventory["pairing_anomaly_events"]:
        assert {
            "category",
            "tick",
            "pitch",
            "velocity",
            "channel",
            "ordinal",
            "source_path",
            "source_sha256",
            "affected_block_onsets",
            "affected_span_ids",
            "affected_interval",
        } <= event.keys()
    assert report["strict"] == manifest["readiness"]
    assert report["strict"]["evidence_contract_ready"] is True
    assert report["strict"]["production_adapter_ready"] is False

    rows = {row["song_id"]: row for row in report["per_file"]}
    for case in manifest["cases"]:
        row = rows[case["song_id"]]
        assert row["relative_path"] == case["relative_path"]
        assert row["sha256"] == case["sha256"]
        assert [item["category"] for item in row["instrument_contract"]["failures"]] == case[
            "instrument_failure_categories"
        ]
        assert row["score_projection"]["status"] == case["score_projection_status"]
        for key, expected in case.get("expected_facts", {}).items():
            assert row[key] == expected, (case["song_id"], key)
        if case["song_id"] in {"367", "658"}:
            assert row["chord_annotations"]["status"] == "unavailable"
            assert all(
                available is False
                for available in row["chord_annotations"]["task_availability"].values()
            )
        if case["song_id"] == "172":
            assert row["score_projection"]["acceptance"] == "quarantined"
        if "meter_change" in case:
            observed = next(
                item for item in row["meter_boundary_evidence"] if item["tick"] == 85_080
            )
            for key, expected in case["meter_change"].items():
                assert observed[key] == expected, key

    report_output = os.environ.get("MUSIC_CRITIC_POP909_CL_REPORT")
    if report_output:
        output = Path(report_output)
        ensure_output_outside_root(root, output)
        assert not output.resolve().is_relative_to(REPO_ROOT.resolve())
        write_report(report, output)
