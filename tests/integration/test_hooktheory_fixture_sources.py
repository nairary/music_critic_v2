from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.audit_hooktheory_legacy import (
    build_report,
    iter_jsonl,
    iter_top_level_object,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hooktheory"
CASE_ROOT = FIXTURE_ROOT / "cases"
RUN_ENV = "MUSIC_CRITIC_RUN_HOOKTHEORY_AUDIT"


pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_ENV) != "1",
    reason=f"set {RUN_ENV}=1 to verify HookTheory golden fixtures against local data",
)


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((FIXTURE_ROOT / "golden_manifest.json").read_text(encoding="utf-8"))
    cases = [
        json.loads((CASE_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in manifest["cases"]
    ]
    return manifest, cases


def record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_mapping_subset(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key, value in expected.items():
        assert key in actual
        if isinstance(value, dict):
            assert isinstance(actual[key], dict)
            assert_mapping_subset(value, actual[key])
        else:
            assert actual[key] == value


def select_records(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    selected = {}
    for key, record in iter_top_level_object(path):
        if key in wanted:
            assert isinstance(record, dict)
            selected[key] = record
    assert set(selected) == wanted
    return selected


def verify_legacy_excerpt(expected: dict[str, Any], actual: dict[str, Any] | None) -> None:
    if not expected["exists"]:
        assert actual is None
        return
    assert actual is not None
    assert_mapping_subset(expected["meta_excerpt"], actual["meta"])
    for event in expected.get("melody_events", []):
        assert_mapping_subset(event["value"], actual["melody"][event["index"]])
    for event in expected.get("chord_events", []):
        assert_mapping_subset(event["value"], actual["chords"][event["index"]])


def test_committed_sources_and_excerpts_match_local_artifacts() -> None:
    manifest, cases = load_inputs()
    for relative, expected_hash in manifest["source_file_hashes"].items():
        path = REPO_ROOT / relative
        assert path.is_file(), f"required local audit source is missing: {relative}"
        assert sha256_file(path) == expected_hash

    wanted = {case["source_reference"]["clip_id"] for case in cases}
    raw_path = REPO_ROOT / "data/HookTheory/Hooktheory_Raw.json/4_merged.json"
    raw_records = select_records(raw_path, wanted)
    for case in cases:
        reference = case["source_reference"]
        clip_id = reference["clip_id"]
        record = raw_records[clip_id]
        assert record.get("split", "").strip().lower().replace("valid", "val") == reference["split"]
        assert record_hash(record) == case["evidence"]["source_record_sha256"]
        excerpt = case["raw_excerpt"]
        payload = record.get("json")
        if excerpt.get("json_present") is False:
            assert payload is None
            continue
        assert isinstance(payload, dict)
        for region_name, regions in excerpt["regions"].items():
            assert payload[region_name] == regions
        for source_name, payload_name in (("notes", "notes"), ("chords", "chords")):
            for event in excerpt[source_name]:
                assert payload[payload_name][event["index"]] == event["value"]

    structure_by_clip: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = REPO_ROOT / f"data/HookTheory/HookTheoryStructure.{split}.jsonl"
        for _, row in iter_jsonl(path):
            clip_id = Path(row.get("audio_path", "")).stem
            if clip_id in wanted:
                structure_by_clip[clip_id] = {"split": split, **row}
    for case in cases:
        clip_id = case["source_reference"]["clip_id"]
        expected = case["structure_excerpt"]
        if expected is None:
            assert clip_id not in structure_by_clip
            continue
        row = structure_by_clip[clip_id]
        assert row["split"] == expected["split"]
        assert Path(row["audio_path"]).stem == expected["clip_id"]
        assert row["ori_uid"] == expected["ori_uid"]
        assert row["label"] == expected["labels"]
        for name in ("duration", "segment_start", "segment_end"):
            assert row[name] == expected[name]

    processed_path = REPO_ROOT / "data/HTCanon/HK_processed/hooktheory_processed.json"
    processed_wanted = {
        case["source_reference"]["clip_id"]
        for case in cases if case["legacy_selected_expected"]["exists"]
    }
    processed = select_records(processed_path, processed_wanted)
    for case in cases:
        clip_id = case["source_reference"]["clip_id"]
        verify_legacy_excerpt(case["legacy_selected_expected"], processed.get(clip_id))

    canonical_path = REPO_ROOT / "data/HTCanon/HK_processed/canonical_full/hooktheory_canonical.json"
    canonical_wanted = {
        case["source_reference"]["clip_id"]
        for case in cases if case["legacy_canonical_expected"]["exists"]
    }
    canonical = select_records(canonical_path, canonical_wanted)
    for case in cases:
        clip_id = case["source_reference"]["clip_id"]
        verify_legacy_excerpt(case["legacy_canonical_expected"], canonical.get(clip_id))

    simplified_cases = [
        case for case in cases if "upstream_simplified_excerpt" in case["evidence"]
    ]
    simplified_wanted = {
        case["source_reference"]["clip_id"] for case in simplified_cases
    }
    simplified = select_records(
        REPO_ROOT / "data/HookTheory/Hooktheory.json", simplified_wanted
    )
    for case in simplified_cases:
        clip_id = case["source_reference"]["clip_id"]
        actual = simplified[clip_id]
        expected = case["evidence"]["upstream_simplified_excerpt"]
        assert actual["split"] == expected["split"]
        assert actual["hooktheory"]["id"] == expected["hooktheory_id"] == clip_id
        assert isinstance(actual.get("alignment"), dict) is expected["alignment_available"]
        assert_mapping_subset(expected["annotations"], actual["annotations"])


def test_build_report_corpus_counts_and_semantic_meter_crosswalk() -> None:
    report = build_report(Namespace(
        hooktheory_root=REPO_ROOT / "data/HookTheory",
        htcanon_root=REPO_ROOT / "data/HTCanon",
        legacy_root=Path(os.environ.get(
            "MUSIC_CRITIC_LEGACY_ROOT",
            "/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic",
        )),
        candidate_limit=12,
    ))
    assert report["report_schema_version"] == "hooktheory_legacy_audit_v3"
    raw = report["raw_audit"]
    assert raw["record_counts_by_split"] == {
        "test": 2761, "train": 21233, "val": 2184
    }
    assert raw["records_missing_json"] == 3
    assert raw["event_counts"]["meters"] == 27217
    assert raw["derived_pitch_v1_compatibility"]["counts"] == {
        "success": 1228046,
        "missing_inputs": 8,
        "out_of_range": 0,
        "unresolved_active_key": 9,
        "rest": 110283,
    }
    assert {
        name: finding["count"]
        for name, finding in raw["exact_duplicate_regions"].items()
    } == {"keys": 0, "tempos": 0, "meters": 0}

    crosswalk = report["simplified_schema_crosswalk"]
    assert crosswalk["matched_identifiers"] == 26175
    assert crosswalk["raw_only_identifiers"] == 3
    assert crosswalk["simplified_only_identifiers"] == 0
    meter = crosswalk["meter_semantic_comparison"]
    assert meter["records_compared"] == 26175
    assert meter["raw_meter_regions"] == 27217
    assert meter["simplified_meter_regions"] == 27216
    assert meter["total_compared_meter_regions"] == 27216
    assert meter["exact_matches"] == 27216
    assert meter["missing_raw_regions"] == 0
    assert meter["missing_simplified_regions"] == 1
    assert meter["count_mismatches"] == 1
    assert meter["value_mismatches"] == 0
    assert meter["records_missing_raw_summary"] == 0
    assert meter["records_missing_raw_meter_collection"] == 0
    assert meter["records_missing_simplified_meter_collection"] == 0
    assert meter["canonical_mapping"] == {
        "accepted": True,
        "numerator": "raw numBeats",
        "denominator": {"beatUnit=1": 4, "beatUnit=3": 8},
        "basis": (
            "all paired regions match exactly and there are no simplified-only "
            "regions; raw-only regions are reported as simplified coverage loss"
        ),
    }
    assert meter["bounded_mismatch_examples"] == [
        {
            "kind": "count_mismatch",
            "clip_id": "nvgy-WaRgkA",
            "raw_count": 2,
            "simplified_count": 1,
        },
        {
            "kind": "missing_simplified_region",
            "clip_id": "nvgy-WaRgkA",
            "index": 1,
            "raw": {"beat": 25, "beatUnit": 1, "numBeats": 4},
            "expected": {"beat": "24/1", "beat_unit": 4, "beats_per_bar": 4},
        },
    ]

    inventory = {item["path"]: item["role"] for item in report["source_inventory"]}
    assert inventory["data/HookTheory/Hooktheory_Raw.json/4_merged.json"] == (
        "map_raw_theorytab_source"
    )
    assert inventory["data/HookTheory/Hooktheory.json"] == (
        "upstream_sheetsage_simplified"
    )
