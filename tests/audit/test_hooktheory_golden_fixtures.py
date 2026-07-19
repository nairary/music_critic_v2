from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hooktheory"
CASE_ROOT = FIXTURE_ROOT / "cases"
SUPPORTED_SCHEMA = "hooktheory_golden_v1"
VALID_TAGS = {
    "alternate_underscore",
    "alternate_unresolved",
    "applied_deferred",
    "beat_unit_3",
    "borrowed_empty",
    "borrowed_mode",
    "borrowed_pcset",
    "borrowed_unknown",
    "chord_decorations",
    "compound_meter_grouping",
    "derived_pitch",
    "extended_chord",
    "first_beat",
    "fractional_timing",
    "inversion",
    "melody_rest",
    "minor_modal",
    "missing_pitch_input",
    "missing_payload",
    "multiple_keys",
    "multiple_meters",
    "multiple_tempos",
    "negative_root",
    "null_note_beat",
    "null_note_octave",
    "num_beats_8",
    "ordinary_major",
    "root_zero_non_rest",
    "root_zero_rest",
    "root_anomaly",
    "seventh_chord",
    "shared_ori_uid",
    "structure_matched",
    "structure_unmatched_symbolic",
    "double_flat_bb1",
    "resolved_meter",
}
REQUIRED_OBSERVED_TAGS = VALID_TAGS - {"borrowed_empty"}
FORBIDDEN_RAW_FEATURES = {"sd_id", "root_id", "type_id", "inversion_id", "applied_id"}
SD_TO_CHROMATIC = {
    "1": 0, "b1": 11, "#1": 1,
    "2": 2, "b2": 1, "#2": 3,
    "3": 4, "b3": 3, "#3": 5,
    "4": 5, "b4": 4, "#4": 6,
    "5": 7, "b5": 6, "#5": 8,
    "6": 9, "b6": 8, "#6": 10,
    "7": 11, "b7": 10, "#7": 0,
    "bb1": 10,
}


def load_manifest() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "golden_manifest.json").read_text(encoding="utf-8"))


def load_cases() -> list[dict[str, Any]]:
    manifest = load_manifest()
    return [
        json.loads((CASE_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in manifest["cases"]
    ]


def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def normalized_fraction(value: str) -> bool:
    if not re.fullmatch(r"-?\d+/[1-9]\d*", value):
        return False
    fraction = Fraction(value)
    return value == f"{fraction.numerator}/{fraction.denominator}"


def test_manifest_and_case_file_bijection() -> None:
    manifest = load_manifest()
    assert manifest["fixture_schema_version"] == SUPPORTED_SCHEMA
    assert manifest["legacy_commit"] == "2d8281f31cc9ad9c8fecaf332da0c61e0e949415"
    assert len(manifest["cases"]) >= 19
    assert len(manifest["cases"]) == len(set(manifest["cases"]))
    actual = {path.stem for path in CASE_ROOT.glob("*.json")}
    assert set(manifest["cases"]) == actual


def test_case_shape_ids_tags_and_coverage() -> None:
    manifest = load_manifest()
    cases = load_cases()
    required = {
        "case_id", "coverage_tags", "source_reference", "raw_excerpt",
        "structure_excerpt", "legacy_selected_expected",
        "legacy_canonical_expected", "expected_v2_contract",
        "expected_diagnostics", "evidence",
    }
    assert [case["case_id"] for case in cases] == manifest["cases"]
    assert len({case["case_id"] for case in cases}) == len(cases)
    observed_tags = set()
    for case in cases:
        assert set(case) == required
        assert len(case["coverage_tags"]) == len(set(case["coverage_tags"]))
        assert set(case["coverage_tags"]) <= VALID_TAGS
        observed_tags.update(case["coverage_tags"])
        reference = case["source_reference"]
        assert set(reference) == {
            "dataset_relative_path", "split", "clip_id", "ori_uid",
            "source_record_locator",
        }
        assert reference["dataset_relative_path"] == (
            "data/HookTheory/Hooktheory_Raw.json/4_merged.json"
        )
    assert REQUIRED_OBSERVED_TAGS <= observed_tags
    assert "raw_root_8_bvii" in manifest["not_observed_categories"]
    assert "borrowed_stringified_pitch_class_list" in manifest["not_observed_categories"]
    classifications = manifest["evidence_classifications"]
    assert classifications["root_8_bvii"]["corpus_status"] == "not_observed"
    assert classifications["root_8_bvii"]["classification"] == "Music Critic V1 synthetic compatibility behavior"
    assert classifications["midi_anchor_60"]["classification"] == "upstream_semantics"
    assert classifications["midi_anchor_72"]["classification"] == "legacy_compatibility"
    assert manifest["upstream_sheetsage"]["commit"] == "bbdd7b7b6a5fb845828f82790acdceb03a197779"


def test_no_absolute_paths_or_encoded_ids_as_v2_raw_features() -> None:
    for path in [FIXTURE_ROOT / "golden_manifest.json", *CASE_ROOT.glob("*.json")]:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "\\Users\\" not in text
    for case in load_cases():
        contract_keys = {item for item in walk(case["expected_v2_contract"]) if isinstance(item, str)}
        assert not (FORBIDDEN_RAW_FEATURES & contract_keys)


def test_rational_strings_and_one_based_conversion_are_exact() -> None:
    saw_first = False
    saw_fractional = False
    for case in load_cases():
        contract = case["expected_v2_contract"]
        for timing in contract["timing"]:
            onset = timing["canonical_onset_qn"]
            duration = timing["canonical_duration_qn"]
            assert normalized_fraction(onset)
            assert normalized_fraction(duration)
            assert Fraction(onset) == Fraction(timing["raw_beat_decimal"]) - 1
            assert Fraction(duration) == Fraction(timing["raw_duration_decimal"])
            saw_first |= timing["raw_beat_decimal"] == "1" and onset == "0/1"
            saw_fractional |= "." in timing["raw_beat_decimal"]
        for regions in contract.get("regions", {}).values():
            if isinstance(regions, list) and all(isinstance(item, str) and "/" in item for item in regions):
                assert all(normalized_fraction(item) for item in regions)
    assert saw_first and saw_fractional


def test_derived_pitch_formula_rest_and_range_guard_contract() -> None:
    saw_rest = False
    for case in load_cases():
        for melody in case["expected_v2_contract"]["melody"]:
            assert melody["raw_sd"] in SD_TO_CHROMATIC
            assert isinstance(melody["active_scale_degree_offset"], int)
            assert isinstance(melody["accidental_offset"], int)
            assert melody["pitch_source"] == "derived"
            assert melody["provenance_method"] == "hooktheory_scale_degree_to_midi_upstream"
            expected = (
                60
                + 12 * melody["octave"]
                + melody["tonic_pc"]
                + melody["active_scale_degree_offset"]
                + melody["accidental_offset"]
            )
            if melody["is_rest"]:
                saw_rest = True
                assert melody["derived_pitch"] is None
            else:
                assert 0 <= expected <= 127
                assert melody["derived_pitch"] == expected
    assert saw_rest
    out_of_range = 60 + 12 * 4 + 11 + 10
    assert out_of_range > 127
    assert "derived_pitch_out_of_range" in load_manifest()["not_observed_categories"]


def test_root_mapping_zero_eight_and_applied_policy() -> None:
    mapping = load_manifest()["root_mapping_contract"]
    assert mapping == {
        "0": None, "1": 0, "2": 1, "3": 2, "4": 3,
        "5": 4, "6": 5, "7": 6, "8": "bVII",
    }
    saw_zero = False
    saw_negative = False
    for case in load_cases():
        for chord in case["expected_v2_contract"]["chords"]:
            raw = chord["raw_root"]
            assert chord["applied_target"] is None
            if raw == 0:
                saw_zero = True
                assert chord["canonical_functional_degree"] is None
            elif 1 <= raw <= 7:
                assert chord["canonical_functional_degree"] == raw - 1
            elif raw < 0:
                saw_negative = True
                assert chord["canonical_functional_degree"] is None
                assert chord["is_rest"] is True
    assert saw_zero
    assert saw_negative
    assert mapping["8"] == "bVII"
    assert all(
        event["value"].get("root") != 8
        for case in load_cases()
        for event in case["raw_excerpt"].get("chords", [])
    )


def test_observed_meter_and_anomaly_cases_are_real_and_mask_safe() -> None:
    by_tag = {
        tag: case
        for case in load_cases()
        for tag in case["coverage_tags"]
    }
    compound = by_tag["beat_unit_3"]["expected_v2_contract"]["meter"][0]
    assert compound == {
        "raw_num_beats": 12,
        "raw_beat_unit": 3,
        "felt_group_size_source_beats": 3,
        "canonical_numerator": 12,
        "canonical_denominator": 8,
        "status": "resolved_by_semantic_crosswalk",
    }
    simple = by_tag["num_beats_8"]["expected_v2_contract"]["meter"][0]
    assert simple == {
        "raw_num_beats": 8,
        "raw_beat_unit": 1,
        "felt_group_size_source_beats": 1,
        "canonical_numerator": 8,
        "canonical_denominator": 4,
        "status": "resolved_by_semantic_crosswalk",
    }
    assert by_tag["alternate_underscore"]["raw_excerpt"]["chords"][0]["value"]["alternate"] == "_"
    null_case = by_tag["null_note_beat"]
    assert null_case["raw_excerpt"]["notes"][0]["value"]["beat"] is None
    assert null_case["raw_excerpt"]["notes"][1]["value"]["octave"] is None
    assert null_case["expected_v2_contract"]["melody"] == []
    bb1 = by_tag["double_flat_bb1"]
    assert bb1["raw_excerpt"]["notes"][0]["value"]["sd"] == "bb1"
    assert bb1["expected_v2_contract"]["melody"][0]["provenance_method"] == "hooktheory_scale_degree_to_midi_upstream"


def test_structure_seconds_grouping_masks_and_diagnostics() -> None:
    group_to_source_ids: dict[str, set[str]] = {}
    for case in load_cases():
        reference = case["source_reference"]
        contract = case["expected_v2_contract"]
        assert contract["source_group_id"] == reference["ori_uid"]
        if reference["ori_uid"] is not None:
            group_to_source_ids.setdefault(reference["ori_uid"], set()).add(contract["source_group_id"])
        structure = contract["structure"]
        assert structure == {
            "coordinate_unit": "audio_seconds",
            "section_alignment_status": "unresolved_audio_seconds",
            "target_array_expected": False,
            "annotation_span_expected": False,
        }
        unavailable = contract["unavailable_target"]
        assert unavailable == {
            "mask": False, "value": None, "source": None,
            "confidence": None, "provenance": None,
        }
        diagnostics = case["expected_diagnostics"]
        assert len(diagnostics) == len(set(diagnostics))
        assert all(re.fullmatch(r"hooktheory\.[a-z0-9_]+", item) for item in diagnostics)
    assert all(len(source_ids) == 1 for source_ids in group_to_source_ids.values())
    shared = [case for case in load_cases() if "shared_ori_uid" in case["coverage_tags"]]
    assert len(shared) >= 2
    assert len({case["source_reference"]["ori_uid"] for case in shared}) == 1


def test_fixture_json_has_deterministic_canonical_serialization() -> None:
    for path in [FIXTURE_ROOT / "golden_manifest.json", *sorted(CASE_ROOT.glob("*.json"))]:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        first = json.dumps(loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        second = json.dumps(json.loads(first), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert first == second
