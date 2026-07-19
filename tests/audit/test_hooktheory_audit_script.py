from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from scripts.audit_hooktheory_legacy import (
    SHEETSAGE_COMMIT,
    audit_raw,
    derive_v1_compatibility_pitch,
    exact_duplicate_regions,
    iter_jsonl,
    iter_top_level_object,
)


def raw_record(payload: dict) -> dict:
    return {"hash": "clip", "split": "train", "json": payload}


def test_incremental_reader_supports_complete_fragment_and_exact_decimal(tmp_path: Path) -> None:
    complete = tmp_path / "complete.json"
    complete.write_text('{"a":{"beat":4.50},"b":{"beat":1}}', encoding="utf-8")
    values = dict(iter_top_level_object(complete, parse_decimal=True))
    assert values["a"]["beat"] == Decimal("4.50")
    assert str(values["a"]["beat"]) == "4.50"

    fragment = tmp_path / "fragment.json"
    fragment.write_text('"a":{"beat":1.25},"b":{"beat":2}', encoding="utf-8")
    fragment_values = dict(iter_top_level_object(fragment, parse_decimal=True))
    assert fragment_values["a"]["beat"] == Decimal("1.25")
    assert set(fragment_values) == {"a", "b"}


def test_jsonl_reader_can_preserve_decimal_lexemes(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"duration":0.10}\n\n{"duration":2}\n', encoding="utf-8")
    rows = list(iter_jsonl(path, parse_decimal=True))
    assert rows[0] == (1, {"duration": Decimal("0.10")})
    assert rows[1] == (3, {"duration": 2})


def test_duplicate_regions_are_exact_and_bounded() -> None:
    count, examples = exact_duplicate_regions([
        {"beat": 1, "tonic": "C"},
        {"tonic": "C", "beat": 1},
        {"beat": 2, "tonic": "G"},
    ])
    assert count == 1
    assert examples == [{"value": {"beat": 1, "tonic": "C"}, "occurrences": 2}]


def test_derived_pitch_accounts_for_success_missing_range_and_key_resolution() -> None:
    keys = [{"beat": 1, "tonic": "C"}, {"beat": 9, "tonic": "G"}]
    assert derive_v1_compatibility_pitch(
        {"beat": 9, "sd": "1", "octave": 0, "isRest": False}, keys
    ) == ("success", 79)
    assert derive_v1_compatibility_pitch(
        {"beat": 1, "sd": "", "octave": None, "isRest": False}, keys
    ) == ("missing_inputs", None)
    assert derive_v1_compatibility_pitch(
        {"beat": 1, "sd": "#6", "octave": 4, "isRest": False},
        [{"beat": 1, "tonic": "B"}],
    ) == ("out_of_range", 141)
    assert derive_v1_compatibility_pitch(
        {"beat": None, "sd": "1", "octave": 0, "isRest": False}, keys
    ) == ("unresolved_active_key", None)


def test_synthetic_audit_checks_root8_meter_borrowed_and_decimal_timing(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    payload = {
        "keys": [{"beat": 1, "tonic": "C", "scale": "major"}] * 2,
        "tempos": [{"beat": 1, "bpm": 120}],
        "meters": [
            {"beat": 1, "numBeats": 12, "beatUnit": 3},
            {"beat": 13, "numBeats": 8, "beatUnit": 1},
        ],
        "notes": [{"beat": 4.50, "duration": 0.25, "sd": "bb1", "octave": 0, "isRest": False}],
        "chords": [
            {"beat": 1, "duration": 1, "root": 8, "borrowed": "[0,2,4,5,7,9,11]", "alternate": "_", "pedal": None},
            {"beat": 2, "duration": 1, "root": -2, "borrowed": {"bad": True}, "alternate": "", "pedal": 1},
        ],
    }
    path.write_text(json.dumps({"clip": raw_record(payload)}), encoding="utf-8")
    audit, _, candidates = audit_raw(path, candidate_limit=12)
    findings = audit["audited_findings"]
    assert findings["raw_root_8"]["corpus_status"] == "observed"
    assert findings["negative_root"]["count"] == 1
    assert findings["alternate_underscore"]["count"] == 1
    assert findings["non_null_pedal"]["count"] == 1
    assert findings["borrowed_stringified_list"]["count"] == 1
    assert findings["borrowed_unexpected_type"]["count"] == 1
    assert findings["beat_unit_3"]["count"] == 1
    assert findings["num_beats_8"]["count"] == 1
    assert audit["exact_duplicate_regions"]["keys"]["count"] == 1
    assert {item["lexeme"] for item in audit["exact_timing"]["lexemes"]} >= {"4.5", "0.25"}
    assert candidates["root_eight_bvii"] == ["clip"]


def test_upstream_pin_is_exact() -> None:
    assert SHEETSAGE_COMMIT == "bbdd7b7b6a5fb845828f82790acdceb03a197779"
