from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json
import os
from pathlib import Path

import pytest

from music_critic.adapters._json_stream import iter_jsonl, iter_object_records
from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.data import RationalTime, dumps_piece, loads_piece, validate_piece


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hooktheory"
STRUCTURE_ROOT = Path(
    os.environ.get(
        "MUSIC_CRITIC_HOOKTHEORY_ROOT",
        REPO_ROOT / "data/HookTheory",
    )
)
RAW_PATH = STRUCTURE_ROOT / "Hooktheory_Raw.json/4_merged.json"


pytestmark = pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_HOOKTHEORY_ADAPTER") != "1",
    reason="set MUSIC_CRITIC_RUN_HOOKTHEORY_ADAPTER=1 for real HookTheory adapter tests",
)


def exact(value: object, *, subtract_one: bool = False) -> RationalTime:
    assert isinstance(value, (int, Decimal)) and not isinstance(value, bool)
    result = Fraction(str(value)) - (1 if subtract_one else 0)
    return RationalTime(result.numerator, result.denominator)


def exact_text(value: str) -> RationalTime:
    result = Fraction(value)
    return RationalTime(result.numerator, result.denominator)


def load_cases() -> list[dict]:
    manifest = json.loads(
        (FIXTURE_ROOT / "golden_manifest.json").read_text(encoding="utf-8")
    )
    return [
        json.loads((FIXTURE_ROOT / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in manifest["cases"]
    ]


def select_raw(ids: set[str]) -> dict[str, dict]:
    selected = {}
    for clip_id, record in iter_object_records(RAW_PATH):
        if clip_id in ids:
            selected[clip_id] = record
    assert set(selected) == ids
    return selected


def select_structures(ids: set[str]) -> dict[str, dict]:
    selected = {}
    for split in ("train", "val", "test"):
        path = STRUCTURE_ROOT / f"HookTheoryStructure.{split}.jsonl"
        for _line, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            if isinstance(audio_path, str) and Path(audio_path).stem in ids:
                selected[Path(audio_path).stem] = row
    return selected


def target(piece, task: str):
    return next(value for value in piece.targets if value.task == task)


def test_all_phase_2b0_golden_cases_convert_against_raw_production_source() -> None:
    cases = load_cases()
    assert len(cases) == 19
    ids = {case["source_reference"]["clip_id"] for case in cases}
    raw = select_raw(ids)
    structures = select_structures(ids)
    converted = 0
    missing_payload = 0

    for case in cases:
        reference = case["source_reference"]
        clip_id = reference["clip_id"]
        record = raw[clip_id]
        if record.get("json") is None:
            missing_payload += 1
            with pytest.raises(HookTheoryAdapterError, match="no usable json payload"):
                convert_hooktheory_record(
                    clip_id,
                    record,
                    config=HookTheoryAdapterConfig("hooktheory"),
                    source_path=str(RAW_PATH),
                )
            continue

        piece = convert_hooktheory_record(
            clip_id,
            record,
            config=HookTheoryAdapterConfig("hooktheory"),
            structure_row=structures.get(clip_id),
            source_path=str(RAW_PATH),
        )
        converted += 1
        assert piece.piece_id == f"piece:hooktheory-{clip_id}"
        assert piece.split == reference["split"]
        assert piece.source_path == str(RAW_PATH)
        expected_group = reference["ori_uid"] or piece.piece_id
        assert piece.source_group_id == expected_group
        assert not validate_piece(piece).errors
        assert loads_piece(dumps_piece(piece)) == piece
        assert piece.key_signature_events == ()

        flag_codes = {flag.code for flag in piece.quality_flags}
        if clip_id in structures:
            assert "hooktheory.structure_alignment_unresolved" in flag_codes
        else:
            assert "hooktheory.structure_unmatched_symbolic_clip" in flag_codes

        payload = record["json"]
        for note_excerpt in case["raw_excerpt"].get("notes", []):
            source_index = note_excerpt["index"]
            raw_note = payload["notes"][source_index]
            expected = next(
                (
                    value
                    for value in case["expected_v2_contract"].get("melody", [])
                    if value["raw_sd"] == raw_note.get("sd")
                    and value["octave"] == raw_note.get("octave")
                    and value["is_rest"] is raw_note.get("isRest")
                ),
                None,
            )
            if expected is None or expected["derived_pitch"] is None:
                continue
            canonical = next(
                note
                for note in piece.notes
                if note.note_id == f"note:melody-{source_index:06d}"
            )
            expected_onset = expected.get("canonical_onset_qn")
            expected_duration = expected.get("canonical_duration_qn")
            assert canonical.onset_qn == (
                exact_text(expected_onset)
                if expected_onset is not None
                else exact(raw_note["beat"], subtract_one=True)
            )
            assert canonical.duration_qn == (
                exact_text(expected_duration)
                if expected_duration is not None
                else exact(raw_note["duration"])
            )
            assert canonical.pitch == expected["derived_pitch"]

        for expected_meter in case["expected_v2_contract"].get("meter", []):
            assert any(
                event.numerator == expected_meter["canonical_numerator"]
                and event.denominator == expected_meter["canonical_denominator"]
                for event in piece.meter_events
            )

        root_target = target(piece, "theory.chord.root_degree")
        roots_by_span = dict(zip(root_target.entity_ids, root_target.values, strict=True))
        for chord_excerpt in case["raw_excerpt"].get("chords", []):
            source_index = chord_excerpt["index"]
            expected = next(
                (
                    value
                    for value in case["expected_v2_contract"].get("chords", [])
                    if value["raw_root"] == payload["chords"][source_index]["root"]
                    and value["is_rest"] is payload["chords"][source_index]["isRest"]
                ),
                None,
            )
            if expected is None:
                continue
            degree = expected["canonical_functional_degree"]
            expected_value = None if degree is None else str(degree)
            assert roots_by_span[f"span:chord-{source_index:06d}"] == expected_value

        for aligned_target in piece.targets:
            for index, available in enumerate(aligned_target.mask):
                if available:
                    assert aligned_target.values[index] is not None
                    assert aligned_target.source[index] == "dataset"
                    assert aligned_target.provenance[index] == "prov:annotation"
                else:
                    assert (
                        aligned_target.values[index],
                        aligned_target.confidence[index],
                        aligned_target.source[index],
                        aligned_target.provenance[index],
                    ) == (None, None, None, None)

    assert converted == 18
    assert missing_payload == 1
