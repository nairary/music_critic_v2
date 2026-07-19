from __future__ import annotations

import ast
from dataclasses import fields
from decimal import Decimal
import json
from pathlib import Path

import pytest

import music_critic.adapters.hooktheory as hooktheory_module
from music_critic.adapters import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
    load_hooktheory_piece,
)
from music_critic.data import (
    CanonicalPiece,
    RationalTime,
    ValidationIssue,
    ValidationReport,
    dumps_piece,
    loads_piece,
    validate_piece,
)


CONFIG = HookTheoryAdapterConfig(dataset_name="hooktheory-test")


def raw_record(
    *,
    notes: list[dict] | None = None,
    chords: list[dict] | None = None,
    keys: list[dict] | None = None,
    tempos: list[dict] | None = None,
    meters: list[dict] | None = None,
    end_beat: int | Decimal = 9,
    split: object = "train",
    clip_id: str = "clip",
) -> dict:
    return {
        "hash": clip_id,
        "split": split,
        "json": {
            "endBeat": end_beat,
            "keys": keys if keys is not None else [
                {"beat": 1, "tonic": "C", "scale": "major"}
            ],
            "tempos": tempos if tempos is not None else [{"beat": 1, "bpm": 120}],
            "meters": meters if meters is not None else [
                {"beat": 1, "numBeats": 4, "beatUnit": 1}
            ],
            "notes": notes if notes is not None else [
                {
                    "beat": 1,
                    "duration": 1,
                    "sd": "1",
                    "octave": 0,
                    "isRest": False,
                }
            ],
            "chords": chords if chords is not None else [],
        },
    }


def chord(**overrides: object) -> dict:
    value = {
        "beat": 1,
        "duration": 2,
        "root": 1,
        "type": 5,
        "inversion": 0,
        "adds": [],
        "omits": [],
        "alterations": [],
        "suspensions": [],
        "borrowed": None,
        "isRest": False,
        "applied": 0,
        "alternate": "",
        "pedal": None,
    }
    value.update(overrides)
    return value


def convert(record: dict, *, include_targets: bool = True, structure: dict | None = None):
    return convert_hooktheory_record(
        record["hash"],
        record,
        config=HookTheoryAdapterConfig("hooktheory-test", include_targets),
        structure_row=structure,
        source_path="4_merged.json",
    )


def target(piece: CanonicalPiece, task: str):
    return next(value for value in piece.targets if value.task == task)


def codes(piece: CanonicalPiece) -> set[str]:
    return {flag.code for flag in piece.quality_flags}


def test_exact_timing_first_beat_active_keys_rests_bb1_and_round_trip() -> None:
    record = raw_record(
        keys=[
            {"beat": 1, "tonic": "D", "scale": "major"},
            {"beat": 3, "tonic": "Eb", "scale": "dorian"},
        ],
        notes=[
            {"beat": 1, "duration": 1, "sd": "1", "octave": 0, "isRest": True},
            {
                "beat": Decimal("1.25"),
                "duration": Decimal("0.50"),
                "sd": "bb1",
                "octave": 0,
                "isRest": False,
            },
            {"beat": 3, "duration": 1, "sd": "1", "octave": 0, "isRest": False},
        ],
    )
    piece = convert(record)
    assert [(note.onset_qn, note.duration_qn, note.pitch) for note in piece.notes] == [
        (RationalTime(1, 4), RationalTime(1, 2), 84),
        (RationalTime(2), RationalTime(1), 75),
    ]
    assert target(piece, "theory.melody.scale_degree").values == ("bb1", "1")
    assert target(piece, "theory.local_key.mode").values == ("major", "dorian")
    assert "hooktheory.double_flat_v1_compatibility" in codes(piece)
    assert not validate_piece(piece).errors
    assert loads_piece(dumps_piece(piece)) == piece


def test_invalid_notes_are_omitted_with_required_diagnostics() -> None:
    record = raw_record(
        keys=[],
        notes=[
            {"beat": None, "duration": 1, "sd": "1", "octave": 0, "isRest": False},
            {"beat": 1, "duration": 1, "sd": "1", "octave": None, "isRest": False},
            {"beat": 1, "duration": 1, "sd": "x", "octave": 0, "isRest": False},
            {"beat": 1, "duration": 0, "sd": "1", "octave": 0, "isRest": False},
            {"beat": 1.5, "duration": 1, "sd": "1", "octave": 0, "isRest": False},
            {"beat": 2, "duration": 1, "sd": "1", "octave": 0, "isRest": False},
        ],
    )
    piece = convert(record)
    assert piece.notes == ()
    assert {
        "hooktheory.note_timing_invalid",
        "hooktheory.note_octave_missing",
        "hooktheory.scale_degree_unsupported",
        "hooktheory.note_duration_invalid",
        "hooktheory.pitch_active_key_unresolved",
    } <= codes(piece)


def test_out_of_range_pitch_is_not_clamped() -> None:
    piece = convert(
        raw_record(
            keys=[{"beat": 1, "tonic": "B", "scale": "minor"}],
            notes=[
                {"beat": 1, "duration": 1, "sd": "#6", "octave": 4, "isRest": False}
            ],
        )
    )
    assert piece.notes == ()
    assert "hooktheory.pitch_out_of_range" in codes(piece)


def test_tempo_regions_exact_half_up_dedup_conflict_and_default() -> None:
    piece = convert(
        raw_record(
            tempos=[
                {"beat": 1, "bpm": 512},
                {"beat": 2, "bpm": 512},
                {"beat": 3, "bpm": 120},
                {"beat": 3, "bpm": 90},
            ]
        )
    )
    assert [event.microseconds_per_quarter for event in piece.tempo_events] == [
        117_188,
        500_000,
    ]
    assert "hooktheory.tempo_conflict" in codes(piece)

    defaulted = convert(raw_record(tempos=[{"beat": 2, "bpm": 60}]))
    assert defaulted.tempo_events[0].microseconds_per_quarter == 500_000
    assert "hooktheory.default_tempo" in codes(defaulted)


def test_meter_mapping_changes_and_incomplete_bar_are_exact() -> None:
    piece = convert(
        raw_record(
            end_beat=10,
            meters=[
                {"beat": 1, "numBeats": 4, "beatUnit": 1},
                {"beat": 3, "numBeats": 12, "beatUnit": 3},
                {"beat": 7, "numBeats": 8, "beatUnit": 1},
            ],
        )
    )
    assert [(event.onset_qn, event.numerator, event.denominator) for event in piece.meter_events] == [
        (RationalTime(0), 4, 4),
        (RationalTime(2), 12, 8),
        (RationalTime(6), 8, 4),
    ]
    assert piece.bars[0].duration_qn == RationalTime(2)
    assert piece.bars[0].is_incomplete is True
    assert "hooktheory.meter_change_incomplete_bar" in codes(piece)
    assert not validate_piece(piece).errors


def test_unsupported_meter_is_omitted_and_defaulted() -> None:
    piece = convert(
        raw_record(meters=[{"beat": 1, "numBeats": 7, "beatUnit": 2}])
    )
    assert [(event.numerator, event.denominator) for event in piece.meter_events] == [(4, 4)]
    assert {"hooktheory.meter_invalid", "hooktheory.default_meter"} <= codes(piece)


def test_chord_targets_masks_decorations_borrowed_and_synthetic_root_8() -> None:
    piece = convert(
        raw_record(
            chords=[
                chord(root=0, isRest=True),
                chord(beat=3, root=0),
                chord(beat=5, root=-2),
                chord(
                    beat=7,
                    root=8,
                    type=13,
                    inversion=3,
                    adds=[9, 4],
                    omits=[5, 3],
                    alterations=["b13", "#5"],
                    suspensions=[4, 2],
                    borrowed=[14, 0, 12, -1],
                ),
            ],
            end_beat=11,
        )
    )
    assert target(piece, "theory.chord.presence").values == (
        "false", "true", "true", "true"
    )
    roots = target(piece, "theory.chord.root_degree")
    assert roots.values == (None, None, None, "bVII")
    assert roots.mask == (False, False, False, True)
    assert target(piece, "theory.chord.extent").values[-1] == "13"
    assert target(piece, "theory.chord.inversion").values[-1] == "3"
    assert target(piece, "theory.chord.adds").values[-1] == ("4", "9")
    assert target(piece, "theory.chord.omits").values[-1] == ("3", "5")
    assert target(piece, "theory.chord.alterations").values[-1] == ("#5", "b13")
    assert target(piece, "theory.chord.suspensions").values[-1] == ("2", "4")
    assert target(piece, "theory.chord.borrowed").values[-1] == "pcset:0,2,11"
    assert {"hooktheory.chord_root_zero_non_rest", "hooktheory.chord_root_invalid"} <= codes(piece)


@pytest.mark.parametrize(
    ("borrowed", "expected"),
    [
        ("dorian", "mode:dorian"),
        ("super:2", "unknown:super:2"),
        (None, "none"),
    ],
)
def test_borrowed_string_encodings(borrowed: object, expected: str) -> None:
    piece = convert(raw_record(chords=[chord(borrowed=borrowed)]))
    assert target(piece, "theory.chord.borrowed").values == (expected,)


def test_unexpected_borrowed_and_deferred_fields_are_diagnostic() -> None:
    piece = convert(
        raw_record(
            chords=[
                chord(borrowed={"bad": True}, applied=5, alternate="_", pedal=1)
            ]
        )
    )
    borrowed = target(piece, "theory.chord.borrowed")
    assert borrowed.values == (None,)
    assert borrowed.mask == (False,)
    assert {
        "hooktheory.borrowed_invalid",
        "hooktheory.applied_deferred",
        "hooktheory.alternate_unresolved",
        "hooktheory.pedal_unresolved",
    } <= codes(piece)


def test_structure_grouping_missing_structure_and_missing_ori_uid() -> None:
    matched = convert(
        raw_record(),
        structure={"ori_uid": "song-1", "segment_start": Decimal("1.2")},
    )
    assert matched.source_group_id == "song-1"
    assert "hooktheory.structure_alignment_unresolved" in codes(matched)
    assert matched.annotations
    assert all(span.annotation_type != "section" for span in matched.annotations)

    missing_uid = convert(raw_record(), structure={"ori_uid": ""})
    assert missing_uid.source_group_id == missing_uid.piece_id
    assert "hooktheory.structure_missing_ori_uid" in codes(missing_uid)

    unmatched = convert(raw_record())
    assert unmatched.source_group_id == unmatched.piece_id
    assert "hooktheory.structure_unmatched_symbolic_clip" in codes(unmatched)


def test_split_hash_source_and_dataset_identity_validation() -> None:
    record = raw_record(split="VALID")
    piece = convert(record)
    assert piece.split == "val"
    assert piece.dataset_name == "hooktheory-test"
    assert piece.metadata.source_format == "json"
    assert piece.source_resolution is None

    unknown = convert(raw_record(split="other"))
    assert unknown.split is None
    assert "hooktheory.split_unknown" in codes(unknown)

    with pytest.raises(HookTheoryAdapterError, match="does not match"):
        convert_hooktheory_record("other", record, config=CONFIG)
    with pytest.raises(HookTheoryAdapterError, match="raw 4_merged"):
        convert_hooktheory_record(
            "clip", record, config=CONFIG, source_path="data/HookTheory/Hooktheory.json"
        )


def test_target_inventory_masks_and_complete_hiding() -> None:
    visible = convert(raw_record(chords=[chord(root=0)]), include_targets=True)
    hidden = convert(raw_record(chords=[chord(root=0)]), include_targets=False)
    assert [value.task for value in visible.targets] == [
        "theory.chord.adds",
        "theory.chord.alterations",
        "theory.chord.borrowed",
        "theory.chord.extent",
        "theory.chord.inversion",
        "theory.chord.omits",
        "theory.chord.presence",
        "theory.chord.root_degree",
        "theory.chord.suspensions",
        "theory.local_key.mode",
        "theory.local_key.tonic_pc",
        "theory.melody.scale_degree",
    ]
    root = target(visible, "theory.chord.root_degree")
    assert (root.values, root.mask, root.confidence, root.source, root.provenance) == (
        (None,), (False,), (None,), (None,), (None,)
    )
    assert hidden.annotations == ()
    assert hidden.targets == ()
    raw_fields = {
        field.name
        for field in fields(CanonicalPiece)
        if field.name not in {"annotations", "targets", "provenance"}
    }
    assert {name: getattr(visible, name) for name in raw_fields} == {
        name: getattr(hidden, name) for name in raw_fields
    }
    assert {record.provenance_id for record in visible.provenance} - {
        record.provenance_id for record in hidden.provenance
    } == {"prov:annotation"}
    assert not validate_piece(hidden).errors


def test_duration_is_derived_or_extended_from_valid_content() -> None:
    missing = raw_record(notes=[{"beat": 5, "duration": 2, "sd": "1", "octave": 0, "isRest": False}])
    missing["json"].pop("endBeat")
    piece = convert(missing)
    assert piece.duration_qn == RationalTime(6)
    assert "hooktheory.duration_derived" in codes(piece)

    short = convert(raw_record(end_beat=2, chords=[chord(beat=4, duration=3)]))
    assert short.duration_qn == RationalTime(6)
    assert "hooktheory.duration_extended" in codes(short)


def test_loader_uses_production_streamer_and_optional_structure_index(tmp_path: Path) -> None:
    raw_path = tmp_path / "4_merged.json"
    raw_path.write_text('"clip":' + json.dumps(raw_record()), encoding="utf-8")
    structure_root = tmp_path / "structure"
    structure_root.mkdir()
    (structure_root / "HookTheoryStructure.train.jsonl").write_text(
        json.dumps({"audio_path": "audio/clip.mp3", "ori_uid": "group"}) + "\n",
        encoding="utf-8",
    )
    piece = load_hooktheory_piece(
        raw_path, "clip", config=CONFIG, structure_root=structure_root
    )
    assert piece.source_group_id == "group"
    assert piece.source_path == str(raw_path)


def test_validation_failures_are_wrapped_with_complete_report(monkeypatch) -> None:
    report = ValidationReport(
        issues=(
            ValidationIssue(
                code="FIELD_VALUE_INVALID",
                severity="error",
                message="synthetic",
                path="/dataset_name",
                entity_id="piece:hooktheory-clip",
            ),
        )
    )
    monkeypatch.setattr(hooktheory_module, "validate_piece", lambda _piece: report)
    with pytest.raises(HookTheoryAdapterError) as caught:
        convert(raw_record())
    assert caught.value.clip_id == "clip"
    assert caught.value.validation_report is report


def test_hooktheory_production_imports_are_isolated() -> None:
    source_path = Path(hooktheory_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_import_prefixes = (
        "scripts.audit_hooktheory_legacy",
        "sheetsage",
        "numpy",
        "torch",
        "torch_geometric",
    )
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imports
        for prefix in forbidden_import_prefixes
    )
    assert "data/HTCanon" not in source
    assert "Fine-tune-text2midi" not in source
