from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import inf
from typing import get_args

import pytest

import music_critic.data.validation as validation
from music_critic.data import (
    AnnotationSpan,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    CanonicalValidationError,
    KeySignatureEvent,
    MeterEvent,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
    TargetArray,
    TempoEvent,
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    validate_or_raise,
    validate_piece,
)


SERIALIZATION_ONLY_CODES = {
    "JSON_UNKNOWN_FIELD",
    "JSON_MISSING_FIELD",
    "JSON_TYPE_INVALID",
    "RATIONAL_INVALID",
    "RATIONAL_NOT_NORMALIZED",
}

VALIDATOR_TESTED_CODES = {
    "SCHEMA_VERSION_UNSUPPORTED",
    "FIELD_VALUE_INVALID",
    "ENTITY_ID_INVALID",
    "ENTITY_ID_PREFIX_INVALID",
    "ENTITY_ID_DUPLICATE",
    "ENTITY_REFERENCE_INVALID",
    "COLLECTION_ORDER_INVALID",
    "VALUE_NOT_FINITE",
    "TIME_NEGATIVE",
    "DURATION_NEGATIVE",
    "ZERO_DURATION_NON_GRACE",
    "PITCH_OUT_OF_RANGE",
    "VELOCITY_OUT_OF_RANGE",
    "CHANNEL_OUT_OF_RANGE",
    "PROGRAM_OUT_OF_RANGE",
    "SOURCE_INDEX_INVALID",
    "PERCUSSION_MISMATCH",
    "PIECE_DURATION_TOO_SHORT",
    "TEMPO_INVALID",
    "TEMPO_INITIAL_MISSING",
    "TEMPO_DUPLICATE_ONSET",
    "METER_INVALID",
    "METER_INITIAL_MISSING",
    "METER_DUPLICATE_ONSET",
    "METER_NOT_AT_BAR_START",
    "BAR_INVALID",
    "BAR_COVERAGE_INVALID",
    "BAR_METER_MISMATCH",
    "BEAT_INVALID",
    "BEAT_GRID_INVALID",
    "ANNOTATION_INVALID",
    "TARGET_VIEW_INVALID",
    "TARGET_VIEW_DUPLICATE",
    "TARGET_LENGTH_MISMATCH",
    "TARGET_ENTITY_DUPLICATE",
    "TARGET_ALIGNMENT_INVALID",
    "TARGET_ENTITY_INVALID",
    "TARGET_VALUE_INVALID",
    "TARGET_MASK_INVALID",
    "TARGET_CONFIDENCE_INVALID",
    "TARGET_SOURCE_INVALID",
    "TARGET_PROVENANCE_INVALID",
    "QUALITY_FLAG_CODE_INVALID",
    "PROVENANCE_DETAIL_INVALID",
    "PROVENANCE_MISSING",
    "PROVENANCE_PARENT_INVALID",
    "PROVENANCE_CYCLE",
    "EMPTY_PIECE",
    "EMPTY_TRACK",
    "SOURCE_RESOLUTION_UNAVAILABLE",
    "INCOMPLETE_FINAL_BAR",
    "OVERLAPPING_SAME_PITCH_NOTES",
    "MID_BAR_TEMPO_CHANGE",
    "LOW_CONFIDENCE_TARGET",
    "UNREFERENCED_PROVENANCE",
    "EMPTY_OBSERVATION",
    "PIECE_TRAILING_SILENCE",
}


def _codes(piece: CanonicalPiece) -> set[str]:
    return {issue.code for issue in validate_piece(piece).issues}


def _assert_code(piece: CanonicalPiece, code: str) -> ValidationReport:
    report = validate_piece(piece)
    assert code in {issue.code for issue in report.issues}, report.issues
    return report


def _replace_note(
    piece: CanonicalPiece, index: int = 0, **changes: object
) -> CanonicalPiece:
    notes = list(piece.notes)
    notes[index] = replace(notes[index], **changes)
    return replace(piece, notes=tuple(notes))


def _replace_track(
    piece: CanonicalPiece, index: int = 0, **changes: object
) -> CanonicalPiece:
    tracks = list(piece.tracks)
    tracks[index] = replace(tracks[index], **changes)
    return replace(piece, tracks=tuple(tracks))


def _replace_target(
    piece: CanonicalPiece, index: int = 0, **changes: object
) -> CanonicalPiece:
    targets = list(piece.targets)
    targets[index] = replace(targets[index], **changes)
    return replace(piece, targets=tuple(targets))


def _replace_provenance(
    piece: CanonicalPiece, index: int = 0, **changes: object
) -> CanonicalPiece:
    provenance = list(piece.provenance)
    provenance[index] = replace(provenance[index], **changes)
    return replace(piece, provenance=tuple(provenance))


def _overlap_warnings(piece: CanonicalPiece) -> tuple[ValidationIssue, ...]:
    return tuple(
        issue
        for issue in validate_piece(piece).issues
        if issue.code == "OVERLAPPING_SAME_PITCH_NOTES"
    )


def _pitched_note(
    template: CanonicalNote,
    *,
    note_id: str,
    onset: RationalTime,
    duration: RationalTime,
    pitch: int = 60,
) -> CanonicalNote:
    return replace(
        template,
        note_id=note_id,
        track_id="track:melody",
        pitch=pitch,
        onset_qn=onset,
        duration_qn=duration,
        is_percussion=False,
        is_grace=duration == RationalTime(0),
        source_onset_ticks=None,
        source_duration_ticks=None,
        source_onset_seconds=None,
        source_duration_seconds=None,
    )


def _provenance_record(
    provenance_id: str,
    parents: tuple[str, ...] = (),
) -> ProvenanceRecord:
    return ProvenanceRecord(
        provenance_id=provenance_id,
        kind="derivation",
        source="ordering_fixture",
        record_id=None,
        uri=None,
        version=None,
        checksum_sha256=None,
        created_at=None,
        parents=parents,
        details=(),
    )


def _targets_for_beats(
    targets: tuple[TargetArray, ...], beats: tuple[CanonicalBeat, ...]
) -> tuple[TargetArray, ...]:
    entity_ids = tuple(beat.beat_id for beat in beats)
    return tuple(
        replace(
            target,
            entity_ids=entity_ids,
            values=tuple("major" for _ in beats),
            mask=tuple(True for _ in beats),
            confidence=tuple(1.0 for _ in beats),
            source=tuple("human" for _ in beats),
            provenance=tuple("prov:annotation" for _ in beats),
        )
        for target in targets
    )


def _pickup_piece(piece: CanonicalPiece) -> CanonicalPiece:
    bars = (
        replace(
            piece.bars[0],
            bar_id="bar:000",
            duration_qn=RationalTime(1),
            metric_offset_qn=RationalTime(3),
            is_pickup=True,
            is_incomplete=True,
            display_number="0",
        ),
        CanonicalBar(
            bar_id="bar:001",
            index=1,
            start_qn=RationalTime(1),
            duration_qn=RationalTime(4),
            meter_event_id="meter:000",
            metric_offset_qn=RationalTime(0),
            is_pickup=False,
            is_incomplete=False,
            display_number="1",
            provenance_id="prov:source",
        ),
    )
    beats = (
        replace(
            piece.beats[0],
            beat_id="beat:000",
            index_in_bar=3,
            position_in_bar_qn=RationalTime(3),
            is_downbeat=False,
        ),
        *tuple(
            CanonicalBeat(
                beat_id=f"beat:{index + 1:03d}",
                bar_id="bar:001",
                meter_event_id="meter:000",
                index_in_bar=index,
                start_qn=RationalTime(index + 1),
                duration_qn=RationalTime(1),
                position_in_bar_qn=RationalTime(index),
                is_downbeat=index == 0,
                strength=1.0 if index == 0 else 0.5,
                provenance_id="prov:source",
            )
            for index in range(4)
        ),
    )
    notes = (
        replace(piece.notes[0], duration_qn=RationalTime(5)),
        replace(piece.notes[1], onset_qn=RationalTime(1)),
    )
    return replace(
        piece,
        duration_qn=RationalTime(5),
        notes=notes,
        bars=bars,
        beats=beats,
        targets=_targets_for_beats(piece.targets, beats),
    )


def _six_eight_piece(piece: CanonicalPiece) -> CanonicalPiece:
    meter = replace(piece.meter_events[0], numerator=6, denominator=8)
    bar = replace(piece.bars[0], duration_qn=RationalTime(3))
    beats = tuple(
        CanonicalBeat(
            beat_id=f"beat:{index:03d}",
            bar_id="bar:000",
            meter_event_id="meter:000",
            index_in_bar=index,
            start_qn=RationalTime(index, 2),
            duration_qn=RationalTime(1, 2),
            position_in_bar_qn=RationalTime(index, 2),
            is_downbeat=index == 0,
            strength=1.0 if index == 0 else 0.5,
            provenance_id="prov:source",
        )
        for index in range(6)
    )
    notes = (
        replace(piece.notes[0], duration_qn=RationalTime(3)),
        piece.notes[1],
    )
    return replace(
        piece,
        duration_qn=RationalTime(3),
        notes=notes,
        bars=(bar,),
        beats=beats,
        meter_events=(meter,),
        targets=_targets_for_beats(piece.targets, beats),
    )


def test_validation_code_coverage_is_complete() -> None:
    declared = set(get_args(ValidationCode))
    assert declared == VALIDATOR_TESTED_CODES | SERIALIZATION_ONLY_CODES
    assert VALIDATOR_TESTED_CODES.isdisjoint(SERIALIZATION_ONLY_CODES)


def test_validation_module_exports_only_public_api() -> None:
    assert set(validation.__all__) == {
        "ValidationCode",
        "ValidationIssue",
        "ValidationReport",
        "CanonicalValidationError",
        "validate_piece",
        "validate_or_raise",
    }
    assert all(not name.startswith("_") for name in validation.__all__)


def test_empty_report_properties_and_frozen_slots() -> None:
    report = ValidationReport(())
    assert report.errors == ()
    assert report.warnings == ()
    assert report.is_valid
    assert not hasattr(report, "__dict__")
    with pytest.raises(FrozenInstanceError):
        report.issues = ()  # type: ignore[misc]


def test_clean_piece_has_no_issues(valid_piece: CanonicalPiece) -> None:
    assert validate_piece(valid_piece) == ValidationReport(())
    validate_or_raise(valid_piece)


def test_report_properties_warnings_and_exception_retention(
    valid_piece: CanonicalPiece,
) -> None:
    warning_piece = replace(valid_piece, source_resolution=None)
    warning_report = validate_piece(warning_piece)
    assert warning_report.errors == ()
    assert warning_report.warnings
    assert warning_report.is_valid
    validate_or_raise(warning_piece)

    invalid_piece = replace(warning_piece, schema_version="3.0.0")
    expected = validate_piece(invalid_piece)
    with pytest.raises(CanonicalValidationError) as captured:
        validate_or_raise(invalid_piece)
    assert captured.value.report == expected
    assert captured.value.report.errors
    assert captured.value.report.warnings


def test_issue_sorting_repetition_and_input_immutability(
    valid_piece: CanonicalPiece,
) -> None:
    piece = replace(
        valid_piece,
        schema_version="bad",
        dataset_name="",
        source_resolution=None,
    )
    before = repr(piece)
    first = validate_piece(piece)
    second = validate_piece(piece)
    assert first == second
    assert repr(piece) == before
    assert first.issues == tuple(
        sorted(
            first.issues,
            key=lambda issue: (
                issue.path,
                issue.severity,
                issue.code,
                issue.entity_id or "",
                issue.message,
            ),
        )
    )
    assert all(issue.path.startswith("/") for issue in first.issues)


def test_validation_issue_is_frozen_and_slotted() -> None:
    issue = ValidationIssue("FIELD_VALUE_INVALID", "error", "bad", "/x", None)
    assert not hasattr(issue, "__dict__")
    with pytest.raises(FrozenInstanceError):
        issue.message = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("piece_factory", "expected_code"),
    [
        (lambda piece: replace(piece, schema_version="2.0.1"), "SCHEMA_VERSION_UNSUPPORTED"),
        (lambda piece: replace(piece, dataset_name=" bad "), "FIELD_VALUE_INVALID"),
        (
            lambda piece: replace(
                piece, metadata=replace(piece.metadata, source_format="wav")
            ),
            "FIELD_VALUE_INVALID",
        ),
        (
            lambda piece: _replace_note(piece, source_onset_seconds=inf),
            "VALUE_NOT_FINITE",
        ),
        (
            lambda piece: replace(
                piece,
                key_signature_events=(
                    KeySignatureEvent(
                        "keysig:000",
                        RationalTime(0),
                        8,
                        "major",
                        None,
                        "prov:source",
                    ),
                ),
            ),
            "FIELD_VALUE_INVALID",
        ),
        (
            lambda piece: _replace_note(piece, spelling_step=None, spelling_alter=1),
            "FIELD_VALUE_INVALID",
        ),
        (
            lambda piece: _replace_provenance(piece, created_at="2026-02-30T00:00:00Z"),
            "FIELD_VALUE_INVALID",
        ),
        (
            lambda piece: _replace_provenance(piece, checksum_sha256="ABC"),
            "FIELD_VALUE_INVALID",
        ),
        (
            lambda piece: replace(
                piece,
                quality_flags=(
                    QualityFlag(
                        "Bad-Code",
                        "warning",
                        "diagnostic",
                        (piece.piece_id,),
                        "prov:source",
                    ),
                ),
            ),
            "QUALITY_FLAG_CODE_INVALID",
        ),
    ],
)
def test_schema_and_general_field_validation(
    valid_piece: CanonicalPiece, piece_factory: object, expected_code: str
) -> None:
    _assert_code(piece_factory(valid_piece), expected_code)  # type: ignore[operator]


def test_key_signature_modes_and_other_raw_value(valid_piece: CanonicalPiece) -> None:
    valid_modal = replace(
        valid_piece,
        key_signature_events=(
            KeySignatureEvent(
                "keysig:000",
                RationalTime(0),
                0,
                "dorian",
                "D dorian",
                "prov:source",
            ),
        ),
    )
    assert validate_piece(valid_modal).is_valid
    invalid_other = replace(
        valid_modal,
        key_signature_events=(
            replace(valid_modal.key_signature_events[0], mode="other", raw_value=None),
        ),
    )
    _assert_code(invalid_other, "FIELD_VALUE_INVALID")


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (lambda piece: _replace_track(piece, track_id="bad id"), "ENTITY_ID_INVALID"),
        (
            lambda piece: _replace_track(piece, track_id="note:wrong"),
            "ENTITY_ID_PREFIX_INVALID",
        ),
        (
            lambda piece: _replace_track(
                piece, 1, track_id=piece.tracks[0].track_id
            ),
            "ENTITY_ID_DUPLICATE",
        ),
        (
            lambda piece: _replace_note(piece, track_id="track:missing"),
            "ENTITY_REFERENCE_INVALID",
        ),
        (
            lambda piece: replace(piece, tracks=tuple(reversed(piece.tracks))),
            "COLLECTION_ORDER_INVALID",
        ),
    ],
)
def test_entity_ids_references_and_order(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


def test_note_order_uses_canonical_track_order_independent_of_track_tuple(
    valid_piece: CanonicalPiece,
) -> None:
    piece = replace(
        valid_piece,
        tracks=tuple(reversed(valid_piece.tracks)),
        notes=tuple(reversed(valid_piece.notes)),
    )
    order_paths = {
        issue.path
        for issue in validate_piece(piece).issues
        if issue.code == "COLLECTION_ORDER_INVALID"
    }
    assert {"/tracks", "/notes"} <= order_paths


def test_canonical_note_order_is_not_rejected_when_tracks_are_unsorted(
    valid_piece: CanonicalPiece,
) -> None:
    piece = replace(valid_piece, tracks=tuple(reversed(valid_piece.tracks)))
    order_paths = {
        issue.path
        for issue in validate_piece(piece).issues
        if issue.code == "COLLECTION_ORDER_INVALID"
    }
    assert "/tracks" in order_paths
    assert "/notes" not in order_paths


def test_canonical_track_and_note_order_remains_clean(
    valid_piece: CanonicalPiece,
) -> None:
    assert not {
        issue.path
        for issue in validate_piece(valid_piece).issues
        if issue.code == "COLLECTION_ORDER_INVALID"
    }


@pytest.mark.parametrize(
    "variant",
    [
        lambda piece: replace(
            piece,
            beats=(replace(piece.beats[0], bar_id="bar:missing"),) + piece.beats[1:],
        ),
        lambda piece: replace(
            piece,
            bars=(replace(piece.bars[0], meter_event_id="meter:missing"),),
        ),
        lambda piece: _replace_note(piece, provenance_id="prov:missing"),
        lambda piece: replace(
            piece,
            quality_flags=(
                QualityFlag(
                    "canonical.missing_entity",
                    "warning",
                    "missing entity",
                    ("note:missing",),
                    "prov:source",
                ),
            ),
        ),
    ],
)
def test_missing_raw_and_quality_flag_references(
    valid_piece: CanonicalPiece, variant: object
) -> None:
    _assert_code(variant(valid_piece), "ENTITY_REFERENCE_INVALID")  # type: ignore[operator]


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (
            lambda piece: replace(
                piece,
                key_signature_events=(
                    KeySignatureEvent(
                        "keysig:000",
                        RationalTime(-1),
                        0,
                        "major",
                        None,
                        "prov:source",
                    ),
                ),
            ),
            "TIME_NEGATIVE",
        ),
        (
            lambda piece: _replace_note(piece, duration_qn=RationalTime(-1)),
            "DURATION_NEGATIVE",
        ),
        (
            lambda piece: _replace_note(
                piece, duration_qn=RationalTime(0), is_grace=False
            ),
            "ZERO_DURATION_NON_GRACE",
        ),
        (lambda piece: _replace_note(piece, pitch=128), "PITCH_OUT_OF_RANGE"),
        (lambda piece: _replace_note(piece, velocity=128), "VELOCITY_OUT_OF_RANGE"),
        (lambda piece: _replace_note(piece, channel=16), "CHANNEL_OUT_OF_RANGE"),
        (lambda piece: _replace_note(piece, program=128), "PROGRAM_OUT_OF_RANGE"),
        (
            lambda piece: _replace_track(piece, source_track_index=-1),
            "SOURCE_INDEX_INVALID",
        ),
        (
            lambda piece: _replace_note(piece, is_percussion=True),
            "PERCUSSION_MISMATCH",
        ),
        (
            lambda piece: replace(piece, duration_qn=RationalTime(3)),
            "PIECE_DURATION_TOO_SHORT",
        ),
    ],
)
def test_note_time_range_and_percussion_rules(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


def test_positive_grace_crossing_and_different_pitch_overlap_are_valid(
    valid_piece: CanonicalPiece,
) -> None:
    pickup = _pickup_piece(valid_piece)
    grace = replace(
        valid_piece.notes[0],
        note_id="note:grace",
        onset_qn=RationalTime(2),
        duration_qn=RationalTime(0),
        pitch=61,
        is_grace=True,
        source_onset_ticks=960,
        source_duration_ticks=0,
        source_onset_seconds=1.0,
        source_duration_seconds=0.0,
    )
    other_pitch = replace(
        valid_piece.notes[0],
        note_id="note:overlap-other-pitch",
        onset_qn=RationalTime(1),
        duration_qn=RationalTime(1),
        pitch=62,
        source_onset_ticks=480,
        source_duration_ticks=480,
        source_onset_seconds=0.5,
        source_duration_seconds=0.5,
    )
    piece = replace(
        pickup,
        notes=(pickup.notes[0], other_pitch, pickup.notes[1], grace),
    )
    report = validate_piece(piece)
    assert report.is_valid
    assert "OVERLAPPING_SAME_PITCH_NOTES" not in {issue.code for issue in report.issues}


def test_touching_half_open_intervals_do_not_overlap(
    valid_piece: CanonicalPiece,
) -> None:
    notes = (
        _pitched_note(
            valid_piece.notes[0],
            note_id="note:touch-000",
            onset=RationalTime(0),
            duration=RationalTime(1),
        ),
        _pitched_note(
            valid_piece.notes[0],
            note_id="note:touch-001",
            onset=RationalTime(1),
            duration=RationalTime(1),
        ),
    )
    assert _overlap_warnings(replace(valid_piece, notes=notes)) == ()


def test_simple_and_nested_same_pitch_overlaps_warn_once_per_later_note(
    valid_piece: CanonicalPiece,
) -> None:
    notes = (
        _pitched_note(
            valid_piece.notes[0],
            note_id="note:nested-000",
            onset=RationalTime(0),
            duration=RationalTime(4),
        ),
        _pitched_note(
            valid_piece.notes[0],
            note_id="note:nested-001",
            onset=RationalTime(1),
            duration=RationalTime(1),
        ),
        _pitched_note(
            valid_piece.notes[0],
            note_id="note:nested-002",
            onset=RationalTime(2),
            duration=RationalTime(1),
        ),
    )
    warnings = _overlap_warnings(replace(valid_piece, notes=notes))
    assert [warning.path for warning in warnings] == ["/notes/1", "/notes/2"]


def test_overlap_chain_is_detected_without_duplicate_warnings(
    valid_piece: CanonicalPiece,
) -> None:
    notes = tuple(
        _pitched_note(
            valid_piece.notes[0],
            note_id=f"note:chain-{index:03d}",
            onset=RationalTime(index),
            duration=RationalTime(2),
        )
        for index in range(3)
    )
    warnings = _overlap_warnings(replace(valid_piece, notes=notes))
    assert [warning.path for warning in warnings] == ["/notes/1", "/notes/2"]
    assert len(warnings) == len(set(warnings))


def test_overlap_groups_separate_tracks_and_pitches(
    valid_piece: CanonicalPiece,
) -> None:
    melody = _pitched_note(
        valid_piece.notes[0],
        note_id="note:melody-overlap-group",
        onset=RationalTime(0),
        duration=RationalTime(2),
        pitch=60,
    )
    drums = replace(
        valid_piece.notes[1],
        note_id="note:drums-overlap-group",
        pitch=60,
        onset_qn=RationalTime(1),
        duration_qn=RationalTime(1),
        source_onset_ticks=None,
        source_duration_ticks=None,
        source_onset_seconds=None,
        source_duration_seconds=None,
    )
    other_pitch = _pitched_note(
        valid_piece.notes[0],
        note_id="note:other-pitch-group",
        onset=RationalTime(1),
        duration=RationalTime(1),
        pitch=61,
    )
    piece = replace(valid_piece, notes=(melody, other_pitch, drums))
    assert _overlap_warnings(piece) == ()


def test_percussion_same_pitch_overlap_uses_the_same_sweep(
    valid_piece: CanonicalPiece,
) -> None:
    first = replace(
        valid_piece.notes[1],
        note_id="note:drums-sweep-000",
        onset_qn=RationalTime(0),
        duration_qn=RationalTime(2),
        source_onset_ticks=None,
        source_duration_ticks=None,
        source_onset_seconds=None,
        source_duration_seconds=None,
    )
    second = replace(
        first,
        note_id="note:drums-sweep-001",
        onset_qn=RationalTime(1),
        duration_qn=RationalTime(1),
    )
    warnings = _overlap_warnings(replace(valid_piece, notes=(first, second)))
    assert [warning.path for warning in warnings] == ["/notes/1"]


def test_zero_duration_grace_notes_do_not_participate_in_overlap_sweep(
    valid_piece: CanonicalPiece,
) -> None:
    sounding = _pitched_note(
        valid_piece.notes[0],
        note_id="note:sounding",
        onset=RationalTime(0),
        duration=RationalTime(2),
    )
    grace = _pitched_note(
        valid_piece.notes[0],
        note_id="note:grace-overlap",
        onset=RationalTime(1),
        duration=RationalTime(0),
    )
    assert _overlap_warnings(replace(valid_piece, notes=(sounding, grace))) == ()


def test_unsorted_notes_still_receive_correct_overlap_warning(
    valid_piece: CanonicalPiece,
) -> None:
    later = _pitched_note(
        valid_piece.notes[0],
        note_id="note:unsorted-later",
        onset=RationalTime(1),
        duration=RationalTime(1),
    )
    earlier = _pitched_note(
        valid_piece.notes[0],
        note_id="note:unsorted-earlier",
        onset=RationalTime(0),
        duration=RationalTime(2),
    )
    report = validate_piece(replace(valid_piece, notes=(later, earlier)))
    overlap_paths = [
        issue.path
        for issue in report.issues
        if issue.code == "OVERLAPPING_SAME_PITCH_NOTES"
    ]
    assert overlap_paths == ["/notes/0"]
    assert any(
        issue.code == "COLLECTION_ORDER_INVALID" and issue.path == "/notes"
        for issue in report.issues
    )


def test_large_overlap_group_has_linear_warning_count(
    valid_piece: CanonicalPiece,
) -> None:
    count = 2_000
    notes = tuple(
        _pitched_note(
            valid_piece.notes[0],
            note_id=f"note:large-{index:04d}",
            onset=RationalTime(index),
            duration=RationalTime(2),
        )
        for index in range(count)
    )
    piece = replace(
        valid_piece,
        duration_qn=RationalTime(count + 1),
        notes=notes,
    )
    warnings = _overlap_warnings(piece)
    assert len(warnings) == count - 1
    assert len({warning.path for warning in warnings}) == count - 1


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (
            lambda piece: replace(
                piece,
                tempo_events=(
                    replace(piece.tempo_events[0], microseconds_per_quarter=0),
                ),
            ),
            "TEMPO_INVALID",
        ),
        (
            lambda piece: replace(
                piece,
                tempo_events=(
                    replace(piece.tempo_events[0], onset_qn=RationalTime(1)),
                ),
            ),
            "TEMPO_INITIAL_MISSING",
        ),
        (
            lambda piece: replace(
                piece,
                tempo_events=(
                    piece.tempo_events[0],
                    replace(
                        piece.tempo_events[0],
                        tempo_event_id="tempo:001",
                        microseconds_per_quarter=600_000,
                    ),
                ),
            ),
            "TEMPO_DUPLICATE_ONSET",
        ),
        (
            lambda piece: replace(
                piece,
                meter_events=(replace(piece.meter_events[0], denominator=3),),
            ),
            "METER_INVALID",
        ),
        (
            lambda piece: replace(
                piece,
                meter_events=(
                    replace(piece.meter_events[0], onset_qn=RationalTime(1)),
                ),
            ),
            "METER_INITIAL_MISSING",
        ),
        (
            lambda piece: replace(
                piece,
                meter_events=(
                    piece.meter_events[0],
                    replace(
                        piece.meter_events[0],
                        meter_event_id="meter:001",
                    ),
                ),
            ),
            "METER_DUPLICATE_ONSET",
        ),
        (
            lambda piece: replace(
                piece,
                meter_events=(
                    piece.meter_events[0],
                    MeterEvent(
                        "meter:001",
                        RationalTime(2),
                        3,
                        4,
                        "prov:source",
                    ),
                ),
            ),
            "METER_NOT_AT_BAR_START",
        ),
    ],
)
def test_tempo_and_meter_validation(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (
            lambda piece: replace(
                piece, bars=(replace(piece.bars[0], index=1),)
            ),
            "BAR_INVALID",
        ),
        (
            lambda piece: replace(
                piece, bars=(replace(piece.bars[0], start_qn=RationalTime(1)),)
            ),
            "BAR_COVERAGE_INVALID",
        ),
        (
            lambda piece: replace(
                piece, bars=(replace(piece.bars[0], duration_qn=RationalTime(3)),)
            ),
            "BAR_METER_MISMATCH",
        ),
        (
            lambda piece: replace(
                piece, beats=(replace(piece.beats[0], index_in_bar=-1),) + piece.beats[1:]
            ),
            "BEAT_INVALID",
        ),
        (
            lambda piece: replace(
                piece,
                beats=(replace(piece.beats[0], is_downbeat=False),) + piece.beats[1:],
            ),
            "BEAT_GRID_INVALID",
        ),
    ],
)
def test_bar_and_beat_errors(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


def test_invalid_pickup_offset_and_nonfinite_beat_strength(
    valid_piece: CanonicalPiece,
) -> None:
    invalid_pickup = replace(
        valid_piece,
        bars=(
            replace(
                valid_piece.bars[0],
                duration_qn=RationalTime(1),
                metric_offset_qn=RationalTime(5, 2),
                is_pickup=True,
                is_incomplete=True,
            ),
        ),
    )
    _assert_code(invalid_pickup, "BAR_INVALID")
    nonfinite_strength = replace(
        valid_piece,
        beats=(replace(valid_piece.beats[0], strength=inf),) + valid_piece.beats[1:],
    )
    report = _assert_code(nonfinite_strength, "VALUE_NOT_FINITE")
    assert "BEAT_INVALID" not in {issue.code for issue in report.issues}


def test_pickup_and_compound_meter_grids_are_valid(
    valid_piece: CanonicalPiece,
) -> None:
    assert validate_piece(_pickup_piece(valid_piece)).issues == ()
    assert validate_piece(_six_eight_piece(valid_piece)).issues == ()


def test_annotation_rules_and_target_alignment_span(
    valid_piece: CanonicalPiece,
) -> None:
    forbidden = replace(
        valid_piece,
        annotations=(
            AnnotationSpan(
                "span:bad",
                "theory.chord",
                "observation",
                RationalTime(0),
                RationalTime(1),
                None,
                "C",
                "prov:source",
            ),
        ),
    )
    _assert_code(forbidden, "ANNOTATION_INVALID")

    span = AnnotationSpan(
        "span:cadence",
        "theory.cadence",
        "target_alignment",
        RationalTime(4),
        RationalTime(4),
        None,
        None,
        "prov:source",
    )
    target = TargetArray(
        target_id="target:cadence",
        task="theory.cadence",
        annotation_view_id=None,
        alignment_type="annotation_span",
        entity_ids=("span:cadence",),
        value_type="categorical",
        class_labels=("authentic", "no_cadence"),
        values=("authentic",),
        mask=(True,),
        confidence=(None,),
        source=("human",),
        provenance=("prov:annotation",),
    )
    aligned = replace(
        valid_piece,
        annotations=(span,),
        targets=(target,) + valid_piece.targets,
    )
    assert validate_piece(aligned).issues == ()


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (
            lambda piece: _replace_target(piece, annotation_view_id=" bad "),
            "TARGET_VIEW_INVALID",
        ),
        (
            lambda piece: _replace_target(piece, 1, annotation_view_id=None),
            "TARGET_VIEW_DUPLICATE",
        ),
        (
            lambda piece: _replace_target(piece, mask=(True,)),
            "TARGET_LENGTH_MISMATCH",
        ),
        (
            lambda piece: _replace_target(
                piece,
                entity_ids=(
                    piece.targets[0].entity_ids[0],
                    piece.targets[0].entity_ids[0],
                    *piece.targets[0].entity_ids[2:],
                ),
            ),
            "TARGET_ENTITY_DUPLICATE",
        ),
        (
            lambda piece: _replace_target(
                piece,
                alignment_type="track",
                entity_ids=piece.targets[0].entity_ids,
            ),
            "TARGET_ALIGNMENT_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                entity_ids=(
                    "beat:missing",
                    *piece.targets[0].entity_ids[1:],
                ),
            ),
            "TARGET_ENTITY_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                values=("dominant", *piece.targets[0].values[1:]),
            ),
            "TARGET_VALUE_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                mask=(False, *piece.targets[0].mask[1:]),
            ),
            "TARGET_MASK_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                confidence=(2.0, *piece.targets[0].confidence[1:]),
            ),
            "TARGET_CONFIDENCE_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                source=(None, *piece.targets[0].source[1:]),
            ),
            "TARGET_SOURCE_INVALID",
        ),
        (
            lambda piece: _replace_target(
                piece,
                provenance=("prov:missing", *piece.targets[0].provenance[1:]),
            ),
            "TARGET_PROVENANCE_INVALID",
        ),
    ],
)
def test_target_error_semantics(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


@pytest.mark.parametrize("value_type", ["categorical", "scalar", "multi_label", "distribution"])
def test_all_target_value_types_and_unknown_confidence(
    valid_piece: CanonicalPiece, value_type: str
) -> None:
    target = valid_piece.targets[0]
    changes: dict[str, object]
    if value_type == "categorical":
        changes = {}
    elif value_type == "scalar":
        changes = {
            "value_type": "scalar",
            "class_labels": None,
            "values": (1.0, 2, 3.5, 4),
        }
    elif value_type == "multi_label":
        changes = {
            "value_type": "multi_label",
            "class_labels": ("a", "b"),
            "values": (("a",), ("a", "b"), (), ("b",)),
        }
    else:
        changes = {
            "value_type": "distribution",
            "class_labels": ("a", "b"),
            "values": ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (0.25, 0.75)),
        }
    changes["confidence"] = (None, 1.0, 1.0, 1.0)
    piece = _replace_target(valid_piece, **changes)
    report = validate_piece(piece)
    assert report.is_valid, report.issues
    assert "TARGET_CONFIDENCE_INVALID" not in {issue.code for issue in report.issues}
    assert "LOW_CONFIDENCE_TARGET" not in {issue.code for issue in report.issues}
    assert target.annotation_view_id is None
    assert valid_piece.targets[1].annotation_view_id == "analysis.alternative"


@pytest.mark.parametrize(
    "changes",
    [
        {"value_type": "scalar", "class_labels": ("bad",), "values": (1, 2, 3, 4)},
        {
            "value_type": "multi_label",
            "class_labels": ("a", "b"),
            "values": (("b", "a"), ("a",), ("a",), ("a",)),
        },
        {
            "value_type": "distribution",
            "class_labels": ("a", "b"),
            "values": ((0.8, 0.8), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
        },
        {
            "value_type": "multi_label",
            "class_labels": None,
            "values": (("a",), ("a",), ("a",), ("a",)),
        },
    ],
)
def test_invalid_target_value_type_encodings(
    valid_piece: CanonicalPiece, changes: dict[str, object]
) -> None:
    _assert_code(_replace_target(valid_piece, **changes), "TARGET_VALUE_INVALID")


def test_valid_unavailable_target_entries_use_null_aligned_values(
    valid_piece: CanonicalPiece,
) -> None:
    piece = _replace_target(
        valid_piece,
        values=(None, *valid_piece.targets[0].values[1:]),
        mask=(False, *valid_piece.targets[0].mask[1:]),
        confidence=(None, *valid_piece.targets[0].confidence[1:]),
        source=(None, *valid_piece.targets[0].source[1:]),
        provenance=(None, *valid_piece.targets[0].provenance[1:]),
    )
    assert validate_piece(piece).issues == ()


def test_piece_target_alignment_rule(valid_piece: CanonicalPiece) -> None:
    target = replace(
        valid_piece.targets[0],
        target_id="target:piece",
        task="quality.overall",
        alignment_type="piece",
        entity_ids=(valid_piece.piece_id,),
        value_type="scalar",
        class_labels=None,
        values=(1.0,),
        mask=(True,),
        confidence=(None,),
        source=("human",),
        provenance=("prov:annotation",),
    )
    piece = replace(valid_piece, targets=(target,) + valid_piece.targets)
    assert validate_piece(piece).issues == ()
    invalid = replace(
        piece,
        targets=(replace(target, entity_ids=(valid_piece.piece_id, valid_piece.piece_id)),)
        + valid_piece.targets,
    )
    _assert_code(invalid, "TARGET_ALIGNMENT_INVALID")


def test_explicit_negative_target_is_available_not_masked(
    valid_piece: CanonicalPiece,
) -> None:
    piece = _replace_target(
        valid_piece,
        class_labels=("major", "no_chord"),
        values=("no_chord", "major", "major", "major"),
    )
    assert validate_piece(piece).is_valid


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        (lambda piece: replace(piece, provenance=()), "PROVENANCE_MISSING"),
        (
            lambda piece: _replace_provenance(
                piece, 1, parents=("prov:missing",)
            ),
            "PROVENANCE_PARENT_INVALID",
        ),
        (
            lambda piece: replace(
                piece,
                provenance=(
                    replace(piece.provenance[0], parents=("prov:annotation",)),
                    replace(piece.provenance[1], parents=("prov:source",)),
                ),
            ),
            "PROVENANCE_CYCLE",
        ),
        (
            lambda piece: _replace_provenance(
                piece, details=(("z", 1), ("a", 2))
            ),
            "PROVENANCE_DETAIL_INVALID",
        ),
    ],
)
def test_provenance_errors(
    valid_piece: CanonicalPiece, variant: object, code: str
) -> None:
    _assert_code(variant(valid_piece), code)  # type: ignore[operator]


def test_provenance_order_uses_smallest_ready_id_one_at_a_time(
    valid_piece: CanonicalPiece,
) -> None:
    canonical = (
        _provenance_record("prov:a"),
        _provenance_record("prov:b", ("prov:a",)),
        _provenance_record("prov:z"),
    )
    report = validate_piece(replace(valid_piece, provenance=canonical))
    assert not any(
        issue.code == "COLLECTION_ORDER_INVALID" and issue.path == "/provenance"
        for issue in report.issues
    )

    layered = (canonical[0], canonical[2], canonical[1])
    report = validate_piece(replace(valid_piece, provenance=layered))
    assert any(
        issue.code == "COLLECTION_ORDER_INVALID" and issue.path == "/provenance"
        for issue in report.issues
    )
    assert not any(
        issue.code == "PROVENANCE_PARENT_INVALID" for issue in report.issues
    )


def test_independent_provenance_roots_are_lexicographic(
    valid_piece: CanonicalPiece,
) -> None:
    canonical = tuple(
        _provenance_record(provenance_id)
        for provenance_id in ("prov:a", "prov:m", "prov:z")
    )
    report = validate_piece(replace(valid_piece, provenance=canonical))
    assert not any(
        issue.code == "COLLECTION_ORDER_INVALID" and issue.path == "/provenance"
        for issue in report.issues
    )
    reversed_report = validate_piece(
        replace(valid_piece, provenance=tuple(reversed(canonical)))
    )
    assert any(
        issue.code == "COLLECTION_ORDER_INVALID" and issue.path == "/provenance"
        for issue in reversed_report.issues
    )


def test_valid_branching_provenance_dag_is_accepted(
    valid_piece: CanonicalPiece,
) -> None:
    provenance = (
        _provenance_record("prov:a"),
        _provenance_record("prov:b", ("prov:a",)),
        _provenance_record("prov:c", ("prov:a",)),
        _provenance_record("prov:d", ("prov:b", "prov:c")),
    )
    report = validate_piece(replace(valid_piece, provenance=provenance))
    assert not any(
        issue.code in {"COLLECTION_ORDER_INVALID", "PROVENANCE_CYCLE"}
        and issue.path == "/provenance"
        for issue in report.issues
    )


@pytest.mark.parametrize(
    "provenance",
    [
        lambda piece: (
            piece.provenance[0],
            replace(
                piece.provenance[1],
                parents=("prov:source", "prov:source"),
            ),
        ),
        lambda piece: (
            piece.provenance[0],
            replace(
                piece.provenance[1],
                parents=("prov:annotation",),
            ),
        ),
        lambda piece: tuple(reversed(piece.provenance)),
    ],
)
def test_provenance_parent_duplicates_self_reference_and_order(
    valid_piece: CanonicalPiece, provenance: object
) -> None:
    piece = replace(valid_piece, provenance=provenance(valid_piece))  # type: ignore[operator]
    _assert_code(piece, "PROVENANCE_PARENT_INVALID")


def test_provenance_cycle_detection_is_iterative_for_deep_graphs(
    valid_piece: CanonicalPiece,
) -> None:
    count = 1_100
    provenance = tuple(
        ProvenanceRecord(
            provenance_id=f"prov:{index:04d}",
            kind="derivation",
            source="cycle_fixture",
            record_id=None,
            uri=None,
            version=None,
            checksum_sha256=None,
            created_at=None,
            parents=(f"prov:{(index - 1) % count:04d}",),
            details=(),
        )
        for index in range(count)
    )
    _assert_code(replace(valid_piece, provenance=provenance), "PROVENANCE_CYCLE")


def test_quality_flag_severity_and_message_policy(valid_piece: CanonicalPiece) -> None:
    flag = QualityFlag(
        "canonical.test",
        "fatal",
        "",
        (valid_piece.piece_id,),
        "prov:source",
    )
    report = _assert_code(replace(valid_piece, quality_flags=(flag,)), "FIELD_VALUE_INVALID")
    assert sum(issue.code == "FIELD_VALUE_INVALID" for issue in report.issues) >= 2


@pytest.mark.parametrize(
    ("variant", "warning_code"),
    [
        (lambda piece: replace(piece, notes=()), "EMPTY_PIECE"),
        (
            lambda piece: replace(
                piece,
                tracks=piece.tracks
                + (
                    CanonicalTrack(
                        "track:empty",
                        2,
                        None,
                        None,
                        None,
                        None,
                        False,
                        "prov:source",
                    ),
                ),
            ),
            "EMPTY_TRACK",
        ),
        (
            lambda piece: replace(piece, source_resolution=None),
            "SOURCE_RESOLUTION_UNAVAILABLE",
        ),
        (
            lambda piece: replace(
                piece,
                bars=(
                    replace(
                        piece.bars[0],
                        duration_qn=RationalTime(3),
                        is_incomplete=True,
                    ),
                ),
            ),
            "INCOMPLETE_FINAL_BAR",
        ),
        (
            lambda piece: replace(
                piece,
                notes=piece.notes
                + (
                    replace(
                        piece.notes[0],
                        note_id="note:overlap",
                        onset_qn=RationalTime(1),
                        duration_qn=RationalTime(1),
                        source_onset_ticks=480,
                        source_duration_ticks=480,
                        source_onset_seconds=0.5,
                        source_duration_seconds=0.5,
                    ),
                ),
            ),
            "OVERLAPPING_SAME_PITCH_NOTES",
        ),
        (
            lambda piece: replace(
                piece,
                tempo_events=piece.tempo_events
                + (
                    TempoEvent(
                        "tempo:001",
                        RationalTime(2),
                        600_000,
                        "prov:source",
                    ),
                ),
            ),
            "MID_BAR_TEMPO_CHANGE",
        ),
        (
            lambda piece: _replace_target(
                piece,
                confidence=(0.49, *piece.targets[0].confidence[1:]),
            ),
            "LOW_CONFIDENCE_TARGET",
        ),
        (
            lambda piece: replace(
                piece,
                provenance=piece.provenance
                + (
                    ProvenanceRecord(
                        "prov:unused",
                        "derivation",
                        "unused_tool",
                        None,
                        None,
                        None,
                        None,
                        None,
                        ("prov:source",),
                        (),
                    ),
                ),
            ),
            "UNREFERENCED_PROVENANCE",
        ),
        (
            lambda piece: replace(
                piece,
                annotations=(
                    AnnotationSpan(
                        "span:empty",
                        "text.lyric",
                        "observation",
                        RationalTime(0),
                        RationalTime(1),
                        "track:melody",
                        "",
                        "prov:source",
                    ),
                ),
            ),
            "EMPTY_OBSERVATION",
        ),
        (
            lambda piece: _replace_note(piece, duration_qn=RationalTime(3)),
            "PIECE_TRAILING_SILENCE",
        ),
    ],
)
def test_every_documented_warning(
    valid_piece: CanonicalPiece, variant: object, warning_code: str
) -> None:
    report = _assert_code(variant(valid_piece), warning_code)  # type: ignore[operator]
    issue = next(issue for issue in report.issues if issue.code == warning_code)
    assert issue.severity == "warning"


def test_trailing_silence_ignores_structural_and_zero_duration_content(
    valid_piece: CanonicalPiece,
) -> None:
    grace = replace(
        valid_piece.notes[0],
        duration_qn=RationalTime(0),
        is_grace=True,
        source_duration_ticks=0,
        source_duration_seconds=0.0,
    )
    point = AnnotationSpan(
        "span:point",
        "text.rehearsal",
        "observation",
        RationalTime(4),
        RationalTime(4),
        None,
        "A",
        "prov:source",
    )
    piece = replace(valid_piece, notes=(grace,), annotations=(point,))
    assert "PIECE_TRAILING_SILENCE" in _codes(piece)


def test_programmatic_malformed_collections_do_not_crash(
    valid_piece: CanonicalPiece,
) -> None:
    malformed_target = replace(
        valid_piece.targets[0],
        value_type=[],  # type: ignore[arg-type]
        source=([], *valid_piece.targets[0].source[1:]),  # type: ignore[arg-type]
    )
    malformed = replace(
        valid_piece,
        metadata=replace(valid_piece.metadata, source_format=[]),  # type: ignore[arg-type]
        tracks=["bad"],  # type: ignore[arg-type]
        notes=[valid_piece.notes[0]],  # type: ignore[arg-type]
        targets=[malformed_target],  # type: ignore[arg-type]
        provenance=[
            replace(valid_piece.provenance[0], kind=[]),  # type: ignore[arg-type]
        ],  # type: ignore[arg-type]
    )
    report = validate_piece(malformed)
    assert report.issues
    assert "FIELD_VALUE_INVALID" in {issue.code for issue in report.issues}
    assert len(report.issues) == len(set(report.issues))
    assert report.issues == tuple(
        sorted(
            report.issues,
            key=lambda issue: (
                issue.path,
                issue.severity,
                issue.code,
                issue.entity_id or "",
                issue.message,
            ),
        )
    )
    assert report == validate_piece(malformed)


def test_issue_deduplication_preserves_distinct_diagnostics(
    valid_piece: CanonicalPiece,
) -> None:
    malformed = replace(
        valid_piece,
        notes=[],  # type: ignore[arg-type]
        bars=(replace(valid_piece.bars[0], duration_qn=RationalTime(-1)),),
    )
    report = validate_piece(malformed)
    assert len(report.issues) == len(set(report.issues))
    notes_issues = [issue for issue in report.issues if issue.path == "/notes"]
    assert {issue.code for issue in notes_issues} >= {
        "FIELD_VALUE_INVALID",
        "EMPTY_PIECE",
    }
    duration_issues = [
        issue for issue in report.issues if issue.path == "/bars/0/duration_qn"
    ]
    assert {issue.code for issue in duration_issues} >= {
        "DURATION_NEGATIVE",
        "BAR_INVALID",
    }
