"""Validation for immutable canonical symbolic-music records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import re
from typing import Any, Literal, Sequence, TypeAlias

from music_critic.data.schema import (
    SCHEMA_VERSION,
    AlignmentType,
    AnnotationSpan,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    IssueSeverity,
    JsonScalar,
    KeySignatureEvent,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    QualityFlag,
    TargetArray,
    TempoEvent,
)
from music_critic.data.timing import RationalTime


__all__ = [
    "ValidationCode",
    "ValidationIssue",
    "ValidationReport",
    "CanonicalValidationError",
    "validate_piece",
    "validate_or_raise",
]


ValidationCode: TypeAlias = Literal[
    "SCHEMA_VERSION_UNSUPPORTED",
    "JSON_UNKNOWN_FIELD",
    "JSON_MISSING_FIELD",
    "JSON_TYPE_INVALID",
    "FIELD_VALUE_INVALID",
    "RATIONAL_INVALID",
    "RATIONAL_NOT_NORMALIZED",
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
]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    severity: IssueSeverity
    message: str
    path: str
    entity_id: str | None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors


class CanonicalValidationError(ValueError):
    report: ValidationReport

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(
            f"canonical validation failed with {len(report.errors)} error(s) "
            f"and {len(report.warnings)} warning(s)"
        )


_ID_LOCAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_QUALITY_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_SOURCE_FORMATS = {
    "midi",
    "musicxml",
    "json",
    "jsonl",
    "tsv",
    "synthetic",
    "other",
}
_KEY_MODES = {
    "major",
    "minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "locrian",
    "other",
    "unknown",
}
_ANNOTATION_LAYERS = {"observation", "target_alignment"}
_OBSERVATION_PREFIXES = ("text.", "performance.", "notation.", "other.")
_ALIGNMENT_PREFIXES: dict[str, str] = {
    "track": "track",
    "note": "note",
    "bar": "bar",
    "beat": "beat",
    "bar_boundary": "bar",
    "beat_boundary": "beat",
    "annotation_span": "span",
}
_TARGET_VALUE_TYPES = {"categorical", "scalar", "multi_label", "distribution"}
_TARGET_SOURCES = {
    "human",
    "dataset",
    "algorithm",
    "pseudo_label",
    "derived",
    "synthetic",
}
_PROVENANCE_KINDS = {
    "source",
    "conversion",
    "annotation",
    "derivation",
    "default",
    "synthetic",
}
_QUALITY_SEVERITIES = {"info", "warning"}


@dataclass(slots=True)
class _Context:
    piece: CanonicalPiece
    issues: list[ValidationIssue]
    entity_paths: dict[str, str]
    entity_prefixes: dict[str, str]
    track_by_id: dict[str, CanonicalTrack]
    bar_by_id: dict[str, CanonicalBar]
    beat_by_id: dict[str, CanonicalBeat]
    meter_by_id: dict[str, MeterEvent]
    provenance_by_id: dict[str, ProvenanceRecord]
    annotation_by_id: dict[str, AnnotationSpan]

    def add(
        self,
        code: ValidationCode,
        message: str,
        path: str,
        entity_id: str | None = None,
        *,
        severity: IssueSeverity = "error",
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                path=path,
                entity_id=entity_id,
            )
        )


def _entity_id(record: object, attribute: str) -> str | None:
    value = getattr(record, attribute, None)
    return value if isinstance(value, str) else None


def _as_sequence(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> Sequence[Any]:
    if isinstance(value, tuple):
        return value
    ctx.add(
        "FIELD_VALUE_INVALID",
        "collection-valued schema fields must be tuples",
        path,
        entity_id,
    )
    if isinstance(value, list):
        return value
    return ()


def _typed_records(
    ctx: _Context,
    value: object,
    expected_type: type[Any],
    path: str,
) -> list[tuple[int, Any]]:
    records: list[tuple[int, Any]] = []
    for index, record in enumerate(_as_sequence(ctx, value, path)):
        if isinstance(record, expected_type):
            records.append((index, record))
        else:
            ctx.add(
                "FIELD_VALUE_INVALID",
                f"expected {expected_type.__name__}",
                f"{path}/{index}",
            )
    return records


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or isfinite(value))
    )


def _has_ascii_control(value: str) -> bool:
    return any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value)


def _is_open_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and not _has_ascii_control(value)
    )


def _check_open_string(
    ctx: _Context,
    value: object,
    path: str,
    name: str,
    entity_id: str | None = None,
) -> bool:
    if _is_open_string(value):
        return True
    ctx.add(
        "FIELD_VALUE_INVALID",
        f"{name} must be a non-empty, trimmed string without ASCII controls",
        path,
        entity_id,
    )
    return False


def _check_optional_string(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> None:
    if value is not None and not isinstance(value, str):
        ctx.add(
            "FIELD_VALUE_INVALID",
            "value must be a string or None",
            path,
            entity_id,
        )


def _check_bool(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> bool:
    if isinstance(value, bool):
        return True
    ctx.add("FIELD_VALUE_INVALID", "value must be a bool", path, entity_id)
    return False


def _check_finite(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> bool:
    if isinstance(value, float) and not isfinite(value):
        ctx.add("VALUE_NOT_FINITE", "value must be finite", path, entity_id)
        return False
    return True


def _rational(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> RationalTime | None:
    if isinstance(value, RationalTime):
        return value
    ctx.add(
        "FIELD_VALUE_INVALID",
        "value must be a RationalTime",
        path,
        entity_id,
    )
    return None


def _split_id(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    prefix, local = value.split(":", 1)
    if not prefix or not _ID_LOCAL_RE.fullmatch(local):
        return None
    return prefix, local


def _check_id(
    ctx: _Context,
    value: object,
    expected_prefix: str,
    path: str,
    entity_id: str | None = None,
) -> bool:
    parts = _split_id(value)
    if parts is None:
        ctx.add(
            "ENTITY_ID_INVALID",
            "entity ID must have a valid '<prefix>:<local-id>' form",
            path,
            entity_id,
        )
        return False
    if parts[0] != expected_prefix:
        ctx.add(
            "ENTITY_ID_PREFIX_INVALID",
            f"entity ID must use the '{expected_prefix}:' prefix",
            path,
            entity_id,
        )
        return False
    return True


def _register_id(
    ctx: _Context,
    value: object,
    expected_prefix: str,
    path: str,
    entity_id: str | None = None,
) -> None:
    if not _check_id(ctx, value, expected_prefix, path, entity_id):
        return
    assert isinstance(value, str)
    if value in ctx.entity_paths:
        ctx.add(
            "ENTITY_ID_DUPLICATE",
            f"entity ID duplicates {ctx.entity_paths[value]}",
            path,
            value,
        )
        return
    ctx.entity_paths[value] = path
    ctx.entity_prefixes[value] = expected_prefix


def _check_reference(
    ctx: _Context,
    value: object,
    expected_prefix: str,
    path: str,
    entity_id: str | None = None,
) -> bool:
    if not _check_id(ctx, value, expected_prefix, path, entity_id):
        return False
    assert isinstance(value, str)
    if ctx.entity_prefixes.get(value) != expected_prefix:
        ctx.add(
            "ENTITY_REFERENCE_INVALID",
            f"reference does not identify an existing {expected_prefix} entity",
            path,
            entity_id,
        )
        return False
    return True


def _check_optional_provenance_reference(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None,
) -> None:
    if value is None:
        return
    _check_reference(ctx, value, "prov", path, entity_id)


def _check_any_entity_reference(
    ctx: _Context,
    value: object,
    path: str,
    entity_id: str | None = None,
) -> bool:
    parts = _split_id(value)
    if parts is None:
        ctx.add(
            "ENTITY_ID_INVALID",
            "entity reference must have a valid '<prefix>:<local-id>' form",
            path,
            entity_id,
        )
        return False
    if parts[0] not in {
        "piece",
        "track",
        "note",
        "bar",
        "beat",
        "tempo",
        "meter",
        "keysig",
        "span",
        "target",
        "prov",
    }:
        ctx.add(
            "ENTITY_ID_PREFIX_INVALID",
            "entity reference uses an unsupported prefix",
            path,
            entity_id,
        )
        return False
    assert isinstance(value, str)
    if value not in ctx.entity_paths:
        ctx.add(
            "ENTITY_REFERENCE_INVALID",
            "entity reference does not exist",
            path,
            entity_id,
        )
        return False
    return True


def _records(ctx: _Context, name: str, expected_type: type[Any]) -> list[tuple[int, Any]]:
    return _typed_records(ctx, getattr(ctx.piece, name, ()), expected_type, f"/{name}")


def _validate_ids(ctx: _Context) -> None:
    _register_id(ctx, ctx.piece.piece_id, "piece", "/piece_id", _entity_id(ctx.piece, "piece_id"))
    specifications = (
        ("tracks", CanonicalTrack, "track_id", "track"),
        ("notes", CanonicalNote, "note_id", "note"),
        ("bars", CanonicalBar, "bar_id", "bar"),
        ("beats", CanonicalBeat, "beat_id", "beat"),
        ("tempo_events", TempoEvent, "tempo_event_id", "tempo"),
        ("meter_events", MeterEvent, "meter_event_id", "meter"),
        (
            "key_signature_events",
            KeySignatureEvent,
            "key_signature_event_id",
            "keysig",
        ),
        ("annotations", AnnotationSpan, "annotation_id", "span"),
        ("targets", TargetArray, "target_id", "target"),
        ("provenance", ProvenanceRecord, "provenance_id", "prov"),
    )
    for collection, record_type, attribute, prefix in specifications:
        for index, record in _records(ctx, collection, record_type):
            path = f"/{collection}/{index}/{attribute}"
            value = getattr(record, attribute, None)
            _register_id(ctx, value, prefix, path, _entity_id(record, attribute))

    for _, track in _records(ctx, "tracks", CanonicalTrack):
        track_id = _entity_id(track, "track_id")
        if track_id is not None and ctx.entity_prefixes.get(track_id) == "track":
            ctx.track_by_id[track_id] = track
    for _, bar in _records(ctx, "bars", CanonicalBar):
        bar_id = _entity_id(bar, "bar_id")
        if bar_id is not None and ctx.entity_prefixes.get(bar_id) == "bar":
            ctx.bar_by_id[bar_id] = bar
    for _, beat in _records(ctx, "beats", CanonicalBeat):
        beat_id = _entity_id(beat, "beat_id")
        if beat_id is not None and ctx.entity_prefixes.get(beat_id) == "beat":
            ctx.beat_by_id[beat_id] = beat
    for _, meter in _records(ctx, "meter_events", MeterEvent):
        meter_id = _entity_id(meter, "meter_event_id")
        if meter_id is not None and ctx.entity_prefixes.get(meter_id) == "meter":
            ctx.meter_by_id[meter_id] = meter
    for _, provenance in _records(ctx, "provenance", ProvenanceRecord):
        provenance_id = _entity_id(provenance, "provenance_id")
        if (
            provenance_id is not None
            and ctx.entity_prefixes.get(provenance_id) == "prov"
        ):
            ctx.provenance_by_id[provenance_id] = provenance
    for _, annotation in _records(ctx, "annotations", AnnotationSpan):
        annotation_id = _entity_id(annotation, "annotation_id")
        if annotation_id is not None and ctx.entity_prefixes.get(annotation_id) == "span":
            ctx.annotation_by_id[annotation_id] = annotation


def _validate_top_level(ctx: _Context) -> None:
    piece = ctx.piece
    piece_id = _entity_id(piece, "piece_id")
    if piece.schema_version != SCHEMA_VERSION:
        ctx.add(
            "SCHEMA_VERSION_UNSUPPORTED",
            f"schema_version must be exactly {SCHEMA_VERSION!r}",
            "/schema_version",
            piece_id,
        )
    _check_open_string(ctx, piece.dataset_name, "/dataset_name", "dataset_name", piece_id)
    _check_open_string(
        ctx,
        piece.source_group_id,
        "/source_group_id",
        "source_group_id",
        piece_id,
    )
    if piece.split is not None and not isinstance(piece.split, str):
        ctx.add(
            "FIELD_VALUE_INVALID",
            "split must be a string or None",
            "/split",
            piece_id,
        )
    _check_optional_string(ctx, piece.source_path, "/source_path", piece_id)
    if piece.source_resolution is None:
        ctx.add(
            "SOURCE_RESOLUTION_UNAVAILABLE",
            "source resolution is unavailable",
            "/source_resolution",
            piece_id,
            severity="warning",
        )
    elif not _is_int(piece.source_resolution) or piece.source_resolution <= 0:
        ctx.add(
            "FIELD_VALUE_INVALID",
            "source_resolution must be a positive integer or None",
            "/source_resolution",
            piece_id,
        )
    duration = _rational(ctx, piece.duration_qn, "/duration_qn", piece_id)
    if duration is not None and duration < RationalTime(0):
        ctx.add(
            "DURATION_NEGATIVE",
            "piece duration must be non-negative",
            "/duration_qn",
            piece_id,
        )


def _validate_metadata(ctx: _Context) -> None:
    metadata = ctx.piece.metadata
    piece_id = _entity_id(ctx.piece, "piece_id")
    if not isinstance(metadata, PieceMetadata):
        ctx.add("FIELD_VALUE_INVALID", "metadata must be PieceMetadata", "/metadata", piece_id)
        return
    if (
        not isinstance(metadata.source_format, str)
        or metadata.source_format not in _SOURCE_FORMATS
    ):
        ctx.add(
            "FIELD_VALUE_INVALID",
            "source_format is not a supported SourceFormat value",
            "/metadata/source_format",
            piece_id,
        )
    for field in (
        "title",
        "collection",
        "movement_title",
        "movement_number",
        "copyright",
        "language",
    ):
        _check_optional_string(
            ctx,
            getattr(metadata, field),
            f"/metadata/{field}",
            piece_id,
        )
    for field in ("creators", "genres"):
        value = getattr(metadata, field)
        if value is None:
            continue
        for index, item in enumerate(_as_sequence(ctx, value, f"/metadata/{field}", piece_id)):
            if not isinstance(item, str):
                ctx.add(
                    "FIELD_VALUE_INVALID",
                    "metadata collection entries must be strings",
                    f"/metadata/{field}/{index}",
                    piece_id,
                )


def _validate_tracks(ctx: _Context) -> None:
    for index, track in _records(ctx, "tracks", CanonicalTrack):
        base = f"/tracks/{index}"
        track_id = _entity_id(track, "track_id")
        if track.source_track_index is not None:
            if not _is_int(track.source_track_index) or track.source_track_index < 0:
                ctx.add(
                    "SOURCE_INDEX_INVALID",
                    "source_track_index must be a non-negative integer or None",
                    f"{base}/source_track_index",
                    track_id,
                )
        for field in ("name", "instrument_name"):
            _check_optional_string(ctx, getattr(track, field), f"{base}/{field}", track_id)
        if track.program is not None and (
            not _is_int(track.program) or not 0 <= track.program <= 127
        ):
            ctx.add(
                "PROGRAM_OUT_OF_RANGE",
                "program must be in [0, 127]",
                f"{base}/program",
                track_id,
            )
        if track.channel is not None and (
            not _is_int(track.channel) or not 0 <= track.channel <= 15
        ):
            ctx.add(
                "CHANNEL_OUT_OF_RANGE",
                "channel must be in [0, 15]",
                f"{base}/channel",
                track_id,
            )
        _check_bool(ctx, track.is_percussion, f"{base}/is_percussion", track_id)
        _check_optional_provenance_reference(
            ctx, track.provenance_id, f"{base}/provenance_id", track_id
        )


def _validate_notes(ctx: _Context) -> None:
    zero = RationalTime(0)
    for index, note in _records(ctx, "notes", CanonicalNote):
        base = f"/notes/{index}"
        note_id = _entity_id(note, "note_id")
        _check_reference(ctx, note.track_id, "track", f"{base}/track_id", note_id)
        if not _is_int(note.pitch) or not 0 <= note.pitch <= 127:
            ctx.add(
                "PITCH_OUT_OF_RANGE",
                "pitch must be an integer in [0, 127]",
                f"{base}/pitch",
                note_id,
            )
        onset = _rational(ctx, note.onset_qn, f"{base}/onset_qn", note_id)
        duration = _rational(ctx, note.duration_qn, f"{base}/duration_qn", note_id)
        if onset is not None and onset < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "note onset must be non-negative",
                f"{base}/onset_qn",
                note_id,
            )
        if duration is not None:
            if duration < zero:
                ctx.add(
                    "DURATION_NEGATIVE",
                    "note duration must be non-negative",
                    f"{base}/duration_qn",
                    note_id,
                )
            elif (
                duration == zero
                and isinstance(note.is_grace, bool)
                and not note.is_grace
            ):
                ctx.add(
                    "ZERO_DURATION_NON_GRACE",
                    "zero duration is valid only for grace notes",
                    f"{base}/duration_qn",
                    note_id,
                )
        if note.velocity is not None and (
            not _is_int(note.velocity) or not 0 <= note.velocity <= 127
        ):
            ctx.add(
                "VELOCITY_OUT_OF_RANGE",
                "velocity must be in [0, 127]",
                f"{base}/velocity",
                note_id,
            )
        if note.channel is not None and (
            not _is_int(note.channel) or not 0 <= note.channel <= 15
        ):
            ctx.add(
                "CHANNEL_OUT_OF_RANGE",
                "channel must be in [0, 15]",
                f"{base}/channel",
                note_id,
            )
        if note.program is not None and (
            not _is_int(note.program) or not 0 <= note.program <= 127
        ):
            ctx.add(
                "PROGRAM_OUT_OF_RANGE",
                "program must be in [0, 127]",
                f"{base}/program",
                note_id,
            )
        percussion_valid = _check_bool(
            ctx, note.is_percussion, f"{base}/is_percussion", note_id
        )
        _check_bool(ctx, note.is_grace, f"{base}/is_grace", note_id)
        track = ctx.track_by_id.get(note.track_id) if isinstance(note.track_id, str) else None
        if (
            percussion_valid
            and track is not None
            and isinstance(track.is_percussion, bool)
            and note.is_percussion != track.is_percussion
        ):
            ctx.add(
                "PERCUSSION_MISMATCH",
                "note percussion flag must match its canonical track",
                f"{base}/is_percussion",
                note_id,
            )
        if note.spelling_step is not None and (
            not isinstance(note.spelling_step, str)
            or note.spelling_step not in {"A", "B", "C", "D", "E", "F", "G"}
        ):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "spelling_step must be A through G or None",
                f"{base}/spelling_step",
                note_id,
            )
        if note.spelling_alter is not None and not _is_int(note.spelling_alter):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "spelling_alter must be an integer or None",
                f"{base}/spelling_alter",
                note_id,
            )
        if note.spelling_step is None and note.spelling_alter is not None:
            ctx.add(
                "FIELD_VALUE_INVALID",
                "spelling_alter must be None when spelling_step is None",
                f"{base}/spelling_alter",
                note_id,
            )
        for field in ("staff", "voice"):
            value = getattr(note, field)
            if value is not None and (not _is_int(value) or value < 0):
                ctx.add(
                    "SOURCE_INDEX_INVALID",
                    f"{field} must be a non-negative integer or None",
                    f"{base}/{field}",
                    note_id,
                )
        if note.articulations is not None:
            for item_index, item in enumerate(
                _as_sequence(ctx, note.articulations, f"{base}/articulations", note_id)
            ):
                if not isinstance(item, str):
                    ctx.add(
                        "FIELD_VALUE_INVALID",
                        "articulations entries must be strings",
                        f"{base}/articulations/{item_index}",
                        note_id,
                    )
        _check_optional_string(ctx, note.dynamic, f"{base}/dynamic", note_id)
        if note.source_onset_ticks is not None and (
            not _is_int(note.source_onset_ticks) or note.source_onset_ticks < 0
        ):
            ctx.add(
                "SOURCE_INDEX_INVALID",
                "source_onset_ticks must be non-negative or None",
                f"{base}/source_onset_ticks",
                note_id,
            )
        if note.source_duration_ticks is not None and (
            not _is_int(note.source_duration_ticks) or note.source_duration_ticks < 0
        ):
            ctx.add(
                "DURATION_NEGATIVE",
                "source_duration_ticks must be non-negative or None",
                f"{base}/source_duration_ticks",
                note_id,
            )
        for field, code in (
            ("source_onset_seconds", "TIME_NEGATIVE"),
            ("source_duration_seconds", "DURATION_NEGATIVE"),
        ):
            value = getattr(note, field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                ctx.add(
                    "FIELD_VALUE_INVALID",
                    f"{field} must be a number or None",
                    f"{base}/{field}",
                    note_id,
                )
            elif _check_finite(ctx, value, f"{base}/{field}", note_id) and value < 0:
                ctx.add(code, f"{field} must be non-negative", f"{base}/{field}", note_id)
        _check_optional_provenance_reference(
            ctx, note.provenance_id, f"{base}/provenance_id", note_id
        )


def _valid_meter(meter: MeterEvent) -> bool:
    return (
        isinstance(meter.onset_qn, RationalTime)
        and meter.onset_qn >= RationalTime(0)
        and _is_int(meter.numerator)
        and meter.numerator > 0
        and _is_int(meter.denominator)
        and meter.denominator > 0
        and meter.denominator & (meter.denominator - 1) == 0
    )


def _validate_tempo_events(ctx: _Context) -> None:
    zero = RationalTime(0)
    onsets: dict[RationalTime, int] = {}
    has_initial = False
    for index, event in _records(ctx, "tempo_events", TempoEvent):
        base = f"/tempo_events/{index}"
        event_id = _entity_id(event, "tempo_event_id")
        onset = _rational(ctx, event.onset_qn, f"{base}/onset_qn", event_id)
        tempo_valid = _is_int(event.microseconds_per_quarter) and (
            event.microseconds_per_quarter > 0
        )
        if onset is not None and onset < zero:
            ctx.add(
                "TEMPO_INVALID",
                "tempo onset must be non-negative",
                f"{base}/onset_qn",
                event_id,
            )
        if not tempo_valid:
            ctx.add(
                "TEMPO_INVALID",
                "microseconds_per_quarter must be a positive integer",
                f"{base}/microseconds_per_quarter",
                event_id,
            )
        if onset is not None:
            has_initial |= onset == zero
            if onset in onsets:
                ctx.add(
                    "TEMPO_DUPLICATE_ONSET",
                    f"tempo onset duplicates /tempo_events/{onsets[onset]}",
                    f"{base}/onset_qn",
                    event_id,
                )
            else:
                onsets[onset] = index
        _check_optional_provenance_reference(
            ctx, event.provenance_id, f"{base}/provenance_id", event_id
        )
    if not has_initial:
        ctx.add(
            "TEMPO_INITIAL_MISSING",
            "a tempo event is required at 0/1",
            "/tempo_events",
            _entity_id(ctx.piece, "piece_id"),
        )


def _validate_meter_events(ctx: _Context) -> None:
    zero = RationalTime(0)
    onsets: dict[RationalTime, int] = {}
    has_initial = False
    bar_starts = {
        bar.start_qn
        for _, bar in _records(ctx, "bars", CanonicalBar)
        if isinstance(bar.start_qn, RationalTime)
    }
    has_bars = bool(_records(ctx, "bars", CanonicalBar))
    for index, event in _records(ctx, "meter_events", MeterEvent):
        base = f"/meter_events/{index}"
        event_id = _entity_id(event, "meter_event_id")
        onset = _rational(ctx, event.onset_qn, f"{base}/onset_qn", event_id)
        if onset is not None and onset < zero:
            ctx.add(
                "METER_INVALID",
                "meter onset must be non-negative",
                f"{base}/onset_qn",
                event_id,
            )
        if not _is_int(event.numerator) or event.numerator <= 0:
            ctx.add(
                "METER_INVALID",
                "meter numerator must be a positive integer",
                f"{base}/numerator",
                event_id,
            )
        if (
            not _is_int(event.denominator)
            or event.denominator <= 0
            or event.denominator & (event.denominator - 1)
        ):
            ctx.add(
                "METER_INVALID",
                "meter denominator must be a positive power of two",
                f"{base}/denominator",
                event_id,
            )
        if onset is not None:
            has_initial |= onset == zero
            if onset in onsets:
                ctx.add(
                    "METER_DUPLICATE_ONSET",
                    f"meter onset duplicates /meter_events/{onsets[onset]}",
                    f"{base}/onset_qn",
                    event_id,
                )
            else:
                onsets[onset] = index
            if has_bars and onset not in bar_starts:
                ctx.add(
                    "METER_NOT_AT_BAR_START",
                    "meter events must occur at canonical bar starts",
                    f"{base}/onset_qn",
                    event_id,
                )
        _check_optional_provenance_reference(
            ctx, event.provenance_id, f"{base}/provenance_id", event_id
        )
    if not has_initial:
        ctx.add(
            "METER_INITIAL_MISSING",
            "a meter event is required at 0/1",
            "/meter_events",
            _entity_id(ctx.piece, "piece_id"),
        )


def _validate_key_signatures(ctx: _Context) -> None:
    zero = RationalTime(0)
    seen_onsets: set[RationalTime] = set()
    for index, event in _records(ctx, "key_signature_events", KeySignatureEvent):
        base = f"/key_signature_events/{index}"
        event_id = _entity_id(event, "key_signature_event_id")
        onset = _rational(ctx, event.onset_qn, f"{base}/onset_qn", event_id)
        if onset is not None:
            if onset < zero:
                ctx.add(
                    "TIME_NEGATIVE",
                    "key-signature onset must be non-negative",
                    f"{base}/onset_qn",
                    event_id,
                )
            if onset in seen_onsets:
                ctx.add(
                    "FIELD_VALUE_INVALID",
                    "key-signature events may not share an onset",
                    f"{base}/onset_qn",
                    event_id,
                )
            seen_onsets.add(onset)
        if not _is_int(event.fifths) or not -7 <= event.fifths <= 7:
            ctx.add(
                "FIELD_VALUE_INVALID",
                "key-signature fifths must be an integer in [-7, 7]",
                f"{base}/fifths",
                event_id,
            )
        if not isinstance(event.mode, str) or event.mode not in _KEY_MODES:
            ctx.add(
                "FIELD_VALUE_INVALID",
                "mode is not a supported KeySignatureMode value",
                f"{base}/mode",
                event_id,
            )
        _check_optional_string(ctx, event.raw_value, f"{base}/raw_value", event_id)
        if event.mode == "other" and not _is_open_string(event.raw_value):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "mode='other' requires a non-empty trimmed raw_value",
                f"{base}/raw_value",
                event_id,
            )
        _check_optional_provenance_reference(
            ctx, event.provenance_id, f"{base}/provenance_id", event_id
        )


def _effective_meter(ctx: _Context, onset: RationalTime) -> MeterEvent | None:
    candidates = [
        meter
        for _, meter in _records(ctx, "meter_events", MeterEvent)
        if _valid_meter(meter) and meter.onset_qn <= onset
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda meter: meter.onset_qn)


def _nominal_bar_duration(meter: MeterEvent) -> RationalTime:
    return RationalTime(meter.numerator * 4, meter.denominator)


def _validate_bars(ctx: _Context) -> None:
    bars = _records(ctx, "bars", CanonicalBar)
    zero = RationalTime(0)
    previous_end: RationalTime | None = None
    piece_duration = (
        ctx.piece.duration_qn
        if isinstance(ctx.piece.duration_qn, RationalTime)
        else None
    )
    for ordinal, (index, bar) in enumerate(bars):
        base = f"/bars/{index}"
        bar_id = _entity_id(bar, "bar_id")
        start = _rational(ctx, bar.start_qn, f"{base}/start_qn", bar_id)
        duration = _rational(ctx, bar.duration_qn, f"{base}/duration_qn", bar_id)
        offset = _rational(
            ctx, bar.metric_offset_qn, f"{base}/metric_offset_qn", bar_id
        )
        if not _is_int(bar.index) or bar.index != ordinal:
            ctx.add(
                "BAR_INVALID",
                "bar indices must be chronological zero-based ordinals",
                f"{base}/index",
                bar_id,
            )
        if start is not None and start < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "bar start must be non-negative",
                f"{base}/start_qn",
                bar_id,
            )
        if duration is not None and duration <= zero:
            if duration < zero:
                ctx.add(
                    "DURATION_NEGATIVE",
                    "bar duration must be positive",
                    f"{base}/duration_qn",
                    bar_id,
                )
            ctx.add(
                "BAR_INVALID",
                "bar duration must be positive",
                f"{base}/duration_qn",
                bar_id,
            )
        if offset is not None and offset < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "bar metric offset must be non-negative",
                f"{base}/metric_offset_qn",
                bar_id,
            )
        pickup_valid = _check_bool(ctx, bar.is_pickup, f"{base}/is_pickup", bar_id)
        incomplete_valid = _check_bool(
            ctx, bar.is_incomplete, f"{base}/is_incomplete", bar_id
        )
        _check_optional_string(ctx, bar.display_number, f"{base}/display_number", bar_id)
        reference_valid = _check_reference(
            ctx, bar.meter_event_id, "meter", f"{base}/meter_event_id", bar_id
        )
        meter = (
            ctx.meter_by_id.get(bar.meter_event_id)
            if reference_valid and isinstance(bar.meter_event_id, str)
            else None
        )
        effective = _effective_meter(ctx, start) if start is not None else None
        if meter is not None and effective is not None and meter is not effective:
            ctx.add(
                "BAR_METER_MISMATCH",
                "bar must reference the meter effective at its start",
                f"{base}/meter_event_id",
                bar_id,
            )
        if meter is not None and _valid_meter(meter) and duration is not None and offset is not None:
            nominal = _nominal_bar_duration(meter)
            unit = RationalTime(4, meter.denominator)
            offset_units = offset.to_fraction() / unit.to_fraction()
            if offset_units.denominator != 1:
                ctx.add(
                    "BAR_INVALID",
                    "metric_offset_qn must align to the denominator-unit beat grid",
                    f"{base}/metric_offset_qn",
                    bar_id,
                )
            if offset >= nominal:
                ctx.add(
                    "BAR_INVALID",
                    "metric_offset_qn must be inside the nominal meter",
                    f"{base}/metric_offset_qn",
                    bar_id,
                )
            if pickup_valid and incomplete_valid:
                if bar.is_pickup:
                    if (
                        ordinal != 0
                        or start != zero
                        or not bar.is_incomplete
                        or offset <= zero
                        or duration >= nominal
                        or offset + duration != nominal
                    ):
                        ctx.add(
                            "BAR_INVALID",
                            "pickup bars must be the first shortened bar and fill the "
                            "nominal meter with metric_offset_qn",
                            base,
                            bar_id,
                        )
                elif bar.is_incomplete:
                    if (
                        ordinal != len(bars) - 1
                        or offset != zero
                        or duration >= nominal
                    ):
                        ctx.add(
                            "BAR_INVALID",
                            "a non-pickup incomplete bar must be a shortened final bar "
                            "with zero metric offset",
                            base,
                            bar_id,
                        )
                elif offset != zero:
                    ctx.add(
                        "BAR_INVALID",
                        "complete non-pickup bars require zero metric offset",
                        f"{base}/metric_offset_qn",
                        bar_id,
                    )
                if bar.is_incomplete:
                    if duration > nominal:
                        ctx.add(
                            "BAR_METER_MISMATCH",
                            "incomplete bar duration may not exceed nominal duration",
                            f"{base}/duration_qn",
                            bar_id,
                        )
                elif duration != nominal:
                    ctx.add(
                        "BAR_METER_MISMATCH",
                        "complete bar duration must equal nominal meter duration",
                        f"{base}/duration_qn",
                        bar_id,
                    )
        if start is not None and duration is not None:
            if ordinal == 0 and start != zero:
                ctx.add(
                    "BAR_COVERAGE_INVALID",
                    "bar coverage must begin at 0/1",
                    f"{base}/start_qn",
                    bar_id,
                )
            if previous_end is not None and start != previous_end:
                ctx.add(
                    "BAR_COVERAGE_INVALID",
                    "bars must form contiguous non-overlapping coverage",
                    f"{base}/start_qn",
                    bar_id,
                )
            previous_end = start + duration
        _check_optional_provenance_reference(
            ctx, bar.provenance_id, f"{base}/provenance_id", bar_id
        )
    if bars and previous_end is not None and piece_duration is not None:
        if previous_end != piece_duration:
            ctx.add(
                "BAR_COVERAGE_INVALID",
                "bar coverage must end at piece duration",
                "/bars",
                _entity_id(ctx.piece, "piece_id"),
            )


def _expected_beats(
    bar: CanonicalBar, meter: MeterEvent
) -> tuple[tuple[int, RationalTime, RationalTime, RationalTime, bool], ...] | None:
    if (
        not isinstance(bar.start_qn, RationalTime)
        or not isinstance(bar.duration_qn, RationalTime)
        or not isinstance(bar.metric_offset_qn, RationalTime)
        or bar.duration_qn <= RationalTime(0)
        or not _valid_meter(meter)
    ):
        return None
    unit = RationalTime(4, meter.denominator)
    quotient = bar.metric_offset_qn.to_fraction() / unit.to_fraction()
    if quotient.denominator != 1:
        return None
    position = bar.metric_offset_qn
    bar_metric_end = position + bar.duration_qn
    start = bar.start_qn
    index = quotient.numerator
    expected: list[tuple[int, RationalTime, RationalTime, RationalTime, bool]] = []
    while position < bar_metric_end:
        remaining = bar_metric_end - position
        duration = unit if unit <= remaining else remaining
        expected.append((index, start, duration, position, position == RationalTime(0)))
        start = start + duration
        position = position + unit
        index += 1
    return tuple(expected)


def _validate_beats(ctx: _Context) -> None:
    beats_by_bar: dict[str, list[tuple[int, CanonicalBeat]]] = {}
    zero = RationalTime(0)
    for index, beat in _records(ctx, "beats", CanonicalBeat):
        base = f"/beats/{index}"
        beat_id = _entity_id(beat, "beat_id")
        bar_ref = _check_reference(ctx, beat.bar_id, "bar", f"{base}/bar_id", beat_id)
        meter_ref = _check_reference(
            ctx, beat.meter_event_id, "meter", f"{base}/meter_event_id", beat_id
        )
        if isinstance(beat.bar_id, str):
            beats_by_bar.setdefault(beat.bar_id, []).append((index, beat))
        if not _is_int(beat.index_in_bar) or beat.index_in_bar < 0:
            ctx.add(
                "BEAT_INVALID",
                "index_in_bar must be a non-negative integer",
                f"{base}/index_in_bar",
                beat_id,
            )
        start = _rational(ctx, beat.start_qn, f"{base}/start_qn", beat_id)
        duration = _rational(ctx, beat.duration_qn, f"{base}/duration_qn", beat_id)
        position = _rational(
            ctx, beat.position_in_bar_qn, f"{base}/position_in_bar_qn", beat_id
        )
        if start is not None and start < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "beat start must be non-negative",
                f"{base}/start_qn",
                beat_id,
            )
        if duration is not None and duration <= zero:
            if duration < zero:
                ctx.add(
                    "DURATION_NEGATIVE",
                    "beat duration must be positive",
                    f"{base}/duration_qn",
                    beat_id,
                )
            ctx.add(
                "BEAT_INVALID",
                "beat duration must be positive",
                f"{base}/duration_qn",
                beat_id,
            )
        if position is not None and position < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "beat position must be non-negative",
                f"{base}/position_in_bar_qn",
                beat_id,
            )
        _check_bool(ctx, beat.is_downbeat, f"{base}/is_downbeat", beat_id)
        if beat.strength is not None:
            if not isinstance(beat.strength, (int, float)) or isinstance(
                beat.strength, bool
            ):
                ctx.add(
                    "BEAT_INVALID",
                    "strength must be a number in [0, 1] or None",
                    f"{base}/strength",
                    beat_id,
                )
            elif _check_finite(ctx, beat.strength, f"{base}/strength", beat_id) and not (
                0 <= beat.strength <= 1
            ):
                ctx.add(
                    "BEAT_INVALID",
                    "strength must be in [0, 1]",
                    f"{base}/strength",
                    beat_id,
                )
        bar = ctx.bar_by_id.get(beat.bar_id) if bar_ref else None
        if (
            bar is not None
            and meter_ref
            and beat.meter_event_id != bar.meter_event_id
        ):
            ctx.add(
                "BEAT_INVALID",
                "beat meter_event_id must match its bar",
                f"{base}/meter_event_id",
                beat_id,
            )
        if (
            bar is not None
            and start is not None
            and duration is not None
            and isinstance(bar.start_qn, RationalTime)
            and isinstance(bar.duration_qn, RationalTime)
            and (
                start < bar.start_qn
                or start + duration > bar.start_qn + bar.duration_qn
            )
        ):
            ctx.add(
                "BEAT_INVALID",
                "beat must lie inside its referenced bar",
                base,
                beat_id,
            )
        _check_optional_provenance_reference(
            ctx, beat.provenance_id, f"{base}/provenance_id", beat_id
        )

    for bar_index, bar in _records(ctx, "bars", CanonicalBar):
        bar_id = _entity_id(bar, "bar_id")
        if bar_id is None:
            continue
        meter = (
            ctx.meter_by_id.get(bar.meter_event_id)
            if isinstance(bar.meter_event_id, str)
            else None
        )
        expected = _expected_beats(bar, meter) if meter is not None else None
        actual = beats_by_bar.get(bar_id, [])
        if expected is None:
            continue
        if len(actual) != len(expected):
            ctx.add(
                "BEAT_GRID_INVALID",
                "beats must cover the actual bar extent on the denominator-unit grid",
                f"/bars/{bar_index}",
                bar_id,
            )
            continue
        for (beat_index, beat), expected_values in zip(actual, expected, strict=True):
            expected_index, start, duration, position, downbeat = expected_values
            if (
                beat.index_in_bar != expected_index
                or beat.start_qn != start
                or beat.duration_qn != duration
                or beat.position_in_bar_qn != position
                or beat.is_downbeat != downbeat
                or beat.meter_event_id != bar.meter_event_id
            ):
                ctx.add(
                    "BEAT_GRID_INVALID",
                    "beat does not match the denominator-unit grid",
                    f"/beats/{beat_index}",
                    _entity_id(beat, "beat_id"),
                )


def _validate_annotations(ctx: _Context) -> None:
    zero = RationalTime(0)
    for index, annotation in _records(ctx, "annotations", AnnotationSpan):
        base = f"/annotations/{index}"
        annotation_id = _entity_id(annotation, "annotation_id")
        _check_open_string(
            ctx,
            annotation.annotation_type,
            f"{base}/annotation_type",
            "annotation_type",
            annotation_id,
        )
        start = _rational(ctx, annotation.start_qn, f"{base}/start_qn", annotation_id)
        end = _rational(ctx, annotation.end_qn, f"{base}/end_qn", annotation_id)
        if start is not None and start < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "annotation start must be non-negative",
                f"{base}/start_qn",
                annotation_id,
            )
        if end is not None and end < zero:
            ctx.add(
                "TIME_NEGATIVE",
                "annotation end must be non-negative",
                f"{base}/end_qn",
                annotation_id,
            )
        if start is not None and end is not None and end < start:
            ctx.add(
                "ANNOTATION_INVALID",
                "annotation end must not precede start",
                base,
                annotation_id,
            )
        if annotation.track_id is not None:
            _check_reference(
                ctx, annotation.track_id, "track", f"{base}/track_id", annotation_id
            )
        _check_optional_string(ctx, annotation.value, f"{base}/value", annotation_id)
        if (
            not isinstance(annotation.layer, str)
            or annotation.layer not in _ANNOTATION_LAYERS
        ):
            ctx.add(
                "ANNOTATION_INVALID",
                "annotation layer is unsupported",
                f"{base}/layer",
                annotation_id,
            )
        elif annotation.layer == "observation":
            if not (
                isinstance(annotation.annotation_type, str)
                and annotation.annotation_type.startswith(_OBSERVATION_PREFIXES)
            ):
                ctx.add(
                    "ANNOTATION_INVALID",
                    "observation annotation_type must use an observable namespace",
                    f"{base}/annotation_type",
                    annotation_id,
                )
        elif annotation.value is not None:
            ctx.add(
                "ANNOTATION_INVALID",
                "target_alignment annotations require value=None",
                f"{base}/value",
                annotation_id,
            )
        _check_optional_provenance_reference(
            ctx, annotation.provenance_id, f"{base}/provenance_id", annotation_id
        )


def _target_sequences(
    ctx: _Context, target: TargetArray, base: str, target_id: str | None
) -> dict[str, Sequence[Any]]:
    result = {
        name: _as_sequence(ctx, getattr(target, name), f"{base}/{name}", target_id)
        for name in ("entity_ids", "values", "mask", "confidence", "source", "provenance")
    }
    lengths = {len(value) for value in result.values()}
    if len(lengths) > 1:
        ctx.add(
            "TARGET_LENGTH_MISMATCH",
            "aligned target fields must have identical lengths",
            base,
            target_id,
        )
    return result


def _validate_class_labels(
    ctx: _Context, target: TargetArray, base: str, target_id: str | None
) -> tuple[str, ...] | None:
    if target.class_labels is None:
        return None
    values = _as_sequence(ctx, target.class_labels, f"{base}/class_labels", target_id)
    labels: list[str] = []
    for index, label in enumerate(values):
        if not isinstance(label, str) or not label:
            ctx.add(
                "TARGET_VALUE_INVALID",
                "class labels must be non-empty strings",
                f"{base}/class_labels/{index}",
                target_id,
            )
        else:
            labels.append(label)
    if len(labels) != len(set(labels)):
        ctx.add(
            "TARGET_VALUE_INVALID",
            "class_labels may not contain duplicates",
            f"{base}/class_labels",
            target_id,
        )
    return tuple(labels)


def _validate_target_value(
    ctx: _Context,
    target: TargetArray,
    value: object,
    labels: tuple[str, ...] | None,
    path: str,
    target_id: str | None,
) -> None:
    if target.value_type == "categorical":
        if not isinstance(value, str) or (labels is not None and value not in labels):
            ctx.add(
                "TARGET_VALUE_INVALID",
                "categorical value must be a string in class_labels when supplied",
                path,
                target_id,
            )
    elif target.value_type == "scalar":
        if not _is_number(value):
            if isinstance(value, float) and not isfinite(value):
                ctx.add("VALUE_NOT_FINITE", "scalar target must be finite", path, target_id)
            ctx.add(
                "TARGET_VALUE_INVALID",
                "scalar value must be a finite int or float",
                path,
                target_id,
            )
    elif target.value_type == "multi_label":
        sequence = _as_sequence(ctx, value, path, target_id)
        if labels is None:
            return
        seen: set[str] = set()
        positions: list[int] = []
        valid = True
        for item in sequence:
            if not isinstance(item, str) or item not in labels or item in seen:
                valid = False
            else:
                seen.add(item)
                positions.append(labels.index(item))
        if not valid or positions != sorted(positions):
            ctx.add(
                "TARGET_VALUE_INVALID",
                "multi_label values must be unique and in canonical class-label order",
                path,
                target_id,
            )
    elif target.value_type == "distribution":
        sequence = _as_sequence(ctx, value, path, target_id)
        valid = labels is not None and len(sequence) == len(labels)
        total = 0.0
        for offset, probability in enumerate(sequence):
            probability_path = f"{path}/{offset}"
            if not _is_number(probability):
                if isinstance(probability, float) and not isfinite(probability):
                    ctx.add(
                        "VALUE_NOT_FINITE",
                        "distribution probability must be finite",
                        probability_path,
                        target_id,
                    )
                valid = False
            elif not 0 <= probability <= 1:
                valid = False
            else:
                total += float(probability)
        if abs(total - 1.0) > 1e-9:
            valid = False
        if not valid:
            ctx.add(
                "TARGET_VALUE_INVALID",
                "distribution must match class_labels and sum to 1 within 1e-9",
                path,
                target_id,
            )


def _validate_targets(ctx: _Context) -> None:
    views: dict[tuple[str, str | None], int] = {}
    piece_id = _entity_id(ctx.piece, "piece_id")
    for index, target in _records(ctx, "targets", TargetArray):
        base = f"/targets/{index}"
        target_id = _entity_id(target, "target_id")
        task_valid = _check_open_string(
            ctx, target.task, f"{base}/task", "task", target_id
        )
        view_valid = target.annotation_view_id is None or _is_open_string(
            target.annotation_view_id
        )
        if not view_valid:
            ctx.add(
                "TARGET_VIEW_INVALID",
                "annotation_view_id must be None or a non-empty trimmed string "
                "without ASCII controls",
                f"{base}/annotation_view_id",
                target_id,
            )
        if task_valid and view_valid:
            key = (target.task, target.annotation_view_id)
            if key in views:
                ctx.add(
                    "TARGET_VIEW_DUPLICATE",
                    f"target view duplicates /targets/{views[key]}",
                    base,
                    target_id,
                )
            else:
                views[key] = index
        sequences = _target_sequences(ctx, target, base, target_id)
        entity_ids = sequences["entity_ids"]
        comparable_ids = [value for value in entity_ids if isinstance(value, str)]
        if len(comparable_ids) != len(set(comparable_ids)):
            ctx.add(
                "TARGET_ENTITY_DUPLICATE",
                "entity_ids may not repeat within one target array",
                f"{base}/entity_ids",
                target_id,
            )
        alignment_valid = target.alignment_type == "piece" or (
            isinstance(target.alignment_type, str)
            and target.alignment_type in _ALIGNMENT_PREFIXES
        )
        if not alignment_valid:
            ctx.add(
                "TARGET_ALIGNMENT_INVALID",
                "alignment_type is unsupported",
                f"{base}/alignment_type",
                target_id,
            )
        elif target.alignment_type == "piece":
            if len(entity_ids) != 1 or entity_ids[0] != piece_id:
                ctx.add(
                    "TARGET_ALIGNMENT_INVALID",
                    "piece alignment must contain exactly the containing piece_id",
                    f"{base}/entity_ids",
                    target_id,
                )
        else:
            expected_prefix = _ALIGNMENT_PREFIXES[target.alignment_type]
            for entity_index, entity_value in enumerate(entity_ids):
                entity_path = f"{base}/entity_ids/{entity_index}"
                parts = _split_id(entity_value)
                if parts is None or parts[0] != expected_prefix:
                    ctx.add(
                        "TARGET_ALIGNMENT_INVALID",
                        f"alignment requires '{expected_prefix}:' entity IDs",
                        entity_path,
                        target_id,
                    )
                    continue
                assert isinstance(entity_value, str)
                if ctx.entity_prefixes.get(entity_value) != expected_prefix:
                    ctx.add(
                        "TARGET_ENTITY_INVALID",
                        "target entity does not exist in the containing piece",
                        entity_path,
                        target_id,
                    )
                elif target.alignment_type == "annotation_span":
                    annotation = ctx.annotation_by_id.get(entity_value)
                    if (
                        annotation is None
                        or annotation.layer != "target_alignment"
                        or annotation.annotation_type != target.task
                    ):
                        ctx.add(
                            "TARGET_ALIGNMENT_INVALID",
                            "annotation_span targets require matching target_alignment "
                            "spans for the same task",
                            entity_path,
                            target_id,
                        )
        value_type_valid = (
            isinstance(target.value_type, str)
            and target.value_type in _TARGET_VALUE_TYPES
        )
        if not value_type_valid:
            ctx.add(
                "TARGET_VALUE_INVALID",
                "value_type is unsupported",
                f"{base}/value_type",
                target_id,
            )
        labels = _validate_class_labels(ctx, target, base, target_id)
        if target.value_type == "scalar" and target.class_labels is not None:
            ctx.add(
                "TARGET_VALUE_INVALID",
                "scalar targets require class_labels=None",
                f"{base}/class_labels",
                target_id,
            )
        if (
            value_type_valid
            and target.value_type in {"multi_label", "distribution"}
            and labels is None
        ):
            ctx.add(
                "TARGET_VALUE_INVALID",
                f"{target.value_type} targets require class_labels",
                f"{base}/class_labels",
                target_id,
            )
        aligned_length = min((len(value) for value in sequences.values()), default=0)
        for entry_index in range(aligned_length):
            mask = sequences["mask"][entry_index]
            value = sequences["values"][entry_index]
            confidence = sequences["confidence"][entry_index]
            source = sequences["source"][entry_index]
            provenance = sequences["provenance"][entry_index]
            if not isinstance(mask, bool):
                ctx.add(
                    "TARGET_MASK_INVALID",
                    "mask entries must be bool",
                    f"{base}/mask/{entry_index}",
                    target_id,
                )
                continue
            if not mask:
                if value is not None:
                    ctx.add(
                        "TARGET_MASK_INVALID",
                        "unavailable target values must be None",
                        f"{base}/values/{entry_index}",
                        target_id,
                    )
                if confidence is not None:
                    ctx.add(
                        "TARGET_CONFIDENCE_INVALID",
                        "unavailable target confidence must be None",
                        f"{base}/confidence/{entry_index}",
                        target_id,
                    )
                if source is not None:
                    ctx.add(
                        "TARGET_SOURCE_INVALID",
                        "unavailable target source must be None",
                        f"{base}/source/{entry_index}",
                        target_id,
                    )
                if provenance is not None:
                    ctx.add(
                        "TARGET_PROVENANCE_INVALID",
                        "unavailable target provenance must be None",
                        f"{base}/provenance/{entry_index}",
                        target_id,
                    )
                continue
            if value is None:
                ctx.add(
                    "TARGET_MASK_INVALID",
                    "available target values must be non-null",
                    f"{base}/values/{entry_index}",
                    target_id,
                )
            elif value_type_valid:
                _validate_target_value(
                    ctx,
                    target,
                    value,
                    labels,
                    f"{base}/values/{entry_index}",
                    target_id,
                )
            if confidence is not None:
                if not _is_number(confidence):
                    if isinstance(confidence, float) and not isfinite(confidence):
                        ctx.add(
                            "VALUE_NOT_FINITE",
                            "target confidence must be finite",
                            f"{base}/confidence/{entry_index}",
                            target_id,
                        )
                    ctx.add(
                        "TARGET_CONFIDENCE_INVALID",
                        "available confidence must be None or a finite number in [0, 1]",
                        f"{base}/confidence/{entry_index}",
                        target_id,
                    )
                elif not 0 <= confidence <= 1:
                    ctx.add(
                        "TARGET_CONFIDENCE_INVALID",
                        "available confidence must be in [0, 1]",
                        f"{base}/confidence/{entry_index}",
                        target_id,
                    )
            if not isinstance(source, str) or source not in _TARGET_SOURCES:
                ctx.add(
                    "TARGET_SOURCE_INVALID",
                    "available target source is missing or unsupported",
                    f"{base}/source/{entry_index}",
                    target_id,
                )
            if not isinstance(provenance, str) or provenance not in ctx.provenance_by_id:
                ctx.add(
                    "TARGET_PROVENANCE_INVALID",
                    "available target provenance must reference an existing record",
                    f"{base}/provenance/{entry_index}",
                    target_id,
                )


def _valid_rfc3339(value: str) -> bool:
    if not _RFC3339_RE.fullmatch(value):
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_provenance(ctx: _Context) -> None:
    records = _records(ctx, "provenance", ProvenanceRecord)
    if not records:
        ctx.add(
            "PROVENANCE_MISSING",
            "at least one provenance record is required",
            "/provenance",
            _entity_id(ctx.piece, "piece_id"),
        )
        return
    positions = {
        record.provenance_id: ordinal
        for ordinal, (_, record) in enumerate(records)
        if isinstance(record.provenance_id, str)
    }
    graph: dict[str, tuple[str, ...]] = {}
    for ordinal, (index, record) in enumerate(records):
        base = f"/provenance/{index}"
        provenance_id = _entity_id(record, "provenance_id")
        if not isinstance(record.kind, str) or record.kind not in _PROVENANCE_KINDS:
            ctx.add(
                "FIELD_VALUE_INVALID",
                "kind is not a supported ProvenanceKind value",
                f"{base}/kind",
                provenance_id,
            )
        _check_open_string(
            ctx, record.source, f"{base}/source", "provenance source", provenance_id
        )
        for field in ("record_id", "uri", "version"):
            _check_optional_string(
                ctx, getattr(record, field), f"{base}/{field}", provenance_id
            )
        if record.checksum_sha256 is not None and (
            not isinstance(record.checksum_sha256, str)
            or not _SHA256_RE.fullmatch(record.checksum_sha256)
        ):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "checksum_sha256 must be 64 lowercase hexadecimal characters",
                f"{base}/checksum_sha256",
                provenance_id,
            )
        if record.created_at is not None and (
            not isinstance(record.created_at, str)
            or not _valid_rfc3339(record.created_at)
        ):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "created_at must be a valid RFC 3339 timestamp with an offset",
                f"{base}/created_at",
                provenance_id,
            )
        parents = _as_sequence(ctx, record.parents, f"{base}/parents", provenance_id)
        valid_parents = tuple(parent for parent in parents if isinstance(parent, str))
        if len(valid_parents) != len(set(valid_parents)):
            ctx.add(
                "PROVENANCE_PARENT_INVALID",
                "provenance parents may not contain duplicates",
                f"{base}/parents",
                provenance_id,
            )
        for parent_index, parent in enumerate(parents):
            path = f"{base}/parents/{parent_index}"
            if not isinstance(parent, str) or parent not in positions:
                ctx.add(
                    "PROVENANCE_PARENT_INVALID",
                    "provenance parent does not exist",
                    path,
                    provenance_id,
                )
            elif parent == provenance_id:
                ctx.add(
                    "PROVENANCE_PARENT_INVALID",
                    "provenance record may not parent itself",
                    path,
                    provenance_id,
                )
            elif positions[parent] >= ordinal:
                ctx.add(
                    "PROVENANCE_PARENT_INVALID",
                    "provenance parents must precede children",
                    path,
                    provenance_id,
                )
        if provenance_id is not None:
            graph[provenance_id] = valid_parents
        details = _as_sequence(ctx, record.details, f"{base}/details", provenance_id)
        keys: list[str] = []
        for detail_index, detail in enumerate(details):
            detail_path = f"{base}/details/{detail_index}"
            if (
                not isinstance(detail, tuple)
                or len(detail) != 2
                or not isinstance(detail[0], str)
                or not detail[0]
            ):
                ctx.add(
                    "PROVENANCE_DETAIL_INVALID",
                    "details must contain non-empty string-key pairs",
                    detail_path,
                    provenance_id,
                )
                continue
            key, value = detail
            keys.append(key)
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                ctx.add(
                    "PROVENANCE_DETAIL_INVALID",
                    "provenance detail values must be JSON scalars",
                    f"{detail_path}/1",
                    provenance_id,
                )
            elif isinstance(value, float) and not isfinite(value):
                ctx.add(
                    "VALUE_NOT_FINITE",
                    "provenance detail value must be finite",
                    f"{detail_path}/1",
                    provenance_id,
                )
                ctx.add(
                    "PROVENANCE_DETAIL_INVALID",
                    "provenance detail values must be finite JSON scalars",
                    f"{detail_path}/1",
                    provenance_id,
                )
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            ctx.add(
                "PROVENANCE_DETAIL_INVALID",
                "detail keys must be unique and lexicographically sorted",
                f"{base}/details",
                provenance_id,
            )

    remaining = set(graph)
    while remaining:
        ready = {
            node
            for node in remaining
            if all(parent not in remaining for parent in graph.get(node, ()))
        }
        if not ready:
            break
        remaining.difference_update(ready)
    if remaining:
        ctx.add(
            "PROVENANCE_CYCLE",
            "provenance parent links contain a cycle",
            "/provenance",
            min(remaining),
        )


def _validate_quality_flags(ctx: _Context) -> None:
    for index, flag in _records(ctx, "quality_flags", QualityFlag):
        base = f"/quality_flags/{index}"
        if not isinstance(flag.code, str) or not _QUALITY_CODE_RE.fullmatch(flag.code):
            ctx.add(
                "QUALITY_FLAG_CODE_INVALID",
                "quality flag code must be a lowercase dotted namespace",
                f"{base}/code",
            )
        if (
            not isinstance(flag.severity, str)
            or flag.severity not in _QUALITY_SEVERITIES
        ):
            ctx.add(
                "FIELD_VALUE_INVALID",
                "quality flag severity must be 'info' or 'warning'",
                f"{base}/severity",
            )
        _check_open_string(ctx, flag.message, f"{base}/message", "quality flag message")
        for entity_index, entity_id in enumerate(
            _as_sequence(ctx, flag.entity_ids, f"{base}/entity_ids")
        ):
            _check_any_entity_reference(
                ctx,
                entity_id,
                f"{base}/entity_ids/{entity_index}",
            )
        _check_optional_provenance_reference(
            ctx, flag.provenance_id, f"{base}/provenance_id", None
        )


def _validate_piece_bounds(ctx: _Context) -> None:
    duration = (
        ctx.piece.duration_qn
        if isinstance(ctx.piece.duration_qn, RationalTime)
        else None
    )
    if duration is None:
        return
    endpoints: list[tuple[str, RationalTime, str | None]] = []
    for index, note in _records(ctx, "notes", CanonicalNote):
        if isinstance(note.onset_qn, RationalTime) and isinstance(
            note.duration_qn, RationalTime
        ):
            endpoints.append(
                (
                    f"/notes/{index}/duration_qn",
                    note.onset_qn + note.duration_qn,
                    _entity_id(note, "note_id"),
                )
            )
    for collection, record_type, start_field, duration_field, id_field in (
        ("bars", CanonicalBar, "start_qn", "duration_qn", "bar_id"),
        ("beats", CanonicalBeat, "start_qn", "duration_qn", "beat_id"),
    ):
        for index, record in _records(ctx, collection, record_type):
            start = getattr(record, start_field)
            length = getattr(record, duration_field)
            if isinstance(start, RationalTime) and isinstance(length, RationalTime):
                endpoints.append(
                    (
                        f"/{collection}/{index}/{duration_field}",
                        start + length,
                        _entity_id(record, id_field),
                    )
                )
    for collection, record_type, id_field in (
        ("tempo_events", TempoEvent, "tempo_event_id"),
        ("meter_events", MeterEvent, "meter_event_id"),
        ("key_signature_events", KeySignatureEvent, "key_signature_event_id"),
    ):
        for index, record in _records(ctx, collection, record_type):
            if isinstance(record.onset_qn, RationalTime):
                endpoints.append(
                    (
                        f"/{collection}/{index}/onset_qn",
                        record.onset_qn,
                        _entity_id(record, id_field),
                    )
                )
    for index, annotation in _records(ctx, "annotations", AnnotationSpan):
        if isinstance(annotation.end_qn, RationalTime):
            endpoints.append(
                (
                    f"/annotations/{index}/end_qn",
                    annotation.end_qn,
                    _entity_id(annotation, "annotation_id"),
                )
            )
    for path, endpoint, entity_id in endpoints:
        if endpoint > duration:
            ctx.add(
                "PIECE_DURATION_TOO_SHORT",
                "piece duration is before contained content or event extent",
                path,
                entity_id,
            )


def _safe_track_order(ctx: _Context) -> dict[str, int]:
    return {
        track.track_id: position
        for position, (_, track) in enumerate(_records(ctx, "tracks", CanonicalTrack))
        if isinstance(track.track_id, str)
    }


def _check_sorted(
    ctx: _Context,
    collection_name: str,
    values: list[Any],
    keys: list[Any],
) -> None:
    try:
        sorted_values = [
            value
            for _, value in sorted(
                zip(keys, values, strict=True),
                key=lambda pair: pair[0],
            )
        ]
    except (TypeError, ValueError):
        return
    if values != sorted_values:
        ctx.add(
            "COLLECTION_ORDER_INVALID",
            f"{collection_name} is not in canonical order",
            f"/{collection_name}",
            _entity_id(ctx.piece, "piece_id"),
        )


def _validate_collection_order(ctx: _Context) -> None:
    tracks = [record for _, record in _records(ctx, "tracks", CanonicalTrack)]
    if all(
        (track.source_track_index is None or _is_int(track.source_track_index))
        and isinstance(track.track_id, str)
        for track in tracks
    ):
        _check_sorted(
            ctx,
            "tracks",
            tracks,
            [
                (
                    track.source_track_index is None,
                    track.source_track_index if track.source_track_index is not None else 0,
                    track.track_id,
                )
                for track in tracks
            ],
        )
    track_order = _safe_track_order(ctx)
    notes = [record for _, record in _records(ctx, "notes", CanonicalNote)]
    if all(
        isinstance(note.onset_qn, RationalTime)
        and isinstance(note.duration_qn, RationalTime)
        and isinstance(note.track_id, str)
        and note.track_id in track_order
        and _is_int(note.pitch)
        and isinstance(note.note_id, str)
        for note in notes
    ):
        _check_sorted(
            ctx,
            "notes",
            notes,
            [
                (
                    note.onset_qn,
                    track_order[note.track_id],
                    note.pitch,
                    note.duration_qn,
                    note.note_id,
                )
                for note in notes
            ],
        )
    collection_keys: tuple[tuple[str, type[Any], Any], ...] = (
        (
            "bars",
            CanonicalBar,
            lambda value: (value.start_qn, value.index, value.bar_id),
        ),
        (
            "beats",
            CanonicalBeat,
            lambda value: (
                value.start_qn,
                value.bar_id,
                value.index_in_bar,
                value.beat_id,
            ),
        ),
        (
            "tempo_events",
            TempoEvent,
            lambda value: (value.onset_qn, value.tempo_event_id),
        ),
        (
            "meter_events",
            MeterEvent,
            lambda value: (value.onset_qn, value.meter_event_id),
        ),
        (
            "key_signature_events",
            KeySignatureEvent,
            lambda value: (value.onset_qn, value.key_signature_event_id),
        ),
        (
            "annotations",
            AnnotationSpan,
            lambda value: (value.start_qn, value.end_qn, value.annotation_id),
        ),
        (
            "targets",
            TargetArray,
            lambda value: (
                value.task,
                value.annotation_view_id is not None,
                value.annotation_view_id or "",
                value.target_id,
            ),
        ),
        (
            "quality_flags",
            QualityFlag,
            lambda value: (value.code, value.entity_ids, value.message),
        ),
    )
    for name, record_type, key_function in collection_keys:
        values = [record for _, record in _records(ctx, name, record_type)]
        try:
            keys = [key_function(value) for value in values]
        except (AttributeError, TypeError):
            continue
        _check_sorted(ctx, name, values, keys)

    provenance = [
        record for _, record in _records(ctx, "provenance", ProvenanceRecord)
    ]
    if provenance and all(
        isinstance(record.provenance_id, str)
        and isinstance(record.parents, tuple)
        and all(isinstance(parent, str) for parent in record.parents)
        for record in provenance
    ):
        by_id = {record.provenance_id: record for record in provenance}
        remaining = set(by_id)
        emitted: set[str] = set()
        expected_ids: list[str] = []
        while remaining:
            ready = sorted(
                identifier
                for identifier in remaining
                if all(parent in emitted or parent not in by_id for parent in by_id[identifier].parents)
            )
            if not ready:
                break
            expected_ids.extend(ready)
            emitted.update(ready)
            remaining.difference_update(ready)
        actual_ids = [record.provenance_id for record in provenance]
        if not remaining and actual_ids != expected_ids:
            ctx.add(
                "COLLECTION_ORDER_INVALID",
                "provenance is not in canonical topological order",
                "/provenance",
                _entity_id(ctx.piece, "piece_id"),
            )


def _collect_warnings(ctx: _Context) -> None:
    notes = _records(ctx, "notes", CanonicalNote)
    tracks = _records(ctx, "tracks", CanonicalTrack)
    if not notes:
        ctx.add(
            "EMPTY_PIECE",
            "piece contains no notes",
            "/notes",
            _entity_id(ctx.piece, "piece_id"),
            severity="warning",
        )
    note_track_ids = {
        note.track_id for _, note in notes if isinstance(note.track_id, str)
    }
    for index, track in tracks:
        if isinstance(track.track_id, str) and track.track_id not in note_track_ids:
            ctx.add(
                "EMPTY_TRACK",
                "canonical track contains no notes",
                f"/tracks/{index}",
                _entity_id(track, "track_id"),
                severity="warning",
            )
    bars = _records(ctx, "bars", CanonicalBar)
    if bars:
        index, final_bar = bars[-1]
        meter = (
            ctx.meter_by_id.get(final_bar.meter_event_id)
            if isinstance(final_bar.meter_event_id, str)
            else None
        )
        if (
            isinstance(final_bar.is_incomplete, bool)
            and final_bar.is_incomplete
            and isinstance(final_bar.is_pickup, bool)
            and not final_bar.is_pickup
            and meter is not None
            and _valid_meter(meter)
            and isinstance(final_bar.duration_qn, RationalTime)
            and final_bar.duration_qn < _nominal_bar_duration(meter)
        ):
            ctx.add(
                "INCOMPLETE_FINAL_BAR",
                "piece ends with a shortened incomplete final bar",
                f"/bars/{index}",
                _entity_id(final_bar, "bar_id"),
                severity="warning",
            )
    positive_notes = [
        (index, note)
        for index, note in notes
        if isinstance(note.onset_qn, RationalTime)
        and isinstance(note.duration_qn, RationalTime)
        and note.duration_qn > RationalTime(0)
        and isinstance(note.track_id, str)
        and _is_int(note.pitch)
    ]
    for left_position, (left_index, left) in enumerate(positive_notes):
        left_end = left.onset_qn + left.duration_qn
        for right_index, right in positive_notes[left_position + 1 :]:
            if (
                left.track_id == right.track_id
                and left.pitch == right.pitch
                and right.onset_qn < left_end
                and left.onset_qn < right.onset_qn + right.duration_qn
            ):
                ctx.add(
                    "OVERLAPPING_SAME_PITCH_NOTES",
                    "positive-duration notes of the same pitch overlap on one track",
                    f"/notes/{right_index}",
                    _entity_id(right, "note_id"),
                    severity="warning",
                )
    for event_index, event in _records(ctx, "tempo_events", TempoEvent):
        if not isinstance(event.onset_qn, RationalTime):
            continue
        for _, bar in bars:
            if (
                isinstance(bar.start_qn, RationalTime)
                and isinstance(bar.duration_qn, RationalTime)
                and bar.start_qn < event.onset_qn < bar.start_qn + bar.duration_qn
            ):
                ctx.add(
                    "MID_BAR_TEMPO_CHANGE",
                    "tempo event occurs strictly inside a bar",
                    f"/tempo_events/{event_index}/onset_qn",
                    _entity_id(event, "tempo_event_id"),
                    severity="warning",
                )
                break
    for target_index, target in _records(ctx, "targets", TargetArray):
        masks = target.mask if isinstance(target.mask, tuple) else ()
        confidence = target.confidence if isinstance(target.confidence, tuple) else ()
        for entry_index, (available, value) in enumerate(zip(masks, confidence)):
            if (
                available is True
                and _is_number(value)
                and value is not None
                and value < 0.5
            ):
                ctx.add(
                    "LOW_CONFIDENCE_TARGET",
                    "available target confidence is below 0.5",
                    f"/targets/{target_index}/confidence/{entry_index}",
                    _entity_id(target, "target_id"),
                    severity="warning",
                )
    referenced_provenance: set[str] = set()
    for collection, record_type in (
        ("tracks", CanonicalTrack),
        ("notes", CanonicalNote),
        ("bars", CanonicalBar),
        ("beats", CanonicalBeat),
        ("tempo_events", TempoEvent),
        ("meter_events", MeterEvent),
        ("key_signature_events", KeySignatureEvent),
        ("annotations", AnnotationSpan),
        ("quality_flags", QualityFlag),
    ):
        for _, record in _records(ctx, collection, record_type):
            value = getattr(record, "provenance_id", None)
            if isinstance(value, str):
                referenced_provenance.add(value)
    for _, target in _records(ctx, "targets", TargetArray):
        if isinstance(target.provenance, tuple):
            referenced_provenance.update(
                value for value in target.provenance if isinstance(value, str)
            )
    for _, record in _records(ctx, "provenance", ProvenanceRecord):
        if isinstance(record.parents, tuple):
            referenced_provenance.update(
                parent for parent in record.parents if isinstance(parent, str)
            )
    for index, record in _records(ctx, "provenance", ProvenanceRecord):
        if (
            isinstance(record.provenance_id, str)
            and record.provenance_id not in referenced_provenance
        ):
            ctx.add(
                "UNREFERENCED_PROVENANCE",
                "provenance record is not referenced",
                f"/provenance/{index}",
                _entity_id(record, "provenance_id"),
                severity="warning",
            )
    for index, annotation in _records(ctx, "annotations", AnnotationSpan):
        if annotation.layer == "observation" and annotation.value == "":
            ctx.add(
                "EMPTY_OBSERVATION",
                "observation annotation contains an available empty string",
                f"/annotations/{index}/value",
                _entity_id(annotation, "annotation_id"),
                severity="warning",
            )
    content_end = RationalTime(0)
    for _, note in positive_notes:
        content_end = max(content_end, note.onset_qn + note.duration_qn)
    for _, annotation in _records(ctx, "annotations", AnnotationSpan):
        if (
            annotation.layer == "observation"
            and isinstance(annotation.start_qn, RationalTime)
            and isinstance(annotation.end_qn, RationalTime)
            and annotation.end_qn > annotation.start_qn
        ):
            content_end = max(content_end, annotation.end_qn)
    if (
        isinstance(ctx.piece.duration_qn, RationalTime)
        and ctx.piece.duration_qn > content_end
    ):
        ctx.add(
            "PIECE_TRAILING_SILENCE",
            "piece duration extends beyond sounding or observation content",
            "/duration_qn",
            _entity_id(ctx.piece, "piece_id"),
            severity="warning",
        )


def validate_piece(piece: CanonicalPiece) -> ValidationReport:
    """Return every detectable validation issue without mutating ``piece``."""

    ctx = _Context(
        piece=piece,
        issues=[],
        entity_paths={},
        entity_prefixes={},
        track_by_id={},
        bar_by_id={},
        beat_by_id={},
        meter_by_id={},
        provenance_by_id={},
        annotation_by_id={},
    )
    _validate_ids(ctx)
    _validate_top_level(ctx)
    _validate_metadata(ctx)
    _validate_tracks(ctx)
    _validate_notes(ctx)
    _validate_tempo_events(ctx)
    _validate_meter_events(ctx)
    _validate_key_signatures(ctx)
    _validate_bars(ctx)
    _validate_beats(ctx)
    _validate_annotations(ctx)
    _validate_targets(ctx)
    _validate_provenance(ctx)
    _validate_quality_flags(ctx)
    _validate_piece_bounds(ctx)
    _validate_collection_order(ctx)
    _collect_warnings(ctx)
    issues = tuple(
        sorted(
            ctx.issues,
            key=lambda issue: (
                issue.path,
                issue.severity,
                issue.code,
                issue.entity_id or "",
                issue.message,
            ),
        )
    )
    return ValidationReport(issues=issues)


def validate_or_raise(piece: CanonicalPiece) -> None:
    """Raise with the complete report only when validation errors exist."""

    report = validate_piece(piece)
    if not report.is_valid:
        raise CanonicalValidationError(report)
