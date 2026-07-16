"""Strict deterministic JSON serialization for canonical pieces."""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
import json
from os import PathLike
from typing import Any, Mapping, TypeAlias

from music_critic.data.schema import (
    SCHEMA_VERSION,
    AnnotationSpan,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    KeySignatureEvent,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    QualityFlag,
    TargetArray,
    TempoEvent,
)
from music_critic.data.timing import RationalTime
from music_critic.data.validation import (
    CanonicalValidationError,
    ValidationCode,
    ValidationIssue,
    ValidationReport,
    validate_or_raise,
)


JsonObject = dict[str, Any]

__all__ = [
    "JsonObject",
    "piece_to_dict",
    "piece_from_dict",
    "dumps_piece",
    "loads_piece",
    "dump_piece",
    "load_piece",
]


def _encode_rational(value: RationalTime) -> JsonObject:
    return {"num": value.num, "den": value.den}


def _encode_metadata(value: PieceMetadata) -> JsonObject:
    return {
        "source_format": value.source_format,
        "title": value.title,
        "creators": None if value.creators is None else list(value.creators),
        "collection": value.collection,
        "movement_title": value.movement_title,
        "movement_number": value.movement_number,
        "genres": None if value.genres is None else list(value.genres),
        "copyright": value.copyright,
        "language": value.language,
    }


def _encode_track(value: CanonicalTrack) -> JsonObject:
    return {
        "track_id": value.track_id,
        "source_track_index": value.source_track_index,
        "name": value.name,
        "instrument_name": value.instrument_name,
        "program": value.program,
        "channel": value.channel,
        "is_percussion": value.is_percussion,
        "provenance_id": value.provenance_id,
    }


def _encode_note(value: CanonicalNote) -> JsonObject:
    return {
        "note_id": value.note_id,
        "track_id": value.track_id,
        "pitch": value.pitch,
        "onset_qn": _encode_rational(value.onset_qn),
        "duration_qn": _encode_rational(value.duration_qn),
        "velocity": value.velocity,
        "channel": value.channel,
        "program": value.program,
        "is_percussion": value.is_percussion,
        "is_grace": value.is_grace,
        "spelling_step": value.spelling_step,
        "spelling_alter": value.spelling_alter,
        "staff": value.staff,
        "voice": value.voice,
        "articulations": (
            None if value.articulations is None else list(value.articulations)
        ),
        "dynamic": value.dynamic,
        "source_onset_ticks": value.source_onset_ticks,
        "source_duration_ticks": value.source_duration_ticks,
        "source_onset_seconds": value.source_onset_seconds,
        "source_duration_seconds": value.source_duration_seconds,
        "provenance_id": value.provenance_id,
    }


def _encode_bar(value: CanonicalBar) -> JsonObject:
    return {
        "bar_id": value.bar_id,
        "index": value.index,
        "start_qn": _encode_rational(value.start_qn),
        "duration_qn": _encode_rational(value.duration_qn),
        "meter_event_id": value.meter_event_id,
        "metric_offset_qn": _encode_rational(value.metric_offset_qn),
        "is_pickup": value.is_pickup,
        "is_incomplete": value.is_incomplete,
        "display_number": value.display_number,
        "provenance_id": value.provenance_id,
    }


def _encode_beat(value: CanonicalBeat) -> JsonObject:
    return {
        "beat_id": value.beat_id,
        "bar_id": value.bar_id,
        "meter_event_id": value.meter_event_id,
        "index_in_bar": value.index_in_bar,
        "start_qn": _encode_rational(value.start_qn),
        "duration_qn": _encode_rational(value.duration_qn),
        "position_in_bar_qn": _encode_rational(value.position_in_bar_qn),
        "is_downbeat": value.is_downbeat,
        "strength": value.strength,
        "provenance_id": value.provenance_id,
    }


def _encode_tempo_event(value: TempoEvent) -> JsonObject:
    return {
        "tempo_event_id": value.tempo_event_id,
        "onset_qn": _encode_rational(value.onset_qn),
        "microseconds_per_quarter": value.microseconds_per_quarter,
        "provenance_id": value.provenance_id,
    }


def _encode_meter_event(value: MeterEvent) -> JsonObject:
    return {
        "meter_event_id": value.meter_event_id,
        "onset_qn": _encode_rational(value.onset_qn),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "provenance_id": value.provenance_id,
    }


def _encode_key_signature_event(value: KeySignatureEvent) -> JsonObject:
    return {
        "key_signature_event_id": value.key_signature_event_id,
        "onset_qn": _encode_rational(value.onset_qn),
        "fifths": value.fifths,
        "mode": value.mode,
        "raw_value": value.raw_value,
        "provenance_id": value.provenance_id,
    }


def _encode_annotation(value: AnnotationSpan) -> JsonObject:
    return {
        "annotation_id": value.annotation_id,
        "annotation_type": value.annotation_type,
        "layer": value.layer,
        "start_qn": _encode_rational(value.start_qn),
        "end_qn": _encode_rational(value.end_qn),
        "track_id": value.track_id,
        "value": value.value,
        "provenance_id": value.provenance_id,
    }


def _encode_target_value(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value


def _encode_target(value: TargetArray) -> JsonObject:
    return {
        "target_id": value.target_id,
        "task": value.task,
        "annotation_view_id": value.annotation_view_id,
        "alignment_type": value.alignment_type,
        "entity_ids": list(value.entity_ids),
        "value_type": value.value_type,
        "class_labels": (
            None if value.class_labels is None else list(value.class_labels)
        ),
        "values": [_encode_target_value(item) for item in value.values],
        "mask": list(value.mask),
        "confidence": list(value.confidence),
        "source": list(value.source),
        "provenance": list(value.provenance),
    }


def _encode_provenance(value: ProvenanceRecord) -> JsonObject:
    return {
        "provenance_id": value.provenance_id,
        "kind": value.kind,
        "source": value.source,
        "record_id": value.record_id,
        "uri": value.uri,
        "version": value.version,
        "checksum_sha256": value.checksum_sha256,
        "created_at": value.created_at,
        "parents": list(value.parents),
        "details": {key: detail_value for key, detail_value in value.details},
    }


def _encode_quality_flag(value: QualityFlag) -> JsonObject:
    return {
        "code": value.code,
        "severity": value.severity,
        "message": value.message,
        "entity_ids": list(value.entity_ids),
        "provenance_id": value.provenance_id,
    }


def _encode_piece(value: CanonicalPiece) -> JsonObject:
    return {
        "schema_version": SCHEMA_VERSION,
        "piece_id": value.piece_id,
        "dataset_name": value.dataset_name,
        "source_group_id": value.source_group_id,
        "split": value.split,
        "source_path": value.source_path,
        "source_resolution": value.source_resolution,
        "duration_qn": _encode_rational(value.duration_qn),
        "metadata": _encode_metadata(value.metadata),
        "tracks": [_encode_track(item) for item in value.tracks],
        "notes": [_encode_note(item) for item in value.notes],
        "bars": [_encode_bar(item) for item in value.bars],
        "beats": [_encode_beat(item) for item in value.beats],
        "tempo_events": [_encode_tempo_event(item) for item in value.tempo_events],
        "meter_events": [_encode_meter_event(item) for item in value.meter_events],
        "key_signature_events": [
            _encode_key_signature_event(item)
            for item in value.key_signature_events
        ],
        "annotations": [_encode_annotation(item) for item in value.annotations],
        "targets": [_encode_target(item) for item in value.targets],
        "provenance": [_encode_provenance(item) for item in value.provenance],
        "quality_flags": [
            _encode_quality_flag(item) for item in value.quality_flags
        ],
    }


def piece_to_dict(piece: CanonicalPiece) -> JsonObject:
    """Validate and encode a canonical piece as an explicit JSON mapping."""

    validate_or_raise(piece)
    return _encode_piece(piece)


_INVALID = object()
_Decoded: TypeAlias = Any
_Decoder: TypeAlias = Callable[["_DecodeContext", object, str], _Decoded]


class _DecodeContext:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []
        self._issue_set: set[ValidationIssue] = set()

    def add(self, code: ValidationCode, message: str, path: str) -> None:
        issue = ValidationIssue(
            code=code,
            severity="error",
            message=message,
            path=path,
            entity_id=None,
        )
        if issue not in self._issue_set:
            self._issue_set.add(issue)
            self.issues.append(issue)

    def raise_if_invalid(self) -> None:
        if not self.issues:
            return
        issues = tuple(
            sorted(
                self.issues,
                key=lambda issue: (
                    issue.path,
                    issue.severity,
                    issue.code,
                    issue.entity_id or "",
                    issue.message,
                ),
            )
        )
        raise CanonicalValidationError(ValidationReport(issues=issues))


def _pointer(path: str, segment: str | int) -> str:
    escaped = str(segment).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _expect_mapping(
    ctx: _DecodeContext, value: object, path: str
) -> Mapping[str, Any] | None:
    if value is _INVALID:
        return None
    if not isinstance(value, Mapping):
        ctx.add("JSON_TYPE_INVALID", "expected a JSON object", path)
        return None
    return value


def _expect_exact_fields(
    ctx: _DecodeContext,
    value: object,
    path: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any] | None:
    mapping = _expect_mapping(ctx, value, path)
    if mapping is None:
        return None
    allowed = set(fields)
    for key in mapping:
        if not isinstance(key, str):
            ctx.add(
                "JSON_TYPE_INVALID",
                "JSON object keys must be strings",
                path,
            )
        elif key not in allowed:
            ctx.add(
                "JSON_UNKNOWN_FIELD",
                f"unknown field {key!r}",
                _pointer(path, key),
            )
    for field in fields:
        if field not in mapping:
            ctx.add(
                "JSON_MISSING_FIELD",
                f"missing required field {field!r}",
                _pointer(path, field),
            )
    return mapping


def _expect_string(ctx: _DecodeContext, value: object, path: str) -> str | object:
    if value is _INVALID:
        return _INVALID
    if not isinstance(value, str):
        ctx.add("JSON_TYPE_INVALID", "expected a string", path)
        return _INVALID
    return value


def _expect_optional_string(
    ctx: _DecodeContext, value: object, path: str
) -> str | None | object:
    if value is _INVALID:
        return _INVALID
    if value is None:
        return None
    return _expect_string(ctx, value, path)


def _expect_bool(ctx: _DecodeContext, value: object, path: str) -> bool | object:
    if value is _INVALID:
        return _INVALID
    if not isinstance(value, bool):
        ctx.add("JSON_TYPE_INVALID", "expected a boolean", path)
        return _INVALID
    return value


def _expect_int(ctx: _DecodeContext, value: object, path: str) -> int | object:
    if value is _INVALID:
        return _INVALID
    if isinstance(value, bool) or not isinstance(value, int):
        ctx.add("JSON_TYPE_INVALID", "expected an integer", path)
        return _INVALID
    return value


def _expect_optional_int(
    ctx: _DecodeContext, value: object, path: str
) -> int | None | object:
    if value is _INVALID:
        return _INVALID
    if value is None:
        return None
    return _expect_int(ctx, value, path)


def _expect_number(
    ctx: _DecodeContext, value: object, path: str
) -> int | float | object:
    if value is _INVALID:
        return _INVALID
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        ctx.add("JSON_TYPE_INVALID", "expected a number", path)
        return _INVALID
    return value


def _expect_float_number(
    ctx: _DecodeContext, value: object, path: str
) -> float | object:
    number = _expect_number(ctx, value, path)
    if number is _INVALID:
        return _INVALID
    try:
        return float(number)
    except OverflowError:
        ctx.add(
            "VALUE_NOT_FINITE",
            "integer is outside the finite float range",
            path,
        )
        return _INVALID


def _expect_optional_float(
    ctx: _DecodeContext, value: object, path: str
) -> float | None | object:
    if value is _INVALID:
        return _INVALID
    if value is None:
        return None
    return _expect_float_number(ctx, value, path)


def _expect_list(
    ctx: _DecodeContext, value: object, path: str
) -> list[Any] | None:
    if value is _INVALID:
        return None
    if not isinstance(value, list):
        ctx.add("JSON_TYPE_INVALID", "expected a JSON array", path)
        return None
    return value


def _decode_list(
    ctx: _DecodeContext,
    value: object,
    path: str,
    decoder: _Decoder,
) -> tuple[Any, ...] | object:
    sequence = _expect_list(ctx, value, path)
    if sequence is None:
        return _INVALID
    decoded: list[Any] = []
    valid = True
    for index, item in enumerate(sequence):
        result = decoder(ctx, item, _pointer(path, index))
        if result is _INVALID:
            valid = False
        else:
            decoded.append(result)
    return tuple(decoded) if valid else _INVALID


def _decode_optional_string_list(
    ctx: _DecodeContext, value: object, path: str
) -> tuple[str, ...] | None | object:
    if value is _INVALID:
        return _INVALID
    if value is None:
        return None
    return _decode_list(ctx, value, path, _expect_string)


def _decode_string_list(
    ctx: _DecodeContext, value: object, path: str
) -> tuple[str, ...] | object:
    return _decode_list(ctx, value, path, _expect_string)


def _all_valid(*values: object) -> bool:
    return all(value is not _INVALID for value in values)


def _decode_rational(
    ctx: _DecodeContext, value: object, path: str
) -> RationalTime | object:
    mapping = _expect_exact_fields(ctx, value, path, ("num", "den"))
    if mapping is None:
        return _INVALID
    num = _expect_int(ctx, mapping.get("num", _INVALID), _pointer(path, "num"))
    den = _expect_int(ctx, mapping.get("den", _INVALID), _pointer(path, "den"))
    if not _all_valid(num, den):
        return _INVALID
    assert isinstance(num, int) and not isinstance(num, bool)
    assert isinstance(den, int) and not isinstance(den, bool)
    if den == 0:
        ctx.add("RATIONAL_INVALID", "rational denominator must not be zero", path)
        return _INVALID
    if den < 0:
        ctx.add(
            "RATIONAL_NOT_NORMALIZED",
            "rational denominator must be positive",
            path,
        )
        return _INVALID
    if (num == 0 and den != 1) or (num != 0 and gcd(abs(num), den) != 1):
        ctx.add(
            "RATIONAL_NOT_NORMALIZED",
            "rational value must already be normalized",
            path,
        )
        return _INVALID
    return RationalTime(num, den)


def _decode_metadata(
    ctx: _DecodeContext, value: object, path: str
) -> PieceMetadata | object:
    mapping = _expect_exact_fields(
        ctx,
        value,
        path,
        (
            "source_format",
            "title",
            "creators",
            "collection",
            "movement_title",
            "movement_number",
            "genres",
            "copyright",
            "language",
        ),
    )
    if mapping is None:
        return _INVALID
    source_format = _expect_string(
        ctx, mapping.get("source_format", _INVALID), _pointer(path, "source_format")
    )
    title = _expect_optional_string(
        ctx, mapping.get("title", _INVALID), _pointer(path, "title")
    )
    creators = _decode_optional_string_list(
        ctx, mapping.get("creators", _INVALID), _pointer(path, "creators")
    )
    collection = _expect_optional_string(
        ctx, mapping.get("collection", _INVALID), _pointer(path, "collection")
    )
    movement_title = _expect_optional_string(
        ctx,
        mapping.get("movement_title", _INVALID),
        _pointer(path, "movement_title"),
    )
    movement_number = _expect_optional_string(
        ctx,
        mapping.get("movement_number", _INVALID),
        _pointer(path, "movement_number"),
    )
    genres = _decode_optional_string_list(
        ctx, mapping.get("genres", _INVALID), _pointer(path, "genres")
    )
    copyright_value = _expect_optional_string(
        ctx, mapping.get("copyright", _INVALID), _pointer(path, "copyright")
    )
    language = _expect_optional_string(
        ctx, mapping.get("language", _INVALID), _pointer(path, "language")
    )
    values = (
        source_format,
        title,
        creators,
        collection,
        movement_title,
        movement_number,
        genres,
        copyright_value,
        language,
    )
    if not _all_valid(*values):
        return _INVALID
    return PieceMetadata(*values)


def _decode_track(
    ctx: _DecodeContext, value: object, path: str
) -> CanonicalTrack | object:
    mapping = _expect_exact_fields(
        ctx,
        value,
        path,
        (
            "track_id",
            "source_track_index",
            "name",
            "instrument_name",
            "program",
            "channel",
            "is_percussion",
            "provenance_id",
        ),
    )
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(ctx, mapping.get("track_id", _INVALID), _pointer(path, "track_id")),
        _expect_optional_int(
            ctx,
            mapping.get("source_track_index", _INVALID),
            _pointer(path, "source_track_index"),
        ),
        _expect_optional_string(ctx, mapping.get("name", _INVALID), _pointer(path, "name")),
        _expect_optional_string(
            ctx,
            mapping.get("instrument_name", _INVALID),
            _pointer(path, "instrument_name"),
        ),
        _expect_optional_int(ctx, mapping.get("program", _INVALID), _pointer(path, "program")),
        _expect_optional_int(ctx, mapping.get("channel", _INVALID), _pointer(path, "channel")),
        _expect_bool(
            ctx, mapping.get("is_percussion", _INVALID), _pointer(path, "is_percussion")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return CanonicalTrack(*values) if _all_valid(*values) else _INVALID


def _decode_note(
    ctx: _DecodeContext, value: object, path: str
) -> CanonicalNote | object:
    fields = (
        "note_id",
        "track_id",
        "pitch",
        "onset_qn",
        "duration_qn",
        "velocity",
        "channel",
        "program",
        "is_percussion",
        "is_grace",
        "spelling_step",
        "spelling_alter",
        "staff",
        "voice",
        "articulations",
        "dynamic",
        "source_onset_ticks",
        "source_duration_ticks",
        "source_onset_seconds",
        "source_duration_seconds",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(ctx, mapping.get("note_id", _INVALID), _pointer(path, "note_id")),
        _expect_string(ctx, mapping.get("track_id", _INVALID), _pointer(path, "track_id")),
        _expect_int(ctx, mapping.get("pitch", _INVALID), _pointer(path, "pitch")),
        _decode_rational(ctx, mapping.get("onset_qn", _INVALID), _pointer(path, "onset_qn")),
        _decode_rational(
            ctx, mapping.get("duration_qn", _INVALID), _pointer(path, "duration_qn")
        ),
        _expect_optional_int(ctx, mapping.get("velocity", _INVALID), _pointer(path, "velocity")),
        _expect_optional_int(ctx, mapping.get("channel", _INVALID), _pointer(path, "channel")),
        _expect_optional_int(ctx, mapping.get("program", _INVALID), _pointer(path, "program")),
        _expect_bool(
            ctx, mapping.get("is_percussion", _INVALID), _pointer(path, "is_percussion")
        ),
        _expect_bool(ctx, mapping.get("is_grace", _INVALID), _pointer(path, "is_grace")),
        _expect_optional_string(
            ctx, mapping.get("spelling_step", _INVALID), _pointer(path, "spelling_step")
        ),
        _expect_optional_int(
            ctx, mapping.get("spelling_alter", _INVALID), _pointer(path, "spelling_alter")
        ),
        _expect_optional_int(ctx, mapping.get("staff", _INVALID), _pointer(path, "staff")),
        _expect_optional_int(ctx, mapping.get("voice", _INVALID), _pointer(path, "voice")),
        _decode_optional_string_list(
            ctx, mapping.get("articulations", _INVALID), _pointer(path, "articulations")
        ),
        _expect_optional_string(
            ctx, mapping.get("dynamic", _INVALID), _pointer(path, "dynamic")
        ),
        _expect_optional_int(
            ctx,
            mapping.get("source_onset_ticks", _INVALID),
            _pointer(path, "source_onset_ticks"),
        ),
        _expect_optional_int(
            ctx,
            mapping.get("source_duration_ticks", _INVALID),
            _pointer(path, "source_duration_ticks"),
        ),
        _expect_optional_float(
            ctx,
            mapping.get("source_onset_seconds", _INVALID),
            _pointer(path, "source_onset_seconds"),
        ),
        _expect_optional_float(
            ctx,
            mapping.get("source_duration_seconds", _INVALID),
            _pointer(path, "source_duration_seconds"),
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return CanonicalNote(*values) if _all_valid(*values) else _INVALID


def _decode_bar(
    ctx: _DecodeContext, value: object, path: str
) -> CanonicalBar | object:
    fields = (
        "bar_id",
        "index",
        "start_qn",
        "duration_qn",
        "meter_event_id",
        "metric_offset_qn",
        "is_pickup",
        "is_incomplete",
        "display_number",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(ctx, mapping.get("bar_id", _INVALID), _pointer(path, "bar_id")),
        _expect_int(ctx, mapping.get("index", _INVALID), _pointer(path, "index")),
        _decode_rational(ctx, mapping.get("start_qn", _INVALID), _pointer(path, "start_qn")),
        _decode_rational(
            ctx, mapping.get("duration_qn", _INVALID), _pointer(path, "duration_qn")
        ),
        _expect_string(
            ctx, mapping.get("meter_event_id", _INVALID), _pointer(path, "meter_event_id")
        ),
        _decode_rational(
            ctx,
            mapping.get("metric_offset_qn", _INVALID),
            _pointer(path, "metric_offset_qn"),
        ),
        _expect_bool(ctx, mapping.get("is_pickup", _INVALID), _pointer(path, "is_pickup")),
        _expect_bool(
            ctx, mapping.get("is_incomplete", _INVALID), _pointer(path, "is_incomplete")
        ),
        _expect_optional_string(
            ctx, mapping.get("display_number", _INVALID), _pointer(path, "display_number")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return CanonicalBar(*values) if _all_valid(*values) else _INVALID


def _decode_beat(
    ctx: _DecodeContext, value: object, path: str
) -> CanonicalBeat | object:
    fields = (
        "beat_id",
        "bar_id",
        "meter_event_id",
        "index_in_bar",
        "start_qn",
        "duration_qn",
        "position_in_bar_qn",
        "is_downbeat",
        "strength",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(ctx, mapping.get("beat_id", _INVALID), _pointer(path, "beat_id")),
        _expect_string(ctx, mapping.get("bar_id", _INVALID), _pointer(path, "bar_id")),
        _expect_string(
            ctx, mapping.get("meter_event_id", _INVALID), _pointer(path, "meter_event_id")
        ),
        _expect_int(
            ctx, mapping.get("index_in_bar", _INVALID), _pointer(path, "index_in_bar")
        ),
        _decode_rational(ctx, mapping.get("start_qn", _INVALID), _pointer(path, "start_qn")),
        _decode_rational(
            ctx, mapping.get("duration_qn", _INVALID), _pointer(path, "duration_qn")
        ),
        _decode_rational(
            ctx,
            mapping.get("position_in_bar_qn", _INVALID),
            _pointer(path, "position_in_bar_qn"),
        ),
        _expect_bool(
            ctx, mapping.get("is_downbeat", _INVALID), _pointer(path, "is_downbeat")
        ),
        _expect_optional_float(
            ctx, mapping.get("strength", _INVALID), _pointer(path, "strength")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return CanonicalBeat(*values) if _all_valid(*values) else _INVALID


def _decode_tempo_event(
    ctx: _DecodeContext, value: object, path: str
) -> TempoEvent | object:
    fields = (
        "tempo_event_id",
        "onset_qn",
        "microseconds_per_quarter",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(
            ctx, mapping.get("tempo_event_id", _INVALID), _pointer(path, "tempo_event_id")
        ),
        _decode_rational(ctx, mapping.get("onset_qn", _INVALID), _pointer(path, "onset_qn")),
        _expect_int(
            ctx,
            mapping.get("microseconds_per_quarter", _INVALID),
            _pointer(path, "microseconds_per_quarter"),
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return TempoEvent(*values) if _all_valid(*values) else _INVALID


def _decode_meter_event(
    ctx: _DecodeContext, value: object, path: str
) -> MeterEvent | object:
    fields = (
        "meter_event_id",
        "onset_qn",
        "numerator",
        "denominator",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(
            ctx, mapping.get("meter_event_id", _INVALID), _pointer(path, "meter_event_id")
        ),
        _decode_rational(ctx, mapping.get("onset_qn", _INVALID), _pointer(path, "onset_qn")),
        _expect_int(ctx, mapping.get("numerator", _INVALID), _pointer(path, "numerator")),
        _expect_int(
            ctx, mapping.get("denominator", _INVALID), _pointer(path, "denominator")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return MeterEvent(*values) if _all_valid(*values) else _INVALID


def _decode_key_signature_event(
    ctx: _DecodeContext, value: object, path: str
) -> KeySignatureEvent | object:
    fields = (
        "key_signature_event_id",
        "onset_qn",
        "fifths",
        "mode",
        "raw_value",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(
            ctx,
            mapping.get("key_signature_event_id", _INVALID),
            _pointer(path, "key_signature_event_id"),
        ),
        _decode_rational(ctx, mapping.get("onset_qn", _INVALID), _pointer(path, "onset_qn")),
        _expect_int(ctx, mapping.get("fifths", _INVALID), _pointer(path, "fifths")),
        _expect_string(ctx, mapping.get("mode", _INVALID), _pointer(path, "mode")),
        _expect_optional_string(
            ctx, mapping.get("raw_value", _INVALID), _pointer(path, "raw_value")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return KeySignatureEvent(*values) if _all_valid(*values) else _INVALID


def _decode_annotation(
    ctx: _DecodeContext, value: object, path: str
) -> AnnotationSpan | object:
    fields = (
        "annotation_id",
        "annotation_type",
        "layer",
        "start_qn",
        "end_qn",
        "track_id",
        "value",
        "provenance_id",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(
            ctx, mapping.get("annotation_id", _INVALID), _pointer(path, "annotation_id")
        ),
        _expect_string(
            ctx,
            mapping.get("annotation_type", _INVALID),
            _pointer(path, "annotation_type"),
        ),
        _expect_string(ctx, mapping.get("layer", _INVALID), _pointer(path, "layer")),
        _decode_rational(ctx, mapping.get("start_qn", _INVALID), _pointer(path, "start_qn")),
        _decode_rational(ctx, mapping.get("end_qn", _INVALID), _pointer(path, "end_qn")),
        _expect_optional_string(
            ctx, mapping.get("track_id", _INVALID), _pointer(path, "track_id")
        ),
        _expect_optional_string(ctx, mapping.get("value", _INVALID), _pointer(path, "value")),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return AnnotationSpan(*values) if _all_valid(*values) else _INVALID


def _decode_target_value(
    ctx: _DecodeContext, value: object, path: str, value_type: object
) -> object:
    if value is None:
        return None
    if value_type == "categorical":
        return _expect_string(ctx, value, path)
    if value_type == "scalar":
        return _expect_number(ctx, value, path)
    if value_type == "multi_label":
        return _decode_string_list(ctx, value, path)
    if value_type == "distribution":
        return _decode_list(ctx, value, path, _expect_float_number)

    if isinstance(value, str) or (
        not isinstance(value, bool) and isinstance(value, (int, float))
    ):
        return value
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return tuple(value)
        return _decode_list(ctx, value, path, _expect_float_number)
    ctx.add("JSON_TYPE_INVALID", "invalid target value JSON shape", path)
    return _INVALID


def _decode_target_values(
    ctx: _DecodeContext, value: object, path: str, value_type: object
) -> tuple[Any, ...] | object:
    sequence = _expect_list(ctx, value, path)
    if sequence is None:
        return _INVALID
    decoded: list[Any] = []
    valid = True
    for index, item in enumerate(sequence):
        result = _decode_target_value(ctx, item, _pointer(path, index), value_type)
        if result is _INVALID:
            valid = False
        else:
            decoded.append(result)
    return tuple(decoded) if valid else _INVALID


def _decode_nullable_string(
    ctx: _DecodeContext, value: object, path: str
) -> str | None | object:
    return _expect_optional_string(ctx, value, path)


def _decode_nullable_float(
    ctx: _DecodeContext, value: object, path: str
) -> float | None | object:
    return _expect_optional_float(ctx, value, path)


def _decode_target(
    ctx: _DecodeContext, value: object, path: str
) -> TargetArray | object:
    fields = (
        "target_id",
        "task",
        "annotation_view_id",
        "alignment_type",
        "entity_ids",
        "value_type",
        "class_labels",
        "values",
        "mask",
        "confidence",
        "source",
        "provenance",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    value_type = _expect_string(
        ctx, mapping.get("value_type", _INVALID), _pointer(path, "value_type")
    )
    values = (
        _expect_string(
            ctx, mapping.get("target_id", _INVALID), _pointer(path, "target_id")
        ),
        _expect_string(ctx, mapping.get("task", _INVALID), _pointer(path, "task")),
        _expect_optional_string(
            ctx,
            mapping.get("annotation_view_id", _INVALID),
            _pointer(path, "annotation_view_id"),
        ),
        _expect_string(
            ctx,
            mapping.get("alignment_type", _INVALID),
            _pointer(path, "alignment_type"),
        ),
        _decode_string_list(
            ctx, mapping.get("entity_ids", _INVALID), _pointer(path, "entity_ids")
        ),
        value_type,
        _decode_optional_string_list(
            ctx, mapping.get("class_labels", _INVALID), _pointer(path, "class_labels")
        ),
        _decode_target_values(
            ctx, mapping.get("values", _INVALID), _pointer(path, "values"), value_type
        ),
        _decode_list(ctx, mapping.get("mask", _INVALID), _pointer(path, "mask"), _expect_bool),
        _decode_list(
            ctx,
            mapping.get("confidence", _INVALID),
            _pointer(path, "confidence"),
            _decode_nullable_float,
        ),
        _decode_list(
            ctx,
            mapping.get("source", _INVALID),
            _pointer(path, "source"),
            _decode_nullable_string,
        ),
        _decode_list(
            ctx,
            mapping.get("provenance", _INVALID),
            _pointer(path, "provenance"),
            _decode_nullable_string,
        ),
    )
    return TargetArray(*values) if _all_valid(*values) else _INVALID


def _decode_details(
    ctx: _DecodeContext, value: object, path: str
) -> tuple[tuple[str, Any], ...] | object:
    mapping = _expect_mapping(ctx, value, path)
    if mapping is None:
        return _INVALID
    details: list[tuple[str, Any]] = []
    valid = True
    for key, detail_value in mapping.items():
        if not isinstance(key, str):
            ctx.add("JSON_TYPE_INVALID", "JSON object keys must be strings", path)
            valid = False
            continue
        detail_path = _pointer(path, key)
        if isinstance(detail_value, (list, Mapping)):
            ctx.add(
                "JSON_TYPE_INVALID",
                "provenance detail values must be JSON scalars",
                detail_path,
            )
            valid = False
            continue
        if detail_value is not None and not isinstance(
            detail_value, (str, int, float, bool)
        ):
            ctx.add(
                "JSON_TYPE_INVALID",
                "provenance detail values must be JSON scalars",
                detail_path,
            )
            valid = False
            continue
        details.append((key, detail_value))
    return tuple(sorted(details, key=lambda item: item[0])) if valid else _INVALID


def _decode_provenance(
    ctx: _DecodeContext, value: object, path: str
) -> ProvenanceRecord | object:
    fields = (
        "provenance_id",
        "kind",
        "source",
        "record_id",
        "uri",
        "version",
        "checksum_sha256",
        "created_at",
        "parents",
        "details",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
        _expect_string(ctx, mapping.get("kind", _INVALID), _pointer(path, "kind")),
        _expect_string(ctx, mapping.get("source", _INVALID), _pointer(path, "source")),
        _expect_optional_string(
            ctx, mapping.get("record_id", _INVALID), _pointer(path, "record_id")
        ),
        _expect_optional_string(ctx, mapping.get("uri", _INVALID), _pointer(path, "uri")),
        _expect_optional_string(
            ctx, mapping.get("version", _INVALID), _pointer(path, "version")
        ),
        _expect_optional_string(
            ctx,
            mapping.get("checksum_sha256", _INVALID),
            _pointer(path, "checksum_sha256"),
        ),
        _expect_optional_string(
            ctx, mapping.get("created_at", _INVALID), _pointer(path, "created_at")
        ),
        _decode_string_list(
            ctx, mapping.get("parents", _INVALID), _pointer(path, "parents")
        ),
        _decode_details(ctx, mapping.get("details", _INVALID), _pointer(path, "details")),
    )
    return ProvenanceRecord(*values) if _all_valid(*values) else _INVALID


def _decode_quality_flag(
    ctx: _DecodeContext, value: object, path: str
) -> QualityFlag | object:
    fields = ("code", "severity", "message", "entity_ids", "provenance_id")
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    values = (
        _expect_string(ctx, mapping.get("code", _INVALID), _pointer(path, "code")),
        _expect_string(
            ctx, mapping.get("severity", _INVALID), _pointer(path, "severity")
        ),
        _expect_string(ctx, mapping.get("message", _INVALID), _pointer(path, "message")),
        _decode_string_list(
            ctx, mapping.get("entity_ids", _INVALID), _pointer(path, "entity_ids")
        ),
        _expect_optional_string(
            ctx, mapping.get("provenance_id", _INVALID), _pointer(path, "provenance_id")
        ),
    )
    return QualityFlag(*values) if _all_valid(*values) else _INVALID


def _decode_piece(
    ctx: _DecodeContext, value: object, path: str = ""
) -> CanonicalPiece | object:
    fields = (
        "schema_version",
        "piece_id",
        "dataset_name",
        "source_group_id",
        "split",
        "source_path",
        "source_resolution",
        "duration_qn",
        "metadata",
        "tracks",
        "notes",
        "bars",
        "beats",
        "tempo_events",
        "meter_events",
        "key_signature_events",
        "annotations",
        "targets",
        "provenance",
        "quality_flags",
    )
    mapping = _expect_exact_fields(ctx, value, path, fields)
    if mapping is None:
        return _INVALID
    schema_version = _expect_string(
        ctx,
        mapping.get("schema_version", _INVALID),
        _pointer(path, "schema_version"),
    )
    if schema_version is not _INVALID and schema_version != SCHEMA_VERSION:
        ctx.add(
            "SCHEMA_VERSION_UNSUPPORTED",
            f"expected schema version {SCHEMA_VERSION!r}",
            _pointer(path, "schema_version"),
        )
        schema_version = _INVALID
    values = (
        schema_version,
        _expect_string(ctx, mapping.get("piece_id", _INVALID), _pointer(path, "piece_id")),
        _expect_string(
            ctx, mapping.get("dataset_name", _INVALID), _pointer(path, "dataset_name")
        ),
        _expect_string(
            ctx,
            mapping.get("source_group_id", _INVALID),
            _pointer(path, "source_group_id"),
        ),
        _expect_optional_string(ctx, mapping.get("split", _INVALID), _pointer(path, "split")),
        _expect_optional_string(
            ctx, mapping.get("source_path", _INVALID), _pointer(path, "source_path")
        ),
        _expect_optional_int(
            ctx,
            mapping.get("source_resolution", _INVALID),
            _pointer(path, "source_resolution"),
        ),
        _decode_rational(
            ctx, mapping.get("duration_qn", _INVALID), _pointer(path, "duration_qn")
        ),
        _decode_metadata(ctx, mapping.get("metadata", _INVALID), _pointer(path, "metadata")),
        _decode_list(ctx, mapping.get("tracks", _INVALID), _pointer(path, "tracks"), _decode_track),
        _decode_list(ctx, mapping.get("notes", _INVALID), _pointer(path, "notes"), _decode_note),
        _decode_list(ctx, mapping.get("bars", _INVALID), _pointer(path, "bars"), _decode_bar),
        _decode_list(ctx, mapping.get("beats", _INVALID), _pointer(path, "beats"), _decode_beat),
        _decode_list(
            ctx,
            mapping.get("tempo_events", _INVALID),
            _pointer(path, "tempo_events"),
            _decode_tempo_event,
        ),
        _decode_list(
            ctx,
            mapping.get("meter_events", _INVALID),
            _pointer(path, "meter_events"),
            _decode_meter_event,
        ),
        _decode_list(
            ctx,
            mapping.get("key_signature_events", _INVALID),
            _pointer(path, "key_signature_events"),
            _decode_key_signature_event,
        ),
        _decode_list(
            ctx,
            mapping.get("annotations", _INVALID),
            _pointer(path, "annotations"),
            _decode_annotation,
        ),
        _decode_list(
            ctx, mapping.get("targets", _INVALID), _pointer(path, "targets"), _decode_target
        ),
        _decode_list(
            ctx,
            mapping.get("provenance", _INVALID),
            _pointer(path, "provenance"),
            _decode_provenance,
        ),
        _decode_list(
            ctx,
            mapping.get("quality_flags", _INVALID),
            _pointer(path, "quality_flags"),
            _decode_quality_flag,
        ),
    )
    return CanonicalPiece(*values) if _all_valid(*values) else _INVALID


def piece_from_dict(data: Mapping[str, Any]) -> CanonicalPiece:
    """Strictly decode and validate a canonical piece mapping."""

    ctx = _DecodeContext()
    piece = _decode_piece(ctx, data)
    ctx.raise_if_invalid()
    if piece is _INVALID:
        raise RuntimeError("decoder could not construct a piece without an issue")
    assert isinstance(piece, CanonicalPiece)
    validate_or_raise(piece)
    return piece


def dumps_piece(piece: CanonicalPiece, *, indent: int | None = None) -> str:
    """Return deterministic canonical JSON without a terminal newline."""

    return json.dumps(
        piece_to_dict(piece),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def loads_piece(payload: str | bytes | bytearray) -> CanonicalPiece:
    """Decode UTF-8 canonical JSON and validate the resulting piece."""

    text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload

    def reject_constant(value: str) -> None:
        position = text.find(value)
        raise json.JSONDecodeError(
            f"non-standard JSON constant {value!r}", text, max(position, 0)
        )

    data = json.loads(text, parse_constant=reject_constant)
    return piece_from_dict(data)


def dump_piece(
    piece: CanonicalPiece,
    path: str | PathLike[str],
    *,
    indent: int | None = 2,
) -> None:
    """Write deterministic UTF-8 canonical JSON with one terminal newline."""

    payload = dumps_piece(piece, indent=indent)
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.write("\n")


def load_piece(path: str | PathLike[str]) -> CanonicalPiece:
    """Read a UTF-8 canonical JSON file."""

    with open(path, encoding="utf-8") as stream:
        return loads_piece(stream.read())
