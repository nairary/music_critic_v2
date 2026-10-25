"""Bounded, dependency-free raw projection of MusicRender metrical JSON.

The observed upstream definitions are pinned below. This parser is
fixture-validated, not a claim of production release compatibility. No
serialized Python object is restored.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from music_critic.data import (
    SCHEMA_VERSION, CanonicalBar, CanonicalBeat, CanonicalNote, CanonicalPiece,
    CanonicalTrack, KeySignatureEvent, MeterEvent, PieceMetadata,
    ProvenanceRecord, QualityFlag, RationalTime, TempoEvent, validate_piece,
)


PDMX_ADAPTER_VERSION = "1.0.1"
PDMX_PROJECTION_VERSION = "1.0.1"
PDMX_INTEGER_NUMBER_POLICY = "finite-integral-decimal-exact-int-v1"
# Match Python's default JSON integer digit envelope before expanding compact
# Decimal exponents. This is a resource bound, never a rounding threshold.
PDMX_MAX_DECIMAL_INTEGER_DIGITS = 4300
PDMX_SCHEMA_PROFILE_ID = "musicrender-metrical-json-raw-v1"
PDMX_SCHEMA_UPSTREAM_REVISION = "5c6a25b613e4497469e855e8bd1423b1d834ff45"
PDMX_SCHEMA_PRIMARY_URLS = (
    f"https://github.com/pnlong/PDMX/blob/{PDMX_SCHEMA_UPSTREAM_REVISION}/reading/music.py",
    f"https://github.com/pnlong/PDMX/blob/{PDMX_SCHEMA_UPSTREAM_REVISION}/reading/classes.py",
)
PDMX_SCHEMA_EVIDENCE_STATUS = "pinned_primary_definitions_fixture_validated"
# Pin the accepted raw feature envelope without importing graph/__init__ and
# its torch/PyG runtime into the source parser. The contract test compares all
# meter entries below with RAW_FEATURE_REGISTRY, so registry drift fails CI.
PDMX_RAW_FEATURE_REGISTRY_VERSION = "1.0.0"
PDMX_METER_NUMERATOR_VOCABULARY_SIZE = 256
PDMX_METER_DENOMINATOR_LOG2_VOCABULARY_SIZE = 128


class PdmxAdapterError(ValueError):
    """Expected source rejection; programming exceptions remain uncaught."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


@dataclass(frozen=True, slots=True)
class PdmxAdapterConfig:
    max_source_bytes: int = 64 * 1024 * 1024
    max_notes: int = 200_000
    max_bars: int = 50_000
    max_tracks: int = 1_024
    max_beats: int = 500_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PdmxAdapterError("invalid_schema", f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PdmxAdapterError("invalid_schema", f"{label} must be an array")
    return value


def _int(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, Decimal):
        if not value.is_finite() or value < minimum or (maximum is not None and value > maximum):
            raise PdmxAdapterError("invalid_schema", f"{label} must be an integer in range")
        if value and value.adjusted() >= PDMX_MAX_DECIMAL_INTEGER_DIGITS:
            raise PdmxAdapterError("resource_numeric_expansion", f"{label} integer exceeds decimal digit bound")
        if value != value.to_integral_value():
            raise PdmxAdapterError("invalid_schema", f"{label} must be an integer in range")
        # JSON 13060.0 and 13060 denote the same exact integer. Reject fractions
        # above; int() alone would silently truncate them.
        value = int(value)
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise PdmxAdapterError("invalid_schema", f"{label} must be an integer in range")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PdmxAdapterError("invalid_schema", f"{label} must be boolean")
    return value


def _text(value: Any, label: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise PdmxAdapterError("invalid_schema", f"{label} must be a string or null")
    return value.strip() or None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PdmxAdapterError("invalid_json", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise PdmxAdapterError("invalid_json", f"non-finite JSON number: {value}")


def _events(data: dict[str, Any], key: str) -> list[tuple[int, dict[str, Any]]]:
    events = []
    for value in _list(data.get(key, []), key):
        event = _object(value, key)
        events.append((_int(event.get("time"), f"{key}.time"), event))
    events.sort(key=lambda item: item[0])
    return events


def _meters(data: dict[str, Any], resolution: int, defaults: set[str]) -> tuple[MeterEvent, ...]:
    values: dict[int, tuple[int, int]] = {}
    for tick, event in _events(data, "time_signatures"):
        numerator = _int(event.get("numerator"), "meter.numerator", 1)
        denominator = _int(event.get("denominator"), "meter.denominator", 1)
        if denominator & (denominator - 1):
            raise PdmxAdapterError("invalid_meter", "denominator must be a power of two")
        if (
            numerator >= PDMX_METER_NUMERATOR_VOCABULARY_SIZE
            or denominator.bit_length() - 1 >= PDMX_METER_DENOMINATOR_LOG2_VOCABULARY_SIZE
        ):
            raise PdmxAdapterError(
                "unsupported_meter", "meter lies outside the pinned raw feature registry vocabulary"
            )
        value = numerator, denominator
        if tick in values and values[tick] != value:
            raise PdmxAdapterError("conflicting_meter", f"different meters at tick {tick}")
        values[tick] = value
    if 0 not in values:
        values[0] = (4, 4)
        defaults.add("meter")
    return tuple(MeterEvent(
        f"meter:{index:06d}", RationalTime(tick, resolution), *value,
        "prov:default-meter" if tick == 0 and "meter" in defaults else "prov:projection",
    ) for index, (tick, value) in enumerate(sorted(values.items())))


def _tempos(data: dict[str, Any], resolution: int, defaults: set[str]) -> tuple[TempoEvent, ...]:
    values: dict[int, int] = {}
    for tick, event in _events(data, "tempos"):
        qpm = event.get("qpm", 120)
        if isinstance(qpm, bool) or not isinstance(qpm, (int, Decimal)) or qpm <= 0:
            raise PdmxAdapterError("invalid_tempo", "qpm must be a positive finite number")
        if qpm < Decimal("0.000001") or qpm > Decimal("120000000"):
            raise PdmxAdapterError("invalid_tempo", "qpm exceeds supported numeric range")
        exact = Fraction(60_000_000) / Fraction(qpm)
        micros = (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)
        if micros <= 0:
            raise PdmxAdapterError("invalid_tempo", "qpm exceeds integer tempo precision")
        if tick in values and values[tick] != micros:
            raise PdmxAdapterError("conflicting_tempo", f"different tempos at tick {tick}")
        values[tick] = micros
        if "qpm" not in event:
            defaults.add("tempo-qpm")
    if 0 not in values:
        values[0] = 500_000
        defaults.add("tempo")
    return tuple(TempoEvent(
        f"tempo:{index:06d}", RationalTime(tick, resolution), value,
        "prov:default-tempo" if tick == 0 and "tempo" in defaults else "prov:projection",
    ) for index, (tick, value) in enumerate(sorted(values.items())))


def _timeline(
    data: dict[str, Any], resolution: int, duration: RationalTime,
    meters: tuple[MeterEvent, ...], config: PdmxAdapterConfig,
) -> tuple[tuple[CanonicalBar, ...], tuple[CanonicalBeat, ...], str]:
    explicit = {RationalTime(tick, resolution) for tick, _ in _events(data, "barlines")}
    for tick, beat in _events(data, "beats"):
        if _bool(beat.get("is_downbeat", False), "beat.is_downbeat"):
            explicit.add(RationalTime(tick, resolution))
    source_boundaries = sorted(value for value in explicit if RationalTime(0) < value < duration)
    meter_starts = [meter.onset_qn for meter in meters]
    boundaries = sorted(set(source_boundaries).union(meter_starts, (duration,)))
    bars: list[CanonicalBar] = []
    beats: list[CanonicalBeat] = []
    cursor = RationalTime(0)
    while cursor < duration:
        if len(bars) >= config.max_bars:
            raise PdmxAdapterError("resource_max_bars", "bar count exceeds configured limit")
        meter = meters[bisect_right(meter_starts, cursor) - 1]
        nominal = RationalTime(4 * meter.numerator, meter.denominator)
        unit = RationalTime(4, meter.denominator)
        end = min(cursor + nominal, boundaries[bisect_right(boundaries, cursor)])
        span = end - cursor
        incomplete = span < nominal
        pickup = (
            not bars and incomplete and end < duration and end in explicit
            and end not in meter_starts
        )
        offset = nominal - span if pickup else RationalTime(0)
        if (offset.to_fraction() / unit.to_fraction()).denominator != 1 or (
            incomplete and not pickup and end != duration and end not in meter_starts
        ):
            raise PdmxAdapterError(
                "unsupported_metric_geometry", "source bar cannot be represented by canonical bar contract"
            )
        bar_id = f"bar:{len(bars):06d}"
        bars.append(CanonicalBar(
            bar_id, len(bars), cursor, span, meter.meter_event_id, offset,
            pickup, incomplete, None, "prov:projection",
        ))
        position = RationalTime(0)
        index = int(offset.to_fraction() / unit.to_fraction())
        while position < span:
            if len(beats) >= config.max_beats:
                raise PdmxAdapterError("resource_max_beats", "beat count exceeds configured limit")
            beat_span = min(unit, span - position)
            metric_position = offset + position
            beats.append(CanonicalBeat(
                f"beat:{len(beats):07d}", bar_id, meter.meter_event_id, index,
                cursor + position, beat_span, metric_position,
                metric_position == RationalTime(0), None, "prov:projection",
            ))
            position += beat_span
            index += 1
        cursor = end
    return tuple(bars), tuple(beats), "source_boundaries_plus_meter_grid" if explicit else "meter_grid"


def _keys(data: dict[str, Any], resolution: int, duration: RationalTime) -> tuple[KeySignatureEvent, ...]:
    values: dict[int, tuple[int, str]] = {}
    modes = {"major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian", "unknown"}
    for tick, event in _events(data, "key_signatures"):
        # Root alone is insufficient evidence for a notated fifths spelling.
        if event.get("fifths") is None:
            continue
        fifths = _int(event["fifths"], "key.fifths", -7, 7)
        mode = _text(event.get("mode"), "key.mode") or "unknown"
        if RationalTime(tick, resolution) > duration:
            raise PdmxAdapterError("key_outside_raw_timeline", "key signature follows all raw timeline evidence")
        if tick in values and values[tick] != (fifths, mode):
            raise PdmxAdapterError("conflicting_key_signature", f"different key signatures at tick {tick}")
        values[tick] = fifths, mode
    return tuple(KeySignatureEvent(
            f"keysig:{index:06d}", RationalTime(tick, resolution), fifths,
            mode if mode in modes else "other", mode if mode not in modes else None,
            "prov:projection",
        ) for index, (tick, (fifths, mode)) in enumerate(sorted(values.items())))


def load_pdmx_piece(
    path: Path, *, source_id: str, source_group_id: str, split: str,
    config: PdmxAdapterConfig = PdmxAdapterConfig(),
) -> CanonicalPiece:
    """Read one explicitly authorized TRAIN/VALIDATION metrical JSON source.

    Timing comes only from notes and raw meter/tempo/barline/beat events.
    ``song_length`` is retained only as source evidence: upstream computes it
    using lyrics and annotations too, which must not enlarge the raw graph.
    Source note identity is its stable track/notes-array ordinal, not pitch.
    """
    if split not in {"train", "validation"}:
        raise PdmxAdapterError("split_forbidden", "only explicit train/validation input is authorized")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a nonempty string")
    if not isinstance(source_group_id, str) or not source_group_id.strip():
        raise ValueError("source_group_id must be a nonempty string")
    path = Path(path)
    try:
        with path.open("rb") as stream:
            payload = stream.read(config.max_source_bytes + 1)
    except OSError as exc:
        raise PdmxAdapterError("source_unreadable", str(exc)) from exc
    if len(payload) > config.max_source_bytes:
        raise PdmxAdapterError("resource_max_source_bytes", "source exceeds configured byte limit")
    try:
        data = json.loads(payload, parse_float=Decimal, parse_constant=_invalid_constant,
                          object_pairs_hook=_unique_object)
    except PdmxAdapterError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError, InvalidOperation) as exc:
        raise PdmxAdapterError("invalid_json", str(exc)) from exc
    data = _object(data, "MusicRender")
    if _bool(data.get("absolute_time", False), "absolute_time"):
        raise PdmxAdapterError("unsupported_absolute_time", "exact metrical ticks are required")
    resolution = _int(data.get("resolution"), "resolution", 1)
    source_tracks = _list(data.get("tracks", []), "tracks")
    if len(source_tracks) > config.max_tracks:
        raise PdmxAdapterError("resource_max_tracks", "track count exceeds configured limit")
    defaults: set[str] = set()
    if "absolute_time" not in data:
        defaults.add("metrical-time")
    tracks = []
    notes = []
    for track_index, value in enumerate(source_tracks):
        track = _object(value, "track")
        track_id = f"track:{track_index:06d}"
        program = None if track.get("program") is None else _int(track["program"], "track.program", 0, 127)
        percussion = _bool(track.get("is_drum", False), "track.is_drum")
        if "is_drum" not in track:
            defaults.add("non-percussion")
        tracks.append(CanonicalTrack(
            track_id, track_index, _text(track.get("name"), "track.name"), None,
            program, None, percussion, "prov:projection",
        ))
        for ordinal, value in enumerate(_list(track.get("notes", []), "track.notes")):
            if len(notes) >= config.max_notes:
                raise PdmxAdapterError("resource_max_notes", "note count exceeds configured limit")
            note = _object(value, "note")
            tick = _int(note.get("time"), "note.time")
            duration_tick = _int(note.get("duration"), "note.duration")
            grace = _bool(note.get("is_grace", False), "note.is_grace")
            if duration_tick == 0 and not grace:
                raise PdmxAdapterError("invalid_note_duration", "zero-duration note requires explicit grace evidence")
            pitch = _int(note.get("pitch"), "note.pitch", 0, 127)
            spelling = _text(note.get("pitch_str"), "note.pitch_str")
            matched = re.fullmatch(r"([A-G])([#b]*)(?:-?\d+)?", spelling) if spelling else None
            step = matched[1] if matched else None
            alter = matched[2].count("#") - matched[2].count("b") if matched else None
            notes.append(CanonicalNote(
                f"note:{track_index:06d}-{ordinal:09d}", track_id, pitch,
                RationalTime(tick, resolution), RationalTime(duration_tick, resolution),
                None if note.get("velocity") is None else _int(note["velocity"], "note.velocity", 0, 127),
                None, program, percussion, grace, step, alter, None, None, None, None,
                tick, duration_tick, None, None, "prov:projection",
            ))
    if not notes:
        raise PdmxAdapterError("empty_score", "source has no raw notes; chords are not expanded")
    meters = _meters(data, resolution, defaults)
    tempos = _tempos(data, resolution, defaults)
    raw_end = max(note.onset_qn + note.duration_qn for note in notes)
    for name in ("barlines", "beats"):
        raw_end = max(raw_end, *(RationalTime(tick, resolution) for tick, _ in _events(data, name)), RationalTime(0))
    raw_end = max(raw_end, meters[-1].onset_qn, tempos[-1].onset_qn)
    if raw_end == RationalTime(0):
        raw_end = RationalTime(4 * meters[0].numerator, meters[0].denominator)
        defaults.add("point-only-timeline")
    bars, beats, grid_policy = _timeline(data, resolution, raw_end, meters, config)
    metadata = _object({} if data.get("metadata") is None else data["metadata"], "metadata")
    creators = metadata.get("creators")
    if creators is not None:
        creators = tuple(_text(value, "metadata.creator") for value in _list(creators, "metadata.creators"))
        if any(value is None for value in creators):
            raise PdmxAdapterError("invalid_schema", "creator strings must be nonempty")
    piece_id = "piece:pdmx-" + sha256(source_id.encode("utf-8")).hexdigest()
    checksum = sha256(payload).hexdigest()
    provenance = [
        ProvenanceRecord("prov:source", "source", "pdmx_musicrender_json", source_id,
                         None, None, checksum, None, (), ()),
        ProvenanceRecord("prov:projection", "conversion", "music_critic.adapters.pdmx", None,
                         PDMX_SCHEMA_PRIMARY_URLS[0], PDMX_PROJECTION_VERSION, None, None,
                         ("prov:source",), tuple(sorted({
                             "adapter_version": PDMX_ADAPTER_VERSION,
                             "bar_grid_policy": grid_policy,
                             "ignored_annotations": True,
                             "ignored_song_length": True,
                             "integer_number_policy": PDMX_INTEGER_NUMBER_POLICY,
                             "max_decimal_integer_digits": PDMX_MAX_DECIMAL_INTEGER_DIGITS,
                             "raw_feature_registry_version": PDMX_RAW_FEATURE_REGISTRY_VERSION,
                             "schema_evidence_status": PDMX_SCHEMA_EVIDENCE_STATUS,
                             "schema_profile_id": PDMX_SCHEMA_PROFILE_ID,
                             "schema_upstream_revision": PDMX_SCHEMA_UPSTREAM_REVISION,
                             "source_note_identity": "track_index,notes_array_ordinal",
                             "voice_policy": "unavailable_no_inference",
                         }.items()))),
    ]
    flags = []
    for name in sorted(defaults):
        provenance.append(ProvenanceRecord(
            f"prov:default-{name}", "default", "music_critic.adapters.pdmx", None, None,
            PDMX_PROJECTION_VERSION, None, None, ("prov:projection",), (("default", name),),
        ))
        flags.append(QualityFlag("pdmx.explicit_default", "warning", name,
                                 (piece_id,), f"prov:default-{name}"))
    piece = CanonicalPiece(
        SCHEMA_VERSION, piece_id, "pdmx", source_group_id, split, str(path), resolution,
        raw_end, PieceMetadata("json", _text(metadata.get("title"), "metadata.title"),
        creators, _text(metadata.get("collection"), "metadata.collection"), None, None,
        None, _text(metadata.get("copyright"), "metadata.copyright"), None),
        tuple(tracks), tuple(sorted(notes, key=lambda note: (
            note.onset_qn, note.track_id, note.pitch, note.duration_qn, note.note_id))),
        bars, beats, tempos, meters, _keys(data, resolution, raw_end), (), (), tuple(provenance), tuple(flags),
    )
    report = validate_piece(piece)
    if report.errors:
        # Invalid projected canonical data indicate a parser bug, not bad input.
        raise RuntimeError("PDMX projection invariant failed: " + "; ".join(
            f"{issue.code}@{issue.path}" for issue in report.errors[:8]))
    return piece
