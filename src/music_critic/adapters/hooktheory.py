"""Production m-a-p HookTheory TheoryTab to canonical-piece adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import os
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

from music_critic.adapters._json_stream import (
    JSONStreamError,
    find_object_record,
    iter_jsonl,
)
from music_critic.data import (
    SCHEMA_VERSION,
    AnnotationSpan,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
    TargetArray,
    TempoEvent,
    ValidationReport,
    validate_piece,
)


__all__ = [
    "HookTheoryAdapterConfig",
    "HookTheoryAdapterError",
    "convert_hooktheory_record",
    "load_hooktheory_piece",
]


_TRACK_ID = "track:melody"
_DEFAULT_TEMPO = 500_000
_DEFAULT_METER = (4, 4)
_MAX_METRIC_RECORDS = 1_000_000

_SCALE_STEPS = MappingProxyType(
    {
        "major": (2, 2, 1, 2, 2, 2),
        "dorian": (2, 1, 2, 2, 2, 1),
        "phrygian": (1, 2, 2, 2, 1, 2),
        "lydian": (2, 2, 2, 1, 2, 2),
        "mixolydian": (2, 2, 1, 2, 2, 1),
        "minor": (2, 1, 2, 2, 1, 2),
        "locrian": (1, 2, 2, 1, 2, 2),
        "harmonic minor": (2, 1, 2, 2, 1, 3),
        "phrygian dominant": (1, 3, 1, 2, 1, 2),
    }
)
_ACCIDENTAL_OFFSETS = MappingProxyType({"bb": -2, "b": -1, "": 0, "#": 1, "##": 2})
_SCALE_DEGREE_LABELS = tuple(
    f"{accidental}{degree}"
    for degree in range(1, 8)
    for accidental in ("", "b", "#", "bb", "##")
)
_TONIC_LABELS = tuple(str(value) for value in range(12))
_ROOT_LABELS = tuple(str(value) for value in range(7)) + ("bVII",)
_EXTENT_LABELS = ("5", "7", "9", "11", "13")
_INVERSION_LABELS = ("0", "1", "2", "3")
_DECORATION_LABELS = {
    "adds": ("4", "6", "9"),
    "omits": ("3", "5"),
    "alterations": ("b5", "#5", "b9", "#9", "#11", "b13"),
    "suspensions": ("2", "4"),
}
_KNOWN_MODES = frozenset(_SCALE_STEPS)
_NATURAL_TONICS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True, slots=True)
class HookTheoryAdapterConfig:
    dataset_name: str
    include_targets: bool = True


class HookTheoryAdapterError(Exception):
    """Raised when a HookTheory record cannot produce a valid CanonicalPiece."""

    clip_id: str | None
    validation_report: ValidationReport | None

    def __init__(
        self,
        message: str,
        *,
        clip_id: str | None = None,
        validation_report: ValidationReport | None = None,
    ) -> None:
        self.clip_id = clip_id
        self.validation_report = validation_report
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _KeyRegion:
    source_index: int
    raw_onset: Fraction
    onset: RationalTime
    tonic_pc: int | None
    mode: str | None


@dataclass(frozen=True, slots=True)
class _RawMeterRegion:
    source_index: int
    raw_onset: Fraction
    numerator: int
    denominator: int
    qn_per_raw_beat: Fraction


@dataclass(frozen=True, slots=True)
class _MappedMeterRegion:
    raw: _RawMeterRegion
    canonical_onset: Fraction


@dataclass(frozen=True, slots=True)
class _HookTheoryTimeline:
    regions: tuple[_MappedMeterRegion, ...]

    def raw_beat_to_fraction(self, raw_beat: Fraction) -> Fraction:
        active = self.regions[0]
        for region in self.regions[1:]:
            if region.raw.raw_onset > raw_beat:
                break
            active = region
        return active.canonical_onset + (
            raw_beat - active.raw.raw_onset
        ) * active.raw.qn_per_raw_beat

    def raw_beat_to_qn(self, raw_beat: Fraction) -> RationalTime:
        return RationalTime.from_fraction(self.raw_beat_to_fraction(raw_beat))

    def raw_interval_to_qn(
        self, raw_onset: Fraction, raw_duration: Fraction
    ) -> tuple[RationalTime, RationalTime]:
        onset = self.raw_beat_to_fraction(raw_onset)
        end = self.raw_beat_to_fraction(raw_onset + raw_duration)
        return RationalTime.from_fraction(onset), RationalTime.from_fraction(end - onset)

    def active(self, raw_beat: Fraction) -> _RawMeterRegion:
        active = self.regions[0].raw
        for region in self.regions[1:]:
            if region.raw.raw_onset > raw_beat:
                break
            active = region.raw
        return active


@dataclass(frozen=True, slots=True)
class _ValidatedStructureMatch:
    clip_id: str
    ori_uid: str | None
    audio_path: str
    split: str | None
    row: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _MelodyValue:
    source_index: int
    note: CanonicalNote
    scale_degree: str


@dataclass(frozen=True, slots=True)
class _ChordRegion:
    source_index: int
    onset: RationalTime
    duration: RationalTime
    raw: Mapping[str, Any]
    annotation_id: str


def _flag(
    code: str,
    message: str,
    piece_id: str,
    *,
    entity_id: str | None = None,
    provenance_id: str = "prov:conversion",
) -> QualityFlag:
    return QualityFlag(
        code=code,
        severity="warning",
        message=message,
        entity_ids=(entity_id or piece_id,),
        provenance_id=provenance_id,
    )


def _fraction(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError("exact HookTheory numbers must be int or Decimal")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact HookTheory number") from exc


def _normalize_split(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "valid":
        return "val"
    return normalized if normalized in {"train", "val", "test"} else None


def _tonic_pc(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or token[0].upper() not in _NATURAL_TONICS:
        return None
    pitch = _NATURAL_TONICS[token[0].upper()]
    for accidental in token[1:]:
        if accidental == "#":
            pitch += 1
        elif accidental in {"b", "♭"}:
            pitch -= 1
        else:
            return None
    return pitch % 12


def _normalized_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    expanded: list[str] = []
    for index, character in enumerate(value.strip().replace("_", " ")):
        if index and character.isupper() and expanded and expanded[-1] != " ":
            expanded.append(" ")
        expanded.append(character.lower())
    normalized = " ".join("".join(expanded).split())
    return normalized or None


def _scale_degree_relative(
    scale_degree: Any,
    octave: Any,
    tonic: int,
    mode: str,
) -> int | None:
    if (
        not isinstance(scale_degree, str)
        or not scale_degree
        or scale_degree[-1] not in "1234567"
        or scale_degree[:-1] not in _ACCIDENTAL_OFFSETS
        or isinstance(octave, bool)
        or not isinstance(octave, int)
    ):
        return None
    steps = _SCALE_STEPS.get(mode)
    if steps is None:
        return None
    degree = int(scale_degree[-1]) - 1
    return (
        12 * octave
        + tonic
        + sum(steps[:degree])
        + _ACCIDENTAL_OFFSETS[scale_degree[:-1]]
    )


def _round_half_up(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + (1 if remainder * 2 >= value.denominator else 0)


def _tempo_value(bpm: Any, meter: _RawMeterRegion) -> int:
    exact_bpm = _fraction(bpm)
    if exact_bpm <= 0:
        raise ValueError("BPM must be positive")
    numerator = 40_000_000 if meter.denominator == 8 else 60_000_000
    return _round_half_up(Fraction(numerator, 1) / exact_bpm)


def _source_path_valid(source_path: str | None) -> bool:
    if source_path is None:
        return True
    try:
        path = Path(source_path)
    except (TypeError, ValueError):
        return False
    return path.name == "4_merged.json" and "htcanon" not in {
        part.lower() for part in path.parts
    }


def _select_tempos(
    payload: Mapping[str, Any], timeline: _HookTheoryTimeline, piece_id: str
) -> tuple[tuple[TempoEvent, ...], list[QualityFlag], bool]:
    flags: list[QualityFlag] = []
    candidates: list[tuple[RationalTime, int, int]] = []
    raw_tempos = payload.get("tempos")
    if isinstance(raw_tempos, Sequence) and not isinstance(raw_tempos, (str, bytes)):
        for source_index, raw in enumerate(raw_tempos):
            try:
                if not isinstance(raw, Mapping):
                    raise ValueError("tempo region must be an object")
                raw_onset = _fraction(raw.get("beat"))
                if raw_onset < 1:
                    raise ValueError("tempo onset must be non-negative")
                onset = timeline.raw_beat_to_qn(raw_onset)
                value = _tempo_value(raw.get("bpm"), timeline.active(raw_onset))
            except (TypeError, ValueError):
                flags.append(
                    _flag(
                        "hooktheory.tempo_invalid",
                        f"Omitted invalid tempo region at source index {source_index}.",
                        piece_id,
                    )
                )
                continue
            candidates.append((onset, source_index, value))
    elif raw_tempos is not None:
        flags.append(
            _flag(
                "hooktheory.tempo_invalid",
                "Omitted malformed tempo collection.",
                piece_id,
            )
        )

    selected: list[tuple[RationalTime, int, int]] = []
    for candidate in sorted(candidates):
        onset, source_index, value = candidate
        if selected and selected[-1][0] == onset:
            if selected[-1][2] != value:
                flags.append(
                    _flag(
                        "hooktheory.tempo_conflict",
                        f"Conflicting tempo at {onset.num}/{onset.den}; kept first source value.",
                        piece_id,
                    )
                )
            continue
        if selected and selected[-1][2] == value:
            continue
        selected.append((onset, source_index, value))

    defaulted = not selected or selected[0][0] != RationalTime(0)
    if defaulted:
        selected.insert(0, (RationalTime(0), -1, _DEFAULT_TEMPO))
        flags.append(
            _flag(
                "hooktheory.default_tempo",
                "Inserted default tempo 500000 microseconds per quarter at onset zero.",
                piece_id,
                provenance_id="prov:default-tempo",
            )
        )
    events = tuple(
        TempoEvent(
            tempo_event_id=f"tempo:{index:04d}",
            onset_qn=onset,
            microseconds_per_quarter=value,
            provenance_id="prov:default-tempo" if source_index < 0 else "prov:source",
        )
        for index, (onset, source_index, value) in enumerate(selected)
    )
    return events, flags, defaulted


def _select_meters(
    payload: Mapping[str, Any], piece_id: str
) -> tuple[_HookTheoryTimeline, tuple[MeterEvent, ...], list[QualityFlag], bool]:
    flags: list[QualityFlag] = []
    candidates: list[_RawMeterRegion] = []
    raw_meters = payload.get("meters")
    if isinstance(raw_meters, Sequence) and not isinstance(raw_meters, (str, bytes)):
        for source_index, raw in enumerate(raw_meters):
            try:
                if not isinstance(raw, Mapping):
                    raise ValueError("meter region must be an object")
                raw_onset = _fraction(raw.get("beat"))
                numerator = raw.get("numBeats")
                beat_unit = raw.get("beatUnit")
                if (
                    isinstance(numerator, bool)
                    or not isinstance(numerator, int)
                    or numerator <= 0
                    or beat_unit not in {1, 3}
                    or isinstance(beat_unit, bool)
                    or raw_onset < 1
                ):
                    raise ValueError("unsupported meter")
                denominator = 4 if beat_unit == 1 else 8
            except (TypeError, ValueError):
                flags.append(
                    _flag(
                        "hooktheory.meter_invalid",
                        f"Omitted invalid meter region at source index {source_index}.",
                        piece_id,
                    )
                )
                continue
            candidates.append(
                _RawMeterRegion(
                    source_index=source_index,
                    raw_onset=raw_onset,
                    numerator=numerator,
                    denominator=denominator,
                    qn_per_raw_beat=Fraction(4, denominator),
                )
            )
    elif raw_meters is not None:
        flags.append(
            _flag(
                "hooktheory.meter_invalid",
                "Omitted malformed meter collection.",
                piece_id,
            )
        )

    selected: list[_RawMeterRegion] = []
    for candidate in sorted(candidates, key=lambda value: (value.raw_onset, value.source_index)):
        if selected and selected[-1].raw_onset == candidate.raw_onset:
            if (selected[-1].numerator, selected[-1].denominator) != (
                candidate.numerator,
                candidate.denominator,
            ):
                flags.append(
                    _flag(
                        "hooktheory.meter_conflict",
                        f"Conflicting meter at raw beat {candidate.raw_onset}; kept first source value.",
                        piece_id,
                    )
                )
            continue
        if selected and (selected[-1].numerator, selected[-1].denominator) == (
            candidate.numerator,
            candidate.denominator,
        ):
            continue
        selected.append(candidate)

    defaulted = not selected or selected[0].raw_onset != 1
    if defaulted:
        selected.insert(
            0,
            _RawMeterRegion(-1, Fraction(1), 4, 4, Fraction(1)),
        )
        flags.append(
            _flag(
                "hooktheory.default_meter",
                "Inserted default meter 4/4 at onset zero.",
                piece_id,
                provenance_id="prov:default-meter",
            )
        )
    mapped: list[_MappedMeterRegion] = [
        _MappedMeterRegion(selected[0], Fraction(0))
    ]
    for region in selected[1:]:
        previous = mapped[-1]
        mapped.append(
            _MappedMeterRegion(
                region,
                previous.canonical_onset
                + (region.raw_onset - previous.raw.raw_onset)
                * previous.raw.qn_per_raw_beat,
            )
        )
    timeline = _HookTheoryTimeline(tuple(mapped))
    events = tuple(
        MeterEvent(
            meter_event_id=f"meter:{index:04d}",
            onset_qn=RationalTime.from_fraction(region.canonical_onset),
            numerator=region.raw.numerator,
            denominator=region.raw.denominator,
            provenance_id="prov:default-meter" if region.raw.source_index < 0 else "prov:source",
        )
        for index, region in enumerate(mapped)
    )
    return timeline, events, flags, defaulted


def _key_regions(
    payload: Mapping[str, Any], timeline: _HookTheoryTimeline, piece_id: str
) -> tuple[list[_KeyRegion], list[QualityFlag]]:
    regions: list[_KeyRegion] = []
    flags: list[QualityFlag] = []
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
        if raw_keys is not None:
            flags.append(
                _flag("hooktheory.key_timing_invalid", "Malformed key collection.", piece_id)
            )
        return regions, flags
    for source_index, raw in enumerate(raw_keys):
        if not isinstance(raw, Mapping):
            flags.append(
                _flag(
                    "hooktheory.key_timing_invalid",
                    f"Omitted malformed key region at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        try:
            raw_onset = _fraction(raw.get("beat"))
            if raw_onset < 1:
                raise ValueError
            onset = timeline.raw_beat_to_qn(raw_onset)
        except (TypeError, ValueError):
            flags.append(
                _flag(
                    "hooktheory.key_timing_invalid",
                    f"Omitted key region with invalid timing at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        tonic = _tonic_pc(raw.get("tonic"))
        mode = _normalized_mode(raw.get("scale"))
        if tonic is None:
            flags.append(
                _flag(
                    "hooktheory.key_tonic_invalid",
                    f"Key region {source_index} has an unavailable tonic.",
                    piece_id,
                )
            )
        if mode is None or mode not in _SCALE_STEPS:
            flags.append(
                _flag(
                    "hooktheory.key_mode_invalid",
                    f"Key region {source_index} has an unavailable or unsupported mode.",
                    piece_id,
                )
            )
        regions.append(_KeyRegion(source_index, raw_onset, onset, tonic, mode))
    selected: list[_KeyRegion] = []
    for region in sorted(regions, key=lambda value: (value.raw_onset, value.source_index)):
        if selected and selected[-1].raw_onset == region.raw_onset:
            previous_valid = (
                selected[-1].tonic_pc is not None
                and selected[-1].mode in _SCALE_STEPS
            )
            current_valid = region.tonic_pc is not None and region.mode in _SCALE_STEPS
            if not previous_valid and current_valid:
                selected[-1] = region
            elif previous_valid and current_valid and (
                selected[-1].tonic_pc,
                selected[-1].mode,
            ) != (region.tonic_pc, region.mode):
                flags.append(
                    _flag(
                        "hooktheory.key_conflict",
                        f"Conflicting key at raw beat {region.raw_onset}; kept first source value.",
                        piece_id,
                    )
                )
            continue
        selected.append(region)
    return selected, flags


def _active_key(keys: Sequence[_KeyRegion], raw_onset: Fraction) -> _KeyRegion | None:
    active: _KeyRegion | None = None
    for key in keys:
        if key.raw_onset > raw_onset:
            break
        if active is None or key.raw_onset > active.raw_onset:
            active = key
    return active


def _make_notes(
    payload: Mapping[str, Any],
    keys: Sequence[_KeyRegion],
    timeline: _HookTheoryTimeline,
    piece_id: str,
) -> tuple[list[_MelodyValue], list[QualityFlag], RationalTime]:
    values: list[_MelodyValue] = []
    flags: list[QualityFlag] = []
    maximum = RationalTime(0)
    raw_notes = payload.get("notes")
    if not isinstance(raw_notes, Sequence) or isinstance(raw_notes, (str, bytes)):
        if raw_notes is not None:
            flags.append(
                _flag("hooktheory.note_timing_invalid", "Malformed note collection.", piece_id)
            )
        return values, flags, maximum
    for source_index, raw in enumerate(raw_notes):
        if not isinstance(raw, Mapping):
            flags.append(
                _flag(
                    "hooktheory.note_timing_invalid",
                    f"Omitted malformed note at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        try:
            raw_onset = _fraction(raw.get("beat"))
            raw_duration = _fraction(raw.get("duration"))
            if raw_onset < 1:
                raise ValueError
            onset, duration = timeline.raw_interval_to_qn(raw_onset, raw_duration)
        except (TypeError, ValueError):
            flags.append(
                _flag(
                    "hooktheory.note_timing_invalid",
                    f"Omitted note with invalid timing at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        if duration <= RationalTime(0):
            flags.append(
                _flag(
                    "hooktheory.note_duration_invalid",
                    f"Omitted non-positive note at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        maximum = max(maximum, onset + duration)
        if raw.get("isRest") is True:
            continue
        octave = raw.get("octave")
        if isinstance(octave, bool) or not isinstance(octave, int):
            flags.append(
                _flag(
                    "hooktheory.note_octave_missing",
                    f"Omitted note without an integer octave at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        scale_degree = raw.get("sd")
        if not isinstance(scale_degree, str) or scale_degree not in _SCALE_DEGREE_LABELS:
            flags.append(
                _flag(
                    "hooktheory.scale_degree_unsupported",
                    f"Omitted note with unsupported scale degree at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        active_key = _active_key(keys, raw_onset)
        if (
            active_key is None
            or active_key.tonic_pc is None
            or active_key.mode not in _SCALE_STEPS
        ):
            flags.append(
                _flag(
                    "hooktheory.pitch_active_key_unresolved",
                    f"Omitted note without an active key at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        relative = _scale_degree_relative(
            scale_degree,
            octave,
            active_key.tonic_pc,
            active_key.mode,
        )
        if relative is None:
            flags.append(
                _flag(
                    "hooktheory.pitch_active_key_unresolved",
                    f"Omitted note with unresolved scale semantics at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        pitch = 60 + relative
        if not 0 <= pitch <= 127:
            flags.append(
                _flag(
                    "hooktheory.pitch_out_of_range",
                    f"Omitted derived pitch {pitch} at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        note_id = f"note:melody-{source_index:06d}"
        note = CanonicalNote(
            note_id=note_id,
            track_id=_TRACK_ID,
            pitch=pitch,
            onset_qn=onset,
            duration_qn=duration,
            velocity=None,
            channel=None,
            program=None,
            is_percussion=False,
            is_grace=False,
            spelling_step=None,
            spelling_alter=None,
            staff=None,
            voice=None,
            articulations=None,
            dynamic=None,
            source_onset_ticks=None,
            source_duration_ticks=None,
            source_onset_seconds=None,
            source_duration_seconds=None,
            provenance_id="prov:pitch",
        )
        values.append(_MelodyValue(source_index, note, scale_degree))
    values.sort(
        key=lambda value: (
            value.note.onset_qn,
            value.note.pitch,
            value.note.duration_qn,
            value.source_index,
        )
    )
    return values, flags, maximum


def _chord_regions(
    payload: Mapping[str, Any], timeline: _HookTheoryTimeline, piece_id: str
) -> tuple[list[_ChordRegion], list[QualityFlag], RationalTime]:
    regions: list[_ChordRegion] = []
    flags: list[QualityFlag] = []
    maximum = RationalTime(0)
    raw_chords = payload.get("chords")
    if not isinstance(raw_chords, Sequence) or isinstance(raw_chords, (str, bytes)):
        if raw_chords is not None:
            flags.append(
                _flag("hooktheory.chord_timing_invalid", "Malformed chord collection.", piece_id)
            )
        return regions, flags, maximum
    for source_index, raw in enumerate(raw_chords):
        if not isinstance(raw, Mapping):
            flags.append(
                _flag(
                    "hooktheory.chord_timing_invalid",
                    f"Omitted malformed chord at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        try:
            raw_onset = _fraction(raw.get("beat"))
            raw_duration = _fraction(raw.get("duration"))
            if raw_onset < 1 or raw_duration <= 0:
                raise ValueError
            onset, duration = timeline.raw_interval_to_qn(raw_onset, raw_duration)
        except (TypeError, ValueError):
            flags.append(
                _flag(
                    "hooktheory.chord_timing_invalid",
                    f"Omitted chord with invalid timing at source index {source_index}.",
                    piece_id,
                )
            )
            continue
        annotation_id = f"span:chord-{source_index:06d}"
        maximum = max(maximum, onset + duration)
        regions.append(_ChordRegion(source_index, onset, duration, raw, annotation_id))
        if raw.get("applied") not in {None, 0}:
            flags.append(
                _flag(
                    "hooktheory.applied_deferred",
                    f"Deferred applied harmony at chord source index {source_index}.",
                    piece_id,
                )
            )
        if raw.get("alternate") == "_":
            flags.append(
                _flag(
                    "hooktheory.alternate_unresolved",
                    f"Unresolved alternate marker at chord source index {source_index}.",
                    piece_id,
                )
            )
        if raw.get("pedal") is not None:
            flags.append(
                _flag(
                    "hooktheory.pedal_unresolved",
                    f"Unresolved pedal value at chord source index {source_index}.",
                    piece_id,
                )
            )
    regions.sort(key=lambda value: (value.onset, value.source_index))
    return regions, flags, maximum


def _raw_duration(
    payload: Mapping[str, Any],
    timeline: _HookTheoryTimeline,
    content_end: RationalTime,
    piece_id: str,
) -> tuple[RationalTime, list[QualityFlag]]:
    flags: list[QualityFlag] = []
    try:
        raw_end = _fraction(payload.get("endBeat"))
        if raw_end < 1:
            raise ValueError
        duration = timeline.raw_beat_to_qn(raw_end)
    except (TypeError, ValueError):
        duration = content_end
        flags.append(
            _flag(
                "hooktheory.duration_derived",
                "Derived piece duration because endBeat was missing or invalid.",
                piece_id,
            )
        )
    if duration < content_end:
        duration = content_end
        flags.append(
            _flag(
                "hooktheory.duration_extended",
                "Extended piece duration to contain valid source events.",
                piece_id,
            )
        )
    return duration, flags


def _active_meter(meters: Sequence[MeterEvent], onset: RationalTime) -> MeterEvent:
    active = meters[0]
    for meter in meters[1:]:
        if meter.onset_qn > onset:
            break
        active = meter
    return active


def _bars_and_beats(
    duration: RationalTime,
    meters: Sequence[MeterEvent],
    piece_id: str,
) -> tuple[tuple[CanonicalBar, ...], tuple[CanonicalBeat, ...], list[QualityFlag]]:
    bars: list[CanonicalBar] = []
    beats: list[CanonicalBeat] = []
    flags: list[QualityFlag] = []
    cursor = RationalTime(0)
    while cursor < duration:
        meter = _active_meter(meters, cursor)
        nominal = RationalTime(meter.numerator * 4, meter.denominator)
        next_meter = next(
            (value.onset_qn for value in meters if value.onset_qn > cursor),
            duration,
        )
        interval_end = min(duration, next_meter)
        if interval_end <= cursor:
            break
        bar_duration = min(nominal, interval_end - cursor)
        incomplete = bar_duration < nominal
        bar_index = len(bars)
        bar_id = f"bar:{bar_index:06d}"
        bars.append(
            CanonicalBar(
                bar_id=bar_id,
                index=bar_index,
                start_qn=cursor,
                duration_qn=bar_duration,
                meter_event_id=meter.meter_event_id,
                metric_offset_qn=RationalTime(0),
                is_pickup=False,
                is_incomplete=incomplete,
                display_number=str(bar_index + 1),
                provenance_id="prov:conversion",
            )
        )
        if incomplete and cursor + bar_duration < duration:
            flags.append(
                _flag(
                    "hooktheory.meter_change_incomplete_bar",
                    "Meter change ended the preceding canonical bar before its nominal duration.",
                    piece_id,
                    entity_id=bar_id,
                )
            )
        unit = RationalTime(4, meter.denominator)
        position = RationalTime(0)
        beat_index = 0
        while position < bar_duration:
            beat_duration = min(unit, bar_duration - position)
            beats.append(
                CanonicalBeat(
                    beat_id=f"beat:{len(beats):08d}",
                    bar_id=bar_id,
                    meter_event_id=meter.meter_event_id,
                    index_in_bar=beat_index,
                    start_qn=cursor + position,
                    duration_qn=beat_duration,
                    position_in_bar_qn=position,
                    is_downbeat=beat_index == 0,
                    strength=1.0 if beat_index == 0 else 0.5,
                    provenance_id="prov:conversion",
                )
            )
            if len(bars) + len(beats) > _MAX_METRIC_RECORDS:
                raise HookTheoryAdapterError(
                    "metric-grid safety limit exceeded",
                    clip_id=piece_id.removeprefix("piece:hooktheory-"),
                )
            position = position + unit
            beat_index += 1
        cursor = cursor + bar_duration
    return tuple(bars), tuple(beats), flags


def _available(value: Any) -> tuple[Any, bool, None, str | None, str | None]:
    if value is None:
        return None, False, None, None, None
    return value, True, None, "dataset", "prov:annotation"


def _target(
    task: str,
    alignment_type: str,
    entity_ids: Sequence[str],
    value_type: str,
    class_labels: tuple[str, ...] | None,
    entries: Sequence[tuple[Any, bool, None, str | None, str | None]],
) -> TargetArray:
    return TargetArray(
        target_id=f"target:{task}",
        task=task,
        annotation_view_id=None,
        alignment_type=alignment_type,  # type: ignore[arg-type]
        entity_ids=tuple(entity_ids),
        value_type=value_type,  # type: ignore[arg-type]
        class_labels=class_labels,
        values=tuple(entry[0] for entry in entries),
        mask=tuple(entry[1] for entry in entries),
        confidence=tuple(entry[2] for entry in entries),
        source=tuple(entry[3] for entry in entries),  # type: ignore[arg-type]
        provenance=tuple(entry[4] for entry in entries),
    )


def _multi_label(
    value: Any,
    labels: tuple[str, ...],
) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized: set[str] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            return None
        token = str(item)
        if token not in labels:
            return None
        normalized.add(token)
    return tuple(label for label in labels if label in normalized)


def _borrowed(value: Any) -> tuple[str | None, bool]:
    if value is None or value == "":
        return "none", True
    if isinstance(value, str):
        normalized = _normalized_mode(value)
        if normalized is None:
            return None, False
        if normalized in _KNOWN_MODES:
            return f"mode:{normalized}", True
        return f"unknown:{value}", True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pcs: set[int] = set()
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                return None, False
            pcs.add(item % 12)
        return "pcset:" + ",".join(str(item) for item in sorted(pcs)), True
    return None, False


def _annotations_and_targets(
    keys: Sequence[_KeyRegion],
    melody: Sequence[_MelodyValue],
    chords: Sequence[_ChordRegion],
    duration: RationalTime,
    piece_id: str,
) -> tuple[tuple[AnnotationSpan, ...], tuple[TargetArray, ...], list[QualityFlag]]:
    annotations: list[AnnotationSpan] = []
    targets: list[TargetArray] = []
    flags: list[QualityFlag] = []

    key_ids: list[str] = []
    tonic_entries = []
    mode_entries = []
    for index, key in enumerate(keys):
        annotation_id = f"span:key-{key.source_index:06d}"
        end = keys[index + 1].onset if index + 1 < len(keys) else duration
        end = max(key.onset, min(end, duration))
        annotations.append(
            AnnotationSpan(
                annotation_id=annotation_id,
                annotation_type="theory.local_key",
                layer="target_alignment",
                start_qn=key.onset,
                end_qn=end,
                track_id=None,
                value=None,
                provenance_id="prov:annotation",
            )
        )
        key_ids.append(annotation_id)
        tonic_entries.append(_available(None if key.tonic_pc is None else str(key.tonic_pc)))
        mode_entries.append(_available(key.mode))
    targets.extend(
        (
            _target(
                "theory.local_key.tonic_pc",
                "annotation_span",
                key_ids,
                "categorical",
                _TONIC_LABELS,
                tonic_entries,
            ),
            _target(
                "theory.local_key.mode",
                "annotation_span",
                key_ids,
                "categorical",
                None,
                mode_entries,
            ),
        )
    )

    targets.append(
        _target(
            "theory.melody.scale_degree",
            "note",
            [value.note.note_id for value in melody],
            "categorical",
            _SCALE_DEGREE_LABELS,
            [_available(value.scale_degree) for value in melody],
        )
    )

    chord_ids: list[str] = []
    chord_values: dict[str, list[tuple[Any, bool, None, str | None, str | None]]] = {
        name: []
        for name in (
            "presence",
            "root_degree",
            "extent",
            "inversion",
            "adds",
            "omits",
            "alterations",
            "suspensions",
            "borrowed",
        )
    }
    for chord in chords:
        annotations.append(
            AnnotationSpan(
                annotation_id=chord.annotation_id,
                annotation_type="theory.chord",
                layer="target_alignment",
                start_qn=chord.onset,
                end_qn=chord.onset + chord.duration,
                track_id=None,
                value=None,
                provenance_id="prov:annotation",
            )
        )
        chord_ids.append(chord.annotation_id)
        raw = chord.raw
        is_rest = raw.get("isRest") is True
        chord_values["presence"].append(_available("false" if is_rest else "true"))

        root = raw.get("root")
        root_value: str | None = None
        if not is_rest and isinstance(root, int) and not isinstance(root, bool):
            if 1 <= root <= 7:
                root_value = str(root - 1)
            elif root == 8:
                root_value = "bVII"
        if not is_rest and root_value is None:
            code = (
                "hooktheory.chord_root_zero_non_rest"
                if root == 0
                else "hooktheory.chord_root_invalid"
            )
            flags.append(
                _flag(
                    code,
                    f"Unavailable root at chord source index {chord.source_index}.",
                    piece_id,
                )
            )
        elif is_rest and isinstance(root, int) and root < 0:
            flags.append(
                _flag(
                    "hooktheory.negative_root_rest_anomaly",
                    f"Preserved negative root anomaly on rest chord {chord.source_index}.",
                    piece_id,
                )
            )
        chord_values["root_degree"].append(_available(root_value))

        extent = raw.get("type")
        extent_value = str(extent) if not is_rest and str(extent) in _EXTENT_LABELS else None
        chord_values["extent"].append(_available(extent_value))
        inversion = raw.get("inversion")
        inversion_value = (
            str(inversion) if not is_rest and str(inversion) in _INVERSION_LABELS else None
        )
        chord_values["inversion"].append(_available(inversion_value))

        for name, labels in _DECORATION_LABELS.items():
            normalized = None if is_rest else _multi_label(raw.get(name), labels)
            if normalized is None and not is_rest:
                flags.append(
                    _flag(
                        f"hooktheory.chord_{name}_invalid",
                        f"Unavailable {name} at chord source index {chord.source_index}.",
                        piece_id,
                    )
                )
            chord_values[name].append(_available(normalized))

        borrowed, borrowed_valid = _borrowed(raw.get("borrowed"))
        if is_rest:
            borrowed, borrowed_valid = None, False
        if not borrowed_valid and not is_rest:
            flags.append(
                _flag(
                    "hooktheory.borrowed_invalid",
                    f"Unavailable borrowed value at chord source index {chord.source_index}.",
                    piece_id,
                )
            )
        elif (
            not is_rest
            and isinstance(raw.get("borrowed"), str)
            and borrowed is not None
            and borrowed.startswith("unknown:")
        ):
            flags.append(
                _flag(
                    "hooktheory.borrowed_unknown_string",
                    f"Preserved unknown borrowed string at chord source index {chord.source_index}.",
                    piece_id,
                )
            )
        chord_values["borrowed"].append(_available(borrowed if borrowed_valid else None))

    specifications = (
        ("presence", "categorical", ("false", "true")),
        ("root_degree", "categorical", _ROOT_LABELS),
        ("extent", "categorical", _EXTENT_LABELS),
        ("inversion", "categorical", _INVERSION_LABELS),
        ("adds", "multi_label", _DECORATION_LABELS["adds"]),
        ("omits", "multi_label", _DECORATION_LABELS["omits"]),
        ("alterations", "multi_label", _DECORATION_LABELS["alterations"]),
        ("suspensions", "multi_label", _DECORATION_LABELS["suspensions"]),
        ("borrowed", "categorical", None),
    )
    for name, value_type, labels in specifications:
        targets.append(
            _target(
                f"theory.chord.{name}",
                "annotation_span",
                chord_ids,
                value_type,
                labels,
                chord_values[name],
            )
        )
    annotations.sort(key=lambda value: (value.start_qn, value.end_qn, value.annotation_id))
    targets.sort(
        key=lambda value: (
            value.task,
            value.annotation_view_id is not None,
            value.annotation_view_id or "",
            value.target_id,
        )
    )
    return tuple(annotations), tuple(targets), flags


def _provenance(
    clip_id: str,
    source_path: str | None,
    *,
    structure_match: _ValidatedStructureMatch | None,
    default_tempo: bool,
    default_meter: bool,
    include_targets: bool,
) -> tuple[ProvenanceRecord, ...]:
    records = [
        ProvenanceRecord(
            provenance_id="prov:source",
            kind="source",
            source="map_raw_theorytab_source",
            record_id=clip_id,
            uri=source_path,
            version=None,
            checksum_sha256=None,
            created_at=None,
            parents=(),
            details=(("numeric_lexemes", "int_or_decimal"),),
        )
    ]
    conversion_parents = ["prov:source"]
    if structure_match is not None:
        records.append(
            ProvenanceRecord(
                provenance_id="prov:structure",
                kind="source",
                source="hooktheory_structure_jsonl",
                record_id=clip_id,
                uri=None,
                version=None,
                checksum_sha256=None,
                created_at=None,
                parents=(),
                details=(
                    ("coordinate_unit", "audio_seconds"),
                    ("ori_uid_available", bool(structure_match.ori_uid)),
                ),
            )
        )
        conversion_parents.append("prov:structure")
    records.append(
        ProvenanceRecord(
            provenance_id="prov:conversion",
            kind="conversion",
            source="music_critic.adapters.hooktheory",
            record_id=None,
            uri=None,
            version="1",
            checksum_sha256=None,
            created_at=None,
            parents=tuple(sorted(conversion_parents)),
            details=(
                ("bpm_rounding", "nearest_integer_half_up"),
                ("meter_mapping", "numBeats;beatUnit1=4;beatUnit3=8"),
                ("onset", "piecewise_raw_beat_to_qn;beatUnit1=1;beatUnit3=1/2"),
                ("source_order", "raw_event_index"),
                ("tempo", "quarter_bpm_simple;felt_pulse_bpm_compound"),
            ),
        )
    )
    if include_targets:
        records.append(
            ProvenanceRecord(
                provenance_id="prov:annotation",
                kind="annotation",
                source="map_hooktheory_theorytab_annotations",
                record_id=clip_id,
                uri=source_path,
                version=None,
                checksum_sha256=None,
                created_at=None,
                parents=("prov:conversion",),
                details=(("annotation_view_id", None), ("source", "dataset")),
            )
        )
    if default_meter:
        records.append(
            ProvenanceRecord(
                provenance_id="prov:default-meter",
                kind="default",
                source="music_critic.adapters.hooktheory",
                record_id=None,
                uri=None,
                version="1",
                checksum_sha256=None,
                created_at=None,
                parents=("prov:conversion",),
                details=(("meter", "4/4"),),
            )
        )
    if default_tempo:
        records.append(
            ProvenanceRecord(
                provenance_id="prov:default-tempo",
                kind="default",
                source="music_critic.adapters.hooktheory",
                record_id=None,
                uri=None,
                version="1",
                checksum_sha256=None,
                created_at=None,
                parents=("prov:conversion",),
                details=(("microseconds_per_quarter", _DEFAULT_TEMPO),),
            )
        )
    records.append(
        ProvenanceRecord(
            provenance_id="prov:pitch",
            kind="derivation",
            source="hooktheory_scale_degree_to_midi_upstream",
            record_id=None,
            uri=None,
            version="1",
            checksum_sha256=None,
            created_at=None,
            parents=("prov:conversion",),
            details=(
                ("classification", "upstream_semantics"),
                ("formula", "60+12*octave+tonic_pc+active_scale_offset+accidental"),
            ),
        )
    )
    by_id = {record.provenance_id: record for record in records}
    remaining = set(by_id)
    emitted: set[str] = set()
    ordered: list[ProvenanceRecord] = []
    while remaining:
        ready = sorted(
            identifier
            for identifier in remaining
            if all(parent in emitted for parent in by_id[identifier].parents)
        )
        if not ready:
            raise RuntimeError("internal provenance cycle")
        identifier = ready[0]
        ordered.append(by_id[identifier])
        emitted.add(identifier)
        remaining.remove(identifier)
    return tuple(ordered)


def _validate_structure_match(
    structure_row: Mapping[str, Any] | None,
    clip_id: str,
    symbolic_split: str | None,
) -> _ValidatedStructureMatch | None:
    if structure_row is None:
        return None
    if not isinstance(structure_row, Mapping):
        raise HookTheoryAdapterError(
            "structure row belongs to another clip: row is not a mapping",
            clip_id=clip_id,
        )
    audio_path = structure_row.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path or Path(audio_path).stem != clip_id:
        raise HookTheoryAdapterError(
            "structure row belongs to another clip: audio_path stem does not match clip_id",
            clip_id=clip_id,
        )
    structure_split = None
    for field in ("split", "dataset_split", "partition"):
        if field not in structure_row:
            continue
        structure_split = _normalize_split(structure_row.get(field))
        if structure_split != symbolic_split:
            raise HookTheoryAdapterError(
                f"structure row split mismatch in {field!r}",
                clip_id=clip_id,
            )
        break
    ori_uid = structure_row.get("ori_uid")
    normalized_ori_uid = ori_uid.strip() if isinstance(ori_uid, str) and ori_uid.strip() else None
    return _ValidatedStructureMatch(
        clip_id=clip_id,
        ori_uid=normalized_ori_uid,
        audio_path=audio_path,
        split=structure_split,
        row=structure_row,
    )


def _structure_flags(
    structure_match: _ValidatedStructureMatch | None, piece_id: str
) -> tuple[str, list[QualityFlag]]:
    if structure_match is None:
        return piece_id, [
            _flag(
                "hooktheory.structure_unmatched_symbolic_clip",
                "No matching HookTheoryStructure row was supplied.",
                piece_id,
            )
        ]
    flags = [
        _flag(
            "hooktheory.structure_alignment_unresolved",
            "Structure coordinates remain audio seconds and were not converted to symbolic spans.",
            piece_id,
            provenance_id="prov:structure",
        )
    ]
    if structure_match.ori_uid is not None:
        return structure_match.ori_uid, flags
    flags.append(
        _flag(
            "hooktheory.structure_missing_ori_uid",
            "Structure row has no usable ori_uid; using a unique per-clip group.",
            piece_id,
            provenance_id="prov:structure",
        )
    )
    return piece_id, flags


def _validation_error(clip_id: str, report: ValidationReport) -> HookTheoryAdapterError:
    examples = ", ".join(f"{issue.code}@{issue.path}" for issue in report.errors[:5])
    message = f"HookTheory clip {clip_id!r} failed canonical validation"
    if examples:
        message += f"; {examples}"
    return HookTheoryAdapterError(message, clip_id=clip_id, validation_report=report)


def convert_hooktheory_record(
    clip_id: str,
    record: Mapping[str, Any],
    *,
    config: HookTheoryAdapterConfig,
    structure_row: Mapping[str, Any] | None = None,
    source_path: str | None = None,
) -> CanonicalPiece:
    """Convert one usable raw m-a-p HookTheory record into a validated piece."""

    if not isinstance(clip_id, str) or not clip_id:
        raise HookTheoryAdapterError("clip_id must be a non-empty string", clip_id=None)
    if not isinstance(record, Mapping):
        raise HookTheoryAdapterError("HookTheory record must be a mapping", clip_id=clip_id)
    if not isinstance(config, HookTheoryAdapterConfig):
        raise HookTheoryAdapterError("config must be HookTheoryAdapterConfig", clip_id=clip_id)
    if not _source_path_valid(source_path):
        raise HookTheoryAdapterError(
            "source_path must refer to the raw 4_merged.json source",
            clip_id=clip_id,
        )
    record_hash = record.get("hash")
    if record_hash is not None and record_hash != clip_id:
        raise HookTheoryAdapterError(
            f"record hash {record_hash!r} does not match clip_id {clip_id!r}",
            clip_id=clip_id,
        )
    payload = record.get("json")
    if not isinstance(payload, Mapping):
        raise HookTheoryAdapterError(
            "HookTheory record has no usable json payload",
            clip_id=clip_id,
        )

    piece_id = f"piece:hooktheory-{clip_id}"
    split = _normalize_split(record.get("split"))
    flags: list[QualityFlag] = []
    if split is None:
        flags.append(
            _flag(
                "hooktheory.split_unknown",
                "Source split was missing or unsupported and normalized to None.",
                piece_id,
            )
        )
    structure_match = _validate_structure_match(structure_row, clip_id, split)
    source_group_id, structure_flags = _structure_flags(structure_match, piece_id)
    flags.extend(structure_flags)

    timeline, meters, meter_flags, default_meter = _select_meters(payload, piece_id)
    tempos, tempo_flags, default_tempo = _select_tempos(payload, timeline, piece_id)
    keys, key_flags = _key_regions(payload, timeline, piece_id)
    melody, note_flags, note_end = _make_notes(payload, keys, timeline, piece_id)
    chords, chord_flags, chord_end = _chord_regions(payload, timeline, piece_id)
    flags.extend(tempo_flags)
    flags.extend(meter_flags)
    flags.extend(key_flags)
    flags.extend(note_flags)
    flags.extend(chord_flags)

    content_end = max(
        note_end,
        chord_end,
        max((key.onset for key in keys), default=RationalTime(0)),
        max((event.onset_qn for event in tempos), default=RationalTime(0)),
        max((event.onset_qn for event in meters), default=RationalTime(0)),
    )
    duration, duration_flags = _raw_duration(payload, timeline, content_end, piece_id)
    flags.extend(duration_flags)
    bars, beats, bar_flags = _bars_and_beats(duration, meters, piece_id)
    flags.extend(bar_flags)

    visible_annotations, visible_targets, target_flags = _annotations_and_targets(
        keys, melody, chords, duration, piece_id
    )
    flags.extend(target_flags)
    annotations = visible_annotations if config.include_targets else ()
    targets = visible_targets if config.include_targets else ()

    provenance = _provenance(
        clip_id,
        source_path,
        structure_match=structure_match,
        default_tempo=default_tempo,
        default_meter=default_meter,
        include_targets=config.include_targets,
    )
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=piece_id,
        dataset_name=config.dataset_name,
        source_group_id=source_group_id,
        split=split,
        source_path=source_path,
        source_resolution=None,
        duration_qn=duration,
        metadata=PieceMetadata(
            source_format="json",
            title=None,
            creators=None,
            collection=None,
            movement_title=None,
            movement_number=None,
            genres=None,
            copyright=None,
            language=None,
        ),
        tracks=(
            CanonicalTrack(
                track_id=_TRACK_ID,
                source_track_index=0,
                name="melody",
                instrument_name=None,
                program=None,
                channel=None,
                is_percussion=False,
                provenance_id="prov:conversion",
            ),
        ),
        notes=tuple(value.note for value in melody),
        bars=bars,
        beats=beats,
        tempo_events=tempos,
        meter_events=meters,
        key_signature_events=(),
        annotations=annotations,
        targets=targets,
        provenance=provenance,
        quality_flags=tuple(
            sorted(flags, key=lambda value: (value.code, value.entity_ids, value.message))
        ),
    )
    report = validate_piece(piece)
    if report.errors:
        raise _validation_error(clip_id, report)
    return piece


def _find_structure_row(
    structure_root: str | PathLike[str], split: str | None, clip_id: str
) -> Mapping[str, Any] | None:
    if split is None:
        return None
    root = Path(structure_root)
    path = root / f"HookTheoryStructure.{split}.jsonl"
    if not path.is_file():
        raise HookTheoryAdapterError(
            f"structure source is missing: {path}",
            clip_id=clip_id,
        )
    matches: list[Mapping[str, Any]] = []
    try:
        for _line_number, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            if isinstance(audio_path, str) and Path(audio_path).stem == clip_id:
                matches.append(row)
    except JSONStreamError as exc:
        raise HookTheoryAdapterError(str(exc), clip_id=clip_id) from exc
    if len(matches) > 1:
        raise HookTheoryAdapterError(
            f"duplicate structure rows for clip {clip_id!r}",
            clip_id=clip_id,
        )
    return matches[0] if matches else None


def load_hooktheory_piece(
    raw_path: str | PathLike[str],
    clip_id: str,
    *,
    config: HookTheoryAdapterConfig,
    structure_root: str | PathLike[str] | None = None,
) -> CanonicalPiece:
    """Locate one raw merged record incrementally and convert it."""

    try:
        path_value = os.fspath(raw_path)
    except TypeError as exc:
        raise HookTheoryAdapterError(
            "raw_path must be string-like", clip_id=clip_id
        ) from exc
    if not isinstance(path_value, str) or Path(path_value).name != "4_merged.json":
        raise HookTheoryAdapterError(
            "raw_path must refer to the production 4_merged.json source",
            clip_id=clip_id,
        )
    try:
        record = find_object_record(path_value, clip_id)
    except JSONStreamError as exc:
        raise HookTheoryAdapterError(str(exc), clip_id=clip_id) from exc
    if not isinstance(record, Mapping):
        raise HookTheoryAdapterError(
            "requested HookTheory record is not an object",
            clip_id=clip_id,
        )
    structure_row = None
    if structure_root is not None:
        structure_row = _find_structure_row(
            structure_root, _normalize_split(record.get("split")), clip_id
        )
    return convert_hooktheory_record(
        clip_id,
        record,
        config=config,
        structure_row=structure_row,
        source_path=path_value,
    )
