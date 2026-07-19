"""Minimal deterministic Standard MIDI File to canonical-piece adapter."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import mido

from music_critic.data import (
    SCHEMA_VERSION,
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
    RationalTime,
    TempoEvent,
    ValidationReport,
    validate_piece,
)


__all__ = ["MidiAdapterConfig", "MidiAdapterError", "load_midi_piece"]

_DEFAULT_TEMPO = 500_000
_DEFAULT_NUMERATOR = 4
_DEFAULT_DENOMINATOR = 4

_MAJOR_FIFTHS = {
    "Cb": -7,
    "Gb": -6,
    "Db": -5,
    "Ab": -4,
    "Eb": -3,
    "Bb": -2,
    "F": -1,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
}
_MINOR_FIFTHS = {
    "Abm": -7,
    "Ebm": -6,
    "Bbm": -5,
    "Fm": -4,
    "Cm": -3,
    "Gm": -2,
    "Dm": -1,
    "Am": 0,
    "Em": 1,
    "Bm": 2,
    "F#m": 3,
    "C#m": 4,
    "G#m": 5,
    "D#m": 6,
    "A#m": 7,
}


@dataclass(frozen=True, slots=True)
class MidiAdapterConfig:
    dataset_name: str
    source_group_id: str | None = None
    split: str | None = None


class MidiAdapterError(Exception):
    """Raised when a MIDI source cannot be converted to a valid CanonicalPiece."""

    validation_report: object | None

    def __init__(
        self,
        message: str,
        *,
        validation_report: object | None = None,
    ) -> None:
        self.validation_report = validation_report
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _LocatedMessage:
    tick: int
    source_track_index: int
    message_index: int
    message: Any

    @property
    def order(self) -> tuple[int, int, int]:
        return self.tick, self.source_track_index, self.message_index


@dataclass(frozen=True, slots=True)
class _OpenNote:
    tick: int
    velocity: int
    program: int | None
    ordinal: int


@dataclass(frozen=True, slots=True)
class _PairedNote:
    source_track_index: int
    channel: int
    pitch: int
    onset_tick: int
    offset_tick: int
    velocity: int
    program: int | None
    ordinal: int


@dataclass(frozen=True, slots=True)
class _SelectedEvent:
    tick: int
    value: Any
    provenance_id: str


def _flag(
    code: str,
    message: str,
    piece_id: str,
    *,
    provenance_id: str = "prov:conversion",
) -> QualityFlag:
    return QualityFlag(
        code=code,
        severity="warning",
        message=message,
        entity_ids=(piece_id,),
        provenance_id=provenance_id,
    )


def _read_midi(path_string: str) -> tuple[bytes, Any]:
    try:
        payload = Path(path_string).read_bytes()
    except (OSError, ValueError) as exc:
        raise MidiAdapterError(
            f"{path_string}: unreadable MIDI source: {exc}"
        ) from exc
    try:
        midi_file = mido.MidiFile(file=BytesIO(payload))
    except (OSError, EOFError, KeyError, ValueError) as exc:
        raise MidiAdapterError(
            f"{path_string}: corrupted or unreadable MIDI file: {exc}"
        ) from exc
    return payload, midi_file


def _collect_messages(midi_file: Any, path_string: str) -> tuple[list[_LocatedMessage], int]:
    located: list[_LocatedMessage] = []
    source_end_tick = 0
    for source_track_index, track in enumerate(midi_file.tracks):
        absolute_tick = 0
        for message_index, message in enumerate(track):
            delta = message.time
            if not isinstance(delta, int) or isinstance(delta, bool) or delta < 0:
                raise MidiAdapterError(
                    f"{path_string}: invalid non-negative integer delta time "
                    f"at track {source_track_index}, message {message_index}"
                )
            absolute_tick += delta
            source_end_tick = max(source_end_tick, absolute_tick)
            located.append(
                _LocatedMessage(
                    tick=absolute_tick,
                    source_track_index=source_track_index,
                    message_index=message_index,
                    message=message,
                )
            )
    return located, source_end_tick


def _event_value(message: Any, event_type: str) -> Any:
    if event_type == "set_tempo":
        return message.tempo
    if event_type == "time_signature":
        return message.numerator, message.denominator
    return message.key


def _select_global_events(
    messages: Iterable[_LocatedMessage],
    event_type: str,
    *,
    piece_id: str,
    conflict_code: str,
    label: str,
) -> tuple[list[_SelectedEvent], list[QualityFlag]]:
    by_tick: dict[int, list[_LocatedMessage]] = defaultdict(list)
    for located in messages:
        if located.message.type == event_type:
            by_tick[located.tick].append(located)

    selected: list[_SelectedEvent] = []
    flags: list[QualityFlag] = []
    for tick in sorted(by_tick):
        candidates = sorted(by_tick[tick], key=lambda item: item.order)
        first_value = _event_value(candidates[0].message, event_type)
        distinct_values: list[Any] = []
        for candidate in candidates:
            value = _event_value(candidate.message, event_type)
            if value not in distinct_values:
                distinct_values.append(value)
        if len(distinct_values) > 1:
            flags.append(
                _flag(
                    conflict_code,
                    f"Conflicting {label} events at tick {tick}; selected "
                    f"{first_value!r} by source-track/message order from "
                    f"{len(distinct_values)} distinct values.",
                    piece_id,
                )
            )
        selected.append(
            _SelectedEvent(
                tick=tick,
                value=first_value,
                provenance_id="prov:source",
            )
        )
    return selected, flags


def _pair_notes(
    messages: Iterable[_LocatedMessage],
    piece_id: str,
) -> tuple[
    list[_PairedNote],
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], int | None],
    list[QualityFlag],
]:
    messages_by_track: dict[int, list[_LocatedMessage]] = defaultdict(list)
    for located in messages:
        messages_by_track[located.source_track_index].append(located)

    paired: list[_PairedNote] = []
    program_changes: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    selected_programs: dict[tuple[int, int], int | None] = {}
    unmatched: list[tuple[int, int, int, int]] = []
    dangling: list[tuple[int, int, int, int]] = []

    for track_index in sorted(messages_by_track):
        current_program: dict[int, int] = {}
        first_note_order: dict[int, tuple[int, int]] = {}
        first_note_program: dict[int, int | None] = {}
        first_program: dict[int, int] = {}
        open_notes: dict[tuple[int, int], deque[_OpenNote]] = defaultdict(deque)
        note_ordinals: dict[tuple[int, int], int] = defaultdict(int)

        for located in sorted(messages_by_track[track_index], key=lambda item: item.order):
            message = located.message
            channel = getattr(message, "channel", None)
            if not isinstance(channel, int):
                continue
            track_channel = track_index, channel
            if message.type == "program_change":
                program = int(message.program)
                first_program.setdefault(channel, program)
                current_program[channel] = program
                program_changes[track_channel].append((located.tick, program))
                continue

            is_note_on = message.type == "note_on" and message.velocity > 0
            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )
            if is_note_on:
                first_note_order.setdefault(channel, (located.tick, located.message_index))
                first_note_program.setdefault(channel, current_program.get(channel))
                key = channel, int(message.note)
                ordinal = note_ordinals[key]
                note_ordinals[key] += 1
                open_notes[key].append(
                    _OpenNote(
                        tick=located.tick,
                        velocity=int(message.velocity),
                        program=current_program.get(channel),
                        ordinal=ordinal,
                    )
                )
            elif is_note_off:
                key = channel, int(message.note)
                if not open_notes[key]:
                    unmatched.append((track_index, channel, int(message.note), located.tick))
                    continue
                opened = open_notes[key].popleft()
                paired.append(
                    _PairedNote(
                        source_track_index=track_index,
                        channel=channel,
                        pitch=int(message.note),
                        onset_tick=opened.tick,
                        offset_tick=located.tick,
                        velocity=opened.velocity,
                        program=opened.program,
                        ordinal=opened.ordinal,
                    )
                )

        channels = set(first_program) | set(first_note_order)
        for channel in channels:
            selected_programs[(track_index, channel)] = (
                first_note_program[channel]
                if channel in first_note_order
                else first_program.get(channel)
            )
        for (channel, pitch), queue in open_notes.items():
            for opened in queue:
                dangling.append((track_index, channel, pitch, opened.tick))

    flags: list[QualityFlag] = []
    if unmatched:
        ticks = ",".join(str(item[3]) for item in unmatched[:12])
        flags.append(
            _flag(
                "midi.unmatched_note_off",
                f"Ignored {len(unmatched)} unmatched note-off event(s); ticks={ticks}.",
                piece_id,
            )
        )
    if dangling:
        ticks = ",".join(str(item[3]) for item in dangling[:12])
        flags.append(
            _flag(
                "midi.dangling_note_on",
                f"Omitted {len(dangling)} dangling note-on event(s); ticks={ticks}.",
                piece_id,
            )
        )

    for track_channel, changes in sorted(program_changes.items()):
        track_index, channel = track_channel
        track_messages = messages_by_track[track_index]
        first_note = next(
            (
                item
                for item in sorted(track_messages, key=lambda candidate: candidate.order)
                if getattr(item.message, "channel", None) == channel
                and item.message.type == "note_on"
                and item.message.velocity > 0
            ),
            None,
        )
        if first_note is None:
            continue
        later = [
            (tick, program)
            for tick, program in changes
            if tick > first_note.tick
            or (
                tick == first_note.tick
                and any(
                    item.message.type == "program_change"
                    and getattr(item.message, "channel", None) == channel
                    and item.tick == tick
                    and item.message_index > first_note.message_index
                    and item.message.program == program
                    for item in track_messages
                )
            )
        ]
        if later:
            ticks = ",".join(str(tick) for tick, _ in later)
            flags.append(
                _flag(
                    "midi.mid_track_program_change",
                    f"Source track {track_index}, channel {channel} has "
                    f"{len(later)} program change(s) after note activity; ticks={ticks}.",
                    piece_id,
                )
            )

    return paired, program_changes, selected_programs, flags


def _track_id(source_track_index: int, channel: int | None) -> str:
    suffix = "empty" if channel is None else f"c{channel:02d}"
    return f"track:t{source_track_index:04d}-{suffix}"


def _make_tracks(
    midi_file: Any,
    messages: Iterable[_LocatedMessage],
    selected_programs: dict[tuple[int, int], int | None],
) -> tuple[tuple[CanonicalTrack, ...], dict[tuple[int, int], str]]:
    by_track: dict[int, list[_LocatedMessage]] = defaultdict(list)
    for located in messages:
        by_track[located.source_track_index].append(located)

    tracks: list[CanonicalTrack] = []
    channel_ids: dict[tuple[int, int], str] = {}
    for track_index, _source_track in enumerate(midi_file.tracks):
        located_messages = sorted(by_track.get(track_index, []), key=lambda item: item.order)
        channels = sorted(
            {
                int(item.message.channel)
                for item in located_messages
                if isinstance(getattr(item.message, "channel", None), int)
            }
        )
        track_name = next(
            (
                item.message.name
                for item in located_messages
                if item.message.type == "track_name"
            ),
            None,
        )
        instrument_name = next(
            (
                item.message.name
                for item in located_messages
                if item.message.type == "instrument_name"
            ),
            None,
        )
        logical_channels: list[int | None] = channels if channels else [None]
        for channel in logical_channels:
            identifier = _track_id(track_index, channel)
            if channel is not None:
                channel_ids[(track_index, channel)] = identifier
            tracks.append(
                CanonicalTrack(
                    track_id=identifier,
                    source_track_index=track_index,
                    name=track_name,
                    instrument_name=instrument_name,
                    program=(
                        selected_programs.get((track_index, channel))
                        if channel is not None
                        else None
                    ),
                    channel=channel,
                    is_percussion=channel == 9,
                    provenance_id="prov:source",
                )
            )
    tracks.sort(
        key=lambda track: (
            track.source_track_index is None,
            track.source_track_index if track.source_track_index is not None else 0,
            track.track_id,
        )
    )
    return tuple(tracks), channel_ids


def _make_notes(
    paired: Iterable[_PairedNote],
    channel_ids: dict[tuple[int, int], str],
    ticks_per_beat: int,
    piece_id: str,
) -> tuple[tuple[CanonicalNote, ...], list[QualityFlag]]:
    notes: list[CanonicalNote] = []
    flags: list[QualityFlag] = []
    for item in paired:
        duration_ticks = item.offset_tick - item.onset_tick
        local = (
            f"t{item.source_track_index:04d}-c{item.channel:02d}-"
            f"p{item.pitch:03d}-o{item.onset_tick:012d}-n{item.ordinal:04d}"
        )
        note_id = f"note:{local}"
        is_zero_duration = duration_ticks == 0
        notes.append(
            CanonicalNote(
                note_id=note_id,
                track_id=channel_ids[(item.source_track_index, item.channel)],
                pitch=item.pitch,
                onset_qn=RationalTime(item.onset_tick, ticks_per_beat),
                duration_qn=RationalTime(duration_ticks, ticks_per_beat),
                velocity=item.velocity,
                channel=item.channel,
                program=item.program,
                is_percussion=item.channel == 9,
                is_grace=is_zero_duration,
                spelling_step=None,
                spelling_alter=None,
                staff=None,
                voice=None,
                articulations=None,
                dynamic=None,
                source_onset_ticks=item.onset_tick,
                source_duration_ticks=duration_ticks,
                source_onset_seconds=None,
                source_duration_seconds=None,
                provenance_id="prov:source",
            )
        )
        if is_zero_duration:
            flags.append(
                QualityFlag(
                    code="midi.zero_duration_note",
                    severity="warning",
                    message="Preserved a real same-tick note-on/note-off as grace-like.",
                    entity_ids=(note_id,),
                    provenance_id="prov:source",
                )
            )

    track_order = {
        identifier: index
        for index, identifier in enumerate(
            sorted(set(channel_ids.values()))
        )
    }
    notes.sort(
        key=lambda note: (
            note.onset_qn,
            track_order[note.track_id],
            note.pitch,
            note.duration_qn,
            note.note_id,
        )
    )
    return tuple(notes), flags


def _ensure_defaults(
    tempo: list[_SelectedEvent],
    meter: list[_SelectedEvent],
    piece_id: str,
) -> tuple[list[_SelectedEvent], list[_SelectedEvent], list[QualityFlag], bool, bool]:
    flags: list[QualityFlag] = []
    default_tempo = not tempo or tempo[0].tick != 0
    default_meter = not meter or meter[0].tick != 0
    if default_tempo:
        tempo.insert(0, _SelectedEvent(0, _DEFAULT_TEMPO, "prov:default-tempo"))
        flags.append(
            _flag(
                "midi.default_tempo",
                "Inserted the Standard MIDI default tempo of 500000 us/qn at tick 0.",
                piece_id,
                provenance_id="prov:default-tempo",
            )
        )
    if default_meter:
        meter.insert(
            0,
            _SelectedEvent(
                0,
                (_DEFAULT_NUMERATOR, _DEFAULT_DENOMINATOR),
                "prov:default-meter",
            ),
        )
        flags.append(
            _flag(
                "midi.default_meter",
                "Inserted the MIDI-adapter default meter of 4/4 at tick 0.",
                piece_id,
                provenance_id="prov:default-meter",
            )
        )
    return tempo, meter, flags, default_tempo, default_meter


def _validate_meter_boundaries(
    meter: list[_SelectedEvent],
    ticks_per_beat: int,
    path_string: str,
) -> None:
    cursor = RationalTime(0)
    numerator, denominator = meter[0].value
    for event in meter[1:]:
        onset = RationalTime(event.tick, ticks_per_beat)
        nominal = RationalTime(numerator * 4, denominator)
        while cursor < onset:
            next_boundary = cursor + nominal
            if next_boundary > onset:
                raise MidiAdapterError(
                    f"{path_string}: meter change at tick {event.tick} is inside "
                    f"a bar under active meter {numerator}/{denominator}"
                )
            cursor = next_boundary
        if cursor != onset:
            raise MidiAdapterError(
                f"{path_string}: meter change at tick {event.tick} is not on a "
                f"bar boundary under active meter {numerator}/{denominator}"
            )
        numerator, denominator = event.value


def _make_meter_events(
    selected: Iterable[_SelectedEvent], ticks_per_beat: int
) -> tuple[MeterEvent, ...]:
    return tuple(
        MeterEvent(
            meter_event_id=f"meter:{index:04d}",
            onset_qn=RationalTime(event.tick, ticks_per_beat),
            numerator=int(event.value[0]),
            denominator=int(event.value[1]),
            provenance_id=event.provenance_id,
        )
        for index, event in enumerate(selected)
    )


def _make_tempo_events(
    selected: Iterable[_SelectedEvent], ticks_per_beat: int
) -> tuple[TempoEvent, ...]:
    return tuple(
        TempoEvent(
            tempo_event_id=f"tempo:{index:04d}",
            onset_qn=RationalTime(event.tick, ticks_per_beat),
            microseconds_per_quarter=int(event.value),
            provenance_id=event.provenance_id,
        )
        for index, event in enumerate(selected)
    )


def _parse_key(raw_value: Any) -> tuple[int, str] | None:
    if not isinstance(raw_value, str):
        return None
    if raw_value in _MAJOR_FIFTHS:
        return _MAJOR_FIFTHS[raw_value], "major"
    if raw_value in _MINOR_FIFTHS:
        return _MINOR_FIFTHS[raw_value], "minor"
    return None


def _make_key_events(
    selected: Iterable[_SelectedEvent],
    ticks_per_beat: int,
    piece_id: str,
) -> tuple[tuple[KeySignatureEvent, ...], list[QualityFlag]]:
    events: list[KeySignatureEvent] = []
    flags: list[QualityFlag] = []
    for selected_index, event in enumerate(selected):
        parsed = _parse_key(event.value)
        if parsed is None:
            flags.append(
                _flag(
                    "midi.unsupported_key_signature",
                    f"Omitted unsupported key-signature value {event.value!r} "
                    f"at tick {event.tick}.",
                    piece_id,
                )
            )
            continue
        fifths, mode = parsed
        events.append(
            KeySignatureEvent(
                key_signature_event_id=f"keysig:{selected_index:04d}",
                onset_qn=RationalTime(event.tick, ticks_per_beat),
                fifths=fifths,
                mode=mode,
                raw_value=event.value,
                provenance_id="prov:source",
            )
        )
    return tuple(events), flags


def _active_meter(meters: tuple[MeterEvent, ...], onset: RationalTime) -> MeterEvent:
    active = meters[0]
    for meter in meters[1:]:
        if meter.onset_qn > onset:
            break
        active = meter
    return active


def _make_bars_and_beats(
    source_end_tick: int,
    ticks_per_beat: int,
    meters: tuple[MeterEvent, ...],
) -> tuple[RationalTime, tuple[CanonicalBar, ...], tuple[CanonicalBeat, ...]]:
    source_end = RationalTime(source_end_tick, ticks_per_beat)
    if source_end == RationalTime(0):
        initial = meters[0]
        duration = RationalTime(initial.numerator * 4, initial.denominator)
    else:
        duration = source_end
        if meters[-1].onset_qn == source_end and meters[-1].onset_qn > RationalTime(0):
            duration = duration + RationalTime(
                meters[-1].numerator * 4,
                meters[-1].denominator,
            )

    bars: list[CanonicalBar] = []
    beats: list[CanonicalBeat] = []
    cursor = RationalTime(0)
    while cursor < duration:
        meter = _active_meter(meters, cursor)
        nominal = RationalTime(meter.numerator * 4, meter.denominator)
        bar_duration = min(nominal, duration - cursor)
        bar_index = len(bars)
        bar_id = f"bar:{bar_index:04d}"
        incomplete = bar_duration < nominal
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
        beat_unit = RationalTime(4, meter.denominator)
        position = RationalTime(0)
        beat_index = 0
        while position < bar_duration:
            beat_duration = min(beat_unit, bar_duration - position)
            beats.append(
                CanonicalBeat(
                    beat_id=f"beat:{len(beats):06d}",
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
            position = position + beat_unit
            beat_index += 1
        cursor = cursor + bar_duration
    return duration, tuple(bars), tuple(beats)


def _make_provenance(
    checksum: str,
    *,
    default_tempo: bool,
    default_meter: bool,
) -> tuple[ProvenanceRecord, ...]:
    records = [
        ProvenanceRecord(
            provenance_id="prov:source",
            kind="source",
            source="standard_midi_file",
            record_id=checksum,
            uri=None,
            version=None,
            checksum_sha256=checksum,
            created_at=None,
            parents=(),
            details=(("timing", "ppqn"),),
        ),
        ProvenanceRecord(
            provenance_id="prov:conversion",
            kind="conversion",
            source="music_critic.adapters.midi",
            record_id=None,
            uri=None,
            version="1",
            checksum_sha256=None,
            created_at=None,
            parents=("prov:source",),
            details=(
                ("event_order", "tick,source_track_index,message_index"),
                ("note_pairing", "fifo_by_source_track_channel_pitch"),
            ),
        ),
    ]
    if default_meter:
        records.append(
            ProvenanceRecord(
                provenance_id="prov:default-meter",
                kind="default",
                source="standard_midi_default",
                record_id=None,
                uri=None,
                version=None,
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
                source="standard_midi_default",
                record_id=None,
                uri=None,
                version=None,
                checksum_sha256=None,
                created_at=None,
                parents=("prov:conversion",),
                details=(("microseconds_per_quarter", _DEFAULT_TEMPO),),
            )
        )
    return tuple(records)


def _validation_error(path_string: str, report: ValidationReport) -> MidiAdapterError:
    examples = ", ".join(
        f"{issue.code}@{issue.path}" for issue in report.errors[:5]
    )
    message = (
        f"{path_string}: canonical validation failed with "
        f"{len(report.errors)} error(s)"
    )
    if examples:
        message += f"; {examples}"
    return MidiAdapterError(message, validation_report=report)


def load_midi_piece(
    path: str | PathLike[str],
    *,
    config: MidiAdapterConfig,
) -> CanonicalPiece:
    """Convert a type-0/type-1 PPQN MIDI file into a validated canonical piece."""

    path_value = os.fspath(path)
    if not isinstance(path_value, str):
        raise MidiAdapterError("MIDI source path must resolve to a string")
    payload, midi_file = _read_midi(path_value)
    if midi_file.type == 2:
        raise MidiAdapterError(f"{path_value}: MIDI type 2 is unsupported")
    if midi_file.type not in {0, 1}:
        raise MidiAdapterError(f"{path_value}: unsupported MIDI type {midi_file.type}")
    ticks_per_beat = midi_file.ticks_per_beat
    if not isinstance(ticks_per_beat, int) or isinstance(ticks_per_beat, bool):
        raise MidiAdapterError(f"{path_value}: invalid ticks_per_beat value")
    if ticks_per_beat <= 0:
        reason = "SMPTE/non-PPQN timing" if ticks_per_beat < 0 else "ticks_per_beat <= 0"
        raise MidiAdapterError(f"{path_value}: {reason} is unsupported")

    checksum = sha256(payload).hexdigest()
    piece_id = f"piece:midi-{checksum}"
    messages, source_end_tick = _collect_messages(midi_file, path_value)

    tempo, flags = _select_global_events(
        messages,
        "set_tempo",
        piece_id=piece_id,
        conflict_code="midi.tempo_conflict",
        label="tempo",
    )
    meter, meter_flags = _select_global_events(
        messages,
        "time_signature",
        piece_id=piece_id,
        conflict_code="midi.meter_conflict",
        label="meter",
    )
    keys, key_conflict_flags = _select_global_events(
        messages,
        "key_signature",
        piece_id=piece_id,
        conflict_code="midi.key_signature_conflict",
        label="key-signature",
    )
    flags.extend(meter_flags)
    flags.extend(key_conflict_flags)
    tempo, meter, default_flags, default_tempo, default_meter = _ensure_defaults(
        tempo, meter, piece_id
    )
    flags.extend(default_flags)
    _validate_meter_boundaries(meter, ticks_per_beat, path_value)

    paired, _program_changes, selected_programs, note_flags = _pair_notes(
        messages, piece_id
    )
    flags.extend(note_flags)
    tracks, channel_ids = _make_tracks(midi_file, messages, selected_programs)
    notes, zero_duration_flags = _make_notes(
        paired, channel_ids, ticks_per_beat, piece_id
    )
    flags.extend(zero_duration_flags)

    tempo_events = _make_tempo_events(tempo, ticks_per_beat)
    meter_events = _make_meter_events(meter, ticks_per_beat)
    key_events, unsupported_key_flags = _make_key_events(
        keys, ticks_per_beat, piece_id
    )
    flags.extend(unsupported_key_flags)
    duration, bars, beats = _make_bars_and_beats(
        source_end_tick, ticks_per_beat, meter_events
    )

    quality_flags = tuple(
        sorted(flags, key=lambda flag: (flag.code, flag.entity_ids, flag.message))
    )
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=piece_id,
        dataset_name=config.dataset_name,
        source_group_id=(
            config.source_group_id
            if config.source_group_id is not None
            else piece_id
        ),
        split=config.split,
        source_path=path_value,
        source_resolution=ticks_per_beat,
        duration_qn=duration,
        metadata=PieceMetadata(
            source_format="midi",
            title=None,
            creators=None,
            collection=None,
            movement_title=None,
            movement_number=None,
            genres=None,
            copyright=None,
            language=None,
        ),
        tracks=tracks,
        notes=notes,
        bars=bars,
        beats=beats,
        tempo_events=tempo_events,
        meter_events=meter_events,
        key_signature_events=key_events,
        annotations=(),
        targets=(),
        provenance=_make_provenance(
            checksum,
            default_tempo=default_tempo,
            default_meter=default_meter,
        ),
        quality_flags=quality_flags,
    )
    report = validate_piece(piece)
    if report.errors:
        raise _validation_error(path_value, report)
    return piece
