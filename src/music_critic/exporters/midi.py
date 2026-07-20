"""Deterministic Standard MIDI File rendering for canonical pieces."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from math import lcm
from os import PathLike
from pathlib import Path
from typing import Iterable

import mido

from music_critic.data.schema import AnnotationSpan, CanonicalPiece
from music_critic.data.timing import RationalTime
from music_critic.data.validation import validate_piece


__all__ = [
    "MidiRenderConfig",
    "MidiRenderError",
    "MidiRenderReport",
    "piece_to_midi_bytes",
    "write_piece_midi",
]

_MAX_MIDI_PPQ = 32_767


class MidiRenderError(ValueError):
    """Raised when a canonical piece cannot be rendered under the policy."""


@dataclass(frozen=True, slots=True)
class MidiRenderConfig:
    """Controls track content and exact-timing behavior."""

    ticks_per_quarter: int | None = None
    require_exact_timing: bool = True
    quantized_ticks_per_quarter: int = 960
    melody_velocity: int = 96
    melody_program: int = 0
    include_click_track: bool = True
    include_target_markers: bool = True
    click_channel: int = 9
    click_downbeat_note: int = 37
    click_beat_note: int = 31
    click_velocity: int = 80


@dataclass(frozen=True, slots=True)
class MidiRenderReport:
    """Auditable facts about one MIDI rendering."""

    piece_id: str
    ticks_per_quarter: int
    exact_timing: bool
    timing_quantized: bool
    maximum_quantization_error_qn: RationalTime
    rendered_tracks: int
    rendered_notes: int
    rendered_clicks: int
    rendered_tempo_events: int
    rendered_meter_events: int
    rendered_markers: int


@dataclass(frozen=True, slots=True)
class _Marker:
    onset: RationalTime
    text: str


def _validate_config(config: MidiRenderConfig) -> None:
    integer_ranges = (
        ("melody_velocity", config.melody_velocity, 1, 127),
        ("melody_program", config.melody_program, 0, 127),
        ("click_channel", config.click_channel, 0, 15),
        ("click_downbeat_note", config.click_downbeat_note, 0, 127),
        ("click_beat_note", config.click_beat_note, 0, 127),
        ("click_velocity", config.click_velocity, 1, 127),
        (
            "quantized_ticks_per_quarter",
            config.quantized_ticks_per_quarter,
            1,
            _MAX_MIDI_PPQ,
        ),
    )
    for name, value, minimum, maximum in integer_ranges:
        if isinstance(value, bool) or not isinstance(value, int):
            raise MidiRenderError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise MidiRenderError(f"{name} must be in [{minimum}, {maximum}]")
    for name, value in (
        ("require_exact_timing", config.require_exact_timing),
        ("include_click_track", config.include_click_track),
        ("include_target_markers", config.include_target_markers),
    ):
        if not isinstance(value, bool):
            raise MidiRenderError(f"{name} must be a bool")
    if config.ticks_per_quarter is not None:
        value = config.ticks_per_quarter
        if isinstance(value, bool) or not isinstance(value, int):
            raise MidiRenderError("ticks_per_quarter must be an integer or None")
        if not 1 <= value <= _MAX_MIDI_PPQ:
            raise MidiRenderError(
                f"ticks_per_quarter must be in [1, {_MAX_MIDI_PPQ}]"
            )


def _target_values(piece: CanonicalPiece) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for target in piece.targets:
        if target.alignment_type != "annotation_span":
            continue
        by_entity = values.setdefault(target.task, {})
        for entity_id, value, available in zip(
            target.entity_ids, target.values, target.mask, strict=True
        ):
            if available and value is not None:
                if isinstance(value, tuple):
                    by_entity[entity_id] = ",".join(str(item) for item in value)
                else:
                    by_entity[entity_id] = str(value)
    return values


def _annotation_markers(piece: CanonicalPiece) -> tuple[_Marker, ...]:
    values = _target_values(piece)
    markers: list[_Marker] = []
    chords: list[AnnotationSpan] = []
    for annotation in piece.annotations:
        if annotation.annotation_type == "theory.local_key":
            tonic = values.get("theory.local_key.tonic_pc", {}).get(
                annotation.annotation_id, "?"
            )
            mode = values.get("theory.local_key.mode", {}).get(
                annotation.annotation_id, "?"
            )
            markers.append(_Marker(annotation.start_qn, f"KEY tonic={tonic} mode={mode}"))
        elif annotation.annotation_type == "theory.chord":
            chords.append(annotation)
            fields = []
            for label, task in (
                ("root", "theory.chord.root_degree"),
                ("extent", "theory.chord.extent"),
                ("inversion", "theory.chord.inversion"),
            ):
                value = values.get(task, {}).get(annotation.annotation_id, "?")
                fields.append(f"{label}={value}")
            text = (
                "CHORD unavailable"
                if all(field.endswith("=?") for field in fields)
                else "CHORD " + " ".join(fields)
            )
            markers.append(_Marker(annotation.start_qn, text))

    chord_starts = {chord.start_qn for chord in chords}
    for chord in chords:
        if chord.end_qn not in chord_starts:
            markers.append(_Marker(chord.end_qn, "CHORD_END"))
    return tuple(sorted(markers, key=lambda item: (item.onset, item.text)))


def _timing_points(
    piece: CanonicalPiece, markers: Iterable[_Marker], *, include_click: bool
) -> tuple[RationalTime, ...]:
    points = [RationalTime(0), piece.duration_qn]
    for note in piece.notes:
        points.extend((note.onset_qn, note.onset_qn + note.duration_qn))
    for event in piece.tempo_events:
        points.append(event.onset_qn)
    for event in piece.meter_events:
        points.append(event.onset_qn)
    for marker in markers:
        points.append(marker.onset)
    for bar in piece.bars:
        points.extend((bar.start_qn, bar.start_qn + bar.duration_qn))
    for beat in piece.beats:
        points.extend((beat.start_qn, beat.start_qn + beat.duration_qn))
        if include_click:
            points.append(
                beat.start_qn
                + min(beat.duration_qn / 4, RationalTime(1, 8))
            )
    return tuple(points)


def _required_ppq(points: Iterable[RationalTime]) -> int:
    result = 1
    for point in points:
        result = lcm(result, point.den)
        if result > _MAX_MIDI_PPQ:
            return result
    return result


def _select_ppq(
    points: tuple[RationalTime, ...], config: MidiRenderConfig
) -> tuple[int, bool]:
    required = _required_ppq(points)
    if config.ticks_per_quarter is not None:
        ppq = config.ticks_per_quarter
        exact = all((point.num * ppq) % point.den == 0 for point in points)
        if not exact and config.require_exact_timing:
            raise MidiRenderError(
                f"ticks_per_quarter={ppq} cannot represent all canonical times exactly"
            )
        return ppq, exact
    if required <= _MAX_MIDI_PPQ:
        return required, True
    if config.require_exact_timing:
        raise MidiRenderError(
            f"exact canonical timing requires PPQ {required}, above MIDI limit {_MAX_MIDI_PPQ}"
        )
    return config.quantized_ticks_per_quarter, False


class _TickConverter:
    def __init__(self, ppq: int) -> None:
        self.ppq = ppq
        self.max_error = Fraction(0)

    def __call__(self, value: RationalTime) -> int:
        scaled = value.to_fraction() * self.ppq
        if scaled.denominator == 1:
            tick = scaled.numerator
        else:
            # Canonical times are non-negative. This is deterministic half-up rounding.
            tick = (2 * scaled.numerator + scaled.denominator) // (
                2 * scaled.denominator
            )
        error = abs(Fraction(tick, self.ppq) - value.to_fraction())
        self.max_error = max(self.max_error, error)
        return tick


def _message_track(
    events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]],
    *,
    end_tick: int,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    previous = 0
    for tick, group, sequence, message in sorted(
        events, key=lambda item: (item[0], item[1], item[2])
    ):
        del group, sequence
        if tick < previous:
            raise MidiRenderError("internal event ordering produced a negative MIDI delta")
        track.append(message.copy(time=tick - previous))
        previous = tick
    if previous > end_tick:
        raise MidiRenderError("a rendered MIDI event exceeds canonical piece duration")
    track.append(mido.MetaMessage("end_of_track", time=end_tick - previous))
    return track


def _conductor_track(
    piece: CanonicalPiece,
    markers: tuple[_Marker, ...],
    to_tick: _TickConverter,
    end_tick: int,
) -> mido.MidiTrack:
    events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = [
        (0, -1, 0, mido.MetaMessage("track_name", name="Canonical Conductor"))
    ]
    sequence = 1
    for meter in piece.meter_events:
        events.append(
            (
                to_tick(meter.onset_qn),
                0,
                sequence,
                mido.MetaMessage(
                    "time_signature",
                    numerator=meter.numerator,
                    denominator=meter.denominator,
                    clocks_per_click=24,
                    notated_32nd_notes_per_beat=8,
                ),
            )
        )
        sequence += 1
    for tempo in piece.tempo_events:
        events.append(
            (
                to_tick(tempo.onset_qn),
                1,
                sequence,
                mido.MetaMessage(
                    "set_tempo", tempo=tempo.microseconds_per_quarter
                ),
            )
        )
        sequence += 1
    for marker in markers:
        events.append(
            (
                to_tick(marker.onset),
                2,
                sequence,
                mido.MetaMessage("marker", text=marker.text),
            )
        )
        sequence += 1
    return _message_track(events, end_tick=end_tick)


def _note_track(
    piece: CanonicalPiece,
    track_id: str,
    to_tick: _TickConverter,
    end_tick: int,
    config: MidiRenderConfig,
) -> mido.MidiTrack:
    canonical_track = next(track for track in piece.tracks if track.track_id == track_id)
    name = canonical_track.name or canonical_track.track_id
    events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = [
        (0, -1, 0, mido.MetaMessage("track_name", name=name))
    ]
    notes = sorted(
        (note for note in piece.notes if note.track_id == track_id),
        key=lambda note: (note.onset_qn, note.pitch, note.note_id),
    )
    program_by_channel: dict[int, int] = {}
    for sequence, note in enumerate(notes, start=1):
        onset = to_tick(note.onset_qn)
        end = to_tick(note.onset_qn + note.duration_qn)
        if note.duration_qn > RationalTime(0) and end <= onset:
            raise MidiRenderError(
                f"quantization collapses positive duration for {note.note_id}"
            )
        channel = (
            note.channel
            if note.channel is not None
            else (
                canonical_track.channel
                if canonical_track.channel is not None
                else (9 if note.is_percussion else 0)
            )
        )
        program = (
            note.program
            if note.program is not None
            else (
                canonical_track.program
                if canonical_track.program is not None
                else config.melody_program
            )
        )
        start_order = sequence * 2
        if program_by_channel.get(channel) != program:
            events.append(
                (
                    onset,
                    1,
                    start_order,
                    mido.Message("program_change", channel=channel, program=program),
                )
            )
            program_by_channel[channel] = program
            start_order += 1
        velocity = note.velocity if note.velocity is not None else config.melody_velocity
        events.append(
            (
                onset,
                1,
                start_order,
                mido.Message(
                    "note_on", channel=channel, note=note.pitch, velocity=velocity
                ),
            )
        )
        events.append(
            (
                end,
                2 if note.duration_qn.num == 0 else 0,
                sequence,
                mido.Message("note_off", channel=channel, note=note.pitch, velocity=0),
            )
        )
    return _message_track(events, end_tick=end_tick)


def _click_track(
    piece: CanonicalPiece,
    to_tick: _TickConverter,
    end_tick: int,
    config: MidiRenderConfig,
) -> mido.MidiTrack:
    events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = [
        (0, -1, 0, mido.MetaMessage("track_name", name="Canonical Click"))
    ]
    for sequence, beat in enumerate(piece.beats, start=1):
        onset = to_tick(beat.start_qn)
        duration = min(beat.duration_qn / 4, RationalTime(1, 8))
        end = to_tick(beat.start_qn + duration)
        if end <= onset:
            raise MidiRenderError(f"quantization collapses click {beat.beat_id}")
        note = config.click_downbeat_note if beat.is_downbeat else config.click_beat_note
        velocity = config.click_velocity
        events.extend(
            (
                (
                    onset,
                    1,
                    sequence,
                    mido.Message(
                        "note_on",
                        channel=config.click_channel,
                        note=note,
                        velocity=velocity,
                    ),
                ),
                (
                    end,
                    0,
                    sequence,
                    mido.Message(
                        "note_off",
                        channel=config.click_channel,
                        note=note,
                        velocity=0,
                    ),
                ),
            )
        )
    return _message_track(events, end_tick=end_tick)


def _render(
    piece: CanonicalPiece, config: MidiRenderConfig
) -> tuple[bytes, MidiRenderReport]:
    _validate_config(config)
    validation = validate_piece(piece)
    if validation.errors:
        first = validation.errors[0]
        raise MidiRenderError(
            f"canonical piece is invalid: {first.code} at {first.path}: {first.message}"
        )
    for note in piece.notes:
        if note.duration_qn <= RationalTime(0):
            raise MidiRenderError(
                f"MIDI rendering requires positive note duration: {note.note_id}"
            )
    markers = _annotation_markers(piece) if config.include_target_markers else ()
    points = _timing_points(piece, markers, include_click=config.include_click_track)
    ppq, selected_exact = _select_ppq(points, config)
    to_tick = _TickConverter(ppq)
    end_tick = to_tick(piece.duration_qn)

    midi = mido.MidiFile(type=1, ticks_per_beat=ppq)
    midi.tracks.append(_conductor_track(piece, markers, to_tick, end_tick))
    melody_tracks = [track for track in piece.tracks if not track.is_percussion]
    if melody_tracks:
        for track in melody_tracks:
            midi.tracks.append(
                _note_track(piece, track.track_id, to_tick, end_tick, config)
            )
    else:
        midi.tracks.append(
            _message_track(
                [
                    (
                        0,
                        -1,
                        0,
                        mido.MetaMessage("track_name", name="Canonical Melody"),
                    )
                ],
                end_tick=end_tick,
            )
        )
    click_count = 0
    if config.include_click_track:
        midi.tracks.append(_click_track(piece, to_tick, end_tick, config))
        click_count = len(piece.beats)

    buffer = BytesIO()
    midi.save(file=buffer)
    max_error = RationalTime.from_fraction(to_tick.max_error)
    exact = selected_exact and max_error.num == 0
    report = MidiRenderReport(
        piece_id=piece.piece_id,
        ticks_per_quarter=ppq,
        exact_timing=exact,
        timing_quantized=not exact,
        maximum_quantization_error_qn=max_error,
        rendered_tracks=len(midi.tracks),
        rendered_notes=sum(not note.is_percussion for note in piece.notes),
        rendered_clicks=click_count,
        rendered_tempo_events=len(piece.tempo_events),
        rendered_meter_events=len(piece.meter_events),
        rendered_markers=len(markers),
    )
    return buffer.getvalue(), report


def piece_to_midi_bytes(
    piece: CanonicalPiece,
    *,
    config: MidiRenderConfig = MidiRenderConfig(),
) -> tuple[bytes, MidiRenderReport]:
    """Render ``piece`` and return deterministic MIDI bytes plus its report."""

    return _render(piece, config)


def write_piece_midi(
    piece: CanonicalPiece,
    path: str | PathLike[str],
    *,
    config: MidiRenderConfig = MidiRenderConfig(),
) -> MidiRenderReport:
    """Render ``piece`` and write it to ``path``, returning an audit report."""

    payload, report = _render(piece, config)
    Path(path).write_bytes(payload)
    return report
