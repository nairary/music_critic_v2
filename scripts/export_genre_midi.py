#!/usr/bin/env python3
"""Exact diagnostic multitrack SMF export with independent note lifetimes.

GM programs are retained, drums use channel 9, and each route owns its channel
on a MIDI port for the entire file. Additional lanes prevent overlapping copies
of one pitch from terminating each other. Ports are necessary beyond 15 tonal
routes or one drum route: players ignoring MIDI-port metadata cannot reliably
play those files. The companion rendered WAV is the primary listening reference.

No target labels, markers, click notes, inferred instruments, or quantization.
Only zero-duration grace notes are omitted. A positive note with velocity zero
is rejected, since SMF encodes that as note-off, not a silent note lifetime.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from io import BytesIO
from math import lcm
from typing import Any

import mido

from music_critic.data.schema import CanonicalPiece
from music_critic.data.timing import RationalTime as Q
from music_critic.data.validation import validate_piece


EXPORT_VERSION = "DiagnosticGMExactMidi@1.0.0"
TONAL_CHANNELS = tuple(c for c in range(16) if c != 9)


def _message_track(events, end_tick):
    track = mido.MidiTrack()
    previous = 0
    for tick, priority, order, message in sorted(events, key=lambda e: e[:3]):
        if tick < previous or tick > end_tick:
            raise ValueError("MIDI event outside the canonical piece")
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=end_tick - previous))
    return track


def inspect_preview_midi(payload: bytes) -> dict[str, Any]:
    """Independently parse SMF and enforce global port/channel/pitch lifetimes.

    Returned Counters use Fraction times and are for in-process verification,
    not JSON serialization. No canonical note-to-route mapping is trusted here.
    """
    midi = mido.MidiFile(file=BytesIO(payload))
    if midi.type != 1 or not 0 < midi.ticks_per_beat <= 32767:
        raise ValueError("Expected format-1 MIDI with positive legal PPQ")
    ppq = midi.ticks_per_beat
    events, tempos, meters, ends = [], Counter(), Counter(), []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        port = 0
        for sequence, message in enumerate(track):
            tick += message.time
            if message.type == "midi_port":
                if not 0 <= message.port <= 127:
                    raise ValueError("MIDI port outside 0..127")
                port = message.port
            elif message.type == "set_tempo":
                tempos[(Fraction(tick, ppq), message.tempo)] += 1
            elif message.type == "time_signature":
                meters[(Fraction(tick, ppq), message.numerator, message.denominator)] += 1
            elif message.type in ("note_on", "note_off", "program_change"):
                priority = 0 if message.type == "note_off" or message.type == "note_on" and message.velocity == 0 else 1 if message.type == "program_change" else 2
                events.append((tick, priority, track_index, sequence, port, message))
            elif message.type in ("marker", "lyrics", "cue_marker"):
                raise ValueError("Unexpected semantic/target marker in preview MIDI")
            elif message.type == "end_of_track":
                ends.append(Fraction(tick, ppq))
    active, programs, rows = {}, {}, Counter()
    for tick, _, _, _, port, message in sorted(events, key=lambda e: e[:4]):
        channel_key = port, message.channel
        if message.type == "program_change":
            if any(key[:2] == channel_key for key in active) and programs.get(channel_key, 0) != message.program:
                raise ValueError("Program changes under active notes")
            programs[channel_key] = message.program
            continue
        key = port, message.channel, message.note
        if message.type == "note_on" and message.velocity > 0:
            if key in active:
                raise ValueError("Overlapping global port/channel/pitch lifetimes")
            active[key] = tick, programs.get(channel_key, 0), message.velocity
        else:
            if key not in active:
                raise ValueError("Unmatched note-off")
            start, program, velocity = active.pop(key)
            if tick <= start:
                raise ValueError("Nonpositive rendered lifetime")
            rows[(message.note, Fraction(start, ppq), Fraction(tick - start, ppq), program, message.channel == 9, velocity)] += 1
    if active:
        raise ValueError("Unterminated note lifetimes")
    return dict(notes=rows, tempos=tempos, meters=meters, track_end_qn=ends, ticks_per_quarter=ppq, tracks=len(midi.tracks))


def export_preview_midi(piece: CanonicalPiece) -> tuple[bytes, dict[str, Any]]:
    """Export without modifying canonical notes; reread and verify before return."""
    validation = validate_piece(piece)
    if validation.errors:
        first = validation.errors[0]
        raise ValueError(f"Invalid canonical piece: {first.code} at {first.path}: {first.message}")
    tracks = {t.track_id: t for t in piece.tracks}
    points = [Q(0), piece.duration_qn]
    points += [p for n in piece.notes for p in (n.onset_qn, n.onset_qn + n.duration_qn)]
    points += [e.onset_qn for e in (*piece.tempo_events, *piece.meter_events, *piece.key_signature_events)]
    points += [p for b in (*piece.bars, *piece.beats) for p in (b.start_qn, b.start_qn + b.duration_qn)]
    ppq = 1
    for point in points:
        ppq = lcm(ppq, point.den)
        if ppq > 32767:
            raise ValueError(f"Exact canonical timing requires PPQ {ppq}, above MIDI limit 32767; no quantization permitted")

    def ticks(time):
        value = time.to_fraction() * ppq
        if value.denominator != 1:
            raise ValueError("Internal nonintegral tick conversion")
        return value.numerator

    omitted, default_velocity, default_program, inherited_program = [], [], [], []
    groups, expected = defaultdict(list), Counter()
    for note in piece.notes:
        if note.duration_qn == Q(0) and note.is_grace:
            omitted.append(note.note_id)
            continue
        if note.duration_qn <= Q(0):
            raise ValueError(f"Nonpositive non-grace duration: {note.note_id}")
        velocity = 80 if note.velocity is None else note.velocity
        if velocity == 0:
            raise ValueError(f"Positive note with velocity=0 cannot preserve a MIDI note lifetime: {note.note_id}")
        if note.velocity is None:
            default_velocity.append(note.note_id)
        program = note.program
        if program is None:
            program = tracks[note.track_id].program
            if program is None:
                program = 0
                default_program.append(note.note_id)
            else:
                inherited_program.append(note.note_id)
        group = (note.track_id, program, note.is_percussion, -1 if note.staff is None else note.staff, -1 if note.voice is None else note.voice)
        groups[group].append((note, velocity))
        expected[(note.pitch, note.onset_qn.to_fraction(), note.duration_qn.to_fraction(), program, note.is_percussion, velocity)] += 1

    route_specs = []
    tonal_index = drum_index = 0
    for group in sorted(groups):
        lanes = []
        for note, velocity in sorted(groups[group], key=lambda e: (e[0].onset_qn, e[0].pitch, e[0].note_id)):
            for lane in lanes:
                if lane["pitch_end"].get(note.pitch, Q(0)) <= note.onset_qn:
                    break
            else:
                lane = dict(pitch_end={}, notes=[])
                lanes.append(lane)
            lane["pitch_end"][note.pitch] = note.onset_qn + note.duration_qn
            lane["notes"].append((note, velocity))
        for lane_index, lane in enumerate(lanes):
            if group[2]:
                port, channel = drum_index, 9
                drum_index += 1
            else:
                port, channel = divmod(tonal_index, len(TONAL_CHANNELS))
                channel = TONAL_CHANNELS[channel]
                tonal_index += 1
            if port > 127:
                raise ValueError("Independent voice routes exceed MIDI port limit 0..127")
            route_specs.append(dict(group=group, lane=lane_index, port=port, channel=channel, notes=lane["notes"]))

    end_tick = ticks(piece.duration_qn)
    conductor = [(0, -1, 0, mido.MetaMessage("track_name", name="Canonical conductor"))]
    for i, meter in enumerate(piece.meter_events):
        conductor.append((ticks(meter.onset_qn), 0, i, mido.MetaMessage("time_signature", numerator=meter.numerator, denominator=meter.denominator)))
    for i, tempo in enumerate(piece.tempo_events):
        conductor.append((ticks(tempo.onset_qn), 1, i, mido.MetaMessage("set_tempo", tempo=tempo.microseconds_per_quarter)))
    midi = mido.MidiFile(type=1, ticks_per_beat=ppq)
    midi.tracks.append(_message_track(conductor, end_tick))
    routes = []
    for index, route in enumerate(route_specs):
        group, channel = route["group"], route["channel"]
        events = [(0, -3, 0, mido.MetaMessage("midi_port", port=route["port"])), (0, -2, 0, mido.MetaMessage("track_name", name=f"Preview route {index:04d}")), (0, -1, 0, mido.Message("program_change", channel=channel, program=group[1]))]
        for i, (note, velocity) in enumerate(route["notes"]):
            events.append((ticks(note.onset_qn), 1, i, mido.Message("note_on", channel=channel, note=note.pitch, velocity=velocity)))
            events.append((ticks(note.onset_qn + note.duration_qn), 0, i, mido.Message("note_off", channel=channel, note=note.pitch, velocity=0)))
        midi.tracks.append(_message_track(events, end_tick))
        routes.append(dict(track_id=group[0], program=group[1], is_percussion=group[2], staff=None if group[3] == -1 else group[3], voice=None if group[4] == -1 else group[4], lane=route["lane"], port=route["port"], channel=channel, note_ids=[n.note_id for n, _ in route["notes"]]))
    output = BytesIO()
    midi.save(file=output)
    payload = output.getvalue()
    audited = inspect_preview_midi(payload)
    expected_tempos = Counter((e.onset_qn.to_fraction(), e.microseconds_per_quarter) for e in piece.tempo_events)
    expected_meters = Counter((e.onset_qn.to_fraction(), e.numerator, e.denominator) for e in piece.meter_events)
    if audited["notes"] != expected or audited["tempos"] != expected_tempos or audited["meters"] != expected_meters:
        raise ValueError("MIDI reread differs from canonical notes/programs/tempo/meter")
    if len(audited["track_end_qn"]) != len(midi.tracks) or any(t != piece.duration_qn.to_fraction() for t in audited["track_end_qn"]):
        raise ValueError("MIDI track end differs from canonical duration")
    ports = sorted({r["port"] for r in routes})
    report = dict(export_version=EXPORT_VERSION, piece_id=piece.piece_id, ticks_per_quarter=ppq, exact_timing=True, timing_quantized=False, notes=sum(expected.values()), tracks=len(midi.tracks), routes=routes, ports=ports, requires_multiport_player=len(ports) > 1, player_limitation="Players ignoring MIDI-port metadata may merge instruments or note lifetimes; companion WAV is the primary listening reference", omitted_zero_duration_grace_ids=sorted(omitted), default_velocity=80, default_velocity_note_ids=sorted(default_velocity), default_program=0, default_program_note_ids=sorted(default_program), inherited_track_program_note_ids=sorted(inherited_program), note_roundtrip=True, tempo_roundtrip=True, meter_roundtrip=True, global_port_channel_pitch_lifetimes_valid=True, target_markers=0, click_notes=0, key_signature_metadata_exported=False, midi_sha256=hashlib.sha256(payload).hexdigest(), midi_byte_size=len(payload))
    return payload, report
