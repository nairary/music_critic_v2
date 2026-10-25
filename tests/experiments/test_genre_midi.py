"""Synthetic exact SMF export and independently reread routing checks."""
from dataclasses import replace
from fractions import Fraction
import importlib.util
from io import BytesIO
from pathlib import Path

import mido
import pytest

from music_critic.data import load_piece, RationalTime as Q


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("genre_midi", ROOT / "scripts/export_genre_midi.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def piece():
    source = load_piece(ROOT / "tests/fixtures/data/canonical_piece_v2.json")
    note = next(n for n in source.notes if not n.is_percussion)
    return replace(source, notes=(replace(note, onset_qn=Q(0), duration_qn=Q(1)),), targets=(), annotations=(), quality_flags=())


def test_two_tracks_different_programs_and_percussion_preserved():
    source = piece()
    template = source.notes[0]
    guitar = replace(source.tracks[0], track_id="track:guitar", source_track_index=0, program=25)
    bass = replace(guitar, track_id="track:bass", source_track_index=1, program=33)
    drums = replace(guitar, track_id="track:drums", source_track_index=2, program=0, is_percussion=True, channel=9)
    notes = tuple(replace(template, note_id="note:" + t.track_id.split(":")[1], track_id=t.track_id, program=t.program, channel=t.channel, is_percussion=t.is_percussion, pitch=36 if t.is_percussion else 60) for t in (guitar, bass, drums))
    source = replace(source, tracks=(guitar, bass, drums), notes=notes)
    payload, report = exporter.export_preview_midi(source)
    decoded = exporter.inspect_preview_midi(payload)
    assert {row[3] for row in decoded["notes"]} == {0, 25, 33}
    assert sum(v for row, v in decoded["notes"].items() if row[4]) == 1
    assert all(r["channel"] == 9 if r["is_percussion"] else r["channel"] != 9 for r in report["routes"])
    assert report["notes"] == 3 and report["target_markers"] == report["click_notes"] == 0
    assert report["note_roundtrip"] and report["global_port_channel_pitch_lifetimes_valid"]


@pytest.mark.parametrize("percussion", [False, True])
def test_overlapping_same_pitch_has_independent_lifetimes(percussion):
    source = piece()
    track = replace(source.tracks[0], is_percussion=percussion)
    template = replace(source.notes[0], track_id=track.track_id, is_percussion=percussion)
    notes = (replace(template, note_id="note:a", onset_qn=Q(0), duration_qn=Q(2)), replace(template, note_id="note:b", onset_qn=Q(1, 2), duration_qn=Q(1)))
    payload, report = exporter.export_preview_midi(replace(source, tracks=(track,), notes=notes))
    assert len(report["routes"]) == 2
    if percussion:
        assert report["ports"] == [0, 1]
        assert report["requires_multiport_player"]
    else:
        assert {r["channel"] for r in report["routes"]} == {0, 1}
    assert sum(exporter.inspect_preview_midi(payload)["notes"].values()) == 2


def test_repeated_pitch_at_exact_boundary_can_reuse_route():
    source = piece()
    a = source.notes[0]
    b = replace(a, note_id="note:b", onset_qn=Q(1))
    _, report = exporter.export_preview_midi(replace(source, notes=(a, b)))
    assert len(report["routes"]) == 1


def test_program_change_gets_separate_route_while_prior_note_is_active():
    source = piece()
    a = replace(source.notes[0], duration_qn=Q(2), program=25)
    b = replace(a, note_id="note:b", onset_qn=Q(1), duration_qn=Q(1), program=33)
    _, report = exporter.export_preview_midi(replace(source, notes=(a, b)))
    assert len(report["routes"]) == 2
    assert {r["program"] for r in report["routes"]} == {25, 33}


def test_tempo_meter_and_fractional_timing_preserved():
    source = piece()
    note = replace(source.notes[0], onset_qn=Q(1, 7), duration_qn=Q(2, 7))
    tempo = replace(source.tempo_events[0], tempo_event_id="tempo:new", onset_qn=Q(1, 3), microseconds_per_quarter=625000)
    payload, report = exporter.export_preview_midi(replace(source, notes=(note,), tempo_events=(*source.tempo_events, tempo)))
    decoded = exporter.inspect_preview_midi(payload)
    assert report["ticks_per_quarter"] == 21
    assert decoded["tempos"][(Fraction(1, 3), 625000)] == 1
    assert decoded["meters"][(Fraction(0), 4, 4)] == 1
    row = next(iter(decoded["notes"]))
    assert row[1:3] == (Fraction(1, 7), Fraction(2, 7))
    assert report["exact_timing"] and not report["timing_quantized"]


def test_meter_change_preserved_at_exact_bar_boundary():
    source = piece()
    meter = replace(source.meter_events[0], meter_event_id="meter:cut-time", onset_qn=Q(1), numerator=2, denominator=2)
    bar = replace(source.bars[1], meter_event_id=meter.meter_event_id)
    first = replace(source.beats[1], meter_event_id=meter.meter_event_id, duration_qn=Q(2))
    second = replace(first, beat_id="beat:cut-second", start_qn=Q(3), position_in_bar_qn=Q(2), index_in_bar=1, is_downbeat=False)
    changed = replace(source, meter_events=(*source.meter_events, meter), bars=(source.bars[0], bar), beats=(source.beats[0], first, second))
    payload, _ = exporter.export_preview_midi(changed)
    decoded = exporter.inspect_preview_midi(payload)
    assert decoded["meters"][(Fraction(1), 2, 2)] == 1
    assert decoded["meters"][(Fraction(0), 4, 4)] == 1


def test_only_zero_duration_graces_omitted_and_piece_unchanged():
    source = piece()
    grace = replace(source.notes[0], note_id="note:grace", onset_qn=Q(1), duration_qn=Q(0), is_grace=True)
    source = replace(source, notes=(*source.notes, grace))
    _, report = exporter.export_preview_midi(source)
    assert report["omitted_zero_duration_grace_ids"] == ["note:grace"]
    assert report["notes"] == 1 and len(source.notes) == 2
    with pytest.raises(ValueError, match="ZERO_DURATION_NON_GRACE"):
        exporter.export_preview_midi(replace(source, notes=(replace(grace, is_grace=False),)))


def test_missing_velocity_and_program_defaults_are_reported():
    source = piece()
    track = replace(source.tracks[0], program=None)
    note = replace(source.notes[0], velocity=None, program=None)
    payload, report = exporter.export_preview_midi(replace(source, tracks=(track,), notes=(note,)))
    row = next(iter(exporter.inspect_preview_midi(payload)["notes"]))
    assert row[3] == 0 and row[5] == 80
    assert report["default_program_note_ids"] == report["default_velocity_note_ids"] == [note.note_id]
    _, inherited = exporter.export_preview_midi(replace(source, tracks=(replace(track, program=40),), notes=(note,)))
    assert inherited["inherited_track_program_note_ids"] == [note.note_id]
    assert inherited["default_program_note_ids"] == []


def test_zero_velocity_not_silently_deleted_or_made_audible():
    source = piece()
    with pytest.raises(ValueError, match="velocity=0"):
        exporter.export_preview_midi(replace(source, notes=(replace(source.notes[0], velocity=0),)))


def test_ppq_limit_rejects_quantization():
    source = piece()
    with pytest.raises(ValueError, match="no quantization"):
        exporter.export_preview_midi(replace(source, notes=(replace(source.notes[0], duration_qn=Q(1, 40009)),)))


def test_many_tonal_routes_use_ports_and_never_drum_channel():
    source = piece()
    notes = tuple(replace(source.notes[0], note_id=f"note:{i:02d}", voice=i) for i in range(17))
    _, report = exporter.export_preview_midi(replace(source, notes=notes))
    assert report["ports"] == [0, 1]
    assert len(report["routes"]) == 17
    assert all(r["channel"] != 9 for r in report["routes"])
    assert "WAV" in report["player_limitation"]


def test_port_limit_enforced():
    source = piece()
    track = replace(source.tracks[0], is_percussion=True)
    notes = tuple(replace(source.notes[0], note_id=f"note:drum-{i:03d}", is_percussion=True) for i in range(129))
    with pytest.raises(ValueError, match="port limit"):
        exporter.export_preview_midi(replace(source, tracks=(track,), notes=notes))


def test_output_deterministic_and_invalid_input_order_rejected():
    source = piece()
    notes = (source.notes[0], replace(source.notes[0], note_id="note:b", onset_qn=Q(1), program=25))
    a, _ = exporter.export_preview_midi(replace(source, notes=notes))
    b, _ = exporter.export_preview_midi(replace(source, notes=notes))
    assert a == b
    with pytest.raises(ValueError, match="COLLECTION_ORDER_INVALID"):
        exporter.export_preview_midi(replace(source, notes=notes[::-1], tracks=source.tracks[::-1]))


def bad_midi(messages):
    midi = mido.MidiFile(type=1, ticks_per_beat=4)
    midi.tracks.append(mido.MidiTrack(messages))
    data = BytesIO()
    midi.save(file=data)
    return data.getvalue()


@pytest.mark.parametrize("middle,ending,expected", [
    ([mido.Message("note_on", channel=0, note=60, velocity=90, time=1)], [mido.Message("note_off", channel=0, note=60, time=1)], "Overlapping"),
    ([mido.Message("program_change", channel=0, program=1, time=1)], [mido.Message("note_off", channel=0, note=60, time=1)], "Program changes"),
    ([], [], "Unterminated"),
])
def test_independent_reader_rejects_lifetime_corruption(middle, ending, expected):
    payload = bad_midi([mido.Message("program_change", channel=0, program=0), mido.Message("note_on", channel=0, note=60, velocity=80), *middle, *ending])
    with pytest.raises(ValueError, match=expected):
        exporter.inspect_preview_midi(payload)


def test_reader_checks_same_channel_collision_across_tracks():
    source = piece()
    a = replace(source.notes[0], duration_qn=Q(2))
    b = replace(a, note_id="note:b", onset_qn=Q(1), duration_qn=Q(1))
    payload, _ = exporter.export_preview_midi(replace(source, notes=(a, b)))
    midi = mido.MidiFile(file=BytesIO(payload))
    for message in midi.tracks[-1]:
        if hasattr(message, "channel"):
            message.channel = 0
    data = BytesIO()
    midi.save(file=data)
    with pytest.raises(ValueError, match="Overlapping"):
        exporter.inspect_preview_midi(data.getvalue())
