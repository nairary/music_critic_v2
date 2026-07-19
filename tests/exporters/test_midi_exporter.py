from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import mido
import pytest

from music_critic.adapters import MidiAdapterConfig, load_midi_piece
from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    convert_hooktheory_record,
)
from music_critic.data import (
    AnnotationSpan,
    RationalTime,
    TargetArray,
    TempoEvent,
    load_piece,
)
from music_critic.exporters import (
    MidiRenderConfig,
    MidiRenderError,
    piece_to_midi_bytes,
    write_piece_midi,
)


@pytest.fixture
def valid_piece():
    piece = load_piece(
        Path(__file__).parents[1] / "fixtures" / "data" / "canonical_piece_v2.json"
    )
    return replace(
        piece,
        notes=tuple(note for note in piece.notes if note.duration_qn > RationalTime(0)),
    )


def _absolute_messages(track: mido.MidiTrack) -> list[tuple[int, mido.Message]]:
    tick = 0
    result = []
    for message in track:
        tick += message.time
        result.append((tick, message))
    return result


def test_renders_format_one_conductor_notes_and_click(valid_piece) -> None:
    payload, report = piece_to_midi_bytes(
        valid_piece,
        config=MidiRenderConfig(
            include_click_track=True, include_target_markers=False
        ),
    )
    midi = mido.MidiFile(file=BytesIO(payload))

    assert midi.type == 1
    assert midi.ticks_per_beat == 8
    assert len(midi.tracks) == 3
    assert midi.tracks[0].name == "Canonical Conductor"
    assert midi.tracks[-1].name == "Canonical Click"
    conductor = _absolute_messages(midi.tracks[0])
    assert [(tick, message.tempo) for tick, message in conductor if message.type == "set_tempo"] == [(0, 500_000)]
    assert [
        (tick, message.numerator, message.denominator)
        for tick, message in conductor
        if message.type == "time_signature"
    ] == [(0, 4, 4)]
    click_ons = [
        (tick, message.note, message.velocity)
        for tick, message in _absolute_messages(midi.tracks[-1])
        if message.type == "note_on" and message.velocity > 0
    ]
    assert click_ons == [
        (
            beat.start_qn.num * 8 // beat.start_qn.den,
            37 if beat.is_downbeat else 31,
            80,
        )
        for beat in valid_piece.beats
    ]
    assert report.rendered_notes == sum(
        not note.is_percussion for note in valid_piece.notes
    )
    assert report.rendered_clicks == len(valid_piece.beats)


def test_auto_ppq_is_denominator_lcm(valid_piece) -> None:
    notes = (
        replace(valid_piece.notes[0], duration_qn=RationalTime(1, 7)),
        valid_piece.notes[1],
    )
    piece = replace(valid_piece, notes=notes)

    payload, _ = piece_to_midi_bytes(
        piece, config=MidiRenderConfig(include_click_track=False)
    )
    midi = mido.MidiFile(file=BytesIO(payload))

    assert midi.ticks_per_beat == 7


def test_explicit_exact_ppq_is_preserved(valid_piece) -> None:
    payload, report = piece_to_midi_bytes(
        valid_piece,
        config=MidiRenderConfig(
            ticks_per_quarter=8,
            include_click_track=False,
            include_target_markers=False,
        ),
    )

    assert mido.MidiFile(file=BytesIO(payload)).ticks_per_beat == 8
    assert report.exact_timing
    assert report.maximum_quantization_error_qn == RationalTime(0)


def test_explicit_inexact_ppq_requires_opt_in(valid_piece, tmp_path) -> None:
    piece = replace(
        valid_piece,
        tempo_events=valid_piece.tempo_events
        + (
            TempoEvent(
                tempo_event_id="tempo:fractional",
                onset_qn=RationalTime(1, 2),
                microseconds_per_quarter=400_000,
                provenance_id="prov:source",
            ),
        ),
    )
    with pytest.raises(MidiRenderError, match="cannot represent"):
        piece_to_midi_bytes(
            piece,
            config=MidiRenderConfig(
                ticks_per_quarter=3, include_click_track=False
            ),
        )

    report = write_piece_midi(
        piece,
        tmp_path / "quantized.mid",
        config=MidiRenderConfig(
            ticks_per_quarter=3,
            require_exact_timing=False,
            include_click_track=False,
        ),
    )

    assert report.ticks_per_quarter == 3
    assert report.exact_timing is False
    assert report.timing_quantized is True
    assert report.maximum_quantization_error_qn == RationalTime(1, 6)


def test_excessive_exact_ppq_fails_or_uses_quantization_fallback(
    valid_piece, tmp_path
) -> None:
    piece = replace(
        valid_piece,
        tempo_events=valid_piece.tempo_events
        + (
            TempoEvent(
                tempo_event_id="tempo:large-denominator",
                onset_qn=RationalTime(1, 32_771),
                microseconds_per_quarter=400_000,
                provenance_id="prov:source",
            ),
        ),
    )
    with pytest.raises(MidiRenderError, match="above MIDI limit"):
        piece_to_midi_bytes(
            piece, config=MidiRenderConfig(include_click_track=False)
        )

    report = write_piece_midi(
        piece,
        tmp_path / "fallback.mid",
        config=MidiRenderConfig(
            require_exact_timing=False, include_click_track=False
        ),
    )

    assert report.ticks_per_quarter == 960
    assert report.timing_quantized
    assert report.maximum_quantization_error_qn == RationalTime(1, 32_771)


def test_round_trip_preserves_note_tempo_meter_and_duration(valid_piece, tmp_path) -> None:
    path = tmp_path / "roundtrip.mid"
    report = write_piece_midi(
        valid_piece,
        path,
        config=MidiRenderConfig(
            include_click_track=False, include_target_markers=False
        ),
    )
    loaded = load_midi_piece(path, config=MidiAdapterConfig(dataset_name="roundtrip"))

    expected_notes = sorted(
        (
            note.pitch,
            note.onset_qn,
            note.duration_qn,
            note.velocity,
            note.channel,
            note.program if note.program is not None else 0,
            note.is_percussion,
        )
        for note in valid_piece.notes
        if not note.is_percussion
    )
    actual_notes = sorted(
        (
            note.pitch,
            note.onset_qn,
            note.duration_qn,
            note.velocity,
            note.channel,
            note.program,
            note.is_percussion,
        )
        for note in loaded.notes
    )
    assert actual_notes == expected_notes
    assert [
        (event.onset_qn, event.microseconds_per_quarter)
        for event in loaded.tempo_events
    ] == [(RationalTime(0), 500_000)]
    assert [
        (event.onset_qn, event.numerator, event.denominator)
        for event in loaded.meter_events
    ] == [(RationalTime(0), 4, 4)]
    assert loaded.duration_qn == valid_piece.duration_qn
    assert report.exact_timing


def test_invalid_piece_and_config_fail_clearly(valid_piece) -> None:
    invalid = replace(valid_piece, duration_qn=RationalTime(1))
    with pytest.raises(MidiRenderError, match="canonical piece is invalid"):
        piece_to_midi_bytes(invalid)
    with pytest.raises(MidiRenderError, match="melody_velocity"):
        piece_to_midi_bytes(
            valid_piece, config=MidiRenderConfig(melody_velocity=0)
        )
    grace = replace(
        valid_piece.notes[0], duration_qn=RationalTime(0), is_grace=True
    )
    grace_piece = replace(valid_piece, notes=(grace,) + valid_piece.notes[1:])
    with pytest.raises(MidiRenderError, match="positive note duration"):
        piece_to_midi_bytes(grace_piece)


def test_target_markers_encode_key_chord_and_chord_end(valid_piece) -> None:
    key = AnnotationSpan(
        annotation_id="span:key-marker",
        annotation_type="theory.local_key",
        layer="target_alignment",
        start_qn=RationalTime(0),
        end_qn=valid_piece.duration_qn,
        track_id=None,
        value=None,
        provenance_id="prov:theory",
    )
    chord = AnnotationSpan(
        annotation_id="span:chord-marker",
        annotation_type="theory.chord",
        layer="target_alignment",
        start_qn=RationalTime(1),
        end_qn=RationalTime(3),
        track_id=None,
        value=None,
        provenance_id="prov:theory",
    )

    def marker_target(task: str, entity_id: str, value: str) -> TargetArray:
        return TargetArray(
            target_id=f"target:marker-{task.rsplit('.', 1)[-1]}",
            task=task,
            annotation_view_id=None,
            alignment_type="annotation_span",
            entity_ids=(entity_id,),
            value_type="categorical",
            class_labels=None,
            values=(value,),
            mask=(True,),
            confidence=(None,),
            source=("dataset",),
            provenance=("prov:theory",),
        )

    targets = valid_piece.targets + (
        marker_target("theory.local_key.tonic_pc", key.annotation_id, "0"),
        marker_target("theory.local_key.mode", key.annotation_id, "major"),
        marker_target("theory.chord.root_degree", chord.annotation_id, "0"),
        marker_target("theory.chord.extent", chord.annotation_id, "7"),
        marker_target("theory.chord.inversion", chord.annotation_id, "1"),
    )
    piece = replace(
        valid_piece,
        annotations=tuple(
            sorted(
                valid_piece.annotations + (key, chord),
                key=lambda item: (item.start_qn, item.end_qn, item.annotation_id),
            )
        ),
        targets=tuple(
            sorted(
                targets,
                key=lambda item: (
                    item.task,
                    item.annotation_view_id is not None,
                    item.annotation_view_id or "",
                    item.target_id,
                ),
            )
        ),
    )

    payload, _ = piece_to_midi_bytes(
        piece, config=MidiRenderConfig(include_click_track=False)
    )
    midi = mido.MidiFile(file=BytesIO(payload))
    markers = [
        (tick, message.text)
        for tick, message in _absolute_messages(midi.tracks[0])
        if message.type == "marker"
    ]

    assert markers == [
        (0, "KEY tonic=0 mode=major"),
        (2, "CHORD root=0 extent=7 inversion=1"),
        (6, "CHORD_END"),
    ]


@pytest.mark.parametrize(
    ("num_beats", "beat_unit", "end_beat", "expected_ppq", "expected_seconds"),
    ((12, 3, 13, 8, 1.999998), (4, 1, 5, 8, 2.0)),
)
def test_complete_bar_playback_duration_for_compound_and_simple_tempo(
    num_beats,
    beat_unit,
    end_beat,
    expected_ppq,
    expected_seconds,
) -> None:
    piece = convert_hooktheory_record(
        "tempo-control",
        {
            "hash": "tempo-control",
            "split": "train",
            "json": {
                "endBeat": end_beat,
                "meters": [
                    {"beat": 1, "numBeats": num_beats, "beatUnit": beat_unit}
                ],
                "tempos": [{"beat": 1, "bpm": 120}],
                "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
                "notes": [
                    {
                        "beat": 1,
                        "duration": end_beat - 1,
                        "sd": "1",
                        "octave": 0,
                        "isRest": False,
                    }
                ],
                "chords": [],
            },
        },
        config=HookTheoryAdapterConfig(dataset_name="synthetic"),
        source_path="data/HookTheory/Hooktheory_Raw.json/4_merged.json",
    )

    payload, _ = piece_to_midi_bytes(piece)
    midi = mido.MidiFile(file=BytesIO(payload))

    assert midi.ticks_per_beat == expected_ppq
    assert midi.length == pytest.approx(expected_seconds, abs=1e-9)


def test_multiple_global_events_note_boundary_order_and_end_of_track() -> None:
    piece = convert_hooktheory_record(
        "event-order",
        {
            "hash": "event-order",
            "split": "train",
            "json": {
                "endBeat": 8,
                "meters": [
                    {"beat": 1, "numBeats": 4, "beatUnit": 1},
                    {"beat": 5, "numBeats": 6, "beatUnit": 3},
                ],
                "tempos": [
                    {"beat": 1, "bpm": 120},
                    {"beat": 3, "bpm": 100},
                ],
                "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
                "notes": [
                    {
                        "beat": 1,
                        "duration": 1,
                        "sd": "1",
                        "octave": 0,
                        "isRest": False,
                    },
                    {
                        "beat": 2,
                        "duration": 1,
                        "sd": "1",
                        "octave": 0,
                        "isRest": False,
                    },
                ],
                "chords": [],
            },
        },
        config=HookTheoryAdapterConfig(dataset_name="synthetic"),
        source_path="data/HookTheory/Hooktheory_Raw.json/4_merged.json",
    )

    payload, report = piece_to_midi_bytes(
        piece,
        config=MidiRenderConfig(
            include_click_track=False, include_target_markers=False
        ),
    )
    midi = mido.MidiFile(file=BytesIO(payload))
    conductor = _absolute_messages(midi.tracks[0])

    assert [
        (tick, message.tempo)
        for tick, message in conductor
        if message.type == "set_tempo"
    ] == [(0, 500_000), (4, 600_000)]
    assert [
        (tick, message.numerator, message.denominator)
        for tick, message in conductor
        if message.type == "time_signature"
    ] == [(0, 4, 4), (8, 6, 8)]
    boundary = [
        message.type
        for tick, message in _absolute_messages(midi.tracks[1])
        if tick == 2 and message.type in {"note_on", "note_off"}
    ]
    assert boundary == ["note_off", "note_on"]
    end_tick = piece.duration_qn.num * midi.ticks_per_beat // piece.duration_qn.den
    assert all(
        _absolute_messages(track)[-1] == (end_tick, track[-1])
        for track in midi.tracks
    )
    assert report.rendered_tempo_events == 2
    assert report.rendered_meter_events == 2


@pytest.mark.parametrize(
    "mode",
    (
        "major",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "minor",
        "locrian",
        "harmonic minor",
        "phrygian dominant",
    ),
)
def test_all_observed_scale_families_survive_hooktheory_midi_round_trip(
    mode: str, tmp_path: Path
) -> None:
    clip_id = f"mode-{mode.replace(' ', '-')}"
    piece = convert_hooktheory_record(
        clip_id,
        {
            "hash": clip_id,
            "split": "train",
            "json": {
                "endBeat": 5,
                "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
                "tempos": [{"beat": 1, "bpm": 120}],
                "keys": [{"beat": 1, "tonic": "C", "scale": mode}],
                "notes": [
                    {
                        "beat": 1,
                        "duration": 1,
                        "sd": "3",
                        "octave": 0,
                        "isRest": False,
                    }
                ],
                "chords": [],
            },
        },
        config=HookTheoryAdapterConfig(dataset_name="synthetic"),
        source_path="data/HookTheory/Hooktheory_Raw.json/4_merged.json",
    )
    path = tmp_path / f"{mode.replace(' ', '-')}.mid"
    report = write_piece_midi(
        piece,
        path,
        config=MidiRenderConfig(
            include_click_track=False, include_target_markers=False
        ),
    )

    loaded = load_midi_piece(
        path, config=MidiAdapterConfig(dataset_name="mode-roundtrip")
    )

    assert report.exact_timing
    assert [(note.pitch, note.onset_qn, note.duration_qn) for note in loaded.notes] == [
        (piece.notes[0].pitch, piece.notes[0].onset_qn, piece.notes[0].duration_qn)
    ]
