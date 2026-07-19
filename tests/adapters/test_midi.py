from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mido
import pytest

from music_critic.adapters import (
    MidiAdapterConfig,
    MidiAdapterError,
    load_midi_piece,
)
from music_critic.data import (
    RationalTime,
    dump_piece,
    dumps_piece,
    load_piece,
    loads_piece,
    validate_piece,
)


CONFIG = MidiAdapterConfig(dataset_name="test-midi")


def _write_midi(
    tmp_path: Path,
    tracks: list[list[mido.Message | mido.MetaMessage]],
    *,
    midi_type: int = 0,
    ticks_per_beat: int = 480,
    name: str = "fixture.mid",
) -> Path:
    midi_file = mido.MidiFile(type=midi_type, ticks_per_beat=ticks_per_beat)
    for source_messages in tracks:
        track = mido.MidiTrack()
        track.extend(source_messages)
        midi_file.tracks.append(track)
    path = tmp_path / name
    midi_file.save(path)
    return path


def _basic_note_messages(
    *,
    channel: int = 0,
    pitch: int = 60,
    duration: int = 480,
) -> list[mido.Message]:
    return [
        mido.Message("note_on", channel=channel, note=pitch, velocity=90, time=0),
        mido.Message("note_off", channel=channel, note=pitch, velocity=0, time=duration),
    ]


def _load(path: Path, *, config: MidiAdapterConfig = CONFIG):
    return load_midi_piece(str(path), config=config)


def _quality_codes(piece) -> set[str]:
    return {flag.code for flag in piece.quality_flags}


def test_type_0_single_track_midi(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    assert len(piece.tracks) == 1
    assert len(piece.notes) == 1
    assert piece.notes[0].pitch == 60


def test_type_1_multitrack_midi(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [
            [mido.MetaMessage("set_tempo", tempo=500_000, time=0)],
            _basic_note_messages(pitch=60),
            _basic_note_messages(pitch=67),
        ],
        midi_type=1,
    )
    piece = _load(path)
    assert len(piece.tracks) == 3
    assert {note.pitch for note in piece.notes} == {60, 67}


def test_one_source_track_with_two_midi_channels(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
            mido.Message("note_on", channel=1, note=64, velocity=81, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
            mido.Message("note_off", channel=1, note=64, velocity=0, time=0),
        ]],
    )
    piece = _load(path)
    assert [(track.source_track_index, track.channel) for track in piece.tracks] == [
        (0, 0),
        (0, 1),
    ]


def test_note_on_velocity_zero_is_note_off(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=60, velocity=77, time=0),
            mido.Message("note_on", note=60, velocity=0, time=360),
        ]],
    )
    note = _load(path).notes[0]
    assert note.velocity == 77
    assert note.source_duration_ticks == 360


def test_ordinary_note_off_pairs_note(tmp_path: Path) -> None:
    note = _load(_write_midi(tmp_path, [_basic_note_messages(duration=240)])).notes[0]
    assert note.duration_qn == RationalTime(1, 2)


def test_exact_rational_time_for_fractional_quarter_note(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=62, velocity=70, time=160),
            mido.Message("note_off", note=62, velocity=0, time=80),
        ]],
        ticks_per_beat=480,
    )
    note = _load(path).notes[0]
    assert note.onset_qn == RationalTime(1, 3)
    assert note.duration_qn == RationalTime(1, 6)


def test_tempo_event_at_tick_zero(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("set_tempo", tempo=600_000, time=0), *_basic_note_messages()]],
    )
    piece = _load(path)
    assert [(event.onset_qn, event.microseconds_per_quarter) for event in piece.tempo_events] == [
        (RationalTime(0), 600_000)
    ]
    assert piece.tempo_events[0].provenance_id == "prov:source"


def test_tempo_change_after_tick_zero(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.MetaMessage("set_tempo", tempo=400_000, time=240),
        ]],
    )
    piece = _load(path)
    assert [event.onset_qn for event in piece.tempo_events] == [
        RationalTime(0),
        RationalTime(1, 2),
    ]


def test_missing_tempo_produces_default(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    assert piece.tempo_events[0].microseconds_per_quarter == 500_000
    assert piece.tempo_events[0].provenance_id == "prov:default-tempo"
    assert "midi.default_tempo" in _quality_codes(piece)


def test_first_tempo_later_than_zero_produces_default_plus_observed(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("set_tempo", tempo=420_000, time=240)]],
    )
    piece = _load(path)
    assert [(event.onset_qn, event.microseconds_per_quarter) for event in piece.tempo_events] == [
        (RationalTime(0), 500_000),
        (RationalTime(1, 2), 420_000),
    ]


def test_meter_event_at_tick_zero(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0)]],
    )
    meter = _load(path).meter_events[0]
    assert (meter.numerator, meter.denominator) == (3, 4)
    assert meter.provenance_id == "prov:source"


def test_meter_change_on_bar_boundary(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=1920),
        ]],
    )
    piece = _load(path)
    assert [event.onset_qn for event in piece.meter_events] == [
        RationalTime(0),
        RationalTime(4),
    ]
    assert [bar.duration_qn for bar in piece.bars] == [RationalTime(4), RationalTime(3)]


def test_meter_change_inside_bar_raises_midi_adapter_error(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=960),
        ]],
    )
    with pytest.raises(MidiAdapterError, match=r"tick 960.*active meter 4/4"):
        _load(path)


def test_missing_meter_produces_default_4_4(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    meter = piece.meter_events[0]
    assert (meter.numerator, meter.denominator) == (4, 4)
    assert meter.provenance_id == "prov:default-meter"
    assert "midi.default_meter" in _quality_codes(piece)


def test_major_key_signature(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [[mido.MetaMessage("key_signature", key="Bb", time=0)]])
    event = _load(path).key_signature_events[0]
    assert (event.fifths, event.mode, event.raw_value) == (-2, "major", "Bb")


def test_minor_key_signature(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [[mido.MetaMessage("key_signature", key="F#m", time=0)]])
    event = _load(path).key_signature_events[0]
    assert (event.fifths, event.mode, event.raw_value) == (3, "minor", "F#m")


def test_unsupported_key_signature_produces_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_midi(tmp_path, [[]])
    fake_track = [SimpleNamespace(type="key_signature", key="H", time=0)]
    fake_midi = SimpleNamespace(type=0, ticks_per_beat=480, tracks=[fake_track])
    monkeypatch.setattr("music_critic.adapters.midi.mido.MidiFile", lambda **_: fake_midi)
    piece = _load(path)
    assert piece.key_signature_events == ()
    assert "midi.unsupported_key_signature" in _quality_codes(piece)


def test_track_name_is_preserved(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("track_name", name="Lead", time=0), *_basic_note_messages()]],
    )
    assert _load(path).tracks[0].name == "Lead"


def test_instrument_name_is_preserved(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("instrument_name", name="Clarinet", time=0), *_basic_note_messages()]],
    )
    assert _load(path).tracks[0].instrument_name == "Clarinet"


def test_initial_program_change_is_preserved(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.Message("program_change", program=41, time=0), *_basic_note_messages()]],
    )
    piece = _load(path)
    assert piece.tracks[0].program == 41
    assert piece.notes[0].program == 41


def test_mid_track_program_change_produces_diagnostic(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("program_change", program=1, time=0),
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.Message("program_change", program=2, time=120),
            mido.Message("note_off", note=60, velocity=0, time=360),
        ]],
    )
    piece = _load(path)
    assert piece.tracks[0].program == 1
    assert "midi.mid_track_program_change" in _quality_codes(piece)
    assert "ticks=120" in next(
        flag.message for flag in piece.quality_flags if flag.code == "midi.mid_track_program_change"
    )


def test_channel_9_is_percussion(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages(channel=9, pitch=36)]))
    assert piece.tracks[0].is_percussion is True
    assert piece.notes[0].is_percussion is True


def test_overlapping_same_pitch_notes_use_fifo_pairing(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=60, velocity=70, time=0),
            mido.Message("note_on", note=60, velocity=80, time=120),
            mido.Message("note_off", note=60, velocity=0, time=120),
            mido.Message("note_off", note=60, velocity=0, time=120),
        ]],
    )
    notes = _load(path).notes
    assert [(note.source_onset_ticks, note.source_duration_ticks, note.velocity) for note in notes] == [
        (0, 240, 70),
        (120, 240, 80),
    ]


def test_identical_pitch_on_different_channels_is_not_mixed(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", channel=0, note=60, velocity=70, time=0),
            mido.Message("note_on", channel=1, note=60, velocity=80, time=0),
            mido.Message("note_off", channel=1, note=60, velocity=0, time=120),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=120),
        ]],
    )
    notes = _load(path).notes
    assert {(note.channel, note.source_duration_ticks) for note in notes} == {(0, 240), (1, 120)}


def test_identical_pitch_on_different_source_tracks_is_not_mixed(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [
            _basic_note_messages(pitch=60, duration=120),
            _basic_note_messages(pitch=60, duration=360),
        ],
        midi_type=1,
    )
    notes = _load(path).notes
    assert {(note.track_id, note.source_duration_ticks) for note in notes} == {
        ("track:t0000-c00", 120),
        ("track:t0001-c00", 360),
    }


def test_unmatched_note_off_produces_diagnostic_without_invented_note(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.Message("note_off", note=60, velocity=0, time=120)]],
    )
    piece = _load(path)
    assert piece.notes == ()
    assert "midi.unmatched_note_off" in _quality_codes(piece)


def test_dangling_note_on_produces_diagnostic_without_invented_note(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.Message("note_on", note=60, velocity=90, time=120)]],
    )
    piece = _load(path)
    assert piece.notes == ()
    assert "midi.dangling_note_on" in _quality_codes(piece)


def test_real_source_zero_duration_note_is_grace_like(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.Message("note_off", note=60, velocity=0, time=0),
        ]],
    )
    piece = _load(path)
    assert len(piece.notes) == 1
    assert piece.notes[0].duration_qn == RationalTime(0)
    assert piece.notes[0].is_grace is True
    assert "midi.zero_duration_note" in _quality_codes(piece)


def test_note_crossing_bar_boundary_is_not_split(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=60, velocity=90, time=1680),
            mido.Message("note_off", note=60, velocity=0, time=480),
        ]],
    )
    piece = _load(path)
    assert len(piece.notes) == 1
    assert piece.notes[0].onset_qn == RationalTime(7, 2)
    assert piece.notes[0].duration_qn == RationalTime(1)
    assert len(piece.bars) == 2


def test_note_crossing_tempo_boundary_is_not_split(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.MetaMessage("set_tempo", tempo=400_000, time=240),
            mido.Message("note_off", note=60, velocity=0, time=240),
        ]],
    )
    piece = _load(path)
    assert len(piece.notes) == 1
    assert piece.notes[0].duration_qn == RationalTime(1)
    assert len(piece.tempo_events) == 2


def test_note_crossing_meter_boundary_is_not_split(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.Message("note_on", note=60, velocity=90, time=1440),
            mido.MetaMessage("time_signature", numerator=3, denominator=4, time=480),
            mido.Message("note_off", note=60, velocity=0, time=480),
        ]],
    )
    piece = _load(path)
    assert len(piece.notes) == 1
    assert piece.notes[0].onset_qn == RationalTime(3)
    assert piece.notes[0].duration_qn == RationalTime(2)
    assert len(piece.meter_events) == 2


def test_incomplete_final_bar(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages(duration=600)]))
    assert piece.bars[-1].is_incomplete is True
    assert piece.bars[-1].duration_qn == RationalTime(5, 4)
    assert "INCOMPLETE_FINAL_BAR" in {issue.code for issue in validate_piece(piece).warnings}


def test_empty_source_track_creates_empty_canonical_track(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[mido.MetaMessage("track_name", name="Conductor", time=0)]],
    )
    piece = _load(path)
    assert len(piece.tracks) == 1
    assert piece.tracks[0].channel is None
    assert piece.tracks[0].name == "Conductor"


def test_empty_midi_has_minimal_valid_timeline(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [[]]))
    assert piece.notes == ()
    assert len(piece.bars) == 1
    assert piece.duration_qn == RationalTime(4)
    assert len(piece.beats) == 4
    assert not validate_piece(piece).errors


def test_midi_type_2_is_rejected(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [[], []], midi_type=2)
    with pytest.raises(MidiAdapterError, match="MIDI type 2"):
        _load(path)


def test_smpte_timing_is_rejected(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [[]])
    payload = bytearray(path.read_bytes())
    payload[12:14] = bytes((0xE7, 0x28))
    path.write_bytes(payload)
    with pytest.raises(MidiAdapterError, match="SMPTE/non-PPQN"):
        _load(path)


def test_deterministic_ids_do_not_depend_on_absolute_path(tmp_path: Path) -> None:
    first = _write_midi(tmp_path, [_basic_note_messages()], name="first.mid")
    second = tmp_path / "second.mid"
    second.write_bytes(first.read_bytes())
    piece_a = _load(first)
    piece_b = _load(second)
    assert piece_a.piece_id == piece_b.piece_id
    assert [track.track_id for track in piece_a.tracks] == [track.track_id for track in piece_b.tracks]
    assert [note.note_id for note in piece_a.notes] == [note.note_id for note in piece_b.notes]
    assert [bar.bar_id for bar in piece_a.bars] == [bar.bar_id for bar in piece_b.bars]
    assert [beat.beat_id for beat in piece_a.beats] == [beat.beat_id for beat in piece_b.beats]


def test_repeated_conversion_is_deterministic(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [_basic_note_messages()])
    assert _load(path) == _load(path)


def test_generic_midi_annotations_are_empty(tmp_path: Path) -> None:
    assert _load(_write_midi(tmp_path, [_basic_note_messages()])).annotations == ()


def test_generic_midi_targets_are_empty(tmp_path: Path) -> None:
    assert _load(_write_midi(tmp_path, [_basic_note_messages()])).targets == ()


def test_output_passes_validate_piece(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    assert validate_piece(piece).errors == ()


def test_json_string_round_trip_preserves_equality(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    assert loads_piece(dumps_piece(piece)) == piece


def test_json_file_round_trip_preserves_equality(tmp_path: Path) -> None:
    piece = _load(_write_midi(tmp_path, [_basic_note_messages()]))
    output = tmp_path / "piece.json"
    dump_piece(piece, output)
    assert load_piece(output) == piece


def test_importing_music_critic_data_does_not_import_mido() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import music_critic.data; assert 'mido' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_importing_adapters_exposes_accepted_api() -> None:
    import music_critic.adapters as adapters

    assert adapters.MidiAdapterConfig is MidiAdapterConfig
    assert adapters.MidiAdapterError is MidiAdapterError
    assert adapters.load_midi_piece is load_midi_piece


def test_conflicting_global_events_use_deterministic_source_order(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [
            [
                mido.MetaMessage("set_tempo", tempo=510_000, time=0),
                mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
                mido.MetaMessage("key_signature", key="C", time=0),
            ],
            [
                mido.MetaMessage("set_tempo", tempo=610_000, time=0),
                mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0),
                mido.MetaMessage("key_signature", key="F", time=0),
            ],
        ],
        midi_type=1,
    )
    piece = _load(path)
    assert piece.tempo_events[0].microseconds_per_quarter == 510_000
    assert (piece.meter_events[0].numerator, piece.meter_events[0].denominator) == (4, 4)
    assert piece.key_signature_events[0].raw_value == "C"
    assert {
        "midi.tempo_conflict",
        "midi.meter_conflict",
        "midi.key_signature_conflict",
    } <= _quality_codes(piece)


def test_exact_duplicate_global_events_are_deduplicated(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [
            [mido.MetaMessage("set_tempo", tempo=500_000, time=0)],
            [mido.MetaMessage("set_tempo", tempo=500_000, time=0)],
        ],
        midi_type=1,
    )
    piece = _load(path)
    assert len(piece.tempo_events) == 1
    assert "midi.tempo_conflict" not in _quality_codes(piece)


def test_program_active_at_first_note_on_is_preserved(tmp_path: Path) -> None:
    path = _write_midi(
        tmp_path,
        [[
            mido.Message("program_change", program=2, time=0),
            mido.Message("program_change", program=12, time=120),
            mido.Message("note_on", note=60, velocity=90, time=120),
            mido.Message("note_off", note=60, velocity=0, time=240),
        ]],
    )
    piece = _load(path)
    assert piece.tracks[0].program == 12
    assert piece.notes[0].program == 12


def test_validation_failure_is_exposed_on_adapter_error(tmp_path: Path) -> None:
    path = _write_midi(tmp_path, [_basic_note_messages()])
    with pytest.raises(MidiAdapterError) as caught:
        _load(path, config=MidiAdapterConfig(dataset_name=""))
    assert caught.value.validation_report is not None
    assert caught.value.validation_report.errors
    assert "FIELD_VALUE_INVALID" in str(caught.value)


def test_corrupted_midi_is_wrapped_in_adapter_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.mid"
    path.write_bytes(b"not a midi file")
    with pytest.raises(MidiAdapterError, match="corrupted or unreadable"):
        _load(path)
