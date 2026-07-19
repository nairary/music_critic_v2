from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from music_critic.data import RationalTime, load_piece
from scripts.audit_hooktheory_midi_ambiguities import analyze_piece_ambiguities


def _piece():
    value = load_piece(
        Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"
    )
    track = replace(value.tracks[0], track_id="track:a", channel=0, program=0)
    return replace(value, tracks=(track,), notes=())


def _note(template, note_id, track_id, onset, duration, *, channel=0, program=0, pitch=60):
    return replace(
        template,
        note_id=note_id,
        track_id=track_id,
        onset_qn=RationalTime(*onset),
        duration_qn=RationalTime(*duration),
        channel=channel,
        program=program,
        pitch=pitch,
    )


def test_same_pitch_overlap_scope_and_nested_classification() -> None:
    source = load_piece(
        Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"
    )
    template = source.notes[0]
    piece = _piece()

    nonoverlap = replace(
        piece,
        notes=(
            _note(template, "a", "track:a", (0, 1), (1, 1)),
            _note(template, "b", "track:a", (2, 1), (1, 1)),
        ),
    )
    adjacent = replace(
        piece,
        notes=(
            _note(template, "a", "track:a", (0, 1), (1, 1)),
            _note(template, "b", "track:a", (1, 1), (1, 1)),
        ),
    )
    partial = replace(
        piece,
        notes=(
            _note(template, "a", "track:a", (0, 1), (2, 1)),
            _note(template, "b", "track:a", (1, 1), (2, 1)),
        ),
    )
    nested = replace(
        piece,
        notes=(
            _note(template, "a", "track:a", (0, 1), (3, 1)),
            _note(template, "b", "track:a", (1, 1), (1, 1)),
        ),
    )
    track_b = replace(piece.tracks[0], track_id="track:b")
    different_tracks = replace(
        piece,
        tracks=piece.tracks + (track_b,),
        notes=(
            _note(template, "a", "track:a", (0, 1), (2, 1)),
            _note(template, "b", "track:b", (1, 1), (2, 1)),
        ),
    )
    different_channels = replace(
        piece,
        notes=(
            _note(template, "a", "track:a", (0, 1), (2, 1), channel=0),
            _note(template, "b", "track:a", (1, 1), (2, 1), channel=1),
        ),
    )

    assert analyze_piece_ambiguities(nonoverlap)["same_pitch_overlap_pairs"] == 0
    assert analyze_piece_ambiguities(adjacent)["same_pitch_overlap_pairs"] == 0
    partial_report = analyze_piece_ambiguities(partial)
    assert partial_report["same_pitch_overlap_pairs"] == 1
    assert partial_report["same_pitch_nested_pairs"] == 0
    nested_report = analyze_piece_ambiguities(nested)
    assert nested_report["same_pitch_overlap_pairs"] == 1
    assert nested_report["same_pitch_nested_pairs"] == 1
    assert nested_report["same_pitch_overlap_examples"][0]["overlap_type"] == "nested"
    assert analyze_piece_ambiguities(different_tracks)["same_pitch_overlap_pairs"] == 0
    assert analyze_piece_ambiguities(different_channels)["same_pitch_overlap_pairs"] == 0


def test_program_conflict_requires_overlap_same_channel_and_different_program() -> None:
    source = load_piece(
        Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"
    )
    template = source.notes[0]
    piece = _piece()

    def report(*, right_channel=0, right_program=1, right_onset=(1, 1)):
        candidate = replace(
            piece,
            notes=(
                _note(template, "a", "track:a", (0, 1), (2, 1), channel=0, program=0),
                _note(
                    template,
                    "b",
                    "track:a",
                    right_onset,
                    (2, 1),
                    channel=right_channel,
                    program=right_program,
                    pitch=64,
                ),
            ),
        )
        return analyze_piece_ambiguities(candidate)

    assert report(right_program=0)["channel_program_conflict_pairs"] == 0
    assert report(right_channel=1)["channel_program_conflict_pairs"] == 0
    assert report(right_onset=(2, 1))["channel_program_conflict_pairs"] == 0
    conflict = report()
    assert conflict["channel_program_conflict_pairs"] == 1
    assert conflict["channel_program_conflict_examples"][0]["channel"] == 0
    assert conflict["channel_program_conflict_examples"][0]["programs"] == [0, 1]
