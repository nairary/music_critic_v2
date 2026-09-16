from __future__ import annotations

from dataclasses import replace

import pytest

from music_critic.data import CanonicalPiece, RationalTime
from music_critic.data.voice_continuation import (
    VoiceAnnotation, VoiceAnnotationView, VoiceCandidatePolicy,
    build_voice_candidates, map_voice_continuation_targets,
)


@pytest.fixture
def crossing_piece(valid_piece: CanonicalPiece) -> CanonicalPiece:
    """Two voices cross pitch rank while sharing one physical MIDI track."""
    template = valid_piece.notes[0]
    notes = tuple(
        replace(
            template, note_id=f"note:{name}", pitch=pitch,
            onset_qn=RationalTime(onset), duration_qn=RationalTime(3, 2),
            voice=voice, staff=None,
        )
        for name, onset, pitch, voice in (
            ("a0", 0, 60, 1), ("b0", 0, 72, 2),
            ("a1", 1, 74, 1), ("b1", 1, 58, 2),
            ("a2", 2, 76, 1), ("b2", 2, 56, 2),
        )
    )
    return replace(valid_piece, notes=notes, targets=())


def _view(piece: CanonicalPiece, *, confidence=0.8) -> VoiceAnnotationView:
    # This mapping is hand-authored fixture truth, not inferred from pitch/track.
    assignments = {"a0": "1", "a1": "1", "a2": "1", "b0": "2", "b1": "2", "b2": "2"}
    return VoiceAnnotationView(
        piece.piece_id, "synthetic.manual_voices",
        tuple(VoiceAnnotation(
            note.note_id, ("score-part:one", assignments[note.note_id[5:]]),
            "synthetic", "prov:source", confidence,
        ) for note in piece.notes),
        piece.provenance,
    )


def _targets(piece: CanonicalPiece, view: VoiceAnnotationView | None = None, **policy):
    return map_voice_continuation_targets(
        build_voice_candidates(piece, VoiceCandidatePolicy(**policy)), view or _view(piece),
    )


def test_crossing_voices_multiple_voices_one_track(crossing_piece):
    result = _targets(crossing_piece)
    available = {row.source_note_id: row.next_note_id for row in result.targets if row.mask}
    assert available == {
        "note:a0": "note:a1", "note:a1": "note:a2",
        "note:b0": "note:b1", "note:b1": "note:b2",
    }
    # Pitch continuity/next-in-track chooses b1 first, but gold voice a chooses a1.
    row = next(row for row in result.targets if row.source_note_id == "note:a0")
    assert row.candidate_index == 1
    assert row.confidence == 0.8
    assert all(row.reason == "no_observed_successor" for row in result.targets if not row.mask)
    assert all(not row.mask and row.next_note_id is None for row in result.targets[-2:])


def test_candidates_ignore_all_optional_annotation_metadata_and_track(crossing_piece):
    original = build_voice_candidates(crossing_piece)
    changed = replace(
        crossing_piece, notes=tuple(replace(
            note, voice=None, staff=90, spelling_step=None, spelling_alter=None,
            track_id="track:merged", provenance_id=None,
        ) for note in reversed(crossing_piece.notes)),
        tracks=(), annotations=(), targets=(), provenance=(),
    )
    assert build_voice_candidates(changed) == original
    assert _targets(changed, _view(crossing_piece)).targets == _targets(crossing_piece).targets


def test_scalar_voice_is_never_automatically_gold(crossing_piece):
    view = replace(_view(crossing_piece), annotations=())
    result = _targets(crossing_piece, view)
    assert all(not row.mask and row.reason == "missing_annotation" for row in result.targets)
    assert all(row.confidence is None and row.evidence == () for row in result.targets)


def test_same_voice_can_cross_tracks(crossing_piece):
    changed = replace(crossing_piece, notes=tuple(
        replace(note, track_id="track:other") if note.note_id == "note:a1" else note
        for note in crossing_piece.notes
    ))
    assert _targets(changed).targets == _targets(crossing_piece).targets


def test_repeated_voice_number_in_different_source_scopes_is_distinct(crossing_piece):
    view = _view(crossing_piece)
    view = replace(view, annotations=tuple(replace(
        annotation, scoped_voice_id=(
            "part:a" if annotation.note_id.startswith("note:a") else "part:b", "1",
        ),
    ) for annotation in view.annotations))
    assert {row.source_note_id: row.next_note_id for row in _targets(crossing_piece, view).targets} == {
        row.source_note_id: row.next_note_id for row in _targets(crossing_piece).targets
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_intermediate_annotation_does_not_skip_unknown_note(crossing_piece, explicit):
    view = _view(crossing_piece)
    view = replace(view, annotations=tuple(
        VoiceAnnotation(row.note_id, None, None, None) if row.note_id == "note:b1" else row
        for row in view.annotations if explicit or row.note_id != "note:b1"
    ))
    result = _targets(crossing_piece, view)
    by_id = {row.source_note_id: row for row in result.targets}
    assert by_id["note:b0"].reason == "incomplete_annotation_interval"
    assert by_id["note:a0"].reason == "incomplete_annotation_interval"
    assert by_id["note:b1"].reason == "missing_annotation"
    for row in result.targets:
        if not row.mask:
            assert row.next_note_id is row.candidate_index is row.confidence is None
            assert row.evidence == ()


def test_missing_annotation_after_successor_does_not_mask_earlier_link(crossing_piece):
    view = replace(_view(crossing_piece), annotations=tuple(
        row for row in _view(crossing_piece).annotations if row.note_id != "note:b2"
    ))
    result = _targets(crossing_piece, view)
    assert next(row for row in result.targets if row.source_note_id == "note:a0").mask


def test_same_voice_simultaneity_is_ambiguous(crossing_piece):
    view = _view(crossing_piece)
    view = replace(view, annotations=tuple(
        replace(row, scoped_voice_id=("score-part:one", "1"))
        if row.note_id == "note:b1" else row for row in view.annotations
    ))
    rows = {row.source_note_id: row for row in _targets(crossing_piece, view).targets}
    assert rows["note:a0"].reason == "ambiguous_successor_onset"
    assert rows["note:a1"].reason == rows["note:b1"].reason == "ambiguous_source_onset"


@pytest.mark.parametrize("policy", [
    {"max_candidates_per_note": 1}, {"max_onset_delta_qn": RationalTime(1, 2)},
])
def test_candidate_miss_masks_instead_of_rewriting_gold(crossing_piece, policy):
    rows = {row.source_note_id: row for row in _targets(crossing_piece, **policy).targets}
    assert rows["note:a0"].reason == "successor_outside_candidates"
    assert not rows["note:a0"].mask
    assert rows["note:a0"].next_note_id is None


def test_unknown_confidence_is_not_one(crossing_piece):
    view = _view(crossing_piece, confidence=None)
    result = _targets(crossing_piece, view)
    assert any(row.mask for row in result.targets)
    assert all(row.confidence is None for row in result.targets)
    assert all(item.confidence is None for row in result.targets for item in row.evidence)


def test_confidence_preserves_all_dependency_evidence(crossing_piece):
    view = _view(crossing_piece)
    view = replace(view, annotations=tuple(replace(row, confidence=0.2)
        if row.note_id == "note:b1" else row for row in view.annotations))
    row = next(row for row in _targets(crossing_piece, view).targets if row.source_note_id == "note:a0")
    assert row.confidence == 0.2
    assert {item.note_id for item in row.evidence} == {"note:a0", "note:b0", "note:a1", "note:b1"}


def test_exact_timing_equal_onsets_excluded_and_horizon_inclusive(crossing_piece):
    piece = replace(crossing_piece, notes=tuple(replace(
        note, onset_qn=RationalTime(note.onset_qn.num, 3),
    ) for note in crossing_piece.notes))
    candidates = build_voice_candidates(piece, VoiceCandidatePolicy(max_onset_delta_qn=RationalTime(1, 3)))
    assert candidates.destination_ids[0] == ("note:b1", "note:a1")
    assert len(candidates.destination_ids[0]) == 2  # includes sustained overlap
    assert candidates.destination_ids[-1] == ()


def test_piece_boundary_is_checked_before_alignment(crossing_piece):
    with pytest.raises(ValueError, match="piece_id"):
        _targets(crossing_piece, replace(_view(crossing_piece), piece_id="piece:another"))


@pytest.mark.parametrize("mutation, message", [
    (lambda view: replace(view, annotations=view.annotations + view.annotations[:1]), "duplicate"),
    (lambda view: replace(view, annotations=(replace(view.annotations[0], note_id="note:foreign"),)), "outside"),
    (lambda view: replace(view, provenance=()), "provenance"),
])
def test_malformed_sidecars_reject(crossing_piece, mutation, message):
    with pytest.raises(ValueError, match=message):
        _targets(crossing_piece, mutation(_view(crossing_piece)))


@pytest.mark.parametrize("source", ["algorithm", "pseudo_label", "derived"])
def test_heuristic_labels_are_not_gold(source):
    with pytest.raises(ValueError, match="not heuristics"):
        VoiceAnnotation("note:a", ("scope", "voice"), source, "prov:source")


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_confidence_rejects(confidence):
    with pytest.raises(ValueError, match="confidence"):
        VoiceAnnotation("note:a", ("scope", "voice"), "dataset", "prov:source", confidence)


def test_missing_annotation_cannot_retain_evidence():
    with pytest.raises(ValueError, match="null"):
        VoiceAnnotation("note:a", None, "dataset", "prov:source")


def test_raw_bound_and_forged_candidates_fail_closed(crossing_piece):
    with pytest.raises(ValueError, match="max_notes_per_piece"):
        build_voice_candidates(crossing_piece, VoiceCandidatePolicy(max_notes_per_piece=2))
    candidates = build_voice_candidates(crossing_piece)
    forged = replace(candidates, destination_ids=(("note:a0",),) + candidates.destination_ids[1:])
    with pytest.raises(ValueError, match="raw policy"):
        map_voice_continuation_targets(forged, _view(crossing_piece))


def test_empty_piece_is_valid(crossing_piece):
    piece = replace(crossing_piece, notes=())
    result = _targets(piece)
    assert result.targets == result.candidates.notes == result.candidates.destination_ids == ()
