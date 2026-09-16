from __future__ import annotations

from dataclasses import replace
import json

import pytest

from music_critic.data import RationalTime, TempoEvent, validate_or_raise
from music_critic.experiments.context_diagnostics_pairs import (
    build_context_pair_suite,
    context_pair_fingerprint,
    context_pair_to_dict,
    crop_raw_context,
    raw_context_candidates,
    sounding_note_signature,
    validate_context_pair,
)


@pytest.fixture
def suite():
    return build_context_pair_suite()


def _sort_notes(piece, notes):
    order = {t.track_id: i for i, t in enumerate(piece.tracks)}
    return tuple(sorted(notes, key=lambda n: (
        n.onset_qn, order[n.track_id], n.pitch, n.duration_qn, n.note_id,
    )))


def test_suite_is_bounded_deterministic_and_canonically_valid(suite):
    assert len(suite) == 5
    assert {pair.family for pair in suite} == {
        "harmonic_surroundings", "dissonance_preparation_resolution",
        "vertical_sonority_voice_motion", "track_repartition",
    }
    assert suite == build_context_pair_suite()
    assert len({p.pair_id for p in suite}) == len(suite)
    assert len({piece.piece_id for p in suite for piece in (p.left, p.right)}) == 10
    for pair in suite:
        assert validate_context_pair(pair).mechanically_valid
        assert not validate_context_pair(pair).failed_checks
        for piece in (pair.left, pair.right):
            validate_or_raise(piece)
            assert len(piece.notes) <= 16
            assert not piece.annotations and not piece.targets
            assert all(n.voice is None and n.staff is None for n in piece.notes)


def test_interpretation_and_review_are_not_mechanical_or_quality_labels(suite):
    for pair in suite:
        payload = context_pair_to_dict(pair)
        assert payload["mechanically_valid"] is True
        assert payload["interpretation_status"] == "hypothesis_only"
        assert payload["musician_review_status"] == "not_reviewed"
        assert payload["preferred_side"] is None
        assert payload["functional_label_transfer_allowed"] is False
        assert payload["limitations"]
        assert json.loads(json.dumps(payload)) == payload
        assert context_pair_fingerprint(pair) == context_pair_fingerprint(pair)
    assert len({context_pair_fingerprint(p) for p in suite}) == len(suite)


@pytest.mark.parametrize("index", range(5))
def test_focus_is_identical_exact_one_raw_beat(suite, index):
    pair = suite[index]
    assert (pair.focus_start_qn, pair.focus_end_qn) == (RationalTime(1), RationalTime(2))
    left = sounding_note_signature(pair.left, pair.focus_start_qn, pair.focus_end_qn)
    right = sounding_note_signature(pair.right, pair.focus_start_qn, pair.focus_end_qn)
    assert left == right
    assert all(row[0] == RationalTime(1) and row[1] == RationalTime(1) for row in left)
    if index == 0:
        assert tuple(row[2] for row in left) == (55, 59, 62, 65)
    for piece in (pair.left, pair.right):
        focus_beats = [b.beat_id for b in piece.beats if b.start_qn == pair.focus_start_qn]
        assert focus_beats == ["beat:001"]


def test_track_views_preserve_complete_sounding_event_multiset(suite):
    renumber, split = suite[3:]
    for pair in (renumber, split):
        assert sounding_note_signature(pair.left) == sounding_note_signature(pair.right)
    assert {t.track_id for t in renumber.left.tracks}.isdisjoint(
        t.track_id for t in renumber.right.tracks
    )
    assert [t.track_id for t in renumber.right.tracks] == [
        "track:020", "track:019", "track:018", "track:017",
    ]
    assert len(split.left.tracks) == 1 and len(split.right.tracks) == 4


def test_dissonance_and_motion_oracles_do_not_rely_on_theory_metadata(suite):
    dissonance, motion = suite[1:3]
    upper = lambda p: tuple(n.pitch for n in p.notes if n.track_id == "track:003")
    assert upper(dissonance.left) == (65, 65, 64, 64)
    assert upper(dissonance.right) == (69, 65, 69, 69)
    lower = lambda p: tuple(n.pitch for n in p.notes if n.track_id == "track:000")
    assert lower(motion.left) == (48, 50, 52, 53)
    assert lower(motion.right) == (55, 50, 45, 43)
    assert any("reattacked" in text for text in dissonance.limitations)
    assert any("Only the focal vertical" in text for text in motion.limitations)


def test_invariants_detect_local_mutation_and_source_or_partition_mix(suite):
    pair = suite[0]
    changed = replace(pair.right.notes[4], pitch=56)
    right = replace(pair.right, notes=_sort_notes(pair.right, (changed,) + pair.right.notes[:4] + pair.right.notes[5:]))
    result = validate_context_pair(replace(pair, right=right))
    assert not result.mechanically_valid
    assert "equal_focus_sounding_notes" in result.failed_checks
    assert "same_source_group" in validate_context_pair(replace(
        pair, right=replace(pair.right, source_group_id="different-work"),
    )).failed_checks
    assert "same_dataset_and_partition" in validate_context_pair(replace(
        pair, right=replace(pair.right, split="validation"),
    )).failed_checks
    with pytest.raises(ValueError, match="unknown context-pair invariant"):
        validate_context_pair(replace(pair, declared_invariants=("musically_better",)))


def test_omitted_focus_invariant_cannot_claim_mechanical_validity(suite):
    report = validate_context_pair(replace(suite[0], declared_invariants=()))
    assert not report.mechanically_valid
    assert "required_invariants_declared" in report.failed_checks


def test_candidates_ignore_target_annotations_and_keep_fractional_onsets(suite):
    piece = suite[0].left
    changed = replace(piece.notes[0], onset_qn=RationalTime(1, 3), duration_qn=RationalTime(2, 3))
    piece = replace(piece, notes=_sort_notes(piece, (changed,) + piece.notes[1:]))
    candidates = raw_context_candidates(piece)
    assert [(c.start_qn, c.end_qn) for c in candidates] == [
        (RationalTime(0), RationalTime(1, 3)),
        (RationalTime(1, 3), RationalTime(1)),
        (RationalTime(1), RationalTime(2)),
        (RationalTime(2), RationalTime(3)),
        (RationalTime(3), RationalTime(4)),
    ]
    assert candidates[1].onset_note_ids == (changed.note_id,)
    # Poison sidecars prove candidate construction does not even validate them.
    assert raw_context_candidates(replace(piece, annotations=(object(),), targets=(object(),))) == candidates


def test_grace_only_onsets_are_not_sounding_candidates(suite):
    piece = suite[0].left
    grace = replace(piece.notes[0], note_id="note:grace", onset_qn=RationalTime(1, 7),
                    duration_qn=RationalTime(0), is_grace=True)
    with_grace = replace(piece, notes=_sort_notes(piece, piece.notes + (grace,)))
    assert raw_context_candidates(with_grace) == raw_context_candidates(piece)


def _two_bar_piece(piece):
    second = tuple(replace(n, note_id=n.note_id + "-second", onset_qn=n.onset_qn + RationalTime(4))
                   for n in piece.notes)
    bar = replace(piece.bars[0], bar_id="bar:001", index=1, start_qn=RationalTime(4))
    beats = tuple(replace(b, beat_id=b.beat_id + "-second", bar_id="bar:001", start_qn=b.start_qn + RationalTime(4))
                  for b in piece.beats)
    # A crossing rational-duration note must be clipped, never rounded/dropped.
    crossing = replace(piece.notes[0], note_id="note:crossing", onset_qn=RationalTime(11, 3),
                       duration_qn=RationalTime(2), pitch=61)
    return replace(
        piece, duration_qn=RationalTime(8), notes=_sort_notes(piece, piece.notes + second + (crossing,)),
        bars=piece.bars + (bar,), beats=piece.beats + beats,
        tempo_events=piece.tempo_events + (TempoEvent("tempo:001", RationalTime(2), 600000,
                                                     piece.provenance[0].provenance_id),),
    )


def test_crop_rebases_bars_preserves_exact_overlap_and_raw_events(suite):
    original = _two_bar_piece(suite[0].left)
    validate_or_raise(original)
    cropped = crop_raw_context(original, RationalTime(4), RationalTime(8))
    piece = cropped.piece
    validate_or_raise(piece)
    assert piece.duration_qn == RationalTime(4)
    assert piece.bars[0].bar_id == "bar:001" and piece.bars[0].index == 0
    assert tuple(b.start_qn for b in piece.beats) == tuple(RationalTime(i) for i in range(4))
    crossing = next(n for n in piece.notes if n.note_id == "note:crossing")
    assert crossing.onset_qn == RationalTime(0)
    assert crossing.duration_qn == RationalTime(5, 3)
    assert piece.tempo_events[0].microseconds_per_quarter == 600000
    assert piece.tempo_events[0].onset_qn == RationalTime(0)
    shifted_expected = tuple((row[0] - RationalTime(4), *row[1:]) for row in
                             sounding_note_signature(original, RationalTime(4), RationalTime(8)))
    assert sounding_note_signature(piece) == shifted_expected
    assert (piece.dataset_name, piece.source_group_id, piece.split) == (
        original.dataset_name, original.source_group_id, original.split,
    )
    assert cropped.source_piece_id == original.piece_id
    assert cropped.source_start_qn == RationalTime(4)
    assert "intervention" in cropped.distribution_shift_warning
    assert original.duration_qn == RationalTime(8)


def test_crop_drops_annotations_and_rejects_partial_bars_and_float_time(suite):
    piece = suite[0].left
    poisoned = replace(piece, annotations=(object(),), targets=(object(),))
    crop = crop_raw_context(poisoned, RationalTime(0), RationalTime(4))
    assert not crop.piece.annotations and not crop.piece.targets
    with pytest.raises(ValueError, match="complete raw bar boundaries"):
        crop_raw_context(piece, RationalTime(1), RationalTime(4))
    with pytest.raises(TypeError, match="RationalTime"):
        crop_raw_context(piece, 0.0, RationalTime(4))
    with pytest.raises(ValueError, match="nonempty"):
        crop_raw_context(piece, RationalTime(1), RationalTime(1))


def test_test_partition_is_rejected_before_reading_notes(suite):
    class ForbiddenNotes:
        def __iter__(self):
            raise AssertionError("TEST note access")

    piece = replace(suite[0].left, split="TEST", notes=ForbiddenNotes())
    for action in (
        lambda: raw_context_candidates(piece),
        lambda: sounding_note_signature(piece),
        lambda: crop_raw_context(piece, RationalTime(0), RationalTime(4)),
    ):
        with pytest.raises(ValueError, match="only TRAIN"):
            action()
