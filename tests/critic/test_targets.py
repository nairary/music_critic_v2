"""Tiny synthetic intervention targets; never opens corpus records."""

from pathlib import Path
from dataclasses import replace
from fractions import Fraction

import pytest

from music_critic.data import RationalTime as Q, load_piece
from music_critic.experiments.music_interventions import (
    Intervention, NotePatch, apply_interventions,
)
from music_critic.critic.targets import InterventionTargetBuilder, VerifiedPatch, verify_actual_diff


FIXTURE = Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"


def test_verified_diff_uses_actual_changes_not_eligibility_support():
    original = load_piece(FIXTURE)
    patch = NotePatch("note:melody-000", pitch=68)
    variant = apply_interventions(original, (Intervention("pitch", (patch,)),)).piece
    proof = VerifiedPatch(
        original.piece_id, variant.piece_id, "synthetic:work", (patch,),
        ("harmony",), "pitch", exhaustive=True,
        support_ids=("note:melody-000", "note:melody-001"),
    )
    actual = verify_actual_diff(original, variant, proof)
    assert actual.changed_note_ids == ("note:melody-000",)
    assert actual.deleted_note_ids == ()
    assert actual.exhaustive


def test_exhaustive_verification_rejects_hidden_edits_and_wrong_values():
    original = load_piece(FIXTURE)
    declared = NotePatch("note:melody-000", pitch=68)
    hidden = NotePatch("note:melody-001", pitch=73)
    variant = apply_interventions(original, (Intervention("two", (declared, hidden)),)).piece
    proof = VerifiedPatch(original.piece_id, variant.piece_id, "work", (declared,), ("harmony",), "pitch")
    with pytest.raises(ValueError, match="unreported"):
        verify_actual_diff(original, variant, proof)
    partial = verify_actual_diff(original, variant, replace(proof, exhaustive=False))
    assert partial.changed_note_ids == ("note:melody-000",)
    assert not partial.exhaustive
    with pytest.raises(ValueError, match="match verified"):
        verify_actual_diff(original, variant, replace(proof, patches=(replace(declared, pitch=69),), exhaustive=False))


def test_patch_requires_boolean_completeness_and_known_aspects():
    with pytest.raises(ValueError, match="exhaustive"):
        VerifiedPatch("a", "b", "work", (), (), "unknown", exhaustive="false")
    with pytest.raises(ValueError, match="aspect"):
        VerifiedPatch("a", "b", "work", (), ("harmny",), "pitch")


def test_empty_diff_cannot_manufacture_an_original_preference():
    original = load_piece(FIXTURE)
    variant = replace(original, piece_id="piece:same-raw-input", targets=(), annotations=(), quality_flags=())
    proof = VerifiedPatch(original.piece_id, variant.piece_id, "work", (), ("harmony",), "pitch")
    with pytest.raises(ValueError, match="actual edit"):
        InterventionTargetBuilder().build(original, variant, proof)


def test_pitch_targets_mark_actual_notes_and_soundings_without_cross_aspect_negatives():
    original = load_piece(FIXTURE)
    patch = NotePatch("note:melody-000", pitch=68)
    variant = apply_interventions(original, (Intervention("pitch", (patch,)),)).piece
    proof = VerifiedPatch(original.piece_id, variant.piece_id, "work", (patch,),
                          ("harmony", "melody"), "pitch",
                          support_ids=("note:melody-000", "note:melody-001"))
    targets = InterventionTargetBuilder().build(original, variant, proof)
    assert targets.pair_mask == (True, False, True, False)
    assert targets.preference == (1, 0, 1, 0)
    original_local, edited_local = targets.local_targets
    assert not any(original_local["melody/note"].values)
    note_targets = edited_local["melody/note"]
    positive = {c.candidate_id for c, value in zip(note_targets.candidates, note_targets.values) if value}
    assert positive == {"note:melody-000"}
    assert not any(edited_local["rhythm/beat"].mask)
    assert any(edited_local["harmony/slice"].values)
    assert targets.swapped().swapped() == targets


def _targets(patch, *, aspects=("rhythm",), exhaustive=True):
    original = load_piece(FIXTURE)
    variant = apply_interventions(original, (Intervention("one", (patch,)),)).piece
    proof = VerifiedPatch(original.piece_id, variant.piece_id, "work", (patch,), aspects,
                          "synthetic", exhaustive=exhaustive)
    return InterventionTargetBuilder().build(original, variant, proof)


def _positive_ids(target):
    return {c.candidate_id for c, v, m in zip(target.candidates, target.values, target.mask) if m and v}


def test_time_support_is_old_new_union_and_uses_exact_half_open_boundaries():
    targets = _targets(NotePatch("note:melody-000", onset_qn=Q(2)))
    beats = targets.local_targets[1]["rhythm/beat"]
    assert _positive_ids(beats) == {"beat:000", "beat:001", "beat:002", "beat:003"}
    assert "beat:004" not in _positive_ids(beats)


def test_deletion_projects_to_retained_metric_grid_without_deleted_candidates():
    targets = _targets(NotePatch("note:melody-000", delete=True))
    edited = targets.local_targets[1]
    assert _positive_ids(edited["rhythm/beat"]) == {"beat:000", "beat:001"}
    assert all("note:melody-000" not in c.note_ids for c in edited["rhythm/event"].candidates)


def test_incomplete_diff_supplies_positive_evidence_but_no_negative_labels():
    targets = _targets(NotePatch("note:melody-000", delete=True), exhaustive=False)
    assert not any(any(t.mask) for t in targets.local_targets[0].values())
    beats = targets.local_targets[1]["rhythm/beat"]
    assert _positive_ids(beats) == {"beat:000", "beat:001"}
    assert all(value for value, mask in zip(beats.values, beats.mask) if mask)


def test_grace_pitch_targets_existing_events_only_not_fake_sounding_support():
    targets = _targets(NotePatch("note:melody-002", pitch=85), aspects=("harmony", "melody"))
    edited = targets.local_targets[1]
    assert not _positive_ids(edited["harmony/slice"])
    assert not _positive_ids(edited["melody/note"])
    assert _positive_ids(edited["melody/event"])
    grace = next(i for i, c in enumerate(edited["melody/note"].candidates) if c.candidate_id == "note:melody-002")
    assert not edited["melody/note"].mask[grace]


def test_deleted_grace_projects_only_to_retained_same_track_attack_event():
    targets = _targets(NotePatch("note:melody-002", delete=True), aspects=("harmony", "melody"))
    edited = targets.local_targets[1]
    assert not _positive_ids(edited["harmony/slice"])
    assert not _positive_ids(edited["melody/note"])
    positive = _positive_ids(edited["melody/event"])
    assert len(positive) == 1
    event = next(c for c in edited["melody/event"].candidates if c.candidate_id in positive)
    assert event.start == Fraction(2) and event.note_ids == ("note:melody-003",)


def test_arrangement_is_diagnostic_and_missing_aspects_are_masked():
    targets = _targets(NotePatch("note:melody-000", delete=True), aspects=("arrangement",))
    assert not any(targets.pair_mask)
    assert not any(any(t.mask) for view in targets.local_targets for t in view.values())


def _bar_swap():
    source = load_piece(FIXTURE)
    bars = tuple(replace(source.bars[1], bar_id=f"bar:{i}", index=i, start_qn=Q(i * 4))
                 for i in range(4))
    notes = tuple(replace(source.notes[0], note_id=f"note:{i}", onset_qn=Q(i * 4 + 1),
                          duration_qn=Q(1), pitch=60 + i) for i in range(4))
    beats = tuple(replace(source.beats[1], beat_id=f"beat:{i}", bar_id=f"bar:{i // 4}",
                          start_qn=Q(i), index_in_bar=i % 4, position_in_bar_qn=Q(i % 4),
                          is_downbeat=i % 4 == 0) for i in range(16))
    original = replace(source, bars=bars, beats=beats, notes=notes, duration_qn=Q(16), targets=(), annotations=())
    patches = (NotePatch("note:1", onset_qn=Q(9)), NotePatch("note:2", onset_qn=Q(5)))
    variant = apply_interventions(original, (Intervention("swap", patches),)).piece
    proof = VerifiedPatch(original.piece_id, variant.piece_id, "work", patches,
                          ("structure",), "bar_order_swap", swap_boundaries=(Fraction(4), Fraction(8), Fraction(12)))
    return original, variant, proof


def test_bar_swap_labels_existing_entrance_join_and_exit_boundaries():
    original, variant, proof = _bar_swap()
    targets = InterventionTargetBuilder().build(original, variant, proof)
    assert _positive_ids(targets.local_targets[1]["structure/boundary"]) == {
        "boundary:bar:1", "boundary:bar:2", "boundary:bar:3"}
    assert not any(targets.local_targets[0]["structure/boundary"].values)
    with pytest.raises(ValueError, match="bar swap"):
        InterventionTargetBuilder().build(original, variant, replace(proof, swap_boundaries=(Fraction(0), Fraction(4), Fraction(8))))


def test_bar_swap_requires_complete_permutation_even_when_general_diff_is_partial():
    original, variant, proof = _bar_swap()
    with pytest.raises(ValueError, match="bar swap"):
        InterventionTargetBuilder().build(original, variant, replace(proof, exhaustive=False, patches=proof.patches[:1]))
