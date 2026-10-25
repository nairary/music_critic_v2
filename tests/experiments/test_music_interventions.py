from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from music_critic.data import AnnotationSpan, RationalTime, dumps_piece, load_piece, validate_or_raise
from music_critic.experiments.music_interventions import (
    Intervention, InterventionError, NotePatch, apply_interventions, check_compatibility,
)


FIXTURE = Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"
N0, N1 = "note:melody-000", "note:melody-001"


@pytest.fixture
def piece():
    return load_piece(FIXTURE)


def edit(identifier, note_id=N0, *, context=(), **fields):
    return Intervention(identifier, (NotePatch(note_id, **fields),), context)


def note(piece, identifier=N0):
    return next(item for item in piece.notes if item.note_id == identifier)


def test_pitch_timing_compose_in_both_orders_without_source_mutation(piece):
    before = dumps_piece(piece)
    pitch = edit("pitch", pitch=68)
    timing = edit("timing", onset_qn=RationalTime(1, 8))
    left = apply_interventions(piece, (pitch, timing))
    right = apply_interventions(piece, (timing, pitch))
    assert left == right
    assert left.compatibility.status == "compatible"
    assert note(left.piece).pitch == 68
    assert note(left.piece).onset_qn == RationalTime(1, 8)
    assert left.piece.piece_id != piece.piece_id
    assert left.piece.source_group_id == piece.source_group_id
    assert left.piece.split == piece.split
    assert dumps_piece(piece) == before
    assert left.piece.provenance[-1].source == "MusicInterventions@1.0.0"
    assert "quality_claim" in dict(left.piece.provenance[-1].details)
    validate_or_raise(left.piece)


def test_context_edits_require_review_even_for_disjoint_note_ids(piece):
    a = edit("harmony", pitch=68, context=(N1,))
    b = edit("melody", N1, pitch=73)
    assert check_compatibility(a, b) == check_compatibility(b, a)
    result = apply_interventions(piece, (a, b))
    assert result.compatibility.status == "semantic_revalidation"
    assert result.compatibility.reasons == (f"edited context note: {N1}",)


def test_shared_read_only_context_does_not_alone_conflict(piece):
    context = ("note:drums-000",)
    a = edit("a", pitch=68, context=context)
    b = edit("b", N1, pitch=73, context=context)
    assert apply_interventions(piece, (a, b)).compatibility.status == "compatible"


@pytest.mark.parametrize("second", [NotePatch(N0, pitch=69), NotePatch(N0, delete=True)])
def test_same_field_or_double_deletion_conflict(piece, second):
    first = edit("a", delete=True) if second.delete else edit("a", pitch=68)
    other = Intervention("b", (second,))
    assert check_compatibility(first, other).status == "conflict"
    with pytest.raises(InterventionError):
        apply_interventions(piece, (first, other))


def test_delete_and_edit_conflict_in_either_order(piece):
    delete, pitch = edit("delete", delete=True), edit("pitch", pitch=68)
    assert check_compatibility(delete, pitch) == check_compatibility(pitch, delete)
    for selected in ((delete, pitch), (pitch, delete)):
        with pytest.raises(InterventionError, match="delete/edit"):
            apply_interventions(piece, selected)


def test_delete_preserves_identity_of_remaining_notes_and_records_tombstone(piece):
    result = apply_interventions(piece, (edit("delete", delete=True),)).piece
    assert [n.note_id for n in result.notes] == [n.note_id for n in piece.notes if n.note_id != N0]
    plan = dict(result.provenance[-1].details)["plan_json"]
    assert '"delete":true' in plan and N0 in plan


@pytest.mark.parametrize("fields", [
    {"duration_qn": RationalTime(-1)},
    {"duration_qn": RationalTime(0)},
    {"onset_qn": RationalTime(-1, 8)},
    {"onset_qn": RationalTime(5)},
    {"duration_qn": RationalTime(6)},
])
def test_invalid_final_timing_is_rejected(piece, fields):
    with pytest.raises(InterventionError, match="invalid intervention result"):
        apply_interventions(piece, (edit("bad", **fields),))


def test_combined_final_bounds_checked_not_only_single_patches(piece):
    # Each patch alone is valid; together the note ends beyond the source span.
    onset = edit("onset", onset_qn=RationalTime(3))
    duration = edit("duration", duration_qn=RationalTime(3))
    apply_interventions(piece, (onset,))
    apply_interventions(piece, (duration,))
    with pytest.raises(InterventionError, match="invalid intervention result"):
        apply_interventions(piece, (onset, duration))


def test_stale_annotations_targets_and_pitch_spelling_are_removed(piece):
    target_annotation = AnnotationSpan(
        "span:target", "theory.harmony", "target_alignment", RationalTime(0),
        RationalTime(2), None, None, piece.provenance[0].provenance_id,
    )
    source = replace(piece, annotations=tuple(sorted(
        (*piece.annotations, target_annotation),
        key=lambda item: (item.start_qn, item.end_qn, item.annotation_id),
    )))
    result = apply_interventions(source, (edit("pitch", pitch=68),)).piece
    assert source.targets and not result.targets
    assert result.annotations == piece.annotations
    assert note(result).spelling_step is None
    assert note(result).spelling_alter is None
    assert note(result).source_onset_ticks == note(source).source_onset_ticks
    assert note(result, N1) == note(source, N1)


def test_time_metadata_cleared_but_raw_metric_context_kept(piece):
    result = apply_interventions(piece, (edit("time", onset_qn=RationalTime(1, 8)),)).piece
    for field in ("source_onset_ticks", "source_duration_ticks", "source_onset_seconds", "source_duration_seconds"):
        assert getattr(note(result), field) is None
    assert note(result).spelling_step == note(piece).spelling_step
    assert result.bars == piece.bars and result.beats == piece.beats
    assert result.tempo_events == piece.tempo_events


@pytest.mark.parametrize("fields", [
    {"pitch": -1}, {"pitch": 128}, {"pitch": True}, {"onset_qn": 0.25},
    {"duration_qn": 1.0}, {"delete": 1}, {"delete": True, "pitch": 68}, {},
])
def test_malformed_patch_fails_early(fields):
    with pytest.raises(InterventionError):
        NotePatch(N0, **fields)


def test_duplicate_ids_and_empty_plan_rejected(piece):
    a = edit("same", pitch=68)
    with pytest.raises(InterventionError, match="duplicate intervention"):
        apply_interventions(piece, (a, edit("same", N1, pitch=73)))
    with pytest.raises(InterventionError, match="duplicate note"):
        Intervention("bad", (NotePatch(N0, pitch=68), NotePatch(N0, duration_qn=RationalTime(1))))
    with pytest.raises(InterventionError, match="nonempty tuple"):
        apply_interventions(piece, ())


def test_unknown_note_or_context_and_noop_rejected(piece):
    for intervention in (
        edit("unknown", "note:missing", pitch=68),
        edit("context", pitch=68, context=("note:missing",)),
        edit("unchanged", pitch=note(piece).pitch),
    ):
        with pytest.raises(InterventionError):
            apply_interventions(piece, (intervention,))


def test_source_must_be_valid_and_frozen_objects_cannot_mutate(piece):
    patch = NotePatch(N0, pitch=68)
    with pytest.raises(FrozenInstanceError):
        patch.pitch = 70
    with pytest.raises(InterventionError, match="invalid source"):
        apply_interventions(replace(piece, duration_qn=RationalTime(0)), (Intervention("a", (patch,)),))


def test_three_way_context_dependencies_are_all_reported(piece):
    selected = (
        edit("a", pitch=68, context=(N1,)),
        edit("b", N1, pitch=73, context=("note:melody-003",)),
        edit("c", "note:melody-003", pitch=77),
    )
    result = apply_interventions(piece, selected)
    assert result.compatibility.status == "semantic_revalidation"
    assert len(result.compatibility.reasons) == 2
