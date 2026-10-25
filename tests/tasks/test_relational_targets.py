"""Synthetic canonical-event supervision; no musical accuracy claim."""

from dataclasses import replace
import json

import pytest

from music_critic.data import RationalTime
from music_critic.graph import graph_fingerprint
from music_critic.tasks import (
    collate_multisource_samples,
    dumps_target_bundle,
    loads_target_bundle,
    prepare_multisource_sample,
)
from music_critic.tasks.relational_targets import (
    RelationalTargetAssignment,
    RelationalTargetError,
    build_relational_target_bundle,
    dumps_relational_assignments,
    loads_relational_assignments,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _piece():
    base = _hook_piece()
    note = base.notes[0]
    return replace(
        base,
        notes=(
            replace(note, note_id="note:held", duration_qn=RationalTime(2)),
            replace(note, note_id="note:arrival", onset_qn=RationalTime(2)),
        ),
        annotations=(),
        targets=(),
    )


def _assignment(piece, **overrides):
    fields = dict(
        assignment_id="assignment:1",
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        note_id="note:held",
        flag="held_common_tone",
        start_qn=RationalTime(1, 2),
        end_qn=RationalTime(3, 2),
        relation_id="relation:1",
        source_event_ids=("source:fragment:1", "source:fragment:2"),
        related_note_ids=("note:arrival",),
        source_version_id="synthetic-source-v1",
        annotation_origin="astra_inference",
        annotation_status="astra_inference",
        admitted_weak=True,
        model_requested="gpt-6-astra",
        reasoning_effort="ultra",
    )
    return RelationalTargetAssignment(**(fields | overrides))


def test_positive_note_flag_preserves_unknown_rows_and_other_flags():
    piece = _piece()
    bundle = build_relational_target_bundle(
        piece, (_assignment(piece),), analysis_view_id="relations:synthetic"
    )
    assert len(bundle.targets) == 9
    target = next(
        item for item in bundle.targets
        if item.task_id == "theory.relations.held_common_tone"
    )
    assert target.entity_ids == ("note:held", "note:arrival")
    assert target.values == ("present", None)
    assert target.availability_mask == (True, False)
    assert target.source == ("pseudo_label", None)
    assert target.confidence == (None, None)
    assert target.provenance_ids[0] is not None
    assert target.provenance_ids[1] is None
    for other in bundle.targets:
        if other is not target:
            assert other.values == (None, None)
            assert other.availability_mask == (False, False)
            assert other.provenance_ids == (None, None)


def test_bundle_roundtrip_and_existing_collator_encode_positive_without_graph_change():
    piece = _piece()
    original = prepare_multisource_sample(piece)
    bundle = build_relational_target_bundle(
        piece, (_assignment(piece),), analysis_view_id="relations:synthetic"
    )
    restored = loads_target_bundle(dumps_target_bundle(bundle))
    assert restored == bundle
    sample = prepare_multisource_sample(piece, target_sidecar=restored)
    assert graph_fingerprint(sample.raw_graph) == graph_fingerprint(original.raw_graph)
    batch = collate_multisource_samples((sample,))
    assert len(batch.target_batches) == 27
    target = next(
        item for item in batch.target_batches
        if item.task_id == "theory.relations.held_common_tone"
    )
    assert target.values.tolist() == [1, -1]
    assert target.availability_mask.tolist() == [True, False]
    assert target.supervision_eligibility_mask.tolist() == [True, False]
    # Existing alignment leaves unavailable entries unaligned, with sentinel -1.
    assert target.entity_indices.tolist() == [0, -1]
    assert target.entity_index_mask.tolist() == [True, False]
    assert target.supervision_regime == "positive_unlabeled"
    assert target.confidence is None


def test_duplicate_assignment_ids_are_rejected_even_when_equal():
    piece = _piece()
    assignment = _assignment(piece)
    with pytest.raises(RelationalTargetError, match="duplicate assignment_id"):
        build_relational_target_bundle(
            piece, (assignment, assignment), analysis_view_id="relations:synthetic"
        )


def test_multiple_roles_and_subspans_survive_and_project_once_per_note_flag():
    piece = _piece()
    first = _assignment(piece)
    second = _assignment(
        piece, assignment_id="assignment:2", start_qn=RationalTime(3, 2),
        end_qn=RationalTime(2), relation_id="relation:2",
    )
    other_role = _assignment(
        piece, assignment_id="assignment:3", flag="anticipation",
        start_qn=RationalTime(7, 4), end_qn=RationalTime(2),
    )
    assignments = (first, second, other_role)
    payload = dumps_relational_assignments(reversed(assignments))
    assert loads_relational_assignments(payload) == assignments
    bundle = build_relational_target_bundle(
        piece, assignments, analysis_view_id="relations:synthetic"
    )
    assert sum(sum(target.availability_mask) for target in bundle.targets) == 2
    held = next(p for p in bundle.provenance if dict(p.details)["flag"] == "held_common_tone")
    evidence = loads_relational_assignments(dict(held.details)["assignment_evidence_json"])
    assert evidence == (first, second)
    assert evidence[0].source_event_ids == ("source:fragment:1", "source:fragment:2")
    assert evidence[1].start_qn == RationalTime(3, 2)
    assert evidence[0].model_requested == "gpt-6-astra"
    assert evidence[0].reasoning_effort == "ultra"


@pytest.mark.parametrize("status", [
    "rejected", "ambiguous", "missing", "missing_evidence", "candidate",
    "rejected_semantic", "unknown_status",
])
def test_unadmitted_assignments_remain_unknown_and_roundtrip(status):
    piece = _piece()
    assignment = _assignment(piece, admitted_weak=False, annotation_status=status)
    assert loads_relational_assignments(dumps_relational_assignments((assignment,))) == (assignment,)
    bundle = build_relational_target_bundle(
        piece, (assignment,), analysis_view_id="relations:synthetic"
    )
    assert not bundle.provenance
    assert all(not any(target.availability_mask) for target in bundle.targets)
    assert all(all(value is None for value in target.values) for target in bundle.targets)
    with pytest.raises(RelationalTargetError, match="cannot be admitted"):
        replace(assignment, admitted_weak=True)


@pytest.mark.parametrize("split", [None, "validation", "test", "external_research_only"])
def test_non_train_canonical_pieces_rejected(split):
    piece = replace(_piece(), split=split)
    with pytest.raises(RelationalTargetError, match="TRAIN"):
        build_relational_target_bundle(piece, (), analysis_view_id="relations:synthetic")


@pytest.mark.parametrize("changes,message", [
    ({"piece_id": "another-piece"}, "different canonical piece"),
    ({"dataset_id": "another-dataset"}, "different canonical piece"),
    ({"note_id": "invented-note"}, "absent canonical note"),
    ({"related_note_ids": ("invented-note",)}, "absent canonical note"),
    ({"start_qn": RationalTime(-1)}, "outside its canonical note"),
    ({"end_qn": RationalTime(5, 2)}, "outside its canonical note"),
])
def test_invalid_binding_or_outside_interval_rejected(changes, message):
    piece = _piece()
    with pytest.raises(RelationalTargetError, match=message):
        build_relational_target_bundle(
            piece, (_assignment(piece, **changes),), analysis_view_id="relations:synthetic"
        )


@pytest.mark.parametrize("changes,message", [
    ({"flag": "quality_bad"}, "unknown relational flag"),
    ({"start_qn": 0.5}, "RationalTime"),
    ({"end_qn": RationalTime(1, 2)}, "nonempty exact interval"),
    ({"source_event_ids": ()}, "requires source_event_ids"),
    ({"admitted_weak": 1}, "boolean"),
    ({"source_event_ids": ([],)}, "unique exact IDs"),
])
def test_malformed_assignment_rejected(changes, message):
    with pytest.raises(RelationalTargetError, match=message):
        _assignment(_piece(), **changes)


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(contract_version="2.0.0"),
    lambda data: data["assignments"][0].update(extra="invented"),
    lambda data: data["assignments"][0]["start_qn"].update(num=0.5),
    lambda data: data["assignments"][0].update(source_event_ids="source:fragment:1"),
    lambda data: data["assignments"].append(data["assignments"][0]),
])
def test_assignment_decoder_rejects_malformed_payload(mutation):
    data = json.loads(dumps_relational_assignments((_assignment(_piece()),)))
    mutation(data)
    with pytest.raises(RelationalTargetError):
        loads_relational_assignments(json.dumps(data))


def test_extension_is_opt_in_and_preserves_core_fingerprints():
    from music_critic.tasks.encoding import target_encoding_contract_fingerprint
    from music_critic.tasks.registry import target_families_for_registries
    from music_critic.tasks.relational_registry import RELATIONAL_TARGET_REGISTRY_ID

    assert len(target_families_for_registries()) == 18
    assert len(target_families_for_registries((RELATIONAL_TARGET_REGISTRY_ID,))) == 27
    assert target_encoding_contract_fingerprint() == (
        "386aceef18b6ba7da5e91d406cefdcdc21d46b6839ded873312402940b507e01"
    )
