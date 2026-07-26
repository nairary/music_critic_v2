from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from music_critic.data import RationalTime, TargetArray
from music_critic.adapters import HookTheoryAdapterConfig, convert_hooktheory_record
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.tasks import (
    ALIGNMENT_CONFLICT_DIAGNOSTIC,
    TARGET_ENCODINGS,
    TARGET_ENCODING_REGISTRY_VERSION,
    AlignmentOperationCounts,
    MultiSourceContractError,
    align_sample_targets,
    benchmark_multisource_collator,
    benchmark_target_alignment,
    build_alignment_index,
    build_multisource_sample,
    collate_multisource_samples,
    prepare_multisource_sample,
    project_multisource_targets,
    target_encoding_contract_fingerprint,
)
from tests.tasks.test_multisource_contract import _hook_piece, _pop_piece
from scripts.audit_multisource_targets import _hook_record


def _sample(piece):
    return prepare_multisource_sample(piece)


def _target(batch, task_id: str):
    return next(
        target for target in batch.target_batches if target.task_id == task_id
    )


def _replace_target(piece, replacement: TargetArray):
    return replace(
        piece,
        targets=tuple(
            replacement if target.task == replacement.task else target
            for target in piece.targets
        ),
    )


def _add_overlapping_root(
    piece,
    *,
    value: str,
    start: RationalTime | None = None,
    end: RationalTime | None = None,
):
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    annotation = next(
        item for item in piece.annotations if item.annotation_id == root.entity_ids[0]
    )
    duplicate_id = "span:phase5b1-extra-root"
    duplicate = replace(
        annotation,
        annotation_id=duplicate_id,
        start_qn=start or annotation.start_qn,
        end_qn=end or annotation.end_qn,
    )
    expanded = replace(
        root,
        entity_ids=(*root.entity_ids, duplicate_id),
        values=(*root.values, value),
        mask=(*root.mask, True),
        confidence=(*root.confidence, root.confidence[0]),
        source=(*root.source, root.source[0]),
        provenance=(*root.provenance, root.provenance[0]),
    )
    piece = _replace_target(piece, expanded)
    return replace(
        piece,
        annotations=tuple(
            sorted(
                (*piece.annotations, duplicate),
                key=lambda item: (
                    item.start_qn,
                    item.end_qn,
                    item.annotation_id,
                ),
            )
        ),
    )


def _many_equal_root_annotations(piece, count: int):
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    source_annotation = next(
        annotation
        for annotation in piece.annotations
        if annotation.annotation_id == root.entity_ids[0]
    )
    additions = tuple(
        replace(
            source_annotation,
            annotation_id=f"span:phase5b1-scaling-{index:04d}",
        )
        for index in range(count)
    )
    expanded = replace(
        root,
        entity_ids=(
            *root.entity_ids,
            *(annotation.annotation_id for annotation in additions),
        ),
        values=(*root.values, *((root.values[0],) * count)),
        mask=(*root.mask, *((True,) * count)),
        confidence=(*root.confidence, *((root.confidence[0],) * count)),
        source=(*root.source, *((root.source[0],) * count)),
        provenance=(*root.provenance, *((root.provenance[0],) * count)),
    )
    piece = _replace_target(piece, expanded)
    return replace(
        piece,
        annotations=tuple(
            sorted(
                (*piece.annotations, *additions),
                key=lambda item: (
                    item.start_qn,
                    item.end_qn,
                    item.annotation_id,
                ),
            )
        ),
    )


def test_encoding_registry_is_versioned_complete_and_explicit() -> None:
    assert TARGET_ENCODING_REGISTRY_VERSION == "1.0.0"
    assert len(TARGET_ENCODINGS) == 18
    assert tuple(spec.task_id for spec in TARGET_ENCODINGS) == tuple(
        sorted(spec.task_id for spec in TARGET_ENCODINGS)
    )
    open_specs = tuple(spec for spec in TARGET_ENCODINGS if not spec.model_ready)
    assert {spec.task_id for spec in open_specs} == {
        "theory.chord.borrowed",
        "theory.local_key.mode",
    }
    assert all(
        spec.encoding_kind == "open_string_cpu"
        and spec.dtype == "cpu.str"
        and spec.vocabulary is None
        and spec.deferred_reason
        for spec in open_specs
    )
    by_task = {spec.task_id: spec for spec in TARGET_ENCODINGS}
    positive_unlabeled_tasks = {
        "pop909_cl.chord.boundary",
        "pop909_cl.chord.no_chord",
    }
    assert all(
        by_task[task_id].model_ready
        and by_task[task_id].supervision_regime == "positive_unlabeled"
        for task_id in positive_unlabeled_tasks
    )
    assert by_task["pop909_cl.chord.no_chord"].vocabulary == ("N",)
    assert {
        spec.supervision_regime for spec in open_specs
    } == {"deferred_open_vocabulary"}
    assert all(
        spec.supervision_regime == "fully_supervised"
        for spec in TARGET_ENCODINGS
        if spec.model_ready and spec.task_id not in positive_unlabeled_tasks
    )
    assert not hasattr(
        by_task["pop909_cl.chord.no_chord"],
        "standard_bce_eligible",
    )
    assert target_encoding_contract_fingerprint() == (
        "386aceef18b6ba7da5e91d406cefdcdc21d46b6839ded873312402940b507e01"
    )


def test_mixed_hook_pop_raw_only_batch_has_all_tasks_and_exact_statistics(
    tmp_path: Path,
) -> None:
    hook = _sample(_hook_piece())
    pop = _sample(_pop_piece(tmp_path))
    raw_piece = replace(_hook_piece(), annotations=(), targets=())
    raw = _sample(raw_piece)
    batch = collate_multisource_samples((hook, pop, raw))

    assert batch.raw_graph_batch.num_graphs == 3
    assert len(batch.target_batches) == 18
    assert batch.dataset_ids == ("hooktheory", "pop909_cl", "hooktheory")
    assert batch.statistics.sample_count == batch.statistics.graph_count == 3
    assert batch.statistics.source_target_entry_count == 17
    assert batch.statistics.target_row_count == 76
    assert batch.statistics.aligned_available_count == 76
    assert batch.statistics.available_unaligned_row_count == 0
    assert batch.statistics.masked_row_count == 0
    assert batch.statistics.conflict_row_count == 0
    assert batch.statistics.model_ready_task_count == 16
    assert batch.statistics.deferred_open_vocabulary_task_count == 2
    assert batch.statistics.model_encodable_row_count == 66
    assert batch.statistics.supervision_eligible_row_count == 66
    assert batch.statistics.deferred_open_vocabulary_row_count == 10
    assert _target(batch, "pop909_cl.chord.no_chord").entry_count == 0


def test_bounded_real_source_hooktheory_fixture_collates() -> None:
    fixture = Path(
        "tests/fixtures/hooktheory/cases/ordinary_major_fractional.json"
    )
    case = json.loads(fixture.read_text(encoding="utf-8"))
    clip_id, record, structure = _hook_record(case)
    piece = convert_hooktheory_record(
        clip_id,
        record,
        config=HookTheoryAdapterConfig(dataset_name="hooktheory"),
        structure_row=structure,
        source_path="4_merged.json",
    )
    batch = collate_multisource_samples((_sample(piece),))
    assert batch.dataset_ids == ("hooktheory",)
    assert batch.statistics.source_target_entry_count > 0
    assert batch.statistics.aligned_available_count > 0


def test_note_identity_and_exact_onset_beat_bar_span_expansion() -> None:
    sample = _sample(_hook_piece())
    batch = collate_multisource_samples((sample,))

    note = _target(batch, "theory.melody.scale_degree")
    assert note.entity_node_types == ("note",)
    assert note.entity_indices.tolist() == [0]
    root = _target(batch, "theory.chord.root_degree")
    assert root.entity_node_types == ("onset", "beat", "beat", "bar")
    assert root.entity_indices.tolist() == [0, 0, 1, 0]
    assert root.supervision_eligibility_mask.tolist() == [True] * 4


def test_alignment_index_is_built_once_and_lookups_scale_with_entries() -> None:
    small = _sample(_many_equal_root_annotations(_hook_piece(), 8))
    large = _sample(_many_equal_root_annotations(_hook_piece(), 128))
    observations = []
    for sample in (small, large):
        operations = AlignmentOperationCounts()
        aligned = align_sample_targets(
            sample.canonical_piece,
            sample.raw_graph,
            sample,
            instrumentation=operations,
        )
        observations.append(operations)
        assert len(aligned) == 18
        assert operations.index_build_count == 1
        assert operations.note_index_entry_count == len(
            sample.canonical_piece.notes
        )
        assert operations.annotation_index_entry_count == len(
            sample.canonical_piece.annotations
        )
        assert operations.source_entry_lookup_count == sum(
            len(target.entity_ids) for target in sample.target_bundle
        )
        assert operations.annotation_lookup_count == sum(
            sum(target.availability_mask)
            for target in sample.target_bundle
            if target.alignment_type == "annotation_span"
        )
    assert (
        observations[0].candidate_index_entry_count
        == observations[1].candidate_index_entry_count
    )
    assert observations[1].annotation_lookup_count > (
        observations[0].annotation_lookup_count
    )
    evidence = benchmark_target_alignment((small,), repeats=1)
    operation_counts = dict(evidence.operation_counts)
    assert operation_counts["index_build_count"] == 1
    assert operation_counts["source_entry_lookup_count"] == (
        evidence.source_target_entry_count
    )
    assert operation_counts["merge_candidate_slot_visit_count"] > 0
    assert evidence.emitted_row_count > evidence.source_target_entry_count


def test_alignment_index_mappings_are_immutable() -> None:
    piece = _hook_piece()
    index = build_alignment_index(piece)
    with pytest.raises(TypeError):
        index.note_index_by_id["replacement"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        index.annotation_by_id["replacement"] = (  # type: ignore[index]
            piece.annotations[0]
        )
    with pytest.raises(TypeError):
        index.candidate_by_node_type["beat"].exact_indices_by_time[
            RationalTime(0, 1)
        ] = (0,)  # type: ignore[index]


def test_half_open_adjacent_spans_assign_boundary_to_following_span() -> None:
    piece = _hook_piece()
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    piece = _add_overlapping_root(
        piece,
        value=root.values[0],
        start=RationalTime(2, 1),
        end=RationalTime(4, 1),
    )
    sample = _sample(piece)
    aligned = next(
        family
        for family in align_sample_targets(
            piece,
            sample.raw_graph,
            sample,
        )
        if family.task_id == "theory.chord.root_degree"
    )
    beat_indices = [
        row.local_entity_index
        for row in aligned.rows
        if row.entity_node_type == "beat"
    ]
    assert beat_indices == [0, 1, 2, 3]


def test_boundary_exact_event_expands_all_types_without_synthetic_negatives(
    tmp_path: Path,
) -> None:
    batch = collate_multisource_samples((_sample(_pop_piece(tmp_path)),))
    boundary = _target(batch, "pop909_cl.chord.boundary")
    assert boundary.entity_node_types == ("onset", "beat", "bar")
    assert boundary.availability_mask.tolist() == [True, True, True]
    assert boundary.values.tolist() == [0, 0, 0]
    assert boundary.supervision_regime == "positive_unlabeled"
    assert boundary.supervision_eligibility_mask.tolist() == [True] * 3
    assert not hasattr(boundary, "standard_bce_eligibility_mask")
    assert boundary.entry_count == boundary.source_entry_count * 3


def test_no_chord_is_positive_unlabeled_without_synthetic_negatives(
    tmp_path: Path,
) -> None:
    batch = collate_multisource_samples(
        (_sample(_pop_piece(tmp_path, leading_gap_ticks=480)),)
    )
    no_chord = _target(batch, "pop909_cl.chord.no_chord")
    assert no_chord.supervision_regime == "positive_unlabeled"
    assert no_chord.source_entry_count == 1
    assert no_chord.entry_count == 3
    assert no_chord.entity_node_types == ("onset", "beat", "bar")
    assert no_chord.availability_mask.tolist() == [True, True, True]
    assert no_chord.values.tolist() == [0, 0, 0]
    assert no_chord.supervision_eligibility_mask.tolist() == [True, True, True]
    assert not hasattr(no_chord, "standard_bce_eligibility_mask")
    assert set(no_chord.values.tolist()) == {0}
    assert no_chord.entry_count < sum(
        int(batch.raw_graph_batch[node_type].num_nodes)
        for node_type in ("onset", "beat", "bar")
    )


def test_available_unaligned_boundary_is_retained_without_snap(
    tmp_path: Path,
) -> None:
    piece = _pop_piece(tmp_path)
    boundary = next(
        target
        for target in piece.targets
        if target.task == "pop909_cl.chord.boundary"
    )
    boundary_entity = boundary.entity_ids[0]
    piece = replace(
        piece,
        annotations=tuple(
            replace(annotation, start_qn=RationalTime(1, 2))
            if annotation.annotation_id == boundary_entity
            else annotation
            for annotation in piece.annotations
        ),
    )
    target = _target(
        collate_multisource_samples((_sample(piece),)),
        "pop909_cl.chord.boundary",
    )
    assert target.entry_count == 1
    assert target.availability_mask.tolist() == [True]
    assert target.entity_indices.tolist() == [-1]
    assert target.entity_index_mask.tolist() == [False]
    assert target.entity_node_types == (None,)
    assert target.supervision_eligibility_mask.tolist() == [False]
    assert (
        collate_multisource_samples((_sample(piece),))
        .statistics.available_unaligned_row_count
        == 1
    )


def test_equal_duplicate_spans_merge_and_preserve_both_sources() -> None:
    piece = _hook_piece()
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    batch = collate_multisource_samples(
        (_sample(_add_overlapping_root(piece, value=root.values[0])),)
    )
    target = _target(batch, root.task)
    assert target.source_entry_count == 2
    assert target.entry_count == 4
    assert all(
        len(provenance.source_entity_ids) == 2
        for provenance in target.provenance_cpu
    )
    assert target.diagnostics_cpu == ((), (), (), ())


def test_conflicting_duplicate_spans_are_masked_with_diagnostic() -> None:
    piece = _hook_piece()
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    conflict_value = next(
        label for label in (root.class_labels or ()) if label != root.values[0]
    )
    batch = collate_multisource_samples(
        (_sample(_add_overlapping_root(piece, value=conflict_value)),)
    )
    target = _target(batch, root.task)
    assert target.source_entry_count == 2
    assert target.entry_count == 4
    assert target.availability_mask.tolist() == [False] * 4
    assert target.values.tolist() == [-1] * 4
    assert batch.statistics.conflict_row_count == 4
    assert batch.statistics.masked_row_count == 0
    task_statistics = next(
        statistics
        for statistics in batch.statistics.task_counts
        if statistics.task_id == root.task
    )
    assert task_statistics.conflict_row_count == 4
    assert task_statistics.supervision_eligible_row_count == 0
    assert {
        diagnostic.code
        for diagnostics in target.diagnostics_cpu
        for diagnostic in diagnostics
    } == {ALIGNMENT_CONFLICT_DIAGNOSTIC}


def test_masked_source_entry_stays_one_unaligned_row() -> None:
    target = _target(
        collate_multisource_samples((_sample(_hook_piece(root=0)),)),
        "theory.chord.root_degree",
    )
    assert target.source_entry_count == target.entry_count == 1
    assert target.availability_mask.tolist() == [False]
    assert target.entity_indices.tolist() == [-1]
    assert target.entity_index_mask.tolist() == [False]
    assert target.values.tolist() == [-1]
    assert not target.supervision_eligibility_mask.any()


def test_closed_multilabel_and_open_strings_use_declared_encodings() -> None:
    batch = collate_multisource_samples((_sample(_hook_piece()),))
    multi = _target(batch, "theory.chord.alterations")
    assert multi.values.dtype == torch.bool
    assert tuple(multi.values.shape) == (4, 6)
    assert not multi.values.any()

    mode = _target(batch, "theory.local_key.mode")
    assert mode.values == ("major",) * 6
    assert not isinstance(mode.values, torch.Tensor)
    assert not mode.model_ready
    assert mode.deferred_reason
    assert not mode.supervision_eligibility_mask.any()


def test_masked_multilabel_all_false_is_not_supervision_eligible() -> None:
    piece = _hook_piece()
    alterations = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.alterations"
    )
    piece = _replace_target(
        piece,
        replace(
            alterations,
            values=(None,),
            mask=(False,),
            confidence=(None,),
            source=(None,),
            provenance=(None,),
        ),
    )
    target = _target(
        collate_multisource_samples((_sample(piece),)),
        alterations.task,
    )
    assert target.entry_count == 1
    assert target.values.shape == (1, len(alterations.class_labels or ()))
    assert not target.values.any()
    assert target.availability_mask.tolist() == [False]
    assert not target.supervision_eligibility_mask.any()


def test_second_sample_indices_use_ptr_offsets_and_batch_membership(
    tmp_path: Path,
) -> None:
    batch = collate_multisource_samples(
        (_sample(_hook_piece()), _sample(_pop_piece(tmp_path)))
    )
    root = _target(batch, "pop909_cl.chord.root")
    for row, node_type in enumerate(root.entity_node_types):
        assert node_type is not None
        index = int(root.entity_indices[row].item())
        assert index >= int(batch.raw_graph_batch[node_type].ptr[1].item())
        assert int(batch.raw_graph_batch[node_type].batch[index].item()) == 1


def test_collator_is_deterministic_and_does_not_mutate_raw_graphs(
    tmp_path: Path,
) -> None:
    samples = (_sample(_hook_piece()), _sample(_pop_piece(tmp_path)))
    before = tuple(graph_fingerprint(sample.raw_graph) for sample in samples)
    left = collate_multisource_samples(samples)
    right = collate_multisource_samples(samples)
    after = tuple(graph_fingerprint(sample.raw_graph) for sample in samples)
    assert before == after
    assert left.statistics == right.statistics
    for left_target, right_target in zip(
        left.target_batches, right.target_batches
    ):
        assert left_target.task_id == right_target.task_id
        if isinstance(left_target.values, torch.Tensor):
            assert torch.equal(left_target.values, right_target.values)
        else:
            assert left_target.values == right_target.values
        assert torch.equal(
            left_target.entity_indices, right_target.entity_indices
        )
        assert left_target.provenance_cpu == right_target.provenance_cpu


def test_target_projection_has_no_raw_graph_and_factory_proof_is_required() -> None:
    piece = _hook_piece()
    projection = project_multisource_targets(piece)
    assert not hasattr(projection, "raw_graph")
    sample = prepare_multisource_sample(piece)
    with pytest.raises(
        MultiSourceContractError,
        match="verified preparation factory",
    ):
        replace(sample, _binding_token=None)


def test_target_provenance_and_diagnostic_changes_do_not_change_raw_graph() -> None:
    piece = _hook_piece()
    presence = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.presence"
    )
    changed_presence = replace(
        presence,
        values=(
            "false" if presence.values[0] == "true" else "true",
        ),
    )
    changed_provenance = tuple(
        replace(
            record,
            details=tuple(
                sorted(
                    (*record.details, ("phase5b1_test", "changed")),
                    key=lambda item: item[0],
                )
            ),
        )
        if record.provenance_id == presence.provenance[0]
        else record
        for record in piece.provenance
    )
    changed_flags = tuple(
        replace(flag, message=f"{flag.message} changed")
        for flag in piece.quality_flags
    )
    changed = replace(
        _replace_target(piece, changed_presence),
        provenance=changed_provenance,
        quality_flags=changed_flags,
    )
    original_graph = build_raw_graph(piece)
    changed_graph = build_raw_graph(changed)
    assert graph_fingerprint(original_graph) == graph_fingerprint(changed_graph)
    original_batch = collate_multisource_samples(
        (build_multisource_sample(piece, original_graph),)
    )
    changed_batch = collate_multisource_samples(
        (build_multisource_sample(changed, changed_graph),)
    )
    assert original_batch.raw_graph_batch.node_types == (
        changed_batch.raw_graph_batch.node_types
    )
    assert original_batch.raw_graph_batch.edge_types == (
        changed_batch.raw_graph_batch.edge_types
    )
    assert all(
        "target" not in key
        and "provenance" not in key
        and "diagnostic" not in key
        for store in (
            original_batch.raw_graph_batch._global_store,
            *(
                original_batch.raw_graph_batch[node_type]
                for node_type in original_batch.raw_graph_batch.node_types
            ),
            *(
                original_batch.raw_graph_batch[edge_type]
                for edge_type in original_batch.raw_graph_batch.edge_types
            ),
        )
        for key in store.keys()
    )


@pytest.mark.parametrize(
    "mutation",
    ("categorical", "continuous", "topology"),
)
def test_prepared_sample_rejects_raw_graph_mutation(mutation: str) -> None:
    sample = _sample(_hook_piece())
    graph = sample.raw_graph
    if mutation == "categorical":
        graph["note"].x_cat[0, 0] = 61
    elif mutation == "continuous":
        graph["note"].x_cont[0, 0] = 0.125
    else:
        forward = ("beat", "next_beat", "beat")
        reverse = ("beat", "previous_beat", "beat")
        permutation = torch.arange(
            graph[forward].edge_index.shape[1] - 1,
            -1,
            -1,
        )
        graph[forward].edge_index = graph[forward].edge_index[:, permutation]
        graph[reverse].edge_index = graph[forward].edge_index.flip(0)
    with pytest.raises(
        MultiSourceContractError,
        match="multisource.raw_graph_binding_mismatch",
    ):
        collate_multisource_samples((sample,))


def test_external_graph_factory_has_no_binding_bypass() -> None:
    piece = _hook_piece()
    graph = build_raw_graph(piece)
    graph["note"].x_cat[0, 0] = 61
    with pytest.raises(
        MultiSourceContractError,
        match="multisource.raw_graph_binding_mismatch",
    ):
        build_multisource_sample(piece, graph)


def test_malformed_inputs_tensors_indices_and_graph_leakage_are_rejected() -> None:
    with pytest.raises(MultiSourceContractError, match="cannot be empty"):
        collate_multisource_samples(())
    with pytest.raises(MultiSourceContractError, match="not MultiSourceSample"):
        collate_multisource_samples((object(),))

    sample = _sample(_hook_piece())
    sample.raw_graph.targets = object()
    with pytest.raises(MultiSourceContractError, match="raw-only"):
        collate_multisource_samples((sample,))

    clean = _sample(_hook_piece())
    batch = collate_multisource_samples((clean,))
    target = _target(batch, "theory.chord.presence")
    with pytest.raises(MultiSourceContractError, match="rank-one long"):
        replace(target, values=target.values.to(torch.float32))
    with pytest.raises(MultiSourceContractError, match="registry order"):
        replace(batch, target_batches=tuple(reversed(batch.target_batches)))
    with pytest.raises(
        MultiSourceContractError,
        match="supervision_eligible_row_count",
    ):
        replace(
            batch.statistics,
            supervision_eligible_row_count=(
                batch.statistics.supervision_eligible_row_count + 1
            ),
        )
    with pytest.raises(MultiSourceContractError, match="global entity index"):
        replace(
            batch,
            target_batches=tuple(
                replace(
                    item,
                    entity_indices=item.entity_indices + 10_000,
                )
                if item.task_id == target.task_id
                else item
                for item in batch.target_batches
            ),
        )

    malformed_graph_batch = deepcopy(batch.raw_graph_batch)
    malformed_graph_batch["beat"].ptr[1] += 1
    with pytest.raises(MultiSourceContractError, match="raw-only contract"):
        replace(batch, raw_graph_batch=malformed_graph_batch)

    masked = _target(
        collate_multisource_samples((_sample(_hook_piece(root=0)),)),
        "theory.chord.root_degree",
    )
    with pytest.raises(MultiSourceContractError, match="sentinel -1"):
        replace(masked, values=torch.zeros(1, dtype=torch.long))


def test_lightweight_benchmark_covers_dozen_scale_batch() -> None:
    piece = replace(_hook_piece(), annotations=(), targets=())
    sample = _sample(piece)
    evidence = benchmark_multisource_collator((sample,) * 24, repeats=1)
    assert evidence.sample_count == 24
    assert evidence.repeat_count == 1
    assert evidence.node_count > 0
    assert evidence.edge_count > 0
    assert evidence.target_row_count == 0
    assert evidence.full_collation_seconds_per_repeat >= 0
