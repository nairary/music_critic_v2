from __future__ import annotations

from dataclasses import replace
import inspect

import torch

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models import (
    ACTIVE_TASK_IDS,
    EXCLUDED_TASK_REASONS,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    SourceNativeTaskHeads,
    active_task_head_specs,
    aggregate_task_losses,
    join_task_supervision,
    routing_operation_counts,
)
from music_critic.tasks import (
    ENTITY_NODE_TYPE_TO_CODE,
    TARGET_ENCODING_BY_TASK,
    collate_multisource_samples,
    prepare_multisource_sample,
)
from tests.tasks.test_multisource_collator import (
    _add_overlapping_root,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _model() -> LocalHeterogeneousBaseline:
    return LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )


def _batch(piece):
    return collate_multisource_samples((prepare_multisource_sample(piece),))


def _masked_root_piece():
    piece = _hook_piece()
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    masked = replace(
        root,
        values=(None,),
        mask=(False,),
        confidence=(None,),
        source=(None,),
        provenance=(None,),
    )
    return replace(
        piece,
        targets=tuple(
            masked if target.task == root.task else target
            for target in piece.targets
        ),
    )


def _replaced_root_piece():
    piece = _hook_piece()
    root = next(
        target
        for target in piece.targets
        if target.task == "theory.chord.root_degree"
    )
    replacement = replace(root, values=("2",))
    return replace(
        piece,
        targets=tuple(
            replacement if target.task == root.task else target
            for target in piece.targets
        ),
    )


def _repeat_batch_target_rows(target, repeats: int):
    def repeated(value):
        if value is None:
            return None
        return value.repeat_interleave(repeats, dim=0)

    return replace(
        target,
        values=repeated(target.values),
        availability_mask=repeated(target.availability_mask),
        entity_indices=repeated(target.entity_indices),
        entity_index_mask=repeated(target.entity_index_mask),
        entity_node_type_codes=repeated(target.entity_node_type_codes),
        entity_node_types=tuple(
            node_type
            for node_type in target.entity_node_types
            for _ in range(repeats)
        ),
        sample_indices=repeated(target.sample_indices),
        confidence=repeated(target.confidence),
        confidence_mask=repeated(target.confidence_mask),
        entry_count=target.entry_count * repeats,
        source_entry_count=target.source_entry_count * repeats,
        provenance_cpu=tuple(
            row for row in target.provenance_cpu for _ in range(repeats)
        ),
        diagnostics_cpu=tuple(
            row for row in target.diagnostics_cpu for _ in range(repeats)
        ),
    )


def test_head_registry_contains_only_fully_supervised_source_native_tasks() -> None:
    specs = active_task_head_specs()
    assert tuple(spec.task_id for spec in specs) == ACTIVE_TASK_IDS
    assert len(specs) == 14
    assert set(EXCLUDED_TASK_REASONS) == {
        "theory.local_key.mode",
        "theory.chord.borrowed",
        "pop909_cl.chord.boundary",
        "pop909_cl.chord.no_chord",
    }
    assert all(
        TARGET_ENCODING_BY_TASK[spec.task_id].supervision_regime
        == "fully_supervised"
        for spec in specs
    )


def test_predictions_cover_all_raw_candidates_before_supervision(mixed_batch) -> None:
    output = _model()(mixed_batch)
    assert tuple(item.task_id for item in output.predictions) == ACTIVE_TASK_IDS
    assert tuple(item.task_id for item in output.supervisions) == ACTIVE_TASK_IDS
    assert tuple(item.task_id for item in output.harmonic_loss.task_losses) == (
        ACTIVE_TASK_IDS
    )
    for prediction, spec in zip(output.predictions, active_task_head_specs()):
        expected_count = sum(
            mixed_batch.raw_graph_batch[node_type].num_nodes
            for node_type in spec.node_types
        )
        assert prediction.logits.shape == (expected_count, spec.output_dim)
        assert prediction.allowed_node_types == spec.node_types
        expected_offsets = {
            node_type: sum(
                mixed_batch.raw_graph_batch[earlier].num_nodes
                for earlier in spec.node_types[:index]
            )
            for index, node_type in enumerate(spec.node_types)
        }
        for node_type in MANDATORY_NODE_TYPES:
            code = ENTITY_NODE_TYPE_TO_CODE[node_type]
            count = (
                mixed_batch.raw_graph_batch[node_type].num_nodes
                if node_type in spec.node_types
                else 0
            )
            expected_offset = expected_offsets.get(node_type, -1)
            assert int(prediction.candidate_offsets_by_node_type[code]) == (
                expected_offset
            )
            assert int(prediction.candidate_counts_by_node_type[code]) == count
        cursor = 0
        for node_type in spec.node_types:
            code = ENTITY_NODE_TYPE_TO_CODE[node_type]
            count = mixed_batch.raw_graph_batch[node_type].num_nodes
            segment = slice(cursor, cursor + count)
            assert torch.equal(
                prediction.candidate_node_type_codes[segment],
                torch.full((count,), code, dtype=torch.long),
            )
            assert torch.equal(
                prediction.global_entity_indices[segment],
                torch.arange(count),
            )
            assert torch.equal(
                prediction.sample_indices[segment],
                mixed_batch.raw_graph_batch[node_type].batch,
            )
            cursor += count


def test_supervision_join_retains_unreduced_local_losses(mixed_batch) -> None:
    output = _model()(mixed_batch)
    by_task = {item.task_id: item for item in output.predictions}
    for supervision in output.supervisions:
        prediction = by_task[supervision.task_id]
        assert supervision.per_row_loss.shape == supervision.candidate_indices.shape
        assert torch.equal(
            prediction.candidate_node_type_codes.index_select(
                0, supervision.candidate_indices
            ),
            supervision.node_type_codes,
        )
        assert torch.equal(
            prediction.global_entity_indices.index_select(
                0, supervision.candidate_indices
            ),
            supervision.global_entity_indices,
        )
        assert torch.equal(
            prediction.sample_indices.index_select(
                0, supervision.candidate_indices
            ),
            supervision.sample_indices,
        )
        assert torch.isfinite(supervision.per_row_loss).all()
    for task_loss in output.harmonic_loss.task_losses:
        assert task_loss.group_mean_losses.numel() > 0
        assert (task_loss.group_row_counts > 0).all()
        assert torch.isfinite(task_loss.mean_loss)


def test_raw_only_batch_emits_candidate_logits_without_harmonic_loss() -> None:
    piece = replace(_hook_piece(), annotations=(), targets=())
    batch = _batch(piece)
    model = _model().eval()
    output = model(batch)
    assert output.harmonic_loss.total_loss is None
    assert output.harmonic_loss.task_losses == ()
    assert output.supervisions == ()
    assert all(prediction.logits.shape[0] > 0 for prediction in output.predictions)
    encoded, predictions = model.predict(batch.raw_graph_batch)
    assert encoded.final_output.embeddings["note"].shape[0] == 1
    assert all(prediction.logits.shape[0] > 0 for prediction in predictions)


def test_dense_rows_reduce_by_task_node_type_and_sample(mixed_batch) -> None:
    report = _model()(mixed_batch).harmonic_loss
    root = next(
        item
        for item in report.task_losses
        if item.task_id == "theory.chord.root_degree"
    )
    groups = {
        (int(code), int(sample))
        for code, sample in zip(
            root.group_node_type_codes, root.group_sample_indices
        )
    }
    assert groups == {
        (ENTITY_NODE_TYPE_TO_CODE["bar"], 0),
        (ENTITY_NODE_TYPE_TO_CODE["beat"], 0),
        (ENTITY_NODE_TYPE_TO_CODE["onset"], 0),
    }
    assert torch.isfinite(report.total_loss)


def test_target_replace_delete_mask_and_add_leave_candidate_logits_unchanged() -> None:
    base = _hook_piece()
    variants = (
        _replaced_root_piece(),
        replace(base, annotations=(), targets=()),
        _masked_root_piece(),
        _add_overlapping_root(base, value="1"),
    )
    torch.manual_seed(41)
    model = _model().eval()
    reference = model(_batch(base), include_reconstruction=False).predictions
    for variant in variants:
        actual = model(_batch(variant), include_reconstruction=False).predictions
        for left, right in zip(reference, actual):
            assert left.task_id == right.task_id
            assert torch.equal(
                left.candidate_node_type_codes,
                right.candidate_node_type_codes,
            )
            assert torch.equal(
                left.global_entity_indices, right.global_entity_indices
            )
            assert torch.equal(left.sample_indices, right.sample_indices)
            assert torch.equal(left.logits, right.logits)


def test_forward_and_loss_routing_has_no_per_row_python_materialization() -> None:
    sources = "\n".join(
        inspect.getsource(function)
        for function in (
            SourceNativeTaskHeads.forward,
            join_task_supervision,
            aggregate_task_losses,
        )
    )
    for forbidden in (".cpu(", ".tolist(", ".item("):
        assert forbidden not in sources

    batch = _batch(_hook_piece())
    model = _model()
    predictions = model.task_heads(
        model.encode(batch.raw_graph_batch).final_output
    )
    small_supervisions = join_task_supervision(
        predictions, batch.target_batches
    )
    large_targets = tuple(
        _repeat_batch_target_rows(target, 128)
        if target.task_id == "theory.chord.root_degree"
        else target
        for target in batch.target_batches
    )
    large_supervisions = join_task_supervision(predictions, large_targets)
    small_root = next(
        item
        for item in small_supervisions
        if item.task_id == "theory.chord.root_degree"
    )
    large_root = next(
        item
        for item in large_supervisions
        if item.task_id == "theory.chord.root_degree"
    )
    assert large_root.per_row_loss.shape[0] == (
        128 * small_root.per_row_loss.shape[0]
    )
    small_ops = routing_operation_counts(model.task_specs, small_supervisions)
    large_ops = routing_operation_counts(model.task_specs, large_supervisions)
    assert small_ops == large_ops
    assert small_ops.prediction_task_visits == len(ACTIVE_TASK_IDS)
    assert small_ops.candidate_node_type_visits == sum(
        len(spec.node_types) for spec in active_task_head_specs()
    )
    assert small_ops.supervision_task_visits == len(ACTIVE_TASK_IDS)
    assert 0 < small_ops.tensor_group_reductions <= len(ACTIVE_TASK_IDS)
