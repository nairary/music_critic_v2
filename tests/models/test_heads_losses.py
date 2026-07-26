from __future__ import annotations

from dataclasses import replace

import torch

from music_critic.models import (
    ACTIVE_TASK_IDS,
    EXCLUDED_TASK_REASONS,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    active_task_head_specs,
)
from music_critic.tasks import (
    TARGET_ENCODING_BY_TASK,
    collate_multisource_samples,
    prepare_multisource_sample,
)
from tests.tasks.test_multisource_contract import _hook_piece


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


def test_task_outputs_retain_routing_and_unreduced_local_losses(mixed_batch) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant="feature_only",
            hidden_dim=16,
            gnn_layers=1,
            dropout=0.0,
        )
    )
    output = model(mixed_batch)
    assert tuple(item.task_id for item in output.tasks) == ACTIVE_TASK_IDS
    assert tuple(item.task_id for item in output.harmonic_loss.task_losses) == (
        ACTIVE_TASK_IDS
    )
    for item in output.tasks:
        assert item.per_row_loss is not None
        assert item.logits.shape[0] == item.global_entity_indices.shape[0]
        assert item.logits.shape[0] == item.sample_indices.shape[0]
        assert item.eligibility_mask.all()
        assert all(
            int(mixed_batch.raw_graph_batch[node_type].batch[index]) == sample
            for node_type, index, sample in zip(
                item.node_types,
                item.global_entity_indices.tolist(),
                item.sample_indices.tolist(),
            )
        )
    for task_loss in output.harmonic_loss.task_losses:
        assert task_loss.groups
        assert all(group.row_count > 0 for group in task_loss.groups)
        assert torch.isfinite(task_loss.mean_loss)


def test_raw_only_batch_has_no_harmonic_loss() -> None:
    piece = _hook_piece()
    piece = replace(piece, annotations=(), targets=())
    batch = collate_multisource_samples((prepare_multisource_sample(piece),))
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    output = model(batch)
    assert output.harmonic_loss.total_loss is None
    assert output.harmonic_loss.task_losses == ()
    assert all(task.logits.shape[0] == 0 for task in output.tasks)


def test_dense_rows_reduce_by_task_node_type_and_sample(mixed_batch) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    report = model(mixed_batch).harmonic_loss
    root = next(
        item for item in report.task_losses
        if item.task_id == "theory.chord.root_degree"
    )
    assert {(group.node_type, group.sample_index) for group in root.groups} == {
        ("bar", 0),
        ("beat", 0),
        ("onset", 0),
    }
    assert torch.isfinite(report.total_loss)
