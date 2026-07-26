from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F
import pytest

from music_critic.data import validate_piece
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    build_raw_graph,
    graph_fingerprint,
    validate_raw_graph,
)
from music_critic.models import (
    ACTIVE_TASK_IDS,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    oversmoothing_by_group,
    perturb_canonical_note_pitch,
    single_note_sensitivity,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _task_losses(output):
    return {
        task.task_id: float(task.mean_loss.detach())
        for task in output.harmonic_loss.task_losses
    }


def test_canonical_single_note_change_rebuilds_both_production_graphs() -> None:
    torch.manual_seed(19)
    original = replace(_hook_piece(), annotations=(), targets=())
    note_id = original.notes[0].note_id
    perturbed = perturb_canonical_note_pitch(original, note_id)
    assert not validate_piece(original).errors
    assert not validate_piece(perturbed).errors
    assert original.notes[0].note_id == perturbed.notes[0].note_id == note_id
    assert perturbed.notes[0].pitch == original.notes[0].pitch + 1

    left = build_raw_graph(original)
    right = build_raw_graph(perturbed)
    validate_raw_graph(left)
    validate_raw_graph(right)
    assert graph_fingerprint(left) != graph_fingerprint(right)
    assert all(
        left[node_type].entity_id == right[node_type].entity_id
        for node_type in MANDATORY_NODE_TYPES
    )
    assert all(
        torch.equal(left[edge_type].edge_index, right[edge_type].edge_index)
        for edge_type in MANDATORY_EDGE_TYPES
    )

    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=2, dropout=0.0)
    )
    report = single_note_sensitivity(
        model, original, perturbed, note_id=note_id
    )
    assert report.changed_note_identity == note_id
    assert report.original_graph_fingerprint == graph_fingerprint(left)
    assert report.perturbed_graph_fingerprint == graph_fingerprint(right)
    assert report.topology_equal
    assert {
        (change.node_type, change.feature_kind, change.feature_name)
        for change in report.raw_feature_changes
    } == {
        ("track", "continuous", "mean_pitch"),
        ("track", "continuous", "min_pitch"),
        ("track", "continuous", "max_pitch"),
        ("note", "categorical", "pitch"),
        ("note", "categorical", "pitch_class"),
    }
    assert all(change.entity_ids for change in report.raw_feature_changes)
    note_deltas = [
        delta for delta in report.deltas if delta.node_type == "note"
    ]
    assert [delta.scale for delta in note_deltas] == [
        "feature",
        "gnn_layer_1",
        "gnn_layer_2",
        "final_skip",
    ]
    assert all(delta.l2 > 0 for delta in note_deltas)
    assert report.reconstruction_logit_l2_delta > 0
    assert len(report.oversmoothing) == (
        2 * len(MANDATORY_NODE_TYPES) * 4
    )
    assert "quality" in report.interpretation


def test_oversmoothing_never_mixes_samples_or_node_types(mixed_batch) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    encoded = model.encode(mixed_batch.raw_graph_batch, return_layers=True)
    report = oversmoothing_by_group(encoded)
    assert len(report) == 3 * 3 * len(MANDATORY_NODE_TYPES)
    feature = {
        (item.sample_index, item.node_type): item
        for item in report
        if item.scale == "feature"
    }
    assert feature[(0, "note")].node_count == 1
    assert feature[(0, "note")].status == "fewer_than_two_nodes"
    assert feature[(0, "note")].mean_pairwise_cosine is None
    assert feature[(1, "song")].status == "fewer_than_two_nodes"

    membership = encoded.feature_output.batch_membership["beat"]
    values = encoded.feature_output.embeddings["beat"][membership == 0]
    normalized = F.normalize(values, dim=-1)
    expected = (
        normalized.sum(dim=0).square().sum() - values.shape[0]
    ) / (values.shape[0] * (values.shape[0] - 1))
    actual = feature[(0, "beat")]
    assert actual.status == "available"
    assert actual.policy == "exact_linear_normalized_sum"
    assert actual.mean_pairwise_cosine == pytest.approx(
        float(expected.detach())
    )


def test_deterministic_one_batch_training_acceptance(mixed_batch) -> None:
    torch.manual_seed(29)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial_output = model(mixed_batch)
    initial = _task_losses(initial_output)
    initial_reconstruction = float(initial_output.reconstruction_loss.detach())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        output = model(mixed_batch)
        assert output.harmonic_loss.total_loss is not None
        assert output.reconstruction_loss is not None
        loss = output.harmonic_loss.total_loss + output.reconstruction_loss
        loss.backward()
        optimizer.step()
    final_output = model(mixed_batch)
    final = _task_losses(final_output)
    assert float(final_output.reconstruction_loss.detach()) < initial_reconstruction
    for task_id in (
        "theory.chord.root_degree",
        "theory.chord.adds",
        "pop909_cl.chord.root",
    ):
        assert final[task_id] < initial[task_id]
    assert set(final) == set(ACTIVE_TASK_IDS)
    for index, _ in enumerate(model.task_specs):
        head = model.task_heads.heads[f"task_{index:02d}"]
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in head.parameters()
        )
    for node_type in MANDATORY_NODE_TYPES:
        encoder = model.encoder.feature_encoder.node_encoders[node_type]
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in encoder.parameters()
        )


def test_feature_only_and_gnn_are_controlled_ablation(mixed_batch) -> None:
    configs = {
        variant: LocalBaselineConfig(
            variant=variant,
            hidden_dim=16,
            gnn_layers=1,
            dropout=0.0,
        )
        for variant in ("feature_only", "local_gnn")
    }
    models = {
        variant: LocalHeterogeneousBaseline(config)
        for variant, config in configs.items()
    }
    outputs = {
        variant: model(mixed_batch, include_reconstruction=False)
        for variant, model in models.items()
    }
    assert all(
        tuple(task.task_id for task in output.predictions) == ACTIVE_TASK_IDS
        for output in outputs.values()
    )
    feature_params = sum(
        parameter.numel() for parameter in models["feature_only"].parameters()
    )
    gnn_params = sum(
        parameter.numel() for parameter in models["local_gnn"].parameters()
    )
    assert gnn_params > feature_params
