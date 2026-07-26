from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from music_critic.graph import MANDATORY_NODE_TYPES, build_raw_graph
from music_critic.models import (
    ACTIVE_TASK_IDS,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    single_note_sensitivity,
)
from music_critic.tasks import (
    collate_multisource_samples,
    prepare_multisource_sample,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _task_losses(output):
    return {
        task.task_id: float(task.mean_loss.detach())
        for task in output.harmonic_loss.task_losses
    }


def test_single_note_change_remains_local_and_observable() -> None:
    torch.manual_seed(19)
    graph = build_raw_graph(_hook_piece())
    perturbed = deepcopy(graph)
    column = perturbed["note"].cat_feature_names.index("pitch")
    perturbed["note"].x_cat[0, column] += 1
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=2, dropout=0.0)
    )
    report = single_note_sensitivity(
        model, graph, perturbed, note_index=0
    )
    assert report.changed_note_identity == graph["note"].entity_id[0]
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
    assert len(report.oversmoothing) == 4
    assert "quality" in report.interpretation


def test_target_mutation_cannot_change_raw_encoder_embeddings(mixed_batch) -> None:
    hook = _hook_piece()
    root = next(
        target for target in hook.targets
        if target.task == "theory.chord.root_degree"
    )
    replacement = replace(root, values=("2",))
    mutated = replace(
        hook,
        targets=tuple(
            replacement if target.task == root.task else target
            for target in hook.targets
        ),
    )
    left = collate_multisource_samples((prepare_multisource_sample(hook),))
    right = collate_multisource_samples((prepare_multisource_sample(mutated),))
    torch.manual_seed(23)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    ).eval()
    left_output = model.encode(left.raw_graph_batch).final_output.embeddings
    right_output = model.encode(right.raw_graph_batch).final_output.embeddings
    assert all(
        torch.equal(left_output[node_type], right_output[node_type])
        for node_type in MANDATORY_NODE_TYPES
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
        tuple(task.task_id for task in output.tasks) == ACTIVE_TASK_IDS
        for output in outputs.values()
    )
    feature_params = sum(
        parameter.numel() for parameter in models["feature_only"].parameters()
    )
    gnn_params = sum(
        parameter.numel() for parameter in models["local_gnn"].parameters()
    )
    assert gnn_params > feature_params
