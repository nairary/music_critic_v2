from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import textwrap

import pytest
import torch
from torch.nn import functional as F

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
from music_critic.models.diagnostics import (
    OversmoothingContractError,
    _contiguous_sample_boundaries,
    _linear_mean_pairwise_cosine,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _dense_off_diagonal_cosine_mean(values: torch.Tensor) -> float:
    """Independent quadratic oracle used only by bounded tests."""

    matrix = F.cosine_similarity(
        values[:, None, :],
        values[None, :, :],
        dim=-1,
    )
    off_diagonal = ~torch.eye(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    return float(matrix[off_diagonal].mean().detach())


def _task_losses(output):
    return {
        task.task_id: float(task.mean_loss.detach())
        for task in output.harmonic_loss.task_losses
    }


def _replace_membership(output, node_type: str, membership: torch.Tensor):
    batch_membership = dict(output.batch_membership)
    batch_membership[node_type] = membership
    return replace(output, batch_membership=batch_membership)


def test_contiguous_boundaries_cover_four_samples_once() -> None:
    membership = torch.tensor(
        [0, 0, 1, 1, 1, 2, 3, 3], dtype=torch.long
    )
    assert _contiguous_sample_boundaries(
        membership, membership.shape[0], sample_count=4
    ) == (0, 2, 5, 6, 8)
    assert _contiguous_sample_boundaries(
        torch.tensor([0, 0, 2], dtype=torch.long),
        3,
        sample_count=4,
    ) == (0, 2, 2, 3, 3)


@pytest.mark.parametrize(
    ("membership", "message"),
    (
        (torch.tensor([[0, 1]], dtype=torch.long), "membership_invalid"),
        (torch.tensor([0, 1], dtype=torch.int32), "membership_invalid"),
        (torch.tensor([0, -1], dtype=torch.long), "membership_negative"),
        (
            torch.tensor([0, 2, 1], dtype=torch.long),
            "membership_not_monotonic",
        ),
    ),
)
def test_contiguous_boundaries_reject_malformed_membership(
    membership: torch.Tensor, message: str
) -> None:
    with pytest.raises(OversmoothingContractError, match=message):
        _contiguous_sample_boundaries(
            membership, membership.numel(), sample_count=3
        )


def test_linear_oversmoothing_matches_dense_oracle_for_random_nonzero() -> None:
    generator = torch.Generator().manual_seed(211)
    values = torch.randn(17, 11, generator=generator) + 0.125
    assert torch.linalg.vector_norm(values, dim=-1).count_nonzero() == 17
    actual, zero_norm_count = _linear_mean_pairwise_cosine(values)
    assert zero_norm_count == 0
    assert actual == pytest.approx(
        _dense_off_diagonal_cosine_mean(values), abs=1e-7
    )


def test_linear_oversmoothing_matches_dense_oracle_with_one_zero() -> None:
    values = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        dtype=torch.float32,
    )
    actual, zero_norm_count = _linear_mean_pairwise_cosine(values)
    assert zero_norm_count == 1
    assert actual == pytest.approx(
        _dense_off_diagonal_cosine_mean(values), abs=1e-7
    )


def test_linear_oversmoothing_reports_all_zero_collapse() -> None:
    values = torch.zeros(5, 7)
    actual, zero_norm_count = _linear_mean_pairwise_cosine(values)
    assert zero_norm_count == values.shape[0]
    assert actual == pytest.approx(
        _dense_off_diagonal_cosine_mean(values), abs=1e-7
    )
    assert actual == 0.0


def test_linear_oversmoothing_single_row_is_unavailable() -> None:
    actual, zero_norm_count = _linear_mean_pairwise_cosine(
        torch.zeros(1, 4)
    )
    assert actual is None
    assert zero_norm_count == 1


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
    assert feature[(0, "note")].zero_norm_count == 0
    assert feature[(0, "note")].status == "fewer_than_two_nodes"
    assert feature[(0, "note")].mean_pairwise_cosine is None
    assert feature[(1, "song")].status == "fewer_than_two_nodes"

    boundaries_by_node_type = {
        node_type: _contiguous_sample_boundaries(
            encoded.feature_output.batch_membership[node_type],
            encoded.feature_output.embeddings[node_type].shape[0],
            sample_count=3,
        )
        for node_type in MANDATORY_NODE_TYPES
    }
    for sample_index in range(3):
        for node_type in MANDATORY_NODE_TYPES:
            embeddings = encoded.feature_output.embeddings[node_type]
            boundaries = boundaries_by_node_type[node_type]
            values = embeddings[
                boundaries[sample_index] : boundaries[sample_index + 1]
            ]
            assert (
                values.untyped_storage().data_ptr()
                == embeddings.untyped_storage().data_ptr()
            )
            actual = feature[(sample_index, node_type)]
            assert actual.node_count == values.shape[0]
            assert actual.zero_norm_count == int(
                (
                    torch.linalg.vector_norm(values, dim=-1) == 0
                ).sum()
            )
            if values.shape[0] < 2:
                assert actual.status == "fewer_than_two_nodes"
                assert actual.mean_pairwise_cosine is None
            else:
                assert actual.status == "available"
                assert actual.policy == "exact_linear_normalized_sum"
                assert actual.mean_pairwise_cosine == pytest.approx(
                    _dense_off_diagonal_cosine_mean(values), abs=1e-6
                )


def test_oversmoothing_reports_zero_rows_through_production_path(
    mixed_batch,
) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    encoded = model.encode(mixed_batch.raw_graph_batch, return_layers=True)
    boundaries = _contiguous_sample_boundaries(
        encoded.feature_output.batch_membership["beat"],
        encoded.feature_output.embeddings["beat"].shape[0],
        sample_count=3,
    )
    feature_embeddings = dict(encoded.feature_output.embeddings)
    beat_values = feature_embeddings["beat"].clone()
    beat_values[boundaries[0] : boundaries[1]] = 0
    feature_embeddings["beat"] = beat_values
    mutated_feature = replace(
        encoded.feature_output, embeddings=feature_embeddings
    )
    mutated = replace(encoded, feature_output=mutated_feature)
    item = next(
        value
        for value in oversmoothing_by_group(mutated)
        if (
            value.scale,
            value.sample_index,
            value.node_type,
        )
        == ("feature", 0, "beat")
    )
    assert item.node_count == boundaries[1] - boundaries[0] == 4
    assert item.zero_norm_count == item.node_count
    assert item.mean_pairwise_cosine == 0.0


def test_oversmoothing_rejects_noncontiguous_membership(mixed_batch) -> None:
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    )
    encoded = model.encode(mixed_batch.raw_graph_batch, return_layers=True)
    membership = encoded.feature_output.batch_membership["beat"].clone()
    membership[4] = 2
    membership[5] = 1
    malformed = replace(
        encoded,
        feature_output=_replace_membership(
            encoded.feature_output, "beat", membership
        ),
        layer_outputs=tuple(
            _replace_membership(output, "beat", membership)
            for output in encoded.layer_outputs
        ),
        final_output=_replace_membership(
            encoded.final_output, "beat", membership
        ),
    )
    with pytest.raises(
        OversmoothingContractError, match="membership_not_monotonic"
    ):
        oversmoothing_by_group(malformed)


def test_production_oversmoothing_uses_only_basic_embedding_slices() -> None:
    source = textwrap.dedent(inspect.getsource(oversmoothing_by_group))
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, (ast.Compare, ast.BoolOp))
        for node in ast.walk(tree)
    )
    assert "masked_select" not in source
    assert "index_select" not in source
    assert "scale_embeddings[node_type][start:end]" in source


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
