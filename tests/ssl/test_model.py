from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from music_critic.data import load_piece
from music_critic.graph import (
    RAW_FEATURE_REGISTRY,
    build_raw_graph,
    graph_fingerprint,
)
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.data import (
    SSLBatch,
    SSLRawSample,
    build_ssl_data_runtime,
    collate_ssl_samples,
)
from music_critic.ssl.contracts import MaskPlan, SSLContractError
from music_critic.ssl.masking import (
    build_mask_plan,
    prepare_mask_binding,
)
from music_critic.ssl.model import (
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)
from music_critic.ssl.views import build_feature_mask_overlay
from music_critic.training.config import DataConfig
from music_critic.training.data import _bounded_samples


_CANONICAL_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "data"
    / "canonical_piece_v2.json"
)


def _model() -> MaskedGraphSSLModel:
    return MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    )


def _batch() -> SSLBatch:
    return build_ssl_data_runtime(DataConfig(), seed=42).first_train_batch


def _binding(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    global_seed: int,
    epoch: int,
    stage: str = "train",
):
    return prepare_mask_binding(
        batch,
        global_seed=global_seed,
        epoch=epoch,
        stage=stage,
        requested_mask_rate=model.ssl_config.mask_rate,
    )


def _forward(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    global_seed: int,
    epoch: int,
    stage: str = "train",
):
    binding = _binding(
        model,
        batch,
        global_seed=global_seed,
        epoch=epoch,
        stage=stage,
    )
    return model(batch, prepared_mask_binding=binding)


def _replace_graph(batch: SSLBatch, graph: object) -> SSLBatch:
    return SSLBatch(
        raw_graph_batch=graph,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        sample_count=batch.sample_count,
        node_count=batch.node_count,
        edge_count=batch.edge_count,
    )


def _column(node_type: str, kind: str, name: str) -> int:
    return RAW_FEATURE_REGISTRY.names(node_type, kind).index(name)


def _mutate_masked_slots(
    batch: SSLBatch,
    plans: tuple[object, ...],
) -> SSLBatch:
    graph = deepcopy(batch.raw_graph_batch)
    note_ptr = graph["note"].ptr
    track_ptr = graph["track"].ptr
    categorical = ("pitch", "pitch_class", "octave")
    continuous = ("track_relative_pitch",)
    for sample_index, plan in enumerate(plans):
        for local_index in plan.selected_local_node_indices:
            row = int(note_ptr[sample_index].item()) + local_index
            for name in categorical:
                column = _column("note", "categorical", name)
                spec = RAW_FEATURE_REGISTRY.for_node(
                    "note", "categorical"
                )[column]
                graph["note"].x_cat[row, column] = (
                    int(graph["note"].x_cat[row, column].item()) + 1
                ) % int(spec.vocabulary_size)
            for name in continuous:
                column = _column("note", "continuous", name)
                available = graph["note"].x_cont_available[row, column]
                if bool(available):
                    graph["note"].x_cont[row, column] += 7.0
                else:
                    graph["note"].x_cont_available[row, column] = True
                    graph["note"].x_cont[row, column] = 0.75
        for collateral in plan.collateral_feature_masks:
            ptr = (
                note_ptr
                if collateral.node_type == "note"
                else track_ptr
            )
            for local_index in collateral.local_node_indices:
                row = int(ptr[sample_index].item()) + local_index
                for field in collateral.features:
                    column = _column(
                        collateral.node_type,
                        "continuous",
                        field.feature_name,
                    )
                    graph[collateral.node_type].x_cont[
                        row, column
                    ] += 9.0
                    graph[collateral.node_type].x_cont_available[
                        row, column
                    ] = True
    return _replace_graph(batch, graph)


def _mutate_unmasked_velocity(batch: SSLBatch, sample_index: int = 0) -> SSLBatch:
    graph = deepcopy(batch.raw_graph_batch)
    start = int(graph["note"].ptr[sample_index].item())
    end = int(graph["note"].ptr[sample_index + 1].item())
    assert start < end
    column = _column("note", "continuous", "velocity")
    graph["note"].x_cont[start, column] += 5.0
    graph["note"].x_cont_available[start, column] = True
    return _replace_graph(batch, graph)


def _fused_equal(left: object, right: object) -> bool:
    return all(
        torch.equal(
            left.online_encoder.fused.embeddings[node_type],
            right.online_encoder.fused.embeddings[node_type],
        )
        for node_type in left.online_encoder.fused.embeddings
    )


def test_masked_mutation_is_hidden_online_but_changes_target_and_loss() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    batch = _batch()
    original = _forward(model, batch, global_seed=23, epoch=2)
    changed_batch = _mutate_masked_slots(batch, original.mask_plans)
    changed_binding = _binding(
        model,
        changed_batch,
        global_seed=23,
        epoch=2,
    )
    assert changed_binding.mask_plans == original.mask_plans
    changed = model(
        changed_batch,
        prepared_mask_binding=changed_binding,
    )

    assert _fused_equal(original, changed)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original.decoder_predictions,
            changed.decoder_predictions,
            strict=True,
        )
    )
    assert not torch.equal(original.targets.note, changed.targets.note)
    assert not torch.equal(
        original.note_loss.mean, changed.note_loss.mean
    )
    assert original.feature_overlay.fingerprint == (
        changed.feature_overlay.fingerprint
    )


def test_coherent_pitch_rebuild_hides_owner_track_peer_dependencies() -> None:
    piece = load_piece(_CANONICAL_FIXTURE)
    graph = build_raw_graph(piece)
    batch = collate_ssl_samples(
        (
            SSLRawSample(
                raw_graph=graph,
                raw_graph_fingerprint=graph_fingerprint(graph),
                dataset_id=piece.dataset_name,
                piece_id=piece.piece_id,
            ),
        )
    )
    torch.manual_seed(18)
    model = _model().eval()
    original = _forward(model, batch, global_seed=24, epoch=0)
    plan = original.mask_plans[0]
    peer_mask = next(
        mask
        for mask in plan.collateral_feature_masks
        if mask.node_type == "note"
    )
    assert peer_mask.local_node_indices
    selected = plan.selected_local_node_indices[0]
    source_note = piece.notes[selected]
    replacement_pitch = (
        source_note.pitch + 1
        if source_note.pitch < 127
        else source_note.pitch - 1
    )
    notes = list(piece.notes)
    notes[selected] = replace(
        source_note,
        pitch=replacement_pitch,
    )
    changed_piece = replace(piece, notes=tuple(notes))
    changed_graph = build_raw_graph(changed_piece)
    changed_batch = collate_ssl_samples(
        (
            SSLRawSample(
                raw_graph=changed_graph,
                raw_graph_fingerprint=graph_fingerprint(changed_graph),
                dataset_id=piece.dataset_name,
                piece_id=piece.piece_id,
            ),
        )
    )
    changed_binding = _binding(
        model,
        changed_batch,
        global_seed=24,
        epoch=0,
    )
    assert changed_binding.mask_plans == original.mask_plans
    changed = model(
        changed_batch,
        prepared_mask_binding=changed_binding,
    )

    assert _fused_equal(original, changed)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            original.decoder_predictions,
            changed.decoder_predictions,
            strict=True,
        )
    )
    assert not torch.equal(original.targets.note, changed.targets.note)
    assert not torch.equal(
        original.note_loss.mean,
        changed.note_loss.mean,
    )


def test_unmasked_raw_value_can_change_online_output() -> None:
    torch.manual_seed(19)
    model = _model().eval()
    batch = _batch()
    original = _forward(model, batch, global_seed=29, epoch=0)
    changed_batch = _mutate_unmasked_velocity(batch)
    changed = _forward(
        model,
        changed_batch,
        global_seed=29,
        epoch=0,
    )
    assert not _fused_equal(original, changed)


def test_mutating_one_sample_leaves_other_samples_bit_exact() -> None:
    torch.manual_seed(21)
    model = _model().eval()
    batch = _batch()
    original = _forward(model, batch, global_seed=31, epoch=0)
    changed_batch = _mutate_unmasked_velocity(
        batch,
        sample_index=0,
    )
    changed = _forward(
        model,
        changed_batch,
        global_seed=31,
        epoch=0,
    )
    for node_type, before in original.online_encoder.fused.embeddings.items():
        membership = original.online_encoder.fused.batch_membership[
            node_type
        ]
        other_rows = torch.nonzero(
            membership != 0, as_tuple=False
        ).flatten()
        assert torch.equal(
            before.index_select(0, other_rows),
            changed.online_encoder.fused.embeddings[node_type].index_select(
                0, other_rows
            ),
        )


def test_overlay_construction_preserves_raw_graph_fingerprint() -> None:
    sample = _bounded_samples()[0][0]
    before = graph_fingerprint(sample.raw_graph)
    plan = build_mask_plan(
        sample.raw_graph,
        dataset_id=sample.dataset_id,
        piece_id=sample.piece_id,
        global_seed=37,
        epoch=0,
    )
    overlay = build_feature_mask_overlay(sample.raw_graph, (plan,))
    assert overlay.slot_masks
    assert graph_fingerprint(sample.raw_graph) == before
    assert sample.raw_graph_fingerprint == before


def test_stop_gradient_and_required_trainable_paths_receive_gradients() -> None:
    torch.manual_seed(23)
    model = _model().train()
    output = _forward(
        model,
        _batch(),
        global_seed=41,
        epoch=0,
    )
    assert output.objective.total_loss is not None
    assert not output.targets.note.requires_grad
    assert not output.targets.bar.requires_grad
    assert not output.targets.song.requires_grad
    assert not output.bar_latent.target.requires_grad
    assert not output.song_latent.target.requires_grad

    output.objective.total_loss.backward()
    named = dict(model.named_parameters())
    required_prefixes = (
        "encoder.local_baseline.encoder.",
        "encoder.context_encoder.pooling.",
        "encoder.context_encoder.transformer.",
        "encoder.context_encoder.fusion.",
        "decoder.network.",
        "bar_projector_predictor.projector.",
        "bar_projector_predictor.predictor.",
        "song_projector_predictor.projector.",
        "song_projector_predictor.predictor.",
    )
    for prefix in required_prefixes:
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(parameter.grad.abs().sum() > 0)
            for name, parameter in named.items()
            if name.startswith(prefix)
        ), prefix
    assert model.feature_mask_token.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in named.items()
        if ".task_heads." in name or ".reconstruction_heads." in name
    )


def test_no_mask_supervised_encode_remains_bit_exact() -> None:
    torch.manual_seed(27)
    model = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        )
    ).eval()
    graph = _batch().raw_graph_batch
    ordinary = model.encode(graph)
    explicit = model.encode(graph, feature_overlay=None)
    for node_type in ordinary.fused.embeddings:
        assert torch.equal(
            ordinary.fused.embeddings[node_type],
            explicit.fused.embeddings[node_type],
        )


def test_model_rejects_noncanonical_encoder_view_plan() -> None:
    batch = _batch()
    model = _model().eval()
    with pytest.raises(
        SSLContractError,
        match="require encoder view zero",
    ):
        prepare_mask_binding(
            batch,
            global_seed=37,
            epoch=0,
            encoder_view_index=1,
            requested_mask_rate=model.ssl_config.mask_rate,
        )


def test_model_rejects_validly_fingerprinted_alternate_selection() -> None:
    piece = load_piece(_CANONICAL_FIXTURE)
    graph = build_raw_graph(piece)
    batch = collate_ssl_samples(
        (
            SSLRawSample(
                raw_graph=graph,
                raw_graph_fingerprint=graph_fingerprint(graph),
                dataset_id=piece.dataset_name,
                piece_id=piece.piece_id,
            ),
        )
    )
    model = _model().eval()
    binding = _binding(
        model,
        batch,
        global_seed=41,
        epoch=0,
    )
    canonical = binding.mask_plans[0]
    different_epoch = build_mask_plan(
        graph,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        global_seed=41,
        epoch=1,
        requested_mask_rate=model.ssl_config.mask_rate,
    )
    assert (
        different_epoch.selected_local_node_indices
        != canonical.selected_local_node_indices
    )
    alternate = MaskPlan.create(
        mask_policy=canonical.mask_policy,
        mask_policy_version=canonical.mask_policy_version,
        dataset_id=canonical.dataset_id,
        piece_id=canonical.piece_id,
        stage=canonical.stage,
        epoch=canonical.epoch,
        encoder_view_index=canonical.encoder_view_index,
        selected_node_type=canonical.selected_node_type,
        selected_local_node_indices=(
            different_epoch.selected_local_node_indices
        ),
        primary_feature_group=canonical.primary_feature_group,
        collateral_feature_masks=(
            different_epoch.collateral_feature_masks
        ),
        requested_mask_rate=canonical.requested_mask_rate,
        maskable_node_count=canonical.maskable_node_count,
        realized_mask_rate=different_epoch.realized_mask_rate,
        global_seed=canonical.global_seed,
        stable_seed=canonical.stable_seed,
        stable_seed_sha256=canonical.stable_seed_sha256,
    )

    forged = copy(binding)
    object.__setattr__(forged, "mask_plans", (alternate,))
    with pytest.raises(
        SSLContractError,
        match="ordered plan fingerprints",
    ):
        model(
            batch,
            prepared_mask_binding=forged,
        )


def test_tiny_one_batch_all_ssl_losses_materially_decrease() -> None:
    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        batch = _batch()
        binding = _binding(
            model,
            batch,
            global_seed=42,
            epoch=0,
        )
        trajectory = []
        for step in range(31):
            output = model(
                batch,
                prepared_mask_binding=binding,
            )
            loss = output.objective.total_loss
            assert loss is not None and torch.isfinite(loss)
            trajectory.append(
                (
                    float(loss.detach()),
                    float(output.note_loss.mean.detach()),
                    float(output.bar_latent.loss.mean.detach()),
                    float(output.song_latent.loss.mean.detach()),
                )
            )
            if step < 30:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)
    initial, final = trajectory[0], trajectory[-1]
    assert final[0] < initial[0] * 0.25
    assert final[1] < initial[1] * 0.25
    assert final[2] < initial[2] * 0.25
    assert final[3] < initial[3] * 0.25
