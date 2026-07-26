from __future__ import annotations

from dataclasses import replace

import torch

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models import (
    ACTIVE_TASK_IDS,
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.tasks import (
    collate_multisource_samples,
    prepare_multisource_sample,
)
from tests.models.test_heads_losses import (
    _add_overlapping_root,
    _masked_root_piece,
    _replaced_root_piece,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _model() -> HierarchicalHeterogeneousBaseline:
    return HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )


def _batch(piece):
    return collate_multisource_samples(
        (prepare_multisource_sample(piece),)
    )


def test_candidate_counts_and_raw_only_inference_match_phase6a(
    mixed_batch,
) -> None:
    model = _model().eval()
    output = model(mixed_batch)
    assert tuple(item.task_id for item in output.predictions) == ACTIVE_TASK_IDS
    assert sum(item.logits.shape[0] for item in output.predictions) == 237
    raw = replace(_hook_piece(), annotations=(), targets=())
    raw_output = model(_batch(raw))
    assert sum(
        item.logits.shape[0] for item in raw_output.predictions
    ) == 79
    assert raw_output.supervisions == ()
    assert raw_output.harmonic_loss.total_loss is None
    encoded, predictions = model.predict(
        _batch(raw).raw_graph_batch
    )
    assert encoded.fused.embeddings["note"].shape[0] == 1
    assert sum(item.logits.shape[0] for item in predictions) == 79


def test_target_replace_delete_mask_add_cannot_change_eval_logits() -> None:
    base = _hook_piece()
    variants = (
        _replaced_root_piece(),
        replace(base, annotations=(), targets=()),
        _masked_root_piece(),
        _add_overlapping_root(base, value="1"),
    )
    torch.manual_seed(719)
    model = _model().eval()
    reference = model(_batch(base), include_reconstruction=False)
    reference_logits = {
        item.task_id: item.logits.detach().clone()
        for item in reference.predictions
    }
    for piece in variants:
        output = model(_batch(piece), include_reconstruction=False)
        assert all(
            torch.equal(reference_logits[item.task_id], item.logits)
            for item in output.predictions
        )


def _has_nonzero_gradient(module) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_gradients_reach_every_new_and_inherited_component(
    mixed_batch,
) -> None:
    torch.manual_seed(727)
    model = _model().train()
    output = model(mixed_batch)
    assert output.harmonic_loss.total_loss is not None
    assert output.reconstruction_loss is not None
    (
        output.harmonic_loss.total_loss + output.reconstruction_loss
    ).backward()
    pooling = model.context_encoder.pooling
    for module in (
        pooling.bar_beat,
        pooling.bar_onset,
        pooling.bar_note,
        pooling.track_note,
        pooling.bar_builder,
        pooling.track_builder,
    ):
        assert _has_nonzero_gradient(module)
    layer = model.context_encoder.transformer.encoder.layers[0]
    assert _has_nonzero_gradient(layer.self_attn)
    assert _has_nonzero_gradient(layer.linear1)
    assert _has_nonzero_gradient(layer.linear2)
    for node_type in MANDATORY_NODE_TYPES:
        assert _has_nonzero_gradient(
            model.local_baseline.encoder.feature_encoder.node_encoders[
                node_type
            ]
        )
        assert _has_nonzero_gradient(
            model.context_encoder.fusion.fusions[node_type]
        )
    for head in model.local_baseline.task_heads.heads.values():
        assert _has_nonzero_gradient(head)


def test_deterministic_tiny_overfit_decreases_both_losses(
    mixed_batch,
) -> None:
    torch.manual_seed(733)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = model(mixed_batch)
    assert initial.harmonic_loss.total_loss is not None
    assert initial.reconstruction_loss is not None
    initial_harmonic = float(initial.harmonic_loss.total_loss.detach())
    initial_reconstruction = float(initial.reconstruction_loss.detach())
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        output = model(mixed_batch)
        assert output.harmonic_loss.total_loss is not None
        assert output.reconstruction_loss is not None
        (
            output.harmonic_loss.total_loss + output.reconstruction_loss
        ).backward()
        optimizer.step()
    final = model(mixed_batch)
    assert float(final.harmonic_loss.total_loss.detach()) < initial_harmonic
    assert (
        float(final.reconstruction_loss.detach())
        < initial_reconstruction
    )
