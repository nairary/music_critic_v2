"""Synthetic fixed-width WS2 learning contracts, not model-quality evidence.

Reuse the existing synthetic WS2 setup and labels. No corpus factories, trained
checkpoints, historical class weights, or production training loop are used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import torch

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import ACTIVE_HEADS
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.masking import prepare_mask_binding
from music_critic.ssl.model import MaskedGraphSSLConfig
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel,
    export_theory_warmstart_state,
    load_theory_warmstart,
)
from tests.ssl.test_theory_retention_harness import (
    _CANONICAL_FIXTURE,
    _INACTIVE_PREFIXES,
    _SHARED_PREFIXES,
    _SSL_PREFIXES,
    _THEORY_PREFIXES,
    _assert_gradient_group,
    _synthetic_batch_without_notes,
    _synthetic_sidecar,
    _tree_equal,
    setup as ws2_setup,
)


def _report(case: str, **metrics) -> None:
    print(json.dumps({"case": case, **metrics}, sort_keys=True))


def test_fixed_width_coherent_hidden_pitch_mutation_cannot_change_online_view(ws2_setup):
    _, harness, batch, binding, _, _ = ws2_setup
    harness.eval()
    graph_before = deepcopy(batch.raw_graph_batch.to_dict())
    with torch.no_grad():
        original = harness(
            lambda_ssl=1.0, lambda_theory=0.0,
            ssl_batch=batch, prepared_mask_binding=binding,
        ).ssl_output
    assert original.targets.note.shape[1] == 128
    plan = original.mask_plans[0]
    assert any(mask.local_node_indices for mask in plan.collateral_feature_masks
               if mask.node_type == "note")
    piece = load_piece(_CANONICAL_FIXTURE)
    selected = plan.selected_local_node_indices[0]
    source_note = piece.notes[selected]
    assert source_note.note_id == batch.raw_graph_batch["note"].entity_id[0][selected]
    notes = list(piece.notes)
    notes[selected] = replace(
        source_note, pitch=source_note.pitch + (1 if source_note.pitch < 127 else -1)
    )
    changed_graph = build_raw_graph(replace(piece, notes=tuple(notes)))
    changed_batch = collate_ssl_samples((SSLRawSample(
        raw_graph=changed_graph,
        raw_graph_fingerprint=graph_fingerprint(changed_graph),
        dataset_id=piece.dataset_name, piece_id=piece.piece_id,
    ),))
    changed_binding = prepare_mask_binding(
        changed_batch, global_seed=binding.global_seed, epoch=binding.epoch,
        stage=binding.stage, requested_mask_rate=binding.requested_mask_rate,
    )
    assert changed_binding.mask_plans == binding.mask_plans
    with torch.no_grad():
        changed = harness(
            lambda_ssl=1.0, lambda_theory=0.0,
            ssl_batch=changed_batch, prepared_mask_binding=changed_binding,
        ).ssl_output
    for node_type, embeddings in original.online_encoder.fused.embeddings.items():
        assert torch.equal(embeddings, changed.online_encoder.fused.embeddings[node_type])
    for before, after in zip(original.decoder_predictions, changed.decoder_predictions, strict=True):
        assert torch.equal(before, after)
    assert original.feature_overlay.fingerprint == changed.feature_overlay.fingerprint
    target_distance = torch.linalg.vector_norm(original.targets.note - changed.targets.note)
    assert torch.isfinite(target_distance) and target_distance > 0
    assert original.note_loss.mean is not None and changed.note_loss.mean is not None
    loss_delta = changed.note_loss.mean - original.note_loss.mean
    assert torch.isfinite(loss_delta) and loss_delta != 0
    _tree_equal(batch.raw_graph_batch.to_dict(), graph_before)
    assert piece.notes[selected] == source_note
    _report(
        "fixed128_hidden_pitch", selected_notes=len(plan.selected_local_node_indices),
        online_bit_exact=True, decoder_bit_exact=True,
        target_l2=float(target_distance), reconstruction_loss_delta=float(loss_delta),
    )


def test_unsupported_theory_classes_leave_only_ssl_loss_and_gradients(ws2_setup):
    _, harness, batch, binding, sidecar, weights = ws2_setup
    harness.train()
    harness.zero_grad(set_to_none=True)
    unsupported = replace(
        weights,
        supported={task: torch.zeros_like(values) for task, values in weights.supported.items()},
        payload_fingerprint=fingerprint({
            "provenance": "synthetic_in_memory",
            "weighting": "all_registry_classes_unsupported",
            "parent": weights.payload_fingerprint,
        }),
    )
    output = harness(
        lambda_ssl=0.7, lambda_theory=0.3,
        ssl_batch=batch, prepared_mask_binding=binding,
        theory_raw_batch=batch.raw_graph_batch,
        theory_sidecars=(sidecar,), class_weights=unsupported,
    )
    assert output.theory_loss is None and output.theory_report.total is None
    assert output.theory_report.zero_valid_heads == ACTIVE_HEADS
    assert output.counts["theory_unsupported_rows"] == len(ACTIVE_HEADS)
    assert output.counts["theory_valid_rows"] == output.counts["theory_masked_rows"] == 0
    assert output.counts["theory_active_heads"] == 0
    assert "theory_no_valid_targets" in output.diagnostics
    for head in output.theory_report.heads.values():
        assert head.unsupported_row_count == 1 and head.weighted_ce is None
    assert output.ssl_loss is not None and output.total is not None
    torch.testing.assert_close(output.total, 0.7 * output.ssl_loss, rtol=0, atol=0)
    output.total.backward()
    for prefix in _SHARED_PREFIXES:
        _assert_gradient_group(harness.theory_model, prefix, active=True)
    for prefix in _SSL_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=True)
    for prefix in _THEORY_PREFIXES:
        _assert_gradient_group(harness.theory_model, prefix, active=False)
    for prefix in _INACTIVE_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=False)
    assert all(bool(value.all()) for value in weights.supported.values())
    _report(
        "all_theory_classes_unsupported", unsupported_rows=output.counts["theory_unsupported_rows"],
        theory_loss=None, theory_parameter_gradients=0,
        ssl_loss=float(output.ssl_loss.detach()), total=float(output.total.detach()),
    )


def test_unavailable_ssl_objective_preserves_theory_coefficient_and_gradients(ws2_setup):
    source, _, batch, _, sidecar, weights = ws2_setup
    harness = TheoryRetentionHarness(load_theory_warmstart(
        TheoryWarmstartSSLModel(MaskedGraphSSLConfig(mask_rate=0.0)),
        export_theory_warmstart_state(source),
    ))
    harness.train()
    harness.zero_grad(set_to_none=True)
    binding = prepare_mask_binding(
        batch, global_seed=2204, epoch=0, stage="train", requested_mask_rate=0.0,
    )
    output = harness(
        lambda_ssl=0.7, lambda_theory=0.3,
        ssl_batch=batch, prepared_mask_binding=binding,
        theory_raw_batch=batch.raw_graph_batch,
        theory_sidecars=(sidecar,), class_weights=weights,
    )
    assert output.counts["ssl_samples"] == 1 and output.counts["ssl_selected_notes"] == 0
    assert output.ssl_loss is None and output.ssl_output.note_loss.mean is None
    assert output.ssl_output.bar_latent.loss.mean is not None
    assert output.ssl_output.song_latent.loss.mean is not None
    assert {name for name, _ in output.ssl_output.objective.unavailable_components} == {
        "note_reconstruction"
    }
    assert ("ssl_unavailable", "required_component_unavailable") in output.diagnostics
    assert output.theory_loss is not None and output.total is not None
    torch.testing.assert_close(output.total, 0.3 * output.theory_loss, rtol=0, atol=0)
    output.total.backward()
    for prefix in (*_SHARED_PREFIXES, *_THEORY_PREFIXES):
        _assert_gradient_group(harness.theory_model, prefix, active=True)
    for prefix in (*_SSL_PREFIXES, *_INACTIVE_PREFIXES):
        _assert_gradient_group(harness.ssl_model, prefix, active=False)
    _report(
        "ssl_required_objective_unavailable", selected_notes=0, ssl_loss=None,
        ssl_parameter_gradients=0, theory_loss=float(output.theory_loss.detach()),
        total=float(output.total.detach()),
    )


def test_objective_transitions_update_only_active_groups_and_skip_unavailable(ws2_setup):
    """Exercise a caller-owned optimizer recipe, not a production runner."""
    _, harness, batch, binding, sidecar, weights = ws2_setup
    harness.train()
    optimizer = torch.optim.AdamW(harness.optimizer_parameters(), lr=1e-3, weight_decay=0.1)
    empty_batch = _synthetic_batch_without_notes()
    empty_binding = prepare_mask_binding(
        empty_batch, global_seed=2205, epoch=0, stage="train",
        requested_mask_rate=harness.ssl_model.ssl_config.mask_rate,
    )
    masked_sidecar = _synthetic_sidecar(batch, masked=True)
    inventory = harness.parameter_inventory()
    parameters = dict(harness.named_parameters())
    applied_steps = 0
    for mode, lambda_ssl, lambda_theory, active in (
        ("mixed", 0.7, 0.3, {"shared", "ssl", "theory"}),
        ("theory_only", 0.0, 1.0, {"shared", "theory"}),
        ("ssl_only", 1.0, 0.0, {"shared", "ssl"}),
        ("both_unavailable", 0.7, 0.3, set()),
    ):
        optimizer.zero_grad(set_to_none=True)
        assert all(parameter.grad is None for parameter in parameters.values())
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        last = mode == "both_unavailable"
        optimizer_before = deepcopy(optimizer.state_dict()) if last else None
        output = harness(
            lambda_ssl=lambda_ssl, lambda_theory=lambda_theory,
            ssl_batch=empty_batch if last else batch,
            prepared_mask_binding=empty_binding if last else binding,
            theory_raw_batch=batch.raw_graph_batch,
            theory_sidecars=(masked_sidecar if last else sidecar,), class_weights=weights,
        )
        if output.total is not None:
            assert torch.isfinite(output.total)
            output.total.backward()
            optimizer.step()
            applied_steps += 1
        else:
            assert last and output.ssl_loss is None and output.theory_loss is None
            assert output.counts["theory_masked_rows"] == len(ACTIVE_HEADS)
            assert ("ssl_unavailable", "required_component_unavailable") in output.diagnostics
            assert "theory_no_valid_targets" in output.diagnostics
            _tree_equal(optimizer.state_dict(), optimizer_before)
        changes = {}
        for role, group in inventory.items():
            names = group["names"]
            gradients = [parameters[name].grad for name in names]
            changes[role] = sum(not torch.equal(before[name], parameters[name]) for name in names)
            if role in active:
                present = [gradient for gradient in gradients if gradient is not None]
                assert present and all(bool(torch.isfinite(gradient).all()) for gradient in present)
                assert any(bool(gradient.abs().sum() > 0) for gradient in present)
                assert changes[role] > 0
            else:
                assert all(gradient is None for gradient in gradients)
                assert changes[role] == 0
            for name, gradient in zip(names, gradients, strict=True):
                if gradient is None:
                    assert torch.equal(before[name], parameters[name])
        _report("optimizer_transition", mode=mode, changed_parameter_tensors=changes,
                applied_steps=applied_steps, loss_available=output.total is not None)
    assert applied_steps == 3
