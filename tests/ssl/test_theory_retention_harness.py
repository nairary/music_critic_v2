"""Synthetic CPU contracts for the fixed-width theory/SSL bridge.

The tiny canonical fixture is explicitly synthetic. Labels and uniform class
weights below are generated from the live registry, without corpus factories,
historical class-weight artifacts, or trained checkpoint access.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    FrozenClassWeights,
    align_target_sidecars_after_prediction,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    PRIMARY_HEADS,
)
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.hierarchical_masking import HierarchyMaskPolicyConfig
from music_critic.ssl.masking import (
    prepare_hierarchy_mask_binding,
    prepare_mask_binding,
)
from music_critic.ssl.model import Phase8AHierarchySSLForwardOutput
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartError,
    TheoryWarmstartSSLModel,
    export_theory_warmstart_state,
    load_theory_warmstart,
)


_CANONICAL_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "data" / "canonical_piece_v2.json"
)
_SHARED_PREFIXES = (
    "local_baseline.encoder.",
    "context_encoder.pooling.",
    "context_encoder.transformer.",
    "context_encoder.fusion.",
)
_SSL_PREFIXES = (
    "feature_mask_token",
    "decoder.",
    "bar_projector_predictor.",
    "song_projector_predictor.",
)
_THEORY_PREFIXES = ("sequence_decoder.", "task_heads.")
_INACTIVE_PREFIXES = (
    "encoder.local_baseline.task_heads.",
    "encoder.local_baseline.reconstruction_heads.",
)


def _synthetic_batch():
    piece = load_piece(_CANONICAL_FIXTURE)
    assert piece.dataset_name == "synthetic"
    graph = build_raw_graph(piece)
    sample = SSLRawSample(
        raw_graph=graph,
        raw_graph_fingerprint=graph_fingerprint(graph),
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
    )
    return collate_ssl_samples((sample,))


def _synthetic_batch_without_notes():
    original = load_piece(_CANONICAL_FIXTURE)
    assert original.notes
    piece = replace(original, notes=())
    graph = build_raw_graph(piece)
    assert original.notes and piece.bars == original.bars and piece.beats == original.beats
    batch = collate_ssl_samples((SSLRawSample(
        raw_graph=graph,
        raw_graph_fingerprint=graph_fingerprint(graph),
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
    ),))
    assert batch.raw_graph_batch["note"].num_nodes == 0
    assert batch.raw_graph_batch["onset"].num_nodes == 0
    assert batch.raw_graph_batch["beat"].num_nodes > 0
    return batch


def _synthetic_weights() -> FrozenClassWeights:
    sizes = {task: TASK_BY_ID[task].class_count for task in ACTIVE_HEADS}
    return FrozenClassWeights(
        values={task: torch.ones(size, dtype=torch.float32)
                for task, size in sizes.items()},
        supported={task: torch.ones(size, dtype=torch.bool)
                   for task, size in sizes.items()},
        payload_fingerprint=fingerprint({
            "provenance": "synthetic_in_memory",
            "weighting": "uniform_all_registry_classes_supported",
            "class_counts": sizes,
        }),
    )


def _synthetic_sidecar(batch, *, label_index: int = 1, masked: bool = False):
    graph = batch.raw_graph_batch
    onset_id = graph["onset"].entity_id[0][0]
    numerator, denominator = map(int, onset_id.removeprefix("onset:").split("_"))
    entity_ids = {
        "harmonic_event": "harmonic-event:ws2-synthetic",
        "onset": "onset-sidecar:ws2-synthetic",
        "note": graph["note"].entity_id[0][0],
    }
    entities = []
    for level, entity_id in entity_ids.items():
        targets = {}
        for task in ACTIVE_HEADS:
            if TASK_BY_ID[task].prediction_level != level:
                continue
            labels = get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels
            label = labels[label_index % len(labels)]
            targets[task] = {
                "available": not masked,
                "masked": masked,
                "canonical_value": None if masked else label,
                "source_value": None if masked else label,
                "missing_reason": "synthetic_missing" if masked else None,
                "provenance": {"origin": "synthetic_in_memory"},
            }
        entities.append({
            "entity_type": level,
            "canonical_entity_id": entity_id,
            "onset_qn": {"num": numerator, "den": denominator},
            "targets": targets,
        })
    sidecar = {
        "schema_version": "analysisgnn-source-native-target-sidecar-v1",
        "record_id": "synthetic:ws2-record",
        "source_component_id": "synthetic:ws2-component",
        "dialect": "dlc",
        "entities": entities,
        "relations": [{
            "relation": "harmonic_event_to_beat",
            "source_entity_id": entity_ids["harmonic_event"],
            "target_entity_id": graph["beat"].entity_id[0][0],
        }],
        "provenance": "synthetic_in_memory",
    }
    sidecar["fingerprint"] = fingerprint(sidecar)
    return sidecar


@pytest.fixture
def setup():
    torch.manual_seed(2201)
    source = CorrectedAnalysisGNNModel()
    destination = TheoryWarmstartSSLModel()
    warmstart = load_theory_warmstart(
        destination, export_theory_warmstart_state(source)
    )
    harness = TheoryRetentionHarness(warmstart)
    batch = _synthetic_batch()
    binding = prepare_mask_binding(
        batch,
        global_seed=2202,
        epoch=0,
        stage="train",
        requested_mask_rate=harness.ssl_model.ssl_config.mask_rate,
    )
    return source, harness, batch, binding, _synthetic_sidecar(batch), _synthetic_weights()


def _tree_equal(actual, expected) -> None:
    if isinstance(expected, Tensor):
        assert isinstance(actual, Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _tree_equal(left, right)
    else:
        assert actual == expected


def _parameters_with_prefix(model, prefix):
    parameters = [value for name, value in model.named_parameters()
                  if name.startswith(prefix)]
    assert parameters, prefix
    return parameters


def _assert_gradient_group(model, prefix, *, active: bool):
    parameters = _parameters_with_prefix(model, prefix)
    if not active:
        assert all(parameter.grad is None for parameter in parameters), prefix
        return
    gradients = [parameter.grad for parameter in parameters
                 if parameter.grad is not None]
    assert gradients, prefix
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients), prefix
    assert any(bool(gradient.abs().sum() > 0) for gradient in gradients), prefix


def _forward(harness, batch, binding, sidecar, weights, *, lambda_ssl, lambda_theory):
    return harness(
        lambda_ssl=lambda_ssl,
        lambda_theory=lambda_theory,
        ssl_batch=batch if lambda_ssl else None,
        prepared_mask_binding=binding if lambda_ssl else None,
        theory_raw_batch=batch.raw_graph_batch if lambda_theory else None,
        theory_sidecars=(sidecar,) if lambda_theory else None,
        class_weights=weights if lambda_theory else None,
    )


def test_shared_parameter_identity_and_optimizer_inventory(setup):
    source, harness, _, _, _, _ = setup
    source_ids = {id(parameter) for parameter in source.parameters()}
    theory_named = dict(harness.theory_model.named_parameters())
    ssl_named = dict(harness.ssl_model.named_parameters())
    for name, parameter in theory_named.items():
        if name.startswith(_SHARED_PREFIXES):
            assert ssl_named[f"encoder.{name}"] is parameter
            assert id(parameter) not in source_ids
    inventory = harness.parameter_inventory()
    assert {"shared", "ssl", "theory", "inactive"} <= inventory.keys()
    parameters = harness.optimizer_parameters()
    assert isinstance(parameters, tuple)
    ids = [id(parameter) for parameter in parameters]
    assert len(ids) == len(set(ids))
    expected = {
        id(parameter) for name, parameter in theory_named.items()
        if name.startswith((*_SHARED_PREFIXES, *_THEORY_PREFIXES))
    } | {
        id(parameter) for name, parameter in ssl_named.items()
        if name.startswith(_SSL_PREFIXES)
    }
    assert set(ids) == expected
    assert not source_ids & set(ids)
    assert not {
        id(parameter) for name, parameter in ssl_named.items()
        if name.startswith(_INACTIVE_PREFIXES)
    } & set(ids)
    assert harness.ssl_model.encoder_config == source.config.hierarchy_config()
    assert harness.theory_model.config == source.config


@pytest.mark.parametrize("lambda_ssl,lambda_theory", [(1.0, 0.0), (0.0, 1.0), (0.7, 0.3)])
def test_three_objective_regimes_have_only_active_branch_gradients(
    setup, lambda_ssl, lambda_theory
):
    _, harness, batch, binding, sidecar, weights = setup
    harness.train()
    harness.zero_grad(set_to_none=True)
    output = _forward(
        harness, batch, binding, sidecar, weights,
        lambda_ssl=lambda_ssl, lambda_theory=lambda_theory,
    )
    assert output.total is not None and bool(torch.isfinite(output.total))
    assert isinstance(output.counts, dict)
    assert isinstance(output.diagnostics, tuple)
    assert output.parameter_inventory == harness.parameter_inventory()
    assert output.counts["ssl_samples"] == (1 if lambda_ssl else 0)
    assert output.counts["theory_samples"] == (1 if lambda_theory else 0)
    assert output.counts["theory_valid_rows"] == (18 if lambda_theory else 0)
    assert output.counts["theory_active_heads"] == (18 if lambda_theory else 0)
    assert output.counts["theory_masked_rows"] == 0
    assert output.counts["theory_unsupported_rows"] == 0
    assert (output.ssl_loss is not None) == bool(lambda_ssl)
    assert (output.theory_loss is not None) == bool(lambda_theory)
    expected = sum(
        weight * loss for weight, loss in (
            (lambda_ssl, output.ssl_loss), (lambda_theory, output.theory_loss)
        ) if loss is not None
    )
    torch.testing.assert_close(output.total, expected)
    output.total.backward()
    for prefix in _SHARED_PREFIXES:
        _assert_gradient_group(harness.theory_model, prefix, active=True)
    for prefix in _SSL_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=bool(lambda_ssl))
    for prefix in _THEORY_PREFIXES:
        _assert_gradient_group(harness.theory_model, prefix, active=bool(lambda_theory))
    if lambda_theory:
        for head in harness.theory_model.task_heads.heads.values():
            assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                       for parameter in head.parameters())
        assert output.theory_report.primary is not None
        assert output.theory_report.auxiliary is not None
        assert sum(report.valid_row_count for report in output.theory_report.heads.values()) == 18
        torch.testing.assert_close(
            output.theory_loss,
            output.theory_report.primary + 0.25 * output.theory_report.auxiliary,
        )
    for prefix in _INACTIVE_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=False)
    if lambda_ssl:
        ssl = output.ssl_output
        assert output.counts["ssl_selected_notes"] == ssl.selected_global_note_indices.numel()
        for target in (ssl.targets.note, ssl.targets.bar, ssl.targets.song,
                       ssl.bar_latent.target, ssl.song_latent.target):
            assert not target.requires_grad and target.grad_fn is None


def test_mixed_then_raw_only_step_preserves_source_theory_heads_and_fixtures(setup):
    source, harness, batch, binding, sidecar, weights = setup
    harness.train()
    source_before = model_state_fingerprint(source)
    source_modes = [(module, module.training) for module in source.modules()]
    graph_before = deepcopy(batch.raw_graph_batch.to_dict())
    sidecar_before = deepcopy(sidecar)
    weight_values_before = deepcopy(dict(weights.values))
    weight_support_before = deepcopy(dict(weights.supported))
    shared_before = {
        name: value.detach().clone()
        for name, value in harness.theory_model.state_dict().items()
        if name.startswith(_SHARED_PREFIXES)
    }
    # A fixed coefficient and learning rate exercise mechanics only.
    optimizer = torch.optim.AdamW(harness.optimizer_parameters(), lr=1e-3)
    mixed = _forward(
        harness, batch, binding, sidecar, weights,
        lambda_ssl=0.7, lambda_theory=0.3,
    )
    mixed.total.backward()
    optimizer.step()
    assert any(not torch.equal(value, shared_before[name])
               for name, value in harness.theory_model.state_dict().items()
               if name in shared_before)
    theory_only_before = {
        name: value.detach().clone()
        for name, value in harness.theory_model.state_dict().items()
        if name.startswith(_THEORY_PREFIXES)
    }
    optimizer.zero_grad(set_to_none=True)
    raw_only = _forward(
        harness, batch, binding, sidecar, weights,
        lambda_ssl=1.0, lambda_theory=0.0,
    )
    raw_only.total.backward()
    optimizer.step()
    for name, value in theory_only_before.items():
        assert torch.equal(harness.theory_model.state_dict()[name], value)
    assert model_state_fingerprint(source) == source_before
    assert all(module.training == mode for module, mode in source_modes)
    _tree_equal(batch.raw_graph_batch.to_dict(), graph_before)
    _tree_equal(sidecar, sidecar_before)
    _tree_equal(dict(weights.values), weight_values_before)
    _tree_equal(dict(weights.supported), weight_support_before)


def test_sidecar_replacement_masks_and_removal_leave_raw_logits_identical(setup):
    _, harness, batch, _, sidecar, weights = setup
    harness.eval()
    common = {"lambda_ssl": 0.0, "lambda_theory": 1.0,
              "theory_raw_batch": batch.raw_graph_batch}
    original = harness(**common, theory_sidecars=(sidecar,), class_weights=weights)
    replaced = harness(
        **common, theory_sidecars=(_synthetic_sidecar(batch, label_index=0),),
        class_weights=weights,
    )
    masked = harness(
        **common, theory_sidecars=(_synthetic_sidecar(batch, masked=True),),
        class_weights=weights,
    )
    removed = harness(**common)
    for output in (replaced, masked, removed):
        assert output.theory_output is not None
        for task, logits in original.theory_output.logits.items():
            assert torch.equal(output.theory_output.logits[task], logits)
    assert original.theory_loss is not None and replaced.theory_loss is not None
    assert not torch.equal(original.theory_loss, replaced.theory_loss)
    assert masked.theory_loss is None and masked.total is None
    assert all(report.valid_row_count == 0 for report in masked.theory_report.heads.values())
    assert masked.counts["theory_masked_rows"] == 18
    assert masked.counts["theory_valid_rows"] == 0
    assert removed.theory_loss is None and removed.total is None
    assert removed.diagnostics


def test_absent_theory_inputs_do_not_become_class_zero_loss(setup):
    _, harness, batch, binding, _, _ = setup
    output = harness(
        lambda_ssl=1.0, lambda_theory=1.0,
        ssl_batch=batch, prepared_mask_binding=binding,
    )
    assert output.theory_loss is None
    assert output.theory_report is None
    assert output.alignment is None
    assert output.diagnostics
    torch.testing.assert_close(output.total, output.ssl_loss)


def test_zero_weight_theory_branch_is_not_executed(setup, monkeypatch):
    _, harness, batch, binding, sidecar, weights = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-weight theory branch was executed")

    monkeypatch.setattr(harness.theory_model, "predict", forbidden)
    output = harness(
        lambda_ssl=1.0, lambda_theory=0.0,
        ssl_batch=batch, prepared_mask_binding=binding,
        theory_raw_batch=batch.raw_graph_batch, theory_sidecars=(sidecar,),
        class_weights=weights,
    )
    assert output.theory_output is None and output.theory_loss is None
    assert output.total is not None


def test_zero_weight_ssl_branch_is_not_executed(setup, monkeypatch):
    _, harness, batch, binding, sidecar, weights = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-weight SSL branch was executed")

    monkeypatch.setattr(harness.ssl_model, "forward", forbidden)
    output = harness(
        lambda_ssl=0.0, lambda_theory=1.0,
        ssl_batch=batch, prepared_mask_binding=binding,
        theory_raw_batch=batch.raw_graph_batch, theory_sidecars=(sidecar,),
        class_weights=weights,
    )
    assert output.ssl_output is None and output.ssl_loss is None
    assert output.total is not None


def test_ssl_and_labeled_branches_accept_distinct_raw_batches(setup, monkeypatch):
    _, harness, ssl_batch, binding, _, weights = setup
    theory_batch = _synthetic_batch()
    assert theory_batch.raw_graph_batch is not ssl_batch.raw_graph_batch
    seen_theory_graphs = []
    original = harness.theory_model.predict

    def inspected(graph, **kwargs):
        seen_theory_graphs.append(graph)
        return original(graph, **kwargs)

    monkeypatch.setattr(harness.theory_model, "predict", inspected)
    output = harness(
        lambda_ssl=1.0, lambda_theory=1.0,
        ssl_batch=ssl_batch, prepared_mask_binding=binding,
        theory_raw_batch=theory_batch.raw_graph_batch,
        theory_sidecars=(_synthetic_sidecar(theory_batch),),
        class_weights=weights,
    )
    assert len(seen_theory_graphs) == 1
    assert seen_theory_graphs[0] is theory_batch.raw_graph_batch
    assert output.ssl_loss is not None and output.theory_loss is not None


@pytest.mark.parametrize("case", ["missing_weights", "orphan_sidecar", "orphan_binding"])
def test_incomplete_input_combinations_fail_closed(setup, case):
    _, harness, batch, binding, sidecar, _ = setup
    arguments = {"lambda_ssl": 0.0, "lambda_theory": 1.0}
    if case == "missing_weights":
        arguments.update(theory_raw_batch=batch.raw_graph_batch, theory_sidecars=(sidecar,))
    elif case == "orphan_sidecar":
        arguments.update(theory_sidecars=(sidecar,))
    else:
        arguments.update(lambda_ssl=1.0, lambda_theory=0.0, prepared_mask_binding=binding)
    before = model_state_fingerprint(harness)
    with pytest.raises((ValueError, TypeError)):
        harness(**arguments)
    assert model_state_fingerprint(harness) == before


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf"), True])
def test_invalid_objective_coefficients_are_rejected(setup, coefficient):
    _, harness, _, _, _, _ = setup
    with pytest.raises((ValueError, TypeError)):
        harness(lambda_ssl=coefficient, lambda_theory=0.0)
    with pytest.raises((ValueError, TypeError)):
        harness(lambda_ssl=0.0, lambda_theory=coefficient)


def test_target_pass_preserves_each_module_mode_before_online_pass(setup, monkeypatch):
    _, harness, batch, binding, _, _ = setup
    harness.train()
    encoder = harness.ssl_model.encoder
    encoder.context_encoder.transformer.eval()
    encoder.local_baseline.encoder.feature_encoder.eval()
    before = [(module, module.training) for module in encoder.modules()]
    calls = []
    original = encoder._encode_prepared

    def inspected(*args, **kwargs):
        calls.append((torch.is_grad_enabled(), tuple(module.training for module, _ in before)))
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "_encode_prepared", inspected)
    output = harness(
        lambda_ssl=1.0, lambda_theory=0.0,
        ssl_batch=batch, prepared_mask_binding=binding,
    )
    assert len(calls) == 2
    assert calls[0][0] is False and not any(calls[0][1])
    assert calls[1] == (True, tuple(mode for _, mode in before))
    assert all(module.training == mode for module, mode in before)
    assert output.total is not None


def test_target_pass_restores_each_module_mode_on_exception(setup, monkeypatch):
    _, harness, batch, binding, _, _ = setup
    harness.train()
    encoder = harness.ssl_model.encoder
    encoder.context_encoder.transformer.eval()
    before = [(module, module.training) for module in encoder.modules()]

    def failed_target(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert not any(module.training for module, _ in before)
        raise RuntimeError("synthetic target-pass failure")

    monkeypatch.setattr(encoder, "_encode_prepared", failed_target)
    with pytest.raises(RuntimeError, match="synthetic target-pass failure"):
        harness(
            lambda_ssl=1.0, lambda_theory=0.0,
            ssl_batch=batch, prepared_mask_binding=binding,
        )
    assert all(module.training == mode for module, mode in before)


def test_hierarchy_binding_uses_public_hierarchy_objectives(setup):
    _, harness, batch, _, _, _ = setup
    binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=HierarchyMaskPolicyConfig.create(weights={"onset_pitch_descendants": 1.0}),
        global_seed=2203,
        epoch=0,
        requested_mask_rate=harness.ssl_model.ssl_config.mask_rate,
    )
    output = harness(
        lambda_ssl=1.0, lambda_theory=0.0,
        ssl_batch=batch, prepared_mask_binding=binding,
    )
    assert isinstance(output.ssl_output, Phase8AHierarchySSLForwardOutput)
    assert output.total is not None and bool(torch.isfinite(output.total))
    output.total.backward()
    _assert_gradient_group(harness.theory_model, _SHARED_PREFIXES, active=True)


def test_ssl_targets_are_before_theory_bigru(setup, monkeypatch):
    _, harness, batch, binding, _, _ = setup
    calls = []
    original = harness.theory_model.sequence_decoder.forward

    def inspected(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(harness.theory_model.sequence_decoder, "forward", inspected)
    output = harness(
        lambda_ssl=1.0, lambda_theory=0.0,
        ssl_batch=batch, prepared_mask_binding=binding,
    )
    assert not calls
    assert output.ssl_output.targets.note.shape[1] == 128
    assert output.ssl_output.online_encoder.fused.embeddings["note"].shape[1] == 128


def _mutate_runtime_configuration(model, mutation, *, ssl=False):
    if mutation == "head_routes":
        specs = list(model.task_heads.specs)
        root = next(index for index, spec in enumerate(specs) if spec.task_id == "root")
        bass = next(index for index, spec in enumerate(specs) if spec.task_id == "bass")
        assert specs[root].output_dim == specs[bass].output_dim
        specs[root], specs[bass] = specs[bass], specs[root]
        model.task_heads.specs = tuple(specs)
    elif mutation == "head_precision":
        heads = model.encoder.local_baseline.task_heads if ssl else model.task_heads
        heads.force_float32 = not heads.force_float32
    elif mutation == "attention_dropout":
        attention = next(module for module in model.modules()
                         if isinstance(module, torch.nn.MultiheadAttention))
        attention.dropout = 0.5
    elif mutation == "activation":
        layer = next(module for module in model.modules()
                     if isinstance(module, torch.nn.TransformerEncoderLayer))
        layer.activation = torch.nn.functional.relu
    else:
        local = model.encoder.local_baseline.encoder if ssl else model.local_baseline.encoder
        if mutation == "feature_order":
            feature = local.feature_encoder.node_encoders["note"]
            feature.categorical_specs = tuple(reversed(feature.categorical_specs))
        else:
            assert mutation == "feature_normalization"
            feature = local.feature_encoder.node_encoders["track"]
            specs = list(feature.continuous_specs)
            index = next(index for index, spec in enumerate(specs)
                         if spec.normalization == "log1p")
            specs[index] = replace(specs[index], normalization="none")
            feature.continuous_specs = tuple(specs)


@pytest.mark.parametrize("mutation", [
    "head_routes", "head_precision", "attention_dropout", "activation",
    "feature_order", "feature_normalization",
])
def test_actual_source_runtime_configuration_drift_is_rejected(setup, mutation):
    source, _, _, _, _, _ = setup
    before = model_state_fingerprint(source)
    _mutate_runtime_configuration(source, mutation)
    # No state keys, tensor shapes, dtypes, or bytes expose this config drift.
    assert model_state_fingerprint(source) == before
    with pytest.raises(TheoryWarmstartError, match="source.model_contract_incompatible"):
        export_theory_warmstart_state(source)
    assert model_state_fingerprint(source) == before


@pytest.mark.parametrize("mutation", [
    "head_precision", "attention_dropout", "activation",
    "feature_order", "feature_normalization",
])
def test_actual_destination_runtime_configuration_drift_is_rejected(setup, mutation):
    source, harness, _, _, _, _ = setup
    destination = harness.ssl_model
    payload = export_theory_warmstart_state(source)
    before = model_state_fingerprint(destination)
    flags = [(parameter, parameter.requires_grad) for parameter in destination.parameters()]
    modes = [(module, module.training) for module in destination.modules()]
    _mutate_runtime_configuration(destination, mutation, ssl=True)
    assert model_state_fingerprint(destination) == before
    with pytest.raises(TheoryWarmstartError, match="destination.model_contract_incompatible"):
        load_theory_warmstart(destination, payload)
    assert model_state_fingerprint(destination) == before
    assert all(parameter.requires_grad == flag for parameter, flag in flags)
    assert all(module.training == mode for module, mode in modes)


def _assert_original_empty_candidate_alignment(output, theory_batch, sidecar):
    expected = align_target_sidecars_after_prediction(
        output.theory_output, theory_batch.raw_graph_batch, (sidecar,)
    )
    _tree_equal(asdict(output.alignment), asdict(expected))
    missing_tasks = {
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level != "harmonic_event"
    }
    assert {row.task_id for row in output.alignment.diagnostics} == missing_tasks
    assert all(row.category == "exact_alignment_failed"
               for row in output.alignment.diagnostics)
    assert all(row in output.diagnostics for row in output.alignment.diagnostics)
    for task in missing_tasks:
        rows = output.alignment.heads[task]
        assert rows.values.shape == (1,) and rows.candidate_indices.shape == (1,)
        assert len(rows.entity_ids) == 1
        assert rows.alignment_failure_count == 1
        assert not bool(rows.valid_mask.any())
        assert output.theory_output.logits[task].shape == (0, TASK_BY_ID[task].class_count)
        head = output.theory_report.heads[task]
        assert head.valid_row_count == 0 and head.masked_row_count == 1
        assert head.unsupported_row_count == 0
        assert head.weighted_ce is None and head.normalized_contribution is None


def test_all_masked_targets_with_no_note_onset_candidates_remain_unavailable(setup):
    _, harness, ordinary_batch, _, _, weights = setup
    theory_batch = _synthetic_batch_without_notes()
    sidecar = _synthetic_sidecar(ordinary_batch, masked=True)
    graph_before = deepcopy(theory_batch.raw_graph_batch.to_dict())
    sidecar_before = deepcopy(sidecar)
    harness.eval()
    output = harness(
        lambda_ssl=0.0, lambda_theory=1.0,
        theory_raw_batch=theory_batch.raw_graph_batch,
        theory_sidecars=(sidecar,), class_weights=weights,
    )
    assert output.theory_output is not None
    assert output.theory_loss is None and output.total is None
    assert output.theory_report.primary is None and output.theory_report.auxiliary is None
    assert output.theory_report.zero_valid_heads == ACTIVE_HEADS
    assert output.counts["theory_samples"] == 1
    assert output.counts["theory_masked_rows"] == len(ACTIVE_HEADS)
    assert output.counts["theory_valid_rows"] == 0
    assert output.counts["theory_active_heads"] == 0
    assert output.counts["theory_unsupported_rows"] == 0
    assert all(head.masked_row_count == 1 and head.valid_row_count == 0
               for head in output.theory_report.heads.values())
    _assert_original_empty_candidate_alignment(output, theory_batch, sidecar)
    _tree_equal(theory_batch.raw_graph_batch.to_dict(), graph_before)
    _tree_equal(sidecar, sidecar_before)


@pytest.mark.parametrize("mixed", [False, True], ids=["theory_only", "separate_ssl_batch"])
def test_harmonic_targets_without_note_onset_candidates_have_exact_loss_and_gradients(
    setup, mixed
):
    _, harness, ssl_batch, binding, sidecar, weights = setup
    theory_batch = _synthetic_batch_without_notes()
    theory_graph_before = deepcopy(theory_batch.raw_graph_batch.to_dict())
    ssl_graph_before = deepcopy(ssl_batch.raw_graph_batch.to_dict())
    sidecar_before = deepcopy(sidecar)
    harness.train()
    harness.zero_grad(set_to_none=True)
    lambda_ssl, lambda_theory = (0.7, 0.3) if mixed else (0.0, 1.0)
    output = harness(
        lambda_ssl=lambda_ssl, lambda_theory=lambda_theory,
        ssl_batch=ssl_batch if mixed else None,
        prepared_mask_binding=binding if mixed else None,
        theory_raw_batch=theory_batch.raw_graph_batch,
        theory_sidecars=(sidecar,), class_weights=weights,
    )
    harmonic_tasks = {
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level == "harmonic_event"
    }
    # The synthetic sidecar addresses the first raw beat with class index 1;
    # uniform weights make each available head an ordinary one-row CE.
    manual_ce = {
        task: torch.nn.functional.cross_entropy(
            output.theory_output.logits[task][:1], torch.tensor([1], dtype=torch.long)
        ) for task in harmonic_tasks
    }
    primary = torch.stack([manual_ce[task] for task in PRIMARY_HEADS if task in manual_ce]).mean()
    auxiliary = torch.stack([manual_ce[task] for task in AUXILIARY_HEADS if task in manual_ce]).mean()
    expected_theory = primary + 0.25 * auxiliary
    assert output.theory_loss is not None and bool(torch.isfinite(output.theory_loss))
    torch.testing.assert_close(output.theory_report.primary, primary)
    torch.testing.assert_close(output.theory_report.auxiliary, auxiliary)
    torch.testing.assert_close(output.theory_loss, expected_theory)
    expected_total = lambda_theory * expected_theory
    if mixed:
        assert output.ssl_loss is not None and bool(torch.isfinite(output.ssl_loss))
        expected_total = expected_total + lambda_ssl * output.ssl_loss
        assert output.counts["ssl_samples"] == 1 and output.counts["ssl_selected_notes"] > 0
    else:
        assert output.ssl_loss is None and output.ssl_output is None
    torch.testing.assert_close(output.total, expected_total)
    assert output.counts["theory_valid_rows"] == len(harmonic_tasks)
    assert output.counts["theory_active_heads"] == len(harmonic_tasks)
    assert output.counts["theory_masked_rows"] == len(ACTIVE_HEADS) - len(harmonic_tasks)
    assert output.counts["theory_unsupported_rows"] == 0
    assert set(output.theory_report.zero_valid_heads) == set(ACTIVE_HEADS) - harmonic_tasks
    _assert_original_empty_candidate_alignment(output, theory_batch, sidecar)
    output.total.backward()
    for prefix in _SHARED_PREFIXES:
        _assert_gradient_group(harness.theory_model, prefix, active=True)
    for index, spec in enumerate(harness.theory_model.task_specs):
        _assert_gradient_group(
            harness.theory_model, f"task_heads.heads.task_{index:02d}.",
            active=spec.task_id in harmonic_tasks,
        )
    for prefix in _SSL_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=mixed)
    for prefix in _INACTIVE_PREFIXES:
        _assert_gradient_group(harness.ssl_model, prefix, active=False)
    _tree_equal(theory_batch.raw_graph_batch.to_dict(), theory_graph_before)
    _tree_equal(ssl_batch.raw_graph_batch.to_dict(), ssl_graph_before)
    _tree_equal(sidecar, sidecar_before)
