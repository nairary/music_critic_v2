from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from music_critic.models import (
    ACTIVE_TASK_IDS,
    DILEMMADATA_ACTIVE_TASK_IDS,
    DILEMMADATA_OPEN_TASK_IDS,
    DILEMMADATA_PU_TASK_IDS,
    DilemmadataHierarchicalConfig,
    DilemmadataHierarchicalModel,
    TaskSupervision,
    aggregate_dilemmadata_source_entry_losses,
    class_weight_artifact,
    class_weight_tensors,
    load_dilemmadata_encoder_state,
)
from music_critic.tasks import collate_multisource_samples
from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK
from tests.adapters.test_dilemmadata_targets import _sample, _target
from tests.adapters.test_dilemmadata import CORPUS


def _model() -> DilemmadataHierarchicalModel:
    return DilemmadataHierarchicalModel(
        DilemmadataHierarchicalConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )


def _batch():
    return collate_multisource_samples(
        (
            _sample(_target(CORPUS, "an:training:same")),
            _sample(_target(CORPUS, "dlc:demo:same")),
        )
    )


def _supervision(task_id: str, losses, samples, entries):
    count = len(losses)
    sequence = torch.arange(count, dtype=torch.long)
    return TaskSupervision(
        task_id=task_id,
        target_row_indices=sequence,
        candidate_indices=sequence,
        node_type_codes=torch.zeros(count, dtype=torch.long),
        global_entity_indices=sequence,
        sample_indices=torch.tensor(samples, dtype=torch.long),
        source_entry_indices=torch.tensor(entries, dtype=torch.long),
        per_row_loss=torch.tensor(losses, dtype=torch.float32),
    )


def test_exact_four_distinct_heads_and_no_pu_or_open_head() -> None:
    model = _model()
    assert tuple(spec.task_id for spec in model.task_specs) == (
        DILEMMADATA_ACTIVE_TASK_IDS
    )
    assert {
        spec.task_id.split(".")[1] for spec in model.task_specs
    } == {"an", "dlc"}
    assert not set(DILEMMADATA_PU_TASK_IDS) & set(
        spec.task_id for spec in model.task_specs
    )
    assert not set(DILEMMADATA_OPEN_TASK_IDS) & set(
        spec.task_id for spec in model.task_specs
    )
    assert not set(ACTIVE_TASK_IDS) & set(
        spec.task_id for spec in model.task_specs
    )
    assert (
        DILEMMADATA_TARGET_ENCODING_BY_TASK[
            "dilemmadata.an.chord.quality"
        ].vocabulary
        != DILEMMADATA_TARGET_ENCODING_BY_TASK[
            "dilemmadata.dlc.chord.quality"
        ].vocabulary
    )
    assert model.task_heads.heads["task_00"] is not model.task_heads.heads[
        "task_02"
    ]
    state_names = tuple(model.state_dict())
    assert not any(task_id in name for task_id in DILEMMADATA_PU_TASK_IDS for name in state_names)
    assert not any(task_id in name for task_id in DILEMMADATA_OPEN_TASK_IDS for name in state_names)


def test_target_mutation_cannot_change_logits_and_gradients_reach_encoder_heads() -> None:
    model = _model()
    batch = _batch()
    model.eval()
    with torch.no_grad():
        encoded, raw_predictions = model.predict(batch.raw_graph_batch)
    mutated_targets = []
    for target in batch.target_batches:
        if target.task_id in DILEMMADATA_ACTIVE_TASK_IDS:
            values = target.values.clone()
            values[target.availability_mask] = 0
            mutated_targets.append(replace(target, values=values))
        else:
            mutated_targets.append(target)
    mutated = replace(batch, target_batches=tuple(mutated_targets))
    with torch.no_grad():
        original_output = model.supervise(
            encoded, raw_predictions, batch.target_batches
        )
        output = model.supervise(
            encoded, raw_predictions, mutated.target_batches
        )
    assert original_output.predictions is raw_predictions
    assert output.predictions is raw_predictions
    assert all(
        row.logits is prediction.logits
        for row, prediction in zip(
            output.predictions, raw_predictions, strict=True
        )
    )
    assert output.encoder.local_encoder.final_output.embeddings
    assert output.encoder.coarse
    assert output.encoder.fused.embeddings

    model.train()
    trained = model(batch)
    assert trained.harmonic_loss.total_loss is not None
    trained.harmonic_loss.total_loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(
        value is not None and torch.count_nonzero(value)
        for name, value in gradients.items()
        if name.startswith("local_baseline.encoder.")
    )
    assert sum(
        any(
            value is not None and torch.count_nonzero(value)
            for name, value in gradients.items()
            if name.startswith(f"task_heads.heads.task_{index:02d}.")
        )
        for index in range(4)
    ) == 2

    model.zero_grad(set_to_none=True)
    _, predictions = model.predict(batch.raw_graph_batch)
    synthetic = tuple(
        TaskSupervision(
            task_id=prediction.task_id,
            target_row_indices=torch.zeros(1, dtype=torch.long),
            candidate_indices=torch.zeros(1, dtype=torch.long),
            node_type_codes=prediction.candidate_node_type_codes[:1],
            global_entity_indices=prediction.global_entity_indices[:1],
            sample_indices=prediction.sample_indices[:1],
            source_entry_indices=torch.zeros(1, dtype=torch.long),
            per_row_loss=F.cross_entropy(
                prediction.logits[:1],
                torch.zeros(1, dtype=torch.long),
                reduction="none",
            ),
        )
        for prediction in predictions
    )
    all_head_loss = aggregate_dilemmadata_source_entry_losses(
        synthetic,
        task_weights={task_id: 1.0 for task_id in DILEMMADATA_ACTIVE_TASK_IDS},
    )
    assert all_head_loss.total_loss is not None
    all_head_loss.total_loss.backward()
    gradients = dict(model.named_parameters())
    for index in range(4):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for name, parameter in gradients.items()
            if name.startswith(f"task_heads.heads.task_{index:02d}.")
        )


def test_original_and_mutated_joins_reuse_one_raw_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    batch = _batch()
    model.eval()
    with torch.no_grad():
        encoded, predictions = model.predict(batch.raw_graph_batch)
    mutated_targets = []
    for target in batch.target_batches:
        if target.task_id in DILEMMADATA_ACTIVE_TASK_IDS:
            values = target.values.clone()
            values[target.availability_mask] = (
                values[target.availability_mask] + 1
            ) % next(
                spec.output_dim
                for spec in model.task_specs
                if spec.task_id == target.task_id
            )
            mutated_targets.append(replace(target, values=values))
        else:
            mutated_targets.append(target)

    def forbidden_predict(*args, **kwargs):
        del args, kwargs
        raise AssertionError("supervise must not replay target-dependent predict")

    monkeypatch.setattr(model, "predict", forbidden_predict)
    with torch.no_grad():
        original = model.supervise(encoded, predictions, batch.target_batches)
        mutated = model.supervise(encoded, predictions, tuple(mutated_targets))
    assert original.predictions is predictions
    assert mutated.predictions is predictions
    assert original.harmonic_loss.total_loss is not None
    assert mutated.harmonic_loss.total_loss is not None
    assert not torch.equal(
        original.harmonic_loss.total_loss, mutated.harmonic_loss.total_loss
    )


def test_autocast_keeps_dilemmadata_heads_and_loss_fp32_with_gradients() -> None:
    model = _model()
    batch = _batch()
    model.train()
    with torch.amp.autocast("cpu", enabled=True, dtype=torch.bfloat16):
        output = model(batch)
        assert all(row.logits.dtype == torch.float32 for row in output.predictions)
        assert all(
            row.per_row_loss.dtype == torch.float32
            for row in output.supervisions
        )
        assert all(
            row.entry_mean_losses.dtype == torch.float32
            and row.mean_loss.dtype == torch.float32
            for row in output.harmonic_loss.task_losses
        )
        assert output.harmonic_loss.total_loss is not None
        assert output.harmonic_loss.total_loss.dtype == torch.float32
        loss = output.harmonic_loss.total_loss + sum(
            row.logits.square().mean() for row in output.predictions
        )
    loss.backward()
    gradients = dict(model.named_parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for name, parameter in gradients.items()
        if name.startswith("local_baseline.encoder.")
    )
    for index in range(4):
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for name, parameter in gradients.items()
            if name.startswith(f"task_heads.heads.task_{index:02d}.")
        )


def test_plain_cpu_fp32_forward_remains_predict_then_supervise() -> None:
    model = _model()
    batch = _batch()
    model.eval()
    with torch.no_grad():
        direct = model(batch)
        encoded, predictions = model.predict(batch.raw_graph_batch)
        composed = model.supervise(encoded, predictions, batch.target_batches)
    assert all(row.logits.dtype == torch.float32 for row in direct.predictions)
    assert direct.harmonic_loss.total_loss is not None
    assert direct.harmonic_loss.total_loss.dtype == torch.float32
    assert all(
        torch.equal(left.logits, right.logits)
        for left, right in zip(
            direct.predictions, composed.predictions, strict=True
        )
    )
    assert torch.equal(
        direct.harmonic_loss.total_loss, composed.harmonic_loss.total_loss
    )


def test_source_entry_normalization_is_candidate_count_invariant() -> None:
    first, second = DILEMMADATA_ACTIVE_TASK_IDS[:2]
    report = aggregate_dilemmadata_source_entry_losses(
        (
            _supervision(first, [1.0, 3.0, 5.0], [0, 0, 1], [0, 0, 0]),
            _supervision(second, [7.0], [0], [0]),
        ),
        task_weights={first: 1.0, second: 1.0},
    )
    assert report.task_losses[0].expanded_row_count == 3
    assert report.task_losses[0].effective_source_entry_count == 2
    assert report.task_losses[0].entry_row_counts.tolist() == [2, 1]
    assert report.task_losses[0].mean_loss.item() == pytest.approx(3.5)
    assert report.total_loss.item() == pytest.approx(10.5)

    duplicated = aggregate_dilemmadata_source_entry_losses(
        (
            _supervision(
                first,
                [1.0, 3.0, 1.0, 3.0, 5.0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ),
        ),
        task_weights={first: 1.0},
    )
    assert duplicated.task_losses[0].mean_loss.item() == pytest.approx(3.5)


def test_encoder_only_transfer_is_exact_fresh_and_failure_atomic() -> None:
    source = _model()
    destination = _model()
    prefixes = (
        "local_baseline.encoder.",
        "context_encoder.pooling.",
        "context_encoder.transformer.",
        "context_encoder.fusion.",
    )
    export = {
        "encoder_state": {
            name: value.detach().clone()
            for name, value in source.state_dict().items()
            if name.startswith(prefixes)
        }
    }
    fresh = {
        name: value.detach().clone()
        for name, value in destination.state_dict().items()
        if not name.startswith(prefixes)
    }
    report = load_dilemmadata_encoder_state(
        destination,
        export,
        source_kind="phase8b_multilevel_ssl",
        source_checkpoint_sha256="a" * 64,
        transfer_mode="frozen_probe",
    )
    assert report.encoder_frozen is True
    assert report.supervised_heads_transferred is False
    assert report.ssl_heads_transferred is False
    assert set(report.loaded_tensors) == set(export["encoder_state"])
    assert not set(report.loaded_tensors) & set(report.optimizer_parameter_names)
    assert all(
        torch.equal(destination.state_dict()[name], value)
        for name, value in fresh.items()
    )

    invalid = _model()
    before = {
        name: value.detach().clone() for name, value in invalid.state_dict().items()
    }
    malformed = {"encoder_state": dict(export["encoder_state"])}
    first_name = next(iter(malformed["encoder_state"]))
    malformed["encoder_state"][first_name] = torch.zeros(1)
    with pytest.raises(ValueError, match="tensor_incompatible"):
        load_dilemmadata_encoder_state(
            invalid,
            malformed,
            source_kind="phase7a_ssl",
            source_checkpoint_sha256="b" * 64,
            transfer_mode="full_finetune",
        )
    assert all(
        torch.equal(invalid.state_dict()[name], value)
        for name, value in before.items()
    )


def test_optional_class_weights_are_train_only_and_fingerprinted() -> None:
    counts = {
        task_id: tuple(
            1
            for _ in DILEMMADATA_TARGET_ENCODING_BY_TASK[
                task_id
            ].vocabulary
        )
        for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    }
    artifact = class_weight_artifact(
        counts,
        policy="inverse_sqrt_frequency",
        train_membership_fingerprint="a" * 64,
    )
    assert artifact["source_split"] == "train_only"
    assert artifact["policy"] == "inverse_sqrt_frequency"
    assert len(artifact["fingerprint"]) == 64
    with pytest.raises(ValueError, match="config_invalid"):
        class_weight_artifact(
            counts,
            policy="unweighted",
            train_membership_fingerprint="validation",
        )


def test_supported_inverse_sqrt_weights_are_train_only_and_zero_safe() -> None:
    counts = {
        task_id: tuple(
            0 if index == 0 else index
            for index, _ in enumerate(
                DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary
            )
        )
        for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    }
    artifact = class_weight_artifact(
        counts,
        policy="inverse_sqrt_frequency_supported",
        train_membership_fingerprint="a" * 64,
    )
    tensors, evidence = class_weight_tensors(artifact, device=torch.device("cpu"))
    assert evidence["source_split"] == "train_only"
    for task_id, values in tensors.items():
        assert values.dtype == torch.float32
        assert float(values[0]) == 0.0
        counts_tensor = torch.tensor(counts[task_id], dtype=torch.float32)
        assert torch.isclose((counts_tensor * values).sum() / counts_tensor.sum(), torch.tensor(1.0))
    artifact["weights"][DILEMMADATA_ACTIVE_TASK_IDS[0]][1] = -1.0
    with pytest.raises(ValueError, match="artifact_invalid"):
        class_weight_tensors(artifact, device=torch.device("cpu"))
