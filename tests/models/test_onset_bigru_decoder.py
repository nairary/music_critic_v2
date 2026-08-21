from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
import torch
from torch.nn import functional as F

from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    DilemmadataDecoderConfig,
    DilemmadataDecoderConfigError,
    DilemmadataHierarchicalConfig,
    DilemmadataHierarchicalModel,
    dilemmadata_model_contract_fingerprint,
    load_dilemmadata_encoder_state,
)
from music_critic.tasks import collate_multisource_samples
from tests.adapters.test_dilemmadata import CORPUS
from tests.adapters.test_dilemmadata_targets import _sample, _target


OLD_MLP_CONTRACT_FINGERPRINT = (
    "9ba93993ae5fa0e78841c4c0f60b7f9e605d250baf91b03c6ad9f587377748db"
)
OLD_MLP_STATE_FINGERPRINT = (
    "e86b577b4c91c55d66f68f22d5faceb078910c0219a179ea8590dbc7fe6f36d4"
)
OLD_MLP_OUTPUT_FINGERPRINT = (
    "efd5c7e09424247e1d82984e65ff3cf59088efd6c65bb11b4e53acd2751d06b1"
)


def _add_tensor(digest, value: torch.Tensor) -> None:
    host = value.detach().cpu().contiguous()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())


def _state_fingerprint(model: DilemmadataHierarchicalModel) -> str:
    digest = sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        _add_tensor(digest, value)
    return digest.hexdigest()


def _output_fingerprint(model: DilemmadataHierarchicalModel) -> str:
    digest = sha256()
    model.eval()
    with torch.no_grad():
        output = model(_batch(0, 1))
    for prediction in output.predictions:
        for value in (
            prediction.candidate_node_type_codes,
            prediction.global_entity_indices,
            prediction.sample_indices,
            prediction.logits,
        ):
            _add_tensor(digest, value)
    assert output.harmonic_loss.total_loss is not None
    _add_tensor(digest, output.harmonic_loss.total_loss)
    return digest.hexdigest()


def _config(kind: str) -> DilemmadataHierarchicalConfig:
    return DilemmadataHierarchicalConfig(
        hidden_dim=16,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        decoder=DilemmadataDecoderConfig(kind=kind),
    )


def _model(kind: str = "onset_bigru") -> DilemmadataHierarchicalModel:
    return DilemmadataHierarchicalModel(_config(kind))


def _samples():
    return (
        _sample(_target(CORPUS, "an:training:same")),
        _sample(_target(CORPUS, "dlc:demo:same")),
    )


def _batch(*indices: int):
    samples = _samples()
    return collate_multisource_samples(tuple(samples[index] for index in indices))


def _all_head_loss(model: DilemmadataHierarchicalModel, batch) -> torch.Tensor:
    _, predictions = model.predict(batch.raw_graph_batch)
    return torch.stack(
        [
            F.cross_entropy(
                prediction.logits,
                torch.zeros(
                    prediction.logits.shape[0],
                    dtype=torch.long,
                    device=prediction.logits.device,
                ),
            )
            for prediction in predictions
        ]
    ).sum()


def test_decoder_configuration_is_structured_and_mlp_contract_is_bit_exact() -> None:
    with pytest.raises(
        DilemmadataDecoderConfigError,
        match="dilemmadata.decoder.kind_invalid",
    ):
        DilemmadataDecoderConfig(kind="transformer")  # type: ignore[arg-type]
    with pytest.raises(
        DilemmadataDecoderConfigError,
        match="dilemmadata.decoder.hidden_dim_must_be_positive_even",
    ):
        DilemmadataHierarchicalModel(
            DilemmadataHierarchicalConfig(
                hidden_dim=15,
                attention_heads=3,
                decoder=DilemmadataDecoderConfig(kind="onset_bigru"),
            )
        )
    torch.manual_seed(17)
    mlp = DilemmadataHierarchicalModel()
    assert mlp.sequence_decoder is None
    assert not any(name.startswith("sequence_decoder.") for name in mlp.state_dict())
    assert dilemmadata_model_contract_fingerprint(mlp) == OLD_MLP_CONTRACT_FINGERPRINT
    torch.manual_seed(1701)
    fixed = DilemmadataHierarchicalModel(_config("mlp"))
    assert _state_fingerprint(fixed) == OLD_MLP_STATE_FINGERPRINT
    assert _output_fingerprint(fixed) == OLD_MLP_OUTPUT_FINGERPRINT


def test_packed_sequences_restore_rows_isolate_samples_and_ignore_padding() -> None:
    torch.manual_seed(3)
    decoder = _model().sequence_decoder
    assert decoder is not None
    decoder.eval()
    rows = torch.randn(5, 16)
    membership = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    with torch.no_grad():
        together = decoder._sequence_context(rows, membership, 3)
        first = decoder._sequence_context(rows[:2], torch.zeros(2, dtype=torch.long), 1)
        second = decoder._sequence_context(rows[2:], torch.zeros(3, dtype=torch.long), 1)
        repeated = decoder._sequence_context(rows, membership, 3)
        reversed_context = decoder._sequence_context(
            rows.flip(0), torch.tensor([0, 0, 0, 1, 1]), 2
        )
    assert torch.equal(together[:2], first)
    assert torch.equal(together[2:], second)
    assert torch.equal(together, repeated)
    assert not torch.equal(together, reversed_context.flip(0))


def test_sample_eval_output_is_identical_alone_and_in_mixed_batch() -> None:
    torch.manual_seed(5)
    model = _model()
    model.eval()
    alone = _batch(0)
    mixed = _batch(0, 1)
    with torch.no_grad():
        alone_encoded, alone_predictions = model.predict(alone.raw_graph_batch)
        mixed_encoded, mixed_predictions = model.predict(mixed.raw_graph_batch)
    for node_type in ("onset", "beat", "bar"):
        count = alone_encoded.fused.embeddings[node_type].shape[0]
        torch.testing.assert_close(
            alone_encoded.fused.embeddings[node_type],
            mixed_encoded.fused.embeddings[node_type][:count],
            atol=1e-6,
            rtol=1e-6,
        )
    for left, right in zip(alone_predictions, mixed_predictions, strict=True):
        count = left.logits.shape[0]
        onset_count = alone.raw_graph_batch["onset"].num_nodes
        beat_count = alone.raw_graph_batch["beat"].num_nodes
        bar_count = alone.raw_graph_batch["bar"].num_nodes
        assert count == onset_count + beat_count + bar_count
        # Candidate stores remain onset, beat, bar in their original row order.
        sample_rows = torch.nonzero(
            right.sample_indices == 0, as_tuple=False
        ).flatten()
        torch.testing.assert_close(
            left.logits,
            right.logits.index_select(0, sample_rows),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(
            left.candidate_node_type_codes,
            right.candidate_node_type_codes.index_select(0, sample_rows),
        )
        assert torch.equal(
            left.global_entity_indices,
            right.global_entity_indices.index_select(0, sample_rows),
        )


def test_target_mutation_reuses_one_bigru_prediction_and_raw_only_path() -> None:
    torch.manual_seed(7)
    model = _model()
    batch = _batch(0, 1)
    model.eval()
    with torch.no_grad():
        encoded, predictions = model.predict(batch.raw_graph_batch)
    mutated = []
    for target in batch.target_batches:
        if target.task_id in DILEMMADATA_ACTIVE_TASK_IDS:
            values = target.values.clone()
            values[target.availability_mask] = 0
            mutated.append(replace(target, values=values))
        else:
            mutated.append(target)
    with torch.no_grad():
        original = model.supervise(encoded, predictions, batch.target_batches)
        changed = model.supervise(encoded, predictions, tuple(mutated))
    assert original.predictions is predictions
    assert changed.predictions is predictions
    for left, right in zip(original.predictions, changed.predictions, strict=True):
        assert left.logits is right.logits
        assert torch.equal(left.candidate_node_type_codes, right.candidate_node_type_codes)
        assert torch.equal(left.global_entity_indices, right.global_entity_indices)


def test_gradients_and_optimizer_reach_gru_fusions_encoder_and_four_heads() -> None:
    torch.manual_seed(11)
    model = _model()
    batch = _batch(0, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = _all_head_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {name: value.grad for name, value in model.named_parameters()}
    prefixes = (
        "sequence_decoder.gru.",
        "sequence_decoder.onset_projection.",
        "sequence_decoder.beat_context_projection.",
        "sequence_decoder.bar_context_projection.",
        "local_baseline.encoder.",
    )
    for prefix in prefixes:
        selected = [value for name, value in gradients.items() if name.startswith(prefix)]
        assert selected and any(
            value is not None
            and bool(torch.isfinite(value).all())
            and bool(torch.count_nonzero(value))
            for value in selected
        )
    for index in range(4):
        assert any(
            value is not None and bool(torch.count_nonzero(value))
            for name, value in gradients.items()
            if name.startswith(f"task_heads.heads.task_{index:02d}.")
        )
    optimizer.step()
    for group_prefix in ("sequence_decoder.gru.", "task_heads.", "local_baseline.encoder."):
        assert any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if name.startswith(group_prefix)
        )


def test_bounded_four_head_overfit_is_finite_and_changes_every_head() -> None:
    torch.manual_seed(13)
    model = _model()
    batch = _batch(0, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    losses = []
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = _all_head_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.8
    assert any(
        not torch.equal(before[name], value.detach())
        for name, value in model.named_parameters()
        if name.startswith("sequence_decoder.gru.")
    )
    for index in range(4):
        assert any(
            not torch.equal(before[name], value.detach())
            for name, value in model.named_parameters()
            if name.startswith(f"task_heads.heads.task_{index:02d}.")
        )


def test_encoder_transfer_keeps_bigru_and_heads_fresh_and_cross_kind_rejects() -> None:
    torch.manual_seed(17)
    source = _model()
    torch.manual_seed(19)
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
        if name.startswith(("sequence_decoder.", "task_heads."))
    }
    report = load_dilemmadata_encoder_state(
        destination,
        export,
        source_kind="phase8b_multilevel_ssl",
        source_checkpoint_sha256="a" * 64,
        transfer_mode="full_finetune",
    )
    assert not set(report.loaded_tensors) & set(fresh)
    assert all(torch.equal(destination.state_dict()[name], value) for name, value in fresh.items())
    round_trip = _model()
    round_trip.load_state_dict(destination.state_dict(), strict=True)
    assert all(
        torch.equal(left, right)
        for left, right in zip(destination.state_dict().values(), round_trip.state_dict().values(), strict=True)
    )
    with pytest.raises(RuntimeError):
        _model("mlp").load_state_dict(destination.state_dict(), strict=True)


def test_onset_bigru_epoch_resume_matches_uninterrupted_and_cross_kind_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from music_critic.training import engine as training_engine
    from music_critic.training.engine import TrainingContractError, run_training
    from tests.training.test_dilemmadata_training import _compose, _runtime

    monkeypatch.setattr(
        training_engine, "build_data_runtime", lambda config, seed: _runtime()
    )

    def config(output, kind: str = "onset_bigru"):
        return _compose(
            "experiment=dilemmadata_smoke",
            "model=hierarchical",
            "data=dilemmadata",
            "device=cpu",
            "+model.decoder.kind=" + kind,
            "model.hidden_dim=16",
            "model.local_gnn_layers=1",
            "model.transformer_layers=1",
            "model.attention_heads=4",
            "model.ffn_multiplier=2",
            "model.dropout=0",
            "experiment.epochs=2",
            "experiment.steps=1",
            "optimizer.learning_rate=0.001",
            f"output_dir={output}",
        )

    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    run_training(config(uninterrupted))
    run_training(config(resumed), stop_after_epoch=1)
    resume_config = config(resumed)
    resume_config.experiment.resume_from = str(resumed / "last.pt")
    run_training(resume_config)
    left = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)["model_state"]
    right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)["model_state"]
    assert set(left) == set(right)
    assert all(torch.equal(left[name], right[name]) for name in left)

    wrong = config(resumed, "mlp")
    wrong.experiment.resume_from = str(resumed / "last.pt")
    with pytest.raises(
        TrainingContractError,
        match="run_manifest_binding_mismatch|checkpoint.metadata_mismatch",
    ):
        run_training(wrong)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_bigru_smoke_is_fp32_at_heads_and_finite() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(23)
    model = _model().to(device)
    batch = _batch(0, 1)
    from music_critic.training.device import move_multisource_batch

    batch = move_multisource_batch(batch, device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(batch)
    assert all(prediction.logits.dtype == torch.float32 for prediction in output.predictions)
    assert output.harmonic_loss.total_loss is not None
    assert output.harmonic_loss.total_loss.dtype == torch.float32
    output.harmonic_loss.total_loss.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    model.eval()
    allocated = []
    for _ in range(3):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            _, predictions = model.predict(batch.raw_graph_batch)
        assert all(bool(torch.isfinite(row.logits).all()) for row in predictions)
        del predictions
        torch.cuda.synchronize(device)
        allocated.append(torch.cuda.memory_allocated(device))
    assert allocated[-1] <= max(allocated[:-1])
