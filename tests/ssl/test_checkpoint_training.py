from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize
import pytest
import torch

from music_critic.ssl.checkpoint import (
    SSLCheckpointError,
    load_ssl_checkpoint,
    save_ssl_checkpoint,
)
from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.engine import (
    SSLTrainingError,
    _Accumulator,
    _optimize_batch,
    _plain_config,
    _training_scope_evidence,
    run_ssl_training,
)
from music_critic.ssl.model import build_ssl_model
from music_critic.training.checkpoint import capture_rng_state


_CHECKPOINT_CONFIG = {"purpose": "phase7a-checkpoint-test"}
_DATA_FINGERPRINTS = {
    "kind": "bounded",
    "bounded_fixture_fingerprint": "a" * 64,
    "split_fingerprint": "b" * 64,
    "train_composition_fingerprint": "d" * 64,
    "validation_composition_fingerprint": "e" * 64,
    "validation_membership_fingerprint": "c" * 64,
}


def _journal_row(
    *,
    epoch: int = 0,
    validation_loss: float | None = 0.75,
) -> dict[str, object]:
    return {
        "metric_row_version": "1.2.0",
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "learning_rate_used": 0.01,
        "next_learning_rate": 0.005,
        "train": {"total_ssl_loss": 0.75},
        "validation": (
            None
            if validation_loss is None
            else {"total_ssl_loss": validation_loss}
        ),
        "gradient_coverage": None,
    }


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _config(
    output: Path,
    experiment: str,
    *,
    dropout: float = 0.0,
):
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="ssl_training",
            overrides=[
                f"experiment={experiment}",
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=2",
                "model.ffn_multiplier=2",
                f"model.dropout={dropout}",
                "data=bounded",
                "data.batch_size=3",
                "data.epoch_size=3",
                "data.validation_epoch_size=2",
                "device=cpu",
                "optimizer.learning_rate=0.02",
                "ssl.mask_rate=0.5",
                "ssl.decoder_views=1",
                "ssl.decoder_remask_prob=0",
                "ssl.projector_hidden_dim=8",
                "ssl.decoder_hidden_dim=8",
                f"output_dir={output}",
                *(
                    ["experiment.steps=8"]
                    if experiment == "one_batch"
                    else [
                        "experiment.epochs=2",
                        "experiment.collect_gradient_evidence=false",
                    ]
                ),
            ],
        )


def _assert_state_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def test_ssl_one_batch_default_learning_rate_is_phase7a_specific() -> None:
    config = _config(
        Path("/tmp/unused-phase7a-learning-rate"),
        "one_batch",
    )
    config.optimizer.learning_rate = None
    assert _plain_config(config)["optimizer"]["learning_rate"] == pytest.approx(
        3e-4
    )

    config.optimizer.learning_rate = 0.001
    assert _plain_config(config)["optimizer"]["learning_rate"] == pytest.approx(
        0.001
    )


def _checkpoint_objects():
    config = _config(Path("/tmp/unused-phase7a-checkpoint"), "pretrain")
    model = build_ssl_model(config.model, config.ssl)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(
        [
            parameter.float().square().mean()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    ).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return model, optimizer, scheduler, scaler


def _mutate_live_state(model, optimizer, scheduler, scaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(
        [
            (parameter.float() + 0.25).square().mean()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    ).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    scaler_state = scaler.state_dict()
    scaler_state["scale"] = float(scaler_state["scale"]) / 2.0
    scaler_state["_growth_tracker"] = (
        int(scaler_state["_growth_tracker"]) + 1
    )
    scaler.load_state_dict(scaler_state)
    random.random()
    torch.rand(11)


def _snapshot(model, optimizer, scheduler, scaler) -> dict[str, object]:
    return {
        "model": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "rng": capture_rng_state(),
    }


def _save_checkpoint(path: Path, model, optimizer, scheduler, scaler) -> None:
    save_ssl_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=1,
        best_validation_loss=0.75,
        epoch_journal=(_journal_row(),),
        resolved_config=_CHECKPOINT_CONFIG,
        data_fingerprints=_DATA_FINGERPRINTS,
    )


def _load_checkpoint(path: Path, model, optimizer, scheduler, scaler):
    return load_ssl_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        maximum_next_epoch=2,
        resolved_config=_CHECKPOINT_CONFIG,
        data_fingerprints=_DATA_FINGERPRINTS,
    )


def test_ssl_checkpoint_round_trip_restores_every_deterministic_state(
    tmp_path: Path,
) -> None:
    random.seed(101)
    torch.manual_seed(101)
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    checkpoint = tmp_path / "round-trip.pt"
    _save_checkpoint(checkpoint, model, optimizer, scheduler, scaler)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    _mutate_live_state(model, optimizer, scheduler, scaler)
    state = _load_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
    )

    assert state.next_epoch == 1
    assert state.best_validation_loss == 0.75
    assert state.epoch_journal == (_journal_row(),)
    _assert_state_equal(model.state_dict(), payload["model_state"])
    _assert_state_equal(optimizer.state_dict(), payload["optimizer_state"])
    _assert_state_equal(scheduler.state_dict(), payload["scheduler_state"])
    _assert_state_equal(scaler.state_dict(), payload["scaler_state"])
    _assert_state_equal(capture_rng_state(), payload["rng_state"])


def test_ssl_checkpoint_rejects_previous_contract_version_atomically(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    current = tmp_path / "current.pt"
    outdated = tmp_path / "outdated.pt"
    _save_checkpoint(current, model, optimizer, scheduler, scaler)
    payload = torch.load(
        current,
        map_location="cpu",
        weights_only=True,
    )
    payload["metadata"]["ssl_checkpoint_contract_version"] = "1.1.0"
    torch.save(payload, outdated)
    before = _snapshot(model, optimizer, scheduler, scaler)

    with pytest.raises(
        SSLCheckpointError,
        match="ssl.checkpoint.metadata_mismatch",
    ):
        _load_checkpoint(
            outdated,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )


@pytest.mark.parametrize(
    ("scenario", "forged_best_validation_loss"),
    (
        ("best_without_validation", 0.75),
        ("missing_best_with_validation", None),
        ("stale_nonminimum_best", 0.75),
    ),
)
def test_ssl_checkpoint_rejects_forged_inconsistent_best_validation_loss(
    tmp_path: Path,
    scenario: str,
    forged_best_validation_loss: float | None,
) -> None:
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    if scenario == "best_without_validation":
        journal = (_journal_row(validation_loss=None),)
        valid_best = None
    elif scenario == "missing_best_with_validation":
        journal = (_journal_row(validation_loss=0.75),)
        valid_best = 0.75
    else:
        journal = (
            _journal_row(validation_loss=0.75),
            _journal_row(epoch=1, validation_loss=0.50),
        )
        valid_best = 0.50

    valid_path = tmp_path / f"{scenario}-valid.pt"
    save_ssl_checkpoint(
        valid_path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=len(journal),
        best_validation_loss=valid_best,
        epoch_journal=journal,
        resolved_config=_CHECKPOINT_CONFIG,
        data_fingerprints=_DATA_FINGERPRINTS,
    )
    payload = torch.load(
        valid_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["best_validation_loss"] = forged_best_validation_loss
    forged_path = tmp_path / f"{scenario}-forged.pt"
    torch.save(payload, forged_path)
    before = _snapshot(model, optimizer, scheduler, scaler)

    with pytest.raises(
        SSLCheckpointError,
        match="best_validation_loss_inconsistent",
    ):
        _load_checkpoint(
            forged_path,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )


def test_ssl_checkpoint_rejects_noncanonical_epoch_journal_rows(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    invalid = _journal_row()
    invalid["epoch"] = False

    with pytest.raises(
        SSLCheckpointError,
        match="epoch_journal_order_invalid",
    ):
        save_ssl_checkpoint(
            tmp_path / "invalid-journal.pt",
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=1,
            best_validation_loss=None,
            epoch_journal=(invalid,),
            resolved_config=_CHECKPOINT_CONFIG,
            data_fingerprints=_DATA_FINGERPRINTS,
        )


def test_ssl_checkpoint_save_failure_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    destination = tmp_path / "atomic.pt"
    destination.write_bytes(b"previous-checkpoint")

    with patch(
        "music_critic.ssl.checkpoint.torch.save",
        side_effect=OSError("injected save failure"),
    ), pytest.raises(OSError, match="injected save failure"):
        _save_checkpoint(
            destination,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    assert destination.read_bytes() == b"previous-checkpoint"
    assert tuple(tmp_path.glob(".atomic.pt.*.tmp")) == ()


def test_ssl_checkpoint_application_failure_rolls_back_all_live_state(
    tmp_path: Path,
) -> None:
    random.seed(103)
    torch.manual_seed(103)
    model, optimizer, scheduler, scaler = _checkpoint_objects()
    checkpoint = tmp_path / "application-failure.pt"
    _save_checkpoint(checkpoint, model, optimizer, scheduler, scaler)
    _mutate_live_state(model, optimizer, scheduler, scaler)
    before = _snapshot(model, optimizer, scheduler, scaler)

    scheduler_type = type(scheduler)
    original_load = scheduler_type.load_state_dict
    live_scheduler_id = id(scheduler)
    live_calls = 0

    def apply_then_fail(self, state_dict):
        nonlocal live_calls
        result = original_load(self, state_dict)
        if id(self) == live_scheduler_id:
            live_calls += 1
            if live_calls == 1:
                raise RuntimeError("injected SSL scheduler application failure")
        return result

    with patch.object(
        scheduler_type,
        "load_state_dict",
        apply_then_fail,
    ):
        with pytest.raises(
            SSLCheckpointError,
            match="ssl.checkpoint.application_failed",
        ):
            _load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                scaler,
            )

    assert live_calls == 2
    _assert_state_equal(
        _snapshot(model, optimizer, scheduler, scaler),
        before,
    )


def test_epoch_boundary_resume_matches_uninterrupted_state_and_metrics(
    tmp_path: Path,
) -> None:
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"

    def resume_config(path: Path):
        config = _config(
            path,
            "pretrain",
            dropout=0.2,
        )
        config.scheduler.name = "cosine"
        # CPU resume determinism is independent of the CUDA-only AMP
        # contract. Real AMP resume remains covered by optional CUDA tests.
        config.device.amp = False
        return config

    uninterrupted = run_ssl_training(
        resume_config(uninterrupted_dir)
    )
    first_part = run_ssl_training(
        resume_config(resumed_dir),
        stop_after_epoch=1,
    )
    assert first_part["completed_epochs"] == 1

    resumed_config = resume_config(resumed_dir)
    resumed_config.experiment.resume_from = str(
        resumed_dir / "last.pt"
    )
    resumed = run_ssl_training(resumed_config)

    assert uninterrupted["completed_epochs"] == 2
    assert resumed["start_epoch"] == 1
    assert resumed["completed_epochs"] == 2
    assert resumed["resume_boundary"] == "epoch_only"
    assert resumed["mid_epoch_resume_supported"] is False
    assert uninterrupted["best_validation_loss"] == resumed[
        "best_validation_loss"
    ]
    assert (
        uninterrupted_dir / "metrics.jsonl"
    ).read_bytes() == (resumed_dir / "metrics.jsonl").read_bytes()

    uninterrupted_state = torch.load(
        uninterrupted_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        resumed_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "next_epoch",
        "best_validation_loss",
        "epoch_journal",
        "rng_state",
    ):
        _assert_state_equal(
            uninterrupted_state[key],
            resumed_state[key],
        )

    rows = [
        json.loads(line)
        for line in (resumed_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["epoch"] for row in rows] == [0, 1]
    assert all("stage_timing" not in row for row in rows)
    assert (resumed_dir / "epoch_performance.jsonl").is_file()
    validation_fingerprints = [
        row["validation"]["masking"]["plan_fingerprints"]
        for row in rows
    ]
    assert validation_fingerprints[1] == validation_fingerprints[0]
    assert validation_fingerprints[0]


def test_rejected_resume_restores_entry_rng_state(
    tmp_path: Path,
) -> None:
    random.seed(107)
    torch.manual_seed(107)
    before = capture_rng_state()
    config = _config(tmp_path / "rejected-resume", "pretrain")
    config.experiment.resume_from = str(
        tmp_path / "missing-checkpoint.pt"
    )

    with pytest.raises(
        SSLTrainingError,
        match="artifact_unreadable",
    ):
        run_ssl_training(config)

    _assert_state_equal(capture_rng_state(), before)


def test_one_batch_report_has_trajectory_reload_transfer_and_cpu_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "one-batch"
    config = _config(output, "one_batch", dropout=0.2)
    config.model.hidden_dim = 128
    config.model.local_gnn_layers = 3
    config.model.transformer_layers = 2
    config.model.attention_heads = 4
    config.model.ffn_multiplier = 4
    config.model.dropout = 0.1
    config.ssl.decoder_hidden_dim = 128
    config.ssl.projector_hidden_dim = 128
    config.ssl.decoder_views = 3
    config.ssl.decoder_remask_prob = 0.20
    config.ssl.mask_rate = 0.30
    config.experiment.steps = 40
    config.optimizer.learning_rate = None
    report = run_ssl_training(config)

    assert report["evidence_kind"] == "bounded_phase7a_ssl_plumbing"
    assert report["sample_count"] == 3
    assert report["node_count"] == 114
    assert report["edge_count"] == 740
    assert report["steps"] == 40
    assert report["learning_rate_used"] == pytest.approx(3e-4)
    assert report["trajectory_measurement_mode"] == "eval_no_grad"
    assert report["final"]["total_ssl_loss"] < report["initial"][
        "total_ssl_loss"
    ]
    for component in (
        "note_reconstruction",
        "bar_latent",
        "song_latent",
    ):
        assert report["final"][component]["mean"] < report["initial"][
            component
        ]["mean"]
        assert report["initial"][component]["denominator"] > 0
    assert report["initial"]["masking"]["primary_masked_count"] > 0
    assert report["initial"]["masking"]["primary_masked_count"] == 13
    assert report["initial"]["masking"]["collateral_note_count"] == 35
    assert report["initial"]["masking"]["collateral_track_count"] == 7
    assert report["initial"]["masking"]["realized_mask_rate"] == (
        pytest.approx(13 / 48)
    )
    assert (
        report["initial"]["masking"]["requested_mask_rate"]
        == config.ssl.mask_rate
    )
    assert len(report["initial"]["decoder_view_losses"]) == (
        config.ssl.decoder_views
    )
    for view in report["initial"]["decoder_view_losses"]:
        assert view["denominator"] > 0
        assert view["stable_seeds"]
        assert view["plan_fingerprints"]
    for level in ("note", "bar", "song"):
        diagnostics = report["initial"]["anti_collapse"][level]
        assert diagnostics["row_count"] > 0
        assert diagnostics["pairwise_policy"] == (
            "exact_linear_normalized_sum"
        )
        assert report["initial"]["non_collapse_acceptance"]["levels"][
            level
        ]["passed"] is True
    assert report["initial"]["non_collapse_acceptance"]["passed"] is True
    assert all(
        value["target_zero_norm_count"] == 0
        and value["prediction_zero_norm_count"] == 0
        for value in report["final"]["anti_collapse"].values()
    )
    assert all(report["deterministic_repeat"].values())
    leakage = report["no_leakage_mutation_evidence"]
    assert leakage["applicable"] is True
    assert leakage["fixed_mask_plan"] is True
    assert leakage["raw_graph_stores_bit_exact_after_view"] is True
    assert (
        leakage[
            "online_embeddings_bit_exact_after_masked_mutation"
        ]
        is True
    )
    assert (
        leakage[
            "online_predictions_bit_exact_after_masked_mutation"
        ]
        is True
    )
    assert leakage["full_view_target_changed"] is True
    assert leakage["reconstruction_loss_changed"] is True
    assert leakage["fixed_prepared_binding_fingerprint"] is True
    assert leakage["mutation_contract_version"] == "1.0.0"
    assert leakage["mutation_policy"] == "midi_axis_reflection_v1"
    assert leakage["mutation_policy_fingerprint"] == (
        "55c9c82b10153c21d158fb3287c3c01deea10b2a427b08d1266e1c89cdc32227"
    )
    assert leakage["runtime_source_binding"]["passed"] is True
    assert all(
        row["passed"] is True
        for row in leakage["runtime_source_binding"]["per_sample"]
    )
    assert leakage["raw_graph_store_immutability"] == {
        "original_cpu": True,
        "original_device": True,
        "mutated_cpu": True,
        "mutated_device": True,
    }
    assert all(
        mutation["mutation_instance_fingerprint"]
        and mutation["mask_plan_fingerprint"]
        and mutation["selected_note_ids"]
        and len(mutation["source_pitches"])
        == len(mutation["mutated_pitches"])
        for mutation in leakage["coherent_mutations"]
    )
    assert leakage["metrics_finite"] is True
    assert leakage["positive_margin"] is True
    assert leakage["correct_minus_mutated_margin"] > (
        leakage["positive_margin_floor"]
    )
    assert leakage["target_to_mutated_target_mean_l2_distance"] > 0
    assert leakage["passed"] is True
    assert report["checkpoint_reload"] == {
        "next_epoch": 0,
        "bit_exact": True,
    }
    assert report["encoder_transfer"]["loaded_parameter_count"] > 0
    assert report["encoder_transfer"]["untouched_parameter_count"] > 0
    assert report["encoder_transfer"]["supervised_heads_unchanged"] is True
    for group in (
        "online_local_encoder",
        "hierarchy_pooling",
        "transformer",
        "fusion",
        "decoder",
        "bar_projector",
        "bar_predictor",
        "song_projector",
        "song_predictor",
    ):
        evidence = report["gradient_coverage"][group]
        assert evidence["parameter_count"] > 0
        assert evidence["finite_gradient_count"] > 0
        assert evidence["nonzero_gradient_count"] > 0
    assert report["gradient_coverage"]["unused_supervised_heads"][
        "with_gradient_count"
    ] == 0
    assert report["device"]["resolved_device"] == "cpu"
    assert report["device"]["deterministic_algorithms"] is True
    assert report["device"]["peak_allocated_bytes"] is None
    assert report["device"]["peak_reserved_bytes"] is None
    assert report["amp_enabled"] is False
    assert report["scaler_enabled"] is False
    assert report["stage_timing"]["checkpoint_binding_participation"] is False
    assert report["stage_timing"]["mask_plan_preparation_seconds"] > 0
    assert (
        report["prepared_mask_binding"]["fingerprint"]
        == leakage["prepared_mask_binding_fingerprint"]
    )
    assert "retained_memory_counters" not in report
    assert report["data_source_kind"] == "bounded"
    assert report["production_cache_data_used"] is False
    assert report["run_scope"] == "one_batch_plumbing"
    assert report["optimization_step_count"] == report["steps"]
    assert report["production_ssl_training_performed"] is False
    assert report["full_corpus_ssl_training_performed"] is False
    assert report["full_corpus_ssl_training_unavailable_reason"] is None
    assert (output / "one_batch_report.json").is_file()
    assert (output / "one_batch.pt").is_file()


def test_training_scope_evidence_distinguishes_data_use_and_coverage() -> None:
    production_smoke = _training_scope_evidence(
        data_source_kind="mixed",
        run_scope="one_batch_plumbing",
        optimization_step_count=1,
    )
    assert production_smoke["production_cache_data_used"] is True
    assert production_smoke["production_ssl_training_performed"] is True
    assert production_smoke["full_corpus_ssl_training_performed"] is False

    production_epochs = _training_scope_evidence(
        data_source_kind="mixed",
        run_scope="epoch_pretraining",
        optimization_step_count=2,
    )
    assert production_epochs["production_ssl_training_performed"] is True
    assert production_epochs["full_corpus_ssl_training_performed"] is None
    assert (
        production_epochs["full_corpus_ssl_training_unavailable_reason"]
        == "full_corpus_identity_coverage_not_tracked"
    )


def test_nonfinite_total_loss_fails_before_optimizer_mutation() -> None:
    class NonfiniteModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *_args, **_kwargs):
            return SimpleNamespace(
                objective=SimpleNamespace(
                    total_loss=self.weight * torch.tensor(float("nan"))
                )
            )

    model = NonfiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = model.weight.detach().clone()

    with pytest.raises(
        SSLTrainingError,
        match="ssl.training.nonfinite_total_loss",
    ):
        _optimize_batch(
            model,
            object(),
            optimizer,
            scaler,
            {
                "seed": 1,
                "device": {"amp": False},
                "optimizer": {"gradient_clip_norm": 1.0},
            },
            torch.device("cpu"),
            epoch=0,
            collect_gradient_evidence=False,
        )

    assert torch.equal(model.weight.detach(), before)
    assert model.weight.grad is None


def test_zero_mask_one_batch_report_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "zero-mask", "one_batch")
    config.experiment.steps = 1
    config.ssl.mask_rate = 0.0

    report = run_ssl_training(config)

    for stage in ("initial", "final"):
        assert report[stage]["total_ssl_loss"] is None
        assert (
            report[stage]["total_unavailable_reason"]
            == "required_component_unavailable"
        )
        assert report[stage]["note_reconstruction"]["mean"] is None
        assert (
            report[stage]["note_reconstruction"][
                "unavailable_reason"
            ]
            == "no_eligible_rows"
        )
        assert report[stage]["masking"]["primary_masked_count"] == 0
        assert report[stage]["masking"]["realized_mask_rate"] == 0.0
    assert all(report["deterministic_repeat"].values())
    leakage = report["no_leakage_mutation_evidence"]
    assert leakage["applicable"] is False
    assert leakage["unavailable_reason"] == "no_masked_rows"
    assert leakage["fixed_mask_plan"] is True
    assert leakage["raw_graph_stores_bit_exact_after_view"] is True
    assert leakage["passed"] is None
    assert leakage["correct_minus_mutated_margin"] is None
    assert report["checkpoint_reload"]["bit_exact"] is True


def test_epoch_metrics_recompose_row_weighted_objective_and_unavailable_state(
) -> None:
    accumulator = _Accumulator(
        1,
        note_weight=2.0,
        bar_weight=0.5,
        song_weight=0.25,
    )
    accumulator.note_sum = 4.0
    accumulator.note_count = 4
    accumulator.note_zero_norm_count = 1
    accumulator.bar_sum = 9.0
    accumulator.bar_count = 3
    accumulator.bar_zero_norm_count = 2
    accumulator.song_sum = 8.0
    accumulator.song_count = 2
    accumulator.view_sum[0] = 4.0
    accumulator.view_count[0] = 4
    accumulator.view_zero_norm_count[0] = 1
    for diagnostics in accumulator.diagnostics.values():
        diagnostics.update(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        )

    metric = accumulator.finalize()

    assert metric["total_ssl_loss"] == pytest.approx(
        2.0 * (4.0 / 4) + 0.5 * (9.0 / 3) + 0.25 * (8.0 / 2)
    )
    assert metric["total_unavailable_reason"] is None
    assert metric["unavailable_components"] == []
    assert metric["note_reconstruction"]["zero_norm_count"] == 1
    assert metric["bar_latent"]["zero_norm_count"] == 2
    assert metric["song_latent"]["zero_norm_count"] == 0
    assert metric["decoder_view_losses"][0]["zero_norm_count"] == 1

    unavailable = _Accumulator(
        1,
        note_weight=1.0,
        bar_weight=1.0,
        song_weight=1.0,
    )
    unavailable.bar_sum = 1.0
    unavailable.bar_count = 1
    unavailable.song_sum = 1.0
    unavailable.song_count = 1
    for diagnostics in unavailable.diagnostics.values():
        diagnostics.update(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        )
    missing = unavailable.finalize()
    assert missing["total_ssl_loss"] is None
    assert (
        missing["total_unavailable_reason"]
        == "required_component_unavailable"
    )
    assert missing["unavailable_components"] == [
        "note_reconstruction"
    ]
