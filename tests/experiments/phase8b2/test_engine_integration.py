from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from hydra import compose, initialize
import pytest
import torch

from music_critic.experiments.phase8b2.config import Phase8B2Config
from music_critic.experiments.phase8b2.accounting import (
    compute_accounting_from_ssl_report,
)
from music_critic.experiments.phase8b2.runner import (
    build_experiment_plan,
    official_downstream_overrides,
    official_evaluation_overrides,
    official_ssl_cell_overrides,
)
from music_critic.evaluation.config import register_evaluation_configs
from music_critic.evaluation.engine import _plain_config as evaluation_plain_config
from music_critic.experiments.phase8b2.artifacts import file_sha256
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.data import SSLDataRuntime, strip_multisource_batch
from music_critic.ssl.engine import _plain_config, run_ssl_training
from music_critic.ssl import phase8b_engine
from music_critic.ssl.phase8b_engine import (
    _encoder_state_fingerprint,
    _prepare,
)
from music_critic.ssl.transfer import save_pretrained_encoder_export
from music_critic.training.config import register_training_configs
from music_critic.training.config import DataConfig
from music_critic.training.data import build_data_runtime
from music_critic.training.engine import run_training
from music_critic.training.engine import _plain_config as training_plain_config


def _config(plan: dict[str, object], cell: dict[str, object], output: Path):
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="ssl_training",
            overrides=official_ssl_cell_overrides(
                plan, cell["cell_id"], str(output)
            ),
        )


def test_non_default_protocol_fields_reach_all_resolved_runtimes(
    tmp_path: Path,
) -> None:
    requested = Phase8B2Config()
    requested.data.mixture_weights = {
        "hooktheory": 30.0,
        "pop909_cl": 1.0,
    }
    requested.data.workers = 2
    requested.model.residual = False
    requested.scheduler.name = "cosine"
    requested.scheduler.minimum_learning_rate = 0.001
    requested.device.amp_dtype = "bfloat16"
    requested.downstream_task_ids = [
        "theory.local_key.tonic_pc",
        "pop909_cl.chord.root",
    ]
    plan = build_experiment_plan(requested)
    protocol = plan["protocol"]

    ssl_cell = plan["ssl_cells"][0]
    ssl_config = _config(plan, ssl_cell, tmp_path / "ssl")
    resolved_ssl = _plain_config(ssl_config)
    assert resolved_ssl["data"]["mixture_weights"] == {
        "hooktheory": 30.0,
        "pop909_cl": 1.0,
    }
    assert resolved_ssl["data"]["workers"] == 2
    assert resolved_ssl["model"]["residual"] is False
    assert resolved_ssl["scheduler"] == {
        "name": "cosine",
        "minimum_learning_rate": 0.001,
    }
    assert resolved_ssl["device"]["amp_dtype"] == "bfloat16"

    downstream_cell = next(
        row
        for row in plan["downstream_cells"]
        if row["transfer_mode"] == "supervised_scratch"
    )
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        downstream_config = compose(
            config_name="training",
            overrides=official_downstream_overrides(
                plan,
                downstream_cell["cell_id"],
                str(tmp_path / "downstream"),
            ),
        )
    resolved_downstream = training_plain_config(downstream_config)
    assert resolved_downstream["data"]["workers"] == 2
    assert resolved_downstream["model"]["residual"] is False
    assert resolved_downstream["scheduler"]["minimum_learning_rate"] == 0.001
    assert resolved_downstream["device"]["amp_dtype"] == "bfloat16"
    assert resolved_downstream["downstream_task_ids"] == [
        "pop909_cl.chord.root",
        "theory.local_key.tonic_pc",
    ]

    evaluation_cell = next(
        row
        for row in plan["evaluation_cells"]
        if row["downstream_cell_id"] == downstream_cell["cell_id"]
    )
    register_evaluation_configs()
    with initialize(version_base="1.3", config_path=None):
        evaluation_config = compose(
            config_name="evaluation",
            overrides=official_evaluation_overrides(
                plan,
                checkpoint=str(tmp_path / "candidate.pt"),
                output_directory=str(tmp_path / "evaluation"),
                cell_id=evaluation_cell["cell_id"],
            ),
        )
    resolved_evaluation = evaluation_plain_config(evaluation_config)
    assert resolved_evaluation["seed"] == evaluation_cell["evaluation_seed"]
    assert resolved_evaluation["data"]["workers"] == 2
    assert resolved_evaluation["data"]["max_evaluation_samples"] == 0
    assert resolved_evaluation["data"]["validation_seed"] == 20260815
    assert resolved_evaluation["device"]["amp_dtype"] == "bfloat16"
    assert resolved_evaluation["downstream_task_ids"] == [
        "pop909_cl.chord.root",
        "theory.local_key.tonic_pc",
    ]
    assert protocol["data"]["actual_train_size"] > 0
    assert protocol["data"]["actual_validation_size"] > 0


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_official_engine_uses_paired_encoder_initialization(
    tmp_path: Path,
) -> None:
    plan = build_experiment_plan(Phase8B2Config())
    cells = [
        row
        for row in plan["ssl_cells"]
        if row["seed"] == 17
    ]
    fingerprints = []
    sample_schedules = []
    validation_memberships = []
    for index, cell in enumerate(cells):
        prepared = _prepare(
            _plain_config(_config(plan, cell, tmp_path / f"cell-{index}"))
        )
        model = prepared[3]
        runtime = prepared[2]
        fingerprints.append(_encoder_state_fingerprint(model))
        sample_schedules.append(
            tuple(
                identity
                for epoch in range(2)
                for batch in runtime.train_loader(epoch)
                for identity in zip(
                    batch.dataset_ids, batch.piece_ids, strict=True
                )
            )
        )
        validation_memberships.append(
            runtime.validation_membership.membership_fingerprint
        )
    assert len(fingerprints) == 4
    assert len(set(fingerprints)) == 1
    assert len(set(sample_schedules)) == 1
    assert len(set(validation_memberships)) == 1


@pytest.mark.parametrize(
    ("variant_id", "views_per_update", "forwards_per_view"),
    (("phase7a_control", 6, 2), ("onset_latent", 4, 3)),
)
def test_official_engine_records_exact_matched_forward_budget(
    tmp_path: Path,
    variant_id: str,
    views_per_update: int,
    forwards_per_view: int,
) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plan = build_experiment_plan(Phase8B2Config())
        cell = next(
            row
            for row in plan["ssl_cells"]
            if row["seed"] == 17 and row["variant_id"] == variant_id
        )
        report = run_ssl_training(
            _config(plan, cell, tmp_path / f"matched-{variant_id}")
        )
    finally:
        torch.set_num_threads(previous)
    assert report["phase8b2_started"]
    assert report["compute_matched_comparison"]
    assert report["accounting"]["optimizer_step_count"] == 2
    assert report["accounting"]["encoder_forward_count"] == 24
    assert report["phase8b2_schedule"][
        "encoder_forwards_per_policy_view"
    ] == forwards_per_view
    assert len(report["phase8b2_schedule"]["policy_views"]) == (
        views_per_update
    )
    assert report["accounting"]["sample_count"] == 4
    assert report["phase8b2_schedule"]["loss_renormalization"] == (
        "unchanged_family_global_aggregation"
    )
    assert report["encoder_state_fingerprints"]["initial"]
    assert report["encoder_state_fingerprints"]["final"]
    exact = compute_accounting_from_ssl_report(report)
    assert exact.encoder_forwards == 24
    assert exact.policy_views == 2 * views_per_update


def test_official_downstream_engine_freezes_transferred_encoder(
    tmp_path: Path,
) -> None:
    plan = build_experiment_plan(Phase8B2Config())
    cell = next(
        row
        for row in plan["downstream_cells"]
        if row["seed"] == 17
        and row["variant_id"] == "onset_latent"
        and row["transfer_mode"] == "frozen_probe"
    )
    source = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )
    export_path = tmp_path / "encoder.pt"
    save_pretrained_encoder_export(export_path, source)
    overrides = official_downstream_overrides(
        plan,
        cell["cell_id"],
        str(tmp_path / "downstream"),
        encoder_export_path=str(export_path),
        encoder_export_sha256=file_sha256(export_path),
        source_ssl_checkpoint_sha256="a" * 64,
    )
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(config_name="training", overrides=overrides)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_training(config)
    finally:
        torch.set_num_threads(previous)
    assert report["phase8b2_transfer"]["transfer_mode"] == "frozen_probe"
    assert report["frozen_encoder_final"]["bit_exact"]
    assert set(
        report["phase8b2_transfer"]["loaded_trainable_parameter_names"]
    ).isdisjoint(
        report["phase8b2_transfer"]["optimizer_parameter_names"]
    )


def test_comparison_downstream_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    plan = build_experiment_plan(Phase8B2Config())
    cell = next(
        row
        for row in plan["downstream_cells"]
        if row["seed"] == 17
        and row["variant_id"] == "onset_latent"
        and row["transfer_mode"] == "frozen_probe"
    )
    torch.manual_seed(5)
    source = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )
    export_path = tmp_path / "resume-encoder.pt"
    save_pretrained_encoder_export(export_path, source)

    def config(output: Path):
        overrides = official_downstream_overrides(
            plan,
            cell["cell_id"],
            str(output),
            encoder_export_path=str(export_path),
            encoder_export_sha256=file_sha256(export_path),
            source_ssl_checkpoint_sha256="a" * 64,
        )
        register_training_configs()
        with initialize(version_base="1.3", config_path=None):
            return compose(config_name="training", overrides=overrides)

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        uninterrupted_dir = tmp_path / "downstream-uninterrupted"
        uninterrupted = run_training(config(uninterrupted_dir))
        resumed_dir = tmp_path / "downstream-resumed"
        first = run_training(config(resumed_dir), stop_after_epoch=1)
        assert first["completed_epochs"] == 1
        resume_config = config(resumed_dir)
        resume_config.experiment.resume_from = str(resumed_dir / "last.pt")
        resumed = run_training(resume_config)
        uninterrupted_state = torch.load(
            uninterrupted_dir / "last.pt",
            map_location="cpu",
            weights_only=True,
        )
        resumed_state = torch.load(
            resumed_dir / "last.pt", map_location="cpu", weights_only=True
        )
    finally:
        torch.set_num_threads(previous)
    assert uninterrupted["frozen_encoder_final"] == resumed[
        "frozen_encoder_final"
    ]
    assert uninterrupted["observed_downstream_schedule_fingerprint"] == (
        resumed["observed_downstream_schedule_fingerprint"]
    )
    assert (uninterrupted_dir / "metrics.jsonl").read_bytes() == (
        resumed_dir / "metrics.jsonl"
    ).read_bytes()
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "next_epoch",
        "best_validation_loss",
        "committed_metric_rows",
        "rng_state",
    ):
        _assert_tree_equal(uninterrupted_state[key], resumed_state[key])


def test_comparison_schedule_epoch_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    plan = build_experiment_plan(Phase8B2Config())
    cell = next(
        row
        for row in plan["ssl_cells"]
        if row["seed"] == 17 and row["variant_id"] == "onset_latent"
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        uninterrupted_dir = tmp_path / "uninterrupted"
        uninterrupted = run_ssl_training(
            _config(plan, cell, uninterrupted_dir)
        )
        uninterrupted_checkpoint = deepcopy(
            torch.load(
                uninterrupted_dir / "last.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        resumed_dir = tmp_path / "resumed"
        interrupted_config = _config(plan, cell, resumed_dir)
        first = run_ssl_training(interrupted_config, stop_after_epoch=1)
        assert first["completed_epochs"] == 1
        resume_config = _config(plan, cell, resumed_dir)
        resume_config.experiment.resume_from = str(resumed_dir / "last.pt")
        resumed = run_ssl_training(resume_config)
        resumed_checkpoint = torch.load(
            resumed_dir / "last.pt", map_location="cpu", weights_only=True
        )
    finally:
        torch.set_num_threads(previous)
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "next_epoch",
        "best_validation_loss",
        "epoch_journal",
    ):
        _assert_tree_equal(
            uninterrupted_checkpoint[key], resumed_checkpoint[key]
        )
    assert uninterrupted["accounting"] == resumed["accounting"]
    assert (uninterrupted_dir / "metrics.jsonl").read_bytes() == (
        resumed_dir / "metrics.jsonl"
    ).read_bytes()


def _mutate_target_sidecars(batch: object, mutation_kind: str) -> None:
    if mutation_kind == "removed":
        object.__setattr__(batch, "target_batches", ())
        object.__setattr__(batch, "diagnostics_cpu", ())
        return
    targets = list(batch.target_batches)
    if mutation_kind == "replaced":
        targets.reverse()
        object.__setattr__(batch, "target_batches", tuple(targets))
    for target in targets:
        for name in (
            "values",
            "availability_mask",
            "entity_indices",
            "entity_index_mask",
            "entity_node_type_codes",
            "sample_indices",
            "confidence",
            "confidence_mask",
        ):
            value = getattr(target, name)
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            if value.dtype == torch.bool:
                value.logical_not_()
            elif value.is_floating_point():
                value.add_(17.0)
            else:
                value.add_(17)
        object.__setattr__(target, "provenance_cpu", ())


def test_actual_target_mutations_leave_official_comparison_run_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = build_experiment_plan(Phase8B2Config())
    cell = next(
        row
        for row in plan["ssl_cells"]
        if row["seed"] == 17 and row["variant_id"] == "phase7a_control"
    )
    data_config = DataConfig(
        batch_size=2, epoch_size=2, validation_epoch_size=2
    )
    supervised_batch = build_data_runtime(
        data_config, seed=17
    ).first_train_batch
    reference_runtime = phase8b_engine.build_ssl_data_runtime(
        data_config, seed=17
    )
    outcomes = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for mutation_kind in ("original", "changed", "removed", "replaced"):
            candidate = deepcopy(supervised_batch)
            if mutation_kind != "original":
                _mutate_target_sidecars(candidate, mutation_kind)
            raw_batch = strip_multisource_batch(candidate)
            runtime = SSLDataRuntime(
                first_train_batch=raw_batch,
                train_loader=lambda _epoch, value=raw_batch: (value,),
                validation_loader=lambda value=raw_batch: (value,),
                validation_membership=reference_runtime.validation_membership,
                fingerprints=reference_runtime.fingerprints,
                mixture_statistics=reference_runtime.mixture_statistics,
            )
            monkeypatch.setattr(
                phase8b_engine,
                "build_ssl_data_runtime",
                lambda *_args, value=runtime, **_kwargs: value,
            )
            output = tmp_path / mutation_kind
            report = run_ssl_training(_config(plan, cell, output))
            checkpoint = torch.load(
                output / "last.pt", map_location="cpu", weights_only=True
            )
            outcomes.append(
                {
                    "plans": [
                        row["train"]["mask_plan_fingerprints"]
                        for row in checkpoint["epoch_journal"]
                    ],
                    "sample_schedule": report[
                        "observed_ssl_sample_schedule_fingerprint"
                    ],
                    "predictions": report[
                        "observed_prediction_fingerprint"
                    ],
                    "losses": [
                        row["train"]["total_ssl_loss"]
                        for row in checkpoint["epoch_journal"]
                    ],
                    "gradients": report[
                        "observed_gradient_fingerprint"
                    ],
                    "checkpoint": checkpoint,
                    "transferred_encoder": report[
                        "encoder_state_fingerprints"
                    ]["final"],
                }
            )
    finally:
        torch.set_num_threads(previous)
    reference = outcomes[0]
    for outcome in outcomes[1:]:
        for name in (
            "plans",
            "sample_schedule",
            "predictions",
            "losses",
            "gradients",
            "transferred_encoder",
        ):
            assert outcome[name] == reference[name]
        for key in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_state",
            "next_epoch",
            "best_validation_loss",
            "epoch_journal",
        ):
            _assert_tree_equal(
                outcome["checkpoint"][key], reference["checkpoint"][key]
            )
