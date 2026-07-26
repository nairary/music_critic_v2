from __future__ import annotations

from hydra import compose, initialize
import pytest

from music_critic.models import (
    HierarchicalHeterogeneousBaseline,
    LocalHeterogeneousBaseline,
)
from music_critic.training.config import register_training_configs
from music_critic.training.device import move_multisource_batch
from music_critic.training.engine import (
    TrainingContractError,
    _plain_config,
    _resolve_presets,
    _validate_config,
)
from music_critic.training.models import build_baseline_model


def _compose(*overrides: str):
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=list(overrides),
        )


@pytest.mark.parametrize(
    ("group", "expected_type", "variant"),
    (
        ("feature_only", LocalHeterogeneousBaseline, "feature_only"),
        ("local_gnn", LocalHeterogeneousBaseline, "local_gnn"),
        (
            "hierarchical",
            HierarchicalHeterogeneousBaseline,
            "hierarchical",
        ),
    ),
)
def test_model_group_selects_existing_baseline(
    group: str,
    expected_type: type,
    variant: str,
) -> None:
    config = _compose(
        f"model={group}",
        "model.hidden_dim=16",
        "model.dropout=0",
    )
    model = build_baseline_model(config.model)
    assert isinstance(model, expected_type)
    if isinstance(model, LocalHeterogeneousBaseline):
        assert model.config.variant == variant


def test_all_structured_groups_compose_with_fixed_overrides() -> None:
    config = _compose(
        "model=hierarchical",
        "data=mixed",
        "experiment=train",
        "optimizer=adamw",
        "objective=supervised_harmonic",
        "scheduler=cosine",
        "device=auto",
        "seed=73",
        "data.batch_size=7",
        "data.workers=2",
        "data.epoch_size=19",
        "data.mixture_weights.hooktheory=3",
        "optimizer.learning_rate=0.0003",
        "optimizer.weight_decay=0.01",
        "optimizer.gradient_clip_norm=2",
        "experiment.epochs=4",
        "experiment.checkpoint_interval=2",
        "experiment.validation_interval=2",
        "device.amp=false",
        "output_dir=artifacts/phase6c-test",
    )
    assert config.model.name == "hierarchical"
    assert config.data.name == "mixed"
    assert config.experiment.name == "train"
    assert config.optimizer.name == "adamw"
    assert config.objective.name == "supervised_harmonic"
    assert config.scheduler.name == "cosine"
    assert config.device.name == "auto"
    assert config.seed == 73
    assert config.data.batch_size == 7
    assert config.data.workers == 2
    assert config.data.epoch_size == 19
    assert config.data.mixture_weights.hooktheory == 3
    assert config.optimizer.learning_rate == 0.0003
    assert config.optimizer.weight_decay == 0.01
    assert config.optimizer.gradient_clip_norm == 2
    assert config.experiment.epochs == 4
    assert config.experiment.checkpoint_interval == 2
    assert config.experiment.validation_interval == 2
    assert config.output_dir == "artifacts/phase6c-test"


@pytest.mark.parametrize(
    (
        "experiment",
        "experiment_name",
        "learning_rate",
        "objective",
        "harmonic_weight",
        "reconstruction_weight",
    ),
    (
        ("one_batch", "one_batch", 0.02, "one_batch_joint", 1.0, 1.0),
        ("smoke", "smoke", 3e-4, "supervised_harmonic", 1.0, 0.0),
        ("train", "train", 3e-4, "supervised_harmonic", 1.0, 0.0),
        (
            "supervised_baseline",
            "supervised_baseline",
            3e-4,
            "supervised_harmonic",
            1.0,
            0.0,
        ),
        (
            "joint_visible_reconstruction",
            "joint_visible_reconstruction",
            3e-4,
            "joint_visible_reconstruction",
            1.0,
            1.0,
        ),
    ),
)
def test_fixed_experiment_presets_resolve_before_execution(
    experiment: str,
    experiment_name: str,
    learning_rate: float,
    objective: str,
    harmonic_weight: float,
    reconstruction_weight: float,
) -> None:
    config = _plain_config(_compose(f"experiment={experiment}"))
    _resolve_presets(config)
    assert config["experiment"]["name"] == experiment_name
    assert config["optimizer"]["learning_rate"] == learning_rate
    assert config["objective"] == {
        "name": objective,
        "harmonic_weight": harmonic_weight,
        "reconstruction_weight": reconstruction_weight,
        "task_weights": {},
    }


def test_objective_task_weights_enter_model_contract() -> None:
    config = _compose(
        "model=feature_only",
        "objective=supervised_harmonic",
        "+objective.task_weights={theory.chord.extent:2.5}",
        "model.hidden_dim=8",
    )
    model = build_baseline_model(
        config.model,
        task_weights=dict(config.objective.task_weights),
    )
    assert model.config.task_weights == (
        ("theory.chord.extent", 2.5),
    )


@pytest.mark.parametrize(
    "task_id",
    (
        "pop909_cl.chord.no_chord",
        "pop909_cl.chord.boundary",
        "theory.local_key.mode",
    ),
)
def test_objective_cannot_enable_pu_or_open_vocabulary_tasks(
    task_id: str,
) -> None:
    config = _plain_config(_compose("experiment=train"))
    _resolve_presets(config)
    config["objective"]["task_weights"] = {task_id: 1.0}
    with pytest.raises(
        TrainingContractError,
        match="training.config.task_weights_invalid",
    ):
        _validate_config(config)


@pytest.mark.parametrize(
    "group", ("feature_only", "local_gnn", "hierarchical")
)
def test_each_selected_baseline_runs_cpu_forward_backward(
    group: str,
    bounded_batch,
) -> None:
    config = _compose(
        f"model={group}",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=4",
        "model.dropout=0",
    )
    model = build_baseline_model(config.model)
    batch = move_multisource_batch(bounded_batch, "cpu")
    output = model(batch)
    assert output.harmonic_loss.total_loss is not None
    assert output.reconstruction_loss is not None
    total = (
        output.harmonic_loss.total_loss
        + output.reconstruction_loss
    )
    total.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )
