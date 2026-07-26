"""Hydra structured configuration groups for Phase 6C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = "hierarchical"
    hidden_dim: int = 128
    local_gnn_layers: int = 3
    transformer_layers: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    residual: bool = True


@dataclass
class DataConfig:
    name: str = "bounded"
    index_paths: list[str] = field(default_factory=list)
    cache_roots: list[str] = field(default_factory=list)
    split_manifest: str = ""
    train_split: str = "train"
    validation_split: str = "validation"
    batch_size: int = 3
    workers: int = 0
    epoch_size: int = 6
    validation_epoch_size: int = 3
    mixture_weights: dict[str, float] = field(
        default_factory=lambda: {
            "hooktheory": 1.0,
            "pop909_cl": 1.0,
        }
    )


@dataclass
class ExperimentConfig:
    name: str = "one_batch"
    steps: int = 40
    epochs: int = 1
    checkpoint_interval: int = 1
    validation_interval: int = 1
    resume_from: str = ""


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 0.02
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0


@dataclass
class SchedulerConfig:
    name: str = "none"
    minimum_learning_rate: float = 0.0


@dataclass
class DeviceConfig:
    name: str = "cpu"
    amp: bool = False
    non_blocking: bool = False


@dataclass
class TrainingConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"model": "hierarchical"},
            {"data": "bounded"},
            {"experiment": "one_batch"},
            {"optimizer": "adamw"},
            {"scheduler": "none"},
            {"device": "cpu"},
            "_self_",
        ]
    )
    seed: int = 42
    output_dir: str = "outputs/phase6c"
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    experiment: ExperimentConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    device: DeviceConfig = MISSING


def _model(name: str) -> ModelConfig:
    return ModelConfig(name=name)


def _data(
    name: str,
    *,
    index_paths: list[str],
    cache_roots: list[str],
    weights: dict[str, float],
) -> DataConfig:
    return DataConfig(
        name=name,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=(
            "" if name == "bounded" else "data/cache/global.split.json"
        ),
        mixture_weights=weights,
    )


def register_training_configs() -> None:
    """Register every fixed Phase 6C group in Hydra's ConfigStore."""

    store = ConfigStore.instance()
    store.store(name="training", node=TrainingConfig)
    for name in ("feature_only", "local_gnn", "hierarchical"):
        store.store(group="model", name=name, node=_model(name))
    store.store(
        group="data",
        name="bounded",
        node=_data(
            "bounded", index_paths=[], cache_roots=[], weights={
                "hooktheory": 1.0,
                "pop909_cl": 1.0,
            }
        ),
    )
    store.store(
        group="data",
        name="hooktheory",
        node=_data(
            "hooktheory",
            index_paths=["data/cache/hooktheory.index.json"],
            cache_roots=["data/cache/hooktheory"],
            weights={"hooktheory": 1.0},
        ),
    )
    store.store(
        group="data",
        name="pop909_cl",
        node=_data(
            "pop909_cl",
            index_paths=["data/cache/pop909_cl.index.json"],
            cache_roots=["data/cache/pop909_cl"],
            weights={"pop909_cl": 1.0},
        ),
    )
    store.store(
        group="data",
        name="mixed",
        node=_data(
            "mixed",
            index_paths=[
                "data/cache/hooktheory.index.json",
                "data/cache/pop909_cl.index.json",
            ],
            cache_roots=[
                "data/cache/hooktheory",
                "data/cache/pop909_cl",
            ],
            weights={"hooktheory": 1.0, "pop909_cl": 1.0},
        ),
    )
    store.store(
        group="experiment",
        name="one_batch",
        node=ExperimentConfig(
            name="one_batch",
            steps=40,
            epochs=1,
            checkpoint_interval=1,
            validation_interval=1,
        ),
    )
    store.store(
        group="experiment",
        name="smoke",
        node=ExperimentConfig(
            name="smoke",
            steps=1,
            epochs=2,
            checkpoint_interval=1,
            validation_interval=1,
        ),
    )
    store.store(
        group="experiment",
        name="train",
        node=ExperimentConfig(
            name="train",
            steps=1,
            epochs=20,
            checkpoint_interval=1,
            validation_interval=1,
        ),
    )
    store.store(group="optimizer", name="adamw", node=OptimizerConfig())
    store.store(group="scheduler", name="none", node=SchedulerConfig())
    store.store(
        group="scheduler",
        name="cosine",
        node=SchedulerConfig(name="cosine", minimum_learning_rate=0.0),
    )
    for name in ("cpu", "cuda", "auto"):
        store.store(
            group="device",
            name=name,
            node=DeviceConfig(name=name),
        )


register_training_configs()


__all__ = [
    "DataConfig",
    "DeviceConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "register_training_configs",
]
