"""Hydra structured configuration for Phase 6D-A supervised evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EvaluationDataConfig:
    name: str = "bounded"
    index_paths: list[str] = field(default_factory=list)
    cache_roots: list[str] = field(default_factory=list)
    split_manifest: str = ""
    train_split: str = "train"
    batch_size: int = 3
    workers: int = 0
    # Zero means the complete fixed split. Positive values are explicit
    # bounded smoke subsets and remain fingerprinted as such.
    max_train_samples: int = 0
    max_evaluation_samples: int = 0
    validation_seed: int = -1


@dataclass
class EvaluationDeviceConfig:
    name: str = "cpu"
    amp: bool = False
    non_blocking: bool = False
    amp_dtype: str = "float16"


@dataclass
class EvaluationConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"data": "bounded"},
            {"device": "cpu"},
            "_self_",
        ]
    )
    checkpoint: str = ""
    split: str = "validation"
    acknowledge_test_evaluation: bool = False
    output_dir: str = "outputs/phase6d/evaluation"
    seed: int = 42
    overwrite_output: bool = False
    train_priors_path: str = ""
    data: EvaluationDataConfig = MISSING
    device: EvaluationDeviceConfig = MISSING
    downstream_task_ids: list[str] = field(default_factory=list)


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_path: str = "outputs/phase6d/performance_report.json"
    seed: int = 42
    input_mode: str = "synthetic"
    max_batches: int = 2
    dataset_values: list[str] = field(
        default_factory=lambda: ["hooktheory", "pop909_cl", "mixed"]
    )
    model_values: list[str] = field(
        default_factory=lambda: [
            "feature_only",
            "local_gnn",
            "hierarchical",
        ]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4])
    worker_values: list[int] = field(default_factory=lambda: [0, 2])
    hidden_dim: int = 128
    local_gnn_layers: int = 3
    transformer_layers: int = 2
    attention_heads: int = 4
    # Production profiling is opt-in and read-only. Paths must be explicit,
    # absolute, and empty in synthetic mode.
    production_index_paths: list[str] = field(default_factory=list)
    production_cache_roots: list[str] = field(default_factory=list)
    production_split_manifest: str = ""
    production_split: str = "validation"
    production_max_samples_per_dataset: int = 2


def _data(
    name: str,
    *,
    index_paths: list[str],
    cache_roots: list[str],
) -> EvaluationDataConfig:
    return EvaluationDataConfig(
        name=name,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=(
            "" if name == "bounded" else "data/cache/global.split.json"
        ),
    )


def register_evaluation_configs() -> None:
    store = ConfigStore.instance()
    store.store(name="evaluation", node=EvaluationConfig)
    store.store(name="profiler", node=ProfilerConfig)
    store.store(
        group="data",
        name="bounded",
        node=_data("bounded", index_paths=[], cache_roots=[]),
    )
    store.store(
        group="data",
        name="hooktheory",
        node=_data(
            "hooktheory",
            index_paths=["data/cache/hooktheory.index.json"],
            cache_roots=["data/cache/hooktheory"],
        ),
    )
    store.store(
        group="data",
        name="pop909_cl",
        node=_data(
            "pop909_cl",
            index_paths=["data/cache/pop909_cl.index.json"],
            cache_roots=["data/cache/pop909_cl"],
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
        ),
    )
    for name in ("cpu", "cuda", "auto"):
        store.store(
            group="device",
            name=name,
            node=EvaluationDeviceConfig(name=name),
        )


register_evaluation_configs()


__all__ = [
    "EvaluationConfig",
    "EvaluationDataConfig",
    "EvaluationDeviceConfig",
    "ProfilerConfig",
    "register_evaluation_configs",
]
