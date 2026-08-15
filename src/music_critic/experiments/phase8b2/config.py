"""Hydra structured presets for Phase 8B.2A orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from music_critic.experiments.phase8b2.contracts import (
    DOWNSTREAM_TASK_IDS,
    SSL_VARIANTS,
    fingerprint,
)


@dataclass
class ComparisonPresetConfig:
    name: str = "bounded_acceptance"
    comparison_mode: str = "encoder_forward_matched"
    variants: list[str] = field(
        default_factory=lambda: [
            "phase7a_control",
            "phase8a_mask_only",
            "onset_latent",
            "multilevel_equal",
        ]
    )
    seeds: list[int] = field(default_factory=lambda: [17, 29])
    transfer_modes: list[str] = field(
        default_factory=lambda: ["frozen_probe", "full_finetune", "supervised_scratch"]
    )
    architecture: str = "hierarchical"
    ssl_optimizer_steps: int = 2
    downstream_optimizer_steps: int = 2
    matched_encoder_forwards_per_update: int = 12
    bootstrap_replicates: int = 200
    minimum_production_seeds: int = 3


@dataclass
class ComparisonDataConfig:
    index_paths: list[str] = field(default_factory=list)
    cache_roots: list[str] = field(default_factory=list)
    split_manifest: str = ""
    index_fingerprints: dict[str, str] = field(default_factory=dict)
    cache_fingerprints: dict[str, str] = field(default_factory=dict)
    split_manifest_fingerprint: str = field(
        default_factory=lambda: fingerprint(
            {"bounded_fixture": "split_manifest"}
        )
    )
    train_membership_fingerprint: str = field(
        default_factory=lambda: fingerprint(
            {"bounded_fixture": "train_membership"}
        )
    )
    validation_membership_fingerprint: str = field(
        default_factory=lambda: fingerprint(
            {"bounded_fixture": "validation_membership"}
        )
    )
    test_membership_fingerprint: str = field(
        default_factory=lambda: fingerprint(
            {"bounded_fixture": "test_membership"}
        )
    )
    mixture_weights: dict[str, float] = field(
        default_factory=lambda: {"hooktheory": 1.0, "pop909_cl": 1.0}
    )
    batch_size: int = 2
    workers: int = 0


@dataclass
class ComparisonModelConfig:
    name: str = "hierarchical"
    hidden_dim: int = 8
    local_gnn_layers: int = 1
    transformer_layers: int = 1
    attention_heads: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.0
    residual: bool = True


@dataclass
class ComparisonSSLConfig:
    mask_rate: float = 0.5
    decoder_views: int = 1
    decoder_remask_prob: float = 0.0
    note_weight: float = 1.0
    bar_weight: float = 1.0
    song_weight: float = 1.0
    epsilon: float = 1e-8
    projector_hidden_dim: int = 8
    decoder_hidden_dim: int = 8


@dataclass
class ComparisonOptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 0.02
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0


@dataclass
class ComparisonSchedulerConfig:
    name: str = "none"
    minimum_learning_rate: float = 0.0


@dataclass
class ComparisonDeviceConfig:
    name: str = "cpu"
    amp: bool = False
    amp_dtype: str = "float16"
    non_blocking: bool = False


@dataclass
class Phase8B2Config:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"comparison": "bounded_acceptance"},
            "_self_",
        ]
    )
    action: str = "plan"
    output_root: str = "outputs/phase8b2"
    acknowledge_test_evaluation: bool = False
    selected_checkpoint: str = ""
    comparison: ComparisonPresetConfig = field(
        default_factory=ComparisonPresetConfig
    )
    data: ComparisonDataConfig = field(default_factory=ComparisonDataConfig)
    model: ComparisonModelConfig = field(default_factory=ComparisonModelConfig)
    ssl: ComparisonSSLConfig = field(default_factory=ComparisonSSLConfig)
    optimizer: ComparisonOptimizerConfig = field(
        default_factory=ComparisonOptimizerConfig
    )
    scheduler: ComparisonSchedulerConfig = field(
        default_factory=ComparisonSchedulerConfig
    )
    device: ComparisonDeviceConfig = field(
        default_factory=ComparisonDeviceConfig
    )
    downstream_task_ids: list[str] = field(
        default_factory=lambda: list(DOWNSTREAM_TASK_IDS)
    )


def _preset(
    name: str,
    *,
    mode: str,
    seeds: list[int],
    ssl_steps: int,
    downstream_steps: int,
    bootstrap_replicates: int,
) -> ComparisonPresetConfig:
    return ComparisonPresetConfig(
        name=name,
        comparison_mode=mode,
        variants=list(SSL_VARIANTS),
        seeds=seeds,
        ssl_optimizer_steps=ssl_steps,
        downstream_optimizer_steps=downstream_steps,
        bootstrap_replicates=bootstrap_replicates,
    )


_REGISTERED = False


def register_phase8b2_configs() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    store = ConfigStore.instance()
    store.store(name="phase8b2", node=Phase8B2Config)
    store.store(
        group="comparison",
        name="bounded_acceptance",
        node=ComparisonPresetConfig(),
    )
    store.store(
        group="comparison",
        name="production_pilot",
        node=_preset(
            "production_pilot",
            mode="encoder_forward_matched",
            seeds=[17, 29, 43],
            ssl_steps=1000,
            downstream_steps=1000,
            bootstrap_replicates=2000,
        ),
    )
    store.store(
        group="comparison",
        name="production_paper",
        node=_preset(
            "production_paper",
            mode="encoder_forward_matched",
            seeds=[17, 29, 43, 61, 73],
            ssl_steps=10000,
            downstream_steps=10000,
            bootstrap_replicates=10000,
        ),
    )
    store.store(
        group="comparison",
        name="natural_schedule_diagnostic",
        node=_preset(
            "natural_schedule_diagnostic",
            mode="natural_schedule",
            seeds=[17, 29, 43],
            ssl_steps=1000,
            downstream_steps=1000,
            bootstrap_replicates=2000,
        ),
    )
    _REGISTERED = True


register_phase8b2_configs()


__all__ = [
    "ComparisonDataConfig",
    "ComparisonDeviceConfig",
    "ComparisonModelConfig",
    "ComparisonOptimizerConfig",
    "ComparisonPresetConfig",
    "ComparisonSSLConfig",
    "ComparisonSchedulerConfig",
    "Phase8B2Config",
    "register_phase8b2_configs",
]
