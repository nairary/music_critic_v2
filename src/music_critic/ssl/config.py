"""Hydra structured configuration for deterministic Phase 7A SSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from music_critic.training.config import (
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    register_training_configs,
)


@dataclass
class Phase8BObjectiveModeConfig:
    """Hydra-facing fixed Phase 8B.1 objective registry weights."""

    contract_version: str = "1.0.0"
    mode: str = "phase7a_control"
    phase7a_note_reconstruction: float = 1.0
    phase7a_bar_latent: float = 1.0
    phase7a_song_latent: float = 1.0
    onset_latent: float = 0.0
    beat_latent: float = 0.0
    hierarchy_bar_latent: float = 0.0
    track_latent: float = 0.0


@dataclass
class SSLObjectiveConfig:
    mask_rate: float = 0.30
    decoder_views: int = 3
    decoder_remask_prob: float = 0.20
    note_weight: float = 1.0
    bar_weight: float = 1.0
    song_weight: float = 1.0
    epsilon: float = 1e-8
    projector_hidden_dim: int = 128
    decoder_hidden_dim: int = 128


@dataclass
class SSLTrainingConfig:
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
    output_dir: str = "outputs/phase7a"
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    experiment: ExperimentConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    device: DeviceConfig = MISSING
    ssl: SSLObjectiveConfig = field(default_factory=SSLObjectiveConfig)
    phase8b_objective: Phase8BObjectiveModeConfig | None = None


_REGISTERED = False


def register_ssl_configs() -> None:
    """Register only the Phase 7A root and its one new global preset."""

    # Hydra's ConfigStore is process-global. Evaluation uses group names such
    # as ``data=bounded`` with a different schema, so refresh the Phase 6C
    # groups on every SSL composition even after the SSL root was registered.
    register_training_configs()
    global _REGISTERED
    if _REGISTERED:
        return
    store = ConfigStore.instance()
    store.store(name="ssl_training", node=SSLTrainingConfig)
    store.store(
        group="experiment",
        name="pretrain",
        node=ExperimentConfig(
            name="pretrain",
            preset="pretrain",
            steps=1,
            epochs=20,
            checkpoint_interval=1,
            validation_interval=1,
            default_learning_rate=3e-4,
            default_objective="masked_graph_ssl",
            default_harmonic_weight=0.0,
            default_reconstruction_weight=0.0,
            collect_gradient_evidence=False,
        ),
    )
    phase8b_modes = {
        "phase7a_control": Phase8BObjectiveModeConfig(),
        "onset_only": Phase8BObjectiveModeConfig(
            mode="onset_only",
            phase7a_note_reconstruction=0.0,
            phase7a_bar_latent=0.0,
            phase7a_song_latent=0.0,
            onset_latent=1.0,
        ),
        "beat_only": Phase8BObjectiveModeConfig(
            mode="beat_only",
            phase7a_note_reconstruction=0.0,
            phase7a_bar_latent=0.0,
            phase7a_song_latent=0.0,
            beat_latent=1.0,
        ),
        "bar_only": Phase8BObjectiveModeConfig(
            mode="bar_only",
            phase7a_note_reconstruction=0.0,
            phase7a_bar_latent=0.0,
            phase7a_song_latent=0.0,
            hierarchy_bar_latent=1.0,
        ),
        "track_only": Phase8BObjectiveModeConfig(
            mode="track_only",
            phase7a_note_reconstruction=0.0,
            phase7a_bar_latent=0.0,
            phase7a_song_latent=0.0,
            track_latent=1.0,
        ),
        "multilevel_equal_weight": Phase8BObjectiveModeConfig(
            mode="multilevel_equal_weight",
            phase7a_note_reconstruction=0.0,
            phase7a_bar_latent=0.0,
            phase7a_song_latent=0.0,
            onset_latent=1.0,
            beat_latent=1.0,
            hierarchy_bar_latent=1.0,
            track_latent=1.0,
        ),
    }
    for name, node in phase8b_modes.items():
        store.store(group="phase8b_objective", name=name, node=node)
    _REGISTERED = True


register_ssl_configs()


__all__ = [
    "Phase8BObjectiveModeConfig",
    "SSLObjectiveConfig",
    "SSLTrainingConfig",
    "register_ssl_configs",
]
