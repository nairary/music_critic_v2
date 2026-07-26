"""Reproducible supervised baseline training harness for Phase 6C."""

from music_critic.training.config import (
    DataConfig,
    DeviceConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    register_training_configs,
)
from music_critic.training.device import (
    DEVICE_TRANSFER_CONTRACT_VERSION,
    DeviceTransferError,
    move_multisource_batch,
    validate_device_batch,
)
from music_critic.training.engine import (
    TRAINING_CHECKPOINT_VERSION,
    TrainingContractError,
    run_training,
)
from music_critic.training.models import build_baseline_model


__all__ = [
    "DEVICE_TRANSFER_CONTRACT_VERSION",
    "TRAINING_CHECKPOINT_VERSION",
    "DataConfig",
    "DeviceConfig",
    "DeviceTransferError",
    "ExperimentConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "TrainingContractError",
    "build_baseline_model",
    "move_multisource_batch",
    "register_training_configs",
    "run_training",
    "validate_device_batch",
]
