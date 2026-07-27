"""Hydra entry point for reproducible Phase 6C baseline training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from music_critic.training.config import register_training_configs
from music_critic.training.engine import run_training


register_training_configs()


@hydra.main(version_base="1.3", config_name="training")
def main(config: DictConfig) -> None:
    run_training(config)


if __name__ == "__main__":
    main()
