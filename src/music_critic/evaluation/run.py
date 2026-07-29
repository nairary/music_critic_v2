"""Hydra CLI for deterministic Phase 6D-A supervised evaluation."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from music_critic.evaluation.config import register_evaluation_configs
from music_critic.evaluation.engine import run_evaluation


register_evaluation_configs()


@hydra.main(version_base="1.3", config_name="evaluation")
def main(config: DictConfig) -> None:
    run_evaluation(config)


if __name__ == "__main__":
    main()
