"""Hydra entry point for the explicit bounded Phase 6D-A profiler."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from music_critic.evaluation.config import register_evaluation_configs
from music_critic.evaluation.profiler import run_profiler


register_evaluation_configs()


@hydra.main(version_base="1.3", config_name="profiler")
def main(config: DictConfig) -> None:
    run_profiler(config)


if __name__ == "__main__":
    main()
