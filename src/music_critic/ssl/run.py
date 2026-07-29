"""Hydra CLI for deterministic Phase 7A masked graph SSL."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from music_critic.ssl.config import register_ssl_configs


register_ssl_configs()


@hydra.main(version_base="1.3", config_name="ssl_training")
def main(config: DictConfig) -> None:
    from music_critic.ssl.engine import run_ssl_training

    run_ssl_training(config)


if __name__ == "__main__":
    main()
