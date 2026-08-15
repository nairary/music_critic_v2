"""Hydra CLI for Phase 8B.2A planning and protocol inspection."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from music_critic.experiments.phase8b2.config import (
    register_phase8b2_configs,
)
from music_critic.experiments.phase8b2.runner import build_experiment_plan


register_phase8b2_configs()


@hydra.main(version_base="1.3", config_name="phase8b2")
def main(config: DictConfig) -> None:
    if config.action != "plan":
        raise ValueError(
            "phase8b2.cli.action_unsupported: Phase 8B.2A CLI currently "
            "accepts action=plan; cell execution uses the emitted official "
            "SSL/training/evaluation engine bindings"
        )
    plan = build_experiment_plan(config)
    print(
        json.dumps(
            plan,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
