"""Hydra CLI for the executable Phase 8B.2A comparison matrix."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from music_critic.experiments.phase8b2.config import (
    register_phase8b2_configs,
)
from music_critic.experiments.phase8b2.runner import build_experiment_plan


register_phase8b2_configs()


@hydra.main(version_base="1.3", config_name="phase8b2")
def main(config: DictConfig) -> None:
    if config.action not in {"plan", "run", "resume", "aggregate", "select"}:
        raise ValueError(
            "phase8b2.cli.action_unsupported: expected one of "
            "plan,run,resume,aggregate,select"
        )
    plan = build_experiment_plan(config)
    if config.action == "plan":
        result = plan
    else:
        from music_critic.experiments.phase8b2.orchestrator import (
            execute_matrix,
        )

        plain = OmegaConf.to_container(config, resolve=True)
        assert isinstance(plain, dict)
        plain.pop("defaults", None)
        result = execute_matrix(plain, plan, action=str(config.action))
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
