"""Preregistered pilot bounds; no corpus, checkpoint or accelerator access."""

import json
import math
from pathlib import Path

import pytest

from music_critic.experiments.analysisgnn.harmonic_fusion_training import (
    FusionTrainingConfig,
    schedule_factor,
)


ROOT = Path(__file__).resolve().parents[2]


def _pilot():
    path = ROOT / "configs/experiments/harmonic_fusion_pilot.json"
    return FusionTrainingConfig(**json.loads(path.read_text(encoding="utf-8")))


def test_pilot_config_has_complete_equal_budgets_within_predeclared_slot():
    config = _pilot()
    record = json.loads((ROOT / "docs/experiments/records/EXP-9EP1-GPU-EVIDENCE-015.json").read_text())
    rate = next(arm["steady_updates_per_second"] for arm in record["results"]["arms"]
                if arm["arm_key"] == "original-workers0")
    # This is only a checked planning calculation. It never converts the C0
    # measurement into C2 throughput or a passed scientific launch gate.
    planning_updates = 256 * math.floor(1200 * rate / (2 * 1.10 * 256))
    assert config.total_updates == planning_updates == 512
    assert config.batch_size * config.total_updates == 1024
    assert config.validation_mode == "required"
    assert config.validation_interval == 256
    assert config.checkpoint_interval == config.warmup_updates == 64
    assert config.total_updates % config.validation_interval == 0
    assert config.total_updates % config.checkpoint_interval == 0
    contract = config.to_dict()
    assert contract["early_stopping"] is False
    assert contract["test_enabled"] is False
    assert contract["loader_workers"] == 0
    assert contract["continuation"] == "fresh_optimizer_not_historical_resume"


def test_frozen_pilot_lr_covers_full_budget_and_ends_at_zero():
    config = _pilot()
    rates = [config.peak_lr * schedule_factor(config, step)
             for step in range(config.total_updates + 1)]
    assert config.peak_lr == 0.0003
    assert rates[0] == pytest.approx(0.0003 / 64)
    assert rates[63] == rates[64] == 0.0003
    assert rates[-1] == 0
    assert all(a <= b for a, b in zip(rates[:63], rates[1:64]))
    assert all(a >= b for a, b in zip(rates[64:-1], rates[65:]))
    assert all(math.isfinite(rate) and 0 <= rate <= 0.0003 for rate in rates)
