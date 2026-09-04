#!/usr/bin/env python3
"""Source-free audit for the Phase 9E-B5D full-training boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_AUDIT_SCHEMA,
    full_runtime_config,
    full_training_contract,
    full_validation_updates,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5d_full_training.json"


def _load_fixture() -> dict[str, object]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint")
    if fingerprint(value) != observed:
        raise ValueError("phase9eb5d fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    return value


def audit() -> dict[str, object]:
    fixture = _load_fixture()
    preflight = fixture["preflight"]
    pilots = fixture["completed_b5c_pilots"]
    status = fixture["status"]
    contract = full_training_contract()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    initial_fingerprint = model_state_fingerprint(CorrectedAnalysisGNNModel())
    c0 = full_runtime_config("C0").to_dict()
    c1 = full_runtime_config("C1").to_dict()
    record_schedules = preflight["record_schedule_fingerprints"]
    shift_schedules = preflight["transposition_schedule_fingerprints"]
    expected_evidence = {
        "contract_fingerprint": fixture["contract_fingerprint"],
        "preflight": fixture["preflight"],
        "completed_b5c_pilots": fixture["completed_b5c_pilots"],
        "commands": fixture["commands"],
        "status": fixture["status"],
    }
    checks = {
        "contract_fingerprint": contract["fingerprint"]
        == fixture["contract_fingerprint"],
        "exact_budget": contract["applied_update_budget"] == 10_000
        and contract["train_draw_budget"] == 20_000
        and contract["batch_size"] == 2,
        "exact_schedule": contract["warmup_applied_updates"] == 500
        and contract["validation_interval"] == 500
        and contract["checkpoint_interval"] == 100
        and tuple(contract["validation_updates"]) == full_validation_updates(),
        "optimizer_envelope": contract["optimizer"]["optimizer"] == "AdamW"
        and contract["optimizer"]["learning_rate"] == 0.005
        and contract["optimizer"]["weight_decay"] == 0.0005
        and contract["optimizer"]["gradient_clip_norm"] == 1.0,
        "resolved_C0": c0["fingerprint"] == preflight["resolved_C0_fingerprint"],
        "resolved_C1": c1["fingerprint"] == preflight["resolved_C1_fingerprint"],
        "paired_initial_state": initial_fingerprint
        == preflight["initial_model_state_fingerprint"],
        "paired_record_schedule": record_schedules["C0"]
        == record_schedules["C1"]
        and len(record_schedules["C0"]) == 64,
        "separate_transposition_schedule": shift_schedules["C0"]
        != shift_schedules["C1"]
        and len(shift_schedules["C0"]) == 64
        and len(shift_schedules["C1"]) == 64,
        "b5c_pilots_valid": pilots["C0"]["valid"] is True
        and pilots["C1"]["valid"] is True
        and pilots["C0"]["applied_updates"] == 500
        and pilots["C1"]["applied_updates"] == 500,
        "b5c_delta_exact": pilots["C1"]["final_primary_score"]
        - pilots["C0"]["final_primary_score"]
        == pilots["final_primary_score_delta_C1_minus_C0"],
        "test_closed": contract["test_enabled"] is False
        and preflight["test_loader_created"] is False
        and preflight["test_targets_read"] is False
        and preflight["test_metrics_computed"] is False
        and pilots["test_evaluated"] is False
        and status["test_evaluated"] is False,
        "no_fabricated_full_result": status["c0_full_training_completed"] is False
        and status["c1_full_training_completed"] is False
        and status["comparison_completed"] is False,
        "ready": status["runner_implemented"] is True
        and status["preflight_completed"] is True
        and status["ready_for_paired_cuda_full_training"] is True,
        "no_statistical_claim": contract["multi_seed_claim"] is False
        and pilots["multi_seed_run"] is False
        and status["multi_seed_run"] is False,
        "evidence_fingerprint": fixture["evidence_fingerprint"]
        == fingerprint(expected_evidence),
    }
    valid = bool(fixture["valid"] and all(checks.values()))
    return {
        "schema": FULL_AUDIT_SCHEMA,
        "valid": valid,
        "ready_for_paired_cuda_full_training": status[
            "ready_for_paired_cuda_full_training"
        ],
        "c0_full_training_completed": status["c0_full_training_completed"],
        "c1_full_training_completed": status["c1_full_training_completed"],
        "comparison_completed": status["comparison_completed"],
        "test_evaluated": status["test_evaluated"],
        "multi_seed_run": status["multi_seed_run"],
        "checks": checks,
        "fixture_fingerprint": fixture["fixture_fingerprint"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
