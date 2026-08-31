#!/usr/bin/env python3
"""Source-free audit for the Phase 9E-B5C implementation/pilot boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    corrected_parameter_inventory,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedRuntimeConfig,
    implementation_fingerprints,
)
from music_critic.experiments.analysisgnn.training_policy import (
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5c_model_pilot.json"


def _load_fixture() -> dict[str, object]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint")
    if fingerprint(value) != observed:
        raise ValueError("phase9eb5c fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    return value


def audit() -> dict[str, object]:
    fixture = _load_fixture()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = CorrectedAnalysisGNNModel()
    implementations = implementation_fingerprints(model)
    c0 = CorrectedRuntimeConfig(
        profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        applied_update_budget=500,
    ).to_dict()
    c1 = CorrectedRuntimeConfig(
        profile_id=CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
        applied_update_budget=500,
    ).to_dict()
    expected = fixture["fingerprints"]
    checks = {
        "model_architecture": corrected_model_contract(model)["fingerprint"] == expected["model_architecture"],
        "parameter_inventory": corrected_parameter_inventory(model)["fingerprint"] == expected["parameter_inventory"],
        "active_head_inventory": implementations["active_head_inventory"] == expected["active_head_inventory"],
        "routing_contract": implementations["routing_contract"] == expected["routing_contract"],
        "loss_implementation": implementations["loss_implementation"] == expected["loss_implementation"],
        "class_weight_contract": implementations["class_weight_contract"] == expected["class_weight_contract"],
        "optimizer_envelope": implementations["optimizer_envelope"] == expected["optimizer_envelope"],
        "metric_implementation": implementations["metric_implementation"] == expected["metric_implementation"],
        "joint_metric_implementation": implementations["joint_metric_implementation"] == expected["joint_metric_implementation"],
        "test_lock": implementations["test_lock"] == expected["test_lock"],
        "initial_model_state": model_state_fingerprint(model) == expected["initial_model_state"],
        "resolved_C0": c0["fingerprint"] == expected["resolved_C0"],
        "resolved_C1": c1["fingerprint"] == expected["resolved_C1"],
        "C0_C1_record_schedule": expected["C0_record_schedule"]
        == expected["C1_record_schedule"]
        and len(expected["C0_record_schedule"]) == 64,
        "C0_C1_transposition_schedule": expected["C0_transposition_schedule"]
        != expected["C1_transposition_schedule"]
        and len(expected["C0_transposition_schedule"]) == 64
        and len(expected["C1_transposition_schedule"]) == 64,
    }
    status = fixture["status"]
    smoke = fixture["smoke_evidence"]
    checks.update(
        {
            "combined_audit_fingerprint": fixture["combined_audit_fingerprint"]
            == fingerprint(
                {
                    "status": fixture["status"],
                    "fingerprints": fixture["fingerprints"],
                    "smoke_evidence": fixture["smoke_evidence"],
                }
            ),
            "exact_18_head_coverage": smoke["real_train_coverage"]["finite_loss_head_count"] == 18
            and smoke["real_train_coverage"]["nonzero_gradient_head_count"] == 18,
            "shared_encoder_gradient": smoke["real_train_coverage"]["shared_encoder_nonzero_gradient"] is True,
            "test_closed": status["test_evaluated"] is False
            and status["test_targets_used_for_evaluation"] is False,
            "no_full_or_multiseed": status["full_training_run"] is False
            and status["multi_seed_run"] is False,
            "cuda_status_honest": status["cuda_smoke_passed"] is False
            and status["pilot_not_run_reason"] == "cuda_unavailable"
            and status["ready_for_cuda_pilot"] is True,
            "pilots_not_fabricated": status["c0_pilot_completed"] is False
            and status["c1_pilot_completed"] is False
            and status["comparison_completed"] is False,
        }
    )
    valid = bool(fixture["valid"] and all(checks.values()))
    return {
        "valid": valid,
        "model_implemented": status["model_implemented"],
        "trainer_implemented": status["trainer_implemented"],
        "cpu_smoke_passed": status["cpu_smoke_passed"],
        "real_train_coverage_smoke_passed": status["real_train_coverage_smoke_passed"],
        "cuda_smoke_passed": status["cuda_smoke_passed"],
        "ready_for_cuda_pilot": status["ready_for_cuda_pilot"],
        "pilot_not_run_reason": status["pilot_not_run_reason"],
        "c0_pilot_completed": status["c0_pilot_completed"],
        "c1_pilot_completed": status["c1_pilot_completed"],
        "comparison_completed": status["comparison_completed"],
        "test_evaluated": status["test_evaluated"],
        "test_targets_used_for_evaluation": status["test_targets_used_for_evaluation"],
        "full_training_run": status["full_training_run"],
        "multi_seed_run": status["multi_seed_run"],
        "checks": checks,
        "fixture_fingerprint": fixture["fixture_fingerprint"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    del args
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
