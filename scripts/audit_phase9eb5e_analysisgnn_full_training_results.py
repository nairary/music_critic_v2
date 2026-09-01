#!/usr/bin/env python3
"""Source-free audit for the completed Phase 9E-B5D paired results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.full_training import (
    build_full_comparison,
    full_training_contract,
    full_validation_updates,
)
from music_critic.experiments.analysisgnn.training_policy import PRIMARY_HEADS


RESULT_SCHEMA = "Phase9EB5EFullTrainingResults@1.0.0"
AUDIT_SCHEMA = "Phase9EB5EFullTrainingResultsAudit@1.0.0"
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5e_full_training_results.json"


def _load_fixture() -> dict[str, object]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint")
    if fingerprint(value) != observed:
        raise ValueError("phase9eb5e fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    return value


def audit() -> dict[str, object]:
    fixture = _load_fixture()
    source = fixture["source"]
    summaries = fixture["run_summaries"]
    comparison = fixture["comparison"]
    metrics = fixture["final_validation_metrics"]
    heads = fixture["primary_head_macro_f1"]
    decision = fixture["decision"]
    artifacts = fixture["artifact_policy"]
    reconstructed = build_full_comparison(
        summaries=summaries,
        validation=comparison["validation_curves"],
    )
    evidence_payload = {
        key: fixture[key]
        for key in (
            "source",
            "run_summaries",
            "comparison",
            "final_validation_metrics",
            "primary_head_macro_f1",
            "decision",
            "artifact_policy",
        )
    }
    c0 = metrics["C0"]
    c1 = metrics["C1"]
    delta = metrics["delta_C1_minus_C0"]
    checks = {
        "schema": fixture["schema"] == RESULT_SCHEMA
        and fixture["phase"] == "9E-B5E",
        "evidence_fingerprint": fixture["evidence_fingerprint"]
        == fingerprint(evidence_payload),
        "b5d_contract": source["full_training_contract_fingerprint"]
        == full_training_contract()["fingerprint"],
        "source_archive_attested": source["result_archive_sha256"]
        == "a9901c3ab9dd6914415a8ca7205f4247596c4aa261be9abe084d6a9523c7374a",
        "comparison_reconstructed": reconstructed == comparison
        and reconstructed["fingerprint"] == source["comparison_fingerprint"],
        "profiles_exact": set(summaries) == {"C0", "C1"}
        and summaries["C0"]["profile"] == "C0"
        and summaries["C1"]["profile"] == "C1",
        "paired_10000_update_run": summaries["C0"]["applied_updates"] == 10_000
        and summaries["C1"]["applied_updates"] == 10_000
        and summaries["C0"]["train_draws"] == 20_000
        and summaries["C1"]["train_draws"] == 20_000
        and summaries["C0"]["initial_model_state_fingerprint"]
        == summaries["C1"]["initial_model_state_fingerprint"]
        and summaries["C0"]["record_schedule_fingerprint"]
        == summaries["C1"]["record_schedule_fingerprint"]
        and summaries["C0"]["transposition_schedule_fingerprint"]
        != summaries["C1"]["transposition_schedule_fingerprint"],
        "validation_schedule": tuple(summaries["C0"]["validation_updates"])
        == full_validation_updates()
        and tuple(summaries["C1"]["validation_updates"])
        == full_validation_updates(),
        "primary_scores_exact": c0["corrected_primary_macro_score"]
        == 0.3548871111124754
        and c1["corrected_primary_macro_score"] == 0.2715279571712017
        and delta["corrected_primary_macro_score"]
        == -0.08335915394127369,
        "corrected_joint_exact": c0["corrected_harmonic_event_joint_accuracy"]
        == 0.11430474921480918
        and c1["corrected_harmonic_event_joint_accuracy"]
        == 0.01408584753021795
        and c0["corrected_joint_support"] == c1["corrected_joint_support"]
        == 10_507,
        "unseen_tuple_disclosed": c0["unseen_tuple_joint_accuracy"] == 0.0
        and c1["unseen_tuple_joint_accuracy"] == 0.0
        and c0["unseen_tuple_support"] == c1["unseen_tuple_support"] == 1_090
        and c0["unseen_tuple_count"] == c1["unseen_tuple_count"] == 187,
        "all_primary_heads_favor_C0": tuple(heads["C0"]) == PRIMARY_HEADS
        and tuple(heads["C1"]) == PRIMARY_HEADS
        and all(heads["C0"][task] > heads["C1"][task] for task in PRIMARY_HEADS),
        "C0_selected": decision["selected_profile"] == "C0"
        and decision["selected_model_state_fingerprint"]
        == summaries["C0"]["final_model_state_fingerprint"]
        and decision["baseline_status"]
        == "current_corrected_analysisgnn_baseline",
        "C1_deferred_not_deleted": decision["C1_status"]
        == "experimental_deferred"
        and decision["C1_deleted"] is False
        and decision["transposition_implementation_invalidated"] is False
        and decision["transposition_benefit_claim"] is False,
        "single_seed_only": decision["selection_scope"]
        == "seed_17_validation_only"
        and decision["multi_seed_run"] is False
        and decision["statistical_improvement_claim"] is False,
        "test_closed": comparison["test_evaluated"] is False
        and summaries["C0"]["test_evaluated"] is False
        and summaries["C1"]["test_evaluated"] is False
        and decision["test_evaluated"] is False,
        "compact_artifacts_only": artifacts["compact_result_committed"] is True
        and artifacts["comparison_json_committed"] is True
        and artifacts["checkpoints_committed"] is False
        and artifacts["training_logs_committed"] is False
        and artifacts["result_archive_committed"] is False
        and artifacts["datasets_committed"] is False,
    }
    valid = bool(fixture["valid"] and all(checks.values()))
    return {
        "schema": AUDIT_SCHEMA,
        "valid": valid,
        "selected_profile": decision["selected_profile"],
        "C1_status": decision["C1_status"],
        "final_primary_scores": comparison["final_primary_scores"],
        "final_primary_score_delta_C1_minus_C0": comparison[
            "final_primary_score_delta_C1_minus_C0"
        ],
        "corrected_joint_accuracy": {
            profile: metrics[profile]["corrected_harmonic_event_joint_accuracy"]
            for profile in ("C0", "C1")
        },
        "unseen_tuple_joint_accuracy": {
            profile: metrics[profile]["unseen_tuple_joint_accuracy"]
            for profile in ("C0", "C1")
        },
        "test_evaluated": decision["test_evaluated"],
        "multi_seed_run": decision["multi_seed_run"],
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
