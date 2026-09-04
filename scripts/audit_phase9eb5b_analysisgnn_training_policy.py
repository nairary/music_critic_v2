#!/usr/bin/env python3
"""Build or source-free verify the Phase 9E-B5B training-policy fixture."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.training_policy import (
    B3_SEMANTIC_FINGERPRINT,
    B4_SEMANTIC_FINGERPRINT,
    B5A_SEMANTIC_FINGERPRINT,
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    OFFICIAL_TRAINING_PROFILE_ID,
    TRAINING_POLICY_SCHEMA,
    AnalysisGNNTrainingPolicyError,
    build_class_weight_payload,
    build_training_profiles,
    combined_training_policy_contract,
    component_sampler_contract,
    corrected_loss_contract,
    corrected_metric_contract,
    corrected_profile_comparison,
    experiment_matrix_contract,
    head_role_contract,
    official_contracts,
    stop_gate_contract,
    validate_class_weight_payload,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID


DEFAULT_B3_ROOT = (
    REPO_ROOT / "outputs/phase9eb3/analysisgnn-multitask-contract-01290f5"
)
DEFAULT_B4_ROOT = (
    REPO_ROOT / "outputs/phase9eb4/analysisgnn-class-balance-671097b"
)
DEFAULT_B4_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb4_class_balance_audit.json"
)
DEFAULT_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json"
)
FIXTURE_SCHEMA = "phase9eb5b-analysisgnn-training-policy-fixture-v1"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _same_json(left: object, right: object) -> bool:
    return canonical_json(left) == canonical_json(right)


def _load_b4_train_rows(
    class_counts_path: Path, b4_fixture_path: Path
) -> dict[str, list[dict[str, object]]]:
    fixture = json.loads(b4_fixture_path.read_text(encoding="utf-8"))
    observed_fixture = fixture.pop("fixture_fingerprint", None)
    if observed_fixture != fingerprint(fixture):
        raise AnalysisGNNTrainingPolicyError("B4 fixture fingerprint mismatch")
    if fixture.get("semantic_fingerprint") != B4_SEMANTIC_FINGERPRINT:
        raise AnalysisGNNTrainingPolicyError("B4 semantic fingerprint drift")
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _read_jsonl(class_counts_path):
        if row.get("split") == "train" and row.get("task_id") in TASK_BY_ID:
            rows[str(row["task_id"])].append(row)
    return dict(rows)


def _component_sampling_evidence(split_path: Path) -> dict[str, object]:
    component_records: dict[str, list[str]] = defaultdict(list)
    held_out_records = 0
    for row in _read_jsonl(split_path):
        if row.get("split") == "train":
            component_records[str(row["source_component_id"])].append(
                str(row["record_id"])
            )
        else:
            held_out_records += 1
    sizes = Counter(len(records) for records in component_records.values())
    record_count = sum(len(records) for records in component_records.values())
    payload: dict[str, object] = {
        "train_component_count": len(component_records),
        "train_record_count": record_count,
        "held_out_record_count": held_out_records,
        "component_size_counts": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "minimum_component_size": min(sizes),
        "maximum_component_size": max(sizes),
        "component_selection_probability": f"1/{len(component_records)}",
        "record_selection_probability_formula": (
            f"1/({len(component_records)}*records_in_selected_component)"
        ),
        "draws_per_epoch": 1295,
        "test_draw_count": 0,
        "validation_oversampled": False,
        "test_oversampled": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def build_fixture(
    *,
    b3_root: Path = DEFAULT_B3_ROOT,
    b4_root: Path = DEFAULT_B4_ROOT,
    b4_fixture: Path = DEFAULT_B4_FIXTURE,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, object]:
    train_rows = _load_b4_train_rows(
        b4_root / "class_counts.jsonl", b4_fixture
    )
    class_weights = build_class_weight_payload(train_rows)
    profiles = build_training_profiles(class_weights)
    comparison = corrected_profile_comparison(profiles["C0"], profiles["C1"])
    component_evidence = _component_sampling_evidence(
        b3_root / "split_assignments.jsonl"
    )
    combined = combined_training_policy_contract(class_weights)
    contracts = {
        "head_roles": head_role_contract(),
        "loss": corrected_loss_contract(),
        "class_weights": class_weights["contract"],
        "sampler": component_sampler_contract(),
        "metrics": corrected_metric_contract(),
        "stop_gates": stop_gate_contract(),
        "experiment_matrix": experiment_matrix_contract(),
    }
    payload: dict[str, object] = {
        "fixture_schema": FIXTURE_SCHEMA,
        "schema": TRAINING_POLICY_SCHEMA,
        "input_fingerprints": {
            "b3_semantic": B3_SEMANTIC_FINGERPRINT,
            "b4_semantic": B4_SEMANTIC_FINGERPRINT,
            "b5a_semantic": B5A_SEMANTIC_FINGERPRINT,
        },
        "contracts": contracts,
        "class_weight_payload": class_weights,
        "component_sampling_evidence": component_evidence,
        "profiles": {key: profile.to_dict() for key, profile in profiles.items()},
        "profile_ids": {
            "O": OFFICIAL_TRAINING_PROFILE_ID,
            "C0": CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
            "C1": CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
        },
        "corrected_profile_comparison": comparison,
        "official_contract_fingerprint": official_contracts()["fingerprint"],
        "combined_training_policy": combined,
        "fingerprints": {
            "head_roles": contracts["head_roles"]["fingerprint"],
            "loss": contracts["loss"]["fingerprint"],
            "class_weights": class_weights["fingerprint"],
            "sampler": contracts["sampler"]["fingerprint"],
            "metrics": contracts["metrics"]["fingerprint"],
            "profile_O": profiles["O"].semantic_fingerprint,
            "profile_C0": profiles["C0"].semantic_fingerprint,
            "profile_C1": profiles["C1"].semantic_fingerprint,
            "combined": combined["fingerprint"],
        },
        "valid": True,
        "ready_for_model_implementation": True,
        "training_run": False,
        "validation_inference_run": False,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
    }
    payload["audit_semantic_fingerprint"] = fingerprint(payload)
    payload["fixture_fingerprint"] = fingerprint(payload)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        canonical_json(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def _validate_component_evidence(value: Mapping[str, object]) -> None:
    observed = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if observed != fingerprint(body):
        raise AnalysisGNNTrainingPolicyError("component evidence fingerprint mismatch")
    sizes = value.get("component_size_counts")
    if not isinstance(sizes, dict):
        raise AnalysisGNNTrainingPolicyError("component-size evidence is absent")
    component_count = sum(int(count) for count in sizes.values())
    record_count = sum(int(size) * int(count) for size, count in sizes.items())
    if component_count != 1209 or record_count != 1295:
        raise AnalysisGNNTrainingPolicyError("component sampler counts drifted")
    if value.get("test_draw_count") != 0:
        raise AnalysisGNNTrainingPolicyError("TEST draws are forbidden")


def check_fixture(fixture_path: Path = DEFAULT_FIXTURE) -> dict[str, object]:
    text = fixture_path.read_text(encoding="utf-8")
    fixture = json.loads(text)
    observed_fixture = fixture.pop("fixture_fingerprint", None)
    if observed_fixture != fingerprint(fixture):
        raise AnalysisGNNTrainingPolicyError("B5B fixture fingerprint mismatch")
    fixture["fixture_fingerprint"] = observed_fixture
    canonical = canonical_json(fixture, indent=2) + "\n"
    if canonical != text:
        raise AnalysisGNNTrainingPolicyError("B5B fixture bytes are not canonical")

    semantic = dict(fixture)
    semantic.pop("fixture_fingerprint")
    observed_semantic = semantic.pop("audit_semantic_fingerprint", None)
    if observed_semantic != fingerprint(semantic):
        raise AnalysisGNNTrainingPolicyError("audit semantic fingerprint mismatch")

    if fixture.get("fixture_schema") != FIXTURE_SCHEMA:
        raise AnalysisGNNTrainingPolicyError("fixture schema mismatch")
    if fixture.get("schema") != TRAINING_POLICY_SCHEMA:
        raise AnalysisGNNTrainingPolicyError("training-policy schema mismatch")
    if fixture.get("input_fingerprints") != {
        "b3_semantic": B3_SEMANTIC_FINGERPRINT,
        "b4_semantic": B4_SEMANTIC_FINGERPRINT,
        "b5a_semantic": B5A_SEMANTIC_FINGERPRINT,
    }:
        raise AnalysisGNNTrainingPolicyError("upstream semantic binding mismatch")

    class_weights = fixture.get("class_weight_payload")
    if not isinstance(class_weights, dict):
        raise AnalysisGNNTrainingPolicyError("class-weight payload missing")
    validate_class_weight_payload(class_weights)
    profiles = build_training_profiles(class_weights)
    expected_profiles = {key: profile.to_dict() for key, profile in profiles.items()}
    if not _same_json(fixture.get("profiles"), expected_profiles):
        raise AnalysisGNNTrainingPolicyError("serialized profile drift")
    comparison = corrected_profile_comparison(profiles["C0"], profiles["C1"])
    if not _same_json(fixture.get("corrected_profile_comparison"), comparison):
        raise AnalysisGNNTrainingPolicyError("C0/C1 comparison drift")
    if comparison["only_transposition_differs"] is not True:
        raise AnalysisGNNTrainingPolicyError("C0/C1 differ outside transposition")

    contracts = fixture.get("contracts")
    if not isinstance(contracts, dict):
        raise AnalysisGNNTrainingPolicyError("contract table missing")
    expected_contracts = {
        "head_roles": head_role_contract(),
        "loss": corrected_loss_contract(),
        "class_weights": class_weights["contract"],
        "sampler": component_sampler_contract(),
        "metrics": corrected_metric_contract(),
        "stop_gates": stop_gate_contract(),
        "experiment_matrix": experiment_matrix_contract(),
    }
    if not _same_json(contracts, expected_contracts):
        raise AnalysisGNNTrainingPolicyError("contract table drift")
    combined = combined_training_policy_contract(class_weights)
    if not _same_json(fixture.get("combined_training_policy"), combined):
        raise AnalysisGNNTrainingPolicyError("combined policy drift")
    component_evidence = fixture.get("component_sampling_evidence")
    if not isinstance(component_evidence, dict):
        raise AnalysisGNNTrainingPolicyError("component evidence missing")
    _validate_component_evidence(component_evidence)
    if fixture.get("official_contract_fingerprint") != official_contracts()[
        "fingerprint"
    ]:
        raise AnalysisGNNTrainingPolicyError("official contract drift")
    required_flags = {
        "valid": True,
        "ready_for_model_implementation": True,
        "training_run": False,
        "validation_inference_run": False,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
    }
    if any(fixture.get(key) is not value for key, value in required_flags.items()):
        raise AnalysisGNNTrainingPolicyError("execution-state flags are invalid")
    return fixture


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-root", type=Path, default=DEFAULT_B3_ROOT)
    parser.add_argument("--b4-root", type=Path, default=DEFAULT_B4_ROOT)
    parser.add_argument("--b4-fixture", type=Path, default=DEFAULT_B4_FIXTURE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        value = check_fixture(args.fixture)
    else:
        value = build_fixture(
            b3_root=args.b3_root,
            b4_root=args.b4_root,
            b4_fixture=args.b4_fixture,
            fixture_path=args.fixture,
        )
    print(
        canonical_json(
            {
                "valid": value["valid"],
                "ready_for_model_implementation": value[
                    "ready_for_model_implementation"
                ],
                "training_run": value["training_run"],
                "validation_inference_run": value["validation_inference_run"],
                "test_evaluated": value["test_evaluated"],
                "test_targets_used_for_evaluation": value[
                    "test_targets_used_for_evaluation"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
