#!/usr/bin/env python3
"""Verify the independent Phase 8B.2A RTX 3090 bounded smoke bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


EXPECTED_DATASET_IDS = {"hooktheory", "pop909_cl"}
EXPECTED_VALIDATION_SAMPLES = 128
EXPECTED_CUDA_MEMORY_LIFECYCLE_VERSION = "1.0.0"


def _json_value(value: object) -> object:
    if isinstance(value, set):
        return sorted(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _dataset_ids(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            str(key)
            for key, count in value.items()
            if isinstance(count, int) and not isinstance(count, bool) and count > 0
        }
    if isinstance(value, list):
        result = set()
        for row in value:
            if (
                isinstance(row, list)
                and len(row) == 2
                and isinstance(row[1], int)
                and not isinstance(row[1], bool)
                and row[1] > 0
            ):
                result.add(str(row[0]))
        return result
    return set()


def _dataset_count_total(value: object) -> int | None:
    if isinstance(value, Mapping):
        counts = tuple(value.values())
    elif isinstance(value, list):
        counts = tuple(
            row[1]
            for row in value
            if isinstance(row, list) and len(row) == 2
        )
        if len(counts) != len(value):
            return None
    else:
        return None
    if not all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in counts
    ):
        return None
    return sum(counts)


class _Checks:
    def __init__(self) -> None:
        self.failures: list[dict[str, object]] = []

    def equal(self, code: str, observed: object, expected: object) -> None:
        if observed != expected:
            self.failures.append(
                {
                    "check": code,
                    "expected": _json_value(expected),
                    "observed": _json_value(observed),
                }
            )

    def true(self, code: str, condition: object, observed: object) -> None:
        if condition is not True:
            self.failures.append(
                {
                    "check": code,
                    "expected": True,
                    "observed": _json_value(observed),
                }
            )


def verify_gate(
    output_root: Path,
    *,
    expected_sha: str,
    invocation_config: Path,
    expected_device_name: str = "RTX 3090",
) -> dict[str, object]:
    """Return complete verification evidence without reading corpus payloads."""

    root = output_root.resolve()
    bundle = root / "final_bundle"
    checks = _Checks()
    required = {
        "plan": root / "plan.json",
        "schedule": root / "actual_sample_schedule.json",
        "report": bundle / "final_comparison_report.json",
        "compute": bundle / "compute_accounting.json",
        "manifest": bundle / "run_manifest.json",
        "protocol": bundle / "comparison_protocol.json",
        "invocation": invocation_config.resolve(),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in required.items():
        if not path.is_file():
            checks.failures.append(
                {"check": f"artifact.{name}.present", "path": str(path)}
            )
            continue
        try:
            payloads[name] = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checks.failures.append(
                {
                    "check": f"artifact.{name}.valid_json_object",
                    "path": str(path),
                    "error": str(exc),
                }
            )

    if checks.failures:
        return {
            "status": "failed",
            "output_root": str(root),
            "failures": checks.failures,
        }

    plan = payloads["plan"]
    schedule = payloads["schedule"]
    report = payloads["report"]
    compute = payloads["compute"]
    manifest = payloads["manifest"]
    protocol = payloads["protocol"]
    invocation = payloads["invocation"]

    for field, expected in (
        ("executed_cell_count", 8),
        ("expected_cell_count", 8),
        ("verified_runtime_binding_cell_count", 8),
        ("checkpoint_to_evaluation_verified_cell_count", 3),
        ("test_inference_performed", False),
        ("test_targets_accessed", False),
        ("test_metrics_accessed", False),
        ("bounded_results_are_scientific_superiority_evidence", False),
    ):
        checks.equal(f"final_report.{field}", report.get(field), expected)
    checks.equal("final_report.status", report.get("status"), "complete")
    checks.equal(
        "final_report.ssl_attempted_updates",
        report.get("ssl_compute_totals", {}).get("attempted_updates"),
        1,
    )
    checks.equal(
        "final_report.ssl_applied_updates",
        report.get("ssl_compute_totals", {}).get("applied_updates"),
        1,
    )
    checks.equal(
        "final_report.ssl_skipped_updates",
        report.get("ssl_compute_totals", {}).get("skipped_updates"),
        0,
    )

    checks.equal(
        "invocation.run_label",
        invocation.get("run_label"),
        "production-format real-corpus bounded smoke",
    )
    checks.equal(
        "invocation.comparison", invocation.get("comparison"), "bounded_acceptance"
    )
    checks.equal("invocation.variants", invocation.get("variants"), ["phase7a_control"])
    checks.equal("invocation.seeds", invocation.get("seeds"), [17])
    checks.equal(
        "invocation.ssl_optimizer_steps",
        invocation.get("ssl_optimizer_steps"),
        1,
    )
    checks.equal(
        "invocation.downstream_optimizer_steps",
        invocation.get("downstream_optimizer_steps"),
        1,
    )
    checks.equal("invocation.device", invocation.get("device"), "cuda:0")
    checks.equal("invocation.amp", invocation.get("amp"), True)
    checks.equal("invocation.amp_dtype", invocation.get("amp_dtype"), "float16")
    checks.equal(
        "invocation.validation_samples",
        invocation.get("validation_samples"),
        EXPECTED_VALIDATION_SAMPLES,
    )
    checks.equal(
        "invocation.expected_git_sha",
        invocation.get("expected_git_sha"),
        expected_sha,
    )
    checks.equal("invocation.output_root", invocation.get("output_root"), str(root))

    plan_protocol = plan.get("protocol", {})
    checks.equal("final_protocol.matches_plan", protocol, plan_protocol)
    checks.equal("plan.seeds", plan_protocol.get("seeds"), [17])
    checks.equal("plan.variants", plan_protocol.get("variants"), ["phase7a_control"])
    checks.equal("plan.ssl_cell_count", len(plan.get("ssl_cells", [])), 1)
    checks.equal("plan.downstream_cell_count", len(plan.get("downstream_cells", [])), 3)
    checks.equal("plan.evaluation_cell_count", len(plan.get("evaluation_cells", [])), 3)
    runtime_config = plan_protocol.get("runtime_execution_config", {})
    checks.equal(
        "plan.ssl_attempted_updates",
        runtime_config.get("ssl_attempted_logical_updates"),
        1,
    )
    checks.equal(
        "plan.downstream_attempted_updates",
        runtime_config.get("downstream_attempted_logical_updates"),
        1,
    )
    checks.equal(
        "plan.validation_samples",
        runtime_config.get("validation_samples"),
        EXPECTED_VALIDATION_SAMPLES,
    )
    amp_config = plan_protocol.get("amp_device_config", {})
    checks.equal("plan.device", amp_config.get("name"), "cuda:0")
    checks.equal("plan.amp", amp_config.get("amp"), True)
    checks.equal("plan.amp_dtype", amp_config.get("amp_dtype"), "float16")
    test_lock = plan_protocol.get("test_unlock_state", {})
    for field, expected in (
        ("acknowledged", False),
        ("unlocked", False),
        ("test_inference_performed", False),
        ("test_targets_accessed", False),
        ("test_metrics_accessed", False),
    ):
        checks.equal(f"plan.test_lock.{field}", test_lock.get(field), expected)
    data_attestation = plan.get("data_attestation", {})
    for field in (
        "test_inference_performed",
        "test_targets_accessed",
        "test_metrics_accessed",
    ):
        checks.equal(
            f"plan.data_attestation.{field}",
            data_attestation.get(field),
            False,
        )
    test_summary = data_attestation.get("test_membership_summary", {})
    checks.equal(
        "plan.test_membership_has_no_identities",
        "selected_identities" in test_summary,
        False,
    )
    validation_membership = data_attestation.get("validation_membership", {})
    validation_fingerprint = validation_membership.get(
        "membership_fingerprint"
    )
    checks.true(
        "plan.validation_membership_fingerprint_present",
        isinstance(validation_fingerprint, str)
        and len(validation_fingerprint) == 64
        and all(
            character in "0123456789abcdef"
            for character in validation_fingerprint
        ),
        validation_fingerprint,
    )
    checks.equal(
        "plan.validation_membership_selected_count",
        validation_membership.get("selected_count"),
        EXPECTED_VALIDATION_SAMPLES,
    )
    checks.equal(
        "plan.validation_membership_subset_limit",
        validation_membership.get("subset_limit"),
        EXPECTED_VALIDATION_SAMPLES,
    )
    checks.equal(
        "plan.validation_membership_dataset_ids",
        _dataset_ids(validation_membership.get("dataset_counts")),
        EXPECTED_DATASET_IDS,
    )
    checks.equal(
        "plan.validation_membership_dataset_count_total",
        _dataset_count_total(validation_membership.get("dataset_counts")),
        EXPECTED_VALIDATION_SAMPLES,
    )
    selected_identities = validation_membership.get("selected_identities", [])
    checks.equal(
        "plan.validation_membership_identity_count",
        len(selected_identities),
        EXPECTED_VALIDATION_SAMPLES,
    )
    checks.equal(
        "protocol.validation_membership_fingerprint",
        plan_protocol.get("data", {}).get(
            "validation_membership_fingerprint"
        ),
        validation_fingerprint,
    )

    runtime_paths = plan.get("runtime_paths", {})
    checks.equal("plan.index_path_count", len(runtime_paths.get("index_paths", [])), 2)
    checks.equal("plan.cache_root_count", len(runtime_paths.get("cache_roots", [])), 2)
    checks.true(
        "plan.split_manifest_bound",
        bool(runtime_paths.get("split_manifest")),
        runtime_paths.get("split_manifest"),
    )
    checks.equal(
        "invocation.index_paths_match_plan",
        invocation.get("index_paths"),
        runtime_paths.get("index_paths"),
    )
    checks.equal(
        "invocation.cache_roots_match_plan",
        invocation.get("cache_roots"),
        runtime_paths.get("cache_roots"),
    )
    checks.equal(
        "invocation.split_manifest_matches_plan",
        invocation.get("split_manifest"),
        runtime_paths.get("split_manifest"),
    )

    for kind in ("ssl", "downstream"):
        rows = schedule.get(kind, [])
        checks.equal(f"schedule.{kind}.row_count", len(rows), 1)
        for ordinal, row in enumerate(rows):
            slot_ids = {
                str(slot.get("dataset_id"))
                for slot in row.get("slots", [])
                if isinstance(slot, dict)
            }
            checks.equal(
                f"schedule.{kind}.{ordinal}.train_slots_dataset_ids",
                slot_ids,
                EXPECTED_DATASET_IDS,
            )
            projection = row.get("data_semantic_projection", {})
            checks.equal(
                f"schedule.{kind}.{ordinal}.train_composition_dataset_ids",
                _dataset_ids(
                    projection.get("train_composition", {}).get("dataset_counts")
                ),
                EXPECTED_DATASET_IDS,
            )
            projected_validation = projection.get(
                "validation_membership", {}
            )
            checks.equal(
                f"schedule.{kind}.{ordinal}.validation_membership_dataset_ids",
                _dataset_ids(projected_validation.get("dataset_counts")),
                EXPECTED_DATASET_IDS,
            )
            checks.equal(
                f"schedule.{kind}.{ordinal}.validation_dataset_count_total",
                _dataset_count_total(projected_validation.get("dataset_counts")),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"schedule.{kind}.{ordinal}.validation_selected_count",
                projected_validation.get("selected_count"),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"schedule.{kind}.{ordinal}.validation_subset_limit",
                projected_validation.get("subset_limit"),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"schedule.{kind}.{ordinal}.validation_membership_fingerprint",
                projected_validation.get("membership_fingerprint"),
                validation_fingerprint,
            )

    repository = manifest.get("repository", {})
    environment = manifest.get("environment", {})
    checks.equal("manifest.git_sha", repository.get("git_sha"), expected_sha)
    checks.equal("manifest.device", environment.get("device"), "cuda:0")
    checks.equal("manifest.cuda_available", environment.get("cuda_available"), True)
    checks.equal(
        "manifest.cuda_logical_device_index",
        environment.get("cuda_logical_device_index"),
        0,
    )
    checks.equal(
        "manifest.cuda_memory_statistics_lifecycle_contract_version",
        environment.get(
            "cuda_memory_statistics_lifecycle_contract_version"
        ),
        EXPECTED_CUDA_MEMORY_LIFECYCLE_VERSION,
    )
    device_name = environment.get("cuda_device_name")
    checks.true(
        "manifest.cuda_device_name",
        isinstance(device_name, str) and expected_device_name in device_name,
        device_name,
    )

    compute_cells = compute.get("cells", [])
    checks.equal("compute.ssl_cell_count", len(compute_cells), 1)
    for ordinal, cell in enumerate(compute_cells):
        checks.equal(
            f"compute.{ordinal}.logical_updates",
            cell.get("logical_updates"),
            1,
        )
        checks.equal(
            f"compute.{ordinal}.optimizer_updates_applied",
            cell.get("optimizer_updates_applied"),
            1,
        )
        checks.equal(
            f"compute.{ordinal}.optimizer_updates_skipped",
            cell.get("optimizer_updates_skipped"),
            0,
        )
        peak = cell.get("cuda_peak_memory", {})
        checks.equal(
            f"compute.{ordinal}.cuda_peak_available",
            peak.get("available"),
            True,
        )
        checks.equal(
            f"compute.{ordinal}.cuda_logical_device_index",
            peak.get("cuda_logical_device_index"),
            0,
        )
        checks.true(
            f"compute.{ordinal}.peak_allocated_positive",
            isinstance(peak.get("peak_allocated_bytes"), int)
            and not isinstance(peak.get("peak_allocated_bytes"), bool)
            and peak["peak_allocated_bytes"] > 0,
            peak.get("peak_allocated_bytes"),
        )
        checks.true(
            f"compute.{ordinal}.peak_reserved_positive",
            isinstance(peak.get("peak_reserved_bytes"), int)
            and not isinstance(peak.get("peak_reserved_bytes"), bool)
            and peak["peak_reserved_bytes"] > 0,
            peak.get("peak_reserved_bytes"),
        )

    manifests = tuple(sorted((root / "cells").rglob("cell_manifest.json")))
    runtime_verified = 0
    evaluation_verified = 0
    for path in manifests:
        cell = _read_json(path)
        binding = cell.get("runtime_binding_evidence", {})
        if binding.get("verified") is True:
            runtime_verified += 1
        if binding.get("checkpoint_to_evaluation_data_verified") is True:
            evaluation_verified += 1
    checks.equal("cells.runtime_binding_verified_count", runtime_verified, 8)
    checks.equal(
        "cells.checkpoint_to_evaluation_verified_count",
        evaluation_verified,
        3,
    )
    preflight_manifests = tuple(
        path
        for path in manifests
        if "preflight" in path.relative_to(root / "cells").parts
    )
    checks.equal(
        "cells.preflight_manifest_count",
        len(preflight_manifests),
        1,
    )
    for ordinal, path in enumerate(preflight_manifests):
        binding = _read_json(path).get("runtime_binding_evidence", {})
        lifecycle = binding.get("cuda_memory_statistics_lifecycle", {})
        checks.equal(
            f"preflight.{ordinal}.resolved_device",
            binding.get("resolved_device"),
            "cuda:0",
        )
        checks.equal(
            f"preflight.{ordinal}.cuda_lifecycle_contract_version",
            lifecycle.get("contract_version"),
            EXPECTED_CUDA_MEMORY_LIFECYCLE_VERSION,
        )
        checks.equal(
            f"preflight.{ordinal}.cuda_lifecycle_logical_index",
            lifecycle.get("logical_device_index"),
            0,
        )
        checks.true(
            f"preflight.{ordinal}.cuda_lifecycle_initialized_before",
            type(lifecycle.get("initialized_before")) is bool,
            lifecycle.get("initialized_before"),
        )
        checks.equal(
            f"preflight.{ordinal}.cuda_lifecycle_initialized_after",
            lifecycle.get("initialized_after"),
            True,
        )

    report_paths = {
        "ssl": tuple(sorted((root / "cells" / "ssl").rglob("training_report.json"))),
        "downstream": tuple(
            sorted((root / "cells" / "downstream").rglob("training_report.json"))
        ),
    }
    checks.equal("runtime.ssl_report_count", len(report_paths["ssl"]), 1)
    checks.equal("runtime.downstream_report_count", len(report_paths["downstream"]), 3)
    for kind, paths in report_paths.items():
        for ordinal, path in enumerate(paths):
            runtime_report = _read_json(path)
            device = runtime_report.get("device", {})
            checks.equal(
                f"runtime.{kind}.{ordinal}.resolved_device",
                device.get("resolved_device"),
                "cuda:0",
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.cuda_available",
                device.get("cuda_available"),
                True,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.cuda_logical_device_index",
                device.get("cuda_logical_device_index"),
                0,
            )
            lifecycle = device.get(
                "cuda_memory_statistics_lifecycle", {}
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.cuda_lifecycle_contract_version",
                lifecycle.get("contract_version"),
                EXPECTED_CUDA_MEMORY_LIFECYCLE_VERSION,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.cuda_lifecycle_logical_index",
                lifecycle.get("logical_device_index"),
                0,
            )
            checks.true(
                f"runtime.{kind}.{ordinal}.cuda_lifecycle_initialized_before",
                type(lifecycle.get("initialized_before")) is bool,
                lifecycle.get("initialized_before"),
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.cuda_lifecycle_initialized_after",
                lifecycle.get("initialized_after"),
                True,
            )
            runtime_device_name = device.get("cuda_device_name")
            checks.true(
                f"runtime.{kind}.{ordinal}.cuda_device_name",
                isinstance(runtime_device_name, str)
                and expected_device_name in runtime_device_name,
                runtime_device_name,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.amp_enabled",
                runtime_report.get("amp_enabled"),
                True,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.scaler_enabled",
                runtime_report.get("scaler_enabled"),
                True,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.validation_dataset_ids",
                _dataset_ids(
                    runtime_report.get("validation_membership", {}).get(
                        "dataset_counts"
                    )
                ),
                EXPECTED_DATASET_IDS,
            )
            runtime_validation = runtime_report.get(
                "validation_membership", {}
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.validation_dataset_count_total",
                _dataset_count_total(runtime_validation.get("dataset_counts")),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.validation_selected_count",
                runtime_validation.get("selected_count"),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.validation_subset_limit",
                runtime_validation.get("subset_limit"),
                EXPECTED_VALIDATION_SAMPLES,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.validation_membership_fingerprint",
                runtime_validation.get("membership_fingerprint"),
                validation_fingerprint,
            )
            checks.equal(
                f"runtime.{kind}.{ordinal}.logical_update_budget_complete",
                runtime_report.get("logical_update_budget_complete"),
                True,
            )
            if kind == "ssl":
                accounting = runtime_report.get("accounting", {})
                checks.equal(
                    "runtime.ssl.attempted_updates",
                    accounting.get("optimizer_step_attempt_count"),
                    1,
                )
                checks.equal(
                    "runtime.ssl.applied_updates",
                    accounting.get("optimizer_step_applied_count"),
                    1,
                )
                checks.equal(
                    "runtime.ssl.skipped_updates",
                    accounting.get("optimizer_step_skipped_count"),
                    0,
                )
            else:
                checks.equal(
                    f"runtime.downstream.{ordinal}.attempted_updates",
                    runtime_report.get("optimizer_step_attempt_count"),
                    1,
                )
                checks.equal(
                    f"runtime.downstream.{ordinal}.applied_updates",
                    runtime_report.get("optimizer_step_applied_count"),
                    1,
                )
                checks.equal(
                    f"runtime.downstream.{ordinal}.skipped_updates",
                    runtime_report.get("optimizer_step_skipped_count"),
                    0,
                )

    resolved_configs = tuple(
        sorted((root / "cells" / "ssl").rglob("resolved_config.json"))
    ) + tuple(
        sorted(
            (root / "cells" / "downstream").rglob("resolved_config.json")
        )
    )
    checks.equal("runtime.resolved_training_config_count", len(resolved_configs), 4)
    for ordinal, path in enumerate(resolved_configs):
        config = _read_json(path)
        device = config.get("device", {})
        checks.equal(f"resolved_config.{ordinal}.device", device.get("name"), "cuda:0")
        checks.equal(f"resolved_config.{ordinal}.amp", device.get("amp"), True)
        checks.equal(
            f"resolved_config.{ordinal}.amp_dtype",
            device.get("amp_dtype"),
            "float16",
        )

    downstream_configs = tuple(
        sorted(
            (root / "cells" / "downstream").rglob(
                "resolved_config.json"
            )
        )
    )
    checks.equal(
        "runtime.resolved_downstream_config_count",
        len(downstream_configs),
        3,
    )
    for ordinal, path in enumerate(downstream_configs):
        config = _read_json(path)
        checks.equal(
            f"downstream_config.{ordinal}.validation_epoch_size",
            config.get("data", {}).get("validation_epoch_size"),
            EXPECTED_VALIDATION_SAMPLES,
        )

    evaluation_configs = tuple(
        sorted(
            (root / "cells" / "evaluation").rglob(
                "resolved_evaluation_config.json"
            )
        )
    )
    checks.equal("runtime.resolved_evaluation_config_count", len(evaluation_configs), 3)
    for ordinal, path in enumerate(evaluation_configs):
        config = _read_json(path)
        device = config.get("device", {})
        checks.equal(
            f"evaluation_config.{ordinal}.device",
            device.get("name"),
            "cuda:0",
        )
        checks.equal(f"evaluation_config.{ordinal}.amp", device.get("amp"), True)
        checks.equal(
            f"evaluation_config.{ordinal}.amp_dtype",
            device.get("amp_dtype"),
            "float16",
        )
        checks.equal(
            f"evaluation_config.{ordinal}.max_evaluation_samples",
            config.get("data", {}).get("max_evaluation_samples"),
            EXPECTED_VALIDATION_SAMPLES,
        )
        engine = path.parent
        evaluation_report = _read_json(engine / "evaluation_report.json")
        checks.equal(
            f"evaluation_report.{ordinal}.sample_count",
            evaluation_report.get("sample_count"),
            EXPECTED_VALIDATION_SAMPLES,
        )
        data_verification = evaluation_report.get("data_verification", {})
        checks.equal(
            f"evaluation_report.{ordinal}.data_verification",
            data_verification.get("verified"),
            True,
        )
        checks.true(
            f"evaluation_report.{ordinal}.membership_field_matched",
            "validation_membership_fingerprint"
            in data_verification.get("matched_fields", []),
            data_verification.get("matched_fields"),
        )
        metrics = _read_json(engine / "metrics.json")
        checks.equal(
            f"evaluation_report.{ordinal}.validation_membership_fingerprint",
            metrics.get("bindings", {}).get(
                "evaluation_membership_fingerprint"
            ),
            validation_fingerprint,
        )
        checks.equal(
            f"evaluation_report.{ordinal}.metrics_sample_count",
            metrics.get("counts", {}).get("sample_count"),
            EXPECTED_VALIDATION_SAMPLES,
        )
        checks.equal(
            f"evaluation_report.{ordinal}.dataset_ids",
            _dataset_ids(metrics.get("dataset_sample_counts")),
            EXPECTED_DATASET_IDS,
        )
        checks.equal(
            f"evaluation_report.{ordinal}.dataset_sample_count_total",
            _dataset_count_total(metrics.get("dataset_sample_counts")),
            EXPECTED_VALIDATION_SAMPLES,
        )
        checkpoint_evidence = _read_json(
            engine / "checkpoint_evidence.json"
        )
        checks.equal(
            f"evaluation_report.{ordinal}.checkpoint_validation_fingerprint",
            checkpoint_evidence.get("training_data_fingerprints", {}).get(
                "validation_membership_fingerprint"
            ),
            validation_fingerprint,
        )

    return {
        "status": "passed" if not checks.failures else "failed",
        "run_label": "production-format real-corpus bounded smoke",
        "output_root": str(root),
        "expected_git_sha": expected_sha,
        "expected_device_name_contains": expected_device_name,
        "verified_runtime_binding_cell_count": runtime_verified,
        "checkpoint_to_evaluation_verified_cell_count": evaluation_verified,
        "test_split_opened": False if not checks.failures else None,
        "failures": checks.failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--invocation-config", required=True, type=Path)
    parser.add_argument("--expected-device-name", default="RTX 3090")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = verify_gate(
        args.output_root,
        expected_sha=args.expected_sha,
        invocation_config=args.invocation_config,
        expected_device_name=args.expected_device_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
