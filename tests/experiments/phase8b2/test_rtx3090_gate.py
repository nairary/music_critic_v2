from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.verify_phase8b2a_rtx3090_bounded_smoke import verify_gate


EXACT_SHA = "a" * 40
VALIDATION_FINGERPRINT = "f" * 64
DATASET_COUNTS = {"hooktheory": 64, "pop909_cl": 64}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _schedule_row(kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "slots": [
            {"dataset_id": "hooktheory"},
            {"dataset_id": "pop909_cl"},
        ],
        "data_semantic_projection": {
            "train_composition": {
                "dataset_counts": [["hooktheory", 1], ["pop909_cl", 1]]
            },
            "validation_membership": {
                "dataset_counts": [
                    ["hooktheory", 64],
                    ["pop909_cl", 64],
                ],
                "membership_fingerprint": VALIDATION_FINGERPRINT,
                "selected_count": 128,
                "subset_limit": 128,
            },
        },
    }


def _training_report(*, ssl: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "amp_enabled": True,
        "scaler_enabled": True,
        "logical_update_budget_complete": True,
        "validation_membership": {
            "dataset_counts": DATASET_COUNTS,
            "membership_fingerprint": VALIDATION_FINGERPRINT,
            "selected_count": 128,
            "subset_limit": 128,
        },
        "device": {
            "resolved_device": "cuda:0",
            "cuda_available": True,
            "cuda_logical_device_index": 0,
            "cuda_device_name": "NVIDIA GeForce RTX 3090",
        },
    }
    if ssl:
        report["accounting"] = {
            "optimizer_step_attempt_count": 1,
            "optimizer_step_applied_count": 1,
            "optimizer_step_skipped_count": 0,
        }
    else:
        report.update(
            {
                "optimizer_step_attempt_count": 1,
                "optimizer_step_applied_count": 1,
                "optimizer_step_skipped_count": 0,
            }
        )
    return report


def _gate_fixture(root: Path) -> tuple[Path, Path]:
    output = root / "run"
    invocation = root / "invocation.json"
    protocol = {
        "seeds": [17],
        "variants": ["phase7a_control"],
        "amp_device_config": {
            "name": "cuda:0",
            "amp": True,
            "amp_dtype": "float16",
        },
        "runtime_execution_config": {
            "ssl_attempted_logical_updates": 1,
            "downstream_attempted_logical_updates": 1,
            "validation_samples": 128,
        },
        "data": {
            "validation_membership_fingerprint": VALIDATION_FINGERPRINT
        },
        "test_unlock_state": {
            "acknowledged": False,
            "unlocked": False,
            "test_inference_performed": False,
            "test_targets_accessed": False,
            "test_metrics_accessed": False,
        },
    }
    _write(
        output / "plan.json",
        {
            "protocol": protocol,
            "ssl_cells": [{}],
            "downstream_cells": [{}, {}, {}],
            "evaluation_cells": [{}, {}, {}],
            "runtime_paths": {
                "index_paths": ["hook.index.json", "pop.index.json"],
                "cache_roots": ["hook-cache", "pop-cache"],
                "split_manifest": "global.split.json",
            },
            "data_attestation": {
                "validation_membership": {
                    "dataset_counts": DATASET_COUNTS,
                    "membership_fingerprint": VALIDATION_FINGERPRINT,
                    "selected_count": 128,
                    "selected_identities": [
                        [
                            "hooktheory" if ordinal < 64 else "pop909_cl",
                            f"validation-piece-{ordinal:03d}",
                        ]
                        for ordinal in range(128)
                    ],
                    "subset_limit": 128,
                },
                "test_inference_performed": False,
                "test_targets_accessed": False,
                "test_metrics_accessed": False,
                "test_membership_summary": {"selected_count": 2},
            },
        },
    )
    schedule = {
        "ssl": [_schedule_row("ssl")],
        "downstream": [_schedule_row("downstream")],
    }
    _write(output / "actual_sample_schedule.json", schedule)
    _write(output / "final_bundle" / "comparison_protocol.json", protocol)
    _write(
        output / "final_bundle" / "final_comparison_report.json",
        {
            "status": "complete",
            "executed_cell_count": 8,
            "expected_cell_count": 8,
            "verified_runtime_binding_cell_count": 8,
            "checkpoint_to_evaluation_verified_cell_count": 3,
            "test_inference_performed": False,
            "test_targets_accessed": False,
            "test_metrics_accessed": False,
            "bounded_results_are_scientific_superiority_evidence": False,
            "ssl_compute_totals": {
                "attempted_updates": 1,
                "applied_updates": 1,
                "skipped_updates": 0,
            },
        },
    )
    _write(
        output / "final_bundle" / "compute_accounting.json",
        {
            "cells": [
                {
                    "logical_updates": 1,
                    "optimizer_updates_applied": 1,
                    "optimizer_updates_skipped": 0,
                    "cuda_peak_memory": {
                        "available": True,
                        "cuda_logical_device_index": 0,
                        "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048,
                    },
                }
            ]
        },
    )
    _write(
        output / "final_bundle" / "run_manifest.json",
        {
            "repository": {"git_sha": EXACT_SHA, "dirty": True},
            "environment": {
                "device": "cuda:0",
                "cuda_available": True,
                "cuda_logical_device_index": 0,
                "cuda_device_name": "NVIDIA GeForce RTX 3090",
            },
        },
    )
    _write(
        invocation,
        {
            "run_label": "production-format real-corpus bounded smoke",
            "comparison": "bounded_acceptance",
            "expected_git_sha": EXACT_SHA,
            "variants": ["phase7a_control"],
            "seeds": [17],
            "ssl_optimizer_steps": 1,
            "downstream_optimizer_steps": 1,
            "validation_samples": 128,
            "device": "cuda:0",
            "amp": True,
            "amp_dtype": "float16",
            "output_root": str(output.resolve()),
            "index_paths": ["hook.index.json", "pop.index.json"],
            "cache_roots": ["hook-cache", "pop-cache"],
            "split_manifest": "global.split.json",
        },
    )

    cell_roots = [
        output / "cells" / "ssl" / "control",
        output / "cells" / "encoder_export" / "control",
        *(
            output / "cells" / "downstream" / mode
            for mode in ("frozen_probe", "full_finetune", "supervised_scratch")
        ),
        *(
            output / "cells" / "evaluation" / mode
            for mode in ("frozen_probe", "full_finetune", "supervised_scratch")
        ),
    ]
    for cell_root in cell_roots:
        evaluation = "evaluation" in cell_root.parts
        _write(
            cell_root / "cell_manifest.json",
            {
                "runtime_binding_evidence": {
                    "verified": True,
                    "checkpoint_to_evaluation_data_verified": evaluation,
                }
            },
        )
    _write(
        output / "cells" / "ssl" / "control" / "engine" / "training_report.json",
        _training_report(ssl=True),
    )
    _write(
        output / "cells" / "ssl" / "control" / "engine" / "resolved_config.json",
        {
            "data": {"validation_epoch_size": 128},
            "device": {"name": "cuda:0", "amp": True, "amp_dtype": "float16"},
        },
    )
    for mode in ("frozen_probe", "full_finetune", "supervised_scratch"):
        engine = output / "cells" / "downstream" / mode / "engine"
        _write(engine / "training_report.json", _training_report(ssl=False))
        _write(
            engine / "resolved_config.json",
            {
                "data": {"validation_epoch_size": 128},
                "device": {
                    "name": "cuda:0",
                    "amp": True,
                    "amp_dtype": "float16",
                },
            },
        )
        evaluation_engine = (
            output / "cells" / "evaluation" / mode / "engine"
        )
        _write(
            evaluation_engine / "resolved_evaluation_config.json",
            {
                "data": {"max_evaluation_samples": 128},
                "device": {
                    "name": "cuda:0",
                    "amp": True,
                    "amp_dtype": "float16",
                },
            },
        )
        _write(
            evaluation_engine / "evaluation_report.json",
            {
                "sample_count": 128,
                "data_verification": {
                    "verified": True,
                    "matched_fields": [
                        "validation_membership_fingerprint"
                    ],
                },
            },
        )
        _write(
            evaluation_engine / "metrics.json",
            {
                "bindings": {
                    "evaluation_membership_fingerprint": (
                        VALIDATION_FINGERPRINT
                    )
                },
                "counts": {"sample_count": 128},
                "dataset_sample_counts": DATASET_COUNTS,
            },
        )
        _write(
            evaluation_engine / "checkpoint_evidence.json",
            {
                "training_data_fingerprints": {
                    "validation_membership_fingerprint": (
                        VALIDATION_FINGERPRINT
                    )
                }
            },
        )
    return output, invocation


def test_rtx3090_verifier_accepts_complete_bounded_evidence(
    tmp_path: Path,
) -> None:
    output, invocation = _gate_fixture(tmp_path)

    result = verify_gate(
        output,
        expected_sha=EXACT_SHA,
        invocation_config=invocation,
    )

    assert result["status"] == "passed"
    assert result["verified_runtime_binding_cell_count"] == 8
    assert result["checkpoint_to_evaluation_verified_cell_count"] == 3
    assert result["test_split_opened"] is False


def test_rtx3090_verifier_rejects_hidden_cpu_fallback(tmp_path: Path) -> None:
    output, invocation = _gate_fixture(tmp_path)
    manifest_path = output / "final_bundle" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["device"] = "cpu"
    _write(manifest_path, manifest)

    result = verify_gate(
        output,
        expected_sha=EXACT_SHA,
        invocation_config=invocation,
    )

    assert result["status"] == "failed"
    assert {row["check"] for row in result["failures"]} >= {
        "manifest.device"
    }


@pytest.mark.parametrize("validation_samples", [0, 127, 129])
def test_rtx3090_verifier_rejects_wrong_invocation_validation_bound(
    tmp_path: Path,
    validation_samples: int,
) -> None:
    output, invocation = _gate_fixture(tmp_path)
    payload = json.loads(invocation.read_text(encoding="utf-8"))
    payload["validation_samples"] = validation_samples
    _write(invocation, payload)
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["protocol"]["runtime_execution_config"][
        "validation_samples"
    ] = validation_samples
    _write(plan_path, plan)
    _write(
        output / "final_bundle" / "comparison_protocol.json",
        plan["protocol"],
    )

    result = verify_gate(
        output,
        expected_sha=EXACT_SHA,
        invocation_config=invocation,
    )

    assert result["status"] == "failed"
    assert {row["check"] for row in result["failures"]} >= {
        "invocation.validation_samples",
        "plan.validation_samples",
    }


@pytest.mark.parametrize(
    ("relative_path", "field_path", "expected_check"),
    [
        (
            Path("cells/downstream/frozen_probe/engine/training_report.json"),
            ("validation_membership", "membership_fingerprint"),
            "runtime.downstream.0.validation_membership_fingerprint",
        ),
        (
            Path("cells/evaluation/frozen_probe/engine/metrics.json"),
            ("bindings", "evaluation_membership_fingerprint"),
            "evaluation_report.0.validation_membership_fingerprint",
        ),
    ],
)
def test_rtx3090_verifier_rejects_validation_membership_fingerprint_mismatch(
    tmp_path: Path,
    relative_path: Path,
    field_path: tuple[str, str],
    expected_check: str,
) -> None:
    output, invocation = _gate_fixture(tmp_path)
    artifact = output / relative_path
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload[field_path[0]][field_path[1]] = "e" * 64
    _write(artifact, payload)

    result = verify_gate(
        output,
        expected_sha=EXACT_SHA,
        invocation_config=invocation,
    )

    assert result["status"] == "failed"
    assert {row["check"] for row in result["failures"]} >= {
        expected_check
    }


def test_published_gate_script_is_subshell_safe_and_untracked_tolerant() -> None:
    script = Path("scripts/run_phase8b2a_rtx3090_bounded_smoke.sh")
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert syntax.returncode == 0, syntax.stdout + syntax.stderr
    source = script.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "\n(\nset -euo pipefail\n" in source
    assert "git diff --quiet" in source
    assert "git diff --cached --quiet" in source
    assert "git ls-files --others --exclude-standard" in source
    assert "comparison=bounded_acceptance" in source
    assert "comparison.variants=[phase7a_control]" in source
    assert "comparison.seeds=[17]" in source
    assert "comparison.validation_samples=128" in source
    assert '"validation_samples":128' in source
    assert "comparison=production_pilot" not in source
    assert "git status --porcelain" not in source
    assert "rm " not in source
    assert "mv " not in source
