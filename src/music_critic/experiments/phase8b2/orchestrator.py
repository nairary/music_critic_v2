"""Dependency-aware, resumable Phase 8B.2A matrix execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

import torch

from music_critic.experiments.phase8b2.accounting import (
    compute_accounting_from_ssl_report,
)
from music_critic.experiments.phase8b2.artifacts import (
    REQUIRED_ARTIFACTS,
    environment_evidence,
    file_sha256,
    read_json,
    repository_evidence,
    write_complete_artifact_bundle,
    write_json_once,
)
from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_ARTIFACT_CONTRACT_VERSION,
    Phase8B2ContractError,
    fingerprint,
)
from music_critic.experiments.phase8b2.runner import (
    official_downstream_overrides,
    official_evaluation_overrides,
    official_ssl_cell_overrides,
)
from music_critic.experiments.phase8b2.selection import (
    select_validation_checkpoint,
)
from music_critic.experiments.phase8b2.statistics import (
    aggregate_piece_sufficient_statistics,
)


MATRIX_RUNNER_CONTRACT_VERSION = "1.1.0"
CELL_MANIFEST_CONTRACT_VERSION = "1.1.0"


def _read_json_dict(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise Phase8B2ContractError(
            f"phase8b2.runner.json_mapping_required:{path.name}"
        )
    return value


def _cell_path(root: Path, cell_id: str) -> Path:
    parts = cell_id.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise Phase8B2ContractError("phase8b2.runner.cell_id_invalid")
    return root / "cells" / Path(*parts)


def _staging_path(root: Path, cell_id: str) -> Path:
    return root / ".staging" / cell_id.replace("/", "__")


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "cell_manifest.json"
    }


def _validate_published_cell(
    directory: Path,
    *,
    cell_id: str,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    manifest = _read_json_dict(directory / "cell_manifest.json")
    if (
        manifest.get("cell_manifest_contract_version")
        != CELL_MANIFEST_CONTRACT_VERSION
        or manifest.get("cell_id") != cell_id
        or manifest.get("protocol_fingerprint") != protocol_fingerprint
        or manifest.get("status") != "complete"
        or manifest.get("artifact_sha256") != _artifact_hashes(directory)
    ):
        raise Phase8B2ContractError(
            f"phase8b2.runner.stale_or_incomplete_cell:{cell_id}"
        )
    return manifest


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _execute_cell(
    root: Path,
    *,
    cell_id: str,
    protocol_fingerprint: str,
    command: list[str],
    validate: Callable[[Path], dict[str, object]],
) -> dict[str, Any]:
    published = _cell_path(root, cell_id)
    if published.exists():
        return _validate_published_cell(
            published,
            cell_id=cell_id,
            protocol_fingerprint=protocol_fingerprint,
        )
    staging = _staging_path(root, cell_id)
    if staging.exists():
        raise Phase8B2ContractError(
            f"phase8b2.runner.incomplete_staging_requires_inspection:{cell_id}"
        )
    staging.mkdir(parents=True)
    write_json_once(staging / "command.json", command)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    process = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_text(staging / "stdout.log", process.stdout)
    _write_text(staging / "stderr.log", process.stderr)
    process_evidence = {
        "returncode": process.returncode,
        "argv": command,
        "shell": False,
    }
    write_json_once(staging / "process.json", process_evidence)
    if process.returncode:
        write_json_once(
            staging / "failure.json",
            {
                "cell_id": cell_id,
                "status": "failed",
                **process_evidence,
            },
        )
        raise Phase8B2ContractError(
            f"phase8b2.runner.cell_failed:{cell_id}:exit={process.returncode}"
        )
    binding_evidence = validate(staging)
    manifest = {
        "cell_manifest_contract_version": CELL_MANIFEST_CONTRACT_VERSION,
        "cell_id": cell_id,
        "protocol_fingerprint": protocol_fingerprint,
        "status": "complete",
        "runtime_binding_evidence": binding_evidence,
        "artifact_sha256": _artifact_hashes(staging),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json_once(staging / "cell_manifest.json", manifest)
    published.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, published)
    return manifest


def _assert_mapping_subset(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    category: str,
) -> None:
    mismatched = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if mismatched:
        raise Phase8B2ContractError(
            f"phase8b2.runner.{category}_runtime_mismatch:"
            + ",".join(sorted(mismatched))
        )


def _ssl_runtime_binding(
    plan: Mapping[str, Any], cell: Mapping[str, Any], staging: Path
) -> dict[str, object]:
    engine = staging / "engine"
    report = _read_json_dict(engine / "training_report.json")
    resolved = _read_json_dict(engine / "resolved_config.json")
    protocol = plan["protocol"]
    if (
        report.get("actual_sample_schedule_verified") is not True
        or report.get("logical_update_budget_complete") is not True
        or report.get("phase8b2_schedule", {}).get("protocol_fingerprint")
        != protocol["fingerprint"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.ssl_scientific_evidence_invalid"
        )
    expected_schedule = next(
        row
        for row in plan["actual_sample_schedule"]["ssl"]
        if row["seed"] == cell["seed"]
    )
    if (
        fingerprint(report.get("fingerprints"))
        != fingerprint(expected_schedule["runtime_data_fingerprints"])
        or fingerprint(report.get("validation_membership"))
        != fingerprint(expected_schedule["validation_membership"])
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.ssl_data_attestation_mismatch"
        )
    _assert_mapping_subset(
        resolved["model"], protocol["encoder_model_config"], category="ssl_model"
    )
    _assert_mapping_subset(
        resolved["ssl"], protocol["ssl_objective_config"], category="ssl_objective"
    )
    _assert_mapping_subset(
        resolved["optimizer"], protocol["optimizer_config"], category="ssl_optimizer"
    )
    _assert_mapping_subset(
        resolved["scheduler"], protocol["scheduler_config"], category="ssl_scheduler"
    )
    _assert_mapping_subset(
        resolved["device"], protocol["amp_device_config"], category="ssl_device"
    )
    expected_data = {
        "batch_size": protocol["compute"]["batch_size"],
        "workers": protocol["data"]["workers"],
        "validation_epoch_size": protocol["runtime_execution_config"][
            "validation_samples"
        ],
        "validation_seed": protocol["runtime_execution_config"][
            "fixed_validation_seed"
        ],
        "mixture_weights": dict(protocol["data"]["mixture_weights"]),
    }
    _assert_mapping_subset(resolved["data"], expected_data, category="ssl_data")
    schedule = cell["schedule"]
    expected_steps_per_epoch = min(
        protocol["runtime_execution_config"]["optimizer_steps_per_epoch"],
        schedule["logical_updates"],
    )
    _assert_mapping_subset(
        resolved["experiment"],
        {
            "steps": schedule["logical_updates"],
            "optimizer_steps_per_epoch": expected_steps_per_epoch,
            "validation_interval": protocol["runtime_execution_config"][
                "validation_interval_epochs"
            ],
        },
        category="ssl_experiment",
    )
    expected_accounting = {
        "optimizer_step_attempt_count": schedule["logical_updates"],
        "optimizer_step_applied_count": schedule["logical_updates"],
        "optimizer_step_skipped_count": 0,
        "encoder_forward_count": schedule["encoder_forward_count"],
        "sample_count": schedule["raw_sample_exposures"],
    }
    _assert_mapping_subset(
        report["accounting"], expected_accounting, category="ssl_accounting"
    )
    return {
        "verified": True,
        "resolved_runtime_fingerprint": fingerprint(resolved),
        "protocol_fingerprint": protocol["fingerprint"],
        "sample_schedule_fingerprint": report[
            "observed_ssl_sample_schedule_fingerprint"
        ],
        "encoder_counter_kind": "instrumented_encoder_method_invocations",
        "expected_accounting": expected_accounting,
    }


def _downstream_runtime_binding(
    plan: Mapping[str, Any], cell: Mapping[str, Any], staging: Path
) -> dict[str, object]:
    engine = staging / "engine"
    report = _read_json_dict(engine / "training_report.json")
    resolved = _read_json_dict(engine / "resolved_config.json")
    protocol = plan["protocol"]
    if (
        report.get("actual_sample_schedule_verified") is not True
        or report.get("logical_update_budget_complete") is not True
        or report.get("optimizer_step_skipped_count") != 0
        or report.get("phase8b2_transfer", {}).get(
            "comparison_protocol_fingerprint"
        )
        != protocol["fingerprint"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.downstream_scientific_evidence_invalid"
        )
    expected_schedule = next(
        row
        for row in plan["actual_sample_schedule"]["downstream"]
        if row["seed"] == cell["seed"]
    )
    if (
        fingerprint(report.get("fingerprints"))
        != fingerprint(expected_schedule["runtime_data_fingerprints"])
        or fingerprint(report.get("validation_membership"))
        != fingerprint(expected_schedule["validation_membership"])
        or report.get("validation_membership", {}).get(
            "membership_fingerprint"
        )
        != protocol["data"]["validation_membership_fingerprint"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.downstream_data_attestation_mismatch"
        )
    composition = expected_schedule["runtime_data_composition"]
    if (
        sum(composition["train_dataset_counts"].values())
        != protocol["data"]["actual_train_size"]
        or report["validation_membership"]["selected_count"]
        != protocol["data"]["actual_validation_size"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.downstream_data_size_mismatch"
    )
    _assert_mapping_subset(
        resolved["model"],
        protocol["encoder_model_config"],
        category="downstream_model",
    )
    _assert_mapping_subset(
        resolved["optimizer"],
        protocol["optimizer_config"],
        category="downstream_optimizer",
    )
    _assert_mapping_subset(
        resolved["scheduler"],
        protocol["scheduler_config"],
        category="downstream_scheduler",
    )
    _assert_mapping_subset(
        resolved["device"],
        protocol["amp_device_config"],
        category="downstream_device",
    )
    runtime = protocol["runtime_execution_config"]
    expected_steps_per_epoch = min(
        runtime["optimizer_steps_per_epoch"],
        protocol["downstream_optimizer_steps"],
    )
    _assert_mapping_subset(
        resolved["experiment"],
        {
            "steps": protocol["downstream_optimizer_steps"],
            "optimizer_steps_per_epoch": expected_steps_per_epoch,
            "validation_interval": runtime[
                "validation_interval_epochs"
            ],
        },
        category="downstream_experiment",
    )
    _assert_mapping_subset(
        resolved["data"],
        {
            "batch_size": protocol["compute"]["batch_size"],
            "workers": protocol["data"]["workers"],
            "validation_epoch_size": runtime["validation_samples"],
            "validation_seed": runtime["fixed_validation_seed"],
            "mixture_weights": dict(protocol["data"]["mixture_weights"]),
        },
        category="downstream_data",
    )
    expected_tasks = [row["task_id"] for row in protocol["downstream_tasks"]]
    if resolved.get("downstream_task_ids") != expected_tasks:
        raise Phase8B2ContractError(
            "phase8b2.runner.downstream_task_runtime_mismatch"
        )
    return {
        "verified": True,
        "resolved_runtime_fingerprint": fingerprint(resolved),
        "protocol_fingerprint": protocol["fingerprint"],
        "sample_schedule_fingerprint": report[
            "observed_downstream_schedule_fingerprint"
        ],
        "checkpoint": report["last_checkpoint"],
    }


def _evaluation_runtime_binding(
    plan: Mapping[str, Any], cell: Mapping[str, Any], staging: Path
) -> dict[str, object]:
    engine = staging / "engine"
    report = _read_json_dict(engine / "evaluation_report.json")
    resolved = _read_json_dict(engine / "resolved_evaluation_config.json")
    checkpoint = _read_json_dict(engine / "checkpoint_evidence.json")
    metrics = _read_json_dict(engine / "metrics.json")
    protocol = plan["protocol"]
    if (
        report.get("status") != "completed"
        or report.get("split") != "validation"
        or report.get("data_verification", {}).get("verified") is not True
        or report.get("data_verification", {}).get("matched_fields") is None
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.checkpoint_evaluation_verification_failed"
        )
    if report.get("sample_count") != protocol["data"][
        "actual_validation_size"
    ]:
        raise Phase8B2ContractError(
            "phase8b2.runner.evaluation_data_size_mismatch"
        )
    if resolved.get("seed") != cell["evaluation_seed"]:
        raise Phase8B2ContractError(
            "phase8b2.runner.evaluation_seed_runtime_mismatch"
        )
    expected_tasks = [row["task_id"] for row in protocol["downstream_tasks"]]
    if resolved.get("downstream_task_ids") != expected_tasks:
        raise Phase8B2ContractError(
            "phase8b2.runner.evaluation_task_runtime_mismatch"
        )
    runtime = protocol["runtime_execution_config"]
    _assert_mapping_subset(
        resolved["device"],
        protocol["amp_device_config"],
        category="evaluation_device",
    )
    _assert_mapping_subset(
        resolved["data"],
        {
            "batch_size": protocol["compute"]["batch_size"],
            "workers": protocol["data"]["workers"],
            "max_evaluation_samples": runtime["validation_samples"],
            "validation_seed": runtime["fixed_validation_seed"],
        },
        category="evaluation_data",
    )
    bindings = metrics.get("bindings", {})
    expected_bindings = {
        "split_manifest_fingerprint": protocol["data"][
            "split_manifest_fingerprint"
        ],
        "train_membership_fingerprint": protocol["data"][
            "train_membership_fingerprint"
        ],
        "evaluation_membership_fingerprint": protocol["data"][
            "validation_membership_fingerprint"
        ],
    }
    _assert_mapping_subset(
        bindings, expected_bindings, category="evaluation_data_attestation"
    )
    if plan["runtime_paths"]["index_paths"]:
        if (
            dict(bindings.get("index_fingerprints", []))
            != dict(protocol["data"]["dataset_indices"])
            or dict(bindings.get("cache_fingerprints", []))
            != dict(protocol["data"]["cache_identities"])
        ):
            raise Phase8B2ContractError(
                "phase8b2.runner.evaluation_index_cache_attestation_mismatch"
            )
    return {
        "verified": True,
        "resolved_runtime_fingerprint": fingerprint(resolved),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_to_evaluation_data_verified": True,
        "membership_mismatch": False,
    }


def _export_binding(
    protocol_fingerprint: str,
    source_checkpoint_sha256: str,
    staging: Path,
) -> dict[str, object]:
    export_path = staging / "engine" / "encoder.pt"
    if not export_path.is_file():
        raise Phase8B2ContractError(
            "phase8b2.runner.encoder_export_missing"
        )
    stdout = (staging / "stdout.log").read_text(encoding="utf-8").strip()
    evidence = json.loads(stdout.splitlines()[-1])
    if evidence.get("protocol_fingerprint") != protocol_fingerprint:
        raise Phase8B2ContractError(
            "phase8b2.runner.encoder_export_protocol_mismatch"
        )
    return {
        "verified": True,
        "encoder_export_sha256": file_sha256(export_path),
        "source_ssl_checkpoint_sha256": source_checkpoint_sha256,
        **evidence,
    }


def _preflight_binding(
    protocol_fingerprint: str, staging: Path
) -> dict[str, object]:
    stdout = (staging / "stdout.log").read_text(encoding="utf-8").strip()
    evidence = json.loads(stdout.splitlines()[-1])
    if (
        evidence.get("status") != "passed"
        or evidence.get("protocol_fingerprint") != protocol_fingerprint
        or evidence.get("objective_available_for_every_planned_batch")
        is not True
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.preflight_evidence_invalid"
        )
    return evidence


def _write_overrides(staging: Path, overrides: list[str]) -> Path:
    path = staging.parent / f".{staging.name}.overrides.json"
    if path.exists():
        existing = read_json(path)
        if existing != overrides:
            raise Phase8B2ContractError(
                "phase8b2.runner.override_manifest_stale"
            )
    else:
        write_json_once(path, overrides)
    return path


def _engine_path(root: Path, cell_id: str) -> Path:
    return _cell_path(root, cell_id) / "engine"


def _maybe_stop(config: Mapping[str, Any], stage: str) -> bool:
    value = str(config.get("stop_after_stage", ""))
    return bool(value and value == stage)


def _run_matrix_cells(
    config: Mapping[str, Any], plan: Mapping[str, Any], root: Path
) -> dict[str, object]:
    protocol_fingerprint = plan["protocol"]["fingerprint"]
    schedule_path = root / "actual_sample_schedule.json"
    executed = 0

    for cell in plan["ssl_cells"]:
        cell_id = "preflight/" + cell["cell_id"].removeprefix("ssl/")
        staging = _staging_path(root, cell_id)
        overrides = official_ssl_cell_overrides(
            plan,
            cell["cell_id"],
            str(staging / "unused-engine-output"),
            actual_sample_schedule_path=str(schedule_path.resolve()),
        )
        overrides_path = _write_overrides(staging, overrides)
        _execute_cell(
            root,
            cell_id=cell_id,
            protocol_fingerprint=protocol_fingerprint,
            command=[
                sys.executable,
                "-m",
                "music_critic.experiments.phase8b2.worker",
                "preflight-ssl",
                str(overrides_path),
            ],
            validate=lambda directory: _preflight_binding(
                protocol_fingerprint, directory
            ),
        )
    if _maybe_stop(config, "preflight"):
        return {"status": "stopped", "stage": "preflight"}

    for cell in plan["ssl_cells"]:
        staging = _staging_path(root, cell["cell_id"])
        overrides = official_ssl_cell_overrides(
            plan,
            cell["cell_id"],
            str(staging / "engine"),
            actual_sample_schedule_path=str(schedule_path.resolve()),
        )
        _execute_cell(
            root,
            cell_id=cell["cell_id"],
            protocol_fingerprint=protocol_fingerprint,
            command=[sys.executable, "-m", "music_critic.ssl.run", *overrides],
            validate=lambda directory, row=cell: _ssl_runtime_binding(
                plan, row, directory
            ),
        )
        executed += 1
        if int(config.get("fail_after_cell", 0)) == executed:
            raise Phase8B2ContractError(
                "phase8b2.runner.injected_interruption_after_published_cell"
            )
    if _maybe_stop(config, "ssl"):
        return {"status": "stopped", "stage": "ssl"}

    ssl_by_id = {row["cell_id"]: row for row in plan["ssl_cells"]}
    for cell in plan["encoder_export_cells"]:
        ssl_engine = _engine_path(root, cell["ssl_cell_id"])
        source_checkpoint = ssl_engine / "last.pt"
        source_sha = file_sha256(source_checkpoint)
        staging = _staging_path(root, cell["cell_id"])
        destination = staging / "engine" / "encoder.pt"
        _execute_cell(
            root,
            cell_id=cell["cell_id"],
            protocol_fingerprint=protocol_fingerprint,
            command=[
                sys.executable,
                "-m",
                "music_critic.experiments.phase8b2.worker",
                "export-encoder",
                str(ssl_engine),
                str(destination),
            ],
            validate=lambda directory, sha=source_sha: _export_binding(
                protocol_fingerprint, sha, directory
            ),
        )
    if _maybe_stop(config, "encoder_export"):
        return {"status": "stopped", "stage": "encoder_export"}

    export_by_ssl = {
        row["ssl_cell_id"]: row for row in plan["encoder_export_cells"]
    }
    for cell in plan["downstream_cells"]:
        encoder_path = ""
        encoder_sha = ""
        source_sha = ""
        if cell["ssl_cell_id"] is not None:
            export_cell = export_by_ssl[cell["ssl_cell_id"]]
            encoder = _engine_path(root, export_cell["cell_id"]) / "encoder.pt"
            encoder_path = str(encoder.resolve())
            encoder_sha = file_sha256(encoder)
            source_sha = file_sha256(
                _engine_path(root, cell["ssl_cell_id"]) / "last.pt"
            )
        staging = _staging_path(root, cell["cell_id"])
        overrides = official_downstream_overrides(
            plan,
            cell["cell_id"],
            str(staging / "engine"),
            encoder_export_path=encoder_path,
            encoder_export_sha256=encoder_sha,
            source_ssl_checkpoint_sha256=source_sha,
            actual_sample_schedule_path=str(schedule_path.resolve()),
        )
        _execute_cell(
            root,
            cell_id=cell["cell_id"],
            protocol_fingerprint=protocol_fingerprint,
            command=[sys.executable, "-m", "music_critic.training.run", *overrides],
            validate=lambda directory, row=cell: _downstream_runtime_binding(
                plan, row, directory
            ),
        )
    if _maybe_stop(config, "downstream"):
        return {"status": "stopped", "stage": "downstream"}

    downstream_by_id = {
        row["cell_id"]: row for row in plan["downstream_cells"]
    }
    for cell in plan["evaluation_cells"]:
        downstream = downstream_by_id[cell["downstream_cell_id"]]
        checkpoint = _engine_path(root, downstream["cell_id"]) / "last.pt"
        staging = _staging_path(root, cell["cell_id"])
        overrides = official_evaluation_overrides(
            plan,
            checkpoint=str(checkpoint.resolve()),
            output_directory=str(staging / "engine"),
            cell_id=cell["cell_id"],
        )
        _execute_cell(
            root,
            cell_id=cell["cell_id"],
            protocol_fingerprint=protocol_fingerprint,
            command=[sys.executable, "-m", "music_critic.evaluation.run", *overrides],
            validate=lambda directory, row=cell: _evaluation_runtime_binding(
                plan, row, directory
            ),
        )
    if _maybe_stop(config, "evaluation"):
        return {"status": "stopped", "stage": "evaluation"}
    return {"status": "cells_complete", "stage": "evaluation"}


def _selection_metrics(metrics: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    endpoints: dict[str, float] = {}
    nll_values: list[float] = []
    for dataset_id in ("hooktheory", "pop909_cl"):
        tasks = metrics.get("datasets", {}).get(dataset_id, {})
        f1_values = []
        for evidence in tasks.values():
            model = evidence["model"]
            macro = model.get("macro_f1", {}).get("value")
            if macro is not None:
                f1_values.append(float(macro))
            nll = model.get("nll", model.get("bce_nll", {})).get("value")
            if nll is not None:
                nll_values.append(float(nll))
        if not f1_values:
            raise Phase8B2ContractError(
                f"phase8b2.runner.selection_endpoint_unavailable:{dataset_id}"
            )
        endpoints[dataset_id] = sum(f1_values) / len(f1_values)
    if not nll_values:
        raise Phase8B2ContractError(
            "phase8b2.runner.validation_nll_unavailable"
        )
    return endpoints, sum(nll_values) / len(nll_values)


def aggregate_verified_outputs(
    plan: Mapping[str, Any], root: Path
) -> dict[str, object]:
    protocol = plan["protocol"]
    candidates = []
    piece_rows = []
    downstream_metrics = []
    checkpoint_evidence = []
    transfer_evidence = []
    for cell in plan["evaluation_cells"]:
        _validate_published_cell(
            _cell_path(root, cell["cell_id"]),
            cell_id=cell["cell_id"],
            protocol_fingerprint=protocol["fingerprint"],
        )
        engine = _engine_path(root, cell["cell_id"])
        metrics = _read_json_dict(engine / "metrics.json")
        piece = _read_json_dict(engine / "piece_statistics.json")
        checkpoint = _read_json_dict(engine / "checkpoint_evidence.json")
        endpoints, nll = _selection_metrics(metrics)
        downstream_engine = _engine_path(root, cell["downstream_cell_id"])
        downstream_report = _read_json_dict(
            downstream_engine / "training_report.json"
        )
        checkpoint_path = downstream_engine / "last.pt"
        compute = 0
        if cell["variant_id"] != "supervised_scratch":
            ssl_id = next(
                row["ssl_cell_id"]
                for row in plan["downstream_cells"]
                if row["cell_id"] == cell["downstream_cell_id"]
            )
            ssl_report = _read_json_dict(
                _engine_path(root, ssl_id) / "training_report.json"
            )
            compute = int(ssl_report["accounting"]["encoder_forward_count"])
        candidates.append(
            {
                "seed": cell["seed"],
                "variant_id": cell["variant_id"],
                "transfer_mode": cell["transfer_mode"],
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "protocol_fingerprint": protocol["fingerprint"],
                "split": "validation",
                "dataset_endpoints": endpoints,
                "validation_nll": nll,
                "encoder_forward_count": compute,
            }
        )
        downstream_metrics.append(
            {
                "cell_id": cell["cell_id"],
                "metrics": metrics,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            }
        )
        checkpoint_evidence.append(
            {"cell_id": cell["cell_id"], **checkpoint}
        )
        transfer_evidence.append(
            {
                "cell_id": cell["downstream_cell_id"],
                **downstream_report["phase8b2_transfer"],
                "frozen_encoder_final": downstream_report.get(
                    "frozen_encoder_final"
                ),
            }
        )
        for row in piece["pieces"]:
            piece_rows.append(
                {
                    **row,
                    "seed": cell["seed"],
                    "variant_id": cell["variant_id"],
                    "transfer_mode": cell["transfer_mode"],
                    "checkpoint_sha256": checkpoint[
                        "checkpoint_sha256"
                    ],
                }
            )
    statistics = aggregate_piece_sufficient_statistics(
        piece_rows,
        declared_seeds=protocol["seeds"],
        bootstrap_seed=min(protocol["seeds"]),
        bootstrap_replicates=int(
            plan["protocol"]["runtime_execution_config"].get(
                "bootstrap_replicates",
                200,
            )
        ),
        minimum_scientific_seeds=(
            3 if len(protocol["seeds"]) >= 3 else 3
        ),
    )
    selection = select_validation_checkpoint(
        candidates,
        protocol_fingerprint=protocol["fingerprint"],
        declared_seeds=protocol["seeds"],
    )
    compute_rows = []
    ssl_metric_rows = []
    ssl_checkpoint_evidence = []
    for cell in plan["ssl_cells"]:
        engine = _engine_path(root, cell["cell_id"])
        report = _read_json_dict(engine / "training_report.json")
        accounting = compute_accounting_from_ssl_report(report)
        compute_rows.append(
            {
                "cell_id": cell["cell_id"],
                **accounting.to_dict(),
                "cuda_peak_memory": report["cuda_peak_memory"],
            }
        )
        ssl_checkpoint_evidence.append(
            {
                "cell_id": cell["cell_id"],
                "last_checkpoint": report["last_checkpoint"],
                "checkpoint_sha256": file_sha256(engine / "last.pt"),
                "encoder_state_fingerprints": report[
                    "encoder_state_fingerprints"
                ],
                "actual_sample_schedule_verified": report[
                    "actual_sample_schedule_verified"
                ],
            }
        )
        for line in (engine / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line:
                ssl_metric_rows.append(
                    {"cell_id": cell["cell_id"], **json.loads(line)}
                )
    compute = {
        "compute_accounting_contract_version": "1.1.0",
        "instrumentation": "actual_engine_counters",
        "cells": compute_rows,
        "all_primary_cells_matched": len(
            {row["encoder_forwards"] for row in compute_rows}
        )
        == 1,
    }
    if not compute["all_primary_cells_matched"]:
        raise Phase8B2ContractError(
            "phase8b2.runner.aggregate_compute_not_matched"
        )
    return {
        "candidates": candidates,
        "selection": selection,
        "statistics": statistics,
        "compute": compute,
        "piece_statistics": {
            "statistics_contract_version": "1.1.0",
            "rows": piece_rows,
            "retained_cuda_tensor_count": 0,
        },
        "downstream_metrics": downstream_metrics,
        "ssl_checkpoint_evidence": ssl_checkpoint_evidence,
        "transfer_evidence": transfer_evidence,
        "checkpoint_evidence": checkpoint_evidence,
        "ssl_metric_rows": ssl_metric_rows,
    }


def _finalize_bundle(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    root: Path,
    aggregate: Mapping[str, Any],
) -> dict[str, object]:
    final = root / "final_bundle"
    if final.exists():
        manifest = _read_json_dict(final / "run_manifest.json")
        expected_artifacts = set(REQUIRED_ARTIFACTS)
        actual_artifacts = {
            path.name for path in final.iterdir() if path.is_file()
        }
        recorded_hashes = manifest.get("artifact_sha256")
        current_cells = sorted(
            (
                _read_json_dict(path)
                for path in (root / "cells").rglob(
                    "cell_manifest.json"
                )
            ),
            key=lambda row: str(row["cell_id"]),
        )
        if (
            manifest.get("artifact_contract_version")
            != PHASE8B2_ARTIFACT_CONTRACT_VERSION
            or manifest.get("protocol_fingerprint")
            != plan["protocol"]["fingerprint"]
            or not isinstance(recorded_hashes, dict)
            or set(recorded_hashes)
            != expected_artifacts - {"run_manifest.json"}
            or actual_artifacts != expected_artifacts
            or manifest.get("cells") != current_cells
            or any(
                file_sha256(final / name) != value
                for name, value in recorded_hashes.items()
            )
        ):
            raise Phase8B2ContractError(
                "phase8b2.runner.final_bundle_stale_or_incomplete"
            )
        return {
            "status": "complete",
            "output_directory": str(final.resolve()),
            "resumed": True,
        }
    expected_cell_ids = {
        row["cell_id"]
        for key in (
            "ssl_cells",
            "encoder_export_cells",
            "downstream_cells",
            "evaluation_cells",
        )
        for row in plan[key]
    }
    actual_cell_ids = {
        str(path.relative_to(root / "cells").parent).replace(os.sep, "/")
        for path in (root / "cells").rglob("cell_manifest.json")
        if not str(path.relative_to(root / "cells")).startswith("preflight/")
    }
    if actual_cell_ids != expected_cell_ids:
        raise Phase8B2ContractError(
            "phase8b2.runner.partial_matrix_cannot_complete"
        )
    cells = [
        _read_json_dict(path)
        for path in sorted((root / "cells").rglob("cell_manifest.json"))
    ]
    ssl_compute_rows = aggregate["compute"]["cells"]
    final_report = {
        "matrix_runner_contract_version": MATRIX_RUNNER_CONTRACT_VERSION,
        "status": "complete",
        "protocol_fingerprint": plan["protocol"]["fingerprint"],
        "executed_cell_count": len(expected_cell_ids),
        "expected_cell_count": len(expected_cell_ids),
        "preflight_cell_count": len(plan["ssl_cells"]),
        "verified_runtime_binding_cell_count": sum(
            int(
                row.get("runtime_binding_evidence", {}).get("verified")
                is True
            )
            for row in cells
        ),
        "checkpoint_to_evaluation_verified_cell_count": len(
            plan["evaluation_cells"]
        ),
        "ssl_compute_totals": {
            "attempted_updates": sum(
                row["logical_updates"] for row in ssl_compute_rows
            ),
            "applied_updates": sum(
                row["optimizer_updates_applied"]
                for row in ssl_compute_rows
            ),
            "skipped_updates": sum(
                row["optimizer_updates_skipped"]
                for row in ssl_compute_rows
            ),
            "raw_samples": sum(
                row["raw_samples_seen"] for row in ssl_compute_rows
            ),
            "encoder_forwards": sum(
                row["encoder_forwards"] for row in ssl_compute_rows
            ),
        },
        "actual_schedule_fingerprints": {
            "ssl": {
                str(row["seed"]): {
                    "attestation_fingerprint": row["fingerprint"],
                    "sample_identity_fingerprint": row[
                        "sample_schedule_fingerprint"
                    ],
                }
                for row in plan["actual_sample_schedule"]["ssl"]
            },
            "downstream": {
                str(row["seed"]): {
                    "attestation_fingerprint": row["fingerprint"],
                    "sample_identity_fingerprint": row[
                        "sample_schedule_fingerprint"
                    ],
                }
                for row in plan["actual_sample_schedule"]["downstream"]
            },
        },
        "test_accessed": False,
        "validation_only_selection": True,
        "selected_configuration_id": aggregate["selection"][
            "selected_configuration_id"
        ],
        "selected_checkpoints": aggregate["selection"][
            "selected_checkpoints"
        ],
        "bounded_results_are_scientific_superiority_evidence": False,
        "pdmx_evidence": False,
        "full_scale_pdmx_owner": "Phase 10",
    }
    repository = repository_evidence(
        Path.cwd(),
        require_clean=config["comparison"]["name"] != "bounded_acceptance",
    )
    requested_device = str(plan["protocol"]["amp_device_config"]["name"])
    if requested_device == "auto":
        bundle_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        bundle_device = torch.device(requested_device)
    result = write_complete_artifact_bundle(
        final,
        protocol_fingerprint=plan["protocol"]["fingerprint"],
        repository=repository,
        environment=environment_evidence(bundle_device),
        cells=cells,
        json_artifacts={
            "comparison_protocol.json": plan["protocol"],
            "actual_sample_schedule.json": plan["actual_sample_schedule"],
            "ssl_checkpoint_evidence.json": aggregate[
                "ssl_checkpoint_evidence"
            ],
            "transfer_evidence.json": aggregate["transfer_evidence"],
            "downstream_metrics.json": aggregate["downstream_metrics"],
            "piece_statistics.json": aggregate["piece_statistics"],
            "validation_selection.json": aggregate["selection"],
            "statistical_summary.json": aggregate["statistics"],
            "compute_accounting.json": aggregate["compute"],
            "final_comparison_report.json": final_report,
        },
        ssl_metric_rows=aggregate["ssl_metric_rows"],
        allow_dirty_repository=(
            config["comparison"]["name"] == "bounded_acceptance"
        ),
    )
    return {"status": "complete", **result, "resumed": False}


def execute_matrix(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    action: str,
) -> dict[str, object]:
    """Execute or inspect the exact precomputed plan without test access."""

    if (
        config["comparison"]["name"] != "bounded_acceptance"
        and action in {"run", "resume"}
        and not plan["runtime_paths"]["index_paths"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.production_paths_required"
        )

    root = Path(str(config["output_root"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("plan.json", plan),
        ("actual_sample_schedule.json", plan["actual_sample_schedule"]),
    ):
        path = root / name
        if path.exists():
            if fingerprint(read_json(path)) != fingerprint(payload):
                raise Phase8B2ContractError(
                    f"phase8b2.runner.existing_{name}_mismatch"
                )
        else:
            write_json_once(path, payload)
    if action in {"run", "resume"}:
        state = _run_matrix_cells(config, plan, root)
        if state["status"] == "stopped":
            return {**state, "output_root": str(root)}
    aggregate = aggregate_verified_outputs(plan, root)
    if action == "aggregate":
        return {
            "status": "aggregated",
            "statistics": aggregate["statistics"],
            "compute": aggregate["compute"],
        }
    if action == "select":
        return {"status": "selected", "selection": aggregate["selection"]}
    return _finalize_bundle(config, plan, root, aggregate)


__all__ = [
    "CELL_MANIFEST_CONTRACT_VERSION",
    "MATRIX_RUNNER_CONTRACT_VERSION",
    "aggregate_verified_outputs",
    "execute_matrix",
]
