"""Executable, resumable Phase 9C-A control plane."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

from .artifacts import (
    build_artifact_manifest,
    file_sha256,
    line_plot_png,
    publish_staged_cell,
    read_json,
    verify_bundle,
    verify_completed_cell,
    write_bytes_once,
    write_comparison_tables,
    write_json_once,
)
from .contracts import (
    CLAIM_BOUNDARIES,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PHASE9C_PROFILE_VERSION,
    Phase9CContractError,
    fingerprint,
    validate_protocol,
)
from .metrics import (
    component_bootstrap_primary_delta,
    primary_validation_summary,
)


def _cell_directory(root: Path, cell_id: str) -> Path:
    parts = cell_id.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise Phase9CContractError("phase9c.runner.cell_id_invalid")
    return root / "cells" / Path(*parts)


def _staging_directory(root: Path, cell_id: str) -> Path:
    return root / ".staging" / cell_id.replace("/", "__")


def _all_cells(plan: Mapping[str, Any]) -> list[dict[str, object]]:
    return [
        *plan["ssl_cells"],
        *plan["encoder_export_cells"],
        *plan["train_prior_cells"],
        *plan["downstream_cells"],
        *plan["validation_cells"],
    ]


def _dependency_complete(root: Path, plan: Mapping[str, Any], cell: Mapping[str, object]) -> None:
    dependency = cell.get("depends_on")
    dependencies = [dependency, cell.get("prior_dependency")]
    for current in dependencies:
        if current is None:
            continue
        verify_completed_cell(
            _cell_directory(root, str(current)),
            cell_id=str(current),
            protocol_fingerprint=str(plan["protocol"]["fingerprint"]),
        )


def _run_subprocess(command: list[str], staging: Path) -> None:
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
    write_bytes_once(staging / "stdout.log", process.stdout.encode("utf-8"))
    write_bytes_once(staging / "stderr.log", process.stderr.encode("utf-8"))
    write_json_once(
        staging / "process.json",
        {"argv": command, "shell": False, "returncode": process.returncode},
    )
    if process.returncode:
        write_json_once(
            staging / "failure.json",
            {"status": "failed", "returncode": process.returncode},
        )
        raise Phase9CContractError(
            f"phase9c.runner.cell_failed:exit={process.returncode}"
        )


def _bounded_cell_command(spec_path: Path, staging: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "music_critic.experiments.phase9c.worker",
        "bounded-cell",
        str(spec_path),
        str(staging),
    ]


def _source_kind(plan: Mapping[str, Any], variant: str) -> str:
    for row in plan["encoder_export_cells"]:
        if row["variant_id"] == variant:
            return str(row["source_kind"])
    raise Phase9CContractError("phase9c.runner.encoder_source_missing")


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def _ssl_command(plan: Mapping[str, Any], cell: Mapping[str, object], staging: Path) -> list[str]:
    protocol = plan["protocol"]
    preset = protocol["preset"]
    schedule = cell["schedule"]
    paths = plan["runtime_paths"]
    steps = int(preset["ssl_updates"])
    batch_size = int(preset["batch_size"])
    views = schedule["policy_views"]
    command = [
        sys.executable,
        "-m",
        "music_critic.ssl.run",
        f"+phase8b_objective={schedule['objective_mode']}",
        f"+phase8b_masking={schedule['masking_mode']}",
        "+phase8b2_schedule=comparison",
        "experiment=pretrain",
        f"experiment.steps={steps}",
        f"experiment.epochs={max(1, int(preset['downstream_epochs']))}",
        f"experiment.optimizer_steps_per_epoch={min(steps, int(preset['downstream_steps_per_epoch']))}",
        "experiment.validation_interval=1",
        f"data.batch_size={batch_size}",
        f"data.epoch_size={batch_size * min(steps, int(preset['downstream_steps_per_epoch']))}",
        "data.validation_epoch_size=0",
        "data.workers=0",
        "data=mixed",
        f"data.index_paths={_hydra_list(paths['ssl_index_paths'])}",
        f"data.cache_roots={_hydra_list(paths['ssl_cache_roots'])}",
        f"data.split_manifest={paths['ssl_split_manifest']}",
        "+data.mixture_weights={dilemmadata:0.3333333333333333,hooktheory:0.3333333333333333,pop909_cl:0.3333333333333333}",
        "model=hierarchical",
        "optimizer=adamw",
        "optimizer.learning_rate=0.0003",
        "scheduler=none",
        "device=cuda",
        "device.name=cuda:0",
        "device.amp=true",
        "device.amp_dtype=float16",
        "seed=17",
        f"output_dir={staging / 'engine'}",
        "phase8b2_schedule.contract_version=1.1.0",
        "phase8b2_schedule.comparison_mode=encoder_forward_matched",
        f"phase8b2_schedule.variant_id={cell['variant_id']}",
        f"phase8b2_schedule.protocol_fingerprint={protocol['fingerprint']}",
        f"phase8b2_schedule.sample_schedule_fingerprint={schedule['sample_schedule_fingerprint']}",
        f"phase8b2_schedule.model_initialization_seed={protocol['paired_initialization']['initial_encoder_seed']}",
        f"phase8b2_schedule.data_order_seed={protocol['paired_initialization']['ssl_data_order_seed']}",
        f"phase8b2_schedule.logical_updates={steps}",
        "phase8b2_schedule.policy_view_names=" + _hydra_list([str(row["policy"]) for row in views]),
        "phase8b2_schedule.policy_view_seeds=" + _hydra_list([str(row["seed"]) for row in views]),
    ]
    return command


def _export_command(root: Path, cell: Mapping[str, object], staging: Path) -> list[str]:
    variant = str(cell["variant_id"])
    if variant == "scratch":
        return [
            sys.executable,
            "-m",
            "music_critic.experiments.phase9c.worker",
            "export-initial-encoder",
            str(staging / "engine" / "encoder.pt"),
        ]
    ssl_output = _cell_directory(root, f"ssl/{variant}") / "engine"
    return [
        sys.executable,
        "-m",
        "music_critic.experiments.phase8b2.worker",
        "export-encoder",
        str(ssl_output),
        str(staging / "engine" / "encoder.pt"),
    ]


def _downstream_command(
    root: Path, plan: Mapping[str, Any], cell: Mapping[str, object], staging: Path
) -> list[str]:
    protocol = plan["protocol"]
    preset = protocol["preset"]
    paths = plan["runtime_paths"]
    variant = str(cell["variant_id"])
    engine_mode = str(cell["engine_transfer_mode"])
    dependency = cell.get("depends_on")
    encoder_path = None if dependency is None else _cell_directory(root, str(dependency)) / "engine" / "encoder.pt"
    source_checkpoint = (
        encoder_path
        if variant == "scratch"
        else _cell_directory(root, f"ssl/{variant}") / "engine" / "last.pt"
    )
    logical_updates = int(preset["downstream_epochs"]) * int(preset["downstream_steps_per_epoch"])
    task_weights = ",".join(f"{task}:1.0" for task in protocol["task_ids"])
    command = [
        sys.executable,
        "-m",
        "music_critic.training.run",
        "experiment=dilemmadata_scratch_vs_ssl",
        f"experiment.steps={logical_updates}",
        f"experiment.epochs={preset['downstream_epochs']}",
        f"experiment.optimizer_steps_per_epoch={preset['downstream_steps_per_epoch']}",
        f"experiment.validation_interval={preset['downstream_epochs']}",
        "experiment.collect_gradient_evidence=true",
        "objective=supervised_harmonic",
        f"+objective.task_weights={{{task_weights}}}",
        "model=hierarchical",
        "data=dilemmadata",
        f"data.index_paths=[{paths['downstream_raw_index']}]",
        f"data.cache_roots=[{paths['downstream_raw_cache_root']}]",
        f"data.target_cache_index={paths['target_cache_index']}",
        f"data.target_cache_root={paths['target_cache_root']}",
        f"data.split_manifest={paths['downstream_split_manifest']}",
        f"data.batch_size={preset['batch_size']}",
        f"data.epoch_size={int(preset['batch_size']) * int(preset['downstream_steps_per_epoch'])}",
        "data.validation_epoch_size=0",
        "data.workers=0",
        "optimizer=adamw",
        "optimizer.learning_rate=0.0003",
        "scheduler=none",
        "device=cuda",
        "device.name=cuda:0",
        "device.amp=true",
        "device.amp_dtype=float16",
        "seed=17",
        f"output_dir={staging / 'engine'}",
        "transfer.contract_version=1.2.0",
        f"transfer.mode={engine_mode}",
        f"transfer.comparison_protocol_fingerprint={protocol['fingerprint']}",
        f"transfer.downstream_initialization_seed={protocol['paired_initialization']['fresh_head_seed']}",
        f"transfer.downstream_data_order_seed={protocol['paired_initialization']['downstream_data_order_seed']}",
        f"transfer.sample_schedule_fingerprint={cell['sample_schedule_fingerprint']}",
        f"transfer.logical_updates={logical_updates}",
        "downstream_task_ids=" + _hydra_list(list(protocol["task_ids"])),
    ]
    if encoder_path is not None:
        command.extend(
            [
                f"transfer.encoder_export_path={encoder_path}",
                f"transfer.encoder_export_sha256={file_sha256(encoder_path)}",
                f"transfer.source_ssl_checkpoint_sha256={file_sha256(source_checkpoint)}",
                f"transfer.source_kind={_source_kind(plan, variant)}",
            ]
        )
    return command


def _train_prior_command(
    plan: Mapping[str, Any], staging: Path
) -> list[str]:
    paths = plan["runtime_paths"]
    return [
        sys.executable,
        "-m",
        "music_critic.experiments.phase9c.worker",
        "build-train-priors",
        "--raw-index",
        paths["downstream_raw_index"],
        "--raw-cache-root",
        paths["downstream_raw_cache_root"],
        "--target-index",
        paths["target_cache_index"],
        "--target-cache-root",
        paths["target_cache_root"],
        "--split-manifest",
        paths["downstream_split_manifest"],
        "--batch-size",
        str(plan["protocol"]["preset"]["batch_size"]),
        "--output",
        str(staging / "train_priors.json"),
    ]


def _fixed_budget_checkpoint_binding(
    root: Path, cell: Mapping[str, object]
) -> tuple[Path, dict[str, object]]:
    dependency_cell = _cell_directory(root, str(cell["depends_on"]))
    dependency = dependency_cell / "engine"
    if not dependency.is_dir():
        dependency = dependency_cell
    training_report = read_json(dependency / "training_report.json")
    expected_updates = int(cell["optimizer_update_budget"])
    attempted = int(
        training_report.get(
            "optimizer_step_attempt_count",
            training_report.get("attempted_optimizer_updates", -1),
        )
    )
    applied = int(
        training_report.get(
            "optimizer_step_applied_count",
            training_report.get("applied_optimizer_updates", -1),
        )
    )
    skipped = int(
        training_report.get(
            "optimizer_step_skipped_count",
            training_report.get("skipped_optimizer_updates", -1),
        )
    )
    checkpoint_name = str(cell["comparison_checkpoint"])
    checkpoint = dependency / checkpoint_name
    if (
        checkpoint_name != "last.pt"
        or attempted != expected_updates
        or applied != expected_updates
        or skipped != 0
        or not checkpoint.is_file()
    ):
        raise Phase9CContractError(
            f"phase9c.comparison.fixed_budget_binding_invalid:{cell['cell_id']}"
        )
    return checkpoint, {
        "policy": "last_after_fixed_budget",
        "filename": "last.pt",
        "sha256": file_sha256(checkpoint),
        "expected_optimizer_updates": expected_updates,
        "attempted_optimizer_updates": attempted,
        "applied_optimizer_updates": applied,
        "skipped_optimizer_updates": skipped,
    }


def _validation_command(
    root: Path, plan: Mapping[str, Any], cell: Mapping[str, object], staging: Path
) -> list[str]:
    paths = plan["runtime_paths"]
    checkpoint, _ = _fixed_budget_checkpoint_binding(root, cell)
    return [
        sys.executable,
        "-m",
        "music_critic.evaluation.dilemmadata_run",
        "--checkpoint",
        str(checkpoint),
        "--raw-index",
        paths["downstream_raw_index"],
        "--raw-cache-root",
        paths["downstream_raw_cache_root"],
        "--target-index",
        paths["target_cache_index"],
        "--target-cache-root",
        paths["target_cache_root"],
        "--split-manifest",
        paths["downstream_split_manifest"],
        "--split",
        "validation",
        "--batch-size",
        str(plan["protocol"]["preset"]["batch_size"]),
        "--device",
        "cuda:0",
        "--train-priors",
        str(_cell_directory(root, "train_priors/dilemmadata") / "train_priors.json"),
        "--output",
        str(staging / "validation_report.json"),
    ]


def _production_cell_command(
    root: Path, plan: Mapping[str, Any], cell: Mapping[str, object], staging: Path
) -> list[str]:
    kind = str(cell["cell_id"]).split("/", 1)[0]
    if kind == "ssl":
        return _ssl_command(plan, cell, staging)
    if kind == "encoder_export":
        (staging / "engine").mkdir(parents=True, exist_ok=True)
        return _export_command(root, cell, staging)
    if kind == "downstream":
        return _downstream_command(root, plan, cell, staging)
    if kind == "train_priors":
        return _train_prior_command(plan, staging)
    if kind == "validation":
        return _validation_command(root, plan, cell, staging)
    raise Phase9CContractError("phase9c.runner.cell_kind_unknown")


def _execute_cell(
    root: Path,
    plan: Mapping[str, Any],
    cell: Mapping[str, object],
    *,
    bounded: bool,
) -> dict[str, object]:
    cell_id = str(cell["cell_id"])
    destination = _cell_directory(root, cell_id)
    protocol_fingerprint = str(plan["protocol"]["fingerprint"])
    if destination.exists():
        return verify_completed_cell(
            destination,
            cell_id=cell_id,
            protocol_fingerprint=protocol_fingerprint,
        )
    _dependency_complete(root, plan, cell)
    staging = _staging_directory(root, cell_id)
    binding = {
        "cell_id": cell_id,
        "protocol_fingerprint": protocol_fingerprint,
        "cell_fingerprint": fingerprint(cell),
    }
    if staging.exists():
        prior = read_json(staging / "resume_binding.json")
        if prior != binding:
            raise Phase9CContractError(f"phase9c.resume.binding_mismatch:{cell_id}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json_once(staging / "resume_binding.json", binding)
    write_json_once(staging / "cell_spec.json", cell)
    command = (
        _bounded_cell_command(staging / "cell_spec.json", staging)
        if bounded
        else _production_cell_command(root, plan, cell, staging)
    )
    _run_subprocess(command, staging)
    return publish_staged_cell(
        staging,
        destination,
        cell_id=cell_id,
        protocol_fingerprint=protocol_fingerprint,
    )


def profile_experiment(root: Path, plan: Mapping[str, Any]) -> dict[str, object]:
    """Run every batch candidate in a fresh subprocess and publish recommendations."""

    results = []
    production = plan["data_semantic_projection"]["kind"] == "production"
    rebuild_config_path = root / ".profile" / "rebuild_config.json"
    if production:
        write_json_once(rebuild_config_path, plan["profile_rebuild_config"])
    for candidate in plan["profile_candidates"]:
        candidate = int(candidate)
        output = root / ".profile" / f"batch-{candidate}.json"
        candidate_root = root / ".profile" / f"candidate-{candidate}"
        output.parent.mkdir(parents=True, exist_ok=True)
        if production:
            candidate_root.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "music_critic.experiments.phase9c.worker",
                "profile-production-candidate",
                str(rebuild_config_path),
                str(candidate),
                str(candidate_root),
                str(output),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "music_critic.experiments.phase9c.worker",
                "profile-candidate",
                str(candidate),
                str(output),
            ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if production:
            write_bytes_once(
                candidate_root / "profile_subprocess_stdout.log",
                process.stdout.encode("utf-8"),
            )
            write_bytes_once(
                candidate_root / "profile_subprocess_stderr.log",
                process.stderr.encode("utf-8"),
            )
            write_json_once(
                candidate_root / "profile_subprocess.json",
                {"argv": command, "shell": False, "returncode": process.returncode},
            )
        if output.exists():
            row = read_json(output)
        else:
            row = {"status": "failed", "batch_size": candidate}
        row["returncode"] = process.returncode
        if production:
            row["candidate_root"] = str(candidate_root)
            row["candidate_root_preserved"] = candidate_root.is_dir()
        results.append(row)
    passed = [row for row in results if row.get("status") == "passed"]
    recommended = max((int(row["batch_size"]) for row in passed), default=None)
    projected = None
    if passed:
        selected = next(row for row in passed if int(row["batch_size"]) == recommended)
        cells = len(plan["ssl_cells"]) + len(plan["downstream_cells"])
        projected = float(selected["seconds_per_downstream_epoch"]) * cells
    payload = {
        "contract_version": PHASE9C_PROFILE_VERSION,
        "status": "complete" if passed else "no_candidate_passed",
        "hardware_required": "NVIDIA GeForce RTX 3090 cuda:0",
        "candidate_process_isolation": True,
        "oom_cleanup_boundary": "subprocess_exit",
        "results": results,
        "recommended_batch_size": recommended,
        "projected_primary_pilot_seconds": projected,
        "recommendation_is_not_production_config": True,
        "production_requires_explicit_immutable_batch_size": True,
        "production_started": False,
        "test_access": False,
    }
    report = {**payload, "fingerprint": fingerprint(payload)}
    write_json_once(root / "profile_report.json", report)
    return report


def run_production_profile_candidate(
    plan: Mapping[str, Any], root: Path, *, batch_size: int
) -> dict[str, object]:
    """Execute one real bounded RTX profile matrix inside its candidate process."""

    started = time.perf_counter()
    _initialize_root(root, plan)
    timings: dict[str, float] = {}
    for cell in _all_cells(plan):
        kind = str(cell["cell_id"]).split("/", 1)[0]
        cell_started = time.perf_counter()
        _execute_cell(root, plan, cell, bounded=False)
        timings[kind] = timings.get(kind, 0.0) + (time.perf_counter() - cell_started)
    elapsed = time.perf_counter() - started
    ssl_updates = int(plan["protocol"]["preset"]["ssl_updates"])
    ssl_cells = len(plan["ssl_cells"])
    forwards = ssl_updates * ssl_cells * PHASE9C_ENCODER_FORWARDS_PER_UPDATE
    downstream_samples = (
        batch_size
        * int(plan["protocol"]["preset"]["downstream_steps_per_epoch"])
        * len(plan["downstream_cells"])
    )
    allocated: list[int] = []
    reserved: list[int] = []
    for path in root.glob("cells/**/engine/training_report.json"):
        report = read_json(path)
        device = report.get("device", {}) if isinstance(report, dict) else {}
        if isinstance(device, dict):
            if isinstance(device.get("peak_allocated_bytes"), int):
                allocated.append(device["peak_allocated_bytes"])
            if isinstance(device.get("peak_reserved_bytes"), int):
                reserved.append(device["peak_reserved_bytes"])
    return {
        "status": "passed",
        "batch_size": batch_size,
        "warmup_steps": 1,
        "measured_steps": max(1, ssl_updates - 1),
        "separate_ssl_variants": [row["variant_id"] for row in plan["ssl_cells"]],
        "frozen_probe_profiled": True,
        "full_finetune_profiled": True,
        "validation_traversal_profiled": True,
        "peak_allocated_vram_bytes": max(allocated) if allocated else None,
        "peak_reserved_vram_bytes": max(reserved) if reserved else None,
        "samples_per_second": downstream_samples / max(timings.get("downstream", 0.0), 1e-9),
        "encoder_forwards_per_second": forwards / max(timings.get("ssl", 0.0), 1e-9),
        "seconds_per_downstream_epoch": timings.get("downstream", 0.0),
        "validation_seconds": timings.get("validation", 0.0),
        "total_seconds": elapsed,
        "stage_seconds": timings,
        "subprocess_isolation": True,
        "state_cleaned_by_subprocess_exit": True,
        "production_training_started": False,
        "profile_short_dag_only": True,
        "test_access": False,
    }


def _load_validation_rows(root: Path, plan: Mapping[str, Any]) -> list[dict[str, object]]:
    rows = []
    for cell in plan["validation_cells"]:
        _, checkpoint_binding = _fixed_budget_checkpoint_binding(root, cell)
        report = read_json(_cell_directory(root, cell["cell_id"]) / "validation_report.json")
        summary = primary_validation_summary(report)
        rows.append(
            {
                "variant_id": cell["variant_id"],
                "transfer_mode": cell["transfer_mode"],
                "checkpoint_identity": str(cell["depends_on"]) + "/last.pt",
                "checkpoint_binding": checkpoint_binding,
                "validation_summary": summary,
                "validation_report": report,
            }
        )
    if len(
        {
            int(row["checkpoint_binding"]["applied_optimizer_updates"])
            for row in rows
        }
    ) != 1:
        raise Phase9CContractError("phase9c.comparison.optimizer_budget_mismatch")
    return rows


def aggregate_experiment(root: Path, plan: Mapping[str, Any]) -> dict[str, object]:
    rows = _load_validation_rows(root, plan)
    table_rows = []
    for row in rows:
        summary = row["validation_summary"]
        table_rows.append(
            {
                "variant_id": row["variant_id"],
                "transfer_mode": row["transfer_mode"],
                "primary_score": summary["primary_score"],
                "mean_macro_f1": summary["mean_macro_f1"],
                "mean_task_nll": summary["mean_task_nll"],
                "checkpoint_policy": "last_after_fixed_budget",
                "optimizer_updates": row["checkpoint_binding"]["applied_optimizer_updates"],
            }
        )
    bootstrap = []
    by_key = {(row["variant_id"], row["transfer_mode"]): row for row in rows}
    for row in rows:
        if row["variant_id"] == "scratch":
            continue
        suffix = "frozen_probe" if "frozen" in str(row["transfer_mode"]) else "full_finetune"
        scratch_mode = "scratch_frozen_probe" if suffix == "frozen_probe" else "scratch_full_finetune"
        reference = by_key[("scratch", scratch_mode)]
        bootstrap.append(
            {
                "variant_id": row["variant_id"],
                "transfer_mode": row["transfer_mode"],
                "reference": f"scratch/{scratch_mode}",
                "report": component_bootstrap_primary_delta(
                    reference["validation_report"],
                    row["validation_report"],
                    seed=int(plan["protocol"]["bootstrap"]["seed"]),
                    replicates=int(plan["protocol"]["bootstrap"]["replicates"]),
                ),
            }
        )
    bootstrap_payload = {
        "unit": "component",
        "comparisons": bootstrap,
        "interpretation": CLAIM_BOUNDARIES["bootstrap_interpretation"],
    }
    bootstrap_report = {**bootstrap_payload, "fingerprint": fingerprint(bootstrap_payload)}
    selection_payload = {
        "comparison_split": "validation",
        "comparison_metric": "mean_task_nll_div_log_class_count",
        "checkpoint_policy": "last_after_fixed_budget",
        "checkpoint_selection_between_epochs": False,
        "validation_compares_final_checkpoints_only": True,
        "configurations": [
            {
                "variant_id": row["variant_id"],
                "transfer_mode": row["transfer_mode"],
                "checkpoint_identity": row["checkpoint_identity"],
                "checkpoint_binding": row["checkpoint_binding"],
                "validation_summary": row["validation_summary"],
            }
            for row in rows
        ],
        "test_access": False,
    }
    selection_report = {**selection_payload, "fingerprint": fingerprint(selection_payload)}
    final_payload = {
        "phase": "9C-A",
        "seed": 17,
        "one_seed_exploratory": True,
        "rows": table_rows,
        "frozen_probe_and_full_finetune_separate": True,
        "bootstrap_report_fingerprint": bootstrap_report["fingerprint"],
        "selection_report_fingerprint": selection_report["fingerprint"],
        "claim_boundaries": CLAIM_BOUNDARIES,
        "production_pilot_executed": plan["data_semantic_projection"]["kind"] == "production",
        "test_access": False,
    }
    final_report = {**final_payload, "fingerprint": fingerprint(final_payload)}
    write_json_once(root / "bootstrap_report.json", bootstrap_report)
    write_json_once(root / "selection_report.json", selection_report)
    write_json_once(root / "final_comparison_report.json", final_report)
    write_comparison_tables(root, table_rows)
    loss_values = [float(row["mean_task_nll"]) for row in table_rows]
    primary_values = [float(row["primary_score"]) for row in table_rows]
    write_json_once(root / "curves" / "comparison_curves.json", {"loss": loss_values, "primary": primary_values})
    write_bytes_once(root / "curves" / "loss.png", line_plot_png(loss_values))
    write_bytes_once(root / "curves" / "primary_validation_metric.png", line_plot_png(primary_values))
    return {"bootstrap": bootstrap_report, "selection": selection_report, "final": final_report}


def _initialize_root(root: Path, plan: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("experiment_plan.json", plan),
        ("protocol.json", plan["protocol"]),
        ("data_semantic_projection.json", plan["data_semantic_projection"]),
        ("claim_boundaries.json", CLAIM_BOUNDARIES),
    ):
        write_json_once(root / name, payload)


def execute_experiment(
    root: Path,
    plan: Mapping[str, Any],
    *,
    action: str,
    fail_after_cell: int = 0,
) -> dict[str, object]:
    validate_protocol(plan["protocol"])
    _initialize_root(root, plan)
    bounded = plan["data_semantic_projection"]["kind"] == "bounded_synthetic_fixture"
    if action == "profile":
        return profile_experiment(root, plan)
    if action in {"run", "resume"}:
        if not bounded:
            preset = plan["protocol"]["preset"]
            if not preset["production_budget_resolved"]:
                raise Phase9CContractError("phase9c.runner.production_budget_unresolved")
            profile_path = root / "profile_report.json"
            if not profile_path.exists():
                external_profile = Path(str(plan.get("profile_report_path", "")))
                if not external_profile.is_file():
                    raise Phase9CContractError("phase9c.runner.profile_required")
                write_json_once(profile_path, read_json(external_profile))
            profile = read_json(profile_path)
            if profile.get("status") != "complete":
                raise Phase9CContractError("phase9c.runner.profile_required")
            if int(preset["batch_size"]) not in {
                int(row["batch_size"])
                for row in profile["results"]
                if row.get("status") == "passed"
            }:
                raise Phase9CContractError("phase9c.runner.batch_size_not_profiled")
        elif not (root / "profile_report.json").exists():
            profile_experiment(root, plan)
        completed = 0
        for cell in _all_cells(plan):
            _execute_cell(root, plan, cell, bounded=bounded)
            completed += 1
            if fail_after_cell and completed == fail_after_cell:
                return {"status": "stopped", "completed_cells": completed}
    aggregate = aggregate_experiment(root, plan)
    if action == "aggregate":
        return {"status": "aggregated", **aggregate}
    if action == "select":
        return {"status": "selected", "selection": aggregate["selection"]}
    manifest = build_artifact_manifest(root)
    write_json_once(root / "artifact_manifest.json", manifest)
    verification = verify_bundle(root)
    return {
        "status": "complete",
        "output_root": str(root),
        "verification": verification,
        "production_pilot_executed": not bounded,
        "test_access": False,
    }


__all__ = [
    "aggregate_experiment",
    "execute_experiment",
    "profile_experiment",
    "run_production_profile_candidate",
]
