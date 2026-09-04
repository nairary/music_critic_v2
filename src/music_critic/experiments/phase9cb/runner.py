"""Execution, aggregation, and independent evidence checks for Phase 9C-B."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Mapping

import torch

from .contracts import (
    PHASE9CB_CELLS,
    Phase9CBError,
    canonical_bytes,
    file_sha256,
    fingerprint,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase9CBError(f"phase9cb.artifact.unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise Phase9CBError(f"phase9cb.artifact.mapping_required:{path}")
    return value


def _hydra_tasks() -> str:
    tasks = (
        "dilemmadata.an.chord.inversion",
        "dilemmadata.an.chord.quality",
        "dilemmadata.dlc.chord.inversion",
        "dilemmadata.dlc.chord.quality",
    )
    return ",".join(f"{task}:1.0" for task in tasks)


def _training_command(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    profile: bool,
    resume: bool,
) -> list[str]:
    protocol = plan["protocol"]
    schedule = protocol["schedule"]
    bindings = protocol["bindings"]
    epochs = (
        int(schedule["profile_epochs"])
        if profile
        else int(schedule["epochs"])
    )
    steps = (
        int(schedule["profile_steps_per_epoch"])
        if profile
        else int(schedule["steps_per_epoch"])
    )
    epoch_size = (
        int(schedule["profile_epoch_size"])
        if profile
        else int(schedule["batch_size"]) * steps
    )
    logical_updates = epochs * steps
    command = [
        sys.executable,
        "-m",
        "music_critic.training.run",
        "experiment=dilemmadata_scratch_vs_ssl",
        f"experiment.steps={logical_updates}",
        f"experiment.epochs={epochs}",
        f"experiment.optimizer_steps_per_epoch={steps}",
        "experiment.validation_interval=1",
        "experiment.collect_gradient_evidence=true",
        "objective=supervised_harmonic",
        f"+objective.task_weights={{{_hydra_tasks()}}}",
        "objective.class_weight_artifact_path="
        + str(bindings["class_weight_artifact"]["path"]),
        "model=hierarchical",
        f"+model.decoder.kind={cell['decoder_kind']}",
        "data=dilemmadata",
        f"data.index_paths=[{bindings['raw_index']['path']}]",
        f"data.cache_roots=[{bindings['raw_cache_root']}]",
        f"data.target_cache_index={bindings['target_index']['path']}",
        f"data.target_cache_root={bindings['target_cache_root']}",
        f"data.split_manifest={bindings['split_manifest']['path']}",
        f"data.batch_size={schedule['batch_size']}",
        f"data.epoch_size={epoch_size}",
        "data.validation_epoch_size=0",
        "data.workers=0",
        "optimizer=adamw",
        f"optimizer.learning_rate={schedule['learning_rate']}",
        "scheduler=none",
        "device=cuda",
        "device.name=cuda:0",
        "device.amp=true",
        "device.amp_dtype=float16",
        "seed=17",
        f"output_dir={output}",
        "transfer.contract_version=1.2.0",
        f"transfer.mode={cell['transfer_mode']}",
        f"transfer.comparison_protocol_fingerprint={protocol['fingerprint']}",
        "transfer.downstream_initialization_seed="
        + str(schedule["downstream_initialization_seed"]),
        "transfer.downstream_data_order_seed="
        + str(schedule["downstream_data_order_seed"]),
        "transfer.sample_schedule_fingerprint="
        + str(
            schedule[
                "profile_sample_schedule_fingerprint"
                if profile
                else "sample_schedule_fingerprint"
            ]
        ),
        f"transfer.logical_updates={logical_updates}",
        "downstream_task_ids=[dilemmadata.an.chord.inversion,"
        "dilemmadata.an.chord.quality,dilemmadata.dlc.chord.inversion,"
        "dilemmadata.dlc.chord.quality]",
    ]
    if cell["encoder_initialization"] == "ssl":
        checkpoint = bindings["ssl_checkpoint"]
        command.extend(
            (
                f"transfer.encoder_export_path={checkpoint['encoder_export_path']}",
                f"transfer.encoder_export_sha256={checkpoint['encoder_export_sha256']}",
                f"transfer.source_ssl_checkpoint_sha256={checkpoint['sha256']}",
                f"transfer.source_kind={checkpoint['source_kind']}",
            )
        )
    if resume:
        checkpoint = output / "last.pt"
        if checkpoint.is_file():
            command.append(f"experiment.resume_from={checkpoint}")
    return command


def _evaluation_command(
    plan: Mapping[str, object], checkpoint: Path, output: Path
) -> list[str]:
    protocol = plan["protocol"]
    bindings = protocol["bindings"]
    return [
        sys.executable,
        "-m",
        "music_critic.evaluation.dilemmadata_run",
        "--checkpoint",
        str(checkpoint),
        "--raw-index",
        str(bindings["raw_index"]["path"]),
        "--raw-cache-root",
        str(bindings["raw_cache_root"]),
        "--target-index",
        str(bindings["target_index"]["path"]),
        "--target-cache-root",
        str(bindings["target_cache_root"]),
        "--split-manifest",
        str(bindings["split_manifest"]["path"]),
        "--split",
        "validation",
        "--batch-size",
        str(protocol["schedule"]["batch_size"]),
        "--device",
        "cuda:0",
        "--train-priors",
        str(bindings["train_priors"]["path"]),
        "--output",
        str(output),
    ]


def _run(command: list[str], log: Path) -> float:
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise Phase9CBError(
            f"phase9cb.subprocess.failed:{process.returncode}:{log}"
        )
    return time.perf_counter() - started


def _cell_kind(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = payload["metadata"]["model_contract"]
    return str(contract.get("decoder", {}).get("kind", "mlp"))


def _parameter_counts(checkpoint: Path) -> dict[str, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    encoder_prefixes = (
        "local_baseline.encoder.",
        "context_encoder.pooling.",
        "context_encoder.transformer.",
        "context_encoder.fusion.",
    )
    return {
        "encoder": sum(
            value.numel()
            for name, value in state.items()
            if name.startswith(encoder_prefixes)
        ),
        "gru": sum(
            value.numel()
            for name, value in state.items()
            if name.startswith("sequence_decoder.gru.")
        ),
        "decoder_total": sum(
            value.numel()
            for name, value in state.items()
            if name.startswith("sequence_decoder.")
        ),
        "heads": sum(
            value.numel()
            for name, value in state.items()
            if name.startswith("task_heads.")
        ),
    }


def _execute_cell(
    root: Path,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    profile: bool,
    resume: bool,
) -> dict[str, object]:
    base = root / ("profile_cells" if profile else "cells")
    destination = base / str(cell["cell_id"])
    if (destination / "cell_report.json").is_file():
        return _read(destination / "cell_report.json")
    staging = base / ".staging" / str(cell["cell_id"])
    staging.mkdir(parents=True, exist_ok=True)
    engine = staging / "engine"
    train_seconds = _run(
        _training_command(plan, cell, engine, profile=profile, resume=resume),
        staging / "training.log",
    )
    checkpoint = engine / "last.pt"
    if not checkpoint.is_file() or _cell_kind(checkpoint) != cell["decoder_kind"]:
        raise Phase9CBError("phase9cb.cell.checkpoint_decoder_mismatch")
    validation_seconds = _run(
        _evaluation_command(plan, checkpoint, staging / "validation_report.json"),
        staging / "validation.log",
    )
    training = _read(engine / "training_report.json")
    attempted = training.get("optimizer_step_attempt_count")
    applied = training.get("optimizer_step_applied_count")
    skipped = training.get("optimizer_step_skipped_count")
    report = {
        "cell_id": cell["cell_id"],
        "decoder_kind": cell["decoder_kind"],
        "encoder_initialization": cell["encoder_initialization"],
        "schedule_fingerprint": cell["schedule_fingerprint"],
        "checkpoint": {
            "path": "engine/last.pt",
            "sha256": file_sha256(checkpoint),
            "policy": "last_after_equal_applied_updates",
        },
        "validation_checkpoint_sha256": file_sha256(checkpoint),
        "attempted_updates": attempted,
        "applied_updates": applied,
        "skipped_updates": skipped,
        "actual_sample_schedule_fingerprint": training.get(
            "observed_downstream_schedule_fingerprint"
        ),
        "fresh_supervised_initialization_fingerprint": training.get(
            "phase8b2_transfer", {}
        ).get("fresh_supervised_initialization_fingerprint"),
        "transfer": training.get("phase8b2_transfer"),
        "wall_clock_train_seconds": train_seconds,
        "updates_per_second": (
            float(applied) / train_seconds
            if isinstance(applied, int) and train_seconds > 0
            else None
        ),
        "wall_clock_validation_seconds": validation_seconds,
        "parameter_counts": _parameter_counts(checkpoint),
        "peak_allocated_bytes": training.get("device", {}).get(
            "peak_allocated_bytes"
        ),
        "peak_reserved_bytes": training.get("device", {}).get(
            "peak_reserved_bytes"
        ),
        "test_lock": plan["protocol"]["test_lock"],
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write(staging / "cell_report.json", report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return report


def _delta(
    right: Mapping[str, object], left: Mapping[str, object]
) -> dict[str, float]:
    return {
        "normalized_nll_raw_delta_right_minus_left": (
            float(right["mean_normalized_nll"])
            - float(left["mean_normalized_nll"])
        ),
        "normalized_nll_improvement_left_minus_right": (
            float(left["mean_normalized_nll"])
            - float(right["mean_normalized_nll"])
        ),
        "macro_f1_improvement_right_minus_left": (
            float(right["mean_macro_f1"]) - float(left["mean_macro_f1"])
        ),
        "balanced_accuracy_improvement_right_minus_left": (
            float(right["mean_balanced_accuracy"])
            - float(left["mean_balanced_accuracy"])
        ),
        "accuracy_improvement_right_minus_left": (
            float(right["mean_accuracy"])
            - float(left["mean_accuracy"])
        ),
    }


def _task_delta(
    right: Mapping[str, object], left: Mapping[str, object]
) -> dict[str, float]:
    return {
        "normalized_nll_raw_delta_right_minus_left": (
            float(right["normalized_nll"])
            - float(left["normalized_nll"])
        ),
        "normalized_nll_improvement_left_minus_right": (
            float(left["normalized_nll"])
            - float(right["normalized_nll"])
        ),
        "macro_f1_improvement_right_minus_left": (
            float(right["macro_f1"]) - float(left["macro_f1"])
        ),
        "balanced_accuracy_improvement_right_minus_left": (
            float(right["balanced_accuracy"])
            - float(left["balanced_accuracy"])
        ),
        "accuracy_improvement_right_minus_left": (
            float(right["accuracy"]) - float(left["accuracy"])
        ),
    }


def aggregate(root: Path, plan: Mapping[str, object]) -> dict[str, object]:
    metrics = {}
    task_metrics = {}
    cells = {}
    for cell_id in PHASE9CB_CELLS:
        directory = root / "cells" / cell_id
        cells[cell_id] = _read(directory / "cell_report.json")
        validation = _read(directory / "validation_report.json")
        aggregate_row = validation.get("aggregate")
        if not isinstance(aggregate_row, dict) or aggregate_row.get("task_count") != 4:
            raise Phase9CBError(f"phase9cb.aggregate.metrics_incomplete:{cell_id}")
        metrics[cell_id] = aggregate_row
        task_metrics[cell_id] = validation["tasks"]
    deltas = {
        "decoder_effect_under_scratch": _delta(metrics["scratch_onset_bigru"], metrics["scratch_mlp"]),
        "decoder_effect_under_ssl": _delta(metrics["ssl_onset_bigru"], metrics["ssl_mlp"]),
        "ssl_effect_with_mlp": _delta(metrics["ssl_mlp"], metrics["scratch_mlp"]),
        "ssl_effect_with_onset_bigru": _delta(metrics["ssl_onset_bigru"], metrics["scratch_onset_bigru"]),
    }
    pairings = {
        "decoder_effect_under_scratch": (
            "scratch_onset_bigru",
            "scratch_mlp",
        ),
        "decoder_effect_under_ssl": ("ssl_onset_bigru", "ssl_mlp"),
        "ssl_effect_with_mlp": ("ssl_mlp", "scratch_mlp"),
        "ssl_effect_with_onset_bigru": (
            "ssl_onset_bigru",
            "scratch_onset_bigru",
        ),
    }
    task_deltas = {
        comparison: {
            task_id: _task_delta(
                task_metrics[right][task_id], task_metrics[left][task_id]
            )
            for task_id in task_metrics[right]
        }
        for comparison, (right, left) in pairings.items()
    }
    gru_helps_scratch = deltas["decoder_effect_under_scratch"]["normalized_nll_improvement_left_minus_right"] > 0
    gru_helps_ssl = deltas["decoder_effect_under_ssl"]["normalized_nll_improvement_left_minus_right"] > 0
    ssl_helps_gru = deltas["ssl_effect_with_onset_bigru"]["normalized_nll_improvement_left_minus_right"] > 0
    interpretations = []
    if gru_helps_scratch and gru_helps_ssl:
        interpretations.append("old_decoder_probable_bottleneck")
    if (
        ssl_helps_gru
        and deltas["ssl_effect_with_mlp"][
            "normalized_nll_improvement_left_minus_right"
        ]
        <= 0
    ):
        interpretations.append(
            "ssl_sequence_information_visible_only_with_bigru"
        )
    if (gru_helps_scratch or gru_helps_ssl) and not ssl_helps_gru:
        interpretations.append(
            "decoder_helps_but_ssl_objectives_or_distribution_next"
        )
    if not gru_helps_scratch and not gru_helps_ssl:
        interpretations.append(
            "target_formulation_noise_alignment_or_subset_next"
        )
    report = {
        "contract_version": "1.0.0",
        "plan_fingerprint": plan["fingerprint"],
        "cells": cells,
        "aggregate_metrics": metrics,
        "deltas": deltas,
        "per_task_deltas": task_deltas,
        "descriptive_interpretations": interpretations,
        "statistical_significance_claimed": False,
        "scientific_superiority_claimed": False,
        "test_lock": plan["protocol"]["test_lock"],
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write(root / "aggregate_report.json", report)
    _write_manifest(root)
    return report


def _write_manifest(root: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    manifest = {"contract_version": "1.0.0", "files": files}
    manifest = {**manifest, "fingerprint": fingerprint(manifest)}
    _write(root / "bundle_manifest.json", manifest)
    return manifest


def verify_bundle(root: Path, *, expected_sha: str | None = None) -> dict[str, object]:
    plan = _read(root / "experiment_plan.json")
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict) or tuple(protocol.get("cells", ())) != PHASE9CB_CELLS:
        raise Phase9CBError("phase9cb.verify.cell_inventory_invalid")
    if expected_sha is not None and protocol.get("git_head") != expected_sha:
        raise Phase9CBError("phase9cb.verify.git_head_mismatch")
    if any(protocol.get("test_lock", {}).get(name) is not False for name in (
        "test_inference", "test_targets_accessed", "test_metrics_accessed", "test_unlock"
    )):
        raise Phase9CBError("phase9cb.verify.test_lock_invalid")
    schedule = protocol["schedule"]
    expected_updates = int(schedule["logical_updates"])
    fresh = {}
    actual_schedules = set()
    checkpoint_hashes = {}
    data_fingerprints = set()
    for cell in plan["cells"]:
        cell_id = cell["cell_id"]
        directory = root / "cells" / cell_id
        report = _read(directory / "cell_report.json")
        if (
            report.get("decoder_kind") != cell["decoder_kind"]
            or report.get("schedule_fingerprint")
            != schedule["sample_schedule_fingerprint"]
            or report.get("attempted_updates") != expected_updates
            or report.get("applied_updates") != expected_updates
            or report.get("skipped_updates") != 0
        ):
            raise Phase9CBError(f"phase9cb.verify.cell_schedule_invalid:{cell_id}")
        checkpoint = directory / "engine" / "last.pt"
        if report["checkpoint"]["sha256"] != file_sha256(checkpoint):
            raise Phase9CBError(f"phase9cb.verify.checkpoint_binding_invalid:{cell_id}")
        if report.get("validation_checkpoint_sha256") != report["checkpoint"]["sha256"]:
            raise Phase9CBError(f"phase9cb.verify.checkpoint_evaluation_binding_invalid:{cell_id}")
        if _cell_kind(checkpoint) != cell["decoder_kind"]:
            raise Phase9CBError(f"phase9cb.verify.decoder_checkpoint_invalid:{cell_id}")
        transfer = report.get("transfer")
        if not isinstance(transfer, dict) or transfer.get("fresh_supervised_preserved_after_transfer") is not True:
            raise Phase9CBError(f"phase9cb.verify.transfer_invalid:{cell_id}")
        loaded = transfer.get("loaded_tensors")
        if cell["encoder_initialization"] == "scratch":
            if loaded != [] or transfer.get("source_kind") != "supervised_scratch":
                raise Phase9CBError(f"phase9cb.verify.scratch_transfer_invalid:{cell_id}")
        else:
            prefixes = (
                "local_baseline.encoder.",
                "context_encoder.pooling.",
                "context_encoder.transformer.",
                "context_encoder.fusion.",
            )
            if (
                not isinstance(loaded, list)
                or not loaded
                or any(not str(name).startswith(prefixes) for name in loaded)
                or any(str(name).startswith(("sequence_decoder.", "task_heads.")) for name in loaded)
            ):
                raise Phase9CBError(f"phase9cb.verify.encoder_only_transfer_invalid:{cell_id}")
        training = _read(directory / "engine" / "training_report.json")
        data_fingerprints.add(
            fingerprint(training.get("fingerprints", {}))
        )
        fresh[cell_id] = report["fresh_supervised_initialization_fingerprint"]
        actual_schedules.add(report["actual_sample_schedule_fingerprint"])
        checkpoint_hashes[cell_id] = report["checkpoint"]["sha256"]
        validation = _read(directory / "validation_report.json")
        for task in validation.get("tasks", {}).values():
            required = {
                "normalized_nll", "macro_f1", "balanced_accuracy", "accuracy",
                "per_class", "confusion_matrix", "true_class_support",
                "predicted_class_distribution", "prediction_entropy", "alignment_counts",
            }
            if not isinstance(task, dict) or not required <= set(task):
                raise Phase9CBError(f"phase9cb.verify.metrics_incomplete:{cell_id}")
            if any(
                not math.isfinite(float(task[name]))
                for name in (
                    "normalized_nll",
                    "macro_f1",
                    "balanced_accuracy",
                    "accuracy",
                    "prediction_entropy",
                )
            ):
                raise Phase9CBError(f"phase9cb.verify.metric_non_finite:{cell_id}")
    if fresh["scratch_mlp"] != fresh["ssl_mlp"] or fresh["scratch_onset_bigru"] != fresh["ssl_onset_bigru"]:
        raise Phase9CBError("phase9cb.verify.paired_fresh_initialization_mismatch")
    if len(actual_schedules) != 1 or None in actual_schedules:
        raise Phase9CBError("phase9cb.verify.batch_schedule_mismatch")
    if len(data_fingerprints) != 1:
        raise Phase9CBError("phase9cb.verify.data_fingerprint_mismatch")
    manifest = _read(root / "bundle_manifest.json")
    payload = dict(manifest)
    observed = payload.pop("fingerprint", None)
    if observed != fingerprint(payload):
        raise Phase9CBError("phase9cb.verify.manifest_fingerprint_invalid")
    for relative, digest in manifest["files"].items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise Phase9CBError(f"phase9cb.verify.bundle_hash_invalid:{relative}")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    if actual_files != set(manifest["files"]):
        raise Phase9CBError("phase9cb.verify.bundle_file_inventory_mismatch")
    return {
        "status": "verified",
        "cell_count": 4,
        "checkpoint_hashes": checkpoint_hashes,
        "test_access": False,
        "manifest_fingerprint": manifest["fingerprint"],
    }


def execute(root: Path, plan: Mapping[str, object], *, action: str) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "experiment_plan.json").exists():
        _write(root / "experiment_plan.json", plan)
        _write(root / "protocol.json", plan["protocol"])
    elif _read(root / "experiment_plan.json") != plan:
        raise Phase9CBError("phase9cb.resume.plan_mismatch")
    if action == "plan":
        return {"status": "planned", "production_started": False, "test_access": False, "plan_fingerprint": plan["fingerprint"]}
    if action == "verify":
        return verify_bundle(root)
    if action == "aggregate":
        return aggregate(root, plan)
    if action not in {"profile", "run", "resume"}:
        raise Phase9CBError("phase9cb.action_invalid")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise Phase9CBError("phase9cb.hardware.rtx3090_cuda0_required")
    results = [
        _execute_cell(root, plan, cell, profile=action == "profile", resume=action == "resume")
        for cell in plan["cells"]
    ]
    if action == "profile":
        report = {
            "status": "complete",
            "production_started": False,
            "cells": results,
            "recommended_batch_size": plan["protocol"]["schedule"]["batch_size"],
            "test_access": False,
        }
        _write(root / "profile_report.json", report)
        return report
    return aggregate(root, plan)


def create_evidence_tar(root: Path, destination: Path) -> str:
    verify_bundle(root)
    with tarfile.open(destination, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(root), recursive=False)
    digest = file_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return digest


__all__ = ["aggregate", "create_evidence_tar", "execute", "verify_bundle"]
