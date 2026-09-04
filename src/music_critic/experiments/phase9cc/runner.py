"""Execution, convergence aggregation, and independent Phase 9C-C checks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    raw_downstream_sample_schedule_fingerprint,
)

from .contracts import (
    PHASE9CC_BASE_SHA,
    PHASE9CC_CELLS,
    PHASE9CC_TASKS,
    Phase9CCError,
    canonical_bytes,
    file_sha256,
    fingerprint,
)
from .training import _schedule, model_state_fingerprint, run_cell_training


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
        raise Phase9CCError(f"phase9cc.artifact.unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise Phase9CCError(f"phase9cc.artifact.mapping_required:{path}")
    return value


def _rows(path: Path) -> list[dict[str, object]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase9CCError(f"phase9cc.artifact.unreadable:{path}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise Phase9CCError(f"phase9cc.artifact.rows_invalid:{path}")
    return values


def _evaluation_command(
    plan: Mapping[str, object], checkpoint: Path, output: Path, device: str
) -> list[str]:
    bindings = plan["protocol"]["bindings"]
    return [
        sys.executable,
        "-m",
        "music_critic.evaluation.dilemmadata_run",
        "--checkpoint",
        str(checkpoint),
        "--raw-index",
        bindings["raw_index"]["path"],
        "--raw-cache-root",
        bindings["raw_cache_root"],
        "--target-index",
        bindings["target_index"]["path"],
        "--target-cache-root",
        bindings["target_cache_root"],
        "--split-manifest",
        bindings["split_manifest"]["path"],
        "--split",
        "validation",
        "--batch-size",
        str(plan["protocol"]["schedule"]["batch_size"]),
        "--workers",
        "0",
        "--device",
        device,
        "--train-priors",
        bindings["train_priors"]["path"],
        "--output",
        str(output),
    ]


def _evaluate_milestones(
    root: Path,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    device: str,
) -> dict[str, object]:
    directory = root / "cells" / str(cell["cell_id"])
    rows = []
    for update in plan["protocol"]["schedule"]["validation_milestones"]:
        checkpoint = directory / "checkpoints" / f"update-{update}.pt"
        if not checkpoint.is_file():
            raise Phase9CCError(
                f"phase9cc.validation.checkpoint_missing:{cell['cell_id']}:{update}"
            )
        report_path = directory / "milestones" / f"update-{update}.json"
        log_path = directory / "milestones" / f"update-{update}.log"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not report_path.is_file():
            process = subprocess.run(
                _evaluation_command(plan, checkpoint, report_path, device),
                capture_output=True,
                text=True,
                check=False,
            )
            log_path.write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            if process.returncode:
                raise Phase9CCError(
                    "phase9cc.validation.subprocess_failed:"
                    f"{cell['cell_id']}:{update}:{process.returncode}"
                )
        validation = _read(report_path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        rows.append(
            {
                "update": update,
                "checkpoint_path": f"checkpoints/update-{update}.pt",
                "checkpoint_sha256": file_sha256(checkpoint),
                "model_state_fingerprint": model_state_fingerprint(
                    payload["model_state"]
                ),
                "checkpoint_declared_model_state_fingerprint": payload.get(
                    "model_state_fingerprint"
                ),
                "validation_report_path": f"milestones/update-{update}.json",
                "validation_report_sha256": file_sha256(report_path),
                "validation_report_fingerprint": validation.get("fingerprint"),
                "validation_membership_fingerprint": validation.get(
                    "membership_fingerprint"
                ),
                "aggregate": validation.get("aggregate"),
                "tasks": validation.get("tasks"),
            }
        )
    result = {
        "contract_version": "1.0.0",
        "cell_id": cell["cell_id"],
        "split": "validation",
        "milestones": rows,
        "test_access": False,
    }
    result = {**result, "fingerprint": fingerprint(result)}
    _write(directory / "validation_milestones.json", result)
    return result


_METRICS = (
    "mean_normalized_nll",
    "mean_macro_f1",
    "mean_balanced_accuracy",
    "mean_accuracy",
    "mean_prediction_entropy",
)


def _metric_delta(
    right: Mapping[str, object], left: Mapping[str, object]
) -> dict[str, float | None]:
    return {
        name: (
            None
            if right.get(name) is None or left.get(name) is None
            else float(right[name]) - float(left[name])
        )
        for name in _METRICS
    }


def aggregate(root: Path, plan: Mapping[str, object]) -> dict[str, object]:
    validation_by_cell = {
        cell_id: _read(
            root / "cells" / cell_id / "validation_milestones.json"
        )
        for cell_id in PHASE9CC_CELLS
    }
    train_by_cell = {
        cell_id: _read(root / "cells" / cell_id / "training_report.json")
        for cell_id in PHASE9CC_CELLS
    }
    telemetry_by_cell = {
        cell_id: _rows(
            root / "cells" / cell_id / "train_telemetry.jsonl"
        )
        for cell_id in PHASE9CC_CELLS
    }
    indexed = {
        cell_id: {
            int(row["update"]): row
            for row in validation_by_cell[cell_id]["milestones"]
        }
        for cell_id in PHASE9CC_CELLS
    }
    transitions = ((1000, 3000), (3000, 6000), (6000, 9000))
    available = set(indexed["scratch_mlp"])
    transitions = tuple(
        pair for pair in transitions if pair[0] in available and pair[1] in available
    )
    within_cell = {
        cell_id: {
            f"{left}_to_{right}": _metric_delta(
                indexed[cell_id][right]["aggregate"],
                indexed[cell_id][left]["aggregate"],
            )
            for left, right in transitions
        }
        for cell_id in PHASE9CC_CELLS
    }
    ssl_minus_scratch = {
        str(update): _metric_delta(
            indexed["ssl_mlp"][update]["aggregate"],
            indexed["scratch_mlp"][update]["aggregate"],
        )
        for update in sorted(available)
    }
    best = {}
    final = max(available)
    for cell_id in PHASE9CC_CELLS:
        best_update = min(
            available,
            key=lambda update: float(
                indexed[cell_id][update]["aggregate"][
                    "mean_normalized_nll"
                ]
            ),
        )
        best[cell_id] = {
            "selection_metric": "mean_normalized_nll_descriptive_only",
            "best_milestone": best_update,
            "final_milestone": final,
            "final_minus_best": _metric_delta(
                indexed[cell_id][final]["aggregate"],
                indexed[cell_id][best_update]["aggregate"],
            ),
        }
    report = {
        "contract_version": "1.0.0",
        "plan_fingerprint": plan["fingerprint"],
        "cells": {
            cell_id: {
                "training": train_by_cell[cell_id],
                "milestones": validation_by_cell[cell_id]["milestones"],
                "train_loss_moving_averages": telemetry_by_cell[cell_id],
                "update_accounting": {
                    name: train_by_cell[cell_id][name]
                    for name in (
                        "applied_updates",
                        "attempted_updates",
                        "skipped_updates",
                    )
                },
            }
            for cell_id in PHASE9CC_CELLS
        },
        "within_cell_milestone_deltas": within_cell,
        "ssl_minus_scratch_gaps": ssl_minus_scratch,
        "best_milestone_and_final_vs_best": best,
        "automatic_plateau_verdict": None,
        "test_access": False,
        "scientific_superiority_claimed": False,
        "statistical_significance_claimed": False,
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write(root / "convergence_report.json", report)
    _write_manifest(root)
    return report


def _write_manifest(root: Path) -> dict[str, object]:
    excluded = {"manifest.json", "payload.sha256"}
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    manifest = {
        "contract_version": "1.0.0",
        "files": files,
        "file_count": len(files),
    }
    manifest = {**manifest, "fingerprint": fingerprint(manifest)}
    _write(root / "manifest.json", manifest)
    digest = file_sha256(root / "manifest.json")
    (root / "payload.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8"
    )
    return manifest


def _checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Phase9CCError(f"phase9cc.verify.checkpoint_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise Phase9CCError(f"phase9cc.verify.checkpoint_invalid:{path}")
    return payload


def _verify_metrics(cell_id: str, update: int, row: Mapping[str, object]) -> None:
    aggregate_row = row.get("aggregate")
    tasks = row.get("tasks")
    if not isinstance(aggregate_row, dict) or not isinstance(tasks, dict):
        raise Phase9CCError(
            f"phase9cc.verify.milestone_metrics_missing:{cell_id}:{update}"
        )
    if aggregate_row.get("task_count") != 4 or any(
        aggregate_row.get(name) is None
        or not math.isfinite(float(aggregate_row[name]))
        for name in _METRICS
    ):
        raise Phase9CCError(
            f"phase9cc.verify.aggregate_metrics_invalid:{cell_id}:{update}"
        )
    if set(tasks) != set(PHASE9CC_TASKS):
        raise Phase9CCError(
            f"phase9cc.verify.task_inventory_invalid:{cell_id}:{update}"
        )
    for task in tasks.values():
        required = {
            "normalized_nll",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "true_class_support",
            "predicted_class_distribution",
            "prediction_entropy",
        }
        if not isinstance(task, dict) or not required <= set(task):
            raise Phase9CCError(
                f"phase9cc.verify.task_metrics_missing:{cell_id}:{update}"
            )
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
            raise Phase9CCError(
                f"phase9cc.verify.metric_non_finite:{cell_id}:{update}"
            )


def verify_bundle(
    root: Path, *, expected_sha: str | None = None
) -> dict[str, object]:
    plan = _read(root / "experiment_plan.json")
    unsigned = dict(plan)
    observed_plan_fingerprint = unsigned.pop("fingerprint", None)
    if observed_plan_fingerprint != fingerprint(unsigned):
        raise Phase9CCError("phase9cc.verify.plan_fingerprint_invalid")
    protocol = plan.get("protocol")
    if (
        not isinstance(protocol, dict)
        or tuple(protocol.get("cells", ())) != PHASE9CC_CELLS
        or tuple(row.get("cell_id") for row in plan.get("cells", ()))
        != PHASE9CC_CELLS
        or protocol.get("seed") != 17
        or protocol.get("phase9cb_base_sha") != PHASE9CC_BASE_SHA
    ):
        raise Phase9CCError("phase9cc.verify.protocol_invalid")
    unsigned_protocol = dict(protocol)
    observed_protocol_fingerprint = unsigned_protocol.pop("fingerprint", None)
    if observed_protocol_fingerprint != fingerprint(unsigned_protocol):
        raise Phase9CCError("phase9cc.verify.protocol_fingerprint_invalid")
    if _read(root / "protocol.json") != protocol:
        raise Phase9CCError("phase9cc.verify.protocol_artifact_mismatch")
    if expected_sha is not None and protocol.get("git_head") != expected_sha:
        raise Phase9CCError("phase9cc.verify.git_head_mismatch")
    if any(value is not False for value in protocol["test_lock"].values()):
        raise Phase9CCError("phase9cc.verify.test_access")
    schedule = protocol["schedule"]
    if not protocol.get("bounded_test_protocol") and (
        schedule.get("epochs") != 1
        or schedule.get("optimizer_steps_per_epoch") != 9000
        or schedule.get("required_applied_updates") != 9000
        or schedule.get("batch_size") != 2
        or schedule.get("telemetry_interval_applied") != 100
        or schedule.get("checkpoint_interval_applied") != 1000
        or schedule.get("validation_milestones")
        != [0, 1000, 3000, 6000, 9000]
    ):
        raise Phase9CCError("phase9cc.verify.production_protocol_invalid")
    for name, binding in protocol["bindings"].items():
        if name in {"raw_cache_root", "target_cache_root", "ssl_checkpoint"}:
            continue
        path = Path(binding["path"])
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise Phase9CCError(
                f"phase9cc.verify.input_binding_invalid:{name}"
            )
    ssl_binding = protocol["bindings"]["ssl_checkpoint"]
    for path_name, sha_name in (
        ("path", "sha256"),
        ("encoder_export_path", "encoder_export_sha256"),
    ):
        path = Path(ssl_binding[path_name])
        if not path.is_file() or file_sha256(path) != ssl_binding[sha_name]:
            raise Phase9CCError(
                f"phase9cc.verify.input_binding_invalid:{path_name}"
            )
    rebuilt_identities = _schedule(plan)
    required_updates = int(schedule["required_applied_updates"])
    milestones = tuple(schedule["validation_milestones"])
    checkpoint_interval = int(schedule["checkpoint_interval_applied"])
    required_checkpoints = set(range(0, required_updates + 1, checkpoint_interval)) | set(milestones)
    schedule_fingerprints = set()
    fresh_heads = set()
    data_fingerprints = set()
    validation_memberships = set()
    checkpoint_hashes = {}
    for cell in plan["cells"]:
        cell_id = cell["cell_id"]
        directory = root / "cells" / cell_id
        training = _read(directory / "training_report.json")
        unsigned_training = dict(training)
        observed_training_fingerprint = unsigned_training.pop(
            "fingerprint", None
        )
        if observed_training_fingerprint != fingerprint(unsigned_training):
            raise Phase9CCError(
                f"phase9cc.verify.training_fingerprint_invalid:{cell_id}"
            )
        if (
            training.get("complete") is not True
            or training.get("applied_updates") != required_updates
            or training.get("attempted_updates")
            != required_updates + int(training.get("skipped_updates", -1))
            or training.get("sample_schedule_position") != required_updates
            or training.get("actual_sample_schedule_fingerprint")
            != schedule["sample_schedule_fingerprint"]
            or training.get("test_access") is not False
            or not isinstance(training.get("resume_evidence"), dict)
            or training["resume_evidence"].get(
                "schedule_rebuilt_then_rng_restored"
            )
            is not True
        ):
            raise Phase9CCError(
                f"phase9cc.verify.training_accounting_invalid:{cell_id}"
            )
        telemetry = _rows(directory / "train_telemetry.jsonl")
        expected_telemetry = required_updates // int(
            schedule["telemetry_interval_applied"]
        )
        if (
            len(telemetry) != expected_telemetry
            or [row.get("applied_updates") for row in telemetry]
            != list(
                range(
                    int(schedule["telemetry_interval_applied"]),
                    required_updates + 1,
                    int(schedule["telemetry_interval_applied"]),
                )
            )
            or fingerprint(telemetry) != training.get("telemetry_fingerprint")
        ):
            raise Phase9CCError(
                f"phase9cc.verify.telemetry_invalid:{cell_id}"
            )
        for row in telemetry:
            if any(
                not math.isfinite(float(row[name]))
                for name in (
                    "mean_objective_loss",
                    "learning_rate",
                    "grad_scaler_scale",
                    "mean_gradient_norm_before_clip",
                    "last_gradient_norm_before_clip",
                )
            ):
                raise Phase9CCError(
                    f"phase9cc.verify.telemetry_non_finite:{cell_id}"
                )
            consumed = int(row["sample_count_consumed"])
            if row.get("sample_schedule_prefix_fingerprint") != (
                raw_downstream_sample_schedule_fingerprint(
                    rebuilt_identities[:consumed]
                )
            ):
                raise Phase9CCError(
                    f"phase9cc.verify.telemetry_schedule_invalid:{cell_id}"
                )
        observed_checkpoints = {
            int(path.stem.split("-", 1)[1]): path
            for path in (directory / "checkpoints").glob("update-*.pt")
        }
        if set(observed_checkpoints) != required_checkpoints:
            raise Phase9CCError(
                f"phase9cc.verify.checkpoint_inventory_invalid:{cell_id}"
            )
        for update, path in observed_checkpoints.items():
            payload = _checkpoint(path)
            metadata = payload.get("metadata", {})
            progress = payload.get("progress", {})
            if (
                metadata.get("phase9cc_checkpoint_version") != "1.0.0"
                or metadata.get("cell_id") != cell_id
                or metadata.get("plan_fingerprint") != plan["fingerprint"]
                or metadata.get("schedule_fingerprint")
                != schedule["sample_schedule_fingerprint"]
                or metadata.get("resume_boundary")
                != "applied_update_mid_epoch"
                or progress.get("applied_updates") != update
                or progress.get("schedule_position") != update
                or progress.get("attempted_updates")
                != update + int(progress.get("skipped_updates", -1))
                or progress.get("schedule_prefix_fingerprint")
                != raw_downstream_sample_schedule_fingerprint(
                    rebuilt_identities[
                        : update * int(schedule["batch_size"])
                    ]
                )
                or not isinstance(payload.get("rng_state"), dict)
                or set(payload["rng_state"]) != {
                    "python",
                    "torch_cpu",
                    "torch_cuda",
                }
                or model_state_fingerprint(payload.get("model_state"))
                != payload.get("model_state_fingerprint")
            ):
                raise Phase9CCError(
                    f"phase9cc.verify.checkpoint_binding_invalid:{cell_id}:{update}"
                )
        transfer = training.get("transfer")
        if not isinstance(transfer, dict) or transfer.get(
            "fresh_supervised_preserved_after_transfer"
        ) is not True:
            raise Phase9CCError(f"phase9cc.verify.transfer_invalid:{cell_id}")
        loaded = transfer.get("loaded_tensors")
        if cell["encoder_initialization"] == "scratch":
            if loaded != [] or transfer.get("source_kind") != "supervised_scratch":
                raise Phase9CCError("phase9cc.verify.scratch_transfer_invalid")
        else:
            encoder_prefixes = (
                "local_baseline.encoder.",
                "context_encoder.pooling.",
                "context_encoder.transformer.",
                "context_encoder.fusion.",
            )
            if (
                not isinstance(loaded, list)
                or not loaded
                or any(not str(name).startswith(encoder_prefixes) for name in loaded)
                or any(
                    str(name).startswith(("sequence_decoder.", "task_heads."))
                    for name in loaded
                )
            ):
                raise Phase9CCError(
                    "phase9cc.verify.encoder_only_transfer_invalid"
                )
            if (
                transfer.get("source_ssl_checkpoint_sha256")
                != ssl_binding["sha256"]
                or transfer.get("encoder_export_sha256")
                != ssl_binding["encoder_export_sha256"]
            ):
                raise Phase9CCError(
                    "phase9cc.verify.ssl_transfer_binding_invalid"
                )
        validation = _read(directory / "validation_milestones.json")
        unsigned_validation = dict(validation)
        observed_validation_fingerprint = unsigned_validation.pop(
            "fingerprint", None
        )
        if observed_validation_fingerprint != fingerprint(unsigned_validation):
            raise Phase9CCError(
                f"phase9cc.verify.milestone_fingerprint_invalid:{cell_id}"
            )
        rows = validation.get("milestones")
        if (
            validation.get("test_access") is not False
            or not isinstance(rows, list)
            or tuple(row.get("update") for row in rows) != milestones
        ):
            raise Phase9CCError(
                f"phase9cc.verify.milestones_invalid:{cell_id}"
            )
        for row in rows:
            update = int(row["update"])
            checkpoint = observed_checkpoints[update]
            payload = _checkpoint(checkpoint)
            report_path = directory / row["validation_report_path"]
            validation_report = _read(report_path)
            unsigned_report = dict(validation_report)
            observed_report_fingerprint = unsigned_report.pop(
                "fingerprint", None
            )
            if (
                row.get("checkpoint_sha256") != file_sha256(checkpoint)
                or row.get("model_state_fingerprint")
                != payload.get("model_state_fingerprint")
                or row.get("validation_report_sha256")
                != file_sha256(report_path)
                or row.get("validation_report_fingerprint")
                != validation_report.get("fingerprint")
                or row.get("validation_membership_fingerprint")
                != protocol["validation_membership"][
                    "evaluation_membership_fingerprint"
                ]
                or validation_report.get("split") != "validation"
                or observed_report_fingerprint != fingerprint(unsigned_report)
            ):
                raise Phase9CCError(
                    f"phase9cc.verify.checkpoint_evaluation_binding_invalid:{cell_id}:{update}"
                )
            _verify_metrics(cell_id, update, row)
            validation_memberships.add(
                row["validation_membership_fingerprint"]
            )
        schedule_fingerprints.add(
            training["actual_sample_schedule_fingerprint"]
        )
        fresh = training.get("fresh_supervised_initialization_fingerprint")
        if not isinstance(fresh, str) or len(fresh) != 64:
            raise Phase9CCError(
                f"phase9cc.verify.fresh_initialization_invalid:{cell_id}"
            )
        fresh_heads.add(fresh)
        data_fingerprints.add(fingerprint(training["data_fingerprints"]))
        checkpoint_hashes[cell_id] = file_sha256(
            observed_checkpoints[required_updates]
        )
    if len(schedule_fingerprints) != 1 or len(fresh_heads) != 1:
        raise Phase9CCError("phase9cc.verify.pairing_mismatch")
    if len(data_fingerprints) != 1 or len(validation_memberships) != 1:
        raise Phase9CCError("phase9cc.verify.data_membership_mismatch")
    convergence = _read(root / "convergence_report.json")
    unsigned_convergence = dict(convergence)
    observed_convergence_fingerprint = unsigned_convergence.pop(
        "fingerprint", None
    )
    if (
        convergence.get("plan_fingerprint") != plan["fingerprint"]
        or convergence.get("test_access") is not False
        or convergence.get("scientific_superiority_claimed") is not False
        or convergence.get("statistical_significance_claimed") is not False
        or convergence.get("automatic_plateau_verdict") is not None
        or observed_convergence_fingerprint
        != fingerprint(unsigned_convergence)
    ):
        raise Phase9CCError("phase9cc.verify.convergence_claim_invalid")
    manifest = _read(root / "manifest.json")
    unsigned_manifest = dict(manifest)
    observed_manifest_fingerprint = unsigned_manifest.pop("fingerprint", None)
    if observed_manifest_fingerprint != fingerprint(unsigned_manifest):
        raise Phase9CCError("phase9cc.verify.manifest_fingerprint_invalid")
    for relative, digest in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise Phase9CCError(
                f"phase9cc.verify.artifact_hash_invalid:{relative}"
            )
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "payload.sha256"}
    }
    if actual_files != set(manifest.get("files", {})):
        raise Phase9CCError("phase9cc.verify.file_inventory_invalid")
    expected_payload = f"{file_sha256(root / 'manifest.json')}  manifest.json\n"
    if (root / "payload.sha256").read_text(encoding="utf-8") != expected_payload:
        raise Phase9CCError("phase9cc.verify.payload_digest_invalid")
    return {
        "status": "verified",
        "cell_count": 2,
        "required_applied_updates": required_updates,
        "checkpoint_hashes": checkpoint_hashes,
        "resume_boundary": "applied_update_mid_epoch",
        "test_access": False,
        "manifest_fingerprint": manifest["fingerprint"],
    }


def execute(
    root: Path,
    plan: Mapping[str, object],
    *,
    action: str,
    device: str | None = None,
) -> dict[str, object]:
    if action not in {"plan", "run", "resume", "verify"}:
        raise Phase9CCError("phase9cc.action_invalid")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "experiment_plan.json"
    if not plan_path.exists():
        _write(plan_path, plan)
        _write(root / "protocol.json", plan["protocol"])
    elif _read(plan_path) != plan:
        raise Phase9CCError("phase9cc.resume.plan_mismatch")
    if action == "plan":
        return {
            "status": "planned",
            "production_started": False,
            "test_access": False,
            "plan_fingerprint": plan["fingerprint"],
        }
    if action == "verify":
        return verify_bundle(root)
    production = not bool(plan["protocol"].get("bounded_test_protocol"))
    resolved_device = device or ("cuda:0" if production else "cpu")
    if production and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090"
        or resolved_device != "cuda:0"
    ):
        raise Phase9CCError("phase9cc.hardware.rtx3090_cuda0_required")
    for cell in plan["cells"]:
        directory = root / "cells" / cell["cell_id"]
        report_path = directory / "training_report.json"
        report = _read(report_path) if report_path.is_file() else None
        if report is None or report.get("complete") is not True:
            run_cell_training(
                plan,
                cell,
                directory,
                action=(
                    "resume"
                    if action == "resume" and directory.exists()
                    else "run"
                ),
                device=resolved_device,
            )
        _evaluate_milestones(
            root, plan, cell, device=resolved_device
        )
    report = aggregate(root, plan)
    verified = verify_bundle(root)
    return {
        "status": "complete",
        "convergence_report_fingerprint": report["fingerprint"],
        "verification": verified,
        "test_access": False,
    }


__all__ = ["aggregate", "execute", "verify_bundle"]
