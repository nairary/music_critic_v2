"""Exact stateful training continuation from Phase 9C-C update 9000."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cc.runner import _read
from music_critic.experiments.phase9cc.training import (
    _save_torch,
    _telemetry_row,
    _write_json,
    _write_jsonl,
    model_state_fingerprint,
    training_config as parent_training_config,
)
from music_critic.training import engine as training_engine
from music_critic.training.checkpoint import capture_rng_state, restore_rng_state
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import model_contract_metadata

from .contracts import (
    Phase9CCContinuationError,
    _production_schedule,
    file_sha256,
    fingerprint,
)


CONTINUATION_CHECKPOINT_VERSION = "1.0.0"
CONTINUATION_TRAINING_REPORT_VERSION = "1.0.0"


def schedule_identities(
    plan: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    protocol = plan["protocol"]
    bounded = protocol.get("bounded_schedule_identities")
    if protocol.get("bounded_test_protocol") and isinstance(bounded, list):
        identities = tuple(tuple(value) for value in bounded)
    else:
        parent_root = Path(protocol["parent_binding"]["root"])
        parent_plan = _read(parent_root / "experiment_plan.json")
        identities = _production_schedule(
            parent_plan,
            int(protocol["schedule"]["target_applied_update"]),
        )
    if raw_downstream_sample_schedule_fingerprint(identities) != protocol[
        "schedule"
    ]["full_schedule_fingerprint"]:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.training.schedule_rebuild_mismatch"
        )
    return identities


def continuation_training_config(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    parent_root = Path(plan["protocol"]["parent_binding"]["root"])
    parent_plan = _read(parent_root / "experiment_plan.json")
    parent_cell = next(
        row for row in parent_plan["cells"] if row["cell_id"] == cell["cell_id"]
    )
    config = parent_training_config(
        parent_plan, parent_cell, output, device=device
    )
    schedule = plan["protocol"]["schedule"]
    target = int(schedule["target_applied_update"])
    config["data"]["epoch_size"] = int(schedule["epoch_size"])
    config["experiment"]["steps"] = target
    config["experiment"]["optimizer_steps_per_epoch"] = target
    config["transfer"].update(
        {
            "mode": "supervised_scratch",
            "encoder_export_path": "",
            "encoder_export_sha256": "",
            "source_ssl_checkpoint_sha256": "",
            "source_kind": "phase7a_ssl",
            "comparison_protocol_fingerprint": plan["protocol"]["fingerprint"],
            "sample_schedule_fingerprint": schedule[
                "full_schedule_fingerprint"
            ],
            "logical_updates": target,
            "downstream_initialization_seed": schedule[
                "downstream_initialization_seed"
            ],
            "downstream_data_order_seed": schedule[
                "downstream_data_order_seed"
            ],
        }
    )
    return config


def _load_payload(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Phase9CCContinuationError(
            f"phase9cc.continuation.resume.checkpoint_unreadable:{path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.checkpoint_invalid"
        )
    return payload


def _restore_state(
    payload: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
) -> None:
    if (
        model_state_fingerprint(payload.get("model_state"))
        != payload.get("model_state_fingerprint")
        or payload.get("metadata", {}).get("model_contract")
        != model_contract_metadata(model)
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.model_binding_invalid"
        )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is None:
            if payload.get("scheduler_state") is not None:
                raise ValueError("unexpected scheduler state")
        else:
            scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload["scaler_state"])
    except Exception as exc:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.state_invalid"
        ) from exc


def _restore_parent(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    binding = cell["parent_checkpoint"]
    path = Path(binding["path"])
    if file_sha256(path) != binding["sha256"]:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.parent_checkpoint_sha256_mismatch"
        )
    payload = _load_payload(path)
    metadata = payload.get("metadata", {})
    progress = payload.get("progress", {})
    parent = plan["protocol"]["parent_binding"]
    start = int(plan["protocol"]["schedule"]["start_applied_update"])
    if (
        metadata.get("plan_fingerprint") != parent["plan_fingerprint"]
        or metadata.get("protocol_fingerprint")
        != parent["protocol_fingerprint"]
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("schedule_fingerprint")
        != parent["schedule_fingerprint"]
        or progress.get("applied_updates") != start
        or progress.get("schedule_position") != start
        or payload.get("model_state_fingerprint")
        != binding["model_state_fingerprint"]
        or not isinstance(payload.get("rng_state"), dict)
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.parent_checkpoint_binding_mismatch"
        )
    _restore_state(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    return payload


def _checkpoint_payload(
    *,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    config: Mapping[str, object],
    runtime: object,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
    applied: int,
    attempted: int,
    skipped: int,
    schedule_prefix_fingerprint: str,
    telemetry_rows: list[dict[str, object]],
) -> dict[str, object]:
    schedule = plan["protocol"]["schedule"]
    return {
        "metadata": {
            "phase9cc_continuation_checkpoint_version": (
                CONTINUATION_CHECKPOINT_VERSION
            ),
            "model_contract": model_contract_metadata(model),
            "plan_fingerprint": plan["fingerprint"],
            "protocol_fingerprint": plan["protocol"]["fingerprint"],
            "parent_binding_fingerprint": plan["protocol"]["parent_binding"][
                "fingerprint"
            ],
            "parent_checkpoint_sha256": cell["parent_checkpoint"]["sha256"],
            "cell_id": cell["cell_id"],
            "schedule_fingerprint": schedule["full_schedule_fingerprint"],
            "data_fingerprints": runtime.fingerprints,
            "config_fingerprint": fingerprint(config),
            "start_applied_update": schedule["start_applied_update"],
            "resume_boundary": "applied_update_mid_epoch",
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "progress": {
            "epoch": 0,
            "schedule_position": applied,
            "applied_updates": applied,
            "attempted_updates": attempted,
            "skipped_updates": skipped,
            "schedule_prefix_fingerprint": schedule_prefix_fingerprint,
            "telemetry_row_count": len(telemetry_rows),
            "telemetry_fingerprint": fingerprint(telemetry_rows),
        },
        "telemetry_rows": telemetry_rows,
        "model_state_fingerprint": model_state_fingerprint(model),
    }


def _latest_checkpoint(directory: Path) -> Path | None:
    candidates = []
    for path in directory.glob("update-*.pt"):
        try:
            candidates.append((int(path.stem.split("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return max(candidates)[1] if candidates else None


def _restore_continuation(
    path: Path,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    config: Mapping[str, object],
    runtime: object,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    payload = _load_payload(path)
    metadata = payload.get("metadata", {})
    progress = payload.get("progress", {})
    expected = {
        "phase9cc_continuation_checkpoint_version": (
            CONTINUATION_CHECKPOINT_VERSION
        ),
        "plan_fingerprint": plan["fingerprint"],
        "protocol_fingerprint": plan["protocol"]["fingerprint"],
        "parent_binding_fingerprint": plan["protocol"]["parent_binding"][
            "fingerprint"
        ],
        "parent_checkpoint_sha256": cell["parent_checkpoint"]["sha256"],
        "cell_id": cell["cell_id"],
        "schedule_fingerprint": plan["protocol"]["schedule"][
            "full_schedule_fingerprint"
        ],
        "data_fingerprints": runtime.fingerprints,
        "config_fingerprint": fingerprint(config),
        "start_applied_update": plan["protocol"]["schedule"][
            "start_applied_update"
        ],
        "resume_boundary": "applied_update_mid_epoch",
    }
    if (
        not isinstance(metadata, dict)
        or any(metadata.get(name) != value for name, value in expected.items())
        or not isinstance(progress, dict)
        or progress.get("epoch") != 0
        or progress.get("schedule_position") != progress.get("applied_updates")
        or progress.get("telemetry_row_count")
        != len(payload.get("telemetry_rows", ()))
        or progress.get("telemetry_fingerprint")
        != fingerprint(payload.get("telemetry_rows", ()))
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.checkpoint_binding_mismatch"
        )
    _restore_state(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    return payload


def run_cell_training(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    action: str,
    device: str,
    stop_after_applied: int | None = None,
) -> dict[str, object]:
    """Continue one exact parent trajectory, with resumable global updates."""

    if action not in {"run", "resume"}:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.training.action_invalid"
        )
    config = continuation_training_config(
        plan, cell, output, device=device
    )
    (
        _unused_output,
        resolved_device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        _cuda_memory,
        class_weights,
    ) = training_engine._prepare(config)
    if scheduler is not None:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.training.scheduler_must_be_none"
        )
    schedule = plan["protocol"]["schedule"]
    identities = schedule_identities(plan)
    batch_size = int(schedule["batch_size"])
    start = int(schedule["start_applied_update"])
    target = int(schedule["target_applied_update"])
    telemetry_interval = int(schedule["telemetry_interval_applied"])
    checkpoint_interval = int(schedule["checkpoint_interval_applied"])
    checkpoints = output / "checkpoints"
    telemetry_path = output / "train_telemetry.jsonl"
    output.mkdir(parents=True, exist_ok=True)

    latest = _latest_checkpoint(checkpoints) if action == "resume" else None
    if latest is None:
        if action == "run" and (
            telemetry_path.exists() or (output / "training_report.json").exists()
        ):
            raise Phase9CCContinuationError(
                "phase9cc.continuation.training.fresh_output_required"
            )
        payload = _restore_parent(
            plan,
            cell,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        progress = payload["progress"]
        applied = start
        attempted = int(progress["attempted_updates"])
        skipped = int(progress["skipped_updates"])
        telemetry_rows: list[dict[str, object]] = []
        entry_rng = payload["rng_state"]
        resumed_from = {
            "source": "parent",
            "update": start,
            "checkpoint_sha256": cell["parent_checkpoint"]["sha256"],
        }
    else:
        payload = _restore_continuation(
            latest,
            plan,
            cell,
            config,
            runtime,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        progress = payload["progress"]
        applied = int(progress["applied_updates"])
        attempted = int(progress["attempted_updates"])
        skipped = int(progress["skipped_updates"])
        telemetry_rows = copy.deepcopy(payload["telemetry_rows"])
        entry_rng = payload["rng_state"]
        resumed_from = {
            "source": "continuation",
            "update": applied,
            "checkpoint_sha256": file_sha256(latest),
        }
    if applied < start or applied > target:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.progress_invalid"
        )
    disk_rows = []
    if telemetry_path.is_file():
        try:
            disk_rows = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.resume.telemetry_unreadable"
            ) from exc
    if disk_rows[: len(telemetry_rows)] != telemetry_rows:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.resume.telemetry_mismatch"
        )
    _write_jsonl(telemetry_path, telemetry_rows)

    loader_iterator = iter(runtime.train_loader(0))
    try:
        for _ in range(applied):
            next(loader_iterator)
    except StopIteration as exc:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.training.schedule_exhausted"
        ) from exc
    restore_rng_state(entry_rng)
    initial_model_fingerprint = model_state_fingerprint(model)
    window: list[dict[str, object]] = []
    consecutive_skips = 0
    while applied < target:
        if stop_after_applied is not None and applied >= stop_after_applied:
            break
        try:
            cpu_batch = next(loader_iterator)
        except StopIteration as exc:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.training.schedule_exhausted"
            ) from exc
        expected_batch = identities[
            applied * batch_size : (applied + 1) * batch_size
        ]
        observed_batch = tuple(
            zip(cpu_batch.dataset_ids, cpu_batch.piece_ids, strict=True)
        )
        if observed_batch != expected_batch:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.training.schedule_identity_mismatch"
            )
        while True:
            attempted += 1
            batch = move_multisource_batch(
                cpu_batch,
                resolved_device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            _output, metric, was_skipped = training_engine._optimize_batch(
                model,
                batch,
                optimizer,
                scaler,
                config,
                resolved_device,
                collect_gradient_evidence=False,
                collect_update_metric=True,
                categorical_class_weights=class_weights,
            )
            if not isinstance(metric, dict):
                raise Phase9CCContinuationError(
                    "phase9cc.continuation.training.update_metric_missing"
                )
            if was_skipped:
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips > int(schedule["maximum_consecutive_skips"]):
                    raise Phase9CCContinuationError(
                        "phase9cc.continuation.training.persistent_amp_overflow"
                    )
                continue
            consecutive_skips = 0
            applied += 1
            window.append(metric)
            break
        if applied % telemetry_interval == 0:
            row = _telemetry_row(
                window=window,
                applied=applied,
                attempted=attempted,
                skipped=skipped,
                optimizer=optimizer,
                scaler=scaler,
                schedule_identities=identities,
                batch_size=batch_size,
            )
            if (
                not plan["protocol"].get("bounded_test_protocol")
                and any(value is None for value in row["mean_task_losses"].values())
            ):
                raise Phase9CCContinuationError(
                    "phase9cc.continuation.telemetry.production_task_window_empty"
                )
            telemetry_rows.append(row)
            _write_jsonl(telemetry_path, telemetry_rows)
            window.clear()
        if applied % checkpoint_interval == 0:
            payload = _checkpoint_payload(
                plan=plan,
                cell=cell,
                config=config,
                runtime=runtime,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                applied=applied,
                attempted=attempted,
                skipped=skipped,
                schedule_prefix_fingerprint=(
                    raw_downstream_sample_schedule_fingerprint(
                        identities[: applied * batch_size]
                    )
                ),
                telemetry_rows=telemetry_rows,
            )
            _save_torch(checkpoints / f"update-{applied}.pt", payload)

    complete = applied == target
    actual_fingerprint = raw_downstream_sample_schedule_fingerprint(
        identities[: applied * batch_size]
    )
    if complete and actual_fingerprint != schedule["full_schedule_fingerprint"]:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.training.final_schedule_mismatch"
        )
    parent_training = _read(
        Path(plan["protocol"]["parent_binding"]["root"])
        / "cells"
        / str(cell["cell_id"])
        / "training_report.json"
    )
    report = {
        "contract_version": CONTINUATION_TRAINING_REPORT_VERSION,
        "cell_id": cell["cell_id"],
        "epoch": 0,
        "start_applied_update": start,
        "target_applied_update": target,
        "additional_applied_updates": applied - start,
        "applied_updates": applied,
        "attempted_updates": attempted,
        "skipped_updates": skipped,
        "parent_attempted_updates": cell["parent_checkpoint"][
            "attempted_updates"
        ],
        "parent_skipped_updates": cell["parent_checkpoint"]["skipped_updates"],
        "complete": complete,
        "sample_schedule_position": applied,
        "full_sample_schedule_fingerprint": schedule[
            "full_schedule_fingerprint"
        ],
        "actual_sample_schedule_fingerprint": actual_fingerprint,
        "continuation_schedule_fingerprint": schedule[
            "continuation_schedule_fingerprint"
        ],
        "telemetry_interval_applied": telemetry_interval,
        "telemetry_row_count": len(telemetry_rows),
        "telemetry_fingerprint": fingerprint(telemetry_rows),
        "initial_or_resume_model_state_fingerprint": initial_model_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "parent_checkpoint_sha256": cell["parent_checkpoint"]["sha256"],
        "parent_transfer": parent_training["transfer"],
        "restore_mode": "model_optimizer_scaler_scheduler_rng_sampler",
        "encoder_export_reloaded": False,
        "data_fingerprints": runtime.fingerprints,
        "validation_membership": asdict(runtime.validation_membership),
        "resume_evidence": {
            "resumed_from": resumed_from,
            "checkpoint_updates": sorted(
                int(path.stem.split("-", 1)[1])
                for path in checkpoints.glob("update-*.pt")
            ),
            "loader_advanced_before_rng_restore": True,
        },
        "test_access": False,
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write_json(output / "training_report.json", report)
    return report


__all__ = [
    "CONTINUATION_CHECKPOINT_VERSION",
    "continuation_training_config",
    "run_cell_training",
    "schedule_identities",
]
