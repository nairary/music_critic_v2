"""Applied-update training runtime for the Phase 9C-C diagnostic.

This module intentionally leaves the generic Phase 6C epoch checkpoint
contract unchanged.  It reuses the canonical data/model/optimizer step while
adding a protocol-bound mid-epoch checkpoint for one continuous epoch.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    build_raw_downstream_sample_schedule,
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.training import engine as training_engine
from music_critic.training.checkpoint import capture_rng_state, restore_rng_state
from music_critic.training.config import DataConfig
from music_critic.training.data import build_corpus_data_views
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import model_contract_metadata

from .contracts import PHASE9CC_TASKS, Phase9CCError, canonical_bytes, fingerprint


PHASE9CC_CHECKPOINT_VERSION = "1.0.0"
PHASE9CC_TELEMETRY_VERSION = "1.0.0"


def _write_json(path: Path, value: object) -> None:
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


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _save_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def model_state_fingerprint(model_or_state: object) -> str:
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, torch.nn.Module)
        else model_or_state
    )
    if not isinstance(state, Mapping):
        raise Phase9CCError("phase9cc.model_state.mapping_required")
    digest = sha256()
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise Phase9CCError("phase9cc.model_state.tensor_mapping_required")
        detached = value.detach().to(device="cpu").contiguous()
        digest.update(
            canonical_bytes(
                {
                    "name": name,
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                }
            )
        )
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def training_config(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    protocol = plan["protocol"]
    schedule = protocol["schedule"]
    bindings = protocol["bindings"]
    production = not bool(protocol.get("bounded_test_protocol"))
    use_cuda = device.startswith("cuda")
    if production and not use_cuda:
        raise Phase9CCError("phase9cc.training.production_cuda_required")
    transfer = {
        "contract_version": "1.2.0",
        "mode": cell["transfer_mode"],
        "encoder_export_path": "",
        "encoder_export_sha256": "",
        "source_ssl_checkpoint_sha256": "",
        "source_kind": "phase7a_ssl",
        "comparison_protocol_fingerprint": protocol["fingerprint"],
        "downstream_initialization_seed": schedule[
            "downstream_initialization_seed"
        ],
        "downstream_data_order_seed": schedule[
            "downstream_data_order_seed"
        ],
        "actual_sample_schedule_path": "",
        "sample_schedule_fingerprint": schedule[
            "sample_schedule_fingerprint"
        ],
        "logical_updates": schedule["required_applied_updates"],
    }
    if cell["encoder_initialization"] == "ssl":
        checkpoint = bindings["ssl_checkpoint"]
        transfer.update(
            {
                "encoder_export_path": checkpoint["encoder_export_path"],
                "encoder_export_sha256": checkpoint[
                    "encoder_export_sha256"
                ],
                "source_ssl_checkpoint_sha256": checkpoint["sha256"],
                "source_kind": checkpoint["source_kind"],
            }
        )
    return {
        "seed": 17,
        "output_dir": str(output),
        "model": copy.deepcopy(protocol["model"]),
        "data": {
            "name": "dilemmadata",
            "index_paths": [bindings["raw_index"]["path"]],
            "cache_roots": [bindings["raw_cache_root"]],
            "split_manifest": bindings["split_manifest"]["path"],
            "target_cache_index": bindings["target_index"]["path"],
            "target_cache_root": bindings["target_cache_root"],
            "require_target_sidecars": True,
            "train_split": "train",
            "validation_split": "validation",
            "batch_size": schedule["batch_size"],
            "workers": 0,
            "epoch_size": schedule["epoch_size"],
            "validation_epoch_size": 0,
            "validation_seed": -1,
            "mixture_weights": {"dilemmadata": 1.0},
        },
        "experiment": {
            "name": "dilemmadata_scratch_vs_ssl",
            "preset": "dilemmadata_scratch_vs_ssl",
            "steps": schedule["required_applied_updates"],
            "epochs": 1,
            "checkpoint_interval": 1,
            "validation_interval": 1,
            "resume_from": "",
            "overwrite_output": False,
            "default_learning_rate": schedule["learning_rate"],
            "default_objective": "supervised_harmonic",
            "default_harmonic_weight": 1.0,
            "default_reconstruction_weight": 0.0,
            "collect_gradient_evidence": False,
            "optimizer_steps_per_epoch": schedule[
                "optimizer_steps_per_epoch"
            ],
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": schedule["learning_rate"],
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "objective": {
            "name": "supervised_harmonic",
            "harmonic_weight": 1.0,
            "reconstruction_weight": 0.0,
            "task_weights": {task: 1.0 for task in PHASE9CC_TASKS},
            "class_weight_artifact_path": bindings[
                "class_weight_artifact"
            ]["path"],
        },
        "scheduler": {"name": "none", "minimum_learning_rate": 0.0},
        "device": {
            "name": device,
            "amp": use_cuda,
            "non_blocking": False,
            "amp_dtype": "float16",
        },
        "transfer": transfer,
        "downstream_task_ids": list(PHASE9CC_TASKS),
    }


def _schedule(plan: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    protocol = plan["protocol"]
    schedule = protocol["schedule"]
    if protocol.get("bounded_test_protocol") and isinstance(
        protocol.get("bounded_schedule_identities"), list
    ):
        identities = tuple(
            tuple(value) for value in protocol["bounded_schedule_identities"]
        )
        if (
            raw_downstream_sample_schedule_fingerprint(identities)
            != schedule["sample_schedule_fingerprint"]
        ):
            raise Phase9CCError(
                "phase9cc.training.bounded_schedule_binding_mismatch"
            )
        return identities
    bindings = protocol["bindings"]
    data = DataConfig(
        name="dilemmadata",
        index_paths=[bindings["raw_index"]["path"]],
        cache_roots=[bindings["raw_cache_root"]],
        split_manifest=bindings["split_manifest"]["path"],
        target_cache_index=bindings["target_index"]["path"],
        target_cache_root=bindings["target_cache_root"],
        require_target_sidecars=True,
        batch_size=int(schedule["batch_size"]),
        workers=0,
        epoch_size=int(schedule["epoch_size"]),
        validation_epoch_size=0,
        mixture_weights={"dilemmadata": 1.0},
    )
    views = build_corpus_data_views(data)
    built = build_raw_downstream_sample_schedule(
        views.train,
        weights={"dilemmadata": 1.0},
        seed=int(schedule["downstream_data_order_seed"]),
        first_epoch=0,
        epochs=1,
        steps_per_epoch=int(schedule["optimizer_steps_per_epoch"]),
        batch_size=int(schedule["batch_size"]),
    )
    if built.fingerprint != schedule["sample_schedule_fingerprint"]:
        raise Phase9CCError("phase9cc.training.schedule_rebuild_mismatch")
    return built.identities


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
    return {
        "metadata": {
            "phase9cc_checkpoint_version": PHASE9CC_CHECKPOINT_VERSION,
            "model_contract": model_contract_metadata(model),
            "plan_fingerprint": plan["fingerprint"],
            "protocol_fingerprint": plan["protocol"]["fingerprint"],
            "cell_id": cell["cell_id"],
            "schedule_fingerprint": plan["protocol"]["schedule"][
                "sample_schedule_fingerprint"
            ],
            "data_fingerprints": runtime.fingerprints,
            "config_fingerprint": fingerprint(config),
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


def _load_checkpoint(
    path: Path,
    *,
    expected: Mapping[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Phase9CCError("phase9cc.resume.checkpoint_unreadable") from exc
    required = {
        "metadata",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "progress",
        "telemetry_rows",
        "model_state_fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise Phase9CCError("phase9cc.resume.checkpoint_fields_invalid")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(name) != value for name, value in expected.items()
    ):
        raise Phase9CCError("phase9cc.resume.checkpoint_binding_mismatch")
    if metadata.get("model_contract") != model_contract_metadata(model):
        raise Phase9CCError("phase9cc.resume.model_contract_mismatch")
    if model_state_fingerprint(payload["model_state"]) != payload.get(
        "model_state_fingerprint"
    ):
        raise Phase9CCError("phase9cc.resume.model_state_fingerprint_invalid")
    progress = payload.get("progress")
    rows = payload.get("telemetry_rows")
    if (
        not isinstance(progress, dict)
        or not isinstance(rows, list)
        or progress.get("telemetry_row_count") != len(rows)
        or progress.get("telemetry_fingerprint") != fingerprint(rows)
        or progress.get("epoch") != 0
        or progress.get("schedule_position")
        != progress.get("applied_updates")
    ):
        raise Phase9CCError("phase9cc.resume.progress_invalid")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is None:
            if payload["scheduler_state"] is not None:
                raise ValueError("unexpected scheduler")
        else:
            scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload["scaler_state"])
    except Exception as exc:
        raise Phase9CCError("phase9cc.resume.state_invalid") from exc
    return payload


def _latest_checkpoint(directory: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("update-*.pt"):
        try:
            update = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((update, path))
    if not candidates:
        raise Phase9CCError("phase9cc.resume.checkpoint_missing")
    return max(candidates)[1]


def _mean(values: list[float]) -> float:
    if not values:
        raise Phase9CCError("phase9cc.telemetry.empty_window")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise Phase9CCError("phase9cc.telemetry.non_finite")
    return result


def _telemetry_row(
    *,
    window: list[dict[str, object]],
    applied: int,
    attempted: int,
    skipped: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    schedule_identities: tuple[tuple[str, str], ...],
    batch_size: int,
) -> dict[str, object]:
    task_values = {
        task: [
            float(row["task_losses"][task])
            for row in window
            if task in row["task_losses"]
        ]
        for task in PHASE9CC_TASKS
    }
    task_losses = {
        task: (_mean(values) if values else None)
        for task, values in task_values.items()
    }
    gradient_norms = [
        float(row["gradient_norm_before_clip"]) for row in window
    ]
    if any(not math.isfinite(value) for value in gradient_norms):
        raise Phase9CCError("phase9cc.telemetry.gradient_norm_non_finite")
    prefix = schedule_identities[: applied * batch_size]
    return {
        "contract_version": PHASE9CC_TELEMETRY_VERSION,
        "epoch": 0,
        "applied_updates": applied,
        "attempted_updates": attempted,
        "skipped_updates": skipped,
        "window_applied_updates": len(window),
        "mean_objective_loss": _mean(
            [float(row["total_loss"]) for row in window]
        ),
        "mean_task_losses": task_losses,
        "task_loss_observation_counts": {
            task: len(values) for task, values in task_values.items()
        },
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "grad_scaler_scale": float(scaler.get_scale()),
        "mean_gradient_norm_before_clip": _mean(gradient_norms),
        "last_gradient_norm_before_clip": gradient_norms[-1],
        "sample_schedule_position": applied,
        "sample_count_consumed": len(prefix),
        "sample_schedule_prefix_fingerprint": (
            raw_downstream_sample_schedule_fingerprint(prefix)
        ),
    }


def run_cell_training(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    action: str,
    device: str,
    stop_after_applied: int | None = None,
    telemetry_enabled: bool = True,
) -> dict[str, object]:
    """Train one cell continuously, or resume its last committed update."""

    if action not in {"run", "resume"}:
        raise Phase9CCError("phase9cc.training.action_invalid")
    config = training_config(plan, cell, output, device=device)
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
        raise Phase9CCError("phase9cc.training.scheduler_must_be_none")
    schedule_contract = plan["protocol"]["schedule"]
    identities = _schedule(plan)
    batch_size = int(schedule_contract["batch_size"])
    required = int(schedule_contract["required_applied_updates"])
    telemetry_interval = int(
        schedule_contract["telemetry_interval_applied"]
    )
    checkpoint_interval = int(
        schedule_contract["checkpoint_interval_applied"]
    )
    milestones = set(schedule_contract["validation_milestones"])
    checkpoints = output / "checkpoints"
    telemetry_path = output / "train_telemetry.jsonl"
    expected_metadata = {
        "phase9cc_checkpoint_version": PHASE9CC_CHECKPOINT_VERSION,
        "plan_fingerprint": plan["fingerprint"],
        "protocol_fingerprint": plan["protocol"]["fingerprint"],
        "cell_id": cell["cell_id"],
        "schedule_fingerprint": schedule_contract[
            "sample_schedule_fingerprint"
        ],
        "data_fingerprints": runtime.fingerprints,
        "config_fingerprint": fingerprint(config),
        "resume_boundary": "applied_update_mid_epoch",
    }
    if action == "run":
        if output.exists() and any(output.iterdir()):
            raise Phase9CCError("phase9cc.training.fresh_output_required")
        output.mkdir(parents=True, exist_ok=True)
        applied = attempted = skipped = 0
        telemetry_rows: list[dict[str, object]] = []
        initial = _checkpoint_payload(
            plan=plan,
            cell=cell,
            config=config,
            runtime=runtime,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            applied=0,
            attempted=0,
            skipped=0,
            schedule_prefix_fingerprint=(
                raw_downstream_sample_schedule_fingerprint(())
            ),
            telemetry_rows=telemetry_rows,
        )
        _save_torch(checkpoints / "update-0.pt", initial)
        entry_rng = capture_rng_state()
        resumed_from_update = None
    else:
        checkpoint = _latest_checkpoint(checkpoints)
        payload = _load_checkpoint(
            checkpoint,
            expected=expected_metadata,
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
        disk_rows = []
        if telemetry_path.is_file():
            try:
                disk_rows = [
                    json.loads(line)
                    for line in telemetry_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line
                ]
            except (OSError, json.JSONDecodeError) as exc:
                raise Phase9CCError(
                    "phase9cc.resume.telemetry_unreadable"
                ) from exc
        if disk_rows[: len(telemetry_rows)] != telemetry_rows:
            raise Phase9CCError("phase9cc.resume.telemetry_mismatch")
        _write_jsonl(telemetry_path, telemetry_rows)
        entry_rng = payload["rng_state"]
        resumed_from_update = applied

    loader_iterator = iter(runtime.train_loader(0))
    try:
        for _ in range(applied):
            next(loader_iterator)
    except StopIteration as exc:
        raise Phase9CCError("phase9cc.training.schedule_exhausted") from exc
    restore_rng_state(entry_rng)
    window: list[dict[str, object]] = []
    consecutive_skips = 0
    initial_model_fingerprint = model_state_fingerprint(model)
    transfer = copy.deepcopy(config["phase8b2_transfer_runtime"])
    while applied < required:
        if stop_after_applied is not None and applied >= stop_after_applied:
            break
        try:
            cpu_batch = next(loader_iterator)
        except StopIteration as exc:
            raise Phase9CCError("phase9cc.training.schedule_exhausted") from exc
        expected_batch = identities[
            applied * batch_size : (applied + 1) * batch_size
        ]
        observed_batch = tuple(
            zip(cpu_batch.dataset_ids, cpu_batch.piece_ids, strict=True)
        )
        if observed_batch != expected_batch:
            raise Phase9CCError("phase9cc.training.schedule_identity_mismatch")
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
                collect_update_metric=telemetry_enabled,
                categorical_class_weights=class_weights,
            )
            if telemetry_enabled and not isinstance(metric, dict):
                raise Phase9CCError("phase9cc.training.update_metric_missing")
            if was_skipped:
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips > int(
                    schedule_contract["maximum_consecutive_skips"]
                ):
                    raise Phase9CCError(
                        "phase9cc.training.persistent_amp_overflow"
                    )
                continue
            consecutive_skips = 0
            applied += 1
            if metric is not None:
                window.append(metric)
            break
        if applied % telemetry_interval == 0:
            if telemetry_enabled:
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
                    and any(
                        value is None
                        for value in row["mean_task_losses"].values()
                    )
                ):
                    raise Phase9CCError(
                        "phase9cc.telemetry.production_task_window_empty"
                    )
                telemetry_rows.append(row)
                _write_jsonl(telemetry_path, telemetry_rows)
            window.clear()
        if applied % checkpoint_interval == 0 or applied in milestones:
            prefix_fingerprint = raw_downstream_sample_schedule_fingerprint(
                identities[: applied * batch_size]
            )
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
                schedule_prefix_fingerprint=prefix_fingerprint,
                telemetry_rows=telemetry_rows,
            )
            _save_torch(checkpoints / f"update-{applied}.pt", payload)

    complete = applied == required
    actual_schedule_fingerprint = raw_downstream_sample_schedule_fingerprint(
        identities[: applied * batch_size]
    )
    if complete and actual_schedule_fingerprint != schedule_contract[
        "sample_schedule_fingerprint"
    ]:
        raise Phase9CCError("phase9cc.training.final_schedule_mismatch")
    report = {
        "contract_version": "1.0.0",
        "cell_id": cell["cell_id"],
        "epochs": 1,
        "required_applied_updates": required,
        "applied_updates": applied,
        "attempted_updates": attempted,
        "skipped_updates": skipped,
        "complete": complete,
        "sample_schedule_fingerprint": schedule_contract[
            "sample_schedule_fingerprint"
        ],
        "actual_sample_schedule_fingerprint": actual_schedule_fingerprint,
        "sample_schedule_position": applied,
        "telemetry_interval_applied": telemetry_interval,
        "telemetry_row_count": len(telemetry_rows),
        "telemetry_fingerprint": fingerprint(telemetry_rows),
        "initial_or_resume_model_state_fingerprint": initial_model_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "fresh_supervised_initialization_fingerprint": transfer.get(
            "fresh_supervised_initialization_fingerprint"
        ),
        "transfer": transfer,
        "data_fingerprints": runtime.fingerprints,
        "validation_membership": asdict(runtime.validation_membership),
        "resume_boundary": "applied_update_mid_epoch",
        "resume_evidence": {
            "resumed_from_update": resumed_from_update,
            "checkpoint_updates": sorted(
                int(path.stem.split("-", 1)[1])
                for path in checkpoints.glob("update-*.pt")
            ),
            "schedule_rebuilt_then_rng_restored": True,
        },
        "test_access": False,
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write_json(output / "training_report.json", report)
    return report


__all__ = [
    "PHASE9CC_CHECKPOINT_VERSION",
    "model_state_fingerprint",
    "run_cell_training",
    "training_config",
]
