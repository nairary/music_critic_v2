"""Deterministic one-batch and epoch-boundary Phase 6C runners."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from music_critic.models import ACTIVE_TASK_IDS
from music_critic.tasks import MultiSourceBatch
from music_critic.training.checkpoint import (
    TRAINING_CHECKPOINT_VERSION,
    load_training_checkpoint,
    save_training_checkpoint,
    training_checkpoint_metadata,
)
from music_critic.training.data import DataRuntime, build_data_runtime
from music_critic.training.device import move_multisource_batch
from music_critic.training.metrics import EpochMetricAccumulator
from music_critic.training.models import (
    BaselineModel,
    build_baseline_model,
    model_contract_fingerprint,
)


class TrainingContractError(ValueError):
    """Stable Phase 6C configuration or runtime contract failure."""


class InjectedTrainingCrash(RuntimeError):
    """Bounded-test crash at a named epoch-commit boundary."""


def _plain_config(config: object) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, dict):
        value = copy.deepcopy(config)
    else:
        raise TrainingContractError("training.config.type_invalid")
    if not isinstance(value, dict):
        raise TrainingContractError("training.config.root_invalid")
    value.pop("defaults", None)
    return value


def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude only the location used to request an otherwise exact resume."""

    result = copy.deepcopy(config)
    result["experiment"]["resume_from"] = ""
    return result


def _resolve_presets(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    optimizer = config["optimizer"]
    objective = config["objective"]
    if optimizer["learning_rate"] is None:
        optimizer["learning_rate"] = experiment[
            "default_learning_rate"
        ]
    if objective["name"] == "preset":
        objective["name"] = experiment["default_objective"]
        objective["harmonic_weight"] = experiment[
            "default_harmonic_weight"
        ]
        objective["reconstruction_weight"] = experiment[
            "default_reconstruction_weight"
        ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_config(config: dict[str, Any]) -> None:
    accepted = {
        "model": {"feature_only", "local_gnn", "hierarchical"},
        "data": {"bounded", "hooktheory", "pop909_cl", "mixed"},
        "experiment": {
            "one_batch",
            "smoke",
            "train",
            "supervised_baseline",
            "joint_visible_reconstruction",
        },
        "optimizer": {"adamw"},
        "objective": {
            "one_batch_joint",
            "supervised_harmonic",
            "joint_visible_reconstruction",
        },
        "scheduler": {"none", "cosine"},
        "device": {"cpu", "cuda", "auto"},
    }
    for group, names in accepted.items():
        if config[group]["name"] not in names:
            raise TrainingContractError(
                f"training.config.{group}_invalid"
            )
    integers = (
        ("seed", config["seed"], 0),
        ("batch_size", config["data"]["batch_size"], 1),
        ("workers", config["data"]["workers"], 0),
        ("epoch_size", config["data"]["epoch_size"], 1),
        (
            "validation_epoch_size",
            config["data"]["validation_epoch_size"],
            0,
        ),
        ("steps", config["experiment"]["steps"], 1),
        ("epochs", config["experiment"]["epochs"], 1),
        (
            "checkpoint_interval",
            config["experiment"]["checkpoint_interval"],
            1,
        ),
        (
            "validation_interval",
            config["experiment"]["validation_interval"],
            1,
        ),
    )
    for name, value, minimum in integers:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise TrainingContractError(
                f"training.config.{name}_invalid"
            )
    positive = (
        ("learning_rate", config["optimizer"]["learning_rate"]),
        (
            "gradient_clip_norm",
            config["optimizer"]["gradient_clip_norm"],
        ),
    )
    for name, value in positive:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise TrainingContractError(
                f"training.config.{name}_invalid"
            )
    weight_decay = config["optimizer"]["weight_decay"]
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise TrainingContractError(
            "training.config.weight_decay_invalid"
        )
    weights = config["data"]["mixture_weights"]
    if not weights or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for key, value in weights.items()
    ):
        raise TrainingContractError(
            "training.config.mixture_weights_invalid"
        )
    objective_weights = (
        config["objective"]["harmonic_weight"],
        config["objective"]["reconstruction_weight"],
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in objective_weights
    ) or not any(value > 0 for value in objective_weights):
        raise TrainingContractError(
            "training.config.objective_weights_invalid"
        )
    task_weights = config["objective"]["task_weights"]
    if any(
        task_id not in ACTIVE_TASK_IDS
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
        for task_id, weight in task_weights.items()
    ):
        raise TrainingContractError(
            "training.config.task_weights_invalid"
        )


def _resolve_device(config: dict[str, Any]) -> torch.device:
    name = config["device"]["name"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainingContractError("training.device.cuda_unavailable")
    if config["device"]["amp"] and name != "cuda":
        raise TrainingContractError("training.device.amp_requires_cuda")
    return torch.device(name)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _optimizer(
    model: BaselineModel, config: dict[str, Any]
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    one_batch: bool,
) -> Any:
    if config["scheduler"]["name"] == "none":
        return None
    duration = (
        config["experiment"]["steps"]
        if one_batch
        else config["experiment"]["epochs"]
    )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=duration,
        eta_min=float(config["scheduler"]["minimum_learning_rate"]),
    )


def _losses(
    output: Any,
    config: dict[str, Any],
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    task_weights = config["objective"]["task_weights"]
    active_tasks = [
        (
            supervision.per_row_loss.mean(),
            float(task_weights.get(supervision.task_id, 1.0)),
        )
        for supervision in output.supervisions
        if supervision.per_row_loss.numel()
    ]
    harmonic = None
    if active_tasks:
        harmonic = torch.stack(
            [loss * weight for loss, weight in active_tasks]
        ).sum() / sum(weight for _, weight in active_tasks)
    reconstruction = output.reconstruction_loss
    terms = []
    if (
        harmonic is not None
        and config["objective"]["harmonic_weight"] > 0
    ):
        terms.append(
            harmonic * config["objective"]["harmonic_weight"]
        )
    if (
        reconstruction is not None
        and config["objective"]["reconstruction_weight"] > 0
    ):
        terms.append(
            reconstruction
            * config["objective"]["reconstruction_weight"]
        )
    total = None if not terms else torch.stack(terms).sum()
    if total is not None:
        torch._assert_async(
            torch.isfinite(total),
            "training.loss.non_finite",
        )
    return harmonic, reconstruction, total


def _scalar(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach())


def _gradient_evidence(model: BaselineModel) -> dict[str, object]:
    covered = []
    missing = []
    non_finite = []
    modules: Counter[str] = Counter()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None or not bool(torch.count_nonzero(gradient)):
            missing.append(name)
            continue
        if not bool(torch.isfinite(gradient).all()):
            non_finite.append(name)
            continue
        covered.append(name)
        modules[name.split(".", 1)[0]] += 1
    if non_finite:
        raise TrainingContractError(
            "training.gradient.non_finite:" + ",".join(non_finite)
        )
    return {
        "covered_parameter_count": len(covered),
        "trainable_parameter_count": len(covered) + len(missing),
        "covered_parameters": covered,
        "missing_parameters": missing,
        "covered_top_level_modules": dict(sorted(modules.items())),
    }


def _candidate_counts(output: Any) -> dict[str, int]:
    return {
        prediction.task_id: int(prediction.logits.shape[0])
        for prediction in output.predictions
    }


def _availability_counts(output: Any) -> dict[str, int]:
    return {
        supervision.task_id: int(supervision.per_row_loss.shape[0])
        for supervision in output.supervisions
    }


def _task_losses(output: Any) -> dict[str, float]:
    return {
        item.task_id: float(item.mean_loss.detach())
        for item in output.harmonic_loss.task_losses
    }


def _eval_logits(
    model: BaselineModel, batch: MultiSourceBatch
) -> tuple[tuple[str, torch.Tensor], ...]:
    model.eval()
    with torch.no_grad():
        output = model(batch)
    return tuple(
        (prediction.task_id, prediction.logits.detach().cpu().clone())
        for prediction in output.predictions
    )


def _equal_logits(
    left: tuple[tuple[str, torch.Tensor], ...],
    right: tuple[tuple[str, torch.Tensor], ...],
) -> bool:
    return len(left) == len(right) and all(
        left_id == right_id and torch.equal(left_value, right_value)
        for (left_id, left_value), (right_id, right_value) in zip(
            left, right, strict=True
        )
    )


def _optimize_batch(
    model: BaselineModel,
    batch: MultiSourceBatch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    *,
    collect_gradient_evidence: bool,
) -> tuple[Any, dict[str, object] | None, bool]:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=bool(config["device"]["amp"]),
    ):
        output = model(
            batch,
            include_reconstruction=(
                config["objective"]["reconstruction_weight"] > 0
            ),
        )
        harmonic, reconstruction, total = _losses(output, config)
    if total is None:
        return output, None, True
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["optimizer"]["gradient_clip_norm"]),
        error_if_nonfinite=collect_gradient_evidence,
    )
    gradient = (
        _gradient_evidence(model)
        if collect_gradient_evidence
        else None
    )
    scaler.step(optimizer)
    scaler.update()
    if not collect_gradient_evidence:
        return output, None, False
    return output, {
        "harmonic_loss": _scalar(harmonic),
        "reconstruction_loss": _scalar(reconstruction),
        "total_loss": _scalar(total),
        "gradient_norm_before_clip": float(clipped_norm),
        "gradient_coverage": gradient,
        "task_losses": _task_losses(output),
        "availability_counts": _availability_counts(output),
    }, False


def _validation_epoch(
    model: BaselineModel,
    batches: Iterable[MultiSourceBatch],
    *,
    config: dict[str, Any],
    device: torch.device,
    membership_evidence: dict[str, object],
) -> dict[str, object]:
    model.eval()
    accumulator = EpochMetricAccumulator(
        harmonic_weight=config["objective"]["harmonic_weight"],
        reconstruction_weight=config["objective"][
            "reconstruction_weight"
        ],
        task_weights=config["objective"]["task_weights"],
    )
    with torch.no_grad():
        for cpu_batch in batches:
            batch = move_multisource_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=bool(config["device"]["amp"]),
            ):
                output = model(
                    batch,
                    include_reconstruction=(
                        config["objective"][
                            "reconstruction_weight"
                        ]
                        > 0
                    ),
                )
                _losses(output, config)
            accumulator.add(output, batch)
    result = accumulator.finalize()
    if result["batch_count"] == 0:
        raise TrainingContractError("training.validation.empty")
    result["membership"] = membership_evidence
    return result


def _device_evidence(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "resolved_device": str(device),
        "cuda_available": True,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _prepare(
    config: dict[str, Any],
) -> tuple[
    Path,
    torch.device,
    DataRuntime,
    BaselineModel,
    torch.optim.Optimizer,
    Any,
    torch.amp.GradScaler,
]:
    _validate_config(config)
    device = _resolve_device(config)
    _set_determinism(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime = build_data_runtime(
        OmegaConf.create(config["data"]), seed=config["seed"]
    )
    model = build_baseline_model(
        OmegaConf.create(config["model"]),
        task_weights=config["objective"]["task_weights"],
    ).to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(
        optimizer,
        config,
        one_batch=config["experiment"]["name"] == "one_batch",
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(config["device"]["amp"]),
    )
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output, device, runtime, model, optimizer, scheduler, scaler


def _artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: DataRuntime,
    model: BaselineModel,
) -> None:
    _write_json(output / "resolved_config.json", config)
    _write_json(
        output / "fingerprints.json",
        {
            "data": runtime.fingerprints,
            "model_contract_fingerprint": model_contract_fingerprint(
                model
            ),
            "checkpoint_binding": training_checkpoint_metadata(
                model,
                resolved_config=_checkpoint_config(config),
                data_fingerprints=runtime.fingerprints,
            ),
        },
    )
    _write_json(
        output / "mixture_statistics.json",
        runtime.mixture_statistics,
    )


def _run_one_batch(config: dict[str, Any]) -> dict[str, object]:
    (
        output_dir,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
    ) = _prepare(config)
    _artifacts(output_dir, config, runtime, model)
    batch = move_multisource_batch(
        runtime.first_train_batch,
        device,
        non_blocking=bool(config["device"]["non_blocking"]),
    )
    model.eval()
    with torch.no_grad():
        initial_output = model(
            batch,
            include_reconstruction=(
                config["objective"]["reconstruction_weight"] > 0
            ),
        )
        initial_harmonic, initial_reconstruction, _ = _losses(
            initial_output, config
        )
    initial = {
        "harmonic_loss": _scalar(initial_harmonic),
        "reconstruction_loss": _scalar(initial_reconstruction),
    }
    curve = []
    final_gradient = None
    for step in range(config["experiment"]["steps"]):
        model.train()
        _, metric, skipped = _optimize_batch(
            model,
            batch,
            optimizer,
            scaler,
            config,
            device,
            collect_gradient_evidence=True,
        )
        if skipped or metric is None:
            raise TrainingContractError(
                "training.one_batch.objective_unavailable"
            )
        metric["step"] = step
        metric["learning_rate"] = optimizer.param_groups[0]["lr"]
        curve.append(metric)
        final_gradient = metric["gradient_coverage"]
        if scheduler is not None:
            scheduler.step()
    final_logits = _eval_logits(model, batch)
    model.eval()
    with torch.no_grad():
        final_output = model(
            batch,
            include_reconstruction=(
                config["objective"]["reconstruction_weight"] > 0
            ),
        )
        final_harmonic, final_reconstruction, final_total = _losses(
            final_output, config
        )
    final = {
        "harmonic_loss": _scalar(final_harmonic),
        "reconstruction_loss": _scalar(final_reconstruction),
        "total_loss": _scalar(final_total),
    }
    if initial["harmonic_loss"] is None or initial[
        "reconstruction_loss"
    ] is None:
        raise TrainingContractError(
            "training.one_batch.requires_both_objectives"
        )
    if not (
        final["harmonic_loss"] < initial["harmonic_loss"]
        and final["reconstruction_loss"]
        < initial["reconstruction_loss"]
    ):
        raise TrainingContractError(
            "training.one_batch.objectives_did_not_both_decrease"
        )
    checkpoint = output_dir / "one_batch.pt"
    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        committed_metric_rows=0,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    reloaded = build_baseline_model(
        OmegaConf.create(config["model"]),
        task_weights=config["objective"]["task_weights"],
    ).to(device)
    reloaded_optimizer = _optimizer(reloaded, config)
    reloaded_scheduler = _scheduler(
        reloaded_optimizer, config, one_batch=True
    )
    reloaded_scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(config["device"]["amp"]),
    )
    load_training_checkpoint(
        checkpoint,
        reloaded,
        reloaded_optimizer,
        scheduler=reloaded_scheduler,
        scaler=reloaded_scaler,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    reloaded_logits = _eval_logits(reloaded, batch)
    if not _equal_logits(final_logits, reloaded_logits):
        raise TrainingContractError(
            "training.one_batch.checkpoint_logits_mismatch"
        )
    report = {
        "evidence_kind": "optimization_plumbing_not_generalization",
        "seed": config["seed"],
        "model": config["model"]["name"],
        "data": config["data"]["name"],
        "steps": config["experiment"]["steps"],
        "objective": config["objective"],
        "initial": initial,
        "curve": curve,
        "final": final,
        "candidate_counts": _candidate_counts(final_output),
        "final_gradient_coverage": final_gradient,
        "checkpoint": str(checkpoint),
        "checkpoint_reload_bit_exact": True,
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "optimizer_step_count": config["experiment"]["steps"],
        "device": _device_evidence(device),
        "fingerprints": runtime.fingerprints,
    }
    _write_json(output_dir / "one_batch_report.json", report)
    return report


def _metric_directory(output_dir: Path) -> Path:
    return output_dir / "epoch_metrics"


def _pending_metric_path(output_dir: Path) -> Path:
    return _metric_directory(output_dir) / "pending.json"


def _committed_metric_path(
    output_dir: Path, next_epoch: int
) -> Path:
    return _metric_directory(output_dir) / (
        f"epoch-{next_epoch:04d}.json"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingContractError(
            f"training.journal.unreadable:{path.name}:{exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TrainingContractError(
            f"training.journal.invalid:{path.name}"
        )
    return value


def _committed_metric_envelopes(
    output_dir: Path,
) -> list[dict[str, Any]]:
    directory = _metric_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(directory.glob("epoch-*.json"))
    envelopes = []
    for expected, path in enumerate(files, start=1):
        if path != _committed_metric_path(output_dir, expected):
            raise TrainingContractError(
                "training.journal.non_contiguous"
            )
        envelope = _read_json(path)
        if envelope.get("next_epoch") != expected:
            raise TrainingContractError(
                "training.journal.epoch_mismatch"
            )
        envelopes.append(envelope)
    return envelopes


def _rebuild_metrics_jsonl(output_dir: Path) -> None:
    rows = [
        envelope["row"]
        for envelope in _committed_metric_envelopes(output_dir)
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TrainingContractError("training.journal.row_invalid")
    payload = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for row in rows
    )
    _write_text_atomic(output_dir / "metrics.jsonl", payload)


def _reset_metric_journal(output_dir: Path) -> None:
    directory = _metric_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("epoch-*.json"):
        path.unlink()
    _pending_metric_path(output_dir).unlink(missing_ok=True)
    _write_text_atomic(output_dir / "metrics.jsonl", "")


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _finalize_pending_metric(
    output_dir: Path,
    envelope: dict[str, Any],
) -> None:
    next_epoch = envelope.get("next_epoch")
    if (
        isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or next_epoch <= 0
        or not isinstance(envelope.get("row"), dict)
    ):
        raise TrainingContractError(
            "training.journal.pending_invalid"
        )
    last = output_dir / "last.pt"
    if envelope.get("is_best"):
        _atomic_copy(last, output_dir / "best.pt")
    if envelope.get("write_interval_checkpoint"):
        _atomic_copy(
            last, output_dir / f"epoch-{next_epoch:04d}.pt"
        )
    os.replace(
        _pending_metric_path(output_dir),
        _committed_metric_path(output_dir, next_epoch),
    )
    _rebuild_metrics_jsonl(output_dir)


def _recover_metric_journal(
    output_dir: Path,
    *,
    committed_metric_rows: int,
) -> None:
    committed = _committed_metric_envelopes(output_dir)
    pending_path = _pending_metric_path(output_dir)
    pending = _read_json(pending_path) if pending_path.exists() else None
    if committed_metric_rows == len(committed):
        # A staged metric without its checkpoint is not committed. The loaded
        # checkpoint restores the prior epoch and the epoch will be replayed.
        pending_path.unlink(missing_ok=True)
    elif (
        committed_metric_rows == len(committed) + 1
        and pending is not None
        and pending.get("next_epoch") == committed_metric_rows
    ):
        # The checkpoint is authoritative and the staged deterministic row is
        # now committed without re-running or duplicating the epoch.
        _finalize_pending_metric(output_dir, pending)
    else:
        raise TrainingContractError(
            "training.journal.checkpoint_metric_mismatch"
        )
    _rebuild_metrics_jsonl(output_dir)


def _commit_epoch(
    output_dir: Path,
    *,
    row: dict[str, object],
    is_best: bool,
    write_interval_checkpoint: bool,
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    best_validation_loss: float | None,
    config: dict[str, Any],
    runtime: DataRuntime,
    crash_after: str | None,
) -> None:
    next_epoch = int(row["next_epoch"])
    envelope = {
        "next_epoch": next_epoch,
        "row": row,
        "is_best": is_best,
        "write_interval_checkpoint": write_interval_checkpoint,
    }
    _write_json_atomic(_pending_metric_path(output_dir), envelope)
    if crash_after == "metric_write":
        raise InjectedTrainingCrash(
            "training.crash.after_metric_write"
        )
    save_training_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=next_epoch,
        best_validation_loss=best_validation_loss,
        committed_metric_rows=next_epoch,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    if crash_after == "checkpoint_write":
        raise InjectedTrainingCrash(
            "training.crash.after_checkpoint_write"
        )
    _finalize_pending_metric(output_dir, envelope)


def _run_epochs(
    config: dict[str, Any],
    *,
    stop_after_epoch: int | None,
    crash_after: str | None,
) -> dict[str, object]:
    (
        output_dir,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
    ) = _prepare(config)
    _artifacts(output_dir, config, runtime, model)
    resume = config["experiment"]["resume_from"]
    start_epoch = 0
    best: float | None = None
    if resume:
        (
            start_epoch,
            best,
            committed_metric_rows,
        ) = load_training_checkpoint(
            resume,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        )
        _recover_metric_journal(
            output_dir,
            committed_metric_rows=committed_metric_rows,
        )
    else:
        _reset_metric_journal(output_dir)
    epochs = int(config["experiment"]["epochs"])
    if start_epoch > epochs:
        raise TrainingContractError(
            "training.resume.epoch_beyond_config"
        )
    completed = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        train_accumulator = EpochMetricAccumulator(
            harmonic_weight=config["objective"]["harmonic_weight"],
            reconstruction_weight=config["objective"][
                "reconstruction_weight"
            ],
            task_weights=config["objective"]["task_weights"],
        )
        for cpu_batch in runtime.train_loader(epoch):
            batch = move_multisource_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            output, gradient_metric, skipped = _optimize_batch(
                model,
                batch,
                optimizer,
                scaler,
                config,
                device,
                collect_gradient_evidence=bool(
                    config["experiment"][
                        "collect_gradient_evidence"
                    ]
                ),
            )
            train_accumulator.gradient_evidence_scan_count += int(
                gradient_metric is not None
            )
            train_accumulator.add(output, batch, skipped=skipped)
        train_metric = train_accumulator.finalize()
        if train_metric["batch_count"] == 0:
            raise TrainingContractError("training.epoch.empty")
        if scheduler is not None:
            scheduler.step()
        validation = None
        if (
            (epoch + 1)
            % int(config["experiment"]["validation_interval"])
            == 0
            or epoch + 1 == epochs
        ):
            validation = _validation_epoch(
                model,
                runtime.validation_loader(),
                config=config,
                device=device,
                membership_evidence=asdict(
                    runtime.validation_membership
                ),
            )
        row = {
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metric,
            "validation": validation,
        }
        validation_objective = (
            None
            if validation is None
            else validation["objective_loss"]
        )
        is_best = bool(
            validation_objective is not None
            and (
                best is None
                or float(validation_objective) < best
            )
        )
        if is_best:
            best = float(validation_objective)
        write_interval_checkpoint = (
            (epoch + 1)
            % int(config["experiment"]["checkpoint_interval"])
            == 0
        )
        _commit_epoch(
            output_dir,
            row=row,
            is_best=is_best,
            write_interval_checkpoint=write_interval_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_validation_loss=best,
            config=config,
            runtime=runtime,
            crash_after=crash_after,
        )
        completed = epoch + 1
        if stop_after_epoch is not None and completed >= stop_after_epoch:
            break
    report = {
        "evidence_kind": "bounded_supervised_training_plumbing",
        "resume_boundary": "epoch_only",
        "mid_epoch_resume_supported": False,
        "start_epoch": start_epoch,
        "completed_epochs": completed,
        "configured_epochs": epochs,
        "best_validation_loss": best,
        "metrics": str(output_dir / "metrics.jsonl"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "device": _device_evidence(device),
        "fingerprints": runtime.fingerprints,
        "validation_membership": asdict(
            runtime.validation_membership
        ),
        "objective": config["objective"],
    }
    _write_json(output_dir / "training_report.json", report)
    return report


def run_training(
    config: object,
    *,
    stop_after_epoch: int | None = None,
    crash_after: str | None = None,
) -> dict[str, object]:
    """Run one accepted Phase 6C experiment.

    ``stop_after_epoch`` is a bounded-test hook that only stops immediately
    after a checkpointed epoch boundary; it does not implement mid-epoch
    resume.
    """

    plain = _plain_config(config)
    _resolve_presets(plain)
    if crash_after not in {
        None,
        "metric_write",
        "checkpoint_write",
    }:
        raise TrainingContractError(
            "training.crash.injection_point_invalid"
        )
    if plain["experiment"]["name"] == "one_batch":
        if stop_after_epoch is not None or crash_after is not None:
            raise TrainingContractError(
                "training.one_batch.test_hook_invalid"
            )
        return _run_one_batch(plain)
    return _run_epochs(
        plain,
        stop_after_epoch=stop_after_epoch,
        crash_after=crash_after,
    )


__all__ = [
    "TRAINING_CHECKPOINT_VERSION",
    "InjectedTrainingCrash",
    "TrainingContractError",
    "run_training",
]
