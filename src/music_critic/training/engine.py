"""Deterministic one-batch and epoch-boundary Phase 6C runners."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from music_critic.tasks import MultiSourceBatch
from music_critic.training.checkpoint import (
    TRAINING_CHECKPOINT_VERSION,
    load_training_checkpoint,
    save_training_checkpoint,
    training_checkpoint_metadata,
)
from music_critic.training.data import DataRuntime, build_data_runtime
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import (
    BaselineModel,
    build_baseline_model,
    model_contract_fingerprint,
)


class TrainingContractError(ValueError):
    """Stable Phase 6C configuration or runtime contract failure."""


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


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )


def _validate_config(config: dict[str, Any]) -> None:
    accepted = {
        "model": {"feature_only", "local_gnn", "hierarchical"},
        "data": {"bounded", "hooktheory", "pop909_cl", "mixed"},
        "experiment": {"one_batch", "smoke", "train"},
        "optimizer": {"adamw"},
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
            1,
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


def _losses(output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    harmonic = output.harmonic_loss.total_loss
    reconstruction = output.reconstruction_loss
    active = tuple(
        value for value in (harmonic, reconstruction) if value is not None
    )
    if not active:
        raise TrainingContractError("training.loss.no_active_objective")
    total = active[0]
    for value in active[1:]:
        total = total + value
    if not bool(torch.isfinite(total)):
        raise TrainingContractError("training.loss.non_finite")
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
) -> tuple[Any, dict[str, object]]:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=bool(config["device"]["amp"]),
    ):
        output = model(batch)
        harmonic, reconstruction, total = _losses(output)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    gradient = _gradient_evidence(model)
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["optimizer"]["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    return output, {
        "harmonic_loss": _scalar(harmonic),
        "reconstruction_loss": _scalar(reconstruction),
        "total_loss": _scalar(total),
        "gradient_norm_before_clip": float(clipped_norm),
        "gradient_coverage": gradient,
        "task_losses": _task_losses(output),
        "availability_counts": _availability_counts(output),
    }


def _validation_epoch(
    model: BaselineModel,
    batches: Iterable[MultiSourceBatch],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    totals: list[float] = []
    harmonic_values: list[float] = []
    reconstruction_values: list[float] = []
    task_values: dict[str, list[float]] = defaultdict(list)
    availability: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    with torch.no_grad():
        for cpu_batch in batches:
            datasets.update(cpu_batch.dataset_ids)
            batch = move_multisource_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=bool(config["device"]["amp"]),
            ):
                output = model(batch)
                harmonic, reconstruction, total = _losses(output)
            totals.append(float(total))
            if harmonic is not None:
                harmonic_values.append(float(harmonic))
            if reconstruction is not None:
                reconstruction_values.append(float(reconstruction))
            for key, value in _task_losses(output).items():
                task_values[key].append(value)
            availability.update(_availability_counts(output))
    if not totals:
        raise TrainingContractError("training.validation.empty")
    return {
        "total_loss": sum(totals) / len(totals),
        "harmonic_loss": (
            None
            if not harmonic_values
            else sum(harmonic_values) / len(harmonic_values)
        ),
        "reconstruction_loss": (
            None
            if not reconstruction_values
            else sum(reconstruction_values)
            / len(reconstruction_values)
        ),
        "task_losses": {
            key: sum(values) / len(values)
            for key, values in sorted(task_values.items())
        },
        "availability_counts": dict(sorted(availability.items())),
        "dataset_counts": dict(sorted(datasets.items())),
        "batch_count": len(totals),
    }


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
        OmegaConf.create(config["model"])
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
        initial_output = model(batch)
        initial_harmonic, initial_reconstruction, _ = _losses(
            initial_output
        )
    initial = {
        "harmonic_loss": _scalar(initial_harmonic),
        "reconstruction_loss": _scalar(initial_reconstruction),
    }
    curve = []
    final_gradient = None
    for step in range(config["experiment"]["steps"]):
        model.train()
        _, metric = _optimize_batch(
            model, batch, optimizer, scaler, config, device
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
        final_output = model(batch)
        final_harmonic, final_reconstruction, final_total = _losses(
            final_output
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
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    reloaded = build_baseline_model(
        OmegaConf.create(config["model"])
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
        "initial": initial,
        "curve": curve,
        "final": final,
        "candidate_counts": _candidate_counts(final_output),
        "final_gradient_coverage": final_gradient,
        "checkpoint": str(checkpoint),
        "checkpoint_reload_bit_exact": True,
        "device": _device_evidence(device),
        "fingerprints": runtime.fingerprints,
    }
    _write_json(output_dir / "one_batch_report.json", report)
    return report


def _run_epochs(
    config: dict[str, Any],
    *,
    stop_after_epoch: int | None,
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
    metrics_path = output_dir / "metrics.jsonl"
    resume = config["experiment"]["resume_from"]
    start_epoch = 0
    best: float | None = None
    if resume:
        start_epoch, best = load_training_checkpoint(
            resume,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        )
    else:
        metrics_path.write_text("", encoding="utf-8")
    epochs = int(config["experiment"]["epochs"])
    if start_epoch > epochs:
        raise TrainingContractError(
            "training.resume.epoch_beyond_config"
        )
    completed = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        totals: list[float] = []
        harmonic_values: list[float] = []
        reconstruction_values: list[float] = []
        task_values: dict[str, list[float]] = defaultdict(list)
        availability: Counter[str] = Counter()
        datasets: Counter[str] = Counter()
        final_gradient = None
        for cpu_batch in runtime.train_loader(epoch):
            datasets.update(cpu_batch.dataset_ids)
            batch = move_multisource_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            _, metric = _optimize_batch(
                model, batch, optimizer, scaler, config, device
            )
            totals.append(metric["total_loss"])
            if metric["harmonic_loss"] is not None:
                harmonic_values.append(metric["harmonic_loss"])
            if metric["reconstruction_loss"] is not None:
                reconstruction_values.append(
                    metric["reconstruction_loss"]
                )
            for key, value in metric["task_losses"].items():
                task_values[key].append(value)
            availability.update(metric["availability_counts"])
            final_gradient = metric["gradient_coverage"]
        if not totals:
            raise TrainingContractError("training.epoch.empty")
        if scheduler is not None:
            scheduler.step()
        train_metric = {
            "total_loss": sum(totals) / len(totals),
            "harmonic_loss": (
                None
                if not harmonic_values
                else sum(harmonic_values) / len(harmonic_values)
            ),
            "reconstruction_loss": (
                None
                if not reconstruction_values
                else sum(reconstruction_values)
                / len(reconstruction_values)
            ),
            "task_losses": {
                key: sum(values) / len(values)
                for key, values in sorted(task_values.items())
            },
            "availability_counts": dict(sorted(availability.items())),
            "dataset_counts": dict(sorted(datasets.items())),
            "batch_count": len(totals),
            "final_gradient_coverage": final_gradient,
        }
        validation = None
        if (
            (epoch + 1)
            % int(config["experiment"]["validation_interval"])
            == 0
            or epoch + 1 == epochs
        ):
            validation = _validation_epoch(
                model,
                runtime.validation_loader(epoch),
                config=config,
                device=device,
            )
        row = {
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metric,
            "validation": validation,
        }
        _append_jsonl(metrics_path, row)
        if validation is not None and (
            best is None or validation["total_loss"] < best
        ):
            best = float(validation["total_loss"])
            save_training_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler=scheduler,
                scaler=scaler,
                next_epoch=epoch + 1,
                best_validation_loss=best,
                resolved_config=_checkpoint_config(config),
                data_fingerprints=runtime.fingerprints,
            )
        save_training_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=epoch + 1,
            best_validation_loss=best,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        )
        if (
            (epoch + 1)
            % int(config["experiment"]["checkpoint_interval"])
            == 0
        ):
            save_training_checkpoint(
                output_dir / f"epoch-{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler=scheduler,
                scaler=scaler,
                next_epoch=epoch + 1,
                best_validation_loss=best,
                resolved_config=_checkpoint_config(config),
                data_fingerprints=runtime.fingerprints,
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
        "metrics": str(metrics_path),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "device": _device_evidence(device),
        "fingerprints": runtime.fingerprints,
    }
    _write_json(output_dir / "training_report.json", report)
    return report


def run_training(
    config: object,
    *,
    stop_after_epoch: int | None = None,
) -> dict[str, object]:
    """Run one accepted Phase 6C experiment.

    ``stop_after_epoch`` is a bounded-test hook that only stops immediately
    after a checkpointed epoch boundary; it does not implement mid-epoch
    resume.
    """

    plain = _plain_config(config)
    if plain["experiment"]["name"] == "one_batch":
        if stop_after_epoch is not None:
            raise TrainingContractError(
                "training.one_batch.stop_after_epoch_invalid"
            )
        return _run_one_batch(plain)
    return _run_epochs(plain, stop_after_epoch=stop_after_epoch)


__all__ = [
    "TRAINING_CHECKPOINT_VERSION",
    "TrainingContractError",
    "run_training",
]
