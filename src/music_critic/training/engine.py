"""Deterministic one-batch and epoch-boundary Phase 6C runners."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from itertools import islice
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from music_critic.cuda_memory import (
    CudaMemoryStatisticsLifecycleEvidence,
    initialize_cuda_memory_statistics,
)
from music_critic.device import (
    CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION,
    RuntimeDeviceError,
    resolve_cuda_device_index,
    resolve_runtime_device,
)
from music_critic.models import ACTIVE_TASK_IDS
from music_critic.tasks import MultiSourceBatch
from music_critic.training.checkpoint import (
    TRAINING_CHECKPOINT_VERSION,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
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


RUN_MANIFEST_VERSION = "1.0.0"
EPOCH_PERFORMANCE_VERSION = "1.0.0"
_MANAGED_FILENAMES = (
    "resolved_config.json",
    "fingerprints.json",
    "mixture_statistics.json",
    "run_manifest.json",
    "one_batch_report.json",
    "one_batch.pt",
    "metrics.jsonl",
    "epoch_performance.jsonl",
    "training_report.json",
    "last.pt",
    "best.pt",
)
_INTERVAL_CHECKPOINT_PATTERN = re.compile(r"epoch-[0-9]{4,}\.pt")
_EPOCH_METRIC_PATTERN = re.compile(r"epoch-[0-9]{4,}\.json")


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
    """Exclude operational lifecycle flags from compatibility binding."""

    result = copy.deepcopy(config)
    result["experiment"]["resume_from"] = ""
    result["experiment"]["overwrite_output"] = False
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


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _managed_artifacts(output: Path) -> tuple[Path, ...]:
    if not output.exists():
        return ()
    result = [
        output / name
        for name in _MANAGED_FILENAMES
        if (output / name).exists()
    ]
    result.extend(
        path
        for path in output.glob("epoch-*.pt")
        if _INTERVAL_CHECKPOINT_PATTERN.fullmatch(path.name)
    )
    metric_directory = _metric_directory(output)
    if metric_directory.is_dir():
        pending = metric_directory / "pending.json"
        if pending.exists():
            result.append(pending)
        result.extend(
            path
            for path in metric_directory.glob("epoch-*.json")
            if _EPOCH_METRIC_PATTERN.fullmatch(path.name)
        )
    return tuple(sorted(set(result)))


def _preflight_output_lifecycle(config: dict[str, Any]) -> Path:
    output = Path(config["output_dir"]).resolve()
    resume = bool(config["experiment"]["resume_from"])
    overwrite = bool(config["experiment"]["overwrite_output"])
    if resume:
        if overwrite:
            raise TrainingContractError(
                "training.output.resume_overwrite_conflict"
            )
        if not output.is_dir():
            raise TrainingContractError(
                "training.output.resume_directory_missing"
            )
        return output
    managed = _managed_artifacts(output)
    if managed and not overwrite:
        raise TrainingContractError(
            "training.output.managed_artifact_collision"
        )
    return output


def _initialize_fresh_output(
    output: Path,
    *,
    overwrite: bool,
) -> None:
    if output.exists() and not output.is_dir():
        raise TrainingContractError(
            "training.output.path_not_directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for name in _MANAGED_FILENAMES:
        (output / name).unlink(missing_ok=True)
    for path in output.glob("epoch-*.pt"):
        if _INTERVAL_CHECKPOINT_PATTERN.fullmatch(path.name):
            path.unlink()
    metric_directory = _metric_directory(output)
    if metric_directory.is_dir():
        (metric_directory / "pending.json").unlink(
            missing_ok=True
        )
        for path in metric_directory.glob("epoch-*.json"):
            if _EPOCH_METRIC_PATTERN.fullmatch(path.name):
                path.unlink()
        try:
            metric_directory.rmdir()
        except OSError:
            # Unknown user files are deliberately preserved.
            pass


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
    }
    for group, names in accepted.items():
        if config[group]["name"] not in names:
            raise TrainingContractError(
                f"training.config.{group}_invalid"
            )
    for name in (
        "overwrite_output",
        "collect_gradient_evidence",
    ):
        if not isinstance(config["experiment"][name], bool):
            raise TrainingContractError(
                f"training.config.{name}_invalid"
            )
    if (
        config["experiment"]["name"] == "one_batch"
        and config["experiment"]["resume_from"]
    ):
        raise TrainingContractError(
            "training.one_batch.resume_unsupported"
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
        ("validation_seed", config["data"].get("validation_seed", -1), -1),
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
        (
            "optimizer_steps_per_epoch",
            config["experiment"].get("optimizer_steps_per_epoch", 0),
            0,
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
    if config["device"].get("amp_dtype", "float16") not in {
        "float16",
        "bfloat16",
    }:
        raise TrainingContractError("training.config.amp_dtype_invalid")
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
        or weight < 0
        for task_id, weight in task_weights.items()
    ):
        raise TrainingContractError(
            "training.config.task_weights_invalid"
        )
    explicit_requested_tasks = config.get("downstream_task_ids")
    requested_tasks = explicit_requested_tasks or list(ACTIVE_TASK_IDS)
    if (
        not isinstance(requested_tasks, list)
        or not requested_tasks
        or len(requested_tasks) != len(set(requested_tasks))
        or any(task_id not in ACTIVE_TASK_IDS for task_id in requested_tasks)
    ):
        raise TrainingContractError(
            "training.config.downstream_task_ids_invalid"
        )
    if explicit_requested_tasks and task_weights and {
        task_id for task_id, weight in task_weights.items() if weight > 0
    } != set(requested_tasks):
        raise TrainingContractError(
            "training.config.downstream_task_runtime_mismatch"
        )
    transfer = config.get("transfer")
    if not isinstance(transfer, dict) or transfer.get(
        "contract_version"
    ) != "1.1.0" or transfer.get("mode") not in {
        "supervised_scratch",
        "frozen_probe",
        "full_finetune",
    }:
        raise TrainingContractError(
            "training.config.transfer_invalid"
        )
    comparison_bound = bool(transfer.get("comparison_protocol_fingerprint"))
    if comparison_bound:
        schedule_fingerprint = transfer.get("sample_schedule_fingerprint")
        logical_updates = transfer.get("logical_updates")
        if (
            not isinstance(schedule_fingerprint, str)
            or len(schedule_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in schedule_fingerprint
            )
            or isinstance(logical_updates, bool)
            or not isinstance(logical_updates, int)
            or logical_updates <= 0
        ):
            raise TrainingContractError(
                "training.config.phase8b2_schedule_binding_invalid"
            )
    pretrained = transfer["mode"] != "supervised_scratch"
    if pretrained and (
        config["model"]["name"] != "hierarchical"
        or not transfer.get("encoder_export_path")
        or not transfer.get("encoder_export_sha256")
        or not transfer.get("source_ssl_checkpoint_sha256")
        or not transfer.get("comparison_protocol_fingerprint")
    ):
        raise TrainingContractError(
            "training.config.pretrained_transfer_binding_incomplete"
        )
    for name in (
        "encoder_export_sha256",
        "source_ssl_checkpoint_sha256",
        "comparison_protocol_fingerprint",
    ):
        value = transfer.get(name)
        if pretrained and (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise TrainingContractError(
                f"training.config.{name}_invalid"
            )
    if not pretrained and any(
        transfer.get(name)
        for name in (
            "encoder_export_path",
            "encoder_export_sha256",
            "source_ssl_checkpoint_sha256",
        )
    ):
        raise TrainingContractError(
            "training.config.scratch_transfer_source_forbidden"
        )
    for name in (
        "downstream_initialization_seed",
        "downstream_data_order_seed",
    ):
        value = transfer.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TrainingContractError(
                f"training.config.{name}_invalid"
            )


def _resolve_device(config: dict[str, Any]) -> torch.device:
    name = config["device"]["name"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = resolve_runtime_device(name)
    except RuntimeDeviceError as exc:
        if exc.category == "runtime.device.cuda_unavailable":
            raise TrainingContractError(
                "training.device.cuda_unavailable"
            ) from exc
        raise TrainingContractError(
            f"training.device.invalid:{exc}"
        ) from exc
    if config["device"].get("amp", False) and device.type != "cuda":
        raise TrainingContractError("training.device.amp_requires_cuda")
    return device


def _amp_dtype(config: dict[str, Any]) -> torch.dtype:
    name = config["device"].get("amp_dtype", "float16")
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise TrainingContractError("training.config.amp_dtype_invalid")


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
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise TrainingContractError(
            "training.optimizer.trainable_parameter_set_empty"
        )
    return torch.optim.AdamW(
        parameters,
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
        dtype=_amp_dtype(config),
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
    scale_before = float(scaler.get_scale())
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
    scale_after = float(scaler.get_scale())
    skipped = scaler.is_enabled() and scale_after < scale_before
    if not collect_gradient_evidence:
        return output, None, skipped
    return output, {
        "harmonic_loss": _scalar(harmonic),
        "reconstruction_loss": _scalar(reconstruction),
        "total_loss": _scalar(total),
        "gradient_norm_before_clip": float(clipped_norm),
        "gradient_coverage": gradient,
        "task_losses": _task_losses(output),
        "availability_counts": _availability_counts(output),
        "amp_scale_before": scale_before,
        "amp_scale_after": scale_after,
        "optimizer_step_applied": not skipped,
    }, skipped


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
                dtype=_amp_dtype(config),
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


def _device_evidence(
    device: torch.device,
    cuda_memory_lifecycle: (
        CudaMemoryStatisticsLifecycleEvidence | None
    ),
) -> dict[str, object]:
    common = {
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_memory_statistics_lifecycle": (
            None
            if cuda_memory_lifecycle is None
            else cuda_memory_lifecycle.to_dict()
        ),
    }
    if device.type != "cuda":
        return {
            **common,
            "resolved_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    cuda_device_index = resolve_cuda_device_index(device)
    return {
        **common,
        "resolved_device": str(device),
        "cuda_available": True,
        "cuda_runtime_device_index_contract_version": (
            CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION
        ),
        "cuda_logical_device_index": cuda_device_index,
        "cuda_device_name": torch.cuda.get_device_name(cuda_device_index),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
            cuda_device_index
        ),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(
            cuda_device_index
        ),
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
    CudaMemoryStatisticsLifecycleEvidence | None,
]:
    _validate_config(config)
    device = _resolve_device(config)
    transfer = config["transfer"]
    comparison_bound = bool(transfer["comparison_protocol_fingerprint"])
    data_seed = (
        int(transfer["downstream_data_order_seed"])
        if comparison_bound
        else int(config["seed"])
    )
    _set_determinism(data_seed)
    cuda_memory_lifecycle = (
        initialize_cuda_memory_statistics(device)
        if device.type == "cuda"
        else None
    )
    runtime = build_data_runtime(
        OmegaConf.create(config["data"]), seed=data_seed
    )
    schedule_path = transfer.get("actual_sample_schedule_path", "")
    if comparison_bound and schedule_path:
        try:
            artifact = json.loads(
                Path(schedule_path).read_text(encoding="utf-8")
            )
            candidates = [
                row
                for row in artifact["downstream"]
                if row["data_order_seed"] == data_seed
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise TrainingContractError(
                "training.phase8b2.actual_sample_schedule_unreadable"
            ) from exc
        if (
            artifact.get("protocol_fingerprint")
            != transfer["comparison_protocol_fingerprint"]
            or len(candidates) != 1
            or candidates[0].get("sample_schedule_fingerprint")
            != transfer["sample_schedule_fingerprint"]
            or candidates[0].get("logical_updates")
            != transfer["logical_updates"]
        ):
            raise TrainingContractError(
                "training.phase8b2.actual_sample_schedule_binding_mismatch"
            )
    if comparison_bound:
        _set_determinism(int(transfer["downstream_initialization_seed"]))
    model = build_baseline_model(
        OmegaConf.create(config["model"]),
        task_weights=config["objective"]["task_weights"],
    ).to(device)
    encoder_export = None
    if transfer["mode"] != "supervised_scratch":
        export_path = Path(transfer["encoder_export_path"]).resolve()
        if _file_sha256(export_path) != transfer["encoder_export_sha256"]:
            raise TrainingContractError(
                "training.transfer.encoder_export_sha256_mismatch"
            )
        try:
            encoder_export = torch.load(
                export_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise TrainingContractError(
                f"training.transfer.encoder_export_unreadable:{exc}"
            ) from exc
    from music_critic.experiments.phase8b2.contracts import (
        Phase8B2ContractError,
    )
    from music_critic.experiments.phase8b2.transfer import (
        prepare_downstream_model,
    )

    try:
        _, transfer_evidence = prepare_downstream_model(
            model,
            transfer_mode=transfer["mode"],
            encoder_export=encoder_export,
        )
    except Phase8B2ContractError as exc:
        raise TrainingContractError(str(exc)) from exc
    transfer_evidence["comparison_protocol_fingerprint"] = transfer[
        "comparison_protocol_fingerprint"
    ] or None
    transfer_evidence["source_ssl_checkpoint_sha256"] = transfer[
        "source_ssl_checkpoint_sha256"
    ] or None
    transfer_evidence["encoder_export_sha256"] = transfer[
        "encoder_export_sha256"
    ] or None
    config["phase8b2_transfer_runtime"] = transfer_evidence
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
    return (
        output,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        cuda_memory_lifecycle,
    )


def _artifact_payloads(
    output: Path,
    config: dict[str, Any],
    runtime: DataRuntime,
    model: BaselineModel,
) -> dict[str, object]:
    del output
    resolved = copy.deepcopy(config)
    fingerprints = {
        "data": runtime.fingerprints,
        "model_contract_fingerprint": model_contract_fingerprint(
            model
        ),
        "checkpoint_binding": training_checkpoint_metadata(
            model,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        ),
    }
    mixture = runtime.mixture_statistics
    return {
        "resolved_config.json": resolved,
        "fingerprints.json": fingerprints,
        "mixture_statistics.json": mixture,
        "run_manifest.json": {
            "run_manifest_version": RUN_MANIFEST_VERSION,
            "checkpoint_binding": fingerprints[
                "checkpoint_binding"
            ],
            "artifact_fingerprints": {
                "resolved_config.json": _json_fingerprint(resolved),
                "fingerprints.json": _json_fingerprint(
                    fingerprints
                ),
                "mixture_statistics.json": _json_fingerprint(
                    mixture
                ),
            },
        },
    }


def _write_initial_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: DataRuntime,
    model: BaselineModel,
) -> None:
    payloads = _artifact_payloads(output, config, runtime, model)
    for name in (
        "resolved_config.json",
        "fingerprints.json",
        "mixture_statistics.json",
        "run_manifest.json",
    ):
        _write_json_atomic(output / name, payloads[name])


def _read_evidence_artifact(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingContractError(
            f"training.output.evidence_unreadable:{path.name}"
        ) from exc


def _validate_resume_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: DataRuntime,
    model: BaselineModel,
) -> None:
    manifest_path = output / "run_manifest.json"
    manifest = _read_evidence_artifact(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != {
        "run_manifest_version",
        "checkpoint_binding",
        "artifact_fingerprints",
    }:
        raise TrainingContractError(
            "training.output.run_manifest_invalid"
        )
    if manifest["run_manifest_version"] != RUN_MANIFEST_VERSION:
        raise TrainingContractError(
            "training.output.run_manifest_version_mismatch"
        )
    expected_binding = training_checkpoint_metadata(
        model,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    if _json_fingerprint(
        manifest["checkpoint_binding"]
    ) != _json_fingerprint(expected_binding):
        raise TrainingContractError(
            "training.output.run_manifest_binding_mismatch"
        )
    artifact_fingerprints = manifest["artifact_fingerprints"]
    expected_names = {
        "resolved_config.json",
        "fingerprints.json",
        "mixture_statistics.json",
    }
    if (
        not isinstance(artifact_fingerprints, dict)
        or set(artifact_fingerprints) != expected_names
    ):
        raise TrainingContractError(
            "training.output.run_manifest_artifacts_invalid"
        )
    for name in sorted(expected_names):
        actual = _read_evidence_artifact(output / name)
        if _json_fingerprint(actual) != artifact_fingerprints[name]:
            raise TrainingContractError(
                f"training.output.evidence_fingerprint_mismatch:{name}"
            )


def _run_one_batch(config: dict[str, Any]) -> dict[str, object]:
    from music_critic.experiments.phase8b2.contracts import (
        Phase8B2ContractError,
    )
    from music_critic.experiments.phase8b2.transfer import (
        prepare_downstream_model,
        verify_frozen_encoder,
    )

    started_at = time.perf_counter()
    (
        output_dir,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        cuda_memory_lifecycle,
    ) = _prepare(config)
    _initialize_fresh_output(
        output_dir,
        overwrite=config["experiment"]["overwrite_output"],
    )
    _write_initial_artifacts(
        output_dir, config, runtime, model
    )
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
        learning_rate_used = optimizer.param_groups[0]["lr"]
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
        metric["learning_rate_used"] = learning_rate_used
        final_gradient = metric["gradient_coverage"]
        if scheduler is not None:
            scheduler.step()
        metric["next_learning_rate"] = optimizer.param_groups[0][
            "lr"
        ]
        curve.append(metric)
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
    reload_export = None
    if config["transfer"]["mode"] != "supervised_scratch":
        reload_export = torch.load(
            Path(config["transfer"]["encoder_export_path"]).resolve(),
            map_location="cpu",
            weights_only=True,
        )
    try:
        prepare_downstream_model(
            reloaded,
            transfer_mode=config["transfer"]["mode"],
            encoder_export=reload_export,
        )
    except Phase8B2ContractError as exc:
        raise TrainingContractError(str(exc)) from exc
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
        maximum_next_epoch=0,
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
        "duration_seconds": time.perf_counter() - started_at,
        "device": _device_evidence(device, cuda_memory_lifecycle),
        "fingerprints": runtime.fingerprints,
        "phase8b2_transfer": config["phase8b2_transfer_runtime"],
        "frozen_encoder_final": (
            verify_frozen_encoder(
                model, config["phase8b2_transfer_runtime"]
            )
            if config["transfer"]["mode"] == "frozen_probe"
            else None
        ),
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


def _read_performance_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "epoch_performance.jsonl"
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingContractError(
                "training.performance.unreadable:"
                f"{line_number}:{exc}"
            ) from exc
        if (
            not isinstance(row, dict)
            or row.get("next_epoch") != line_number
            or row.get("epoch_performance_version")
            != EPOCH_PERFORMANCE_VERSION
        ):
            raise TrainingContractError(
                "training.performance.non_contiguous"
            )
        rows.append(row)
    return rows


def _write_performance_rows(
    output_dir: Path, rows: list[dict[str, Any]]
) -> None:
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
    _write_text_atomic(output_dir / "epoch_performance.jsonl", payload)


def _sync_performance_rows(
    output_dir: Path, *, committed_epochs: int
) -> None:
    """Align the non-binding timing sidecar with checkpoint authority."""

    rows = _read_performance_rows(output_dir)[:committed_epochs]
    while len(rows) < committed_epochs:
        next_epoch = len(rows) + 1
        rows.append(
            {
                "epoch_performance_version": EPOCH_PERFORMANCE_VERSION,
                "epoch": next_epoch - 1,
                "next_epoch": next_epoch,
                "train": None,
                "validation": None,
                "unavailable": {
                    "category": "recovered_epoch_without_timing",
                    "reason": (
                        "the deterministic epoch was checkpoint-committed "
                        "before its non-binding timing sidecar was written"
                    ),
                },
                "checkpoint_binding_participation": False,
            }
        )
    _write_performance_rows(output_dir, rows)


def _append_performance_row(
    output_dir: Path, row: dict[str, Any]
) -> None:
    rows = _read_performance_rows(output_dir)
    if row.get("next_epoch") != len(rows) + 1:
        raise TrainingContractError(
            "training.performance.append_epoch_mismatch"
        )
    rows.append(row)
    _write_performance_rows(output_dir, rows)


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
    from music_critic.experiments.phase8b2.transfer import (
        verify_frozen_encoder,
    )

    started_at = time.perf_counter()
    resume = config["experiment"]["resume_from"]
    entry_rng = capture_rng_state() if resume else None
    try:
        (
            output_dir,
            device,
            runtime,
            model,
            optimizer,
            scheduler,
            scaler,
            cuda_memory_lifecycle,
        ) = _prepare(config)
    except Exception:
        if entry_rng is not None:
            restore_rng_state(entry_rng)
        raise
    start_epoch = 0
    best: float | None = None
    if resume:
        try:
            _validate_resume_artifacts(
                output_dir, config, runtime, model
            )
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
                maximum_next_epoch=int(
                    config["experiment"]["epochs"]
                ),
                resolved_config=_checkpoint_config(config),
                data_fingerprints=runtime.fingerprints,
            )
            _recover_metric_journal(
                output_dir,
                committed_metric_rows=committed_metric_rows,
            )
        except Exception:
            if entry_rng is not None:
                restore_rng_state(entry_rng)
            raise
    else:
        _initialize_fresh_output(
            output_dir,
            overwrite=config["experiment"]["overwrite_output"],
        )
        _write_initial_artifacts(
            output_dir, config, runtime, model
        )
        _reset_metric_journal(output_dir)
        _write_performance_rows(output_dir, [])
    if resume:
        _sync_performance_rows(
            output_dir, committed_epochs=start_epoch
        )
    epochs = int(config["experiment"]["epochs"])
    comparison_bound = bool(
        config["transfer"]["comparison_protocol_fingerprint"]
    )
    logical_update_budget = (
        int(config["experiment"]["steps"])
        if comparison_bound
        else None
    )
    steps_per_epoch = int(
        config["experiment"].get("optimizer_steps_per_epoch", 0)
    ) or int(config["experiment"]["steps"])
    completed = start_epoch
    for epoch in range(start_epoch, epochs):
        committed_before_epoch = _committed_metric_envelopes(output_dir)
        attempted_before_epoch = sum(
            int(row["row"]["train"]["batch_count"])
            for row in committed_before_epoch
        )
        remaining_updates = (
            None
            if logical_update_budget is None
            else logical_update_budget - attempted_before_epoch
        )
        if remaining_updates is not None and remaining_updates <= 0:
            break
        learning_rate_used = optimizer.param_groups[0]["lr"]
        model.train()
        train_accumulator = EpochMetricAccumulator(
            harmonic_weight=config["objective"]["harmonic_weight"],
            reconstruction_weight=config["objective"][
                "reconstruction_weight"
            ],
            task_weights=config["objective"]["task_weights"],
        )
        downstream_batch_identities: list[list[list[str]]] = []
        train_started_at = time.perf_counter()
        epoch_batches = runtime.train_loader(epoch)
        if remaining_updates is not None:
            epoch_batches = islice(
                epoch_batches,
                min(steps_per_epoch, remaining_updates),
            )
        for cpu_batch in epoch_batches:
            downstream_batch_identities.append(
                [
                    [dataset_id, piece_id]
                    for dataset_id, piece_id in zip(
                        cpu_batch.dataset_ids,
                        cpu_batch.piece_ids,
                        strict=True,
                    )
                ]
            )
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
        train_wall_seconds = time.perf_counter() - train_started_at
        if train_metric["batch_count"] == 0:
            raise TrainingContractError("training.epoch.empty")
        if config["transfer"]["comparison_protocol_fingerprint"] and int(
            train_metric["skipped_batch_count"]
        ):
            raise TrainingContractError(
                "training.phase8b2.scientific_cell_optimizer_step_invalid"
            )
        if scheduler is not None:
            scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        validation = None
        validation_wall_seconds = None
        if (
            (epoch + 1)
            % int(config["experiment"]["validation_interval"])
            == 0
            or (
                logical_update_budget is not None
                and attempted_before_epoch + int(train_metric["batch_count"])
                == logical_update_budget
            )
            or (logical_update_budget is None and epoch + 1 == epochs)
        ):
            validation_started_at = time.perf_counter()
            validation = _validation_epoch(
                model,
                runtime.validation_loader(),
                config=config,
                device=device,
                membership_evidence=asdict(
                    runtime.validation_membership
                ),
            )
            validation_wall_seconds = (
                time.perf_counter() - validation_started_at
            )
        row = {
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": next_learning_rate,
            "train": train_metric,
            "validation": validation,
        }
        if config["transfer"]["comparison_protocol_fingerprint"]:
            row["phase8b2_downstream_sample_identities"] = [
                identity
                for batch_identities in downstream_batch_identities
                for identity in batch_identities
            ]
            row["phase8b2_downstream_schedule_fingerprint"] = (
                _json_fingerprint(
                    {
                        "contract_version": "1.2.0",
                        "epoch": epoch,
                        "batch_identities": downstream_batch_identities,
                    }
                )
            )
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
        train_sample_count = sum(
            int(value)
            for value in train_metric["dataset_counts"].values()
        )
        validation_sample_count = (
            0
            if validation is None
            else sum(
                int(value)
                for value in validation["dataset_counts"].values()
            )
        )
        _append_performance_row(
            output_dir,
            {
                "epoch_performance_version": EPOCH_PERFORMANCE_VERSION,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "train": {
                    "wall_seconds": train_wall_seconds,
                    "samples_per_second": (
                        train_sample_count / train_wall_seconds
                    ),
                    "batches_per_second": (
                        int(train_metric["batch_count"])
                        / train_wall_seconds
                    ),
                },
                "validation": (
                    None
                    if validation is None
                    or validation_wall_seconds is None
                    else {
                        "wall_seconds": validation_wall_seconds,
                        "samples_per_second": (
                            validation_sample_count
                            / validation_wall_seconds
                        ),
                        "batches_per_second": (
                            int(validation["batch_count"])
                            / validation_wall_seconds
                        ),
                    }
                ),
                "unavailable": None,
                "checkpoint_binding_participation": False,
            },
        )
        completed = epoch + 1
        if stop_after_epoch is not None and completed >= stop_after_epoch:
            break
    committed_rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    observed_downstream_identities = [
        identity
        for row in committed_rows
        for identity in row.get(
            "phase8b2_downstream_sample_identities", []
        )
    ]
    from music_critic.experiments.phase8b2.contracts import (
        fingerprint as phase8b2_fingerprint,
    )
    from music_critic.experiments.phase8b2.schedule import (
        SCHEDULE_CONTRACT_VERSION,
    )

    observed_schedule_fingerprint = phase8b2_fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_downstream_sample_schedule",
            "identities": observed_downstream_identities,
        }
    )
    expected_schedule_fingerprint = config["transfer"].get(
        "sample_schedule_fingerprint"
    ) or None
    attempted_updates = sum(
        int(row["train"]["batch_count"]) for row in committed_rows
    )
    skipped_updates = sum(
        int(row["train"]["skipped_batch_count"])
        for row in committed_rows
    )
    applied_updates = attempted_updates - skipped_updates
    budget_complete = (
        True
        if logical_update_budget is None
        else attempted_updates == logical_update_budget
    )
    schedule_verified = (
        not config["transfer"]["comparison_protocol_fingerprint"]
        or observed_schedule_fingerprint == expected_schedule_fingerprint
    )
    if (
        config["transfer"]["comparison_protocol_fingerprint"]
        and budget_complete
        and not schedule_verified
    ):
        raise TrainingContractError(
            "training.phase8b2.actual_sample_schedule_mismatch"
        )
    report = {
        "evidence_kind": "bounded_supervised_training_plumbing",
        "resume_boundary": "epoch_only",
        "mid_epoch_resume_supported": False,
        "start_epoch": start_epoch,
        "completed_epochs": completed,
        "configured_epochs": epochs,
        "configured_logical_updates": logical_update_budget,
        "logical_update_budget_complete": budget_complete,
        "optimizer_step_attempt_count": attempted_updates,
        "optimizer_step_applied_count": applied_updates,
        "optimizer_step_skipped_count": skipped_updates,
        "best_validation_loss": best,
        "metrics": str(output_dir / "metrics.jsonl"),
        "epoch_performance": str(
            output_dir / "epoch_performance.jsonl"
        ),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "duration_seconds": time.perf_counter() - started_at,
        "detailed_profiler_enabled": False,
        "device": _device_evidence(device, cuda_memory_lifecycle),
        "fingerprints": runtime.fingerprints,
        "validation_membership": asdict(
            runtime.validation_membership
        ),
        "objective": config["objective"],
        "phase8b2_transfer": config["phase8b2_transfer_runtime"],
        "observed_downstream_schedule_fingerprint": (
            observed_schedule_fingerprint
            if config["transfer"]["comparison_protocol_fingerprint"]
            else None
        ),
        "expected_downstream_schedule_fingerprint": (
            expected_schedule_fingerprint
        ),
        "actual_sample_schedule_verified": schedule_verified,
        "observed_sample_identities": observed_downstream_identities,
        "frozen_encoder_final": (
            verify_frozen_encoder(
                model, config["phase8b2_transfer_runtime"]
            )
            if config["transfer"]["mode"] == "frozen_probe"
            else None
        ),
    }
    _write_json_atomic(output_dir / "training_report.json", report)
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
    _validate_config(plain)
    _preflight_output_lifecycle(plain)
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
