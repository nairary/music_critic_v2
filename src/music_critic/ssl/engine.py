"""Deterministic bounded training harness for the Phase 7A SSL baseline."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf

from music_critic.graph import RAW_FEATURE_REGISTRY
from music_critic.models import (
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.checkpoint import (
    SSL_METRIC_ROW_VERSION,
    SSLResumeState,
    load_ssl_checkpoint,
    save_ssl_checkpoint,
    ssl_checkpoint_metadata,
)
from music_critic.ssl.data import (
    SSLBatch,
    SSLDataRuntime,
    build_ssl_data_runtime,
    move_ssl_batch,
)
from music_critic.ssl.model import (
    MaskedGraphSSLModel,
    SSLForwardOutput,
    build_ssl_model,
)
from music_critic.ssl.transfer import (
    export_pretrained_encoder_state,
    load_pretrained_encoder_state,
)
from music_critic.training.checkpoint import (
    capture_rng_state,
    restore_rng_state,
)


SSL_RUN_MANIFEST_VERSION = "1.0.0"
SSL_TRAINING_REPORT_VERSION = "1.0.0"
SSL_PERFORMANCE_ROW_VERSION = "1.0.0"


class SSLTrainingError(ValueError):
    """Raised for a deterministic Phase 7A training-contract violation."""


def _plain_config(config: object) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, dict):
        value = copy.deepcopy(config)
    else:
        raise SSLTrainingError("ssl.training.config_type_invalid")
    if not isinstance(value, dict):
        raise SSLTrainingError("ssl.training.config_root_invalid")
    value.pop("defaults", None)
    if value["optimizer"]["learning_rate"] is None:
        value["optimizer"]["learning_rate"] = value["experiment"][
            "default_learning_rate"
        ]
    return value


def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["experiment"]["resume_from"] = ""
    result["experiment"]["overwrite_output"] = False
    return result


def _training_scope_evidence(
    *,
    data_source_kind: str,
    run_scope: str,
    optimization_step_count: int,
) -> dict[str, object]:
    if (
        not isinstance(data_source_kind, str)
        or not data_source_kind
        or run_scope not in {"one_batch_plumbing", "epoch_pretraining"}
        or isinstance(optimization_step_count, bool)
        or not isinstance(optimization_step_count, int)
        or optimization_step_count < 0
    ):
        raise SSLTrainingError("ssl.training.scope_evidence_invalid")
    production_cache_data_used = data_source_kind != "bounded"
    production_training = (
        production_cache_data_used and optimization_step_count > 0
    )
    full_corpus: bool | None
    full_corpus_unavailable_reason: str | None
    if not production_cache_data_used or run_scope == "one_batch_plumbing":
        full_corpus = False
        full_corpus_unavailable_reason = None
    else:
        full_corpus = None
        full_corpus_unavailable_reason = (
            "full_corpus_identity_coverage_not_tracked"
        )
    return {
        "data_source_kind": data_source_kind,
        "production_cache_data_used": production_cache_data_used,
        "run_scope": run_scope,
        "optimization_step_count": optimization_step_count,
        "production_ssl_training_performed": production_training,
        "full_corpus_ssl_training_performed": full_corpus,
        "full_corpus_ssl_training_unavailable_reason": (
            full_corpus_unavailable_reason
        ),
    }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def _write_jsonl_atomic(path: Path, rows: Iterable[object]) -> None:
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _validate_config(config: dict[str, Any]) -> None:
    if config["model"]["name"] != "hierarchical":
        raise SSLTrainingError("ssl.training.hierarchical_model_required")
    if config["experiment"]["name"] not in {"one_batch", "pretrain"}:
        raise SSLTrainingError("ssl.training.experiment_invalid")
    for name in ("steps", "epochs", "checkpoint_interval", "validation_interval"):
        value = config["experiment"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SSLTrainingError(
                f"ssl.training.experiment_{name}_invalid"
            )
    seed = config["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (1 << 63) - 1
    ):
        raise SSLTrainingError("ssl.training.seed_invalid")
    learning_rate = config["optimizer"]["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise SSLTrainingError("ssl.training.learning_rate_invalid")
    gradient_clip = config["optimizer"]["gradient_clip_norm"]
    if (
        isinstance(gradient_clip, bool)
        or not isinstance(gradient_clip, (int, float))
        or not math.isfinite(float(gradient_clip))
        or gradient_clip <= 0
    ):
        raise SSLTrainingError("ssl.training.gradient_clip_invalid")
    if config["ssl"]["epsilon"] != 1e-8:
        raise SSLTrainingError("ssl.training.cosine_epsilon_incompatible")


def _resolve_device(config: dict[str, Any]) -> torch.device:
    name = config["device"]["name"]
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SSLTrainingError("ssl.training.cuda_unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise SSLTrainingError(f"ssl.training.device_unknown:{name}")


def _configure_cublas_determinism() -> None:
    if torch.cuda.is_available():
        cublas_workspace = os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        if cublas_workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif cublas_workspace not in {":4096:8", ":16:8"}:
            raise SSLTrainingError(
                "ssl.training.cublas_workspace_config_invalid"
            )


def _set_determinism(seed: int) -> None:
    _configure_cublas_determinism()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _optimizer(
    model: MaskedGraphSSLModel,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    if config["optimizer"]["name"] != "adamw":
        raise SSLTrainingError("ssl.training.optimizer_unknown")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> Any:
    name = config["scheduler"]["name"]
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["experiment"]["epochs"]),
            eta_min=float(config["scheduler"]["minimum_learning_rate"]),
        )
    raise SSLTrainingError(f"ssl.training.scheduler_unknown:{name}")


def _scalar(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().cpu().item())


def _optional_tensor_equal(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return torch.equal(left, right)


def _diagnostics(value: object) -> dict[str, object]:
    return {
        "row_count": value.row_count,
        "embedding_dim": value.embedding_dim,
        "target_embedding_variance": _scalar(
            value.target_embedding_variance
        ),
        "prediction_embedding_variance": _scalar(
            value.prediction_embedding_variance
        ),
        "target_mean_norm": _scalar(value.target_mean_norm),
        "prediction_mean_norm": _scalar(value.prediction_mean_norm),
        "target_zero_norm_count": value.target_zero_norm_count,
        "prediction_zero_norm_count": value.prediction_zero_norm_count,
        "target_mean_off_diagonal_cosine": _scalar(
            value.target_mean_off_diagonal_cosine
        ),
        "prediction_mean_off_diagonal_cosine": _scalar(
            value.prediction_mean_off_diagonal_cosine
        ),
        "unavailable_reason": value.unavailable_reason,
        "pairwise_unavailable_reason": (
            value.pairwise_unavailable_reason
        ),
        "pairwise_policy": value.pairwise_policy,
    }


def _gradient_evidence(model: MaskedGraphSSLModel) -> dict[str, object]:
    groups = {
        "online_local_encoder": "encoder.local_baseline.encoder.",
        "hierarchy_pooling": "encoder.context_encoder.pooling.",
        "transformer": "encoder.context_encoder.transformer.",
        "fusion": "encoder.context_encoder.fusion.",
        "decoder": "decoder.",
        "bar_projector": "bar_projector_predictor.projector.",
        "bar_predictor": "bar_projector_predictor.predictor.",
        "song_projector": "song_projector_predictor.projector.",
        "song_predictor": "song_projector_predictor.predictor.",
    }
    result: dict[str, object] = {}
    named = tuple(model.named_parameters())
    for group, prefix in groups.items():
        parameters = tuple(
            parameter
            for name, parameter in named
            if name.startswith(prefix)
        )
        with_gradient = tuple(
            parameter for parameter in parameters if parameter.grad is not None
        )
        result[group] = {
            "parameter_count": len(parameters),
            "with_gradient_count": len(with_gradient),
            "finite_gradient_count": sum(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in with_gradient
            ),
            "nonzero_gradient_count": sum(
                bool(parameter.grad.abs().sum() > 0)
                for parameter in with_gradient
            ),
        }
    target_only = tuple(
        parameter
        for name, parameter in named
        if ".task_heads." in name or ".reconstruction_heads." in name
    )
    result["unused_supervised_heads"] = {
        "parameter_count": len(target_only),
        "with_gradient_count": sum(
            parameter.grad is not None for parameter in target_only
        ),
    }
    result["feature_mask_token"] = {
        "with_gradient": model.feature_mask_token.grad is not None,
        "finite": bool(
            model.feature_mask_token.grad is not None
            and torch.isfinite(model.feature_mask_token.grad).all()
        ),
    }
    return result


def _batch_metric(
    output: SSLForwardOutput,
    batch: SSLBatch,
) -> dict[str, object]:
    primary_count = sum(
        len(plan.selected_local_node_indices)
        for plan in output.mask_plans
    )
    maskable_count = sum(
        plan.maskable_node_count for plan in output.mask_plans
    )
    collateral_note_count = sum(
        len(mask.local_node_indices)
        for plan in output.mask_plans
        for mask in plan.collateral_feature_masks
        if mask.node_type == "note"
    )
    collateral_track_count = sum(
        len(mask.local_node_indices)
        for plan in output.mask_plans
        for mask in plan.collateral_feature_masks
        if mask.node_type == "track"
    )
    return {
        "total_ssl_loss": _scalar(output.objective.total_loss),
        "total_unavailable_reason": output.objective.unavailable_reason,
        "note_reconstruction": {
            "numerator": _scalar(output.note_loss.numerator),
            "denominator": output.note_loss.denominator,
            "mean": _scalar(output.note_loss.mean),
            "unavailable_reason": output.note_loss.unavailable_reason,
            "zero_norm_count": output.note_loss.zero_norm_count,
        },
        "decoder_view_losses": [
            {
                "decoder_view_index": view.decoder_view_index,
                "numerator": _scalar(view.loss.numerator),
                "denominator": view.loss.denominator,
                "mean": _scalar(view.loss.mean),
                "unavailable_reason": view.loss.unavailable_reason,
                "stable_seeds": [
                    plan.stable_seed
                    for plan in output.decoder_remask_plans[
                        view.decoder_view_index
                    ]
                ],
                "plan_fingerprints": [
                    plan.fingerprint
                    for plan in output.decoder_remask_plans[
                        view.decoder_view_index
                    ]
                ],
            }
            for view in output.note_loss.view_losses
        ],
        "bar_latent": {
            "numerator": _scalar(output.bar_latent.loss.numerator),
            "denominator": output.bar_latent.loss.denominator,
            "mean": _scalar(output.bar_latent.loss.mean),
            "unavailable_reason": output.bar_latent.loss.unavailable_reason,
            "zero_norm_count": output.bar_latent.loss.zero_norm_count,
        },
        "song_latent": {
            "numerator": _scalar(output.song_latent.loss.numerator),
            "denominator": output.song_latent.loss.denominator,
            "mean": _scalar(output.song_latent.loss.mean),
            "unavailable_reason": output.song_latent.loss.unavailable_reason,
            "zero_norm_count": output.song_latent.loss.zero_norm_count,
        },
        "masking": {
            "sample_count": batch.sample_count,
            "maskable_note_count": maskable_count,
            "primary_masked_count": primary_count,
            "collateral_note_count": collateral_note_count,
            "collateral_track_count": collateral_track_count,
            "collateral_masked_count": (
                collateral_note_count + collateral_track_count
            ),
            "requested_mask_rate": output.mask_plans[
                0
            ].requested_mask_rate,
            "realized_mask_rate": (
                primary_count / maskable_count if maskable_count else 0.0
            ),
            "plan_fingerprints": [
                plan.fingerprint for plan in output.mask_plans
            ],
            "overlay_fingerprint": output.feature_overlay.fingerprint,
        },
        "anti_collapse": {
            "note": _diagnostics(output.note_diagnostics),
            "bar": _diagnostics(output.bar_latent.diagnostics),
            "song": _diagnostics(output.song_latent.diagnostics),
        },
        "sample_count": batch.sample_count,
        "node_count": batch.node_count,
        "edge_count": batch.edge_count,
    }


def _optimize_batch(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    *,
    epoch: int,
    collect_gradient_evidence: bool,
) -> tuple[
    SSLForwardOutput,
    dict[str, object] | None,
    dict[str, float],
]:
    optimizer.zero_grad(set_to_none=True)
    forward_started = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        enabled=bool(config["device"]["amp"]),
    ):
        output = model(
            batch,
            global_seed=int(config["seed"]),
            epoch=epoch,
        )
    forward_seconds = time.perf_counter() - forward_started
    loss = output.objective.total_loss
    if loss is None:
        return output, None, {
            "forward_seconds": forward_seconds,
            "backward_seconds": 0.0,
        }
    if not bool(torch.isfinite(loss)):
        raise SSLTrainingError("ssl.training.nonfinite_total_loss")
    backward_started = time.perf_counter()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["optimizer"]["gradient_clip_norm"]),
    )
    gradient = (
        _gradient_evidence(model) if collect_gradient_evidence else None
    )
    scaler.step(optimizer)
    scaler.update()
    backward_seconds = time.perf_counter() - backward_started
    return output, gradient, {
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
    }


class _Accumulator:
    def __init__(
        self,
        decoder_views: int,
        *,
        note_weight: float,
        bar_weight: float,
        song_weight: float,
    ) -> None:
        self.batch_count = 0
        self.available_batch_count = 0
        self.skipped_batch_count = 0
        self.sample_count = 0
        self.node_count = 0
        self.edge_count = 0
        self.primary_count = 0
        self.collateral_note_count = 0
        self.collateral_track_count = 0
        self.maskable_count = 0
        self.note_sum = 0.0
        self.note_count = 0
        self.note_zero_norm_count = 0
        self.bar_sum = 0.0
        self.bar_count = 0
        self.bar_zero_norm_count = 0
        self.song_sum = 0.0
        self.song_count = 0
        self.song_zero_norm_count = 0
        self.view_sum = [0.0] * decoder_views
        self.view_count = [0] * decoder_views
        self.view_zero_norm_count = [0] * decoder_views
        self.diagnostic_rows: list[dict[str, object]] = []
        self.last_plan_fingerprints: list[str] = []
        self.requested_mask_rate: float | None = None
        self.weights = {
            "note_reconstruction": float(note_weight),
            "bar_latent": float(bar_weight),
            "song_latent": float(song_weight),
        }

    def add(self, output: SSLForwardOutput, batch: SSLBatch) -> None:
        metric = _batch_metric(output, batch)
        self.batch_count += 1
        self.sample_count += batch.sample_count
        self.node_count += batch.node_count
        self.edge_count += batch.edge_count
        masking = metric["masking"]
        assert isinstance(masking, dict)
        self.primary_count += int(masking["primary_masked_count"])
        self.collateral_note_count += int(
            masking["collateral_note_count"]
        )
        self.collateral_track_count += int(
            masking["collateral_track_count"]
        )
        self.maskable_count += int(masking["maskable_note_count"])
        self.last_plan_fingerprints = list(masking["plan_fingerprints"])
        requested_rate = float(masking["requested_mask_rate"])
        if (
            self.requested_mask_rate is not None
            and self.requested_mask_rate != requested_rate
        ):
            raise SSLTrainingError(
                "ssl.training.requested_mask_rate_changed_within_epoch"
            )
        self.requested_mask_rate = requested_rate
        if output.objective.total_loss is None:
            self.skipped_batch_count += 1
        else:
            self.available_batch_count += 1
        self.note_sum += float(output.note_loss.numerator.detach())
        self.note_count += output.note_loss.denominator
        self.note_zero_norm_count += output.note_loss.zero_norm_count
        self.bar_sum += float(output.bar_latent.loss.numerator.detach())
        self.bar_count += output.bar_latent.loss.denominator
        self.bar_zero_norm_count += (
            output.bar_latent.loss.zero_norm_count
        )
        self.song_sum += float(output.song_latent.loss.numerator.detach())
        self.song_count += output.song_latent.loss.denominator
        self.song_zero_norm_count += (
            output.song_latent.loss.zero_norm_count
        )
        for index, view in enumerate(output.note_loss.view_losses):
            self.view_sum[index] += float(view.loss.numerator.detach())
            self.view_count[index] += view.loss.denominator
            self.view_zero_norm_count[index] += (
                view.loss.zero_norm_count
            )
        # Retain one bounded detached row, never predictions or per-batch tensors.
        self.diagnostic_rows[:] = [metric["anti_collapse"]]

    @staticmethod
    def _mean(numerator: float, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    def finalize(self) -> dict[str, object]:
        component_values = {
            "note_reconstruction": self._mean(
                self.note_sum, self.note_count
            ),
            "bar_latent": self._mean(
                self.bar_sum, self.bar_count
            ),
            "song_latent": self._mean(
                self.song_sum, self.song_count
            ),
        }
        unavailable_components = [
            name
            for name, value in component_values.items()
            if self.weights[name] > 0 and value is None
        ]
        total_ssl_loss = (
            None
            if unavailable_components
            else sum(
                self.weights[name] * value
                for name, value in component_values.items()
                if self.weights[name] > 0 and value is not None
            )
        )
        return {
            "batch_count": self.batch_count,
            "available_batch_count": self.available_batch_count,
            "skipped_or_unavailable_batch_count": self.skipped_batch_count,
            "sample_count": self.sample_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "total_ssl_loss": total_ssl_loss,
            "total_unavailable_reason": (
                "required_component_unavailable"
                if unavailable_components
                else None
            ),
            "unavailable_components": unavailable_components,
            "objective_weights": dict(self.weights),
            "note_reconstruction": {
                "numerator": self.note_sum,
                "denominator": self.note_count,
                "mean": component_values["note_reconstruction"],
                "zero_norm_count": self.note_zero_norm_count,
                "unavailable_reason": (
                    "no_eligible_rows" if self.note_count == 0 else None
                ),
            },
            "bar_latent": {
                "numerator": self.bar_sum,
                "denominator": self.bar_count,
                "mean": component_values["bar_latent"],
                "zero_norm_count": self.bar_zero_norm_count,
                "unavailable_reason": (
                    "no_eligible_rows" if self.bar_count == 0 else None
                ),
            },
            "song_latent": {
                "numerator": self.song_sum,
                "denominator": self.song_count,
                "mean": component_values["song_latent"],
                "zero_norm_count": self.song_zero_norm_count,
                "unavailable_reason": (
                    "no_eligible_rows" if self.song_count == 0 else None
                ),
            },
            "decoder_view_losses": [
                {
                    "decoder_view_index": index,
                    "numerator": numerator,
                    "denominator": self.view_count[index],
                    "mean": self._mean(
                        numerator, self.view_count[index]
                    ),
                    "zero_norm_count": (
                        self.view_zero_norm_count[index]
                    ),
                    "unavailable_reason": (
                        "no_eligible_rows"
                        if self.view_count[index] == 0
                        else None
                    ),
                }
                for index, numerator in enumerate(self.view_sum)
            ],
            "masking": {
                "requested_mask_rate": self.requested_mask_rate,
                "realized_mask_rate": (
                    self.primary_count / self.maskable_count
                    if self.maskable_count
                    else 0.0
                ),
                "maskable_note_count": self.maskable_count,
                "primary_masked_count": self.primary_count,
                "collateral_note_count": self.collateral_note_count,
                "collateral_track_count": self.collateral_track_count,
                "collateral_masked_count": (
                    self.collateral_note_count
                    + self.collateral_track_count
                ),
                "last_plan_fingerprints": self.last_plan_fingerprints,
            },
            "anti_collapse_last_batch": (
                None
                if not self.diagnostic_rows
                else self.diagnostic_rows[0]
            ),
            "retained_memory_counters": {
                "peak_live_batches": 1 if self.batch_count else 0,
                "retained_prediction_tensors": 0,
                "retained_diagnostic_rows": len(self.diagnostic_rows),
            },
        }


@torch.no_grad()
def _evaluate(
    model: MaskedGraphSSLModel,
    loader: Iterable[SSLBatch],
    *,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, object]:
    model.eval()
    accumulator = _Accumulator(
        int(config["ssl"]["decoder_views"]),
        note_weight=float(config["ssl"]["note_weight"]),
        bar_weight=float(config["ssl"]["bar_weight"]),
        song_weight=float(config["ssl"]["song_weight"]),
    )
    for cpu_batch in loader:
        batch = move_ssl_batch(
            cpu_batch,
            device,
            non_blocking=bool(config["device"]["non_blocking"]),
        )
        with torch.autocast(
            device_type=device.type,
            enabled=bool(config["device"]["amp"]),
        ):
            output = model(
                batch,
                global_seed=int(config["seed"]),
                epoch=epoch,
                validation=True,
            )
        accumulator.add(output, batch)
    return accumulator.finalize()


def _device_evidence(device: torch.device) -> dict[str, object]:
    common = {
        "resolved_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "amp_supported": device.type in {"cpu", "cuda"},
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }
    if device.type != "cuda":
        return {
            **common,
            "cuda_device_name": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        **common,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _prepare(
    config: dict[str, Any],
) -> tuple[
    Path,
    torch.device,
    SSLDataRuntime,
    MaskedGraphSSLModel,
    torch.optim.Optimizer,
    Any,
    torch.amp.GradScaler,
]:
    _validate_config(config)
    _set_determinism(int(config["seed"]))
    device = _resolve_device(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime = build_ssl_data_runtime(
        OmegaConf.create(config["data"]),
        seed=int(config["seed"]),
    )
    model = build_ssl_model(
        OmegaConf.create(config["model"]),
        OmegaConf.create(config["ssl"]),
    ).to(device)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(config["device"]["amp"]),
    )
    return (
        Path(config["output_dir"]).resolve(),
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
    )


def _managed_paths(output: Path) -> tuple[Path, ...]:
    return tuple(
        output / name
        for name in (
            "resolved_config.json",
            "fingerprints.json",
            "run_manifest.json",
            "metrics.jsonl",
            "epoch_performance.jsonl",
            "last.pt",
            "one_batch.pt",
            "one_batch_report.json",
            "training_report.json",
        )
    )


def _initialize_output(
    output: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = tuple(path for path in _managed_paths(output) if path.exists())
    if resume:
        return
    if existing and not overwrite:
        raise SSLTrainingError(
            "ssl.training.output_exists_without_overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _write_initial_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: SSLDataRuntime,
    model: MaskedGraphSSLModel,
) -> None:
    resolved = copy.deepcopy(config)
    fingerprints = {
        "data": runtime.fingerprints,
        "model_contract_fingerprint": _fingerprint(
            model.ssl_contract_metadata()
        ),
        "checkpoint_binding": ssl_checkpoint_metadata(
            model,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        ),
    }
    manifest = {
        "run_manifest_version": SSL_RUN_MANIFEST_VERSION,
        "artifact_fingerprints": {
            "resolved_config.json": _fingerprint(resolved),
            "fingerprints.json": _fingerprint(fingerprints),
        },
        "checkpoint_binding": fingerprints["checkpoint_binding"],
    }
    _write_json_atomic(output / "resolved_config.json", resolved)
    _write_json_atomic(output / "fingerprints.json", fingerprints)
    _write_json_atomic(output / "run_manifest.json", manifest)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SSLTrainingError(
            f"ssl.training.artifact_unreadable:{path.name}"
        ) from exc


def _validate_resume_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: SSLDataRuntime,
    model: MaskedGraphSSLModel,
) -> None:
    manifest = _read_json(output / "run_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {
        "run_manifest_version",
        "artifact_fingerprints",
        "checkpoint_binding",
    }:
        raise SSLTrainingError("ssl.training.run_manifest_invalid")
    if manifest["run_manifest_version"] != SSL_RUN_MANIFEST_VERSION:
        raise SSLTrainingError(
            "ssl.training.run_manifest_version_incompatible"
        )
    expected_binding = ssl_checkpoint_metadata(
        model,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    if _fingerprint(manifest["checkpoint_binding"]) != _fingerprint(
        expected_binding
    ):
        raise SSLTrainingError(
            "ssl.training.run_manifest_checkpoint_binding_mismatch"
        )
    artifact_fingerprints = manifest["artifact_fingerprints"]
    expected_names = {"resolved_config.json", "fingerprints.json"}
    if (
        not isinstance(artifact_fingerprints, dict)
        or set(artifact_fingerprints) != expected_names
    ):
        raise SSLTrainingError(
            "ssl.training.run_manifest_artifacts_invalid"
        )
    for name in sorted(expected_names):
        if _fingerprint(_read_json(output / name)) != artifact_fingerprints[
            name
        ]:
            raise SSLTrainingError(
                f"ssl.training.artifact_fingerprint_mismatch:{name}"
            )


def _deterministic_repeat(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    seed: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            first = model(batch, global_seed=seed, epoch=0)
            second = model(batch, global_seed=seed, epoch=0)
    return {
        "mask_plans_bit_exact": first.mask_plans == second.mask_plans,
        "online_embeddings_bit_exact": all(
            torch.equal(
                first.online_encoder.fused.embeddings[node_type],
                second.online_encoder.fused.embeddings[node_type],
            )
            for node_type in first.online_encoder.fused.embeddings
        ),
        "decoder_predictions_bit_exact": all(
            torch.equal(left, right)
            for left, right in zip(
                first.decoder_predictions,
                second.decoder_predictions,
                strict=True,
            )
        ),
        "loss_bit_exact": _optional_tensor_equal(
            first.objective.total_loss,
            second.objective.total_loss,
        ),
    }


def _masked_mutation_evidence(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    seed: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    """Bounded proof that planned raw pitch slots cannot reach online outputs."""

    model.eval()
    graph_before = {
        (store_index, name): value.detach().clone()
        for store_index, store in enumerate(batch.raw_graph_batch.stores)
        for name, value in store.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            original = model(batch, global_seed=seed, epoch=0)
    if not any(
        plan.selected_local_node_indices
        for plan in original.mask_plans
    ):
        return {
            "applicable": False,
            "unavailable_reason": "no_masked_rows",
            "fixed_mask_plan": True,
            "raw_graph_stores_bit_exact_after_view": True,
            "online_embeddings_bit_exact_after_masked_mutation": None,
            "online_predictions_bit_exact_after_masked_mutation": None,
            "full_view_target_changed": None,
            "reconstruction_loss_changed": None,
            "passed": None,
        }
    graph = copy.deepcopy(batch.raw_graph_batch)
    note_ptr = graph["note"].ptr
    track_ptr = graph["track"].ptr
    for sample_index, plan in enumerate(original.mask_plans):
        for local_index in plan.selected_local_node_indices:
            row = int(note_ptr[sample_index].item()) + local_index
            for name in ("pitch", "pitch_class", "octave"):
                specs = RAW_FEATURE_REGISTRY.for_node(
                    "note", "categorical"
                )
                column = RAW_FEATURE_REGISTRY.names(
                    "note", "categorical"
                ).index(name)
                vocabulary_size = int(specs[column].vocabulary_size or 0)
                graph["note"].x_cat[row, column] = (
                    int(graph["note"].x_cat[row, column].item()) + 1
                ) % vocabulary_size
            column = RAW_FEATURE_REGISTRY.names(
                "note", "continuous"
            ).index("track_relative_pitch")
            if bool(graph["note"].x_cont_available[row, column]):
                graph["note"].x_cont[row, column] += 7.0
            else:
                graph["note"].x_cont_available[row, column] = True
                graph["note"].x_cont[row, column] = 0.75
        for collateral in plan.collateral_feature_masks:
            ptr = (
                note_ptr
                if collateral.node_type == "note"
                else track_ptr
            )
            for local_index in collateral.local_node_indices:
                row = int(ptr[sample_index].item()) + local_index
                for field in collateral.features:
                    if field.kind != "continuous":
                        raise SSLTrainingError(
                            "ssl.training.unexpected_collateral_kind"
                        )
                    column = RAW_FEATURE_REGISTRY.names(
                        collateral.node_type, "continuous"
                    ).index(field.feature_name)
                    graph[collateral.node_type].x_cont[
                        row, column
                    ] += 9.0
                    graph[collateral.node_type].x_cont_available[
                        row, column
                    ] = True
    mutated = SSLBatch(
        raw_graph_batch=graph,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        sample_count=batch.sample_count,
        node_count=batch.node_count,
        edge_count=batch.edge_count,
    )
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            changed = model(
                mutated,
                global_seed=seed,
                epoch=0,
                mask_plans=original.mask_plans,
            )
    raw_unchanged = all(
        torch.equal(
            value,
            batch.raw_graph_batch.stores[store_index][name],
        )
        for (store_index, name), value in graph_before.items()
    )
    online_equal = all(
        torch.equal(
            original.online_encoder.fused.embeddings[node_type],
            changed.online_encoder.fused.embeddings[node_type],
        )
        for node_type in original.online_encoder.fused.embeddings
    )
    predictions_equal = all(
        torch.equal(left, right)
        for left, right in zip(
            original.decoder_predictions,
            changed.decoder_predictions,
            strict=True,
        )
    )
    target_changed = not torch.equal(
        original.targets.note, changed.targets.note
    )
    loss_changed = not torch.equal(
        original.note_loss.mean, changed.note_loss.mean
    )
    return {
        "applicable": True,
        "unavailable_reason": None,
        "fixed_mask_plan": (
            original.mask_plans == changed.mask_plans
        ),
        "raw_graph_stores_bit_exact_after_view": raw_unchanged,
        "online_embeddings_bit_exact_after_masked_mutation": online_equal,
        "online_predictions_bit_exact_after_masked_mutation": (
            predictions_equal
        ),
        "full_view_target_changed": target_changed,
        "reconstruction_loss_changed": loss_changed,
        "passed": all(
            (
                raw_unchanged,
                online_equal,
                predictions_equal,
                target_changed,
                loss_changed,
            )
        ),
    }


def _one_batch_transfer_evidence(
    model: MaskedGraphSSLModel,
) -> dict[str, object]:
    exported = export_pretrained_encoder_state(model)
    supervised = HierarchicalHeterogeneousBaseline(
        model.encoder_config
    )
    before = {
        name: value.detach().clone()
        for name, value in supervised.state_dict().items()
    }
    report = load_pretrained_encoder_state(supervised, exported)
    after = supervised.state_dict()
    return {
        "encoder_export_contract_version": exported["metadata"][
            "encoder_export_contract_version"
        ],
        "loaded_parameter_count": len(report.loaded_parameters),
        "untouched_parameter_count": len(report.untouched_parameters),
        "supervised_heads_unchanged": all(
            torch.equal(after[name], before[name])
            for name in report.untouched_parameters
        ),
        "loaded_parameters": list(report.loaded_parameters),
        "untouched_parameters": list(report.untouched_parameters),
    }


def _run_one_batch(config: dict[str, Any]) -> dict[str, object]:
    started = time.perf_counter()
    (
        output,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
    ) = _prepare(config)
    del scheduler
    _initialize_output(
        output,
        resume=False,
        overwrite=bool(config["experiment"]["overwrite_output"]),
    )
    _write_initial_artifacts(output, config, runtime, model)
    transfer_started = time.perf_counter()
    batch = move_ssl_batch(
        runtime.first_train_batch,
        device,
        non_blocking=bool(config["device"]["non_blocking"]),
    )
    transfer_seconds = time.perf_counter() - transfer_started
    steps = int(config["experiment"]["steps"])
    initial: dict[str, object] | None = None
    final: dict[str, object] | None = None
    gradient: dict[str, object] | None = None
    optimization_step_count = 0
    forward_seconds = 0.0
    backward_seconds = 0.0

    def measure() -> SSLForwardOutput:
        model.eval()
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                enabled=bool(config["device"]["amp"]),
            ):
                return model(
                    batch,
                    global_seed=int(config["seed"]),
                    epoch=0,
                )

    initial = _batch_metric(measure(), batch)
    for step in range(steps):
        model.train()
        optimized_output, current_gradient, timing = _optimize_batch(
            model,
            batch,
            optimizer,
            scaler,
            config,
            device,
            epoch=0,
            collect_gradient_evidence=(
                step == 0
                or bool(
                    config["experiment"][
                        "collect_gradient_evidence"
                    ]
                )
            ),
        )
        if optimized_output.objective.total_loss is not None:
            optimization_step_count += 1
        forward_seconds += timing["forward_seconds"]
        backward_seconds += timing["backward_seconds"]
        if current_gradient is not None:
            gradient = current_gradient
    final = _batch_metric(measure(), batch)
    repeat = _deterministic_repeat(
        model,
        batch,
        seed=int(config["seed"]),
        device=device,
        amp_enabled=bool(config["device"]["amp"]),
    )
    leakage = _masked_mutation_evidence(
        model,
        batch,
        seed=int(config["seed"]),
        device=device,
        amp_enabled=bool(config["device"]["amp"]),
    )
    checkpoint_path = output / "one_batch.pt"
    save_ssl_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        epoch_journal=(),
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    clone = build_ssl_model(
        OmegaConf.create(config["model"]),
        OmegaConf.create(config["ssl"]),
    ).to(device)
    clone_optimizer = _optimizer(clone, config)
    clone_scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(config["device"]["amp"]),
    )
    state = load_ssl_checkpoint(
        checkpoint_path,
        clone,
        clone_optimizer,
        scheduler=None,
        scaler=clone_scaler,
        maximum_next_epoch=0,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    model.eval()
    clone.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=bool(config["device"]["amp"]),
        ):
            original_reload = model(
                batch, global_seed=int(config["seed"]), epoch=0
            )
            cloned_reload = clone(
                batch, global_seed=int(config["seed"]), epoch=0
            )
    reload_exact = all(
        torch.equal(left, right)
        for left, right in zip(
            original_reload.decoder_predictions,
            cloned_reload.decoder_predictions,
            strict=True,
        )
    ) and _optional_tensor_equal(
        original_reload.objective.total_loss,
        cloned_reload.objective.total_loss,
    )
    transfer = _one_batch_transfer_evidence(model)
    report = {
        "training_report_version": SSL_TRAINING_REPORT_VERSION,
        "evidence_kind": (
            "bounded_phase7a_ssl_plumbing"
            if config["data"]["name"] == "bounded"
            else "production_cache_phase7a_ssl_one_batch_smoke"
        ),
        **_training_scope_evidence(
            data_source_kind=str(config["data"]["name"]),
            run_scope="one_batch_plumbing",
            optimization_step_count=optimization_step_count,
        ),
        "scientific_claim": (
            "one_batch_overfit_proves_plumbing_not_ssl_effectiveness"
        ),
        "sample_count": batch.sample_count,
        "node_count": batch.node_count,
        "edge_count": batch.edge_count,
        "steps": steps,
        "trajectory_measurement_mode": "eval_no_grad",
        "initial": initial,
        "final": final,
        "gradient_coverage": gradient,
        "deterministic_repeat": repeat,
        "no_leakage_mutation_evidence": leakage,
        "checkpoint_reload": {
            "next_epoch": state.next_epoch,
            "bit_exact": reload_exact,
        },
        "encoder_transfer": transfer,
        "learning_rate_used": float(
            config["optimizer"]["learning_rate"]
        ),
        "next_learning_rate": optimizer.param_groups[0]["lr"],
        "stage_timing": {
            "device_transfer_seconds": transfer_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "total_seconds": time.perf_counter() - started,
            "checkpoint_binding_participation": False,
        },
        "retained_memory_counters": {
            "peak_live_batches": 1,
            "retained_prediction_tensors": 0,
            "retained_batch_metric_rows": 2,
        },
        "device": _device_evidence(device),
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "fingerprints": runtime.fingerprints,
        "phase8_started": False,
        "pdmx_added": False,
        "pll_implemented": False,
        "critic_or_quality_score_implemented": False,
    }
    _write_json_atomic(output / "one_batch_report.json", report)
    return report


def _run_epochs(
    config: dict[str, Any],
    *,
    stop_after_epoch: int | None,
) -> dict[str, object]:
    started = time.perf_counter()
    (
        output,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
    ) = _prepare(config)
    resume_path = str(config["experiment"]["resume_from"])
    _initialize_output(
        output,
        resume=bool(resume_path),
        overwrite=bool(config["experiment"]["overwrite_output"]),
    )
    start_epoch = 0
    best: float | None = None
    journal: tuple[dict[str, object], ...] = ()
    if resume_path:
        _validate_resume_artifacts(output, config, runtime, model)
        state: SSLResumeState = load_ssl_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            maximum_next_epoch=int(config["experiment"]["epochs"]),
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        )
        start_epoch = state.next_epoch
        best = state.best_validation_loss
        journal = state.epoch_journal
        # The checkpoint journal is authoritative after any interrupted
        # checkpoint/metric commit boundary.
        _write_jsonl_atomic(output / "metrics.jsonl", journal)
    else:
        _write_initial_artifacts(output, config, runtime, model)
        _write_jsonl_atomic(output / "metrics.jsonl", ())
        _write_jsonl_atomic(output / "epoch_performance.jsonl", ())
    performance_rows: list[dict[str, object]] = []
    performance_path = output / "epoch_performance.jsonl"
    if resume_path and performance_path.exists():
        for line in performance_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if int(row["epoch"]) < start_epoch:
                    performance_rows.append(row)
    recorded_performance_epochs = {
        int(row["epoch"]) for row in performance_rows
    }
    for missing_epoch in range(start_epoch):
        if missing_epoch in recorded_performance_epochs:
            continue
        performance_rows.append(
            {
                "performance_row_version": (
                    SSL_PERFORMANCE_ROW_VERSION
                ),
                "epoch": missing_epoch,
                "next_epoch": missing_epoch + 1,
                "stage_timing": None,
                "unavailable_reason": (
                    "timing_not_committed_before_resume"
                ),
                "checkpoint_binding_participation": False,
            }
        )
    performance_rows.sort(key=lambda row: int(row["epoch"]))
    completed = start_epoch
    epochs = int(config["experiment"]["epochs"])
    for epoch in range(start_epoch, epochs):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        accumulator = _Accumulator(
            int(config["ssl"]["decoder_views"]),
            note_weight=float(config["ssl"]["note_weight"]),
            bar_weight=float(config["ssl"]["bar_weight"]),
            song_weight=float(config["ssl"]["song_weight"]),
        )
        train_started = time.perf_counter()
        forward_seconds = 0.0
        backward_seconds = 0.0
        transfer_seconds = 0.0
        gradient_evidence = None
        for cpu_batch in runtime.train_loader(epoch):
            transfer_started = time.perf_counter()
            batch = move_ssl_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            transfer_seconds += time.perf_counter() - transfer_started
            model.train()
            output_batch, gradient, timing = _optimize_batch(
                model,
                batch,
                optimizer,
                scaler,
                config,
                device,
                epoch=epoch,
                collect_gradient_evidence=(
                    gradient_evidence is None
                    and bool(
                        config["experiment"][
                            "collect_gradient_evidence"
                        ]
                    )
                ),
            )
            if gradient is not None:
                gradient_evidence = gradient
            forward_seconds += timing["forward_seconds"]
            backward_seconds += timing["backward_seconds"]
            accumulator.add(output_batch, batch)
        train_metric = accumulator.finalize()
        if train_metric["batch_count"] == 0:
            raise SSLTrainingError("ssl.training.empty_train_epoch")
        train_seconds = time.perf_counter() - train_started
        if scheduler is not None:
            scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        validation = None
        validation_seconds = None
        if (
            (epoch + 1)
            % int(config["experiment"]["validation_interval"])
            == 0
            or epoch + 1 == epochs
        ):
            validation_started = time.perf_counter()
            validation = _evaluate(
                model,
                runtime.validation_loader(),
                config=config,
                device=device,
                epoch=epoch,
            )
            validation_seconds = time.perf_counter() - validation_started
        row = {
            "metric_row_version": SSL_METRIC_ROW_VERSION,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": next_learning_rate,
            "train": train_metric,
            "validation": validation,
            "gradient_coverage": gradient_evidence,
        }
        validation_loss = (
            None
            if validation is None
            else validation["total_ssl_loss"]
        )
        if validation_loss is not None and (
            best is None or float(validation_loss) < best
        ):
            best = float(validation_loss)
        journal = (*journal, row)
        save_ssl_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=epoch + 1,
            best_validation_loss=best,
            epoch_journal=journal,
            resolved_config=_checkpoint_config(config),
            data_fingerprints=runtime.fingerprints,
        )
        _write_jsonl_atomic(output / "metrics.jsonl", journal)
        performance_rows.append(
            {
                "performance_row_version": SSL_PERFORMANCE_ROW_VERSION,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "stage_timing": {
                    "device_transfer_seconds": transfer_seconds,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "train_total_seconds": train_seconds,
                    "validation_total_seconds": validation_seconds,
                },
                "unavailable_reason": None,
                "checkpoint_binding_participation": False,
            }
        )
        _write_jsonl_atomic(
            output / "epoch_performance.jsonl", performance_rows
        )
        completed = epoch + 1
        if stop_after_epoch is not None and completed >= stop_after_epoch:
            break
    report = {
        "training_report_version": SSL_TRAINING_REPORT_VERSION,
        "evidence_kind": (
            "bounded_phase7a_ssl_pretraining_plumbing"
            if config["data"]["name"] == "bounded"
            else "production_cache_phase7a_ssl_pretraining_run"
        ),
        **_training_scope_evidence(
            data_source_kind=str(config["data"]["name"]),
            run_scope="epoch_pretraining",
            optimization_step_count=sum(
                int(row["train"]["available_batch_count"])
                for row in journal
            ),
        ),
        "start_epoch": start_epoch,
        "completed_epochs": completed,
        "configured_epochs": epochs,
        "best_validation_loss": best,
        "metrics": str(output / "metrics.jsonl"),
        "epoch_performance": str(output / "epoch_performance.jsonl"),
        "last_checkpoint": str(output / "last.pt"),
        "resume_boundary": "epoch_only",
        "mid_epoch_resume_supported": False,
        "validation_membership": asdict(runtime.validation_membership),
        "fingerprints": runtime.fingerprints,
        "device": _device_evidence(device),
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "duration_seconds": time.perf_counter() - started,
        "observed_train_sample_count": sum(
            int(row["train"]["sample_count"]) for row in journal
        ),
        "observed_train_node_count": sum(
            int(row["train"]["node_count"]) for row in journal
        ),
        "observed_train_edge_count": sum(
            int(row["train"]["edge_count"]) for row in journal
        ),
        "phase8_started": False,
        "pdmx_added": False,
        "pll_implemented": False,
        "critic_or_quality_score_implemented": False,
    }
    _write_json_atomic(output / "training_report.json", report)
    return report


def run_ssl_training(
    config: object,
    *,
    stop_after_epoch: int | None = None,
) -> dict[str, object]:
    """Run one bounded overfit or exact epoch-boundary SSL pretraining."""

    _configure_cublas_determinism()
    entry_rng = capture_rng_state()
    try:
        resolved = _plain_config(config)
        if resolved["experiment"]["name"] == "one_batch":
            if stop_after_epoch is not None:
                raise SSLTrainingError(
                    "ssl.training.stop_after_epoch_pretrain_only"
                )
            return _run_one_batch(resolved)
        return _run_epochs(
            resolved,
            stop_after_epoch=stop_after_epoch,
        )
    except Exception:
        restore_rng_state(entry_rng)
        raise


__all__ = [
    "SSL_METRIC_ROW_VERSION",
    "SSL_PERFORMANCE_ROW_VERSION",
    "SSL_RUN_MANIFEST_VERSION",
    "SSL_TRAINING_REPORT_VERSION",
    "SSLTrainingError",
    "run_ssl_training",
]
