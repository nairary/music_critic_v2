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
from torch.nn import functional as F

from music_critic.graph import graph_fingerprint
from music_critic.models import (
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.bounded_fixture import (
    CoherentPitchGroupMutation,
    PHASE7A_PITCH_MUTATION_CONTRACT_VERSION,
    PHASE7A_PITCH_MUTATION_POLICY,
    PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT,
    build_phase7a_bounded_fixture,
    mutate_piece_pitch_group,
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
    collate_ssl_samples,
)
from music_critic.ssl.masking import (
    PreparedMaskBinding,
    move_ssl_batch_with_prepared_binding,
    prepare_mask_binding,
)
from music_critic.ssl.model import (
    MaskedGraphSSLModel,
    SSLForwardOutput,
    build_ssl_model,
)
from music_critic.ssl.objective import (
    StreamingAntiCollapseDiagnostics,
)
from music_critic.ssl.transfer import (
    export_pretrained_encoder_state,
    load_pretrained_encoder_state,
)
from music_critic.training.checkpoint import (
    capture_rng_state,
    restore_rng_state,
)


SSL_RUN_MANIFEST_VERSION = "1.2.0"
SSL_TRAINING_REPORT_VERSION = "1.2.0"
SSL_PERFORMANCE_ROW_VERSION = "1.2.0"
SSL_ONE_BATCH_DEFAULT_LEARNING_RATE = 3e-4


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
        value["optimizer"]["learning_rate"] = (
            SSL_ONE_BATCH_DEFAULT_LEARNING_RATE
            if value["experiment"]["name"] == "one_batch"
            else value["experiment"]["default_learning_rate"]
        )
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


def _non_collapse_mechanics_check(
    diagnostics_by_level: dict[str, object],
) -> dict[str, object]:
    """Apply conservative dtype-aware mechanics checks to bounded evidence."""

    if set(diagnostics_by_level) != {"note", "bar", "song"}:
        raise SSLTrainingError(
            "ssl.training.non_collapse_levels_invalid"
        )
    levels: dict[str, object] = {}
    for level in ("note", "bar", "song"):
        diagnostics = diagnostics_by_level[level]
        if not isinstance(diagnostics, dict):
            raise SSLTrainingError(
                "ssl.training.non_collapse_diagnostics_invalid"
            )
        dtype_name = diagnostics.get("source_dtype")
        dtype = getattr(torch, str(dtype_name), None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise SSLTrainingError(
                "ssl.training.non_collapse_dtype_invalid"
            )
        finfo = torch.finfo(dtype)
        variance_floor = max(finfo.tiny, finfo.eps**2)
        mean_norm_floor = max(finfo.tiny, finfo.eps)
        maximum_near_identical_cosine = 1.0 - max(
            1.0e-4,
            8.0 * finfo.eps**2,
        )
        finite_fields = (
            "target_embedding_variance",
            "prediction_embedding_variance",
            "target_mean_norm",
            "prediction_mean_norm",
            "target_mean_off_diagonal_cosine",
            "prediction_mean_off_diagonal_cosine",
        )
        values = {
            name: diagnostics.get(name) for name in finite_fields
        }
        finite = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        )
        zero_norms = (
            diagnostics.get("target_zero_norm_count") == 0
            and diagnostics.get("prediction_zero_norm_count") == 0
        )
        variance_nondegenerate = all(
            isinstance(values[name], (int, float))
            and float(values[name]) > variance_floor
            for name in (
                "target_embedding_variance",
                "prediction_embedding_variance",
            )
        )
        norm_nondegenerate = all(
            isinstance(values[name], (int, float))
            and float(values[name]) > mean_norm_floor
            for name in (
                "target_mean_norm",
                "prediction_mean_norm",
            )
        )
        not_all_near_identical = all(
            isinstance(values[name], (int, float))
            and float(values[name])
            < maximum_near_identical_cosine
            for name in (
                "target_mean_off_diagonal_cosine",
                "prediction_mean_off_diagonal_cosine",
            )
        )
        passed = all(
            (
                finite,
                zero_norms,
                variance_nondegenerate,
                norm_nondegenerate,
                not_all_near_identical,
            )
        )
        levels[level] = {
            "source_dtype": dtype_name,
            "variance_floor": variance_floor,
            "mean_norm_floor": mean_norm_floor,
            "maximum_near_identical_cosine": (
                maximum_near_identical_cosine
            ),
            "finite": finite,
            "zero_norm_count_is_zero": zero_norms,
            "variance_nondegenerate": variance_nondegenerate,
            "mean_norm_nondegenerate": norm_nondegenerate,
            "not_all_near_identical": not_all_near_identical,
            "passed": passed,
        }
    return {
        "scope": "mechanics_non_collapse_diagnostic",
        "levels": levels,
        "passed": all(
            bool(level["passed"]) for level in levels.values()
        ),
    }


def _bounded_held_out_acceptance(
    initial_validation: dict[str, object],
    journal: tuple[dict[str, object], ...],
    *,
    completed_epochs: int,
    configured_epochs: int,
    best_validation_loss: float | None,
) -> dict[str, object]:
    """Summarize finite fixed-validation mechanics without efficacy claims."""

    validation_metrics = [
        initial_validation,
        *[
            row["validation"]
            for row in journal
            if row["validation"] is not None
        ],
    ]
    stage_metrics = [
        initial_validation,
        *[row["train"] for row in journal],
        *[
            row["validation"]
            for row in journal
            if row["validation"] is not None
        ],
    ]
    finite_losses = all(
        isinstance(metric["total_ssl_loss"], (int, float))
        and not isinstance(metric["total_ssl_loss"], bool)
        and math.isfinite(float(metric["total_ssl_loss"]))
        for metric in stage_metrics
    )
    diagnostic_float_fields = (
        "target_embedding_variance",
        "prediction_embedding_variance",
        "target_mean_norm",
        "prediction_mean_norm",
        "target_mean_off_diagonal_cosine",
        "prediction_mean_off_diagonal_cosine",
    )
    finite_diagnostics = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for metric in stage_metrics
        for diagnostics in metric["anti_collapse_aggregate"].values()
        for value in (
            diagnostics[field] for field in diagnostic_float_fields
        )
    )
    non_collapsed = all(
        metric["non_collapse_acceptance"]["passed"] is True
        for metric in stage_metrics
    )
    validation_plan_fingerprints = [
        metric["masking"]["plan_fingerprints"]
        for metric in validation_metrics
    ]
    validation_binding_fingerprints = [
        metric["masking"]["prepared_mask_binding_fingerprints"]
        for metric in validation_metrics
    ]
    fixed_validation_plans = bool(validation_plan_fingerprints) and all(
        fingerprints == validation_plan_fingerprints[0]
        for fingerprints in validation_plan_fingerprints[1:]
    )
    fixed_validation_bindings = bool(
        validation_binding_fingerprints
    ) and all(
        fingerprints == validation_binding_fingerprints[0]
        for fingerprints in validation_binding_fingerprints[1:]
    )
    every_epoch_has_validation = (
        len(validation_metrics) == len(journal) + 1
    )
    initial_optimizer_step_count = initial_validation.get(
        "optimizer_step_count_at_measurement"
    )
    initial_measurement_before_optimizer = (
        initial_optimizer_step_count == 0
    )
    finite_validation_losses = [
        float(metric["total_ssl_loss"])
        for metric in validation_metrics[1:]
        if isinstance(metric["total_ssl_loss"], (int, float))
        and not isinstance(metric["total_ssl_loss"], bool)
        and math.isfinite(float(metric["total_ssl_loss"]))
    ]
    validation_checkpoint_selection_only = (
        bool(finite_validation_losses)
        and best_validation_loss == min(finite_validation_losses)
    )
    multiple_epochs = (
        configured_epochs >= 2
        and completed_epochs >= 2
        and len(journal) >= 2
    )
    trajectory_complete = (
        completed_epochs == configured_epochs == len(journal)
    )
    return {
        "scope": "bounded_held_out_mechanics_non_collapse_only",
        "initial_measurement_before_optimizer": (
            initial_measurement_before_optimizer
        ),
        "initial_optimizer_step_count": (
            initial_optimizer_step_count
        ),
        "validation_mask_epoch": 0,
        "validation_checkpoint_selection_only": (
            validation_checkpoint_selection_only
        ),
        "checkpoint_selection_metric": (
            "minimum_fixed_validation_total_ssl_loss"
        ),
        "finite_losses": finite_losses,
        "finite_aggregate_diagnostics": finite_diagnostics,
        "finite_losses_and_diagnostics": (
            finite_losses and finite_diagnostics
        ),
        "non_collapse_checks_passed": non_collapsed,
        "fixed_validation_plan_fingerprints": (
            validation_plan_fingerprints[0]
            if validation_plan_fingerprints
            else []
        ),
        "fixed_validation_plans_across_trajectory": (
            fixed_validation_plans
        ),
        "fixed_validation_prepared_binding_fingerprints": (
            validation_binding_fingerprints[0]
            if validation_binding_fingerprints
            else []
        ),
        "fixed_validation_bindings_across_trajectory": (
            fixed_validation_bindings
        ),
        "every_epoch_has_validation": every_epoch_has_validation,
        "multiple_epochs": multiple_epochs,
        "trajectory_complete": trajectory_complete,
        "completed_epochs": completed_epochs,
        "configured_epochs": configured_epochs,
        "initial_validation_loss": initial_validation[
            "total_ssl_loss"
        ],
        "epoch_trajectory": [
            {
                "epoch": row["epoch"],
                "train_total_ssl_loss": row["train"][
                    "total_ssl_loss"
                ],
                "validation_total_ssl_loss": (
                    None
                    if row["validation"] is None
                    else row["validation"]["total_ssl_loss"]
                ),
            }
            for row in journal
        ],
        "effectiveness_claim": False,
        "passed": all(
            (
                finite_losses,
                finite_diagnostics,
                non_collapsed,
                fixed_validation_plans,
                fixed_validation_bindings,
                every_epoch_has_validation,
                multiple_epochs,
                trajectory_complete,
                initial_measurement_before_optimizer,
                validation_checkpoint_selection_only,
            )
        ),
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
    note_stream = StreamingAntiCollapseDiagnostics().update(
        output.targets.note.index_select(
            0,
            output.selected_global_note_indices,
        ),
        torch.stack(output.decoder_predictions, dim=0).mean(dim=0),
    )
    bar_stream = StreamingAntiCollapseDiagnostics().update(
        output.bar_latent.target,
        output.bar_latent.prediction,
    )
    song_stream = StreamingAntiCollapseDiagnostics().update(
        output.song_latent.target,
        output.song_latent.prediction,
    )
    anti_collapse = {
        "note": note_stream.to_dict(),
        "bar": bar_stream.to_dict(),
        "song": song_stream.to_dict(),
    }
    non_collapse_available = all(
        int(diagnostics["row_count"]) >= 2
        and diagnostics["unavailable_reason"] is None
        and diagnostics["pairwise_unavailable_reason"] is None
        for diagnostics in anti_collapse.values()
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
            "prepared_mask_binding_fingerprint": (
                output.prepared_mask_binding_fingerprint
            ),
            "overlay_fingerprint": output.feature_overlay.fingerprint,
        },
        "anti_collapse": anti_collapse,
        "non_collapse_acceptance": (
            _non_collapse_mechanics_check(anti_collapse)
            if non_collapse_available
            else {
                "scope": "mechanics_non_collapse_diagnostic",
                "levels": None,
                "passed": None,
                "unavailable_reason": (
                    "fewer_than_two_rows_in_at_least_one_level"
                ),
            }
        ),
        "sample_count": batch.sample_count,
        "node_count": batch.node_count,
        "edge_count": batch.edge_count,
    }


def _prepare_and_move_batch(
    cpu_batch: SSLBatch,
    model: MaskedGraphSSLModel,
    config: dict[str, Any],
    device: torch.device,
    *,
    epoch: int,
    validation: bool,
) -> tuple[SSLBatch, PreparedMaskBinding, dict[str, float]]:
    """Prepare the target-blind CPU binding before any device transfer."""

    plan_started = time.perf_counter()
    binding = prepare_mask_binding(
        cpu_batch,
        global_seed=int(config["seed"]),
        epoch=epoch,
        stage="validation" if validation else "train",
        requested_mask_rate=model.ssl_config.mask_rate,
    )
    plan_seconds = time.perf_counter() - plan_started
    transfer_started = time.perf_counter()
    batch, moved_binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        binding,
        device,
        non_blocking=bool(config["device"]["non_blocking"]),
    )
    transfer_seconds = time.perf_counter() - transfer_started
    if not isinstance(batch, SSLBatch):
        raise SSLTrainingError(
            "ssl.training.prepared_device_batch_invalid"
        )
    return batch, moved_binding, {
        "mask_plan_preparation_seconds": plan_seconds,
        "device_transfer_seconds": transfer_seconds,
    }


def _optimize_batch(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    *,
    prepared_mask_binding: PreparedMaskBinding | None = None,
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
        if prepared_mask_binding is None:
            if isinstance(model, MaskedGraphSSLModel):
                raise SSLTrainingError(
                    "ssl.training.prepared_mask_binding_required"
                )
            output = model(batch)
        else:
            output = model(
                batch,
                prepared_mask_binding=prepared_mask_binding,
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
        self.diagnostics = {
            "note": StreamingAntiCollapseDiagnostics(),
            "bar": StreamingAntiCollapseDiagnostics(),
            "song": StreamingAntiCollapseDiagnostics(),
        }
        self.plan_fingerprints: list[str] = []
        self.prepared_binding_fingerprints: list[str] = []
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
        self.plan_fingerprints.extend(masking["plan_fingerprints"])
        self.prepared_binding_fingerprints.append(
            output.prepared_mask_binding_fingerprint
        )
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
        selected_target = output.targets.note.index_select(
            0,
            output.selected_global_note_indices,
        )
        mean_note_prediction = torch.stack(
            output.decoder_predictions,
            dim=0,
        ).mean(dim=0)
        self.diagnostics["note"].update(
            selected_target,
            mean_note_prediction,
        )
        self.diagnostics["bar"].update(
            output.bar_latent.target,
            output.bar_latent.prediction,
        )
        self.diagnostics["song"].update(
            output.song_latent.target,
            output.song_latent.prediction,
        )

    @staticmethod
    def _mean(numerator: float, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    def finalize(self) -> dict[str, object]:
        aggregate_diagnostics = {
            level: diagnostics.to_dict()
            for level, diagnostics in self.diagnostics.items()
        }
        non_collapse_available = all(
            int(diagnostics["row_count"]) >= 2
            and diagnostics["unavailable_reason"] is None
            and diagnostics["pairwise_unavailable_reason"] is None
            for diagnostics in aggregate_diagnostics.values()
        )
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
                "plan_fingerprints": sorted(self.plan_fingerprints),
                "prepared_mask_binding_fingerprints": sorted(
                    self.prepared_binding_fingerprints
                ),
            },
            "anti_collapse_aggregate": aggregate_diagnostics,
            "non_collapse_acceptance": (
                _non_collapse_mechanics_check(
                    aggregate_diagnostics
                )
                if non_collapse_available
                else {
                    "scope": (
                        "mechanics_non_collapse_diagnostic"
                    ),
                    "levels": None,
                    "passed": None,
                    "unavailable_reason": (
                        "fewer_than_two_rows_in_at_least_one_level"
                    ),
                }
            ),
            "diagnostic_accumulator_retained_state": {
                "scope": "anti_collapse_sufficient_statistics_only",
                "retained_embedding_history_rows": 0,
                "retained_prediction_history_tensors": 0,
                "retained_diagnostic_tensor_elements": sum(
                    diagnostics.retained_tensor_elements
                    for diagnostics in self.diagnostics.values()
                ),
                "dimension_linear_not_row_linear": True,
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
    stage_timing: dict[str, float] | None = None,
) -> dict[str, object]:
    model.eval()
    accumulator = _Accumulator(
        int(config["ssl"]["decoder_views"]),
        note_weight=float(config["ssl"]["note_weight"]),
        bar_weight=float(config["ssl"]["bar_weight"]),
        song_weight=float(config["ssl"]["song_weight"]),
    )
    for cpu_batch in loader:
        batch, binding, preparation_timing = _prepare_and_move_batch(
            cpu_batch,
            model,
            config,
            device,
            epoch=epoch,
            validation=True,
        )
        forward_started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            enabled=bool(config["device"]["amp"]),
        ):
            output = model(
                batch,
                prepared_mask_binding=binding,
            )
        forward_seconds = time.perf_counter() - forward_started
        if stage_timing is not None:
            stage_timing["mask_plan_preparation_seconds"] = (
                stage_timing.get(
                    "mask_plan_preparation_seconds",
                    0.0,
                )
                + preparation_timing[
                    "mask_plan_preparation_seconds"
                ]
            )
            stage_timing["device_transfer_seconds"] = (
                stage_timing.get("device_transfer_seconds", 0.0)
                + preparation_timing["device_transfer_seconds"]
            )
            stage_timing["forward_seconds"] = (
                stage_timing.get("forward_seconds", 0.0)
                + forward_seconds
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
            "initial_validation.json",
            "metrics.jsonl",
            "epoch_performance.jsonl",
            "last.pt",
            "best.pt",
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
    *,
    initial_validation: dict[str, object] | None = None,
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
    artifacts = {
        "resolved_config.json": _fingerprint(resolved),
        "fingerprints.json": _fingerprint(fingerprints),
    }
    if initial_validation is not None:
        artifacts["initial_validation.json"] = _fingerprint(
            initial_validation
        )
    manifest = {
        "run_manifest_version": SSL_RUN_MANIFEST_VERSION,
        "artifact_fingerprints": artifacts,
        "checkpoint_binding": fingerprints["checkpoint_binding"],
    }
    _write_json_atomic(output / "resolved_config.json", resolved)
    _write_json_atomic(output / "fingerprints.json", fingerprints)
    if initial_validation is not None:
        _write_json_atomic(
            output / "initial_validation.json",
            initial_validation,
        )
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
    if config["experiment"]["name"] == "pretrain":
        expected_names.add("initial_validation.json")
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
    prepared_mask_binding: PreparedMaskBinding,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            first = model(
                batch,
                prepared_mask_binding=prepared_mask_binding,
            )
            second = model(
                batch,
                prepared_mask_binding=prepared_mask_binding,
            )
    return {
        "prepared_mask_binding_fingerprint_bit_exact": (
            first.prepared_mask_binding_fingerprint
            == second.prepared_mask_binding_fingerprint
            == prepared_mask_binding.fingerprint
        ),
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


def _snapshot_store_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def _snapshot_store_value_equal(
    current: object,
    snapshot: object,
) -> bool:
    if torch.is_tensor(snapshot):
        return torch.is_tensor(current) and torch.equal(
            current,
            snapshot,
        )
    if isinstance(snapshot, dict):
        return (
            isinstance(current, dict)
            and current.keys() == snapshot.keys()
            and all(
                _snapshot_store_value_equal(
                    current[key],
                    snapshot[key],
                )
                for key in snapshot
            )
        )
    if isinstance(snapshot, (list, tuple)):
        return (
            type(current) is type(snapshot)
            and len(current) == len(snapshot)
            and all(
                _snapshot_store_value_equal(left, right)
                for left, right in zip(
                    current,
                    snapshot,
                    strict=True,
                )
            )
        )
    return bool(current == snapshot)


def _graph_store_snapshot(
    graph: Any,
) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
    """Snapshot tensor values and non-tensor metadata in every PyG store."""

    return tuple(
        (
            repr(getattr(store, "_key", None)),
            tuple(
                (str(name), _snapshot_store_value(value))
                for name, value in store.items()
            ),
        )
        for store in graph.stores
    )


def _graph_store_snapshot_matches(
    graph: Any,
    snapshot: tuple[
        tuple[str, tuple[tuple[str, object], ...]],
        ...,
    ],
) -> bool:
    current = tuple(
        (
            repr(getattr(store, "_key", None)),
            tuple((str(name), value) for name, value in store.items()),
        )
        for store in graph.stores
    )
    if len(current) != len(snapshot):
        return False
    for (current_key, current_items), (
        snapshot_key,
        snapshot_items,
    ) in zip(current, snapshot, strict=True):
        if current_key != snapshot_key or len(current_items) != len(
            snapshot_items
        ):
            return False
        for (current_name, current_value), (
            snapshot_name,
            snapshot_value,
        ) in zip(current_items, snapshot_items, strict=True):
            if (
                current_name != snapshot_name
                or not _snapshot_store_value_equal(
                    current_value,
                    snapshot_value,
                )
            ):
                return False
    return True


def _per_sample_cpu_raw_graph_fingerprints(
    batch: SSLBatch,
) -> tuple[str, ...]:
    """Fingerprint the actual validated CPU samples used for plan binding."""

    if any(
        value.device.type != "cpu"
        for store in batch.raw_graph_batch.stores
        for value in store.values()
        if torch.is_tensor(value)
    ):
        raise SSLTrainingError(
            "ssl.training.runtime_source_binding_requires_cpu_batch"
        )
    graphs = batch.raw_graph_batch.to_data_list()
    for graph in graphs:
        if torch.is_tensor(graph.raw_only):
            graph.raw_only = bool(graph.raw_only.item())
    return tuple(graph_fingerprint(graph) for graph in graphs)


def _runtime_mutation_source_binding(
    cpu_batch: SSLBatch,
    mutations: tuple[CoherentPitchGroupMutation, ...],
) -> dict[str, object]:
    """Bind rebuilt alternative targets to the exact runtime source graphs."""

    runtime_fingerprints = _per_sample_cpu_raw_graph_fingerprints(
        cpu_batch
    )
    if len(runtime_fingerprints) != len(mutations):
        raise SSLTrainingError(
            "ssl.training.runtime_source_binding_length_mismatch"
        )
    rows = []
    for dataset_id, piece_id, runtime_fingerprint, mutation in zip(
        cpu_batch.dataset_ids,
        cpu_batch.piece_ids,
        runtime_fingerprints,
        mutations,
        strict=True,
    ):
        identity_exact = (
            dataset_id == mutation.source_piece.dataset_name
            and piece_id == mutation.source_piece.piece_id
        )
        fingerprint_exact = (
            runtime_fingerprint
            == mutation.source_raw_graph_fingerprint
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "piece_id": piece_id,
                "runtime_raw_graph_fingerprint": (
                    runtime_fingerprint
                ),
                "rebuilt_source_raw_graph_fingerprint": (
                    mutation.source_raw_graph_fingerprint
                ),
                "identity_exact": identity_exact,
                "fingerprint_exact": fingerprint_exact,
                "passed": identity_exact and fingerprint_exact,
            }
        )
    return {
        "scope": "actual_cpu_runtime_to_rebuilt_canonical_source",
        "per_sample": rows,
        "passed": all(bool(row["passed"]) for row in rows),
    }


def _masked_mutation_evidence(
    model: MaskedGraphSSLModel,
    cpu_batch: SSLBatch,
    batch: SSLBatch,
    prepared_mask_binding: PreparedMaskBinding,
    *,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    """Compare predictions with correct and coherently pitch-mutated targets."""

    cpu_graph_snapshot = _graph_store_snapshot(
        cpu_batch.raw_graph_batch
    )
    device_graph_snapshot = _graph_store_snapshot(
        batch.raw_graph_batch
    )
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            original = model(
                batch,
                prepared_mask_binding=prepared_mask_binding,
            )
    original_cpu_unchanged = _graph_store_snapshot_matches(
        cpu_batch.raw_graph_batch,
        cpu_graph_snapshot,
    )
    original_device_unchanged = _graph_store_snapshot_matches(
        batch.raw_graph_batch,
        device_graph_snapshot,
    )
    if not any(
        plan.selected_local_node_indices
        for plan in original.mask_plans
    ):
        return {
            "applicable": False,
            "unavailable_reason": "no_masked_rows",
            "fixed_mask_plan": True,
            "raw_graph_stores_bit_exact_after_view": (
                original_cpu_unchanged and original_device_unchanged
            ),
            "raw_graph_store_immutability": {
                "original_cpu": original_cpu_unchanged,
                "original_device": original_device_unchanged,
                "mutated_cpu": None,
                "mutated_device": None,
            },
            "online_embeddings_bit_exact_after_masked_mutation": None,
            "online_predictions_bit_exact_after_masked_mutation": None,
            "full_view_target_changed": None,
            "reconstruction_loss_changed": None,
            "cosine_prediction_correct_target": None,
            "cosine_prediction_pitch_mutated_target": None,
            "correct_minus_mutated_margin": None,
            "target_to_mutated_target_cosine_distance": None,
            "target_to_mutated_target_mean_l2_distance": None,
            "positive_margin_floor": None,
            "passed": None,
        }
    if config["data"]["name"] != "bounded":
        return {
            "applicable": False,
            "unavailable_reason": (
                "coherent_canonical_mutation_is_bounded_fixture_only"
            ),
            "fixed_mask_plan": None,
            "passed": None,
        }
    fixture = build_phase7a_bounded_fixture()
    mutations = tuple(
        mutate_piece_pitch_group(
            fixture.piece_by_identity(dataset_id, piece_id),
            plan.selected_local_node_indices,
        )
        for dataset_id, piece_id, plan in zip(
            cpu_batch.dataset_ids,
            cpu_batch.piece_ids,
            original.mask_plans,
            strict=True,
        )
    )
    runtime_source_binding = _runtime_mutation_source_binding(
        cpu_batch,
        mutations,
    )
    mutated_cpu_batch = collate_ssl_samples(
        tuple(mutation.raw_sample(mutated=True) for mutation in mutations)
    )
    mutated_batch, mutated_binding, _timing = _prepare_and_move_batch(
        mutated_cpu_batch,
        model,
        config,
        device,
        epoch=0,
        validation=False,
    )
    mutated_cpu_graph_snapshot = _graph_store_snapshot(
        mutated_cpu_batch.raw_graph_batch
    )
    mutated_device_graph_snapshot = _graph_store_snapshot(
        mutated_batch.raw_graph_batch
    )
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            changed = model(
                mutated_batch,
                prepared_mask_binding=mutated_binding,
            )
    original_cpu_unchanged = (
        original_cpu_unchanged
        and _graph_store_snapshot_matches(
            cpu_batch.raw_graph_batch,
            cpu_graph_snapshot,
        )
    )
    original_device_unchanged = (
        original_device_unchanged
        and _graph_store_snapshot_matches(
            batch.raw_graph_batch,
            device_graph_snapshot,
        )
    )
    mutated_cpu_unchanged = _graph_store_snapshot_matches(
        mutated_cpu_batch.raw_graph_batch,
        mutated_cpu_graph_snapshot,
    )
    mutated_device_unchanged = _graph_store_snapshot_matches(
        mutated_batch.raw_graph_batch,
        mutated_device_graph_snapshot,
    )
    raw_unchanged = all(
        (
            original_cpu_unchanged,
            original_device_unchanged,
            mutated_cpu_unchanged,
            mutated_device_unchanged,
        )
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
        original.targets.note.index_select(
            0,
            original.selected_global_note_indices,
        ),
        changed.targets.note.index_select(
            0,
            changed.selected_global_note_indices,
        ),
    )
    loss_changed = not torch.equal(
        original.note_loss.mean, changed.note_loss.mean
    )
    prediction = torch.stack(
        original.decoder_predictions,
        dim=0,
    ).mean(dim=0)
    correct_target = original.targets.note.index_select(
        0,
        original.selected_global_note_indices,
    )
    mutated_target = changed.targets.note.index_select(
        0,
        changed.selected_global_note_indices,
    )
    correct_cosine = F.cosine_similarity(
        prediction,
        correct_target,
        dim=-1,
        eps=1e-8,
    ).mean()
    mutated_cosine = F.cosine_similarity(
        prediction,
        mutated_target,
        dim=-1,
        eps=1e-8,
    ).mean()
    target_mutated_cosine = F.cosine_similarity(
        correct_target,
        mutated_target,
        dim=-1,
        eps=1e-8,
    ).mean()
    target_mutated_l2 = torch.linalg.vector_norm(
        correct_target - mutated_target,
        dim=-1,
    ).mean()
    margin = correct_cosine - mutated_cosine
    margin_floor = 8.0 * torch.finfo(prediction.dtype).eps
    metrics_finite = bool(
        torch.isfinite(
            torch.stack(
                (
                    correct_cosine,
                    mutated_cosine,
                    target_mutated_cosine,
                    target_mutated_l2,
                    margin,
                )
            )
        ).all()
    )
    fixed_binding = (
        prepared_mask_binding.fingerprint
        == mutated_binding.fingerprint
    )
    positive_margin = bool(margin > margin_floor)
    positive_target_distance = bool(
        target_mutated_l2 > torch.finfo(prediction.dtype).eps
    )
    return {
        "applicable": True,
        "unavailable_reason": None,
        "fixed_mask_plan": (
            original.mask_plans == changed.mask_plans
        ),
        "fixed_prepared_binding_fingerprint": fixed_binding,
        "prepared_mask_binding_fingerprint": (
            prepared_mask_binding.fingerprint
        ),
        "mutation_contract_version": (
            PHASE7A_PITCH_MUTATION_CONTRACT_VERSION
        ),
        "mutation_policy": PHASE7A_PITCH_MUTATION_POLICY,
        "mutation_policy_fingerprint": (
            PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT
        ),
        "raw_graph_stores_bit_exact_after_view": raw_unchanged,
        "raw_graph_store_immutability": {
            "original_cpu": original_cpu_unchanged,
            "original_device": original_device_unchanged,
            "mutated_cpu": mutated_cpu_unchanged,
            "mutated_device": mutated_device_unchanged,
        },
        "runtime_source_binding": runtime_source_binding,
        "online_embeddings_bit_exact_after_masked_mutation": online_equal,
        "online_predictions_bit_exact_after_masked_mutation": (
            predictions_equal
        ),
        "full_view_target_changed": target_changed,
        "reconstruction_loss_changed": loss_changed,
        "source_dtype": str(prediction.dtype).removeprefix("torch."),
        "cosine_prediction_correct_target": _scalar(
            correct_cosine
        ),
        "cosine_prediction_pitch_mutated_target": _scalar(
            mutated_cosine
        ),
        "correct_minus_mutated_margin": _scalar(margin),
        "target_to_mutated_target_cosine_distance": _scalar(
            1.0 - target_mutated_cosine
        ),
        "target_to_mutated_target_mean_l2_distance": _scalar(
            target_mutated_l2
        ),
        "positive_margin_floor": margin_floor,
        "metrics_finite": metrics_finite,
        "positive_margin": positive_margin,
        "positive_target_distance": positive_target_distance,
        "coherent_mutations": [
            {
                "dataset_id": mutation.source_piece.dataset_name,
                "piece_id": mutation.source_piece.piece_id,
                "selected_local_node_indices": list(
                    mutation.selected_local_node_indices
                ),
                "selected_note_ids": list(
                    mutation.selected_note_ids
                ),
                "source_pitches": list(mutation.source_pitches),
                "mutated_pitches": list(mutation.mutated_pitches),
                "mutation_instance_fingerprint": (
                    mutation.mutation_instance_fingerprint
                ),
                "mask_plan_fingerprint": plan.fingerprint,
                "source_raw_graph_fingerprint": (
                    mutation.source_raw_graph_fingerprint
                ),
                "mutated_raw_graph_fingerprint": (
                    mutation.mutated_raw_graph_fingerprint
                ),
                "changed_feature_slots": [
                    list(slot)
                    for slot in mutation.changed_feature_slots
                ],
            }
            for mutation, plan in zip(
                mutations,
                original.mask_plans,
                strict=True,
            )
        ],
        "passed": all(
            (
                raw_unchanged,
                runtime_source_binding["passed"],
                fixed_binding,
                online_equal,
                predictions_equal,
                target_changed,
                loss_changed,
                metrics_finite,
                positive_margin,
                positive_target_distance,
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
    cpu_batch = runtime.first_train_batch
    batch, prepared_mask_binding, preparation_timing = (
        _prepare_and_move_batch(
            cpu_batch,
            model,
            config,
            device,
            epoch=0,
            validation=False,
        )
    )
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
                    prepared_mask_binding=prepared_mask_binding,
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
            prepared_mask_binding=prepared_mask_binding,
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
        prepared_mask_binding,
        device=device,
        amp_enabled=bool(config["device"]["amp"]),
    )
    leakage = _masked_mutation_evidence(
        model,
        cpu_batch,
        batch,
        prepared_mask_binding,
        config=config,
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
                batch,
                prepared_mask_binding=prepared_mask_binding,
            )
            cloned_reload = clone(
                batch,
                prepared_mask_binding=prepared_mask_binding,
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
        "prepared_mask_binding": prepared_mask_binding.to_dict(),
        "gradient_coverage": gradient,
        "deterministic_repeat": repeat,
        "no_leakage_mutation_evidence": leakage,
        "pitch_sensitive_reconstruction_evidence": leakage,
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
            "mask_plan_preparation_seconds": preparation_timing[
                "mask_plan_preparation_seconds"
            ],
            "device_transfer_seconds": preparation_timing[
                "device_transfer_seconds"
            ],
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "total_seconds": time.perf_counter() - started,
            "checkpoint_binding_participation": False,
        },
        "device": _device_evidence(device),
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "fingerprints": runtime.fingerprints,
        "data_composition": runtime.mixture_statistics,
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
    optimization_step_count = 0
    initial_validation: dict[str, object]
    if resume_path:
        _validate_resume_artifacts(output, config, runtime, model)
        loaded_initial_validation = _read_json(
            output / "initial_validation.json"
        )
        if not isinstance(loaded_initial_validation, dict):
            raise SSLTrainingError(
                "ssl.training.initial_validation_invalid"
            )
        initial_validation = loaded_initial_validation
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
        optimization_step_count = sum(
            int(row["train"]["available_batch_count"])
            for row in journal
        )
        # The checkpoint journal is authoritative after any interrupted
        # checkpoint/metric commit boundary.
        _write_jsonl_atomic(output / "metrics.jsonl", journal)
    else:
        # The fixed validation baseline is measured before the first optimizer
        # mutation and uses the canonical validation epoch-zero mask binding.
        initial_validation = _evaluate(
            model,
            runtime.validation_loader(),
            config=config,
            device=device,
            epoch=0,
        )
        initial_validation["optimizer_step_count_at_measurement"] = (
            optimization_step_count
        )
        _write_initial_artifacts(
            output,
            config,
            runtime,
            model,
            initial_validation=initial_validation,
        )
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
        plan_preparation_seconds = 0.0
        gradient_evidence = None
        for cpu_batch in runtime.train_loader(epoch):
            batch, prepared_mask_binding, preparation_timing = (
                _prepare_and_move_batch(
                    cpu_batch,
                    model,
                    config,
                    device,
                    epoch=epoch,
                    validation=False,
                )
            )
            plan_preparation_seconds += preparation_timing[
                "mask_plan_preparation_seconds"
            ]
            transfer_seconds += preparation_timing[
                "device_transfer_seconds"
            ]
            model.train()
            output_batch, gradient, timing = _optimize_batch(
                model,
                batch,
                optimizer,
                scaler,
                config,
                device,
                prepared_mask_binding=prepared_mask_binding,
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
        optimization_step_count += int(
            train_metric["available_batch_count"]
        )
        train_seconds = time.perf_counter() - train_started
        if scheduler is not None:
            scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        validation = None
        validation_seconds = None
        validation_stage_timing: dict[str, float] = {}
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
                stage_timing=validation_stage_timing,
            )
            validation["optimizer_step_count_at_measurement"] = (
                optimization_step_count
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
        improved = validation_loss is not None and (
            best is None or float(validation_loss) < best
        )
        if improved:
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
        if improved:
            save_ssl_checkpoint(
                output / "best.pt",
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
                    "mask_plan_preparation_seconds": (
                        plan_preparation_seconds
                    ),
                    "device_transfer_seconds": transfer_seconds,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                    "train_total_seconds": train_seconds,
                    "validation_total_seconds": validation_seconds,
                    "validation_mask_plan_preparation_seconds": (
                        validation_stage_timing.get(
                            "mask_plan_preparation_seconds"
                        )
                    ),
                    "validation_device_transfer_seconds": (
                        validation_stage_timing.get(
                            "device_transfer_seconds"
                        )
                    ),
                    "validation_forward_seconds": (
                        validation_stage_timing.get(
                            "forward_seconds"
                        )
                    ),
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
            "bounded_phase7a_ssl_held_out_noncollapse"
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
        "held_out_acceptance": (
            _bounded_held_out_acceptance(
                initial_validation,
                journal,
                completed_epochs=completed,
                configured_epochs=epochs,
                best_validation_loss=best,
            )
            if config["data"]["name"] == "bounded"
            else {
                "scope": "production_cache_training",
                "passed": None,
                "unavailable_reason": (
                    "bounded_acceptance_not_applicable"
                ),
            }
        ),
        "initial_validation": initial_validation,
        "initial_validation_artifact": str(
            output / "initial_validation.json"
        ),
        "best_validation_loss": best,
        "best_validation_epoch": (
            None
            if best is None
            else min(
                (
                    int(row["epoch"])
                    for row in journal
                    if row["validation"] is not None
                    and row["validation"]["total_ssl_loss"] == best
                ),
                default=None,
            )
        ),
        "best_checkpoint_selection": (
            "minimum_fixed_validation_total_ssl_loss"
        ),
        "best_checkpoint": (
            None if best is None else str(output / "best.pt")
        ),
        "metrics": str(output / "metrics.jsonl"),
        "epoch_performance": str(output / "epoch_performance.jsonl"),
        "last_checkpoint": str(output / "last.pt"),
        "resume_boundary": "epoch_only",
        "mid_epoch_resume_supported": False,
        "validation_membership": asdict(runtime.validation_membership),
        "fingerprints": runtime.fingerprints,
        "data_composition": runtime.mixture_statistics,
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
    "SSL_ONE_BATCH_DEFAULT_LEARNING_RATE",
    "SSL_PERFORMANCE_ROW_VERSION",
    "SSL_RUN_MANIFEST_VERSION",
    "SSL_TRAINING_REPORT_VERSION",
    "SSLTrainingError",
    "run_ssl_training",
]
