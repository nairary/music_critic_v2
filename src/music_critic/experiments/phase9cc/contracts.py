"""Immutable protocol and plan for the Phase 9C-C convergence diagnostic."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Mapping

from music_critic.data.validation_membership import fixed_validation_membership
from music_critic.experiments.phase8b2.schedule import (
    SeedDomains,
    build_raw_downstream_sample_schedule,
)
from music_critic.experiments.phase9cb.contracts import (
    PHASE9CB_SOURCE_KINDS,
    _validate_encoder_export,
    _validate_weight_artifacts,
    canonical_bytes,
    file_sha256,
    fingerprint,
)
from music_critic.training.config import DataConfig
from music_critic.training.data import build_corpus_data_views


PHASE9CC_PROTOCOL_VERSION = "1.0.0"
PHASE9CC_PLAN_VERSION = "1.0.0"
PHASE9CC_SEED = 17
PHASE9CC_BASE_SHA = "786d0dd9320545f2eee50b6d59e609e72d96da49"
PHASE9CC_CELLS = ("scratch_mlp", "ssl_mlp")
PHASE9CC_TASKS = (
    "dilemmadata.an.chord.inversion",
    "dilemmadata.an.chord.quality",
    "dilemmadata.dlc.chord.inversion",
    "dilemmadata.dlc.chord.quality",
)
PHASE9CC_MILESTONES = (0, 1000, 3000, 6000, 9000)


class Phase9CCError(ValueError):
    """Stable Phase 9C-C planning, runtime, or verification failure."""


def _required_file(config: Mapping[str, object], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise Phase9CCError(f"phase9cc.plan.{name}_required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise Phase9CCError(f"phase9cc.plan.{name}_missing:{path}")
    return path


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _base_is_ancestor() -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", PHASE9CC_BASE_SHA, "HEAD"],
            check=False,
        ).returncode
        == 0
    )


def _git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _test_lock() -> dict[str, bool]:
    return {
        "test_inference": False,
        "test_targets_accessed": False,
        "test_metrics_accessed": False,
        "test_unlock": False,
    }


def build_plan(
    config: Mapping[str, object],
    *,
    _bounded_protocol: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind two fixed MLP cells without starting CUDA or training.

    ``_bounded_protocol`` exists only for executable CPU regressions. The
    public production contract always fixes the 9000-update protocol.
    """

    bounded = dict(_bounded_protocol or {})
    updates = int(bounded.get("updates", 9000))
    telemetry_interval = int(bounded.get("telemetry_interval", 100))
    checkpoint_interval = int(bounded.get("checkpoint_interval", 1000))
    milestones = tuple(
        int(value)
        for value in bounded.get("milestones", PHASE9CC_MILESTONES)
    )
    batch_size = int(bounded.get("batch_size", 2))
    if not bounded and (
        updates != 9000
        or telemetry_interval != 100
        or checkpoint_interval != 1000
        or milestones != PHASE9CC_MILESTONES
        or batch_size != 2
    ):
        raise Phase9CCError("phase9cc.plan.production_protocol_invalid")
    if (
        updates <= 0
        or telemetry_interval <= 0
        or checkpoint_interval <= 0
        or updates % telemetry_interval
        or updates % checkpoint_interval
        or not milestones
        or milestones[0] != 0
        or milestones[-1] != updates
        or tuple(sorted(set(milestones))) != milestones
        or any(value < 0 or value > updates for value in milestones)
        or batch_size <= 0
    ):
        raise Phase9CCError("phase9cc.plan.budget_invalid")

    ssl_checkpoint = _required_file(config, "ssl_checkpoint")
    ssl_encoder_export = _required_file(config, "ssl_encoder_export")
    observed_ssl_sha = file_sha256(ssl_checkpoint)
    observed_export_sha = file_sha256(ssl_encoder_export)
    if config.get("ssl_checkpoint_sha256") != observed_ssl_sha:
        raise Phase9CCError("phase9cc.plan.ssl_checkpoint_sha256_mismatch")
    if config.get("ssl_encoder_export_sha256") != observed_export_sha:
        raise Phase9CCError(
            "phase9cc.plan.ssl_encoder_export_sha256_mismatch"
        )
    if config.get("ssl_source_kind") not in PHASE9CB_SOURCE_KINDS:
        raise Phase9CCError("phase9cc.plan.ssl_source_kind_required")
    try:
        _validate_encoder_export(ssl_encoder_export)
    except ValueError as exc:
        raise Phase9CCError(
            "phase9cc.plan.ssl_encoder_export_invalid"
        ) from exc

    required = {
        name: _required_file(config, name)
        for name in (
            "raw_index",
            "target_index",
            "split_manifest",
            "class_weight_artifact",
            "train_priors",
        )
    }
    for name, path in required.items():
        declared = config.get(f"{name}_sha256")
        if declared != file_sha256(path):
            raise Phase9CCError(
                f"phase9cc.plan.{name}_sha256_mismatch"
            )
    try:
        _validate_weight_artifacts(
            required["class_weight_artifact"], required["train_priors"]
        )
    except ValueError as exc:
        raise Phase9CCError(
            "phase9cc.plan.class_weight_binding_invalid"
        ) from exc
    raw_cache_root = Path(str(config.get("raw_cache_root", ""))).resolve()
    target_cache_root = Path(
        str(config.get("target_cache_root", ""))
    ).resolve()
    if not raw_cache_root.is_dir() or not target_cache_root.is_dir():
        raise Phase9CCError("phase9cc.plan.cache_root_missing")
    learning_rate = float(config.get("learning_rate", 0.0003))
    if learning_rate <= 0:
        raise Phase9CCError("phase9cc.plan.learning_rate_invalid")
    observed_git_head = _git_head()
    git_head = str(config.get("git_head") or observed_git_head)
    if (
        not _git_sha(git_head)
        or git_head != observed_git_head
        or not _base_is_ancestor()
    ):
        raise Phase9CCError("phase9cc.plan.git_head_invalid")

    domains = SeedDomains.create(PHASE9CC_SEED)
    data_config = DataConfig(
        name="dilemmadata",
        index_paths=[str(required["raw_index"])],
        cache_roots=[str(raw_cache_root)],
        split_manifest=str(required["split_manifest"]),
        target_cache_index=str(required["target_index"]),
        target_cache_root=str(target_cache_root),
        require_target_sidecars=True,
        batch_size=batch_size,
        workers=0,
        epoch_size=batch_size * updates,
        validation_epoch_size=0,
        mixture_weights={"dilemmadata": 1.0},
    )
    views = build_corpus_data_views(data_config)
    schedule = build_raw_downstream_sample_schedule(
        views.train,
        weights={"dilemmadata": 1.0},
        seed=domains.downstream_data_order,
        first_epoch=0,
        epochs=1,
        steps_per_epoch=updates,
        batch_size=batch_size,
    )
    validation_identities = tuple(
        views.validation.record_identity(index)
        for index in range(len(views.validation))
    )
    selection = fixed_validation_membership(
        validation_identities,
        limit=0,
        seed=domains.downstream_data_order,
    )
    evaluation_membership_fingerprint = fingerprint(
        {
            "split": "validation",
            "split_manifest_fingerprint": (
                views.manifest.manifest_fingerprint
            ),
            "identities": validation_identities,
        }
    )
    bindings = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in required.items()
    }
    bindings.update(
        {
            "raw_cache_root": str(raw_cache_root),
            "target_cache_root": str(target_cache_root),
            "ssl_checkpoint": {
                "path": str(ssl_checkpoint),
                "sha256": observed_ssl_sha,
                "source_kind": config["ssl_source_kind"],
                "encoder_export_path": str(ssl_encoder_export),
                "encoder_export_sha256": observed_export_sha,
            },
        }
    )
    schedule_contract = {
        "epochs": 1,
        "optimizer_steps_per_epoch": updates,
        "required_applied_updates": updates,
        "batch_size": batch_size,
        "epoch_size": batch_size * updates,
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "scheduler": "none",
        "amp": "cuda_float16_fp32_loss_head",
        "telemetry_interval_applied": telemetry_interval,
        "checkpoint_interval_applied": checkpoint_interval,
        "validation_milestones": list(milestones),
        "maximum_consecutive_skips": int(
            bounded.get("maximum_consecutive_skips", 8)
        ),
        "downstream_initialization_seed": (
            domains.downstream_initialization
        ),
        "downstream_data_order_seed": domains.downstream_data_order,
        "sample_schedule_fingerprint": schedule.fingerprint,
        "sample_count": len(schedule.identities),
        "identity_contract_version": "1.2.0",
        "resume_boundary": "applied_update_mid_epoch",
    }
    schedule_contract = {
        **schedule_contract,
        "fingerprint": fingerprint(schedule_contract),
    }
    model_contract = {
        "name": "hierarchical",
        "hidden_dim": int(bounded.get("hidden_dim", 128)),
        "local_gnn_layers": int(bounded.get("local_gnn_layers", 3)),
        "transformer_layers": int(bounded.get("transformer_layers", 2)),
        "attention_heads": int(bounded.get("attention_heads", 4)),
        "ffn_multiplier": int(bounded.get("ffn_multiplier", 4)),
        "dropout": float(bounded.get("dropout", 0.1)),
        "residual": True,
        "decoder": {"kind": "mlp"},
    }
    protocol = {
        "contract_version": PHASE9CC_PROTOCOL_VERSION,
        "phase": "9C-C",
        "hypothesis": "phase9cb_mlp_comparison_may_have_stopped_early",
        "seed": PHASE9CC_SEED,
        "git_head": git_head,
        "phase9cb_base_sha": PHASE9CC_BASE_SHA,
        "cells": list(PHASE9CC_CELLS),
        "tasks": list(PHASE9CC_TASKS),
        "schedule": schedule_contract,
        "model": model_contract,
        "bindings": bindings,
        "validation_membership": {
            "split": "validation",
            "identities": [list(value) for value in selection.identities],
            "membership_fingerprint": selection.membership_fingerprint,
            "evaluation_membership_fingerprint": (
                evaluation_membership_fingerprint
            ),
            "selected_count": selection.selected_count,
        },
        "test_lock": _test_lock(),
        "claim_boundary": "one_seed_descriptive_no_plateau_verdict",
        "bounded_test_protocol": bool(bounded),
    }
    if bounded:
        protocol["bounded_schedule_identities"] = [
            list(value) for value in schedule.identities
        ]
    protocol = {**protocol, "fingerprint": fingerprint(protocol)}
    cells = [
        {
            "cell_id": cell_id,
            "encoder_initialization": (
                "ssl" if cell_id == "ssl_mlp" else "scratch"
            ),
            "decoder_kind": "mlp",
            "transfer_mode": (
                "full_finetune"
                if cell_id == "ssl_mlp"
                else "supervised_scratch"
            ),
            "schedule_fingerprint": schedule.fingerprint,
            "validation_milestones": list(milestones),
            "comparison_checkpoint_policy": "fixed_update_milestones",
        }
        for cell_id in PHASE9CC_CELLS
    ]
    plan = {
        "contract_version": PHASE9CC_PLAN_VERSION,
        "protocol": protocol,
        "cells": cells,
        "production_started": False,
    }
    return {**plan, "fingerprint": fingerprint(plan)}


__all__ = [
    "PHASE9CC_CELLS",
    "PHASE9CC_BASE_SHA",
    "PHASE9CC_MILESTONES",
    "PHASE9CC_PLAN_VERSION",
    "PHASE9CC_PROTOCOL_VERSION",
    "PHASE9CC_SEED",
    "PHASE9CC_TASKS",
    "Phase9CCError",
    "build_plan",
    "canonical_bytes",
    "file_sha256",
    "fingerprint",
]
