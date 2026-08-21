"""Immutable planning contract for the one-seed decoder matrix."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Mapping

import torch

from music_critic.evaluation import validate_dilemmadata_train_priors
from music_critic.experiments.phase8b2.schedule import (
    SeedDomains,
    build_raw_downstream_sample_schedule,
)
from music_critic.models import class_weight_tensors
from music_critic.training.config import DataConfig
from music_critic.training.data import build_corpus_data_views


PHASE9CB_PROTOCOL_VERSION = "1.0.1"
PHASE9CB_PLAN_VERSION = "1.0.1"
PHASE9CB_SEED = 17
PHASE9CB_CELLS = (
    "scratch_mlp",
    "ssl_mlp",
    "scratch_onset_bigru",
    "ssl_onset_bigru",
)
PHASE9CB_SOURCE_KINDS = (
    "phase7a_ssl",
    "phase8b_multilevel_ssl",
    "phase6_hierarchical",
)


class Phase9CBError(ValueError):
    """Stable Phase 9C-B planning, execution, or verification failure."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _required_file(config: Mapping[str, object], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise Phase9CBError(f"phase9cb.plan.{name}_required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise Phase9CBError(f"phase9cb.plan.{name}_missing:{path}")
    return path


def _validate_weight_artifacts(
    class_weight_path: Path, train_prior_path: Path
) -> None:
    try:
        class_weights = json.loads(class_weight_path.read_text(encoding="utf-8"))
        train_priors = json.loads(train_prior_path.read_text(encoding="utf-8"))
        _, evidence = class_weight_tensors(
            class_weights, device=torch.device("cpu")
        )
        validate_dilemmadata_train_priors(train_priors)
    except Exception as exc:
        raise Phase9CBError("phase9cb.plan.class_weight_binding_invalid") from exc
    if (
        evidence.get("policy") != "inverse_sqrt_frequency_supported"
        or evidence.get("train_membership_fingerprint")
        != train_priors.get("train_membership_fingerprint")
    ):
        raise Phase9CBError("phase9cb.plan.class_weight_binding_invalid")


def build_plan(config: Mapping[str, object]) -> dict[str, object]:
    """Bind the four fixed cells without reading target contents or test rows."""

    ssl_checkpoint = _required_file(config, "ssl_checkpoint")
    declared_ssl_sha = config.get("ssl_checkpoint_sha256")
    observed_ssl_sha = file_sha256(ssl_checkpoint)
    source_kind = config.get("ssl_source_kind")
    if declared_ssl_sha != observed_ssl_sha:
        raise Phase9CBError("phase9cb.plan.ssl_checkpoint_sha256_mismatch")
    if source_kind not in PHASE9CB_SOURCE_KINDS:
        raise Phase9CBError("phase9cb.plan.ssl_source_kind_required")
    export_value = config.get("ssl_encoder_export")
    if export_value is None:
        ssl_encoder_export = ssl_checkpoint
        declared_export_sha = observed_ssl_sha
    else:
        ssl_encoder_export = _required_file(config, "ssl_encoder_export")
        declared_export_sha = config.get("ssl_encoder_export_sha256")
    observed_export_sha = file_sha256(ssl_encoder_export)
    if declared_export_sha != observed_export_sha:
        raise Phase9CBError("phase9cb.plan.ssl_encoder_export_sha256_mismatch")
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
    _validate_weight_artifacts(
        required["class_weight_artifact"], required["train_priors"]
    )
    raw_cache_root = Path(str(config.get("raw_cache_root", ""))).resolve()
    target_cache_root = Path(str(config.get("target_cache_root", ""))).resolve()
    if not raw_cache_root.is_dir() or not target_cache_root.is_dir():
        raise Phase9CBError("phase9cb.plan.cache_root_missing")
    integers = {
        "epochs": int(config.get("epochs", 1)),
        "steps_per_epoch": int(config.get("steps_per_epoch", 3000)),
        "batch_size": int(config.get("batch_size", 2)),
    }
    if any(value <= 0 for value in integers.values()):
        raise Phase9CBError("phase9cb.plan.budget_invalid")
    learning_rate = float(config.get("learning_rate", 0.0003))
    if learning_rate <= 0:
        raise Phase9CBError("phase9cb.plan.learning_rate_invalid")
    git_head = str(config.get("git_head") or _git_head())
    if not _git_sha(git_head):
        raise Phase9CBError("phase9cb.plan.git_head_invalid")
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
                "source_kind": source_kind,
                "encoder_export_path": str(ssl_encoder_export),
                "encoder_export_sha256": observed_export_sha,
            },
        }
    )
    domains = SeedDomains.create(PHASE9CB_SEED)
    data_config = DataConfig(
        name="dilemmadata",
        index_paths=[str(required["raw_index"])],
        cache_roots=[str(raw_cache_root)],
        split_manifest=str(required["split_manifest"]),
        target_cache_index=str(required["target_index"]),
        target_cache_root=str(target_cache_root),
        require_target_sidecars=True,
        batch_size=integers["batch_size"],
        epoch_size=(
            integers["batch_size"] * integers["steps_per_epoch"]
        ),
        mixture_weights={"dilemmadata": 1.0},
    )
    train = build_corpus_data_views(data_config).train
    logical_updates = integers["epochs"] * integers["steps_per_epoch"]
    production_schedule = build_raw_downstream_sample_schedule(
        train,
        weights={"dilemmadata": 1.0},
        seed=domains.downstream_data_order,
        first_epoch=0,
        epochs=integers["epochs"],
        steps_per_epoch=integers["steps_per_epoch"],
        batch_size=integers["batch_size"],
    )
    profile_steps = min(3, integers["steps_per_epoch"])
    profile_schedule = build_raw_downstream_sample_schedule(
        train,
        weights={"dilemmadata": 1.0},
        seed=domains.downstream_data_order,
        first_epoch=0,
        epochs=1,
        steps_per_epoch=profile_steps,
        batch_size=integers["batch_size"],
    )
    schedule = {
        "seed": PHASE9CB_SEED,
        "epochs": integers["epochs"],
        "steps_per_epoch": integers["steps_per_epoch"],
        "logical_updates": logical_updates,
        "batch_size": integers["batch_size"],
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "scheduler": "none",
        "amp": "float16",
        "class_weight_formula": "inverse_sqrt_frequency_supported_unchanged",
        "checkpoint_policy": "last_after_equal_applied_updates",
        "validation_protocol": "complete_validation_each_epoch_primary_last_pt",
        "downstream_initialization_seed": domains.downstream_initialization,
        "downstream_data_order_seed": domains.downstream_data_order,
        "sample_schedule_fingerprint": production_schedule.fingerprint,
        "profile_sample_schedule_fingerprint": (
            profile_schedule.fingerprint
        ),
        "sample_count": len(production_schedule.identities),
        "profile_epochs": 1,
        "profile_steps_per_epoch": profile_steps,
        "profile_epoch_size": integers["batch_size"] * profile_steps,
        "profile_sample_count": len(profile_schedule.identities),
        "targets_read_for_schedule": False,
        "target_sidecar_index_validated_for_schedule": True,
    }
    schedule = {**schedule, "fingerprint": fingerprint(schedule)}
    protocol = {
        "contract_version": PHASE9CB_PROTOCOL_VERSION,
        "phase": "9C-B",
        "hypothesis": "independent_mlp_may_hide_ssl_sequence_information",
        "seed": PHASE9CB_SEED,
        "git_head": git_head,
        "cells": list(PHASE9CB_CELLS),
        "schedule": schedule,
        "bindings": bindings,
        "primary_metric": "mean_task_nll_div_log_class_count",
        "test_lock": {
            "test_inference": False,
            "test_targets_accessed": False,
            "test_metrics_accessed": False,
            "test_unlock": False,
        },
        "claim_boundary": "one_seed_descriptive_diagnostic_no_significance",
    }
    protocol = {**protocol, "fingerprint": fingerprint(protocol)}
    cells = []
    for cell_id in PHASE9CB_CELLS:
        cells.append(
            {
                "cell_id": cell_id,
                "encoder_initialization": (
                    "ssl" if cell_id.startswith("ssl_") else "scratch"
                ),
                "decoder_kind": (
                    "onset_bigru"
                    if cell_id.endswith("onset_bigru")
                    else "mlp"
                ),
                "transfer_mode": (
                    "full_finetune"
                    if cell_id.startswith("ssl_")
                    else "supervised_scratch"
                ),
                "schedule_fingerprint": production_schedule.fingerprint,
                "comparison_checkpoint": "last.pt",
            }
        )
    plan = {
        "contract_version": PHASE9CB_PLAN_VERSION,
        "protocol": protocol,
        "cells": cells,
        "production_started": False,
    }
    return {**plan, "fingerprint": fingerprint(plan)}


__all__ = [
    "PHASE9CB_CELLS",
    "PHASE9CB_PLAN_VERSION",
    "PHASE9CB_PROTOCOL_VERSION",
    "PHASE9CB_SEED",
    "Phase9CBError",
    "build_plan",
    "canonical_bytes",
    "file_sha256",
    "fingerprint",
]
