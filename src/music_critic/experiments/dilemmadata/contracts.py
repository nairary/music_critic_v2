"""Closed RTX 3090 plan for Phase 9B.2B; this module never launches training."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping

from music_critic.models import DILEMMADATA_ACTIVE_TASK_IDS


DILEMMADATA_EXPERIMENT_PLAN_VERSION = "1.0.0"
DILEMMADATA_COMMAND_MATRIX_VERSION = "1.0.0"
DILEMMADATA_REPORT_BUNDLE_VERSION = "1.0.0"
DILEMMADATA_SEEDS = (17, 29, 43)
DILEMMADATA_PRIMARY_VARIANTS = (
    "supervised_scratch",
    "ssl_hook_pop",
    "ssl_hook_pop_dilemmadata",
)
DILEMMADATA_OPTIONAL_VARIANTS = (
    "ssl_multilevel_equal_hook_pop_dilemmadata",
)


class DilemmadataExperimentPlanError(ValueError):
    """Stable immutable-plan validation failure."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_dilemmadata_experiment_plan(
    *,
    raw_index_fingerprint: str,
    target_cache_index_fingerprint: str,
    split_manifest_fingerprint: str,
    sample_schedule_fingerprint: str,
    phase7a_encoder_export_path: str,
    phase7a_encoder_export_sha256: str,
    phase7a_source_checkpoint_sha256: str,
    phase8b_encoder_export_path: str,
    phase8b_encoder_export_sha256: str,
    phase8b_source_checkpoint_sha256: str,
    optional_equal_encoder_export_path: str | None = None,
    optional_equal_encoder_export_sha256: str | None = None,
    optional_equal_source_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    bindings = {
        "raw_index_fingerprint": raw_index_fingerprint,
        "target_cache_index_fingerprint": target_cache_index_fingerprint,
        "split_manifest_fingerprint": split_manifest_fingerprint,
        "sample_schedule_fingerprint": sample_schedule_fingerprint,
    }
    if not all(_sha(value) for value in bindings.values()):
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.binding_invalid"
        )
    variants: list[dict[str, object]] = [
        {
            "variant_id": "supervised_scratch",
            "primary": True,
            "transfer_mode": "supervised_scratch",
            "source_kind": None,
            "encoder_export_path": None,
            "encoder_export_sha256": None,
            "source_checkpoint_sha256": None,
        },
        {
            "variant_id": "ssl_hook_pop",
            "primary": True,
            "transfer_mode": "full_finetune",
            "source_kind": "phase7a_ssl",
            "encoder_export_path": phase7a_encoder_export_path,
            "encoder_export_sha256": phase7a_encoder_export_sha256,
            "source_checkpoint_sha256": phase7a_source_checkpoint_sha256,
        },
        {
            "variant_id": "ssl_hook_pop_dilemmadata",
            "primary": True,
            "transfer_mode": "full_finetune",
            "source_kind": "phase8b_multilevel_ssl",
            "encoder_export_path": phase8b_encoder_export_path,
            "encoder_export_sha256": phase8b_encoder_export_sha256,
            "source_checkpoint_sha256": phase8b_source_checkpoint_sha256,
        },
    ]
    if len(
        {
            optional_equal_encoder_export_path is None,
            optional_equal_encoder_export_sha256 is None,
            optional_equal_source_checkpoint_sha256 is None,
        }
    ) != 1:
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.optional_export_incomplete"
        )
    if optional_equal_encoder_export_path is not None:
        variants.append(
            {
                "variant_id": "ssl_multilevel_equal_hook_pop_dilemmadata",
                "primary": False,
                "transfer_mode": "full_finetune",
                "source_kind": "phase8b_multilevel_ssl",
                "encoder_export_path": optional_equal_encoder_export_path,
                "encoder_export_sha256": optional_equal_encoder_export_sha256,
                "source_checkpoint_sha256": (
                    optional_equal_source_checkpoint_sha256
                ),
            }
        )
    plan = {
        "contract_version": DILEMMADATA_EXPERIMENT_PLAN_VERSION,
        "phase": "9B.2B",
        "execution_state": "planned_not_executed",
        "hardware": {
            "accelerator": "NVIDIA GeForce RTX 3090",
            "minimum_vram_gib": 24,
            "device": "cuda",
            "amp": True,
        },
        "bindings": bindings,
        "seeds": list(DILEMMADATA_SEEDS),
        "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
        "excluded_supervision": {
            "positive_unlabeled_tasks": "no_ce_head_or_loss",
            "open_string_cpu_tasks": "no_vocab_hash_head_or_loss",
        },
        "variants": variants,
        "fixed_training": {
            "model": "hierarchical",
            "data": "dilemmadata",
            "experiment": "dilemmadata_scratch_vs_ssl",
            "objective": "four_task_fixed_equal_weight_sum",
            "class_weight_policy": "unweighted",
            "reconstruction_weight": 0.0,
            "optimizer": "adamw",
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "epochs": 20,
            "logical_updates": 2000,
            "optimizer_steps_per_epoch": 100,
            "batch_size": 3,
            "selection_split": "validation",
            "test_split_locked": True,
        },
        "comparison": {
            "unit": "connected_component",
            "paired_bootstrap_replicates": 2000,
            "primary_metric": "mean_four_task_source_entry_nll",
            "secondary_metrics": [
                "top1_accuracy",
                "macro_f1",
                "balanced_accuracy",
            ],
            "equal_sample_schedule": True,
            "equal_optimizer_budget": True,
            "fresh_heads_and_optimizer_per_cell": True,
        },
    }
    result = {**plan, "plan_fingerprint": _fingerprint(plan)}
    validate_dilemmadata_experiment_plan(result)
    return result


def validate_dilemmadata_experiment_plan(
    plan: Mapping[str, object],
) -> None:
    payload = dict(plan)
    fingerprint = payload.pop("plan_fingerprint", None)
    variants = payload.get("variants")
    if (
        fingerprint != _fingerprint(payload)
        or payload.get("contract_version")
        != DILEMMADATA_EXPERIMENT_PLAN_VERSION
        or payload.get("phase") != "9B.2B"
        or payload.get("execution_state") != "planned_not_executed"
        or payload.get("seeds") != list(DILEMMADATA_SEEDS)
        or payload.get("active_task_ids") != list(DILEMMADATA_ACTIVE_TASK_IDS)
        or not isinstance(variants, list)
        or tuple(
            row.get("variant_id")
            for row in variants
            if row.get("primary") is True
        )
        != DILEMMADATA_PRIMARY_VARIANTS
    ):
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.plan_invalid"
        )
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or not all(
        _sha(value) for value in bindings.values()
    ):
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.binding_invalid"
        )
    for row in variants:
        if row["variant_id"] == "supervised_scratch":
            if row["encoder_export_path"] is not None or row[
                "encoder_export_sha256"
            ] is not None or row["source_checkpoint_sha256"] is not None:
                raise DilemmadataExperimentPlanError(
                    "dilemmadata.experiment.scratch_export_forbidden"
                )
        elif (
            not isinstance(row["encoder_export_path"], str)
            or not _sha(row["encoder_export_sha256"])
            or not _sha(row["source_checkpoint_sha256"])
        ):
            raise DilemmadataExperimentPlanError(
                "dilemmadata.experiment.encoder_export_invalid"
            )


def dilemmadata_command_matrix(
    plan: Mapping[str, object],
) -> dict[str, object]:
    validate_dilemmadata_experiment_plan(plan)
    commands = []
    for variant in plan["variants"]:
        for seed in DILEMMADATA_SEEDS:
            overrides = [
                "python",
                "-m",
                "music_critic.training.run",
                "experiment=dilemmadata_scratch_vs_ssl",
                "model=hierarchical",
                "data=dilemmadata",
                "device=cuda",
                f"seed={seed}",
                f"transfer.mode={variant['transfer_mode']}",
                f"transfer.source_kind={variant['source_kind'] or 'phase7a_ssl'}",
                f"transfer.comparison_protocol_fingerprint={plan['plan_fingerprint']}",
                f"transfer.sample_schedule_fingerprint={plan['bindings']['sample_schedule_fingerprint']}",
                "transfer.logical_updates=2000",
                "experiment.steps=2000",
                "experiment.optimizer_steps_per_epoch=100",
                f"transfer.downstream_initialization_seed={seed}",
                f"transfer.downstream_data_order_seed={seed}",
                f"output_dir=outputs/phase9b2b/{variant['variant_id']}/seed-{seed}",
            ]
            if variant["encoder_export_path"] is not None:
                overrides.extend(
                    (
                        f"transfer.encoder_export_path={variant['encoder_export_path']}",
                        f"transfer.encoder_export_sha256={variant['encoder_export_sha256']}",
                        f"transfer.source_ssl_checkpoint_sha256={variant['source_checkpoint_sha256']}",
                    )
                )
            commands.append(
                {
                    "variant_id": variant["variant_id"],
                    "seed": seed,
                    "primary": variant["primary"],
                    "argv": overrides,
                    "execution_state": "not_started",
                }
            )
    matrix = {
        "contract_version": DILEMMADATA_COMMAND_MATRIX_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "long_training_executed": False,
        "commands": commands,
    }
    return {**matrix, "matrix_fingerprint": _fingerprint(matrix)}


def validate_dilemmadata_command_matrix(
    plan: Mapping[str, object], matrix: Mapping[str, object]
) -> None:
    validate_dilemmadata_experiment_plan(plan)
    payload = dict(matrix)
    fingerprint = payload.pop("matrix_fingerprint", None)
    if (
        fingerprint != _fingerprint(payload)
        or dict(matrix) != dilemmadata_command_matrix(plan)
    ):
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.command_matrix_invalid"
        )


def dilemmadata_report_bundle_manifest(
    plan: Mapping[str, object], matrix: Mapping[str, object]
) -> dict[str, object]:
    validate_dilemmadata_command_matrix(plan, matrix)
    manifest = {
        "contract_version": DILEMMADATA_REPORT_BUNDLE_VERSION,
        "phase": "9B.2B",
        "execution_state": "planned_not_executed",
        "long_training_executed": False,
        "artifacts": {
            "plan.json": plan["plan_fingerprint"],
            "command_matrix.json": matrix["matrix_fingerprint"],
        },
    }
    return {**manifest, "manifest_fingerprint": _fingerprint(manifest)}


def verify_dilemmadata_report_bundle(
    plan: Mapping[str, object],
    matrix: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    expected = dilemmadata_report_bundle_manifest(plan, matrix)
    if dict(manifest) != expected:
        raise DilemmadataExperimentPlanError(
            "dilemmadata.experiment.report_bundle_invalid"
        )


__all__ = [
    "DILEMMADATA_COMMAND_MATRIX_VERSION",
    "DILEMMADATA_EXPERIMENT_PLAN_VERSION",
    "DILEMMADATA_OPTIONAL_VARIANTS",
    "DILEMMADATA_PRIMARY_VARIANTS",
    "DILEMMADATA_REPORT_BUNDLE_VERSION",
    "DILEMMADATA_SEEDS",
    "DilemmadataExperimentPlanError",
    "build_dilemmadata_experiment_plan",
    "dilemmadata_command_matrix",
    "dilemmadata_report_bundle_manifest",
    "validate_dilemmadata_command_matrix",
    "validate_dilemmadata_experiment_plan",
    "verify_dilemmadata_report_bundle",
]
