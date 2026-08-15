"""Official Phase 8B.2A matrix planner and engine configuration adapter."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from music_critic.models import ACTIVE_TASK_IDS

from music_critic.experiments.phase8b2.artifacts import (
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    read_json,
)
from music_critic.experiments.phase8b2.attestation import (
    attest_data_binding,
    resolve_actual_downstream_schedule,
    resolve_actual_ssl_schedule,
)
from music_critic.experiments.phase8b2.config import Phase8B2Config
from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_ARTIFACT_CONTRACT_VERSION,
    PHASE8B2_COMPARISON_PROTOCOL_VERSION,
    ComparisonProtocol,
    ComputeBudget,
    DataBinding,
    Phase8B2ContractError,
    default_selection_rule,
    downstream_task_manifest,
    fingerprint,
    locked_test_state,
)
from music_critic.experiments.phase8b2.schedule import (
    SeedDomains,
    VariantSchedule,
    build_variant_schedule,
    validate_paired_schedules,
)
PLAN_CONTRACT_VERSION = "1.1.0"


def _plain(config: object) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, dict):
        value = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    else:
        raise Phase8B2ContractError("phase8b2.config.type_invalid")
    if not isinstance(value, dict):
        raise Phase8B2ContractError("phase8b2.config.root_invalid")
    value.pop("defaults", None)
    return value


def _protocol(
    config: Mapping[str, Any],
    data: DataBinding,
    *,
    ssl_schedule_fingerprints: Mapping[int, str],
    downstream_schedule_fingerprints: Mapping[int, str],
) -> ComparisonProtocol:
    comparison = config["comparison"]
    compute = ComputeBudget(
        batch_size=int(config["data"]["batch_size"]),
        optimizer_steps=int(comparison["ssl_optimizer_steps"]),
        raw_sample_exposures=(
            int(config["data"]["batch_size"])
            * int(comparison["ssl_optimizer_steps"])
        ),
        encoder_forwards_per_update=(
            int(comparison["matched_encoder_forwards_per_update"])
            if comparison["comparison_mode"] == "encoder_forward_matched"
            else None
        ),
    )
    downstream_schedule = fingerprint(
        {
            "kind": "paired_downstream_schedule",
            "per_seed": [
                [seed, downstream_schedule_fingerprints[seed]]
                for seed in sorted(downstream_schedule_fingerprints)
            ],
        }
    )
    return ComparisonProtocol(
        protocol_version=PHASE8B2_COMPARISON_PROTOCOL_VERSION,
        comparison_mode=comparison["comparison_mode"],
        variants=tuple(sorted(comparison["variants"])),
        seeds=tuple(sorted(comparison["seeds"])),
        encoder_model_config=dict(config["model"]),
        ssl_objective_config=dict(config["ssl"]),
        masking_policy_config={
            "source_contract": "Phase8AMaskPolicy@1.0.0",
            "matched_encoder_forwards_per_update": comparison[
                "matched_encoder_forwards_per_update"
            ],
            "loss_renormalization": "forbidden",
        },
        data=data,
        compute=compute,
        optimizer_config=dict(config["optimizer"]),
        scheduler_config=dict(config["scheduler"]),
        amp_device_config=dict(config["device"]),
        transfer_modes=tuple(comparison["transfer_modes"]),
        downstream_tasks=downstream_task_manifest(
            tuple(config["downstream_task_ids"])
        ),
        validation_selection_rule=default_selection_rule(),
        test_unlock_state=locked_test_state(),
        downstream_optimizer_steps=int(
            comparison["downstream_optimizer_steps"]
        ),
        downstream_schedule_fingerprint=downstream_schedule,
        ssl_sample_schedule_fingerprints=tuple(
            sorted(ssl_schedule_fingerprints.items())
        ),
        downstream_sample_schedule_fingerprints=tuple(
            sorted(downstream_schedule_fingerprints.items())
        ),
        runtime_execution_config={
            "optimizer_steps_per_epoch": int(
                comparison["optimizer_steps_per_epoch"]
            ),
            "validation_interval_epochs": int(
                comparison["validation_interval_epochs"]
            ),
            "validation_samples": int(comparison["validation_samples"]),
            "fixed_validation_seed": int(
                comparison["fixed_validation_seed"]
            ),
            "bootstrap_replicates": int(
                comparison["bootstrap_replicates"]
            ),
            "ssl_attempted_logical_updates": int(
                comparison["ssl_optimizer_steps"]
            ),
            "downstream_attempted_logical_updates": int(
                comparison["downstream_optimizer_steps"]
            ),
        },
    )


def build_experiment_plan(config: object) -> dict[str, object]:
    """Build the complete deterministic matrix without training or writes."""

    plain = _plain(config)
    comparison = plain["comparison"]
    if comparison["architecture"] != "hierarchical":
        raise Phase8B2ContractError(
            "phase8b2.config.primary_architecture_must_be_hierarchical"
        )
    if comparison["name"] != "bounded_acceptance" and len(
        comparison["seeds"]
    ) < int(comparison["minimum_production_seeds"]):
        raise Phase8B2ContractError(
            "phase8b2.config.production_seed_minimum_not_met"
        )
    if plain["acknowledge_test_evaluation"]:
        raise Phase8B2ContractError(
            "phase8b2.plan.test_unlock_forbidden"
        )
    _validate_runtime_support(plain)
    data, data_attestation = attest_data_binding(
        plain["data"],
        validation_samples=int(comparison["validation_samples"]),
        validation_seed=int(comparison["fixed_validation_seed"]),
    )
    actual_ssl_schedules = {
        int(seed): resolve_actual_ssl_schedule(
            plain["data"],
            seed=int(seed),
            logical_updates=int(comparison["ssl_optimizer_steps"]),
            optimizer_steps_per_epoch=int(
                comparison["optimizer_steps_per_epoch"]
            ),
            validation_samples=int(comparison["validation_samples"]),
            validation_seed=int(comparison["fixed_validation_seed"]),
        )
        for seed in sorted(comparison["seeds"])
    }
    actual_downstream_schedules = {
        int(seed): resolve_actual_downstream_schedule(
            plain["data"],
            seed=int(seed),
            logical_updates=int(comparison["downstream_optimizer_steps"]),
            optimizer_steps_per_epoch=int(
                comparison["optimizer_steps_per_epoch"]
            ),
            validation_samples=int(comparison["validation_samples"]),
            validation_seed=int(comparison["fixed_validation_seed"]),
        )
        for seed in sorted(comparison["seeds"])
    }
    protocol = _protocol(
        plain,
        data,
        ssl_schedule_fingerprints={
            seed: str(row["sample_schedule_fingerprint"])
            for seed, row in actual_ssl_schedules.items()
        },
        downstream_schedule_fingerprints={
            seed: str(row["sample_schedule_fingerprint"])
            for seed, row in actual_downstream_schedules.items()
        },
    )
    schedules: dict[int, tuple[VariantSchedule, ...]] = {}
    ssl_cells = []
    downstream_cells = []
    for seed in protocol.seeds:
        domains = SeedDomains.create(seed)
        identities = tuple(
            (str(row["dataset_id"]), str(row["piece_id"]))
            for row in actual_ssl_schedules[seed]["slots"]
        )
        seed_schedules = tuple(
            build_variant_schedule(
                variant_id,
                comparison_mode=protocol.comparison_mode,
                logical_updates=protocol.compute.optimizer_steps,
                batch_size=protocol.compute.batch_size,
                matched_encoder_forwards_per_update=(
                    int(comparison["matched_encoder_forwards_per_update"])
                ),
                sample_identity_schedule=identities,
                mask_seed=domains.ssl_mask_planning,
            )
            for variant_id in protocol.variants
        )
        paired = validate_paired_schedules(seed_schedules)
        schedules[seed] = seed_schedules
        for schedule in seed_schedules:
            ssl_cell_id = (
                f"ssl/{protocol.comparison_mode}/{seed}/{schedule.variant_id}"
            )
            ssl_cells.append(
                {
                    "cell_id": ssl_cell_id,
                    "seed": seed,
                    "seed_domains": domains.to_dict(),
                    "variant_id": schedule.variant_id,
                    "initial_encoder_pairing_fingerprint": fingerprint(
                        {
                            "model_initialization_seed": (
                                domains.model_initialization
                            ),
                            "encoder_model_config": dict(
                                protocol.encoder_model_config
                            ),
                        }
                    ),
                    "schedule": schedule.to_dict(),
                    "actual_schedule_fingerprint": actual_ssl_schedules[
                        seed
                    ]["fingerprint"],
                    "paired_schedule_evidence": paired,
                }
            )
            for transfer_mode in protocol.transfer_modes:
                if transfer_mode == "supervised_scratch":
                    continue
                downstream_cells.append(
                    {
                        "cell_id": (
                            f"downstream/{seed}/{schedule.variant_id}/"
                            f"{transfer_mode}"
                        ),
                        "seed": seed,
                        "variant_id": schedule.variant_id,
                        "transfer_mode": transfer_mode,
                        "ssl_cell_id": ssl_cell_id,
                        "downstream_schedule_fingerprint": (
                            actual_downstream_schedules[seed][
                                "sample_schedule_fingerprint"
                            ]
                        ),
                        "evaluation_seed": domains.downstream_data_order,
                    }
                )
        downstream_cells.append(
            {
                "cell_id": f"downstream/{seed}/supervised_scratch/supervised_scratch",
                "seed": seed,
                "variant_id": "supervised_scratch",
                "transfer_mode": "supervised_scratch",
                "ssl_cell_id": None,
                "downstream_schedule_fingerprint": (
                    actual_downstream_schedules[seed][
                        "sample_schedule_fingerprint"
                    ]
                ),
                "evaluation_seed": domains.downstream_data_order,
            }
        )
    evaluation_cells = [
        {
            "cell_id": "evaluation/" + row["cell_id"].removeprefix(
                "downstream/"
            ),
            "downstream_cell_id": row["cell_id"],
            "seed": row["seed"],
            "variant_id": row["variant_id"],
            "transfer_mode": row["transfer_mode"],
            "evaluation_seed": row["evaluation_seed"],
        }
        for row in downstream_cells
    ]
    encoder_export_cells = [
        {
            "cell_id": "encoder_export/" + row["cell_id"].removeprefix(
                "ssl/"
            ),
            "ssl_cell_id": row["cell_id"],
            "seed": row["seed"],
            "variant_id": row["variant_id"],
        }
        for row in ssl_cells
    ]
    ssl_post_training_validation_passes = (
        1
        + (int(comparison["ssl_optimizer_steps"]) - 1)
        // (
            int(comparison["optimizer_steps_per_epoch"])
            * int(comparison["validation_interval_epochs"])
        )
    )
    downstream_validation_passes = (
        1
        + (int(comparison["downstream_optimizer_steps"]) - 1)
        // (
            int(comparison["optimizer_steps_per_epoch"])
            * int(comparison["validation_interval_epochs"])
        )
    )
    plan = {
        "plan_contract_version": PLAN_CONTRACT_VERSION,
        "dry_run": True,
        "training_performed": False,
        "test_accessed": False,
        "protocol": protocol.to_dict(),
        "data_attestation": data_attestation,
        "actual_sample_schedule": {
            "contract_version": "1.1.0",
            "protocol_fingerprint": protocol.fingerprint,
            "ssl": [
                actual_ssl_schedules[seed]
                for seed in sorted(actual_ssl_schedules)
            ],
            "downstream": [
                actual_downstream_schedules[seed]
                for seed in sorted(actual_downstream_schedules)
            ],
        },
        "ssl_cells": sorted(ssl_cells, key=lambda row: row["cell_id"]),
        "encoder_export_cells": sorted(
            encoder_export_cells, key=lambda row: row["cell_id"]
        ),
        "downstream_cells": sorted(
            downstream_cells, key=lambda row: row["cell_id"]
        ),
        "evaluation_cells": sorted(
            evaluation_cells, key=lambda row: row["cell_id"]
        ),
        "artifact_schema": {
            "contract_version": PHASE8B2_ARTIFACT_CONTRACT_VERSION,
            "required": list(REQUIRED_ARTIFACTS),
            "optional": list(OPTIONAL_ARTIFACTS),
        },
        "runtime_paths": {
            "index_paths": list(plain["data"]["index_paths"]),
            "cache_roots": list(plain["data"]["cache_roots"]),
            "split_manifest": plain["data"]["split_manifest"],
        },
        "production_read_only_smoke": production_smoke_status(plain),
        "summary": {
            "ssl_cell_count": len(ssl_cells),
            "encoder_export_cell_count": len(encoder_export_cells),
            "downstream_cell_count": len(downstream_cells),
            "evaluation_cell_count": len(evaluation_cells),
            "ssl_raw_sample_budget_per_seed": (
                int(comparison["ssl_optimizer_steps"])
                * int(plain["data"]["batch_size"])
            ),
            "ssl_encoder_forward_budget_per_cell": (
                None
                if protocol.compute.encoder_forward_count is None
                else protocol.compute.encoder_forward_count
            ),
            "estimated_ssl_validation_passes_per_cell": (
                1 + ssl_post_training_validation_passes
            ),
            "estimated_downstream_validation_passes_per_cell": (
                downstream_validation_passes
            ),
            "estimated_validation_passes_total": (
                len(ssl_cells)
                * (1 + ssl_post_training_validation_passes)
                + len(downstream_cells) * downstream_validation_passes
                + len(evaluation_cells)
            ),
            "output_root": str(Path(plain["output_root"]).resolve()),
        },
        "claims": {
            "bounded_acceptance_is_scientific_superiority_evidence": False,
            "pdmx_evidence": False,
            "production_training_performed": False,
        },
    }
    plan["fingerprint"] = fingerprint(plan)
    return plan


def _validate_runtime_support(config: Mapping[str, Any]) -> None:
    """Reject protocol fields the official engines cannot execute exactly."""

    comparison = config["comparison"]
    data = config["data"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    device = config["device"]
    if optimizer["name"] != "adamw":
        raise Phase8B2ContractError(
            "phase8b2.runtime.optimizer_unsupported"
        )
    if scheduler["name"] not in {"none", "cosine"}:
        raise Phase8B2ContractError(
            "phase8b2.runtime.scheduler_unsupported"
        )
    if scheduler["name"] == "none" and float(
        scheduler["minimum_learning_rate"]
    ) != 0.0:
        raise Phase8B2ContractError(
            "phase8b2.runtime.scheduler_minimum_lr_unsupported_for_none"
        )
    if device["name"] not in {"cpu", "auto", "cuda"} and not str(
        device["name"]
    ).startswith("cuda:"):
        raise Phase8B2ContractError(
            "phase8b2.runtime.device_unsupported"
        )
    if device["amp_dtype"] not in {"float16", "bfloat16"}:
        raise Phase8B2ContractError(
            "phase8b2.runtime.amp_dtype_unsupported"
        )
    if bool(device["amp"]) and device["name"] == "cpu":
        raise Phase8B2ContractError(
            "phase8b2.runtime.amp_requires_cuda"
        )
    for name, minimum in (
        ("batch_size", 1),
        ("workers", 0),
    ):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise Phase8B2ContractError(
                f"phase8b2.runtime.{name}_invalid"
            )
    for name in (
        "ssl_optimizer_steps",
        "downstream_optimizer_steps",
        "optimizer_steps_per_epoch",
        "validation_interval_epochs",
    ):
        value = comparison[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Phase8B2ContractError(
                f"phase8b2.runtime.{name}_invalid"
            )
    if int(comparison["validation_samples"]) < 0:
        raise Phase8B2ContractError(
            "phase8b2.runtime.validation_samples_invalid"
        )
    if config["ssl"]["epsilon"] != 1e-8:
        raise Phase8B2ContractError(
            "phase8b2.runtime.ssl_epsilon_unsupported"
        )
    task_ids = config["downstream_task_ids"]
    if (
        not any(str(task_id).startswith("theory.") for task_id in task_ids)
        or not any(
            str(task_id).startswith("pop909_cl.") for task_id in task_ids
        )
    ):
        raise Phase8B2ContractError(
            "phase8b2.runtime.task_subset_primary_datasets_incomplete"
        )


def official_ssl_cell_overrides(
    plan: Mapping[str, object],
    cell_id: str,
    output_directory: str,
    *,
    actual_sample_schedule_path: str = "",
) -> list[str]:
    """Translate a plan cell into the official ``music_critic.ssl.run`` API."""

    cells = {
        row["cell_id"]: row
        for row in plan["ssl_cells"]
        if isinstance(row, dict)
    }
    if cell_id not in cells:
        raise Phase8B2ContractError("phase8b2.runner.ssl_cell_unknown")
    cell = cells[cell_id]
    schedule = cell["schedule"]
    protocol = plan["protocol"]
    views = schedule["policy_views"]
    objective_mode = schedule["objective_mode"]
    masking_mode = schedule["masking_mode"]
    runtime = protocol["runtime_execution_config"]
    steps_per_epoch = min(
        int(runtime["optimizer_steps_per_epoch"]),
        int(schedule["logical_updates"]),
    )
    epochs = (
        int(schedule["logical_updates"]) + steps_per_epoch - 1
    ) // steps_per_epoch
    device_name = str(protocol["amp_device_config"]["name"])
    device_group = "cuda" if device_name.startswith("cuda") else device_name
    overrides = [
        f"+phase8b_objective={objective_mode}",
        f"+phase8b_masking={masking_mode}",
        "+phase8b2_schedule=comparison",
        "experiment=pretrain",
        f"experiment.steps={schedule['logical_updates']}",
        f"experiment.epochs={epochs}",
        f"experiment.optimizer_steps_per_epoch={steps_per_epoch}",
        "experiment.validation_interval="
        f"{runtime['validation_interval_epochs']}",
        f"data.batch_size={schedule['batch_size']}",
        f"data.epoch_size={schedule['batch_size'] * steps_per_epoch}",
        "data.validation_epoch_size="
        f"{runtime['validation_samples']}",
        f"data.validation_seed={runtime['fixed_validation_seed']}",
        f"data.workers={protocol['data']['workers']}",
        "data.mixture_weights={"
        + ",".join(
            f"{dataset_id}:{weight}"
            for dataset_id, weight in protocol["data"]["mixture_weights"]
        )
        + "}",
        f"model.name={protocol['encoder_model_config']['name']}",
        f"model.hidden_dim={protocol['encoder_model_config']['hidden_dim']}",
        "model.local_gnn_layers="
        f"{protocol['encoder_model_config']['local_gnn_layers']}",
        "model.transformer_layers="
        f"{protocol['encoder_model_config']['transformer_layers']}",
        "model.attention_heads="
        f"{protocol['encoder_model_config']['attention_heads']}",
        "model.ffn_multiplier="
        f"{protocol['encoder_model_config']['ffn_multiplier']}",
        f"model.dropout={protocol['encoder_model_config']['dropout']}",
        "model.residual="
        f"{str(protocol['encoder_model_config']['residual']).lower()}",
        f"optimizer={protocol['optimizer_config']['name']}",
        f"optimizer.learning_rate={protocol['optimizer_config']['learning_rate']}",
        f"optimizer.weight_decay={protocol['optimizer_config']['weight_decay']}",
        "optimizer.gradient_clip_norm="
        f"{protocol['optimizer_config']['gradient_clip_norm']}",
        f"scheduler={protocol['scheduler_config']['name']}",
        "scheduler.minimum_learning_rate="
        f"{protocol['scheduler_config']['minimum_learning_rate']}",
        f"device={device_group}",
        f"device.name={device_name}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
        "device.amp_dtype="
        f"{protocol['amp_device_config']['amp_dtype']}",
        "device.non_blocking="
        f"{str(protocol['amp_device_config']['non_blocking']).lower()}",
        f"seed={cell['seed']}",
        f"ssl.mask_rate={protocol['ssl_objective_config']['mask_rate']}",
        "ssl.decoder_views="
        f"{protocol['ssl_objective_config']['decoder_views']}",
        "ssl.decoder_remask_prob="
        f"{protocol['ssl_objective_config']['decoder_remask_prob']}",
        f"ssl.note_weight={protocol['ssl_objective_config']['note_weight']}",
        f"ssl.bar_weight={protocol['ssl_objective_config']['bar_weight']}",
        f"ssl.song_weight={protocol['ssl_objective_config']['song_weight']}",
        "ssl.epsilon="
        f"{protocol['ssl_objective_config']['epsilon']}",
        "ssl.projector_hidden_dim="
        f"{protocol['ssl_objective_config']['projector_hidden_dim']}",
        "ssl.decoder_hidden_dim="
        f"{protocol['ssl_objective_config']['decoder_hidden_dim']}",
        f"output_dir={output_directory}",
        "phase8b2_schedule.contract_version=1.1.0",
        "phase8b2_schedule.comparison_mode="
        f"{schedule['comparison_mode']}",
        f"phase8b2_schedule.variant_id={schedule['variant_id']}",
        "phase8b2_schedule.protocol_fingerprint="
        f"{protocol['fingerprint']}",
        "phase8b2_schedule.sample_schedule_fingerprint="
        f"{schedule['sample_schedule_fingerprint']}",
        "phase8b2_schedule.model_initialization_seed="
        f"{cell['seed_domains']['model_initialization']}",
        "phase8b2_schedule.data_order_seed="
        f"{cell['seed_domains']['ssl_data_order']}",
        f"phase8b2_schedule.logical_updates={schedule['logical_updates']}",
        "phase8b2_schedule.policy_view_names=["
        + ",".join(view["policy"] for view in views)
        + "]",
        "phase8b2_schedule.policy_view_seeds=["
        + ",".join(str(view["seed"]) for view in views)
        + "]",
    ]
    if actual_sample_schedule_path:
        overrides.append(
            "phase8b2_schedule.actual_sample_schedule_path="
            f"{actual_sample_schedule_path}"
        )
    paths = plan["runtime_paths"]
    if paths["index_paths"]:
        overrides.extend(
            (
                "data=mixed",
                "data.index_paths=["
                + ",".join(paths["index_paths"])
                + "]",
                "data.cache_roots=["
                + ",".join(paths["cache_roots"])
                + "]",
                f"data.split_manifest={paths['split_manifest']}",
            )
        )
    return overrides


def production_smoke_status(config: Mapping[str, Any]) -> dict[str, object]:
    """Check only explicitly named paths; never scan or rebuild a cache."""

    data = config["data"]
    paths = [
        *data["index_paths"],
        *data["cache_roots"],
        *([data["split_manifest"]] if data["split_manifest"] else []),
    ]
    if not paths:
        return {
            "status": "skipped",
            "reason": "production_paths_not_configured",
            "cache_writes": False,
            "test_split_access": False,
        }
    missing = sorted(path for path in paths if not Path(path).exists())
    if missing:
        return {
            "status": "skipped",
            "reason": "configured_production_paths_missing",
            "missing_paths": missing,
            "cache_writes": False,
            "test_split_access": False,
        }
    return {
        "status": "available_not_executed_in_plan",
        "maximum_pieces_per_dataset_split": 3,
        "splits": ["train", "validation"],
        "cache_writes": False,
        "test_split_access": False,
        "directory_scans": False,
    }


def official_downstream_overrides(
    plan: Mapping[str, object],
    cell_id: str,
    output_directory: str,
    *,
    encoder_export_path: str = "",
    encoder_export_sha256: str = "",
    source_ssl_checkpoint_sha256: str = "",
    actual_sample_schedule_path: str = "",
) -> list[str]:
    """Translate one transfer cell into the official training engine API."""

    cells = {
        row["cell_id"]: row
        for row in plan["downstream_cells"]
        if isinstance(row, dict)
    }
    if cell_id not in cells:
        raise Phase8B2ContractError(
            "phase8b2.runner.downstream_cell_unknown"
        )
    cell = cells[cell_id]
    protocol = plan["protocol"]
    mode = cell["transfer_mode"]
    pretrained = mode != "supervised_scratch"
    if pretrained and not all(
        (
            encoder_export_path,
            encoder_export_sha256,
            source_ssl_checkpoint_sha256,
        )
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.transfer_source_binding_incomplete"
        )
    if not pretrained and any(
        (
            encoder_export_path,
            encoder_export_sha256,
            source_ssl_checkpoint_sha256,
        )
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.scratch_transfer_source_forbidden"
        )
    domains = SeedDomains.create(int(cell["seed"]))
    model = protocol["encoder_model_config"]
    optimizer = protocol["optimizer_config"]
    batch_size = int(protocol["compute"]["batch_size"])
    updates = int(protocol["downstream_optimizer_steps"])
    runtime = protocol["runtime_execution_config"]
    steps_per_epoch = min(
        int(runtime["optimizer_steps_per_epoch"]), updates
    )
    epochs = (updates + steps_per_epoch - 1) // steps_per_epoch
    device_name = str(protocol["amp_device_config"]["name"])
    device_group = "cuda" if device_name.startswith("cuda") else device_name
    selected_tasks = [row["task_id"] for row in protocol["downstream_tasks"]]
    task_weights = {
        task_id: float(task_id in set(selected_tasks))
        for task_id in ACTIVE_TASK_IDS
    }
    overrides = [
        "experiment=smoke",
        f"experiment.steps={updates}",
        f"experiment.epochs={epochs}",
        f"experiment.optimizer_steps_per_epoch={steps_per_epoch}",
        "experiment.validation_interval="
        f"{runtime['validation_interval_epochs']}",
        "experiment.collect_gradient_evidence=false",
        "objective=supervised_harmonic",
        "model=hierarchical",
        f"model.hidden_dim={model['hidden_dim']}",
        f"model.local_gnn_layers={model['local_gnn_layers']}",
        f"model.transformer_layers={model['transformer_layers']}",
        f"model.attention_heads={model['attention_heads']}",
        f"model.ffn_multiplier={model['ffn_multiplier']}",
        f"model.dropout={model['dropout']}",
        f"model.residual={str(model['residual']).lower()}",
        f"data.batch_size={batch_size}",
        f"data.epoch_size={batch_size * steps_per_epoch}",
        f"data.validation_epoch_size={runtime['validation_samples']}",
        f"data.validation_seed={runtime['fixed_validation_seed']}",
        f"data.workers={protocol['data']['workers']}",
        "data.mixture_weights={"
        + ",".join(
            f"{dataset_id}:{weight}"
            for dataset_id, weight in protocol["data"]["mixture_weights"]
        )
        + "}",
        f"optimizer={optimizer['name']}",
        f"optimizer.learning_rate={optimizer['learning_rate']}",
        f"optimizer.weight_decay={optimizer['weight_decay']}",
        f"optimizer.gradient_clip_norm={optimizer['gradient_clip_norm']}",
        f"scheduler={protocol['scheduler_config']['name']}",
        "scheduler.minimum_learning_rate="
        f"{protocol['scheduler_config']['minimum_learning_rate']}",
        f"device={device_group}",
        f"device.name={device_name}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
        "device.amp_dtype="
        f"{protocol['amp_device_config']['amp_dtype']}",
        "device.non_blocking="
        f"{str(protocol['amp_device_config']['non_blocking']).lower()}",
        "downstream_task_ids=[" + ",".join(selected_tasks) + "]",
        "+objective.task_weights={"
        + ",".join(f"{key}:{value}" for key, value in task_weights.items())
        + "}",
        f"seed={cell['seed']}",
        f"output_dir={output_directory}",
        "transfer.contract_version=1.1.0",
        f"transfer.mode={mode}",
        "transfer.comparison_protocol_fingerprint="
        f"{protocol['fingerprint']}",
        "transfer.downstream_initialization_seed="
        f"{domains.downstream_initialization}",
        "transfer.downstream_data_order_seed="
        f"{domains.downstream_data_order}",
        "transfer.sample_schedule_fingerprint="
        f"{cell['downstream_schedule_fingerprint']}",
        f"transfer.logical_updates={updates}",
    ]
    if actual_sample_schedule_path:
        overrides.append(
            "transfer.actual_sample_schedule_path="
            f"{actual_sample_schedule_path}"
        )
    if pretrained:
        overrides.extend(
            (
                f"transfer.encoder_export_path={encoder_export_path}",
                f"transfer.encoder_export_sha256={encoder_export_sha256}",
                "transfer.source_ssl_checkpoint_sha256="
                f"{source_ssl_checkpoint_sha256}",
            )
        )
    # Production paths remain caller-supplied and are never hard-coded.
    paths = plan["runtime_paths"]
    if paths["index_paths"]:
        overrides.extend(
            (
                "data=mixed",
                "data.index_paths=["
                + ",".join(paths["index_paths"])
                + "]",
                "data.cache_roots=["
                + ",".join(paths["cache_roots"])
                + "]",
                f"data.split_manifest={paths['split_manifest']}",
            )
        )
    return overrides


def official_evaluation_overrides(
    plan: Mapping[str, object],
    *,
    checkpoint: str,
    output_directory: str,
    split: str = "validation",
    cell_id: str | None = None,
) -> list[str]:
    """Use candidate-first evaluation; comparison test access stays locked."""

    if split != "validation":
        raise Phase8B2ContractError(
            "phase8b2.runner.test_requires_test_lock_authorization"
        )
    protocol = plan["protocol"]
    cells = {
        row["cell_id"]: row
        for row in plan.get("evaluation_cells", [])
        if isinstance(row, dict)
    }
    cell = None if cell_id is None else cells.get(cell_id)
    if cell_id is not None and cell is None:
        raise Phase8B2ContractError(
            "phase8b2.runner.evaluation_cell_unknown"
        )
    evaluation_seed = (
        int(protocol["seeds"][0])
        if cell is None
        else int(cell["evaluation_seed"])
    )
    runtime = protocol["runtime_execution_config"]
    device_name = str(protocol["amp_device_config"]["name"])
    device_group = "cuda" if device_name.startswith("cuda") else device_name
    selected_tasks = [row["task_id"] for row in protocol["downstream_tasks"]]
    overrides = [
        f"checkpoint={checkpoint}",
        "split=validation",
        "acknowledge_test_evaluation=false",
        f"seed={evaluation_seed}",
        f"device={device_group}",
        f"device.name={device_name}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
        "device.amp_dtype="
        f"{protocol['amp_device_config']['amp_dtype']}",
        "device.non_blocking="
        f"{str(protocol['amp_device_config']['non_blocking']).lower()}",
        f"data.batch_size={protocol['compute']['batch_size']}",
        f"data.workers={protocol['data']['workers']}",
        "data.max_evaluation_samples="
        f"{runtime['validation_samples']}",
        f"data.validation_seed={runtime['fixed_validation_seed']}",
        "downstream_task_ids=[" + ",".join(selected_tasks) + "]",
        f"output_dir={output_directory}",
    ]
    paths = plan["runtime_paths"]
    if paths["index_paths"]:
        overrides.extend(
            (
                "data=mixed",
                "data.index_paths=["
                + ",".join(paths["index_paths"])
                + "]",
                "data.cache_roots=["
                + ",".join(paths["cache_roots"])
                + "]",
                f"data.split_manifest={paths['split_manifest']}",
            )
        )
    return overrides


def official_test_evaluation_overrides(
    plan: Mapping[str, object],
    authorization: Mapping[str, object],
) -> list[str]:
    """Bind a consumed, single-use authorization to official test evaluation."""

    protocol = plan["protocol"]
    if (
        authorization.get("authorized") is not True
        or authorization.get("authorization_stage") != "pre_inference"
        or authorization.get("acknowledged") is not True
        or authorization.get("protocol_fingerprint")
        != protocol["fingerprint"]
        or authorization.get("test_membership_fingerprint")
        != protocol["data"]["test_membership_fingerprint"]
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.test_authorization_binding_invalid"
        )
    marker_value = authorization.get("single_use_marker")
    output_value = authorization.get("output_directory")
    checkpoint = authorization.get("selected_checkpoint")
    if not all(
        isinstance(value, str) and value
        for value in (marker_value, output_value, checkpoint)
    ):
        raise Phase8B2ContractError(
            "phase8b2.runner.test_authorization_binding_invalid"
        )
    output = Path(output_value)
    if not output.is_dir() or any(output.iterdir()):
        raise Phase8B2ContractError(
            "phase8b2.runner.test_output_not_new_and_empty"
        )
    marker = read_json(marker_value)
    expected_marker = {
        "test_lock_contract_version": "1.1.0",
        "authorization_stage": "consumed_pre_inference",
        "single_use_identity": authorization["single_use_identity"],
        "protocol_fingerprint": authorization["protocol_fingerprint"],
        "selection_fingerprint": authorization["selection_fingerprint"],
        "selected_variant_id": authorization["selected_variant_id"],
        "selected_checkpoint": checkpoint,
        "selected_checkpoint_sha256": authorization[
            "selected_checkpoint_sha256"
        ],
        "selected_checkpoint_seed": authorization[
            "selected_checkpoint_seed"
        ],
        "test_membership_fingerprint": authorization[
            "test_membership_fingerprint"
        ],
    }
    if marker != expected_marker:
        raise Phase8B2ContractError(
            "phase8b2.runner.test_authorization_marker_mismatch"
        )
    overrides = [
        f"checkpoint={checkpoint}",
        "split=test",
        "acknowledge_test_evaluation=true",
        f"seed={authorization['selected_checkpoint_seed']}",
        f"device={protocol['amp_device_config']['name']}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
        f"output_dir={output_value}",
        "overwrite_output=false",
    ]
    paths = plan["runtime_paths"]
    if paths["index_paths"]:
        overrides.extend(
            (
                "data=mixed",
                "data.index_paths=[" + ",".join(paths["index_paths"]) + "]",
                "data.cache_roots=[" + ",".join(paths["cache_roots"]) + "]",
                f"data.split_manifest={paths['split_manifest']}",
            )
        )
    return overrides


__all__ = [
    "PLAN_CONTRACT_VERSION",
    "build_experiment_plan",
    "official_downstream_overrides",
    "official_evaluation_overrides",
    "official_ssl_cell_overrides",
    "official_test_evaluation_overrides",
    "production_smoke_status",
]
