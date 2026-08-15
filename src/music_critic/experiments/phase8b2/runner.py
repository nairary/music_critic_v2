"""Official Phase 8B.2A matrix planner and engine configuration adapter."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from music_critic.experiments.phase8b2.artifacts import (
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    read_json,
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
PLAN_CONTRACT_VERSION = "1.0.0"


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


def _data_binding(config: Mapping[str, Any]) -> DataBinding:
    data = config["data"]
    dataset_ids = tuple(sorted(data["mixture_weights"]))
    index_fingerprints = dict(data["index_fingerprints"])
    cache_fingerprints = dict(data["cache_fingerprints"])
    production = bool(data["index_paths"] or data["cache_roots"])
    if production:
        if (
            len(data["index_paths"]) != len(dataset_ids)
            or len(data["cache_roots"]) != len(dataset_ids)
            or set(index_fingerprints) != set(dataset_ids)
            or set(cache_fingerprints) != set(dataset_ids)
            or any(not Path(path).is_absolute() for path in data["index_paths"])
            or any(not Path(path).is_absolute() for path in data["cache_roots"])
            or not Path(data["split_manifest"]).is_absolute()
        ):
            raise Phase8B2ContractError(
                "phase8b2.config.production_data_binding_incomplete"
            )
    else:
        index_fingerprints = {
            dataset_id: fingerprint(
                {"bounded_fixture_index": dataset_id}
            )
            for dataset_id in dataset_ids
        }
        cache_fingerprints = {
            dataset_id: fingerprint(
                {"bounded_fixture_cache": dataset_id}
            )
            for dataset_id in dataset_ids
        }
    return DataBinding(
        dataset_indices=tuple(sorted(index_fingerprints.items())),
        cache_identities=tuple(sorted(cache_fingerprints.items())),
        split_manifest_fingerprint=data["split_manifest_fingerprint"],
        train_membership_fingerprint=data["train_membership_fingerprint"],
        validation_membership_fingerprint=(
            data["validation_membership_fingerprint"]
        ),
        test_membership_fingerprint=data["test_membership_fingerprint"],
        mixture_weights=tuple(
            sorted(
                (dataset_id, float(weight))
                for dataset_id, weight in data["mixture_weights"].items()
            )
        ),
    )


def _sample_schedule(
    *,
    dataset_ids: tuple[str, ...],
    seed_domains: SeedDomains,
    count: int,
) -> tuple[tuple[str, str], ...]:
    # Concrete production identities are resolved by the official data runtime
    # and checked against this data-order domain.  Dry-run uses opaque slots,
    # never target values or split labels.
    return tuple(
        (
            dataset_ids[index % len(dataset_ids)],
            f"schedule-slot-{seed_domains.ssl_data_order:016x}-{index:08d}",
        )
        for index in range(count)
    )


def _protocol(config: Mapping[str, Any], data: DataBinding) -> ComparisonProtocol:
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
            "seeds": sorted(comparison["seeds"]),
            "batch_size": config["data"]["batch_size"],
            "optimizer_steps": comparison["downstream_optimizer_steps"],
            "data_order_domain": "downstream_data_order",
            "membership": data.train_membership_fingerprint,
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
    data = _data_binding(plain)
    protocol = _protocol(plain, data)
    dataset_ids = tuple(key for key, _ in data.dataset_indices)
    schedules: dict[int, tuple[VariantSchedule, ...]] = {}
    ssl_cells = []
    downstream_cells = []
    for seed in protocol.seeds:
        domains = SeedDomains.create(seed)
        identities = _sample_schedule(
            dataset_ids=dataset_ids,
            seed_domains=domains,
            count=protocol.compute.raw_sample_exposures,
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
                            protocol.downstream_schedule_fingerprint
                        ),
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
                    protocol.downstream_schedule_fingerprint
                ),
            }
        )
    plan = {
        "plan_contract_version": PLAN_CONTRACT_VERSION,
        "dry_run": True,
        "training_performed": False,
        "test_accessed": False,
        "protocol": protocol.to_dict(),
        "ssl_cells": sorted(ssl_cells, key=lambda row: row["cell_id"]),
        "downstream_cells": sorted(
            downstream_cells, key=lambda row: row["cell_id"]
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
        "claims": {
            "bounded_acceptance_is_scientific_superiority_evidence": False,
            "pdmx_evidence": False,
            "production_training_performed": False,
        },
    }
    plan["fingerprint"] = fingerprint(plan)
    return plan


def official_ssl_cell_overrides(
    plan: Mapping[str, object], cell_id: str, output_directory: str
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
    overrides = [
        f"+phase8b_objective={objective_mode}",
        f"+phase8b_masking={masking_mode}",
        "+phase8b2_schedule=comparison",
        "experiment=pretrain",
        f"experiment.epochs={schedule['logical_updates']}",
        f"data.batch_size={schedule['batch_size']}",
        f"data.epoch_size={schedule['batch_size']}",
        f"data.validation_epoch_size={schedule['batch_size']}",
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
        f"optimizer.learning_rate={protocol['optimizer_config']['learning_rate']}",
        f"optimizer.weight_decay={protocol['optimizer_config']['weight_decay']}",
        "optimizer.gradient_clip_norm="
        f"{protocol['optimizer_config']['gradient_clip_norm']}",
        f"scheduler={protocol['scheduler_config']['name']}",
        f"device={protocol['amp_device_config']['name']}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
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
        "phase8b2_schedule.contract_version=1.0.0",
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
        "phase8b2_schedule.policy_view_names=["
        + ",".join(view["policy"] for view in views)
        + "]",
        "phase8b2_schedule.policy_view_seeds=["
        + ",".join(str(view["seed"]) for view in views)
        + "]",
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
    overrides = [
        "experiment=smoke",
        f"experiment.epochs={updates}",
        "experiment.collect_gradient_evidence=false",
        "objective=supervised_harmonic",
        "model=hierarchical",
        f"model.hidden_dim={model['hidden_dim']}",
        f"model.local_gnn_layers={model['local_gnn_layers']}",
        f"model.transformer_layers={model['transformer_layers']}",
        f"model.attention_heads={model['attention_heads']}",
        f"model.ffn_multiplier={model['ffn_multiplier']}",
        f"model.dropout={model['dropout']}",
        f"data.batch_size={batch_size}",
        f"data.epoch_size={batch_size}",
        f"data.validation_epoch_size={batch_size}",
        f"optimizer.learning_rate={optimizer['learning_rate']}",
        f"optimizer.weight_decay={optimizer['weight_decay']}",
        f"optimizer.gradient_clip_norm={optimizer['gradient_clip_norm']}",
        f"scheduler={protocol['scheduler_config']['name']}",
        f"device={protocol['amp_device_config']['name']}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
        f"seed={cell['seed']}",
        f"output_dir={output_directory}",
        "transfer.contract_version=1.0.0",
        f"transfer.mode={mode}",
        "transfer.comparison_protocol_fingerprint="
        f"{protocol['fingerprint']}",
        "transfer.downstream_initialization_seed="
        f"{domains.downstream_initialization}",
        "transfer.downstream_data_order_seed="
        f"{domains.downstream_data_order}",
    ]
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
) -> list[str]:
    """Use candidate-first evaluation; comparison test access stays locked."""

    if split != "validation":
        raise Phase8B2ContractError(
            "phase8b2.runner.test_requires_test_lock_authorization"
        )
    protocol = plan["protocol"]
    overrides = [
        f"checkpoint={checkpoint}",
        "split=validation",
        "acknowledge_test_evaluation=false",
        f"seed={protocol['seeds'][0]}",
        f"device={protocol['amp_device_config']['name']}",
        f"device.amp={str(protocol['amp_device_config']['amp']).lower()}",
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
        "test_lock_contract_version": "1.0.0",
        "authorization_stage": "consumed_pre_inference",
        "single_use_identity": authorization["single_use_identity"],
        "protocol_fingerprint": authorization["protocol_fingerprint"],
        "selection_fingerprint": authorization["selection_fingerprint"],
        "selected_variant_id": authorization["selected_variant_id"],
        "selected_checkpoint": checkpoint,
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
        f"seed={protocol['seeds'][0]}",
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
