"""Target-blind planning for the Phase 9C-A dependency DAG."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from music_critic.experiments.phase8b2.schedule import (
    SCHEDULE_CONTRACT_VERSION,
    SeedDomains,
    build_variant_schedule,
    derive_seed,
)
from music_critic.experiments.phase8b2.attestation import (
    resolve_actual_downstream_schedule,
    resolve_actual_ssl_schedule,
)
from music_critic.tasks import load_corpus_index, load_split_manifest

from .contracts import (
    CLAIM_BOUNDARIES,
    DOWNSTREAM_MODES,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PHASE9C_PLAN_VERSION,
    PHASE9C_PROTOCOL_VERSION,
    PHASE9C_SEED,
    PRIMARY_VARIANTS,
    SSL_PRIMARY_VARIANTS,
    TASK_IDS,
    Phase9CContractError,
    fingerprint,
    locked_test_state,
    resolve_preset,
    validate_protocol,
)
from .sampling import build_source_balanced_schedule


DEFAULT_MIXTURE = {
    "dilemmadata": 1.0 / 3.0,
    "hooktheory": 1.0 / 3.0,
    "pop909_cl": 1.0 / 3.0,
}


def _repository_evidence(*, require_clean: bool) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase9CContractError("phase9c.plan.repository_unavailable") from exc
    clean = status == ""
    if require_clean and not clean:
        raise Phase9CContractError("phase9c.plan.production_clean_head_required")
    return {"git_head": head, "clean": clean, "exact_head_required": True}


def _file_sha256(path: str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_identities() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    ssl = {
        dataset_id: tuple(f"{dataset_id}:fixture:{index}" for index in range(4))
        for dataset_id in sorted(DEFAULT_MIXTURE)
    }
    downstream = tuple(f"dilemmadata:train:fixture:{index}" for index in range(6))
    return ssl, downstream


def _production_data(config: Mapping[str, Any]) -> tuple[
    dict[str, tuple[str, ...]], tuple[str, ...], dict[str, object], dict[str, object]
]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise Phase9CContractError("phase9c.plan.production_data_missing")
    index_paths = tuple(str(path) for path in data.get("ssl_index_paths", ()))
    cache_roots = tuple(str(path) for path in data.get("ssl_cache_roots", ()))
    if len(index_paths) != 3 or len(cache_roots) != 3:
        raise Phase9CContractError("phase9c.plan.ssl_three_sources_required")
    ssl_split_path = str(data.get("ssl_split_manifest", ""))
    downstream_split_path = str(data.get("downstream_split_manifest", ""))
    downstream_raw_index = str(data.get("downstream_raw_index", ""))
    downstream_raw_cache = str(data.get("downstream_raw_cache_root", ""))
    target_index = str(data.get("target_cache_index", ""))
    target_cache = str(data.get("target_cache_root", ""))
    all_paths = (
        *index_paths,
        *cache_roots,
        ssl_split_path,
        downstream_split_path,
        downstream_raw_index,
        downstream_raw_cache,
        target_index,
        target_cache,
    )
    missing = sorted(path for path in all_paths if not path or not Path(path).exists())
    if missing:
        raise Phase9CContractError("phase9c.plan.production_path_missing:" + ",".join(missing))

    indices = tuple(load_corpus_index(path) for path in index_paths)
    by_dataset = {index.header.dataset_id: index for index in indices}
    if set(by_dataset) != set(DEFAULT_MIXTURE):
        raise Phase9CContractError("phase9c.plan.ssl_dataset_inventory_invalid")
    ssl_split = load_split_manifest(ssl_split_path)
    train_keys = {
        (row.dataset_id, row.piece_id)
        for row in ssl_split.assignments
        if row.split == "train"
    }
    ssl_identities = {
        dataset_id: tuple(
            row.piece_id
            for row in index.records
            if (dataset_id, row.piece_id) in train_keys
        )
        for dataset_id, index in sorted(by_dataset.items())
    }
    if any(not rows for rows in ssl_identities.values()):
        raise Phase9CContractError("phase9c.plan.ssl_train_source_empty")

    downstream_index = load_corpus_index(downstream_raw_index)
    if downstream_index.header.dataset_id != "dilemmadata":
        raise Phase9CContractError("phase9c.plan.downstream_dataset_invalid")
    downstream_split = load_split_manifest(downstream_split_path)
    split_rows: dict[str, list[object]] = {name: [] for name in ("train", "validation", "test")}
    for row in downstream_split.assignments:
        if row.dataset_id == "dilemmadata" and row.split in split_rows:
            split_rows[row.split].append(row)
    record_counts = {name: len(rows) for name, rows in split_rows.items()}
    component_counts = {
        name: len({row.component_fingerprint for row in rows})
        for name, rows in split_rows.items()
    }
    if record_counts != {"train": 577, "validation": 71, "test": 71} or component_counts != {
        "train": 565,
        "validation": 71,
        "test": 71,
    }:
        raise Phase9CContractError("phase9c.plan.dilemmadata_split_counts_invalid")
    downstream_train = tuple(sorted(row.piece_id for row in split_rows["train"]))
    validation_identities = tuple(sorted(row.piece_id for row in split_rows["validation"]))
    test_identities = tuple(sorted(row.piece_id for row in split_rows["test"]))
    data_projection = {
        "kind": "production",
        "ssl_indices": [
            {
                "dataset_id": index.header.dataset_id,
                "index_fingerprint": index.header.index_fingerprint,
                "cache_root": str(Path(cache_roots[index_paths.index(path)]).resolve()),
            }
            for path, index in zip(index_paths, indices, strict=True)
        ],
        "ssl_split_manifest_fingerprint": ssl_split.manifest_fingerprint,
        "downstream_raw_index_fingerprint": downstream_index.header.index_fingerprint,
        "downstream_target_index_sha256": _file_sha256(target_index),
        "downstream_split_manifest_fingerprint": downstream_split.manifest_fingerprint,
        "downstream_split_record_counts": record_counts,
        "downstream_split_component_counts": component_counts,
        "validation_membership_fingerprint": fingerprint(
            {"split": "validation", "piece_ids": validation_identities}
        ),
        "test_membership_fingerprint": fingerprint(
            {"split": "test", "piece_ids": test_identities}
        ),
        "test_membership_count": len(test_identities),
        "test_identities_serialized": False,
        "target_bundles_loaded_during_planning": False,
    }
    runtime_paths = {
        "ssl_index_paths": list(index_paths),
        "ssl_cache_roots": list(cache_roots),
        "ssl_split_manifest": ssl_split_path,
        "downstream_raw_index": downstream_raw_index,
        "downstream_raw_cache_root": downstream_raw_cache,
        "target_cache_index": target_index,
        "target_cache_root": target_cache,
        "downstream_split_manifest": downstream_split_path,
    }
    return ssl_identities, downstream_train, data_projection, runtime_paths


def _downstream_schedule(identities: tuple[str, ...], count: int, seed: int) -> dict[str, object]:
    schedule = build_source_balanced_schedule(
        {"dilemmadata": identities},
        weights={"dilemmadata": 1.0},
        sample_count=count,
        seed=seed,
    )
    engine_fingerprint = fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_downstream_sample_schedule",
            "identities": [
                [str(row["dataset_id"]), str(row["piece_id"])]
                for row in schedule["slots"]
            ],
        }
    )
    return {**schedule, "engine_schedule_fingerprint": engine_fingerprint}


def _official_schedule_evidence(
    actual: Mapping[str, object],
    *,
    source_sizes: Mapping[str, int],
    weights: Mapping[str, float],
    engine_key: str,
) -> dict[str, object]:
    slots = list(actual["slots"])
    counts = Counter(str(row["dataset_id"]) for row in slots)
    unique = {
        source: len(
            {
                str(row["piece_id"])
                for row in slots
                if row["dataset_id"] == source
            }
        )
        for source in sorted(source_sizes)
    }
    payload = {
        "contract_version": "1.0.0",
        "seed": actual["data_order_seed"],
        "weights": [[source, float(weights[source])] for source in sorted(weights)],
        "slots": slots,
        "dataset_counts": {source: counts[source] for source in sorted(source_sizes)},
        "unique_record_counts": unique,
        "repeat_counts": {
            source: counts[source] - unique[source] for source in sorted(source_sizes)
        },
        "completed_or_entered_cycle_counts": {
            source: (
                0
                if counts[source] == 0
                else (counts[source] + source_sizes[source] - 1) // source_sizes[source]
            )
            for source in sorted(source_sizes)
        },
        "replacement_within_cycle": False,
        "target_or_provenance_access": False,
        "official_sampler": actual["sampler"],
        "engine_schedule_fingerprint": actual["sample_schedule_fingerprint"],
        engine_key: actual["sample_schedule_fingerprint"],
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def build_experiment_plan(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve the complete immutable one-seed plan without target-bundle access."""

    preset_name = str(config.get("preset", "bounded_acceptance"))
    preset = resolve_preset(
        preset_name,
        ssl_updates=config.get("ssl_updates"),
        downstream_epochs=config.get("downstream_epochs"),
        downstream_steps_per_epoch=config.get("downstream_steps_per_epoch"),
        batch_size=config.get("batch_size"),
        bootstrap_replicates=config.get("bootstrap_replicates"),
    )
    production = preset_name != "bounded_acceptance"
    if production:
        ssl_identities, downstream_ids, data_projection, runtime_paths = _production_data(config)
    else:
        ssl_identities, downstream_ids = _bounded_identities()
        data_projection = {
            "kind": "bounded_synthetic_fixture",
            "downstream_split_record_counts": {"train": 6, "validation": 4, "test": 4},
            "downstream_split_component_counts": {"train": 6, "validation": 4, "test": 4},
            "validation_membership_fingerprint": fingerprint({"fixture": "validation"}),
            "test_membership_fingerprint": fingerprint({"fixture": "test"}),
            "test_membership_count": 4,
            "test_identities_serialized": False,
            "target_bundles_loaded_during_planning": False,
        }
        runtime_paths = {}

    batch_size = preset.batch_size or 1
    ssl_updates = preset.ssl_updates or 1
    downstream_epochs = preset.downstream_epochs or 1
    downstream_steps = preset.downstream_steps_per_epoch or 1
    mixture = dict(config.get("mixture_weights", DEFAULT_MIXTURE))
    if production:
        ssl_data_config = {
            "index_paths": runtime_paths["ssl_index_paths"],
            "cache_roots": runtime_paths["ssl_cache_roots"],
            "split_manifest": runtime_paths["ssl_split_manifest"],
            "batch_size": batch_size,
            "workers": 0,
            "mixture_weights": mixture,
        }
        actual_ssl = resolve_actual_ssl_schedule(
            ssl_data_config,
            data_semantic_projection=data_projection,
            data_binding={},
            seed=PHASE9C_SEED,
            logical_updates=ssl_updates,
            optimizer_steps_per_epoch=min(ssl_updates, downstream_steps),
            validation_samples=0,
            validation_seed=20260817,
        )
        schedule = _official_schedule_evidence(
            actual_ssl,
            source_sizes={source: len(rows) for source, rows in ssl_identities.items()},
            weights=mixture,
            engine_key="sample_schedule_fingerprint",
        )
        downstream_data_config = {
            "index_paths": [runtime_paths["downstream_raw_index"]],
            "cache_roots": [runtime_paths["downstream_raw_cache_root"]],
            "split_manifest": runtime_paths["downstream_split_manifest"],
            "batch_size": batch_size,
            "workers": 0,
            "mixture_weights": {"dilemmadata": 1.0},
        }
        actual_downstream = resolve_actual_downstream_schedule(
            downstream_data_config,
            data_semantic_projection=data_projection,
            data_binding={},
            seed=PHASE9C_SEED,
            logical_updates=downstream_epochs * downstream_steps,
            optimizer_steps_per_epoch=downstream_steps,
            validation_samples=0,
            validation_seed=20260817,
        )
        downstream_schedule = _official_schedule_evidence(
            actual_downstream,
            source_sizes={"dilemmadata": len(downstream_ids)},
            weights={"dilemmadata": 1.0},
            engine_key="engine_schedule_fingerprint",
        )
    else:
        schedule = build_source_balanced_schedule(
            ssl_identities,
            weights=mixture,
            sample_count=ssl_updates * batch_size,
            seed=derive_seed(PHASE9C_SEED, "phase9c/ssl_data_order"),
        )
        downstream_schedule = _downstream_schedule(
            downstream_ids,
            downstream_epochs * downstream_steps * batch_size,
            derive_seed(PHASE9C_SEED, "phase9c/downstream_data_order"),
        )
    domains = SeedDomains.create(PHASE9C_SEED)
    ssl_variants = tuple(variant for variant in preset.variants if variant != "scratch")
    variant_schedules = {
        variant: build_variant_schedule(
            variant,
            comparison_mode="encoder_forward_matched",
            logical_updates=ssl_updates,
            batch_size=batch_size,
            matched_encoder_forwards_per_update=PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
            sample_identity_schedule=tuple(
                (str(row["dataset_id"]), str(row["piece_id"]))
                for row in schedule["slots"]
            ),
            mask_seed=domains.ssl_mask_planning,
        ).to_dict()
        for variant in ssl_variants
    }
    protocol_payload = {
        "contract_version": PHASE9C_PROTOCOL_VERSION,
        "phase": "9C-A",
        "seed": PHASE9C_SEED,
        "preset": preset.to_dict(),
        "repository": _repository_evidence(require_clean=production),
        "primary_variants": list(PRIMARY_VARIANTS),
        "executed_variants": list(preset.variants),
        "optional_variants_automatic": False,
        "task_ids": list(TASK_IDS),
        "data": data_projection,
        "mixture": {
            "weights": [[key, float(mixture[key])] for key in sorted(mixture)],
            "schedule_fingerprint": schedule["fingerprint"],
            "target_blind": True,
            "validation_and_test_excluded": True,
        },
        "paired_initialization": {
            "initial_encoder_seed": domains.model_initialization,
            "initial_encoder_fingerprint": fingerprint(
                {"domain": "initial_encoder", "seed": domains.model_initialization}
            ),
            "fresh_head_seed": domains.downstream_initialization,
            "fresh_head_fingerprint": fingerprint(
                {"domain": "fresh_heads", "seed": domains.downstream_initialization}
            ),
            "ssl_data_order_seed": domains.ssl_data_order,
            "downstream_data_order_seed": domains.downstream_data_order,
        },
        "compute": {
            "unit": "instrumented_encoder_forward",
            "encoder_forwards_per_logical_update": PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
            "ssl_logical_updates": preset.ssl_updates,
            "downstream_epochs": preset.downstream_epochs,
            "downstream_steps_per_epoch": preset.downstream_steps_per_epoch,
            "batch_size": preset.batch_size,
            "fixed_ssl_budget_no_downstream_early_stopping": True,
        },
        "supervised": {
            "reduction": "candidate_rows_mean_then_source_entries_mean_then_fixed_equal_task_sum",
            "fp32_boundary": "encoder_amp_float16_heads_logits_ce_total_fp32",
            "grad_scaler_initial_scale": 16384,
            "scheduler_after_applied_update_only": True,
            "positive_unlabeled_and_open_string_heads": False,
        },
        "selection": {
            "split": "validation",
            "primary_metric": "mean_task_nll_div_log_class_count",
            "lower_is_better": True,
            "tie_breakers": [
                "higher_mean_macro_f1",
                "lower_mean_task_nll",
                "earlier_epoch",
                "lexicographic_checkpoint_identity",
            ],
            "fixed_before_results": True,
        },
        "bootstrap": {
            "unit": "component",
            "seed": domains.bootstrap,
            "replicates": preset.bootstrap_replicates,
            "reference": "scratch_same_transfer_mode",
        },
        "test_lock": locked_test_state(),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    protocol = {**protocol_payload, "fingerprint": fingerprint(protocol_payload)}
    validate_protocol(protocol)

    ssl_cells = [
        {
            "cell_id": f"ssl/{variant}",
            "variant_id": variant,
            "schedule": variant_schedules[variant],
            "initial_encoder_fingerprint": protocol["paired_initialization"]["initial_encoder_fingerprint"],
        }
        for variant in ssl_variants
    ]
    export_cells = [
        {
            "cell_id": "encoder_export/initial_scratch",
            "variant_id": "scratch",
            "depends_on": None,
            "source_kind": "phase6_hierarchical",
        },
        *[
        {
            "cell_id": f"encoder_export/{variant}",
            "variant_id": variant,
            "depends_on": f"ssl/{variant}",
            "source_kind": (
                "phase7a_ssl"
                if variant in {"phase7a_control", "phase8a_mask_only"}
                else "phase8b_multilevel_ssl"
            ),
        }
        for variant in ssl_variants
        ],
    ]
    downstream_cells = []
    for variant in preset.variants:
        for mode in DOWNSTREAM_MODES:
            downstream_cells.append(
                {
                    "cell_id": f"downstream/{variant}/{mode}",
                    "variant_id": variant,
                    "transfer_mode": (
                        f"scratch_{mode}" if variant == "scratch" else mode
                    ),
                    "engine_transfer_mode": (
                        "supervised_scratch" if variant == "scratch" else mode
                    ),
                    "depends_on": (
                        "encoder_export/initial_scratch"
                        if variant == "scratch" and mode == "frozen_probe"
                        else (None if variant == "scratch" else f"encoder_export/{variant}")
                    ),
                    "sample_schedule_fingerprint": downstream_schedule["engine_schedule_fingerprint"],
                    "fresh_head_fingerprint": protocol["paired_initialization"]["fresh_head_fingerprint"],
                }
            )
    validation_cells = [
        {
            "cell_id": f"validation/{row['variant_id']}/{row['transfer_mode']}",
            "variant_id": row["variant_id"],
            "transfer_mode": row["transfer_mode"],
            "depends_on": row["cell_id"],
            "prior_dependency": "train_priors/dilemmadata",
            "split": "validation",
            "membership_fingerprint": data_projection["validation_membership_fingerprint"],
        }
        for row in downstream_cells
    ]
    plan_payload = {
        "contract_version": PHASE9C_PLAN_VERSION,
        "protocol": protocol,
        "data_semantic_projection": data_projection,
        "runtime_paths": runtime_paths,
        "ssl_sample_schedule": schedule,
        "downstream_sample_schedule": downstream_schedule,
        "variant_schedules": variant_schedules,
        "ssl_cells": ssl_cells,
        "encoder_export_cells": export_cells,
        "downstream_cells": downstream_cells,
        "train_prior_cells": [
            {
                "cell_id": "train_priors/dilemmadata",
                "split": "train",
                "target_access": "downstream_train_only",
                "depends_on": None,
            }
        ],
        "validation_cells": validation_cells,
        "profile_candidates": list(config.get("profile_batch_candidates", [1, 2, 3, 4, 6, 8])),
        "profile_report_path": str(config.get("profile_report_path", "")),
        "profile_rebuild_config": (
            None
            if not production
            else {
                **dict(config),
                "preset": "rtx_profile",
                "profile_report_path": "",
            }
        ),
        "production_run_requires_explicit_action": True,
        "test_action_implemented": False,
    }
    return {**plan_payload, "fingerprint": fingerprint(plan_payload)}


__all__ = ["DEFAULT_MIXTURE", "build_experiment_plan"]
