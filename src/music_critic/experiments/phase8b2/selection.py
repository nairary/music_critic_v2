"""Validation-only ranking and single-use held-out test lock."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

from music_critic.experiments.phase8b2.artifacts import read_json
from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_SELECTION_CONTRACT_VERSION,
    PHASE8B2_TEST_LOCK_CONTRACT_VERSION,
    Phase8B2ContractError,
    canonical_json_bytes,
    fingerprint,
)


PRIMARY_DATASETS = ("hooktheory", "pop909_cl")


def _number(value: object, category: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise Phase8B2ContractError(f"phase8b2.selection.{category}")
    return float(value)


def select_validation_checkpoint(
    candidates: Iterable[Mapping[str, object]],
    *,
    protocol_fingerprint: str,
    selection_scope: str = "primary_hierarchical",
    declared_seeds: Iterable[int] | None = None,
) -> dict[str, object]:
    """Aggregate paired seeds first, then rank configurations on validation."""

    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        raise Phase8B2ContractError(
            "phase8b2.selection.candidate_set_empty"
        )
    required = {
        "variant_id",
        "checkpoint",
        "protocol_fingerprint",
        "split",
        "dataset_endpoints",
        "validation_nll",
        "encoder_forward_count",
    }
    explicit_seeds = declared_seeds is not None
    seeds = tuple(sorted(declared_seeds or (0,)))
    if not seeds or len(seeds) != len(set(seeds)):
        raise Phase8B2ContractError(
            "phase8b2.selection.declared_seeds_invalid"
        )
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        if not required <= set(row):
            raise Phase8B2ContractError(
                "phase8b2.selection.candidate_fields_missing"
            )
        variant_id = row["variant_id"]
        transfer_mode = row.get("transfer_mode", "full_finetune")
        seed = row.get("seed", 0)
        identity = (seed, variant_id, transfer_mode)
        if (
            not isinstance(variant_id, str)
            or not isinstance(transfer_mode, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or identity in seen
        ):
            raise Phase8B2ContractError(
                "phase8b2.selection.duplicate_downstream_cell"
            )
        seen.add(identity)
        row["seed"] = seed
        row["transfer_mode"] = transfer_mode
        if not isinstance(row["checkpoint"], str) or not row["checkpoint"]:
            raise Phase8B2ContractError(
                "phase8b2.selection.checkpoint_invalid"
            )
        if row["protocol_fingerprint"] != protocol_fingerprint:
            raise Phase8B2ContractError(
                "phase8b2.selection.protocol_fingerprint_mismatch"
            )
        if row["split"] != "validation":
            raise Phase8B2ContractError(
                "phase8b2.selection.validation_only"
            )
        endpoints = row["dataset_endpoints"]
        if not isinstance(endpoints, Mapping) or set(endpoints) != set(
            PRIMARY_DATASETS
        ):
            raise Phase8B2ContractError(
                "phase8b2.selection.primary_endpoints_incomplete"
            )
        row["validation_nll"] = _number(
            row["validation_nll"], "validation_nll_invalid"
        )
        if row["validation_nll"] < 0:
            raise Phase8B2ContractError(
                "phase8b2.selection.validation_nll_invalid"
            )
        compute = row["encoder_forward_count"]
        if (
            isinstance(compute, bool)
            or not isinstance(compute, int)
            or compute < 0
        ):
            raise Phase8B2ContractError(
                "phase8b2.selection.encoder_forward_count_invalid"
            )
        row["_endpoint_scores"] = {
            dataset_id: _number(
                endpoints[dataset_id],
                f"endpoint_invalid:{dataset_id}",
            )
            for dataset_id in PRIMARY_DATASETS
        }
        checkpoint_sha = row.get("checkpoint_sha256")
        if checkpoint_sha is None:
            checkpoint_sha = "0" * 64
        if (
            not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint_sha)
        ):
            raise Phase8B2ContractError(
                "phase8b2.selection.checkpoint_sha256_invalid"
            )
        row["checkpoint_sha256"] = checkpoint_sha
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["variant_id"]), str(row["transfer_mode"])), []
        ).append(row)
    aggregated = []
    for (variant_id, transfer_mode), group in sorted(groups.items()):
        group_seeds = tuple(sorted(int(row["seed"]) for row in group))
        expected = seeds if explicit_seeds else group_seeds
        if group_seeds != expected:
            raise Phase8B2ContractError(
                "phase8b2.selection.paired_seed_evidence_incomplete"
            )
        configuration_id = f"{variant_id}/{transfer_mode}"
        aggregated.append(
            {
                "configuration_id": configuration_id,
                "variant_id": variant_id,
                "transfer_mode": transfer_mode,
                "declared_seeds": list(expected),
                "seed_checkpoints": [
                    {
                        "seed": row["seed"],
                        "checkpoint": row["checkpoint"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "dataset_endpoints": row["_endpoint_scores"],
                        "validation_nll": row["validation_nll"],
                        "encoder_forward_count": row[
                            "encoder_forward_count"
                        ],
                    }
                    for row in sorted(group, key=lambda item: int(item["seed"]))
                ],
                "dataset_endpoints": {
                    dataset_id: mean(
                        float(row["_endpoint_scores"][dataset_id])
                        for row in group
                    )
                    for dataset_id in PRIMARY_DATASETS
                },
                "validation_nll": mean(
                    float(row["validation_nll"]) for row in group
                ),
                "encoder_forward_count": int(
                    mean(float(row["encoder_forward_count"]) for row in group)
                ),
            }
        )
    ranks: dict[str, dict[str, float]] = {
        str(row["configuration_id"]): {} for row in aggregated
    }
    for dataset_id in PRIMARY_DATASETS:
        ordered = sorted(
            aggregated,
            key=lambda row: (
                -row["dataset_endpoints"][dataset_id],
                str(row["configuration_id"]),
            ),
        )
        index = 0
        while index < len(ordered):
            end = index + 1
            score = ordered[index]["dataset_endpoints"][dataset_id]
            while (
                end < len(ordered)
                and ordered[end]["dataset_endpoints"][dataset_id] == score
            ):
                end += 1
            rank = (index + 1 + end) / 2.0
            for row in ordered[index:end]:
                ranks[str(row["configuration_id"])][dataset_id] = rank
            index = end
    ranked = []
    for row in aggregated:
        configuration_id = str(row["configuration_id"])
        mean_rank = sum(ranks[configuration_id].values()) / len(PRIMARY_DATASETS)
        ranked.append(
            {
                **row,
                "dataset_ranks": ranks[configuration_id],
                "mean_dataset_rank": mean_rank,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["mean_dataset_rank"],
            row["validation_nll"],
            row["encoder_forward_count"],
            row["configuration_id"],
        )
    )
    selected = ranked[0]
    artifact = {
        "selection_contract_version": PHASE8B2_SELECTION_CONTRACT_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "selection_scope": selection_scope,
        "source_split": "validation",
        "test_used": False,
        "primary_datasets": list(PRIMARY_DATASETS),
        "rule": "mean_dataset_rank",
        "tie_breaks": [
            "lower_validation_nll",
            "lower_encoder_forward_count",
            "lexicographic_configuration_id",
        ],
        "selection_identity": ["variant_id", "transfer_mode"],
        "seed_aggregation": "paired_arithmetic_mean_before_ranking",
        "declared_seeds": list(seeds if explicit_seeds else selected["declared_seeds"]),
        "ranked_candidates": ranked,
        "selected_variant_id": selected["variant_id"],
        "selected_transfer_mode": selected["transfer_mode"],
        "selected_configuration_id": selected["configuration_id"],
        "selected_checkpoints": selected["seed_checkpoints"],
        "selected_checkpoint": selected["seed_checkpoints"][0]["checkpoint"],
        "selected_count": len(selected["seed_checkpoints"]),
        "complete_paired_seed_evidence": True,
    }
    artifact["fingerprint"] = fingerprint(artifact)
    return artifact


@dataclass(frozen=True, slots=True)
class TestUnlockRequest:
    __test__ = False

    protocol_fingerprint: str
    experiment_identity: str
    selection_artifact: str
    output_directory: str
    test_membership_fingerprint: str
    acknowledge: bool
    checkpoint_seed: int | None = None


def authorize_test_evaluation(
    request: TestUnlockRequest,
) -> dict[str, object]:
    """Validate every lock condition before any inference can begin."""

    if not request.acknowledge:
        raise Phase8B2ContractError(
            "phase8b2.test_lock.acknowledgement_required"
        )
    if (
        not isinstance(request.protocol_fingerprint, str)
        or not request.protocol_fingerprint
        or not isinstance(request.experiment_identity, str)
        or not request.experiment_identity
    ):
        raise Phase8B2ContractError(
            "phase8b2.test_lock.identity_invalid"
        )
    output = Path(request.output_directory).resolve()
    if output.exists():
        raise Phase8B2ContractError(
            "phase8b2.test_lock.new_output_directory_required"
        )
    selection_path = Path(request.selection_artifact)
    selection = read_json(selection_path)
    if not isinstance(selection, dict):
        raise Phase8B2ContractError(
            "phase8b2.test_lock.selection_artifact_invalid"
        )
    recorded_selection_fingerprint = selection.get("fingerprint")
    selection_payload = dict(selection)
    selection_payload.pop("fingerprint", None)
    if (
        not isinstance(recorded_selection_fingerprint, str)
        or fingerprint(selection_payload) != recorded_selection_fingerprint
    ):
        raise Phase8B2ContractError(
            "phase8b2.test_lock.selection_artifact_stale"
        )
    if selection.get("selection_contract_version") != (
        PHASE8B2_SELECTION_CONTRACT_VERSION
    ) or selection.get("source_split") != "validation" or selection.get(
        "test_used"
    ) is not False:
        raise Phase8B2ContractError(
            "phase8b2.test_lock.selection_artifact_invalid"
        )
    if selection.get("protocol_fingerprint") != request.protocol_fingerprint:
        raise Phase8B2ContractError(
            "phase8b2.test_lock.protocol_fingerprint_mismatch"
        )
    selected_checkpoints = selection.get("selected_checkpoints")
    if not isinstance(selected_checkpoints, list):
        legacy = selection.get("selected_checkpoint")
        selected_checkpoints = (
            [{"seed": 0, "checkpoint": legacy, "checkpoint_sha256": "0" * 64}]
            if isinstance(legacy, str) and legacy
            else []
        )
    if (
        selection.get("selected_count") != len(selected_checkpoints)
        or not selected_checkpoints
        or len({row.get("seed") for row in selected_checkpoints})
        != len(selected_checkpoints)
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("seed"), int)
            or not isinstance(row.get("checkpoint"), str)
            or not row.get("checkpoint")
            or not isinstance(row.get("checkpoint_sha256"), str)
            or len(row["checkpoint_sha256"]) != 64
            for row in selected_checkpoints
        )
    ):
        raise Phase8B2ContractError(
            "phase8b2.test_lock.selected_checkpoint_manifest_invalid"
        )
    checkpoint_seed = request.checkpoint_seed
    if checkpoint_seed is None:
        if len(selected_checkpoints) != 1:
            raise Phase8B2ContractError(
                "phase8b2.test_lock.checkpoint_seed_required"
            )
        selected_checkpoint = selected_checkpoints[0]
    else:
        matches = [
            row for row in selected_checkpoints
            if row["seed"] == checkpoint_seed
        ]
        if len(matches) != 1:
            raise Phase8B2ContractError(
                "phase8b2.test_lock.checkpoint_seed_not_selected"
            )
        selected_checkpoint = matches[0]
    if not request.test_membership_fingerprint:
        raise Phase8B2ContractError(
            "phase8b2.test_lock.test_membership_missing"
        )
    lock_identity = fingerprint(
        {
            "contract_version": PHASE8B2_TEST_LOCK_CONTRACT_VERSION,
            "experiment_identity": request.experiment_identity,
            "protocol_fingerprint": request.protocol_fingerprint,
            "selection_fingerprint": recorded_selection_fingerprint,
            "test_membership_fingerprint": (
                request.test_membership_fingerprint
            ),
            "checkpoint_seed": selected_checkpoint["seed"],
            "checkpoint_sha256": selected_checkpoint[
                "checkpoint_sha256"
            ],
        }
    )
    # The sibling marker is outside the not-yet-created output directory and
    # makes one-use identity checks possible before inference.
    marker = output.parent / f".{output.name}.{lock_identity}.test-used"
    if marker.exists():
        raise Phase8B2ContractError(
            "phase8b2.test_lock.experiment_identity_already_used"
        )
    return {
        "test_lock_contract_version": PHASE8B2_TEST_LOCK_CONTRACT_VERSION,
        "authorized": True,
        "authorization_stage": "pre_inference",
        "protocol_fingerprint": request.protocol_fingerprint,
        "experiment_identity": request.experiment_identity,
        "selection_fingerprint": recorded_selection_fingerprint,
        "selected_variant_id": selection.get("selected_variant_id"),
        "selected_transfer_mode": selection.get("selected_transfer_mode"),
        "selected_checkpoint": selected_checkpoint["checkpoint"],
        "selected_checkpoint_sha256": selected_checkpoint[
            "checkpoint_sha256"
        ],
        "selected_checkpoint_seed": selected_checkpoint["seed"],
        "authorized_checkpoint_manifest": selected_checkpoints,
        "test_membership_fingerprint": request.test_membership_fingerprint,
        "test_membership_metadata_resolved": True,
        "test_inference_performed": False,
        "test_targets_accessed": False,
        "test_metrics_accessed": False,
        "output_directory": str(output),
        "single_use_identity": lock_identity,
        "single_use_marker": str(marker),
        "acknowledged": True,
    }


def consume_test_authorization(authorization: Mapping[str, object]) -> None:
    """Atomically consume a validated identity immediately before inference."""

    if authorization.get("authorized") is not True or authorization.get(
        "authorization_stage"
    ) != "pre_inference":
        raise Phase8B2ContractError(
            "phase8b2.test_lock.authorization_invalid"
        )
    marker_value = authorization.get("single_use_marker")
    output_value = authorization.get("output_directory")
    if not isinstance(marker_value, str) or not isinstance(output_value, str):
        raise Phase8B2ContractError(
            "phase8b2.test_lock.authorization_invalid"
        )
    marker = Path(marker_value)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = marker.open("xb")
    except FileExistsError as exc:
        raise Phase8B2ContractError(
            "phase8b2.test_lock.experiment_identity_already_used"
        ) from exc
    with descriptor:
        descriptor.write(
            canonical_json_bytes(
                {
                    "test_lock_contract_version": (
                        PHASE8B2_TEST_LOCK_CONTRACT_VERSION
                    ),
                    "authorization_stage": "consumed_pre_inference",
                    "single_use_identity": authorization[
                        "single_use_identity"
                    ],
                    "protocol_fingerprint": authorization[
                        "protocol_fingerprint"
                    ],
                    "selection_fingerprint": authorization[
                        "selection_fingerprint"
                    ],
                    "selected_variant_id": authorization[
                        "selected_variant_id"
                    ],
                    "selected_checkpoint": authorization[
                        "selected_checkpoint"
                    ],
                    "selected_checkpoint_sha256": authorization[
                        "selected_checkpoint_sha256"
                    ],
                    "selected_checkpoint_seed": authorization[
                        "selected_checkpoint_seed"
                    ],
                    "test_membership_fingerprint": authorization[
                        "test_membership_fingerprint"
                    ],
                    "test_membership_metadata_resolved": True,
                    "test_inference_performed": False,
                    "test_targets_accessed": False,
                    "test_metrics_accessed": False,
                },
                pretty=True,
            )
        )
        descriptor.flush()
    output = Path(output_value)
    try:
        output.mkdir(parents=False, exist_ok=False)
    except Exception:
        marker.unlink(missing_ok=True)
        raise


__all__ = [
    "PRIMARY_DATASETS",
    "TestUnlockRequest",
    "authorize_test_evaluation",
    "consume_test_authorization",
    "select_validation_checkpoint",
]
