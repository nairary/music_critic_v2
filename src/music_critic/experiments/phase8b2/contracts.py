"""Fail-closed contracts for Phase 8B.2A scientific comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Literal, Mapping, Sequence

from music_critic.models import ACTIVE_TASK_IDS
from music_critic.tasks import TARGET_ENCODING_BY_TASK, TARGET_FAMILY_BY_ID


PHASE8B2_COMPARISON_PROTOCOL_VERSION = "1.1.0"
PHASE8B2_ARTIFACT_CONTRACT_VERSION = "1.1.0"
PHASE8B2_COMPUTE_ACCOUNTING_VERSION = "1.1.0"
PHASE8B2_SELECTION_CONTRACT_VERSION = "1.1.0"
PHASE8B2_STATISTICS_CONTRACT_VERSION = "1.1.0"
PHASE8B2_DIAGNOSTICS_CONTRACT_VERSION = "1.1.0"
PHASE8B2_TEST_LOCK_CONTRACT_VERSION = "1.1.0"

ComparisonMode = Literal["natural_schedule", "encoder_forward_matched"]
TransferMode = Literal[
    "frozen_probe", "full_finetune", "supervised_scratch"
]

SSL_VARIANTS = (
    "phase7a_control",
    "phase8a_mask_only",
    "onset_latent",
    "beat_latent",
    "hierarchy_bar_latent",
    "track_latent",
    "multilevel_equal",
)
PRETRAINED_VARIANTS = SSL_VARIANTS
TRANSFER_MODES = (
    "frozen_probe",
    "full_finetune",
    "supervised_scratch",
)
PRIMARY_ARCHITECTURE = "hierarchical"
EXCLUDED_DOWNSTREAM_TASKS = {
    "theory.local_key.mode": "deferred_open_vocabulary",
    "theory.chord.borrowed": "deferred_open_vocabulary",
    "pop909_cl.chord.boundary": "positive_unlabeled",
    "pop909_cl.chord.no_chord": "positive_unlabeled",
}
DOWNSTREAM_TASK_IDS = tuple(ACTIVE_TASK_IDS)

_VARIANT_MODES = {
    "phase7a_control": ("phase7a_control", "phase7a_control"),
    "phase8a_mask_only": ("phase7a_control", "phase8a_mask_only"),
    "onset_latent": ("onset_only", "onset_only"),
    "beat_latent": ("beat_only", "beat_only"),
    "hierarchy_bar_latent": ("bar_only", "bar_only"),
    "track_latent": ("track_only", "track_only"),
    "multilevel_equal": (
        "multilevel_equal_weight",
        "multilevel_equal_weight",
    ),
}
_NATURAL_POLICY_VIEWS = {
    "phase7a_control": 1,
    "phase8a_mask_only": 4,
    "onset_latent": 1,
    "beat_latent": 1,
    "hierarchy_bar_latent": 1,
    "track_latent": 1,
    "multilevel_equal": 4,
}
_ENCODER_FORWARDS_PER_POLICY_VIEW = {
    "phase7a_control": 2,
    "phase8a_mask_only": 2,
    "onset_latent": 3,
    "beat_latent": 3,
    "hierarchy_bar_latent": 3,
    "track_latent": 3,
    "multilevel_equal": 3,
}


class Phase8B2ContractError(ValueError):
    """A stable Phase 8B.2A fail-closed contract violation."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase8B2ContractError(
            f"phase8b2.json.non_finite:{path}"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Phase8B2ContractError(
                    f"phase8b2.json.non_string_key:{path}"
                )
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    """Encode one finite JSON value using the Phase 8B.2A byte contract."""

    _validate_finite(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    """Return SHA-256 over compact canonical JSON without ambiguity."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def variant_modes(variant_id: str) -> tuple[str, str]:
    try:
        return _VARIANT_MODES[variant_id]
    except KeyError as exc:
        raise Phase8B2ContractError(
            f"phase8b2.protocol.variant_unknown:{variant_id}"
        ) from exc


def natural_policy_views(variant_id: str) -> int:
    variant_modes(variant_id)
    return _NATURAL_POLICY_VIEWS[variant_id]


def encoder_forwards_per_policy_view(variant_id: str) -> int:
    """Count every online and stop-gradient encoder invocation per view."""

    variant_modes(variant_id)
    return _ENCODER_FORWARDS_PER_POLICY_VIEW[variant_id]


def downstream_task_manifest(
    task_ids: Sequence[str] = DOWNSTREAM_TASK_IDS,
) -> tuple[dict[str, str], ...]:
    """Materialize the source-isolated fully supervised head set."""

    if len(task_ids) != len(set(task_ids)):
        raise Phase8B2ContractError(
            "phase8b2.protocol.downstream_task_duplicate"
        )
    rows: list[dict[str, str]] = []
    for task_id in task_ids:
        if task_id in EXCLUDED_DOWNSTREAM_TASKS:
            raise Phase8B2ContractError(
                f"phase8b2.protocol.downstream_task_excluded:{task_id}"
            )
        if task_id not in ACTIVE_TASK_IDS:
            raise Phase8B2ContractError(
                f"phase8b2.protocol.downstream_task_inactive:{task_id}"
            )
        encoding = TARGET_ENCODING_BY_TASK[task_id]
        family = TARGET_FAMILY_BY_ID[task_id]
        if not encoding.model_ready or encoding.supervision_regime != (
            "fully_supervised"
        ):
            raise Phase8B2ContractError(
                f"phase8b2.protocol.downstream_task_not_fully_supervised:{task_id}"
            )
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": family.source_adapter,
                "task_family": task_id.rsplit(".", 1)[0],
                "encoding_kind": encoding.encoding_kind,
            }
        )
    return tuple(sorted(rows, key=lambda row: row["task_id"]))


@dataclass(frozen=True, slots=True)
class DataBinding:
    """Exact corpus/cache/split and membership identity."""

    dataset_indices: tuple[tuple[str, str], ...]
    cache_identities: tuple[tuple[str, str], ...]
    split_manifest_fingerprint: str
    train_membership_fingerprint: str
    validation_membership_fingerprint: str
    test_membership_fingerprint: str
    mixture_weights: tuple[tuple[str, float], ...]
    workers: int
    actual_train_size: int
    actual_validation_size: int
    actual_test_size: int
    validation_subset_limit: int
    fixed_validation_seed: int

    def __post_init__(self) -> None:
        for name, rows in (
            ("dataset_indices", self.dataset_indices),
            ("cache_identities", self.cache_identities),
            ("mixture_weights", self.mixture_weights),
        ):
            keys = tuple(key for key, _ in rows)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise Phase8B2ContractError(
                    f"phase8b2.protocol.{name}_not_unique_sorted"
                )
        if not self.dataset_indices or not self.cache_identities:
            raise Phase8B2ContractError(
                "phase8b2.protocol.data_binding_empty"
            )
        if set(dict(self.dataset_indices)) != set(dict(self.cache_identities)):
            raise Phase8B2ContractError(
                "phase8b2.protocol.index_cache_dataset_mismatch"
            )
        if set(dict(self.mixture_weights)) != set(dict(self.dataset_indices)):
            raise Phase8B2ContractError(
                "phase8b2.protocol.mixture_dataset_mismatch"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for _, value in self.mixture_weights
        ):
            raise Phase8B2ContractError(
                "phase8b2.protocol.mixture_weight_invalid"
            )
        for name, value, minimum in (
            ("workers", self.workers, 0),
            ("actual_train_size", self.actual_train_size, 1),
            ("actual_validation_size", self.actual_validation_size, 1),
            ("actual_test_size", self.actual_test_size, 1),
            ("validation_subset_limit", self.validation_subset_limit, 0),
            ("fixed_validation_seed", self.fixed_validation_seed, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise Phase8B2ContractError(
                    f"phase8b2.protocol.data_runtime_invalid:{name}"
                )
        identities = (
            *(value for _, value in self.dataset_indices),
            *(value for _, value in self.cache_identities),
            self.split_manifest_fingerprint,
            self.train_membership_fingerprint,
            self.validation_membership_fingerprint,
            self.test_membership_fingerprint,
        )
        if any(not _is_sha256(value) for value in identities):
            raise Phase8B2ContractError(
                "phase8b2.protocol.data_fingerprint_invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_indices": [list(row) for row in self.dataset_indices],
            "cache_identities": [list(row) for row in self.cache_identities],
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "train_membership_fingerprint": (
                self.train_membership_fingerprint
            ),
            "validation_membership_fingerprint": (
                self.validation_membership_fingerprint
            ),
            "test_membership_fingerprint": self.test_membership_fingerprint,
            "mixture_weights": [list(row) for row in self.mixture_weights],
            "workers": self.workers,
            "actual_train_size": self.actual_train_size,
            "actual_validation_size": self.actual_validation_size,
            "actual_test_size": self.actual_test_size,
            "validation_subset_limit": self.validation_subset_limit,
            "fixed_validation_seed": self.fixed_validation_seed,
        }


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    """Budget fixed before a comparison run begins."""

    batch_size: int
    optimizer_steps: int
    raw_sample_exposures: int
    encoder_forwards_per_update: int | None
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "encoder_forwards_per_update" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Phase8B2ContractError(
                    f"phase8b2.protocol.compute_budget_invalid:{name}"
                )
        expected = self.batch_size * self.optimizer_steps
        if self.raw_sample_exposures != expected:
            raise Phase8B2ContractError(
                "phase8b2.protocol.raw_exposure_budget_inconsistent"
            )
        if self.gradient_accumulation_steps != 1:
            raise Phase8B2ContractError(
                "phase8b2.protocol.gradient_accumulation_unsupported"
            )

    @property
    def encoder_forward_count(self) -> int | None:
        if self.encoder_forwards_per_update is None:
            return None
        return self.optimizer_steps * self.encoder_forwards_per_update

    def to_dict(self) -> dict[str, int | None]:
        result = asdict(self)
        result["encoder_forward_count"] = self.encoder_forward_count
        return result


@dataclass(frozen=True, slots=True)
class ComparisonProtocol:
    """Complete binding for one compatible Phase 8B.2A experiment family."""

    protocol_version: str
    comparison_mode: ComparisonMode
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    encoder_model_config: Mapping[str, object]
    ssl_objective_config: Mapping[str, object]
    masking_policy_config: Mapping[str, object]
    data: DataBinding
    compute: ComputeBudget
    optimizer_config: Mapping[str, object]
    scheduler_config: Mapping[str, object]
    amp_device_config: Mapping[str, object]
    transfer_modes: tuple[str, ...]
    downstream_tasks: tuple[dict[str, str], ...]
    validation_selection_rule: Mapping[str, object]
    test_unlock_state: Mapping[str, object]
    downstream_optimizer_steps: int
    downstream_schedule_fingerprint: str
    ssl_sample_schedule_fingerprints: tuple[tuple[int, str], ...]
    downstream_sample_schedule_fingerprints: tuple[tuple[int, str], ...]
    runtime_execution_config: Mapping[str, object]
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.protocol_version != PHASE8B2_COMPARISON_PROTOCOL_VERSION:
            raise Phase8B2ContractError(
                "phase8b2.protocol.version_incompatible"
            )
        if self.comparison_mode not in {
            "natural_schedule",
            "encoder_forward_matched",
        }:
            raise Phase8B2ContractError(
                "phase8b2.protocol.comparison_mode_invalid"
            )
        if (
            not self.variants
            or len(self.variants) != len(set(self.variants))
            or any(variant not in SSL_VARIANTS for variant in self.variants)
        ):
            raise Phase8B2ContractError(
                "phase8b2.protocol.variant_manifest_invalid"
            )
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                for seed in self.seeds
            )
        ):
            raise Phase8B2ContractError(
                "phase8b2.protocol.seed_manifest_invalid"
            )
        if tuple(sorted(set(self.transfer_modes))) != tuple(
            sorted(self.transfer_modes)
        ) or any(mode not in TRANSFER_MODES for mode in self.transfer_modes):
            raise Phase8B2ContractError(
                "phase8b2.protocol.transfer_modes_invalid"
            )
        if "supervised_scratch" not in self.transfer_modes:
            raise Phase8B2ContractError(
                "phase8b2.protocol.scratch_control_required"
            )
        expected_tasks = downstream_task_manifest(
            tuple(row["task_id"] for row in self.downstream_tasks)
        )
        if self.downstream_tasks != expected_tasks:
            raise Phase8B2ContractError(
                "phase8b2.protocol.downstream_task_manifest_invalid"
            )
        if self.downstream_optimizer_steps <= 0:
            raise Phase8B2ContractError(
                "phase8b2.protocol.downstream_budget_invalid"
            )
        if not self.downstream_schedule_fingerprint:
            raise Phase8B2ContractError(
                "phase8b2.protocol.downstream_schedule_missing"
            )
        expected_seed_rows = tuple(sorted(self.seeds))
        for name, rows in (
            (
                "ssl_sample_schedule_fingerprints",
                self.ssl_sample_schedule_fingerprints,
            ),
            (
                "downstream_sample_schedule_fingerprints",
                self.downstream_sample_schedule_fingerprints,
            ),
        ):
            if (
                tuple(seed for seed, _value in rows) != expected_seed_rows
                or any(not _is_sha256(value) for _seed, value in rows)
            ):
                raise Phase8B2ContractError(
                    f"phase8b2.protocol.{name}_invalid"
                )
        if self.comparison_mode == "encoder_forward_matched":
            target = self.compute.encoder_forwards_per_update
            if target is None:
                raise Phase8B2ContractError(
                    "phase8b2.protocol.matched_forward_budget_missing"
                )
            if any(
                target % encoder_forwards_per_policy_view(variant) != 0
                or target // encoder_forwards_per_policy_view(variant)
                < natural_policy_views(variant)
                for variant in self.variants
            ):
                raise Phase8B2ContractError(
                    "phase8b2.protocol.compute_budget_unmatchable"
                )
        elif self.compute.encoder_forwards_per_update is not None:
            raise Phase8B2ContractError(
                "phase8b2.protocol.natural_schedule_forward_target_forbidden"
            )
        expected = fingerprint(self.binding_dict())
        if self.fingerprint and self.fingerprint != expected:
            raise Phase8B2ContractError(
                "phase8b2.protocol.fingerprint_mismatch"
            )
        object.__setattr__(self, "fingerprint", expected)

    def binding_dict(self) -> dict[str, object]:
        """Return every field whose mutation changes protocol identity."""

        return {
            "protocol_contract": (
                f"Phase8B2ComparisonProtocol@{self.protocol_version}"
            ),
            "comparison_mode": self.comparison_mode,
            "variants": list(self.variants),
            "variant_bindings": {
                variant: {
                    "objective_mode": variant_modes(variant)[0],
                    "masking_mode": variant_modes(variant)[1],
                    "natural_policy_views_per_update": (
                        natural_policy_views(variant)
                    ),
                    "encoder_forwards_per_policy_view": (
                        encoder_forwards_per_policy_view(variant)
                    ),
                }
                for variant in self.variants
            },
            "seeds": list(self.seeds),
            "encoder_model_config": dict(self.encoder_model_config),
            "ssl_objective_config": dict(self.ssl_objective_config),
            "masking_policy_config": dict(self.masking_policy_config),
            "data": self.data.to_dict(),
            "compute": self.compute.to_dict(),
            "optimizer_config": dict(self.optimizer_config),
            "scheduler_config": dict(self.scheduler_config),
            "amp_device_config": dict(self.amp_device_config),
            "transfer_modes": list(self.transfer_modes),
            "downstream_tasks": list(self.downstream_tasks),
            "downstream_optimizer_steps": self.downstream_optimizer_steps,
            "downstream_schedule_fingerprint": (
                self.downstream_schedule_fingerprint
            ),
            "ssl_sample_schedule_fingerprints": [
                [seed, value]
                for seed, value in self.ssl_sample_schedule_fingerprints
            ],
            "downstream_sample_schedule_fingerprints": [
                [seed, value]
                for seed, value in self.downstream_sample_schedule_fingerprints
            ],
            "runtime_execution_config": dict(self.runtime_execution_config),
            "validation_selection_rule": dict(
                self.validation_selection_rule
            ),
            "test_unlock_state": dict(self.test_unlock_state),
        }

    def to_dict(self) -> dict[str, object]:
        result = self.binding_dict()
        result["fingerprint"] = self.fingerprint
        return result

    def with_fingerprint_verified(self, expected: str) -> ComparisonProtocol:
        if expected != self.fingerprint:
            raise Phase8B2ContractError(
                "phase8b2.protocol.resume_fingerprint_mismatch"
            )
        return replace(self, fingerprint=expected)


def default_selection_rule() -> dict[str, object]:
    return {
        "contract_version": PHASE8B2_SELECTION_CONTRACT_VERSION,
        "split": "validation",
        "primary_endpoints": [
            "hooktheory.dataset_macro_summary",
            "pop909_cl.dataset_macro_summary",
        ],
        "rank_rule": "mean_dataset_rank",
        "tie_breaks": [
            "lower_validation_nll",
            "lower_encoder_forward_count",
            "lexicographic_configuration_id",
        ],
        "identity": ["variant_id", "transfer_mode"],
        "seed_aggregation": "paired_arithmetic_mean",
        "diagnostics_in_selection": False,
        "test_metrics_in_selection": False,
    }


def locked_test_state() -> dict[str, object]:
    return {
        "contract_version": PHASE8B2_TEST_LOCK_CONTRACT_VERSION,
        "unlocked": False,
        "acknowledged": False,
        "single_use": True,
    }


__all__ = [
    "DOWNSTREAM_TASK_IDS",
    "EXCLUDED_DOWNSTREAM_TASKS",
    "PHASE8B2_ARTIFACT_CONTRACT_VERSION",
    "PHASE8B2_COMPARISON_PROTOCOL_VERSION",
    "PHASE8B2_COMPUTE_ACCOUNTING_VERSION",
    "PHASE8B2_DIAGNOSTICS_CONTRACT_VERSION",
    "PHASE8B2_SELECTION_CONTRACT_VERSION",
    "PHASE8B2_STATISTICS_CONTRACT_VERSION",
    "PHASE8B2_TEST_LOCK_CONTRACT_VERSION",
    "PRETRAINED_VARIANTS",
    "PRIMARY_ARCHITECTURE",
    "SSL_VARIANTS",
    "TRANSFER_MODES",
    "ComparisonMode",
    "ComparisonProtocol",
    "ComputeBudget",
    "DataBinding",
    "Phase8B2ContractError",
    "TransferMode",
    "canonical_json_bytes",
    "default_selection_rule",
    "downstream_task_manifest",
    "encoder_forwards_per_policy_view",
    "fingerprint",
    "locked_test_state",
    "natural_policy_views",
    "variant_modes",
]
