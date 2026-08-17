"""Immutable contracts for the Phase 9C-A one-seed production pilot."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


PHASE9C_PROTOCOL_VERSION = "1.0.0"
PHASE9C_PLAN_VERSION = "1.0.0"
PHASE9C_ARTIFACT_VERSION = "1.0.0"
PHASE9C_PROFILE_VERSION = "1.0.0"
PHASE9C_SELECTION_VERSION = "1.0.0"
PHASE9C_TEST_LOCK_VERSION = "1.0.0"
PHASE9C_SEED = 17
PHASE9C_ENCODER_FORWARDS_PER_UPDATE = 12

PRIMARY_VARIANTS = (
    "scratch",
    "phase7a_control",
    "phase8a_mask_only",
    "multilevel_equal",
)
SSL_PRIMARY_VARIANTS = PRIMARY_VARIANTS[1:]
OPTIONAL_VARIANTS = (
    "onset_latent",
    "beat_latent",
    "hierarchy_bar_latent",
    "track_latent",
)
ALL_VARIANTS = PRIMARY_VARIANTS + OPTIONAL_VARIANTS
DOWNSTREAM_MODES = ("frozen_probe", "full_finetune")
TASK_IDS = (
    "dilemmadata.an.chord.inversion",
    "dilemmadata.an.chord.quality",
    "dilemmadata.dlc.chord.inversion",
    "dilemmadata.dlc.chord.quality",
)
PRESETS = (
    "bounded_acceptance",
    "rtx_profile",
    "one_seed_primary_pilot",
    "one_seed_full_ablation",
)
ACTIONS = ("plan", "profile", "run", "resume", "aggregate", "select", "verify")


class Phase9CContractError(ValueError):
    """Stable fail-closed Phase 9C-A contract error."""


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


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def locked_test_state() -> dict[str, object]:
    payload = {
        "contract_version": PHASE9C_TEST_LOCK_VERSION,
        "split": "test",
        "test_inference": False,
        "test_targets_accessed": False,
        "test_metrics_computed": False,
        "test_unlock_used": False,
        "full_test_identities_serialized": False,
        "permitted_actions": [
            "plan",
            "profile",
            "run",
            "resume",
            "aggregate",
            "select",
            "verify",
        ],
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def validate_test_lock(value: Mapping[str, object]) -> None:
    payload = dict(value)
    observed = payload.pop("fingerprint", None)
    if (
        observed != fingerprint(payload)
        or payload.get("contract_version") != PHASE9C_TEST_LOCK_VERSION
        or any(
            payload.get(name) is not False
            for name in (
                "test_inference",
                "test_targets_accessed",
                "test_metrics_computed",
                "test_unlock_used",
                "full_test_identities_serialized",
            )
        )
    ):
        raise Phase9CContractError("phase9c.test_lock.invalid")


@dataclass(frozen=True, slots=True)
class PilotPreset:
    name: str
    variants: tuple[str, ...]
    ssl_updates: int | None
    downstream_epochs: int | None
    downstream_steps_per_epoch: int | None
    batch_size: int | None
    bootstrap_replicates: int
    profile_only: bool = False

    def __post_init__(self) -> None:
        if self.name not in PRESETS:
            raise Phase9CContractError("phase9c.preset.unknown")
        if self.variants != tuple(dict.fromkeys(self.variants)) or any(
            variant not in ALL_VARIANTS for variant in self.variants
        ):
            raise Phase9CContractError("phase9c.preset.variants_invalid")
        if self.name == "one_seed_primary_pilot" and self.variants != PRIMARY_VARIANTS:
            raise Phase9CContractError("phase9c.preset.primary_matrix_changed")
        if self.name != "one_seed_full_ablation" and any(
            variant in OPTIONAL_VARIANTS for variant in self.variants
        ):
            raise Phase9CContractError("phase9c.preset.optional_ablation_implicit")
        for value in (
            self.ssl_updates,
            self.downstream_epochs,
            self.downstream_steps_per_epoch,
            self.batch_size,
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise Phase9CContractError("phase9c.preset.budget_invalid")
        if self.bootstrap_replicates <= 0:
            raise Phase9CContractError("phase9c.preset.bootstrap_invalid")
        if self.name.startswith("one_seed_") and self.bootstrap_replicates < 1000:
            raise Phase9CContractError("phase9c.preset.production_bootstrap_too_small")

    @property
    def production_budget_resolved(self) -> bool:
        return all(
            value is not None
            for value in (
                self.ssl_updates,
                self.downstream_epochs,
                self.downstream_steps_per_epoch,
                self.batch_size,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "variants": list(self.variants),
            "ssl_updates": self.ssl_updates,
            "downstream_epochs": self.downstream_epochs,
            "downstream_steps_per_epoch": self.downstream_steps_per_epoch,
            "batch_size": self.batch_size,
            "bootstrap_replicates": self.bootstrap_replicates,
            "profile_only": self.profile_only,
            "production_budget_resolved": self.production_budget_resolved,
        }


def resolve_preset(
    name: str,
    *,
    ssl_updates: int | None = None,
    downstream_epochs: int | None = None,
    downstream_steps_per_epoch: int | None = None,
    batch_size: int | None = None,
    bootstrap_replicates: int | None = None,
) -> PilotPreset:
    if name == "bounded_acceptance":
        return PilotPreset(
            name=name,
            variants=PRIMARY_VARIANTS,
            ssl_updates=ssl_updates or 1,
            downstream_epochs=downstream_epochs or 1,
            downstream_steps_per_epoch=downstream_steps_per_epoch or 1,
            batch_size=batch_size or 2,
            bootstrap_replicates=bootstrap_replicates or 32,
        )
    if name == "rtx_profile":
        return PilotPreset(
            name=name,
            variants=PRIMARY_VARIANTS,
            ssl_updates=ssl_updates or 3,
            downstream_epochs=downstream_epochs or 1,
            downstream_steps_per_epoch=downstream_steps_per_epoch or 3,
            batch_size=batch_size,
            bootstrap_replicates=bootstrap_replicates or 32,
            profile_only=True,
        )
    if name == "one_seed_primary_pilot":
        variants = PRIMARY_VARIANTS
    elif name == "one_seed_full_ablation":
        variants = ALL_VARIANTS
    else:
        raise Phase9CContractError("phase9c.preset.unknown")
    return PilotPreset(
        name=name,
        variants=variants,
        ssl_updates=ssl_updates,
        downstream_epochs=downstream_epochs,
        downstream_steps_per_epoch=downstream_steps_per_epoch,
        batch_size=batch_size,
        bootstrap_replicates=bootstrap_replicates or 2000,
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    payload = dict(protocol)
    observed = payload.pop("fingerprint", None)
    if observed != fingerprint(payload):
        raise Phase9CContractError("phase9c.protocol.fingerprint_mismatch")
    if (
        payload.get("contract_version") != PHASE9C_PROTOCOL_VERSION
        or payload.get("phase") != "9C-A"
        or payload.get("seed") != PHASE9C_SEED
        or payload.get("primary_variants") != list(PRIMARY_VARIANTS)
        or payload.get("task_ids") != list(TASK_IDS)
        or payload.get("selection", {}).get("primary_metric")
        != "mean_task_nll_div_log_class_count"
        or payload.get("compute", {}).get("encoder_forwards_per_logical_update")
        != PHASE9C_ENCODER_FORWARDS_PER_UPDATE
    ):
        raise Phase9CContractError("phase9c.protocol.invalid")
    validate_test_lock(payload.get("test_lock", {}))


CLAIM_BOUNDARIES = {
    "permitted": [
        "production pipeline executable",
        "one-seed exploratory validation comparison",
        "observed validation difference relative to scratch",
        "compute and VRAM evidence",
    ],
    "forbidden": [
        "test quality",
        "statistical superiority",
        "generalization superiority",
        "final SSL benefit",
        "paper-level significance",
        "PDMX-scale evidence",
        "complete music critic readiness",
    ],
    "bootstrap_interpretation": (
        "Component bootstrap reflects validation-sample uncertainty only; "
        "it does not measure optimization-seed uncertainty and is not a final "
        "significance claim."
    ),
}


__all__ = [
    "ACTIONS",
    "ALL_VARIANTS",
    "CLAIM_BOUNDARIES",
    "DOWNSTREAM_MODES",
    "OPTIONAL_VARIANTS",
    "PHASE9C_ARTIFACT_VERSION",
    "PHASE9C_ENCODER_FORWARDS_PER_UPDATE",
    "PHASE9C_PLAN_VERSION",
    "PHASE9C_PROFILE_VERSION",
    "PHASE9C_PROTOCOL_VERSION",
    "PHASE9C_SEED",
    "PHASE9C_SELECTION_VERSION",
    "PRESETS",
    "PRIMARY_VARIANTS",
    "Phase9CContractError",
    "PilotPreset",
    "SSL_PRIMARY_VARIANTS",
    "TASK_IDS",
    "canonical_bytes",
    "fingerprint",
    "is_sha256",
    "locked_test_state",
    "resolve_preset",
    "validate_protocol",
    "validate_test_lock",
]
