"""Public Phase 9C-A API with lazy imports for isolated worker startup."""

from __future__ import annotations

from importlib import import_module

from .contracts import (
    ACTIONS,
    ALL_VARIANTS,
    CLAIM_BOUNDARIES,
    DOWNSTREAM_MODES,
    OPTIONAL_VARIANTS,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PHASE9C_PROTOCOL_VERSION,
    PHASE9C_SEED,
    PRESETS,
    PRIMARY_VARIANTS,
    Phase9CContractError,
    TASK_IDS,
    resolve_preset,
)

_LAZY = {
    "build_experiment_plan": ("planner", "build_experiment_plan"),
    "compose_ssl_split_manifest": ("planner", "compose_ssl_split_manifest"),
    "materialize_ssl_split_manifest": ("planner", "materialize_ssl_split_manifest"),
    "build_source_balanced_schedule": ("sampling", "build_source_balanced_schedule"),
    "component_bootstrap_primary_delta": ("metrics", "component_bootstrap_primary_delta"),
    "create_evidence_tar": ("artifacts", "create_evidence_tar"),
    "execute_experiment": ("runner", "execute_experiment"),
    "primary_validation_summary": ("metrics", "primary_validation_summary"),
    "profile_experiment": ("runner", "profile_experiment"),
    "safe_extract_members": ("artifacts", "safe_extract_members"),
    "verify_bundle": ("artifacts", "verify_bundle"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ACTIONS",
    "ALL_VARIANTS",
    "CLAIM_BOUNDARIES",
    "DOWNSTREAM_MODES",
    "OPTIONAL_VARIANTS",
    "PHASE9C_ENCODER_FORWARDS_PER_UPDATE",
    "PHASE9C_PROTOCOL_VERSION",
    "PHASE9C_SEED",
    "PRESETS",
    "PRIMARY_VARIANTS",
    "Phase9CContractError",
    "TASK_IDS",
    "build_experiment_plan",
    "compose_ssl_split_manifest",
    "build_source_balanced_schedule",
    "component_bootstrap_primary_delta",
    "create_evidence_tar",
    "execute_experiment",
    "primary_validation_summary",
    "profile_experiment",
    "materialize_ssl_split_manifest",
    "resolve_preset",
    "safe_extract_members",
    "verify_bundle",
]
