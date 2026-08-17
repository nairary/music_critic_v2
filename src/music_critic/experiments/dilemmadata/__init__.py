"""Immutable Phase 9B.2B scratch-vs-SSL experiment plans."""

from music_critic.experiments.dilemmadata.contracts import (
    DILEMMADATA_EXPERIMENT_PLAN_VERSION,
    DILEMMADATA_PRIMARY_VARIANTS,
    DILEMMADATA_REPORT_BUNDLE_VERSION,
    DILEMMADATA_SEEDS,
    DilemmadataExperimentPlanError,
    build_dilemmadata_experiment_plan,
    dilemmadata_command_matrix,
    dilemmadata_report_bundle_manifest,
    validate_dilemmadata_command_matrix,
    validate_dilemmadata_experiment_plan,
    verify_dilemmadata_report_bundle,
)

__all__ = [
    "DILEMMADATA_EXPERIMENT_PLAN_VERSION",
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
