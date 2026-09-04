"""Supervised baseline evaluation and performance evidence."""

from music_critic.evaluation.checkpoint import load_evaluation_checkpoint
from music_critic.evaluation.config import (
    EvaluationConfig,
    EvaluationDataConfig,
    EvaluationDeviceConfig,
    ProfilerConfig,
    register_evaluation_configs,
)
from music_critic.evaluation.contracts import (
    EVALUATION_ARTIFACT_VERSION,
    EVALUATION_CONTRACT_VERSION,
    MACRO_SUMMARY_CONTRACT_VERSION,
    PROFILER_CONTRACT_VERSION,
    TRAIN_PRIOR_CONTRACT_VERSION,
    EvaluationContractError,
)
from music_critic.evaluation.engine import (
    build_macro_summaries,
    run_evaluation,
)
from music_critic.evaluation.dilemmadata import (
    DILEMMADATA_COMPONENT_BOOTSTRAP_VERSION,
    DILEMMADATA_EVALUATION_CONTRACT_VERSION,
    DILEMMADATA_TEST_LOCK_VERSION,
    DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION,
    DilemmadataEvaluationError,
    build_dilemmadata_train_priors,
    evaluate_dilemmadata_model,
    make_dilemmadata_test_unlock,
    paired_component_bootstrap,
    validate_dilemmadata_train_priors,
)
from music_critic.evaluation.metrics import (
    CategoricalMetricAccumulator,
    MultilabelMetricAccumulator,
)
from music_critic.evaluation.priors import (
    TrainPriorBuilder,
    TrivialBaselineAccumulator,
    validate_train_priors,
)
from music_critic.evaluation.profiler import run_profiler


__all__ = [
    "EVALUATION_ARTIFACT_VERSION",
    "EVALUATION_CONTRACT_VERSION",
    "MACRO_SUMMARY_CONTRACT_VERSION",
    "PROFILER_CONTRACT_VERSION",
    "TRAIN_PRIOR_CONTRACT_VERSION",
    "CategoricalMetricAccumulator",
    "DILEMMADATA_COMPONENT_BOOTSTRAP_VERSION",
    "DILEMMADATA_EVALUATION_CONTRACT_VERSION",
    "DILEMMADATA_TEST_LOCK_VERSION",
    "DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION",
    "DilemmadataEvaluationError",
    "EvaluationConfig",
    "EvaluationContractError",
    "EvaluationDataConfig",
    "EvaluationDeviceConfig",
    "ProfilerConfig",
    "MultilabelMetricAccumulator",
    "TrainPriorBuilder",
    "TrivialBaselineAccumulator",
    "load_evaluation_checkpoint",
    "build_macro_summaries",
    "build_dilemmadata_train_priors",
    "evaluate_dilemmadata_model",
    "make_dilemmadata_test_unlock",
    "paired_component_bootstrap",
    "register_evaluation_configs",
    "run_evaluation",
    "run_profiler",
    "validate_train_priors",
    "validate_dilemmadata_train_priors",
]
