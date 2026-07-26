"""Trainable local raw-graph baselines for Music Critic Phase 6A."""

from music_critic.models.baseline import (
    BaselineOutput,
    LocalHeterogeneousBaseline,
)
from music_critic.models.checkpoint import (
    CheckpointContractError,
    checkpoint_metadata,
    feature_registry_fingerprint,
    load_baseline_checkpoint,
    save_baseline_checkpoint,
)
from music_critic.models.contracts import (
    ACTIVE_TASK_IDS,
    BASELINE_LOSS_CONTRACT_VERSION,
    CHECKPOINT_CONTRACT_VERSION,
    ENCODER_OUTPUT_VERSION,
    EXCLUDED_TASK_REASONS,
    MODEL_CONTRACT_VERSION,
    RAW_RECONSTRUCTION_CONTRACT_VERSION,
    LocalBaselineConfig,
    TaskHeadSpec,
    active_task_head_specs,
)
from music_critic.models.diagnostics import (
    EmbeddingDelta,
    OversmoothingValue,
    SingleNoteDiagnostic,
    single_note_sensitivity,
)
from music_critic.models.encoder import (
    EncoderOutput,
    LocalHeterogeneousEncoder,
    LocalRelationLayer,
    MultiScaleEncoderOutput,
    RawFeatureEncoder,
    normalize_continuous,
)
from music_critic.models.heads import (
    BaselineLossReport,
    LossGroup,
    SourceNativeTaskHeads,
    TaskLoss,
    TaskOutput,
    aggregate_task_losses,
)
from music_critic.models.reconstruction import (
    RECONSTRUCTION_FIELDS,
    RawReconstructionHeads,
    ReconstructionOutput,
    reconstruction_loss,
)

__all__ = [
    "ACTIVE_TASK_IDS",
    "BASELINE_LOSS_CONTRACT_VERSION",
    "BaselineLossReport",
    "BaselineOutput",
    "CHECKPOINT_CONTRACT_VERSION",
    "CheckpointContractError",
    "ENCODER_OUTPUT_VERSION",
    "EXCLUDED_TASK_REASONS",
    "EmbeddingDelta",
    "EncoderOutput",
    "LocalBaselineConfig",
    "LocalHeterogeneousBaseline",
    "LocalHeterogeneousEncoder",
    "LocalRelationLayer",
    "LossGroup",
    "MODEL_CONTRACT_VERSION",
    "MultiScaleEncoderOutput",
    "OversmoothingValue",
    "RAW_RECONSTRUCTION_CONTRACT_VERSION",
    "RECONSTRUCTION_FIELDS",
    "RawFeatureEncoder",
    "RawReconstructionHeads",
    "ReconstructionOutput",
    "SingleNoteDiagnostic",
    "SourceNativeTaskHeads",
    "TaskHeadSpec",
    "TaskLoss",
    "TaskOutput",
    "active_task_head_specs",
    "aggregate_task_losses",
    "checkpoint_metadata",
    "feature_registry_fingerprint",
    "load_baseline_checkpoint",
    "normalize_continuous",
    "reconstruction_loss",
    "save_baseline_checkpoint",
    "single_note_sensitivity",
]
