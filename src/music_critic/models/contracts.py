"""Versioned contracts for the Phase 6A trainable local baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Literal

from music_critic.tasks import TARGET_ENCODING_BY_TASK, TARGET_FAMILY_BY_ID


MODEL_CONTRACT_VERSION = "1.0.0"
ENCODER_OUTPUT_VERSION = "1.0.0"
BASELINE_LOSS_CONTRACT_VERSION = "1.0.0"
RAW_RECONSTRUCTION_CONTRACT_VERSION = "1.0.0"
CHECKPOINT_CONTRACT_VERSION = "1.0.0"

ModelVariant = Literal["feature_only", "local_gnn"]

ACTIVE_TASK_IDS = (
    "pop909_cl.chord.bass",
    "pop909_cl.chord.inversion",
    "pop909_cl.chord.quality",
    "pop909_cl.chord.root",
    "theory.chord.adds",
    "theory.chord.alterations",
    "theory.chord.extent",
    "theory.chord.inversion",
    "theory.chord.omits",
    "theory.chord.presence",
    "theory.chord.root_degree",
    "theory.chord.suspensions",
    "theory.local_key.tonic_pc",
    "theory.melody.scale_degree",
)
EXCLUDED_TASK_REASONS = MappingProxyType(
    {
        "pop909_cl.chord.boundary": "positive_unlabeled",
        "pop909_cl.chord.no_chord": "positive_unlabeled",
        "theory.chord.borrowed": "deferred_open_vocabulary",
        "theory.local_key.mode": "deferred_open_vocabulary",
    }
)


@dataclass(frozen=True, slots=True)
class LocalBaselineConfig:
    """Configuration shared by feature-only and relation-aware variants."""

    variant: ModelVariant = "local_gnn"
    hidden_dim: int = 128
    gnn_layers: int = 3
    dropout: float = 0.1
    residual: bool = True
    activation: Literal["gelu"] = "gelu"
    normalization: Literal["layer_norm"] = "layer_norm"
    relation_aggregation: Literal["sum"] = "sum"
    task_hidden_dim: int | None = None
    task_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in {"feature_only", "local_gnn"}:
            raise ValueError("variant must be feature_only or local_gnn")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if (
            isinstance(self.gnn_layers, bool)
            or not isinstance(self.gnn_layers, int)
            or self.gnn_layers < 0
        ):
            raise ValueError("gnn_layers must be a non-negative integer")
        if self.variant == "local_gnn" and self.gnn_layers == 0:
            raise ValueError("local_gnn requires at least one message-passing layer")
        if not isinstance(self.dropout, (int, float)) or isinstance(
            self.dropout, bool
        ) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.task_hidden_dim is not None and (
            isinstance(self.task_hidden_dim, bool)
            or not isinstance(self.task_hidden_dim, int)
            or self.task_hidden_dim <= 0
        ):
            raise ValueError("task_hidden_dim must be null or a positive integer")
        keys = tuple(task_id for task_id, _ in self.task_weights)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("task weights must be uniquely sorted by task ID")
        for task_id, weight in self.task_weights:
            if (
                task_id not in ACTIVE_TASK_IDS
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("task weights require active tasks and positive values")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["task_weights"] = [list(item) for item in self.task_weights]
        return result


@dataclass(frozen=True, slots=True)
class TaskHeadSpec:
    """One actually instantiated source-native Phase 6A output head."""

    task_id: str
    source_adapter: str
    encoding_kind: str
    output_dim: int
    node_types: tuple[str, ...]
    supervision_regime: Literal["fully_supervised"] = "fully_supervised"

    def __post_init__(self) -> None:
        if self.task_id not in ACTIVE_TASK_IDS:
            raise ValueError("Phase 6A head is not in the accepted active task set")
        family = TARGET_FAMILY_BY_ID[self.task_id]
        encoding = TARGET_ENCODING_BY_TASK[self.task_id]
        if (
            self.source_adapter != family.source_adapter
            or self.encoding_kind != encoding.encoding_kind
            or encoding.supervision_regime != "fully_supervised"
            or not encoding.model_ready
            or self.output_dim != len(encoding.vocabulary or ())
            or self.node_types != family.alignment_policy.candidate_node_types
        ):
            raise ValueError("task head spec differs from ontology/encoding contracts")


def active_task_head_specs() -> tuple[TaskHeadSpec, ...]:
    """Build the ordered registry directly from accepted target contracts."""

    return tuple(
        TaskHeadSpec(
            task_id=task_id,
            source_adapter=TARGET_FAMILY_BY_ID[task_id].source_adapter,
            encoding_kind=TARGET_ENCODING_BY_TASK[task_id].encoding_kind,
            output_dim=len(TARGET_ENCODING_BY_TASK[task_id].vocabulary or ()),
            node_types=TARGET_FAMILY_BY_ID[
                task_id
            ].alignment_policy.candidate_node_types,
        )
        for task_id in ACTIVE_TASK_IDS
    )


__all__ = [
    "ACTIVE_TASK_IDS",
    "BASELINE_LOSS_CONTRACT_VERSION",
    "CHECKPOINT_CONTRACT_VERSION",
    "ENCODER_OUTPUT_VERSION",
    "EXCLUDED_TASK_REASONS",
    "MODEL_CONTRACT_VERSION",
    "RAW_RECONSTRUCTION_CONTRACT_VERSION",
    "LocalBaselineConfig",
    "ModelVariant",
    "TaskHeadSpec",
    "active_task_head_specs",
]
