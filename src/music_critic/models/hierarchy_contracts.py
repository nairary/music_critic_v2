"""Versioned contracts for the Phase 6B hierarchy baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from music_critic.models.contracts import LocalBaselineConfig


HIERARCHY_POOLING_CONTRACT_VERSION = "1.0.0"
COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION = "1.0.0"
HIERARCHICAL_ENCODER_OUTPUT_VERSION = "1.0.0"
TOP_DOWN_FUSION_CONTRACT_VERSION = "1.0.0"
HIERARCHICAL_MODEL_CONTRACT_VERSION = "1.0.0"
HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class HierarchicalBaselineConfig:
    """Complete Phase 6B architecture and inherited Phase 6A configuration."""

    hidden_dim: int = 128
    local_gnn_layers: int = 3
    transformer_layers: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    local_residual: bool = True
    task_hidden_dim: int | None = None
    task_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        integer_fields = (
            ("hidden_dim", self.hidden_dim, 1),
            ("local_gnn_layers", self.local_gnn_layers, 1),
            ("transformer_layers", self.transformer_layers, 1),
            ("attention_heads", self.attention_heads, 1),
            ("ffn_multiplier", self.ffn_multiplier, 1),
        )
        for name, value, minimum in integer_fields:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.attention_heads:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads"
            )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must lie in [0, 1)")
        # Reuse Phase 6A validation for task dimensions, weights, and local
        # encoder settings without changing its public contract.
        self.local_config()

    def local_config(self) -> LocalBaselineConfig:
        return LocalBaselineConfig(
            variant="local_gnn",
            hidden_dim=self.hidden_dim,
            gnn_layers=self.local_gnn_layers,
            dropout=self.dropout,
            residual=self.local_residual,
            task_hidden_dim=self.task_hidden_dim,
            task_weights=self.task_weights,
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["task_weights"] = [list(item) for item in self.task_weights]
        return result


__all__ = [
    "COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION",
    "HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION",
    "HIERARCHICAL_ENCODER_OUTPUT_VERSION",
    "HIERARCHICAL_MODEL_CONTRACT_VERSION",
    "HIERARCHY_POOLING_CONTRACT_VERSION",
    "TOP_DOWN_FUSION_CONTRACT_VERSION",
    "HierarchicalBaselineConfig",
]
