"""Model selection without changing Phase 6A/6B runtime semantics."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from torch import nn

from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    checkpoint_metadata,
    hierarchical_checkpoint_metadata,
)
from music_critic.training.config import ModelConfig


BaselineModel = LocalHeterogeneousBaseline | HierarchicalHeterogeneousBaseline


def build_baseline_model(config: ModelConfig | Any) -> BaselineModel:
    """Select one accepted baseline using its existing configuration."""

    if config.name in {"feature_only", "local_gnn"}:
        return LocalHeterogeneousBaseline(
            LocalBaselineConfig(
                variant=config.name,
                hidden_dim=config.hidden_dim,
                gnn_layers=config.local_gnn_layers,
                dropout=config.dropout,
                residual=config.residual,
            )
        )
    if config.name == "hierarchical":
        return HierarchicalHeterogeneousBaseline(
            HierarchicalBaselineConfig(
                hidden_dim=config.hidden_dim,
                local_gnn_layers=config.local_gnn_layers,
                transformer_layers=config.transformer_layers,
                attention_heads=config.attention_heads,
                ffn_multiplier=config.ffn_multiplier,
                dropout=config.dropout,
                local_residual=config.residual,
            )
        )
    raise ValueError(f"training.model.unknown:{config.name}")


def model_contract_metadata(model: nn.Module) -> dict[str, object]:
    if isinstance(model, HierarchicalHeterogeneousBaseline):
        return hierarchical_checkpoint_metadata(model)
    if isinstance(model, LocalHeterogeneousBaseline):
        return checkpoint_metadata(model)
    raise TypeError("training model is not an accepted baseline")


def model_contract_fingerprint(model: nn.Module) -> str:
    payload = json.dumps(
        model_contract_metadata(model),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BaselineModel",
    "build_baseline_model",
    "model_contract_fingerprint",
    "model_contract_metadata",
]
