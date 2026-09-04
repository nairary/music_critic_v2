"""Loss, uncertainty weighting, and applied-update schedule for Phase 9E-B1."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from music_critic.experiments.analysisgnn.contracts import (
    COMMON_BENCHMARK_CONFIG,
    Phase9EB1Config,
)


class TwoTaskUncertaintyLoss(nn.Module):
    """Public AnalysisGNN two-task weighting with explicit missing-label masks."""

    TASKS = ("quality", "inversion")

    def __init__(self, config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.scales = nn.Parameter(torch.ones(len(self.TASKS)))

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        terms: list[torch.Tensor] = []
        for index, task in enumerate(self.TASKS):
            target = labels[task]
            mask = target.ne(self.config.ignore_index)
            if not bool(mask.any()):
                continue
            loss = F.cross_entropy(
                logits[task],
                target,
                ignore_index=self.config.ignore_index,
                label_smoothing=self.config.label_smoothing,
            )
            losses[task] = loss
            scale = self.scales[index]
            terms.append(0.5 * loss / scale.square() + torch.log1p(scale.square()))
        if not terms:
            return None, losses
        return torch.stack(terms).sum(), losses


def learning_rate_at_update(
    applied_update: int,
    config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG,
) -> float:
    """Return warmup-then-cosine LR for a 1-indexed applied update."""

    if applied_update < 1 or applied_update > config.applied_update_budget:
        raise ValueError("applied update is outside the frozen run budget")
    if applied_update <= config.warmup_applied_updates:
        return config.learning_rate * applied_update / config.warmup_applied_updates
    progress = (applied_update - config.warmup_applied_updates) / (
        config.applied_update_budget - config.warmup_applied_updates
    )
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def configure_optimizer(
    model: nn.Module,
    objective: TwoTaskUncertaintyLoss,
    config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG,
) -> torch.optim.AdamW:
    """Construct the no-class-weight AdamW optimizer."""

    return torch.optim.AdamW(
        (*model.parameters(), *objective.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def apply_update_learning_rate(
    optimizer: torch.optim.Optimizer,
    applied_update: int,
    config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG,
) -> float:
    value = learning_rate_at_update(applied_update, config)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


__all__ = [
    "TwoTaskUncertaintyLoss",
    "apply_update_learning_rate",
    "configure_optimizer",
    "learning_rate_at_update",
]
