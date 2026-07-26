"""Visible raw-input reconstruction for Phase 6A plumbing checks only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.graph import RAW_FEATURE_REGISTRY
from music_critic.models.contracts import RAW_RECONSTRUCTION_CONTRACT_VERSION
from music_critic.models.encoder import EncoderOutput, normalize_continuous


RECONSTRUCTION_FIELDS = (
    ("song", "duration_qn"),
    ("track", "program"),
    ("bar", "meter_numerator"),
    ("beat", "is_downbeat"),
    ("onset", "start_qn"),
    ("note", "pitch"),
)


@dataclass(frozen=True, slots=True)
class ReconstructionOutput:
    """Local reconstruction values and unreduced per-node masked losses."""

    contract_version: str
    node_type: str
    feature_name: str
    kind: str
    logits: Tensor
    availability_mask: Tensor
    per_node_loss: Tensor

    def __post_init__(self) -> None:
        if self.contract_version != RAW_RECONSTRUCTION_CONTRACT_VERSION:
            raise ValueError("raw reconstruction contract version is incompatible")
        count = int(self.availability_mask.shape[0])
        if (
            self.availability_mask.dtype != torch.bool
            or self.per_node_loss.ndim != 1
            or int(self.per_node_loss.shape[0]) != count
            or int(self.logits.shape[0]) != count
        ):
            raise ValueError("reconstruction output changed local node rows")


class RawReconstructionHeads(nn.Module):
    """One inference-safe local field per mandatory node type."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.specs = tuple(
            next(
                spec
                for spec in RAW_FEATURE_REGISTRY.for_node(node_type)
                if spec.name == feature_name
            )
            for node_type, feature_name in RECONSTRUCTION_FIELDS
        )
        self.heads = nn.ModuleDict(
            {
                f"field_{index:02d}": nn.Linear(
                    hidden_dim,
                    (
                        int(spec.vocabulary_size or 0)
                        if spec.kind == "categorical"
                        else 1
                    ),
                )
                for index, spec in enumerate(self.specs)
            }
        )

    def forward(
        self, output: EncoderOutput, graph: object
    ) -> tuple[ReconstructionOutput, ...]:
        results = []
        for index, spec in enumerate(self.specs):
            store = graph[spec.node_type]
            logits = self.heads[f"field_{index:02d}"](
                output.embeddings[spec.node_type]
            )
            if spec.kind == "categorical":
                column = RAW_FEATURE_REGISTRY.names(
                    spec.node_type, "categorical"
                ).index(spec.name)
                availability = store.x_cat_available[:, column]
                per_node = torch.zeros(
                    int(store.num_nodes), device=logits.device
                )
                positions = torch.nonzero(
                    availability, as_tuple=False
                ).flatten()
                if positions.numel():
                    values = store.x_cat[:, column]
                    losses = F.cross_entropy(
                        logits.index_select(0, positions),
                        values.index_select(0, positions),
                        reduction="none",
                    )
                    per_node.index_copy_(0, positions, losses)
            else:
                column = RAW_FEATURE_REGISTRY.names(
                    spec.node_type, "continuous"
                ).index(spec.name)
                availability = store.x_cont_available[:, column]
                target = normalize_continuous(store.x_cont[:, column], spec)
                per_node = torch.zeros(
                    int(store.num_nodes), device=logits.device
                )
                positions = torch.nonzero(
                    availability, as_tuple=False
                ).flatten()
                if positions.numel():
                    losses = F.smooth_l1_loss(
                        logits.flatten().index_select(0, positions),
                        target.index_select(0, positions),
                        reduction="none",
                    )
                    per_node.index_copy_(0, positions, losses)
            results.append(
                ReconstructionOutput(
                    contract_version=RAW_RECONSTRUCTION_CONTRACT_VERSION,
                    node_type=spec.node_type,
                    feature_name=spec.name,
                    kind=spec.kind,
                    logits=logits,
                    availability_mask=availability,
                    per_node_loss=per_node,
                )
            )
        return tuple(results)


def reconstruction_loss(
    outputs: tuple[ReconstructionOutput, ...],
) -> Tensor | None:
    """Mean active-field loss; missing fields contribute no artificial target."""

    active = [
        output.per_node_loss[output.availability_mask].mean()
        for output in outputs
        if output.availability_mask.any()
    ]
    return torch.stack(active).mean() if active else None


__all__ = [
    "RECONSTRUCTION_FIELDS",
    "RawReconstructionHeads",
    "ReconstructionOutput",
    "reconstruction_loss",
]
