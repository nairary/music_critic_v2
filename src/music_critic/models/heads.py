"""Source-native local task heads and inspectable loss reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.contracts import (
    BASELINE_LOSS_CONTRACT_VERSION,
    TaskHeadSpec,
)
from music_critic.models.encoder import EncoderOutput
from music_critic.tasks import BatchTarget


@dataclass(frozen=True, slots=True)
class TaskOutput:
    """Local logits and row-level supervision evidence for one active task."""

    task_id: str
    source_adapter: str
    node_types: tuple[str, ...]
    batch_row_indices: Tensor
    global_entity_indices: Tensor
    sample_indices: Tensor
    eligibility_mask: Tensor
    logits: Tensor
    per_row_loss: Tensor | None

    def __post_init__(self) -> None:
        row_count = int(self.batch_row_indices.shape[0])
        if (
            self.batch_row_indices.dtype != torch.long
            or self.global_entity_indices.dtype != torch.long
            or self.sample_indices.dtype != torch.long
            or self.eligibility_mask.dtype != torch.bool
            or self.logits.ndim != 2
            or any(
                int(value.shape[0]) != row_count
                for value in (
                    self.global_entity_indices,
                    self.sample_indices,
                    self.eligibility_mask,
                    self.logits,
                )
            )
            or len(self.node_types) != row_count
        ):
            raise ValueError("task output rows are inconsistent")
        if self.per_row_loss is not None and (
            self.per_row_loss.ndim != 1
            or int(self.per_row_loss.shape[0]) != row_count
        ):
            raise ValueError("task output row loss is inconsistent")


@dataclass(frozen=True, slots=True)
class LossGroup:
    """Mean eligible row loss for one task/node-type/sample group."""

    task_id: str
    node_type: str
    sample_index: int
    row_count: int
    mean_loss: Tensor


@dataclass(frozen=True, slots=True)
class TaskLoss:
    """Mean of active local groups for one task."""

    task_id: str
    weight: float
    groups: tuple[LossGroup, ...]
    mean_loss: Tensor


@dataclass(frozen=True, slots=True)
class BaselineLossReport:
    """Versioned deterministic task -> local group -> row reduction."""

    contract_version: str
    task_losses: tuple[TaskLoss, ...]
    total_loss: Tensor | None

    def __post_init__(self) -> None:
        if self.contract_version != BASELINE_LOSS_CONTRACT_VERSION:
            raise ValueError("baseline loss report version is incompatible")
        if (self.total_loss is None) != (not self.task_losses):
            raise ValueError("total loss exists exactly when active task groups exist")


def _head_key(index: int) -> str:
    return f"task_{index:02d}"


class SourceNativeTaskHeads(nn.Module):
    """One actual MLP per accepted Phase 6A task; no unused output heads."""

    def __init__(
        self,
        specs: tuple[TaskHeadSpec, ...],
        hidden_dim: int,
        task_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.task_to_index = {
            spec.task_id: index for index, spec in enumerate(specs)
        }
        self.heads = nn.ModuleDict(
            {
                _head_key(index): nn.Sequential(
                    nn.Linear(hidden_dim, task_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(task_hidden_dim, spec.output_dim),
                )
                for index, spec in enumerate(specs)
            }
        )
        self.node_type_embeddings = nn.ModuleDict(
            {
                _head_key(index): nn.Embedding(
                    len(MANDATORY_NODE_TYPES), hidden_dim
                )
                for index, spec in enumerate(specs)
                if len(spec.node_types) > 1
            }
        )
        self.node_type_index = {
            node_type: index
            for index, node_type in enumerate(MANDATORY_NODE_TYPES)
        }

    def forward(
        self,
        encoder_output: EncoderOutput,
        targets: tuple[BatchTarget, ...] | None,
    ) -> tuple[TaskOutput, ...]:
        if targets is None:
            return ()
        targets_by_task = {target.task_id: target for target in targets}
        outputs = []
        for spec_index, spec in enumerate(self.specs):
            target = targets_by_task[spec.task_id]
            if (
                not target.model_ready
                or target.supervision_regime != "fully_supervised"
            ):
                raise ValueError("active task target changed supervision semantics")
            aligned_rows = torch.nonzero(
                target.entity_index_mask, as_tuple=False
            ).flatten()
            device = encoder_output.embeddings[
                spec.node_types[0]
            ].device
            aligned_rows = aligned_rows.to(device)
            node_types = tuple(
                target.entity_node_types[int(row)]
                for row in aligned_rows.cpu().tolist()
            )
            node_type_codes = torch.tensor(
                [self.node_type_index[node_type] for node_type in node_types],
                dtype=torch.long,
                device=device,
            )
            routed = torch.empty(
                (int(aligned_rows.shape[0]), next(iter(
                    encoder_output.embeddings.values()
                )).shape[1]),
                device=device,
            )
            global_indices = target.entity_indices.to(device).index_select(
                0, aligned_rows
            )
            sample_indices = target.sample_indices.to(device).index_select(
                0, aligned_rows
            )
            for node_type in spec.node_types:
                positions = torch.nonzero(
                    node_type_codes == self.node_type_index[node_type],
                    as_tuple=False,
                ).flatten()
                if positions.numel() == 0:
                    continue
                indices = global_indices.index_select(0, positions)
                values = encoder_output.embeddings[node_type].index_select(
                    0, indices
                )
                if len(spec.node_types) > 1:
                    type_ids = torch.full(
                        (positions.shape[0],),
                        self.node_type_index[node_type],
                        dtype=torch.long,
                        device=device,
                    )
                    values = values + self.node_type_embeddings[
                        _head_key(spec_index)
                    ](type_ids)
                routed.index_copy_(0, positions, values)
            logits = self.heads[_head_key(spec_index)](routed)
            eligibility = (
                target.availability_mask
                & target.entity_index_mask
                & bool(target.model_ready)
            ).to(device).index_select(0, aligned_rows)
            per_row_loss = torch.zeros(
                int(aligned_rows.shape[0]), device=device
            )
            eligible_positions = torch.nonzero(
                eligibility, as_tuple=False
            ).flatten()
            if eligible_positions.numel():
                batch_rows = aligned_rows.index_select(
                    0, eligible_positions
                )
                if spec.encoding_kind == "closed_categorical_index":
                    values = target.values.to(device).index_select(
                        0, batch_rows
                    )
                    losses = F.cross_entropy(
                        logits.index_select(0, eligible_positions),
                        values,
                        reduction="none",
                    )
                elif spec.encoding_kind == "closed_multilabel":
                    values = target.values.to(device).index_select(
                        0, batch_rows
                    ).float()
                    losses = F.binary_cross_entropy_with_logits(
                        logits.index_select(0, eligible_positions),
                        values,
                        reduction="none",
                    ).mean(dim=-1)
                else:
                    raise ValueError("active task has no baseline loss")
                per_row_loss.index_copy_(0, eligible_positions, losses)
            outputs.append(
                TaskOutput(
                    task_id=spec.task_id,
                    source_adapter=spec.source_adapter,
                    node_types=tuple(str(value) for value in node_types),
                    batch_row_indices=aligned_rows,
                    global_entity_indices=global_indices,
                    sample_indices=sample_indices,
                    eligibility_mask=eligibility,
                    logits=logits,
                    per_row_loss=per_row_loss,
                )
            )
        return tuple(outputs)


def aggregate_task_losses(
    outputs: tuple[TaskOutput, ...],
    *,
    task_weights: Mapping[str, float] | None = None,
) -> BaselineLossReport:
    """Reduce rows without letting dense samples dominate automatically."""

    task_weights = task_weights or {}
    task_losses = []
    for output in outputs:
        if output.per_row_loss is None or not output.eligibility_mask.any():
            continue
        groups = []
        group_keys = sorted(
            {
                (output.node_types[row], int(output.sample_indices[row].item()))
                for row in torch.nonzero(
                    output.eligibility_mask, as_tuple=False
                ).flatten().tolist()
            }
        )
        for node_type, sample_index in group_keys:
            positions = torch.tensor(
                [
                    row
                    for row, (row_node_type, eligible) in enumerate(
                        zip(
                            output.node_types,
                            output.eligibility_mask.tolist(),
                        )
                    )
                    if eligible
                    and row_node_type == node_type
                    and int(output.sample_indices[row].item()) == sample_index
                ],
                dtype=torch.long,
                device=output.logits.device,
            )
            group_loss = output.per_row_loss.index_select(0, positions).mean()
            groups.append(
                LossGroup(
                    task_id=output.task_id,
                    node_type=node_type,
                    sample_index=sample_index,
                    row_count=int(positions.numel()),
                    mean_loss=group_loss,
                )
            )
        mean_loss = torch.stack([group.mean_loss for group in groups]).mean()
        task_losses.append(
            TaskLoss(
                task_id=output.task_id,
                weight=float(task_weights.get(output.task_id, 1.0)),
                groups=tuple(groups),
                mean_loss=mean_loss,
            )
        )
    total = None
    if task_losses:
        weighted = torch.stack(
            [item.mean_loss * item.weight for item in task_losses]
        )
        total = weighted.sum() / sum(item.weight for item in task_losses)
    return BaselineLossReport(
        contract_version=BASELINE_LOSS_CONTRACT_VERSION,
        task_losses=tuple(task_losses),
        total_loss=total,
    )


__all__ = [
    "BaselineLossReport",
    "LossGroup",
    "SourceNativeTaskHeads",
    "TaskLoss",
    "TaskOutput",
    "aggregate_task_losses",
]
