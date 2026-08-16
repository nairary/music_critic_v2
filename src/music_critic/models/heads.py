"""Candidate-first harmonic prediction and tensorized supervision losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.contracts import (
    BASELINE_LOSS_CONTRACT_VERSION,
    TASK_PREDICTION_CONTRACT_VERSION,
    TaskHeadSpec,
)
from music_critic.models.encoder import EncoderOutput
from music_critic.tasks import BatchTarget, ENTITY_NODE_TYPE_TO_CODE


@dataclass(frozen=True, slots=True)
class TaskPrediction:
    """Target-independent logits for every raw-graph candidate of one task."""

    contract_version: str
    task_id: str
    source_adapter: str
    allowed_node_types: tuple[str, ...]
    candidate_node_type_codes: Tensor
    global_entity_indices: Tensor
    sample_indices: Tensor
    candidate_offsets_by_node_type: Tensor
    candidate_counts_by_node_type: Tensor
    logits: Tensor

    def __post_init__(self) -> None:
        if self.contract_version != TASK_PREDICTION_CONTRACT_VERSION:
            raise ValueError("task prediction contract version is incompatible")
        row_count = self.logits.shape[0]
        if (
            self.candidate_node_type_codes.dtype != torch.long
            or self.global_entity_indices.dtype != torch.long
            or self.sample_indices.dtype != torch.long
            or self.logits.ndim != 2
            or any(
                value.ndim != 1 or value.shape[0] != row_count
                for value in (
                    self.candidate_node_type_codes,
                    self.global_entity_indices,
                    self.sample_indices,
                )
            )
            or self.candidate_offsets_by_node_type.dtype != torch.long
            or self.candidate_counts_by_node_type.dtype != torch.long
            or self.candidate_offsets_by_node_type.shape
            != (len(MANDATORY_NODE_TYPES),)
            or self.candidate_counts_by_node_type.shape
            != (len(MANDATORY_NODE_TYPES),)
        ):
            raise ValueError("task prediction candidate tensors are inconsistent")


@dataclass(frozen=True, slots=True)
class TaskSupervision:
    """Tensorized join from eligible BatchTarget rows to raw candidates."""

    task_id: str
    target_row_indices: Tensor
    candidate_indices: Tensor
    node_type_codes: Tensor
    global_entity_indices: Tensor
    sample_indices: Tensor
    per_row_loss: Tensor

    def __post_init__(self) -> None:
        row_count = self.target_row_indices.shape[0]
        if any(
            value.ndim != 1 or value.shape[0] != row_count
            for value in (
                self.target_row_indices,
                self.candidate_indices,
                self.node_type_codes,
                self.global_entity_indices,
                self.sample_indices,
                self.per_row_loss,
            )
        ) or any(
            value.dtype != torch.long
            for value in (
                self.target_row_indices,
                self.candidate_indices,
                self.node_type_codes,
                self.global_entity_indices,
                self.sample_indices,
            )
        ):
            raise ValueError("task supervision tensors are inconsistent")


@dataclass(frozen=True, slots=True)
class TaskLoss:
    """Tensorized task/node-type/sample group report."""

    task_id: str
    weight: float
    group_node_type_codes: Tensor
    group_sample_indices: Tensor
    group_row_counts: Tensor
    group_mean_losses: Tensor
    mean_loss: Tensor

    def __post_init__(self) -> None:
        group_count = self.group_mean_losses.shape[0]
        if (
            self.group_node_type_codes.dtype != torch.long
            or self.group_sample_indices.dtype != torch.long
            or self.group_row_counts.dtype != torch.long
            or any(
                value.ndim != 1 or value.shape[0] != group_count
                for value in (
                    self.group_node_type_codes,
                    self.group_sample_indices,
                    self.group_row_counts,
                    self.group_mean_losses,
                )
            )
            or self.mean_loss.ndim != 0
        ):
            raise ValueError("task loss group tensors are inconsistent")


@dataclass(frozen=True, slots=True)
class BaselineLossReport:
    """Versioned deterministic task -> tensorized group -> row reduction."""

    contract_version: str
    task_losses: tuple[TaskLoss, ...]
    total_loss: Tensor | None

    def __post_init__(self) -> None:
        if self.contract_version != BASELINE_LOSS_CONTRACT_VERSION:
            raise ValueError("baseline loss report version is incompatible")
        if (self.total_loss is None) != (not self.task_losses):
            raise ValueError("total loss exists exactly when active task groups exist")


@dataclass(frozen=True, slots=True)
class RoutingOperationCounts:
    """Python work bounded by fixed task/node-type registries."""

    prediction_task_visits: int
    candidate_node_type_visits: int
    supervision_task_visits: int
    tensor_group_reductions: int


def _head_key(index: int) -> str:
    return f"task_{index:02d}"


class SourceNativeTaskHeads(nn.Module):
    """One candidate-first MLP per accepted Phase 6A task."""

    def __init__(
        self,
        specs: tuple[TaskHeadSpec, ...],
        hidden_dim: int,
        task_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.specs = specs
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

    def forward(
        self,
        encoder_output: EncoderOutput,
    ) -> tuple[TaskPrediction, ...]:
        """Enumerate candidates from raw embeddings, independent of targets."""

        predictions = []
        for spec_index, spec in enumerate(self.specs):
            embeddings = []
            node_type_codes = []
            entity_indices = []
            sample_indices = []
            reference = encoder_output.embeddings[spec.node_types[0]]
            offsets = torch.full(
                (len(MANDATORY_NODE_TYPES),),
                -1,
                dtype=torch.long,
                device=reference.device,
            )
            counts = torch.zeros_like(offsets)
            candidate_offset = 0
            for node_type in spec.node_types:
                node_type_code = ENTITY_NODE_TYPE_TO_CODE[node_type]
                values = encoder_output.embeddings[node_type]
                count = values.shape[0]
                offsets[node_type_code] = candidate_offset
                counts[node_type_code] = count
                indices = torch.arange(
                    count, dtype=torch.long, device=values.device
                )
                type_codes = torch.full_like(indices, node_type_code)
                if len(spec.node_types) > 1:
                    values = values + self.node_type_embeddings[
                        _head_key(spec_index)
                    ](type_codes)
                embeddings.append(values)
                node_type_codes.append(type_codes)
                entity_indices.append(indices)
                sample_indices.append(
                    encoder_output.batch_membership[node_type]
                )
                candidate_offset += count
            routed = torch.cat(embeddings, dim=0)
            predictions.append(
                TaskPrediction(
                    contract_version=TASK_PREDICTION_CONTRACT_VERSION,
                    task_id=spec.task_id,
                    source_adapter=spec.source_adapter,
                    allowed_node_types=spec.node_types,
                    candidate_node_type_codes=torch.cat(
                        node_type_codes, dim=0
                    ),
                    global_entity_indices=torch.cat(entity_indices, dim=0),
                    sample_indices=torch.cat(sample_indices, dim=0),
                    candidate_offsets_by_node_type=offsets,
                    candidate_counts_by_node_type=counts,
                    logits=self.heads[_head_key(spec_index)](routed),
                )
            )
        return tuple(predictions)


def join_task_supervision(
    predictions: tuple[TaskPrediction, ...],
    targets: tuple[BatchTarget, ...],
) -> tuple[TaskSupervision, ...]:
    """Join eligible targets to existing candidates using tensors only."""

    targets_by_task = {target.task_id: target for target in targets}
    supervisions = []
    for prediction in predictions:
        target = targets_by_task[prediction.task_id]
        if (
            not target.model_ready
            or target.supervision_regime != "fully_supervised"
        ):
            raise ValueError("active task target changed supervision semantics")
        device = prediction.logits.device
        eligibility = (
            target.availability_mask
            & target.entity_index_mask
            & target.model_ready
        ).to(device)
        target_rows = torch.nonzero(eligibility, as_tuple=False).flatten()
        if target_rows.numel() == 0:
            continue
        node_type_codes = target.entity_node_type_codes.to(device).index_select(
            0, target_rows
        )
        entity_indices = target.entity_indices.to(device).index_select(
            0, target_rows
        )
        sample_indices = target.sample_indices.to(device).index_select(
            0, target_rows
        )
        candidate_indices = (
            prediction.candidate_offsets_by_node_type.index_select(
                0, node_type_codes
            )
            + entity_indices
        )
        if (
            not torch.equal(
                prediction.candidate_node_type_codes.index_select(
                    0, candidate_indices
                ),
                node_type_codes,
            )
            or not torch.equal(
                prediction.global_entity_indices.index_select(
                    0, candidate_indices
                ),
                entity_indices,
            )
            or not torch.equal(
                prediction.sample_indices.index_select(0, candidate_indices),
                sample_indices,
            )
        ):
            raise ValueError("target-to-candidate mapping is inconsistent")
        logits = prediction.logits.index_select(0, candidate_indices)
        values = target.values.to(device).index_select(0, target_rows)
        if target.encoding_kind == "closed_categorical_index":
            losses = F.cross_entropy(logits, values, reduction="none")
        elif target.encoding_kind == "closed_multilabel":
            losses = F.binary_cross_entropy_with_logits(
                logits, values.float(), reduction="none"
            ).mean(dim=-1)
        else:
            raise ValueError("active task has no baseline loss")
        supervisions.append(
            TaskSupervision(
                task_id=prediction.task_id,
                target_row_indices=target_rows,
                candidate_indices=candidate_indices,
                node_type_codes=node_type_codes,
                global_entity_indices=entity_indices,
                sample_indices=sample_indices,
                per_row_loss=losses,
            )
        )
    return tuple(supervisions)


def aggregate_task_losses(
    supervisions: tuple[TaskSupervision, ...],
    *,
    task_weights: Mapping[str, float] | None = None,
) -> BaselineLossReport:
    """Vectorize row -> task/node-type/sample -> task reduction."""

    task_weights = task_weights or {}
    task_losses = []
    for supervision in supervisions:
        weight = float(task_weights.get(supervision.task_id, 1.0))
        if weight == 0.0:
            continue
        group_keys = torch.stack(
            (supervision.node_type_codes, supervision.sample_indices), dim=1
        )
        unique_groups, inverse, counts = torch.unique(
            group_keys,
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        sums = torch.zeros(
            unique_groups.shape[0],
            dtype=supervision.per_row_loss.dtype,
            device=supervision.per_row_loss.device,
        )
        sums.index_add_(0, inverse, supervision.per_row_loss)
        group_means = sums / counts.to(sums.dtype)
        task_losses.append(
            TaskLoss(
                task_id=supervision.task_id,
                weight=weight,
                group_node_type_codes=unique_groups[:, 0],
                group_sample_indices=unique_groups[:, 1],
                group_row_counts=counts,
                group_mean_losses=group_means,
                mean_loss=group_means.mean(),
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


def routing_operation_counts(
    specs: tuple[TaskHeadSpec, ...],
    supervisions: tuple[TaskSupervision, ...],
) -> RoutingOperationCounts:
    """Return fixed-registry Python operation evidence."""

    return RoutingOperationCounts(
        prediction_task_visits=len(specs),
        candidate_node_type_visits=sum(len(spec.node_types) for spec in specs),
        supervision_task_visits=len(specs),
        tensor_group_reductions=len(supervisions),
    )


__all__ = [
    "BaselineLossReport",
    "RoutingOperationCounts",
    "SourceNativeTaskHeads",
    "TaskLoss",
    "TaskPrediction",
    "TaskSupervision",
    "aggregate_task_losses",
    "join_task_supervision",
    "routing_operation_counts",
]
