"""Batch-partition-invariant epoch metrics for Phase 6C."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from music_critic.tasks import MultiSourceBatch


@dataclass(frozen=True, slots=True)
class _ScalarRecord:
    dataset_id: str
    piece_id: str
    family: str
    name: str
    numerator: Tensor
    denominator: Tensor


class EpochMetricAccumulator:
    """Retain detached per-sample scalars and synchronize once at finalize."""

    def __init__(
        self,
        *,
        harmonic_weight: float,
        reconstruction_weight: float,
        task_weights: dict[str, float],
    ) -> None:
        self.harmonic_weight = harmonic_weight
        self.reconstruction_weight = reconstruction_weight
        self.task_weights = dict(task_weights)
        self.records: list[_ScalarRecord] = []
        self.dataset_counts: Counter[str] = Counter()
        self.batch_count = 0
        self.skipped_batch_count = 0
        self.device_to_host_sync_count = 0
        self.gradient_evidence_scan_count = 0

    def add(
        self,
        output: Any,
        batch: MultiSourceBatch,
        *,
        skipped: bool = False,
    ) -> None:
        self.batch_count += 1
        self.skipped_batch_count += int(skipped)
        self.dataset_counts.update(batch.dataset_ids)
        sample_count = len(batch.dataset_ids)
        for supervision in output.supervisions:
            losses = supervision.per_row_loss.detach()
            if losses.numel() == 0:
                continue
            sample_sums = torch.zeros(
                sample_count,
                dtype=losses.dtype,
                device=losses.device,
            )
            sample_sums.index_add_(
                0, supervision.sample_indices, losses
            )
            sample_counts = torch.bincount(
                supervision.sample_indices,
                minlength=sample_count,
            ).to(dtype=losses.dtype)
            for sample_index, (dataset_id, piece_id) in enumerate(
                zip(batch.dataset_ids, batch.piece_ids, strict=True)
            ):
                self.records.append(
                    _ScalarRecord(
                        dataset_id=dataset_id,
                        piece_id=piece_id,
                        family="task",
                        name=supervision.task_id,
                        numerator=sample_sums[sample_index],
                        denominator=sample_counts[sample_index],
                    )
                )
        for reconstruction in output.reconstruction:
            losses = reconstruction.per_node_loss.detach()
            membership = batch.raw_graph_batch[
                reconstruction.node_type
            ].batch
            sample_sums = torch.zeros(
                sample_count,
                dtype=losses.dtype,
                device=losses.device,
            )
            sample_sums.index_add_(0, membership, losses)
            sample_counts = torch.zeros_like(sample_sums)
            sample_counts.index_add_(
                0,
                membership,
                reconstruction.availability_mask.to(losses.dtype),
            )
            name = (
                f"{reconstruction.node_type}."
                f"{reconstruction.feature_name}"
            )
            for sample_index, (dataset_id, piece_id) in enumerate(
                zip(batch.dataset_ids, batch.piece_ids, strict=True)
            ):
                self.records.append(
                    _ScalarRecord(
                        dataset_id=dataset_id,
                        piece_id=piece_id,
                        family="reconstruction",
                        name=name,
                        numerator=sample_sums[sample_index],
                        denominator=sample_counts[sample_index],
                    )
                )

    def finalize(self) -> dict[str, object]:
        ordered = sorted(
            self.records,
            key=lambda item: (
                item.dataset_id,
                item.piece_id,
                item.family,
                item.name,
            ),
        )
        values: list[list[float]] = []
        if ordered:
            packed = torch.stack(
                [
                    torch.stack(
                        (
                            item.numerator.to(torch.float64),
                            item.denominator.to(torch.float64),
                        )
                    )
                    for item in ordered
                ]
            )
            values = packed.cpu().tolist()
            self.device_to_host_sync_count += 1
        global_values: dict[
            str, dict[str, list[tuple[float, float]]]
        ] = {
            "task": defaultdict(list),
            "reconstruction": defaultdict(list),
        }
        dataset_values: dict[
            str,
            dict[str, dict[str, list[tuple[float, float]]]],
        ] = {}
        for record, pair in zip(ordered, values, strict=True):
            numerator, denominator = pair
            if denominator == 0:
                continue
            global_values[record.family][record.name].append(
                (numerator, denominator)
            )
            by_family = dataset_values.setdefault(
                record.dataset_id,
                {
                    "task": defaultdict(list),
                    "reconstruction": defaultdict(list),
                },
            )
            by_family[record.family][record.name].append(
                (numerator, denominator)
            )
        task_metrics = _family_metrics(global_values["task"])
        reconstruction_metrics = _family_metrics(
            global_values["reconstruction"]
        )
        objective = _objective(
            task_metrics,
            reconstruction_metrics,
            harmonic_weight=self.harmonic_weight,
            reconstruction_weight=self.reconstruction_weight,
            task_weights=self.task_weights,
        )
        per_dataset = {}
        for dataset_id, families in sorted(dataset_values.items()):
            tasks = _family_metrics(families["task"])
            reconstruction = _family_metrics(
                families["reconstruction"]
            )
            per_dataset[dataset_id] = {
                "sample_count": self.dataset_counts[dataset_id],
                "tasks": tasks,
                "reconstruction": reconstruction,
                **_objective(
                    tasks,
                    reconstruction,
                    harmonic_weight=self.harmonic_weight,
                    reconstruction_weight=self.reconstruction_weight,
                    task_weights=self.task_weights,
                ),
            }
        return {
            **objective,
            "tasks": task_metrics,
            "reconstruction": reconstruction_metrics,
            "per_dataset": per_dataset,
            "dataset_counts": dict(sorted(self.dataset_counts.items())),
            "batch_count": self.batch_count,
            "skipped_batch_count": self.skipped_batch_count,
            "hot_path_instrumentation": {
                "gradient_evidence_scans": (
                    self.gradient_evidence_scan_count
                ),
                "per_parameter_host_syncs": 0,
                "per_task_host_syncs": 0,
                "per_family_host_syncs": 0,
                "epoch_finalize_device_to_host_syncs": (
                    self.device_to_host_sync_count
                ),
            },
        }


def _family_metrics(
    values: dict[str, list[tuple[float, float]]],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, rows in sorted(values.items()):
        numerator = math.fsum(row[0] for row in rows)
        denominator_float = math.fsum(row[1] for row in rows)
        denominator = int(denominator_float)
        if denominator <= 0 or denominator_float != denominator:
            raise ValueError("training.metrics.denominator_invalid")
        result[name] = {
            "loss_numerator": numerator,
            "eligible_row_count": denominator,
            "mean_loss": numerator / denominator,
        }
    return result


def _objective(
    tasks: dict[str, dict[str, float | int]],
    reconstruction: dict[str, dict[str, float | int]],
    *,
    harmonic_weight: float,
    reconstruction_weight: float,
    task_weights: dict[str, float],
) -> dict[str, float | None]:
    weighted_tasks = [
        (
            float(value["mean_loss"]),
            float(task_weights.get(task_id, 1.0)),
        )
        for task_id, value in sorted(tasks.items())
    ]
    harmonic = (
        None
        if not weighted_tasks
        else math.fsum(loss * weight for loss, weight in weighted_tasks)
        / math.fsum(weight for _, weight in weighted_tasks)
    )
    field_means = [
        float(value["mean_loss"])
        for _, value in sorted(reconstruction.items())
    ]
    reconstruction_loss = (
        None
        if not field_means
        else math.fsum(field_means) / len(field_means)
    )
    terms = []
    if harmonic is not None and harmonic_weight > 0:
        terms.append(harmonic * harmonic_weight)
    if reconstruction_loss is not None and reconstruction_weight > 0:
        terms.append(reconstruction_loss * reconstruction_weight)
    return {
        "harmonic_loss": harmonic,
        "reconstruction_loss": reconstruction_loss,
        "objective_loss": None if not terms else math.fsum(terms),
    }


__all__ = ["EpochMetricAccumulator"]
