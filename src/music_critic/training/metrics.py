"""Bounded-memory, row-weighted epoch metrics for Phase 6C."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from music_critic.tasks import MultiSourceBatch


@dataclass(slots=True)
class _Aggregate:
    numerator: float = 0.0
    denominator: int = 0


class EpochMetricAccumulator:
    """Fold one packed device transfer per batch into bounded CPU buckets."""

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
        self._aggregates: dict[
            tuple[str, str, str], _Aggregate
        ] = {}
        self.dataset_counts: Counter[str] = Counter()
        self.batch_count = 0
        self.skipped_batch_count = 0
        self.packed_host_materialization_count = 0
        self.packed_device_to_host_transfer_count = 0
        self.packed_host_scalar_count = 0
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
        dataset_ids = tuple(sorted(set(batch.dataset_ids)))
        dataset_to_index = {
            dataset_id: index
            for index, dataset_id in enumerate(dataset_ids)
        }
        metadata: list[tuple[str, str, str]] = []
        values: list[tuple[Tensor, Tensor]] = []
        reference_device = _output_device(output)
        sample_dataset_indices = torch.tensor(
            [
                dataset_to_index[dataset_id]
                for dataset_id in batch.dataset_ids
            ],
            dtype=torch.long,
            device=reference_device,
        )

        for supervision in output.supervisions:
            losses = supervision.per_row_loss.detach()
            if losses.numel() == 0:
                continue
            row_datasets = sample_dataset_indices.index_select(
                0, supervision.sample_indices
            )
            sums = torch.zeros(
                len(dataset_ids),
                dtype=losses.dtype,
                device=losses.device,
            )
            sums.index_add_(0, row_datasets, losses)
            counts = torch.bincount(
                row_datasets,
                minlength=len(dataset_ids),
            )
            _append_family_values(
                metadata,
                values,
                dataset_ids=dataset_ids,
                family="task",
                name=supervision.task_id,
                numerators=sums,
                denominators=counts,
            )

        for reconstruction in output.reconstruction:
            losses = reconstruction.per_node_loss.detach()
            membership = batch.raw_graph_batch[
                reconstruction.node_type
            ].batch
            row_datasets = sample_dataset_indices.index_select(
                0, membership
            )
            sums = torch.zeros(
                len(dataset_ids),
                dtype=losses.dtype,
                device=losses.device,
            )
            sums.index_add_(0, row_datasets, losses)
            counts = torch.zeros(
                len(dataset_ids),
                dtype=torch.long,
                device=losses.device,
            )
            counts.index_add_(
                0,
                row_datasets,
                reconstruction.availability_mask.to(torch.long),
            )
            _append_family_values(
                metadata,
                values,
                dataset_ids=dataset_ids,
                family="reconstruction",
                name=(
                    f"{reconstruction.node_type}."
                    f"{reconstruction.feature_name}"
                ),
                numerators=sums,
                denominators=counts,
            )

        if not values:
            return
        packed = torch.stack(
            [
                torch.stack(
                    (
                        numerator.to(torch.float64),
                        denominator.to(torch.float64),
                    )
                )
                for numerator, denominator in values
            ]
        )
        # This is the only device-to-host metric transfer site. Nothing from
        # ``packed`` or its scalar views is retained after this method returns.
        host_values = packed.to(device="cpu").tolist()
        self.packed_host_materialization_count += 1
        self.packed_device_to_host_transfer_count += int(
            packed.device.type != "cpu"
        )
        self.packed_host_scalar_count += int(packed.numel())
        for key, (numerator, denominator_float) in zip(
            metadata, host_values, strict=True
        ):
            denominator = int(denominator_float)
            if denominator_float != denominator or denominator < 0:
                raise ValueError(
                    "training.metrics.denominator_invalid"
                )
            if denominator == 0:
                continue
            aggregate = self._aggregates.setdefault(key, _Aggregate())
            aggregate.numerator += numerator
            aggregate.denominator += denominator

    def storage_evidence(self) -> dict[str, int]:
        """Report actual retained metric state after a completed ``add``."""

        retained_tensors = tuple(
            value
            for value in self.__dict__.values()
            if isinstance(value, Tensor)
        )
        return {
            "aggregate_bucket_count": len(self._aggregates),
            "retained_tensor_count": len(retained_tensors),
            "retained_device_tensor_count": sum(
                tensor.device.type != "cpu"
                for tensor in retained_tensors
            ),
            "retained_device_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in retained_tensors
                if tensor.device.type != "cpu"
            ),
        }

    def finalize(self) -> dict[str, object]:
        global_values = _global_aggregates(self._aggregates)
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
        for dataset_id in sorted(self.dataset_counts):
            tasks = _family_metrics(
                _dataset_family(
                    self._aggregates, dataset_id, "task"
                )
            )
            reconstruction = _family_metrics(
                _dataset_family(
                    self._aggregates,
                    dataset_id,
                    "reconstruction",
                )
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
            "runtime_transfer_evidence": {
                "gradient_evidence_scans": (
                    self.gradient_evidence_scan_count
                ),
                "metric_packed_device_to_host_transfers": (
                    self.packed_device_to_host_transfer_count
                ),
                "metric_packed_host_materializations": (
                    self.packed_host_materialization_count
                ),
                "metric_packed_host_scalars": (
                    self.packed_host_scalar_count
                ),
                **self.storage_evidence(),
            },
        }


def _output_device(output: Any) -> torch.device:
    for supervision in output.supervisions:
        return supervision.per_row_loss.device
    for reconstruction in output.reconstruction:
        return reconstruction.per_node_loss.device
    return torch.device("cpu")


def _append_family_values(
    metadata: list[tuple[str, str, str]],
    values: list[tuple[Tensor, Tensor]],
    *,
    dataset_ids: tuple[str, ...],
    family: str,
    name: str,
    numerators: Tensor,
    denominators: Tensor,
) -> None:
    for dataset_index, dataset_id in enumerate(dataset_ids):
        metadata.append((dataset_id, family, name))
        values.append(
            (
                numerators[dataset_index],
                denominators[dataset_index],
            )
        )


def _dataset_family(
    aggregates: dict[tuple[str, str, str], _Aggregate],
    dataset_id: str,
    family: str,
) -> dict[str, _Aggregate]:
    return {
        name: value
        for (candidate_dataset, candidate_family, name), value in (
            aggregates.items()
        )
        if candidate_dataset == dataset_id
        and candidate_family == family
    }


def _global_aggregates(
    aggregates: dict[tuple[str, str, str], _Aggregate],
) -> dict[str, dict[str, _Aggregate]]:
    grouped: dict[str, dict[str, list[_Aggregate]]] = {
        "task": {},
        "reconstruction": {},
    }
    for (_, family, name), aggregate in sorted(aggregates.items()):
        grouped[family].setdefault(name, []).append(aggregate)
    return {
        family: {
            name: _Aggregate(
                numerator=math.fsum(
                    item.numerator for item in items
                ),
                denominator=sum(
                    item.denominator for item in items
                ),
            )
            for name, items in names.items()
        }
        for family, names in grouped.items()
    }


def _family_metrics(
    values: dict[str, _Aggregate],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, aggregate in sorted(values.items()):
        if aggregate.denominator <= 0:
            raise ValueError("training.metrics.denominator_invalid")
        result[name] = {
            "loss_numerator": aggregate.numerator,
            "eligible_row_count": aggregate.denominator,
            "mean_loss": (
                aggregate.numerator / aggregate.denominator
            ),
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
        "objective_loss": (
            None if not terms else math.fsum(terms)
        ),
    }


__all__ = ["EpochMetricAccumulator"]
