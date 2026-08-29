"""Frozen source-entry evaluator for the Phase 9E-B1 common benchmark."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

import torch

from music_critic.experiments.analysisgnn.contracts import TASK_CLASS_COUNTS


@dataclass(frozen=True, slots=True)
class EntryPrediction:
    record_id: str
    piece_id: str
    split: str
    task: str
    entity_id: str
    entry_index: int
    target: int
    mask: bool
    logits: tuple[float, ...]
    prediction: int


def aggregate_entry_predictions(
    *,
    record_id: str,
    piece_id: str,
    split: str,
    task: str,
    note_logits: torch.Tensor,
    note_targets: torch.Tensor,
    note_membership_index: torch.Tensor,
    entity_ids: Sequence[str],
    entry_masks: Sequence[bool],
) -> tuple[EntryPrediction, ...]:
    """Aggregate notes by mean log probability, exactly once per source entry."""

    if task not in TASK_CLASS_COUNTS:
        raise ValueError(f"unsupported task {task!r}")
    expected_classes = TASK_CLASS_COUNTS[task]
    if note_logits.ndim != 2 or note_logits.shape[1] != expected_classes:
        raise ValueError("note logits have the wrong class surface")
    if note_targets.ndim != 1 or note_targets.shape[0] != note_logits.shape[0]:
        raise ValueError("note targets do not align with logits")
    if note_membership_index.ndim != 2 or note_membership_index.shape[0] != 2:
        raise ValueError("note membership index must have shape [2, membership_count]")
    if note_membership_index.dtype != torch.long:
        raise ValueError("note membership index must use torch.long")
    if len(entity_ids) != len(entry_masks):
        raise ValueError("source-entry identities and masks must have equal length")
    if any(not isinstance(mask, bool) for mask in entry_masks):
        raise ValueError("source-entry masks must contain booleans")
    log_probabilities = note_logits.detach().float().log_softmax(dim=-1).cpu()
    targets = note_targets.detach().cpu()
    memberships = note_membership_index.detach().long().cpu()
    note_indices = memberships[0]
    entry_indices = memberships[1]
    membership_pairs = tuple(zip(note_indices.tolist(), entry_indices.tolist()))
    if membership_pairs != tuple(sorted(set(membership_pairs))):
        raise ValueError("note membership pairs must be unique and lexicographically sorted")
    if any(
        note_index < 0
        or note_index >= note_logits.shape[0]
        or entry_index < 0
        or entry_index >= len(entity_ids)
        for note_index, entry_index in membership_pairs
    ):
        raise ValueError("note membership index is out of bounds")
    if any(
        not entry_masks[entry_index]
        for _note_index, entry_index in membership_pairs
    ):
        raise ValueError("unavailable source entries cannot have note memberships")
    rows: list[EntryPrediction] = []
    for entry_index, (entity_id, entry_mask) in enumerate(zip(entity_ids, entry_masks)):
        entry_membership_mask = entry_indices.eq(entry_index)
        member_note_indices = note_indices[entry_membership_mask]
        available = bool(entry_mask) and bool(member_note_indices.numel())
        if available:
            target_values = targets[member_note_indices]
            valid_targets = target_values[target_values.ne(-1)].unique()
            if valid_targets.numel() != 1:
                raise ValueError("an available source entry must have one note-level target")
            target = int(valid_targets.item())
            aggregated = log_probabilities[member_note_indices].mean(dim=0)
            aggregated = aggregated - torch.logsumexp(aggregated, dim=-1)
            logits = tuple(float(value) for value in aggregated.tolist())
            prediction = int(aggregated.argmax().item())
        else:
            target = -1
            logits = tuple(float("nan") for _ in range(expected_classes))
            prediction = -1
        rows.append(
            EntryPrediction(
                record_id=record_id,
                piece_id=piece_id,
                split=split,
                task=task,
                entity_id=entity_id,
                entry_index=entry_index,
                target=target,
                mask=available,
                logits=logits,
                prediction=prediction,
            )
        )
    return tuple(rows)


def _confusion(rows: Sequence[EntryPrediction], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for row in rows:
        if row.mask:
            matrix[row.target][row.prediction] += 1
    return matrix


def task_metrics(rows: Iterable[EntryPrediction], task: str) -> dict[str, object]:
    selected = tuple(row for row in rows if row.task == task and row.mask)
    classes = TASK_CLASS_COUNTS[task]
    confusion = _confusion(selected, classes)
    support = [sum(confusion[index]) for index in range(classes)]
    total = sum(support)
    if total == 0:
        raise ValueError(f"no available {task} entries")
    accuracy = sum(confusion[index][index] for index in range(classes)) / total
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(classes):
        if support[index] == 0:
            continue
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(classes)) - true_positive
        false_negative = support[index] - true_positive
        recall = true_positive / support[index]
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recalls.append(recall)
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    nll = -sum(row.logits[row.target] for row in selected) / total
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "confusion_matrix": confusion,
        "macro_f1": sum(f1s) / len(f1s),
        "majority_class_baseline_accuracy": max(support) / total,
        "nll": nll,
        "per_class_support": support,
        "support": total,
        "supported_class_count": len(recalls),
    }


def joint_accuracy(rows: Iterable[EntryPrediction]) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, EntryPrediction]] = defaultdict(dict)
    for row in rows:
        if row.mask:
            grouped[(row.record_id, row.entity_id)][row.task] = row
    pairs = tuple(values for values in grouped.values() if set(values) == set(TASK_CLASS_COUNTS))
    correct = sum(
        all(row.prediction == row.target for row in values.values()) for values in pairs
    )
    return {"accuracy": correct / len(pairs) if pairs else float("nan"), "support": len(pairs)}


def benchmark_metrics(rows: Iterable[EntryPrediction]) -> dict[str, object]:
    materialized = tuple(rows)
    quality = task_metrics(materialized, "quality")
    inversion = task_metrics(materialized, "inversion")
    return {
        "inversion": inversion,
        "joint_quality_inversion": joint_accuracy(materialized),
        "normalized_mean_nll": 0.5
        * (
            float(quality["nll"]) / math.log(TASK_CLASS_COUNTS["quality"])
            + float(inversion["nll"]) / math.log(TASK_CLASS_COUNTS["inversion"])
        ),
        "quality": quality,
    }


def grouped_bootstrap(
    rows: Sequence[EntryPrediction],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """95% percentile intervals from record-grouped resampling."""

    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    by_record: dict[str, list[EntryPrediction]] = defaultdict(list)
    for row in rows:
        by_record[row.record_id].append(row)
    record_ids = sorted(by_record)
    if not record_ids:
        raise ValueError("cannot bootstrap an empty prediction table")
    rng = random.Random(seed)
    values: dict[str, dict[str, list[float]]] = {
        task: defaultdict(list) for task in ("quality", "inversion")
    }
    joint_values: list[float] = []
    normalized_nll_values: list[float] = []
    for _ in range(samples):
        sampled: list[EntryPrediction] = []
        for draw in range(len(record_ids)):
            record_id = rng.choice(record_ids)
            # Entity identities must be unique after sampling the same record twice.
            for row in by_record[record_id]:
                sampled.append(
                    EntryPrediction(
                        **{
                            **asdict(row),
                            "record_id": f"{draw}:{row.record_id}",
                        }
                    )
                )
        metrics = benchmark_metrics(sampled)
        for task in values:
            for name in (
                "nll",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "majority_class_baseline_accuracy",
            ):
                values[task][name].append(float(metrics[task][name]))  # type: ignore[index]
        joint_values.append(float(metrics["joint_quality_inversion"]["accuracy"]))  # type: ignore[index]
        normalized_nll_values.append(float(metrics["normalized_mean_nll"]))

    def percentile(series: list[float], q: float) -> float:
        ordered = sorted(series)
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    intervals: dict[str, object] = {
        task: {
            metric: (percentile(series, 0.025), percentile(series, 0.975))
            for metric, series in task_values.items()
        }
        for task, task_values in values.items()
    }
    intervals["joint_quality_inversion"] = {
        "accuracy": (
            percentile(joint_values, 0.025),
            percentile(joint_values, 0.975),
        )
    }
    intervals["normalized_mean_nll"] = (
        percentile(normalized_nll_values, 0.025),
        percentile(normalized_nll_values, 0.975),
    )
    return intervals


def summarize_seeds(seed_metrics: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    if tuple(sorted(seed_metrics)) != (17, 23, 42):
        raise ValueError("summary requires exactly seeds 17, 23, and 42")
    summary: dict[str, object] = {}
    for task in ("quality", "inversion"):
        summary[task] = {}
        for metric in (
            "nll",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "majority_class_baseline_accuracy",
        ):
            values = [float(seed_metrics[seed][task][metric]) for seed in sorted(seed_metrics)]  # type: ignore[index]
            summary[task][metric] = {"mean": mean(values), "std": stdev(values)}  # type: ignore[index]
    joint = [
        float(seed_metrics[seed]["joint_quality_inversion"]["accuracy"])  # type: ignore[index]
        for seed in sorted(seed_metrics)
    ]
    summary["joint_quality_inversion"] = {
        "accuracy": {"mean": mean(joint), "std": stdev(joint)}
    }
    normalized_nll = [
        float(seed_metrics[seed]["normalized_mean_nll"])
        for seed in sorted(seed_metrics)
    ]
    summary["normalized_mean_nll"] = {
        "mean": mean(normalized_nll),
        "std": stdev(normalized_nll),
    }
    return summary


__all__ = [
    "EntryPrediction",
    "aggregate_entry_predictions",
    "benchmark_metrics",
    "grouped_bootstrap",
    "joint_accuracy",
    "summarize_seeds",
    "task_metrics",
]
