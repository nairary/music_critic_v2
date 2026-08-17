"""Source-entry Dilemmadata evaluation and component-level comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import random
from typing import Iterable, Mapping

import torch
from torch.nn import functional as F

from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    DilemmadataHierarchicalModel,
    join_task_supervision,
)
from music_critic.tasks import MultiSourceBatch


DILEMMADATA_EVALUATION_CONTRACT_VERSION = "1.0.0"
DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION = "1.0.0"
DILEMMADATA_COMPONENT_BOOTSTRAP_VERSION = "1.0.0"
DILEMMADATA_TEST_LOCK_VERSION = "1.0.0"


class DilemmadataEvaluationError(ValueError):
    """Stable evaluation boundary error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _with_fingerprint(value: dict[str, object]) -> dict[str, object]:
    return {**value, "fingerprint": _fingerprint(value)}


def make_dilemmadata_test_unlock(
    test_membership_fingerprint: str,
) -> dict[str, object]:
    if not _is_sha256(test_membership_fingerprint):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.test_membership_fingerprint_invalid"
        )
    artifact = {
        "contract_version": DILEMMADATA_TEST_LOCK_VERSION,
        "split": "test",
        "test_membership_fingerprint": test_membership_fingerprint,
        "authorization": "explicit_post_selection_test_evaluation",
    }
    return _with_fingerprint(artifact)


def _validate_split_lock(
    split: str,
    membership_fingerprint: str,
    test_unlock: Mapping[str, object] | None,
) -> None:
    if not _is_sha256(membership_fingerprint):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.membership_fingerprint_invalid"
        )
    if split == "validation":
        if test_unlock is not None:
            raise DilemmadataEvaluationError(
                "dilemmadata.evaluation.validation_unlock_forbidden"
            )
        return
    if split != "test" or test_unlock is None:
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.test_locked"
        )
    payload = dict(test_unlock)
    fingerprint = payload.pop("fingerprint", None)
    if (
        fingerprint != _fingerprint(payload)
        or payload.get("contract_version") != DILEMMADATA_TEST_LOCK_VERSION
        or payload.get("split") != "test"
        or payload.get("test_membership_fingerprint")
        != membership_fingerprint
    ):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.test_unlock_binding_mismatch"
        )


def _entry_rows(
    model: DilemmadataHierarchicalModel,
    batch: MultiSourceBatch,
    component_by_identity: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    # Prediction is deliberately completed before the first target-sidecar read.
    _, predictions = model.predict(batch.raw_graph_batch)
    supervisions = join_task_supervision(predictions, batch.target_batches)
    targets = {target.task_id: target for target in batch.target_batches}
    prediction_by_task = {row.task_id: row for row in predictions}
    rows: list[dict[str, object]] = []
    for supervision in supervisions:
        prediction = prediction_by_task[supervision.task_id]
        target = targets[supervision.task_id]
        logits = prediction.logits.index_select(
            0, supervision.candidate_indices
        )
        log_probabilities = F.log_softmax(logits, dim=-1)
        labels = target.values.to(logits.device).index_select(
            0, supervision.target_row_indices
        )
        keys = torch.stack(
            (supervision.sample_indices, supervision.source_entry_indices),
            dim=1,
        )
        unique, inverse, counts = torch.unique(
            keys,
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        sums = torch.zeros(
            (unique.shape[0], log_probabilities.shape[1]),
            dtype=log_probabilities.dtype,
            device=log_probabilities.device,
        )
        sums.index_add_(0, inverse, log_probabilities)
        means = sums / counts[:, None].to(sums.dtype)
        first_positions = torch.full(
            (unique.shape[0],),
            labels.shape[0],
            dtype=torch.long,
            device=labels.device,
        )
        first_positions.scatter_reduce_(
            0,
            inverse,
            torch.arange(labels.shape[0], device=labels.device),
            reduce="amin",
            include_self=True,
        )
        entry_labels = labels.index_select(0, first_positions)
        if not torch.equal(entry_labels.index_select(0, inverse), labels):
            raise DilemmadataEvaluationError(
                "dilemmadata.evaluation.source_entry_label_conflict"
            )
        host_unique = unique.detach().cpu().tolist()
        host_counts = counts.detach().cpu().tolist()
        host_labels = entry_labels.detach().cpu().tolist()
        host_means = means.detach().cpu().tolist()
        for key, count, label, log_probs in zip(
            host_unique,
            host_counts,
            host_labels,
            host_means,
            strict=True,
        ):
            sample_index, source_entry_index = key
            identity = (
                batch.dataset_ids[sample_index],
                batch.piece_ids[sample_index],
            )
            component = component_by_identity.get(identity)
            if not isinstance(component, str) or not component:
                raise DilemmadataEvaluationError(
                    "dilemmadata.evaluation.component_binding_missing:"
                    + ":".join(identity)
                )
            rows.append(
                {
                    "task_id": supervision.task_id,
                    "dataset_id": identity[0],
                    "piece_id": identity[1],
                    "component_fingerprint": component,
                    "source_entry_index": source_entry_index,
                    "expanded_row_count": count,
                    "label": label,
                    "log_probabilities": log_probs,
                }
            )
    return rows


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _categorical_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "available": False,
            "undefined_reason": "zero_source_entries",
            "source_entry_count": 0,
            "expanded_row_count": 0,
        }
    class_count = len(rows[0]["log_probabilities"])
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    nll = 0.0
    top3_correct = 0
    is_quality = str(rows[0]["task_id"]).endswith(".quality")
    for row in rows:
        probabilities = row["log_probabilities"]
        if len(probabilities) != class_count:
            raise DilemmadataEvaluationError(
                "dilemmadata.evaluation.class_count_changed"
            )
        label = int(row["label"])
        prediction = max(range(class_count), key=probabilities.__getitem__)
        confusion[label][prediction] += 1
        nll -= float(probabilities[label])
        if is_quality:
            top = sorted(
                range(class_count),
                key=probabilities.__getitem__,
                reverse=True,
            )[: min(3, class_count)]
            top3_correct += int(label in top)
    per_class = []
    f1_values = []
    weighted_f1_sum = 0.0
    recalls = []
    total = len(rows)
    correct = sum(confusion[index][index] for index in range(class_count))
    for class_id in range(class_count):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted = sum(row[class_id] for row in confusion)
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        if support and f1 is None:
            f1 = 0.0
        if f1 is not None:
            f1_values.append(f1)
            weighted_f1_sum += support * f1
        if recall is not None:
            recalls.append(recall)
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "predicted_count": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "undefined_reason": (
                    "zero_support" if support == 0 else None
                ),
            }
        )
    return {
        "available": True,
        "undefined_reason": None,
        "macro_f1_rule": "supported_true_classes_v1",
        "source_entry_count": total,
        "expanded_row_count": sum(
            int(row["expanded_row_count"]) for row in rows
        ),
        "class_count": class_count,
        "nll": nll / total,
        "top1_accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "weighted_f1": weighted_f1_sum / total,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "top3_accuracy": (
            top3_correct / total
            if is_quality
            else None
        ),
        "top3_undefined_reason": (
            None if is_quality else "not_applicable_non_quality_task"
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def build_dilemmadata_train_priors(
    rows: Iterable[Mapping[str, object]],
    *,
    train_membership_fingerprint: str,
) -> dict[str, object]:
    if not _is_sha256(train_membership_fingerprint):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.train_membership_fingerprint_invalid"
        )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    if set(grouped) != set(DILEMMADATA_ACTIVE_TASK_IDS):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.train_prior_inventory_incomplete"
        )
    tasks = {}
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        task_rows = grouped[task_id]
        class_count = len(task_rows[0]["log_probabilities"])
        counts = [0 for _ in range(class_count)]
        for row in task_rows:
            counts[int(row["label"])] += 1
        total = sum(counts)
        tasks[task_id] = {
            "class_counts": counts,
            "class_probabilities": [count / total for count in counts],
            "majority_class_id": max(
                range(class_count), key=counts.__getitem__
            ),
            "source_entry_count": total,
        }
    artifact = {
        "contract_version": DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION,
        "source_split": "train_only",
        "train_membership_fingerprint": train_membership_fingerprint,
        "tasks": tasks,
    }
    return _with_fingerprint(artifact)


def _validate_priors(priors: Mapping[str, object]) -> None:
    payload = dict(priors)
    fingerprint = payload.pop("fingerprint", None)
    if (
        fingerprint != _fingerprint(payload)
        or payload.get("contract_version")
        != DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION
        or payload.get("source_split") != "train_only"
        or set(payload.get("tasks", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.train_prior_invalid"
        )


def _baseline_metrics(
    rows: list[dict[str, object]], priors: Mapping[str, object]
) -> dict[str, object]:
    task = priors["tasks"][rows[0]["task_id"]]
    probabilities = task["class_probabilities"]
    majority = int(task["majority_class_id"])
    correct = sum(int(row["label"] == majority) for row in rows)
    finite_nll = []
    zero_probability_count = 0
    for row in rows:
        probability = float(probabilities[int(row["label"])])
        if probability == 0:
            zero_probability_count += 1
        else:
            finite_nll.append(-math.log(probability))
    return {
        "source": "train_only",
        "majority_top1_accuracy": correct / len(rows),
        "empirical_prior_nll": (
            sum(finite_nll) / len(rows)
            if not zero_probability_count
            else None
        ),
        "empirical_prior_nll_undefined_reason": (
            "zero_train_probability"
            if zero_probability_count
            else None
        ),
        "zero_train_probability_count": zero_probability_count,
    }


def evaluate_dilemmadata_model(
    model: DilemmadataHierarchicalModel,
    batches: Iterable[MultiSourceBatch],
    *,
    component_by_identity: Mapping[tuple[str, str], str],
    split: str = "validation",
    membership_fingerprint: str,
    train_priors: Mapping[str, object] | None = None,
    test_unlock: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_split_lock(split, membership_fingerprint, test_unlock)
    if train_priors is not None:
        _validate_priors(train_priors)
    model.eval()
    rows: list[dict[str, object]] = []
    alignment_counts: dict[str, Counter[str]] = {
        task_id: Counter() for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    }
    with torch.no_grad():
        for batch in batches:
            rows.extend(_entry_rows(model, batch, component_by_identity))
            for statistics in batch.statistics.task_counts:
                if statistics.task_id not in alignment_counts:
                    continue
                alignment_counts[statistics.task_id].update(
                    {
                        "source_entry_count": statistics.source_entry_count,
                        "expanded_target_row_count": statistics.target_row_count,
                        "eligible_expanded_row_count": (
                            statistics.supervision_eligible_row_count
                        ),
                        "effective_source_entry_count": (
                            statistics.effective_source_entry_count
                        ),
                        "masked_row_count": statistics.masked_row_count,
                        "conflict_row_count": statistics.conflict_row_count,
                        "unaligned_available_row_count": (
                            statistics.available_unaligned_row_count
                        ),
                    }
                )
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    tasks = {}
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        task_rows = by_task.get(task_id, [])
        metrics = _categorical_metrics(task_rows)
        record_metrics = {
            key: _categorical_metrics(group)
            for key, group in sorted(
                _group_rows(task_rows, "piece_id").items()
            )
        }
        component_metrics = {
            key: _categorical_metrics(group)
            for key, group in sorted(
                _group_rows(task_rows, "component_fingerprint").items()
            )
        }
        tasks[task_id] = {
            **metrics,
            "alignment_counts": dict(
                sorted(alignment_counts[task_id].items())
            ),
            "record_metrics": record_metrics,
            "component_metrics": component_metrics,
            "train_only_baselines": (
                None
                if train_priors is None or not task_rows
                else _baseline_metrics(task_rows, train_priors)
            ),
        }
    projection = {
        "contract_version": DILEMMADATA_EVALUATION_CONTRACT_VERSION,
        "split": split,
        "membership_fingerprint": membership_fingerprint,
        "validation_only_default": True,
        "test_unlock_fingerprint": (
            None if test_unlock is None else test_unlock["fingerprint"]
        ),
        "train_prior_fingerprint": (
            None if train_priors is None else train_priors["fingerprint"]
        ),
        "tasks": tasks,
        "counts": {
            "source_entry_count": len(rows),
            "expanded_row_count": sum(
                int(row["expanded_row_count"]) for row in rows
            ),
            "record_count": len({row["piece_id"] for row in rows}),
            "component_count": len(
                {row["component_fingerprint"] for row in rows}
            ),
            "dataset_counts": dict(
                sorted(Counter(str(row["dataset_id"]) for row in rows).items())
            ),
            "eligible_expanded_row_count": sum(
                values["eligible_expanded_row_count"]
                for values in alignment_counts.values()
            ),
            "masked_row_count": sum(
                values["masked_row_count"]
                for values in alignment_counts.values()
            ),
            "conflict_row_count": sum(
                values["conflict_row_count"]
                for values in alignment_counts.values()
            ),
            "unaligned_available_row_count": sum(
                values["unaligned_available_row_count"]
                for values in alignment_counts.values()
            ),
        },
        "entry_predictions": rows,
    }
    return _with_fingerprint(projection)


def _group_rows(
    rows: list[dict[str, object]], key: str
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def paired_component_bootstrap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    seed: int,
    replicates: int = 2000,
) -> dict[str, object]:
    if replicates <= 0 or isinstance(replicates, bool):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.bootstrap_replicates_invalid"
        )
    left_rows = left.get("entry_predictions")
    right_rows = right.get("entry_predictions")
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.bootstrap_predictions_missing"
        )
    def identity(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            row["task_id"],
            row["dataset_id"],
            row["piece_id"],
            row["source_entry_index"],
            row["label"],
        )
    left_by_id = {identity(row): row for row in left_rows}
    right_by_id = {identity(row): row for row in right_rows}
    if set(left_by_id) != set(right_by_id):
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.bootstrap_pairing_mismatch"
        )
    component_deltas: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_id):
        left_row = left_by_id[key]
        right_row = right_by_id[key]
        component = str(left_row["component_fingerprint"])
        if component != right_row["component_fingerprint"]:
            raise DilemmadataEvaluationError(
                "dilemmadata.evaluation.bootstrap_component_mismatch"
            )
        label = int(left_row["label"])
        left_correct = int(
            max(
                range(len(left_row["log_probabilities"])),
                key=left_row["log_probabilities"].__getitem__,
            )
            == label
        )
        right_correct = int(
            max(
                range(len(right_row["log_probabilities"])),
                key=right_row["log_probabilities"].__getitem__,
            )
            == label
        )
        component_deltas[component].append(right_correct - left_correct)
    components = sorted(component_deltas)
    if not components:
        raise DilemmadataEvaluationError(
            "dilemmadata.evaluation.bootstrap_components_empty"
        )
    values = [
        sum(component_deltas[key]) / len(component_deltas[key])
        for key in components
    ]
    generator = random.Random(seed)
    draws = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    lower = draws[int(0.025 * (replicates - 1))]
    upper = draws[int(0.975 * (replicates - 1))]
    artifact = {
        "contract_version": DILEMMADATA_COMPONENT_BOOTSTRAP_VERSION,
        "unit": "connected_component",
        "metric": "top1_accuracy_delta_right_minus_left",
        "seed": seed,
        "replicates": replicates,
        "component_count": len(values),
        "mean_delta": sum(values) / len(values),
        "confidence_interval_95": [lower, upper],
    }
    return _with_fingerprint(artifact)


__all__ = [
    "DILEMMADATA_COMPONENT_BOOTSTRAP_VERSION",
    "DILEMMADATA_EVALUATION_CONTRACT_VERSION",
    "DILEMMADATA_TEST_LOCK_VERSION",
    "DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION",
    "DilemmadataEvaluationError",
    "build_dilemmadata_train_priors",
    "evaluate_dilemmadata_model",
    "make_dilemmadata_test_unlock",
    "paired_component_bootstrap",
]
