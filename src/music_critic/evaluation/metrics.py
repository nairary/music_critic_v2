"""Streaming supervised metrics with explicit undefined-value semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from music_critic.evaluation.contracts import (
    EvaluationContractError,
    metric_value,
)


def _ratio(
    numerator: int,
    denominator: int,
    *,
    category: str,
    reason: str,
) -> dict[str, Any]:
    if denominator == 0:
        return metric_value(None, category=category, reason=reason)
    return metric_value(numerator / denominator)


def _mean_defined(
    values: list[dict[str, Any]],
    *,
    category: str,
    reason: str,
) -> dict[str, Any]:
    defined = [
        float(item["value"])
        for item in values
        if item["value"] is not None
    ]
    if not defined:
        return metric_value(None, category=category, reason=reason)
    return metric_value(math.fsum(defined) / len(defined))


@dataclass(slots=True)
class _ExactFloatSum:
    """Order- and batch-partition-invariant sum of binary64 observations."""

    value: Fraction = field(default_factory=Fraction)

    def add_tensor(self, values: Tensor) -> None:
        for value in values.detach().to(dtype=torch.float64, device="cpu").view(
            -1
        ):
            number = float(value)
            if not math.isfinite(number):
                raise EvaluationContractError(
                    "evaluation.metrics.non_finite_observation"
                )
            self.value += Fraction.from_float(number)

    def as_float(self) -> float:
        return float(self.value)


@dataclass(slots=True)
class CategoricalMetricAccumulator:
    """Fixed-memory categorical confusion and likelihood accumulator."""

    class_labels: tuple[str, ...]
    confusion: list[list[int]] = field(init=False)
    nll_sum: _ExactFloatSum = field(default_factory=_ExactFloatSum)
    row_count: int = 0
    top3_correct: int = 0

    def __post_init__(self) -> None:
        if not self.class_labels or len(self.class_labels) != len(
            set(self.class_labels)
        ):
            raise EvaluationContractError(
                "evaluation.metrics.class_labels_invalid"
            )
        self.confusion = [
            [0 for _ in self.class_labels] for _ in self.class_labels
        ]

    @property
    def retained_prediction_tensor_count(self) -> int:
        return 0

    @property
    def retained_prediction_element_count(self) -> int:
        return 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        class_count = len(self.class_labels)
        if (
            logits.ndim != 2
            or logits.shape[1] != class_count
            or targets.ndim != 1
            or targets.shape[0] != logits.shape[0]
            or targets.dtype != torch.long
        ):
            raise EvaluationContractError(
                "evaluation.metrics.categorical_shape_invalid"
            )
        if targets.numel() and (
            int(targets.min()) < 0
            or int(targets.max()) >= class_count
        ):
            raise EvaluationContractError(
                "evaluation.metrics.categorical_target_invalid"
            )
        if logits.shape[0] == 0:
            return
        detached_logits = logits.detach()
        detached_targets = targets.detach().to(logits.device)
        log_probabilities = F.log_softmax(detached_logits, dim=-1)
        losses = -log_probabilities.gather(
            1, detached_targets[:, None]
        ).squeeze(1)
        self.nll_sum.add_tensor(losses)
        predictions = detached_logits.argmax(dim=-1)
        flat = (
            detached_targets.to(torch.long) * class_count
            + predictions.to(torch.long)
        )
        counts = torch.bincount(
            flat.to("cpu"), minlength=class_count * class_count
        ).reshape(class_count, class_count)
        for truth in range(class_count):
            for prediction in range(class_count):
                self.confusion[truth][prediction] += int(
                    counts[truth, prediction]
                )
        if class_count >= 3:
            top3 = detached_logits.topk(3, dim=-1).indices
            self.top3_correct += int(
                (top3 == detached_targets[:, None]).any(dim=1).sum()
            )
        self.row_count += int(logits.shape[0])

    def finalize(self) -> dict[str, Any]:
        class_count = len(self.class_labels)
        per_class = []
        recalls: list[dict[str, Any]] = []
        f1_values: list[dict[str, Any]] = []
        correct = sum(self.confusion[index][index] for index in range(class_count))
        for index, label in enumerate(self.class_labels):
            tp = self.confusion[index][index]
            support = sum(self.confusion[index])
            predicted = sum(row[index] for row in self.confusion)
            fp = predicted - tp
            fn = support - tp
            precision = _ratio(
                tp,
                tp + fp,
                category="zero_predicted_positive",
                reason="the class has no predicted rows",
            )
            recall = _ratio(
                tp,
                tp + fn,
                category="zero_true_support",
                reason="the class has no eligible true rows",
            )
            if precision["value"] is None or recall["value"] is None:
                f1 = metric_value(
                    None,
                    category="undefined_precision_or_recall",
                    reason=(
                        "class F1 requires both defined precision and recall"
                    ),
                )
            else:
                denominator = float(precision["value"]) + float(
                    recall["value"]
                )
                f1 = (
                    metric_value(0.0)
                    if denominator == 0
                    else metric_value(
                        2
                        * float(precision["value"])
                        * float(recall["value"])
                        / denominator
                    )
                )
            recalls.append(recall)
            f1_values.append(f1)
            per_class.append(
                {
                    "class_index": index,
                    "class_label": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                    "predicted_count": predicted,
                }
            )
        if self.row_count:
            nll = metric_value(self.nll_sum.as_float() / self.row_count)
            accuracy = metric_value(correct / self.row_count)
        else:
            nll = metric_value(
                None,
                category="no_eligible_rows",
                reason="no eligible categorical rows were observed",
            )
            accuracy = metric_value(
                None,
                category="no_eligible_rows",
                reason="no eligible categorical rows were observed",
            )
        top3 = (
            metric_value(
                None,
                category="class_count_below_k",
                reason="top-3 requires at least three classes",
            )
            if class_count < 3
            else (
                metric_value(self.top3_correct / self.row_count)
                if self.row_count
                else metric_value(
                    None,
                    category="no_eligible_rows",
                    reason="no eligible categorical rows were observed",
                )
            )
        )
        return {
            "kind": "closed_categorical",
            "eligible_row_count": self.row_count,
            "nll": nll,
            "top1_accuracy": accuracy,
            "top3_accuracy": top3,
            "balanced_accuracy": _mean_defined(
                recalls,
                category="no_supported_classes",
                reason="no class has eligible true support",
            ),
            "macro_f1": _mean_defined(
                f1_values,
                category="no_defined_class_f1",
                reason="no class has a defined F1",
            ),
            "micro_f1": accuracy,
            "per_class": per_class,
            "confusion_matrix": [list(row) for row in self.confusion],
            "retained_prediction_tensor_count": 0,
            "retained_prediction_element_count": 0,
        }


@dataclass(slots=True)
class MultilabelMetricAccumulator:
    """Fixed-memory multilabel threshold metrics and BCE accumulator."""

    class_labels: tuple[str, ...]
    threshold: float = 0.5
    tp: list[int] = field(init=False)
    fp: list[int] = field(init=False)
    fn: list[int] = field(init=False)
    tn: list[int] = field(init=False)
    bce_sum: _ExactFloatSum = field(default_factory=_ExactFloatSum)
    row_count: int = 0
    exact_match_count: int = 0

    def __post_init__(self) -> None:
        if (
            not self.class_labels
            or len(self.class_labels) != len(set(self.class_labels))
            or self.threshold != 0.5
        ):
            raise EvaluationContractError(
                "evaluation.metrics.multilabel_contract_invalid"
            )
        count = len(self.class_labels)
        self.tp = [0] * count
        self.fp = [0] * count
        self.fn = [0] * count
        self.tn = [0] * count

    @property
    def retained_prediction_tensor_count(self) -> int:
        return 0

    @property
    def retained_prediction_element_count(self) -> int:
        return 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        class_count = len(self.class_labels)
        if (
            logits.ndim != 2
            or logits.shape[1] != class_count
            or targets.shape != logits.shape
            or targets.dtype != torch.bool
        ):
            raise EvaluationContractError(
                "evaluation.metrics.multilabel_shape_invalid"
            )
        if logits.shape[0] == 0:
            return
        detached_logits = logits.detach()
        truth = targets.detach().to(device=logits.device)
        losses = F.binary_cross_entropy_with_logits(
            detached_logits, truth.float(), reduction="none"
        )
        self.bce_sum.add_tensor(losses)
        predicted = torch.sigmoid(detached_logits) >= self.threshold
        self.exact_match_count += int(
            (predicted == truth).all(dim=1).sum()
        )
        for index in range(class_count):
            pred = predicted[:, index]
            actual = truth[:, index]
            self.tp[index] += int((pred & actual).sum())
            self.fp[index] += int((pred & ~actual).sum())
            self.fn[index] += int((~pred & actual).sum())
            self.tn[index] += int((~pred & ~actual).sum())
        self.row_count += int(logits.shape[0])

    def finalize(self) -> dict[str, Any]:
        precisions = []
        recalls = []
        f1_values = []
        per_class = []
        for index, label in enumerate(self.class_labels):
            precision = _ratio(
                self.tp[index],
                self.tp[index] + self.fp[index],
                category="zero_predicted_positive",
                reason="the class has no predicted positive rows",
            )
            recall = _ratio(
                self.tp[index],
                self.tp[index] + self.fn[index],
                category="zero_true_positive",
                reason="the class has no eligible positive labels",
            )
            if precision["value"] is None or recall["value"] is None:
                f1 = metric_value(
                    None,
                    category="undefined_precision_or_recall",
                    reason=(
                        "class F1 requires both defined precision and recall"
                    ),
                )
            else:
                denominator = float(precision["value"]) + float(
                    recall["value"]
                )
                f1 = (
                    metric_value(0.0)
                    if denominator == 0
                    else metric_value(
                        2
                        * float(precision["value"])
                        * float(recall["value"])
                        / denominator
                    )
                )
            precisions.append(precision)
            recalls.append(recall)
            f1_values.append(f1)
            per_class.append(
                {
                    "class_index": index,
                    "class_label": label,
                    "tp": self.tp[index],
                    "fp": self.fp[index],
                    "fn": self.fn[index],
                    "tn": self.tn[index],
                    "support": self.tp[index] + self.fn[index],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        total_tp = sum(self.tp)
        total_fp = sum(self.fp)
        total_fn = sum(self.fn)
        micro_precision = _ratio(
            total_tp,
            total_tp + total_fp,
            category="zero_predicted_positive",
            reason="no positive label was predicted",
        )
        micro_recall = _ratio(
            total_tp,
            total_tp + total_fn,
            category="zero_true_positive",
            reason="no eligible positive label exists",
        )
        micro_f1 = _ratio(
            2 * total_tp,
            2 * total_tp + total_fp + total_fn,
            category="zero_positive_union",
            reason="neither truth nor prediction contains a positive label",
        )
        class_observations = self.row_count * len(self.class_labels)
        bce = (
            metric_value(self.bce_sum.as_float() / class_observations)
            if class_observations
            else metric_value(
                None,
                category="no_eligible_rows",
                reason="no eligible multilabel rows were observed",
            )
        )
        exact = (
            metric_value(self.exact_match_count / self.row_count)
            if self.row_count
            else metric_value(
                None,
                category="no_eligible_rows",
                reason="no eligible multilabel rows were observed",
            )
        )
        return {
            "kind": "closed_multilabel",
            "threshold": self.threshold,
            "eligible_row_count": self.row_count,
            "eligible_label_count": class_observations,
            "bce_nll": bce,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": _mean_defined(
                precisions,
                category="no_defined_class_precision",
                reason="no class has defined precision",
            ),
            "macro_recall": _mean_defined(
                recalls,
                category="no_positive_class_support",
                reason="no class has eligible positive support",
            ),
            "macro_f1": _mean_defined(
                f1_values,
                category="no_defined_class_f1",
                reason="no class has a defined F1",
            ),
            "exact_match_accuracy": exact,
            "per_class": per_class,
            "retained_prediction_tensor_count": 0,
            "retained_prediction_element_count": 0,
        }


def make_metric_accumulator(
    encoding_kind: str,
    class_labels: tuple[str, ...],
) -> CategoricalMetricAccumulator | MultilabelMetricAccumulator:
    if encoding_kind == "closed_categorical_index":
        return CategoricalMetricAccumulator(class_labels)
    if encoding_kind == "closed_multilabel":
        return MultilabelMetricAccumulator(class_labels)
    raise EvaluationContractError(
        f"evaluation.metrics.encoding_unsupported:{encoding_kind}"
    )


__all__ = [
    "CategoricalMetricAccumulator",
    "MultilabelMetricAccumulator",
    "make_metric_accumulator",
]
