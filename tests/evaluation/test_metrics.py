from __future__ import annotations

import math

import pytest
import torch

from music_critic.evaluation.metrics import (
    CategoricalMetricAccumulator,
    MultilabelMetricAccumulator,
)


def _categorical_fixture():
    # Truth/prediction pairs: 0/0, 0/1, 1/1, 2/1, 2/2.
    targets = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    return logits, targets


def _f1_oracle(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2 * tp / denominator


def _categorical_f1_counterexample():
    # Class 0: supported, never predicted.
    # Class 1: TP=1, FP=1.
    # Class 2: unsupported, one false-positive prediction.
    # Class 3: absent from both truth and predictions.
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    predictions = torch.tensor([1, 2, 1], dtype=torch.long)
    logits = torch.full((3, 4), -3.0)
    logits[torch.arange(3), predictions] = 3.0
    return logits, targets


def _multilabel_f1_counterexample():
    targets = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
        ]
    )
    predictions = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, True, False, False],
        ]
    )
    logits = torch.where(
        predictions,
        torch.full(predictions.shape, 3.0),
        torch.full(predictions.shape, -3.0),
    )
    return logits, targets


def test_hand_computed_categorical_confusion_matrix() -> None:
    accumulator = CategoricalMetricAccumulator(("a", "b", "c"))
    logits, targets = _categorical_fixture()
    accumulator.update(logits, targets)
    result = accumulator.finalize()

    assert result["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]
    assert result["top1_accuracy"]["value"] == 3 / 5
    assert result["balanced_accuracy"]["value"] == (
        0.5 + 1.0 + 0.5
    ) / 3
    assert result["per_class"][1]["precision"]["value"] == 1 / 3
    assert result["per_class"][1]["recall"]["value"] == 1.0


def test_hand_computed_multilabel_counts() -> None:
    # sigmoid(logit) >= 0.5 is equivalent to logit >= 0.
    logits = torch.tensor(
        [
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, -2.0],
        ]
    )
    targets = torch.tensor(
        [
            [True, False],
            [False, True],
            [True, False],
        ]
    )
    accumulator = MultilabelMetricAccumulator(("x", "y"))
    accumulator.update(logits, targets)
    result = accumulator.finalize()

    assert [
        (
            item["tp"],
            item["fp"],
            item["fn"],
            item["tn"],
            item["support"],
        )
        for item in result["per_class"]
    ] == [(1, 1, 1, 0, 2), (1, 0, 0, 2, 1)]
    assert result["exact_match_accuracy"]["value"] == 1 / 3
    assert result["per_class"][0]["average_precision"]["value"] == (
        pytest.approx(7 / 12)
    )
    assert result["per_class"][1]["average_precision"]["value"] == 1.0
    assert result["average_precision"]["value"] == pytest.approx(19 / 24)
    assert result["exact_ap_score_group_count"] == 4


def test_categorical_f1_uses_direct_confusion_count_denominator() -> None:
    logits, targets = _categorical_f1_counterexample()
    accumulator = CategoricalMetricAccumulator(("a", "b", "c", "d"))
    accumulator.update(logits, targets)
    result = accumulator.finalize()
    confusion = result["confusion_matrix"]
    oracle = []
    for index in range(4):
        tp = confusion[index][index]
        fp = sum(row[index] for row in confusion) - tp
        fn = sum(confusion[index]) - tp
        oracle.append(_f1_oracle(tp, fp, fn))

    assert oracle == [0.0, 2 / 3, 0.0, None]
    assert [
        item["f1"]["value"] for item in result["per_class"]
    ] == oracle
    assert result["per_class"][0]["precision"]["value"] is None
    assert result["per_class"][0]["recall"]["value"] == 0.0
    assert result["per_class"][2]["precision"]["value"] == 0.0
    assert result["per_class"][2]["recall"]["value"] is None
    assert result["per_class"][3]["f1"]["undefined"]["category"] == (
        "zero_f1_denominator"
    )
    # The old precision/recall-gated implementation retained only class 1.
    assert result["macro_f1"]["value"] == 2 / 9
    assert 2 / 3 != result["macro_f1"]["value"]


def test_multilabel_f1_uses_direct_confusion_count_denominator() -> None:
    logits, targets = _multilabel_f1_counterexample()
    accumulator = MultilabelMetricAccumulator(("a", "b", "c", "d"))
    accumulator.update(logits, targets)
    result = accumulator.finalize()
    oracle = [
        _f1_oracle(item["tp"], item["fp"], item["fn"])
        for item in result["per_class"]
    ]

    assert oracle == [0.0, 2 / 3, 0.0, None]
    assert [
        item["f1"]["value"] for item in result["per_class"]
    ] == oracle
    assert result["per_class"][0]["precision"]["value"] is None
    assert result["per_class"][2]["recall"]["value"] is None
    assert result["per_class"][3]["f1"]["undefined"]["category"] == (
        "zero_f1_denominator"
    )
    assert result["macro_f1"]["value"] == 2 / 9
    assert 2 / 3 != result["macro_f1"]["value"]


def test_categorical_batch_partition_and_order_invariance() -> None:
    logits, targets = _categorical_fixture()
    whole = CategoricalMetricAccumulator(("a", "b", "c"))
    whole.update(logits, targets)

    partitioned = CategoricalMetricAccumulator(("a", "b", "c"))
    order = torch.tensor([4, 1, 3, 0, 2])
    partitioned.update(logits[order[:2]], targets[order[:2]])
    partitioned.update(logits[order[2:]], targets[order[2:]])

    assert partitioned.finalize() == whole.finalize()


def test_multilabel_batch_partition_and_order_invariance() -> None:
    logits, targets = _multilabel_f1_counterexample()
    whole = MultilabelMetricAccumulator(("a", "b", "c", "d"))
    whole.update(logits, targets)

    partitioned = MultilabelMetricAccumulator(("a", "b", "c", "d"))
    order = torch.tensor([2, 0, 1])
    partitioned.update(logits[order[:1]], targets[order[:1]])
    partitioned.update(logits[order[1:]], targets[order[1:]])

    assert partitioned.finalize() == whole.finalize()


def test_undefined_multilabel_metrics_are_null_with_reason() -> None:
    accumulator = MultilabelMetricAccumulator(("never_positive",))
    accumulator.update(
        torch.tensor([[-2.0], [-3.0]]),
        torch.tensor([[False], [False]]),
    )
    result = accumulator.finalize()

    for name in (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        assert result[name]["value"] is None
        assert result[name]["undefined"]["category"]
        assert result[name]["undefined"]["reason"]
    assert result["exact_match_accuracy"]["value"] == 1.0
    assert math.isfinite(result["bce_nll"]["value"])


def test_accumulators_retain_no_prediction_tensors() -> None:
    categorical = CategoricalMetricAccumulator(("a", "b", "c"))
    multilabel = MultilabelMetricAccumulator(("a", "b"))
    for _ in range(50):
        logits, targets = _categorical_fixture()
        categorical.update(logits, targets)
        multilabel.update(
            torch.zeros((5, 2)),
            torch.zeros((5, 2), dtype=torch.bool),
        )
    assert categorical.retained_prediction_tensor_count == 0
    assert categorical.retained_prediction_element_count == 0
    assert multilabel.retained_prediction_tensor_count == 0
    assert multilabel.retained_prediction_element_count == 0
