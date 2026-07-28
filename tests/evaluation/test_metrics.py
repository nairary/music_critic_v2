from __future__ import annotations

import math

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


def test_batch_partition_and_order_invariance() -> None:
    logits, targets = _categorical_fixture()
    whole = CategoricalMetricAccumulator(("a", "b", "c"))
    whole.update(logits, targets)

    partitioned = CategoricalMetricAccumulator(("a", "b", "c"))
    order = torch.tensor([4, 1, 3, 0, 2])
    partitioned.update(logits[order[:2]], targets[order[:2]])
    partitioned.update(logits[order[2:]], targets[order[2:]])

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
