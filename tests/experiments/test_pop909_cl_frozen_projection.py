"""Synthetic CPU checks for the frozen POP909-CL crosswalk."""

from dataclasses import replace

import pytest
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.pop909_cl_frozen import projection


def test_enharmonic_sum_counterexample_and_conservation():
    labels = projection.root_class_labels()
    weights = torch.full((37,), 0.001, dtype=torch.float64)
    weights[labels.index("C")] = 0.3
    weights[labels.index("B#")] = 0.3
    weights[labels.index("D")] = 0.4
    logits = weights.log()
    probabilities = projection.project_root_logits(logits)
    assert projection.root_pc_from_label(labels[int(logits.argmax())]) == 2
    assert int(projection.predict_root_ids(logits)) == 0
    assert probabilities.shape == (12,)
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    expected = torch.zeros(12, dtype=torch.float64)
    for class_id, label in enumerate(labels):
        expected[projection.root_pc_from_label(label)] += weights[class_id] / weights.sum()
    torch.testing.assert_close(probabilities, expected)


def test_projection_handles_batches_empty_rows_and_extreme_finite_logits():
    logits = torch.arange(2 * 3 * 37, dtype=torch.float64).reshape(2, 3, 37) * 10
    result = projection.project_root_logits(logits)
    assert result.shape == (2, 3, 12)
    torch.testing.assert_close(result.sum(-1), torch.ones(2, 3, dtype=torch.float64))
    assert projection.project_root_logits(torch.empty(0, 37)).shape == (0, 12)
    assert projection.predict_quality_ids(torch.empty(0, 17)).shape == (0,)


def test_root_tie_policy_is_lowest_pc_not_spelled_class_order():
    # A and C each have three spelled classes. A occurs first in registry;
    # identical maximum aggregate probability must nevertheless select PC C=0.
    labels = projection.root_class_labels()
    logits = torch.full((37,), -1000.0, dtype=torch.float64)
    for class_id, label in enumerate(labels):
        if projection.root_pc_from_label(label) in {0, 9}:
            logits[class_id] = 0
    probabilities = projection.project_root_logits(logits)
    assert probabilities[0] == probabilities[9]
    assert int(projection.predict_root_ids(logits)) == 0


@pytest.mark.parametrize("label,expected", [("C", 0), ("B#", 0), ("C###", 3), ("Bbbb", 8), ("Fbb", 3)])
def test_spelled_root_mapping(label, expected):
    assert projection.root_pc_from_label(label) == expected


@pytest.mark.parametrize("label", ["", "H", "c", "C#b", "C♯", None, 0])
def test_unknown_root_is_rejected(label):
    with pytest.raises(projection.ProjectionContractError, match="unsupported spelled root"):
        projection.root_pc_from_label(label)


def test_quality_interval_crosswalk_is_explicit_and_preserves_registry_ids():
    rows = projection.projection_contract()["quality"]["subset9"]
    assert [(row["pop909_label"], row["corrected_class_id"]) for row in rows] == [
        ("M", 0), ("m", 1), ("o", 2), ("+", 3), ("D7", 6),
        ("M7", 5), ("m7", 4), ("/o7", 9), ("o7", 8),
    ]
    expected_templates = [
        [0, 4, 7], [0, 3, 7], [0, 3, 6], [0, 4, 8], [0, 4, 7, 10],
        [0, 4, 7, 11], [0, 3, 7, 10], [0, 3, 6, 10], [0, 3, 6, 9],
    ]
    assert [row["intervals"] for row in rows] == expected_templates
    labels = projection.quality_class_labels()
    for row in rows:
        assert projection.quality_gold_class_id(row["pop909_label"]) == row["corrected_class_id"]
        assert labels[row["corrected_class_id"]] == row["corrected_label"]


@pytest.mark.parametrize("gold", ["sus2", "sus4", "mM7", "+7"])
def test_explicit_unsupported_gold(gold):
    assert projection.quality_gold_class_id(gold) is None


def test_unknown_quality_is_not_silently_excluded():
    with pytest.raises(projection.ProjectionContractError, match="unknown POP909-CL quality"):
        projection.quality_gold_class_id("major")


def test_quality_winner_outside_subset_is_retained_as_error():
    logits = torch.zeros((2, 17))
    logits[0, 0] = 5  # Best supported class is M; full17 wins augmented sixth.
    logits[0, 10] = 9
    logits[1, 15] = 8  # +7 is outside frozen subset even though corrected has it.
    prediction = projection.predict_quality_ids(logits)
    assert prediction.tolist() == [10, 15]
    assert 10 not in projection.quality_subset_ids()
    assert int(prediction[0]) != projection.quality_gold_class_id("M")
    assert int(projection.predict_quality_ids(torch.zeros(17))) == 0


def test_contract_is_fresh_deterministic_and_self_fingerprinted():
    first = projection.projection_contract()
    second = projection.projection_contract()
    assert first == second
    assert first["registry_fingerprint"] == projection.EXPECTED_REGISTRY_FINGERPRINT
    expected_fp = first.pop("fingerprint")
    assert fingerprint(first) == expected_fp
    first["root"]["class_labels"].reverse()
    assert projection.projection_contract() == second


@pytest.mark.parametrize("task_id", ["root", "quality"])
def test_registry_class_order_drift_is_rejected(monkeypatch, task_id):
    vocabulary_id = TASK_BY_ID[task_id].vocabulary_id
    original = get_vocabulary(vocabulary_id)
    altered = replace(original, labels=original.labels[::-1])
    monkeypatch.setattr(projection, "get_vocabulary", lambda key: altered if key == vocabulary_id else get_vocabulary(key))
    with pytest.raises(projection.ProjectionContractError, match="registry drift"):
        projection.projection_contract()
    with pytest.raises(projection.ProjectionContractError, match="registry drift"):
        projection.project_root_logits(torch.zeros(37))


def test_registry_task_route_drift_is_rejected(monkeypatch):
    monkeypatch.setitem(projection.TASK_BY_ID, "root", replace(TASK_BY_ID["root"], prediction_level="note"))
    with pytest.raises(projection.ProjectionContractError, match="registry drift"):
        projection.predict_quality_ids(torch.zeros(17))


@pytest.mark.parametrize("bad", [torch.zeros(38), torch.zeros(()), torch.zeros(37, dtype=torch.int64), torch.full((37,), float("nan")), torch.full((37,), float("inf"))])
def test_invalid_root_logits_rejected(bad):
    with pytest.raises(projection.ProjectionContractError):
        projection.project_root_logits(bad)


@pytest.mark.parametrize("bad", [torch.zeros(9), torch.zeros(15), torch.zeros(17, dtype=torch.int64), torch.full((17,), -float("inf"))])
def test_invalid_quality_logits_rejected(bad):
    with pytest.raises(projection.ProjectionContractError):
        projection.predict_quality_ids(bad)
