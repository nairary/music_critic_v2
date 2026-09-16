"""Synthetic mathematical oracles; no datasets, model loading or inference."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import random

import pytest

from music_critic.experiments.context_diagnostics import Member
from music_critic.experiments import context_local_uncertainty as module
from music_critic.experiments.context_local_uncertainty import paired_component_report


def member(piece, component=None):
    return Member(piece, component or piece, "validation")


def observation(piece, identity="e", *, component=None, gold=0, local=0, reference=1, **changes):
    row = {"piece_id": piece, "component_id": component or piece, "observation_id": identity,
           "candidate_id": "raw:beat:1", "gold": gold, "local": local, "reference": reference,
           "target_provenance": "synthetic_manual", "target_confidence": .8}
    row.update(changes)
    return row


def weighted_fixture():
    members = [member("a", "work-a"), member("b", "work-a"), member("c", "work-b")]
    rows = [observation("a", "e1", component="work-a"), observation("a", "e2", component="work-a"),
            observation("b", component="work-a"), observation("c", component="work-b", local=1, reference=0)]
    return rows, members


def test_hand_point_oracle_same_rows_and_record_component_support():
    rows, members = weighted_fixture()
    report = paired_component_report(rows, ["present", "absent"], members, draws=5)
    local, reference = report["point"]["local"], report["point"]["reference"]
    assert local["confusion_gold_rows_prediction_columns"] == [[3, 1], [0, 0]]
    assert reference["confusion_gold_rows_prediction_columns"] == [[1, 3], [0, 0]]
    assert local["accuracy"] == .75 and reference["accuracy"] == .25
    assert local["observed_class_macro_f1"] == 6 / 7
    assert reference["observed_class_macro_f1"] == 2 / 5
    assert report["point"]["reference_minus_local"]["accuracy"] == -.5
    assert report["point"]["reference_minus_local"]["observed_class_macro_f1"] == pytest.approx(-16 / 35)
    assert report["support"]["eligible_records"] == 3
    assert report["support"]["eligible_components"] == 2
    assert local["classes"][0]["gold_record_support"] == 3
    assert local["classes"][0]["gold_component_support"] == 2
    assert local["classes"][0]["prediction_record_support"] == 2
    assert local["classes"][0]["prediction_component_support"] == 1
    assert local["classes"][1]["fp"] == 1  # Keep false positives for absent gold classes.
    for arm in ("local", "reference", "reference_minus_local"):
        intervals = report["bootstrap"]["per_class"][1]["intervals"][arm]
        assert all(value["point_estimate"] is None and value["lower"] is None for value in intervals.values())


def test_paired_whole_component_draws_and_manual_percentile_oracle(monkeypatch):
    # Five exact shared draws: AA, AB, BB, AB, AB. Components, never events.
    order = iter([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    class OracleRandom:
        def randrange(self, count):
            assert count == 2
            return next(order)
    monkeypatch.setattr(module.random, "Random", lambda seed: OracleRandom())
    rows, members = weighted_fixture()
    report = paired_component_report(rows, ["a", "b"], members, draws=5)
    draws = report["bootstrap"]["draws"]
    assert [row["component_multiplicities"] for row in draws] == [[2, 0], [1, 1], [0, 2], [1, 1], [1, 1]]
    assert [row["eligible_observations"] for row in draws] == [6, 4, 2, 4, 4]
    assert [row["reference_minus_local"]["accuracy"] for row in draws] == [-1, -.5, 1, -.5, -.5]
    for row in draws:
        a, b = row["component_multiplicities"]
        assert row["local"]["accuracy"] == 3 * a / (3 * a + b)
        assert row["reference"]["accuracy"] == b / (3 * a + b)
        assert row["reference_minus_local"]["observed_class_macro_f1"] == pytest.approx(
            2 * b / (3 * a + 2 * b) - 6 * a / (6 * a + b))
    interval = report["bootstrap"]["intervals"]["reference_minus_local"]["accuracy"]
    assert interval["lower"] == pytest.approx(-.95)
    assert interval["upper"] == pytest.approx(.85)
    assert interval["defined_draws"] == 5 and interval["undefined_draws"] == 0
    assert not interval["conditional_on_defined_draws"]


def test_fixed_macro_classes_do_not_change_when_a_draw_lacks_one_gold_class(monkeypatch):
    order = iter([0, 0, 0, 1, 1, 1])
    class OracleRandom:
        def randrange(self, _count):
            return next(order)
    monkeypatch.setattr(module.random, "Random", lambda seed: OracleRandom())
    rows = [observation("a", gold=0, local=0, reference=1),
            observation("b", gold=1, local=1, reference=0)]
    report = paired_component_report(rows, ["a", "b", "point_absent"], [member("a"), member("b")], draws=3)
    assert report["bootstrap"]["fixed_macro_gold_class_ids"] == [0, 1]
    for index in (0, 2):
        for arm in ("local", "reference", "reference_minus_local"):
            assert report["bootstrap"]["draws"][index][arm]["observed_class_macro_f1"] is None
    delta = report["bootstrap"]["draws"][1]["reference_minus_local"]["observed_class_macro_f1"]
    assert delta == -1
    interval = report["bootstrap"]["intervals"]["reference_minus_local"]["observed_class_macro_f1"]
    assert interval["defined_draws"] == 1 and interval["undefined_draws"] == 2
    assert interval["lower"] is None and interval["undefined_reason"] == "fewer_than_two_defined_draws"
    # Each class occurs in one work: two total works do not justify class intervals.
    for row in report["bootstrap"]["per_class"][:2]:
        assert row["intervals"]["local"]["recall"]["undefined_reason"] == "fewer_than_two_eligible_components"


def test_zero_eligible_components_remain_in_official_sampling_frame(monkeypatch):
    order = iter([0, 1, 2, 2, 2, 2, 0, 0, 1])
    class OracleRandom:
        def randrange(self, count):
            assert count == 3
            return next(order)
    monkeypatch.setattr(module.random, "Random", lambda seed: OracleRandom())
    report = paired_component_report([observation("a"), observation("b")], ["a", "b"],
                                    [member("a"), member("b"), member("z")], draws=3)
    assert report["support"]["official_validation_components"] == 3
    assert report["support"]["eligible_components"] == 2
    assert report["members"][-1]["eligible_observations"] == 0
    assert report["components"][-1]["confusion_counts"] == {"local": [], "reference": []}
    empty = report["bootstrap"]["draws"][1]
    assert empty["eligible_observations"] == 0
    assert all(empty[arm][metric] is None for arm in ("local", "reference", "reference_minus_local")
               for metric in ("accuracy", "observed_class_macro_f1"))
    interval = report["bootstrap"]["intervals"]["reference_minus_local"]["accuracy"]
    assert interval["lower"] == interval["upper"] == -1
    assert interval["defined_draws"] == 2 and interval["undefined_draws"] == 1
    assert interval["conditional_on_defined_draws"]


@pytest.mark.parametrize("rows", [[], [observation("a")]])
def test_empty_or_single_eligible_component_has_no_interval(rows):
    report = paired_component_report(rows, ["a", "b"], [member("a"), member("empty")], draws=10)
    for arm in ("local", "reference", "reference_minus_local"):
        for interval in report["bootstrap"]["intervals"][arm].values():
            assert interval["lower"] is interval["upper"] is None
    if not rows:
        assert report["point"]["local"]["accuracy"] is None
        assert report["point"]["reference_minus_local"]["observed_class_macro_f1"] is None
        assert report["bootstrap"]["fixed_macro_gold_class_ids"] == []
    json.dumps(report, allow_nan=False)


def test_two_records_in_one_component_are_not_two_independent_groups():
    report = paired_component_report([observation("a", component="same"), observation("b", component="same")],
                                    ["a", "b"], [member("a", "same"), member("b", "same")])
    assert report["support"]["eligible_records"] == 2
    assert report["support"]["eligible_components"] == 1
    interval = report["bootstrap"]["intervals"]["reference_minus_local"]["accuracy"]
    assert interval["lower"] is None


def test_determinism_order_independence_rng_and_input_preservation():
    rows, members = weighted_fixture()
    saved_rows, saved_members = deepcopy(rows), deepcopy(members)
    state = random.getstate()
    first = paired_component_report(rows, ["a", "b"], members, seed=17, draws=20)
    second = paired_component_report(list(reversed(rows)), ["a", "b"], list(reversed(members)), seed=17, draws=20)
    assert first == second
    assert rows == saved_rows and members == saved_members
    assert random.getstate() == state
    assert paired_component_report(rows, ["a", "b"], members, seed=18, draws=20)["report_fingerprint"] != first["report_fingerprint"]
    rows[0]["target_provenance"] = "another_manual_source"
    third = paired_component_report(rows, ["a", "b"], members, draws=20)
    assert third["observations_sha256"] != first["observations_sha256"]
    assert third["point"] == first["point"]


def test_unknown_confidence_retained_and_distinct_source_events_on_one_candidate_allowed():
    rows = [observation("a", "one", target_confidence=None), observation("a", "two", target_confidence=None)]
    report = paired_component_report(rows, ["a", "b"], [member("a")], draws=1)
    assert report["support"]["eligible_observations"] == 2
    assert report["support"]["unknown_target_confidence"] == 2


@pytest.mark.parametrize("change", [
    {"gold": None}, {"gold": True}, {"gold": -1}, {"gold": 2}, {"gold": float("nan")},
    {"local": None}, {"local": 1.}, {"reference": float("inf")}, {"reference": False},
    {"target_provenance": ""}, {"target_provenance": " "}, {"target_provenance": None},
    {"candidate_id": ""}, {"observation_id": ""}, {"component_id": "wrong"}, {"piece_id": "unknown"},
    {"split": "test"},
    {"target_confidence": float("nan")}, {"target_confidence": float("inf")}, {"target_confidence": -.1},
    {"target_confidence": 1.1}, {"target_confidence": True}, {"target_confidence": "unknown"},
    {"target_confidence": 10**400},
    {"available": False}, {"available": 1}, {"masked": True}, {"masked": 0}, {"reason": "missing"},
])
def test_invalid_or_masked_observations_fail_closed(change):
    with pytest.raises(ValueError):
        paired_component_report([observation("a", **change)], ["a", "b"], [member("a")])


def test_duplicate_observation_rejected_even_if_candidate_or_prediction_differs():
    with pytest.raises(ValueError, match="duplicate_observation"):
        paired_component_report([observation("a"), observation("a", candidate_id="other", local=1)],
                                ["a", "b"], [member("a")])


@pytest.mark.parametrize("members", [[], [replace(member("a"), split="train")],
                                    [replace(member("a"), split="test")], [member("a"), member("a")],
                                    [member(" ")], [member("a", " ")]])
def test_invalid_validation_membership_rejected(members):
    with pytest.raises(ValueError):
        paired_component_report([], ["a", "b"], members)


@pytest.mark.parametrize("classes", [[], ["a"], ["a", "a"], ["a", ""], ["a", None], "ab", ["a", []]])
def test_invalid_class_order_rejected(classes):
    with pytest.raises(ValueError):
        paired_component_report([], classes, [member("a")])


@pytest.mark.parametrize("option", [{"draws": 0}, {"draws": 201}, {"draws": True}, {"seed": True},
                                  {"seed": -1}, {"seed": 2**32}, {"seed": math.nan}])
def test_bounded_options(option):
    with pytest.raises(ValueError):
        paired_component_report([], ["a", "b"], [member("a")], **option)


def test_input_budgets_before_iteration_and_class_matrix_expansion(monkeypatch):
    with pytest.raises(ValueError, match="membership_budget"):
        paired_component_report([], ["a", "b"], [member(str(i)) for i in range(33)])
    with pytest.raises(ValueError, match="invalid_classes"):
        paired_component_report([], [str(i) for i in range(257)], [member("a")])
    class HugeRows(list):
        def __len__(self):
            return 640_001
        def __iter__(self):
            pytest.fail("oversized rows must not be consumed")
    with pytest.raises(ValueError, match="observation_budget"):
        paired_component_report(HugeRows(), ["a", "b"], [member("a")])
    assert module.MAX_OBSERVATIONS_PER_PIECE == 20_000 and module.MAX_OBSERVATIONS == 640_000
    monkeypatch.setattr(module, "MAX_OBSERVATIONS_PER_PIECE", 2)
    with pytest.raises(ValueError, match="per_piece_observation_budget"):
        paired_component_report([observation("a", str(i)) for i in range(3)], ["a", "b"], [member("a")])


def test_existing_event_metric_function_only_runs_for_two_point_estimates(monkeypatch):
    real = module.classification_metrics
    calls = []
    def counted(rows, classes):
        calls.append(len(rows))
        return real(rows, classes)
    monkeypatch.setattr(module, "classification_metrics", counted)
    rows, members = weighted_fixture()
    paired_component_report(rows, ["a", "b"], members, draws=200)
    assert calls == [4, 4]  # Resamples use component sufficient statistics.
