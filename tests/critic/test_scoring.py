"""Exact synthetic score-reference examples; not musical-quality evidence."""

import pytest
from fractions import Fraction
import torch

from music_critic.critic.contracts import CriticOutput, SamplePrediction

from music_critic.critic.scoring import (
    AspectScorePolicy,
    ReferenceRow,
    TrainReferenceBank,
)


def test_reference_is_work_balanced_and_uses_midpoint_ties():
    # Work a contributes total mass 1/2 despite having two variants.
    bank = TrainReferenceBank(
        reference_id="synthetic-train-v1",
        rows=(
            ReferenceRow("a", {"harmony": 0.0}, "original"),
            ReferenceRow("a", {"harmony": 2.0}, "variant"),
            ReferenceRow("b", {"harmony": 2.0}, "original"),
        ),
        horizon="8-bars",
        support_policy="pitched-v1",
        type_weights={"original": 1.0, "variant": 1.0}, predictor_id="fixture-critic",
    )
    result = AspectScorePolicy().apply(
        {"harmony": 2.0}, bank, {"harmony": 1.0},
        horizon="8-bars", support_policy="pitched-v1", predictor_id="fixture-critic",
    )
    # Strictly lower: 1/4; ties: 3/4. Midpoint = 1/4 + 3/8.
    assert result.percentiles["harmony"] == pytest.approx(0.625)
    assert result.overall == pytest.approx(0.625)
    assert result.weighted_coverage == 1.0


def test_output_adapter_preserves_unavailable_aspects_and_exact_horizon():
    output = CriticOutput((SamplePrediction(
        "x", torch.tensor([2.0, float("nan"), 1.0, 4.0]),
        torch.tensor([True, False, True, False]), {}, Fraction(8, 3), "pitched-v1",
    ),))
    result = AspectScorePolicy().from_output(output, sample_index=0)
    assert result.raw_scores == {"harmony": 2.0, "rhythm": None,
                                 "melody": 1.0, "structure": None}
    assert result.horizon == "8/3"
    assert result.overall is None
    assert result.weighted_coverage == 0
    assert result.unavailable_reasons["harmony"] == "reference_unavailable"


def reference():
    return TrainReferenceBank("train-v1", (
        ReferenceRow("a", {"harmony": 0.0, "rhythm": 0.0}, "original"),
        ReferenceRow("b", {"harmony": 2.0, "rhythm": 2.0}, "original"),
    ), "8-bars", "pitched-v1", {"original": 1.0}, predictor_id="fixture-critic")


def test_pair_comparison_uses_explicit_common_support_not_unmatched_overalls():
    policy = AspectScorePolicy()
    bank = reference()
    weights = {"harmony": 1.0, "rhythm": 3.0}
    left = policy.apply({"harmony": 2.0, "rhythm": -1.0}, bank, weights,
                        horizon="8-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    right = policy.apply({"harmony": 1.0}, bank, weights,
                         horizon="8-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    assert left.overall < right.overall  # The unmatched support reverses order.
    compared = policy.compare(left, right)
    assert compared.common_aspects == ("harmony",)
    assert compared.left_overall == 0.75
    assert compared.right_overall == 0.5
    assert compared.weighted_coverage == 0.25
    assert right.weighted_coverage == 0.25


@pytest.mark.parametrize("weights", [{}, {"harmony": 0}, {"harmony": -1},
                                     {"harmony": float("nan")}, {"harmony": float("inf")}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match="weights_invalid"):
        AspectScorePolicy().apply({}, None, weights, horizon="8-bars", support_policy="v1")


def test_scope_reference_and_missing_support_fail_explicitly():
    policy = AspectScorePolicy()
    with pytest.raises(ValueError, match="reference_scope"):
        policy.apply({"harmony": 1.0}, reference(), horizon="4-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    with pytest.raises(ValueError, match="binding_invalid"):
        TrainReferenceBank("heldout", (), "8-bars", "v1", {"original": 1}, split="validation")
    absent = policy.apply({}, reference(), horizon="8-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    assert absent.overall is None and absent.weighted_coverage == 0
    assert policy.compare(absent, absent).unavailable_reason == "no_common_reference_support"
    unsupported = policy.apply({"melody": 1}, reference(), horizon="8-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    assert unsupported.unavailable_reasons["melody"] == "reference_aspect_unavailable"
    other = policy.apply({}, None, horizon="4-bars", support_policy="pitched-v1", predictor_id="fixture-critic")
    with pytest.raises(ValueError, match="comparison_binding"):
        policy.compare(absent, other)


def test_reference_type_weights_and_sealing():
    bank = TrainReferenceBank("weighted", (
        ReferenceRow("a", {"harmony": 0}, "original"),
        ReferenceRow("a", {"harmony": 2}, "variant"),
        ReferenceRow("b", {"harmony": 0}, "original"),
    ), "8-bars", "v1", {"original": 1, "variant": 3}, predictor_id="fixture-critic")
    assert bank.percentile("harmony", 1) == 0.625
    with pytest.raises(TypeError):
        bank.rows[0].scores["harmony"] = 9
    with pytest.raises(TypeError):
        bank.type_weights["original"] = 5


def test_reference_is_bound_to_the_selected_predictor():
    policy = AspectScorePolicy()
    with pytest.raises(ValueError, match="predictor"):
        policy.apply({"harmony": 1}, reference(), horizon="8-bars", support_policy="pitched-v1",
                     predictor_id="different-critic")


def test_large_finite_weights_do_not_overflow_normalization():
    result = AspectScorePolicy().apply(
        {"harmony": 2.0, "rhythm": 2.0}, reference(),
        {"harmony": 1e308, "rhythm": 1e308}, horizon="8-bars", support_policy="pitched-v1",
        predictor_id="fixture-critic",
    )
    assert result.overall == 0.75
    assert result.weighted_coverage == 1.0


def test_each_supported_work_retains_equal_mass_at_extreme_row_weights():
    bank = TrainReferenceBank("extreme", (
        ReferenceRow("a", {"harmony": 0}, "original", weight=1e-300),
        ReferenceRow("b", {"harmony": 2}, "original", weight=1e300),
    ), "8-bars", "v1", {"original": 1}, predictor_id="fixture-critic")
    assert bank.percentile("harmony", 1) == 0.5
