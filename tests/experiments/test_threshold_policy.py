"""Synthetic operating-point tests, without claims about real-data quality."""
from copy import deepcopy
from fractions import Fraction
from itertools import combinations_with_replacement, product
import math
import unittest

from music_critic.experiments.harmonic_event_reasoner.metrics import (
    choose_threshold,
    choose_threshold_exact,
    choose_threshold_fpr,
    score_metrics,
)


def row(scores, labels, mask=None, **metadata):
    return {
        "work": "synthetic", "h_id": "h", "scores": list(scores),
        "y": list(labels), "mask": list(mask) if mask is not None else [True] * len(scores),
        **metadata,
    }


class ExactThresholdTests(unittest.TestCase):
    def test_separates_known_scores_above_historical_grid(self):
        rows = [row([0.99, 0.96, 0.95], [1, 0, 0])]
        self.assertEqual(choose_threshold(rows), 0.5)
        self.assertEqual(choose_threshold_exact(rows), 0.99)
        historical = score_metrics(rows, choose_threshold(rows))["global"]
        exact = score_metrics(rows, choose_threshold_exact(rows))["global"]
        self.assertEqual(historical["macro_f1"], 0.25)
        self.assertEqual(historical["fpr"], 1)
        self.assertEqual(exact["macro_f1"], 1)
        self.assertEqual(exact["fpr"], 0)

    def test_fpr_constraint_trades_recall_for_negative_error_limit(self):
        rows = [row([0.98, 0.95, 0.94, 0.92, 0.91], [1, 0, 1, 0, 0])]
        self.assertEqual(choose_threshold_fpr(rows), 0.98)
        strict = score_metrics(rows, choose_threshold_fpr(rows))["global"]
        relaxed = score_metrics(rows, choose_threshold_fpr(rows, max_fpr=1 / 3))["global"]
        self.assertEqual((strict["recall"], strict["fpr"]), (0.5, 0))
        self.assertEqual((relaxed["recall"], relaxed["fpr"]), (1, 1 / 3))

    def test_bounds_include_endpoints_without_observed_scores(self):
        rows = [row([0.99, 0.96, 0.95], [1, 0, 0])]
        self.assertEqual(choose_threshold_exact(rows, min_threshold=0.1, max_threshold=0.9), 0.5)
        self.assertEqual(choose_threshold_exact(rows, min_threshold=0.97, max_threshold=0.98), 0.97)
        self.assertEqual(choose_threshold_exact(rows, min_threshold=0.98, max_threshold=0.98), 0.98)

    def test_grid_resolution_failure_can_be_isolated_with_historical_bounds(self):
        rows = [row([0.87, 0.85], [1, 0])]
        self.assertEqual(choose_threshold(rows), 0.5)
        threshold = choose_threshold_exact(rows, min_threshold=0.1, max_threshold=0.9)
        self.assertEqual(threshold, 0.87)
        self.assertEqual(score_metrics(rows, threshold)["global"]["macro_f1"], 1)

    def test_historical_grid_is_nested_to_preserve_plateau_tie_preference(self):
        rows = [row([0.9, 0.1], [1, 0])]
        self.assertEqual(choose_threshold(rows), 0.5)
        self.assertEqual(choose_threshold_exact(rows), 0.5)
        self.assertEqual(choose_threshold_exact(rows, min_threshold=0.1, max_threshold=0.9), 0.5)

    def test_tied_scores_move_together_and_input_order_does_not_matter(self):
        rows = [row([0.8, 0.8, 0.2], [1, 0, 0])]
        self.assertEqual(choose_threshold_exact(rows), 0.5)
        metrics = score_metrics(rows, 0.8)["global"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]), (1, 1, 1, 0))
        self.assertEqual(metrics["macro_f1"], 2 / 3)
        reversed_rows = [row([0.2, 0.8, 0.8], [0, 0, 1])]
        self.assertEqual(choose_threshold_exact(reversed_rows), 0.5)

    def test_constant_scores_and_objective_tie_have_declared_boundary_policy(self):
        rows = [row([0.5, 0.5], [1, 0])]
        # All-positive and all-negative each have fixed-class macro F1=1/3;
        # the empirical .5 boundary is nearest .5 among candidate thresholds.
        self.assertEqual(choose_threshold_exact(rows), 0.5)
        threshold = choose_threshold_fpr(rows, max_fpr=0)
        self.assertEqual(threshold, math.nextafter(0.5, 1.0))
        self.assertEqual(score_metrics(rows, threshold)["global"]["fpr"], 0)
        # Historical grid points are also eligible on an unchanged plateau.
        outside = [row([0.99, 0.97], [1, 0])]
        self.assertEqual(choose_threshold_exact(outside, min_threshold=0.25, max_threshold=0.75), 0.5)

    def test_single_class_support_and_all_negative_representability(self):
        positives = [row([0.1, 0.8], [1, 1])]
        negatives = [row([0.2, 0.8], [0, 0])]
        self.assertEqual(choose_threshold_exact(positives), 0.1)
        threshold = choose_threshold_exact(negatives)
        self.assertEqual(threshold, math.nextafter(0.8, 1.0))
        self.assertEqual(score_metrics(negatives, threshold)["global"]["macro_f1"], 0.5)
        adjacent = [row([math.nextafter(1.0, 0.0)], [0])]
        self.assertEqual(choose_threshold_exact(adjacent), 1.0)
        saturated = [row([1.0, 1.0], [0, 0])]
        threshold = choose_threshold_exact(saturated)
        self.assertTrue(0 <= threshold <= 1)
        self.assertEqual(score_metrics(saturated, threshold)["global"]["fp"], 2)
        self.assertEqual(score_metrics(saturated, threshold)["global"]["macro_f1"], 0)

    def test_masked_values_add_no_boundaries_or_support_and_inputs_are_unchanged(self):
        rows = [row([0.7, 0.6, 0.55], [1, 0, 0], [True, True, False], split="TEST")]
        changed = deepcopy(rows)
        changed[0]["scores"][-1], changed[0]["y"][-1] = 1.0, 1
        before = deepcopy(rows)
        self.assertEqual(choose_threshold_exact(rows), choose_threshold_exact(changed))
        self.assertEqual(choose_threshold_fpr(rows, 0), choose_threshold_fpr(changed, 0))
        self.assertEqual(rows, before)
        # Split admission belongs to the caller: arbitrary row metadata cannot
        # either authorize or prohibit access in this pure metric function.
        self.assertEqual(choose_threshold_exact(rows), 0.7)

    def test_fpr_infeasible_saturated_negative_is_explicit(self):
        rows = [row([0.9, 1.0], [1, 0])]
        with self.assertRaisesRegex(ValueError, "no attainable threshold.*score==1"):
            choose_threshold_fpr(rows)
        self.assertEqual(choose_threshold_fpr(rows, 1), 0.5)
        # A score==1 negative is feasible when its count fits the error limit.
        rows = [row([0.9, 1.0, 0.1], [1, 0, 0])]
        self.assertEqual(choose_threshold_fpr(rows, 0.5), 0.5)

    def test_unsupported_labels_invalid_bounds_and_limits_raise(self):
        for rows in ([], [row([], [])], [row([0.8], [1], [False])]):
            for chooser in (choose_threshold_exact, choose_threshold_fpr):
                with self.subTest(rows=rows, chooser=chooser), self.assertRaisesRegex(ValueError, "known labels"):
                    chooser(rows)
        for rows in ([row([0.1], [0])], [row([0.9], [1])],
                     [row([0.9, 0.1], [1, 0], [True, False])]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "both known"):
                choose_threshold_fpr(rows)
        rows = [row([0.9, 0.1], [1, 0])]
        for value in (-0.1, 1.1, math.nan, math.inf, True, "0.1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_fpr"):
                choose_threshold_fpr(rows, value)
            for name in ("min_threshold", "max_threshold"):
                with self.subTest(value=value, name=name), self.assertRaisesRegex(ValueError, name):
                    choose_threshold_exact(rows, **{name: value})
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            choose_threshold_exact(rows, min_threshold=0.9, max_threshold=0.1)
        for chooser in (choose_threshold_exact, choose_threshold_fpr):
            with self.subTest(chooser=chooser), self.assertRaisesRegex(ValueError, "duplicate"):
                chooser(rows + rows)

    def test_sweep_matches_exhaustive_attainable_boundaries(self):
        # Independent brute-force oracle: classify every candidate from scratch
        # and use rational arithmetic for the fixed-two-class F1 objective.
        for scores in combinations_with_replacement([0.0, 0.25, 0.5, 0.75, 1.0], 3):
            for labels in product([0, 1], repeat=3):
                rows = [row(scores, labels)]
                for lower, upper in ((0.0, 1.0), (0.1, 0.9), (0.3, 0.6)):
                    candidates = {lower, upper, *(s for s in scores if lower <= s <= upper)}
                    candidates.update(tick / 10 for tick in range(1, 10)
                                      if lower <= tick / 10 <= upper)
                    if max(scores) < 1:
                        boundary = math.nextafter(max(scores), 1.0)
                        if lower <= boundary <= upper:
                            candidates.add(boundary)
                    outcomes = {}
                    for threshold in candidates:
                        tp = sum(y == 1 and s >= threshold for s, y in zip(scores, labels))
                        fp = sum(y == 0 and s >= threshold for s, y in zip(scores, labels))
                        fn, tn = sum(labels) - tp, 3 - sum(labels) - fp
                        objective = (Fraction(2 * tp, 2 * tp + fp + fn or 1)
                                     + Fraction(2 * tn, 2 * tn + fp + fn or 1)) / 2
                        outcomes[threshold] = objective, tp, fp
                    expected = max(candidates, key=lambda t: (outcomes[t][0], -abs(t - 0.5), -t))
                    with self.subTest(scores=scores, labels=labels, bounds=(lower, upper)):
                        self.assertEqual(choose_threshold_exact(rows, min_threshold=lower, max_threshold=upper), expected)
                    if (lower, upper) != (0.0, 1.0) or sum(labels) in (0, 3):
                        continue
                    for limit in (0.0, 0.5, 1.0):
                        feasible = [t for t in candidates if outcomes[t][2] / (3 - sum(labels)) <= limit]
                        with self.subTest(scores=scores, labels=labels, limit=limit):
                            if feasible:
                                expected = max(feasible, key=lambda t: (outcomes[t][1], -abs(t - 0.5), -t))
                                self.assertEqual(choose_threshold_fpr(rows, limit), expected)
                            else:
                                with self.assertRaisesRegex(ValueError, "no attainable threshold"):
                                    choose_threshold_fpr(rows, limit)


if __name__ == "__main__":
    unittest.main()
