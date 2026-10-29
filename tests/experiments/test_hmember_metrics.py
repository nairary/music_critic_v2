"""Synthetic contract tests; these establish no musical quality claims."""
from copy import deepcopy
import math
import unittest

from music_critic.experiments.harmonic_event_reasoner.metrics import choose_threshold, score_metrics


def row(work="w", h_id="h", scores=(0.9, 0.1), y=(1, 0), mask=None):
    return {
        "work": work, "h_id": h_id, "scores": list(scores), "y": list(y),
        "mask": list(mask) if mask is not None else [True] * len(scores),
    }


class MembershipMetricsTests(unittest.TestCase):
    def test_primary_is_nested_work_average_not_flat_query_or_candidate_average(self):
        rows = [row("a", "h1"), row("a", "h2"), row("a", "h3"),
                row("b", "h1", scores=(0.1, 0.9)),
                row("c", "unsupported", scores=(0.8,), y=(1,))]
        result = score_metrics(rows)
        self.assertEqual(result["primary_auc"], 0.5)
        self.assertEqual(result["eligible_h"], 4)
        self.assertEqual(result["eligible_works"], 2)
        self.assertEqual(result["per_work"]["a"]["primary_auc"], 1)
        self.assertEqual(result["per_work"]["b"]["primary_auc"], 0)
        self.assertIsNone(result["per_work"]["c"]["primary_auc"])
        self.assertEqual(result["controls"]["source_only_primary_auc"], 0.5)

    def test_rank_auc_matches_pair_definition_with_partial_ties(self):
        scores, labels = [0.1, 0.3, 0.3, 0.9, 0.9], [0, 1, 0, 1, 0]
        expected = sum(
            1 if a > b else 0.5 if a == b else 0
            for a, ya in zip(scores, labels) if ya
            for b, yb in zip(scores, labels) if not yb
        ) / 6
        result = score_metrics([row(scores=scores, y=labels)])
        self.assertAlmostEqual(result["primary_auc"], expected)
        self.assertEqual(score_metrics([row(scores=(0.4, 0.4))])["primary_auc"], 0.5)

    def test_unknown_scores_and_labels_do_not_change_classification_or_auc(self):
        first = row(scores=(0.7, 0.3, 0.1), y=(1, 0, 0), mask=(True, True, False))
        second = deepcopy(first)
        second["scores"][2], second["y"][2] = 0.99, 1
        left, right = score_metrics([first]), score_metrics([second])
        self.assertEqual(left["global"], right["global"])
        self.assertEqual(left["primary_auc"], right["primary_auc"])
        self.assertEqual(left["retrieval"]["recall_at_1"], 1)
        self.assertEqual(right["retrieval"]["recall_at_1"], 0)
        self.assertEqual(left["coverage"], right["coverage"])
        self.assertEqual(right["coverage"]["unknown"], 1)

    def test_unknown_remains_in_top_five_and_recall_counts_all_known_positives(self):
        result = score_metrics([row(
            scores=(0.99, 0.98, 0.97, 0.96, 0.8, 0.7, 0.1),
            y=(0, 0, 0, 0, 1, 1, 0),
            mask=(False, False, False, False, True, True, True),
        )])
        self.assertEqual(result["retrieval"]["recall_at_5"], 0.5)
        self.assertEqual(result["retrieval"]["recall_at_1"], 0)
        self.assertEqual(result["primary_auc"], 1)

    def test_retrieval_uses_stable_candidate_order_for_ties(self):
        result = score_metrics([row(scores=(0.4, 0.4), y=(0, 1))])
        self.assertEqual(result["retrieval"]["recall_at_1"], 0)
        self.assertEqual(result["retrieval"]["recall_at_5"], 1)

    def test_retrieval_has_nested_work_and_query_macro_averages(self):
        rows = [row("a", "h1"), row("a", "h2"), row("a", "h3"),
                row("b", "h1", scores=(0.1, 0.9))]
        result = score_metrics(rows)["retrieval"]
        self.assertEqual(result["recall_at_1"], 0.5)
        self.assertEqual(result["query_macro_recall_at_1"], 0.75)
        self.assertEqual(result["eligible_h"], 4)
        self.assertEqual(result["eligible_works"], 2)

    def test_all_member_collapse_reports_false_positives(self):
        result = score_metrics([row(scores=(0.9, 0.9, 0.9), y=(1, 0, 0))])
        metrics = result["global"]
        self.assertEqual(result["primary_auc"], 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]), (1, 2, 0, 0))
        self.assertEqual(metrics["macro_f1"], 0.25)
        self.assertEqual(metrics["precision"], 1 / 3)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["fpr"], 1)
        self.assertIsNone(metrics["per_class"]["0"]["precision"])
        self.assertEqual(metrics["per_class"]["0"]["recall"], 0)

    def test_all_nonmember_collapse_and_brier(self):
        result = score_metrics([row(scores=(0.1, 0.1), y=(1, 0))])
        metrics = result["global"]
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["fpr"], 0)
        self.assertAlmostEqual(metrics["brier"], 0.41)
        self.assertAlmostEqual(metrics["bce"], (-math.log(0.1) - math.log(0.9)) / 2)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)

    def test_single_class_fixed_macro_f1_and_explicit_unsupported_ratios(self):
        result = score_metrics([row(scores=(0.8,), y=(1,))])
        self.assertIsNone(result["primary_auc"])
        self.assertEqual(result["global"]["macro_f1"], 0.5)
        self.assertIsNone(result["global"]["per_class"]["0"]["f1"])
        self.assertIsNone(result["global"]["fpr"])

    def test_empty_support_is_not_zero_quality(self):
        for rows in ([], [row(scores=(), y=())], [row(mask=(False, False))]):
            with self.subTest(rows=rows):
                result = score_metrics(rows)
                self.assertIsNone(result["primary_auc"])
                self.assertEqual(result["eligible_h"], 0)
                self.assertEqual(result["eligible_works"], 0)
                for metric in ("macro_f1", "precision", "recall", "fpr", "brier", "bce"):
                    self.assertIsNone(result["global"][metric])
                self.assertIsNone(result["retrieval"]["recall_at_1"])
                self.assertIsNone(result["controls"]["source_only_primary_auc"])

    def test_exact_probability_extremes_yield_finite_bce(self):
        result = score_metrics([row(scores=(0.0, 1.0))])
        self.assertTrue(math.isfinite(result["global"]["bce"]))
        self.assertEqual(result["global"]["brier"], 1)

    def test_threshold_is_inclusive(self):
        result = score_metrics([row(scores=(0.5, 0.4))], threshold=0.5)
        self.assertEqual(result["global"]["macro_f1"], 1)

    def test_duplicate_identity_rejected_even_when_scores_differ(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_metrics([row(), row(scores=(0.6, 0.4))])
        self.assertEqual(score_metrics([row("a"), row("b")])["coverage"]["total_h"], 2)

    def test_invalid_shapes_labels_masks_probabilities_and_ids_rejected(self):
        changes = [
            {"scores": [0.5]}, {"scores": [float("nan"), 0.5]},
            {"scores": [float("inf"), 0.5]}, {"scores": [-0.1, 0.5]},
            {"scores": [1.1, 0.5]}, {"scores": [True, 0.5]},
            {"scores": ["0.9", 0.5]}, {"y": [2, 0]}, {"y": [0.5, 0]},
            {"mask": [1, 0]}, {"work": ""}, {"h_id": None}, {"family": ""},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_metrics([{**row(), **change}])
        for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                score_metrics([row()], threshold=threshold)

    def test_family_metadata_preserved_and_conflicts_rejected(self):
        result = score_metrics([{**row(), "family": "f1"}])
        self.assertEqual(result["per_work"]["w"]["family"], "f1")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            score_metrics([{**row(h_id="a"), "family": "f1"}, {**row(h_id="b"), "family": "f2"}])

    def test_threshold_grid_and_nearest_half_tie(self):
        self.assertEqual(choose_threshold([row(scores=(0.9, 0.1))]), 0.5)
        self.assertEqual(choose_threshold([row(scores=(0.4, 0.2))]), 0.4)
        self.assertEqual(choose_threshold([row(scores=(0.9, 0.7))]), 0.8)
        self.assertEqual(choose_threshold([row(scores=(0.5, 0.5))]), 0.5)

    def test_threshold_unknown_mask_invariance(self):
        base = row(scores=(0.4, 0.2, 0.1), y=(1, 0, 0), mask=(True, True, False))
        modified = deepcopy(base)
        modified["scores"][2], modified["y"][2] = 0.99, 1
        self.assertEqual(choose_threshold([base]), choose_threshold([modified]))

    def test_threshold_selection_rejects_missing_known_support_and_duplicates(self):
        for rows in ([], [row(mask=(False, False))], [row(), row()]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                choose_threshold(rows)


if __name__ == "__main__":
    unittest.main()
