"""Synthetic contracts for shared H assignment; no corpus is read."""
from fractions import Fraction
import unittest

from music_critic.experiments.hierarchical_graph_analyzer.matching import (
    canonical_time, interval_iou, match_events, known_negative_indices,
)


class MatchingTests(unittest.TestCase):
    def test_rational_timing_and_half_open_overlap(self):
        self.assertEqual(canonical_time({'num': 1, 'den': 3}), Fraction(1, 3))
        self.assertEqual(interval_iou((0, 1), (1, 2)), 0)
        with self.assertRaises(ValueError):
            canonical_time(0.1)

    def test_training_assignment_does_not_use_metric_overlap_gate(self):
        pred = {'spans': [[8, 9]]}
        targets = {'ids': ['h'], 'spans': [[0, 1]]}
        self.assertEqual(match_events(pred, targets, mode='training'), [(0, 0)])
        self.assertEqual(match_events(pred, targets, mode='metric'), [])

    def test_metric_assignment_maximizes_cardinality_before_cost(self):
        # Broad P overlaps both H; short P can match only the first H. A
        # greedy broad->first choice would incorrectly recover only one H.
        pred = {'spans': [[0, 3], [0, 1]]}
        targets = {'ids': ['a', 'b'], 'spans': [[0, 2], [2, 3]]}
        self.assertEqual(match_events(pred, targets, mode='metric', iou_threshold=0.3),
                         [(0, 1), (1, 0)])

    def test_permutation_keeps_unique_geometric_assignment(self):
        targets = {'ids': ['b', 'a'], 'spans': [[2, 3], [0, 1]]}
        pred = {'spans': [[0, 1], [2, 3], [7, 8]]}
        self.assertEqual(match_events(pred, targets), [(0, 1), (1, 0)])
        permuted = {'spans': [pred['spans'][i] for i in [2, 1, 0]]}
        self.assertEqual(match_events(permuted, targets), [(1, 0), (2, 1)])

    def test_duplicate_candidates_and_missing_event_masks(self):
        pred = {'spans': [[0, 1], [0, 1]]}
        targets = {'ids': ['a', 'b'], 'spans': [[0, 1], [0, 1]],
                   'event_mask': [True, False]}
        self.assertEqual(len(match_events(pred, targets)), 1)

    def test_view_and_scope_mismatch_is_never_matched(self):
        pred = {'spans': [[0, 1]], 'view_id': 'a', 'scope_id': 'whole'}
        for key in ('view_id', 'scope_id'):
            targets = dict(pred, ids=['h'])
            targets[key] = 'different'
            self.assertEqual(match_events(pred, targets), [])

    def test_negatives_require_complete_containment_and_no_unknown_overlap(self):
        pred = {'spans': [[0, 1], [1, 2], [2, 4], [4, 5]]}
        targets = {'ids': ['unknown'], 'spans': [[1, 2]], 'event_mask': [False],
                   'view_id': 'a', 'scope_id': 'whole',
                   'known_negative_spans': [{'span': [0, 3], 'complete': True},
                                            {'span': [4, 5], 'complete': False}]}
        self.assertEqual(known_negative_indices(pred, targets), [0])
        targets['ambiguous_spans'] = [[0, 0.5]]
        self.assertEqual(known_negative_indices(pred, targets), [])
        targets['known_negative_spans'] = [[0, 5]]
        self.assertEqual(known_negative_indices(pred, targets), [])

    def test_negative_scope_cannot_relabel_selected_view(self):
        pred = {'spans': [[0, 1]]}
        targets = {'ids': [], 'spans': [], 'view_id': 'a', 'scope_id': 'whole',
                   'known_negative_spans': [{'span': [0, 1], 'complete': True,
                                            'view_id': 'other', 'scope_id': 'whole'}]}
        self.assertEqual(known_negative_indices(pred, targets), [])

    def test_empty_and_point_events(self):
        self.assertEqual(match_events({'spans': []}, {'ids': ['h'], 'spans': [[0, 1]]}), [])
        with self.assertRaisesRegex(ValueError, 'positive duration'):
            match_events({'spans': [[0, 0]]}, {'ids': [], 'spans': []})


if __name__ == '__main__':
    unittest.main()
