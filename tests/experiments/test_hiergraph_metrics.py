"""Synthetic evaluation fixtures; recovery is distinct from object precision."""
import unittest

from music_critic.experiments.hierarchical_graph_analyzer.metrics import (
    aggregate_metrics, evaluate_piece,
)


class MetricTests(unittest.TestCase):
    def test_missed_known_event_stays_in_recovery_and_joint_denominators(self):
        pred = {'spans': [[0., .25]], 'object_logits': [3.],
                'field_logits': {'H.function': [[1., 0.]]}}
        targets = {'ids': ['a', 'b'], 'spans': [[0., .25], [.5, .75]],
                   'fields': {'H.function': {'values': [0, 1], 'mask': [True, True]}},
                   'family_id': 'f'}
        result = evaluate_piece(pred, targets, duration_quarters=4)
        self.assertEqual(result['known_h_recovery'], .5)
        self.assertEqual(result['joint_h_f_recovery'], .5)
        self.assertEqual(result['conditional_f_macro_f1'], 1.)
        self.assertIsNone(result['precision'])

    def test_budget_depends_on_raw_duration_and_scores_not_gold_count(self):
        pred = {'spans': [[0, .2], [.3, .5], [.6, .8]], 'object_logits': [1., 5., 2.],
                'field_logits': {}}
        targets = {'ids': ['a', 'b', 'c'], 'spans': pred['spans'], 'fields': {}}
        result = evaluate_piece(pred, targets, duration_quarters=.25)
        self.assertEqual(result['budget'], 1)
        self.assertEqual(result['selected_indices'], [1])
        self.assertAlmostEqual(result['known_h_recovery'], 1 / 3)
        targets = {'ids': [], 'spans': [], 'fields': {}}
        self.assertEqual(evaluate_piece(pred, targets, duration_quarters=.25)['budget'], 1)

    def test_duplicates_do_not_inflate_recovery_and_raw_ceiling_is_separate(self):
        pred = {'spans': [[0., .3], [0., .3]], 'object_logits': [3., 2.], 'field_logits': {}}
        targets = {'ids': ['a', 'b'], 'spans': [[0., .3], [.5, .8]], 'fields': {}}
        result = evaluate_piece(pred, targets, duration_quarters=4,
                                raw_candidates={'spans': targets['spans']})
        self.assertEqual(result['known_h_recovery'], .5)
        self.assertEqual(result['raw_candidate_ceiling'], 1)
        self.assertEqual(result['duplicate_known_h_candidates'], 1)
        self.assertEqual(result['duplicate_rate'], .5)
        self.assertIsNone(result['precision'])

    def test_precision_only_inside_proven_complete_scopes(self):
        pred = {'spans': [[0., .2], [.3, .4], [.7, .8]], 'object_logits': [3., 2., 1.],
                'field_logits': {}}
        targets = {'ids': ['a'], 'spans': [[0., .2]], 'fields': {},
                   'complete_scopes': [{'span': [0., .5], 'complete': True}]}
        result = evaluate_piece(pred, targets, duration_quarters=4)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall_complete'], 1)
        self.assertAlmostEqual(result['f1'], 2 / 3)
        self.assertEqual(result['complete_predictions'], 2)
        targets['ambiguous_spans'] = [[.3, .4]]
        self.assertIsNone(evaluate_piece(pred, targets, duration_quarters=4)['precision'])

    def test_wrong_scope_cannot_license_precision(self):
        pred = {'spans': [[0., .2]], 'object_logits': [3.], 'field_logits': {}}
        targets = {'ids': ['a'], 'spans': [[0., .2]], 'fields': {}, 'view_id': 'a',
                   'complete_scopes': [{'span': [0., 1.], 'complete': True, 'view_id': 'b'}]}
        self.assertIsNone(evaluate_piece(pred, targets, duration_quarters=4)['precision'])

    def test_unknown_event_invalidates_complete_scope(self):
        pred = {'spans': [[0., .2]], 'object_logits': [3.], 'field_logits': {}}
        targets = {'ids': ['a'], 'spans': [[0., .2]], 'fields': {}, 'event_mask': [False],
                   'complete_scopes': [{'span': [0., 1.], 'complete': True}]}
        result = evaluate_piece(pred, targets, duration_quarters=4)
        self.assertIsNone(result['known_h_recovery'])
        self.assertIsNone(result['precision'])
        self.assertEqual(result['excluded_h'], 1)

    def test_boundary_error_uses_raw_duration(self):
        pred = {'spans': [[.1, .5]], 'object_logits': [3.], 'field_logits': {}}
        targets = {'ids': ['a'], 'spans': [[0., .4]], 'fields': {}}
        result = evaluate_piece(pred, targets, duration_quarters=8)
        self.assertAlmostEqual(result['boundary_error_normalized'], .1)
        self.assertAlmostEqual(result['boundary_error_quarters'], .8)

    def test_outside_halo_owned_events_remain_primary_misses(self):
        pred = {'spans': [[0., .25]], 'object_logits': [3.],
                'field_logits': {'H.function': [[1., 0.]]}}
        targets = {'ids': ['a'], 'spans': [[0., .25]],
                   'fields': {'H.function': {'values': [0], 'mask': [True]}},
                   'coverage': {'outside_halo': 2, 'outside_halo_known_fields': {'H.function': 1}}}
        result = evaluate_piece(pred, targets, duration_quarters=4, span_scale_quarters=8)
        self.assertAlmostEqual(result['known_h_recovery'], 1 / 3)
        self.assertEqual(result['admitted_known_h_recovery'], 1)
        self.assertEqual(result['joint_h_f_recovery'], .5)

    def test_missing_function_field_has_no_metric_denominator(self):
        pred = {'spans': [[0., 1.]], 'object_logits': [3.], 'field_logits': {'H.function': [[0., 1.]]}}
        targets = {'ids': ['a'], 'spans': [[0., 1.]],
                   'fields': {'H.function': {'values': [-1], 'mask': [False]}}}
        result = evaluate_piece(pred, targets, duration_quarters=4)
        self.assertIsNone(result['joint_h_f_recovery'])
        self.assertIsNone(result['conditional_f_macro_f1'])

    def test_family_macro_weights_independent_families_equally(self):
        pred = {'spans': [[0., .2]], 'object_logits': [3.], 'field_logits': {}}
        target = {'ids': ['a'], 'spans': [[0., .2]], 'fields': {}, 'family_id': 'large'}
        good = evaluate_piece(pred, target, duration_quarters=1)
        bad = evaluate_piece({'spans': [], 'object_logits': [], 'field_logits': {}},
                             dict(target, family_id='small'), duration_quarters=1)
        aggregate = aggregate_metrics([good, good, good, bad])
        self.assertEqual(aggregate['family_macro_known_h_recovery'], .5)
        self.assertEqual(aggregate['supported_families_known_h_recovery'], 2)

    def test_empty_predictions_and_zero_raw_duration(self):
        pred = {'spans': [], 'object_logits': [], 'field_logits': {}}
        targets = {'ids': [], 'spans': [], 'fields': {}, 'family_id': 'f'}
        result = evaluate_piece(pred, targets, duration_quarters=0)
        self.assertEqual(result['budget'], 0)
        self.assertIsNone(result['known_h_recovery'])
        self.assertIsNone(aggregate_metrics([result])['family_macro_known_h_recovery'])

    def test_tied_duplicate_prediction_permutation_keeps_semantic_metric(self):
        pred = {'spans': [[0., .2], [0., .2]], 'object_logits': [3., 3.],
                'field_logits': {'H.function': [[1., 0.], [0., 1.]]}}
        target = {'ids': ['a'], 'spans': [[0., .2]],
                  'fields': {'H.function': {'values': [0], 'mask': [True]}}}
        original = evaluate_piece(pred, target, duration_quarters=.25)
        reverse = dict(pred, field_logits={'H.function': pred['field_logits']['H.function'][::-1]})
        other = evaluate_piece(reverse, target, duration_quarters=.25)
        self.assertEqual(original['joint_h_f_recovery'], other['joint_h_f_recovery'])


if __name__ == '__main__':
    unittest.main()
