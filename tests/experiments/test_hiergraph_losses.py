"""Synthetic gradient tests for partial H/F labels."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from music_critic.experiments.hierarchical_graph_analyzer.losses import compute_loss


@unittest.skipIf(torch is None, 'native PyTorch is required for gradient tests')
class LossTests(unittest.TestCase):
    def prediction(self):
        return {'spans': torch.tensor([[0.0, .3], [.6, .8]], requires_grad=True),
                'object_logits': torch.zeros(2, requires_grad=True),
                'field_logits': {'H.function': torch.zeros(2, 3, requires_grad=True)}}

    def test_all_unknown_is_differentiable_zero(self):
        pred = self.prediction()
        targets = {'ids': [], 'spans': [], 'fields': {}}
        result = compute_loss(pred, targets)
        self.assertEqual(result['loss'].item(), 0)
        result['loss'].backward()
        for tensor in [pred['spans'], pred['object_logits'], *pred['field_logits'].values()]:
            self.assertIsNotNone(tensor.grad)
            self.assertEqual(tensor.grad.abs().sum().item(), 0)

    def test_masked_fields_and_unmatched_candidates_have_zero_gradient(self):
        pred = self.prediction()
        targets = {'ids': ['h'], 'spans': [[0.0, .4]],
                   'fields': {'H.function': {'values': [-999], 'mask': [False]}}}
        result = compute_loss(pred, targets)
        result['loss'].backward()
        self.assertGreater(pred['spans'].grad[0].abs().sum().item(), 0)
        self.assertLess(pred['object_logits'].grad[0].item(), 0)
        self.assertEqual(pred['object_logits'].grad[1].item(), 0)
        self.assertEqual(pred['spans'].grad[1].abs().sum().item(), 0)
        self.assertEqual(pred['field_logits']['H.function'].grad.abs().sum().item(), 0)
        self.assertEqual(result['support']['fields']['H.function'], 0)

    def test_known_field_ce_is_independent_of_other_field_mask(self):
        pred = self.prediction()
        pred['field_logits']['H.root'] = torch.zeros(2, 4, requires_grad=True)
        targets = {'ids': ['h'], 'spans': [[0.0, .4]], 'fields': {
            'H.function': {'values': [2], 'mask': [True]},
            'H.root': {'values': [-1], 'mask': [False]}}}
        result = compute_loss(pred, targets)
        result['loss'].backward()
        function = pred['field_logits']['H.function'].grad
        self.assertLess(function[0, 2].item(), 0)
        self.assertGreater(function[0, 0].item(), 0)
        self.assertEqual(function[1].abs().sum().item(), 0)
        self.assertEqual(pred['field_logits']['H.root'].grad.abs().sum().item(), 0)

    def test_explicit_complete_negative_has_negative_object_gradient(self):
        pred = self.prediction()
        targets = {'ids': ['h'], 'spans': [[0.0, .4]], 'fields': {},
                   'known_negative_spans': [{'span': [.5, 1.0], 'complete': True}]}
        result = compute_loss(pred, targets)
        result['loss'].backward()
        self.assertEqual(result['support']['object_negative'], 1)
        self.assertGreater(pred['object_logits'].grad[1].item(), 0)

    def test_disjoint_initial_prediction_receives_boundary_gradient(self):
        pred = {'spans': torch.tensor([[.8, .9]], requires_grad=True),
                'object_logits': torch.zeros(1, requires_grad=True), 'field_logits': {}}
        result = compute_loss(pred, {'ids': ['h'], 'spans': [[0, .1]], 'fields': {}})
        result['loss'].backward()
        self.assertGreater(pred['spans'].grad.abs().sum().item(), 0)

    def test_unmatched_gold_does_not_leak_field_label_to_other_predictions(self):
        pred = {'spans': torch.tensor([[0., .1]], requires_grad=True),
                'object_logits': torch.zeros(1, requires_grad=True),
                'field_logits': {'H.function': torch.zeros(1, 3, requires_grad=True)}}
        targets = {'ids': ['a', 'b'], 'spans': [[0., .1], [.8, 1.]],
                   'fields': {'H.function': {'values': [-999, 1], 'mask': [False, True]}}}
        result = compute_loss(pred, targets)
        result['loss'].backward()
        self.assertEqual(result['support']['known_events'], 2)
        self.assertEqual(result['support']['matched_events'], 1)
        self.assertEqual(pred['field_logits']['H.function'].grad.abs().sum().item(), 0)

    def test_permutation_preserves_loss(self):
        pred = self.prediction()
        targets = {'ids': ['h', 'h2'], 'spans': [[0., .4], [.7, .9]],
                   'fields': {'H.function': {'values': [2, 1], 'mask': [True, True]}}}
        original = compute_loss(pred, targets)['loss'].item()
        permuted = {'spans': pred['spans'].flip(0), 'object_logits': pred['object_logits'].flip(0),
                    'field_logits': {k: v.flip(0) for k, v in pred['field_logits'].items()}}
        self.assertAlmostEqual(compute_loss(permuted, targets)['loss'].item(), original)

    def test_known_oov_is_rejected_and_masked_oov_ignored(self):
        targets = {'ids': ['h'], 'spans': [[0., .4]],
                   'fields': {'H.function': {'values': [-1], 'mask': [True]}}}
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            compute_loss(self.prediction(), targets)
        targets['fields']['H.function']['mask'] = [False]
        self.assertTrue(torch.isfinite(compute_loss(self.prediction(), targets)['loss']))

    def test_training_match_cannot_also_receive_negative_objectness(self):
        pred = {'spans': torch.tensor([[.7, .8]], requires_grad=True),
                'object_logits': torch.zeros(1, requires_grad=True), 'field_logits': {}}
        target = {'ids': ['h'], 'spans': [[.1, .2]], 'fields': {},
                  'known_negative_spans': [{'span': [.6, .9], 'complete': True}]}
        result = compute_loss(pred, target)
        self.assertEqual(result['support']['object_negative'], 0)
        result['loss'].backward()
        self.assertLess(pred['object_logits'].grad[0].item(), 0)

    def test_empty_loader_fields_are_differentiable_zero(self):
        pred = self.prediction()
        result = compute_loss(pred, {'ids': [], 'spans': [],
                                    'fields': {'H.function': {'values': [], 'mask': []}}})
        self.assertEqual(result['loss'].item(), 0)
        result['loss'].backward()

    def test_exact_known_endpoint_learns_while_unknown_grid_is_masked(self):
        pred = self.prediction()
        pred['boundary_logits'] = torch.tensor([-8., -8., -8.], requires_grad=True)
        target = {'ids': ['h'], 'spans': [[0., .4]], 'fields': {},
                  'known_boundary_indices': [0, 2, 2],
                  'boundary_endpoint_coverage': {'knownunique': 2, 'offgrid': 0}}
        result = compute_loss(pred, target)
        result['loss'].backward()
        self.assertEqual(result['support']['boundary_positive'], 1)
        grad = pred['boundary_logits'].grad
        self.assertLess(grad[1].item(), 0)
        self.assertEqual(grad[[0, 2]].abs().sum().item(), 0)

    def test_boundary_labels_never_fall_back_to_float_geometry(self):
        pred = self.prediction()
        pred['boundary_logits'] = torch.tensor([-8.], requires_grad=True)
        pred['boundary_positions'] = torch.tensor([.4])
        result = compute_loss(pred, {'ids': ['h'], 'spans': [[0., .4]], 'fields': {}})
        result['loss'].backward()
        self.assertTrue(result['support']['boundary_targets_missing'])
        self.assertEqual(pred['boundary_logits'].grad.item(), 0)

    def test_e2_no_cut_receives_known_boundary_confidence_gradient(self):
        from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer
        raw = {'x': torch.ones(3, 2), 'node_type': torch.tensor([6, 6, 6]),
               'node_time': torch.tensor([[0., 0.], [.5, .5], [1., 1.]]),
               'time_index': torch.tensor([0, 1, 2]), 'time_positions': torch.tensor([0., .5, 1.]),
               'bank_indices': torch.tensor([[0, 1], [1, 2]]),
               'edge_index': torch.empty(2, 0, dtype=torch.long),
               'edge_type': torch.empty(0, dtype=torch.long), 'pool_mask': torch.ones(3, dtype=torch.bool)}
        model = HAnalyzer('E2', {}, input_width=2, hidden=4, rounds=1, max_events=3)
        with torch.no_grad():
            model.boundary_head[-1].weight.zero_(); model.boundary_head[-1].bias.fill_(-8.)
        pred = model(raw); pred['boundary_logits'].retain_grad()
        self.assertEqual(pred['spans'].tolist(), [[0., 1.]])
        result = compute_loss(pred, {'ids': ['a', 'b'], 'spans': [[0., .5], [.5, 1.]],
                                    'fields': {}, 'known_boundary_indices': [0, 1, 2]},
                              weights={'object_positive': 0})
        result['loss'].backward()
        self.assertLess(pred['boundary_logits'].grad.item(), 0)


if __name__ == '__main__':
    unittest.main()
