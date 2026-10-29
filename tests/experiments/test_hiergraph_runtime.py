"""Native synthetic optimizer/replay contracts; no real samples or training."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch import nn

from music_critic.experiments.hierarchical_graph_analyzer.runtime import (
    evaluate, filter_core_predictions, initialize_shared_parameters, load_checkpoint,
    sample_schedule, save_checkpoint, tensorize_targets, train_steps,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 5)
        self.drop = nn.Dropout(.3)

    def forward(self, raw):
        values = self.linear(self.drop(raw['x']))
        start = values[:, 0].sigmoid() * .4
        end = .5 + values[:, 1].sigmoid() * .5
        return {'spans': torch.stack((start, end), -1), 'object_logits': values[:, 2],
                'field_logits': {'H.function': values[:, 3:]}}


def sample(work='w', view='v', split='fit'):
    return ({'x': torch.tensor([[1., .2]])},
            {'ids': ['h'], 'spans': [[.1, .8]],
             'fields': {'H.function': {'values': [1], 'mask': [True]}},
             'work_id': work, 'family_id': work, 'view_id': view, 'scope_id': 'harmonies',
             'core_span': (Fraction(0), Fraction(1)),
             'halo_span': (Fraction(0), Fraction(1)), 'split': split})


class RuntimeTests(unittest.TestCase):
    def test_tensorizer_preserves_exact_metadata_and_masks(self):
        target = sample()[1]
        converted = tensorize_targets(target, device='cpu')
        self.assertEqual(converted['spans'].shape, (1, 2))
        self.assertEqual(converted['fields']['H.function']['mask'].dtype, torch.bool)
        self.assertEqual(converted['core_span'], (Fraction(0), Fraction(1)))
        self.assertIsInstance(target['spans'], list)

    def test_shared_initialization_is_independent_of_module_registration_order(self):
        class Common(nn.Module):
            def __init__(self, reverse):
                super().__init__()
                if reverse:
                    self.extra = nn.Linear(3, 3)
                self.shared = nn.Linear(2, 3)
                if not reverse:
                    self.extra = nn.Linear(3, 3)
        first, second = Common(False), Common(True)
        before = torch.get_rng_state().clone()
        initialize_shared_parameters(first, 42)
        initialize_shared_parameters(second, 42)
        self.assertTrue(torch.equal(first.shared.weight, second.shared.weight))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_schedule_is_order_invariant_and_uniform_at_work_level(self):
        samples = [sample('a'), sample('b')]
        expanded = [sample('a', f'v{i}') for i in range(5)] + [sample('b')]
        base_works = [samples[i][1]['work_id'] for i in sample_schedule(samples, count=40)]
        expanded_works = [expanded[i][1]['work_id'] for i in sample_schedule(expanded, count=40)]
        self.assertEqual(base_works, expanded_works)
        reversed_samples = list(reversed(expanded))
        first = [(expanded[i][1]['work_id'], expanded[i][1]['view_id'])
                 for i in sample_schedule(expanded, count=40)]
        second = [(reversed_samples[i][1]['work_id'], reversed_samples[i][1]['view_id'])
                  for i in sample_schedule(reversed_samples, count=40)]
        self.assertEqual(first, second)

    def test_full_adam_and_rng_resume_matches_uninterrupted_updates(self):
        samples = [sample('a'), sample('b')]
        original = initialize_shared_parameters(TinyModel(), 17)
        interrupted = deepcopy(original)
        optimizer = torch.optim.Adam(original.parameters(), lr=.02)
        other_optimizer = torch.optim.Adam(interrupted.parameters(), lr=.02)
        random.seed(42); torch.manual_seed(42)
        full = train_steps(original, samples, optimizer, steps=4)
        expected_random, expected_torch = random.random(), torch.rand(3)
        random.seed(42); torch.manual_seed(42)
        first = train_steps(interrupted, samples, other_optimizer, steps=2)
        bindings = {'source_sha256': 'a'*64, 'config_sha256': 'b'*64, 'data_sha256': 'c'*64}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'step2.pt'
            receipt = save_checkpoint(path, interrupted, other_optimizer, first['state'], bindings=bindings)
            restored = TinyModel()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.02)
            state = load_checkpoint(path, restored, restored_optimizer, bindings=bindings,
                                    expected_sha256=receipt['sha256'])
            second = train_steps(restored, samples, restored_optimizer, steps=2, state=state)
        self.assertEqual(second['state'], full['state'])
        self.assertEqual(first['history'] + second['history'], full['history'])
        for name, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
        for key, row in optimizer.state_dict()['state'].items():
            for name, value in row.items():
                other = restored_optimizer.state_dict()['state'][key][name]
                self.assertTrue(torch.equal(value, other), name)
        self.assertEqual(random.random(), expected_random)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_checkpoint_binding_mismatch_does_not_mutate_model_or_rng(self):
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters())
        state = train_steps(model, [sample()], optimizer, steps=1)['state']
        bindings = {'source_sha256': 'a'*64, 'config_sha256': 'b'*64, 'data_sha256': 'c'*64}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'step1.pt'
            save_checkpoint(path, model, optimizer, state, bindings=bindings)
            before, rng = deepcopy(model.state_dict()), torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, 'bindings mismatch'):
                load_checkpoint(path, model, optimizer, bindings=dict(bindings, data_sha256='d'*64))
            for name, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, state, bindings=bindings)

    def test_all_unknown_draw_skips_adam_momentum_and_weight_decay(self):
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters(), weight_decay=.3)
        raw, target = sample()
        target = dict(target, ids=[], spans=[], fields={'H.function': {'values': [], 'mask': []}})
        before = deepcopy(model.state_dict())
        result = train_steps(model, [(raw, target)], optimizer, steps=2)
        self.assertEqual(result['state']['applied_updates'], 0)
        self.assertEqual(result['state']['skipped_unknown'], 2)
        self.assertEqual(optimizer.state_dict()['state'], {})
        for name, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_training_selection_and_default_dev_evaluation_are_rejected(self):
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(ValueError, 'fit'):
            train_steps(model, [sample(split='selection')], optimizer, steps=1)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            evaluate(model, [sample(split='dev')], split='dev')
        result = evaluate(model, [sample(split='dev')], split='dev', allow_dev=True)
        self.assertEqual(result['split'], 'dev')

    def test_core_filter_uses_start_ownership_and_preserves_gradients(self):
        spans = torch.tensor([[0., .7], [.25, .75], [.75, 1.]], requires_grad=True)
        pred = {'spans': spans, 'object_logits': torch.zeros(3), 'field_logits': {}}
        target = dict(sample()[1], core_span=(Fraction(1), Fraction(3)),
                      halo_span=(Fraction(0), Fraction(4)))
        selected = filter_core_predictions(pred, target)
        self.assertEqual(selected['spans'].shape[0], 1)
        selected['spans'].sum().backward()
        self.assertEqual(spans.grad.tolist(), [[0., 0.], [1., 1.], [0., 0.]])

    def test_evaluation_preserves_mixed_module_modes_even_after_failure(self):
        model = TinyModel().train()
        model.drop.eval()
        flags = [module.training for module in model.modules()]
        evaluate(model, [sample(split='selection')])
        self.assertEqual(flags, [module.training for module in model.modules()])
        with self.assertRaisesRegex(ValueError, 'split'):
            evaluate(model, [sample(split='fit')])
        self.assertEqual(flags, [module.training for module in model.modules()])

    def test_resume_seed_or_inventory_change_rejected(self):
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters())
        state = train_steps(model, [sample()], optimizer, steps=1)['state']
        with self.assertRaisesRegex(ValueError, 'schedule'):
            train_steps(model, [sample()], optimizer, steps=1, seed=42, state=state)
        with self.assertRaisesRegex(ValueError, 'schedule'):
            train_steps(model, [sample('changed')], optimizer, steps=1, state=state)

    def test_reader_identity_drift_is_rejected_before_optimizer_mutation(self):
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters())
        before = deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, 'identity'):
            train_steps(model, [sample('actual')], optimizer, steps=1,
                        sample_metadata=[sample('declared')[1]])
        self.assertEqual(optimizer.state_dict()['state'], {})
        for name, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_new_run_seed_controls_stochastic_model_rng(self):
        first = initialize_shared_parameters(TinyModel(), 17)
        second = deepcopy(first)
        torch.manual_seed(314)
        left = train_steps(first, [sample()], torch.optim.Adam(first.parameters()), steps=2, seed=73)
        torch.manual_seed(159)
        right = train_steps(second, [sample()], torch.optim.Adam(second.parameters()), steps=2, seed=73)
        self.assertEqual(left['history'], right['history'])

    def test_same_shape_different_e2_capacity_checkpoint_is_rejected(self):
        from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer
        first = HAnalyzer('E2', {}, input_width=2, hidden=4, rounds=1, max_events=3)
        second = HAnalyzer('E2', {}, input_width=2, hidden=4, rounds=1, max_events=1)
        bindings = {'source_sha256': 'a'*64, 'config_sha256': 'b'*64, 'data_sha256': 'c'*64}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.pt'
            save_checkpoint(path, first, torch.optim.Adam(first.parameters()), {}, bindings=bindings)
            before = deepcopy(second.state_dict())
            with self.assertRaisesRegex(ValueError, 'model contract'):
                load_checkpoint(path, second, torch.optim.Adam(second.parameters()), bindings=bindings)
            for name, value in before.items():
                self.assertTrue(torch.equal(value, second.state_dict()[name]))

    def test_per_step_callback_retains_partial_updates_before_reader_failure(self):
        class DriftingSamples:
            metadata = [sample()[1]]
            calls = 0
            def __len__(self): return 1
            def __getitem__(self, index):
                self.calls += 1
                if self.calls > 1: raise ValueError('synthetic reader failure')
                return sample()
        model = TinyModel(); optimizer = torch.optim.Adam(model.parameters())
        observed = []
        with self.assertRaisesRegex(ValueError, 'reader failure'):
            train_steps(model, DriftingSamples(), optimizer, steps=3,
                        on_step=lambda state, row: observed.append((state, row)))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0]['applied_updates'], 1)
        self.assertEqual(observed[0][0]['attempted_steps'], 1)

    def test_evaluation_reports_shared_raw_bank_reference_ceiling(self):
        raw, target = sample(split='selection')
        raw = dict(raw, time_positions=torch.tensor([0., .1, .8, 1.]),
                   bank_indices=torch.tensor([[0, 1], [1, 2], [2, 3]]))
        result = evaluate(TinyModel(), [(raw, target)])
        report = result['reports'][0]
        self.assertEqual(report['raw_candidate_ceiling'], 1)
        self.assertEqual(report['raw_candidate_reference'], 'shared_raw_span_bank')


if __name__ == '__main__':
    unittest.main()
