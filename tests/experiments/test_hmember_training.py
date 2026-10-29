"""Synthetic schedule/admission tests, plus native Torch smoke when installed."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from music_critic.experiments.harmonic_event_reasoner.training import (
    _Deadline, _manifest, _TileCache, _WorkSampler, train_run,
)

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def binding(tile_id, split, work=None, family=None):
    return dict(id=tile_id, split=split, work=work or split, family=family or split,
                path=tile_id + '.pt', sha256='a' * 64, size_bytes=1)


def cache_manifest(tiles, **overrides):
    return dict(schema='HMemberCache@1', status='completed',
                mode='oracle_H_geometry_frozen_raw_C3', test_locked=True,
                human_gold=False, tiles=tiles) | overrides


class TrainingContractTests(unittest.TestCase):
    def test_sampler_is_seeded_independent_and_work_uniform(self):
        queries = {'small': [('a', 0)], 'large': [('b', x) for x in range(99)]}
        left, right = _WorkSampler(queries, 17), _WorkSampler(queries, 17)
        self.assertEqual([left.next() for _ in range(100)], [right.next() for _ in range(100)])
        sampler = _WorkSampler(queries, 17)
        counts = Counter(sampler.next()[0] for _ in range(10000))
        self.assertTrue(4500 < counts['small'] < 5500, counts)
        self.assertTrue(4500 < counts['large'] < 5500, counts)

    def test_sampler_h_uniform_and_order_stable(self):
        first = _WorkSampler({'w': [('t', 1), ('t', 0)]}, 73)
        second = _WorkSampler({'w': [('t', 0), ('t', 1)]}, 73)
        self.assertEqual([first.next() for _ in range(100)], [second.next() for _ in range(100)])
        sampler = _WorkSampler({'w': [('t', 0), ('t', 1)]}, 73)
        counts = Counter(sampler.next()[2] for _ in range(10000))
        self.assertTrue(4500 < counts[0] < 5500, counts)
        for bad in ({}, {'w': []}):
            with self.assertRaises(ValueError):
                _WorkSampler(bad, 17)

    def test_manifest_preserves_disjoint_splits_and_records_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(cache_manifest([binding(x, x) for x in ('train', 'selection', 'dev')])))
            _, manifest, tiles = _manifest(directory)
            self.assertEqual(set(tiles), {'train', 'selection', 'dev'})
            self.assertEqual(manifest['_binding']['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_manifest_rejects_work_family_split_overlap_and_unsafe_paths(self):
        cases = [
            [binding('a', 'train', work='shared'), binding('b', 'selection', work='shared'), binding('c', 'dev')],
            [binding('a', 'train', family='shared'), binding('b', 'selection', family='shared'), binding('c', 'dev')],
            [binding('a', 'train'), binding('b', 'selection'), {**binding('c', 'dev'), 'path': '../outside.pt'}],
            [binding('a', 'train'), binding('b', 'selection'), binding('c', 'test')],
            [binding('a', 'train'), binding('a', 'selection'), binding('c', 'dev')],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for tiles in cases:
                with self.subTest(tiles=tiles):
                    path.write_text(json.dumps(cache_manifest(tiles)))
                    with self.assertRaises(ValueError):
                        _manifest(directory)

    def test_manifest_rejects_partial_or_wrong_contract_even_with_usable_tiles(self):
        rows = [binding(x, x) for x in ('train', 'selection', 'dev')]
        cases = [dict(schema='OtherCache@1'), dict(status='running'), dict(status='failed'),
                 dict(mode='raw_H_prediction'), dict(test_locked=False), dict(test_locked=1),
                 dict(human_gold=True), dict(human_gold=0)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for override in cases:
                with self.subTest(override=override):
                    path.write_text(json.dumps(cache_manifest(rows, **override)))
                    with self.assertRaisesRegex(ValueError, 'completed HMemberCache'):
                        _manifest(directory)
            path.write_text(json.dumps({'tiles': rows}))
            with self.assertRaisesRegex(ValueError, 'completed HMemberCache'):
                _manifest(directory)

    def test_hash_mismatch_is_rejected_before_pickle_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tile.pt'
            path.write_bytes(b'not a pickle')
            torch = Mock()
            cache = _TileCache({'t': dict(resolved_path=path, size_bytes=12, sha256='0' * 64)}, torch, 266)
            with self.assertRaisesRegex(ValueError, 'SHA/size'):
                cache.get('t')
            torch.load.assert_not_called()

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires Unix alarm')
    def test_deadline_interrupts_wait_and_restores_handler(self):
        original = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with _Deadline(0.01):
                time.sleep(0.1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_invalid_args_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            for kwargs in ({'kind': 'bad'}, {'steps': 0}, {'seed': -1}, {'max_seconds': 0}, {'deterministic': 1}):
                args = dict(cache='missing', output=output, kind='mlp', seed=17, steps=1, device='cpu')
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    train_run(**(args | kwargs))
                self.assertFalse(output.exists())

    def test_invalid_weighting_and_threshold_policy_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            for extra in ({'class_weights': (0., 1.)}, {'class_weights': (1.,)},
                          {'class_weights': (True, 1.)}, {'class_weights': (float('nan'), 1.)},
                          {'threshold_policy': 'DEV'}, {'threshold_policy': None}, {'retain_logits': 1}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    train_run('missing', output, 'mlp', 17, 1, device='cpu', **extra)
                self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                train_run('unused', directory, 'mlp', 17, 1, device='cpu')


@unittest.skipUnless(TORCH_AVAILABLE, 'native Torch unavailable; run during native gate')
class NativeTrainingTests(unittest.TestCase):
    def cache(self, directory, *, invalid_mask=False):
        import torch
        root = Path(directory) / 'cache'
        root.mkdir()
        tiles = []
        for split in ('train', 'selection', 'dev'):
            path = root / (split + '.pt')
            tile = dict(x=torch.arange(3 * 266, dtype=torch.float32).reshape(3, 266) / 800,
                        node_type=torch.tensor([0, 0, 1]), edge_index=torch.tensor([[0, 1], [2, 2]]),
                        edge_type=torch.tensor([0, 0]), queries=[dict(h_id='h', source=2,
                            candidates=[0, 1], y=[1, 0], mask=[1, 0] if invalid_mask else [True, True])])
            torch.save(tile, path)
            raw = path.read_bytes()
            tiles.append(dict(binding(split, split), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
        (root / 'manifest.json').write_text(json.dumps(cache_manifest(tiles)))
        return root

    def test_one_update_writes_complete_metrics_and_predictions(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            output = Path(directory) / 'run'
            result = train_run(cache, output, 'mlp', 17, 1, max_seconds=30, device='cpu')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['applied_updates'], 1)
            self.assertEqual(result['known_pair_exposure'], 2)
            self.assertEqual(result['evaluation_queries'], dict(initial_selection=1, final_selection=1, final_dev=1))
            self.assertEqual(result['final']['dev']['eligible_h'], 1)
            self.assertEqual(result['controls']['all_member']['primary_auc'], .5)
            self.assertEqual(result['controls']['all_nonmember']['primary_auc'], .5)
            self.assertEqual(result['controls']['duration_only']['primary_auc'], 0.)
            self.assertEqual(result['threshold_source'], 'inner_fit_selection_only')
            self.assertEqual(len((output / 'predictions.jsonl').read_text().splitlines()), 2)
            self.assertTrue(all('logits' not in json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()))
            self.assertEqual(json.loads((output / 'progress.json').read_text())['status'], 'completed')
            checkpoint = result['last_checkpoint']
            raw = (output / checkpoint['path']).read_bytes()
            self.assertEqual(checkpoint['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(checkpoint['size_bytes'], len(raw))
            state = torch.load(output / checkpoint['path'], weights_only=True)
            self.assertEqual(state['schema'], 'HMemberHeadCheckpoint@1')
            self.assertEqual(state['applied_updates'], 1)
            self.assertEqual(state['exposure_sha256'], result['exposure_sha256'])
            self.assertEqual(state['cache_manifest'], result['cache_manifest'])
            self.assertEqual(state['reason'], 'final_before_evaluation')
            self.assertFalse(state['resumable'])
            self.assertTrue(all(value.device.type == 'cpu' for value in state['model_state'].values()))

    def test_strict_graph_replays_state_and_records_runtime_with_restored_flags(self):
        import torch
        torch.set_num_threads(1)
        previous = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            results = [train_run(cache, Path(directory)/name, 'graph', 17, 2,
                max_seconds=30, device='cpu', deterministic=True) for name in ('left', 'right')]
            left, right = results
            self.assertTrue(left['runtime']['deterministic_algorithms'])
            self.assertFalse(left['runtime']['warn_only'])
            self.assertEqual(left['runtime'], left['config']['runtime'])
            states = [torch.load(Path(directory)/name/result['last_checkpoint']['path'], weights_only=True)
                for name, result in zip(('left', 'right'), results)]
            for key, value in states[0]['model_state'].items():
                torch.testing.assert_close(value, states[1]['model_state'][key], rtol=0, atol=0)
            self.assertEqual(left['exposure_sha256'], right['exposure_sha256'])
            self.assertEqual(left['final'], right['final'])
        self.assertEqual(previous, (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    def test_strict_runtime_restores_flags_and_retains_settings_on_training_error(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        previous = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'failed'
            with patch.object(training, '_forward', side_effect=RuntimeError('synthetic strict failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic strict failure'):
                    train_run(self.cache(directory), output, 'graph', 17, 1,
                        max_seconds=30, device='cpu', deterministic=True)
            result = json.loads((output/'result.json').read_text())
            self.assertTrue(result['runtime']['requested'])
            self.assertTrue(result['runtime']['deterministic_algorithms'])
            self.assertFalse(result['runtime']['warn_only'])
        self.assertEqual(previous, (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))

    def test_weighted_exact_policy_and_real_gradient_clipping_accounting(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        torch.set_num_threads(1)
        original_evaluate = training._evaluate
        original_clip = torch.nn.utils.clip_grad_norm_
        norms = []
        def evaluate(*args, **kwargs):
            rows = original_evaluate(*args, **kwargs)
            if args[3] == 'selection':
                rows[0]['scores'] = [.99, .98]
            return rows
        def clip(*args, **kwargs):
            norm = original_clip(*args, **kwargs)
            norms.append(float(norm))
            return norm
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training, '_evaluate', side_effect=evaluate), patch.object(
                    torch.nn.utils, 'clip_grad_norm_', side_effect=clip):
                result = train_run(self.cache(directory), Path(directory)/'weighted', 'mlp', 17, 2,
                    max_seconds=30, device='cpu', deterministic=True,
                    class_weights=(2., .5), threshold_policy='exact', selection_updates=(1, 2))
            self.assertEqual(result['config']['class_weights'], [2., .5])
            self.assertEqual(result['config']['threshold_policy'], 'exact')
            self.assertEqual(result['initial']['selection']['threshold'], .99)
            self.assertEqual([p['threshold'] for p in result['selection_curve']], [.99, .99])
            self.assertEqual(result['threshold'], .99)
            self.assertEqual(result['selected_update'], 1)
            self.assertEqual(result['evaluation_queries']['curve_selection'], 2)
            self.assertEqual(result['evaluation_queries']['final_selection'], 0)
            self.assertEqual(result['evaluation_queries']['final_dev'], 1)
            self.assertEqual(len(norms), result['applied_updates'])
            self.assertEqual(result['gradient_clipping_count'], sum(n > 1. for n in norms))
            self.assertEqual(result['gradient_clipping_fraction'], sum(n > 1. for n in norms)/len(norms))

    def test_retained_raw_logits_match_scores_without_additional_forwards(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        torch.set_num_threads(1)
        original = training._forward
        calls = 0
        def forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'logits'
            with patch.object(training, '_forward', side_effect=forward):
                result = train_run(self.cache(directory), output, 'graph', 17, 2,
                    max_seconds=30, device='cpu', deterministic=True, retain_logits=True,
                    selection_updates=(1, 2))
            self.assertTrue(result['config']['retain_logits'])
            self.assertEqual(calls, 6)  # initial + two updates + two curve points + DEV
            rows = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual({row['split'] for row in rows}, {'selection', 'dev'})
            for row in rows:
                self.assertEqual(len(row['logits']), len(row['scores']))
                torch.testing.assert_close(torch.tensor(row['logits']).sigmoid(),
                    torch.tensor(row['scores']), rtol=1e-6, atol=1e-7)

    def test_fixed_update_is_retained_and_reused_by_identical_final(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            output = Path(directory) / 'run'
            result = train_run(cache, output, 'mlp', 17, 50, max_seconds=30, device='cpu')
            self.assertEqual([x['applied_updates'] for x in result['checkpoints']], [50])
            self.assertEqual(list(output.glob('checkpoint-*.pt')), [output / 'checkpoint-000050.pt'])
            state = torch.load(output / 'checkpoint-000050.pt', weights_only=True)
            self.assertEqual(state['reason'], 'fixed_update')
            self.assertEqual(state['exposure_sha256'], result['exposure_sha256'])

    def test_final_weights_survive_evaluation_timeout(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        torch.set_num_threads(1)
        original = training._evaluate
        calls = 0

        def timed_evaluation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError('synthetic final-evaluation deadline')
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            output = Path(directory) / 'run'
            with patch.object(training, '_evaluate', side_effect=timed_evaluation):
                with self.assertRaisesRegex(TimeoutError, 'synthetic'):
                    train_run(cache, output, 'mlp', 17, 1, max_seconds=30, device='cpu')
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['status'], 'aborted')
            self.assertEqual(result['applied_updates'], 1)
            self.assertEqual(len(result['checkpoints']), 1)
            state = torch.load(output / result['last_checkpoint']['path'], weights_only=True)
            self.assertEqual(state['reason'], 'final_before_evaluation')

    def test_training_error_retains_partial_weights_and_original_failure(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        torch.set_num_threads(1)
        original = training._forward
        calls = 0

        def failed_forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:  # initial selection, first update, second update
                raise RuntimeError('synthetic training failure')
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            output = Path(directory) / 'run'
            with patch.object(training, '_forward', side_effect=failed_forward):
                with self.assertRaisesRegex(RuntimeError, 'synthetic training failure'):
                    train_run(cache, output, 'mlp', 17, 3, max_seconds=30, device='cpu')
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['applied_updates'], 1)
            self.assertEqual(result['attempted_updates'], 2)
            state = torch.load(output / result['last_checkpoint']['path'], weights_only=True)
            self.assertEqual(state['applied_updates'], 1)
            self.assertEqual(state['query_exposure'], 1)
            self.assertEqual(state['known_pair_exposure'], 2)
            self.assertEqual(state['reason'], 'partial_after_RuntimeError')

    def test_invalid_mask_writes_failed_evidence_and_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, invalid_mask=True)
            output = Path(directory) / 'run'
            with self.assertRaisesRegex(ValueError, 'boolean'):
                train_run(cache, output, 'mlp', 17, 1, max_seconds=30, device='cpu')
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['applied_updates'], 0)

    def test_wall_abort_has_compact_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory)
            output = Path(directory) / 'run'
            with self.assertRaises(TimeoutError):
                train_run(cache, output, 'graph', 17, 100000, max_seconds=0.02, device='cpu')
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['status'], 'aborted')
            self.assertLess(result['applied_updates'], 100000)


if __name__ == '__main__':
    unittest.main()
