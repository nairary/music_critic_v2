"""Independent synthetic review of explicit launch admission and gate receipts."""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from music_critic.experiments.hierarchical_graph_analyzer import pipeline

_spec = importlib.util.spec_from_file_location('hiergraph_review_cli',
    Path(__file__).parents[2] / 'scripts/run_hiergraph.py')
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


class LaunchAdmissionReviewTests(unittest.TestCase):
    def fixture(self, root, mode='cache'):
        config = dict(source_manifest_sha256='a' * 64, target_manifest_sha256='b' * 64,
                      cache_manifest_sha256='c' * 64, cache_wall_seconds=60,
                      gate_wall_seconds=40, condition_wall_seconds=90)
        raw = json.dumps(config, sort_keys=True).encode()
        output = root / 'fresh-bound-output'
        cap = {'cache': 60, 'gate': 40, 'train': 90}[mode]
        launch = dict(mode=mode, config_sha256=sha256(raw).hexdigest(),
                      source_manifest_sha256=config['source_manifest_sha256'],
                      target_manifest_sha256=config['target_manifest_sha256'],
                      output_name=output.name, wall_seconds=cap)
        if mode != 'cache':
            launch['cache_manifest_sha256'] = config['cache_manifest_sha256']
        if mode == 'train':
            launch.update(kind='E3', seed=42)
        record = dict(protocol={'launch': launch}, budget={'planned': {'total_wall_seconds': 100}})
        return raw, config, record, output

    def test_exact_binding_accepts_each_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('cache', 'gate', 'train'):
                raw, config, record, output = self.fixture(Path(directory), mode)
                result = cli.admit_launch(raw, config, record, mode, output, 'E3', 42)
                self.assertEqual(result, record['protocol']['launch'])
                self.assertFalse(output.exists())

    def test_source_config_data_cache_condition_and_leaf_drift_rejected(self):
        mutations = {'config_sha256': '0' * 64, 'source_manifest_sha256': '0' * 64,
                     'target_manifest_sha256': '0' * 64, 'cache_manifest_sha256': '0' * 64,
                     'kind': 'E5', 'seed': 17, 'output_name': 'different',
                     'wall_seconds': 89, 'mode': 'gate'}
        with tempfile.TemporaryDirectory() as directory:
            for field, value in mutations.items():
                raw, config, record, output = self.fixture(Path(directory), 'train')
                record['protocol']['launch'][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'frozen planned'):
                    cli.admit_launch(raw, config, record, 'train', output, 'E3', 42)
                self.assertFalse(output.exists())

    def test_missing_binding_overbudget_and_existing_output_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, config, record, output = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'exact launch'):
                cli.admit_launch(raw, config, {'protocol': {}}, 'cache', output, None, 17)
            record['budget']['planned']['total_wall_seconds'] = 59
            with self.assertRaisesRegex(ValueError, 'budget'):
                cli.admit_launch(raw, config, record, 'cache', output, None, 17)
            record['budget']['planned']['total_wall_seconds'] = 100
            output.mkdir()
            sentinel = output / 'preserve.txt'
            sentinel.write_text('existing immutable attempt')
            with self.assertRaisesRegex(FileExistsError, 'fresh output'):
                cli.admit_launch(raw, config, record, 'cache', output, None, 17)
            self.assertEqual(sentinel.read_text(), 'existing immutable attempt')

    def test_fractional_or_boolean_cap_cannot_disable_outer_alarm(self):
        with tempfile.TemporaryDirectory() as directory:
            for cap in (.5, True):
                raw, config, record, output = self.fixture(Path(directory))
                config['cache_wall_seconds'] = cap
                raw = json.dumps(config, sort_keys=True).encode()
                record['protocol']['launch'].update(config_sha256=sha256(raw).hexdigest(), wall_seconds=cap)
                with self.subTest(cap=cap), self.assertRaises(ValueError):
                    cli.admit_launch(raw, config, record, 'cache', output, None, 17)

    def test_source_verification_authenticates_bytes_before_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'model.py'
            source.write_bytes(b'synthetic source only\n')
            binding = dict(path=source.name, sha256=sha256(source.read_bytes()).hexdigest(),
                           size_bytes=source.stat().st_size)
            manifest = root / 'sources.json'
            manifest.write_text(json.dumps({'files': [binding]}))
            digest = sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(cli.verify_source(root, manifest, digest), digest)
            source.write_bytes(b'changed source bytes\n')
            with self.assertRaises(ValueError):
                cli.verify_source(root, manifest, digest)


class CoverageReviewTests(unittest.TestCase):
    def test_point_coverage_aggregates_owned_events_only(self):
        class Dataset:
            def targets_for_tile(self, work, **metadata):
                return [dict(coverage=dict(total_events=10, not_owned=7,
                    point_events=3, point_events_owned=metadata['owned_points'], admitted_events=2))]
        cache = SimpleNamespace(works={'w': dict(split='fit', exclusions=[], tiles=[
            dict(id='left', metadata={'owned_points': 1}),
            dict(id='right', metadata={'owned_points': 2})])})
        samples = pipeline.Samples(Dataset(), cache, 'fit')
        self.assertEqual(samples.coverage['point_events_owned'], 3)
        self.assertNotIn('point_events', samples.coverage)
        self.assertNotIn('total_events', samples.coverage)
        self.assertEqual(samples.coverage['admitted_events'], 4)


class PartialGateReviewTests(unittest.TestCase):
    def test_failed_second_sample_retains_completed_first_sample_measurement(self):
        class Samples:
            coverage = {}
            rows = [('w', {'id': f'tile{i}', 'nodes': i + 2}) for i in range(3)]
            def __init__(self, *args):
                pass
            def __len__(self):
                return 3
            def __getitem__(self, index):
                if index == 1:
                    raise ValueError('synthetic reader failure after first gate sample')
                raw = dict(x=torch.zeros(2, 266), node_type=torch.tensor([6, 6]),
                    edge_index=torch.empty((2, 0), dtype=torch.long), edge_type=torch.empty(0, dtype=torch.long),
                    node_time=torch.tensor([[0., 0.], [1., 1.]]), time_index=torch.tensor([0, 1]),
                    time_positions=torch.tensor([0., 1.]), bank_indices=torch.tensor([[0, 1]]),
                    pool_mask=torch.tensor([True, True]))
                target = dict(ids=['h'], spans=[[0., 1.]],
                    fields={'H.function': {'values': [1], 'mask': [True]}},
                    work_id='w', family_id='f', view_id='v', scope_id='harmonies', split='fit',
                    core_span=(Fraction(0), Fraction(1)), halo_span=(Fraction(0), Fraction(1)))
                return raw, target
        dataset = SimpleNamespace(head_sizes={'H.function': 2})
        cache = SimpleNamespace(works={'w': {'exclusions': []}})
        config = dict(device='cpu', target_manifest_sha256='a' * 64,
                      cache_manifest_sha256='b' * 64, gate_wall_seconds=30,
                      model=dict(hidden=8, rounds=1, max_events=2))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pipeline, 'open_inputs', return_value=(dataset, cache)), \
                patch.object(pipeline, 'Samples', Samples):
            output = Path(directory) / 'gate'
            with self.assertRaisesRegex(ValueError, 'synthetic reader failure'):
                pipeline.native_gate(config, output)
            result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['applied_updates'], 0)
        self.assertEqual(len(result['conditions']), 1)
        condition = result['conditions'][0]
        self.assertEqual(condition['kind'], 'E1')
        self.assertEqual(len(condition['samples']), 1)
        self.assertEqual(condition['samples'][0]['tile'], 'tile0')
        self.assertGreater(condition['samples'][0]['gradient_tensors'], 0)


if __name__ == '__main__':
    unittest.main()
