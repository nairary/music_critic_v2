"""Independent synthetic launch/recovery review; never opens project corpora."""
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from music_critic.experiments.hierarchical_graph_analyzer import pipeline, runtime
from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record
from test_registry import _record

SPEC = importlib.util.spec_from_file_location('hiergraph_cli_review',
    Path(__file__).resolve().parents[2] / 'scripts/run_hiergraph.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3))
    def forward(self, raw):
        return {'spans': self.logits.new_tensor([[0., 1.]]),
                'object_logits': self.logits[:1], 'field_logits': {'H.function': self.logits[1:].reshape(1, 2)}}
    def oracle_readout(self, raw, spans):
        return dict(self.forward(raw), spans=spans)


class SyntheticSamples(list):
    def __init__(self, dataset, cache, split):
        target = dict(ids=['h'], spans=[[0., 1.]], fields={'H.function': {'values': [1], 'mask': [True]}},
                      split=split, work_id=split, family_id=split, view_id='v', scope_id='harmonies',
                      core_span=(Fraction(0), Fraction(1)), halo_span=(Fraction(0), Fraction(1)))
        super().__init__([({'x': torch.ones(1, 1)}, target)])
        self.metadata = [target]
        self.coverage = {'admitted_events': 1}
    def missing_reports(self): return []


def config():
    return dict(device='cpu', learning_rate=.01, condition_wall_seconds=30,
                selection_steps=[1, 2], clip_grad_norm=1., evaluation={},
                source_manifest='synthetic-source.json', source_manifest_sha256='a'*64,
                target_manifest_sha256='b'*64, cache_manifest_sha256='c'*64)


def bind(config_raw, cfg, output):
    return dict(mode='train', config_sha256=sha256(config_raw).hexdigest(),
                source_manifest_sha256=cfg['source_manifest_sha256'],
                target_manifest_sha256=cfg['target_manifest_sha256'],
                cache_manifest_sha256=cfg['cache_manifest_sha256'],
                output_name=output.name, wall_seconds=cfg['condition_wall_seconds'], kind='E1', seed=17)


class PipelineReviewTests(unittest.TestCase):
    def test_frozen_launch_rejects_config_drift_and_larger_wall_budget(self):
        cfg = config(); raw = json.dumps(cfg).encode()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            record = {'protocol': {'launch': bind(raw, cfg, output)},
                      'budget': {'planned': {'total_wall_seconds': 30}}}
            cli.admit_launch(raw, cfg, record, 'train', output, 'E1', 17)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                cli.admit_launch(raw+b' ', cfg, record, 'train', output, 'E1', 17)
            record['budget']['planned']['total_wall_seconds'] = 10
            with self.assertRaisesRegex(ValueError, 'budget'):
                cli.admit_launch(raw, cfg, record, 'train', output, 'E1', 17)

    def test_synthetic_training_selects_before_single_final_dev(self):
        from types import SimpleNamespace
        calls = []
        original = runtime.evaluate
        def track(*args, **kwargs):
            calls.append((kwargs.get('split'), kwargs.get('oracle', False)))
            return original(*args, **kwargs)
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as temp, \
                 patch.object(pipeline, 'open_inputs', return_value=(SimpleNamespace(head_sizes={'H.function': 2}), None)), \
                 patch.object(pipeline, 'Samples', SyntheticSamples), \
                 patch.object(pipeline, 'make_model', side_effect=lambda *args: TinyModel()), \
                 patch.object(runtime, 'evaluate', side_effect=track):
                output = Path(temp) / 'run'
                result = cli.run_train(config(), output, 'E1', 17)
                exposure = (output / 'exposure.jsonl').read_text().splitlines()
                self.assertEqual(len(exposure), 2)
                self.assertEqual(result['actual_state']['applied_updates'], 2)
                self.assertEqual(result['selected']['attempted_steps'], 1)
                self.assertEqual(calls, [('selection', False), ('selection', False), ('dev', False), ('dev', True)])
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_cli_failure_preserves_actual_applied_updates_in_terminal_record(self):
        cfg = config(); raw = json.dumps(cfg).encode()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root/'run'; cfg_path = root/'config.json'; cfg_path.write_bytes(raw)
            record = _record(experiment_id='synthetic-hiergraph-failure', status='planned')
            record.update(protocol={'kind': 'E1', 'launch': bind(raw, cfg, output)},
                          budget={'planned': {'total_wall_seconds': 30}, 'actual': {}})
            path = root/'synthetic-hiergraph-failure.json'
            path.write_bytes(canonical_json_bytes(seal_record(record)) + b'\n')
            def fail(config, directory, kind, seed):
                output.mkdir()
                (output/'result.json').write_text(json.dumps({'status': 'failed',
                    'actual_state': {'attempted_steps': 7, 'applied_updates': 6}}))
                raise ValueError('synthetic update failure')
            with patch.object(cli, 'verify_source', return_value='a'*64), patch.object(cli, 'run_train', side_effect=fail):
                with self.assertRaisesRegex(ValueError, 'synthetic update failure'):
                    cli.main(['train', '--config', str(cfg_path), '--record', str(path),
                              '--output', str(output), '--kind', 'E1'])
            terminal = load_record(path)
            self.assertEqual(terminal['status'], 'failed')
            self.assertEqual(terminal['budget']['actual']['applied_updates'], 6)
            self.assertEqual(terminal['results']['actual_state']['attempted_steps'], 7)
