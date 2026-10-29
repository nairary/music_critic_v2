"""Runtime control-flow tests; tensor/native replay is tested by the probe."""
from contextlib import contextmanager
import importlib
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class FakeTorch:
    __version__ = 'test-backend'
    version = SimpleNamespace(cuda=None)

    def __init__(self):
        self.algorithms = False
        self.warn_only = True
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, deterministic=False))
        self.cuda = SimpleNamespace(is_initialized=lambda: False)

    def are_deterministic_algorithms_enabled(self):
        return self.algorithms

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, enabled, *, warn_only):
        self.algorithms, self.warn_only = enabled, warn_only


@contextmanager
def runtime_backend():
    backend = FakeTorch()
    with patch.dict(sys.modules, torch=backend):
        module = importlib.import_module('music_critic.experiments.harmonic_event_reasoner.reproducibility')
        yield module, backend


class ReproducibilityTests(unittest.TestCase):
    def test_disabled_runtime_preserves_historical_settings(self):
        with runtime_backend() as (module, backend):
            with module.deterministic_runtime('cpu') as state:
                self.assertIs(state['requested'], False)
                self.assertIs(state['deterministic_algorithms'], False)
                self.assertIs(state['warn_only'], True)
                self.assertIs(backend.backends.cudnn.benchmark, True)
            self.assertIs(backend.algorithms, False)
            self.assertIs(backend.warn_only, True)

    def test_enabled_runtime_is_strict_and_restores_after_failure(self):
        with runtime_backend() as (module, backend):
            with self.assertRaisesRegex(RuntimeError, 'synthetic body failure'):
                with module.deterministic_runtime('cpu', enabled=True) as state:
                    self.assertIs(state['requested'], True)
                    self.assertIs(state['deterministic_algorithms'], True)
                    self.assertIs(state['warn_only'], False)
                    self.assertIs(backend.backends.cudnn.benchmark, False)
                    self.assertIs(backend.backends.cudnn.deterministic, True)
                    raise RuntimeError('synthetic body failure')
            self.assertIs(backend.algorithms, False)
            self.assertIs(backend.warn_only, True)
            self.assertIs(backend.backends.cudnn.benchmark, True)
            self.assertIs(backend.backends.cudnn.deterministic, False)

    def test_cuda_requires_launch_provided_cublas_environment(self):
        for workspace in (None, '', ':invalid:'):
            with self.subTest(workspace=workspace), runtime_backend() as (module, backend), patch.dict(os.environ):
                if workspace is None:
                    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
                else:
                    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
                before = dict(os.environ)
                with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                    with module.deterministic_runtime('cuda:0', enabled=True):
                        self.fail('must reject before entering body')
                self.assertEqual(dict(os.environ), before)
                self.assertIs(backend.algorithms, False)
                self.assertIs(backend.warn_only, True)

    def test_cuda_accepts_valid_workspace_and_records_initialization(self):
        for workspace in (':4096:8', ':16:8'):
            with self.subTest(workspace=workspace), runtime_backend() as (module, backend), patch.dict(os.environ, CUBLAS_WORKSPACE_CONFIG=workspace):
                backend.cuda.is_initialized = lambda: True
                with module.deterministic_runtime('cuda:0', enabled=True) as state:
                    self.assertEqual(state['cublas_workspace_config'], workspace)
                    self.assertIs(state['cuda_initialized_on_entry'], True)
                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], workspace)

    def test_invalid_toggle_rejected(self):
        with runtime_backend() as (module, _), self.assertRaisesRegex(ValueError, 'boolean'):
            with module.deterministic_runtime('cpu', enabled=1):
                self.fail('must not enter body')


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'native Torch unavailable')
class NativeReplayTests(unittest.TestCase):
    @staticmethod
    def probe_module():
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('hmember_determinism_probe',
            Path(__file__).resolve().parents[2] / 'scripts' / 'probe_hmember_determinism.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_content_hash_is_named_shape_dtype_and_byte_bound(self):
        import torch
        probe = self.probe_module()
        value = torch.tensor([1., 2.])
        original = probe.tensor_content_hash(dict(x=value))
        self.assertEqual(original, probe.tensor_content_hash(dict(x=value.clone())))
        for changed in (dict(y=value), dict(x=value.reshape(1, 2)),
                        dict(x=value.double()), dict(x=torch.tensor([1., 3.]))):
            self.assertNotEqual(original, probe.tensor_content_hash(changed))
        comparison = probe.compare_tensors(dict(x=value), dict(x=torch.tensor([1., 3.])))
        self.assertIs(comparison['exact_equal'], False)
        self.assertEqual(comparison['max_abs_diff'], 1.)

    def test_tiny_synthetic_cpu_training_replays_all_observed_stages(self):
        import torch
        probe = self.probe_module()
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            result = probe.run_probe(device='cpu', deterministic=True, repeats=2,
                steps=2, nodes=8, edges=24, reduction_edges=64, kinds=('self', 'graph'))
        finally:
            torch.set_num_threads(previous)
        self.assertIs(result['corpus_accessed'], False)
        self.assertIs(result['index_add']['comparisons_to_first'][0]['exact_equal'], True)
        for kind, stages in result['models'].items():
            for stage, summary in stages.items():
                with self.subTest(kind=kind, stage=stage):
                    self.assertEqual(len(set(summary['content_sha256'])), 1)
                    self.assertIs(summary['comparisons_to_first'][0]['exact_equal'], True)

    def test_real_runtime_preserves_rng_and_restores_flags(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner.reproducibility import deterministic_runtime
        rng = torch.get_rng_state().clone()
        previous = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        with deterministic_runtime('cpu', enabled=True):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(previous, (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))


if __name__ == '__main__':
    unittest.main()
