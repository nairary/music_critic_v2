"""Shared RNG restores streams atomically without experimental model imports."""
from copy import deepcopy
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from music_critic.experiments.analysisgnn import training_rng as rng


def draws():
    return random.random(), np.random.random(), torch.rand(4)


def equal_draws(a, b):
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])


def test_restore_repeats_all_streams_and_preserves_historical_schema():
    with rng.preserve_caller_rng():
        state = rng.capture_safe_rng()
        assert state["schema_version"] == "HarmonicFusionSafeRNG@1.0.0"
        expected = draws()
        draws()
        rng.restore_safe_rng(state, device_type="cpu")
        equal_draws(draws(), expected)


def test_invalid_numpy_state_cannot_partially_change_python_or_torch():
    with rng.preserve_caller_rng():
        before = rng.capture_safe_rng()
        expected = draws()
        rng.restore_safe_rng(before, device_type="cpu")
        invalid = deepcopy(before)
        invalid["numpy"]["position"] = -1
        with pytest.raises(ValueError):
            rng.restore_safe_rng(invalid, device_type="cpu")
        equal_draws(draws(), expected)


def test_context_restores_after_exception():
    with rng.preserve_caller_rng():
        before = rng.capture_safe_rng()
        expected = draws()
        rng.restore_safe_rng(before, device_type="cpu")
        with pytest.raises(RuntimeError):
            with rng.preserve_caller_rng():
                draws()
                raise RuntimeError("interrupted")
        equal_draws(draws(), expected)


def test_temporal_cold_import_without_any_fusion_module():
    code = """
import importlib.abc, json, sys
class RejectFusion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('music_critic.') and any(x in fullname for x in (
            'harmonic_fusion', 'harmonic_logit_fusion', 'fusion_accelerated',
            'task_attention_fusion', 'grouped_mmoe')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectFusion())
from music_critic.experiments.analysisgnn.temporal_slices import campaign, checkpoint
from music_critic.experiments.analysisgnn.temporal_slices.model import TemporalSliceModel
model = TemporalSliceModel()
assert campaign.scientific_config()['workers'] == 4
print(json.dumps({'ok': True}))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True, env=os.environ.copy())
    assert json.loads(result.stdout)["ok"] is True
