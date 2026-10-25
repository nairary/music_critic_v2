"""Synthetic whole-context integration with the existing registered native gate."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record
from music_critic.pll.cli import load_run_config
from music_critic.pll.cloud import gate, gate_bindings
from music_critic.pll.whole_data import WholeDataManifest
from tests.pll.test_whole_runtime import whole_run


def test_whole_runtime_does_not_import_windowed_pipeline_or_hydra():
    code = '''
import sys, importlib.abc
class BlockWindowed(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "hydra" or fullname.startswith("music_critic.pdmx_pipeline"):
            raise ImportError("Unexpected windowed preparation dependency: " + fullname)
sys.meta_path.insert(0, BlockWindowed())
from music_critic.pll.whole_data import WholeDataManifest
from music_critic.pll.whole_model import WholePieceHCM
from music_critic.pll.cloud import _model
'''
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable,'-c',code], cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_registered_gate_uses_whole_model_and_sampler(tmp_path):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        cfg, manifest, record, output = whole_run(tmp_path, total=3)
        loaded = WholeDataManifest.load(manifest)
        cases = dict(schema='PLLNativeGateCases@1', cases=[dict(record_id=loaded.records[0].record_id,
            start_bar=10, bars=1, role='dense')])
        case_path = tmp_path/'cases.json'; case_path.write_text(json.dumps(cases))
        plan = load_record(record); plan['budget']['planned_updates'] = 3
        plan['pll_gate'] = gate_bindings(load_run_config(cfg), loaded, 'cpu', cases)
        record.write_bytes(canonical_json_bytes(seal_record(plan))+b'\n')
        receipt = gate(cfg, manifest, record, output, cases_path=case_path, device='cpu')
        assert receipt['results']['resume_verified'] is True
        assert receipt['results']['native_cuda_admitted'] is False
        metric = json.loads((output/'gate-step-01.json').read_text())['metrics']
        assert metric['context_diagnostics'][0]['source_bars'] == 40
        assert 'bar:32' in metric['context_diagnostics'][0]['motifs']['scale_counts']
        assert load_record(record)['status'] == 'completed'
    finally:
        torch.set_num_threads(previous)
