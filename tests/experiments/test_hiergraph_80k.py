"""Synthetic requested-update accounting; no corpus or GPU workload."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from music_critic.experiments.hierarchical_graph_analyzer import pipeline,runtime
from test_hiergraph_pipeline_review import cli,TinyModel,config,SyntheticSamples

class MixedSamples(SyntheticSamples):
    def __init__(self,dataset,cache,split):
        super().__init__(dataset,cache,split)
        if split=='fit':
            empty=dict(self[0][1]);empty.update(ids=[],spans=[],fields={'H.function':dict(values=[],mask=[])},view_id='unknown')
            self.insert(0,({'x':torch.ones(1,1)},empty));self.metadata=[row[1] for row in self]

class AppliedUpdateTests(unittest.TestCase):
    def run_case(self,directory,cap):
        cfg=config();cfg.update(selection_unit='applied_updates',max_attempted_steps=cap)
        with patch.object(pipeline,'open_inputs',return_value=(SimpleNamespace(head_sizes={'H.function':2}),None)),patch.object(pipeline,'Samples',MixedSamples),patch.object(pipeline,'make_model',side_effect=lambda *args:TinyModel()),patch.object(runtime,'_draw',side_effect=lambda groups,seed,cursor:cursor%2):
            return cli.run_train(cfg,directory,'E1',17)
    def test_requested_updates_exclude_unknown_draws(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_case(Path(d)/'run',8)
            self.assertEqual(result['actual_state']['applied_updates'],2)
            self.assertEqual(result['actual_state']['attempted_steps'],4)
            self.assertEqual(result['selected']['applied_updates'],1)
            self.assertEqual(result['selected']['attempted_steps'],2)
    def test_draw_cap_fails_with_retained_confirmed_updates(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run'
            with self.assertRaisesRegex(RuntimeError,'attempted'):
                self.run_case(output,3)
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['actual_state']['applied_updates'],1)
            self.assertEqual(result['actual_state']['attempted_steps'],3)
