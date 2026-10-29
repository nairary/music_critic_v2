"""Duration selection uses selection only and preserves a continuous optimizer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from music_critic.experiments.harmonic_event_reasoner.training import (
    _selection_schedule, _select_curve_point, train_run,
)


class CurveContractTests(unittest.TestCase):
    def test_schedule_is_explicit_sorted_unique_and_includes_final(self):
        self.assertEqual(_selection_schedule((2, 5, 10), 10), (2, 5, 10))
        self.assertEqual(_selection_schedule((), 10), ())
        for values in ((0, 10), (True, 10), (5, 2, 10), (2, 2, 10), (2, 5), (2, 11)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _selection_schedule(values, 10)

    def test_selects_highest_selection_auc_with_earliest_tie(self):
        points=[dict(update=i, selection=dict(primary_auc=a)) for i,a in [(2,.8),(5,.9),(10,.9)]]
        self.assertEqual(_select_curve_point(points)['update'],5)
        for bad in ([], [dict(update=2,selection=dict(primary_auc=None))],
                    [dict(update=2,selection=dict(primary_auc=float('nan'))) ]):
            with self.assertRaises(ValueError):_select_curve_point(bad)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'native Torch unavailable')
class NativeCurveTests(unittest.TestCase):
    def cache(self, directory):
        from test_hmember_training import NativeTrainingTests
        return NativeTrainingTests().cache(directory)

    def test_dev_once_uses_selected_weights_not_last_weights(self):
        import torch
        from music_critic.experiments.harmonic_event_reasoner import training
        torch.set_num_threads(1)
        original=training._evaluate;selection_calls=0;dev_calls=0;chosen=None
        def evaluate(model, store, bindings, split, *args):
            nonlocal selection_calls,dev_calls,chosen
            rows=original(model,store,bindings,split,*args)
            if split=='selection':
                selection_calls+=1
                if selection_calls==2:
                    chosen={k:v.detach().clone() for k,v in model.state_dict().items()}
                if selection_calls>1:
                    rows[0]['scores']=[.9,.1] if selection_calls==2 else [.1,.9]
            else:
                dev_calls+=1
                for k,v in model.state_dict().items():torch.testing.assert_close(v,chosen[k],rtol=0,atol=0)
            return rows
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(training,'_evaluate',side_effect=evaluate):
                result=train_run(self.cache(directory),Path(directory)/'run','mlp',17,3,
                    max_seconds=30,device='cpu',selection_updates=(1,2,3))
            self.assertEqual(result['applied_updates'],3)
            self.assertEqual(result['selected_update'],1)
            self.assertEqual(selection_calls,4)
            self.assertEqual(dev_calls,1)
            self.assertEqual([x['update'] for x in result['selection_curve']],[1,2,3])
            self.assertEqual(result['final']['selection']['primary_auc'],1.)
            self.assertEqual(result['selected_checkpoint']['applied_updates'],1)
            self.assertEqual(result['last_checkpoint']['applied_updates'],3)
            self.assertEqual(result['selected_predictions_checkpoint'], result['selected_checkpoint'])
            self.assertTrue(result['selected_selection_rows_reused_from_curve'])
            rows=[json.loads(line) for line in (Path(directory)/'run'/'predictions.jsonl').read_text().splitlines()]
            retained=[row for row in rows if row['split']=='selection']
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0]['scores'], [.9,.1])
            from music_critic.experiments.harmonic_event_reasoner.metrics import score_metrics
            self.assertEqual(score_metrics(retained,result['threshold']), result['final']['selection'])
            self.assertEqual(len([row for row in rows if row['split']=='dev']),1)

    def test_selection_evaluations_do_not_change_training_schedule_or_final_weights(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            cache=self.cache(directory);p=Path(directory)
            a=train_run(cache,p/'plain','mlp',17,3,max_seconds=30,device='cpu')
            b=train_run(cache,p/'curve','mlp',17,3,max_seconds=30,device='cpu',selection_updates=(1,2,3))
            self.assertEqual(a['exposure_sha256'],b['exposure_sha256'])
            left=torch.load(p/'plain'/a['last_checkpoint']['path'],weights_only=True)
            right=torch.load(p/'curve'/b['last_checkpoint']['path'],weights_only=True)
            for k,v in left['model_state'].items():torch.testing.assert_close(v,right['model_state'][k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
