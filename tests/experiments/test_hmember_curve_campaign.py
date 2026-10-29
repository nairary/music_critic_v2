import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
SPEC=importlib.util.spec_from_file_location('curve_campaign',ROOT/'scripts/run_hmember_curve_campaign.py')
campaign=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(campaign)


class CurveCampaignTests(unittest.TestCase):
    def test_80k_profile_isolated_ids_and_fits_campaign_cap(self):
        values=campaign.profile_settings('80k')
        self.assertEqual(campaign.identifier('graph',17,'80k'),'EXP-HMEMBER-80K-S17-GRAPH-001')
        self.assertEqual(values['updates'],(2000,5000,10000,20000,30000,40000,60000,80000))
        self.assertLess(values['worker_seconds'],values['condition_seconds'])
        self.assertLess(12*(values['condition_seconds']+10),campaign.CAMPAIGN_SECONDS)
        self.assertIn('80k',campaign.worker_command('runner','cache','output','graph',17,'80k'))
        with self.assertRaises(ValueError):campaign.profile_settings('unknown')

    def test_80k_cli_passes_full_schedule_and_budget_to_unchanged_trainer(self):
        torch=SimpleNamespace(set_num_threads=Mock())
        argv=['runner','run','--cache','cache','--output','out','--kind','mlp','--seed','17','--profile','80k']
        with patch.dict(sys.modules,{'torch':torch}),patch.object(sys,'argv',argv),\
             patch('music_critic.experiments.harmonic_event_reasoner.training.train_run') as train:
            self.assertEqual(campaign.main(),0)
        self.assertEqual(train.call_args.args,('cache','out','mlp',17,80000))
        self.assertEqual(train.call_args.kwargs['max_seconds'],1090)
        self.assertEqual(train.call_args.kwargs['selection_updates'],campaign.profile_settings('80k')['updates'])

    def test_80k_cannot_launch_with_old30k_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);args=SimpleNamespace(profile='80k',records=p/'records',cache=p/'cache',
                binding=p/'binding',campaign=p/'campaign')
            with patch.object(campaign,'load_record',return_value={'status':'planned'}),\
                 patch.object(campaign,'verified_inputs',return_value={}):
                with self.assertRaisesRegex(ValueError,'80k binding'):campaign.supervise(args)
            self.assertFalse((p/'campaign').exists())

    def test_new_ids_fixed_schedule_and_bounded_caps(self):
        self.assertEqual(campaign.identifier('graph',17),'EXP-HMEMBER-LONG-S17-GRAPH-001')
        self.assertEqual(campaign.UPDATES,(2000,5000,10000,20000,30000))
        self.assertLess(campaign.WORKER_SECONDS,campaign.CONDITION_SECONDS)
        self.assertLess(12*(campaign.CONDITION_SECONDS+10),campaign.CAMPAIGN_SECONDS)
        cmd=campaign.worker_command('runner','cache','output','graph',17)
        self.assertEqual(cmd[-2:],['--seed','17'])
        self.assertNotIn('--checkpoint',cmd)

    def test_input_hash_binding_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'manifest.json').write_text('{}')
            (p/'binding.json').write_text(json.dumps(dict(cache_manifest=dict(size_bytes=2,sha256='0'*64))))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                campaign.verified_inputs(p,p/'binding.json')

    def test_compact_curve_uses_selection_not_dev_for_point_scores(self):
        point=dict(update=2000,selection={'primary_auc':.8,'global':dict(macro_f1=.7,fpr=.2)},
            threshold=.6,unique_train_h=10,known_pair_exposure=100,exposure_sha256='a'*64)
        result=campaign.compact_result(dict(selection_curve=[point],selected_update=2000))
        self.assertEqual(result['curve'][0]['primary_auc'],.8)
        self.assertNotIn('dev',result['curve'][0])

    def test_supervisor_interrupt_retains_worker_update_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);states={};writes=[]
            def load(path):return dict(status=states.get(path.stem,'planned'))
            def transition(path,status,**kwargs):
                states[path.stem]=status;writes.append((path.stem,status,kwargs))
            def interrupt(command,log,timeout):
                output=p/'campaign'/campaign.identifier('mlp',17);output.mkdir()
                (output/'progress.json').write_text(json.dumps(dict(kind='mlp',seed=17,
                    status='running',requested_updates=30000,applied_updates=15,checkpoints=[])))
                raise KeyboardInterrupt('synthetic interrupt')
            args=SimpleNamespace(records=p/'records',campaign=p/'campaign',cache=p/'cache',binding=p/'binding.json')
            with patch.object(campaign,'load_record',side_effect=load),patch.object(campaign,'transition',side_effect=transition),\
                 patch.object(campaign,'verified_inputs',return_value={}),patch.object(campaign,'_run_owned',side_effect=interrupt),\
                 patch.object(campaign,'_abort_remaining'):
                with self.assertRaises(KeyboardInterrupt):campaign.supervise(args)
            terminal=next(v for eid,status,v in writes if eid==campaign.identifier('mlp',17) and status=='aborted')
            self.assertEqual(terminal['results']['applied_updates'],15)
            self.assertTrue(terminal['results']['progress_fallback'])
            self.assertTrue(terminal['artifacts'])

if __name__=='__main__':unittest.main()
