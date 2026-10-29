import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

scripts=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('repair_campaign',scripts/'run_hmember_repair_campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
from music_critic.experiments.harmonic_event_reasoner.evidence import plan,binding
from music_critic.experiments.registry import load_record


class RepairCampaignTests(unittest.TestCase):
    def test_fixed_conditions_and_worker_authority(self):
        self.assertEqual(len({campaign.identifier(*v)for v in campaign.CONDITIONS}),4)
        cmd=campaign.worker_command('script','cache','out','authority','graph','balanced')
        self.assertIn('run',cmd);self.assertEqual(cmd[cmd.index('--authority')+1],'authority')
        with self.assertRaises(ValueError):campaign.identifier('self','balanced')

    def test_weights_must_come_from_bound_mass_and_source(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'manifest.json').write_text('{}');(p/'source.tar.gz').write_bytes(b'synthetic')
            a=dict(seed=17,steps=30000,conditions=[list(v)for v in campaign.CONDITIONS],
                deterministic=True,threshold_policy='exact',negative_mass=.1,class_weights=[5.,1/1.8],
                cache_manifest=binding(p/'manifest.json'),source_archive=binding(p/'source.tar.gz'))
            f=p/'authority.json';f.write_text(json.dumps(a));campaign.verify_authority(f,p)
            for change in ({'class_weights':[1.,1.]},{'steps':80000},{'negative_mass':True}):
                f.write_text(json.dumps(dict(a,**change)))
                with self.subTest(change=change),self.assertRaises(ValueError):campaign.verify_authority(f,p)
            f.write_text(json.dumps(a));(p/'source.tar.gz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):campaign.verify_authority(f,p)

    def test_first_failed_run_terminalizes_unlaunched_conditions(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);records=p/'records';records.mkdir();authority=p/'authority';authority.write_text('{}')
            for kind,loss in campaign.CONDITIONS:
                eid=campaign.identifier(kind,loss)
                plan(records/(eid+'.json'),eid,'engineering_gate',{'fixture':True,'authority':binding(authority)},{'commit':'fixture'},seeds=[17])
            args=SimpleNamespace(records=records,campaign=p/'campaign',cache=p/'cache',authority=p/'authority')
            receipt=dict(returncode=1,leader_reaped=True,timed_out=False)
            with patch.object(campaign,'verify_authority',return_value={}),patch.object(campaign,'_run_owned',return_value=receipt)as run:
                with self.assertRaisesRegex(RuntimeError,'condition failed'):campaign.supervise(args)
            self.assertEqual(run.call_count,1)
            states=[load_record(records/(campaign.identifier(*v)+'.json'))['status']for v in campaign.CONDITIONS]
            self.assertEqual(states,['failed','aborted','aborted','aborted'])

    def test_missing_record_authority_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); records=p/'records'; records.mkdir()
            for condition in campaign.CONDITIONS:
                eid=campaign.identifier(*condition)
                plan(records/(eid+'.json'),eid,'engineering_gate',{'fixture':True},{'commit':'fixture'},seeds=[17])
            args=SimpleNamespace(records=records,campaign=p/'campaign',cache=p/'cache',authority=p/'authority')
            with patch.object(campaign,'verify_authority',return_value={}),patch.object(campaign,'_run_owned') as launch:
                with self.assertRaisesRegex(ValueError,'authority binding'):
                    campaign.supervise(args)
            launch.assert_not_called()
            self.assertFalse((p/'campaign').exists())

    def test_unsupported_raw_auc_is_missing(self):
        self.assertIsNone(campaign.raw_primary_auc([]))
        self.assertIsNone(campaign.raw_primary_auc([dict(work='a',logits=[1.,2.],y=[1.,1.],mask=[True,True])]))
        rows=[dict(work='a',logits=[0.,1.],y=[0.,1.],mask=[True,True]),
              dict(work='b',logits=[1.,0.],y=[0.,1.],mask=[True,True])]
        self.assertEqual(campaign.raw_primary_auc(rows),.5)


if __name__=='__main__':unittest.main()
