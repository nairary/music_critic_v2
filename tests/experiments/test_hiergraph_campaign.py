"""Real subprocess sequencing on synthetic records, no corpus or accelerator."""
from pathlib import Path
import sys,tempfile,json
import unittest
from unittest.mock import patch
from music_critic.experiments.hierarchical_graph_analyzer import campaign
from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes
from test_registry import _record

WORKER='''from pathlib import Path
import sys,time
from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes
record,mode,previous=sys.argv[1:]
if previous!='none':assert load_record(previous)['status'] in {'completed','failed','aborted'}
if mode=='fail':raise SystemExit(3)
if mode=='sleep':time.sleep(60)
r=load_record(record);r['status']='completed';r['results']={'summary':'synthetic completed','actual_state':{'applied_updates':80000,'attempted_steps':80003}}
Path(record).write_bytes(canonical_json_bytes(seal_record(r))+b'\\n')
'''

class CampaignTests(unittest.TestCase):
    def fixture(self,path,modes):
        root=path/'campaign';root.mkdir();(path/'source-test').mkdir();worker=path/'worker.py';worker.write_text(WORKER)
        parent=root/'parent.json';parent.write_bytes(canonical_json_bytes(seal_record(_record(experiment_id='parent',status='planned')))+b'\n')
        conditions=[]
        for i,mode in enumerate(modes):
            rid='child'+str(i);p=root/(rid+'.json');p.write_bytes(canonical_json_bytes(seal_record(_record(experiment_id=rid,status='planned')))+b'\n')
            conditions.append(dict(kind='E'+str(i+1),experiment_id=rid,seed=17,record=p.name,output='run'+str(i),mode=mode))
        plan=dict(campaign_record='parent.json',campaign_wall_seconds=5,condition_wall_seconds=.02,worker_grace_seconds=.1,source_directory='source-test',environment={'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src'),'PYTHONDONTWRITEBYTECODE':'1'},poll_seconds=.01,conditions=conditions)
        def command(root,plan,row):
            i=conditions.index(row);previous=str(root/conditions[i-1]['record']) if i else 'none'
            return [sys.executable,str(worker),str(root/row['record']),row['mode'],previous]
        return root,plan,command
    def test_failed_worker_finalized_before_next_and_no_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root,plan,command=self.fixture(Path(temp),['fail','ok'])
            plan['condition_wall_seconds']=1
            with patch.object(campaign,'command_for',side_effect=command):result=campaign.run_campaign(root,plan)
            self.assertEqual([r['status'] for r in result['conditions']],['failed','completed'])
            self.assertEqual(result['status'],'completed_with_failures')
            self.assertEqual(load_record(root/'child0.json')['status'],'failed')
            self.assertEqual(load_record(root/'child1.json')['results']['actual_state']['applied_updates'],80000)
            with self.assertRaises(FileExistsError):campaign.run_campaign(root,plan)
    def test_timeout_worker_group_stopped_next_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root,plan,command=self.fixture(Path(temp),['sleep','ok'])
            with patch.object(campaign,'command_for',side_effect=command):result=campaign.run_campaign(root,plan)
            self.assertEqual(result['conditions'][0]['status'],'aborted')
            self.assertEqual(result['conditions'][1]['status'],'completed')
    def test_gpu_lock_rejects_second_holder(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'gpu.lock'
            with campaign.gpu_lock(path):
                with self.assertRaisesRegex(RuntimeError,'GPU lock'):
                    with campaign.gpu_lock(path):pass
    def test_interrupted_record_retains_confirmed_partial_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root,plan,_=self.fixture(Path(temp),['ok']);row=plan['conditions'][0];out=root/row['output'];out.mkdir()
            (out/'progress.json').write_text(json.dumps(dict(state=dict(attempted_steps=100,applied_updates=87))))
            record=campaign.finalize_interrupted(root,row,'synthetic timeout','aborted')
            self.assertEqual(record['results']['actual_state']['applied_updates'],87)
            self.assertEqual(record['budget']['actual']['attempted_steps'],100)
            original=(root/row['record']).read_bytes();campaign.finalize_interrupted(root,row,'again','failed')
            self.assertEqual((root/row['record']).read_bytes(),original)
