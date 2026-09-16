import importlib.util
import json
from pathlib import Path
from music_critic.experiments.registry import seal_record,canonical_json_bytes,load_record


def module():
    spec=importlib.util.spec_from_file_location('mmoe_queue',Path('scripts/grouped_mmoe_campaign_queue.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_gate_then_one_training_and_stop_on_failed_gate(tmp_path):
    m=module();seen=[]
    def fail(root,python,name):seen.append(name);return False
    m.run_stage=fail
    args=['--root',str(tmp_path),'--python','fixture-python']
    assert m.main(args)==1 and seen==['gate-mmoe']
    seen.clear()
    def ok(root,python,name):seen.append(name);return True
    m.run_stage=ok
    assert m.main(args)==0 and seen==['gate-mmoe','train-mmoe']
    assert json.loads((tmp_path/'queue-completed.json').read_text())['baseline_retrained'] is False


def test_gpu_preflight_failure_is_terminal_and_preserves_planned_record(tmp_path,monkeypatch):
    m=module()
    r=json.loads(Path('docs/experiments/records/EXP-TASKATTN-GATE-001.json').read_text())
    r.update(status='planned',results={},artifacts=[],validity={'valid':None,'reason':'fixture'})
    r['data_access'].update(access_evidence_status='not_applicable_planning_record',validation_accessed=None,
        test_inputs_accessed=None,test_targets_accessed=None,test_metrics_computed=None,test_used_for_selection=None)
    r['budget']['actual']=None;r=seal_record(r)
    p=tmp_path/(r['experiment_id']+'.json');p.write_bytes(canonical_json_bytes(r)+b'\n')
    (tmp_path/'gate-mmoe-template.json').write_text(json.dumps({'record':str(p),'mode':'gate','model_kind':'mmoe'}))
    def busy(*args,**kwargs):return '4000, 90'
    monkeypatch.setattr(m.subprocess,'check_output',busy)
    assert m.run_stage(tmp_path,Path('python'),'gate-mmoe') is False
    terminal=load_record(tmp_path/'gate-mmoe'/p.name)
    assert terminal['status']=='failed' and 'gpu_busy' in terminal['results']['error']['reason']
    assert load_record(p)['status']=='planned'
