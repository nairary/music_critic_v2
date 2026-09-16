import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
from music_critic.experiments.registry import seal_record, canonical_json_bytes, validate_record


def load(tmp_path):
    spec=importlib.util.spec_from_file_location('attention_queue',Path(__file__).parents[2]/'scripts/task_attention_campaign_queue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.R=tmp_path
    return module


def test_queue_stops_after_failed_gate(tmp_path):
    m=load(tmp_path);calls=[]
    def run(name):calls.append(name);return False
    m.run_stage=run
    assert m.main()==1 and calls==['gate-attention'] and not (tmp_path/'queue-completed.json').exists()


def test_queue_order_and_no_mmoe(tmp_path):
    m=load(tmp_path);calls=[]
    def run(name):calls.append(name);return True
    m.run_stage=run
    assert m.main()==0
    assert calls==['gate-attention','gate-control','train-attention','train-control']
    assert json.loads((tmp_path/'queue-completed.json').read_text())['MMoE_launched'] is False


def test_preflight_failure_preserves_terminal_evidence(tmp_path,monkeypatch):
    m=load(tmp_path)
    original=Path(__file__).parents[2]/'docs/experiments/records/EXP-FUSION-ACCEL-C2-GATE-002.json'
    record=json.loads(original.read_text());record.update(status='planned',results={},artifacts=[])
    record['budget']['actual']=None;record['validity']={'valid':None,'reason':'fixture'}
    record=seal_record(record);record_path=tmp_path/'planned.json';record_path.write_bytes(canonical_json_bytes(record))
    config={'record':str(record_path),'mode':'gate','model_kind':'attention'}
    (tmp_path/'gate-attention-template.json').write_text(json.dumps(config))
    def unavailable(*a,**kw):raise RuntimeError('fixture GPU unavailable')
    monkeypatch.setattr(m.subprocess,'check_output',unavailable)
    assert m.run_stage('gate-attention') is False
    terminal=json.loads((tmp_path/'gate-attention'/(record['experiment_id']+'.json')).read_text())
    validate_record(terminal,filename=record['experiment_id']+'.json')
    assert terminal['status']=='failed' and terminal['results']['error']['reason']=='fixture GPU unavailable'
    assert json.loads(record_path.read_text())['status']=='planned'
