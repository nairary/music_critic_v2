"""One authorized MMoE gate then fresh80k run; no baseline retraining or polling agent."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

from music_critic.experiments.registry import seal_record, validate_record, canonical_json_bytes, load_record

GPU = 'GPU-1d7c6d9f-751f-7ea8-a951-d1b8a7991b42'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def durable_updates(out):
    count=0
    journal=out/'run/training_metrics.jsonl'
    if journal.exists():
        with journal.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'):break
                try:row=json.loads(line)
                except ValueError:break
                if row.get('update')!=count+1:break
                count+=1
    return count


def run_stage(root, python, name):
    config=json.loads((root/(name+'-template.json')).read_text())
    record=load_record(Path(config['record']))
    if record['status']!='planned' or config['model_kind']!='mmoe':
        raise ValueError('mmoe_queue.planned_mmoe_required')
    out=root/name;out.mkdir(exist_ok=False)
    started=time.monotonic();rc=None;error=None;report=None;resource=None
    try:
        if config['mode']=='train':
            gate_path=root/'gate-mmoe/run/report.json'
            gate=json.loads(gate_path.read_text())
            terminal=load_record(root/'gate-mmoe/EXP-MMOE-GATE-001.json')
            if not gate.get('passed') or gate.get('model_kind')!='mmoe' or terminal['status']!='completed':
                raise ValueError('mmoe_queue.completed_gate_required')
            if gate['initial_base_fingerprint']!=config['expected_base_fingerprint']:
                raise ValueError('mmoe_queue.base_initialization')
            config.update(gate_report=str(gate_path),gate_sha256=digest(gate_path))
        write(out/'execution-config.json',config)
        health=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        mem,util=map(int,health.split(','))
        if mem>=1024 or util!=0:raise ValueError('mmoe_queue.gpu_busy:'+health)
        wall=1800 if config['mode']=='gate' else 86400
        command=[str(python),str(root/'source/scripts/research_pilot_resource_wrapper.py'),
            '--ownership-wrapper',str(root/'source/tests/fixtures/research_supervision/accepted_context_wrapper.py'),
            '--gpu-uuid',GPU,'--wall-seconds',str(wall),'--memory-mib','16384',
            '--receipt',str(out/'resource-receipt.json'),'--stdout',str(out/'stdout.log'),
            '--stderr',str(out/'stderr.log'),'--',str(python),'-m',
            'music_critic.experiments.analysisgnn.fusion_accelerated_campaign',
            '--config',str(out/'execution-config.json'),'--output',str(out/'run')]
        write(out/'launch.json',{'command':command,'cwd':str(root/'source'),'gpu_health':health,'wall_limit_seconds':wall})
        env=dict(os.environ,PYTHONPATH=str(root/'source/src'),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES=GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (out/'supervisor.stdout.log').open('xb') as stdout,(out/'supervisor.stderr.log').open('xb') as stderr:
            rc=subprocess.run(command,cwd=root/'source',env=env,stdout=stdout,stderr=stderr).returncode
        if (out/'run/report.json').exists():report=json.loads((out/'run/report.json').read_text())
        if (out/'resource-receipt.json').exists():resource=json.loads((out/'resource-receipt.json').read_text())
    except Exception as exc:
        error={'type':type(exc).__name__,'reason':str(exc)}
    success=rc==0 and report is not None and (report.get('passed') if config['mode']=='gate' else
        report.get('completed') and report.get('start_update')==0 and report.get('final_update')==80000 and report.get('applied_updates')==80000)
    status='completed' if success else 'aborted' if resource and resource.get('process_status')=='aborted' else 'failed'
    write(out/'launch-receipt.json',{'exit_code':rc,'wall_seconds':time.monotonic()-started,'success':success,'error':error})
    actual=report.get('applied_updates') if report else durable_updates(out) if config['mode']=='train' else None
    record.update(status=status,results=report if success else {'reason':'Stage stopped; inspect retained receipts/journals','exit_code':rc,'error':error,
        'durable_updates':actual,'accounting_scope':'Journal prefix; in-flight work unknown'},
        validity={'valid':bool(success),'reason':'Execution checks passed; scientific quality remains separate' if success else 'Stage did not complete'})
    record['budget']['actual']=actual
    if success:
        record['data_access'].update(access_evidence_status='verified_primary_evidence',validation_accessed=config['mode']=='train',
            test_inputs_accessed=False,test_targets_accessed=False,test_metrics_computed=False,test_used_for_selection=False)
    else:
        record['data_access'].update(access_evidence_status='pending_primary_evidence')
    for path in sorted(out.rglob('*')):
        if path.is_file():record['artifacts'].append({'role':'stage_evidence','locator':str(path.relative_to(root)),
            'sha256':digest(path),'size_bytes':path.stat().st_size,'availability':'external_configured'})
    record=seal_record(record);validate_record(record,filename=record['experiment_id']+'.json')
    (out/(record['experiment_id']+'.json')).write_bytes(canonical_json_bytes(record)+b'\n')
    print(json.dumps({'stage':name,'status':status,'seconds':time.monotonic()-started}),flush=True)
    return success


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--python',type=Path,required=True)
    args=p.parse_args(argv)
    for name in ('gate-mmoe','train-mmoe'):
        print(json.dumps({'stage':name,'event':'starting','time':time.time()}),flush=True)
        if not run_stage(args.root,args.python,name):return 1
    write(args.root/'queue-completed.json',{'completed':True,'time':time.time(),'baseline_retrained':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
