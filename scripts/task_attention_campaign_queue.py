"""User-authorized four-stage sequential GPU queue; no agent polling."""
from pathlib import Path
import subprocess,os,sys,json,time,hashlib
R=Path('/home/humtech/music_critic_task_attention_20260914')
P=Path('/home/humtech/music_critic_research_20260913/environment/bin/python')
sys.path.insert(0,str(R/'source/src'))
from music_critic.experiments.registry import seal_record,validate_record,canonical_json_bytes
UUID='GPU-1d7c6d9f-751f-7ea8-a951-d1b8a7991b42'
env=dict(os.environ,PYTHONPATH=str(R/'source/src'),PYTHONDONTWRITEBYTECODE='1',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 CUDA_VISIBLE_DEVICES=UUID,CUBLAS_WORKSPACE_CONFIG=':4096:8')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def write(path,data):
 with path.open('x') as f:json.dump(data,f,indent=2)

def run_stage(name):
 cfg=json.loads((R/(name+'-template.json')).read_text())
 rec=json.loads(Path(cfg['record']).read_text());assert rec['status']=='planned'
 O=R/name;O.mkdir(exist_ok=False);started=time.monotonic();rc=1;report=None;error=None
 try:
  if cfg['mode']=='train':
   gate_name='gate-attention' if cfg['model_kind']=='attention' else 'gate-control'
   gate=R/gate_name/'run/report.json';g=json.loads(gate.read_text())
   assert g['passed'] and g['model_kind']==cfg['model_kind']
   other=json.loads((R/'gate-control/run/report.json').read_text())
   original=json.loads((R/'gate-attention/run/report.json').read_text())
   assert original['initial_base_fingerprint']==other['initial_base_fingerprint']
   cfg.update(gate_report=str(gate),gate_sha256=digest(gate),expected_base_fingerprint=g['initial_base_fingerprint'])
  config=O/'execution-config.json';write(config,cfg)
  health=subprocess.check_output(['nvidia-smi','--id='+UUID,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
  mem,util=map(int,health.split(','));assert mem<1024 and util==0,health
  seconds=1800 if cfg['mode']=='gate' else 86400
  cmd=[str(P),str(R/'source/scripts/research_pilot_resource_wrapper.py'),'--ownership-wrapper',str(R/'source/tests/fixtures/research_supervision/accepted_context_wrapper.py'),
   '--gpu-uuid',UUID,'--wall-seconds',str(seconds),'--memory-mib','16384','--receipt',str(O/'resource-receipt.json'),
   '--stdout',str(O/'stdout.log'),'--stderr',str(O/'stderr.log'),'--',str(P),'-m','music_critic.experiments.analysisgnn.fusion_accelerated_campaign',
   '--config',str(config),'--output',str(O/'run')]
  write(O/'launch.json',{'command':cmd,'cwd':str(R/'source'),'gpu_health':health,'stage':name,'wall_limit_seconds':seconds})
  with (O/'supervisor.stdout.log').open('xb') as out,(O/'supervisor.stderr.log').open('xb') as err:
   rc=subprocess.run(cmd,cwd=R/'source',env=env,stdout=out,stderr=err).returncode
  if (O/'run/report.json').exists():report=json.loads((O/'run/report.json').read_text())
 except Exception as exc:error={'type':type(exc).__name__,'reason':str(exc)}
 success=rc==0 and report is not None and (report.get('passed') if cfg['mode']=='gate' else report.get('completed') and report.get('final_update')==80000 and report.get('applied_updates')==80000)
 write(O/'launch-receipt.json',{'exit_code':rc,'wall_seconds':time.monotonic()-started,'success':success,'error':error})
 rec.update(status='completed' if success else 'failed',results=report if success else {'reason':'stage failed; inspect retained logs','exit_code':rc,'error':error},
  validity={'valid':bool(success),'reason':'Execution and internal checks complete; quality review remains separate' if success else 'Execution failed'})
 rec['budget']['actual']=report.get('applied_updates') if report else None
 if success:rec['data_access'].update(access_evidence_status='verified_primary_evidence',validation_accessed=cfg['mode']=='train',test_inputs_accessed=False,test_targets_accessed=False,test_metrics_computed=False,test_used_for_selection=False)
 for file in sorted(O.rglob('*')):
  if file.is_file():rec['artifacts'].append({'role':'stage_evidence','locator':str(file.relative_to(R)),'sha256':digest(file),'size_bytes':file.stat().st_size,'availability':'external_configured'})
 rec=seal_record(rec);validate_record(rec,filename=rec['experiment_id']+'.json')
 (O/(rec['experiment_id']+'.json')).write_bytes(canonical_json_bytes(rec)+b'\n')
 print(json.dumps({'stage':name,'status':rec['status'],'seconds':time.monotonic()-started}),flush=True)
 return success

def main():
 for name in ('gate-attention','gate-control','train-attention','train-control'):
  print(json.dumps({'stage':name,'event':'starting','time':time.time()}),flush=True)
  if not run_stage(name):return 1
 write(R/'queue-completed.json',{'completed':True,'time':time.time(),'MMoE_launched':False})
 return 0

if __name__=='__main__':raise SystemExit(main())
