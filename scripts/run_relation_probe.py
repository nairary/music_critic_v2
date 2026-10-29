"""Owned-process, time-bounded pilot supervisor; no new paid resources or LLM calls."""
from __future__ import annotations
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from hashlib import sha256
from pathlib import Path
from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes


def save_record(path,record):
    path=Path(path);record=seal_record(record)
    temp=path.with_suffix('.json.tmp');temp.write_bytes(canonical_json_bytes(record)+b'\n');temp.replace(path)


def stop_owned(proc):
    if proc is None or proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:proc.wait();return
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def close_record(path,status,result,artifact_root,reason):
    from music_critic.relation_probe.runtime import binding
    r=load_record(path)
    if r['status'] in ('completed','failed','invalid','aborted'):raise ValueError('terminal record immutable')
    r['status']=status;r['results']=result;r['validity']={'valid':status=='completed','reason':reason}
    r['budget']['actual_wall_seconds']=result.get('wall_seconds')
    r['budget']['actual']={'applied_updates':result.get('applied_updates')}
    r['data_access'].update(access_evidence_status='verified_primary_evidence',validation_accessed=False,
        test_inputs_accessed=False,test_targets_accessed=False,test_metrics_computed=False,test_used_for_selection=False,
        observation_scope='Only hashed prepared artifacts from named global TRAIN admission; downstream-dev internal to TRAIN.')
    artifacts=[]
    for p in sorted(Path(artifact_root).rglob('*')):
        if p.is_file() and p.suffix in ('.json','.jsonl','.pt'):
            b=binding(p,str(p.relative_to(Path(artifact_root).parent)))
            artifacts.append(dict(role='relation_probe_evidence',locator=b['path'],sha256=b['sha256'],size_bytes=b['size_bytes'],availability='external'))
    r['artifacts']=artifacts
    save_record(path,r)


def single(args):
    from music_critic.relation_probe.runtime import train_run,dump_json
    config=json.loads(Path(args.config).read_text());r=load_record(args.record)
    if r['status']!='planned':raise ValueError('fresh planned run required')
    r['status']='running';save_record(args.record,r)
    root=Path(config['workspace']);spec=r['protocol']['condition'];out=root/'runs'/r['experiment_id']
    started=time.monotonic()
    try:
        result=train_run(root/'cache001',out,spec['mode'],spec['decoder'],spec['seed'],config['checkpoint'],
                         steps=config['updates'],max_seconds=config['run_seconds'],device='cuda')
        close_record(args.record,result['status'],result,out,result['reason'])
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True)
        result=dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        dump_json(out/'failure.json',result);close_record(args.record,'failed',result,out,'training_exception');raise


def gate(config):
    from music_critic.relation_probe.runtime import build_cache,TileStore,device_item,dump_json
    from music_critic.relation_probe.model import Probe,masked_pair_loss
    import torch
    root=Path(config['workspace']);record=root/'records'/config['gate_record'];r=load_record(record)
    if r['status']!='planned':raise ValueError('fresh gate required')
    r['status']='running';save_record(record,r);out=root/'gate001';out.mkdir(exist_ok=False);start=time.monotonic()
    try:
        cache=build_cache(root/'dataset',root/'cache001')
        torch.set_num_threads(4)
        store=TileStore(root/'cache001');first=next(row for row in cache['tiles'] if row['split']=='fit')
        batch=device_item(store.get(first),'cuda')
        details=[]
        for mode in ('random','finetune'):
            model=Probe(mode,'transe',17,config['checkpoint']).cuda().train();optimizer=torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),lr=1e-4)
            torch.cuda.reset_peak_memory_stats();ts=time.monotonic()
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                h=model.features(batch['graph'],batch['events'],batch['boundary'])
                score,presence=model.head(h,batch['source'],batch['target'],batch['exist_source'])
                loss=masked_pair_loss(score,batch['labels'],batch['mask'])+masked_pair_loss(
                    presence,batch['exist_y'],torch.ones_like(batch['exist_y'],dtype=torch.bool))
                if not torch.isfinite(loss):raise ValueError('gate nonfinite loss')
                loss.backward();optimizer.step()
            details.append(dict(mode=mode,seconds_per_update=(time.monotonic()-ts)/4,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),finite=True))
            del model,optimizer;torch.cuda.empty_cache()
        counts=cache['counts']
        if any(counts.get(k,0)==0 for k in ('fit_positive','fit_none','dev_positive','dev_none')):
            raise ValueError('both classes required in fit/dev')
        result=dict(wall_seconds=time.monotonic()-start,counts=counts,tiles=len(cache['tiles']),
                    exclusions=cache['exclusions'],discarded_updates=8,engineering=details,
                    cache_manifest_sha256=sha256((root/'cache001/manifest.json').read_bytes()).hexdigest())
        dump_json(out/'result.json',result);close_record(record,'completed',result,out,'bounded_native_gate_passed')
        return result
    except BaseException as error:
        result=dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start)
        dump_json(out/'failure.json',result);close_record(record,'failed',result,out,'gate_exception');raise


def campaign(args):
    from music_critic.relation_probe.runtime import dump_json
    config=json.loads(Path(args.config).read_text());root=Path(config['workspace'])
    root.mkdir(exist_ok=True);state=root/'campaign.json'
    if state.exists():raise ValueError('campaign already has state; refusing duplicate launch')
    started=time.time();deadline=started+config['campaign_seconds'];proc=None;active_record=None
    def interrupted(signum,frame):raise InterruptedError(f'supervisor_signal_{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    progress=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        deadline_epoch=deadline,completed=[],total=len(config['training_records']),phase='gate',pid=os.getpid())
    dump_json(state,progress)
    try:
        # Gate is a child so the wall cap also bounds preprocessing/native admission.
        command=[sys.executable,__file__,'gate','--config',args.config]
        with (root/'gate.stdout.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(config['gate_seconds'],max(1,deadline-time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError('gate wall cap')
        if code:raise RuntimeError('native gate failed; training not started')
        for name in config['training_records']:
            if deadline-time.time()<60:progress['status']='aborted';progress['reason']='campaign_wall_cap';break
            record=root/'records'/name;r=load_record(record)
            active_record=record
            r['protocol']['cache_manifest_sha256']=sha256((root/'cache001/manifest.json').read_bytes()).hexdigest()
            save_record(record,r)
            progress.update(phase='training',current=r['experiment_id']);dump_json(state,progress)
            command=[sys.executable,__file__,'single','--config',args.config,'--record',str(record)]
            with (root/(r['experiment_id']+'.stdout.log')).open('x') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                progress['child_pid']=proc.pid;dump_json(state,progress)
                try:code=proc.wait(timeout=min(config['run_seconds']+120,max(1,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    out=root/'runs'/r['experiment_id']
                    close_record(record,'aborted',dict(reason='supervisor_wall_cap'),out,'supervisor_wall_cap')
                    code=0
            terminal=load_record(record)
            progress['completed'].append(dict(id=r['experiment_id'],status=terminal['status']))
            dump_json(state,progress)
            if code or terminal['status']=='failed':raise RuntimeError('training condition failed; remaining runs not launched')
        else:progress['status']='completed'
    except BaseException as error:
        progress.update(status='failed',reason=repr(error),traceback=traceback.format_exc())
    finally:
        stop_owned(proc)
        gate_path=root/'records'/config['gate_record']
        if gate_path.exists() and load_record(gate_path)['status']=='running':
            close_record(gate_path,'aborted',dict(reason='supervisor_gate_stop'),root/'gate001','supervisor_gate_stop')
        if active_record is not None and load_record(active_record)['status']=='running':
            active=load_record(active_record)
            close_record(active_record,'aborted',dict(reason='supervisor_exit'),root/'runs'/active['experiment_id'],'supervisor_exit')
        progress.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.time()-started)
        dump_json(state,progress)


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('campaign','gate','single'))
    p.add_argument('--config',required=True);p.add_argument('--record');args=p.parse_args()
    if args.operation=='campaign':campaign(args)
    elif args.operation=='gate':gate(json.loads(Path(args.config).read_text()))
    else:single(args)


if __name__=='__main__':main()
