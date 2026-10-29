"""Bounded sequential supervisor for one explicitly authorized H campaign.

This module never retries/resumes a condition. A worker seals its own evidence;
if it is interrupted, only its planned/running record can be finalized here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime,timezone
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from music_critic.experiments.registry import load_record,seal_record,validate_record,canonical_json_bytes
from music_critic.experiments.hiergraph_export import checked_read,safe_relative,file_binding
from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json

TERMINAL={'completed','failed','aborted','invalid'}


def now():return datetime.now(timezone.utc).isoformat()


def transition(path,status,*,results=None,reason=None,actual=None,artifacts=()):
    record=load_record(path)
    if record['status'] in TERMINAL:raise ValueError('terminal record is immutable')
    if status=='running':record['started_at']=now()
    else:
        record['finished_at']=now();record['results']=results or {'summary':reason or status}
        record['validity'].update(valid=status=='completed',reason=None if status=='completed' else str(reason or status))
        record['budget']['actual']=actual or {}
        record['artifacts'].extend(artifacts)
    record['status']=status;sealed=seal_record(record);validate_record(sealed)
    tmp=Path(path).with_suffix('.tmp');tmp.write_bytes(canonical_json_bytes(sealed)+b'\n');tmp.replace(path)
    return sealed


@contextmanager
def gpu_lock(path):
    with Path(path).open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('another H campaign owns the GPU lock') from error
        try:yield handle.fileno()
        finally:pass  # close releases only this descriptor; inherited worker FD keeps lock on supervisor death


def validate_plan(root,raw,digest):
    if sha256(raw).hexdigest()!=digest:raise ValueError('campaign plan SHA mismatch')
    plan=json.loads(raw)
    if plan.get('schema')!='Hiergraph80KCampaign@1':raise ValueError('campaign schema mismatch')
    if (type(plan.get('condition_wall_seconds')) is not int or plan['condition_wall_seconds']!=21600
        or type(plan.get('campaign_wall_seconds')) is not int or plan['campaign_wall_seconds']!=111600
        or plan.get('cleanup_seconds',10)!=10 or plan.get('worker_grace_seconds',40)!=40):raise ValueError('authorized integer campaign deadlines required')
    rows=plan['conditions']
    if [r['kind'] for r in rows]!=['E1','E2','E3','E4','E5'] or any(r['seed']!=17 for r in rows):raise ValueError('authorized five conditions/seed mismatch')
    config_raw=checked_read(root,plan['config'],1024**2);config=json.loads(config_raw)
    if config.get('selection_unit')!='applied_updates' or config['selection_steps']!=[2000,5000,10000,20000,40000,60000,80000] or config['max_attempted_steps']!=160000:raise ValueError('exact80k optimizer protocol required')
    if config['condition_wall_seconds']!=plan['condition_wall_seconds']:raise ValueError('worker deadline mismatch')
    if plan['campaign_wall_seconds']<len(rows)*(plan['condition_wall_seconds']+20):raise ValueError('campaign cap insufficient for declared queue')
    source=(Path(root).parent/plan['source_directory']).resolve()
    if source.parent!=Path(root).resolve().parent or not source.name.startswith('source'):raise ValueError('source must be a sibling frozen snapshot')
    manifest_raw=(source/'source-manifest.json').read_bytes()
    if sha256(manifest_raw).hexdigest()!=config['source_manifest_sha256']:raise ValueError('frozen source SHA mismatch')
    for binding in json.loads(manifest_raw)['files']:checked_read(source,binding,8*1024**2)
    parent=load_record(safe_relative(root,plan['campaign_record']))
    if parent['status']!='planned' or parent['protocol']['plan_sha256']!=digest:raise ValueError('fresh bound campaign record required')
    if parent['budget']['planned']['total_wall_seconds']<plan['campaign_wall_seconds']:raise ValueError('campaign cap exceeds registered budget')
    for row in rows:
        record=load_record(safe_relative(root,row['record']))
        if (record['experiment_id']!=row['experiment_id'] or record['protocol']['kind']!=row['kind'] or record['seeds']!=[17]
            or record['budget']['planned']['total_wall_seconds']<plan['condition_wall_seconds']):raise ValueError('child identity/seed/budget mismatch')
        launch=record['protocol']['launch']
        expected=dict(mode='train',config_sha256=plan['config']['sha256'],source_manifest_sha256=config['source_manifest_sha256'],target_manifest_sha256=config['target_manifest_sha256'],cache_manifest_sha256=config['cache_manifest_sha256'],output_name=Path(row['output']).name,wall_seconds=config['condition_wall_seconds'],kind=row['kind'],seed=17)
        if record['status']!='planned' or launch!=expected:raise ValueError('child record differs from frozen plan')
        if safe_relative(root,row['output']).exists():raise FileExistsError('fresh child output required')
    return plan


def command_for(root,plan,row):
    return ['timeout','--signal=TERM','--kill-after=10s',str(plan['condition_wall_seconds']+20)+'s',
            plan['python'],'scripts/run_hiergraph.py','train','--config',str(Path(root)/plan['config']['path']),
            '--record',str(Path(root)/row['record']),'--output',str(Path(root)/row['output']),
            '--kind',row['kind'],'--seed',str(row['seed'])]


def terminate(process,grace_seconds=10):
    """Reap the leader and stop its entire session group, including orphans."""
    def alive():
        process.poll()
        proc=Path('/proc')
        if proc.is_dir():
            # Linux zombies retain a process-group number but no GPU resources.
            for entry in proc.iterdir():
                if not entry.name.isdigit():continue
                try:
                    stat=(entry/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
                    if int(fields[2])==process.pid and fields[0] not in ('Z','X'):return True
                except (OSError,ValueError,IndexError):continue
            return False
        listing=subprocess.run(['ps','-axo','pid=,pgid=,stat='],capture_output=True,text=True,check=True)
        for line in listing.stdout.splitlines():
            fields=line.split()
            if len(fields)>=3 and int(fields[1])==process.pid and not fields[2].startswith('Z'):return True
        return False
    if not alive():return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    end=time.monotonic()+grace_seconds
    while alive() and time.monotonic()<end:time.sleep(.05)
    if alive():
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        end=time.monotonic()+2
        while alive() and time.monotonic()<end:time.sleep(.05)
        if alive():raise RuntimeError('worker process group remains live after SIGKILL')
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill();process.wait(timeout=10)


def finalize_interrupted(root,row,reason,status):
    path=Path(root)/row['record'];record=load_record(path)
    if record['status'] in TERMINAL:return record
    output=Path(root)/row['output'];partial={};artifacts=[]
    for name in ('result.json','progress.json'):
        p=output/name
        if p.is_file():
            d=json.loads(p.read_text());partial.update(d)
            b=file_binding(p,root);artifacts.append(dict(role='interrupted_'+name,sha256=b['sha256'],size_bytes=b['size_bytes'],availability='retained_remote',locator=b['path']))
    state=partial.get('actual_state') or partial.get('state')
    result=dict(summary=reason,error=reason,actual_state=state,
                update_accounting='last retained completed step; an interrupted in-flight update is unknown')
    actual=dict(applied_updates=None if state is None else state.get('applied_updates'),attempted_steps=None if state is None else state.get('attempted_steps'))
    return transition(path,status,results=result,reason=reason,actual=actual,artifacts=artifacts)


def run_campaign(root,plan):
    root=Path(root).resolve();parent=root/plan['campaign_record'];started=time.monotonic();process=None;active=None
    progress=dict(schema='Hiergraph80KCampaignProgress@1',status='running',started_at=now(),active=None,conditions=[],applied_target_per_condition=80000)
    def flush():
        progress['wall_seconds']=time.monotonic()-started;atomic_json(root/'progress.json',progress)
    with gpu_lock(root.parent/'hiergraph-gpu.lock') as lock_fd:
        if (root/'progress.json').exists():raise FileExistsError('campaign already has progress; no implicit resume')
        transition(parent,'running');flush()
        try:
            for row in plan['conditions']:
                if time.monotonic()-started>=plan['campaign_wall_seconds']:raise TimeoutError('campaign deadline')
                active=row;entry=dict(experiment_id=row['experiment_id'],kind=row['kind'],seed=row['seed'],status='starting',started_at=now());progress['conditions'].append(entry)
                command=command_for(root,plan,row);entry['command']=command;begin=time.monotonic()
                log_path=root/(row['kind']+'.log')
                source=root.parent/plan['source_directory']
                with log_path.open('xb') as log:
                    process=subprocess.Popen(command,cwd=source,env=dict(os.environ,**plan['environment']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
                entry.update(status='running',pid=process.pid);progress['active']=dict(experiment_id=row['experiment_id'],pid=process.pid,output=row['output']);flush()
                forced=False
                while process.poll() is None:
                    if time.monotonic()-begin>plan['condition_wall_seconds']+plan.get('worker_grace_seconds',40) or time.monotonic()-started>=plan['campaign_wall_seconds']:
                        terminate(process,plan.get('cleanup_seconds',10));forced=True;break
                    time.sleep(plan.get('poll_seconds',2))
                exit_code=process.wait();terminate(process,plan.get('cleanup_seconds',10));record=load_record(root/row['record'])
                if record['status'] not in TERMINAL:
                    record=finalize_interrupted(root,row,'worker timeout/termination' if forced or exit_code in (124,137) else 'worker exited without terminal record', 'aborted' if forced or exit_code in (124,137) else 'failed')
                result=record['results'];state=result.get('actual_state') or {}
                entry.update(status=record['status'],exit_code=exit_code,finished_at=now(),wall_seconds=time.monotonic()-begin,applied_updates=state.get('applied_updates'),attempted_steps=state.get('attempted_steps'),selected=result.get('selected'))
                if record['status']=='completed' and (exit_code!=0 or state.get('applied_updates')!=80000):
                    entry.update(status='invalid',validation_error='completed worker did not meet exact80k/exit0 contract')
                process=None;active=None;progress['active']=None;flush()
            failures=[r for r in progress['conditions'] if r['status']!='completed']
            progress.update(status='completed_with_failures' if failures else 'completed',finished_at=now());flush()
            b=file_binding(root/'progress.json',root)
            transition(parent,'completed',results=dict(summary='All five conditions processed; inspect child statuses.',campaign_status=progress['status'],conditions=progress['conditions']),actual=dict(allocated_worker_wall_seconds=sum(r['wall_seconds'] for r in progress['conditions']),wall_seconds=progress['wall_seconds']),artifacts=[dict(role='campaign_progress',sha256=b['sha256'],size_bytes=b['size_bytes'],availability='retained_remote',locator=b['path'])])
        except BaseException as error:
            if process is not None:terminate(process,plan.get('cleanup_seconds',10))
            if active is not None:finalize_interrupted(root,active,str(error),'aborted')
            progress.update(status='aborted',error=dict(type=type(error).__name__,reason=str(error)),active=None,finished_at=now());flush()
            if load_record(parent)['status'] not in TERMINAL:transition(parent,'aborted',results=dict(summary='Campaign interrupted; launched worker finalized, pending conditions not started.',progress=progress),reason=str(error),actual=dict(wall_seconds=progress['wall_seconds']))
            raise
    return progress


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args(argv);root=Path(args.campaign).resolve();path=root/'campaign.json'
    if path.is_symlink() or path.stat().st_size>1024**2:raise ValueError('bounded regular campaign plan required')
    plan=validate_plan(root,path.read_bytes(),args.plan_sha256)
    def stop(signum,frame):raise KeyboardInterrupt('supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    run_campaign(root,plan)
