#!/usr/bin/env python3
"""Four preregistered fixed-budget loss controls; no adaptive run selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from run_hmember_campaign import _run_owned, _claim_campaign, _verify_bytes, _abort_remaining
from run_hmember_curve_campaign import retained_artifacts
from music_critic.experiments.registry import load_record
from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json, transition, artifact, stamp
from music_critic.experiments.harmonic_event_reasoner.metrics import choose_threshold_fpr, score_metrics

CONDITIONS=(('mlp','plain'),('mlp','balanced'),('graph','plain'),('graph','balanced'))


def identifier(kind, loss):
    if (kind,loss) not in CONDITIONS:raise ValueError('unsupported repair condition')
    return f'EXP-HMEMBER-REPAIR-S17-{kind.upper()}-{loss.upper()}-001'


def worker_command(script, cache, output, authority, kind, loss):
    identifier(kind,loss)
    return [sys.executable,str(script),'run','--cache',str(cache),'--output',str(output),
            '--authority',str(authority),'--kind',kind,'--loss',loss]


def verify_authority(path, cache):
    path=Path(path).resolve();a=json.loads(path.read_text())
    if (a.get('seed')!=17 or a.get('steps')!=30000 or a.get('conditions')!=[list(v)for v in CONDITIONS]
            or a.get('deterministic') is not True or a.get('threshold_policy')!='exact'):
        raise ValueError('repair authority must bind the fixed four-condition protocol')
    pi=a['negative_mass']
    if not isinstance(pi,(int,float)) or isinstance(pi,bool) or not 0<pi<1:
        raise ValueError('supported TRAIN class mass required')
    expected=[1/(2*pi),1/(2*(1-pi))]
    if a['class_weights']!=expected:raise ValueError('class weights differ from TRAIN mass')
    _verify_bytes(Path(cache)/'manifest.json',a['cache_manifest'])
    _verify_bytes(path.parent/a['source_archive']['path'],a['source_archive'])
    return a


def raw_primary_auc(rows):
    from music_critic.experiments.harmonic_event_reasoner.metrics import _auc
    import math
    works={}
    for row in rows:
        auc=_auc([v for v,m in zip(row['logits'],row['mask'])if m],
                 [v for v,m in zip(row['y'],row['mask'])if m])
        if auc is not None:works.setdefault(row['work'],[]).append(auc)
    return math.fsum(math.fsum(v)/len(v)for v in works.values())/len(works) if works else None


def run(args):
    identifier(args.kind,args.loss)
    from music_critic.experiments.harmonic_event_reasoner.training import train_run
    a=verify_authority(args.authority,args.cache)
    result=train_run(args.cache,args.output,args.kind,17,30000,max_seconds=540,device='cuda',
        deterministic=True,class_weights=a['class_weights']if args.loss=='balanced'else None,
        threshold_policy='exact',retain_logits=True)
    rows=[json.loads(line)for line in (Path(args.output)/'predictions.jsonl').read_text().splitlines()]
    selection=[r for r in rows if r['split']=='selection'];dev=[r for r in rows if r['split']=='dev']
    # Both operating points were specified before the campaign; no DEV selection.
    try:threshold=choose_threshold_fpr(selection,max_fpr=.1);error=None
    except ValueError as exc:threshold=None;error=str(exc)
    operating=dict(fpr10_threshold=threshold,fpr10_infeasible=error,
        selection=score_metrics(selection,threshold)if threshold is not None else None,
        dev=score_metrics(dev,threshold)if threshold is not None else None)
    operating['raw_logit_primary_auc']={split:raw_primary_auc(part) for split,part in [('selection',selection),('dev',dev)]}
    atomic_json(Path(args.output)/'operating-points.json',operating)
    return result


def supervise(args):
    a=verify_authority(args.authority,args.cache)
    records=Path(args.records);ids=[identifier(*x)for x in CONDITIONS]
    for eid in ids:
        record=load_record(records/(eid+'.json'))
        if record['status']!='planned':raise ValueError('all repair runs must be new planned attempts')
        if 'authority' not in record['protocol']:
            raise ValueError('planned repair record requires authority binding')
        _verify_bytes(args.authority,record['protocol']['authority'])
    root=_claim_campaign(args.campaign);started=time.monotonic();active=None
    state=dict(status='running',completed=0,total=4,started_at=stamp(),active=None,
               cap_seconds=2400,per_condition_seconds=600,worker_seconds=540)
    atomic_json(root/'campaign.json',state);atomic_json(root/'progress.json',state)
    previous=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('termination')))
    try:
        for kind,loss in CONDITIONS:
            if 2400-(time.monotonic()-started)<610:raise TimeoutError('insufficient campaign budget')
            active=identifier(kind,loss);output=root/active;record=records/(active+'.json')
            transition(record,'running');state['active']=active;atomic_json(root/'progress.json',state)
            receipt=_run_owned(worker_command(Path(__file__).resolve(),args.cache,output,args.authority,kind,loss),
                               root/(active+'.log'),600)
            result=json.loads((output/'result.json').read_text())if(output/'result.json').exists()else{}
            complete=(receipt['returncode']==0 and receipt['leader_reaped']and result.get('status')=='completed'
                      and result.get('applied_updates')==30000 and result.get('exposure_sha256')==a['exposure_sha256']
                      and result.get('known_pair_exposure')==a['known_pair_exposure']
                      and result.get('runtime',{}).get('deterministic_algorithms')is True
                      and(output/'operating-points.json').exists())
            status='completed'if complete else('aborted'if receipt['timed_out']else'failed')
            artifacts=retained_artifacts(output,root,result)
            if(output/'operating-points.json').exists():artifacts.append(artifact(output/'operating-points.json',root,'operating_points'))
            compact={k:v for k,v in result.items()if k not in ['controls','initial','final','selection_curve']}
            compact['final']={split:{k:v for k,v in m.items()if k!='per_work'}for split,m in result.get('final',{}).items()}
            transition(record,status,results=compact,reason=None if complete else result.get('error','incomplete repair condition'),
                artifacts=artifacts,actual=receipt,observed_train_only='cache_manifest'in result)
            active=None
            if not complete:raise RuntimeError('repair condition failed; no follow-up runs')
            state['completed']+=1;state['wall_seconds']=time.monotonic()-started
        state.update(status='completed',active=None,finished_at=stamp(),wall_seconds=time.monotonic()-started)
    except BaseException as error:
        if active and load_record(records/(active+'.json'))['status']in('planned','running'):
            transition(records/(active+'.json'),'aborted',reason=str(error))
        _abort_remaining(records,ids,str(error))
        state.update(status='aborted'if isinstance(error,(KeyboardInterrupt,TimeoutError))else'failed',
                     error=str(error),active=None,wall_seconds=time.monotonic()-started)
        raise
    finally:
        atomic_json(root/'campaign.json',state);atomic_json(root/'progress.json',state)
        signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','supervise'])
    for key in ('cache','authority'):p.add_argument('--'+key,required=True)
    for key in ('output','kind','loss','records','campaign'):p.add_argument('--'+key)
    args=p.parse_args();run(args)if args.mode=='run'else supervise(args)
