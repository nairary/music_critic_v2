#!/usr/bin/env python3
"""Fresh, bounded duration comparison on the existing frozen-C3 cache."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from run_hmember_campaign import (_claim_campaign, _run_owned, _verify_bytes,
    _condition_result, _compact_result, _abort_remaining)
from music_critic.experiments.harmonic_event_reasoner.evidence import (
    atomic_json, artifact, stamp, transition,
)
from music_critic.experiments.registry import load_record

KINDS=('mlp','transe','self','graph')
SEEDS=(17,42,73)
UPDATES=(2000,5000,10000,20000,30000)
CAMPAIGN_SECONDS=14400
CONDITION_SECONDS=900
WORKER_SECONDS=890


def profile_settings(profile):
    if profile=='30k':
        return dict(prefix='LONG',updates=UPDATES,condition_seconds=CONDITION_SECONDS,worker_seconds=WORKER_SECONDS)
    if profile=='80k':
        return dict(prefix='80K',updates=UPDATES+(40000,60000,80000),condition_seconds=1100,worker_seconds=1090)
    raise ValueError('unsupported duration profile')


def identifier(kind,seed,profile='30k'):
    return f"EXP-HMEMBER-{profile_settings(profile)['prefix']}-S{seed}-{kind.upper()}-001"


def verified_inputs(cache,binding_path):
    path=Path(binding_path).resolve();data=json.loads(path.read_text())
    _verify_bytes(Path(cache)/'manifest.json',data['cache_manifest'])
    _verify_bytes(path.parent/data['source_archive']['path'],data['source_archive'])
    return data


def compact_result(result):
    value=_compact_result(result)
    for key in ('selected_update','selected_checkpoint','selection_policy','unique_train_h'):
        if key in result:value[key]=result[key]
    value['curve']=[dict(update=p['update'],primary_auc=p['selection']['primary_auc'],
        macro_f1=p['selection']['global']['macro_f1'],fpr=p['selection']['global']['fpr'],
        threshold=p['threshold'],unique_train_h=p['unique_train_h'],
        known_pair_exposure=p['known_pair_exposure'],exposure_sha256=p['exposure_sha256'])
        for p in result.get('selection_curve',[])]
    return value


def worker_command(script,cache,output,kind,seed,profile='30k'):
    profile_settings(profile)
    return [sys.executable,str(script),'run','--cache',str(cache),'--output',str(output),
        '--profile',profile,'--kind',kind,'--seed',str(seed)]


def retained_artifacts(output,campaign,result):
    artifacts=[]
    for name,role in [('result.json','result'),('progress.json','progress'),
                      ('selection-curve.json','selection_curve'),('predictions.jsonl','selected_predictions')]:
        p=output/name
        if p.exists():artifacts.append(artifact(p,campaign,role))
    for checkpoint in result.get('checkpoints',[]):
        path=(output/checkpoint['path']).resolve()
        if not path.is_relative_to(output.resolve()):raise ValueError('unsafe checkpoint locator')
        _verify_bytes(path,checkpoint);artifacts.append(artifact(path,campaign,'head_checkpoint'))
    return artifacts


def supervise(args):
    profile=getattr(args,'profile','30k');settings=profile_settings(profile)
    updates=settings['updates'];condition_seconds=settings['condition_seconds'];worker_seconds=settings['worker_seconds']
    records=Path(args.records).resolve();ids=[identifier(k,s,profile) for s in SEEDS for k in KINDS]
    for eid in ids:
        if load_record(records/(eid+'.json'))['status']!='planned':
            raise ValueError('all duration records must be fresh planned attempts')
    authority=verified_inputs(args.cache,args.binding)
    if profile=='80k' and (authority.get('profile')!='80k' or 'prefix_30k' not in authority):
        raise ValueError('80k binding must declare profile and prior30k schedule fingerprints')
    campaign=_claim_campaign(args.campaign);started=time.monotonic()
    state=dict(status='running',phase='training',completed=0,total=12,active=None,
        started_at=stamp(),profile=profile,updates=list(updates),selection='highest inner-selection AUROC; earliest tie',
        caps=dict(campaign=CAMPAIGN_SECONDS,condition=condition_seconds,worker=worker_seconds,cleanup=10))
    atomic_json(campaign/'campaign.json',state);atomic_json(campaign/'progress.json',state)
    active=None;previous=signal.getsignal(signal.SIGTERM)
    def interrupted(*_):raise KeyboardInterrupt('duration supervisor termination')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for seed in SEEDS:
            for kind in KINDS:
                if CAMPAIGN_SECONDS-(time.monotonic()-started)<condition_seconds+10:
                    raise TimeoutError('insufficient remaining wall budget for full duration condition')
                active=identifier(kind,seed,profile);output=campaign/active
                transition(records/(active+'.json'),'running')
                state.update(active=active,wall_seconds=time.monotonic()-started)
                atomic_json(campaign/'progress.json',state)
                command=worker_command(Path(__file__).resolve(),Path(args.cache).resolve(),output,kind,seed,profile)
                receipt=_run_owned(command,campaign/(active+'.log'),condition_seconds)
                result=_condition_result(output,kind,seed,updates[-1],receipt['timed_out'])
                complete=(receipt['returncode']==0 and receipt['leader_reaped']
                    and result.get('status')=='completed' and result.get('applied_updates')==updates[-1]
                    and tuple(p['update'] for p in result.get('selection_curve',[]))==updates
                    and result.get('selected_update') in updates)
                if complete:
                    expected=authority['prefix_schedule'][str(seed)]
                    first=result['selection_curve'][0]
                    if (first['exposure_sha256']!=expected['exposure_sha256']
                        or first['known_pair_exposure']!=expected['known_pair_exposure']):
                        complete=False;result['error']='2k sampler prefix differs from completed short pilot'
                    if complete and profile=='80k':
                        expected=authority['prefix_30k'][str(seed)]
                        point=next(p for p in result['selection_curve'] if p['update']==30000)
                        if (point['exposure_sha256']!=expected['exposure_sha256']
                            or point['known_pair_exposure']!=expected['known_pair_exposure']):
                            complete=False;result['error']='30k sampler prefix differs from completed duration pilot'
                status='completed' if complete else ('aborted' if receipt['timed_out'] or result.get('status')=='aborted' else 'failed')
                artifacts=retained_artifacts(output,campaign,result)
                transition(records/(active+'.json'),status,results=compact_result(result),actual=receipt,
                    reason=None if complete else result.get('error','incomplete duration condition'),
                    artifacts=artifacts,observed_train_only='cache_manifest' in result)
                active=None
                if not complete:raise RuntimeError('duration condition failed; stop before more launches')
                state['completed']+=1
        state.update(status='completed',phase='completed',active=None,finished_at=stamp(),wall_seconds=time.monotonic()-started)
        atomic_json(campaign/'campaign.json',state);atomic_json(campaign/'progress.json',state)
        return 0
    except BaseException as error:
        reason=f'{type(error).__name__}: {error}'
        if active:
            record=load_record(records/(active+'.json'))
            if record['status'] in ('planned','running'):
                try:
                    saved=_condition_result(campaign/active,kind,seed,updates[-1],True)
                    retained=retained_artifacts(campaign/active,campaign,saved)
                    evidence=compact_result(saved)
                except Exception as salvage_error:
                    evidence=dict(evidence_salvage_error=f'{type(salvage_error).__name__}: {salvage_error}')
                    retained=[]
                evidence['interrupted']=True
                transition(records/(active+'.json'),'aborted',reason=reason,results=evidence,artifacts=retained)
        _abort_remaining(records,ids,reason)
        state.update(status='aborted' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',
            phase='terminal',active=None,error=reason,finished_at=stamp(),wall_seconds=time.monotonic()-started)
        atomic_json(campaign/'campaign.json',state);atomic_json(campaign/'progress.json',state)
        raise
    finally:signal.signal(signal.SIGTERM,previous)


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    parent=commands.add_parser('supervise')
    for key in ('campaign','cache','records','binding'):parent.add_argument('--'+key,required=True)
    parent.add_argument('--profile',choices=('30k','80k'),default='30k')
    worker=commands.add_parser('run')
    for key in ('cache','output'):worker.add_argument('--'+key,required=True)
    worker.add_argument('--kind',choices=KINDS,required=True);worker.add_argument('--seed',type=int,required=True)
    worker.add_argument('--profile',choices=('30k','80k'),default='30k')
    args=parser.parse_args()
    if args.command=='supervise':return supervise(args)
    import torch
    from music_critic.experiments.harmonic_event_reasoner.training import train_run
    torch.set_num_threads(1)
    settings=profile_settings(args.profile)
    train_run(args.cache,args.output,args.kind,args.seed,settings['updates'][-1],max_seconds=settings['worker_seconds'],
        selection_updates=settings['updates'])
    return 0


if __name__=='__main__':raise SystemExit(main())
