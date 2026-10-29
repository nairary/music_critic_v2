#!/usr/bin/env python3
"""Explicit bounded preparation/training CLI; each real command needs a planned record."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import signal
import time


def verify_source(root,binding_path,expected):
    path=Path(binding_path);raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected:raise ValueError('source manifest SHA mismatch')
    doc=json.loads(raw)
    from music_critic.experiments.hiergraph_export import checked_read
    for row in doc['files']:checked_read(root,row,8*1024**2)
    return expected


def run_train(config,output,kind,seed):
    import torch
    from music_critic.experiments.hierarchical_graph_analyzer.pipeline import open_inputs,Samples,make_model
    from music_critic.experiments.hierarchical_graph_analyzer.runtime import train_steps,evaluate,save_checkpoint,load_checkpoint
    from music_critic.experiments.hierarchical_graph_analyzer.metrics import aggregate_metrics
    from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json
    torch.set_num_threads(1);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    ds,cache=open_inputs(config);fit=Samples(ds,cache,'fit');selection=Samples(ds,cache,'selection')
    model=make_model(kind,ds.head_sizes,config,seed);optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    config_sha=sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    bindings=dict(source_sha256=config['source_manifest_sha256'],config_sha256=config_sha,
                  data_sha256=config['target_manifest_sha256'],cache_sha256=config['cache_manifest_sha256'])
    unit=config.get('selection_unit','attempted_steps')
    if unit not in ('attempted_steps','applied_updates'):raise ValueError('unsupported selection unit')
    milestones=config['selection_steps']
    if (not milestones or any(type(v) is not int or v<=0 for v in milestones)
        or milestones!=sorted(set(milestones))):raise ValueError('strictly increasing positive selection steps required')
    draw_cap=config.get('max_attempted_steps',milestones[-1] if unit=='attempted_steps' else 2*milestones[-1])
    if type(draw_cap) is not int or draw_cap<milestones[-1]:raise ValueError('attempted-step cap must cover requested updates')
    state=None;curve=[];best=None
    def retain_step(current,row):
        nonlocal state
        state=current
        with (out/'exposure.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
        atomic_json(out/'progress.json',dict(status='running',kind=kind,seed=seed,state=state,wall_seconds=time.monotonic()-started))
    try:
        for target_step in config['selection_steps']:
            current=0 if state is None else state[unit]
            while current<target_step:
                attempted=0 if state is None else state['attempted_steps']
                if attempted>=draw_cap:raise RuntimeError('attempted draw cap reached before requested optimizer updates')
                if time.monotonic()-started>config['condition_wall_seconds']:raise TimeoutError('condition wall cap')
                run=train_steps(model,fit,optimizer,steps=min(10,target_step-current,draw_cap-attempted),seed=seed,state=state,clip_grad_norm=config['clip_grad_norm'],on_step=retain_step)
                state=run['state'];current=state[unit]
            report=evaluate(model,selection,split='selection',**config['evaluation'])
            report['reports']+=selection.missing_reports();report['aggregate']=aggregate_metrics(report['reports'])
            report['primary']=report['aggregate']['family_macro_known_h_recovery']
            atomic_json(out/f'selection-{target_step:06d}.json',report)
            checkpoint=save_checkpoint(out/f'checkpoint-{target_step:06d}.pt',model,optimizer,state,bindings=bindings)
            checkpoint['path']=Path(checkpoint['path']).name
            row=dict(selection_step=target_step,selection_unit=unit,attempted_steps=state['attempted_steps'],applied_updates=state['applied_updates'],primary=report['primary'],checkpoint=checkpoint,exposure_sha256=state['exposure_sha256'])
            curve.append(row);atomic_json(out/'selection-curve.json',dict(rows=curve))
            if report['primary'] is not None and (best is None or report['primary']>best['primary']):best=row
        if best is None:raise ValueError('no supported inner-selection primary score')
        load_checkpoint(out/best['checkpoint']['path'],model,optimizer,bindings=bindings,expected_sha256=best['checkpoint']['sha256'])
        dev=Samples(ds,cache,'dev')
        # One final checkpoint. Oracle interpretation is an explicitly separate view.
        final=evaluate(model,dev,split='dev',allow_dev=True,**config['evaluation'])
        final['reports']+=dev.missing_reports();final['aggregate']=aggregate_metrics(final['reports'])
        final['primary']=final['aggregate']['family_macro_known_h_recovery']
        atomic_json(out/'dev-selected.json',final)
        oracle=evaluate(model,dev,split='dev',allow_dev=True,oracle=True,**config['evaluation'])
        atomic_json(out/'dev-oracle-H.json',oracle)
        result=dict(status='completed',kind=kind,seed=seed,actual_state=state,selected=best,selection_unit=unit,requested_updates=milestones[-1] if unit=='applied_updates' else None,
                    dev=final['aggregate'],oracle=oracle['aggregate'],wall_seconds=time.monotonic()-started,
                    fit_coverage=dict(fit.coverage),selection_coverage=dict(selection.coverage),dev_coverage=dict(dev.coverage),
                    selection_rule='earliest maximum inner-selection family-macro known-H recovery',test_locked=True)
    except BaseException as error:
        result=dict(status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed',kind=kind,seed=seed,
                    actual_state=state,selected=best,wall_seconds=time.monotonic()-started,error=dict(type=type(error).__name__,reason=str(error)))
        atomic_json(out/'result.json',result);raise
    atomic_json(out/'result.json',result);return result


def admit_launch(config_raw,config,record,mode,output,kind,seed):
    launch=record.get('protocol',{}).get('launch')
    if not isinstance(launch,dict):raise ValueError('planned record needs exact launch binding')
    cap_key={'cache':'cache_wall_seconds','gate':'gate_wall_seconds','train':'condition_wall_seconds'}[mode]
    expected=dict(mode=mode,config_sha256=sha256(config_raw).hexdigest(),
        source_manifest_sha256=config['source_manifest_sha256'],
        target_manifest_sha256=config['target_manifest_sha256'],output_name=Path(output).name,
        wall_seconds=config[cap_key])
    if mode!='cache':expected['cache_manifest_sha256']=config['cache_manifest_sha256']
    if mode=='train':expected.update(kind=kind,seed=seed)
    if launch!=expected:raise ValueError('actual launch differs from frozen planned record')
    if type(config[cap_key]) is not int or not 0<config[cap_key]<=record['budget']['planned']['total_wall_seconds']:
        raise ValueError('wall deadline exceeds registered budget')
    if Path(output).exists():raise FileExistsError('fresh output required before launch')
    return launch


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cache','gate','train'])
    p.add_argument('--config',required=True);p.add_argument('--record',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['E1','E2','E3','E4','E5']);p.add_argument('--seed',type=int,default=17)
    args=p.parse_args(argv);config_raw=Path(args.config).read_bytes();config=json.loads(config_raw)
    root=Path(__file__).resolve().parents[1]
    from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes,validate_record
    from music_critic.experiments.hiergraph_export import file_binding
    record_path=Path(args.record);record=load_record(record_path)
    if record['status']!='planned':raise ValueError('fresh planned experiment record required')
    if args.mode=='train' and (not args.kind or record.get('protocol',{}).get('kind')!=args.kind or args.seed not in record['seeds']):raise ValueError('training condition differs from planned record')
    admit_launch(config_raw,config,record,args.mode,args.output,args.kind,args.seed)
    verify_source(root,config['source_manifest'],config['source_manifest_sha256'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    def transition(status,result=None):
        current=load_record(record_path)
        if current['status'] not in ('planned','running'):raise ValueError('terminal record is immutable')
        current['status']=status
        if status=='running':current['started_at']=datetime.now(timezone.utc).isoformat()
        else:
            current['finished_at']=datetime.now(timezone.utc).isoformat();current['results']=result or {}
            current['validity']['valid']=status=='completed';current['validity']['reason']=None if status=='completed' else (result or {}).get('error','execution failed')
            current['budget']['actual']=dict(wall_seconds=(result or {}).get('wall_seconds'),applied_updates=((result or {}).get('actual_state') or {}).get('applied_updates',0),attempted_steps=((result or {}).get('actual_state') or {}).get('attempted_steps',0))
            current['data_access'].update(access_evidence_status='verified_primary_evidence',validation_accessed=False,test_inputs_accessed=False,test_targets_accessed=False,test_metrics_computed=False,test_used_for_selection=False)
            evidence=Path(args.output)/('manifest.json' if args.mode=='cache' else 'result.json')
            if evidence.is_file():
                b=file_binding(evidence);current['artifacts'].append(dict(role=evidence.name,sha256=b['sha256'],size_bytes=b['size_bytes'],availability='retained_remote',locator=current.get('protocol',{}).get('output_root',Path(args.output).name)+'/'+b['path']))
        if status in ('completed','failed','aborted') and args.mode=='train':
            for name in ('selection-curve.json','progress.json'):
                p=Path(args.output)/name
                if p.is_file():
                    b=file_binding(p);current['artifacts'].append(dict(role=name,sha256=b['sha256'],size_bytes=b['size_bytes'],availability='retained_remote',locator=current.get('protocol',{}).get('output_root',Path(args.output).name)+'/'+b['path']))
        sealed=seal_record(current);validate_record(sealed,filename=record_path.name)
        raw=canonical_json_bytes(sealed)+b'\n';tmp=record_path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(record_path)
    def deadline(signum,frame):raise TimeoutError('outer condition wall deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(int(config['cache_wall_seconds'] if args.mode=='cache' else config['gate_wall_seconds'] if args.mode=='gate' else config['condition_wall_seconds']))
    transition('running');started=time.monotonic()
    try:
        if args.mode=='cache':
            from music_critic.experiments.hierarchical_graph_analyzer.data import load_target_dataset
            from music_critic.experiments.hierarchical_graph_analyzer.cache import build_raw_cache
            ds=load_target_dataset(config['targets'],manifest_sha256=config['target_manifest_sha256'])
            result=build_raw_cache(ds,config['source_root'],args.output,config['checkpoint'],config['checkpoint_sha256'],config['checkpoint_size_bytes'],device=config['device'],wall_seconds=config['cache_wall_seconds'])
            summary=dict(status=result['status'],works=len(result['works']),tiles=sum(len(w['tiles']) for w in result['works']),exclusions=sum(len(w['exclusions']) for w in result['works']),wall_seconds=result['wall_seconds'],summary='Raw C3 preparation completed; no training updates.')
        elif args.mode=='gate':
            from music_critic.experiments.hierarchical_graph_analyzer.pipeline import native_gate
            result=native_gate(config,args.output);summary=result
        else:result=run_train(config,args.output,args.kind,args.seed);summary=result
        signal.alarm(0)
        transition('completed',summary);print(json.dumps(summary,sort_keys=True))
    except BaseException as error:
        signal.alarm(0)
        partial={}
        partial_path=Path(args.output)/'result.json'
        if partial_path.is_file():partial=json.loads(partial_path.read_text())
        partial.update(error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started,summary='Preparation/training did not complete.')
        transition('aborted' if isinstance(error,KeyboardInterrupt) else 'failed',partial)
        raise

if __name__=='__main__':main()
