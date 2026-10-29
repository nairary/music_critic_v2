#!/usr/bin/env python3
"""Explicit registered whole-work native gate, training, or unlabelled inference."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path,PurePosixPath
import signal
import time


def admit_launch(config_raw,config,record,mode,output,seed,*,evidence_base=None):
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import validate_full_config
    validate_full_config(config,mode)
    protocol=record.get('protocol',{});anchor=protocol.get('evidence_root');relative=protocol.get('output_root')
    if not isinstance(anchor,str) or not isinstance(relative,str):raise ValueError('portable relative evidence_root and output_root required')
    rootpart=PurePosixPath(anchor)
    if rootpart.is_absolute() or not rootpart.parts or '..' in rootpart.parts:raise ValueError('portable relative evidence_root required')
    part=PurePosixPath(relative)
    if part.is_absolute() or not part.parts or '..' in part.parts:raise ValueError('safe relative output_root required')
    base=Path(evidence_base).resolve() if evidence_base is not None else Path(__file__).resolve().parents[2]
    portable=str(rootpart/part)
    anchor=(base/anchor).resolve();actual=Path(output).resolve();declared=(anchor/relative).resolve()
    if not anchor.is_relative_to(base):raise ValueError('evidence root escapes evidence base')
    if not declared.is_relative_to(anchor) or actual!=declared:raise ValueError('actual output path differs from registered evidence root')
    repository=Path(__file__).resolve().parents[1]
    if actual.is_relative_to(repository) and not any(actual.is_relative_to(repository/p) for p in ('outputs','artifacts/experiments')):raise ValueError('repository evidence must remain in ignored output roots')
    expected=dict(mode=mode,config_sha256=sha256(config_raw).hexdigest(),
        source_manifest_sha256=config['source_manifest_sha256'],output_name=Path(output).name,output_path=portable,
        wall_seconds=config['wall_seconds'],seed=seed)
    if mode!='infer':expected.update(target_manifest_sha256=config['target_manifest_sha256'],cache_manifest_sha256=config['cache_manifest_sha256'])
    else:expected.update(input_sha256=config['input']['sha256'],model_checkpoint_sha256=config['model_checkpoint']['expected_sha256'])
    if record.get('status')!='planned' or record.get('protocol',{}).get('launch')!=expected:raise ValueError('fresh planned record with exact launch binding required')
    if seed not in record['seeds']:raise ValueError('seed outside planned record')
    if config['wall_seconds']>record['budget']['planned']['total_wall_seconds']:raise ValueError('wall cap exceeds registered budget')
    if Path(output).exists():raise FileExistsError('fresh output required')
    return expected


def verify_source(root,path,expected):
    from music_critic.experiments.hiergraph_export import checked_read
    raw=Path(path).read_bytes()
    if sha256(raw).hexdigest()!=expected:raise ValueError('source manifest SHA mismatch')
    rows=json.loads(raw)['files']
    if not rows:raise ValueError('empty source manifest')
    for row in rows:checked_read(root,row,8*1024**2)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['gate','train','infer'])
    for name in ('config','record','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--evidence-base',type=Path,help='Machine-local parent for portable evidence_root; defaults to repository parent')
    parser.add_argument('--seed',type=int,default=17);args=parser.parse_args(argv)
    raw=Path(args.config).read_bytes();config=json.loads(raw);record_path=Path(args.record)
    from music_critic.experiments.registry import load_record,seal_record,validate_record,canonical_json_bytes
    from music_critic.experiments.hiergraph_export import file_binding
    record=load_record(record_path);admit_launch(raw,config,record,args.mode,args.output,args.seed,evidence_base=args.evidence_base)
    verify_source(Path(__file__).resolve().parents[1],config['source_manifest'],config['source_manifest_sha256'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    def transition(status,result=None):
        current=load_record(record_path)
        if current['status'] not in ('planned','running'):raise ValueError('terminal experiment record is immutable')
        current['status']=status
        if status=='running':current['started_at']=datetime.now(timezone.utc).isoformat()
        else:
            current['finished_at']=datetime.now(timezone.utc).isoformat();current['results']=result or {}
            current['validity']['valid']=status=='completed';current['validity']['reason']=None if status=='completed' else str((result or {}).get('error','execution failed'))
            state=(result or {}).get('actual_state') or {}
            current['budget']['actual']=dict(wall_seconds=(result or {}).get('wall_seconds'),applied_updates=state.get('applied_updates',0),attempted_steps=state.get('attempted_steps',0),skipped_unknown=state.get('skipped_unknown',0))
            for name in ('result.json','selection-curve.json','progress.json','resolved.json','prediction.json'):
                path=Path(args.output)/name
                if path.is_file():
                    binding=file_binding(path)
                    current['artifacts'].append(dict(role=name,sha256=binding['sha256'],size_bytes=binding['size_bytes'],availability='retained_local',locator=current.get('protocol',{}).get('output_root',Path(args.output).name)+'/'+name))
        sealed=seal_record(current);validate_record(sealed,filename=record_path.name)
        temporary=record_path.with_suffix('.tmp');temporary.write_bytes(canonical_json_bytes(sealed)+b'\n');temporary.replace(record_path)
    def timeout(signum,frame):raise TimeoutError('outer full-model wall deadline')
    def interrupt(signum,frame):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,interrupt)
    transition('running');started=time.monotonic();signal.alarm(config['wall_seconds'])
    try:
        if args.mode=='train':
            from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import run_full_train
            result=run_full_train(config,args.output,seed=args.seed)
        elif args.mode=='gate':
            from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import full_native_gate
            result=full_native_gate(config,args.output,seed=args.seed)
        else:
            from music_critic.experiments.hierarchical_graph_analyzer.full_inference import run_full_inference
            result=run_full_inference(config,args.output,seed=args.seed)
        signal.alarm(0);transition('completed',result);print(json.dumps(result,sort_keys=True,allow_nan=False))
    except BaseException as error:
        signal.alarm(0);partial_path=Path(args.output)/'result.json';partial=json.loads(partial_path.read_text()) if partial_path.is_file() else {}
        partial.update(error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started,summary='Full model execution did not complete.')
        transition('aborted' if isinstance(error,KeyboardInterrupt) else 'failed',partial);raise


if __name__=='__main__':main()
