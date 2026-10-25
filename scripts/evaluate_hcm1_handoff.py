"""Registered native migration gate and fixed development diagnostic.

The frozen package is unchanged. Gate updates are disposable, and evaluation
always reloads the authenticated terminal parent rather than gate weights.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

from scripts.hcm1_stage2 import load_parent,continue_full_state
from scripts.hcm1_diagnostic_data import read_bound,query_from_json,paired_examples


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def tree_equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):return torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,dict):return left.keys()==right.keys() and all(tree_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(tree_equal(a,b) for a,b in zip(left,right))
    return left==right


def parent(args):
    trainer,sampler=load_parent(args.parent_config,args.manifest,args.parent_checkpoint,args.parent_sha256,
        args.parent_binding,args.parent_record,device=args.device)
    if getattr(args,'segment_checkpoint',None):
        from music_critic.experiments.registry import load_record
        from scripts.hcm1_completed_segment import restore_segment
        trainer,state=restore_segment(trainer,args.segment_checkpoint,args.segment_binding,
            load_record(args.segment_record),parent_sha256=args.parent_sha256)
        sampler.load_state_dict(state)
        if sampler.draws!=trainer.draws:raise ValueError('segment sampler progress mismatch')
    return trainer,sampler


def gate(args,output,stop,evidence):
    import torch
    from music_critic.cuda_memory import initialize_cuda_memory_statistics
    from music_critic.pll.cloud import _compare_tensors
    from music_critic.pll.runtime import _cpu_tree
    lifecycle=initialize_cuda_memory_statistics(torch.device(args.device))
    device_index=lifecycle.logical_device_index
    model_parent,sampler=parent(args);evidence['sampler']=sampler
    rng=model_parent._rng_state();optimizer=_cpu_tree(model_parent.optimizer.state_dict())
    state={k:p.detach().cpu().clone() for k,p in model_parent.model.state_dict().items()}
    child=continue_full_state(model_parent,additional_updates=10000,protocol_id='HCM1-3090-STAGE2-001',
        parent_sha256=args.parent_sha256,implementation_sha256=digest(Path(__file__).with_name('hcm1_stage2.py')))
    if not (tree_equal(state,child.model.state_dict()) and tree_equal(optimizer,child.optimizer.state_dict())
        and tree_equal(rng,child._rng_state()) and (child.updates,child.draws)==(model_parent.updates,model_parent.draws)):
        raise ValueError('full-state transition mismatch')
    del model_parent,state,optimizer
    initial=output/'gate-initial.pt';initial_sha=child.save_checkpoint(initial,sampler_state=sampler.state_dict())
    sampler.load_state_dict(child.load_checkpoint(initial,expected_sha256=initial_sha))
    times=[];receipts=[]
    def step():
        if stop():raise InterruptedError('gate stopped')
        samples=tuple(sampler.draw() for _ in range(child.config.accumulation))
        receipts.append([s.receipt() for s in samples]);start=time.monotonic()
        metrics=child.step(tuple(s.example for s in samples));torch.cuda.synchronize(device_index)
        times.append(time.monotonic()-start)
        (output/f'step-{len(times)}.json').write_text(json.dumps(dict(metrics=metrics,draws=receipts[-1]),sort_keys=True))
        return metrics
    first=step()
    checkpoint=output/'gate-one-step.pt';checksum=child.save_checkpoint(checkpoint,sampler_state=sampler.state_dict())
    expected_metrics=step();expected={k:p.detach().cpu().clone() for k,p in child.model.state_dict().items()}
    expected_optimizer=_cpu_tree(child.optimizer.state_dict());expected_rng=child._rng_state()
    sampler.load_state_dict(child.load_checkpoint(checkpoint,expected_sha256=checksum));replayed=step()
    if receipts[1]!=receipts[2]:raise ValueError('sampler replay mismatch')
    difference=_compare_tensors(expected,child.model.state_dict(),atol=1e-6,rtol=1e-5,label='migration_resume')
    if not math.isclose(expected_metrics['block_nll'],replayed['block_nll'],rel_tol=1e-5,abs_tol=1e-4):raise ValueError('loss replay mismatch')
    actual=child.optimizer.state_dict();moment_difference=0.
    if expected_optimizer['param_groups']!=actual['param_groups']:raise ValueError('optimizer groups replay mismatch')
    for key,values in expected_optimizer['state'].items():
        moment_difference=max(moment_difference,_compare_tensors(values,actual['state'][key],atol=1e-6,rtol=1e-5,label='adam_replay'))
    if not tree_equal(expected_rng,child._rng_state()):raise ValueError('RNG replay mismatch')
    peak=torch.cuda.max_memory_reserved(device_index);total=torch.cuda.get_device_properties(device_index).total_memory
    if peak>.9*total:raise ValueError('native memory headroom')
    if digest(args.parent_checkpoint)!=args.parent_sha256:raise ValueError('parent checkpoint changed')
    return dict(native_admitted=True,full_state_exact=True,resume_verified=True,physical_updates=3,
        first=first,max_parameter_difference=difference,max_optimizer_difference=moment_difference,
        update_seconds=times,peak_reserved_bytes=peak,cuda_total_bytes=total,
        cuda_memory_lifecycle=lifecycle.to_dict(),gate_weights_promoted=False)


def bounded_generate(model,context,*,seed,max_tokens,wall_seconds,stop=lambda:False):
    import torch
    from music_critic.pll.codec import EOS
    started=time.monotonic();prefix=()
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    with torch.no_grad():
        if stop():raise InterruptedError('generation stopped')
        memory=model.encode(context)
        for _ in range(max_tokens):
            if stop():raise InterruptedError('generation stopped')
            if time.monotonic()-started>=wall_seconds:
                return dict(status='incomplete',reason='example_wall_budget',tokens=list(prefix),elapsed_seconds=time.monotonic()-started)
            probs=model.decoder.next_token_log_probs(prefix,memory,span=context.query.span,tracks=context.allowed_tracks)
            prefix+=(int(torch.multinomial(probs.exp(),1,generator=generator).item()),)
            if prefix[-1]==EOS:
                events=model.codec.decode(prefix,span=context.query.span,tracks=context.allowed_tracks)
                return dict(status='completed',events=[asdict(e) for e in events],tokens=list(prefix),elapsed_seconds=time.monotonic()-started)
    return dict(status='incomplete',reason='token_budget',tokens=list(prefix),elapsed_seconds=time.monotonic()-started)


def evaluate(args,output,stop,evidence):
    from music_critic.data import loads_piece
    from music_critic.pll.scoring import score_piece
    from music_critic.pll.masking import split_masked_piece
    from music_critic.experiments.registry import load_record
    case_path=Path(args.cases);case_raw=case_path.read_bytes();record=load_record(args.preparation_record)
    if record['status']!='completed' or not any(a['sha256']==sha256(case_raw).hexdigest() and a['size_bytes']==len(case_raw) for a in record['artifacts']):
        raise ValueError('cases not bound to completed preparation')
    cases=json.loads(case_raw);root=case_path.parent
    if cases['status']!='completed':raise ValueError('incomplete preparation')
    for c in cases['cases']:
        if c['split'] not in ('train','validation','campaign_head_dev'):raise ValueError('case split forbidden')
        if c['kind']=='pair' and c['source_scope'] not in ('train','external_development'):raise ValueError('authored scope forbidden')
    trainer,sampler=parent(args);evidence['sampler']=sampler;model=trainer.model.eval();rows=[];generations=[]
    def load(case,key):
        attempt=dict(case_id=case['id'],split=None,verified=False)
        evidence.setdefault('reads',[]).append(attempt)
        piece=loads_piece(read_bound(root,case[key]).decode())
        actual=(piece.split or '').lower()
        attempt['split']=actual
        if actual not in ('','train','validation','val','dev'):raise ValueError('unexpected embedded evaluation split')
        if case['split']=='validation' and actual not in ('validation','val','dev'):raise ValueError('validation identity mismatch')
        attempt['verified']=True
        return piece
    with (output/'scores.jsonl').open('x') as log:
        for case in cases['cases']:
            if stop():raise InterruptedError('evaluation stopped')
            row=dict(id=case['id'],kind=case['kind'],split=case['split'],source=case['source'],work_id=case['work_id'],supported=case['supported'])
            if case['kind']=='pair':row.update(aspect=case['aspect'],genre=case['genre'])
            if not case['supported']:row['reason']=case['reason']
            else:
                original=load(case,'original');q=query_from_json(case['query'])
                if case['kind']=='pair':
                    variant=load(case,'variant');a,b,identity=paired_examples(original,variant,q,model.context_profile)
                    if identity!=case['context_sha256']:raise ValueError('prepared context identity changed')
                    row['context_sha256']=identity
                    row['original']=score_piece(model,original,(q,),allow_validation=True)['blocks'][0]
                    row['variant']=score_piece(model,variant,(q,),allow_validation=True)['blocks'][0]
                    row['supported']=row['original']['supported'] and row['variant']['supported']
                    if row['supported']:
                        row['delta_nll']=row['variant']['nll']-row['original']['nll']
                        row['delta_nats_per_event']=None if not a.targets or not b.targets else row['variant']['nats_per_event']-row['original']['nats_per_event']
                else:
                    row['original']=score_piece(model,original,(q,),allow_validation=True)['blocks'][0]
                    row['supported']=row['original']['supported']
            rows.append(row);log.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
    by_id={c['id']:c for c in cases['cases']}
    for i,identifier in enumerate([] if getattr(args,'skip_generation',False) else cases['generation_ids']):
        if stop():raise InterruptedError('evaluation stopped before generation')
        case=by_id[identifier]
        if not case['supported']:result=dict(status='unsupported',reason=case['reason'])
        else:
            piece=load(case,'original');example=split_masked_piece(piece,query_from_json(case['query']),model.context_profile)
            result=bounded_generate(model,example.context,seed=args.seed+i,max_tokens=4096,wall_seconds=180,stop=stop)
        if stop():raise InterruptedError('evaluation stopped after generation')
        result.update(id=identifier,seed=args.seed+i);generations.append(result)
        (output/f'generation-{i+1}.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    return dict(parent_update=trainer.updates,scores=rows,generations=generations,requested=len(rows),supported=sum(r['supported'] for r in rows),
        interpretation='Conditional symbolic likelihood; no human quality label or full-distribution generalization claim')


def binding(args):
    from music_critic.pll.runtime import implementation_fingerprint
    result=dict(mode=args.mode,parent_checkpoint_sha256=args.parent_sha256,implementation_sha256=implementation_fingerprint(),
        script_sha256=digest(__file__),stage2_sha256=digest(Path(__file__).with_name('hcm1_stage2.py')),
        data_helpers_sha256=digest(Path(__file__).with_name('hcm1_diagnostic_data.py')),
        cases_sha256=digest(args.cases) if args.mode=='evaluate' else None,seed=args.seed,wall_seconds=args.wall_seconds)
    if getattr(args,'segment_checkpoint',None):
        if args.mode!='evaluate' or not args.segment_binding or not args.segment_record:
            raise ValueError('segment evaluation requires checkpoint, binding and terminal record')
        result['segment']=dict(checkpoint_sha256=digest(args.segment_checkpoint),binding_sha256=digest(args.segment_binding),
            record_sha256=digest(args.segment_record),loader_sha256=digest(Path(__file__).with_name('hcm1_completed_segment.py')))
    if getattr(args,'skip_generation',False):result['skip_generation']=True
    return result


def access_observations(original,sampler,reads):
    from music_critic.experiments.registry import DATA_ACCESS_FIELDS
    unknown=sampler is None or bool(sampler.manifest.unverified_reads) or any(not r['verified'] for r in reads)
    access={**original,**{k:None if unknown else False for k in DATA_ACCESS_FIELDS}}
    if any(r['split'] in ('val','validation','dev') for r in reads):access['validation_accessed']=True
    test=any(r['split']=='test' for r in reads)
    if test:
        access.update(test_inputs_accessed=True,test_targets_accessed=True,test_metrics_computed=False,test_used_for_selection=False,
            policy_violations=['test_inputs_accessed','test_targets_accessed'])
    values=[access[k] for k in DATA_ACCESS_FIELDS]
    access['access_evidence_status']=('pending_primary_evidence' if all(v is None for v in values) else
        'verified_primary_evidence' if all(isinstance(v,bool) for v in values) else 'partial_primary_evidence')
    return access,test


def compact_results(result):
    """Generated notes/tokens and full per-example scores stay in ignored output."""
    from statistics import mean
    if 'scores' not in result:return result
    groups=[]
    for split,source in sorted({(r['split'],r['source']) for r in result['scores']}):
        rows=[r for r in result['scores'] if (r['split'],r['source'])==(split,source)]
        good=[r for r in rows if r['supported']]
        work_values={}
        for row in good:
            value=row['original']['nats_per_event']
            if value is not None:work_values.setdefault(row['work_id'],[]).append(value)
        groups.append(dict(split=split,source=source,requested=len(rows),supported=len(good),
            works=len({r['work_id'] for r in good}),nonempty_work_macro_nats_per_event=mean(mean(v) for v in work_values.values()) if work_values else None))
    pairs=[r for r in result['scores'] if r['kind']=='pair' and r['supported']]
    return dict(parent_update=result['parent_update'],requested=result['requested'],supported=result['supported'],groups=groups,
        pair_sign_counts=dict(original=sum(r['delta_nll']>0 for r in pairs),edited=sum(r['delta_nll']<0 for r in pairs),equal=sum(r['delta_nll']==0 for r in pairs)),
        generations=[dict(id=g['id'],status=g['status'],reason=g.get('reason'),events=len(g.get('events',[])),tokens=len(g.get('tokens',[]))) for g in result['generations']],
        interpretation=result['interpretation'])


def run(args):
    import torch
    from music_critic.experiments.registry import load_record,DATA_ACCESS_FIELDS
    from music_critic.pll.cli import _admit_output,_record_lock,_write_new,_artifact,_transition,_now
    from music_critic.pll.cloud import stop_signals
    _admit_output(args.output)
    if args.wall_seconds<=0:raise ValueError('positive wall budget required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rp=Path(args.record);out=Path(args.output);ledger=rp.parents[2]/'EXPERIMENT_LEDGER.md'
    with _record_lock(rp):
        record=load_record(rp)
        if record['status']!='planned' or record.get('hcm_diagnostic')!=binding(args):raise ValueError('planned diagnostic binding')
        policy=record['data_access']['declared_policy']
        if any(policy[k] for k in ('test_inputs_allowed','test_targets_allowed','test_metrics_allowed','test_selection_allowed')):
            raise ValueError('TEST must remain closed')
        if bool(policy['validation_allowed'])!=(args.mode=='evaluate'):raise ValueError('validation policy mismatch')
        out.mkdir(parents=True,exist_ok=False);_write_new(out/'planned-record.json',record)
        record=_transition(rp,record,dict(record,status='running',execution=dict(started_at=_now(),device=args.device,
            torch=str(torch.__version__),gpu=torch.cuda.get_device_name())),ledger)
        started=time.monotonic();result={};error=None;status='failed';evidence={}
        try:
            with stop_signals() as requested:
                stop=lambda:requested() or time.monotonic()-started>=args.wall_seconds
                result=(gate if args.mode=='gate' else evaluate)(args,out,stop,evidence)
                status='completed'
        except InterruptedError as exc:status='aborted';error=dict(type=type(exc).__name__,message=str(exc))
        except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
        finally:
            sampler=evidence.pop('sampler',None);reads=evidence.get('reads',[])
            access,test=access_observations(record['data_access'],sampler,reads)
            if test:status='invalid'
            receipt=dict(status=status,error=error,results=result,access=evidence,data_access=access,
                binding=binding(args),elapsed_seconds=time.monotonic()-started)
            _write_new(out/'receipt.json',receipt)
            after=dict(record,status=status,results=compact_results(result),data_access=access,
                execution={**record['execution'],'ended_at':_now()},validity={'reason':None if status=='completed' else str(error or status)},
                budget={**record['budget'],'actual_wall_seconds':receipt['elapsed_seconds']},
                artifacts=record['artifacts']+[_artifact(p,rp,p.name) for p in sorted(out.iterdir()) if p.is_file()])
            _transition(rp,record,after,ledger)
        if status!='completed':raise RuntimeError(f'{status}: {error}')
        return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('gate','evaluate'))
    for name in ('parent-config','manifest','parent-checkpoint','parent-sha256','parent-binding','parent-record','record','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--cases');p.add_argument('--preparation-record');p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=20260928);p.add_argument('--wall-seconds',type=int,default=3600)
    p.add_argument('--segment-checkpoint');p.add_argument('--segment-binding');p.add_argument('--segment-record')
    p.add_argument('--skip-generation',action='store_true')
    run(p.parse_args())
