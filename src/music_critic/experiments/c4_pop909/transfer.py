"""Registered, frozen C4 -> Dilemmadata VALIDATION with POP-compatible metrics."""
from __future__ import annotations
from collections import Counter
from io import BytesIO
from hashlib import sha256
from pathlib import Path
import json
import time
import torch
from music_critic.critic.c3_backbone import C3ThreePassRelationGateModel, validate_c3_source_model
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.frozen_training_data import FrozenTrainingData, read_bound_bytes
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from music_critic.experiments.analysisgnn.temporal_slices.cache import read_temporal_record
from music_critic.experiments.pop909_cl_frozen.projection import quality_subset_ids
from music_critic.experiments.registry import load_record
from .training import CHECKPOINT_SCHEMA, HEADS, class_count, pc_class_mapping, pc_log_probabilities, forward_validated, write_json, configure_runtime

SCHEMA='C4DilemmaTransfer@1.0.0'


def implementation_fingerprint():
    root=Path(__file__).resolve().parents[4]
    files={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest()
           for folder in ('src/music_critic','scripts') for p in sorted((root/folder).rglob('*.py'))}
    return fingerprint({'schema':SCHEMA,'files':files})


def require(condition, reason):
    if not condition:
        raise ValueError('c4_transfer.'+reason)


def validate_request(request, training_record):
    require(request.get('split')=='validation', 'validation_only')
    require(training_record.get('status')=='completed', 'training_not_completed')


def load_frozen_c4(binding, training_record, device):
    """Authenticate the selected C4 envelope; restore weights only, never RNG/Adam."""
    read_bound_bytes(training_record)
    parent=load_record(training_record['path'])
    validate_request({'split':'validation'},parent)
    best=parent['results']['producer_report']['best_checkpoint']
    require(best['sha256']==binding['sha256'] and best['size_bytes']==binding['size_bytes'], 'selected_checkpoint_binding')
    payload=torch.load(BytesIO(read_bound_bytes(binding)),map_location='cpu',weights_only=True)
    require(payload.get('schema_version')==CHECKPOINT_SCHEMA,'checkpoint_schema')
    require(payload.get('state_fingerprint')==recursive_state_fingerprint({k:v for k,v in payload.items() if k!='state_fingerprint'}),'checkpoint_digest')
    require(payload['progress']['applied_updates']==best['update'],'checkpoint_update')
    for key,field in (('source','source_fingerprint'),('manifest','manifest_fingerprint'),('config','c4_config_fingerprint')):
        require(payload['bindings'][key]==parent['protocol'][field],'checkpoint_'+key)
    state=payload['model_state']
    require(all(torch.isfinite(v).all() for v in state.values()),'nonfinite_checkpoint')
    require(model_state_fingerprint(state)==payload['model_state_fingerprint'],'model_digest')
    with torch.random.fork_rng(devices=[]):
        model=C3ThreePassRelationGateModel()
    validate_c3_source_model(model)
    model.load_state_dict(state,strict=True)
    return model.to(device).requires_grad_(False).eval(),best


def aligned_events(output, sidecar):
    """Keep native multiplicity; join only source-declared existing raw beats."""
    beats={}
    for row in sidecar['relations']:
        if row.get('relation')=='harmonic_event_to_beat':
            src,dst=row['source_entity_id'],row['target_entity_id']
            require(src not in beats or beats[src]==dst,'ambiguous_beat_relation')
            beats[src]=dst
    lookup={h:{key:i for i,key in enumerate(output.candidate_keys[h])} for h in HEADS if h in output.logits}
    for h in lookup:
        require(len(lookup[h])==len(output.candidate_keys[h]),'duplicate_candidate')
    predictions={h:(pc_log_probabilities(output.logits[h],h) if h in ('root','bass') else output.logits[h]).argmax(-1).cpu().tolist() for h in lookup}
    events=[];seen=set()
    for entity in sidecar['entities']:
        if entity['entity_type']!='harmonic_event':
            continue
        eid=entity['canonical_entity_id'];require(eid not in seen,'duplicate_event');seen.add(eid)
        row={'event_id':eid,'beat_id':beats.get(eid),'targets':{}}
        for h in lookup:
            state=entity.get('targets',{}).get(h,{})
            label=state.get('canonical_value');labels=get_vocabulary(TASK_BY_ID[h].vocabulary_id).labels
            candidate=lookup[h].get((0,'beat',beats.get(eid)))
            reason=('source_unavailable' if state.get('available') is not True or state.get('masked') is not False
                    else 'unknown_target' if label not in labels else 'unaligned' if candidate is None else 'aligned')
            gold=labels.index(label) if reason=='aligned' else None
            if gold is not None and h in ('root','bass'):
                gold=pc_class_mapping(h)[gold]
            row['targets'][h]={'gold':gold,'predicted':predictions[h][candidate] if candidate is not None else None,'reason':reason,
                'source_available':state.get('available') is True and state.get('masked') is False,
                'canonical_known':label in labels,'candidate_aligned':candidate is not None}
        events.append(row)
    return events


def summarize(matrix, *, allowed_gold=None):
    matrix=matrix.clone();excluded=0
    if allowed_gold is not None:
        outside=[i for i in range(matrix.shape[0]) if i not in allowed_gold]
        excluded=int(matrix[outside].sum());matrix[outside]=0
    support=matrix.sum(1);predicted=matrix.sum(0);diagonal=matrix.diag();observed=support>0
    f1=2.*diagonal.double()/(support+predicted).clamp_min(1).double()
    return {'eligible_events':int(support.sum()),'excluded_gold_events':excluded,
        'accuracy':float(diagonal.sum()/support.sum()) if bool(support.sum()) else None,
        'observed_true_class_macro_f1':float(f1[observed].mean()) if bool(observed.any()) else None,
        'observed_class_ids':observed.nonzero().flatten().tolist(),'per_class_f1':f1.tolist(),
        'class_support':support.tolist(),'confusion':matrix.tolist()}


def evaluate(request, *, experiment_record, output):
    record=load_record(experiment_record)
    require(record['status']=='planned' and record['kind']=='evaluation','registered_plan_required')
    require(record['protocol']['request_fingerprint']==fingerprint(request),'request_record_binding')
    policy=record['data_access']['declared_policy']
    require(policy.get('validation_allowed') is True and all(policy.get(k) is False for k in
        ('test_inputs_allowed','test_targets_allowed','test_metrics_allowed','test_selection_allowed')),'access_policy')
    require(request['source_fingerprint']==implementation_fingerprint(),'implementation_binding')
    read_bound_bytes(request['training_record'])
    parent=load_record(request['training_record']['path']);validate_request(request,parent)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();device=request['device'];env=configure_runtime(device)
    access={'validation_accessed':False,'test_inputs_accessed':False,'test_targets_accessed':False,
            'test_metrics_computed':False,'test_used_for_selection':False,'train_optimizer_updates':0,
            'validation_source_reads':0,'validation_target_reads':0,'validation_model_forwards':0}
    report={'schema_version':SCHEMA,'experiment_id':record['experiment_id'],'status':'failed',
        'source_fingerprint':request['source_fingerprint'],'access':access,'error':None}
    try:
        data=FrozenTrainingData(json.loads(read_bound_bytes(request['data_manifest'])))
        require(len(data.validation_record_ids)==162,'complete_validation_required')
        report['manifest_fingerprint']=data.manifest_fingerprint
        model,best=load_frozen_c4(request['checkpoint'],request['training_record'],device)
        initial=model_state_fingerprint(model)
        matrices={h:torch.zeros(class_count(h),class_count(h),dtype=torch.int64) for h in HEADS}
        reasons={h:Counter() for h in HEADS};coverage={h:Counter() for h in HEADS}
        joint={'all':Counter(),'quality_subset9':Counter()}
        dialects=Counter();subset=set(quality_subset_ids())
        with torch.no_grad(), (output/'events.jsonl').open('x') as stream:
            for index,rid in enumerate(data.validation_record_ids,1):
                require(time.monotonic()-started<request['wall_seconds'],'wall_bound')
                access['validation_accessed']=True
                access['validation_source_reads']+=1;access['validation_target_reads']+=1
                graph,sidecar=read_temporal_record(data,rid,split='validation')
                predicted=forward_validated(model,graph.to(device));access['validation_model_forwards']+=1
                require(all(torch.isfinite(v).all() for v in predicted.logits.values()),'nonfinite_forward')
                rows=aligned_events(predicted,sidecar);dialects[sidecar['dialect']]+=1
                for row in rows:
                    for h,cell in row['targets'].items():
                        reasons[h][cell['reason']]+=1
                        coverage[h]['source_rows']+=1
                        for axis in ('source_available','canonical_known','candidate_aligned'):
                            coverage[h][axis]+=int(cell[axis])
                        if cell['gold'] is not None:
                            matrices[h][cell['gold'],cell['predicted']]+=1
                    root,quality=row['targets']['root'],row['targets']['quality']
                    if root['gold'] is not None and quality['gold'] is not None:
                        correct=root['gold']==root['predicted'] and quality['gold']==quality['predicted']
                        for scope in ('all','quality_subset9'):
                            if scope=='all' or quality['gold'] in subset:
                                joint[scope]['eligible']+=1;joint[scope]['correct']+=int(correct)
                stream.write(json.dumps({'record_id':rid,'dialect':sidecar['dialect'],
                    'source_component_id':sidecar['source_component_id'],'target_fingerprint':sidecar['fingerprint'],
                    'events':rows},sort_keys=True)+'\n');stream.flush()
                print(json.dumps({'event':'scored','records':index,'record_id':rid,'elapsed_seconds':time.monotonic()-started}),flush=True)
                del graph,predicted,sidecar,rows
        require(model_state_fingerprint(model)==initial,'weights_changed')
        metrics={h:{**summarize(m),'availability':dict(reasons[h]),'independent_coverage':dict(coverage[h])} for h,m in matrices.items()}
        common=summarize(matrices['quality'],allowed_gold=subset)
        report.update(status='completed',metrics=metrics,quality_subset9=common,
            joint={s:{**dict(v),'accuracy':v['correct']/v['eligible'] if v['eligible'] else None} for s,v in joint.items()},
            record_count=162,dialect_counts=dict(dialects),selected_checkpoint=best,
            evaluated_model_fingerprint=initial,weights_unchanged=True,applied_updates=0,
            model_forwards=access['validation_model_forwards'],environment=env,
            candidate_policy='source_event_to_existing_raw_beat',identity_only=True)
    except BaseException as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)}
    report['wall_seconds']=time.monotonic()-started
    write_json(output/'report.json',report)
    return report
