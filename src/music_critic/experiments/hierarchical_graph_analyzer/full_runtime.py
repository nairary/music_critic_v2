"""Whole-work H/h/P execution; target metadata never enters model.forward.

This module owns deterministic work/view sampling and execution accounting.
It launches no experiment on import. Real-data CLI admission is separate.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

import torch
from music_critic.cuda_memory import initialize_cuda_memory_statistics
from .runtime import save_checkpoint, load_checkpoint


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def move_tensors(value, device):
    """Move nested raw tensor structures; retain exact identity/time sidecars."""
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move_tensors(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(move_tensors(v,device) for v in value)
    if isinstance(value,list):return [move_tensors(v,device) for v in value]
    return value


def _identity(row):
    work=row.get('work_id');view=row.get('view_id')
    if not isinstance(work,str) or not work:raise ValueError('stable work identity required')
    if view is not None and not isinstance(view,str):raise ValueError('scalar analytic view required')
    return _json([work,view]).decode()


def _inventory(samples,sample_metadata=None):
    metadata=sample_metadata if sample_metadata is not None else getattr(samples,'metadata',None)
    if metadata is None:metadata=[target for _,target in samples]
    if len(metadata)!=len(samples) or not len(metadata):raise ValueError('nonempty aligned work metadata required')
    groups=defaultdict(list);identities={};seen=set()
    for i,row in enumerate(metadata):
        if row.get('split')!='fit':raise ValueError('training schedule admits fit works only')
        ident=_identity(row)
        if ident in seen:raise ValueError('duplicate work/view identity')
        seen.add(ident);identities[i]=ident;groups[row['work_id']].append(i)
    ordered={w:sorted(indices,key=identities.__getitem__) for w,indices in sorted(groups.items())}
    return ordered,identities,sha256(_json(sorted(identities.values()))).hexdigest()


def _draw(groups,seed,cursor):
    rng=random.Random(int.from_bytes(sha256(f'full:{seed}:{cursor}'.encode()).digest(),'big'))
    return rng.choice(groups[rng.choice(list(groups))])


def full_sample_schedule(samples,*,count,seed=17,start=0,sample_metadata=None):
    if any(type(x) is not int or x<0 for x in (count,seed,start)):raise ValueError('nonnegative counts/seed required')
    groups,_,_=_inventory(samples,sample_metadata)
    return [_draw(groups,seed,cursor) for cursor in range(start,start+count)]


def _validate_state(state,seed,fingerprint):
    if type(seed) is not int or seed<0 or not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):raise ValueError('invalid seed/schedule fingerprint')
    if state.get('schema')!='HiergraphFullTrainingState@1' or state.get('seed')!=seed or state.get('schedule_fingerprint')!=fingerprint:
        raise ValueError('whole-work resume schedule/seed mismatch')
    keys=('attempted_steps','applied_updates','skipped_unknown')
    if any(type(state.get(k)) is not int or state[k]<0 for k in keys):raise ValueError('invalid training counts')
    if state['attempted_steps']!=state['applied_updates']+state['skipped_unknown']:raise ValueError('inconsistent training counts')
    digest=state.get('exposure_sha256','')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('invalid exposure hash')


def train_full_steps(model,samples,optimizer,*,steps,seed=17,state=None,weights=None,
                     sample_metadata=None,clip_grad_norm=1.,on_step=None,loss_fn=None,
                     deadline=None):
    """Run bounded attempted whole-work draws; only supported loss steps Adam.

    ``deadline`` is a monotonic absolute deadline checked before each draw.
    An outer process deadline must separately bound a single expensive forward.
    """
    if any(type(x) is not int or x<0 for x in (steps,seed)):raise ValueError('nonnegative steps/seed required')
    if clip_grad_norm is not None and (isinstance(clip_grad_norm,bool) or not isinstance(clip_grad_norm,(int,float)) or not math.isfinite(clip_grad_norm) or clip_grad_norm<=0):raise ValueError('positive gradient norm required')
    groups,identities,fingerprint=_inventory(samples,sample_metadata)
    current=deepcopy(state) if state is not None else dict(schema='HiergraphFullTrainingState@1',seed=seed,
        schedule_fingerprint=fingerprint,attempted_steps=0,applied_updates=0,skipped_unknown=0,exposure_sha256=sha256(b'').hexdigest())
    _validate_state(current,seed,fingerprint)
    if loss_fn is None:
        from .full_losses import full_loss
        loss_fn=full_loss
    parameter=next(model.parameters());history=[]
    if state is None:random.seed(seed);torch.manual_seed(seed)
    model.train()
    for _ in range(steps):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('whole-work training wall cap')
        index=_draw(groups,seed,current['attempted_steps']);raw,target=samples[index]
        if target.get('split')!='fit' or _identity(target)!=identities[index]:raise ValueError('sample reader differs from fit inventory')
        raw=move_tensors(raw,parameter.device)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(raw)
        result=loss_fn(prediction,target,weights=weights)
        value=result['loss'];supported=result.get('supported')
        if type(supported) is not bool:raise ValueError('loss must declare supported boolean')
        if not isinstance(value,torch.Tensor) or value.numel()!=1 or not bool(torch.isfinite(value)):
            raise ValueError('finite scalar loss required; no optimizer update applied')
        if supported:
            if not value.requires_grad:raise ValueError('supported loss has no gradient path')
            value.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('non-finite gradient; no optimizer update applied')
            if clip_grad_norm is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad_norm,error_if_nonfinite=True)
            optimizer.step();current['applied_updates']+=1
        else:current['skipped_unknown']+=1
        current['attempted_steps']+=1
        current['exposure_sha256']=sha256(bytes.fromhex(current['exposure_sha256'])+identities[index].encode()).hexdigest()
        row=dict(sample_identity=identities[index],loss=float(value.detach()),applied=supported,
            support=deepcopy(result.get('support',{})),components={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in result.get('components',{}).items()})
        history.append(row)
        if on_step is not None:on_step(deepcopy(current),deepcopy(row))
    return dict(state=current,history=history)


def full_checkpoint_bindings(model,bindings):
    """Bind semantic routing and non-parameter architecture, not shapes alone."""
    from dataclasses import asdict
    from .runtime import _model_signature,_bindings
    contract={'tensor_signature':_model_signature(model)}
    if hasattr(model,'constructor') and hasattr(model,'note_fields') and hasattr(model,'process_fields'):
        contract.update(config=asdict(model.config),
            H=list(zip(model.constructor.field_names,[h.out_features for h in model.constructor.field_heads])),
            h=list(zip(model.note_fields.names,[h.out_features for h in model.note_fields.heads])),
            P=list(zip(model.process_fields.names,[h.out_features for h in model.process_fields.heads])),
            relation_types=list(model.relation_types))
    digest=sha256(_json(contract)).hexdigest();result=dict(bindings)
    if 'full_model_sha256' in result and result['full_model_sha256']!=digest:raise ValueError('full model semantic/config binding mismatch')
    result['full_model_sha256']=digest
    return _bindings(result)


def save_full_checkpoint(path,model,optimizer,state,*,bindings):
    _validate_state(state,state.get('seed'),state.get('schedule_fingerprint'))
    if 'cache_sha256' not in bindings:raise ValueError('whole-work checkpoint requires cache binding')
    return save_checkpoint(path,model,optimizer,state,bindings=full_checkpoint_bindings(model,bindings))


def load_full_checkpoint(path,model,optimizer,*,bindings,expected_sha256):
    if 'cache_sha256' not in bindings or not expected_sha256:raise ValueError('whole-work checkpoint requires SHA and cache binding')
    # Validate training state before reusable loader mutates parameters/RNG.
    p=Path(path)
    if sha256(p.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('checkpoint SHA256 mismatch')
    payload=torch.load(p,map_location='cpu',weights_only=True);state=payload.get('state',{})
    _validate_state(state,state.get('seed'),state.get('schedule_fingerprint'))
    return load_checkpoint(p,model,optimizer,bindings=full_checkpoint_bindings(model,bindings),expected_sha256=expected_sha256)


def warmstart_e3(constructor,path,*,expected_sha256,expected_size_bytes):
    """Authenticated E3 weights only; optimizer and all RNG states stay untouched."""
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected_size_bytes:raise ValueError('E3 checkpoint size/type mismatch')
    if sha256(p.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('E3 checkpoint SHA mismatch')
    payload=torch.load(p,map_location='cpu',weights_only=True)
    if payload.get('schema')!='HiergraphCheckpoint@1' or payload.get('model_signature',{}).get('kind')!='E3' or getattr(constructor,'kind',None)!='E3':raise ValueError('E3 checkpoint/model kind required')
    from .runtime import _model_signature
    if payload.get('model_signature')!=_model_signature(constructor):raise ValueError('E3 checkpoint semantic field/architecture contract mismatch')
    weights=payload.get('model',{});current=constructor.state_dict()
    if set(weights)!=set(current) or any(not isinstance(weights[k],torch.Tensor) or weights[k].shape!=v.shape or weights[k].dtype!=v.dtype or not bool(torch.isfinite(weights[k]).all()) for k,v in current.items()):raise ValueError('E3 checkpoint parameter contract mismatch')
    constructor.load_state_dict(weights,strict=True)
    return dict(schema='HiergraphE3Warmstart@1',sha256=expected_sha256,size_bytes=expected_size_bytes,
        applied_updates=payload.get('state',{}).get('applied_updates'),optimizer_restored=False)


def structured_prediction(pred,raw,*,work_id,field_values=None,edge_threshold=.5):
    """Export addressed candidate graph, never hidden states or target sidecars.

    Event intervals are learned estimates represented as rationals in original
    source coordinates. Keeping every candidate event makes all retained edge
    endpoints resolvable. Existence scores are uncalibrated model scores.
    """
    if not isinstance(work_id,str) or not work_id:raise ValueError('work identity required')
    if not isinstance(edge_threshold,(int,float)) or not 0<=edge_threshold<=1:raise ValueError('edge threshold outside [0,1]')
    origin,duration=raw['origin'],raw['duration']
    if not isinstance(origin,Fraction) or not isinstance(duration,Fraction) or duration<=0:raise ValueError('exact source origin/duration required')
    values=field_values or {};note_ids=tuple(raw['note_ids'])
    def listing(v):return v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else list(v)
    def probabilities(logits):
        if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite exported score')
        return listing(logits.sigmoid())
    def span(value):
        a,b=map(float,value)
        if not(math.isfinite(a) and math.isfinite(b) and 0<=a<b<=1):raise ValueError('invalid predicted normalized span')
        return [[x.numerator,x.denominator] for x in (origin+duration*Fraction(a),origin+duration*Fraction(b))]
    def fields(group,index):
        out={}
        for name,logits in group.get('field_logits',{}).items():
            z=logits[index].detach().cpu()
            if not bool(torch.isfinite(z).all()):raise ValueError('nonfinite exported field logits')
            if z.ndim!=1 or not len(z) or not bool(torch.isfinite(z).all()):raise ValueError('invalid output field logits')
            label=int(z.argmax());row=dict(class_index=label,score=float(z.softmax(-1)[label]))
            if name in values:
                if len(values[name])!=len(z):raise ValueError('output class vocabulary mismatch')
                row['value']=deepcopy(values[name][label])
            out[name]=row
        return out
    result=dict(schema='HiergraphWholePrediction@1',work_id=work_id,
        time_semantics='learned estimates in exact source rational coordinates',
        score_semantics='uncalibrated model scores',H=[],h=[],P=[],participants=[],relations=[])
    for level in ('H','P'):
        group=pred[level];spans=listing(group['spans']);scores=probabilities(group['object_logits'])
        if len(spans)!=len(scores):raise ValueError('unaligned event scores')
        for i,(geometry,score) in enumerate(zip(spans,scores)):
            result[level].append(dict(id=f'{work_id}:{level}:{i}',span_qn=span(geometry),score=score,fields=fields(group,i)))
    group=pred['h'];spans=listing(group['spans']);notes=listing(group['note_index']);harmonies=listing(group['H_index'])
    if not len(spans)==len(notes)==len(harmonies):raise ValueError('unaligned h routing')
    for i,(geometry,n,h) in enumerate(zip(spans,notes,harmonies)):
        if type(n) is not int or not 0<=n<len(note_ids) or type(h) is not int or not 0<=h<len(result['H']):raise ValueError('invalid h endpoint')
        result['h'].append(dict(id=f'{work_id}:h:{i}',span_qn=span(geometry),note_id=note_ids[n],H_id=result['H'][h]['id'],fields=fields(group,i)))
    def endpoint(level,index):
        if type(index) is not int:raise ValueError('integer endpoint index required')
        if level in ('n','note'):
            if not 0<=index<len(note_ids):raise ValueError('unknown physical note endpoint')
            return note_ids[index]
        if level not in ('H','h','P') or not 0<=index<len(result[level]):raise ValueError('unknown graph endpoint')
        return result[level][index]['id']
    group=pred['participants'];parents=listing(group['P_index']);indices=listing(group['endpoint_index']);levels=tuple(group['endpoint_level']);scores=probabilities(group['existence_logits'])
    if not len(parents)==len(indices)==len(levels)==len(scores):raise ValueError('unaligned participant routing')
    for i,(p,l,j,score) in enumerate(zip(parents,levels,indices,scores)):
        parent=endpoint('P',p);child=endpoint(l,j)
        if score>=edge_threshold:result['participants'].append(dict(P_id=parent,endpoint=child,score=score,fields=fields(group,i)))
    group=pred['relations'];sources=group['sources'];targets=group['targets']
    if len(sources)!=len(targets):raise ValueError('unaligned relation routing')
    for name,logits in group['logits'].items():
        scores=probabilities(logits)
        if len(scores)!=len(sources):raise ValueError('unaligned typed relation scores')
        for source,target,score in zip(sources,targets,scores):
            src=[endpoint(x['kind'],x['index']) for x in source];dst=[endpoint(x['kind'],x['index']) for x in target]
            if score>=edge_threshold:result['relations'].append(dict(type=name,source=src,target=dst,score=score))
    result['coverage']={'raw':deepcopy(raw.get('coverage',{})),'model':deepcopy(pred.get('coverage',{}))}
    return result


def raw_work_from_cache(piece,cache,work_id,*,limits=None):
    """Raw-only assembly, requiring no target shard or analytical annotation."""
    from .full_data import assemble_work_raw
    from music_critic.relation_probe.data import sanitize
    if work_id not in cache.works:raise ValueError('raw work absent from cache')
    work=cache.works[work_id]
    limits=dict(limits or {})
    if set(limits)-{'max_tiles','max_notes'}:raise ValueError('unknown raw whole-work capacity limit')
    try:
        raw=assemble_work_raw(sanitize(piece),(cache.load(work_id,row) for row in work['tiles']),**limits)
    except (ValueError,RuntimeError) as error:
        raise ValueError(f'raw whole-work admission failed for {work_id}: {error}') from error
    raw['coverage']['cache_exclusions']=deepcopy(work.get('exclusions',[]))
    return raw


def _work_geometry(work):
    from .data import exact_time
    starts=[exact_time(row['metadata']['halo_start']) for row in work['tiles']]
    starts.extend(exact_time(row['halo_start']) for row in work.get('exclusions',[]) if 'halo_start' in row)
    if not starts:raise ValueError('raw work has no geometry evidence')
    duration=exact_time(work['duration'])
    if duration<=0:raise ValueError('positive whole-work duration required')
    return min(starts),duration


class WholeWorkSamples:
    """Authenticated target-view inventory and one-work raw cache.

    Numerical target shards and canonical observed sources have separate roots.
    The latter always pass sanitization before entering the graph assembler.
    Capacity failures carry the work identity and abort admission; no truncation.
    """
    def __init__(self,dataset,cache,split,*,source_root,limits=None,max_source_bytes=128*1024**2):
        from .full_data import full_head_sizes,targets_for_work
        if split not in ('fit','selection','dev'):raise ValueError('global evaluation split forbidden')
        self.dataset,self.cache,self.split=dataset,cache,split
        self.source_root=Path(source_root);self.limits=dict(limits or {});self.max_source_bytes=max_source_bytes
        self.head_sizes=full_head_sizes(dataset);self.active_fields=tuple(sorted(n for group in self.head_sizes.values() for n in group))
        self.metadata=[];self._raw_work_id=None;self._raw=None
        if set(dataset.works)!=set(cache.works):raise ValueError('raw/target whole-work inventory mismatch')
        for wid,work in sorted(dataset.works.items()):
            cached=cache.works[wid]
            if any(work[key]!=cached[key] for key in ('split','family_id')):raise ValueError('raw/target family split mismatch')
            if work['split']!=split:continue
            origin,duration=_work_geometry(cached)
            views=targets_for_work(dataset.load_work(wid),active_fields=self.active_fields,origin=origin,duration=duration)
            for target in views:self._validate_head_ranges(target)
            seen=set()
            for target in views:
                view=target['view_id']
                if view in seen:raise ValueError('duplicate whole-work analytical view')
                seen.add(view)
                self.metadata.append(dict(work_id=wid,piece_id=work['piece_id'],family_id=work['family_id'],view_id=view,split=split))
        self.coverage=dict(works=len({x['work_id'] for x in self.metadata}),views=len(self.metadata))
    def _validate_head_ranges(self,target):
        for level,sizes in self.head_sizes.items():
            for field,size in sizes.items():
                values=target[level]['fields'][field]
                if any(mask and not 0<=value<size for value,mask in zip(values['values'],values['mask'])):raise ValueError('active field ID outside fitted vocabulary: '+field)
    def __len__(self):return len(self.metadata)
    def raw_for_work(self,wid):
        from music_critic.data import loads_piece
        from music_critic.experiments.hiergraph_export import checked_read
        if self._raw_work_id!=wid:
            self._raw=None;self._raw_work_id=None
            binding=self.dataset.works[wid]['source_bindings']['canonical_piece']
            piece=loads_piece(checked_read(self.source_root,binding,self.max_source_bytes))
            if piece.piece_id!=self.dataset.works[wid]['piece_id']:raise ValueError('canonical piece identity mismatch')
            self._raw=raw_work_from_cache(piece,self.cache,wid,limits=self.limits);self._raw_work_id=wid
        return self._raw
    def __getitem__(self,index):
        from .full_data import targets_for_work
        metadata=self.metadata[index];wid=metadata['work_id'];raw=self.raw_for_work(wid)
        numeric=self.dataset.load_work(wid)
        note_ids={row['canonical_note_index']:row['id'] for row in numeric['targets'].get('participants',[]) if row.get('kind')=='note'}
        if note_ids!={i:n for i,n in enumerate(raw['note_ids'])}:raise ValueError('numeric/canonical physical note identity mismatch')
        views=targets_for_work(numeric,active_fields=self.active_fields,origin=raw['origin'],duration=raw['duration'])
        target=next((x for x in views if x['view_id']==metadata['view_id']),None)
        if target is None:raise ValueError('analytical view disappeared from target shard')
        self._validate_head_ranges(target)
        target.update(metadata);target['raw_coverage']=deepcopy(raw['coverage'])
        return raw,target


def make_full_model(head_sizes,config,seed=17):
    from .full_model import FullGraphAnalyzer,FullGraphConfig
    from .runtime import initialize_shared_parameters
    if config.get('augmentation','none')!='none':raise ValueError('full baseline augmentation is separately scoped')
    model=FullGraphAnalyzer(head_sizes['H'],head_sizes['h'],head_sizes['P'],
        relation_types=tuple(config.get('relation_types',('resolves_to','prepares'))),
        config=FullGraphConfig(**config.get('model',{})))
    initialize_shared_parameters(model,seed)
    return model.to(config.get('device','cpu'))


def evaluate_full(model,samples,*,split='selection',allow_dev=False,geometry='predicted',
                  deadline=None,metric_options=None):
    from .full_metrics import evaluate_full_piece,aggregate_full_metrics
    if split not in ('selection','dev') or (split=='dev' and not allow_dev):raise ValueError('selection only; final dev requires explicit allow_dev')
    if geometry not in ('predicted','oracle'):raise ValueError('unknown geometry diagnostic')
    if getattr(samples,'split',split)!=split:raise ValueError('sample split mismatch')
    previous=model.training;model.eval();device=next(model.parameters()).device;reports=[]
    try:
        with torch.no_grad():
            for raw,target in samples:
                if target.get('split')!=split:raise ValueError('reader returned wrong evaluation split')
                if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('whole-work evaluation wall cap')
                raw=move_tensors(raw,device)
                if geometry=='oracle':
                    if not hasattr(model,'oracle_forward'):raise ValueError('explicit oracle diagnostic unsupported by this model')
                    from .full_losses import event_supervision_targets
                    spans=[event_supervision_targets(target[level])[0]['spans'] for level in ('H','P')]
                    pred=model.oracle_forward(raw,*(torch.as_tensor(x,dtype=torch.float32,device=device).reshape(-1,2) for x in spans))
                else:pred=model(raw)
                report=evaluate_full_piece(pred,target,geometry=geometry,duration_quarters=float(raw['duration']),**(metric_options or {}))
                report.update(work_id=target['work_id'],family_id=target['family_id'],view_id=target.get('view_id'))
                reports.append(report)
    finally:model.train(previous)
    return dict(schema='HiergraphFullEvaluation@1',split=split,geometry=geometry,reports=reports,aggregate=aggregate_full_metrics(reports),test_locked=True)


def infer_full(model,raw,*,work_id,field_values=None,edge_threshold=.5):
    """The training forward path, used without any target bundle."""
    previous=model.training;model.eval()
    try:
        with torch.no_grad():
            prediction=model(move_tensors(raw,next(model.parameters()).device))
            return structured_prediction(prediction,raw,work_id=work_id,field_values=field_values,edge_threshold=edge_threshold)
    finally:model.train(previous)


def full_vocabulary_sha256(head_sizes,field_values):
    """Fingerprint class meanings in their fitted index order for deployment."""
    if not isinstance(head_sizes,dict) or set(head_sizes)!={'H','h','P'} or not isinstance(field_values,dict):raise ValueError('complete fitted vocabulary required')
    sizes={name:size for scope in head_sizes.values() for name,size in scope.items()}
    if set(sizes)!=set(field_values):raise ValueError('complete active fitted vocabulary required')
    if any(type(size) is not int or size<1 or not isinstance(field_values[name],list) or len(field_values[name])!=size for name,size in sizes.items()):raise ValueError('fitted vocabulary/head size mismatch')
    return sha256(_json(field_values)).hexdigest()


def validate_full_config(config,mode):
    if config.get('schema')!='HiergraphFullRun@1' or config.get('prepared') is not True:raise ValueError('prepared HiergraphFullRun@1 config required')
    if config.get('test_locked') is not True or config.get('augmentation')!='none' or config.get('freeze_policy')!='frozen_C3':raise ValueError('full baseline requires TEST lock, no augmentation and frozen C3')
    if mode not in ('gate','train','infer'):raise ValueError('unknown full execution mode')
    for key in (('source_manifest_sha256',) if mode=='infer' else ('source_manifest_sha256','target_manifest_sha256','cache_manifest_sha256')):
        value=config.get(key,'')
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError(f'prepared {key} required')
    cap=config.get('wall_seconds')
    if type(cap) is not int or cap<=0:raise ValueError('positive explicit wall cap required')
    if mode=='infer':
        if config.get('model_checkpoint',{}).get('bindings',{}).get('c3_sha256')!=config.get('c3_checkpoint',{}).get('sha256') or not config.get('c3_checkpoint',{}).get('sha256'):raise ValueError('inference C3 must match trained checkpoint binding')
        if full_vocabulary_sha256(config.get('head_sizes'),config.get('field_values'))!=config['model_checkpoint']['bindings'].get('output_vocabulary_sha256'):raise ValueError('inference fitted vocabulary binding mismatch')
    if mode=='train':
        milestones=config.get('selection_steps');target=config.get('max_applied_updates');draws=config.get('max_attempted_steps')
        if not isinstance(milestones,list) or not milestones or any(type(n) is not int or n<=0 for n in milestones) or milestones!=sorted(set(milestones)):raise ValueError('explicit increasing selection milestones required')
        if type(target) is not int or target!=milestones[-1] or type(draws) is not int or draws<target:raise ValueError('explicit applied/attempted budgets required')
        if config.get('selection_metric') not in ('H.known_h_recovery','P.known_p_recovery','H.joint_h_f_recovery'):raise ValueError('predeclared supported selection metric required')
    return config


def _metric(aggregate,path):
    value=aggregate
    for key in path.split('.'):
        if not isinstance(value,dict) or key not in value:raise ValueError('declared selection metric unavailable')
        value=value[key]
    if value is not None and (not isinstance(value,(float,int)) or not math.isfinite(value)):raise ValueError('selection metric must be finite or unavailable')
    return value


def run_full_train(config,output,*,seed=17):
    """Explicit fresh training, called only after external registry admission."""
    from .pipeline import open_inputs
    from .full_data import full_head_sizes
    from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json
    validate_full_config(config,'train');out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+config['wall_seconds'];state=None;best=None;curve=[]
    torch.set_num_threads(config.get('cpu_threads',1));torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    try:
        dataset,cache=open_inputs(config)
        heads=full_head_sizes(dataset);model=make_full_model(heads,config,seed)
        warm=None
        if config.get('warmstart'):
            warm=warmstart_e3(model.constructor,**config['warmstart'])
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        bindings=dict(source_sha256=config['source_manifest_sha256'],config_sha256=sha256(_json(config)).hexdigest(),
            data_sha256=config['target_manifest_sha256'],cache_sha256=config['cache_manifest_sha256'],c3_sha256=cache.manifest['checkpoint']['sha256'])
        field_values=dict(dataset.value_maps);field_values.update(dataset.object_value_maps.get('harmonic_process',{}))
        field_values={name:field_values[name] for scope in heads.values() for name in scope}
        bindings['output_vocabulary_sha256']=full_vocabulary_sha256(heads,field_values)
        bindings=full_checkpoint_bindings(model,bindings)
        fit=WholeWorkSamples(dataset,cache,'fit',source_root=config['source_root'],limits=config.get('raw_limits'))
        selection=WholeWorkSamples(dataset,cache,'selection',source_root=config['source_root'],limits=config.get('raw_limits'))
        atomic_json(out/'resolved.json',dict(config=config,head_sizes=heads,field_values=field_values,warmstart=warm,bindings=bindings))
        def retain(current,row):
            nonlocal state
            state=current
            with (out/'exposure.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            atomic_json(out/'progress.json',dict(status='running',state=state,wall_seconds=time.monotonic()-start))
        for milestone in config['selection_steps']:
            while state is None or state['applied_updates']<milestone:
                attempted=0 if state is None else state['attempted_steps']
                if attempted>=config['max_attempted_steps']:raise RuntimeError('attempted draw cap before requested applied updates')
                run=train_full_steps(model,fit,optimizer,steps=1,seed=seed,state=state,weights=config.get('loss_weights'),
                    clip_grad_norm=config.get('clip_grad_norm',1.),on_step=retain,deadline=deadline)
                state=run['state']
            report=evaluate_full(model,selection,deadline=deadline,metric_options=config.get('evaluation'))
            atomic_json(out/f'selection-{milestone:06d}.json',report)
            primary=_metric(report['aggregate'],config['selection_metric'])
            checkpoint=save_full_checkpoint(out/f'checkpoint-{milestone:06d}.pt',model,optimizer,state,bindings=bindings);checkpoint['path']=Path(checkpoint['path']).name
            row=dict(applied_updates=milestone,attempted_steps=state['attempted_steps'],primary=primary,checkpoint=checkpoint,exposure_sha256=state['exposure_sha256'])
            curve.append(row);atomic_json(out/'selection-curve.json',dict(metric=config['selection_metric'],rows=curve))
            if primary is not None and (best is None or primary>best['primary']):best=row
        if best is None:raise ValueError('no supported selection metric at any checkpoint')
        final_state=deepcopy(state)
        load_full_checkpoint(out/best['checkpoint']['path'],model,optimizer,bindings=bindings,expected_sha256=best['checkpoint']['sha256'])
        dev=WholeWorkSamples(dataset,cache,'dev',source_root=config['source_root'],limits=config.get('raw_limits'))
        final=evaluate_full(model,dev,split='dev',allow_dev=True,deadline=deadline,metric_options=config.get('evaluation'))
        atomic_json(out/'dev-selected.json',final)
        oracle=None
        if config.get('oracle_diagnostic',False):
            oracle=evaluate_full(model,dev,split='dev',allow_dev=True,geometry='oracle',deadline=deadline,metric_options=config.get('evaluation'))
            atomic_json(out/'dev-oracle.json',oracle)
        result=dict(status='completed',summary='Whole-work model training completed; see masked per-task metrics and coverage.',seed=seed,actual_state=final_state,selected=best,
            selection_metric=config['selection_metric'],dev=final['aggregate'],oracle=None if oracle is None else oracle['aggregate'],head_sizes=heads,warmstart=warm,
            wall_seconds=time.monotonic()-start,test_locked=True,augmentation='none')
    except BaseException as error:
        result=dict(status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed',summary='Whole-work execution did not complete.',actual_state=state,selected=best,
            error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-start)
        atomic_json(out/'result.json',result);raise
    atomic_json(out/'result.json',result);return result


def full_native_gate(config,output,*,seed=17):
    """Registered fit-only admission with backward and zero optimizer steps."""
    from .pipeline import open_inputs
    from .full_data import full_head_sizes
    from .full_losses import full_loss
    from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json
    validate_full_config(config,'gate');out=Path(output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();rows=[];torch.set_num_threads(config.get('cpu_threads',1));torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    try:
        dataset,cache=open_inputs(config);samples=WholeWorkSamples(dataset,cache,'fit',source_root=config['source_root'],limits=config.get('raw_limits'))
        model=make_full_model(full_head_sizes(dataset),config,seed)
        if config.get('warmstart'):warmstart_e3(model.constructor,**config['warmstart'])
        if type(config.get('gate_works')) is not int or config['gate_works']<1:raise ValueError('positive explicit gate works required')
        chosen={}
        for i,row in enumerate(samples.metadata):chosen.setdefault(row['work_id'],i)
        ranking=sorted(chosen,key=lambda wid:(sum(row['nodes'] for row in cache.works[wid]['tiles']),wid))
        count=min(config['gate_works'],len(ranking))
        selected=[ranking[round(j*(len(ranking)-1)/max(1,count-1))] for j in range(count)]
        for i in [chosen[wid] for wid in selected]:
            if time.monotonic()-start>=config['wall_seconds']:raise TimeoutError('whole-work gate wall cap')
            raw,target=samples[i];raw=move_tensors(raw,next(model.parameters()).device);model.zero_grad(set_to_none=True)
            begin=time.monotonic()
            if config['device'].startswith('cuda'):
                cuda_statistics=initialize_cuda_memory_statistics(next(model.parameters()).device)
                torch.cuda.synchronize(cuda_statistics.logical_device_index)
            pred=model(raw)
            if config['device'].startswith('cuda'):torch.cuda.synchronize(cuda_statistics.logical_device_index)
            forward_seconds=time.monotonic()-begin;loss_begin=time.monotonic()
            loss=full_loss(pred,target)
            if config['device'].startswith('cuda'):torch.cuda.synchronize(cuda_statistics.logical_device_index)
            loss_seconds=time.monotonic()-loss_begin;backward_begin=time.monotonic()
            if not bool(torch.isfinite(loss['loss'])):raise ValueError('nonfinite native gate loss')
            if loss['supported']:loss['loss'].backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('nonfinite native gate gradient')
            if config['device'].startswith('cuda'):torch.cuda.synchronize(cuda_statistics.logical_device_index)
            rows.append(dict(work_id=target['work_id'],known_supervision=loss['supported'],support=loss['support'],loss=float(loss['loss'].detach()),
                H=len(pred['H']['spans']),h=len(pred['h']['spans']),P=len(pred['P']['spans']),raw_coverage=raw['coverage'],wall_seconds=time.monotonic()-begin,
                forward_seconds=forward_seconds,loss_seconds=loss_seconds,backward_seconds=time.monotonic()-backward_begin,
                nonzero_gradient_tensors=sum(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated(cuda_statistics.logical_device_index) if config['device'].startswith('cuda') else None))
            atomic_json(out/'progress.json',dict(status='running',samples=rows))
        if not rows or not any(x['known_supervision'] and x['nonzero_gradient_tensors'] for x in rows):raise ValueError('no supported finite-gradient gate work')
        result=dict(status='completed',summary='Bounded whole-work fit gate completed; no optimizer updates or model quality claim.',samples=rows,applied_updates=0,wall_seconds=time.monotonic()-start,test_locked=True)
    except BaseException as error:
        result=dict(status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed',summary='Whole-work gate failed.',samples=rows,applied_updates=0,error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-start);atomic_json(out/'result.json',result);raise
    atomic_json(out/'result.json',result);return result
