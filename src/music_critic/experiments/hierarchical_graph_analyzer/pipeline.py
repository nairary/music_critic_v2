"""Authenticated data/cache integration; no autonomous training launch."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import time
import torch

from .data import load_target_dataset
from .cache import RawCache,raw_bank
from .model import HAnalyzer
from .runtime import tensorize_targets,initialize_shared_parameters
from .losses import compute_loss
from .metrics import evaluate_piece,aggregate_metrics
from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json


def bind_boundaries(target,raw_times):
    """Target-only endpoint lookup uses exact rationals; never float equality."""
    out=dict(target);times=[Fraction(*t) for t in raw_times];lookup={t:i for i,t in enumerate(times)}
    ends={t for pair in target['exact_spans'] for t in pair if times[0]<t<times[-1]}
    out['known_boundary_indices']=sorted({lookup[t] for t in ends if t in lookup})
    out['boundary_endpoint_coverage']=dict(known_interior=len(ends),on_grid=len(out['known_boundary_indices']),off_grid=sum(t not in lookup for t in ends))
    return out


class Samples:
    """Target-only inventory, lazy SHA-verified raw tensors, one raw tile cache."""
    def __init__(self,dataset,cache,split):
        self.dataset,self.cache,self.split=dataset,cache,split
        self.metadata=[];self.rows=[];self.missing=[];self.coverage=Counter()
        for wid,work in sorted(cache.works.items()):
            if work['split']!=split:continue
            self.coverage['works']+=1
            for row in work['tiles']:
                targets=dataset.targets_for_tile(wid,**row['metadata'])
                for target in targets:
                    self.metadata.append(target);self.rows.append((wid,row))
                    for k,v in target['coverage'].items():
                        if isinstance(v,int) and k not in ('total_events','not_owned','point_events'):self.coverage[k]+=v
            for exclusion in work['exclusions']:
                geometry={k:exclusion[k] for k in ('core_start','core_end','halo_start','halo_end')}
                for target in dataset.targets_for_tile(wid,**geometry):
                    self.missing.append(dict(target=target,reason=exclusion['reason']))
                    self.coverage['raw_excluded_owned_h']+=len(target['ids'])+target['coverage']['outside_halo']
        self._last=None;self._raw=None;self._raw_metadata=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        wid,row=self.rows[i]
        if self._last!=(wid,row['id']):
            self._raw,self._raw_metadata=self.cache.load(wid,row);self._last=(wid,row['id'])
        return self._raw,bind_boundaries(self.metadata[i],self._raw_metadata['raw_times'])
    def missing_reports(self):
        reports=[]
        for row in self.missing:
            target=row['target'];core=target['core_span'];halo=target['halo_span']
            empty=dict(spans=[],object_logits=[],field_logits={},view_id=target['view_id'],scope_id=target['scope_id'])
            report=evaluate_piece(empty,target,duration_quarters=float(core[1]-core[0]),span_scale_quarters=float(halo[1]-halo[0]))
            report.update(work_id=target['work_id'],family_id=target['family_id'],view_id=target['view_id'],raw_exclusion=row['reason'])
            reports.append(report)
        return reports


def open_inputs(config):
    dataset=load_target_dataset(config['targets'],manifest_sha256=config['target_manifest_sha256'])
    cache=RawCache(config['cache'],config['cache_manifest_sha256'],dataset_fingerprint=dataset.manifest['fingerprint'])
    if set(dataset.works)!=set(cache.works):raise ValueError('raw cache work coverage differs from admitted targets')
    for wid,work in dataset.works.items():
        if any(work[k]!=cache.works[wid][k] for k in ('split','family_id')):raise ValueError('raw/target split mismatch')
    return dataset,cache


def make_model(kind,heads,config,seed):
    model=HAnalyzer(kind,heads,**config['model']).to(config['device'])
    return initialize_shared_parameters(model,seed)


def native_gate(config,output):
    """Real fit-only engineering forward/backward; zero optimizer steps."""
    out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(1)
    dataset,cache=open_inputs(config);samples=Samples(dataset,cache,'fit')
    # Raw size, never labels, chooses small/median/largest successful tiles.
    representatives={}
    for i,(wid,row) in enumerate(samples.rows):representatives.setdefault((wid,row['id']),i)
    ranked=sorted(representatives.values(),key=lambda i:(samples.rows[i][1]['nodes'],samples.rows[i][1]['id']))
    if not ranked:raise ValueError('no fit raw tiles')
    chosen=sorted(set([ranked[0],ranked[len(ranked)//2],ranked[-1]]))
    result=dict(schema='HiergraphNativeGate@1',status='running',applied_updates=0,conditions=[],
                checkpoint_selection=False,dev_metrics_computed=False,test_locked=True,
                target_manifest_sha256=config['target_manifest_sha256'],cache_manifest_sha256=config['cache_manifest_sha256'],
                heads=dataset.head_sizes,fit_coverage=dict(samples.coverage),raw_exclusions=sum(len(w['exclusions']) for w in cache.works.values()),
                torch=str(torch.__version__),cuda=torch.version.cuda,device=config['device'],augmentation='none')
    if config['device'].startswith('cuda'):result['gpu']=torch.cuda.get_device_name()
    atomic_json(out/'result.json',result)
    try:
        for kind in ('E1','E2','E3','E4','E5'):
            model=make_model(kind,dataset.head_sizes,config,17);rows=[]
            result['conditions'].append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),samples=rows))
            atomic_json(out/'result.json',result)
            for i in chosen:
                if time.monotonic()-start>config.get('gate_wall_seconds',600):raise TimeoutError('native gate wall cap')
                raw,target=samples[i];raw={k:v.to(config['device']) for k,v in raw.items()}
                target=tensorize_targets(target,device=config['device']);model.zero_grad(set_to_none=True)
                if config['device'].startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin=time.monotonic();pred=model(raw);loss=compute_loss(pred,target)
                if not torch.isfinite(loss['loss']):raise ValueError('nonfinite real gate loss')
                loss['loss'].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite real gate gradients')
                if config['device'].startswith('cuda'):torch.cuda.synchronize()
                rows.append(dict(work_id=target['work_id'],tile=samples.rows[i][1]['id'],known_h=len(target['ids']),
                    nodes=len(raw['x']),edges=len(raw['edge_type']),raw_times=len(raw['time_positions']),
                    predicted_candidates=len(pred['spans']),wall_seconds=time.monotonic()-begin,
                    loss=float(loss['loss'].detach()),support=loss['support'],
                    gradient_tensors=sum(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if config['device'].startswith('cuda') else None))
                atomic_json(out/'result.json',result)
                del raw,pred,loss
            if not any(r['gradient_tensors']>0 and r['support']['matched_events']>0 for r in rows):raise ValueError('native gate has no supervised nonzero gradient sample')
            del model
            if config['device'].startswith('cuda'):torch.cuda.empty_cache()
            atomic_json(out/'result.json',result)
        result.update(status='completed',wall_seconds=time.monotonic()-start)
    except BaseException as error:
        result.update(status='failed',wall_seconds=time.monotonic()-start,error=dict(type=type(error).__name__,reason=str(error)))
        atomic_json(out/'result.json',result);raise
    atomic_json(out/'result.json',result)
    return result
