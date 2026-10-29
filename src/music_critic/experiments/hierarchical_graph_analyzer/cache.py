"""Target-blind frozen-C3 cache with exact raw fragments and full-work bars.

No theory export is passed to assemble_raw_tile. The dataset manifest supplies
only authenticated canonical piece bindings and family splits to the builder.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import time

import torch

from music_critic.experiments.hiergraph_export import checked_read, file_binding, fingerprint
from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json

BASE_NODES = ('note', 'onset', 'beat', 'bar', 'track', 'song', 'slice')
NODE_TYPES = BASE_NODES + ('fragment', 'global_bar')
EXTRA_EDGES = (('fragment','of_note','note'),('note','has_fragment','fragment'),
               ('fragment','at_slice','slice'),('slice','has_fragment','fragment'),
               ('global_bar','next_global','global_bar'),('global_bar','previous_global','global_bar'),
               ('bar','global_parent','global_bar'),('global_bar','local_child','bar'))
LIMITS = dict(core_bars=4, halo_bars=1, max_tracks=64, max_notes_per_tile=2048,
              max_fragments=8192, max_times=512, max_bank=4096,
              max_flat_edges=300000, max_global_bars=4096)
RAW_KEYS = ('x','node_type','edge_index','edge_type','node_time','time_index',
            'time_positions','bank_indices','pool_mask')


@dataclass(frozen=True, slots=True)
class RawGraphProfile:
    """Raw tiling/graph capacity, separate from the 16-track PLL event codec.

    Only build_observed_tiles/build_tile_graph consume this structural profile.
    C3's direct _encode_graph seam uses variable track rows and dynamic coarse
    tokens; no Event, codec, PLL decoder or shared ContextProfile is changed.
    All tracks are retained up to the explicit cap; oversized works are excluded.
    """
    context_bars: int = 32
    core_bars: int = 4
    halo_bars: int = 1
    max_tracks: int = 64
    max_notes_per_tile: int = 2048
    max_edges_per_tile: int = 200000
    max_slices_per_tile: int = 8192

    def __post_init__(self):
        for name in ('context_bars', 'core_bars', 'max_tracks', 'max_notes_per_tile',
                     'max_edges_per_tile', 'max_slices_per_tile'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError('hiergraph.invalid_raw_graph_profile:' + name)
        if type(self.halo_bars) is not int or self.halo_bars < 0:
            raise ValueError('hiergraph.invalid_raw_graph_profile:halo_bars')
        if self.core_bars > self.context_bars or self.max_tracks > 64:
            raise ValueError('hiergraph.invalid_raw_graph_profile:capacity')


class UnsupportedRawGeometry(ValueError):
    pass


def pair(t):
    return [t.numerator,t.denominator]


def edge_schema():
    from music_critic.graph.relations import MANDATORY_EDGE_TYPES
    from music_critic.experiments.analysisgnn.temporal_slices.graph import EDGES
    schema = tuple(MANDATORY_EDGE_TYPES) + tuple(EDGES) + EXTRA_EDGES
    if len(schema) > 64 or len(set(schema)) != len(schema):
        raise ValueError('unique raw edge schema must fit64 slots')
    return schema


def raw_bank(times):
    """Raw adjacent intervals plus nearest raw endpoints at fixed qn scales."""
    bank = set(zip(range(len(times)-1),range(1,len(times))))
    for i,a in enumerate(times[:-1]):
        for duration in (Fraction(1,2),Fraction(1),Fraction(2),Fraction(4),Fraction(8)):
            j = bisect_left(times,a+duration)
            if i < j < len(times):
                bank.add((i,j))
    if len(bank)>LIMITS['max_bank']:
        raise UnsupportedRawGeometry('raw_bank_capacity')
    return sorted(bank)


def assemble_raw_tile(piece,tile,graph,encoded):
    """Exact physical identities are side metadata; tensors contain raw music."""
    left,right=tile.global_start.to_fraction(),tile.global_end.to_fraction()
    duration=right-left
    if duration<=0:
        raise UnsupportedRawGeometry('nonpositive_tile')
    times=[left+t for t,_ in graph.slots[0].time_slots]
    if len(times)>LIMITS['max_times']:
        raise UnsupportedRawGeometry('raw_time_capacity')
    if times[0]!=left or times[-1]!=right:
        raise ValueError('raw time grid endpoints differ')
    schema=edge_schema();offset={};features=[];types=[];spans=[];cursor=0
    notes={n.note_id:n for n in piece.notes}
    bars={b.bar_id:b for b in piece.bars};beats={b.beat_id:b for b in piece.beats}
    geometry={g.note_id:g for g in tile.note_geometry}
    onsets=sorted({g.onset.to_fraction() for g in tile.note_geometry if not g.left_censored})
    for kind,name in enumerate(BASE_NODES):
        value=encoded[name].detach().cpu().float()
        expected=graph.slice_features.shape[0] if name=='slice' else graph.base[name].num_nodes
        if value.shape!=(expected,128) or not torch.isfinite(value).all():
            raise ValueError('finite aligned128d C3 states required')
        offset[name]=cursor;cursor+=expected
        features.append(torch.cat((value,torch.zeros((expected,138))),dim=1));types.extend([kind]*expected)
        for i,(_,identity) in enumerate(graph.keys.get(name,[(0,str(j)) for j in range(expected)])):
            if name=='note':
                g=geometry[identity];a=g.onset.to_fraction();b=a+g.duration.to_fraction()
            elif name=='onset': a=b=onsets[i]
            elif name in ('beat','bar'):
                obj=(beats if name=='beat' else bars)[identity]
                a=obj.start_qn.to_fraction();b=a+obj.duration_qn.to_fraction()
            elif name=='slice': a=b=times[i]
            else: a,b=left,right
            spans.append((a,b))
    edges=[];tags=[]
    for tag,edge in enumerate(schema[:-len(EXTRA_EDGES)]):
        values=(graph.edges[edge] if edge in graph.edges else graph.base[edge].edge_index).detach().cpu().clone()
        values[0]+=offset[edge[0]];values[1]+=offset[edge[2]]
        edges.append(values);tags.extend([tag]*values.shape[1])
    extra=[];extra_tags=[];fragments=[];fragment_features=[]
    track_index={t.track_id:i for i,t in enumerate(piece.tracks)}
    for ni,(_,identity) in enumerate(graph.keys['note']):
        note=notes[identity];a=note.onset_qn.to_fraction();b=a+note.duration_qn.to_fraction()
        if note.is_percussion or a==b: continue
        for si,(u,v) in enumerate(zip(times,times[1:])):
            start,end=max(u,a),min(v,b)
            if start>=end: continue
            if len(fragments)>=LIMITS['max_fragments']:
                raise UnsupportedRawGeometry('fragment_capacity')
            n=cursor+len(fragments);age=float(start-a);dur=float(b-a)
            raw=[note.pitch/127,math.sin(note.pitch%12*math.pi/6),math.cos(note.pitch%12*math.pi/6),
                 dur/(1+dur),float(start-left)/32,track_index[note.track_id]/16,
                 float(note.is_percussion),float(note.is_grace),age/(1+abs(age)),float(a-left)/32]
            fragment_features.append(torch.cat((encoded['note'][ni].cpu(),encoded['slice'][si].cpu(),torch.tensor(raw))).float())
            fragments.append(dict(note_id=identity,start=pair(start),end=pair(end),node=n))
            spans.append((start,end));types.append(7)
            for s,t,e in ((n,offset['note']+ni,0),(offset['note']+ni,n,1),
                          (n,offset['slice']+si,2),(offset['slice']+si,n,3)):
                extra.append((s,t));extra_tags.append(schema.index(EXTRA_EDGES[e]))
    if fragment_features: features.append(torch.stack(fragment_features))
    if extra: edges.append(torch.tensor(extra,dtype=torch.long).T.contiguous());tags.extend(extra_tags)
    if len(tags)>LIMITS['max_flat_edges']: raise UnsupportedRawGeometry('raw_edge_capacity')
    core=[bars[i] for i in tile.core_bar_ids]
    ca=core[0].start_qn.to_fraction();cb=(core[-1].start_qn+core[-1].duration_qn).to_fraction()
    pool=[kind in (6,7) for kind in types]
    raw=dict(x=torch.cat(features),node_type=torch.tensor(types,dtype=torch.long),
             edge_index=torch.cat(edges,dim=1),edge_type=torch.tensor(tags,dtype=torch.long),
             node_time=torch.tensor([[float((a-left)/duration),float((b-left)/duration)] for a,b in spans]),
             time_index=torch.arange(len(times),dtype=torch.long)+offset['slice'],
             time_positions=torch.tensor([float((t-left)/duration) for t in times]),
             bank_indices=torch.tensor(raw_bank(times),dtype=torch.long).reshape(-1,2),
             pool_mask=torch.tensor(pool,dtype=torch.bool))
    metadata=dict(core_start=pair(ca),core_end=pair(cb),halo_start=pair(left),halo_end=pair(right),
                  raw_times=[pair(t) for t in times],fragments=fragments,
                  local_bars=[dict(id=identity,node=offset['bar']+i) for i,(_,identity) in enumerate(graph.keys['bar'])],
                  point_pitched_notes=sum(not n.is_percussion and n.duration_qn.to_fraction()==0 for n in tile.piece.notes))
    return dict(raw=raw,metadata=metadata)


def attach_global_bars(payload,context):
    """Every tile retains every available whole-piece bar state individually."""
    raw={k:v.clone() for k,v in payload['raw'].items()};m=payload['metadata']
    count=len(context['ids']);n=raw['x'].shape[0]
    if count>LIMITS['max_global_bars']: raise UnsupportedRawGeometry('global_bar_capacity')
    left,right=map(lambda k:Fraction(*m[k]),('halo_start','halo_end'));d=right-left
    raw['x']=torch.cat((raw['x'],torch.cat((context['states'],torch.zeros((count,138))),dim=1)))
    raw['node_type']=torch.cat((raw['node_type'],torch.full((count,),8,dtype=torch.long)))
    raw['node_time']=torch.cat((raw['node_time'],torch.tensor([[float((Fraction(*a)-left)/d),float((Fraction(*b)-left)/d)] for a,b in context['spans']]).reshape(-1,2)))
    raw['pool_mask']=torch.cat((raw['pool_mask'],torch.zeros(count,dtype=torch.bool)))
    idx={identity:n+i for i,identity in enumerate(context['ids'])};edges=[];tags=[];schema=edge_schema()
    def add(a,b,e): edges.append((a,b));tags.append(schema.index(EXTRA_EDGES[e]))
    for i in range(count-1):add(n+i,n+i+1,4);add(n+i+1,n+i,5)
    for row in m['local_bars']:
        if row['id'] in idx:add(row['node'],idx[row['id']],6);add(idx[row['id']],row['node'],7)
    if edges:
        raw['edge_index']=torch.cat((raw['edge_index'],torch.tensor(edges,dtype=torch.long).T),dim=1)
        raw['edge_type']=torch.cat((raw['edge_type'],torch.tensor(tags,dtype=torch.long)))
    if raw['edge_type'].numel()>LIMITS['max_flat_edges']:
        raise UnsupportedRawGeometry('final_raw_edge_capacity')
    return raw


def build_raw_cache(dataset,source_root,output,checkpoint,checkpoint_sha,checkpoint_size,*,device='cuda',wall_seconds=1800):
    from music_critic.data import loads_piece
    from music_critic.graph import GraphBuildError
    from music_critic.relation_probe.data import sanitize
    from music_critic.pll.contracts import MaskQuery
    from music_critic.pll.encoder import PLLContextEncoder
    from music_critic.pll.graph import build_tile_graph
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.observed_view import ObservedMusicView,UnsupportedContextError
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(schema='HiergraphRawCache@1',status='running',works=[],exclusions=[],
                dataset_fingerprint=dataset.manifest['fingerprint'],source_bindings=[],
                checkpoint=dict(sha256=checkpoint_sha,size_bytes=checkpoint_size),limits=LIMITS,
                node_schema=NODE_TYPES,edge_schema=edge_schema(),test_locked=True,
                feature_contract='raw_C3_frozen_all_tiles_full_work_bars',torch_version=str(torch.__version__))
    def flush(status):
        result.update(status=status,wall_seconds=time.monotonic()-started)
        result['fingerprint']=fingerprint(result);atomic_json(out/'manifest.json',result)
    flush('running')
    try:
        torch.set_num_threads(1)
        encoder=PLLContextEncoder.from_pretrained(checkpoint,expected_sha256=checkpoint_sha,
                  expected_size_bytes=checkpoint_size,freeze_policy='frozen').to(device)
        encoder.eval();encoder.requires_grad_(False)
        for wid,work in sorted(dataset.works.items()):
            if time.monotonic()-started>=wall_seconds: raise TimeoutError('raw cache wall cap')
            binding=work['source_bindings']['canonical_piece']
            piece=sanitize(loads_piece(checked_read(source_root,binding,128*1024**2)))
            result['source_bindings'].append(dict(binding,work_id=wid))
            if not piece.bars: raise ValueError('raw piece needs observed meter bars')
            row=dict(work_id=wid,family_id=work['family_id'],split=work['split'],tiles=[],exclusions=[])
            result['works'].append(row)
            profile=RawGraphProfile(context_bars=max(32,len(piece.bars)),core_bars=LIMITS['core_bars'],halo_bars=LIMITS['halo_bars'],max_tracks=LIMITS['max_tracks'],max_notes_per_tile=LIMITS['max_notes_per_tile'])
            start=piece.bars[0].start_qn;end=piece.bars[-1].start_qn+piece.bars[-1].duration_qn
            row['duration']=pair((end-start).to_fraction())
            view=ObservedMusicView(piece,MaskQuery(start,end,tuple(t.track_id for t in piece.tracks)),tuple(range(len(piece.bars))))
            try: tiles=build_observed_tiles(view,profile)
            except UnsupportedContextError as error:
                row['exclusions'].append(dict(reason=str(error),domain='whole_work',core_start=pair(start.to_fraction()),core_end=pair(end.to_fraction()),halo_start=pair(start.to_fraction()),halo_end=pair(end.to_fraction())));flush('running');continue
            contexts={};bar_lookup={b.bar_id:b for b in piece.bars}
            for ordinal,tile in enumerate(tiles):
                if time.monotonic()-started>=wall_seconds: raise TimeoutError('raw cache wall cap')
                try:
                    try:
                        graph=build_tile_graph(tile,profile)
                    except GraphBuildError as error:
                        # C3 cannot represent every admitted source measure after
                        # target-free cropping. Preserve that geometry exclusion,
                        # but never hide framework/safety failures or encoder errors.
                        if not str(error).startswith('build_raw_graph requires a validator-clean CanonicalPiece:'):
                            raise
                        raise UnsupportedRawGeometry(str(error)) from error
                    flags=torch.tensor(tile.boundary_flags,dtype=torch.float32,device=device).reshape(-1,2)
                    with torch.no_grad(): states=encoder._encode_graph(graph.to(device),flags)
                    states={k:v.detach().cpu() for k,v in states.items()}
                    for i,(_,identity) in enumerate(graph.keys['bar']):
                        if identity in tile.core_bar_ids: contexts[identity]=states['bar'][i]
                    payload=assemble_raw_tile(piece,tile,graph,states)
                except (UnsupportedRawGeometry,UnsupportedContextError) as error:
                    owned=[bar_lookup[i] for i in tile.core_bar_ids]
                    row['exclusions'].append(dict(tile=ordinal,reason=str(error),domain='raw_tile',core_start=pair(owned[0].start_qn.to_fraction()),core_end=pair((owned[-1].start_qn+owned[-1].duration_qn).to_fraction()),halo_start=pair(tile.global_start.to_fraction()),halo_end=pair(tile.global_end.to_fraction())));continue
                path=out/f'{wid}-{ordinal:04d}.pt';torch.save(payload,path)
                row['tiles'].append(dict(id=f'{wid}-{ordinal:04d}',**file_binding(path),metadata={k:payload['metadata'][k] for k in ('core_start','core_end','halo_start','halo_end')},nodes=payload['raw']['x'].shape[0],candidates=len(payload['raw']['bank_indices'])))
            ids=[b.bar_id for b in piece.bars if b.bar_id in contexts]
            if len(ids)>LIMITS['max_global_bars']: raise UnsupportedRawGeometry('global_bar_capacity')
            context=dict(ids=ids,states=torch.stack([contexts[i] for i in ids]) if ids else torch.zeros((0,128)),
                         spans=[[pair(bar_lookup[i].start_qn.to_fraction()),pair((bar_lookup[i].start_qn+bar_lookup[i].duration_qn).to_fraction())] for i in ids])
            path=out/f'{wid}-context.pt';torch.save(context,path);row['context']=file_binding(path)
            row['context_coverage']=dict(observed_bars=len(piece.bars),encoded_bars=len(ids))
            flush('running');print(json.dumps(dict(cached_work=wid,tiles=len(row['tiles']),excluded=len(row['exclusions']))),flush=True)
        flush('completed')
    except Exception as error:
        result['error']=dict(type=type(error).__name__,reason=str(error));flush('failed');raise
    return result


class RawCache:
    def __init__(self,root,manifest_sha256,*,dataset_fingerprint):
        self.root=Path(root);data=(self.root/'manifest.json').read_bytes()
        if sha256(data).hexdigest()!=manifest_sha256:raise ValueError('cache manifest SHA mismatch')
        self.manifest=json.loads(data)
        if (self.manifest['status']!='completed' or self.manifest['schema']!='HiergraphRawCache@1'
            or self.manifest['fingerprint']!=fingerprint(self.manifest)
            or self.manifest['dataset_fingerprint']!=dataset_fingerprint
            or self.manifest.get('node_schema')!=list(NODE_TYPES)
            or self.manifest.get('edge_schema')!=[list(e) for e in edge_schema()]
            or self.manifest.get('feature_contract')!='raw_C3_frozen_all_tiles_full_work_bars'
            or self.manifest.get('limits')!=LIMITS
            or self.manifest.get('test_locked') is not True):raise ValueError('cache identity mismatch')
        self.works={r['work_id']:r for r in self.manifest['works']}
        self._context_id=None;self._context=None
    def load(self,work_id,tile_row):
        if self._context_id!=work_id:
            self._context=torch.load(io.BytesIO(checked_read(self.root,self.works[work_id]['context'],128*1024**2)),weights_only=True,map_location='cpu');self._context_id=work_id
        payload=torch.load(io.BytesIO(checked_read(self.root,tile_row,256*1024**2)),weights_only=True,map_location='cpu')
        return attach_global_bars(payload,self._context),payload['metadata']
