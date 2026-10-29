"""Whole-piece E3 H/h/P graph with addressed feedback and mutable geometry.

Raw observations are the only forward input. Candidate budgets are explicit
policies; missing candidate endpoints remain evaluation coverage misses. Physical
note IDs route outputs and never index learned embeddings. C3 features are frozen
upstream; the E3 constructor and every new module remain trainable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import HAnalyzer, _geometry


@dataclass(frozen=True)
class FullGraphConfig:
    input_width: int = 266
    hidden: int = 128
    attention_heads: int = 4
    constructor_rounds: int = 3
    rounds: int = 3
    h_per_tile: int = 64
    max_h: int = 4096
    max_p: int = 4096
    max_notes: int = 100000
    max_contexts: int = 131072
    max_participants: int = 524288
    max_relations: int = 524288
    context_slots: int = 2
    context_h_neighbors: int = 8
    participant_neighbors: int = 64
    relation_neighbors: int = 16
    relation_far_neighbors: int = 8
    attention_chunk: int = 128
    process_scales: tuple[int, ...] = (1, 2, 4, 8, 16)
    soft_feedback: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in ('process_scales', 'soft_feedback'):
                continue
            if type(value) is not int or value < 1:
                raise ValueError('full.invalid_config:' + key)
        if self.hidden % self.attention_heads:
            raise ValueError('full.hidden_not_divisible_by_heads')
        if not self.process_scales or any(type(x) is not int or x < 1 for x in self.process_scales):
            raise ValueError('full.invalid_process_scales')
        if len(set(self.process_scales)) != len(self.process_scales):
            raise ValueError('full.duplicate_process_scales')
        if type(self.soft_feedback) is not bool:
            raise ValueError('full.invalid_soft_feedback')


class FullGraphCapacityError(ValueError):
    """An entire work exceeds an explicit bound; callers must retain coverage."""


def _scatter(values, index, count, weights=None):
    result = values.new_zeros((count, values.shape[-1]))
    if not index.numel():
        return result
    if weights is None:
        weights = values.new_ones(index.numel())
    total = values.new_zeros(count).index_add(0, index, weights)
    return result.index_add(0, index, values * weights[:, None]) / total[:, None].clamp_min(1e-8)


def _refine_spans(spans, delta):
    """Residual ordered interval update in whole-work coordinates, not tile bounds."""
    eps = max(1e-5, torch.finfo(spans.dtype).eps * 16)
    start = spans[:, 0].clamp(eps, 1 - eps)
    width = ((spans[:, 1] - start) / (1 - start)).clamp(eps, 1 - eps)
    start = (1 - eps) * (torch.logit(start) + delta[:, 0]).sigmoid()
    width = (torch.logit(width) + delta[:, 1]).sigmoid()
    end = start + eps + (1 - eps - start) * width
    return torch.stack((start, end), -1)


class _Fields(nn.Module):
    def __init__(self, width, sizes):
        super().__init__()
        if any(not isinstance(k, str) or not k or type(v) is not int or v < 1 for k, v in sizes.items()):
            raise ValueError('full.invalid_field_sizes')
        self.names = tuple(sorted(sizes))
        self.heads = nn.ModuleList(nn.Linear(width, sizes[k]) for k in self.names)
        self.feedback = nn.Linear(sum(sizes.values()), width, bias=False) if sizes else None

    def forward(self, states):
        return {k: h(states) for k, h in zip(self.names, self.heads)}

    def soft(self, states):
        if self.feedback is None:
            return torch.zeros_like(states)
        return self.feedback(torch.cat([h(states).softmax(-1) for h in self.heads], -1))


class _Fusion(nn.Module):
    def __init__(self, width, channels):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(width * (channels + 1), width), nn.GELU(), nn.Linear(width, width))
        self.gate = nn.Linear(width * (channels + 1), width)
        self.norm = nn.LayerNorm(width)

    def forward(self, state, *messages):
        joined = torch.cat((state, *messages), -1)
        return self.norm(state + self.update(joined) * self.gate(joined).sigmoid())


class _GlobalAttention(nn.Module):
    """Each query accesses separate nodes globally; chunking bounds score allocation."""
    def __init__(self, width, heads, chunk):
        super().__init__()
        self.query, self.key, self.value = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.out = nn.Linear(width, width)
        self.heads, self.chunk = heads, chunk
        self.time_scale = nn.Parameter(torch.zeros(heads))

    def forward(self, query, context, q_spans, c_spans):
        if not len(query) or not len(context):
            return torch.zeros_like(query)
        d = query.shape[-1] // self.heads
        key = self.key(context).reshape(-1, self.heads, d).transpose(0, 1)
        value = self.value(context).reshape(-1, self.heads, d).transpose(0, 1)
        centers = c_spans.mean(-1)
        results = []
        for start in range(0, len(query), self.chunk):
            q = self.query(query[start:start + self.chunk]).reshape(-1, self.heads, d).transpose(0, 1)
            distance = (q_spans[start:start + self.chunk].mean(-1)[:, None] - centers[None, :]).abs()
            bias = -F.softplus(self.time_scale)[:, None, None] * distance[None]
            attended = F.scaled_dot_product_attention(q, key, value, attn_mask=bias, dropout_p=0.)
            results.append(attended.transpose(0, 1).reshape(-1, query.shape[-1]))
        return self.out(torch.cat(results, 0))


class _Round(nn.Module):
    def __init__(self, cfg):
        super().__init__(); w = cfg.hidden
        self.global_h = _GlobalAttention(w, cfg.attention_heads, cfg.attention_chunk)
        self.global_p = _GlobalAttention(w, cfg.attention_heads, cfg.attention_chunk)
        self.process_reads_h = _GlobalAttention(w, cfg.attention_heads, cfg.attention_chunk)
        self.h_bottom = _Fusion(w, 3)
        self.H_bottom = _Fusion(w, 3)
        self.P_bottom = _Fusion(w, 3)
        self.H_top = _Fusion(w, 2)
        self.h_top = _Fusion(w, 3)
        self.note_top = _Fusion(w, 1)
        self.participant_score = nn.Sequential(nn.Linear(w * 2 + 4, w), nn.GELU(), nn.Linear(w, 1))
        self.H_delta, self.P_delta, self.h_delta = (nn.Linear(w, 2) for _ in range(3))
        self.H_to_P, self.h_to_P, self.P_to_H, self.P_to_h = (nn.Linear(w, w) for _ in range(4))
        self.h_to_H, self.h_to_note = nn.Linear(w, w), nn.Linear(w, w)


class FullGraphAnalyzer(nn.Module):
    RAW_KEYS = frozenset(('tiles', 'tile_spans', 'core_spans', 'owned_time_indices', 'note_index_by_tile',
                         'notes', 'bar_spans', 'note_ids', 'origin', 'duration', 'coverage'))
    NOTE_KEYS = frozenset(('spans', 'pitch', 'track_index', 'is_percussion', 'is_grace'))

    def __init__(self, h_head_sizes, note_head_sizes, process_head_sizes,
                 relation_types=('prepares', 'resolves_to'), *, config=None):
        super().__init__(); self.config = config or FullGraphConfig(); c = self.config; w = c.hidden
        self.constructor = HAnalyzer('E3', h_head_sizes, input_width=c.input_width,
            hidden=w, rounds=c.constructor_rounds, max_events=c.h_per_tile)
        self.note_projection = nn.Sequential(nn.Linear(10, w), nn.GELU(), nn.LayerNorm(w))
        self.H_geometry, self.P_geometry, self.h_geometry = (nn.Linear(4, w) for _ in range(3))
        self.context_slots = nn.Parameter(torch.randn(c.context_slots, w) * .02)
        self.context_initial = nn.Sequential(nn.Linear(w * 2, w), nn.GELU(), nn.LayerNorm(w))
        self.note_fields, self.process_fields = _Fields(w, note_head_sizes), _Fields(w, process_head_sizes)
        self.H_feedback = nn.Linear(sum(h_head_sizes.values()), w, bias=False) if h_head_sizes else None
        self.P_object = nn.Linear(w, 1)
        self.round_layers = nn.ModuleList(_Round(c) for _ in range(c.rounds))
        self.participant_head = nn.Sequential(nn.Linear(w * 2 + 4, w), nn.GELU(), nn.Linear(w, 1))
        self.relation_types = tuple(relation_types)
        if len(set(self.relation_types)) != len(self.relation_types) or any(not isinstance(s, str) or not s for s in self.relation_types):
            raise ValueError('full.invalid_relation_types')
        self.relation_heads = nn.ModuleList(nn.Sequential(nn.Linear(w * 2 + 4, w), nn.GELU(), nn.Linear(w, 1))
                                          for _ in self.relation_types)

    def _validate(self, raw):
        if not isinstance(raw, Mapping) or set(raw) != self.RAW_KEYS:
            raise ValueError('full.raw_schema_only')
        if not isinstance(raw['notes'], Mapping) or set(raw['notes']) != self.NOTE_KEYS:
            raise ValueError('full.note_schema_only')
        if not raw['tiles'] or not len(raw['tiles']) == len(raw['note_index_by_tile']) == len(raw['owned_time_indices']):
            raise ValueError('full.tile_mapping')
        parameter = next(self.parameters()); n = len(raw['notes']['spans'])
        if n > self.config.max_notes: raise FullGraphCapacityError('full.note_capacity')
        if len(raw['note_ids']) != n or len(set(raw['note_ids'])) != n:
            raise ValueError('full.note_identity')
        if not isinstance(raw['origin'], Fraction) or not isinstance(raw['duration'], Fraction) or raw['duration'] <= 0:
            raise ValueError('full.exact_time_metadata')
        for name, shape in [('tile_spans', (len(raw['tiles']), 2)), ('core_spans', (len(raw['tiles']), 2)),
                            ('bar_spans', (len(raw['bar_spans']), 2))]:
            value = raw[name]
            if value.shape != shape or value.dtype != parameter.dtype or value.device != parameter.device or not torch.isfinite(value).all():
                raise ValueError('full.geometry:' + name)
            if ((value[:, 0] < 0) | (value[:, 1] > 1) | (value[:, 0] >= value[:, 1])).any():
                raise ValueError('full.invalid_interval:' + name)
        if not len(raw['bar_spans']): raise ValueError('full.no_bars')
        for name, value in raw['notes'].items():
            expected = (n, 2) if name == 'spans' else (n,)
            if value.shape != expected or value.device != parameter.device:
                raise ValueError('full.note_shape:' + name)
        ns = raw['notes']['spans']
        if ns.dtype != parameter.dtype or not torch.isfinite(ns).all() or (ns[:, 0] > ns[:, 1]).any() or (ns < 0).any() or (ns > 1).any():
            raise ValueError('full.note_geometry')
        for k in ('pitch', 'track_index'):
            if raw['notes'][k].dtype != torch.long: raise ValueError('full.note_integer:' + k)
        if ((raw['notes']['pitch'] < 0) | (raw['notes']['pitch'] > 127)).any() or (raw['notes']['track_index'] < 0).any():
            raise ValueError('full.note_value')
        for k in ('is_percussion', 'is_grace'):
            if raw['notes'][k].dtype != torch.bool: raise ValueError('full.note_boolean:' + k)
        for tile, index, owned in zip(raw['tiles'], raw['note_index_by_tile'], raw['owned_time_indices']):
            self.constructor._validate(tile)
            if (owned.dtype != torch.long or owned.device != parameter.device or owned.ndim != 1
                or (owned < 0).any() or (owned >= len(tile['time_positions'])).any()
                or (owned[1:] <= owned[:-1]).any()):
                raise ValueError('full.exact_owned_anchors')
            if index.dtype != torch.long or index.device != parameter.device or index.shape != (int((tile['node_type'] == 0).sum()),):
                raise ValueError('full.note_routing_shape')
            if index.numel() and ((index < 0).any() or (index >= n).any()): raise ValueError('full.note_routing_range')

    def _H_fields(self, state):
        return {k: h(state) for k, h in zip(self.constructor.field_names, self.constructor.field_heads)}

    def _semantic_H(self, state):
        if not self.config.soft_feedback or self.H_feedback is None: return torch.zeros_like(state)
        return self.H_feedback(torch.cat([v.softmax(-1) for v in self._H_fields(state).values()], -1))

    def _encode(self, raw):
        c = self.config
        # Reject known capacities before retaining any local autograd graph.
        if sum(min(len(x),c.h_per_tile) for x in raw['owned_time_indices'])>c.max_h:
            raise FullGraphCapacityError('full.H_capacity')
        hs=[]; spans=[]; ids=[]; note_states=[]; indices=[]
        proposed=0
        for tile_index, tile in enumerate(raw['tiles']):
            out = self.constructor(tile); a,b = raw['tile_spans'][tile_index]; scale=b-a
            # Core ownership was resolved from exact source Fractions upstream.
            # Reconstructing this decision from normalized float anchors can
            # lose a true boundary event or assign it to both adjacent cores.
            eligible = raw['owned_time_indices'][tile_index]
            proposed += len(eligible)
            rank = torch.argsort(out['object_logits'][eligible].detach(), descending=True, stable=True)[:c.h_per_tile]
            selected = eligible[rank].sort().values
            hs.append(out['states'][selected]);spans.append(a + scale * out['spans'][selected])
            ids.extend((tile_index, int(i)) for i in selected.detach().cpu())
            note_states.append(out['raw_states'][tile['node_type'] == 0]); indices.append(raw['note_index_by_tile'][tile_index])
        H=torch.cat(hs); intervals=torch.cat(spans)
        if not len(H): raise ValueError('full.no_owned_H_anchors')
        if len(H)>c.max_h: raise FullGraphCapacityError('full.H_capacity')
        notes=raw['notes']; s=notes['spans'];pitch=notes['pitch'].to(H.dtype);track=notes['track_index'].to(H.dtype)
        features=torch.cat((s, (s[:,1]-s[:,0])[:,None], (pitch/127)[:,None],
                            torch.sin(pitch*math.pi/6)[:,None],torch.cos(pitch*math.pi/6)[:,None],
                            (track/(1+track))[:,None],notes['is_percussion'].to(H.dtype)[:,None],
                            notes['is_grace'].to(H.dtype)[:,None],s.mean(-1)[:,None]),-1)
        n=self.note_projection(features)+_scatter(torch.cat(note_states),torch.cat(indices),len(s))
        return H+self.H_geometry(_geometry(intervals)),intervals,n,ids,proposed

    def _process_bank(self, raw, H, Hspans):
        bars=raw['bar_spans'];pairs=sorted({(i,min(len(bars)-1,i+s-1)) for i in range(len(bars)) for s in self.config.process_scales})
        if len(pairs)>self.config.max_p: raise FullGraphCapacityError('full.P_capacity')
        spans=torch.stack([torch.stack((bars[i,0],bars[j,1])) for i,j in pairs])
        state=self.P_geometry(_geometry(spans))
        state=state+self.round_layers[0].process_reads_h(state,H,spans,Hspans)
        return state,spans,tuple(pairs)

    @staticmethod
    def _overlap_candidates(queries, bank, limit, *, penalize_distance=False):
        """Bounded score blocks, one host transfer per block, stable tie policy.

        Avoid a CUDA synchronization/kernel series for every physical note.
        Only candidate selection detaches; returned geometry retains gradients.
        """
        if not len(bank):
            for _ in range(len(queries)):yield [], 0
            return
        chunk=max(1,min(128,1048576//len(bank)))
        bank=bank.detach();queries=queries.detach()
        for start in range(0,len(queries),chunk):
            q=queries[start:start+chunk]
            overlap=torch.minimum(q[:,None,1],bank[None,:,1])-torch.maximum(q[:,None,0],bank[None,:,0])
            score=overlap
            if penalize_distance:
                score=score-(q.mean(-1)[:,None]-bank.mean(-1)[None,:]).abs()*.01
            score=score.masked_fill(overlap<=0,float('-inf'))
            rank=torch.argsort(score,dim=-1,descending=True,stable=True)[:,:limit].sort(dim=-1).values
            keep=overlap.gather(1,rank)>0
            totals=(overlap>0).sum(-1).cpu().tolist()
            for candidates,valid,total in zip(rank.cpu().tolist(),keep.cpu().tolist(),totals):
                selected=[j for j,ok in zip(candidates,valid) if ok]
                yield selected,total-len(selected)

    def _contexts(self, raw, H, Hspans, notes, previous=None):
        """Refresh geometric candidates; never infer semantic chord membership."""
        c=self.config;ns=raw['notes']['spans'];nidx=[];hidx=[];slots=[];keys=[];excluded=0
        for i,(candidates,dropped) in enumerate(self._overlap_candidates(ns,Hspans,c.context_h_neighbors)):
            excluded+=dropped
            if len(keys) + len(candidates) * c.context_slots > c.max_contexts:
                raise FullGraphCapacityError('full.h_capacity')
            for j in candidates:
                for slot in range(c.context_slots):nidx.append(i);hidx.append(j);slots.append(slot);keys.append((i,j,slot))
        if len(keys)>c.max_contexts: raise FullGraphCapacityError('full.h_capacity')
        ni=torch.tensor(nidx,dtype=torch.long,device=H.device);hi=torch.tensor(hidx,dtype=torch.long,device=H.device)
        si=torch.tensor(slots,dtype=torch.long,device=H.device)
        initial=self.context_initial(torch.cat((notes[ni],H[hi]),-1))+self.context_slots[si]
        spans=torch.stack((torch.maximum(ns[ni,0],Hspans[hi,0]),torch.minimum(ns[ni,1],Hspans[hi,1])),-1)
        state=initial+self.h_geometry(_geometry(spans))
        local=spans.new_tensor([[.001,.999]]).expand(len(spans),-1)
        if previous is not None:
            lookup={key:i for i,key in enumerate(previous['keys'])};dest=[];source=[]
            for i,key in enumerate(keys):
                if key in lookup:dest.append(i);source.append(lookup[key])
            if dest:
                di=torch.tensor(dest,device=H.device);sr=torch.tensor(source,device=H.device)
                state=state.index_copy(0,di,previous['states'][sr]+self.h_geometry(_geometry(spans[di])))
                local=local.index_copy(0,di,previous['local_spans'][sr])
        bounds=spans
        spans=bounds[:,:1]+(bounds[:,1:]-bounds[:,:1])*local
        return dict(states=state,spans=spans,bounds=bounds,local_spans=local,note_index=ni,H_index=hi,keys=tuple(keys),excluded_pairs=excluded)

    def _participants(self, Pspans, Hspans, hspans):
        pi=[];ei=[];levels=[];excluded=0
        for level,spans in [('H',Hspans),('h',hspans)]:
            for i,(chosen,dropped) in enumerate(self._overlap_candidates(Pspans,spans,self.config.participant_neighbors,penalize_distance=True)):
                excluded+=dropped
                if len(pi)+len(chosen)>self.config.max_participants:
                    raise FullGraphCapacityError('full.participant_capacity')
                pi.extend([i]*len(chosen));ei.extend(chosen);levels.extend([level]*len(chosen))
        if len(pi)>self.config.max_participants:raise FullGraphCapacityError('full.participant_capacity')
        device=Pspans.device
        return dict(P_index=torch.tensor(pi,dtype=torch.long,device=device),endpoint_index=torch.tensor(ei,dtype=torch.long,device=device),endpoint_level=tuple(levels),excluded=excluded)

    @staticmethod
    def _edge_values(part, P, H, h, Ps, Hs, hs):
        pi,ei=part['P_index'],part['endpoint_index'];end=P.new_zeros((len(pi),P.shape[-1]));sp=Ps.new_zeros((len(pi),2))
        for level,states,spans in [('H',H,Hs),('h',h,hs)]:
            selected=torch.tensor([i for i,k in enumerate(part['endpoint_level']) if k==level],dtype=torch.long,device=P.device)
            if selected.numel():
                end=end.index_copy(0,selected,states[ei[selected]]);sp=sp.index_copy(0,selected,spans[ei[selected]])
        return torch.cat((P[pi],end,Ps[pi],sp),-1)

    def _relation_candidates(self, levels):
        sources=[];targets=[];features=[];total_possible=0
        for level,state,spans in levels:
            count=len(state);total_possible+=count*max(0,count-1)
            if count<2:continue
            # One temporal sort, then bounded neighbours. Never materialise or sort
            # an M by M distance matrix for note contexts.
            centers=spans.mean(-1).detach().cpu().tolist()
            order=sorted(range(count),key=lambda i:(centers[i],i))
            inverse={i:p for p,i in enumerate(order)}
            for i in range(count):
                position=inverse[i];left=position-1;right=position+1;near=[]
                for _ in range(min(self.config.relation_neighbors,count-1)):
                    if left<0:chosen=right;right+=1
                    elif right>=count:chosen=left;left-=1
                    elif (abs(centers[order[left]]-centers[i]),order[left]) <= (abs(centers[order[right]]-centers[i]),order[right]):
                        chosen=left;left-=1
                    else:chosen=right;right+=1
                    near.append(order[chosen])
                # Logarithmic distant positions preserve cross-window access.
                step=1;far=[]
                while step<count:
                    for p in (position-step,position+step):
                        if 0<=p<count:far.append(order[p])
                    step*=2
                if len(far)>self.config.relation_far_neighbors:
                    budget=self.config.relation_far_neighbors
                    far=[far[round(k*(len(far)-1)/max(1,budget-1))] for k in range(budget)]
                indices=sorted(set(near+far)-{i})
                if len(sources)+len(indices)>self.config.max_relations:raise FullGraphCapacityError('full.relation_capacity')
                for j in indices:
                    sources.append(({'kind':level,'index':i},));targets.append(({'kind':level,'index':j},))
                if indices:
                    idx=torch.tensor(indices,dtype=torch.long,device=state.device)
                    features.append(torch.cat((state[i].expand(len(idx),-1),state[idx],spans[i].expand(len(idx),-1),spans[idx]),-1))
        w=next(self.parameters());x=torch.cat(features) if features else w.new_empty((0,self.config.hidden*2+4))
        return dict(sources=tuple(sources),targets=tuple(targets),logits={name:head(x).flatten() for name,head in zip(self.relation_types,self.relation_heads)},candidate_count=len(sources),possible_singletons=total_possible)

    def forward(self, raw):
        return self._forward(raw)

    @torch.no_grad()
    def oracle_forward(self, raw, H_spans, P_spans):
        """Geometry-only diagnostic; ordinary inference never receives targets.

        This fixes both proposal banks and boundaries. It is a conditional
        diagnostic, not the measured gain from a boundary-only intervention.
        """
        return self._forward(raw, (H_spans, P_spans))

    def _forward(self, raw, oracle=None):
        self._validate(raw)
        if oracle is None:
            bars=len(raw['bar_spans']);c=self.config
            if len({(i,min(bars-1,i+s-1)) for i in range(bars) for s in c.process_scales})>c.max_p:
                raise FullGraphCapacityError('full.P_capacity')
        H,Hspans,notes,Hids,proposed=self._encode(raw)
        if oracle is None:
            P,Pspans,Pids=self._process_bank(raw,H,Hspans)
        else:
            for name,value,cap in [('H',oracle[0],self.config.max_h),('P',oracle[1],self.config.max_p)]:
                if (not isinstance(value,Tensor) or value.ndim!=2 or value.shape[1]!=2
                    or value.dtype!=H.dtype or value.device!=H.device or not torch.isfinite(value).all()
                    or ((value[:,0]<0)|(value[:,1]>1)|(value[:,0]>=value[:,1])).any()):
                    raise ValueError('full.oracle_geometry:'+name)
                if len(value)>cap:raise FullGraphCapacityError('full.oracle_capacity:'+name)
            q=self.H_geometry(_geometry(oracle[0]))
            H=q+self.round_layers[0].global_h(q,H,oracle[0],Hspans)
            Hspans=oracle[0];Hids=tuple((-1,i) for i in range(len(H)))
            Pspans=oracle[1];P=self.P_geometry(_geometry(Pspans))
            P=P+self.round_layers[0].process_reads_h(P,H,Pspans,Hspans)
            Pids=tuple((-1,i) for i in range(len(P)))
        context=None;history=[]
        for layer in self.round_layers:
            history.append(dict(H_spans=Hspans,P_spans=Pspans,H_states=H,P_states=P))
            context=self._contexts(raw,H,Hspans,notes,context);h=context['states'];ni=context['note_index'];hi=context['H_index']
            hsoft=self.note_fields.soft(h) if self.config.soft_feedback else torch.zeros_like(h)
            h=layer.h_bottom(h,notes[ni],H[hi],hsoft)
            bottom=_scatter(layer.h_to_H(h),hi,len(H))
            H=layer.H_bottom(H,bottom,layer.global_h(H,H,Hspans,Hspans),self._semantic_H(H))
            part=self._participants(Pspans,Hspans,context['spans'])
            edge=self._edge_values(part,P,H,h,Pspans,Hspans,context['spans']);weights=layer.participant_score(edge).flatten().sigmoid()
            messages=P.new_zeros((len(weights),P.shape[-1]))
            for level,states,projection in [('H',H,layer.H_to_P),('h',h,layer.h_to_P)]:
                ix=torch.tensor([i for i,k in enumerate(part['endpoint_level']) if k==level],dtype=torch.long,device=P.device)
                if ix.numel():messages=messages.index_copy(0,ix,projection(states[part['endpoint_index'][ix]]))
            psoft=self.process_fields.soft(P) if self.config.soft_feedback else torch.zeros_like(P)
            P=layer.P_bottom(P,_scatter(messages,part['P_index'],len(P),weights),
                layer.process_reads_h(P,H,Pspans,Hspans)+layer.global_p(P,P,Pspans,Pspans),psoft)
            feedback={}
            for level,count,projection in [('H',len(H),layer.P_to_H),('h',len(h),layer.P_to_h)]:
                ix=torch.tensor([i for i,k in enumerate(part['endpoint_level']) if k==level],dtype=torch.long,device=P.device)
                feedback[level]=_scatter(projection(P[part['P_index'][ix]]),part['endpoint_index'][ix],count,weights[ix])
            H=layer.H_top(H,feedback['H'],self._semantic_H(H))
            h=layer.h_top(h,feedback['h'],H[hi],notes[ni])
            notes=layer.note_top(notes,_scatter(layer.h_to_note(h),ni,len(notes)))
            if oracle is None:
                Hspans=_refine_spans(Hspans,layer.H_delta(H));Pspans=_refine_spans(Pspans,layer.P_delta(P))
            # h endpoints are learned within its raw note/H overlap, preserving observations.
            local=_refine_spans(context['local_spans'],layer.h_delta(h))
            bounds=context['bounds'];hspans=bounds[:,:1]+(bounds[:,1:]-bounds[:,:1])*local
            context.update(states=h,spans=hspans,local_spans=local)
        # Rebuild participants after geometry changes. Every final h fits final H and note.
        previous=context;context=self._contexts(raw,H,Hspans,notes,previous)
        h=context['states'];bounds=context['bounds'];local=_refine_spans(context['local_spans'],self.round_layers[-1].h_delta(h))
        context['spans']=bounds[:,:1]+(bounds[:,1:]-bounds[:,:1])*local
        part=self._participants(Pspans,Hspans,context['spans'])
        part['existence_logits']=self.participant_head(self._edge_values(part,P,H,h,Pspans,Hspans,context['spans'])).flatten()
        relations=self._relation_candidates([('H',H,Hspans),('h',h,context['spans'])])
        return dict(H=dict(spans=Hspans,states=H,object_logits=self.constructor.object_head(H).flatten(),field_logits=self._H_fields(H),ids=tuple(f'H:{t}:{a}' for t,a in Hids)),
            h=dict(spans=context['spans'],states=h,note_index=context['note_index'],H_index=context['H_index'],field_logits=self.note_fields(h),ids=tuple(f'h:{n}:{j}:{s}' for n,j,s in context['keys'])),
            P=dict(spans=Pspans,states=P,object_logits=self.P_object(P).flatten(),field_logits=self.process_fields(P),ids=tuple(f'P:{a}:{b}' for a,b in Pids)),
            participants=part,relations=relations,notes=dict(states=notes,ids=raw['note_ids'],spans=raw['notes']['spans']),rounds=tuple(history),geometry='predicted' if oracle is None else 'oracle',oracle_levels=() if oracle is None else ('H','P'),
            coverage=dict(proposed_H=proposed,retained_H=len(H),P=len(P),h=len(h),context_pair_budget_exclusions=context['excluded_pairs'],participant_budget_exclusions=part['excluded'],
                          point_notes_without_h=int((raw['notes']['spans'][:,0]==raw['notes']['spans'][:,1]).sum()),raw=raw['coverage']))
