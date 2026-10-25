"""Bounded temporal tile graphs; artificial crop boundaries are never attacks.

C3 containment assigns carry-in notes to the tile's first bar so every note
has a local owner. This is explicitly censored containment, not onset evidence.
Exact original times are in ObservedTile.note_geometry; the note duration
feature is restored and unobserved onset positions are masked unavailable.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from fractions import Fraction
from statistics import fmean

import torch
from torch_geometric.data import Batch

from music_critic.data.raw_time_readout_slots import ReadoutSlots
from music_critic.graph import build_raw_graph, validate_raw_graph
from music_critic.experiments.analysisgnn.temporal_slices.graph import (
    TemporalGraphBatch, EDGES, FEATURES, NODES, owner,
)
from .contracts import ContextProfile
from .observed_view import ObservedTile, UnsupportedContextError


def _time(value):
    return value.to_fraction()


def _set_cont(graph,node,name,row,value,available=True):
    i=graph[node].cont_feature_names.index(name)
    graph[node].x_cont[row,i]=value
    graph[node].x_cont_available[row,i]=available


def _tensor(pairs):
    return torch.tensor(pairs,dtype=torch.long).reshape(-1,2).t().contiguous()


def _correct_base(base,tile):
    """Remove false starts before the graph reaches any trainable operator."""
    piece=tile.piece
    onset_counts=Counter(n.onset_qn for n,g in zip(piece.notes,tile.note_geometry) if not g.left_censored)
    actual_onsets=set(onset_counts)
    ordered_onsets=sorted(actual_onsets)
    prefix_counts=[0]
    for t in ordered_onsets:
        prefix_counts.append(prefix_counts[-1]+onset_counts[t])
    surrogate_onsets=sorted({n.onset_qn for n in piece.notes})
    keep=torch.tensor([i for i,t in enumerate(surrogate_onsets) if t in actual_onsets],dtype=torch.long)
    mapping=torch.full((len(surrogate_onsets),),-1,dtype=torch.long)
    mapping[keep]=torch.arange(len(keep))
    store=base['onset']
    for name in ('x_cat','x_cat_available','x_cont','x_cont_available','candidate_slot'):
        store[name]=store[name][keep]
    store.entity_id=tuple(store.entity_id[i] for i in keep.tolist())
    store.num_nodes=len(keep)
    for edge_type in base.edge_types:
        source,_,target=edge_type
        edge=base[edge_type].edge_index
        valid=torch.ones(edge.shape[1],dtype=torch.bool)
        if source=='onset': valid &= mapping[edge[0]]>=0
        if target=='onset': valid &= mapping[edge[1]]>=0
        edge=edge[:,valid].clone()
        if source=='onset': edge[0]=mapping[edge[0]]
        if target=='onset': edge[1]=mapping[edge[1]]
        base[edge_type].edge_index=edge
    carry=torch.tensor([g.left_censored for g in tile.note_geometry],dtype=torch.bool)
    for edge_type in (('onset','starts_note','note'),('note','in_onset','onset')):
        e=base[edge_type].edge_index
        note_axis=1 if edge_type[0]=='onset' else 0
        base[edge_type].edge_index=e[:,~carry[e[note_axis]]]
    for i,g in enumerate(tile.note_geometry):
        _set_cont(base,'note','duration_qn',i,float(_time(g.duration)))
        if g.left_censored:
            # Do not turn the surrogate onset zero into an observed attack.
            _set_cont(base,'note','onset_qn',i,0.,False)
            _set_cont(base,'note','position_in_bar_qn',i,0.,False)
    _set_cont(base,'song','onset_count',0,len(actual_onsets))
    for node,rows in (('bar',piece.bars),('beat',piece.beats)):
        for i,row in enumerate(rows):
            a,b=row.start_qn,row.start_qn+row.duration_qn
            lo,hi=bisect_left(ordered_onsets,a),bisect_left(ordered_onsets,b)
            count=prefix_counts[hi]-prefix_counts[lo]
            _set_cont(base,node,'starting_note_count',i,count)
            if node=='bar':
                _set_cont(base,node,'onset_count',i,hi-lo)
    for i,t in enumerate(ordered_onsets):
        count=onset_counts[t]
        _set_cont(base,'onset','starting_note_count',i,count)
        beat=next((b for b in piece.beats if b.start_qn<=t<b.start_qn+b.duration_qn),None)
        _set_cont(base,'onset','onsets_in_beat',i,
                  bisect_left(ordered_onsets,beat.start_qn+beat.duration_qn)-bisect_left(ordered_onsets,beat.start_qn) if beat else 0.,bool(beat))
    next_pairs=[]
    for i,track in enumerate(piece.tracks):
        indices=[j for j,n in enumerate(piece.notes) if n.track_id==track.track_id]
        if indices:
            _set_cont(base,'track','mean_duration_qn',i,fmean(float(_time(tile.note_geometry[j].duration)) for j in indices))
        starts={}
        for j in indices:
            if not tile.note_geometry[j].left_censored:
                starts.setdefault(piece.notes[j].onset_qn,[]).append(j)
        _set_cont(base,'track','polyphony_ratio',i,
                  sum(len(v)>1 for v in starts.values())/len(starts) if starts else 0.,bool(starts))
        indices.sort(key=lambda j:(tile.note_geometry[j].onset,piece.notes[j].pitch,
                                   tile.note_geometry[j].duration,piece.notes[j].note_id))
        next_pairs.extend(zip(indices,indices[1:]))
    base['note','next_in_track','note'].edge_index=_tensor(next_pairs)
    base['note','previous_in_track','note'].edge_index=_tensor(next_pairs).flip(0)
    validate_raw_graph(base)
    return base


def build_tile_graph(tile:ObservedTile,profile:ContextProfile=ContextProfile())->TemporalGraphBatch:
    """Build one tile, enforcing note/slice/total directed-edge budgets.

    Both real endpoints of a long visible note remain in its geometry; only
    endpoints inside this tile create temporal nodes or starts/ends edges.
    Boundary slices from the known meter remain even when no notes are visible.
    """
    piece=tile.piece
    if len(piece.notes)>profile.max_notes_per_tile:
        raise UnsupportedContextError('pll.tile_note_limit')
    if len(piece.tracks)>profile.max_tracks:
        raise UnsupportedContextError('pll.context_track_limit')
    duration=_time(tile.global_end-tile.global_start)
    zero=Fraction(0)
    starts,ends=[],[]
    for g in tile.note_geometry:
        a=_time(g.onset-tile.global_start)
        starts.append(a)
        ends.append(a+_time(g.duration))
    actual_starts={s for s in starts if zero<=s<duration}
    actual_ends={s for s in ends if zero<=s<=duration}
    beat_bounds={_time(b.start_qn) for b in piece.beats}|{_time(b.start_qn+b.duration_qn) for b in piece.beats}
    bar_bounds={_time(b.start_qn) for b in piece.bars}|{_time(b.start_qn+b.duration_qn) for b in piece.bars}
    times=sorted({zero,duration}|actual_starts|actual_ends|beat_bounds|bar_bounds)
    if len(times)>profile.max_slices_per_tile:
        raise UnsupportedContextError('pll.tile_slice_limit')
    base=_correct_base(build_raw_graph(piece),tile)
    count=sum(base[e].edge_index.shape[1] for e in base.edge_types)
    if count>profile.max_edges_per_tile:
        raise UnsupportedContextError('pll.tile_edge_limit')
    edges={e:[] for e in EDGES}
    def add(source,relation,target,a,b):
        nonlocal count
        count+=2
        if count>profile.max_edges_per_tile:
            raise UnsupportedContextError('pll.tile_edge_limit')
        edges[source,relation,target].append((a,b))
        edges[target,'rev_'+relation,source].append((b,a))
    time_index={t:i for i,t in enumerate(times)}
    for i,(a,b) in enumerate(zip(starts,ends)):
        if a in actual_starts: add('note','starts','slice',i,time_index[a])
        if b in actual_ends: add('note','ends','slice',i,time_index[b])
        for j in range(bisect_right(times,a),bisect_left(times,b)):
            add('note','holds','slice',i,j)
    for i,t in enumerate(sorted(actual_starts)):
        add('onset','coincides','slice',i,time_index[t])
    spans={node:([_time(r.start_qn) for r in rows],[_time(r.start_qn+r.duration_qn) for r in rows])
           for node,rows in (('beat',piece.beats),('bar',piece.bars))}
    meters={m.meter_event_id:m for m in piece.meter_events}
    features=[]
    for j,t in enumerate(times):
        owners={node:owner(t,*spans[node]) for node in ('beat','bar')}
        for node,i in owners.items():
            if i is not None:add(node,'contains','slice',i,j)
        beat=piece.beats[owners['beat']] if owners['beat'] is not None else None
        bar=piece.bars[owners['bar']] if owners['bar'] is not None else None
        meter=meters[bar.meter_event_id] if bar else None
        features.append([
            float((t-_time(beat.start_qn))/_time(beat.duration_qn)) if beat else 0.,
            float(t-_time(bar.start_qn)+_time(bar.metric_offset_qn)) if bar else 0.,
            beat.index_in_bar if beat else 0.,meter.numerator if meter else 0.,meter.denominator if meter else 0.,
            float(t-times[j-1]) if j else 0.,float(times[j+1]-t) if j+1<len(times) else 0.,
            t in actual_starts,t in actual_ends,t in beat_bounds,beat is not None,bar is not None,
            bar.is_pickup if bar else False,bar.is_incomplete if bar else False])
        if j:add('slice','next','slice',j-1,j)
    slots=ReadoutSlots(tuple((t,f'rawtime:{t.numerator}_{t.denominator}') for t in times),
                       frozenset(b.beat_id for b in piece.beats))
    ids={node:tuple(slot_id for _,slot_id in slots.time_slots) if node=='slice' else base[node].entity_id for node in NODES}
    result=TemporalGraphBatch(Batch.from_data_list([base]),
        torch.tensor(features,dtype=torch.float32).reshape(-1,len(FEATURES)),
        {e:_tensor(v) for e,v in edges.items()},
        {node:tuple((0,i) for i in ids[node]) for node in NODES},
        {node:(0,len(ids[node])) for node in NODES},(slots,))
    result.validate()
    return result
