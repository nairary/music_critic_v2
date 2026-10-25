"""Immutable TRAIN-only HCM preparation from authenticated canonical sources.

This preparer preserves caller work identities and reports exclusions. Exact
observation unions within the admitted TRAIN sample do not establish complete
cross-corpus deduplication, nor do they open any held-out musical payload.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Mapping

from music_critic.data import loads_piece, dumps_piece
from music_critic.data.serialization import canonical_json_sha256
from .codec import EventCodec
from .contracts import CodecProfile, ContextProfile, MaskQuery
from .data import DataManifest, seal_manifest
from .masking import split_masked_piece, build_observed_tiles
from .graph import build_tile_graph
from .observed_view import ObservedMusicView


@dataclass(frozen=True)
class SourceCandidate:
    record_id: str
    source: str
    piece_id: str
    source_group_id: str
    work_id: str
    split: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PrepareConfig:
    window_bars: int = 16
    max_notes_per_window: int = 1024
    max_windows_per_source: int = 4
    mask_lengths: tuple[int, ...] = (1, 2, 4)
    context: ContextProfile = ContextProfile()
    codec: CodecProfile = CodecProfile()

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.window_bars,self.max_notes_per_window,self.max_windows_per_source)):
            raise ValueError('pll.prepare_positive_budgets')
        if self.window_bars > self.context.context_bars or not self.mask_lengths or any(
                k not in (1,2,4) for k in self.mask_lengths):
            raise ValueError('pll.prepare_geometry_budget')


def _json_once(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload,stream,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n')


def _observation(piece):
    def q(v): return [v.num,v.den]
    tracks = []
    for track in piece.tracks:
        events = sorted((n.pitch,q(n.onset_qn),q(n.duration_qn),n.is_percussion,n.is_grace)
                        for n in piece.notes if n.track_id == track.track_id)
        tracks.append(events)
    return canonical_json_sha256(dict(tracks=sorted(tracks),
        meter=[(q(m.onset_qn),m.numerator,m.denominator) for m in piece.meter_events],
        bars=[(q(b.start_qn),q(b.duration_qn)) for b in piece.bars]))


def _admit_window(piece, config):
    """All mask/codec geometries plus a conservative full-visible graph bound.

    Removing events cannot increase the number of note/onset/slice nodes or
    containment/active/hold edges. Sequential note/onset/slice edges can change
    identity, but their counts cannot increase. Therefore the full-visible tile
    graph bounds every masked tile's resource counts under the same geometry.
    """
    codec = EventCodec(config.codec)
    tracks = tuple(t.track_id for t in piece.tracks)
    choices = tuple(dict.fromkeys((tracks, *((t,) for t in tracks))))
    queries, max_events, graph_count = 0, 0, 0
    # Every possible mask used by the sampler is checked before admission.
    for length in config.mask_lengths:
        for start in range(len(piece.bars)-length+1):
            first,last=piece.bars[start],piece.bars[start+length-1]
            for selected in choices:
                ex=split_masked_piece(piece,MaskQuery(first.start_qn,
                    last.start_qn+last.duration_qn,selected),config.context)
                codec.encode(ex.targets,span=ex.span,tracks=ex.tracks)
                max_events=max(max_events,len(ex.targets))
                queries+=1
    if not queries:
        raise ValueError('pll.prepare_no_query')
    first,last=piece.bars[0],piece.bars[-1]
    names={t.track_id:f'track:{i}' for i,t in enumerate(piece.tracks)}
    blank=split_masked_piece(replace(piece,notes=()),
        MaskQuery(first.start_qn,last.start_qn+last.duration_qn,tracks),config.context).context.piece
    full_piece=replace(blank,notes=tuple(replace(n,note_id=f'note:{i:06d}',
        track_id=names[n.track_id],provenance_id=None,spelling_step=None,spelling_alter=None,
        staff=None,voice=None,articulations=None,dynamic=None,source_onset_ticks=None,
        source_duration_ticks=None,source_onset_seconds=None,source_duration_seconds=None)
        for i,n in enumerate(piece.notes)))
    full=ObservedMusicView(full_piece,MaskQuery(first.start_qn,last.start_qn+last.duration_qn,tuple(names.values())),
                           tuple(range(len(piece.bars))))
    for tile in build_observed_tiles(full,config.context):
        build_tile_graph(tile,config.context)
        graph_count+=1
    return dict(queries_checked=queries,unique_tile_graphs_checked=graph_count,
                graph_bound='full-visible node and edge counts bound every event subset',
                max_target_events=max_events)


def prepare_sources(candidates, output, *, config=PrepareConfig(),
                    source_weights: Mapping[str,float], split_provenance: str,
                    input_bindings=()):
    """Prepare a new output directory; caller must register any real-data run.

    Windows with clipped notes are excluded rather than changing targets. Empty
    source windows remain explicit exclusions; empty sampled masks in admitted
    windows remain valid. This is a bounded engineering-pilot population.
    """
    # Whole-song runtime reuses only the source contract and JSON writer;
    # importing those must not load the old windowed/SSL training pipeline.
    from music_critic.pdmx_pipeline.windowing import WindowConfig, iter_bar_windows
    candidates=tuple(candidates)
    if not split_provenance.strip() or len({c.record_id for c in candidates})!=len(candidates):
        raise ValueError('pll.prepare_identity_required')
    # Inspect all declared groups before opening any payload, including later rows.
    groups={}
    for c in candidates:
        if any(not isinstance(v,str) or not v.strip() for v in
               (c.record_id,c.source,c.piece_id,c.source_group_id,c.work_id)):
            raise ValueError('pll.prepare_identity_required')
        groups.setdefault(c.work_id,set()).add(c.split)
        groups.setdefault((c.source,c.source_group_id),set()).add(c.split)
    if any(len(splits)>1 for splits in groups.values()):
        raise ValueError('pll.prepare_group_crosses_split')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'pieces').mkdir()
    started=time.monotonic()
    report=dict(schema='PLLPreparationReceipt@1',status='running',config=asdict(config),
        input_bindings=list(input_bindings),train_payloads_read=0,held_out_payloads_read=0,
        metadata_excluded_nontrain=sum(c.split!='train' for c in candidates),
        accesses=[],exclusions=[],admitted=[],cross_source_near_duplicate_coverage='unknown',
        full_corpus_claim=False)
    records=[]
    # Union exact TRAIN observations at source-work level, before windows are samples.
    parents={c.work_id:c.work_id for c in candidates if c.split=='train'}
    def root(work):
        while parents[work]!=work:
            parents[work]=parents[parents[work]]
            work=parents[work]
        return work
    seen_observations={}
    try:
        for c in sorted(candidates,key=lambda c:c.record_id):
            if c.split!='train': continue
            access=dict(record_id=c.record_id,path=c.path,declared_split=c.split,
                        observed_split=None,verified=False)
            report['accesses'].append(access)
            raw=Path(c.path).read_bytes()
            report['train_payloads_read']+=1
            if len(raw)!=c.size_bytes or sha256(raw).hexdigest()!=c.sha256:
                raise ValueError('pll.prepare_source_hash_mismatch')
            piece=loads_piece(raw.decode())
            access['observed_split']=piece.split
            if piece.split not in (None,'train'):
                report['held_out_payloads_read']+=1
                raise ValueError('pll.prepare_embedded_nontrain')
            if (piece.piece_id,piece.dataset_name,piece.source_group_id)!=(c.piece_id,c.source,c.source_group_id):
                raise ValueError('pll.prepare_source_identity_mismatch')
            access['verified']=True
            if c.source=='pop909_cl' and (any(n.channel==1 for n in piece.notes)
                    or any(t.channel==1 for t in piece.tracks)):
                raise ValueError('pll.prepare_pop909_annotation_channel_forbidden')
            piece=replace(piece,split='train',targets=(),annotations=(),key_signature_events=(),quality_flags=())
            # Raw geometry is inherited explicitly; no label-derived bar repairs.
            # Provenance stays in the artifact for canonical validation and is
            # removed by the existing mask-before-graph boundary at inference.
            admission=0
            windows=iter_bar_windows(piece,WindowConfig(bar_count=config.window_bars,
                stride=config.window_bars,max_notes_per_window=config.max_notes_per_window))
            for window in windows:
                identity=dict(record_id=c.record_id,bar_start=window.source_bar_start,
                              bar_stop=window.source_bar_stop)
                if admission>=config.max_windows_per_source:
                    report['exclusions'].append({**identity,'reason':'declared_window_cap'})
                    continue
                if window.piece is None or window.clipped_note_count:
                    report['exclusions'].append({**identity,'reason':window.reason or 'clipped_notes_forbidden'})
                    continue
                clip=window.piece
                try:
                    coverage=_admit_window(clip,config)
                except ValueError as exc:
                    report['exclusions'].append({**identity,'reason':str(exc)})
                    continue
                observation=_observation(clip)
                if observation in seen_observations:
                    previous=seen_observations[observation]
                    left,right=root(c.work_id),root(previous)
                    parents[max(left,right)]=min(left,right)
                else: seen_observations[observation]=c.work_id
                data=dumps_piece(clip).encode()
                digest=sha256(data).hexdigest()
                relative='pieces/'+digest+'.json'
                target=output/relative
                if not target.exists(): target.write_bytes(data)
                records.append(dict(record_id=c.record_id+':bars:'+str(window.source_bar_start)+'-'+str(window.source_bar_stop),
                    source=c.source,work_id=c.work_id,source_group_id=clip.source_group_id,
                    piece_id=clip.piece_id,split='train',path=relative,sha256=digest,size_bytes=len(data)))
                report['admitted'].append({**identity,**coverage,'observation_sha256':observation})
                admission+=1
            print(json.dumps(dict(source=c.source,record_id=c.record_id,admitted=admission,
                train_reads=report['train_payloads_read'])),flush=True)
        for record in records: record['work_id']=root(record['work_id'])
        if not records or set(source_weights)!={r['source'] for r in records}:
            raise ValueError('pll.prepare_all_requested_sources_required')
        manifest=seal_manifest(dict(schema='PLLDataManifest@1',records=records,
            source_weights=dict(source_weights),group_before_window=True,
            cross_source_components_declared=True,split_provenance=split_provenance))
        DataManifest(output,manifest)
        _json_once(output/'manifest.json',manifest)
        report.update(status='completed',manifest_fingerprint=manifest['fingerprint'],
            records=len(records),works=len({r['work_id'] for r in records}),
            source_counts=dict(Counter(r['source'] for r in records)),
            source_work_counts={s:len({r['work_id'] for r in records if r['source']==s}) for s in source_weights})
        return report
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        _json_once(output/'prepare-receipt.json',report)
