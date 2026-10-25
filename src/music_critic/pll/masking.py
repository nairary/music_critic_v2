"""Mask raw events before deriving any graph, local feature or identifier."""
from __future__ import annotations

from dataclasses import replace

from music_critic.data.schema import CanonicalPiece, PieceMetadata, ProvenanceRecord
from music_critic.data.timing import RationalTime as Q
from .contracts import ContextProfile, Event, MaskQuery
from .observed_view import (MaskedExample, NoteGeometry, ObservedMusicView,
                            ObservedTile, UnsupportedContextError)


def _intersects(note, start, end):
    return (start <= note.onset_qn < end if note.duration_qn == Q(0)
            else note.onset_qn < end and note.onset_qn + note.duration_qn > start)


def _note_key(note):
    return (note.onset_qn, int(note.track_id.split(':')[-1]),
            note.pitch, note.duration_qn, note.is_percussion, note.is_grace, -1 if note.velocity is None else note.velocity,
            -1 if note.channel is None else note.channel, -1 if note.program is None else note.program)


def split_masked_piece(piece: CanonicalPiece, query: MaskQuery,
                       profile: ContextProfile = ContextProfile()) -> MaskedExample:
    """Return disjoint predictor/target records for an onset-anchored query.

    Explicit bars/meters are declared known geometry. Source `duration_qn`,
    note tails, semantic segmentation, IDs and annotation targets do not select
    the context window. Upstream adapters must not derive the declared metric
    geometry from the particular hidden filling under comparison.
    """
    if len(piece.tracks) > profile.max_tracks:
        raise UnsupportedContextError('pll.context_track_limit')
    track_map = {t.track_id: i for i, t in enumerate(piece.tracks)}
    if len(track_map) != len(piece.tracks) or any(t not in track_map for t in query.track_ids):
        raise ValueError('pll.unknown_or_duplicate_track')
    if not piece.bars:
        raise UnsupportedContextError('pll.metric_geometry_required')
    bars = piece.bars
    if any(b.duration_qn <= Q(0) for b in bars) or any(a.start_qn+a.duration_qn != b.start_qn for a,b in zip(bars,bars[1:])):
        raise UnsupportedContextError('pll.contiguous_metric_geometry_required')
    hits = [i for i,b in enumerate(bars) if b.start_qn < query.end and b.start_qn+b.duration_qn > query.start]
    if (not hits or query.start < bars[0].start_qn
            or query.end > bars[-1].start_qn + bars[-1].duration_qn):
        raise UnsupportedContextError('pll.query_outside_declared_geometry')
    if len(hits) > profile.context_bars:
        raise UnsupportedContextError('pll.query_context_limit')
    spare = profile.context_bars - len(hits)
    left = max(0, hits[0] - spare // 2)
    right = min(len(bars), left + profile.context_bars)
    left = max(0, right-profile.context_bars)
    chosen = bars[left:right]
    # The inherited canonical builder admits some irregular source measures
    # only through adapter repair provenance. Such labels cannot enter this
    # target-free view. Until a dedicated irregular-tile validator exists,
    # decline their *geometry* explicitly so scoring retains unsupported slots.
    meters = {m.meter_event_id: m for m in piece.meter_events}
    meter_changes = {m.onset_qn for m in piece.meter_events}
    for ordinal in range(left, right):
        bar = bars[ordinal]
        meter = meters[bar.meter_event_id]
        nominal = Q(4 * meter.numerator, meter.denominator)
        unit = Q(4, meter.denominator)
        off_grid = (bar.metric_offset_qn.to_fraction() / unit.to_fraction()).denominator != 1
        irregular_incomplete = (bar.is_incomplete and not bar.is_pickup and (
            bar.metric_offset_qn != Q(0) or bar.duration_qn >= nominal or
            (ordinal != len(bars)-1 and bar.start_qn + bar.duration_qn not in meter_changes)))
        if off_grid or irregular_incomplete:
            raise UnsupportedContextError('pll.irregular_metric_geometry')
    start, end = chosen[0].start_qn, chosen[-1].start_qn+chosen[-1].duration_qn
    targets, visible = [], []
    masked = set(query.track_ids)
    for note in piece.notes:
        if note.track_id not in track_map:
            raise ValueError('pll.note_unknown_track')
        if note.track_id in masked and query.start <= note.onset_qn < query.end:
            targets.append(Event(note.onset_qn-query.start, track_map[note.track_id],
                                 2*int(note.is_percussion)+int(note.is_grace), note.pitch, note.duration_qn))
        elif _intersects(note, start, end):
            visible.append(replace(note, track_id=f'track:{track_map[note.track_id]}',
                spelling_step=None, spelling_alter=None, staff=None, voice=None,
                articulations=None, dynamic=None, source_onset_ticks=None,
                source_duration_ticks=None, source_onset_seconds=None,
                source_duration_seconds=None, provenance_id=None))
    visible = tuple(replace(n, note_id=f'note:{i:06d}') for i,n in enumerate(sorted(visible,key=_note_key)))
    bar_ids = {b.bar_id: f'bar:{i}' for i,b in enumerate(chosen)}
    # Preserve meter schedule independently of any removed note or annotation.
    meter_ids = {m.meter_event_id: f'meter:{i}' for i,m in enumerate(piece.meter_events)}
    clean_bars = tuple(replace(b,bar_id=bar_ids[b.bar_id],index=i,
        meter_event_id=meter_ids[b.meter_event_id],display_number=None,provenance_id=None)
        for i,b in enumerate(chosen))
    clean = replace(piece,piece_id='piece:pll',dataset_name='pll',source_group_id='pll',
        split=None,source_path=None,source_resolution=None,duration_qn=end,
        metadata=PieceMetadata('synthetic',None,None,None,None,None,None,None,None),
        tracks=tuple(replace(t,track_id=f'track:{i}',source_track_index=i,name=None,
            instrument_name=None,provenance_id=None) for i,t in enumerate(piece.tracks)),
        notes=visible,bars=clean_bars,
        beats=tuple(replace(b,beat_id=f'beat:{i}',bar_id=bar_ids[b.bar_id],
            meter_event_id=meter_ids[b.meter_event_id],provenance_id=None)
            for i,b in enumerate(b for b in piece.beats if b.bar_id in bar_ids)),
        meter_events=tuple(replace(m,meter_event_id=meter_ids[m.meter_event_id],provenance_id=None) for m in piece.meter_events),
        tempo_events=tuple(replace(t,tempo_event_id=f'tempo:{i}',provenance_id=None) for i,t in enumerate(piece.tempo_events)),
        key_signature_events=(),annotations=(),targets=(),provenance=(),quality_flags=())
    clean_query = MaskQuery(query.start,query.end,tuple(f'track:{track_map[t]}' for t in query.track_ids))
    return MaskedExample(ObservedMusicView(clean,clean_query,tuple(range(left,right))),
                         tuple(sorted(targets,key=lambda e:e.key)), tuple(sorted(track_map[t] for t in query.track_ids)))


def _schedule(events,start,end):
    prior = [e for e in events if e.onset_qn <= start]
    if not prior:
        raise UnsupportedContextError('pll.effective_raw_schedule_required')
    selected = [replace(prior[-1],onset_qn=Q(0))]
    selected.extend(replace(e,onset_qn=e.onset_qn-start) for e in events if start<e.onset_qn<end)
    return tuple(selected)


def build_observed_tiles(view: ObservedMusicView,
                         profile: ContextProfile = ContextProfile()) -> tuple[ObservedTile,...]:
    """Fixed bar tiles with halo and bounded surrogate geometry for C3.

    Surrogate clipping establishes valid containment only. `build_tile_graph`
    removes clipped attacks/releases and restores original note duration.
    Consumers must use note_geometry for the actual event geometry.
    """
    piece, result = view.piece, []
    if len(piece.bars)>profile.context_bars:
        raise UnsupportedContextError('pll.context_bar_limit')
    if len(piece.tracks)>profile.max_tracks:
        raise UnsupportedContextError('pll.context_track_limit')
    for core_start in range(0,len(piece.bars),profile.core_bars):
        core_end = min(len(piece.bars),core_start+profile.core_bars)
        left,right = max(0,core_start-profile.halo_bars), min(len(piece.bars),core_end+profile.halo_bars)
        bars = piece.bars[left:right]
        start,end = bars[0].start_qn,bars[-1].start_qn+bars[-1].duration_qn
        selected = [n for n in piece.notes if _intersects(n,start,end)]
        if len(selected)>profile.max_notes_per_tile:
            raise UnsupportedContextError('pll.tile_note_limit')
        # Stable ordering after clipping is required by the canonical builder.
        pairs = [(replace(n,onset_qn=max(n.onset_qn,start)-start,
                          duration_qn=min(n.onset_qn+n.duration_qn,end)-max(n.onset_qn,start)),
                  NoteGeometry(n.note_id,n.onset_qn,n.duration_qn,n.onset_qn<start,n.onset_qn+n.duration_qn>end)) for n in selected]
        pairs.sort(key=lambda pair:(pair[0].onset_qn, int(pair[0].track_id.split(':')[-1]), pair[0].pitch, pair[0].duration_qn, pair[0].note_id))
        ids = {b.bar_id for b in bars}
        local = replace(piece,duration_qn=end-start,notes=tuple(p[0] for p in pairs),
            bars=tuple(replace(b,start_qn=b.start_qn-start,index=i) for i,b in enumerate(bars)),
            beats=tuple(replace(b,start_qn=b.start_qn-start) for b in piece.beats if b.bar_id in ids),
            meter_events=_schedule(piece.meter_events,start,end),
            tempo_events=_schedule(piece.tempo_events,start,end),
            provenance=(ProvenanceRecord('prov:pll','derivation','music_critic.pll',None,None,None,None,None,(),()),))
        result.append(ObservedTile(local,start,end,tuple(b.bar_id for b in piece.bars[core_start:core_end]),
            view.global_bar_indices[left:right],tuple(p[1] for p in pairs),view.query))
    return tuple(result)
