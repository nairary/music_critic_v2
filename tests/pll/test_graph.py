"""Synthetic C3-compatible tile graph invariants, no corpus evaluation."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece, RationalTime as Q
from music_critic.pll.contracts import ContextProfile, MaskQuery
from music_critic.pll.masking import split_masked_piece, build_observed_tiles
from music_critic.pll.graph import build_tile_graph
from music_critic.pll.observed_view import UnsupportedContextError


def piece():
    return load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')


def test_empty_visible_graph_has_metric_slices_and_no_note_events():
    p=replace(piece(),notes=())
    e=split_masked_piece(p,MaskQuery(Q(1),Q(5),tuple(t.track_id for t in p.tracks)))
    graph=build_tile_graph(build_observed_tiles(e.context)[0])
    graph.validate()
    assert graph.base['note'].num_nodes == graph.base['onset'].num_nodes == 0
    assert graph.slice_features.shape[0] == 6
    assert not graph.slice_features[:,7:9].any()


def test_clip_boundaries_are_not_attacks_or_releases():
    p=piece()
    n=replace(p.notes[0],onset_qn=Q(0),duration_qn=Q(20))
    p=replace(p,notes=(n,))
    e=split_masked_piece(p,MaskQuery(Q(1),Q(5),(p.tracks[1].track_id,)))
    profile=ContextProfile(core_bars=1,halo_bars=0)
    tiles=build_observed_tiles(e.context,profile)
    graph=build_tile_graph(tiles[1],profile)
    assert tiles[1].boundary_flags == ((1.0,1.0),)
    assert graph.base['onset'].num_nodes == 0
    assert graph.edges['note','starts','slice'].shape == (2,0)
    assert graph.edges['note','ends','slice'].shape == (2,0)
    assert graph.edges['note','holds','slice'].shape[1] == 5
    assert not graph.slice_features[:,7:9].any()
    col=graph.base['note'].cont_feature_names[0].index('duration_qn')
    assert graph.base['note'].x_cont[0,col] == 20
    graph.validate()


def test_hidden_filling_changes_do_not_change_any_graph_tensor():
    p=piece(); q=MaskQuery(Q(1),Q(5),tuple(t.track_id for t in p.tracks))
    changed=replace(p,notes=tuple(n for n in p.notes if n.onset_qn<Q(1)))
    a,b=[build_tile_graph(build_observed_tiles(split_masked_piece(x,q).context)[0]) for x in (p,changed)]
    assert a.fingerprint() == b.fingerprint()


def test_graph_limits_reject_instead_of_truncating():
    p=piece(); q=MaskQuery(Q(1),Q(5),tuple(t.track_id for t in p.tracks))
    tile=build_observed_tiles(split_masked_piece(p,q).context)[0]
    with pytest.raises(UnsupportedContextError,match='slice_limit'):
        build_tile_graph(tile,ContextProfile(max_slices_per_tile=1))
    with pytest.raises(UnsupportedContextError,match='edge_limit'):
        build_tile_graph(tile,ContextProfile(max_edges_per_tile=1))


def test_meter_change_survives_tile_origin_shift():
    p=piece()
    meters=(p.meter_events[0],replace(p.meter_events[0],meter_event_id='meter:new',onset_qn=Q(1),numerator=3))
    bars=(p.bars[0],replace(p.bars[1],duration_qn=Q(3),meter_event_id='meter:new'))
    beats=tuple(replace(b,meter_event_id='meter:new') if b.start_qn>=Q(1) else b for b in p.beats if b.start_qn<Q(4))
    p=replace(p,notes=(),bars=bars,beats=beats,meter_events=meters,duration_qn=Q(4))
    profile=ContextProfile(core_bars=1,halo_bars=0)
    e=split_masked_piece(p,MaskQuery(Q(1),Q(4),(p.tracks[0].track_id,)))
    second=build_observed_tiles(e.context,profile)[1]
    assert second.piece.meter_events[0].onset_qn == Q(0)
    assert second.piece.meter_events[0].numerator == 3
    graph=build_tile_graph(second,profile)
    assert (graph.slice_features[:,3]==3).all()
    assert (graph.slice_features[:,4]==4).all()


def test_observed_real_attack_at_crop_boundary_is_kept_with_carry_in():
    p=piece()
    sustain=replace(p.notes[0],onset_qn=Q(0),duration_qn=Q(20),pitch=50)
    attack=replace(sustain,note_id='note:attack',onset_qn=Q(1),duration_qn=Q(1),pitch=70)
    p=replace(p,notes=(sustain,attack))
    profile=ContextProfile(core_bars=1,halo_bars=0)
    e=split_masked_piece(p,MaskQuery(Q(1),Q(5),(p.tracks[1].track_id,)))
    graph=build_tile_graph(build_observed_tiles(e.context,profile)[1],profile)
    assert graph.base['onset'].num_nodes == 1
    assert graph.base['onset','starts_note','note'].edge_index.shape[1] == 1
    assert graph.edges['note','starts','slice'].shape[1] == 1
    assert graph.slice_features[0,7] == 1
    col=graph.base['onset'].cont_feature_names[0].index('starting_note_count')
    assert graph.base['onset'].x_cont[0,col] == 1


def test_known_long_sustain_does_not_expand_cropped_context():
    p=piece()
    bar=p.bars[-1]
    bars=tuple(replace(bar,bar_id=f'bar:{i}',index=i,start_qn=Q(4*i),duration_qn=Q(4)) for i in range(40))
    beat=p.beats[1]
    beats=tuple(replace(beat,beat_id=f'beat:{i}:{j}',bar_id=f'bar:{i}',start_qn=Q(4*i+j),
        index_in_bar=j,position_in_bar_qn=Q(j),is_downbeat=j==0) for i in range(40) for j in range(4))
    sustain=replace(p.notes[0],onset_qn=Q(0),duration_qn=Q(100))
    p=replace(p,notes=(sustain,),bars=bars,beats=beats,duration_qn=Q(160))
    e=split_masked_piece(p,MaskQuery(Q(80),Q(84),(p.tracks[1].track_id,)))
    assert e.context.global_start==Q(20)
    assert len(e.context.piece.bars)==32
    tiles=build_observed_tiles(e.context)
    assert len(tiles)==8
    assert max(len(t.piece.bars) for t in tiles)==6
    assert tiles[0].global_start==Q(20)
    assert tiles[0].note_geometry[0].onset==Q(0)
    graph=build_tile_graph(tiles[0])
    assert graph.base['onset'].num_nodes==0
    assert len(graph.slots[0].time_slots)==21
