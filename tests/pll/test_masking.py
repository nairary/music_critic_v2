"""Synthetic-only contracts for mask-before-graph input construction."""
from dataclasses import replace
from pathlib import Path

import pytest

from music_critic.data import load_piece, RationalTime as Q
from music_critic.pll.contracts import ContextProfile, MaskQuery
from music_critic.pll.masking import split_masked_piece, build_observed_tiles


def fixture_piece():
    return load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')


def test_hidden_content_and_annotations_cannot_change_observed_input():
    piece = fixture_piece()
    query = MaskQuery(Q(1), Q(5), tuple(t.track_id for t in piece.tracks))
    example = split_masked_piece(piece, query)
    hidden = next(n for n in piece.notes if Q(1) <= n.onset_qn < Q(5))
    changed = replace(hidden, note_id='note:secret', pitch=117, onset_qn=Q(7, 3), duration_qn=Q(100))
    variant = replace(piece, notes=tuple(n for n in piece.notes if not Q(1) <= n.onset_qn < Q(5)) + (changed,), duration_qn=Q(1000), annotations=(), targets=(), source_path='secret')
    other = split_masked_piece(variant, query)
    assert example.context == other.context
    assert example.targets != other.targets
    assert not example.context.piece.targets
    assert not example.context.piece.annotations
    assert not example.context.piece.provenance


def test_empty_answer_is_supported_and_query_has_unknown_coverage():
    piece = replace(fixture_piece(), notes=())
    example = split_masked_piece(piece, MaskQuery(Q(1), Q(5), (piece.tracks[0].track_id,)))
    assert example.targets == ()
    assert example.context.coverage('bar:1', 'track:0') == 0.0
    assert example.context.coverage('bar:0', 'track:0') == 1.0
    assert build_observed_tiles(example.context)[0].piece.notes == ()


def test_mask_is_track_specific_onset_anchored_and_keeps_full_target_tail():
    p=fixture_piece()
    first=replace(p.notes[0],track_id=p.tracks[0].track_id,onset_qn=Q(0),duration_qn=Q(20))
    target=replace(first,note_id='note:hidden',onset_qn=Q(4),pitch=61,duration_qn=Q(100))
    other_track=replace(target,note_id='note:visible',track_id=p.tracks[1].track_id,pitch=62)
    p=replace(p,notes=(first,target,other_track))
    e=split_masked_piece(p,MaskQuery(Q(1),Q(5),(p.tracks[0].track_id,)))
    assert e.tracks == e.context.allowed_tracks == (0,)
    assert len(e.targets)==1
    assert (e.targets[0].onset,e.targets[0].duration)==(Q(3),Q(100))
    assert [(n.pitch,n.duration_qn) for n in e.context.piece.notes]==[(first.pitch,Q(20)),(62,Q(100))]
    assert e.context.coverage('bar:1','track:1')==1.


def test_invalid_or_unsupported_queries_do_not_shrink_silently():
    p=fixture_piece()
    with pytest.raises(ValueError,match='unknown'):
        split_masked_piece(p,MaskQuery(Q(1),Q(5),('track:missing',)))
    with pytest.raises(ValueError,match='outside_declared_geometry'):
        split_masked_piece(p,MaskQuery(Q(1),Q(6),(p.tracks[0].track_id,)))
    with pytest.raises(ValueError,match='query_context_limit'):
        split_masked_piece(p,MaskQuery(Q(0),Q(5),(p.tracks[0].track_id,)),ContextProfile(context_bars=1,core_bars=1))
    with pytest.raises(ValueError,match='track_limit'):
        split_masked_piece(p,MaskQuery(Q(0),Q(1),(p.tracks[0].track_id,)),ContextProfile(max_tracks=1))


def test_core_ownership_is_unique_with_halo_overlap():
    p=fixture_piece()
    e=split_masked_piece(p,MaskQuery(Q(1),Q(2),(p.tracks[0].track_id,)))
    tiles=build_observed_tiles(e.context,ContextProfile(core_bars=1,halo_bars=1))
    assert [t.core_bar_ids for t in tiles]==[('bar:0',),('bar:1',)]
    assert all(len(t.piece.bars)==2 for t in tiles)
    core=[note_id for t in tiles for note_id in t.core_note_ids]
    assert len(core)==len(set(core))==len(e.context.piece.notes)


def test_partial_query_coverage_is_exactly_derived_from_declared_interval():
    p=replace(fixture_piece(),notes=())
    e=split_masked_piece(p,MaskQuery(Q(2),Q(3),(p.tracks[0].track_id,)))
    assert e.context.coverage('bar:1','track:0')==.75


def test_tile_admission_checks_profile_before_graph_encoding():
    p=fixture_piece()
    e=split_masked_piece(p,MaskQuery(Q(1),Q(5),(p.tracks[0].track_id,)))
    with pytest.raises(ValueError,match='context_bar_limit'):
        build_observed_tiles(e.context,ContextProfile(context_bars=1,core_bars=1))
    with pytest.raises(ValueError,match='context_track_limit'):
        build_observed_tiles(e.context,ContextProfile(max_tracks=1))


def test_adapter_authorized_irregular_bar_is_explicitly_unsupported():
    """Synthetic valid repaired geometry must not become a late graph crash."""
    from music_critic.data import validate_piece
    from music_critic.pll.observed_view import UnsupportedContextError
    p=fixture_piece()
    repair=replace(p.provenance[0],provenance_id='prov:zzrepair',kind='derivation',
        source='music_critic.dilemmadata_adapter',parents=(),
        details=(('raw_repair_evidence_version','synthetic-fixture'),))
    middle=replace(p.bars[1],duration_qn=Q(2),is_incomplete=True,provenance_id=repair.provenance_id)
    last=replace(p.bars[1],bar_id='bar:002',index=2,start_qn=Q(3))
    beats=[p.beats[0]]
    for bar in (middle,last):
        for i in range(bar.duration_qn.num):
            beats.append(replace(p.beats[1],beat_id=f'beat:{bar.index}-{i}',bar_id=bar.bar_id,
                start_qn=bar.start_qn+Q(i),index_in_bar=i,position_in_bar_qn=Q(i),is_downbeat=i==0))
    p=replace(p,notes=(),annotations=(),targets=(),bars=(p.bars[0],middle,last),
              beats=tuple(beats),duration_qn=Q(7),provenance=(*p.provenance,repair))
    assert validate_piece(p).errors == ()
    query=MaskQuery(Q(1),Q(3),(p.tracks[0].track_id,))
    with pytest.raises(UnsupportedContextError,match='irregular_metric_geometry'):
        split_masked_piece(p,query)
    # Predictor admission depends on the declared geometry, not repair labels.
    with pytest.raises(UnsupportedContextError,match='irregular_metric_geometry'):
        split_masked_piece(replace(p,provenance=()),query)
    from types import SimpleNamespace
    from music_critic.pll.scoring import score_piece
    # The unsupported block is reported before any encoder/model access.
    scored=score_piece(SimpleNamespace(training=False,context_profile=ContextProfile()),p,(query,))
    assert scored['supported'] == 0
    assert scored['blocks'][0]['reason'] == 'pll.irregular_metric_geometry'
    assert scored['blocks'][0]['nll'] is None
