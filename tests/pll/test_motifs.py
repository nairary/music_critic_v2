"""Synthetic motif-memory mechanics; no musical-quality or GPU claims."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import RationalTime as Q, load_piece
from music_critic.pll.contracts import ContextProfile, MaskQuery, ModelConfig
from music_critic.pll.masking import split_masked_piece
from music_critic.pll.motifs import MotifMemory, fragment_bank


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def synthetic_piece(bars=40):
    source = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    bar = source.bars[1]
    beat = source.beats[1]
    metric = tuple(replace(bar, bar_id=f'bar:{i}', index=i, start_qn=Q(4*i),
                           duration_qn=Q(4), metric_offset_qn=Q(0), is_pickup=False,
                           is_incomplete=False) for i in range(bars))
    beats = tuple(replace(beat, beat_id=f'beat:{i}-{j}', bar_id=f'bar:{i}',
                         start_qn=Q(4*i+j), duration_qn=Q(1), index_in_bar=j,
                         position_in_bar_qn=Q(j), is_downbeat=j == 0)
                  for i in range(bars) for j in range(4))
    note = source.notes[0]
    notes = tuple(replace(note, note_id=f'note:{i}-{j}', onset_qn=Q(4*i+j),
                         duration_qn=Q(1), pitch=60+(j*2), track_id=source.tracks[0].track_id)
                  for i in range(bars) for j in range(4))
    return replace(source, bars=metric, beats=beats, notes=notes, duration_qn=Q(4*bars),
                   annotations=(), targets=())


def observed(piece=None, start=0, end=4):
    piece = synthetic_piece() if piece is None else piece
    return split_masked_piece(piece, MaskQuery(Q(start), Q(end), (piece.tracks[0].track_id,)),
                              ContextProfile(context_bars=len(piece.bars), core_bars=min(4,len(piece.bars)))).context


def tiny():
    return ModelConfig(width=32, heads=4, global_layers=1, ff_dim=64, dropout=0)


def test_overlapping_bank_keeps_six_bar_scales_and_declines_mask_overlap():
    view = observed()
    bank = fragment_bank(view)
    assert {c.scale for c in bank if c.unit == 'bar'} == {1, 2, 4, 8, 16, 32}
    assert {c.scale for c in bank if c.unit == 'beat'} == {1, 2}
    assert all(c.end <= view.query.start or c.start >= view.query.end for c in bank)
    long = [c for c in bank if c.unit == 'bar' and c.scale == 32]
    assert [(c.first_bar, c.last_bar) for c in long] == [(i, i+31) for i in range(1, 9)]
    assert len({c.candidate_id for c in bank}) == len(bank)


def test_empty_bank_returns_finite_null_memory_and_explicit_coverage():
    view = observed(synthetic_piece(1))
    memory = MotifMemory(tiny()).eval()
    bars = torch.randn(1, 4, 32)
    result = memory((), view, bar_values=bars, global_values=torch.randn(2, 32))
    assert result.candidates == ()
    assert result.selected_indices == ()
    torch.testing.assert_close(result.weights, torch.ones(1))
    assert torch.isfinite(result.values).all()
    assert result.values.shape == (1, 32)
    assert result.scale_counts == {'bar:1': 0, 'bar:2': 0, 'bar:4': 0,
                                   'bar:8': 0, 'bar:16': 0, 'bar:32': 0,
                                   'beat:1': 0, 'beat:2': 0}


def test_soft_retrieval_keeps_gradients_for_unselected_candidates_and_keys():
    torch.manual_seed(8)
    view = observed(synthetic_piece(10))
    memory = MotifMemory(tiny()).eval()
    bars = torch.randn(10, 4, 32, requires_grad=True)
    result = memory((), view, bar_values=bars, global_values=torch.randn(2, 32))
    assert 0 < len(result.selected_indices) <= 4
    assert result.weights.shape == (len(result.candidates)+1,)
    torch.testing.assert_close(result.weights.sum(), torch.tensor(1.))
    result.candidate_keys.retain_grad()
    (result.values[0] * torch.arange(32)).sum().backward()
    assert bars.grad is not None and bars.grad.abs().sum() > 0
    assert result.candidate_keys.grad is not None
    unselected = [i for i in range(len(result.candidates)) if i not in result.selected_indices]
    assert result.candidate_keys.grad[unselected].abs().sum() > 0
    assert memory.key_projection.weight.grad.abs().sum() > 0


def test_hidden_replacement_preserves_bank_keys_weights_alignment_and_details():
    piece = synthetic_piece(10)
    changed = replace(piece, notes=tuple(n for n in piece.notes if n.onset_qn >= Q(4)) +
                      (replace(piece.notes[0], pitch=117, duration_qn=Q(1000)),))
    first, second = observed(piece), observed(changed)
    assert first == second
    torch.manual_seed(9)
    module = MotifMemory(tiny()).eval()
    bars, global_values = torch.randn(10, 4, 32), torch.randn(4, 32)
    a, b = [module((),v,bar_values=bars,global_values=global_values) for v in (first,second)]
    assert a.candidates == b.candidates
    assert a.selected_indices == b.selected_indices
    assert a.metadata == b.metadata
    for left,right in ((a.candidate_keys,b.candidate_keys),(a.weights,b.weights),(a.values,b.values)):
        torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_32_bar_details_cover_entire_candidate_with_finite_alignment_gradient():
    from music_critic.pll.whole_contracts import WholePieceConfig
    config = WholePieceConfig(motif_scales=(32,), motif_beat_scales=(), motif_top_k=1,
                              motif_details_per_candidate=4)
    view = observed(synthetic_piece(33))
    module = MotifMemory(tiny(),config).eval()
    bars = torch.randn(33,4,32,requires_grad=True)
    result = module((),view,bar_values=bars,global_values=torch.randn(2,32))
    assert result.scale_counts == {'bar:32': 1}
    assert result.selected_scale_counts == {'bar:32': 1}
    aligned, = result.alignments
    assert aligned.input_rows == 32 and aligned.output_rows == 4
    assert aligned.positions[0][0] == Q(4)
    assert aligned.positions[-1][1] == Q(132)
    assert sum(m.pooled_rows for m in result.metadata[1:]) == 32
    assert aligned.attention.shape[0] == 4
    torch.testing.assert_close(aligned.attention.sum(-1),torch.ones(4))
    (result.values[1:] * torch.arange(32)).sum().backward()
    assert module.aligner.in_proj_weight.grad.abs().sum() > 0
    assert bars.grad[-1].abs().sum() > 0


def test_bank_limit_fails_without_dropping_scales_and_selection_deduplicates():
    from music_critic.pll.observed_view import UnsupportedContextError
    view = observed()
    with pytest.raises(UnsupportedContextError,match='motif_candidate_limit'):
        fragment_bank(view,max_candidates=1)
    candidates = fragment_bank(view,scales=(1,2),beat_scales=())
    weights = torch.ones(len(candidates)+1)
    selected = MotifMemory._selected(candidates,weights,4,.5)
    assert len(selected) == 4
    for i,index in enumerate(selected):
        for other in selected[i+1:]:
            assert candidates[index].end <= candidates[other].start or candidates[other].end <= candidates[index].start


def test_raw_views_preserve_relative_pitch_under_transposition_and_temporal_order():
    from music_critic.pll.motifs import _content_views
    notes = tuple(replace(n,track_id="track:0") for n in synthetic_piece(1).notes)
    shifted = tuple(replace(n,pitch=n.pitch+7) for n in notes)
    first = torch.tensor(_content_views(notes,Q(0),Q(4)))
    second = torch.tensor(_content_views(shifted,Q(0),Q(4)))
    torch.testing.assert_close(first[1],second[1])
    assert not torch.equal(first[2],second[2])
    reverse = tuple(replace(n,pitch=notes[-i-1].pitch) for i,n in enumerate(notes))
    assert not torch.equal(first[1],torch.tensor(_content_views(reverse,Q(0),Q(4)))[1])
    # These checks constrain observable features, not a trained retrieval score.


def test_short_beat_detail_geometry_does_not_expand_to_whole_bar():
    from music_critic.pll.whole_contracts import WholePieceConfig
    config = WholePieceConfig(motif_scales=(32,),motif_beat_scales=(1,),motif_top_k=1)
    view = observed(synthetic_piece(2))
    module = MotifMemory(tiny(),config).eval()
    bars = torch.randn(2,4,32)
    result = module((),view,bar_values=bars,global_values=torch.randn(2,32))
    candidate = result.candidates[result.selected_indices[0]]
    assert candidate.unit == 'beat'
    alignment, = result.alignments
    assert alignment.positions == ((candidate.start,candidate.end),)


def test_synthetic_encoded_notes_receive_gradient_and_diagnostics_are_json_safe():
    import json
    from music_critic.pll.encoder import EncodedTile
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.whole_contracts import WholePieceConfig
    view = observed(synthetic_piece(3))
    config = WholePieceConfig(motif_scales=(2,),motif_beat_scales=(),motif_top_k=1,
                              motif_details_per_candidate=3,bar_slots=2)
    tiles = build_observed_tiles(view,ContextProfile(context_bars=3,core_bars=3))
    notes = torch.randn(len(tiles[0].piece.notes),128,requires_grad=True)
    encoded = (EncodedTile(tiles[0],None,{'note':notes},{}),)
    module = MotifMemory(tiny(),config).eval()
    result = module(encoded,view,bar_values=torch.randn(3,2,32),global_values=torch.randn(2,32))
    alignment, = result.alignments
    assert alignment.input_rows == 10  # two bar states plus eight observed notes
    assert [m.pooled_rows for m in result.metadata[1:]] == [3,3,4]
    (result.values[1:] * torch.arange(32)).sum().backward()
    assert notes.grad.abs().sum() > 0
    receipt = json.loads(json.dumps(result.diagnostics(),allow_nan=False))
    assert receipt['selected_scale_counts'] == {'bar:2':1}
    assert receipt['alignments'][0]['input_rows'] == 10


def test_sustained_only_candidate_is_observed_sound_and_keeps_carry_in_detail():
    from music_critic.pll.encoder import EncodedTile
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.whole_contracts import WholePieceConfig
    piece = synthetic_piece(3)
    sustain = replace(piece.notes[0], onset_qn=Q(0),duration_qn=Q(12))
    piece = replace(piece,notes=(sustain,))
    # Mask another track, so the onset-owned sustain remains observed.
    view = split_masked_piece(piece,MaskQuery(Q(0),Q(4),(piece.tracks[1].track_id,)),
                              ContextProfile(context_bars=3,core_bars=3)).context
    config = WholePieceConfig(motif_scales=(2,),motif_beat_scales=(),motif_top_k=1)
    tiles = build_observed_tiles(view,ContextProfile(context_bars=3,core_bars=3))
    encoded = (EncodedTile(tiles[0],None,{'note':torch.randn(1,128)},{}),)
    result = MotifMemory(tiny(),config).eval()(encoded,view,bar_values=torch.randn(3,4,32),
                                              global_values=torch.randn(2,32))
    candidate, = result.candidates
    assert candidate.observed_note_count == 1
    assert result.alignments[0].input_rows == 3  # two bars plus carry-in sustain
    assert all(not m.empty for m in result.metadata)


def test_rhythm_bins_use_exact_fraction_boundaries_and_separate_carry_from_rest():
    from music_critic.pll.motifs import _content_views
    note = replace(synthetic_piece(1).notes[0],track_id='track:0',duration_qn=Q(1,16),
                   onset_qn=Q(2**60-1,8*2**60))
    rows = _content_views((note,),Q(0),Q(1))
    assert rows[0][0] == 1 and rows[0][1] == 0
    carry = replace(note,onset_qn=Q(0),duration_qn=Q(4))
    rows = _content_views((carry,),Q(1),Q(2))
    assert rows[0][:8] == [0.]*8
    assert rows[0][8] == 1.
    assert rows[0][22:] == [1.,0.]  # no attack, but observed sounding content
    assert _content_views((),Q(1),Q(2))[0][22:] == [1.,1.]
