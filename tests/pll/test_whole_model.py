"""In-process whole-song model mechanics on synthesized scores, not real music."""
from dataclasses import replace

import pytest
import torch

from music_critic.pll.contracts import MaskQuery
from music_critic.pll.masking import split_masked_piece
from music_critic.pll.whole_contracts import split_whole_piece
from music_critic.pll.whole_model import WholePieceHCM
from tests.pll.whole_fixtures import full_piece, small_codec, small_config, small_whole, whole_example
from music_critic.data import RationalTime as Q


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def small_model(**whole):
    return WholePieceHCM(small_config(), codec_profile=small_codec(),
                         whole_config=small_whole(**whole))


def test_actual_c3_whole_song_motifs_decoder_backward():
    torch.manual_seed(19)
    model = small_model()
    ex = whole_example(40)
    result = model(ex)
    assert result.nll.isfinite()
    result.nll.backward()
    for name in ('memory','motifs','decoder'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in getattr(model,name).parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert model.last_diagnostics['source_bars'] == 40
    assert model.last_diagnostics['motifs']['scale_counts']['bar:32'] > 0
    assert 'whole_piece' in model.contract()


def test_hidden_replacement_cannot_change_memories_or_retrieval():
    torch.manual_seed(11)
    model = small_model().eval()
    piece = full_piece(8)
    query = MaskQuery(Q(4), Q(8), tuple(t.track_id for t in piece.tracks))
    deleted = replace(piece, notes=tuple(n for n in piece.notes if not Q(4) <= n.onset_qn < Q(8)))
    a,b = [split_whole_piece(p,query,whole_config=model.whole_config).context for p in (piece,deleted)]
    with torch.no_grad():
        ma,mb = model.encode(a),model.encode(b)
    torch.testing.assert_close(ma.values,mb.values,rtol=0,atol=0)
    assert ma.metadata == mb.metadata and ma.diagnostics == mb.diagnostics
    with pytest.raises(TypeError, match='whole_observed'):
        model.encode(split_masked_piece(piece,query).context)


def test_bound_ablations_and_generation_keep_the_public_event_interface():
    from music_critic.pll.codec import EOS
    model = small_model(enable_motifs=False, enable_global=False).eval()
    ex = whole_example(4)
    memory = model.encode(ex.context)
    assert set(memory.branches) == {'local'}
    assert model.last_diagnostics['motifs'] is None
    with torch.no_grad():
        model.decoder.output.weight.zero_(); model.decoder.output.bias.zero_()
        model.decoder.output.bias[EOS] = 100
    generated = model.generate(ex.context, generator=torch.Generator().manual_seed(2))
    assert generated == ()
    assert model(replace(ex, targets=generated)).nll.isfinite()
def test_disabled_global_branch_does_not_reserve_decoder_tokens():
    from music_critic.pll.contracts import ContextProfile
    model = WholePieceHCM(small_config(), whole_config=small_whole(enable_global=False, enable_motifs=False),
                         codec_profile=small_codec(), context_profile=ContextProfile(max_memory=32)).eval()
    # Query already uses normalized track IDs, so use the split fixture view.
    memory = model.encode(whole_example(17).context)
    assert set(memory.branches) == {'local'}
    assert len(memory.values) <= 32


def _tile_tensors(tiles):
    return [value.detach().clone() for tile in tiles for value in tile.embeddings.values()]


def test_encode_from_reuses_frozen_tiles_bitwise():
    torch.manual_seed(23)
    model = small_model().eval()
    ex = whole_example(12)
    with torch.no_grad():
        tiles = model.encode_tiles(ex.context)
        direct = model.encode(ex.context)
        reused = model.encode_from(tiles, ex.context)
    torch.testing.assert_close(direct.values, reused.values, rtol=0, atol=0)
    assert direct.metadata == reused.metadata
    assert direct.diagnostics == reused.diagnostics
    assert model.last_diagnostics == reused.diagnostics
    assert direct.query == reused.query
    assert set(direct.branches) == set(reused.branches) == {'local', 'global', 'motif'}
    for name in direct.branches:
        torch.testing.assert_close(direct.branches[name], reused.branches[name], rtol=0, atol=0)
    for name in direct.families:
        torch.testing.assert_close(direct.families[name], reused.families[name], rtol=0, atol=0)


def test_cached_tile_stage_survives_trained_weight_changes():
    """Cache reuse must equal recomputation while trained scopes keep moving."""
    torch.manual_seed(29)
    model = small_model().eval()
    ex = whole_example(12)
    with torch.no_grad():
        cached = model.encode_tiles(ex.context)
        cached_tensors = _tile_tensors(cached)
        baseline = model.encode_from(cached, ex.context)
        for module in (model.memory, model.motifs, model.decoder):
            for parameter in module.parameters():
                parameter.add_(0.05)
        refreshed = model.encode_tiles(ex.context)
        reused = model.encode_from(cached, ex.context)
        recomputed = model.encode_from(refreshed, ex.context)
    assert cached_tensors and len(_tile_tensors(refreshed)) == len(cached_tensors)
    assert all(torch.equal(a, b) for a, b in zip(cached_tensors, _tile_tensors(refreshed)))
    torch.testing.assert_close(reused.values, recomputed.values, rtol=0, atol=0)
    assert reused.diagnostics == recomputed.diagnostics
    assert not torch.equal(reused.values, baseline.values)


def test_encode_tiles_keeps_the_whole_view_contract():
    from music_critic.pll.observed_view import UnsupportedContextError
    bounded = small_model(max_piece_bars=4).eval()
    ex = whole_example(8)
    with pytest.raises(UnsupportedContextError, match='bar_limit'):
        bounded.encode_tiles(ex.context)
    with pytest.raises(UnsupportedContextError, match='bar_limit'):
        bounded.encode(ex.context)
    piece = full_piece(8)
    masked = split_masked_piece(piece, MaskQuery(Q(4), Q(8),
                                                 tuple(t.track_id for t in piece.tracks))).context
    with pytest.raises(TypeError, match='whole_observed'):
        small_model().eval().encode_tiles(masked)
    with pytest.raises(TypeError, match='whole_observed'):
        small_model().eval().encode_from((), masked)
