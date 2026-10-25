"""Synthetic whole-piece memory mechanics, never model-quality evidence."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import RationalTime as Q, load_piece
from music_critic.pll.contracts import ContextProfile, MaskQuery, ModelConfig
from music_critic.pll.encoder import PLLContextEncoder
from music_critic.pll.masking import split_masked_piece
from music_critic.pll.whole_contracts import WholePieceConfig
from music_critic.pll.whole_memory import WholeSongMemory


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config():
    return ModelConfig(width=32, heads=4, ff_dim=64, dropout=0, global_layers=1)


def fixture_piece():
    return load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")


def fixture_input(*, empty=False, profile=None):
    piece = fixture_piece()
    if empty:
        piece = replace(piece, notes=(), annotations=(), targets=())
    profile = profile or ContextProfile(context_bars=512, core_bars=1, halo_bars=1)
    view = split_masked_piece(piece, MaskQuery(Q(1), Q(5), (piece.tracks[0].track_id,)), profile).context
    encoder = PLLContextEncoder(profile).eval()
    return view, encoder(view), encoder, profile


def module(profile):
    return WholeSongMemory(tiny_config(), profile, WholePieceConfig(
        block_layers=1, song_layers=1, near_bars=1)).eval()


def test_whole_memory_preserves_bars_tracks_and_owned_note_geometry():
    view, encoded, _, profile = fixture_input()
    result = module(profile)(encoded, view)
    assert result.bar_values.shape == (2, 4, 32)
    assert [(m.start, m.end, m.coverage) for m in result.bar_metadata] == [
        (Q(0), Q(1), 1), (Q(1), Q(5), .5)]
    assert result.global_memory.values.shape == (12, 32)
    assert result.local.values.shape[1] == 32
    assert set(result.local.families) == {"activity", "rhythm", "vertical", "melodic"}
    notes = [m for tile in result.fine_tiles for m in tile.metadata if m.level == "note"]
    assert len(notes) == len(view.piece.notes)
    assert (Q(0), Q(2)) in {(m.start, m.end) for m in notes}
    assert sum(m.level == "query" for m in result.local.metadata) == 1


def test_empty_cells_distinguish_unknown_from_observed_rest():
    view, encoded, _, profile = fixture_input(empty=True)
    result = module(profile)(encoded, view)
    cells = [(i, m) for i, m in enumerate(result.local.metadata) if m.level == "track_bar"]
    assert all(m.empty and m.pooled_rows == 0 for _, m in cells)
    assert sorted(m.coverage for _, m in cells) == [0., 1., 1., 1.]
    unknown = next(i for i, m in cells if m.coverage == 0)
    rest = next(i for i, m in cells if m.start == Q(1) and m.coverage == 1)
    assert not torch.equal(result.local.values[unknown], result.local.values[rest])
    assert torch.isfinite(result.global_memory.values).all()


def test_replacing_hidden_events_preserves_all_memory_and_owned_details():
    piece = fixture_piece()
    other = replace(piece, notes=tuple(n for n in piece.notes if n.onset_qn < Q(1)), annotations=(), targets=())
    profile = ContextProfile(context_bars=512, core_bars=1, halo_bars=1)
    query = MaskQuery(Q(1), Q(5), tuple(t.track_id for t in piece.tracks))
    views = [split_masked_piece(p, query, profile).context for p in (piece, other)]
    encoder, memory = PLLContextEncoder(profile).eval(), module(profile)
    a, b = [memory(encoder(v), v) for v in views]
    assert a.bar_metadata == b.bar_metadata
    torch.testing.assert_close(a.bar_values, b.bar_values, rtol=0, atol=0)
    for first, second in ((a.local, b.local), (a.global_memory, b.global_memory)):
        assert first.metadata == second.metadata
        torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)
    for first, second in zip(a.fine_tiles, b.fine_tiles):
        assert first.metadata == second.metadata
        torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)


def test_position_aware_bar_attention_distinguishes_reordered_contents():
    torch.manual_seed(24)
    view, encoded, _, profile = fixture_input()
    memory = module(profile)
    # Exchange note states at two different attack positions while holding
    # their multiset and all other graph states/geometry exactly fixed.
    changed = []
    for tile in encoded:
        values = dict(tile.embeddings)
        for level in ("note", "note_local"):
            values[level] = values[level].flip(0)
        changed.append(replace(tile, embeddings=values))
    a, b = memory(encoded, view), memory(tuple(changed), view)
    assert not torch.allclose(a.bar_values, b.bar_values)


def long_piece(count=17):
    base = fixture_piece()
    bars = tuple(replace(base.bars[1], bar_id=f"bar:{i}", index=i, start_qn=Q(4 * i)) for i in range(count))
    beats = tuple(replace(base.beats[1 + j], bar_id=f"bar:{i}", beat_id=f"beat:{i}:{j}",
                          start_qn=Q(4 * i + j)) for i in range(count) for j in range(4))
    notes = tuple(replace(base.notes[0], note_id=f"note:{i}", onset_qn=Q(4 * i), duration_qn=Q(1),
                          pitch=60 + i % 12) for i in range(count))
    return replace(base, bars=bars, beats=beats, notes=notes, duration_qn=Q(4 * count), annotations=(), targets=())


def test_far_visible_bar_changes_global_and_query_states_and_receives_gradient():
    torch.manual_seed(30)
    piece = long_piece()
    profile = ContextProfile(context_bars=512)
    view = split_masked_piece(piece, MaskQuery(Q(4), Q(8), (piece.tracks[0].track_id,)), profile).context
    encoder = PLLContextEncoder(profile).eval()
    encoded = encoder(view)
    memory = module(profile)
    first = memory(encoded, view)
    # Final core is beyond both the near region and the first tile's halo.
    last = encoded[-1]
    values = {name: value.detach().clone() for name, value in last.embeddings.items()}
    values["bar"] = (values["bar"] + torch.arange(128).to(values["bar"]) / 10).requires_grad_()
    second = memory(encoded[:-1] + (replace(last, embeddings=values),), view)
    torch.testing.assert_close(first.bar_values[:3], second.bar_values[:3], rtol=0, atol=0)
    assert not torch.allclose(first.global_memory.values, second.global_memory.values)
    assert not torch.allclose(first.query_values, second.query_values)
    second.query_values[:, 0].sum().backward()
    assert values["bar"].grad is not None and bool(values["bar"].grad.abs().sum())
    for parameter in (memory.bar_queries, memory.block_queries, memory.song_queries,
                      memory.geometry_projection[0].weight, memory.near_attention.in_proj_weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert bool(parameter.grad.abs().sum())


def test_core_coverage_duplicates_and_resource_limits_fail_explicitly():
    from music_critic.pll.observed_view import UnsupportedContextError
    view, encoded, _, profile = fixture_input()
    memory = module(profile)
    with pytest.raises(ValueError, match="exactly one tile owner"):
        memory(encoded + encoded, view)
    with pytest.raises(ValueError, match="coverage_mismatch"):
        memory(encoded[:1], view)
    capped = WholeSongMemory(tiny_config(), profile, WholePieceConfig(max_piece_bars=1))
    with pytest.raises(UnsupportedContextError, match="bar_limit"):
        capped(encoded, view)
    capped = WholeSongMemory(tiny_config(), profile, WholePieceConfig(max_local_tokens=1))
    with pytest.raises(UnsupportedContextError, match="local_summary_limit"):
        capped(encoded, view)


def test_detail_pooling_preserves_exact_source_counts_without_halo_duplicates():
    view, encoded, _, profile = fixture_input()
    memory = WholeSongMemory(tiny_config(), profile, WholePieceConfig(
        max_detail_tokens_per_bar=8, max_local_tokens=32, block_layers=1, song_layers=1)).eval()
    result = memory(encoded, view)
    assert len(result.local.values) <= 32
    assert all(len(tile.values) <= 8 for tile in result.fine_tiles)
    note_count = sum(m.pooled_rows for tile in result.fine_tiles for m in tile.metadata if m.level == "note")
    assert note_count == len(view.piece.notes)
    assert all(m.coverage == 1 for tile in result.fine_tiles for m in tile.metadata if m.level in {"note", "note_local"})


def test_disabled_global_path_cannot_change_local_memory_from_distant_states():
    piece = long_piece()
    profile = ContextProfile(context_bars=512)
    view = split_masked_piece(piece, MaskQuery(Q(4), Q(8), (piece.tracks[0].track_id,)), profile).context
    encoded = PLLContextEncoder(profile).eval()(view)
    memory = WholeSongMemory(tiny_config(), profile, WholePieceConfig(
        block_layers=1, song_layers=1, near_bars=1, enable_global=False)).eval()
    last = encoded[-1]
    changed = replace(last, embeddings={name: value + 5 for name, value in last.embeddings.items()})
    a, b = memory(encoded, view), memory(encoded[:-1] + (changed,), view)
    assert a.local.metadata == b.local.metadata
    torch.testing.assert_close(a.local.values, b.local.values, rtol=0, atol=0)
    torch.testing.assert_close(a.query_values, b.query_values, rtol=0, atol=0)


def test_total_branch_memory_budget_is_explicit():
    from music_critic.pll.observed_view import UnsupportedContextError
    view, encoded, _, profile = fixture_input()
    with pytest.raises(UnsupportedContextError, match="whole_memory_limit"):
        module(replace(profile, max_memory=10))(encoded, view)


def test_configurable_block_geometry_and_all_global_relation_families():
    view, encoded, _, profile = fixture_input()
    memory = WholeSongMemory(tiny_config(), profile, WholePieceConfig(
        block_bars=1, block_layers=1, song_layers=1)).eval()
    result = memory(encoded, view)
    assert sum(m.level == "group1" for m in result.global_memory.metadata) == 8
    assert sum(m.level == "context" for m in result.global_memory.metadata) == 8
    for values in result.global_memory.families.values():
        torch.testing.assert_close(values, result.global_memory.values, rtol=0, atol=0)


def test_more_than_32_bars_remain_available_in_pre_global_memory():
    piece = long_piece(34)
    profile = ContextProfile(context_bars=512)
    view = split_masked_piece(piece, MaskQuery(Q(4), Q(8), (piece.tracks[0].track_id,)), profile).context
    encoded = PLLContextEncoder(profile).eval()(view)
    result = module(profile)(encoded, view)
    assert result.bar_values.shape == (34, 4, 32)
    assert result.bar_metadata[-1].end == Q(136)
    assert result.global_memory.values.shape == (28, 32)
    assert sum(m.pooled_rows for tile in result.fine_tiles for m in tile.metadata if m.level == "note") == 33


def test_fractional_mask_retains_exact_bounds_and_partial_unknown_coverage():
    piece = replace(fixture_piece(), notes=(), annotations=(), targets=())
    profile = ContextProfile(context_bars=512)
    view = split_masked_piece(piece, MaskQuery(Q(4, 3), Q(5, 3), (piece.tracks[0].track_id,)), profile).context
    result = module(profile)(PLLContextEncoder(profile).eval()(view), view)
    assert result.bar_metadata[1].coverage == 23 / 24
    unknown_cell = next(m for m in result.local.metadata if m.level == "track_bar" and m.start == Q(1) and m.track == 0)
    assert unknown_cell.coverage == 11 / 12
    query = next(m for m in result.local.metadata if m.level == "query")
    assert query.start == Q(4, 3) and query.end == Q(5, 3)
