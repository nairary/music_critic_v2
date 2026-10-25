"""Synthetic multiscale memory contracts; no corpus or checkpoint use."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece, RationalTime as Q
from music_critic.critic.c3_backbone import C3_ADAPTER_VERSION
from music_critic.critic.features import expected_stages
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
from music_critic.pll.contracts import ContextProfile, MaskQuery, ModelConfig
from music_critic.pll.encoder import EncodedTile, PLLContextEncoder
from music_critic.pll.masking import build_observed_tiles, split_masked_piece
from music_critic.pll.memory import HierarchicalMemory


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(*, empty=False, track=0):
    piece = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    if empty:
        piece = replace(piece, notes=(), annotations=(), targets=())
    view = split_masked_piece(piece, MaskQuery(Q(1), Q(5), (piece.tracks[track].track_id,))).context
    encoder = PLLContextEncoder(freeze_policy="full").eval()
    encoded = []
    for tile in build_observed_tiles(view):
        graph = build_temporal_batch((tile.piece,))
        encoded.append(EncodedTile(tile, graph, encoder._encode_graph(graph), expected_stages(C3_ADAPTER_VERSION)))
    return view, tuple(encoded), encoder


def tiny_config():
    return ModelConfig(width=32, global_layers=1, heads=4, ff_dim=64, dropout=0)


def test_memory_retains_track_bar_empty_cells_and_scales_with_gradients():
    view, encoded, encoder = inputs()
    module = HierarchicalMemory(tiny_config()).eval()
    memory = module(encoded, view)
    levels = [row.level for row in memory.metadata]
    assert levels.count("bar") == 2
    assert levels.count("track_bar") == 4
    assert levels.count("track") == 2
    assert levels.count("group2") == 1
    assert levels.count("group4") == levels.count("group8") == 0
    assert levels.count("context") == levels.count("query") == 1
    assert set(memory.families) == {"activity", "rhythm", "vertical", "melodic"}
    assert memory.values.shape == (len(memory.metadata), 32)
    assert memory.query == view.query
    assert all(v.ndim == 2 and v.shape[0] >= 1 and v.shape[1] == 32 for v in memory.families.values())
    cells = [m for m in memory.metadata if m.level == "track_bar"]
    assert any(m.coverage == 0 for m in cells)
    assert any(m.coverage == 1 for m in cells)
    memory.values[:, 0].sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in encoder.backbone.base.local_baseline.parameters())


def test_empty_music_keeps_unknown_cells_distinct_from_observed_silence():
    view, encoded, _ = inputs(empty=True)
    module = HierarchicalMemory(tiny_config()).eval()
    memory = module(encoded, view)
    cells = [(i,m) for i,m in enumerate(memory.metadata) if m.level == "track_bar"]
    assert all(m.empty for _,m in cells)
    assert sorted(m.coverage for _,m in cells) == [0, 1, 1, 1]
    hidden = next(i for i,m in cells if m.coverage == 0)
    silence = next(i for i,m in cells if m.start == Q(1) and m.coverage == 1)
    assert not torch.equal(memory.values[hidden], memory.values[silence])
    assert torch.isfinite(memory.values).all()


def test_fine_pooling_retains_levels_and_obeys_explicit_budget():
    view, encoded, _ = inputs()
    module = HierarchicalMemory(tiny_config(), ContextProfile(max_fine_memory=10)).eval()
    memory = module(encoded, view)
    fine = [m for m in memory.metadata if m.level in {"note_local", "note", "onset", "beat", "slice"}]
    assert len(fine) <= 10
    assert {m.level for m in fine} == {"note_local", "note", "onset", "beat", "slice"}
    assert any(m.pooled_rows > 1 for m in fine)
    assert len(memory.metadata) <= module.profile.max_memory


def test_memory_rejects_unsupported_limit_and_duplicate_core_owner():
    from music_critic.pll.observed_view import UnsupportedContextError
    view, encoded, _ = inputs()
    with pytest.raises(UnsupportedContextError, match="global_memory_limit"):
        HierarchicalMemory(tiny_config(), ContextProfile(max_memory=2))(encoded, view)
    with pytest.raises(ValueError, match="exactly one tile owner"):
        HierarchicalMemory(tiny_config())(encoded + encoded, view)


def test_halo_does_not_duplicate_global_cells_or_fine_events():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    profile = ContextProfile(core_bars=1, halo_bars=1)
    view = split_masked_piece(p, MaskQuery(Q(1), Q(5), (p.tracks[0].track_id,)), profile).context
    encoded = PLLContextEncoder(profile).eval()(view)
    result = HierarchicalMemory(tiny_config(), profile).eval()(encoded, view)
    assert sum(m.level == "bar" for m in result.metadata) == 2
    assert sum(m.level == "track_bar" for m in result.metadata) == 4
    incidences = sum((b.start_qn <= n.onset_qn < b.start_qn+b.duration_qn) if n.duration_qn == Q(0)
                     else (n.onset_qn < b.start_qn+b.duration_qn and n.onset_qn+n.duration_qn > b.start_qn)
                     for b in view.piece.bars for n in view.piece.notes)
    assert sum(m.pooled_rows for m in result.metadata if m.level == "track_bar") == incidences
    assert sum(m.level == "note" for m in result.metadata) == len(view.piece.notes)


def test_hidden_filling_has_no_effect_on_complete_memory():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    q = MaskQuery(Q(1), Q(5), tuple(t.track_id for t in p.tracks))
    other = replace(p, notes=tuple(n for n in p.notes if n.onset_qn < Q(1)), annotations=(), targets=())
    encoder, memory = PLLContextEncoder().eval(), HierarchicalMemory(tiny_config()).eval()
    a, b = [split_masked_piece(x, q).context for x in (p, other)]
    first, second = memory(encoder(a), a), memory(encoder(b), b)
    assert first.metadata == second.metadata
    torch.testing.assert_close(first.values, second.values, rtol=0, atol=0)


def test_unknown_metric_fine_tokens_do_not_claim_observed_silence():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    p = replace(p, notes=(), annotations=(), targets=())
    view = split_masked_piece(p, MaskQuery(Q(1), Q(5), tuple(t.track_id for t in p.tracks))).context
    encoded = PLLContextEncoder().eval()(view)
    memory = HierarchicalMemory(tiny_config()).eval()(encoded, view)
    unknown = [m for m in memory.metadata if m.level in {"beat", "slice"} and Q(1) <= m.start < Q(5)]
    assert unknown and all(m.coverage == 0 for m in unknown)
    known = [m for m in memory.metadata if m.level == "beat" and m.start == Q(0)]
    assert known and all(m.coverage == 1 for m in known)


def test_sustained_only_track_bar_and_carry_in_fine_note_remain_visible():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    track = p.notes[0].track_id
    p = replace(p, notes=(replace(p.notes[0], onset_qn=Q(0), duration_qn=Q(20)),))
    other = next(t.track_id for t in p.tracks if t.track_id != track)
    profile = ContextProfile(context_bars=1, core_bars=1, halo_bars=0)
    view = split_masked_piece(p, MaskQuery(Q(1), Q(5), (other,)), profile).context
    result = HierarchicalMemory(tiny_config(), profile).eval()(PLLContextEncoder(profile).eval()(view), view)
    cells = [m for m in result.metadata if m.level == "track_bar"]
    assert sum(not m.empty for m in cells) == 1
    notes = [m for m in result.metadata if m.level == "note"]
    assert len(notes) == 1
    assert notes[0].start == Q(0) and notes[0].end == Q(20)
