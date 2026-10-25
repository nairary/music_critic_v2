"""Whole-context contracts use only checked-in synthetic music."""
from dataclasses import replace
from pathlib import Path

import pytest

from music_critic.data import RationalTime as Q, load_piece
from music_critic.pll.contracts import ContextProfile, MaskQuery
from music_critic.pll.whole_contracts import WholePieceConfig, split_whole_piece


def test_whole_profile_is_bound_and_long_scales_are_configurable():
    config = WholePieceConfig()
    assert config.motif_scales == (1, 2, 4, 8, 16, 32)
    assert config.fingerprint() != replace(config, motif_scales=(1, 2, 4, 8)).fingerprint()
    for change in ({'max_piece_bars': 0}, {'motif_scales': (1, 1)},
                   {'motif_top_k': True}, {'motif_overlap_threshold': float('nan')}):
        with pytest.raises(ValueError):
            replace(config, **change)


def test_whole_masking_never_uses_the_old_context_crop():
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    query = MaskQuery(Q(1), Q(5), (piece.tracks[0].track_id,))
    result = split_whole_piece(piece, query, context_profile=ContextProfile(context_bars=1, core_bars=1))
    assert len(result.context.piece.bars) == len(piece.bars)
    assert result.context.global_bar_indices == tuple(range(len(piece.bars)))
    with pytest.raises(ValueError, match='whole_piece_bar_limit'):
        split_whole_piece(piece, query, whole_config=WholePieceConfig(max_piece_bars=1),
                          context_profile=ContextProfile(core_bars=1))


def test_hidden_note_count_does_not_select_context_or_resource_admission():
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    query = MaskQuery(Q(1), Q(5), tuple(t.track_id for t in piece.tracks))
    sparse = replace(piece, notes=tuple(n for n in piece.notes if not Q(1) <= n.onset_qn < Q(5)))
    whole = WholePieceConfig(max_piece_notes=len(sparse.notes))
    a = split_whole_piece(piece, query, whole_config=whole)
    b = split_whole_piece(sparse, query, whole_config=whole)
    assert a.context == b.context
