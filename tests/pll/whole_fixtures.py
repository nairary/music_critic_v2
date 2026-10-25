"""Small score geometry synthesized from a checked-in fixture, never corpus data."""
from dataclasses import replace
from pathlib import Path

from music_critic.data import RationalTime as Q, load_piece
from music_critic.pll.contracts import CodecProfile, MaskQuery, ModelConfig
from music_critic.pll.whole_contracts import WholePieceConfig, split_whole_piece


def full_piece(bars=40):
    p = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    metric = tuple(replace(p.bars[1], bar_id=f'bar:{i}', index=i, start_qn=Q(4*i),
                           display_number=None, is_pickup=False, is_incomplete=False)
                   for i in range(bars))
    beats = tuple(replace(p.beats[1], beat_id=f'beat:{i}-{j}', bar_id=f'bar:{i}',
                          start_qn=Q(4*i+j), index_in_bar=j, position_in_bar_qn=Q(j),
                          is_downbeat=j == 0, strength=1. if j == 0 else .5)
                  for i in range(bars) for j in range(4))
    notes = tuple(replace(p.notes[0], note_id=f'note:{i}', onset_qn=Q(4*i),
                          duration_qn=Q(1), pitch=60+i % 5,
                          source_onset_ticks=None, source_duration_ticks=None,
                          source_onset_seconds=None, source_duration_seconds=None)
                  for i in range(bars))
    return replace(p, duration_qn=Q(4*bars), bars=metric, beats=beats, notes=notes,
                   annotations=(), targets=(), key_signature_events=(), split='train')


def small_config(dropout=0.):
    return ModelConfig(width=32, heads=4, event_layers=1, global_layers=1, ff_dim=64,
                       field_width=16, field_heads=4, field_ff_dim=32, relation_dim=8,
                       relation_components=2, dropout=dropout)


def small_whole(**kwargs):
    return WholePieceConfig(block_layers=1, song_layers=1, max_local_tokens=32,
        max_detail_tokens_per_bar=16, motif_details_per_candidate=8, near_bars=1, **kwargs)


def small_codec():
    return CodecProfile(max_events=8, max_denominator=12, max_numerator=1024)


def whole_example(bars=40):
    piece = full_piece(bars)
    query = MaskQuery(Q(4), Q(8), (piece.tracks[0].track_id,))
    return split_whole_piece(piece, query, whole_config=small_whole())
