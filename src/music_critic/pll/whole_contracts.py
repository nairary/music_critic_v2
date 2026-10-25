"""Opt-in whole-source context, bounded without silently cutting a song."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math

from .contracts import ContextProfile, MaskQuery, _integer
from .masking import split_masked_piece
from .observed_view import MaskedExample, ObservedMusicView, UnsupportedContextError

WHOLE_SCHEMA = 'HierarchicalConditionalWholeMusicModel@1.0.0'


@dataclass(frozen=True, slots=True)
class WholeObservedMusicView(ObservedMusicView):
    """Declared source geometry proves this view was not locally cropped."""
    source_bar_count: int

    def __post_init__(self):
        if (type(self.source_bar_count) is not int or self.source_bar_count < 1 or
                len(self.piece.bars) != self.source_bar_count or
                self.global_bar_indices != tuple(range(self.source_bar_count))):
            raise ValueError('pll.whole_piece_context_was_cropped')


@dataclass(frozen=True, slots=True)
class WholePieceConfig:
    max_piece_bars: int = 512
    # A predictor-side resource bound: hidden target events do not count here.
    max_piece_notes: int = 65536
    bar_slots: int = 4
    block_bars: int = 8
    block_slots: int = 4
    song_slots: int = 8
    block_layers: int = 2
    song_layers: int = 2
    near_bars: int = 4
    max_local_tokens: int = 512
    max_detail_tokens_per_bar: int = 128
    motif_scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    motif_beat_scales: tuple[int, ...] = (1, 2)
    motif_top_k: int = 4
    motif_details_per_candidate: int = 64
    motif_overlap_threshold: float = .5
    max_motif_candidates: int = 8192
    freeze_local: bool = True
    enable_global: bool = True
    enable_motifs: bool = True

    def __post_init__(self):
        for name in ('max_piece_bars', 'max_piece_notes', 'bar_slots', 'block_bars',
                     'block_slots', 'song_slots', 'block_layers', 'song_layers',
                     'max_local_tokens', 'max_detail_tokens_per_bar', 'motif_top_k',
                     'motif_details_per_candidate', 'max_motif_candidates'):
            _integer(name, getattr(self, name), 1)
        _integer('near_bars', self.near_bars, 0)
        for name in ('freeze_local', 'enable_global', 'enable_motifs'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('pll.whole_boolean_required:' + name)
        for name in ('motif_scales', 'motif_beat_scales'):
            values = getattr(self, name)
            if (not isinstance(values, (tuple, list)) or
                    any(type(v) is not int or v < 1 for v in values) or
                    tuple(values) != tuple(sorted(set(values))) or
                    (name == 'motif_scales' and not values)):
                raise ValueError('pll.whole_unique_positive_sorted_scales:' + name)
            object.__setattr__(self, name, tuple(values))
        if (isinstance(self.motif_overlap_threshold, bool) or
                not isinstance(self.motif_overlap_threshold, (int, float)) or
                not math.isfinite(self.motif_overlap_threshold) or
                not 0 <= self.motif_overlap_threshold <= 1):
            raise ValueError('pll.whole_motif_overlap_threshold')

    def fingerprint(self) -> str:
        return sha256(json.dumps({'schema': WHOLE_SCHEMA, **asdict(self)},
                                 sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def whole_context_profile(context_profile: ContextProfile,
                          whole_config: WholePieceConfig) -> ContextProfile:
    return replace(context_profile, context_bars=whole_config.max_piece_bars,
                   core_bars=min(context_profile.core_bars, whole_config.max_piece_bars))


def split_whole_piece(piece, query: MaskQuery, *,
                      whole_config: WholePieceConfig = WholePieceConfig(),
                      context_profile: ContextProfile = ContextProfile()) -> MaskedExample:
    """Keep all source bars, sanitize once, and cap only observable notes.

    Full *source* context is not evidence that the source contains a full song;
    the preparation/manifest coverage contract provides that provenance.
    """
    if len(piece.bars) > whole_config.max_piece_bars:
        raise UnsupportedContextError('pll.whole_piece_bar_limit')
    profile = whole_context_profile(context_profile, whole_config)
    example = split_masked_piece(piece, query, profile)
    if example.context.global_bar_indices != tuple(range(len(piece.bars))):
        raise ValueError('pll.whole_piece_context_was_cropped')
    if len(example.context.piece.notes) > whole_config.max_piece_notes:
        raise UnsupportedContextError('pll.whole_piece_note_limit')
    view = example.context
    whole_view = WholeObservedMusicView(view.piece, view.query, view.global_bar_indices,
                                        len(piece.bars))
    return replace(example, context=whole_view)
