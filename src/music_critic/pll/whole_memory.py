"""Position-aware, tiled whole-piece memory built only from masked observations.

Attention is local to track/bar cells and eight-bar blocks before a small
whole-piece Transformer. Halo rows help encode a core; only its owner exports
that bar or event. All interval decisions use exact canonical times.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from music_critic.data.timing import RationalTime as Q
from .contracts import ContextProfile, ModelConfig
from .encoder import EncodedTile
from .memory import ContextMemory, FAMILY_LEVELS, MemoryToken
from .observed_view import ObservedMusicView, UnsupportedContextError
from .whole_contracts import WholePieceConfig


_LEVELS = ("note_local", "note", "onset", "beat", "slice", "bar", "track_bar", "query")


@dataclass(frozen=True)
class VisibleFineTile:
    core_bar_ids: tuple[str, ...]
    values: Tensor
    metadata: tuple[MemoryToken, ...]


@dataclass(frozen=True)
class WholeSongMemoryOutput:
    local: ContextMemory
    global_memory: ContextMemory
    bar_values: Tensor
    bar_metadata: tuple[MemoryToken, ...]
    fine_tiles: tuple[VisibleFineTile, ...]
    query_values: Tensor


def _intersects(start: Q, end: Q, left: Q, right: Q) -> bool:
    return left <= start < right if start == end else start < right and end > left


def _coverage(view, start, end, track=None):
    if track is not None and view.piece.tracks[track].track_id not in view.query.track_ids:
        return 1.
    if start == end:
        hidden = float(view.query.start <= start < view.query.end)
    else:
        overlap = min(end, view.query.end) - max(start, view.query.start)
        hidden = max(0., float(overlap.to_fraction() / (end - start).to_fraction()))
    return 1. - hidden * (len(view.query.track_ids) / len(view.piece.tracks) if track is None else 1.)


def _context(values, metadata, query, *, all_families=False):
    # Whole-song block/song summaries carry every family, while local rows use
    # the same family support as the original context-memory contract.
    families = {}
    for family, levels in FAMILY_LEVELS.items():
        indices = [i for i, token in enumerate(metadata) if token.level in levels]
        families[family] = values[indices] if indices and not all_families else values
    return ContextMemory(values, families, tuple(metadata), query)


class WholeSongMemory(nn.Module):
    """Expose pre-global bar slots for retrieval and separate decoder branches."""

    def __init__(self, config: ModelConfig, profile: ContextProfile,
                 whole_config: WholePieceConfig):
        super().__init__()
        self.config, self.profile, self.whole_config = config, profile, whole_config
        width = config.width
        self.projections = nn.ModuleDict({name: nn.Linear(128, width) for name in _LEVELS})
        self.level_embedding = nn.Embedding(len(_LEVELS), width)
        self.track_embedding = nn.Embedding(profile.max_tracks + 1, width)
        self.geometry_projection = nn.Sequential(nn.Linear(13, width), nn.GELU(), nn.Linear(width, width))
        self.cell_query = nn.Parameter(torch.randn(1, width) * .02)
        self.bar_queries = nn.Parameter(torch.randn(whole_config.bar_slots, width) * .02)
        self.block_queries = nn.Parameter(torch.randn(whole_config.block_slots, width) * .02)
        self.song_queries = nn.Parameter(torch.randn(whole_config.song_slots, width) * .02)
        self.query_vector = nn.Parameter(torch.randn(width) * .02)
        self.cell_attention = nn.MultiheadAttention(width, config.heads, config.dropout, batch_first=True)
        self.bar_attention = nn.MultiheadAttention(width, config.heads, config.dropout, batch_first=True)
        self.near_attention = nn.MultiheadAttention(width, config.heads, config.dropout, batch_first=True)
        self.cell_norm, self.bar_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.near_norm, self.fine_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.block_position = nn.Linear(2, width)
        self.block_transformer = self._transformer(whole_config.block_layers)
        self.song_transformer = self._transformer(whole_config.song_layers)

    def _transformer(self, layers):
        c = self.config
        layer = nn.TransformerEncoderLayer(c.width, c.heads, c.ff_dim, c.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(c.width), enable_nested_tensor=False)

    def _project(self, value, token, view, bar, beat_starts):
        total = (view.global_end - view.global_start).to_fraction()
        duration = bar.duration_qn.to_fraction()
        local_start = float((token.start - bar.start_qn).to_fraction() / duration)
        local_end = float((token.end - bar.start_qn).to_fraction() / duration)
        beat_index = bisect_right(beat_starts, token.start) - 1
        beat = view.piece.beats[beat_index] if beat_index >= 0 else None
        if beat is not None and token.start >= beat.start_qn + beat.duration_qn:
            beat = None
        phase = float((token.start - beat.start_qn).to_fraction() / beat.duration_qn.to_fraction()) if beat else 0.
        selected = token.track is not None and view.piece.tracks[token.track].track_id in view.query.track_ids
        features = value.new_tensor((
            float((token.start - view.global_start).to_fraction() / total),
            float((token.end - view.global_start).to_fraction() / total),
            local_start, local_end, math.sin(2 * math.pi * local_start), math.cos(2 * math.pi * local_start),
            phase, math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase),
            float(beat.index_in_bar) if beat else -1., token.coverage, float(token.empty), float(selected)))
        track = self.profile.max_tracks if token.track is None else token.track
        return (self.projections[token.level](value) + self.geometry_projection(features)
                + self.level_embedding.weight[_LEVELS.index(token.level)] + self.track_embedding.weight[track])

    def _rows(self, encoded, view):
        """Every tile row has actual global geometry, including censored tails."""
        tile = encoded.tile
        tracks = {t.track_id: i for i, t in enumerate(view.piece.tracks)}
        result = []
        for level in ("note_local", "note", "onset", "beat", "slice"):
            if level in {"note_local", "note"}:
                geometry = [(g.onset, g.onset + g.duration, tracks[n.track_id], g.note_id)
                    for n, g in zip(tile.piece.notes, tile.note_geometry)]
            elif level == "onset":
                geometry = []
                for _, key in encoded.graph.keys["onset"]:
                    num, den = key.removeprefix("onset:").split("_")
                    time = Q(int(num), int(den)) + tile.global_start
                    geometry.append((time, time, None, str(time)))
            elif level == "beat":
                geometry = [(b.start_qn + tile.global_start, b.start_qn + b.duration_qn + tile.global_start,
                             None, b.beat_id) for b in tile.piece.beats]
            else:
                times = [Q.from_fraction(t) + tile.global_start for t, _ in encoded.graph.slots[0].time_slots]
                geometry = [(t, times[i + 1] if i + 1 < len(times) else t, None, str(t)) for i, t in enumerate(times)]
            values = encoded.embeddings[level]
            if len(geometry) != len(values):
                raise ValueError("pll.whole_fine_geometry_mismatch")
            for value, (start, end, track, key) in zip(values, geometry):
                coverage = 1. if level in {"note_local", "note"} else _coverage(view, start, end)
                result.append((value, MemoryToken(level, start, end, track, coverage,
                              stage=encoded.stages[level]), key))
        return result

    @staticmethod
    def _pool(rows, budget, view):
        """Bound details using consecutive groups; retain all rows' support."""
        if len(rows) <= budget:
            return rows
        groups = defaultdict(list)
        for value, token in rows:
            groups[token.level, token.track].append((value, token))
        if budget < len(groups):
            raise UnsupportedContextError("pll.whole_detail_budget_cannot_retain_levels_and_tracks")
        quota = budget // len(groups)
        pooled = []
        for group in groups.values():
            group.sort(key=lambda row: (row[1].start, row[1].end))
            count = min(quota, len(group))
            for i in range(count):
                chunk = group[i * len(group) // count:(i + 1) * len(group) // count]
                first = chunk[0][1]
                start, end = min(t.start for _, t in chunk), max(t.end for _, t in chunk)
                coverage = 1. if first.level in {"note_local", "note"} else _coverage(view, start, end, first.track)
                pooled.append((torch.stack([v for v, _ in chunk]).mean(0),
                    MemoryToken(first.level, start, end, first.track, coverage,
                        all(t.empty for _, t in chunk), first.stage, sum(t.pooled_rows for _, t in chunk))))
        return pooled

    def forward(self, encoded_tiles: tuple[EncodedTile, ...], view: ObservedMusicView) -> WholeSongMemoryOutput:
        if not encoded_tiles or not view.piece.bars or not view.piece.tracks:
            raise ValueError("pll.whole_memory_requires_tiles_bars_tracks")
        wc, bars = self.whole_config, view.piece.bars
        if len(bars) > min(wc.max_piece_bars, self.profile.context_bars):
            raise UnsupportedContextError("pll.whole_piece_bar_limit")
        if len(view.piece.notes) > wc.max_piece_notes or len(view.piece.tracks) > self.profile.max_tracks:
            raise UnsupportedContextError("pll.whole_piece_note_or_track_limit")
        hit = [i for i, b in enumerate(bars) if _intersects(b.start_qn, b.start_qn + b.duration_qn, view.query.start, view.query.end)]
        if not hit:
            raise ValueError("pll.whole_query_outside_bars")
        left, right = max(0, hit[0] - wc.near_bars), min(len(bars), hit[-1] + 1 + wc.near_bars)
        global_count = (((len(bars) + wc.block_bars - 1) // wc.block_bars) * wc.block_slots + wc.song_slots
                        if wc.enable_global else 0)
        reserved = (right - left) * (wc.bar_slots + len(view.piece.tracks)) + 1
        if reserved > wc.max_local_tokens:
            raise UnsupportedContextError("pll.whole_local_summary_limit")
        if reserved + global_count > self.profile.max_memory:
            raise UnsupportedContextError("pll.whole_memory_limit")
        owners, tile_rows = {}, []
        for ti, encoded in enumerate(encoded_tiles):
            if encoded.tile.query != view.query:
                raise ValueError("pll.whole_tile_query_disagreement")
            local_indices = {b.bar_id: i for i, b in enumerate(encoded.tile.piece.bars)}
            for bar_id in encoded.tile.core_bar_ids:
                if bar_id in owners or bar_id not in local_indices:
                    raise ValueError("PLL core bars must have exactly one tile owner")
                owners[bar_id] = ti, local_indices[bar_id]
            tile_rows.append(self._rows(encoded, view))
        if set(owners) != {b.bar_id for b in bars}:
            raise ValueError("pll.whole_core_coverage_mismatch")
        zero = self.query_vector.new_zeros(128)
        beat_starts = tuple(beat.start_qn for beat in view.piece.beats)
        def project(value, token, bar):
            return self._project(value, token, view, bar, beat_starts)
        bar_values, bar_metadata, cells, detail_by_tile = [], [], [], defaultdict(list)
        seen = set()
        for bi, bar in enumerate(bars):
            start, end = bar.start_qn, bar.start_qn + bar.duration_qn
            ti, row = owners[bar.bar_id]
            encoded = encoded_tiles[ti]
            rows = tile_rows[ti]
            visible_notes = [r for r in rows if r[1].level == "note" and _intersects(r[1].start, r[1].end, start, end)]
            coverage = _coverage(view, start, end)
            bar_token = MemoryToken("bar", start, end, None, coverage, not visible_notes)
            bar_metadata.append(bar_token)
            bar_inputs = [project(encoded.embeddings["bar"][row], bar_token, bar)]
            for track, description in enumerate(view.piece.tracks):
                track_rows = [(value, token) for value, token, _ in rows
                    if token.track == track and _intersects(token.start, token.end, start, end)]
                token = MemoryToken("track_bar", start, end, track,
                    view.coverage(bar.bar_id, description.track_id), not track_rows, pooled_rows=len(track_rows) // 2)
                local_track = next(i for i, t in enumerate(encoded.tile.piece.tracks) if t.track_id == description.track_id)
                anchor = project(encoded.embeddings["track"][local_track], token, bar)
                values = torch.stack([anchor] + [project(v, m, bar) for v, m in track_rows])
                query = self.cell_query + anchor
                attended = self.cell_attention(query.unsqueeze(0), values.unsqueeze(0), values.unsqueeze(0), need_weights=False)[0][0, 0]
                cell = self.cell_norm(anchor + attended)
                cells.append((bi, cell, token))
                bar_inputs.append(cell)
            for value, token, _ in rows:
                if token.track is None and start <= token.start < end:
                    bar_inputs.append(project(value, token, bar))
            values = torch.stack(bar_inputs).unsqueeze(0)
            slots = self.bar_attention(self.bar_queries.unsqueeze(0), values, values, need_weights=False)[0].squeeze(0)
            bar_values.append(self.bar_norm(slots + self.bar_queries))
            # Details have a single core owner, even when a sustain participates
            # in several bar summaries. Halo-only rows never become extra events.
            details = []
            for value, token, key in rows:
                owned = (start <= token.start < end or
                    (bi == 0 and token.start < start and token.end > start) or
                    (bi == len(bars) - 1 and token.start == end == view.global_end and token.track is None))
                if owned and (token.level, key) not in seen:
                    seen.add((token.level, key))
                    details.append((self.fine_norm(project(value, token, bar)), token))
            detail_by_tile[ti].extend(self._pool(details, wc.max_detail_tokens_per_bar, view))
        bar_tensor = torch.stack(bar_values)
        block_values, block_metadata = [], []
        for offset in range(0, len(bars), wc.block_bars):
            end_index = min(len(bars), offset + wc.block_bars)
            sequence = bar_tensor[offset:end_index].flatten(0, 1)
            positions = sequence.new_tensor([[i / wc.block_bars, slot / wc.bar_slots]
                for i in range(end_index - offset) for slot in range(wc.bar_slots)])
            sequence = sequence + self.block_position(positions)
            states = self.block_transformer(torch.cat((self.block_queries, sequence)).unsqueeze(0))[0, :wc.block_slots]
            block_values.append(states)
            a, b = bars[offset].start_qn, bars[end_index - 1].start_qn + bars[end_index - 1].duration_qn
            block_metadata.extend(MemoryToken(f"group{wc.block_bars}", a, b, None, _coverage(view, a, b)) for _ in range(wc.block_slots))
        blocks = torch.cat(block_values)
        global_values = self.song_transformer(torch.cat((blocks, self.song_queries)).unsqueeze(0))[0]
        global_metadata = block_metadata + [MemoryToken("context", view.global_start, view.global_end,
            None, _coverage(view, view.global_start, view.global_end)) for _ in range(wc.song_slots)]
        global_memory = _context(global_values, global_metadata, view.query, all_families=True)
        near_values = bar_tensor[left:right].flatten(0, 1)
        query_token = MemoryToken("query", view.query.start, view.query.end, None, 0.)
        query = self.query_vector + project(zero, query_token, bars[hit[0]])
        selected = [i for i, t in enumerate(view.piece.tracks) if t.track_id in view.query.track_ids]
        query = query + self.track_embedding.weight[selected].mean(0)
        near_values = torch.cat((near_values, query.unsqueeze(0)))
        if wc.enable_global:
            update = self.near_attention(near_values.unsqueeze(0), global_values.unsqueeze(0), global_values.unsqueeze(0), need_weights=False)[0][0]
            near_values = self.near_norm(near_values + update)
        near_metadata = [bar_metadata[i] for i in range(left, right) for _ in range(wc.bar_slots)] + [query_token]
        local_cells = [(v, t) for i, v, t in cells if left <= i < right]
        fine_tiles = tuple(VisibleFineTile(encoded.tile.core_bar_ids,
            torch.stack([v for v, _ in detail_by_tile[i]]) if detail_by_tile[i] else global_values.new_empty((0, self.config.width)),
            tuple(t for _, t in detail_by_tile[i])) for i, encoded in enumerate(encoded_tiles))
        near_start, near_end = bars[left].start_qn, bars[right - 1].start_qn + bars[right - 1].duration_qn
        details = [(v, t) for tile in fine_tiles for v, t in zip(tile.values, tile.metadata)
            if _intersects(t.start, t.end, near_start, near_end)]
        details = self._pool(details, min(wc.max_local_tokens - reserved,
            self.profile.max_memory - global_count - reserved), view)
        local_rows = local_cells + details
        local_values = torch.cat((near_values, torch.stack([v for v, _ in local_rows]))) if local_rows else near_values
        local_metadata = near_metadata + [t for _, t in local_rows]
        local = _context(local_values, local_metadata, view.query)
        return WholeSongMemoryOutput(local, global_memory, bar_tensor, tuple(bar_metadata), fine_tiles, near_values)
