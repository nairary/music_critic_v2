"""Bounded multiscale context memory with explicit observed/unknown coverage.

Core bars have exactly one owner. Halo states aid local computation but never
create duplicate global cells or statistics. All pooling remains differentiable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import torch
from torch import Tensor, nn

from music_critic.data.timing import RationalTime as Q
from .contracts import ContextProfile, MaskQuery, ModelConfig
from .encoder import EncodedTile
from .observed_view import ObservedMusicView, UnsupportedContextError

LEVELS = ("bar", "track_bar", "track", "group2", "group4", "group8",
          "context", "query", "note_local", "note", "onset", "beat", "slice")
STAGES = ("local_gnn", "hierarchical_context", "c3_relationgate_round_2",
          "c3_relationgate_round_3", "pooled")
FAMILY_LEVELS = {
    "activity": {"track_bar", "track", "group2", "group4", "group8", "context", "query"},
    "rhythm": {"bar", "group2", "group4", "group8", "onset", "beat", "slice", "query"},
    "vertical": {"bar", "track_bar", "group2", "group4", "group8", "note", "slice", "query"},
    "melodic": {"track", "track_bar", "note_local", "note", "onset", "query"},
}


@dataclass(frozen=True)
class MemoryToken:
    level: str
    start: Q
    end: Q
    track: int | None
    coverage: float
    empty: bool = False
    stage: str = "pooled"
    pooled_rows: int = 1


@dataclass(frozen=True)
class ContextMemory:
    values: Tensor
    families: Mapping[str, Tensor]
    metadata: tuple[MemoryToken, ...]
    query: MaskQuery

    def __post_init__(self):
        if self.values.ndim != 2 or len(self.metadata) != self.values.shape[0]:
            raise ValueError("PLL memory rows must align with metadata")
        if set(self.families) != set(FAMILY_LEVELS):
            raise ValueError("PLL memory requires four relation families")
        if not torch.isfinite(self.values).all():
            raise ValueError("PLL memory must be finite")
        for values in self.families.values():
            if (values.ndim != 2 or values.shape[1] != self.values.shape[1]
                    or values.shape[0] < 1 or values.device != self.values.device
                    or not torch.isfinite(values).all()):
                raise ValueError("PLL relation family memory is invalid")


class HierarchicalMemory(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig(),
                 profile: ContextProfile = ContextProfile()):
        super().__init__()
        self.config, self.profile = config, profile
        self.projections = nn.ModuleDict({level: nn.Linear(128, config.width) for level in LEVELS})
        self.level_embedding = nn.Embedding(len(LEVELS), config.width)
        self.stage_embedding = nn.Embedding(len(STAGES), config.width)
        self.track_embedding = nn.Embedding(profile.max_tracks + 1, config.width)
        self.geometry_projection = nn.Linear(5, config.width)
        self.query_vector = nn.Parameter(torch.zeros(128))
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.ff_dim,
                    config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.global_transformer = nn.TransformerEncoder(
            layer, config.global_layers, norm=nn.LayerNorm(config.width), enable_nested_tensor=False)
        self.fine_norm = nn.LayerNorm(config.width)

    def _project(self, value: Tensor, token: MemoryToken, view: ObservedMusicView) -> Tensor:
        span = (view.global_end - view.global_start).to_fraction()
        start = float((token.start - view.global_start).to_fraction() / span)
        end = float((token.end - view.global_start).to_fraction() / span)
        selected = (token.track is not None and
                    view.piece.tracks[token.track].track_id in view.query.track_ids)
        geometry = value.new_tensor((start, end, token.coverage, float(token.empty), float(selected)))
        track = self.profile.max_tracks if token.track is None else token.track
        return (self.projections[token.level](value)
                + self.level_embedding.weight[LEVELS.index(token.level)]
                + self.stage_embedding.weight[STAGES.index(token.stage)]
                + self.track_embedding.weight[track] + self.geometry_projection(geometry))

    @staticmethod
    def _mean(values: list[Tensor], zero: Tensor) -> Tensor:
        return torch.stack(values).mean(0) if values else zero

    def forward(self, encoded_tiles: tuple[EncodedTile, ...], view: ObservedMusicView) -> ContextMemory:
        if not encoded_tiles or not view.piece.bars or not view.piece.tracks:
            raise ValueError("PLL memory needs tiles, metric bars and known tracks")
        if len(view.piece.bars) > self.profile.context_bars or len(view.piece.tracks) > self.profile.max_tracks:
            raise UnsupportedContextError("pll.memory_context_limit")
        owners = {}
        for encoded in encoded_tiles:
            if encoded.tile.query != view.query:
                raise ValueError("PLL tile and view query disagree")
            indices = {bar.bar_id: i for i, bar in enumerate(encoded.tile.piece.bars)}
            for bar_id in encoded.tile.core_bar_ids:
                if bar_id in owners or bar_id not in indices:
                    raise ValueError("PLL core bars must have exactly one tile owner")
                owners[bar_id] = encoded, indices[bar_id]
        if set(owners) != {b.bar_id for b in view.piece.bars}:
            raise ValueError("PLL core tile coverage does not match the observed window")
        reference = encoded_tiles[0].embeddings["bar"]
        zero = reference.new_zeros(128)
        raw: list[Tensor] = []
        metadata: list[MemoryToken] = []
        bars = view.piece.bars
        tracks = view.piece.tracks
        cell_values: dict[tuple[int, int], Tensor] = {}
        bar_values: list[Tensor] = []

        def add(value: Tensor, token: MemoryToken):
            raw.append(value)
            metadata.append(token)

        for bi, bar in enumerate(bars):
            encoded, row = owners[bar.bar_id]
            value = encoded.embeddings["bar"][row]
            bar_values.append(value)
            coverage = sum(view.coverage(bar.bar_id, t.track_id) for t in tracks) / len(tracks)
            add(value, MemoryToken("bar", bar.start_qn, bar.start_qn + bar.duration_qn,
                                   None, coverage, stage=encoded.stages["bar"]))
            by_track: dict[str, list[Tensor]] = defaultdict(list)
            for ni, (note, geometry) in enumerate(zip(encoded.tile.piece.notes, encoded.tile.note_geometry)):
                # A cell represents sounding content, including carry-in. The
                # same sustain may belong to different bars, but each cell is
                # built only from its one core owner, never from another halo.
                if self._intersects(geometry.onset, geometry.onset + geometry.duration,
                                    bar.start_qn, bar.start_qn + bar.duration_qn):
                    by_track[note.track_id].append(encoded.embeddings["note"][ni])
            for ti, track in enumerate(tracks):
                rows = by_track[track.track_id]
                value = self._mean(rows, zero)
                cell_values[bi, ti] = value
                add(value, MemoryToken("track_bar", bar.start_qn, bar.start_qn + bar.duration_qn,
                    ti, view.coverage(bar.bar_id, track.track_id), not rows,
                    encoded.stages["note"], len(rows)))

        for ti, track in enumerate(tracks):
            # Each cell occurs once, including known-empty and unknown cells.
            value = torch.stack([cell_values[bi, ti] for bi in range(len(bars))]).mean(0)
            local_track = []
            for encoded in encoded_tiles:
                i = next(i for i,t in enumerate(encoded.tile.piece.tracks) if t.track_id == track.track_id)
                local_track.extend([encoded.embeddings["track"][i]] * len(encoded.tile.core_bar_ids))
            value = value + torch.stack(local_track).mean(0)
            coverage = sum(view.coverage(b.bar_id, track.track_id) for b in bars) / len(bars)
            add(value, MemoryToken("track", view.global_start, view.global_end, ti,
                                   coverage, stage="pooled"))

        for scale in (2, 4, 8):
            for offset in range(0, len(bars) - scale + 1, scale):
                group = bars[offset:offset + scale]
                coverage = sum(view.coverage(b.bar_id, t.track_id) for b in group for t in tracks) / (scale * len(tracks))
                add(torch.stack(bar_values[offset:offset + scale]).mean(0),
                    MemoryToken(f"group{scale}", group[0].start_qn,
                    group[-1].start_qn + group[-1].duration_qn, None, coverage))
        songs = [e.embeddings["song"][0] for e in encoded_tiles for _ in e.tile.core_bar_ids]
        coverage = sum(view.coverage(b.bar_id, t.track_id) for b in bars for t in tracks) / (len(bars) * len(tracks))
        add(torch.stack(songs).mean(0) + torch.stack(bar_values).mean(0),
            MemoryToken("context", view.global_start, view.global_end, None, coverage))
        query_tracks = [i for i,t in enumerate(tracks) if t.track_id in view.query.track_ids]
        add(self.query_vector, MemoryToken("query", view.query.start, view.query.end, None, 0))
        if len(raw) > self.profile.max_memory:
            raise UnsupportedContextError("pll.global_memory_limit")
        projected = torch.stack([self._project(v, m, view) for v,m in zip(raw, metadata)])
        # Query identifies the known target track subset without file IDs or labels.
        projected = torch.cat((projected[:-1], (projected[-1] +
            self.track_embedding.weight[query_tracks].mean(0)).unsqueeze(0)))
        global_values = self.global_transformer(projected.unsqueeze(0)).squeeze(0)
        budget = min(self.profile.max_fine_memory, self.profile.max_memory - len(raw))
        if self.profile.max_fine_memory and budget == 0:
            raise UnsupportedContextError("pll.no_capacity_for_fine_memory")
        fine = self._fine_rows(encoded_tiles, view, budget)
        if fine:
            fine_values = self.fine_norm(torch.stack([self._project(v,m,view) for v,m in fine]))
            values = torch.cat((global_values, fine_values), dim=0)
            metadata.extend(m for _,m in fine)
        else:
            values = global_values
        families = {name: values[[i for i,m in enumerate(metadata) if m.level in levels]]
                    for name,levels in FAMILY_LEVELS.items()}
        return ContextMemory(values, families, tuple(metadata), view.query)

    @staticmethod
    def _intersects(start, end, left, right):
        return left <= start < right if start == end else start < right and end > left

    @staticmethod
    def _interval_coverage(view, start, end):
        """Observed fraction for a metric token, including point boundaries."""
        if start == end:
            hidden = float(view.query.start <= start < view.query.end)
        else:
            overlap = min(end, view.query.end) - max(start, view.query.start)
            hidden = max(0., float(overlap.to_fraction() / (end-start).to_fraction()))
        return 1. - hidden * len(view.query.track_ids) / len(view.piece.tracks)

    def _fine_rows(self, encoded_tiles, view, budget):
        if not budget:
            return []
        bars = view.piece.bars
        hit = [i for i,b in enumerate(bars) if b.start_qn < view.query.end
               and b.start_qn + b.duration_qn > view.query.start]
        first, last = max(0, hit[0]-1), min(len(bars)-1, hit[-1]+1)
        start, end = bars[first].start_qn, bars[last].start_qn + bars[last].duration_qn
        track_indices = {t.track_id:i for i,t in enumerate(view.piece.tracks)}
        groups = defaultdict(list)
        seen = set()
        for encoded in encoded_tiles:
            tile = encoded.tile
            core = [b for b in tile.piece.bars if b.bar_id in tile.core_bar_ids]
            core_start = core[0].start_qn + tile.global_start
            core_end = core[-1].start_qn + core[-1].duration_qn + tile.global_start
            for level in ("note_local", "note", "onset", "beat", "slice"):
                rows = encoded.embeddings[level]
                if level in {"note_local", "note"}:
                    geometries = [(g.onset, g.onset + g.duration, track_indices[n.track_id], g.note_id)
                                  for n,g in zip(tile.piece.notes, tile.note_geometry)]
                elif level == "onset":
                    geometries = []
                    for _,key in encoded.graph.keys["onset"]:
                        num,den = key.removeprefix("onset:").split("_")
                        time = Q(int(num), int(den)) + tile.global_start
                        geometries.append((time,time,None,str(time)))
                elif level == "beat":
                    geometries = [(b.start_qn + tile.global_start,
                        b.start_qn + b.duration_qn + tile.global_start, None,b.beat_id) for b in tile.piece.beats]
                else:
                    times = [Q.from_fraction(t) + tile.global_start for t,_ in encoded.graph.slots[0].time_slots]
                    geometries = [(t, times[i+1] if i+1<len(times) else t,None,str(t)) for i,t in enumerate(times)]
                if len(geometries) != len(rows):
                    raise ValueError("PLL fine geometry does not align with encoder rows")
                for value,(a,b,track,key) in zip(rows,geometries):
                    # Core ownership removes halo duplicates; terminal slices are
                    # retained only by the last core, never by both neighbors.
                    if level in {"note_local", "note"}:
                        owned = (core_start <= a < core_end or
                                 (a < view.global_start == core_start and b > core_start))
                        near = self._intersects(a, b, start, end)
                    else:
                        owned = core_start <= a < core_end or (a == core_end == view.global_end)
                        near = start <= a <= end
                    if not owned or not near or (level,key) in seen:
                        continue
                    seen.add((level,key))
                    coverage = 1. if level in {"note_local", "note"} else self._interval_coverage(view,a,b)
                    token = MemoryToken(level,a,b,track,coverage,stage=encoded.stages[level])
                    groups[level,track].append((value,token))
        if not groups:
            return []
        if sum(map(len,groups.values())) <= budget:
            return [row for rows in groups.values() for row in rows]
        if budget < len(groups):
            raise UnsupportedContextError("pll.fine_memory_cannot_retain_each_level_and_track")
        # Fixed geometry bins, separately per level/track. Occupancy chooses
        # which known raw rows are pooled, never target-derived bin boundaries.
        quota = budget // len(groups)
        result = []
        width = (end-start).to_fraction()
        for (level,track), rows in groups.items():
            if len(rows) <= quota:
                result.extend(rows)
                continue
            bins = defaultdict(list)
            for value,token in rows:
                index = max(0, min(quota-1, int((token.start-start).to_fraction() / width * quota)))
                bins[index].append((value,token))
            for index, chunk in sorted(bins.items()):
                a = start + Q.from_fraction(width * Fraction(index,quota))
                b = start + Q.from_fraction(width * Fraction(index+1,quota))
                result.append((torch.stack([v for v,_ in chunk]).mean(0),
                    MemoryToken(level,a,b,track,
                        1. if level in {"note_local", "note"} else self._interval_coverage(view,a,b),
                        stage=chunk[0][1].stage,pooled_rows=len(chunk))))
        return result
