"""Observed-only overlapping motif bank and differentiable repeat memory.

A candidate is a fully observed metric interval, not a semantic assertion that
it is a motif or refrain. All candidates remain in soft retrieval; geometric
selection only limits detailed reads. No target event enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import math

import torch
from torch import Tensor

from torch import nn

from music_critic.data.timing import RationalTime as Q
from .contracts import ModelConfig
from .encoder import EncodedTile
from .memory import MemoryToken
from .observed_view import ObservedMusicView, UnsupportedContextError


@dataclass(frozen=True)
class FragmentMetadata:
    candidate_id: str
    start: Q
    end: Q
    scale: int
    unit: str
    first_bar: int
    last_bar: int
    observed_note_count: int


def fragment_bank(view: ObservedMusicView, *, scales=(1, 2, 4, 8, 16, 32),
                  beat_scales=(1, 2), max_candidates=8192) -> tuple[FragmentMetadata, ...]:
    """Stride-one windows; any temporal overlap with the mask is excluded.

    Exclusion applies across every track even for a track-specific mask, so
    a retained candidate always has complete observed coverage. Insufficient
    length is represented by zero candidates at that scale, never by clipping.
    """
    if not isinstance(view, ObservedMusicView):
        raise TypeError('pll.motif_observed_view_required')
    for sizes in (scales, beat_scales):
        if len(set(sizes)) != len(sizes) or any(type(s) is not int or s < 1 for s in sizes):
            raise ValueError('pll.invalid_motif_scales')
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('pll.invalid_motif_candidate_limit')
    bars = view.piece.bars
    result = []
    sounding_starts = sorted(n.onset_qn for n in view.piece.notes if n.duration_qn > Q(0))
    sounding_ends = sorted(n.onset_qn+n.duration_qn for n in view.piece.notes if n.duration_qn > Q(0))
    grace_onsets = sorted(n.onset_qn for n in view.piece.notes if n.duration_qn == Q(0))
    for unit, geometry, sizes in (('bar', bars, scales), ('beat', view.piece.beats, beat_scales)):
        for scale in sizes:
            for i in range(len(geometry)-scale+1):
                window = geometry[i:i+scale]
                if any(a.start_qn+a.duration_qn != b.start_qn for a,b in zip(window, window[1:])):
                    continue
                start, end = window[0].start_qn, window[-1].start_qn+window[-1].duration_qn
                if start < view.query.end and end > view.query.start:
                    continue
                if len(result) >= max_candidates:
                    raise UnsupportedContextError('pll.motif_candidate_limit')
                hit = [bi for bi,b in enumerate(bars) if b.start_qn < end and b.start_qn+b.duration_qn > start]
                count = (bisect_left(sounding_starts,end)-bisect_right(sounding_ends,start)
                         + bisect_left(grace_onsets,end)-bisect_left(grace_onsets,start))
                result.append(FragmentMetadata(f'{unit}:{scale}:{i}', start, end, scale, unit,
                                                hit[0], hit[-1], count))
    return tuple(result)


@dataclass(frozen=True)
class CandidateAlignment:
    candidate_index: int
    input_rows: int
    output_rows: int
    positions: tuple[tuple[Q, Q], ...]
    query_positions: tuple[tuple[Q, Q], ...]
    attention: Tensor


@dataclass(frozen=True)
class MotifResult:
    values: Tensor
    metadata: tuple[MemoryToken, ...]
    candidates: tuple[FragmentMetadata, ...]
    weights: Tensor  # Last entry is the learned null branch.
    selected_indices: tuple[int, ...]
    alignments: tuple[CandidateAlignment, ...]
    candidate_keys: Tensor
    scale_counts: dict[str, int]
    selected_scale_counts: dict[str, int]

    def diagnostics(self):
        """JSON-safe receipts only; never retain a training autograd graph."""
        def time(value):
            return [value.num, value.den]
        weights = self.weights.detach().cpu().tolist()
        return {
            'scale_counts': dict(self.scale_counts),
            'selected_scale_counts': dict(self.selected_scale_counts),
            'null_weight': weights[-1],
            'candidates': [dict(candidate_id=c.candidate_id, start=time(c.start),
                end=time(c.end), scale=c.scale, unit=c.unit, first_bar=c.first_bar,
                last_bar=c.last_bar, observed_note_count=c.observed_note_count,
                weight=weights[i]) for i,c in enumerate(self.candidates)],
            'selected_indices': list(self.selected_indices),
            'alignments': [dict(candidate_index=a.candidate_index,
                input_rows=a.input_rows, output_rows=a.output_rows,
                positions=[[time(x),time(y)] for x,y in a.positions],
                query_positions=[[time(x),time(y)] for x,y in a.query_positions])
                for a in self.alignments],
        }


def _content_views(notes, start, end):
    """Four 24D raw views; onset bins preserve coarse order, not voice labels.

    Relative pitch is an onset-bin centroid centered on the fragment's pitched
    notes. It is not an invented monophonic melody extracted from a polyphonic
    track. The independent absolute-pitch view prevents forced invariance.
    """
    result = [[0.] * 24 for _ in range(4)]
    span = (end-start).to_fraction()
    sounding = [n for n in notes if _intersects(n.onset_qn,n.onset_qn+n.duration_qn,start,end)]
    pitched = [n for n in sounding if not n.is_percussion]
    center = sum(n.pitch for n in pitched)/len(pitched) if pitched else 0.
    minimum_pitch = min((n.pitch for n in pitched),default=0)
    bin_counts = [0] * 8
    attack_count = 0
    for note in sounding:
        attack = start <= note.onset_qn < end
        attack_count += int(attack)
        phase = (max(note.onset_qn,start)-start).to_fraction()/span
        slot = min(7, (phase*8).numerator//(phase*8).denominator)
        percussion_slot = min(5, (phase*6).numerator//(phase*6).denominator)
        duration = float((min(note.onset_qn+note.duration_qn,end)-max(note.onset_qn,start)).to_fraction()/span)
        result[0][slot] += float(attack)
        result[0][8+slot] += duration
        result[3][int(note.track_id.split(':')[-1])] += 1.
        result[3][16+percussion_slot] += float(note.is_percussion)
        if not note.is_percussion:
            bin_counts[slot] += 1
            result[1][slot] += (note.pitch-center)/127.
            result[1][8+(note.pitch-minimum_pitch) % 12] += 1.
            result[2][slot] += note.pitch/127.
            result[2][8+note.pitch % 12] += 1.
    for slot, count in enumerate(bin_counts):
        if count:
            result[1][slot] /= count
            result[2][slot] /= count
    count = max(1,len(sounding))
    for j in range(16):
        result[0][j] /= count
        result[3][j] /= count
    for j in range(16,24):
        result[3][j] /= count
    for j in range(8,20):
        result[1][j] /= max(1,len(pitched))
        result[2][j] /= max(1,len(pitched))
    for row in result:
        row[22] = float(attack_count == 0)
        row[23] = float(not sounding)
    result[0][16] = math.log1p(attack_count)/8.
    result[0][17] = math.log1p(float(span))/8.
    result[0][18] = (len(sounding)-attack_count)/count
    if pitched:
        result[1][20] = (max(n.pitch for n in pitched)-minimum_pitch)/127.
        result[2][20] = center/127.
    return result


def _intersects(start,end,left,right):
    return left <= start < right if start == end else start < right and end > left


class MotifMemory(nn.Module):
    """Soft retrieval over every eligible candidate plus bounded detail reads."""

    def __init__(self, config: ModelConfig = ModelConfig(), whole=None):
        super().__init__()
        if whole is None:
            from .whole_contracts import WholePieceConfig
            whole = WholePieceConfig()
        self.config, self.whole = config, whole
        w = config.width
        self.raw_projections = nn.ModuleList(nn.Linear(24, w) for _ in range(4))
        self.sequence_projection = nn.Linear(4*w, w)
        self.key_projection = nn.Linear(w, w)
        self.query_projection = nn.Linear(w, w)
        self.geometry_projection = nn.Linear(5, w)
        self.view_logits = nn.Parameter(torch.zeros(4))
        self.value_projection = nn.Linear(w, w)
        self.note_projection = nn.Linear(128, w)
        self.position_projection = nn.Linear(4, w)
        self.null_key = nn.Parameter(torch.zeros(w))
        self.null_value = nn.Parameter(torch.zeros(w))
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.aligner = nn.MultiheadAttention(w, config.heads, config.dropout, batch_first=True)
        self.detail_norm = nn.LayerNorm(w)

    def _summary(self, bars: Tensor) -> Tensor:
        # Ordered four-bin pooling retains the full candidate, including 16/32
        # bars. It is deliberately a summary, not a lossless sequence encoding.
        rows = bars.mean(1)
        bins = []
        for i in range(4):
            left, right = i*len(rows)//4, (i+1)*len(rows)//4
            bins.append(rows[left:right].mean(0) if right > left else rows[min(left,len(rows)-1)])
        return self.sequence_projection(torch.cat(bins))

    @staticmethod
    def _selected(candidates, weights, top_k, threshold):
        selected = []
        for index in torch.argsort(weights[:-1].detach(), descending=True, stable=True).tolist():
            candidate = candidates[index]
            duplicate = False
            for other_index in selected:
                other = candidates[other_index]
                overlap = min(candidate.end,other.end)-max(candidate.start,other.start)
                ratio = float(overlap.to_fraction()/min(candidate.end-candidate.start,
                                                         other.end-other.start).to_fraction())
                if ratio > threshold:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(index)
                if len(selected) == top_k:
                    break
        return tuple(selected)

    def _query(self, view, bars, globals):
        hits = [i for i,b in enumerate(view.piece.bars) if b.start_qn < view.query.end
                and b.start_qn+b.duration_qn > view.query.start]
        indices = sorted(set(range(max(0,hits[0]-2), min(len(bars),hits[-1]+3))))
        # These are already masked states. Unknown cells are not target notes.
        local = bars[indices].mean((0,1))
        span = float((view.global_end-view.global_start).to_fraction())
        a = float((view.query.start-view.global_start).to_fraction())/span
        b = float((view.query.end-view.global_start).to_fraction())/span
        geometry = bars.new_tensor([a,b,len(view.query.track_ids)/len(view.piece.tracks),
                                    float(hits[0] == 0),float(hits[-1] == len(bars)-1)])
        query = self.query_projection(local+globals.mean(0)+self.geometry_projection(geometry))
        positions = tuple((view.piece.bars[i].start_qn,
                           view.piece.bars[i].start_qn+view.piece.bars[i].duration_qn) for i in indices)
        query_rows = bars[indices].mean(1)
        encoded = [self._position(a,b,view.query.start,view.query.end,query_rows) for a,b in positions]
        return query, query_rows+torch.stack(encoded), positions

    def _position(self, start, end, origin, finish, ref):
        span = (finish-origin).to_fraction()
        left = float((start-origin).to_fraction()/span)
        right = float((end-origin).to_fraction()/span)
        # tanh keeps very distant positions bounded without float comparisons.
        return self.position_projection(ref.new_tensor([math.tanh(left),math.tanh(right),
                                                        math.sin(math.pi*left),math.cos(math.pi*left)]))

    @staticmethod
    def _notes(encoded_tiles, view):
        rows, seen = [], set()
        for encoded in encoded_tiles:
            if encoded.tile.query != view.query:
                raise ValueError('pll.motif_tile_query_mismatch')
            core = set(encoded.tile.core_note_ids)
            for index, note in enumerate(encoded.tile.note_geometry):
                if note.note_id not in core:
                    continue
                if note.note_id in seen:
                    raise ValueError('pll.motif_duplicate_core_note')
                seen.add(note.note_id)
                rows.append((note.onset, note.onset+note.duration, encoded.embeddings['note'][index]))
        return rows

    def _details(self, candidate, bars, note_rows, view):
        rows = [(max(b.start_qn,candidate.start),
                 min(b.start_qn+b.duration_qn,candidate.end),bars[i].mean(0))
                for i,b in enumerate(view.piece.bars)
                if b.start_qn < candidate.end and b.start_qn+b.duration_qn > candidate.start]
        rows += [(a,b,self.note_projection(value)) for a,b,value in note_rows
                 if _intersects(a,b,candidate.start,candidate.end)]
        if not rows:  # A sub-bar observed rest still has a valid contextual state.
            rows = [(candidate.start,candidate.end,bars[candidate.first_bar].mean(0))]
        rows.sort(key=lambda row:(row[0],row[1]))
        count = len(rows)
        budget = self.whole.motif_details_per_candidate
        output, positions = [], []
        for i in range(min(budget,count)):
            group = rows[i*count//min(budget,count):(i+1)*count//min(budget,count)]
            start, end = min(v[0] for v in group),max(v[1] for v in group)
            # Positional encodings are applied before pooling, so every source
            # row contributes. A long candidate never becomes a prefix-only read.
            output.append(torch.stack([v+self._position(a,b,candidate.start,candidate.end,v)
                                       for a,b,v in group]).mean(0))
            positions.append((start,end))
        return torch.stack(output), tuple(positions), count

    def forward(self, encoded_tiles: tuple[EncodedTile,...], view: ObservedMusicView, *,
                bar_values: Tensor, global_values: Tensor) -> MotifResult:
        w = self.config.width
        if (bar_values.shape != (len(view.piece.bars),self.whole.bar_slots,w) or global_values.ndim != 2
                or global_values.shape[1] != w or not len(global_values)
                or global_values.device != bar_values.device
                or not torch.isfinite(bar_values).all() or not torch.isfinite(global_values).all()):
            raise ValueError('pll.motif_memory_input_shape')
        candidates = fragment_bank(view, scales=self.whole.motif_scales,
                    beat_scales=self.whole.motif_beat_scales,
                    max_candidates=self.whole.max_motif_candidates)
        scale_counts = {f'{unit}:{s}':sum(c.unit == unit and c.scale == s for c in candidates)
                        for unit,sizes in (('bar',self.whole.motif_scales),
                                           ('beat',self.whole.motif_beat_scales)) for s in sizes}
        query, query_rows, query_positions = self._query(view,bar_values,global_values)
        features, values = [], []
        view_weights = torch.softmax(self.view_logits,0)
        observed_notes = sorted(view.piece.notes,key=lambda n:n.onset_qn)
        onsets = [n.onset_qn for n in observed_notes]
        longest = max((n.duration_qn for n in observed_notes),default=Q(0))
        for candidate in candidates:
            rows = bar_values[candidate.first_bar:candidate.last_bar+1]
            content = bar_values.new_tensor(_content_views(observed_notes[bisect_left(onsets,candidate.start-longest):bisect_left(onsets,candidate.end)],candidate.start,candidate.end))
            views = torch.stack([self.raw_projections[i](content[i])+rows[:,i % self.whole.bar_slots].mean(0)
                                 for i in range(4)])
            summary = self._summary(rows)
            features.append((views*view_weights[:,None]).sum(0)+summary)
            values.append(self.value_projection(views.mean(0)+summary))
        keys = (self.key_projection(torch.stack(features)) if features
                else bar_values.new_empty((0,w)))
        logits = keys @ query/math.sqrt(w)
        null_logit = (self.null_key @ query/math.sqrt(w)+self.null_bias).reshape(1)
        weights = torch.softmax(torch.cat((logits,null_logit)),0)
        projected = torch.stack(values) if values else bar_values.new_empty((0,w))
        soft = (weights[:-1,None]*projected).sum(0)+weights[-1]*self.null_value
        result = [soft]
        metadata = [MemoryToken('motif_summary',view.query.start,view.query.end,None,1.,
                                empty=not bool(candidates),pooled_rows=len(candidates))]
        selected = self._selected(candidates,weights,self.whole.motif_top_k,
                                  self.whole.motif_overlap_threshold)
        note_rows = self._notes(encoded_tiles,view)
        alignments = []
        for index in selected:
            candidate = candidates[index]
            details, positions, count = self._details(candidate,bar_values,note_rows,view)
            aligned, attention = self.aligner(details[None],query_rows[None],query_rows[None],
                                               need_weights=True,average_attn_weights=True)
            detailed = self.detail_norm(details+aligned[0])*weights[index]
            result.extend(detailed.unbind(0))
            metadata.extend(MemoryToken('motif_detail',a,b,None,1.,
                            empty=candidate.observed_note_count == 0,
                            pooled_rows=(j+1)*count//len(positions)-j*count//len(positions))
                            for j,(a,b) in enumerate(positions))
            alignments.append(CandidateAlignment(index,count,len(positions),positions,
                                                   query_positions,attention[0]))
        return MotifResult(torch.stack(result),tuple(metadata),candidates,weights,selected,
                           tuple(alignments),keys,scale_counts,
                           {key:sum(f'{candidates[i].unit}:{candidates[i].scale}' == key
                                    for i in selected) for key in scale_counts})
