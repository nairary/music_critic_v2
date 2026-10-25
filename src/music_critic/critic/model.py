"""Three recurrent scales with raw-incidence fusion and aspect-specific readouts."""
from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import (ASPECTS, CriticFeaturePyramid, CriticOutput, FeatureSample,
                        LocalPrediction, SamplePrediction)
from .geometry_plan import geometry_plan
from .device_geometry import prepare_constants

LEVELS = ('note', 'event', 'slice', 'beat', 'bar', 'span2', 'span4', 'span8', 'track', 'song')
INPUT_LEVELS = ('note_local', 'note', 'onset', 'beat', 'bar', 'slice', 'track', 'song')


class SetPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.projection = nn.Linear(2 * width + 2, width)

    def forward(self, values, constants=None):
        if not len(values):
            return values.new_zeros(self.width)
        if constants is None:
            count, one = values.new_tensor([len(values)]).log1p(), values.new_ones(1)
        else:
            count, one = constants.scalars(len(values))
        return F.gelu(self.projection(torch.cat((values.mean(0), values.amax(0), count, one))))


class GatedMessage(nn.Module):
    def __init__(self, width, relations=8):
        super().__init__()
        self.relations = nn.Embedding(relations, width)
        self.gate = nn.Linear(3 * width, width)
        self.message = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, receiver, message, relation):
        r = self.relations.weight[relation]
        gate = torch.sigmoid(self.gate(torch.cat((receiver, message, r))))
        return self.norm(receiver + gate * self.message(message))


class AspectReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(len(LEVELS), width) / sqrt(width))
        self.level_gate = nn.Linear(width, 1)
        self.score = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, levels, zero):
        pooled = []
        for i, name in enumerate(LEVELS):
            states = levels.get(name)
            if states is None or not len(states):
                continue
            attention = torch.softmax(states @ self.queries[i] / sqrt(states.shape[-1]), dim=0)
            pooled.append((attention[:, None] * states).sum(0))
        if not pooled:
            return zero.sum() * 0
        stacked = torch.stack(pooled)
        gates = torch.softmax(self.level_gate(stacked).squeeze(-1), dim=0)
        return self.score((gates[:, None] * stacked).sum(0)).squeeze(-1)


class MultiscaleAspectCritic(nn.Module):
    """Per-piece processing makes sequence ownership independent of batch padding.

    No encoder is registered here: training this module cannot change a backbone.
    Input feature tensors are detached defensively. Metadata is used for exact joins.
    """
    def __init__(self, feature_dim=128, width=128):
        super().__init__()
        if width < 2 or width % 2 or feature_dim < 1:
            raise ValueError('positive feature_dim and even width >=2 required')
        self.feature_dim, self.width = feature_dim, width
        self.projections = nn.ModuleDict({k: nn.Linear(feature_dim, width) for k in INPUT_LEVELS})
        self.note_mix = nn.Linear(2 * width + 4, width)
        self.pool = SetPool(width)
        self.event_input = nn.Linear(2 * width + 4, width)
        self.slice_input = nn.Linear(6 * width + 4, width)
        self.bar_input = nn.Linear(4 * width + 4, width)
        self.track_gru = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.slice_gru = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.bar_gru = nn.GRU(width, width // 2, bidirectional=True, batch_first=True)
        self.fusion = GatedMessage(width)
        self.readouts = nn.ModuleDict({a: AspectReadout(width) for a in ASPECTS})
        self.local_heads = nn.ModuleDict({a: nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))
                                          for a in ASPECTS})
        self.boundary = nn.Linear(2 * width, width)

    def forward(self, pyramid: CriticFeaturePyramid) -> CriticOutput:
        return CriticOutput(tuple(self._sample(s) for s in pyramid.samples))

    def _sample(self, sample: FeatureSample) -> SamplePrediction:
        w = self.width
        zero = self.note_mix.weight.new_zeros(w)
        empty = zero.new_empty((0, w))
        plan = geometry_plan(sample.geometry)
        constants = prepare_constants(plan, zero)
        geo = plan.geometry
        notes = geo.notes
        projected = {}
        for name, values in sample.embeddings.items():
            if name not in self.projections:
                raise ValueError(f'unsupported feature level: {name}')
            expected = (len(notes) if name in ('note', 'note_local') else
                        len(geo.tracks) if name == 'track' else 1 if name == 'song' else
                        len(geo.intervals.get(name, ())))
            if values.shape != (expected, self.feature_dim) or not torch.isfinite(values).all():
                raise ValueError(f'feature shape/finite mismatch: {name}')
            projected[name] = F.gelu(self.projections[name](values.detach().to(zero)))
        if notes and 'note' not in projected:
            raise ValueError('note representation required')
        get = lambda name: projected.get(name, empty)
        def rows(values, indices):
            return values[constants.indices(indices)] if indices else empty

        def pool(values):
            return self.pool(values, constants.pool)

        def pool_indices(values, indices):
            return pool(rows(values, indices))

        def sequence(gru, values):
            return gru(values.unsqueeze(0))[0].squeeze(0) if len(values) else empty

        def weighted(values, weights):
            if not weights.entries:
                return zero
            tensor = constants.weights(weights)
            return (tensor[:, None] * values).sum(0)

        local = get('note_local')
        note = torch.stack([F.gelu(self.note_mix(torch.cat((get('note')[i],
            local[i] if len(local) else zero, constants.timing(n.timing)))))
            for i, n in enumerate(plan.notes)]) if notes else empty
        events = plan.events
        track = get('track')
        event = torch.stack([F.gelu(self.event_input(torch.cat((
            pool_indices(note, e.notes),
            track[e.track] if len(track) else zero, constants.timing(e.timing)))))
            for e in events]) if events else empty
        if len(event):
            updated = event.new_zeros(event.shape)
            for t in plan.tracks:
                indexes = t.events
                if indexes:
                    index = constants.indices(indexes)
                    updated = updated.index_copy(0, index, sequence(self.track_gru, event[index]))
            event = updated
        horizontal_notes = torch.stack([self.fusion(note[i], event[n.event], 0)
                                        for i, n in enumerate(plan.notes)]) if notes else empty
        slices = plan.slices
        bars = plan.bars
        slice_rows = []
        for i, s in enumerate(slices):
            slice_rows.append(F.gelu(self.slice_input(torch.cat((
                pool_indices(horizontal_notes, s.attacks), pool_indices(horizontal_notes, s.holds),
                pool_indices(horizontal_notes, s.releases), get('slice')[i] if len(get('slice')) else zero,
                pool_indices(get('beat'), s.beats) if len(get('beat')) else zero,
                pool_indices(get('onset'), s.onsets) if len(get('onset')) else zero,
                constants.timing(s.timing))))))
        vertical = sequence(self.slice_gru, torch.stack(slice_rows) if slice_rows else empty)
        old_tracks = rows(get('track'), plan.active_tracks) if len(get('track')) else empty
        old_global = pool(torch.cat((old_tracks, get('song')), dim=0))
        bar_rows = []
        for i, b in enumerate(bars):
            bar_rows.append(F.gelu(self.bar_input(torch.cat((weighted(vertical, b.weights),
                pool_indices(event, b.events), get('bar')[i] if len(get('bar')) else zero,
                old_global, constants.timing(b.timing))))))
        bar = sequence(self.bar_gru, torch.stack(bar_rows) if bar_rows else empty)

        def make_spans(states):
            spans = {}
            for size, groups in zip((2, 4, 8), plan.span_groups):
                spans[f'span{size}'] = torch.stack([states[a:b].mean(0) for a, b in groups]) if groups else empty
            return spans

        spans = make_spans(bar)
        refined = []
        for i, s in enumerate(slices):
            value = vertical[i]
            parents = s.bars
            if parents:
                value = self.fusion(value, pool_indices(bar, parents), 1)
            for r, name in enumerate(('span2', 'span4', 'span8'), 2):
                indexes = s.spans[r-2]
                if indexes:
                    value = self.fusion(value, pool_indices(spans[name], indexes), r)
            refined.append(value)
        vertical = torch.stack(refined) if refined else empty
        note_rows = []
        for i, n in enumerate(plan.notes):
            indexes = n.slices
            note_rows.append(self.fusion(horizontal_notes[i], pool_indices(vertical, indexes), 5)
                             if indexes else horizontal_notes[i])
        note = torch.stack(note_rows) if note_rows else empty
        event_rows = []
        for i, e in enumerate(events):
            indexes = e.slices
            event_rows.append(self.fusion(event[i], pool_indices(vertical, indexes), 5)
                              if indexes else event[i])
        event = torch.stack(event_rows) if event_rows else empty
        # Repool upward only after the updated lower-level states are available.
        vertical = torch.stack([self.fusion(vertical[i], pool(torch.cat((
            rows(note, s.notes), rows(event, s.events)), dim=0)), 6)
            for i,s in enumerate(slices)]) if slices else empty
        bar = torch.stack([self.fusion(bar[i], pool(torch.stack((
            weighted(vertical, b.weights), pool_indices(event, b.events)))), 7)
            for i,b in enumerate(bars)]) if bars else empty
        beat = torch.stack([weighted(vertical, b) for b in plan.beat_weights]) if plan.beat_weights else empty
        new_track = torch.stack([pool_indices(note, plan.tracks[i].notes)
                                 for i in plan.active_tracks]) if notes else empty
        spans = make_spans(bar)
        song = pool(torch.cat((bar, new_track, note), dim=0)).unsqueeze(0) if notes else empty
        levels = dict(note=note, event=event, slice=vertical, beat=beat, bar=bar,
                      track=new_track, song=song, **spans)
        candidates = dict(plan.candidates)
        available = torch.tensor(plan.availability, device=zero.device)
        scores = torch.stack([self.readouts[a](levels, zero) for a in ASPECTS])
        scores = torch.where(available, scores, torch.zeros_like(scores))
        boundaries = torch.stack([F.gelu(self.boundary(torch.cat((bar[i],bar[i+1]))))
            for i in plan.boundaries]) if candidates['structure/boundary'] else empty
        local_states = {'note':note, 'event':event, 'slice':vertical, 'beat':beat, 'boundary':boundaries}
        predictions = {}
        for (key, items), mask in zip(plan.candidates, plan.local_masks):
            aspect, kind = key.split('/')
            predictions[key] = LocalPrediction(aspect, kind, items,
                self.local_heads[aspect](local_states[kind]).squeeze(-1),
                torch.tensor(mask, dtype=torch.bool, device=zero.device))
        return SamplePrediction(geo.sample_id, scores, available, predictions, geo.duration,
                                contextual=levels)
