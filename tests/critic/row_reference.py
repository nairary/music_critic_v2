"""Test reference: source004 _sample verbatim, before row batching.

Source model SHA256: cecfd06e673f2b0a5b9256c79bcbbed68906bfd3d8937f793e350ad519d1d9fe.
Architecture construction and single-row GatedMessage.forward remain unchanged.
This fixture freezes execution order; no ignored source archive is needed at runtime.
"""
import torch
from torch.nn import functional as F

from music_critic.critic.contracts import ASPECTS, FeatureSample, LocalPrediction, SamplePrediction
from music_critic.critic.geometry_plan import geometry_plan
from music_critic.critic.model import MultiscaleAspectCritic


class Source004Critic(MultiscaleAspectCritic):
    def _sample(self, sample: FeatureSample) -> SamplePrediction:
        w = self.width
        zero = self.note_mix.weight.new_zeros(w)
        empty = zero.new_empty((0, w))
        plan = geometry_plan(sample.geometry)
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
            return values[list(indices)] if indices else empty

        def pool_indices(values, indices):
            return self.pool(rows(values, indices))

        def sequence(gru, values):
            return gru(values.unsqueeze(0))[0].squeeze(0) if len(values) else empty

        def weighted(values, weights):
            if not weights.entries:
                return zero
            tensor = zero.new_tensor(weights.dense())
            return (tensor[:, None] * values).sum(0)

        local = get('note_local')
        note = torch.stack([F.gelu(self.note_mix(torch.cat((get('note')[i],
            local[i] if len(local) else zero, zero.new_tensor(n.timing)))))
            for i, n in enumerate(plan.notes)]) if notes else empty
        events = plan.events
        track = get('track')
        event = torch.stack([F.gelu(self.event_input(torch.cat((
            pool_indices(note, e.notes),
            track[e.track] if len(track) else zero, zero.new_tensor(e.timing)))))
            for e in events]) if events else empty
        if len(event):
            updated = event.new_zeros(event.shape)
            for t in plan.tracks:
                indexes = t.events
                if indexes:
                    index = torch.tensor(indexes, device=event.device)
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
                zero.new_tensor(s.timing))))))
        vertical = sequence(self.slice_gru, torch.stack(slice_rows) if slice_rows else empty)
        old_tracks = rows(get('track'), plan.active_tracks) if len(get('track')) else empty
        old_global = self.pool(torch.cat((old_tracks, get('song')), dim=0))
        bar_rows = []
        for i, b in enumerate(bars):
            bar_rows.append(F.gelu(self.bar_input(torch.cat((weighted(vertical, b.weights),
                pool_indices(event, b.events), get('bar')[i] if len(get('bar')) else zero,
                old_global, zero.new_tensor(b.timing))))))
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
        vertical = torch.stack([self.fusion(vertical[i], self.pool(torch.cat((
            rows(note, s.notes), rows(event, s.events)), dim=0)), 6)
            for i,s in enumerate(slices)]) if slices else empty
        bar = torch.stack([self.fusion(bar[i], self.pool(torch.stack((
            weighted(vertical, b.weights), pool_indices(event, b.events)))), 7)
            for i,b in enumerate(bars)]) if bars else empty
        beat = torch.stack([weighted(vertical, b) for b in plan.beat_weights]) if plan.beat_weights else empty
        new_track = torch.stack([pool_indices(note, plan.tracks[i].notes)
                                 for i in plan.active_tracks]) if notes else empty
        spans = make_spans(bar)
        song = self.pool(torch.cat((bar, new_track, note), dim=0)).unsqueeze(0) if notes else empty
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
