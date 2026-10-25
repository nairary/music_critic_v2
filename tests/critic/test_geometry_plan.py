"""Exact geometry optimization mechanics; synthetic inputs only."""
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction as F
import random
from types import MappingProxyType
from unittest.mock import patch

import pytest
import torch

from music_critic.critic.contracts import (
    FeatureSample, RawInterval, RawNote, SampleGeometry, local_candidates, track_events,
)
from music_critic.critic.geometry_plan import (
    BarPlan, EventPlan, GeometryPlan, NotePlan, SlicePlan, TrackPlan, WeightedOverlap,
    _cached_plan, geometry_plan,
)
from music_critic.critic.model import MultiscaleAspectCritic
from .test_model import fixture_sample, pyramid


def naive_plan(geo):
    """Independent original scan formulas, including their row/reduction order."""
    notes, bars = geo.notes, geo.intervals.get('bar', ())
    slices, beats = geo.intervals.get('slice', ()), geo.intervals.get('beat', ())
    onsets, events = geo.intervals.get('onset', ()), track_events(geo)
    ids = {n.note_id: i for i, n in enumerate(notes)}
    duration = max(float(geo.duration), 1e-12)

    def timing(start, end, extra=0., flag=0.):
        return (float(start) / duration,
                float(end - start) / (1 + abs(float(end - start))), extra, flag)

    def weights(start, end):
        values = [max(F(0), min(s.end, end) - max(s.start, start)) for s in slices]
        total = sum(values)
        return WeightedOverlap(len(slices), tuple((i, float(v / total))
            for i, v in enumerate(values) if v)) if total else WeightedOverlap(len(slices), ())

    event_tracks = tuple(notes[ids[e.note_ids[0]]].track_id for e in events)
    note_event = {k: i for i, e in enumerate(events) for k in e.note_ids}
    groups = tuple(tuple((i, i + size) for i in range(len(bars) - size + 1)
        if all(b.complete for b in bars[i:i+size])
        and all(bars[j].end == bars[j+1].start for j in range(i, i+size-1))) for size in (2, 4, 8))
    ranges = tuple(tuple((bars[a].start, bars[b-1].end) for a, b in group) for group in groups)
    candidates = tuple(local_candidates(geo).items())
    boundaries = tuple(i for i in range(len(bars)-1) if bars[i].end == bars[i+1].start)
    return GeometryPlan(
        replace(geo, intervals=MappingProxyType(dict(geo.intervals))),
        tuple(NotePlan(note_event[n.note_id], tuple(j for j, s in enumerate(slices)
            if n.start <= s.start < n.end or s.start == n.start),
            timing(n.start, n.end, n.pitch / 127., float(n.is_drum))) for n in notes),
        tuple(EventPlan(tuple(ids[k] for k in e.note_ids), geo.tracks.index(t),
            tuple(j for j, s in enumerate(slices) if s.start == e.start), timing(e.start, e.end))
            for e, t in zip(events, event_tracks)),
        tuple(SlicePlan(
            tuple(j for j, n in enumerate(notes) if n.start == s.start),
            tuple(j for j, n in enumerate(notes) if n.start < s.start < n.end),
            tuple(j for j, n in enumerate(notes) if n.end == s.start and n.end > n.start),
            tuple(j for j, b in enumerate(beats) if b.start <= s.start < b.end),
            tuple(j for j, o in enumerate(onsets) if o.start == s.start),
            tuple(j for j, b in enumerate(bars) if b.start <= s.start < b.end),
            tuple(tuple(j for j, (a, b) in enumerate(rs) if a <= s.start < b) for rs in ranges),
            tuple(j for j, n in enumerate(notes) if n.start <= s.start < n.end or n.start == s.start),
            tuple(j for j, e in enumerate(events) if e.start == s.start), timing(s.start, s.end))
            for s in slices),
        tuple(BarPlan(tuple(j for j, e in enumerate(events) if b.start <= e.start < b.end),
            weights(b.start, b.end), timing(b.start, b.end)) for b in bars),
        tuple(weights(b.start, b.end) for b in beats),
        tuple(TrackPlan(tuple(i for i, n in enumerate(notes) if n.track_id == t),
            tuple(i for i, owner in enumerate(event_tracks) if owner == t)) for t in geo.tracks),
        tuple(i for i, t in enumerate(geo.tracks) if any(n.track_id == t for n in notes)),
        groups, candidates,
        tuple(tuple(any(not notes[ids[k]].is_drum for k in item.note_ids)
            if key.split('/')[0] in ('melody', 'harmony') else bool(notes) for item in items)
            for key, items in candidates),
        (any(not n.is_drum and n.end > n.start for n in notes), bool(notes) and bool(beats),
         any(not n.is_drum for n in notes), bool(notes) and bool(boundaries)), boundaries,
    )


def rich_sample():
    """Fractional, unordered, overlapping, grace and incomplete geometry."""
    notes = (
        RawNote('z', 'second', F(1, 3), F(19, 3), 64),
        RawNote('a', 'second', F(1, 3), F(2, 3), 72),
        RawNote('grace', 'first', F(4, 3), F(4, 3), 80),
        RawNote('drum', 'first', F(0), F(1, 3), 36, True),
        RawNote('later', 'first', F(11, 3), F(13, 3), 67),
    )
    slices = tuple(RawInterval(f's{i}', F(i, 3), F(i+1, 3)) for i in (4, 0, 1, 2, 3, *range(5, 25)))
    intervals = {
        'slice': slices + (RawInterval('empty', F(4, 3), F(4, 3)),),
        'beat': (RawInterval('overlap', F(1, 6), F(7, 3)),
                 RawInterval('grace-inside', F(7, 6), F(3, 2)),
                 RawInterval('zero', F(4, 3), F(4, 3)))
                + tuple(RawInterval(f'b{i}', F(i), F(i+1)) for i in range(9)),
        'onset': tuple(RawInterval(f'o{i}', s, s) for i, s in enumerate((F(4, 3), F(1, 3), F(0), F(4, 3)))),
        'bar': tuple(RawInterval(f'bar{i}', F(i), F(i+1)) for i in range(9))
               + (RawInterval('overlap-bar', F(1, 6), F(7, 3), False),),
    }
    geo = SampleGeometry('rich', notes, ('second', 'empty', 'first'), intervals, F(9))
    gen = torch.Generator().manual_seed(41)
    sizes = dict(note=len(notes), note_local=len(notes), track=3, song=1,
                 **{k: len(v) for k, v in intervals.items()})
    values = {k: torch.randn(n, 8, generator=gen, requires_grad=True) for k, n in sizes.items()}
    return FeatureSample(geo, values, 'synthetic', {})


@pytest.mark.parametrize('sample', [rich_sample(), fixture_sample(), fixture_sample(empty=True),
    FeatureSample(SampleGeometry('empty', (), (), {}, F(0)), {}, 'synthetic', {})])
def test_plan_matches_independent_scan_formulas(sample):
    actual, expected = geometry_plan(sample.geometry), naive_plan(sample.geometry)
    assert actual == expected
    for weighted in (*actual.beat_weights, *(b.weights for b in actual.bars)):
        dense = weighted.dense()
        assert len(dense) == len(sample.geometry.intervals.get('slice', ()))
        assert dense == [dict(weighted.entries).get(i, 0.) for i in range(weighted.length)]


def test_exact_joins_match_scans_for_random_unsorted_and_overlapping_intervals():
    rng = random.Random(191)
    for case in range(30):
        def interval():
            start = F(rng.randrange(-9, 24), rng.choice((2, 3, 7)))
            return start, start + F(rng.randrange(8), rng.choice((2, 3, 7)))

        notes = tuple(RawNote(f'n{i}', rng.choice(('a', 'b')), *interval(),
                              rng.randrange(128), bool(rng.randrange(2))) for i in range(rng.randrange(25)))
        intervals = {key: tuple(RawInterval(f'{key}{i}', *interval(), bool(rng.randrange(2)))
                               for i in range(rng.randrange(25))) for key in ('slice', 'beat', 'bar', 'onset')}
        geo = SampleGeometry(f'random-{case}', notes, ('b', 'unused', 'a'), intervals, F(10))
        assert geometry_plan(geo) == naive_plan(geo)


def assert_prediction_equal(actual, expected):
    assert actual.sample_id == expected.sample_id and actual.horizon == expected.horizon
    for a, b in ((actual.raw_scores, expected.raw_scores), (actual.availability, expected.availability)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual.contextual.keys() == expected.contextual.keys()
    for key in actual.contextual:
        torch.testing.assert_close(actual.contextual[key], expected.contextual[key], rtol=0, atol=0)
    assert actual.local.keys() == expected.local.keys()
    for key, a in actual.local.items():
        b = expected.local[key]
        assert a.candidates == b.candidates
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        torch.testing.assert_close(a.availability, b.availability, rtol=0, atol=0)


def test_outputs_gradients_and_optimizer_step_match_naive_geometry():
    torch.manual_seed(90)
    actual = MultiscaleAspectCritic(feature_dim=8, width=8)
    expected = MultiscaleAspectCritic(feature_dim=8, width=8)
    expected.load_state_dict(actual.state_dict())
    samples = (rich_sample(), fixture_sample(empty=True))
    outputs = []
    for model, planner in ((actual, geometry_plan), (expected, naive_plan)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with patch('music_critic.critic.model.geometry_plan', planner):
            result = model(pyramid(*samples))
        loss = sum(s.raw_scores.square().sum() + sum(p.logits.square().sum() for p in s.local.values())
                   for s in result.samples)
        loss.backward()
        grads = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
        optimizer.step()
        outputs.append((result, grads, optimizer.state_dict()))
    for a, b in zip(outputs[0][0].samples, outputs[1][0].samples):
        assert_prediction_equal(a, b)
    for key, a in outputs[0][1].items():
        b = outputs[1][1][key]
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    for key, state in outputs[0][2]['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, outputs[1][2]['state'][key][name], rtol=0, atol=0)
    assert all(v.grad is None for s in samples for v in s.embeddings.values())


def test_content_key_prevents_stale_mutable_interval_cache():
    geo = rich_sample().geometry
    before = geometry_plan(geo)
    assert geometry_plan(replace(geo, intervals=dict(geo.intervals))) is before
    geo.intervals['bar'] = tuple(replace(b, complete=False) for b in geo.intervals['bar'])
    after = geometry_plan(geo)
    assert after is not before and not any(after.span_groups)
    assert any(before.span_groups)
    with pytest.raises(TypeError):
        before.geometry.intervals['bar'] = ()
    for change in (
        dict(duration=F(10)), dict(tracks=tuple(reversed(geo.tracks))),
        dict(notes=tuple(reversed(geo.notes))),
        dict(notes=(replace(geo.notes[0], pitch=65), *geo.notes[1:])),
        dict(notes=(replace(geo.notes[0], is_drum=True), *geo.notes[1:])),
        dict(notes=(replace(geo.notes[0], note_id='renamed'), *geo.notes[1:])),
    ):
        assert geometry_plan(replace(geo, **change)) is not after


def test_cache_is_bounded_and_contains_no_tensors():
    _cached_plan.cache_clear()
    geo = SampleGeometry('0', (), (), {}, F(0))
    first = geometry_plan(geo)
    for i in range(1, 129):
        geometry_plan(replace(geo, sample_id=str(i)))
    assert _cached_plan.cache_info().currsize == 128
    assert geometry_plan(geo) is not first

    def check(value):
        assert not isinstance(value, torch.Tensor)
        if is_dataclass(value):
            for field in fields(value):
                check(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        elif isinstance(value, MappingProxyType):
            for item in value.values():
                check(item)
    check(geometry_plan(rich_sample().geometry))


@pytest.mark.parametrize('change,match', [
    (lambda g: replace(g, notes=(g.notes[0], g.notes[0])), 'duplicate'),
    (lambda g: replace(g, tracks=('second', 'second')), 'duplicate'),
    (lambda g: replace(g, tracks=('second',)), 'unknown'),
])
def test_identity_validation_before_plan(change, match):
    with pytest.raises(ValueError, match=match):
        geometry_plan(change(rich_sample().geometry))
