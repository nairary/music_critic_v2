"""Bounded, tensor-free plans for the critic's exact raw-geometry joins.

Cache keys snapshot content because SampleGeometry.intervals may be mutable.
All indices keep the original row order, including overlapping parent intervals.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType

from .contracts import LocalCandidate, SampleGeometry, track_events


@dataclass(frozen=True)
class WeightedOverlap:
    length: int
    entries: tuple[tuple[int, float], ...]

    def dense(self):
        # Keep the full tensor shape and zero positions used by the original sum.
        values = [0.] * self.length
        for index, weight in self.entries:
            values[index] = weight
        return values


@dataclass(frozen=True)
class NotePlan:
    event: int
    slices: tuple[int, ...]
    timing: tuple[float, ...]


@dataclass(frozen=True)
class EventPlan:
    notes: tuple[int, ...]
    track: int
    slices: tuple[int, ...]
    timing: tuple[float, ...]


@dataclass(frozen=True)
class SlicePlan:
    attacks: tuple[int, ...]
    holds: tuple[int, ...]
    releases: tuple[int, ...]
    beats: tuple[int, ...]
    onsets: tuple[int, ...]
    bars: tuple[int, ...]
    spans: tuple[tuple[int, ...], ...]
    notes: tuple[int, ...]
    events: tuple[int, ...]
    timing: tuple[float, ...]


@dataclass(frozen=True)
class BarPlan:
    events: tuple[int, ...]
    weights: WeightedOverlap
    timing: tuple[float, ...]


@dataclass(frozen=True)
class TrackPlan:
    notes: tuple[int, ...]
    events: tuple[int, ...]


@dataclass(frozen=True)
class GeometryPlan:
    geometry: SampleGeometry
    notes: tuple[NotePlan, ...]
    events: tuple[EventPlan, ...]
    slices: tuple[SlicePlan, ...]
    bars: tuple[BarPlan, ...]
    beat_weights: tuple[WeightedOverlap, ...]
    tracks: tuple[TrackPlan, ...]
    active_tracks: tuple[int, ...]
    span_groups: tuple[tuple[tuple[int, int], ...], ...]
    candidates: tuple[tuple[str, tuple[LocalCandidate, ...]], ...]
    local_masks: tuple[tuple[bool, ...], ...]
    availability: tuple[bool, ...]
    boundaries: tuple[int, ...]


def _at_points(ranges, points):
    """Half-open containment via exact sweep; return all parents in row order."""
    starts = sorted((a, i) for i, (a, b) in enumerate(ranges) if b > a)
    ends = sorted((b, i) for i, (a, b) in enumerate(ranges) if b > a)
    active, found = set(), {}
    a = b = 0
    for point in sorted(set(points)):
        while a < len(starts) and starts[a][0] <= point:
            active.add(starts[a][1])
            a += 1
        while b < len(ends) and ends[b][0] <= point:
            active.discard(ends[b][1])
            b += 1
        found[point] = tuple(sorted(active))
    return tuple(found[p] for p in points)


def _equal(keys, queries):
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    return tuple(tuple(groups.get(q, ())) for q in queries)


def _within(points, ranges):
    ordered = sorted((p, i) for i, p in enumerate(points))
    keys = [p for p, _ in ordered]
    return tuple(tuple(sorted(i for _, i in ordered[bisect_left(keys, a):bisect_left(keys, b)]))
                 for a, b in ranges)


def _overlapping(ranges, queries):
    """Strict endpoint test, also preserving interior grace-note candidates."""
    starts = sorted((a, i) for i, (a, _) in enumerate(ranges))
    ends = sorted((b, i) for i, (_, b) in enumerate(ranges))
    start_keys, end_keys = [a for a, _ in starts], [b for b, _ in ends]
    found = []
    for a, b in queries:
        left, right = bisect_left(start_keys, b), bisect_right(end_keys, a)
        if left <= len(ends) - right:
            indices = (i for _, i in starts[:left] if ranges[i][1] > a)
        else:
            indices = (i for _, i in ends[right:] if ranges[i][0] < b)
        found.append(tuple(sorted(indices)))
    return tuple(found)


def _weights(ranges, queries):
    result = []
    for (start, end), indices in zip(queries, _overlapping(ranges, queries)):
        overlaps = tuple((i, max(Fraction(0), min(ranges[i][1], end) - max(ranges[i][0], start)))
                         for i in indices)
        total = sum(value for _, value in overlaps)
        result.append(WeightedOverlap(len(ranges), tuple((i, float(value / total))
            for i, value in overlaps if value) if total else ()))
    return tuple(result)


def geometry_plan(geometry: SampleGeometry) -> GeometryPlan:
    key = (geometry.sample_id, tuple(geometry.notes), tuple(geometry.tracks),
           tuple((name, tuple(rows)) for name, rows in sorted(geometry.intervals.items())),
           geometry.duration)
    return _cached_plan(key)


@lru_cache(maxsize=128)
def _cached_plan(key):
    sample_id, notes, tracks, intervals, duration_exact = key
    geo = SampleGeometry(sample_id, notes, tracks, MappingProxyType(dict(intervals)), duration_exact)
    ids = {n.note_id: i for i, n in enumerate(notes)}
    track_ids = {t: i for i, t in enumerate(tracks)}
    if len(ids) != len(notes) or len(track_ids) != len(tracks):
        raise ValueError('duplicate raw identities')
    if any(n.track_id not in track_ids for n in notes):
        raise ValueError('note has unknown raw track')
    slices, beats = geo.intervals.get('slice', ()), geo.intervals.get('beat', ())
    bars, onsets = geo.intervals.get('bar', ()), geo.intervals.get('onset', ())
    events = track_events(geo)
    duration = max(float(geo.duration), 1e-12)

    def timing(start, end, extra=0., flag=0.):
        return (float(start) / duration,
                float(end - start) / (1 + abs(float(end - start))), extra, flag)

    note_ranges = tuple((n.start, n.end) for n in notes)
    slice_ranges = tuple((s.start, s.end) for s in slices)
    beat_ranges = tuple((b.start, b.end) for b in beats)
    bar_ranges = tuple((b.start, b.end) for b in bars)
    slice_points = tuple(s.start for s in slices)
    event_points = tuple(e.start for e in events)
    sounding = _at_points(note_ranges, slice_points)
    attacks = _equal(tuple(n.start for n in notes), slice_points)
    release_map = {}
    for i, n in enumerate(notes):
        if n.end > n.start:
            release_map.setdefault(n.end, []).append(i)
    slice_notes = tuple(tuple(sorted(set(active).union(on))) for active, on in zip(sounding, attacks))
    note_slices = [[] for _ in notes]
    for j, indices in enumerate(slice_notes):
        for i in indices:
            note_slices[i].append(j)
    event_slices = _equal(slice_points, event_points)
    slice_events = _equal(event_points, slice_points)
    slice_beats = _at_points(beat_ranges, slice_points)
    slice_bars = _at_points(bar_ranges, slice_points)
    slice_onsets = _equal(tuple(o.start for o in onsets), slice_points)
    bar_events = _within(event_points, bar_ranges)
    bar_weights, beat_weights = _weights(slice_ranges, bar_ranges), _weights(slice_ranges, beat_ranges)
    groups = tuple(tuple((i, i + size) for i in range(len(bars) - size + 1)
        if all(b.complete for b in bars[i:i+size])
        and all(bars[j].end == bars[j+1].start for j in range(i, i+size-1))) for size in (2, 4, 8))
    span_parents = tuple(_at_points(tuple((bars[a].start, bars[b-1].end) for a, b in group),
                                  slice_points) for group in groups)
    event_tracks = tuple(notes[ids[e.note_ids[0]]].track_id for e in events)
    note_event = {k: i for i, e in enumerate(events) for k in e.note_ids}
    track_notes = _equal(tuple(n.track_id for n in notes), tracks)
    track_events_ = _equal(event_tracks, tracks)
    boundaries = tuple(i for i in range(len(bars)-1) if bars[i].end == bars[i+1].start)
    candidates = (
        ('harmony/slice', tuple(LocalCandidate(s.entity_id, s.start, s.end,
            tuple(notes[j].note_id for j in active)) for s, active in zip(slices, sounding))),
        ('melody/note', tuple(LocalCandidate(n.note_id, n.start, n.end, (n.note_id,)) for n in notes)),
        ('melody/event', events), ('rhythm/event', events),
        ('rhythm/beat', tuple(LocalCandidate(b.entity_id, b.start, b.end,
            tuple(notes[j].note_id for j in active))
            for b, active in zip(beats, _overlapping(note_ranges, beat_ranges)))),
        ('structure/boundary', tuple(LocalCandidate('boundary:' + bars[i+1].entity_id,
            bars[i+1].start, bars[i+1].start) for i in boundaries)),
    )
    return GeometryPlan(
        geo,
        tuple(NotePlan(note_event[n.note_id], tuple(note_slices[i]),
            timing(n.start, n.end, n.pitch / 127., float(n.is_drum))) for i, n in enumerate(notes)),
        tuple(EventPlan(tuple(ids[k] for k in e.note_ids), track_ids[t], event_slices[i],
            timing(e.start, e.end)) for i, (e, t) in enumerate(zip(events, event_tracks))),
        tuple(SlicePlan(attacks[i], tuple(j for j in sounding[i] if notes[j].start < s.start),
            tuple(release_map.get(s.start, ())), slice_beats[i], slice_onsets[i], slice_bars[i],
            tuple(parents[i] for parents in span_parents), slice_notes[i], slice_events[i],
            timing(s.start, s.end)) for i, s in enumerate(slices)),
        tuple(BarPlan(bar_events[i], bar_weights[i], timing(b.start, b.end)) for i, b in enumerate(bars)),
        beat_weights, tuple(TrackPlan(ns, es) for ns, es in zip(track_notes, track_events_)),
        tuple(i for i, ns in enumerate(track_notes) if ns), groups, candidates,
        tuple(tuple(any(not notes[ids[k]].is_drum for k in item.note_ids)
            if key.split('/')[0] in ('melody', 'harmony') else bool(notes) for item in items)
            for key, items in candidates),
        (any(not n.is_drum and n.end > n.start for n in notes), bool(notes) and bool(beats),
         any(not n.is_drum for n in notes), bool(notes) and bool(boundaries)), boundaries,
    )
