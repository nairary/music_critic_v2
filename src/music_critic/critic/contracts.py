"""Raw-only interfaces for the bounded multi-aspect critic.

Identity and exact time are alignment metadata, never learned categorical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import json
from typing import Mapping

import torch
from torch import Tensor

ASPECTS = ('harmony', 'rhythm', 'melody', 'structure')
PYRAMID_SCHEMA = 'CriticFeaturePyramid@1.0.0'
OUTPUT_SCHEMA = 'MultiscaleAspectCritic@1.0.0'
SUPPORT_POLICY = 'raw-clip-local-order-v1'


@dataclass(frozen=True)
class FeatureBinding:
    source_id: str
    checkpoint_id: str
    adapter_version: str
    dtype: str = 'float32'
    schema: str = PYRAMID_SCHEMA
    model_state_id: str = ''

    def __post_init__(self):
        if self.schema != PYRAMID_SCHEMA:
            raise ValueError('feature schema mismatch')
        for value in (self.source_id, self.checkpoint_id, self.adapter_version, self.dtype):
            if (not isinstance(value, str) or not value.strip()
                    or value.strip().lower() in {'unknown', 'none', 'null'}):
                raise ValueError('known feature binding required')
        if not isinstance(self.model_state_id, str):
            raise ValueError('model state identity must be a string')


@dataclass(frozen=True)
class RawNote:
    note_id: str
    track_id: str
    start: Fraction
    end: Fraction
    pitch: int
    is_drum: bool = False

    def __post_init__(self):
        if (not self.note_id or not self.track_id or type(self.is_drum) is not bool
                or type(self.pitch) is not int):
            raise ValueError('invalid raw note identity or type')
        if not isinstance(self.start, Fraction) or not isinstance(self.end, Fraction):
            raise ValueError('exact Fraction times required')
        if self.end < self.start or not 0 <= self.pitch <= 127:
            raise ValueError('invalid raw note')


@dataclass(frozen=True)
class RawInterval:
    entity_id: str
    start: Fraction
    end: Fraction
    complete: bool = True

    def __post_init__(self):
        if not self.entity_id or type(self.complete) is not bool:
            raise ValueError('invalid raw interval identity or completeness')
        if not isinstance(self.start, Fraction) or not isinstance(self.end, Fraction):
            raise ValueError('exact Fraction times required')
        if self.end < self.start:
            raise ValueError('negative raw interval')


@dataclass(frozen=True)
class SampleGeometry:
    sample_id: str
    notes: tuple[RawNote, ...]
    tracks: tuple[str, ...]
    intervals: Mapping[str, tuple[RawInterval, ...]]
    duration: Fraction


@dataclass(frozen=True)
class FeatureSample:
    geometry: SampleGeometry
    embeddings: Mapping[str, Tensor]
    raw_hash: str
    stages: Mapping[str, str]

    @property
    def sample_id(self):
        return self.geometry.sample_id

    def to(self, device):
        return replace(self, embeddings={k: v.to(device) for k, v in self.embeddings.items()})


@dataclass(frozen=True)
class CriticFeaturePyramid:
    samples: tuple[FeatureSample, ...]
    binding: FeatureBinding

    def to(self, device):
        return replace(self, samples=tuple(s.to(device) for s in self.samples))


@dataclass(frozen=True)
class LocalCandidate:
    candidate_id: str
    start: Fraction
    end: Fraction
    note_ids: tuple[str, ...] = ()
    complete: bool = True


def track_events(geometry: SampleGeometry) -> tuple[LocalCandidate, ...]:
    groups: dict[tuple[str, Fraction], list[RawNote]] = {}
    for note in geometry.notes:
        groups.setdefault((note.track_id, note.start), []).append(note)
    return tuple(LocalCandidate(
        json.dumps([track, start.numerator, start.denominator], separators=(',', ':')),
        start, max(n.end for n in notes), tuple(sorted(n.note_id for n in notes)),
    ) for (track, start), notes in sorted(groups.items()))


def local_candidates(geometry: SampleGeometry) -> Mapping[str, tuple[LocalCandidate, ...]]:
    """Same raw candidate namespaces for predictor and target-side alignment."""
    notes = tuple(LocalCandidate(n.note_id, n.start, n.end, (n.note_id,)) for n in geometry.notes)
    events = track_events(geometry)
    slices = tuple(LocalCandidate(i.entity_id, i.start, i.end,
        tuple(n.note_id for n in geometry.notes if n.start <= i.start < n.end))
        for i in geometry.intervals.get('slice', ()))
    beats = tuple(LocalCandidate(i.entity_id, i.start, i.end,
        tuple(n.note_id for n in geometry.notes if n.start < i.end and n.end > i.start))
        for i in geometry.intervals.get('beat', ()))
    bars = geometry.intervals.get('bar', ())
    boundaries = tuple(LocalCandidate('boundary:' + b.entity_id, b.start, b.start)
                       for a, b in zip(bars, bars[1:]) if a.end == b.start)
    return {'harmony/slice': slices, 'melody/note': notes, 'melody/event': events,
            'rhythm/event': events, 'rhythm/beat': beats, 'structure/boundary': boundaries}


@dataclass(frozen=True)
class LocalPrediction:
    aspect: str
    candidate_type: str
    candidates: tuple[LocalCandidate, ...]
    logits: Tensor
    availability: Tensor


@dataclass(frozen=True)
class SamplePrediction:
    sample_id: str
    raw_scores: Tensor
    availability: Tensor
    local: Mapping[str, LocalPrediction]
    horizon: Fraction
    support_policy: str = SUPPORT_POLICY
    contextual: Mapping[str, Tensor] = field(default_factory=dict)
    musical_error_location: None = None
    musical_error_reason: str = 'no_validated_targets'


@dataclass(frozen=True)
class CriticOutput:
    samples: tuple[SamplePrediction, ...]
    schema: str = OUTPUT_SCHEMA
