"""Versioned, exact, target-independent HCM-1 public value contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

from music_critic.data.timing import RationalTime

SCHEMA = 'HierarchicalConditionalMusicModel@1.0.0'


def _integer(name: str, value: int, low: int, high: int | None = None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f'pll.invalid_{name}')


@dataclass(frozen=True, slots=True)
class Event:
    """One projected event; onset is relative to its MaskQuery start.

    Kind: 0 pitched, 1 pitched grace, 2 percussion, 3 percussion grace.
    Pitch is MIDI note/key number in all cases, never program ID.
    """
    onset: RationalTime
    track: int
    kind: int
    pitch: int
    duration: RationalTime

    def __post_init__(self):
        if not isinstance(self.onset, RationalTime) or not isinstance(self.duration, RationalTime):
            raise TypeError('pll.exact_event_time_required')
        _integer('track', self.track, 0, 15)
        _integer('kind', self.kind, 0, 3)
        _integer('pitch', self.pitch, 0, 127)
        if self.onset.num < 0 or self.duration.num < 0 or (self.duration.num == 0 and self.kind % 2 == 0):
            raise ValueError('pll.invalid_event_time')

    @property
    def key(self):
        return self.onset, self.track, self.kind, self.pitch, self.duration


@dataclass(frozen=True, slots=True)
class MaskQuery:
    start: RationalTime
    end: RationalTime
    track_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.start, RationalTime) or not isinstance(self.end, RationalTime):
            raise TypeError('pll.exact_query_time_required')
        if self.start.num < 0 or self.end <= self.start:
            raise ValueError('pll.invalid_query_interval')
        if (not isinstance(self.track_ids, tuple) or not self.track_ids
                or any(not isinstance(t, str) or not t for t in self.track_ids)
                or len(set(self.track_ids)) != len(self.track_ids)):
            raise ValueError('pll.invalid_query_tracks')

    @property
    def span(self):
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class CodecProfile:
    max_events: int = 512
    max_tracks: int = 16
    max_denominator: int = 65535
    max_numerator: int = 2**64 - 1

    def __post_init__(self):
        _integer('max_events', self.max_events, 1)
        _integer('max_tracks', self.max_tracks, 1, 16)
        _integer('max_denominator', self.max_denominator, 1, 65535)
        _integer('max_numerator', self.max_numerator, 1, 2**64 - 1)


@dataclass(frozen=True, slots=True)
class ContextProfile:
    context_bars: int = 32
    core_bars: int = 4
    halo_bars: int = 1
    max_tracks: int = 16
    max_notes_per_tile: int = 2048
    max_edges_per_tile: int = 200000
    max_slices_per_tile: int = 8192
    max_memory: int = 1152
    max_fine_memory: int = 512

    def __post_init__(self):
        for name in ('context_bars', 'core_bars', 'max_tracks', 'max_notes_per_tile',
                     'max_edges_per_tile', 'max_slices_per_tile', 'max_memory'):
            _integer(name, getattr(self, name), 1)
        _integer('halo_bars', self.halo_bars, 0)
        _integer('max_fine_memory', self.max_fine_memory, 0)
        if self.core_bars > self.context_bars or self.max_tracks > 16:
            raise ValueError('pll.invalid_context_profile')


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 256
    global_layers: int = 2
    event_layers: int = 6
    heads: int = 8
    ff_dim: int = 1024
    field_width: int = 128
    field_heads: int = 4
    field_ff_dim: int = 512
    dropout: float = .1
    relation_dim: int = 64
    relation_components: int = 4
    max_fields: int = 64
    relations: str = 'transr'

    def __post_init__(self):
        for name in ('width', 'global_layers', 'event_layers', 'heads', 'ff_dim',
                     'field_width', 'field_heads', 'field_ff_dim', 'relation_dim',
                     'relation_components', 'max_fields'):
            _integer(name, getattr(self, name), 1)
        if self.width % self.heads or self.field_width % self.field_heads:
            raise ValueError('pll.invalid_attention_width')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('pll.invalid_dropout')
        if self.relations not in ('none', 'transr', 'bilinear'):
            raise ValueError('pll.invalid_relation_mode')

    def fingerprint(self):
        return sha256(json.dumps({'schema': SCHEMA, **asdict(self)}, sort_keys=True,
                                 separators=(',', ':')).encode()).hexdigest()

    def validate_codec(self, profile: CodecProfile) -> None:
        # Five pitched-event fields plus two length-prefixed rational times.
        # A conservative bound covers every supported value, not just a target
        # encountered in training; a legal NOTE prefix must remain completable.
        denominator_bytes = (profile.max_denominator.bit_length() + 7) // 8
        numerator_bytes = (profile.max_numerator.bit_length() + 7) // 8
        required = 5 + 2 * (2 + denominator_bytes + numerator_bytes)
        if self.max_fields < required:
            raise ValueError(f'pll.field_capacity_must_cover_codec:{required}')
