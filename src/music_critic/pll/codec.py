"""Exact event serialization with a prefix-valid finite grammar.

The grammar depends only on the public query and the consumed prefix. Time
uses reduced, denominator-first rational integers; no corpus grid is inferred.
The cached denominator scan is an exact reference implementation, not a claim
of production throughput for the 65535-denominator profile.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from music_critic.data.timing import RationalTime
from music_critic.pll.contracts import CodecProfile, Event

NOTE, EOS = 0, 1
LEN_BASE, BYTE_BASE = 2, 10
TRACK_BASE, KIND_BASE = 266, 282
PC_BASE, OCTAVE_BASE, DRUM_BASE = 286, 298, 309
BOS, PAD, VOCAB_SIZE = 437, 438, 439
FIELD_NAMES = (
    "event", "onset_den_length", "onset_den_byte", "onset_num_length",
    "onset_num_byte", "track", "kind", "pitch_class", "octave", "drum",
    "duration_den_length", "duration_den_byte", "duration_num_length",
    "duration_num_byte",
)


@dataclass(frozen=True, slots=True)
class GrammarStep:
    """State immediately before a token, including the exact legal support."""

    field: str
    allowed: tuple[int, ...]
    event_index: int
    field_index: int


@dataclass(frozen=True, slots=True)
class CodecState:
    events: tuple[Event, ...]
    step: GrammarStep | None
    complete: bool
    current_tokens: tuple[int, ...]


def _key(event: Event) -> tuple:
    return event.onset, event.track, event.kind, event.pitch, event.duration


def _integer(value: int) -> tuple[int, ...]:
    length = max(1, (value.bit_length() + 7) // 8)
    return (LEN_BASE + length - 1,) + tuple(
        BYTE_BASE + byte for byte in value.to_bytes(length, "big")
    )


def _rational(value: RationalTime) -> tuple[int, ...]:
    return _integer(value.den) + _integer(value.num)


@lru_cache(maxsize=65535)
def _mobius_terms(denominator: int) -> tuple[tuple[int, int], ...]:
    """Square-free products with signs, for exact coprime interval counts."""
    factors = []
    remaining, divisor = denominator, 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    terms = [(1, 1)]
    for factor in factors:
        terms += [(product * factor, -sign) for product, sign in tuple(terms)]
    return tuple(terms)


def _has_coprime(low: int, high: int, denominator: int) -> bool:
    if low > high:
        return False
    if high - low + 1 >= denominator:
        return True  # Every d consecutive integers contain a residue coprime to d.
    # Counts zero correctly: gcd(0, d) == 1 only for d == 1.
    return sum(
        sign * (high // divisor - (low - 1) // divisor)
        for divisor, sign in _mobius_terms(denominator)
    ) > 0


def _numerator_bounds(
    denominator: int, lower: RationalTime, upper: RationalTime | None,
    positive: bool, maximum: int,
) -> tuple[int, int]:
    low = (lower.num * denominator + lower.den - 1) // lower.den
    low = max(low, int(positive))
    high = maximum if upper is None else min(
        maximum, (upper.num * denominator - 1) // upper.den
    )
    return low, high


@lru_cache(maxsize=64)
def _feasible_denominators(
    maximum_denominator: int, maximum_numerator: int,
    lower: RationalTime, upper: RationalTime | None, positive: bool,
) -> array:
    # A packed array bounds this cache to at most roughly 8 MiB, unlike a
    # tuple of 65535 Python integers per distinct query bound.
    if lower.num == 0:
        # For d > 1 the smallest nonnegative reduced numerator is 1. For
        # d == 1, zero is also available unless positivity is required.
        first = 1 if upper is None else upper.den // upper.num + 1
        result = array("H", range(max(1, first), maximum_denominator + 1))
        if not positive and first > 1:
            result.insert(0, 1)
        return result
    result = array("H")
    for denominator in range(1, maximum_denominator + 1):
        low, high = _numerator_bounds(
            denominator, lower, upper, positive, maximum_numerator
        )
        if _has_coprime(low, high, denominator):
            result.append(denominator)
    return result


def _byte_interval(length: int, prefix: tuple[int, ...]) -> tuple[int, int]:
    remaining = length - len(prefix)
    value = int.from_bytes(bytes(prefix), "big") if prefix else 0
    low = value * (256 ** remaining)
    high = (value + 1) * (256 ** remaining) - 1
    return max(low, 0 if length == 1 else 256 ** (length - 1)), high


class EventCodec:
    """Canonical multiset codec; unsupported values raise instead of rounding."""

    def __init__(self, profile: CodecProfile | None = None):
        self.profile = profile or CodecProfile()
        for name, minimum, maximum in (
            ("max_events", 1, self.profile.max_events), ("max_tracks", 1, 16),
            ("max_denominator", 1, 65535),
            ("max_numerator", 1, 2**64 - 1),
        ):
            value = getattr(self.profile, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} outside codec support")

    def _query(self, span: RationalTime, tracks: tuple[int, ...]) -> tuple[int, ...]:
        if not isinstance(span, RationalTime) or span <= RationalTime(0):
            raise ValueError("span must be a positive RationalTime")
        if not isinstance(tracks, tuple) or not tracks:
            raise ValueError("tracks must be a nonempty tuple")
        if any(isinstance(t, bool) or not isinstance(t, int) or
               not 0 <= t < self.profile.max_tracks for t in tracks):
            raise ValueError("track ID outside codec profile")
        if len(set(tracks)) != len(tracks):
            raise ValueError("duplicate query track IDs")
        return tuple(sorted(tracks))

    def encode(
        self, events: Iterable[Event], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> tuple[int, ...]:
        tracks = self._query(span, tracks)
        events = tuple(events)
        if len(events) > self.profile.max_events:
            raise ValueError("unsupported event count; targets are never truncated")
        for event in events:
            self._validate_event(event, span, tracks)
        result: list[int] = []
        for event in sorted(events, key=_key):
            result.extend((NOTE, *_rational(event.onset), TRACK_BASE + event.track,
                           KIND_BASE + event.kind))
            if event.kind < 2:
                result.extend((PC_BASE + event.pitch % 12,
                               OCTAVE_BASE + event.pitch // 12))
            else:
                result.append(DRUM_BASE + event.pitch)
            result.extend(_rational(event.duration))
        return (*result, EOS)

    def _validate_event(self, event: Event, span: RationalTime, tracks: tuple[int, ...]):
        if not isinstance(event, Event):
            raise TypeError("events must contain Event values")
        for value in (event.onset, event.duration):
            if not isinstance(value, RationalTime) or not (
                0 <= value.num <= self.profile.max_numerator and
                value.den <= self.profile.max_denominator
            ):
                raise ValueError("unsupported exact rational value")
        if not RationalTime(0) <= event.onset < span:
            raise ValueError("event onset outside half-open query span")
        for name, value, upper in (("track", event.track, self.profile.max_tracks),
                                   ("kind", event.kind, 4), ("pitch", event.pitch, 128)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
                raise ValueError(f"unsupported {name}")
        if event.track not in tracks:
            raise ValueError("event track is not queried")
        if event.duration.num == 0 and event.kind not in (1, 3):
            raise ValueError("zero duration is only valid for grace events")

    def _parser(self, span: RationalTime, tracks: tuple[int, ...]) -> _Parser:
        return _Parser(self, span, self._query(span, tracks))

    def parse(
        self, prefix: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> CodecState:
        parser = self._parser(span, tracks)
        for token in prefix:
            parser.feed(token)
        return CodecState(tuple(parser.events), parser.step(), parser.complete,
                          tuple(parser.current_tokens))

    def allowed(
        self, prefix: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> tuple[int, ...]:
        state = self.parse(prefix, span=span, tracks=tracks)
        return () if state.step is None else state.step.allowed

    def decode(
        self, tokens: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> tuple[Event, ...]:
        state = self.parse(tokens, span=span, tracks=tracks)
        if not state.complete:
            raise ValueError("incomplete event stream: terminal EOS required")
        return state.events

    def trace(
        self, tokens: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...], require_complete: bool = False,
    ) -> tuple[GrammarStep, ...]:
        parser = self._parser(span, tracks)
        steps = []
        for token in tokens:
            step = parser.step()
            if step is None:
                raise ValueError("token after terminal EOS")
            steps.append(step)
            parser.feed(token, step.allowed)
        if require_complete and not parser.complete:
            raise ValueError("incomplete event stream: terminal EOS required")
        return tuple(steps)

    def trace_prefix(
        self, prefix: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> tuple[GrammarStep, ...]:
        """Trace a possibly unfinished prefix without requiring future fields."""
        return self.trace(prefix, span=span, tracks=tracks, require_complete=False)

    def event_token_groups(
        self, tokens: Iterable[int], *, span: RationalTime,
        tracks: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return complete event groups followed by a singleton EOS group."""
        tokens = tuple(tokens)
        trace = self.trace(tokens, span=span, tracks=tracks, require_complete=True)
        groups: list[list[int]] = []
        for token, step in zip(tokens, trace, strict=True):
            if step.field_index == 0:
                groups.append([])
            groups[-1].append(token)
        return tuple(tuple(group) for group in groups)


class _Parser:
    def __init__(self, codec: EventCodec, span: RationalTime, tracks: tuple[int, ...]):
        self.codec, self.span, self.tracks = codec, span, tracks
        self.events: list[Event] = []
        self.complete = False
        self.field = "event"
        self.current_tokens: list[int] = []
        self.current: dict = {}
        self.length = 0
        self.byte_prefix: tuple[int, ...] = ()
        self.denominator = 0

    def _same(self, *names: str) -> bool:
        return bool(self.events) and all(
            self.current.get(name) == getattr(self.events[-1], name) for name in names
        )

    def _bounds(self) -> tuple[RationalTime, RationalTime | None, bool]:
        if self.field.startswith("onset"):
            return (self.events[-1].onset if self.events else RationalTime(0),
                    self.span, False)
        lower = self.events[-1].duration if self._same("onset", "track", "kind", "pitch") else RationalTime(0)
        return lower, None, self.current["kind"] not in (1, 3)

    def _integer_possible(self, length: int, prefix: tuple[int, ...]) -> bool:
        low, high = _byte_interval(length, prefix)
        if low > high:
            return False
        lower, upper, positive = self._bounds()
        if "_den_" in self.field:
            feasible = _feasible_denominators(
                self.codec.profile.max_denominator, self.codec.profile.max_numerator,
                lower, upper, positive,
            )
            return bisect_left(feasible, low) < bisect_right(feasible, high)
        nlow, nhigh = _numerator_bounds(
            self.denominator, lower, upper, positive,
            self.codec.profile.max_numerator,
        )
        return _has_coprime(max(low, nlow), min(high, nhigh), self.denominator)

    def step(self) -> GrammarStep | None:
        if self.complete:
            return None
        if self.field == "event":
            allowed = (NOTE, EOS) if len(self.events) < self.codec.profile.max_events else (EOS,)
        elif self.field.endswith("_length"):
            maximum_length = 2 if "_den_" in self.field else 8
            allowed = tuple(LEN_BASE + n - 1 for n in range(1, maximum_length + 1)
                            if self._integer_possible(n, ()))
        elif self.field.endswith("_byte"):
            allowed = tuple(BYTE_BASE + byte for byte in range(256)
                            if self._integer_possible(self.length, (*self.byte_prefix, byte)))
        elif self.field == "track":
            lower = self.events[-1].track if self._same("onset") else 0
            allowed = tuple(TRACK_BASE + track for track in self.tracks if track >= lower)
        elif self.field == "kind":
            lower = self.events[-1].kind if self._same("onset", "track") else 0
            allowed = tuple(KIND_BASE + kind for kind in range(lower, 4))
        else:
            minimum_pitch = self.events[-1].pitch if self._same("onset", "track", "kind") else 0
            if self.field == "pitch_class":
                allowed = tuple(PC_BASE + pc for pc in range(12)
                                if any(minimum_pitch <= 12 * octave + pc <= 127
                                       for octave in range(11)))
            elif self.field == "octave":
                allowed = tuple(OCTAVE_BASE + octave for octave in range(11)
                                if minimum_pitch <= 12 * octave + self.current["pc"] <= 127)
            elif self.field == "drum":
                allowed = tuple(DRUM_BASE + pitch for pitch in range(minimum_pitch, 128))
            else:
                raise AssertionError(f"unhandled grammar field {self.field}")
        if not allowed:
            raise AssertionError(f"grammar dead end at {self.field}")
        return GrammarStep(self.field, allowed, len(self.events), len(self.current_tokens))

    def feed(self, token: int, allowed: tuple[int, ...] | None = None):
        if self.complete:
            raise ValueError("token after terminal EOS")
        if allowed is None:
            allowed = self.step().allowed
        if isinstance(token, bool) or not isinstance(token, int) or token not in allowed:
            raise ValueError(f"invalid token {token!r} in {self.field}")
        self.current_tokens.append(token)
        if self.field == "event":
            if token == EOS:
                self.complete = True
            else:
                self.field = "onset_den_length"
            return
        if self.field.endswith("_length"):
            self.length = token - LEN_BASE + 1
            self.byte_prefix = ()
            self.field = self.field.replace("_length", "_byte")
            return
        if self.field.endswith("_byte"):
            self.byte_prefix = (*self.byte_prefix, token - BYTE_BASE)
            if len(self.byte_prefix) < self.length:
                return
            number = int.from_bytes(bytes(self.byte_prefix), "big")
            name = self.field.split("_", 1)[0]
            if "_den_" in self.field:
                self.denominator = number
                self.field = name + "_num_length"
            else:
                self.current[name] = RationalTime(number, self.denominator)
                if name == "onset":
                    self.field = "track"
                else:
                    self.events.append(Event(**self.current))
                    self.current = {}
                    self.current_tokens = []
                    self.field = "event"
            return
        if self.field == "track":
            self.current["track"] = token - TRACK_BASE
            self.field = "kind"
        elif self.field == "kind":
            self.current["kind"] = token - KIND_BASE
            self.field = "pitch_class" if token - KIND_BASE < 2 else "drum"
        elif self.field == "pitch_class":
            self.current["pc"] = token - PC_BASE
            self.field = "octave"
        elif self.field == "octave":
            self.current["pitch"] = 12 * (token - OCTAVE_BASE) + self.current.pop("pc")
            self.field = "duration_den_length"
        elif self.field == "drum":
            self.current["pitch"] = token - DRUM_BASE
            self.field = "duration_den_length"
