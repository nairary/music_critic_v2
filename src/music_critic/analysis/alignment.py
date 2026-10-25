"""Exact source-event alignment primitives for research sidecars."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class AlignmentError(ValueError):
    """An event or alignment request violates the exact-time contract."""


def _rational(value: object, field: str) -> Fraction:
    if not isinstance(value, str):
        raise AlignmentError(f"{field} must be a canonical rational string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise AlignmentError(f"{field} must be a canonical rational string") from exc
    if value != f"{result.numerator}/{result.denominator}":
        raise AlignmentError(f"{field} must be a canonical rational string")
    return result


@dataclass(frozen=True)
class _Event:
    identifier: str
    start: Fraction
    duration: Fraction
    pitch: int


@dataclass(frozen=True)
class MotifMatch:
    """A candidate mapping, not an automatic alignment or musical label."""

    event_ids: tuple[str, ...]
    time_scale: str
    time_offset: str
    pitch_transposition: int


def _events(values: list[dict]) -> list[_Event]:
    if not isinstance(values, list):
        raise AlignmentError("events must be a list of dictionaries")
    parsed, identifiers = [], set()
    for value in values:
        if not isinstance(value, dict):
            raise AlignmentError("event must be a dictionary")
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise AlignmentError("event IDs must be nonempty and unique within the stream")
        identifiers.add(identifier)
        start = _rational(value.get("start"), "event start")
        duration = _rational(value.get("duration"), "event duration")
        if duration < 0:
            raise AlignmentError("event duration must be nonnegative")
        pitch = value.get("pitch")
        if type(pitch) is not int or not 0 <= pitch <= 127:
            raise AlignmentError("event pitch must be an integer MIDI pitch")
        parsed.append(_Event(identifier, start, duration, pitch))
    return parsed


def select_overlapping_events(events: list[dict], start: str, end: str) -> list[str]:
    """Return IDs sounding in ``[start, end)`` in their original input order.

    Both bounds and note times must be reduced rational strings (``"1/3"``).
    The interval must be nonempty. Negative pickups are supported. Zero-duration
    grace events are excluded: a future explicit point mode would be needed to
    select them. This clock-agnostic operation does not convert time units; the
    caller must supply bounds and events in the same declared clock.
    """
    left, right = _rational(start, "start"), _rational(end, "end")
    if left >= right:
        raise AlignmentError("interval start must be strictly before end")
    return [e.identifier for e in _events(events)
            if e.duration > 0 and e.start < right and e.start + e.duration > left]


def find_exact_motif_matches(
    pattern: list[dict], events: list[dict], *, time_unit: str = "quarter_note",
) -> list[MotifMatch]:
    """Return all exact contiguous blocks in the supplied ordered event stream.

    Each candidate has ``target.start = scale * pattern.start + offset``,
    ``target.duration = scale * pattern.duration`` and a constant integer pitch
    transposition. The rational scale must be positive. A pattern must contain
    at least one positive duration to determine scale; other zero durations
    are compared exactly. ``measured_seconds`` and unknown clocks are refused.

    Both lists must already have nondecreasing starts. No event is skipped or
    reordered, including simultaneous notes and duplicate pitches with distinct
    IDs. Their supplied order is part of this deliberately strict contract:
    this is neither unordered chord matching nor a search within inferred
    voices. The caller chooses the event stream explicitly. Extra fields are
    ignored, and the inputs are not mutated.

    There is one candidate anchor per possible block start; a positive pattern
    duration determines its scale. Search costs O(len(events) * len(pattern))
    in the worst case, without combinatorial subsequences or a top-one cutoff.
    Repeated/overlapping candidates are all retained in stream order. None is
    automatically accepted as the correct alignment, harmony or gold target.
    """
    if time_unit != "quarter_note":
        raise AlignmentError("exact motif matching requires the quarter_note clock")
    motif, stream = _events(pattern), _events(events)
    if not motif:
        raise AlignmentError("motif pattern must be nonempty")
    for name, sequence in (("pattern", motif), ("events", stream)):
        if any(right.start < left.start for left, right in zip(sequence, sequence[1:])):
            raise AlignmentError(f"{name} must be ordered by nondecreasing start")
    pivot = next((i for i, note in enumerate(motif) if note.duration > 0), None)
    if pivot is None:
        raise AlignmentError("pattern needs a positive duration to determine time scale")
    matches = []
    for first in range(len(stream) - len(motif) + 1):
        scale = stream[first + pivot].duration / motif[pivot].duration
        if scale <= 0:
            continue
        offset = stream[first].start - scale * motif[0].start
        transposition = stream[first].pitch - motif[0].pitch
        if all(stream[first + i].start == scale * note.start + offset
               and stream[first + i].duration == scale * note.duration
               and stream[first + i].pitch == note.pitch + transposition
               for i, note in enumerate(motif)):
            matches.append(MotifMatch(
                tuple(note.identifier for note in stream[first:first + len(motif)]),
                f"{scale.numerator}/{scale.denominator}",
                f"{offset.numerator}/{offset.denominator}", transposition,
            ))
    return matches
