"""Exact raw-only time candidates, independent of graphs and theory targets.

The only observed piece attributes are ``notes`` and ``beats``; note records
contribute ``onset_qn``/``duration_qn`` and beat records contribute
``start_qn``/``duration_qn``. Beat boundaries include both starts and ends,
including the final end. Grace notes and percussion are included without
consulting their flags. There is no rounding, clipping, interpolation, or
filtering against target times.

This is an isolated candidate generator, not a new model readout. It does not
validate an entire CanonicalPiece, read annotation metadata, or infer times of
intermediate tied segments that are absent from the raw note records.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Protocol

from music_critic.data.timing import RationalTime


class _RawNote(Protocol):
    onset_qn: RationalTime | Fraction
    duration_qn: RationalTime | Fraction


class _RawBeat(Protocol):
    start_qn: RationalTime | Fraction
    duration_qn: RationalTime | Fraction


class _RawTimedPiece(Protocol):
    notes: Iterable[_RawNote]
    beats: Iterable[_RawBeat]


def _exact_fraction(value: object, field: str) -> Fraction:
    if isinstance(value, RationalTime):
        return Fraction(value.num, value.den)
    if isinstance(value, Fraction):
        return value
    raise TypeError(f"{field} must be exact RationalTime or Fraction")


def raw_time_sources(piece: _RawTimedPiece) -> dict[str, frozenset[Fraction]]:
    """Return deduplicated raw attacks, offsets, and beat starts plus ends.

    Durations must be nonnegative. All supplied exact start times are retained;
    no piece bounds, target masks, track roles, or note attributes are consulted.
    Input order is immaterial. Each source is scanned once in O(N + B) time.
    The immutable sets have no semantic iteration order; sort them to serialize
    an ordered candidate sequence.
    """

    onsets: set[Fraction] = set()
    offsets: set[Fraction] = set()
    boundaries: set[Fraction] = set()
    for index, note in enumerate(piece.notes):
        onset = _exact_fraction(note.onset_qn, f"notes[{index}].onset_qn")
        duration = _exact_fraction(note.duration_qn, f"notes[{index}].duration_qn")
        if duration < 0:
            raise ValueError(f"notes[{index}].duration_qn must be nonnegative")
        onsets.add(onset)
        offsets.add(onset + duration)
    for index, beat in enumerate(piece.beats):
        start = _exact_fraction(beat.start_qn, f"beats[{index}].start_qn")
        duration = _exact_fraction(beat.duration_qn, f"beats[{index}].duration_qn")
        if duration < 0:
            raise ValueError(f"beats[{index}].duration_qn must be nonnegative")
        boundaries.update((start, start + duration))
    return {
        "note_onsets": frozenset(onsets),
        "note_offsets": frozenset(offsets),
        "beat_boundaries": frozenset(boundaries),
    }


def candidate_time_sets(piece: _RawTimedPiece) -> dict[str, frozenset[Fraction]]:
    """Return four fixed unions; ``beats`` always means starts AND ends.

    These alternatives are declared independently of targets. Empty sources
    remain empty; duplicates (including zero-duration note onsets/offsets) do
    not create additional candidate times.
    """

    sources = raw_time_sources(piece)
    onsets = sources["note_onsets"]
    offsets = sources["note_offsets"]
    beats = sources["beat_boundaries"]
    return {
        "note_onsets": onsets,
        "note_onsets_offsets": onsets | offsets,
        "note_onsets_beats": onsets | beats,
        "note_onsets_offsets_beats": onsets | offsets | beats,
    }


__all__ = ["candidate_time_sets", "raw_time_sources"]
