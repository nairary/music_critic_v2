"""Isolated raw-time slots plus the existing beat fallback namespace.

Construction is target-blind. Alignment is a later operation and cannot create
slots. The beat namespace refers only to existing raw beats; it does not add
semantic chord nodes. This module neither executes nor changes a model.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Literal, Protocol

from music_critic.data.raw_time_candidates import candidate_time_sets
from music_critic.data.timing import RationalTime


CANDIDATE_POLICIES = (
    "note_onsets", "note_onsets_offsets", "note_onsets_beats",
    "note_onsets_offsets_beats",
)
Route = Literal["exact_time", "legacy_beat", "unmatched"]
AlignmentReason = Literal[
    "available_exact_time", "available_legacy_beat_fallback",
    "source_unavailable", "unknown_target", "fallback_not_allowed",
    "missing_legacy_beat_id", "unknown_legacy_beat_id",
]


class _RawNote(Protocol):
    onset_qn: RationalTime | Fraction
    duration_qn: RationalTime | Fraction


class _RawBeat(Protocol):
    beat_id: str
    start_qn: RationalTime | Fraction
    duration_qn: RationalTime | Fraction


class _RawPiece(Protocol):
    notes: Iterable[_RawNote]
    beats: Iterable[_RawBeat]


@dataclass(frozen=True, slots=True)
class _RawSnapshot:
    notes: Iterable[_RawNote]
    beats: tuple[_RawBeat, ...]


def _time_id(time: Fraction) -> str:
    return f"rawtime:{time.numerator}_{time.denominator}"


def _beat_id(value: object) -> str:
    if (
        not isinstance(value, str) or not value or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("raw beat ID must be a nonempty trimmed identifier")
    return value


@dataclass(frozen=True, slots=True)
class ReadoutSlots:
    """Immutable candidate inventory; all IDs are scoped to this one piece.

    ``time_slots`` is sorted by exact time. ``beat_ids`` contains original raw
    beat IDs, while ``legacy_beat_slots`` contains their namespaced output IDs.
    The two namespaces remain distinct even at a beat-start time. There is no
    claim that all source annotation times occur in ``time_slots``.
    """

    time_slots: tuple[tuple[Fraction, str], ...]
    beat_ids: frozenset[str]
    policy: str = "note_onsets_offsets_beats"
    legacy_beat_slots: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.policy not in CANDIDATE_POLICIES:
            raise ValueError("unknown raw time candidate policy")
        if not isinstance(self.time_slots, tuple) or any(
            not isinstance(row, tuple) or len(row) != 2 for row in self.time_slots
        ):
            raise TypeError("time_slots must be an immutable tuple of pairs")
        times = []
        for time, slot_id in self.time_slots:
            if not isinstance(time, Fraction) or slot_id != _time_id(time):
                raise ValueError("time slot must use exact Fraction and its rawtime ID")
            times.append(time)
        if times != sorted(set(times)):
            raise ValueError("time slots must be sorted and unique")
        if not isinstance(self.beat_ids, frozenset):
            raise TypeError("beat_ids must be an immutable frozenset")
        for beat_id in self.beat_ids:
            _beat_id(beat_id)
        object.__setattr__(
            self, "legacy_beat_slots",
            frozenset("legacybeat:" + beat_id for beat_id in self.beat_ids),
        )


@dataclass(frozen=True, slots=True)
class TargetSlotAlignment:
    slot_id: str | None
    route: Route
    reason: AlignmentReason


def build_readout_slots(
    piece: _RawPiece, policy: str = "note_onsets_offsets_beats",
) -> ReadoutSlots:
    """Generate raw-only slots without reading labels, masks or target times.

    Only notes' onset/duration and beats' ID/start/duration are read. A tuple
    snapshot of beats avoids consuming a single-use iterable twice. All raw
    beats remain available for an explicitly permitted later fallback, even if
    the selected fine-time policy omits beat boundaries. Duplicate raw beat IDs
    fail rather than being silently merged.
    """

    if policy not in CANDIDATE_POLICIES:
        raise ValueError("unknown raw time candidate policy")
    beats = tuple(piece.beats)
    beat_ids = tuple(_beat_id(beat.beat_id) for beat in beats)
    if len(beat_ids) != len(set(beat_ids)):
        raise ValueError("duplicate raw beat IDs")
    times = candidate_time_sets(_RawSnapshot(piece.notes, beats))[policy]
    return ReadoutSlots(
        tuple((time, _time_id(time)) for time in sorted(times)),
        frozenset(beat_ids), policy,
    )


def align_time_target(
    slots: ReadoutSlots,
    onset: RationalTime | Fraction,
    *,
    legacy_beat_id: str | None,
    available: bool,
    known: bool,
    allow_beat_fallback: bool,
) -> TargetSlotAlignment:
    """Join one time target after raw-only prediction slots have been built.

    Callers must pass ``available = source_available and not source_masked``.
    Unavailable/unknown targets cannot gain supervision through either route.
    Exact time wins; fallback is permitted only for a caller-authorized harmonic
    head with a verified existing legacy beat mapping. Cadence callers must use
    ``allow_beat_fallback=False``; note targets do not use this API.

    This does not select a label, merge same-time disagreements, or claim that
    two targets sharing a fallback beat become distinguishable. ``legacy_beat_id``
    is an explicit pre-existing mapping, never inferred from gold by construction.
    A match uses O(log T) time for time slots and O(1) for beat membership.
    """

    if any(type(value) is not bool for value in (available, known, allow_beat_fallback)):
        raise TypeError("availability, vocabulary and fallback flags must be bool")
    if not available:
        return TargetSlotAlignment(None, "unmatched", "source_unavailable")
    if not known:
        return TargetSlotAlignment(None, "unmatched", "unknown_target")
    if isinstance(onset, RationalTime):
        time = Fraction(onset.num, onset.den)
    elif isinstance(onset, Fraction):
        time = onset
    else:
        raise TypeError("target onset must be exact RationalTime or Fraction")
    index = bisect_left(slots.time_slots, (time, ""))
    if index < len(slots.time_slots) and slots.time_slots[index][0] == time:
        return TargetSlotAlignment(slots.time_slots[index][1], "exact_time", "available_exact_time")
    if not allow_beat_fallback:
        return TargetSlotAlignment(None, "unmatched", "fallback_not_allowed")
    if legacy_beat_id is None:
        return TargetSlotAlignment(None, "unmatched", "missing_legacy_beat_id")
    legacy_beat_id = _beat_id(legacy_beat_id)
    if legacy_beat_id not in slots.beat_ids:
        return TargetSlotAlignment(None, "unmatched", "unknown_legacy_beat_id")
    return TargetSlotAlignment(
        "legacybeat:" + legacy_beat_id, "legacy_beat", "available_legacy_beat_fallback",
    )


__all__ = [
    "ReadoutSlots", "TargetSlotAlignment", "build_readout_slots", "align_time_target",
]
