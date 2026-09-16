from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace

import pytest

from music_critic.data import RationalTime
from music_critic.data.raw_time_readout_slots import (
    ReadoutSlots, align_time_target, build_readout_slots,
)


def _piece():
    return SimpleNamespace(
        notes=(SimpleNamespace(onset_qn=RationalTime(0), duration_qn=RationalTime(2)),),
        beats=(SimpleNamespace(beat_id="beat:a", start_qn=RationalTime(0), duration_qn=RationalTime(2)),),
    )


def _align(slots, onset, **overrides):
    return align_time_target(slots, onset, **{
        "legacy_beat_id": "beat:a", "available": True, "known": True,
        "allow_beat_fallback": True, **overrides,
    })


def test_exact_slot_is_preferred_and_namespaces_are_distinct():
    slots = build_readout_slots(_piece())
    assert slots.time_slots == ((Fraction(0), "rawtime:0_1"), (Fraction(2), "rawtime:2_1"))
    assert slots.beat_ids == frozenset(("beat:a",))
    assert slots.legacy_beat_slots == frozenset(("legacybeat:beat:a",))
    assert not set(slot for _, slot in slots.time_slots).intersection(slots.legacy_beat_slots)
    exact = _align(slots, RationalTime(0))
    assert (exact.slot_id, exact.route, exact.reason) == ("rawtime:0_1", "exact_time", "available_exact_time")
    assert _align(slots, Fraction(2), legacy_beat_id=None).route == "exact_time"


def test_two_missing_times_keep_supervision_but_still_alias_at_legacy_beat():
    slots = build_readout_slots(_piece())
    first = _align(slots, Fraction(1, 2))
    second = _align(slots, Fraction(3, 2))
    assert first.slot_id == second.slot_id == "legacybeat:beat:a"
    assert first.route == second.route == "legacy_beat"
    assert all(time not in (Fraction(1, 2), Fraction(3, 2)) for time, _ in slots.time_slots)
    assert len(slots.time_slots) == 2  # Alignment never inserts target times.


def test_same_time_annotations_are_not_chosen_or_disambiguated():
    slots = build_readout_slots(_piece())
    assert _align(slots, Fraction(0)) == _align(slots, RationalTime(0))


@pytest.mark.parametrize(("available", "known", "reason"), [
    (False, True, "source_unavailable"),
    (False, False, "source_unavailable"),
    (True, False, "unknown_target"),
])
def test_unavailable_masked_or_unknown_target_does_not_read_its_time(available, known, reason):
    # An unavailable/masked row passes available=False, regardless of raw coverage.
    aligned = _align(build_readout_slots(_piece()), object(), available=available, known=known)
    assert (aligned.slot_id, aligned.route, aligned.reason) == (None, "unmatched", reason)


def test_cadence_cannot_borrow_beat_but_can_use_exact_raw_time():
    slots = build_readout_slots(_piece())
    missed = _align(slots, RationalTime(1), allow_beat_fallback=False)
    assert (missed.slot_id, missed.route, missed.reason) == (None, "unmatched", "fallback_not_allowed")
    assert _align(slots, RationalTime(0), allow_beat_fallback=False).route == "exact_time"


@pytest.mark.parametrize(("legacy_id", "reason"), [
    (None, "missing_legacy_beat_id"), ("beat:absent", "unknown_legacy_beat_id"),
])
def test_fallback_requires_an_existing_explicit_legacy_beat(legacy_id, reason):
    aligned = _align(build_readout_slots(_piece()), Fraction(1), legacy_beat_id=legacy_id)
    assert aligned.slot_id is None and aligned.reason == reason


def test_duplicate_raw_beat_ids_fail_closed():
    piece = _piece()
    piece.beats = piece.beats * 2
    with pytest.raises(ValueError, match="duplicate raw beat IDs"):
        build_readout_slots(piece)


def test_selected_fine_policy_does_not_remove_the_legacy_beat_namespace():
    slots = build_readout_slots(_piece(), policy="note_onsets")
    assert slots.time_slots == ((Fraction(0), "rawtime:0_1"),)
    assert slots.legacy_beat_slots == frozenset(("legacybeat:beat:a",))
    assert _align(slots, RationalTime(1)).route == "legacy_beat"


class _ReadTrap:
    def __init__(self, **allowed):
        object.__setattr__(self, "_allowed", allowed)

    def __getattribute__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise AssertionError(f"forbidden field: {name}")
        return allowed[name]


def test_construction_reads_only_whitelisted_raw_attributes():
    piece = _ReadTrap(
        notes=(_ReadTrap(onset_qn=RationalTime(1, 3), duration_qn=RationalTime(0)),),
        beats=(_ReadTrap(beat_id="beat:raw", start_qn=RationalTime(0), duration_qn=RationalTime(1)),),
    )
    slots = build_readout_slots(piece)
    assert slots.time_slots == (
        (Fraction(0), "rawtime:0_1"), (Fraction(1, 3), "rawtime:1_3"), (Fraction(1), "rawtime:1_1"),
    )
    assert slots.beat_ids == frozenset(("beat:raw",))


def test_beat_iterable_is_consumed_once_and_keeps_fallbacks():
    piece = _piece()
    piece.beats = iter(piece.beats)
    slots = build_readout_slots(piece)
    assert slots.beat_ids == frozenset(("beat:a",))
    assert _align(slots, RationalTime(1)).route == "legacy_beat"


def test_slot_inventory_and_alignment_results_are_immutable():
    slots = build_readout_slots(_piece())
    with pytest.raises(FrozenInstanceError):
        slots.time_slots = ()
    with pytest.raises(AttributeError):
        slots.beat_ids.add("beat:injected")
    aligned = _align(slots, Fraction(0))
    with pytest.raises(FrozenInstanceError):
        aligned.slot_id = "rawtime:1_2"


def test_empty_inventory_never_creates_a_target_candidate():
    slots = build_readout_slots(SimpleNamespace(notes=(), beats=()))
    assert slots.time_slots == () and slots.legacy_beat_slots == frozenset()
    assert _align(slots, RationalTime(1)).slot_id is None


@pytest.mark.parametrize("invalid", [0.5, 0, True, "1/2"])
def test_available_target_time_must_be_exact(invalid):
    with pytest.raises(TypeError, match="target onset must be exact"):
        _align(build_readout_slots(_piece()), invalid)


@pytest.mark.parametrize("flag", ["available", "known", "allow_beat_fallback"])
def test_flags_require_bool_not_truthiness(flag):
    with pytest.raises(TypeError, match="flags must be bool"):
        _align(build_readout_slots(_piece()), Fraction(0), **{flag: 1})


def test_unknown_policy_and_manually_malformed_inventory_fail_closed():
    with pytest.raises(ValueError, match="unknown raw time candidate policy"):
        build_readout_slots(_piece(), "from_target_times")
    with pytest.raises(ValueError, match="rawtime ID"):
        ReadoutSlots(((Fraction(0), "legacybeat:beat:a"),), frozenset())
    with pytest.raises(ValueError, match="sorted and unique"):
        ReadoutSlots(((Fraction(0), "rawtime:0_1"),) * 2, frozenset())
