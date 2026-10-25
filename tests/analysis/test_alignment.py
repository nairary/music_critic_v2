"""Tiny synthetic events test mechanics, not musical analysis correctness."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from music_critic.analysis.alignment import (
    AlignmentError,
    MotifMatch,
    find_exact_motif_matches,
    select_overlapping_events,
)


def event(identifier, start, duration, pitch=60):
    return {"id": identifier, "start": start, "duration": duration, "pitch": pitch}


def test_span_includes_held_notes_preserves_order_and_excludes_boundaries_and_grace():
    events = [
        event("later", "1/1", "1/1"),
        event("held", "-1/1", "4/1"),
        event("ends_at_start", "-2/1", "2/1"),
        event("starts_at_end", "2/1", "1/1"),
        event("grace", "1/1", "0/1"),
    ]
    assert select_overlapping_events(events, "0/1", "2/1") == ["later", "held"]
    assert select_overlapping_events(events, "-3/2", "-1/1") == ["ends_at_start"]


def test_all_repeated_transposed_affine_candidates_are_preserved():
    pattern = [event("p1", "-1/2", "1/2", 60), event("p2", "1/2", "1/4", 64)]
    stream = [event("a", "3/1", "1/1", 65), event("b", "5/1", "1/2", 69),
              event("c", "10/1", "1/2", 58), event("d", "11/1", "1/4", 62)]
    assert find_exact_motif_matches(pattern, stream) == [
        MotifMatch(("a", "b"), "2/1", "4/1", 5),
        MotifMatch(("c", "d"), "1/1", "21/2", -2),
    ]


def test_simultaneous_duplicate_pitches_keep_ids_and_overlapping_ambiguity():
    pattern = [event("p1", "0/1", "1/1"), event("p2", "0/1", "1/1")]
    stream = [event("a", "4/1", "2/1", 65), event("b", "4/1", "2/1", 65),
              event("c", "4/1", "2/1", 65)]
    assert find_exact_motif_matches(pattern, stream) == [
        MotifMatch(("a", "b"), "2/1", "4/1", 5),
        MotifMatch(("b", "c"), "2/1", "4/1", 5),
    ]
    assert find_exact_motif_matches(pattern, stream[:1]) == []


def test_no_skipping_reordering_or_pitch_set_only_match():
    pattern = [event("p1", "0/1", "1/1", 60), event("p2", "1/1", "1/1", 64)]
    skipped = [event("a", "0/1", "1/1", 60), event("extra", "1/2", "1/2", 72),
               event("b", "1/1", "1/1", 64)]
    reversed_pitch = [event("a", "0/1", "1/1", 64), event("b", "1/1", "1/1", 60)]
    assert find_exact_motif_matches(pattern, skipped) == []
    assert find_exact_motif_matches(pattern, reversed_pitch) == []
    simultaneous = [event("p1", "0/1", "1/1", 60), event("p2", "0/1", "2/1", 64)]
    assert find_exact_motif_matches(simultaneous, simultaneous[::-1]) == []


@pytest.mark.parametrize("changed", [
    [event("a", "0/1", "1/1", 60), event("b", "2/1", "1/1", 64)],
    [event("a", "0/1", "1/1", 60), event("b", "1/1", "2/1", 64)],
    [event("a", "0/1", "1/1", 60), event("b", "1/1", "1/1", 65)],
])
def test_all_onsets_durations_and_pitch_transpositions_must_agree(changed):
    pattern = [event("p1", "0/1", "1/1", 60), event("p2", "1/1", "1/1", 64)]
    assert find_exact_motif_matches(pattern, changed) == []


def test_zero_duration_leading_note_uses_later_duration_anchor():
    pattern = [event("p1", "0/1", "0/1", 60), event("p2", "1/1", "1/1", 62)]
    stream = [event("a", "3/1", "0/1", 64), event("b", "5/1", "2/1", 66)]
    assert find_exact_motif_matches(pattern, stream) == [MotifMatch(("a", "b"), "2/1", "3/1", 4)]
    assert find_exact_motif_matches([event("p", "0/1", "1/1")], [event("z", "0/1", "0/1")]) == []
    with pytest.raises(AlignmentError, match="positive duration"):
        find_exact_motif_matches([event("p", "0/1", "0/1")], stream)


@pytest.mark.parametrize("clock", ["measured_seconds", "seconds", "ticks", "", None])
def test_motif_refuses_measured_seconds_and_unknown_clocks(clock):
    data = [event("p", "0/1", "1/1")]
    with pytest.raises(AlignmentError, match="quarter_note"):
        find_exact_motif_matches(data, data, time_unit=clock)


def test_span_can_select_within_a_declared_measured_seconds_clock_without_quantization():
    # These rational strings preserve decimal measured seconds; no beat conversion.
    data = [event("seconds-note", "2469/200", "67/1000")]
    assert select_overlapping_events(data, "1239/100", "62/5") == ["seconds-note"]
    assert select_overlapping_events(data, "3103/250", "13/1") == []


@pytest.mark.parametrize("bad", ["1", "2/2", "-0/1", "+1/1", "01/1", "1/-2", "1/0", " 1/2 ", 0.5, True, None])
def test_span_bounds_require_canonical_rational_strings(bad):
    with pytest.raises(AlignmentError, match="canonical rational"):
        select_overlapping_events([], bad, "3/1")
    with pytest.raises(AlignmentError, match="canonical rational"):
        select_overlapping_events([], "0/1", bad)


@pytest.mark.parametrize("start,end", [("0/1", "0/1"), ("1/1", "0/1")])
def test_span_must_be_nonempty(start, end):
    with pytest.raises(AlignmentError, match="strictly before"):
        select_overlapping_events([], start, end)


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("id", None), ("start", "1/2 "), ("start", 0.5),
    ("duration", "-1/2"), ("duration", "2/4"), ("duration", "1/0"),
    ("pitch", True), ("pitch", 60.0), ("pitch", -1), ("pitch", 128),
])
def test_event_contract_is_shared_by_span_and_motif(field, value):
    invalid = event("a", "0/1", "1/1")
    invalid[field] = value
    with pytest.raises(AlignmentError):
        select_overlapping_events([invalid], "0/1", "2/1")
    with pytest.raises(AlignmentError):
        find_exact_motif_matches([event("p", "0/1", "1/1")], [invalid])


def test_duplicate_ids_are_errors_instead_of_silent_event_collapse():
    duplicates = [event("same", "0/1", "1/1"), event("same", "1/1", "1/1")]
    with pytest.raises(AlignmentError, match="unique"):
        select_overlapping_events(duplicates, "0/1", "2/1")
    with pytest.raises(AlignmentError, match="unique"):
        find_exact_motif_matches(duplicates, [])
    with pytest.raises(AlignmentError, match="unique"):
        find_exact_motif_matches([event("p", "0/1", "1/1")], duplicates)


def test_equal_pitches_do_not_hide_different_simultaneous_durations():
    pattern = [event("p1", "0/1", "1/1"), event("p2", "0/1", "2/1")]
    swapped = [event("a", "0/1", "2/1"), event("b", "0/1", "1/1")]
    assert find_exact_motif_matches(pattern, swapped) == []


def test_motif_rejects_out_of_order_input_instead_of_sorting_it():
    unordered = [event("a", "1/1", "1/1"), event("b", "0/1", "1/1")]
    with pytest.raises(AlignmentError, match="ordered"):
        find_exact_motif_matches(unordered, [])
    with pytest.raises(AlignmentError, match="ordered"):
        find_exact_motif_matches([event("p", "0/1", "1/1")], unordered)


def test_empty_pattern_rejected_empty_stream_returns_no_candidates():
    assert select_overlapping_events([], "0/1", "1/1") == []
    assert find_exact_motif_matches([event("p", "0/1", "1/1")], []) == []
    with pytest.raises(AlignmentError, match="nonempty"):
        find_exact_motif_matches([], [])


def test_inputs_are_unchanged_and_returned_candidate_is_immutable():
    pattern = [event("p", "-1/3", "2/3")]
    pattern[0]["unknown_theory"] = {"mask": False, "label": None}
    stream = [event("a", "1/3", "1/3", 67)]
    original = deepcopy((pattern, stream))
    result = find_exact_motif_matches(pattern, stream)
    select_overlapping_events(stream, "0/1", "1/1")
    assert (pattern, stream) == original
    assert result == [MotifMatch(("a",), "1/2", "1/2", 7)]
    with pytest.raises(FrozenInstanceError):
        result[0].time_scale = "1/1"
