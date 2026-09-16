from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from music_critic.data import RationalTime
from music_critic.data.raw_time_candidates import candidate_time_sets, raw_time_sources


def _piece(notes=(), beats=()):
    return SimpleNamespace(
        notes=tuple(SimpleNamespace(onset_qn=start, duration_qn=duration) for start, duration in notes),
        beats=tuple(SimpleNamespace(start_qn=start, duration_qn=duration) for start, duration in beats),
    )


def test_exact_fractions_and_all_declared_unions():
    piece = _piece(
        notes=((RationalTime(1, 3), RationalTime(2, 7)), (Fraction(2, 3), Fraction(1, 5))),
        beats=((RationalTime(0), Fraction(3, 5)), (Fraction(3, 5), RationalTime(7, 5))),
    )
    onsets = frozenset((Fraction(1, 3), Fraction(2, 3)))
    offsets = frozenset((Fraction(13, 21), Fraction(13, 15)))
    beats = frozenset((Fraction(0), Fraction(3, 5), Fraction(2)))
    assert raw_time_sources(piece) == {
        "note_onsets": onsets, "note_offsets": offsets, "beat_boundaries": beats,
    }
    candidates = candidate_time_sets(piece)
    assert candidates == {
        "note_onsets": onsets,
        "note_onsets_offsets": onsets | offsets,
        "note_onsets_beats": onsets | beats,
        "note_onsets_offsets_beats": onsets | offsets | beats,
    }
    assert all(isinstance(values, frozenset) for values in candidates.values())
    assert all(type(value) is Fraction for values in candidates.values() for value in values)


def test_deduplication_and_input_order_do_not_change_candidates():
    notes = ((RationalTime(1, 3), RationalTime(1, 6)), (RationalTime(2, 3), RationalTime(1, 3)))
    beats = ((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2)))
    expected = candidate_time_sets(_piece(notes, beats))
    assert candidate_time_sets(_piece(notes[::-1] * 2, beats[::-1] * 3)) == expected
    assert sorted(expected["note_onsets_offsets_beats"]) == [
        Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1),
    ]


def test_grace_percussion_and_sustained_offsets_are_included(valid_piece):
    melodic, percussion = valid_piece.notes
    piece = replace(valid_piece, notes=(
        replace(melodic, onset_qn=RationalTime(1, 3), duration_qn=RationalTime(0), is_grace=True),
        replace(percussion, onset_qn=RationalTime(2, 3), duration_qn=RationalTime(1, 6)),
        replace(melodic, onset_qn=RationalTime(0), duration_qn=RationalTime(4)),
    ))
    sources = raw_time_sources(piece)
    assert sources["note_onsets"] == frozenset((Fraction(0), Fraction(1, 3), Fraction(2, 3)))
    assert sources["note_offsets"] == frozenset((Fraction(1, 3), Fraction(5, 6), Fraction(4)))
    assert sources["beat_boundaries"] == frozenset(Fraction(i) for i in range(5))
    # A held note yields its actual end, not invented intermediate attacks.
    assert Fraction(1) not in sources["note_offsets"]


def test_empty_piece_has_all_keys_and_empty_sets():
    sources = raw_time_sources(_piece())
    candidates = candidate_time_sets(_piece())
    assert tuple(sources) == ("note_onsets", "note_offsets", "beat_boundaries")
    assert tuple(candidates) == (
        "note_onsets", "note_onsets_offsets", "note_onsets_beats", "note_onsets_offsets_beats",
    )
    assert all(value == frozenset() for value in (*sources.values(), *candidates.values()))


def test_beat_only_piece_retains_final_end_and_gaps_without_interpolation():
    piece = _piece(beats=((Fraction(0), Fraction(1)), (Fraction(3), Fraction(1, 2))))
    candidates = candidate_time_sets(piece)
    assert candidates["note_onsets_offsets"] == frozenset()
    assert candidates["note_onsets_beats"] == frozenset((Fraction(0), Fraction(1), Fraction(3), Fraction(7, 2)))
    assert Fraction(2) not in candidates["note_onsets_beats"]


class _ReadTrap:
    """Any read outside explicitly supplied raw fields fails, even flags/IDs."""

    def __init__(self, **allowed):
        object.__setattr__(self, "_allowed", allowed)

    def __getattribute__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        if name not in allowed:
            raise AssertionError(f"forbidden non-raw field read: {name}")
        return allowed[name]


def test_only_the_four_raw_time_fields_are_read():
    piece = _ReadTrap(
        notes=(_ReadTrap(onset_qn=RationalTime(1, 3), duration_qn=RationalTime(0)),),
        beats=(_ReadTrap(start_qn=RationalTime(0), duration_qn=RationalTime(1)),),
    )
    assert raw_time_sources(piece)["note_offsets"] == frozenset((Fraction(1, 3),))
    assert candidate_time_sets(piece)["note_onsets_offsets_beats"] == frozenset((Fraction(0), Fraction(1, 3), Fraction(1)))


@pytest.mark.parametrize("field", ["note_onset", "note_duration", "beat_start", "beat_duration"])
@pytest.mark.parametrize("invalid", [0.5, float("nan"), float("inf")])
def test_float_timing_is_rejected_without_rounding(field, invalid):
    values = {name: RationalTime(1) for name in ("note_onset", "note_duration", "beat_start", "beat_duration")}
    values[field] = invalid
    piece = _piece(
        notes=((values["note_onset"], values["note_duration"]),),
        beats=((values["beat_start"], values["beat_duration"]),),
    )
    with pytest.raises(TypeError, match="must be exact RationalTime or Fraction"):
        candidate_time_sets(piece)


@pytest.mark.parametrize("invalid", [0, True, "1/3", None])
def test_unsupported_time_types_are_not_coerced(invalid):
    with pytest.raises(TypeError, match="must be exact RationalTime or Fraction"):
        raw_time_sources(_piece(notes=((invalid, RationalTime(1)),)))


@pytest.mark.parametrize("entity", ["notes", "beats"])
def test_negative_durations_fail_closed(entity):
    with pytest.raises(ValueError, match="duration_qn must be nonnegative"):
        raw_time_sources(_piece(**{entity: ((RationalTime(0), RationalTime(-1, 3)),)}))


def test_exact_starts_are_not_clipped_to_inferred_piece_bounds():
    piece = _piece(notes=((Fraction(-1, 3), Fraction(1, 6)),), beats=())
    assert candidate_time_sets(piece)["note_onsets_offsets"] == frozenset((Fraction(-1, 3), Fraction(-1, 6)))
