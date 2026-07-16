from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from music_critic.data import RationalTime


def test_default_denominator_and_normalization() -> None:
    assert RationalTime(3) == RationalTime(3, 1)
    assert RationalTime(6, 8) == RationalTime(3, 4)
    assert RationalTime(-6, 8) == RationalTime(-3, 4)


@pytest.mark.parametrize("denominator", [2, 99, 10**30])
def test_zero_is_canonicalized(denominator: int) -> None:
    value = RationalTime(0, denominator)
    assert (value.num, value.den) == (0, 1)


def test_denominator_remains_positive() -> None:
    value = RationalTime(-3, 7)
    assert value.num == -3
    assert value.den == 7


@pytest.mark.parametrize(
    ("num", "den"),
    [(True, 1), (False, 1), (1, True), (1, False)],
)
def test_bool_construction_is_rejected(num: object, den: object) -> None:
    with pytest.raises(TypeError):
        RationalTime(num, den)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num", "den"),
    [(1.0, 1), ("1", 1), (1, 2.0), (1, "2"), (None, 1)],
)
def test_non_integer_construction_is_rejected(num: object, den: object) -> None:
    with pytest.raises(TypeError):
        RationalTime(num, den)  # type: ignore[arg-type]


@pytest.mark.parametrize("denominator", [0, -1, -10])
def test_non_positive_denominator_is_rejected(denominator: int) -> None:
    with pytest.raises(ValueError):
        RationalTime(1, denominator)


def test_equality_and_hash_use_normalized_values() -> None:
    values = {RationalTime(1, 2), RationalTime(2, 4), RationalTime(50, 100)}
    assert values == {RationalTime(1, 2)}
    assert hash(RationalTime(1, 2)) == hash(RationalTime(2, 4))


def test_numeric_ordering_including_negative_values() -> None:
    values = [
        RationalTime(10, 11),
        RationalTime(-1, 2),
        RationalTime(2, 3),
        RationalTime(-5, 4),
        RationalTime(0),
    ]
    assert sorted(values) == [
        RationalTime(-5, 4),
        RationalTime(-1, 2),
        RationalTime(0),
        RationalTime(2, 3),
        RationalTime(10, 11),
    ]


def test_addition_subtraction_and_negation() -> None:
    left = RationalTime(5, 6)
    right = RationalTime(7, 10)
    assert left + right == RationalTime(23, 15)
    assert left - right == RationalTime(2, 15)
    assert -left == RationalTime(-5, 6)
    assert -RationalTime(0, 9) == RationalTime(0)


def test_multiplication_and_division_are_normalized() -> None:
    value = RationalTime(3, 10)
    assert value * 5 == RationalTime(3, 2)
    assert value * -5 == RationalTime(-3, 2)
    assert value * 0 == RationalTime(0)
    assert value / 3 == RationalTime(1, 10)
    assert value / -3 == RationalTime(-1, 10)


def test_bool_arithmetic_is_rejected() -> None:
    value = RationalTime(1, 2)
    with pytest.raises(TypeError):
        value * True
    with pytest.raises(TypeError):
        value / False


def test_zero_division() -> None:
    with pytest.raises(ZeroDivisionError):
        RationalTime(1, 2) / 0


def test_unsupported_operands_return_not_implemented() -> None:
    value = RationalTime(1, 2)
    assert value.__lt__(1) is NotImplemented
    assert value.__add__(1) is NotImplemented
    assert value.__sub__(1) is NotImplemented
    assert value.__mul__(1.5) is NotImplemented
    assert value.__truediv__(1.5) is NotImplemented

    with pytest.raises(TypeError):
        _ = value + 1  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = value * 1.5  # type: ignore[operator]


def test_fraction_conversion_round_trip_is_exact() -> None:
    fraction = Fraction(-(10**80 + 7), 10**60 + 9)
    value = RationalTime.from_fraction(fraction)
    assert value.to_fraction() == fraction
    assert RationalTime.from_fraction(value.to_fraction()) == value


def test_from_fraction_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        RationalTime.from_fraction(0.5)  # type: ignore[arg-type]


def test_very_large_integer_arithmetic_has_no_float_precision_dependency() -> None:
    large = 10**200 + 123456789
    left = RationalTime(large, 10**100 + 3)
    right = RationalTime(large - 1, 10**100 + 2)

    assert (left + right).to_fraction() == (
        Fraction(large, 10**100 + 3) + Fraction(large - 1, 10**100 + 2)
    )
    assert (left < right) == (
        large * (10**100 + 2) < (large - 1) * (10**100 + 3)
    )


def test_rational_time_is_frozen_and_slotted() -> None:
    value = RationalTime(1, 2)
    with pytest.raises(FrozenInstanceError):
        value.num = 4  # type: ignore[misc]
    assert not hasattr(value, "__dict__")
