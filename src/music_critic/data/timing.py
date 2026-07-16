"""Exact rational timing in quarter-note units."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering
from math import gcd


@total_ordering
@dataclass(frozen=True, slots=True)
class RationalTime:
    num: int
    den: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.num, bool) or not isinstance(self.num, int):
            raise TypeError("num must be an int, not bool or another type")
        if isinstance(self.den, bool) or not isinstance(self.den, int):
            raise TypeError("den must be an int, not bool or another type")
        if self.den <= 0:
            raise ValueError("den must be a positive integer")

        if self.num == 0:
            object.__setattr__(self, "den", 1)
            return

        common_divisor = gcd(abs(self.num), self.den)
        object.__setattr__(self, "num", self.num // common_divisor)
        object.__setattr__(self, "den", self.den // common_divisor)

    def __lt__(self, other: RationalTime) -> bool:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return self.num * other.den < other.num * self.den

    def __add__(self, other: RationalTime) -> RationalTime:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return RationalTime(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )

    def __sub__(self, other: RationalTime) -> RationalTime:
        if not isinstance(other, RationalTime):
            return NotImplemented
        return RationalTime(
            self.num * other.den - other.num * self.den,
            self.den * other.den,
        )

    def __neg__(self) -> RationalTime:
        return RationalTime(-self.num, self.den)

    def __mul__(self, factor: int) -> RationalTime:
        if isinstance(factor, bool):
            raise TypeError("factor must be an int, not bool")
        if not isinstance(factor, int):
            return NotImplemented
        return RationalTime(self.num * factor, self.den)

    def __truediv__(self, divisor: int) -> RationalTime:
        if isinstance(divisor, bool):
            raise TypeError("divisor must be an int, not bool")
        if not isinstance(divisor, int):
            return NotImplemented
        if divisor == 0:
            raise ZeroDivisionError("cannot divide RationalTime by zero")
        if divisor < 0:
            return RationalTime(-self.num, self.den * -divisor)
        return RationalTime(self.num, self.den * divisor)

    def to_fraction(self) -> Fraction:
        return Fraction(self.num, self.den)

    @classmethod
    def from_fraction(cls, value: Fraction) -> RationalTime:
        if not isinstance(value, Fraction):
            raise TypeError("value must be a fractions.Fraction")
        return cls(value.numerator, value.denominator)
