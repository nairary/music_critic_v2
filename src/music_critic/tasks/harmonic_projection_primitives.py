"""Shared literal spelling vocabulary; no source interpretation or tensor I/O."""
import re
from types import MappingProxyType

# Moved unchanged from dense_harmonic_targets and dilemmadata_common.
DEGREES = tuple(f"{a}{d}" for d in range(1, 8)
                for a in ("bbb", "bb", "b", "", "#", "##", "###"))
PITCH_RE = re.compile(r"^([A-Ga-g])((?:#{1,3}|-{1,3}|b{1,3})?)$")
NATURAL_PC = MappingProxyType({"A": 9, "B": 11, "C": 0, "D": 2, "E": 4, "F": 5, "G": 7})


def spelling_pitch_class(value):
    match = PITCH_RE.fullmatch(value)
    if match is None:
        return None
    step, accidental = match.groups()
    alteration = accidental.count("#") - accidental.count("-") - accidental.count("b")
    return (NATURAL_PC[step.upper()] + alteration) % 12


# Frozen common quality templates, also used solely for inversion consistency.
COMMON_QUALITY_INTERVALS = MappingProxyType({
    "augmented triad": (0, 4, 8),
    "diminished seventh chord": (0, 3, 6, 9),
    "diminished triad": (0, 3, 6),
    "dominant seventh chord": (0, 4, 7, 10),
    "half-diminished seventh chord": (0, 3, 6, 10),
    "major seventh chord": (0, 4, 7, 11),
    "major triad": (0, 4, 7),
    "minor seventh chord": (0, 3, 7, 10),
    "minor triad": (0, 3, 7),
    "note": (0,),
})
