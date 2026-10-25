"""Target-free views with explicit geometry and tile-boundary censoring.

The view's note times are global and never clipped. Its `piece` is an internal
observation container, not a canonical export: selected bars need not start at
zero and visible notes may cross the window. Only an ObservedTile's surrogate
piece is passed to the existing canonical graph builder.
"""
from __future__ import annotations

from dataclasses import dataclass

from music_critic.data.schema import CanonicalPiece
from music_critic.data.timing import RationalTime
from .contracts import Event, MaskQuery


class UnsupportedContextError(ValueError):
    """The requested conditional task exceeds support; nothing was truncated."""


def observed_fraction(bar, track_id: str, query: MaskQuery) -> float:
    if track_id not in query.track_ids:
        return 1.0
    overlap = min(bar.start_qn + bar.duration_qn, query.end) - max(bar.start_qn, query.start)
    return 1.0 - max(0.0, float(overlap.to_fraction() / bar.duration_qn.to_fraction()))


@dataclass(frozen=True, slots=True)
class ObservedMusicView:
    piece: CanonicalPiece
    query: MaskQuery
    global_bar_indices: tuple[int, ...]

    @property
    def allowed_tracks(self) -> tuple[int, ...]:
        return tuple(i for i,t in enumerate(self.piece.tracks) if t.track_id in self.query.track_ids)

    @property
    def global_start(self) -> RationalTime:
        return self.piece.bars[0].start_qn

    @property
    def global_end(self) -> RationalTime:
        return self.piece.bars[-1].start_qn + self.piece.bars[-1].duration_qn

    def coverage(self, bar_id: str, track_id: str) -> float:
        if track_id not in {t.track_id for t in self.piece.tracks}:
            raise ValueError('pll.unknown_track')
        return observed_fraction(next(b for b in self.piece.bars if b.bar_id == bar_id), track_id, self.query)


@dataclass(frozen=True, slots=True)
class MaskedExample:
    context: ObservedMusicView
    targets: tuple[Event, ...]
    tracks: tuple[int, ...]

    @property
    def span(self) -> RationalTime:
        return self.context.query.span


@dataclass(frozen=True, slots=True)
class NoteGeometry:
    note_id: str
    onset: RationalTime
    duration: RationalTime
    left_censored: bool
    right_censored: bool


@dataclass(frozen=True, slots=True)
class ObservedTile:
    piece: CanonicalPiece
    global_start: RationalTime
    global_end: RationalTime
    core_bar_ids: tuple[str, ...]
    global_bar_indices: tuple[int, ...]
    note_geometry: tuple[NoteGeometry, ...]
    query: MaskQuery

    @property
    def boundary_flags(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(n.left_censored), float(n.right_censored)) for n in self.note_geometry)

    @property
    def core_note_ids(self) -> tuple[str, ...]:
        bars = [b for b in self.piece.bars if b.bar_id in self.core_bar_ids]
        start = bars[0].start_qn + self.global_start
        end = bars[-1].start_qn + bars[-1].duration_qn + self.global_start
        return tuple(n.note_id for n in self.note_geometry if start <= n.onset < end)
