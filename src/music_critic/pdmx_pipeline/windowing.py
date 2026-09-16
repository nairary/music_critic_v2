"""Exact, bounded bar windows for already grouped raw PDMX canonical pieces.

No reader is invoked here. The source/cache layer must reject TEST before it
opens a piece; this layer repeats the guard before canonical validation. Notes
are half-open, except zero-duration grace notes at the source terminal boundary
belong to the last window. An ordinary boundary grace belongs to the window
starting at that boundary. Source tick/second fields always describe the
original source note, including on explicitly clipped derivatives.
Tempo, meter and key events use the same half-open nonterminal boundaries;
events at the full source endpoint remain for terminal grace semantics.

The iterator retains one source piece and current window candidates, never all
output pieces. Overlap does not increase the independent-work denominator.
``allow_validation_overlap=True`` acknowledges that any later evaluation must
aggregate by original source piece, not treat windows as independent examples.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
import heapq
import json
from typing import Iterator

from music_critic.data import (
    CanonicalNote,
    CanonicalPiece,
    CanonicalValidationError,
    KeySignatureEvent,
    MeterEvent,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
    TempoEvent,
    validate_or_raise,
)


WINDOW_VERSION = "1.0.0"
_ZERO = RationalTime(0)
_Event = TempoEvent | MeterEvent | KeySignatureEvent


class WindowingError(ValueError):
    """Expected source/window rejection; unexpected exceptions are not caught."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        self.category = reason
        super().__init__(f"{reason}: {message}")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    bar_count: int = 8
    stride: int = 8
    allow_validation_overlap: bool = False
    max_windows_per_piece: int = 100_000
    max_notes_per_window: int = 100_000

    def __post_init__(self) -> None:
        for name in ("bar_count", "stride", "max_windows_per_piece", "max_notes_per_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.allow_validation_overlap, bool):
            raise ValueError("allow_validation_overlap must be a bool")


@dataclass(frozen=True, slots=True)
class WindowResult:
    piece: CanonicalPiece | None
    source_piece_id: str
    source_group_id: str
    source_bar_ids: tuple[str, ...]
    source_bar_start: int
    source_bar_stop: int
    global_offset_qn: RationalTime
    original_note_ids: tuple[str, ...]
    clipped_note_count: int
    reason: str | None = None
    evaluation_unit: str = "source_piece"


def _identity(prefix: str, *parts: str | int) -> str:
    payload = json.dumps((WINDOW_VERSION, *parts), separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}:pdmx-window-{sha256(payload.encode()).hexdigest()}"


def _derived_provenance(
    identifier: str,
    source_piece: CanonicalPiece,
    parent: str | None,
    details: tuple[tuple[str, str | int | bool | None], ...],
) -> ProvenanceRecord:
    return ProvenanceRecord(
        provenance_id=identifier,
        kind="derivation",
        source="music_critic.pdmx_pipeline.windowing",
        record_id=source_piece.piece_id,
        uri=None,
        version=WINDOW_VERSION,
        checksum_sha256=None,
        created_at=None,
        parents=(parent,) if parent is not None else (),
        details=tuple(sorted(details)),
    )


def _context_events(
    events: tuple[_Event, ...],
    onsets: tuple[RationalTime, ...],
    id_field: str,
    start: RationalTime,
    end: RationalTime,
    window_id: str,
    source_piece: CanonicalPiece,
    derivations: dict[str, ProvenanceRecord],
) -> tuple[tuple[_Event, ...], dict[str, str]]:
    """Include start context, excluding the next window's boundary events."""
    active_index = bisect_right(onsets, start) - 1
    selected: list[_Event] = []
    replacements: dict[str, str] = {}
    if active_index >= 0:
        active = events[active_index]
        original_id = getattr(active, id_field)
        if active.onset_qn < start:
            identifier = _identity(original_id.split(":", 1)[0], window_id, original_id, "context")
            provenance_id = _identity("prov", window_id, original_id, "context")
            derivations[provenance_id] = _derived_provenance(
                provenance_id, source_piece, active.provenance_id,
                (("operation", "effective_start_context"), ("original_event_id", original_id),
                 ("original_onset_qn", _time_text(active.onset_qn))),
            )
            active = replace(active, **{id_field: identifier, "provenance_id": provenance_id})
            replacements[original_id] = identifier
        selected.append(replace(active, onset_qn=_ZERO))
    # A nonterminal boundary event belongs to the next window. Including it
    # would change song-level event aggregates even though it has no duration
    # here. A shortened final projected bar validates at its piece endpoint
    # without importing the next meter. True source-terminal events remain
    # applicable to terminal grace notes and preserve full-piece projection.
    begin = bisect_right(onsets, start)
    stop = bisect_right(onsets, end) if end == source_piece.duration_qn else bisect_left(onsets, end)
    selected.extend(replace(event, onset_qn=event.onset_qn - start) for event in events[begin:stop])
    return tuple(selected), replacements


def _time_text(value: RationalTime) -> str:
    return f"{value.num}/{value.den}"


def _provenance_closure(
    all_records: dict[str, ProvenanceRecord], roots: set[str]
) -> tuple[ProvenanceRecord, ...]:
    """Retain only referenced ancestors in the schema's lexical topological order."""
    required: set[str] = set()
    pending = list(roots)
    while pending:
        identifier = pending.pop()
        if identifier not in required:
            required.add(identifier)
            pending.extend(all_records[identifier].parents)
    children: dict[str, list[str]] = {identifier: [] for identifier in required}
    indegree: dict[str, int] = {}
    for identifier in required:
        parents = all_records[identifier].parents
        indegree[identifier] = len(parents)
        for parent in parents:
            children[parent].append(identifier)
    ready = [identifier for identifier in required if indegree[identifier] == 0]
    heapq.heapify(ready)
    result: list[ProvenanceRecord] = []
    while ready:
        identifier = heapq.heappop(ready)
        result.append(all_records[identifier])
        for child in children[identifier]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(required):
        raise RuntimeError("window provenance unexpectedly contains a cycle")
    return tuple(result)


def _make_window(
    piece: CanonicalPiece,
    first: int,
    stop: int,
    candidates: tuple[CanonicalNote, ...],
    beat_onsets: tuple[RationalTime, ...],
    event_onsets: tuple[tuple[RationalTime, ...], ...],
    source_provenance: dict[str, ProvenanceRecord],
) -> WindowResult:
    source_bars = piece.bars[first:stop]
    start = source_bars[0].start_qn
    end = source_bars[-1].start_qn + source_bars[-1].duration_qn
    window_id = _identity("piece", piece.piece_id, first, stop)
    derivations: dict[str, ProvenanceRecord] = {}
    window_provenance_id = _identity("prov", window_id, "window")
    derivations[window_provenance_id] = _derived_provenance(
        window_provenance_id, piece, None,
        (("operation", "bar_window"), ("source_bar_start", first), ("source_bar_stop", stop),
         ("global_offset_qn", _time_text(start)), ("end_qn", _time_text(end)),
         ("event_end_boundary", "closed_source_terminal" if end == piece.duration_qn else "open_window_boundary")),
    )
    tempo, tempo_ids = _context_events(
        piece.tempo_events, event_onsets[0], "tempo_event_id", start, end,
        window_id, piece, derivations,
    )
    meter, meter_ids = _context_events(
        piece.meter_events, event_onsets[1], "meter_event_id", start, end,
        window_id, piece, derivations,
    )
    keys, key_ids = _context_events(
        piece.key_signature_events, event_onsets[2], "key_signature_event_id", start, end,
        window_id, piece, derivations,
    )
    bars = tuple(replace(
        bar, index=index, start_qn=bar.start_qn - start,
        meter_event_id=meter_ids.get(bar.meter_event_id, bar.meter_event_id),
    ) for index, bar in enumerate(source_bars))
    beats = tuple(replace(
        beat, start_qn=beat.start_qn - start,
        meter_event_id=meter_ids.get(beat.meter_event_id, beat.meter_event_id),
    ) for beat in piece.beats[bisect_left(beat_onsets, start):bisect_left(beat_onsets, end)])
    projected: list[tuple[CanonicalNote, str]] = []
    note_ids: dict[str, str] = {}
    clipped_count = 0
    for note in candidates:
        onset = max(start, note.onset_qn)
        offset = min(end, note.onset_qn + note.duration_qn)
        clipped_left = onset != note.onset_qn
        clipped_right = offset != note.onset_qn + note.duration_qn
        if clipped_left or clipped_right:
            clipped_count += 1
            identifier = _identity("note", window_id, note.note_id, "clip")
            provenance_id = _identity("prov", window_id, note.note_id, "clip")
            derivations[provenance_id] = _derived_provenance(
                provenance_id, piece, note.provenance_id,
                (("operation", "clip_note"), ("original_note_id", note.note_id),
                 ("original_onset_qn", _time_text(note.onset_qn)),
                 ("original_duration_qn", _time_text(note.duration_qn)),
                 ("clipped_left", clipped_left), ("clipped_right", clipped_right)),
            )
            projected_note = replace(note, note_id=identifier, onset_qn=onset - start,
                                     duration_qn=offset - onset, provenance_id=provenance_id)
            note_ids[note.note_id] = identifier
        else:
            projected_note = replace(note, onset_qn=onset - start)
        projected.append((projected_note, note.note_id))
    track_order = {track.track_id: index for index, track in enumerate(piece.tracks)}
    projected.sort(key=lambda pair: (
        pair[0].onset_qn, track_order[pair[0].track_id], pair[0].pitch,
        pair[0].duration_qn, pair[0].note_id,
    ))
    notes = tuple(pair[0] for pair in projected)
    # Only flags whose entities survive the projection remain applicable.
    surviving_ids = {window_id}
    for records, field in ((piece.tracks, "track_id"), (notes, "note_id"),
                           (bars, "bar_id"), (beats, "beat_id"), (tempo, "tempo_event_id"),
                           (meter, "meter_event_id"), (keys, "key_signature_event_id")):
        surviving_ids.update(getattr(record, field) for record in records)
    replacement_ids = {piece.piece_id: window_id, **note_ids, **tempo_ids, **meter_ids, **key_ids}
    retained_flags = [replace(
        flag, entity_ids=tuple(replacement_ids.get(entity_id, entity_id) for entity_id in flag.entity_ids)
    ) for flag in piece.quality_flags if all(
        replacement_ids.get(entity_id, entity_id) in surviving_ids for entity_id in flag.entity_ids
    )]
    retained_flags.append(QualityFlag(
        code="pdmx.window_projection", severity="info",
        message="Exact source bar window; source timing fields retain original note coordinates.",
        entity_ids=(window_id,), provenance_id=window_provenance_id,
    ))
    flags = tuple(sorted(retained_flags, key=lambda flag: (flag.code, flag.entity_ids, flag.message)))
    provenance_roots: set[str] = set()
    for records in (piece.tracks, notes, bars, beats, tempo, meter, keys, flags):
        provenance_roots.update(record.provenance_id for record in records if record.provenance_id is not None)
    all_provenance = {**source_provenance, **derivations}
    window = replace(
        piece, piece_id=window_id, duration_qn=end - start, notes=notes,
        bars=bars, beats=beats, tempo_events=tempo, meter_events=meter,
        key_signature_events=keys, annotations=(), targets=(),
        provenance=_provenance_closure(all_provenance, provenance_roots), quality_flags=flags,
    )
    # An invalid projection is a programming error, not a bad-input quarantine.
    validate_or_raise(window)
    return WindowResult(
        piece=window, source_piece_id=piece.piece_id, source_group_id=piece.source_group_id,
        source_bar_ids=tuple(bar.bar_id for bar in source_bars), source_bar_start=first,
        source_bar_stop=stop, global_offset_qn=start,
        original_note_ids=tuple(pair[1] for pair in projected), clipped_note_count=clipped_count,
    )


def iter_bar_windows(
    piece: CanonicalPiece, config: WindowConfig = WindowConfig()
) -> Iterator[WindowResult]:
    """Yield windows and explicit empty/resource rejections in source-bar order.

    A trailing window with fewer than ``bar_count`` source bars is retained.
    Pickup/incomplete source bars keep their exact duration and metric offset.
    The total-window budget rejects the source before emitting any window;
    per-window note limits emit a reason-bearing result without a canonical
    payload. No truncation is silently interpreted as a complete source.
    """
    if piece.split not in ("train", "validation"):
        raise WindowingError("pdmx.split_not_allowed", "only explicit train/validation pieces are permitted")
    if piece.split == "validation" and config.stride < config.bar_count and not config.allow_validation_overlap:
        raise WindowingError("pdmx.validation_overlap_forbidden", "validation overlap requires explicit source-piece aggregation policy")
    if piece.annotations or piece.targets:
        raise WindowingError("pdmx.window_not_raw", "windowing requires the explicit annotation-free raw projection")
    try:
        validate_or_raise(piece)
    except CanonicalValidationError as exc:
        raise WindowingError("pdmx.canonical_invalid", str(exc)) from exc
    window_count = (len(piece.bars) + config.stride - 1) // config.stride
    if window_count > config.max_windows_per_piece:
        raise WindowingError("pdmx.resource_windows", "source exceeds max_windows_per_piece")
    if not piece.bars:
        yield WindowResult(None, piece.piece_id, piece.source_group_id, (), 0, 0, _ZERO, (), 0, "pdmx.window_no_bars")
        return
    beat_onsets = tuple(beat.start_qn for beat in piece.beats)
    event_onsets = tuple(tuple(event.onset_qn for event in events) for events in (
        piece.tempo_events, piece.meter_events, piece.key_signature_events,
    ))
    source_provenance = {record.provenance_id: record for record in piece.provenance}
    active: dict[int, CanonicalNote] = {}
    cursor = 0
    for first in range(0, len(piece.bars), config.stride):
        stop = min(first + config.bar_count, len(piece.bars))
        source_bars = piece.bars[first:stop]
        start = source_bars[0].start_qn
        end = source_bars[-1].start_qn + source_bars[-1].duration_qn
        terminal = end == piece.duration_qn
        active = {index: note for index, note in active.items() if (
            note.onset_qn + note.duration_qn > start
            or (note.duration_qn == _ZERO and note.onset_qn >= start)
        )}
        while cursor < len(piece.notes):
            note = piece.notes[cursor]
            if note.onset_qn > end or (note.onset_qn == end and not terminal):
                break
            if note.onset_qn + note.duration_qn > start or (
                note.duration_qn == _ZERO and note.onset_qn >= start
            ):
                active[cursor] = note
            cursor += 1
        if not active or len(active) > config.max_notes_per_window:
            reason = "pdmx.window_empty" if not active else "pdmx.resource_window_notes"
            yield WindowResult(None, piece.piece_id, piece.source_group_id,
                               tuple(bar.bar_id for bar in source_bars), first, stop,
                               start, (), 0, reason)
            continue
        yield _make_window(piece, first, stop, tuple(active.values()), beat_onsets,
                           event_onsets, source_provenance)
