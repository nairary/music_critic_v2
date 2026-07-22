"""Deterministic conversion from validated canonical observations to PyG."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from statistics import fmean
from typing import Iterable, TypeAlias

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalNote,
    CanonicalPiece,
    RationalTime,
    validate_piece,
)
from music_critic.graph.feature_registry import (
    RAW_FEATURE_REGISTRY,
    FeatureRegistry,
    NodeType,
)
from music_critic.graph.relations import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    REVERSE_EDGE_TYPES,
)


FeatureValue: TypeAlias = tuple[int | float | None, bool]
FeatureRow: TypeAlias = dict[str, FeatureValue]
EdgePair: TypeAlias = tuple[int, int]
EdgeType: TypeAlias = tuple[str, str, str]


class GraphBuildError(ValueError):
    """Raised when canonical raw structure cannot form the graph contract."""


def _float_time(value: RationalTime) -> float:
    """Convert exact structural time only at the tensor feature boundary."""

    return value.num / value.den


def _offset(note: CanonicalNote) -> RationalTime:
    return note.onset_qn + note.duration_qn


def _point_owner(
    point: RationalTime,
    starts: tuple[RationalTime, ...],
    ends: tuple[RationalTime, ...],
) -> int | None:
    """Find a half-open interval owner in O(log intervals)."""

    if not starts:
        return None
    index = bisect_right(starts, point) - 1
    if index >= 0 and point < ends[index]:
        return index
    if point == ends[-1]:
        return len(starts) - 1
    return None


def _effective_values(
    points: Iterable[RationalTime],
    event_starts: tuple[RationalTime, ...],
    event_values: tuple[int, ...],
) -> list[int]:
    values: list[int] = []
    for point in points:
        event_index = bisect_right(event_starts, point) - 1
        if event_index < 0:
            raise GraphBuildError("canonical piece has no effective event at qn 0")
        values.append(event_values[event_index])
    return values


def _pack_features(
    data: HeteroData,
    node_type: NodeType,
    rows: list[FeatureRow],
    entity_ids: tuple[str, ...],
    registry: FeatureRegistry,
) -> None:
    categorical = registry.for_node(node_type, "categorical")
    continuous = registry.for_node(node_type, "continuous")
    expected = {spec.name for spec in (*categorical, *continuous)}

    for row in rows:
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise GraphBuildError(
                f"{node_type} feature row mismatch: missing={missing}, extra={extra}"
            )

    x_cat = torch.empty((len(rows), len(categorical)), dtype=torch.long)
    x_cat_available = torch.empty((len(rows), len(categorical)), dtype=torch.bool)
    x_cont = torch.empty((len(rows), len(continuous)), dtype=torch.float32)
    x_cont_available = torch.empty((len(rows), len(continuous)), dtype=torch.bool)

    for row_index, row in enumerate(rows):
        for column, spec in enumerate(categorical):
            value, available = row[spec.name]
            try:
                encoded = spec.encode_category(
                    value if isinstance(value, int) and not isinstance(value, bool) else None,
                    available=available,
                )
            except ValueError as exc:
                raise GraphBuildError(str(exc)) from exc
            x_cat[row_index, column] = encoded
            x_cat_available[row_index, column] = available
        for column, spec in enumerate(continuous):
            value, available = row[spec.name]
            if available:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise GraphBuildError(
                        f"available {node_type}.{spec.name} must be numeric"
                    )
                encoded_continuous = float(value)
            else:
                encoded_continuous = spec.unavailable_continuous_value
            x_cont[row_index, column] = encoded_continuous
            x_cont_available[row_index, column] = available

    store = data[node_type]
    store.num_nodes = len(rows)
    store.x_cat = x_cat
    store.x_cat_available = x_cat_available
    store.x_cont = x_cont
    store.x_cont_available = x_cont_available
    store.entity_id = entity_ids
    store.cat_feature_names = tuple(spec.name for spec in categorical)
    store.cont_feature_names = tuple(spec.name for spec in continuous)


def _edge_tensor(pairs: list[EdgePair]) -> Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _add_forward_and_reverse(
    edge_pairs: dict[EdgeType, list[EdgePair]],
    edge_type: EdgeType,
    pairs: Iterable[EdgePair],
) -> None:
    pair_list = list(pairs)
    edge_pairs[edge_type].extend(pair_list)
    reverse = REVERSE_EDGE_TYPES[edge_type]
    edge_pairs[reverse].extend((destination, source) for source, destination in pair_list)


@dataclass(slots=True)
class _PieceIndex:
    track_index: dict[str, int]
    bar_index: dict[str, int]
    note_indices_by_track: dict[str, list[int]]
    note_indices_by_track_onset: dict[str, dict[RationalTime, list[int]]]
    onset_times: tuple[RationalTime, ...]
    onset_index: dict[RationalTime, int]
    note_indices_by_onset: dict[RationalTime, list[int]]
    onset_indices_by_track: dict[str, list[int]]
    onset_bar_owner: tuple[int | None, ...]
    onset_beat_owner: tuple[int | None, ...]
    note_bar_owner: tuple[int | None, ...]
    note_indices_by_bar: list[list[int]]
    note_indices_by_beat: list[list[int]]
    onset_indices_by_bar: list[list[int]]
    onset_indices_by_beat: list[list[int]]
    bar_active_note_count: list[int]
    bar_active_track_count: list[int]
    beat_active_note_count: list[int]
    beat_active_track_count: list[int]
    onset_active_note_count: list[int]
    onset_active_track_count: list[int]
    active_bar_indices_by_track: dict[str, set[int]]
    sustained_pairs: list[EdgePair]


def _build_piece_index(piece: CanonicalPiece) -> _PieceIndex:
    track_index = {track.track_id: index for index, track in enumerate(piece.tracks)}
    bar_index = {bar.bar_id: index for index, bar in enumerate(piece.bars)}
    note_indices_by_track = {track.track_id: [] for track in piece.tracks}
    note_indices_by_track_onset: dict[
        str, dict[RationalTime, list[int]]
    ] = {track.track_id: defaultdict(list) for track in piece.tracks}
    note_indices_by_onset: dict[RationalTime, list[int]] = defaultdict(list)
    for note_index, note in enumerate(piece.notes):
        if note.track_id not in note_indices_by_track:
            raise GraphBuildError(f"note {note.note_id} references an unknown track")
        note_indices_by_track[note.track_id].append(note_index)
        note_indices_by_track_onset[note.track_id][note.onset_qn].append(note_index)
        note_indices_by_onset[note.onset_qn].append(note_index)

    onset_times = tuple(note_indices_by_onset)
    onset_index = {time: index for index, time in enumerate(onset_times)}
    onset_indices_by_track = {
        track_id: [onset_index[time] for time in onset_groups]
        for track_id, onset_groups in note_indices_by_track_onset.items()
    }
    bar_starts = tuple(bar.start_qn for bar in piece.bars)
    bar_ends = tuple(bar.start_qn + bar.duration_qn for bar in piece.bars)
    beat_starts = tuple(beat.start_qn for beat in piece.beats)
    beat_ends = tuple(beat.start_qn + beat.duration_qn for beat in piece.beats)

    onset_bar_owner = tuple(
        _point_owner(time, bar_starts, bar_ends) for time in onset_times
    )
    onset_beat_owner = tuple(
        _point_owner(time, beat_starts, beat_ends) for time in onset_times
    )
    note_bar_owner = tuple(
        onset_bar_owner[onset_index[note.onset_qn]] for note in piece.notes
    )

    note_indices_by_bar = [[] for _ in piece.bars]
    note_indices_by_beat = [[] for _ in piece.beats]
    onset_indices_by_bar = [[] for _ in piece.bars]
    onset_indices_by_beat = [[] for _ in piece.beats]
    for current_onset_index, owner in enumerate(onset_bar_owner):
        if owner is not None:
            onset_indices_by_bar[owner].append(current_onset_index)
    for current_onset_index, owner in enumerate(onset_beat_owner):
        if owner is not None:
            onset_indices_by_beat[owner].append(current_onset_index)
    for note_index, note in enumerate(piece.notes):
        onset = onset_index[note.onset_qn]
        bar_owner = onset_bar_owner[onset]
        beat_owner = onset_beat_owner[onset]
        if bar_owner is not None:
            note_indices_by_bar[bar_owner].append(note_index)
        if beat_owner is not None:
            note_indices_by_beat[beat_owner].append(note_index)

    bar_active_note_count = [0] * len(piece.bars)
    bar_active_tracks = [set() for _ in piece.bars]
    beat_active_note_count = [0] * len(piece.beats)
    beat_active_tracks = [set() for _ in piece.beats]
    active_bar_indices_by_track = {
        track.track_id: set() for track in piece.tracks
    }
    sustained_pairs: list[EdgePair] = []

    for note_index, note in enumerate(piece.notes):
        if note.duration_qn.num == 0:
            owner = note_bar_owner[note_index]
            if owner is not None:
                bar_active_note_count[owner] += 1
                bar_active_tracks[owner].add(note.track_id)
                active_bar_indices_by_track[note.track_id].add(owner)
            continue

        note_end = _offset(note)
        first_bar = bisect_right(bar_ends, note.onset_qn)
        stop_bar = bisect_left(bar_starts, note_end)
        for current_bar in range(first_bar, stop_bar):
            bar_active_note_count[current_bar] += 1
            bar_active_tracks[current_bar].add(note.track_id)
            active_bar_indices_by_track[note.track_id].add(current_bar)

        first_beat = bisect_left(beat_starts, note.onset_qn)
        stop_beat = bisect_left(beat_starts, note_end)
        for current_beat in range(first_beat, stop_beat):
            beat_active_note_count[current_beat] += 1
            beat_active_tracks[current_beat].add(note.track_id)
            sustained_pairs.append((note_index, current_beat))

    onset_active_note_count: list[int] = []
    onset_active_track_count: list[int] = []
    active_heap: list[tuple[RationalTime, int]] = []
    active_count_by_track: dict[str, int] = defaultdict(int)
    for time in onset_times:
        for note_index in note_indices_by_onset[time]:
            note = piece.notes[note_index]
            if note.duration_qn.num > 0:
                heappush(active_heap, (_offset(note), note_index))
                active_count_by_track[note.track_id] += 1
        while active_heap and active_heap[0][0] <= time:
            _, expired_index = heappop(active_heap)
            expired_track = piece.notes[expired_index].track_id
            active_count_by_track[expired_track] -= 1
            if active_count_by_track[expired_track] == 0:
                del active_count_by_track[expired_track]
        onset_active_note_count.append(len(active_heap))
        onset_active_track_count.append(len(active_count_by_track))

    return _PieceIndex(
        track_index=track_index,
        bar_index=bar_index,
        note_indices_by_track=note_indices_by_track,
        note_indices_by_track_onset=note_indices_by_track_onset,
        onset_times=onset_times,
        onset_index=onset_index,
        note_indices_by_onset=note_indices_by_onset,
        onset_indices_by_track=onset_indices_by_track,
        onset_bar_owner=onset_bar_owner,
        onset_beat_owner=onset_beat_owner,
        note_bar_owner=note_bar_owner,
        note_indices_by_bar=note_indices_by_bar,
        note_indices_by_beat=note_indices_by_beat,
        onset_indices_by_bar=onset_indices_by_bar,
        onset_indices_by_beat=onset_indices_by_beat,
        bar_active_note_count=bar_active_note_count,
        bar_active_track_count=[len(tracks) for tracks in bar_active_tracks],
        beat_active_note_count=beat_active_note_count,
        beat_active_track_count=[len(tracks) for tracks in beat_active_tracks],
        onset_active_note_count=onset_active_note_count,
        onset_active_track_count=onset_active_track_count,
        active_bar_indices_by_track=active_bar_indices_by_track,
        sustained_pairs=sustained_pairs,
    )


def _track_rows(piece: CanonicalPiece, index: _PieceIndex) -> list[FeatureRow]:
    duration = _float_time(piece.duration_qn)
    rows: list[FeatureRow] = []
    for track in piece.tracks:
        track_notes = [
            piece.notes[note_index]
            for note_index in index.note_indices_by_track[track.track_id]
        ]
        pitches = [note.pitch for note in track_notes]
        durations = [_float_time(note.duration_qn) for note in track_notes]
        velocities = [note.velocity for note in track_notes if note.velocity is not None]
        onset_groups = index.note_indices_by_track_onset[track.track_id]
        track_onsets = index.onset_indices_by_track[track.track_id]
        mean_pitch = fmean(pitches) if pitches else 0.0
        pitch_std = (
            sqrt(fmean((pitch - mean_pitch) ** 2 for pitch in pitches))
            if pitches
            else 0.0
        )
        rows.append(
            {
                "program": (track.program, track.program is not None),
                "channel": (track.channel, track.channel is not None),
                "is_percussion": (int(track.is_percussion), True),
                "source_track_index": (
                    track.source_track_index,
                    track.source_track_index is not None,
                ),
                "note_count": (len(track_notes), True),
                "mean_pitch": (mean_pitch if pitches else None, bool(pitches)),
                "pitch_std": (pitch_std if pitches else None, bool(pitches)),
                "min_pitch": (min(pitches) if pitches else None, bool(pitches)),
                "max_pitch": (max(pitches) if pitches else None, bool(pitches)),
                "note_density": (
                    len(track_notes) / duration if duration > 0 else None,
                    duration > 0,
                ),
                "polyphony_ratio": (
                    sum(
                        len(note_indices) > 1
                        for note_indices in onset_groups.values()
                    )
                    / len(track_onsets)
                    if track_onsets
                    else None,
                    bool(track_onsets),
                ),
                "active_bar_ratio": (
                    len(index.active_bar_indices_by_track[track.track_id])
                    / len(piece.bars)
                    if piece.bars
                    else None,
                    bool(piece.bars),
                ),
                "mean_duration_qn": (
                    fmean(durations) if durations else None,
                    bool(durations),
                ),
                "mean_velocity": (
                    fmean(velocities) if velocities else None,
                    bool(velocities),
                ),
            }
        )
    return rows


def _require_valid_piece(piece: CanonicalPiece) -> None:
    report = validate_piece(piece)
    if report.errors:
        summary = ", ".join(
            f"{issue.code}@{issue.path}" for issue in report.errors[:8]
        )
        suffix = "" if len(report.errors) <= 8 else f" (+{len(report.errors) - 8} more)"
        raise GraphBuildError(
            "build_raw_graph requires a validator-clean CanonicalPiece: "
            f"{summary}{suffix}"
        )


def build_raw_graph(
    piece: CanonicalPiece,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
    *,
    raw_only: bool = True,
    assume_valid: bool = False,
) -> HeteroData:
    """Build the Phase 3A raw-only graph.

    By default the complete ``CanonicalPiece`` is validated before construction.
    ``assume_valid=True`` is an explicit fast path for a piece that the caller
    has already checked with :func:`music_critic.data.validate_piece`; behavior
    is undefined for invalid or non-canonically ordered input on that path.

    Supervisory fields are validated as part of the default input precondition,
    but targets, annotations, provenance, source identity, grouping, and split
    are never read while constructing features or relations.
    """

    if not isinstance(piece, CanonicalPiece):
        raise GraphBuildError("piece must be a CanonicalPiece")
    if not raw_only:
        raise GraphBuildError("Phase 3A implements only raw_only=True graphs")
    if not assume_valid:
        _require_valid_piece(piece)
    if piece.schema_version != SCHEMA_VERSION:
        raise GraphBuildError(
            f"unsupported canonical schema {piece.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if any(not spec.raw_inference_safe for spec in registry.specs):
        raise GraphBuildError("feature registry contains a raw-inference-unsafe field")

    piece_index = _build_piece_index(piece)
    meters = {
        event.meter_event_id: (event.numerator, event.denominator)
        for event in piece.meter_events
    }
    tempo_starts = tuple(event.onset_qn for event in piece.tempo_events)
    tempo_values = tuple(
        event.microseconds_per_quarter for event in piece.tempo_events
    )
    bar_tempos = _effective_values(
        (bar.start_qn for bar in piece.bars), tempo_starts, tempo_values
    )
    beat_tempos = _effective_values(
        (beat.start_qn for beat in piece.beats), tempo_starts, tempo_values
    )

    data = HeteroData()
    data.schema_version = piece.schema_version
    data.graph_schema_version = GRAPH_SCHEMA_VERSION
    data.feature_registry_version = registry.version
    data.graph_builder_version = GRAPH_BUILDER_VERSION
    data.raw_only = True

    song_rows: list[FeatureRow] = [
        {
            "duration_qn": (_float_time(piece.duration_qn), True),
            "track_count": (len(piece.tracks), True),
            "bar_count": (len(piece.bars), True),
            "beat_count": (len(piece.beats), True),
            "onset_count": (len(piece_index.onset_times), True),
            "note_count": (len(piece.notes), True),
            "tempo_mean_us_per_qn": (
                fmean(tempo_values) if tempo_values else None,
                bool(tempo_values),
            ),
            "tempo_min_us_per_qn": (
                min(tempo_values) if tempo_values else None,
                bool(tempo_values),
            ),
            "tempo_max_us_per_qn": (
                max(tempo_values) if tempo_values else None,
                bool(tempo_values),
            ),
            "tempo_change_count": (max(0, len(piece.tempo_events) - 1), True),
            "meter_change_count": (max(0, len(piece.meter_events) - 1), True),
        }
    ]
    _pack_features(data, "song", song_rows, (piece.piece_id,), registry)
    _pack_features(
        data,
        "track",
        _track_rows(piece, piece_index),
        tuple(track.track_id for track in piece.tracks),
        registry,
    )

    bar_rows: list[FeatureRow] = []
    for bar_position, bar in enumerate(piece.bars):
        numerator, denominator = meters[bar.meter_event_id]
        bar_rows.append(
            {
                "meter_numerator": (numerator, True),
                "meter_denominator_log2": (denominator.bit_length() - 1, True),
                "is_pickup": (int(bar.is_pickup), True),
                "is_incomplete": (int(bar.is_incomplete), True),
                "index": (bar.index, True),
                "start_qn": (_float_time(bar.start_qn), True),
                "duration_qn": (_float_time(bar.duration_qn), True),
                "metric_offset_qn": (_float_time(bar.metric_offset_qn), True),
                "tempo_us_per_qn": (bar_tempos[bar_position], True),
                "starting_note_count": (
                    len(piece_index.note_indices_by_bar[bar_position]),
                    True,
                ),
                "active_note_count": (
                    piece_index.bar_active_note_count[bar_position],
                    True,
                ),
                "onset_count": (
                    len(piece_index.onset_indices_by_bar[bar_position]),
                    True,
                ),
                "active_track_count": (
                    piece_index.bar_active_track_count[bar_position],
                    True,
                ),
            }
        )
    _pack_features(
        data,
        "bar",
        bar_rows,
        tuple(bar.bar_id for bar in piece.bars),
        registry,
    )

    beat_rows: list[FeatureRow] = []
    for beat_position, beat in enumerate(piece.beats):
        numerator, denominator = meters[beat.meter_event_id]
        beat_rows.append(
            {
                "meter_numerator": (numerator, True),
                "meter_denominator_log2": (denominator.bit_length() - 1, True),
                "is_downbeat": (int(beat.is_downbeat), True),
                "index_in_bar": (beat.index_in_bar, True),
                "start_qn": (_float_time(beat.start_qn), True),
                "duration_qn": (_float_time(beat.duration_qn), True),
                "position_in_bar_qn": (
                    _float_time(beat.position_in_bar_qn),
                    True,
                ),
                "strength": (beat.strength, beat.strength is not None),
                "tempo_us_per_qn": (beat_tempos[beat_position], True),
                "starting_note_count": (
                    len(piece_index.note_indices_by_beat[beat_position]),
                    True,
                ),
                "active_note_count": (
                    piece_index.beat_active_note_count[beat_position],
                    True,
                ),
                "active_track_count": (
                    piece_index.beat_active_track_count[beat_position],
                    True,
                ),
            }
        )
    _pack_features(
        data,
        "beat",
        beat_rows,
        tuple(beat.beat_id for beat in piece.beats),
        registry,
    )
    data["beat"].candidate_slot = torch.ones(len(piece.beats), dtype=torch.bool)

    onset_rows: list[FeatureRow] = []
    for current_onset, time in enumerate(piece_index.onset_times):
        owner_bar = piece_index.onset_bar_owner[current_onset]
        owner_beat = piece_index.onset_beat_owner[current_onset]
        position = (
            time - piece.bars[owner_bar].start_qn
            if owner_bar is not None
            else None
        )
        onset_rows.append(
            {
                "start_qn": (_float_time(time), True),
                "position_in_bar_qn": (
                    _float_time(position) if position is not None else None,
                    position is not None,
                ),
                "starting_note_count": (
                    len(piece_index.note_indices_by_onset[time]),
                    True,
                ),
                "active_note_count": (
                    piece_index.onset_active_note_count[current_onset],
                    True,
                ),
                "active_track_count": (
                    piece_index.onset_active_track_count[current_onset],
                    True,
                ),
                "onsets_in_beat": (
                    len(piece_index.onset_indices_by_beat[owner_beat])
                    if owner_beat is not None
                    else None,
                    owner_beat is not None,
                ),
            }
        )
    onset_ids = tuple(
        f"onset:{time.num}_{time.den}" for time in piece_index.onset_times
    )
    _pack_features(data, "onset", onset_rows, onset_ids, registry)
    data["onset"].candidate_slot = torch.ones(
        len(piece_index.onset_times), dtype=torch.bool
    )

    track_pitch_stats: dict[str, tuple[float, float]] = {}
    for track_id, note_indices in piece_index.note_indices_by_track.items():
        pitches = [piece.notes[note_index].pitch for note_index in note_indices]
        mean = fmean(pitches) if pitches else 0.0
        std = (
            sqrt(fmean((pitch - mean) ** 2 for pitch in pitches))
            if pitches
            else 0.0
        )
        track_pitch_stats[track_id] = (mean, std)

    note_rows: list[FeatureRow] = []
    for note_index, note in enumerate(piece.notes):
        owner_bar = piece_index.note_bar_owner[note_index]
        position = (
            note.onset_qn - piece.bars[owner_bar].start_qn
            if owner_bar is not None
            else None
        )
        mean_pitch, pitch_std = track_pitch_stats[note.track_id]
        note_rows.append(
            {
                "pitch": (note.pitch, True),
                "pitch_class": (note.pitch % 12, True),
                "octave": (note.pitch // 12, True),
                "program": (note.program, note.program is not None),
                "channel": (note.channel, note.channel is not None),
                "is_percussion": (int(note.is_percussion), True),
                "is_grace": (int(note.is_grace), True),
                "onset_qn": (_float_time(note.onset_qn), True),
                "duration_qn": (_float_time(note.duration_qn), True),
                "velocity": (note.velocity, note.velocity is not None),
                "position_in_bar_qn": (
                    _float_time(position) if position is not None else None,
                    position is not None,
                ),
                "track_relative_pitch": (
                    (note.pitch - mean_pitch) / pitch_std if pitch_std > 0 else None,
                    pitch_std > 0,
                ),
            }
        )
    _pack_features(
        data,
        "note",
        note_rows,
        tuple(note.note_id for note in piece.notes),
        registry,
    )

    edge_pairs: dict[EdgeType, list[EdgePair]] = {
        edge_type: [] for edge_type in MANDATORY_EDGE_TYPES
    }
    _add_forward_and_reverse(
        edge_pairs,
        ("song", "contains_track", "track"),
        ((0, track) for track in range(len(piece.tracks))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("song", "contains_bar", "bar"),
        ((0, bar) for bar in range(len(piece.bars))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("track", "contains_note", "note"),
        (
            (piece_index.track_index[note.track_id], note_index)
            for note_index, note in enumerate(piece.notes)
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_beat", "beat"),
        (
            (piece_index.bar_index[beat.bar_id], beat_index)
            for beat_index, beat in enumerate(piece.beats)
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_onset", "onset"),
        (
            (owner, onset_index)
            for onset_index, owner in enumerate(piece_index.onset_bar_owner)
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_note", "note"),
        (
            (owner, note_index)
            for note_index, owner in enumerate(piece_index.note_bar_owner)
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("beat", "contains_onset", "onset"),
        (
            (owner, onset_index)
            for onset_index, owner in enumerate(piece_index.onset_beat_owner)
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("onset", "starts_note", "note"),
        (
            (piece_index.onset_index[note.onset_qn], note_index)
            for note_index, note in enumerate(piece.notes)
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "next_bar", "bar"),
        ((bar, bar + 1) for bar in range(max(0, len(piece.bars) - 1))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("beat", "next_beat", "beat"),
        ((beat, beat + 1) for beat in range(max(0, len(piece.beats) - 1))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("onset", "next_onset", "onset"),
        (
            (onset, onset + 1)
            for onset in range(max(0, len(piece_index.onset_times) - 1))
        ),
    )
    for track in piece.tracks:
        note_indices = piece_index.note_indices_by_track[track.track_id]
        _add_forward_and_reverse(
            edge_pairs,
            ("note", "next_in_track", "note"),
            zip(note_indices, note_indices[1:]),
        )
    _add_forward_and_reverse(
        edge_pairs,
        ("note", "active_at", "beat"),
        piece_index.sustained_pairs,
    )

    for edge_type in MANDATORY_EDGE_TYPES:
        data[edge_type].edge_index = _edge_tensor(edge_pairs[edge_type])

    from music_critic.graph.validation import validate_raw_graph

    validate_raw_graph(data, registry=registry)
    return data


__all__ = ["GraphBuildError", "build_raw_graph"]
