"""Deterministic conversion from canonical raw observations to PyG graphs."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import fmean
from typing import Iterable, TypeAlias

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from music_critic.data import SCHEMA_VERSION, CanonicalPiece, RationalTime
from music_critic.graph.feature_registry import (
    RAW_FEATURE_REGISTRY,
    FeatureRegistry,
    NodeType,
)
from music_critic.graph.relations import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    REVERSE_EDGE_TYPES,
)


FeatureValue: TypeAlias = tuple[int | float, bool]
FeatureRow: TypeAlias = dict[str, FeatureValue]
EdgePair: TypeAlias = tuple[int, int]


class GraphBuildError(ValueError):
    """Raised when raw canonical structure cannot form the graph contract."""


def _float_time(value: RationalTime) -> float:
    return value.num / value.den


def _offset(note: object) -> RationalTime:
    return note.onset_qn + note.duration_qn  # type: ignore[attr-defined]


def _contains_point(
    start: RationalTime,
    duration: RationalTime,
    point: RationalTime,
) -> bool:
    return start <= point < start + duration


def _interval_overlaps(
    left_start: RationalTime,
    left_end: RationalTime,
    right_start: RationalTime,
    right_end: RationalTime,
) -> bool:
    return left_start < right_end and right_start < left_end


def _owner_index(
    point: RationalTime,
    intervals: Iterable[tuple[RationalTime, RationalTime]],
) -> int | None:
    materialized = tuple(intervals)
    for index, (start, duration) in enumerate(materialized):
        if _contains_point(start, duration, point):
            return index
    if materialized:
        final_start, final_duration = materialized[-1]
        if point == final_start + final_duration:
            return len(materialized) - 1
    return None


def _tempo_at(piece: CanonicalPiece, point: RationalTime) -> int:
    active = None
    for event in piece.tempo_events:
        if event.onset_qn > point:
            break
        active = event.microseconds_per_quarter
    if active is None:
        raise GraphBuildError("canonical piece has no effective tempo at qn 0")
    return active


def _meter_values(piece: CanonicalPiece) -> dict[str, tuple[int, int]]:
    return {
        event.meter_event_id: (event.numerator, event.denominator)
        for event in piece.meter_events
    }


def _bounded_category(value: int, vocabulary_size: int, unknown_id: int) -> int:
    if 0 <= value < vocabulary_size and value != unknown_id:
        return value
    return unknown_id


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
            encoded = int(value)
            if spec.vocabulary_size is None:
                raise GraphBuildError(f"{spec.name} has no categorical vocabulary")
            if not 0 <= encoded < spec.vocabulary_size:
                raise GraphBuildError(
                    f"{node_type}.{spec.name}={encoded} is outside its vocabulary"
                )
            x_cat[row_index, column] = encoded
            x_cat_available[row_index, column] = available
        for column, spec in enumerate(continuous):
            value, available = row[spec.name]
            x_cont[row_index, column] = float(value)
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
    edge_pairs: dict[tuple[str, str, str], list[EdgePair]],
    edge_type: tuple[str, str, str],
    pairs: Iterable[EdgePair],
) -> None:
    pair_list = list(pairs)
    edge_pairs[edge_type].extend(pair_list)
    reverse = REVERSE_EDGE_TYPES[edge_type]
    edge_pairs[reverse].extend((dst, src) for src, dst in pair_list)


def _track_feature_rows(
    piece: CanonicalPiece,
    notes_by_track: dict[str, list[object]],
) -> list[FeatureRow]:
    duration = _float_time(piece.duration_qn)
    rows: list[FeatureRow] = []
    bar_intervals = tuple(
        (bar.start_qn, bar.start_qn + bar.duration_qn) for bar in piece.bars
    )

    for track in piece.tracks:
        track_notes = notes_by_track[track.track_id]
        pitches = [note.pitch for note in track_notes]
        durations = [_float_time(note.duration_qn) for note in track_notes]
        velocities = [note.velocity for note in track_notes if note.velocity is not None]
        onset_counts: dict[RationalTime, int] = defaultdict(int)
        for note in track_notes:
            onset_counts[note.onset_qn] += 1
        polyphonic_onsets = sum(count > 1 for count in onset_counts.values())
        active_bars = 0
        for bar_start, bar_end in bar_intervals:
            if any(
                (
                    note.duration_qn.num > 0
                    and _interval_overlaps(
                        note.onset_qn, _offset(note), bar_start, bar_end
                    )
                )
                or (
                    note.duration_qn.num == 0
                    and bar_start <= note.onset_qn < bar_end
                )
                for note in track_notes
            ):
                active_bars += 1

        mean_pitch = fmean(pitches) if pitches else 0.0
        pitch_std = (
            sqrt(fmean((pitch - mean_pitch) ** 2 for pitch in pitches))
            if pitches
            else 0.0
        )
        rows.append(
            {
                "program": (track.program or 0, track.program is not None),
                "channel": (track.channel or 0, track.channel is not None),
                "is_percussion": (int(track.is_percussion), True),
                "source_track_index": (
                    track.source_track_index or 0,
                    track.source_track_index is not None,
                ),
                "note_count": (len(track_notes), True),
                "mean_pitch": (mean_pitch, bool(pitches)),
                "pitch_std": (pitch_std, bool(pitches)),
                "min_pitch": (min(pitches) if pitches else 0, bool(pitches)),
                "max_pitch": (max(pitches) if pitches else 0, bool(pitches)),
                "note_density": (
                    len(track_notes) / duration if duration > 0 else 0.0,
                    duration > 0,
                ),
                "polyphony_ratio": (
                    polyphonic_onsets / len(onset_counts) if onset_counts else 0.0,
                    bool(onset_counts),
                ),
                "active_bar_ratio": (
                    active_bars / len(piece.bars) if piece.bars else 0.0,
                    bool(piece.bars),
                ),
                "mean_duration_qn": (
                    fmean(durations) if durations else 0.0,
                    bool(durations),
                ),
                "mean_velocity": (
                    fmean(velocities) if velocities else 0.0,
                    bool(velocities),
                ),
            }
        )
    return rows


def build_raw_graph(
    piece: CanonicalPiece,
    registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
    *,
    raw_only: bool = True,
) -> HeteroData:
    """Build the Phase 3A raw-only graph without reading supervisory fields.

    Targets, annotations, provenance, source paths, source groups, dataset names,
    and split values are intentionally never inspected while constructing node
    features or relations.
    """

    if not raw_only:
        raise GraphBuildError("Phase 3A implements only raw_only=True graphs")
    if piece.schema_version != SCHEMA_VERSION:
        raise GraphBuildError(
            f"unsupported canonical schema {piece.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if any(not spec.raw_inference_safe for spec in registry.specs):
        raise GraphBuildError("feature registry contains a raw-inference-unsafe field")

    track_index = {track.track_id: index for index, track in enumerate(piece.tracks)}
    bar_index = {bar.bar_id: index for index, bar in enumerate(piece.bars)}
    beat_index = {beat.beat_id: index for index, beat in enumerate(piece.beats)}
    if len(track_index) != len(piece.tracks):
        raise GraphBuildError("duplicate track IDs")
    if len(bar_index) != len(piece.bars) or len(beat_index) != len(piece.beats):
        raise GraphBuildError("duplicate bar or beat IDs")

    notes_by_track: dict[str, list[object]] = {
        track.track_id: [] for track in piece.tracks
    }
    for note in piece.notes:
        if note.track_id not in notes_by_track:
            raise GraphBuildError(f"note {note.note_id} references an unknown track")
        notes_by_track[note.track_id].append(note)

    onset_times = tuple(sorted({note.onset_qn for note in piece.notes}))
    onset_index = {time: index for index, time in enumerate(onset_times)}
    notes_at_onset: dict[RationalTime, list[int]] = defaultdict(list)
    for index, note in enumerate(piece.notes):
        notes_at_onset[note.onset_qn].append(index)

    bar_intervals = tuple((bar.start_qn, bar.duration_qn) for bar in piece.bars)
    beat_intervals = tuple((beat.start_qn, beat.duration_qn) for beat in piece.beats)
    onset_bar_owner = {
        time: _owner_index(time, bar_intervals) for time in onset_times
    }
    onset_beat_owner = {
        time: _owner_index(time, beat_intervals) for time in onset_times
    }
    note_bar_owner = [
        _owner_index(note.onset_qn, bar_intervals) for note in piece.notes
    ]

    meters = _meter_values(piece)
    for bar in piece.bars:
        if bar.meter_event_id not in meters:
            raise GraphBuildError(f"bar {bar.bar_id} references an unknown meter")
    for beat in piece.beats:
        if beat.bar_id not in bar_index:
            raise GraphBuildError(f"beat {beat.beat_id} references an unknown bar")
        if beat.meter_event_id not in meters:
            raise GraphBuildError(f"beat {beat.beat_id} references an unknown meter")

    data = HeteroData()
    data.schema_version = piece.schema_version
    data.graph_schema_version = GRAPH_SCHEMA_VERSION
    data.feature_registry_version = registry.version
    data.graph_builder_version = GRAPH_BUILDER_VERSION
    data.raw_only = True

    tempo_values = [event.microseconds_per_quarter for event in piece.tempo_events]
    song_rows: list[FeatureRow] = [
        {
            "duration_qn": (_float_time(piece.duration_qn), True),
            "track_count": (len(piece.tracks), True),
            "bar_count": (len(piece.bars), True),
            "beat_count": (len(piece.beats), True),
            "onset_count": (len(onset_times), True),
            "note_count": (len(piece.notes), True),
            "tempo_mean_us_per_qn": (
                fmean(tempo_values) if tempo_values else 0.0,
                bool(tempo_values),
            ),
            "tempo_min_us_per_qn": (
                min(tempo_values) if tempo_values else 0,
                bool(tempo_values),
            ),
            "tempo_max_us_per_qn": (
                max(tempo_values) if tempo_values else 0,
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
        _track_feature_rows(piece, notes_by_track),
        tuple(track.track_id for track in piece.tracks),
        registry,
    )

    bar_rows: list[FeatureRow] = []
    for index, bar in enumerate(piece.bars):
        numerator, denominator = meters[bar.meter_event_id]
        denominator_log2 = denominator.bit_length() - 1
        bar_end = bar.start_qn + bar.duration_qn
        assigned_onsets = [
            time for time in onset_times if onset_bar_owner[time] == index
        ]
        starting_notes = [
            note for note, owner in zip(piece.notes, note_bar_owner) if owner == index
        ]
        active_notes = [
            note
            for note in piece.notes
            if (
                note.duration_qn.num > 0
                and _interval_overlaps(
                    note.onset_qn, _offset(note), bar.start_qn, bar_end
                )
            )
            or (
                note.duration_qn.num == 0
                and bar.start_qn <= note.onset_qn < bar_end
            )
        ]
        bar_rows.append(
            {
                "meter_numerator": (
                    _bounded_category(numerator, 256, 255),
                    True,
                ),
                "meter_denominator_log2": (
                    _bounded_category(denominator_log2, 128, 127),
                    True,
                ),
                "is_pickup": (int(bar.is_pickup), True),
                "is_incomplete": (int(bar.is_incomplete), True),
                "index": (bar.index, True),
                "start_qn": (_float_time(bar.start_qn), True),
                "duration_qn": (_float_time(bar.duration_qn), True),
                "metric_offset_qn": (_float_time(bar.metric_offset_qn), True),
                "tempo_us_per_qn": (_tempo_at(piece, bar.start_qn), True),
                "starting_note_count": (len(starting_notes), True),
                "active_note_count": (len(active_notes), True),
                "onset_count": (len(assigned_onsets), True),
                "active_track_count": (
                    len({note.track_id for note in active_notes}),
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
    for index, beat in enumerate(piece.beats):
        numerator, denominator = meters[beat.meter_event_id]
        denominator_log2 = denominator.bit_length() - 1
        assigned_onsets = [
            time for time in onset_times if onset_beat_owner[time] == index
        ]
        starting_notes = [
            note for note in piece.notes if onset_beat_owner[note.onset_qn] == index
        ]
        active_notes = [
            note
            for note in piece.notes
            if note.duration_qn.num > 0
            and note.onset_qn <= beat.start_qn < _offset(note)
        ]
        beat_rows.append(
            {
                "meter_numerator": (
                    _bounded_category(numerator, 256, 255),
                    True,
                ),
                "meter_denominator_log2": (
                    _bounded_category(denominator_log2, 128, 127),
                    True,
                ),
                "is_downbeat": (int(beat.is_downbeat), True),
                "index_in_bar": (beat.index_in_bar, True),
                "start_qn": (_float_time(beat.start_qn), True),
                "duration_qn": (_float_time(beat.duration_qn), True),
                "position_in_bar_qn": (
                    _float_time(beat.position_in_bar_qn),
                    True,
                ),
                "strength": (beat.strength or 0.0, beat.strength is not None),
                "tempo_us_per_qn": (_tempo_at(piece, beat.start_qn), True),
                "starting_note_count": (len(starting_notes), True),
                "active_note_count": (len(active_notes), True),
                "active_track_count": (
                    len({note.track_id for note in active_notes}),
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
    onsets_per_beat: dict[int, int] = defaultdict(int)
    for owner in onset_beat_owner.values():
        if owner is not None:
            onsets_per_beat[owner] += 1
    for time in onset_times:
        owner_bar = onset_bar_owner[time]
        position = (
            time - piece.bars[owner_bar].start_qn
            if owner_bar is not None
            else RationalTime(0)
        )
        active_notes = [
            note
            for note in piece.notes
            if note.duration_qn.num > 0 and note.onset_qn <= time < _offset(note)
        ]
        beat_owner = onset_beat_owner[time]
        onset_rows.append(
            {
                "start_qn": (_float_time(time), True),
                "position_in_bar_qn": (
                    _float_time(position),
                    owner_bar is not None,
                ),
                "starting_note_count": (len(notes_at_onset[time]), True),
                "active_note_count": (len(active_notes), True),
                "active_track_count": (
                    len({note.track_id for note in active_notes}),
                    True,
                ),
                "onsets_in_beat": (
                    onsets_per_beat[beat_owner] if beat_owner is not None else 0,
                    beat_owner is not None,
                ),
            }
        )
    onset_ids = tuple(f"onset:{time.num}_{time.den}" for time in onset_times)
    _pack_features(data, "onset", onset_rows, onset_ids, registry)
    data["onset"].candidate_slot = torch.ones(len(onset_times), dtype=torch.bool)

    track_pitch_stats: dict[str, tuple[float, float]] = {}
    for track_id, track_notes in notes_by_track.items():
        pitches = [note.pitch for note in track_notes]
        mean = fmean(pitches) if pitches else 0.0
        std = sqrt(fmean((pitch - mean) ** 2 for pitch in pitches)) if pitches else 0.0
        track_pitch_stats[track_id] = (mean, std)

    note_rows: list[FeatureRow] = []
    for index, note in enumerate(piece.notes):
        owner_bar = note_bar_owner[index]
        position = (
            note.onset_qn - piece.bars[owner_bar].start_qn
            if owner_bar is not None
            else RationalTime(0)
        )
        mean_pitch, pitch_std = track_pitch_stats[note.track_id]
        note_rows.append(
            {
                "pitch": (note.pitch, True),
                "pitch_class": (note.pitch % 12, True),
                "octave": (note.pitch // 12, True),
                "program": (note.program or 0, note.program is not None),
                "channel": (note.channel or 0, note.channel is not None),
                "is_percussion": (int(note.is_percussion), True),
                "is_grace": (int(note.is_grace), True),
                "onset_qn": (_float_time(note.onset_qn), True),
                "duration_qn": (_float_time(note.duration_qn), True),
                "velocity": (note.velocity or 0, note.velocity is not None),
                "position_in_bar_qn": (
                    _float_time(position),
                    owner_bar is not None,
                ),
                "track_relative_pitch": (
                    (note.pitch - mean_pitch) / pitch_std if pitch_std > 0 else 0.0,
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

    edge_pairs: dict[tuple[str, str, str], list[EdgePair]] = {
        edge_type: [] for edge_type in MANDATORY_EDGE_TYPES
    }
    _add_forward_and_reverse(
        edge_pairs,
        ("song", "contains_track", "track"),
        ((0, index) for index in range(len(piece.tracks))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("song", "contains_bar", "bar"),
        ((0, index) for index in range(len(piece.bars))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("track", "contains_note", "note"),
        ((track_index[note.track_id], index) for index, note in enumerate(piece.notes)),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_beat", "beat"),
        ((bar_index[beat.bar_id], index) for index, beat in enumerate(piece.beats)),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_onset", "onset"),
        (
            (owner, onset_index[time])
            for time, owner in onset_bar_owner.items()
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "contains_note", "note"),
        (
            (owner, index)
            for index, owner in enumerate(note_bar_owner)
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("beat", "contains_onset", "onset"),
        (
            (owner, onset_index[time])
            for time, owner in onset_beat_owner.items()
            if owner is not None
        ),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("onset", "starts_note", "note"),
        (
            (onset_index[note.onset_qn], index)
            for index, note in enumerate(piece.notes)
        ),
    )

    _add_forward_and_reverse(
        edge_pairs,
        ("bar", "next_bar", "bar"),
        ((index, index + 1) for index in range(max(0, len(piece.bars) - 1))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("beat", "next_beat", "beat"),
        ((index, index + 1) for index in range(max(0, len(piece.beats) - 1))),
    )
    _add_forward_and_reverse(
        edge_pairs,
        ("onset", "next_onset", "onset"),
        ((index, index + 1) for index in range(max(0, len(onset_times) - 1))),
    )
    for track in piece.tracks:
        indices = [
            index for index, note in enumerate(piece.notes) if note.track_id == track.track_id
        ]
        _add_forward_and_reverse(
            edge_pairs,
            ("note", "next_in_track", "note"),
            zip(indices, indices[1:]),
        )

    sustained_pairs: list[EdgePair] = []
    for note_index, note in enumerate(piece.notes):
        if note.duration_qn.num == 0:
            continue
        note_end = _offset(note)
        for candidate_index, beat in enumerate(piece.beats):
            if note.onset_qn <= beat.start_qn < note_end:
                sustained_pairs.append((note_index, candidate_index))
    _add_forward_and_reverse(
        edge_pairs,
        ("note", "active_at", "beat"),
        sustained_pairs,
    )

    for edge_type in MANDATORY_EDGE_TYPES:
        data[edge_type].edge_index = _edge_tensor(edge_pairs[edge_type])

    from music_critic.graph.validation import validate_raw_graph

    validate_raw_graph(data, registry=registry)
    return data


__all__ = ["GraphBuildError", "build_raw_graph"]
