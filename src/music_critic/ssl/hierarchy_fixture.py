"""Deterministic Phase 8A hierarchy oracle beside the Phase 7A fixture.

The existing Phase 7A bounded fixture is an immutable compatibility control.
This module wraps it without modifying its pieces or fingerprints and appends
one target-free raw piece that supplies exact hierarchy mechanics:

* one polyphonic onset;
* one beat containing multiple onset nodes;
* one note sustained across a bar boundary;
* non-empty track/bar intersections in a multitrack, multibar sample.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Literal

from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    RationalTime,
    TempoEvent,
    dumps_piece,
    validate_piece,
)
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.bounded_fixture import (
    PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
    PHASE7A_BOUNDED_FIXTURE_POLICY,
    Phase7ABoundedFixture,
    build_phase7a_bounded_fixture,
)

if TYPE_CHECKING:
    from music_critic.ssl.data import SSLRawSample


PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION = "1.0.0"
PHASE8A_HIERARCHY_FIXTURE_POLICY = (
    "phase8a_phase7a_base_plus_hierarchy_oracle_v1"
)
PHASE8A_HIERARCHY_DATASET_ID = "phase8a-bounded"
PHASE8A_HIERARCHY_ORACLE_PIECE_ID = "piece:phase8a-hierarchy-oracle"
PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID = (
    "group:phase8a-hierarchy-oracle"
)
PHASE8A_BOUND_PHASE7A_FIXTURE_FINGERPRINT = (
    "9f959d91d6805101983711511abcf89450e24b1886417632ea37fd0dc96ba922"
)

FixtureSplit = Literal["train", "validation"]
EdgeType = tuple[str, str, str]
EdgePair = tuple[int, int]

ONSET_STARTS_NOTE_EDGE: EdgeType = ("onset", "starts_note", "note")
BEAT_CONTAINS_ONSET_EDGE: EdgeType = ("beat", "contains_onset", "onset")
BAR_CONTAINS_ONSET_EDGE: EdgeType = ("bar", "contains_onset", "onset")
BAR_CONTAINS_NOTE_EDGE: EdgeType = ("bar", "contains_note", "note")
TRACK_CONTAINS_NOTE_EDGE: EdgeType = ("track", "contains_note", "note")
NOTE_ACTIVE_AT_BEAT_EDGE: EdgeType = ("note", "active_at", "beat")

PHASE8A_ORACLE_ONSET_STARTS_NOTE: tuple[EdgePair, ...] = (
    (0, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (5, 8),
)
PHASE8A_ORACLE_BEAT_CONTAINS_ONSET: tuple[EdgePair, ...] = (
    (0, 0),
    (0, 1),
    (3, 2),
    (4, 3),
    (6, 4),
    (8, 5),
)
PHASE8A_ORACLE_BAR_CONTAINS_ONSET: tuple[EdgePair, ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (2, 5),
)
PHASE8A_ORACLE_BAR_CONTAINS_NOTE: tuple[EdgePair, ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 7),
    (2, 8),
)
PHASE8A_ORACLE_TRACK_CONTAINS_NOTE: tuple[EdgePair, ...] = (
    (0, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 5),
    (1, 6),
    (0, 7),
    (1, 8),
)
PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT: tuple[EdgePair, ...] = (
    (0, 0),
    (1, 0),
    (3, 3),
    (3, 4),
    (4, 4),
    (5, 4),
    (6, 6),
    (7, 8),
    (8, 8),
)


class Phase8AHierarchyFixtureError(ValueError):
    """Raised when the bounded hierarchy evidence contract is violated."""


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _piece_fingerprint(piece: CanonicalPiece) -> str:
    return sha256(dumps_piece(piece).encode("utf-8")).hexdigest()


def _edge_pairs(graph: Any, edge_type: EdgeType) -> tuple[EdgePair, ...]:
    return tuple(
        (int(source), int(destination))
        for source, destination in graph[edge_type].edge_index.t().tolist()
    )


@dataclass(frozen=True, slots=True)
class Phase8AHierarchyPolicyOracle:
    """One exact policy-specific descendant/collateral expectation."""

    policy: str
    selected_unit_node_type: str
    selected_local_unit_indices: tuple[int, ...]
    selected_local_onset_indices: tuple[int, ...]
    selected_local_note_descendants: tuple[int, ...]
    selected_local_track_index: int | None
    span_start_bar_index: int | None
    span_end_bar_index: int | None
    primary_masked_count: int
    visible_pitched_note_count: int
    collateral_peer_note_indices: tuple[int, ...]
    collateral_owner_track_indices: tuple[int, ...]
    realized_mask_fraction: tuple[int, int]

    def __post_init__(self) -> None:
        for name, values in (
            ("selected_local_unit_indices", self.selected_local_unit_indices),
            ("selected_local_onset_indices", self.selected_local_onset_indices),
            (
                "selected_local_note_descendants",
                self.selected_local_note_descendants,
            ),
            (
                "collateral_peer_note_indices",
                self.collateral_peer_note_indices,
            ),
            (
                "collateral_owner_track_indices",
                self.collateral_owner_track_indices,
            ),
        ):
            if values != tuple(sorted(set(values))):
                raise Phase8AHierarchyFixtureError(
                    f"phase8a.fixture.oracle_{name}_invalid"
                )
        if self.primary_masked_count != len(
            self.selected_local_note_descendants
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_primary_count_invalid"
            )
        numerator, denominator = self.realized_mask_fraction
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or numerator < 0
            or denominator <= 0
            or numerator > denominator
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_realized_fraction_invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "selected_unit_node_type": self.selected_unit_node_type,
            "selected_local_unit_indices": list(
                self.selected_local_unit_indices
            ),
            "selected_local_onset_indices": list(
                self.selected_local_onset_indices
            ),
            "selected_local_note_descendants": list(
                self.selected_local_note_descendants
            ),
            "selected_local_track_index": self.selected_local_track_index,
            "span_start_bar_index": self.span_start_bar_index,
            "span_end_bar_index": self.span_end_bar_index,
            "primary_masked_count": self.primary_masked_count,
            "visible_pitched_note_count": self.visible_pitched_note_count,
            "collateral_peer_note_indices": list(
                self.collateral_peer_note_indices
            ),
            "collateral_owner_track_indices": list(
                self.collateral_owner_track_indices
            ),
            "realized_mask_fraction": list(self.realized_mask_fraction),
        }


PHASE8A_HIERARCHY_POLICY_ORACLES = (
    Phase8AHierarchyPolicyOracle(
        policy="onset_pitch_descendants",
        selected_unit_node_type="onset",
        selected_local_unit_indices=(0,),
        selected_local_onset_indices=(0,),
        selected_local_note_descendants=(0, 1),
        selected_local_track_index=None,
        span_start_bar_index=None,
        span_end_bar_index=None,
        primary_masked_count=2,
        visible_pitched_note_count=7,
        collateral_peer_note_indices=(2, 3, 4, 5, 6, 7, 8),
        collateral_owner_track_indices=(0, 1),
        realized_mask_fraction=(2, 9),
    ),
    Phase8AHierarchyPolicyOracle(
        policy="beat_pitch_descendants",
        selected_unit_node_type="beat",
        selected_local_unit_indices=(0,),
        selected_local_onset_indices=(0, 1),
        selected_local_note_descendants=(0, 1, 2),
        selected_local_track_index=None,
        span_start_bar_index=None,
        span_end_bar_index=None,
        primary_masked_count=3,
        visible_pitched_note_count=6,
        collateral_peer_note_indices=(3, 4, 5, 6, 7, 8),
        collateral_owner_track_indices=(0, 1),
        realized_mask_fraction=(1, 3),
    ),
    Phase8AHierarchyPolicyOracle(
        policy="contiguous_bar_pitch_span",
        selected_unit_node_type="bar",
        selected_local_unit_indices=(1,),
        selected_local_onset_indices=(3, 4),
        selected_local_note_descendants=(4, 5, 6),
        selected_local_track_index=None,
        span_start_bar_index=1,
        span_end_bar_index=1,
        primary_masked_count=3,
        visible_pitched_note_count=6,
        collateral_peer_note_indices=(0, 1, 2, 3, 7, 8),
        collateral_owner_track_indices=(0, 1),
        realized_mask_fraction=(1, 3),
    ),
    Phase8AHierarchyPolicyOracle(
        policy="track_bar_pitch_span",
        selected_unit_node_type="bar",
        selected_local_unit_indices=(1,),
        selected_local_onset_indices=(3, 4),
        selected_local_note_descendants=(5, 6),
        selected_local_track_index=1,
        span_start_bar_index=1,
        span_end_bar_index=1,
        primary_masked_count=2,
        visible_pitched_note_count=7,
        collateral_peer_note_indices=(1, 8),
        collateral_owner_track_indices=(1,),
        realized_mask_fraction=(2, 9),
    ),
)


@dataclass(frozen=True, slots=True)
class Phase8AHierarchyOracleComposition:
    """Portable exact evidence for the supplemental hierarchy piece."""

    dataset_id: str
    piece_id: str
    source_group_id: str
    track_count: int
    bar_count: int
    beat_count: int
    onset_count: int
    note_count: int
    onset_starts_note: tuple[EdgePair, ...]
    beat_contains_onset: tuple[EdgePair, ...]
    bar_contains_onset: tuple[EdgePair, ...]
    bar_contains_note: tuple[EdgePair, ...]
    track_contains_note: tuple[EdgePair, ...]
    note_active_at_beat: tuple[EdgePair, ...]
    policy_oracles: tuple[Phase8AHierarchyPolicyOracle, ...]
    canonical_piece_fingerprint: str
    raw_graph_fingerprint: str
    fingerprint: str

    def payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "piece_id": self.piece_id,
            "source_group_id": self.source_group_id,
            "track_count": self.track_count,
            "bar_count": self.bar_count,
            "beat_count": self.beat_count,
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "relations": {
                "onset_starts_note": [
                    list(pair) for pair in self.onset_starts_note
                ],
                "beat_contains_onset": [
                    list(pair) for pair in self.beat_contains_onset
                ],
                "bar_contains_onset": [
                    list(pair) for pair in self.bar_contains_onset
                ],
                "bar_contains_note": [
                    list(pair) for pair in self.bar_contains_note
                ],
                "track_contains_note": [
                    list(pair) for pair in self.track_contains_note
                ],
                "note_active_at_beat": [
                    list(pair) for pair in self.note_active_at_beat
                ],
            },
            "policy_oracles": [
                oracle.to_dict() for oracle in self.policy_oracles
            ],
            "canonical_piece_fingerprint": (
                self.canonical_piece_fingerprint
            ),
            "raw_graph_fingerprint": self.raw_graph_fingerprint,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "fingerprint": self.fingerprint}

    def __post_init__(self) -> None:
        if self.dataset_id != PHASE8A_HIERARCHY_DATASET_ID:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_dataset_id_incompatible"
            )
        if self.piece_id != PHASE8A_HIERARCHY_ORACLE_PIECE_ID:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_piece_id_incompatible"
            )
        if self.source_group_id != (
            PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_source_group_incompatible"
            )
        if (
            self.track_count,
            self.bar_count,
            self.beat_count,
            self.onset_count,
            self.note_count,
        ) != (2, 3, 12, 6, 9):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_counts_incompatible"
            )
        expected_relations = (
            PHASE8A_ORACLE_ONSET_STARTS_NOTE,
            PHASE8A_ORACLE_BEAT_CONTAINS_ONSET,
            PHASE8A_ORACLE_BAR_CONTAINS_ONSET,
            PHASE8A_ORACLE_BAR_CONTAINS_NOTE,
            PHASE8A_ORACLE_TRACK_CONTAINS_NOTE,
            PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT,
        )
        actual_relations = (
            self.onset_starts_note,
            self.beat_contains_onset,
            self.bar_contains_onset,
            self.bar_contains_note,
            self.track_contains_note,
            self.note_active_at_beat,
        )
        if actual_relations != expected_relations:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_relations_incompatible"
            )
        if self.policy_oracles != PHASE8A_HIERARCHY_POLICY_ORACLES:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_policy_evidence_incompatible"
            )
        if self.fingerprint != _canonical_fingerprint(self.payload()):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_fingerprint_invalid"
            )


@dataclass(frozen=True, slots=True)
class _OracleNoteSpec:
    track_index: int
    pitch: int
    onset_qn: RationalTime
    duration_qn: RationalTime
    velocity: int


_ORACLE_NOTE_SPECS = (
    _OracleNoteSpec(0, 60, RationalTime(0), RationalTime(1), 72),
    _OracleNoteSpec(1, 72, RationalTime(0), RationalTime(1), 76),
    _OracleNoteSpec(
        0,
        64,
        RationalTime(1, 2),
        RationalTime(1, 2),
        80,
    ),
    _OracleNoteSpec(0, 67, RationalTime(3), RationalTime(2), 84),
    _OracleNoteSpec(0, 62, RationalTime(4), RationalTime(1), 74),
    _OracleNoteSpec(1, 74, RationalTime(4), RationalTime(1), 78),
    _OracleNoteSpec(1, 76, RationalTime(6), RationalTime(1), 82),
    _OracleNoteSpec(0, 65, RationalTime(8), RationalTime(1), 86),
    _OracleNoteSpec(1, 77, RationalTime(8), RationalTime(1), 88),
)


def build_phase8a_hierarchy_oracle_piece() -> CanonicalPiece:
    """Build the exact target-free supplemental hierarchy oracle."""

    token = "phase8a-hierarchy-oracle"
    provenance_id = f"prov:{token}.source"
    meter_id = f"meter:{token}.000"
    tracks = (
        CanonicalTrack(
            track_id=f"track:{token}.00",
            source_track_index=0,
            name="Oracle track 1",
            instrument_name=None,
            program=0,
            channel=0,
            is_percussion=False,
            provenance_id=provenance_id,
        ),
        CanonicalTrack(
            track_id=f"track:{token}.01",
            source_track_index=1,
            name="Oracle track 2",
            instrument_name=None,
            program=40,
            channel=1,
            is_percussion=False,
            provenance_id=provenance_id,
        ),
    )
    bars = tuple(
        CanonicalBar(
            bar_id=f"bar:{token}.{bar_index:03d}",
            index=bar_index,
            start_qn=RationalTime(4 * bar_index),
            duration_qn=RationalTime(4),
            meter_event_id=meter_id,
            metric_offset_qn=RationalTime(0),
            is_pickup=False,
            is_incomplete=False,
            display_number=str(bar_index + 1),
            provenance_id=provenance_id,
        )
        for bar_index in range(3)
    )
    beats = tuple(
        CanonicalBeat(
            beat_id=f"beat:{token}.{bar_index:03d}.{beat_index}",
            bar_id=f"bar:{token}.{bar_index:03d}",
            meter_event_id=meter_id,
            index_in_bar=beat_index,
            start_qn=RationalTime(4 * bar_index + beat_index),
            duration_qn=RationalTime(1),
            position_in_bar_qn=RationalTime(beat_index),
            is_downbeat=beat_index == 0,
            strength=(
                1.0
                if beat_index == 0
                else 0.75
                if beat_index == 2
                else 0.5
            ),
            provenance_id=provenance_id,
        )
        for bar_index in range(3)
        for beat_index in range(4)
    )
    notes = tuple(
        CanonicalNote(
            note_id=f"note:{token}.{note_index:03d}",
            track_id=tracks[spec.track_index].track_id,
            pitch=spec.pitch,
            onset_qn=spec.onset_qn,
            duration_qn=spec.duration_qn,
            velocity=spec.velocity,
            channel=tracks[spec.track_index].channel,
            program=tracks[spec.track_index].program,
            is_percussion=False,
            is_grace=False,
            spelling_step=None,
            spelling_alter=None,
            staff=None,
            voice=None,
            articulations=(),
            dynamic=None,
            source_onset_ticks=(
                spec.onset_qn.num * 480 // spec.onset_qn.den
            ),
            source_duration_ticks=(
                spec.duration_qn.num * 480 // spec.duration_qn.den
            ),
            source_onset_seconds=None,
            source_duration_seconds=None,
            provenance_id=provenance_id,
        )
        for note_index, spec in enumerate(_ORACLE_NOTE_SPECS)
    )
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=PHASE8A_HIERARCHY_ORACLE_PIECE_ID,
        dataset_name=PHASE8A_HIERARCHY_DATASET_ID,
        source_group_id=PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID,
        split="train",
        source_path=None,
        source_resolution=480,
        duration_qn=RationalTime(12),
        metadata=PieceMetadata(
            source_format="synthetic",
            title="Phase 8A hierarchy oracle",
            creators=("Music Critic V2",),
            collection="Phase 8A deterministic bounded evidence",
            movement_title=None,
            movement_number=None,
            genres=(),
            copyright=None,
            language=None,
        ),
        tracks=tracks,
        notes=notes,
        bars=bars,
        beats=beats,
        tempo_events=(
            TempoEvent(
                tempo_event_id=f"tempo:{token}.000",
                onset_qn=RationalTime(0),
                microseconds_per_quarter=500_000,
                provenance_id=provenance_id,
            ),
        ),
        meter_events=(
            MeterEvent(
                meter_event_id=meter_id,
                onset_qn=RationalTime(0),
                numerator=4,
                denominator=4,
                provenance_id=provenance_id,
            ),
        ),
        key_signature_events=(),
        annotations=(),
        targets=(),
        provenance=(
            ProvenanceRecord(
                provenance_id=provenance_id,
                kind="synthetic",
                source=PHASE8A_HIERARCHY_FIXTURE_POLICY,
                record_id=token,
                uri=None,
                version=PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
                checksum_sha256=None,
                created_at="2026-07-30T00:00:00+03:00",
                parents=(),
                details=(
                    ("purpose", "hierarchy_masking_oracle"),
                    ("split", "train"),
                ),
            ),
        ),
        quality_flags=(),
    )
    report = validate_piece(piece)
    if report.errors:
        errors = ",".join(
            f"{issue.code}@{issue.path}" for issue in report.errors
        )
        raise Phase8AHierarchyFixtureError(
            f"phase8a.fixture.oracle_piece_invalid:{errors}"
        )
    return piece


def _oracle_composition(
    piece: CanonicalPiece,
) -> Phase8AHierarchyOracleComposition:
    graph = build_raw_graph(piece, assume_valid=True)
    payload = {
        "dataset_id": piece.dataset_name,
        "piece_id": piece.piece_id,
        "source_group_id": piece.source_group_id,
        "track_count": len(piece.tracks),
        "bar_count": len(piece.bars),
        "beat_count": len(piece.beats),
        "onset_count": int(graph["onset"].num_nodes),
        "note_count": len(piece.notes),
        "onset_starts_note": _edge_pairs(
            graph,
            ONSET_STARTS_NOTE_EDGE,
        ),
        "beat_contains_onset": _edge_pairs(
            graph,
            BEAT_CONTAINS_ONSET_EDGE,
        ),
        "bar_contains_onset": _edge_pairs(
            graph,
            BAR_CONTAINS_ONSET_EDGE,
        ),
        "bar_contains_note": _edge_pairs(
            graph,
            BAR_CONTAINS_NOTE_EDGE,
        ),
        "track_contains_note": _edge_pairs(
            graph,
            TRACK_CONTAINS_NOTE_EDGE,
        ),
        "note_active_at_beat": _edge_pairs(
            graph,
            NOTE_ACTIVE_AT_BEAT_EDGE,
        ),
        "policy_oracles": PHASE8A_HIERARCHY_POLICY_ORACLES,
        "canonical_piece_fingerprint": _piece_fingerprint(piece),
        "raw_graph_fingerprint": graph_fingerprint(graph),
    }
    provisional = Phase8AHierarchyOracleComposition(
        **payload,
        fingerprint=_canonical_fingerprint(
            {
                "dataset_id": payload["dataset_id"],
                "piece_id": payload["piece_id"],
                "source_group_id": payload["source_group_id"],
                "track_count": payload["track_count"],
                "bar_count": payload["bar_count"],
                "beat_count": payload["beat_count"],
                "onset_count": payload["onset_count"],
                "note_count": payload["note_count"],
                "relations": {
                    "onset_starts_note": [
                        list(pair)
                        for pair in payload["onset_starts_note"]
                    ],
                    "beat_contains_onset": [
                        list(pair)
                        for pair in payload["beat_contains_onset"]
                    ],
                    "bar_contains_onset": [
                        list(pair)
                        for pair in payload["bar_contains_onset"]
                    ],
                    "bar_contains_note": [
                        list(pair)
                        for pair in payload["bar_contains_note"]
                    ],
                    "track_contains_note": [
                        list(pair)
                        for pair in payload["track_contains_note"]
                    ],
                    "note_active_at_beat": [
                        list(pair)
                        for pair in payload["note_active_at_beat"]
                    ],
                },
                "policy_oracles": [
                    oracle.to_dict()
                    for oracle in PHASE8A_HIERARCHY_POLICY_ORACLES
                ],
                "canonical_piece_fingerprint": payload[
                    "canonical_piece_fingerprint"
                ],
                "raw_graph_fingerprint": payload[
                    "raw_graph_fingerprint"
                ],
            }
        ),
    )
    return provisional


def _count_summary(
    pieces: tuple[CanonicalPiece, ...],
    *,
    split: str,
) -> dict[str, object]:
    track_count = 0
    bar_count = 0
    beat_count = 0
    onset_count = 0
    note_count = 0
    track_bar_cell_count = 0
    nonempty_track_bar_cell_count = 0
    polyphonic_onset_count = 0
    multi_onset_beat_count = 0
    cross_bar_sustained_note_count = 0

    for piece in pieces:
        graph = build_raw_graph(piece, assume_valid=True)
        track_count += len(piece.tracks)
        bar_count += len(piece.bars)
        beat_count += len(piece.beats)
        onset_count += int(graph["onset"].num_nodes)
        note_count += len(piece.notes)
        track_bar_cell_count += len(piece.tracks) * len(piece.bars)

        onset_sizes = Counter(
            source
            for source, _ in _edge_pairs(
                graph,
                ONSET_STARTS_NOTE_EDGE,
            )
        )
        beat_sizes = Counter(
            source
            for source, _ in _edge_pairs(
                graph,
                BEAT_CONTAINS_ONSET_EDGE,
            )
        )
        polyphonic_onset_count += sum(
            size > 1 for size in onset_sizes.values()
        )
        multi_onset_beat_count += sum(
            size > 1 for size in beat_sizes.values()
        )

        notes_by_bar = {
            bar: set() for bar in range(len(piece.bars))
        }
        for bar, note in _edge_pairs(graph, BAR_CONTAINS_NOTE_EDGE):
            notes_by_bar[bar].add(note)
        notes_by_track = {
            track: set() for track in range(len(piece.tracks))
        }
        for track, note in _edge_pairs(graph, TRACK_CONTAINS_NOTE_EDGE):
            notes_by_track[track].add(note)
        nonempty_track_bar_cell_count += sum(
            bool(notes_by_track[track] & notes_by_bar[bar])
            for track in notes_by_track
            for bar in notes_by_bar
        )

        boundary_starts = tuple(bar.start_qn for bar in piece.bars[1:])
        cross_bar_sustained_note_count += sum(
            any(
                note.onset_qn
                < boundary
                < note.onset_qn + note.duration_qn
                for boundary in boundary_starts
            )
            for note in piece.notes
        )

    return {
        "split": split,
        "piece_count": len(pieces),
        "track_count": track_count,
        "bar_count": bar_count,
        "beat_count": beat_count,
        "onset_count": onset_count,
        "note_count": note_count,
        "track_bar_cell_count": track_bar_cell_count,
        "nonempty_track_bar_cell_count": (
            nonempty_track_bar_cell_count
        ),
        "polyphonic_onset_count": polyphonic_onset_count,
        "multi_onset_beat_count": multi_onset_beat_count,
        "cross_bar_sustained_note_count": (
            cross_bar_sustained_note_count
        ),
    }


_EXPECTED_COUNT_SUMMARIES = {
    "all": {
        "split": "all",
        "piece_count": 6,
        "track_count": 14,
        "bar_count": 15,
        "beat_count": 60,
        "onset_count": 42,
        "note_count": 93,
        "track_bar_cell_count": 34,
        "nonempty_track_bar_cell_count": 34,
        "polyphonic_onset_count": 39,
        "multi_onset_beat_count": 1,
        "cross_bar_sustained_note_count": 1,
    },
    "train": {
        "split": "train",
        "piece_count": 4,
        "track_count": 9,
        "bar_count": 10,
        "beat_count": 40,
        "onset_count": 27,
        "note_count": 57,
        "track_bar_cell_count": 22,
        "nonempty_track_bar_cell_count": 22,
        "polyphonic_onset_count": 24,
        "multi_onset_beat_count": 1,
        "cross_bar_sustained_note_count": 1,
    },
    "validation": {
        "split": "validation",
        "piece_count": 2,
        "track_count": 5,
        "bar_count": 5,
        "beat_count": 20,
        "onset_count": 15,
        "note_count": 36,
        "track_bar_cell_count": 12,
        "nonempty_track_bar_cell_count": 12,
        "polyphonic_onset_count": 15,
        "multi_onset_beat_count": 0,
        "cross_bar_sustained_note_count": 0,
    },
}


def _fixture_composition_payload(
    *,
    contract_version: str,
    policy: str,
    phase7a_fixture: Phase7ABoundedFixture,
    supplemental_piece: CanonicalPiece,
    oracle_composition: Phase8AHierarchyOracleComposition,
) -> dict[str, object]:
    train_pieces = phase7a_fixture.train_pieces + (supplemental_piece,)
    validation_pieces = phase7a_fixture.validation_pieces
    return {
        "contract_version": contract_version,
        "policy": policy,
        "phase7a_base": {
            "contract_version": phase7a_fixture.contract_version,
            "policy": phase7a_fixture.policy,
            "fingerprints": phase7a_fixture.fingerprint_bundle(),
        },
        "supplemental_oracle": oracle_composition.to_dict(),
        "split_identities": {
            "train": [
                [piece.dataset_name, piece.piece_id]
                for piece in train_pieces
            ],
            "validation": [
                [piece.dataset_name, piece.piece_id]
                for piece in validation_pieces
            ],
        },
        "counts": {
            "all": _count_summary(
                train_pieces + validation_pieces,
                split="all",
            ),
            "train": _count_summary(train_pieces, split="train"),
            "validation": _count_summary(
                validation_pieces,
                split="validation",
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class Phase8AHierarchyFixture:
    """Phase 7A compatibility base plus one Phase 8A hierarchy oracle."""

    contract_version: str
    policy: str
    phase7a_fixture: Phase7ABoundedFixture
    supplemental_piece: CanonicalPiece
    oracle_composition: Phase8AHierarchyOracleComposition
    fixture_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.contract_version_incompatible"
            )
        if self.policy != PHASE8A_HIERARCHY_FIXTURE_POLICY:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.policy_incompatible"
            )
        if (
            self.phase7a_fixture.contract_version
            != PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION
            or self.phase7a_fixture.policy
            != PHASE7A_BOUNDED_FIXTURE_POLICY
            or self.phase7a_fixture.fixture_fingerprint
            != PHASE8A_BOUND_PHASE7A_FIXTURE_FINGERPRINT
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.phase7a_base_incompatible"
            )
        if (
            self.supplemental_piece.dataset_name
            != PHASE8A_HIERARCHY_DATASET_ID
            or self.supplemental_piece.piece_id
            != PHASE8A_HIERARCHY_ORACLE_PIECE_ID
            or self.supplemental_piece.source_group_id
            != PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID
            or self.supplemental_piece.split != "train"
            or self.supplemental_piece.annotations
            or self.supplemental_piece.targets
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.supplemental_piece_incompatible"
            )
        report = validate_piece(self.supplemental_piece)
        if report.errors:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.supplemental_piece_invalid"
            )
        if _oracle_composition(self.supplemental_piece) != (
            self.oracle_composition
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.oracle_composition_invalid"
            )
        for split in ("all", "train", "validation"):
            if self.count_summary(
                None if split == "all" else split
            ) != _EXPECTED_COUNT_SUMMARIES[split]:
                raise Phase8AHierarchyFixtureError(
                    f"phase8a.fixture.{split}_counts_incompatible"
                )
        if self.fixture_fingerprint != _canonical_fingerprint(
            self.composition_payload()
        ):
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.fingerprint_invalid"
            )

    @property
    def train_pieces(self) -> tuple[CanonicalPiece, ...]:
        return self.phase7a_fixture.train_pieces + (
            self.supplemental_piece,
        )

    @property
    def validation_pieces(self) -> tuple[CanonicalPiece, ...]:
        return self.phase7a_fixture.validation_pieces

    def pieces(self, split: FixtureSplit) -> tuple[CanonicalPiece, ...]:
        if split == "train":
            return self.train_pieces
        if split == "validation":
            return self.validation_pieces
        raise Phase8AHierarchyFixtureError(
            "phase8a.fixture.split_invalid"
        )

    def identities(
        self,
        split: FixtureSplit,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (piece.dataset_name, piece.piece_id)
            for piece in self.pieces(split)
        )

    def raw_samples(
        self,
        split: FixtureSplit,
    ) -> tuple[SSLRawSample, ...]:
        from music_critic.ssl.data import SSLRawSample

        samples = []
        for piece in self.pieces(split):
            graph = build_raw_graph(piece, assume_valid=True)
            samples.append(
                SSLRawSample(
                    raw_graph=graph,
                    raw_graph_fingerprint=graph_fingerprint(graph),
                    dataset_id=piece.dataset_name,
                    piece_id=piece.piece_id,
                )
            )
        return tuple(samples)

    def count_summary(
        self,
        split: FixtureSplit | None = None,
    ) -> dict[str, object]:
        if split is None:
            pieces = self.train_pieces + self.validation_pieces
            label = "all"
        elif split in {"train", "validation"}:
            pieces = self.pieces(split)
            label = split
        else:
            raise Phase8AHierarchyFixtureError(
                "phase8a.fixture.split_invalid"
            )
        return _count_summary(pieces, split=label)

    def composition_payload(self) -> dict[str, object]:
        return _fixture_composition_payload(
            contract_version=self.contract_version,
            policy=self.policy,
            phase7a_fixture=self.phase7a_fixture,
            supplemental_piece=self.supplemental_piece,
            oracle_composition=self.oracle_composition,
        )

    def fingerprint_bundle(self) -> dict[str, str]:
        return {
            "kind": "phase8a_hierarchy_bounded",
            "hierarchy_fixture_fingerprint": self.fixture_fingerprint,
            "phase7a_base_fixture_fingerprint": (
                self.phase7a_fixture.fixture_fingerprint
            ),
            "supplemental_oracle_fingerprint": (
                self.oracle_composition.fingerprint
            ),
            "supplemental_raw_graph_fingerprint": (
                self.oracle_composition.raw_graph_fingerprint
            ),
        }


def build_phase8a_hierarchy_fixture() -> Phase8AHierarchyFixture:
    """Build the exact Phase 7A base plus Phase 8A hierarchy evidence."""

    phase7a_fixture = build_phase7a_bounded_fixture()
    supplemental_piece = build_phase8a_hierarchy_oracle_piece()
    oracle_composition = _oracle_composition(supplemental_piece)
    payload = _fixture_composition_payload(
        contract_version=PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
        policy=PHASE8A_HIERARCHY_FIXTURE_POLICY,
        phase7a_fixture=phase7a_fixture,
        supplemental_piece=supplemental_piece,
        oracle_composition=oracle_composition,
    )
    return Phase8AHierarchyFixture(
        contract_version=PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
        policy=PHASE8A_HIERARCHY_FIXTURE_POLICY,
        phase7a_fixture=phase7a_fixture,
        supplemental_piece=supplemental_piece,
        oracle_composition=oracle_composition,
        fixture_fingerprint=_canonical_fingerprint(payload),
    )


__all__ = [
    "BAR_CONTAINS_NOTE_EDGE",
    "BAR_CONTAINS_ONSET_EDGE",
    "BEAT_CONTAINS_ONSET_EDGE",
    "NOTE_ACTIVE_AT_BEAT_EDGE",
    "ONSET_STARTS_NOTE_EDGE",
    "PHASE8A_BOUND_PHASE7A_FIXTURE_FINGERPRINT",
    "PHASE8A_HIERARCHY_DATASET_ID",
    "PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION",
    "PHASE8A_HIERARCHY_FIXTURE_POLICY",
    "PHASE8A_HIERARCHY_ORACLE_PIECE_ID",
    "PHASE8A_HIERARCHY_ORACLE_SOURCE_GROUP_ID",
    "PHASE8A_HIERARCHY_POLICY_ORACLES",
    "PHASE8A_ORACLE_BAR_CONTAINS_NOTE",
    "PHASE8A_ORACLE_BAR_CONTAINS_ONSET",
    "PHASE8A_ORACLE_BEAT_CONTAINS_ONSET",
    "PHASE8A_ORACLE_NOTE_ACTIVE_AT_BEAT",
    "PHASE8A_ORACLE_ONSET_STARTS_NOTE",
    "PHASE8A_ORACLE_TRACK_CONTAINS_NOTE",
    "TRACK_CONTAINS_NOTE_EDGE",
    "Phase8AHierarchyFixture",
    "Phase8AHierarchyFixtureError",
    "Phase8AHierarchyOracleComposition",
    "Phase8AHierarchyPolicyOracle",
    "build_phase8a_hierarchy_fixture",
    "build_phase8a_hierarchy_oracle_piece",
]
