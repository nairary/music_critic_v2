"""Deterministic multi-note bounded evidence fixture for Phase 7A.

The fixture is intentionally target-free.  It provides canonical pieces as
the authoritative source and can materialize :class:`SSLRawSample` instances
through a local import, which keeps this module safe for use by
``music_critic.ssl.data`` without creating an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Literal

import torch

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
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    build_raw_graph,
    graph_fingerprint,
)

if TYPE_CHECKING:
    from music_critic.ssl.data import SSLRawSample


PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION = "1.0.0"
PHASE7A_BOUNDED_DATASET_ID = "phase7a-bounded"
PHASE7A_BOUNDED_FIXTURE_POLICY = "phase7a_multinote_raw_only_v1"
PHASE7A_PITCH_MUTATION_CONTRACT_VERSION = "1.0.0"
PHASE7A_PITCH_MUTATION_POLICY = "midi_axis_reflection_v1"

FixtureSplit = Literal["train", "validation"]


class Phase7ABoundedFixtureError(ValueError):
    """Raised when the deterministic bounded fixture contract is violated."""


def _canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT = _canonical_fingerprint(
    {
        "contract_version": PHASE7A_PITCH_MUTATION_CONTRACT_VERSION,
        "policy": PHASE7A_PITCH_MUTATION_POLICY,
        "source_domain": [0, 127],
        "transformation": "mutated_pitch=127-source_pitch",
        "rebuild": "canonical_piece_and_all_dependent_raw_features",
    }
)


def _piece_fingerprint(piece: CanonicalPiece) -> str:
    return sha256(dumps_piece(piece).encode("utf-8")).hexdigest()


def _time_payload(value: RationalTime) -> tuple[int, int]:
    return (value.num, value.den)


@dataclass(frozen=True, slots=True)
class Phase7APieceComposition:
    """Exact raw composition metadata for one bounded fixture piece."""

    split: FixtureSplit
    dataset_id: str
    piece_id: str
    source_group_id: str
    track_count: int
    bar_count: int
    beat_count: int
    onset_count: int
    note_count: int
    pitch_min: int
    pitch_max: int
    pitch_classes: tuple[int, ...]
    octaves: tuple[int, ...]
    durations_qn: tuple[tuple[int, int], ...]
    positions_in_bar_qn: tuple[tuple[int, int], ...]
    canonical_piece_fingerprint: str
    raw_graph_fingerprint: str

    @property
    def identity(self) -> tuple[str, str]:
        return (self.dataset_id, self.piece_id)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-ready fingerprint payload."""

        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "piece_id": self.piece_id,
            "source_group_id": self.source_group_id,
            "track_count": self.track_count,
            "bar_count": self.bar_count,
            "beat_count": self.beat_count,
            "onset_count": self.onset_count,
            "note_count": self.note_count,
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "pitch_classes": list(self.pitch_classes),
            "octaves": list(self.octaves),
            "durations_qn": [
                list(duration) for duration in self.durations_qn
            ],
            "positions_in_bar_qn": [
                list(position) for position in self.positions_in_bar_qn
            ],
            "canonical_piece_fingerprint": (
                self.canonical_piece_fingerprint
            ),
            "raw_graph_fingerprint": self.raw_graph_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class Phase7ABoundedFixture:
    """Target-free train/validation pieces and their exact bindings."""

    contract_version: str
    policy: str
    train_pieces: tuple[CanonicalPiece, ...]
    validation_pieces: tuple[CanonicalPiece, ...]
    composition: tuple[Phase7APieceComposition, ...]
    split_fingerprint: str
    train_composition_fingerprint: str
    validation_composition_fingerprint: str
    fixture_fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.contract_version_incompatible"
            )
        if self.policy != PHASE7A_BOUNDED_FIXTURE_POLICY:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.policy_incompatible"
            )
        if not self.train_pieces or not self.validation_pieces:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.split_empty"
            )
        train_identities = set(self.identities("train"))
        validation_identities = set(self.identities("validation"))
        if train_identities & validation_identities:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.identity_overlap"
            )
        train_groups = {
            piece.source_group_id for piece in self.train_pieces
        }
        validation_groups = {
            piece.source_group_id for piece in self.validation_pieces
        }
        if train_groups & validation_groups:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.source_group_overlap"
            )

    def pieces(self, split: FixtureSplit) -> tuple[CanonicalPiece, ...]:
        """Return one immutable canonical split."""

        if split == "train":
            return self.train_pieces
        if split == "validation":
            return self.validation_pieces
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.split_invalid"
        )

    def identities(
        self,
        split: FixtureSplit,
    ) -> tuple[tuple[str, str], ...]:
        """Return ordered ``(dataset_id, piece_id)`` identities."""

        return tuple(
            (piece.dataset_name, piece.piece_id)
            for piece in self.pieces(split)
        )

    def piece_lookup(
        self,
    ) -> dict[tuple[str, str], CanonicalPiece]:
        """Return a deterministic identity-to-canonical-piece lookup."""

        pairs = tuple(
            (
                (piece.dataset_name, piece.piece_id),
                piece,
            )
            for piece in self.train_pieces + self.validation_pieces
        )
        lookup = dict(pairs)
        if len(lookup) != len(pairs):
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.identity_duplicate"
            )
        return lookup

    def piece_by_identity(
        self,
        dataset_id: str,
        piece_id: str,
    ) -> CanonicalPiece:
        """Resolve the canonical source used by a prepared batch identity."""

        try:
            return self.piece_lookup()[(dataset_id, piece_id)]
        except KeyError as exc:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.identity_unknown"
            ) from exc

    def raw_samples(
        self,
        split: FixtureSplit,
    ) -> tuple[SSLRawSample, ...]:
        """Build raw-only samples without importing supervised data paths."""

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
        """Aggregate exact structural and feature-variety counts."""

        if split is not None and split not in {"train", "validation"}:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.split_invalid"
            )
        selected = tuple(
            item
            for item in self.composition
            if split is None or item.split == split
        )
        pitch_classes = {
            pitch_class
            for item in selected
            for pitch_class in item.pitch_classes
        }
        octaves = {
            octave for item in selected for octave in item.octaves
        }
        durations = {
            duration
            for item in selected
            for duration in item.durations_qn
        }
        positions = {
            position
            for item in selected
            for position in item.positions_in_bar_qn
        }
        return {
            "split": split if split is not None else "all",
            "piece_count": len(selected),
            "track_count": sum(item.track_count for item in selected),
            "bar_count": sum(item.bar_count for item in selected),
            "beat_count": sum(item.beat_count for item in selected),
            "onset_count": sum(item.onset_count for item in selected),
            "note_count": sum(item.note_count for item in selected),
            "pitch_min": min(item.pitch_min for item in selected),
            "pitch_max": max(item.pitch_max for item in selected),
            "distinct_pitch_class_count": len(pitch_classes),
            "distinct_octave_count": len(octaves),
            "distinct_duration_count": len(durations),
            "distinct_position_in_bar_count": len(positions),
        }

    def fingerprint_bundle(self) -> dict[str, str]:
        """Return checkpoint/report-ready deterministic fingerprints."""

        return {
            "kind": "bounded",
            "bounded_fixture_fingerprint": self.fixture_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "train_composition_fingerprint": (
                self.train_composition_fingerprint
            ),
            "validation_composition_fingerprint": (
                self.validation_composition_fingerprint
            ),
        }

    def composition_payload(self) -> dict[str, object]:
        """Return all structural evidence without model or target data."""

        return {
            "contract_version": self.contract_version,
            "policy": self.policy,
            "train": [
                item.to_dict()
                for item in self.composition
                if item.split == "train"
            ],
            "validation": [
                item.to_dict()
                for item in self.composition
                if item.split == "validation"
            ],
        }


@dataclass(frozen=True, slots=True)
class CoherentPitchGroupMutation:
    """A fixed-identity canonical pitch mutation and rebuilt raw graphs."""

    source_piece: CanonicalPiece
    mutated_piece: CanonicalPiece
    contract_version: str
    policy: str
    policy_fingerprint: str
    mutation_instance_fingerprint: str
    selected_local_node_indices: tuple[int, ...]
    selected_note_ids: tuple[str, ...]
    source_pitches: tuple[int, ...]
    mutated_pitches: tuple[int, ...]
    source_raw_graph: Any
    mutated_raw_graph: Any
    source_raw_graph_fingerprint: str
    mutated_raw_graph_fingerprint: str
    changed_feature_slots: tuple[tuple[str, str, str], ...]

    def raw_sample(self, *, mutated: bool) -> SSLRawSample:
        """Materialize either graph as an identity-compatible raw sample."""

        from music_critic.ssl.data import SSLRawSample

        graph = self.mutated_raw_graph if mutated else self.source_raw_graph
        fingerprint = (
            self.mutated_raw_graph_fingerprint
            if mutated
            else self.source_raw_graph_fingerprint
        )
        return SSLRawSample(
            raw_graph=graph,
            raw_graph_fingerprint=fingerprint,
            dataset_id=self.source_piece.dataset_name,
            piece_id=self.source_piece.piece_id,
        )


@dataclass(frozen=True, slots=True)
class _PieceSpec:
    split: FixtureSplit
    ordinal: int
    bar_count: int
    track_pitch_bases: tuple[int, ...]
    programs: tuple[int, ...]
    pitch_rotation: int


_PIECE_SPECS = (
    _PieceSpec("train", 0, 3, (43, 67), (32, 0), 0),
    _PieceSpec("train", 1, 2, (38, 58, 76), (42, 24, 73), 2),
    _PieceSpec("train", 2, 2, (50, 70), (48, 40), 4),
    _PieceSpec("validation", 0, 3, (45, 69), (43, 5), 1),
    _PieceSpec("validation", 1, 2, (40, 60, 74), (33, 26, 68), 5),
)

_PITCH_OFFSETS = (0, 2, 7, 11, 14, 5, 17, 9, 23)
_POSITIONS = (RationalTime(0), RationalTime(3, 2), RationalTime(3))
_DURATION_CHOICES = (
    (RationalTime(1, 2), RationalTime(1), RationalTime(3, 2)),
    (RationalTime(3, 4), RationalTime(1), RationalTime(3, 2)),
    (RationalTime(1, 2), RationalTime(1)),
)


def _piece_token(spec: _PieceSpec) -> str:
    return f"phase7a-{spec.split}-{spec.ordinal:02d}"


def _make_piece(spec: _PieceSpec) -> CanonicalPiece:
    token = _piece_token(spec)
    provenance_id = f"prov:{token}.source"
    meter_id = f"meter:{token}.000"
    duration_qn = RationalTime(spec.bar_count * 4)
    tracks = tuple(
        CanonicalTrack(
            track_id=f"track:{token}.{track_index:02d}",
            source_track_index=track_index,
            name=f"Raw part {track_index + 1}",
            instrument_name=None,
            program=program,
            channel=track_index,
            is_percussion=False,
            provenance_id=provenance_id,
        )
        for track_index, program in enumerate(spec.programs)
    )
    bars = tuple(
        CanonicalBar(
            bar_id=f"bar:{token}.{bar_index:03d}",
            index=bar_index,
            start_qn=RationalTime(bar_index * 4),
            duration_qn=RationalTime(4),
            meter_event_id=meter_id,
            metric_offset_qn=RationalTime(0),
            is_pickup=False,
            is_incomplete=False,
            display_number=str(bar_index + 1),
            provenance_id=provenance_id,
        )
        for bar_index in range(spec.bar_count)
    )
    beats = tuple(
        CanonicalBeat(
            beat_id=f"beat:{token}.{bar_index:03d}.{beat_index}",
            bar_id=f"bar:{token}.{bar_index:03d}",
            meter_event_id=meter_id,
            index_in_bar=beat_index,
            start_qn=RationalTime(bar_index * 4 + beat_index),
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
        for bar_index in range(spec.bar_count)
        for beat_index in range(4)
    )
    notes = []
    for bar_index in range(spec.bar_count):
        for position_index, position in enumerate(_POSITIONS):
            for track_index, track in enumerate(tracks):
                event_index = bar_index * len(_POSITIONS) + position_index
                pitch_offset_index = (
                    event_index
                    + 2 * track_index
                    + spec.pitch_rotation
                ) % len(_PITCH_OFFSETS)
                pitch = (
                    spec.track_pitch_bases[track_index]
                    + _PITCH_OFFSETS[pitch_offset_index]
                )
                duration_options = _DURATION_CHOICES[position_index]
                duration = duration_options[
                    (
                        bar_index
                        + track_index
                        + spec.pitch_rotation
                    )
                    % len(duration_options)
                ]
                onset = RationalTime(bar_index * 4) + position
                note_index = event_index * len(tracks) + track_index
                notes.append(
                    CanonicalNote(
                        note_id=f"note:{token}.{note_index:03d}",
                        track_id=track.track_id,
                        pitch=pitch,
                        onset_qn=onset,
                        duration_qn=duration,
                        velocity=(
                            54
                            + (
                                7 * event_index
                                + 11 * track_index
                                + 3 * spec.ordinal
                            )
                            % 58
                        ),
                        channel=track.channel,
                        program=track.program,
                        is_percussion=False,
                        is_grace=False,
                        spelling_step=None,
                        spelling_alter=None,
                        staff=None,
                        voice=None,
                        articulations=(),
                        dynamic=None,
                        source_onset_ticks=(
                            onset.num * 480 // onset.den
                        ),
                        source_duration_ticks=(
                            duration.num * 480 // duration.den
                        ),
                        source_onset_seconds=None,
                        source_duration_seconds=None,
                        provenance_id=provenance_id,
                    )
                )
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=f"piece:{token}",
        dataset_name=PHASE7A_BOUNDED_DATASET_ID,
        source_group_id=f"group:{token}",
        split=spec.split,
        source_path=None,
        source_resolution=480,
        duration_qn=duration_qn,
        metadata=PieceMetadata(
            source_format="synthetic",
            title=f"Phase 7A bounded {spec.split} {spec.ordinal}",
            creators=("Music Critic V2",),
            collection="Phase 7A deterministic bounded evidence",
            movement_title=None,
            movement_number=None,
            genres=(),
            copyright=None,
            language=None,
        ),
        tracks=tracks,
        notes=tuple(notes),
        bars=bars,
        beats=beats,
        tempo_events=(
            TempoEvent(
                tempo_event_id=f"tempo:{token}.000",
                onset_qn=RationalTime(0),
                microseconds_per_quarter=(
                    480_000 + 10_000 * spec.ordinal
                ),
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
                source=PHASE7A_BOUNDED_FIXTURE_POLICY,
                record_id=token,
                uri=None,
                version=PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
                checksum_sha256=None,
                created_at="2026-07-29T00:00:00+03:00",
                parents=(),
                details=(
                    ("ordinal", spec.ordinal),
                    ("split", spec.split),
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
        raise Phase7ABoundedFixtureError(
            f"phase7a.fixture.canonical_piece_invalid:{errors}"
        )
    return piece


def _composition(
    piece: CanonicalPiece,
    split: FixtureSplit,
) -> Phase7APieceComposition:
    graph = build_raw_graph(piece, assume_valid=True)
    bar_starts = {
        bar.bar_id: bar.start_qn for bar in piece.bars
    }
    positions = set()
    for note in piece.notes:
        owner = max(
            (
                bar
                for bar in piece.bars
                if bar.start_qn <= note.onset_qn
            ),
            key=lambda bar: bar.start_qn,
        )
        positions.add(note.onset_qn - bar_starts[owner.bar_id])
    pitches = tuple(note.pitch for note in piece.notes)
    return Phase7APieceComposition(
        split=split,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        source_group_id=piece.source_group_id,
        track_count=len(piece.tracks),
        bar_count=len(piece.bars),
        beat_count=len(piece.beats),
        onset_count=len({note.onset_qn for note in piece.notes}),
        note_count=len(piece.notes),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        pitch_classes=tuple(sorted({pitch % 12 for pitch in pitches})),
        octaves=tuple(sorted({pitch // 12 for pitch in pitches})),
        durations_qn=tuple(
            sorted(
                {
                    _time_payload(note.duration_qn)
                    for note in piece.notes
                }
            )
        ),
        positions_in_bar_qn=tuple(
            sorted(_time_payload(position) for position in positions)
        ),
        canonical_piece_fingerprint=_piece_fingerprint(piece),
        raw_graph_fingerprint=graph_fingerprint(graph),
    )


def build_phase7a_bounded_fixture() -> Phase7ABoundedFixture:
    """Build the exact target-free multi-note Phase 7A evidence fixture."""

    train_pieces = tuple(
        _make_piece(spec)
        for spec in _PIECE_SPECS
        if spec.split == "train"
    )
    validation_pieces = tuple(
        _make_piece(spec)
        for spec in _PIECE_SPECS
        if spec.split == "validation"
    )
    composition = tuple(
        _composition(piece, "train") for piece in train_pieces
    ) + tuple(
        _composition(piece, "validation")
        for piece in validation_pieces
    )
    split_payload = {
        "contract_version": PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
        "policy": PHASE7A_BOUNDED_FIXTURE_POLICY,
        "train": [
            {
                "dataset_id": piece.dataset_name,
                "piece_id": piece.piece_id,
                "source_group_id": piece.source_group_id,
            }
            for piece in train_pieces
        ],
        "validation": [
            {
                "dataset_id": piece.dataset_name,
                "piece_id": piece.piece_id,
                "source_group_id": piece.source_group_id,
            }
            for piece in validation_pieces
        ],
    }
    split_fingerprint = _canonical_fingerprint(split_payload)
    train_composition_fingerprint = _canonical_fingerprint(
        [
            item.to_dict()
            for item in composition
            if item.split == "train"
        ]
    )
    validation_composition_fingerprint = _canonical_fingerprint(
        [
            item.to_dict()
            for item in composition
            if item.split == "validation"
        ]
    )
    fixture_fingerprint = _canonical_fingerprint(
        {
            "contract_version": PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
            "policy": PHASE7A_BOUNDED_FIXTURE_POLICY,
            "split_fingerprint": split_fingerprint,
            "train_composition_fingerprint": (
                train_composition_fingerprint
            ),
            "validation_composition_fingerprint": (
                validation_composition_fingerprint
            ),
        }
    )
    return Phase7ABoundedFixture(
        contract_version=PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION,
        policy=PHASE7A_BOUNDED_FIXTURE_POLICY,
        train_pieces=train_pieces,
        validation_pieces=validation_pieces,
        composition=composition,
        split_fingerprint=split_fingerprint,
        train_composition_fingerprint=train_composition_fingerprint,
        validation_composition_fingerprint=(
            validation_composition_fingerprint
        ),
        fixture_fingerprint=fixture_fingerprint,
    )


def _changed_feature_slots(
    source_graph: Any,
    mutated_graph: Any,
) -> tuple[tuple[str, str, str], ...]:
    changed = []
    for node_type in MANDATORY_NODE_TYPES:
        for kind, value_name, availability_name in (
            ("categorical", "x_cat", "x_cat_available"),
            ("continuous", "x_cont", "x_cont_available"),
        ):
            names = RAW_FEATURE_REGISTRY.names(node_type, kind)
            source_values = source_graph[node_type][value_name]
            mutated_values = mutated_graph[node_type][value_name]
            source_available = source_graph[node_type][availability_name]
            mutated_available = mutated_graph[node_type][availability_name]
            for column, name in enumerate(names):
                if (
                    not torch.equal(
                        source_values[:, column],
                        mutated_values[:, column],
                    )
                    or not torch.equal(
                        source_available[:, column],
                        mutated_available[:, column],
                    )
                ):
                    changed.append((node_type, kind, name))
    return tuple(changed)


def mutate_piece_pitch_group(
    piece: CanonicalPiece,
    selected_local_node_indices: Sequence[int],
) -> CoherentPitchGroupMutation:
    """Mutate selected canonical pitches and rebuild all dependent features.

    Timing, node identity, ownership, and topology stay fixed, making the
    result suitable as an alternative full-view target for one fixed
    :class:`MaskPlan`.  The versioned policy reflects each pitch around the
    MIDI range axis (``pitch -> 127 - pitch``), producing an auditable,
    non-post-hoc alternative rather than a tuned semitone delta.
    """

    if not isinstance(piece, CanonicalPiece):
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_piece_invalid"
        )
    if piece.annotations or piece.targets:
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_requires_target_free_piece"
        )
    if (
        isinstance(selected_local_node_indices, (str, bytes))
        or not isinstance(selected_local_node_indices, Sequence)
    ):
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_indices_invalid"
        )
    selected = tuple(selected_local_node_indices)
    if (
        not selected
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in selected
        )
        or selected != tuple(sorted(set(selected)))
        or selected[0] < 0
        or selected[-1] >= len(piece.notes)
    ):
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_indices_invalid"
        )
    source_graph = build_raw_graph(piece, assume_valid=True)
    source_fingerprint = graph_fingerprint(source_graph)
    source_pitches = tuple(
        piece.notes[index].pitch for index in selected
    )
    selected_note_ids = tuple(
        piece.notes[index].note_id for index in selected
    )
    mutated_pitches = tuple(127 - pitch for pitch in source_pitches)
    mutation_instance_fingerprint = _canonical_fingerprint(
        {
            "contract_version": (
                PHASE7A_PITCH_MUTATION_CONTRACT_VERSION
            ),
            "policy_fingerprint": (
                PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT
            ),
            "source_raw_graph_fingerprint": source_fingerprint,
            "selected_local_node_indices": list(selected),
            "selected_note_ids": list(selected_note_ids),
            "source_pitches": list(source_pitches),
            "mutated_pitches": list(mutated_pitches),
        }
    )
    mutation_provenance_id = (
        f"prov:phase7a-pitch-mutation."
        f"{piece.piece_id.removeprefix('piece:')}."
        f"{mutation_instance_fingerprint[:16]}"
    )
    notes = list(piece.notes)
    for index, candidate in zip(
        selected,
        mutated_pitches,
        strict=True,
    ):
        source = notes[index]
        if not 0 <= candidate <= 127 or candidate == source.pitch:
            raise Phase7ABoundedFixtureError(
                "phase7a.fixture.mutation_pitch_out_of_range"
            )
        notes[index] = replace(
            source,
            pitch=candidate,
            spelling_step=None,
            spelling_alter=None,
            provenance_id=mutation_provenance_id,
        )
    mutation_provenance = ProvenanceRecord(
        provenance_id=mutation_provenance_id,
        kind="derivation",
        source="phase7a.coherent_pitch_group_mutation",
        record_id=piece.piece_id,
        uri=None,
        version=PHASE7A_PITCH_MUTATION_CONTRACT_VERSION,
        checksum_sha256=None,
        created_at="2026-07-29T00:00:00+03:00",
        parents=tuple(
            sorted(
                {
                    note.provenance_id
                    for note in piece.notes
                    if note.provenance_id is not None
                }
            )
        ),
        details=(
            (
                "mutation_instance_fingerprint",
                mutation_instance_fingerprint,
            ),
            ("mutation_policy", PHASE7A_PITCH_MUTATION_POLICY),
            (
                "mutation_policy_fingerprint",
                PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT,
            ),
            ("selected_count", len(selected)),
        ),
    )
    mutated_piece = replace(
        piece,
        notes=tuple(notes),
        provenance=piece.provenance + (mutation_provenance,),
    )
    report = validate_piece(mutated_piece)
    if report.errors:
        errors = ",".join(
            f"{issue.code}@{issue.path}" for issue in report.errors
        )
        raise Phase7ABoundedFixtureError(
            f"phase7a.fixture.mutated_piece_invalid:{errors}"
        )

    mutated_graph = build_raw_graph(mutated_piece, assume_valid=True)
    if any(
        source_graph[node_type].entity_id
        != mutated_graph[node_type].entity_id
        for node_type in MANDATORY_NODE_TYPES
    ):
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_changed_node_identity"
        )
    if any(
        not torch.equal(
            source_graph[edge_type].edge_index,
            mutated_graph[edge_type].edge_index,
        )
        for edge_type in MANDATORY_EDGE_TYPES
    ):
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_changed_topology"
        )
    mutated_fingerprint = graph_fingerprint(mutated_graph)
    if source_fingerprint == mutated_fingerprint:
        raise Phase7ABoundedFixtureError(
            "phase7a.fixture.mutation_graph_unchanged"
        )
    return CoherentPitchGroupMutation(
        source_piece=piece,
        mutated_piece=mutated_piece,
        contract_version=PHASE7A_PITCH_MUTATION_CONTRACT_VERSION,
        policy=PHASE7A_PITCH_MUTATION_POLICY,
        policy_fingerprint=(
            PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT
        ),
        mutation_instance_fingerprint=(
            mutation_instance_fingerprint
        ),
        selected_local_node_indices=selected,
        selected_note_ids=selected_note_ids,
        source_pitches=source_pitches,
        mutated_pitches=mutated_pitches,
        source_raw_graph=source_graph,
        mutated_raw_graph=mutated_graph,
        source_raw_graph_fingerprint=source_fingerprint,
        mutated_raw_graph_fingerprint=mutated_fingerprint,
        changed_feature_slots=_changed_feature_slots(
            source_graph,
            mutated_graph,
        ),
    )


__all__ = [
    "PHASE7A_BOUNDED_DATASET_ID",
    "PHASE7A_BOUNDED_FIXTURE_CONTRACT_VERSION",
    "PHASE7A_BOUNDED_FIXTURE_POLICY",
    "PHASE7A_PITCH_MUTATION_CONTRACT_VERSION",
    "PHASE7A_PITCH_MUTATION_POLICY",
    "PHASE7A_PITCH_MUTATION_POLICY_FINGERPRINT",
    "CoherentPitchGroupMutation",
    "Phase7ABoundedFixture",
    "Phase7ABoundedFixtureError",
    "Phase7APieceComposition",
    "build_phase7a_bounded_fixture",
    "mutate_piece_pitch_group",
]
