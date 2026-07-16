"""Immutable canonical symbolic-music schema records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from music_critic.data.timing import RationalTime


SCHEMA_VERSION = "2.0.0"

JsonScalar: TypeAlias = str | int | float | bool | None
ProvenanceDetail: TypeAlias = tuple[str, JsonScalar]

Split: TypeAlias = str | None
SourceFormat: TypeAlias = Literal[
    "midi", "musicxml", "json", "jsonl", "tsv", "synthetic", "other"
]
KeySignatureMode: TypeAlias = Literal[
    "major",
    "minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "locrian",
    "other",
    "unknown",
]
AnnotationLayer: TypeAlias = Literal["observation", "target_alignment"]
AlignmentType: TypeAlias = Literal[
    "piece",
    "track",
    "note",
    "bar",
    "beat",
    "bar_boundary",
    "beat_boundary",
    "annotation_span",
]
TargetValueType: TypeAlias = Literal[
    "categorical", "scalar", "multi_label", "distribution"
]
TargetSource: TypeAlias = Literal[
    "human",
    "dataset",
    "algorithm",
    "pseudo_label",
    "derived",
    "synthetic",
]
TargetValue: TypeAlias = (
    str | int | float | tuple[str, ...] | tuple[float, ...]
)
ProvenanceKind: TypeAlias = Literal[
    "source", "conversion", "annotation", "derivation", "default", "synthetic"
]
IssueSeverity: TypeAlias = Literal["error", "warning"]
QualitySeverity: TypeAlias = Literal["info", "warning"]
QualityFlagCode: TypeAlias = str


@dataclass(frozen=True, slots=True)
class PieceMetadata:
    source_format: SourceFormat
    title: str | None
    creators: tuple[str, ...] | None
    collection: str | None
    movement_title: str | None
    movement_number: str | None
    genres: tuple[str, ...] | None
    copyright: str | None
    language: str | None


@dataclass(frozen=True, slots=True)
class CanonicalTrack:
    track_id: str
    source_track_index: int | None
    name: str | None
    instrument_name: str | None
    program: int | None
    channel: int | None
    is_percussion: bool
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class CanonicalNote:
    note_id: str
    track_id: str
    pitch: int
    onset_qn: RationalTime
    duration_qn: RationalTime
    velocity: int | None
    channel: int | None
    program: int | None
    is_percussion: bool
    is_grace: bool
    spelling_step: str | None
    spelling_alter: int | None
    staff: int | None
    voice: int | None
    articulations: tuple[str, ...] | None
    dynamic: str | None
    source_onset_ticks: int | None
    source_duration_ticks: int | None
    source_onset_seconds: float | None
    source_duration_seconds: float | None
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class CanonicalBar:
    bar_id: str
    index: int
    start_qn: RationalTime
    duration_qn: RationalTime
    meter_event_id: str
    metric_offset_qn: RationalTime
    is_pickup: bool
    is_incomplete: bool
    display_number: str | None
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class CanonicalBeat:
    beat_id: str
    bar_id: str
    meter_event_id: str
    index_in_bar: int
    start_qn: RationalTime
    duration_qn: RationalTime
    position_in_bar_qn: RationalTime
    is_downbeat: bool
    strength: float | None
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class TempoEvent:
    tempo_event_id: str
    onset_qn: RationalTime
    microseconds_per_quarter: int
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class MeterEvent:
    meter_event_id: str
    onset_qn: RationalTime
    numerator: int
    denominator: int
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class KeySignatureEvent:
    key_signature_event_id: str
    onset_qn: RationalTime
    fifths: int
    mode: KeySignatureMode
    raw_value: str | None
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class AnnotationSpan:
    annotation_id: str
    annotation_type: str
    layer: AnnotationLayer
    start_qn: RationalTime
    end_qn: RationalTime
    track_id: str | None
    value: str | None
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class TargetArray:
    target_id: str
    task: str
    annotation_view_id: str | None
    alignment_type: AlignmentType
    entity_ids: tuple[str, ...]
    value_type: TargetValueType
    class_labels: tuple[str, ...] | None
    values: tuple[TargetValue | None, ...]
    mask: tuple[bool, ...]
    confidence: tuple[float | None, ...]
    source: tuple[TargetSource | None, ...]
    provenance: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    provenance_id: str
    kind: ProvenanceKind
    source: str
    record_id: str | None
    uri: str | None
    version: str | None
    checksum_sha256: str | None
    created_at: str | None
    parents: tuple[str, ...]
    details: tuple[ProvenanceDetail, ...]


@dataclass(frozen=True, slots=True)
class QualityFlag:
    code: QualityFlagCode
    severity: QualitySeverity
    message: str
    entity_ids: tuple[str, ...]
    provenance_id: str | None


@dataclass(frozen=True, slots=True)
class CanonicalPiece:
    schema_version: str
    piece_id: str
    dataset_name: str
    source_group_id: str
    split: str | None
    source_path: str | None
    source_resolution: int | None
    duration_qn: RationalTime
    metadata: PieceMetadata
    tracks: tuple[CanonicalTrack, ...]
    notes: tuple[CanonicalNote, ...]
    bars: tuple[CanonicalBar, ...]
    beats: tuple[CanonicalBeat, ...]
    tempo_events: tuple[TempoEvent, ...]
    meter_events: tuple[MeterEvent, ...]
    key_signature_events: tuple[KeySignatureEvent, ...]
    annotations: tuple[AnnotationSpan, ...]
    targets: tuple[TargetArray, ...]
    provenance: tuple[ProvenanceRecord, ...]
    quality_flags: tuple[QualityFlag, ...]
