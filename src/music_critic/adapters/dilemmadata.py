"""Production Phase 9B.1 Dilemmadata raw adapter.

The adapter deliberately stops at a target-independent canonical piece.  TSV
columns outside the evidenced raw projection are never read by conversion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import os
from os import PathLike
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    KeySignatureEvent,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
    TempoEvent,
    ValidationReport,
    dumps_piece,
    validate_piece,
)


DILEMMADATA_ADAPTER_VERSION = "1.0.1"
DILEMMADATA_CORPUS_IDENTITY_VERSION = "1.0.0"
DILEMMADATA_RAW_PROJECTION_VERSION = "1.0.0"
DILEMMADATA_GROUPING_VERSION = "1.0.0"
DILEMMADATA_RECORD_BINDING_VERSION = "1.0.0"
DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION = "1.0.0"
DILEMMADATA_ACCEPTANCE_REPORT_VERSION = "1.1.0"
DILEMMADATA_PRODUCTION_MANIFEST_VERSION = "1.1.0"
DILEMMADATA_DATASET_NAME = "dilemmadata"
DILEMMADATA_RELEASE_VERSION = "v1.0"
DILEMMADATA_RELEASE_COMMIT = "d60ee75b4a9495e932a4a7be39381578be17e222"
DILEMMADATA_INSTALLATION_FILE_COUNT = 2_743
DILEMMADATA_CONTENT_FINGERPRINT = (
    "8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312"
)
DILEMMADATA_PRIMARY_RECORD_COUNT = 1_633
DILEMMADATA_AN_RECORD_COUNT = 353
DILEMMADATA_DLC_RECORD_COUNT = 1_280

_DEFAULT_TEMPO = 500_000
_MAX_METRIC_RECORDS = 2_000_000
_MISSING = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
_TRUE = frozenset({"1", "True", "true", "TRUE"})
_FALSE = frozenset({"0", "False", "false", "FALSE"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NOTE_MULTISET_DOMAIN = b"dilemmadata.midi-note-event-multiset-grouping.1\0"
_RAW_PROJECTION_DOMAIN = b"dilemmadata.target-independent-raw-projection.1\0"
_RECORD_BINDING_DOMAIN = b"dilemmadata.discovery-record-binding.1\0"

_TRACK_POLICY = "single_source_neutral_pitched_track_v1"
_TIE_POLICY = "merge_exact_contiguous_same_pitch_source_voice_v1"
_GRACE_POLICY = "retain_zero_duration_as_grace_v1"
_METER_POLICY = "exact_measure_anchor_or_metric_grid_v1"
_DEFAULT_POLICY = "explicit_unknowns_and_default_tempo_v1"

_AN_REQUIRED_FIELDS = (
    "onset_div",
    "duration_div",
    "s_offset_frac",
    "s_duration_frac",
    "s_midi",
    "s_isOnset",
    "ks_fifths",
    "ts_beats",
    "ts_beat_type",
)
_DLC_REQUIRED_FIELDS = (
    "onset_div",
    "duration_div",
    "quarterbeats_playthrough",
    "duration",
    "pitch",
    "is_note_onset",
    "ks_fifths",
    "ts_beats",
    "ts_beat_type",
)


class DilemmadataAdapterError(ValueError):
    """Base class for stable Dilemmadata adapter failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        record_id: str | None = None,
    ) -> None:
        self.category = category
        self.record_id = record_id
        super().__init__(f"[{category}] {message}")


class DilemmadataCorpusIdentityError(DilemmadataAdapterError):
    """Fatal failure raised before full-corpus record iteration."""

    def __init__(self, discovery: DilemmadataCorpusDiscovery) -> None:
        self.discovery = discovery
        categories = ", ".join(issue.category for issue in discovery.issues)
        super().__init__(
            f"Dilemmadata corpus identity failed: {categories}",
            category="dilemmadata.corpus_identity",
        )


class DilemmadataConversionError(DilemmadataAdapterError):
    """Raised only for invalid API use; source defects are quarantined."""


@dataclass(frozen=True, slots=True)
class DilemmadataCorpusIdentity:
    version: str = DILEMMADATA_CORPUS_IDENTITY_VERSION
    release_version: str = DILEMMADATA_RELEASE_VERSION
    release_commit: str = DILEMMADATA_RELEASE_COMMIT
    installation_file_count: int = DILEMMADATA_INSTALLATION_FILE_COUNT
    content_fingerprint: str = DILEMMADATA_CONTENT_FINGERPRINT
    primary_record_count: int = DILEMMADATA_PRIMARY_RECORD_COUNT
    an_record_count: int = DILEMMADATA_AN_RECORD_COUNT
    dlc_record_count: int = DILEMMADATA_DLC_RECORD_COUNT


@dataclass(frozen=True, slots=True)
class DilemmadataAdapterConfig:
    dataset_name: str = DILEMMADATA_DATASET_NAME
    track_policy: Literal["single_source_neutral_pitched_track_v1"] = _TRACK_POLICY
    tie_policy: Literal[
        "merge_exact_contiguous_same_pitch_source_voice_v1"
    ] = _TIE_POLICY
    grace_policy: Literal["retain_zero_duration_as_grace_v1"] = _GRACE_POLICY
    meter_policy: Literal[
        "exact_measure_anchor_or_metric_grid_v1"
    ] = _METER_POLICY
    default_policy: Literal[
        "explicit_unknowns_and_default_tempo_v1"
    ] = _DEFAULT_POLICY
    default_tempo_microseconds_per_quarter: int = _DEFAULT_TEMPO

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise DilemmadataAdapterError(
                "dataset_name must be a non-empty string",
                category="dilemmadata.config_invalid",
            )
        for field_name, value, implemented in (
            ("track_policy", self.track_policy, _TRACK_POLICY),
            ("tie_policy", self.tie_policy, _TIE_POLICY),
            ("grace_policy", self.grace_policy, _GRACE_POLICY),
            ("meter_policy", self.meter_policy, _METER_POLICY),
            ("default_policy", self.default_policy, _DEFAULT_POLICY),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, str)
                or value != implemented
            ):
                raise DilemmadataAdapterError(
                    f"{field_name} must select the implemented policy",
                    category="dilemmadata.config_invalid",
                )
        if (
            isinstance(self.default_tempo_microseconds_per_quarter, bool)
            or not isinstance(self.default_tempo_microseconds_per_quarter, int)
            or self.default_tempo_microseconds_per_quarter <= 0
        ):
            raise DilemmadataAdapterError(
                "default tempo must be a positive integer",
                category="dilemmadata.config_invalid",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCorpusIssue:
    category: str
    record_id: str | None
    paths: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class DilemmadataCorpusRecord:
    record_id: str
    piece_id: str
    dialect: Literal["an_joint", "dlc"]
    path: Path
    relative_path: str
    collection: str
    piece_name: str
    suggested_split: str | None
    physical_source_sha256: str
    raw_projection_sha256: str
    raw_equivalence_id: str
    grouping_fingerprint: str
    source_group_id: str
    lineage_group_id: str
    source_resolution: int | None
    score_path: Path | None
    score_relative_path: str | None
    score_sha256: str | None
    raw_issue_categories: tuple[str, ...]
    note_row_count: int
    tie_continuation_row_count: int
    zero_duration_row_count: int
    corpus_identity: DilemmadataCorpusIdentity
    record_binding_version: str
    record_binding_sha256: str
    dataset_name: str = DILEMMADATA_DATASET_NAME


@dataclass(frozen=True, slots=True)
class DilemmadataCorpusDiscovery:
    root: Path
    records: tuple[DilemmadataCorpusRecord, ...]
    content_fingerprint: str
    installation_file_count: int
    installation_byte_count: int
    release_version: str | None
    issues: tuple[DilemmadataCorpusIssue, ...]
    auxiliary_tsv_paths: tuple[str, ...]
    component_count: int
    multi_record_component_count: int
    explicit_overlap_count: int
    suggested_split_conflict_count: int
    corpus_identity: DilemmadataCorpusIdentity
    identity_version: str = DILEMMADATA_CORPUS_IDENTITY_VERSION
    grouping_version: str = DILEMMADATA_GROUPING_VERSION
    record_binding_version: str = DILEMMADATA_RECORD_BINDING_VERSION

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class DilemmadataConversionStatistics:
    source_note_row_count: int
    canonical_note_count: int
    tie_continuation_row_count: int
    tie_merge_count: int
    grace_note_count: int
    meter_event_count: int
    bar_count: int
    beat_count: int
    pickup_bar_count: int
    incomplete_bar_count: int


@dataclass(frozen=True, slots=True)
class DilemmadataSourceRowBinding:
    """Target-neutral exact binding from one TSV row to a canonical note."""

    ordinal: int
    line: int
    onset_qn: RationalTime
    canonical_note_id: str
    tie_continuation: bool


@dataclass(frozen=True, slots=True)
class DilemmadataRawTargetAlignmentEvidence:
    """Versioned raw-adapter evidence consumed by the target-only adapter."""

    version: str
    record_id: str
    piece_id: str
    record_binding_sha256: str
    raw_projection_sha256: str
    canonical_piece_sha256: str
    rows: tuple[DilemmadataSourceRowBinding, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DilemmadataAccepted:
    status: Literal["accepted"]
    record: DilemmadataCorpusRecord
    piece: CanonicalPiece
    validation_report: ValidationReport
    statistics: DilemmadataConversionStatistics
    alignment_evidence: DilemmadataRawTargetAlignmentEvidence


@dataclass(frozen=True, slots=True)
class DilemmadataQuarantine:
    status: Literal["quarantined"]
    record: DilemmadataCorpusRecord
    categories: tuple[str, ...]
    messages: tuple[str, ...]


DilemmadataConversionResult = DilemmadataAccepted | DilemmadataQuarantine


@dataclass(frozen=True, slots=True)
class _RawObservation:
    ordinal: int
    line: int
    onset: Fraction
    duration: Fraction
    pitch: int
    tie_onset: bool
    spelling_step: str | None
    spelling_alter: int | None
    staff: int | None
    voice: int | None
    source_voice_key: tuple[str, ...]
    meter: tuple[int, int]
    key_fifths: int | None
    measure_key: str | None
    measure_anchor: Fraction | None
    display_measure: str | None


@dataclass(frozen=True, slots=True)
class _RawParse:
    observations: tuple[_RawObservation, ...]
    raw_projection_sha256: str
    grouping_fingerprint: str
    source_resolution: int | None
    categories: tuple[str, ...]
    messages: tuple[str, ...]
    note_row_count: int
    tie_continuation_row_count: int
    zero_duration_row_count: int


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class _BoundedMultisetFingerprint:
    _MODULUS = 1 << 256

    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.squared_total = 0
        self.xor = 0

    def add(self, value: bytes) -> None:
        integer = int.from_bytes(value, "big")
        self.count += 1
        self.total = (self.total + integer) % self._MODULUS
        self.squared_total = (
            self.squared_total + integer * integer
        ) % self._MODULUS
        self.xor ^= integer

    def hexdigest(self) -> str:
        digest = sha256(_NOTE_MULTISET_DOMAIN)
        digest.update(self.count.to_bytes(8, "big"))
        digest.update(self.total.to_bytes(32, "big"))
        digest.update(self.squared_total.to_bytes(32, "big"))
        digest.update(self.xor.to_bytes(32, "big"))
        return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _alignment_evidence_payload(
    evidence: DilemmadataRawTargetAlignmentEvidence,
) -> dict[str, object]:
    return {
        "canonical_piece_sha256": evidence.canonical_piece_sha256,
        "piece_id": evidence.piece_id,
        "raw_projection_sha256": evidence.raw_projection_sha256,
        "record_binding_sha256": evidence.record_binding_sha256,
        "record_id": evidence.record_id,
        "rows": [
            {
                "canonical_note_id": row.canonical_note_id,
                "line": row.line,
                "onset_qn": {
                    "den": row.onset_qn.den,
                    "num": row.onset_qn.num,
                },
                "ordinal": row.ordinal,
                "tie_continuation": row.tie_continuation,
            }
            for row in evidence.rows
        ],
        "version": evidence.version,
    }


def _alignment_evidence_fingerprint(
    evidence: DilemmadataRawTargetAlignmentEvidence,
) -> str:
    return _fingerprint(_alignment_evidence_payload(evidence))


def _make_alignment_evidence(
    record: DilemmadataCorpusRecord,
    piece: CanonicalPiece,
    rows: tuple[DilemmadataSourceRowBinding, ...],
) -> DilemmadataRawTargetAlignmentEvidence:
    evidence = DilemmadataRawTargetAlignmentEvidence(
        version=DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION,
        record_id=record.record_id,
        piece_id=piece.piece_id,
        record_binding_sha256=record.record_binding_sha256,
        raw_projection_sha256=record.raw_projection_sha256,
        canonical_piece_sha256=sha256(dumps_piece(piece).encode("utf-8")).hexdigest(),
        rows=rows,
        fingerprint="",
    )
    return replace(evidence, fingerprint=_alignment_evidence_fingerprint(evidence))


def validate_dilemmadata_alignment_evidence(
    record: DilemmadataCorpusRecord,
    piece: CanonicalPiece,
    evidence: DilemmadataRawTargetAlignmentEvidence,
) -> bool:
    """Check exact record/canonical/row binding without reading target columns."""

    try:
        if evidence.version != DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION:
            return False
        if (
            evidence.record_id != record.record_id
            or evidence.piece_id != piece.piece_id
            or evidence.record_binding_sha256 != record.record_binding_sha256
            or evidence.raw_projection_sha256 != record.raw_projection_sha256
        ):
            return False
        if evidence.canonical_piece_sha256 != sha256(
            dumps_piece(piece).encode("utf-8")
        ).hexdigest():
            return False
        if evidence.fingerprint != _alignment_evidence_fingerprint(evidence):
            return False
        if tuple(row.ordinal for row in evidence.rows) != tuple(range(len(evidence.rows))):
            return False
        note_ids = {note.note_id for note in piece.notes}
        if any(
            row.line != row.ordinal + 2
            or row.canonical_note_id not in note_ids
            or not isinstance(row.tie_continuation, bool)
            for row in evidence.rows
        ):
            return False
        return len(evidence.rows) == record.note_row_count
    except (AttributeError, TypeError, ValueError):
        return False


def _path_locator_sha256(path: Path) -> str:
    """Bind a discovered path without exposing it in diagnostics or manifests."""

    digest = sha256(_RECORD_BINDING_DOMAIN)
    digest.update(os.fsencode(path.resolve(strict=False)))
    return digest.hexdigest()


def _record_binding_payload(record: DilemmadataCorpusRecord) -> dict[str, object]:
    return {
        "binding_version": record.record_binding_version,
        "corpus_identity": asdict(record.corpus_identity),
        "record": {
            "collection": record.collection,
            "dataset_name": record.dataset_name,
            "dialect": record.dialect,
            "grouping_fingerprint": record.grouping_fingerprint,
            "lineage_group_id": record.lineage_group_id,
            "note_row_count": record.note_row_count,
            "physical_source_sha256": record.physical_source_sha256,
            "piece_id": record.piece_id,
            "piece_name": record.piece_name,
            "raw_equivalence_id": record.raw_equivalence_id,
            "raw_issue_categories": list(record.raw_issue_categories),
            "raw_projection_sha256": record.raw_projection_sha256,
            "record_id": record.record_id,
            "score_relative_path": record.score_relative_path,
            "score_sha256": record.score_sha256,
            "score_source_locator_sha256": (
                None
                if record.score_path is None
                else _path_locator_sha256(record.score_path)
            ),
            "source_group_id": record.source_group_id,
            "source_path": record.relative_path,
            "source_resolution": record.source_resolution,
            "source_locator_sha256": _path_locator_sha256(record.path),
            "suggested_split": record.suggested_split,
            "tie_continuation_row_count": record.tie_continuation_row_count,
            "zero_duration_row_count": record.zero_duration_row_count,
        },
    }


def _record_binding_sha256(record: DilemmadataCorpusRecord) -> str:
    digest = sha256(_RECORD_BINDING_DOMAIN)
    digest.update(_canonical_bytes(_record_binding_payload(record)))
    return digest.hexdigest()


def _portable_record_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and value == path.as_posix()
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def _record_binding_is_valid(record: DilemmadataCorpusRecord) -> bool:
    """Validate that a record is the exact product of one discovery pass."""

    try:
        identity = record.corpus_identity
        if not isinstance(identity, DilemmadataCorpusIdentity):
            return False
        if (
            identity.version != DILEMMADATA_CORPUS_IDENTITY_VERSION
            or identity.release_version != DILEMMADATA_RELEASE_VERSION
            or identity.release_commit != DILEMMADATA_RELEASE_COMMIT
        ):
            return False
        if record.record_binding_version != DILEMMADATA_RECORD_BINDING_VERSION:
            return False
        if record.dataset_name != DILEMMADATA_DATASET_NAME:
            return False
        if record.dialect not in ("an_joint", "dlc"):
            return False
        if not isinstance(record.path, Path) or not _portable_record_path(
            record.relative_path
        ):
            return False
        if record.piece_id != _piece_id(record.record_id, record.dialect):
            return False
        if not all(
            isinstance(value, str) and value
            for value in (
                record.record_id,
                record.collection,
                record.piece_name,
                record.source_group_id,
                record.lineage_group_id,
            )
        ):
            return False
        if not record.source_group_id.startswith("dilemmadata-component:"):
            return False
        if not record.lineage_group_id.startswith("dilemmadata-lineage:"):
            return False
        if not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in (
                record.physical_source_sha256,
                record.raw_projection_sha256,
                record.grouping_fingerprint,
                record.record_binding_sha256,
            )
        ):
            return False
        if record.raw_equivalence_id != (
            f"dilemmadata-raw:{record.raw_projection_sha256}"
        ):
            return False
        if record.suggested_split is not None and (
            not isinstance(record.suggested_split, str)
            or not record.suggested_split.strip()
        ):
            return False
        if record.source_resolution is not None and (
            isinstance(record.source_resolution, bool)
            or not isinstance(record.source_resolution, int)
            or record.source_resolution <= 0
        ):
            return False
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                record.note_row_count,
                record.tie_continuation_row_count,
                record.zero_duration_row_count,
            )
        ):
            return False
        score_values = (
            record.score_path,
            record.score_relative_path,
            record.score_sha256,
        )
        if any(value is None for value in score_values) != all(
            value is None for value in score_values
        ):
            return False
        if record.score_path is not None:
            if not isinstance(record.score_path, Path) or not _portable_record_path(
                record.score_relative_path
            ):
                return False
            if not isinstance(record.score_sha256, str) or not _SHA256_RE.fullmatch(
                record.score_sha256
            ):
                return False
        return _record_binding_sha256(record) == record.record_binding_sha256
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _bind_record(record: DilemmadataCorpusRecord) -> DilemmadataCorpusRecord:
    return replace(record, record_binding_sha256=_record_binding_sha256(record))


def validate_dilemmadata_record_binding(record: DilemmadataCorpusRecord) -> bool:
    """Public read-only verification for downstream target-sidecar adapters."""

    return _record_binding_is_valid(record)


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_files(root: Path) -> Iterator[Path]:
    resolved = root.resolve()
    for directory, dirnames, filenames in os.walk(resolved, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        for name in sorted(filenames):
            path = directory_path / name
            try:
                path.resolve().relative_to(resolved)
            except (OSError, ValueError):
                continue
            if path.is_file():
                yield path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _bool_value(value: str) -> bool | None:
    normalized = value.strip()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def _fraction(value: str) -> Fraction:
    normalized = value.strip()
    if normalized in _MISSING:
        raise ValueError("missing rational value")
    return Fraction(normalized)


def _optional_int(value: str) -> int | None:
    normalized = value.strip()
    if normalized in _MISSING:
        return None
    return int(normalized)


def _note_hash(onset: Fraction, duration: Fraction, pitch: int) -> bytes:
    return sha256(
        _canonical_bytes(
            [
                [onset.numerator, onset.denominator],
                [duration.numerator, duration.denominator],
                pitch,
            ]
        )
    ).digest()


def _ratio(value: str, musical: Fraction) -> Fraction | None:
    if musical == 0:
        return None
    return Fraction(int(value), 1) / musical


def _raw_projection_row(observation: _RawObservation) -> dict[str, object]:
    def rational(value: Fraction | None) -> list[int] | None:
        if value is None:
            return None
        return [value.numerator, value.denominator]

    return {
        "duration_qn": rational(observation.duration),
        "key_fifths": observation.key_fifths,
        "measure_anchor_qn": rational(observation.measure_anchor),
        "measure_key": observation.measure_key,
        "midi_pitch": observation.pitch,
        "onset_qn": rational(observation.onset),
        "source_staff": observation.staff,
        "source_tie_onset": observation.tie_onset,
        "source_voice": observation.voice,
        "source_voice_key": list(observation.source_voice_key),
        "spelling_alter": observation.spelling_alter,
        "spelling_step": observation.spelling_step,
        "time_signature": list(observation.meter),
    }


def _parse_raw_file(
    path: Path,
    dialect: Literal["an_joint", "dlc"],
) -> _RawParse:
    required = _AN_REQUIRED_FIELDS if dialect == "an_joint" else _DLC_REQUIRED_FIELDS
    categories: list[str] = []
    messages: list[str] = []
    observations: list[_RawObservation] = []
    resolutions: set[int] = set()
    grouping = _BoundedMultisetFingerprint()
    raw_digest = sha256(_RAW_PROJECTION_DOMAIN)
    raw_digest.update(dialect.encode("ascii"))
    raw_digest.update(b"\0")
    previous_onset: Fraction | None = None
    tie_count = 0
    grace_count = 0
    row_count = 0

    def issue(category: str, line: int | None, message: str) -> None:
        categories.append(category)
        location = "header" if line is None else f"line {line}"
        messages.append(f"{location}: {' '.join(message.split())[:180]}")

    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        return _RawParse(
            (),
            raw_digest.hexdigest(),
            grouping.hexdigest(),
            None,
            ("dilemmadata.raw_fingerprint_mismatch",),
            (f"source unreadable: {type(exc).__name__}",),
            0,
            0,
            0,
        )
    with handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        if len(header) != len(set(header)):
            issue("dilemmadata.duplicate_header_field", None, "duplicate header field")
        missing = sorted(set(required) - set(header))
        if missing:
            issue(
                "dilemmadata.missing_required_raw_field",
                None,
                f"missing required raw fields: {', '.join(missing[:12])}",
            )
        indices = {name: index for index, name in enumerate(header)}
        for line, values in enumerate(reader, start=2):
            row_count += 1
            if len(values) != len(header):
                issue(
                    "dilemmadata.row_width_mismatch",
                    line,
                    f"expected {len(header)} columns, observed {len(values)}",
                )
                continue
            row = {name: values[index] for name, index in indices.items()}
            try:
                if dialect == "an_joint":
                    onset = _fraction(row["s_offset_frac"])
                    duration = _fraction(row["s_duration_frac"])
                    pitch = int(row["s_midi"])
                    tie = _bool_value(row["s_isOnset"])
                    onset_ratio = _ratio(row["onset_div"], onset)
                    duration_ratio = _ratio(row["duration_div"], duration)
                    part_raw = (row.get("s_part_id") or "").strip()
                    voice_raw = (row.get("s_voice_id") or "").strip()
                    source_voice_key = (
                        part_raw or "<unknown-part>",
                        voice_raw or "<unknown-voice>",
                    )
                    staff = None
                    voice = _optional_int(voice_raw)
                    step_raw = (row.get("s_step") or "").strip()
                    alter_raw = (row.get("s_alter") or "").strip()
                    measure_key = (row.get("s_measure") or "").strip() or None
                    display_measure = (
                        (row.get("measureNumberWithSuffix") or "").strip() or None
                    )
                    anchor = None
                    mn_onset = (row.get("mn_onset") or "").strip()
                    if measure_key is not None and mn_onset not in _MISSING:
                        anchor = onset - 4 * _fraction(mn_onset)
                else:
                    onset = _fraction(row["quarterbeats_playthrough"])
                    source_duration = _fraction(row["duration"])
                    duration = source_duration * 4
                    pitch = int(row["pitch"])
                    tie = _bool_value(row["is_note_onset"])
                    onset_ratio = _ratio(row["onset_div"], onset)
                    duration_ratio_source = _ratio(row["duration_div"], source_duration)
                    duration_ratio = (
                        None
                        if duration_ratio_source is None
                        else duration_ratio_source / 4
                    )
                    staff_raw = (row.get("staff") or "").strip()
                    voice_raw = (row.get("voice") or "").strip()
                    source_voice_key = (
                        staff_raw or "<unknown-staff>",
                        voice_raw or "<unknown-voice>",
                    )
                    staff = _optional_int(staff_raw)
                    voice = _optional_int(voice_raw)
                    step_raw = (row.get("step") or "").strip()
                    alter_raw = (row.get("alter") or "").strip()
                    measure_key = (
                        (row.get("mn_playthrough") or row.get("mc_playthrough") or "")
                        .strip()
                        or None
                    )
                    display_measure = (row.get("mn") or "").strip() or measure_key
                    downbeat = (row.get("downbeat") or "").strip()
                    anchor = onset if measure_key is not None and downbeat == "1" else None
                if onset < 0 or duration < 0:
                    raise ValueError("negative onset or duration")
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                issue(
                    "dilemmadata.malformed_rational_time",
                    line,
                    f"raw note parse failed: {type(exc).__name__}: {exc}",
                )
                continue
            if not 0 <= pitch <= 127:
                issue(
                    "dilemmadata.pitch_out_of_range",
                    line,
                    f"pitch {pitch} is outside MIDI range",
                )
                continue
            if tie is None:
                issue(
                    "dilemmadata.malformed_tie_flag",
                    line,
                    "tie onset flag is neither true nor false",
                )
                continue
            if duration == 0:
                grace_count += 1
                if not tie:
                    issue(
                        "dilemmadata.grace_conflict",
                        line,
                        "zero-duration tie continuation is contradictory",
                    )
            if not tie:
                tie_count += 1
            if previous_onset is not None and onset < previous_onset:
                issue(
                    "dilemmadata.nonmonotonic_note_order",
                    line,
                    "note onset precedes the previous source row",
                )
            previous_onset = onset
            ratios = tuple(value for value in (onset_ratio, duration_ratio) if value is not None)
            if any(value <= 0 or value.denominator != 1 for value in ratios) or len(set(ratios)) > 1:
                issue(
                    "dilemmadata.resolution_mismatch",
                    line,
                    "division coordinates disagree with exact qn values",
                )
            for value in ratios:
                if value > 0 and value.denominator == 1:
                    resolutions.add(value.numerator)
            try:
                numerator = int(row["ts_beats"])
                denominator = int(row["ts_beat_type"])
                if numerator <= 0 or denominator <= 0 or denominator & (denominator - 1):
                    raise ValueError("invalid time signature")
            except (KeyError, ValueError) as exc:
                issue(
                    "dilemmadata.meter_conflict",
                    line,
                    f"meter parse failed: {exc}",
                )
                continue
            key_fifths = None
            try:
                key_fifths = _optional_int(row["ks_fifths"])
                if key_fifths is not None and not -7 <= key_fifths <= 7:
                    raise ValueError("key signature fifths outside [-7, 7]")
            except (KeyError, ValueError) as exc:
                issue(
                    "dilemmadata.missing_required_raw_field",
                    line,
                    f"key signature parse failed: {exc}",
                )
                continue
            spelling_step = None
            spelling_alter = None
            if step_raw not in _MISSING and alter_raw not in _MISSING:
                try:
                    candidate_alter = int(alter_raw)
                    if step_raw not in "ABCDEFG" or len(step_raw) != 1:
                        raise ValueError("invalid spelling step")
                    spelling_step = step_raw
                    spelling_alter = candidate_alter
                except ValueError:
                    spelling_step = None
                    spelling_alter = None
            if anchor is not None and anchor < 0:
                issue(
                    "dilemmadata.bar_reconstruction_failed",
                    line,
                    "measure anchor precedes piece origin",
                )
                continue
            observation = _RawObservation(
                ordinal=row_count - 1,
                line=line,
                onset=onset,
                duration=duration,
                pitch=pitch,
                tie_onset=tie,
                spelling_step=spelling_step,
                spelling_alter=spelling_alter,
                staff=staff,
                voice=voice,
                source_voice_key=source_voice_key,
                meter=(numerator, denominator),
                key_fifths=key_fifths,
                measure_key=measure_key,
                measure_anchor=anchor,
                display_measure=display_measure,
            )
            observations.append(observation)
            grouping.add(_note_hash(onset, duration, pitch))
            raw_digest.update(_canonical_bytes(_raw_projection_row(observation)))
            raw_digest.update(b"\n")
    if len(resolutions) != 1:
        issue(
            "dilemmadata.resolution_mismatch",
            None,
            f"expected one positive source resolution, observed {sorted(resolutions)}",
        )
    unique_categories = tuple(sorted(set(categories)))
    bounded_messages = tuple(messages[:16])
    return _RawParse(
        observations=tuple(observations),
        raw_projection_sha256=raw_digest.hexdigest(),
        grouping_fingerprint=grouping.hexdigest(),
        source_resolution=next(iter(resolutions)) if len(resolutions) == 1 else None,
        categories=unique_categories,
        messages=bounded_messages,
        note_row_count=row_count,
        tie_continuation_row_count=tie_count,
        zero_duration_row_count=grace_count,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _piece_token(record_id: str) -> str:
    return sha256(record_id.encode("utf-8")).hexdigest()[:24]


def _piece_id(record_id: str, dialect: str) -> str:
    label = "an" if dialect == "an_joint" else "dlc"
    return f"piece:dilemmadata-{label}-{_piece_token(record_id)}"


def _inventory(root: Path) -> tuple[tuple[Path, ...], int, str]:
    files = tuple(sorted(_safe_files(root), key=lambda path: _relative(path, root)))
    rows: list[list[object]] = []
    byte_count = 0
    for path in files:
        size = path.stat().st_size
        byte_count += size
        rows.append([_relative(path, root), size, _hash_file(path)])
    return files, byte_count, _fingerprint(rows)


def _release_version(root: Path) -> str | None:
    try:
        value = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    version = value.get("version") if isinstance(value, dict) else None
    return version if isinstance(version, str) else None


def discover_dilemmadata_corpus(
    root: str | PathLike[str],
    *,
    identity: DilemmadataCorpusIdentity = DilemmadataCorpusIdentity(),
    require_valid: bool = True,
) -> DilemmadataCorpusDiscovery:
    """Discover both dialects, reconstruct grouping, and verify corpus identity."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise DilemmadataAdapterError(
            "Dilemmadata root is not a directory",
            category="dilemmadata.corpus_identity",
        )
    root_path = root_path.resolve()
    files, byte_count, content_fingerprint = _inventory(root_path)
    records_data: list[dict[str, object]] = []
    primary_paths: set[Path] = set()

    an_root = root_path / "pitch_arrays" / "AN"
    if an_root.is_dir():
        for path in sorted(an_root.rglob("*_joint.tsv")):
            split = path.parent.name
            piece = path.name[: -len("_joint.tsv")]
            score_candidates = tuple(
                candidate
                for suffix in (".mxl", ".musicxml")
                if (candidate := path.parent / f"{piece}{suffix}").is_file()
            )
            score = score_candidates[0] if len(score_candidates) == 1 else None
            records_data.append(
                {
                    "record_id": f"an:{split}:{piece}",
                    "dialect": "an_joint",
                    "path": path.resolve(),
                    "relative_path": _relative(path, root_path),
                    "collection": piece.split("-", 1)[0],
                    "piece_name": piece,
                    "suggested_split": split,
                    "score_path": None if score is None else score.resolve(),
                    "score_relative_path": None if score is None else _relative(score, root_path),
                }
            )
            primary_paths.add(path.resolve())
    dlc_root = root_path / "pitch_arrays" / "DLC"
    if dlc_root.is_dir():
        for path in sorted(dlc_root.rglob("*.tsv")):
            relative = path.relative_to(dlc_root)
            collection = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            records_data.append(
                {
                    "record_id": f"dlc:{collection}:{path.stem}",
                    "dialect": "dlc",
                    "path": path.resolve(),
                    "relative_path": _relative(path, root_path),
                    "collection": collection,
                    "piece_name": path.stem,
                    "suggested_split": None,
                    "score_path": None,
                    "score_relative_path": None,
                }
            )
            primary_paths.add(path.resolve())

    all_pitch_tsv = {
        path.resolve() for path in (root_path / "pitch_arrays").rglob("*.tsv")
    } if (root_path / "pitch_arrays").is_dir() else set()
    auxiliary = tuple(sorted(_relative(path, root_path) for path in all_pitch_tsv - primary_paths))
    unexpected = tuple(
        path
        for path in auxiliary
        if not (path.endswith("_slices.tsv") or path.endswith("dataset_summary.tsv"))
    )

    dlc_metadata = {
        ((row.get("corpus") or "").strip(), (row.get("piece") or "").strip()):
        ((row.get("split") or "").strip() or None)
        for row in _read_rows(
            root_path / "processing" / "DLC" / "distant_listening_corpus.metadata.tsv"
        )
    }
    scans: dict[str, _RawParse] = {}
    physical: dict[str, str] = {}
    scores: dict[str, str | None] = {}
    for row in records_data:
        record_id = str(row["record_id"])
        dialect = str(row["dialect"])
        parse = _parse_raw_file(
            Path(row["path"]),
            "an_joint" if dialect == "an_joint" else "dlc",
        )
        scans[record_id] = parse
        physical[record_id] = _hash_file(Path(row["path"]))
        score_path = row["score_path"]
        scores[record_id] = _hash_file(Path(score_path)) if score_path is not None else None
        if dialect == "dlc":
            row["suggested_split"] = dlc_metadata.get(
                (str(row["collection"]), str(row["piece_name"]))
            )

    ids = tuple(str(row["record_id"]) for row in records_data)
    union = _UnionFind(ids)
    for selector in (
        lambda record_id: scans[record_id].grouping_fingerprint,
        lambda record_id: scores[record_id] if record_id.startswith("an:") else None,
    ):
        groups: dict[str, list[str]] = defaultdict(list)
        for record_id in ids:
            key = selector(record_id)
            if key:
                groups[key].append(record_id)
        for members in groups.values():
            for other in sorted(members)[1:]:
                union.union(sorted(members)[0], other)

    an_lookup = {
        str(row["piece_name"]): str(row["record_id"])
        for row in records_data
        if row["dialect"] == "an_joint"
    }
    dlc_lookup = {
        (str(row["collection"]), str(row["piece_name"])): str(row["record_id"])
        for row in records_data
        if row["dialect"] == "dlc"
    }
    explicit_overlap_count = 0
    for row in _read_rows(root_path / "processing" / "merged_summary.tsv"):
        an_id = an_lookup.get((row.get("id_v100") or "").strip())
        dlc_id = dlc_lookup.get(
            ((row.get("corpus_dlc") or "").strip(), (row.get("piece") or "").strip())
        )
        if an_id and dlc_id:
            union.union(an_id, dlc_id)
            explicit_overlap_count += 1

    components: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(ids):
        components[union.find(record_id)].append(record_id)
    component_identity: dict[str, tuple[str, str]] = {}
    split_conflicts = 0
    suggested_by_id = {
        str(row["record_id"]): row["suggested_split"] for row in records_data
    }
    for members in components.values():
        ordered = sorted(members)
        digest = _fingerprint(ordered)
        source_group = f"dilemmadata-component:{digest}"
        lineage = f"dilemmadata-lineage:{digest}"
        for record_id in ordered:
            component_identity[record_id] = (source_group, lineage)
        splits = {suggested_by_id[record_id] for record_id in ordered if suggested_by_id[record_id]}
        split_conflicts += int(len(splits) > 1)

    records: list[DilemmadataCorpusRecord] = []
    for row in records_data:
        record_id = str(row["record_id"])
        parse = scans[record_id]
        source_group, lineage = component_identity[record_id]
        dialect = "an_joint" if row["dialect"] == "an_joint" else "dlc"
        record = DilemmadataCorpusRecord(
            record_id=record_id,
            piece_id=_piece_id(record_id, dialect),
            dialect=dialect,
            path=Path(row["path"]),
            relative_path=str(row["relative_path"]),
            collection=str(row["collection"]),
            piece_name=str(row["piece_name"]),
            suggested_split=(
                str(row["suggested_split"])
                if row["suggested_split"] is not None
                else None
            ),
            physical_source_sha256=physical[record_id],
            raw_projection_sha256=parse.raw_projection_sha256,
            raw_equivalence_id=f"dilemmadata-raw:{parse.raw_projection_sha256}",
            grouping_fingerprint=parse.grouping_fingerprint,
            source_group_id=source_group,
            lineage_group_id=lineage,
            source_resolution=parse.source_resolution,
            score_path=(
                None if row["score_path"] is None else Path(row["score_path"])
            ),
            score_relative_path=(
                None
                if row["score_relative_path"] is None
                else str(row["score_relative_path"])
            ),
            score_sha256=scores[record_id],
            raw_issue_categories=parse.categories,
            note_row_count=parse.note_row_count,
            tie_continuation_row_count=parse.tie_continuation_row_count,
            zero_duration_row_count=parse.zero_duration_row_count,
            corpus_identity=identity,
            record_binding_version=DILEMMADATA_RECORD_BINDING_VERSION,
            record_binding_sha256="",
        )
        records.append(_bind_record(record))

    release = _release_version(root_path)
    counts = Counter(record.dialect for record in records)
    issues: list[DilemmadataCorpusIssue] = []
    if unexpected:
        issues.append(
            DilemmadataCorpusIssue(
                "dilemmadata.unknown_dialect",
                None,
                unexpected[:16],
                f"unclassified pitch-array TSV paths: {len(unexpected)}",
            )
        )
    identity_mismatches: list[str] = []
    observed = {
        "version": DILEMMADATA_CORPUS_IDENTITY_VERSION,
        "release_version": release,
        "installation_file_count": len(files),
        "content_fingerprint": content_fingerprint,
        "primary_record_count": len(records),
        "an_record_count": counts["an_joint"],
        "dlc_record_count": counts["dlc"],
    }
    for field in (
        "version",
        "release_version",
        "installation_file_count",
        "content_fingerprint",
        "primary_record_count",
        "an_record_count",
        "dlc_record_count",
    ):
        expected = getattr(identity, field)
        if observed[field] != expected:
            identity_mismatches.append(field)
    if identity.release_commit != DILEMMADATA_RELEASE_COMMIT:
        identity_mismatches.append("release_commit_config")
    if identity_mismatches:
        issues.append(
            DilemmadataCorpusIssue(
                "dilemmadata.corpus_identity",
                None,
                (),
                f"identity mismatches: {', '.join(sorted(identity_mismatches))}",
            )
        )
    record_ids = tuple(record.record_id for record in records)
    if len(record_ids) != len(set(record_ids)):
        issues.append(
            DilemmadataCorpusIssue(
                "dilemmadata.duplicate_record_identity",
                None,
                (),
                "duplicate record identities discovered",
            )
        )
    discovery = DilemmadataCorpusDiscovery(
        root=root_path,
        records=tuple(sorted(records, key=lambda record: record.record_id)),
        content_fingerprint=content_fingerprint,
        installation_file_count=len(files),
        installation_byte_count=byte_count,
        release_version=release,
        issues=tuple(
            sorted(issues, key=lambda issue: (issue.category, issue.record_id or "", issue.paths))
        ),
        auxiliary_tsv_paths=auxiliary,
        component_count=len(components),
        multi_record_component_count=sum(len(members) > 1 for members in components.values()),
        explicit_overlap_count=explicit_overlap_count,
        suggested_split_conflict_count=split_conflicts,
        corpus_identity=identity,
    )
    if require_valid and not discovery.is_valid:
        raise DilemmadataCorpusIdentityError(discovery)
    return discovery


def _rt(value: Fraction) -> RationalTime:
    return RationalTime(value.numerator, value.denominator)


def _active_meter(
    events: Sequence[tuple[Fraction, tuple[int, int]]], time: Fraction
) -> tuple[Fraction, tuple[int, int]]:
    active = events[0]
    for event in events:
        if event[0] > time:
            break
        active = event
    return active


def _meter_events_and_anchors(
    observations: Sequence[_RawObservation],
) -> tuple[
    tuple[tuple[Fraction, tuple[int, int]], ...],
    tuple[Fraction, ...],
    Fraction | None,
    tuple[str, ...],
]:
    by_onset: dict[Fraction, set[tuple[int, int]]] = defaultdict(set)
    for row in observations:
        by_onset[row.onset].add(row.meter)
    if any(len(values) != 1 for values in by_onset.values()):
        return (), (), None, ("dilemmadata.meter_conflict",)
    ordered = tuple((onset, next(iter(values))) for onset, values in sorted(by_onset.items()))
    if not ordered:
        return (), (), None, ("dilemmadata.meter_conflict",)
    anchors_by_measure: dict[str, set[Fraction]] = defaultdict(set)
    for row in observations:
        if row.measure_key is not None and row.measure_anchor is not None:
            anchors_by_measure[row.measure_key].add(row.measure_anchor)
    if any(len(values) != 1 for values in anchors_by_measure.values()):
        return (), (), None, ("dilemmadata.bar_reconstruction_failed",)
    anchors = tuple(sorted({next(iter(values)) for values in anchors_by_measure.values()}))
    if anchors and anchors[0] != 0:
        return (), (), None, ("dilemmadata.bar_reconstruction_failed",)

    first_meter = ordered[0][1]
    events: list[tuple[Fraction, tuple[int, int]]] = [(Fraction(0), first_meter)]
    current = first_meter
    anchor_set = set(anchors)
    for onset, meter in ordered:
        if meter == current:
            continue
        matching_anchors = [
            row.measure_anchor
            for row in observations
            if row.onset == onset and row.meter == meter and row.measure_anchor is not None
        ]
        event_onset = matching_anchors[0] if matching_anchors else onset
        if matching_anchors and len(set(matching_anchors)) != 1:
            return (), (), None, ("dilemmadata.meter_conflict",)
        if event_onset < events[-1][0]:
            return (), (), None, ("dilemmadata.meter_conflict",)
        events.append((event_onset, meter))
        anchor_set.add(event_onset)
        current = meter

    numerator, denominator = first_meter
    nominal = Fraction(numerator * 4, denominator)
    pickup_end = None
    if len(anchors) >= 2 and 0 < anchors[1] < nominal:
        pickup_end = anchors[1]
    return tuple(events), tuple(sorted(anchor_set)), pickup_end, ()


def _metric_grid(
    duration: Fraction,
    meter_evidence: Sequence[tuple[Fraction, tuple[int, int]]],
    anchors: Sequence[Fraction],
    pickup_end: Fraction | None,
    token: str,
) -> tuple[
    tuple[MeterEvent, ...],
    tuple[CanonicalBar, ...],
    tuple[CanonicalBeat, ...],
    tuple[str, ...],
]:
    event_by_onset: dict[Fraction, tuple[int, int]] = {}
    for onset, meter in meter_evidence:
        existing = event_by_onset.get(onset)
        if existing is not None and existing != meter:
            return (), (), (), ("dilemmadata.meter_conflict",)
        event_by_onset[onset] = meter
    events = tuple(sorted(event_by_onset.items()))
    meter_events = tuple(
        MeterEvent(
            meter_event_id=f"meter:dilemmadata-{token}-{index:04d}",
            onset_qn=_rt(onset),
            numerator=meter[0],
            denominator=meter[1],
            provenance_id="prov:dilemmadata-conversion",
        )
        for index, (onset, meter) in enumerate(events)
    )
    boundaries: list[Fraction] = [Fraction(0)]
    if pickup_end is not None:
        boundaries.append(pickup_end)
    cursor = boundaries[-1]
    event_onsets = [onset for onset, _meter in events[1:] if onset < duration]
    for endpoint in [*event_onsets, duration]:
        if endpoint < cursor:
            return (), (), (), ("dilemmadata.bar_reconstruction_failed",)
        while cursor < endpoint:
            _meter_onset, meter = _active_meter(events, cursor)
            nominal = Fraction(meter[0] * 4, meter[1])
            proposed = cursor + nominal
            boundary = min(proposed, endpoint)
            if boundary <= cursor:
                return (), (), (), ("dilemmadata.bar_reconstruction_failed",)
            boundaries.append(boundary)
            cursor = boundary
    boundary_set = set(boundaries)
    if any(anchor <= duration and anchor not in boundary_set for anchor in anchors):
        return (), (), (), ("dilemmadata.bar_reconstruction_failed",)

    bars: list[CanonicalBar] = []
    beats: list[CanonicalBeat] = []
    meter_id_by_onset = {
        onset: meter_events[index].meter_event_id
        for index, (onset, _meter) in enumerate(events)
    }
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        meter_onset, meter = _active_meter(events, start)
        numerator, denominator = meter
        nominal = Fraction(numerator * 4, denominator)
        actual = end - start
        is_pickup = index == 0 and pickup_end is not None
        metric_offset = nominal - actual if is_pickup else Fraction(0)
        beat_unit = Fraction(4, denominator)
        if metric_offset / beat_unit != int(metric_offset / beat_unit):
            return (), (), (), ("dilemmadata.bar_reconstruction_failed",)
        bar_id = f"bar:dilemmadata-{token}-{index:06d}"
        bars.append(
            CanonicalBar(
                bar_id=bar_id,
                index=index,
                start_qn=_rt(start),
                duration_qn=_rt(actual),
                meter_event_id=meter_id_by_onset[meter_onset],
                metric_offset_qn=_rt(metric_offset),
                is_pickup=is_pickup,
                is_incomplete=actual < nominal,
                display_number=str(index + (0 if is_pickup else 1)),
                provenance_id="prov:dilemmadata-conversion",
            )
        )
        position = metric_offset
        beat_index = int(metric_offset / beat_unit)
        while position - metric_offset < actual:
            elapsed = position - metric_offset
            beat_duration = min(beat_unit, actual - elapsed)
            beats.append(
                CanonicalBeat(
                    beat_id=f"beat:dilemmadata-{token}-{len(beats):08d}",
                    bar_id=bar_id,
                    meter_event_id=meter_id_by_onset[meter_onset],
                    index_in_bar=beat_index,
                    start_qn=_rt(start + elapsed),
                    duration_qn=_rt(beat_duration),
                    position_in_bar_qn=_rt(position),
                    is_downbeat=beat_index == 0,
                    strength=1.0 if beat_index == 0 else 0.5,
                    provenance_id="prov:dilemmadata-conversion",
                )
            )
            if len(bars) + len(beats) > _MAX_METRIC_RECORDS:
                return (), (), (), ("dilemmadata.bar_reconstruction_failed",)
            position += beat_unit
            beat_index += 1
    return meter_events, tuple(bars), tuple(beats), ()


def _merge_notes(
    observations: Sequence[_RawObservation], token: str, track_id: str
) -> tuple[
    tuple[CanonicalNote, ...],
    int,
    tuple[str, ...],
    tuple[DilemmadataSourceRowBinding, ...],
]:
    builders: list[dict[str, object]] = []
    by_pitch_voice: dict[tuple[int, tuple[str, ...]], list[int]] = defaultdict(list)
    merge_count = 0
    categories: list[str] = []
    for row in observations:
        key = (row.pitch, row.source_voice_key)
        if row.tie_onset:
            builders.append(
                {
                    "first_ordinal": row.ordinal,
                    "onset": row.onset,
                    "duration": row.duration,
                    "pitch": row.pitch,
                    "spelling_step": row.spelling_step,
                    "spelling_alter": row.spelling_alter,
                    "staff": row.staff,
                    "voice": row.voice,
                    "source_rows": [row],
                }
            )
            by_pitch_voice[key].append(len(builders) - 1)
            continue
        candidates = [
            index
            for index in by_pitch_voice.get(key, ())
            if builders[index]["onset"] + builders[index]["duration"] == row.onset
            and builders[index]["duration"] > 0
        ]
        if not candidates:
            categories.append("dilemmadata.tie_predecessor_missing")
            continue
        if len(candidates) != 1:
            categories.append("dilemmadata.tie_timing_conflict")
            continue
        builder = builders[candidates[0]]
        builder["duration"] = builder["duration"] + row.duration
        builder["source_rows"].append(row)
        merge_count += 1
    if categories:
        return (), merge_count, tuple(sorted(set(categories))), ()
    ordered = sorted(
        builders,
        key=lambda row: (
            row["onset"],
            row["pitch"],
            row["duration"],
            row["first_ordinal"],
        ),
    )
    notes = tuple(
        CanonicalNote(
            note_id=f"note:dilemmadata-{token}-{int(row['first_ordinal']):08d}",
            track_id=track_id,
            pitch=int(row["pitch"]),
            onset_qn=_rt(row["onset"]),
            duration_qn=_rt(row["duration"]),
            velocity=None,
            channel=None,
            program=None,
            is_percussion=False,
            is_grace=row["duration"] == 0,
            spelling_step=row["spelling_step"],
            spelling_alter=row["spelling_alter"],
            staff=row["staff"],
            voice=row["voice"],
            articulations=None,
            dynamic=None,
            source_onset_ticks=None,
            source_duration_ticks=None,
            source_onset_seconds=None,
            source_duration_seconds=None,
            provenance_id="prov:dilemmadata-conversion",
        )
        for row in ordered
    )
    row_bindings = tuple(
        sorted(
            (
                DilemmadataSourceRowBinding(
                    ordinal=source_row.ordinal,
                    line=source_row.line,
                    onset_qn=_rt(source_row.onset),
                    canonical_note_id=(
                        f"note:dilemmadata-{token}-{int(row['first_ordinal']):08d}"
                    ),
                    tie_continuation=not source_row.tie_onset,
                )
                for row in ordered
                for source_row in row["source_rows"]
            ),
            key=lambda binding: binding.ordinal,
        )
    )
    return notes, merge_count, (), row_bindings


def _key_events(
    observations: Sequence[_RawObservation], token: str
) -> tuple[tuple[KeySignatureEvent, ...], tuple[str, ...]]:
    by_onset: dict[Fraction, set[int | None]] = defaultdict(set)
    for row in observations:
        by_onset[row.onset].add(row.key_fifths)
    if any(len(values) != 1 for values in by_onset.values()):
        return (), ("dilemmadata.key_signature_conflict",)
    first = next(iter(by_onset[min(by_onset)]))
    if first is None:
        return (), ()
    values: list[tuple[Fraction, int]] = [(Fraction(0), first)]
    current = first
    for onset, candidates in sorted(by_onset.items()):
        value = next(iter(candidates))
        if value is None:
            continue
        if value != current:
            values.append((onset, value))
            current = value
    return (
        tuple(
            KeySignatureEvent(
                key_signature_event_id=f"keysig:dilemmadata-{token}-{index:04d}",
                onset_qn=_rt(onset),
                fifths=fifths,
                mode="unknown",
                raw_value=None,
                provenance_id="prov:dilemmadata-conversion",
            )
            for index, (onset, fifths) in enumerate(values)
        ),
        (),
    )


def _quarantine(
    record: DilemmadataCorpusRecord,
    categories: Iterable[str],
    messages: Iterable[str] = (),
) -> DilemmadataQuarantine:
    return DilemmadataQuarantine(
        status="quarantined",
        record=record,
        categories=tuple(sorted(set(categories))),
        messages=tuple(" ".join(message.split())[:220] for message in tuple(messages)[:16]),
    )


def convert_dilemmadata_record(
    record: DilemmadataCorpusRecord,
    *,
    config: DilemmadataAdapterConfig = DilemmadataAdapterConfig(),
) -> DilemmadataConversionResult:
    """Convert one discovered record without reading any theory column."""

    if not isinstance(record, DilemmadataCorpusRecord):
        raise DilemmadataConversionError(
            "record must be DilemmadataCorpusRecord",
            category="dilemmadata.record_invalid",
        )
    if not isinstance(config, DilemmadataAdapterConfig):
        raise DilemmadataConversionError(
            "config must be DilemmadataAdapterConfig",
            category="dilemmadata.config_invalid",
        )
    if not _record_binding_is_valid(record):
        return _quarantine(
            record,
            ("dilemmadata.record_binding_mismatch",),
            ("record does not match its versioned discovery binding",),
        )
    parse = _parse_raw_file(record.path, record.dialect)
    if parse.categories:
        return _quarantine(record, parse.categories, parse.messages)
    if parse.raw_projection_sha256 != record.raw_projection_sha256:
        return _quarantine(
            record,
            ("dilemmadata.raw_fingerprint_mismatch",),
            ("target-independent raw projection changed after discovery",),
        )
    if parse.source_resolution != record.source_resolution:
        return _quarantine(
            record,
            ("dilemmadata.resolution_mismatch",),
            ("source resolution changed after discovery",),
        )
    token = _piece_token(record.record_id)
    track_id = f"track:dilemmadata-{token}-raw"
    notes, tie_merges, note_categories, row_bindings = _merge_notes(
        parse.observations,
        token,
        track_id,
    )
    if note_categories:
        return _quarantine(record, note_categories)
    duration = max(
        (note.onset_qn.to_fraction() + note.duration_qn.to_fraction() for note in notes),
        default=Fraction(0),
    )
    meter_evidence, anchors, pickup_end, meter_categories = _meter_events_and_anchors(
        parse.observations
    )
    if meter_categories:
        return _quarantine(record, meter_categories)
    meters, bars, beats, bar_categories = _metric_grid(
        duration, meter_evidence, anchors, pickup_end, token
    )
    if bar_categories:
        return _quarantine(record, bar_categories)
    key_events, key_categories = _key_events(parse.observations, token)
    if key_categories:
        return _quarantine(record, key_categories)

    source_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-source",
        kind="source",
        source="dilemmadata_v1",
        record_id=record.record_id,
        uri=record.relative_path,
        version=DILEMMADATA_RELEASE_VERSION,
        checksum_sha256=None,
        created_at=None,
        parents=(),
        details=(
            ("dialect", record.dialect),
            ("lineage_group_id", record.lineage_group_id),
            ("physical_checksum_external", True),
            ("source_group_id", record.source_group_id),
        ),
    )
    conversion_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-conversion",
        kind="conversion",
        source="music_critic.dilemmadata_adapter",
        record_id=None,
        uri=None,
        version=DILEMMADATA_ADAPTER_VERSION,
        checksum_sha256=parse.raw_projection_sha256,
        created_at=None,
        parents=("prov:dilemmadata-source",),
        details=(
            ("grace_policy", config.grace_policy),
            ("meter_policy", config.meter_policy),
            ("raw_projection_version", DILEMMADATA_RAW_PROJECTION_VERSION),
            ("tie_policy", config.tie_policy),
            ("track_policy", config.track_policy),
        ),
    )
    tempo_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-default-tempo",
        kind="default",
        source="music_critic.canonical_default",
        record_id=None,
        uri=None,
        version=config.default_policy,
        checksum_sha256=None,
        created_at=None,
        parents=("prov:dilemmadata-conversion",),
        details=(
            (
                "microseconds_per_quarter",
                config.default_tempo_microseconds_per_quarter,
            ),
            ("source_observed", False),
        ),
    )
    percussion_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-default-percussion",
        kind="default",
        source="music_critic.canonical_default",
        record_id=None,
        uri=None,
        version=config.default_policy,
        checksum_sha256=None,
        created_at=None,
        parents=("prov:dilemmadata-conversion",),
        details=(("is_percussion", False), ("source_observed", False)),
    )
    flags = [
        QualityFlag(
            code="dilemmadata.default_tempo_inserted",
            severity="info",
            message="Tempo is absent in Dilemmadata; the canonical default is not an observation.",
            entity_ids=(record.piece_id,),
            provenance_id="prov:dilemmadata-default-tempo",
        ),
        QualityFlag(
            code="dilemmadata.percussion_unknown_default_false",
            severity="warning",
            message=(
                "Percussion identity is absent; schema-required false is conservative "
                "and is not a semantic instrument claim."
            ),
            entity_ids=(record.piece_id, track_id),
            provenance_id="prov:dilemmadata-default-percussion",
        ),
    ]
    if tie_merges:
        flags.append(
            QualityFlag(
                code="dilemmadata.tie_continuations_merged",
                severity="info",
                message=f"Merged {tie_merges} exact contiguous tie continuation rows.",
                entity_ids=(record.piece_id,),
                provenance_id="prov:dilemmadata-conversion",
            )
        )
    grace_count = sum(note.is_grace for note in notes)
    if grace_count:
        flags.append(
            QualityFlag(
                code="dilemmadata.zero_duration_grace_retained",
                severity="info",
                message=f"Retained {grace_count} source-zero-duration grace observations.",
                entity_ids=(record.piece_id,),
                provenance_id="prov:dilemmadata-conversion",
            )
        )
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=record.piece_id,
        dataset_name=config.dataset_name,
        source_group_id=record.source_group_id,
        split=None,
        source_path=record.relative_path,
        source_resolution=parse.source_resolution,
        duration_qn=_rt(duration),
        metadata=PieceMetadata(
            source_format="tsv",
            title=None,
            creators=None,
            collection=record.collection,
            movement_title=None,
            movement_number=None,
            genres=None,
            copyright=None,
            language=None,
        ),
        tracks=(
            CanonicalTrack(
                track_id=track_id,
                source_track_index=0,
                name=None,
                instrument_name=None,
                program=None,
                channel=None,
                is_percussion=False,
                provenance_id="prov:dilemmadata-default-percussion",
            ),
        ),
        notes=notes,
        bars=bars,
        beats=beats,
        tempo_events=(
            TempoEvent(
                tempo_event_id=f"tempo:dilemmadata-{token}-default",
                onset_qn=RationalTime(0),
                microseconds_per_quarter=config.default_tempo_microseconds_per_quarter,
                provenance_id="prov:dilemmadata-default-tempo",
            ),
        ),
        meter_events=meters,
        key_signature_events=key_events,
        annotations=(),
        targets=(),
        provenance=(
            source_provenance,
            conversion_provenance,
            percussion_provenance,
            tempo_provenance,
        ),
        quality_flags=tuple(
            sorted(flags, key=lambda flag: (flag.code, flag.entity_ids, flag.message))
        ),
    )
    report = validate_piece(piece)
    if report.errors:
        examples = tuple(
            f"{issue.code}@{issue.path}" for issue in report.errors[:8]
        )
        return _quarantine(
            record,
            ("dilemmadata.canonical_validation_failed",),
            examples,
        )
    statistics = DilemmadataConversionStatistics(
        source_note_row_count=parse.note_row_count,
        canonical_note_count=len(notes),
        tie_continuation_row_count=parse.tie_continuation_row_count,
        tie_merge_count=tie_merges,
        grace_note_count=grace_count,
        meter_event_count=len(meters),
        bar_count=len(bars),
        beat_count=len(beats),
        pickup_bar_count=sum(bar.is_pickup for bar in bars),
        incomplete_bar_count=sum(bar.is_incomplete for bar in bars),
    )
    accepted_record = _bind_record(
        replace(record, physical_source_sha256=_hash_file(record.path))
    )
    return DilemmadataAccepted(
        status="accepted",
        record=accepted_record,
        piece=piece,
        validation_report=report,
        statistics=statistics,
        alignment_evidence=_make_alignment_evidence(
            accepted_record,
            piece,
            row_bindings,
        ),
    )


def iter_dilemmadata_corpus(
    root: str | PathLike[str],
    *,
    config: DilemmadataAdapterConfig = DilemmadataAdapterConfig(),
    identity: DilemmadataCorpusIdentity = DilemmadataCorpusIdentity(),
) -> Iterator[DilemmadataConversionResult]:
    """Verify full identity once, then emit exactly one outcome per record."""

    discovery = discover_dilemmadata_corpus(root, identity=identity, require_valid=True)
    for record in discovery.records:
        yield convert_dilemmadata_record(record, config=config)


__all__ = [
    "DILEMMADATA_ACCEPTANCE_REPORT_VERSION",
    "DILEMMADATA_AN_RECORD_COUNT",
    "DILEMMADATA_ADAPTER_VERSION",
    "DILEMMADATA_CONTENT_FINGERPRINT",
    "DILEMMADATA_CORPUS_IDENTITY_VERSION",
    "DILEMMADATA_DATASET_NAME",
    "DILEMMADATA_DLC_RECORD_COUNT",
    "DILEMMADATA_GROUPING_VERSION",
    "DILEMMADATA_INSTALLATION_FILE_COUNT",
    "DILEMMADATA_PRIMARY_RECORD_COUNT",
    "DILEMMADATA_PRODUCTION_MANIFEST_VERSION",
    "DILEMMADATA_RAW_PROJECTION_VERSION",
    "DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION",
    "DILEMMADATA_RECORD_BINDING_VERSION",
    "DILEMMADATA_RELEASE_COMMIT",
    "DILEMMADATA_RELEASE_VERSION",
    "DilemmadataAccepted",
    "DilemmadataAdapterConfig",
    "DilemmadataAdapterError",
    "DilemmadataConversionError",
    "DilemmadataConversionResult",
    "DilemmadataConversionStatistics",
    "DilemmadataCorpusDiscovery",
    "DilemmadataCorpusIdentity",
    "DilemmadataCorpusIdentityError",
    "DilemmadataCorpusIssue",
    "DilemmadataCorpusRecord",
    "DilemmadataQuarantine",
    "DilemmadataRawTargetAlignmentEvidence",
    "DilemmadataSourceRowBinding",
    "convert_dilemmadata_record",
    "discover_dilemmadata_corpus",
    "iter_dilemmadata_corpus",
    "validate_dilemmadata_alignment_evidence",
    "validate_dilemmadata_record_binding",
]
