"""Production Phase 9B.2A source-native Dilemmadata target sidecars."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Literal

from music_critic.adapters.dilemmadata import (
    DILEMMADATA_DATASET_NAME,
    DILEMMADATA_RELEASE_VERSION,
    DilemmadataAccepted,
    DilemmadataAdapterConfig,
    DilemmadataCorpusIdentity,
    DilemmadataCorpusRecord,
    DilemmadataRawTargetAlignmentEvidence,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
    validate_dilemmadata_alignment_evidence,
    validate_dilemmadata_record_binding,
)
from music_critic.data import (
    AnnotationSpan,
    CanonicalPiece,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_FAMILIES,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
    DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
    DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
    DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
    DILEMMADATA_TASK_IDS_BY_DIALECT,
    DilemmadataSourceFamilySpec,
)
from music_critic.tasks.multisource import (
    TARGET_BUNDLE_CONTRACT_VERSION,
    SampleTarget,
    TargetBundle,
    target_bundle_fingerprint,
)
from music_critic.tasks.registry import DILEMMADATA_TARGET_REGISTRY_ID


DILEMMADATA_TARGET_ADAPTER_VERSION = "1.1.0"
DILEMMADATA_REMEDIATED_TARGET_ADAPTER_VERSION = "1.2.0"
DILEMMADATA_TARGET_SIDECAR_VERSION = "1.0.0"
DILEMMADATA_TARGET_AUDIT_REPORT_VERSION = "1.1.0"
DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION = "1.1.0"
DILEMMADATA_TARGET_METADATA_VERSION = "1.0.0"

_COLUMN_POLICY = "phase9a_evidenced_target_columns_only_v1"
_SPAN_POLICY = "exact_source_identity_next_boundary_no_terminal_inference_v1"
_POINT_POLICY = "exact_rational_onset_no_snap_v1"
_TIE_POLICY = "all_merged_source_rows_must_agree_v1"
_DUPLICATE_POLICY = "merge_equal_mask_conflicts_v1"
_MISSING = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
_TRUE = frozenset({"1", "True", "true", "TRUE"})
_FALSE = frozenset({"0", "False", "false", "FALSE"})

TargetState = Literal["available", "masked", "missing", "unsupported"]


class DilemmadataTargetAdapterError(ValueError):
    """Invalid API/configuration use at the target-only boundary."""

    def __init__(self, message: str, *, category: str) -> None:
        self.category = category
        super().__init__(f"[{category}] {message}")


@dataclass(frozen=True, slots=True)
class DilemmadataTargetAdapterConfig:
    target_column_policy: Literal[
        "phase9a_evidenced_target_columns_only_v1"
    ] = _COLUMN_POLICY
    span_policy: Literal[
        "exact_source_identity_next_boundary_no_terminal_inference_v1"
    ] = _SPAN_POLICY
    point_policy: Literal["exact_rational_onset_no_snap_v1"] = _POINT_POLICY
    tie_policy: Literal[
        "all_merged_source_rows_must_agree_v1"
    ] = _TIE_POLICY
    duplicate_policy: Literal[
        "merge_equal_mask_conflicts_v1"
    ] = _DUPLICATE_POLICY

    def __post_init__(self) -> None:
        for field_name, value, implemented in (
            ("target_column_policy", self.target_column_policy, _COLUMN_POLICY),
            ("span_policy", self.span_policy, _SPAN_POLICY),
            ("point_policy", self.point_policy, _POINT_POLICY),
            ("tie_policy", self.tie_policy, _TIE_POLICY),
            ("duplicate_policy", self.duplicate_policy, _DUPLICATE_POLICY),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, str)
                or value != implemented
            ):
                raise DilemmadataTargetAdapterError(
                    f"{field_name} must select the implemented policy",
                    category="dilemmadata.target.config_invalid",
                )


@dataclass(frozen=True, slots=True)
class DilemmadataTargetMetadata:
    record_id: str
    fields: tuple[tuple[str, str], ...]
    ambiguous_fields: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DilemmadataTargetMetadataIndex:
    version: str
    records: tuple[DilemmadataTargetMetadata, ...]
    source_files: tuple[tuple[str, str], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.version != DILEMMADATA_TARGET_METADATA_VERSION:
            raise DilemmadataTargetAdapterError(
                "target metadata index version is incompatible",
                category="dilemmadata.target.metadata_invalid",
            )
        record_ids = tuple(record.record_id for record in self.records)
        if record_ids != tuple(sorted(record_ids)) or len(record_ids) != len(
            set(record_ids)
        ):
            raise DilemmadataTargetAdapterError(
                "target metadata records must be unique and sorted",
                category="dilemmadata.target.metadata_invalid",
            )
        if self.source_files != tuple(sorted(self.source_files)):
            raise DilemmadataTargetAdapterError(
                "target metadata source files must be sorted",
                category="dilemmadata.target.metadata_invalid",
            )
        if self.fingerprint != _target_metadata_index_fingerprint(self):
            raise DilemmadataTargetAdapterError(
                "target metadata index fingerprint differs from its contents",
                category="dilemmadata.target.metadata_invalid",
            )

    def for_record(self, record_id: str) -> DilemmadataTargetMetadata | None:
        return next(
            (record for record in self.records if record.record_id == record_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class DilemmadataTargetFamilyStatistics:
    task_id: str
    source_row_count: int
    available_count: int
    masked_count: int
    missing_count: int
    ambiguous_count: int
    unsupported_count: int
    source_entry_count: int
    emitted_entry_count: int
    available_entry_count: int
    masked_entry_count: int
    equal_duplicate_merge_count: int
    conflict_count: int
    merged_tie_agreement_count: int
    merged_tie_conflict_count: int
    distinct_value_count: int


@dataclass(frozen=True, slots=True)
class DilemmadataTargetStatistics:
    source_row_count: int
    target_source_sha256: str
    family_statistics: tuple[DilemmadataTargetFamilyStatistics, ...]
    alignment_span_count: int
    available_entry_count: int
    masked_entry_count: int
    alt_label_present_count: int
    alt_label_fingerprint: str
    analyst_metadata_field_count: int
    analyst_metadata_fingerprint: str


@dataclass(frozen=True, slots=True)
class DilemmadataTargetAccepted:
    status: Literal["accepted"]
    record: DilemmadataCorpusRecord
    piece_id: str
    target_bundle: TargetBundle
    sidecar_fingerprint: str
    statistics: DilemmadataTargetStatistics


@dataclass(frozen=True, slots=True)
class DilemmadataTargetQuarantine:
    status: Literal["quarantined"]
    record: DilemmadataCorpusRecord
    piece_id: str
    categories: tuple[str, ...]
    messages: tuple[str, ...]


DilemmadataTargetConversionResult = (
    DilemmadataTargetAccepted | DilemmadataTargetQuarantine
)


@dataclass(frozen=True, slots=True)
class _TargetRow:
    ordinal: int
    line: int
    onset_qn: RationalTime
    canonical_note_id: str | None
    tie_continuation: bool
    repair_mask_scope: Literal["none", "note", "all"]
    values: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _FamilyRow:
    row: _TargetRow
    state: TargetState
    value: str | None
    identity: str | None


@dataclass(frozen=True, slots=True)
class _EmittedEntry:
    start_qn: RationalTime | None
    end_qn: RationalTime | None
    canonical_note_id: str | None
    value: str | None
    available: bool
    source_rows: tuple[int, ...]
    conflict: bool


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _target_metadata_record_payload(
    metadata: DilemmadataTargetMetadata,
) -> dict[str, object]:
    return {
        "ambiguous_fields": list(metadata.ambiguous_fields),
        "fields": [list(field) for field in metadata.fields],
        "record_id": metadata.record_id,
    }


def _target_metadata_index_payload(
    index: DilemmadataTargetMetadataIndex,
) -> dict[str, object]:
    return _target_metadata_index_values_payload(
        version=index.version,
        records=index.records,
        source_files=index.source_files,
    )


def _target_metadata_index_values_payload(
    *,
    version: str,
    records: tuple[DilemmadataTargetMetadata, ...],
    source_files: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "records": [
            {
                **_target_metadata_record_payload(record),
                "fingerprint": record.fingerprint,
            }
            for record in records
        ],
        "source_files": [list(row) for row in source_files],
        "version": version,
    }


def _target_metadata_index_fingerprint(
    index: DilemmadataTargetMetadataIndex,
) -> str:
    return sha256(_canonical_bytes(_target_metadata_index_payload(index))).hexdigest()


def _metadata_record(
    record_id: str,
    source_rows: Sequence[Mapping[str, str]],
    fields: Mapping[str, tuple[str, ...]],
) -> DilemmadataTargetMetadata:
    values: list[tuple[str, str]] = []
    ambiguous: list[str] = []
    for output_field, source_fields in sorted(fields.items()):
        observed = tuple(
            sorted(
                {
                    value
                    for row in source_rows
                    for source_field in source_fields
                    if (value := row.get(source_field, "").strip())
                    and value not in _MISSING
                }
            )
        )
        if not observed:
            continue
        if len(observed) > 1:
            ambiguous.append(output_field)
        values.append(
            (
                output_field,
                observed[0]
                if len(observed) == 1
                else json.dumps(
                    observed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    metadata = DilemmadataTargetMetadata(
        record_id=record_id,
        fields=tuple(values),
        ambiguous_fields=tuple(ambiguous),
        fingerprint="",
    )
    return DilemmadataTargetMetadata(
        record_id=metadata.record_id,
        fields=metadata.fields,
        ambiguous_fields=metadata.ambiguous_fields,
        fingerprint=sha256(
            _canonical_bytes(_target_metadata_record_payload(metadata))
        ).hexdigest(),
    )


def _read_metadata_rows(path: Path) -> tuple[dict[str, str], ...]:
    if not path.is_file():
        return ()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(
                set(reader.fieldnames)
            ):
                raise DilemmadataTargetAdapterError(
                    "target metadata header is absent or duplicated",
                    category="dilemmadata.target.metadata_invalid",
                )
            return tuple(dict(row) for row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DilemmadataTargetAdapterError(
            f"target metadata is unreadable: {type(exc).__name__}",
            category="dilemmadata.target.metadata_invalid",
        ) from exc


def load_dilemmadata_target_metadata_index(
    root: str | PathLike[str],
    records: Iterable[DilemmadataCorpusRecord],
) -> DilemmadataTargetMetadataIndex:
    """Load target-only analyst/reviewer evidence without changing raw records."""

    root_path = Path(root).resolve()
    record_rows = tuple(records)
    if len({record.record_id for record in record_rows}) != len(record_rows):
        raise DilemmadataTargetAdapterError(
            "target metadata input records must be unique",
            category="dilemmadata.target.metadata_invalid",
        )
    an_path = root_path / "pitch_arrays" / "AN" / "dataset_summary.tsv"
    dlc_path = (
        root_path
        / "processing"
        / "DLC"
        / "distant_listening_corpus.metadata.tsv"
    )
    an_rows = _read_metadata_rows(an_path)
    dlc_rows = _read_metadata_rows(dlc_path)
    an_by_piece: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    dlc_by_piece: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in an_rows:
        an_by_piece[row.get("file", "").strip()].append(row)
    for row in dlc_rows:
        dlc_by_piece[
            (row.get("corpus", "").strip(), row.get("piece", "").strip())
        ].append(row)
    metadata_records = tuple(
        sorted(
            (
                _metadata_record(
                    record.record_id,
                    (
                        an_by_piece.get(record.piece_name, [])
                        if record.dialect == "an_joint"
                        else dlc_by_piece.get(
                            (record.collection, record.piece_name),
                            [],
                        )
                    ),
                    (
                        {
                            "analyst": ("a_analyst",),
                            "composer": ("s_composer", "a_composer"),
                            "movement": ("s_movementName",),
                            "proofreader": ("a_proofreader",),
                            "title": ("s_title", "a_title"),
                        }
                        if record.dialect == "an_joint"
                        else {
                            "annotators": ("annotators",),
                            "composer": ("composer", "composer_text"),
                            "movement": ("movementTitle", "movementNumber"),
                            "reviewers": ("reviewers",),
                            "source_url": ("last_modified_url",),
                            "source_version": ("last_modified",),
                            "suggested_split": ("split",),
                            "title": ("workTitle", "title_text"),
                        }
                    ),
                )
                for record in record_rows
            ),
            key=lambda row: row.record_id,
        )
    )
    source_files = tuple(
        sorted(
            (
                relative,
                _file_sha256(path),
            )
            for relative, path in (
                ("pitch_arrays/AN/dataset_summary.tsv", an_path),
                (
                    "processing/DLC/distant_listening_corpus.metadata.tsv",
                    dlc_path,
                ),
            )
            if path.is_file()
        )
    )
    return DilemmadataTargetMetadataIndex(
        version=DILEMMADATA_TARGET_METADATA_VERSION,
        records=metadata_records,
        source_files=source_files,
        fingerprint=sha256(
            _canonical_bytes(
                _target_metadata_index_values_payload(
                    version=DILEMMADATA_TARGET_METADATA_VERSION,
                    records=metadata_records,
                    source_files=source_files,
                )
            )
        ).hexdigest(),
    )


def _root_for_record(record: DilemmadataCorpusRecord) -> Path:
    relative = PurePosixPath(record.relative_path)
    root = record.path.resolve()
    for _part in relative.parts:
        root = root.parent
    if (root / relative).resolve() != record.path.resolve():
        raise DilemmadataTargetAdapterError(
            "record path cannot be resolved against its corpus-relative path",
            category="dilemmadata.target.metadata_invalid",
        )
    return root


def _file_sha256(path: PathLike[str]) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_value(value: str) -> bool | None:
    normalized = value.strip()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def _quarantine(
    record: DilemmadataCorpusRecord,
    piece_id: str,
    categories: Iterable[str],
    messages: Iterable[str],
) -> DilemmadataTargetQuarantine:
    return DilemmadataTargetQuarantine(
        status="quarantined",
        record=record,
        piece_id=piece_id,
        categories=tuple(sorted(set(categories))),
        messages=tuple(messages)[:16],
    )


def _selected_fields(dialect: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                "alt_label",
                *(
                    field
                    for spec in DILEMMADATA_SOURCE_FAMILIES
                    if spec.dialect == dialect
                    for field in (
                        *spec.source_fields,
                        *((spec.gate_field,) if spec.gate_field is not None else ()),
                        *((spec.source_identity_field,)
                          if spec.source_identity_field is not None else ()),
                    )
                ),
            }
        )
    )


def _read_target_rows(
    record: DilemmadataCorpusRecord,
    alignment: DilemmadataRawTargetAlignmentEvidence,
) -> tuple[tuple[_TargetRow, ...], str, int, str] | DilemmadataTargetQuarantine:
    selected = _selected_fields(record.dialect)
    digest = sha256(
        f"dilemmadata.target-columns.{record.dialect}.1\0".encode("ascii")
    )
    alt_digest = sha256(b"dilemmadata.alt-label.1\0")
    alt_count = 0
    rows: list[_TargetRow] = []
    try:
        handle = record.path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as exc:
        return _quarantine(
            record,
            record.piece_id,
            ("dilemmadata.target.source_unreadable",),
            (f"target source unreadable: {type(exc).__name__}",),
        )
    try:
        with handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            if len(header) != len(set(header)):
                return _quarantine(
                    record,
                    record.piece_id,
                    ("dilemmadata.target.duplicate_header_field",),
                    ("target source contains duplicate header fields",),
                )
            indices = {name: index for index, name in enumerate(header)}
            present = tuple(field for field in selected if field in indices)
            digest.update(_canonical_bytes({"present_fields": present}))
            for ordinal, raw_values in enumerate(reader):
                line = ordinal + 2
                if len(raw_values) != len(header):
                    return _quarantine(
                        record,
                        record.piece_id,
                        ("dilemmadata.target.row_width_mismatch",),
                        (
                            f"line {line}: expected {len(header)} columns, "
                            f"observed {len(raw_values)}",
                        ),
                    )
                if ordinal >= len(alignment.rows):
                    return _quarantine(
                        record,
                        record.piece_id,
                        ("dilemmadata.target.row_binding_mismatch",),
                        ("target source has more rows than raw alignment evidence",),
                    )
                binding = alignment.rows[ordinal]
                if binding.ordinal != ordinal or binding.line != line:
                    return _quarantine(
                        record,
                        record.piece_id,
                        ("dilemmadata.target.row_binding_mismatch",),
                        (f"line {line}: raw alignment row order differs",),
                    )
                values = {
                    field: raw_values[indices[field]]
                    for field in present
                }
                digest.update(
                    _canonical_bytes(
                        [[field, values[field]] for field in present]
                    )
                )
                digest.update(b"\n")
                alt_value = values.get("alt_label", "").strip()
                if alt_value not in _MISSING:
                    alt_count += 1
                    alt_digest.update(alt_value.encode("utf-8"))
                    alt_digest.update(b"\0")
                rows.append(
                    _TargetRow(
                        ordinal=ordinal,
                        line=line,
                        onset_qn=binding.onset_qn,
                        canonical_note_id=binding.canonical_note_id,
                        tie_continuation=binding.tie_continuation,
                        repair_mask_scope=binding.repair_mask_scope,
                        values=values,
                    )
                )
    except (csv.Error, UnicodeError) as exc:
        return _quarantine(
            record,
            record.piece_id,
            ("dilemmadata.target.tabular_parse_failed",),
            (f"target source parse failed: {type(exc).__name__}",),
        )
    if len(rows) != len(alignment.rows):
        return _quarantine(
            record,
            record.piece_id,
            ("dilemmadata.target.row_binding_mismatch",),
            ("target source row count differs from raw alignment evidence",),
        )
    return tuple(rows), digest.hexdigest(), alt_count, alt_digest.hexdigest()


def _family_row(spec: DilemmadataSourceFamilySpec, row: _TargetRow) -> _FamilyRow:
    if row.repair_mask_scope == "all" or (
        row.repair_mask_scope == "note"
        and spec.coordinate == "canonical_note_identity"
    ):
        return _FamilyRow(row, "masked", None, None)
    if spec.primary_field not in row.values:
        return _FamilyRow(row, "unsupported", None, None)
    if spec.gate_field is not None:
        if spec.gate_field not in row.values:
            return _FamilyRow(row, "unsupported", None, None)
        gate_raw = row.values[spec.gate_field].strip()
        gate = _bool_value(gate_raw)
        if gate is None:
            state: TargetState = (
                "masked" if gate_raw in _MISSING else "unsupported"
            )
            return _FamilyRow(row, state, None, None)
        if not gate:
            return _FamilyRow(row, "masked", None, None)
    raw_value = row.values[spec.primary_field].strip()
    if raw_value in _MISSING:
        return _FamilyRow(row, "missing", None, None)
    value = raw_value
    if spec.encoding_mode == "positive_unlabeled" and spec.family != "cadence":
        boolean = _bool_value(raw_value)
        if boolean is False:
            return _FamilyRow(row, "masked", None, None)
        if boolean is None:
            return _FamilyRow(row, "unsupported", None, None)
        value = "present"
    if spec.vocabulary is not None and value not in spec.vocabulary:
        return _FamilyRow(row, "unsupported", None, None)
    identity = None
    if spec.source_identity_field is not None:
        raw_identity = row.values.get(spec.source_identity_field, "").strip()
        if raw_identity not in _MISSING:
            identity = raw_identity
    return _FamilyRow(row, "available", value, identity)


def _resolve_rows(rows: Sequence[_FamilyRow]) -> tuple[str | None, bool, bool]:
    """Return value, available, conflict for one exact source entity."""

    if not rows or any(row.state != "available" for row in rows):
        return None, False, False
    values = {row.value for row in rows}
    if len(values) != 1:
        return None, False, True
    return next(iter(values)), True, False


def _emit_note_entries(
    rows: tuple[_FamilyRow, ...],
) -> tuple[tuple[_EmittedEntry, ...], int, int, int, int]:
    grouped: dict[str, list[_FamilyRow]] = defaultdict(list)
    for row in rows:
        if row.row.canonical_note_id is None:
            continue
        grouped[row.row.canonical_note_id].append(row)
    entries: list[_EmittedEntry] = []
    tie_agreement = 0
    tie_conflict = 0
    equal_merges = 0
    conflicts = 0
    for note_id, candidates in sorted(grouped.items()):
        value, available, conflict = _resolve_rows(candidates)
        if len(candidates) > 1:
            if available:
                tie_agreement += 1
            else:
                tie_conflict += 1
        if len(candidates) > 1 and available:
            equal_merges += len(candidates) - 1
        if conflict:
            conflicts += 1
        entries.append(
            _EmittedEntry(
                start_qn=None,
                end_qn=None,
                canonical_note_id=note_id,
                value=value,
                available=available,
                source_rows=tuple(item.row.ordinal for item in candidates),
                conflict=conflict or (len(candidates) > 1 and not available),
            )
        )
    return tuple(entries), equal_merges, conflicts, tie_agreement, tie_conflict


def _emit_point_entries(
    rows: tuple[_FamilyRow, ...],
) -> tuple[tuple[_EmittedEntry, ...], int, int]:
    grouped: dict[RationalTime, list[_FamilyRow]] = defaultdict(list)
    for row in rows:
        if row.state == "available":
            grouped[row.row.onset_qn].append(row)
    entries: list[_EmittedEntry] = []
    equal_merges = 0
    conflicts = 0
    for onset, candidates in sorted(grouped.items()):
        value, available, conflict = _resolve_rows(candidates)
        if len(candidates) > 1 and available:
            equal_merges += len(candidates) - 1
        if conflict:
            conflicts += 1
        entries.append(
            _EmittedEntry(
                start_qn=onset,
                end_qn=onset,
                canonical_note_id=None,
                value=value,
                available=available,
                source_rows=tuple(item.row.ordinal for item in candidates),
                conflict=conflict,
            )
        )
    return tuple(entries), equal_merges, conflicts


def _span_groups(
    spec: DilemmadataSourceFamilySpec,
    rows: tuple[_FamilyRow, ...],
) -> tuple[tuple[RationalTime, tuple[_FamilyRow, ...]], ...]:
    if spec.coordinate == "global_span":
        return ((RationalTime(0), rows),)
    if spec.family == "local_key":
        by_onset: dict[RationalTime, list[_FamilyRow]] = defaultdict(list)
        for row in rows:
            by_onset[row.row.onset_qn].append(row)
        resolved: list[tuple[RationalTime, tuple[_FamilyRow, ...]]] = []
        for onset, candidates in sorted(by_onset.items()):
            resolved.append((onset, tuple(candidates)))
        runs: list[tuple[RationalTime, list[_FamilyRow]]] = []
        previous_value: str | None = None
        previous_available = False
        for onset, candidates in resolved:
            value, available, _ = _resolve_rows(candidates)
            if runs and available and previous_available and value == previous_value:
                runs[-1][1].extend(candidates)
            else:
                runs.append((onset, list(candidates)))
            previous_value = value
            previous_available = available
        return tuple((onset, tuple(candidates)) for onset, candidates in runs)
    by_identity: dict[str, list[_FamilyRow]] = defaultdict(list)
    for row in rows:
        identity = row.identity or (
            f"onset:{row.row.onset_qn.num}/{row.row.onset_qn.den}"
        )
        by_identity[identity].append(row)
    ordered = sorted(
        by_identity.items(),
        key=lambda item: (
            min(candidate.row.onset_qn for candidate in item[1]),
            item[0],
        ),
    )
    return tuple(
        (
            min(item.row.onset_qn for item in candidates),
            tuple(candidates),
        )
        for _identity, candidates in ordered
    )


def _emit_span_entries(
    spec: DilemmadataSourceFamilySpec,
    rows: tuple[_FamilyRow, ...],
    piece: CanonicalPiece,
) -> tuple[tuple[_EmittedEntry, ...], int, int]:
    groups = _span_groups(spec, rows)
    entries: list[_EmittedEntry] = []
    equal_merges = 0
    conflicts = 0
    for index, (start, candidates) in enumerate(groups):
        value, available, conflict = _resolve_rows(candidates)
        if len(candidates) > 1 and available:
            equal_merges += len(candidates) - 1
        if conflict:
            conflicts += 1
        if spec.coordinate == "global_span":
            start = RationalTime(0)
            end = piece.duration_qn
        else:
            end = groups[index + 1][0] if index + 1 < len(groups) else start
        entries.append(
            _EmittedEntry(
                start_qn=start,
                end_qn=end,
                canonical_note_id=None,
                value=value,
                available=available,
                source_rows=tuple(item.row.ordinal for item in candidates),
                conflict=conflict,
            )
        )
    return tuple(entries), equal_merges, conflicts


def _family_target(
    spec: DilemmadataSourceFamilySpec,
    rows: tuple[_TargetRow, ...],
    piece: CanonicalPiece,
    token: str,
) -> tuple[
    SampleTarget,
    tuple[AnnotationSpan, ...],
    DilemmadataTargetFamilyStatistics,
]:
    family_rows = tuple(_family_row(spec, row) for row in rows)
    state_counts = Counter(row.state for row in family_rows)
    if spec.coordinate == "canonical_note_identity":
        entries, equal_merges, conflicts, tie_agreement, tie_conflict = (
            _emit_note_entries(family_rows)
        )
    elif spec.coordinate == "exact_onset_event":
        entries, equal_merges, conflicts = _emit_point_entries(family_rows)
        tie_agreement = 0
        tie_conflict = 0
    else:
        entries, equal_merges, conflicts = _emit_span_entries(
            spec,
            family_rows,
            piece,
        )
        tie_agreement = 0
        tie_conflict = 0
    task_token = sha256(spec.task_id.encode("utf-8")).hexdigest()[:12]
    spans: list[AnnotationSpan] = []
    entity_ids: list[str] = []
    for index, entry in enumerate(entries):
        if entry.canonical_note_id is not None:
            entity_ids.append(entry.canonical_note_id)
            continue
        assert entry.start_qn is not None and entry.end_qn is not None
        span_id = f"span:dilemmadata-target-{token}-{task_token}-{index:08d}"
        spans.append(
            AnnotationSpan(
                annotation_id=span_id,
                annotation_type=spec.task_id,
                layer="target_alignment",
                start_qn=entry.start_qn,
                end_qn=entry.end_qn,
                track_id=None,
                value=None,
                provenance_id="prov:dilemmadata-target-annotation",
            )
        )
        entity_ids.append(span_id)
    available_mask = tuple(entry.available for entry in entries)
    target = SampleTarget(
        task_id=spec.task_id,
        annotation_view_id=None,
        alignment_type=spec.ontology_spec.source_alignment_type,
        entity_ids=tuple(entity_ids),
        values=tuple(entry.value if entry.available else None for entry in entries),
        availability_mask=available_mask,
        confidence=tuple(None for _ in entries),
        source=tuple("dataset" if entry.available else None for entry in entries),
        provenance_ids=tuple(
            "prov:dilemmadata-target-annotation" if entry.available else None
            for entry in entries
        ),
    )
    distinct_values = {
        entry.value for entry in entries if entry.available and entry.value is not None
    }
    merged_conflict_count = (
        tie_conflict
        if spec.coordinate == "canonical_note_identity"
        else conflicts
    )
    statistics = DilemmadataTargetFamilyStatistics(
        task_id=spec.task_id,
        source_row_count=len(rows),
        available_count=state_counts["available"],
        masked_count=state_counts["masked"],
        missing_count=state_counts["missing"],
        ambiguous_count=merged_conflict_count,
        unsupported_count=state_counts["unsupported"],
        source_entry_count=len(entries),
        emitted_entry_count=len(entries),
        available_entry_count=sum(available_mask),
        masked_entry_count=len(entries) - sum(available_mask),
        equal_duplicate_merge_count=equal_merges,
        conflict_count=merged_conflict_count,
        merged_tie_agreement_count=tie_agreement,
        merged_tie_conflict_count=tie_conflict,
        distinct_value_count=len(distinct_values),
    )
    return target, tuple(spans), statistics


def _target_token(record: DilemmadataCorpusRecord) -> str:
    return sha256(record.record_id.encode("utf-8")).hexdigest()[:24]


def convert_dilemmadata_target_sidecar(
    record: DilemmadataCorpusRecord,
    piece: CanonicalPiece,
    alignment_evidence: DilemmadataRawTargetAlignmentEvidence,
    *,
    config: DilemmadataTargetAdapterConfig = DilemmadataTargetAdapterConfig(),
    metadata_index: DilemmadataTargetMetadataIndex | None = None,
) -> DilemmadataTargetConversionResult:
    """Read only evidenced target columns and build an external exact sidecar."""

    if not isinstance(record, DilemmadataCorpusRecord):
        raise DilemmadataTargetAdapterError(
            "record must be DilemmadataCorpusRecord",
            category="dilemmadata.target.record_invalid",
        )
    if not isinstance(piece, CanonicalPiece):
        raise DilemmadataTargetAdapterError(
            "piece must be CanonicalPiece",
            category="dilemmadata.target.piece_invalid",
        )
    if not isinstance(config, DilemmadataTargetAdapterConfig):
        raise DilemmadataTargetAdapterError(
            "config must be DilemmadataTargetAdapterConfig",
            category="dilemmadata.target.config_invalid",
        )
    if metadata_index is not None and not isinstance(
        metadata_index,
        DilemmadataTargetMetadataIndex,
    ):
        raise DilemmadataTargetAdapterError(
            "metadata_index must be DilemmadataTargetMetadataIndex",
            category="dilemmadata.target.metadata_invalid",
        )
    if not validate_dilemmadata_record_binding(record):
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.record_binding_mismatch",),
            ("record does not match its versioned raw discovery binding",),
        )
    try:
        current_source_sha256 = _file_sha256(record.path)
    except OSError as exc:
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.source_unreadable",),
            (f"target source unreadable: {type(exc).__name__}",),
        )
    if current_source_sha256 != record.physical_source_sha256:
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.source_changed_after_raw_acceptance",),
            (
                "source bytes changed after Phase 9B.1 acceptance; rerun the raw "
                "adapter so raw mutations cannot be hidden by target extraction",
            ),
        )
    if not validate_dilemmadata_alignment_evidence(
        record,
        piece,
        alignment_evidence,
    ):
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.alignment_binding_mismatch",),
            ("raw source-row alignment evidence differs from record/canonical input",),
        )
    if (
        piece.dataset_name != DILEMMADATA_DATASET_NAME
        or piece.piece_id != record.piece_id
        or piece.source_group_id != record.source_group_id
        or piece.targets
        or piece.annotations
    ):
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.raw_piece_mismatch",),
            ("target adapter requires the exact raw-only Phase 9B.1 canonical piece",),
        )
    if metadata_index is None:
        try:
            metadata_index = load_dilemmadata_target_metadata_index(
                _root_for_record(record),
                (record,),
            )
        except DilemmadataTargetAdapterError as exc:
            return _quarantine(
                record,
                piece.piece_id,
                (exc.category,),
                (str(exc),),
            )
    analyst_metadata = metadata_index.for_record(record.record_id)
    if analyst_metadata is None:
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.metadata_record_missing",),
            ("target metadata index does not cover the requested record",),
        )
    parsed = _read_target_rows(record, alignment_evidence)
    if isinstance(parsed, DilemmadataTargetQuarantine):
        return parsed
    rows, target_source_sha256, alt_count, alt_fingerprint = parsed
    token = _target_token(record)
    targets: list[SampleTarget] = []
    spans: list[AnnotationSpan] = []
    family_statistics: list[DilemmadataTargetFamilyStatistics] = []
    for task_id in DILEMMADATA_TASK_IDS_BY_DIALECT[record.dialect]:
        spec = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
        target, family_spans, statistics = _family_target(
            spec,
            rows,
            piece,
            token,
        )
        targets.append(target)
        spans.extend(family_spans)
        family_statistics.append(statistics)
    conflict_count = sum(item.conflict_count for item in family_statistics)
    diagnostics: list[QualityFlag] = []
    if analyst_metadata.fields:
        diagnostics.append(
            QualityFlag(
                code="dilemmadata.target.analyst_metadata_present",
                severity="info",
                message=(
                    f"Retained {len(analyst_metadata.fields)} target-only "
                    "analyst/reviewer metadata fields as provenance."
                ),
                entity_ids=(piece.piece_id,),
                provenance_id="prov:dilemmadata-target-source",
            )
        )
    if analyst_metadata.ambiguous_fields:
        diagnostics.append(
            QualityFlag(
                code="dilemmadata.target.analyst_metadata_ambiguous",
                severity="warning",
                message=(
                    "Retained all distinct values for ambiguous metadata fields: "
                    + ", ".join(analyst_metadata.ambiguous_fields)
                ),
                entity_ids=(piece.piece_id,),
                provenance_id="prov:dilemmadata-target-source",
            )
        )
    for statistics in family_statistics:
        for state, count in (
            ("missing", statistics.missing_count),
            ("masked", statistics.masked_count),
            ("ambiguous", statistics.ambiguous_count),
            ("unsupported", statistics.unsupported_count),
        ):
            if count:
                diagnostics.append(
                    QualityFlag(
                        code=f"dilemmadata.target.state.{state}",
                        severity=("warning" if state in {"ambiguous", "unsupported"} else "info"),
                        message=(
                            f"{statistics.task_id}: retained {count} source rows in "
                            f"the {state} target state."
                        ),
                        entity_ids=(piece.piece_id,),
                        provenance_id="prov:dilemmadata-target-annotation",
                    )
                )
        spec = DILEMMADATA_SOURCE_FAMILY_BY_TASK[statistics.task_id]
        if spec.encoding_mode == "open_string_cpu":
            diagnostics.append(
                QualityFlag(
                    code="dilemmadata.target.state.deferred_open_vocabulary",
                    severity="info",
                    message=(
                        f"{statistics.task_id}: source strings are retained on CPU "
                        "without dynamic class IDs."
                    ),
                    entity_ids=(piece.piece_id,),
                    provenance_id="prov:dilemmadata-target-annotation",
                )
            )
    if alt_count:
        diagnostics.append(
            QualityFlag(
                code="dilemmadata.target.alt_label_present",
                severity="info",
                message=(
                    f"Retained {alt_count} row-level alternative-label observations "
                    "as target-only diagnostic evidence."
                ),
                entity_ids=(piece.piece_id,),
                provenance_id="prov:dilemmadata-target-annotation",
            )
        )
    if conflict_count:
        diagnostics.append(
            QualityFlag(
                code="dilemmadata.target.alignment_conflict",
                severity="warning",
                message=(
                    f"Masked {conflict_count} conflicting duplicate or merged-tie "
                    "target entities."
                ),
                entity_ids=(piece.piece_id,),
                provenance_id="prov:dilemmadata-target-annotation",
            )
        )
    source_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-target-source",
        kind="source",
        source="dilemmadata_v1_target_columns",
        record_id=record.record_id,
        uri=record.relative_path,
        version=DILEMMADATA_RELEASE_VERSION,
        checksum_sha256=target_source_sha256,
        created_at=None,
        parents=("prov:dilemmadata-source",),
        details=tuple(
            sorted(
                (
                    ("alt_label_fingerprint", alt_fingerprint),
                    ("alt_label_present_count", alt_count),
                    (
                        "analyst_metadata_fingerprint",
                        analyst_metadata.fingerprint,
                    ),
                    ("dialect", record.dialect),
                    *(
                        (f"metadata.{key}", value)
                        for key, value in analyst_metadata.fields
                    ),
                    (
                        "metadata_contract_version",
                        DILEMMADATA_TARGET_METADATA_VERSION,
                    ),
                )
            )
        ),
    )
    conversion_details: tuple[tuple[str, object], ...] = (
        ("duplicate_policy", config.duplicate_policy),
        ("point_policy", config.point_policy),
        ("span_policy", config.span_policy),
        ("target_column_policy", config.target_column_policy),
        ("tie_policy", config.tie_policy),
    )
    if alignment_evidence.raw_repair_evidence_fingerprint is not None:
        conversion_details = (
            *conversion_details,
            (
                "raw_repair_evidence_fingerprint",
                alignment_evidence.raw_repair_evidence_fingerprint,
            ),
            ("raw_target_alignment_version", alignment_evidence.version),
        )
    conversion_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-target-conversion",
        kind="conversion",
        source="music_critic.dilemmadata_target_adapter",
        record_id=None,
        uri=None,
        version=(
            DILEMMADATA_TARGET_ADAPTER_VERSION
            if alignment_evidence.raw_repair_evidence_fingerprint is None
            else DILEMMADATA_REMEDIATED_TARGET_ADAPTER_VERSION
        ),
        checksum_sha256=None,
        created_at=None,
        parents=("prov:dilemmadata-target-source",),
        details=conversion_details,
    )
    annotation_provenance = ProvenanceRecord(
        provenance_id="prov:dilemmadata-target-annotation",
        kind="annotation",
        source="dilemmadata_v1_source_native_theory",
        record_id=record.record_id,
        uri=None,
        version=DILEMMADATA_TARGET_SIDECAR_VERSION,
        checksum_sha256=target_source_sha256,
        created_at=None,
        parents=("prov:dilemmadata-target-conversion",),
        details=(
            (
                "alignment_rules_version",
                DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
            ),
            (
                "encoding_registry_version",
                DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
            ),
            (
                "family_registry_version",
                DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
            ),
        ),
    )
    try:
        bundle = TargetBundle(
            contract_version=TARGET_BUNDLE_CONTRACT_VERSION,
            registry_extension_ids=(DILEMMADATA_TARGET_REGISTRY_ID,),
            dataset_id=piece.dataset_name,
            piece_id=piece.piece_id,
            analysis_view_id=f"dilemmadata.{record.dialect}.{record.record_id}",
            alignment_spans=tuple(
                sorted(
                    spans,
                    key=lambda span: (
                        span.start_qn,
                        span.end_qn,
                        span.annotation_id,
                    ),
                )
            ),
            targets=tuple(sorted(targets, key=lambda target: target.task_id)),
            provenance=(
                source_provenance,
                conversion_provenance,
                annotation_provenance,
            ),
            diagnostics=tuple(
                sorted(
                    diagnostics,
                    key=lambda flag: (flag.code, flag.entity_ids, flag.message),
                )
            ),
        )
        fingerprint = target_bundle_fingerprint(bundle)
    except (TypeError, ValueError) as exc:
        return _quarantine(
            record,
            piece.piece_id,
            ("dilemmadata.target.bundle_invalid",),
            (f"target bundle validation failed: {type(exc).__name__}: {exc}",),
        )
    statistics = DilemmadataTargetStatistics(
        source_row_count=len(rows),
        target_source_sha256=target_source_sha256,
        family_statistics=tuple(family_statistics),
        alignment_span_count=len(bundle.alignment_spans),
        available_entry_count=sum(
            item.available_entry_count for item in family_statistics
        ),
        masked_entry_count=sum(
            item.masked_entry_count for item in family_statistics
        ),
        alt_label_present_count=alt_count,
        alt_label_fingerprint=alt_fingerprint,
        analyst_metadata_field_count=len(analyst_metadata.fields),
        analyst_metadata_fingerprint=analyst_metadata.fingerprint,
    )
    return DilemmadataTargetAccepted(
        status="accepted",
        record=record,
        piece_id=piece.piece_id,
        target_bundle=bundle,
        sidecar_fingerprint=fingerprint,
        statistics=statistics,
    )


def build_dilemmadata_target_sidecar(
    accepted: DilemmadataAccepted,
    *,
    config: DilemmadataTargetAdapterConfig = DilemmadataTargetAdapterConfig(),
    metadata_index: DilemmadataTargetMetadataIndex | None = None,
) -> DilemmadataTargetConversionResult:
    """Build one sidecar from the exact accepted Phase 9B.1 result."""

    if not isinstance(accepted, DilemmadataAccepted):
        raise DilemmadataTargetAdapterError(
            "accepted must be DilemmadataAccepted",
            category="dilemmadata.target.accepted_invalid",
        )
    return convert_dilemmadata_target_sidecar(
        accepted.record,
        accepted.piece,
        accepted.alignment_evidence,
        config=config,
        metadata_index=metadata_index,
    )


def iter_dilemmadata_target_sidecars(
    root: str | PathLike[str],
    *,
    raw_config: DilemmadataAdapterConfig = DilemmadataAdapterConfig(),
    target_config: DilemmadataTargetAdapterConfig = DilemmadataTargetAdapterConfig(),
    identity: DilemmadataCorpusIdentity = DilemmadataCorpusIdentity(),
) -> Iterator[DilemmadataTargetConversionResult]:
    """Yield one target outcome for every raw record accepted by Phase 9B.1."""

    discovery = discover_dilemmadata_corpus(
        root,
        identity=identity,
        require_valid=True,
    )
    metadata_index = load_dilemmadata_target_metadata_index(
        discovery.root,
        discovery.records,
    )
    for record in discovery.records:
        raw_outcome = convert_dilemmadata_record(record, config=raw_config)
        if isinstance(raw_outcome, DilemmadataAccepted):
            yield build_dilemmadata_target_sidecar(
                raw_outcome,
                config=target_config,
                metadata_index=metadata_index,
            )


__all__ = [
    "DILEMMADATA_REMEDIATED_TARGET_ADAPTER_VERSION",
    "DILEMMADATA_TARGET_ADAPTER_VERSION",
    "DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION",
    "DILEMMADATA_TARGET_AUDIT_REPORT_VERSION",
    "DILEMMADATA_TARGET_SIDECAR_VERSION",
    "DILEMMADATA_TARGET_METADATA_VERSION",
    "DilemmadataTargetAccepted",
    "DilemmadataTargetAdapterConfig",
    "DilemmadataTargetAdapterError",
    "DilemmadataTargetConversionResult",
    "DilemmadataTargetFamilyStatistics",
    "DilemmadataTargetMetadata",
    "DilemmadataTargetMetadataIndex",
    "DilemmadataTargetQuarantine",
    "DilemmadataTargetStatistics",
    "build_dilemmadata_target_sidecar",
    "convert_dilemmadata_target_sidecar",
    "iter_dilemmadata_target_sidecars",
    "load_dilemmadata_target_metadata_index",
]
