"""Exact source-row binding for Dilemmadata duplicate-time transitions.

The common cache deliberately contains target-neutral canonical pieces and
separate target sidecars.  The raw adapter's structured row lineage is not
serialized there, so this experiment contract recovers the immutable ordinal
from the canonical note ID and validates it against the pinned source TSV.
No target value or class index participates in the recovery algorithm.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.dataset import CommonDatasetRecord
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
    DilemmadataCommonHarmonicProjection,
)
from music_critic.tasks.multisource import TargetBundle


SOURCE_ROW_BINDING_VERSION = "1.0.0"
SOURCE_ROW_BINDING_SCHEMA = "phase9eb1-dilemmadata-source-row-binding"
_TASK_NAME = {
    COMMON_INVERSION_TASK: "inversion",
    COMMON_QUALITY_TASK: "quality",
}
_NOTE_ID = re.compile(r"^note:dilemmadata-([0-9a-f]{24})-([0-9]{8})$")
_TRUE = frozenset({"1", "True", "true", "TRUE"})


class SourceRowBindingError(ValueError):
    """A deterministic source row cannot be recovered without ambiguity."""

    def __init__(self, category: str, diagnostic: Mapping[str, object]) -> None:
        if category not in {"ambiguous", "unresolved"}:
            raise ValueError("source-row error category must be ambiguous or unresolved")
        self.category = category
        self.diagnostic = dict(sorted(diagnostic.items()))
        super().__init__(
            f"source-row provenance {category}: "
            f"{self.diagnostic!r}"
        )


@dataclass(frozen=True, slots=True)
class SourceRowEntry:
    entry_index: int
    entity_id: str
    source_identity: str
    span_start: RationalTime
    span_end: RationalTime


@dataclass(frozen=True, slots=True)
class SourceRowAssignment:
    note_index: int
    note_id: str
    source_row_ordinal: int
    source_identity: str
    entry_index: int
    entity_id: str


@dataclass(frozen=True, slots=True)
class SourceRowGroup:
    task: str
    start_qn: RationalTime
    entries: tuple[SourceRowEntry, ...]
    assignments: tuple[SourceRowAssignment, ...]
    binding_basis: str = (
        "validated_note_id_ordinal_plus_raw_identity_and_typed_entry_order"
    )

    def __post_init__(self) -> None:
        if self.task not in {"quality", "inversion"}:
            raise ValueError("source-row group task is invalid")
        entry_keys = tuple((row.entry_index, row.entity_id) for row in self.entries)
        if entry_keys != tuple(sorted(set(entry_keys))) or len(self.entries) != 2:
            raise ValueError("source-row group requires two uniquely sorted entries")
        if len({row.source_identity for row in self.entries}) != len(self.entries):
            raise ValueError("source-row entries require unique source identities")
        assignment_keys = tuple(
            (row.note_index, row.entry_index) for row in self.assignments
        )
        if assignment_keys != tuple(sorted(set(assignment_keys))):
            raise ValueError(
                "source-row assignment pairs must be unique and lexicographically sorted"
            )
        note_indices = tuple(row.note_index for row in self.assignments)
        if len(note_indices) != len(set(note_indices)):
            raise ValueError("one source-row group cannot assign a note to two entries")
        ordinals = tuple(row.source_row_ordinal for row in self.assignments)
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("source-row assignments require unique source ordinals")
        entry_by_index = {row.entry_index: row for row in self.entries}
        for row in self.assignments:
            entry = entry_by_index.get(row.entry_index)
            if entry is None or (
                row.entity_id != entry.entity_id
                or row.source_identity != entry.source_identity
            ):
                raise ValueError("source-row assignment differs from its entry identity")


@dataclass(frozen=True, slots=True)
class DilemmadataSourceRowBinding:
    contract_version: str
    schema: str
    record_id: str
    piece_id: str
    dialect: str
    groups: tuple[SourceRowGroup, ...]
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version != SOURCE_ROW_BINDING_VERSION
            or self.schema != SOURCE_ROW_BINDING_SCHEMA
            or self.dialect != "dlc"
        ):
            raise ValueError("source-row binding version/schema/dialect is invalid")
        keys = tuple(
            (row.task, row.start_qn, tuple(entry.entry_index for entry in row.entries))
            for row in self.groups
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("source-row groups must be uniquely sorted")
        payload = source_row_binding_payload(self, include_fingerprint=False)
        if self.semantic_fingerprint != fingerprint(payload):
            raise ValueError("source-row binding fingerprint is invalid")


def _time_payload(value: RationalTime) -> dict[str, int]:
    return {"den": value.den, "num": value.num}


def _time_from_payload(value: object) -> RationalTime:
    if not isinstance(value, dict) or set(value) != {"den", "num"}:
        raise ValueError("source-row rational time payload is invalid")
    return RationalTime(int(value["num"]), int(value["den"]))


def _binding_payload(
    *,
    contract_version: str,
    dialect: str,
    groups: Sequence[SourceRowGroup],
    piece_id: str,
    record_id: str,
    schema: str,
) -> dict[str, object]:
    return {
        "contract_version": contract_version,
        "dialect": dialect,
        "groups": [
            {
                "assignments": [asdict(row) for row in group.assignments],
                "binding_basis": group.binding_basis,
                "entries": [
                    {
                        **asdict(entry),
                        "span_end": _time_payload(entry.span_end),
                        "span_start": _time_payload(entry.span_start),
                    }
                    for entry in group.entries
                ],
                "start_qn": _time_payload(group.start_qn),
                "task": group.task,
            }
            for group in groups
        ],
        "piece_id": piece_id,
        "record_id": record_id,
        "schema": schema,
    }


def source_row_binding_payload(
    binding: DilemmadataSourceRowBinding,
    *,
    include_fingerprint: bool = True,
) -> dict[str, object]:
    payload = _binding_payload(
        contract_version=binding.contract_version,
        dialect=binding.dialect,
        groups=binding.groups,
        piece_id=binding.piece_id,
        record_id=binding.record_id,
        schema=binding.schema,
    )
    if include_fingerprint:
        payload["semantic_fingerprint"] = binding.semantic_fingerprint
    return payload


def source_row_binding_from_payload(value: object) -> DilemmadataSourceRowBinding:
    if not isinstance(value, dict) or set(value) != {
        "contract_version",
        "dialect",
        "groups",
        "piece_id",
        "record_id",
        "schema",
        "semantic_fingerprint",
    }:
        raise ValueError("source-row binding payload has missing or unknown fields")
    groups: list[SourceRowGroup] = []
    if not isinstance(value["groups"], list):
        raise ValueError("source-row binding groups must be a list")
    for raw_group in value["groups"]:
        if not isinstance(raw_group, dict) or set(raw_group) != {
            "assignments",
            "binding_basis",
            "entries",
            "start_qn",
            "task",
        }:
            raise ValueError("source-row group payload is invalid")
        entries = tuple(
            SourceRowEntry(
                entry_index=int(entry["entry_index"]),
                entity_id=str(entry["entity_id"]),
                source_identity=str(entry["source_identity"]),
                span_start=_time_from_payload(entry["span_start"]),
                span_end=_time_from_payload(entry["span_end"]),
            )
            for entry in raw_group["entries"]
        )
        assignments = tuple(
            SourceRowAssignment(
                note_index=int(row["note_index"]),
                note_id=str(row["note_id"]),
                source_row_ordinal=int(row["source_row_ordinal"]),
                source_identity=str(row["source_identity"]),
                entry_index=int(row["entry_index"]),
                entity_id=str(row["entity_id"]),
            )
            for row in raw_group["assignments"]
        )
        groups.append(
            SourceRowGroup(
                task=str(raw_group["task"]),
                start_qn=_time_from_payload(raw_group["start_qn"]),
                entries=entries,
                assignments=assignments,
                binding_basis=str(raw_group["binding_basis"]),
            )
        )
    return DilemmadataSourceRowBinding(
        contract_version=str(value["contract_version"]),
        schema=str(value["schema"]),
        record_id=str(value["record_id"]),
        piece_id=str(value["piece_id"]),
        dialect=str(value["dialect"]),
        groups=tuple(groups),
        semantic_fingerprint=str(value["semantic_fingerprint"]),
    )


def _source_path(corpus_root: Path, record: CommonDatasetRecord) -> Path:
    try:
        dialect, collection, piece_name = record.record_id.split(":", 2)
    except ValueError as exc:
        raise SourceRowBindingError(
            "unresolved",
            {"reason": "malformed_record_id", "record_id": record.record_id},
        ) from exc
    if dialect != "dlc" or record.dialect != "dlc":
        raise SourceRowBindingError(
            "unresolved",
            {"reason": "dialect_identity_mismatch", "record_id": record.record_id},
        )
    return corpus_root / "pitch_arrays" / "DLC" / collection / f"{piece_name}.tsv"


def _read_rows(path: Path, record: CommonDatasetRecord) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            required = {
                "is_note_onset",
                "pitch",
                "quarterbeats_playthrough",
                "staff",
                "unfolded_harmony_index",
                "voice",
            }
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not required <= set(reader.fieldnames)
            ):
                raise SourceRowBindingError(
                    "unresolved",
                    {
                        "reason": "source_header_invalid",
                        "record_id": record.record_id,
                    },
                )
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise SourceRowBindingError(
            "unresolved",
            {
                "reason": "source_tsv_unavailable",
                "record_id": record.record_id,
                "source_name": path.name,
            },
        ) from exc


def _fraction(raw: str, *, field: str, record_id: str) -> Fraction:
    try:
        return Fraction(raw.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise SourceRowBindingError(
            "unresolved",
            {"field": field, "reason": "source_value_invalid", "record_id": record_id},
        ) from exc


def _rt(value: Fraction) -> RationalTime:
    return RationalTime(value.numerator, value.denominator)


def detect_row_transition_entries(
    source: TargetBundle,
    projection: DilemmadataCommonHarmonicProjection,
) -> tuple[tuple[str, RationalTime, tuple[tuple[int, str, RationalTime], ...]], ...]:
    """Find class-independent same-start point/interval entry clusters."""

    span_by_id = {span.annotation_id: span for span in source.alignment_spans}
    result: list[
        tuple[str, RationalTime, tuple[tuple[int, str, RationalTime], ...]]
    ] = []
    for target in projection.targets:
        task = _TASK_NAME.get(target.task_id)
        if task is None:
            continue
        by_start: dict[RationalTime, list[tuple[int, str, RationalTime]]] = defaultdict(list)
        for entry_index, entry in enumerate(target.entries):
            span = span_by_id.get(entry.entity_id)
            if span is not None:
                by_start[span.start_qn].append(
                    (entry_index, entry.entity_id, span.end_qn)
                )
        for start, entries in sorted(by_start.items()):
            point = [row for row in entries if row[2] == start]
            interval = [row for row in entries if row[2] != start]
            if point and interval:
                result.append((task, start, tuple(sorted(entries))))
    return tuple(result)


def _note_ordinal(note: Any, *, token: str, record_id: str) -> int:
    match = _NOTE_ID.fullmatch(str(note.note_id))
    if match is None or match.group(1) != token:
        raise SourceRowBindingError(
            "unresolved",
            {
                "note_id": str(note.note_id),
                "reason": "canonical_note_id_has_no_validated_source_ordinal",
                "record_id": record_id,
            },
        )
    return int(match.group(2))


def _optional_int(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped else None


def _validate_note_row(
    note: Any,
    row: Mapping[str, str],
    ordinal: int,
    *,
    record_id: str,
) -> str:
    try:
        onset = _fraction(
            row["quarterbeats_playthrough"],
            field="quarterbeats_playthrough",
            record_id=record_id,
        )
        pitch = int(row["pitch"])
        staff = _optional_int(row["staff"])
        voice = _optional_int(row["voice"])
    except (KeyError, ValueError) as exc:
        raise SourceRowBindingError(
            "unresolved",
            {
                "reason": "source_note_identity_invalid",
                "record_id": record_id,
                "source_row_ordinal": ordinal,
            },
        ) from exc
    identity = row["unfolded_harmony_index"].strip()
    if (
        row["is_note_onset"].strip() not in _TRUE
        or _rt(onset) != note.onset_qn
        or pitch != int(note.pitch)
        or staff != note.staff
        or voice != note.voice
        or not identity
    ):
        raise SourceRowBindingError(
            "unresolved",
            {
                "note_id": str(note.note_id),
                "reason": "canonical_note_does_not_match_source_row",
                "record_id": record_id,
                "source_row_ordinal": ordinal,
            },
        )
    return identity


def build_source_row_binding(
    record: CommonDatasetRecord,
    piece: Any,
    source: TargetBundle,
    projection: DilemmadataCommonHarmonicProjection,
    corpus_root: str | Path,
    *,
    notes: Sequence[Any],
) -> DilemmadataSourceRowBinding | None:
    """Recover and validate all DLC duplicate-row transitions in one record."""

    if record.dialect != "dlc":
        return None
    if piece.piece_id != record.piece_id or source.piece_id != record.piece_id:
        raise SourceRowBindingError(
            "unresolved",
            {"reason": "record_piece_sidecar_identity_mismatch", "record_id": record.record_id},
        )
    transitions = detect_row_transition_entries(source, projection)
    if not transitions:
        return None
    raw_rows = _read_rows(_source_path(Path(corpus_root), record), record)
    token = record.piece_id.rsplit("-", 1)[-1]
    identity_rows: dict[str, list[int]] = defaultdict(list)
    first_onset: dict[str, Fraction] = {}
    onset_by_ordinal: list[Fraction] = []
    for ordinal, row in enumerate(raw_rows):
        identity = row["unfolded_harmony_index"].strip()
        onset = _fraction(
            row["quarterbeats_playthrough"],
            field="quarterbeats_playthrough",
            record_id=record.record_id,
        )
        onset_by_ordinal.append(onset)
        if not identity:
            continue
        identity_rows[identity].append(ordinal)
        first_onset[identity] = min(onset, first_onset.get(identity, onset))
    ordered_identities = tuple(
        sorted(identity_rows, key=lambda value: (first_onset[value], value))
    )
    next_onset: dict[str, Fraction] = {}
    for index, identity in enumerate(ordered_identities):
        next_onset[identity] = (
            first_onset[ordered_identities[index + 1]]
            if index + 1 < len(ordered_identities)
            else Fraction(piece.duration_qn.num, piece.duration_qn.den)
        )

    groups: list[SourceRowGroup] = []
    for task, start_qn, transition_entries in transitions:
        diagnostic_base = {
            "piece_id": record.piece_id,
            "record_id": record.record_id,
            "start_qn": _time_payload(start_qn),
            "task": task,
        }
        if len(transition_entries) != 2:
            raise SourceRowBindingError(
                "ambiguous",
                {
                    **diagnostic_base,
                    "candidate_entry_indices": [row[0] for row in transition_entries],
                    "reason": "duplicate_transition_entry_count_is_not_two",
                },
            )
        source_identities = tuple(
            identity
            for identity in ordered_identities
            if _rt(first_onset[identity]) == start_qn
        )
        if len(source_identities) != 2:
            raise SourceRowBindingError(
                "unresolved" if len(source_identities) < 2 else "ambiguous",
                {
                    **diagnostic_base,
                    "candidate_source_identities": list(source_identities),
                    "reason": "raw_duplicate_harmony_identity_count_is_not_two",
                },
            )
        bound_entries: list[SourceRowEntry] = []
        entry_by_identity: dict[str, SourceRowEntry] = {}
        ordered_transition_entries = tuple(sorted(transition_entries))
        raw_point_flags = tuple(
            _rt(next_onset[identity]) == start_qn for identity in source_identities
        )
        cached_point_flags = tuple(
            end_qn == start_qn
            for _entry_index, _entity_id, end_qn in ordered_transition_entries
        )
        if raw_point_flags != (True, False) or cached_point_flags != (True, False):
            raise SourceRowBindingError(
                "ambiguous",
                {
                    **diagnostic_base,
                    "cached_point_flags": list(cached_point_flags),
                    "raw_point_flags": list(raw_point_flags),
                    "reason": "source_and_typed_entry_order_do_not_form_point_interval",
                },
            )
        for identity, cached in zip(
            source_identities, ordered_transition_entries, strict=True
        ):
            entry_index, entity_id, end_qn = cached
            if end_qn < start_qn:
                raise SourceRowBindingError(
                    "unresolved",
                    {
                        **diagnostic_base,
                        "reason": "cached_entry_span_is_negative",
                        "source_identity": identity,
                    },
                )
            entry = SourceRowEntry(
                entry_index=entry_index,
                entity_id=entity_id,
                source_identity=identity,
                span_start=start_qn,
                span_end=end_qn,
            )
            bound_entries.append(entry)
            entry_by_identity[identity] = entry
        if len({entry.entry_index for entry in bound_entries}) != 2:
            raise SourceRowBindingError(
                "ambiguous",
                {**diagnostic_base, "reason": "source_identities_map_to_same_entry"},
            )

        boundary_notes = tuple(
            (note_index, note)
            for note_index, note in enumerate(notes)
            if note.onset_qn == start_qn
        )
        assignments: list[SourceRowAssignment] = []
        observed_ordinals: set[int] = set()
        for note_index, note in boundary_notes:
            ordinal = _note_ordinal(note, token=token, record_id=record.record_id)
            if ordinal < 0 or ordinal >= len(raw_rows):
                raise SourceRowBindingError(
                    "unresolved",
                    {
                        **diagnostic_base,
                        "note_id": str(note.note_id),
                        "reason": "source_row_ordinal_out_of_bounds",
                        "source_row_ordinal": ordinal,
                    },
                )
            if ordinal in observed_ordinals:
                raise SourceRowBindingError(
                    "ambiguous",
                    {
                        **diagnostic_base,
                        "reason": "duplicate_note_source_row_ordinal",
                        "source_row_ordinal": ordinal,
                    },
                )
            observed_ordinals.add(ordinal)
            identity = _validate_note_row(
                note, raw_rows[ordinal], ordinal, record_id=record.record_id
            )
            entry = entry_by_identity.get(identity)
            if entry is None:
                raise SourceRowBindingError(
                    "unresolved",
                    {
                        **diagnostic_base,
                        "note_id": str(note.note_id),
                        "reason": "note_source_identity_has_no_transition_entry",
                        "source_identity": identity,
                        "source_row_ordinal": ordinal,
                    },
                )
            assignments.append(
                SourceRowAssignment(
                    note_index=note_index,
                    note_id=str(note.note_id),
                    source_row_ordinal=ordinal,
                    source_identity=identity,
                    entry_index=entry.entry_index,
                    entity_id=entry.entity_id,
                )
            )
        expected_ordinals = {
            ordinal
            for identity in source_identities
            for ordinal in identity_rows[identity]
            if onset_by_ordinal[ordinal]
            == Fraction(start_qn.num, start_qn.den)
            and raw_rows[ordinal]["is_note_onset"].strip() in _TRUE
        }
        if observed_ordinals != expected_ordinals or not assignments:
            raise SourceRowBindingError(
                "unresolved",
                {
                    **diagnostic_base,
                    "expected_source_row_ordinals": sorted(expected_ordinals),
                    "observed_source_row_ordinals": sorted(observed_ordinals),
                    "reason": "row_aligned_boundary_notes_are_not_exactly_covered",
                },
            )
        groups.append(
            SourceRowGroup(
                task=task,
                start_qn=start_qn,
                entries=tuple(sorted(bound_entries, key=lambda row: row.entry_index)),
                assignments=tuple(
                    sorted(assignments, key=lambda row: (row.note_index, row.entry_index))
                ),
            )
        )

    base = {
        "contract_version": SOURCE_ROW_BINDING_VERSION,
        "dialect": record.dialect,
        "groups": tuple(
            sorted(
                groups,
                key=lambda row: (
                    row.task,
                    row.start_qn,
                    tuple(entry.entry_index for entry in row.entries),
                ),
            )
        ),
        "piece_id": record.piece_id,
        "record_id": record.record_id,
        "schema": SOURCE_ROW_BINDING_SCHEMA,
    }
    payload = _binding_payload(**base)
    return DilemmadataSourceRowBinding(
        **base,
        semantic_fingerprint=fingerprint(payload),
    )


__all__ = [
    "DilemmadataSourceRowBinding",
    "SOURCE_ROW_BINDING_SCHEMA",
    "SOURCE_ROW_BINDING_VERSION",
    "SourceRowAssignment",
    "SourceRowBindingError",
    "SourceRowEntry",
    "SourceRowGroup",
    "build_source_row_binding",
    "detect_row_transition_entries",
    "source_row_binding_from_payload",
    "source_row_binding_payload",
]
