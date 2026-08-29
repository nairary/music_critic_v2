"""Deterministic, read-only forensics for Phase 9E-B1 label conflicts.

This module reports evidence only.  It does not build a model, mutate graph
semantics, authorize training, or calculate evaluation metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    ANALYSISGNN_REPOSITORY,
    DILEMMADATA_COMMIT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_FINGERPRINT,
    TRANSPOSITIONS,
    fingerprint,
)
from music_critic.experiments.analysisgnn.dataset import (
    CommonDatasetManifest,
    CommonDatasetRecord,
    load_common_record,
)
from music_critic.experiments.analysisgnn.graph import (
    AnalysisGNNGraphError,
    LabelOverlap,
    bind_entry_supervision,
    ordered_analysis_notes,
)
from music_critic.experiments.analysisgnn.preflight import (
    LABEL_BINDING_PREFLIGHT_VERSION,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
)
from music_critic.tasks.multisource import SampleTarget, TargetBundle


LABEL_BINDING_FORENSIC_AUDIT_VERSION = "1.0.0"
LABEL_BINDING_FORENSIC_COMPACT_VERSION = "1.0.0"
_SCHEMA = "phase9eb1-label-binding-conflict-forensics"
_COMPACT_SCHEMA = "phase9eb1-label-binding-conflict-forensics-compact"
_UNAVAILABLE = "unavailable"
_COMMON_TASK = {
    "inversion": COMMON_INVERSION_TASK,
    "quality": COMMON_QUALITY_TASK,
}
_MISSING = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
_TRUE = frozenset({"1", "True", "true", "TRUE"})
_EXPECTED_PREFLIGHT_FINGERPRINT = (
    "da1d02a0ab58ce9ad765a37822e59e96ddbdbce2fc20302ce46ebb5c82faa500"
)


def _time(value: RationalTime) -> dict[str, int]:
    return {"den": value.den, "num": value.num}


def _duration(start: RationalTime, end: RationalTime) -> dict[str, int]:
    return _time(end - start)


def _manifest_is_frozen(manifest: CommonDatasetManifest) -> bool:
    return (
        manifest.record_count == EXPECTED_RECORD_COUNT
        and len(manifest.records) == EXPECTED_RECORD_COUNT
        and manifest.split_counts == EXPECTED_SPLIT_COUNTS
        and dict(Counter(row.split for row in manifest.records))
        == EXPECTED_SPLIT_COUNTS
        and manifest.source_split_fingerprint == EXPECTED_SPLIT_FINGERPRINT
    )


def _load_structural_preflight(
    path: str | Path,
    manifest: CommonDatasetManifest,
) -> dict[str, object]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = report.pop("preflight_fingerprint", None)
    observed = fingerprint(report)
    report["preflight_fingerprint"] = expected
    counts = report.get("counts")
    if (
        not _manifest_is_frozen(manifest)
        or expected != observed
        or expected != _EXPECTED_PREFLIGHT_FINGERPRINT
        or report.get("contract_version") != LABEL_BINDING_PREFLIGHT_VERSION
        or report.get("dataset_manifest_fingerprint")
        != manifest.manifest_fingerprint
        or report.get("source_split_fingerprint")
        != manifest.source_split_fingerprint
        or report.get("split_counts") != EXPECTED_SPLIT_COUNTS
        or not isinstance(counts, dict)
        or counts.get("record_count") != EXPECTED_RECORD_COUNT
        or counts.get("transposition_view_count") != 7_066
    ):
        raise AnalysisGNNGraphError(
            "forensic audit requires the exact frozen conflicting preflight"
        )
    return report


def source_conflict_identity(
    *,
    record: CommonDatasetRecord,
    task: str,
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return order-independent source-native identity for one conflict group."""

    ordered_entries = sorted(entries, key=lambda value: str(value["entity_id"]))
    return {
        "dialect": record.dialect,
        "entries": [
            {
                "entity_id": entry["entity_id"],
                "source_task_ids": entry["source_task_ids"],
                "source_values": entry["source_values"],
                "span_end": entry["span_end"],
                "span_start": entry["span_start"],
            }
            for entry in ordered_entries
        ],
        "piece_id": record.piece_id,
        "record_id": record.record_id,
        "task_id": _COMMON_TASK[task],
    }


def classify_conflict_group(
    entries: Sequence[Mapping[str, object]],
    notes: Sequence[Mapping[str, object]],
    *,
    source_row_membership_complete: bool,
) -> dict[str, object]:
    """Classify temporal and note structure without choosing a target class."""

    point_flags = tuple(bool(entry["point_span"]) for entry in entries)
    starts = tuple(entry["span_start"] for entry in entries)
    if len(entries) == 2 and sum(point_flags) == 1:
        span_type = (
            "point_vs_interval_same_start"
            if len(set(json.dumps(value, sort_keys=True) for value in starts)) == 1
            else "point_vs_interval"
        )
    elif entries and all(point_flags):
        span_type = "point_vs_point"
    elif entries and not any(point_flags):
        span_type = "interval_vs_interval"
    else:
        span_type = "other"
    timestamp_type = (
        "duplicate_timestamp_transition"
        if span_type == "point_vs_interval_same_start"
        else "other"
    )
    grace_flags = tuple(
        bool(note["is_grace"]) or bool(note["zero_duration"]) for note in notes
    )
    if grace_flags and all(grace_flags):
        note_type = "grace_note_specific"
    elif grace_flags and not any(grace_flags) and any(point_flags):
        note_type = "ordinary_note_at_point_boundary"
    else:
        note_type = "other"
    return {
        "note_type": note_type,
        "official_source_row_membership": (
            "complete" if source_row_membership_complete else "unresolved"
        ),
        "span_type": span_type,
        "timestamp_type": timestamp_type,
    }


def _target(source: TargetBundle, task_id: str) -> SampleTarget:
    return next(target for target in source.targets if target.task_id == task_id)


def _span_payload(span: Any) -> dict[str, object]:
    return {
        "duration": _duration(span.start_qn, span.end_qn),
        "end": _time(span.end_qn),
        "point_span": span.start_qn == span.end_qn,
        "start": _time(span.start_qn),
    }


def _entry_summary(
    entry_index: int,
    common_target: Any,
    source: TargetBundle,
    span_by_id: Mapping[str, Any],
) -> dict[str, object]:
    entry = common_target.entries[entry_index]
    source_positions: list[dict[str, object]] = []
    for source_task_id, source_value in zip(
        entry.source_task_ids, entry.source_values, strict=True
    ):
        source_target = _target(source, source_task_id)
        source_entry_order = source_target.entity_ids.index(entry.entity_id)
        source_positions.append(
            {
                "availability": source_target.availability_mask[source_entry_order],
                "source_entry_order": source_entry_order,
                "source_task_id": source_task_id,
                "source_value": source_value,
            }
        )
    span = span_by_id[entry.entity_id]
    return {
        "availability": entry.state in {"exact", "coarsened"},
        "common_value": entry.common_value,
        "entity_id": entry.entity_id,
        "entry_index": entry_index,
        "mapping_state": entry.state,
        "source_positions": source_positions,
        "source_task_ids": list(entry.source_task_ids),
        "source_values": list(entry.source_values),
        "span_end": _time(span.end_qn),
        "span_start": _time(span.start_qn),
        **_span_payload(span),
    }


def _neighbor(
    entry_index: int,
    common_target: Any,
    source: TargetBundle,
    span_by_id: Mapping[str, Any],
) -> dict[str, object] | str:
    if entry_index < 0 or entry_index >= len(common_target.entries):
        return _UNAVAILABLE
    row = _entry_summary(entry_index, common_target, source, span_by_id)
    return {
        key: row[key]
        for key in (
            "common_value",
            "entity_id",
            "entry_index",
            "mapping_state",
            "source_values",
            "span_end",
            "span_start",
        )
    }


def _source_path(corpus_root: Path, record: CommonDatasetRecord) -> Path:
    if record.dialect != "dlc":
        raise AnalysisGNNGraphError(
            "source-row forensics currently requires the DLC pitch-array dialect"
        )
    dialect, collection, piece_name = record.record_id.split(":", 2)
    if dialect != "dlc":
        raise AnalysisGNNGraphError("record identity differs from its DLC dialect")
    return corpus_root / "pitch_arrays" / "DLC" / collection / f"{piece_name}.tsv"


def _fraction(raw: str) -> Fraction:
    return Fraction(raw.strip())


def _source_row_evidence(
    corpus_root: Path,
    record: CommonDatasetRecord,
    source: TargetBundle,
    source_task_ids: Iterable[str],
) -> dict[tuple[str, str], dict[str, object]]:
    """Recover original TSV row ordinals for selected TRAIN/VALIDATION entries."""

    path = _source_path(corpus_root, record)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise AnalysisGNNGraphError("forensic source TSV header is invalid")
        required = {"quarterbeats_playthrough", "unfolded_harmony_index"}
        if not required <= set(reader.fieldnames):
            raise AnalysisGNNGraphError("forensic source TSV lacks identity fields")
        rows = tuple(dict(row) for row in reader)

    result: dict[tuple[str, str], dict[str, object]] = {}
    for task_id in sorted(set(source_task_ids)):
        spec = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
        grouped: dict[str, list[tuple[int, Mapping[str, str]]]] = defaultdict(list)
        first_onset: dict[str, Fraction] = {}
        for ordinal, row in enumerate(rows):
            onset = _fraction(row["quarterbeats_playthrough"])
            primary = row.get(spec.primary_field, "").strip()
            gate = (
                row.get(spec.gate_field, "").strip()
                if spec.gate_field is not None
                else "True"
            )
            available = (
                gate in _TRUE
                and primary not in _MISSING
                and (spec.vocabulary is None or primary in spec.vocabulary)
            )
            identity = (
                row.get(spec.source_identity_field, "").strip()
                if available and spec.source_identity_field is not None
                else ""
            )
            if identity in _MISSING:
                identity = f"onset:{onset.numerator}/{onset.denominator}"
            grouped[identity].append((ordinal, row))
            first_onset[identity] = min(onset, first_onset.get(identity, onset))
        ordered_groups = tuple(
            grouped[identity]
            for identity in sorted(
                grouped, key=lambda value: (first_onset[value], value)
            )
        )
        target = _target(source, task_id)
        if len(target.entity_ids) != len(ordered_groups):
            raise AnalysisGNNGraphError(
                "source-row group count differs from cached source target"
            )
        for entry_order, (entity_id, group_rows) in enumerate(
            zip(target.entity_ids, ordered_groups, strict=True)
        ):
            raw_values = tuple(
                sorted(
                    {
                        row.get(
                            "chord_type" if task_id.endswith("quality") else "figbass",
                            "",
                        ).strip()
                        for _ordinal, row in group_rows
                    }
                    - {""}
                )
            )
            result[(task_id, entity_id)] = {
                "source_entry_order": entry_order,
                "source_identity": group_rows[0][1]["unfolded_harmony_index"],
                "source_row_lines": [ordinal + 2 for ordinal, _row in group_rows],
                "source_row_ordinals": [ordinal for ordinal, _row in group_rows],
                "source_row_value_set": list(raw_values),
            }
    return result


def _note_source_ordinal(note_id: str) -> int | str:
    try:
        return int(note_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return _UNAVAILABLE


def _entry_forensics(
    entry_index: int,
    common_target: Any,
    source: TargetBundle,
    span_by_id: Mapping[str, Any],
    source_rows: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    row = _entry_summary(entry_index, common_target, source, span_by_id)
    raw_rows = [
        source_rows.get((position["source_task_id"], row["entity_id"]))
        for position in row["source_positions"]
    ]
    if any(value is None for value in raw_rows):
        original_rows: object = _UNAVAILABLE
    else:
        original_rows = raw_rows
    row.update(
        {
            "common_class_index": None,
            "next_source_annotation": _neighbor(
                entry_index + 1, common_target, source, span_by_id
            ),
            "original_row_order": original_rows,
            "previous_source_annotation": _neighbor(
                entry_index - 1, common_target, source, span_by_id
            ),
        }
    )
    return row


def _note_forensics(
    note: Any,
    entry_rows: Sequence[Mapping[str, object]],
    class_ids: Sequence[int],
) -> dict[str, object]:
    ordinal = _note_source_ordinal(str(note.note_id))
    official_entries: list[str] = []
    if isinstance(ordinal, int):
        for entry in entry_rows:
            original = entry["original_row_order"]
            if not isinstance(original, list):
                continue
            if any(
                ordinal in raw["source_row_ordinals"]
                for raw in original
                if isinstance(raw, dict)
            ):
                official_entries.append(str(entry["entity_id"]))
    return {
        "covered_entries": [
            {
                "class_index": class_id,
                "common_value": entry["common_value"],
                "entity_id": entry["entity_id"],
                "entry_index": entry["entry_index"],
            }
            for entry, class_id in zip(entry_rows, class_ids, strict=True)
        ],
        "duration": _time(note.duration_qn),
        "is_grace": bool(note.is_grace),
        "note_id": str(note.note_id),
        "offset": _time(note.onset_qn + note.duration_qn),
        "official_source_row_entries": official_entries,
        "onset": _time(note.onset_qn),
        "part": _UNAVAILABLE,
        "pitch": int(note.pitch),
        "source_row_ordinal": ordinal,
        "staff": note.staff if note.staff is not None else _UNAVAILABLE,
        "track": note.track_id if note.track_id else _UNAVAILABLE,
        "voice": note.voice if note.voice is not None else _UNAVAILABLE,
        "zero_duration": note.duration_qn == RationalTime(0),
    }


def _group_overlaps(
    overlaps: Iterable[LabelOverlap],
) -> tuple[
    dict[tuple[str, tuple[int, ...]], list[LabelOverlap]],
    set[tuple[str, tuple[int, ...]]],
]:
    conflicts: dict[tuple[str, tuple[int, ...]], list[LabelOverlap]] = defaultdict(list)
    equivalents: set[tuple[str, tuple[int, ...]]] = set()
    for overlap in overlaps:
        key = (overlap.task, tuple(sorted(overlap.entry_indices)))
        if overlap.comparison == "conflicting_available_class":
            conflicts[key].append(overlap)
        elif overlap.comparison == "equivalent_available_class":
            equivalents.add(key)
    if set(conflicts) & equivalents:
        raise AnalysisGNNGraphError(
            "one source group cannot be both equivalent and conflicting"
        )
    return conflicts, equivalents


def _record_conflicts(
    record: CommonDatasetRecord,
    piece: Any,
    source: TargetBundle,
    projection: Any,
    *,
    corpus_root: Path | None,
) -> tuple[list[dict[str, object]], set[tuple[str, tuple[int, ...]]]]:
    notes = ordered_analysis_notes(piece)
    _tensors, _entries, overlaps = bind_entry_supervision(
        notes,
        source,
        projection,
        record_id=record.record_id,
        piece_id=record.piece_id,
        dialect=record.dialect,
        allow_historical_span_forensics=True,
        fail_on_conflict=False,
    )
    grouped, equivalents = _group_overlaps(overlaps)
    if not grouped:
        return [], equivalents
    span_by_id = {span.annotation_id: span for span in source.alignment_spans}
    required_source_tasks = {
        task_id
        for task, entry_indices in grouped
        for entry_index in entry_indices
        for task_id in next(
            target for target in projection.targets if target.task_id == _COMMON_TASK[task]
        ).entries[entry_index].source_task_ids
    }
    source_rows = (
        _source_row_evidence(corpus_root, record, source, required_source_tasks)
        if corpus_root is not None and record.split in {"train", "validation"}
        else {}
    )
    result: list[dict[str, object]] = []
    for (task, entry_indices), note_overlaps in sorted(grouped.items()):
        common_target = next(
            target for target in projection.targets if target.task_id == _COMMON_TASK[task]
        )
        entry_rows = [
            _entry_forensics(
                entry_index, common_target, source, span_by_id, source_rows
            )
            for entry_index in entry_indices
        ]
        class_by_index = dict(
            zip(note_overlaps[0].entry_indices, note_overlaps[0].class_ids, strict=True)
        )
        class_ids = [class_by_index[index] for index in entry_indices]
        for entry, class_id in zip(entry_rows, class_ids, strict=True):
            entry["common_class_index"] = class_id
        note_rows = [
            _note_forensics(notes[overlap.note_index], entry_rows, class_ids)
            for overlap in sorted(note_overlaps, key=lambda value: value.note_id)
        ]
        source_row_membership_complete = bool(note_rows) and all(
            len(note["official_source_row_entries"]) == 1 for note in note_rows
        )
        identity = source_conflict_identity(
            record=record,
            task=task,
            entries=entry_rows,
        )
        identity_fingerprint = fingerprint(identity)
        transpositions = TRANSPOSITIONS if record.split == "train" else ("P1",)
        result.append(
            {
                "classification": classify_conflict_group(
                    entry_rows,
                    note_rows,
                    source_row_membership_complete=source_row_membership_complete,
                ),
                "common_projection_fingerprint": (
                    record.common_projection_fingerprint
                ),
                "conflicting_notes": note_rows,
                "dataset_source_group_id": record.source_group_id,
                "dialect": record.dialect,
                "effective_conflicting_note_count": len(note_rows)
                * len(transpositions),
                "effective_view_count": len(transpositions),
                "entries": entry_rows,
                "identity": identity,
                "piece_id": record.piece_id,
                "raw_projection_sha256": record.raw_projection_sha256,
                "record_id": record.record_id,
                "source_conflict_group_id": (
                    f"label-binding-conflict:{identity_fingerprint}"
                ),
                "source_conflict_identity_fingerprint": identity_fingerprint,
                "source_conflicting_note_count": len(note_rows),
                "split": record.split,
                "target_bundle_fingerprint": record.target_bundle_fingerprint,
                "task": task,
                "task_id": _COMMON_TASK[task],
                "transpositions": list(transpositions),
            }
        )
    return result, equivalents


def _count_groups(groups: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records = {str(group["record_id"]) for group in groups}
    entry_ids = {
        (str(group["record_id"]), str(group["task"]), str(entry["entity_id"]))
        for group in groups
        for entry in group["entries"]  # type: ignore[union-attr]
    }
    return {
        "effective_conflicting_note_occurrence_count": sum(
            int(group["effective_conflicting_note_count"]) for group in groups
        ),
        "effective_group_occurrence_count": sum(
            int(group["effective_view_count"]) for group in groups
        ),
        "source_conflicting_note_count": sum(
            int(group["source_conflicting_note_count"]) for group in groups
        ),
        "source_entry_count": len(entry_ids),
        "source_group_count": len(groups),
        "source_record_count": len(records),
    }


def _dimensional_counts(
    groups: Sequence[Mapping[str, object]], key: str
) -> dict[str, dict[str, int]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        if key == "classification":
            dimensions = group["classification"]  # type: ignore[assignment]
            assert isinstance(dimensions, dict)
            labels = (
                f"note_type:{dimensions['note_type']}",
                f"span_type:{dimensions['span_type']}",
                f"timestamp_type:{dimensions['timestamp_type']}",
            )
        else:
            labels = (str(group[key]),)
        for label in labels:
            values[label]["source_group_count"] += 1
            values[label]["source_note_count"] += int(
                group["source_conflicting_note_count"]
            )
            values[label]["effective_group_occurrence_count"] += int(
                group["effective_view_count"]
            )
            values[label]["effective_note_occurrence_count"] += int(
                group["effective_conflicting_note_count"]
            )
    return {
        label: dict(sorted(counts.items()))
        for label, counts in sorted(values.items())
    }


def _official_source_evidence() -> dict[str, object]:
    return {
        "analysisgnn": {
            "commit": ANALYSISGNN_COMMIT,
            "repository": ANALYSISGNN_REPOSITORY,
            "rows": [
                {
                    "applicability": "DLC TSV rows are retained in file order; equal timestamps are neither sorted nor deduplicated here.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "load_labeled_pitch_array",
                    "lines": "78-94",
                    "semantics": "pandas.read_csv reads the TSV; optional processing only drops rows missing selected fields.",
                },
                {
                    "applicability": "Every retained dataframe row becomes one note-array row; duration is derived row-wise and no same-onset conflict resolver runs.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "create_graph_from_df",
                    "lines": "127-160",
                    "semantics": "Uses dataframe onset/duration/pitch fields directly to construct the note array.",
                },
                {
                    "applicability": "Quality and inversion labels remain aligned one-to-one with the retained source rows.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "create_graph_from_df",
                    "lines": "195-203",
                    "semantics": "Attaches every label array directly to graph note nodes without span expansion or forward fill.",
                },
                {
                    "applicability": "Quality and inversion are encoded independently for each dataframe row; the function defines no precedence between chord rows.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "create_labels_dlc",
                    "lines": "374-388",
                    "semantics": "Encodes DLC quality and inversion columns row-wise.",
                },
                {
                    "applicability": "Defines inversion class conversion only; it does not resolve temporal overlap.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "process_inversion_from_chord",
                    "lines": "279-290",
                    "semantics": "Maps figured-bass strings to root/first/second/third inversion indices.",
                },
                {
                    "applicability": "This is prediction pooling after row-wise label construction, not a source-label conflict policy.",
                    "file": "analysisgnn/models/analysis.py",
                    "file_sha256": "c5efc9086ce101a0732fb883ef508d09040a7109abc6b73bdee4bc9ba396dc5a",
                    "function": "onsetwise_logit_aggregation",
                    "lines": "44-101",
                    "semantics": "Pools predicted logits across onset edges and does not choose among source chord labels.",
                },
                {
                    "applicability": "Transposition multiplies graph views, not source-native annotation identities.",
                    "file": "analysisgnn/data/datasets/dlc.py",
                    "file_sha256": "3144af37692c708916f4d90924bc8b3dd63beceb2859013c6ef3b9db62853e36",
                    "function": "DLCGraphDataset._process_single",
                    "lines": "365-405",
                    "semantics": "Creates one graph per configured interval except its own test nicknames, which use P1.",
                },
                {
                    "applicability": "No explicit grace/zero-duration label rule exists in the audited DLC loader or label encoder; downstream graph handling is unresolved for policy selection.",
                    "file": "analysisgnn/utils/dcl_tsv_utils.py",
                    "file_sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
                    "function": "create_graph_from_df/create_labels_dlc",
                    "lines": "97-203,374-444",
                    "semantics": "No grace-specific branch or zero-duration label precedence is defined.",
                },
            ],
        },
        "dilemmadata_pitch_array_producer": {
            "commit": DILEMMADATA_COMMIT,
            "rows": [
                {
                    "applicability": "At one timestamp, multiple harmony rows can duplicate note rows; each resulting TSV row carries its own harmony identity/value.",
                    "file": "processing/utils.py",
                    "file_sha256": "6940742bd5b34d780b5db64083b6755234b0ed54d3bb016c6559dbddc96b9aad",
                    "function": "make_labeled_pitch_array",
                    "lines": "1109-1188",
                    "semantics": "Outer-merges notes and labels with sort=True, then forward-fills harmony fields within each unfolded harmony identity before dropping label-only rows.",
                }
            ],
        },
        "conclusion": {
            "official_rule": "row_aligned_source_label",
            "span_precedence": "unresolved_and_not_implemented_by_official_source",
            "v2_applicability": "Source-row ordinals retained by V2 note IDs can distinguish the duplicated same-onset rows in all detailed TRAIN/VALIDATION conflicts.",
        },
    }


def _policy_comparison(
    train_validation_groups: Sequence[Mapping[str, object]],
    sealed_test_counts: Mapping[str, object],
) -> list[dict[str, object]]:
    visible = _count_groups(train_validation_groups)
    notes = int(visible["source_conflicting_note_count"])
    entries = int(visible["source_entry_count"])
    records = int(visible["source_record_count"])
    sealed_notes = int(sealed_test_counts["source_conflicting_note_count"])
    sealed_entries = int(sealed_test_counts["source_entry_count"])
    common = {
        "changes_persisted_source_targets": False,
        "policy_selection_scope": "train_and_validation_only",
        "potential_sealed_test_entry_count": sealed_entries,
        "potential_sealed_test_note_count": sealed_notes,
        "test_values_used_for_comparison": False,
    }
    return [
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "high",
            "entry_level_evaluation": "technically_possible_but_interval_supervision_is_overridden",
            "leakage_risk": "low_if_fixed_without_test_inspection",
            "official_support": "contradicted_by_row_aligned_labels",
            "policy": "point_span_precedence",
            "subset_or_split_change": False,
        },
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "high",
            "entry_level_evaluation": "technically_possible_but_point_supervision_is_overridden",
            "leakage_risk": "low_if_fixed_without_test_inspection",
            "official_support": "contradicted_by_row_aligned_labels",
            "policy": "interval_span_precedence",
            "subset_or_split_change": False,
        },
        {
            **common,
            "affected_source_entry_count": 0,
            "affected_source_note_count": 0,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "medium",
            "entry_level_evaluation": "unresolved_conflicts_remain",
            "leakage_risk": "low",
            "official_support": "no_grace_specific_rule_found",
            "policy": "point_span_only_for_grace_or_zero_duration_notes",
            "subset_or_split_change": False,
            "unresolved_source_note_count": notes,
        },
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "low_only_for_exact_source_row_identity_otherwise_high",
            "entry_level_evaluation": "viable_if_each_source_row_retains_its_entry_membership",
            "leakage_risk": "low_if_bound_only_to_source_row_provenance",
            "official_support": "conditional_exact_support_for_source_row_identity_not_first_or_last_precedence",
            "policy": "deterministic_same_onset_ordering",
            "subset_or_split_change": False,
        },
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "low",
            "entry_level_evaluation": "possible_but_conflicting_entries_lack_unambiguous_training_supervision",
            "leakage_risk": "low_if_task_local_and_fixed_before_test_unlock",
            "official_support": "not_equivalent_to_official_semantics_but_minimal_unresolved_fallback",
            "policy": "mask_conflicting_task_note_targets",
            "subset_or_split_change": False,
        },
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "applies_equally_to_quality_and_inversion": True,
            "arbitrary_selection_risk": "low",
            "entry_level_evaluation": "impossible_for_masked_source_entries",
            "leakage_risk": "low_if_fixed_before_test_unlock",
            "official_support": "not_equivalent_to_official_semantics_and_discards_source_targets",
            "policy": "mask_conflicting_source_entries",
            "subset_or_split_change": False,
        },
        {
            **common,
            "affected_source_entry_count": entries,
            "affected_source_note_count": notes,
            "affected_source_record_count": records,
            "applies_equally_to_quality_and_inversion": False,
            "arbitrary_selection_risk": "high_and_overbroad",
            "entry_level_evaluation": "impossible_for_all_tasks_in_excluded_records",
            "leakage_risk": "high_if_test_membership_influences_exclusion_policy",
            "official_support": "unsupported",
            "policy": "exclude_entire_record",
            "subset_or_split_change": True,
        },
    ]


def label_binding_forensic_audit(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    preflight_path: str | Path,
    *,
    corpus_root: str | Path,
) -> dict[str, object]:
    """Run one bounded source-native audit over the existing 719-record cache."""

    preflight = _load_structural_preflight(preflight_path, manifest)
    all_groups: list[dict[str, object]] = []
    equivalent_groups: set[tuple[str, str, tuple[int, ...]]] = set()
    corpus_path = Path(corpus_root)
    for record in sorted(
        manifest.records, key=lambda value: (value.record_id, value.piece_id)
    ):
        piece, source, projection = load_common_record(cache_root, record)
        groups, equivalent = _record_conflicts(
            record,
            piece,
            source,
            projection,
            corpus_root=corpus_path,
        )
        all_groups.extend(groups)
        equivalent_groups.update(
            (record.record_id, task, indices) for task, indices in equivalent
        )
    all_groups.sort(key=lambda value: str(value["source_conflict_group_id"]))
    train_validation_groups = [
        group for group in all_groups if group["split"] in {"train", "validation"}
    ]
    test_groups = [group for group in all_groups if group["split"] == "test"]
    aggregate = _count_groups(all_groups)
    visible_counts = _count_groups(train_validation_groups)
    sealed_test_counts = _count_groups(test_groups)
    preflight_counts = preflight["counts"]
    assert isinstance(preflight_counts, dict)
    if (
        aggregate["source_group_count"]
        != preflight_counts.get("conflicting_overlap_group_count")
        or aggregate["source_conflicting_note_count"]
        != preflight_counts.get("conflicting_overlap_note_count")
    ):
        raise AnalysisGNNGraphError(
            "forensic source counts differ from the structural preflight"
        )
    aggregate.update(
        {
            "equivalent_source_group_count": len(equivalent_groups),
            "preflight_corpus_effective_view_count": preflight_counts[
                "transposition_view_count"
            ],
            "transposition_repetition_extra_group_count": (
                int(aggregate["effective_group_occurrence_count"])
                - int(aggregate["source_group_count"])
            ),
            "transposition_repetition_extra_note_count": (
                int(aggregate["effective_conflicting_note_occurrence_count"])
                - int(aggregate["source_conflicting_note_count"])
            ),
        }
    )
    sealed_test = {
        **sealed_test_counts,
        "details_sealed": True,
        "model_evaluation_performed": False,
        "note_level_diagnostics_exposed": False,
        "policy_selection_used_test_targets": False,
        "source_entry_classes_exposed": False,
        "source_values_exposed": False,
    }
    policy_comparison = _policy_comparison(train_validation_groups, sealed_test)
    payload: dict[str, object] = {
        "acceptance": False,
        "aggregate_counts": aggregate,
        "classification_counts": _dimensional_counts(all_groups, "classification"),
        "count_interpretation": {
            "correction": "The preflight computed overlap rows once per accepted source record; 18 groups and 128 notes were already source-native counts, while 7066 described the whole corpus view schedule.",
            "effective_occurrences": "Each TRAIN source conflict repeats over 12 pitch-only transpositions; validation and TEST use P1 only.",
        },
        "deduplication_contract": {
            "entry_order": "entity_id_lexicographic",
            "identity_fields": [
                "record_id",
                "piece_id",
                "dialect",
                "task_id",
                "source_entry_entity_ids",
                "source_native_span_boundaries",
                "source_native_source_values",
            ],
            "source_conflict_group_id": "label-binding-conflict:sha256(canonical_identity_json)",
            "transpositions_are_not_source_identities": True,
        },
        "input_bindings": {
            "assignment_fingerprint": manifest.assignment_fingerprint,
            "common_registry_fingerprint": manifest.common_registry_fingerprint,
            "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
            "dilemmadata_commit": DILEMMADATA_COMMIT,
            "preflight_fingerprint": preflight["preflight_fingerprint"],
            "raw_index_fingerprint": manifest.raw_index_fingerprint,
            "records_fingerprint": manifest.records_fingerprint,
            "source_split_fingerprint": manifest.source_split_fingerprint,
            "split_counts": manifest.split_counts,
        },
        "official_source_evidence": _official_source_evidence(),
        "policy_comparison": policy_comparison,
        "recommendation": {
            "confidence": "high_for_observed_train_validation_groups",
            "fallback_if_source_row_identity_cannot_be_contract_bound": "mask_conflicting_task_note_targets",
            "policy": "preserve_exact_source_row_to_entry_membership",
            "rationale": "Pinned AnalysisGNN consumes one label per TSV row. Every detailed conflict has exact one-entry source-row membership recoverable from the immutable source ordinal retained in its V2 note ID; point/interval precedence would erase valid row-aligned supervision.",
            "scope": "next_separate_remediation_not_implemented_by_this_audit",
            "test_targets_used": False,
        },
        "schema": _SCHEMA,
        "sealed_test": sealed_test,
        "split_counts": _dimensional_counts(all_groups, "split"),
        "task_counts": _dimensional_counts(all_groups, "task"),
        "test_targets_used_for_model_evaluation": False,
        "training_authorized": False,
        "train_validation_counts": visible_counts,
        "train_validation_conflicts": train_validation_groups,
        "version": LABEL_BINDING_FORENSIC_AUDIT_VERSION,
    }
    payload["semantic_fingerprint"] = fingerprint(payload)
    return payload


def compact_forensic_evidence(report: Mapping[str, object]) -> dict[str, object]:
    """Create a reviewable committed projection without TEST or note-level rows."""

    groups: list[dict[str, object]] = []
    visible = report["train_validation_conflicts"]
    assert isinstance(visible, list)
    for group in visible:
        assert isinstance(group, dict)
        entries = []
        for entry in group["entries"]:
            assert isinstance(entry, dict)
            entries.append(
                {
                    key: entry[key]
                    for key in (
                        "common_class_index",
                        "common_value",
                        "entity_id",
                        "entry_index",
                        "mapping_state",
                        "point_span",
                        "source_task_ids",
                        "source_values",
                        "span_end",
                        "span_start",
                    )
                }
            )
        groups.append(
            {
                "classification": group["classification"],
                "effective_conflicting_note_count": group[
                    "effective_conflicting_note_count"
                ],
                "effective_view_count": group["effective_view_count"],
                "entries": entries,
                "piece_id": group["piece_id"],
                "record_id": group["record_id"],
                "source_conflict_group_id": group["source_conflict_group_id"],
                "source_conflicting_note_count": group[
                    "source_conflicting_note_count"
                ],
                "split": group["split"],
                "task": group["task"],
                "transpositions": group["transpositions"],
            }
        )
    compact: dict[str, object] = {
        key: report[key]
        for key in (
            "acceptance",
            "aggregate_counts",
            "classification_counts",
            "count_interpretation",
            "deduplication_contract",
            "input_bindings",
            "official_source_evidence",
            "policy_comparison",
            "recommendation",
            "sealed_test",
            "split_counts",
            "task_counts",
            "test_targets_used_for_model_evaluation",
            "training_authorized",
            "train_validation_counts",
        )
    }
    compact.update(
        {
            "full_artifact_semantic_fingerprint": report["semantic_fingerprint"],
            "schema": _COMPACT_SCHEMA,
            "train_validation_conflicts": groups,
            "version": LABEL_BINDING_FORENSIC_COMPACT_VERSION,
        }
    )
    compact["semantic_fingerprint"] = fingerprint(compact)
    return compact


__all__ = [
    "LABEL_BINDING_FORENSIC_AUDIT_VERSION",
    "LABEL_BINDING_FORENSIC_COMPACT_VERSION",
    "classify_conflict_group",
    "compact_forensic_evidence",
    "label_binding_forensic_audit",
    "source_conflict_identity",
]
