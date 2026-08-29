"""Structural label-binding preflight for the frozen Phase 9E-B1 corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

from music_critic.experiments.analysisgnn.contracts import (
    EXPECTED_RECORD_COUNT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_FINGERPRINT,
    TRANSPOSITIONS,
    fingerprint,
    graph_schema_fingerprint,
)
from music_critic.experiments.analysisgnn.dataset import (
    CommonDatasetManifest,
    load_common_record,
)
from music_critic.experiments.analysisgnn.graph import (
    AnalysisGNNGraphError,
    bind_entry_supervision,
    ordered_analysis_notes,
)


LABEL_BINDING_PREFLIGHT_VERSION = "1.0.0"
_TEST_ACCESS_POLICY = (
    "structural_label_binding_only_no_model_predictions_metrics_or_selection"
)
_TRANSPOSITION_BINDING = (
    "computed_once_per_accepted_record_and_reused_for_pitch_only_views"
)
_EXPECTED_VIEW_COUNT = 577 * len(TRANSPOSITIONS) + 71 + 71


def _assert_frozen_manifest(manifest: CommonDatasetManifest) -> None:
    observed_splits = dict(Counter(row.split for row in manifest.records))
    if (
        manifest.record_count != EXPECTED_RECORD_COUNT
        or len(manifest.records) != EXPECTED_RECORD_COUNT
        or manifest.split_counts != EXPECTED_SPLIT_COUNTS
        or observed_splits != EXPECTED_SPLIT_COUNTS
        or manifest.source_split_fingerprint != EXPECTED_SPLIT_FINGERPRINT
    ):
        raise AnalysisGNNGraphError(
            "label-binding preflight requires the frozen 719-record 577/71/71 manifest"
        )


def label_binding_preflight(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
) -> dict[str, object]:
    """Audit every accepted record without model construction or evaluation."""

    _assert_frozen_manifest(manifest)
    counts = Counter()
    task_counts: dict[str, Counter[str]] = {
        "inversion": Counter(),
        "quality": Counter(),
    }
    overlap_rows: list[dict[str, object]] = []
    equivalent_records: set[str] = set()
    conflicting_records: set[str] = set()
    equivalent_groups: set[tuple[str, str, tuple[int, ...]]] = set()
    conflicting_groups: set[tuple[str, str, tuple[int, ...]]] = set()
    for row in sorted(
        manifest.records, key=lambda value: (value.record_id, value.piece_id)
    ):
        piece, targets, projection = load_common_record(cache_root, row)
        notes = ordered_analysis_notes(piece)
        tensors, entries, overlaps = bind_entry_supervision(
            notes,
            targets,
            projection,
            record_id=row.record_id,
            piece_id=row.piece_id,
            fail_on_conflict=False,
        )
        counts["record_count"] += 1
        counts["note_count"] += len(notes)
        for task in ("quality", "inversion"):
            task_entries = tuple(entry for entry in entries if entry.task == task)
            available_count = sum(entry.mask for entry in task_entries)
            unavailable_count = len(task_entries) - available_count
            supervised_note_count = int(tensors[task].ne(-1).sum().item())
            membership_count = int(
                tensors[f"{task}_membership_index"].shape[1]
            )
            for key, value in (
                ("available_source_entry_count", available_count),
                ("unavailable_source_entry_count", unavailable_count),
                ("supervised_note_count", supervised_note_count),
                ("membership_count", membership_count),
            ):
                counts[key] += value
                task_counts[task][key] += value
        for overlap in overlaps:
            evidence = {
                **asdict(overlap),
                "split": row.split,
            }
            overlap_rows.append(evidence)
            key = (
                "equivalent_overlap_note_count"
                if overlap.comparison == "equivalent_available_class"
                else "conflicting_overlap_note_count"
            )
            counts[key] += 1
            task_counts[overlap.task][key] += 1
            counts[f"{key.removesuffix('_note_count')}_extra_membership_count"] += (
                len(overlap.entry_indices) - 1
            )
            if overlap.comparison == "equivalent_available_class":
                equivalent_records.add(row.record_id)
                equivalent_groups.add(
                    (row.record_id, overlap.task, overlap.entry_indices)
                )
            else:
                conflicting_records.add(row.record_id)
                conflicting_groups.add(
                    (row.record_id, overlap.task, overlap.entry_indices)
                )

    for key in (
        "equivalent_overlap_note_count",
        "conflicting_overlap_note_count",
        "equivalent_overlap_extra_membership_count",
        "conflicting_overlap_extra_membership_count",
    ):
        counts.setdefault(key, 0)
    for task in task_counts.values():
        task.setdefault("equivalent_overlap_note_count", 0)
        task.setdefault("conflicting_overlap_note_count", 0)

    split_counts = dict(Counter(row.split for row in manifest.records))
    transposition_view_count = (
        split_counts["train"] * len(TRANSPOSITIONS)
        + split_counts["validation"]
        + split_counts["test"]
    )
    payload: dict[str, object] = {
        "acceptance": counts["conflicting_overlap_note_count"] == 0,
        "contract_version": LABEL_BINDING_PREFLIGHT_VERSION,
        "counts": {
            **dict(sorted(counts.items())),
            "conflicting_overlap_group_count": len(conflicting_groups),
            "equivalent_overlap_group_count": len(equivalent_groups),
            "records_with_conflicting_overlaps": len(conflicting_records),
            "records_with_equivalent_overlaps": len(equivalent_records),
            "transposition_view_count": transposition_view_count,
        },
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "graph_schema_fingerprint": graph_schema_fingerprint(),
        "overlaps": overlap_rows,
        "source_split_fingerprint": manifest.source_split_fingerprint,
        "split_counts": split_counts,
        "task_counts": {
            task: dict(sorted(values.items()))
            for task, values in sorted(task_counts.items())
        },
        "test_access_policy": _TEST_ACCESS_POLICY,
        "transposition_binding": _TRANSPOSITION_BINDING,
    }
    payload["preflight_fingerprint"] = fingerprint(payload)
    return payload


def require_zero_conflicting_overlaps(report: Mapping[str, object]) -> None:
    counts = report.get("counts")
    conflicts = (
        counts.get("conflicting_overlap_note_count")
        if isinstance(counts, dict)
        else None
    )
    if report.get("acceptance") is not True or conflicts != 0:
        overlap_rows = report.get("overlaps")
        first_conflict = (
            next(
                (
                    row
                    for row in overlap_rows
                    if isinstance(row, dict)
                    and row.get("comparison") == "conflicting_available_class"
                ),
                None,
            )
            if isinstance(overlap_rows, list)
            else None
        )
        raise AnalysisGNNGraphError(
            "label-binding preflight found conflicting available-class overlaps: "
            f"count={conflicts!r}, first={json.dumps(first_conflict, sort_keys=True)}"
        )


def validate_label_binding_preflight(
    path: str | Path,
    manifest: CommonDatasetManifest,
) -> dict[str, object]:
    """Verify that a persisted preflight is exact, current, and conflict-free."""

    _assert_frozen_manifest(manifest)
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fingerprint = report.pop("preflight_fingerprint", None)
    observed_fingerprint = fingerprint(report)
    report["preflight_fingerprint"] = expected_fingerprint
    if (
        expected_fingerprint != observed_fingerprint
        or report.get("contract_version") != LABEL_BINDING_PREFLIGHT_VERSION
        or report.get("dataset_manifest_fingerprint")
        != manifest.manifest_fingerprint
        or report.get("source_split_fingerprint")
        != manifest.source_split_fingerprint
        or report.get("graph_schema_fingerprint") != graph_schema_fingerprint()
        or report.get("split_counts") != EXPECTED_SPLIT_COUNTS
        or not isinstance(report.get("counts"), dict)
        or report["counts"].get("record_count") != EXPECTED_RECORD_COUNT
        or report["counts"].get("transposition_view_count")
        != _EXPECTED_VIEW_COUNT
        or not isinstance(report.get("overlaps"), list)
        or not isinstance(report.get("task_counts"), dict)
        or report.get("test_access_policy") != _TEST_ACCESS_POLICY
        or report.get("transposition_binding") != _TRANSPOSITION_BINDING
    ):
        raise AnalysisGNNGraphError(
            "label-binding preflight artifact is malformed, stale, or misbound"
        )
    require_zero_conflicting_overlaps(report)
    return report


__all__ = [
    "LABEL_BINDING_PREFLIGHT_VERSION",
    "label_binding_preflight",
    "require_zero_conflicting_overlaps",
    "validate_label_binding_preflight",
]
