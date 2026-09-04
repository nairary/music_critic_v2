"""Source-row label-binding preflight for the frozen Phase 9E-B1 corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from music_critic.experiments.analysisgnn.source_row_binding import (
    DilemmadataSourceRowBinding,
    SOURCE_ROW_BINDING_VERSION,
    SourceRowBindingError,
    build_source_row_binding,
    detect_row_transition_entries,
    source_row_binding_from_payload,
    source_row_binding_payload,
)


LABEL_BINDING_PREFLIGHT_VERSION = "2.0.0"
_TEST_ACCESS_POLICY = (
    "sealed_structural_source_row_binding_no_model_predictions_metrics_or_selection"
)
_TRANSPOSITION_BINDING = (
    "source_row_identity_computed_once_and_reused_for_pitch_only_views"
)
_EXPECTED_VIEW_COUNT = 577 * len(TRANSPOSITIONS) + 71 + 71
_SUPERSEDED_FORENSIC_FINGERPRINT = (
    "b425c470da9cd9754a4cbedb240a44835391ad2dac9dbcd11ba108aef66d40e9"
)
_SUPERSEDED_COMPACT_FINGERPRINT = (
    "4dc5db61525be807a49677893b6f5b338eb35b0f59854ed508cf0d38f5f012ba"
)


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


def _effective_multiplier(split: str) -> int:
    return len(TRANSPOSITIONS) if split == "train" else 1


def _add(
    source: Counter[str],
    effective: Counter[str],
    key: str,
    value: int,
    *,
    split: str,
) -> None:
    source[key] += value
    effective[key] += value * _effective_multiplier(split)


def _plain(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def label_binding_preflight(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    *,
    corpus_root: str | Path,
) -> dict[str, object]:
    """Audit all 719 records before graph/model/optimizer construction."""

    _assert_frozen_manifest(manifest)
    source_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    split_source: dict[str, Counter[str]] = defaultdict(Counter)
    split_effective: dict[str, Counter[str]] = defaultdict(Counter)
    task_source: dict[str, Counter[str]] = defaultdict(Counter)
    task_effective: dict[str, Counter[str]] = defaultdict(Counter)
    tv_bindings: list[dict[str, object]] = []
    tv_diagnostics: list[dict[str, object]] = []
    row_records: set[str] = set()
    equivalent_records: set[str] = set()
    conflict_records: set[str] = set()
    equivalent_groups: set[tuple[str, str, tuple[int, ...]]] = set()
    conflict_groups: set[tuple[str, str, tuple[int, ...]]] = set()
    split_by_record = {row.record_id: row.split for row in manifest.records}

    for row in sorted(
        manifest.records, key=lambda value: (value.record_id, value.piece_id)
    ):
        piece, targets, projection = load_common_record(cache_root, row)
        notes = ordered_analysis_notes(piece)
        transitions = (
            detect_row_transition_entries(targets, projection)
            if row.dialect == "dlc"
            else ()
        )
        source_counts["records_checked"] += 1
        source_counts["record_count"] += 1
        source_counts["note_count"] += len(notes)
        split_source[row.split]["records_checked"] += 1
        split_source[row.split]["note_count"] += len(notes)
        effective_counts["record_views_checked"] += _effective_multiplier(row.split)
        split_effective[row.split]["record_views_checked"] += _effective_multiplier(
            row.split
        )
        _add(
            source_counts,
            effective_counts,
            "row_aligned_groups_detected",
            len(transitions),
            split=row.split,
        )
        split_source[row.split]["row_aligned_groups_detected"] += len(transitions)
        split_effective[row.split]["row_aligned_groups_detected"] += (
            len(transitions) * _effective_multiplier(row.split)
        )
        for task, _start, _entries in transitions:
            task_source[task]["row_aligned_groups_detected"] += 1
            task_effective[task]["row_aligned_groups_detected"] += (
                _effective_multiplier(row.split)
            )

        try:
            binding = build_source_row_binding(
                row,
                piece,
                targets,
                projection,
                corpus_root,
                notes=notes,
            )
        except SourceRowBindingError as exc:
            key = f"{exc.category}_row_groups"
            value = max(len(transitions), 1)
            _add(source_counts, effective_counts, key, value, split=row.split)
            split_source[row.split][key] += value
            split_effective[row.split][key] += value * _effective_multiplier(row.split)
            if row.split != "test":
                tv_diagnostics.append(
                    {
                        "category": exc.category,
                        "diagnostic": exc.diagnostic,
                        "piece_id": row.piece_id,
                        "record_id": row.record_id,
                        "split": row.split,
                    }
                )
            continue

        if binding is not None:
            if len(binding.groups) != len(transitions):
                raise AnalysisGNNGraphError(
                    "source-row builder did not resolve every detected transition"
                )
            row_records.add(row.record_id)
            if row.split != "test":
                tv_bindings.append(
                    {"split": row.split, "binding": source_row_binding_payload(binding)}
                )
            for group in binding.groups:
                assignment_count = len(group.assignments)
                _add(
                    source_counts,
                    effective_counts,
                    "row_aligned_groups_resolved",
                    1,
                    split=row.split,
                )
                _add(
                    source_counts,
                    effective_counts,
                    "row_aligned_notes_resolved",
                    assignment_count,
                    split=row.split,
                )
                split_source[row.split]["row_aligned_groups_resolved"] += 1
                split_source[row.split]["row_aligned_notes_resolved"] += assignment_count
                split_effective[row.split]["row_aligned_groups_resolved"] += (
                    _effective_multiplier(row.split)
                )
                split_effective[row.split]["row_aligned_notes_resolved"] += (
                    assignment_count * _effective_multiplier(row.split)
                )
                task_source[group.task]["row_aligned_groups_resolved"] += 1
                task_source[group.task]["row_aligned_notes_resolved"] += assignment_count
                task_effective[group.task]["row_aligned_groups_resolved"] += (
                    _effective_multiplier(row.split)
                )
                task_effective[group.task]["row_aligned_notes_resolved"] += (
                    assignment_count * _effective_multiplier(row.split)
                )

        tensors, entries, overlaps = bind_entry_supervision(
            notes,
            targets,
            projection,
            record_id=row.record_id,
            piece_id=row.piece_id,
            dialect=row.dialect,
            source_row_binding=binding,
            fail_on_conflict=False,
        )
        for task in ("quality", "inversion"):
            task_entries = tuple(entry for entry in entries if entry.task == task)
            values = {
                "available_source_entry_count": sum(entry.mask for entry in task_entries),
                "unavailable_source_entry_count": sum(
                    not entry.mask for entry in task_entries
                ),
                "supervised_note_count": int(tensors[task].ne(-1).sum().item()),
                "membership_count": int(
                    tensors[f"{task}_membership_index"].shape[1]
                ),
            }
            for key, value in values.items():
                _add(source_counts, effective_counts, key, value, split=row.split)
                split_source[row.split][key] += value
                split_effective[row.split][key] += value * _effective_multiplier(
                    row.split
                )
                task_source[task][key] += value
                task_effective[task][key] += value * _effective_multiplier(row.split)
        for overlap in overlaps:
            group_key = (row.record_id, overlap.task, overlap.entry_indices)
            if overlap.comparison == "equivalent_available_class":
                key = "equivalent_non_row_overlap_notes"
                equivalent_groups.add(group_key)
                equivalent_records.add(row.record_id)
            else:
                key = "remaining_conflicting_notes"
                conflict_groups.add(group_key)
                conflict_records.add(row.record_id)
            _add(source_counts, effective_counts, key, 1, split=row.split)
            split_source[row.split][key] += 1
            split_effective[row.split][key] += _effective_multiplier(row.split)
            task_source[overlap.task][key] += 1
            task_effective[overlap.task][key] += _effective_multiplier(row.split)

    source_counts["equivalent_non_row_overlap_groups"] = len(equivalent_groups)
    source_counts["remaining_conflicting_groups"] = len(conflict_groups)
    source_counts["records_with_row_aligned_groups"] = len(row_records)
    source_counts["records_with_equivalent_non_row_overlaps"] = len(equivalent_records)
    source_counts["records_with_remaining_conflicts"] = len(conflict_records)
    effective_counts["equivalent_non_row_overlap_groups"] = sum(
        _effective_multiplier(split_by_record[record_id])
        for record_id, _task, _entries in equivalent_groups
    )
    effective_counts["remaining_conflicting_groups"] = sum(
        _effective_multiplier(split_by_record[record_id])
        for record_id, _task, _entries in conflict_groups
    )
    for key in (
        "ambiguous_row_groups",
        "equivalent_non_row_overlap_groups",
        "equivalent_non_row_overlap_notes",
        "remaining_conflicting_groups",
        "remaining_conflicting_notes",
        "row_aligned_groups_detected",
        "row_aligned_groups_resolved",
        "row_aligned_notes_resolved",
        "unresolved_row_groups",
    ):
        source_counts.setdefault(key, 0)
        effective_counts.setdefault(key, 0)
    gate = (
        source_counts["remaining_conflicting_groups"] == 0
        and source_counts["remaining_conflicting_notes"] == 0
        and source_counts["unresolved_row_groups"] == 0
        and source_counts["ambiguous_row_groups"] == 0
        and source_counts["row_aligned_groups_detected"]
        == source_counts["row_aligned_groups_resolved"]
    )
    payload: dict[str, object] = {
        "acceptance": gate,
        "contract_version": LABEL_BINDING_PREFLIGHT_VERSION,
        "counts": _plain(source_counts),
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "effective_view_counts": _plain(effective_counts),
        "graph_schema_fingerprint": graph_schema_fingerprint(),
        "source_row_binding_version": SOURCE_ROW_BINDING_VERSION,
        "source_split_fingerprint": manifest.source_split_fingerprint,
        "split_aggregates": {
            split: {
                "effective": _plain(split_effective[split]),
                "source_native": _plain(split_source[split]),
            }
            for split in ("train", "validation")
        },
        "sealed_test": {
            "details_sealed": True,
            "effective": _plain(split_effective["test"]),
            "record_ids_exposed": False,
            "source_native": _plain(split_source["test"]),
            "source_values_or_classes_exposed": False,
        },
        "supersedes_blocker_evidence": {
            "compact_forensic_fingerprint": _SUPERSEDED_COMPACT_FINGERPRINT,
            "full_forensic_fingerprint": _SUPERSEDED_FORENSIC_FINGERPRINT,
            "historical_artifacts_rewritten": False,
        },
        "task_aggregates": {
            task: {
                "effective": _plain(task_effective[task]),
                "source_native": _plain(task_source[task]),
            }
            for task in ("inversion", "quality")
        },
        "test_access_policy": _TEST_ACCESS_POLICY,
        "test_targets_used_for_model_evaluation": False,
        "training_authorized": gate,
        "train_validation_bindings": tv_bindings,
        "train_validation_diagnostics": tv_diagnostics,
        "transposition_binding": _TRANSPOSITION_BINDING,
        "transposition_view_count": _EXPECTED_VIEW_COUNT,
    }
    payload["preflight_fingerprint"] = fingerprint(payload)
    return payload


def require_zero_conflicting_overlaps(report: Mapping[str, object]) -> None:
    counts = report.get("counts")
    gate_counts = counts if isinstance(counts, dict) else {}
    if (
        report.get("acceptance") is not True
        or report.get("training_authorized") is not True
        or gate_counts.get("remaining_conflicting_groups") != 0
        or gate_counts.get("remaining_conflicting_notes") != 0
        or gate_counts.get("unresolved_row_groups") != 0
        or gate_counts.get("ambiguous_row_groups") != 0
    ):
        diagnostics = report.get("train_validation_diagnostics")
        first = diagnostics[0] if isinstance(diagnostics, list) and diagnostics else None
        raise AnalysisGNNGraphError(
            "label-binding preflight did not authorize training: "
            f"counts={json.dumps(gate_counts, sort_keys=True)}, "
            f"first={json.dumps(first, sort_keys=True)}"
        )


def source_row_bindings_from_report(
    report: Mapping[str, object],
) -> dict[str, DilemmadataSourceRowBinding]:
    rows = report.get("train_validation_bindings")
    if not isinstance(rows, list):
        raise AnalysisGNNGraphError("preflight source-row bindings are missing")
    result: dict[str, DilemmadataSourceRowBinding] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"binding", "split"} or row[
            "split"
        ] not in {"train", "validation"}:
            raise AnalysisGNNGraphError("preflight source-row binding row is invalid")
        try:
            binding = source_row_binding_from_payload(row["binding"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisGNNGraphError(
                "preflight source-row binding payload is invalid"
            ) from exc
        if binding.record_id in result:
            raise AnalysisGNNGraphError("preflight repeats a source-row record binding")
        result[binding.record_id] = binding
    return result


def validate_label_binding_preflight(
    path: str | Path,
    manifest: CommonDatasetManifest,
) -> dict[str, object]:
    """Verify that a persisted preflight is exact, current, and gate-open."""

    _assert_frozen_manifest(manifest)
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fingerprint = report.pop("preflight_fingerprint", None)
    observed_fingerprint = fingerprint(report)
    report["preflight_fingerprint"] = expected_fingerprint
    counts = report.get("counts")
    sealed_test = report.get("sealed_test")
    if (
        expected_fingerprint != observed_fingerprint
        or report.get("contract_version") != LABEL_BINDING_PREFLIGHT_VERSION
        or report.get("dataset_manifest_fingerprint")
        != manifest.manifest_fingerprint
        or report.get("source_split_fingerprint")
        != manifest.source_split_fingerprint
        or report.get("graph_schema_fingerprint") != graph_schema_fingerprint()
        or report.get("source_row_binding_version") != SOURCE_ROW_BINDING_VERSION
        or not isinstance(counts, dict)
        or counts.get("record_count") != EXPECTED_RECORD_COUNT
        or counts.get("records_checked") != EXPECTED_RECORD_COUNT
        or report.get("transposition_view_count") != _EXPECTED_VIEW_COUNT
        or not isinstance(report.get("effective_view_counts"), dict)
        or not isinstance(report.get("split_aggregates"), dict)
        or not isinstance(report.get("task_aggregates"), dict)
        or not isinstance(report.get("train_validation_bindings"), list)
        or not isinstance(report.get("train_validation_diagnostics"), list)
        or not isinstance(sealed_test, dict)
        or sealed_test.get("details_sealed") is not True
        or sealed_test.get("record_ids_exposed") is not False
        or sealed_test.get("source_values_or_classes_exposed") is not False
        or report.get("test_access_policy") != _TEST_ACCESS_POLICY
        or report.get("test_targets_used_for_model_evaluation") is not False
        or report.get("transposition_binding") != _TRANSPOSITION_BINDING
    ):
        raise AnalysisGNNGraphError(
            "label-binding preflight artifact is malformed, stale, or misbound"
        )
    source_row_bindings_from_report(report)
    require_zero_conflicting_overlaps(report)
    return report


__all__ = [
    "LABEL_BINDING_PREFLIGHT_VERSION",
    "label_binding_preflight",
    "require_zero_conflicting_overlaps",
    "source_row_bindings_from_report",
    "validate_label_binding_preflight",
]
