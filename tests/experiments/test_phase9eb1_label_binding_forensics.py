from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.contracts import (
    TRANSPOSITIONS,
    fingerprint,
)
from music_critic.experiments.analysisgnn.dataset import CommonDatasetRecord
from music_critic.experiments.analysisgnn.forensics import (
    _group_overlaps,
    _record_conflicts,
    classify_conflict_group,
    compact_forensic_evidence,
    label_binding_forensic_audit,
    source_conflict_identity,
)
from music_critic.experiments.analysisgnn.graph import LabelOverlap
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
)


SOURCE_QUALITY_TASK = "dilemmadata.dlc.chord.quality"
RECORD_ID = "dlc:corelli:op03n04a"
PIECE_ID = "piece:dilemmadata-dlc-a88f753949b33e7705b36448"
POINT_ENTITY_ID = (
    "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000137"
)
INTERVAL_ENTITY_ID = (
    "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000138"
)


def _record(split: str = "train") -> CommonDatasetRecord:
    return CommonDatasetRecord(
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        dialect="dlc",
        source_group_id="dilemmadata-component:fixture",
        split=split,
        raw_projection_sha256="a" * 64,
        target_bundle_fingerprint="b" * 64,
        common_projection_fingerprint="c" * 64,
    )


def _identity_entries() -> list[dict[str, object]]:
    return [
        {
            "entity_id": POINT_ENTITY_ID,
            "source_task_ids": [SOURCE_QUALITY_TASK],
            "source_values": ["M"],
            "span_start": {"den": 1, "num": 86},
            "span_end": {"den": 1, "num": 86},
        },
        {
            "entity_id": INTERVAL_ENTITY_ID,
            "source_task_ids": [SOURCE_QUALITY_TASK],
            "source_values": ["m"],
            "span_start": {"den": 1, "num": 86},
            "span_end": {"den": 1, "num": 87},
        },
    ]


def test_source_identity_deduplicates_transpositions_and_entry_order() -> None:
    forward = source_conflict_identity(
        record=_record(), task="quality", entries=_identity_entries()
    )
    reverse = source_conflict_identity(
        record=_record(), task="quality", entries=list(reversed(_identity_entries()))
    )
    assert forward == reverse
    groups = {
        fingerprint(
            source_conflict_identity(
                record=_record(), task="quality", entries=_identity_entries()
            )
        )
        for _transposition in TRANSPOSITIONS
    }
    assert len(groups) == 1


def _overlap(
    *, comparison: str, entries: tuple[int, int], values: tuple[str, str]
) -> LabelOverlap:
    return LabelOverlap(
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        task="quality",
        note_index=0,
        note_id="note:fixture-00000000",
        entry_indices=entries,
        entity_ids=(POINT_ENTITY_ID, INTERVAL_ENTITY_ID),
        class_ids=(30, 30 if values[0] == values[1] else 32),
        common_classes=values,
        comparison=comparison,
    )


def test_corelli_137_138_equivalent_is_excluded_and_conflict_is_included() -> None:
    equivalent = _overlap(
        comparison="equivalent_available_class",
        entries=(137, 138),
        values=("major triad", "major triad"),
    )
    conflict = _overlap(
        comparison="conflicting_available_class",
        entries=(139, 140),
        values=("major triad", "minor triad"),
    )
    groups, equivalents = _group_overlaps((equivalent, conflict))
    assert ("quality", (137, 138)) in equivalents
    assert ("quality", (137, 138)) not in groups
    assert list(groups) == [("quality", (139, 140))]


def test_point_interval_classification_is_deterministic() -> None:
    entries = [
        {"point_span": False, "span_start": {"den": 1, "num": 86}},
        {"point_span": True, "span_start": {"den": 1, "num": 86}},
    ]
    notes = [{"is_grace": False, "zero_duration": False}] * 6
    expected = {
        "note_type": "ordinary_note_at_point_boundary",
        "official_source_row_membership": "complete",
        "span_type": "point_vs_interval_same_start",
        "timestamp_type": "duplicate_timestamp_transition",
    }
    assert (
        classify_conflict_group(
            entries, notes, source_row_membership_complete=True
        )
        == expected
    )
    assert (
        classify_conflict_group(
            list(reversed(entries)), notes, source_row_membership_complete=True
        )
        == expected
    )


def _note(index: int, onset: int) -> SimpleNamespace:
    return SimpleNamespace(
        note_id=f"note:dilemmadata-fixture-{index:08d}",
        track_id="track:fixture",
        pitch=60 + index,
        onset_qn=RationalTime(onset),
        duration_qn=RationalTime(1),
        is_grace=False,
        staff=1,
        voice=1,
    )


def _common_entry(entity_id: str, value: str) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id=entity_id,
        source_task_ids=(SOURCE_QUALITY_TASK,),
        source_values=("M" if value == "major triad" else "m",),
        state="exact",
        common_value=value,
    )


def test_train_conflict_has_details_while_equivalent_group_stays_out() -> None:
    entities = ("eq-point", "eq-interval", "conflict-point", "conflict-interval")
    spans = (
        SimpleNamespace(
            annotation_id=entities[0],
            start_qn=RationalTime(86),
            end_qn=RationalTime(86),
        ),
        SimpleNamespace(
            annotation_id=entities[1],
            start_qn=RationalTime(86),
            end_qn=RationalTime(87),
        ),
        SimpleNamespace(
            annotation_id=entities[2],
            start_qn=RationalTime(90),
            end_qn=RationalTime(90),
        ),
        SimpleNamespace(
            annotation_id=entities[3],
            start_qn=RationalTime(90),
            end_qn=RationalTime(91),
        ),
    )
    source_quality = SimpleNamespace(
        task_id=SOURCE_QUALITY_TASK,
        entity_ids=entities,
        values=("M", "M", "M", "m"),
        availability_mask=(True, True, True, True),
    )
    source = SimpleNamespace(alignment_spans=spans, targets=(source_quality,))
    projection = SimpleNamespace(
        targets=(
            SimpleNamespace(
                task_id=COMMON_INVERSION_TASK,
                entries=(),
            ),
            SimpleNamespace(
                task_id=COMMON_QUALITY_TASK,
                entries=(
                    _common_entry(entities[0], "major triad"),
                    _common_entry(entities[1], "major triad"),
                    _common_entry(entities[2], "major triad"),
                    _common_entry(entities[3], "minor triad"),
                ),
            ),
        )
    )
    piece = SimpleNamespace(notes=(_note(0, 86), _note(1, 90)))
    groups, equivalents = _record_conflicts(
        _record(), piece, source, projection, corpus_root=None
    )
    assert equivalents == {("quality", (0, 1))}
    assert len(groups) == 1
    group = groups[0]
    assert group["source_conflicting_note_count"] == 1
    assert group["effective_view_count"] == 12
    assert group["effective_conflicting_note_count"] == 12
    assert len(group["entries"]) == 2
    assert len(group["conflicting_notes"]) == 1
    assert group["classification"]["span_type"] == (
        "point_vs_interval_same_start"
    )


def _compact_input() -> dict[str, object]:
    group = {
        "classification": {
            "note_type": "ordinary_note_at_point_boundary",
            "official_source_row_membership": "complete",
            "span_type": "point_vs_interval_same_start",
            "timestamp_type": "duplicate_timestamp_transition",
        },
        "conflicting_notes": [{"source_value_that_must_not_escape": "minor"}],
        "effective_conflicting_note_count": 12,
        "effective_view_count": 12,
        "entries": [
            {
                "common_class_index": 30,
                "common_value": "major triad",
                "entity_id": "entry-a",
                "entry_index": 1,
                "mapping_state": "exact",
                "point_span": True,
                "source_task_ids": [SOURCE_QUALITY_TASK],
                "source_values": ["M"],
                "span_end": {"den": 1, "num": 86},
                "span_start": {"den": 1, "num": 86},
            }
        ],
        "piece_id": PIECE_ID,
        "record_id": RECORD_ID,
        "source_conflict_group_id": "label-binding-conflict:fixture",
        "source_conflicting_note_count": 1,
        "split": "train",
        "task": "quality",
        "transpositions": list(TRANSPOSITIONS),
    }
    report: dict[str, object] = {
        "acceptance": False,
        "aggregate_counts": {"source_group_count": 1},
        "classification_counts": {},
        "count_interpretation": {},
        "deduplication_contract": {},
        "input_bindings": {},
        "official_source_evidence": {},
        "policy_comparison": [],
        "recommendation": {},
        "sealed_test": {
            "details_sealed": True,
            "source_group_count": 6,
            "source_values_exposed": False,
        },
        "semantic_fingerprint": "d" * 64,
        "split_counts": {},
        "task_counts": {},
        "test_targets_used_for_model_evaluation": False,
        "training_authorized": False,
        "train_validation_counts": {},
        "train_validation_conflicts": [group],
    }
    return report


def test_compact_evidence_keeps_train_details_and_test_sealed() -> None:
    compact = compact_forensic_evidence(_compact_input())
    assert compact["train_validation_conflicts"][0]["record_id"] == RECORD_ID
    assert "conflicting_notes" not in compact["train_validation_conflicts"][0]
    assert compact["sealed_test"] == {
        "details_sealed": True,
        "source_group_count": 6,
        "source_values_exposed": False,
    }
    serialized = str(compact)
    assert "source_value_that_must_not_escape" not in serialized


def test_compact_fingerprint_is_deterministic_and_gates_stay_closed() -> None:
    first = compact_forensic_evidence(_compact_input())
    second = compact_forensic_evidence(_compact_input())
    assert first == second
    semantic_fingerprint = first.pop("semantic_fingerprint")
    assert semantic_fingerprint == fingerprint(first)
    assert first["acceptance"] is False
    assert first["training_authorized"] is False
    assert first["test_targets_used_for_model_evaluation"] is False


def test_audit_surface_constructs_no_model_optimizer_predictions_or_metrics() -> None:
    source = inspect.getsource(label_binding_forensic_audit)
    module_source = inspect.getsource(
        inspect.getmodule(label_binding_forensic_audit)
    )
    for forbidden in (
        "AnalysisGNNCommonModel",
        "configure_optimizer",
        "aggregate_entry_predictions",
        "summarize_metrics",
    ):
        assert forbidden not in source
        assert forbidden not in module_source


def test_committed_evidence_is_bound_and_contains_no_test_details() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads(
        (
            root
            / "tests/fixtures/analysisgnn/phase9eb1_label_binding_forensics.json"
        ).read_text(encoding="utf-8")
    )
    semantic_fingerprint = evidence.pop("semantic_fingerprint")
    assert semantic_fingerprint == fingerprint(evidence)
    assert semantic_fingerprint == (
        "4dc5db61525be807a49677893b6f5b338eb35b0f59854ed508cf0d38f5f012ba"
    )
    assert evidence["aggregate_counts"]["source_group_count"] == 18
    assert evidence["aggregate_counts"]["source_conflicting_note_count"] == 128
    assert len(evidence["train_validation_conflicts"]) == 12
    assert {
        group["split"] for group in evidence["train_validation_conflicts"]
    } == {"train"}
    assert evidence["sealed_test"]["source_group_count"] == 6
    assert evidence["sealed_test"]["source_conflicting_note_count"] == 48
    assert evidence["sealed_test"]["details_sealed"] is True
    assert evidence["sealed_test"]["source_values_exposed"] is False
    assert evidence["test_targets_used_for_model_evaluation"] is False
    assert evidence["training_authorized"] is False
