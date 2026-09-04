from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import json

import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.dataset import CommonDatasetRecord
from music_critic.experiments.analysisgnn.graph import (
    AnalysisGNNGraphError,
    bind_entry_supervision,
)
from music_critic.experiments.analysisgnn.metrics import aggregate_entry_predictions
from music_critic.experiments.analysisgnn.preflight import label_binding_preflight
from music_critic.experiments.analysisgnn.source_row_binding import (
    SourceRowBindingError,
    build_source_row_binding,
    source_row_binding_from_payload,
    source_row_binding_payload,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
    DILEMMADATA_COMMON_FAMILY_BY_TASK,
)


RECORD_ID = "dlc:corelli:op03n04a"
PIECE_ID = "piece:dilemmadata-dlc-a88f753949b33e7705b36448"
TOKEN = "a88f753949b33e7705b36448"
POINT = "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000137"
INTERVAL = "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000138"


def _record() -> CommonDatasetRecord:
    return CommonDatasetRecord(
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        dialect="dlc",
        source_group_id="dilemmadata-component:fixture",
        split="train",
        raw_projection_sha256="a" * 64,
        target_bundle_fingerprint="b" * 64,
        common_projection_fingerprint="c" * 64,
    )


def _notes(*, unknown_ordinal: bool = False) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            note_id=(
                f"note:dilemmadata-{TOKEN}-"
                f"{(99999999 if unknown_ordinal and offset == 0 else 530 + offset):08d}"
            ),
            onset_qn=RationalTime(86),
            pitch=60 + offset // 2,
            staff=(3, 3, 4, 4, 1, 1)[offset],
            voice=1,
        )
        for offset in range(6)
    )


def _source_projection(
    *,
    point_value: str = "major triad",
    interval_value: str = "major triad",
    interval_state: str = "exact",
) -> tuple[SimpleNamespace, SimpleNamespace]:
    spans = (
        SimpleNamespace(
            annotation_id=POINT,
            start_qn=RationalTime(86),
            end_qn=RationalTime(86),
        ),
        SimpleNamespace(
            annotation_id=INTERVAL,
            start_qn=RationalTime(86),
            end_qn=RationalTime(87),
        ),
    )
    entries = tuple(
        SimpleNamespace(
            entity_id=f"unavailable:{index}",
            state="masked",
            common_value=None,
        )
        for index in range(137)
    ) + (
        SimpleNamespace(entity_id=POINT, state="exact", common_value=point_value),
        SimpleNamespace(
            entity_id=INTERVAL,
            state=interval_state,
            common_value=interval_value if interval_state == "exact" else None,
        ),
    )
    source = SimpleNamespace(piece_id=PIECE_ID, alignment_spans=spans)
    projection = SimpleNamespace(
        targets=(
            SimpleNamespace(task_id=COMMON_INVERSION_TASK, entries=()),
            SimpleNamespace(task_id=COMMON_QUALITY_TASK, entries=entries),
        )
    )
    return source, projection


def _piece(notes: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
    return SimpleNamespace(piece_id=PIECE_ID, duration_qn=RationalTime(87), notes=notes)


def _write_source(root: Path) -> None:
    path = root / "pitch_arrays" / "DLC" / "corelli" / "op03n04a.tsv"
    path.parent.mkdir(parents=True)
    header = (
        "quarterbeats_playthrough\tis_note_onset\tpitch\tstaff\tvoice\t"
        "unfolded_harmony_index\n"
    )
    rows = "".join(
        f"86\tTrue\t{60 + offset // 2}\t{(3, 3, 4, 4, 1, 1)[offset]}\t1\t"
        f"{133 if offset % 2 == 0 else 134}\n"
        for offset in range(6)
    )
    prefix = "0\tFalse\t0\t\t\t\n" * 530
    path.write_text(header + prefix + rows, encoding="utf-8")


def _built_binding(tmp_path: Path, *, notes: tuple[SimpleNamespace, ...] | None = None):
    _write_source(tmp_path)
    notes = _notes() if notes is None else notes
    source, projection = _source_projection()
    return build_source_row_binding(
        _record(),
        _piece(notes),
        source,
        projection,
        tmp_path,
        notes=notes,
    )


def test_real_corelli_rows_bind_three_plus_three_without_class_selection(
    tmp_path: Path,
) -> None:
    notes = _notes()
    binding = _built_binding(tmp_path, notes=notes)
    assert binding is not None
    group = binding.groups[0]
    assert [entry.entry_index for entry in group.entries] == [137, 138]
    assert [entry.source_identity for entry in group.entries] == ["133", "134"]
    assert [
        sum(row.entry_index == entry_index for row in group.assignments)
        for entry_index in (137, 138)
    ] == [3, 3]
    assert [row.source_row_ordinal for row in group.assignments] == list(
        range(530, 536)
    )

    source, projection = _source_projection(interval_value="minor triad")
    tensors, entries, overlaps = bind_entry_supervision(
        notes,
        source,
        projection,
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        dialect="dlc",
        source_row_binding=binding,
    )
    vocabulary = DILEMMADATA_COMMON_FAMILY_BY_TASK[COMMON_QUALITY_TASK].vocabulary
    assert vocabulary is not None
    major = vocabulary.index("major triad")
    minor = vocabulary.index("minor triad")
    assert tensors["quality"].tolist() == [major, minor, major, minor, major, minor]
    assert tuple(zip(*tensors["quality_membership_index"].tolist())) == tuple(
        (index, 137 if index % 2 == 0 else 138) for index in range(6)
    )
    assert overlaps == ()

    task_entries = tuple(entry for entry in entries if entry.task == "quality")
    logits = torch.zeros((6, 50))
    logits[::2, major] = 5
    logits[1::2, minor] = 5
    predictions = aggregate_entry_predictions(
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        split="train",
        task="quality",
        note_logits=logits,
        note_targets=tensors["quality"],
        note_membership_index=tensors["quality_membership_index"],
        entity_ids=tuple(entry.entity_id for entry in task_entries),
        entry_masks=tuple(entry.mask for entry in task_entries),
    )
    assert predictions[137].mask and predictions[137].prediction == major
    assert predictions[138].mask and predictions[138].prediction == minor


def test_binding_is_class_independent_and_transposition_invariant(tmp_path: Path) -> None:
    binding = _built_binding(tmp_path)
    assert binding is not None
    first = source_row_binding_payload(binding)
    source, changed_projection = _source_projection(
        point_value="diminished triad", interval_value="major triad"
    )
    rebuilt = build_source_row_binding(
        _record(),
        _piece(_notes()),
        source,
        changed_projection,
        tmp_path,
        notes=_notes(),
    )
    assert rebuilt is not None
    assert rebuilt.semantic_fingerprint == binding.semantic_fingerprint
    assert source_row_binding_payload(rebuilt) == first
    # Pitch-only graph views reuse canonical note indices and this exact binding.
    assert tuple(
        (row.note_index, row.entry_index) for row in rebuilt.groups[0].assignments
    ) == tuple(
        (row.note_index, row.entry_index) for row in binding.groups[0].assignments
    )


def test_unknown_source_ordinal_and_missing_binding_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(SourceRowBindingError, match="out_of_bounds"):
        _built_binding(tmp_path, notes=_notes(unknown_ordinal=True))
    source, projection = _source_projection()
    with pytest.raises(AnalysisGNNGraphError, match="source-row provenance"):
        bind_entry_supervision(
            _notes(),
            source,
            projection,
            record_id=RECORD_ID,
            piece_id=PIECE_ID,
            dialect="dlc",
        )


def test_malformed_duplicate_provenance_and_incomplete_entry_mapping_fail_closed(
    tmp_path: Path,
) -> None:
    binding = _built_binding(tmp_path)
    assert binding is not None
    payload = source_row_binding_payload(binding)
    payload["groups"][0]["entries"][1]["source_identity"] = "133"
    payload_without_fingerprint = dict(payload)
    payload_without_fingerprint.pop("semantic_fingerprint")
    payload["semantic_fingerprint"] = fingerprint(payload_without_fingerprint)
    with pytest.raises(ValueError, match="unique source identities"):
        source_row_binding_from_payload(payload)

    double_bound = source_row_binding_payload(binding)
    duplicate = dict(double_bound["groups"][0]["assignments"][0])
    duplicate["entry_index"] = 138
    duplicate["entity_id"] = INTERVAL
    duplicate["source_identity"] = "134"
    double_bound["groups"][0]["assignments"].append(duplicate)
    double_bound["groups"][0]["assignments"].sort(
        key=lambda row: (row["note_index"], row["entry_index"])
    )
    unsigned = dict(double_bound)
    unsigned.pop("semantic_fingerprint")
    double_bound["semantic_fingerprint"] = fingerprint(unsigned)
    with pytest.raises(ValueError, match="assign a note to two entries"):
        source_row_binding_from_payload(double_bound)

    incomplete = source_row_binding_payload(binding)
    incomplete["groups"][0]["assignments"].pop()
    unsigned = dict(incomplete)
    unsigned.pop("semantic_fingerprint")
    incomplete["semantic_fingerprint"] = fingerprint(unsigned)
    malformed = source_row_binding_from_payload(incomplete)
    source, projection = _source_projection()
    with pytest.raises(AnalysisGNNGraphError, match="incomplete"):
        bind_entry_supervision(
            _notes(),
            source,
            projection,
            record_id=RECORD_ID,
            piece_id=PIECE_ID,
            dialect="dlc",
            source_row_binding=malformed,
        )


def test_masked_transition_row_does_not_replace_available_supervision(
    tmp_path: Path,
) -> None:
    binding = _built_binding(tmp_path)
    source, projection = _source_projection(interval_state="masked")
    tensors, _entries, overlaps = bind_entry_supervision(
        _notes(),
        source,
        projection,
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
        dialect="dlc",
        source_row_binding=binding,
    )
    assert tensors["quality_mask"].tolist() == [True, False, True, False, True, False]
    assert tuple(zip(*tensors["quality_membership_index"].tolist())) == (
        (0, 137),
        (2, 137),
        (4, 137),
    )
    assert overlaps == ()


def test_non_dlc_point_interval_conflict_keeps_generic_fail_closed_rule() -> None:
    source, projection = _source_projection(interval_value="minor triad")
    with pytest.raises(AnalysisGNNGraphError, match="conflicting available"):
        bind_entry_supervision(
            _notes(),
            source,
            projection,
            record_id="an:fixture",
            piece_id=PIECE_ID,
            dialect="an_joint",
        )


def test_non_row_equivalent_overlap_keeps_sparse_multi_membership() -> None:
    source, projection = _source_projection()
    tensors, _entries, overlaps = bind_entry_supervision(
        _notes(),
        source,
        projection,
        record_id="an:fixture",
        piece_id=PIECE_ID,
        dialect="an_joint",
    )
    assert tuple(zip(*tensors["quality_membership_index"].tolist())) == tuple(
        (note_index, entry_index)
        for note_index in range(6)
        for entry_index in (137, 138)
    )
    assert len(overlaps) == 6
    assert all(row.comparison == "equivalent_available_class" for row in overlaps)


def test_committed_real_train_cases_are_exact_disjoint_and_fingerprint_bound() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/analysisgnn/phase9eb1_source_row_remediation.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))
    semantic_fingerprint = evidence.pop("semantic_fingerprint")
    assert semantic_fingerprint == fingerprint(evidence)
    assert semantic_fingerprint == (
        "13054693d85ec859026dc92452bf8c853d6bedb705e2256964544db2b01078e2"
    )
    cases = {row["record_id"]: row for row in evidence["cases"]}
    assert set(cases) == {
        "dlc:corelli:op01n07c",
        "dlc:corelli:op03n03b",
        "dlc:corelli:op03n03d",
        "dlc:corelli:op03n04a",
    }
    for row in cases.values():
        pairs = tuple(tuple(pair) for pair in row["membership_pairs"])
        assert pairs == tuple(sorted(set(pairs)))
        assert len({note_index for note_index, _entry_index in pairs}) == len(pairs)
        assert sum(count for _entry, count in row["entry_note_counts"]) == len(pairs)
        assert all(count > 0 for _entry, count in row["entry_note_counts"])
    corelli = cases["dlc:corelli:op03n04a"]
    assert corelli["entry_note_counts"] == [[137, 3], [138, 3]]
    assert corelli["common_classes"] == ["major triad", "major triad"]
    assert len(corelli["source_row_ordinals"]) == 6
    assert evidence["preflight_counts"]["remaining_conflicting_groups"] == 0
    assert evidence["training_authorized"] is True
    assert evidence["sealed_test"]["test_targets_used_for_model_evaluation"] is False


def test_available_entry_without_memberships_has_explicit_masked_prediction() -> None:
    rows = aggregate_entry_predictions(
        record_id="record",
        piece_id="piece",
        split="validation",
        task="inversion",
        note_logits=torch.zeros((1, 4)),
        note_targets=torch.tensor([-1]),
        note_membership_index=torch.empty((2, 0), dtype=torch.long),
        entity_ids=("entry",),
        entry_masks=(True,),
    )
    assert len(rows) == 1
    assert rows[0].mask is False
    assert rows[0].target == -1 and rows[0].prediction == -1
    assert all(value != value for value in rows[0].logits)


def test_preflight_surface_constructs_no_graph_model_optimizer_or_metrics() -> None:
    source = inspect.getsource(label_binding_preflight)
    module_source = inspect.getsource(inspect.getmodule(label_binding_preflight))
    for forbidden in (
        "build_analysisgnn_graph",
        "AnalysisGNNCommonModel",
        "configure_optimizer",
        "aggregate_entry_predictions",
        "benchmark_metrics",
    ):
        assert forbidden not in source
        assert forbidden not in module_source
