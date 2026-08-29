from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.contracts import (
    COMMON_BENCHMARK_CONFIG,
    EXPECTED_SPLIT_FINGERPRINT,
    fingerprint,
    graph_schema_fingerprint,
)
from music_critic.experiments.analysisgnn.graph import (
    AnalysisGNNGraphError,
    bind_entry_supervision,
)
from music_critic.experiments.analysisgnn.metrics import aggregate_entry_predictions
from music_critic.experiments.analysisgnn.preflight import (
    LABEL_BINDING_PREFLIGHT_VERSION,
    validate_label_binding_preflight,
)
from music_critic.experiments.analysisgnn.training import (
    require_deterministic_cuda_environment,
    train_seed,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
    DILEMMADATA_COMMON_FAMILY_BY_TASK,
)


RECORD_ID = "dlc:corelli:op03n04a"
PIECE_ID = "piece:dilemmadata-dlc-a88f753949b33e7705b36448"
TRANSPOSITION = "m6"
POINT_ENTITY_ID = (
    "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000137"
)
INTERVAL_ENTITY_ID = (
    "span:dilemmadata-target-a88f753949b33e7705b36448-917564b62cd5-00000138"
)


def _entry(index: int, state: str, value: str | None) -> SimpleNamespace:
    entity_id = f"unavailable:{index:03d}"
    if index == 137:
        entity_id = POINT_ENTITY_ID
    elif index == 138:
        entity_id = INTERVAL_ENTITY_ID
    elif index == 139:
        entity_id = "masked-overlap:139"
    return SimpleNamespace(entity_id=entity_id, state=state, common_value=value)


def _real_overlap_fixture(
    *, interval_class: str = "major triad"
) -> tuple[tuple[SimpleNamespace, ...], SimpleNamespace, SimpleNamespace]:
    note_ids = tuple(
        f"note:dilemmadata-a88f753949b33e7705b36448-{index:08d}"
        for index in range(530, 536)
    )
    notes = tuple(
        SimpleNamespace(note_id=note_id, onset_qn=RationalTime(86))
        for note_id in note_ids
    )
    spans = (
        SimpleNamespace(
            annotation_id=POINT_ENTITY_ID,
            start_qn=RationalTime(86),
            end_qn=RationalTime(86),
        ),
        SimpleNamespace(
            annotation_id=INTERVAL_ENTITY_ID,
            start_qn=RationalTime(86),
            end_qn=RationalTime(87),
        ),
        SimpleNamespace(
            annotation_id="masked-overlap:139",
            start_qn=RationalTime(86),
            end_qn=RationalTime(87),
        ),
    )
    quality_entries = tuple(
        _entry(index, "masked", None) for index in range(137)
    ) + (
        _entry(137, "exact", "major triad"),
        _entry(138, "exact", interval_class),
        _entry(139, "masked", None),
    )
    projection = SimpleNamespace(
        targets=(
            SimpleNamespace(task_id=COMMON_QUALITY_TASK, entries=quality_entries),
            SimpleNamespace(task_id=COMMON_INVERSION_TASK, entries=()),
        )
    )
    return notes, SimpleNamespace(alignment_spans=spans), projection


def test_real_corelli_equivalent_point_interval_overlap_keeps_both_entries() -> None:
    notes, source, projection = _real_overlap_fixture()
    tensors, entries, overlaps = bind_entry_supervision(
        notes,
        source,
        projection,
        record_id=RECORD_ID,
        piece_id=PIECE_ID,
    )
    vocabulary = DILEMMADATA_COMMON_FAMILY_BY_TASK[COMMON_QUALITY_TASK].vocabulary
    assert vocabulary is not None
    major_class = vocabulary.index("major triad")
    assert TRANSPOSITION == "m6"  # Recorded failing view; binding is pitch-invariant.
    assert tensors["quality"].tolist() == [major_class] * 6
    assert tensors["quality_mask"].tolist() == [True] * 6
    expected_memberships = tuple(
        (note_index, entry_index)
        for note_index in range(6)
        for entry_index in (137, 138)
    )
    assert tuple(
        zip(*tensors["quality_membership_index"].tolist())
    ) == expected_memberships
    assert len(overlaps) == 6
    assert all(
        overlap.entry_indices == (137, 138)
        and overlap.common_classes == ("major triad", "major triad")
        and overlap.comparison == "equivalent_available_class"
        for overlap in overlaps
    )

    task_entries = tuple(entry for entry in entries if entry.task == "quality")
    logits = torch.zeros((6, 50))
    logits[:, major_class] = 5.0
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
    for entry_index in (137, 138):
        assert predictions[entry_index].mask is True
        assert predictions[entry_index].target == major_class
        assert predictions[entry_index].prediction == major_class
    assert predictions[139].mask is False


def test_real_corelli_conflicting_available_classes_fail_closed_with_diagnostics() -> None:
    notes, source, projection = _real_overlap_fixture(interval_class="minor triad")
    with pytest.raises(AnalysisGNNGraphError) as caught:
        bind_entry_supervision(
            notes,
            source,
            projection,
            record_id=RECORD_ID,
            piece_id=PIECE_ID,
        )
    diagnostic = str(caught.value)
    for expected in (
        RECORD_ID,
        PIECE_ID,
        '"task":"quality"',
        '"entry_index":137',
        '"entry_index":138',
        "major triad",
        "minor triad",
    ):
        assert expected in diagnostic


def _frozen_manifest() -> SimpleNamespace:
    records = tuple(
        SimpleNamespace(split=split)
        for split, count in (("train", 577), ("validation", 71), ("test", 71))
        for _ in range(count)
    )
    return SimpleNamespace(
        record_count=719,
        records=records,
        split_counts={"train": 577, "validation": 71, "test": 71},
        source_split_fingerprint=EXPECTED_SPLIT_FINGERPRINT,
        manifest_fingerprint="manifest-fingerprint",
    )


def test_training_preflight_artifact_is_hash_bound_and_conflict_free(
    tmp_path: Path,
) -> None:
    manifest = _frozen_manifest()
    report = {
        "acceptance": True,
        "contract_version": LABEL_BINDING_PREFLIGHT_VERSION,
        "counts": {
            "conflicting_overlap_note_count": 0,
            "record_count": 719,
            "transposition_view_count": 7_066,
        },
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "graph_schema_fingerprint": graph_schema_fingerprint(),
        "overlaps": [],
        "source_split_fingerprint": manifest.source_split_fingerprint,
        "split_counts": manifest.split_counts,
        "task_counts": {},
        "test_access_policy": (
            "structural_label_binding_only_no_model_predictions_metrics_or_selection"
        ),
        "transposition_binding": (
            "computed_once_per_accepted_record_and_reused_for_pitch_only_views"
        ),
    }
    report["preflight_fingerprint"] = fingerprint(report)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_label_binding_preflight(path, manifest)
    assert validated["preflight_fingerprint"] == report["preflight_fingerprint"]

    report["acceptance"] = False
    report["counts"]["conflicting_overlap_note_count"] = 1
    report.pop("preflight_fingerprint")
    report["preflight_fingerprint"] = fingerprint(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AnalysisGNNGraphError, match="conflicting"):
        validate_label_binding_preflight(path, manifest)


def test_cuda_requires_exact_cublas_workspace_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        require_deterministic_cuda_environment("cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        require_deterministic_cuda_environment("cuda")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require_deterministic_cuda_environment("cuda")
    require_deterministic_cuda_environment("cpu")


def test_train_validates_preflight_before_run_directory_and_optimizer() -> None:
    source = inspect.getsource(train_seed)
    validation = source.index("validate_label_binding_preflight")
    run_directory = source.index("output.mkdir")
    optimizer = source.index("configure_optimizer")
    assert validation < run_directory < optimizer


def test_reconstruction_lock_adds_music21_without_changing_graph_pins() -> None:
    root = Path(__file__).resolve().parents[2]
    lock = (root / "configs/phase9eb1/reconstruction-lock.txt").read_text()
    assert "music21==9.3.0" in lock
    assert "torch==2.2.2" in lock
    assert "torch-geometric==2.6.1" in lock
    assert (
        "graphmuse @ git+https://github.com/manoskary/graphmuse@"
        "c36eedba811a24c0addf96bdd3d1df449cf753c1"
    ) in lock
    config = json.loads(
        (root / "configs/phase9eb1/common_subset.json").read_text()
    )
    assert config["contract_version"] == "1.0.1"
    assert config["config_fingerprint"] == (
        COMMON_BENCHMARK_CONFIG.config_fingerprint
    )
    assert config["graph_schema_fingerprint"] == graph_schema_fingerprint()
