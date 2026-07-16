from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest

import music_critic.data as data
from music_critic.data import (
    SCHEMA_VERSION,
    AnnotationSpan,
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
    TargetArray,
    TempoEvent,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "docs" / "DATA_CONTRACT.md"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "data" / "canonical_piece_v2.json"

SCHEMA_CLASSES = (
    PieceMetadata,
    CanonicalTrack,
    CanonicalNote,
    CanonicalBar,
    CanonicalBeat,
    TempoEvent,
    MeterEvent,
    KeySignatureEvent,
    AnnotationSpan,
    TargetArray,
    ProvenanceRecord,
    QualityFlag,
    CanonicalPiece,
)

EXPECTED_FIELDS = {
    PieceMetadata: (
        "source_format",
        "title",
        "creators",
        "collection",
        "movement_title",
        "movement_number",
        "genres",
        "copyright",
        "language",
    ),
    CanonicalTrack: (
        "track_id",
        "source_track_index",
        "name",
        "instrument_name",
        "program",
        "channel",
        "is_percussion",
        "provenance_id",
    ),
    CanonicalNote: (
        "note_id",
        "track_id",
        "pitch",
        "onset_qn",
        "duration_qn",
        "velocity",
        "channel",
        "program",
        "is_percussion",
        "is_grace",
        "spelling_step",
        "spelling_alter",
        "staff",
        "voice",
        "articulations",
        "dynamic",
        "source_onset_ticks",
        "source_duration_ticks",
        "source_onset_seconds",
        "source_duration_seconds",
        "provenance_id",
    ),
    CanonicalBar: (
        "bar_id",
        "index",
        "start_qn",
        "duration_qn",
        "meter_event_id",
        "metric_offset_qn",
        "is_pickup",
        "is_incomplete",
        "display_number",
        "provenance_id",
    ),
    CanonicalBeat: (
        "beat_id",
        "bar_id",
        "meter_event_id",
        "index_in_bar",
        "start_qn",
        "duration_qn",
        "position_in_bar_qn",
        "is_downbeat",
        "strength",
        "provenance_id",
    ),
    TempoEvent: (
        "tempo_event_id",
        "onset_qn",
        "microseconds_per_quarter",
        "provenance_id",
    ),
    MeterEvent: (
        "meter_event_id",
        "onset_qn",
        "numerator",
        "denominator",
        "provenance_id",
    ),
    KeySignatureEvent: (
        "key_signature_event_id",
        "onset_qn",
        "fifths",
        "mode",
        "raw_value",
        "provenance_id",
    ),
    AnnotationSpan: (
        "annotation_id",
        "annotation_type",
        "layer",
        "start_qn",
        "end_qn",
        "track_id",
        "value",
        "provenance_id",
    ),
    TargetArray: (
        "target_id",
        "task",
        "annotation_view_id",
        "alignment_type",
        "entity_ids",
        "value_type",
        "class_labels",
        "values",
        "mask",
        "confidence",
        "source",
        "provenance",
    ),
    ProvenanceRecord: (
        "provenance_id",
        "kind",
        "source",
        "record_id",
        "uri",
        "version",
        "checksum_sha256",
        "created_at",
        "parents",
        "details",
    ),
    QualityFlag: (
        "code",
        "severity",
        "message",
        "entity_ids",
        "provenance_id",
    ),
    CanonicalPiece: (
        "schema_version",
        "piece_id",
        "dataset_name",
        "source_group_id",
        "split",
        "source_path",
        "source_resolution",
        "duration_qn",
        "metadata",
        "tracks",
        "notes",
        "bars",
        "beats",
        "tempo_events",
        "meter_events",
        "key_signature_events",
        "annotations",
        "targets",
        "provenance",
        "quality_flags",
    ),
}


def _contract_fixture() -> object:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    section = contract.index("## 11. Complete canonical JSON example")
    start = contract.index("```json\n", section) + len("```json\n")
    end = contract.index("\n```", start)
    return json.loads(contract[start:end])


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_schema_version_is_exact() -> None:
    assert SCHEMA_VERSION == "2.0.0"


def test_all_schema_records_are_frozen_slotted_dataclasses() -> None:
    for schema_class in SCHEMA_CLASSES:
        assert dataclasses.is_dataclass(schema_class)
        assert schema_class.__dataclass_params__.frozen
        assert "__slots__" in schema_class.__dict__

    metadata = PieceMetadata("synthetic", None, None, None, None, None, None, None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.title = "changed"  # type: ignore[misc]
    assert not hasattr(metadata, "__dict__")


def test_dataclass_field_names_and_order_match_contract() -> None:
    for schema_class, expected in EXPECTED_FIELDS.items():
        assert tuple(field.name for field in dataclasses.fields(schema_class)) == expected


def test_dataclass_type_annotations_match_contract() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    for schema_class in SCHEMA_CLASSES:
        marker = f"class {schema_class.__name__}:"
        marker_index = contract.index(marker)
        block_start = contract.rfind("```python\n", 0, marker_index) + len("```python\n")
        block_end = contract.index("\n```", marker_index)
        module = ast.parse(contract[block_start:block_end])
        class_node = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == schema_class.__name__
        )
        documented = {
            statement.target.id: ast.unparse(statement.annotation)
            for statement in class_node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        assert schema_class.__annotations__ == documented


def test_public_schema_aliases_match_contract() -> None:
    assert data.JsonScalar == str | int | float | bool | None
    assert data.ProvenanceDetail == tuple[str, data.JsonScalar]
    assert data.Split == str | None
    assert get_args(data.SourceFormat) == (
        "midi",
        "musicxml",
        "json",
        "jsonl",
        "tsv",
        "synthetic",
        "other",
    )
    assert get_args(data.KeySignatureMode) == (
        "major",
        "minor",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "locrian",
        "other",
        "unknown",
    )
    assert get_args(data.AnnotationLayer) == ("observation", "target_alignment")
    assert get_args(data.AlignmentType) == (
        "piece",
        "track",
        "note",
        "bar",
        "beat",
        "bar_boundary",
        "beat_boundary",
        "annotation_span",
    )
    assert get_args(data.TargetValueType) == (
        "categorical",
        "scalar",
        "multi_label",
        "distribution",
    )
    assert get_args(data.TargetSource) == (
        "human",
        "dataset",
        "algorithm",
        "pseudo_label",
        "derived",
        "synthetic",
    )
    assert data.TargetValue == (
        str | int | float | tuple[str, ...] | tuple[float, ...]
    )
    assert get_args(data.ProvenanceKind) == (
        "source",
        "conversion",
        "annotation",
        "derivation",
        "default",
        "synthetic",
    )
    assert get_args(data.IssueSeverity) == ("error", "warning")
    assert get_args(data.QualitySeverity) == ("info", "warning")
    assert data.QualityFlagCode is str


def test_collection_fields_accept_tuples_and_preserve_optional_distinctions() -> None:
    metadata = PieceMetadata(
        source_format="synthetic",
        title=None,
        creators=(),
        collection="",
        movement_title=None,
        movement_number=None,
        genres=(),
        copyright=None,
        language=None,
    )
    piece = CanonicalPiece(
        schema_version="invalid-on-purpose",
        piece_id="invalid",
        dataset_name="",
        source_group_id="",
        split=None,
        source_path=None,
        source_resolution=None,
        duration_qn=RationalTime(0),
        metadata=metadata,
        tracks=(),
        notes=(),
        bars=(),
        beats=(),
        tempo_events=(),
        meter_events=(),
        key_signature_events=(),
        annotations=(),
        targets=(),
        provenance=(),
        quality_flags=(),
    )
    assert metadata.title is None
    assert metadata.collection == ""
    assert metadata.creators == ()
    assert piece.split is None
    for field_name in (
        "tracks",
        "notes",
        "bars",
        "beats",
        "tempo_events",
        "meter_events",
        "key_signature_events",
        "annotations",
        "targets",
        "provenance",
        "quality_flags",
    ):
        assert isinstance(getattr(piece, field_name), tuple)


def test_invalid_programmatic_records_are_constructible_for_future_validation() -> None:
    track = CanonicalTrack(
        track_id="not-prefixed",
        source_track_index=-1,
        name=None,
        instrument_name=None,
        program=999,
        channel=99,
        is_percussion=False,
        provenance_id="missing",
    )
    target = TargetArray(
        target_id="bad",
        task="",
        annotation_view_id=" ",
        alignment_type="track",
        entity_ids=("missing",),
        value_type="categorical",
        class_labels=None,
        values=(None,),
        mask=(True,),
        confidence=(2.0,),
        source=(None,),
        provenance=(None,),
    )
    assert track.program == 999
    assert target.mask == (True,)


def test_raw_record_fields_exclude_theory_and_semantic_roles() -> None:
    note_fields = set(EXPECTED_FIELDS[CanonicalNote])
    track_fields = set(EXPECTED_FIELDS[CanonicalTrack])
    metadata_fields = set(EXPECTED_FIELDS[PieceMetadata])
    forbidden = {
        "scale_degree",
        "harmony",
        "chord",
        "chord_quality",
        "roman_numeral",
        "local_key",
        "non_chord_tone",
        "semantic_role",
        "role",
        "cadence",
        "phrase",
        "section_function",
    }
    assert not (note_fields & forbidden)
    assert not (track_fields & forbidden)
    assert not (metadata_fields & forbidden)
    assert {"target_id", "annotation_view_id"} <= set(EXPECTED_FIELDS[TargetArray])


def test_normative_fixture_parses_and_matches_documentation() -> None:
    fixture = _fixture()
    assert fixture == _contract_fixture()


def test_normative_fixture_target_views_masks_and_order() -> None:
    fixture = _fixture()
    targets = fixture["targets"]
    assert isinstance(targets, list)
    assert len(targets) == 3
    assert [target["task"] for target in targets] == [
        "theory.chord_quality",
        "theory.chord_quality",
        "track.role",
    ]
    assert [target["annotation_view_id"] for target in targets[:2]] == [
        None,
        "analysis.alternative",
    ]
    assert len({target["target_id"] for target in targets}) == 3
    assert targets == sorted(
        targets,
        key=lambda target: (
            target["task"],
            target["annotation_view_id"] is not None,
            target["annotation_view_id"] or "",
            target["target_id"],
        ),
    )

    default_view = targets[0]
    assert any(
        available and confidence is None
        for available, confidence in zip(
            default_view["mask"], default_view["confidence"], strict=True
        )
    )
    assert any(
        not available
        and value is None
        and confidence is None
        and source is None
        and provenance is None
        for available, value, confidence, source, provenance in zip(
            default_view["mask"],
            default_view["values"],
            default_view["confidence"],
            default_view["source"],
            default_view["provenance"],
            strict=True,
        )
    )


def test_normative_fixture_has_no_theory_keys_in_raw_records() -> None:
    fixture = _fixture()
    forbidden = {
        "scale_degree",
        "harmony",
        "chord",
        "chord_quality",
        "roman_numeral",
        "local_key",
        "non_chord_tone",
        "semantic_role",
        "role",
        "cadence",
        "phrase",
        "section_function",
    }
    raw_records = [fixture["metadata"], *fixture["tracks"], *fixture["notes"]]
    for record in raw_records:
        assert not (set(record) & forbidden)


def test_public_api_is_explicit_and_excludes_unfinished_modules() -> None:
    expected = {
        "SCHEMA_VERSION",
        "AlignmentType",
        "AnnotationLayer",
        "AnnotationSpan",
        "CanonicalBar",
        "CanonicalBeat",
        "CanonicalNote",
        "CanonicalPiece",
        "CanonicalTrack",
        "CanonicalValidationError",
        "IssueSeverity",
        "JsonObject",
        "JsonScalar",
        "KeySignatureEvent",
        "KeySignatureMode",
        "MeterEvent",
        "PieceMetadata",
        "ProvenanceDetail",
        "ProvenanceKind",
        "ProvenanceRecord",
        "QualityFlag",
        "QualityFlagCode",
        "QualitySeverity",
        "RationalTime",
        "SourceFormat",
        "Split",
        "TargetArray",
        "TargetSource",
        "TargetValue",
        "TargetValueType",
        "TempoEvent",
        "ValidationCode",
        "ValidationIssue",
        "ValidationReport",
        "dump_piece",
        "dumps_piece",
        "load_piece",
        "loads_piece",
        "piece_from_dict",
        "piece_to_dict",
        "validate_or_raise",
        "validate_piece",
    }
    assert set(data.__all__) == expected


def test_importing_data_package_is_standard_library_only(tmp_path: Path) -> None:
    code = """
import json
import sys
import music_critic.data

forbidden = (
    "torch",
    "torch_geometric",
    "numpy",
    "mido",
    "pretty_midi",
    "hydra",
    "src",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "[]"
