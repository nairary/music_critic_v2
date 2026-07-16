from __future__ import annotations

from collections import UserDict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

import music_critic.data as data_api
from music_critic.data import (
    CanonicalPiece,
    CanonicalValidationError,
    JsonObject,
    RationalTime,
    dump_piece,
    dumps_piece,
    load_piece,
    loads_piece,
    piece_from_dict,
    piece_to_dict,
    validate_piece,
)
from music_critic.data import serialization


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "data" / "canonical_piece_v2.json"
)


@pytest.fixture
def fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def fixture_dict(fixture_text: str) -> JsonObject:
    return json.loads(fixture_text)


@pytest.fixture
def fixture_piece(fixture_dict: JsonObject) -> CanonicalPiece:
    return piece_from_dict(fixture_dict)


def _assert_issue(
    error: CanonicalValidationError, code: str, path: str
) -> None:
    assert any(
        issue.code == code and issue.path == path for issue in error.report.issues
    ), error.report.issues


def _decode_error(value: object) -> CanonicalValidationError:
    with pytest.raises(CanonicalValidationError) as caught:
        piece_from_dict(value)  # type: ignore[arg-type]
    return caught.value


def test_serialization_all_is_exact() -> None:
    assert serialization.__all__ == [
        "JsonObject",
        "piece_to_dict",
        "piece_from_dict",
        "dumps_piece",
        "loads_piece",
        "dump_piece",
        "load_piece",
    ]


def test_data_package_exports_serialization_api() -> None:
    for name in serialization.__all__:
        assert getattr(data_api, name) is getattr(serialization, name)


def test_later_phase_apis_remain_absent() -> None:
    for name in ("MidiAdapter", "build_graph", "CanonicalDataset", "CriticModel"):
        assert not hasattr(data_api, name)


def test_json_object_alias_is_public() -> None:
    assert JsonObject == dict[str, Any]


def test_normative_fixture_decodes(fixture_dict: JsonObject) -> None:
    assert isinstance(piece_from_dict(fixture_dict), CanonicalPiece)


def test_warnings_do_not_prevent_loading(fixture_piece: CanonicalPiece) -> None:
    report = validate_piece(fixture_piece)
    assert report.is_valid
    assert report.warnings


def test_normative_fixture_mapping_round_trip(
    fixture_dict: JsonObject, fixture_piece: CanonicalPiece
) -> None:
    assert piece_to_dict(fixture_piece) == fixture_dict


def test_fixture_text_and_utf8_buffer_inputs(
    fixture_text: str, fixture_piece: CanonicalPiece
) -> None:
    payload = fixture_text.encode("utf-8")
    assert loads_piece(fixture_text) == fixture_piece
    assert loads_piece(payload) == fixture_piece
    assert loads_piece(bytearray(payload)) == fixture_piece


def test_json_round_trip_returns_equal_piece(fixture_piece: CanonicalPiece) -> None:
    assert loads_piece(dumps_piece(fixture_piece)) == fixture_piece


def test_decoded_collections_are_immutable_tuples(
    fixture_piece: CanonicalPiece,
) -> None:
    assert isinstance(fixture_piece.tracks, tuple)
    assert isinstance(fixture_piece.notes, tuple)
    assert isinstance(fixture_piece.metadata.creators, tuple)
    assert isinstance(fixture_piece.metadata.genres, tuple)
    assert isinstance(fixture_piece.notes[0].articulations, tuple)
    assert isinstance(fixture_piece.targets[0].entity_ids, tuple)
    assert isinstance(fixture_piece.targets[0].values, tuple)
    assert isinstance(fixture_piece.targets[0].mask, tuple)
    assert isinstance(fixture_piece.provenance[0].parents, tuple)
    assert isinstance(fixture_piece.quality_flags[0].entity_ids, tuple)


def test_decoded_times_are_rational(fixture_piece: CanonicalPiece) -> None:
    assert isinstance(fixture_piece.duration_qn, RationalTime)
    assert isinstance(fixture_piece.notes[0].onset_qn, RationalTime)
    assert isinstance(fixture_piece.bars[0].metric_offset_qn, RationalTime)


def test_provenance_details_are_sorted_tuples(fixture_piece: CanonicalPiece) -> None:
    assert fixture_piece.provenance[0].details == (
        ("description", "hand-authored Phase 1A contract fixture"),
        ("resolution", 480),
    )


def test_target_views_masks_and_unknown_confidence_survive(
    fixture_piece: CanonicalPiece,
) -> None:
    default, alternative, _ = fixture_piece.targets
    assert default.task == alternative.task
    assert default.annotation_view_id is None
    assert alternative.annotation_view_id == "analysis.alternative"
    assert default.confidence[0] is None
    assert default.mask[2] is False
    assert default.values[2] is None
    assert default.source[2] is None
    assert default.provenance[2] is None


@pytest.mark.parametrize(
    ("value_type", "class_labels", "values", "expected"),
    [
        ("scalar", None, [1, 2.5, 3, 4], (1, 2.5, 3, 4)),
        ("multi_label", ["a", "b"], [["a"], ["b"], ["a", "b"], ["a"]],
         (("a",), ("b",), ("a", "b"), ("a",))),
        ("distribution", ["a", "b"], [[1, 0], [0.5, 0.5], [0, 1], [0.25, 0.75]],
         ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (0.25, 0.75))),
    ],
)
def test_target_value_types_decode_by_declared_shape(
    valid_piece: CanonicalPiece,
    value_type: str,
    class_labels: list[str] | None,
    values: list[object],
    expected: tuple[object, ...],
) -> None:
    mapping = piece_to_dict(valid_piece)
    target = mapping["targets"][0]
    target["task"] = f"test.{value_type}"
    target["value_type"] = value_type
    target["class_labels"] = class_labels
    target["values"] = values
    decoded = piece_from_dict(mapping)
    assert decoded.targets[0].values == expected


def test_piece_to_dict_is_repeatable_and_nonmutating(
    fixture_piece: CanonicalPiece,
) -> None:
    before = fixture_piece
    first = piece_to_dict(fixture_piece)
    second = piece_to_dict(fixture_piece)
    assert first == second
    assert fixture_piece == before
    assert tuple(note.note_id for note in fixture_piece.notes) == (
        "note:melody-000",
        "note:melody-001",
        "note:drums-000",
        "note:melody-002",
        "note:melody-003",
        "note:drums-001",
    )


@pytest.mark.parametrize("indent", [None, 2, 4])
def test_dumps_piece_is_byte_deterministic(
    fixture_piece: CanonicalPiece, indent: int | None
) -> None:
    assert dumps_piece(fixture_piece, indent=indent) == dumps_piece(
        fixture_piece, indent=indent
    )


def test_compact_dump_uses_exact_json_options(fixture_piece: CanonicalPiece) -> None:
    expected = json.dumps(
        piece_to_dict(fixture_piece),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )
    assert dumps_piece(fixture_piece) == expected


def test_dump_preserves_unicode_and_sorts_object_keys(
    valid_piece: CanonicalPiece,
) -> None:
    metadata = replace(valid_piece.metadata, title="Музыкальная пьеса")
    piece = replace(valid_piece, metadata=metadata)
    payload = dumps_piece(piece)
    assert "Музыкальная пьеса" in payload
    assert "\\u" not in payload
    assert payload.startswith('{"annotations":')
    assert payload.index('"collection"', payload.index('"metadata"')) < payload.index(
        '"copyright"', payload.index('"metadata"')
    )


@pytest.mark.parametrize("indent", [None, 2])
def test_dumps_piece_has_no_terminal_newline(
    fixture_piece: CanonicalPiece, indent: int | None
) -> None:
    assert not dumps_piece(fixture_piece, indent=indent).endswith("\n")


@pytest.mark.parametrize("indent", [None, 2, 4])
def test_dump_piece_file_format_and_load(
    tmp_path: Path, fixture_piece: CanonicalPiece, indent: int | None
) -> None:
    path = tmp_path / f"piece-{indent}.json"
    dump_piece(fixture_piece, path, indent=indent)
    raw = path.read_bytes()
    assert raw == (dumps_piece(fixture_piece, indent=indent) + "\n").encode("utf-8")
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert load_piece(path) == fixture_piece


def test_dump_piece_default_indent_is_two(
    tmp_path: Path, fixture_piece: CanonicalPiece
) -> None:
    path = tmp_path / "piece.json"
    dump_piece(fixture_piece, path)
    assert path.read_text(encoding="utf-8") == dumps_piece(
        fixture_piece, indent=2
    ) + "\n"


def test_load_piece_propagates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_piece(tmp_path / "missing.json")


def test_dump_piece_does_not_create_parent_directories(
    tmp_path: Path, fixture_piece: CanonicalPiece
) -> None:
    with pytest.raises(FileNotFoundError):
        dump_piece(fixture_piece, tmp_path / "missing" / "piece.json")


def test_invalid_piece_is_rejected_before_every_writer(
    tmp_path: Path, valid_piece: CanonicalPiece
) -> None:
    invalid = replace(
        valid_piece,
        notes=(replace(valid_piece.notes[0], pitch=999), *valid_piece.notes[1:]),
    )
    path = tmp_path / "piece.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(CanonicalValidationError):
        piece_to_dict(invalid)
    with pytest.raises(CanonicalValidationError):
        dumps_piece(invalid)
    with pytest.raises(CanonicalValidationError):
        dump_piece(invalid, path)
    assert path.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize(
    ("mutation", "code", "path"),
    [
        (lambda data: data.__setitem__("extra~/field", 1), "JSON_UNKNOWN_FIELD", "/extra~0~1field"),
        (lambda data: data.pop("piece_id"), "JSON_MISSING_FIELD", "/piece_id"),
        (lambda data: data["metadata"].__setitem__("extra", 1), "JSON_UNKNOWN_FIELD", "/metadata/extra"),
        (lambda data: data["metadata"].pop("title"), "JSON_MISSING_FIELD", "/metadata/title"),
        (lambda data: data["notes"][0].__setitem__("extra", 1), "JSON_UNKNOWN_FIELD", "/notes/0/extra"),
        (lambda data: data["notes"][0].pop("pitch"), "JSON_MISSING_FIELD", "/notes/0/pitch"),
        (lambda data: data["targets"][0].__setitem__("extra", 1), "JSON_UNKNOWN_FIELD", "/targets/0/extra"),
        (lambda data: data["targets"][0].pop("mask"), "JSON_MISSING_FIELD", "/targets/0/mask"),
        (lambda data: data["duration_qn"].__setitem__("extra", 1), "JSON_UNKNOWN_FIELD", "/duration_qn/extra"),
        (lambda data: data["duration_qn"].pop("den"), "JSON_MISSING_FIELD", "/duration_qn/den"),
    ],
)
def test_unknown_and_missing_fields(
    fixture_dict: JsonObject,
    mutation: Any,
    code: str,
    path: str,
) -> None:
    mutation(fixture_dict)
    _assert_issue(_decode_error(fixture_dict), code, path)


@pytest.mark.parametrize(
    ("mutation", "path"),
    [
        (lambda data: data.__setitem__("tracks", {}), "/tracks"),
        (lambda data: data["tracks"].__setitem__(0, "bad"), "/tracks/0"),
        (lambda data: data.__setitem__("source_resolution", "480"), "/source_resolution"),
        (lambda data: data.__setitem__("source_resolution", True), "/source_resolution"),
        (lambda data: data.__setitem__("tracks", tuple(data["tracks"])), "/tracks"),
        (lambda data: data["provenance"][0]["details"].__setitem__("nested", []), "/provenance/0/details/nested"),
        (lambda data: data["targets"][0]["values"].__setitem__(0, ["major"]), "/targets/0/values/0"),
        (lambda data: data["targets"][0]["mask"].__setitem__(0, 1), "/targets/0/mask/0"),
        (lambda data: data["targets"][0]["confidence"].__setitem__(0, "unknown"), "/targets/0/confidence/0"),
        (lambda data: data["targets"][0].__setitem__("annotation_view_id", 3), "/targets/0/annotation_view_id"),
    ],
)
def test_json_runtime_type_errors(
    fixture_dict: JsonObject, mutation: Any, path: str
) -> None:
    mutation(fixture_dict)
    _assert_issue(_decode_error(fixture_dict), "JSON_TYPE_INVALID", path)


def test_top_level_must_be_mapping() -> None:
    _assert_issue(_decode_error([]), "JSON_TYPE_INVALID", "")


def test_non_string_mapping_key_is_rejected(fixture_dict: JsonObject) -> None:
    fixture_dict[3] = "invalid"  # type: ignore[index]
    _assert_issue(_decode_error(fixture_dict), "JSON_TYPE_INVALID", "")


def test_custom_mapping_is_supported_without_mutation(fixture_dict: JsonObject) -> None:
    original = deepcopy(fixture_dict)
    custom = UserDict(fixture_dict)
    piece = piece_from_dict(custom)
    assert isinstance(piece, CanonicalPiece)
    assert custom.data == original


def test_decoder_does_not_mutate_nested_inputs(fixture_dict: JsonObject) -> None:
    original = deepcopy(fixture_dict)
    piece_from_dict(fixture_dict)
    assert fixture_dict == original


@pytest.mark.parametrize(
    ("rational", "code"),
    [
        ({"num": 2, "den": 4}, "RATIONAL_NOT_NORMALIZED"),
        ({"num": 0, "den": 7}, "RATIONAL_NOT_NORMALIZED"),
        ({"num": 1, "den": -2}, "RATIONAL_NOT_NORMALIZED"),
        ({"num": True, "den": 1}, "JSON_TYPE_INVALID"),
        ({"num": 1, "den": 0}, "RATIONAL_INVALID"),
        ({"num": 1}, "JSON_MISSING_FIELD"),
        ({"num": 1, "den": 2, "extra": 3}, "JSON_UNKNOWN_FIELD"),
    ],
)
def test_rational_decoding_errors(
    fixture_dict: JsonObject, rational: JsonObject, code: str
) -> None:
    fixture_dict["duration_qn"] = rational
    error = _decode_error(fixture_dict)
    assert code in {issue.code for issue in error.report.issues}


def test_normalized_rational_is_accepted(fixture_dict: JsonObject) -> None:
    fixture_dict["duration_qn"] = {"num": 5, "den": 1}
    assert piece_from_dict(fixture_dict).duration_qn == RationalTime(5)


@pytest.mark.parametrize(
    ("version", "code"),
    [
        (2, "JSON_TYPE_INVALID"),
        ("2.0.1", "SCHEMA_VERSION_UNSUPPORTED"),
        ("3.0.0", "SCHEMA_VERSION_UNSUPPORTED"),
    ],
)
def test_schema_version_is_exact(
    fixture_dict: JsonObject, version: object, code: str
) -> None:
    fixture_dict["schema_version"] = version
    _assert_issue(_decode_error(fixture_dict), code, "/schema_version")


def test_missing_schema_version(fixture_dict: JsonObject) -> None:
    fixture_dict.pop("schema_version")
    _assert_issue(
        _decode_error(fixture_dict), "JSON_MISSING_FIELD", "/schema_version"
    )


def test_exact_schema_version_is_accepted(fixture_dict: JsonObject) -> None:
    assert piece_from_dict(fixture_dict).schema_version == "2.0.0"


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "{} trailing",
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
    ],
)
def test_invalid_json_syntax_and_constants_raise_json_decode_error(
    payload: str,
) -> None:
    with pytest.raises(json.JSONDecodeError):
        loads_piece(payload)


def test_invalid_utf8_propagates_unicode_decode_error() -> None:
    with pytest.raises(UnicodeDecodeError):
        loads_piece(b"\xff")


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    ("location", "path"),
    [
        ("note_seconds", "/notes/0/source_onset_seconds"),
        ("beat_strength", "/beats/0/strength"),
        ("target_confidence", "/targets/0/confidence/0"),
        ("distribution_value", "/targets/0/values/0/0"),
    ],
)
def test_huge_integer_float_fields_report_value_not_finite_without_mutation(
    fixture_dict: JsonObject,
    location: str,
    path: str,
    sign: int,
) -> None:
    huge_integer = sign * 10**10000
    if location == "note_seconds":
        fixture_dict["notes"][0]["source_onset_seconds"] = huge_integer
    elif location == "beat_strength":
        fixture_dict["beats"][0]["strength"] = huge_integer
    elif location == "target_confidence":
        fixture_dict["targets"][0]["confidence"][0] = huge_integer
    else:
        target = fixture_dict["targets"][0]
        target["value_type"] = "distribution"
        target["class_labels"] = ["major", "minor"]
        target["values"] = [
            [0.5, 0.5],
            [0.5, 0.5],
            None,
            [0.5, 0.5],
            None,
        ]
        target["values"][0][0] = huge_integer

    before = deepcopy(fixture_dict)
    error = _decode_error(fixture_dict)

    _assert_issue(error, "VALUE_NOT_FINITE", path)
    assert fixture_dict == before


def test_json_numeric_exponent_infinity_reaches_semantic_validation(
    fixture_text: str,
) -> None:
    payload = fixture_text.replace(
        '"source_onset_seconds": null',
        '"source_onset_seconds": 1e9999',
        1,
    )
    with pytest.raises(CanonicalValidationError) as caught:
        loads_piece(payload)
    _assert_issue(
        caught.value,
        "VALUE_NOT_FINITE",
        "/notes/0/source_onset_seconds",
    )


@pytest.mark.parametrize(
    ("mutation", "code", "path"),
    [
        (lambda data: data["notes"][0].__setitem__("pitch", 999), "PITCH_OUT_OF_RANGE", "/notes/0/pitch"),
        (lambda data: data["targets"][0]["confidence"].__setitem__(0, 2.0), "TARGET_CONFIDENCE_INVALID", "/targets/0/confidence/0"),
        (lambda data: data["targets"][0].__setitem__("annotation_view_id", " bad "), "TARGET_VIEW_INVALID", "/targets/0/annotation_view_id"),
        (lambda data: data["notes"][0].__setitem__("track_id", "track:missing"), "ENTITY_REFERENCE_INVALID", "/notes/0/track_id"),
        (lambda data: data.__setitem__("notes", list(reversed(data["notes"]))), "COLLECTION_ORDER_INVALID", "/notes"),
    ],
)
def test_semantic_errors_are_delegated_to_validator(
    fixture_dict: JsonObject, mutation: Any, code: str, path: str
) -> None:
    mutation(fixture_dict)
    error = _decode_error(fixture_dict)
    _assert_issue(error, code, path)
    assert not any(
        issue.code == "JSON_TYPE_INVALID" and issue.path == path
        for issue in error.report.issues
    )


def test_invalid_modal_string_reaches_validator(valid_piece: CanonicalPiece) -> None:
    mapping = piece_to_dict(valid_piece)
    mapping["key_signature_events"] = [
        {
            "key_signature_event_id": "keysig:000",
            "onset_qn": {"num": 0, "den": 1},
            "fifths": 0,
            "mode": "aeolian",
            "raw_value": None,
            "provenance_id": "prov:source",
        }
    ]
    _assert_issue(
        _decode_error(mapping),
        "FIELD_VALUE_INVALID",
        "/key_signature_events/0/mode",
    )


def test_modal_runtime_type_is_decoder_error(valid_piece: CanonicalPiece) -> None:
    mapping = piece_to_dict(valid_piece)
    mapping["key_signature_events"] = [
        {
            "key_signature_event_id": "keysig:000",
            "onset_qn": {"num": 0, "den": 1},
            "fifths": 0,
            "mode": 7,
            "raw_value": None,
            "provenance_id": "prov:source",
        }
    ]
    _assert_issue(
        _decode_error(mapping), "JSON_TYPE_INVALID", "/key_signature_events/0/mode"
    )


def test_scalar_bool_is_decoder_error(valid_piece: CanonicalPiece) -> None:
    mapping = piece_to_dict(valid_piece)
    target = mapping["targets"][0]
    target["task"] = "test.scalar"
    target["value_type"] = "scalar"
    target["class_labels"] = None
    target["values"][0] = True
    _assert_issue(
        _decode_error(mapping), "JSON_TYPE_INVALID", "/targets/0/values/0"
    )


def test_categorical_unknown_string_reaches_validator(
    fixture_dict: JsonObject,
) -> None:
    fixture_dict["targets"][0]["values"][0] = "unknown"
    _assert_issue(
        _decode_error(fixture_dict),
        "TARGET_VALUE_INVALID",
        "/targets/0/values/0",
    )


def test_nonfinite_direct_mapping_detail_reaches_validator(
    fixture_dict: JsonObject,
) -> None:
    fixture_dict["provenance"][0]["details"]["score"] = float("nan")
    _assert_issue(
        _decode_error(fixture_dict),
        "VALUE_NOT_FINITE",
        "/provenance/0/details/2/1",
    )


def test_multiple_structural_issues_are_sorted_deterministically(
    fixture_dict: JsonObject,
) -> None:
    fixture_dict["metadata"].pop("title")
    fixture_dict["tracks"] = {}
    fixture_dict["zzz"] = 1
    first = _decode_error(fixture_dict).report
    second = _decode_error(fixture_dict).report
    assert first == second
    assert first.issues == tuple(
        sorted(
            first.issues,
            key=lambda issue: (
                issue.path,
                issue.severity,
                issue.code,
                issue.entity_id or "",
                issue.message,
            ),
        )
    )
