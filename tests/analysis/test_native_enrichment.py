"""Synthetic tests of native targets, not musical quality."""
from copy import deepcopy
from fractions import Fraction

import pytest

from music_critic.analysis.native_enrichment import build_native_enrichment


def q(num, den=1):
    return {"num": num, "den": den}


def fixture_piece():
    return {
        "dataset_name": "hooktheory", "piece_id": "piece:fixture", "split": None,
        "metadata": {}, "provenance": [], "quality_flags": [],
        "notes": [
            {"note_id": "n1", "onset_qn": q(0), "duration_qn": q(3), "pitch": 60, "track_id": "melody"},
            {"note_id": "n2", "onset_qn": q(3), "duration_qn": q(0), "pitch": 62, "track_id": "melody", "is_grace": True},
        ],
        "annotations": [
            {"annotation_id": "h1", "annotation_type": "theory.chord", "start_qn": q(0), "end_qn": q(1)},
            {"annotation_id": "h2", "annotation_type": "theory.chord", "start_qn": q(1), "end_qn": q(2)},
            {"annotation_id": "k1", "annotation_type": "theory.local_key", "start_qn": q(0), "end_qn": q(3, 2)},
            {"annotation_id": "k2", "annotation_type": "theory.local_key", "start_qn": q(3, 2), "end_qn": q(3)},
        ],
        "targets": [
            target("theory.chord.adds", ["h1", "h2"], [["9"], []], [True, True]),
            target("theory.chord.borrowed", ["h1", "h2"], ["minor", None], [True, False]),
            target("theory.local_key.mode", ["k1", "k2"], ["major", "minor"], [True, True]),
            target("theory.melody.scale_degree", ["n1", "n2"], ["1", "2"], [True, True], alignment="note"),
        ],
    }


def target(task, ids, values, masks, alignment="annotation_span", view=None):
    return {"target_id": task + ":" + str(view), "task": task, "annotation_view_id": view,
            "alignment_type": alignment, "entity_ids": ids, "values": values, "mask": masks,
            "confidence": [None] * len(ids), "source": ["dataset_annotation"] * len(ids),
            "provenance": ["native"] * len(ids)}


def build(piece=None):
    return build_native_enrichment(piece or fixture_piece(), canonical_binding={
        "path": "fixture.json", "sha256": "a" * 64, "size_bytes": 1},
        source_bindings=[], admission={"current_split": "train", "native_split": "TRAIN"})


def test_partitions_held_note_at_harmony_key_and_gap_boundaries():
    result = build()
    rows = [r for r in result["note_contexts"] if r["note_id"] == "n1"]
    assert [(r["start_qn"], r["end_qn"], r["harmony_id"]) for r in rows] == [
        (q(0), q(1), "h1"), (q(1), q(3, 2), "h2"),
        (q(3, 2), q(2), "h2"), (q(2), q(3), None)]
    assert [r["fields"]["theory.local_key.mode"]["value"] for r in rows] == ["major", "major", "minor", "minor"]
    assert result["coverage"]["note_count"] == 2
    assert result["coverage"]["unique_notes_accounted"] == 2


def test_native_decorations_and_unknowns_are_not_simplified():
    result = build()
    a, b = result["harmonies"]
    assert a["fields"]["theory.chord.adds"]["value"] == ["9"]
    assert a["fields"]["theory.chord.borrowed"]["value"] == "minor"
    assert b["fields"]["theory.chord.adds"]["available"] is True
    assert b["fields"]["theory.chord.adds"]["value"] == []
    assert b["fields"]["theory.chord.borrowed"]["available"] is False
    assert b["fields"]["theory.chord.borrowed"]["value"] is None
    assert a["fields"]["analysis.harmonic_function"]["available"] is False
    assert result["relations"] == []


def test_point_notes_retained_with_no_sounding_harmony():
    row = build()["note_contexts"][-1]
    assert row["note_id"] == "n2" and row["start_qn"] == row["end_qn"] == q(3)
    assert row["harmony_id"] is None
    assert row["fields"]["analysis.temporal_role"]["unknown_reason"] == "point_event_no_sounding_interval"


def test_alternate_views_remain_separate():
    piece = fixture_piece()
    original = piece["targets"][0]
    original["annotation_view_id"] = "analyst_a"
    alt = deepcopy(original)
    alt["annotation_view_id"] = "analyst_b"
    alt["values"] = [["11"], []]
    piece["targets"].append(alt)
    result = build(piece)
    assert {h["harmony_view"] for h in result["harmonies"]} == {None, "analyst_a", "analyst_b"}
    assert len({r["id"] for r in result["note_contexts"]}) == len(result["note_contexts"])
    assert {h["fields"]["theory.chord.adds"]["value"][0] for h in result["harmonies"]
            if h["source_annotation_id"] == "h1" and h["harmony_view"] is not None} == {"9", "11"}


def test_default_view_survives_conflicting_alternative_without_cross_view_fields():
    piece = fixture_piece()
    piece["targets"].append(target("theory.chord.adds", ["h1"], [["11"]], [True], view="alternative"))
    piece["targets"].append(target("theory.melody.scale_degree", ["n1"], ["b3"], [True], alignment="note", view="alternative"))
    result = build(piece)
    harmonies = {h["harmony_view"]: h for h in result["harmonies"] if h["source_annotation_id"] == "h1"}
    assert harmonies[None]["fields"]["theory.chord.adds"]["value"] == ["9"]
    assert harmonies["alternative"]["fields"]["theory.chord.adds"]["value"] == ["11"]
    assert "theory.chord.borrowed" not in harmonies["alternative"]["fields"]
    rows = {r["harmony_view"]: r for r in result["note_contexts"] if r["note_id"] == "n1" and r["start_qn"] == q(0)}
    assert rows[None]["fields"]["theory.melody.scale_degree"]["value"] == "1"
    assert rows["alternative"]["fields"]["theory.melody.scale_degree"]["value"] == "b3"
    assert rows["alternative"]["fields"]["theory.local_key.mode"]["available"] is False
    for view in (None, "alternative"):
        assert sum(Fraction(r["end_qn"]["num"], r["end_qn"]["den"]) - Fraction(r["start_qn"]["num"], r["start_qn"]["den"])
                   for r in result["note_contexts"] if r["note_id"] == "n1" and r["harmony_view"] == view) == 3


def test_zero_chord_span_remains_nonsounding_object_with_no_note_membership():
    piece = fixture_piece()
    piece["annotations"][0]["end_qn"] = q(0)
    result = build(piece)
    assert "h1" not in {h["id"] for h in result["harmonies"]}
    obj, = result["objects"]
    assert obj["source_annotation_id"] == "h1"
    assert obj["start_qn"] == obj["end_qn"] == q(0)
    assert obj["applicability"] == "nonsounding_native_annotation"
    assert obj["fields"]["theory.chord.adds"]["value"] == ["9"]
    assert obj["fields"]["theory.chord.borrowed"]["value"] == "minor"
    row = result["note_contexts"][0]
    assert row["harmony_id"] is None
    assert row["fields"]["analysis.temporal_role"]["available"] is False
    assert result["coverage"]["unique_notes_accounted"] == 2


def test_overlapping_harmony_not_arbitrarily_selected():
    piece = fixture_piece()
    piece["annotations"][0]["end_qn"] = q(2)
    result = build(piece)
    row = next(r for r in result["note_contexts"] if r["start_qn"] == q(1))
    assert row["harmony_id"] is None
    assert set(row["active_harmony_ids"]) == {"h1", "h2"}


@pytest.mark.parametrize("change", ["bad_note_id", "non_reduced_time", "negative_duration", "masked_value"])
def test_rejects_corrupt_source_geometry_or_binding(change):
    piece = fixture_piece()
    if change == "bad_note_id": piece["targets"][-1]["entity_ids"][0] = "absent"
    if change == "non_reduced_time": piece["notes"][0]["onset_qn"] = q(0, 2)
    if change == "negative_duration": piece["notes"][0]["duration_qn"] = q(-1)
    if change == "masked_value": piece["targets"][1]["values"][1] = "unknown"
    with pytest.raises(ValueError): build(piece)


def test_input_immutable_and_known_false_survives():
    piece = fixture_piece()
    piece["targets"].append(target("theory.synthetic_flag", ["n1"], [False], [True], alignment="note"))
    before = deepcopy(piece)
    result = build(piece)
    assert piece == before
    field = result["note_contexts"][0]["fields"]["theory.synthetic_flag"]
    assert field["value"] is False and field["available"] is True


def test_requires_train_without_changing_native_split():
    with pytest.raises(ValueError):
        build_native_enrichment(fixture_piece(), canonical_binding={}, source_bindings=[],
                                admission={"current_split": "test", "native_split": "TRAIN"})
