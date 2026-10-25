"""Pure synthetic authoring recipes; no musical payload reads or annotation."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from music_critic.data import RationalTime
from music_critic.tasks.event_theory_enrichment import EnrichmentError
from scripts.author_corpus400_analysis import authored_field, build_authored_view
from tests.tasks.test_event_theory_enrichment import fixture as event_fixture


def q(value):
    value = Fraction(value)
    return dict(num=value.numerator, den=value.denominator)


def fixture():
    piece, _ = event_fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], pitch=60, spelling_step="C", spelling_alter=0),))
    binding = dict(path="synthetic:canonical", sha256="a" * 64, size_bytes=100)
    input_binding = dict(path="synthetic:packet", sha256="b" * 64, size_bytes=80)
    author = dict(kind="model", author_id="astra:synthetic", annotation_id="synthetic:run",
        model_requested="gpt-6-astra", reasoning_effort_requested="ultra", actual_backend_version=None,
        human_gold=False, input_sha256="b" * 64, output_sha256="c" * 64, evidence=["synthetic:authored-result"])
    recipe = dict(schema="Corpus400AuthoredRecipe@1", dataset_id=piece.dataset_name,
        piece_id=piece.piece_id, canonical_piece_sha256=binding["sha256"], view_id="astra:whole",
        author=author, work=dict(work_family_id="synthetic:family", genre="classical",
            genre_evidence=["synthetic:catalog"], source_extent="unknown"),
        harmonies=[dict(id="h0", start_qn=q(0), end_qn=q(2), fields={
            "analysis.sonority": authored_field({"tones": ["C", "E", "G"]}, ["synthetic:h0"])}),
            dict(id="h1", start_qn=q(2), end_qn=q(4), fields={
            "analysis.sonority": authored_field({"tones": ["D", "F", "A"]}, ["synthetic:h1"])})],
        note_judgments=[], objects=[], relations=[], layer_reviews={})
    return piece, binding, input_binding, recipe


def test_held_note_is_partitioned_at_authored_harmony_change_with_unknown_roles():
    piece, binding, packet, recipe = fixture()
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    rows = result["bundle"]["note_contexts"]
    assert [(r["start_qn"], r["end_qn"], r["harmony_id"]) for r in rows] == [
        (q(0), q(2), "h0"), (q(2), q(4), "h1")]
    assert {r["note_id"] for r in rows} == {piece.notes[0].note_id}
    assert all(not r["fields"]["analysis.temporal_role"]["available"] for r in rows)
    assert all(not r["fields"]["analysis.chord_membership"]["available"] for r in rows)
    assert result["assessment"]["independent_reviews"] == []
    assert result["report"]["complete_for_declared_scope"] is False


def test_sol_recipe_retains_generic_model_origin_and_unknown_masks():
    piece, binding, packet, recipe = fixture()
    recipe["author"].update(model_requested="gpt-6.1-sol", reasoning_effort_requested="high")
    for harmony in recipe["harmonies"]:
        harmony["fields"]["analysis.sonority"] = authored_field(
            {"tones": ["C", "E", "G"]}, ["synthetic:sol-decision"], origin="model_individual_context")
    add_template(recipe["harmonies"][0])
    recipe["harmonies"][0]["fields"]["analysis.key"] = authored_field(
        "C major", ["synthetic:sol-key"], origin="model_individual_context")
    before = deepcopy(recipe)
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    bundle = result["bundle"]
    assert bundle["provenance"]["author"]["model_requested"] == "gpt-6.1-sol"
    assert bundle["harmonies"][0]["fields"]["analysis.sonority"]["origin"] == "model_individual_context"
    fields = bundle["note_contexts"][0]["fields"]
    assert fields["analysis.chord_membership"]["derivation"]["upstream_origin"] == "model_individual_context"
    assert fields["analysis.preparation"]["available"] is False
    assert fields["analysis.preparation"]["value"] is None
    assert recipe == before


def test_new_sol_recipe_cannot_misattribute_decisions_to_astra():
    piece, binding, packet, recipe = fixture()
    recipe["author"].update(model_requested="gpt-6.1-sol", reasoning_effort_requested="high")
    with pytest.raises(EnrichmentError, match="origin.*model"):
        build_authored_view(piece, binding, recipe, input_binding=packet)


def add_template(harmony, *, complete=True):
    harmony["fields"]["analysis.key"] = authored_field("C major", ["synthetic:key"])
    harmony["note_template"] = dict(schema="AuthoredNoteTemplate@1", id="template:" + harmony["id"],
        basis="spelled_pitch_class", members=["C:0", "E:0", "G:0"], members_complete=complete,
        chord_degrees={"C:0": "1", "E:0": "3", "G:0": "5"},
        key_degrees={"C:0": "1", "D:0": "2"}, evidence=["synthetic:explicit-pitch-map"])


def test_only_explicit_template_projects_membership_and_degrees_not_temporal_meaning():
    piece, binding, packet, recipe = fixture()
    add_template(recipe["harmonies"][0])
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    first, second = result["bundle"]["note_contexts"]
    for name, expected in (("analysis.chord_membership", True), ("analysis.chord_degree", "1"), ("analysis.key_degree", "1")):
        field = first["fields"][name]
        assert field["value"] == expected and field["available"] is True
        assert field["origin"] == "source_derived"
        assert field["derivation"]["upstream_origin"] == "astra_individual_context"
        assert field["evidence"]
        assert second["fields"][name]["available"] is False
    for name in ("analysis.temporal_role", "analysis.preparation", "analysis.resolution"):
        assert first["fields"][name]["value"] is None
        assert first["fields"][name]["available"] is False
    assert result["bundle"]["harmonies"][0]["fields"]["analysis.function"]["available"] is False


@pytest.mark.parametrize("complete,available", [(False, False), (True, True)])
def test_absence_from_open_template_is_unknown_not_a_negative(complete, available):
    piece, binding, packet, recipe = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], pitch=62, spelling_step="D"),))
    add_template(recipe["harmonies"][0], complete=complete)
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    member = result["bundle"]["note_contexts"][0]["fields"]["analysis.chord_membership"]
    assert member["available"] is available
    assert member["value"] is (False if available else None)


def test_authored_subinterval_changes_only_that_context_and_preserves_unknown_tail():
    piece, binding, packet, recipe = fixture()
    recipe["note_judgments"] = [dict(note_id=piece.notes[0].note_id, start_qn=q(0), end_qn=q(1),
        fields={"analysis.temporal_role": authored_field("stable_support", ["synthetic:individual-role"])})]
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    rows = result["bundle"]["note_contexts"]
    assert [(r["start_qn"], r["end_qn"]) for r in rows] == [(q(0), q(1)), (q(1), q(2)), (q(2), q(4))]
    assert [r["fields"]["analysis.temporal_role"]["available"] for r in rows] == [True, False, False]


def test_exact_address_resolvers_reject_wrong_points_and_ambiguous_unisons():
    from scripts.author_corpus400_analysis import resolve_note_id, resolve_context_id
    piece, binding, packet, recipe = fixture()
    note = piece.notes[0]
    selector = dict(onset_qn=q(0), track_id=note.track_id, pitch=60)
    assert resolve_note_id(piece, selector) == note.note_id
    with pytest.raises(EnrichmentError, match="mismatched"):
        resolve_note_id(piece, dict(note_id=note.note_id, onset_qn=q(1)))
    duplicate = replace(note, note_id="note:duplicate")
    with pytest.raises(EnrichmentError, match="ambiguous"):
        resolve_note_id(replace(piece, notes=(note, duplicate)), selector)
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    with pytest.raises(EnrichmentError, match="mismatched"):
        resolve_context_id(result["bundle"], dict(note_id=note.note_id, start_qn=q(0), end_qn=q(0)))


@pytest.mark.parametrize("change,match", [
    (lambda p, r: r.update(independent_reviews=[]), "separate reviewer"),
    (lambda p, r: r["author"].update(input_sha256="f" * 64), "input packet"),
    (lambda p, r: r["author"].update(kind="source"), "Astra"),
    (lambda p, r: r["harmonies"][0]["fields"]["analysis.sonority"].update(evidence=[]), "evidence"),
    (lambda p, r: r["harmonies"][0]["fields"]["analysis.sonority"].update(origin="source_native"), "authored weak"),
    (lambda p, r: r["harmonies"][1].update(start_qn=q(1)), "overlapping harmonies"),
    (lambda p, r: r.update(note_judgments=[dict(note_id=p.notes[0].note_id, start_qn=q(1), end_qn=q(1),
        fields={"analysis.temporal_role": authored_field("passing", ["synthetic:point"])})]), "point/interval"),
])
def test_invalid_authorship_or_geometry_fails_closed(change, match):
    piece, binding, packet, recipe = fixture()
    change(piece, recipe)
    with pytest.raises(EnrichmentError, match=match):
        build_authored_view(piece, binding, recipe, input_binding=packet)


def test_writer_is_pure_and_never_fills_gaps_from_native_fields(monkeypatch):
    from pathlib import Path
    piece, binding, packet, recipe = fixture()
    before = deepcopy((piece, binding, packet, recipe))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("unexpected payload read"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("unexpected payload read"))
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    assert (piece, binding, packet, recipe) == before
    assert result["bundle"]["provenance"]["native_view_policy"] == "separate_file_never_used_to_fill_authored_unknowns"


def test_harmony_gap_has_dense_explicit_unknown_context():
    piece, binding, packet, recipe = fixture()
    recipe["harmonies"][0]["end_qn"] = q(1)
    recipe["harmonies"][1]["start_qn"] = q(3)
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    gap = result["bundle"]["note_contexts"][1]
    assert (gap["start_qn"], gap["end_qn"], gap["harmony_id"]) == (q(1), q(3), None)
    assert gap["fields"]["analysis.chord_membership"]["unknown_reason"] == "no_authored_harmony_for_interval"


def test_relations_and_objects_bind_to_the_new_contexts_and_keep_semantic_targets():
    from music_critic.tasks.whole_work_annotation import build_object_vocabulary, encode_object_fields
    piece, binding, packet, recipe = fixture()
    left = {"context": dict(note_id=piece.notes[0].note_id, start_qn=q(0), end_qn=q(2), harmony_id="h0")}
    right = {"context": dict(note_id=piece.notes[0].note_id, start_qn=q(2), end_qn=q(4), harmony_id="h1")}
    recipe["note_judgments"] = [dict(note_id=piece.notes[0].note_id, start_qn=q(0), end_qn=q(2),
        fields={"analysis.resolution": authored_field({"status": "linked", "relation_ids": ["r0"]}, ["synthetic:resolution"])})]
    recipe["relations"] = [dict(id="r0", type="resolves_to", source_refs=[left], target_refs=[right],
        start_qn=q(0), end_qn=q(4), **authored_field(True, ["synthetic:directed-resolution"]))]
    recipe["objects"] = [dict(id="phrase:0", object_type="phrase", member_refs=[left, right],
        start_qn=q(0), end_qn=q(4), fields={"analysis.function": authored_field("antecedent", ["synthetic:phrase"])})]
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    bundle = result["bundle"]
    first, second = bundle["note_contexts"]
    assert bundle["relations"][0]["source_ids"] == [first["id"]]
    assert bundle["relations"][0]["target_ids"] == [second["id"]]
    assert bundle["objects"][0]["member_ids"] == [first["id"], second["id"]]
    numeric = encode_object_fields(piece, bundle, build_object_vocabulary([bundle]))
    assert numeric["groups"]["phrase"]["masks"]["analysis.function"] == [True]
    assert numeric["groups"]["phrase"]["masks"]["analysis.question_answer"] == [False]


def test_point_note_has_no_inferred_sounding_membership_and_cannot_bind_to_nonzero_interval():
    piece, binding, packet, recipe = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], duration_qn=RationalTime(0), is_grace=True),))
    add_template(recipe["harmonies"][0])
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    row = result["bundle"]["note_contexts"][0]
    assert row["start_qn"] == row["end_qn"] == q(0)
    assert row["fields"]["analysis.chord_membership"]["unknown_reason"] == "point_event_no_sounding_interval"
    recipe["note_judgments"] = [dict(note_id=piece.notes[0].note_id, start_qn=q(0), end_qn=q(1),
        fields={"analysis.voice": authored_field("melody", ["synthetic:voice"])})]
    with pytest.raises(EnrichmentError, match="point/interval"):
        build_authored_view(piece, binding, recipe, input_binding=packet)


def test_spelled_template_never_falls_back_to_pitch_class_without_author_decision():
    piece, binding, packet, recipe = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], spelling_step=None, spelling_alter=None),))
    add_template(recipe["harmonies"][0])
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    field = result["bundle"]["note_contexts"][0]["fields"]["analysis.chord_degree"]
    assert field["available"] is False and field["unknown_reason"] == "template_requires_unavailable_note_spelling"
    template = recipe["harmonies"][0]["note_template"]
    template.update(basis="midi_pitch_class", members=["0", "4", "7"], chord_degrees={"0": "1"}, key_degrees={"0": "1"})
    with pytest.raises(EnrichmentError, match="enharmonic policy"):
        build_authored_view(piece, binding, recipe, input_binding=packet)
    template["enharmonic_policy"] = "author_declared_pc_mapping"
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    assert result["bundle"]["note_contexts"][0]["fields"]["analysis.chord_degree"]["value"] == "1"


@pytest.mark.parametrize("change,match", [
    (lambda r: r["harmonies"][0]["fields"].pop("analysis.key"), "available key"),
    (lambda r: r["harmonies"][0]["note_template"].update(members_complete=None), "completeness"),
    (lambda r: r["harmonies"][0]["note_template"].update(evidence=[]), "evidence"),
    (lambda r: r["harmonies"][0]["note_template"]["chord_degrees"].update({"C:0": True}), "degree map"),
])
def test_templates_need_explicit_complete_and_keyed_evidence(change, match):
    piece, binding, packet, recipe = fixture()
    add_template(recipe["harmonies"][0])
    change(recipe)
    with pytest.raises(EnrichmentError, match=match):
        build_authored_view(piece, binding, recipe, input_binding=packet)


def test_conflicting_authored_membership_cannot_silently_override_template():
    piece, binding, packet, recipe = fixture()
    add_template(recipe["harmonies"][0])
    recipe["note_judgments"] = [dict(note_id=piece.notes[0].note_id, start_qn=q(0), end_qn=q(2),
        fields={"analysis.chord_membership": authored_field(False, ["synthetic:conflict"])})]
    with pytest.raises(EnrichmentError, match="contradicts"):
        build_authored_view(piece, binding, recipe, input_binding=packet)


@pytest.mark.parametrize("members,complete", [(["C:0", "E:0"], True), (["C:0", "F:0"], False)])
def test_template_cannot_contradict_or_silently_drop_explicit_sonority_tones(members, complete):
    piece, binding, packet, recipe = fixture()
    add_template(recipe["harmonies"][0])
    recipe["harmonies"][0]["note_template"].update(members=members, members_complete=complete)
    with pytest.raises(EnrichmentError, match="sonority"):
        build_authored_view(piece, binding, recipe, input_binding=packet)


@pytest.mark.parametrize("tone,pitch,step,alter", [("B#3", 60, "B", 1), ("Cb4", 59, "C", -1)])
def test_exact_pitch_template_preserves_enharmonic_octave_crossings(tone, pitch, step, alter):
    piece, binding, packet, recipe = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], pitch=pitch, spelling_step=step, spelling_alter=alter),))
    harmony = recipe["harmonies"][0]
    add_template(harmony)
    harmony["fields"]["analysis.sonority"]["value"] = {"tones": [tone]}
    harmony["note_template"].update(basis="midi_pitch", members=[str(pitch)],
        chord_degrees={str(pitch): "1"}, key_degrees={str(pitch): "1"})
    result = build_authored_view(piece, binding, recipe, input_binding=packet)
    field = result["bundle"]["note_contexts"][0]["fields"]["analysis.chord_membership"]
    assert field["available"] is True and field["value"] is True
