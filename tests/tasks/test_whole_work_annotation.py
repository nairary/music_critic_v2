"""Synthetic whole-work admission checks; no evidence of musical accuracy."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256

import pytest

from tests.tasks.test_event_theory_enrichment import fixture as event_fixture
from music_critic.tasks.event_theory_enrichment import EnrichmentError, canonical
from music_critic.data import RationalTime
from music_critic.tasks.whole_work_annotation import (
    default_whole_work_profile,
    validate_annotation_completeness,
)


def rational(value):
    value = Fraction(value)
    return {"num": value.numerator, "den": value.denominator}


def known(value=True):
    return dict(value=value, available=True, origin="astra_individual_context",
                confidence=None, evidence=["synthetic:analysis"], unknown_reason=None)


def semantic_fields(names):
    values = {
        "analysis.chord_membership": True,
        "analysis.sonority": {"tones": ["C", "E", "G"], "root": "C", "quality": "major"},
        "analysis.extensions": ["7"],
        "analysis.transformations": ["literal_repetition"],
        "analysis.parts_voices": ["voice:synthetic"],
        "analysis.interactions": ["parallel_motion"],
        "analysis.repetition_contrast": ["repetition"],
        "analysis.preparation": {"status": "none", "relation_ids": [], "reason": "No preparation in this synthetic context."},
        "analysis.resolution": {"status": "none", "relation_ids": [], "reason": "No resolution in this synthetic context."},
    }
    return {name: known(deepcopy(values.get(name, "synthetic_label"))) for name in names}


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def fixture():
    piece, bundle = event_fixture()
    end = rational(piece.duration_qn.to_fraction())
    span = dict(start_qn=rational(0), end_qn=end)
    bundle["canonical_piece"] = dict(path="synthetic.json", sha256="a" * 64, size_bytes=1)
    profile = default_whole_work_profile()
    view_id = "interpretation:a"
    for row in bundle["note_contexts"]:
        row["view_id"] = view_id
        row["harmony_id"] = "harmony:0"
        row["fields"] = semantic_fields(profile["layers"]["notes"]["required_fields"])
    bundle["harmonies"] = [dict(id="harmony:0", view_id=view_id, **span,
        fields=semantic_fields(profile["layers"]["harmony"]["required_fields"]))]
    bundle["objects"] = []
    for layer in profile["layers"].values():
        if layer["scope"] == "objects":
            bundle["objects"].append(dict(id=layer["object_type"] + ":0", view_id=view_id,
                object_type=layer["object_type"], member_ids=["note:n0"], **span,
                fields=semantic_fields(layer["required_fields"])))
    author = dict(author_id="annotator:a", kind="model", annotation_id="run:a",
        model_requested="gpt-6-astra", reasoning_effort_requested="ultra",
        actual_backend_version=None, human_gold=False, input_sha256="b" * 64,
        output_sha256="c" * 64, evidence=["synthetic:run:a"])
    assessment = dict(schema="WholeWorkAnnotationAssessment@1", dataset_id=piece.dataset_name,
        piece_id=piece.piece_id, canonical_piece_sha256="a" * 64,
        bundle_fingerprint=digest(bundle), profile_fingerprint=digest(profile),
        work=dict(work_family_id="synthetic:work", genre="classical", genre_evidence=["synthetic:catalog"],
            source_extent="full_work", full_work_evidence=["synthetic:complete-source"], **span),
        views=[dict(view_id=view_id, author=author, layers={name: dict(review_status="reviewed",
            reviewed_intervals=[span], evidence=["synthetic:reviewed-interval"])
            for name in profile["layers"]})],
        independent_reviews=[dict(view_id=view_id, reviewer_id="reviewer:b", reviewer_kind="model",
            result="accepted", bundle_fingerprint=digest(bundle), evidence=["synthetic:review:b"])])
    return piece, bundle, profile, assessment


def rebind(bundle, assessment):
    assessment["bundle_fingerprint"] = digest(bundle)
    for review in assessment["independent_reviews"]:
        review["bundle_fingerprint"] = digest(bundle)


def test_full_geometry_does_not_make_empty_semantic_fields_complete():
    piece, bundle, profile, assessment = fixture()
    for row in bundle["note_contexts"]:
        row["fields"] = {}
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["full_work_verified"] is True
    assert report["all_layers_reviewed"] is True
    assert report["all_required_content_accounted"] is False
    assert report["complete_for_declared_scope"] is False
    notes = report["views"]["interpretation:a"]["layers"]["notes"]
    assert notes["structural_coverage_complete"] is True
    assert notes["fields"]["analysis.temporal_role"]["counts"]["missing"] == 2
    with pytest.raises(EnrichmentError, match="incomplete"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile, require_complete=True)


def test_complete_synthetic_receipts_are_not_a_human_gold_claim():
    piece, bundle, profile, assessment = fixture()
    before = deepcopy((bundle, profile, assessment))
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile, require_complete=True)
    assert report["complete_for_declared_scope"] is True
    assert report["human_gold"] is False
    assert (bundle, profile, assessment) == before


def test_sol_high_author_is_preserved_without_claiming_known_backend_or_gold():
    piece, bundle, profile, assessment = fixture()
    author = assessment["views"][0]["author"]
    author.update(model_requested="gpt-6.1-sol", reasoning_effort_requested="high")
    before = deepcopy(assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["human_gold"] is False
    assert assessment == before
    assert author["actual_backend_version"] is None


@pytest.mark.parametrize("changes", [
    dict(model_requested="gpt-6.1-sol", reasoning_effort_requested="ultra"),
    dict(model_requested="unknown-model", reasoning_effort_requested="high"),
    dict(model_requested="gpt-6.1-sol", reasoning_effort_requested="high", human_gold=True),
])
def test_new_author_settings_do_not_relax_weak_provenance(changes):
    piece, bundle, profile, assessment = fixture()
    assessment["views"][0]["author"].update(changes)
    with pytest.raises(EnrichmentError, match="weak provenance"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)


def test_partial_views_cannot_union_their_note_intervals():
    piece, bundle, profile, assessment = fixture()
    bundle["note_contexts"][1]["view_id"] = "interpretation:b"
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["views"]["interpretation:a"]["layers"]["notes"]["structural_coverage_complete"] is False
    assert report["complete_for_declared_scope"] is False


def test_unknown_fields_are_counted_without_becoming_negative_targets():
    piece, bundle, profile, assessment = fixture()
    field = known(False)
    bundle["note_contexts"][0]["fields"]["analysis.chord_membership"] = field
    bundle["note_contexts"][1]["fields"]["analysis.chord_membership"] = dict(
        value=None, available=False, origin="unknown", confidence=None, evidence=[],
        unknown_reason="not reviewed", annotation_state="unreviewed")
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    summary = report["views"]["interpretation:a"]["layers"]["notes"]["fields"]["analysis.chord_membership"]
    assert summary["counts"]["known"] == 1
    assert summary["counts"]["unreviewed"] == 1
    assert summary["duration_qn"]["known"] == rational(2)
    assert summary["duration_qn"]["unreviewed"] == rational(2)
    assert report["all_layers_reviewed"] is True
    assert report["all_required_content_accounted"] is False


def test_field_not_applicable_requires_reason_and_remains_unavailable():
    piece, bundle, profile, assessment = fixture()
    field = dict(value=None, available=False, origin="unknown", confidence=None,
        unknown_reason="non-tonal context", annotation_state="not_applicable", evidence=["synthetic:atonal"])
    bundle["note_contexts"][0]["fields"]["analysis.key_degree"] = field
    rebind(bundle, assessment)
    with pytest.raises(EnrichmentError, match="musical reason"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    field["applicability_reason"] = "This interpretation makes no tonal-center claim in this interval."
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    summary = report["views"]["interpretation:a"]["layers"]["notes"]["fields"]["analysis.key_degree"]
    assert summary["counts"]["not_applicable"] == 1
    assert summary["counts"]["known"] == 1
    assert report["complete_for_declared_scope"] is True
    assert field["available"] is False


def test_empty_layer_is_unknown_until_explicit_permitted_nonapplicability():
    piece, bundle, profile, assessment = fixture()
    bundle["objects"] = [o for o in bundle["objects"] if o["object_type"] != "ensemble"]
    rebind(bundle, assessment)
    assert not validate_annotation_completeness(piece, bundle, assessment, profile=profile)["complete_for_declared_scope"]
    layer = assessment["views"][0]["layers"]["ensemble"]
    layer["review_status"] = "not_applicable"
    with pytest.raises(EnrichmentError, match="musical reason"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    layer["applicability_reason"] = "The selected version contains a single monophonic line."
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["complete_for_declared_scope"] is True
    assert report["views"]["interpretation:a"]["layers"]["ensemble"]["fields"] == {}


@pytest.mark.parametrize("change", [
    lambda p, b, a: a["work"].update(source_extent="excerpt"),
    lambda p, b, a: a.update(independent_reviews=[]),
    lambda p, b, a: a["views"][0]["layers"]["form"].update(reviewed_intervals=[
        dict(start_qn=rational(0), end_qn=rational(2))]),
    lambda p, b, a: b["objects"][-2]["fields"].clear(),
])
def test_evidence_or_review_gaps_block_full_completion(change):
    piece, bundle, profile, assessment = fixture()
    change(piece, bundle, assessment)
    rebind(bundle, assessment)
    assert not validate_annotation_completeness(piece, bundle, assessment, profile=profile)["complete_for_declared_scope"]


@pytest.mark.parametrize("change,match", [
    (lambda b, p, a: a.update(canonical_piece_sha256="f" * 64), "canonical hash"),
    (lambda b, p, a: a.update(bundle_fingerprint="f" * 64), "bundle fingerprint"),
    (lambda b, p, a: a["work"].update(full_work_evidence=[]), "source evidence"),
    (lambda b, p, a: a["work"].update(cropped=True), "cropped"),
    (lambda b, p, a: a["work"].update(genre_evidence=[]), "genre"),
    (lambda b, p, a: a["work"].update(work_family_id=""), "work-family"),
    (lambda b, p, a: a["views"][0]["author"].update(human_gold=True), "weak provenance"),
    (lambda b, p, a: a["views"][0]["author"].pop("output_sha256"), "input/output"),
    (lambda b, p, a: a["independent_reviews"][0].update(reviewer_id="annotator:a"), "independent reviewer"),
    (lambda b, p, a: p["layers"]["form"].update(required_fields=[]), "required fields"),
])
def test_false_completion_provenance_is_rejected(change, match):
    piece, bundle, profile, assessment = fixture()
    change(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match=match):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)


def test_object_sidecar_preserves_fields_false_zero_unknown_and_views():
    from music_critic.tasks.whole_work_annotation import build_object_vocabulary, encode_object_fields
    piece, bundle, _, _ = fixture()
    first = bundle["objects"][0]
    first["fields"] = {"analysis.flag": known(False)}
    second = deepcopy(first)
    second["id"] = "process:alternative"
    second["view_id"] = "interpretation:b"
    second["fields"]["analysis.flag"] = known(0)
    third = deepcopy(first)
    third["id"] = "process:unknown"
    third["fields"]["analysis.flag"] = dict(value=None, available=False, origin="unknown",
        confidence=None, evidence=[], unknown_reason="not reviewed")
    bundle["objects"] = [first, second, third]
    vocabulary = build_object_vocabulary([bundle])
    numeric = encode_object_fields(piece, bundle, vocabulary)
    group = numeric["groups"]["harmonic_process"]
    assert group["view_ids"] == ["interpretation:a", "interpretation:b", "interpretation:a"]
    assert group["values"]["analysis.flag"][0] != group["values"]["analysis.flag"][1]
    assert group["values"]["analysis.flag"][2] == -1
    assert group["masks"]["analysis.flag"] == [True, True, False]
    assert group["member_ids"] == [["note:n0"]] * 3
    assert numeric["oov_counts"] == {}


def test_unresolved_independent_review_is_not_overridden_by_another_acceptance():
    piece, bundle, profile, assessment = fixture()
    review = deepcopy(assessment["independent_reviews"][0])
    review.update(reviewer_id="reviewer:c", result="changes_requested")
    assessment["independent_reviews"].append(review)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["independently_reviewed"] is False
    assert report["complete_for_declared_scope"] is False


def test_complete_notes_require_binding_to_their_own_harmonic_view():
    piece, bundle, profile, assessment = fixture()
    bundle["note_contexts"][0]["harmony_id"] = None
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["complete_for_declared_scope"] is False
    assert "unbound_harmonic_context" in report["views"]["interpretation:a"]["layers"]["notes"]["issues"]


def test_whole_harmony_interval_includes_time_outside_sounding_notes():
    piece, bundle, profile, assessment = fixture()
    # All canonical notes still have harmonic coverage; the full work has a tail.
    piece = replace(piece, notes=(replace(piece.notes[0], duration_qn=RationalTime(2)),))
    bundle["note_contexts"][0]["end_qn"] = rational(1)
    bundle["note_contexts"][1].update(start_qn=rational(1), end_qn=rational(2))
    bundle["harmonies"][0]["end_qn"] = rational(2)
    assert piece.duration_qn.to_fraction() > 2
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["views"]["interpretation:a"]["layers"]["notes"]["structural_coverage_complete"] is True
    assert report["views"]["interpretation:a"]["layers"]["harmony"]["structural_coverage_complete"] is False
    assert report["complete_for_declared_scope"] is False


def test_numeric_objects_mask_new_values_fields_and_types_and_authenticate_vocabulary():
    from music_critic.tasks.whole_work_annotation import build_object_vocabulary, encode_object_fields
    piece, bundle, _, _ = fixture()
    obj = bundle["objects"][0]
    bundle["objects"] = [obj]
    obj["fields"] = {"analysis.flag": known(False)}
    vocabulary = build_object_vocabulary([bundle])
    obj["fields"]["analysis.flag"] = known(True)
    obj["fields"]["analysis.new_field"] = known("new")
    alternate = deepcopy(obj)
    alternate.update(id="new:object", object_type="novel_type")
    bundle["objects"].append(alternate)
    numeric = encode_object_fields(piece, bundle, vocabulary)
    assert numeric["oov_counts"] == {
        "harmonic_process:analysis.flag": 1, "harmonic_process:analysis.new_field": 1,
        "novel_type:analysis.flag": 1, "novel_type:analysis.new_field": 1,
    }
    assert numeric["groups"]["novel_type"]["values"]["analysis.flag"] == [-1]
    assert numeric["groups"]["harmonic_process"]["masks"]["analysis.new_field"] == [False]
    vocabulary["fields"]["harmonic_process"]["analysis.flag"].append(True)
    with pytest.raises(EnrichmentError, match="fingerprint"):
        encode_object_fields(piece, bundle, vocabulary)


def test_mandatory_semantics_cannot_all_be_marked_not_applicable():
    piece, bundle, profile, assessment = fixture()
    bundle["objects"][0]["fields"]["analysis.process"] = dict(value=None, available=False,
        origin="unknown", confidence=None, evidence=["synthetic:review"], unknown_reason="not applicable",
        annotation_state="not_applicable", applicability_reason="No interpretation was chosen.")
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    assert report["all_required_content_accounted"] is False
    assert report["complete_for_declared_scope"] is False


def test_known_empty_values_need_an_explicit_negative_decision_reason():
    piece, bundle, profile, assessment = fixture()
    field = known([])
    bundle["harmonies"][0]["fields"]["analysis.extensions"] = field
    rebind(bundle, assessment)
    with pytest.raises(EnrichmentError, match="empty value"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    field["decision_reason"] = "The analyst explicitly found no added tones in this sonority."
    rebind(bundle, assessment)
    assert validate_annotation_completeness(piece, bundle, assessment, profile=profile)["complete_for_declared_scope"]


def test_incomplete_view_reports_unrepresented_note_time():
    piece, bundle, profile, assessment = fixture()
    bundle["note_contexts"][1]["view_id"] = "interpretation:b"
    rebind(bundle, assessment)
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    coverage = report["views"]["interpretation:a"]["layers"]["notes"]["coverage"]
    assert coverage["expected_duration_qn"] == rational(4)
    assert coverage["covered_duration_qn"] == rational(2)
    assert coverage["unrepresented_duration_qn"] == rational(2)
    assert coverage["fully_covered_notes"] == 0


def linked_fixture():
    piece, bundle, profile, assessment = fixture()
    edge = dict(id="resolution:0", type="resolves_to", view_id="interpretation:a",
        source_ids=["c0"], target_ids=["c1"], start_qn=rational(0), end_qn=rational(4), **known(True))
    bundle["relations"] = [edge]
    bundle["note_contexts"][0]["fields"]["analysis.resolution"] = known({
        "status": "linked", "relation_ids": [edge["id"]]})
    rebind(bundle, assessment)
    return piece, bundle, profile, assessment


def test_resolution_addresses_a_positive_edge_with_actual_note_context_endpoints():
    piece, bundle, profile, assessment = linked_fixture()
    assert validate_annotation_completeness(piece, bundle, assessment, profile=profile)["complete_for_declared_scope"]


@pytest.mark.parametrize("change", [
    lambda b: b["note_contexts"][0]["fields"]["analysis.resolution"].update(value=True),
    lambda b: b["note_contexts"][0]["fields"]["analysis.resolution"]["value"].update(relation_ids=["missing"]),
    lambda b: b["relations"][0].update(view_id="other"),
    lambda b: b["note_contexts"][1].update(view_id="other"),
    lambda b: b["relations"][0].update(source_ids=["c1"], target_ids=["c0"]),
    lambda b: b["relations"][0].update(value=False),
    lambda b: b["relations"][0].update(type="prepares"),
    lambda b: b["note_contexts"][0]["fields"]["analysis.resolution"].update(value={
        "status": "none", "relation_ids": [], "reason": "Claimed absent despite positive edge."}),
])
def test_dangling_cross_view_or_blanket_positive_resolution_is_rejected(change):
    piece, bundle, profile, assessment = linked_fixture()
    change(bundle)
    rebind(bundle, assessment)
    with pytest.raises(EnrichmentError, match="directed|relation|resolution"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)


def test_preparation_targets_incoming_edges_and_rejects_unreasoned_absence():
    piece, bundle, profile, assessment = fixture()
    edge = dict(id="preparation:0", type="prepares", view_id="interpretation:a", source_ids=["c0"],
        target_ids=["c1"], start_qn=rational(0), end_qn=rational(4), **known(True))
    bundle["relations"] = [edge]
    bundle["note_contexts"][1]["fields"]["analysis.preparation"] = known({
        "status": "linked", "relation_ids": [edge["id"]]})
    rebind(bundle, assessment)
    assert validate_annotation_completeness(piece, bundle, assessment, profile=profile)["complete_for_declared_scope"]
    bundle["note_contexts"][0]["fields"]["analysis.preparation"]["value"].pop("reason")
    rebind(bundle, assessment)
    with pytest.raises(EnrichmentError, match="reason"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)


@pytest.mark.parametrize("scope,index,name", [
    ("harmonies", 0, "analysis.sonority"),
    ("harmonies", 0, "analysis.roman"),
    ("harmonies", 0, "analysis.extensions"),
    ("objects", 0, "analysis.process"),
    ("note_contexts", 0, "analysis.temporal_role"),
])
def test_boolean_is_not_a_complete_semantic_analysis(scope, index, name):
    piece, bundle, profile, assessment = fixture()
    bundle[scope][index]["fields"][name] = known(True)
    rebind(bundle, assessment)
    with pytest.raises(EnrichmentError, match="semantic|sonority"):
        validate_annotation_completeness(piece, bundle, assessment, profile=profile)
