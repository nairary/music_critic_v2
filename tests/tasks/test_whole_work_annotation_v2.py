"""Synthetic v2 accounting; no corpus payload or claim of musical accuracy."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from music_critic.tasks.event_theory_enrichment import EnrichmentError
from music_critic.tasks.whole_work_annotation import annotation_fingerprint
from music_critic.data import RationalTime
from music_critic.data.schema import ProvenanceRecord
from music_critic.tasks.whole_work_annotation_v2 import (
    assessment_fingerprint,
    default_whole_work_profile_v2,
    validate_annotation_completeness_v2,
)
from tests.tasks.test_whole_work_annotation import fixture as v1_fixture
from tests.tasks.test_whole_work_annotation import rational


def fixture():
    piece, bundle, _, assessment = v1_fixture()
    profile = default_whole_work_profile_v2()
    assessment["schema"] = "WholeWorkAnnotationAssessment@2"
    assessment["work"]["musical_scope"] = {
        key: deepcopy(assessment["work"][key]) for key in ("start_qn", "end_qn")
    }
    assessment["work"]["non_musical_intervals"] = []
    assessment["views"][0]["reviewed_unknowns"] = []
    assessment["views"][0]["layer_dispositions"] = []
    rebind(bundle, profile, assessment)
    return piece, bundle, profile, assessment


def rebind(bundle, profile, assessment):
    assessment["bundle_fingerprint"] = annotation_fingerprint(bundle)
    assessment["profile_fingerprint"] = annotation_fingerprint(profile)
    for review in assessment["independent_reviews"]:
        review.update(bundle_fingerprint=assessment["bundle_fingerprint"],
                      profile_fingerprint=assessment["profile_fingerprint"],
                      assessment_fingerprint=assessment_fingerprint(assessment))


def unresolved(bundle, assessment, row=0, field="analysis.resolution"):
    context = bundle["note_contexts"][row]
    context["fields"][field] = dict(value=None, available=False, origin="unknown",
        confidence=None, evidence=["synthetic:ambiguous-exit"],
        unknown_reason="Two contextual exits remain defensible.", annotation_state="unresolved")
    assessment["views"][0]["reviewed_unknowns"].append(dict(
        row_id=context["id"], field=field, reason="No unique directed exit selected.",
        evidence=["synthetic:individually-reviewed-exits"]))


def test_opt_in_v2_complete_does_not_mutate_inputs():
    piece, bundle, profile, assessment = fixture()
    before = deepcopy((bundle, profile, assessment))
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["complete_for_declared_scope"] is True
    assert report["full_work_review_complete"] is True
    assert (bundle, profile, assessment) == before


def test_reviewed_unknown_is_masked_but_does_not_prevent_complete_review():
    piece, bundle, profile, assessment = fixture()
    unresolved(bundle, assessment)
    rebind(bundle, profile, assessment)
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["all_required_decisions_reviewed"] is True
    assert report["all_required_content_accounted"] is False
    assert report["full_work_review_complete"] is True
    assert report["complete_for_declared_scope"] is False
    assert bundle["note_contexts"][0]["fields"]["analysis.resolution"]["available"] is False


def test_unaddressed_unknown_and_entirely_unknown_field_cannot_pass_review_gate():
    piece, bundle, profile, assessment = fixture()
    unresolved(bundle, assessment)
    assessment["views"][0]["reviewed_unknowns"] = []
    rebind(bundle, profile, assessment)
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["full_work_review_complete"] is False
    unresolved(bundle, assessment, row=1)
    # Both unknowns are explicitly documented, but no target supports this field.
    assessment["views"][0]["reviewed_unknowns"].append(dict(row_id="c0",
        field="analysis.resolution", reason="Distinct ambiguity at the first context.",
        evidence=["synthetic:first-context"]))
    rebind(bundle, profile, assessment)
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["all_required_decisions_reviewed"] is True
    assert report["all_required_fields_supported"] is False
    assert report["full_work_review_complete"] is False


def padded_fixture():
    piece, bundle, profile, assessment = fixture()
    note = replace(piece.notes[0], onset_qn=RationalTime(1, 2), duration_qn=RationalTime(7, 2))
    conversion = ProvenanceRecord(provenance_id="synthetic:padding", kind="conversion",
        source="synthetic:adapter", record_id=None, uri=None, version="1",
        checksum_sha256=None, created_at=None, parents=(), details=(
            ("source_to_canonical_offset_qn", "1/2"),
            ("raw_repair_evidence_fingerprint", "e" * 64)))
    piece = replace(piece, notes=(note,), provenance=piece.provenance + (conversion,))
    bundle["note_contexts"][0]["start_qn"] = rational(Fraction(1, 2))
    for row in bundle["harmonies"] + bundle["objects"]:
        row["start_qn"] = rational(Fraction(1, 2))
    assessment["work"]["musical_scope"]["start_qn"] = rational(Fraction(1, 2))
    assessment["work"]["non_musical_intervals"] = [dict(
        start_qn=rational(0), end_qn=rational(Fraction(1, 2)),
        kind="adapter_structural_padding", canonical_sha256="a" * 64,
        provenance_id="synthetic:padding", adapter_source="synthetic:adapter", adapter_version="1",
        source_binding=dict(sha256="d" * 64, size_bytes=12, path="synthetic-source.tsv"),
        repair_evidence_fingerprint="e" * 64,
        evidence=["synthetic:leading-partial-measure-proof"])]
    for layer in assessment["views"][0]["layers"].values():
        layer["reviewed_intervals"] = [deepcopy(assessment["work"]["musical_scope"])]
    rebind(bundle, profile, assessment)
    return piece, bundle, profile, assessment


def test_proven_prefix_changes_accounting_not_canonical_coordinates():
    piece, bundle, profile, assessment = padded_fixture()
    before = deepcopy(bundle)
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["complete_for_declared_scope"] is True
    coverage = report["views"]["interpretation:a"]["layers"]["harmony"]["coverage"]
    assert coverage["expected_duration_qn"] == rational(Fraction(7, 2))
    assert coverage["excluded_non_musical_duration_qn"] == rational(Fraction(1, 2))
    assert bundle == before


@pytest.mark.parametrize("mutation", [
    lambda p, b, a: a["work"].update(non_musical_intervals=[]),
    lambda p, b, a: a["work"]["non_musical_intervals"][0].update(evidence=[]),
    lambda p, b, a: a["work"]["non_musical_intervals"][0].update(canonical_sha256="f" * 64),
    lambda p, b, a: a["work"]["non_musical_intervals"][0].update(end_qn=rational(1)),
    lambda p, b, a: a["work"]["non_musical_intervals"][0].update(start_qn=rational(Fraction(1, 4))),
    lambda p, b, a: a["work"]["non_musical_intervals"][0].update(repair_evidence_fingerprint="f" * 64),
])
def test_prefix_requires_exact_bound_provenance_not_first_note_silence(mutation):
    piece, bundle, profile, assessment = padded_fixture()
    mutation(piece, bundle, assessment)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


@pytest.mark.parametrize("point", [False, True])
def test_prefix_cannot_hide_sounding_or_grace_event(point):
    piece, bundle, profile, assessment = padded_fixture()
    extra = replace(piece.notes[0], note_id="note:excluded", onset_qn=RationalTime(1, 4),
                    duration_qn=RationalTime(0 if point else 1), is_grace=point)
    piece = replace(piece, notes=piece.notes + (extra,))
    row = deepcopy(bundle["note_contexts"][0])
    row.update(id="context:excluded", note_id=extra.note_id, harmony_id=None,
               start_qn=rational(Fraction(1, 4)), end_qn=rational(Fraction(1, 4) + (0 if point else 1)))
    bundle["note_contexts"].append(row)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="musical scope|padding"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def dispositions(assessment, end=4, activity="unpitched_events_only", note_ids=None):
    assessment["views"][0]["layer_dispositions"] = [dict(
        layer=layer, start_qn=rational(0), end_qn=rational(end),
        status="not_applicable", activity=activity, note_ids=note_ids or [],
        reason="No active pitched events or asserted conceptual harmonic continuation here.",
        evidence=["synthetic:inspected-activity-and-applicability"])
        for layer in ("harmony", "harmonic_processes")]


def percussion_fixture():
    piece, bundle, profile, assessment = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], is_percussion=True),))
    bundle["harmonies"] = []
    bundle["objects"] = [o for o in bundle["objects"] if o["object_type"] != "harmonic_process"]
    for row in bundle["note_contexts"]:
        row["harmony_id"] = None
        for name in ("analysis.chord_membership", "analysis.key_degree", "analysis.chord_degree"):
            row["fields"][name] = dict(value=None, available=False, origin="unknown",
                confidence=None, evidence=["synthetic:unpitched-instrument"],
                unknown_reason="Unpitched percussion", annotation_state="not_applicable",
                applicability_reason="This event is an unpitched percussion strike.")
    dispositions(assessment, note_ids=["note:n0"])
    rebind(bundle, profile, assessment)
    return piece, bundle, profile, assessment


def test_percussion_full_source_review_without_fabricated_harmony():
    piece, bundle, profile, assessment = percussion_fixture()
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["full_work_verified"] is True
    assert report["has_pitched_events"] is False
    assert report["full_work_review_complete"] is True
    assert report["complete_for_declared_scope"] is True
    assert not bundle["harmonies"]
    harmonic = report["views"]["interpretation:a"]["layers"]["harmony"]
    assert harmonic["coverage"]["not_applicable_duration_qn"] == rational(4)
    assert harmonic["rows"] == 0


@pytest.mark.parametrize("fault", ["pitched", "wrong_activity", "wrong_notes", "missing_reason", "overlap"])
def test_harmonic_disposition_rejects_contradictions(fault):
    piece, bundle, profile, assessment = percussion_fixture()
    d = assessment["views"][0]["layer_dispositions"][0]
    if fault == "pitched":
        piece = replace(piece, notes=(replace(piece.notes[0], is_percussion=False),))
    elif fault == "wrong_activity":
        d["activity"] = "silence"
    elif fault == "wrong_notes":
        d["note_ids"] = []
    elif fault == "missing_reason":
        d["reason"] = ""
    else:
        assessment["views"][0]["layer_dispositions"].append(deepcopy(d))
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_real_silence_remains_in_scope_and_can_have_conceptual_harmony():
    piece, bundle, profile, assessment = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], onset_qn=RationalTime(2),
        duration_qn=RationalTime(2)),))
    bundle["note_contexts"][0].update(start_qn=rational(2), end_qn=rational(3))
    bundle["note_contexts"][1]["start_qn"] = rational(3)
    rebind(bundle, profile, assessment)
    retained = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert retained["complete_for_declared_scope"] is True
    assert retained["musical_scope"]["start_qn"] == rational(0)
    bundle["harmonies"][0]["start_qn"] = rational(2)
    next(o for o in bundle["objects"] if o["object_type"] == "harmonic_process")["start_qn"] = rational(2)
    dispositions(assessment, end=2, activity="silence")
    rebind(bundle, profile, assessment)
    explicit = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert explicit["complete_for_declared_scope"] is True
    assert explicit["musical_scope"]["start_qn"] == rational(0)


@pytest.mark.parametrize("fault", ["profile", "scope", "bundle", "decisions"])
def test_review_receipt_cannot_be_reused_after_bound_basis_changes(fault):
    piece, bundle, profile, assessment = fixture()
    if fault == "profile":
        profile["profile_id"] = "synthetic-extended@2"
        assessment["profile_fingerprint"] = annotation_fingerprint(profile)
    elif fault == "scope":
        assessment["work"]["musical_scope"]["start_qn"] = rational(1)
    elif fault == "bundle":
        bundle["note_contexts"][0]["fields"]["analysis.temporal_role"]["value"] = "neighbor"
        assessment["bundle_fingerprint"] = annotation_fingerprint(bundle)
    else:
        assessment["views"][0]["layers"]["notes"]["evidence"] = ["synthetic:changed-review"]
    with pytest.raises(EnrichmentError, match="fingerprint"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_incomplete_views_do_not_fill_each_others_unknowns():
    piece, bundle, profile, assessment = fixture()
    for collection in ("note_contexts", "harmonies", "objects"):
        for row in list(bundle[collection]):
            copied = deepcopy(row)
            copied["id"] += ":b"
            copied["view_id"] = "interpretation:b"
            if copied.get("harmony_id"):
                copied["harmony_id"] += ":b"
            bundle[collection].append(copied)
    view = deepcopy(assessment["views"][0])
    view.update(view_id="interpretation:b")
    assessment["views"].append(view)
    review = deepcopy(assessment["independent_reviews"][0])
    review["view_id"] = "interpretation:b"
    assessment["independent_reviews"].append(review)
    bundle["note_contexts"][0]["fields"].pop("analysis.temporal_role")
    bundle["note_contexts"][3]["fields"].pop("analysis.temporal_role")
    rebind(bundle, profile, assessment)
    report = validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)
    assert report["full_work_review_complete"] is False
    assert report["all_required_content_accounted"] is False


@pytest.mark.parametrize("fault", ["receipt_evidence", "receipt_reason", "field_evidence",
                                  "wrong_view_address", "duplicate", "unreviewed"])
def test_unknown_review_is_exact_and_evidenced(fault):
    piece, bundle, profile, assessment = fixture()
    unresolved(bundle, assessment)
    receipt = assessment["views"][0]["reviewed_unknowns"][0]
    field = bundle["note_contexts"][0]["fields"]["analysis.resolution"]
    if fault == "receipt_evidence":
        receipt["evidence"] = []
    elif fault == "receipt_reason":
        receipt["reason"] = ""
    elif fault == "field_evidence":
        field["evidence"] = []
    elif fault == "wrong_view_address":
        receipt["row_id"] = "other-view:c0"
    elif fault == "duplicate":
        assessment["views"][0]["reviewed_unknowns"].append(deepcopy(receipt))
    else:
        field["annotation_state"] = "unreviewed"
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_unknown_encoding_stays_masked_and_review_does_not_create_target():
    from music_critic.tasks.event_theory_enrichment import build_vocabulary, encode_enrichment
    piece, bundle, profile, assessment = fixture()
    unresolved(bundle, assessment)
    rebind(bundle, profile, assessment)
    before = deepcopy(bundle)
    validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
                                       require_review_complete=True)
    encoded = encode_enrichment(piece, bundle, build_vocabulary([bundle]))
    assert encoded["note_contexts"]["masks"]["analysis.resolution"] == [False, True]
    assert encoded["note_contexts"]["values"]["analysis.resolution"][0] == -1
    assert bundle == before
    with pytest.raises(EnrichmentError, match="strict"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
                                           require_complete=True)


def test_no_independent_acceptance_or_contested_review_never_passes():
    piece, bundle, profile, assessment = fixture()
    assessment["independent_reviews"][0]["result"] = "changes_requested"
    assert not validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)["full_work_review_complete"]
    assessment["independent_reviews"] = []
    assert not validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)["full_work_review_complete"]


def test_known_fields_without_evidence_do_not_establish_completion():
    piece, bundle, profile, assessment = fixture()
    bundle["note_contexts"][0]["fields"]["analysis.temporal_role"]["evidence"] = []
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="evidence"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


@pytest.mark.parametrize("collection", ["note_contexts", "harmonies"])
def test_same_view_overlapping_targets_are_rejected(collection):
    piece, bundle, profile, assessment = fixture()
    duplicate = deepcopy(bundle[collection][0])
    duplicate["id"] += ":contradiction"
    if collection == "note_contexts":
        duplicate["fields"]["analysis.chord_membership"]["value"] = False
    else:
        duplicate["fields"]["analysis.function"]["value"] = "different_function"
    bundle[collection].append(duplicate)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="overlapping.*same view"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
                                           require_complete=True)


@pytest.mark.parametrize("collection", ["note_contexts", "harmonies"])
def test_partial_overlap_is_rejected_without_requiring_identical_bounds(collection):
    piece, bundle, profile, assessment = fixture()
    if collection == "note_contexts":
        bundle[collection][1]["start_qn"] = rational(1)
    else:
        duplicate = deepcopy(bundle[collection][0])
        duplicate.update(id="overlapping:harmony", start_qn=rational(1))
        bundle[collection].append(duplicate)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="overlapping.*same view"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


@pytest.mark.parametrize("at", [0, 1, 2, 3])
def test_point_context_cannot_duplicate_an_active_note_context(at):
    piece, bundle, profile, assessment = fixture()
    point = deepcopy(bundle["note_contexts"][0])
    point.update(id="point:duplicate", start_qn=rational(at), end_qn=rational(at),
                 applicability="source_note_annotation")
    bundle["note_contexts"].append(point)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="overlapping note_contexts in same view"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_exclusive_right_boundary_point_without_outgoing_context_is_allowed():
    piece, bundle, profile, assessment = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], duration_qn=RationalTime(3)),))
    bundle["note_contexts"][1]["end_qn"] = rational(3)
    point = deepcopy(bundle["note_contexts"][1])
    point.update(id="point:end", start_qn=rational(3),
                 applicability="source_note_annotation")
    bundle["note_contexts"].append(point)
    rebind(bundle, profile, assessment)
    assert validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
        require_complete=True)["complete_for_declared_scope"]


def grace_boundary_fixture():
    piece, bundle, profile, assessment = fixture()
    outgoing = deepcopy(bundle["harmonies"][0])
    outgoing.update(id="harmony:outgoing", start_qn=rational(2))
    bundle["harmonies"][0]["end_qn"] = rational(2)
    bundle["harmonies"].append(outgoing)
    bundle["note_contexts"][1]["harmony_id"] = outgoing["id"]
    grace = replace(piece.notes[0], note_id="note:grace", onset_qn=RationalTime(2),
                    duration_qn=RationalTime(0), is_grace=True)
    piece = replace(piece, notes=piece.notes + (grace,))
    context = deepcopy(bundle["note_contexts"][1])
    context.update(id="context:grace", note_id=grace.note_id, end_qn=rational(2))
    bundle["note_contexts"].append(context)
    rebind(bundle, profile, assessment)
    return piece, bundle, profile, assessment


def test_adjacent_harmonies_and_grace_at_outgoing_boundary_are_allowed():
    piece, bundle, profile, assessment = grace_boundary_fixture()
    assert validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
        require_complete=True)["complete_for_declared_scope"]
    bundle["note_contexts"][-1]["harmony_id"] = "harmony:0"
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="context outside harmony"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_duplicate_grace_points_are_rejected_even_with_null_harmony_binding():
    piece, bundle, profile, assessment = grace_boundary_fixture()
    duplicate = deepcopy(bundle["note_contexts"][-1])
    duplicate.update(id="context:duplicate-grace", harmony_id=None)
    bundle["note_contexts"].append(duplicate)
    rebind(bundle, profile, assessment)
    with pytest.raises(EnrichmentError, match="overlapping note_contexts in same view"):
        validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile)


def test_overlapping_objects_and_distinct_views_remain_independent_and_allowed():
    piece, bundle, profile, assessment = fixture()
    for obj in list(bundle["objects"]):
        duplicate = deepcopy(obj)
        duplicate["id"] += ":overlapping-object"
        bundle["objects"].append(duplicate)
    for collection in ("note_contexts", "harmonies", "objects"):
        for row in list(bundle[collection]):
            duplicate = deepcopy(row)
            duplicate["id"] += ":alternative-view"
            duplicate["view_id"] = "interpretation:b"
            if duplicate.get("harmony_id"):
                duplicate["harmony_id"] += ":alternative-view"
            bundle[collection].append(duplicate)
    view = deepcopy(assessment["views"][0])
    view["view_id"] = "interpretation:b"
    assessment["views"].append(view)
    review = deepcopy(assessment["independent_reviews"][0])
    review["view_id"] = "interpretation:b"
    assessment["independent_reviews"].append(review)
    rebind(bundle, profile, assessment)
    assert validate_annotation_completeness_v2(piece, bundle, assessment, profile=profile,
        require_complete=True)["complete_for_declared_scope"]
