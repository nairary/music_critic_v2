"""Opt-in T08 review accounting over immutable EventTheoryEnrichment@1.

This module performs no I/O and does not authenticate external evidence bytes.
Callers must authenticate the admitted canonical piece and retained receipts.
Review completion preserves unresolved masks; it is neither target availability
nor musical correctness, and never establishes corpus400 quota eligibility.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Mapping

from music_critic.tasks import whole_work_annotation as v1
from music_critic.tasks.event_theory_enrichment import fraction, require


PROFILE_SCHEMA = "WholeWorkAnnotationProfile@2"
ASSESSMENT_SCHEMA = "WholeWorkAnnotationAssessment@2"
REPORT_SCHEMA = "WholeWorkAnnotationReport@2"


def default_whole_work_profile_v2() -> dict:
    """Separate, opt-in policy; v1 defaults and required field lists are intact."""
    profile = v1.default_whole_work_profile()
    profile.update(schema=PROFILE_SCHEMA, profile_id="corpus400@2",
                   require_support_for_each_required_field=True)
    return profile


def assessment_fingerprint(assessment: Mapping) -> str:
    """Bind the exact review basis, excluding independent reviews themselves."""
    return v1.annotation_fingerprint({key: value for key, value in assessment.items()
                                      if key != "independent_reviews"})


def _profile(profile):
    require(profile.get("schema") == PROFILE_SCHEMA
            and v1._text(profile.get("profile_id")), "invalid v2 profile")
    require(profile.get("require_support_for_each_required_field") is True,
            "v2 field support policy cannot be weakened")
    compatibility = deepcopy(profile)
    compatibility.update(schema=v1.PROFILE_SCHEMA)
    v1._validate_profile(compatibility)
    return compatibility


def _scope(piece, bundle, assessment):
    duration = piece.duration_qn.to_fraction()
    work = assessment["work"]
    scope = work.get("musical_scope")
    require(isinstance(scope, dict), "explicit musical scope required")
    left, right = v1._interval(scope, duration)
    require(left < right and right == duration, "musical scope must end at canonical duration")
    exclusions = work.get("non_musical_intervals", [])
    require(isinstance(exclusions, list), "non-musical intervals must be a list")
    require(len(exclusions) == (1 if left else 0), "exact prefix padding complement required")
    if left:
        proof = exclusions[0]
        require(isinstance(proof, dict) and v1._interval(proof, duration) == (0, left)
                and proof.get("kind") == "adapter_structural_padding",
                "only exact leading structural padding may be excluded")
        require(proof.get("canonical_sha256") == bundle["canonical_piece"]["sha256"]
                and v1._evidence(proof.get("evidence")), "padding canonical binding/evidence required")
        source = proof.get("source_binding", {})
        require(isinstance(source, dict) and v1._digest(source.get("sha256"))
                and type(source.get("size_bytes")) is int and source["size_bytes"] > 0
                and v1._text(source.get("path")), "padding source byte binding required")
        records = [p for p in piece.provenance if p.provenance_id == proof.get("provenance_id")]
        require(len(records) == 1, "padding conversion provenance required")
        record = records[0]
        details = dict(record.details)
        require(record.kind == "conversion" and v1._text(record.version)
                and record.version == proof.get("adapter_version")
                and record.source == proof.get("adapter_source"), "padding adapter identity mismatch")
        require(details.get("source_to_canonical_offset_qn") == f"{left.numerator}/{left.denominator}"
                and v1._digest(proof.get("repair_evidence_fingerprint"))
                and proof["repair_evidence_fingerprint"] == details.get("raw_repair_evidence_fingerprint"),
                "padding transform evidence mismatch")
    for note in piece.notes:
        onset = note.onset_qn.to_fraction()
        require(left <= onset and onset + note.duration_qn.to_fraction() <= right,
                "canonical event outside musical scope; padding cannot hide notes or grace events")
    view_ids = {view["view_id"] for view in assessment["views"]}
    for collection in ("note_contexts", "harmonies", "objects", "relations"):
        for row in bundle.get(collection, []):
            require(v1._view(row) in view_ids, "undeclared analytical view")
            a, b = v1._interval(row, duration)
            require(left <= a <= b <= right, "analytical row outside musical scope")
    return left, right


def _rows(bundle, view_id, layer):
    return [row for row in bundle.get(layer["scope"], [])
            if v1._view(row) == view_id and (layer["scope"] != "objects"
            or v1._object_type(row) == layer["object_type"])]


def _exclusive_contexts(bundle):
    """One global harmonic frame and one decision context per note in a view.

    Positive spans are half-open. A point conflicts with another point at the
    same time or a span containing that time, including its left boundary.
    The exclusive right boundary alone is not an overlap. Analytical objects
    may overlap, and alternative views remain independent.
    """
    for collection in ("harmonies", "note_contexts"):
        groups = defaultdict(list)
        for row in bundle.get(collection, []):
            key = (v1._view(row), row.get("note_id") if collection == "note_contexts" else None)
            groups[key].append((fraction(row["start_qn"]), fraction(row["end_qn"])))
        for spans in groups.values():
            ordered = sorted(spans)
            for (a, b), (c, _) in zip(ordered, ordered[1:]):
                require(b <= c and not (a == b == c),
                        f"overlapping {collection} in same view")


def _unknown_reviews(bundle, view):
    rows = {row["id"]: row for scope in ("note_contexts", "harmonies", "objects")
            for row in bundle.get(scope, []) if v1._view(row) == view["view_id"]}
    receipts = view.get("reviewed_unknowns", [])
    require(isinstance(receipts, list), "reviewed unknowns must be addressed records")
    reviewed = set()
    for receipt in receipts:
        require(isinstance(receipt, dict), "unknown review receipt must be an object")
        key = (receipt.get("row_id"), receipt.get("field"))
        require(all(v1._text(item) for item in key) and key not in reviewed,
                "duplicate or invalid unknown address")
        row = rows.get(key[0])
        field = row.get("fields", {}).get(key[1]) if row else None
        require(field is not None and v1._decision(field) in {"unresolved", "conflicting"},
                "unknown review must address an explicit same-view unresolved field")
        require(v1._text(field.get("unknown_reason")) and v1._evidence(field.get("evidence"))
                and v1._text(receipt.get("reason")) and v1._evidence(receipt.get("evidence")),
                "addressed unknown requires reason and evidence")
        reviewed.add(key)
    return reviewed


def _dispositions(piece, bundle, view, profile, scope):
    """Verify literal activity plus an explicit author's applicability decision.

    Silence can still have a conceptual harmony; no disposition is generated
    from geometry. Unpitched-only spans may contain rests between drum events.
    A pitched point grace or a note sustained from outside the span counts.
    """
    rows = view.get("layer_dispositions", [])
    require(isinstance(rows, list), "layer dispositions must be a list")
    result = defaultdict(list)
    for row in rows:
        require(isinstance(row, dict) and row.get("layer") in {"harmony", "harmonic_processes"},
                "disposition is restricted to harmonic layers")
        require(row.get("status") == "not_applicable" and v1._text(row.get("reason"))
                and v1._evidence(row.get("evidence")), "harmonic disposition needs reason and evidence")
        left, right = v1._interval(row, piece.duration_qn.to_fraction())
        require(scope[0] <= left < right <= scope[1], "disposition outside musical scope")
        active = []
        for note in piece.notes:
            start = note.onset_qn.to_fraction()
            end = start + note.duration_qn.to_fraction()
            if (start == end and left <= start < right) or (start < right and end > left):
                active.append(note)
        require(not any(not note.is_percussion for note in active),
                "pitched event contradicts harmonic disposition")
        ids = row.get("note_ids")
        require(isinstance(ids, list) and all(v1._text(item) for item in ids)
                and len(ids) == len(set(ids)) and set(ids) == {note.note_id for note in active},
                "disposition must address every active canonical event")
        activity = row.get("activity")
        require(activity in {"silence", "unpitched_events_only"}, "unknown literal activity state")
        require((activity == "silence" and not active)
                or (activity == "unpitched_events_only" and bool(active)),
                "literal activity contradicts canonical events")
        layer = row["layer"]
        for analytical in _rows(bundle, view["view_id"], profile["layers"][layer]):
            a, b = fraction(analytical["start_qn"]), fraction(analytical["end_qn"])
            require(not (a < right and b > left), "disposition contradicts authored harmonic context")
        result[layer].append((left, right))
    for spans in result.values():
        ordered = sorted(spans)
        require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])),
                "overlapping harmonic dispositions")
    return result


def _layer_report(piece, bundle, view, name, policy, previous, scope, unknowns, dispositions):
    report = deepcopy(previous)
    rows = _rows(bundle, view["view_id"], policy)
    left, right = scope
    review = view.get("layers", {}).get(name, {})
    reviewed = [v1._interval(row, piece.duration_qn.to_fraction())
                for row in review.get("reviewed_intervals", [])]
    full_review = v1._covered(reviewed, left, right)
    coverage = report["coverage"]
    coverage.update(review_extent_qn=v1._rational(right-left),
        reviewed_duration_qn=v1._rational(v1._duration([(max(left, a), min(right, b))
            for a, b in reviewed if max(left, a) <= min(right, b)])),
        excluded_non_musical_duration_qn=v1._rational(left))
    issues = [issue for issue in report["issues"]
              if issue not in {"analytical_geometry_gap", "review_interval_gap"}]
    inactive = dispositions.get(name, [])
    fully_inactive = v1._covered(inactive, left, right)
    if fully_inactive:
        issues = [issue for issue in issues if issue != "missing_layer_rows"]
    absence = report["review_status"] in {"not_applicable", "reviewed_absence"}
    structural = report["structural_coverage_complete"]
    if policy["coverage"] == "interval" and not absence:
        spans = [(fraction(row["start_qn"]), fraction(row["end_qn"])) for row in rows]
        structural = v1._covered(spans + inactive, left, right)
        covered = v1._duration(spans + inactive)
        coverage.update(expected_duration_qn=v1._rational(right-left),
                        covered_duration_qn=v1._rational(covered),
                        annotated_duration_qn=v1._rational(v1._duration(spans)),
                        not_applicable_duration_qn=v1._rational(v1._duration(inactive)),
                        unrepresented_duration_qn=v1._rational(right-left-covered))
    if not structural and not absence:
        issues.append("analytical_geometry_gap")
    if not full_review:
        issues.append("review_interval_gap")
    report.update(reviewed_full_interval=full_review, structural_coverage_complete=structural,
                  issues=issues, content_accounted=not issues)
    decisions_reviewed = True
    fields_supported = True
    for field_name in policy["required_fields"]:
        support = fully_inactive
        for row in rows:
            field = row.get("fields", {}).get(field_name)
            state = v1._decision(field) if field is not None else "missing"
            if state in {"known", "not_applicable"}:
                support = True
                decisions_reviewed &= v1._evidence(field.get("evidence"))
            elif state in {"unresolved", "conflicting"}:
                decisions_reviewed &= (row["id"], field_name) in unknowns
            else:
                decisions_reviewed = False
        fields_supported &= support
    if report["review_status"] in {"not_applicable", "reviewed_absence"}:
        decisions_reviewed = fields_supported = True
    report.update(all_required_decisions_reviewed=bool(decisions_reviewed),
                  all_required_fields_supported=bool(fields_supported))
    review_issues = [issue for issue in report["issues"]
                     if not issue.startswith("unaccounted_field:")]
    if not decisions_reviewed:
        review_issues.append("unreviewed_required_decisions")
    if not fields_supported:
        review_issues.append("required_field_without_supported_targets")
    report["review_completion_issues"] = review_issues
    return report


def validate_annotation_completeness_v2(piece, bundle: Mapping, assessment: Mapping, *,
                                      profile=None, require_complete=False,
                                      require_review_complete=False) -> dict:
    """Validate v2 receipts and compute separate strict/review completion gates.

    Explicit reviewed unknowns need one exact row/field address each. At least
    one known or legitimately NA decision must support every applicable required
    field: an all-unknown field cannot create a complete reviewed annotation.
    This support floor is not a claim of dense target availability; exact v1
    field counts and durations remain visible and masks are never modified.
    """
    profile = default_whole_work_profile_v2() if profile is None else profile
    compatibility_profile = _profile(profile)
    require(assessment.get("schema") == ASSESSMENT_SCHEMA, "v2 assessment schema mismatch")
    require(assessment.get("profile_fingerprint") == v1.annotation_fingerprint(profile),
            "assessment v2 profile fingerprint mismatch")
    basis = assessment_fingerprint(assessment)
    reviews = assessment.get("independent_reviews", [])
    require(isinstance(reviews, list), "independent reviews must be a list")
    for review in reviews:
        require(isinstance(review, dict), "independent review must be an object")
        require(review.get("profile_fingerprint") == assessment["profile_fingerprint"],
                "independent review profile fingerprint mismatch")
        require(review.get("assessment_fingerprint") == basis,
                "independent review assessment fingerprint mismatch")
    compatibility = deepcopy(assessment)
    compatibility.update(schema=v1.ASSESSMENT_SCHEMA,
                         profile_fingerprint=v1.annotation_fingerprint(compatibility_profile))
    report = v1.validate_annotation_completeness(piece, bundle, compatibility,
                                               profile=compatibility_profile)
    scope = _scope(piece, bundle, assessment)
    _exclusive_contexts(bundle)
    for view in assessment["views"]:
        unknowns = _unknown_reviews(bundle, view)
        dispositions = _dispositions(piece, bundle, view, profile, scope)
        for name, policy in profile["layers"].items():
            previous = report["views"][view["view_id"]]["layers"][name]
            report["views"][view["view_id"]]["layers"][name] = _layer_report(
                piece, bundle, view, name, policy, previous, scope, unknowns, dispositions)
    layers = [layer for view in report["views"].values() for layer in view["layers"].values()]
    report["all_layers_reviewed"] = all(layer["reviewed_full_interval"] and
        layer["review_status"] in {"reviewed", "not_applicable", "reviewed_absence"} for layer in layers)
    report["all_required_content_accounted"] = all(layer["content_accounted"] for layer in layers)
    source_complete = assessment["work"]["source_extent"] == "full_work"
    review_complete = source_complete and report["independently_reviewed"] and all(
        not layer["review_completion_issues"] for layer in layers)
    report.update(schema=REPORT_SCHEMA, profile_fingerprint=assessment["profile_fingerprint"],
        assessment_fingerprint=basis, full_work_verified=source_complete,
        has_pitched_events=any(not note.is_percussion for note in piece.notes),
        all_required_decisions_reviewed=all(layer["all_required_decisions_reviewed"] for layer in layers),
        all_required_fields_supported=all(layer["all_required_fields_supported"] for layer in layers),
        full_work_review_complete=bool(review_complete),
        complete_for_declared_scope=bool(review_complete and report["all_required_content_accounted"]),
        musical_scope=deepcopy(assessment["work"]["musical_scope"]),
        claim_boundary="Review accounting with explicit masks; not musical accuracy or quota eligibility.")
    if require_complete:
        require(report["complete_for_declared_scope"], "strict whole-work annotation is incomplete")
    if require_review_complete:
        require(review_complete, "whole-work review is incomplete")
    return report
