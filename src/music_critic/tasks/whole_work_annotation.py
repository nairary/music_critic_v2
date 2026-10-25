"""Strict, opt-in whole-work accounting above the unchanged T07 sidecar.

Coverage and review receipts establish accounting, never musical correctness.
No files are opened here: callers authenticate/admit canonical inputs first.
Fingerprints below hash canonical JSON without a trailing newline, not file
bytes. Full-work and genre evidence are retained attestations, not inferred
from MIDI duration, track names, or the presence of annotation rows.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from typing import Iterable, Mapping

from music_critic.tasks.event_theory_enrichment import (
    SCHEMA, MODEL_ANNOTATOR_SETTINGS, _field, canonical, fraction, require, validate_enrichment,
)


PROFILE_SCHEMA = "WholeWorkAnnotationProfile@1"
ASSESSMENT_SCHEMA = "WholeWorkAnnotationAssessment@1"
REPORT_SCHEMA = "WholeWorkAnnotationReport@1"
OBJECT_VOCABULARY_SCHEMA = "EventTheoryObjectVocabulary@1"
OBJECT_TARGETS_SCHEMA = "EventTheoryObjectTargets@1"
GENRES = frozenset({"classical", "jazz", "rock", "pop"})
STATES = ("known", "not_applicable", "unreviewed", "unresolved", "conflicting")


def _layer(scope, fields, *, object_type=None, coverage="interval",
           allow_not_applicable=False, allow_reviewed_absence=False, not_applicable_fields=()):
    return dict(scope=scope, required_fields=fields, object_type=object_type,
                coverage=coverage, allow_not_applicable=allow_not_applicable,
                allow_reviewed_absence=allow_reviewed_absence,
                not_applicable_fields=list(not_applicable_fields))


_LAYERS = {
    "notes": _layer("note_contexts", [
        "analysis.chord_membership", "analysis.key_degree", "analysis.chord_degree",
        "analysis.voice", "analysis.temporal_role", "analysis.preparation", "analysis.resolution",
    ], coverage="notes", not_applicable_fields=("analysis.key_degree", "analysis.chord_degree",
        "analysis.chord_membership", "analysis.preparation", "analysis.resolution")),
    "harmony": _layer("harmonies", [
        "analysis.sonority", "analysis.extensions", "analysis.inversion",
        "analysis.key", "analysis.roman", "analysis.function",
    ], not_applicable_fields=("analysis.key", "analysis.roman", "analysis.function")),
    "harmonic_processes": _layer("objects", ["analysis.process", "analysis.function"],
                                 object_type="harmonic_process", not_applicable_fields=("analysis.function",)),
    "motifs": _layer("objects", ["analysis.identity", "analysis.transformations"],
                     object_type="motif", coverage="reviewed_interval", allow_reviewed_absence=True),
    "phrases": _layer("objects", ["analysis.function", "analysis.question_answer"], object_type="phrase",
                      not_applicable_fields=("analysis.question_answer",)),
    "form": _layer("objects", ["analysis.section", "analysis.repetition_contrast"], object_type="section"),
    "ensemble": _layer("objects", ["analysis.parts_voices", "analysis.texture", "analysis.interactions"],
                        object_type="ensemble", allow_not_applicable=True),
}


def annotation_fingerprint(value: Mapping) -> str:
    """Stable semantic identity of a profile, bundle, or vocabulary document."""
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def default_whole_work_profile() -> dict:
    """Return an independent copy; profiles may add but not remove requirements."""
    return dict(schema=PROFILE_SCHEMA, profile_id="corpus400@1", layers=deepcopy(_LAYERS))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _evidence(value):
    return isinstance(value, list) and bool(value) and all(_text(item) for item in value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _rational(value):
    value = Fraction(value)
    return dict(num=value.numerator, den=value.denominator)


def _interval(row, duration):
    left, right = fraction(row["start_qn"]), fraction(row["end_qn"])
    require(0 <= left <= right <= duration, "review interval outside canonical piece")
    return left, right


def _union(intervals):
    result = []
    for left, right in sorted(intervals):
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(right, result[-1][1]))
        else:
            result.append((left, right))
    return result


def _covered(intervals, left, right):
    return any(start <= left <= right <= end for start, end in _union(intervals))


def _duration(intervals):
    return sum((right - left for left, right in _union(intervals)), Fraction(0))


def _view(row):
    return row.get("view_id", row.get("harmony_view"))


def _object_type(row):
    return row.get("object_type", row.get("type", row.get("kind")))


def _validate_profile(profile):
    require(profile.get("schema") == PROFILE_SCHEMA and _text(profile.get("profile_id")), "invalid profile")
    layers = profile.get("layers")
    require(isinstance(layers, dict) and set(layers) == set(_LAYERS), "all required analytical layers must remain")
    for name, baseline in _LAYERS.items():
        layer = layers[name]
        require(isinstance(layer, dict), "layer profile must be an object")
        fields = layer.get("required_fields")
        require(isinstance(fields, list) and fields and all(_text(f) for f in fields)
                and len(set(fields)) == len(fields)
                and set(baseline["required_fields"]) <= set(fields), "required fields cannot be removed")
        for key in ("scope", "object_type", "coverage", "allow_not_applicable", "allow_reviewed_absence", "not_applicable_fields"):
            require(layer.get(key) == baseline[key], "layer coverage policy cannot be weakened")


def _decision(field):
    """T07 availability remains authoritative; state adds review/applicability."""
    _field(field)
    state = field.get("annotation_state", "known" if field["available"] else "unreviewed")
    require(state in STATES, "unknown annotation state")
    require(field["available"] == (state == "known"), "annotation state contradicts availability")
    if state == "not_applicable":
        require(_text(field.get("applicability_reason")) and _evidence(field.get("evidence")),
                "not-applicable field requires musical reason and evidence")
    empty = isinstance(field["value"], (list, dict)) and not field["value"]
    empty = empty or (isinstance(field["value"], str) and not field["value"].strip())
    if state == "known" and empty:
        require(_text(field.get("decision_reason")), "known empty value requires explicit decision reason")
    return state


def _validate_objects(bundle):
    for obj in bundle.get("objects", []):
        require(_text(_object_type(obj)), "analysis object type required")
        fields = obj.get("fields", {})
        require(isinstance(fields, dict), "object fields must be an object")
        for name, field in fields.items():
            require(_text(name), "object field name required")
            _decision(field)


def _validate_author(author):
    require(isinstance(author, dict), "view author receipt required")
    require(author.get("kind") in {"model", "human", "source"}
            and _text(author.get("author_id")) and _text(author.get("annotation_id"))
            and _evidence(author.get("evidence")), "view author identity/evidence required")
    require(_digest(author.get("input_sha256")) and _digest(author.get("output_sha256")),
            "view author input/output hashes required")
    if author["kind"] == "model":
        require((author.get("model_requested"), author.get("reasoning_effort_requested")) in MODEL_ANNOTATOR_SETTINGS
                and author.get("human_gold") is False, "model annotations require explicit supported model/effort weak provenance")
        require("actual_backend_version" in author
                and (author["actual_backend_version"] is None or _text(author["actual_backend_version"])),
                "actual backend must be explicit, including unknown null")


def _semantic_indexes(piece, bundle):
    participants = {note.note_id: ("note", None, note.onset_qn.to_fraction()) for note in piece.notes}
    for scope in ("harmonies", "note_contexts", "objects"):
        participants.update({row["id"]: (scope, _view(row), fraction(row["start_qn"]))
                             for row in bundle.get(scope, [])})
    relations = {row["id"]: row for row in bundle.get("relations", [])}
    directed = defaultdict(set)
    for relation in relations.values():
        if _decision(relation) == "known" and relation["value"] is True:
            for direction in ("source_ids", "target_ids"):
                for participant in relation[direction]:
                    directed[(_view(relation), relation["type"], direction, participant)].add(relation["id"])
    return participants, relations, directed


def _directed_note_decision(row, name, value, indexes):
    """Preparation is incoming; resolution is outgoing; absence is explicit."""
    require(isinstance(value, dict) and value.get("status") in {"none", "linked"},
            name + " requires a directed relation decision")
    references = value.get("relation_ids")
    require(isinstance(references, list) and all(_text(ref) for ref in references)
            and len(references) == len(set(references)), "directed relation IDs must be unique")
    participants, relations, directed = indexes
    relation_type, direction = (("prepares", "target_ids") if name == "analysis.preparation"
                                else ("resolves_to", "source_ids"))
    view_id = _view(row)
    addressed = set().union(*(directed[(view_id, relation_type, direction, identity)]
                              for identity in (row["id"], row["note_id"])))
    if value["status"] == "none":
        require(not references and _text(value.get("reason")), "absent directed relation needs an explicit reason")
        require(not addressed, "absent decision contradicts a positive directed relation")
        return
    require(bool(references), "linked directed decision needs relation IDs")
    for reference in references:
        relation = relations.get(reference)
        require(relation is not None, "unknown directed relation ID")
        require(relation["available"] and relation["value"] is True
                and relation["type"] == relation_type and _view(relation) == view_id,
                "directed relation requires matching positive type/view")
        require(set(relation[direction]) & {row["id"], row["note_id"]}, "directed relation has wrong note role")
        require(not set(relation["source_ids"]) & set(relation["target_ids"]), "directed relation cannot link to itself")
        for identity in relation["source_ids"] + relation["target_ids"]:
            kind, participant_view, _ = participants[identity]
            require(kind in {"note", "note_contexts"} and (kind == "note" or participant_view == view_id),
                    "directed relation participant must be a note/context in this view")
        require(max(participants[identity][2] for identity in relation["source_ids"])
                <= min(participants[identity][2] for identity in relation["target_ids"]),
                "directed relation target precedes its preparation/tension")
    require(set(references) == addressed, "directed decision omits a positive relation")


def _semantic_values(row, required_fields, indexes):
    """Minimum shape, not a universal theory ontology or accuracy test.

    Categorical analyses are strings or objects with a nonempty label. Sonority
    retains its explicit tones, extensions remain a list, and note predicates
    remain boolean. Object relationship descriptions are lists; actual edges
    remain separate T07 relations and require independent semantic review.
    """
    list_fields = {"analysis.extensions", "analysis.transformations", "analysis.parts_voices",
                   "analysis.interactions", "analysis.repetition_contrast"}
    core_fields = {name for layer in _LAYERS.values() for name in layer["required_fields"]}
    category = lambda value: _text(value) or (isinstance(value, dict) and _text(value.get("label")))
    for name in required_fields:
        field = row.get("fields", {}).get(name)
        if field is None or _decision(field) != "known" or name not in core_fields:
            continue
        value = field["value"]
        if name in {"analysis.preparation", "analysis.resolution"}:
            _directed_note_decision(row, name, value, indexes)
        elif name == "analysis.chord_membership":
            require(type(value) is bool, "semantic chord membership must be boolean")
        elif name == "analysis.sonority":
            require(isinstance(value, dict) and isinstance(value.get("tones"), list) and value["tones"]
                    and all(_text(tone) or (type(tone) is int and 0 <= tone <= 127) for tone in value["tones"]),
                    "semantic sonority requires explicit nonempty tones")
        elif name in list_fields:
            require(isinstance(value, list) and all(category(item) for item in value),
                    "semantic list required for " + name)
        else:
            require(category(value), "semantic category required for " + name)


def _field_summary(rows, name, note_scope):
    counts = Counter({state: 0 for state in (*STATES, "missing")})
    spans = defaultdict(lambda: defaultdict(list))
    for row in rows:
        field = row.get("fields", {}).get(name)
        state = "missing" if field is None else _decision(field)
        counts[state] += 1
        group = row["note_id"] if note_scope else "interval"
        spans[state][group].append((fraction(row["start_qn"]), fraction(row["end_qn"])))
    durations = {state: _rational(sum((_duration(value) for value in spans[state].values()), Fraction(0)))
                 for state in (*STATES, "missing")}
    return dict(counts=dict(counts), duration_qn=durations,
                duration_unit="summed_note_time" if note_scope else "union_interval_time")


def _assess_layer(piece, bundle, view_id, name, profile, review, indexes):
    duration = piece.duration_qn.to_fraction()
    require(isinstance(review, dict), "layer review receipt must be an object")
    status = review.get("review_status", "not_reviewed")
    require(status in {"not_reviewed", "in_progress", "reviewed", "not_applicable", "reviewed_absence"},
            "invalid layer review status")
    intervals = review.get("reviewed_intervals", [])
    require(isinstance(intervals, list), "reviewed intervals must be a list")
    intervals = [_interval(row, duration) for row in intervals]
    reviewed_full = _covered(intervals, Fraction(0), duration)
    if status in {"reviewed", "not_applicable", "reviewed_absence"}:
        require(_evidence(review.get("evidence")), "reviewed layer needs evidence")
    rows = [row for row in bundle.get(profile["scope"], []) if _view(row) == view_id
            and (profile["scope"] != "objects" or _object_type(row) == profile["object_type"])]
    if status in {"not_applicable", "reviewed_absence"}:
        permitted = profile["allow_not_applicable" if status == "not_applicable" else "allow_reviewed_absence"]
        reason_key = "applicability_reason" if status == "not_applicable" else "absence_reason"
        require(permitted and _text(review.get(reason_key)), "layer absence/applicability requires permitted musical reason")
        require(not rows, "layer absence/applicability contradicts existing analytical objects")
        return dict(review_status=status, reviewed_full_interval=reviewed_full,
                    structural_coverage_complete=False, content_accounted=reviewed_full,
                    rows=0, fields={}, coverage=dict(reviewed_duration_qn=_rational(_duration(intervals)),
                        review_extent_qn=_rational(duration), analytical_coverage_not_applicable=True),
                    issues=[] if reviewed_full else ["review_interval_gap"])
    issues = []
    coverage = dict(reviewed_duration_qn=_rational(_duration(intervals)), review_extent_qn=_rational(duration))
    if not rows:
        issues.append("missing_layer_rows")
    if profile["coverage"] == "notes":
        by_note = defaultdict(list)
        harmonies = {row["id"]: row for row in bundle.get("harmonies", [])}
        for row in rows:
            by_note[row["note_id"]].append((fraction(row["start_qn"]), fraction(row["end_qn"])))
            harmonic_fields = [row.get("fields", {}).get(field) for field in
                               ("analysis.chord_membership", "analysis.chord_degree", "analysis.key_degree")]
            if any(field is not None and _decision(field) == "known" for field in harmonic_fields):
                harmony = harmonies.get(row.get("harmony_id"))
                if harmony is None or _view(harmony) != view_id:
                    issues.append("unbound_harmonic_context")
        fully_covered = sum(_covered(by_note[n.note_id], n.onset_qn.to_fraction(),
            n.onset_qn.to_fraction() + n.duration_qn.to_fraction()) for n in piece.notes)
        structural = bool(piece.notes) and fully_covered == len(piece.notes)
        expected_duration = sum((n.duration_qn.to_fraction() for n in piece.notes), Fraction(0))
        covered_duration = sum((_duration(spans) for spans in by_note.values()), Fraction(0))
        coverage.update(total_notes=len(piece.notes), fully_covered_notes=fully_covered,
                        duration_unit="summed_note_time")
    elif profile["coverage"] == "interval":
        spans = [(fraction(r["start_qn"]), fraction(r["end_qn"])) for r in rows]
        structural = _covered(spans, Fraction(0), duration)
        expected_duration, covered_duration = duration, _duration(spans)
        coverage["duration_unit"] = "union_interval_time"
    else:
        structural = bool(rows)  # Sparse motifs: the full review, not motif spans, covers absence.
        expected_duration = covered_duration = None
    if expected_duration is not None:
        coverage.update(expected_duration_qn=_rational(expected_duration),
                        covered_duration_qn=_rational(covered_duration),
                        unrepresented_duration_qn=_rational(expected_duration - covered_duration))
    if not structural:
        issues.append("analytical_geometry_gap")
    if profile["scope"] == "objects" and any(not row.get("member_ids") for row in rows):
        issues.append("unaddressed_object")
    for row in rows:
        _semantic_values(row, profile["required_fields"], indexes)
    fields = {field: _field_summary(rows, field, profile["coverage"] == "notes")
              for field in profile["required_fields"]}
    for field, summary in fields.items():
        if any(summary["counts"][state] for state in ("missing", "unreviewed", "unresolved", "conflicting")):
            issues.append("unaccounted_field:" + field)
        if summary["counts"]["not_applicable"] and field not in profile["not_applicable_fields"]:
            issues.append("required_semantics_not_applicable:" + field)
    if not reviewed_full:
        issues.append("review_interval_gap")
    if status != "reviewed":
        issues.append("layer_not_reviewed")
    return dict(review_status=status, reviewed_full_interval=reviewed_full,
                structural_coverage_complete=structural, content_accounted=not issues,
                rows=len(rows), fields=fields, coverage=coverage, issues=issues)


def validate_annotation_completeness(piece, bundle: Mapping, assessment: Mapping, *,
                                     profile=None, require_complete=False) -> dict:
    """Return computed accounting; fail on malformed receipts or requested gate.

    Valid partial annotations produce an incomplete report. A complete report
    needs all declared views independently accounted for and reviewed. It is
    not a human-gold or correctness certificate. Hashes authenticate identities
    only after the caller has verified the input bytes against their bindings.
    """
    profile = default_whole_work_profile() if profile is None else profile
    _validate_profile(profile)
    validate_enrichment(piece, bundle)
    _validate_objects(bundle)
    require(assessment.get("schema") == ASSESSMENT_SCHEMA, "assessment schema mismatch")
    require((assessment.get("dataset_id"), assessment.get("piece_id")) == (piece.dataset_name, piece.piece_id),
            "assessment identity mismatch")
    binding = bundle.get("canonical_piece", {})
    require(_digest(binding.get("sha256")) and assessment.get("canonical_piece_sha256") == binding["sha256"],
            "assessment canonical hash mismatch")
    fingerprint = annotation_fingerprint(bundle)
    require(assessment.get("bundle_fingerprint") == fingerprint, "assessment bundle fingerprint mismatch")
    require(assessment.get("profile_fingerprint") == annotation_fingerprint(profile), "assessment profile fingerprint mismatch")
    work = assessment.get("work")
    require(isinstance(work, dict) and _text(work.get("work_family_id")), "work-family identity required")
    require(work.get("genre") in GENRES and _evidence(work.get("genre_evidence")), "evidenced quota genre required")
    extent = work.get("source_extent")
    require(extent in {"full_work", "excerpt", "unknown"}, "invalid source extent")
    duration = piece.duration_qn.to_fraction()
    if extent == "full_work":
        require(_evidence(work.get("full_work_evidence")), "full-work source evidence required")
        require(_interval(work, duration) == (Fraction(0), duration), "full-work extent differs from canonical interval")
        require(work.get("cropped", False) is False, "cropped material cannot claim full work")
    full_work = extent == "full_work" and any(not note.is_percussion for note in piece.notes) and duration > 0
    views = assessment.get("views")
    require(isinstance(views, list) and bool(views), "explicit analytical views required")
    reports = {}
    indexes = _semantic_indexes(piece, bundle)
    for view in views:
        view_id = view.get("view_id")
        require(_text(view_id) and view_id not in reports, "duplicate or missing view identity")
        _validate_author(view.get("author"))
        reviews = view.get("layers", {})
        require(isinstance(reviews, dict) and set(reviews) <= set(profile["layers"]), "unknown layer review")
        reports[view_id] = dict(layers={name: _assess_layer(piece, bundle, view_id, name, layer, reviews.get(name, {}), indexes)
                                       for name, layer in profile["layers"].items()})
    independent, contested = set(), set()
    reviews = assessment.get("independent_reviews", [])
    require(isinstance(reviews, list), "independent reviews must be a list")
    authors = {view["view_id"]: view["author"]["author_id"] for view in views}
    for review in reviews:
        view_id = review.get("view_id")
        require(view_id in reports and _text(review.get("reviewer_id"))
                and review["reviewer_id"] != authors[view_id], "independent reviewer identity required")
        require(review.get("reviewer_kind") in {"model", "human"}, "reviewer kind required")
        require(review.get("result") in {"accepted", "changes_requested", "unresolved"}
                and _evidence(review.get("evidence")), "independent review result/evidence required")
        require(review.get("bundle_fingerprint") == fingerprint, "independent review bundle fingerprint mismatch")
        if review["result"] == "accepted":
            independent.add(view_id)
        else:
            contested.add(view_id)
    layer_reports = [layer for view in reports.values() for layer in view["layers"].values()]
    all_reviewed = all(layer["reviewed_full_interval"] and layer["review_status"] in
                       {"reviewed", "not_applicable", "reviewed_absence"} for layer in layer_reports)
    accounted = all(layer["content_accounted"] for layer in layer_reports)
    independently_reviewed = set(reports) == independent and not contested
    complete = full_work and all_reviewed and accounted and independently_reviewed
    report = dict(schema=REPORT_SCHEMA, dataset_id=piece.dataset_name, piece_id=piece.piece_id,
                  work_family_id=work["work_family_id"], genre=work["genre"],
                  canonical_piece_sha256=binding["sha256"], bundle_fingerprint=fingerprint,
                  profile_fingerprint=annotation_fingerprint(profile), full_work_verified=full_work,
                  all_layers_reviewed=all_reviewed, all_required_content_accounted=accounted,
                  independently_reviewed=independently_reviewed, complete_for_declared_scope=complete,
                  views=reports, human_gold=False,
                  claim_boundary="Evidence-backed accounting and review receipts; not musical accuracy or human agreement.")
    if require_complete:
        require(complete, "whole-work annotation is incomplete for declared scope")
    return report


def summarize_annotation_completeness(piece, bundle, assessment, *, profile=None):
    """Non-gating alias used for progress ledgers; malformed input still fails."""
    return validate_annotation_completeness(piece, bundle, assessment, profile=profile)


def build_object_vocabulary(bundles: Iterable[Mapping]) -> dict:
    """Freeze typed per-object-type field classes on admitted TRAIN only.

    Annotator/view IDs are metadata. This sidecar does not alter the existing
    T07 vocabulary or add a classifier, graph input, or loss.
    """
    fields = defaultdict(lambda: defaultdict(set))
    for bundle in bundles:
        require(bundle.get("schema") == SCHEMA and bundle.get("current_split") == "train",
                "TRAIN enrichment required for object vocabulary")
        _validate_objects(bundle)
        for obj in bundle.get("objects", []):
            object_type = _object_type(obj)
            fields[object_type]
            for name, field in obj.get("fields", {}).items():
                values = fields[object_type][name]
                if field["available"]:
                    values.add(canonical(field["value"]))
    vocabulary = dict(schema=OBJECT_VOCABULARY_SCHEMA, unknown_value=-1,
        fields={object_type: {name: [json.loads(value) for value in sorted(values)]
                             for name, values in sorted(type_fields.items())}
                for object_type, type_fields in sorted(fields.items())})
    vocabulary["fingerprint"] = annotation_fingerprint(vocabulary)
    return vocabulary


def encode_object_fields(piece, bundle: Mapping, vocabulary: Mapping) -> dict:
    """Export object semantic arrays, preserving exact geometry and masks.

    Unknown states and known OOV values have -1/false; explicit false and zero
    are independent typed classes. Novel fields/types are retained as masked
    arrays and counted as OOV, never silently discarded.
    """
    validate_enrichment(piece, bundle)
    _validate_objects(bundle)
    require(vocabulary.get("schema") == OBJECT_VOCABULARY_SCHEMA, "object vocabulary schema mismatch")
    identity = {key: value for key, value in vocabulary.items() if key != "fingerprint"}
    require(annotation_fingerprint(identity) == vocabulary.get("fingerprint"), "object vocabulary fingerprint mismatch")
    require(vocabulary.get("unknown_value") == -1 and isinstance(vocabulary.get("fields"), dict),
            "invalid object vocabulary fields")
    mappings = {}
    for object_type, fields in vocabulary["fields"].items():
        require(_text(object_type) and isinstance(fields, dict), "invalid object vocabulary type")
        mappings[object_type] = {}
        for name, values in fields.items():
            require(_text(name) and isinstance(values, list), "invalid object vocabulary field")
            encoded = [canonical(value) for value in values]
            require(None not in values and len(encoded) == len(set(encoded)), "invalid duplicate/null vocabulary class")
            mappings[object_type][name] = {value: index for index, value in enumerate(encoded)}
    objects = defaultdict(list)
    for obj in bundle.get("objects", []):
        objects[_object_type(obj)].append(obj)
    groups = {}
    oov = Counter()
    for object_type, rows in sorted(objects.items()):
        fields = set(mappings.get(object_type, {})) | {name for row in rows for name in row.get("fields", {})}
        group = dict(ids=[row["id"] for row in rows], view_ids=[_view(row) for row in rows],
                     intervals=[[row["start_qn"], row["end_qn"]] for row in rows],
                     member_ids=[row.get("member_ids", []) for row in rows], values={}, masks={}, states={})
        for name in sorted(fields):
            values, masks, states = [], [], []
            lookup = mappings.get(object_type, {}).get(name, {})
            for row in rows:
                field = row.get("fields", {}).get(name)
                state = "missing" if field is None else _decision(field)
                available = state == "known"
                value = lookup.get(canonical(field["value"]), -1) if available else -1
                if available and value == -1:
                    oov[object_type + ":" + name] += 1
                values.append(value)
                masks.append(available and value != -1)
                states.append(state)
            group["values"][name] = values
            group["masks"][name] = masks
            group["states"][name] = states
        groups[object_type] = group
    return dict(schema=OBJECT_TARGETS_SCHEMA, dataset_id=piece.dataset_name, piece_id=piece.piece_id,
                current_split="train", vocabulary_fingerprint=vocabulary["fingerprint"],
                bundle_fingerprint=annotation_fingerprint(bundle), groups=groups, oov_counts=dict(oov))
