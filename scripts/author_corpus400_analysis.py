"""Pure construction of one explicit model analysis view; no musical reads.

Callers authenticate and admit the canonical piece and input packet before
calling these helpers. Native analysis remains a separately bound file: it is
never used to fill holes in this view. There is deliberately no CLI that opens
corpus paths and no automatic musical judgement or independent-review receipt.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from fractions import Fraction
import re
from typing import Mapping

from music_critic.tasks.event_theory_enrichment import (
    SCHEMA, MODEL_ANNOTATOR_SETTINGS, MODEL_AUTHORED_ORIGINS, _field, canonical, fraction, require, validate_enrichment,
)
from music_critic.tasks.whole_work_annotation import (
    annotation_fingerprint, default_whole_work_profile, validate_annotation_completeness,
)


RECIPE_SCHEMA = "Corpus400AuthoredRecipe@1"


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _evidence(value):
    return isinstance(value, list) and bool(value) and all(_text(item) for item in value)


def _binding(value):
    require(isinstance(value, dict) and _text(value.get("path"))
            and isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in value["sha256"])
            and type(value.get("size_bytes")) is int and value["size_bytes"] > 0,
            "authenticated input binding required")


def _q(value):
    value = Fraction(value)
    return dict(num=value.numerator, den=value.denominator)


def authored_field(value, evidence, *, annotation_state="known", reason=None, applicability_reason=None,
                   decision_reason=None, origin="astra_individual_context"):
    """Construct weak provenance; this helper does not decide musical values."""
    known = annotation_state == "known"
    require(origin in MODEL_AUTHORED_ORIGINS, "authored field requires a model origin")
    field = dict(value=deepcopy(value), available=known,
                 origin=origin if known else "unknown", confidence=None,
                 evidence=deepcopy(evidence), unknown_reason=None if known else reason,
                 annotation_state=annotation_state)
    if applicability_reason is not None:
        field["applicability_reason"] = applicability_reason
    if decision_reason is not None:
        field["decision_reason"] = decision_reason
    _authored_fields({"field": field})
    return field


def unknown_field(reason="not_judged_by_author"):
    return dict(value=None, available=False, origin="unknown", confidence=None,
                evidence=[], unknown_reason=reason, annotation_state="unreviewed")


def _authored_fields(fields):
    require(isinstance(fields, dict), "authored fields must be an object")
    for name, field in fields.items():
        require(_text(name), "authored field name required")
        _field(field)
        require(field["origin"] in MODEL_AUTHORED_ORIGINS | {"unknown"} and field["confidence"] is None,
                "recipe fields must retain authored weak provenance")
        if field["available"]:
            require(_evidence(field["evidence"]), "available authored field requires evidence references")


def resolve_note_id(piece, selector: Mapping) -> str:
    """Resolve an exact ID or onset+track+pitch address; never pick a tie."""
    allowed = {"note_id", "onset_qn", "duration_qn", "track_id", "pitch"}
    require(isinstance(selector, dict) and set(selector) <= allowed, "invalid note selector")
    require(_text(selector.get("note_id")) or {"onset_qn", "track_id", "pitch"} <= selector.keys(),
            "note selector requires an ID or exact onset/track/pitch")
    onset = fraction(selector["onset_qn"]) if "onset_qn" in selector else None
    duration = fraction(selector["duration_qn"]) if "duration_qn" in selector else None
    if "pitch" in selector:
        require(type(selector["pitch"]) is int and 0 <= selector["pitch"] <= 127, "invalid note selector pitch")
    selected = [note for note in piece.notes
        if ("note_id" not in selector or note.note_id == selector["note_id"])
        and ("track_id" not in selector or note.track_id == selector["track_id"])
        and ("pitch" not in selector or note.pitch == selector["pitch"])
        and (onset is None or note.onset_qn.to_fraction() == onset)
        and (duration is None or note.duration_qn.to_fraction() == duration)]
    require(len(selected) == 1, "note selector is missing, mismatched, or ambiguous")
    return selected[0].note_id


def context_id(view_id, note_id, start_qn, end_qn, harmony_id=None):
    """Stable address derived from the complete view/note/geometry tuple."""
    fraction(start_qn)
    fraction(end_qn)
    return "context:" + annotation_fingerprint(dict(view_id=view_id, note_id=note_id,
        start_qn=start_qn, end_qn=end_qn, harmony_id=harmony_id))[:24]


def resolve_context_id(bundle, selector: Mapping) -> str:
    allowed = {"context_id", "note_id", "start_qn", "end_qn", "harmony_id"}
    require(isinstance(selector, dict) and set(selector) <= allowed, "invalid context selector")
    require(_text(selector.get("context_id")) or {"note_id", "start_qn", "end_qn"} <= selector.keys(),
            "context selector requires an ID or exact note interval")
    left = fraction(selector["start_qn"]) if "start_qn" in selector else None
    right = fraction(selector["end_qn"]) if "end_qn" in selector else None
    selected = [row for row in bundle["note_contexts"]
        if ("context_id" not in selector or row["id"] == selector["context_id"])
        and ("note_id" not in selector or row["note_id"] == selector["note_id"])
        and ("harmony_id" not in selector or row["harmony_id"] == selector["harmony_id"])
        and (left is None or fraction(row["start_qn"]) == left)
        and (right is None or fraction(row["end_qn"]) == right)]
    require(len(selected) == 1, "context selector is missing, mismatched, or ambiguous")
    return selected[0]["id"]


def _reference(piece, bundle, reference):
    if _text(reference):
        return reference
    require(isinstance(reference, dict) and len(reference) == 1, "invalid participant reference")
    if "note" in reference:
        return resolve_note_id(piece, reference["note"])
    require("context" in reference, "participant reference must name a note or context")
    return resolve_context_id(bundle, reference["context"])


def _span(row, duration):
    left, right = fraction(row["start_qn"]), fraction(row["end_qn"])
    require(0 <= left <= right <= duration, "authored interval outside canonical piece")
    return left, right


def _harmonies(recipe, profile, duration):
    rows = []
    for source in recipe.get("harmonies", []):
        row = deepcopy(source)
        require(_text(row.get("id")), "authored harmony ID required")
        left, right = _span(row, duration)
        require(left < right, "authored harmony interval must be positive")
        _authored_fields(row.get("fields", {}))
        row["fields"] = {name: unknown_field() for name in profile["layers"]["harmony"]["required_fields"]} | row.get("fields", {})
        require(row.get("view_id", recipe["view_id"]) == recipe["view_id"], "recipe cannot mix analytic views")
        row["view_id"] = recipe["view_id"]
        if "note_template" in row:
            _validate_template(row["note_template"], row)
        rows.append(row)
    rows.sort(key=lambda row: (fraction(row["start_qn"]), fraction(row["end_qn"]), row["id"]))
    for prior, current in zip(rows, rows[1:]):
        require(fraction(prior["end_qn"]) <= fraction(current["start_qn"]),
                "overlapping harmonies require separate analytic views")
    return rows


def _template_key(key, basis):
    if basis == "spelled_pitch_class":
        match = re.fullmatch(r"([A-G]):(-?\d+)", key) if isinstance(key, str) else None
        return bool(match and str(int(match.group(2))) == match.group(2))
    upper = 11 if basis == "midi_pitch_class" else 127
    return isinstance(key, str) and key.isdigit() and str(int(key)) == key and 0 <= int(key) <= upper


def _sonority_key(tone, basis):
    """Normalize an explicitly written tone; never choose enharmonic spelling."""
    if type(tone) is int:
        require(0 <= tone <= 127 and basis != "spelled_pitch_class",
                "spelled template needs spelled sonority tones")
        return str(tone % 12 if basis == "midi_pitch_class" else tone)
    require(_text(tone), "invalid explicit sonority tone")
    spelling = re.fullmatch(r"([A-G])((?:#+|b+)?)(-?\d+)?", tone.strip().replace("♯", "#").replace("♭", "b"))
    require(spelling is not None, "sonority tone is not an explicit pitch spelling")
    step, accidental, octave = spelling.groups()
    alteration = accidental.count("#") - accidental.count("b")
    if basis == "spelled_pitch_class":
        return step + ":" + str(alteration)
    pitch_class = ({"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + alteration) % 12
    if basis == "midi_pitch_class":
        return str(pitch_class)
    require(octave is not None, "exact MIDI-pitch template requires octave in sonority spelling")
    # Retain octave crossings of enharmonic spelling (B#3 == C4), not modulo arithmetic.
    pitch = 12 * (int(octave) + 1) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + alteration
    require(0 <= pitch <= 127, "sonority spelling outside MIDI pitch range")
    return str(pitch)


def _validate_template(template, harmony):
    require(isinstance(template, dict) and template.get("schema") == "AuthoredNoteTemplate@1"
            and _text(template.get("id")) and _evidence(template.get("evidence")), "explicit authored note template evidence required")
    basis = template.get("basis")
    require(basis in {"spelled_pitch_class", "midi_pitch_class", "midi_pitch"}, "unsupported template pitch basis")
    if basis == "midi_pitch_class":
        require(template.get("enharmonic_policy") == "author_declared_pc_mapping",
                "pitch-class template requires explicit authored enharmonic policy")
    members = template.get("members")
    require(isinstance(members, list) and all(_template_key(key, basis) for key in members)
            and len(members) == len(set(members)) and type(template.get("members_complete")) is bool,
            "template member set/completeness required")
    require(harmony["fields"]["analysis.sonority"]["available"], "template requires an authored available sonority")
    sonority = harmony["fields"]["analysis.sonority"]["value"]
    require(isinstance(sonority, dict) and isinstance(sonority.get("tones"), list) and sonority["tones"],
            "template requires explicit sonority tones")
    sonority_keys = {_sonority_key(tone, basis) for tone in sonority["tones"]}
    require(set(members) <= sonority_keys and (not template["members_complete"] or set(members) == sonority_keys),
            "template membership contradicts the explicit sonority")
    for name in ("chord_degrees", "key_degrees"):
        degrees = template.get(name, {})
        require(isinstance(degrees, dict) and all(_template_key(key, basis) and isinstance(value, str)
                and re.fullmatch(r"[b#]*[1-9]\d*", value) for key, value in degrees.items()),
                "invalid explicit degree map")
    if template.get("key_degrees"):
        require(harmony["fields"]["analysis.key"]["available"], "key-degree template requires an authored available key")


def _template_projection(note, harmony, recipe_hash, *, upstream_origin="astra_individual_context"):
    names = ("analysis.chord_membership", "analysis.chord_degree", "analysis.key_degree")
    def unavailable(reason):
        return {name: unknown_field(reason) for name in names}
    if harmony is None:
        return unavailable("no_authored_harmony_for_interval")
    if "note_template" not in harmony:
        return unavailable("no_explicit_authored_pitch_template")
    if note.is_percussion:
        return unavailable("percussion_pitch_not_a_harmonic_pitch")
    if not note.duration_qn.to_fraction():
        return unavailable("point_event_no_sounding_interval")
    template = harmony["note_template"]
    basis = template["basis"]
    if basis == "spelled_pitch_class":
        if note.spelling_step is None or note.spelling_alter is None:
            return unavailable("template_requires_unavailable_note_spelling")
        key = note.spelling_step + ":" + str(note.spelling_alter)
    else:
        key = str(note.pitch % 12 if basis == "midi_pitch_class" else note.pitch)
    evidence = list(dict.fromkeys(template["evidence"] + harmony["fields"]["analysis.sonority"]["evidence"]
                                 + ["recipe:" + recipe_hash, "template:" + template["id"]]))
    derivation = dict(rule="explicit_authored_pitch_lookup", template_id=template["id"], pitch_key=key,
        matching_basis=basis, upstream_origin=upstream_origin,
        degree_semantics="author_declared_pitch_map_not_computed_interval")
    def derived(value, *, key_evidence=False):
        return dict(value=value, available=True, origin="source_derived", confidence=None,
            evidence=deepcopy(evidence + (harmony["fields"]["analysis.key"]["evidence"] if key_evidence else [])),
            unknown_reason=None, annotation_state="known", derivation=deepcopy(derivation))
    fields = unavailable("pitch_not_declared_in_authored_template")
    if key in template["members"] or template["members_complete"]:
        fields["analysis.chord_membership"] = derived(key in template["members"])
    if key in template.get("chord_degrees", {}):
        fields["analysis.chord_degree"] = derived(template["chord_degrees"][key])
    if key in template.get("key_degrees", {}):
        fields["analysis.key_degree"] = derived(template["key_degrees"][key], key_evidence=True)
    return fields


def _judgments(piece, recipe):
    by_note = {note.note_id: [] for note in piece.notes}
    notes = {note.note_id: note for note in piece.notes}
    for row in recipe.get("note_judgments", []):
        selector = row.get("selector", {"note_id": row.get("note_id")})
        identity = resolve_note_id(piece, selector)
        require(row.get("note_id", identity) == identity, "judgment selector and note ID disagree")
        left, right = _span(row, piece.duration_qn.to_fraction())
        note = notes[identity]
        start = note.onset_qn.to_fraction()
        end = start + note.duration_qn.to_fraction()
        require(start <= left <= right <= end and (left < right or start == end),
                "judgment point/interval does not match canonical note")
        _authored_fields(row.get("fields", {}))
        require(bool(row.get("fields")), "empty note judgment has no semantic decision")
        by_note[identity].append((left, right, row["fields"]))
    return by_note


def _contexts(piece, recipe, harmonies, judgments, profile, recipe_hash):
    rows = []
    harmonic_geometry = [(fraction(h["start_qn"]), fraction(h["end_qn"]), h) for h in harmonies]
    harmonic_starts = [left for left, _, _ in harmonic_geometry]
    harmonic_boundaries = sorted({value for left, right, _ in harmonic_geometry for value in (left, right)})
    for note in piece.notes:
        start = note.onset_qn.to_fraction()
        end = start + note.duration_qn.to_fraction()
        boundaries = {start, end}
        boundaries.update(value for value in harmonic_boundaries[bisect_right(harmonic_boundaries, start):bisect_right(harmonic_boundaries, end)]
                          if value < end)
        for left, right, _ in judgments[note.note_id]:
            boundaries.update((left, right))
        ordered = sorted(boundaries)
        intervals = [(start, end)] if start == end else list(zip(ordered, ordered[1:]))
        for left, right in intervals:
            index = bisect_right(harmonic_starts, left) - 1
            active = harmonic_geometry[index] if index >= 0 else None
            harmony = active[2] if active and active[0] <= left < active[1] and right <= active[1] else None
            hid = harmony["id"] if harmony else None
            fields = {name: unknown_field() for name in profile["layers"]["notes"]["required_fields"]}
            fields.update(_template_projection(note, harmony, recipe_hash,
                upstream_origin="model_individual_context" if recipe["author"]["model_requested"] == "gpt-6.1-sol"
                else "astra_individual_context"))
            supplied = set()
            for jleft, jright, updates in judgments[note.note_id]:
                if jleft <= left <= right <= jright:
                    require(not supplied & updates.keys(), "overlapping authored judgments for same field")
                    for name, update in updates.items():
                        previous = fields.get(name)
                        require(not (previous and previous["available"] and update["available"])
                                or canonical(previous["value"]) == canonical(update["value"]),
                                "individual judgment contradicts explicit pitch template")
                        fields[name] = deepcopy(update)
                        if previous and previous["available"]:
                            fields[name]["prior_template_projection"] = deepcopy(previous)
                    supplied.update(updates)
            rows.append(dict(id=context_id(recipe["view_id"], note.note_id, _q(left), _q(right), hid),
                view_id=recipe["view_id"], note_id=note.note_id, start_qn=_q(left), end_qn=_q(right),
                harmony_id=hid, fields=fields, applicability="point_note" if start == end else "note_harmonic_context"))
    return rows


def build_authored_view(piece, canonical_binding: Mapping, recipe: Mapping, *, input_binding: Mapping) -> dict:
    """Build one new T07 view and an honestly incomplete review assessment.

    Exact note judgement intervals may subdivide harmonic contexts. All missing
    fields remain unknown. Supplied author output hashes describe the retained
    model result, while recipe_fingerprint binds this normalized recipe. This
    function neither verifies remote model execution nor creates its metadata.
    """
    _binding(dict(canonical_binding))
    _binding(dict(input_binding))
    require(recipe.get("schema") == RECIPE_SCHEMA, "unsupported authored recipe schema")
    require(piece.split == "train", "admitted TRAIN canonical piece required")
    require((recipe.get("dataset_id"), recipe.get("piece_id")) == (piece.dataset_name, piece.piece_id), "recipe identity mismatch")
    require(recipe.get("canonical_piece_sha256") == canonical_binding["sha256"], "recipe canonical binding mismatch")
    require(_text(recipe.get("view_id")), "authored view ID required")
    require(recipe.get("author", {}).get("kind") == "model"
            and (recipe["author"].get("model_requested"), recipe["author"].get("reasoning_effort_requested")) in MODEL_ANNOTATOR_SETTINGS
            and recipe["author"].get("human_gold") is False, "explicit supported model weak authorship required (Astra ultra or Sol high)")
    require(recipe.get("author", {}).get("input_sha256") == input_binding["sha256"], "author input packet binding mismatch")
    require("independent_reviews" not in recipe, "independent review must be provided by a separate reviewer")
    if recipe["author"]["model_requested"] == "gpt-6.1-sol":
        supplied = [row.get("fields", {}) for layer in ("harmonies", "note_judgments", "objects")
                    for row in recipe.get(layer, [])]
        supplied += [{"relation": row} for row in recipe.get("relations", [])]
        for fields in supplied:
            _authored_fields(fields)
            for field in fields.values():
                require(not field["available"] or field["origin"] == "model_individual_context",
                        "Sol origin must retain model provenance")
    profile = default_whole_work_profile()
    harmonies = _harmonies(recipe, profile, piece.duration_qn.to_fraction())
    judgments = _judgments(piece, recipe)
    recipe_hash = annotation_fingerprint(recipe)
    bundle = dict(schema=SCHEMA, dataset_id=piece.dataset_name, piece_id=piece.piece_id, current_split="train",
        canonical_piece=deepcopy(dict(canonical_binding)), harmonies=harmonies,
        note_contexts=_contexts(piece, recipe, harmonies, judgments, profile, recipe_hash), objects=[], relations=[],
        source_bindings=[deepcopy(dict(input_binding))], provenance=dict(recipe_fingerprint=recipe_hash,
            author=deepcopy(recipe["author"]), native_view_policy="separate_file_never_used_to_fill_authored_unknowns"))
    object_profiles = {layer["object_type"]: layer for layer in profile["layers"].values() if layer["scope"] == "objects"}
    for source in recipe.get("objects", []):
        row = deepcopy(source)
        require(row.get("view_id", recipe["view_id"]) == recipe["view_id"], "recipe cannot mix analytic views")
        row["view_id"] = recipe["view_id"]
        _authored_fields(row.get("fields", {}))
        if row.get("object_type") in object_profiles:
            row["fields"] = {name: unknown_field() for name in object_profiles[row["object_type"]]["required_fields"]} | row.get("fields", {})
        require(not ("member_ids" in row and "member_refs" in row), "choose member IDs or references")
        row["member_ids"] = [_reference(piece, bundle, ref) for ref in row.pop("member_refs", row.get("member_ids", []))]
        bundle["objects"].append(row)
    for source in recipe.get("relations", []):
        row = deepcopy(source)
        _authored_fields({row.get("type", "relation"): row})
        require(row.get("view_id", recipe["view_id"]) == recipe["view_id"], "recipe cannot mix analytic views")
        row["view_id"] = recipe["view_id"]
        for direction in ("source", "target"):
            require(not (direction + "_ids" in row and direction + "_refs" in row), "choose participant IDs or references")
            row[direction + "_ids"] = [_reference(piece, bundle, ref) for ref in row.pop(direction + "_refs", row.get(direction + "_ids", []))]
        bundle["relations"].append(row)
    validate_enrichment(piece, bundle)
    reviews = recipe.get("layer_reviews", {})
    require(isinstance(reviews, dict) and set(reviews) <= set(profile["layers"]), "unknown reviewed layer")
    assessment = dict(schema="WholeWorkAnnotationAssessment@1", dataset_id=piece.dataset_name, piece_id=piece.piece_id,
        canonical_piece_sha256=canonical_binding["sha256"], bundle_fingerprint=annotation_fingerprint(bundle),
        profile_fingerprint=annotation_fingerprint(profile), work=deepcopy(recipe.get("work")),
        views=[dict(view_id=recipe["view_id"], author=deepcopy(recipe["author"]), layers={
            name: deepcopy(reviews.get(name, dict(review_status="not_reviewed", reviewed_intervals=[])))
            for name in profile["layers"]})], independent_reviews=[])
    report = validate_annotation_completeness(piece, bundle, assessment, profile=profile)
    return dict(bundle=bundle, assessment=assessment, report=report, recipe_fingerprint=recipe_hash)
