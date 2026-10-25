"""Exact contextual projection of authenticated native auxiliary targets.

This pure module neither reads corpora nor infers missing harmony. In particular,
Roman/root labels are not promoted to preparation, resolution or scalar tension.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
from math import gcd


SCHEMA = "EventTheoryEnrichment@1"
DEEP_FIELDS = ("analysis.harmonic_function", "analysis.structural_goal", "analysis.tension")


def exact(value: dict) -> Fraction:
    """Decode canonical reduced integer rational time; never accept floats."""
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("exact rational object required")
    n, d = value["num"], value["den"]
    if type(n) is not int or type(d) is not int or d <= 0 or gcd(n, d) != 1:
        raise ValueError("reduced rational required")
    return Fraction(n, d)


def rational(value: Fraction) -> dict:
    return {"num": value.numerator, "den": value.denominator}


def field(value, *, available=True, origin="source_native", evidence=None, reason=None,
          confidence=None):
    return {"value": value if available else None, "available": available,
            "origin": origin if available else "unknown", "confidence": confidence,
            "evidence": evidence or [], "unknown_reason": None if available else reason}


def unknown(reason):
    return field(None, available=False, reason=reason)


def _targets(piece, note_ids, annotation_ids):
    indexed = defaultdict(dict)
    for ordinal, target in enumerate(piece.get("targets", [])):
        alignment = target["alignment_type"]
        if alignment not in ("note", "annotation_span"):
            raise ValueError(f"unsupported native target alignment: {alignment}")
        ids = target["entity_ids"]
        columns = ("values", "mask", "confidence", "source", "provenance")
        if any(len(target[name]) != len(ids) for name in columns):
            raise ValueError("native target column lengths differ")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate native target entities")
        valid_ids = note_ids if alignment == "note" else annotation_ids
        for i, entity_id in enumerate(ids):
            if entity_id not in valid_ids:
                raise ValueError("native target names an absent entity")
            available, value = target["mask"][i], target["values"][i]
            if type(available) is not bool or (not available and value is not None):
                raise ValueError("invalid native mask/value")
            if available and value is None:
                raise ValueError("available native target has no value")
            key = (alignment, entity_id, target.get("annotation_view_id"))
            task = target["task"]
            if task in indexed[key]:
                raise ValueError("duplicate native target in one analytic view")
            indexed[key][task] = field(value, available=available,
                confidence=target["confidence"][i], reason="native_target_mask_false",
                evidence=[{"canonical_pointer": f"/targets/{ordinal}/values/{i}",
                           "target_id": target["target_id"],
                           "source": target["source"][i], "provenance": target["provenance"][i]}])
    return indexed


def _timeline(harmonies, keys):
    starts, ends = defaultdict(list), defaultdict(list)
    for kind, objects in (("h", harmonies), ("k", keys)):
        for obj in objects:
            a, b = exact(obj["start_qn"]), exact(obj["end_qn"])
            if b <= a:
                continue
            starts[a].append((kind, obj["id"]))
            ends[b].append((kind, obj["id"]))
    cuts = sorted(set(starts) | set(ends))
    active = {"h": set(), "k": set()}
    states = []
    for cut in cuts:
        for kind, entity in ends[cut]: active[kind].remove(entity)
        for kind, entity in starts[cut]: active[kind].add(entity)
        states.append((sorted(active["h"]), sorted(active["k"])))
    return cuts, states


def build_native_enrichment(piece: dict, *, canonical_binding: dict,
                            source_bindings: list, admission: dict) -> dict:
    """Keep every note and native target, partitioning held notes exactly.

    Caller establishes admission and SHA before decoding. Different analytic
    views are independent contexts; overlaps inside one view remain ambiguous.
    Point events retain native facts but no invented sounding interval. An
    unspecified (None) view is independent, not a fallback for named analyses.
    """
    if admission.get("current_split") != "train":
        raise ValueError("current TRAIN required")
    dataset = piece["dataset_name"]
    if dataset not in ("hooktheory", "pop909_cl"):
        raise ValueError("unsupported native dataset")
    if dataset == "hooktheory" and admission.get("native_split") != "TRAIN":
        raise ValueError("native HookTheory TRAIN required")
    notes, annotations = piece["notes"], piece.get("annotations", [])
    note_ids = {n["note_id"] for n in notes}
    annotation_ids = {a["annotation_id"] for a in annotations}
    if len(note_ids) != len(notes) or len(annotation_ids) != len(annotations):
        raise ValueError("duplicate canonical entity IDs")
    indexed = _targets(piece, note_ids, annotation_ids)
    source_views = {view for _, _, view in indexed}
    views = ([None] if None in source_views or not source_views else []) + sorted(
        view for view in source_views if view is not None)
    harmonies, keys, nonsounding = [], [], []
    for view in views:
        for i, a in enumerate(annotations):
            start, end = exact(a["start_qn"]), exact(a["end_qn"])
            if start < 0 or end < start:
                raise ValueError("invalid annotation interval")
            fields = dict(indexed.get(("annotation_span", a["annotation_id"], view), {}))
            is_key = any(k.startswith("theory.local_key.") for k in fields)
            is_harmony = a["annotation_type"] in ("theory.chord", "pop909_cl.chord")
            if not is_key and not is_harmony:
                # Do not silently discard a native span with an unexpected type.
                if fields: raise ValueError(f"unsupported annotation type {a['annotation_type']}")
                continue
            entity_id = a["annotation_id"] if view is None else a["annotation_id"] + "@view:" + view
            obj = {"id": entity_id, "source_annotation_id": a["annotation_id"],
                   "start_qn": rational(start), "end_qn": rational(end),
                   "harmony_view": view, "fields": fields,
                   "evidence": [{"canonical_pointer": f"/annotations/{i}"}]}
            if is_harmony:
                for name in DEEP_FIELDS: fields[name] = unknown("not_annotated_by_native_source")
            if end == start:
                # Canonical clipping may retain a chord at the piece boundary.
                # Its source facts survive without claiming any sounding span.
                obj.update({"type": "native_harmony_annotation" if is_harmony else "native_key_annotation",
                            "applicability": "nonsounding_native_annotation",
                            "nonsounding_reason": "zero_duration_source_span",
                            "source_annotation_type": a["annotation_type"]})
                nonsounding.append(obj)
            elif is_harmony: harmonies.append(obj)
            else: keys.append(obj)
    rows, point_count, rows_with_harmony = [], 0, 0
    key_lookup = {k["id"]: k for k in keys}
    timelines = {view: _timeline([h for h in harmonies if h["harmony_view"] == view],
                                 [k for k in keys if k["harmony_view"] == view]) for view in views}
    for ni, note in enumerate(notes):
        start = exact(note["onset_qn"])
        duration = exact(note["duration_qn"])
        if start < 0 or duration < 0: raise ValueError("invalid note interval")
        end = start + duration
        if not duration: point_count += 1
        for view in views:
            cuts, states = timelines[view]
            left, right = bisect_right(cuts, start), bisect_right(cuts, end)
            bounds = [start, *(c for c in cuts[left:right] if c < end), end]
            for a, b in zip(bounds, bounds[1:]):
                state_index = bisect_right(cuts, a) - 1
                hs, ks = states[state_index] if state_index >= 0 else ([], [])
                if not duration: hs, ks = [], []
                harmony_id = hs[0] if len(hs) == 1 else None
                key_id = ks[0] if len(ks) == 1 else None
                fields = dict(indexed.get(("note", note["note_id"], view), {}))
                if key_id is not None: fields.update(key_lookup[key_id]["fields"])
                else:
                    for task in ("theory.local_key.tonic_pc", "theory.local_key.mode"):
                        fields[task] = unknown("ambiguous_key_context" if len(ks) > 1 else "no_native_key_context")
                fields["analysis.temporal_role"] = unknown("not_annotated_by_native_source" if duration else "point_event_no_sounding_interval")
                rows.append({"id": f"context:{len(rows)}", "note_id": note["note_id"],
                    "start_qn": rational(a), "end_qn": rational(b), "harmony_id": harmony_id,
                    "harmony_view": view, "key_id": key_id, "active_harmony_ids": hs,
                    "active_key_ids": ks, "fields": fields,
                    "applicability": "sounding_note_interval" if duration else "point_event",
                    "evidence": [{"canonical_pointer": f"/notes/{ni}"}]})
                rows_with_harmony += harmony_id is not None
    coverage = {"note_count": len(notes), "unique_notes_accounted": len({r["note_id"] for r in rows}),
        "context_rows": len(rows), "harmonies": len(harmonies), "key_regions": len(keys),
        "analytic_views": len(views), "point_notes": point_count,
        "nonsounding_native_annotations": len(nonsounding),
        "native_available_nonsounding_fields": sum(f["available"] for obj in nonsounding for f in obj["fields"].values()),
        "context_rows_with_unique_harmony": rows_with_harmony,
        "context_rows_without_unique_harmony": len(rows) - rows_with_harmony,
        "native_available_harmony_fields": sum(f["available"] for h in harmonies for f in h["fields"].values()),
        "native_available_context_fields": sum(f["available"] for r in rows for f in r["fields"].values()),
        "authored_harmonic_relations": 0, "authored_note_roles": 0}
    if coverage["unique_notes_accounted"] != len(notes): raise ValueError("not all notes accounted")
    by_note = defaultdict(Fraction)
    for row in rows: by_note[(row["note_id"], row["harmony_view"])] += exact(row["end_qn"]) - exact(row["start_qn"])
    for note in notes:
        for view in views:
            if by_note[(note["note_id"], view)] != exact(note["duration_qn"]):
                raise ValueError("context coverage differs from note duration")
    return {"schema": SCHEMA, "dataset_id": dataset, "piece_id": piece["piece_id"],
        "current_split": "train", "canonical_piece": canonical_binding,
        "source_bindings": source_bindings, "harmonies": harmonies, "key_regions": keys,
        "note_contexts": rows, "objects": nonsounding, "relations": [], "coverage": coverage,
        "provenance": {"method": "exact_native_span_projection", "admission": admission,
            "source_quality_flags": piece.get("quality_flags", []),
            "source_provenance": piece.get("provenance", []),
            "source_metadata": piece.get("metadata", {}),
            "limitations": ["Native labels are not independently reviewed harmonic process analysis.",
                "No chord tones, absent decorations, applied functions or scalar tension inferred.",
                "Harmony/key boundaries are auxiliary target geometry, never required raw input."]}}
