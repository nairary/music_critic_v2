"""AugmentedNet source-note facts with exact existing raw-adapter lineage.

Rows describe source note segments. Their repeated Roman labels do not prove
harmonic change positions, so this module never constructs harmony spans.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256

from music_critic.adapters.dilemmadata import _alignment_evidence_fingerprint
from music_critic.analysis.native_enrichment import field, rational, unknown
from music_critic.data import dumps_piece
from music_critic.tasks.event_theory_enrichment import validate_enrichment


NATIVE_FIELDS = (
    "a_romanNumeral", "a_pitchNames", "a_bass", "a_root", "a_inversion",
    "a_quality", "a_pcset", "a_localKey", "a_tonicizedKey", "a_degree1",
    "a_degree2", "a_simpleNumeral", "a_isOnset", "note_degree",
)
MISSING = {"", "<NA>", "NA", "NaN", "nan", "None", "null"}


def native_fields(row, *, usable, point, evidence):
    result = {}
    for name in NATIVE_FIELDS:
        value = row.get(name)
        reason = None
        if value is None or value in MISSING:
            reason = "source_missing"
        if row.get("valid_chord_label") != "True":
            reason = "source_validity_unavailable"
        if not usable:
            reason = "source_binding_masked"
        if point and name == "note_degree":
            reason = "point_event_no_sounding_role"
        if name == "a_isOnset":
            if value == "True":
                value = True
            elif value == "False":
                reason = reason or "native_boundary_positive_unlabeled"
            else:
                reason = reason or "invalid_source_boolean"
        result["an." + name] = field(value, available=reason is None,
            evidence=evidence, reason=reason)
    result["analysis.temporal_role"] = unknown("not_annotated_by_native_source")
    return result


def build_an_enrichment(piece, alignment, rows, admission, *, canonical_binding,
                        source_bindings, offset=Fraction(0)):
    """Project admitted native TRAIN rows; unknown tails remain unavailable."""
    if (admission.get("status") != "admitted" or admission.get("current_split") != "train"
            or admission.get("native_split") != "training" or piece.split not in (None, "train")):
        raise ValueError("current and native TRAIN admission required")
    rid = admission["record_id"]
    if not rid.startswith("an:training:") or alignment.record_id != rid:
        raise ValueError("AN source identity mismatch")
    if piece.targets or piece.annotations:
        raise ValueError("raw-only canonical required")
    piece_sha = sha256(dumps_piece(piece).encode()).hexdigest()
    if (alignment.piece_id != piece.piece_id or alignment.canonical_piece_sha256 != piece_sha
            or canonical_binding["sha256"] != piece_sha
            or alignment.fingerprint != _alignment_evidence_fingerprint(alignment)
            or len(rows) != len(alignment.rows)):
        raise ValueError("canonical/alignment binding mismatch")
    if admission["source_component_id"] != piece.source_group_id:
        raise ValueError("source component mismatch")
    notes = {n.note_id: (i, n) for i, n in enumerate(piece.notes)}
    contexts, omitted, groups = [], [], {}
    spans = defaultdict(list)
    for ordinal, (row, binding) in enumerate(zip(rows, alignment.rows)):
        if binding.ordinal != ordinal or binding.line != ordinal + 2:
            raise ValueError("source row ordinal mismatch")
        start = Fraction(row["s_offset_frac"]) + offset
        end = start + Fraction(row["s_duration_frac"])
        if end < start or start != binding.onset_qn.to_fraction():
            raise ValueError("source row timing mismatch")
        gid = row.get("a_annotationNumber", "")
        group = groups.setdefault(gid, {"id": gid, "source_rows": [], "raw_field_variants": []})
        group["source_rows"].append(binding.line)
        facts = {k: row[k] for k in (*NATIVE_FIELDS, "a_duration", "a_measure", "valid_chord_label") if k in row}
        if facts not in group["raw_field_variants"]:
            group["raw_field_variants"].append(facts)
        note_id = binding.canonical_note_id
        if note_id is None:
            if binding.repair_mask_scope != "note":
                raise ValueError("unexplained omitted source row")
            omitted.append({"source_row_number": binding.line, "reason": "raw_adapter_omitted_repair_row"})
            continue
        if note_id not in notes:
            raise ValueError("unknown canonical note")
        index, note = notes[note_id]
        left = note.onset_qn.to_fraction()
        right = left + note.duration_qn.to_fraction()
        if int(row["s_midi"]) != note.pitch or not left <= start <= end <= right:
            raise ValueError("source/canonical note geometry mismatch")
        if row["s_isOnset"] not in ("True", "False"):
            raise ValueError("invalid source attack boolean")
        if (row["s_isOnset"] == "False") != binding.tie_continuation:
            raise ValueError("source/canonical tie mismatch")
        usable = binding.repair_mask_scope == "none"
        if not binding.tie_continuation and start != left:
            raise ValueError("source attack differs from canonical attack")
        if binding.tie_continuation and start == left:
            usable = False
        contexts.append({"id": f"{rid}:source-row:{binding.line}", "note_id": note_id,
            "canonical_note_index": index, "start_qn": rational(start), "end_qn": rational(end),
            "harmony_id": None, "applicability": "source_note_annotation",
            "source_row_number": binding.line, "source_annotation_group_id": gid,
            "source_part_id": row.get("s_part_id"), "source_voice_id": row.get("s_voice_id"),
            "tie_continuation": binding.tie_continuation, "binding_available": usable,
            "fields": native_fields(row, usable=usable, point=start == end,
                                    evidence=[{"source_row": binding.line}])})
        spans[note_id].append((start, end))
    for note_id, (index, note) in notes.items():
        left = note.onset_qn.to_fraction(); right = left + note.duration_qn.to_fraction()
        cursor, gaps = left, []
        for a, b in sorted(spans[note_id]):
            if a > cursor: gaps.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < right: gaps.append((cursor, right))
        if left == right and not spans[note_id]: gaps.append((left, right))
        for ordinal, (a, b) in enumerate(gaps):
            contexts.append({"id": f"{rid}:unknown:{index}:{ordinal}", "note_id": note_id,
                "canonical_note_index": index, "start_qn": rational(a), "end_qn": rational(b),
                "harmony_id": None, "applicability": "unavailable",
                "fields": native_fields({}, usable=False, point=a == b, evidence=[])})
    available, unavailable = Counter(), Counter()
    for context in contexts:
        for name, value in context["fields"].items():
            if value["available"]: available[name] += 1
            else: unavailable[value["unknown_reason"]] += 1
    bundle = {"schema": "EventTheoryEnrichment@1", "dataset_id": piece.dataset_name,
        "piece_id": piece.piece_id, "record_id": rid, "current_split": "train",
        "canonical_piece": canonical_binding, "source_bindings": source_bindings,
        "harmonies": [], "note_contexts": contexts, "relations": [],
        "source_annotation_groups": list(groups.values()),
        "coverage": {"canonical_notes": len(notes), "canonical_notes_accounted": len(spans),
            "context_rows": len(contexts), "native_source_rows": len(rows),
            "source_annotation_groups": len(groups), "exact_harmonic_spans": 0,
            "semantic_relations": 0, "available_by_field": dict(available),
            "unknown_by_reason": dict(unavailable), "omitted_source_rows": omitted},
        "provenance": {"origin": "source_native", "admission": admission,
            "source_component_id": piece.source_group_id,
            "method": "exact_AN_source_note_segment_projection",
            "limitations": ["Source note annotations do not establish harmony duration.",
                "Native AN/DLC/TAVERN records may be alternative views of the same work.",
                "No source-derived boundary negatives or interpreted temporal roles invented."]}}
    validate_enrichment(piece, bundle)
    return bundle
