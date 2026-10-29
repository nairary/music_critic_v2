"""Opt-in exact note×harmony supervision, separate from raw graphs and heads.

``non_chord`` means outside the explicitly supplied spelled core/added sets;
it is not a passing-tone, dissonance, tension or quality judgement. Native
source predicates remain separate from source-derived interval predicates.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from music_critic.data import CanonicalPiece, RationalTime, dumps_piece
from music_critic.tasks.harmonic_projection_primitives import DEGREES


CONTRACT_VERSION = "DenseHarmonicTargets@1"
COORDINATE_CONTRACT = MappingProxyType({
    "tpc": "absolute_fifths", "root_tpc": "absolute_fifths",
    "bass_note_tpc": "absolute_fifths", "localkey_tpc": "absolute_fifths",
    "chord_tones": "localkey_relative_fifths", "added_tones": "localkey_relative_fifths",
    "note_degree": "source_scale_degree_not_chord_degree",
})
ROLE_LABELS = ("core", "added", "non_chord")
DENSE_FIELD_VOCABULARIES = MappingProxyType({
    "role": ROLE_LABELS,
    **{key: (False, True) for key in (
        "core_member", "added_member", "chord_member", "is_root", "is_bass",
        "source_chord_member", "source_is_root", "source_is_bass")},
    "chord_degree": DEGREES, "key_degree": DEGREES, "source_note_degree": DEGREES,
    "compound_member": ("9", "11", "13"),
    **{key: tuple(range(12)) for key in (
        "interval_from_root_pc", "root_pc", "bass_pc", "local_key_pc")},
})
_NATURAL_FIFTHS = {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}
_NATURAL_DEGREE_FIFTHS = (0, 2, 4, -1, 1, 3, 5)


class DenseHarmonicError(ValueError):
    """Inconsistent exact geometry, identity or target schema."""


def _require(condition, message):
    if not condition:
        raise DenseHarmonicError(message)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _q(value):
    _require(isinstance(value, dict) and set(value) == {"num", "den"}, "exact rational object required")
    try:
        result = RationalTime(value["num"], value["den"])
    except (TypeError, ValueError) as exc:
        raise DenseHarmonicError("invalid rational time") from exc
    _require((result.num, result.den) == (value["num"], value["den"]), "rational must be reduced")
    return result.to_fraction()


def _qdict(value):
    return {"num": value.numerator, "den": value.denominator}


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            result = int(value)
            if value.strip() in (str(result), f"+{result}"):
                return result
        except ValueError:
            pass
    return None


def _source(harmony, name):
    return harmony["values"].get(name) if harmony["availability"].get(name) is True else None


def _unknown(reason):
    return {"value": None, "available": False, "origin": "unknown",
            "reason": reason, "source_fields": []}


def _known(field, value, origin="source_derived", source_fields=()):
    vocabulary = DENSE_FIELD_VOCABULARIES[field]
    if not any(type(value) is type(v) and value == v for v in vocabulary):
        return _unknown("outside_fixed_vocabulary")
    return {"value": value, "available": True, "origin": origin,
            "reason": None, "source_fields": list(source_fields)}


def _degree(difference):
    index = (4 * difference) % 7
    alteration = (difference - _NATURAL_DEGREE_FIFTHS[index]) // 7
    if not -3 <= alteration <= 3:
        return None
    accidental = "#" * alteration if alteration >= 0 else "b" * -alteration
    return f"{accidental}{index + 1}"


def _tone_set(harmony, field):
    tonic = _integer(_source(harmony, "localkey_tpc"))
    raw = _source(harmony, field)
    if tonic is None or raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(raw, (list, tuple)) or any(type(v) is not int for v in raw):
        return None
    return {v + tonic for v in raw}


def _consistent_segment_value(segments, field, *, theory=False):
    values = []
    for segment in segments:
        value = segment["values"].get(field)
        if segment.get("availability", {}).get(field, True) is not True:
            return None
        if theory and (segment["values"].get("valid_chord_label") is not True
                       or segment.get("availability", {}).get("valid_chord_label", True) is not True
                       or segment.get("availability", {}).get(field, True) is not True):
            return None
        if value is None:
            return None
        values.append(value)
    if not values or any(type(v) is not type(values[0]) or v != values[0] for v in values):
        return None
    return values[0]


def _fields(harmony, segments, *, reason):
    fields = {name: _unknown(reason) for name in DENSE_FIELD_VOCABULARIES}
    if reason != "available":
        return fields
    note_tpc = _integer(_consistent_segment_value(segments, "tpc"))
    root = _integer(_source(harmony, "root_tpc"))
    bass = _integer(_source(harmony, "bass_note_tpc"))
    tonic = _integer(_source(harmony, "localkey_tpc"))
    for field, value, source_field in (
        ("root_pc", root, "root_tpc"), ("bass_pc", bass, "bass_note_tpc"),
        ("local_key_pc", tonic, "localkey_tpc"),
    ):
        fields[field] = (_known(field, (7 * value) % 12, source_fields=(source_field,))
                         if value is not None else _unknown("source_field_unavailable"))
    for name in fields:
        if fields[name]["reason"] == "available":
            fields[name] = _unknown("source_field_unavailable")
    core, added = _tone_set(harmony, "chord_tones"), _tone_set(harmony, "added_tones")
    if note_tpc is not None:
        for name, tones, source_field in (("core_member", core, "chord_tones"),
                                          ("added_member", added, "added_tones")):
            if tones is not None:
                fields[name] = _known(name, note_tpc in tones, source_fields=("tpc", source_field, "localkey_tpc"))
        in_core = note_tpc in core if core is not None else None
        in_added = note_tpc in added if added is not None else None
        if in_core is True or in_added is True:
            fields["chord_member"] = _known("chord_member", True, source_fields=("tpc", "chord_tones", "added_tones"))
        elif in_core is False and in_added is False:
            fields["chord_member"] = _known("chord_member", False, source_fields=("tpc", "chord_tones", "added_tones"))
        if in_core is True:
            fields["role"] = _known("role", "core", source_fields=("tpc", "chord_tones"))
        elif in_core is False and in_added is not None:
            fields["role"] = _known("role", "added" if in_added else "non_chord", source_fields=("tpc", "chord_tones", "added_tones"))
        for name, target, native in (("is_root", root, "root_tpc"), ("is_bass", bass, "bass_note_tpc")):
            if target is not None:
                fields[name] = _known(name, note_tpc == target, source_fields=("tpc", native))
        if root is not None:
            degree = _degree(note_tpc - root)
            if degree is not None:
                fields["chord_degree"] = _known("chord_degree", degree, source_fields=("tpc", "root_tpc"))
            fields["interval_from_root_pc"] = _known("interval_from_root_pc", (7 * (note_tpc - root)) % 12, source_fields=("tpc", "root_tpc"))
        if tonic is not None:
            degree = _degree(note_tpc - tonic)
            if degree is not None:
                fields["key_degree"] = _known("key_degree", degree, source_fields=("tpc", "localkey_tpc"))
        compounds = _source(harmony, "explicit_compound_member_by_tpc")
        if isinstance(compounds, dict) and str(note_tpc) in compounds:
            fields["compound_member"] = _known("compound_member", str(compounds[str(note_tpc)]), "source_native", ("explicit_compound_member_by_tpc",))
    context = harmony.get("harmony_source_index", harmony["values"].get("unfolded_harmony_index"))
    matching = context is not None and all(str(s.get("harmony_source_index")) == str(context) for s in segments)
    for name, source_field in (("source_note_degree", "note_degree"),
                               ("source_chord_member", "tpc_is_in_label"),
                               ("source_is_root", "tpc_is_root"),
                               ("source_is_bass", "tpc_is_bass")):
        value = _consistent_segment_value(segments, source_field, theory=True) if matching else None
        fields[name] = (_known(name, value, "source_native", (source_field,)) if value is not None
                        else _unknown("native_context_mismatch" if not matching else "native_field_unavailable_or_conflicting"))
    return fields


def _source_geometry(piece, packet):
    _require(packet.get("schema") == "DenseHarmonicSourcePacket@1", "source packet schema mismatch")
    _require(packet.get("time_unit") == "quarter_note", "source time unit must be quarter_note")
    _require(packet.get("current_split") == "train" and piece.split in (None, "train"), "current TRAIN required")
    _require((packet.get("dataset_id"), packet.get("piece_id")) == (piece.dataset_name, piece.piece_id), "source piece identity mismatch")
    _require(packet.get("coordinate_contract") == dict(COORDINATE_CONTRACT), "source coordinate contract mismatch")
    _require(isinstance(packet.get("source_version_id"), str) and packet["source_version_id"], "source version required")
    notes = {n.note_id: n for n in piece.notes}
    _require(len(notes) == len(piece.notes), "duplicate canonical note IDs")
    harmonies = []
    identifiers = set()
    for h in packet["harmonies"]:
        _require(isinstance(h.get("id"), str) and h["id"] and h["id"] not in identifiers, "duplicate or missing harmony ID")
        identifiers.add(h["id"])
        start, end = _q(h["start_qn"]), _q(h["end_qn"])
        _require(start < end, "harmony interval must be positive")
        _require(isinstance(h.get("values"), dict) and isinstance(h.get("availability"), dict), "harmony field availability required")
        _require(all(type(v) is bool for v in h["availability"].values()), "field masks must be boolean")
        harmonies.append((start, end, h))
    segments = defaultdict(list)
    identifiers = set()
    for s in packet["note_segments"]:
        _require(s.get("source_event_id") not in identifiers and isinstance(s.get("source_event_id"), str)
                 and bool(s["source_event_id"]), "duplicate or missing source event ID")
        identifiers.add(s["source_event_id"])
        _require(s.get("note_id") in notes, "source segment references absent note")
        _require(type(s.get("binding_available")) is bool, "binding mask must be boolean")
        _require(isinstance(s.get("availability", {}), dict)
                 and all(type(v) is bool for v in s.get("availability", {}).values()), "note field masks must be boolean")
        start, end = _q(s["start_qn"]), _q(s["end_qn"])
        note = notes[s["note_id"]]
        nstart = note.onset_qn.to_fraction()
        _require(nstart <= start <= end <= nstart + note.duration_qn.to_fraction(), "source segment outside canonical note")
        if s["binding_available"]:
            _require(s["values"].get("pitch") == note.pitch, "source/canonical pitch mismatch")
            tpc = _integer(s["values"].get("tpc"))
            _require(tpc is None or 7 * tpc % 12 == note.pitch % 12, "source tpc/pitch mismatch")
        segments[s["note_id"]].append((start, end, s))
    return sorted(harmonies, key=lambda x: (x[0], x[1], x[2]["id"])), segments


def _note_identities(piece, segments):
    """Source Part/staff/voice identities, never raw track or instruments."""
    result = []
    for index, note in enumerate(piece.notes):
        ss = segments.get(note.note_id, [])
        observed = sorted({(s.get("part_id"), s.get("staff"), s.get("voice"))
                           for _, _, s in ss}, key=repr)
        start = note.onset_qn.to_fraction()
        end = start + note.duration_qn.to_fraction()
        cursor = start
        for a, b, _ in sorted(ss, key=lambda x: (x[0], x[1])):
            if a > cursor:
                break
            cursor = max(cursor, b)
        reason = ("zero_duration_note" if start == end else
                  "source_note_coverage_gap" if not ss or cursor != end else
                  "source_binding_masked" if not all(s["binding_available"] for _, _, s in ss) else
                  "conflicting_source_identity" if len(observed) != 1 else None)
        part, staff, voice = observed[0] if len(observed) == 1 else (None, None, None)
        if reason is None and (staff != note.staff or voice != note.voice):
            reason = "source_canonical_voice_mismatch"
        part_available = reason is None and isinstance(part, str) and bool(part)
        voice_available = part_available and type(staff) is int and type(voice) is int
        if reason is None and not voice_available:
            reason = "missing_source_identity"
        result.append({"note_id": note.note_id, "note_index": index,
                       "part_id": part if part_available else None,
                       "staff": staff if voice_available else None,
                       "voice": voice if voice_available else None,
                       "part_available": part_available, "voice_available": voice_available,
                       "reason": reason, "origin": "source_native" if part_available else "unknown",
                       "source_event_ids": sorted(s["source_event_id"] for _, _, s in ss),
                       "observed_source_identities": [dict(zip(("part_id", "staff", "voice"), identity))
                                                       for identity in observed]})
    return result


def _physical_links(notes, identities):
    """Mechanical relations only. Missing pairs never assert non-relation."""
    spans = [(_q(n["onset_qn"]), _q(n["onset_qn"]) + _q(n["duration_qn"])) for n in notes]
    by_voice = defaultdict(set)
    for index, identity in enumerate(identities):
        for observed in identity["observed_source_identities"]:
            key = (observed["part_id"], observed["staff"], observed["voice"])
            if isinstance(key[0], str) and key[0] and type(key[1]) is int and type(key[2]) is int:
                by_voice[key].add(index)
                # A repaired source/canonical voice disagreement blocks both
                # possible voices; it must never be skipped to invent a link.
                canonical_key = (key[0], notes[index]["staff"], notes[index]["voice"])
                if type(canonical_key[1]) is int and type(canonical_key[2]) is int:
                    by_voice[canonical_key].add(index)
    voice_links = []
    for index, identity in enumerate(identities):
        reason = identity["reason"] if not identity["voice_available"] else None
        candidates = []
        target = None
        if reason is None:
            key = (identity["part_id"], identity["staff"], identity["voice"])
            indices = by_voice[key]
            later = [j for j in indices if spans[j][0] > spans[index][0]]
            if sum(spans[j][0] == spans[index][0] for j in indices) != 1:
                reason = "ambiguous_current_onset"
            elif not later:
                reason = "no_successor"
            else:
                onset = min(spans[j][0] for j in later)
                candidates = sorted(j for j in later if spans[j][0] == onset)
                if len(candidates) != 1:
                    reason = "ambiguous_next_onset"
                elif not identities[candidates[0]]["voice_available"]:
                    reason = "untrusted_successor_identity"
                else:
                    candidate = candidates[0]
                    overlaps = any(
                        j != k and spans[j][0] < spans[k][1] and spans[k][0] < spans[j][1]
                        for k in (index, candidate) for j in indices)
                    if overlaps:
                        reason = "overlapping_voice_events"
                    else:
                        target = candidate
        available = target is not None
        voice_links.append({"id": f"voice-next:{index}", "source_note_id": notes[index]["note_id"],
                            "source_note_index": index, "target_note_id": notes[target]["note_id"] if available else None,
                            "target_note_index": target, "candidate_note_ids": [notes[j]["note_id"] for j in candidates],
                            "available": available, "reason": reason,
                            "origin": "source_derived" if available else "unknown",
                            "pitch_difference": notes[target]["pitch"] - notes[index]["pitch"] if available else None,
                            "start_qn": _qdict(spans[index][0]) if available else None,
                            "end_qn": _qdict(spans[target][0]) if available else None,
                            "gap_qn": _qdict(spans[target][0] - spans[index][1]) if available else None})
    sounding, active = [], []
    for j in sorted(range(len(notes)), key=lambda i: (spans[i][0], i)):
        if not identities[j]["part_available"] or spans[j][0] == spans[j][1]:
            continue
        active = [i for i in active if spans[i][1] > spans[j][0]]
        for i in active:
            if identities[i]["part_id"] == identities[j]["part_id"]:
                continue
            left, right = sorted((i, j))
            a, b = max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])
            sounding.append({"id": f"interpart:{left}:{right}",
                             "source_note_id": notes[left]["note_id"], "target_note_id": notes[right]["note_id"],
                             "source_note_index": left, "target_note_index": right,
                             "source_part_id": identities[left]["part_id"], "target_part_id": identities[right]["part_id"],
                             "available": True, "reason": None, "origin": "source_derived",
                             "start_qn": _qdict(a), "end_qn": _qdict(b),
                             "pitch_difference": notes[right]["pitch"] - notes[left]["pitch"]})
        active.append(j)
    sounding.sort(key=lambda r: (_q(r["start_qn"]), _q(r["end_qn"]), r["source_note_index"], r["target_note_index"]))
    return voice_links, sounding


def build_dense_harmonic_targets(piece: CanonicalPiece, packet: Mapping) -> dict:
    """Partition every note over exact known contexts and explicit unknown gaps.

    The caller authenticates input bytes/current split metadata before decoding.
    An archived null split is allowed with that explicit current TRAIN packet.
    No inferred harmony fills missing regions. Grace notes retain zero duration
    and an explicit unknown point row rather than acquiring sounding duration.
    """
    harmonies, segments = _source_geometry(piece, packet)
    fingerprint = sha256(dumps_piece(piece).encode()).hexdigest()
    if packet.get("canonical_piece_sha256") is not None:
        _require(packet["canonical_piece_sha256"] == fingerprint, "canonical piece SHA mismatch")
    rows = []
    for note_index, note in enumerate(piece.notes):
        start = note.onset_qn.to_fraction()
        end = start + note.duration_qn.to_fraction()
        hs = [h for h in harmonies if h[0] < end and start < h[1]]
        ss = segments.get(note.note_id, [])
        boundaries = sorted({start, end} | {max(start, min(end, t)) for a, b, _ in (*hs, *ss) for t in (a, b)})
        spans = list(zip(boundaries, boundaries[1:])) if start < end else [(start, end)]
        for ordinal, (a, b) in enumerate(spans):
            # A point has no positive-duration intersection. Its explicit
            # unknown row must obey the same geometry as the loader.
            active_h = [h for left, right, h in hs if a < b and left <= a and b <= right]
            active_s = [s for left, right, s in ss if left <= a and b <= right]
            reason = ("zero_duration_note" if a == b else
                      "no_harmony_coverage" if not active_h else
                      "conflicting_harmony_context" if len(active_h) != 1 else
                      "no_source_note_coverage" if not active_s else
                      "source_binding_masked" if not all(s["binding_available"] for s in active_s) else "available")
            harmony = active_h[0] if len(active_h) == 1 else None
            rows.append({"row_id": f"{piece.dataset_name}:{piece.piece_id}:row:{note_index}:{ordinal}", "note_id": note.note_id,
                         "note_index": note_index, "start_qn": _qdict(a), "end_qn": _qdict(b),
                         "harmony_id": harmony["id"] if harmony else None,
                         "harmony_ids": [h["id"] for h in active_h],
                         "source_event_ids": sorted(s["source_event_id"] for s in active_s),
                         "context_status": reason,
                         "fields": _fields(harmony, active_s, reason=reason)})
    bundle = {"schema": CONTRACT_VERSION, "field_registry_version": "1.0.0",
              "time_unit": "quarter_note",
              "dataset_id": piece.dataset_name, "piece_id": piece.piece_id,
              "canonical_piece_sha256": fingerprint, "current_split": "train",
              "source_version_id": packet["source_version_id"],
              "source_packet_sha256": sha256(_json(packet).encode()).hexdigest(),
              "coordinate_contract": dict(COORDINATE_CONTRACT),
              "canonical_notes": [{"note_id": n.note_id, "onset_qn": _qdict(n.onset_qn.to_fraction()),
                                   "duration_qn": _qdict(n.duration_qn.to_fraction()), "pitch": n.pitch,
                                   "staff": n.staff, "voice": n.voice} for n in piece.notes],
              "source_note_identities": _note_identities(piece, segments),
              "harmonies": deepcopy(packet["harmonies"]), "rows": rows,
              "field_vocabularies": {k: list(v) for k, v in DENSE_FIELD_VOCABULARIES.items()},
              "semantics": {"role_non_chord": "outside declared spelled core and added sets; not tension or quality",
                            "chord_degree": "generic spelled interval 1..7, not compound extension",
                            "gold_harmony": "auxiliary target geometry only; not raw encoder input",
                            "physical_links": "source-derived timing/pitch facts; omitted interpart pairs are unknown, not negatives",
                            "batch_row_identity": "sample_indices plus piece-qualified row_ids; repeated samples retain their original IDs"}}
    bundle["voice_links"], bundle["interpart_links"] = _physical_links(bundle["canonical_notes"], bundle["source_note_identities"])
    validate_dense_harmonic_targets(bundle, piece=piece)
    return bundle


def validate_dense_harmonic_targets(bundle, *, piece=None):
    _require(bundle.get("schema") == CONTRACT_VERSION, "dense target schema mismatch")
    _require(bundle.get("field_registry_version") == "1.0.0", "field registry mismatch")
    _require(bundle.get("time_unit") == "quarter_note", "dense time unit must be quarter_note")
    _require(bundle.get("field_vocabularies") == {k: list(v) for k, v in DENSE_FIELD_VOCABULARIES.items()}, "field vocabulary mismatch")
    _require(bundle.get("current_split") == "train", "current TRAIN required")
    _require(bundle.get("coordinate_contract") == dict(COORDINATE_CONTRACT), "source coordinate contract mismatch")
    notes = bundle["canonical_notes"]
    ids = [n["note_id"] for n in notes]
    _require(len(ids) == len(set(ids)), "duplicate canonical note IDs")
    if piece is not None:
        _require((bundle["dataset_id"], bundle["piece_id"]) == (piece.dataset_name, piece.piece_id), "bundle piece mismatch")
        _require(bundle["canonical_piece_sha256"] == sha256(dumps_piece(piece).encode()).hexdigest(), "bundle canonical SHA mismatch")
        _require(ids == [n.note_id for n in piece.notes], "bundle note order mismatch")
        _require(all(_q(stored["onset_qn"]) == actual.onset_qn.to_fraction()
                     and _q(stored["duration_qn"]) == actual.duration_qn.to_fraction()
                     for stored, actual in zip(notes, piece.notes)), "bundle canonical note geometry mismatch")
        _require(all((n["pitch"], n["staff"], n["voice"]) == (p.pitch, p.staff, p.voice)
                     for n, p in zip(notes, piece.notes)), "bundle canonical note identity mismatch")
    groups = defaultdict(list)
    row_ids = set()
    harmony_ids = {h["id"] for h in bundle["harmonies"]}
    _require(len(harmony_ids) == len(bundle["harmonies"]), "duplicate harmony ID")
    harmonies = sorted(((_q(h["start_qn"]), _q(h["end_qn"]), h["id"]) for h in bundle["harmonies"]))
    _require(all(a < b for a, b, _ in harmonies), "invalid harmony geometry")
    for row in bundle["rows"]:
        _require(row["row_id"] not in row_ids, "duplicate dense row ID")
        row_ids.add(row["row_id"])
        index = row["note_index"]
        _require(type(index) is int and 0 <= index < len(notes) and row["note_id"] == ids[index], "dense row note alignment mismatch")
        _require(set(row["harmony_ids"]) <= harmony_ids, "absent harmony reference")
        start, end = _q(row["start_qn"]), _q(row["end_qn"])
        expected_harmonies = [hid for a, b, hid in harmonies if a <= start and end <= b and start < end]
        _require(row["harmony_ids"] == expected_harmonies, "row harmony geometry mismatch")
        _require(row["harmony_id"] == (expected_harmonies[0] if len(expected_harmonies) == 1 else None), "row harmony reference mismatch")
        _require(set(row["fields"]) == set(DENSE_FIELD_VOCABULARIES), "dense fields incomplete")
        for name, value in row["fields"].items():
            _require(type(value["available"]) is bool, "field mask must be boolean")
            if value["available"]:
                _require(row["context_status"] == "available" and len(expected_harmonies) == 1, "known field without available harmony context")
                _require(any(type(value["value"]) is type(v) and value["value"] == v for v in DENSE_FIELD_VOCABULARIES[name]), "invalid numeric vocabulary value")
                _require(value["origin"] in ("source_native", "source_derived") and value["reason"] is None, "available origin mismatch")
            else:
                _require(value["value"] is None and value["origin"] == "unknown" and bool(value["reason"]), "unknown field must remain null")
        groups[index].append((_q(row["start_qn"]), _q(row["end_qn"])))
    for index, note in enumerate(notes):
        start, duration = _q(note["onset_qn"]), _q(note["duration_qn"])
        spans = groups[index]
        _require(bool(spans) and spans[0][0] == start and spans[-1][1] == start + duration, "incomplete note coverage")
        _require(all(a < b for a, b in spans) if duration > 0 else spans == [(start, start)], "invalid note span geometry")
        _require(all(a[1] == b[0] for a, b in zip(spans, spans[1:])), "note coverage gap or overlap")
    identities = bundle["source_note_identities"]
    _require(len(identities) == len(notes) and all(
        identity["note_index"] == i and identity["note_id"] == ids[i]
        for i, identity in enumerate(identities)), "source identity note alignment mismatch")
    for note, identity in zip(notes, identities):
        _require(type(identity["part_available"]) is bool and type(identity["voice_available"]) is bool,
                 "source identity masks must be boolean")
        if identity["part_available"]:
            _require(isinstance(identity["part_id"], str) and bool(identity["part_id"])
                     and identity["origin"] == "source_native", "invalid available Part identity")
        else:
            _require(identity["part_id"] is None and identity["origin"] == "unknown", "unknown Part identity must be null")
        if identity["voice_available"]:
            _require(identity["part_available"] and identity["reason"] is None
                     and type(identity["staff"]) is int and type(identity["voice"]) is int
                     and (identity["staff"], identity["voice"]) == (note["staff"], note["voice"]), "invalid available voice identity")
        else:
            _require(identity["staff"] is None and identity["voice"] is None and bool(identity["reason"]), "unknown voice identity must be null")
    voice, sounding = _physical_links(notes, identities)
    _require(bundle["voice_links"] == voice and bundle["interpart_links"] == sounding, "physical link geometry or identity mismatch")


def dumps_dense_harmonic_targets(bundle, *, indent=None):
    validate_dense_harmonic_targets(bundle)
    return json.dumps(bundle, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      indent=indent, separators=None if indent else (",", ":"))


def loads_dense_harmonic_targets(payload, *, piece=None):
    try:
        bundle = json.loads(payload)
        validate_dense_harmonic_targets(bundle, piece=piece)
        return bundle
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise DenseHarmonicError(f"invalid dense target sidecar: {exc}") from exc


@dataclass(frozen=True)
class DensePhysicalLinkBatch:
    source_note_indices: torch.Tensor
    target_note_indices: torch.Tensor
    sample_indices: torch.Tensor
    masks: torch.Tensor
    pitch_differences: torch.Tensor
    exact_intervals: tuple[tuple[RationalTime, RationalTime] | None, ...]


def _collate_links(samples, key):
    source, target, owners, masks, pitches, intervals = [], [], [], [], [], []
    offset = 0
    for owner, (piece, bundle) in enumerate(samples):
        for row in bundle[key]:
            available = row["available"]
            source.append(offset + row["source_note_index"])
            target.append(offset + row["target_note_index"] if available else -1)
            owners.append(owner); masks.append(available)
            pitches.append(row["pitch_difference"] if available else 0)
            intervals.append((RationalTime(**row["start_qn"]), RationalTime(**row["end_qn"])) if available else None)
        offset += len(piece.notes)
    return DensePhysicalLinkBatch(torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long),
                                  torch.tensor(owners, dtype=torch.long), torch.tensor(masks, dtype=torch.bool),
                                  torch.tensor(pitches, dtype=torch.long), tuple(intervals))


@dataclass(frozen=True)
class DenseHarmonicBatch:
    local_note_indices: torch.Tensor
    global_note_indices: torch.Tensor
    sample_indices: torch.Tensor
    row_ids: tuple[str, ...]
    exact_intervals: tuple[tuple[RationalTime, RationalTime], ...]
    harmony_ids: tuple[str | None, ...]
    values: Mapping[str, torch.Tensor]
    masks: Mapping[str, torch.Tensor]
    field_origins: tuple[Mapping[str, str], ...]
    voice_links: DensePhysicalLinkBatch
    interpart_links: DensePhysicalLinkBatch


def collate_dense_harmonic_targets(samples: Sequence[tuple[CanonicalPiece, Mapping]]) -> DenseHarmonicBatch:
    """CPU auxiliary tensors; no graph modification or gold encoder features."""
    local, global_, owners, ids, times, harmonies, origins = [], [], [], [], [], [], []
    values = {k: [] for k in DENSE_FIELD_VOCABULARIES}
    masks = {k: [] for k in DENSE_FIELD_VOCABULARIES}
    offset = 0
    for sample_index, (piece, bundle) in enumerate(samples):
        validate_dense_harmonic_targets(bundle, piece=piece)
        for row in bundle["rows"]:
            local.append(row["note_index"]); global_.append(offset + row["note_index"])
            owners.append(sample_index); ids.append(row["row_id"])
            times.append((RationalTime(**row["start_qn"]), RationalTime(**row["end_qn"])))
            harmonies.append(row["harmony_id"])
            origins.append({name: value["origin"] for name, value in row["fields"].items()})
            for name, vocabulary in DENSE_FIELD_VOCABULARIES.items():
                value = row["fields"][name]
                masks[name].append(value["available"])
                values[name].append(vocabulary.index(value["value"]) if value["available"] else -1)
        offset += len(piece.notes)
    return DenseHarmonicBatch(torch.tensor(local, dtype=torch.long), torch.tensor(global_, dtype=torch.long),
                              torch.tensor(owners, dtype=torch.long), tuple(ids), tuple(times), tuple(harmonies),
                              {k: torch.tensor(v, dtype=torch.long) for k, v in values.items()},
                              {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()}, tuple(origins),
                              _collate_links(samples, "voice_links"), _collate_links(samples, "interpart_links"))
