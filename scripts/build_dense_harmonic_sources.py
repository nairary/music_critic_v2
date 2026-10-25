#!/usr/bin/env python3
"""Extract exact source geometry for the registered three-work TRAIN release.

No harmonic role is inferred here. The original score owns harmony boundaries;
the TSV owns source fields and exact row-to-canonical-note lineage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from music_critic.analysis.target_binding import bind_source_events
from music_critic.data import loads_piece


SCHEMA = "DenseHarmonicSourcePacket@1"
COORDINATES = {
    "tpc": "absolute_fifths", "root_tpc": "absolute_fifths",
    "bass_note_tpc": "absolute_fifths", "localkey_tpc": "absolute_fifths",
    "chord_tones": "localkey_relative_fifths", "added_tones": "localkey_relative_fifths",
    "note_degree": "source_scale_degree_not_chord_degree",
}
DURATIONS = {name: Fraction(value) for name, value in {
    "long": "16", "breve": "8", "whole": "4", "half": "2", "quarter": "1",
    "eighth": "1/2", "16th": "1/4", "32nd": "1/8", "64th": "1/16",
    "128th": "1/32", "256th": "1/64", "512th": "1/128", "1024th": "1/256",
}.items()}
GRACES = {"appoggiatura", "acciaccatura", "grace4", "grace16", "grace32",
          "grace8after", "grace16after", "grace32after"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rt(value):
    value = Fraction(value)
    return {"num": value.numerator, "den": value.denominator}


def frac(value):
    return Fraction(value["num"], value["den"]) if isinstance(value, dict) else Fraction(value)


def text_time(value):
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def parse_score_harmonies(score_xml):
    """Read supported score timing exactly; reject unsupported navigation.

    Repeat traversal handles plain, nonnested start/end repeats. Tuplets in a
    harmony-bearing voice, volta/jump navigation and simultaneous conflicting
    labels are rejected instead of silently assigning approximate boundaries.
    """
    root = ET.fromstring(score_xml)
    score = root.find("Score")
    require(score is not None, "missing Score")
    require(not score.findall(".//Jump") and not score.findall('.//Spanner[@type="Volta"]'),
            "unsupported score navigation")
    staffs = score.findall("Staff")
    require(bool(staffs), "missing score staffs")
    measures = staffs[0].findall("Measure")
    require(bool(measures), "missing score measures")
    lengths, meter = [], None
    for measure in measures:
        signatures = {(int(ts.findtext("sigN")), int(ts.findtext("sigD")))
                      for ts in measure.findall("./voice/TimeSig")}
        require(len(signatures) <= 1, "ambiguous meter")
        if signatures:
            num, den = signatures.pop()
            meter = Fraction(4 * num, den)
        require(meter is not None, "unknown initial meter")
        length = 4 * Fraction(measure.get("len")) if measure.get("len") else meter
        require(length > 0, "invalid measure length")
        lengths.append(length)
    folded = defaultdict(list)
    for staff in staffs:
        staff_measures = staff.findall("Measure")
        require(len(staff_measures) == len(measures), "staff measure count mismatch")
        for mc, measure in enumerate(staff_measures, 1):
            for voice_index, voice in enumerate(measure.findall("voice"), 1):
                if voice.find("Harmony") is None:
                    continue
                require(not voice.findall(".//Tuplet"), "unhandled tuplet in harmony-bearing voice")
                cursor = Fraction(0)
                for item in voice:
                    if item.tag == "location":
                        require(item.find("measures") is None, "unsupported cross-measure location")
                        cursor += 4 * Fraction(item.findtext("fractions") or "0")
                    elif item.tag == "Harmony":
                        label = item.findtext("name")
                        require(isinstance(label, str) and bool(label), "empty harmony label")
                        require(0 <= cursor < lengths[mc - 1], "harmony outside measure")
                        folded[mc].append({"label": label, "offset": cursor,
                                           "source_mc": mc, "score_staff": staff.get("id"),
                                           "score_voice": voice_index})
                    elif item.tag in ("Chord", "Rest"):
                        if any(child.tag in GRACES for child in item):
                            continue
                        dtype = item.findtext("durationType")
                        require(dtype == "measure" or dtype in DURATIONS, "unknown score duration")
                        duration = lengths[mc - 1] if dtype == "measure" else DURATIONS[dtype]
                        dots = int(item.findtext("dots") or "0")
                        require(0 <= dots <= 4, "unsupported dot count")
                        duration *= sum((Fraction(1, 2**i) for i in range(dots + 1)), Fraction(0))
                        cursor += duration
                    require(0 <= cursor <= lengths[mc - 1], "score cursor outside measure")
    traversal, repeat_start = [], 0
    for index, measure in enumerate(measures):
        if measure.find("startRepeat") is not None:
            require(repeat_start == index, "nested or ambiguous repeat")
            repeat_start = index
        traversal.append(index)
        closing = measure.find("endRepeat")
        if closing is not None:
            count = int(closing.text or "2")
            require(1 <= count <= 8, "unsupported repeat count")
            traversal.extend(list(range(repeat_start, index + 1)) * (count - 1))
            repeat_start = index + 1
    playthrough, unfolded, position = [], [], Fraction(0)
    for mc_playthrough, index in enumerate(traversal, 1):
        playthrough.append({"mc": index + 1, "mc_playthrough": mc_playthrough,
                            "start_qn": rt(position), "end_qn": rt(position + lengths[index])})
        for harmony in sorted(folded[index + 1], key=lambda h: (h["offset"], h["score_staff"], h["score_voice"])):
            unfolded.append({**{k: v for k, v in harmony.items() if k != "offset"},
                             "source_harmony_index": str(len(unfolded)),
                             "source_mc_playthrough": mc_playthrough,
                             "start_qn": rt(position + harmony["offset"])})
        position += lengths[index]
    for index, harmony in enumerate(unfolded):
        right = frac(unfolded[index + 1]["start_qn"]) if index + 1 < len(unfolded) else position
        require(frac(harmony["start_qn"]) < right, "simultaneous or reversed harmony boundaries")
        harmony["end_qn"] = rt(right)
        harmony["end_origin"] = "next_score_harmony" if index + 1 < len(unfolded) else "explicit_score_end"
    require(bool(unfolded), "score contains no harmony labels")
    return {"harmonies": unfolded, "playthrough_measures": playthrough,
            "score_end_qn": rt(position), "folded_measure_count": len(measures)}


HARMONY_FIELDS = (
    "unfolded_harmony_index", "label", "globalkey", "localkey", "pedal", "chord",
    "special", "numeral", "form", "figbass", "changes", "relativeroot", "chord_type",
    "globalkey_is_minor", "localkey_is_minor", "chord_tones", "added_tones", "root",
    "bass_note", "globalkey_mode", "localkey_mode", "localkey_resolved", "localkey_and_mode",
    "root_roman", "relativeroot_resolved", "effective_localkey", "effective_localkey_resolved",
    "effective_localkey_is_minor", "chord_reduced", "chord_reduced_and_mode", "pedal_resolved",
    "chord_and_mode", "applied_to_numeral", "numeral_or_applied_to_numeral", "a_simpleNumeral",
    "a_degree1", "a_degree2", "globalkey_tpc", "localkey_tpc", "tonicized_tpc", "root_tpc",
    "bass_note_tpc", "a_root", "a_bass", "a_localKey", "a_tonicizedKey", "a_quality",
    "a_inversion", "valid_chord_label", "valid_cadence_label", "valid_phrase_label",
    "valid_pedal_point_label", "valid_section_start_label",
)
EMPTY_SYNTAX = {"pedal", "special", "form", "figbass", "changes", "relativeroot",
                "relativeroot_resolved", "pedal_resolved", "applied_to_numeral", "a_degree2"}
NATIVE_FIELDS = ("note_degree", "tpc_is_in_label", "tpc_is_root", "tpc_is_bass")


def source_bool(value):
    require(value in ("True", "False", "", None), "invalid source boolean")
    return {"True": True, "False": False, "": None, None: None}[value]


def extract_source_packet(piece, alignment_payload, rows, score_xml, metadata):
    """Normalize one authenticated score/TSV pair, retaining all raw facts.

    ``metadata`` carries the prior authenticated source identity and XML Part
    mapping. This pure function never opens files or chooses dataset membership.
    """
    geometry = parse_score_harmonies(score_xml)
    case, rid, version = (metadata[k] for k in ("case_id", "record_id", "source_version_id"))
    offset = Fraction(metadata["source_to_canonical_offset_qn"])
    partmap = {}
    for part in metadata["parts"]:
        for staff in part["staff_ids"]:
            require(staff not in partmap, "ambiguous source staff/Part mapping")
            partmap[staff] = part["part_id"]
    events, source_groups = [], defaultdict(list)
    measuremap = {m["mc_playthrough"]: m for m in geometry["playthrough_measures"]}
    for line, row in enumerate(rows, 2):
        start, duration = Fraction(row["quarterbeats_playthrough"]), 4 * Fraction(row["duration"])
        mc_playthrough = int(row["mc_playthrough"])
        require(mc_playthrough in measuremap and int(row["mc"]) == measuremap[mc_playthrough]["mc"],
                "source/score unfolded measure identity mismatch")
        measure = measuremap[mc_playthrough]
        require(frac(measure["start_qn"]) <= start < frac(measure["end_qn"]),
                "source note onset outside exact score measure")
        hid = row["unfolded_harmony_index"]
        require(hid.isdecimal() and int(hid) < len(geometry["harmonies"]), "invalid source harmony index")
        require(row["label"] == geometry["harmonies"][int(hid)]["label"], "source/score harmony label mismatch")
        source_groups[hid].append((line, row))
        events.append({"id": f"{case}:l{line}", "source_version_id": version,
                       "source_row_id": f"{rid}#line={line}", "source_row_number": line,
                       "start": text_time(start), "duration": text_time(duration), "end": text_time(start + duration),
                       "pitch": int(row["pitch"]), "source_staff": row["staff"], "source_voice": row["voice"],
                       "part_id": partmap.get(row["staff"]), "is_note_onset": source_bool(row["is_note_onset"]),
                       "time_unit": "quarter_note"})
    packet = {"source_version_id": version, "current_split": metadata["current_split"],
              "source_table": {"sha256": metadata["source_sha256"]}, "events": events}
    bindings = bind_source_events(piece, alignment_payload, packet, record_id=rid,
                                  source_sha256=metadata["source_sha256"], current_split=metadata["current_split"],
                                  source_to_canonical_offset_qn=text_time(offset))
    harmonies, conflicts = [], []
    for harmony in geometry["harmonies"]:
        index = harmony["source_harmony_index"]
        group = source_groups.get(index, [])
        values, availability, field_conflicts = {}, {}, {}
        native_valid = bool(group) and all(row.get("valid_chord_label") == "True" for _, row in group)
        for field in HARMONY_FIELDS:
            candidates = sorted({row[field] for _, row in group if field in row})
            values[field] = candidates[0] if len(candidates) == 1 else None
            availability[field] = len(candidates) == 1 and (bool(candidates[0]) or field in EMPTY_SYNTAX)
            if field not in {"label", "unfolded_harmony_index"} and not field.startswith("valid_"):
                availability[field] = availability[field] and native_valid
            if len(candidates) > 1:
                field_conflicts[field] = candidates
        values["label"], availability["label"] = harmony["label"], True
        values["unfolded_harmony_index"], availability["unfolded_harmony_index"] = index, True
        if field_conflicts:
            conflicts.append({"source_harmony_index": index, "fields": field_conflicts})
        harmonies.append({"id": f"{case}:h{index}", "harmony_source_index": index,
                          "start_qn": rt(frac(harmony["start_qn"]) + offset),
                          "end_qn": rt(frac(harmony["end_qn"]) + offset),
                          "source_start_qn": harmony["start_qn"], "source_end_qn": harmony["end_qn"],
                          "values": values, "availability": availability,
                          "field_conflicts": field_conflicts,
                          "source_row_ids": [f"{rid}#line={line}" for line, _ in group],
                          "geometry_origin": "retained_score_harmony_and_explicit_repeat_traversal",
                          "end_origin": harmony["end_origin"], "score_mc": harmony["source_mc"],
                          "score_mc_playthrough": harmony["source_mc_playthrough"],
                          "score_staff": harmony["score_staff"], "score_voice": harmony["score_voice"],
                          "source_kind": "dataset_annotation", "confidence": None})
    segments, context_mismatches = [], []
    canonical_notes = {note.note_id: note for note in piece.notes}
    for row, binding in zip(rows, bindings):
        hi = row["unfolded_harmony_index"]
        harmony = harmonies[int(hi)]
        anchor = Fraction(binding.canonical_start_qn)
        context_consistent = frac(harmony["start_qn"]) <= anchor < frac(harmony["end_qn"])
        if not context_consistent:
            context_mismatches.append(binding.source_event_id)
        values = {key: int(row[key]) for key in ("pitch", "tpc", "alter", "octave")}
        values["step"] = row["step"]
        values["note_degree"] = row["note_degree"] or None
        values.update({key: source_bool(row[key]) for key in (*NATIVE_FIELDS[1:], "valid_chord_label")})
        availability = {key: value is not None for key, value in values.items()}
        for key in NATIVE_FIELDS:
            availability[key] &= context_consistent and binding.binding_available and values["valid_chord_label"] is True
        segments.append({"note_id": binding.canonical_note_id, "source_event_id": binding.source_event_id,
                         "source_row_id": f"{rid}#line={binding.source_row_number}",
                         "source_row_number": binding.source_row_number,
                         "start_qn": rt(binding.canonical_start_qn), "end_qn": rt(binding.canonical_end_qn),
                         "source_start_qn": rt(binding.source_start_qn), "source_end_qn": rt(binding.source_end_qn),
                         "part_id": binding.source_part_id, "staff": binding.source_staff, "voice": binding.source_voice,
                         "canonical_track_id": binding.canonical_track_id, "is_note_onset": not binding.tie_continuation,
                         "canonical_staff": canonical_notes[binding.canonical_note_id].staff if binding.canonical_note_id else None,
                         "canonical_voice": canonical_notes[binding.canonical_note_id].voice if binding.canonical_note_id else None,
                         "tie_continuation": binding.tie_continuation, "binding_available": binding.binding_available,
                         "mask_reason": binding.mask_reason, "harmony_source_index": hi,
                         "source_harmony_context_consistent": context_consistent,
                         "values": values, "availability": availability,
                         "source_measure": row["mn"], "source_mc": int(row["mc"]),
                         "source_mc_playthrough": int(row["mc_playthrough"]),
                         "source_kind": "dataset_annotation", "confidence": None})
    covered = {segment["note_id"] for segment in segments if segment["note_id"] is not None}
    require(covered == {note.note_id for note in piece.notes}, "not all canonical notes accounted for")
    notes = [{"note_id": note.note_id, "pitch": note.pitch, "track_id": note.track_id,
              "start_qn": rt(Fraction(note.onset_qn.num, note.onset_qn.den)),
              "end_qn": rt(Fraction(note.onset_qn.num, note.onset_qn.den) + Fraction(note.duration_qn.num, note.duration_qn.den)),
              "staff": note.staff, "voice": note.voice, "part_id": partmap.get(str(note.staff)),
              "is_grace": note.is_grace, "spelling_step": note.spelling_step, "spelling_alter": note.spelling_alter}
             for note in piece.notes]
    return {"schema": SCHEMA, "dataset_id": piece.dataset_name, "piece_id": piece.piece_id,
            "case_id": case, "record_id": rid, "current_split": "train", "time_unit": "quarter_note",
            "source_version_id": version, "source_sha256": metadata["source_sha256"],
            "source_to_canonical_offset_qn": text_time(offset), "coordinate_contract": COORDINATES,
            "harmonies": harmonies, "note_segments": segments, "notes": notes, "parts": metadata["parts"],
            "score_geometry": {**{k: v for k, v in geometry.items() if k != "harmonies"},
                               "canonical_score_end_qn": rt(frac(geometry["score_end_qn"]) + offset)},
            "diagnostics": {"source_rows": len(rows), "canonical_notes": len(piece.notes),
                            "source_harmonies": len(harmonies), "canonical_notes_accounted": len(covered),
                            "binding_unavailable": sum(not b.binding_available for b in bindings),
                            "tie_continuation_segments": sum(b.tie_continuation for b in bindings),
                            "source_harmony_context_mismatch_events": context_mismatches,
                            "source_field_conflicts": conflicts},
            "claim_boundary": "Source-native harmonic facts and exact geometry, not Astra labels or music quality"}


def authenticate(root, entry, receipts):
    path = (root / entry["path"]).resolve()
    require(path.is_relative_to(root), "input outside repository")
    raw = path.read_bytes()
    observed = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    require(all(observed[k] == entry[k] for k in observed), f"input changed before decode: {path}")
    receipts.append({"path": str(path.relative_to(root)), **observed})
    return raw


def release(root, output):
    root, output = root.resolve(), output.resolve()
    require(output.is_relative_to(root / "outputs"), "output must remain under ignored outputs")
    output.mkdir(parents=True, exist_ok=False)
    receipts, summaries = [], []
    started = time.perf_counter()
    source_snapshot = {}
    for name in ("scripts/build_dense_harmonic_sources.py", "src/music_critic/analysis/target_binding.py",
                 "tests/test_build_dense_harmonic_sources.py", "tests/analysis/test_target_binding.py"):
        raw = (root / name).read_bytes()
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        source_snapshot[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    (output / "source-snapshot.json").write_text(json.dumps(source_snapshot, indent=2) + "\n")
    try:
        active = json.loads((root / "docs/experiments/records/EXP-DENSE-HARMONIC-TARGETS-001.json").read_text())
        require(active["status"] in {"planned", "running"}, "active planned record required")
        parent = json.loads((root / "docs/experiments/records/EXP-TONAL-RELATIONS-TARGETS-001.json").read_text())
        require(parent["status"] == "completed", "completed T05 parent required")
        parent_files = {a["locator"]: a for a in parent["artifacts"]}
        entry = parent_files["outputs/tonal_relations_targets_20261006_001/export-inputs.json"]
        inputs = json.loads(authenticate(root, {"path": entry["locator"], **entry}, receipts))
        pilot = json.loads((root / "docs/experiments/records/EXP-TONAL-RELATIONS-ASTRA-PILOT-001.json").read_text())
        pilot_files = {a["locator"]: a for a in pilot["artifacts"]}
        require({w["case_id"] for w in inputs["works"]} == {"mozart-k283-i", "mendelssohn-op12-i", "pleyel-b309-i"}, "exact three-work scope required")
        for work in inputs["works"]:
            admission = json.loads(authenticate(root, work["admission"], receipts))
            require(admission["status"] == "admitted_current_train", "TRAIN admission required")
            current_entry = next(m for m in admission["metadata_bindings"] if m["path"] == "outputs/c4_pop909_20261005/dilemma-data-manifest.json")
            current = json.loads(authenticate(root, current_entry, receipts))
            current_row, = [r for r in current["records"] if r["record_id"] == work["record_id"]]
            require(current_row["split"] == "train", "current TRAIN metadata required")
            require(current_row["source_component_id"] == admission["source_expected"]["component_id"], "component mismatch")
            source_bytes = authenticate(root, {"path": admission["allowed_path"], **admission["source_expected"]}, receipts)
            piece_bytes = authenticate(root, work["canonical_piece"], receipts)
            require(sha256(piece_bytes).hexdigest() == current_row["piece"]["sha256"], "current canonical piece mismatch")
            alignment_bytes = authenticate(root, work["raw_alignment"], receipts)
            prior = json.loads(authenticate(root, work["source_events"], receipts))
            score_path = f'outputs/tonal_relations_astra_20261006_001/classical/{work["case_id"]}-score.mscx'
            score_entry = {"path": score_path, **pilot_files[score_path]}
            score_bytes = authenticate(root, score_entry, receipts)
            piece, alignment = loads_piece(piece_bytes), json.loads(alignment_bytes)
            conversion, = [p for p in piece.provenance if p.provenance_id == "prov:dilemmadata-conversion"]
            offset = dict(conversion.details).get("source_to_canonical_offset_qn", "0/1")
            require(offset == work["source_to_canonical_offset_qn"], "stored canonical offset mismatch")
            rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")), delimiter="\t"))
            packet = extract_source_packet(piece, alignment, rows, score_bytes, {
                **work, "source_version_id": prior["source_version_id"], "parts": prior["parts"],
                "current_split": "train", "source_to_canonical_offset_qn": offset})
            packet["canonical_piece"] = work["canonical_piece"]
            packet["raw_alignment"] = work["raw_alignment"]
            packet["source_table"] = {"path": admission["allowed_path"], **admission["source_expected"]}
            packet["source_score"] = {k: score_entry[k] for k in ("path", "sha256", "size_bytes")}
            packet["title"] = prior["title"]
            target = output / f'{work["case_id"]}.json'
            target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
            summaries.append({"case_id": work["case_id"], "path": str(target.relative_to(root)),
                              "sha256": sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size,
                              "diagnostics": packet["diagnostics"]})
            print(json.dumps({"case_id": work["case_id"], "status": "completed", **packet["diagnostics"]}), flush=True)
        receipt = {"schema": "DenseSourceExtractionReceipt@1", "status": "passed", "attempt_id": output.name,
                   "experiment_id": active["experiment_id"], "works": summaries, "authenticated_inputs": receipts,
                   "elapsed_seconds": time.perf_counter() - started, "test_or_validation_payload_opened": False}
        (output / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        return receipt
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "attempt_id": output.name,
            "exception": type(exc).__name__, "message": str(exc), "completed_works": summaries,
            "authenticated_inputs": receipts, "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release(args.root, args.output)
