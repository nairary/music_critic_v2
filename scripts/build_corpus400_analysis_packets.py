#!/usr/bin/env python3
"""Lossless inspection packets for whole-file annotation, never theory inference.

Every note has one core owner. Logical time partitions cover the entire file;
oversized simultaneous groups use separately owned note shards of the same
partition. Context is bounded and explicit, including omitted context IDs.
Original event times, note IDs and native field objects never change.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path

from music_critic.tasks.event_theory_enrichment import canonical, fraction
from music_critic.tasks.whole_work_annotation import default_whole_work_profile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    value = Fraction(value)
    return {"num": value.numerator, "den": value.denominator}


def semantic_hash(value):
    return sha256(canonical(value).encode()).hexdigest()


def overlaps(left, right, start, end, duration):
    if left == right:
        return start <= left < end or left == end == duration
    return left < end and start < right


def _notes(packet):
    require(packet.get("schema") == "Corpus400FullInspectionPacket@1", "unsupported inspection schema")
    require(packet.get("all_notes_included") is True and packet.get("cropped") is False
            and packet.get("inspection_scope") == "entire_materialized_file", "full inspection required")
    columns = packet["note_columns"]
    require(len(columns) == len(set(columns)), "duplicate note column")
    require({"note_id", "track_id", "onset_qn", "duration_qn", "pitch"} <= set(columns), "note columns incomplete")
    duration = fraction(packet["duration_qn"])
    require(duration > 0, "positive piece duration required")
    tracks = {r["track_id"] for r in packet["tracks"]}
    require(len(tracks) == len(packet["tracks"]), "duplicate track identity")
    notes = {}
    for index, values in enumerate(packet["note_rows"]):
        require(len(values) == len(columns), "note row width mismatch")
        note = dict(zip(columns, values, strict=True))
        nid = note["note_id"]
        require(isinstance(nid, str) and bool(nid) and nid not in notes, "duplicate or empty note identity")
        require(note["track_id"] in tracks, "unknown note track")
        start, length = fraction(note["onset_qn"]), fraction(note["duration_qn"])
        require(0 <= start <= start + length <= duration, "note outside piece")
        notes[nid] = {"id": nid, "start": start, "end": start + length, "index": index, "note": note}
    require(packet.get("note_count") == len(notes), "inspection note_count mismatch")
    return duration, notes


def _bars(packet, duration):
    bars, cursor, identities = [], Fraction(0), set()
    for row in sorted(packet.get("bars", []), key=lambda r: fraction(r["start_qn"])):
        start = fraction(row["start_qn"])
        end = start + fraction(row["duration_qn"])
        require(cursor <= start < end <= duration, "overlapping or out-of-bounds source bars")
        bid = row["bar_id"]
        require(bid not in identities, "duplicate bar identity")
        identities.add(bid)
        if cursor < start:
            bars.append({"bar_id": f"unbarred:{len(bars)}", "start": cursor, "end": start, "source_bar": False})
        bars.append({"bar_id": bid, "start": start, "end": end, "source_bar": True})
        cursor = end
    if cursor < duration:
        bars.append({"bar_id": f"unbarred:{len(bars)}", "start": cursor, "end": duration, "source_bar": False})
    return bars


def _partitions(notes, duration, core_limit):
    groups = defaultdict(list)
    for note in notes.values():
        groups[note["start"]].append(note["id"])
    partitions, start, ids = [], Fraction(0), []
    for onset, group in sorted(groups.items()):
        if ids and len(ids) + len(group) > core_limit and start < onset < duration:
            partitions.append((start, onset, ids))
            start, ids = onset, []
        ids.extend(sorted(group))
    partitions.append((start, duration, ids))
    return partitions


def _native(bundle, packet, notes, duration):
    if bundle is None:
        return {"harmonies": [], "note_contexts": [], "objects": [], "relations": []}
    require(bundle.get("schema") == "EventTheoryEnrichment@1", "unsupported native enrichment schema")
    require(bundle.get("current_split") == "train", "native TRAIN binding required")
    source = packet["canonical_piece"]
    require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64
            and type(source.get("size_bytes")) is int and source["size_bytes"] > 0, "canonical binding required")
    require(all(bundle.get("canonical_piece", {}).get(k) == source.get(k) for k in ("sha256", "size_bytes")),
            "native canonical binding mismatch")
    geometry = {nid: (n["start"], n["end"]) for nid, n in notes.items()}
    harmonies = {}
    for kind in ("harmonies", "note_contexts", "objects"):
        for row in bundle.get(kind, []):
            rid = row["id"]
            require(rid not in geometry, "duplicate native identity")
            start, end = fraction(row["start_qn"]), fraction(row["end_qn"])
            require(0 <= start <= end <= duration, "native interval outside piece")
            if kind == "harmonies":
                require(start < end, "harmony requires positive interval")
                harmonies[rid] = (start, end)
            if kind == "note_contexts":
                require(row.get("note_id") in notes, "unknown native context note")
                note = notes[row["note_id"]]
                require(note["start"] <= start <= end <= note["end"], "native context outside note")
                hid = row.get("harmony_id")
                require(hid is None or hid in harmonies, "unknown native context harmony")
                if hid is not None:
                    require(harmonies[hid][0] <= start <= end <= harmonies[hid][1], "native context outside harmony")
            geometry[rid] = start, end
    for row in bundle.get("objects", []):
        require(all(member in geometry for member in row.get("member_ids", [])), "unknown native object member")
    for row in bundle.get("relations", []):
        require(row["id"] not in geometry, "duplicate native relation")
        participants = row.get("source_ids", []) + row.get("target_ids", [])
        require(all(p in geometry for p in participants), "unknown native relation participant")
        start, end = fraction(row["start_qn"]), fraction(row["end_qn"])
        require(0 <= start <= end <= duration, "native relation outside piece")
        geometry[row["id"]] = start, end
    return bundle


def _select_native(bundle, selected_notes, start, end, duration):
    contexts = [r for r in bundle.get("note_contexts", []) if r["note_id"] in selected_notes]
    referenced_harmonies = {r.get("harmony_id") for r in contexts}
    rows = {"note_contexts": contexts}
    for kind in ("harmonies", "objects", "relations"):
        rows[kind] = [r for r in bundle.get(kind, []) if (kind == "harmonies" and r["id"] in referenced_harmonies)
            or overlaps(fraction(r["start_qn"]), fraction(r["end_qn"]), start, end, duration)]
    fieldsets = {}
    packed = {}
    for kind, source_rows in rows.items():
        packed[kind] = []
        for row in source_rows:
            copy = deepcopy(row)
            if "fields" in copy:
                fields = copy.pop("fields")
                fid = "fieldset:" + semantic_hash(fields)[:20]
                require(fid not in fieldsets or fieldsets[fid] == fields, "native fieldset token collision")
                fieldsets[fid] = fields
                copy["fieldset_id"] = fid
            packed[kind].append(copy)
    return packed, fieldsets


def build_analysis_packets(packet, native_bundle=None, *, max_notes=1500, context_qn="4",
                           view_id="analysis:primary", author_id=None):
    """Pure full-file partitioning; no event or native field is sampled away."""
    require(type(max_notes) is int and max_notes > 0, "max_notes must be positive integer")
    context = Fraction(context_qn)
    require(context >= 0, "context span must be nonnegative")
    require(isinstance(view_id, str) and bool(view_id.strip()), "view identity required")
    duration, notes = _notes(packet)
    bars = _bars(packet, duration)
    native = _native(native_bundle, packet, notes, duration)
    profile = default_whole_work_profile()
    config = {"max_notes": max_notes, "context_qn": rational(context),
        "requested_view_id": view_id, "requested_author_id": author_id}
    lineage = {"inspection_semantic_sha256": semantic_hash(packet),
        "native_bundle_semantic_sha256": semantic_hash(native_bundle) if native_bundle is not None else None,
        "canonical_piece": deepcopy(packet["canonical_piece"]), "native_theory": deepcopy(packet.get("native_theory")),
        "source_analytical_extras": deepcopy(packet.get("source_analytical_extras")),
        "profile_semantic_sha256": semantic_hash(profile), "configuration": config}
    lineage["fingerprint"] = semantic_hash(lineage)
    reserve = 0 if max_notes == 1 else min(128, max(1, max_notes // 8))
    core_limit = max(1, max_notes - reserve)
    partitions, segments = [], []
    for ordinal, (start, end, core_ids) in enumerate(_partitions(notes, duration, core_limit)):
        pid = "partition:" + semantic_hash([lineage["fingerprint"], ordinal])[:24]
        shards = [core_ids[i:i + core_limit] for i in range(0, len(core_ids), core_limit)] or [[]]
        partitions.append({"partition_id": pid, "start_qn": rational(start), "end_qn": rational(end),
            "core_note_count": len(core_ids), "shard_count": len(shards)})
        context_start, context_end = max(Fraction(0), start - context), min(duration, end + context)
        possible = [n for n in notes.values() if overlaps(n["start"], n["end"], context_start, context_end, duration)]
        for shard_index, owned in enumerate(shards):
            owned_set = set(owned)
            contextual = [n for n in possible if n["id"] not in owned_set]
            contextual.sort(key=lambda n: (max(Fraction(0), start - n["end"], n["start"] - end), n["start"], n["id"]))
            supplied = contextual[:max_notes - len(owned)]
            omitted = contextual[max_notes - len(owned):]
            selected = owned_set | {n["id"] for n in supplied}
            ordered = sorted((notes[n] for n in selected), key=lambda n: (n["start"], n["id"]))
            native_rows, fieldsets = _select_native(native, selected, context_start, context_end, duration)
            timeline = []
            for bar in bars:
                if bar["start"] >= context_end or bar["end"] <= context_start: continue
                sounding = [n for n in ordered if overlaps(n["start"], n["end"], bar["start"], bar["end"], duration)]
                timeline.append({"bar_id": bar["bar_id"], "source_bar": bar["source_bar"],
                    "start_qn": rational(bar["start"]), "end_qn": rational(bar["end"]),
                    "visible_start_qn": rational(max(bar["start"], context_start)),
                    "visible_end_qn": rational(min(bar["end"], context_end)),
                    "note_ids": [n["id"] for n in sounding],
                    "carry_in_note_ids": [n["id"] for n in sounding if n["start"] < bar["start"]]})
            sid = "segment:" + semantic_hash([pid, shard_index])[:24]
            core_context_ids = [r["id"] for r in native_rows["note_contexts"] if r["note_id"] in owned_set]
            included_ids = selected | {r["id"] for rows in native_rows.values() for r in rows}
            participant_ids = {member for r in native_rows["objects"] for member in r.get("member_ids", [])}
            participant_ids |= {member for r in native_rows["relations"] for name in ("source_ids", "target_ids") for member in r.get(name, [])}
            core_native_ids = {kind: [r["id"] for r in native_rows[kind] if shard_index == 0 and
                (start <= fraction(r["start_qn"]) < end or fraction(r["start_qn"]) == end == duration)]
                for kind in ("harmonies", "objects", "relations")}
            segments.append({"schema": "Corpus400AnalysisSegment@1", "segment_id": sid,
                "candidate_id": packet["candidate_id"], "work_family_id": packet["work_family_id"],
                "partition_id": pid, "shard_index": shard_index, "shard_count": len(shards),
                "core_interval": {"start_qn": rational(start), "end_qn": rational(end)},
                "context_interval": {"start_qn": rational(context_start), "end_qn": rational(context_end)},
                "piece_duration_qn": rational(duration), "lineage": deepcopy(lineage),
                "core_note_ids": list(owned), "core_native_context_ids": core_context_ids,
                "core_native_ids": core_native_ids,
                "core_lifetimes_beyond_context_ids": [nid for nid in owned if notes[nid]["start"] < context_start or notes[nid]["end"] > context_end],
                "context_note_ids": [n["id"] for n in supplied],
                "context_note_coverage": {"requested_count": len(contextual), "supplied_count": len(supplied),
                    "omitted_ids": [n["id"] for n in omitted], "complete": not omitted,
                    "omission_reason": "note_budget_requires_other_segments" if omitted else None},
                "note_columns": deepcopy(packet["note_columns"]),
                "note_rows": [deepcopy(packet["note_rows"][n["index"]]) for n in ordered],
                "tracks": deepcopy(packet["tracks"]), "bar_timeline": timeline,
                "native": native_rows, "native_fieldsets": fieldsets,
                "native_external_participant_ids": sorted(participant_ids - included_ids),
                "native_status": "source_bound_fields_preserved" if native_bundle is not None else "not_supplied",
                "required_annotation_profile": profile,
                "ownership_rule": "Annotate only core note/context IDs; context repeats across segments. Cross-segment structures require whole-work synthesis.",
                "time_axis": "canonical_quarter_notes; original note intervals are not clipped to packet bounds",
                "semantic_inference_performed": False})
    owner_counts = Counter(n for segment in segments for n in segment["core_note_ids"])
    require(owner_counts == Counter(notes.keys()), "internal note ownership mismatch")
    owners = {nid: segment["segment_id"] for segment in segments for nid in segment["core_note_ids"]}
    entities = {"schema": "Corpus400AnalysisEntityIndex@1", "candidate_id": packet["candidate_id"],
        "lineage_fingerprint": lineage["fingerprint"], "tracks": deepcopy(packet["tracks"]),
        "notes": [{"note_id": n["id"], "start_qn": rational(n["start"]), "end_qn": rational(n["end"]),
                   "owner_segment_id": owners[n["id"]]} for n in notes.values()],
        "native_ids": {k: [r["id"] for r in native.get(k, [])] for k in ("harmonies", "note_contexts", "objects", "relations")}}
    overview = {"schema": "Corpus400AnalysisOverview@1", "candidate_id": packet["candidate_id"],
        "work_family_id": packet["work_family_id"], "primary_genre": packet.get("primary_genre"),
        "identity": deepcopy(packet.get("identity", {})), "source_extent": deepcopy(packet.get("source_extent", {})),
        "source_analytical_extras": deepcopy(packet.get("source_analytical_extras")),
        "lineage": lineage, "duration_qn": rational(duration), "note_count": len(notes),
        "tracks": deepcopy(packet["tracks"]), "bar_count": len(packet.get("bars", [])),
        "meter_events": deepcopy(packet.get("meter_events", [])),
        "key_signature_events": deepcopy(packet.get("key_signature_events", [])),
        "partition_count": len(partitions), "segment_count": len(segments),
        "all_notes_have_one_core_owner": True, "all_time_partitioned": True,
        "entity_index_semantic_sha256": semantic_hash(entities), "required_annotation_profile": profile,
        "analytical_sources": deepcopy(packet.get("analytical_sources", [])),
        "annotation_complete": False, "semantic_inference_performed": False,
        "limits": ["Full file inspection does not verify complete work identity",
            "Written versus performed/repeat-expanded time is not inferred; inspect source conversion receipts",
            "Packet boundaries and shards are computational, not musical phrases or sections",
            "Budgeted context may omit events; explicit IDs identify the other packets needed for synthesis"]}
    return {"overview": overview, "partitions": partitions, "segments": segments, "entity_index": entities}


def qtext(value):
    value = fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def render_segment_markdown(segment):
    """Complete readable timeline, with an explicit dictionary for native fields."""
    core, context = segment["core_interval"], segment["context_interval"]
    lines = [f"# {segment['segment_id']}", "",
        f"Core [{qtext(core['start_qn'])}, {qtext(core['end_qn'])}); context [{qtext(context['start_qn'])}, {qtext(context['end_qn'])}).",
        f"Time unit: canonical quarter note. Whole file: {qtext(segment['piece_duration_qn'])}. Shard {segment['shard_index'] + 1}/{segment['shard_count']}.",
        "CORE marks exclusive note ownership. CONTEXT is shared. Original note durations are preserved even outside this packet's interval.",
        "Written/performed time and expanded repeats must be checked against source receipts; no interpretation is inferred here.",
        f"Lineage: {segment['lineage']['fingerprint']}",
        f"Requested view: {segment['lineage']['configuration']['requested_view_id']}",
        "", "## Parts", canonical(segment["tracks"]), "", "## Exact timeline by source bar"]
    # note_id need not be the first input column; repeated track IDs must not
    # collapse rows before the true identity column is located.
    rows = [dict(zip(segment["note_columns"], row, strict=True)) for row in segment["note_rows"]]
    notes = {n["note_id"]: n for n in rows}
    native_contexts = defaultdict(list)
    for row in segment["native"]["note_contexts"]: native_contexts[row["note_id"]].append(row["id"])
    core_ids = set(segment["core_note_ids"])
    for bar in segment["bar_timeline"]:
        lines += ["", f"### {bar['bar_id']} [{qtext(bar['start_qn'])}, {qtext(bar['end_qn'])})" + (" (unbarred interval)" if not bar["source_bar"] else "")]
        if not bar["note_ids"]: lines.append("No supplied notes in this interval; consult context omission accounting below.")
        for nid in bar["note_ids"]:
            n = notes[nid]
            extras = {key: n[key] for key in ("spelling_step", "spelling_alter", "staff", "voice", "is_percussion", "is_grace") if key in n}
            lines.append(f"- {'CORE' if nid in core_ids else 'CONTEXT'} {nid} | part={n['track_id']} | pitch={n['pitch']} | onset={qtext(n['onset_qn'])} duration={qtext(n['duration_qn'])} | {canonical(extras)} | native_contexts={canonical(native_contexts[nid])}")
    lines += ["", "## Context accounting", canonical(segment["context_note_coverage"]),
        "Core note lifetimes extending beyond supplied context: " + canonical(segment["core_lifetimes_beyond_context_ids"]),
        "External native participant IDs: " + canonical(segment["native_external_participant_ids"])]
    for kind, rows in segment["native"].items():
        lines += ["", "## Native " + kind]
        if not rows: lines.append("No supplied rows; this is not a reviewed absence.")
        for row in rows: lines.append(canonical(row))
    lines += ["", "## Native field dictionary (lossless)"]
    for fid, fields in sorted(segment["native_fieldsets"].items()):
        lines += ["", fid]
        for name, field in sorted(fields.items()): lines.append(f"- {name}: {canonical(field)}")
    lines += ["", "## Required analysis profile", canonical(segment["required_annotation_profile"]),
        "", segment["ownership_rule"], "", "No semantic inference or completion claim is made by packetization."]
    return "\n".join(lines) + "\n"


def _save(path, raw):
    path = Path(path)
    with path.open("xb") as stream: stream.write(raw)
    return {"path": str(path), "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _save_json(path, value):
    return _save(path, (canonical(value) + "\n").encode())


def write_analysis_packets(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "segments").mkdir()
    overview = _save_json(output / "overview.json", result["overview"])
    entities = _save_json(output / "entity-index.json", result["entity_index"])
    partitions = _save_json(output / "partitions.json", result["partitions"])
    short = result["overview"]
    overview_text = [f"# {short['candidate_id']}", "", canonical(short["identity"]),
        f"{short['note_count']} notes; {short['bar_count']} source bars; duration {qtext(short['duration_qn'])} canonical quarter notes.",
        f"{short['partition_count']} time partitions, {short['segment_count']} note-budget segments; every note has one core owner.",
        "Full entity IDs and ownership: entity-index.json. Exact temporal boundaries: partitions.json.",
        "Original analytical extras (binding only; not loaded, interpreted or time-aligned here): " + canonical(short.get("source_analytical_extras")),
        "", "## Parts", canonical(short["tracks"]), "", "## Meter events", canonical(short["meter_events"]),
        "", "## Key signature events (source metadata)", canonical(short["key_signature_events"]),
        "", "## Source extent", canonical(short["source_extent"]), "", "## Limits", *short["limits"]]
    overview_markdown = _save(output / "overview.md", ("\n".join(overview_text) + "\n").encode())
    with (output / "segments/index.jsonl").open("x", encoding="utf-8") as index:
        for ordinal, segment in enumerate(result["segments"]):
            json_binding = _save_json(output / f"segments/{ordinal:04d}.json", segment)
            markdown = _save(output / f"segments/{ordinal:04d}.md", render_segment_markdown(segment).encode())
            index.write(canonical({"segment_id": segment["segment_id"], "partition_id": segment["partition_id"],
                "core_interval": segment["core_interval"], "core_note_count": len(segment["core_note_ids"]),
                "supplied_note_count": len(segment["note_rows"]), "json": json_binding, "markdown": markdown}) + "\n")
    summary = {"schema": "Corpus400AnalysisPacketResult@1", "status": "packets_built",
        "candidate_id": short["candidate_id"], "work_family_id": short["work_family_id"],
        "note_count": short["note_count"], "segment_count": len(result["segments"]),
        "partition_count": len(result["partitions"]), "overview": overview, "overview_markdown": overview_markdown,
        "entity_index": entities, "partitions": partitions,
        "lineage_fingerprint": short["lineage"]["fingerprint"], "annotation_complete": False,
        "note_and_time_partition_complete": True}
    _save_json(output / "summary.json", summary)
    return summary


def _read_bound(binding):
    raw = Path(binding["path"]).read_bytes()
    require(len(raw) == binding["size_bytes"] and sha256(raw).hexdigest() == binding["sha256"],
            "input binding hash/size mismatch")
    return raw


def run(materialized_index, output, experiment_record, *, max_notes=1500, context_qn="4",
        view_id="analysis:primary", author_id=None):
    """Authenticate a registered materialization index before any packet reads."""
    from music_critic.experiments.registry import load_record
    record = load_record(experiment_record)
    require(record["status"] in {"planned", "running"} and record["phase"] == "tonal-relations-T08"
            and record["kind"] == "corpus400_analysis_packetization", "active T08 packetization record required")
    expected = record.get("protocol", {}).get("materialized_index")
    require(isinstance(expected, dict) and {"path", "sha256", "size_bytes"} <= expected.keys(),
            "registered materialized_index binding required")
    require(Path(expected["path"]).resolve() == Path(materialized_index).resolve(), "materialization index path mismatch")
    index_bytes = _read_bound(expected)
    rows = [json.loads(line) for line in index_bytes.decode().splitlines() if line.strip()]
    good = [row for row in rows if row.get("status") == "raw_materialized"]
    require(len({row["candidate_id"] for row in good}) == len(good), "duplicate materialized candidate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "input-materialization-index.jsonl", index_bytes)
    _save(output / "active-experiment-record.json", Path(experiment_record).read_bytes())
    _save(output / "packetizer-source.py", Path(__file__).read_bytes())
    successes, failures = 0, 0
    with (output / "index.jsonl").open("x", encoding="utf-8") as ledger:
        for row in good:
            try:
                packet = json.loads(_read_bound(row["inspection_packet"]))
                require(packet["candidate_id"] == row["candidate_id"]
                        and packet["canonical_piece"] == row["canonical_piece"], "inspection/materialization identity mismatch")
                require(packet.get("native_theory") == row.get("native_theory"), "native source binding differs from materialization")
                require(packet.get("source_analytical_extras") == row.get("source_analytical_extras"), "source analytical extras binding differs from materialization")
                native = None
                if row.get("native_theory"):
                    payload = _read_bound(row["native_theory"])
                    if str(row["native_theory"]["path"]).endswith(".gz"): payload = gzip.decompress(payload)
                    native = json.loads(payload)
                result = build_analysis_packets(packet, native, max_notes=max_notes,
                    context_qn=context_qn, view_id=view_id, author_id=author_id)
                receipt = write_analysis_packets(output / "works" / sha256(row["candidate_id"].encode()).hexdigest()[:24], result)
                receipt["input_inspection_packet"] = row["inspection_packet"]
                receipt["input_native_theory"] = row.get("native_theory")
                successes += 1
            except (ValueError, OSError, KeyError, TypeError) as exc:
                receipt = {"candidate_id": row["candidate_id"], "status": "failed",
                    "error": type(exc).__name__ + ": " + str(exc)}
                failures += 1
            ledger.write(canonical(receipt) + "\n"); ledger.flush()
    summary = {"schema": "Corpus400AnalysisPacketCampaign@1", "experiment_id": record["experiment_id"],
        "success": successes, "failed": failures, "excluded_nonmaterialized_rows": len(rows) - len(good),
        "annotation_complete": 0, "semantic_inference_performed": False,
        "configuration": {"max_notes": max_notes, "context_qn": context_qn, "view_id": view_id, "author_id": author_id}}
    _save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-record", type=Path, required=True)
    parser.add_argument("--max-notes", type=int, default=1500)
    parser.add_argument("--context-qn", default="4")
    parser.add_argument("--view-id", default="analysis:primary")
    parser.add_argument("--author-id")
    args = parser.parse_args()
    summary = run(args.materialized_index, args.output, args.experiment_record,
        max_notes=args.max_notes, context_qn=args.context_qn, view_id=args.view_id, author_id=args.author_id)
    print(canonical(summary))


if __name__ == "__main__":
    main()
