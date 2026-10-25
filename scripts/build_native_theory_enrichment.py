#!/usr/bin/env python3
"""Publish immutable native TRAIN contextual targets with metadata admission.

Only index-listed, native/current TRAIN canonical files are decoded. This is a
source projection, not automatic expert harmonic analysis or an alignment search.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import traceback
import unicodedata

from music_critic.analysis.native_enrichment import build_native_enrichment
from music_critic.data import loads_piece
from music_critic.tasks.event_theory_enrichment import validate_enrichment


PRIORITY = ("LAgQZNyExyO", "AQodOweXxDl", "_NgbbvpYgQA", "dPoDnVRRgnM",
            "QLgnkVQpx-V", "qveoYqDGodn", "dZbgOqbbonY", "nLgaeDadgYp")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


def safe_child(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("canonical path escapes dataset root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("canonical symlink escapes root")
    return path


def normalized(value):
    if not isinstance(value, str): return ""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def family(metadata):
    info = metadata.get("metadata", {}).get("hooktheory", {})
    artist, title = normalized(info.get("artist")), normalized(info.get("song"))
    return (artist, title) if artist and title else None


def build_admissions(indexes, split, metadata):
    """ID and work-family checks complete before any musical file is read."""
    fingerprints = dict(split["index_fingerprints"])
    assignments = {}
    component_splits = defaultdict(set)
    for row in split["assignments"]:
        if row["piece_id"] in assignments: raise ValueError("duplicate split assignment")
        assignments[row["piece_id"]] = row
        component_splits[row["component_fingerprint"]].add(row["split"])
    native_family_splits, current_family_splits = defaultdict(set), defaultdict(set)
    for clip_id, row in metadata.items():
        key = family(row)
        if key is not None: native_family_splits[key].add(row.get("metadata", {}).get("split"))
    all_records = []
    for index in indexes:
        ds = index["header"]["dataset_id"]
        if ds not in ("hooktheory", "pop909_cl"):
            raise ValueError("unsupported index dataset")
        if fingerprints.get(ds) != index["header"]["index_fingerprint"]:
            raise ValueError("split/index fingerprint mismatch")
        for record in index["records"]:
            if record["dataset_id"] != ds: raise ValueError("mixed dataset index")
            all_records.append(record)
            if ds == "hooktheory":
                key = family(metadata.get(record["source_identity"], {}))
                if key is not None:
                    current_family_splits[key].add(assignments.get(record["piece_id"], {}).get("split"))
    if len({r["piece_id"] for r in all_records}) != len(all_records):
        raise ValueError("duplicate index identities")
    decisions = []
    for record in all_records:
        ds, identity = record["dataset_id"], record["piece_id"]
        assignment = assignments.get(identity, {})
        current = assignment.get("split")
        reasons, native, work = [], None, None
        if current != "train": reasons.append("current_split_not_train")
        if assignment and assignment.get("dataset_id") != ds: reasons.append("assignment_dataset_mismatch")
        if assignment and component_splits[assignment["component_fingerprint"]] != {"train"}:
            reasons.append("component_not_exclusively_train")
        if ds == "hooktheory":
            source = metadata.get(record["source_identity"], {})
            native = source.get("metadata", {}).get("split")
            work = family(source)
            if native != "TRAIN": reasons.append("native_split_not_train")
            if record.get("suggested_split") != "train": reasons.append("index_native_split_not_train")
            if work is None: reasons.append("missing_work_family")
            else:
                if native_family_splits[work] != {"TRAIN"}: reasons.append("native_family_heldout")
                if current_family_splits[work] != {"train"}: reasons.append("current_family_heldout")
        decisions.append({"piece_id": identity, "dataset_id": ds,
            "source_identity": record["source_identity"], "current_split": current,
            "native_split": native, "work_family": list(work) if work else None,
            "component_fingerprint": assignment.get("component_fingerprint"),
            "admitted": not reasons, "reasons": reasons, "record": record})
    return decisions


def queue_selection(decisions, metadata, count=32):
    admitted = [d for d in decisions if d["admitted"]]
    ht = [d for d in admitted if d["dataset_id"] == "hooktheory"]
    pop = [d for d in admitted if d["dataset_id"] == "pop909_cl"]
    preference = {identity: i for i, identity in enumerate(PRIORITY)}
    # One clip per artist/song family; otherwise lexical selection is reproducible,
    # not an assertion of genre balance or representative coverage.
    ht.sort(key=lambda d: (preference.get(d["source_identity"], 100), d["source_identity"]))
    chosen, works, artists = [], set(), set()
    for d in ht if count > 8 else []:
        work = tuple(d["work_family"])
        if work in works: continue
        if work[0] in artists and d["source_identity"] not in preference: continue
        chosen.append(d["piece_id"]); works.add(work); artists.add(work[0])
        if len(chosen) >= max(0, count - 8): break
    chosen.extend(d["piece_id"] for d in pop[:max(0, count - len(chosen))])
    return set(chosen)


def export_native(indexes, split, metadata, roots, output, source_bindings, *, queue_count=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    decisions = build_admissions(indexes, split, metadata)
    queue_ids = queue_selection(decisions, metadata, queue_count)
    (output / "pieces").mkdir()
    (output / "astra_queue").mkdir()
    with (output / "admission.jsonl").open("x", encoding="utf-8") as stream:
        for decision in decisions: stream.write(canonical(decision) + "\n")
    dump_new(output / "metadata-bindings.json", source_bindings)
    totals, per_dataset, skip_reasons = Counter(), defaultdict(Counter), Counter()
    with (output / "index.jsonl").open("x", encoding="utf-8") as index_out, \
         (output / "failures.jsonl").open("x", encoding="utf-8") as failures, \
         (output / "astra_queue" / "index.jsonl").open("x", encoding="utf-8") as queue_out:
        # Selected packets become available early without a second musical scan.
        for ordinal, decision in enumerate(sorted(decisions, key=lambda d: d["piece_id"] not in queue_ids)):
            ds = decision["dataset_id"]
            totals["records"] += 1; per_dataset[ds]["records"] += 1
            if not decision["admitted"]:
                totals["skipped"] += 1; per_dataset[ds]["skipped"] += 1
                skip_reasons.update(decision["reasons"])
                continue
            record = decision["record"]
            admission = {k: v for k, v in decision.items() if k != "record"}
            try:
                path = safe_child(roots[ds], record["canonical_relative_path"])
                raw = path.read_bytes()
                digest = sha256(raw).hexdigest()
                if digest != record["canonical_sha256"]: raise ValueError("canonical hash mismatch")
                piece = loads_piece(raw.decode("utf-8"))
                if piece.piece_id != decision["piece_id"] or piece.dataset_name != ds:
                    raise ValueError("canonical identity mismatch")
                if piece.split not in (None, "train"): raise ValueError("canonical split conflict")
                source = json.loads(raw)
                b = {"path": str(path), "sha256": digest, "size_bytes": len(raw)}
                bundle = build_native_enrichment(source, canonical_binding=b,
                    source_bindings=source_bindings, admission=admission)
                validate_enrichment(piece, bundle)
                name = ds + "-" + sha256(piece.piece_id.encode()).hexdigest()[:24] + ".json.gz"
                dest = output / "pieces" / name
                payload = gzip.compress((canonical(bundle) + "\n").encode(), compresslevel=6, mtime=0)
                with dest.open("xb") as stream: stream.write(payload)
                entry = {"status": "completed", "piece_id": piece.piece_id, "dataset_id": ds,
                    "bundle": {"path": str(dest), "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload), "compression": "gzip"}, "coverage": bundle["coverage"],
                    "canonical_piece": b, "source_identity": decision["source_identity"],
                    "current_split": "train", "native_split": decision["native_split"],
                    "work_family": decision["work_family"]}
                if piece.piece_id in queue_ids:
                    info = metadata.get(decision["source_identity"], {}).get("metadata", {}) if ds == "hooktheory" else {}
                    packet = {"schema": "NativeTheoryAstraQueue@1", "piece_id": piece.piece_id,
                        "dataset_id": ds, "admission": admission, "canonical_piece": b,
                        "native_bundle": entry, "identity_metadata": info,
                        "source_metadata": source.get("metadata", {}), "genre": None,
                        "genre_unknown_reason": "no_work_level_genre_provenance",
                        "notes": source["notes"], "tracks": source["tracks"],
                        "bars": source["bars"], "harmonies": bundle["harmonies"],
                        "key_regions": bundle["key_regions"], "note_contexts": bundle["note_contexts"],
                        "objects": bundle.get("objects", []),
                        "source_quality_flags": source.get("quality_flags", []),
                        "scope": "complete canonical clip/piece, not necessarily full composition",
                        "instructions": "Analyse the complete progression: preparation, prolongation, local/phrase goals and cadential roles, where the context supports them. Authored analysis must remain separate from native fields. Identify exact harmony and note participants; preserve unknown, alternatives and source limitations. Do not invent accompaniment, voice continuity, applied targets or scalar tension."}
                    queue_path = output / "astra_queue" / name
                    with queue_path.open("xb") as stream:
                        stream.write(gzip.compress((canonical(packet) + "\n").encode(), compresslevel=6, mtime=0))
                    queue_out.write(canonical({**binding(queue_path), "compression": "gzip",
                        "piece_id": piece.piece_id, "dataset_id": ds,
                        "source_identity": decision["source_identity"], "work_family": decision["work_family"],
                        "note_count": len(piece.notes), "harmony_count": len(bundle["harmonies"])}) + "\n")
                    queue_out.flush(); totals["astra_queue_pieces"] += 1
                index_out.write(canonical(entry) + "\n"); index_out.flush()
                totals["completed"] += 1; per_dataset[ds]["completed"] += 1
                for key, value in bundle["coverage"].items():
                    totals[key] += value; per_dataset[ds][key] += value
                totals["compressed_bytes"] += len(payload)
            except Exception as exc:
                # Preserve one informative failure and continue independent pieces.
                totals["failed"] += 1; per_dataset[ds]["failed"] += 1
                failures.write(canonical({"piece_id": decision["piece_id"], "dataset_id": ds,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(), "admission": admission}) + "\n")
                failures.flush()
            if (ordinal + 1) % 1000 == 0:
                print(canonical({"processed": ordinal + 1, "completed": totals["completed"],
                    "failed": totals["failed"], "elapsed_seconds": time.monotonic() - start}), flush=True)
    for key in ("completed", "failed", "skipped", "note_count", "context_rows", "astra_queue_pieces"):
        totals.setdefault(key, 0)
    summary = {"schema": "NativeTheoryEnrichmentSummary@1", **dict(totals),
        "per_dataset": {k: dict(v) for k,v in per_dataset.items()},
        "skip_reasons_nonexclusive": dict(skip_reasons), "elapsed_seconds": time.monotonic() - start,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not totals["failed"] else "completed_with_piece_failures",
        "limitations": ["No expert/Astra semantic functions added by this native batch.",
            "Metadata family matching is conservative exact normalized artist/song, not global music deduplication.",
            "All admitted notes accounted for; unavailable context is not a negative label.",
            "No native/current heldout musical files opened. Genre remains unknown absent provenance."]}
    dump_new(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/cache/global.split.json"))
    parser.add_argument("--hook-index", type=Path, default=Path("data/cache/hooktheory.index.json"))
    parser.add_argument("--pop-index", type=Path, default=Path("data/cache/pop909_cl.index.json"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/corpus_overlap_20261005_001/hooktheory-clips.jsonl"))
    parser.add_argument("--hook-root", type=Path, default=Path("data/cache/hooktheory/multisource-cache-v1.0.0"))
    parser.add_argument("--pop-root", type=Path, default=Path("data/cache/pop909_cl/multisource-cache-v1.0.0"))
    parser.add_argument("--planned-record", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.planned_record.read_text())
    if record.get("experiment_id") != "EXP-MULTIGENRE-THEORY-ENRICHMENT-001" or record.get("status") not in ("planned", "running"):
        raise ValueError("matching planned/running experiment record required")
    inputs = [args.split, args.hook_index, args.pop_index, args.metadata, args.planned_record]
    bindings = [binding(p) for p in inputs]
    bindings.extend(binding(p) for p in (Path(__file__), Path("src/music_critic/analysis/native_enrichment.py"),
                                       Path("src/music_critic/tasks/event_theory_enrichment.py")))
    indexes = [json.loads(p.read_text()) for p in (args.hook_index, args.pop_index)]
    split, metadata = json.loads(args.split.read_text()), {}
    for line in args.metadata.read_text().splitlines():
        row = json.loads(line)
        if row["clip_id"] in metadata: raise ValueError("duplicate metadata clip ID")
        metadata[row["clip_id"]] = row
    summary = export_native(indexes, split, metadata,
        {"hooktheory": args.hook_root, "pop909_cl": args.pop_root}, args.output, bindings)
    print(canonical(summary), flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
