"""Acquire admitted TAVERN files and write immutable exact native targets.

Only pinned Git tree metadata and split metadata are read before family
admission. No downloaded code is executed. Each payload matches its Git blob.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from hashlib import sha1, sha256
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.request

from music_critic.adapters.tavern import (
    VERSION, TavernError, build_enrichment, canonical_piece, parse_joined,
    project_score_spines, raw_signature, source_note_bindings,
)
from music_critic.data import dumps_piece
from music_critic.tasks.event_theory_enrichment import validate_enrichment


def packed(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def binding(path):
    path = Path(path)
    b = path.read_bytes()
    return {"path": str(path), "sha256": sha256(b).hexdigest(), "size_bytes": len(b)}


def admit_families(manifest):
    """Use both current component membership and native heldout hints."""
    by_family, components = defaultdict(list), defaultdict(set)
    for row in manifest["records"]:
        components[row["source_component_id"]].add(row["split"])
        match = re.fullmatch(r"an:([^:]+):tavern-(.+)-([ab])", row["record_id"])
        if match:
            native, family, annotator = match.groups()
            by_family[family].append({"record_id": row["record_id"], "current_split": row["split"],
                                      "native_split": native, "component": row["source_component_id"]})
    result = {}
    for family, rows in sorted(by_family.items()):
        reasons = []
        if any(row["current_split"] != "train" for row in rows):
            reasons.append("current_heldout_family")
        if any(row["native_split"] != "training" for row in rows):
            reasons.append("native_heldout_hint")
        if any(components[row["component"]] != {"train"} for row in rows):
            reasons.append("component_not_exclusively_current_train")
        result[family] = {"admitted": not reasons, "reasons": reasons, "records": rows}
    return result


def family_for_path(path):
    parts = PurePosixPath(path).parts
    if len(parts) < 2:
        return None
    composer, work = parts[:2]
    if composer == "Beethoven":
        if re.fullmatch(r"B\d{3}", work):
            return f"beethoven-woo-{int(work[1:])}"
        if re.fullmatch(r"Opus\d+", work):
            return f"beethoven-op{int(work[4:])}"
    if composer == "Mozart" and re.fullmatch(r"K\d{3}", work):
        return f"mozart-{work.lower()}"
    return None


def catalog(tree, admissions):
    entries = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    groups, skipped = defaultdict(dict), []
    for path, row in sorted(entries.items()):
        if "/Joined/" not in path or not path.endswith(".krn"):
            continue
        family = family_for_path(path)
        admission = admissions.get(family)
        if not admission or not admission["admitted"]:
            skipped.append({"path": path, "family": family, "reason": admission["reasons"] if admission else ["unknown_family_split"]})
            continue
        match = re.fullmatch(r"([A-Z0-9]+_\d+_\d+)([a-z]*)_([ab])\.krn", PurePosixPath(path).name)
        if not match:
            skipped.append({"path": path, "family": family, "reason": ["unsupported_filename"]})
            continue
        phrase, content, annotator = match.groups()
        score_path = str(PurePosixPath(path).parent.parent / "Krn" / f"{phrase}_score.krn")
        if score_path not in entries:
            # Source naming varies: Wo071_V00 versus B071_00 and Bo76 versus BO76.
            variation, number = phrase.split("_")[-2:]
            parent = str(PurePosixPath(path).parent.parent / "Krn") + "/"
            candidates = [p for p in entries if p.startswith(parent)
                          and re.search(rf"_V?{re.escape(variation)}_{re.escape(number)}_score\.krn$", p, re.I)]
            score_path = candidates[0] if len(candidates) == 1 else None
        key = family + ":" + phrase
        groups[key].update({"family": family, "phrase_id": phrase, "score": entries[score_path] if score_path else None})
        groups[key].setdefault("analyses", []).append({**row, "annotator": annotator, "content_group": content})
    return dict(groups), skipped


def acquire(entry, source_root, commit, download):
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or any(ord(c) < 32 for c in str(relative)):
        raise TavernError("unsafe source path")
    path = source_root / relative
    if path.exists():
        payload = path.read_bytes()
    else:
        if not download:
            raise TavernError("source unavailable without --download")
        url = f"https://raw.githubusercontent.com/jcdevaney/TAVERN/{commit}/{relative}"
        payload = urllib.request.urlopen(url, timeout=60).read()
        expected = sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if expected != entry["sha"]:
            raise TavernError("Git source blob mismatch")
        write_new(path, payload)
    if sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() != entry["sha"]:
        raise TavernError("retained source blob mismatch")
    return path


def run(args):
    started = time.monotonic()
    output = Path(args.output)
    manifest_path, tree_path = Path(args.data_manifest), Path(args.source_tree)
    manifest, tree = json.loads(manifest_path.read_text()), json.loads(tree_path.read_text())
    if not re.fullmatch(r"[a-f0-9]{40}", tree["sha"]) or tree.get("truncated"):
        raise TavernError("complete pinned Git tree required")
    admissions = admit_families(manifest)
    groups, skipped = catalog(tree, admissions)
    selected = {}
    for group in groups.values():
        for row in ([group["score"]] if group["score"] else []) + group["analyses"]:
            selected[row["path"]] = row
    write_new(output / "admission.json", packed({"schema": "TavernAdmission@1", "source_tree": binding(tree_path),
        "split_metadata": binding(manifest_path), "commit": tree["sha"], "families": admissions,
        "selected_files": sorted(selected), "skipped": skipped}))
    failures, downloaded = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(acquire, row, Path(args.source_root), tree["sha"], args.download): name for name, row in selected.items()}
        for future in as_completed(jobs):
            name = jobs[future]
            try:
                downloaded[name] = future.result()
            except Exception as exc:
                failures.append({"stage": "download", "path": name, "reason": f"{type(exc).__name__}: {exc}"})
    records = []
    for key, group in sorted(groups.items()):
        try:
            score = None
            if group["score"]:
                try:
                    score_path = downloaded[group["score"]["path"]]
                    score_binding = binding(score_path)
                    score = parse_joined(score_path.read_text(), source_path=group["score"]["path"], source_sha256=score_binding["sha256"])
                except Exception as exc:
                    failures.append({"stage": "score", "path": group["score"]["path"], "reason": f"{type(exc).__name__}: {exc}"})
            variants = {}
            for row in group["analyses"]:
                try:
                    path = downloaded[row["path"]]
                    source_binding = binding(path)
                    text = path.read_text()
                    parsed = parse_joined(text, source_path=row["path"], source_sha256=source_binding["sha256"])
                    projection, line_map = project_score_spines(text)
                    projection_path = output / "projections" / f"{group['phrase_id']}_{row['annotator']}.krn"
                    raw = parse_joined(projection, source_path=str(projection_path), source_sha256=sha256(projection.encode()).hexdigest())
                    if raw_signature(raw) != raw_signature(parsed) or raw.duration != parsed.duration:
                        raise TavernError("Joined/analysis-free-projection note or clock disagreement")
                    identity = (tuple(raw_signature(raw)), raw.duration, tuple(raw.meters), tuple(t for t, _ in raw.barlines))
                    if identity not in variants:
                        if (score is not None and raw_signature(score) == raw_signature(raw) and score.duration == raw.duration
                                and score.meters == raw.meters and [t for t, _ in score.barlines] == [t for t, _ in raw.barlines]):
                            chosen_raw, origin = score, "native_score_only"
                            raw_binding = score_binding
                        else:
                            chosen_raw, origin = raw, "joined_score_projection"
                            raw_binding = {"path": str(projection_path), "sha256": sha256(projection.encode()).hexdigest(),
                                           "size_bytes": len(projection.encode()), "source_joined": source_binding,
                                           "projection_to_joined_lines": line_map}
                        variants[identity] = {"raw": chosen_raw, "origin": origin, "analyses": [],
                                               "source_bindings": [raw_binding], "projection": projection,
                                               "projection_path": projection_path}
                    variant = variants[identity]
                    variant["analyses"].append((row["annotator"], parsed))
                    variant["source_bindings"].append(source_binding)
                except Exception as exc:
                    failures.append({"stage": "analysis", "path": row["path"], "reason": f"{type(exc).__name__}: {exc}"})
            if not variants:
                raise TavernError("no verified native analysis")
            for variant in variants.values():
                try:
                    analyses = variant["analyses"]
                    stem = group["phrase_id"] + ("_raw-" + analyses[0][0] if len(variants) > 1 else "")
                    piece = canonical_piece(variant["raw"], phrase_id=stem, family_id="tavern-family:" + group["family"])
                    piece_path = output / "pieces" / f"{stem}.json"
                    piece_bytes = dumps_piece(piece).encode()
                    canonical_binding = {"path": str(piece_path), "sha256": sha256(piece_bytes).hexdigest(), "size_bytes": len(piece_bytes)}
                    bundle = build_enrichment(piece, analyses, canonical_binding=canonical_binding,
                                             source_bindings=variant["source_bindings"], admission=admissions[group["family"]])
                    bundle["provenance"]["raw_source_origin"] = variant["origin"]
                    bundle["provenance"]["source_note_bindings"] = source_note_bindings(piece, variant["raw"])
                    validate_enrichment(piece, bundle)
                    if variant["origin"] == "joined_score_projection":
                        write_new(variant["projection_path"], variant["projection"].encode())
                    write_new(piece_path, piece_bytes)
                    target_path = output / "targets" / f"{stem}.json.gz"
                    write_new(target_path, gzip.compress(packed(bundle), mtime=0))
                    records.append({"dataset_id": "tavern", "piece_id": piece.piece_id, "family": group["family"],
                                    "source_phrase": group["phrase_id"], "raw_variants_for_phrase": len(variants),
                                    "raw_source_origin": variant["origin"],
                                    "canonical_piece": canonical_binding, "enrichment": binding(target_path),
                                    "coverage": bundle["coverage"], "analyses": [a for a, _ in analyses],
                                    "native_function_spans": sum(len(p.functions) for _, p in analyses),
                                    "native_harmony_spans": sum(len(p.harmonies) for _, p in analyses)})
                except Exception as exc:
                    failures.append({"stage": "variant", "group": key, "reason": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            failures.append({"stage": "piece", "group": key, "reason": f"{type(exc).__name__}: {exc}"})
    result = {"schema": "TavernEnrichmentRelease@1", "adapter": VERSION, "source_commit": tree["sha"],
              "records": records, "failures": failures, "skipped": skipped,
              "counts": {"catalog_groups_admitted": len(groups), "selected_source_files": len(selected),
                         "downloaded_source_files": len(downloaded), "released_phrases": len(records),
                         "unique_source_phrases": len({r["source_phrase"] for r in records}),
                         "released_families": len({r["family"] for r in records}),
                         "canonical_notes": sum(r["coverage"]["canonical_notes"] for r in records),
                         "note_context_rows": sum(r["coverage"]["note_context_rows"] for r in records),
                         "native_function_spans": sum(r["native_function_spans"] for r in records),
                         "native_harmony_spans": sum(r["native_harmony_spans"] for r in records)},
              "elapsed_seconds": time.monotonic() - started,
              "source_bindings": {name: binding(path) for name, path in sorted(downloaded.items())},
              "test_music_read": False, "validation_music_read": False,
              "license": "CC BY-SA 4.0", "time_basis": "phrase_local_written_score_order"}
    write_new(output / "index.json", packed(result))
    print(json.dumps({"counts": result["counts"], "failures": len(failures), "output": str(output)}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be in 1..12")
    run(args)


if __name__ == "__main__":
    main()
