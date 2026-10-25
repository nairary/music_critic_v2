#!/usr/bin/env python3
"""Rank a metadata reservoir, without opening any referenced musical payload.

Candidate and family IDs are metadata identities, never an admission to read
music or proof of full-work/genre/annotation completeness. Referenced file
availability uses stat only. Only explicit input metadata files are hashed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import unicodedata


EXPERIMENT_ID = "EXP-CORPUS400-CATALOG-001"
GENRES = ("classical", "jazz", "rock", "pop")
MISSING = {"", "na", "n/a", "none", "null", "nan"}
ROOTS = {
    "pdmx": "data/pdmx",
    "hooktheory": "data/cache/hooktheory/multisource-cache-v1.0.0",
    "pop909": "data/cache/pop909_cl/multisource-cache-v1.0.0",
    "dilemmadata": "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def text(value):
    if value is None:
        return None
    result = str(value).strip()
    return result if result.casefold() not in MISSING else None


def normalized(value):
    value = unicodedata.normalize("NFKC", text(value) or "").casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def identity(prefix, values):
    return prefix + sha256(canonical(values).encode()).hexdigest()[:24]


def family_id(creator, title):
    return identity("family:", [normalized(creator), normalized(title)])


def split_name(value):
    value = text(value)
    if value is None:
        return None
    return {"valid": "validation", "val": "validation"}.get(value.casefold(), value.casefold())


def unique_rows(rows, key):
    result = {}
    for row in rows:
        identifier = row[key]
        if identifier in result:
            raise ValueError("duplicate metadata identity: " + str(identifier))
        result[identifier] = row
    return result


def aggregate_hook_families(clips, assignments):
    """Use every source clip, plus all component assignments, not just pairs."""
    assigned = unique_rows(assignments, "piece_id")
    component_splits = defaultdict(set)
    for row in assigned.values():
        if row.get("component_fingerprint"):
            component_splits[row["component_fingerprint"]].add(split_name(row.get("split")))
    grouped, by_clip = defaultdict(list), {}
    for row in clips:
        cid = row["clip_id"]
        if cid in by_clip:
            raise ValueError("duplicate HookTheory clip: " + cid)
        meta = row.get("metadata", {})
        source = meta.get("hooktheory", {})
        artist, song = source.get("artist"), source.get("song")
        fid = family_id(artist, song) if text(artist) and text(song) else identity("unknown-family:", ["hooktheory", cid])
        by_clip[cid] = fid
        current = assigned.get("piece:hooktheory-" + cid, {})
        native, split = split_name(meta.get("split")), split_name(current.get("split"))
        grouped[fid].append({"clip_id": cid, "creator": text(artist), "title": text(song),
            "native_split": native, "current_split": split,
            "component_splits": sorted(v for v in component_splits.get(current.get("component_fingerprint"), ()) if v)})
    families = {}
    for fid, rows in grouped.items():
        native = sorted({r["native_split"] for r in rows if r["native_split"]})
        current = sorted({r["current_split"] for r in rows if r["current_split"]})
        heldout = any(s in {"validation", "test"} for r in rows
                      for s in [r["native_split"], r["current_split"], *r["component_splits"]])
        families[fid] = {"work_family_id": fid, "creator": rows[0]["creator"], "title": rows[0]["title"],
            "clip_ids": sorted(r["clip_id"] for r in rows), "clip_count": len(rows),
            "native_splits": native, "current_splits": current,
            "known_heldout": heldout,
            "native_current_conflict": any(r["native_split"] and r["current_split"] and
                                            r["native_split"] != r["current_split"] for r in rows),
            "all_native_current_train": all(r["native_split"] == r["current_split"] == "train"
                                            for r in rows) and not heldout,
            "membership_scope": "all_projected_hooktheory_clips_and_current_components"}
    return families, by_clip


def values(record, field):
    return list(dict.fromkeys(v for item in record.get(field, [])
                             if (v := text(item.get("value")))))


def compact_binding(value):
    return {k: value[k] for k in ("path", "sha256", "size_bytes") if k in value}


def membership_rows(record):
    membership = record.get("membership", {})
    rows = membership.get("records") or [membership]
    result = []
    for row in rows:
        native = split_name(row.get("native_split"))
        current = split_name(row.get("current_split", row.get("current_global_split")))
        archive = split_name(row.get("saved_profile_split", row.get("archive_split")))
        result.append({"source": "projected_catalog_membership", "source_id": row.get("source_id", record["id"]),
            "native_split": native, "current_split": current, "archive_split": archive,
            "native_current_conflict": native != current if native and current else None})
    return result


def _base(record):
    dataset, sid = record["dataset"], record["id"]
    titles, creators = values(record, "titles"), values(record, "creators")
    known = bool(titles and creators)
    fid = family_id(creators[0], titles[0]) if known else identity("unknown-family:", [dataset, sid])
    meta = record.get("source_metadata", {})
    bindings = {}
    for role, binding in record.get("source_bindings", {}).items():
        if role not in {"canonical_piece", "theory_bundle", "source_json", "original_midi", "source_score"}:
            raise ValueError("unsupported metadata binding role: " + role)
        bindings[role] = compact_binding(binding)
    if dataset == "pdmx":
        for field, role in (("path", "source_json"), ("mid", "original_midi"), ("mxl", "source_score")):
            if text(meta.get(field)):
                bindings[role] = {"path": str(Path(ROOTS[dataset]) / meta[field])}
    available = record.get("availability", {})
    if text(available.get("path")) and dataset in ROOTS:
        bindings["projected_source"] = {"path": str(Path(ROOTS[dataset]) / available["path"])}
    genre_evidence = []
    for field in ("genres", "tags"):
        raw = text(meta.get(field))
        if raw:
            tags = {normalized(t) for t in re.split(r"[,;|]", raw)}
            for genre in GENRES:
                if genre in tags:
                    genre_evidence.append({"genre": genre, "status": "candidate_only",
                        "evidence": {"source": dataset, "field": field, "value": raw}})
    source_genre = "pop" if dataset == "pop909" else "classical" if dataset in {"dilemmadata", "tavern"} else None
    if source_genre:
        genre_evidence.append({"genre": source_genre, "status": "candidate_only",
            "evidence": {"source": dataset, "field": "source_repertoire", "value": "corpus_context_unverified_per_work"}})
    return {"schema": "Corpus400Candidate@1", "candidate_id": identity("candidate:", [dataset, sid]),
        "dataset": dataset, "source_id": sid, "work_family_id": fid,
        "titles": titles, "creators": creators,
        "identity_confidence": "source_metadata_unverified" if known else "missing_title_or_creator",
        "genre_candidates": genre_evidence, "source_bindings": bindings,
        "split_evidence": membership_rows(record), "analytical_sources": [], "files": [],
        "full_work_status": "known_clip" if dataset == "hooktheory" else "unverified",
        "complete_for_declared_scope": False,
        "admission": {"state": "supplemental_admission_required", "reasons": [],
            "payload_read_allowed": False, "inherited_receipts": [], "hook_family_ids": []},
        "source_metadata": {k: meta[k] for k in ("license", "license_url", "license_conflict", "n_tracks", "n_notes",
            "song_length.seconds", "song_length.bars", "best_path", "best_arrangement", "best_unique_arrangement",
            "collection", "movement", "score_identity") if k in meta}}


def pdmx_record(row):
    """Project the parent stat-only PDMX CSV reservoir; never embed its full row."""
    if row.get("dataset"):
        return row
    if not text(row.get("path")):
        raise ValueError("PDMX metadata path missing")
    return {"dataset": "pdmx", "id": row["path"],
        "titles": [{"field": k, "value": row[k]} for k in ("song_name", "title") if text(row.get(k))],
        "creators": [{"field": k, "value": row[k]} for k in ("artist_name", "composer_name") if text(row.get(k))],
        "source_metadata": row}


def build_candidates(*, release_rows=(), hook_clips=(), assignments=(), catalogs=(), pairs=(), extra_records=()):
    """Pure metadata joins. Output does not grant payload admission."""
    hook_families, clip_families = aggregate_hook_families(hook_clips, assignments)
    assigned = unique_rows(assignments, "piece_id")
    catalog = {}
    for row in catalogs:
        key = row["dataset"], row["id"]
        if key in catalog:
            raise ValueError("duplicate catalog identity: " + str(key))
        catalog[key] = row
    catalog_heldout_families = set()
    for record in catalog.values():
        titles, creators = values(record, "titles"), values(record, "creators")
        if titles and creators and any(split_name(row.get(k)) in {"test", "validation"}
                for row in membership_rows(record) for k in ("native_split", "current_split", "archive_split")):
            catalog_heldout_families.add(family_id(creators[0], titles[0]))
    candidates = {}

    def add(record):
        key = record["dataset"], record["id"]
        if key not in candidates:
            candidates[key] = _base(record)
        else:
            # Later projections may add tags/locators; never overwrite a known binding.
            newer = _base(record); existing = candidates[key]
            for name, value in newer["source_bindings"].items():
                if name in existing["source_bindings"] and existing["source_bindings"][name] != value:
                    raise ValueError("conflicting source binding: " + str(key))
                existing["source_bindings"][name] = value
            for field in ("genre_candidates", "split_evidence"):
                existing[field] = _unique(existing[field] + newer[field])
            existing["source_metadata"].update(newer["source_metadata"])
        return candidates[key]

    for row in extra_records:
        add(row)
    for row in release_rows:
        dataset = {"pop909_cl": "pop909"}.get(row["dataset_id"], row["dataset_id"])
        meta = row.get("source_metadata", {})
        sid = meta.get("record_id") or meta.get("source_identity")
        if sid is None and dataset == "pop909":
            sid = row["piece_id"].removeprefix("piece:pop909-cl-")
        sid = sid or row["piece_id"]
        source = catalog.get((dataset, sid))
        if source is None:
            family = meta.get("work_family") or []
            source = {"dataset": dataset, "id": sid,
                "titles": [{"value": family[1]}] if len(family) == 2 else [],
                "creators": [{"value": family[0]}] if len(family) == 2 else []}
        candidate = add(source)
        for old, new in (("canonical_piece", "canonical_piece"), ("bundle", "theory_bundle")):
            if row.get(old): candidate["source_bindings"][new] = compact_binding(row[old])
        candidate["admission"]["inherited_receipts"].append({"source": "T07_release_index",
            "index": row.get("input_index"), "piece_id": row["piece_id"],
            "current_split": row.get("current_split"),
            "inherited_admission": meta.get("admission"),
            "augmentation_split": meta.get("augmentation_split"),
            "original_split": meta.get("original_split"), "global_membership": meta.get("global_membership")})
        candidate["split_evidence"].append({"source": "T07_release_index", "native_split": None,
            "current_split": split_name(row.get("current_split")), "piece_id": row["piece_id"]})
        if row["piece_id"] in assigned:
            candidate["split_evidence"].append({"source": "current_global_split", "native_split": None,
                "current_split": split_name(assigned[row["piece_id"]].get("split")), "piece_id": row["piece_id"]})
        if dataset == "hooktheory" and sid in clip_families:
            candidate["admission"]["hook_family_ids"].append(clip_families[sid])

    for pair in pairs:
        left, right = pair["left"], pair["right"]
        candidate = add(right)
        trusted = pair.get("creator_evidence") == "exact_normalized_creator"
        match = "exact_title_creator_candidate" if trusted else "title_only_unverified"
        candidate["analytical_sources"].append({"dataset": left["dataset"], "id": left["id"],
            "titles": values(left, "titles"), "creators": values(left, "creators"),
            "identity_match": match, "music_alignment": "not_attempted",
            "metadata_ref": pair.get("metadata_ref")})
        if left["dataset"] in {"jht", "wjd", "jaah", "chord_jazzification"}:
            candidate["genre_candidates"].append({"genre": "jazz", "status": "candidate_only",
                "evidence": {"source": left["dataset"], "field": "analytical_catalog_retrieval", "value": match}})
        if left["dataset"] == "hooktheory" and trusted:
            fids = {clip_families[c] for c in left.get("clip_ids", []) if c in clip_families}
            # Also look up the normalized family, including clips absent from this pair.
            titles, creators = values(left, "titles"), values(left, "creators")
            if titles and creators:
                fid = family_id(creators[0], titles[0])
                if fid in hook_families: fids.add(fid)
            candidate["admission"]["hook_family_ids"].extend(sorted(fids))
            if not fids:
                candidate["admission"]["reasons"].append("matched_hook_family_membership_unknown")
        # Membership of the actual right source matters even for a title-only edge.
        enrichment = pair.get("local_metadata_enrichment", {})
        for row in enrichment.get("right_splits", []):
            candidate["split_evidence"].append({k: row.get(k) for k in
                ("source_id", "piece_id", "native_split", "current_split", "archive_split", "native_current_conflict")})

    # Freeze family evidence only after every source lane has contributed.
    # A heldout variant from supplemental metadata or an enriched pair blocks
    # its sibling versions just like an entry in the base source catalog.
    candidate_heldout_families = {
        candidate["work_family_id"] for candidate in candidates.values()
        if any(r.get("native_current_conflict") is True or any(
            split_name(r.get(k)) in {"validation", "test"}
            for k in ("native_split", "current_split", "archive_split"))
            for r in candidate["split_evidence"])
    }
    for candidate in candidates.values():
        admission = candidate["admission"]
        # Intrinsic exact metadata identity also connects supplemental sources to
        # every known HookTheory clip, without relying on a selected match edge.
        if candidate["work_family_id"] in hook_families:
            admission["hook_family_ids"].append(candidate["work_family_id"])
        admission["hook_family_ids"] = sorted(set(admission["hook_family_ids"]))
        reasons = admission["reasons"]
        if any(split_name(r.get(k)) in {"validation", "test"} for r in candidate["split_evidence"]
               for k in ("native_split", "current_split", "archive_split")):
            reasons.append("known_source_or_current_heldout")
        if any(r.get("native_current_conflict") is True for r in candidate["split_evidence"]):
            reasons.append("native_current_split_conflict")
        if candidate["work_family_id"] in catalog_heldout_families:
            reasons.append("catalog_family_has_heldout")
        if candidate["work_family_id"] in candidate_heldout_families:
            reasons.append("candidate_family_has_heldout")
        for fid in admission["hook_family_ids"]:
            family = hook_families[fid]
            if family["known_heldout"]: reasons.append("hook_family_has_heldout")
            elif not family["all_native_current_train"]: reasons.append("hook_family_membership_unknown")
        quarantine = any(r in reasons for r in ("known_source_or_current_heldout", "native_current_split_conflict", "hook_family_has_heldout", "catalog_family_has_heldout", "candidate_family_has_heldout"))
        inherited_train = any(r.get("current_split") == "train" for r in admission["inherited_receipts"])
        admission["state"] = "quarantined" if quarantine else (
            "analytical_reference_only" if candidate["dataset"] == "hooktheory" else
            "existing_train_candidate" if inherited_train else "supplemental_admission_required")
        if candidate["identity_confidence"] == "missing_title_or_creator": reasons.append("identity_metadata_incomplete")
        admission["reasons"] = sorted(set(reasons))
        for field in ("analytical_sources", "genre_candidates", "split_evidence"):
            candidate[field] = _unique(candidate[field])
        candidate["priority_score"] = (50 if inherited_train else 0) + 10 * sum(
            r["identity_match"] == "exact_title_creator_candidate" for r in candidate["analytical_sources"])
        candidate["priority_score"] += 5 if candidate["titles"] and candidate["creators"] else 0
        if candidate["full_work_status"] == "known_clip": candidate["priority_score"] -= 100
        if quarantine: candidate["priority_score"] -= 1000
    ranked = sorted(candidates.values(), key=lambda r: (-r["priority_score"], r["work_family_id"], r["candidate_id"]))
    families = defaultdict(list)
    for rank, row in enumerate(ranked, 1):
        group = families[row["work_family_id"]]
        row["rank"], row["family_rank"] = rank, len(group) + 1
        group.append(row["candidate_id"])
    family_rows = [{"work_family_id": fid, "candidate_ids": ids,
        "representative_candidate_id": ids[0], "dedup_scope": "normalized_primary_creator_title_or_source_fallback",
        "larger_work_identity": "unverified", "hook_family_evidence": hook_families.get(fid)}
        for fid, ids in sorted(families.items())]
    return ranked, family_rows


def _unique(rows):
    return [json.loads(k) for k in sorted({canonical(row) for row in rows})]


def file_facts(repo, binding):
    """Availability only: do not open or hash a referenced file."""
    repo = Path(repo).resolve()
    value = binding.get("path")
    result = {"path": value, "method": "stat_only_no_content_read", "exists": None, "size_bytes": None}
    if not text(value):
        return {**result, "status": "unspecified"}
    path = Path(value)
    try:
        path = (repo / path).resolve()
        if not path.is_relative_to(repo): return {**result, "status": "unsafe_path"}
        observed = path.stat()
        regular = stat.S_ISREG(observed.st_mode)
        return {**result, "exists": regular, "status": "available" if regular else "not_regular_file",
                "size_bytes": observed.st_size if regular else None}
    except FileNotFoundError:
        return {**result, "exists": False, "status": "missing"}
    except OSError as exc:
        return {**result, "status": "stat_error", "error": type(exc).__name__}


def write_catalog(output, candidates, families, inputs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("candidates.jsonl", candidates), ("families.jsonl", families)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            for row in rows: stream.write(canonical(row) + "\n")
    summary = {"schema": "Corpus400CatalogSummary@1", "experiment_id": EXPERIMENT_ID,
        "candidate_count": len(candidates), "metadata_family_count": len(families),
        "candidate_states": dict(Counter(r["admission"]["state"] for r in candidates)),
        "datasets": dict(Counter(r["dataset"] for r in candidates)),
        "genre_candidate_counts": {g: sum(any(e["genre"] == g for e in r["genre_candidates"]) for r in candidates) for g in GENRES},
        "complete_works": 0, "quota_complete": False, "musical_payloads_read": False,
        "limits": ["Metadata family representatives are not verified full works", "Genre tags remain candidates",
                   "No annotation, version alignment or payload admission performed", "Original external membership may be unknown"],
        "inputs": inputs}
    (output / "summary.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    return summary


DEFAULT_INPUTS = {
    "release": "outputs/multigenre_theory_enrichment_20261007_001/release/attempt002/index.jsonl",
    "hook_clips": "outputs/corpus_overlap_20261005_001/hooktheory-clips.jsonl",
    "global_split": "data/cache/global.split.json",
    "pop_catalog": "outputs/jazz_corpus_overlap_20261007_001/local-001/pop909.jsonl",
    "dlc_catalog": "outputs/jazz_corpus_overlap_20261007_001/local-001/dilemmadata.jsonl",
    "dlc_current": "outputs/c4_pop909_20261005/dilemma-data-manifest.json",
    "ht_pdmx": "outputs/corpus_overlap_20261005_001/matches/hooktheory_pdmx.jsonl",
    "jazz_pairs": "outputs/jazz_corpus_overlap_20261007_002/enriched-001/candidates.jsonl",
}


def read_jsonl(path, *, pair_refs=False):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row, dict): raise ValueError("metadata row must be object")
            if pair_refs: row["metadata_ref"] = {"path": str(path), "line": line_number}
            rows.append(row)
    return rows


def fingerprint(path, repo):
    path = Path(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return {"path": str(path.relative_to(repo)), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def run(repo, output, experiment_record, *, extra_pdmx=(), extra_catalogs=(), inputs=None):
    """Read only configured metadata, then stat referenced files under repo."""
    from music_critic.experiments.registry import load_record
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("output must be under repository outputs")
    if output.exists(): raise FileExistsError("fresh output directory required")
    record_path = (repo / experiment_record).resolve()
    record = load_record(record_path)
    if record["experiment_id"] != EXPERIMENT_ID or record["status"] not in {"planned", "running"}:
        raise ValueError("active registered corpus400 catalog experiment required")
    paths = {name: (repo / path).resolve() for name, path in (inputs or DEFAULT_INPUTS).items()}
    extra_pdmx = [(repo / path).resolve() for path in extra_pdmx]
    extra_catalogs = [(repo / path).resolve() for path in extra_catalogs]
    all_paths = [record_path, *paths.values(), *extra_pdmx, *extra_catalogs]
    if any(not p.is_relative_to(repo) for p in all_paths):
        raise ValueError("metadata paths must remain within repository")
    bindings = [fingerprint(p, repo) for p in all_paths]
    assignments = json.loads(paths["global_split"].read_text())["assignments"]
    dlc_current = unique_rows(json.loads(paths["dlc_current"].read_text())["records"], "record_id")
    catalogs = read_jsonl(paths["pop_catalog"]) + read_jsonl(paths["dlc_catalog"])
    assigned = unique_rows(assignments, "piece_id")
    for row in catalogs:
        member = row.setdefault("membership", {})
        if row["dataset"] == "dilemmadata" and row["id"] in dlc_current:
            member["current_split"] = dlc_current[row["id"]]["split"]
        elif row["dataset"] == "pop909":
            current = assigned.get("piece:pop909-cl-" + row["id"], {})
            member["current_split"] = current.get("split")
    extras = [pdmx_record(row) for path in extra_pdmx for row in read_jsonl(path)]
    extras += [row for path in extra_catalogs for row in read_jsonl(path)]
    candidates, families = build_candidates(release_rows=read_jsonl(paths["release"]),
        hook_clips=read_jsonl(paths["hook_clips"]), assignments=assignments, catalogs=catalogs,
        pairs=read_jsonl(paths["ht_pdmx"], pair_refs=True) + read_jsonl(paths["jazz_pairs"], pair_refs=True),
        extra_records=extras)
    for candidate in candidates:
        candidate["files"] = [{"role": role, **file_facts(repo, binding)}
            for role, binding in candidate["source_bindings"].items()]
        if any(f["exists"] and f["role"] in {"canonical_piece", "source_json", "original_midi"} for f in candidate["files"]):
            candidate["priority_score"] += 20
    candidates.sort(key=lambda r: (-r["priority_score"], r["work_family_id"], r["candidate_id"]))
    ranked_families = defaultdict(list)
    for rank, candidate in enumerate(candidates, 1):
        group = ranked_families[candidate["work_family_id"]]
        candidate["rank"], candidate["family_rank"] = rank, len(group) + 1
        group.append(candidate["candidate_id"])
    for family in families:
        family["candidate_ids"] = ranked_families[family["work_family_id"]]
        family["representative_candidate_id"] = family["candidate_ids"][0]
    return write_catalog(output, candidates, families, bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-record", type=Path,
                        default=Path("docs/experiments/records/" + EXPERIMENT_ID + ".json"))
    parser.add_argument("--extra-pdmx-catalog", type=Path, action="append", default=[])
    parser.add_argument("--extra-catalog", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = run(args.repo, args.output, args.experiment_record,
                  extra_pdmx=args.extra_pdmx_catalog, extra_catalogs=args.extra_catalog)
    print(canonical({k: v for k, v in summary.items() if k != "inputs"}))


if __name__ == "__main__":
    main()
