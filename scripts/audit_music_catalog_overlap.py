"""Metadata candidate joins; never read music files or transfer annotations.

Inputs are projected catalogs. Equal normalized names are retrieval evidence,
not confirmed composition identities. PDMX arrangement families remain distinct
from the title/artist groups used by the annotation catalog.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import unicodedata


MISSING = {"", "na", "n/a", "null", "none", "nan"}
PDMX_FIELDS = (
    "path", "song_name", "title", "subtitle", "artist_name", "composer_name",
    "best_path", "best_arrangement", "best_unique_arrangement", "license",
    "license_url", "license_conflict", "subset:no_license_conflict", "mid", "mxl",
)


def clean(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.casefold() in MISSING else text


def normalize(value: object) -> str:
    """Preserve scripts/accents; fold width/case and punctuation boundaries."""
    text = unicodedata.normalize("NFKC", clean(value)).casefold()
    text = "".join(c for c in text if c not in "'’ʼ")
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                            else " " for c in text).split())


def values(record: dict, field: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for item in record.get(field, []):
        key = normalize(item["value"])
        if key:
            result[key].append(item)
    return dict(result)


def catalog_index(records: list[dict]) -> dict[str, list[dict]]:
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate catalog IDs")
    index: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for key in values(record, "titles"):
            index[key].append(record)
    return dict(index)


def candidates(index: dict[str, list[dict]], right: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    right_titles = values(right, "titles")
    right_creators = values(right, "creators")
    for key, right_fields in right_titles.items():
        for left in index.get(key, []):
            if left["id"] not in grouped:
                left_creators = values(left, "creators")
                common = sorted(set(left_creators) & set(right_creators))
                field_evidence = [{"key": k, "left_field": a["field"], "right_field": b["field"],
                                   "same_role": creator_role(a["field"]) == creator_role(b["field"])}
                                  for k in common for a in left_creators[k] for b in right_creators[k]]
                grouped[left["id"]] = {
                    "left": left, "right": right, "title_evidence": [],
                    "creator_evidence": "exact_normalized_creator" if common else
                    ("no_exact_creator_agreement" if values(left, "creators") and
                     right_creators else "creator_missing"),
                    "creator_keys": common,
                    "creator_field_evidence": field_evidence,
                    "work_identity": "unverified", "music_alignment": "not_attempted",
                }
            grouped[left["id"]]["title_evidence"].append({
                "normalized_key": key, "left_fields": values(left, "titles")[key],
                "right_fields": right_fields,
            })
    return [grouped[k] for k in sorted(grouped)]


def creator_role(field: str) -> str:
    if "composer" in field.casefold():
        return "composer"
    if "artist" in field.casefold():
        return "artist"
    return "unknown:" + field


def pdmx_record(row: dict) -> dict:
    return {
        "id": row["path"], "dataset": "pdmx",
        "titles": [{"field": k, "value": clean(row.get(k))}
                   for k in ("song_name", "title") if clean(row.get(k))],
        "creators": [{"field": k, "value": clean(row.get(k))}
                     for k in ("artist_name", "composer_name") if clean(row.get(k))],
        "source_metadata": {k: row.get(k) for k in PDMX_FIELDS},
    }


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def read_catalog(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_hooktheory(records) -> tuple[list[dict], list[dict]]:
    """Group projected clip metadata by original artist/song slugs, not work ID."""
    grouped, missing = {}, []
    for record in records:
        metadata = record.metadata
        ht = metadata.get("hooktheory", {})
        artist, song = clean(ht.get("artist")), clean(ht.get("song"))
        if not song:
            missing.append({"clip_id": record.clip_id, "reason": "missing_song_metadata"})
            continue
        key = json.dumps([artist, song], ensure_ascii=False, separators=(",", ":"))
        if key not in grouped:
            grouped[key] = {"dataset": "hooktheory", "id": key,
                            "titles": [{"field": "hooktheory.song", "value": song}],
                            "creators": [{"field": "hooktheory.artist", "value": artist}] if artist else [],
                            "clip_ids": [], "source_clips": [],
                            "group_semantics": "exact source artist/song slug pair; work identity unverified"}
        grouped[key]["clip_ids"].append(record.clip_id)
        grouped[key]["source_clips"].append({"clip_id": record.clip_id, "metadata": metadata})
    return [grouped[k] for k in sorted(grouped)], missing


def summarize(pairs: list[dict]) -> dict:
    output = {}
    lanes = {"all_title_candidates": pairs}
    for label in ("exact_normalized_creator", "no_exact_creator_agreement", "creator_missing"):
        lanes[label] = [p for p in pairs if p["creator_evidence"] == label]
    lanes["exact_same_role_creator"] = [p for p in pairs if any(e["same_role"] for e in p["creator_field_evidence"])]
    lanes["exact_cross_role_creator_only"] = [p for p in pairs if p["creator_field_evidence"] and
                                             not any(e["same_role"] for e in p["creator_field_evidence"])]
    for label, rows in lanes.items():
        output[label] = {
            "pairs": len(rows), "left_records": len({p["left"]["id"] for p in rows}),
            "right_records": len({p["right"]["id"] for p in rows}),
            "left_clip_ids": len({c for p in rows for c in p["left"].get("clip_ids", [])}),
            "pdmx_best_paths": len({p["right"].get("source_metadata", {}).get("best_path")
                                    for p in rows if p["right"].get("source_metadata", {}).get("best_path")}),
            "right_license_conflict_rows": len({p["right"]["id"] for p in rows if
                p["right"].get("source_metadata", {}).get("license_conflict") == "True"}),
        }
    return output


def dump_pairs(path: Path, rows: list[dict]) -> None:
    with path.with_suffix(".jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    columns = ["left_id", "right_id", "left_titles", "right_titles", "left_creators",
               "right_creators", "creator_evidence", "title_keys", "clip_ids",
               "license_conflict", "best_path", "mid", "mxl"]
    with path.with_suffix(".csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for p in rows:
            r = {"left_id": p["left"]["id"], "right_id": p["right"]["id"],
                 "creator_evidence": p["creator_evidence"],
                 "title_keys": " | ".join(e["normalized_key"] for e in p["title_evidence"]),
                 "clip_ids": " | ".join(p["left"].get("clip_ids", []))}
            for side in ("left", "right"):
                for field in ("titles", "creators"):
                    r[f"{side}_{field}"] = " | ".join(v["value"] for v in p[side].get(field, []))
            for key in ("license_conflict", "best_path", "mid", "mxl"):
                r[key] = p["right"].get("source_metadata", {}).get(key, "")
            # These are untrusted catalog strings; suppress spreadsheet formulas.
            writer.writerow({k: "'" + str(v) if str(v).startswith(("=", "+", "-", "@"))
                             else v for k, v in r.items()})


def run(hook: Path, pop: Path, pdmx: Path, output: Path) -> dict:
    started = time.monotonic()
    output.mkdir(exist_ok=False)
    hook_rows, pop_rows = read_catalog(hook), read_catalog(pop)
    ht_index, pop_index = catalog_index(hook_rows), catalog_index(pop_rows)
    ht_pd, pop_pd, ht_pop = [], [], []
    for row in pop_rows:
        ht_pop.extend(candidates(ht_index, row))
    coverage = Counter()
    pdmx_ids = set()
    with pdmx.open(newline="") as handle:
        for row in csv.DictReader(handle):
            record = pdmx_record(row)
            if record["id"] in pdmx_ids:
                raise ValueError("Duplicate PDMX path")
            pdmx_ids.add(record["id"])
            coverage["rows"] += 1
            coverage["with_title"] += bool(values(record, "titles"))
            coverage["with_creator"] += bool(values(record, "creators"))
            ht_pd.extend(candidates(ht_index, record))
            pop_pd.extend(candidates(pop_index, record))
    summary = {"normalization": "NFKC, casefold, remove apostrophes; preserve letters, numbers and combining marks; other characters to spaces. No aliases, translations, transliteration, fuzzy or substring matching.",
               "scope": "All-split catalog metadata only; no musical payload decoding; no model metrics.",
               "catalogs": {"hooktheory_named_groups": len(hook_rows),
                            "hooktheory_clip_ids": len({c for r in hook_rows for c in r.get("clip_ids", [])}),
                            "pop909_rows": len(pop_rows), "pdmx": dict(coverage)},
               "pairs": {}, "inputs": [fingerprint(p) for p in (hook, pop, pdmx)]}
    for name, pairs in (("hooktheory_pdmx", ht_pd), ("pop909_pdmx", pop_pd), ("hooktheory_pop909", ht_pop)):
        dump_pairs(output / name, pairs)
        summary["pairs"][name] = summarize(pairs)
    common = sorted({p["right"]["id"] for p in ht_pd} & {p["right"]["id"] for p in pop_pd})
    summary["pdmx_rows_with_title_candidates_in_both"] = common
    summary["elapsed_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hook", "pop", "pdmx", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.hook, args.pop, args.pdmx, args.output), ensure_ascii=False, indent=2))
