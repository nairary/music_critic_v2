"""Enrich saved candidate pairs with stat-only local file and split evidence.

Read only pair JSONL metadata. Never open, hash or decode referenced music,
canonical cache or raw TSV payloads. The original candidate fields are retained.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import time

try:
    from scripts.audit_jazz_catalog_overlap import csv_cell, fingerprint, read_catalog, safe_file_availability
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_jazz_catalog_overlap import csv_cell, fingerprint, read_catalog, safe_file_availability


PAIR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*__[A-Za-z0-9][A-Za-z0-9_-]*\.jsonl$")
ROOTS = {
    "pdmx": "data/pdmx",
    "hooktheory": "data/cache/hooktheory/multisource-cache-v1.0.0",
    "pop909": "data/cache/pop909_cl/multisource-cache-v1.0.0",
    "dilemmadata": "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75",
}


def _file(repo: Path, dataset: str, path: object, kind: str, field: str,
          source_id: object = None, projected_status: object = None) -> dict:
    return {
        "kind": kind, "source_field": field, "source_id": source_id,
        "root_relative_to_repo": ROOTS[dataset], "projected_status": projected_status,
        "method": "stat_only_no_content_read",
        **safe_file_availability(repo / ROOTS[dataset], path),
    }


def local_files(record: dict, repo: Path) -> list[dict]:
    """Apply only known, source-projected locators; never derive paths from IDs."""
    dataset = record["dataset"]
    if dataset == "pdmx":
        metadata = record.get("source_metadata", {})
        return [_file(repo, dataset, metadata.get(field), kind, "source_metadata." + field)
                for field, kind in (("path", "pdmx_symbolic_json"), ("mid", "pdmx_midi"),
                                    ("mxl", "pdmx_compressed_musicxml"))]
    if dataset == "hooktheory":
        members = record.get("membership", {}).get("records", [])
        if not members:
            return [_file(repo, dataset, None, "canonical_cache_only", "membership.records")]
        files = []
        for member in members:
            available = member.get("canonical_availability", {})
            files.append(_file(repo, dataset, available.get("path"), "canonical_cache_only",
                               "membership.records.canonical_availability.path", member.get("source_id"),
                               available.get("status")))
        return files
    if dataset in ("pop909", "dilemmadata"):
        available = record.get("availability", {})
        kind = "canonical_cache_only" if dataset == "pop909" else "dilemmadata_raw_tsv"
        return [_file(repo, dataset, available.get("path"), kind, "availability.path",
                      record["id"], available.get("status"))]
    return [{"kind": "unspecified", "status": "unavailable", "exists": None,
             "reason": "dataset_locator_not_supported", "method": "no_stat_attempt"}]


def split_evidence(record: dict) -> list[dict]:
    """Preserve source membership observations without filling unknown splits."""
    membership = record.get("membership", {})
    members = membership.get("records")
    if members is None or not members:
        members = [membership]

    def normalized(value):
        if value is None or not str(value).strip():
            return None
        text = str(value).strip().casefold()
        return {"val": "validation", "valid": "validation"}.get(text, text)

    def conflict(first, second):
        a, b = normalized(first), normalized(second)
        return a != b if a is not None and b is not None else None

    evidence = []
    for member in members:
        current_field = "current_split" if "current_split" in member else "current_global_split"
        archive_field = "saved_profile_split" if "saved_profile_split" in member else "archive_split"
        native, current, archive = member.get("native_split"), member.get(current_field), member.get(archive_field)
        observed_conflict = conflict(native, current)
        supplied_conflict = member.get("native_current_conflict")
        evidence.append({
            "source_id": member.get("source_id", record["id"]), "piece_id": member.get("piece_id"),
            "native_split": native, "current_split": current, "archive_split": archive,
            "current_split_field": current_field, "archive_split_field": archive_field,
            "native_current_conflict": observed_conflict,
            "native_archive_conflict": conflict(native, archive),
            "current_archive_conflict": conflict(current, archive),
            "reported_native_current_conflict": supplied_conflict,
            "conflict_evidence_inconsistent": supplied_conflict != observed_conflict
                if supplied_conflict is not None and observed_conflict is not None else None,
            "current_fields_conflict": conflict(member.get("current_split"), member.get("current_global_split")),
            "archive_fields_conflict": conflict(member.get("saved_profile_split"), member.get("archive_split")),
            "scope": membership.get("scope"), "original_membership_record": member,
        })
    return evidence


def _validate_pair(pair: dict, name: str) -> None:
    if not isinstance(pair, dict) or "local_metadata_enrichment" in pair:
        raise ValueError(f"{name}: expected original, unenriched candidate object")
    for side in ("left", "right"):
        record = pair.get(side)
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"{name}: missing {side} source-record ID")
        if not isinstance(record.get("dataset"), str) or not record["dataset"]:
            raise ValueError(f"{name}: missing {side} dataset identity")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dump(path: Path, rows: list[dict]) -> None:
    with path.with_suffix(".jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json(row) + "\n")
    columns = ["left_dataset", "left_id", "right_dataset", "right_id", "left_titles", "right_titles",
               "retrieval_lanes", "creator_evidence", "work_identity", "right_files"]
    for side in ("left", "right"):
        columns += [side + "_" + field for field in
                    ("native_splits", "current_splits", "archive_splits", "split_conflicts", "split_evidence")]
    columns.append("candidate_json")
    with path.with_suffix(".csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            enrichment = row["local_metadata_enrichment"]
            output = {"retrieval_lanes": _json(row.get("retrieval_lanes")),
                      "creator_evidence": row.get("creator_evidence", ""),
                      "work_identity": row.get("work_identity", "unverified"),
                      "right_files": _json(enrichment["right_files"]), "candidate_json": _json(row)}
            for side in ("left", "right"):
                record = row[side]
                output[side + "_dataset"], output[side + "_id"] = record["dataset"], record["id"]
                output[side + "_titles"] = " | ".join(str(v.get("value") or "")
                    for v in record.get("titles", []))
                evidence = enrichment[side + "_splits"]
                output[side + "_split_evidence"] = _json(evidence)
                for field in ("native", "current", "archive"):
                    output[side + "_" + field + "_splits"] = _json([
                        {"source_id": member["source_id"], "split": member[field + "_split"]}
                        for member in evidence])
                output[side + "_split_conflicts"] = _json([
                    {key: value for key, value in member.items() if key == "source_id" or "conflict" in key}
                    for member in evidence])
            writer.writerow({key: csv_cell(value) for key, value in output.items()})


def _corpus_summary(records: dict[tuple[str, str], dict], expected_corpora: set[str]) -> dict:
    by_corpus = defaultdict(dict)
    for dataset in expected_corpora:
        by_corpus[dataset] = {}
    for (dataset, identity), enriched in records.items():
        by_corpus[dataset][identity] = enriched
    result = {}
    for dataset, rows in sorted(by_corpus.items()):
        groups = defaultdict(lambda: {"file_observations": 0, "record_ids": set()})
        for identity, enriched in rows.items():
            for file in enriched["right_files"]:
                item = groups[(file["kind"], file["status"])]
                item["file_observations"] += 1
                item["record_ids"].add(identity)
        available = defaultdict(dict)
        for (kind, status), counts in sorted(groups.items()):
            available[kind][status] = {"file_observations": counts["file_observations"],
                                       "right_records": len(counts["record_ids"])}
        result[dataset] = {"unique_right_records": len(rows), "availability": dict(available),
                           "scope": "unique source IDs among exported candidates only"}
    return result


def run(matches: Path, repo: Path, output: Path) -> dict:
    """Produce a fresh enrichment export; do not modify any input candidate."""
    started = time.monotonic()
    matches, repo, output = Path(matches).resolve(), Path(repo).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = sorted(path for path in matches.iterdir() if PAIR_NAME.fullmatch(path.name))
    if not paths:
        raise ValueError("No pair JSONL catalogs found")
    for path in paths:
        if path.stem.count("__") != 1 or path.resolve().parent != matches or not path.is_file():
            raise ValueError(f"Unsafe pair catalog path: {path.name}")
    pairs, originals, enriched_records, inputs = {}, {}, {}, []
    for path in paths:
        rows, seen = [], set()
        for pair in read_catalog(path):
            _validate_pair(pair, path.name)
            identity = (pair["right"]["dataset"], pair["right"]["id"])
            pair_identity = (pair["left"]["dataset"], pair["left"]["id"], *identity)
            if pair_identity in seen:
                raise ValueError(f"Duplicate candidate pair in {path.name}: {pair_identity}")
            seen.add(pair_identity)
            signature = _json(pair["right"])
            if identity in originals and originals[identity] != signature:
                raise ValueError(f"Conflicting source-record metadata across pairs: {identity}")
            if identity not in originals:
                originals[identity] = signature
                enriched_records[identity] = {"right_files": local_files(pair["right"], repo),
                                              "right_splits": split_evidence(pair["right"])}
            rows.append({**pair, "local_metadata_enrichment": {
                **enriched_records[identity], "left_splits": split_evidence(pair["left"])}})
        pairs[path.stem] = rows
        inputs.append(fingerprint(path))
    output.mkdir(parents=True, exist_ok=False)
    combined = []
    for name, rows in pairs.items():
        _dump(output / name, rows)
        combined.extend(rows)
    _dump(output / "candidates", combined)
    summary = {
        "schema": "JazzCandidateLocalMetadata@1", "inputs": inputs,
        "pairs": {name: {"exported_candidates": len(rows),
                          "unique_left_records": len({r['left']['id'] for r in rows}),
                          "unique_right_records": len({r['right']['id'] for r in rows}),
                          "source_overlap_status": "not_reassessed"} for name, rows in pairs.items()},
        "corpora": _corpus_summary(enriched_records, {name.split("__")[1] for name in pairs}),
        "exported_candidates": len(combined),
        "access": {"pair_catalog_metadata_read": True, "unmatched_catalogs_read": False,
                   "music_payloads_read": False, "canonical_payloads_read": False,
                   "raw_tsv_payloads_read": False, "file_check_method": "stat_only"},
        "claim_boundaries": [
            "Counts refer to exported candidates and source records, never confirmed works.",
            "No source-catalog searchability or zero-overlap conclusion is inferred from an empty export.",
            "Availability statuses for a record with several files are not mutually exclusive.",
            "HookTheory and POP909 cache existence is canonical-cache-only evidence, not original MIDI availability.",
            "Native, current and archived split evidence remain separate; missing membership remains unknown.",
            "WJD and JAAH source records are not independent work inventories.",
        ], "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.matches, args.repo, args.output), ensure_ascii=False, indent=2))
