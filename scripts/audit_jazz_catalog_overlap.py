"""Join projected identity catalogs without reading musical payloads.

All exported rows are retrieval candidates with unverified work identity.
Source records and every title/creator field remain attached to the evidence.
Only JSONL catalog paths passed by the caller are read; source metadata paths
are never opened by the matcher.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import stat
import time
import unicodedata

try:
    from scripts.audit_music_catalog_overlap import clean, fingerprint, normalize, read_catalog
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from audit_music_catalog_overlap import clean, fingerprint, normalize, read_catalog


RETRIEVAL_LANES = ("exact_normalized_title", "deterministic_title_variant")
CREATOR_LANES = ("exact_normalized_creator", "no_exact_creator_agreement", "creator_missing",
                 "exact_same_role_creator", "exact_cross_role_creator_only", "exact_unknown_role_creator")
_TRAILING_BRACKET = re.compile(r"^(.*?)\s*(?:\(([^()]*)\)|\[([^\[\]]*)\])\s*$")
_MARKER = re.compile(
    r"(?:arr(?:anged)?(?: by)? .+|arrangement(?: by .+)?|"
    r"[a-z0-9 ]+ arrangement(?: by .+)?|version(?: [0-9]+| by .+)?|"
    r"[a-z0-9 ]+ version(?: [0-9]+)?)"
)


def _remove_marker(value: str) -> str:
    while True:
        match = _TRAILING_BRACKET.fullmatch(value)
        if not match or not _MARKER.fullmatch(normalize(match[2] or match[3])):
            return value
        if not normalize(match[1]):
            return value
        value = match[1].strip()


def _fold_latin_accents(value: str) -> str:
    output, latin_base = [], False
    for char in unicodedata.normalize("NFD", value):
        if unicodedata.category(char).startswith("M"):
            if not latin_base:
                output.append(char)
        else:
            latin_base = "LATIN" in unicodedata.name(char, "")
            output.append(char)
    return unicodedata.normalize("NFC", "".join(output))


def title_variants(value: object) -> list[dict]:
    """Finite explicit variants; never fuzzy, substring or guessed alias joins.

    Brackets are removed only at the end and only for explicit arrangement or
    version markers. English articles are optional retrieval evidence. Accent
    folding affects Latin characters only, preserving marks in other scripts.
    """
    original = clean(value)
    if not normalize(original):
        return []
    states = [(original, ())]
    operations = (
        ("remove_bracketed_marker", _remove_marker),
        ("move_trailing_article", lambda text: re.sub(
            r"^(.+),\s*(the|an|a)$", r"\2 \1", text, flags=re.IGNORECASE)),
        ("remove_leading_article", lambda text: re.sub(
            r"^(the|an|a)\s+(?=\S)", "", text, flags=re.IGNORECASE)),
        ("accent_fold_latin", _fold_latin_accents),
    )
    for operation, transform in operations:
        for text, previous in list(states):
            changed = transform(text)
            if normalize(changed) and normalize(changed) != normalize(text):
                states.append((changed, previous + (operation,)))
    return [{"normalized_key": normalize(text), "operations": list(operations)}
            for text, operations in states]


def _title_entries(record: dict) -> list[dict]:
    grouped = {}
    for field in record.get("titles", []):
        for variant in title_variants(field.get("value")):
            key = (variant["normalized_key"], tuple(variant["operations"]))
            if key not in grouped:
                grouped[key] = {**variant, "fields": []}
            grouped[key]["fields"].append(field)
    return list(grouped.values())


def creator_role(field: str) -> str:
    """Keep composer, performer and artist source roles distinct."""
    plurals = {role + "s": role for role in
               ("composer", "performer", "artist", "arranger", "lyricist")}
    tokens = {plurals.get(token, token) for token in re.findall(r"[a-z]+", field.casefold())}
    for role in ("composer", "performer", "artist", "arranger", "lyricist"):
        if role in tokens:
            return role
    return "unknown:" + field


def _field_values(record: dict, kind: str) -> dict[str, list[dict]]:
    values: dict[str, list[dict]] = defaultdict(list)
    for field in record.get(kind, []):
        key = normalize(field.get("value"))
        if key:
            values[key].append(field)
    return dict(values)


def _creator_evidence(left: dict, right: dict) -> dict:
    left_values = _field_values(left, "creators")
    right_values = _field_values(right, "creators")
    common = sorted(set(left_values) & set(right_values))
    evidence = []
    for key in common:
        for a in left_values[key]:
            for b in right_values[key]:
                a_role, b_role = creator_role(a["field"]), creator_role(b["field"])
                known_roles = not a_role.startswith("unknown:") and not b_role.startswith("unknown:")
                same_role = a_role == b_role if known_roles else None
                evidence.append({
                    "normalized_key": key, "left_field": a["field"], "right_field": b["field"],
                    "left_value": a["value"], "right_value": b["value"],
                    "left_role": a_role, "right_role": b_role, "same_role": same_role,
                    "same_field": a["field"] == b["field"],
                    "role_comparison": "unknown_role" if not known_roles else (
                        "same_role" if same_role else "different_role"),
                })
    label = "exact_normalized_creator" if common else (
        "no_exact_creator_agreement" if left_values and right_values else "creator_missing")
    return {"creator_evidence": label, "creator_keys": common, "creator_field_evidence": evidence}


def _validate_rows(rows: list[dict], dataset: str) -> None:
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{dataset}: every record needs a nonempty string ID")
        if row["id"] in seen:
            raise ValueError(f"{dataset}: duplicate catalog ID {row['id']!r}")
        seen.add(row["id"])
        for kind in ("titles", "creators"):
            fields = row.get(kind, [])
            if not isinstance(fields, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("field"), str)
                or not isinstance(item.get("value"), (str, type(None))) for item in fields
            ):
                raise ValueError(f"{dataset}: {kind} must be field/value objects")


def match_catalogs(left: list[dict], right: list[dict]) -> list[dict]:
    """Return distinct source-record pairs, never consolidate work identities."""
    index = defaultdict(list)
    for row in left:
        for entry in _title_entries(row):
            index[entry["normalized_key"]].append((row, entry))
    grouped = {}
    for right_row in right:
        for right_entry in _title_entries(right_row):
            key = right_entry["normalized_key"]
            for left_row, left_entry in index.get(key, []):
                lane = ("deterministic_title_variant" if left_entry["operations"] or
                        right_entry["operations"] else "exact_normalized_title")
                pair_key = (left_row["id"], right_row["id"])
                if pair_key not in grouped:
                    grouped[pair_key] = {
                        "left": left_row, "right": right_row, "title_evidence": [],
                        "retrieval_lanes": [],
                        "work_identity": "unverified", "music_alignment": "not_attempted",
                        **_creator_evidence(left_row, right_row),
                    }
                grouped[pair_key]["title_evidence"].append({
                    "lane": lane, "normalized_key": key,
                    "left_fields": left_entry["fields"], "right_fields": right_entry["fields"],
                    "left_operations": left_entry["operations"],
                    "right_operations": right_entry["operations"],
                })
                if lane not in grouped[pair_key]["retrieval_lanes"]:
                    grouped[pair_key]["retrieval_lanes"].append(lane)
    for pair in grouped.values():
        pair["retrieval_lanes"].sort(key=RETRIEVAL_LANES.index)
        pair["primary_retrieval_lane"] = pair["retrieval_lanes"][0]
    return [grouped[key] for key in sorted(grouped)]


def _counts(pairs: list[dict]) -> dict:
    return {"pairs": len(pairs), "left_records": len({p["left"]["id"] for p in pairs}),
            "right_records": len({p["right"]["id"] for p in pairs})}


def _summarize(pairs: list[dict], searchable: bool) -> dict:
    lanes = {"all_title_candidates": pairs}
    for lane in RETRIEVAL_LANES:
        lanes[lane] = [p for p in pairs if lane in p["retrieval_lanes"]]
    lanes["deterministic_variant_only"] = [p for p in pairs if
        p["primary_retrieval_lane"] == "deterministic_title_variant"]
    for lane in CREATOR_LANES[:3]:
        lanes[lane] = [p for p in pairs if p["creator_evidence"] == lane]
    lanes["exact_same_role_creator"] = [p for p in pairs if any(
        e["same_role"] for e in p["creator_field_evidence"])]
    lanes["exact_cross_role_creator_only"] = [p for p in pairs if p["creator_field_evidence"]
        and all(e["same_role"] is False for e in p["creator_field_evidence"])]
    lanes["exact_unknown_role_creator"] = [p for p in pairs if any(
        e["same_role"] is None for e in p["creator_field_evidence"])]
    unknown = {"pairs": None, "left_records": None, "right_records": None}
    return {lane: _counts(rows) if searchable else dict(unknown) for lane, rows in lanes.items()}


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def csv_cell(value: object) -> str:
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _dump_pairs(path: Path, pairs: list[dict]) -> None:
    _dump_jsonl(path.with_suffix(".jsonl"), pairs)
    columns = ["left_id", "right_id", "left_titles", "right_titles", "left_creators",
               "right_creators", "retrieval_lanes", "creator_evidence", "title_evidence",
               "creator_field_evidence", "work_identity", "music_alignment", "left_record", "right_record"]
    with path.with_suffix(".csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            output = {k: pair[k] for k in ("creator_evidence", "work_identity", "music_alignment")}
            for side in ("left", "right"):
                output[side + "_id"] = pair[side]["id"]
                output[side + "_record"] = json.dumps(pair[side], ensure_ascii=False, sort_keys=True)
                for kind in ("titles", "creators"):
                    output[side + "_" + kind] = " | ".join(
                        clean(item.get("value")) for item in pair[side].get(kind, []))
            for kind in ("retrieval_lanes", "title_evidence", "creator_field_evidence"):
                output[kind] = json.dumps(pair[kind], ensure_ascii=False, sort_keys=True)
            writer.writerow({key: csv_cell(value) for key, value in output.items()})


def _load(path: Path | None, name: str) -> tuple[list[dict], dict]:
    if path is None or not Path(path).is_file():
        return [], {"status": "unavailable", "path": str(path) if path is not None else None,
                    "rows": None, "with_title": None, "with_creator": None,
                    "reason": "catalog_not_supplied" if path is None else "catalog_file_missing"}
    rows = read_catalog(Path(path))
    _validate_rows(rows, name)
    named = sum(bool(_field_values(row, "titles")) for row in rows)
    return rows, {"status": "searchable" if named else "unsearchable", "rows": len(rows),
                  "with_title": named, "without_title": len(rows) - named,
                  "with_creator": sum(bool(_field_values(row, "creators")) for row in rows),
                  "input": fingerprint(Path(path)),
                  "reason": None if named else "no_searchable_titles"}


def _safe_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or "__" in name:
        raise ValueError(f"Unsafe dataset output name: {name!r}")


def safe_file_availability(root: Path, relative_path: object) -> dict:
    """Inspect an explicit relative musical-file locator with stat only.

    This optional projection helper is never applied automatically to catalog
    metadata. Traversal, absolute paths and symlinks escaping root fail closed.
    No bytes of the referenced file are read or hashed.
    """
    relative = clean(relative_path)
    if not relative:
        return {"status": "unavailable", "relative_path": None, "exists": None,
                "reason": "source_path_missing"}
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or re.match(r"^[A-Za-z]:", relative):
        raise ValueError(f"Unsafe relative source path: {relative!r}")
    root = Path(root).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source path escapes declared root: {relative!r}") from error
    try:
        metadata = target.stat()
    except FileNotFoundError:
        return {"status": "missing", "relative_path": relative, "exists": False}
    is_file = stat.S_ISREG(metadata.st_mode)
    return {"status": "available" if is_file else "not_regular_file", "relative_path": relative,
            "exists": is_file, "size_bytes": metadata.st_size if is_file else None}


def run(left_catalogs: dict[str, Path | None], right_catalogs: dict[str, Path | None],
        output: Path) -> dict:
    """Write the full Cartesian dataset matrix; missing inputs have null counts.

    Include every expected dataset key, with None or a nonexistent path for a
    missing catalog. Output must not already exist. Identifiers are opaque and
    never used as output paths. Only dataset keys determine safe filenames.
    """
    started = time.monotonic()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name in list(left_catalogs) + list(right_catalogs):
        _safe_name(name)
    left = {name: _load(path, name) for name, path in sorted(left_catalogs.items())}
    right = {name: _load(path, name) for name, path in sorted(right_catalogs.items())}
    output.mkdir(exist_ok=False)
    summary = {
        "schema": "JazzCatalogOverlap@1", "catalogs": {
            "left": {name: status for name, (_, status) in left.items()},
            "right": {name: status for name, (_, status) in right.items()}}, "pairs": {},
        "scope": "Projected all-split metadata only; no musical payload reads, models or annotation transfer.",
        "claim_boundary": "Candidates do not establish work identity or musical version alignment.",
        "normalization": "NFKC, casefold, apostrophe removal, punctuation boundaries; preserve scripts and accents.",
        "variant_operations": ["remove_bracketed_marker", "move_trailing_article",
                               "remove_leading_article", "accent_fold_latin"],
        "lane_accounting": "Evidence lanes may overlap; exact_normalized_title and deterministic_variant_only are disjoint.",
        "source_dependencies": "Do not sum dependent WJD/JAAH catalogs as independent works.",
    }
    for left_name, (left_rows, left_status) in left.items():
        for right_name, (right_rows, right_status) in right.items():
            name = left_name + "__" + right_name
            searchable = left_status["status"] == right_status["status"] == "searchable"
            pairs = match_catalogs(left_rows, right_rows) if searchable else []
            _dump_pairs(output / name, pairs)
            pair_summary = {"left": left_name, "right": right_name,
                            "status": "searched" if searchable else "not_assessed",
                            "left_status": left_status["status"], "right_status": right_status["status"],
                            "lanes": _summarize(pairs, searchable), "unmatched": {}}
            for side, rows in (("left", left_rows), ("right", right_rows)):
                matched = {pair[side]["id"] for pair in pairs}
                unmatched = [{"record": row, "reason": (
                    "missing_searchable_title" if not _field_values(row, "titles") else
                    "no_title_candidate" if searchable else "pair_not_assessed")}
                    for row in rows if row["id"] not in matched]
                _dump_jsonl(output / (name + ".unmatched_" + side + ".jsonl"), unmatched)
                pair_summary["unmatched"][side] = len(unmatched) if searchable else None
            summary["pairs"][name] = pair_summary
    summary["elapsed_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return summary


def _catalog_arguments(items: list[str]) -> dict[str, Path | None]:
    result = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError("Each catalog must use a unique NAME=PATH (or NAME= for unavailable)")
        result[name] = Path(path) if path else None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--right", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(_catalog_arguments(args.left), _catalog_arguments(args.right), args.output),
                     ensure_ascii=False, indent=2))
