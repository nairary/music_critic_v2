"""Project WJD and Chord-Jazzification identity catalogs without music payloads.

WJD SQL access is column-allowlisted. Chord-Jazzification workbook access is
limited to identity columns and their referenced shared strings. No source code
is executed, musical file paths followed, or numbered annotation workbook read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
import time
from xml.etree import ElementTree as ET
from zipfile import ZipFile


SCHEMA = "JazzIdentityCatalog@1.0.0"
WJD_URL = "https://jazzomat.hfm-weimar.de/download/download.html"
WJD_COLUMNS = {
    "solo_info": {"melid", "compid", "title", "performer"},
    "composition_info": {"compid", "title", "composer"},
}
IDENTITY_HEADERS = {"ID", "Piece", "Album", "Composer", "Arranger"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROW = re.compile(rb'<row\b[^>]*>.*?</row>', re.S)
CELL = re.compile(rb'<c\b[^>]*(?:/>|>.*?</c>)', re.S)
SHARED = re.compile(rb'<si(?:\s[^>]*)?>.*?</si>', re.S)


def _safe_xml(raw: bytes) -> bytes:
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("DTD/entities are unsupported")
    return raw


def _cell_fragments(row: bytes, *, columns: set[str] | None = None) -> dict:
    result = {}
    for match in CELL.finditer(row):
        raw = match.group()
        coordinate = re.search(rb'\br="([A-Z]+)([1-9][0-9]*)"', raw.split(b">", 1)[0])
        if coordinate is None:
            raise ValueError("Invalid cell coordinate")
        column, number = (part.decode("ascii") for part in coordinate.groups())
        if columns is not None and column not in columns:
            continue  # Do not XML-decode unselected cell bodies.
        if column in result:
            raise ValueError("Duplicate cell coordinate")
        cell = ET.fromstring(_safe_xml(raw))
        if cell.find("f") is not None:
            raise ValueError("Formula in identity cell")
        result[column] = (cell, int(number))
    return result


def _shared_ids(cells: dict) -> set[int]:
    result = set()
    for cell, _ in cells.values():
        if cell.get("t") == "s":
            raw = cell.findtext("v")
            if raw is None or not re.fullmatch(r"[0-9]+", raw):
                raise ValueError("Invalid shared-string reference")
            result.add(int(raw))
    return result


def _shared_values(raw: bytes, ids: set[int]) -> dict:
    result = {}
    for index, match in enumerate(SHARED.finditer(_safe_xml(raw))):
        if index in ids:
            result[index] = "".join(ET.fromstring(match.group()).itertext())
    if set(result) != ids:
        raise ValueError("Missing shared-string reference")
    return result


def _cell_value(cell: ET.Element, shared: dict):
    kind = cell.get("t", "n")
    if kind == "s":
        return shared[int(cell.findtext("v"))]
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.findall("is//t"))
    if kind not in {"n", "str"}:
        raise ValueError(f"Unsupported identity cell type {kind}")
    return cell.findtext("v")


def read_chord_workbook(path: Path) -> dict:
    """Read only documented identity columns, skipping all other cell bodies.

    The bounded lexical selector supports the pinned Excel-produced unprefixed
    XML layout. It is intentionally not a general spreadsheet evaluation engine.
    """
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 100:
            raise ValueError("Duplicate or excessive workbook members")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name or any(ord(x) < 32 for x in name):
                raise ValueError("Unsafe workbook member")
        if sum(x.file_size for x in archive.infolist()) > 5_000_000:
            raise ValueError("Excessive workbook size")
        book = ET.fromstring(_safe_xml(archive.read("xl/workbook.xml")))
        sheets = book.findall("m:sheets/m:sheet", NS)
        if len(sheets) != 1:
            raise ValueError("Expected one identity worksheet")
        relation_id = sheets[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        relations = ET.fromstring(_safe_xml(archive.read("xl/_rels/workbook.xml.rels")))
        selected = [r for r in relations if r.get("Id") == relation_id]
        if len(selected) != 1 or selected[0].get("TargetMode") == "External":
            raise ValueError("Missing or external worksheet relationship")
        target = selected[0].get("Target", "")
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        if ".." in PurePosixPath(member).parts or member not in names:
            raise ValueError("Unsafe worksheet relationship")
        raw = _safe_xml(archive.read(member))
        shared_raw = archive.read("xl/sharedStrings.xml") if "xl/sharedStrings.xml" in names else b""
        matches = iter(ROW.finditer(raw))
        header = next(matches, None)
        if header is None:
            raise ValueError("Empty or unsupported identity worksheet")
        header_cells = _cell_fragments(header.group())
        header_ids = _shared_ids(header_cells)
        shared = _shared_values(shared_raw, header_ids)
        headers = {column: _cell_value(cell, shared) for column, (cell, _) in header_cells.items()}
        if len(set(headers.values())) != len(headers) or not IDENTITY_HEADERS <= set(headers.values()):
            raise ValueError("Missing or duplicate identity headers")
        columns = {column for column, header in headers.items() if header in IDENTITY_HEADERS}
        projected, data_ids, row_numbers = [], set(), set()
        for match in matches:
            cells = _cell_fragments(match.group(), columns=columns)
            numbers = {number for _, number in cells.values()}
            if not cells:
                continue
            if len(numbers) != 1 or next(iter(numbers)) in row_numbers:
                raise ValueError("Invalid or duplicate identity row coordinate")
            row_numbers.update(numbers)
            projected.append(cells)
            data_ids.update(_shared_ids(cells))
        shared.update(_shared_values(shared_raw, data_ids - header_ids))
        rows, seen = [], set()
        for cells in projected:
            item = {headers[column]: _cell_value(cells[column][0], shared) if column in cells else None
                    for column in sorted(columns)}
            if not any(value is not None and str(value).strip() for value in item.values()):
                continue
            raw_id = item["ID"]
            if raw_id is None or not re.fullmatch(r"[1-9][0-9]*(?:\.0+)?", raw_id):
                raise ValueError("Invalid Chord-Jazzification ID")
            item["ID"] = str(int(raw_id.split(".")[0]))
            if item["ID"] in seen:
                raise ValueError("Duplicate Chord-Jazzification ID")
            seen.add(item["ID"])
            item["row_number"] = next(iter(cells.values()))[1]
            rows.append(item)
    return {"rows": rows, "sheet": sheets[0].get("name"), "sheet_member": member,
            "headers": headers, "skipped_columns": sorted(set(headers) - columns),
            "decoded_shared_string_ids": sorted(header_ids | data_ids),
            "source": fingerprint(path)}


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def wjd_authorizer(action, table, column, database, trigger):
    """Deny annotations, extension execution, writes and unapproved metadata."""
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and database == "main" and not trigger:
        if column in WJD_COLUMNS.get(table, set()):
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _named(field: str, value) -> list[dict]:
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        raise ValueError(f"Non-string identity value: {field}")
    return [{"field": field, "value": value}] if value.strip() else []


def _integer_id(value, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Invalid {field}")
    return value


def project_wjd(snapshot_path: Path, *, sqlite_path: Path | None = None) -> dict:
    """Keep each melid; enrich from authenticated local identity metadata only."""
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, list):
        raise ValueError("Expected WJD identity list")
    sources = {"snapshot": fingerprint(snapshot_path)}
    queries, local, compositions = [], {}, {}
    if sqlite_path is not None:
        sources["sqlite"] = fingerprint(sqlite_path)
        connection = sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        connection.set_authorizer(wjd_authorizer)
        connection.set_trace_callback(queries.append)
        try:
            for item in connection.execute("SELECT melid, compid, title, performer FROM solo_info ORDER BY melid"):
                if item[0] in local:
                    raise ValueError("Duplicate WJD SQL melid")
                local[item[0]] = item
            for item in connection.execute("SELECT compid, title, composer FROM composition_info ORDER BY compid"):
                if item[0] in compositions:
                    raise ValueError("Duplicate WJD SQL compid")
                compositions[item[0]] = item
        finally:
            connection.close()
    rows, seen = [], set()
    allowed = {"melid", "compid", "title", "performer", "instrument", "style"}
    for position, item in enumerate(snapshot):
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("Unexpected WJD snapshot fields")
        melid = _integer_id(item.get("melid"), field="WJD melid")
        compid = _integer_id(item.get("compid"), field="WJD compid")
        if melid in seen:
            raise ValueError("Duplicate WJD melid")
        seen.add(melid)
        titles = _named("title", item.get("title"))
        creators = _named("performer", item.get("performer"))
        metadata = {"source_url": WJD_URL, "source_version": None, "snapshot": sources["snapshot"],
                    "json_pointer": f"/{position}", "melid": melid, "compid": compid,
                    "instrument": item.get("instrument"), "style": item.get("style")}
        if sqlite_path is not None:
            expected = (melid, compid, item.get("title"), item.get("performer"))
            if local.get(melid) != expected:
                raise ValueError(f"WJD identity snapshot disagrees with SQLite for melid {melid}")
            if compid not in compositions:
                raise ValueError(f"Missing WJD composition {compid}")
            composition = compositions[compid]
            if composition[1] != item.get("title"):
                titles += _named("composition_info.title", composition[1])
            creators += _named("composer", composition[2])
            metadata.update({"sqlite": sources["sqlite"], "composition_title": composition[1],
                             "composer": composition[2], "composition_pointer": f"composition_info[compid={compid}]"})
        rows.append({"schema_version": SCHEMA, "dataset": "wjd", "id": str(melid),
                     "titles": titles, "creators": creators, "source_metadata": metadata,
                     "membership": {"native_split": None, "current_split": None},
                     "availability": {"title": bool(titles), "composer": any(x["field"] == "composer" for x in creators),
                                      "split_status": "not_supplied_by_identity_sources"}})
    if sqlite_path is not None and seen != set(local):
        raise ValueError("WJD retained snapshot does not cover SQL identity catalog")
    return {"rows": rows, "sources": sources, "sqlite_queries": queries,
            "source_version": None, "musical_payload_read": False,
            "limitations": ["Retained local snapshot; current upstream completeness was not re-fetched.",
                            "Each record is a solo melid, not a unique composition or independent work.",
                            "Native and current project splits are unavailable in these identity sources."]}


def project_chord(tree_path: Path, workbook_path: Path, *, upstream_commit: str, expected_count: int = 50) -> dict:
    """Bind all source work IDs to a complete pinned tree and identity workbook."""
    if not re.fullmatch(r"[a-f0-9]{40}", upstream_commit):
        raise ValueError("Invalid upstream commit")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated") is not False:
        raise ValueError("Git tree is truncated or completeness unknown")
    entries = {}
    for index, entry in enumerate(tree["tree"]):
        name = entry["path"]
        if name in entries:
            raise ValueError("Duplicate Git tree path")
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name:
            raise ValueError("Unsafe Git tree path")
        entries[name] = (index, entry)
    prefix = "Chord_Jazzification_Dataset/"
    native = {name[len(prefix):-5]: value for name, value in entries.items()
              if re.fullmatch(re.escape(prefix) + r"[1-9][0-9]*\.xlsx", name)}
    expected_ids = {str(i) for i in range(1, expected_count + 1)}
    if set(native) != expected_ids or any(item[1]["type"] != "blob" for item in native.values()):
        raise ValueError("Git tree does not contain complete expected work IDs")
    corpus_path = prefix + "corpus.xlsx"
    if corpus_path not in entries:
        raise ValueError("Missing corpus metadata workbook in tree")
    corpus = entries[corpus_path][1]
    raw = workbook_path.read_bytes()
    blob_sha = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    if corpus.get("type") != "blob" or corpus.get("sha") != blob_sha or corpus.get("size") != len(raw):
        raise ValueError("Corpus metadata Git blob identity mismatch")
    workbook = read_chord_workbook(workbook_path)
    if {item["ID"] for item in workbook["rows"]} != expected_ids:
        raise ValueError("Workbook and tree work ID sets disagree")
    repository = "https://github.com/Tsung-Ping/Chord-Jazzification"
    sources = {"tree": fingerprint(tree_path), "workbook": workbook["source"]}
    rows = []
    for item in sorted(workbook["rows"], key=lambda row: int(row["ID"])):
        index, source_entry = native[item["ID"]]
        titles = _named("Piece", item["Piece"])
        creators = _named("composer", item["Composer"]) + _named("arranger", item["Arranger"])
        rows.append({"schema_version": SCHEMA, "dataset": "chord_jazzification", "id": item["ID"],
                     "titles": titles, "creators": creators,
                     "source_metadata": {"source_url": f"{repository}/blob/{upstream_commit}/{corpus_path}",
                                         "repository_url": repository, "upstream_commit": upstream_commit,
                                         "tree_sha": tree["sha"], "tree_pointer": f"/tree/{index}",
                                         "source_annotation_path": source_entry["path"],
                                         "source_annotation_git_blob": source_entry["sha"],
                                         "source_annotation_size_bytes": source_entry.get("size"),
                                         "source_annotation_opened": False,
                                         "metadata_git_blob": blob_sha, "workbook": sources["workbook"],
                                         "sheet": workbook["sheet"], "row": item["row_number"],
                                         "identity_cells": [f"{column}{item['row_number']}" for column, header in workbook["headers"].items() if header in IDENTITY_HEADERS],
                                         "native_identity": {key: value for key, value in item.items() if key != "row_number"}},
                     "membership": {"native_split": None, "current_split": None},
                     "availability": {"title": bool(titles), "composer": bool(_named("composer", item["Composer"])),
                                      "split_status": "not_supplied_by_identity_sources"}})
    return {"rows": rows, "sources": sources, "upstream_commit": upstream_commit,
            "tree_sha": tree["sha"], "metadata_git_blob_verified": True,
            "workbook_projection": {key: value for key, value in workbook.items() if key not in {"rows", "source"}},
            "musical_payload_read": False,
            "limitations": ["Titles and creators retain source spelling; no external translations or inferred aliases.",
                            "Arranger and composer fields do not identify a particular audio performance.",
                            "Native/current split assignments are unavailable in the identity workbook.",
                            "Numbered annotation workbooks, MIDI, scores and audio were not opened."]}


def write_catalog(result: dict, directory: Path, *, command: list[str]) -> dict:
    """Publish new evidence only; existing artifacts are never replaced."""
    output, receipt_path = directory / "catalog.jsonl", directory / "receipt.json"
    if output.exists() or receipt_path.exists():
        raise ValueError("Refuse to overwrite existing catalog or receipt")
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["rows"])
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    receipt = {key: value for key, value in result.items() if key != "rows"}
    receipt.update({"schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                    "command": command, "python": sys.version, "script": fingerprint(Path(__file__)),
                    "row_count": len(result["rows"]), "catalog": fingerprint(output),
                    "title_available_count": sum(bool(row["titles"]) for row in result["rows"]),
                    "creator_available_count": sum(bool(row["creators"]) for row in result["rows"])})
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return receipt


def _assert_digest(path: Path, expected: str | None) -> None:
    if expected is None or fingerprint(path)["sha256"] != expected:
        raise ValueError(f"Source SHA256 mismatch or missing expected SHA256: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["wjd", "chord"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--expected-snapshot-sha256")
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--expected-sqlite-sha256")
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--expected-tree-sha256")
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--expected-workbook-sha256")
    parser.add_argument("--upstream-commit")
    args = parser.parse_args()
    if (args.output_root / "catalog.jsonl").exists() or (args.output_root / "receipt.json").exists():
        raise ValueError("Refuse to overwrite existing catalog or receipt")
    started = time.monotonic()
    if args.mode == "wjd":
        if args.snapshot is None:
            parser.error("wjd requires --snapshot")
        _assert_digest(args.snapshot, args.expected_snapshot_sha256)
        if args.sqlite is not None:
            _assert_digest(args.sqlite, args.expected_sqlite_sha256)
        result = project_wjd(args.snapshot, sqlite_path=args.sqlite)
    else:
        if args.tree is None or args.workbook is None or args.upstream_commit is None:
            parser.error("chord requires --tree, --workbook and --upstream-commit")
        _assert_digest(args.tree, args.expected_tree_sha256)
        _assert_digest(args.workbook, args.expected_workbook_sha256)
        result = project_chord(args.tree, args.workbook, upstream_commit=args.upstream_commit)
    result["elapsed_seconds"] = time.monotonic() - started
    receipt = write_catalog(result, args.output_root, command=[sys.executable, *sys.argv])
    print(json.dumps({"row_count": receipt["row_count"], "catalog": receipt["catalog"]}, sort_keys=True))


if __name__ == "__main__":
    main()
