"""Project a pinned POP909 metadata workbook; never follow musical-file paths.

This small reader supports XLSX string/numeric metadata, not spreadsheet formulas
or music annotations. Missing cells remain null and original text stays intact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import time
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SCHEMA = "POP909MetadataCatalog@1.0.0"


def _xml(archive: ZipFile, name: str) -> ET.Element:
    content = archive.read(name)
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("DTD/entities are unsupported")
    return ET.fromstring(content)


def _cells(row: ET.Element, strings: list[str]) -> dict[int, str | None]:
    cells = {}
    for cell in row.findall("m:c", NS):
        reference = cell.get("r", "")
        match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", reference)
        if not match or match.group(2) != row.get("r"):
            raise ValueError(f"Invalid cell coordinate: {reference}")
        column = 0
        for letter in match.group(1):
            column = column * 26 + ord(letter) - ord("A") + 1
        if column in cells:
            raise ValueError("Duplicate cell coordinate")
        if cell.find("m:f", NS) is not None:
            raise ValueError("Formula metadata is unsupported")
        kind = cell.get("t", "n")
        value = cell.findtext("m:v", default=None, namespaces=NS)
        if kind == "inlineStr":
            value = "".join(node.text or "" for node in cell.findall("m:is//m:t", NS))
        elif kind == "s":
            if value is None or not re.fullmatch(r"[0-9]+", value) or int(value) >= len(strings):
                raise ValueError("Invalid shared-string index")
            value = strings[int(value)]
        elif kind not in {"n", "str", "b", "d"}:
            raise ValueError(f"Unsupported cell type: {kind}")
        cells[column] = value
    return cells


def read_catalog(path: Path, *, source_url: str, upstream_commit: str) -> dict:
    """Return metadata rows and byte provenance from one POP909 worksheet."""
    content = path.read_bytes()
    if len(content) > 2_000_000:
        raise ValueError("Unexpectedly large metadata workbook")
    sha256 = hashlib.sha256(content).hexdigest()
    with ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 200:
            raise ValueError("Duplicate or excessive workbook members")
        if sum(info.file_size for info in archive.infolist()) > 30_000_000:
            raise ValueError("Unexpectedly large uncompressed workbook")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError("Unsafe workbook member path")
        book = _xml(archive, "xl/workbook.xml")
        sheets = book.findall("m:sheets/m:sheet", NS)
        if len(sheets) != 1 or sheets[0].get("name") != "POP909":
            raise ValueError("Expected exactly one POP909 worksheet")
        relations = _xml(archive, "xl/_rels/workbook.xml.rels")
        relation_id = sheets[0].get(f"{{{REL_NS}}}id")
        matching = [r for r in relations if r.get("Id") == relation_id]
        if len(matching) != 1 or matching[0].get("TargetMode") == "External":
            raise ValueError("Worksheet relationship is missing or external")
        target = matching[0].get("Target", "")
        member = target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
        if ".." in PurePosixPath(member).parts or member not in names:
            raise ValueError("Invalid worksheet relationship path")
        strings = []
        if "xl/sharedStrings.xml" in names:
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in _xml(archive, "xl/sharedStrings.xml").findall("m:si", NS)
            ]
        worksheet = _xml(archive, member)
        source_rows = worksheet.findall("m:sheetData/m:row", NS)
        if not source_rows:
            raise ValueError("Empty metadata worksheet")
        headers = _cells(source_rows[0], strings)
        if any(not value for value in headers.values()) or len(set(headers.values())) != len(headers):
            raise ValueError("Empty or duplicate metadata header")
        if not {"song_id", "name", "artist"} <= set(headers.values()):
            raise ValueError("Required song_id/name/artist headers are missing")
        rows, seen_ids, seen_rows = [], set(), set()
        for source_row in source_rows[1:]:
            cells = _cells(source_row, strings)
            if not any(value is not None and value != "" for value in cells.values()):
                continue
            if set(cells) - set(headers):
                raise ValueError("Metadata cell has no column header")
            row_number = int(source_row.get("r", "0"))
            if row_number <= int(source_rows[0].get("r", "0")) or row_number in seen_rows:
                raise ValueError("Invalid or duplicate worksheet row")
            seen_rows.add(row_number)
            metadata = {header: cells.get(column) for column, header in headers.items()}
            raw_id = metadata["song_id"]
            if raw_id is None or not re.fullmatch(r"[0-9]{1,3}(?:\.0+)?", raw_id):
                raise ValueError("Invalid POP909 song ID")
            number = int(raw_id.split(".")[0])
            if not 1 <= number <= 909:
                raise ValueError("POP909 song ID outside 001..909")
            song_id = f"{number:03d}"
            if song_id in seen_ids:
                raise ValueError("Duplicate POP909 song ID")
            seen_ids.add(song_id)
            if not metadata["name"] or not metadata["name"].strip():
                raise ValueError("Missing POP909 title")
            creators = []
            if metadata["artist"] is not None and metadata["artist"].strip():
                creators.append({"field": "artist", "value": metadata["artist"]})
            rows.append({
                "schema_version": SCHEMA,
                "dataset": "pop909",
                "id": song_id,
                "titles": [{"field": "name", "value": metadata["name"]}],
                "creators": creators,
                "source_metadata": metadata,
                "provenance": {
                    "source_url": source_url,
                    "upstream_commit": upstream_commit,
                    "source_sha256": sha256,
                    "sheet": "POP909",
                    "row": row_number,
                },
            })
    return {"schema_version": SCHEMA, "headers": list(headers.values()), "source_sha256": sha256,
            "source_size_bytes": len(content), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise ValueError("Refuse to overwrite existing catalog or receipt")
    started = time.monotonic()
    result = read_catalog(args.input, source_url=args.source_url, upstream_commit=args.upstream_commit)
    if result["source_sha256"] != args.expected_sha256:
        raise ValueError("Source SHA256 mismatch")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["rows"])
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    receipt = {key: value for key, value in result.items() if key != "rows"}
    receipt.update({
        "command": [sys.executable, *sys.argv],
        "source_url": args.source_url,
        "upstream_commit": args.upstream_commit,
        "source_path": str(args.input),
        "row_count": len(result["rows"]),
        "output_path": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "output_size_bytes": len(payload.encode("utf-8")),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "wall_seconds": time.monotonic() - started,
        "musical_payload_read": False,
        "annotations_read": False,
        "claim": "Metadata catalog only; no confirmed work identity or musical alignment",
    })
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"row_count": receipt["row_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
