"""Fetch pinned official catalogs and lexically project identity metadata only.

Mixed source bytes are retained and hashed; musical subtrees are syntax-checked
and skipped by the existing streaming lexer, never decoded into Python objects.
Only repository tree API responses and this script's metadata manifests use a
whole-document JSON decoder. Downloaded source code is never executed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterator, TextIO
from urllib.parse import quote
from urllib.request import Request, urlopen

from scripts.music_overlap_metadata import MetadataJSONError, _Lexer, _pointer


VERSION = "jazz-web-identity-projection@1.0.0"
REPOSITORIES = {"jht": "DCMLab/JazzHarmonyTreebank", "jaah": "MTG/JAAH"}
TITLE_FIELDS = frozenset({"title", "subtitle", "name", "song_name", "track_title"})
CREATOR_FIELDS = frozenset({
    "composer", "composers", "artist", "artists", "performer", "performers",
    "composer_name", "artist_name", "performer_name",
})
IDENTITY_FIELDS = TITLE_FIELDS | CREATOR_FIELDS | frozenset({
    "id", "uid", "tune_id", "track_id", "recording_id", "mbid", "musicbrainz",
    "musicbrainz_id", "musicbrainz_recording_id", "release_id", "release_mbid",
    "work_id", "work_mbid", "year", "recording_year", "album", "collection",
    "filename", "path", "url", "uri", "youtube_id", "jams_version",
})
CONTAINERS = frozenset({
    ("file_metadata",), ("file_metadata", "identifiers"),
    ("sandbox",), ("metadata",), ("identifiers",),
})


@dataclass(frozen=True)
class IdentityProjection:
    metadata: dict[str, object]
    schema: dict[str, str]


def _project(lexer: _Lexer, prefix: tuple[str, ...] = ()) -> IdentityProjection:
    lexer.expect("{")
    metadata, schema, seen = {}, {}, set()
    lexer.whitespace()
    if lexer.char() == "}":
        lexer.take()
        return IdentityProjection(metadata, schema)
    while True:
        key = lexer.string(True)
        if key in seen:
            raise MetadataJSONError("duplicate key in identity metadata object")
        seen.add(key)
        lexer.expect(":")
        path = (*prefix, key)
        kind = lexer.kind()
        pointer = _pointer(path)
        schema[pointer] = kind
        if key in IDENTITY_FIELDS and kind not in ("object", "array"):
            metadata[pointer] = lexer.scalar(True)
        elif path in CONTAINERS and kind == "object":
            child = _project(lexer, path)
            metadata.update(child.metadata)
            schema.update(child.schema)
        else:
            lexer.skip()
        lexer.whitespace()
        char = lexer.take()
        if char == "}":
            return IdentityProjection(metadata, schema)
        if char != ",":
            raise MetadataJSONError("invalid identity object delimiter")


def _lexer(handle: TextIO, chunk_size: int) -> _Lexer:
    if type(chunk_size) is not int or chunk_size < 1:
        raise MetadataJSONError("chunk_size must be a positive integer")
    return _Lexer(handle, chunk_size, 1048576)


def _finish(lexer: _Lexer) -> None:
    lexer.whitespace()
    if lexer.char():
        raise MetadataJSONError("trailing JSON content")


def iter_jht_metadata(
    source: str | Path | TextIO, *, chunk_size: int = 65536,
) -> Iterator[IdentityProjection]:
    """Read the complete top-level JHT array without constructing music values."""
    context = nullcontext(source) if hasattr(source, "read") else open(source, encoding="utf-8")
    with context as handle:
        lexer = _lexer(handle, chunk_size)
        lexer.expect("[")
        lexer.whitespace()
        if lexer.char() == "]":
            lexer.take()
        else:
            while True:
                yield _project(lexer)
                lexer.whitespace()
                char = lexer.take()
                if char == "]":
                    break
                if char != ",":
                    raise MetadataJSONError("invalid JHT array delimiter")
        _finish(lexer)


def project_jaah_metadata(
    source: str | Path | TextIO, *, chunk_size: int = 65536,
) -> IdentityProjection:
    """Project one native JAAH JSON or JAMS identity object; skip all annotations."""
    context = nullcontext(source) if hasattr(source, "read") else open(source, encoding="utf-8")
    with context as handle:
        lexer = _lexer(handle, chunk_size)
        result = _project(lexer)
        _finish(lexer)
        return result


def catalog_row(dataset: str, source_id: str, projection: IdentityProjection,
                provenance: dict) -> dict:
    def fields(allowed: frozenset[str]) -> list[dict]:
        return [{"field": key, "value": value} for key, value in projection.metadata.items()
                if key.rsplit("/", 1)[-1] in allowed
                and isinstance(value, str) and value.strip()]

    return {
        "dataset": dataset, "id": source_id,
        "titles": fields(TITLE_FIELDS), "creators": fields(CREATOR_FIELDS),
        "source_metadata": {
            **provenance, "projection_version": VERSION,
            "identity_metadata": projection.metadata,
            "shallow_schema": projection.schema,
        },
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _fetch(url: str, destination: Path, root: Path, expected_blob: str | None = None) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    receipt = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
               "destination": str(destination.relative_to(root)),
               "expected_git_blob_sha1": expected_blob}
    try:
        with urlopen(Request(url, headers={"User-Agent": VERSION}), timeout=90) as response:
            raw = response.read()
            receipt.update(http_status=response.status, headers=dict(response.headers))
        digest = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        receipt.update(bytes=len(raw), sha256=digest, git_blob_sha1=blob)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(raw)
        if expected_blob is not None and blob != expected_blob:
            raise ValueError("downloaded bytes do not match pinned Git blob")
        receipt["status"] = "fetched_verified" if expected_blob else "fetched"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
    return receipt


def fetch_catalog_sources(dataset: str, revision: str, root: Path) -> dict:
    """Fetch a pinned tree, README and identity-bearing blobs, with raw receipts."""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("revision must be an explicit 40-character Git commit SHA")
    root.mkdir(parents=True, exist_ok=True)
    repo = REPOSITORIES[dataset]
    tree_path = root / "repository_tree.json"
    tree_receipt = _fetch(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1",
                          tree_path, root)
    receipts = [tree_receipt]
    with (root / "source_fetch_receipts.jsonl").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(tree_receipt) + "\n")
    if tree_receipt["status"] == "failed":
        raise RuntimeError("repository tree acquisition failed; see receipt")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise ValueError("repository tree is truncated; completeness is unavailable")
    selected = []
    for item in tree["tree"]:
        path = item["path"]
        if item["type"] != "blob":
            continue
        identity = path == "treebank.json" if dataset == "jht" else (
            path.startswith("annotations/") and path.endswith(".json"))
        if identity or path.lower() in {"readme.md", "license.md", "license", "license.txt"}:
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts or "\\" in path:
                raise ValueError("unsafe repository-relative path")
            selected.append({"path": path, "git_blob_sha1": item["sha"],
                             "size": item["size"], "identity_source": identity})

    def fetch(item: dict) -> dict:
        return _fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{quote(item['path'])}",
                      root / "raw" / item["path"], root, item["git_blob_sha1"])

    with ThreadPoolExecutor(max_workers=6) as pool:
        for receipt in pool.map(fetch, selected):
            receipts.append(receipt)
            with (root / "source_fetch_receipts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt) + "\n")
    manifest = {"dataset": dataset, "repository": repo, "revision": revision,
                "tree_truncated": False, "selected_files": selected,
                "failed_fetches": [item for item in receipts if item["status"] == "failed"]}
    _write_json(root / "source_manifest.json", manifest)
    return {"selected_files": len(selected), "failed_fetches": len(manifest["failed_fetches"])}


def project_catalog(root: Path) -> dict:
    """Project every identity source in the complete pinned repository manifest."""
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["failed_fetches"]:
        raise ValueError("source fetch failures must remain explicit; projection not published")
    dataset = manifest["dataset"]
    rows = []
    seen_ids = set()
    for item in manifest["selected_files"]:
        if not item["identity_source"]:
            continue
        path = root / "raw" / item["path"]
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if blob != item["git_blob_sha1"] or len(raw) != item["size"]:
            raise ValueError("retained source does not match pinned blob and size")
        provenance = {"repository": manifest["repository"], "version": manifest["revision"],
                      "source_path": item["path"], "git_blob_sha1": blob,
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "source_url": f"https://github.com/{manifest['repository']}/blob/"
                                    f"{manifest['revision']}/{quote(item['path'])}",
                      "native_split": None, "split_evidence": "not supplied in projected identity fields"}
        del raw
        projections = iter_jht_metadata(path) if dataset == "jht" else [project_jaah_metadata(path)]
        for index, projection in enumerate(projections):
            pointer = f"/{index}" if dataset == "jht" else ""
            source_id = f"treebank.json#{pointer}" if dataset == "jht" else item["path"]
            if source_id in seen_ids:
                raise ValueError("duplicate catalog source identity")
            seen_ids.add(source_id)
            rows.append(catalog_row(dataset, source_id, projection,
                                    {**provenance, "record_pointer": pointer}))
    with (root / "catalog.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"dataset": dataset, "revision": manifest["revision"], "rows": len(rows),
               "identity_sources": sum(item["identity_source"] for item in manifest["selected_files"]),
               "records_without_title": sum(not row["titles"] for row in rows),
               "records_without_creator": sum(not row["creators"] for row in rows),
               "unhandled_identity_containers": [
                   {"id": row["id"], "field": key, "type": value}
                   for row in rows for key, value in row["source_metadata"]["shallow_schema"].items()
                   if key.rsplit("/", 1)[-1] in IDENTITY_FIELDS and value in {"object", "array"}],
               "access": {"raw_mixed_bytes_fetched_and_hashed": True,
                          "mixed_source_json_object_decoded": False,
                          "musical_subtrees_lexically_skipped": True,
                          "local_musical_files_read": False,
                          "downloaded_code_executed": False}}
    _write_json(root / "catalog_summary.json", summary)
    return summary


def write_population_views(root: Path) -> dict:
    """Retain full JHT rows separately from the subset with a tree field.

    This only reads the generated identity catalog. Field presence does not
    assert that the excluded tree array is nonempty or scientifically valid.
    The original catalog remains unchanged.
    """
    rows = [json.loads(line) for line in (root / "catalog.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if any(row["dataset"] != "jht" for row in rows):
        raise ValueError("population views are defined for JHT only")
    full, subset = [], []
    for row in rows:
        field_type = row["source_metadata"]["shallow_schema"].get("/trees")
        enriched = {**row, "membership": {
            "official_treebank_json_population": True,
            "trees_field_present": field_type is not None,
            "trees_field_type": field_type,
            "tree_value_inspected": False,
            "evidence": "root JSON key and type only; tree values lexically skipped",
        }}
        full.append(enriched)
        if field_type is not None:
            subset.append(enriched)
    for filename, selected in (("catalog_with_membership.jsonl", full),
                               ("catalog-annotated.jsonl", subset),
                               ("catalog_membership.jsonl", [
                                   {"dataset": row["dataset"], "id": row["id"],
                                    "membership": row["membership"]} for row in full])):
        with (root / filename).open("x", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"full_file_rows": len(full), "tree_field_present_rows": len(subset),
               "tree_values_inspected": False,
               "subset_evidence": "trees field presence only; source README describes trees as analysis field"}
    _write_json(root / "population_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fetch", "project", "population-views"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(REPOSITORIES))
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.operation == "fetch":
        if not args.dataset or not args.revision:
            parser.error("fetch requires --dataset and --revision")
        summary = fetch_catalog_sources(args.dataset, args.revision, args.root)
    elif args.operation == "project":
        summary = project_catalog(args.root)
    else:
        summary = write_population_views(args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
