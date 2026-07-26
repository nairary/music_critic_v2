#!/usr/bin/env python3
"""Deterministically audit a portable corpus index and its cache artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.tasks import (
    CorpusCacheConfig,
    dumps_corpus_index,
    load_cached_piece,
    load_corpus_index,
    load_split_manifest,
    validate_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", type=Path, required=True)
    parser.add_argument(
        "--cache-root", action="append", type=Path, required=True
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional bounded artifact verification count.",
    )
    args = parser.parse_args()

    if len(args.index) != len(args.cache_root):
        raise SystemExit("--index and --cache-root counts must match")
    sources = tuple(
        sorted(
            (
                (
                    load_corpus_index(index_path),
                    index_path,
                    CorpusCacheConfig(cache_root),
                )
                for index_path, cache_root in zip(
                    args.index, args.cache_root, strict=True
                )
            ),
            key=lambda row: row[0].header.dataset_id,
        )
    )
    indices = tuple(index for index, _path, _cache in sources)
    caches = tuple(cache for _index, _path, cache in sources)
    for index, path, _cache in sources:
        canonical_index = dumps_corpus_index(index)
        if (
            args.check
            and path.read_text(encoding="utf-8") != canonical_index
        ):
            raise SystemExit(
                "corpus index is not in canonical deterministic form"
            )
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    record_caches = tuple(
        (record, cache)
        for index, cache in zip(indices, caches, strict=True)
        for record in (
            index.records
            if args.limit is None
            else index.records[: args.limit]
        )
    )
    for record, cache in record_caches:
        load_cached_piece(record, cache)
    manifest_fingerprint = None
    split_counts: dict[str, int] = {}
    if args.split_manifest is not None:
        manifest = load_split_manifest(args.split_manifest)
        validate_split_manifest(manifest, indices)
        manifest_fingerprint = manifest.manifest_fingerprint
        for assignment in manifest.assignments:
            split_counts[assignment.split] = (
                split_counts.get(assignment.split, 0) + 1
            )
    report = {
        "artifact_count": len(record_caches),
        "indexed_count": sum(len(index.records) for index in indices),
        "dataset_ids": [index.header.dataset_id for index in indices],
        "index_fingerprints": [
            [index.header.dataset_id, index.header.index_fingerprint]
            for index in indices
        ],
        "manifest_fingerprint": manifest_fingerprint,
        "split_counts": dict(sorted(split_counts.items())),
        "status": "ok",
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is not None:
        if args.check:
            if (
                not args.output.is_file()
                or args.output.read_text(encoding="utf-8") != payload
            ):
                raise SystemExit("dataset audit output is stale")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
