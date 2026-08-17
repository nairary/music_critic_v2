#!/usr/bin/env python3
"""Build or independently check the immutable Dilemmadata TargetBundle cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from music_critic.tasks import (
    CorpusCacheConfig,
    DilemmadataTargetCacheBuildReport,
    DilemmadataTargetCacheConfig,
    build_dilemmadata_target_cache,
    check_dilemmadata_target_cache,
    dilemmadata_target_cache_manifest,
    dump_dilemmadata_target_cache_index,
    load_corpus_index,
    load_dilemmadata_target_cache_index,
)


def _write_json_atomic(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--raw-index",
        type=Path,
        default=Path("data/cache/dilemmadata.index.json"),
    )
    parser.add_argument(
        "--raw-cache-root",
        type=Path,
        default=Path("data/cache/dilemmadata"),
    )
    parser.add_argument(
        "--target-cache-root",
        type=Path,
        default=Path("data/cache/dilemmadata-target"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/cache/dilemmadata-target.index.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/dilemmadata/target_cache_manifest.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    raw_index = load_corpus_index(args.raw_index)
    cache_config = DilemmadataTargetCacheConfig(args.target_cache_root)
    if args.check:
        index = load_dilemmadata_target_cache_index(args.index)
        checked = check_dilemmadata_target_cache(
            index,
            raw_index=raw_index,
            cache_config=cache_config,
        )
        report = DilemmadataTargetCacheBuildReport(
            record_count=int(checked["record_count"]),
            cache_hit_count=int(checked["record_count"]),
            cache_miss_count=0,
            index_fingerprint=index.index_fingerprint,
            target_bundle_fingerprint=str(
                checked["target_bundle_fingerprint"]
            ),
        )
        expected = dilemmadata_target_cache_manifest(index, report)
        actual = json.loads(args.manifest.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(
                "dilemmadata.target_cache.manifest_mismatch"
            )
        print(json.dumps(checked, sort_keys=True))
        return 0
    if args.source_root is None:
        raise SystemExit("--source-root is required unless --check is used")
    index, report = build_dilemmadata_target_cache(
        args.source_root,
        raw_index=raw_index,
        raw_cache_config=CorpusCacheConfig(args.raw_cache_root),
        target_cache_config=cache_config,
        limit=args.limit,
    )
    dump_dilemmadata_target_cache_index(index, args.index)
    manifest = dilemmadata_target_cache_manifest(index, report)
    _write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
