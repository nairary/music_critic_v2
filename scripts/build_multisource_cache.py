#!/usr/bin/env python3
"""Explicit offline HookTheory/POP909-CL canonical cache builder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from music_critic.tasks import (
    CorpusCacheConfig,
    build_hooktheory_corpus_cache,
    build_pop909_cl_corpus_cache,
    dump_corpus_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional bounded smoke limit; omit only for an intentional full build.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Keep canonical symbolic inputs but omit auxiliary targets.",
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    hook = subparsers.add_parser("hooktheory")
    hook.add_argument("--raw-path", type=Path, required=True)
    hook.add_argument("--structure-root", type=Path)
    hook.add_argument("--dataset-id", default="hooktheory")
    pop = subparsers.add_parser("pop909_cl")
    pop.add_argument("--corpus-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    cache = CorpusCacheConfig(args.cache_root)
    if args.dataset == "hooktheory":
        index, report = build_hooktheory_corpus_cache(
            args.raw_path,
            cache_config=cache,
            dataset_id=args.dataset_id,
            include_targets=not args.raw_only,
            structure_root=args.structure_root,
            limit=args.limit,
        )
    else:
        index, report = build_pop909_cl_corpus_cache(
            args.corpus_root,
            cache_config=cache,
            include_targets=not args.raw_only,
            limit=args.limit,
        )
    dump_corpus_index(index, args.index_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(
            asdict(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": report.accepted_count,
                "dataset_id": report.dataset_id,
                "index_fingerprint": report.index_fingerprint,
                "quarantined": report.quarantined_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
