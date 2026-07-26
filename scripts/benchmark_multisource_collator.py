#!/usr/bin/env python3
"""Run a lightweight prepared-sample Phase 5B.1 collator benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from music_critic.data import load_piece
from music_critic.graph import build_raw_graph
from music_critic.tasks import (
    benchmark_multisource_collator,
    build_multisource_sample,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "canonical_json",
        type=Path,
        nargs="?",
        default=Path("tests/fixtures/data/canonical_piece_v2.json"),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    piece = replace(
        load_piece(args.canonical_json),
        annotations=(),
        targets=(),
    )
    sample = build_multisource_sample(piece, build_raw_graph(piece))
    evidence = benchmark_multisource_collator(
        (sample,) * args.samples,
        repeats=args.repeats,
    )
    print(
        json.dumps(
            asdict(evidence),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
