"""CLI for globally validated source/lineage-closed split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from music_critic.tasks import (
    dump_split_manifest,
    load_corpus_index,
    plan_group_hash_split,
    validate_split_manifest,
)


def _ratio(value: str) -> tuple[str, float]:
    try:
        name, raw = value.split("=", 1)
        ratio = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ratio must use NAME=FLOAT"
        ) from exc
    if not name or ratio <= 0:
        raise argparse.ArgumentTypeError(
            "ratio requires a name and a positive value"
        )
    return name, ratio


def make_split(
    index_paths: Sequence[str | Path],
    *,
    ratios: Sequence[tuple[str, float]],
    seed: int,
    output: str | Path,
) -> None:
    if not index_paths:
        raise ValueError("training.split.indices_required")
    ratio_map = dict(ratios)
    if len(ratio_map) != len(ratios):
        raise ValueError("training.split.duplicate_ratio")
    indices = tuple(load_corpus_index(path) for path in index_paths)
    manifest = plan_group_hash_split(
        indices,
        seed=seed,
        ratios=ratio_map,
    )
    validate_split_manifest(manifest, indices)
    dump_split_manifest(manifest, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one target-blind globally validated corpus split"
        )
    )
    parser.add_argument(
        "--index",
        action="append",
        required=True,
        help="versioned corpus index JSON; repeat for each source",
    )
    parser.add_argument(
        "--ratio",
        action="append",
        required=True,
        type=_ratio,
        help="split allocation such as train=0.8",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    make_split(
        arguments.index,
        ratios=arguments.ratio,
        seed=arguments.seed,
        output=arguments.output,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "make_split"]
