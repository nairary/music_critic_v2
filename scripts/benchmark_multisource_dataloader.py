#!/usr/bin/env python3
"""Run a bounded, non-acceptance benchmark of the production lazy loader."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from music_critic.tasks import (
    CorpusCacheConfig,
    DeterministicQuotaSampler,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    MultiSourceDataLoaderConfig,
    benchmark_multisource_dataloader,
    load_corpus_index,
    load_split_manifest,
    make_multisource_dataloader,
)


def _corpus(value: str) -> tuple[Path, Path]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "corpus must be INDEX:CACHE_ROOT"
        )
    return tuple(Path(part) for part in parts)  # type: ignore[return-value]


def _weight(value: str) -> tuple[str, float]:
    dataset_id, separator, raw = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("weight must be DATASET_ID=VALUE")
    try:
        weight = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be numeric") from exc
    return dataset_id, weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=_corpus, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--weight", action="append", type=_weight, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, choices=(0, 2), default=0)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    datasets = []
    for index_path, cache_root in args.corpus:
        index = load_corpus_index(index_path)
        datasets.append(
            IndexedMultiSourceDataset(
                index, cache_config=CorpusCacheConfig(cache_root)
            )
        )
    mixed = MultiCorpusDataset(
        datasets,
        load_split_manifest(args.split_manifest),
        split=args.split,
    )
    sampler = DeterministicQuotaSampler(
        mixed,
        weights=dict(args.weight),
        seed=args.seed,
        epoch_size=args.epoch_size,
    )
    loader = make_multisource_dataloader(
        mixed,
        sampler=sampler,
        config=MultiSourceDataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            prefetch_factor=2 if args.num_workers else None,
            multiprocessing_context="spawn" if args.num_workers else None,
        ),
    )
    report = benchmark_multisource_dataloader(
        loader, max_batches=args.max_batches
    )
    print(json.dumps(asdict(report), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
