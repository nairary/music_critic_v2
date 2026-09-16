#!/usr/bin/env python3
"""Registered isolated performance replay; never resumes canonical training."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
from music_critic.experiments.analysisgnn.performance_replay import run_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", help="Sealed terminal attempt linked to this new retry")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "production"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--measured-draws", type=int, default=800)
    parser.add_argument("--warmup-draws", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--optimized-data-strategy", choices=("prepared_cache", "frozen"),
                        default="prepared_cache",
                        help="Use frozen data in both arms to isolate the optimized update")
    parser.add_argument("--cache-max-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--cache-max-entries", type=int, default=1024)
    parser.add_argument("--diagnostic", action="store_true")
    for field in fields(ProductionArtifactPaths):
        parser.add_argument("--" + field.name.replace("_", "-"), type=Path)
    args = parser.parse_args(argv)
    paths = None
    if args.mode == "production":
        values = {field.name: getattr(args, field.name) for field in fields(ProductionArtifactPaths)}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            parser.error("production requires explicit paths: " + ", ".join(missing))
        if args.device != "cuda":
            parser.error("production requires --device cuda")
        paths = ProductionArtifactPaths(**values)
    protocol = {
        "mode": args.mode, "device": args.device, "measured_draws": args.measured_draws,
        "warmup_draws": args.warmup_draws, "repeats": args.repeats,
        "batch_sizes": args.batch_sizes, "diagnostic": args.diagnostic,
        "reference_data_strategy": "frozen", "optimized_data_strategy": args.optimized_data_strategy,
        "cache_max_bytes": args.cache_max_bytes, "cache_max_entries": args.cache_max_entries,
        "quality_claims": False,
    }
    result = run_registered_replay(
        experiment_id=args.experiment_id, output_root=args.output_root, protocol=protocol,
        parent_experiment_id=args.parent_experiment_id,
        execute=lambda: run_replay(
            args.output_root, experiment_id=args.experiment_id, mode=args.mode, device=args.device,
            measured_draws=args.measured_draws, warmup_draws=args.warmup_draws,
            repeats=args.repeats, paths=paths, batch_sizes=tuple(args.batch_sizes),
            cache_max_bytes=args.cache_max_bytes, cache_max_entries=args.cache_max_entries,
            diagnostic=args.diagnostic,
            optimized_data_strategy=args.optimized_data_strategy,
        ),
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
