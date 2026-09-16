#!/usr/bin/env python3
"""Registered production-only fresh-process B2 memory/equality gate."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.performance_memory_replay import (
    MemoryIndexInputs, run_registered_memory_replay,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment-id", "parent-experiment-id", "index-experiment-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-records-root", type=Path, required=True)
    for name in ("size-index", "bindings-initial", "bindings-final", "extreme-cases"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("child-timeout-seconds", "rss-sample-interval-seconds"):
        parser.add_argument("--" + name, type=float, required=True)
    for field in fields(ProductionArtifactPaths):
        parser.add_argument("--" + field.name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_registered_memory_replay(experiment_id=args.experiment_id, parent_experiment_id=args.parent_experiment_id,
        output_root=args.output_root, index_inputs=MemoryIndexInputs(args.index_experiment_id, args.index_records_root,
            args.size_index, args.bindings_initial, args.bindings_final, args.extreme_cases),
        paths=ProductionArtifactPaths(**{field.name: getattr(args, field.name) for field in fields(ProductionArtifactPaths)}),
        device="cuda", child_timeout_seconds=args.child_timeout_seconds, rss_sample_interval_seconds=args.rss_sample_interval_seconds,
        command=[sys.executable, __file__, *(sys.argv[1:] if argv is None else argv)])
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
