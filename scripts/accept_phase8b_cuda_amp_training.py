#!/usr/bin/env python3
"""Run and archive independent Phase 8B.1 CUDA FP32/AMP acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from music_critic.ssl.phase8b_cuda_acceptance import (
    PHASE8B_CUDA_PARITY_ABSOLUTE_TOLERANCE,
    PHASE8B_CUDA_PARITY_RELATIVE_TOLERANCE,
    Phase8BCudaAcceptanceError,
    run_phase8b_cuda_training_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--parity-rtol",
        type=float,
        default=PHASE8B_CUDA_PARITY_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--parity-atol",
        type=float,
        default=PHASE8B_CUDA_PARITY_ABSOLUTE_TOLERANCE,
    )
    arguments = parser.parse_args()
    try:
        report = run_phase8b_cuda_training_acceptance(
            arguments.output_dir,
            expected_head=arguments.expected_head,
            seed=arguments.seed,
            steps=arguments.steps,
            relative_tolerance=arguments.parity_rtol,
            absolute_tolerance=arguments.parity_atol,
        )
    except Phase8BCudaAcceptanceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
