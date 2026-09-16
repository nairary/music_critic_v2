#!/usr/bin/env python3
"""Register the fixed 108-update synthetic historical C2 equivalence gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    # The analysisgnn package __init__ imports Torch. Validate before importing
    # that package, not merely before invoking the gate's lazy helpers.
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if args.device == "cuda" and any(os.environ.get(name) != "1" for name in names):
        parser.error("c2_equivalence.cuda_requires_inherited_thread_limits")
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    from music_critic.experiments.analysisgnn.performance_c2_equivalence import run_registered_c2_gate
    result = run_registered_c2_gate(experiment_id=args.experiment_id,
        parent_experiment_id=args.parent_experiment_id, output_root=args.output_root, device=args.device)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
