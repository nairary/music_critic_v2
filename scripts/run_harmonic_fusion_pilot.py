#!/usr/bin/env python3
"""Run the isolated source-free fusion trainer/resume gate. Real pilot dispatch stays gated."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True,
                        help="Generated-fixture 24-update acceptance; never the 512-update scientific pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--experiment-record", type=Path,
                        help="Required pre-existing planned engineering gate for CUDA; coordinator finalizes it")
    args = parser.parse_args(argv)
    if args.device == "cuda":
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if os.environ.get(name) != "1":
                parser.error("CUDA gate requires inherited native thread limits of one")
        if args.experiment_record is None:
            parser.error("CUDA gate requires a planned --experiment-record before Torch initialization")
        from music_critic.experiments.registry import load_record
        record = load_record(args.experiment_record)
        if (record["status"] != "planned" or record["kind"] != "engineering_gate"
                or not record["experiment_id"].startswith("EXP-FUSION-")):
            parser.error("fresh planned EXP-FUSION engineering_gate required")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    from music_critic.experiments.analysisgnn.harmonic_fusion_run import run_synthetic_gate
    result = run_synthetic_gate(args.output_root, device=args.device)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
