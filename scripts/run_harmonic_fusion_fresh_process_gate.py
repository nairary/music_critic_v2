#!/usr/bin/env python3
"""Run the separate generated-fixture process-boundary resume gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cuda-device-count", type=int)
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-size-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=("control", "fusion"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    worker_args = (args.worker_checkpoint, args.checkpoint_sha256, args.checkpoint_size_bytes, args.arm)
    if any(item is not None for item in worker_args) and any(item is None for item in worker_args):
        parser.error("all internal worker checkpoint bindings are required together")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error("fresh-process gate requires inherited native thread limits of one")
    if args.device == "cuda":
        if args.expected_cuda_device_count is None or args.expected_cuda_device_count <= 0:
            parser.error("CUDA requires a positive --expected-cuda-device-count from the coordinator slot")
        if args.experiment_record is None:
            parser.error("CUDA requires a planned --experiment-record before Torch initialization")
        from music_critic.experiments.registry import load_record
        record = load_record(args.experiment_record)
        if (record["status"] != "planned" or record["kind"] != "engineering_gate"
                or not record["experiment_id"].startswith("EXP-FUSION-")
                or record["protocol"].get("fresh_process_resume") is not True
                or record["budget"].get("planned") != 24
                or record["budget"].get("planned_draws") != 48):
            parser.error("a planned EXP-FUSION engineering gate must explicitly declare fresh_process_resume")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    elif args.expected_cuda_device_count is not None:
        parser.error("CUDA device count is inapplicable to CPU execution")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    from music_critic.experiments.analysisgnn.harmonic_fusion_fresh_process import (
        run_fresh_process_gate, run_resume_worker,
    )
    if args.worker_checkpoint is not None:
        result = run_resume_worker(args.output_root, checkpoint=args.worker_checkpoint,
            expected_sha256=args.checkpoint_sha256, expected_size_bytes=args.checkpoint_size_bytes,
            enabled=args.arm == "fusion", device=args.device,
            expected_cuda_device_count=args.expected_cuda_device_count)
    else:
        result = run_fresh_process_gate(args.output_root, device=args.device,
            expected_cuda_device_count=args.expected_cuda_device_count,
            experiment_record=args.experiment_record)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
