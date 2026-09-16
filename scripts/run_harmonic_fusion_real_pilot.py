#!/usr/bin/env python3
"""Execute the gated reference C2/fusion pair; default mode checks admission only."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--request-size-bytes", required=True, type=int)
    parser.add_argument("--execute", action="store_true", help="Coordinator dispatch only;512 updates perarm")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.execute and args.output_root is None:
        parser.error("--execute requires a new --output-root")
    from music_critic.experiments.analysisgnn.harmonic_fusion_pilot import (
        checked_json, admit_request, run_request,
    )
    request = checked_json({"path": str(args.request.absolute()), "sha256": args.request_sha256,
                            "size_bytes": args.request_size_bytes})
    repository_root = Path(__file__).resolve().parents[1]
    if args.execute:
        result = run_request(request, args.output_root, repository_root=repository_root)
    else:
        admitted = admit_request(request, repository_root=repository_root)
        result = {"admission_valid": True, "executed": False, "cuda_initialized": False,
                  "config": admitted["config"].to_dict(), "bindings": admitted["bindings"],
                  "raw_graphs_loaded": 0, "checkpoint_tensors_loaded": 0}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
