#!/usr/bin/env python3
"""Bounded source-only TRAIN preparation; requires a registered live plan."""
import argparse
import json
from pathlib import Path

from music_critic.critic.campaign_data import prepare_frozen_train_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-root", default="outputs/cloud_migration_20260916/prepared")
    parser.add_argument("--max-sources", type=int, default=400)
    parser.add_argument("--max-inputs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    record = json.loads(Path(args.record).read_text())
    if record["status"] not in {"planned", "running"}:
        raise ValueError("new real preparation requires a registered live experiment")
    root = Path(args.input_root)
    manifest = {"path": str(root / "data-manifest.json"), "size_bytes": 1288201,
                "sha256": "48a5301dcf83c12eb6ba5f5a00e658e3d0972221bc9db5eea5fa1e62f384dd38"}
    receipt = {"path": str(root / "receipt.json"), "size_bytes": 4632,
               "sha256": "1904b052e663ccbdc18f49ce72ec119430e6c19ba8ff125d0b5d5d34dec54f15"}
    config = prepare_frozen_train_sources(manifest, receipt, args.output,
        max_sources=args.max_sources, max_inputs=args.max_inputs, seed=args.seed)
    print(json.dumps({"experiment_id": record["experiment_id"], "dataset": config}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
