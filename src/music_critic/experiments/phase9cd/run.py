"""CLI for Phase 9C-D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import build_plan
from .runner import execute, finalize, verify_bundle


def _milestones(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("continue", "finalize", "verify"))
    parser.add_argument("--parent-output-root", type=Path)
    parser.add_argument("--mlp-reference-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--start-update", type=int, default=3000)
    parser.add_argument("--target-update", type=int, default=15000)
    parser.add_argument("--validation-milestones", type=_milestones, default=(3000, 6000, 9000, 12000, 15000))
    args = parser.parse_args()
    if args.action == "continue":
        if args.parent_output_root is None or args.mlp_reference_root is None or args.config is None:
            parser.error("parent, MLP reference, and config are required")
        result = execute(
            args.output_root,
            build_plan(
                args.parent_output_root,
                args.config,
                args.mlp_reference_root,
                start_update=args.start_update,
                target_update=args.target_update,
                validation_milestones=args.validation_milestones,
            ),
        )
    elif args.action == "finalize":
        if args.expected_sha is None:
            parser.error("--expected-sha is required")
        result = finalize(args.output_root, expected_sha=args.expected_sha)
    else:
        result = verify_bundle(args.output_root, expected_sha=args.expected_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
