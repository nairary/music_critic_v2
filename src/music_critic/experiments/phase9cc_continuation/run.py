"""CLI for Phase 9C-C continuation execution and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import build_continuation_plan
from .runner import execute, finalize, verify_bundle


def _milestones(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comma-separated integers required") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("continue", "finalize", "verify"))
    parser.add_argument("--parent-output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--start-update", type=int, default=9000)
    parser.add_argument("--target-update", type=int, default=15000)
    parser.add_argument(
        "--validation-milestones", type=_milestones, default=(9000, 12000, 15000)
    )
    arguments = parser.parse_args()
    root = arguments.output_root.expanduser().resolve()
    if arguments.action == "verify":
        result = verify_bundle(root, expected_sha=arguments.expected_sha)
    elif arguments.action == "finalize":
        if arguments.expected_sha is None:
            parser.error("--expected-sha is required for finalize")
        result = finalize(root, expected_sha=arguments.expected_sha)
    else:
        if arguments.parent_output_root is None or arguments.config is None:
            parser.error("--parent-output-root and --config are required")
        result = execute(
            root,
            build_continuation_plan(
                arguments.parent_output_root,
                arguments.config,
                start_update=arguments.start_update,
                target_update=arguments.target_update,
                validation_milestones=arguments.validation_milestones,
            ),
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
