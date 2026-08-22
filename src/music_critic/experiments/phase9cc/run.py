"""CLI for Phase 9C-C plan/run/resume/verify operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import build_plan
from .runner import execute, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "run", "resume", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha")
    arguments = parser.parse_args()
    if arguments.action == "verify":
        result = verify_bundle(
            arguments.output_root.resolve(),
            expected_sha=arguments.expected_sha,
        )
    else:
        if arguments.config is None:
            parser.error("--config is required for plan/run/resume")
        config = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("phase9cc.cli.config_mapping_required")
        result = execute(
            arguments.output_root.resolve(),
            build_plan(config),
            action=arguments.action,
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
