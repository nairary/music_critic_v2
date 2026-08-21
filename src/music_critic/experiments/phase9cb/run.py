"""CLI for the Phase 9C-B one-seed 2x2 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import build_plan
from .runner import execute


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "profile", "run", "resume", "aggregate", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("phase9cb.cli.config_mapping_required")
    result = execute(arguments.output_root.resolve(), build_plan(config), action=arguments.action)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
