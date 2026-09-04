#!/usr/bin/env python3
"""Independent Phase 9C-D evidence verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.experiments.phase9cd.runner import verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    result = verify_bundle(args.bundle, expected_sha=args.expected_sha)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
