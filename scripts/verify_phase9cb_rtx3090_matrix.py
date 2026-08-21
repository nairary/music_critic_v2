#!/usr/bin/env python3
"""Independent verifier for the Phase 9C-B evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.experiments.phase9cb import verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    arguments = parser.parse_args()
    result = verify_bundle(arguments.bundle, expected_sha=arguments.expected_sha)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
