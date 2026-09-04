#!/usr/bin/env python3
"""Independent source-free verifier for Phase 9C-A evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.experiments.phase9c import safe_extract_members, verify_bundle
from music_critic.experiments.phase9c.artifacts import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--archive", type=Path)
    arguments = parser.parse_args()
    protocol = read_json(arguments.bundle / "protocol.json")
    if protocol.get("repository", {}).get("git_head") != arguments.expected_sha:
        raise ValueError("phase9c.verify.git_head_mismatch")
    if protocol.get("repository", {}).get("clean") is not True:
        raise ValueError("phase9c.verify.clean_head_not_attested")
    result = verify_bundle(arguments.bundle)
    if arguments.archive is not None:
        members = safe_extract_members(arguments.archive)
        result = {**result, "archive_member_count": len(members)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
