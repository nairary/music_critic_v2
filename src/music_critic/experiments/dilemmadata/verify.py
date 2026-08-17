"""Verify an immutable Phase 9B.2B plan bundle without executing training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_critic.experiments.dilemmadata import (
    verify_dilemmadata_report_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.bundle
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        (root / "command_matrix.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    verify_dilemmadata_report_bundle(plan, matrix, manifest)
    print(manifest["manifest_fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
