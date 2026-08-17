"""Materialize a Phase 9B.2B plan and command matrix; never execute cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from music_critic.experiments.dilemmadata import (
    build_dilemmadata_experiment_plan,
    dilemmadata_command_matrix,
    dilemmadata_report_bundle_manifest,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "raw-index-fingerprint",
        "target-cache-index-fingerprint",
        "split-manifest-fingerprint",
        "sample-schedule-fingerprint",
        "phase7a-encoder-export-path",
        "phase7a-encoder-export-sha256",
        "phase7a-source-checkpoint-sha256",
        "phase8b-encoder-export-path",
        "phase8b-encoder-export-sha256",
        "phase8b-source-checkpoint-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    output = values.pop("output")
    plan = build_dilemmadata_experiment_plan(**values)
    matrix = dilemmadata_command_matrix(plan)
    manifest = dilemmadata_report_bundle_manifest(plan, matrix)
    _write(output / "plan.json", plan)
    _write(output / "command_matrix.json", matrix)
    _write(output / "bundle_manifest.json", manifest)
    print(plan["plan_fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
