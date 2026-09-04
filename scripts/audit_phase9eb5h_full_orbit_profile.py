#!/usr/bin/env python3
"""Audit the Phase 9E-B5H full-orbit profile without opening TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from music_critic.experiments.analysisgnn.full_orbit_training import (
    check_full_orbit_fixture,
    compact_full_orbit_fixture,
    excluded_pair_evidence,
    full_orbit_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5h_full_orbit_profile.json"
DEFAULT_OUTPUT = ROOT / "outputs/phase9eb5h/full-orbit-profile"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build(*, fixture: Path, output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preflight = full_orbit_preflight()
    exclusions = excluded_pair_evidence()
    _write_json(output / "preflight.json", preflight)
    _write_json(output / "excluded_pairs.json", {"pairs": list(exclusions)})
    _write_json(fixture, compact_full_orbit_fixture(preflight))
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = (
        check_full_orbit_fixture(args.fixture)
        if args.check
        else build(fixture=args.fixture, output=args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
