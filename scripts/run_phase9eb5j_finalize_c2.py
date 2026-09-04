#!/usr/bin/env python3
"""Finalize an already completed Phase 9E-B5H C2 run after B5J repair."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from music_critic.experiments.analysisgnn.validation_eligibility_repair import (
    run_repaired_full_orbit_diagnostic_validation,
)


RUNNER_PATH = Path(__file__).with_name(
    "run_phase9eb5h_analysisgnn_full_orbit.py"
)


def _patched_runner() -> ModuleType:
    name = "music_critic_phase9eb5h_repaired_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Phase 9E-B5H runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    module.run_full_orbit_diagnostic_validation = (
        run_repaired_full_orbit_diagnostic_validation
    )
    return module


def main() -> int:
    return int(_patched_runner().main())


if __name__ == "__main__":
    raise SystemExit(main())
