#!/usr/bin/env python3
"""Write deterministic bounded Phase 8B.1 mechanics evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from music_critic.ssl.phase8b_acceptance import (
    run_phase8b_bounded_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    arguments = parser.parse_args()
    report = run_phase8b_bounded_comparison(
        seed=arguments.seed,
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["all_variant_train_overfit_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
