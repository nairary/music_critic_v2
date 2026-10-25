#!/usr/bin/env python3
"""Registered C4 POP909 preparation, native admission and training CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_critic.experiments.c4_pop909.training import main


if __name__ == "__main__":
    raise SystemExit(main())
