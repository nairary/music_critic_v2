#!/usr/bin/env python3
"""Verify that the read-only legacy repository matches the captured snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_LEGACY_ROOT = Path(
    "/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic"
)
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "docs" / "legacy_snapshot.json"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return result.stdout


def _status_lines(root: Path) -> list[str]:
    return _git(root, "status", "--porcelain=v1").splitlines()


def main() -> int:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    legacy_root = Path(
        os.environ.get("MUSIC_CRITIC_LEGACY_ROOT", str(DEFAULT_LEGACY_ROOT))
    ).expanduser()

    if not legacy_root.exists():
        print(f"legacy-check: repository missing: {legacy_root}")
        return 2

    try:
        current_head = _git(legacy_root, "rev-parse", "HEAD").strip()
        current_status = _status_lines(legacy_root)
    except RuntimeError as exc:
        print(f"legacy-check: unable to inspect repository: {exc}")
        return 2

    expected_head = str(snapshot["head_commit"])
    expected_status = list(snapshot["status_porcelain_before"])
    changed = False

    if current_head != expected_head:
        print(
            "legacy-check: HEAD changed "
            f"(expected {expected_head}, found {current_head})"
        )
        changed = True

    if current_status != expected_status:
        print("legacy-check: worktree status changed")
        print("expected:")
        for line in expected_status:
            print(f"  {line}")
        print("found:")
        for line in current_status:
            print(f"  {line}")
        changed = True

    if changed:
        return 1

    print(
        "legacy-check: unchanged "
        f"(HEAD {current_head}, {len(current_status)} status entries)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
