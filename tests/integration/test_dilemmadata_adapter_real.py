from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.accept_dilemmadata_adapter import (
    DEFAULT_MANIFEST,
    build_acceptance_report,
    manifest_projection,
)


def _corpus_root() -> Path:
    value = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    if not value:
        pytest.skip("MUSIC_CRITIC_DILEMMADATA_ROOT is not configured")
    return Path(value)


def test_pinned_full_corpus_matches_production_manifest(tmp_path: Path) -> None:
    report = build_acceptance_report(
        _corpus_root(),
        work_dir=tmp_path / "acceptance",
        manifest_path=DEFAULT_MANIFEST,
    )
    expected = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert report["intrinsic_ready"] is True
    assert report["manifest_check"]["passed"] is True
    assert report["ready"] is True
    assert manifest_projection(report) == expected
