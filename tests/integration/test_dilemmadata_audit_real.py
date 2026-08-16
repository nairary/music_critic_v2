from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.audit_dilemmadata import build_report, manifest_projection


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "tests" / "fixtures" / "dilemmadata" / "audit_manifest.json"
RUN_REAL = os.environ.get("MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "set MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS=1 and "
        "MUSIC_CRITIC_DILEMMADATA_ROOT to run the complete evidence audit"
    ),
)


def test_complete_recorded_dilemmadata_audit() -> None:
    supplied = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    assert supplied, "MUSIC_CRITIC_DILEMMADATA_ROOT must identify Dilemmadata v1.0"
    root = Path(supplied)
    assert root.is_dir(), f"explicit Dilemmadata root is missing: {root}"
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))

    report = build_report(root)

    assert manifest_projection(report) == expected
    assert report["readiness"]["evidence_contract_ready"] is True
    assert report["readiness"]["production_adapter_ready"] is False
