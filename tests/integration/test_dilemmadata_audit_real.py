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
        "MUSIC_CRITIC_DILEMMADATA_ROOT plus MUSIC_CRITIC_DILEMMADATA_UPSTREAM_ROOT "
        "to run the complete acceptance-backed evidence audit"
    ),
)


def test_complete_recorded_dilemmadata_audit() -> None:
    supplied = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    assert supplied, "MUSIC_CRITIC_DILEMMADATA_ROOT must identify Dilemmadata v1.0"
    root = Path(supplied)
    assert root.is_dir(), f"explicit Dilemmadata root is missing: {root}"
    upstream_supplied = os.environ.get("MUSIC_CRITIC_DILEMMADATA_UPSTREAM_ROOT")
    assert upstream_supplied, (
        "MUSIC_CRITIC_DILEMMADATA_UPSTREAM_ROOT must identify a separate clean v1.0 checkout"
    )
    upstream_root = Path(upstream_supplied)
    assert upstream_root.is_dir(), f"explicit upstream checkout is missing: {upstream_root}"
    assert upstream_root.resolve() != root.resolve()
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))

    report = build_report(root, upstream_root=upstream_root)

    assert manifest_projection(report) == expected
    comparison = report["corpus_identity"]["upstream_comparison"]
    assert comparison["performed"] is True
    assert comparison["exact_match"] is True
    assert comparison["checkout_clean"] is True
    assert comparison["failure_categories"] == []
    assert report["readiness"]["evidence_contract_ready"] is True
    assert report["readiness"]["acceptance_backed_release_ready"] is True
    assert report["readiness"]["production_adapter_ready"] is False
