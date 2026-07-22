from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.audit_pop909 import build_report


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "fixtures" / "pop909" / "audit_manifest.json"
RUN_REAL_POP909 = os.environ.get("MUSIC_CRITIC_RUN_REAL_POP909_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_POP909,
    reason="set MUSIC_CRITIC_RUN_REAL_POP909_TESTS=1 with MUSIC_CRITIC_POP909_ROOT",
)


def test_complete_recorded_pop909_audit() -> None:
    supplied = os.environ.get("MUSIC_CRITIC_POP909_ROOT")
    assert supplied, "MUSIC_CRITIC_POP909_ROOT must explicitly identify the corpus root"
    root = Path(supplied)
    assert root.is_dir(), f"explicit POP909 root is missing: {root}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = build_report(root)

    for key, expected in manifest["corpus"].items():
        assert report["identity"][key] == expected, key
    crosswalk = report["generic_midi_crosswalk"]
    assert crosswalk["attempted"] == manifest["corpus"]["primary_midi_count"]
    assert crosswalk["converted"] == manifest["expected_crosswalk"]["converted"]
    assert crosswalk["failed"] == manifest["expected_crosswalk"]["failed"]
    assert [failure["song_id"] for failure in crosswalk["failures"]] == manifest[
        "expected_crosswalk"
    ]["failed_song_ids"]
    assert report["annotations"]["parser_failure_count"] == 0
    assert all(
        row["equal"]
        for row in crosswalk["serialization_round_trip_sample"]
    )

    files = {
        row["path"]: row["sha256"] for row in report["discovery"]["files"]
    }
    audited_cases = {row["song_id"]: row for row in report["golden_evidence"]}
    for case in manifest["cases"]:
        assert files[case["primary_midi"]] == case["sha256"]
        assert audited_cases[case["song_id"]]["expected"] == case["expected"]
        assert audited_cases[case["song_id"]]["reasons"] == case["reasons"]
