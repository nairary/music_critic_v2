from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.accept_pop909_cl_adapter import (
    _load_expectations,
    build_acceptance_report,
)


@pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE") != "1",
    reason="set MUSIC_CRITIC_RUN_POP909_CL_PRODUCTION_ACCEPTANCE=1",
)
def test_complete_pop909_cl_production_acceptance() -> None:
    root = Path(
        os.environ.get(
            "MUSIC_CRITIC_POP909_CL_ROOT",
            "data/pop909-cl",
        )
    )
    manifest = Path("tests/fixtures/pop909_cl/production_manifest.json")
    report = build_acceptance_report(root, _load_expectations(manifest))
    assert report["ready"], json.dumps(
        {
            "mismatches": report["mismatches"],
            "fatal_failure_count": report["fatal_failure_count"],
            "fatal_failure_samples": report["fatal_failure_samples"],
        },
        sort_keys=True,
    )
