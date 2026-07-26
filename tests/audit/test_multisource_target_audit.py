from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_multisource_targets import (
    build_report,
    dumps_report,
    report_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "multisource"
    / "target_contract_manifest.json"
)


def test_committed_manifest_matches_deterministic_bounded_audit() -> None:
    report = build_report(REPO_ROOT)
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert committed == report
    assert dumps_report(report) == dumps_report(build_report(REPO_ROOT))
    assert report_fingerprint(report) == report_fingerprint(committed)


def test_audit_uses_bounded_evidence_and_preserves_inventory_counts() -> None:
    report = build_report(REPO_ROOT)
    assert report["scan_policy"] == {
        "manual_corpus_file_reads": False,
        "hooktheory_full_corpus_scan": False,
        "pop909_cl_full_acceptance_rerun": False,
    }
    hook = report["source_inventories"]["hooktheory"]
    assert hook["fixture_cases"] == 19
    assert hook["converted_cases"] == 18
    assert hook["skipped_unusable_cases"] == ["missing_payload"]
    assert len(hook["target_inventory"]) == 12
    pop = report["source_inventories"]["pop909_cl"]
    assert pop["logical_files"] == 909
    assert pop["accepted"] == 908
    assert pop["quarantined"] == 1
    assert pop["ambiguous_blocks"] == 5_801
    assert pop["unsupported_blocks"] == 586
    counts = {
        item["task_id"]: (item["available"], item["masked"])
        for item in pop["target_inventory"]
    }
    assert counts == {
        "pop909_cl.chord.bass": (116_055, 2),
        "pop909_cl.chord.boundary": (116_055, 2),
        "pop909_cl.chord.inversion": (109_668, 6_389),
        "pop909_cl.chord.no_chord": (947, 153),
        "pop909_cl.chord.quality": (109_800, 6_257),
        "pop909_cl.chord.root": (109_668, 6_389),
    }
