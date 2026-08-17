from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.audit_dilemmadata_targets import (
    DEFAULT_MANIFEST,
    build_target_audit_report,
    manifest_projection,
)


RUN_REAL = os.environ.get("MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL,
    reason=(
        "set MUSIC_CRITIC_RUN_REAL_DILEMMADATA_TESTS=1 and "
        "MUSIC_CRITIC_DILEMMADATA_ROOT to run the full target-sidecar audit"
    ),
)


def test_full_pinned_target_manifest_and_real_e2e() -> None:
    supplied = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    assert supplied, "MUSIC_CRITIC_DILEMMADATA_ROOT must identify Dilemmadata v1.0"
    root = Path(supplied)
    assert root.is_dir()
    report = build_target_audit_report(root)
    expected = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest_projection(report) == expected

    end_to_end = report["end_to_end"]
    assert end_to_end["composition"]["an_joint"] >= 2
    assert end_to_end["composition"]["dlc"] >= 2
    assert end_to_end["merged_tie_record_ids"]
    assert end_to_end["cadence_phrase_or_section_record_ids"]
    assert end_to_end["alternative_component_record_ids"]
    assert end_to_end["alternative_target_fingerprints_are_distinct"] is True
    assert end_to_end["raw_cache_miss_count"] == end_to_end["record_count"]
    assert end_to_end["indexed_dataset_count"] == end_to_end["record_count"]
    assert end_to_end["raw_graph_fingerprints_unchanged"] is True
    assert end_to_end["model_input_fingerprints_unchanged"] is True
    assert end_to_end["candidate_identities_unchanged"] is True
    assert end_to_end["all_registered_families_in_availability"] is True
    assert end_to_end["dilemmadata_task_count"] == 22
    assert end_to_end["open_string_cpu_task_count"] > 0
    assert end_to_end["closed_tensor_task_count"] > 0
    assert end_to_end["retained_cuda_tensor_count"] == 0
    assert end_to_end["retained_prediction_tensor_count"] == 0
    assert end_to_end["theory_target_model_input_access_count"] == 0
