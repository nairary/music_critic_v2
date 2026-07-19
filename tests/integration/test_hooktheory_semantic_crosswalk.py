from __future__ import annotations

from argparse import Namespace
import os
from pathlib import Path

import pytest

from scripts.audit_hooktheory_adapter_semantics import build_report


RUN_ENV = "MUSIC_CRITIC_RUN_HOOKTHEORY_SEMANTIC_AUDIT"
pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_ENV) != "1",
    reason=f"set {RUN_ENV}=1 to run the corpus HookTheory semantic crosswalk",
)


def test_corpus_semantic_crosswalk_counts_and_evidence() -> None:
    root = Path(
        os.environ.get(
            "MUSIC_CRITIC_HOOKTHEORY_ROOT",
            Path(__file__).resolve().parents[2] / "data/HookTheory",
        )
    )
    report = build_report(
        Namespace(
            raw_path=root / "Hooktheory_Raw.json/4_merged.json",
            simplified_path=root / "Hooktheory.json",
            dataset_root=root,
            structure_root=root,
            fixture_root=Path(__file__).resolve().parents[2] / "tests/fixtures/hooktheory",
            example_limit=20,
        )
    )
    counts = report["source_accounting"]
    assert counts["raw_records"] == 26_178
    assert counts["missing_payload"] == 3
    assert counts["matched_records"] == 26_175
    assert counts["compared_meter_regions"] == 27_216
    assert counts["exact_meter_region_matches"] == 27_216
    assert counts["missing_simplified_meter_regions"] == 1
    assert counts["meter_value_mismatches"] == 0
    assert counts["paired_melody_notes"] == 1_211_093
    assert counts["pitch_class_matches"] == 1_211_093
    assert counts["pitch_class_mismatches"] == 0
    assert counts["octave_matches"] == 1_211_093
    assert counts["octave_mismatches"] == 0
    assert counts["remediated_pitch_omissions"] == 0
    compound = report["tempo"]["metrics"]
    assert compound["user:A_quarter_bpm:beatUnit=3"]["eligible_intervals"] == 72
    assert compound["user:B_raw_beat_bpm:beatUnit=3"]["eligible_intervals"] == 72
    assert compound["user:C_felt_pulse_bpm:beatUnit=3"]["eligible_intervals"] == 72
    assert compound["user:C_felt_pulse_bpm:beatUnit=3"]["within_10_percent"] == 70
    structure = report["structure"]
    assert structure["matched_rows"] == 11_515
    assert structure.get("identity_mismatches", 0) == 0
    assert structure.get("split_mismatches", 0) == 0
    assert structure.get("duplicates", 0) == 0
