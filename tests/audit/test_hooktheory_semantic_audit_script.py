from __future__ import annotations

import ast
from argparse import Namespace
import json
from pathlib import Path

from scripts.audit_hooktheory_adapter_semantics import build_report


def test_semantic_audit_is_v1_isolated_and_compares_synthetic_semantics(tmp_path: Path) -> None:
    raw_path = tmp_path / "4_merged.json"
    simplified_path = tmp_path / "Hooktheory.json"
    structure_root = tmp_path / "structure"
    structure_root.mkdir()
    raw_path.write_text(
        json.dumps(
            {
                "clip": {
                    "hash": "clip",
                    "split": "train",
                    "json": {
                        "endBeat": 7,
                        "meters": [{"beat": 1, "numBeats": 6, "beatUnit": 3}],
                        "tempos": [{"beat": 1, "bpm": 120, "swingFactor": 0}],
                        "keys": [{"beat": 1, "tonic": "C", "scale": "minor"}],
                        "notes": [{"beat": 1, "duration": 2, "sd": "3", "octave": 0, "isRest": False}],
                        "chords": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    simplified_path.write_text(
        json.dumps(
            {
                "clip": {
                    "split": "TRAIN",
                    "hooktheory": {"id": "clip"},
                    "alignment": {
                        "swing": "STRAIGHT",
                        "user": {"beats": [0, 6], "times": [0, 1]},
                        "refined": None,
                    },
                    "annotations": {
                        "num_beats": 6,
                        "meters": [{"beat": 0, "beats_per_bar": 6, "beat_unit": 8}],
                        "melody": [{"onset": 0, "offset": 2, "pitch_class": 3, "octave": 0}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (structure_root / "HookTheoryStructure.train.jsonl").write_text(
        json.dumps({"audio_path": "audio/clip.mp3", "ori_uid": "group"}) + "\n",
        encoding="utf-8",
    )
    report = build_report(
        Namespace(
            raw_path=raw_path,
            simplified_path=simplified_path,
            dataset_root=tmp_path,
            structure_root=structure_root,
            fixture_root=None,
            example_limit=4,
        )
    )
    counts = report["source_accounting"]
    assert counts["exact_meter_region_matches"] == 1
    assert counts["meter_value_mismatches"] == 0
    assert counts["missing_raw_meter_regions"] == 0
    assert counts["missing_simplified_meter_regions"] == 0
    assert counts["pitch_class_matches"] == 1
    assert counts["pitch_class_mismatches"] == 0
    assert counts["octave_matches"] == 1
    assert counts["octave_mismatches"] == 0
    assert counts["remediated_pitch_omissions"] == 0
    assert report["tempo"]["metrics"]["user:C_felt_pulse_bpm:beatUnit=3"]["within_1_percent"] == 1

    source_path = Path(__file__).resolve().parents[2] / "scripts/audit_hooktheory_adapter_semantics.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("legacy" in value or "music_critic_v1" in value for value in imports)
