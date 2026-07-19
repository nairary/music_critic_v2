from __future__ import annotations

import ast
from argparse import Namespace
import json
from pathlib import Path

import mido

from scripts.compare_hooktheory_midi_rendering import build_report


def _write_compound_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=2)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Canonical Conductor"))
    conductor.append(mido.MetaMessage("time_signature", numerator=6, denominator=8))
    conductor.append(mido.MetaMessage("set_tempo", tempo=333_333))
    conductor.append(mido.MetaMessage("end_of_track", time=6))
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Melody"))
    notes.append(mido.Message("note_on", note=63, velocity=80, channel=0))
    notes.append(mido.Message("note_off", note=63, velocity=0, channel=0, time=2))
    notes.append(mido.MetaMessage("end_of_track", time=4))
    midi.tracks.extend((conductor, notes))
    midi.save(path)


def test_independent_comparison_maps_compound_source_beats(tmp_path: Path) -> None:
    clip_id = "compound"
    simplified_path = tmp_path / "Hooktheory.json"
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    simplified_path.write_text(
        json.dumps(
            {
                clip_id: {
                    "annotations": {
                        "meters": [
                            {"beat": 0, "beats_per_bar": 6, "beat_unit": 8}
                        ],
                        "melody": [
                            {
                                "onset": 0,
                                "offset": 2,
                                "pitch_class": 3,
                                "octave": 0,
                            }
                        ],
                    },
                    "alignment": {
                        "user": {"beats": [0, 6], "times": [10, 11]}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    _write_compound_midi(render_dir / f"{clip_id}.canonical.mid")
    (render_dir / f"{clip_id}.render-report.json").write_text(
        json.dumps(
            {
                "render": {
                    "maximum_quantization_error_qn": {"num": 0, "den": 1}
                }
            }
        ),
        encoding="utf-8",
    )
    (render_dir / "render-manifest.json").write_text(
        json.dumps(
            {
                "results": [
                    {"clip_id": clip_id, "status": "rendered"}
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_report(
        Namespace(
            simplified_path=simplified_path,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            example_limit=4,
        )
    )

    assert report["symbolic_notes_exact_clips"] == 1
    assert report["symbolic_notes_accepted_clips"] == 1
    assert report["meter_regions_exact_clips"] == 1
    comparison = report["comparisons"][0]
    assert comparison["alignment_source"] == "user"
    assert comparison["tempo_alignment_status"] == "within_50ms_p95"
    assert comparison["audio_onset_absolute_error_seconds"]["maximum"] == 0.0


def test_independent_comparison_does_not_import_production_hooktheory_adapter() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "compare_hooktheory_midi_rendering.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "music_critic.adapters.hooktheory" not in imports
    assert "music_critic.adapters" not in imports
