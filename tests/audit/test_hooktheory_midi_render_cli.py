from __future__ import annotations

import json
from pathlib import Path

import mido

from scripts.render_hooktheory_midi import _deterministic_sample_records, main


def test_render_cli_writes_exact_piece_report_manifest_and_skips_missing(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "Hooktheory_Raw.json"
    raw_root.mkdir()
    raw_path = raw_root / "4_merged.json"
    output = tmp_path / "output"
    raw_path.write_text(
        json.dumps(
            {
                "compound": {
                    "hash": "compound",
                    "split": "train",
                    "json": {
                        "endBeat": 7,
                        "meters": [
                            {"beat": 1, "numBeats": 6, "beatUnit": 3}
                        ],
                        "tempos": [{"beat": 1, "bpm": 120}],
                        "keys": [
                            {"beat": 1, "tonic": "C", "scale": "minor"}
                        ],
                        "notes": [
                            {
                                "beat": 1,
                                "duration": 2,
                                "sd": "3",
                                "octave": 0,
                                "isRest": False,
                            }
                        ],
                        "chords": [],
                    },
                },
                "missing": {
                    "hash": "missing",
                    "split": "train",
                    "json": None,
                },
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--raw-path",
            str(raw_path),
            "--clip-id",
            "compound",
            "--clip-id",
            "missing",
            "--output-dir",
            str(output),
            "--no-click",
        ]
    )

    assert result == 0
    assert (output / "compound.canonical.mid").is_file()
    assert (output / "compound.canonical.json").is_file()
    report = json.loads(
        (output / "compound.render-report.json").read_text(encoding="utf-8")
    )
    assert report["piece"]["duration_qn"] == {"num": 3, "den": 1}
    assert report["piece"]["beat_count"] == 6
    assert report["render"]["rendered_clicks"] == 0
    assert report["render"]["exact_timing"] is True
    midi = mido.MidiFile(output / "compound.canonical.mid")
    assert midi.type == 1
    manifest = json.loads(
        (output / "render-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["rendered_clips"] == 1
    assert manifest["skipped_missing_payload"] == 1
    assert manifest["failed_clips"] == 0
    listening = json.loads(
        (output / "listening-manifest.json").read_text(encoding="utf-8")
    )
    assert listening["entries"][0]["meter"] == "6/8"
    assert listening["entries"][0]["mode"] == "minor"


def test_deterministic_sampler_covers_modes_meters_changes_and_shared_group(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "4_merged.json"

    def record(scale, meters, tempos, note_beat=1):
        return {
            "json": {
                "keys": [{"beat": 1, "tonic": "C", "scale": scale}],
                "meters": meters,
                "tempos": tempos,
                "notes": [
                    {
                        "beat": note_beat,
                        "duration": 1,
                        "sd": "1",
                        "octave": 0,
                        "isRest": False,
                    }
                ],
                "chords": [],
            }
        }

    raw_path.write_text(
        json.dumps(
            {
                "a": record(
                    "dorian",
                    [{"beat": 1, "numBeats": 9, "beatUnit": 3}],
                    [{"beat": 1, "bpm": 100}],
                ),
                "b": record(
                    "major",
                    [
                        {"beat": 1, "numBeats": 4, "beatUnit": 1},
                        {"beat": 5, "numBeats": 12, "beatUnit": 3},
                    ],
                    [{"beat": 1, "bpm": 100}, {"beat": 5, "bpm": 120}],
                    note_beat=1.5,
                ),
                "c": record(
                    "minor",
                    [{"beat": 1, "numBeats": 6, "beatUnit": 3}],
                    [{"beat": 1, "bpm": 90}],
                ),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "HookTheoryStructure.train.jsonl").write_text(
        json.dumps({"audio_path": "audio/b.mp3", "ori_uid": "shared"})
        + "\n"
        + json.dumps({"audio_path": "audio/c.mp3", "ori_uid": "shared"})
        + "\n",
        encoding="utf-8",
    )

    selected, records = _deterministic_sample_records(raw_path, tmp_path)
    categories = {
        category
        for _clip_id, _case_id, tags in selected
        for category in tags
    }

    assert set(records) == {"a", "b", "c"}
    assert {
        "major",
        "minor",
        "mode:dorian",
        "meter:6/8",
        "meter:9/8",
        "meter:12/8",
        "multiple_meters",
        "multiple_tempos",
        "fractional_timing",
        "shared_ori_uid",
    } <= categories
