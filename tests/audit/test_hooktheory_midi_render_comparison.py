from __future__ import annotations

import ast
from argparse import Namespace
import json
from pathlib import Path

import mido

from scripts.compare_hooktheory_midi_rendering import build_report, main


def _write_compound_midi(path: Path, *, onset_ticks: int = 0, duration_ticks: int = 2) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=2)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="Canonical Conductor"))
    conductor.append(mido.MetaMessage("time_signature", numerator=6, denominator=8))
    conductor.append(mido.MetaMessage("set_tempo", tempo=333_333))
    conductor.append(mido.MetaMessage("end_of_track", time=6))
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Melody"))
    notes.append(mido.Message("note_on", note=63, velocity=80, channel=0, time=onset_ticks))
    notes.append(mido.Message("note_off", note=63, velocity=0, channel=0, time=duration_ticks))
    notes.append(mido.MetaMessage("end_of_track", time=4))
    midi.tracks.extend((conductor, notes))
    midi.save(path)


def _write_canonical_projection(render_dir: Path, clip_id: str) -> None:
    midi = mido.MidiFile(render_dir / f"{clip_id}.canonical.mid")
    maximum_tick = max(sum(message.time for message in track) for track in midi.tracks)
    duration = {"num": maximum_tick, "den": midi.ticks_per_beat}
    (render_dir / f"{clip_id}.canonical.json").write_text(
        json.dumps(
            {
                "duration_qn": duration,
                "tempo_events": [
                    {
                        "onset_qn": {"num": 0, "den": 1},
                        "microseconds_per_quarter": 333_333,
                    }
                ],
                "meter_events": [
                    {
                        "onset_qn": {"num": 0, "den": 1},
                        "numerator": 6,
                        "denominator": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


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
    _write_canonical_projection(render_dir, clip_id)
    (render_dir / f"{clip_id}.render-report.json").write_text(
        json.dumps(
            {
                "render": {
                    "maximum_quantization_error_qn": {"num": 0, "den": 1},
                    "exact_timing": True,
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
    assert report["canonical_tempo_mismatch_clips"] == 0
    assert report["canonical_meter_mismatch_clips"] == 0
    assert report["canonical_duration_mismatch_clips"] == 0
    comparison = report["comparisons"][0]
    assert comparison["alignment_source"] == "user"
    assert comparison["tempo_alignment_status"] == "within_50ms_p95"
    assert comparison["audio_onset_absolute_error_seconds"]["maximum"] == 0.0
    assert comparison["maximum_observed_onset_error_qn"] == "0"
    assert comparison["maximum_observed_offset_error_qn"] == "0"
    assert comparison["maximum_observed_duration_error_qn"] == "0"
    assert comparison["endpoint_quantization_bound_qn"] == "1/4"
    assert comparison["duration_quantization_bound_qn"] == "1/2"
    assert comparison["audit_passed"] is True


def _comparison_case(
    tmp_path: Path,
    *,
    reference_onset: float | int,
    reference_offset: float | int,
    midi_onset_ticks: int,
    reported_error: tuple[int, int],
    reported_exact: bool,
    midi_duration_ticks: int = 2,
) -> tuple[Path, Path]:
    clip_id = "case"
    simplified_path = tmp_path / "Hooktheory.json"
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    simplified_path.write_text(
        json.dumps(
            {
                clip_id: {
                    "annotations": {
                        "meters": [{"beat": 0, "beats_per_bar": 6, "beat_unit": 8}],
                        "melody": [
                            {
                                "onset": reference_onset,
                                "offset": reference_offset,
                                "pitch_class": 3,
                                "octave": 0,
                            }
                        ],
                    },
                    "alignment": {},
                }
            }
        ),
        encoding="utf-8",
    )
    _write_compound_midi(
        render_dir / f"{clip_id}.canonical.mid",
        onset_ticks=midi_onset_ticks,
        duration_ticks=midi_duration_ticks,
    )
    _write_canonical_projection(render_dir, clip_id)
    (render_dir / f"{clip_id}.render-report.json").write_text(
        json.dumps(
            {
                "render": {
                    "maximum_quantization_error_qn": {
                        "num": reported_error[0],
                        "den": reported_error[1],
                    },
                    "exact_timing": reported_exact,
                }
            }
        ),
        encoding="utf-8",
    )
    (render_dir / "render-manifest.json").write_text(
        json.dumps({"results": [{"clip_id": clip_id, "status": "rendered"}]}),
        encoding="utf-8",
    )
    return simplified_path, render_dir


def test_huge_reported_tolerance_cannot_accept_error_beyond_ppq_bound(
    tmp_path: Path,
) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0,
        reference_offset=2,
        midi_onset_ticks=1,
        reported_error=(100, 1),
        reported_exact=False,
    )
    args = Namespace(
        simplified_path=simplified,
        render_dir=render_dir,
        render_manifest=None,
        output=None,
        audio_disagreement_output=None,
        example_limit=4,
    )

    report = build_report(args)

    assert report["symbolic_accepted_clips"] == 0
    assert report["audit_violation_clips"] == 1
    assert "onset_error_exceeds_ppq_bound" in report["comparisons"][0]["independent_timing_violations"]
    assert main(
        [
            "--simplified-path",
            str(simplified),
            "--render-dir",
            str(render_dir),
        ]
    ) == 1


def test_exact_midi_rejects_false_nonzero_reported_error(tmp_path: Path) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0,
        reference_offset=2,
        midi_onset_ticks=0,
        reported_error=(1, 8),
        reported_exact=True,
    )
    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["symbolic_notes_exact"] is True
    assert comparison["maximum_observed_onset_error_qn"] == "0"
    assert comparison["report_crosscheck_passed"] is False
    assert "exact_report_has_nonzero_error" in comparison["report_crosscheck_violations"]
    assert main(
        [
            "--simplified-path",
            str(simplified),
            "--render-dir",
            str(render_dir),
        ]
    ) == 1


def test_quantized_symbolic_error_within_independent_ppq_bound_is_accepted(
    tmp_path: Path,
) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0.4,
        reference_offset=2.4,
        midi_onset_ticks=0,
        reported_error=(1, 5),
        reported_exact=False,
    )
    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["symbolic_notes_exact"] is False
    assert comparison["symbolic_notes_accepted"] is True
    assert comparison["maximum_observed_onset_error_qn"] == "1/5"
    assert comparison["endpoint_quantization_bound_qn"] == "1/4"
    assert comparison["audit_passed"] is True


def test_opposite_endpoint_rounding_uses_full_tick_duration_bound(
    tmp_path: Path,
) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0.4,
        reference_offset=3.6,
        midi_onset_ticks=0,
        midi_duration_ticks=4,
        reported_error=(1, 5),
        reported_exact=False,
    )

    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["endpoint_quantization_bound_qn"] == "1/4"
    assert comparison["duration_quantization_bound_qn"] == "1/2"
    assert comparison["maximum_observed_onset_error_qn"] == "1/5"
    assert comparison["maximum_observed_offset_error_qn"] == "1/5"
    assert comparison["maximum_observed_duration_error_qn"] == "2/5"
    assert comparison["symbolic_notes_accepted"] is True
    assert comparison["audit_passed"] is True


def test_duration_error_above_one_tick_fails_audit_and_cli(tmp_path: Path) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0,
        reference_offset=2,
        midi_onset_ticks=0,
        midi_duration_ticks=4,
        reported_error=(0, 1),
        reported_exact=False,
    )

    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["maximum_observed_duration_error_qn"] == "1"
    assert "duration_error_exceeds_duration_bound" in comparison["independent_timing_violations"]
    assert comparison["audit_passed"] is False
    assert main(
        [
            "--simplified-path",
            str(simplified),
            "--render-dir",
            str(render_dir),
        ]
    ) == 1


def test_exact_report_rejects_nonzero_observed_duration_error(tmp_path: Path) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0.4,
        reference_offset=1.6,
        midi_onset_ticks=0,
        midi_duration_ticks=2,
        reported_error=(0, 1),
        reported_exact=True,
    )

    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["maximum_observed_duration_error_qn"] == "2/5"
    assert "exact_note_duration_error_nonzero" in comparison["independent_timing_violations"]
    assert "exact_report_has_nonzero_error" in comparison["report_crosscheck_violations"]
    assert comparison["audit_passed"] is False
    assert main(
        [
            "--simplified-path",
            str(simplified),
            "--render-dir",
            str(render_dir),
        ]
    ) == 1


def test_under_reported_endpoint_error_is_detected(tmp_path: Path) -> None:
    simplified, render_dir = _comparison_case(
        tmp_path,
        reference_onset=0.4,
        reference_offset=2.4,
        midi_onset_ticks=0,
        reported_error=(0, 1),
        reported_exact=False,
    )
    report = build_report(
        Namespace(
            simplified_path=simplified,
            render_dir=render_dir,
            render_manifest=None,
            output=None,
            audio_disagreement_output=None,
            example_limit=4,
        )
    )

    comparison = report["comparisons"][0]
    assert comparison["symbolic_notes_accepted"] is True
    assert "reported_error_below_observed_endpoint_error" in comparison["report_crosscheck_violations"]
    assert comparison["audit_passed"] is False


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
