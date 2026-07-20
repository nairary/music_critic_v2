from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from music_critic.adapters import MidiAdapterConfig, load_midi_piece
from music_critic.adapters._json_stream import iter_jsonl, iter_object_records
from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.data import dump_piece, load_piece
from music_critic.exporters import MidiRenderConfig, write_piece_midi
from scripts.compare_hooktheory_midi_rendering import build_parser, build_report
from scripts.audit_hooktheory_midi_ambiguities import (
    build_parser as build_ambiguity_parser,
    build_report as build_ambiguity_report,
)
from scripts.render_hooktheory_midi import main as render_main


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/hooktheory"
DATA_ROOT = Path(
    os.environ.get(
        "MUSIC_CRITIC_HOOKTHEORY_ROOT", REPO_ROOT / "data/HookTheory"
    )
)
RAW_PATH = DATA_ROOT / "Hooktheory_Raw.json/4_merged.json"
SIMPLIFIED_PATH = DATA_ROOT / "Hooktheory.json"


pytestmark = pytest.mark.skipif(
    os.environ.get("MUSIC_CRITIC_RUN_HOOKTHEORY_MIDI_RENDERER") != "1",
    reason=(
        "set MUSIC_CRITIC_RUN_HOOKTHEORY_MIDI_RENDERER=1 for real "
        "HookTheory renderer tests"
    ),
)


def _cases() -> list[dict]:
    manifest = json.loads(
        (FIXTURE_ROOT / "golden_manifest.json").read_text(encoding="utf-8")
    )
    return [
        json.loads(
            (FIXTURE_ROOT / "cases" / f"{case_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for case_id in manifest["cases"]
    ]


def _raw_records(ids: set[str]) -> dict[str, dict]:
    selected = {}
    for clip_id, record in iter_object_records(RAW_PATH):
        if clip_id in ids:
            selected[clip_id] = record
    assert set(selected) == ids
    return selected


def _structure_rows(ids: set[str]) -> dict[str, dict]:
    selected = {}
    for split in ("train", "val", "test"):
        for _line, row in iter_jsonl(
            DATA_ROOT / f"HookTheoryStructure.{split}.jsonl"
        ):
            audio_path = row.get("audio_path")
            if isinstance(audio_path, str) and Path(audio_path).stem in ids:
                selected[Path(audio_path).stem] = row
    return selected


def _effective_notes(piece):
    tracks = {track.track_id: track for track in piece.tracks}
    return sorted(
        (
            note.pitch,
            note.onset_qn.to_fraction(),
            (note.onset_qn + note.duration_qn).to_fraction(),
            note.velocity if note.velocity is not None else 96,
            (
                note.channel
                if note.channel is not None
                else (
                    tracks[note.track_id].channel
                    if tracks[note.track_id].channel is not None
                    else (9 if note.is_percussion else 0)
                )
            ),
            (
                note.program
                if note.program is not None
                else (
                    tracks[note.track_id].program
                    if tracks[note.track_id].program is not None
                    else 0
                )
            ),
            note.is_percussion,
        )
        for note in piece.notes
    )


def _loaded_notes_without_click(piece):
    tracks = {track.track_id: track for track in piece.tracks}
    return sorted(
        (
            note.pitch,
            note.onset_qn.to_fraction(),
            (note.onset_qn + note.duration_qn).to_fraction(),
            note.velocity,
            note.channel,
            note.program,
            note.is_percussion,
        )
        for note in piece.notes
        if tracks[note.track_id].name != "Canonical Click"
    )


def test_all_real_golden_cases_render_round_trip_and_compare_independently(
    tmp_path: Path,
) -> None:
    cases = _cases()
    ids = {case["source_reference"]["clip_id"] for case in cases}
    raw = _raw_records(ids)
    structures = _structure_rows(ids)
    render_results = []
    rendered = skipped = strict_exact = quantized = 0

    for case in sorted(cases, key=lambda item: item["source_reference"]["clip_id"]):
        clip_id = case["source_reference"]["clip_id"]
        record = raw[clip_id]
        if record.get("json") is None:
            skipped += 1
            with pytest.raises(HookTheoryAdapterError, match="no usable json payload"):
                convert_hooktheory_record(
                    clip_id,
                    record,
                    config=HookTheoryAdapterConfig("HookTheory"),
                    source_path=str(RAW_PATH),
                )
            continue

        visible = convert_hooktheory_record(
            clip_id,
            record,
            config=HookTheoryAdapterConfig("HookTheory", include_targets=True),
            structure_row=structures.get(clip_id),
            source_path=str(RAW_PATH),
        )
        hidden = convert_hooktheory_record(
            clip_id,
            record,
            config=HookTheoryAdapterConfig("HookTheory", include_targets=False),
            structure_row=structures.get(clip_id),
            source_path=str(RAW_PATH),
        )
        assert hidden.targets == ()
        assert not any(
            span.layer == "target_alignment" for span in hidden.annotations
        )
        target_count = len(visible.targets)
        annotation_count = len(visible.annotations)

        json_path = tmp_path / f"{clip_id}.canonical.json"
        midi_path = tmp_path / f"{clip_id}.canonical.mid"
        dump_piece(visible, json_path)
        assert load_piece(json_path) == visible
        report = write_piece_midi(
            visible,
            midi_path,
            config=MidiRenderConfig(
                require_exact_timing=False,
                include_click_track=True,
                include_target_markers=True,
            ),
        )
        hidden_report = write_piece_midi(
            hidden,
            tmp_path / f"{clip_id}.hidden.mid",
            config=MidiRenderConfig(
                require_exact_timing=False,
                include_click_track=False,
                include_target_markers=True,
            ),
        )
        assert len(visible.targets) == target_count
        assert len(visible.annotations) == annotation_count
        assert hidden_report.rendered_markers == 0
        assert report.rendered_clicks == len(visible.beats)
        assert report.rendered_notes == len(visible.notes)
        assert report.rendered_markers > 0

        round_trip = load_midi_piece(
            midi_path,
            config=MidiAdapterConfig(dataset_name="HookTheory-render-roundtrip"),
        )
        expected_notes = _effective_notes(visible)
        actual_notes = _loaded_notes_without_click(round_trip)
        assert len(actual_notes) == len(expected_notes)
        tolerance = report.maximum_quantization_error_qn.to_fraction()
        for expected, actual in zip(expected_notes, actual_notes, strict=True):
            assert actual[0] == expected[0]
            assert abs(actual[1] - expected[1]) <= tolerance
            assert abs(actual[2] - expected[2]) <= tolerance
            assert actual[3:] == expected[3:]
        assert abs(
            round_trip.duration_qn.to_fraction() - visible.duration_qn.to_fraction()
        ) <= tolerance
        assert [event.microseconds_per_quarter for event in round_trip.tempo_events] == [
            event.microseconds_per_quarter for event in visible.tempo_events
        ]
        assert [
            (event.numerator, event.denominator) for event in round_trip.meter_events
        ] == [
            (event.numerator, event.denominator) for event in visible.meter_events
        ]

        report_value = asdict(report)
        error = report.maximum_quantization_error_qn
        report_value["maximum_quantization_error_qn"] = {
            "num": error.num,
            "den": error.den,
        }
        (tmp_path / f"{clip_id}.render-report.json").write_text(
            json.dumps({"render": report_value}), encoding="utf-8"
        )
        render_results.append({"clip_id": clip_id, "status": "rendered"})
        rendered += 1
        strict_exact += report.exact_timing
        quantized += report.timing_quantized

    assert (rendered, skipped, strict_exact, quantized) == (18, 1, 17, 1)
    (tmp_path / "render-manifest.json").write_text(
        json.dumps({"results": render_results}), encoding="utf-8"
    )
    args = build_parser().parse_args(
        [
            "--simplified-path",
            str(SIMPLIFIED_PATH),
            "--render-dir",
            str(tmp_path),
            "--example-limit",
            "4",
        ]
    )
    comparison = build_report(args)
    assert comparison["compared_clips"] == 18
    assert comparison["note_count_mismatch_clips"] == 0
    assert comparison["meter_mismatch_clips"] == 0
    assert comparison["canonical_tempo_mismatch_clips"] == 0
    assert comparison["canonical_meter_mismatch_clips"] == 0
    assert comparison["canonical_duration_mismatch_clips"] == 0
    assert comparison["symbolic_notes_exact_clips"] == 17
    assert comparison["symbolic_quantization_accepted_clips"] == 1
    assert comparison["symbolic_notes_accepted_clips"] == 18
    assert comparison["symbolic_mismatch_clips"] == 0
    assert comparison["audit_violation_clips"] == 0
    assert comparison["symbolic_accepted_clips"] == 18
    assert (
        comparison["audio_agreement_clips"]
        + comparison["audio_disagreement_clips"]
        + comparison["audio_ineligible_clips"]
        == 18
    )


def test_real_review_package_generates_every_audit_report(tmp_path: Path) -> None:
    output = tmp_path / "review-package"

    result = render_main(
        [
            "--raw-path",
            str(RAW_PATH),
            "--simplified-path",
            str(SIMPLIFIED_PATH),
            "--structure-root",
            str(DATA_ROOT),
            "--manifest",
            str(FIXTURE_ROOT / "golden_manifest.json"),
            "--output-dir",
            str(output),
            "--allow-timing-quantization",
        ]
    )

    assert result == 0
    for name in (
        "render-manifest.json",
        "listening-manifest.json",
        "comparison-report.json",
        "audio-disagreement-clips.json",
        "ambiguity-report.json",
    ):
        assert (output / name).is_file(), name
    render_manifest = json.loads(
        (output / "render-manifest.json").read_text(encoding="utf-8")
    )
    assert render_manifest["rendered_clips"] == 18
    assert render_manifest["skipped_missing_payload"] == 1
    comparison = json.loads(
        (output / "comparison-report.json").read_text(encoding="utf-8")
    )
    assert comparison["symbolic_accepted_clips"] == 18
    assert comparison["audit_violation_clips"] == 0


def test_full_corpus_ambiguity_audit_streams_all_usable_clips() -> None:
    args = build_ambiguity_parser().parse_args(
        ["--raw-path", str(RAW_PATH), "--example-limit", "4"]
    )

    report = build_ambiguity_report(args)

    assert report["total_clips"] == 26_178
    assert report["usable_clips"] == 26_175
    assert report["missing_payload_clips"] == 3
    assert report["failed_clips"] == 0
    assert report["total_notes"] == 1_228_022
    assert report["clips_with_same_pitch_overlaps"] == 102
    assert report["same_pitch_overlap_pairs"] == 1_802
    assert report["same_pitch_nested_pairs"] == 1_627
    assert report["clips_with_channel_program_conflicts"] == 0
    assert report["channel_program_conflict_pairs"] == 0
