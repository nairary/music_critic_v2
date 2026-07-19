#!/usr/bin/env python3
"""Independently compare rendered MIDI with simplified HookTheory evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import mido

from music_critic.adapters._json_stream import iter_object_records


REPORT_SCHEMA_VERSION = "hooktheory_midi_render_comparison_v1"


def _exact(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("expected a JSON number")
    return Fraction(str(value))


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class _Meter:
    source_beat: Fraction
    onset_qn: Fraction
    numerator: int
    denominator: int


def _meters(annotations: Mapping[str, Any]) -> tuple[_Meter, ...]:
    candidates: list[tuple[Fraction, int, int, int]] = []
    values = annotations.get("meters")
    if isinstance(values, list):
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                continue
            try:
                beat = _exact(value.get("beat"))
                numerator = value.get("beats_per_bar")
                denominator = value.get("beat_unit")
                if (
                    isinstance(numerator, bool)
                    or not isinstance(numerator, int)
                    or isinstance(denominator, bool)
                    or not isinstance(denominator, int)
                    or numerator <= 0
                    or denominator <= 0
                ):
                    continue
                candidates.append((beat, index, numerator, denominator))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    candidates.sort()
    if not candidates:
        candidates = [(Fraction(0), -1, 4, 4)]
    selected: list[tuple[Fraction, int, int]] = []
    for beat, _index, numerator, denominator in candidates:
        if selected and selected[-1][0] == beat:
            continue
        selected.append((beat, numerator, denominator))
    result: list[_Meter] = []
    onset_qn = Fraction(0)
    for index, (beat, numerator, denominator) in enumerate(selected):
        if index:
            previous = result[-1]
            onset_qn = previous.onset_qn + (
                beat - previous.source_beat
            ) * Fraction(4, previous.denominator)
        result.append(_Meter(beat, onset_qn, numerator, denominator))
    return tuple(result)


def _beat_to_qn(beat: Fraction, meters: tuple[_Meter, ...]) -> Fraction:
    active = meters[0]
    for meter in meters[1:]:
        if meter.source_beat > beat:
            break
        active = meter
    return active.onset_qn + (beat - active.source_beat) * Fraction(
        4, active.denominator
    )


def _simplified_index(path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for clip_id, record in iter_object_records(path):
        if clip_id not in selected or not isinstance(record, Mapping):
            continue
        annotations = record.get("annotations")
        alignment = record.get("alignment")
        annotations = annotations if isinstance(annotations, Mapping) else {}
        alignment = alignment if isinstance(alignment, Mapping) else {}
        meters = _meters(annotations)
        melody = []
        values = annotations.get("melody")
        if isinstance(values, list):
            for index, value in enumerate(values):
                if not isinstance(value, Mapping):
                    continue
                try:
                    onset_beat = _exact(value.get("onset"))
                    offset_beat = _exact(value.get("offset"))
                    pitch_class = value.get("pitch_class")
                    octave = value.get("octave")
                    if (
                        isinstance(pitch_class, bool)
                        or not isinstance(pitch_class, int)
                        or isinstance(octave, bool)
                        or not isinstance(octave, int)
                    ):
                        continue
                    melody.append(
                        {
                            "index": index,
                            "pitch": 60 + 12 * octave + pitch_class,
                            "onset_beat": onset_beat,
                            "offset_beat": offset_beat,
                            "onset_qn": _beat_to_qn(onset_beat, meters),
                            "offset_qn": _beat_to_qn(offset_beat, meters),
                        }
                    )
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
        point_sets: dict[str, tuple[tuple[Fraction, Fraction], ...]] = {}
        for name in ("refined", "user"):
            value = alignment.get(name)
            if not isinstance(value, Mapping):
                continue
            beats, times = value.get("beats"), value.get("times")
            if not isinstance(beats, list) or not isinstance(times, list):
                continue
            if len(beats) != len(times) or len(beats) < 2:
                continue
            try:
                points = tuple(
                    zip(
                        (_exact(item) for item in beats),
                        (_exact(item) for item in times),
                        strict=True,
                    )
                )
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if all(points[index][0] < points[index + 1][0] for index in range(len(points) - 1)):
                point_sets[name] = points
        result[clip_id] = {
            "meters": meters,
            "melody": tuple(melody),
            "point_sets": point_sets,
            "swing": alignment.get("swing"),
        }
    return result


def _absolute_tracks(midi: mido.MidiFile) -> Iterable[tuple[int, int, mido.Message]]:
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += message.time
            yield track_index, tick, message


def _midi_projection(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    track_names = {index: track.name for index, track in enumerate(midi.tracks)}
    open_notes: dict[tuple[int, int, int], deque[tuple[int, int]]] = defaultdict(deque)
    notes = []
    meters = []
    tempos = []
    for track_index, tick, message in _absolute_tracks(midi):
        if message.type == "time_signature":
            meters.append(
                (Fraction(tick, midi.ticks_per_beat), message.numerator, message.denominator)
            )
        elif message.type == "set_tempo":
            tempos.append((Fraction(tick, midi.ticks_per_beat), message.tempo))
        elif track_names.get(track_index) != "Canonical Click" and message.type == "note_on" and message.velocity > 0:
            open_notes[(track_index, message.channel, message.note)].append(
                (tick, message.velocity)
            )
        elif track_names.get(track_index) != "Canonical Click" and (
            message.type == "note_off"
            or (message.type == "note_on" and message.velocity == 0)
        ):
            key = (track_index, message.channel, message.note)
            if open_notes[key]:
                onset, velocity = open_notes[key].popleft()
                notes.append(
                    {
                        "pitch": message.note,
                        "onset_qn": Fraction(onset, midi.ticks_per_beat),
                        "offset_qn": Fraction(tick, midi.ticks_per_beat),
                        "velocity": velocity,
                    }
                )
    notes.sort(key=lambda item: (item["onset_qn"], item["pitch"], item["offset_qn"]))
    tempos.sort()
    meters.sort()
    return {
        "ppq": midi.ticks_per_beat,
        "notes": tuple(notes),
        "meters": tuple(meters),
        "tempos": tuple(tempos),
    }


def _interpolate(points: tuple[tuple[Fraction, Fraction], ...], beat: Fraction) -> Fraction | None:
    if beat < points[0][0] or beat > points[-1][0]:
        return None
    for (left_beat, left_time), (right_beat, right_time) in zip(points, points[1:]):
        if left_beat <= beat <= right_beat:
            if right_beat == left_beat:
                return left_time
            ratio = (beat - left_beat) / (right_beat - left_beat)
            return left_time + ratio * (right_time - left_time)
    return points[-1][1]


def _qn_to_seconds(qn: Fraction, tempos: tuple[tuple[Fraction, int], ...]) -> Fraction:
    if not tempos:
        tempos = ((Fraction(0), 500_000),)
    active_onset, active_tempo = tempos[0]
    elapsed = Fraction(0)
    for onset, tempo in tempos[1:]:
        if onset >= qn:
            break
        elapsed += (onset - active_onset) * Fraction(active_tempo, 1_000_000)
        active_onset, active_tempo = onset, tempo
    return elapsed + (qn - active_onset) * Fraction(active_tempo, 1_000_000)


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "maximum": None}
    ordered = sorted(values)

    def percentile(value: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * value))]

    return {
        "count": len(values),
        "median": median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def _clip_comparison(
    clip_id: str,
    simplified: Mapping[str, Any],
    midi: Mapping[str, Any],
    *,
    example_limit: int,
    timing_tolerance_qn: Fraction,
    quality_flags: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    reference_notes = sorted(
        simplified["melody"],
        key=lambda item: (item["onset_qn"], item["pitch"], item["offset_qn"]),
    )
    rendered_notes = midi["notes"]
    reference_projection = [
        (item["pitch"], item["onset_qn"], item["offset_qn"])
        for item in reference_notes
    ]
    rendered_projection = [
        (item["pitch"], item["onset_qn"], item["offset_qn"])
        for item in rendered_notes
    ]
    pitch_exact = len(reference_projection) == len(rendered_projection) and all(
        expected[0] == actual[0]
        for expected, actual in zip(reference_projection, rendered_projection)
    )
    symbolic_errors = [
        max(abs(expected[1] - actual[1]), abs(expected[2] - actual[2]))
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    timing_within_tolerance = (
        len(reference_projection) == len(rendered_projection)
        and all(error <= timing_tolerance_qn for error in symbolic_errors)
    )
    paired = min(len(reference_projection), len(rendered_projection))
    pitch_matches = sum(
        expected[0] == actual[0]
        for expected, actual in zip(reference_projection, rendered_projection)
    )
    symbolic_onset_errors = [
        float(abs(expected[1] - actual[1]))
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    symbolic_duration_errors = [
        float(
            abs(
                (expected[2] - expected[1])
                - (actual[2] - actual[1])
            )
        )
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    expected_meters = [
        (meter.onset_qn, meter.numerator, meter.denominator)
        for meter in simplified["meters"]
    ]
    mismatches = []
    for index, (expected, actual) in enumerate(
        zip(reference_projection, rendered_projection)
    ):
        if expected != actual and len(mismatches) < example_limit:
            mismatches.append(
                {
                    "index": index,
                    "expected": [expected[0], _fraction_text(expected[1]), _fraction_text(expected[2])],
                    "actual": [actual[0], _fraction_text(actual[1]), _fraction_text(actual[2])],
                }
            )

    onset_audio_errors: list[float] = []
    duration_audio_errors: list[float] = []
    point_sets = simplified["point_sets"]
    alignment_name = "refined" if "refined" in point_sets else "user" if "user" in point_sets else None
    audio_ineligibility = None
    if alignment_name is None:
        audio_ineligibility = "alignment_unavailable_or_non_monotonic"
    elif len(simplified["meters"]) != 1:
        audio_ineligibility = "meter_changes"
    elif len(midi["tempos"]) != 1:
        audio_ineligibility = "tempo_changes"
    elif simplified["swing"] not in (None, False, 0, "STRAIGHT", "straight"):
        audio_ineligibility = "unsupported_swing"
    if alignment_name is not None and audio_ineligibility is None:
        points = point_sets[alignment_name]
        alignment_zero = _interpolate(points, Fraction(0))
        if alignment_zero is None:
            alignment_zero = points[0][1]
        for expected, actual in zip(reference_notes, rendered_notes):
            expected_onset = _interpolate(points, expected["onset_beat"])
            expected_offset = _interpolate(points, expected["offset_beat"])
            if expected_onset is None or expected_offset is None:
                continue
            expected_onset -= alignment_zero
            expected_offset -= alignment_zero
            actual_onset = _qn_to_seconds(actual["onset_qn"], midi["tempos"])
            actual_offset = _qn_to_seconds(actual["offset_qn"], midi["tempos"])
            onset_audio_errors.append(float(abs(actual_onset - expected_onset)))
            duration_audio_errors.append(
                float(abs((actual_offset - actual_onset) - (expected_offset - expected_onset)))
            )
    audio_onset_summary = _percentiles(onset_audio_errors)
    if alignment_name is None:
        tempo_alignment_status = "unavailable"
    elif audio_ineligibility is not None:
        tempo_alignment_status = f"ineligible:{audio_ineligibility}"
    elif audio_onset_summary["p95"] is not None and audio_onset_summary["p95"] <= 0.05:
        tempo_alignment_status = "within_50ms_p95"
    else:
        tempo_alignment_status = "disagrees_over_50ms_p95"
    return (
        {
            "clip_id": clip_id,
            "reference_note_count": len(reference_projection),
            "rendered_note_count": len(rendered_projection),
            "paired_notes": paired,
            "unpaired_canonical_notes": len(rendered_projection) - paired,
            "unpaired_reference_notes": len(reference_projection) - paired,
            "pitch_matches": pitch_matches,
            "pitch_mismatches": paired - pitch_matches,
            "note_count_match": len(reference_projection) == len(rendered_projection),
            "symbolic_notes_exact": reference_projection == rendered_projection,
            "symbolic_pitch_exact": pitch_exact,
            "symbolic_timing_tolerance_qn": _fraction_text(timing_tolerance_qn),
            "maximum_symbolic_timing_error_qn": _fraction_text(
                max(symbolic_errors, default=Fraction(0))
            ),
            "symbolic_notes_accepted": pitch_exact and timing_within_tolerance,
            "symbolic_onset_absolute_error_qn": _percentiles(
                symbolic_onset_errors
            ),
            "symbolic_duration_absolute_error_qn": _percentiles(
                symbolic_duration_errors
            ),
            "meter_regions_exact": expected_meters == list(midi["meters"]),
            "meter_status": (
                "exact" if expected_meters == list(midi["meters"]) else "disagrees"
            ),
            "expected_meter_regions": len(expected_meters),
            "rendered_meter_regions": len(midi["meters"]),
            "tempo_event_count": len(midi["tempos"]),
            "alignment_source": alignment_name,
            "tempo_alignment_status": tempo_alignment_status,
            "audio_alignment_ineligibility": audio_ineligibility,
            "audio_aligned_note_count": len(onset_audio_errors),
            "audio_onset_absolute_error_seconds": audio_onset_summary,
            "audio_duration_absolute_error_seconds": _percentiles(
                duration_audio_errors
            ),
            "mismatch_examples": mismatches,
            "quality_flags": quality_flags,
        },
        onset_audio_errors,
        duration_audio_errors,
        symbolic_onset_errors,
        symbolic_duration_errors,
    )


def _selected_clips(render_dir: Path, manifest_path: Path | None) -> list[str]:
    path = manifest_path or render_dir / "render-manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError("render manifest has no results list")
    return sorted(
        item["clip_id"]
        for item in results
        if isinstance(item, Mapping)
        and item.get("status") == "rendered"
        and isinstance(item.get("clip_id"), str)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simplified-path", required=True, type=Path)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=10)
    return parser


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    clips = _selected_clips(args.render_dir, args.render_manifest)
    simplified = _simplified_index(args.simplified_path, set(clips))
    comparisons = []
    onset_errors: list[float] = []
    duration_errors: list[float] = []
    symbolic_onset_errors: list[float] = []
    symbolic_duration_errors: list[float] = []
    missing_simplified = []
    for clip_id in clips:
        reference = simplified.get(clip_id)
        if reference is None:
            missing_simplified.append(clip_id)
            continue
        midi = _midi_projection(args.render_dir / f"{clip_id}.canonical.mid")
        render_report = json.loads(
            (args.render_dir / f"{clip_id}.render-report.json").read_text(
                encoding="utf-8"
            )
        )
        error_value = render_report["render"]["maximum_quantization_error_qn"]
        timing_tolerance = Fraction(error_value["num"], error_value["den"])
        comparison, clip_onsets, clip_durations, clip_symbolic_onsets, clip_symbolic_durations = _clip_comparison(
            clip_id,
            reference,
            midi,
            example_limit=args.example_limit,
            timing_tolerance_qn=timing_tolerance,
            quality_flags=render_report.get("piece", {}).get("quality_flags", []),
        )
        comparisons.append(comparison)
        onset_errors.extend(clip_onsets)
        duration_errors.extend(clip_durations)
        symbolic_onset_errors.extend(clip_symbolic_onsets)
        symbolic_duration_errors.extend(clip_symbolic_durations)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "selected_clips": len(clips),
        "compared_clips": len(comparisons),
        "missing_simplified_clips": missing_simplified[: args.example_limit],
        "symbolic_notes_exact_clips": sum(
            item["symbolic_notes_exact"] for item in comparisons
        ),
        "meter_regions_exact_clips": sum(
            item["meter_regions_exact"] for item in comparisons
        ),
        "note_count_mismatch_clips": sum(
            not item["note_count_match"] for item in comparisons
        ),
        "symbolic_mismatch_clips": sum(
            not item["symbolic_notes_accepted"] for item in comparisons
        ),
        "symbolic_notes_accepted_clips": sum(
            item["symbolic_notes_accepted"] for item in comparisons
        ),
        "symbolic_quantization_accepted_clips": sum(
            item["symbolic_notes_accepted"] and not item["symbolic_notes_exact"]
            for item in comparisons
        ),
        "meter_mismatch_clips": sum(
            not item["meter_regions_exact"] for item in comparisons
        ),
        "pitch_mismatch_count": sum(
            item["pitch_mismatches"] for item in comparisons
        ),
        "tempo_disagreement_clips": sum(
            item["tempo_alignment_status"] == "disagrees_over_50ms_p95"
            for item in comparisons
        ),
        "symbolic_onset_absolute_error_qn": _percentiles(
            symbolic_onset_errors
        ),
        "symbolic_duration_absolute_error_qn": _percentiles(
            symbolic_duration_errors
        ),
        "audio_onset_absolute_error_seconds": _percentiles(onset_errors),
        "audio_duration_absolute_error_seconds": _percentiles(duration_errors),
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.example_limit < 0:
        raise SystemExit("--example-limit must be non-negative")
    try:
        report = build_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if (
        report["missing_simplified_clips"]
        or report["symbolic_mismatch_clips"]
        or report["meter_mismatch_clips"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
