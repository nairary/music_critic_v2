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


REPORT_SCHEMA_VERSION = "hooktheory_midi_render_comparison_v2"


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
    maximum_tick = 0
    for track_index, tick, message in _absolute_tracks(midi):
        maximum_tick = max(maximum_tick, tick)
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
        "duration_qn": Fraction(maximum_tick, midi.ticks_per_beat),
    }


def _rational_object(value: Any) -> Fraction:
    if not isinstance(value, Mapping):
        raise ValueError("expected a rational object")
    numerator, denominator = value.get("num"), value.get("den")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator == 0
    ):
        raise ValueError("invalid rational object")
    return Fraction(numerator, denominator)


def _canonical_projection(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tempos = tuple(
        (
            _rational_object(item["onset_qn"]),
            item["microseconds_per_quarter"],
        )
        for item in document.get("tempo_events", ())
        if isinstance(item, Mapping)
    )
    meters = tuple(
        (
            _rational_object(item["onset_qn"]),
            item["numerator"],
            item["denominator"],
        )
        for item in document.get("meter_events", ())
        if isinstance(item, Mapping)
    )
    return {
        "duration_qn": _rational_object(document["duration_qn"]),
        "tempos": tempos,
        "meters": meters,
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
    canonical: Mapping[str, Any],
    *,
    example_limit: int,
    reported_quantization_error_qn: Fraction,
    reported_exact_timing: bool | None,
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
    endpoint_quantization_bound = Fraction(1, 2 * midi["ppq"])
    duration_quantization_bound = Fraction(1, midi["ppq"])
    # All values below are exact Fractions, so no floating-point tolerance is
    # needed. Keep the named slack explicit to prevent a future float rewrite
    # from silently broadening the acceptance boundary.
    fraction_technical_slack = Fraction(0)
    onset_error_fractions = [
        abs(expected[1] - actual[1])
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    offset_error_fractions = [
        abs(expected[2] - actual[2])
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    duration_error_fractions = [
        abs((expected[2] - expected[1]) - (actual[2] - actual[1]))
        for expected, actual in zip(reference_projection, rendered_projection)
    ]
    maximum_onset_error = max(onset_error_fractions, default=Fraction(0))
    maximum_offset_error = max(offset_error_fractions, default=Fraction(0))
    maximum_duration_error = max(duration_error_fractions, default=Fraction(0))
    canonical_duration_error = abs(canonical["duration_qn"] - midi["duration_qn"])
    tempo_onset_errors = [
        abs(expected[0] - actual[0])
        for expected, actual in zip(canonical["tempos"], midi["tempos"])
    ]
    meter_onset_errors = [
        abs(expected[0] - actual[0])
        for expected, actual in zip(canonical["meters"], midi["meters"])
    ]
    maximum_tempo_onset_error = max(tempo_onset_errors, default=Fraction(0))
    maximum_meter_onset_error = max(meter_onset_errors, default=Fraction(0))
    maximum_endpoint_error = max(
        maximum_onset_error,
        maximum_offset_error,
        maximum_tempo_onset_error,
        maximum_meter_onset_error,
        canonical_duration_error,
    )
    exact_required = reported_exact_timing is True
    endpoint_acceptance_bound = (
        Fraction(0) if exact_required else endpoint_quantization_bound
    )
    duration_acceptance_bound = (
        Fraction(0) if exact_required else duration_quantization_bound
    )
    timing_within_tolerance = (
        len(reference_projection) == len(rendered_projection)
        and maximum_onset_error <= endpoint_acceptance_bound + fraction_technical_slack
        and maximum_offset_error <= endpoint_acceptance_bound + fraction_technical_slack
        and maximum_duration_error <= duration_acceptance_bound + fraction_technical_slack
    )
    canonical_duration_accepted = (
        canonical_duration_error
        <= endpoint_acceptance_bound + fraction_technical_slack
    )
    canonical_tempo_accepted = (
        len(canonical["tempos"]) == len(midi["tempos"])
        and all(
            expected[1] == actual[1]
            for expected, actual in zip(canonical["tempos"], midi["tempos"])
        )
        and maximum_tempo_onset_error
        <= endpoint_acceptance_bound + fraction_technical_slack
    )
    canonical_meter_accepted = (
        len(canonical["meters"]) == len(midi["meters"])
        and all(
            expected[1:] == actual[1:]
            for expected, actual in zip(canonical["meters"], midi["meters"])
        )
        and maximum_meter_onset_error
        <= endpoint_acceptance_bound + fraction_technical_slack
    )
    report_crosscheck_violations = []
    if reported_quantization_error_qn > endpoint_quantization_bound:
        report_crosscheck_violations.append("reported_error_exceeds_ppq_bound")
    if reported_quantization_error_qn < maximum_endpoint_error:
        report_crosscheck_violations.append("reported_error_below_observed_endpoint_error")
    if reported_exact_timing is True and (
        reported_quantization_error_qn != 0
        or maximum_endpoint_error != 0
        or maximum_duration_error != 0
    ):
        report_crosscheck_violations.append("exact_report_has_nonzero_error")
    independent_violations = []
    if maximum_onset_error > endpoint_acceptance_bound + fraction_technical_slack:
        independent_violations.append(
            "exact_note_onset_error_nonzero"
            if exact_required
            else "onset_error_exceeds_ppq_bound"
        )
    if maximum_offset_error > endpoint_acceptance_bound + fraction_technical_slack:
        independent_violations.append(
            "exact_note_offset_error_nonzero"
            if exact_required
            else "offset_error_exceeds_ppq_bound"
        )
    if maximum_duration_error > duration_acceptance_bound + fraction_technical_slack:
        independent_violations.append(
            "exact_note_duration_error_nonzero"
            if exact_required
            else "duration_error_exceeds_duration_bound"
        )
    if not canonical_duration_accepted:
        independent_violations.append(
            "exact_piece_duration_error_nonzero"
            if exact_required
            else "piece_duration_error_exceeds_ppq_bound"
        )
    if not canonical_tempo_accepted:
        independent_violations.append("canonical_tempo_events_disagree")
        if exact_required and maximum_tempo_onset_error:
            independent_violations.append("exact_tempo_onset_error_nonzero")
    if not canonical_meter_accepted:
        independent_violations.append("canonical_meter_events_disagree")
        if exact_required and maximum_meter_onset_error:
            independent_violations.append("exact_meter_onset_error_nonzero")
    paired = min(len(reference_projection), len(rendered_projection))
    pitch_matches = sum(
        expected[0] == actual[0]
        for expected, actual in zip(reference_projection, rendered_projection)
    )
    symbolic_onset_errors = [float(value) for value in onset_error_fractions]
    symbolic_duration_errors = [float(value) for value in duration_error_fractions]
    expected_meters = [
        (meter.onset_qn, meter.numerator, meter.denominator)
        for meter in simplified["meters"]
    ]
    meter_regions_exact = expected_meters == list(midi["meters"])
    meter_regions_accepted = (
        len(expected_meters) == len(midi["meters"])
        and all(
            expected[1:] == actual[1:]
            and abs(expected[0] - actual[0])
            <= endpoint_acceptance_bound + fraction_technical_slack
            for expected, actual in zip(expected_meters, midi["meters"])
        )
    )
    symbolic_accepted = (
        pitch_exact
        and timing_within_tolerance
        and meter_regions_accepted
        and canonical_tempo_accepted
        and canonical_meter_accepted
        and canonical_duration_accepted
    )
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
    audio_alignment_status = (
        "ineligible"
        if tempo_alignment_status == "unavailable" or tempo_alignment_status.startswith("ineligible:")
        else "agreeing"
        if tempo_alignment_status == "within_50ms_p95"
        else "disagreeing"
    )
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
            "midi_ticks_per_quarter": midi["ppq"],
            "endpoint_quantization_bound_qn": _fraction_text(endpoint_quantization_bound),
            "duration_quantization_bound_qn": _fraction_text(duration_quantization_bound),
            "fraction_technical_slack_qn": _fraction_text(fraction_technical_slack),
            "reported_maximum_quantization_error_qn": _fraction_text(reported_quantization_error_qn),
            "reported_exact_timing": reported_exact_timing,
            "maximum_observed_onset_error_qn": _fraction_text(maximum_onset_error),
            "maximum_observed_offset_error_qn": _fraction_text(maximum_offset_error),
            "maximum_observed_duration_error_qn": _fraction_text(maximum_duration_error),
            "maximum_observed_tempo_onset_error_qn": _fraction_text(maximum_tempo_onset_error),
            "maximum_observed_meter_onset_error_qn": _fraction_text(maximum_meter_onset_error),
            "report_crosscheck_passed": not report_crosscheck_violations,
            "report_crosscheck_violations": report_crosscheck_violations,
            "independent_timing_violations": independent_violations,
            "audit_passed": not report_crosscheck_violations and not independent_violations,
            "symbolic_notes_accepted": pitch_exact and timing_within_tolerance,
            "symbolic_onset_absolute_error_qn": _percentiles(
                symbolic_onset_errors
            ),
            "symbolic_duration_absolute_error_qn": _percentiles(
                symbolic_duration_errors
            ),
            "symbolic_accepted": symbolic_accepted,
            "meter_regions_exact": meter_regions_exact,
            "meter_regions_accepted": meter_regions_accepted,
            "meter_status": (
                "exact"
                if meter_regions_exact
                else "quantized"
                if meter_regions_accepted
                else "disagrees"
            ),
            "expected_meter_regions": len(expected_meters),
            "rendered_meter_regions": len(midi["meters"]),
            "tempo_event_count": len(midi["tempos"]),
            "canonical_tempo_events_accepted": canonical_tempo_accepted,
            "canonical_meter_events_accepted": canonical_meter_accepted,
            "canonical_piece_duration_qn": _fraction_text(canonical["duration_qn"]),
            "rendered_piece_duration_qn": _fraction_text(midi["duration_qn"]),
            "canonical_piece_duration_error_qn": _fraction_text(canonical_duration_error),
            "canonical_piece_duration_accepted": canonical_duration_accepted,
            "alignment_source": alignment_name,
            "audio_alignment_status": audio_alignment_status,
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
    parser.add_argument("--audio-disagreement-output", type=Path)
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
        canonical = _canonical_projection(args.render_dir / f"{clip_id}.canonical.json")
        render_report = json.loads(
            (args.render_dir / f"{clip_id}.render-report.json").read_text(
                encoding="utf-8"
            )
        )
        error_value = render_report["render"]["maximum_quantization_error_qn"]
        reported_error = Fraction(error_value["num"], error_value["den"])
        comparison, clip_onsets, clip_durations, clip_symbolic_onsets, clip_symbolic_durations = _clip_comparison(
            clip_id,
            reference,
            midi,
            canonical,
            example_limit=args.example_limit,
            reported_quantization_error_qn=reported_error,
            reported_exact_timing=render_report.get("render", {}).get("exact_timing"),
            quality_flags=render_report.get("piece", {}).get("quality_flags", []),
        )
        comparisons.append(comparison)
        onset_errors.extend(clip_onsets)
        duration_errors.extend(clip_durations)
        symbolic_onset_errors.extend(clip_symbolic_onsets)
        symbolic_duration_errors.extend(clip_symbolic_durations)
    disagreement_details = [
        {
            "clip_id": item["clip_id"],
            "alignment_source": item["alignment_source"],
            "aligned_note_count": item["audio_aligned_note_count"],
            "onset_median_seconds": item["audio_onset_absolute_error_seconds"]["median"],
            "onset_p90_seconds": item["audio_onset_absolute_error_seconds"]["p90"],
            "onset_p95_seconds": item["audio_onset_absolute_error_seconds"]["p95"],
            "duration_median_seconds": item["audio_duration_absolute_error_seconds"]["median"],
            "duration_p90_seconds": item["audio_duration_absolute_error_seconds"]["p90"],
            "duration_p95_seconds": item["audio_duration_absolute_error_seconds"]["p95"],
            "meter_regions": [
                {
                    "onset_qn": _fraction_text(onset),
                    "numerator": numerator,
                    "denominator": denominator,
                }
                for onset, numerator, denominator in midi_item["meters"]
            ],
            "tempo_events": [
                {"onset_qn": _fraction_text(onset), "microseconds_per_quarter": tempo}
                for onset, tempo in midi_item["tempos"]
            ],
            "quality_flags": item["quality_flags"],
        }
        for item, midi_item in (
            (comparison, _midi_projection(args.render_dir / f"{comparison['clip_id']}.canonical.mid"))
            for comparison in comparisons
            if comparison["audio_alignment_status"] == "disagreeing"
        )
    ]
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
        "meter_regions_accepted_clips": sum(
            item["meter_regions_accepted"] for item in comparisons
        ),
        "meter_regions_quantization_accepted_clips": sum(
            item["meter_regions_accepted"] and not item["meter_regions_exact"]
            for item in comparisons
        ),
        "note_count_mismatch_clips": sum(
            not item["note_count_match"] for item in comparisons
        ),
        "symbolic_mismatch_clips": sum(
            not item["symbolic_accepted"] for item in comparisons
        ),
        "symbolic_notes_accepted_clips": sum(
            item["symbolic_notes_accepted"] for item in comparisons
        ),
        "symbolic_quantization_accepted_clips": sum(
            item["symbolic_notes_accepted"] and not item["symbolic_notes_exact"]
            for item in comparisons
        ),
        "meter_mismatch_clips": sum(
            not item["meter_regions_accepted"] for item in comparisons
        ),
        "canonical_tempo_mismatch_clips": sum(
            not item["canonical_tempo_events_accepted"] for item in comparisons
        ),
        "canonical_meter_mismatch_clips": sum(
            not item["canonical_meter_events_accepted"] for item in comparisons
        ),
        "canonical_duration_mismatch_clips": sum(
            not item["canonical_piece_duration_accepted"] for item in comparisons
        ),
        "pitch_mismatch_count": sum(
            item["pitch_mismatches"] for item in comparisons
        ),
        "tempo_disagreement_clips": sum(
            item["tempo_alignment_status"] == "disagrees_over_50ms_p95"
            for item in comparisons
        ),
        "symbolic_accepted_clips": sum(
            item["symbolic_accepted"] for item in comparisons
        ),
        "audio_agreement_clips": sum(
            item["audio_alignment_status"] == "agreeing" for item in comparisons
        ),
        "audio_disagreement_clips": sum(
            item["audio_alignment_status"] == "disagreeing" for item in comparisons
        ),
        "audio_ineligible_clips": sum(
            item["audio_alignment_status"] == "ineligible" for item in comparisons
        ),
        "audit_violation_clips": sum(not item["audit_passed"] for item in comparisons),
        "reported_error_crosscheck_failure_clips": sum(
            not item["report_crosscheck_passed"] for item in comparisons
        ),
        "symbolic_onset_absolute_error_qn": _percentiles(
            symbolic_onset_errors
        ),
        "symbolic_duration_absolute_error_qn": _percentiles(
            symbolic_duration_errors
        ),
        "audio_onset_absolute_error_seconds": _percentiles(onset_errors),
        "audio_duration_absolute_error_seconds": _percentiles(duration_errors),
        "audio_disagreement_details": disagreement_details,
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
    disagreement_path = args.audio_disagreement_output
    if disagreement_path is None and args.output is not None:
        disagreement_path = args.output.parent / "audio-disagreement-clips.json"
    if disagreement_path is not None:
        disagreement_path.write_text(
            json.dumps(
                {
                    "schema_version": "hooktheory_audio_disagreements_v1",
                    "audio_disagreement_clips": report["audio_disagreement_clips"],
                    "clips": report["audio_disagreement_details"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(payload, end="")
    return 1 if (
        report["missing_simplified_clips"]
        or report["symbolic_mismatch_clips"]
        or report["meter_mismatch_clips"]
        or report["audit_violation_clips"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
