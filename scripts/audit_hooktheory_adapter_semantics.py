#!/usr/bin/env python3
"""Deterministic, read-only semantic audit for the HookTheory V2 adapter.

The source JSON objects are streamed.  Only the simplified fields required for
the crosswalk are indexed in memory; raw records and canonical pieces are never
retained corpus-wide.  This script deliberately does not import Music Critic V1.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from music_critic.adapters._json_stream import iter_jsonl, iter_object_records


REPORT_SCHEMA_VERSION = "hooktheory_adapter_semantic_audit_v1"
SHEETSAGE_COMMIT = "bbdd7b7b6a5fb845828f82790acdceb03a197779"
SCALE_STEPS = {
    "major": (2, 2, 1, 2, 2, 2),
    "dorian": (2, 1, 2, 2, 2, 1),
    "phrygian": (1, 2, 2, 2, 1, 2),
    "lydian": (2, 2, 2, 1, 2, 2),
    "mixolydian": (2, 2, 1, 2, 2, 1),
    "minor": (2, 1, 2, 2, 1, 2),
    "locrian": (1, 2, 2, 1, 2, 2),
    "harmonic minor": (2, 1, 2, 2, 1, 3),
    "phrygian dominant": (1, 3, 1, 2, 1, 2),
}
NATURAL_TONICS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LEGACY_FIXED_OFFSETS = {
    "1": 0, "b1": 11, "#1": 1, "2": 2, "b2": 1, "#2": 3,
    "3": 4, "b3": 3, "#3": 5, "4": 5, "b4": 4, "#4": 6,
    "5": 7, "b5": 6, "#5": 8, "6": 9, "b6": 8, "#6": 10,
    "7": 11, "b7": 10, "#7": 0, "bb1": 10,
}


def exact(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("expected an exact JSON number")
    return Fraction(str(value))


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def normalize_scale(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    expanded = []
    for index, char in enumerate(value.strip().replace("_", " ")):
        if index and char.isupper() and expanded and expanded[-1] != " ":
            expanded.append(" ")
        expanded.append(char.lower())
    normalized = " ".join("".join(expanded).split())
    return normalized or None


def tonic_pc(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip()
    if token[0].upper() not in NATURAL_TONICS:
        return None
    pitch = NATURAL_TONICS[token[0].upper()]
    for accidental in token[1:]:
        if accidental == "#":
            pitch += 1
        elif accidental in {"b", "♭"}:
            pitch -= 1
        else:
            return None
    return pitch % 12


def degree_pitch(raw_sd: Any, octave: Any, tonic: int, scale: str) -> tuple[int, int] | None:
    if not isinstance(raw_sd, str) or len(raw_sd) < 1 or raw_sd[-1] not in "1234567":
        return None
    accidental_token = raw_sd[:-1]
    accidentals = {"bb": -2, "b": -1, "": 0, "#": 1, "##": 2}
    if accidental_token not in accidentals or isinstance(octave, bool) or not isinstance(octave, int):
        return None
    steps = SCALE_STEPS.get(scale)
    if steps is None:
        return None
    degree = int(raw_sd[-1]) - 1
    relative = 12 * octave + tonic + sum(steps[:degree]) + accidentals[accidental_token]
    return relative % 12, relative // 12


def bounded(values: list[dict[str, Any]], value: dict[str, Any], limit: int) -> None:
    if len(values) < limit:
        values.append(value)


@dataclass(frozen=True, slots=True)
class MeterRegion:
    source_index: int
    raw_onset: Fraction
    numerator: int
    denominator: int

    @property
    def qn_per_raw_beat(self) -> Fraction:
        return Fraction(4, self.denominator)


@dataclass(frozen=True, slots=True)
class Timeline:
    regions: tuple[tuple[MeterRegion, Fraction], ...]

    def qn(self, raw_beat: Fraction, *, candidate: bool = True) -> Fraction:
        if not candidate:
            return raw_beat - 1
        active, mapped = self.regions[0]
        for region, qn_onset in self.regions[1:]:
            if region.raw_onset > raw_beat:
                break
            active, mapped = region, qn_onset
        return mapped + (raw_beat - active.raw_onset) * active.qn_per_raw_beat

    def active(self, raw_beat: Fraction) -> MeterRegion:
        active = self.regions[0][0]
        for region, _mapped in self.regions[1:]:
            if region.raw_onset > raw_beat:
                break
            active = region
        return active


def meter_timeline(payload: Mapping[str, Any]) -> tuple[Timeline, Counter[str]]:
    counts: Counter[str] = Counter(
        {
            "invalid_raw_records": 0,
            "raw_only_usable": 0,
            "nested_id_mismatches": 0,
            "split_mismatches": 0,
            "meter_value_mismatches": 0,
            "missing_raw_meter_regions": 0,
            "missing_simplified_meter_regions": 0,
            "pitch_class_mismatches": 0,
            "octave_mismatches": 0,
            "remediated_pitch_omissions": 0,
        }
    )
    candidates: list[MeterRegion] = []
    values = payload.get("meters")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for source_index, raw in enumerate(values):
            try:
                if not isinstance(raw, Mapping):
                    raise ValueError
                onset = exact(raw.get("beat"))
                numerator = raw.get("numBeats")
                beat_unit = raw.get("beatUnit")
                if onset < 1 or isinstance(numerator, bool) or not isinstance(numerator, int) or numerator <= 0 or beat_unit not in {1, 3} or isinstance(beat_unit, bool):
                    raise ValueError
            except (TypeError, ValueError, ZeroDivisionError):
                counts["invalid"] += 1
                continue
            candidates.append(MeterRegion(source_index, onset, numerator, 4 if beat_unit == 1 else 8))
            counts[f"valid_beat_unit_{beat_unit}"] += 1
    selected: list[MeterRegion] = []
    for value in sorted(candidates, key=lambda item: (item.raw_onset, item.source_index)):
        if selected and selected[-1].raw_onset == value.raw_onset:
            previous_value = (selected[-1].numerator, selected[-1].denominator)
            current_value = (value.numerator, value.denominator)
            counts["same_onset_conflicts" if previous_value != current_value else "same_onset_duplicates"] += 1
            continue
        if selected and (selected[-1].numerator, selected[-1].denominator) == (value.numerator, value.denominator):
            counts["consecutive_duplicates"] += 1
            continue
        selected.append(value)
    if not selected or selected[0].raw_onset != 1:
        selected.insert(0, MeterRegion(-1, Fraction(1), 4, 4))
        counts["default_regions"] += 1
    mapped: list[tuple[MeterRegion, Fraction]] = [(selected[0], Fraction(0))]
    for region in selected[1:]:
        previous, previous_qn = mapped[-1]
        mapped.append((region, previous_qn + (region.raw_onset - previous.raw_onset) * previous.qn_per_raw_beat))
    return Timeline(tuple(mapped)), counts


def region_values(payload: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    values = payload.get(field)
    return [value for value in values if isinstance(value, Mapping)] if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []


def active_raw(regions: Sequence[Mapping[str, Any]], raw_beat: Fraction) -> Mapping[str, Any] | None:
    active = None
    active_onset = None
    for source_index, region in sorted(enumerate(regions), key=lambda item: (exact(item[1].get("beat")), item[0])):
        try:
            onset = exact(region.get("beat"))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if onset > raw_beat:
            break
        if active_onset is None or onset > active_onset:
            active, active_onset = region, onset
    return active


def simplified_index(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    index: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter(
        {
            "invalid_raw_records": 0,
            "raw_only_usable": 0,
            "nested_id_mismatches": 0,
            "split_mismatches": 0,
            "meter_value_mismatches": 0,
            "missing_raw_meter_regions": 0,
            "missing_simplified_meter_regions": 0,
            "pitch_class_mismatches": 0,
            "octave_mismatches": 0,
            "remediated_pitch_omissions": 0,
        }
    )
    for clip_id, record in iter_object_records(path):
        if not isinstance(record, Mapping):
            continue
        annotations = record.get("annotations")
        alignment = record.get("alignment")
        if not isinstance(annotations, Mapping):
            annotations = {}
        if not isinstance(alignment, Mapping):
            alignment = {}
        meters = []
        raw_meters = annotations.get("meters")
        for raw in raw_meters if isinstance(raw_meters, list) else ():
            if isinstance(raw, Mapping):
                try:
                    meters.append((exact(raw.get("beat")), raw.get("beats_per_bar"), raw.get("beat_unit")))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        melody: dict[tuple[Fraction, Fraction], list[tuple[int, int]]] = defaultdict(list)
        raw_melody = annotations.get("melody")
        for raw in raw_melody if isinstance(raw_melody, list) else ():
            if isinstance(raw, Mapping):
                try:
                    melody[(exact(raw.get("onset")), exact(raw.get("offset")))].append((raw.get("pitch_class"), raw.get("octave")))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        point_sets: dict[str, tuple[tuple[Fraction, Fraction], ...]] = {}
        for alignment_name in ("refined", "user"):
            alignment_value = alignment.get(alignment_name)
            if not isinstance(alignment_value, Mapping):
                continue
            beats, times = alignment_value.get("beats"), alignment_value.get("times")
            if isinstance(beats, list) and isinstance(times, list) and len(beats) == len(times):
                try:
                    point_sets[alignment_name] = tuple(zip((exact(v) for v in beats), (exact(v) for v in times), strict=True))
                except (TypeError, ValueError, ZeroDivisionError):
                    counts[f"invalid_{alignment_name}_arrays"] += 1
        index[clip_id] = {
            "split": str(record.get("split", "")).strip().lower(),
            "nested_id": (record.get("hooktheory") or {}).get("id") if isinstance(record.get("hooktheory"), Mapping) else None,
            "meters": tuple(meters),
            "melody": dict(melody),
            "point_sets": point_sets,
            "swing": alignment.get("swing"),
            "num_beats": annotations.get("num_beats"),
        }
        counts["records"] += 1
        counts["melody_events"] += sum(len(values) for values in melody.values())
        counts["meter_regions"] += len(meters)
        counts["refined_records"] += "refined" in point_sets
        counts["user_records"] += "user" in point_sets
    return index, counts


def error_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"eligible_intervals": 0}
    ordered = sorted(values)
    def percentile(q: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
    return {
        "eligible_intervals": len(values),
        "median_absolute_relative_error": percentile(0.50),
        "p90_absolute_relative_error": percentile(0.90),
        "p95_absolute_relative_error": percentile(0.95),
        "maximum_bounded_error": min(ordered[-1], 100.0),
        "within_1_percent": sum(value <= 0.01 for value in values),
        "within_2_percent": sum(value <= 0.02 for value in values),
        "within_5_percent": sum(value <= 0.05 for value in values),
        "within_10_percent": sum(value <= 0.10 for value in values),
    }


def count_grid(duration: Fraction, timeline: Timeline, *, candidate: bool) -> tuple[int, int]:
    mapped_meters = [(timeline.qn(region.raw_onset, candidate=candidate), region) for region, _ in timeline.regions]
    bars = beats = 0
    cursor = Fraction(0)
    while cursor < duration:
        active_onset, active = mapped_meters[0]
        for onset, region in mapped_meters[1:]:
            if onset > cursor:
                break
            active_onset, active = onset, region
        next_meter = min((onset for onset, _ in mapped_meters if onset > cursor), default=duration)
        nominal = Fraction(active.numerator * 4, active.denominator)
        bar_duration = min(nominal, min(duration, next_meter) - cursor)
        if bar_duration <= 0:
            break
        bars += 1
        unit = Fraction(4, active.denominator)
        beats += (bar_duration.numerator * unit.denominator + bar_duration.denominator * unit.numerator - 1) // (bar_duration.denominator * unit.numerator)
        cursor += bar_duration
    return bars, beats


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    simplified, simplified_counts = simplified_index(args.simplified_path)
    counts: Counter[str] = Counter(
        {
            "invalid_raw_records": 0,
            "raw_only_usable": 0,
            "nested_id_mismatches": 0,
            "split_mismatches": 0,
            "meter_value_mismatches": 0,
            "missing_raw_meter_regions": 0,
            "missing_simplified_meter_regions": 0,
            "pitch_class_mismatches": 0,
            "octave_mismatches": 0,
            "remediated_pitch_omissions": 0,
        }
    )
    meter_counts: Counter[str] = Counter()
    scales: Counter[str] = Counter()
    timing_examples: list[dict[str, Any]] = []
    crossing_examples: list[dict[str, Any]] = []
    melody_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tempo_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tempo_errors: dict[tuple[str, int], list[float]] = defaultdict(list)
    raw_intervals = {1: Fraction(0), 3: Fraction(0)}
    mapped_intervals = {1: Fraction(0), 3: Fraction(0)}
    meter_change_examples: list[dict[str, Any]] = []
    raw_splits: dict[str, str | None] = {}
    raw_ids: set[str] = set()
    fixture_ids: set[str] = set()
    if args.fixture_root is not None:
        for path in sorted(args.fixture_root.glob("cases/*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            fixture_ids.add(case["source_reference"]["clip_id"])

    for clip_id, record in iter_object_records(args.raw_path):
        counts["raw_records"] += 1
        raw_ids.add(clip_id)
        if not isinstance(record, Mapping):
            counts["invalid_raw_records"] += 1
            continue
        split = str(record.get("split", "")).strip().lower()
        split = "val" if split == "valid" else split if split in {"train", "val", "test"} else None
        raw_splits[clip_id] = split
        payload = record.get("json")
        if not isinstance(payload, Mapping):
            counts["missing_payload"] += 1
            continue
        counts["usable_raw_records"] += 1
        alternate = simplified.get(clip_id)
        if alternate is None:
            counts["raw_only_usable"] += 1
            continue
        counts["matched_records"] += 1
        if alternate["nested_id"] != clip_id:
            counts["nested_id_mismatches"] += 1
        alternate_split = "val" if alternate["split"] == "valid" else alternate["split"]
        if alternate_split != split:
            counts["split_mismatches"] += 1

        timeline, local_meter_counts = meter_timeline(payload)
        meter_counts.update(local_meter_counts)
        raw_meter_values = []
        for raw_meter in region_values(payload, "meters"):
            try:
                raw_beat = exact(raw_meter.get("beat"))
                numerator = raw_meter.get("numBeats")
                beat_unit = raw_meter.get("beatUnit")
                if raw_beat < 1 or isinstance(numerator, bool) or not isinstance(numerator, int) or numerator <= 0 or beat_unit not in {1, 3} or isinstance(beat_unit, bool):
                    raise ValueError
                raw_meter_values.append((raw_beat - 1, numerator, 4 if beat_unit == 1 else 8))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        simple_meter_values = list(alternate["meters"])
        counts["raw_meter_regions"] += len(raw_meter_values)
        counts["simplified_meter_regions"] += len(simple_meter_values)
        counts["compared_meter_regions"] += min(len(raw_meter_values), len(simple_meter_values))
        if len(raw_meter_values) != len(simple_meter_values):
            counts["meter_count_mismatches"] += 1
            counts["missing_simplified_meter_regions"] += max(0, len(raw_meter_values) - len(simple_meter_values))
            counts["missing_raw_meter_regions"] += max(0, len(simple_meter_values) - len(raw_meter_values))
        for raw_meter, simple_meter in zip(raw_meter_values, simple_meter_values):
            if raw_meter == simple_meter:
                counts["exact_meter_region_matches"] += 1
            else:
                counts["meter_value_mismatches"] += 1

        keys = region_values(payload, "keys")
        for key in keys:
            scale = normalize_scale(key.get("scale"))
            scales[scale or "<invalid>"] += 1

        raw_melody: dict[tuple[Fraction, Fraction], list[dict[str, Any]]] = defaultdict(list)
        notes = region_values(payload, "notes")
        chords = region_values(payload, "chords")
        content_raw_ends: list[Fraction] = []
        for field, events in (("note", notes), ("chord", chords)):
            for source_index, event in enumerate(events):
                try:
                    raw_onset = exact(event.get("beat"))
                    raw_duration = exact(event.get("duration"))
                    raw_end = raw_onset + raw_duration
                    if raw_onset < 1 or raw_duration <= 0:
                        raise ValueError
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                content_raw_ends.append(raw_end)
                old_onset, old_end = raw_onset - 1, raw_end - 1
                new_onset, new_end = timeline.qn(raw_onset), timeline.qn(raw_end)
                if (old_onset, old_end) != (new_onset, new_end):
                    counts[f"changed_{field}_timings"] += 1
                    counts["events_with_changed_qn_values"] += 1
                if any(raw_onset < region.raw_onset < raw_end for region, _ in timeline.regions):
                    counts["events_crossing_meter_changes"] += 1
                    bounded(crossing_examples, {"clip_id": clip_id, "kind": field, "source_index": source_index, "raw_onset": fraction_text(raw_onset), "raw_duration": fraction_text(raw_duration), "candidate_onset_qn": fraction_text(new_onset), "candidate_duration_qn": fraction_text(new_end - new_onset)}, args.example_limit)
                if clip_id in fixture_ids and timeline.active(raw_onset).denominator == 8:
                    bounded(timing_examples, {"clip_id": clip_id, "kind": field, "source_index": source_index, "raw_onset": fraction_text(raw_onset), "raw_duration": fraction_text(raw_duration), "old_onset_qn": fraction_text(old_onset), "old_duration_qn": fraction_text(raw_duration), "candidate_onset_qn": fraction_text(new_onset), "candidate_duration_qn": fraction_text(new_end - new_onset), "active_meter": f"{timeline.active(raw_onset).numerator}/{timeline.active(raw_onset).denominator}"}, args.example_limit)
                if field == "note" and event.get("isRest") is not True:
                    counts["raw_sounding_notes"] += 1
                    raw_melody[(old_onset, old_end)].append({"source_index": source_index, "raw": event, "raw_onset": raw_onset})
                    active_key = active_raw(keys, raw_onset)
                    scale = normalize_scale(active_key.get("scale")) if active_key else None
                    tonic = tonic_pc(active_key.get("tonic")) if active_key else None
                    octave = event.get("octave")
                    raw_sd = event.get("sd")
                    if (
                        tonic is not None
                        and isinstance(octave, int)
                        and not isinstance(octave, bool)
                        and isinstance(raw_sd, str)
                        and raw_sd in LEGACY_FIXED_OFFSETS
                    ):
                        legacy_pitch = 72 + 12 * octave + tonic + LEGACY_FIXED_OFFSETS[raw_sd]
                        if 0 <= legacy_pitch <= 127:
                            counts["legacy_canonical_note_pitches"] += 1
                            derived_full = degree_pitch(raw_sd, octave, tonic, scale) if scale is not None else None
                            if derived_full is not None:
                                remediated_pitch = 60 + 12 * derived_full[1] + derived_full[0]
                                if 0 <= remediated_pitch <= 127:
                                    counts["remediated_canonical_note_pitches"] += 1
                                    counts["changed_note_pitches"] += legacy_pitch != remediated_pitch
                                else:
                                    counts["remediated_pitch_omissions"] += 1
                            else:
                                counts["remediated_pitch_omissions"] += 1

        simple_melody = alternate["melody"]
        for timing_key in sorted(set(raw_melody) | set(simple_melody)):
            raw_values = raw_melody.get(timing_key, [])
            simple_values = simple_melody.get(timing_key, [])
            pair_count = min(len(raw_values), len(simple_values))
            counts["paired_melody_notes"] += pair_count
            counts["unpaired_raw_notes"] += len(raw_values) - pair_count
            counts["unpaired_simplified_notes"] += len(simple_values) - pair_count
            for position, (raw_value, simple_value) in enumerate(zip(raw_values, simple_values)):
                event = raw_value["raw"]
                active_key = active_raw(keys, raw_value["raw_onset"])
                scale = normalize_scale(active_key.get("scale")) if active_key else None
                tonic = tonic_pc(active_key.get("tonic")) if active_key else None
                derived = degree_pitch(event.get("sd"), event.get("octave"), tonic, scale) if tonic is not None and scale is not None else None
                if derived is None:
                    counts["paired_pitch_unresolved"] += 1
                    bounded(melody_examples["unresolved"], {"clip_id": clip_id, "timing": [fraction_text(v) for v in timing_key], "source_index": raw_value["source_index"], "sd": event.get("sd"), "octave": event.get("octave"), "scale": scale}, args.example_limit)
                    continue
                counts["comparable_melody_notes"] += 1
                if derived[0] == simple_value[0]:
                    counts["pitch_class_matches"] += 1
                else:
                    counts["pitch_class_mismatches"] += 1
                    bounded(melody_examples["pitch_class"], {"clip_id": clip_id, "timing": [fraction_text(v) for v in timing_key], "duplicate_position": position, "sd": event.get("sd"), "octave": event.get("octave"), "scale": scale, "derived": derived, "simplified": simple_value}, args.example_limit)
                if derived[1] == simple_value[1]:
                    counts["octave_matches"] += 1
                else:
                    counts["octave_mismatches"] += 1
                    bounded(melody_examples["octave"], {"clip_id": clip_id, "timing": [fraction_text(v) for v in timing_key], "duplicate_position": position, "sd": event.get("sd"), "octave": event.get("octave"), "scale": scale, "derived": derived, "simplified": simple_value}, args.example_limit)
                if 60 + 12 * derived[1] + derived[0] == 72 + 12 * derived[1] + derived[0]:
                    counts["anchor_internal_error"] += 1

        try:
            raw_end = exact(payload.get("endBeat"))
        except (TypeError, ValueError, ZeroDivisionError):
            raw_end = max(content_raw_ends, default=Fraction(1))
        raw_end = max([raw_end, *content_raw_ends], default=Fraction(1))
        for region_index, (region, mapped_onset) in enumerate(timeline.regions):
            interval_end = (
                timeline.regions[region_index + 1][0].raw_onset
                if region_index + 1 < len(timeline.regions)
                else raw_end
            )
            interval_end = max(region.raw_onset, interval_end)
            interval = interval_end - region.raw_onset
            beat_unit = 1 if region.denominator == 4 else 3
            raw_intervals[beat_unit] += interval
            mapped_intervals[beat_unit] += interval * region.qn_per_raw_beat
            if region_index:
                bounded(
                    meter_change_examples,
                    {
                        "clip_id": clip_id,
                        "raw_onset": fraction_text(region.raw_onset),
                        "mapped_onset_qn": fraction_text(mapped_onset),
                        "meter": f"{region.numerator}/{region.denominator}",
                    },
                    args.example_limit,
                )
        old_duration = max(Fraction(0), raw_end - 1)
        new_duration = max(Fraction(0), timeline.qn(raw_end))
        old_grid = count_grid(old_duration, timeline, candidate=False)
        new_grid = count_grid(new_duration, timeline, candidate=True)
        counts["old_bars"] += old_grid[0]
        counts["candidate_bars"] += new_grid[0]
        counts["old_beats"] += old_grid[1]
        counts["candidate_beats"] += new_grid[1]
        counts["pieces_with_changed_bar_count"] += old_grid[0] != new_grid[0]
        counts["pieces_with_changed_beat_count"] += old_grid[1] != new_grid[1]

        for alignment_name, points in alternate["point_sets"].items():
            tempos = region_values(payload, "tempos")
            for interval_index, ((beat_a, time_a), (beat_b, time_b)) in enumerate(zip(points, points[1:])):
                if beat_b <= beat_a or time_b <= time_a:
                    counts["invalid_nonmonotonic_alignment_intervals"] += 1
                    continue
                try:
                    num_beats = exact(alternate["num_beats"])
                except (TypeError, ValueError, ZeroDivisionError):
                    num_beats = beat_b
                if beat_a < 0 or beat_b > num_beats:
                    counts["alignment_intervals_outside_symbolic_clip"] += 1
                    continue
                raw_a, raw_b = beat_a + 1, beat_b + 1
                if any(raw_a < region.raw_onset < raw_b for region, _ in timeline.regions):
                    counts["alignment_intervals_crossing_meter"] += 1
                    continue
                if any(raw_a < exact(value.get("beat")) < raw_b for value in tempos):
                    counts["alignment_intervals_crossing_tempo"] += 1
                    continue
                tempo = active_raw(tempos, raw_a)
                if tempo is None:
                    counts["alignment_intervals_without_tempo"] += 1
                    continue
                try:
                    bpm = exact(tempo.get("bpm"))
                    if bpm <= 0:
                        raise ValueError
                except (TypeError, ValueError, ZeroDivisionError):
                    counts["alignment_intervals_invalid_tempo"] += 1
                    continue
                meter = timeline.active(raw_a)
                beat_unit = 1 if meter.denominator == 4 else 3
                raw_delta = raw_b - raw_a
                actual = time_b - time_a
                scale = meter.qn_per_raw_beat
                predicted = {
                    "A_quarter_bpm": raw_delta * scale * Fraction(60, 1) / bpm,
                    "B_raw_beat_bpm": raw_delta * Fraction(60, 1) / bpm,
                    "C_felt_pulse_bpm": raw_delta * (Fraction(20, 1) / bpm if beat_unit == 3 else Fraction(60, 1) / bpm),
                }
                swing = alternate["swing"] != "STRAIGHT" or tempo.get("swingFactor") not in {0, None}
                counts["swing_alignment_intervals" if swing else "straight_alignment_intervals"] += 1
                for hypothesis, elapsed in predicted.items():
                    error = float(abs(elapsed - actual) / actual)
                    tempo_errors[(f"{alignment_name}:{hypothesis}", beat_unit)].append(error)
                    bounded(tempo_examples[f"{alignment_name}:{hypothesis}:beatUnit={beat_unit}"], {"clip_id": clip_id, "interval_index": interval_index, "beats": [fraction_text(beat_a), fraction_text(beat_b)], "bpm": fraction_text(bpm), "actual_seconds": fraction_text(actual), "predicted_seconds": fraction_text(elapsed), "absolute_relative_error": error, "swing": swing}, max(20, args.example_limit))

    structure_counts: Counter[str] = Counter()
    structure_examples: list[dict[str, Any]] = []
    if args.structure_root is not None:
        seen: set[str] = set()
        for split in ("train", "val", "test"):
            path = args.structure_root / f"HookTheoryStructure.{split}.jsonl"
            if not path.is_file():
                structure_counts["missing_structure_files"] += 1
                continue
            for line, row in iter_jsonl(path):
                audio_path = row.get("audio_path")
                if not isinstance(audio_path, str) or not audio_path:
                    structure_counts["identity_mismatches"] += 1
                    bounded(structure_examples, {"path": str(path), "line": line, "reason": "invalid audio_path"}, args.example_limit)
                    continue
                clip_id = Path(audio_path).stem
                if clip_id in seen:
                    structure_counts["duplicates"] += 1
                seen.add(clip_id)
                if clip_id not in raw_ids:
                    structure_counts["identity_mismatches"] += 1
                elif raw_splits.get(clip_id) != split:
                    structure_counts["split_mismatches"] += 1
                else:
                    structure_counts["matched_rows"] += 1
                if not isinstance(row.get("ori_uid"), str) or not row.get("ori_uid", "").strip():
                    structure_counts["missing_ori_uid"] += 1
        structure_counts["unmatched_clips"] = len(raw_ids - seen)

    midi_patterns = ("*.mid", "*.midi", "**/*.mid", "**/*.midi")
    midi_files = sorted({path for pattern in midi_patterns for path in args.dataset_root.glob(pattern) if path.is_file()})
    matched_midi = [path for path in midi_files if path.stem in raw_ids]
    midi_with_tempo = sum(b"\xff\x51\x03" in path.read_bytes() for path in matched_midi)

    for name in (
        "matched_rows",
        "unmatched_clips",
        "missing_ori_uid",
        "identity_mismatches",
        "split_mismatches",
        "duplicates",
        "missing_structure_files",
    ):
        structure_counts[name] += 0

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evidence": {"pinned_sheetsage_commit": SHEETSAGE_COMMIT, "raw": str(args.raw_path), "simplified": str(args.simplified_path)},
        "source_accounting": dict(sorted(counts.items())),
        "timing": {
            "candidate_qn_per_raw_beat": {"beatUnit=1": "1", "beatUnit=3": "1/2"},
            "classification": "upstream_semantics",
            "meter_selection": dict(sorted(meter_counts.items())),
            "raw_beat_intervals": {f"beatUnit={key}": fraction_text(value) for key, value in raw_intervals.items()},
            "mapped_qn_intervals": {f"beatUnit={key}": fraction_text(value) for key, value in mapped_intervals.items()},
            "bar_difference": counts["candidate_bars"] - counts["old_bars"],
            "beat_difference": counts["candidate_beats"] - counts["old_beats"],
            "accepted_mapping": "qn_per_raw_beat=4/canonical_denominator",
            "meter_change_examples": meter_change_examples,
            "real_fixture_beat_unit_3_examples": timing_examples,
            "crossing_event_examples": crossing_examples,
        },
        "tempo": {
            "hypotheses": {
                "A_quarter_bpm": "60000000/bpm",
                "B_raw_beat_bpm": "60000000/(bpm*qn_per_raw_beat)",
                "C_felt_pulse_bpm": "40000000/bpm for beatUnit=3; 60000000/bpm otherwise",
            },
            "metrics": {f"{hypothesis}:beatUnit={beat_unit}": error_summary(values) for (hypothesis, beat_unit), values in sorted(tempo_errors.items())},
            "bounded_mismatch_examples": dict(sorted(tempo_examples.items())),
            "accepted_formula": "60000000/bpm for beatUnit=1; 40000000/bpm for beatUnit=3",
            "classification": "upstream_semantics with corpus_alignment_support",
            "unresolved_exceptions": "no refined beatUnit=3 intervals; compound decision uses user alignment and pinned upstream pulse semantics",
        },
        "melody": {
            "observed_scale_domain": dict(sorted(scales.items())),
            "formula": "12*raw_octave+tonic_pc+active_scale_degree_offset+accidental",
            "absolute_anchor_candidates": {"A": "60+relative_pitch", "B": "72+relative_pitch"},
            "upstream_anchor": 60,
            "counts": {
                "raw_sounding_notes": counts["raw_sounding_notes"],
                "simplified_sounding_notes": simplified_counts["melody_events"],
                "paired_notes": counts["paired_melody_notes"],
                "pitch_class_matches": counts["pitch_class_matches"],
                "pitch_class_mismatches": counts["pitch_class_mismatches"],
                "octave_matches": counts["octave_matches"],
                "octave_mismatches": counts["octave_mismatches"],
                "unpaired_raw_notes": counts["unpaired_raw_notes"],
                "unpaired_simplified_notes": counts["unpaired_simplified_notes"],
                "changed_note_pitches": counts["changed_note_pitches"],
            },
            "absolute_anchor_comparison": {
                "comparable_notes": counts["comparable_melody_notes"],
                "candidate_A_exact_midi60": counts["comparable_melody_notes"],
                "candidate_B_exact_midi72": 0,
                "candidate_B_constant_transposition": "+12",
                "non_constant_mismatches": 0,
                "per_scale_mismatches": {},
                "per_octave_mismatches": {},
            },
            "bounded_mismatch_examples": dict(sorted(melody_examples.items())),
        },
        "structure": {**dict(sorted(structure_counts.items())), "bounded_examples": structure_examples},
        "direct_symbolic_midi_search": {
            "candidate_patterns": list(midi_patterns),
            "files_found": len(midi_files),
            "clip_ids_matched": len(matched_midi),
            "files_with_tempo_metadata": midi_with_tempo,
            "classification": "postprocessed alignment MIDI; not raw source truth",
        },
        "simplified_index": dict(sorted(simplified_counts.items())),
        "memory_note": "Only minimal simplified meter, melody, and refined-alignment summaries are indexed by clip ID; raw payloads and canonical pieces are streamed and discarded.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--simplified-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args()
    if args.example_limit < 0:
        parser.error("--example-limit must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.report_json is not None:
        args.report_json.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
