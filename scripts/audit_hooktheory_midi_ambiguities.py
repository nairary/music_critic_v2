#!/usr/bin/env python3
"""Audit canonical HookTheory notes for MIDI representation ambiguities."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Iterable

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.adapters._json_stream import iter_object_records
from music_critic.data import CanonicalPiece


REPORT_SCHEMA_VERSION = "hooktheory_midi_ambiguity_audit_v1"


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _effective_note_values(piece: CanonicalPiece) -> Iterable[dict[str, Any]]:
    tracks = {track.track_id: track for track in piece.tracks}
    for note in piece.notes:
        track = tracks[note.track_id]
        channel = note.channel if note.channel is not None else track.channel
        if channel is None:
            channel = 9 if note.is_percussion else 0
        program = note.program if note.program is not None else track.program
        if program is None:
            program = 0
        yield {
            "note_id": note.note_id,
            "track_id": note.track_id,
            "channel": channel,
            "program": program,
            "pitch": note.pitch,
            "onset": note.onset_qn.to_fraction(),
            "offset": (note.onset_qn + note.duration_qn).to_fraction(),
        }


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["onset"] < right["offset"] and right["onset"] < left["offset"]


def analyze_piece_ambiguities(
    piece: CanonicalPiece, *, example_limit: int = 10
) -> dict[str, Any]:
    """Return note-pair ambiguities without changing or rejecting the piece."""

    notes = tuple(_effective_note_values(piece))
    pitch_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        pitch_groups[(note["track_id"], note["channel"], note["pitch"])].append(note)

    overlap_pairs = 0
    nested_pairs = 0
    overlap_examples: list[dict[str, Any]] = []
    for (track_id, channel, pitch), values in sorted(pitch_groups.items()):
        ordered = sorted(values, key=lambda item: (item["onset"], item["offset"], item["note_id"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right["onset"] >= left["offset"]:
                    break
                if not _overlap(left, right):
                    continue
                overlap_pairs += 1
                nested = right["offset"] <= left["offset"]
                nested_pairs += nested
                if len(overlap_examples) < example_limit:
                    overlap_examples.append(
                        {
                            "clip_id": piece.piece_id,
                            "track_id": track_id,
                            "channel": channel,
                            "pitch": pitch,
                            "left_note_id": left["note_id"],
                            "right_note_id": right["note_id"],
                            "left_onset_qn": _fraction_text(left["onset"]),
                            "left_offset_qn": _fraction_text(left["offset"]),
                            "right_onset_qn": _fraction_text(right["onset"]),
                            "right_offset_qn": _fraction_text(right["offset"]),
                            "overlap_type": "nested" if nested else "partial",
                        }
                    )

    channel_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        channel_groups[note["channel"]].append(note)
    conflict_pairs = 0
    conflict_examples: list[dict[str, Any]] = []
    for channel, values in sorted(channel_groups.items()):
        ordered = sorted(values, key=lambda item: (item["onset"], item["offset"], item["note_id"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right["onset"] >= left["offset"]:
                    break
                if left["program"] == right["program"] or not _overlap(left, right):
                    continue
                conflict_pairs += 1
                if len(conflict_examples) < example_limit:
                    overlap_start = max(left["onset"], right["onset"])
                    overlap_end = min(left["offset"], right["offset"])
                    conflict_examples.append(
                        {
                            "clip_id": piece.piece_id,
                            "channel": channel,
                            "conflicting_tracks": sorted({left["track_id"], right["track_id"]}),
                            "programs": [left["program"], right["program"]],
                            "left_note_id": left["note_id"],
                            "right_note_id": right["note_id"],
                            "overlap_onset_qn": _fraction_text(overlap_start),
                            "overlap_offset_qn": _fraction_text(overlap_end),
                        }
                    )

    warnings = []
    if overlap_pairs:
        warnings.append(
            "same-pitch overlapping notes on one canonical track/channel are not guaranteed to round-trip through MIDI note-off pairing"
        )
    if conflict_pairs:
        warnings.append(
            "simultaneous notes use different programs on one MIDI channel; rendered timbre is not guaranteed"
        )
    return {
        "clip_id": piece.piece_id,
        "note_count": len(notes),
        "same_pitch_overlap_pairs": overlap_pairs,
        "same_pitch_nested_pairs": nested_pairs,
        "channel_program_conflict_pairs": conflict_pairs,
        "same_pitch_overlap_examples": overlap_examples,
        "channel_program_conflict_examples": conflict_examples,
        "warnings": warnings,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    total_records = usable_clips = missing_payload_clips = failed_clips = total_notes = 0
    overlap_clips = conflict_clips = overlap_pairs = nested_pairs = conflict_pairs = 0
    overlap_examples: list[dict[str, Any]] = []
    conflict_examples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    config = HookTheoryAdapterConfig(dataset_name=args.dataset_name, include_targets=False)
    for clip_id, record in iter_object_records(args.raw_path):
        total_records += 1
        if not isinstance(record, dict) or not isinstance(record.get("json"), dict):
            missing_payload_clips += 1
            continue
        try:
            piece = convert_hooktheory_record(
                clip_id,
                record,
                config=config,
                source_path=str(args.raw_path),
            )
        except HookTheoryAdapterError as exc:
            failed_clips += 1
            if len(failures) < args.example_limit:
                failures.append({"clip_id": clip_id, "error": str(exc)})
            continue
        usable_clips += 1
        item = analyze_piece_ambiguities(piece, example_limit=args.example_limit)
        total_notes += item["note_count"]
        pair_count = item["same_pitch_overlap_pairs"]
        program_count = item["channel_program_conflict_pairs"]
        overlap_clips += pair_count > 0
        conflict_clips += program_count > 0
        overlap_pairs += pair_count
        nested_pairs += item["same_pitch_nested_pairs"]
        conflict_pairs += program_count
        overlap_examples.extend(
            {**example, "clip_id": clip_id}
            for example in item["same_pitch_overlap_examples"][
                : max(0, args.example_limit - len(overlap_examples))
            ]
        )
        conflict_examples.extend(
            {**example, "clip_id": clip_id}
            for example in item["channel_program_conflict_examples"][
                : max(0, args.example_limit - len(conflict_examples))
            ]
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "raw_path": str(args.raw_path),
        "total_clips": total_records,
        "usable_clips": usable_clips,
        "missing_payload_clips": missing_payload_clips,
        "failed_clips": failed_clips,
        "total_notes": total_notes,
        "clips_with_same_pitch_overlaps": overlap_clips,
        "same_pitch_overlap_pairs": overlap_pairs,
        "same_pitch_nested_pairs": nested_pairs,
        "clips_with_channel_program_conflicts": conflict_clips,
        "channel_program_conflict_pairs": conflict_pairs,
        "same_pitch_overlap_examples": overlap_examples,
        "channel_program_conflict_examples": conflict_examples,
        "failure_examples": failures,
        "policy": {
            "overlap_scope": "same canonical track, effective channel, and pitch with strict interval overlap",
            "program_conflict_scope": "different effective programs on one channel during a strict interval overlap",
            "channel_9": "reserved for percussion/click",
            "render_action": "report only; do not reject, move, or rewrite notes/channels/programs",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/HookTheory/Hooktheory_Raw.json/4_merged.json"),
    )
    parser.add_argument("--dataset-name", default="HookTheory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.example_limit < 0:
        raise SystemExit("--example-limit must be non-negative")
    if args.raw_path.name != "4_merged.json":
        raise SystemExit("--raw-path must refer to the production 4_merged.json source")
    try:
        report = build_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if report["failed_clips"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
