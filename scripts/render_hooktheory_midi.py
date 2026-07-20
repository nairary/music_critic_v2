#!/usr/bin/env python3
"""Render selected HookTheory clips to canonical JSON and listenable MIDI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.adapters._json_stream import iter_jsonl, iter_object_records
from music_critic.data import CanonicalPiece, dump_piece
from music_critic.exporters import MidiRenderConfig, MidiRenderReport, write_piece_midi
try:
    from scripts.audit_hooktheory_midi_ambiguities import analyze_piece_ambiguities
    from scripts.compare_hooktheory_midi_rendering import (
        build_report as build_comparison_report,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from audit_hooktheory_midi_ambiguities import analyze_piece_ambiguities
    from compare_hooktheory_midi_rendering import build_report as build_comparison_report


REPORT_SCHEMA_VERSION = "hooktheory_midi_render_manifest_v1"


def _report_dict(report: MidiRenderReport) -> dict[str, Any]:
    value = asdict(report)
    error = report.maximum_quantization_error_qn
    value["maximum_quantization_error_qn"] = {
        "num": error.num,
        "den": error.den,
    }
    return value


def _manifest_clips(path: Path) -> list[tuple[str, str, tuple[str, ...]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases") if isinstance(document, dict) else document
    if not isinstance(cases, list):
        raise ValueError("manifest must be a list or an object with a cases list")
    selected: list[tuple[str, str, tuple[str, ...]]] = []
    for item in cases:
        if isinstance(item, str):
            case_path = path.parent / "cases" / f"{item}.json"
            if case_path.is_file():
                case = json.loads(case_path.read_text(encoding="utf-8"))
                source = case.get("source_reference", {})
                clip_id = source.get("clip_id") if isinstance(source, dict) else None
                tags = case.get("coverage_tags", ())
                if isinstance(clip_id, str):
                    selected.append(
                        (
                            clip_id,
                            item,
                            tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
                        )
                    )
                    continue
            selected.append((item, item, ()))
        elif isinstance(item, dict) and isinstance(item.get("clip_id"), str):
            tags = item.get("coverage_tags", ())
            selected.append(
                (
                    item["clip_id"],
                    str(item.get("case_id", item["clip_id"])),
                    tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
                )
            )
        else:
            raise ValueError("each manifest case must be a clip ID or case object")
    return selected


def _selection(args: argparse.Namespace) -> list[tuple[str, str, tuple[str, ...]]]:
    selected = [(clip_id, clip_id, ()) for clip_id in args.clip_id]
    if args.manifest is not None:
        selected.extend(_manifest_clips(args.manifest))
    deduplicated: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for item in selected:
        deduplicated.setdefault(item[0], item)
    ordered = sorted(deduplicated.values(), key=lambda item: item[0])
    if args.sample_limit is not None:
        ordered = ordered[: args.sample_limit]
    if not ordered and not args.deterministic_sample:
        raise ValueError("provide at least one --clip-id or --manifest")
    return ordered


def _normalized_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    expanded = []
    for index, character in enumerate(value.strip().replace("_", " ")):
        if index and character.isupper() and expanded and expanded[-1] != " ":
            expanded.append(" ")
        expanded.append(character.lower())
    return " ".join("".join(expanded).split()) or None


def _has_fractional_timing(payload: dict[str, Any]) -> bool:
    for collection, fields in (
        ("notes", ("beat", "duration")),
        ("chords", ("beat", "duration")),
        ("keys", ("beat",)),
        ("meters", ("beat",)),
        ("tempos", ("beat",)),
    ):
        values = payload.get(collection)
        for value in values if isinstance(values, list) else ():
            if not isinstance(value, dict):
                continue
            for field in fields:
                number = value.get(field)
                if isinstance(number, bool) or not isinstance(
                    number, (int, float, Decimal)
                ):
                    continue
                if Fraction(str(number)).denominator != 1:
                    return True
    return False


def _shared_structure_clip_ids(root: Path | None) -> set[str]:
    if root is None:
        return set()
    groups: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = root / f"HookTheoryStructure.{split}.jsonl"
        if not path.is_file():
            continue
        for _line, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            group_id = row.get("ori_uid")
            if isinstance(audio_path, str) and isinstance(group_id, str):
                groups.setdefault(group_id, []).append(Path(audio_path).stem)
    duplicate_groups = sorted(
        tuple(sorted(set(clip_ids)))
        for clip_ids in groups.values()
        if len(set(clip_ids)) > 1
    )
    return set(duplicate_groups[0][:2]) if duplicate_groups else set()


def _deterministic_sample_records(
    path: Path, structure_root: Path | None
) -> tuple[
    list[tuple[str, str, tuple[str, ...]]], dict[str, dict[str, Any]]
]:
    shared_ids = _shared_structure_clip_ids(structure_root)
    category_winners: dict[str, str] = {}
    seen_shared_ids: set[str] = set()
    for clip_id, record in iter_object_records(path):
        if not isinstance(record, dict) or not isinstance(record.get("json"), dict):
            continue
        payload = record["json"]
        categories: set[str] = set()
        keys = payload.get("keys")
        for key in keys if isinstance(keys, list) else ():
            if isinstance(key, dict):
                mode = _normalized_mode(key.get("scale"))
                if mode is not None:
                    categories.add(f"mode:{mode}")
                    if mode in {"major", "minor"}:
                        categories.add(mode)
        meters = payload.get("meters")
        meter_values = meters if isinstance(meters, list) else []
        for meter in meter_values:
            if not isinstance(meter, dict):
                continue
            numerator = meter.get("numBeats")
            denominator = 8 if meter.get("beatUnit") == 3 else 4
            if numerator in {6, 9, 12} and denominator == 8:
                categories.add(f"meter:{numerator}/8")
        if len(meter_values) > 1:
            categories.add("multiple_meters")
        tempos = payload.get("tempos")
        if isinstance(tempos, list) and len(tempos) > 1:
            categories.add("multiple_tempos")
        if _has_fractional_timing(payload):
            categories.add("fractional_timing")
        if clip_id in shared_ids:
            categories.add("shared_ori_uid")
            seen_shared_ids.add(clip_id)
        for category in categories:
            previous = category_winners.get(category)
            if previous is None or clip_id < previous:
                category_winners[category] = clip_id
    by_clip: dict[str, set[str]] = {}
    for category, clip_id in category_winners.items():
        by_clip.setdefault(clip_id, set()).add(category)
    for clip_id in sorted(seen_shared_ids):
        by_clip.setdefault(clip_id, set()).add("shared_ori_uid")
    selected = [
        (
            clip_id,
            "deterministic_sample",
            tuple(sorted(categories)),
        )
        for clip_id, categories in sorted(by_clip.items())
    ]
    return selected, _selected_raw_records(path, set(by_clip))


def _selected_raw_records(
    path: Path, clip_ids: set[str]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for clip_id, record in iter_object_records(path):
        if clip_id in clip_ids and isinstance(record, dict):
            records[clip_id] = record
            if len(records) == len(clip_ids):
                break
    return records


def _selected_structure_rows(
    root: Path | None, clip_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = root / f"HookTheoryStructure.{split}.jsonl"
        if not path.is_file():
            raise OSError(f"structure source is missing: {path}")
        for _line_number, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            if isinstance(audio_path, str):
                clip_id = Path(audio_path).stem
                if clip_id in clip_ids:
                    if clip_id in rows:
                        raise ValueError(f"duplicate structure rows for {clip_id!r}")
                    rows[clip_id] = dict(row)
    return rows


def _active_value(piece: CanonicalPiece, task: str) -> str | None:
    for target in piece.targets:
        if target.task == task:
            for value, available in zip(target.values, target.mask, strict=True):
                if available and value is not None:
                    return str(value)
    return None


def _piece_summary(piece: CanonicalPiece) -> dict[str, Any]:
    available_targets = sum(sum(target.mask) for target in piece.targets)
    total_targets = sum(len(target.mask) for target in piece.targets)
    return {
        "piece_id": piece.piece_id,
        "source_path": piece.source_path,
        "duration_qn": {
            "num": piece.duration_qn.num,
            "den": piece.duration_qn.den,
        },
        "note_count": len(piece.notes),
        "tempo_event_count": len(piece.tempo_events),
        "meter_event_count": len(piece.meter_events),
        "bar_count": len(piece.bars),
        "beat_count": len(piece.beats),
        "annotation_count": len(piece.annotations),
        "target_array_count": len(piece.targets),
        "target_values_available": available_targets,
        "target_values_unavailable": total_targets - available_targets,
        "quality_flags": [
            {
                "code": flag.code,
                "severity": flag.severity,
                "message": flag.message,
            }
            for flag in piece.quality_flags
        ],
    }


def _time_text(value: Any) -> str:
    fraction = value.to_fraction()
    return str(fraction.numerator) if fraction.denominator == 1 else f"{fraction.numerator}/{fraction.denominator}"


def _listening_entry(
    piece: CanonicalPiece,
    *,
    clip_id: str,
    case_id: str,
    coverage_tags: tuple[str, ...],
    midi_name: str,
    json_name: str,
    report_name: str,
    render_report: MidiRenderReport,
    ambiguities: dict[str, Any],
) -> dict[str, Any]:
    focus = list(coverage_tags) or ["melody", "tempo", "meter"]
    if any(event.denominator == 8 for event in piece.meter_events):
        focus.extend(
            (
                "verify 6/8, 9/8 or 12/8 playback duration",
                "verify denominator-unit click",
            )
        )
    if ambiguities["same_pitch_overlap_pairs"]:
        focus.append("same-pitch overlap: MIDI note-off pairing is diagnostic only")
    if ambiguities["channel_program_conflict_pairs"]:
        focus.append("program conflict: verify channel timbre manually")
    return {
        "clip_id": clip_id,
        "case_id": case_id,
        "coverage_tags": list(coverage_tags),
        "mode": _active_value(piece, "theory.local_key.mode"),
        "meter_regions": [
            {
                "onset_qn": _time_text(event.onset_qn),
                "numerator": event.numerator,
                "denominator": event.denominator,
            }
            for event in piece.meter_events
        ],
        "tempo_events": [
            {
                "onset_qn": _time_text(event.onset_qn),
                "microseconds_per_quarter": event.microseconds_per_quarter,
            }
            for event in piece.tempo_events
        ],
        "duration_qn": _time_text(piece.duration_qn),
        "note_count": len(piece.notes),
        "ticks_per_quarter": render_report.ticks_per_quarter,
        "exact_timing": render_report.exact_timing,
        "timing_quantized": render_report.timing_quantized,
        "maximum_quantization_error_qn": _time_text(render_report.maximum_quantization_error_qn),
        "same_pitch_overlap_pairs": ambiguities["same_pitch_overlap_pairs"],
        "channel_program_conflict_pairs": ambiguities["channel_program_conflict_pairs"],
        "audio_alignment_status": "pending_comparison",
        "audio_onset_p95_seconds": None,
        "midi_path": midi_name,
        "canonical_json_path": json_name,
        "render_report_path": report_name,
        "diagnostic_focus": focus,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/HookTheory/Hooktheory_Raw.json/4_merged.json"),
    )
    parser.add_argument("--structure-root", type=Path)
    parser.add_argument(
        "--simplified-path",
        type=Path,
        default=Path("data/HookTheory/Hooktheory.json"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--deterministic-sample",
        action="store_true",
        help=(
            "select stable representatives for every observed mode, 6/8, 9/8, "
            "12/8, meter/tempo changes, fractional timing, and shared ori_uid"
        ),
    )
    parser.add_argument("--dataset-name", default="HookTheory")
    parser.add_argument("--ticks-per-quarter", type=int)
    parser.add_argument(
        "--allow-quantization",
        "--allow-timing-quantization",
        dest="allow_quantization",
        action="store_true",
    )
    parser.add_argument("--no-click", action="store_true")
    parser.add_argument(
        "--no-markers",
        "--no-target-markers",
        dest="no_target_markers",
        action="store_true",
    )
    parser.add_argument("--hide-targets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        selected = _selection(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.sample_limit is not None and args.sample_limit < 1:
        parser.error("--sample-limit must be positive")
    if args.raw_path.name != "4_merged.json":
        parser.error("--raw-path must refer to the production 4_merged.json source")
    try:
        if args.deterministic_sample:
            if selected:
                parser.error(
                    "--deterministic-sample cannot be combined with --clip-id or --manifest"
                )
            selected, raw_records = _deterministic_sample_records(
                args.raw_path, args.structure_root
            )
            if args.sample_limit is not None:
                selected = selected[: args.sample_limit]
                raw_records = {
                    clip_id: raw_records[clip_id]
                    for clip_id, _case_id, _tags in selected
                }
        else:
            clip_ids = {item[0] for item in selected}
            raw_records = _selected_raw_records(args.raw_path, clip_ids)
        clip_ids = {item[0] for item in selected}
        structure_rows = _selected_structure_rows(args.structure_root, clip_ids)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_config = MidiRenderConfig(
        ticks_per_quarter=args.ticks_per_quarter,
        require_exact_timing=not args.allow_quantization,
        include_click_track=not args.no_click,
        include_target_markers=not args.no_target_markers and not args.hide_targets,
    )
    adapter_config = HookTheoryAdapterConfig(
        dataset_name=args.dataset_name,
        include_targets=not args.hide_targets,
    )
    results: list[dict[str, Any]] = []
    listening: list[dict[str, Any]] = []
    ambiguity_items: list[dict[str, Any]] = []
    failed = 0
    for clip_id, case_id, tags in selected:
        if args.hide_targets and "target_hiding" not in tags:
            tags = tags + ("target_hiding",)
        try:
            record = raw_records.get(clip_id)
            if record is None:
                raise HookTheoryAdapterError(
                    "requested HookTheory record is missing", clip_id=clip_id
                )
            piece = convert_hooktheory_record(
                clip_id,
                record,
                config=adapter_config,
                structure_row=structure_rows.get(clip_id),
                source_path=str(args.raw_path),
            )
            midi_name = f"{clip_id}.canonical.mid"
            json_name = f"{clip_id}.canonical.json"
            report_name = f"{clip_id}.render-report.json"
            dump_piece(piece, args.output_dir / json_name)
            report = write_piece_midi(
                piece, args.output_dir / midi_name, config=render_config
            )
            ambiguities = analyze_piece_ambiguities(piece, example_limit=10)
            ambiguity_items.append(ambiguities)
            report_document = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "clip_id": clip_id,
                "canonical_json_path": json_name,
                "midi_path": midi_name,
                "piece": _piece_summary(piece),
                "render": _report_dict(report),
                "midi_ambiguities": ambiguities,
            }
            (args.output_dir / report_name).write_text(
                json.dumps(report_document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "case_id": case_id,
                    "status": "rendered",
                    "report_path": report_name,
                    **_piece_summary(piece),
                    **_report_dict(report),
                }
            )
            listening.append(
                _listening_entry(
                    piece,
                    clip_id=clip_id,
                    case_id=case_id,
                    coverage_tags=tags,
                    midi_name=midi_name,
                    json_name=json_name,
                    report_name=report_name,
                    render_report=report,
                    ambiguities=ambiguities,
                )
            )
        except HookTheoryAdapterError as exc:
            status = "skipped_missing_payload" if "payload" in str(exc).lower() else "failed"
            failed += status == "failed"
            results.append(
                {
                    "clip_id": clip_id,
                    "case_id": case_id,
                    "status": status,
                    "error": str(exc),
                }
            )
        except (OSError, ValueError) as exc:
            failed += 1
            results.append(
                {
                    "clip_id": clip_id,
                    "case_id": case_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "raw_path": str(args.raw_path),
        "selected_clips": len(selected),
        "rendered_clips": sum(item["status"] == "rendered" for item in results),
        "skipped_missing_payload": sum(
            item["status"] == "skipped_missing_payload" for item in results
        ),
        "failed_clips": failed,
        "results": results,
    }
    (args.output_dir / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ambiguity_report = {
        "schema_version": "hooktheory_midi_ambiguity_audit_v1",
        "scope": "rendered review-package clips",
        "total_clips": len(ambiguity_items),
        "total_notes": sum(item["note_count"] for item in ambiguity_items),
        "clips_with_same_pitch_overlaps": sum(item["same_pitch_overlap_pairs"] > 0 for item in ambiguity_items),
        "same_pitch_overlap_pairs": sum(item["same_pitch_overlap_pairs"] for item in ambiguity_items),
        "same_pitch_nested_pairs": sum(item["same_pitch_nested_pairs"] for item in ambiguity_items),
        "clips_with_channel_program_conflicts": sum(item["channel_program_conflict_pairs"] > 0 for item in ambiguity_items),
        "channel_program_conflict_pairs": sum(item["channel_program_conflict_pairs"] for item in ambiguity_items),
        "same_pitch_overlap_examples": [
            example
            for item in ambiguity_items
            for example in item["same_pitch_overlap_examples"]
        ][:10],
        "channel_program_conflict_examples": [
            example
            for item in ambiguity_items
            for example in item["channel_program_conflict_examples"]
        ][:10],
    }
    (args.output_dir / "ambiguity-report.json").write_text(
        json.dumps(ambiguity_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_failed = False
    if manifest["rendered_clips"]:
        comparison = build_comparison_report(
            argparse.Namespace(
                simplified_path=args.simplified_path,
                render_dir=args.output_dir,
                render_manifest=args.output_dir / "render-manifest.json",
                output=args.output_dir / "comparison-report.json",
                audio_disagreement_output=args.output_dir / "audio-disagreement-clips.json",
                example_limit=10,
            )
        )
        (args.output_dir / "comparison-report.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "audio-disagreement-clips.json").write_text(
            json.dumps(
                {
                    "schema_version": "hooktheory_audio_disagreements_v1",
                    "audio_disagreement_clips": comparison["audio_disagreement_clips"],
                    "clips": comparison["audio_disagreement_details"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        by_clip = {item["clip_id"]: item for item in comparison["comparisons"]}
        disagreement_focus = [
            "compare playback speed with source",
            "inspect alignment drift",
            "verify tempo interpretation",
        ]
        for entry in listening:
            item = by_clip.get(entry["clip_id"])
            if item is None:
                continue
            entry["audio_alignment_status"] = item["audio_alignment_status"]
            entry["audio_onset_p95_seconds"] = item["audio_onset_absolute_error_seconds"]["p95"]
            if item["audio_alignment_status"] == "disagreeing":
                entry["diagnostic_focus"].extend(disagreement_focus)
        comparison_failed = bool(
            comparison["missing_simplified_clips"]
            or comparison["symbolic_mismatch_clips"]
            or comparison["meter_mismatch_clips"]
            or comparison["audit_violation_clips"]
        )
    (args.output_dir / "listening-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "hooktheory_listening_manifest_v1",
                "entries": listening,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if failed or comparison_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
