#!/usr/bin/env python3
"""Read-only full-corpus smoke runner for the production HookTheory adapter."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from music_critic.adapters._json_stream import JSONStreamError, iter_jsonl, iter_object_records
from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.data import CanonicalPiece, dumps_piece, loads_piece, validate_piece


def _structure_index(root: Path | None) -> dict[str, Mapping[str, Any]]:
    if root is None:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = root / f"HookTheoryStructure.{split}.jsonl"
        for line_number, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            if not isinstance(audio_path, str) or not audio_path:
                continue
            clip_id = Path(audio_path).stem
            if clip_id in rows:
                raise JSONStreamError(
                    f"{path}:{line_number}: duplicate structure clip ID {clip_id!r}"
                )
            rows[clip_id] = row
    return rows


def _bounded_example(values: list[dict[str, Any]], value: dict[str, Any], limit: int) -> None:
    if len(values) < limit:
        values.append(value)


def _sample_add(
    sample: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any] | None]],
    clip_id: str,
    record: Mapping[str, Any],
    structure_row: Mapping[str, Any] | None,
    limit: int,
) -> None:
    if limit <= 0:
        return
    score = sha256(clip_id.encode("utf-8")).hexdigest()
    sample.append((score, clip_id, record, structure_row))
    sample.sort(key=lambda value: (value[0], value[1]))
    if len(sample) > limit:
        sample.pop()


def _raw_content_equal(visible: CanonicalPiece, hidden: CanonicalPiece) -> bool:
    excluded = {"annotations", "targets", "provenance"}
    return all(
        getattr(visible, field.name) == getattr(hidden, field.name)
        for field in fields(CanonicalPiece)
        if field.name not in excluded
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw_path.resolve()
    structure_rows = _structure_index(
        args.structure_root.resolve() if args.structure_root is not None else None
    )
    config = HookTheoryAdapterConfig(
        dataset_name=args.dataset_name,
        include_targets=args.include_targets,
    )
    counts: Counter[str] = Counter()
    entity_totals: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    available_masks: Counter[str] = Counter()
    unavailable_masks: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    sample: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any] | None]] = []

    for clip_id, record in iter_object_records(raw_path):
        counts["raw_records"] += 1
        if not isinstance(record, Mapping):
            counts["unexpected_conversion_failures"] += 1
            _bounded_example(
                failure_examples,
                {"clip_id": clip_id, "classification": "record_type", "message": "record is not an object"},
                args.sample_limit,
            )
            continue
        if not isinstance(record.get("json"), Mapping):
            counts["missing_payload"] += 1
            continue
        counts["usable_records"] += 1
        counts["attempted"] += 1
        structure_row = structure_rows.get(clip_id)
        try:
            piece = convert_hooktheory_record(
                clip_id,
                record,
                config=config,
                structure_row=structure_row,
                source_path=str(raw_path),
            )
            report = validate_piece(piece)
            if report.errors:
                raise HookTheoryAdapterError(
                    "post-conversion validation failed",
                    clip_id=clip_id,
                    validation_report=report,
                )
        except HookTheoryAdapterError as exc:
            counts["unexpected_conversion_failures"] += 1
            classification = (
                exc.validation_report.errors[0].code
                if exc.validation_report and exc.validation_report.errors
                else type(exc).__name__
            )
            _bounded_example(
                failure_examples,
                {
                    "clip_id": clip_id,
                    "classification": classification,
                    "message": str(exc)[:300],
                },
                args.sample_limit,
            )
            continue
        counts["valid_pieces"] += 1
        entity_totals.update(
            {
                "notes": len(piece.notes),
                "bars": len(piece.bars),
                "beats": len(piece.beats),
                "tempo_events": len(piece.tempo_events),
                "meter_events": len(piece.meter_events),
                "annotations": len(piece.annotations),
                "targets": len(piece.targets),
            }
        )
        quality_flags.update(flag.code for flag in piece.quality_flags)
        for target in piece.targets:
            available_masks[target.task] += sum(target.mask)
            unavailable_masks[target.task] += len(target.mask) - sum(target.mask)
        _sample_add(sample, clip_id, record, structure_row, args.sample_limit)

    round_trip_examples: list[dict[str, Any]] = []
    hiding_examples: list[dict[str, Any]] = []
    for _score, clip_id, record, structure_row in sample:
        counts["round_trip_attempted"] += 1
        try:
            piece = convert_hooktheory_record(
                clip_id,
                record,
                config=config,
                structure_row=structure_row,
                source_path=str(raw_path),
            )
            if loads_piece(dumps_piece(piece)) != piece:
                raise ValueError("round trip changed canonical piece")
        except (HookTheoryAdapterError, ValueError) as exc:
            counts["round_trip_failures"] += 1
            _bounded_example(
                round_trip_examples,
                {"clip_id": clip_id, "message": str(exc)[:300]},
                args.sample_limit,
            )
        else:
            counts["round_trip_passed"] += 1

        counts["target_hiding_attempted"] += 1
        try:
            visible = convert_hooktheory_record(
                clip_id,
                record,
                config=HookTheoryAdapterConfig(args.dataset_name, True),
                structure_row=structure_row,
                source_path=str(raw_path),
            )
            hidden = convert_hooktheory_record(
                clip_id,
                record,
                config=HookTheoryAdapterConfig(args.dataset_name, False),
                structure_row=structure_row,
                source_path=str(raw_path),
            )
            visible_ids = {item.provenance_id for item in visible.provenance}
            hidden_ids = {item.provenance_id for item in hidden.provenance}
            if (
                not _raw_content_equal(visible, hidden)
                or hidden.annotations
                or hidden.targets
                or visible_ids - hidden_ids != {"prov:annotation"}
            ):
                raise ValueError("target hiding changed non-target canonical content")
        except (HookTheoryAdapterError, ValueError) as exc:
            counts["target_hiding_failures"] += 1
            _bounded_example(
                hiding_examples,
                {"clip_id": clip_id, "message": str(exc)[:300]},
                args.sample_limit,
            )
        else:
            counts["target_hiding_passed"] += 1

    return {
        "report_schema_version": "hooktheory_adapter_smoke_v1",
        "source": str(raw_path),
        "include_targets": args.include_targets,
        "sample_limit": args.sample_limit,
        "counts": {name: counts[name] for name in sorted(counts)},
        "entity_totals": {name: entity_totals[name] for name in sorted(entity_totals)},
        "quality_flags": {name: quality_flags[name] for name in sorted(quality_flags)},
        "target_masks": {
            task: {
                "available": available_masks[task],
                "unavailable": unavailable_masks[task],
            }
            for task in sorted(set(available_masks) | set(unavailable_masks))
        },
        "bounded_failure_examples": failure_examples,
        "bounded_round_trip_failure_examples": round_trip_examples,
        "bounded_target_hiding_failure_examples": hiding_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--dataset-name", default="hooktheory")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--include-targets", dest="include_targets", action="store_true")
    targets.add_argument("--hide-targets", dest="include_targets", action="store_false")
    parser.set_defaults(include_targets=True)
    args = parser.parse_args()
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report_json is not None:
        args.report_json.write_text(rendered, encoding="utf-8")
    counts = report["counts"]
    return 0 if counts.get("unexpected_conversion_failures", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
