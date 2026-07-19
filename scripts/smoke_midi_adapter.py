#!/usr/bin/env python3
"""Bounded smoke test for the generic MIDI adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mido

from music_critic.adapters import MidiAdapterConfig, load_midi_piece
from music_critic.data import dumps_piece, loads_piece, validate_piece


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _discover(path: Path) -> list[Path] | None:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            (
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".mid", ".midi"}
            ),
            key=lambda candidate: candidate.as_posix(),
        )
    return None


def _short_reason(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:240]


def _relative_label(path: Path, root: Path) -> str:
    if root.is_dir():
        return path.relative_to(root).as_posix()
    return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--dataset-name", default="midi-smoke")
    parser.add_argument("--split")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    discovered = _discover(args.path)
    if discovered is None:
        print(f"invalid input path: {args.path}", file=sys.stderr)
        return 2
    selected = discovered[: args.limit] if args.limit is not None else discovered
    summary = {
        "files_seen": len(discovered),
        "attempted": 0,
        "converted": 0,
        "failed": 0,
        "warnings": 0,
        "notes": 0,
        "tracks": 0,
        "type_0": 0,
        "type_1": 0,
    }
    failures: list[str] = []
    config = MidiAdapterConfig(dataset_name=args.dataset_name, split=args.split)
    for midi_path in selected:
        summary["attempted"] += 1
        try:
            piece = load_midi_piece(str(midi_path), config=config)
            report = validate_piece(piece)
            if report.errors:
                raise RuntimeError(f"adapter returned {len(report.errors)} validation errors")
            if loads_piece(dumps_piece(piece)) != piece:
                raise RuntimeError("canonical JSON string round-trip changed the piece")
            midi_type = mido.MidiFile(filename=midi_path).type
            summary[f"type_{midi_type}"] += 1
            summary["converted"] += 1
            summary["warnings"] += len(report.warnings) + sum(
                flag.severity == "warning" for flag in piece.quality_flags
            )
            summary["notes"] += len(piece.notes)
            summary["tracks"] += len(piece.tracks)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            summary["failed"] += 1
            failures.append(
                f"{_relative_label(midi_path, args.path)} | "
                f"{type(exc).__name__} | {_short_reason(exc)}"
            )
            if args.fail_fast:
                break

    for failure in failures:
        print(failure)
    for key in (
        "files_seen",
        "attempted",
        "converted",
        "failed",
        "warnings",
        "notes",
        "tracks",
        "type_0",
        "type_1",
    ):
        print(f"{key}={summary[key]}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
