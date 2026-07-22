#!/usr/bin/env python3
"""Benchmark output-sensitive Phase 3A graph construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
from statistics import fmean
from time import perf_counter
import tracemalloc
from typing import Any, Iterable

import torch

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    MidiAdapterConfig,
    load_hooktheory_piece,
    load_midi_piece,
)
from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    RationalTime,
    TempoEvent,
    load_piece,
    validate_piece,
)
from music_critic.graph import build_raw_graph


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    name: str
    piece: CanonicalPiece


def _process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * 1024)  # Linux reports KiB; Phase 3A CI is Linux.


def _graph_tensor_bytes(graph: Any) -> int:
    total = 0
    for store in graph.stores:
        for value in store.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def benchmark_piece(piece: CanonicalPiece, *, repeats: int = 5) -> dict[str, Any]:
    """Validate once, then time the explicit validated-input fast path."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    validation_started = perf_counter()
    report = validate_piece(piece)
    validation_seconds = perf_counter() - validation_started
    if report.errors:
        codes = ", ".join(issue.code for issue in report.errors[:8])
        raise ValueError(f"benchmark input is not validator-clean: {codes}")

    build_raw_graph(piece, assume_valid=True)
    elapsed_seconds: list[float] = []
    graph = None
    tracemalloc.start()
    for _ in range(repeats):
        started = perf_counter()
        graph = build_raw_graph(piece, assume_valid=True)
        elapsed_seconds.append(perf_counter() - started)
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert graph is not None

    node_counts = {
        node_type: graph[node_type].num_nodes for node_type in graph.node_types
    }
    edge_counts = {
        "|".join(edge_type): int(graph[edge_type].edge_index.shape[1])
        for edge_type in graph.edge_types
    }
    return {
        "schema_version": graph.schema_version,
        "graph_schema_version": graph.graph_schema_version,
        "feature_registry_version": graph.feature_registry_version,
        "graph_builder_version": graph.graph_builder_version,
        "repeats": repeats,
        "input_validation_seconds": validation_seconds,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "total_nodes": sum(node_counts.values()),
        "total_edges": sum(edge_counts.values()),
        "active_at_edges": edge_counts["note|active_at|beat"],
        "output_tensor_bytes": _graph_tensor_bytes(graph),
        "python_peak_traced_bytes": python_peak_bytes,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "construction_seconds": {
            "mean": fmean(elapsed_seconds),
            "min": min(elapsed_seconds),
            "max": max(elapsed_seconds),
        },
    }


def _bars_and_beats(
    bar_count: int,
    *,
    provenance_id: str,
) -> tuple[tuple[CanonicalBar, ...], tuple[CanonicalBeat, ...]]:
    bars: list[CanonicalBar] = []
    beats: list[CanonicalBeat] = []
    for bar_index in range(bar_count):
        bar_id = f"bar:{bar_index:06d}"
        bar_start = RationalTime(bar_index * 4)
        bars.append(
            CanonicalBar(
                bar_id=bar_id,
                index=bar_index,
                start_qn=bar_start,
                duration_qn=RationalTime(4),
                meter_event_id="meter:000",
                metric_offset_qn=RationalTime(0),
                is_pickup=False,
                is_incomplete=False,
                display_number=str(bar_index + 1),
                provenance_id=provenance_id,
            )
        )
        for position in range(4):
            beat_index = bar_index * 4 + position
            beats.append(
                CanonicalBeat(
                    beat_id=f"beat:{beat_index:07d}",
                    bar_id=bar_id,
                    meter_event_id="meter:000",
                    index_in_bar=position,
                    start_qn=RationalTime(beat_index),
                    duration_qn=RationalTime(1),
                    position_in_bar_qn=RationalTime(position),
                    is_downbeat=position == 0,
                    strength=1.0 if position == 0 else 0.5,
                    provenance_id=provenance_id,
                )
            )
    return tuple(bars), tuple(beats)


def make_synthetic_piece(
    note_count: int,
    *,
    layout: str = "sequential",
    sustain_beats: int = 1_000,
) -> CanonicalPiece:
    """Create a validator-clean scaling fixture without datasets or caches."""

    if note_count <= 0:
        raise ValueError("note_count must be positive")
    if layout not in {"sequential", "dense_same_onset", "long_sustained"}:
        raise ValueError(f"unsupported synthetic layout {layout!r}")
    if sustain_beats <= 0:
        raise ValueError("sustain_beats must be positive")

    if layout == "sequential":
        bar_count = max(1, (note_count + 15) // 16)
    elif layout == "dense_same_onset":
        bar_count = 1
    else:
        bar_count = max(1, (sustain_beats + 3) // 4)
    duration_qn = RationalTime(bar_count * 4)
    provenance_id = "prov:benchmark"
    bars, beats = _bars_and_beats(bar_count, provenance_id=provenance_id)

    notes: list[CanonicalNote] = []
    for note_index in range(note_count):
        if layout == "sequential":
            onset = RationalTime(note_index, 4)
            note_duration = RationalTime(1, 4)
        elif layout == "dense_same_onset":
            onset = RationalTime(0)
            note_duration = RationalTime(1)
        else:
            onset = RationalTime(0)
            note_duration = RationalTime(sustain_beats)
        pitch = 36 + note_index % 72
        notes.append(
            CanonicalNote(
                note_id=f"note:{note_index:07d}",
                track_id="track:benchmark",
                pitch=pitch,
                onset_qn=onset,
                duration_qn=note_duration,
                velocity=80,
                channel=0,
                program=0,
                is_percussion=False,
                is_grace=False,
                spelling_step=None,
                spelling_alter=None,
                staff=None,
                voice=None,
                articulations=(),
                dynamic=None,
                source_onset_ticks=None,
                source_duration_ticks=None,
                source_onset_seconds=None,
                source_duration_seconds=None,
                provenance_id=provenance_id,
            )
        )
    notes.sort(
        key=lambda note: (
            note.onset_qn,
            note.pitch,
            note.duration_qn,
            note.note_id,
        )
    )

    detail_items: list[tuple[str, str | int]] = [
        ("layout", layout),
        ("note_count", note_count),
    ]
    if layout == "long_sustained":
        detail_items.append(("sustain_beats", sustain_beats))
    return CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id=f"piece:benchmark-{layout}-{note_count}-{sustain_beats}",
        dataset_name="synthetic_benchmark",
        source_group_id=f"benchmark-{layout}-{note_count}-{sustain_beats}",
        split=None,
        source_path=None,
        source_resolution=None,
        duration_qn=duration_qn,
        metadata=PieceMetadata(
            source_format="synthetic",
            title=None,
            creators=None,
            collection=None,
            movement_title=None,
            movement_number=None,
            genres=None,
            copyright=None,
            language=None,
        ),
        tracks=(
            CanonicalTrack(
                track_id="track:benchmark",
                source_track_index=0,
                name=None,
                instrument_name=None,
                program=0,
                channel=0,
                is_percussion=False,
                provenance_id=provenance_id,
            ),
        ),
        notes=tuple(notes),
        bars=bars,
        beats=beats,
        tempo_events=(
            TempoEvent(
                tempo_event_id="tempo:000",
                onset_qn=RationalTime(0),
                microseconds_per_quarter=500_000,
                provenance_id=provenance_id,
            ),
        ),
        meter_events=(
            MeterEvent(
                meter_event_id="meter:000",
                onset_qn=RationalTime(0),
                numerator=4,
                denominator=4,
                provenance_id=provenance_id,
            ),
        ),
        key_signature_events=(),
        annotations=(),
        targets=(),
        provenance=(
            ProvenanceRecord(
                provenance_id=provenance_id,
                kind="synthetic",
                source="graph_benchmark",
                record_id=None,
                uri=None,
                version="1",
                checksum_sha256=None,
                created_at=None,
                parents=(),
                details=tuple(detail_items),
            ),
        ),
        quality_flags=(),
    )


def synthetic_cases(
    *,
    note_counts: Iterable[int] = (100, 1_000, 10_000),
    dense_count: int = 10_000,
    sustain_note_count: int = 100,
    sustain_beats: int = 1_000,
) -> tuple[SyntheticCase, ...]:
    cases = [
        SyntheticCase(
            f"sequential_{count}",
            make_synthetic_piece(count, layout="sequential"),
        )
        for count in note_counts
    ]
    cases.extend(
        [
            SyntheticCase(
                f"dense_same_onset_{dense_count}",
                make_synthetic_piece(dense_count, layout="dense_same_onset"),
            ),
            SyntheticCase(
                f"long_sustained_{sustain_note_count}x{sustain_beats}",
                make_synthetic_piece(
                    sustain_note_count,
                    layout="long_sustained",
                    sustain_beats=sustain_beats,
                ),
            ),
        ]
    )
    return tuple(cases)


def benchmark_synthetic_suite(
    *,
    repeats: int = 3,
    note_counts: Iterable[int] = (100, 1_000, 10_000),
    dense_count: int = 10_000,
    sustain_note_count: int = 100,
    sustain_beats: int = 1_000,
) -> dict[str, Any]:
    reports = {
        case.name: benchmark_piece(case.piece, repeats=repeats)
        for case in synthetic_cases(
            note_counts=note_counts,
            dense_count=dense_count,
            sustain_note_count=sustain_note_count,
            sustain_beats=sustain_beats,
        )
    }
    peak_case = max(reports, key=lambda name: reports[name]["output_tensor_bytes"])
    return {
        "cases": reports,
        "peak_output_case": peak_case,
        "peak_output_tensor_bytes": reports[peak_case]["output_tensor_bytes"],
        "complexity_contract": "O((N+O) log B + E_active + E_graph)",
    }


def _first_midi_paths(root: Path, limit: int) -> list[Path]:
    selected: list[Path] = []
    for directory, names, files in os.walk(root):
        names.sort()
        for filename in sorted(files):
            if Path(filename).suffix.lower() not in {".mid", ".midi"}:
                continue
            selected.append(Path(directory) / filename)
            if len(selected) == limit:
                return selected
    return selected


def benchmark_real_midi_root(
    root: Path,
    *,
    dataset_name: str,
    limit: int,
    repeats: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _first_midi_paths(root, limit):
        try:
            piece = load_midi_piece(path, config=MidiAdapterConfig(dataset_name))
            reports.append({"path": str(path), **benchmark_piece(piece, repeats=repeats)})
        except Exception as exc:  # Diagnostic smoke must retain per-file failures.
            failures.append({"path": str(path), "error": str(exc)})
    return {"attempted": len(reports) + len(failures), "reports": reports, "failures": failures}


def benchmark_hooktheory(
    source: Path,
    clip_ids: Iterable[str],
    *,
    repeats: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for clip_id in clip_ids:
        try:
            piece = load_hooktheory_piece(
                source,
                clip_id,
                config=HookTheoryAdapterConfig("hooktheory-benchmark"),
            )
            reports.append({"clip_id": clip_id, **benchmark_piece(piece, repeats=repeats)})
        except Exception as exc:  # Diagnostic smoke must retain per-record failures.
            failures.append({"clip_id": clip_id, "error": str(exc)})
    return {"attempted": len(reports) + len(failures), "reports": reports, "failures": failures}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_json", type=Path, nargs="?")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--synthetic-suite", action="store_true")
    parser.add_argument("--note-counts", type=int, nargs="+", default=[100, 1_000, 10_000])
    parser.add_argument("--dense-count", type=int, default=10_000)
    parser.add_argument("--sustain-note-count", type=int, default=100)
    parser.add_argument("--sustain-beats", type=int, default=1_000)
    parser.add_argument("--pop909-root", type=Path)
    parser.add_argument("--pdmx-root", type=Path)
    parser.add_argument("--real-limit", type=int, default=3)
    parser.add_argument("--hooktheory-source", type=Path)
    parser.add_argument("--hooktheory-clip-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output: dict[str, Any] = {}
    if args.canonical_json is not None:
        output["canonical"] = benchmark_piece(
            load_piece(args.canonical_json), repeats=args.repeats
        )
    if args.synthetic_suite:
        output["synthetic"] = benchmark_synthetic_suite(
            repeats=args.repeats,
            note_counts=args.note_counts,
            dense_count=args.dense_count,
            sustain_note_count=args.sustain_note_count,
            sustain_beats=args.sustain_beats,
        )
    if args.pop909_root is not None:
        output["pop909"] = benchmark_real_midi_root(
            args.pop909_root,
            dataset_name="pop909-benchmark",
            limit=args.real_limit,
            repeats=args.repeats,
        )
    if args.pdmx_root is not None:
        output["pdmx"] = benchmark_real_midi_root(
            args.pdmx_root,
            dataset_name="pdmx-benchmark",
            limit=args.real_limit,
            repeats=args.repeats,
        )
    if args.hooktheory_source is not None:
        if not args.hooktheory_clip_id:
            raise SystemExit("--hooktheory-source requires --hooktheory-clip-id")
        output["hooktheory"] = benchmark_hooktheory(
            args.hooktheory_source,
            args.hooktheory_clip_id,
            repeats=args.repeats,
        )
    if not output:
        raise SystemExit(
            "provide canonical_json, --synthetic-suite, or an optional real-data source"
        )
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
