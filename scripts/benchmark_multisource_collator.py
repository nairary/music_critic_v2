#!/usr/bin/env python3
"""Run raw-only baseline or target-heavy Phase 5B.1 collator benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

try:
    from scripts.benchmark_graph_builder import make_synthetic_piece
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from benchmark_graph_builder import make_synthetic_piece

from music_critic.data import (
    AnnotationSpan,
    RationalTime,
    TargetArray,
    load_piece,
)
from music_critic.tasks import (
    benchmark_multisource_collator,
    benchmark_target_alignment,
    prepare_multisource_sample,
    TARGET_FAMILY_BY_ID,
)


def make_target_heavy_piece(note_count: int):
    """Create target-heavy exact-ID/span/boundary benchmark evidence."""

    piece = make_synthetic_piece(note_count, layout="sequential")
    provenance_id = "prov:benchmark"
    presence_annotations = tuple(
        AnnotationSpan(
            annotation_id=f"span:benchmark-presence-{index:06d}",
            annotation_type="theory.chord",
            layer="target_alignment",
            start_qn=bar.start_qn,
            end_qn=bar.start_qn + bar.duration_qn,
            track_id=None,
            value=None,
            provenance_id=provenance_id,
        )
        for index, bar in enumerate(piece.bars)
    )
    selected_beats = piece.beats[::2]
    boundary_annotations = tuple(
        AnnotationSpan(
            annotation_id=f"span:benchmark-boundary-{index:06d}",
            annotation_type="pop909_cl.chord",
            layer="target_alignment",
            start_qn=beat.start_qn,
            end_qn=min(
                beat.start_qn + RationalTime(1, 4),
                piece.duration_qn,
            ),
            track_id=None,
            value=None,
            provenance_id=provenance_id,
        )
        for index, beat in enumerate(selected_beats)
    )
    unaligned = AnnotationSpan(
        annotation_id="span:benchmark-boundary-unaligned",
        annotation_type="pop909_cl.chord",
        layer="target_alignment",
        start_qn=RationalTime(1, 8),
        end_qn=RationalTime(1, 4),
        track_id=None,
        value=None,
        provenance_id=provenance_id,
    )
    masked = AnnotationSpan(
        annotation_id="span:benchmark-boundary-masked",
        annotation_type="pop909_cl.chord",
        layer="target_alignment",
        start_qn=RationalTime(3, 8),
        end_qn=RationalTime(1, 2),
        track_id=None,
        value=None,
        provenance_id=provenance_id,
    )
    annotations = tuple(
        sorted(
            (*presence_annotations, *boundary_annotations, unaligned, masked),
            key=lambda item: (
                item.start_qn,
                item.end_qn,
                item.annotation_id,
            ),
        )
    )

    note_spec = TARGET_FAMILY_BY_ID["theory.melody.scale_degree"]
    presence_spec = TARGET_FAMILY_BY_ID["theory.chord.presence"]
    boundary_spec = TARGET_FAMILY_BY_ID["pop909_cl.chord.boundary"]
    note_count_actual = len(piece.notes)
    presence_count = len(presence_annotations)
    boundary_available_ids = tuple(
        annotation.annotation_id for annotation in boundary_annotations
    ) + (unaligned.annotation_id,)
    boundary_count = len(boundary_available_ids)
    targets = (
        TargetArray(
            target_id="target:benchmark-boundary",
            task=boundary_spec.task_id,
            annotation_view_id=boundary_spec.annotation_view_id,
            alignment_type="annotation_span",
            entity_ids=(*boundary_available_ids, masked.annotation_id),
            value_type="categorical",
            class_labels=boundary_spec.vocabulary,
            values=(*(("present",) * boundary_count), None),
            mask=(*((True,) * boundary_count), False),
            confidence=(None,) * (boundary_count + 1),
            source=(*(("synthetic",) * boundary_count), None),
            provenance=(*((provenance_id,) * boundary_count), None),
        ),
        TargetArray(
            target_id="target:benchmark-presence",
            task=presence_spec.task_id,
            annotation_view_id=presence_spec.annotation_view_id,
            alignment_type="annotation_span",
            entity_ids=tuple(
                annotation.annotation_id
                for annotation in presence_annotations
            ),
            value_type="categorical",
            class_labels=presence_spec.vocabulary,
            values=("true",) * presence_count,
            mask=(True,) * presence_count,
            confidence=(None,) * presence_count,
            source=("synthetic",) * presence_count,
            provenance=(provenance_id,) * presence_count,
        ),
        TargetArray(
            target_id="target:benchmark-note-identity",
            task=note_spec.task_id,
            annotation_view_id=note_spec.annotation_view_id,
            alignment_type="note",
            entity_ids=tuple(note.note_id for note in piece.notes),
            value_type="categorical",
            class_labels=note_spec.vocabulary,
            values=("1",) * note_count_actual,
            mask=(True,) * note_count_actual,
            confidence=(None,) * note_count_actual,
            source=("synthetic",) * note_count_actual,
            provenance=(provenance_id,) * note_count_actual,
        ),
    )
    return replace(piece, annotations=annotations, targets=targets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "canonical_json",
        type=Path,
        nargs="?",
        default=Path("tests/fixtures/data/canonical_piece_v2.json"),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--target-heavy",
        action="store_true",
        help="run small/medium/large target alignment benchmarks",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    if args.target_heavy:
        sizes = (("small", 64), ("medium", 512), ("large", 2_048))
        evidence = {
            name: asdict(
                benchmark_target_alignment(
                    (prepare_multisource_sample(
                        make_target_heavy_piece(note_count)
                    ),),
                    repeats=args.repeats,
                )
            )
            for name, note_count in sizes
        }
    else:
        piece = replace(
            load_piece(args.canonical_json),
            annotations=(),
            targets=(),
        )
        sample = prepare_multisource_sample(piece)
        evidence = asdict(
            benchmark_multisource_collator(
                (sample,) * args.samples,
                repeats=args.repeats,
            )
        )
    print(
        json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
