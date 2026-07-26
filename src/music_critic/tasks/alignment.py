"""Exact canonical target alignment for Phase 5B.1."""

from __future__ import annotations

from dataclasses import dataclass
from torch_geometric.data import HeteroData

from music_critic.data import CanonicalPiece, RationalTime, validate_piece
from music_critic.graph import GraphContractError, validate_raw_graph
from music_critic.tasks.multisource import (
    MultiSourceContractError,
    MultiSourceSample,
    SampleTarget,
    TargetDiagnostic,
    TargetRowProvenance,
)
from music_critic.tasks.ontology import TARGET_FAMILIES, TARGET_FAMILY_BY_ID


ALIGNMENT_CONFLICT_DIAGNOSTIC = "multisource.alignment_conflict"


class TargetAlignmentError(MultiSourceContractError):
    """Raised when canonical targets cannot be aligned exactly and safely."""


@dataclass(frozen=True, slots=True)
class AlignedTargetRow:
    """One local-index target row before PyG batch offsets are applied."""

    value: object | None
    availability: bool
    local_entity_index: int
    entity_node_type: str | None
    confidence: float | None
    provenance: TargetRowProvenance
    diagnostics: tuple[TargetDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.availability, bool):
            raise TargetAlignmentError("aligned row availability must be boolean")
        if self.availability != (self.value is not None):
            raise TargetAlignmentError(
                "aligned row availability must match non-null source value"
            )
        if (self.local_entity_index == -1) != (self.entity_node_type is None):
            raise TargetAlignmentError(
                "local entity index -1 requires and only permits a null node type"
            )
        if self.local_entity_index < -1:
            raise TargetAlignmentError("local entity index cannot be below -1")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise TargetAlignmentError(
                "aligned row confidence must lie in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class AlignedTargetFamily:
    """All deterministic local rows for one source-native task."""

    task_id: str
    source_entry_count: int
    rows: tuple[AlignedTargetRow, ...]

    def __post_init__(self) -> None:
        if self.task_id not in TARGET_FAMILY_BY_ID:
            raise TargetAlignmentError("aligned family task is absent from ontology")
        if (
            isinstance(self.source_entry_count, bool)
            or not isinstance(self.source_entry_count, int)
            or self.source_entry_count < 0
        ):
            raise TargetAlignmentError(
                "aligned family source entry count must be non-negative"
            )


def _onset_times(piece: CanonicalPiece) -> tuple[RationalTime, ...]:
    return tuple(dict.fromkeys(note.onset_qn for note in piece.notes))


def _expected_graph_entity_ids(
    piece: CanonicalPiece,
) -> dict[str, tuple[str, ...]]:
    onset_times = _onset_times(piece)
    return {
        "song": (piece.piece_id,),
        "track": tuple(track.track_id for track in piece.tracks),
        "bar": tuple(bar.bar_id for bar in piece.bars),
        "beat": tuple(beat.beat_id for beat in piece.beats),
        "onset": tuple(
            f"onset:{time.num}_{time.den}" for time in onset_times
        ),
        "note": tuple(note.note_id for note in piece.notes),
    }


def _validate_alignment_inputs(
    piece: CanonicalPiece,
    raw_graph: HeteroData,
    sample: MultiSourceSample,
) -> None:
    report = validate_piece(piece)
    if report.errors:
        summary = ", ".join(
            f"{issue.code}@{issue.path}" for issue in report.errors[:8]
        )
        raise TargetAlignmentError(
            f"exact target alignment requires a validated canonical piece: {summary}"
        )
    if sample.canonical_piece != piece:
        raise TargetAlignmentError(
            "alignment canonical piece differs from prepared sample"
        )
    if sample.raw_graph is not raw_graph:
        raise TargetAlignmentError(
            "alignment raw graph must be the prepared sample raw graph"
        )
    try:
        validate_raw_graph(raw_graph)
    except GraphContractError as exc:
        raise TargetAlignmentError(
            f"alignment requires a valid raw-only graph: {exc}"
        ) from exc
    for node_type, expected in _expected_graph_entity_ids(piece).items():
        actual = getattr(raw_graph[node_type], "entity_id", None)
        if actual != expected:
            raise TargetAlignmentError(
                f"{node_type} entity ordering differs from graph builder contract"
            )


def _row_provenance(
    *,
    entity_id: str,
    provenance_id: str | None,
    source: str | None,
) -> TargetRowProvenance:
    return TargetRowProvenance(
        source_entity_ids=(entity_id,),
        provenance_ids=(provenance_id,) if provenance_id is not None else (),
        sources=(source,) if source is not None else (),
    )


def _candidate_maps(
    piece: CanonicalPiece,
) -> dict[str, tuple[tuple[RationalTime, int], ...]]:
    return {
        "onset": tuple(
            (time, index) for index, time in enumerate(_onset_times(piece))
        ),
        "beat": tuple(
            (beat.start_qn, index) for index, beat in enumerate(piece.beats)
        ),
        "bar": tuple(
            (bar.start_qn, index) for index, bar in enumerate(piece.bars)
        ),
    }


def _available_matches(
    *,
    piece: CanonicalPiece,
    target: SampleTarget,
    source_index: int,
    candidate_maps: dict[str, tuple[tuple[RationalTime, int], ...]],
) -> tuple[tuple[str, int], ...]:
    spec = TARGET_FAMILY_BY_ID[target.task_id]
    entity_id = target.entity_ids[source_index]
    if target.alignment_type == "note":
        note_indices = {
            note.note_id: index for index, note in enumerate(piece.notes)
        }
        note_index = note_indices.get(entity_id)
        return (("note", note_index),) if note_index is not None else ()

    annotations = {
        annotation.annotation_id: annotation for annotation in piece.annotations
    }
    annotation = annotations.get(entity_id)
    if annotation is None:
        raise TargetAlignmentError(
            f"available target entity {entity_id!r} has no canonical annotation"
        )
    matches: list[tuple[str, int]] = []
    for rule in spec.alignment_policy.candidate_rules:
        candidates = candidate_maps.get(rule.node_type, ())
        if rule.match_rule == "exact_event_time":
            matches.extend(
                (rule.node_type, local_index)
                for time, local_index in candidates
                if time == annotation.start_qn
            )
        elif rule.match_rule == "half_open_containment":
            matches.extend(
                (rule.node_type, local_index)
                for time, local_index in candidates
                if annotation.start_qn <= time < annotation.end_qn
            )
        else:
            raise TargetAlignmentError(
                f"unsupported alignment rule {rule.match_rule!r}"
            )
    return tuple(matches)


def _merge_provenance(
    rows: tuple[AlignedTargetRow, ...],
) -> TargetRowProvenance:
    ordered = tuple(
        sorted(rows, key=lambda row: row.provenance.source_entity_ids)
    )
    return TargetRowProvenance(
        source_entity_ids=tuple(
            entity_id
            for row in ordered
            for entity_id in row.provenance.source_entity_ids
        ),
        provenance_ids=tuple(
            provenance_id
            for row in ordered
            for provenance_id in row.provenance.provenance_ids
        ),
        sources=tuple(
            source for row in ordered for source in row.provenance.sources
        ),
    )


def _merge_available_aligned_rows(
    target: SampleTarget,
    rows: tuple[AlignedTargetRow, ...],
) -> tuple[AlignedTargetRow, ...]:
    spec = TARGET_FAMILY_BY_ID[target.task_id]
    node_order = {
        node_type: index
        for index, node_type in enumerate(
            spec.alignment_policy.candidate_node_types
        )
    }
    aligned: dict[tuple[str, int], list[AlignedTargetRow]] = {}
    retained: list[AlignedTargetRow] = []
    for row in rows:
        if (
            row.availability
            and row.entity_node_type is not None
            and row.local_entity_index >= 0
        ):
            aligned.setdefault(
                (row.entity_node_type, row.local_entity_index), []
            ).append(row)
        else:
            retained.append(row)

    merged: list[AlignedTargetRow] = []
    for (node_type, local_index), candidates in sorted(
        aligned.items(),
        key=lambda item: (node_order[item[0][0]], item[0][1]),
    ):
        candidate_rows = tuple(candidates)
        value = candidate_rows[0].value
        values_equal = all(row.value == value for row in candidate_rows)
        provenance = _merge_provenance(candidate_rows)
        diagnostics = tuple(
            diagnostic
            for row in candidate_rows
            for diagnostic in row.diagnostics
        )
        if values_equal:
            confidences = {row.confidence for row in candidate_rows}
            merged.append(
                AlignedTargetRow(
                    value=value,
                    availability=True,
                    local_entity_index=local_index,
                    entity_node_type=node_type,
                    confidence=(
                        next(iter(confidences)) if len(confidences) == 1 else None
                    ),
                    provenance=provenance,
                    diagnostics=diagnostics,
                )
            )
            continue
        conflict = TargetDiagnostic(
            code=ALIGNMENT_CONFLICT_DIAGNOSTIC,
            message=(
                f"conflicting available values for {target.task_id} at "
                f"{node_type}[{local_index}]"
            ),
            source_entity_ids=provenance.source_entity_ids,
        )
        merged.append(
            AlignedTargetRow(
                value=None,
                availability=False,
                local_entity_index=local_index,
                entity_node_type=node_type,
                confidence=None,
                provenance=provenance,
                diagnostics=(*diagnostics, conflict),
            )
        )

    retained.sort(
        key=lambda row: (
            row.provenance.source_entity_ids,
            not row.availability,
        )
    )
    return (*merged, *retained)


def _align_target(
    *,
    piece: CanonicalPiece,
    target: SampleTarget,
    candidate_maps: dict[str, tuple[tuple[RationalTime, int], ...]],
) -> AlignedTargetFamily:
    rows: list[AlignedTargetRow] = []
    for source_index, entity_id in enumerate(target.entity_ids):
        available = target.availability_mask[source_index]
        provenance = _row_provenance(
            entity_id=entity_id,
            provenance_id=target.provenance_ids[source_index],
            source=target.source[source_index],
        )
        if not available:
            rows.append(
                AlignedTargetRow(
                    value=None,
                    availability=False,
                    local_entity_index=-1,
                    entity_node_type=None,
                    confidence=None,
                    provenance=provenance,
                    diagnostics=(),
                )
            )
            continue
        matches = _available_matches(
            piece=piece,
            target=target,
            source_index=source_index,
            candidate_maps=candidate_maps,
        )
        if not matches:
            rows.append(
                AlignedTargetRow(
                    value=target.values[source_index],
                    availability=True,
                    local_entity_index=-1,
                    entity_node_type=None,
                    confidence=target.confidence[source_index],
                    provenance=provenance,
                    diagnostics=(),
                )
            )
            continue
        rows.extend(
            AlignedTargetRow(
                value=target.values[source_index],
                availability=True,
                local_entity_index=local_index,
                entity_node_type=node_type,
                confidence=target.confidence[source_index],
                provenance=provenance,
                diagnostics=(),
            )
            for node_type, local_index in matches
        )
    return AlignedTargetFamily(
        task_id=target.task_id,
        source_entry_count=len(target.entity_ids),
        rows=_merge_available_aligned_rows(target, tuple(rows)),
    )


def align_sample_targets(
    piece: CanonicalPiece,
    raw_graph: HeteroData,
    sample: MultiSourceSample,
) -> tuple[AlignedTargetFamily, ...]:
    """Align every registry task using canonical IDs and exact rational time."""

    _validate_alignment_inputs(piece, raw_graph, sample)
    candidate_maps = _candidate_maps(piece)
    by_task = {target.task_id: target for target in sample.target_bundle}
    return tuple(
        (
            _align_target(
                piece=piece,
                target=by_task[spec.task_id],
                candidate_maps=candidate_maps,
            )
            if spec.task_id in by_task
            else AlignedTargetFamily(
                task_id=spec.task_id,
                source_entry_count=0,
                rows=(),
            )
        )
        for spec in TARGET_FAMILIES
    )


__all__ = [
    "ALIGNMENT_CONFLICT_DIAGNOSTIC",
    "AlignedTargetFamily",
    "AlignedTargetRow",
    "TargetAlignmentError",
    "align_sample_targets",
]
