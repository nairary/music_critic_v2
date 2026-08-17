"""Indexed exact canonical target alignment for Phase 5B.1."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from torch_geometric.data import HeteroData

from music_critic.data import (
    AnnotationSpan,
    CanonicalPiece,
    RationalTime,
    validate_piece,
)
from music_critic.graph import GraphContractError, validate_raw_graph
from music_critic.tasks.multisource import (
    MultiSourceContractError,
    MultiSourceSample,
    SampleTarget,
    TargetDiagnostic,
    TargetRowProvenance,
)
from music_critic.tasks.registry import target_family_spec


ALIGNMENT_CONFLICT_DIAGNOSTIC = "multisource.alignment_conflict"


class TargetAlignmentError(MultiSourceContractError):
    """Raised when canonical targets cannot be aligned exactly and safely."""


@dataclass(slots=True)
class AlignmentOperationCounts:
    """Optional non-timing evidence for indexed alignment complexity."""

    index_build_count: int = 0
    note_index_entry_count: int = 0
    annotation_index_entry_count: int = 0
    candidate_index_entry_count: int = 0
    source_entry_lookup_count: int = 0
    note_identity_lookup_count: int = 0
    annotation_lookup_count: int = 0
    exact_time_lookup_count: int = 0
    span_bisect_count: int = 0
    candidate_match_count: int = 0
    merge_candidate_slot_visit_count: int = 0
    emitted_row_count: int = 0

    def as_sorted_pairs(self) -> tuple[tuple[str, int], ...]:
        """Return an immutable deterministic benchmark representation."""

        return tuple(
            sorted(
                (name, int(getattr(self, name)))
                for name in self.__dataclass_fields__
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateTimeIndex:
    """Immutable sorted/exact lookup for one temporal raw node store."""

    node_type: str
    times: tuple[RationalTime, ...]
    local_indices: tuple[int, ...]
    exact_indices_by_time: Mapping[RationalTime, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.node_type not in {"onset", "beat", "bar"}:
            raise TargetAlignmentError(
                "candidate time index has an unsupported node type"
            )
        if len(self.times) != len(self.local_indices):
            raise TargetAlignmentError(
                "candidate time/index arrays must have equal length"
            )
        if tuple(sorted(zip(self.times, self.local_indices))) != tuple(
            zip(self.times, self.local_indices)
        ):
            raise TargetAlignmentError(
                "candidate time/index arrays must use deterministic sorted order"
            )
        if not isinstance(self.exact_indices_by_time, MappingProxyType):
            raise TargetAlignmentError(
                "candidate exact-time lookup must be immutable"
            )


@dataclass(frozen=True, slots=True)
class AlignmentIndex:
    """One immutable, output-sensitive index over a canonical piece."""

    canonical_piece: CanonicalPiece
    note_index_by_id: Mapping[str, int]
    annotation_by_id: Mapping[str, AnnotationSpan]
    candidate_by_node_type: Mapping[str, CandidateTimeIndex]

    def __post_init__(self) -> None:
        if not isinstance(self.note_index_by_id, MappingProxyType):
            raise TargetAlignmentError("note identity index must be immutable")
        if not isinstance(self.annotation_by_id, MappingProxyType):
            raise TargetAlignmentError("annotation index must be immutable")
        if not isinstance(self.candidate_by_node_type, MappingProxyType):
            raise TargetAlignmentError("candidate index mapping must be immutable")
        if tuple(self.candidate_by_node_type) != ("onset", "beat", "bar"):
            raise TargetAlignmentError(
                "candidate indices must use onset/beat/bar contract order"
            )

    def local_indices(self, node_type: str) -> tuple[int, ...]:
        """Return deterministic local indices for merge/output ordering."""

        if node_type == "note":
            return tuple(range(len(self.note_index_by_id)))
        candidate = self.candidate_by_node_type.get(node_type)
        if candidate is None:
            raise TargetAlignmentError(
                f"alignment index has no candidate store {node_type!r}"
            )
        return candidate.local_indices


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
        try:
            target_family_spec(self.task_id)
        except KeyError:
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


def _validate_piece_for_alignment(piece: CanonicalPiece) -> None:
    report = validate_piece(piece)
    if report.errors:
        summary = ", ".join(
            f"{issue.code}@{issue.path}" for issue in report.errors[:8]
        )
        raise TargetAlignmentError(
            f"exact target alignment requires a validated canonical piece: {summary}"
        )


def _validate_alignment_inputs(
    piece: CanonicalPiece,
    raw_graph: HeteroData,
    sample: MultiSourceSample,
) -> None:
    _validate_piece_for_alignment(piece)
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


def _candidate_time_index(
    node_type: str,
    candidates: tuple[tuple[RationalTime, int], ...],
) -> CandidateTimeIndex:
    ordered = tuple(sorted(candidates))
    exact: dict[RationalTime, list[int]] = {}
    for time, local_index in ordered:
        exact.setdefault(time, []).append(local_index)
    return CandidateTimeIndex(
        node_type=node_type,
        times=tuple(time for time, _ in ordered),
        local_indices=tuple(local_index for _, local_index in ordered),
        exact_indices_by_time=MappingProxyType(
            {
                time: tuple(local_indices)
                for time, local_indices in exact.items()
            }
        ),
    )


def _build_alignment_index(
    piece: CanonicalPiece,
    *,
    target_alignment_spans: tuple[AnnotationSpan, ...] = (),
    instrumentation: AlignmentOperationCounts | None,
) -> AlignmentIndex:
    note_index = {
        note.note_id: index for index, note in enumerate(piece.notes)
    }
    annotations = (*piece.annotations, *target_alignment_spans)
    annotation_index = {
        annotation.annotation_id: annotation
        for annotation in annotations
    }
    if len(annotation_index) != len(annotations):
        raise TargetAlignmentError(
            "canonical and target-sidecar annotation IDs must be unique"
        )
    candidates = {
        "onset": _candidate_time_index(
            "onset",
            tuple(
                (time, index)
                for index, time in enumerate(_onset_times(piece))
            ),
        ),
        "beat": _candidate_time_index(
            "beat",
            tuple(
                (beat.start_qn, index)
                for index, beat in enumerate(piece.beats)
            ),
        ),
        "bar": _candidate_time_index(
            "bar",
            tuple(
                (bar.start_qn, index)
                for index, bar in enumerate(piece.bars)
            ),
        ),
    }
    if instrumentation is not None:
        instrumentation.index_build_count += 1
        instrumentation.note_index_entry_count += len(note_index)
        instrumentation.annotation_index_entry_count += len(annotation_index)
        instrumentation.candidate_index_entry_count += sum(
            len(candidate.times) for candidate in candidates.values()
        )
    return AlignmentIndex(
        canonical_piece=piece,
        note_index_by_id=MappingProxyType(note_index),
        annotation_by_id=MappingProxyType(annotation_index),
        candidate_by_node_type=MappingProxyType(candidates),
    )


def build_alignment_index(
    piece: CanonicalPiece,
    *,
    target_alignment_spans: tuple[AnnotationSpan, ...] = (),
    instrumentation: AlignmentOperationCounts | None = None,
) -> AlignmentIndex:
    """Build one validated immutable index in O(P + C log C)."""

    _validate_piece_for_alignment(piece)
    return _build_alignment_index(
        piece,
        target_alignment_spans=target_alignment_spans,
        instrumentation=instrumentation,
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


def _available_matches(
    *,
    alignment_index: AlignmentIndex,
    target: SampleTarget,
    source_index: int,
    instrumentation: AlignmentOperationCounts | None,
) -> tuple[tuple[str, int], ...]:
    spec = target_family_spec(target.task_id)
    entity_id = target.entity_ids[source_index]
    if instrumentation is not None:
        instrumentation.source_entry_lookup_count += 1
    if target.alignment_type == "note":
        if instrumentation is not None:
            instrumentation.note_identity_lookup_count += 1
        note_index = alignment_index.note_index_by_id.get(entity_id)
        matches = (("note", note_index),) if note_index is not None else ()
        if instrumentation is not None:
            instrumentation.candidate_match_count += len(matches)
        return matches

    if instrumentation is not None:
        instrumentation.annotation_lookup_count += 1
    annotation = alignment_index.annotation_by_id.get(entity_id)
    if annotation is None:
        raise TargetAlignmentError(
            f"available target entity {entity_id!r} has no canonical annotation"
        )
    matches: list[tuple[str, int]] = []
    for rule in spec.alignment_policy.candidate_rules:
        candidates = alignment_index.candidate_by_node_type[rule.node_type]
        if rule.match_rule == "exact_event_time":
            if instrumentation is not None:
                instrumentation.exact_time_lookup_count += 1
            matches.extend(
                (rule.node_type, local_index)
                for local_index in candidates.exact_indices_by_time.get(
                    annotation.start_qn, ()
                )
            )
        elif rule.match_rule == "half_open_containment":
            if instrumentation is not None:
                instrumentation.span_bisect_count += 2
            start = bisect_left(candidates.times, annotation.start_qn)
            end = bisect_left(candidates.times, annotation.end_qn)
            matches.extend(
                (rule.node_type, local_index)
                for local_index in candidates.local_indices[start:end]
            )
        else:
            raise TargetAlignmentError(
                f"unsupported alignment rule {rule.match_rule!r}"
            )
    if instrumentation is not None:
        instrumentation.candidate_match_count += len(matches)
    return tuple(matches)


def _merge_provenance(
    rows: tuple[AlignedTargetRow, ...],
) -> TargetRowProvenance:
    return TargetRowProvenance(
        source_entity_ids=tuple(
            entity_id
            for row in rows
            for entity_id in row.provenance.source_entity_ids
        ),
        provenance_ids=tuple(
            provenance_id
            for row in rows
            for provenance_id in row.provenance.provenance_ids
        ),
        sources=tuple(
            source for row in rows for source in row.provenance.sources
        ),
    )


def _merge_available_aligned_rows(
    target: SampleTarget,
    rows: tuple[AlignedTargetRow, ...],
    alignment_index: AlignmentIndex,
    instrumentation: AlignmentOperationCounts | None,
) -> tuple[AlignedTargetRow, ...]:
    spec = target_family_spec(target.task_id)
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
    for node_type in spec.alignment_policy.candidate_node_types:
        for local_index in alignment_index.local_indices(node_type):
            if instrumentation is not None:
                instrumentation.merge_candidate_slot_visit_count += 1
            candidates = aligned.get((node_type, local_index))
            if candidates is None:
                continue
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
                            next(iter(confidences))
                            if len(confidences) == 1
                            else None
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

    return (*merged, *retained)


def _align_target(
    *,
    alignment_index: AlignmentIndex,
    target: SampleTarget,
    instrumentation: AlignmentOperationCounts | None,
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
            if instrumentation is not None:
                instrumentation.source_entry_lookup_count += 1
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
            alignment_index=alignment_index,
            target=target,
            source_index=source_index,
            instrumentation=instrumentation,
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
    family = AlignedTargetFamily(
        task_id=target.task_id,
        source_entry_count=len(target.entity_ids),
        rows=_merge_available_aligned_rows(
            target,
            tuple(rows),
            alignment_index,
            instrumentation,
        ),
    )
    if instrumentation is not None:
        instrumentation.emitted_row_count += len(family.rows)
    return family


def align_targets_with_index(
    sample: MultiSourceSample,
    alignment_index: AlignmentIndex,
    *,
    instrumentation: AlignmentOperationCounts | None = None,
) -> tuple[AlignedTargetFamily, ...]:
    """Align task rows using a previously validated immutable piece index."""

    if sample.canonical_piece != alignment_index.canonical_piece:
        raise TargetAlignmentError(
            "alignment index canonical piece differs from prepared sample"
        )
    by_task = {target.task_id: target for target in sample.target_bundle}
    task_ids = tuple(item.task_id for item in sample.target_availability)
    return tuple(
        (
            _align_target(
                alignment_index=alignment_index,
                target=by_task[spec.task_id],
                instrumentation=instrumentation,
            )
            if spec.task_id in by_task
            else AlignedTargetFamily(
                task_id=spec.task_id,
                source_entry_count=0,
                rows=(),
            )
        )
        for spec in (target_family_spec(task_id) for task_id in task_ids)
    )


def align_sample_targets(
    piece: CanonicalPiece,
    raw_graph: HeteroData,
    sample: MultiSourceSample,
    *,
    instrumentation: AlignmentOperationCounts | None = None,
) -> tuple[AlignedTargetFamily, ...]:
    """Align every registry task with one sorted immutable piece index."""

    _validate_alignment_inputs(piece, raw_graph, sample)
    alignment_index = _build_alignment_index(
        piece,
        target_alignment_spans=(
            ()
            if sample.attached_target_bundle is None
            else sample.attached_target_bundle.alignment_spans
        ),
        instrumentation=instrumentation,
    )
    return align_targets_with_index(
        sample,
        alignment_index,
        instrumentation=instrumentation,
    )


__all__ = [
    "ALIGNMENT_CONFLICT_DIAGNOSTIC",
    "AlignedTargetFamily",
    "AlignedTargetRow",
    "AlignmentIndex",
    "AlignmentOperationCounts",
    "CandidateTimeIndex",
    "TargetAlignmentError",
    "align_sample_targets",
    "align_targets_with_index",
    "build_alignment_index",
]
