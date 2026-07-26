"""Phase 5A sidecar, grouping, and future batching API contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any

from music_critic.data import (
    CanonicalPiece,
    ProvenanceRecord,
    QualityFlag,
    TargetArray,
)
from music_critic.tasks.ontology import TARGET_FAMILIES, TARGET_FAMILY_BY_ID


class MultiSourceContractError(ValueError):
    """Raised when a sample or grouping record violates the Phase 5A contract."""


@dataclass(frozen=True, slots=True)
class SampleTarget:
    """One source-native target family stored outside the raw graph."""

    task_id: str
    annotation_view_id: str | None
    alignment_type: str
    entity_ids: tuple[str, ...]
    values: tuple[object | None, ...]
    availability_mask: tuple[bool, ...]
    confidence: tuple[float | None, ...]
    source: tuple[str | None, ...]
    provenance_ids: tuple[str | None, ...]

    @classmethod
    def from_target_array(cls, target: TargetArray) -> SampleTarget:
        spec = TARGET_FAMILY_BY_ID.get(target.task)
        if spec is None:
            raise MultiSourceContractError(
                f"task {target.task!r} is absent from target ontology"
            )
        if target.value_type != spec.value_type:
            raise MultiSourceContractError(
                f"task {target.task!r} value type differs from registry"
            )
        if target.class_labels != spec.vocabulary:
            raise MultiSourceContractError(
                f"task {target.task!r} vocabulary differs from registry"
            )
        if target.annotation_view_id != spec.annotation_view_id:
            raise MultiSourceContractError(
                f"task {target.task!r} annotation view differs from registry"
            )
        if target.alignment_type != spec.source_alignment_type:
            raise MultiSourceContractError(
                f"task {target.task!r} alignment type differs from registry"
            )
        return cls(
            task_id=target.task,
            annotation_view_id=target.annotation_view_id,
            alignment_type=target.alignment_type,
            entity_ids=target.entity_ids,
            values=target.values,
            availability_mask=target.mask,
            confidence=target.confidence,
            source=target.source,
            provenance_ids=target.provenance,
        )

    def __post_init__(self) -> None:
        spec = TARGET_FAMILY_BY_ID.get(self.task_id)
        if spec is None:
            raise MultiSourceContractError("sample task is absent from target ontology")
        if self.annotation_view_id != spec.annotation_view_id:
            raise MultiSourceContractError(
                "sample target annotation view differs from registry"
            )
        lengths = {
            len(self.entity_ids),
            len(self.values),
            len(self.availability_mask),
            len(self.confidence),
            len(self.source),
            len(self.provenance_ids),
        }
        if len(lengths) != 1:
            raise MultiSourceContractError("sample target arrays must have equal length")
        for index, available in enumerate(self.availability_mask):
            values = (
                self.values[index],
                self.confidence[index],
                self.source[index],
                self.provenance_ids[index],
            )
            if not available and any(value is not None for value in values):
                raise MultiSourceContractError(
                    "unavailable target entries must remain entirely null"
                )
            if available and (
                self.values[index] is None
                or self.source[index] is None
                or self.provenance_ids[index] is None
            ):
                raise MultiSourceContractError(
                    "available target entries require value, source, and provenance"
                )


@dataclass(frozen=True, slots=True)
class TaskAvailability:
    """Per-sample distinction between an absent task and masked entries."""

    task_id: str
    family_present: bool
    available_count: int
    masked_count: int


@dataclass(frozen=True, slots=True)
class MultiSourceSample:
    """Public Phase 5B sample shape; construction does not batch or tensorize."""

    raw_graph: Any
    dataset_id: str
    piece_id: str
    source_group_id: str
    lineage_group_id: str
    target_bundle: tuple[SampleTarget, ...]
    target_availability: tuple[TaskAvailability, ...]
    target_provenance_sidecar: tuple[ProvenanceRecord, ...]
    diagnostics: tuple[QualityFlag, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.source_group_id,
                self.lineage_group_id,
            )
        ):
            raise MultiSourceContractError("sample identity fields must be non-empty")
        tasks = tuple(target.task_id for target in self.target_bundle)
        if tasks != tuple(sorted(tasks)) or len(tasks) != len(set(tasks)):
            raise MultiSourceContractError(
                "sample target bundle must be uniquely sorted by stable task ID"
            )
        expected_availability = _availability(self.target_bundle)
        if self.target_availability != expected_availability:
            raise MultiSourceContractError(
                "sample target availability differs from target bundle"
            )
        referenced = {
            provenance_id
            for target in self.target_bundle
            for provenance_id in target.provenance_ids
            if provenance_id is not None
        }
        sidecar_ids = {
            record.provenance_id for record in self.target_provenance_sidecar
        }
        if not referenced <= sidecar_ids:
            raise MultiSourceContractError(
                "sample target provenance references are absent from sidecar"
            )


@dataclass(frozen=True, slots=True)
class BatchTarget:
    """Future Phase 5B tensor sidecar, deliberately independent of PyG stores."""

    task_id: str
    values: Any
    availability_mask: Any
    entity_indices: Any
    sample_indices: Any
    confidence: Any | None
    entry_count: int
    provenance_cpu: tuple[object, ...]
    diagnostics_cpu: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.task_id not in TARGET_FAMILY_BY_ID:
            raise MultiSourceContractError("batch task is absent from target ontology")
        if self.entry_count < 0:
            raise MultiSourceContractError("batch target entry_count cannot be negative")
        if self.entry_count == 0 and (
            self.provenance_cpu or self.diagnostics_cpu
        ):
            raise MultiSourceContractError(
                "a completely empty task family has no per-entry CPU metadata"
            )


@dataclass(frozen=True, slots=True)
class MultiSourceBatch:
    """Future Phase 5B batch shape; no collator is implemented in Phase 5A."""

    raw_graph_batch: Any
    target_batches: tuple[BatchTarget, ...]
    dataset_ids: tuple[str, ...]
    piece_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    lineage_group_ids: tuple[str, ...]
    diagnostics_cpu: tuple[tuple[QualityFlag, ...], ...]

    def __post_init__(self) -> None:
        sample_lengths = {
            len(self.dataset_ids),
            len(self.piece_ids),
            len(self.source_group_ids),
            len(self.lineage_group_ids),
            len(self.diagnostics_cpu),
        }
        if len(sample_lengths) != 1:
            raise MultiSourceContractError("batch sample metadata lengths differ")
        tasks = tuple(target.task_id for target in self.target_batches)
        if tasks != tuple(sorted(tasks)) or len(tasks) != len(set(tasks)):
            raise MultiSourceContractError(
                "batch target sidecars must be uniquely sorted by task ID"
            )


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """Group-level split evidence used before any future sampler."""

    dataset_id: str
    piece_id: str
    source_group_id: str
    lineage_group_id: str
    split: str | None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.source_group_id,
                self.lineage_group_id,
            )
        ):
            raise MultiSourceContractError(
                "group assignment identity fields must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class DatasetSamplingWeight:
    """Future per-dataset sampling weight; Phase 5A performs no sampling."""

    dataset_id: str
    weight: float

    def __post_init__(self) -> None:
        if not self.dataset_id or not math.isfinite(self.weight) or self.weight <= 0:
            raise MultiSourceContractError(
                "dataset sampling weights require an ID and positive weight"
            )


def _availability(
    targets: tuple[SampleTarget, ...],
) -> tuple[TaskAvailability, ...]:
    by_task = {target.task_id: target for target in targets}
    return tuple(
        TaskAvailability(
            task_id=spec.task_id,
            family_present=spec.task_id in by_task,
            available_count=(
                sum(by_task[spec.task_id].availability_mask)
                if spec.task_id in by_task
                else 0
            ),
            masked_count=(
                len(by_task[spec.task_id].availability_mask)
                - sum(by_task[spec.task_id].availability_mask)
                if spec.task_id in by_task
                else 0
            ),
        )
        for spec in TARGET_FAMILIES
    )


def _lineage_from_piece(piece: CanonicalPiece) -> str:
    if piece.dataset_name == "pop909_cl":
        candidates = {
            value
            for record in piece.provenance
            for key, value in record.details
            if key == "lineage_group_id" and isinstance(value, str) and value
        }
        if len(candidates) != 1:
            raise MultiSourceContractError(
                "POP909-CL sample requires exactly one lineage_group_id"
            )
        return next(iter(candidates))
    return piece.source_group_id


def _target_provenance(
    piece: CanonicalPiece,
    targets: tuple[SampleTarget, ...],
) -> tuple[ProvenanceRecord, ...]:
    by_id = {record.provenance_id: record for record in piece.provenance}
    selected = {
        provenance_id
        for target in targets
        for provenance_id in target.provenance_ids
        if provenance_id is not None
    }
    pending = list(selected)
    while pending:
        identifier = pending.pop()
        record = by_id.get(identifier)
        if record is None:
            raise MultiSourceContractError(
                f"target provenance {identifier!r} is absent from canonical piece"
            )
        for parent in record.parents:
            if parent not in selected:
                selected.add(parent)
                pending.append(parent)
    return tuple(
        record for record in piece.provenance if record.provenance_id in selected
    )


def build_multisource_sample(
    piece: CanonicalPiece,
    raw_graph: Any,
    *,
    lineage_group_id: str | None = None,
) -> MultiSourceSample:
    """Project canonical targets into an immutable sidecar around an opaque graph."""

    targets = tuple(
        sorted(
            (SampleTarget.from_target_array(target) for target in piece.targets),
            key=lambda target: target.task_id,
        )
    )
    return MultiSourceSample(
        raw_graph=raw_graph,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        source_group_id=piece.source_group_id,
        lineage_group_id=lineage_group_id or _lineage_from_piece(piece),
        target_bundle=targets,
        target_availability=_availability(targets),
        target_provenance_sidecar=_target_provenance(piece, targets),
        diagnostics=piece.quality_flags,
    )


def validate_group_assignments(
    assignments: tuple[GroupAssignment, ...],
) -> None:
    """Reject source or lineage groups assigned to more than one split."""

    for attribute in ("source_group_id", "lineage_group_id"):
        observed: dict[str, set[str]] = {}
        for assignment in assignments:
            if assignment.split is not None:
                observed.setdefault(getattr(assignment, attribute), set()).add(
                    assignment.split
                )
        conflicts = {
            group_id: tuple(sorted(splits))
            for group_id, splits in observed.items()
            if len(splits) > 1
        }
        if conflicts:
            raise MultiSourceContractError(
                f"{attribute} values cross splits: {conflicts}"
            )


def deterministic_group_order(
    assignments: tuple[GroupAssignment, ...],
    *,
    seed: int,
) -> tuple[GroupAssignment, ...]:
    """Return a stable seed-dependent order without assigning or changing splits."""

    def key(assignment: GroupAssignment) -> tuple[str, str, str]:
        identity = "\0".join(
            (
                str(seed),
                assignment.dataset_id,
                assignment.source_group_id,
                assignment.lineage_group_id,
                assignment.piece_id,
            )
        )
        return (
            sha256(identity.encode("utf-8")).hexdigest(),
            assignment.dataset_id,
            assignment.piece_id,
        )

    return tuple(sorted(assignments, key=key))


__all__ = [
    "BatchTarget",
    "DatasetSamplingWeight",
    "GroupAssignment",
    "MultiSourceBatch",
    "MultiSourceContractError",
    "MultiSourceSample",
    "SampleTarget",
    "TaskAvailability",
    "build_multisource_sample",
    "deterministic_group_order",
    "validate_group_assignments",
]
