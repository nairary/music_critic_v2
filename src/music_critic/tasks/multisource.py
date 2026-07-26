"""Phase 5A sidecar, grouping, and future batching API contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
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
        if self.alignment_type != spec.source_alignment_type:
            raise MultiSourceContractError(
                "sample target alignment type differs from registry"
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
        if len(self.entity_ids) != len(set(self.entity_ids)) or not all(
            isinstance(entity_id, str) and entity_id for entity_id in self.entity_ids
        ):
            raise MultiSourceContractError(
                "sample target entity IDs must be non-empty and unique"
            )
        if not all(
            isinstance(available, bool) for available in self.availability_mask
        ):
            raise MultiSourceContractError(
                "sample target availability mask must contain booleans"
            )
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
                or not isinstance(self.source[index], str)
                or not self.source[index]
                or not isinstance(self.provenance_ids[index], str)
                or not self.provenance_ids[index]
            ):
                raise MultiSourceContractError(
                    "available target entries require value and non-empty string "
                    "source/provenance"
                )
            if available:
                _validate_source_value(spec, self.values[index])
                confidence = self.confidence[index]
                if confidence is not None and (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                ):
                    raise MultiSourceContractError(
                        "available target confidence must be null or finite in [0, 1]"
                    )


@dataclass(frozen=True, slots=True)
class TaskAvailability:
    """Per-sample distinction between an absent task and masked entries."""

    task_id: str
    family_present: bool
    available_count: int
    masked_count: int

    def __post_init__(self) -> None:
        if self.task_id not in TARGET_FAMILY_BY_ID:
            raise MultiSourceContractError(
                "task availability is absent from target ontology"
            )
        if not isinstance(self.family_present, bool):
            raise MultiSourceContractError(
                "task availability family_present must be boolean"
            )
        if (
            isinstance(self.available_count, bool)
            or isinstance(self.masked_count, bool)
            or not isinstance(self.available_count, int)
            or not isinstance(self.masked_count, int)
            or self.available_count < 0
            or self.masked_count < 0
        ):
            raise MultiSourceContractError(
                "task availability counts must be non-negative integers"
            )
        if not self.family_present and (
            self.available_count != 0 or self.masked_count != 0
        ):
            raise MultiSourceContractError(
                "an absent target family must have zero counts"
            )


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
    entity_index_mask: Any
    entity_node_types: tuple[str | None, ...]
    sample_indices: Any
    confidence: Any | None
    entry_count: int
    provenance_cpu: tuple[object, ...]
    diagnostics_cpu: tuple[object, ...]

    def __post_init__(self) -> None:
        spec = TARGET_FAMILY_BY_ID.get(self.task_id)
        if spec is None:
            raise MultiSourceContractError("batch task is absent from target ontology")
        if (
            isinstance(self.entry_count, bool)
            or not isinstance(self.entry_count, int)
            or self.entry_count < 0
        ):
            raise MultiSourceContractError(
                "batch target entry_count must be a non-negative integer"
            )
        dimensions = {
            name: _leading_dimension(name, value)
            for name, value in (
                ("values", self.values),
                ("availability_mask", self.availability_mask),
                ("entity_indices", self.entity_indices),
                ("entity_index_mask", self.entity_index_mask),
                ("sample_indices", self.sample_indices),
            )
        }
        dimensions["entity_node_types"] = len(self.entity_node_types)
        dimensions["provenance_cpu"] = len(self.provenance_cpu)
        dimensions["diagnostics_cpu"] = len(self.diagnostics_cpu)
        if self.confidence is not None:
            dimensions["confidence"] = _leading_dimension(
                "confidence", self.confidence
            )
        mismatched = {
            name: length
            for name, length in dimensions.items()
            if length != self.entry_count
        }
        if mismatched:
            raise MultiSourceContractError(
                f"batch target leading dimensions differ from entry_count: {mismatched}"
            )
        sample_indices = _flat_sequence("sample_indices", self.sample_indices)
        if sample_indices is not None and any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in sample_indices
        ):
            raise MultiSourceContractError(
                "batch target sample indices must be non-negative integers"
            )
        availability = _flat_sequence(
            "availability_mask", self.availability_mask
        )
        if availability is not None and not all(
            isinstance(value, bool) for value in availability
        ):
            raise MultiSourceContractError(
                "batch target availability mask must contain booleans"
            )
        entity_mask = _flat_sequence("entity_index_mask", self.entity_index_mask)
        if entity_mask is not None and not all(
            isinstance(value, bool) for value in entity_mask
        ):
            raise MultiSourceContractError(
                "batch target entity-index mask must contain booleans"
            )
        entity_indices = _flat_sequence("entity_indices", self.entity_indices)
        if entity_mask is not None and entity_indices is not None:
            for index, aligned in enumerate(entity_mask):
                entity_index = entity_indices[index]
                node_type = self.entity_node_types[index]
                if aligned:
                    if (
                        isinstance(entity_index, bool)
                        or not isinstance(entity_index, int)
                        or entity_index < 0
                        or node_type not in spec.alignment_policy.candidate_node_types
                    ):
                        raise MultiSourceContractError(
                            "aligned entity indices require a non-negative index "
                            "and an allowed explicit node type"
                        )
                elif entity_index != -1 or node_type is not None:
                    raise MultiSourceContractError(
                        "unaligned entities require index -1 and null node type"
                    )
        if self.entry_count == 0 and self.confidence is not None and (
            _leading_dimension("confidence", self.confidence) != 0
        ):
            raise MultiSourceContractError(
                "an empty family requires empty optional confidence"
            )
        if self.entry_count == 0 and (
            self.provenance_cpu or self.diagnostics_cpu or self.entity_node_types
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
        if len(sample_lengths) != 1 or next(iter(sample_lengths), 0) == 0:
            raise MultiSourceContractError(
                "batch sample metadata lengths must match and be non-zero"
            )
        if not all(
            isinstance(value, str) and value
            for collection in (
                self.dataset_ids,
                self.piece_ids,
                self.source_group_ids,
                self.lineage_group_ids,
            )
            for value in collection
        ):
            raise MultiSourceContractError(
                "batch sample identity strings must be non-empty"
            )
        tasks = tuple(target.task_id for target in self.target_batches)
        if tasks != tuple(sorted(tasks)) or len(tasks) != len(set(tasks)):
            raise MultiSourceContractError(
                "batch target sidecars must be uniquely sorted by task ID"
            )
        if getattr(self.raw_graph_batch, "raw_only", None) is not True:
            raise MultiSourceContractError(
                "batch graph must carry the raw_only=True contract marker"
            )
        forbidden_sidecars = _raw_graph_sidecar_fields(self.raw_graph_batch)
        if forbidden_sidecars:
            raise MultiSourceContractError(
                "batch graph must remain raw-only; target/provenance sidecar "
                f"fields found: {sorted(forbidden_sidecars)}"
            )
        sample_count = len(self.piece_ids)
        for target in self.target_batches:
            sample_indices = _flat_sequence(
                "sample_indices", target.sample_indices
            )
            if sample_indices is not None and any(
                value >= sample_count for value in sample_indices
            ):
                raise MultiSourceContractError(
                    "batch target sample index is outside the batch"
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
        if self.split is not None and (
            not isinstance(self.split, str) or not self.split
        ):
            raise MultiSourceContractError(
                "group assignment split must be null or a non-empty string"
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


def _validate_source_value(spec: Any, value: object) -> None:
    if spec.value_type == "categorical":
        if not isinstance(value, str):
            raise MultiSourceContractError(
                "available categorical sample target values must be strings"
            )
        if spec.vocabulary is not None and value not in spec.vocabulary:
            raise MultiSourceContractError(
                "available categorical sample target value is outside vocabulary"
            )
        return
    if (
        not isinstance(value, tuple)
        or not all(isinstance(item, str) for item in value)
        or len(value) != len(set(value))
    ):
        raise MultiSourceContractError(
            "available multi-label sample target values must be unique string tuples"
        )
    expected = tuple(label for label in spec.vocabulary if label in value)
    if value != expected:
        raise MultiSourceContractError(
            "available multi-label values must follow canonical vocabulary order"
        )


def _leading_dimension(name: str, value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise MultiSourceContractError(f"{name} must have a leading dimension")
        return int(shape[0])
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise MultiSourceContractError(
            f"{name} must expose a tensor/container leading dimension"
        )
    return len(value)


def _flat_sequence(name: str, value: Any) -> tuple[Any, ...] | None:
    converted = value.tolist() if callable(getattr(value, "tolist", None)) else value
    if not isinstance(converted, Sequence) or isinstance(
        converted, (str, bytes)
    ):
        return None
    if any(
        isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        for item in converted
    ):
        raise MultiSourceContractError(f"{name} must be rank one")
    return tuple(converted)


_RAW_GRAPH_SIDECAR_FIELDS = frozenset(
    {
        "annotations",
        "availability_mask",
        "confidence",
        "dataset_ids",
        "entity_index_mask",
        "entity_indices",
        "entity_node_types",
        "lineage_group_ids",
        "piece_ids",
        "provenance",
        "provenance_cpu",
        "source_group_ids",
        "split",
        "target_availability",
        "target_batches",
        "target_bundle",
        "target_provenance_sidecar",
        "targets",
    }
)


def _raw_graph_sidecar_fields(raw_graph_batch: Any) -> frozenset[str]:
    """Return forbidden supervisory keys found anywhere in a graph batch."""

    stores: list[Any] = [raw_graph_batch]
    for attribute in ("_global_store", "node_stores", "edge_stores"):
        value = getattr(raw_graph_batch, attribute, None)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            stores.extend(value)
        else:
            stores.append(value)

    keys: set[str] = set()
    for store in stores:
        key_method = getattr(store, "keys", None)
        if callable(key_method):
            keys.update(key for key in key_method() if isinstance(key, str))
        namespace = getattr(store, "__dict__", None)
        if isinstance(namespace, dict):
            keys.update(key for key in namespace if isinstance(key, str))
    return frozenset(keys & _RAW_GRAPH_SIDECAR_FIELDS)


def _authoritative_lineage(piece: CanonicalPiece) -> str | None:
    raw_candidates = [
        value
        for record in piece.provenance
        for key, value in record.details
        if key == "lineage_group_id"
    ]
    if any(
        not isinstance(value, str) or not value for value in raw_candidates
    ):
        raise MultiSourceContractError(
            "canonical provenance lineage_group_id must be a non-empty string"
        )
    candidates = set(raw_candidates)
    if len(candidates) > 1:
        raise MultiSourceContractError(
            "canonical provenance contains conflicting lineage_group_id values"
        )
    return next(iter(candidates), None)


def _resolved_lineage(piece: CanonicalPiece) -> str:
    return _authoritative_lineage(piece) or piece.source_group_id


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
    """Project targets while treating ``lineage_group_id`` as an assertion.

    Canonical provenance wins when it supplies lineage. Sources without
    authoritative lineage fall back to ``piece.source_group_id``. An explicit
    argument must be non-empty and exactly match the resolved value.
    """

    resolved_lineage = _resolved_lineage(piece)
    if lineage_group_id is not None:
        if not isinstance(lineage_group_id, str) or not lineage_group_id:
            raise MultiSourceContractError(
                "lineage_group_id assertion must be a non-empty string"
            )
        if lineage_group_id != resolved_lineage:
            raise MultiSourceContractError(
                "lineage_group_id assertion differs from authoritative lineage "
                "or source-group fallback"
            )
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
        lineage_group_id=resolved_lineage,
        target_bundle=targets,
        target_availability=_availability(targets),
        target_provenance_sidecar=_target_provenance(piece, targets),
        diagnostics=piece.quality_flags,
    )


def validate_group_assignments(
    assignments: tuple[GroupAssignment, ...],
) -> None:
    """Reject duplicates, piece identity conflicts, and split-crossing groups."""

    if len(assignments) != len(set(assignments)):
        raise MultiSourceContractError(
            "duplicate group assignments are rejected; callers must deduplicate explicitly"
        )
    piece_groups: dict[tuple[str, str], tuple[str, str]] = {}
    for assignment in assignments:
        piece_key = (assignment.dataset_id, assignment.piece_id)
        grouping = (assignment.source_group_id, assignment.lineage_group_id)
        previous = piece_groups.setdefault(piece_key, grouping)
        if previous != grouping:
            raise MultiSourceContractError(
                f"piece {piece_key!r} has conflicting grouping IDs"
            )

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
    """Order transitive source/lineage components as indivisible blocks."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MultiSourceContractError("deterministic group seed must be an integer")
    validate_group_assignments(assignments)
    if not assignments:
        return ()

    parent = list(range(len(assignments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_source: dict[str, int] = {}
    by_lineage: dict[str, int] = {}
    for index, assignment in enumerate(assignments):
        for grouping, observed in (
            (assignment.source_group_id, by_source),
            (assignment.lineage_group_id, by_lineage),
        ):
            if grouping in observed:
                union(index, observed[grouping])
            else:
                observed[grouping] = index

    components: dict[int, list[GroupAssignment]] = {}
    for index, assignment in enumerate(assignments):
        components.setdefault(find(index), []).append(assignment)

    def assignment_key(
        assignment: GroupAssignment,
    ) -> tuple[str, str, str, str, str]:
        return (
            assignment.dataset_id,
            assignment.piece_id,
            assignment.source_group_id,
            assignment.lineage_group_id,
            assignment.split or "",
        )

    blocks = [
        tuple(sorted(component, key=assignment_key))
        for component in components.values()
    ]

    def block_key(
        block: tuple[GroupAssignment, ...],
    ) -> tuple[str, tuple[tuple[str, str, str, str, str], ...]]:
        identity = tuple(assignment_key(assignment) for assignment in block)
        payload = json.dumps(
            {"seed": seed, "group": identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest(), identity

    return tuple(
        assignment
        for block in sorted(blocks, key=block_key)
        for assignment in block
    )


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
