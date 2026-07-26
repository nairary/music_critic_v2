"""Phase 5A sidecar, grouping, and future batching API contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any

import torch

from music_critic.data import (
    CanonicalPiece,
    ProvenanceRecord,
    QualityFlag,
    TargetArray,
)
from music_critic.graph import (
    GraphContractError,
    build_raw_graph,
    graph_fingerprint,
)
from music_critic.tasks.encoding import (
    TARGET_ENCODING_BY_TASK,
)
from music_critic.tasks.ontology import TARGET_FAMILIES, TARGET_FAMILY_BY_ID


class MultiSourceContractError(ValueError):
    """Raised when a sample or grouping record violates the Phase 5A contract."""


_RAW_GRAPH_BINDING_TOKEN = object()


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
class MultiSourceTargetProjection:
    """Target-only canonical projection for audits that need no raw graph."""

    canonical_piece: CanonicalPiece
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
        if (
            self.canonical_piece.dataset_name != self.dataset_id
            or self.canonical_piece.piece_id != self.piece_id
            or self.canonical_piece.source_group_id != self.source_group_id
        ):
            raise MultiSourceContractError(
                "sample identity differs from its canonical piece"
            )
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
        expected_targets = tuple(
            sorted(
                (
                    SampleTarget.from_target_array(target)
                    for target in self.canonical_piece.targets
                ),
                key=lambda target: target.task_id,
            )
        )
        if self.target_bundle != expected_targets:
            raise MultiSourceContractError(
                "sample target bundle differs from its canonical piece"
            )
        if self.diagnostics != self.canonical_piece.quality_flags:
            raise MultiSourceContractError(
                "sample diagnostics differ from its canonical piece"
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
class MultiSourceSample(MultiSourceTargetProjection):
    """Verified canonical/raw binding consumed by the production collator."""

    raw_graph: Any
    raw_graph_fingerprint: str
    _binding_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        MultiSourceTargetProjection.__post_init__(self)
        if self._binding_token is not _RAW_GRAPH_BINDING_TOKEN:
            raise MultiSourceContractError(
                "MultiSourceSample must be created by a verified preparation "
                "factory"
            )
        if (
            not isinstance(self.raw_graph_fingerprint, str)
            or len(self.raw_graph_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.raw_graph_fingerprint
            )
        ):
            raise MultiSourceContractError(
                "sample raw graph fingerprint must be lowercase SHA-256"
            )
        try:
            current = graph_fingerprint(self.raw_graph)
        except GraphContractError as exc:
            raise MultiSourceContractError(
                f"sample raw graph violates the production contract: {exc}"
            ) from exc
        if current != self.raw_graph_fingerprint:
            raise MultiSourceContractError(
                "multisource.raw_graph_binding_mismatch: sample raw graph "
                "differs from its immutable preparation fingerprint"
            )


@dataclass(frozen=True, slots=True)
class TargetDiagnostic:
    """Machine-readable CPU-side diagnostic for one tensorized target row."""

    code: str
    message: str
    source_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise MultiSourceContractError(
                "target diagnostics require non-empty code and message"
            )
        if not self.source_entity_ids or not all(
            isinstance(entity_id, str) and entity_id
            for entity_id in self.source_entity_ids
        ):
            raise MultiSourceContractError(
                "target diagnostics require source entity IDs"
            )


@dataclass(frozen=True, slots=True)
class TargetRowProvenance:
    """Lossless source references retained outside PyG stores for one row."""

    source_entity_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_entity_ids or not all(
            isinstance(value, str) and value for value in self.source_entity_ids
        ):
            raise MultiSourceContractError(
                "target row provenance requires source entity IDs"
            )
        if not all(
            isinstance(value, str) and value
            for values in (self.provenance_ids, self.sources)
            for value in values
        ):
            raise MultiSourceContractError(
                "target row provenance identifiers must be non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class BatchTarget:
    """Versioned tensor/CPU target sidecar, independent of PyG stores."""

    task_id: str
    source_adapter: str
    supervision_context: str
    encoding_registry_version: str
    encoding_kind: str
    model_ready: bool
    deferred_reason: str | None
    supervision_regime: str
    values: Any
    availability_mask: torch.Tensor
    entity_indices: torch.Tensor
    entity_index_mask: torch.Tensor
    entity_node_types: tuple[str | None, ...]
    sample_indices: torch.Tensor
    confidence: torch.Tensor | None
    confidence_mask: torch.Tensor | None
    entry_count: int
    source_entry_count: int
    provenance_cpu: tuple[TargetRowProvenance, ...]
    diagnostics_cpu: tuple[tuple[TargetDiagnostic, ...], ...]

    def __post_init__(self) -> None:
        spec = TARGET_FAMILY_BY_ID.get(self.task_id)
        if spec is None:
            raise MultiSourceContractError("batch task is absent from target ontology")
        encoding = TARGET_ENCODING_BY_TASK[self.task_id]
        expected_encoding = (
            self.encoding_registry_version,
            self.encoding_kind,
            self.model_ready,
            self.deferred_reason,
            self.supervision_regime,
        )
        actual_encoding = (
            encoding.registry_version,
            encoding.encoding_kind,
            encoding.model_ready,
            encoding.deferred_reason,
            encoding.supervision_regime,
        )
        if expected_encoding != actual_encoding:
            raise MultiSourceContractError(
                "batch target encoding metadata differs from registry"
            )
        if (
            self.source_adapter != spec.source_adapter
            or self.supervision_context != spec.supervision_context
        ):
            raise MultiSourceContractError(
                "batch target source semantics differ from ontology"
            )
        if (
            isinstance(self.entry_count, bool)
            or not isinstance(self.entry_count, int)
            or self.entry_count < 0
        ):
            raise MultiSourceContractError(
                "batch target entry_count must be a non-negative integer"
            )
        if (
            isinstance(self.source_entry_count, bool)
            or not isinstance(self.source_entry_count, int)
            or self.source_entry_count < 0
        ):
            raise MultiSourceContractError(
                "batch target source_entry_count must be non-negative"
            )
        dimensions = {}
        for name, value, dtype in (
            ("availability_mask", self.availability_mask, torch.bool),
            ("entity_indices", self.entity_indices, torch.long),
            ("entity_index_mask", self.entity_index_mask, torch.bool),
            ("sample_indices", self.sample_indices, torch.long),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or value.ndim != 1
            ):
                raise MultiSourceContractError(
                    f"batch target {name} must be a rank-one {dtype} tensor"
                )
            dimensions[name] = int(value.shape[0])
        dimensions["entity_node_types"] = len(self.entity_node_types)
        dimensions["provenance_cpu"] = len(self.provenance_cpu)
        dimensions["diagnostics_cpu"] = len(self.diagnostics_cpu)
        if self.confidence is not None:
            if (
                not isinstance(self.confidence, torch.Tensor)
                or self.confidence.dtype != torch.float32
                or self.confidence.ndim != 1
                or not isinstance(self.confidence_mask, torch.Tensor)
                or self.confidence_mask.dtype != torch.bool
                or self.confidence_mask.ndim != 1
            ):
                raise MultiSourceContractError(
                    "batch target confidence requires rank-one float32 values "
                    "and a rank-one bool mask"
                )
            dimensions["confidence"] = int(self.confidence.shape[0])
            dimensions["confidence_mask"] = int(self.confidence_mask.shape[0])
        elif self.confidence_mask is not None:
            raise MultiSourceContractError(
                "batch target confidence_mask requires confidence values"
            )
        if encoding.encoding_kind == "closed_categorical_index":
            if (
                not isinstance(self.values, torch.Tensor)
                or self.values.dtype != torch.long
                or self.values.ndim != 1
            ):
                raise MultiSourceContractError(
                    "closed categorical values must be a rank-one long tensor"
                )
            dimensions["values"] = int(self.values.shape[0])
        elif encoding.encoding_kind == "closed_multilabel":
            if (
                not isinstance(self.values, torch.Tensor)
                or self.values.dtype != torch.bool
                or self.values.ndim != 2
                or self.values.shape[1] != len(encoding.vocabulary or ())
            ):
                raise MultiSourceContractError(
                    "closed multilabel values must be a bool [N, C] tensor"
                )
            dimensions["values"] = int(self.values.shape[0])
        else:
            if not isinstance(self.values, tuple) or not all(
                value is None or isinstance(value, str) for value in self.values
            ):
                raise MultiSourceContractError(
                    "open string values must be a CPU tuple of strings/nulls"
                )
            dimensions["values"] = len(self.values)
        mismatched = {
            name: length
            for name, length in dimensions.items()
            if length != self.entry_count
        }
        if mismatched:
            raise MultiSourceContractError(
                f"batch target leading dimensions differ from entry_count: {mismatched}"
            )
        if self.sample_indices.numel() and self.sample_indices.min().item() < 0:
            raise MultiSourceContractError(
                "batch target sample indices must be non-negative integers"
            )
        if torch.any(self.entity_index_mask & (self.entity_indices < 0)) or torch.any(
            (~self.entity_index_mask) & (self.entity_indices != -1)
        ):
            raise MultiSourceContractError(
                "entity_index=-1 if and only if entity_index_mask is false"
            )
        for index, aligned in enumerate(self.entity_index_mask.tolist()):
            node_type = self.entity_node_types[index]
            if aligned:
                if node_type not in spec.alignment_policy.candidate_node_types:
                    raise MultiSourceContractError(
                        "aligned entity indices require an allowed explicit "
                        "node type"
                    )
            elif node_type is not None:
                raise MultiSourceContractError(
                    "unaligned entities require a null node type"
                )
        unavailable = ~self.availability_mask
        if encoding.encoding_kind == "closed_categorical_index":
            if unavailable.any() and not torch.all(self.values[unavailable] == -1):
                raise MultiSourceContractError(
                    "masked categorical rows must use sentinel -1"
                )
            available_values = self.values[self.availability_mask]
            class_count = len(encoding.vocabulary or ())
            if available_values.numel() and (
                available_values.min().item() < 0
                or available_values.max().item() >= class_count
            ):
                raise MultiSourceContractError(
                    "available categorical value is outside its vocabulary"
                )
        elif encoding.encoding_kind == "closed_multilabel":
            if unavailable.any() and self.values[unavailable].any():
                raise MultiSourceContractError(
                    "masked multilabel rows must use an all-false sentinel row"
                )
        else:
            for available, value in zip(
                self.availability_mask.tolist(), self.values
            ):
                if available != (value is not None):
                    raise MultiSourceContractError(
                        "open string availability must match string/null values"
                    )
        if self.confidence is not None and self.confidence_mask is not None:
            if not torch.isfinite(self.confidence).all():
                raise MultiSourceContractError(
                    "batch target confidence must be finite"
                )
            if self.confidence_mask.any():
                supplied = self.confidence[self.confidence_mask]
                if supplied.min().item() < 0 or supplied.max().item() > 1:
                    raise MultiSourceContractError(
                        "batch target confidence must lie in [0, 1]"
                    )
            if (~self.confidence_mask).any() and not torch.all(
                self.confidence[~self.confidence_mask] == 0
            ):
                raise MultiSourceContractError(
                    "missing confidence uses zero only under a false confidence mask"
                )

    @property
    def supervision_eligibility_mask(self) -> torch.Tensor:
        """Rows eligible for a future task-specific objective."""

        if not self.model_ready:
            return torch.zeros_like(self.availability_mask)
        return self.availability_mask & self.entity_index_mask


@dataclass(frozen=True, slots=True)
class TaskBatchStatistics:
    """Deterministic CPU-side counts for one source-native task."""

    task_id: str
    source_entry_count: int
    target_row_count: int
    aligned_available_count: int
    available_unaligned_row_count: int
    masked_row_count: int
    conflict_row_count: int
    model_encodable_row_count: int
    supervision_eligible_row_count: int
    deferred_open_vocabulary_row_count: int
    node_type_counts: tuple[tuple[str, int], ...]
    model_ready: bool

    def __post_init__(self) -> None:
        if self.task_id not in TARGET_FAMILY_BY_ID:
            raise MultiSourceContractError(
                "task statistics task is absent from target ontology"
            )
        counts = (
            self.source_entry_count,
            self.target_row_count,
            self.aligned_available_count,
            self.available_unaligned_row_count,
            self.masked_row_count,
            self.conflict_row_count,
            self.model_encodable_row_count,
            self.supervision_eligible_row_count,
            self.deferred_open_vocabulary_row_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise MultiSourceContractError(
                "task statistics counts must be non-negative integers"
            )
        if (
            self.aligned_available_count
            + self.available_unaligned_row_count
            + self.masked_row_count
            + self.conflict_row_count
            != self.target_row_count
        ):
            raise MultiSourceContractError(
                "task target rows must partition into aligned, unaligned, "
                "masked, and conflict rows"
            )
        if self.model_ready != TARGET_ENCODING_BY_TASK[self.task_id].model_ready:
            raise MultiSourceContractError(
                "task statistics model readiness differs from encoding registry"
            )
        expected_encodable = self.target_row_count if self.model_ready else 0
        expected_deferred = 0 if self.model_ready else self.target_row_count
        if (
            self.model_encodable_row_count != expected_encodable
            or self.deferred_open_vocabulary_row_count != expected_deferred
            or self.supervision_eligible_row_count
            > self.aligned_available_count
        ):
            raise MultiSourceContractError(
                "task encodable, supervision-eligible, or deferred counts "
                "are inconsistent"
            )
        if tuple(key for key, _ in self.node_type_counts) != tuple(
            sorted(key for key, _ in self.node_type_counts)
        ) or len({key for key, _ in self.node_type_counts}) != len(
            self.node_type_counts
        ):
            raise MultiSourceContractError(
                "task statistics node types must use unique deterministic order"
            )
        if any(
            not isinstance(node_type, str)
            or not node_type
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for node_type, count in self.node_type_counts
        ):
            raise MultiSourceContractError(
                "task statistics node types contain an invalid count"
            )


@dataclass(frozen=True, slots=True)
class BatchStatistics:
    """Deterministic CPU-side mixed-batch statistics."""

    sample_count: int
    graph_count: int
    node_counts: tuple[tuple[str, int], ...]
    edge_counts: tuple[tuple[str, int], ...]
    dataset_counts: tuple[tuple[str, int], ...]
    source_target_entry_count: int
    target_row_count: int
    aligned_available_count: int
    available_unaligned_row_count: int
    masked_row_count: int
    conflict_row_count: int
    node_type_counts: tuple[tuple[str, int], ...]
    task_counts: tuple[TaskBatchStatistics, ...]
    model_ready_task_count: int
    deferred_open_vocabulary_task_count: int
    model_encodable_row_count: int
    supervision_eligible_row_count: int
    deferred_open_vocabulary_row_count: int

    def __post_init__(self) -> None:
        counts = (
            self.sample_count,
            self.graph_count,
            self.source_target_entry_count,
            self.target_row_count,
            self.aligned_available_count,
            self.available_unaligned_row_count,
            self.masked_row_count,
            self.conflict_row_count,
            self.model_ready_task_count,
            self.deferred_open_vocabulary_task_count,
            self.model_encodable_row_count,
            self.supervision_eligible_row_count,
            self.deferred_open_vocabulary_row_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise MultiSourceContractError(
                "batch statistics counts must be non-negative integers"
            )
        expected_tasks = tuple(spec.task_id for spec in TARGET_FAMILIES)
        if tuple(item.task_id for item in self.task_counts) != expected_tasks:
            raise MultiSourceContractError(
                "batch statistics tasks must follow target registry order"
            )
        if self.sample_count != self.graph_count:
            raise MultiSourceContractError(
                "batch statistics require one graph per sample"
            )
        for name, pairs in (
            ("node_counts", self.node_counts),
            ("edge_counts", self.edge_counts),
            ("dataset_counts", self.dataset_counts),
            ("node_type_counts", self.node_type_counts),
        ):
            if tuple(key for key, _ in pairs) != tuple(
                sorted(key for key, _ in pairs)
            ) or len({key for key, _ in pairs}) != len(pairs):
                raise MultiSourceContractError(
                    f"batch statistics {name} must use unique deterministic keys"
                )
            if any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in pairs
            ):
                raise MultiSourceContractError(
                    f"batch statistics {name} contains an invalid count"
                )
        totals = {
            "source_target_entry_count": sum(
                item.source_entry_count for item in self.task_counts
            ),
            "target_row_count": sum(
                item.target_row_count for item in self.task_counts
            ),
            "aligned_available_count": sum(
                item.aligned_available_count for item in self.task_counts
            ),
            "available_unaligned_row_count": sum(
                item.available_unaligned_row_count for item in self.task_counts
            ),
            "masked_row_count": sum(
                item.masked_row_count for item in self.task_counts
            ),
            "conflict_row_count": sum(
                item.conflict_row_count for item in self.task_counts
            ),
            "model_ready_task_count": sum(
                item.model_ready for item in self.task_counts
            ),
            "deferred_open_vocabulary_task_count": sum(
                not item.model_ready for item in self.task_counts
            ),
            "model_encodable_row_count": sum(
                item.model_encodable_row_count for item in self.task_counts
            ),
            "supervision_eligible_row_count": sum(
                item.supervision_eligible_row_count
                for item in self.task_counts
            ),
            "deferred_open_vocabulary_row_count": sum(
                item.deferred_open_vocabulary_row_count
                for item in self.task_counts
            ),
        }
        for name, expected in totals.items():
            if getattr(self, name) != expected:
                raise MultiSourceContractError(
                    f"batch statistics {name} differs from per-task totals"
                )


@dataclass(frozen=True, slots=True)
class MultiSourceBatch:
    """Validated production Phase 5B.1 raw graph plus target sidecars."""

    raw_graph_batch: Any
    target_batches: tuple[BatchTarget, ...]
    dataset_ids: tuple[str, ...]
    piece_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    lineage_group_ids: tuple[str, ...]
    diagnostics_cpu: tuple[tuple[QualityFlag, ...], ...]
    statistics: BatchStatistics

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
        expected_tasks = tuple(spec.task_id for spec in TARGET_FAMILIES)
        if tasks != expected_tasks:
            raise MultiSourceContractError(
                "batch target sidecars must contain every task in registry order"
            )
        sample_count = len(self.piece_ids)
        from music_critic.graph.validation import (
            GraphContractError,
            validate_raw_graph_batch,
        )

        try:
            validate_raw_graph_batch(
                self.raw_graph_batch,
                sample_count=sample_count,
            )
        except GraphContractError as exc:
            raise MultiSourceContractError(
                f"batch graph violates the exact raw-only contract: {exc}"
            ) from exc
        if (
            self.statistics.sample_count != sample_count
            or self.statistics.graph_count != sample_count
        ):
            raise MultiSourceContractError(
                "batch statistics sample/graph count differs from batch metadata"
            )
        actual_node_counts = tuple(
            sorted(
                (
                    node_type,
                    int(self.raw_graph_batch[node_type].num_nodes),
                )
                for node_type in self.raw_graph_batch.node_types
            )
        )
        actual_edge_counts = tuple(
            sorted(
                (
                    "|".join(edge_type),
                    int(self.raw_graph_batch[edge_type].edge_index.shape[1]),
                )
                for edge_type in self.raw_graph_batch.edge_types
            )
        )
        actual_dataset_counts = tuple(sorted(Counter(self.dataset_ids).items()))
        if (
            self.statistics.node_counts != actual_node_counts
            or self.statistics.edge_counts != actual_edge_counts
            or self.statistics.dataset_counts != actual_dataset_counts
        ):
            raise MultiSourceContractError(
                "batch statistics graph or dataset counts differ from batch data"
            )
        for target, task_statistics in zip(
            self.target_batches, self.statistics.task_counts
        ):
            if target.sample_indices.numel() and (
                target.sample_indices.max().item() >= sample_count
            ):
                raise MultiSourceContractError(
                    "batch target sample index is outside the batch"
                )
            for row, aligned in enumerate(target.entity_index_mask.tolist()):
                if not aligned:
                    continue
                node_type = target.entity_node_types[row]
                assert node_type is not None
                entity_index = int(target.entity_indices[row].item())
                sample_index = int(target.sample_indices[row].item())
                if (
                    entity_index >= self.raw_graph_batch[node_type].num_nodes
                    or int(
                        self.raw_graph_batch[node_type].batch[entity_index].item()
                    )
                    != sample_index
                ):
                    raise MultiSourceContractError(
                        "batch target global entity index has the wrong sample offset"
                    )
            aligned = target.availability_mask & target.entity_index_mask
            unaligned = target.availability_mask & ~target.entity_index_mask
            node_type_counts = tuple(
                sorted(
                    Counter(
                        node_type
                        for node_type, has_index in zip(
                            target.entity_node_types,
                            target.entity_index_mask.tolist(),
                        )
                        if has_index and node_type is not None
                    ).items()
                )
            )
            conflict_flags = tuple(
                any(
                    diagnostic.code == "multisource.alignment_conflict"
                    for diagnostic in diagnostics
                )
                for diagnostics in target.diagnostics_cpu
            )
            conflict_count = sum(conflict_flags)
            masked_count = sum(
                not available and not conflict
                for available, conflict in zip(
                    target.availability_mask.tolist(), conflict_flags
                )
            )
            expected_task_statistics = (
                target.task_id,
                target.source_entry_count,
                target.entry_count,
                int(aligned.sum().item()),
                int(unaligned.sum().item()),
                masked_count,
                conflict_count,
                target.entry_count if target.model_ready else 0,
                int(target.supervision_eligibility_mask.sum().item()),
                0 if target.model_ready else target.entry_count,
                node_type_counts,
                target.model_ready,
            )
            actual_task_statistics = (
                task_statistics.task_id,
                task_statistics.source_entry_count,
                task_statistics.target_row_count,
                task_statistics.aligned_available_count,
                task_statistics.available_unaligned_row_count,
                task_statistics.masked_row_count,
                task_statistics.conflict_row_count,
                task_statistics.model_encodable_row_count,
                task_statistics.supervision_eligible_row_count,
                task_statistics.deferred_open_vocabulary_row_count,
                task_statistics.node_type_counts,
                task_statistics.model_ready,
            )
            if actual_task_statistics != expected_task_statistics:
                raise MultiSourceContractError(
                    "batch task statistics differ from tensor sidecar"
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
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise MultiSourceContractError(
                "dataset sampling weight requires a non-empty string dataset ID"
            )
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise MultiSourceContractError(
                "dataset sampling weight must be a finite positive number"
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


def project_multisource_targets(
    piece: CanonicalPiece,
    *,
    lineage_group_id: str | None = None,
) -> MultiSourceTargetProjection:
    """Project target-only audit evidence without constructing a raw graph.

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
    return MultiSourceTargetProjection(
        canonical_piece=piece,
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        source_group_id=piece.source_group_id,
        lineage_group_id=resolved_lineage,
        target_bundle=targets,
        target_availability=_availability(targets),
        target_provenance_sidecar=_target_provenance(piece, targets),
        diagnostics=piece.quality_flags,
    )


def _sample_from_projection(
    projection: MultiSourceTargetProjection,
    *,
    raw_graph: Any,
    raw_graph_fingerprint: str,
) -> MultiSourceSample:
    return MultiSourceSample(
        canonical_piece=projection.canonical_piece,
        dataset_id=projection.dataset_id,
        piece_id=projection.piece_id,
        source_group_id=projection.source_group_id,
        lineage_group_id=projection.lineage_group_id,
        target_bundle=projection.target_bundle,
        target_availability=projection.target_availability,
        target_provenance_sidecar=projection.target_provenance_sidecar,
        diagnostics=projection.diagnostics,
        raw_graph=raw_graph,
        raw_graph_fingerprint=raw_graph_fingerprint,
        _binding_token=_RAW_GRAPH_BINDING_TOKEN,
    )


def prepare_multisource_sample(
    piece: CanonicalPiece,
    *,
    lineage_group_id: str | None = None,
) -> MultiSourceSample:
    """Build and bind the exact Phase 3A raw graph for production collation."""

    projection = project_multisource_targets(
        piece,
        lineage_group_id=lineage_group_id,
    )
    raw_graph = build_raw_graph(piece)
    return _sample_from_projection(
        projection,
        raw_graph=raw_graph,
        raw_graph_fingerprint=graph_fingerprint(raw_graph),
    )


def build_multisource_sample(
    piece: CanonicalPiece,
    raw_graph: Any,
    *,
    lineage_group_id: str | None = None,
) -> MultiSourceSample:
    """Verify an externally built graph against a fresh Phase 3A projection.

    Prefer :func:`prepare_multisource_sample` in production. This compatibility
    factory has no verification bypass: it builds the expected raw graph and
    compares complete deterministic graph fingerprints before binding.
    """

    projection = project_multisource_targets(
        piece,
        lineage_group_id=lineage_group_id,
    )
    expected_graph = build_raw_graph(piece)
    expected_fingerprint = graph_fingerprint(expected_graph)
    try:
        actual_fingerprint = graph_fingerprint(raw_graph)
    except GraphContractError as exc:
        raise MultiSourceContractError(
            f"external raw graph violates the production contract: {exc}"
        ) from exc
    if actual_fingerprint != expected_fingerprint:
        raise MultiSourceContractError(
            "multisource.raw_graph_binding_mismatch: external raw graph is not "
            "the exact Phase 3A projection of the canonical piece"
        )
    return _sample_from_projection(
        projection,
        raw_graph=raw_graph,
        raw_graph_fingerprint=actual_fingerprint,
    )


def validate_group_assignments(
    assignments: tuple[GroupAssignment, ...],
) -> None:
    """Reject duplicate pieces and split-crossing atomic group components."""

    if len(assignments) != len(set(assignments)):
        raise MultiSourceContractError(
            "duplicate group assignments are rejected; callers must deduplicate explicitly"
        )
    piece_assignments: set[tuple[str, str]] = set()
    for assignment in assignments:
        piece_key = (assignment.dataset_id, assignment.piece_id)
        if piece_key in piece_assignments:
            raise MultiSourceContractError(
                f"piece {piece_key!r} must have exactly one GroupAssignment"
            )
        piece_assignments.add(piece_key)

    for component in _atomic_group_components(assignments):
        splits = tuple(
            sorted(
                {
                    assignment.split
                    for assignment in component
                    if assignment.split is not None
                }
            )
        )
        if len(splits) > 1:
            identity = tuple(_assignment_key(assignment) for assignment in component)
            raise MultiSourceContractError(
                f"atomic source/lineage component {identity!r} crosses splits "
                f"{splits!r}"
            )


def _atomic_group_components(
    assignments: tuple[GroupAssignment, ...],
) -> tuple[tuple[GroupAssignment, ...], ...]:
    """Build deterministic transitive components over source and lineage IDs."""

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

    return tuple(
        sorted(
            (
                tuple(sorted(component, key=_assignment_key))
                for component in components.values()
            ),
            key=lambda component: tuple(
                _assignment_key(assignment) for assignment in component
            ),
        )
    )


def _assignment_key(
    assignment: GroupAssignment,
) -> tuple[str, str, str, str, str]:
    return (
        assignment.dataset_id,
        assignment.piece_id,
        assignment.source_group_id,
        assignment.lineage_group_id,
        assignment.split or "",
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
    blocks = _atomic_group_components(assignments)

    def block_key(
        block: tuple[GroupAssignment, ...],
    ) -> tuple[str, tuple[tuple[str, str, str, str, str], ...]]:
        identity = tuple(_assignment_key(assignment) for assignment in block)
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
    "BatchStatistics",
    "BatchTarget",
    "DatasetSamplingWeight",
    "GroupAssignment",
    "MultiSourceBatch",
    "MultiSourceContractError",
    "MultiSourceSample",
    "MultiSourceTargetProjection",
    "SampleTarget",
    "TargetDiagnostic",
    "TargetRowProvenance",
    "TaskBatchStatistics",
    "TaskAvailability",
    "build_multisource_sample",
    "deterministic_group_order",
    "prepare_multisource_sample",
    "project_multisource_targets",
    "validate_group_assignments",
]
