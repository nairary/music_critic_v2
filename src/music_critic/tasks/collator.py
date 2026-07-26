"""Production PyG graph collation and target tensorization for Phase 5B.1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import torch
from torch_geometric.data import Batch

from music_critic.graph import (
    GraphContractError,
    graph_fingerprint,
    validate_raw_graph_batch,
)
from music_critic.graph.relations import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES
from music_critic.tasks.alignment import (
    ALIGNMENT_CONFLICT_DIAGNOSTIC,
    AlignedTargetFamily,
    AlignmentOperationCounts,
    align_sample_targets,
    align_targets_with_index,
    build_alignment_index,
)
from music_critic.tasks.encoding import TARGET_ENCODING_BY_TASK
from music_critic.tasks.multisource import (
    BatchStatistics,
    BatchTarget,
    MultiSourceBatch,
    MultiSourceContractError,
    MultiSourceSample,
    TaskBatchStatistics,
)
from music_critic.tasks.ontology import TARGET_FAMILIES


@dataclass(frozen=True, slots=True)
class CollatorBenchmark:
    """Lightweight, non-acceptance timing evidence for representative samples."""

    sample_count: int
    repeat_count: int
    alignment_seconds_per_repeat: float
    graph_construction_seconds_per_repeat: float
    full_collation_seconds_per_repeat: float
    node_count: int
    edge_count: int
    target_row_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0 or self.repeat_count <= 0:
            raise MultiSourceContractError(
                "collator benchmark counts must be positive"
            )
        if min(
            self.alignment_seconds_per_repeat,
            self.graph_construction_seconds_per_repeat,
            self.full_collation_seconds_per_repeat,
        ) < 0:
            raise MultiSourceContractError(
                "collator benchmark durations must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class TargetAlignmentBenchmark:
    """Target-heavy indexed lookup and complete collation evidence."""

    sample_count: int
    repeat_count: int
    source_target_entry_count: int
    emitted_row_count: int
    index_construction_seconds_per_repeat: float
    target_lookup_seconds_per_repeat: float
    full_collation_seconds_per_repeat: float
    operation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        counts = (
            self.sample_count,
            self.repeat_count,
            self.source_target_entry_count,
            self.emitted_row_count,
        )
        if any(value <= 0 for value in counts):
            raise MultiSourceContractError(
                "target alignment benchmark counts must be positive"
            )
        if min(
            self.index_construction_seconds_per_repeat,
            self.target_lookup_seconds_per_repeat,
            self.full_collation_seconds_per_repeat,
        ) < 0:
            raise MultiSourceContractError(
                "target alignment benchmark durations must be non-negative"
            )
        if tuple(name for name, _ in self.operation_counts) != tuple(
            sorted(name for name, _ in self.operation_counts)
        ):
            raise MultiSourceContractError(
                "target alignment benchmark operations must be deterministic"
            )


def _validate_samples(
    samples: Sequence[MultiSourceSample],
) -> tuple[MultiSourceSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise MultiSourceContractError(
            "collator input must be a sequence of prepared MultiSourceSample values"
        )
    prepared = tuple(samples)
    if not prepared:
        raise MultiSourceContractError("collator input cannot be empty")
    if not all(isinstance(sample, MultiSourceSample) for sample in prepared):
        raise MultiSourceContractError(
            "collator input contains a value that is not MultiSourceSample"
        )
    for sample in prepared:
        try:
            current_fingerprint = graph_fingerprint(sample.raw_graph)
        except GraphContractError as exc:
            raise MultiSourceContractError(
                f"prepared sample raw graph is invalid: {exc}"
            ) from exc
        if current_fingerprint != sample.raw_graph_fingerprint:
            raise MultiSourceContractError(
                "multisource.raw_graph_binding_mismatch: raw graph mutated "
                "after production sample preparation"
            )
    return prepared


def _encode_values(
    task_id: str,
    families: tuple[AlignedTargetFamily, ...],
) -> torch.Tensor | tuple[str | None, ...]:
    encoding = TARGET_ENCODING_BY_TASK[task_id]
    rows = tuple(row for family in families for row in family.rows)
    if encoding.encoding_kind == "closed_categorical_index":
        vocabulary = encoding.vocabulary or ()
        indices = {label: index for index, label in enumerate(vocabulary)}
        return torch.tensor(
            [
                indices[row.value] if row.availability else -1
                for row in rows
            ],
            dtype=torch.long,
        )
    if encoding.encoding_kind == "closed_multilabel":
        vocabulary = encoding.vocabulary or ()
        columns = {label: index for index, label in enumerate(vocabulary)}
        values = torch.zeros((len(rows), len(vocabulary)), dtype=torch.bool)
        for row_index, row in enumerate(rows):
            if not row.availability:
                continue
            assert isinstance(row.value, tuple)
            for label in row.value:
                values[row_index, columns[label]] = True
        return values
    return tuple(
        row.value if row.availability else None
        for row in rows
    )


def tensorize_aligned_targets(
    aligned_by_sample: Sequence[tuple[AlignedTargetFamily, ...]],
    raw_graph_batch: Batch,
) -> tuple[BatchTarget, ...]:
    """Apply exact local-to-global offsets and registry encodings to all tasks."""

    aligned = tuple(aligned_by_sample)
    if not aligned:
        raise MultiSourceContractError("tensorizer input cannot be empty")
    expected_tasks = tuple(spec.task_id for spec in TARGET_FAMILIES)
    for families in aligned:
        if tuple(family.task_id for family in families) != expected_tasks:
            raise MultiSourceContractError(
                "aligned families must contain every task in registry order"
            )
    validate_raw_graph_batch(raw_graph_batch, sample_count=len(aligned))

    target_batches: list[BatchTarget] = []
    for task_position, spec in enumerate(TARGET_FAMILIES):
        families = tuple(
            sample_families[task_position] for sample_families in aligned
        )
        rows_with_sample = tuple(
            (sample_index, row)
            for sample_index, family in enumerate(families)
            for row in family.rows
        )
        entity_indices: list[int] = []
        for sample_index, row in rows_with_sample:
            if row.entity_node_type is None:
                entity_indices.append(-1)
                continue
            ptr = raw_graph_batch[row.entity_node_type].ptr
            global_index = (
                int(ptr[sample_index].item()) + row.local_entity_index
            )
            if (
                global_index >= raw_graph_batch[row.entity_node_type].num_nodes
                or int(
                    raw_graph_batch[row.entity_node_type]
                    .batch[global_index]
                    .item()
                )
                != sample_index
            ):
                raise MultiSourceContractError(
                    "local-to-global target offset escaped its source sample"
                )
            entity_indices.append(global_index)

        confidence_values = tuple(
            row.confidence for _, row in rows_with_sample
        )
        confidence_mask = tuple(
            value is not None for value in confidence_values
        )
        confidence_tensor = (
            torch.tensor(
                [
                    float(value) if value is not None else 0.0
                    for value in confidence_values
                ],
                dtype=torch.float32,
            )
            if any(confidence_mask)
            else None
        )
        confidence_mask_tensor = (
            torch.tensor(confidence_mask, dtype=torch.bool)
            if confidence_tensor is not None
            else None
        )
        encoding = TARGET_ENCODING_BY_TASK[spec.task_id]
        target_batches.append(
            BatchTarget(
                task_id=spec.task_id,
                source_adapter=spec.source_adapter,
                supervision_context=spec.supervision_context,
                encoding_registry_version=encoding.registry_version,
                encoding_kind=encoding.encoding_kind,
                model_ready=encoding.model_ready,
                deferred_reason=encoding.deferred_reason,
                supervision_regime=encoding.supervision_regime,
                values=_encode_values(spec.task_id, families),
                availability_mask=torch.tensor(
                    [row.availability for _, row in rows_with_sample],
                    dtype=torch.bool,
                ),
                entity_indices=torch.tensor(entity_indices, dtype=torch.long),
                entity_index_mask=torch.tensor(
                    [
                        row.entity_node_type is not None
                        for _, row in rows_with_sample
                    ],
                    dtype=torch.bool,
                ),
                entity_node_types=tuple(
                    row.entity_node_type for _, row in rows_with_sample
                ),
                sample_indices=torch.tensor(
                    [
                        sample_index
                        for sample_index, _ in rows_with_sample
                    ],
                    dtype=torch.long,
                ),
                confidence=confidence_tensor,
                confidence_mask=confidence_mask_tensor,
                entry_count=len(rows_with_sample),
                source_entry_count=sum(
                    family.source_entry_count for family in families
                ),
                provenance_cpu=tuple(
                    row.provenance for _, row in rows_with_sample
                ),
                diagnostics_cpu=tuple(
                    row.diagnostics for _, row in rows_with_sample
                ),
            )
        )
    return tuple(target_batches)


def _task_statistics(target: BatchTarget) -> TaskBatchStatistics:
    aligned = target.availability_mask & target.entity_index_mask
    unaligned = target.availability_mask & ~target.entity_index_mask
    node_counts = Counter(
        node_type
        for node_type, has_index in zip(
            target.entity_node_types, target.entity_index_mask.tolist()
        )
        if has_index and node_type is not None
    )
    conflict_flags = tuple(
        any(
            diagnostic.code == ALIGNMENT_CONFLICT_DIAGNOSTIC
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
    return TaskBatchStatistics(
        task_id=target.task_id,
        source_entry_count=target.source_entry_count,
        target_row_count=target.entry_count,
        aligned_available_count=int(aligned.sum().item()),
        available_unaligned_row_count=int(unaligned.sum().item()),
        masked_row_count=masked_count,
        conflict_row_count=conflict_count,
        model_encodable_row_count=(
            target.entry_count if target.model_ready else 0
        ),
        supervision_eligible_row_count=int(
            target.supervision_eligibility_mask.sum().item()
        ),
        deferred_open_vocabulary_row_count=(
            0 if target.model_ready else target.entry_count
        ),
        node_type_counts=tuple(sorted(node_counts.items())),
        model_ready=target.model_ready,
    )


def _batch_statistics(
    *,
    samples: tuple[MultiSourceSample, ...],
    raw_graph_batch: Batch,
    targets: tuple[BatchTarget, ...],
) -> BatchStatistics:
    task_counts = tuple(_task_statistics(target) for target in targets)
    aggregate_node_types: Counter[str] = Counter()
    for task_count in task_counts:
        aggregate_node_types.update(dict(task_count.node_type_counts))
    dataset_counts = Counter(sample.dataset_id for sample in samples)
    return BatchStatistics(
        sample_count=len(samples),
        graph_count=raw_graph_batch.num_graphs,
        node_counts=tuple(
            sorted(
                (
                    node_type,
                    int(raw_graph_batch[node_type].num_nodes),
                )
                for node_type in MANDATORY_NODE_TYPES
            )
        ),
        edge_counts=tuple(
            sorted(
                (
                    "|".join(edge_type),
                    int(raw_graph_batch[edge_type].edge_index.shape[1]),
                )
                for edge_type in MANDATORY_EDGE_TYPES
            )
        ),
        dataset_counts=tuple(sorted(dataset_counts.items())),
        source_target_entry_count=sum(
            item.source_entry_count for item in task_counts
        ),
        target_row_count=sum(item.target_row_count for item in task_counts),
        aligned_available_count=sum(
            item.aligned_available_count for item in task_counts
        ),
        available_unaligned_row_count=sum(
            item.available_unaligned_row_count for item in task_counts
        ),
        masked_row_count=sum(item.masked_row_count for item in task_counts),
        conflict_row_count=sum(
            item.conflict_row_count for item in task_counts
        ),
        node_type_counts=tuple(sorted(aggregate_node_types.items())),
        task_counts=task_counts,
        model_ready_task_count=sum(item.model_ready for item in task_counts),
        deferred_open_vocabulary_task_count=sum(
            not item.model_ready for item in task_counts
        ),
        model_encodable_row_count=sum(
            item.model_encodable_row_count for item in task_counts
        ),
        supervision_eligible_row_count=sum(
            item.supervision_eligible_row_count for item in task_counts
        ),
        deferred_open_vocabulary_row_count=sum(
            item.deferred_open_vocabulary_row_count
            for item in task_counts
        ),
    )


def collate_multisource_samples(
    samples: Sequence[MultiSourceSample],
) -> MultiSourceBatch:
    """Collate prepared samples without injecting targets into raw PyG stores."""

    prepared = _validate_samples(samples)
    aligned = tuple(
        align_sample_targets(
            sample.canonical_piece,
            sample.raw_graph,
            sample,
        )
        for sample in prepared
    )
    raw_graph_batch = Batch.from_data_list(
        [sample.raw_graph for sample in prepared]
    )
    validate_raw_graph_batch(raw_graph_batch, sample_count=len(prepared))
    target_batches = tensorize_aligned_targets(aligned, raw_graph_batch)
    statistics = _batch_statistics(
        samples=prepared,
        raw_graph_batch=raw_graph_batch,
        targets=target_batches,
    )
    return MultiSourceBatch(
        raw_graph_batch=raw_graph_batch,
        target_batches=target_batches,
        dataset_ids=tuple(sample.dataset_id for sample in prepared),
        piece_ids=tuple(sample.piece_id for sample in prepared),
        source_group_ids=tuple(
            sample.source_group_id for sample in prepared
        ),
        lineage_group_ids=tuple(
            sample.lineage_group_id for sample in prepared
        ),
        diagnostics_cpu=tuple(sample.diagnostics for sample in prepared),
        statistics=statistics,
    )


def benchmark_multisource_collator(
    samples: Sequence[MultiSourceSample],
    *,
    repeats: int = 3,
) -> CollatorBenchmark:
    """Measure small representative batches; this is not corpus acceptance."""

    prepared = _validate_samples(samples)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise MultiSourceContractError(
            "collator benchmark repeats must be a positive integer"
        )

    started = perf_counter()
    for _ in range(repeats):
        tuple(
            align_sample_targets(
                sample.canonical_piece,
                sample.raw_graph,
                sample,
            )
            for sample in prepared
        )
    alignment_seconds = perf_counter() - started

    started = perf_counter()
    for _ in range(repeats):
        graph_batch = Batch.from_data_list(
            [sample.raw_graph for sample in prepared]
        )
        validate_raw_graph_batch(graph_batch, sample_count=len(prepared))
    graph_batch_seconds = perf_counter() - started

    started = perf_counter()
    result: MultiSourceBatch | None = None
    for _ in range(repeats):
        result = collate_multisource_samples(prepared)
    full_seconds = perf_counter() - started
    assert result is not None
    return CollatorBenchmark(
        sample_count=len(prepared),
        repeat_count=repeats,
        alignment_seconds_per_repeat=alignment_seconds / repeats,
        graph_construction_seconds_per_repeat=graph_batch_seconds / repeats,
        full_collation_seconds_per_repeat=full_seconds / repeats,
        node_count=sum(count for _, count in result.statistics.node_counts),
        edge_count=sum(count for _, count in result.statistics.edge_counts),
        target_row_count=result.statistics.target_row_count,
    )


def benchmark_target_alignment(
    samples: Sequence[MultiSourceSample],
    *,
    repeats: int = 3,
) -> TargetAlignmentBenchmark:
    """Benchmark target-heavy index construction and lookup independently."""

    prepared = _validate_samples(samples)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise MultiSourceContractError(
            "target alignment benchmark repeats must be a positive integer"
        )

    index_operations: AlignmentOperationCounts | None = None
    indices = ()
    started = perf_counter()
    for _ in range(repeats):
        index_operations = AlignmentOperationCounts()
        indices = tuple(
            build_alignment_index(
                sample.canonical_piece,
                instrumentation=index_operations,
            )
            for sample in prepared
        )
    index_seconds = perf_counter() - started
    assert index_operations is not None

    lookup_operations: AlignmentOperationCounts | None = None
    started = perf_counter()
    for _ in range(repeats):
        lookup_operations = AlignmentOperationCounts()
        tuple(
            align_targets_with_index(
                sample,
                alignment_index,
                instrumentation=lookup_operations,
            )
            for sample, alignment_index in zip(prepared, indices)
        )
    lookup_seconds = perf_counter() - started
    assert lookup_operations is not None

    started = perf_counter()
    result: MultiSourceBatch | None = None
    for _ in range(repeats):
        result = collate_multisource_samples(prepared)
    full_seconds = perf_counter() - started
    assert result is not None

    operation_counts = {
        name: value for name, value in index_operations.as_sorted_pairs()
    }
    for name, value in lookup_operations.as_sorted_pairs():
        operation_counts[name] = operation_counts.get(name, 0) + value
    return TargetAlignmentBenchmark(
        sample_count=len(prepared),
        repeat_count=repeats,
        source_target_entry_count=result.statistics.source_target_entry_count,
        emitted_row_count=result.statistics.target_row_count,
        index_construction_seconds_per_repeat=index_seconds / repeats,
        target_lookup_seconds_per_repeat=lookup_seconds / repeats,
        full_collation_seconds_per_repeat=full_seconds / repeats,
        operation_counts=tuple(sorted(operation_counts.items())),
    )


__all__ = [
    "CollatorBenchmark",
    "TargetAlignmentBenchmark",
    "benchmark_multisource_collator",
    "benchmark_target_alignment",
    "collate_multisource_samples",
    "tensorize_aligned_targets",
]
