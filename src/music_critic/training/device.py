"""Official non-mutating device transfer for Phase 5B.1 batches."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.tasks import (
    ENTITY_NODE_TYPE_TO_CODE,
    TARGET_FAMILIES,
    BatchTarget,
    MultiSourceBatch,
)


DEVICE_TRANSFER_CONTRACT_VERSION = "1.0.0"


class DeviceTransferError(ValueError):
    """Structured failure in the Phase 6C device-transfer boundary."""


@dataclass(slots=True)
class TransferInstrumentation:
    cpu_semantic_validations: int = 0
    device_semantic_validations: int = 0
    device_tensor_to_python_syncs: int = 0


def _without_post_init(
    cls: type[Any],
    source: object,
    replacements: dict[str, object],
) -> Any:
    """Rebuild an already CPU-validated frozen dataclass without revalidation."""

    result = object.__new__(cls)
    for item in fields(cls):
        object.__setattr__(
            result,
            item.name,
            replacements.get(item.name, getattr(source, item.name)),
        )
    return result


def _move_tensor(
    value: Tensor | None,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Tensor | None:
    return (
        None
        if value is None
        else value.to(device=device, non_blocking=non_blocking)
    )


def _move_target(
    target: BatchTarget,
    device: torch.device,
    *,
    non_blocking: bool,
) -> BatchTarget:
    values = (
        target.values.to(device=device, non_blocking=non_blocking)
        if isinstance(target.values, Tensor)
        else target.values
    )
    return _without_post_init(
        BatchTarget,
        target,
        {
            "values": values,
            "availability_mask": _move_tensor(
                target.availability_mask,
                device,
                non_blocking=non_blocking,
            ),
            "entity_indices": _move_tensor(
                target.entity_indices,
                device,
                non_blocking=non_blocking,
            ),
            "entity_index_mask": _move_tensor(
                target.entity_index_mask,
                device,
                non_blocking=non_blocking,
            ),
            "entity_node_type_codes": _move_tensor(
                target.entity_node_type_codes,
                device,
                non_blocking=non_blocking,
            ),
            "sample_indices": _move_tensor(
                target.sample_indices,
                device,
                non_blocking=non_blocking,
            ),
            "confidence": _move_tensor(
                target.confidence,
                device,
                non_blocking=non_blocking,
            ),
            "confidence_mask": _move_tensor(
                target.confidence_mask,
                device,
                non_blocking=non_blocking,
            ),
        },
    )


def _target_tensors(target: BatchTarget) -> tuple[Tensor, ...]:
    values = []
    for value in (
        target.values,
        target.availability_mask,
        target.entity_indices,
        target.entity_index_mask,
        target.entity_node_type_codes,
        target.sample_indices,
        target.confidence,
        target.confidence_mask,
    ):
        if isinstance(value, Tensor):
            values.append(value)
    return tuple(values)


def validate_device_batch(
    batch: MultiSourceBatch,
    device: torch.device | str,
    *,
    source: MultiSourceBatch | None = None,
) -> None:
    """Validate tensor devices, shapes, task order, and graph binding."""

    expected_device = torch.device(device)
    expected_tasks = tuple(spec.task_id for spec in TARGET_FAMILIES)
    if tuple(item.task_id for item in batch.target_batches) != expected_tasks:
        raise DeviceTransferError("training.device.task_order_mismatch")
    for store in batch.raw_graph_batch.stores:
        for value in store.values():
            if isinstance(value, Tensor) and value.device != expected_device:
                raise DeviceTransferError(
                    "training.device.graph_tensor_mismatch"
                )
    for target in batch.target_batches:
        if any(
            value.device != expected_device
            for value in _target_tensors(target)
        ):
            raise DeviceTransferError(
                f"training.device.target_tensor_mismatch:{target.task_id}"
            )
        for node_type in MANDATORY_NODE_TYPES:
            code = ENTITY_NODE_TYPE_TO_CODE[node_type]
            selected = (
                target.entity_index_mask
                & (target.entity_node_type_codes == code)
            )
            rows = torch.nonzero(selected, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            entity_indices = target.entity_indices.index_select(0, rows)
            sample_indices = target.sample_indices.index_select(0, rows)
            node_count = batch.raw_graph_batch[node_type].num_nodes
            if bool((entity_indices >= node_count).any()) or not torch.equal(
                batch.raw_graph_batch[node_type].batch.index_select(
                    0, entity_indices
                ),
                sample_indices,
            ):
                raise DeviceTransferError(
                    f"training.device.graph_binding_mismatch:{target.task_id}"
                )
    if source is None:
        return
    if (
        batch.dataset_ids != source.dataset_ids
        or batch.piece_ids != source.piece_ids
        or batch.source_group_ids != source.source_group_ids
        or batch.lineage_group_ids != source.lineage_group_ids
        or batch.diagnostics_cpu != source.diagnostics_cpu
        or batch.statistics != source.statistics
    ):
        raise DeviceTransferError("training.device.cpu_sidecar_changed")
    if tuple(item.task_id for item in source.target_batches) != expected_tasks:
        raise DeviceTransferError("training.device.source_task_order_mismatch")
    for target, original in zip(
        batch.target_batches, source.target_batches, strict=True
    ):
        if (
            target.entity_node_types != original.entity_node_types
            or target.provenance_cpu != original.provenance_cpu
            or target.diagnostics_cpu != original.diagnostics_cpu
        ):
            raise DeviceTransferError(
                f"training.device.target_cpu_sidecar_changed:{target.task_id}"
            )
        for moved, before in zip(
            _target_tensors(target),
            _target_tensors(original),
            strict=True,
        ):
            if moved.shape != before.shape or moved.dtype != before.dtype:
                raise DeviceTransferError(
                    f"training.device.target_shape_changed:{target.task_id}"
                )
    for node_type in batch.raw_graph_batch.node_types:
        if (
            int(batch.raw_graph_batch[node_type].num_nodes)
            != int(source.raw_graph_batch[node_type].num_nodes)
        ):
            raise DeviceTransferError(
                f"training.device.graph_shape_changed:{node_type}"
            )
    for edge_type in batch.raw_graph_batch.edge_types:
        if (
            batch.raw_graph_batch[edge_type].edge_index.shape
            != source.raw_graph_batch[edge_type].edge_index.shape
        ):
            raise DeviceTransferError(
                "training.device.graph_shape_changed:"
                + "|".join(edge_type)
            )


def move_multisource_batch(
    batch: MultiSourceBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
    debug_validate_device: bool = False,
    instrumentation: TransferInstrumentation | None = None,
) -> MultiSourceBatch:
    """Validate on CPU, then move without CUDA semantic revalidation."""

    if not isinstance(batch, MultiSourceBatch):
        raise DeviceTransferError("training.device.input_type_invalid")
    if any(
        value.device.type != "cpu"
        for target in batch.target_batches
        for value in _target_tensors(target)
    ):
        raise DeviceTransferError("training.device.source_not_cpu")
    validate_device_batch(batch, "cpu")
    if instrumentation is not None:
        instrumentation.cpu_semantic_validations += 1
    target_device = torch.device(device)
    graph = copy.deepcopy(batch.raw_graph_batch)
    # PyG's recursive ``Data.to`` also rewrites tuple-valued CPU metadata
    # (for example entity IDs) into lists. Move only actual tensor
    # attributes so the raw graph contract and CPU sidecars remain exact.
    for store in graph.stores:
        for key, value in tuple(store.items()):
            if isinstance(value, Tensor):
                store[key] = value.to(
                    device=target_device,
                    non_blocking=non_blocking,
                )
    targets = tuple(
        _move_target(
            target,
            target_device,
            non_blocking=non_blocking,
        )
        for target in batch.target_batches
    )
    moved = _without_post_init(
        MultiSourceBatch,
        batch,
        {
            "raw_graph_batch": graph,
            "target_batches": targets,
        },
    )
    _validate_moved_structure(moved, target_device, source=batch)
    if debug_validate_device:
        validate_device_batch(moved, target_device, source=batch)
        if instrumentation is not None:
            instrumentation.device_semantic_validations += 1
    return moved


def _validate_moved_structure(
    batch: MultiSourceBatch,
    device: torch.device,
    *,
    source: MultiSourceBatch,
) -> None:
    """Check transfer invariants without data-dependent tensor predicates."""

    expected_tasks = tuple(spec.task_id for spec in TARGET_FAMILIES)
    if tuple(item.task_id for item in batch.target_batches) != expected_tasks:
        raise DeviceTransferError("training.device.task_order_mismatch")
    if (
        batch.dataset_ids != source.dataset_ids
        or batch.piece_ids != source.piece_ids
        or batch.source_group_ids != source.source_group_ids
        or batch.lineage_group_ids != source.lineage_group_ids
        or batch.diagnostics_cpu != source.diagnostics_cpu
        or batch.statistics != source.statistics
    ):
        raise DeviceTransferError("training.device.cpu_sidecar_changed")
    for store in batch.raw_graph_batch.stores:
        for value in store.values():
            if isinstance(value, Tensor) and value.device != device:
                raise DeviceTransferError(
                    "training.device.graph_tensor_mismatch"
                )
    for target, original in zip(
        batch.target_batches, source.target_batches, strict=True
    ):
        if (
            target.entity_node_types != original.entity_node_types
            or target.provenance_cpu != original.provenance_cpu
            or target.diagnostics_cpu != original.diagnostics_cpu
        ):
            raise DeviceTransferError(
                f"training.device.target_cpu_sidecar_changed:{target.task_id}"
            )
        for moved, before in zip(
            _target_tensors(target),
            _target_tensors(original),
            strict=True,
        ):
            if (
                moved.device != device
                or moved.shape != before.shape
                or moved.dtype != before.dtype
            ):
                raise DeviceTransferError(
                    f"training.device.target_shape_changed:{target.task_id}"
                )
    for node_type in batch.raw_graph_batch.node_types:
        if (
            int(batch.raw_graph_batch[node_type].num_nodes)
            != int(source.raw_graph_batch[node_type].num_nodes)
        ):
            raise DeviceTransferError(
                f"training.device.graph_shape_changed:{node_type}"
            )
    for edge_type in batch.raw_graph_batch.edge_types:
        if (
            batch.raw_graph_batch[edge_type].edge_index.shape
            != source.raw_graph_batch[edge_type].edge_index.shape
        ):
            raise DeviceTransferError(
                "training.device.graph_shape_changed:"
                + "|".join(edge_type)
            )


__all__ = [
    "DEVICE_TRANSFER_CONTRACT_VERSION",
    "DeviceTransferError",
    "TransferInstrumentation",
    "move_multisource_batch",
    "validate_device_batch",
]
