from __future__ import annotations

import copy

import torch

from music_critic.tasks import TARGET_FAMILIES
from music_critic.training.device import (
    move_multisource_batch,
    validate_device_batch,
)


def _graph_tensor_snapshot(graph) -> dict[tuple[object, str], torch.Tensor]:
    result = {}
    for store in graph.stores:
        key_prefix = getattr(store, "_key", None)
        for name, value in store.items():
            if isinstance(value, torch.Tensor):
                result[(key_prefix, name)] = value.detach().clone()
    return result


def test_cpu_transfer_is_non_mutating_and_keeps_sidecars(
    bounded_batch,
) -> None:
    graph_before = _graph_tensor_snapshot(
        bounded_batch.raw_graph_batch
    )
    metadata_before = copy.deepcopy(
        bounded_batch.raw_graph_batch["song"].entity_id
    )
    moved = move_multisource_batch(
        bounded_batch,
        "cpu",
        non_blocking=True,
    )

    assert moved is not bounded_batch
    assert moved.raw_graph_batch is not bounded_batch.raw_graph_batch
    assert moved.dataset_ids == bounded_batch.dataset_ids
    assert moved.piece_ids == bounded_batch.piece_ids
    assert moved.diagnostics_cpu == bounded_batch.diagnostics_cpu
    assert moved.statistics == bounded_batch.statistics
    assert moved.raw_graph_batch["song"].entity_id == metadata_before
    assert bounded_batch.raw_graph_batch["song"].entity_id == metadata_before
    assert tuple(item.task_id for item in moved.target_batches) == tuple(
        family.task_id for family in TARGET_FAMILIES
    )
    validate_device_batch(moved, "cpu", source=bounded_batch)

    for key, value in graph_before.items():
        store_key, attribute = key
        store = (
            bounded_batch.raw_graph_batch._global_store
            if store_key is None
            else bounded_batch.raw_graph_batch[store_key]
        )
        assert torch.equal(store[attribute], value)
    moved.raw_graph_batch["note"].x_cat[0, 0] += 1
    assert torch.equal(
        bounded_batch.raw_graph_batch["note"].x_cat,
        graph_before[("note", "x_cat")],
    )


def test_transfer_does_not_put_targets_in_raw_graph(
    bounded_batch,
) -> None:
    moved = move_multisource_batch(bounded_batch, "cpu")
    for store in moved.raw_graph_batch.stores:
        assert all(
            "target" not in attribute
            for attribute in store.keys()
        )
    for target, original in zip(
        moved.target_batches,
        bounded_batch.target_batches,
        strict=True,
    ):
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
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
        if not isinstance(target.values, torch.Tensor):
            assert target.values == original.values
        assert target.provenance_cpu is not None
