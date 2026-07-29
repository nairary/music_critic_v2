from __future__ import annotations

import copy

import pytest
import torch

from music_critic.device import RuntimeDeviceError
from music_critic.tasks import TARGET_FAMILIES
from music_critic.training.device import (
    DEVICE_TRANSFER_CONTRACT_VERSION,
    TransferInstrumentation,
    move_multisource_batch,
    validate_device_batch,
)
from music_critic.training.engine import _resolve_device


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
    assert DEVICE_TRANSFER_CONTRACT_VERSION == "1.0.1"
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


def test_normal_transfer_validates_semantics_only_on_cpu(
    bounded_batch,
) -> None:
    evidence = TransferInstrumentation()
    move_multisource_batch(
        bounded_batch,
        "cpu",
        instrumentation=evidence,
    )
    assert evidence.source_cpu_semantic_validation_calls == 1
    assert evidence.post_transfer_debug_validation_calls == 0

    debug_evidence = TransferInstrumentation()
    move_multisource_batch(
        bounded_batch,
        "cpu",
        debug_validate_device=True,
        instrumentation=debug_evidence,
    )
    assert (
        debug_evidence.source_cpu_semantic_validation_calls == 1
    )
    assert (
        debug_evidence.post_transfer_debug_validation_calls == 1
    )


def test_training_runtime_resolves_abstract_cuda_to_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    resolved = _resolve_device(
        {"device": {"name": "cuda", "amp": True}}
    )

    assert resolved == torch.device("cuda:1")


def test_direct_transfer_rejects_unavailable_cuda_structurally(
    monkeypatch: pytest.MonkeyPatch,
    bounded_batch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        RuntimeDeviceError,
        match=r"^runtime\.device\.cuda_unavailable:requested=cuda$",
    ):
        move_multisource_batch(bounded_batch, "cuda")
