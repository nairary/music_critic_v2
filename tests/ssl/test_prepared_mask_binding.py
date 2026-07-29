from __future__ import annotations

from copy import copy, deepcopy
import inspect
import pickle
from typing import Any

import pytest
import torch
from torch import Tensor

from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    GraphContractError,
)
from music_critic.models import (
    HierarchicalBaselineConfig,
)
from music_critic.ssl.bounded_fixture import (
    build_phase7a_bounded_fixture,
)
from music_critic.ssl.contracts import (
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    MaskPlan,
    SSLContractError,
    canonical_sha256,
)
from music_critic.ssl.data import (
    SSLBatch,
    build_ssl_data_runtime,
    collate_ssl_samples,
)
from music_critic.ssl.masking import (
    PreparedMaskBinding,
    build_mask_plan,
    move_ssl_batch_with_prepared_binding,
    prepare_mask_binding,
    validate_prepared_mask_binding,
)
from music_critic.ssl.model import (
    SSL_MODEL_CONTRACT_VERSION,
    SSL_MODEL_OUTPUT_CONTRACT_VERSION,
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)
from music_critic.ssl.views import build_feature_mask_overlay
from music_critic.training.config import DataConfig


def _model() -> MaskedGraphSSLModel:
    return MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    )


def _batch() -> SSLBatch:
    return build_ssl_data_runtime(
        DataConfig(),
        seed=42,
    ).first_train_batch


def _binding(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    stage: str = "train",
    epoch: int = 3,
) -> PreparedMaskBinding:
    return prepare_mask_binding(
        batch,
        global_seed=41,
        epoch=epoch,
        requested_mask_rate=model.ssl_config.mask_rate,
        stage=stage,
    )


def _graph_snapshot(
    graph: Any,
) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, Tensor], ...]], ...]:
    return tuple(
        (
            tuple(store.keys()),
            tuple(
                (key, value.detach().clone())
                for key, value in store.items()
                if isinstance(value, Tensor)
            ),
        )
        for store in graph.stores
    )


def _assert_graph_snapshot(
    graph: Any,
    snapshot: tuple[
        tuple[tuple[str, ...], tuple[tuple[str, Tensor], ...]],
        ...,
    ],
) -> None:
    assert len(graph.stores) == len(snapshot)
    for store, (keys, tensors) in zip(
        graph.stores,
        snapshot,
        strict=True,
    ):
        assert tuple(store.keys()) == keys
        for key, before in tensors:
            assert torch.equal(store[key], before)


def _tensor_failure_snapshot(value: Tensor) -> dict[str, object]:
    return {
        "object": value,
        "object_id": id(value),
        "version": int(value._version),
        "shape": tuple(value.shape),
        "dtype": value.dtype,
        "device": value.device,
        "values": value.detach().clone(),
    }


def _assert_tensor_failure_snapshot(
    value: Tensor,
    snapshot: dict[str, object],
) -> None:
    assert value is snapshot["object"]
    assert id(value) == snapshot["object_id"]
    assert int(value._version) == snapshot["version"]
    assert tuple(value.shape) == snapshot["shape"]
    assert value.dtype == snapshot["dtype"]
    assert value.device == snapshot["device"]
    assert torch.equal(value, snapshot["values"])


def _failure_graph_snapshot(graph: Any) -> dict[str, object]:
    stores = []
    for store in graph.stores:
        values = []
        for key, value in store.items():
            if isinstance(value, Tensor):
                values.append(
                    (key, "tensor", _tensor_failure_snapshot(value))
                )
            else:
                values.append(
                    (
                        key,
                        "metadata",
                        {
                            "object": value,
                            "copy": deepcopy(value),
                        },
                    )
                )
        stores.append(
            {
                "object": store,
                "object_id": id(store),
                "keys": tuple(store.keys()),
                "values": tuple(values),
            }
        )
    return {
        "object": graph,
        "object_id": id(graph),
        "node_types": tuple(graph.node_types),
        "edge_types": tuple(graph.edge_types),
        "stores": tuple(stores),
    }


def _assert_failure_graph_snapshot(
    graph: Any,
    snapshot: dict[str, object],
) -> None:
    assert graph is snapshot["object"]
    assert id(graph) == snapshot["object_id"]
    assert tuple(graph.node_types) == snapshot["node_types"]
    assert tuple(graph.edge_types) == snapshot["edge_types"]
    stores = snapshot["stores"]
    assert len(graph.stores) == len(stores)
    for store, captured in zip(
        graph.stores,
        stores,
        strict=True,
    ):
        assert store is captured["object"]
        assert id(store) == captured["object_id"]
        assert tuple(store.keys()) == captured["keys"]
        for key, kind, value_snapshot in captured["values"]:
            assert key in store
            current = store[key]
            if kind == "tensor":
                assert isinstance(current, Tensor)
                _assert_tensor_failure_snapshot(
                    current,
                    value_snapshot,
                )
            else:
                assert not isinstance(current, Tensor)
                if not isinstance(
                    value_snapshot["object"],
                    (type(None), bool, int, float, str, bytes, tuple),
                ):
                    assert current is value_snapshot["object"]
                assert current == value_snapshot["copy"]


def _model_failure_snapshot(
    model: MaskedGraphSSLModel,
) -> dict[str, object]:
    return {
        "parameters": tuple(
            (
                name,
                parameter,
                _tensor_failure_snapshot(parameter),
            )
            for name, parameter in model.named_parameters()
        ),
        "buffers": tuple(
            (
                name,
                buffer,
                _tensor_failure_snapshot(buffer),
            )
            for name, buffer in model.named_buffers()
        ),
        "training": tuple(
            (name, module.training)
            for name, module in model.named_modules()
        ),
    }


def _assert_model_failure_snapshot(
    model: MaskedGraphSSLModel,
    snapshot: dict[str, object],
) -> None:
    parameters = tuple(model.named_parameters())
    assert tuple(name for name, _ in parameters) == tuple(
        name for name, _, _ in snapshot["parameters"]
    )
    for (name, current), (
        captured_name,
        captured_object,
        captured,
    ) in zip(
        parameters,
        snapshot["parameters"],
        strict=True,
    ):
        assert name == captured_name
        assert current is captured_object
        _assert_tensor_failure_snapshot(current, captured)
    buffers = tuple(model.named_buffers())
    assert tuple(name for name, _ in buffers) == tuple(
        name for name, _, _ in snapshot["buffers"]
    )
    for (name, current), (
        captured_name,
        captured_object,
        captured,
    ) in zip(
        buffers,
        snapshot["buffers"],
        strict=True,
    ):
        assert name == captured_name
        assert current is captured_object
        _assert_tensor_failure_snapshot(current, captured)
    assert tuple(
        (name, module.training)
        for name, module in model.named_modules()
    ) == snapshot["training"]


def _binding_failure_snapshot(
    binding: PreparedMaskBinding,
) -> dict[str, object]:
    return {
        "object": binding,
        "public": deepcopy(binding.to_dict()),
        "bound_graph": binding._bound_graph,
        "semantic_attestation": binding._semantic_attestation,
        "runtime_attestation": binding._runtime_attestation,
        "runtime_graph_evidence": binding._runtime_graph_evidence,
        "selected_indices_evidence": binding._selected_indices_evidence,
        "selected_indices": _tensor_failure_snapshot(
            binding.selected_global_note_indices_tensor
        ),
        "mask_plans": binding.mask_plans,
        "feature_overlay": binding.feature_overlay,
    }


def _assert_binding_failure_snapshot(
    binding: PreparedMaskBinding,
    snapshot: dict[str, object],
) -> None:
    assert binding is snapshot["object"]
    assert binding.to_dict() == snapshot["public"]
    assert binding._bound_graph is snapshot["bound_graph"]
    assert (
        binding._semantic_attestation
        == snapshot["semantic_attestation"]
    )
    assert binding._runtime_attestation == snapshot["runtime_attestation"]
    assert (
        binding._runtime_graph_evidence
        is snapshot["runtime_graph_evidence"]
    )
    assert (
        binding._selected_indices_evidence
        is snapshot["selected_indices_evidence"]
    )
    assert binding.mask_plans is snapshot["mask_plans"]
    assert binding.feature_overlay is snapshot["feature_overlay"]
    _assert_tensor_failure_snapshot(
        binding.selected_global_note_indices_tensor,
        snapshot["selected_indices"],
    )


def _graph_storage_guard(
    monkeypatch: pytest.MonkeyPatch,
    graph: Any,
) -> None:
    exact_tensor_ids: set[int] = set()
    storage_tokens: set[tuple[str, int | None, int, int]] = set()
    for store in graph.stores:
        for value in store.values():
            if not isinstance(value, Tensor):
                continue
            exact_tensor_ids.add(id(value))
            storage = value.untyped_storage()
            if storage.nbytes() > 0:
                storage_tokens.add(
                    (
                        value.device.type,
                        value.device.index,
                        storage.data_ptr(),
                        storage.nbytes(),
                    )
                )

    def belongs_to_graph(value: Tensor) -> bool:
        if id(value) in exact_tensor_ids:
            return True
        storage = value.untyped_storage()
        if storage.nbytes() == 0:
            return False
        return (
            value.device.type,
            value.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        ) in storage_tokens

    original_cpu = Tensor.cpu
    original_tolist = Tensor.tolist
    original_item = Tensor.item
    original_to = Tensor.to

    def guarded_cpu(value: Tensor, *args: object, **kwargs: object) -> Tensor:
        if belongs_to_graph(value) or value.numel() > 1:
            raise AssertionError(
                "bulk tensor materialized through Tensor.cpu"
            )
        return original_cpu(value, *args, **kwargs)

    def guarded_tolist(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> list[object]:
        if belongs_to_graph(value) or value.numel() > 1:
            raise AssertionError(
                "bulk tensor materialized through Tensor.tolist"
            )
        return original_tolist(value, *args, **kwargs)

    def guarded_item(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        if belongs_to_graph(value):
            raise AssertionError("graph tensor materialized through Tensor.item")
        return original_item(value, *args, **kwargs)

    def guarded_to(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        requested_device = kwargs.get("device")
        if requested_device is None and args and isinstance(
            args[0],
            (str, torch.device),
        ):
            requested_device = args[0]
        if (
            requested_device is not None
            and torch.device(requested_device).type == "cpu"
            and (
                belongs_to_graph(value)
                or (
                    value.device.type != "cpu"
                    and value.numel() > 1
                )
            )
        ):
            raise AssertionError(
                "bulk tensor materialized through Tensor.to(cpu)"
            )
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(Tensor, "cpu", guarded_cpu)
    monkeypatch.setattr(Tensor, "tolist", guarded_tolist)
    monkeypatch.setattr(Tensor, "item", guarded_item)
    monkeypatch.setattr(Tensor, "to", guarded_to)


def test_preparation_is_deterministic_cpu_sidecar_and_preserves_graph() -> None:
    model = _model().eval()
    batch = _batch()
    before = _graph_snapshot(batch.raw_graph_batch)

    first = _binding(model, batch)
    second = _binding(model, batch)

    assert first.contract_version == (
        PREPARED_MASK_BINDING_CONTRACT_VERSION
    ) == "1.1.0"
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert first.dataset_ids == batch.dataset_ids
    assert first.piece_ids == batch.piece_ids
    assert first.stage == "train"
    assert first.epoch == 3
    assert first.global_seed == 41
    assert first.requested_mask_rate == model.ssl_config.mask_rate
    assert first.ordered_plan_fingerprints == tuple(
        plan.fingerprint for plan in first.mask_plans
    )
    assert first.selected_global_note_indices_tensor.device.type == "cpu"
    assert len(first.validated_structure_sha256) == 64
    assert len(first.note_track_ownership_sha256) == 64
    assert not any(
        token in str(first.to_dict()).lower()
        for token in ("annotation", "target_bundle", "label")
    )
    _assert_graph_snapshot(batch.raw_graph_batch, before)
    assert not any(
        "mask_plan" in key or "prepared_mask" in key
        for store in batch.raw_graph_batch.stores
        for key in store.keys()
    )

    validation = _binding(
        model,
        batch,
        stage="validation",
        epoch=999,
    )
    assert validation.stage == "validation"
    assert validation.epoch == 0
    assert all(plan.epoch == 0 for plan in validation.mask_plans)


def test_prepared_binding_rejects_the_previous_contract_version() -> None:
    model = _model().eval()
    batch = _batch()
    binding = _binding(model, batch, epoch=0)
    outdated = copy(binding)
    object.__setattr__(outdated, "contract_version", "1.0.0")

    with pytest.raises(
        SSLContractError,
        match="contract version is incompatible",
    ):
        model(batch, prepared_mask_binding=outdated)


def test_plans_remain_batch_order_and_worker_transport_invariant() -> None:
    model = _model().eval()
    samples = build_phase7a_bounded_fixture().raw_samples("train")[:2]
    forward_batch = collate_ssl_samples(samples)
    reverse_batch = collate_ssl_samples(tuple(reversed(samples)))
    worker_roundtrip = pickle.loads(pickle.dumps(forward_batch))

    forward = _binding(model, forward_batch)
    reverse = _binding(model, reverse_batch)
    after_worker_transport = _binding(model, worker_roundtrip)
    single = tuple(
        _binding(model, collate_ssl_samples((sample,)))
        for sample in samples
    )

    forward_by_identity = {
        plan.sample_identity: plan for plan in forward.mask_plans
    }
    assert forward_by_identity == {
        plan.sample_identity: plan for plan in reverse.mask_plans
    }
    assert forward_by_identity == {
        binding.mask_plans[0].sample_identity: binding.mask_plans[0]
        for binding in single
    }
    assert after_worker_transport.to_dict() == forward.to_dict()


def _forged_internally_valid_binding(
    binding: PreparedMaskBinding,
    batch: SSLBatch,
    *,
    model: MaskedGraphSSLModel,
) -> PreparedMaskBinding:
    canonical = binding.mask_plans[0]
    graph = batch.raw_graph_batch.to_data_list()[0]
    different: MaskPlan | None = None
    for epoch in range(1, 33):
        candidate = build_mask_plan(
            graph,
            dataset_id=canonical.dataset_id,
            piece_id=canonical.piece_id,
            global_seed=canonical.global_seed,
            epoch=epoch,
            requested_mask_rate=model.ssl_config.mask_rate,
        )
        if (
            candidate.selected_local_node_indices
            != canonical.selected_local_node_indices
        ):
            different = candidate
            break
    assert different is not None
    alternate = MaskPlan.create(
        mask_policy=canonical.mask_policy,
        mask_policy_version=canonical.mask_policy_version,
        dataset_id=canonical.dataset_id,
        piece_id=canonical.piece_id,
        stage=canonical.stage,
        epoch=canonical.epoch,
        encoder_view_index=canonical.encoder_view_index,
        selected_node_type=canonical.selected_node_type,
        selected_local_node_indices=(
            different.selected_local_node_indices
        ),
        primary_feature_group=canonical.primary_feature_group,
        collateral_feature_masks=different.collateral_feature_masks,
        requested_mask_rate=canonical.requested_mask_rate,
        maskable_node_count=canonical.maskable_node_count,
        realized_mask_rate=different.realized_mask_rate,
        global_seed=canonical.global_seed,
        stable_seed=canonical.stable_seed,
        stable_seed_sha256=canonical.stable_seed_sha256,
    )
    overlay = build_feature_mask_overlay(
        batch.raw_graph_batch,
        (alternate,),
    )
    selected = alternate.selected_local_node_indices
    public_payload = binding.to_dict()
    public_payload.pop("fingerprint")
    public_payload["ordered_plan_fingerprints"] = [
        alternate.fingerprint
    ]
    public_payload["feature_overlay_fingerprint"] = overlay.fingerprint
    public_payload["selected_global_note_indices"] = list(selected)
    forged = copy(binding)
    object.__setattr__(forged, "mask_plans", (alternate,))
    object.__setattr__(
        forged,
        "ordered_plan_fingerprints",
        (alternate.fingerprint,),
    )
    object.__setattr__(forged, "feature_overlay", overlay)
    object.__setattr__(
        forged,
        "selected_global_note_indices",
        selected,
    )
    object.__setattr__(
        forged,
        "selected_global_note_indices_tensor",
        torch.tensor(selected, dtype=torch.long),
    )
    object.__setattr__(
        forged,
        "fingerprint",
        canonical_sha256(public_payload),
    )
    return forged


_RUNTIME_INPUT_MUTATIONS = (
    "x_cat_in_place",
    "x_cont_in_place",
    "categorical_availability_in_place",
    "continuous_availability_in_place",
    "candidate_slot_in_place",
    "tensor_replacement_same_shape_dtype",
    "tensor_replacement_different_shape",
    "tensor_replacement_different_dtype",
    "raw_only_in_place",
    "raw_only_replacement",
    "unknown_node_attribute_injection",
    "target_attribute_injection",
    "theory_attribute_injection",
    "provenance_attribute_injection",
    "edge_index_in_place",
    "ptr_in_place",
    "batch_in_place",
    "node_attribute_removal",
    "edge_attribute_injection",
    "global_attribute_removal",
    "unknown_node_store_injection",
    "unknown_edge_store_injection",
    "entity_id_metadata_mutation",
    "feature_name_metadata_mutation",
    "global_schema_metadata_mutation",
    "num_nodes_metadata_mutation",
)


def _mutate_runtime_input(graph: Any, mutation: str) -> None:
    if mutation == "x_cat_in_place":
        graph["note"].x_cat.add_(0)
    elif mutation == "x_cont_in_place":
        graph["note"].x_cont.add_(0.0)
    elif mutation == "categorical_availability_in_place":
        graph["note"].x_cat_available.logical_not_()
    elif mutation == "continuous_availability_in_place":
        graph["note"].x_cont_available.logical_not_()
    elif mutation == "candidate_slot_in_place":
        graph["beat"].candidate_slot.logical_not_()
    elif mutation == "tensor_replacement_same_shape_dtype":
        graph["note"].x_cat = graph["note"].x_cat.clone()
    elif mutation == "tensor_replacement_different_shape":
        current = graph["note"].x_cat
        graph["note"].x_cat = torch.empty(
            (current.shape[0] + 1, current.shape[1]),
            dtype=current.dtype,
            device=current.device,
        )
    elif mutation == "tensor_replacement_different_dtype":
        graph["note"].x_cont = graph["note"].x_cont.to(
            dtype=torch.float64
        )
    elif mutation == "raw_only_in_place":
        graph.raw_only.logical_not_()
    elif mutation == "raw_only_replacement":
        graph.raw_only = graph.raw_only.clone()
    elif mutation == "unknown_node_attribute_injection":
        graph["note"].unexpected_diagnostic = "forbidden"
    elif mutation == "target_attribute_injection":
        graph["note"].target = torch.zeros(
            1,
            device=graph["note"].x_cont.device,
        )
    elif mutation == "theory_attribute_injection":
        graph["bar"].theory_label = ("C:maj",)
    elif mutation == "provenance_attribute_injection":
        graph.provenance = {"source": "forbidden"}
    elif mutation == "edge_index_in_place":
        graph[MANDATORY_EDGE_TYPES[0]].edge_index.add_(0)
    elif mutation == "ptr_in_place":
        graph["note"].ptr.add_(0)
    elif mutation == "batch_in_place":
        graph["note"].batch.add_(0)
    elif mutation == "node_attribute_removal":
        del graph["note"]["x_cat"]
    elif mutation == "edge_attribute_injection":
        graph[MANDATORY_EDGE_TYPES[0]].diagnostic = True
    elif mutation == "global_attribute_removal":
        del graph._global_store["schema_version"]
    elif mutation == "unknown_node_store_injection":
        graph["diagnostic"].num_nodes = 0
    elif mutation == "unknown_edge_store_injection":
        graph[("note", "diagnostic", "note")].edge_index = (
            torch.empty(
                (2, 0),
                dtype=torch.long,
                device=graph["note"].x_cat.device,
            )
        )
    elif mutation == "entity_id_metadata_mutation":
        graph["song"].entity_id[0] = ("piece:forged",)
    elif mutation == "feature_name_metadata_mutation":
        names = graph["note"].cat_feature_names[0]
        graph["note"].cat_feature_names[0] = tuple(reversed(names))
    elif mutation == "global_schema_metadata_mutation":
        graph.schema_version[0] = "forged"
    elif mutation == "num_nodes_metadata_mutation":
        graph["note"].num_nodes += 1
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")


def _first_feature_encoder(model: MaskedGraphSSLModel) -> torch.nn.Module:
    return (
        model.encoder.local_baseline.encoder.feature_encoder.node_encoders[
            "song"
        ]
    )


@pytest.mark.parametrize("mutation", _RUNTIME_INPUT_MUTATIONS)
def test_runtime_input_mutation_fails_before_encoder_and_is_atomic(
    mutation: str,
) -> None:
    model = _model().eval()
    cpu_batch = _batch()
    cpu_binding = _binding(model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cpu",
    )
    graph = batch.raw_graph_batch
    _mutate_runtime_input(graph, mutation)

    graph_before_failure = _failure_graph_snapshot(graph)
    binding_before_failure = _binding_failure_snapshot(binding)
    model_before_failure = _model_failure_snapshot(model)
    first_feature_encoder_calls = 0

    def count_first_feature_encoder(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
    ) -> None:
        nonlocal first_feature_encoder_calls
        first_feature_encoder_calls += 1

    handle = _first_feature_encoder(model).register_forward_pre_hook(
        count_first_feature_encoder
    )
    try:
        with pytest.raises(
            SSLContractError,
            match=r"ssl\.prepared_binding\.runtime_input_changed",
        ):
            model(
                batch,
                prepared_mask_binding=binding,
            )
    finally:
        handle.remove()

    assert first_feature_encoder_calls == 0
    _assert_failure_graph_snapshot(graph, graph_before_failure)
    _assert_binding_failure_snapshot(binding, binding_before_failure)
    _assert_model_failure_snapshot(model, model_before_failure)


def test_transfer_cannot_sign_an_injected_raw_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from music_critic.ssl import data as ssl_data

    model = _model().eval()
    batch = _batch()
    binding = _binding(model, batch, epoch=0)
    graph_before_failure = _failure_graph_snapshot(
        batch.raw_graph_batch
    )
    binding_before_failure = _binding_failure_snapshot(binding)
    original_move = ssl_data.move_ssl_batch

    def injected_move(*args: object, **kwargs: object) -> SSLBatch:
        moved = original_move(*args, **kwargs)
        moved.raw_graph_batch["note"].theory_label = ("forbidden",)
        return moved

    monkeypatch.setattr(ssl_data, "move_ssl_batch", injected_move)

    with pytest.raises(
        SSLContractError,
        match=(
            r"ssl\.prepared_binding\.runtime_input_changed:"
            r"attribute_set:node:note"
        ),
    ):
        move_ssl_batch_with_prepared_binding(
            batch,
            binding,
            "cpu",
        )

    _assert_failure_graph_snapshot(
        batch.raw_graph_batch,
        graph_before_failure,
    )
    _assert_binding_failure_snapshot(binding, binding_before_failure)


def test_cpu_transfer_reissues_distinct_runtime_tensor_evidence() -> None:
    model = _model().eval()
    batch = _batch()
    binding = _binding(model, batch, epoch=0)

    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        batch,
        binding,
        "cpu",
    )

    assert moved_batch.raw_graph_batch is not batch.raw_graph_batch
    assert (
        moved_binding.selected_global_note_indices_tensor
        is not binding.selected_global_note_indices_tensor
    )
    assert (
        moved_binding._runtime_graph_evidence
        is not binding._runtime_graph_evidence
    )
    assert (
        moved_binding._selected_indices_evidence
        is not binding._selected_indices_evidence
    )
    assert (
        moved_binding._runtime_attestation
        != binding._runtime_attestation
    )


@pytest.mark.parametrize(
    "boundary_violation",
    ("foreign_graph", "forged_binding"),
)
def test_prepared_boundary_identity_failure_after_transfer_is_atomic(
    boundary_violation: str,
) -> None:
    model = _model().eval()
    sample = build_phase7a_bounded_fixture().raw_samples("train")[0]
    cpu_batch = collate_ssl_samples((sample,))
    cpu_binding = _binding(model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cpu",
    )
    if boundary_violation == "foreign_graph":
        foreign_graph = deepcopy(batch.raw_graph_batch)
        failure_batch = SSLBatch(
            raw_graph_batch=foreign_graph,
            dataset_ids=batch.dataset_ids,
            piece_ids=batch.piece_ids,
            sample_count=batch.sample_count,
            node_count=batch.node_count,
            edge_count=batch.edge_count,
        )
        failure_binding = binding
        expected_message = "does not belong"
    else:
        failure_batch = batch
        failure_binding = _forged_internally_valid_binding(
            binding,
            batch,
            model=model,
        )
        expected_message = "semantic attestation"

    graph = failure_batch.raw_graph_batch
    graph_before_failure = _failure_graph_snapshot(graph)
    binding_before_failure = _binding_failure_snapshot(
        failure_binding
    )
    model_before_failure = _model_failure_snapshot(model)
    first_feature_encoder_calls = 0

    def count_first_feature_encoder(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
    ) -> None:
        nonlocal first_feature_encoder_calls
        first_feature_encoder_calls += 1

    handle = _first_feature_encoder(model).register_forward_pre_hook(
        count_first_feature_encoder
    )
    try:
        with pytest.raises(
            SSLContractError,
            match=expected_message,
        ):
            model(
                failure_batch,
                prepared_mask_binding=failure_binding,
            )
    finally:
        handle.remove()

    assert first_feature_encoder_calls == 0
    _assert_failure_graph_snapshot(graph, graph_before_failure)
    _assert_binding_failure_snapshot(
        failure_binding,
        binding_before_failure,
    )
    _assert_model_failure_snapshot(model, model_before_failure)


def test_model_requires_exact_attested_binding_and_rejects_mutation() -> None:
    model = _model().eval()
    sample = build_phase7a_bounded_fixture().raw_samples("train")[0]
    batch = collate_ssl_samples((sample,))
    binding = _binding(model, batch, epoch=0)

    with pytest.raises(TypeError, match="prepared_mask_binding"):
        model(batch)

    foreign_graph = deepcopy(batch.raw_graph_batch)
    foreign_batch = SSLBatch(
        raw_graph_batch=foreign_graph,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        sample_count=batch.sample_count,
        node_count=batch.node_count,
        edge_count=batch.edge_count,
    )
    with pytest.raises(
        SSLContractError,
        match="does not belong",
    ):
        model(
            foreign_batch,
            prepared_mask_binding=binding,
        )

    forged = _forged_internally_valid_binding(
        binding,
        batch,
        model=model,
    )
    with pytest.raises(
        SSLContractError,
        match="semantic attestation",
    ):
        model(batch, prepared_mask_binding=forged)
    with pytest.raises(
        SSLContractError,
        match="construction plans are non-canonical",
    ):
        PreparedMaskBinding._create(
            dataset_ids=forged.dataset_ids,
            piece_ids=forged.piece_ids,
            stage=forged.stage,
            epoch=forged.epoch,
            encoder_view_index=forged.encoder_view_index,
            global_seed=forged.global_seed,
            requested_mask_rate=forged.requested_mask_rate,
            sample_count=forged.sample_count,
            node_counts=forged.node_counts,
            node_ptrs=forged.node_ptrs,
            edge_counts=forged.edge_counts,
            validated_structure_sha256=(
                forged.validated_structure_sha256
            ),
            note_track_ownership_sha256=(
                forged.note_track_ownership_sha256
            ),
            mask_plans=forged.mask_plans,
            feature_overlay=forged.feature_overlay,
            selected_global_note_indices=(
                forged.selected_global_note_indices
            ),
            bound_graph=batch.raw_graph_batch,
        )

    edge_index = batch.raw_graph_batch[
        MANDATORY_EDGE_TYPES[0]
    ].edge_index
    edge_index.add_(0)
    with pytest.raises(
        SSLContractError,
        match="runtime_input_changed",
    ):
        model(batch, prepared_mask_binding=binding)


def _public_encoder_entry(
    model: MaskedGraphSSLModel,
    entry: str,
) -> Any:
    if entry == "raw_feature_forward":
        return (
            model.encoder.local_baseline.encoder.feature_encoder.forward
        )
    if entry == "local_encoder_forward":
        return model.encoder.local_baseline.encoder.forward
    if entry == "local_baseline_encode":
        return model.encoder.local_baseline.encode
    if entry == "hierarchical_baseline_encode":
        return model.encoder.encode
    raise AssertionError(f"unknown public encoder entry: {entry}")


@pytest.mark.parametrize(
    "entry",
    (
        "raw_feature_forward",
        "local_encoder_forward",
        "local_baseline_encode",
        "hierarchical_baseline_encode",
    ),
)
def test_public_phase6_encoder_api_has_no_boolean_validation_bypass(
    entry: str,
) -> None:
    model = _model().eval()
    method = _public_encoder_entry(model, entry)

    assert "_prevalidated_input" not in inspect.signature(
        method
    ).parameters
    with pytest.raises(TypeError, match="_prevalidated_input"):
        method(
            _batch().raw_graph_batch,
            _prevalidated_input=True,
        )


@pytest.mark.parametrize(
    "entry",
    (
        "raw_feature_forward",
        "local_encoder_forward",
        "local_baseline_encode",
        "hierarchical_baseline_encode",
    ),
)
def test_ordinary_phase6_encoder_paths_always_revalidate_raw_graph(
    entry: str,
) -> None:
    model = _model().eval()
    graph = deepcopy(_batch().raw_graph_batch)
    graph["note"].target = torch.zeros(1)

    with pytest.raises(GraphContractError):
        _public_encoder_entry(model, entry)(graph)


@pytest.mark.parametrize(
    "token_kind",
    ("plain_object", "forged_attestation", "foreign_graph"),
)
def test_internal_prepared_encoder_rejects_invalid_capability_atomically(
    token_kind: str,
) -> None:
    model = _model().eval()
    cpu_batch = _batch()
    cpu_binding = _binding(model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cpu",
    )
    graph = batch.raw_graph_batch
    valid_token = validate_prepared_mask_binding(
        batch,
        binding,
        expected_mask_rate=model.ssl_config.mask_rate,
    )
    if token_kind == "plain_object":
        token: object = object()
        graph_argument = graph
        expected_message = "validated_token_invalid:type"
    elif token_kind == "forged_attestation":
        token = copy(valid_token)
        object.__setattr__(token, "_attestation", "0" * 64)
        graph_argument = graph
        expected_message = "validated_token_invalid:attestation"
    else:
        token = valid_token
        graph_argument = deepcopy(graph)
        expected_message = "validated_token_invalid:graph"

    graph_before_failure = _failure_graph_snapshot(graph_argument)
    binding_before_failure = _binding_failure_snapshot(binding)
    model_before_failure = _model_failure_snapshot(model)
    first_feature_encoder_calls = 0

    def count_first_feature_encoder(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
    ) -> None:
        nonlocal first_feature_encoder_calls
        first_feature_encoder_calls += 1

    handle = _first_feature_encoder(model).register_forward_pre_hook(
        count_first_feature_encoder
    )
    try:
        with pytest.raises(
            SSLContractError,
            match=expected_message,
        ):
            model.encoder._encode_prepared(
                graph_argument,
                prepared_input_token=token,
            )
    finally:
        handle.remove()

    assert first_feature_encoder_calls == 0
    _assert_failure_graph_snapshot(
        graph_argument,
        graph_before_failure,
    )
    _assert_binding_failure_snapshot(binding, binding_before_failure)
    _assert_model_failure_snapshot(model, model_before_failure)


def test_valid_prepared_encode_is_bit_exact_to_public_phase6_encode() -> None:
    torch.manual_seed(19)
    model = _model().eval()
    cpu_batch = _batch()
    cpu_binding = _binding(model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cpu",
    )
    token = validate_prepared_mask_binding(
        batch,
        binding,
        expected_mask_rate=model.ssl_config.mask_rate,
    )

    with torch.no_grad():
        ordinary = model.encoder.encode(batch.raw_graph_batch)
        prepared = model.encoder._encode_prepared(
            batch.raw_graph_batch,
            prepared_input_token=token,
        )

    for node_type in ordinary.fused.embeddings:
        assert torch.equal(
            ordinary.fused.embeddings[node_type],
            prepared.fused.embeddings[node_type],
        )
        assert torch.equal(
            ordinary.fused.batch_membership[node_type],
            prepared.fused.batch_membership[node_type],
        )


def test_prepared_cpu_forward_never_host_materializes_graph_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    model = _model().eval()
    cpu_batch = _batch()
    cpu_binding = _binding(model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cpu",
    )
    _graph_storage_guard(monkeypatch, batch.raw_graph_batch)

    with torch.no_grad():
        output = model(
            batch,
            prepared_mask_binding=binding,
        )

    assert output.contract_version == (
        SSL_MODEL_OUTPUT_CONTRACT_VERSION
    ) == "1.2.0"
    assert output.prepared_mask_binding_fingerprint == binding.fingerprint
    metadata = model.ssl_contract_metadata()
    assert metadata["ssl_model_contract_version"] == (
        SSL_MODEL_CONTRACT_VERSION
    ) == "1.2.0"
    assert metadata[
        "prepared_mask_binding_contract_version"
    ] == PREPARED_MASK_BINDING_CONTRACT_VERSION


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prepared binding CUDA+AMP acceptance requires CUDA",
)
def test_prepared_cuda_amp_uses_the_same_no_host_materialization_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    cpu_model = _model().eval()
    cpu_batch = _batch()
    cpu_binding = _binding(cpu_model, cpu_batch, epoch=0)
    batch, binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        cpu_binding,
        "cuda",
    )
    model = cpu_model.cuda()
    _graph_storage_guard(monkeypatch, batch.raw_graph_batch)

    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        output = model(
            batch,
            prepared_mask_binding=binding,
        )

    assert output.objective.total_loss is not None
    assert torch.isfinite(output.objective.total_loss)
    assert output.prepared_mask_binding_fingerprint == binding.fingerprint
