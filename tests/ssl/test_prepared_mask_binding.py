from __future__ import annotations

from copy import copy, deepcopy
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
    ) == "1.0.0"
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
        match="runtime attestation",
    ):
        model(batch, prepared_mask_binding=binding)


def test_ordinary_encoder_path_still_revalidates_raw_graph() -> None:
    graph = deepcopy(_batch().raw_graph_batch)
    graph["note"].ptr[1] += 1

    with pytest.raises(GraphContractError):
        _model().encoder.encode(graph)


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
    ) == "1.1.0"
    assert output.prepared_mask_binding_fingerprint == binding.fingerprint
    metadata = model.ssl_contract_metadata()
    assert metadata["ssl_model_contract_version"] == (
        SSL_MODEL_CONTRACT_VERSION
    ) == "1.1.0"
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
