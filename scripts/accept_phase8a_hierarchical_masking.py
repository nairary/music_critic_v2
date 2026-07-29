#!/usr/bin/env python3
"""Emit bounded Phase 8A mechanics evidence; never a quality comparison."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.contracts import (
    MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
    MASK_PLAN_CONTRACT_VERSION,
    MASK_POLICY_VERSION,
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    MaskPlan,
)
from music_critic.ssl.data import SSLBatch, collate_ssl_samples
from music_critic.ssl.hierarchical_masking import (
    HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION,
    HIERARCHY_MASK_POLICIES,
    HIERARCHY_MASK_POLICY_VERSION,
    HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION,
    HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION,
    HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
    HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION,
    HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION,
    HierarchicalMaskPlan,
    HierarchyMaskPolicyConfig,
    build_batched_hierarchy_mask_resolutions,
)
from music_critic.ssl.hierarchy_fixture import (
    PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.masking import (
    PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import (
    PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION,
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)


PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION = "1.0.0"
_GLOBAL_SEED = 42
_EPOCH = 0
_MASK_RATE = 0.30


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    tensor: Tensor
    version: int
    value: Tensor


@dataclass(frozen=True, slots=True)
class _GraphSnapshot:
    surface: tuple[tuple[str, tuple[str, ...]], ...]
    tensors: tuple[tuple[str, _TensorSnapshot], ...]
    metadata: tuple[tuple[str, str], ...]


def _store_items(graph: Any) -> tuple[tuple[str, Any], ...]:
    return (
        (f"global:{graph._global_store._key!r}", graph._global_store),
        *(
            (f"node:{node_type}", graph[node_type])
            for node_type in graph.node_types
        ),
        *(
            (
                "edge:" + "|".join(edge_type),
                graph[edge_type],
            )
            for edge_type in graph.edge_types
        ),
    )


def _metadata_signature(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}:{value!r}"
        )


def _snapshot_graph(graph: Any) -> _GraphSnapshot:
    surface: list[tuple[str, tuple[str, ...]]] = []
    tensors: list[tuple[str, _TensorSnapshot]] = []
    metadata: list[tuple[str, str]] = []
    for store_name, store in _store_items(graph):
        keys = tuple(sorted(str(key) for key in store.keys()))
        surface.append((store_name, keys))
        for key in keys:
            location = f"{store_name}.{key}"
            value = store[key]
            if isinstance(value, Tensor):
                tensors.append(
                    (
                        location,
                        _TensorSnapshot(
                            tensor=value,
                            version=int(value._version),
                            value=value.detach().clone(),
                        ),
                    )
                )
            else:
                metadata.append(
                    (location, _metadata_signature(value))
                )
    return _GraphSnapshot(
        surface=tuple(surface),
        tensors=tuple(tensors),
        metadata=tuple(metadata),
    )


def _graph_matches_snapshot(
    graph: Any,
    snapshot: _GraphSnapshot,
) -> bool:
    current_stores = dict(_store_items(graph))
    expected_store_names = tuple(
        store_name for store_name, _ in snapshot.surface
    )
    if tuple(current_stores) != expected_store_names:
        return False
    if tuple(
        (
            store_name,
            tuple(
                sorted(str(key) for key in current_stores[store_name].keys())
            ),
        )
        for store_name, _ in snapshot.surface
    ) != snapshot.surface:
        return False
    for location, evidence in snapshot.tensors:
        store_name, key = location.rsplit(".", 1)
        store = current_stores.get(store_name)
        if store is None:
            return False
        value = store[key]
        if (
            not isinstance(value, Tensor)
            or value is not evidence.tensor
            or int(value._version) != evidence.version
            or not torch.equal(value, evidence.value)
        ):
            return False
    for location, signature in snapshot.metadata:
        store_name, key = location.rsplit(".", 1)
        store = current_stores.get(store_name)
        if (
            store is None
            or _metadata_signature(store[key]) != signature
        ):
            return False
    return True


def _model() -> MaskedGraphSSLModel:
    torch.manual_seed(811)
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
            mask_rate=_MASK_RATE,
            decoder_views=1,
            decoder_remask_probability=0.0,
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    )


def _single_policy_config(policy: str) -> HierarchyMaskPolicyConfig:
    return HierarchyMaskPolicyConfig.create(
        weights={policy: 1.0},
        min_span_bars=1,
        max_span_bars=2,
    )


def _collateral_counts(plan: MaskPlan | HierarchicalMaskPlan) -> tuple[int, int]:
    peer_note_count = sum(
        len(mask.local_node_indices)
        for mask in plan.collateral_feature_masks
        if mask.node_type == "note"
    )
    owner_track_count = sum(
        len(mask.local_node_indices)
        for mask in plan.collateral_feature_masks
        if mask.node_type == "track"
    )
    return peer_note_count, owner_track_count


def _plan_evidence(
    policy: str,
    plan: MaskPlan | HierarchicalMaskPlan,
) -> dict[str, object]:
    peer_note_count, owner_track_count = _collateral_counts(plan)
    if isinstance(plan, HierarchicalMaskPlan):
        selected_unit_node_type = plan.selected_unit_node_type
        selected_units = plan.selected_local_unit_indices
        visible_note_count = plan.visible_pitched_note_count
        primary_count = plan.primary_masked_count
        plan_policy = plan.resolved_policy
        span = {
            "start_bar_index": plan.span_start_bar_index,
            "end_bar_index": plan.span_end_bar_index,
            "length_bars": plan.span_length_bars,
            "selected_local_track_index": (
                plan.selected_local_track_index
            ),
        }
    else:
        selected_unit_node_type = "note"
        selected_units = plan.selected_local_node_indices
        visible_note_count = plan.maskable_node_count - plan.selected_count
        primary_count = plan.selected_count
        plan_policy = plan.mask_policy
        span = {
            "start_bar_index": None,
            "end_bar_index": None,
            "length_bars": None,
            "selected_local_track_index": None,
        }
    maskable_note_count = plan.maskable_node_count
    primary_set = set(plan.selected_local_node_indices)
    visible_note_indices = tuple(
        local_note_index
        for local_note_index in range(maskable_note_count)
        if local_note_index not in primary_set
    )
    return {
        "dataset_id": plan.dataset_id,
        "piece_id": plan.piece_id,
        "requested_phase8a_policy": policy,
        "portable_plan_policy": plan_policy,
        "selected_unit_node_type": selected_unit_node_type,
        "selected_local_unit_indices": list(selected_units),
        "primary_masked_local_note_indices": list(
            plan.selected_local_node_indices
        ),
        "primary_masked_note_count": primary_count,
        "visible_pitched_note_count": visible_note_count,
        "visible_local_note_indices": list(
            visible_note_indices
        ),
        "realized_mask_rate": plan.realized_mask_rate,
        "peer_note_collateral_count": peer_note_count,
        "owner_track_collateral_count": owner_track_count,
        "span": span,
        "plan_fingerprint": plan.fingerprint,
    }


def _scalar(value: Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach())


def _gradient_evidence(
    model: MaskedGraphSSLModel,
) -> dict[str, object]:
    parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    with_gradient = tuple(
        (name, parameter.grad)
        for name, parameter in parameters
        if parameter.grad is not None
    )
    finite = tuple(
        name
        for name, gradient in with_gradient
        if bool(torch.isfinite(gradient).all())
    )
    nonzero = tuple(
        name
        for name, gradient in with_gradient
        if bool(torch.count_nonzero(gradient))
    )
    gradient_elements = sum(
        int(gradient.numel()) for _, gradient in with_gradient
    )
    parameter_elements = sum(
        int(parameter.numel()) for _, parameter in parameters
    )
    token_gradient = model.feature_mask_token.grad
    return {
        "trainable_parameter_tensor_count": len(parameters),
        "parameter_tensors_with_gradient": len(with_gradient),
        "parameter_tensors_with_finite_gradient": len(finite),
        "parameter_tensors_with_nonzero_gradient": len(nonzero),
        "trainable_parameter_element_count": parameter_elements,
        "gradient_element_count": gradient_elements,
        "tensor_coverage_fraction": (
            len(with_gradient) / len(parameters) if parameters else 0.0
        ),
        "all_present_gradients_finite": len(finite) == len(with_gradient),
        "feature_mask_token_gradient_present": (
            token_gradient is not None
        ),
        "feature_mask_token_gradient_finite": (
            token_gradient is not None
            and bool(torch.isfinite(token_gradient).all())
        ),
        "feature_mask_token_gradient_nonzero": (
            token_gradient is not None
            and bool(torch.count_nonzero(token_gradient))
        ),
    }


def _accept_policy(
    batch: SSLBatch,
    policy: str,
) -> dict[str, object]:
    config = _single_policy_config(policy)
    planner_arguments = {
        "dataset_ids": batch.dataset_ids,
        "piece_ids": batch.piece_ids,
        "global_seed": _GLOBAL_SEED,
        "epoch": _EPOCH,
        "requested_mask_rate": _MASK_RATE,
        "stage": "train",
        "policy_config": config,
    }
    resolutions = build_batched_hierarchy_mask_resolutions(
        batch.raw_graph_batch,
        **planner_arguments,
    )
    binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=config,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        batch,
        binding,
        "cpu",
    )
    graph_before = _snapshot_graph(moved_batch.raw_graph_batch)
    binding_before = deepcopy(moved_binding.to_dict())
    model = _model().train()
    model_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    output = model.forward_hierarchy(
        moved_batch,
        prepared_mask_binding=moved_binding,
    )
    loss = output.objective.total_loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError(
            f"Phase 8A bounded objective unavailable for {policy}"
        )
    loss.backward()

    graph_unchanged = _graph_matches_snapshot(
        moved_batch.raw_graph_batch,
        graph_before,
    )
    binding_unchanged = moved_binding.to_dict() == binding_before
    parameter_values_unchanged = all(
        torch.equal(value, model_before[name])
        for name, value in model.state_dict().items()
    )
    if not (
        graph_unchanged
        and binding_unchanged
        and parameter_values_unchanged
    ):
        raise RuntimeError(
            f"Phase 8A bounded forward mutated state for {policy}"
        )

    frequency = Counter(
        resolution.resolved_policy for resolution in resolutions
    )
    return {
        "policy": policy,
        "policy_config": config.to_dict(),
        "eligibility_and_resolution": [
            {
                "dataset_id": resolution.dataset_id,
                "piece_id": resolution.piece_id,
                "eligible_policies": list(
                    resolution.eligible_policies
                ),
                "eligible_normalized_weights": [
                    [name, weight]
                    for name, weight in (
                        resolution.eligible_normalized_weights
                    )
                ],
                "resolved_policy": resolution.resolved_policy,
                "resolution_fingerprint": resolution.fingerprint,
            }
            for resolution in resolutions
        ],
        "realized_policy_frequency": {
            str(name): count
            for name, count in sorted(
                frequency.items(),
                key=lambda item: str(item[0]),
            )
        },
        "realized_policy_frequency_denominator": len(
            resolutions
        ),
        "realized_policy_fractions": {
            str(name): count / len(resolutions)
            for name, count in sorted(
                frequency.items(),
                key=lambda item: str(item[0]),
            )
        },
        "plans": [
            _plan_evidence(policy, plan)
            for plan in binding.mask_plans
        ],
        "overlay_fingerprint": binding.feature_overlay.fingerprint,
        "prepared_binding_fingerprint": binding.fingerprint,
        "finite_existing_phase7a_objective": True,
        "losses": {
            "total": _scalar(loss),
            "note_reconstruction": _scalar(output.note_loss.mean),
            "bar_latent": _scalar(output.bar_latent.loss.mean),
            "song_latent": _scalar(output.song_latent.loss.mean),
        },
        "gradient_coverage": _gradient_evidence(model),
        "mutation_checks": {
            "raw_graph_and_store_values_unchanged": graph_unchanged,
            "prepared_binding_public_evidence_unchanged": (
                binding_unchanged
            ),
            "model_parameter_values_unchanged_without_optimizer": (
                parameter_values_unchanged
            ),
        },
    }


def build_phase8a_bounded_acceptance_report() -> dict[str, object]:
    fixture = build_phase8a_hierarchy_fixture()
    batch = collate_ssl_samples(fixture.raw_samples("train"))
    graph_before = _snapshot_graph(batch.raw_graph_batch)
    policies = {
        policy: _accept_policy(batch, policy)
        for policy in HIERARCHY_MASK_POLICIES
    }
    if not _graph_matches_snapshot(batch.raw_graph_batch, graph_before):
        raise RuntimeError("Phase 8A acceptance mutated its source batch")
    return {
        "acceptance_contract_version": (
            PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION
        ),
        "status": (
            "Phase 8A bounded mechanics accepted; "
            "Phase 8B not started"
        ),
        "scope": (
            "deterministic bounded CPU mechanics using existing Phase 7A "
            "note/bar/song objectives only"
        ),
        "explicit_non_goals": [
            "representation-quality comparison",
            "new onset/beat/track objective heads",
            "production SSL training",
            "full-corpus scan",
            "GPU performance evidence",
        ],
        "contracts": {
            "phase7a_mask_plan": MASK_PLAN_CONTRACT_VERSION,
            "phase7a_mask_policy": MASK_POLICY_VERSION,
            "feature_overlay": (
                MASKED_FEATURE_OVERLAY_CONTRACT_VERSION
            ),
            "prepared_binding": (
                PREPARED_MASK_BINDING_CONTRACT_VERSION
            ),
            "hierarchy_prepared_binding": (
                PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
            ),
            "hierarchical_mask_plan": (
                HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
            ),
            "hierarchy_mask_policy": HIERARCHY_MASK_POLICY_VERSION,
            "hierarchy_policy_config": (
                HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
            ),
            "hierarchy_policy_mixture": (
                HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
            ),
            "hierarchy_selection_evidence": (
                HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION
            ),
            "hierarchy_unavailable_reason": (
                HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION
            ),
            "hierarchy_prepared_binding_profile": (
                HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
            ),
            "bounded_fixture": (
                PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION
            ),
            "hierarchy_ssl_output": (
                PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION
            ),
        },
        "fixture": {
            "fingerprints": fixture.fingerprint_bundle(),
            "counts": {
                "all": fixture.count_summary(),
                "train": fixture.count_summary("train"),
                "validation": fixture.count_summary("validation"),
            },
            "train_identities": [
                list(identity)
                for identity in fixture.identities("train")
            ],
            "validation_identities": [
                list(identity)
                for identity in fixture.identities("validation")
            ],
        },
        "configuration": {
            "global_seed": _GLOBAL_SEED,
            "epoch": _EPOCH,
            "stage": "train",
            "requested_note_mask_rate": _MASK_RATE,
            "device": "cpu",
        },
        "policies": policies,
        "all_policies_independently_exercised": (
            tuple(policies) == HIERARCHY_MASK_POLICIES
        ),
        "source_batch_unchanged": True,
        "cuda_measurement": None,
        "quality_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path; stdout is always emitted.",
    )
    arguments = parser.parse_args()
    report = build_phase8a_bounded_acceptance_report()
    text = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            text + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(text)


if __name__ == "__main__":
    main()
