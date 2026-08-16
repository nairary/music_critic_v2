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

import music_critic.ssl.hierarchical_masking as hierarchy_masking_module
from music_critic.device import (
    DEVICE_TRANSFER_CONTRACT_VERSION,
    RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION,
)
from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.contracts import (
    MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
    MASK_PLAN_CONTRACT_VERSION,
    MASK_POLICY_VERSION,
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    SSL_CONTRACT_VERSION,
    MaskPlan,
    SampleIdentity,
    canonical_sha256,
)
from music_critic.ssl.data import SSLBatch, collate_ssl_samples
from music_critic.ssl.hierarchical_masking import (
    CONTIGUOUS_BAR_PITCH_SPAN,
    HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION,
    HIERARCHY_MASK_POLICIES,
    HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT,
    HIERARCHY_MASK_POLICY_VERSION,
    HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION,
    HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION,
    HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
    HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION,
    HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION,
    SPAN_FINAL_CHOICE_RANK_METHOD,
    SPAN_POOL_MEMBERSHIP_RANK_METHOD,
    SPAN_SELECTION_METHOD,
    TRACK_BAR_PITCH_SPAN,
    HierarchicalMaskPlan,
    HierarchyMaskPolicyConfig,
    build_hierarchy_mask_plan,
    build_batched_hierarchy_mask_resolutions,
)
from music_critic.ssl.engine import (
    NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION,
    PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION,
    SSL_TRAINING_REPORT_VERSION,
)
from music_critic.ssl.hierarchy_fixture import (
    PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.masking import (
    PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION,
    derive_stable_seed,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import (
    PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION,
    SSL_MODEL_CONTRACT_VERSION,
    SSL_MODEL_OUTPUT_CONTRACT_VERSION,
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)
from music_critic.ssl.objective import (
    ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
    MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION,
    REPRESENTATION_LOSS_CONTRACT_VERSION,
    SSL_OBJECTIVE_CONTRACT_VERSION,
)


PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION = "1.2.0"
PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION = "1.2.2"
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
    """Enumerate existing global/node/edge stores without creating stores."""

    node_items = tuple(graph.node_items())
    edge_items = tuple(graph.edge_items())
    return (
        (f"global:{graph._global_store._key!r}", graph._global_store),
        *(
            (f"node:{node_type}", store)
            for node_type, store in node_items
        ),
        *(
            (
                "edge:" + "|".join(edge_type),
                store,
            )
            for edge_type, store in edge_items
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
            "selection_evidence": plan.selection.to_dict(),
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
            "selection_evidence": None,
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


def _span_diversity_audit(fixture: Any) -> dict[str, object]:
    config = HierarchyMaskPolicyConfig.create(
        weights={TRACK_BAR_PITCH_SPAN: 1.0},
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=4,
        span_budget_error_slack=1,
    )
    identity = SampleIdentity(
        "phase8a-selection-bias-oracle",
        "piece:early-middle-late-multitrack",
    )
    candidates = []
    next_note = 2
    for track in range(3):
        for bar in range(12):
            descendants = (
                (0, 1)
                if track == 0 and bar == 0
                else (next_note,)
            )
            if descendants == (next_note,):
                next_note += 1
            candidates.append((bar, bar, track, descendants))
    canonical_candidates = tuple(candidates)
    old_canonical_prefix = tuple(
        sorted(
            canonical_candidates,
            key=lambda candidate: (
                abs(len(candidate[3]) - 2),
                candidate[2],
                candidate[0],
                candidate[1],
                candidate[3],
            ),
        )[: config.span_selection_pool_size]
    )

    def selector_sequence(
        candidate_order: tuple[
            tuple[int, int, int, tuple[int, ...]], ...
        ],
        *,
        stage: str = "train",
        requested_epochs: range = range(256),
    ) -> tuple[object, ...]:
        selections = []
        for requested_epoch in requested_epochs:
            canonical_epoch = (
                requested_epoch if stage == "train" else 0
            )
            seed = derive_stable_seed(
                namespace=(
                    "music_critic.ssl.acceptance."
                    "span_rank_oracle.v1"
                ),
                global_seed=_GLOBAL_SEED,
                dataset_id=identity.dataset_id,
                piece_id=identity.piece_id,
                epoch=canonical_epoch,
                view_index=0,
                extra={
                    "stage": stage,
                    "policy": TRACK_BAR_PITCH_SPAN,
                    "policy_configuration_fingerprint": (
                        config.fingerprint
                    ),
                },
            )
            selection = hierarchy_masking_module._select_span(
                candidates=candidate_order,
                target_count=2,
                seed=seed,
                config=config,
                identity=identity,
                stage=stage,
                epoch=canonical_epoch,
                encoder_view_index=0,
                global_seed=_GLOBAL_SEED,
                policy=TRACK_BAR_PITCH_SPAN,
            )
            if selection is None:
                raise RuntimeError(
                    "Phase 8A positional-bias oracle empty"
                )
            selections.append(selection)
        return tuple(selections)

    selections = selector_sequence(canonical_candidates)
    repeated = selector_sequence(canonical_candidates)
    reversed_order = selector_sequence(
        tuple(reversed(canonical_candidates))
    )
    permuted_order = selector_sequence(
        canonical_candidates[::2] + canonical_candidates[1::2]
    )
    signatures = tuple(
        selection.candidate for selection in selections
    )
    repeated_signatures = tuple(
        selection.candidate for selection in repeated
    )
    signature_counts = Counter(signatures)
    error_counts = Counter(
        selection.selected_budget_error
        for selection in selections
    )
    candidate_error_counts = Counter(
        abs(len(candidate[3]) - 2)
        for candidate in canonical_candidates
    )
    strict = HierarchyMaskPolicyConfig.create(
        weights={TRACK_BAR_PITCH_SPAN: 1.0},
        min_span_bars=1,
        max_span_bars=1,
        span_selection_pool_size=4,
        span_budget_error_slack=0,
    )
    strict_selections = []
    for epoch in range(256):
        seed = derive_stable_seed(
            namespace=(
                "music_critic.ssl.acceptance.span_rank_oracle.v1"
            ),
            global_seed=_GLOBAL_SEED,
            dataset_id=identity.dataset_id,
            piece_id=identity.piece_id,
            epoch=epoch,
            view_index=0,
            extra={
                "stage": "train",
                "policy": TRACK_BAR_PITCH_SPAN,
                "policy_configuration_fingerprint": strict.fingerprint,
            },
        )
        selected = hierarchy_masking_module._select_span(
            candidates=canonical_candidates,
            target_count=2,
            seed=seed,
            config=strict,
            identity=identity,
            stage="train",
            epoch=epoch,
            encoder_view_index=0,
            global_seed=_GLOBAL_SEED,
            policy=TRACK_BAR_PITCH_SPAN,
        )
        if selected is None:
            raise RuntimeError("Phase 8A strict selector empty")
        strict_selections.append(selected)
    validation_zero = selector_sequence(
        canonical_candidates,
        stage="validation",
        requested_epochs=range(1),
    )
    validation_later = selector_sequence(
        canonical_candidates,
        stage="validation",
        requested_epochs=range(999, 1000),
    )
    retained_union = {
        candidate
        for selection in selections
        for candidate in selection.retained_pool_candidates
    }
    canonical_prefix_escape_count = sum(
        candidate not in old_canonical_prefix
        for candidate in signatures
    )
    if (
        selections != repeated
        or signatures != repeated_signatures
        or selections != reversed_order
        or selections != permuted_order
        or len(signature_counts) <= 1
        or validation_zero != validation_later
        or candidate_error_counts[0] != 1
        or retained_union != set(canonical_candidates)
        or canonical_prefix_escape_count <= 0
        or max(signature[0] for signature in signatures) < 10
        or len({signature[2] for signature in signatures}) <= 1
        or any(
            selection.selected_budget_error
            > selection.best_budget_error
            + config.span_budget_error_slack
            for selection in selections
        )
        or {
            selection.candidate
            for selection in strict_selections
        }
        != {canonical_candidates[0]}
    ):
        raise RuntimeError("Phase 8A span diversity audit failed")

    default_bounded_fixture_rows = []
    for policy in (
        CONTIGUOUS_BAR_PITCH_SPAN,
        TRACK_BAR_PITCH_SPAN,
    ):
        default_config = _single_policy_config(policy)
        for bounded_sample in fixture.raw_samples("train"):
            sequence_kwargs = {
                "dataset_id": bounded_sample.dataset_id,
                "piece_id": bounded_sample.piece_id,
                "policy": policy,
                "global_seed": _GLOBAL_SEED,
                "requested_mask_rate": _MASK_RATE,
                "stage": "train",
                "policy_config": default_config,
            }

            def default_sequence() -> tuple[
                HierarchicalMaskPlan, ...
            ]:
                return tuple(
                    build_hierarchy_mask_plan(
                        bounded_sample.raw_graph,
                        epoch=epoch,
                        **sequence_kwargs,
                    )
                    for epoch in range(256)
                )

            default_plans = default_sequence()
            default_replay = default_sequence()
            default_signatures = tuple(
                (
                    plan.selected_local_track_index,
                    plan.span_start_bar_index,
                    plan.span_end_bar_index,
                    plan.selected_local_unit_indices,
                    plan.selected_local_note_indices,
                )
                for plan in default_plans
            )
            replay_signatures = tuple(
                (
                    plan.selected_local_track_index,
                    plan.span_start_bar_index,
                    plan.span_end_bar_index,
                    plan.selected_local_unit_indices,
                    plan.selected_local_note_indices,
                )
                for plan in default_replay
            )
            selection = default_plans[0].selection
            selected_error_counts = Counter(
                plan.selection.span_selected_budget_error
                for plan in default_plans
            )
            replay_bit_exact = (
                default_plans == default_replay
                and default_signatures == replay_signatures
            )
            within_slack = all(
                plan.selection.span_selected_budget_error
                is not None
                and plan.selection.span_best_budget_error
                is not None
                and plan.selection.span_selected_budget_error
                <= plan.selection.span_best_budget_error
                + default_config.span_budget_error_slack
                for plan in default_plans
            )
            selection_evidence_is_constant = all(
                plan.selection.total_valid_candidate_count
                == selection.total_valid_candidate_count
                and plan.selection.span_best_budget_error
                == selection.span_best_budget_error
                and plan.selection.span_tolerance_candidate_count
                == selection.span_tolerance_candidate_count
                and plan.selection.span_admissible_pool_count
                == selection.span_admissible_pool_count
                and plan.selection.span_pool_membership_rank_method
                == SPAN_POOL_MEMBERSHIP_RANK_METHOD
                and plan.selection.span_final_choice_rank_method
                == SPAN_FINAL_CHOICE_RANK_METHOD
                for plan in default_plans
            )
            actual_unique_count = len(set(default_signatures))
            if (
                not replay_bit_exact
                or not within_slack
                or not selection_evidence_is_constant
                or actual_unique_count <= 1
            ):
                raise RuntimeError(
                    "Phase 8A default bounded span diversity audit failed"
                )
            default_bounded_fixture_rows.append(
                {
                    "policy": policy,
                    "sample_identity": [
                        bounded_sample.dataset_id,
                        bounded_sample.piece_id,
                    ],
                    "policy_config_fingerprint": (
                        default_config.fingerprint
                    ),
                    "configured_pool_size_limit": (
                        default_config.span_selection_pool_size
                    ),
                    "configured_budget_error_slack": (
                        default_config.span_budget_error_slack
                    ),
                    "total_valid_candidate_count": (
                        selection.total_valid_candidate_count
                    ),
                    "best_budget_error": (
                        selection.span_best_budget_error
                    ),
                    "tolerance_candidate_count": (
                        selection.span_tolerance_candidate_count
                    ),
                    "admissible_pool_count": (
                        selection.span_admissible_pool_count
                    ),
                    "actual_unique_selection_count": (
                        actual_unique_count
                    ),
                    "selected_budget_error_distribution": {
                        str(error): count
                        for error, count in sorted(
                            selected_error_counts.items()
                        )
                    },
                    "ordered_actual_selection_sequence_fingerprint": (
                        canonical_sha256(
                            [
                                {
                                    "track": signature[0],
                                    "start_bar": signature[1],
                                    "end_bar": signature[2],
                                    "selected_local_unit_indices": list(
                                        signature[3]
                                    ),
                                    "selected_local_note_indices": list(
                                        signature[4]
                                    ),
                                }
                                for signature in default_signatures
                            ]
                        )
                    ),
                    "fresh_replay_bit_exact": replay_bit_exact,
                    "all_selected_errors_within_best_plus_slack": (
                        within_slack
                    ),
                }
            )
    return {
        "crafted_fixture_identity": [
            identity.dataset_id,
            identity.piece_id,
        ],
        "epoch_range": [0, 255],
        "global_seed": _GLOBAL_SEED,
        "encoder_view_index": 0,
        "requested_hidden_note_count": 2,
        "config": config.to_dict(),
        "selection_method": SPAN_SELECTION_METHOD,
        "pool_membership_rank_method": (
            SPAN_POOL_MEMBERSHIP_RANK_METHOD
        ),
        "final_choice_rank_method": SPAN_FINAL_CHOICE_RANK_METHOD,
        "total_valid_candidate_count": len(canonical_candidates),
        "tolerance_qualified_candidate_count": len(
            canonical_candidates
        ),
        "retained_pool_count_per_epoch": (
            config.span_selection_pool_size
        ),
        "all_tolerance_candidates_entered_a_retained_pool": (
            retained_union == set(canonical_candidates)
        ),
        "distinct_retained_pool_member_count": len(retained_union),
        "candidate_budget_error_distribution": {
            str(error): count
            for error, count in sorted(candidate_error_counts.items())
        },
        "unique_closest_candidate_count": (
            candidate_error_counts[0]
        ),
        "actual_unique_selection_count": len(signature_counts),
        "actual_selection_counts": [
            {
                "track": signature[2],
                "start_bar": signature[0],
                "end_bar": signature[1],
                "selected_local_note_indices": list(signature[3]),
                "count": count,
            }
            for signature, count in sorted(signature_counts.items())
        ],
        "min_selected_start_bar": min(
            signature[0] for signature in signatures
        ),
        "max_selected_start_bar": max(
            signature[0] for signature in signatures
        ),
        "distinct_selected_track_count": len(
            {signature[2] for signature in signatures}
        ),
        "old_canonical_prefix_size": len(old_canonical_prefix),
        "canonical_prefix_escape_count": (
            canonical_prefix_escape_count
        ),
        "selected_budget_error_distribution": {
            str(error): count
            for error, count in sorted(error_counts.items())
        },
        "ordered_actual_selection_sequence_fingerprint": (
            canonical_sha256(
                [
                    {
                        "track": signature[2],
                        "start_bar": signature[0],
                        "end_bar": signature[1],
                        "selected_local_note_indices": list(
                            signature[3]
                        ),
                    }
                    for signature in signatures
                ]
            )
        ),
        "fresh_replay_bit_exact": True,
        "reverse_enumeration_bit_exact": True,
        "permuted_enumeration_bit_exact": True,
        "all_selected_errors_within_best_plus_slack": all(
            selection.selected_budget_error
            <= selection.best_budget_error
            + config.span_budget_error_slack
            for selection in selections
        ),
        "slack_zero_exact_best_control": {
            "config_fingerprint": strict.fingerprint,
            "actual_unique_selection_count": 1,
            "selected_candidate_canonical_identity": {
                "track": canonical_candidates[0][2],
                "start_bar": canonical_candidates[0][0],
                "end_bar": canonical_candidates[0][1],
                "selected_local_note_indices": list(
                    canonical_candidates[0][3]
                ),
            },
        },
        "validation_epochs_0_and_999_bit_exact": True,
        "default_bounded_fixture_audit": {
            "epoch_range": [0, 255],
            "global_seed": _GLOBAL_SEED,
            "encoder_view_index": 0,
            "requested_mask_rate": _MASK_RATE,
            "rows": default_bounded_fixture_rows,
            "all_rows_have_actual_diversity": True,
            "all_rows_replay_bit_exact": True,
            "all_rows_within_configured_budget_slack": True,
        },
        "quality_claim": None,
    }


def build_phase8a_bounded_acceptance_report() -> dict[str, object]:
    fixture = build_phase8a_hierarchy_fixture()
    batch = collate_ssl_samples(fixture.raw_samples("train"))
    graph_before = _snapshot_graph(batch.raw_graph_batch)
    model_metadata = _model().ssl_contract_metadata()
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
            "runtime_device_resolution": (
                RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION
            ),
            "device_transfer": DEVICE_TRANSFER_CONTRACT_VERSION,
            "ssl_umbrella": SSL_CONTRACT_VERSION,
            "ssl_training_report": SSL_TRAINING_REPORT_VERSION,
            "ssl_model": SSL_MODEL_CONTRACT_VERSION,
            "ssl_model_output": SSL_MODEL_OUTPUT_CONTRACT_VERSION,
            "representation_loss": (
                REPRESENTATION_LOSS_CONTRACT_VERSION
            ),
            "multi_view_representation_loss": (
                MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION
            ),
            "ssl_objective": SSL_OBJECTIVE_CONTRACT_VERSION,
            "anti_collapse_diagnostics": (
                ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
            ),
            "no_leakage_mutation_evidence": (
                NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION
            ),
            "pitch_sensitive_reconstruction_evidence": (
                PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION
            ),
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
        "hierarchy_mask_policy_contract_fingerprint": (
            HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT
        ),
        "span_selection_rank_contract": {
            "selection_method": SPAN_SELECTION_METHOD,
            "pool_membership_rank_method": (
                SPAN_POOL_MEMBERSHIP_RANK_METHOD
            ),
            "final_choice_rank_method": (
                SPAN_FINAL_CHOICE_RANK_METHOD
            ),
            "collision_fallback": (
                "track_start_end_descendants_v1"
            ),
        },
        "model_contract_metadata_fingerprint": canonical_sha256(
            model_metadata
        ),
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
        "span_diversity_audit": _span_diversity_audit(fixture),
        "all_policies_independently_exercised": (
            tuple(policies) == HIERARCHY_MASK_POLICIES
        ),
        "source_batch_unchanged": True,
        "mechanics_acceptance_gates": {
            "finite_existing_objectives": True,
            "loss_decrease_required": False,
            "positive_correct_target_preference_required": False,
            "no_leakage_and_pitch_sensitive_evidence_are_separate": (
                True
            ),
        },
        "cuda_measurement": None,
        "optional_cuda_hardware_evidence": {
            "contract_version": (
                PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION
            ),
            "script": "scripts/accept_phase8a_cuda_amp.py",
            "embedded": False,
            "status": "pending_independent_exact_final_rtx_run",
        },
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
