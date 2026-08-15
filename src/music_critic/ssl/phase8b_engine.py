"""Official deterministic training path for explicit Phase 8B.1 configs.

The Phase 7A engine remains the default and is not routed through this module.
This path deliberately reports mechanics and accounting only; it is not a
compute-matched or effectiveness comparison between objective variants.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

from music_critic.ssl.checkpoint import (
    SSL_METRIC_ROW_VERSION,
    SSLResumeState,
    load_ssl_checkpoint,
    save_ssl_checkpoint,
    ssl_checkpoint_metadata,
)
from music_critic.ssl.data import SSLBatch, SSLDataRuntime, build_ssl_data_runtime
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    HIERARCHY_MASK_POLICIES,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchyMaskPolicyConfig,
)
from music_critic.ssl.masking import (
    PreparedHierarchyMaskBinding,
    PreparedMaskBinding,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import (
    MaskedGraphSSLModel,
    Phase8AHierarchySSLForwardOutput,
    SSLForwardOutput,
)
from music_critic.ssl.multilevel import (
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    ONSET_LATENT,
    PHASE7A_BAR_LATENT,
    PHASE7A_NOTE_RECONSTRUCTION,
    PHASE7A_SONG_LATENT,
    PHASE8B_AMP_COMPUTE_CONTRACT,
    PHASE8B_NEW_OBJECTIVE_FAMILIES,
    PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
    PHASE8B_SCHEDULED_VIEW_AGGREGATION,
    TRACK_LATENT,
    Phase8BMultilevelSSLForwardOutput,
    Phase8BMultilevelSSLModel,
    Phase8BObjectiveAccumulator,
    Phase8BObjectiveConfig,
    aggregate_phase8b_policy_pass_losses,
    build_phase8b_model_from_config,
    prepare_phase8b_objective_binding,
)
from music_critic.ssl.multilevel_checkpoint import (
    PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION,
)
from music_critic.ssl.phase8b_acceptance import (
    phase8b_cross_policy_manual_oracle,
)
from music_critic.ssl.transfer import export_pretrained_encoder_state


PHASE8B_ENGINE_CONTRACT_VERSION = "1.2.0"
PHASE8B_MASKING_CONFIG_CONTRACT_VERSION = "1.1.0"
PHASE8B_RUN_MANIFEST_VERSION = "1.2.0"
PHASE8B_TRAINING_REPORT_VERSION = "1.2.0"
PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION = "1.0.0"
PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION = "1.1.0"
PHASE8B_GRAD_SCALER_INITIAL_SCALE = 16384.0

_HIERARCHY_SCHEDULE = (
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)
_EXPECTED_POLICIES = {
    "phase7a_control": (INDEPENDENT_NOTE_PITCH,),
    "phase8a_mask_only": _HIERARCHY_SCHEDULE,
    "onset_only": (ONSET_PITCH_DESCENDANTS,),
    "beat_only": (BEAT_PITCH_DESCENDANTS,),
    "bar_only": (CONTIGUOUS_BAR_PITCH_SPAN,),
    "track_only": (TRACK_BAR_PITCH_SPAN,),
    "multilevel_equal_weight": _HIERARCHY_SCHEDULE,
}
_PHASE8B2_VARIANT_BINDINGS = {
    "phase7a_control": ("phase7a_control", "phase7a_control"),
    "phase8a_mask_only": ("phase7a_control", "phase8a_mask_only"),
    "onset_latent": ("onset_only", "onset_only"),
    "beat_latent": ("beat_only", "beat_only"),
    "hierarchy_bar_latent": ("bar_only", "bar_only"),
    "track_latent": ("track_only", "track_only"),
    "multilevel_equal": (
        "multilevel_equal_weight",
        "multilevel_equal_weight",
    ),
}
_ENCODER_FORWARDS_PER_POLICY_VIEW = {
    "phase7a_control": 2,
    "phase8a_mask_only": 2,
    "onset_latent": 3,
    "beat_latent": 3,
    "hierarchy_bar_latent": 3,
    "track_latent": 3,
    "multilevel_equal": 3,
}


class Phase8BEngineError(ValueError):
    """A fail-closed official Phase 8B.1 engine contract violation."""


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedPhase8BMaskingConfig:
    """Independent Phase 8A policy-mixture and fixed pass schedule binding."""

    contract_version: str
    mode: str
    policy_config: HierarchyMaskPolicyConfig
    scheduled_policies: tuple[str, ...]
    fingerprint: str

    @classmethod
    def from_hydra(cls, config: object) -> ResolvedPhase8BMaskingConfig:
        if config is None:
            raise Phase8BEngineError("phase8b.engine.masking_config_required")
        if getattr(config, "contract_version", None) != (
            PHASE8B_MASKING_CONFIG_CONTRACT_VERSION
        ):
            raise Phase8BEngineError(
                "phase8b.engine.masking_config_version_incompatible"
            )
        mode = getattr(config, "mode", None)
        if mode not in _EXPECTED_POLICIES:
            raise Phase8BEngineError("phase8b.engine.masking_mode_unknown")
        try:
            policy_config = HierarchyMaskPolicyConfig.create(
                weights={
                    policy: getattr(config, policy, None)
                    for policy in HIERARCHY_MASK_POLICIES
                },
                min_span_bars=getattr(config, "min_span_bars", None),
                max_span_bars=getattr(config, "max_span_bars", None),
                span_selection_pool_size=getattr(
                    config, "span_selection_pool_size", None
                ),
                span_budget_error_slack=getattr(
                    config, "span_budget_error_slack", None
                ),
            )
        except Exception as exc:
            raise Phase8BEngineError(
                f"phase8b.engine.masking_config_invalid:{exc}"
            ) from exc
        scheduled = policy_config.enabled_policies()
        if scheduled != _EXPECTED_POLICIES[mode] or any(
            policy_config.weight(policy) != 1.0 for policy in scheduled
        ):
            raise Phase8BEngineError(
                "phase8b.engine.masking_policy_substitution_forbidden"
            )
        payload = {
            "contract_version": PHASE8B_MASKING_CONFIG_CONTRACT_VERSION,
            "mode": mode,
            "phase8a_policy_mixture": policy_config.to_dict(),
            "scheduled_policies": list(scheduled),
            "pass_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
        }
        return cls(
            contract_version=PHASE8B_MASKING_CONFIG_CONTRACT_VERSION,
            mode=mode,
            policy_config=policy_config,
            scheduled_policies=scheduled,
            fingerprint=_fingerprint(payload),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "contract_version": self.contract_version,
            "mode": self.mode,
            "phase8a_policy_mixture": self.policy_config.to_dict(),
            "scheduled_policies": list(self.scheduled_policies),
            "pass_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
        }
        payload["fingerprint"] = self.fingerprint
        return payload

    def pass_config(self, policy: str) -> HierarchyMaskPolicyConfig:
        if policy not in self.scheduled_policies:
            raise Phase8BEngineError(
                "phase8b.engine.unscheduled_policy_requested"
            )
        return HierarchyMaskPolicyConfig.create(
            weights={policy: 1.0},
            min_span_bars=self.policy_config.min_span_bars,
            max_span_bars=self.policy_config.max_span_bars,
            span_selection_pool_size=(
                self.policy_config.span_selection_pool_size
            ),
            span_budget_error_slack=(
                self.policy_config.span_budget_error_slack
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedPhase8B2Schedule:
    """Comparison-owned repeated views without changing Phase 8B.1 policy."""

    contract_version: str
    comparison_mode: str
    variant_id: str
    protocol_fingerprint: str
    sample_schedule_fingerprint: str
    actual_sample_schedule_path: str | None
    logical_updates: int
    model_initialization_seed: int
    data_order_seed: int
    policy_views: tuple[tuple[str, int], ...]
    encoder_forwards_per_policy_view: int
    fingerprint: str

    @classmethod
    def from_config(
        cls,
        value: object,
        *,
        masking: ResolvedPhase8BMaskingConfig,
        objective_mode: str | None = None,
    ) -> ResolvedPhase8B2Schedule | None:
        if value is None:
            return None
        if getattr(value, "contract_version", None) != (
            PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION
        ):
            raise Phase8BEngineError(
                "phase8b2.engine.schedule_version_incompatible"
            )
        comparison_mode = getattr(value, "comparison_mode", None)
        if comparison_mode not in {
            "natural_schedule",
            "encoder_forward_matched",
        }:
            raise Phase8BEngineError(
                "phase8b2.engine.comparison_mode_invalid"
            )
        variant_id = getattr(value, "variant_id", None)
        protocol_fingerprint = getattr(value, "protocol_fingerprint", None)
        sample_schedule_fingerprint = getattr(
            value, "sample_schedule_fingerprint", None
        )
        actual_sample_schedule_path = getattr(
            value, "actual_sample_schedule_path", ""
        )
        logical_updates = getattr(value, "logical_updates", None)
        model_initialization_seed = getattr(
            value, "model_initialization_seed", None
        )
        data_order_seed = getattr(value, "data_order_seed", None)
        names = getattr(value, "policy_view_names", None)
        seeds = getattr(value, "policy_view_seeds", None)
        try:
            names = list(names) if names is not None else None
            seeds = list(seeds) if seeds is not None else None
        except TypeError:
            names = None
            seeds = None
        if (
            not isinstance(variant_id, str)
            or not variant_id
            or not isinstance(protocol_fingerprint, str)
            or not protocol_fingerprint
            or not isinstance(sample_schedule_fingerprint, str)
            or not sample_schedule_fingerprint
            or not isinstance(actual_sample_schedule_path, str)
            or isinstance(logical_updates, bool)
            or not isinstance(logical_updates, int)
            or logical_updates <= 0
            or isinstance(model_initialization_seed, bool)
            or not isinstance(model_initialization_seed, int)
            or model_initialization_seed < 0
            or isinstance(data_order_seed, bool)
            or not isinstance(data_order_seed, int)
            or data_order_seed < 0
            or not isinstance(names, list)
            or not names
            or not isinstance(seeds, list)
            or len(names) != len(seeds)
            or any(
                not isinstance(policy, str)
                or policy not in masking.scheduled_policies
                for policy in names
            )
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                for seed in seeds
            )
        ):
            raise Phase8BEngineError(
                "phase8b2.engine.schedule_binding_invalid"
            )
        expected_binding = _PHASE8B2_VARIANT_BINDINGS.get(variant_id)
        if expected_binding != (objective_mode, masking.mode):
            raise Phase8BEngineError(
                "phase8b2.engine.variant_binding_mismatch"
            )
        forwards_per_view = _ENCODER_FORWARDS_PER_POLICY_VIEW[variant_id]
        if comparison_mode == "natural_schedule" and tuple(names) != (
            masking.scheduled_policies
        ):
            raise Phase8BEngineError(
                "phase8b2.engine.natural_schedule_substitution_forbidden"
            )
        if actual_sample_schedule_path:
            try:
                artifact = json.loads(
                    Path(actual_sample_schedule_path).read_text(
                        encoding="utf-8"
                    )
                )
                candidates = [
                    row
                    for row in artifact["ssl"]
                    if row["data_order_seed"] == data_order_seed
                ]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise Phase8BEngineError(
                    "phase8b2.engine.actual_sample_schedule_unreadable"
                ) from exc
            if (
                artifact.get("protocol_fingerprint") != protocol_fingerprint
                or len(candidates) != 1
                or candidates[0].get("sample_schedule_fingerprint")
                != sample_schedule_fingerprint
                or candidates[0].get("logical_updates") != logical_updates
            ):
                raise Phase8BEngineError(
                    "phase8b2.engine.actual_sample_schedule_binding_mismatch"
                )
        payload = {
            "contract_version": PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION,
            "comparison_mode": comparison_mode,
            "variant_id": variant_id,
            "protocol_fingerprint": protocol_fingerprint,
            "sample_schedule_fingerprint": sample_schedule_fingerprint,
            "logical_updates": logical_updates,
            "model_initialization_seed": model_initialization_seed,
            "data_order_seed": data_order_seed,
            "policy_views": [
                {"index": index, "policy": policy, "seed": seed}
                for index, (policy, seed) in enumerate(
                    zip(names, seeds, strict=True)
                )
            ],
            "loss_renormalization": "unchanged_family_global_aggregation",
            "encoder_forwards_per_policy_view": forwards_per_view,
        }
        return cls(
            contract_version=PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION,
            comparison_mode=comparison_mode,
            variant_id=variant_id,
            protocol_fingerprint=protocol_fingerprint,
            sample_schedule_fingerprint=sample_schedule_fingerprint,
            actual_sample_schedule_path=(
                actual_sample_schedule_path or None
            ),
            logical_updates=logical_updates,
            model_initialization_seed=model_initialization_seed,
            data_order_seed=data_order_seed,
            policy_views=tuple(zip(names, seeds, strict=True)),
            encoder_forwards_per_policy_view=forwards_per_view,
            fingerprint=_fingerprint(payload),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "comparison_mode": self.comparison_mode,
            "variant_id": self.variant_id,
            "protocol_fingerprint": self.protocol_fingerprint,
            "sample_schedule_fingerprint": self.sample_schedule_fingerprint,
            "actual_sample_schedule_path": self.actual_sample_schedule_path,
            "logical_updates": self.logical_updates,
            "model_initialization_seed": self.model_initialization_seed,
            "data_order_seed": self.data_order_seed,
            "policy_views": [
                {"index": index, "policy": policy, "seed": seed}
                for index, (policy, seed) in enumerate(self.policy_views)
            ],
            "encoder_forwards_per_policy_view": (
                self.encoder_forwards_per_policy_view
            ),
            "encoder_forwards_per_batch": (
                len(self.policy_views)
                * self.encoder_forwards_per_policy_view
            ),
            "loss_renormalization": "unchanged_family_global_aggregation",
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class _Accounting:
    cpu_batch_count: int = 0
    optimizer_step_attempt_count: int = 0
    optimizer_step_applied_count: int = 0
    optimizer_step_skipped_count: int = 0
    forward_pass_count: int = 0
    encoder_forward_count: int = 0
    scheduled_policy_pass_count: int = 0
    objective_evaluation_count: int = 0
    family_view_pass_count: int = 0
    eligible_prediction_row_count: int = 0
    sample_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    primary_masked_entity_count: int = 0
    collateral_note_entity_count: int = 0
    collateral_track_entity_count: int = 0

    def add(self, other: _Accounting) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict[str, int]:
        if self.optimizer_step_attempt_count != (
            self.optimizer_step_applied_count
            + self.optimizer_step_skipped_count
        ):
            raise Phase8BEngineError(
                "phase8b.engine.optimizer_step_accounting_invalid"
            )
        result = asdict(self)
        result["optimizer_step_count"] = self.optimizer_step_applied_count
        result["total_masked_entity_count"] = (
            self.primary_masked_entity_count
            + self.collateral_note_entity_count
            + self.collateral_track_entity_count
        )
        return result


@dataclass(frozen=True, slots=True)
class _PreparedPass:
    policy: str
    batch: SSLBatch
    binding: PreparedMaskBinding
    objective_binding: object | None


_EVIDENCE_GROUP_PREFIXES = {
    "online_encoder": ("encoder.",),
    "online_local_encoder": ("encoder.local_baseline.encoder.",),
    "hierarchy_pooling": ("encoder.context_encoder.pooling.",),
    "transformer": ("encoder.context_encoder.transformer.",),
    "fusion": ("encoder.context_encoder.fusion.",),
    "decoder": ("decoder.",),
    "phase7a_bar_projector": ("bar_projector_predictor.projector.",),
    "phase7a_bar_predictor": ("bar_projector_predictor.predictor.",),
    "phase7a_song_projector": ("song_projector_predictor.projector.",),
    "phase7a_song_predictor": ("song_projector_predictor.predictor.",),
    ONSET_LATENT: (f"phase8b_latent_heads.{ONSET_LATENT}.",),
    BEAT_LATENT: (f"phase8b_latent_heads.{BEAT_LATENT}.",),
    HIERARCHY_BAR_LATENT: (
        f"phase8b_latent_heads.{HIERARCHY_BAR_LATENT}.",
    ),
    TRACK_LATENT: (f"phase8b_latent_heads.{TRACK_LATENT}.",),
}


def _evidence_parameter_groups(
    model: MaskedGraphSSLModel,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    named = tuple(model.named_parameters())
    return {
        group: tuple(
            (name, parameter)
            for name, parameter in named
            if any(name.startswith(prefix) for prefix in prefixes)
        )
        for group, prefixes in _EVIDENCE_GROUP_PREFIXES.items()
    }


def _parameter_snapshots(
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    for rows in groups.values():
        for name, parameter in rows:
            if name not in snapshots:
                snapshots[name] = parameter.detach().clone()
    return snapshots


def _expected_group_activity(
    group: str,
    *,
    objective: Phase8BObjectiveConfig,
    parameter_count: int,
) -> str:
    if group in PHASE8B_NEW_OBJECTIVE_FAMILIES:
        if parameter_count == 0:
            return "absent_old_control_model"
        return "active" if objective.weight(group) > 0.0 else "inactive"
    if group == "decoder":
        return (
            "active"
            if objective.weight(PHASE7A_NOTE_RECONSTRUCTION) > 0.0
            else "inactive"
        )
    if group in {"phase7a_bar_projector", "phase7a_bar_predictor"}:
        return (
            "active"
            if objective.weight(PHASE7A_BAR_LATENT) > 0.0
            else "inactive"
        )
    if group in {"phase7a_song_projector", "phase7a_song_predictor"}:
        return (
            "active"
            if objective.weight(PHASE7A_SONG_LATENT) > 0.0
            else "inactive"
        )
    return "required_shared_path" if group == "online_encoder" else "observed"


def _sum_scalar_tensors(
    values: list[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    if not values:
        return reference.new_zeros((), dtype=torch.float32)
    return torch.stack(
        [value.to(dtype=torch.float32) for value in values]
    ).sum()


def _max_scalar_tensors(
    values: list[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    if not values:
        return reference.new_zeros((), dtype=torch.float32)
    return torch.stack(
        [value.to(dtype=torch.float32) for value in values]
    ).max()


def _optimization_step_evidence(
    model: MaskedGraphSSLModel,
    *,
    objective: Phase8BObjectiveConfig,
    groups: dict[str, tuple[tuple[str, torch.nn.Parameter], ...]],
    snapshots: dict[str, torch.Tensor],
    scaler_enabled: bool,
    scale_before: float,
    scale_after: float,
    optimizer_step_applied: bool,
) -> dict[str, object]:
    """Pack finite-gradient and exact parameter-update evidence once."""

    reference = next(model.parameters()).detach()
    packed_values: list[torch.Tensor] = []
    layouts: list[tuple[str, int, int]] = []
    python_counts: dict[str, tuple[int, int]] = {}
    for group, rows in groups.items():
        gradients = tuple(
            parameter.grad.detach()
            for _name, parameter in rows
            if parameter.grad is not None
        )
        finite_flags = [torch.isfinite(gradient).all() for gradient in gradients]
        nonzero_flags = [
            (torch.isfinite(gradient) & gradient.ne(0)).any()
            for gradient in gradients
        ]
        nonfinite_elements = [
            (~torch.isfinite(gradient)).count_nonzero()
            for gradient in gradients
        ]
        finite_nonzero_elements = [
            (torch.isfinite(gradient) & gradient.ne(0)).count_nonzero()
            for gradient in gradients
        ]
        gradient_maxima = [
            torch.where(
                torch.isfinite(gradient),
                gradient.abs(),
                torch.zeros_like(gradient),
            ).max()
            for gradient in gradients
        ]
        changed_flags = []
        changed_elements = []
        update_maxima = []
        finite_parameter_flags = []
        for name, parameter in rows:
            current = parameter.detach()
            difference = current - snapshots[name]
            changed_flags.append(difference.ne(0).any())
            changed_elements.append(difference.ne(0).count_nonzero())
            update_maxima.append(difference.abs().max())
            finite_parameter_flags.append(torch.isfinite(current).all())
        start = len(packed_values)
        packed_values.extend(
            (
                _sum_scalar_tensors(finite_flags, reference),
                _sum_scalar_tensors(nonzero_flags, reference),
                _sum_scalar_tensors(nonfinite_elements, reference),
                _sum_scalar_tensors(finite_nonzero_elements, reference),
                _max_scalar_tensors(gradient_maxima, reference),
                _sum_scalar_tensors(changed_flags, reference),
                _sum_scalar_tensors(changed_elements, reference),
                _max_scalar_tensors(update_maxima, reference),
                _sum_scalar_tensors(finite_parameter_flags, reference),
            )
        )
        layouts.append((group, start, len(packed_values)))
        python_counts[group] = (len(rows), len(gradients))
    packed_device = torch.stack(packed_values)
    packed = packed_device.to(device="cpu", dtype=torch.float64)
    group_evidence: dict[str, object] = {}
    for group, start, _end in layouts:
        parameter_count, with_gradient_count = python_counts[group]
        values = packed[start : start + 9]
        group_evidence[group] = {
            "expected_activity": _expected_group_activity(
                group,
                objective=objective,
                parameter_count=parameter_count,
            ),
            "parameter_count": parameter_count,
            "with_gradient_count": with_gradient_count,
            "finite_gradient_count": int(values[0]),
            "nonzero_gradient_count": int(values[1]),
            "nonfinite_gradient_element_count": int(values[2]),
            "finite_nonzero_gradient_element_count": int(values[3]),
            "maximum_finite_absolute_gradient": float(values[4]),
            "changed_parameter_count": int(values[5]),
            "changed_element_count": int(values[6]),
            "maximum_absolute_parameter_update": float(values[7]),
            "finite_parameter_count_after_step": int(values[8]),
        }
    failures: list[str] = []
    if not optimizer_step_applied:
        failures.append("optimizer_step_not_applied")
    encoder = group_evidence["online_encoder"]
    if (
        encoder["with_gradient_count"] <= 0
        or encoder["finite_gradient_count"]
        != encoder["with_gradient_count"]
        or encoder["nonzero_gradient_count"] <= 0
        or encoder["changed_parameter_count"] <= 0
    ):
        failures.append("online_encoder_finite_nonzero_update_missing")
    for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
        row = group_evidence[family]
        if objective.weight(family) > 0.0:
            if (
                row["parameter_count"] <= 0
                or row["with_gradient_count"] != row["parameter_count"]
                or row["finite_gradient_count"] != row["with_gradient_count"]
                or row["nonzero_gradient_count"] <= 0
                or row["changed_parameter_count"] <= 0
            ):
                failures.append(f"active_head_invalid:{family}")
        elif (
            row["with_gradient_count"] != 0
            or row["changed_parameter_count"] != 0
        ):
            failures.append(f"inactive_head_changed:{family}")
    legacy_groups = (
        "decoder",
        "phase7a_bar_projector",
        "phase7a_bar_predictor",
        "phase7a_song_projector",
        "phase7a_song_predictor",
    )
    for group in legacy_groups:
        row = group_evidence[group]
        expected_active = row["expected_activity"] == "active"
        if expected_active and (
            row["with_gradient_count"] <= 0
            or row["finite_gradient_count"] != row["with_gradient_count"]
            or row["nonzero_gradient_count"] <= 0
            or row["changed_parameter_count"] <= 0
        ):
            failures.append(f"active_legacy_path_invalid:{group}")
        if not expected_active and (
            row["with_gradient_count"] != 0
            or row["changed_parameter_count"] != 0
        ):
            failures.append(f"inactive_legacy_path_changed:{group}")
    if objective.mode == "phase7a_control":
        for group in (
            "online_local_encoder",
            "hierarchy_pooling",
            "transformer",
            "fusion",
        ):
            row = group_evidence[group]
            if (
                row["with_gradient_count"] <= 0
                or row["finite_gradient_count"]
                != row["with_gradient_count"]
                or row["nonzero_gradient_count"] <= 0
                or row["changed_parameter_count"] <= 0
            ):
                failures.append(f"phase7a_control_path_invalid:{group}")
    return {
        "contract_version": PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION,
        "optimizer_step_applied": optimizer_step_applied,
        "scaler": {
            "enabled": scaler_enabled,
            "scale_before": scale_before,
            "scale_after": scale_after,
            "step_skipped_by_scaler": not optimizer_step_applied,
        },
        "groups": group_evidence,
        "packed_host_materialization_count": 1,
        "packed_device_to_host_transfer_count": int(
            packed_device.device.type == "cuda"
        ),
        "retained_cuda_tensor_count": 0,
        "retained_prediction_tensor_count": 0,
        "acceptance": {
            "passed": not failures,
            "failures": failures,
        },
    }


def _optimizer_parameter_coverage(
    model: MaskedGraphSSLModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    named = tuple(model.named_parameters())
    optimizer_parameters = tuple(
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    )
    optimizer_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    optimizer_id_set = set(optimizer_ids)
    trainable = tuple(
        (name, parameter) for name, parameter in named if parameter.requires_grad
    )
    groups = _evidence_parameter_groups(model)
    return {
        "optimizer_parameter_group_count": len(optimizer.param_groups),
        "optimizer_parameter_count": len(optimizer_parameters),
        "duplicate_optimizer_parameter_count": (
            len(optimizer_ids) - len(optimizer_id_set)
        ),
        "trainable_parameter_count": len(trainable),
        "missing_trainable_parameter_count": sum(
            id(parameter) not in optimizer_id_set
            for _name, parameter in trainable
        ),
        "all_trainable_parameters_present_exactly_once": (
            len(optimizer_ids) == len(optimizer_id_set)
            and all(
                id(parameter) in optimizer_id_set
                for _name, parameter in trainable
            )
        ),
        "evidence_groups": {
            group: {
                "parameter_count": len(rows),
                "all_in_optimizer": all(
                    id(parameter) in optimizer_id_set
                    for _name, parameter in rows
                ),
            }
            for group, rows in groups.items()
        },
    }


def _model_state_fingerprint(model: torch.nn.Module) -> str:
    digest = sha256()
    for name, value in model.state_dict().items():
        detached = value.detach().to(device="cpu").contiguous()
        digest.update(
            json.dumps(
                {
                    "name": name,
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _encoder_state_fingerprint(model: torch.nn.Module) -> str:
    export = export_pretrained_encoder_state(model)
    state = export["encoder_state"]
    assert isinstance(state, dict)
    digest = sha256()
    for name in sorted(state):
        value = state[name]
        assert isinstance(value, torch.Tensor)
        detached = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _update_tensor_digest(
    digest: object, label: str, value: torch.Tensor
) -> None:
    detached = value.detach().to(device="cpu").contiguous()
    digest.update(label.encode("utf-8"))
    digest.update(str(tuple(detached.shape)).encode("ascii"))
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes())


def _prediction_fingerprint(output: object) -> str:
    """Hash actual prediction tensors without retaining them in metrics."""

    digest = sha256()
    base = getattr(output, "base_output", output)
    for index, value in enumerate(getattr(base, "decoder_predictions", ())):
        _update_tensor_digest(digest, f"decoder/{index}", value)
    for name in ("bar_latent", "song_latent"):
        value = getattr(base, name, None)
        if value is not None:
            _update_tensor_digest(digest, name, value.prediction)
    for index, value in enumerate(getattr(output, "latent_predictions", ())):
        _update_tensor_digest(
            digest, f"latent/{index}/{value.family}", value.prediction
        )
    return digest.hexdigest()


def _gradient_fingerprint(model: torch.nn.Module) -> str:
    digest = sha256()
    present = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            continue
        present += 1
        _update_tensor_digest(digest, name, parameter.grad)
    if not present:
        raise Phase8BEngineError(
            "phase8b.engine.gradient_fingerprint_empty"
        )
    return digest.hexdigest()


def _read_initial_encoder_fingerprint(output: Path) -> str:
    try:
        payload = json.loads(
            (output / "fingerprints.json").read_text(encoding="utf-8")
        )
        value = payload["initial_encoder_state_fingerprint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Phase8BEngineError(
            "phase8b.engine.initial_encoder_fingerprint_unreadable"
        ) from exc
    if not isinstance(value, str) or len(value) != 64:
        raise Phase8BEngineError(
            "phase8b.engine.initial_encoder_fingerprint_invalid"
        )
    return value


def _journal_train_evidence_fingerprint(
    journal: tuple[dict[str, object], ...], field: str
) -> str:
    return _fingerprint(
        {
            "contract_version": PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION,
            "stage": "train",
            "field": field,
            "epochs": [row["train"][field] for row in journal],
        }
    )


def _input_batch_fingerprint(batch: SSLBatch) -> str:
    return _fingerprint(
        {
            "dataset_ids": list(batch.dataset_ids),
            "piece_ids": list(batch.piece_ids),
            "sample_count": batch.sample_count,
            "node_count": batch.node_count,
            "edge_count": batch.edge_count,
        }
    )


def _cuda_peak_memory(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "available": False,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "available": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _one_batch_mechanics_acceptance(
    report: dict[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    accounting = report["accounting"]
    if accounting["optimizer_step_attempt_count"] != (
        accounting["optimizer_step_applied_count"]
        + accounting["optimizer_step_skipped_count"]
    ):
        failures.append("optimizer_step_accounting_inconsistent")
    if accounting["optimizer_step_applied_count"] <= 0:
        failures.append("no_optimizer_step_applied")
    if not report["loss_decreased"]:
        failures.append("bounded_loss_did_not_decrease")
    gradient = report["gradient_coverage"]
    if (
        not isinstance(gradient, dict)
        or not gradient.get("acceptance", {}).get("passed", False)
    ):
        failures.append("finite_nonzero_gradient_or_update_evidence_missing")
    fingerprints = report["model_state_fingerprints"]
    if fingerprints["initial"] == fingerprints["final"]:
        failures.append("model_state_unchanged")
    if report["initial"]["input_batch_fingerprints"] != report["final"][
        "input_batch_fingerprints"
    ]:
        failures.append("initial_final_input_fixture_mismatch")
    final_loss = report["final"]["total_ssl_loss"]
    if final_loss is None or not math.isfinite(float(final_loss)):
        failures.append("final_loss_nonfinite_or_unavailable")
    coverage = report["optimizer_parameter_coverage"]
    if not coverage["all_trainable_parameters_present_exactly_once"]:
        failures.append("optimizer_parameter_coverage_incomplete")
    return {
        "contract_version": PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION,
        "failure_closed": True,
        "passed": not failures,
        "failures": failures,
    }


def _validate_pair(
    objective: Phase8BObjectiveConfig,
    masking: ResolvedPhase8BMaskingConfig,
) -> str:
    if objective.mode == "phase7a_control":
        if masking.mode == "phase7a_control":
            return "phase7a_control"
        if masking.mode == "phase8a_mask_only":
            return "phase8a_mask_only"
        raise Phase8BEngineError(
            "phase8b.engine.objective_masking_mode_incompatible"
        )
    if objective.mode != masking.mode:
        raise Phase8BEngineError(
            "phase8b.engine.objective_masking_mode_incompatible"
        )
    return objective.mode


def _materialize(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    Phase8BObjectiveConfig,
    ResolvedPhase8BMaskingConfig,
    str,
    ResolvedPhase8B2Schedule | None,
]:
    from music_critic.ssl.engine import _validate_config

    _validate_config(config)
    if config.get("phase8b_objective") is None:
        raise Phase8BEngineError("phase8b.engine.objective_config_required")
    try:
        objective = Phase8BObjectiveConfig.from_hydra(
            OmegaConf.create(config["phase8b_objective"])
        )
    except Exception as exc:
        raise Phase8BEngineError(
            f"phase8b.engine.objective_config_invalid:{exc}"
        ) from exc
    masking = ResolvedPhase8BMaskingConfig.from_hydra(
        OmegaConf.create(config.get("phase8b_masking"))
        if config.get("phase8b_masking") is not None
        else None
    )
    execution_mode = _validate_pair(objective, masking)
    comparison = ResolvedPhase8B2Schedule.from_config(
        OmegaConf.create(config["phase8b2_schedule"])
        if config.get("phase8b2_schedule") is not None
        else None,
        masking=masking,
        objective_mode=objective.mode,
    )
    materialized = copy.deepcopy(config)
    materialized["phase8b_runtime"] = {
        "engine_contract_version": PHASE8B_ENGINE_CONTRACT_VERSION,
        "checkpoint_binding_contract_version": (
            PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
        ),
        "scheduled_view_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
        "amp_compute_contract": PHASE8B_AMP_COMPUTE_CONTRACT,
        "optimizer_evidence_contract_version": (
            PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION
        ),
        "grad_scaler_initial_scale": PHASE8B_GRAD_SCALER_INITIAL_SCALE,
        "execution_mode": execution_mode,
        "objective_registry_fingerprint": (
            PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
        ),
        "objective_config": objective.to_dict(),
        "objective_config_fingerprint": objective.fingerprint,
        "active_objective_families": [
            family
            for family, weight in objective.family_weights
            if weight > 0.0
        ],
        "active_objective_weights": [
            [family, weight]
            for family, weight in objective.family_weights
            if weight > 0.0
        ],
        "masking_config": masking.to_dict(),
        "masking_config_fingerprint": masking.fingerprint,
        "mask_policy_mixture_fingerprint": masking.policy_config.fingerprint,
        "phase8b2_schedule": (
            None if comparison is None else comparison.to_dict()
        ),
    }
    return materialized, objective, masking, execution_mode, comparison


def _prepare(
    config: dict[str, Any],
) -> tuple[
    Path,
    torch.device,
    SSLDataRuntime,
    MaskedGraphSSLModel,
    torch.optim.Optimizer,
    Any,
    torch.amp.GradScaler,
    Phase8BObjectiveConfig,
    ResolvedPhase8BMaskingConfig,
    str,
    dict[str, Any],
]:
    from music_critic.ssl.engine import (
        _optimizer,
        _resolve_device,
        _scheduler,
        _set_determinism,
    )

    (
        resolved,
        objective,
        masking,
        execution_mode,
        comparison,
    ) = _materialize(config)
    if comparison is None:
        _set_determinism(int(resolved["seed"]))
        data_seed = int(resolved["seed"])
    else:
        _set_determinism(comparison.data_order_seed)
        data_seed = comparison.data_order_seed
    device = _resolve_device(resolved)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime = build_ssl_data_runtime(
        OmegaConf.create(resolved["data"]), seed=data_seed
    )
    if comparison is not None:
        _set_determinism(comparison.model_initialization_seed)
    try:
        model = build_phase8b_model_from_config(
            OmegaConf.create(resolved["model"]),
            OmegaConf.create(resolved["ssl"]),
            OmegaConf.create(resolved["phase8b_objective"]),
        ).to(device)
    except Exception as exc:
        raise Phase8BEngineError(
            f"phase8b.engine.model_build_incompatible:{exc}"
        ) from exc
    if execution_mode in {"phase7a_control", "phase8a_mask_only"}:
        if type(model) is not MaskedGraphSSLModel:
            raise Phase8BEngineError(
                "phase8b.engine.old_control_model_contract_mismatch"
            )
    elif type(model) is not Phase8BMultilevelSSLModel:
        raise Phase8BEngineError(
            "phase8b.engine.multilevel_model_contract_mismatch"
        )
    resolved["phase8b_runtime"]["model_class"] = type(model).__name__
    optimizer = _optimizer(model, resolved)
    scheduler = _scheduler(optimizer, resolved)
    scaler = torch.amp.GradScaler(
        device.type,
        init_scale=PHASE8B_GRAD_SCALER_INITIAL_SCALE,
        enabled=bool(resolved["device"]["amp"]),
    )
    return (
        Path(resolved["output_dir"]).resolve(),
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        objective,
        masking,
        execution_mode,
        resolved,
    )


def _binding_summary(binding: PreparedMaskBinding) -> tuple[int, int, int]:
    primary = sum(
        len(plan.selected_local_node_indices) for plan in binding.mask_plans
    )
    collateral_note = sum(
        len(row.local_node_indices)
        for plan in binding.mask_plans
        for row in plan.collateral_feature_masks
        if row.node_type == "note"
    )
    collateral_track = sum(
        len(row.local_node_indices)
        for plan in binding.mask_plans
        for row in plan.collateral_feature_masks
        if row.node_type == "track"
    )
    return primary, collateral_note, collateral_track


def _prepare_pass(
    cpu_batch: SSLBatch,
    *,
    policy: str,
    model: MaskedGraphSSLModel,
    objective: Phase8BObjectiveConfig,
    masking: ResolvedPhase8BMaskingConfig,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    stage: str,
    view_seed: int | None = None,
) -> _PreparedPass:
    binding = prepare_hierarchy_mask_binding(
        cpu_batch,
        policy_config=masking.pass_config(policy),
        global_seed=(
            int(config["seed"]) if view_seed is None else view_seed
        ),
        epoch=0 if stage == "validation" else epoch,
        requested_mask_rate=model.ssl_config.mask_rate,
        stage=stage,
    )
    objective_binding = None
    if type(model) is Phase8BMultilevelSSLModel:
        if type(binding) is not PreparedHierarchyMaskBinding:
            raise Phase8BEngineError(
                "phase8b.engine.hierarchy_binding_required"
            )
        objective_binding = prepare_phase8b_objective_binding(
            binding, objective
        )
    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        cpu_batch,
        binding,
        device,
        non_blocking=bool(config["device"]["non_blocking"]),
    )
    if not isinstance(moved_batch, SSLBatch):
        raise Phase8BEngineError("phase8b.engine.moved_batch_invalid")
    return _PreparedPass(
        policy=policy,
        batch=moved_batch,
        binding=moved_binding,
        objective_binding=objective_binding,
    )


def _forward(
    model: MaskedGraphSSLModel,
    prepared: _PreparedPass,
    *,
    execution_mode: str,
) -> (
    SSLForwardOutput
    | Phase8AHierarchySSLForwardOutput
    | Phase8BMultilevelSSLForwardOutput
):
    if execution_mode == "phase7a_control":
        if type(prepared.binding) is not PreparedMaskBinding:
            raise Phase8BEngineError(
                "phase8b.engine.phase7a_binding_contract_mismatch"
            )
        return model(
            prepared.batch, prepared_mask_binding=prepared.binding
        )
    if execution_mode == "phase8a_mask_only":
        if type(model) is not MaskedGraphSSLModel or type(
            prepared.binding
        ) is not PreparedHierarchyMaskBinding:
            raise Phase8BEngineError(
                "phase8b.engine.mask_only_forward_contract_mismatch"
            )
        return model.forward_hierarchy(
            prepared.batch, prepared_mask_binding=prepared.binding
        )
    if (
        type(model) is not Phase8BMultilevelSSLModel
        or type(prepared.binding) is not PreparedHierarchyMaskBinding
        or prepared.objective_binding is None
    ):
        raise Phase8BEngineError(
            "phase8b.engine.multilevel_forward_contract_mismatch"
        )
    return model.forward_multilevel(
        prepared.batch,
        prepared_mask_binding=prepared.binding,
        prepared_objective_binding=prepared.objective_binding,
    )


def _loss(output: object) -> torch.Tensor | None:
    objective = getattr(output, "objective", None)
    loss = getattr(objective, "total_loss", None)
    if loss is not None and (
        not isinstance(loss, torch.Tensor) or loss.ndim != 0
    ):
        raise Phase8BEngineError("phase8b.engine.total_loss_invalid")
    return loss


def _stage(
    model: MaskedGraphSSLModel,
    loader: Iterable[SSLBatch],
    *,
    objective: Phase8BObjectiveConfig,
    masking: ResolvedPhase8BMaskingConfig,
    execution_mode: str,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    collect_gradient_evidence: bool = False,
    comparison: ResolvedPhase8B2Schedule | None = None,
    maximum_batches: int | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    training = optimizer is not None
    if training != (scaler is not None) or stage not in {"train", "validation"}:
        raise Phase8BEngineError("phase8b.engine.stage_arguments_invalid")
    model.train(training)
    accumulator = Phase8BObjectiveAccumulator(objective)
    accounting = _Accounting()
    available_batch_count = 0
    binding_fingerprints: list[str] = []
    objective_binding_fingerprints: list[str] = []
    plan_fingerprints: list[str] = []
    input_batch_fingerprints: list[str] = []
    input_sample_identities: list[list[str]] = []
    prediction_fingerprints: list[str] = []
    gradient_fingerprints: list[str] = []
    scheduled_views = (
        tuple((policy, None) for policy in masking.scheduled_policies)
        if comparison is None
        else tuple(comparison.policy_views)
    )
    policy_pass_counts = {policy: 0 for policy, _ in scheduled_views}
    gradient_evidence = None
    collected_step_evidence: list[dict[str, object]] = []
    scaler_scale_before_first: float | None = None
    scaler_scale_after_last: float | None = None
    scaler_minimum_scale: float | None = None
    scaler_maximum_scale: float | None = None
    scaler_scale_decrease_count = 0
    scaler_scale_non_decrease_count = 0
    encoder_invocations = [0]

    original_encode_prepared = model.encoder._encode_prepared

    def _counted_encode_prepared(*args: object, **kwargs: object) -> object:
        encoder_invocations[0] += 1
        return original_encode_prepared(*args, **kwargs)

    setattr(model.encoder, "_encode_prepared", _counted_encode_prepared)
    for batch_index, cpu_batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        accounting.cpu_batch_count += 1
        accounting.sample_count += cpu_batch.sample_count
        accounting.node_count += cpu_batch.node_count
        accounting.edge_count += cpu_batch.edge_count
        input_batch_fingerprints.append(_input_batch_fingerprint(cpu_batch))
        input_sample_identities.extend(
            [dataset_id, piece_id]
            for dataset_id, piece_id in zip(
                cpu_batch.dataset_ids, cpu_batch.piece_ids, strict=True
            )
        )
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        policy_outputs: list[
            tuple[
                str,
                SSLForwardOutput
                | Phase8AHierarchySSLForwardOutput
                | Phase8BMultilevelSSLForwardOutput,
            ]
        ] = []
        for view_index, (policy, view_seed) in enumerate(scheduled_views):
            accounting.scheduled_policy_pass_count += 1
            policy_pass_counts[policy] += 1
            prepared = _prepare_pass(
                cpu_batch,
                policy=policy,
                model=model,
                objective=objective,
                masking=masking,
                config=config,
                device=device,
                epoch=epoch,
                stage=stage,
                view_seed=view_seed,
            )
            primary, collateral_note, collateral_track = _binding_summary(
                prepared.binding
            )
            accounting.primary_masked_entity_count += primary
            accounting.collateral_note_entity_count += collateral_note
            accounting.collateral_track_entity_count += collateral_track
            binding_fingerprints.append(prepared.binding.fingerprint)
            if prepared.objective_binding is not None:
                objective_binding_fingerprint = getattr(
                    prepared.objective_binding, "fingerprint", None
                )
                if not isinstance(objective_binding_fingerprint, str):
                    raise Phase8BEngineError(
                        "phase8b.engine.objective_binding_fingerprint_missing"
                    )
                objective_binding_fingerprints.append(
                    objective_binding_fingerprint
                )
            plan_fingerprints.extend(
                plan.fingerprint for plan in prepared.binding.mask_plans
            )
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(config["device"]["amp"]),
                    dtype=(
                        torch.float16
                        if config["device"].get("amp_dtype", "float16")
                        == "float16"
                        else torch.bfloat16
                    ),
                ):
                    output = _forward(
                        model, prepared, execution_mode=execution_mode
                    )
            accounting.forward_pass_count += 1
            observed_forwards = encoder_invocations[0] - (
                accounting.encoder_forward_count
            )
            expected_forwards = (
                comparison.encoder_forwards_per_policy_view
                if comparison is not None
                else (
                    2
                    if execution_mode
                    in {"phase7a_control", "phase8a_mask_only"}
                    else 3
                )
            )
            if observed_forwards != expected_forwards:
                raise Phase8BEngineError(
                    "phase8b.engine.instrumented_encoder_forward_mismatch:"
                    f"expected={expected_forwards},observed={observed_forwards}"
                )
            accounting.encoder_forward_count += observed_forwards
            accounting.objective_evaluation_count += 1
            prediction_fingerprints.append(
                _prediction_fingerprint(output)
            )
            policy_outputs.append(
                (
                    policy
                    if comparison is None
                    else f"{policy}#view={view_index}",
                    output,
                )
            )
        batch_objective = aggregate_phase8b_policy_pass_losses(
            tuple(policy_outputs), objective_config=objective
        )
        accounting.family_view_pass_count += (
            batch_objective.family_view_pass_count
        )
        accounting.eligible_prediction_row_count += (
            batch_objective.eligible_prediction_row_count
        )
        current_loss = batch_objective.total_loss
        if current_loss is not None:
            available_batch_count += 1
        if training and current_loss is not None:
            assert optimizer is not None and scaler is not None
            accounting.optimizer_step_attempt_count += 1
            scale_before = float(scaler.get_scale())
            if scaler_scale_before_first is None:
                scaler_scale_before_first = scale_before
            scaler.scale(current_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["optimizer"]["gradient_clip_norm"]),
            )
            gradient_fingerprints.append(_gradient_fingerprint(model))
            should_collect = (
                collect_gradient_evidence and gradient_evidence is None
            )
            groups = (
                _evidence_parameter_groups(model)
                if should_collect
                else None
            )
            snapshots = (
                _parameter_snapshots(groups)
                if groups is not None
                else None
            )
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            scaler_scale_after_last = scale_after
            scaler_minimum_scale = min(
                scale_before,
                scale_after,
                scaler_minimum_scale
                if scaler_minimum_scale is not None
                else scale_before,
            )
            scaler_maximum_scale = max(
                scale_before,
                scale_after,
                scaler_maximum_scale
                if scaler_maximum_scale is not None
                else scale_before,
            )
            step_skipped = scaler.is_enabled() and scale_after < scale_before
            if step_skipped:
                accounting.optimizer_step_skipped_count += 1
                scaler_scale_decrease_count += 1
            else:
                accounting.optimizer_step_applied_count += 1
                scaler_scale_non_decrease_count += 1
            if groups is not None and snapshots is not None:
                step_evidence = _optimization_step_evidence(
                    model,
                    objective=objective,
                    groups=groups,
                    snapshots=snapshots,
                    scaler_enabled=scaler.is_enabled(),
                    scale_before=scale_before,
                    scale_after=scale_after,
                    optimizer_step_applied=not step_skipped,
                )
                collected_step_evidence.append(step_evidence)
                if not step_skipped:
                    gradient_evidence = step_evidence
                del snapshots, groups
        elif training and comparison is not None:
            # A comparison logical update with zero eligible rows still
            # consumed its fixed raw/forward budget and is an explicit skip.
            accounting.optimizer_step_attempt_count += 1
            accounting.optimizer_step_skipped_count += 1
        # Metrics materialize only after backward/optimizer handling.  The
        # differentiable family-global loss itself is never detached, moved,
        # converted to Python, or replaced before backward.
        try:
            accumulator.update_batch(batch_objective)
        except ValueError as exc:
            if "non-finite" in str(exc):
                raise Phase8BEngineError(
                    "phase8b.engine.nonfinite_total_loss"
                ) from exc
            raise
        del policy_outputs, batch_objective, current_loss
    setattr(model.encoder, "_encode_prepared", original_encode_prepared)
    aggregate = accumulator.finalize()
    metric: dict[str, object] = {
        "engine_contract_version": PHASE8B_ENGINE_CONTRACT_VERSION,
        "stage": stage,
        "epoch_seed_coordinate": 0 if stage == "validation" else epoch,
        "batch_count": accounting.cpu_batch_count,
        "available_batch_count": available_batch_count,
        "skipped_or_unavailable_batch_count": (
            accounting.cpu_batch_count - available_batch_count
        ),
        "sample_count": accounting.sample_count,
        "node_count": accounting.node_count,
        "edge_count": accounting.edge_count,
        "total_ssl_loss": aggregate["total_loss"],
        "total_unavailable_reason": aggregate["unavailable_reason"],
        "objective": aggregate,
        "accounting": accounting.to_dict(),
        "resolved_policies": [policy for policy, _ in scheduled_views],
        "policy_pass_counts": policy_pass_counts,
        "prepared_binding_fingerprints": sorted(binding_fingerprints),
        "prepared_objective_binding_fingerprints": sorted(
            objective_binding_fingerprints
        ),
        "mask_plan_fingerprints": sorted(plan_fingerprints),
        "input_batch_fingerprints": input_batch_fingerprints,
        "input_sample_identities": input_sample_identities,
        "prediction_fingerprints": prediction_fingerprints,
        "gradient_fingerprints": gradient_fingerprints,
        "validation_schedule_batch_partition_independent_coordinates": (
            stage == "validation"
        ),
        "objective_config_fingerprint": objective.fingerprint,
        "masking_config_fingerprint": masking.fingerprint,
        "mask_policy_mixture_fingerprint": masking.policy_config.fingerprint,
        "phase8b2_schedule_fingerprint": (
            None if comparison is None else comparison.fingerprint
        ),
        "retained_cuda_tensor_count": aggregate[
            "retained_cuda_tensor_count"
        ],
        "retained_prediction_tensor_count": aggregate[
            "retained_prediction_tensor_count"
        ],
        "metrics_transfer": {
            "packed_host_materialization_count": aggregate[
                "packed_host_materialization_count"
            ],
            "packed_device_to_host_transfer_count": aggregate[
                "packed_device_to_host_transfer_count"
            ],
            "maximum_packed_d2h_transfers_per_cpu_batch": aggregate[
                "maximum_packed_d2h_transfers_per_cpu_batch"
            ],
            "retained_cuda_tensor_count": aggregate[
                "retained_cuda_tensor_count"
            ],
            "retained_prediction_tensor_count": aggregate[
                "retained_prediction_tensor_count"
            ],
        },
        "amp_scaler_evidence": {
            "enabled": bool(scaler is not None and scaler.is_enabled()),
            "public_scale_api": "torch.amp.GradScaler.get_scale",
            "initial_scale_contract": PHASE8B_GRAD_SCALER_INITIAL_SCALE,
            "scale_before_first_attempt": scaler_scale_before_first,
            "scale_after_last_attempt": scaler_scale_after_last,
            "minimum_observed_scale": scaler_minimum_scale,
            "maximum_observed_scale": scaler_maximum_scale,
            "scale_decrease_skip_count": scaler_scale_decrease_count,
            "scale_non_decrease_applied_count": (
                scaler_scale_non_decrease_count
            ),
        },
        "optimizer_step_evidence": collected_step_evidence,
        "optimizer_evidence_transfer": {
            "packed_host_materialization_count": sum(
                int(row["packed_host_materialization_count"])
                for row in collected_step_evidence
            ),
            "packed_device_to_host_transfer_count": sum(
                int(row["packed_device_to_host_transfer_count"])
                for row in collected_step_evidence
            ),
            "maximum_packed_d2h_transfers_per_collected_cpu_batch": (
                1
                if any(
                    row["packed_device_to_host_transfer_count"]
                    for row in collected_step_evidence
                )
                else 0
            ),
            "retained_cuda_tensor_count": 0,
            "retained_prediction_tensor_count": 0,
        },
    }
    return metric, gradient_evidence


def _phase8b_bindings(
    config: dict[str, Any], model: MaskedGraphSSLModel
) -> dict[str, object]:
    runtime = config["phase8b_runtime"]
    return {
        "engine_contract_version": runtime["engine_contract_version"],
        "checkpoint_binding_contract_version": runtime[
            "checkpoint_binding_contract_version"
        ],
        "scheduled_view_aggregation": runtime[
            "scheduled_view_aggregation"
        ],
        "amp_compute_contract": runtime["amp_compute_contract"],
        "optimizer_evidence_contract_version": runtime[
            "optimizer_evidence_contract_version"
        ],
        "grad_scaler_initial_scale": runtime[
            "grad_scaler_initial_scale"
        ],
        "execution_mode": runtime["execution_mode"],
        "model_class": type(model).__name__,
        "objective_registry_fingerprint": runtime[
            "objective_registry_fingerprint"
        ],
        "objective_config_fingerprint": runtime[
            "objective_config_fingerprint"
        ],
        "active_objective_families": runtime[
            "active_objective_families"
        ],
        "active_objective_weights": runtime["active_objective_weights"],
        "masking_config_fingerprint": runtime[
            "masking_config_fingerprint"
        ],
        "mask_policy_mixture_fingerprint": runtime[
            "mask_policy_mixture_fingerprint"
        ],
        "phase8b2_schedule": runtime["phase8b2_schedule"],
    }


def _write_initial_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: SSLDataRuntime,
    model: MaskedGraphSSLModel,
    *,
    initial_validation: dict[str, object] | None = None,
) -> None:
    from music_critic.ssl.engine import (
        _checkpoint_config,
        _write_json_atomic,
    )

    resolved = copy.deepcopy(config)
    checkpoint_binding = ssl_checkpoint_metadata(
        model,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    fingerprints = {
        "data": runtime.fingerprints,
        "initial_encoder_state_fingerprint": (
            _encoder_state_fingerprint(model)
        ),
        "model_contract_fingerprint": _fingerprint(
            model.ssl_contract_metadata()
        ),
        "checkpoint_binding": checkpoint_binding,
        "phase8b_bindings": _phase8b_bindings(config, model),
    }
    artifacts = {
        "resolved_config.json": _fingerprint(resolved),
        "fingerprints.json": _fingerprint(fingerprints),
    }
    if initial_validation is not None:
        artifacts["initial_validation.json"] = _fingerprint(
            initial_validation
        )
    manifest = {
        "run_manifest_version": PHASE8B_RUN_MANIFEST_VERSION,
        "artifact_fingerprints": artifacts,
        "checkpoint_binding": checkpoint_binding,
        "phase8b_bindings": _phase8b_bindings(config, model),
    }
    _write_json_atomic(output / "resolved_config.json", resolved)
    _write_json_atomic(output / "fingerprints.json", fingerprints)
    if initial_validation is not None:
        _write_json_atomic(
            output / "initial_validation.json", initial_validation
        )
    _write_json_atomic(output / "run_manifest.json", manifest)


def _validate_resume_artifacts(
    output: Path,
    config: dict[str, Any],
    runtime: SSLDataRuntime,
    model: MaskedGraphSSLModel,
) -> None:
    from music_critic.ssl.engine import _checkpoint_config, _read_json

    manifest = _read_json(output / "run_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {
        "run_manifest_version",
        "artifact_fingerprints",
        "checkpoint_binding",
        "phase8b_bindings",
    }:
        raise Phase8BEngineError("phase8b.engine.run_manifest_invalid")
    if manifest["run_manifest_version"] != PHASE8B_RUN_MANIFEST_VERSION:
        raise Phase8BEngineError(
            "phase8b.engine.run_manifest_version_incompatible"
        )
    expected_binding = ssl_checkpoint_metadata(
        model,
        resolved_config=_checkpoint_config(config),
        data_fingerprints=runtime.fingerprints,
    )
    if _fingerprint(manifest["checkpoint_binding"]) != _fingerprint(
        expected_binding
    ):
        raise Phase8BEngineError(
            "phase8b.engine.run_manifest_checkpoint_binding_mismatch"
        )
    if manifest["phase8b_bindings"] != _phase8b_bindings(config, model):
        raise Phase8BEngineError(
            "phase8b.engine.run_manifest_phase8b_binding_mismatch"
        )
    artifacts = manifest["artifact_fingerprints"]
    names = {
        "resolved_config.json",
        "fingerprints.json",
        "initial_validation.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise Phase8BEngineError(
            "phase8b.engine.run_manifest_artifacts_invalid"
        )
    for name in sorted(names):
        if _fingerprint(_read_json(output / name)) != artifacts[name]:
            raise Phase8BEngineError(
                f"phase8b.engine.artifact_fingerprint_mismatch:{name}"
            )


def _report_common(
    *,
    config: dict[str, Any],
    runtime: SSLDataRuntime,
    model: MaskedGraphSSLModel,
    objective: Phase8BObjectiveConfig,
    masking: ResolvedPhase8BMaskingConfig,
    execution_mode: str,
    comparison: ResolvedPhase8B2Schedule | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    from music_critic.ssl.engine import _device_evidence

    return {
        "training_report_version": PHASE8B_TRAINING_REPORT_VERSION,
        "engine_contract_version": PHASE8B_ENGINE_CONTRACT_VERSION,
        "evidence_kind": (
            "official_phase8b1_engine_mechanics"
            if comparison is None
            else "official_phase8b2_comparison_cell"
        ),
        "scientific_claim": "plumbing_and_optimization_mechanics_only",
        "compute_matched_comparison": bool(
            comparison is not None
            and comparison.comparison_mode == "encoder_forward_matched"
        ),
        "effectiveness_comparison": False,
        "science_owner": "Phase_8B_2",
        "execution_mode": execution_mode,
        "model_class": type(model).__name__,
        "active_objective_families": [
            family
            for family, weight in objective.family_weights
            if weight > 0.0
        ],
        "active_objective_weights": [
            [family, weight]
            for family, weight in objective.family_weights
            if weight > 0.0
        ],
        "resolved_mask_policies": (
            list(masking.scheduled_policies)
            if comparison is None
            else [policy for policy, _ in comparison.policy_views]
        ),
        "scheduled_policy_pass_count_per_cpu_batch": len(
            masking.scheduled_policies
            if comparison is None
            else comparison.policy_views
        ),
        "scheduled_view_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
        "amp_compute_contract": PHASE8B_AMP_COMPUTE_CONTRACT,
        "optimizer_evidence_contract_version": (
            PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION
        ),
        "grad_scaler_initial_scale": PHASE8B_GRAD_SCALER_INITIAL_SCALE,
        "optimizer_parameter_coverage": _optimizer_parameter_coverage(
            model, optimizer
        ),
        "cross_policy_manual_oracle": (
            phase8b_cross_policy_manual_oracle()
        ),
        "objective_registry_fingerprint": (
            PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
        ),
        "objective_config_fingerprint": objective.fingerprint,
        "masking_config_fingerprint": masking.fingerprint,
        "mask_policy_mixture_fingerprint": masking.policy_config.fingerprint,
        "validation_membership": asdict(runtime.validation_membership),
        "validation_seed_coordinate": 0,
        "validation_schedule_fixed": True,
        "fingerprints": runtime.fingerprints,
        "data_composition": runtime.mixture_statistics,
        "device": _device_evidence(device),
        "amp_enabled": bool(config["device"]["amp"]),
        "scaler_enabled": scaler.is_enabled(),
        "phase8_started": True,
        "phase8b2_started": comparison is not None,
        "phase8b2_schedule": (
            None if comparison is None else comparison.to_dict()
        ),
        "pdmx_added": False,
        "pll_implemented": False,
        "critic_or_quality_score_implemented": False,
    }


def _run_one_batch(config: dict[str, Any]) -> dict[str, object]:
    from music_critic.ssl.engine import (
        _checkpoint_config,
        _initialize_output,
        _write_json_atomic,
    )

    started = time.perf_counter()
    (
        output,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        objective,
        masking,
        execution_mode,
        resolved,
    ) = _prepare(config)
    comparison = ResolvedPhase8B2Schedule.from_config(
        OmegaConf.create(resolved["phase8b2_schedule"])
        if resolved.get("phase8b2_schedule") is not None
        else None,
        masking=masking,
        objective_mode=objective.mode,
    )
    del scheduler
    _initialize_output(
        output,
        resume=False,
        overwrite=bool(resolved["experiment"]["overwrite_output"]),
    )
    _write_initial_artifacts(output, resolved, runtime, model)
    batch = runtime.first_train_batch
    input_fixture_fingerprint = _fingerprint(
        {
            "data_fingerprints": runtime.fingerprints,
            "first_train_batch": _input_batch_fingerprint(batch),
        }
    )
    initial_model_state_fingerprint = _model_state_fingerprint(model)
    initial, _ = _stage(
        model,
        (batch,),
        objective=objective,
        masking=masking,
        execution_mode=execution_mode,
        config=resolved,
        device=device,
        epoch=0,
        stage="train",
        comparison=comparison,
    )
    total_accounting = _Accounting()
    total_accounting.add(_Accounting(**{
        key: value
        for key, value in initial["accounting"].items()
        if key in _Accounting.__dataclass_fields__
    }))
    gradient = None
    step_metrics = []
    for _step in range(int(resolved["experiment"]["steps"])):
        metric, current_gradient = _stage(
            model,
            (batch,),
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            config=resolved,
            device=device,
            epoch=0,
            stage="train",
            optimizer=optimizer,
            scaler=scaler,
            collect_gradient_evidence=gradient is None,
            comparison=comparison,
        )
        step_metrics.append(metric)
        if current_gradient is not None:
            gradient = current_gradient
        total_accounting.add(_Accounting(**{
            key: value
            for key, value in metric["accounting"].items()
            if key in _Accounting.__dataclass_fields__
        }))
    final, _ = _stage(
        model,
        (batch,),
        objective=objective,
        masking=masking,
        execution_mode=execution_mode,
        config=resolved,
        device=device,
        epoch=0,
        stage="train",
        comparison=comparison,
    )
    total_accounting.add(_Accounting(**{
        key: value
        for key, value in final["accounting"].items()
        if key in _Accounting.__dataclass_fields__
    }))
    checkpoint_path = output / "one_batch.pt"
    save_ssl_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        epoch_journal=(),
        resolved_config=_checkpoint_config(resolved),
        data_fingerprints=runtime.fingerprints,
    )
    clone = build_phase8b_model_from_config(
        OmegaConf.create(resolved["model"]),
        OmegaConf.create(resolved["ssl"]),
        OmegaConf.create(resolved["phase8b_objective"]),
    ).to(device)
    from music_critic.ssl.engine import _optimizer

    clone_optimizer = _optimizer(clone, resolved)
    clone_scaler = torch.amp.GradScaler(
        device.type,
        init_scale=PHASE8B_GRAD_SCALER_INITIAL_SCALE,
        enabled=bool(resolved["device"]["amp"]),
    )
    state = load_ssl_checkpoint(
        checkpoint_path,
        clone,
        clone_optimizer,
        scheduler=None,
        scaler=clone_scaler,
        maximum_next_epoch=0,
        resolved_config=_checkpoint_config(resolved),
        data_fingerprints=runtime.fingerprints,
    )
    reload_exact = all(
        torch.equal(left, clone.state_dict()[name])
        for name, left in model.state_dict().items()
    )
    initial_loss = initial["total_ssl_loss"]
    final_loss = final["total_ssl_loss"]
    final_model_state_fingerprint = _model_state_fingerprint(model)
    report = {
        **_report_common(
            config=resolved,
            runtime=runtime,
            model=model,
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            comparison=comparison,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        ),
        "run_scope": "one_batch_plumbing",
        "steps": int(resolved["experiment"]["steps"]),
        "input_fixture_fingerprint": input_fixture_fingerprint,
        "model_state_fingerprints": {
            "initial": initial_model_state_fingerprint,
            "final": final_model_state_fingerprint,
            "changed": (
                initial_model_state_fingerprint
                != final_model_state_fingerprint
            ),
        },
        "encoder_state_fingerprints": {
            "initial": _read_initial_encoder_fingerprint(output),
            "final": _encoder_state_fingerprint(model),
        },
        "initial": initial,
        "final": final,
        "loss_decreased": (
            initial_loss is not None
            and final_loss is not None
            and float(final_loss) < float(initial_loss)
        ),
        "gradient_coverage": gradient,
        "checkpoint_reload": {
            "next_epoch": state.next_epoch,
            "bit_exact": reload_exact,
        },
        "accounting": total_accounting.to_dict(),
        "optimization_stage_accounting": [
            row["accounting"] for row in step_metrics
        ],
        "optimization_step_evidence": [
            evidence
            for row in step_metrics
            for evidence in row["optimizer_step_evidence"]
        ],
        "cuda_peak_memory": _cuda_peak_memory(device),
        "duration_seconds": time.perf_counter() - started,
    }
    report["mechanics_acceptance"] = _one_batch_mechanics_acceptance(
        report
    )
    _write_json_atomic(output / "one_batch_report.json", report)
    if not report["mechanics_acceptance"]["passed"]:
        raise Phase8BEngineError(
            "phase8b.engine.one_batch_mechanics_acceptance_failed:"
            + ",".join(report["mechanics_acceptance"]["failures"])
        )
    return report


def _run_epochs(
    config: dict[str, Any], *, stop_after_epoch: int | None
) -> dict[str, object]:
    from music_critic.ssl.engine import (
        SSL_PERFORMANCE_ROW_VERSION,
        _checkpoint_config,
        _initialize_output,
        _read_json,
        _write_json_atomic,
        _write_jsonl_atomic,
    )

    started = time.perf_counter()
    (
        output,
        device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        objective,
        masking,
        execution_mode,
        resolved,
    ) = _prepare(config)
    comparison = ResolvedPhase8B2Schedule.from_config(
        OmegaConf.create(resolved["phase8b2_schedule"])
        if resolved.get("phase8b2_schedule") is not None
        else None,
        masking=masking,
        objective_mode=objective.mode,
    )
    resume_path = str(resolved["experiment"]["resume_from"])
    _initialize_output(
        output,
        resume=bool(resume_path),
        overwrite=bool(resolved["experiment"]["overwrite_output"]),
    )
    start_epoch = 0
    best: float | None = None
    journal: tuple[dict[str, object], ...] = ()
    if resume_path:
        _validate_resume_artifacts(output, resolved, runtime, model)
        initial_validation = _read_json(output / "initial_validation.json")
        if not isinstance(initial_validation, dict):
            raise Phase8BEngineError(
                "phase8b.engine.initial_validation_invalid"
            )
        state: SSLResumeState = load_ssl_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            maximum_next_epoch=int(resolved["experiment"]["epochs"]),
            resolved_config=_checkpoint_config(resolved),
            data_fingerprints=runtime.fingerprints,
        )
        start_epoch = state.next_epoch
        best = state.best_validation_loss
        journal = state.epoch_journal
        _write_jsonl_atomic(output / "metrics.jsonl", journal)
    else:
        initial_validation, _ = _stage(
            model,
            runtime.validation_loader(),
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            config=resolved,
            device=device,
            epoch=0,
            stage="validation",
            comparison=comparison,
        )
        initial_validation["optimizer_step_count_at_measurement"] = 0
        _write_initial_artifacts(
            output,
            resolved,
            runtime,
            model,
            initial_validation=initial_validation,
        )
        _write_jsonl_atomic(output / "metrics.jsonl", ())
        _write_jsonl_atomic(output / "epoch_performance.jsonl", ())
    starting_model_state_fingerprint = _model_state_fingerprint(model)
    starting_encoder_state_fingerprint = _encoder_state_fingerprint(model)
    completed = start_epoch
    epochs = int(resolved["experiment"]["epochs"])
    logical_update_budget = (
        comparison.logical_updates if comparison is not None else None
    )
    steps_per_epoch = int(
        resolved["experiment"].get("optimizer_steps_per_epoch", 0)
    ) or int(resolved["experiment"]["steps"])
    performance_rows = []
    performance_path = output / "epoch_performance.jsonl"
    if resume_path and performance_path.exists():
        performance_rows = [
            json.loads(line)
            for line in performance_path.read_text(encoding="utf-8").splitlines()
            if line and int(json.loads(line)["epoch"]) < start_epoch
        ]
    for epoch in range(start_epoch, epochs):
        attempted_before_epoch = sum(
            int(row["train"]["accounting"][
                "optimizer_step_attempt_count"
            ])
            for row in journal
        )
        remaining_updates = (
            None
            if logical_update_budget is None
            else logical_update_budget - attempted_before_epoch
        )
        if remaining_updates is not None and remaining_updates <= 0:
            break
        epoch_started = time.perf_counter()
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train, gradient = _stage(
            model,
            runtime.train_loader(epoch),
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            config=resolved,
            device=device,
            epoch=epoch,
            stage="train",
            optimizer=optimizer,
            scaler=scaler,
            collect_gradient_evidence=bool(
                resolved["experiment"]["collect_gradient_evidence"]
            ),
            comparison=comparison,
            maximum_batches=(
                None
                if remaining_updates is None
                else min(steps_per_epoch, remaining_updates)
            ),
        )
        if train["batch_count"] == 0:
            raise Phase8BEngineError("phase8b.engine.empty_train_epoch")
        if train["accounting"]["optimizer_step_applied_count"] <= 0:
            raise Phase8BEngineError(
                "phase8b.engine.epoch_all_optimizer_steps_skipped"
            )
        if bool(resolved["experiment"]["collect_gradient_evidence"]) and (
            gradient is None
            or not gradient["acceptance"]["passed"]
        ):
            raise Phase8BEngineError(
                "phase8b.engine.epoch_gradient_acceptance_failed"
            )
        if scheduler is not None:
            scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        validation = None
        if (
            (epoch + 1)
            % int(resolved["experiment"]["validation_interval"])
            == 0
            or (
                logical_update_budget is not None
                and attempted_before_epoch
                + int(train["accounting"]["optimizer_step_attempt_count"])
                == logical_update_budget
            )
            or (logical_update_budget is None and epoch + 1 == epochs)
        ):
            validation, _ = _stage(
                model,
                runtime.validation_loader(),
                objective=objective,
                masking=masking,
                execution_mode=execution_mode,
                config=resolved,
                device=device,
                epoch=0,
                stage="validation",
                comparison=comparison,
            )
            validation["optimizer_step_count_at_measurement"] = sum(
                int(row["train"]["accounting"]["optimizer_step_count"])
                for row in journal
            ) + int(train["accounting"]["optimizer_step_count"])
        row = {
            "metric_row_version": SSL_METRIC_ROW_VERSION,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "learning_rate_used": learning_rate_used,
            "next_learning_rate": next_learning_rate,
            "train": train,
            "validation": validation,
            "gradient_coverage": gradient,
        }
        validation_loss = (
            None if validation is None else validation["total_ssl_loss"]
        )
        improved = validation_loss is not None and (
            best is None or float(validation_loss) < best
        )
        if improved:
            best = float(validation_loss)
        journal = (*journal, row)
        save_ssl_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
            next_epoch=epoch + 1,
            best_validation_loss=best,
            epoch_journal=journal,
            resolved_config=_checkpoint_config(resolved),
            data_fingerprints=runtime.fingerprints,
        )
        if improved:
            save_ssl_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                scheduler=scheduler,
                scaler=scaler,
                next_epoch=epoch + 1,
                best_validation_loss=best,
                epoch_journal=journal,
                resolved_config=_checkpoint_config(resolved),
                data_fingerprints=runtime.fingerprints,
            )
        _write_jsonl_atomic(output / "metrics.jsonl", journal)
        performance_rows.append(
            {
                "performance_row_version": SSL_PERFORMANCE_ROW_VERSION,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "stage_timing": {
                    "epoch_total_seconds": time.perf_counter() - epoch_started
                },
                "unavailable_reason": None,
                "checkpoint_binding_participation": False,
            }
        )
        _write_jsonl_atomic(
            output / "epoch_performance.jsonl", performance_rows
        )
        completed = epoch + 1
        if stop_after_epoch is not None and completed >= stop_after_epoch:
            break
    total_accounting = _Accounting()
    for row in journal:
        total_accounting.add(_Accounting(**{
            key: value
            for key, value in row["train"]["accounting"].items()
            if key in _Accounting.__dataclass_fields__
        }))
    observed_identities = [
        identity
        for row in journal
        for identity in row["train"].get("input_sample_identities", [])
    ]
    from music_critic.experiments.phase8b2.contracts import (
        fingerprint as phase8b2_fingerprint,
    )
    from music_critic.experiments.phase8b2.schedule import (
        SCHEDULE_CONTRACT_VERSION,
    )

    observed_identity_schedule_fingerprint = phase8b2_fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_ssl_sample_schedule",
            "identities": observed_identities,
        }
    )
    expected_schedule_fingerprint = (
        None if comparison is None else comparison.sample_schedule_fingerprint
    )
    schedule_verified = (
        comparison is None
        or observed_identity_schedule_fingerprint
        == expected_schedule_fingerprint
    )
    complete_budget = (
        True
        if logical_update_budget is None
        else total_accounting.optimizer_step_attempt_count
        == logical_update_budget
    )
    if comparison is not None and complete_budget and not schedule_verified:
        raise Phase8BEngineError(
            "phase8b2.engine.actual_sample_schedule_mismatch"
        )
    if comparison is not None and complete_budget and (
        total_accounting.optimizer_step_skipped_count != 0
        or total_accounting.optimizer_step_applied_count
        != logical_update_budget
    ):
        raise Phase8BEngineError(
            "phase8b2.engine.scientific_cell_optimizer_step_invalid"
        )
    report = {
        **_report_common(
            config=resolved,
            runtime=runtime,
            model=model,
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            comparison=comparison,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        ),
        "run_scope": "epoch_pretraining",
        "start_epoch": start_epoch,
        "completed_epochs": completed,
        "configured_epochs": epochs,
        "configured_logical_updates": logical_update_budget,
        "logical_update_budget_complete": complete_budget,
        "model_state_fingerprints": {
            "start_or_resume": starting_model_state_fingerprint,
            "final": _model_state_fingerprint(model),
        },
        "encoder_state_fingerprints": {
            "initial": _read_initial_encoder_fingerprint(output),
            "start_or_resume": starting_encoder_state_fingerprint,
            "final": _encoder_state_fingerprint(model),
        },
        "initial_validation": initial_validation,
        "best_validation_loss": best,
        "best_checkpoint_selection": (
            "minimum_family_global_validation_total_ssl_loss"
        ),
        "best_checkpoint": None if best is None else str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "metrics": str(output / "metrics.jsonl"),
        "observed_ssl_sample_schedule_fingerprint": (
            observed_identity_schedule_fingerprint
            if comparison is not None
            else _journal_train_evidence_fingerprint(
                journal, "input_batch_fingerprints"
            )
        ),
        "expected_ssl_sample_schedule_fingerprint": (
            expected_schedule_fingerprint
        ),
        "actual_sample_schedule_verified": schedule_verified,
        "observed_sample_identities": observed_identities,
        "observed_prediction_fingerprint": (
            _journal_train_evidence_fingerprint(
                journal, "prediction_fingerprints"
            )
        ),
        "observed_gradient_fingerprint": (
            _journal_train_evidence_fingerprint(
                journal, "gradient_fingerprints"
            )
        ),
        "cuda_peak_memory": _cuda_peak_memory(device),
        "epoch_performance": str(output / "epoch_performance.jsonl"),
        "resume_boundary": "epoch_only",
        "mid_epoch_resume_supported": False,
        "accounting": total_accounting.to_dict(),
        "duration_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(output / "training_report.json", report)
    return report


def run_phase8b_training(
    config: dict[str, Any], *, stop_after_epoch: int | None = None
) -> dict[str, object]:
    """Run one official explicit Phase 8B.1 job after outer RNG guarding."""

    if config["experiment"]["name"] == "one_batch":
        if stop_after_epoch is not None:
            raise Phase8BEngineError(
                "phase8b.engine.stop_after_epoch_pretrain_only"
            )
        return _run_one_batch(config)
    return _run_epochs(config, stop_after_epoch=stop_after_epoch)


__all__ = [
    "PHASE8B_ENGINE_CONTRACT_VERSION",
    "PHASE8B_GRAD_SCALER_INITIAL_SCALE",
    "PHASE8B_MASKING_CONFIG_CONTRACT_VERSION",
    "PHASE8B_OPTIMIZER_EVIDENCE_CONTRACT_VERSION",
    "PHASE8B2_SCHEDULE_BINDING_CONTRACT_VERSION",
    "PHASE8B_RUN_MANIFEST_VERSION",
    "PHASE8B_TRAINING_REPORT_VERSION",
    "Phase8BEngineError",
    "ResolvedPhase8BMaskingConfig",
    "ResolvedPhase8B2Schedule",
    "run_phase8b_training",
]
