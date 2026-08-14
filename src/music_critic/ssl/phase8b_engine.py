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
    PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
    PHASE8B_SCHEDULED_VIEW_AGGREGATION,
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


PHASE8B_ENGINE_CONTRACT_VERSION = "1.1.0"
PHASE8B_MASKING_CONFIG_CONTRACT_VERSION = "1.1.0"
PHASE8B_RUN_MANIFEST_VERSION = "1.1.0"
PHASE8B_TRAINING_REPORT_VERSION = "1.1.0"

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


@dataclass(slots=True)
class _Accounting:
    cpu_batch_count: int = 0
    optimizer_step_count: int = 0
    forward_pass_count: int = 0
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
        result = asdict(self)
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
    materialized = copy.deepcopy(config)
    materialized["phase8b_runtime"] = {
        "engine_contract_version": PHASE8B_ENGINE_CONTRACT_VERSION,
        "checkpoint_binding_contract_version": (
            PHASE8B_CHECKPOINT_BINDING_CONTRACT_VERSION
        ),
        "scheduled_view_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
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
    }
    return materialized, objective, masking, execution_mode


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

    resolved, objective, masking, execution_mode = _materialize(config)
    _set_determinism(int(resolved["seed"]))
    device = _resolve_device(resolved)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runtime = build_ssl_data_runtime(
        OmegaConf.create(resolved["data"]), seed=int(resolved["seed"])
    )
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
        device.type, enabled=bool(resolved["device"]["amp"])
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
) -> _PreparedPass:
    binding = prepare_hierarchy_mask_binding(
        cpu_batch,
        policy_config=masking.pass_config(policy),
        global_seed=int(config["seed"]),
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
) -> tuple[dict[str, object], dict[str, object] | None]:
    from music_critic.ssl.engine import _gradient_evidence

    training = optimizer is not None
    if training != (scaler is not None) or stage not in {"train", "validation"}:
        raise Phase8BEngineError("phase8b.engine.stage_arguments_invalid")
    model.train(training)
    accumulator = Phase8BObjectiveAccumulator(objective)
    accounting = _Accounting()
    available_batch_count = 0
    binding_fingerprints: list[str] = []
    plan_fingerprints: list[str] = []
    policy_pass_counts = {
        policy: 0 for policy in masking.scheduled_policies
    }
    gradient_evidence = None
    for cpu_batch in loader:
        accounting.cpu_batch_count += 1
        accounting.sample_count += cpu_batch.sample_count
        accounting.node_count += cpu_batch.node_count
        accounting.edge_count += cpu_batch.edge_count
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
        for policy in masking.scheduled_policies:
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
            )
            primary, collateral_note, collateral_track = _binding_summary(
                prepared.binding
            )
            accounting.primary_masked_entity_count += primary
            accounting.collateral_note_entity_count += collateral_note
            accounting.collateral_track_entity_count += collateral_track
            binding_fingerprints.append(prepared.binding.fingerprint)
            plan_fingerprints.extend(
                plan.fingerprint for plan in prepared.binding.mask_plans
            )
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(config["device"]["amp"]),
                ):
                    output = _forward(
                        model, prepared, execution_mode=execution_mode
                    )
            accounting.forward_pass_count += 1
            accounting.objective_evaluation_count += 1
            policy_outputs.append((policy, output))
        batch_objective = aggregate_phase8b_policy_pass_losses(
            tuple(policy_outputs), objective_config=objective
        )
        try:
            accumulator.update_batch(batch_objective)
        except ValueError as exc:
            if "non-finite" in str(exc):
                raise Phase8BEngineError(
                    "phase8b.engine.nonfinite_total_loss"
                ) from exc
            raise
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
            scaler.scale(current_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["optimizer"]["gradient_clip_norm"]),
            )
            if collect_gradient_evidence and gradient_evidence is None:
                gradient_evidence = _gradient_evidence(model)
            scaler.step(optimizer)
            scaler.update()
            accounting.optimizer_step_count += 1
        del policy_outputs, batch_objective, current_loss
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
        "resolved_policies": list(masking.scheduled_policies),
        "policy_pass_counts": policy_pass_counts,
        "prepared_binding_fingerprints": sorted(binding_fingerprints),
        "mask_plan_fingerprints": sorted(plan_fingerprints),
        "validation_schedule_batch_partition_independent_coordinates": (
            stage == "validation"
        ),
        "objective_config_fingerprint": objective.fingerprint,
        "masking_config_fingerprint": masking.fingerprint,
        "mask_policy_mixture_fingerprint": masking.policy_config.fingerprint,
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
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict[str, object]:
    from music_critic.ssl.engine import _device_evidence

    return {
        "training_report_version": PHASE8B_TRAINING_REPORT_VERSION,
        "engine_contract_version": PHASE8B_ENGINE_CONTRACT_VERSION,
        "evidence_kind": "official_phase8b1_engine_mechanics",
        "scientific_claim": "plumbing_and_optimization_mechanics_only",
        "compute_matched_comparison": False,
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
        "resolved_mask_policies": list(masking.scheduled_policies),
        "scheduled_policy_pass_count_per_cpu_batch": len(
            masking.scheduled_policies
        ),
        "scheduled_view_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
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
        "phase8b2_started": False,
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
    del scheduler
    _initialize_output(
        output,
        resume=False,
        overwrite=bool(resolved["experiment"]["overwrite_output"]),
    )
    _write_initial_artifacts(output, resolved, runtime, model)
    batch = runtime.first_train_batch
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
        device.type, enabled=bool(resolved["device"]["amp"])
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
    report = {
        **_report_common(
            config=resolved,
            runtime=runtime,
            model=model,
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            device=device,
            scaler=scaler,
        ),
        "run_scope": "one_batch_plumbing",
        "steps": int(resolved["experiment"]["steps"]),
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
        "duration_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(output / "one_batch_report.json", report)
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
    completed = start_epoch
    epochs = int(resolved["experiment"]["epochs"])
    performance_rows = []
    performance_path = output / "epoch_performance.jsonl"
    if resume_path and performance_path.exists():
        performance_rows = [
            json.loads(line)
            for line in performance_path.read_text(encoding="utf-8").splitlines()
            if line and int(json.loads(line)["epoch"]) < start_epoch
        ]
    for epoch in range(start_epoch, epochs):
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
        )
        if train["batch_count"] == 0:
            raise Phase8BEngineError("phase8b.engine.empty_train_epoch")
        if scheduler is not None:
            scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        validation = None
        if (
            (epoch + 1)
            % int(resolved["experiment"]["validation_interval"])
            == 0
            or epoch + 1 == epochs
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
    report = {
        **_report_common(
            config=resolved,
            runtime=runtime,
            model=model,
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            device=device,
            scaler=scaler,
        ),
        "run_scope": "epoch_pretraining",
        "start_epoch": start_epoch,
        "completed_epochs": completed,
        "configured_epochs": epochs,
        "initial_validation": initial_validation,
        "best_validation_loss": best,
        "best_checkpoint_selection": (
            "minimum_family_global_validation_total_ssl_loss"
        ),
        "best_checkpoint": None if best is None else str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "metrics": str(output / "metrics.jsonl"),
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
    "PHASE8B_MASKING_CONFIG_CONTRACT_VERSION",
    "PHASE8B_RUN_MANIFEST_VERSION",
    "PHASE8B_TRAINING_REPORT_VERSION",
    "Phase8BEngineError",
    "ResolvedPhase8BMaskingConfig",
    "run_phase8b_training",
]
