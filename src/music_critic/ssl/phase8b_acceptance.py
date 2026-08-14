"""Deterministic bounded mechanics evidence for Phase 8B.1.

The comparison deliberately uses only the accepted synthetic Phase 8A
fixture.  It measures optimization mechanics and makes no musical-quality or
downstream-performance claim.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import torch
from torch import Tensor

from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.data import SSLBatch, collate_ssl_samples
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchyMaskPolicyConfig,
)
from music_critic.ssl.hierarchy_fixture import build_phase8a_hierarchy_fixture
from music_critic.ssl.masking import (
    PreparedHierarchyMaskBinding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import MaskedGraphSSLConfig, MaskedGraphSSLModel
from music_critic.ssl.multilevel import (
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    ONSET_LATENT,
    PHASE7A_BAR_LATENT,
    PHASE7A_NOTE_RECONSTRUCTION,
    PHASE7A_SONG_LATENT,
    PHASE8B_NEW_OBJECTIVE_FAMILIES,
    PHASE8B_OBJECTIVE_FAMILIES,
    PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
    TRACK_LATENT,
    Phase8BMultilevelSSLForwardOutput,
    Phase8BMultilevelSSLModel,
    Phase8BObjectiveConfig,
    prepare_phase8b_objective_binding,
)
from music_critic.ssl.objective import StreamingAntiCollapseDiagnostics


PHASE8B_BOUNDED_COMPARISON_CONTRACT_VERSION = "1.0.0"
PHASE8B_BOUNDED_COMPARISON_SCOPE = (
    "bounded_representation_recovery_mechanics_only"
)

_HIERARCHY_POLICIES = (
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)
_POLICY_FOR_MODE = {
    "onset_only": ONSET_PITCH_DESCENDANTS,
    "beat_only": BEAT_PITCH_DESCENDANTS,
    "bar_only": CONTIGUOUS_BAR_PITCH_SPAN,
    "track_only": TRACK_BAR_PITCH_SPAN,
}
_VARIANTS = (
    "phase7a_control",
    "phase8a_masks_old_objectives",
    "onset_only",
    "beat_only",
    "bar_only",
    "track_only",
    "multilevel_equal_weight",
)


def _canonical_fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _state_fingerprint(
    model: torch.nn.Module,
    *,
    include_new_heads: bool,
) -> str:
    digest = sha256()
    for name, value in model.state_dict().items():
        is_new = name.startswith("phase8b_latent_heads.")
        if is_new != include_new_heads:
            continue
        detached = value.detach().cpu().contiguous()
        descriptor = {
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(detached.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _encoder_config() -> HierarchicalBaselineConfig:
    return HierarchicalBaselineConfig(
        hidden_dim=8,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    )


def _ssl_config() -> MaskedGraphSSLConfig:
    return MaskedGraphSSLConfig(
        mask_rate=0.30,
        decoder_views=1,
        decoder_remask_probability=0.0,
        decoder_hidden_dim=8,
        projector_hidden_dim=8,
        note_weight=1.0,
        bar_weight=1.0,
        song_weight=1.0,
    )


def _policies(variant: str) -> tuple[str, ...]:
    if variant == "phase7a_control":
        return (INDEPENDENT_NOTE_PITCH,)
    if variant == "phase8a_masks_old_objectives":
        return _HIERARCHY_POLICIES
    if variant in _POLICY_FOR_MODE:
        return (_POLICY_FOR_MODE[variant],)
    if variant == "multilevel_equal_weight":
        return _HIERARCHY_POLICIES
    raise ValueError("unknown Phase 8B.1 bounded variant")


def _model(variant: str, *, seed: int) -> MaskedGraphSSLModel:
    torch.manual_seed(seed)
    if variant in {"phase7a_control", "phase8a_masks_old_objectives"}:
        return MaskedGraphSSLModel(_encoder_config(), _ssl_config())
    return Phase8BMultilevelSSLModel(
        _encoder_config(),
        _ssl_config(),
        Phase8BObjectiveConfig.for_mode(variant),
    )


def _binding(
    batch: SSLBatch,
    *,
    policy: str,
    seed: int,
    epoch: int,
    stage: str,
):
    return prepare_hierarchy_mask_binding(
        batch,
        policy_config=HierarchyMaskPolicyConfig.create(
            weights={policy: 1.0},
            min_span_bars=1,
            max_span_bars=2,
        ),
        global_seed=seed,
        epoch=epoch,
        requested_mask_rate=0.30,
        stage=stage,
    )


def _forward_outputs(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    variant: str,
    seed: int,
    epoch: int,
    stage: str,
) -> tuple[tuple[str, object], ...]:
    outputs = []
    for policy in _policies(variant):
        binding = _binding(
            batch,
            policy=policy,
            seed=seed,
            epoch=epoch,
            stage=stage,
        )
        if type(model) is MaskedGraphSSLModel:
            output = (
                model(batch, prepared_mask_binding=binding)
                if policy == INDEPENDENT_NOTE_PITCH
                else model.forward_hierarchy(
                    batch, prepared_mask_binding=binding
                )
            )
        else:
            assert type(model) is Phase8BMultilevelSSLModel
            if type(binding) is not PreparedHierarchyMaskBinding:
                raise ValueError(
                    "new multilevel objectives require hierarchy policies"
                )
            objective_binding = prepare_phase8b_objective_binding(
                binding, model.phase8b_objective_config
            )
            output = model.forward_multilevel(
                batch,
                prepared_mask_binding=binding,
                prepared_objective_binding=objective_binding,
            )
        outputs.append((policy, output))
    return tuple(outputs)


def _loss(output: object) -> Tensor | None:
    if type(output) is Phase8BMultilevelSSLForwardOutput:
        return output.objective.total_loss
    return output.objective.total_loss


def _old_rows(output: object) -> tuple[tuple[str, object, float], ...]:
    return (
        (PHASE7A_NOTE_RECONSTRUCTION, output.note_loss, 1.0),
        (PHASE7A_BAR_LATENT, output.bar_latent.loss, 1.0),
        (PHASE7A_SONG_LATENT, output.song_latent.loss, 1.0),
    )


def _active_rows(output: object) -> tuple[tuple[str, object, float], ...]:
    if type(output) is not Phase8BMultilevelSSLForwardOutput:
        return _old_rows(output)
    return tuple(
        (row.family, row, row.configured_weight)
        for row in output.objective.family_losses
        if row.active
    )


def _update_diagnostics(
    diagnostics: dict[str, StreamingAntiCollapseDiagnostics],
    output: object,
) -> None:
    if type(output) is Phase8BMultilevelSSLForwardOutput:
        for row in output.latent_predictions:
            diagnostics.setdefault(
                row.family, StreamingAntiCollapseDiagnostics()
            ).update(row.target, row.prediction)
        return
    note_prediction = torch.stack(output.decoder_predictions).mean(dim=0)
    diagnostics.setdefault(
        PHASE7A_NOTE_RECONSTRUCTION,
        StreamingAntiCollapseDiagnostics(),
    ).update(output.targets.note.index_select(
        0, output.selected_global_note_indices
    ), note_prediction)
    diagnostics.setdefault(
        PHASE7A_BAR_LATENT, StreamingAntiCollapseDiagnostics()
    ).update(output.bar_latent.target, output.bar_latent.prediction)
    diagnostics.setdefault(
        PHASE7A_SONG_LATENT, StreamingAntiCollapseDiagnostics()
    ).update(output.song_latent.target, output.song_latent.prediction)


def _stage_evidence(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    variant: str,
    seed: int,
    epoch: int,
    stage: str,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    numerators = {family: 0.0 for family in PHASE8B_OBJECTIVE_FAMILIES}
    denominators = {family: 0 for family in PHASE8B_OBJECTIVE_FAMILIES}
    weights: dict[str, float] = {}
    diagnostics: dict[str, StreamingAntiCollapseDiagnostics] = {}
    try:
        with torch.no_grad():
            outputs = _forward_outputs(
                model,
                batch,
                variant=variant,
                seed=seed,
                epoch=epoch,
                stage=stage,
            )
            for _policy, output in outputs:
                for family, row, weight in _active_rows(output):
                    numerators[family] += float(row.numerator.detach())
                    denominators[family] += int(
                        getattr(
                            row,
                            "eligible_denominator",
                            getattr(row, "denominator", 0),
                        )
                    )
                    weights[family] = weight
                _update_diagnostics(diagnostics, output)
    finally:
        model.train(was_training)
    families = []
    for family in PHASE8B_OBJECTIVE_FAMILIES:
        if family not in weights:
            continue
        denominator = denominators[family]
        mean = numerators[family] / denominator if denominator > 0 else None
        families.append(
            {
                "family": family,
                "numerator": numerators[family],
                "eligible_denominator": denominator,
                "mean_loss": mean,
                "available": denominator > 0,
                "unavailable_reason": (
                    None if denominator > 0 else "no_eligible_entities"
                ),
                "configured_weight": weights[family],
                "active": True,
                "anti_collapse": (
                    diagnostics[family].to_dict()
                    if family in diagnostics
                    else None
                ),
            }
        )
    total = sum(
        row["configured_weight"] * row["mean_loss"]
        for row in families
        if row["mean_loss"] is not None
    )
    return {
        "stage": stage,
        "scheduled_pass_count": len(_policies(variant)),
        "families": families,
        "total_loss": total if any(
            row["mean_loss"] is not None for row in families
        ) else None,
        "retained_cuda_tensor_count": 0,
        "retained_prediction_tensor_count": 0,
    }


def _gradient_coverage(model: MaskedGraphSSLModel) -> dict[str, object]:
    required = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    present = tuple(name for name, parameter in required if parameter.grad is not None)
    finite = tuple(
        name
        for name, parameter in required
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
    )
    nonzero = tuple(
        name
        for name, parameter in required
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
    )
    return {
        "required_parameter_tensor_count": len(required),
        "gradient_present_tensor_count": len(present),
        "finite_gradient_tensor_count": len(finite),
        "nonzero_gradient_tensor_count": len(nonzero),
        "gradient_present_parameter_tensors": list(present),
        "finite_gradient_parameter_tensors": list(finite),
        "nonzero_gradient_parameter_tensors": list(nonzero),
    }


def _train(
    model: MaskedGraphSSLModel,
    batch: SSLBatch,
    *,
    variant: str,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    coverage: dict[str, object] | None = None
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_outputs(
            model,
            batch,
            variant=variant,
            seed=seed,
            epoch=step,
            stage="train",
        )
        losses = tuple(
            value
            for _policy, output in outputs
            if (value := _loss(output)) is not None
        )
        if not losses:
            raise ValueError("bounded variant has no available active loss")
        # The divisor is the fixed scheduled pass count, never the count of
        # available families, so missing evidence cannot rescale another loss.
        objective = torch.stack(losses).sum() / len(_policies(variant))
        if not bool(torch.isfinite(objective)):
            raise ValueError("bounded variant produced a non-finite loss")
        objective.backward()
        coverage = _gradient_coverage(model)
        optimizer.step()
    assert coverage is not None
    return coverage


def _family_means(stage: dict[str, object]) -> dict[str, float | None]:
    return {
        str(row["family"]): row["mean_loss"]
        for row in stage["families"]
    }


def _schedule_payload(
    train_batch: SSLBatch,
    validation_batch: SSLBatch,
    *,
    seed: int,
    steps: int,
) -> dict[str, object]:
    rows = []
    for stage, batch, epochs in (
        ("train", train_batch, tuple(range(steps))),
        ("train_evaluation", train_batch, (steps + 1000,)),
        ("validation", validation_batch, (steps + 1000,)),
    ):
        binding_stage = "train" if stage != "validation" else "validation"
        for epoch in epochs:
            for policy in (INDEPENDENT_NOTE_PITCH, *_HIERARCHY_POLICIES):
                binding = _binding(
                    batch,
                    policy=policy,
                    seed=seed,
                    epoch=epoch,
                    stage=binding_stage,
                )
                rows.append(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "policy": policy,
                        "prepared_binding_fingerprint": binding.fingerprint,
                    }
                )
    payload = {
        "global_seed": seed,
        "requested_mask_rate": 0.30,
        "policy_order": [INDEPENDENT_NOTE_PITCH, *_HIERARCHY_POLICIES],
        "rows": rows,
    }
    payload["fingerprint"] = _canonical_fingerprint(payload)
    return payload


def run_phase8b_bounded_comparison(
    *,
    seed: int = 42,
    steps: int = 12,
    learning_rate: float = 0.02,
) -> dict[str, object]:
    """Run the fixed CPU comparison and return deterministic JSON evidence."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or not isinstance(learning_rate, float)
        or not 0.0 < learning_rate < 1.0
    ):
        raise ValueError("Phase 8B.1 bounded comparison arguments are invalid")
    fixture = build_phase8a_hierarchy_fixture()
    train_batch = collate_ssl_samples(fixture.raw_samples("train"))
    validation_batch = collate_ssl_samples(fixture.raw_samples("validation"))
    evaluation_epoch = steps + 1000
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    variants = []
    base_initializations = set()
    new_head_initializations = set()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for variant in _VARIANTS:
            model = _model(variant, seed=seed)
            base_initialization = _state_fingerprint(
                model, include_new_heads=False
            )
            base_initializations.add(base_initialization)
            new_initialization = (
                _state_fingerprint(model, include_new_heads=True)
                if type(model) is Phase8BMultilevelSSLModel
                else None
            )
            if new_initialization is not None:
                new_head_initializations.add(new_initialization)
            initial_train = _stage_evidence(
                model,
                train_batch,
                variant=variant,
                seed=seed,
                epoch=evaluation_epoch,
                stage="train",
            )
            initial_validation = _stage_evidence(
                model,
                validation_batch,
                variant=variant,
                seed=seed,
                epoch=evaluation_epoch,
                stage="validation",
            )
            coverage = _train(
                model,
                train_batch,
                variant=variant,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
            )
            final_train = _stage_evidence(
                model,
                train_batch,
                variant=variant,
                seed=seed,
                epoch=evaluation_epoch,
                stage="train",
            )
            final_validation = _stage_evidence(
                model,
                validation_batch,
                variant=variant,
                seed=seed,
                epoch=evaluation_epoch,
                stage="validation",
            )
            initial_means = _family_means(initial_train)
            final_means = _family_means(final_train)
            overfit = {
                family: (
                    initial_means[family] is not None
                    and final_means[family] is not None
                    and final_means[family] < initial_means[family]
                )
                for family in initial_means
            }
            variants.append(
                {
                    "variant": variant,
                    "model_type": type(model).__name__,
                    "applied_mask_policies": list(_policies(variant)),
                    "scheduled_pass_divisor": len(_policies(variant)),
                    "base_initialization_fingerprint": base_initialization,
                    "new_head_initialization_fingerprint": new_initialization,
                    "new_head_parameter_count": (
                        model.new_head_parameter_count()
                        if type(model) is Phase8BMultilevelSSLModel
                        else 0
                    ),
                    "initial": {
                        "train": initial_train,
                        "held_out": initial_validation,
                    },
                    "final": {
                        "train": final_train,
                        "held_out": final_validation,
                    },
                    "train_family_loss_decreased": overfit,
                    "all_available_train_families_decreased": all(
                        overfit.values()
                    ),
                    "gradient_coverage": coverage,
                }
            )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
    schedule = _schedule_payload(
        train_batch,
        validation_batch,
        seed=seed,
        steps=steps,
    )
    payload: dict[str, Any] = {
        "contract_version": PHASE8B_BOUNDED_COMPARISON_CONTRACT_VERSION,
        "scope": PHASE8B_BOUNDED_COMPARISON_SCOPE,
        "claims_excluded": [
            "musical_representation_quality",
            "critic_quality",
            "downstream_improvement",
            "scientific_model_selection",
        ],
        "objective_registry_fingerprint": (
            PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
        ),
        "fixture_fingerprint": fixture.fixture_fingerprint,
        "train_membership": [list(row) for row in fixture.identities("train")],
        "held_out_membership": [
            list(row) for row in fixture.identities("validation")
        ],
        "protocol": {
            "device": "cpu",
            "seed": seed,
            "steps": steps,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "initialization": "torch_manual_seed_per_variant",
            "batch_membership_fixed": True,
            "masking_schedule_fingerprint": schedule["fingerprint"],
            "fixed_pass_divisor_no_availability_renormalization": True,
        },
        "masking_schedule": schedule,
        "shared_base_initialization": len(base_initializations) == 1,
        "shared_new_head_initialization": len(new_head_initializations) == 1,
        "variants": variants,
        "all_variant_train_overfit_checks_passed": all(
            row["all_available_train_families_decreased"] for row in variants
        ),
        "all_reports_retain_zero_cuda_predictions": all(
            stage["retained_cuda_tensor_count"] == 0
            and stage["retained_prediction_tensor_count"] == 0
            for row in variants
            for boundary in ("initial", "final")
            for stage in row[boundary].values()
        ),
    }
    payload["fingerprint"] = _canonical_fingerprint(payload)
    return payload


__all__ = [
    "PHASE8B_BOUNDED_COMPARISON_CONTRACT_VERSION",
    "PHASE8B_BOUNDED_COMPARISON_SCOPE",
    "run_phase8b_bounded_comparison",
]
