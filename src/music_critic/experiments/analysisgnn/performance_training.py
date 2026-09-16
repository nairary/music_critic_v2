"""Opt-in synchronization reductions for the frozen corrected update.

This module does not select a training profile or replace the reference path.
The raw-only model, supervised loss, clipping and optimizer remain unchanged.
Only metadata placement and transfers of diagnostic scalars differ. Callers
advance their sampler exactly when the returned update was applied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn

from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    CorrectedModelOutput,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    AppliedUpdateResult,
    CorrectedTargetAlignment,
    CorrectedTrainingError,
    FrozenClassWeights,
    align_target_sidecars_after_prediction,
    corrected_supervised_loss,
)


Instrumentation = Callable[[str], AbstractContextManager[object]]


def prepare_performance_class_weights(
    class_weights: FrozenClassWeights, device: torch.device | str
) -> FrozenClassWeights:
    """Create an owned, device-resident constant snapshot before timed updates.

    Tensor values, dtypes, support and provenance are preserved. As with the
    frozen weights, callers must treat the resulting tensors as constants.
    No process-global cache or mutation of the source weights is involved.
    """

    return FrozenClassWeights(
        values={key: value.to(device).clone()
                for key, value in class_weights.values.items()},
        supported={key: value.to(device).clone()
                   for key, value in class_weights.supported.items()},
        payload_fingerprint=class_weights.payload_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _CandidateStore:
    ptr: Tensor
    entity_id: object


@dataclass(frozen=True, slots=True)
class _AlignmentMetadataView:
    """Private join-only projection, never a validated model output or input."""
    predictions: tuple


def align_performance_target_sidecars(
    output: CorrectedModelOutput,
    raw_graph_batch: object,
    sidecars: Sequence[Mapping[str, object]],
    *,
    shifts: Sequence[int] | None = None,
) -> CorrectedTargetAlignment:
    """Reuse exact frozen alignment with CPU-only candidate metadata views.

    Called after prediction, this cannot supply targets to the encoder. Logits
    and encoder tensors retain their identities and autograd graphs. The view
    copies only metadata read by the frozen join; it is never a model input.
    """

    # Preserve the reference's early argument failures before reading stores.
    if not sidecars or (shifts is not None and len(shifts) != len(sidecars)):
        return align_target_sidecars_after_prediction(
            output, raw_graph_batch, sidecars, shifts=shifts
        )
    stores: dict[str, _CandidateStore] = {}
    predictions = []
    for prediction in output.predictions:
        node_type = prediction.allowed_node_types[0]
        if node_type not in stores:
            store = raw_graph_batch[node_type]  # type: ignore[index]
            stores[node_type] = _CandidateStore(
                ptr=store.ptr.detach().cpu(), entity_id=store.entity_id
            )
        predictions.append(replace(
            prediction,
            sample_indices=prediction.sample_indices.detach().cpu(),
            global_entity_indices=prediction.global_entity_indices.detach().cpu(),
        ))
    # Fusion outputs require all candidate metadata on the logits device.
    # A CPU join projection is deliberately not reconstructed as such an output.
    metadata_output = _AlignmentMetadataView(tuple(predictions))
    return align_target_sidecars_after_prediction(
        metadata_output, stores, sidecars, shifts=shifts
    )


def _coalesced_host_scalars(values: Sequence[Tensor]) -> list[bool | float]:
    """Transfer scalars together per device, preserving their original order."""

    groups: dict[torch.device, list[tuple[int, Tensor]]] = {}
    for index, value in enumerate(values):
        groups.setdefault(value.device, []).append((index, value))
    result: list[bool | float] = [False] * len(values)
    for entries in groups.values():
        host_values = torch.stack([value for _, value in entries]).detach().cpu().tolist()
        for (index, _), value in zip(entries, host_values, strict=True):
            result[index] = value
    return result


def performance_gradient_diagnostics(model: nn.Module) -> tuple[bool, float]:
    """Keep each FP32 parameter reduction and the reference Python sum order.

    A single global GPU reduction would change rounding. Stacking already
    reduced scalars only changes transfer granularity, not their arithmetic.
    All gradients are checked on every attempted update, before clipping/step.
    """

    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    finite = all(_coalesced_host_scalars([
        torch.isfinite(gradient).all() for gradient in gradients
    ]))
    squared = sum(_coalesced_host_scalars([
        gradient.detach().float().pow(2).sum() for gradient in gradients
    ]))
    return finite, math.sqrt(squared)


def attempt_performance_update(
    *,
    model: CorrectedAnalysisGNNModel,
    raw_graph_batch: object,
    sidecars: Sequence[Mapping[str, object]],
    shifts: Sequence[int],
    class_weights: FrozenClassWeights,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float = 1.0,
    instrumentation: Instrumentation | None = None,
) -> AppliedUpdateResult:
    """Attempt the frozen update with optional named timing contexts.

    Prepare class weights once with ``prepare_performance_class_weights`` for
    residency; ordinary FrozenClassWeights are also accepted. ``diagnostics``
    occurs before and after backward; instrumentation should accumulate both
    occurrences. The callback must not change tensors or RNG state.
    """

    stage = instrumentation if instrumentation is not None else lambda _: nullcontext()
    optimizer.zero_grad(set_to_none=True)
    with stage("forward"):
        output = model(raw_graph_batch)
    with stage("alignment"):
        alignment = align_performance_target_sidecars(
            output, raw_graph_batch, sidecars, shifts=shifts
        )
    with stage("loss"):
        loss = corrected_supervised_loss(output, alignment, class_weights)
    with stage("diagnostics"):
        if loss.total is None or not loss.fp32_boundary:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.no_finite_supervision", repr(loss.zero_valid_heads)
            )
        finite_flags = _coalesced_host_scalars([
            *(torch.isfinite(value).all() for value in output.logits.values()),
            torch.isfinite(loss.total),
        ])
        finite_logits = all(finite_flags[:-1])
        if not finite_logits or not finite_flags[-1]:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.non_finite_forward", "logits or loss are non-finite"
            )
    scale_before = float(scaler.get_scale())
    with stage("backward"):
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
    with stage("diagnostics"):
        finite_gradients, norm = performance_gradient_diagnostics(model)
        if not finite_gradients or not math.isfinite(norm):
            raise CorrectedTrainingError(
                "analysisgnn.corrected.non_finite_gradient", str(norm)
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    with stage("optimizer"):
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        applied = scale_after >= scale_before
        if applied:
            scheduler.step()
        else:
            optimizer.zero_grad(set_to_none=True)
    return AppliedUpdateResult(
        applied=applied,
        loss=loss,
        gradient_norm=norm,
        finite_logits=finite_logits,
        finite_gradients=finite_gradients,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
    )
