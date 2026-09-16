"""Bounded training primitive, not a campaign launcher or live-run integration."""
from dataclasses import replace
import math

import torch

from music_critic.experiments.analysisgnn.corrected_training import (
    AppliedUpdateResult, gradient_norm, transpose_raw_graph_batch_directed,
)
from .alignment import align_temporal_targets, temporal_supervised_loss


def transpose_temporal_batch(graph, transforms):
    """Apply the established directed C2 transform only to raw pitch features.

    Timing, node identity and temporal topology are pitch-independent. The
    caller supplies the same applied shifts to post-prediction alignment.
    """
    moved = replace(graph, base=transpose_raw_graph_batch_directed(graph.base, transforms))
    moved.validate()
    return moved


def attempt_temporal_update(*, model, graph, sidecars, shifts, class_weights,
                            optimizer, scheduler, scaler, gradient_clip_norm=1.):
    optimizer.zero_grad(set_to_none=True)
    output = model(graph)
    alignment = align_temporal_targets(output, sidecars, shifts=shifts)
    loss = temporal_supervised_loss(output, alignment, class_weights)
    if loss.total is None or not loss.fp32_boundary:
        raise ValueError("no temporal supervision or non-fp32 loss")
    finite_logits = all(bool(torch.isfinite(v).all()) for v in output.logits.values())
    if not finite_logits or not bool(torch.isfinite(loss.total)):
        raise ValueError("nonfinite temporal forward")
    before = float(scaler.get_scale())
    scaler.scale(loss.total).backward()
    scaler.unscale_(optimizer)
    finite_gradients = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    norm = gradient_norm(model)
    if not finite_gradients or not math.isfinite(norm):
        raise ValueError("nonfinite temporal gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    applied = after >= before
    if applied:
        scheduler.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return AppliedUpdateResult(applied, loss, norm, finite_logits, finite_gradients, before, after)
