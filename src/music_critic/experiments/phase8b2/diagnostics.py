"""Bounded streaming anti-collapse and local-sensitivity diagnostics."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_DIAGNOSTICS_CONTRACT_VERSION,
    Phase8B2ContractError,
    fingerprint,
)


NODE_TYPES = ("note", "onset", "beat", "bar", "song")


def _matrix(value: object, node_type: str) -> Tensor | None:
    if value is None:
        return None
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise Phase8B2ContractError(
            f"phase8b2.diagnostics.embedding_invalid:{node_type}"
        )
    return value.detach().to(device="cpu", dtype=torch.float32)


def _representation_summary(value: Tensor | None) -> dict[str, object]:
    if value is None or value.shape[0] == 0:
        return {
            "available": False,
            "unavailable": {
                "category": "node_type_absent",
                "reason": "no representations for this node type",
            },
            "row_count": 0,
            "representation_variance": None,
            "effective_rank": None,
            "oversmoothing_adjacent_cosine": None,
            "zero_norm_count": 0,
        }
    norms = torch.linalg.vector_norm(value, dim=-1)
    zero_norm_count = int((norms == 0).sum().item())
    variance = (
        float(value.var(dim=0, unbiased=False).mean().item())
        if value.shape[0] >= 2
        else 0.0
    )
    centered = value - value.mean(dim=0, keepdim=True)
    # SVD is over N x D and returns only singular values; no N x N matrix and
    # no predictions are retained.
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if float(total.item()) == 0.0:
        effective_rank = 0.0
    else:
        probabilities = energy / total
        positive = probabilities > 0
        entropy = -(
            probabilities[positive] * probabilities[positive].log()
        ).sum()
        effective_rank = float(entropy.exp().item())
    if value.shape[0] < 2:
        cosine = None
        cosine_unavailable = {
            "category": "insufficient_rows",
            "reason": "adjacent cosine requires at least two rows",
        }
    else:
        cosine = float(
            F.cosine_similarity(value[:-1], value[1:], dim=-1).mean().item()
        )
        cosine_unavailable = None
    return {
        "available": True,
        "unavailable": None,
        "row_count": int(value.shape[0]),
        "feature_dimension": int(value.shape[1]),
        "representation_variance": variance,
        "effective_rank": effective_rank,
        "oversmoothing_adjacent_cosine": cosine,
        "oversmoothing_unavailable": cosine_unavailable,
        "zero_norm_count": zero_norm_count,
    }


def encoder_diagnostics(
    original: Mapping[str, Tensor | None],
    perturbed: Mapping[str, Tensor | None],
) -> dict[str, object]:
    """Summarize representations and one-note perturbation deltas."""

    unexpected = (set(original) | set(perturbed)) - set(NODE_TYPES)
    if unexpected:
        raise Phase8B2ContractError(
            "phase8b2.diagnostics.node_type_unknown:"
            + ",".join(sorted(unexpected))
        )
    groups: dict[str, object] = {}
    unavailable_count = 0
    retained_rows = 0
    for node_type in NODE_TYPES:
        before = _matrix(original.get(node_type), node_type)
        after = _matrix(perturbed.get(node_type), node_type)
        summary = _representation_summary(before)
        if before is None or after is None or before.shape != after.shape or (
            before.shape[0] == 0
        ):
            delta = None
            delta_unavailable = {
                "category": "perturbation_pair_unavailable",
                "reason": "original and perturbed representations must have equal non-empty shape",
            }
            unavailable_count += 1
        else:
            delta = float(
                torch.linalg.vector_norm(after - before, dim=-1).mean().item()
            )
            delta_unavailable = None
            retained_rows = max(retained_rows, int(before.shape[0]))
        groups[node_type] = {
            **summary,
            "single_note_perturbation_delta": delta,
            "perturbation_unavailable": delta_unavailable,
        }
    artifact = {
        "diagnostics_contract_version": (
            PHASE8B2_DIAGNOSTICS_CONTRACT_VERSION
        ),
        "diagnostic_only": True,
        "participates_in_primary_selection": False,
        "node_types": groups,
        "unavailable_group_count": unavailable_count,
        "unavailable_group_fraction": unavailable_count / len(NODE_TYPES),
        "pairwise_n_by_n_matrix_created": False,
        "retained_prediction_tensor_count": 0,
        "maximum_retained_representation_rows": retained_rows,
    }
    artifact["fingerprint"] = fingerprint(artifact)
    return artifact


__all__ = ["NODE_TYPES", "encoder_diagnostics"]
