"""Deterministic local-sensitivity and shallow oversmoothing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.baseline import LocalHeterogeneousBaseline


@dataclass(frozen=True, slots=True)
class EmbeddingDelta:
    scale: str
    node_type: str
    node_index: int
    l2: float
    cosine_delta: float


@dataclass(frozen=True, slots=True)
class OversmoothingValue:
    scale: str
    original_mean_pairwise_cosine: float
    perturbed_mean_pairwise_cosine: float


@dataclass(frozen=True, slots=True)
class SingleNoteDiagnostic:
    """Evidence that a changed note row and its local effects remain visible."""

    changed_note_identity: str
    changed_note_index: int
    deltas: tuple[EmbeddingDelta, ...]
    changed_node_counts: tuple[tuple[str, int], ...]
    reconstruction_logit_l2_delta: float
    reconstruction_loss_delta: float
    oversmoothing: tuple[OversmoothingValue, ...]
    interpretation: str = (
        "local accessibility diagnostic only; no quality or musical-better label"
    )


def _as_batch(graph: HeteroData) -> Batch:
    return graph if isinstance(graph, Batch) else Batch.from_data_list([graph])


def _cosine_delta(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((1.0 - F.cosine_similarity(left[None], right[None])).item())


def _mean_pairwise_cosine(embeddings: dict[str, Tensor]) -> float:
    values = torch.cat(
        [embeddings[node_type] for node_type in MANDATORY_NODE_TYPES], dim=0
    )
    if values.shape[0] < 2:
        return 1.0
    normalized = F.normalize(values, dim=-1)
    similarities = normalized @ normalized.t()
    mask = ~torch.eye(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    return float(similarities[mask].mean().item())


def _neighbor_indices(graph: HeteroData, note_index: int) -> dict[str, int]:
    neighbors = {"note": note_index}
    for edge_type, node_type in (
        (("note", "in_onset", "onset"), "onset"),
        (("note", "belongs_to_bar", "bar"), "bar"),
    ):
        edge_index = graph[edge_type].edge_index
        positions = torch.nonzero(
            edge_index[0] == note_index, as_tuple=False
        ).flatten()
        if positions.numel():
            neighbors[node_type] = int(edge_index[1, positions[0]].item())
    onset = neighbors.get("onset")
    if onset is not None:
        edge_index = graph[
            ("onset", "belongs_to_beat", "beat")
        ].edge_index
        positions = torch.nonzero(
            edge_index[0] == onset, as_tuple=False
        ).flatten()
        if positions.numel():
            neighbors["beat"] = int(edge_index[1, positions[0]].item())
    return neighbors


@torch.no_grad()
def single_note_sensitivity(
    model: LocalHeterogeneousBaseline,
    original_graph: HeteroData,
    perturbed_graph: HeteroData,
    *,
    note_index: int,
) -> SingleNoteDiagnostic:
    """Compare two valid raw graphs that differ in one note observation."""

    model.eval()
    original = _as_batch(original_graph)
    perturbed = _as_batch(perturbed_graph)
    left = model.encode(original, return_layers=True)
    right = model.encode(perturbed, return_layers=True)
    left_scales = (
        ("feature", left.feature_output),
        *(
            (f"gnn_layer_{index + 1}", output)
            for index, output in enumerate(left.layer_outputs)
        ),
        ("final_skip", left.final_output),
    )
    right_scales = (
        ("feature", right.feature_output),
        *(
            (f"gnn_layer_{index + 1}", output)
            for index, output in enumerate(right.layer_outputs)
        ),
        ("final_skip", right.final_output),
    )
    neighbors = _neighbor_indices(original, note_index)
    deltas = []
    for (scale, left_output), (_, right_output) in zip(
        left_scales, right_scales
    ):
        for node_type, node_index in sorted(neighbors.items()):
            before = left_output.embeddings[node_type][node_index]
            after = right_output.embeddings[node_type][node_index]
            deltas.append(
                EmbeddingDelta(
                    scale=scale,
                    node_type=node_type,
                    node_index=node_index,
                    l2=float(torch.linalg.vector_norm(before - after).item()),
                    cosine_delta=_cosine_delta(before, after),
                )
            )
    changed_counts = []
    for node_type in MANDATORY_NODE_TYPES:
        difference = torch.linalg.vector_norm(
            left.final_output.embeddings[node_type]
            - right.final_output.embeddings[node_type],
            dim=-1,
        )
        changed_counts.append(
            (node_type, int((difference > 0).sum().item()))
        )
    left_reconstruction = model.reconstruction_heads(
        left.final_output, original
    )
    right_reconstruction = model.reconstruction_heads(
        right.final_output, perturbed
    )
    left_pitch = next(
        output
        for output in left_reconstruction
        if (output.node_type, output.feature_name) == ("note", "pitch")
    )
    right_pitch = next(
        output
        for output in right_reconstruction
        if (output.node_type, output.feature_name) == ("note", "pitch")
    )
    oversmoothing = tuple(
        OversmoothingValue(
            scale=scale,
            original_mean_pairwise_cosine=_mean_pairwise_cosine(
                dict(left_output.embeddings)
            ),
            perturbed_mean_pairwise_cosine=_mean_pairwise_cosine(
                dict(right_output.embeddings)
            ),
        )
        for (scale, left_output), (_, right_output) in zip(
            left_scales, right_scales
        )
    )
    identity = original["note"].entity_id[note_index]
    if isinstance(identity, (tuple, list)):
        if len(identity) != 1 or not isinstance(identity[0], str):
            raise ValueError("batched note identity is not a single stable string")
        identity = identity[0]
    if not isinstance(identity, str):
        raise ValueError("note identity is not a stable string")
    return SingleNoteDiagnostic(
        changed_note_identity=identity,
        changed_note_index=note_index,
        deltas=tuple(deltas),
        changed_node_counts=tuple(changed_counts),
        reconstruction_logit_l2_delta=float(
            torch.linalg.vector_norm(
                left_pitch.logits[note_index]
                - right_pitch.logits[note_index]
            ).item()
        ),
        reconstruction_loss_delta=float(
            (
                right_pitch.per_node_loss[note_index]
                - left_pitch.per_node_loss[note_index]
            ).item()
        ),
        oversmoothing=oversmoothing,
    )


__all__ = [
    "EmbeddingDelta",
    "OversmoothingValue",
    "SingleNoteDiagnostic",
    "single_note_sensitivity",
]
