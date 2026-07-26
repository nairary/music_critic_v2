"""Canonical local-sensitivity and within-store oversmoothing diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData

from music_critic.data import CanonicalPiece, validate_piece
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    build_raw_graph,
    graph_fingerprint,
    validate_raw_graph,
)
from music_critic.models.baseline import LocalHeterogeneousBaseline
from music_critic.models.encoder import MultiScaleEncoderOutput


@dataclass(frozen=True, slots=True)
class EmbeddingDelta:
    scale: str
    node_type: str
    node_index: int
    l2: float
    cosine_delta: float


@dataclass(frozen=True, slots=True)
class RawFeatureChange:
    """Exact changed raw feature rows after canonical graph rebuilding."""

    node_type: str
    feature_kind: Literal[
        "categorical",
        "categorical_availability",
        "continuous",
        "continuous_availability",
    ]
    feature_name: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OversmoothingValue:
    """Exact linear-time within-sample, within-node-store statistic."""

    scale: str
    sample_index: int
    node_type: str
    node_count: int
    zero_norm_count: int
    status: Literal["available", "fewer_than_two_nodes"]
    mean_pairwise_cosine: float | None
    policy: str = "exact_linear_normalized_sum"

    def __post_init__(self) -> None:
        if (
            isinstance(self.zero_norm_count, bool)
            or not isinstance(self.zero_norm_count, int)
            or not 0 <= self.zero_norm_count <= self.node_count
        ):
            raise ValueError(
                "oversmoothing zero_norm_count must lie within node count"
            )


@dataclass(frozen=True, slots=True)
class SingleNoteDiagnostic:
    """Canonical evidence that one changed note remains locally visible."""

    changed_note_identity: str
    changed_note_index: int
    original_graph_fingerprint: str
    perturbed_graph_fingerprint: str
    topology_equal: bool
    raw_feature_changes: tuple[RawFeatureChange, ...]
    deltas: tuple[EmbeddingDelta, ...]
    changed_node_counts: tuple[tuple[str, int], ...]
    reconstruction_logit_l2_delta: float
    reconstruction_loss_delta: float
    oversmoothing: tuple[OversmoothingValue, ...]
    interpretation: str = (
        "canonical local accessibility diagnostic only; no quality, anomaly, "
        "or musical-better label"
    )


def perturb_canonical_note_pitch(
    piece: CanonicalPiece,
    note_id: str,
    *,
    semitone_delta: int = 1,
) -> CanonicalPiece:
    """Return a validator-clean canonical pitch perturbation with stable ID."""

    if semitone_delta == 0:
        raise ValueError("pitch perturbation must be non-zero")
    source_report = validate_piece(piece)
    if source_report.errors:
        raise ValueError("original canonical piece is not validator-clean")
    matches = [note for note in piece.notes if note.note_id == note_id]
    if len(matches) != 1:
        raise ValueError("changed note identity must resolve exactly once")
    changed_pitch = matches[0].pitch + semitone_delta
    if not 0 <= changed_pitch <= 127:
        raise ValueError("perturbed canonical pitch is outside MIDI range")
    perturbed = replace(
        piece,
        notes=tuple(
            replace(note, pitch=changed_pitch)
            if note.note_id == note_id
            else note
            for note in piece.notes
        ),
    )
    result_report = validate_piece(perturbed)
    if result_report.errors:
        raise ValueError("perturbed canonical piece is not validator-clean")
    return perturbed


def _cosine_delta(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((1.0 - F.cosine_similarity(left[None], right[None])).item())


def _linear_mean_pairwise_cosine(
    values: Tensor,
) -> tuple[float | None, int]:
    """Match dense off-diagonal cosine in O(ND) time and O(D) memory."""

    count = values.shape[0]
    total = torch.zeros(
        values.shape[1],
        dtype=values.dtype,
        device=values.device,
    )
    diagonal_sum = torch.zeros(
        (), dtype=values.dtype, device=values.device
    )
    zero_norm_count = torch.zeros(
        (), dtype=torch.long, device=values.device
    )
    detached = values.detach()
    for index in range(count):
        row = detached[index]
        zero_norm_count.add_(
            (torch.linalg.vector_norm(row) == 0).to(torch.long)
        )
        normalized_row = F.normalize(row, dim=0)
        total.add_(normalized_row)
        diagonal_sum.add_(normalized_row.square().sum())
    zero_norm_count_value = int(zero_norm_count.item())
    if count < 2:
        return None, zero_norm_count_value
    pair_sum = total.square().sum() - diagonal_sum
    mean = pair_sum / (count * (count - 1))
    return float(mean.item()), zero_norm_count_value


def _scales(
    output: MultiScaleEncoderOutput,
) -> tuple[tuple[str, object], ...]:
    return (
        ("feature", output.feature_output),
        *(
            (f"gnn_layer_{index + 1}", layer)
            for index, layer in enumerate(output.layer_outputs)
        ),
        ("final_skip", output.final_output),
    )


def oversmoothing_by_group(
    output: MultiScaleEncoderOutput,
) -> tuple[OversmoothingValue, ...]:
    """Report only within-sample and within-node-type cosine statistics."""

    song_membership = output.final_output.batch_membership["song"]
    sample_count = (
        int(song_membership.max().item()) + 1
        if song_membership.numel()
        else 0
    )
    values = []
    for scale, scale_output in _scales(output):
        for sample_index in range(sample_count):
            for node_type in MANDATORY_NODE_TYPES:
                membership = scale_output.batch_membership[node_type]
                group = scale_output.embeddings[node_type][
                    membership == sample_index
                ]
                mean, zero_norm_count = _linear_mean_pairwise_cosine(group)
                values.append(
                    OversmoothingValue(
                        scale=scale,
                        sample_index=sample_index,
                        node_type=node_type,
                        node_count=group.shape[0],
                        zero_norm_count=zero_norm_count,
                        status=(
                            "available"
                            if mean is not None
                            else "fewer_than_two_nodes"
                        ),
                        mean_pairwise_cosine=mean,
                    )
                )
    return tuple(values)


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


def _raw_feature_changes(
    original: HeteroData,
    perturbed: HeteroData,
) -> tuple[RawFeatureChange, ...]:
    changes = []
    attributes = (
        ("categorical", "x_cat", "cat_feature_names"),
        (
            "categorical_availability",
            "x_cat_available",
            "cat_feature_names",
        ),
        ("continuous", "x_cont", "cont_feature_names"),
        (
            "continuous_availability",
            "x_cont_available",
            "cont_feature_names",
        ),
    )
    for node_type in MANDATORY_NODE_TYPES:
        for feature_kind, tensor_name, names_name in attributes:
            left = getattr(original[node_type], tensor_name)
            right = getattr(perturbed[node_type], tensor_name)
            names = getattr(original[node_type], names_name)
            for column, feature_name in enumerate(names):
                rows = torch.nonzero(
                    left[:, column] != right[:, column], as_tuple=False
                ).flatten()
                if rows.numel():
                    changes.append(
                        RawFeatureChange(
                            node_type=node_type,
                            feature_kind=feature_kind,
                            feature_name=feature_name,
                            entity_ids=tuple(
                                original[node_type].entity_id[index]
                                for index in rows.tolist()
                            ),
                        )
                    )
    return tuple(changes)


def _topology_equal(left: HeteroData, right: HeteroData) -> bool:
    return all(
        left[node_type].entity_id == right[node_type].entity_id
        and left[node_type].num_nodes == right[node_type].num_nodes
        for node_type in MANDATORY_NODE_TYPES
    ) and all(
        torch.equal(left[edge_type].edge_index, right[edge_type].edge_index)
        for edge_type in MANDATORY_EDGE_TYPES
    )


@torch.no_grad()
def single_note_sensitivity(
    model: LocalHeterogeneousBaseline,
    original_piece: CanonicalPiece,
    perturbed_piece: CanonicalPiece,
    *,
    note_id: str,
) -> SingleNoteDiagnostic:
    """Build both production graphs from canonical evidence and compare."""

    if validate_piece(original_piece).errors or validate_piece(
        perturbed_piece
    ).errors:
        raise ValueError("single-note diagnostic requires validator-clean pieces")
    original_graph = build_raw_graph(original_piece)
    perturbed_graph = build_raw_graph(perturbed_piece)
    validate_raw_graph(original_graph)
    validate_raw_graph(perturbed_graph)
    original_fingerprint = graph_fingerprint(original_graph)
    perturbed_fingerprint = graph_fingerprint(perturbed_graph)
    if original_fingerprint == perturbed_fingerprint:
        raise ValueError("canonical perturbation did not change raw graph identity")
    try:
        note_index = original_graph["note"].entity_id.index(note_id)
    except ValueError as exc:
        raise ValueError("changed note identity is absent from raw graph") from exc
    if perturbed_graph["note"].entity_id[note_index] != note_id:
        raise ValueError("canonical perturbation changed stable note ordering")

    model.eval()
    batch = Batch.from_data_list([original_graph, perturbed_graph])
    encoded = model.encode(batch, return_layers=True)
    scale_values = _scales(encoded)
    neighbors = _neighbor_indices(original_graph, note_index)
    deltas = []
    for scale, scale_output in scale_values:
        for node_type, local_index in sorted(neighbors.items()):
            original_index = int(batch[node_type].ptr[0]) + local_index
            perturbed_index = int(batch[node_type].ptr[1]) + local_index
            before = scale_output.embeddings[node_type][original_index]
            after = scale_output.embeddings[node_type][perturbed_index]
            deltas.append(
                EmbeddingDelta(
                    scale=scale,
                    node_type=node_type,
                    node_index=local_index,
                    l2=float(torch.linalg.vector_norm(before - after).item()),
                    cosine_delta=_cosine_delta(before, after),
                )
            )
    changed_counts = []
    for node_type in MANDATORY_NODE_TYPES:
        left_start = int(batch[node_type].ptr[0])
        middle = int(batch[node_type].ptr[1])
        right_end = int(batch[node_type].ptr[2])
        difference = torch.linalg.vector_norm(
            encoded.final_output.embeddings[node_type][left_start:middle]
            - encoded.final_output.embeddings[node_type][middle:right_end],
            dim=-1,
        )
        changed_counts.append(
            (node_type, int((difference > 0).sum().item()))
        )
    reconstruction = model.reconstruction_heads(
        encoded.final_output, batch
    )
    pitch = next(
        output
        for output in reconstruction
        if (output.node_type, output.feature_name) == ("note", "pitch")
    )
    original_note_index = int(batch["note"].ptr[0]) + note_index
    perturbed_note_index = int(batch["note"].ptr[1]) + note_index
    return SingleNoteDiagnostic(
        changed_note_identity=note_id,
        changed_note_index=note_index,
        original_graph_fingerprint=original_fingerprint,
        perturbed_graph_fingerprint=perturbed_fingerprint,
        topology_equal=_topology_equal(original_graph, perturbed_graph),
        raw_feature_changes=_raw_feature_changes(
            original_graph, perturbed_graph
        ),
        deltas=tuple(deltas),
        changed_node_counts=tuple(changed_counts),
        reconstruction_logit_l2_delta=float(
            torch.linalg.vector_norm(
                pitch.logits[original_note_index]
                - pitch.logits[perturbed_note_index]
            ).item()
        ),
        reconstruction_loss_delta=float(
            (
                pitch.per_node_loss[perturbed_note_index]
                - pitch.per_node_loss[original_note_index]
            ).item()
        ),
        oversmoothing=oversmoothing_by_group(encoded),
    )


__all__ = [
    "EmbeddingDelta",
    "OversmoothingValue",
    "RawFeatureChange",
    "SingleNoteDiagnostic",
    "oversmoothing_by_group",
    "perturb_canonical_note_pitch",
    "single_note_sensitivity",
]
