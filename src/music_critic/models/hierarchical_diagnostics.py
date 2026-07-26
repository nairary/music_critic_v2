"""Canonical local/coarse/fused accessibility evidence for Phase 6B."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch

from music_critic.data import CanonicalPiece, validate_piece
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    build_raw_graph,
    graph_fingerprint,
    validate_raw_graph,
)
from music_critic.models.hierarchical_baseline import (
    HierarchicalHeterogeneousBaseline,
)


@dataclass(frozen=True, slots=True)
class HierarchicalEmbeddingDelta:
    """One explicitly named local, coarse, contextual, or fused delta."""

    stage: str
    node_type: str
    node_index: int
    l2: float
    cosine_delta: float


@dataclass(frozen=True, slots=True)
class HierarchicalSingleNoteDiagnostic:
    """Accessibility evidence without an anomaly, quality, or magnitude label."""

    changed_note_identity: str
    changed_note_index: int
    original_graph_fingerprint: str
    perturbed_graph_fingerprint: str
    topology_equal: bool
    ownership_equal: bool
    cardinality_equal: bool
    local_note_retained: bool
    unrelated_sample_unchanged: bool
    deltas: tuple[HierarchicalEmbeddingDelta, ...]
    reconstruction_logit_l2_delta: float
    interpretation: str = (
        "Phase 6B accessibility diagnostic only; no threshold, anomaly, "
        "quality, or musical-better label"
    )


def _delta(
    stage: str,
    node_type: str,
    node_index: int,
    left: Tensor,
    right: Tensor,
) -> HierarchicalEmbeddingDelta:
    return HierarchicalEmbeddingDelta(
        stage=stage,
        node_type=node_type,
        node_index=node_index,
        l2=float(torch.linalg.vector_norm(left - right).item()),
        cosine_delta=float(
            (
                1.0
                - F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0))
            ).item()
        ),
    )


def _topology_equal(left: object, right: object) -> bool:
    return all(
        left[node_type].entity_id == right[node_type].entity_id
        and int(left[node_type].num_nodes)
        == int(right[node_type].num_nodes)
        for node_type in MANDATORY_NODE_TYPES
    ) and all(
        torch.equal(
            left[edge_type].edge_index,
            right[edge_type].edge_index,
        )
        for edge_type in MANDATORY_EDGE_TYPES
    )


def _related_rows(graph: object, note_index: int) -> dict[str, int]:
    result = {"note": note_index}
    relations = (
        (("note", "in_onset", "onset"), "onset"),
        (("note", "belongs_to_bar", "bar"), "bar"),
        (("note", "belongs_to_track", "track"), "track"),
    )
    for edge_type, node_type in relations:
        edge_index = graph[edge_type].edge_index
        positions = torch.nonzero(
            edge_index[0] == note_index, as_tuple=False
        ).flatten()
        if positions.numel() != 1:
            raise ValueError(
                "changed note relation does not resolve exactly once"
            )
        result[node_type] = int(edge_index[1, positions[0]].item())
    onset = result["onset"]
    edge_index = graph[
        ("onset", "belongs_to_beat", "beat")
    ].edge_index
    positions = torch.nonzero(
        edge_index[0] == onset, as_tuple=False
    ).flatten()
    if positions.numel() != 1:
        raise ValueError("changed onset does not resolve one beat")
    result["beat"] = int(edge_index[1, positions[0]].item())
    return result


def _global_index(
    batch: Batch, node_type: str, sample_index: int, local_index: int
) -> int:
    return int(batch[node_type].ptr[sample_index]) + local_index


def _sample_rows_equal(
    left: Tensor,
    right: Tensor,
    batch: Batch,
    node_type: str,
    left_sample: int,
    right_sample: int,
) -> bool:
    left_start = int(batch[node_type].ptr[left_sample])
    left_end = int(batch[node_type].ptr[left_sample + 1])
    right_start = int(batch[node_type].ptr[right_sample])
    right_end = int(batch[node_type].ptr[right_sample + 1])
    return torch.equal(
        left[left_start:left_end], right[right_start:right_end]
    )


@torch.no_grad()
def hierarchical_single_note_sensitivity(
    model: HierarchicalHeterogeneousBaseline,
    original_piece: CanonicalPiece,
    perturbed_piece: CanonicalPiece,
    *,
    note_id: str,
    unrelated_piece: CanonicalPiece | None = None,
) -> HierarchicalSingleNoteDiagnostic:
    """Compare validator-clean pieces inside one isolated multi-sample batch."""

    unrelated_piece = unrelated_piece or original_piece
    for piece in (original_piece, perturbed_piece, unrelated_piece):
        if validate_piece(piece).errors:
            raise ValueError(
                "hierarchical diagnostic requires validator-clean pieces"
            )
    original_graph = build_raw_graph(original_piece)
    perturbed_graph = build_raw_graph(perturbed_piece)
    unrelated_graph = build_raw_graph(unrelated_piece)
    unrelated_control_graph = build_raw_graph(unrelated_piece)
    for graph in (
        original_graph,
        perturbed_graph,
        unrelated_graph,
        unrelated_control_graph,
    ):
        validate_raw_graph(graph)
    original_fingerprint = graph_fingerprint(original_graph)
    perturbed_fingerprint = graph_fingerprint(perturbed_graph)
    if original_fingerprint == perturbed_fingerprint:
        raise ValueError("canonical perturbation did not change raw graph")
    try:
        note_index = original_graph["note"].entity_id.index(note_id)
    except ValueError as exc:
        raise ValueError("changed note identity is absent") from exc
    if perturbed_graph["note"].entity_id[note_index] != note_id:
        raise ValueError("stable changed-note ordering was not preserved")

    # Samples 1 and 3 are byte-identical unrelated controls inside the same
    # Transformer batch; samples 0 and 2 are the original/perturbed pair.
    batch = Batch.from_data_list(
        [
            original_graph,
            unrelated_graph,
            perturbed_graph,
            unrelated_control_graph,
        ]
    )
    model.eval()
    encoded = model.encode(batch, return_layers=True)
    related = _related_rows(original_graph, note_index)
    left_sample = 0
    right_sample = 2
    deltas = []

    local_note_left = _global_index(
        batch, "note", left_sample, note_index
    )
    local_note_right = _global_index(
        batch, "note", right_sample, note_index
    )
    local = encoded.local_encoder.final_output.embeddings
    deltas.append(
        _delta(
            "phase6a_local",
            "note",
            note_index,
            local["note"][local_note_left],
            local["note"][local_note_right],
        )
    )

    for node_type, stage, values in (
        ("bar", "pooled", encoded.pooling.bar_tokens),
        ("track", "pooled", encoded.pooling.track_tokens),
        ("bar", "contextual", encoded.coarse.bar_embeddings),
        ("track", "contextual", encoded.coarse.track_embeddings),
    ):
        local_index = related[node_type]
        left_index = _global_index(
            batch, node_type, left_sample, local_index
        )
        right_index = _global_index(
            batch, node_type, right_sample, local_index
        )
        deltas.append(
            _delta(
                stage,
                node_type,
                local_index,
                values[left_index],
                values[right_index],
            )
        )
    deltas.append(
        _delta(
            "contextual",
            "song",
            0,
            encoded.coarse.song_embeddings[left_sample],
            encoded.coarse.song_embeddings[right_sample],
        )
    )
    for node_type in ("note", "onset", "beat", "bar", "track"):
        local_index = related[node_type]
        left_index = _global_index(
            batch, node_type, left_sample, local_index
        )
        right_index = _global_index(
            batch, node_type, right_sample, local_index
        )
        deltas.append(
            _delta(
                "fused",
                node_type,
                local_index,
                encoded.fused.embeddings[node_type][left_index],
                encoded.fused.embeddings[node_type][right_index],
            )
        )

    original_owner = {
        name: int(values[_global_index(
            batch,
            {
                "beat_to_bar": "beat",
                "onset_to_bar": "onset",
                "note_to_bar": "note",
                "note_to_track": "note",
                "bar_to_song": "bar",
                "track_to_song": "track",
            }[name],
            left_sample,
            (
                related["beat"]
                if name == "beat_to_bar"
                else related["onset"]
                if name == "onset_to_bar"
                else note_index
                if name.startswith("note_")
                else related["bar"]
                if name == "bar_to_song"
                else related["track"]
            ),
        )].item())
        for name, values in encoded.pooling.ownership.owners.items()
    }
    perturbed_owner = {
        name: int(values[_global_index(
            batch,
            {
                "beat_to_bar": "beat",
                "onset_to_bar": "onset",
                "note_to_bar": "note",
                "note_to_track": "note",
                "bar_to_song": "bar",
                "track_to_song": "track",
            }[name],
            right_sample,
            (
                related["beat"]
                if name == "beat_to_bar"
                else related["onset"]
                if name == "onset_to_bar"
                else note_index
                if name.startswith("note_")
                else related["bar"]
                if name == "bar_to_song"
                else related["track"]
            ),
        )].item())
        for name, values in encoded.pooling.ownership.owners.items()
    }
    # Normalize global owner rows by the corresponding sample parent offset.
    parent_type = {
        "beat_to_bar": "bar",
        "onset_to_bar": "bar",
        "note_to_bar": "bar",
        "note_to_track": "track",
        "bar_to_song": "song",
        "track_to_song": "song",
    }
    ownership_equal = all(
        original_owner[name]
        - int(batch[parent_type[name]].ptr[left_sample])
        == perturbed_owner[name]
        - int(batch[parent_type[name]].ptr[right_sample])
        for name in original_owner
    )
    cardinality_equal = all(
        int(original_graph[node_type].num_nodes)
        == int(perturbed_graph[node_type].num_nodes)
        for node_type in MANDATORY_NODE_TYPES
    )
    unrelated_unchanged = all(
        _sample_rows_equal(
            encoded.local_encoder.final_output.embeddings[node_type],
            encoded.local_encoder.final_output.embeddings[node_type],
            batch,
            node_type,
            1,
            3,
        )
        and _sample_rows_equal(
            encoded.fused.embeddings[node_type],
            encoded.fused.embeddings[node_type],
            batch,
            node_type,
            1,
            3,
        )
        for node_type in MANDATORY_NODE_TYPES
    ) and all(
        _sample_rows_equal(
            values,
            values,
            batch,
            node_type,
            1,
            3,
        )
        for node_type, values in (
            ("bar", encoded.pooling.bar_tokens),
            ("track", encoded.pooling.track_tokens),
            ("bar", encoded.coarse.bar_embeddings),
            ("track", encoded.coarse.track_embeddings),
        )
    ) and torch.equal(
        encoded.coarse.song_embeddings[1],
        encoded.coarse.song_embeddings[3],
    )

    reconstruction = model.local_baseline.reconstruction_heads(
        encoded.fused, batch
    )
    pitch = next(
        value
        for value in reconstruction
        if (value.node_type, value.feature_name) == ("note", "pitch")
    )
    return HierarchicalSingleNoteDiagnostic(
        changed_note_identity=note_id,
        changed_note_index=note_index,
        original_graph_fingerprint=original_fingerprint,
        perturbed_graph_fingerprint=perturbed_fingerprint,
        topology_equal=_topology_equal(
            original_graph, perturbed_graph
        ),
        ownership_equal=ownership_equal,
        cardinality_equal=cardinality_equal,
        local_note_retained=(
            encoded.local_encoder.final_output.embeddings["note"].shape
            == encoded.fused.embeddings["note"].shape
        ),
        unrelated_sample_unchanged=unrelated_unchanged,
        deltas=tuple(deltas),
        reconstruction_logit_l2_delta=float(
            torch.linalg.vector_norm(
                pitch.logits[local_note_left]
                - pitch.logits[local_note_right]
            ).item()
        ),
    )


__all__ = [
    "HierarchicalEmbeddingDelta",
    "HierarchicalSingleNoteDiagnostic",
    "hierarchical_single_note_sensitivity",
]
