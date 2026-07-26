"""Raw feature encoding and local relation-aware message passing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData

from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    validate_raw_graph,
    validate_raw_graph_batch,
)
from music_critic.graph.feature_registry import FeatureSpec
from music_critic.models.contracts import ENCODER_OUTPUT_VERSION


def normalize_continuous(values: Tensor, spec: FeatureSpec) -> Tensor:
    """Apply the versioned Phase 6A bounded scalar transform.

    Log-count fields use signed ``log1p``. Other raw scalars (including
    already within-track z-scored values) use ``x / (1 + |x|)``. Availability
    is represented separately and unavailable scalar values are zeroed before
    projection.
    """

    if spec.normalization == "log1p":
        return torch.sign(values) * torch.log1p(torch.abs(values))
    return values / (1.0 + torch.abs(values))


def _validate_input_graph(graph: HeteroData) -> None:
    if isinstance(graph, Batch):
        validate_raw_graph_batch(graph, sample_count=int(graph.num_graphs))
    else:
        validate_raw_graph(graph)


def _batch_membership(graph: HeteroData, node_type: str) -> Tensor:
    store = graph[node_type]
    if hasattr(store, "batch"):
        return store.batch
    return torch.zeros(
        int(store.num_nodes),
        dtype=torch.long,
        device=store.x_cont.device,
    )


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """One-row-per-input-node local embeddings at one encoder scale."""

    contract_version: str
    embeddings: Mapping[str, Tensor]
    batch_membership: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.contract_version != ENCODER_OUTPUT_VERSION:
            raise ValueError("encoder output contract version is incompatible")
        if tuple(self.embeddings) != MANDATORY_NODE_TYPES:
            raise ValueError("encoder output must retain every mandatory node type")
        if tuple(self.batch_membership) != MANDATORY_NODE_TYPES:
            raise ValueError("encoder output batch membership is incomplete")
        hidden_dims = set()
        for node_type in MANDATORY_NODE_TYPES:
            values = self.embeddings[node_type]
            membership = self.batch_membership[node_type]
            if values.ndim != 2 or membership.ndim != 1:
                raise ValueError("encoder tensors have invalid rank")
            if values.shape[0] != membership.shape[0]:
                raise ValueError("encoder output changed node cardinality")
            hidden_dims.add(int(values.shape[1]))
        if len(hidden_dims) != 1:
            raise ValueError("all encoder stores must share one hidden dimension")


@dataclass(frozen=True, slots=True)
class MultiScaleEncoderOutput:
    """Feature scale, optional GNN layers, and residual-preserving final scale."""

    contract_version: str
    feature_output: EncoderOutput
    layer_outputs: tuple[EncoderOutput, ...]
    final_output: EncoderOutput

    def __post_init__(self) -> None:
        if self.contract_version != ENCODER_OUTPUT_VERSION:
            raise ValueError("multiscale encoder output version is incompatible")
        expected = self.feature_output.batch_membership
        for output in (*self.layer_outputs, self.final_output):
            for node_type in MANDATORY_NODE_TYPES:
                if not torch.equal(
                    output.batch_membership[node_type], expected[node_type]
                ):
                    raise ValueError("encoder scales changed batch membership")


class _NodeFeatureEncoder(nn.Module):
    def __init__(self, node_type: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_type = node_type
        self.categorical_specs = RAW_FEATURE_REGISTRY.for_node(
            node_type, "categorical"
        )
        self.continuous_specs = RAW_FEATURE_REGISTRY.for_node(
            node_type, "continuous"
        )
        self.categorical = nn.ModuleList(
            nn.Embedding(int(spec.vocabulary_size or 0), hidden_dim)
            for spec in self.categorical_specs
        )
        self.categorical_availability = nn.ModuleList(
            nn.Embedding(2, hidden_dim) for _ in self.categorical_specs
        )
        self.continuous = nn.ModuleList(
            nn.Linear(1, hidden_dim) for _ in self.continuous_specs
        )
        self.continuous_availability = nn.ModuleList(
            nn.Embedding(2, hidden_dim) for _ in self.continuous_specs
        )
        self.node_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, store: object) -> Tensor:
        count = int(store.num_nodes)
        encoded = self.node_bias.expand(count, -1)
        for column, (spec, embedding, availability_embedding) in enumerate(
            zip(
                self.categorical_specs,
                self.categorical,
                self.categorical_availability,
            )
        ):
            del spec
            available = store.x_cat_available[:, column]
            encoded = encoded + embedding(store.x_cat[:, column])
            encoded = encoded + availability_embedding(available.long())
        for column, (spec, projection, availability_embedding) in enumerate(
            zip(
                self.continuous_specs,
                self.continuous,
                self.continuous_availability,
            )
        ):
            available = store.x_cont_available[:, column]
            values = normalize_continuous(store.x_cont[:, column], spec)
            values = torch.where(available, values, torch.zeros_like(values))
            encoded = encoded + projection(values.unsqueeze(-1))
            encoded = encoded + availability_embedding(available.long())
        return self.dropout(self.activation(self.normalization(encoded)))


class RawFeatureEncoder(nn.Module):
    """Per-feature, per-node-type raw-only Phase 3A encoder."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_encoders = nn.ModuleDict(
            {
                node_type: _NodeFeatureEncoder(node_type, hidden_dim, dropout)
                for node_type in MANDATORY_NODE_TYPES
            }
        )

    def forward(self, graph: HeteroData) -> EncoderOutput:
        _validate_input_graph(graph)
        embeddings = {
            node_type: self.node_encoders[node_type](graph[node_type])
            for node_type in MANDATORY_NODE_TYPES
        }
        membership = {
            node_type: _batch_membership(graph, node_type)
            for node_type in MANDATORY_NODE_TYPES
        }
        return EncoderOutput(
            contract_version=ENCODER_OUTPUT_VERSION,
            embeddings=embeddings,
            batch_membership=membership,
        )


def _relation_key(index: int) -> str:
    return f"relation_{index:02d}"


class LocalRelationLayer(nn.Module):
    """Inspectable relation-specific sum aggregation with no edge-wise loop."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        *,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.relation_projections = nn.ModuleDict(
            {
                _relation_key(index): nn.Linear(hidden_dim, hidden_dim, bias=False)
                for index, _ in enumerate(MANDATORY_EDGE_TYPES)
            }
        )
        self.self_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim, hidden_dim)
                for node_type in MANDATORY_NODE_TYPES
            }
        )
        self.normalizations = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in MANDATORY_NODE_TYPES
            }
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, embeddings: Mapping[str, Tensor], graph: HeteroData
    ) -> dict[str, Tensor]:
        aggregated = {
            node_type: torch.zeros_like(embeddings[node_type])
            for node_type in MANDATORY_NODE_TYPES
        }
        for relation_index, edge_type in enumerate(MANDATORY_EDGE_TYPES):
            source_type, _, destination_type = edge_type
            edge_index = graph[edge_type].edge_index
            if edge_index.shape[1] == 0:
                continue
            source_index, destination_index = edge_index
            messages = self.relation_projections[
                _relation_key(relation_index)
            ](embeddings[source_type].index_select(0, source_index))
            aggregated[destination_type].index_add_(
                0, destination_index, messages
            )
        output = {}
        for node_type in MANDATORY_NODE_TYPES:
            values = self.self_projections[node_type](embeddings[node_type])
            values = values + aggregated[node_type]
            if self.residual:
                values = values + embeddings[node_type]
            output[node_type] = self.dropout(
                self.activation(self.normalizations[node_type](values))
            )
        return output


class LocalHeterogeneousEncoder(nn.Module):
    """Shared feature encoder with optional exact-relation message passing."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        gnn_layers: int,
        dropout: float,
        residual: bool,
        use_message_passing: bool,
    ) -> None:
        super().__init__()
        self.feature_encoder = RawFeatureEncoder(hidden_dim, dropout)
        self.layers = nn.ModuleList(
            LocalRelationLayer(hidden_dim, dropout, residual=residual)
            for _ in range(gnn_layers if use_message_passing else 0)
        )
        self.final_skip = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim * 2, hidden_dim)
                for node_type in MANDATORY_NODE_TYPES
            }
        )
        self.final_norm = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in MANDATORY_NODE_TYPES
            }
        )
        self.activation = nn.GELU()

    def forward(
        self, graph: HeteroData, *, return_layers: bool = False
    ) -> MultiScaleEncoderOutput:
        feature_output = self.feature_encoder(graph)
        current = dict(feature_output.embeddings)
        layer_outputs = []
        for layer in self.layers:
            current = layer(current, graph)
            if return_layers:
                layer_outputs.append(
                    EncoderOutput(
                        contract_version=ENCODER_OUTPUT_VERSION,
                        embeddings=dict(current),
                        batch_membership=feature_output.batch_membership,
                    )
                )
        final_embeddings = {
            node_type: self.activation(
                self.final_norm[node_type](
                    self.final_skip[node_type](
                        torch.cat(
                            (
                                feature_output.embeddings[node_type],
                                current[node_type],
                            ),
                            dim=-1,
                        )
                    )
                )
            )
            for node_type in MANDATORY_NODE_TYPES
        }
        final = EncoderOutput(
            contract_version=ENCODER_OUTPUT_VERSION,
            embeddings=final_embeddings,
            batch_membership=feature_output.batch_membership,
        )
        return MultiScaleEncoderOutput(
            contract_version=ENCODER_OUTPUT_VERSION,
            feature_output=feature_output,
            layer_outputs=tuple(layer_outputs),
            final_output=final,
        )


__all__ = [
    "EncoderOutput",
    "LocalHeterogeneousEncoder",
    "LocalRelationLayer",
    "MultiScaleEncoderOutput",
    "RawFeatureEncoder",
    "normalize_continuous",
]
