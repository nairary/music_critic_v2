"""Checkpoint-attested AnalysisGNN encoder reduced to the two common heads."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from torch import nn

from music_critic.experiments.analysisgnn.contracts import (
    BASE_FEATURE_NAMES,
    COMMON_BENCHMARK_CONFIG,
    EDGE_TYPES,
    NODE_TYPES,
    Phase9EB1Config,
    canonical_json,
    fingerprint,
)

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


class CrossTaskTransformer(nn.Module):
    """The public AnalysisGNN residual cross-task logit fusion block."""

    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            channels, 4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        return self.norm(values + attended)


class AnalysisGNNCommonModel(nn.Module):
    """Native HybridGNN + BiGRU with only quality and inversion outputs."""

    TASKS = ("quality", "inversion")

    def __init__(self, config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG) -> None:
        super().__init__()
        config.__post_init__()
        try:
            from graphmuse.nn.models.metrical_gnn import HybridGNN
        except ImportError as exc:  # pragma: no cover - environment acceptance path
            raise RuntimeError(
                "pinned GraphMuse is required; run scripts/prepare_phase9eb1_environment.sh"
            ) from exc

        self.config = config
        hidden = config.hidden_channels
        output = config.output_channels
        metadata = (list(NODE_TYPES), list(EDGE_TYPES))
        self.pitch_embedding = nn.Embedding(35, 64)
        self.key_embedding = nn.Embedding(15, 64)
        self.project = nn.ModuleDict(
            {
                node_type: (
                    nn.Sequential(
                        nn.Linear(len(BASE_FEATURE_NAMES) + 128, hidden),
                        nn.ReLU(),
                        nn.LayerNorm(hidden),
                        nn.Dropout(config.dropout),
                        nn.Linear(hidden, hidden),
                    )
                    if node_type == "note"
                    else nn.Sequential(
                        nn.Linear(len(BASE_FEATURE_NAMES), hidden),
                        nn.ReLU(),
                        nn.LayerNorm(hidden),
                        nn.Dropout(config.dropout),
                        nn.Linear(hidden, hidden),
                    )
                )
                for node_type in NODE_TYPES
            }
        )
        self.encoder = HybridGNN(
            metadata=metadata,
            input_channels=hidden,
            hidden_channels=hidden,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_jk=config.use_jk,
        )
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, output),
            nn.ReLU(),
            nn.LayerNorm(output),
            nn.Dropout(config.dropout),
            nn.Linear(output, output),
        )
        classes = {"quality": config.quality_classes, "inversion": config.inversion_classes}
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(output, output // 2),
                    nn.ReLU(),
                    nn.LayerNorm(output // 2),
                    nn.Linear(output // 2, count),
                )
                for task, count in classes.items()
            }
        )
        self.logit_projections = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(count, output // 2),
                    nn.ReLU(),
                    nn.LayerNorm(output // 2),
                )
                for task, count in classes.items()
            }
        )
        self.cross_task = CrossTaskTransformer(output // 2, dropout=config.dropout)
        self.fusion = nn.ModuleDict(
            {task: nn.Linear(output // 2, count) for task, count in classes.items()}
        )

    def encode(self, graph: "HeteroData") -> torch.Tensor:
        note = graph["note"]
        enriched = torch.cat(
            (
                note.x,
                self.pitch_embedding(note.pitch_spelling),
                self.key_embedding(note.key_signature),
            ),
            dim=-1,
        )
        x_dict = {
            node_type: self.project[node_type](
                enriched if node_type == "note" else graph[node_type].x
            )
            for node_type in NODE_TYPES
        }
        batch_dict = {
            node_type: getattr(
                graph[node_type],
                "batch",
                torch.zeros(
                    x_dict[node_type].shape[0],
                    device=x_dict[node_type].device,
                    dtype=torch.long,
                ),
            )
            for node_type in NODE_TYPES
        }
        encoded = self.encoder(
            x_dict=x_dict,
            edge_index_dict=graph.edge_index_dict,
            batch_dict=batch_dict,
            batch_size=note.x.shape[0],
            neighbor_mask_node=None,
            neighbor_mask_edge=None,
            return_edge_index=False,
            edge_attr_dict=None,
        )
        onset = graph["note", "onset", "note"].edge_index
        onset = onset[:, onset[0] != onset[1]]
        pooled = encoded.clone()
        if onset.numel():
            sums = torch.zeros_like(encoded)
            counts = torch.zeros((encoded.shape[0], 1), dtype=encoded.dtype, device=encoded.device)
            sums.index_add_(0, onset[0], encoded[onset[1]])
            counts.index_add_(
                0,
                onset[0],
                torch.ones((onset.shape[1], 1), dtype=encoded.dtype, device=encoded.device),
            )
            present = counts.squeeze(-1).gt(0)
            pooled[present] = sums[present] / counts[present]
        return self.embedding_projection(torch.cat((encoded, pooled), dim=-1))

    def forward(self, graph: "HeteroData") -> dict[str, torch.Tensor]:
        embedding = self.encode(graph)
        raw = {task: self.heads[task](embedding) for task in self.TASKS}
        projected = torch.stack(
            tuple(self.logit_projections[task](raw[task]) for task in self.TASKS), dim=1
        )
        enhanced = self.cross_task(projected)
        return {
            task: self.fusion[task](enhanced[:, index])
            for index, task in enumerate(self.TASKS)
        }

    def architecture_manifest(self) -> dict[str, object]:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        state_shapes = {
            name: list(value.shape) for name, value in sorted(self.state_dict().items())
        }
        payload: dict[str, object] = {
            "config": asdict(self.config),
            "forbidden_heads": [
                "root",
                "bass",
                "local_key",
                "pitch_class_set",
                "auxiliary",
            ],
            "heads": list(self.TASKS),
            "parameter_count": parameter_count,
            "state_shapes": state_shapes,
            "trainable_parameter_count": trainable_count,
        }
        payload["fingerprint"] = fingerprint(payload)
        # Round trip here catches accidental NaN or non-JSON state metadata.
        canonical_json(payload)
        return payload


__all__ = ["AnalysisGNNCommonModel", "CrossTaskTransformer"]
