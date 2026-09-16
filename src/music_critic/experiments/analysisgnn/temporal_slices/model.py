"""C2 followed by two bidirectional temporal graph layers and the same heads."""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from .graph import EDGES, FEATURES, NODES, SCHEMA, TemporalGraphBatch

MODEL_SCHEMA = "C2TemporalSliceModel@1.0.0"


class TemporalRelationLayer(nn.Module):
    """Mean within each relation, sum relations, residual with per-type norm.

    All destinations are updated simultaneously. Two rounds allow
    note -> slice -> note and parent -> slice -> parent communication.
    """
    def __init__(self, width=128, destinations=NODES):
        super().__init__()
        self.destinations = tuple(destinations)
        self.edge_types = tuple(e for e in EDGES if e[2] in self.destinations)
        self.messages = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in self.edge_types)
        self.norms = nn.ModuleDict({n: nn.LayerNorm(width) for n in self.destinations})
        self.dropout = nn.Dropout(.1)

    def forward(self, embeddings, edges):
        messages = {n: torch.zeros_like(embeddings[n]) for n in self.destinations}
        for projection, edge_type in zip(self.messages, self.edge_types):
            source, _, target = edge_type
            a, b = edges[edge_type]
            # Project once per source entity, not once per sustained-note edge.
            values = projection(embeddings[source]).index_select(0, a)
            aggregate = torch.zeros_like(embeddings[target])
            aggregate.index_add_(0, b, values)
            degree = torch.zeros((aggregate.shape[0], 1), device=aggregate.device, dtype=aggregate.dtype)
            degree.index_add_(0, b, torch.ones((b.numel(), 1), device=b.device, dtype=aggregate.dtype))
            messages[target] = messages[target] + aggregate / degree.clamp_min(1)
        result = dict(embeddings)
        result.update({n: self.norms[n](embeddings[n] + self.dropout(torch.nn.functional.gelu(messages[n])))
                       for n in self.destinations})
        return result


@dataclass(frozen=True)
class TemporalOutput:
    logits: Mapping[str, Tensor]
    candidate_keys: Mapping[str, tuple[tuple[int, str, str], ...]]
    embeddings: Mapping[str, Tensor]
    graph: TemporalGraphBatch
    schema: str = MODEL_SCHEMA


class TemporalSliceModel(nn.Module):
    """Separate experiment class. Existing campaign dispatch cannot select it."""
    def __init__(self):
        super().__init__()
        # C2 is constructed first: same seed preserves its initial weights.
        self.base = CorrectedAnalysisGNNModel()
        self.slice_projection = nn.Linear(len(FEATURES), 128)
        # No trainable last-round updates for nodes without downstream use.
        self.temporal_layers = nn.ModuleList((TemporalRelationLayer(),
            TemporalRelationLayer(destinations=("note", "beat", "slice"))))

    def forward(self, graph: TemporalGraphBatch) -> TemporalOutput:
        if graph.schema != SCHEMA:
            raise ValueError("temporal graph schema mismatch")
        encoded = self.base.encode(graph.base).fused.embeddings
        x = graph.slice_features
        embeddings = {n: encoded[n] for n in NODES if n != "slice"}
        embeddings["slice"] = self.slice_projection(x / (1 + x.abs()))
        for layer in self.temporal_layers:
            embeddings = layer(embeddings, graph.edges)
        logits, keys = {}, {}
        for i, spec in enumerate(self.base.task_specs):
            level = TASK_BY_ID[spec.task_id].prediction_level
            routes = ("slice", "beat") if level == "harmonic_event" else (("slice",) if level == "onset" else ("note",))
            values = torch.cat([embeddings[n] for n in routes], dim=0)
            with torch.amp.autocast(values.device.type, enabled=False):
                logits[spec.task_id] = self.base.task_heads.heads[f"task_{i:02d}"](values.float())
            keys[spec.task_id] = tuple((sample, n, entity) for n in routes for sample, entity in graph.keys[n])
        return TemporalOutput(logits, keys, embeddings, graph)
