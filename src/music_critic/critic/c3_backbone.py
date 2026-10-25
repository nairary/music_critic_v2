"""Frozen inference compatibility for one explicitly admitted V2 source snapshot.

Adapted from cloud C3 retry002 relationgate source, commit 6e90163444a528154b38e1813f35e345b5668747.
Attention and gate operators preserve that snapshot's parameter names, initialization
order, arithmetic and output schema. This module does not admit the later 3090
changes variant or restore a historical optimizer/runtime. Source archive SHA256:
d21edcba657eaee451b971e8bd69e9048319699136cc36d2318b269ddc16b90d.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.temporal_slices.graph import EDGES, FEATURES, NODES, SCHEMA as GRAPH_SCHEMA
from music_critic.experiments.analysisgnn.temporal_slices.model import TemporalOutput, TemporalRelationLayer, TemporalSliceModel

C3_SOURCE_ID = 'b73f6f864202427d9458b0f01bf8dd891e82661cebe3b70ec1ebe8f67e91f5d7'
C3_SOURCE_ARCHIVE_SHA256 = 'd21edcba657eaee451b971e8bd69e9048319699136cc36d2318b269ddc16b90d'
C3_ADAPTER_VERSION = 'critic-c3-threepass-relationgate-cloud-20260918-v1'
GATED_MODEL_SCHEMA = 'C3ThreePassRelationGateModel@1.0.0'
GATED_OUTPUT_SCHEMA = 'C3ThreePassRelationGateOutput@1.0.0'
GATED_CHECKPOINT_SCHEMA = 'C3ThreePassRelationGateTrainingCheckpoint@1.0.0'
FINAL_DESTINATIONS = ('note', 'beat', 'slice')



ATTENDED_EDGES = (
    ("note", "starts", "slice"),
    ("note", "holds", "slice"),
    ("note", "ends", "slice"),
    ("slice", "rev_holds", "note"),
    ("slice", "rev_contains", "beat"),
    ("slice", "rev_contains", "bar"),
)


def relation_key(edge_type):
    return "__".join(edge_type)


def receiver_softmax(scores: Tensor, receivers: Tensor, receiver_count: int) -> Tensor:
    """Sparse stable softmax over edges per (receiver, head), never over a batch.

    Storage is E*H + N_destination*H, without an N*N matrix. The maximum is a
    numerical shift only; detaching it preserves the softmax derivative. CUDA
    determinism and memory admission must be tested in the intended runtime.
    """
    if (scores.ndim != 2 or not scores.is_floating_point() or receivers.ndim != 1
            or receivers.dtype != torch.long or receivers.device != scores.device
            or receivers.numel() != scores.shape[0] or type(receiver_count) is not int
            or receiver_count < 0):
        raise ValueError("c3.invalid_receiver_softmax_input")
    if scores.shape[0] == 0:
        return scores
    indices = receivers[:, None].expand_as(scores)
    maxima = scores.new_full((receiver_count, scores.shape[1]), -torch.inf)
    maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
    numerators = (scores - maxima.index_select(0, receivers)).exp()
    denominators = scores.new_zeros(maxima.shape)
    denominators.index_add_(0, receivers, numerators)
    # Every indexed receiver has at least one edge, hence a positive denominator.
    return numerators / denominators.index_select(0, receivers)


class C3RelationAttentionLayer(nn.Module):
    """Wrap a fully initialized mean layer, retaining its V maps and residual.

    Four head-specific attention distributions weight disjoint 32-channel
    slices of each old 128-channel value projection. No extra output projection,
    attention dropout, relation gate, window, or edge is introduced.
    """
    def __init__(self, mean_layer: TemporalRelationLayer, *, attention_heads=4):
        super().__init__()
        if type(mean_layer) is not TemporalRelationLayer:
            raise ValueError("c3.existing_temporal_relation_layer_required")
        width = mean_layer.messages[0].in_features
        if (type(attention_heads) is not int or attention_heads <= 0
                or width % attention_heads != 0):
            raise ValueError("c3.attention_heads_must_divide_width")
        self.width, self.attention_heads = width, attention_heads
        self.head_width = width // attention_heads
        self.destinations, self.edge_types = mean_layer.destinations, mean_layer.edge_types
        self.messages, self.norms, self.dropout = mean_layer.messages, mean_layer.norms, mean_layer.dropout
        self.attended_edges = tuple(e for e in self.edge_types if e in ATTENDED_EDGES)
        self.queries = nn.ModuleDict({relation_key(e): nn.Linear(width, width, bias=False)
                                     for e in self.attended_edges})
        self.keys = nn.ModuleDict({relation_key(e): nn.Linear(width, width, bias=False)
                                  for e in self.attended_edges})
        # Only Q is zero. Nonzero K lets Q learn on the first update; K receives
        # gradients once Q has moved. Initial weights reproduce uniform means
        # up to ordinary floating-point reduction order differences.
        for query in self.queries.values():
            nn.init.zeros_(query.weight)

    def relation_weights(self, edge_type, embeddings, edge_index):
        """Return differentiable E x H weights; never retain them on the module.

        These weights describe computation, not causal musical explanations or
        calibrated confidence. Callers may detach them for bounded diagnostics.
        """
        if edge_type not in self.edge_types:
            raise ValueError("c3.unknown_layer_relation")
        source, _, target = edge_type
        a, b = edge_index
        if edge_type not in self.attended_edges:
            # The raw builder guarantees singleton receivers for these relations.
            return embeddings[target].new_ones((a.numel(), self.attention_heads), dtype=torch.float32)
        key = relation_key(edge_type)
        with torch.amp.autocast(embeddings[target].device.type, enabled=False):
            # Project each node once before gathering its incident edges.
            q = self.queries[key](embeddings[target].float()).view(-1, self.attention_heads, self.head_width)
            k = self.keys[key](embeddings[source].float()).view(-1, self.attention_heads, self.head_width)
            scores = (q.index_select(0, b) * k.index_select(0, a)).sum(-1) / sqrt(self.head_width)
            return receiver_softmax(scores, b, embeddings[target].shape[0])

    def forward(self, embeddings, edges):
        messages = {n: torch.zeros_like(embeddings[n], dtype=torch.float32) for n in self.destinations}
        with torch.amp.autocast(embeddings["slice"].device.type, enabled=False):
            for projection, edge_type in zip(self.messages, self.edge_types):
                source, _, target = edge_type
                a, b = edges[edge_type]
                values = projection(embeddings[source].float()).index_select(0, a)
                aggregate = torch.zeros_like(embeddings[target], dtype=torch.float32)
                if edge_type in self.attended_edges:
                    weights = self.relation_weights(edge_type, embeddings, edges[edge_type])
                    weighted = (values.view(-1, self.attention_heads, self.head_width)
                                * weights.unsqueeze(-1)).reshape(-1, self.width)
                    aggregate.index_add_(0, b, weighted)
                else:
                    # Preserve the historical operator on structural singleton
                    # relations, including its empty-relation behavior.
                    aggregate.index_add_(0, b, values)
                    degree = aggregate.new_zeros((aggregate.shape[0], 1))
                    degree.index_add_(0, b, aggregate.new_ones((b.numel(), 1)))
                    aggregate = aggregate / degree.clamp_min(1)
                messages[target] = messages[target] + aggregate
            result = dict(embeddings)
            result.update({n: self.norms[n](embeddings[n].float()
                + self.dropout(torch.nn.functional.gelu(messages[n]))) for n in self.destinations})
        return result


def masked_relation_weights(scores: Tensor, active: Tensor) -> Tensor:
    """Receiver-local softmax over present relations, scaled by active count.

    Empty receiver rows return exact zeros without taking softmax of all -inf.
    A zero score for each active relation reproduces the historical SUM, rather
    than dividing messages by a node-dependent relation count.
    """
    if (scores.ndim != 2 or not scores.is_floating_point() or active.dtype != torch.bool
            or active.shape != scores.shape or active.device != scores.device
            or scores.shape[1] == 0):
        raise ValueError("c3_threepass.invalid_relation_scores")
    count = active.sum(-1, keepdim=True)
    safe_scores = torch.where(count > 0, scores.masked_fill(~active, -torch.inf),
                              torch.zeros_like(scores))
    return safe_scores.softmax(-1) * active.to(scores.dtype) * count.to(scores.dtype)


class C3RelationGateLayer(C3RelationAttentionLayer):
    """Attention within relations, followed by dynamic receiver-local selection.

    A separate zero-initialized linear scorer reads [receiver, aggregate] for
    each relation. It sees only raw latent vectors, never task labels or logits.
    """
    def __init__(self, attention_layer: C3RelationAttentionLayer):
        if type(attention_layer) is not C3RelationAttentionLayer:
            raise ValueError("c3_threepass.attention_layer_required")
        nn.Module.__init__(self)
        for name in ("width", "attention_heads", "head_width", "destinations", "edge_types", "attended_edges"):
            setattr(self, name, getattr(attention_layer, name))
        for name in ("messages", "norms", "dropout", "queries", "keys"):
            setattr(self, name, getattr(attention_layer, name))
        # Singleton relation types cannot compete: their softmax weight is
        # identically one. Avoid structurally untrainable scorer parameters.
        self.gated_edges = tuple(e for e in self.edge_types
                                 if sum(other[2] == e[2] for other in self.edge_types) > 1)
        self.relation_scorers = nn.ModuleDict({
            relation_key(edge): nn.Linear(2 * self.width, 1) for edge in self.gated_edges
        })
        for scorer in self.relation_scorers.values():
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)

    def _relation_aggregates(self, embeddings, edges):
        aggregates, active = {}, {}
        for projection, edge_type in zip(self.messages, self.edge_types):
            source, _, target = edge_type
            a, b = edges[edge_type]
            values = projection(embeddings[source].float()).index_select(0, a)
            aggregate = torch.zeros_like(embeddings[target], dtype=torch.float32)
            if edge_type in self.attended_edges:
                weights = self.relation_weights(edge_type, embeddings, edges[edge_type])
                weighted = (values.view(-1, self.attention_heads, self.head_width)
                            * weights.unsqueeze(-1)).reshape(-1, self.width)
                aggregate.index_add_(0, b, weighted)
            else:
                aggregate.index_add_(0, b, values)
                degree = aggregate.new_zeros((aggregate.shape[0], 1))
                degree.index_add_(0, b, aggregate.new_ones((b.numel(), 1)))
                aggregate = aggregate / degree.clamp_min(1)
            present = torch.zeros(aggregate.shape[0], dtype=torch.bool, device=aggregate.device)
            present.index_fill_(0, b, True)
            aggregates[edge_type], active[edge_type] = aggregate, present
        return aggregates, active

    def _relation_type_weights(self, embeddings, aggregates, active):
        result = {}
        for target in self.destinations:
            relations = tuple(e for e in self.edge_types if e[2] == target)
            if len(relations) == 1:
                result[relations[0]] = active[relations[0]][:, None].float()
                continue
            scores = torch.cat([self.relation_scorers[relation_key(e)](
                torch.cat((embeddings[target].float(), aggregates[e]), dim=-1)) for e in relations], dim=-1)
            mask = torch.stack([active[e] for e in relations], dim=-1)
            weights = masked_relation_weights(scores, mask)
            result.update({e: weights[:, i:i + 1] for i, e in enumerate(relations)})
        return result

    def relation_type_weights(self, embeddings, edges):
        """Differentiable per-receiver weights for bounded diagnostics only."""
        with torch.amp.autocast(embeddings["slice"].device.type, enabled=False):
            aggregates, active = self._relation_aggregates(embeddings, edges)
            return self._relation_type_weights(embeddings, aggregates, active)

    def forward(self, embeddings, edges):
        with torch.amp.autocast(embeddings["slice"].device.type, enabled=False):
            aggregates, active = self._relation_aggregates(embeddings, edges)
            weights = self._relation_type_weights(embeddings, aggregates, active)
            messages = {n: torch.zeros_like(embeddings[n], dtype=torch.float32) for n in self.destinations}
            # Preserve the old relation addition order at uniform initialization.
            for edge_type in self.edge_types:
                target = edge_type[2]
                messages[target] = messages[target] + weights[edge_type] * aggregates[edge_type]
            result = dict(embeddings)
            result.update({n: self.norms[n](embeddings[n].float()
                + self.dropout(torch.nn.functional.gelu(messages[n]))) for n in self.destinations})
        return result


@dataclass(frozen=True)
class C3ThreePassRelationGateOutput(TemporalOutput):
    schema: str = GATED_OUTPUT_SCHEMA


class C3ThreePassRelationGateModel(TemporalSliceModel):
    """Exactly the selected cloud x3 gated architecture, with rich same-pass export."""
    output_type = C3ThreePassRelationGateOutput

    def __init__(self, *, attention_heads=4):
        if type(attention_heads) is not int or attention_heads != 4:
            raise ValueError('selected cloud C3 source requires four attention heads')
        super().__init__()
        first, last = self.temporal_layers
        middle = TemporalRelationLayer(destinations=NODES)
        self.attention_heads = attention_heads
        attention = [C3RelationAttentionLayer(layer, attention_heads=attention_heads)
                     for layer in (first, middle, last)]
        self.temporal_layers = nn.ModuleList(C3RelationGateLayer(layer) for layer in attention)

    def forward_with_base(self, graph):
        graph.validate()
        output, base = super().forward_with_base(graph)
        return self.output_type(output.logits, output.candidate_keys, output.embeddings, output.graph), base



def c3_relationgate_model_contract(model):
    from music_critic.experiments.analysisgnn.corrected_model import corrected_model_contract
    from music_critic.experiments.analysisgnn.temporal_slices.alignment import ALIGNMENT_SCHEMA

    if type(model) is not C3ThreePassRelationGateModel:
        raise ValueError("c3_threepass.model_required")
    gated = True
    expected_type = C3RelationGateLayer if gated else C3RelationAttentionLayer
    if (type(model.attention_heads) is not int or model.attention_heads <= 0
            or 128 % model.attention_heads or len(model.temporal_layers) != 3
            or model.output_type is not C3ThreePassRelationGateOutput):
        raise ValueError("c3_threepass.inconsistent_architecture")
    for i, layer in enumerate(model.temporal_layers):
        destinations = NODES if i < 2 else FINAL_DESTINATIONS
        edge_types = tuple(e for e in EDGES if e[2] in destinations)
        if (type(layer) is not expected_type or type(layer.width) is not int or layer.width != 128
                or type(layer.attention_heads) is not int or layer.attention_heads != model.attention_heads
                or type(layer.head_width) is not int or layer.head_width * layer.attention_heads != layer.width
                or layer.destinations != destinations or layer.edge_types != edge_types
                or layer.attended_edges != tuple(e for e in edge_types if e in ATTENDED_EDGES)
                or set(layer.queries) != {relation_key(e) for e in layer.attended_edges}
                or set(layer.keys) != {relation_key(e) for e in layer.attended_edges}):
            raise ValueError("c3_threepass.inconsistent_architecture")
        gated_edges = tuple(e for e in edge_types if sum(other[2] == e[2] for other in edge_types) > 1)
        if gated and (layer.gated_edges != gated_edges
                      or set(layer.relation_scorers) != {relation_key(e) for e in gated_edges}
                      or any(type(s) is not nn.Linear or s.in_features != 256 or s.out_features != 1
                             or s.bias is None for s in layer.relation_scorers.values())):
            raise ValueError("c3_threepass.inconsistent_relation_gate")
    contract = dict(
        schema_version=GATED_MODEL_SCHEMA,
        output_schema=GATED_OUTPUT_SCHEMA,
        graph_schema=GRAPH_SCHEMA, target_alignment_schema=ALIGNMENT_SCHEMA,
        base=corrected_model_contract(model.base), features=list(FEATURES), edges=[list(e) for e in EDGES],
        width=128, rounds=3, update_semantics="synchronous_previous_round_only",
        attention_heads=model.attention_heads, attention="receiver_scaled_dot_product",
        normalization="incoming_neighbors_per_relation_per_head",
        initialization="historical_values_then_extra_middle_values_then_all_zero_Q_random_K_then_zero_gate_scorers",
        relation_fusion="active_count_scaled_masked_softmax" if gated else "sum", attention_dropout=0.,
        relation_gate=(dict(score="relation_specific_linear_concat_receiver_aggregate",
                           mask="receiver_has_incoming_edge_of_relation", scale="active_relation_count",
                           empty_receiver="all_zero_weights", initialization="zero_weight_and_bias",
                           singleton_relation="presence_weight_no_scorer") if gated else None),
        layers=[dict(width=layer.width, attention_heads=layer.attention_heads,
                     head_width=layer.head_width, destinations=list(layer.destinations),
                     edge_types=[list(e) for e in layer.edge_types],
                     attended_edges=[list(e) for e in layer.attended_edges]) for layer in model.temporal_layers],
        task_routes="same18;harmonic=slice_then_beat;cadence=slice;note=note",
        parameter_layout=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad)
                          for n, p in model.named_parameters()],
        dropout_probabilities={n: m.p for n, m in model.named_modules() if isinstance(m, nn.Dropout)},
        normalization_epsilons={n: m.eps for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)})
    contract["fingerprint"] = fingerprint(contract)
    return contract


def validate_c3_source_model(model):
    """Keep non-tensor architecture settings bound as well as frozen weights."""
    contract = c3_relationgate_model_contract(model)
    contract.pop('fingerprint')
    for parameter in contract['parameter_layout']:
        parameter['requires_grad'] = True  # Freezing is permitted; architecture changes are not.
    if fingerprint(contract) != '5f391869cbd29161ad92c5b286e2f1a2b87b1aeaec7efbbf005eef620e7aa1a6':
        raise ValueError('selected cloud C3 architecture fingerprint mismatch')


def load_c3_relationgate_adapter(checkpoint_path, *, expected_sha256: str,
                                expected_size_bytes: int, source_id: str = C3_SOURCE_ID,
                                expected_update: int = 80000):
    """Load frozen weights from a byte-bound original cloud checkpoint.

    Validates the full envelope seal and the original architecture/source/config;
    does not restore its optimizer, RNG, training counters, labels or runtime.
    Callers register real checkpoint use through the experiment protocol.
    """
    from hashlib import sha256
    from io import BytesIO
    from pathlib import Path
    import re

    from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    from .contracts import FeatureBinding
    from .features import BackboneFeatureAdapter

    if source_id != C3_SOURCE_ID:
        raise ValueError('selected C3 source manifest binding required')
    if (not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or type(expected_size_bytes) is not int or expected_size_bytes <= 0):
        raise ValueError('exact checkpoint artifact SHA256 and size required')
    path = Path(checkpoint_path)
    if path.stat().st_size != expected_size_bytes:
        raise ValueError('checkpoint artifact size mismatch')
    content = path.read_bytes()
    if len(content) != expected_size_bytes or sha256(content).hexdigest() != expected_sha256:
        raise ValueError('checkpoint artifact SHA256 mismatch')
    with torch.inference_mode(False):
        payload = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('schema_version') != GATED_CHECKPOINT_SCHEMA:
            raise ValueError('selected C3 checkpoint schema required')
        if payload.get('bindings', {}).get('source') != C3_SOURCE_ID:
            raise ValueError('checkpoint C3 source manifest mismatch')
        if (type(expected_update) is not int or expected_update < 0
                or payload.get('progress', {}).get('applied_updates') != expected_update):
            raise ValueError('checkpoint applied update binding mismatch')
        config = payload.get('config', {})
        required = {'model_kind': 'c3_temporal_attention_3pass_relation_gate', 'temporal_width': 128,
                    'temporal_passes': 3, 'relation_gate': True, 'attention_heads': 4}
        if any(config.get(k) != value or type(config.get(k)) is not type(value) for k, value in required.items()):
            raise ValueError('checkpoint is not the selected C3 x3 relation-gate configuration')
        if payload.get('state_fingerprint') != recursive_state_fingerprint(
                {k: v for k, v in payload.items() if k != 'state_fingerprint'}):
            raise ValueError('checkpoint envelope integrity mismatch')
        state = payload.get('model_state')
        if not isinstance(state, dict) or any(not isinstance(v, Tensor) or not torch.isfinite(v).all()
                                              for v in state.values()):
            raise ValueError('checkpoint finite model state required')
        if payload.get('model_state_fingerprint') != model_state_fingerprint(state):
            raise ValueError('checkpoint model state integrity mismatch')
        with torch.random.fork_rng(devices=[]):
            model = C3ThreePassRelationGateModel()
        if payload.get('model_contract') != c3_relationgate_model_contract(model):
            raise ValueError('checkpoint C3 architecture contract mismatch')
        expected = model.state_dict()
        if (state.keys() != expected.keys() or any(
                state[k].shape != value.shape or state[k].dtype != value.dtype for k, value in expected.items())):
            raise ValueError('checkpoint model parameter layout mismatch')
        model.load_state_dict(state, strict=True)
        return BackboneFeatureAdapter(model, FeatureBinding(
            source_id, 'sha256:' + expected_sha256, C3_ADAPTER_VERSION,
            model_state_id=payload['model_state_fingerprint']))
