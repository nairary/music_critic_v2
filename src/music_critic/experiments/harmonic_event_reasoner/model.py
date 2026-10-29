"""Small query readouts over frozen C3 features, with no entity-ID embeddings.

The graph variant is an NBF-inspired query-conditioned message-passing control,
not a reproduction of NBFNet. Oracle H geometry and target-sidecar isolation
belong to the caller. This module receives only numeric observed features,
node/edge types and the explicit source H of a membership query.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_membership_loss(
    logits: Tensor, labels: Tensor, mask: Tensor, *, class_weights=None,
) -> Tensor:
    """Mean BCE over known candidates in one H; the caller averages H/work.

    Optional (negative, positive) weights multiply known BCE terms; the
    denominator remains the known count, never the sum of weights.
    Unknown labels may contain NaN and never enter BCE. An all-unknown query
    returns a differentiable zero, but the trainer must still count/skip it
    explicitly rather than reporting an applied supervised update.
    """
    if class_weights is not None and (
        not isinstance(class_weights, (tuple, list)) or len(class_weights) != 2
        or any(isinstance(w, bool) or not isinstance(w, Real)
               or not math.isfinite(w) or w <= 0 for w in class_weights)
    ):
        raise ValueError('class_weights must be two finite positive numbers (negative, positive)')
    if labels.shape != logits.shape or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("aligned logits, labels and boolean availability are required")
    if labels.device != logits.device or mask.device != logits.device:
        raise ValueError("logits, labels and availability must share a device")
    known_labels = labels[mask]
    if known_labels.numel() == 0:
        return logits[mask].sum()
    if not torch.isfinite(known_labels).all() or not ((known_labels == 0) | (known_labels == 1)).all():
        raise ValueError("known membership labels must be finite binary values")
    if class_weights is None:
        return F.binary_cross_entropy_with_logits(logits[mask], known_labels.to(logits.dtype))
    weights = logits.new_tensor(class_weights)[known_labels.long()]
    return F.binary_cross_entropy_with_logits(logits[mask], known_labels.to(logits.dtype), weight=weights)


class _TypedProjection(nn.Module):
    """Same type-specific feature projection in every experimental condition."""

    def __init__(self, input_width: int, hidden: int, node_types: int):
        super().__init__()
        self.by_type = nn.ModuleList(nn.Linear(input_width, hidden) for _ in range(node_types))
        self.norm = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, x: Tensor, node_type: Tensor) -> Tensor:
        values = x.new_zeros((x.shape[0], self.hidden))
        for type_id, projection in enumerate(self.by_type):
            selected = torch.nonzero(node_type == type_id, as_tuple=False).flatten()
            if selected.numel():
                values = values.index_copy(0, selected, projection(x[selected]))
        return self.norm(values)


class _RefinementRound(nn.Module):
    """Self messages in both controls, plus typed incoming messages in G.

    The final embedding row is the self-message type. Edge types select learned
    diagonal gates and shifts rather than materializing an E x d x d tensor.
    Q and G have exactly the same parameter schema and update operators.
    """

    def __init__(self, hidden: int, edge_types: int):
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.edge_gate = nn.Embedding(edge_types + 1, hidden)
        self.edge_shift = nn.Embedding(edge_types + 1, hidden)
        self.self_type = edge_types
        self.update = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.update_gate = nn.Linear(4 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.edge_gate.weight, std=0.05)
        nn.init.normal_(self.edge_shift.weight, std=0.05)

    def forward(
        self, state: Tensor, initial: Tensor, query: Tensor,
        edge_index: Tensor, edge_type: Tensor, propagate: bool,
    ) -> Tensor:
        # Query is globally broadcast: self-only never loses source access.
        projected = self.message(state + query)
        aggregate = (
            projected * self.edge_gate.weight[self.self_type].sigmoid()
            + self.edge_shift.weight[self.self_type]
        )
        if propagate and edge_type.numel():
            sender, receiver = edge_index.unbind(0)
            incoming = (
                projected[sender] * self.edge_gate(edge_type).sigmoid()
                + self.edge_shift(edge_type)
            )
            aggregate = aggregate.index_add(0, receiver, incoming)
            degree = state.new_ones((state.shape[0], 1))
            degree = degree.index_add(0, receiver, state.new_ones((receiver.numel(), 1)))
            aggregate = aggregate / degree
        update_input = torch.cat((state, initial, aggregate, query.expand_as(state)), dim=-1)
        delta = self.update(update_input) * self.update_gate(update_input).sigmoid()
        return self.norm(state + delta)


class EventReadout(nn.Module):
    """Score candidate note-context nodes for one source-H membership query.

    ``x`` is N x input_width, ``node_type`` is N, ``edge_index`` is 2 x E
    directed sender-to-receiver edges, and ``edge_type`` is E. ``candidates``
    contains K graph indices in the requested answer order; returns K logits.
    Graph indices are routing only. The caller owns seeds, dtype and device.

    Four kinds: static ``mlp`` and ``transe``, or query-conditioned ``self``
    and ``graph``. There is one learned relation (H-to-chord-member) in this
    pilot. Node/edge types are schema categories, never semantic gold labels.
    All kinds construct identical type projections before their own decoder.
    ``self`` and ``graph`` have identical state_dict shapes, allowing exact
    weight-matched propagation ablations including the empty-edge control.
    """

    def __init__(
        self, input_width: int = 266, kind: str = "mlp", hidden: int = 64,
        rounds: int = 3, node_types: int = 9, edge_types: int = 64,
    ):
        super().__init__()
        if kind not in {"mlp", "transe", "self", "graph"}:
            raise ValueError("kind must be mlp, transe, self or graph")
        if min(input_width, hidden, node_types, edge_types) < 1 or rounds < 1:
            raise ValueError("dimensions, type counts and rounds must be positive")
        self.kind = kind
        self.input_width = input_width
        self.node_types = node_types
        self.edge_types = edge_types
        self.projection = _TypedProjection(input_width, hidden, node_types)
        self.relation = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.relation, std=0.05)
        if kind == "transe":
            self.bias = nn.Parameter(torch.tensor(1.0))
            self.raw_temperature = nn.Parameter(torch.tensor(0.0))
        else:
            self.pair = nn.Sequential(
                nn.Linear(4 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, 1)
            )
        if kind in {"self", "graph"}:
            self.query = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU())
            self.source_marker = nn.Parameter(torch.empty(hidden))
            nn.init.normal_(self.source_marker, std=0.05)
            self.refinement = nn.ModuleList(
                _RefinementRound(hidden, edge_types) for _ in range(rounds)
            )

    def _validate_inputs(
        self, x: Tensor, node_type: Tensor, edge_index: Tensor,
        edge_type: Tensor, source: int, candidates: Tensor,
    ) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_width or not x.is_floating_point():
            raise ValueError("x must be a floating N x input_width matrix")
        n = x.shape[0]
        if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < n:
            raise ValueError("source must be a valid integer graph index")
        if node_type.shape != (n,) or node_type.dtype != torch.long:
            raise ValueError("node_type must be a long vector of length N")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a long 2 x E matrix")
        if edge_type.shape != (edge_index.shape[1],) or edge_type.dtype != torch.long:
            raise ValueError("edge_type must be a long vector of length E")
        if candidates.ndim != 1 or candidates.dtype != torch.long:
            raise ValueError("candidates must be a long vector")
        for values, upper, name in (
            (node_type, self.node_types, "node_type"),
            (edge_index, n, "edge_index"),
            (edge_type, self.edge_types, "edge_type"),
            (candidates, n, "candidates"),
        ):
            if values.device != x.device:
                raise ValueError(f"{name} must share the feature device")
            if values.numel() and ((values < 0).any() or (values >= upper).any()):
                raise ValueError(f"{name} contains an out-of-range index")

    def forward(
        self, x: Tensor, node_type: Tensor, edge_index: Tensor,
        edge_type: Tensor, source: int, candidates: Tensor,
    ) -> Tensor:
        self._validate_inputs(x, node_type, edge_index, edge_type, source, candidates)
        initial = self.projection(x, node_type)
        if self.kind == "transe":
            source_state = F.normalize(initial[source], dim=-1, eps=1e-8)
            target_state = F.normalize(initial[candidates], dim=-1, eps=1e-8)
            distance = (source_state + self.relation - target_state).square().sum(dim=-1)
            return self.bias - distance / (F.softplus(self.raw_temperature) + 1e-4)
        state = initial
        if self.kind in {"self", "graph"}:
            query = self.query(torch.cat((initial[source], self.relation), dim=-1))
            marker = torch.zeros_like(initial)
            marker[source] = self.source_marker
            state = initial + marker
            for layer in self.refinement:
                state = layer(state, initial, query, edge_index, edge_type, self.kind == "graph")
        source_state = (state[source] + self.relation).expand(candidates.numel(), -1)
        target_state = state[candidates]
        pair = torch.cat(
            (source_state, target_state, target_state - source_state, source_state * target_state),
            dim=-1,
        )
        return self.pair(pair).squeeze(-1)
