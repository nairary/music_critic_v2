"""Target-independent H constructors over a shared raw graph and H refiner.

The five mechanisms share raw message passing, full-work context, H refinement
and heads. They are constructor packages, not compute-matched architectures.
Exact canonical times and identities belong to the raw adapter/decoder; this
module operates on normalized floating model coordinates only. Assignment and
attention weights are latent weights, never semantic membership probabilities.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class _TypedProjection(nn.Module):
    def __init__(self, width: int, hidden: int, node_types: int):
        super().__init__()
        self.by_type = nn.ModuleList(nn.Linear(width, hidden) for _ in range(node_types))
        self.geometry = nn.Linear(4, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, x: Tensor, types: Tensor, times: Tensor) -> Tensor:
        state = x.new_zeros((x.shape[0], self.hidden))
        for kind, projection in enumerate(self.by_type):
            indices = torch.nonzero(types == kind, as_tuple=False).flatten()
            if indices.numel():
                state = state.index_copy(0, indices, projection(x[indices]))
        return self.norm(state + self.geometry(_geometry(times)))


def _geometry(spans: Tensor) -> Tensor:
    return torch.cat((spans, spans.mean(-1, keepdim=True),
                      spans[:, 1:2] - spans[:, 0:1]), dim=-1)


def _weighted_pool(weights: Tensor, values: Tensor) -> Tensor:
    """K by N weights; normalize over inputs, never over recipient groups."""
    return weights @ values / weights.sum(-1, keepdim=True).clamp_min(1e-8)


def _span_weights(spans: Tensor, times: Tensor) -> Tensor:
    # Includes held intervals and point/grace states. A smooth geometric path
    # exists even when the learned semantic constructor gives a note low weight.
    return (torch.sigmoid(12 * (times[None, :, 1] - spans[:, None, 0]))
            * torch.sigmoid(12 * (spans[:, None, 1] - times[None, :, 0])))


class _RawRound(nn.Module):
    """Directed schema-typed messages plus independent all-node raw context."""
    def __init__(self, hidden: int, edge_types: int):
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.edge_gate = nn.Embedding(edge_types, hidden)
        self.edge_shift = nn.Embedding(edge_types, hidden)
        self.edge_time = nn.Linear(2, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden))
        self.gate = nn.Linear(hidden * 4, hidden)
        self.norm = nn.LayerNorm(hidden)
        nn.init.normal_(self.edge_gate.weight, std=.05)
        nn.init.normal_(self.edge_shift.weight, std=.05)

    def forward(self, state: Tensor, initial: Tensor, times: Tensor,
                edges: Tensor, types: Tensor) -> Tensor:
        projected = self.message(state)
        aggregate = projected
        degree = state.new_ones((state.shape[0], 1))
        if types.numel():
            src, dst = edges.unbind(0)
            delta_time = times[src] - times[dst]
            message = (projected[src] * self.edge_gate(types).sigmoid()
                       + self.edge_shift(types) + self.edge_time(delta_time))
            aggregate = aggregate.index_add(0, dst, message)
            degree = degree.index_add(0, dst, state.new_ones((dst.numel(), 1)))
        context = state.mean(0, keepdim=True).expand_as(state)
        joined = torch.cat((state, initial, aggregate / degree, context), dim=-1)
        return self.norm(state + self.update(joined) * self.gate(joined).sigmoid())


class _HRefiner(nn.Module):
    """The identical geometry-aware raw/H/context access in every arm."""
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden)
        self.geometry = nn.Linear(4, hidden)
        self.update = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden))
        self.gate = nn.Linear(hidden * 5, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.scale = hidden ** -.5

    def forward(self, state: Tensor, initial: Tensor, spans: Tensor,
                raw_state: Tensor, raw_initial: Tensor, times: Tensor) -> Tensor:
        if state.shape[0] == 0:
            return state
        geometric = _span_weights(spans, times)
        raw_context = _weighted_pool(geometric, raw_state + raw_initial)
        # H-to-H access is shared; E5 does not get induced coarse adjacency.
        scores = self.query(state) @ self.key(state).T * self.scale
        centers = spans.mean(-1)
        scores = scores - (centers[:, None] - centers[None, :]).abs()
        neighbor = scores.softmax(-1) @ self.value(state)
        full_context = raw_state.mean(0, keepdim=True).expand_as(state)
        joined = torch.cat((state, initial + self.geometry(_geometry(spans)),
                            raw_context, neighbor, full_context), dim=-1)
        return self.norm(state + self.update(joined) * self.gate(joined).sigmoid())


class HAnalyzer(nn.Module):
    """Construct and refine harmonic-event candidates without any target input.

    ``raw`` has the exact tensor keys in the launch-preparation spec. Unknown
    keys are rejected so a caller cannot accidentally feed a target sidecar.
    Node indices are routing only; learned embeddings encode schema types or
    E4 query slots, never corpus, note, work, view or target identities.

    E1 emits all supplied bank rows; E3 emits all raw time anchors. E2 emits a
    contiguous partition with at most ``max_events`` intervals. E4/E5 use that
    fixed slot capacity, independently of annotation count. Decoding/ranking
    under a raw-duration budget and target matching live outside this module.
    """
    RAW_KEYS = frozenset({"x", "node_type", "edge_index", "edge_type", "node_time",
                          "time_index", "time_positions", "bank_indices", "pool_mask"})

    def __init__(self, kind: str, head_sizes: Mapping[str, int], input_width: int = 266,
                 hidden: int = 128, rounds: int = 3, max_events: int = 64,
                 edge_types: int = 64, node_types: int = 9):
        super().__init__()
        if kind not in {"E1", "E2", "E3", "E4", "E5"}:
            raise ValueError("kind must be E1, E2, E3, E4 or E5")
        dimensions = (input_width, hidden, rounds, max_events, edge_types, node_types)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in dimensions):
            raise ValueError("dimensions, rounds and capacities must be positive integers")
        if not isinstance(head_sizes, Mapping) or any(
            not isinstance(name, str) or not name or isinstance(size, bool)
            or not isinstance(size, int) or size < 1 for name, size in head_sizes.items()
        ):
            raise ValueError("head_sizes must map nonempty field names to positive class counts")
        self.kind, self.input_width, self.hidden = kind, input_width, hidden
        self.node_types, self.edge_types, self.max_events = node_types, edge_types, max_events
        self.projection = _TypedProjection(input_width, hidden, node_types)
        self.raw_rounds = nn.ModuleList(_RawRound(hidden, edge_types) for _ in range(rounds))
        self.event_geometry = nn.Linear(4, hidden)
        self.refiners = nn.ModuleList(_HRefiner(hidden) for _ in range(rounds))
        self.object_head = nn.Linear(hidden, 1)
        # ModuleDict rejects dots, but dotted ontology field names are canonical.
        self.field_names = tuple(sorted(head_sizes))
        self.field_heads = nn.ModuleList(nn.Linear(hidden, head_sizes[name])
                                         for name in self.field_names)
        if kind == "E2":
            self.boundary_head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(),
                                               nn.Linear(hidden, 1))
            self.boundary_context = nn.Linear(hidden, hidden, bias=False)
        elif kind == "E3":
            self.endpoint_head = nn.Linear(hidden, 2)
        elif kind == "E4":
            self.event_queries = nn.Parameter(torch.empty(max_events, hidden))
            nn.init.normal_(self.event_queries, std=.02)
            self.query_key = nn.Linear(hidden, hidden, bias=False)
            self.query_value = nn.Linear(hidden, hidden)
            self.query_norm = nn.LayerNorm(hidden)
            self.endpoint_head = nn.Linear(hidden, 2)
        elif kind == "E5":
            self.assignment_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                                 nn.Linear(hidden, max_events))
            self.endpoint_head = nn.Linear(hidden, 2)

    def _validate(self, raw: Mapping[str, Tensor]) -> None:
        if not isinstance(raw, Mapping) or set(raw) != self.RAW_KEYS:
            raise ValueError("raw must contain exactly the target-free HAnalyzer tensor keys")
        if any(not isinstance(t, Tensor) for t in raw.values()):
            raise ValueError("every raw model input must be a tensor")
        x = raw["x"]
        if x.ndim != 2 or x.shape[1] != self.input_width or not x.is_floating_point():
            raise ValueError("x must be a floating N by input_width matrix")
        n = x.shape[0]
        if n == 0:
            raise ValueError("raw must retain time/context nodes even when it has no notes")
        if any(t.device != x.device for t in raw.values()):
            raise ValueError("all raw tensors must be on the same device")
        for name in ("x", "node_time", "time_positions"):
            t = raw[name]
            if not t.is_floating_point() or t.dtype != x.dtype or not torch.isfinite(t).all():
                raise ValueError(f"{name} must be finite and share x floating dtype")
        for name in ("node_type", "edge_index", "edge_type", "time_index", "bank_indices"):
            if raw[name].dtype != torch.long:
                raise ValueError(f"{name} must have dtype torch.long")
        if raw["node_type"].shape != (n,) or raw["node_time"].shape != (n, 2):
            raise ValueError("node types and node_time must align with x")
        if (raw["node_time"][:, 0] > raw["node_time"][:, 1]).any():
            raise ValueError("node intervals must have start <= end")
        edges, types = raw["edge_index"], raw["edge_type"]
        if edges.ndim != 2 or edges.shape[0] != 2 or types.shape != (edges.shape[1],):
            raise ValueError("edge_index must be 2 by E and edge_type length E")
        positions, indices = raw["time_positions"], raw["time_index"]
        if positions.ndim != 1 or positions.numel() < 2 or indices.shape != positions.shape:
            raise ValueError("at least two aligned raw time boundaries are required")
        if not (positions[1:] > positions[:-1]).all() or positions[0] != 0 or positions[-1] != 1:
            raise ValueError("raw time positions must increase strictly from 0 to 1")
        bank = raw["bank_indices"]
        if bank.ndim != 2 or bank.shape[1] != 2 or (bank[:, 0] >= bank[:, 1]).any():
            raise ValueError("bank_indices must contain ordered endpoint pairs")
        if raw["pool_mask"].dtype != torch.bool or raw["pool_mask"].shape != (n,):
            raise ValueError("pool_mask must be a boolean vector aligned with x")
        if self.kind == "E5" and not raw["pool_mask"].any():
            raise ValueError("E5 needs at least one raw fragment or time state for pooling")
        for name, upper in (("node_type", self.node_types), ("edge_type", self.edge_types),
                            ("edge_index", n), ("time_index", n),
                            ("bank_indices", positions.numel())):
            values = raw[name]
            if values.numel() and ((values < 0).any() or (values >= upper).any()):
                raise ValueError(f"{name} contains an out-of-range routing/type index")

    @staticmethod
    def _intervals(logits: Tensor) -> Tensor:
        # Explicit positive width survives floating point at both tile edges.
        minimum = max(1e-5, torch.finfo(logits.dtype).eps * 16)
        start = (1 - minimum) * logits[:, 0].sigmoid()
        end = start + minimum + (1 - minimum - start) * logits[:, 1].sigmoid()
        return torch.stack((start, end), dim=-1)

    def _segmentation(self, raw: Mapping[str, Tensor], state: Tensor) -> tuple[Tensor, Tensor, dict]:
        positions = raw["time_positions"]
        time_state = state[raw["time_index"]]
        adjacent = torch.cat((time_state[:-2], time_state[1:-1], time_state[2:]), dim=-1)
        logits = self.boundary_head(adjacent).flatten()
        probability = logits.sigmoid()
        cuts = torch.nonzero(probability.detach() >= .5, as_tuple=False).flatten()
        if cuts.numel() > self.max_events - 1:
            # Stable raw-time tie breaking; never use target count or geometry.
            rank = torch.argsort(logits.detach(), descending=True, stable=True)
            cuts = rank[:self.max_events - 1].sort().values
        elif self.max_events == 1:
            cuts = cuts[:0]
        hard = positions[1:-1][cuts]
        if cuts.numel():
            # Smooth local alternatives provide a declared straight-through
            # gradient; forward is still an exact contiguous raw-grid partition.
            interior = positions[1:-1]
            affinity = logits[None, :] - 32 * (interior[None, :] - hard[:, None]).square()
            soft = affinity.softmax(-1) @ interior
            boundaries = hard + (soft - soft.detach())
        else:
            boundaries = positions.new_empty((0,))
        endpoints = torch.cat((positions[:1], boundaries, positions[-1:]))
        spans = torch.stack((endpoints[:-1], endpoints[1:]), dim=-1)
        h = _weighted_pool(_span_weights(spans, raw["node_time"]), state)
        if logits.numel():
            # Without a selected interior cut localization alone has no gradient
            # through a discrete count. The expected boundary profile also enters
            # constructor states, allowing H/F gradients to learn the cut scores.
            profile = _span_weights(spans, torch.stack((positions[1:-1], positions[1:-1]), -1))
            profile = profile * probability[None, :]
            expected = profile @ time_state[1:-1] / max(1, logits.numel())
            h = h + self.boundary_context(expected)
        return spans, h, {"boundary_logits": logits, "boundary_positions": positions[1:-1]}

    def _encode(self, raw: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        self._validate(raw)
        initial = self.projection(raw["x"], raw["node_type"], raw["node_time"])
        state = initial
        for layer in self.raw_rounds:
            state = layer(state, initial, raw["node_time"], raw["edge_index"], raw["edge_type"])
        return initial, state

    def _readout(self, raw: Mapping[str, Tensor], spans: Tensor, h: Tensor,
                 initial: Tensor, state: Tensor) -> dict:
        h_initial = h + self.event_geometry(_geometry(spans))
        h = h_initial
        for layer in self.refiners:
            h = layer(h, h_initial, spans, state, initial, raw["node_time"])
        return {"spans": spans, "object_logits": self.object_head(h).flatten(),
                "field_logits": {name: head(h) for name, head in zip(self.field_names, self.field_heads)},
                "states": h, "raw_states": state}

    @torch.no_grad()
    def oracle_readout(self, raw: Mapping[str, Tensor], oracle_spans: Tensor) -> dict:
        """Diagnostic-only conditional H interpretation using explicit geometry.

        This deliberately disables gradients and bypasses every H constructor.
        The caller must report this as oracle-H, separately from predicted-H
        recovery. The shared raw encoder, geometric pooling, H refiner and heads
        remain unchanged. Oracle geometry never enters ``forward`` or raw states.
        """
        initial, state = self._encode(raw)
        if (not isinstance(oracle_spans, Tensor) or oracle_spans.ndim != 2
                or oracle_spans.shape[1] != 2 or oracle_spans.dtype != state.dtype
                or oracle_spans.device != state.device or not torch.isfinite(oracle_spans).all()
                or (oracle_spans < 0).any() or (oracle_spans > 1).any()
                or (oracle_spans[:, 0] >= oracle_spans[:, 1]).any()):
            raise ValueError("oracle spans must be finite normalized start<end pairs with raw dtype/device")
        spans = oracle_spans.detach()
        h = _weighted_pool(_span_weights(spans, raw["node_time"]), state)
        return self._readout(raw, spans, h, initial, state)

    def forward(self, raw: Mapping[str, Tensor]) -> dict:
        initial, state = self._encode(raw)
        diagnostics = {}
        if self.kind == "E1":
            spans = raw["time_positions"][raw["bank_indices"]]
            h = _weighted_pool(_span_weights(spans, raw["node_time"]), state)
        elif self.kind == "E2":
            spans, h, diagnostics = self._segmentation(raw, state)
        elif self.kind == "E3":
            h = state[raw["time_index"]]
            anchors = raw["time_positions"]
            offsets = self.endpoint_head(h).sigmoid()
            minimum = max(1e-5, torch.finfo(offsets.dtype).eps * 16)
            offsets = minimum + (1 - minimum) * offsets
            spans = torch.stack((anchors * (1 - offsets[:, 0]),
                                 anchors + (1 - anchors) * offsets[:, 1]), dim=-1)
        elif self.kind == "E4":
            scores = self.event_queries @ self.query_key(state).T / math.sqrt(self.hidden)
            attention = scores.softmax(-1)
            h = self.query_norm(self.event_queries + attention @ self.query_value(state))
            spans = self._intervals(self.endpoint_head(h))
            diagnostics = {"query_attention": attention}
        else:
            selected = torch.nonzero(raw["pool_mask"], as_tuple=False).flatten()
            pooling_state = state[selected]
            assignment = self.assignment_head(pooling_state).sigmoid()
            h = _weighted_pool(assignment.T, pooling_state)
            spans = self._intervals(self.endpoint_head(h))
            diagnostics = {"assignment": assignment, "pool_indices": selected}
        return {**self._readout(raw, spans, h, initial, state), **diagnostics}
