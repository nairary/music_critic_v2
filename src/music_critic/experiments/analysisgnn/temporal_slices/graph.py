"""Raw-only exact temporal graph extension around the unchanged C2 graph.

The extension has its own schema: it is not admitted as a seventh node type
to the production six-type encoder. All topology is computed with Fractions.
Only relative scalar features are cast to float at the neural boundary.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch

from music_critic.data.raw_time_candidates import raw_time_sources
from music_critic.data.raw_time_readout_slots import build_readout_slots, ReadoutSlots
from music_critic.graph import build_raw_graph, graph_fingerprint, validate_raw_graph_batch
from music_critic.experiments.analysisgnn.corrected_training import move_raw_graph_batch

SCHEMA = "RawTemporalSliceGraph@1.0.0"
NODES = ("note", "onset", "beat", "bar", "slice")
FORWARD_EDGES = (
    ("note", "starts", "slice"), ("note", "holds", "slice"),
    ("note", "ends", "slice"), ("onset", "coincides", "slice"),
    ("beat", "contains", "slice"), ("bar", "contains", "slice"),
    ("slice", "next", "slice"),
)
EDGES = tuple(edge for s, r, d in FORWARD_EDGES
              for edge in ((s, r, d), (d, "rev_" + r, s)))
FEATURES = (
    "phase_in_beat", "position_in_metric_bar_qn", "beat_index_in_bar",
    "meter_numerator", "meter_denominator", "previous_gap_qn", "next_gap_qn",
    "has_note_start", "has_note_end", "has_beat_boundary", "has_beat_owner",
    "has_bar_owner", "is_pickup", "is_incomplete_bar",
)


def exact(value) -> Fraction:
    return value.to_fraction()


def owner(point, starts, ends):
    i = bisect_right(starts, point) - 1
    if i >= 0 and point < ends[i]:
        return i
    if ends and point == ends[-1]:
        return len(ends) - 1
    return None


@dataclass(frozen=True)
class TemporalGraphBatch:
    base: Batch
    slice_features: Tensor
    edges: Mapping[tuple[str, str, str], Tensor]
    keys: Mapping[str, tuple[tuple[int, str], ...]]
    pointers: Mapping[str, tuple[int, ...]]
    slots: tuple[ReadoutSlots, ...]
    schema: str = SCHEMA

    def to(self, device) -> "TemporalGraphBatch":
        # Production metadata contains tuples which generic PyG .to rewrites.
        return replace(self, base=move_raw_graph_batch(self.base, device),
                       slice_features=self.slice_features.to(device),
                       edges={k: v.to(device) for k, v in self.edges.items()})

    def fingerprint(self) -> str:
        bases = self.base.to_data_list()
        for graph in bases:
            # PyG unbatching retains this global bool as a one-element tensor.
            graph.raw_only = bool(graph.raw_only)
        payload = {"schema": self.schema, "features": FEATURES,
                   "base": [graph_fingerprint(g) for g in bases],
                   "keys": self.keys, "pointers": self.pointers,
                   "x": self.slice_features.detach().cpu().tolist(),
                   "edges": [(e, self.edges[e].detach().cpu().tolist()) for e in EDGES]}
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate(self) -> None:
        if self.schema != SCHEMA or tuple(self.edges) != EDGES or tuple(self.keys) != NODES:
            raise ValueError("temporal graph schema mismatch")
        size = len(self.slots)
        validate_raw_graph_batch(self.base, sample_count=size)
        x = self.slice_features
        if x.shape != (len(self.keys["slice"]), len(FEATURES)) or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("invalid temporal features")
        for node in NODES:
            pointers = self.pointers[node]
            if len(pointers) != size + 1 or pointers[0] != 0 or pointers[-1] != len(self.keys[node]):
                raise ValueError("invalid temporal pointers")
            expected = []
            for sample in range(size):
                ids = ([slot_id for _, slot_id in self.slots[sample].time_slots]
                       if node == "slice" else list(self.base[node].entity_id[sample]))
                expected.extend((sample, entity_id) for entity_id in ids)
                if pointers[sample + 1] - pointers[sample] != len(ids):
                    raise ValueError("reference node count mismatch")
            if tuple(expected) != self.keys[node] or len(set(expected)) != len(expected):
                raise ValueError("reference node identity mismatch")
        for source, relation, target in EDGES:
            e = self.edges[source, relation, target]
            if e.dtype != torch.long or e.ndim != 2 or e.shape[0] != 2 or e.device != x.device:
                raise ValueError("invalid temporal edge tensor")
            if e.numel():
                a, b = e.detach().cpu().tolist()
                if min(a + b) < 0 or max(a) >= len(self.keys[source]) or max(b) >= len(self.keys[target]):
                    raise ValueError("temporal edge out of range")
                if any(self.keys[source][i][0] != self.keys[target][j][0] for i, j in zip(a, b)):
                    raise ValueError("cross-sample temporal edge")
            if not relation.startswith("rev_") and not torch.equal(e.flip(0), self.edges[target, "rev_" + relation, source]):
                raise ValueError("missing reverse temporal edge")


def build_temporal_batch(pieces: Sequence) -> TemporalGraphBatch:
    if not pieces:
        raise ValueError("empty temporal batch")
    bases, inventories, features = [], [], []
    keys = {n: [] for n in NODES}
    pointers = {n: [0] for n in NODES}
    edges = {e: [] for e in EDGES}
    for sample, piece in enumerate(pieces):
        base = build_raw_graph(piece)
        slots = build_readout_slots(piece)
        bases.append(base)
        inventories.append(slots)
        times = [t for t, _ in slots.time_slots]
        sources = raw_time_sources(piece)
        offsets = {n: len(keys[n]) for n in NODES}
        for n in NODES:
            ids = [s for _, s in slots.time_slots] if n == "slice" else base[n].entity_id
            keys[n].extend((sample, i) for i in ids)
            pointers[n].append(len(keys[n]))

        def add(source, relation, target, a, b):
            a, b = a + offsets[source], b + offsets[target]
            edges[source, relation, target].append((a, b))
            edges[target, "rev_" + relation, source].append((b, a))

        time_index = {t: i for i, t in enumerate(times)}
        for i, note in enumerate(piece.notes):
            start = exact(note.onset_qn)
            end = start + exact(note.duration_qn)
            add("note", "starts", "slice", i, time_index[start])
            add("note", "ends", "slice", i, time_index[end])
            # Strict interior: attack / held / release are disjoint roles.
            for j in range(bisect_right(times, start), bisect_left(times, end)):
                add("note", "holds", "slice", i, j)
        for i, t in enumerate(sorted(sources["note_onsets"])):
            add("onset", "coincides", "slice", i, time_index[t])
        spans = {n: ([exact(r.start_qn) for r in rows],
                     [exact(r.start_qn) + exact(r.duration_qn) for r in rows])
                 for n, rows in (("beat", piece.beats), ("bar", piece.bars))}
        meters = {m.meter_event_id: m for m in piece.meter_events}
        for j, t in enumerate(times):
            owners = {n: owner(t, *spans[n]) for n in ("beat", "bar")}
            for n, i in owners.items():
                if i is not None:
                    add(n, "contains", "slice", i, j)
            beat = None if owners["beat"] is None else piece.beats[owners["beat"]]
            bar = None if owners["bar"] is None else piece.bars[owners["bar"]]
            meter = None if bar is None else meters[bar.meter_event_id]
            features.append([
                float((t - exact(beat.start_qn)) / exact(beat.duration_qn)) if beat else 0.,
                float(t - exact(bar.start_qn) + exact(bar.metric_offset_qn)) if bar else 0.,
                beat.index_in_bar if beat else 0., meter.numerator if meter else 0.,
                meter.denominator if meter else 0., float(t - times[j-1]) if j else 0.,
                float(times[j+1] - t) if j+1 < len(times) else 0.,
                t in sources["note_onsets"], t in sources["note_offsets"],
                t in sources["beat_boundaries"], beat is not None, bar is not None,
                bar.is_pickup if bar else False, bar.is_incomplete if bar else False,
            ])
            if j:
                add("slice", "next", "slice", j - 1, j)
    result = TemporalGraphBatch(
        Batch.from_data_list(bases), torch.tensor(features, dtype=torch.float32).reshape(-1, len(FEATURES)),
        {e: torch.tensor(v, dtype=torch.long).reshape(-1, 2).t().contiguous() for e, v in edges.items()},
        {n: tuple(v) for n, v in keys.items()}, {n: tuple(v) for n, v in pointers.items()}, tuple(inventories),
    )
    result.validate()
    return result
