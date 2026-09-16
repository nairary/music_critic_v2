"""Isolated CPU temporal graph cache and exact, primitive-safe IPC payloads.

Entries contain one identity-view graph and the original authenticated sidecar
JSON bytes. Transposition is applied to a fresh collated batch, never an entry.
The LRU limit covers retained Python objects and tensor storage, not process RSS
or temporary graph construction, batches, IPC copies or GPU allocations.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import sys
from threading import RLock

import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.adapters.dilemmadata import _piece_id
from music_critic.data import loads_piece
from music_critic.data.raw_time_readout_slots import ReadoutSlots
from ..contracts import fingerprint
from ..corrected_training import transpose_raw_graph_batch
from ..frozen_training_data import (
    FrozenTrainingData, _FILES, _identity, _json, _require, read_bound_bytes,
)
from ..multitask_contract import TARGET_SIDECAR_VERSION
from .graph import EDGES, NODES, SCHEMA, TemporalGraphBatch, build_temporal_batch

CACHE_SCHEMA = "TemporalSliceIdentityCache@1.0.0"
PAYLOAD_SCHEMA = "TemporalSliceCPUPayload@1.0.0"


def _cpu(graph):
    values = [graph.slice_features, *graph.edges.values()]
    values.extend(v for store in graph.base.stores for v in store.values()
                  if isinstance(v, torch.Tensor))
    if any(value.device.type != "cpu" for value in values):
        raise ValueError("temporal_cache.cpu_graph_required")


def encode_temporal_graph(graph: TemporalGraphBatch) -> dict:
    """IPC envelope, no Fraction/frozenset or custom graph wrapper in payload.

    This is a fresh caller-owned object, not a persistent serialization format.
    Its tensors alias the caller's graph; never pass a private cache entry here.
    Existing bounded CPU pickle transports can inspect all fields recursively.
    """
    graph.validate()
    _cpu(graph)
    return {"schema": PAYLOAD_SCHEMA, "graph_schema": graph.schema,
            "base": graph.base, "slice_features": graph.slice_features,
            "edges": dict(graph.edges), "keys": dict(graph.keys),
            "pointers": dict(graph.pointers),
            "slots": tuple({"policy": s.policy,
                            "times": tuple((t.numerator, t.denominator, key)
                                           for t, key in s.time_slots),
                            "beat_ids": tuple(sorted(s.beat_ids))}
                           for s in graph.slots)}


def decode_temporal_graph(payload: Mapping) -> TemporalGraphBatch:
    """Decode only the dedicated temporal envelope and validate exact topology."""
    expected = {"schema", "graph_schema", "base", "slice_features", "edges",
                "keys", "pointers", "slots"}
    if (not isinstance(payload, Mapping) or set(payload) != expected
            or payload.get("schema") != PAYLOAD_SCHEMA or payload.get("graph_schema") != SCHEMA):
        raise ValueError("temporal_cache.payload_schema_mismatch")
    slots = []
    for row in payload["slots"]:
        if not isinstance(row, Mapping) or set(row) != {"policy", "times", "beat_ids"}:
            raise ValueError("temporal_cache.slot_schema_mismatch")
        times = []
        for num, den, key in row["times"]:
            if type(num) is not int or type(den) is not int or den <= 0:
                raise ValueError("temporal_cache.exact_time_required")
            value = Fraction(num, den)
            if (value.numerator, value.denominator) != (num, den):
                raise ValueError("temporal_cache.canonical_time_required")
            times.append((value, key))
        if len(set(row["beat_ids"])) != len(row["beat_ids"]):
            raise ValueError("temporal_cache.duplicate_beat_id")
        slots.append(ReadoutSlots(tuple(times), frozenset(row["beat_ids"]), row["policy"]))
    graph = TemporalGraphBatch(payload["base"], payload["slice_features"],
                               payload["edges"], payload["keys"], payload["pointers"], tuple(slots))
    graph.validate()
    _cpu(graph)
    return graph


def collate_temporal_graphs(graphs: Sequence[TemporalGraphBatch]) -> TemporalGraphBatch:
    """Collate admitted identity singletons with fresh tensor/metadata ownership."""
    if not graphs or any(len(g.slots) != 1 for g in graphs):
        raise ValueError("temporal_cache.nonempty_singletons_required")
    bases, keys, pointers = [], {n: [] for n in NODES}, {n: [0] for n in NODES}
    edge_rows = {edge: [] for edge in EDGES}
    for sample, graph in enumerate(graphs):
        if graph.schema != SCHEMA:
            raise ValueError("temporal_cache.graph_schema_mismatch")
        _cpu(graph)
        offsets = {n: len(keys[n]) for n in NODES}
        base = graph.base.to_data_list()[0]
        # PyG unbatching stores this global boolean in a one-element tensor.
        base.raw_only = bool(base.raw_only)
        bases.append(base)
        for n in NODES:
            keys[n].extend((sample, entity_id) for _, entity_id in graph.keys[n])
            pointers[n].append(len(keys[n]))
        for source, relation, target in EDGES:
            edge_rows[source, relation, target].append(
                graph.edges[source, relation, target]
                + torch.tensor([[offsets[source]], [offsets[target]]], dtype=torch.long))
    base = Batch.from_data_list(bases)
    # PyG concatenates tensor features; non-tensor metadata can alias its input.
    for store in base.stores:
        for key, value in tuple(store.items()):
            if not isinstance(value, torch.Tensor):
                store[key] = deepcopy(value)
    result = TemporalGraphBatch(
        base, torch.cat([g.slice_features for g in graphs], dim=0),
        {e: torch.cat(v, dim=1) for e, v in edge_rows.items()},
        {n: tuple(v) for n, v in keys.items()},
        {n: tuple(v) for n, v in pointers.items()}, tuple(g.slots[0] for g in graphs),
    )
    result.validate()
    return result


def transpose_temporal_historical(graph: TemporalGraphBatch, shifts) -> TemporalGraphBatch:
    """Preserve the historical C2 pitch transform for a controlled comparison.

    The known historical track pitch-summary behavior is deliberately retained;
    correcting it would introduce another experimental change. Slice topology,
    raw times and metric features are pitch-independent and remain identical.
    The input must be a caller-owned collated batch, not a retained cache entry.
    """
    result = replace(graph, base=transpose_raw_graph_batch(graph.base, tuple(shifts)))
    result.validate()
    return result


def _read_temporal_record(data, record_id, split):
    row = data._gate(record_id, split)  # Split gate precedes all payload access.
    data._check_metadata()
    payloads = {key: read_bound_bytes(row[key]) for key in _FILES}
    metadata, sidecar = _json(payloads["metadata"]), _json(payloads["sidecar"])
    _require(metadata == {"record_id": record_id, "split": split,
                          "piece_sha256": row["piece"]["sha256"],
                          "sidecar_sha256": row["sidecar"]["sha256"]}, "cache_metadata_mismatch")
    _require(type(sidecar) is dict and sidecar.get("record_id") == record_id
             and sidecar.get("source_component_id") == row["source_component_id"], "sidecar_identity_mismatch")
    _require(sidecar.get("schema_version") == TARGET_SIDECAR_VERSION
             and sidecar.get("fingerprint") == fingerprint({key: value for key, value in sidecar.items()
                                                            if key != "fingerprint"}), "sidecar_contract_mismatch")
    _require(sidecar.get("dialect") in ("an_joint", "dlc"), "sidecar_dialect")
    piece = loads_piece(payloads["piece"].decode("utf-8"))
    _require(piece.dataset_name == "dilemmadata" and piece.split in (None, split)
             and piece.source_group_id == row["source_component_id"]
             and piece.piece_id == _piece_id(record_id, sidecar["dialect"]), "canonical_identity_mismatch")
    graph = build_temporal_batch((piece,))
    data._check_metadata()
    for key in _FILES:
        read_bound_bytes(row[key])
    return graph, sidecar, payloads["sidecar"]


def read_temporal_record(data: FrozenTrainingData, record_id: str, *, split: str = "train"):
    """Uncached authenticated canonical reference, TRAIN/VALIDATION, never TEST."""
    if type(data) is not FrozenTrainingData:
        raise ValueError("temporal_cache.frozen_data_required")
    graph, sidecar, _ = _read_temporal_record(data, record_id, split)
    return graph, sidecar


def implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    files = (Path(__file__), Path(__file__).with_name("graph.py"),
             root / "data/raw_time_candidates.py", root / "data/raw_time_readout_slots.py")
    return fingerprint({"schema": CACHE_SCHEMA, "graph": SCHEMA,
                        "files": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in files}})


@dataclass(frozen=True)
class PreparedTemporalRecord:
    graph: TemporalGraphBatch
    sidecar_bytes: bytes
    binding_seal: str


class FrozenTemporalRecordLoader:
    """TRAIN-only loader; source identities checked on hits, hashes on changes.

    Full byte authentication is required at startup/resume through the supplied
    FrozenTrainingData manifest. The cache never discovers sources or writes files.
    """
    synthetic = False

    def __init__(self, data: FrozenTrainingData):
        if type(data) is not FrozenTrainingData:
            raise ValueError("temporal_cache.frozen_data_required")
        self.data = data
        self.implementation_seal = implementation_fingerprint()
        self.source_seal = fingerprint({"schema": CACHE_SCHEMA, "data": data.manifest_fingerprint,
                                        "implementation": self.implementation_seal})
        self._identities = {}

    def check_binding(self, record_id, *, split="train"):
        if split != "train":
            raise ValueError("temporal_cache.train_only")
        row = self.data._gate(record_id, split)
        self.data._check_metadata()
        observed = tuple(_identity(Path(row[key]["path"])) for key in _FILES)
        if self._identities.get(record_id) != observed:
            for key in _FILES:
                read_bound_bytes(row[key])
            final = tuple(_identity(Path(row[key]["path"])) for key in _FILES)
            if final != observed:
                raise ValueError("temporal_cache.source_changed_during_binding")
            self._identities[record_id] = final
        return fingerprint({"source": self.source_seal, "record": record_id,
                            "files": tuple(row[key]["sha256"] for key in _FILES)})

    def get(self, record_id, *, split="train"):
        seal = self.check_binding(record_id, split=split)
        graph, _, sidecar_bytes = _read_temporal_record(self.data, record_id, split)
        if seal != self.check_binding(record_id, split=split):
            raise ValueError("temporal_cache.source_binding_changed")
        return PreparedTemporalRecord(graph, sidecar_bytes, seal)


def accounted_bytes(value) -> int:
    """Retained object/tensor-storage accounting, no RSS or transient promise."""
    seen, storages = set(), set()
    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        result = sys.getsizeof(item)
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages.add(key)
                result += storage.nbytes()
        elif isinstance(item, HeteroData):
            result += sum(size(store._mapping) for store in item.stores)
            for name in ("_slice_dict", "_inc_dict"):
                if hasattr(item, name):
                    result += size(getattr(item, name))
        elif isinstance(item, Mapping):
            result += sum(size(k) + size(v) for k, v in item.items())
        elif isinstance(item, (tuple, list, frozenset, set)):
            result += sum(size(v) for v in item)
        elif isinstance(item, Fraction):
            result += size(item.numerator) + size(item.denominator)
        elif is_dataclass(item) and not isinstance(item, type):
            result += sum(size(getattr(item, f.name)) for f in fields(item))
        return result
    return size(value)


@dataclass(frozen=True)
class PreparedTemporalBatch:
    graph: TemporalGraphBatch
    sidecars: tuple[dict, ...]
    record_ids: tuple[str, ...]
    binding_seals: tuple[str, ...]
    cache_diagnostics: dict[str, int]


class TemporalGraphCache:
    """Process-local bounded LRU; private identity entries never leave the cache."""
    def __init__(self, loader, *, max_bytes: int, max_entries: int, synthetic=False):
        if any(type(v) is not int or v <= 0 for v in (max_bytes, max_entries)):
            raise ValueError("temporal_cache.positive_limits_required")
        if type(synthetic) is not bool or bool(getattr(loader, "synthetic", None)) != synthetic:
            raise ValueError("temporal_cache.loader_mode_mismatch")
        if not synthetic and type(loader) is not FrozenTemporalRecordLoader:
            raise ValueError("temporal_cache.synthetic_opt_in_required")
        self.loader, self.source_seal = loader, loader.source_seal
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries, self._lock = OrderedDict(), RLock()
        self._bytes = self._hits = self._misses = self._evictions = self._oversized = 0

    def _entry(self, record_id, split):
        if split != "train":
            raise ValueError("temporal_cache.train_only")
        if self.source_seal != self.loader.source_seal:
            raise ValueError("temporal_cache.source_binding_changed")
        seal = self.loader.check_binding(record_id, split=split)
        key = (CACHE_SCHEMA, self.source_seal, record_id)
        if key in self._entries:
            record, _ = self._entries[key]
            if seal != record.binding_seal:
                raise ValueError("temporal_cache.source_binding_changed")
            self._hits += 1
            self._entries.move_to_end(key)
            return record
        self._misses += 1
        loaded = self.loader.get(record_id, split=split)
        if (type(loaded) is not PreparedTemporalRecord or loaded.binding_seal != seal
                or type(loaded.sidecar_bytes) is not bytes or len(loaded.graph.slots) != 1):
            raise ValueError("temporal_cache.record_contract_mismatch")
        loaded.graph.validate()
        _cpu(loaded.graph)
        record = deepcopy(loaded)
        size = accounted_bytes(record)
        if size > self.max_bytes:
            self._oversized += 1
            return record
        while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
            _, (_, removed) = self._entries.popitem(last=False)
            self._bytes -= removed
            self._evictions += 1
        self._entries[key] = (record, size)
        self._bytes += size
        return record

    def get(self, record_id, *, split="train") -> PreparedTemporalRecord:
        with self._lock:
            return deepcopy(self._entry(record_id, split))

    def prepare_batch(self, record_ids, *, split="train") -> PreparedTemporalBatch:
        ids = tuple(record_ids)
        if not ids:
            raise ValueError("temporal_cache.nonempty_batch_required")
        with self._lock:
            rows = tuple(self._entry(rid, split) for rid in ids)
            graph = collate_temporal_graphs([row.graph for row in rows])
            sidecars = tuple(_json(row.sidecar_bytes) for row in rows)
            return PreparedTemporalBatch(graph, sidecars, ids, tuple(r.binding_seal for r in rows), self.diagnostics())

    def diagnostics(self):
        with self._lock:
            return {"entries": len(self._entries), "accounted_bytes": self._bytes,
                    "max_bytes": self.max_bytes, "max_entries": self.max_entries,
                    "hits": self._hits, "misses": self._misses, "evictions": self._evictions,
                    "oversized_bypasses": self._oversized}
