"""Pure TRAIN size-index helpers for a separately registered B2 memory gate.

There is no corpus access, loader, cache mutation, CUDA work or experiment
launcher here. Supplied source metadata is an assertion of the caller; its
syntax and internal bindings are checked, but no source file is authenticated.
Accounting mirrors the current cache admission representation after deepcopy.
It is neither process RSS nor a peak/transient or model-activation estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal

import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.performance_data import (
    PreparedRecord, _EncodedSidecarRecord, _accounted_bytes,
)
from music_critic.graph import validate_raw_graph
from music_critic.graph.relations import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES


PERFORMANCE_MEMORY_SCHEMA = "AnalysisGNNPerformanceMemoryIndex@1.0.0"
CacheRepresentation = Literal["decoded_dict", "source_json_bytes"]
EXTREME_AXES = ("notes", "onsets", "beats", "tracks", "nodes", "edges",
                "coarse_length", "accounted_cache_entry_bytes")
MEMORY_GATE_LIMITS = (
    "Coverage is relative to the supplied expected TRAIN identities, not independently verified corpus membership.",
    "One-dimensional observed extrema and their finite ordered pairs do not exhaust joint memory maxima.",
    "Accounted cache-entry bytes exclude cache bookkeeping, process RSS, transient copies and activations.",
    "The attention padding proxy is a token-pair count, not allocated memory bytes.",
    "No GPU update, OOM safety, other-corpus readiness or B4/B8 acceptance is established by these helpers.",
)


class PerformanceMemoryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = self.category = code
        super().__init__(f"{code}: {message}")


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PerformanceMemoryError(f"performance_memory.{code}", message)


def _text(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), "invalid_text", name)


def _count(value: Any, name: str) -> None:
    _require(type(value) is int and value >= 0, "invalid_count", name)


def _digest(value: Any, name: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), "invalid_digest", name)


@dataclass(frozen=True)
class TrainRecordMetadata:
    """Externally supplied TRAIN membership and provenance; never inferred from graph labels."""

    record_id: str
    source_component_id: str
    split: str
    canonical_sha256: str
    sidecar_sha256: str
    metadata_sha256: str
    binding_seal: str
    source_seal: str
    code_seal: str
    split_sha256: str
    source_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require(self.split == "train", "train_only", "explicit canonical train split required")
        _text(self.record_id, "record_id")
        _text(self.source_component_id, "source_component_id")
        for name in ("canonical_sha256", "sidecar_sha256", "metadata_sha256", "binding_seal",
                     "source_seal", "code_seal", "split_sha256"):
            _digest(getattr(self, name), name)
        _require(isinstance(self.source_versions, tuple) and bool(self.source_versions),
                 "invalid_versions", "nonempty immutable source version pairs required")
        names = []
        for pair in self.source_versions:
            _require(isinstance(pair, tuple) and len(pair) == 2, "invalid_versions", "version pair")
            for value in pair:
                _text(value, "source_versions")
            names.append(pair[0])
        _require(len(set(names)) == len(names) and names == sorted(names),
                 "invalid_versions", "unique version names in sorted order required")


@dataclass(frozen=True)
class SizeIndexRow:
    metadata: TrainRecordMetadata
    nodes_by_type: tuple[tuple[str, int], ...]
    edges_by_relation: tuple[tuple[tuple[str, str, str], int], ...]
    graph_versions: tuple[tuple[str, str], ...]
    unique_cpu_tensor_storage_bytes: int
    accounted_decoded_entry_bytes: int
    accounted_compact_entry_bytes: int | None
    cache_representation: CacheRepresentation

    def __post_init__(self) -> None:
        _require(isinstance(self.metadata, TrainRecordMetadata), "invalid_metadata", "TRAIN metadata required")
        self.metadata.__post_init__()
        for counts, expected, name in (
            (self.nodes_by_type, MANDATORY_NODE_TYPES, "nodes"),
            (self.edges_by_relation, MANDATORY_EDGE_TYPES, "edges"),
        ):
            _require(isinstance(counts, tuple) and all(isinstance(pair, tuple) and len(pair) == 2
                                                     for pair in counts), "invalid_counts", name)
            _require(tuple(pair[0] for pair in counts) == expected, "invalid_counts", name)
            for key, value in counts:
                _count(value, str(key))
        _require(dict(self.nodes_by_type)["song"] == 1, "singleton_required", "exactly one song")
        _require(isinstance(self.graph_versions, tuple)
                 and all(isinstance(pair, tuple) and len(pair) == 2
                         and isinstance(pair[0], str) for pair in self.graph_versions)
                 and self.graph_versions == tuple(sorted(self.graph_versions))
                 and tuple(key for key, _ in self.graph_versions) == (
                     "feature_registry_version", "graph_builder_version", "graph_schema_version", "schema_version"),
                 "invalid_versions", "graph version names")
        for _, value in self.graph_versions:
            _text(value, "graph_versions")
        for name in ("unique_cpu_tensor_storage_bytes", "accounted_decoded_entry_bytes"):
            _count(getattr(self, name), name)
        _require(self.accounted_decoded_entry_bytes >= self.unique_cpu_tensor_storage_bytes,
                 "invalid_accounting", "decoded entry is smaller than tensor storage")
        if self.accounted_compact_entry_bytes is not None:
            _count(self.accounted_compact_entry_bytes, "accounted_compact_entry_bytes")
            _require(self.accounted_compact_entry_bytes >= self.unique_cpu_tensor_storage_bytes,
                     "invalid_accounting", "compact entry is smaller than tensor storage")
        _require(self.cache_representation in ("decoded_dict", "source_json_bytes"),
                 "invalid_representation", str(self.cache_representation))
        _require(self.cache_representation != "source_json_bytes" or self.accounted_compact_entry_bytes is not None,
                 "compact_bytes_required", "decoded accounting cannot substitute for compact accounting")

    @property
    def record_id(self) -> str:
        return self.metadata.record_id

    @property
    def nodes(self) -> int:
        return sum(value for _, value in self.nodes_by_type)

    @property
    def edges(self) -> int:
        return sum(value for _, value in self.edges_by_relation)

    @property
    def coarse_length(self) -> int:
        counts = dict(self.nodes_by_type)
        return 1 + counts["bar"] + counts["track"]

    @property
    def accounted_cache_entry_bytes(self) -> int:
        if self.cache_representation == "source_json_bytes":
            assert self.accounted_compact_entry_bytes is not None
            return self.accounted_compact_entry_bytes
        return self.accounted_decoded_entry_bytes

    def axis_value(self, name: str) -> int:
        node_names = {"notes": "note", "onsets": "onset", "beats": "beat", "tracks": "track", "bars": "bar"}
        if name in node_names:
            return dict(self.nodes_by_type)[node_names[name]]
        _require(name in EXTREME_AXES, "invalid_axis", name)
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(schema_version=PERFORMANCE_MEMORY_SCHEMA, record_id=self.record_id,
                      **{name: self.axis_value(name) for name in (*EXTREME_AXES, "bars")})
        result["accounting_policy"] = "performance_data._accounted_bytes_after_defensive_deepcopy"
        return result


def unique_cpu_tensor_storage_bytes(graph: HeteroData) -> int:
    """Count complete backing storages once, including views and zero-length views.

    Python tensor/container headers are excluded. Non-CPU, sparse and quantized
    tensors are rejected rather than moved/materialized or silently undercounted.
    This helper also walks nested metadata containers without mutating them.
    """
    _require(isinstance(graph, HeteroData), "invalid_graph", "HeteroData required")
    seen: set[int] = set()
    storages: set[tuple[int, int]] = set()

    def visit(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            _require(value.device.type == "cpu", "cpu_required", str(value.device))
            _require(value.layout == torch.strided and not value.is_quantized,
                     "unsupported_tensor", str(value.layout))
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in storages:
                return 0
            storages.add(key)
            return storage.nbytes()
        if isinstance(value, Mapping):
            return sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        return 0

    return sum(visit(store._mapping) for store in graph.stores)


def build_size_index_row(
    prepared: PreparedRecord, *, metadata: TrainRecordMetadata,
    cache_representation: CacheRepresentation, compact_sidecar_bytes: bytes | None = None,
) -> SizeIndexRow:
    """Inspect one prepared TRAIN record without loading anything or altering it.

    Source bytes, when supplied, must match the declared sidecar SHA and decoded
    sidecar. They are retained verbatim for the compact accounting; reserialization
    is never used to invent source bytes. A deepcopy of each representation
    mirrors cache admission ownership and is released after its accounting.
    Temporary allocations made by this helper are not measured as a memory gate.
    """
    _require(isinstance(metadata, TrainRecordMetadata), "invalid_metadata", "TRAIN metadata required")
    metadata.__post_init__()  # Gate before touching the graph or sidecar.
    _require(cache_representation in ("decoded_dict", "source_json_bytes"),
             "invalid_representation", str(cache_representation))
    _require(cache_representation != "source_json_bytes" or compact_sidecar_bytes is not None,
             "compact_bytes_required", "original sidecar bytes required")
    _require(isinstance(prepared, PreparedRecord), "invalid_record", "PreparedRecord required")
    _require(prepared.binding_seal == metadata.binding_seal, "binding_mismatch", metadata.record_id)
    _require(isinstance(prepared.sidecar, dict) and prepared.sidecar.get("record_id") == metadata.record_id,
             "record_mismatch", metadata.record_id)
    graph = prepared.raw_graph
    _require(not isinstance(graph, Batch), "singleton_required", "unbatched raw graph required")
    storage_bytes = unique_cpu_tensor_storage_bytes(graph)  # CPU check before graph tensor operations.
    validate_raw_graph(graph)
    for node_type in graph.node_types:
        _count(graph[node_type].num_nodes, node_type)  # Excludes bool, unlike isinstance(int).
    decoded_accounted = _accounted_bytes(deepcopy(prepared))
    compact_accounted = None
    if compact_sidecar_bytes is not None:
        _require(type(compact_sidecar_bytes) is bytes, "invalid_sidecar_bytes", "immutable bytes required")
        _require(sha256(compact_sidecar_bytes).hexdigest() == metadata.sidecar_sha256,
                 "sidecar_hash_mismatch", metadata.record_id)
        try:
            decoded = json.loads(compact_sidecar_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PerformanceMemoryError("performance_memory.invalid_sidecar_json", metadata.record_id) from exc
        try:
            # Python equality conflates True/1 and 1/1.0. Canonical JSON keeps
            # those distinctions while permitting irrelevant key order/spacing.
            signature = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False, allow_nan=False)
            sidecar_equal = isinstance(decoded, dict) and signature(decoded) == signature(prepared.sidecar)
        except (ValueError, TypeError) as exc:
            raise PerformanceMemoryError("performance_memory.invalid_sidecar_json", metadata.record_id) from exc
        _require(sidecar_equal,
                 "sidecar_mismatch", metadata.record_id)
        del decoded
        compact_accounted = _accounted_bytes(deepcopy(
            _EncodedSidecarRecord(graph, compact_sidecar_bytes, prepared.binding_seal)))
    return SizeIndexRow(
        metadata=metadata,
        nodes_by_type=tuple((name, graph[name].num_nodes) for name in graph.node_types),
        edges_by_relation=tuple((name, graph[name].edge_index.shape[1]) for name in graph.edge_types),
        graph_versions=tuple((name, getattr(graph, name)) for name in (
            "feature_registry_version", "graph_builder_version", "graph_schema_version", "schema_version")),
        unique_cpu_tensor_storage_bytes=storage_bytes,
        accounted_decoded_entry_bytes=decoded_accounted,
        accounted_compact_entry_bytes=compact_accounted,
        cache_representation=cache_representation,
    )


@dataclass(frozen=True)
class ObservedExtremum:
    axis: str
    value: int
    record_id: str
    tied_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class B2MemoryCase:
    record_ids: tuple[str, str]
    coarse_lengths: tuple[int, int]
    attention_padding_proxy: int


@dataclass(frozen=True)
class ExtremeSelection:
    schema_version: str
    supplied_index_fingerprint: str
    expected_train_ids_fingerprint: str
    indexed_record_count: int
    indexed_component_count: int
    extrema: tuple[ObservedExtremum, ...]
    selected_rows: tuple[SizeIndexRow, ...]
    ordered_b2_cases: tuple[B2MemoryCase, ...]
    limits: tuple[str, ...] = MEMORY_GATE_LIMITS


def select_extreme_b2_cases(
    rows: Sequence[SizeIndexRow], *, expected_train_ids: Sequence[str],
) -> ExtremeSelection:
    """Select maxima from the entire supplied index, with lexicographic ID ties.

    Explicit expected TRAIN identities are mandatory; omissions/extras/duplicates
    fail. This does not authenticate the caller's membership list. All extrema
    are deduplicated before the sorted Cartesian product (including self-pairs).
    Supplied rows must have a common source/split/code and accounting policy.
    """
    _require(not isinstance(expected_train_ids, (str, bytes)), "invalid_membership", "sequence of IDs required")
    expected = tuple(expected_train_ids)
    _require(bool(expected), "empty_index", "expected TRAIN membership is empty")
    for record_id in expected:
        _text(record_id, "expected_train_ids")
    _require(len(set(expected)) == len(expected), "duplicate_ids", "expected TRAIN identities")
    supplied = tuple(rows)
    _require(bool(supplied), "empty_index", "no size rows")
    for row in supplied:
        _require(isinstance(row, SizeIndexRow), "invalid_row", "SizeIndexRow required")
        row.__post_init__()
    ids = tuple(row.record_id for row in supplied)
    _require(len(set(ids)) == len(ids), "duplicate_ids", "size index")
    _require(set(ids) == set(expected), "incomplete_membership", "supplied index and expected TRAIN IDs differ")
    supplied = tuple(sorted(supplied, key=lambda row: row.record_id))
    bindings = {(row.metadata.source_seal, row.metadata.code_seal, row.metadata.split_sha256,
                 row.metadata.source_versions, row.graph_versions, row.cache_representation)
                for row in supplied}
    _require(len(bindings) == 1, "mixed_bindings", "source, graph or cache policy differs")
    extrema = []
    for axis in EXTREME_AXES:
        maximum = max(row.axis_value(axis) for row in supplied)
        tied = tuple(row.record_id for row in supplied if row.axis_value(axis) == maximum)
        extrema.append(ObservedExtremum(axis, maximum, tied[0], tied))
    selected_ids = {value.record_id for value in extrema}
    selected = tuple(row for row in supplied if row.record_id in selected_ids)
    cases = tuple(B2MemoryCase((left.record_id, right.record_id),
                               (left.coarse_length, right.coarse_length),
                               2 * max(left.coarse_length, right.coarse_length) ** 2)
                  for left in selected for right in selected)
    return ExtremeSelection(
        schema_version=PERFORMANCE_MEMORY_SCHEMA,
        supplied_index_fingerprint=fingerprint([row.to_dict() for row in supplied]),
        expected_train_ids_fingerprint=fingerprint(sorted(expected)),
        indexed_record_count=len(supplied),
        indexed_component_count=len({row.metadata.source_component_id for row in supplied}),
        extrema=tuple(extrema), selected_rows=selected, ordered_b2_cases=cases,
    )


__all__ = ["PERFORMANCE_MEMORY_SCHEMA", "EXTREME_AXES", "MEMORY_GATE_LIMITS",
           "PerformanceMemoryError", "TrainRecordMetadata", "SizeIndexRow", "ObservedExtremum",
           "B2MemoryCase", "ExtremeSelection", "unique_cpu_tensor_storage_bytes",
           "build_size_index_row", "select_extreme_b2_cases"]
