"""Isolated TRAIN-only, identity-view CPU preparation for performance replay.

No source conversion, persistent cache writes, sampling, optimizer, or CUDA work
occurs here. Stat checks are change detectors, not content-integrity proofs:
startup/first binding and explicit resume validation hash complete file bytes.
The retained-cache budget accounts Python containers and tensor storage, not
process RSS or transient caller-owned batches. This is a synchronous cache.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
import sys
from threading import RLock
from typing import Any, Protocol

import torch
import torch_geometric
from torch_geometric.data import Batch

from music_critic.data import SCHEMA_VERSION, loads_piece
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.training_policy import CORRECTED_SPLIT_ASSIGNMENTS_SHA256
from music_critic.graph import (
    FEATURE_REGISTRY_VERSION, GRAPH_BUILDER_VERSION, GRAPH_SCHEMA_VERSION,
    validate_raw_graph, validate_raw_graph_batch,
)
from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample


PERFORMANCE_DATA_SCHEMA = "AnalysisGNNPerformanceData@1.0.0"
ContextFactory = Callable[[str], AbstractContextManager[Any]]


class PerformanceDataError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = self.category = code
        super().__init__(f"{code}: {message}")


def _train(split: str) -> None:
    if not isinstance(split, str) or split.casefold() != "train":
        raise PerformanceDataError("performance_data.train_only", str(split))


def _scope(context: ContextFactory | None, name: str) -> AbstractContextManager[Any]:
    return nullcontext() if context is None else context(name)


def _identity(path: Path) -> tuple[int, ...]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise PerformanceDataError("performance_data.source_unavailable", str(path)) from exc
    if not stat.S_ISREG(value.st_mode):
        raise PerformanceDataError("performance_data.regular_file_required", str(path))
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@dataclass
class _FileBinding:
    path: Path
    digest: str
    identity: tuple[int, ...]


def _read_stable(path: Path) -> tuple[bytes, _FileBinding]:
    before = _identity(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PerformanceDataError("performance_data.source_unavailable", str(path)) from exc
    after = _identity(path)
    if before != after:
        raise PerformanceDataError("performance_data.source_changed_during_read", str(path))
    return payload, _FileBinding(path, sha256(payload).hexdigest(), after)


def _verify(binding: _FileBinding, *, full: bool = False) -> None:
    if not full and _identity(binding.path) == binding.identity:
        return
    _, observed = _read_stable(binding.path)
    if observed.digest != binding.digest:
        raise PerformanceDataError("performance_data.source_binding_changed", str(binding.path))
    binding.identity = observed.identity


# Explicit executable bindings for the frozen preparation/identity-view path.
# The module itself is included. This list is not a hash of the Python runtime
# or every transitive third-party dependency; runtime versions are bound below.
_CODE_FILES = (
    "data/__init__.py", "data/schema.py", "data/serialization.py", "data/timing.py",
    "data/validation.py", "graph/__init__.py", "graph/builder.py", "graph/feature_registry.py",
    "graph/relations.py", "graph/serialization.py", "graph/validation.py",
    "tasks/__init__.py", "tasks/multisource.py", "tasks/collator.py", "tasks/alignment.py",
    "tasks/encoding.py", "tasks/registry.py", "experiments/analysisgnn/contracts.py",
    "experiments/analysisgnn/corrected_training.py", "experiments/analysisgnn/training_policy.py",
    "experiments/analysisgnn/multitask_contract.py", "experiments/analysisgnn/transposition.py",
    "experiments/analysisgnn/performance_data.py",
)


def _code_bindings() -> tuple[str, tuple[_FileBinding, ...]]:
    root = Path(__file__).resolve().parents[2]
    bindings = tuple(_read_stable(root / name)[1] for name in _CODE_FILES)
    seal = fingerprint({
        "schema": PERFORMANCE_DATA_SCHEMA, "canonical": SCHEMA_VERSION,
        "graph": GRAPH_SCHEMA_VERSION, "builder": GRAPH_BUILDER_VERSION,
        "features": FEATURE_REGISTRY_VERSION,
        "torch": str(torch.__version__), "pyg": str(torch_geometric.__version__),
        "files": {name: value.digest for name, value in zip(_CODE_FILES, bindings, strict=True)},
    })
    return seal, bindings


@dataclass(frozen=True)
class PreparedRecord:
    raw_graph: Any
    sidecar: dict[str, object]
    binding_seal: str


@dataclass(frozen=True)
class _EncodedSidecarRecord:
    """Private cache representation; bytes come from the sealed source read.

    Unlike a public PreparedRecord, this retains no decoded sidecar containers.
    Graph ownership still requires the same defensive copy on admission.
    """

    raw_graph: Any
    sidecar: bytes
    binding_seal: str


def _prepare(piece: Any, sidecar: dict[str, object], seal: str,
             context: ContextFactory | None) -> PreparedRecord:
    with _scope(context, "original_prepare_graph_and_fingerprints"):
        sample = prepare_multisource_sample(piece)
    with _scope(context, "original_singleton_collation"):
        batch = collate_multisource_samples((sample,))
        graph = batch.raw_graph_batch.to_data_list()[0]
    return PreparedRecord(graph, sidecar, seal)


class RecordLoader(Protocol):
    synthetic: bool
    source_seal: str

    def get(self, record_id: str, *, split: str = "train") -> PreparedRecord: ...
    def check_binding(self, record_id: str, *, split: str = "train") -> str: ...


class FrozenProductionRecordLoader:
    """Read only existing B5C cache entries after the frozen metadata split gate.

    ``get`` always rebuilds and therefore also serves as an uncached baseline.
    ``validate_all_bindings(expected=previous_manifest)`` rehashes code, split,
    and all previously bound records before resume; new records are permitted.
    No file in a record outside TRAIN is opened, including missing-cache paths.
    """

    synthetic = False

    def __init__(self, *, paths: ProductionArtifactPaths = ProductionArtifactPaths(),
                 split: str = "train", context: ContextFactory | None = None) -> None:
        _train(split)  # Before even code/split metadata reads.
        self._context, self._paths, self._lock = context, paths, RLock()
        self._records: dict[str, tuple[_FileBinding, ...]] = {}
        self._record_seals: dict[str, str] = {}
        self.code_seal, self._code = _code_bindings()
        with _scope(context, "split_read_hash"):
            payload, self._split = _read_stable(paths.b3_root / "split_assignments.jsonl")
            if self._split.digest != CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
                raise PerformanceDataError("performance_data.split_hash_mismatch", str(self._split.path))
            rows = tuple(json.loads(line) for line in payload.splitlines() if line.strip())
            counts = {name: sum(row["split"] == name for row in rows)
                      for name in ("train", "validation", "test")}
            components = {name: len({row["source_component_id"] for row in rows
                                    if row["split"] == name}) for name in counts}
            if counts != {"train": 1295, "validation": 162, "test": 162} or components != {
                "train": 1209, "validation": 147, "test": 151,
            } or len({row["record_id"] for row in rows}) != len(rows):
                raise PerformanceDataError("performance_data.split_count_mismatch", str(counts))
            self._train_ids = frozenset(str(row["record_id"]) for row in rows if row["split"] == "train")
        self.source_seal = fingerprint({"code": self.code_seal, "split": self._split.digest,
                                        "cache_root": str(paths.cache_root.resolve())})

    def _gate(self, record_id: str, split: str) -> None:
        _train(split)
        if record_id not in self._train_ids:
            raise PerformanceDataError("performance_data.record_not_train", record_id)

    def _common(self, *, full: bool = False) -> None:
        _verify(self._split, full=full)
        for binding in self._code:
            _verify(binding, full=full)

    def _read_record(self, record_id: str) -> tuple[bytes, bytes, str]:
        token = sha256(record_id.encode("utf-8")).hexdigest()[:24]
        root = self._paths.cache_root / token
        paths = tuple(root / name for name in ("piece.json", "target_sidecar.json", "metadata.json"))
        # Missing warm files are an error, never permission to convert/write.
        values = tuple(_read_stable(path) for path in paths)
        piece_bytes, sidecar_bytes, metadata_bytes = (item[0] for item in values)
        bindings = tuple(item[1] for item in values)
        metadata = json.loads(metadata_bytes)
        if metadata != {"record_id": record_id, "split": "train",
                        "piece_sha256": bindings[0].digest, "sidecar_sha256": bindings[1].digest}:
            raise PerformanceDataError("performance_data.cache_binding_invalid", record_id)
        seal = fingerprint({"source": self.source_seal, "record_id": record_id,
                            "files": tuple(binding.digest for binding in bindings)})
        if record_id in self._record_seals and self._record_seals[record_id] != seal:
            raise PerformanceDataError("performance_data.source_binding_changed", record_id)
        for binding in bindings:
            _verify(binding)
        self._records[record_id], self._record_seals[record_id] = bindings, seal
        return piece_bytes, sidecar_bytes, seal

    def check_binding(self, record_id: str, *, split: str = "train") -> str:
        self._gate(record_id, split)
        with self._lock, _scope(self._context, "source_identity_check"):
            self._common()
            if record_id not in self._records:
                self._read_record(record_id)
            else:
                for binding in self._records[record_id]:
                    _verify(binding)
            return self._record_seals[record_id]

    def _load(self, record_id: str, *, split: str) -> tuple[PreparedRecord, bytes]:
        self._gate(record_id, split)
        with self._lock:
            self._common()
            with _scope(self._context, "read_hash_decode"):
                piece_bytes, sidecar_bytes, seal = self._read_record(record_id)
                piece = loads_piece(piece_bytes.decode("utf-8"))
                sidecar = json.loads(sidecar_bytes)
                if not isinstance(sidecar, dict) or sidecar.get("record_id") != record_id:
                    raise PerformanceDataError("performance_data.sidecar_record_mismatch", record_id)
                # Dilemmadata conversion deliberately keeps this field None.
                # Runtime membership is sealed by _gate and cache metadata;
                # preserve the original canonical piece rather than relabel it.
                if piece.split not in (None, "train"):
                    raise PerformanceDataError("performance_data.piece_not_train", record_id)
            prepared = _prepare(piece, sidecar, seal, self._context)
            self.check_binding(record_id, split=split)
            return prepared, sidecar_bytes

    def get(self, record_id: str, *, split: str = "train") -> PreparedRecord:
        # Preserve the public loader contract: a fresh graph and decoded dict.
        return self._load(record_id, split=split)[0]

    def _get_for_cache(self, record_id: str, *, split: str) -> _EncodedSidecarRecord:
        prepared, sidecar_bytes = self._load(record_id, split=split)
        # These are the exact bytes already hashed and decoded by _load. Do
        # not reserialize, compact, project away fields, or retain the dict.
        return _EncodedSidecarRecord(prepared.raw_graph, sidecar_bytes, prepared.binding_seal)

    def validate_all_bindings(self, *, expected: Mapping[str, object] | None = None) -> dict[str, object]:
        """Full content check for startup/resume; no graph building or writes."""
        with self._lock:
            self._common(full=True)
            if expected is not None:
                if any(expected.get(key) != value for key, value in {
                    "schema": PERFORMANCE_DATA_SCHEMA, "source_seal": self.source_seal,
                }.items()):
                    raise PerformanceDataError("performance_data.resume_binding_mismatch", "source/code/split")
                records = expected.get("record_bindings")
                if not isinstance(records, dict):
                    raise PerformanceDataError("performance_data.resume_binding_mismatch", "record bindings")
                for record_id, seal in records.items():
                    if self.check_binding(record_id) != seal:
                        raise PerformanceDataError("performance_data.resume_binding_mismatch", record_id)
            for bindings in self._records.values():
                for binding in bindings:
                    _verify(binding, full=True)
            return {"schema": PERFORMANCE_DATA_SCHEMA, "source_seal": self.source_seal,
                    "code_seal": self.code_seal, "split_sha256": self._split.digest,
                    "record_bindings": dict(sorted(self._record_seals.items()))}


class SyntheticRecordLoader:
    """Explicit source-free injection; callback returns (singleton batch, sidecar).

    The supplied seal/binding getter is the fixture author's assertion, not proof
    of real source bytes. This class must never wrap a corpus loader. For a warm
    JSON baseline the callback should deserialize sealed fixture bytes and call
    original prepare_multisource_sample/collate_multisource_samples each time.
    """

    synthetic = True

    def __init__(self, loader: Callable[[str], tuple[Any, dict[str, object]]], *,
                 source_seal: str, synthetic: bool = False,
                 binding_getter: Callable[[str], str] | None = None) -> None:
        if not synthetic or not source_seal:
            raise PerformanceDataError("performance_data.synthetic_opt_in_required", "loader injection")
        self._loader, self._binding_getter = loader, binding_getter
        self.source_seal = fingerprint({"schema": PERFORMANCE_DATA_SCHEMA,
                                       "synthetic_source": source_seal})
        self._seals: dict[str, str] = {}

    def check_binding(self, record_id: str, *, split: str = "train") -> str:
        _train(split)
        value = self.source_seal if self._binding_getter is None else self._binding_getter(record_id)
        seal = fingerprint({"source": self.source_seal, "id": record_id, "binding": value})
        if record_id in self._seals and self._seals[record_id] != seal:
            raise PerformanceDataError("performance_data.source_binding_changed", record_id)
        self._seals[record_id] = seal
        return seal

    def get(self, record_id: str, *, split: str = "train") -> PreparedRecord:
        seal = self.check_binding(record_id, split=split)
        batch, sidecar = self._loader(record_id)
        raw = getattr(batch, "raw_graph_batch", batch)
        graphs = raw.to_data_list()
        if len(graphs) != 1:
            raise PerformanceDataError("performance_data.singleton_required", str(len(graphs)))
        return PreparedRecord(graphs[0], deepcopy(sidecar), seal)

    def validate_all_bindings(self, *, expected: Mapping[str, object] | None = None) -> dict[str, object]:
        """Recheck declared fixture identities without rebuilding/loading samples.

        Unlike production validation this is not a byte-integrity certificate:
        it validates only the explicit fixture source seal and binding getter.
        """
        if expected is not None:
            if any(expected.get(key) != value for key, value in {
                "schema": PERFORMANCE_DATA_SCHEMA, "source_seal": self.source_seal,
            }.items()):
                raise PerformanceDataError("performance_data.resume_binding_mismatch", "synthetic source")
            records = expected.get("record_bindings")
            if not isinstance(records, dict):
                raise PerformanceDataError("performance_data.resume_binding_mismatch", "record bindings")
            for record_id, seal in records.items():
                if self.check_binding(record_id) != seal:
                    raise PerformanceDataError("performance_data.resume_binding_mismatch", record_id)
        for record_id in tuple(self._seals):
            self.check_binding(record_id)
        return {"schema": PERFORMANCE_DATA_SCHEMA, "source_seal": self.source_seal,
                "record_bindings": dict(sorted(self._seals.items()))}


def _accounted_bytes(record: PreparedRecord | _EncodedSidecarRecord) -> int:
    seen: set[int] = set()
    storages: set[tuple[int, int]] = set()

    def size(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages.add(key)
                result += storage.nbytes()
        elif isinstance(value, Mapping):
            result += sum(size(key) + size(item) for key, item in value.items())
        elif isinstance(value, (tuple, list)):
            result += sum(size(item) for item in value)
        return result

    graph = record.raw_graph
    return (sys.getsizeof(record) + sys.getsizeof(graph) + size(record.sidecar)
            + size(record.binding_seal) + sum(sys.getsizeof(store) + size(store._mapping)
                                            for store in graph.stores))


@dataclass(frozen=True)
class PreparedPerformanceBatch:
    raw_graph_batch: Any
    sidecars: tuple[dict[str, object], ...]
    record_ids: tuple[str, ...]
    binding_seals: tuple[str, ...]
    cache_diagnostics: dict[str, int]


class PreparedGraphCache:
    """Bounded LRU of private CPU graphs with caller-owned decoded sidecars.

    The concrete production loader supplies its original verified JSON bytes;
    injected/synthetic loaders retain their defensive dictionary-copy path.
    The byte budget covers retained entries, not transient decoded batches/RSS.
    """

    def __init__(self, loader: RecordLoader, *, max_bytes: int, max_entries: int,
                 synthetic: bool = False, context: ContextFactory | None = None) -> None:
        if max_bytes <= 0 or max_entries <= 0:
            raise PerformanceDataError("performance_data.cache_limit_invalid", "positive limits required")
        if not isinstance(loader, FrozenProductionRecordLoader) and not synthetic:
            raise PerformanceDataError("performance_data.synthetic_opt_in_required", "injected cache loader")
        if bool(loader.synthetic) != synthetic:
            raise PerformanceDataError("performance_data.loader_mode_mismatch", "synthetic")
        self._loader, self._context, self._lock = loader, context, RLock()
        self._source_seal = loader.source_seal
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries: OrderedDict[
            tuple[str, ...], tuple[PreparedRecord | _EncodedSidecarRecord, int]
        ] = OrderedDict()
        self._keys: dict[str, tuple[str, ...]] = {}
        self._bytes = self._hits = self._misses = self._evictions = self._oversized = 0

    def _entry(self, record_id: str, split: str) -> PreparedRecord | _EncodedSidecarRecord:
        _train(split)
        if self._source_seal != self._loader.source_seal:
            raise PerformanceDataError("performance_data.source_binding_changed", "loader source seal")
        # A miss enters the source reader once, without a redundant binding read.
        prefix = (PERFORMANCE_DATA_SCHEMA, self._source_seal, record_id)
        existing = self._keys.get(record_id)
        if existing is not None:
            seal = self._loader.check_binding(record_id, split=split)
            if existing[3] != seal:
                raise PerformanceDataError("performance_data.source_binding_changed", record_id)
            self._hits += 1
            self._entries.move_to_end(existing)
            return self._entries[existing][0]
        self._misses += 1
        # Only the concrete source reader can attest these bytes came from its
        # own read/hash/decode boundary. Subclasses and synthetic injections
        # keep the original get()/defensive-copy behavior.
        loaded = (self._loader._get_for_cache(record_id, split=split)
                  if type(self._loader) is FrozenProductionRecordLoader else
                  self._loader.get(record_id, split=split))
        validate_raw_graph(loaded.raw_graph)
        if any(value.device.type != "cpu" for store in loaded.raw_graph.stores
               for value in store.values() if isinstance(value, torch.Tensor)):
            raise PerformanceDataError("performance_data.cpu_graph_required", record_id)
        # Injection and original singleton unbatching may retain aliases.
        with _scope(self._context, "cache_entry_copy"):
            record = deepcopy(loaded)
        with _scope(self._context, "cache_entry_accounting"):
            accounted = _accounted_bytes(record)
        if accounted > self.max_bytes:
            self._oversized += 1
            return record
        while self._entries and (len(self._entries) >= self.max_entries
                                 or self._bytes + accounted > self.max_bytes):
            removed_key, (_, removed) = self._entries.popitem(last=False)
            del self._keys[removed_key[2]]
            self._bytes -= removed
            self._evictions += 1
        key = (*prefix, record.binding_seal)
        self._entries[key] = (record, accounted)
        self._keys[record_id] = key
        self._bytes += accounted
        return record

    def get(self, record_id: str, *, split: str = "train") -> PreparedRecord:
        _train(split)
        with self._lock:
            record = self._entry(record_id, split)
            if isinstance(record, PreparedRecord):
                return deepcopy(record)
            return PreparedRecord(deepcopy(record.raw_graph), self._sidecar(record), record.binding_seal)

    def _sidecar(self, record: PreparedRecord | _EncodedSidecarRecord) -> dict[str, object]:
        with _scope(self._context, "sidecar_materialize"):
            # Every occurrence (including duplicate IDs in one batch) gets an
            # independent dict. bytes are immutable and were validated at read.
            return (json.loads(record.sidecar) if isinstance(record, _EncodedSidecarRecord)
                    else deepcopy(record.sidecar))

    def prepare_batch(self, record_ids: Sequence[str], *, split: str = "train") -> PreparedPerformanceBatch:
        _train(split)
        ids = tuple(record_ids)
        if not ids:
            raise PerformanceDataError("performance_data.empty_batch", "at least one record required")
        with self._lock:
            records = tuple(self._entry(record_id, split) for record_id in ids)
            with _scope(self._context, "direct_collation"):
                batch = Batch.from_data_list([record.raw_graph for record in records])
                # PyG concatenates tensors; copy metadata which can contain
                # references to input lists/tuples. No input view is mutated.
                source_storages = {value.untyped_storage().data_ptr()
                                   for record in records for store in record.raw_graph.stores
                                   for value in store.values() if isinstance(value, torch.Tensor)
                                   and value.numel()}
                for store in batch.stores:
                    for key, value in tuple(store.items()):
                        if not isinstance(value, torch.Tensor):
                            store[key] = deepcopy(value)
                        elif value.numel() and value.untyped_storage().data_ptr() in source_storages:
                            store[key] = value.clone()
                validate_raw_graph_batch(batch, sample_count=len(ids))
            return PreparedPerformanceBatch(batch, tuple(self._sidecar(row) for row in records),
                                            ids, tuple(row.binding_seal for row in records), self.diagnostics())

    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "accounted_bytes": self._bytes,
                    "max_bytes": self.max_bytes, "max_entries": self.max_entries,
                    "hits": self._hits, "misses": self._misses,
                    "evictions": self._evictions, "oversized_bypasses": self._oversized}


__all__ = ["PERFORMANCE_DATA_SCHEMA", "PerformanceDataError", "PreparedRecord", "RecordLoader",
           "FrozenProductionRecordLoader", "SyntheticRecordLoader", "PreparedGraphCache",
           "PreparedPerformanceBatch"]
