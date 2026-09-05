"""Bounded preprocessing composed around the existing canonical corpus cache."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Iterator, TypeVar

from music_critic.adapters.pdmx import (
    PDMX_ADAPTER_VERSION, PDMX_PROJECTION_VERSION, PDMX_SCHEMA_PROFILE_ID,
    PDMX_SCHEMA_UPSTREAM_REVISION,
    PdmxAdapterConfig, PdmxAdapterError, load_pdmx_piece,
)
from music_critic.data.schema import CanonicalPiece
from music_critic.data.serialization import dumps_canonical_json
from music_critic.tasks.corpus import (
    CanonicalCorpusInput, CorpusCacheConfig, CorpusIndex, cache_canonical_corpus,
    corpus_index_dict, load_cached_piece, loads_corpus_index, make_corpus_index,
)
from music_critic.tasks.loading import (
    SplitManifest, create_split_manifest, dumps_split_manifest,
    loads_split_manifest, validate_split_manifest,
)

from .manifest import (
    PdmxPipelineError, SelectionConfig, SourceManifest, SourceRecord, fingerprint,
    guard_split, plan_selection, safe_path, strict_json,
)
from .windowing import WINDOW_VERSION, WindowConfig, WindowingError, iter_bar_windows

PIPELINE_VERSION = "1.0.0"
SHARD_VERSION = "1.0.0"
JOURNAL_VERSION = "1.0.0"
_UNVERIFIED_SOURCE_REASONS = frozenset((
    "pdmx.source_unreadable", "pdmx.source_hash_mismatch", "pdmx.source_bytes_limit",
))

T = TypeVar("T")
U = TypeVar("U")


def bounded_map(function: Callable[[T], U], values: Iterable[T], *, workers: int,
                max_in_flight: int) -> Iterator[U]:
    """At most max_in_flight tasks/results, emitted in input order.

    Threads deliberately avoid copying decoded scores across processes. CPU
    throughput is not claimed; the worker count is an execution-only setting.
    """
    if type(workers) is not int or workers < 1 or type(max_in_flight) is not int or max_in_flight < workers:
        raise PdmxPipelineError("pdmx.worker_budget_invalid")
    if workers == 1:
        for value in values:
            yield function(value)
        return
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pdmx") as pool:
        pending = deque()
        for _ in range(max_in_flight):
            try:
                value = next(iterator)
            except StopIteration:
                break
            pending.append(pool.submit(function, value))
        try:
            while pending:
                result = pending.popleft().result()
                yield result
                try:
                    value = next(iterator)
                except StopIteration:
                    continue
                pending.append(pool.submit(function, value))
        finally:
            for future in pending:
                future.cancel()


def _bytes(value: object) -> bytes:
    return (dumps_canonical_json(value) + "\n").encode("utf-8")


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise PdmxPipelineError("pdmx.artifact_corrupt", str(path.name))
        return
    _atomic(path, payload)


def _sealed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "fingerprint": fingerprint(value)}


def _read_sealed(path: Path) -> dict[str, object]:
    try:
        value = strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("object required")
        digest = value.pop("fingerprint")
        if fingerprint(value) != digest:
            raise ValueError("fingerprint mismatch")
        return value
    except (OSError, ValueError, KeyError) as exc:
        raise PdmxPipelineError("pdmx.artifact_corrupt", path.name) from exc


def _verify_source(path: Path, row: SourceRecord, max_bytes: int) -> None:
    # A resumed cache hit still verifies the source bytes named by the manifest.
    digest = sha256()
    total = 0
    try:
        if path.stat().st_size > max_bytes:
            raise PdmxPipelineError("pdmx.source_bytes_limit")
        with path.open("rb") as stream:
            while chunk := stream.read(min(1024 * 1024, max_bytes + 1 - total)):
                total += len(chunk)
                if total > max_bytes:
                    raise PdmxPipelineError("pdmx.source_bytes_limit")
                digest.update(chunk)
    except OSError as exc:
        raise PdmxPipelineError("pdmx.source_unreadable") from exc
    if digest.hexdigest() != row.sha256:
        raise PdmxPipelineError("pdmx.source_hash_mismatch")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    selection: SelectionConfig
    window: WindowConfig = WindowConfig()
    adapter: PdmxAdapterConfig = PdmxAdapterConfig()
    workers: int = 1
    max_in_flight: int = 2
    split_seed: int = 0
    graph_statistics: bool = False

    def __post_init__(self) -> None:
        if type(self.selection) is not SelectionConfig or type(self.window) is not WindowConfig or type(self.adapter) is not PdmxAdapterConfig:
            raise PdmxPipelineError("pdmx.config_invalid")
        if type(self.workers) is not int or self.workers < 1 or type(self.max_in_flight) is not int or self.max_in_flight < self.workers:
            raise PdmxPipelineError("pdmx.worker_budget_invalid")
        if type(self.split_seed) is not int or type(self.graph_statistics) is not bool:
            raise PdmxPipelineError("pdmx.config_invalid")

    def semantic_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("workers")
        value.pop("max_in_flight")
        return value


@dataclass(frozen=True, slots=True)
class PipelineResult:
    root: Path
    fingerprint: str
    index: CorpusIndex
    split_manifest: SplitManifest
    report: dict[str, object]
    cache_hit_records: int
    cache_hit_windows: int

    @property
    def cache_config(self) -> CorpusCacheConfig:
        return CorpusCacheConfig(self.root, "canonical")


def preprocess_pdmx(
    manifest: SourceManifest, split_by_source: dict[str, str], *,
    source_root: Path, output_root: Path, config: PipelineConfig,
    reader: Callable[..., CanonicalPiece] | None = None,
    on_record_committed: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Explicit preprocessing. Source and cache readers follow split validation.

    `on_record_committed` is a synchronous checkpoint/progress callback; an
    exception stops publication and leaves completed journal entries resumable.
    Unknown/programming exceptions propagate rather than becoming bad inputs.
    """
    plan = plan_selection(manifest, config.selection, split_by_source)
    if manifest.schema_profile != PDMX_SCHEMA_PROFILE_ID:
        raise PdmxPipelineError("pdmx.schema_profile_unsupported")
    if manifest.mode == "production" and not config.selection.require_license_evidence:
        raise PdmxPipelineError("pdmx.production_license_evidence_required")
    components = dict(plan.components)
    splits = dict(plan.splits)
    creation_policy = f"pdmx_{manifest.mode}_explicit"
    contract = {
        "pipeline_version": PIPELINE_VERSION, "shard_version": SHARD_VERSION,
        "journal_version": JOURNAL_VERSION, "adapter_version": PDMX_ADAPTER_VERSION,
        "projection_version": PDMX_PROJECTION_VERSION, "window_version": WINDOW_VERSION,
        "schema_upstream_revision": PDMX_SCHEMA_UPSTREAM_REVISION,
        "manifest": manifest.to_dict(), "selection_fingerprint": plan.fingerprint,
        "selection": {"source_ids": [row.source_id for row in plan.records],
                      "exclusions": list(plan.exclusions), "components": list(plan.components)},
        "splits": list(plan.splits), "config": config.semantic_dict(),
    }
    # Include all shared canonical/graph/target-cache contracts without changing
    # their public APIs. A downstream contract change creates a new namespace.
    empty_index = make_corpus_index(
        dataset_id="pdmx", adapter_name="pdmx", adapter_version=PDMX_ADAPTER_VERSION,
        adapter_config_fingerprint=fingerprint(config.semantic_dict()),
        source_identity="pdmx-explicit-source-manifest", source_fingerprint=manifest.fingerprint,
        creation_policy=creation_policy, records=(),
    )
    contract["shared_contracts"] = asdict(empty_index.header)
    run_fingerprint = fingerprint(contract)
    root = Path(output_root) / f"pdmx-v{PIPELINE_VERSION}" / run_fingerprint
    _immutable(root / "contract.json", _bytes(_sealed(contract)))
    cache = CorpusCacheConfig(root, "canonical")
    source_reader = load_pdmx_piece if reader is None else reader

    def process(row: SourceRecord) -> tuple[dict[str, object], bool]:
        guard_split(splits[row.source_id])
        key = fingerprint(row.source_id)
        journal_path = root / "journal" / f"{key}.json"
        source_path = safe_path(Path(source_root), row.relative_path)
        def existing_shard() -> dict[str, object]:
            journal = _read_sealed(journal_path)
            if journal.get("version") != JOURNAL_VERSION or journal.get("contract") != run_fingerprint or journal.get("source_id") != row.source_id:
                raise PdmxPipelineError("pdmx.journal_binding_mismatch")
            shard_path = safe_path(root, journal["shard_path"])
            try:
                payload = shard_path.read_bytes()
            except OSError as exc:
                raise PdmxPipelineError("pdmx.artifact_corrupt") from exc
            if sha256(payload).hexdigest() != journal["shard_sha256"]:
                raise PdmxPipelineError("pdmx.artifact_corrupt")
            shard = _read_sealed(shard_path)
            if shard.get("version") != SHARD_VERSION or shard.get("contract") != run_fingerprint or shard.get("source_id") != row.source_id:
                raise PdmxPipelineError("pdmx.shard_binding_mismatch")
            if shard["index"] is not None:
                index = loads_corpus_index(dumps_canonical_json(shard["index"]))
                for record in index.records:
                    guard_split(record.suggested_split)
                    piece = load_cached_piece(record, cache)
                    if piece.split != splits[row.source_id] or piece.targets or piece.annotations:
                        raise PdmxPipelineError("pdmx.cached_raw_boundary_invalid")
            return shard

        try:
            _verify_source(source_path, row, config.adapter.max_source_bytes)
        except PdmxPipelineError as exc:
            # Re-observe the same bounded rejection without claiming a source
            # hash or cache hit. A formerly accepted or differently rejected
            # source changing in place is an integrity failure.
            if journal_path.exists():
                old = existing_shard()
                if old["reason"] != exc.category:
                    raise PdmxPipelineError("pdmx.resume_source_changed", exc.category) from exc
                return old, False
            return commit(row, None, [], [], exc.category), False
        if journal_path.exists():
            old = existing_shard()
            if old.get("source_hash_observed") is not True or old["reason"] in _UNVERIFIED_SOURCE_REASONS:
                # This invocation has evidence the earlier attempt lacked.
                # Preserve its immutable quarantine rather than relabeling it
                # as a verified cache hit or replacing it with a new result.
                raise PdmxPipelineError(
                    "pdmx.resume_source_changed",
                    "source now verifies after an unverified attempt; retry with a new output root",
                )
            return old, True
        windows: list[dict[str, object]] = []
        graph_counts: list[tuple[int, int]] = []
        try:
            piece = source_reader(
                source_path, source_id=row.source_id, source_group_id=components[row.source_id],
                split=splits[row.source_id], config=config.adapter,
            )
            # Recheck bytes after the adapter read, including custom readers.
            _verify_source(source_path, row, config.adapter.max_source_bytes)
            if piece.split != splits[row.source_id] or piece.source_group_id != components[row.source_id] or piece.targets or piece.annotations:
                raise PdmxPipelineError("pdmx.reader_contract_invalid")
            if not any(prov.kind == "source" and prov.record_id == row.source_id
                       and prov.checksum_sha256 == row.sha256 for prov in piece.provenance):
                raise PdmxPipelineError("pdmx.reader_source_binding_invalid")
            # Standalone adapters may record their physical path. Published
            # canonical windows bind the manifest's portable relative locator.
            piece = replace(piece, source_path=row.relative_path)

            def inputs() -> Iterator[CanonicalCorpusInput]:
                for window in iter_bar_windows(piece, config=config.window):
                    sidecar = {
                        field.name: (asdict(value) if is_dataclass(value) else value)
                        for field in fields(window) if field.name != "piece"
                        for value in (getattr(window, field.name),)
                    }
                    windows.append(sidecar)
                    if window.piece is None:
                        continue
                    if config.graph_statistics:
                        from music_critic.graph.builder import build_raw_graph
                        graph = build_raw_graph(window.piece)
                        graph_counts.append((int(graph.num_nodes), int(graph.num_edges)))
                        del graph
                    yield CanonicalCorpusInput(
                        piece=window.piece, lineage_group_id=components[row.source_id],
                        source_identity=row.source_id, source_relative_path=row.relative_path,
                        source_sha256=row.sha256, suggested_split=splits[row.source_id],
                        cache_input_sha256=fingerprint({"contract": run_fingerprint, "window": sidecar}),
                    )

            index, _ = cache_canonical_corpus(
                inputs(), cache_config=cache, dataset_id="pdmx", adapter_name="pdmx",
                adapter_version=PDMX_ADAPTER_VERSION, adapter_config={"contract": run_fingerprint},
                source_identity=row.source_id, source_fingerprint=row.sha256,
                creation_policy=creation_policy,
            )
        except (PdmxAdapterError, WindowingError) as exc:
            return commit(row, None, [], [], exc.category), False
        return commit(row, index, windows, graph_counts, None), False

    def commit(row: SourceRecord, index: CorpusIndex | None, windows: list[dict],
               graph_counts: list[tuple[int, int]], reason: str | None) -> dict[str, object]:
        shard = {
            "version": SHARD_VERSION, "contract": run_fingerprint, "source_id": row.source_id,
            "source_sha256": row.sha256, "source_group_id": components[row.source_id],
            "source_hash_observed": reason not in _UNVERIFIED_SOURCE_REASONS,
            "split": splits[row.source_id], "reason": reason,
            "index": None if index is None else corpus_index_dict(index),
            "windows": windows, "graph_counts": graph_counts,
        }
        payload = _bytes(_sealed(shard))
        digest = sha256(payload).hexdigest()
        relative = f"shards/{fingerprint(row.source_id)}.{digest}.json"
        _immutable(root / relative, payload)
        journal = {"version": JOURNAL_VERSION, "contract": run_fingerprint,
                   "source_id": row.source_id, "shard_path": relative, "shard_sha256": digest}
        _immutable(root / "journal" / f"{fingerprint(row.source_id)}.json", _bytes(_sealed(journal)))
        return shard

    records = []
    shard_refs = []
    reasons: Counter[str] = Counter(reason for _, reason in plan.exclusions)
    window_reasons: Counter[str] = Counter()
    accepted_sources, quarantined, clipped, cache_hits, cache_windows = 0, 0, 0, 0, 0
    accepted_groups = set()
    graph_counts = []
    for shard, hit in bounded_map(process, plan.records, workers=config.workers, max_in_flight=config.max_in_flight):
        cache_hits += int(hit)
        source_id = shard["source_id"]
        shard_refs.append({"source_id": source_id, "fingerprint": fingerprint(shard)})
        if shard["reason"] is not None:
            quarantined += 1
            reasons[shard["reason"]] += 1
        else:
            index = loads_corpus_index(dumps_canonical_json(shard["index"]))
            records.extend(index.records)
            accepted_sources += 1
            if index.records:
                accepted_groups.add(shard["source_group_id"])
            cache_windows += len(index.records) if hit else 0
            for window in shard["windows"]:
                if window["reason"]:
                    window_reasons[window["reason"]] += 1
                clipped += window["clipped_note_count"]
            graph_counts.extend(shard["graph_counts"])
        if on_record_committed is not None:
            on_record_committed(source_id)
    index = make_corpus_index(
        dataset_id="pdmx", adapter_name="pdmx", adapter_version=PDMX_ADAPTER_VERSION,
        adapter_config_fingerprint=fingerprint({"contract": run_fingerprint}),
        source_identity="pdmx-explicit-source-manifest", source_fingerprint=manifest.fingerprint,
        creation_policy=creation_policy, records=records,
    )
    split_manifest = create_split_manifest(
        (index,), {(row.dataset_id, row.piece_id): row.suggested_split for row in index.records},
        seed=config.split_seed, policy="pdmx_pre_window_components",
        policy_config={"source_selection_fingerprint": plan.fingerprint},
    )
    validate_split_manifest(split_manifest, (index,))
    report = {
        "version": PIPELINE_VERSION,
        "scope": "synthetic_fixture_only" if manifest.mode == "fixture" else "declared_production_preprocessing",
        "production_ready": False,
        "release_identity": manifest.release_identity, "cross_corpus_links": "unknown",
        "input_records": len(manifest.records), "input_groups": len(set(components.values())),
        "selected_records": len(plan.records), "filtered_records": len(plan.exclusions),
        "accepted_records": accepted_sources, "quarantined_records": quarantined,
        "window_groups": len(accepted_groups), "windows": len(index.records),
        "reasons": dict(sorted(reasons.items())), "window_reasons": dict(sorted(window_reasons.items())),
        "resource_rejected_records": sum(count for reason, count in reasons.items()
            if "resource" in reason or reason == "pdmx.source_bytes_limit"),
        "resource_rejected_windows": sum(count for reason, count in window_reasons.items()
            if "resource" in reason),
        "clipped_note_occurrences": clipped,
        "graph_sizes": _graph_statistics(graph_counts),
        "independence_unit": "source_component", "throughput": "not_computed",
        "corpus_distribution": "not_computed",
    }
    publication = _sealed({"version": PIPELINE_VERSION, "contract": run_fingerprint,
        "index": corpus_index_dict(index), "split_manifest": strict_json(dumps_split_manifest(split_manifest)),
        "source_shards": shard_refs, "report": report})
    _immutable(root / "publication.json", _bytes(publication))
    return PipelineResult(root, run_fingerprint, index, split_manifest, report, cache_hits, cache_windows)


def _graph_statistics(rows: list[tuple[int, int]]) -> dict[str, object]:
    if not rows:
        return {"status": "not_computed", "denominator_graphs": 0}
    # Exact nearest-rank percentiles over graphs actually constructed, not
    # note counts or extrapolation from source metadata.
    from math import ceil
    result: dict[str, object] = {"status": "computed", "denominator_graphs": len(rows)}
    for column, name in enumerate(("nodes", "edges")):
        values = sorted(row[column] for row in rows)
        result[name] = {f"p{p}": values[max(0, ceil(p / 100 * len(values)) - 1)] for p in (50, 90, 99)}
    return result


def validate_pdmx_contract(root: Path, *, split: str) -> str:
    guard_split(split)
    contract = _read_sealed(Path(root) / "contract.json")
    current = {"pipeline_version": PIPELINE_VERSION, "shard_version": SHARD_VERSION,
               "journal_version": JOURNAL_VERSION, "adapter_version": PDMX_ADAPTER_VERSION,
               "projection_version": PDMX_PROJECTION_VERSION, "window_version": WINDOW_VERSION,
               "schema_upstream_revision": PDMX_SCHEMA_UPSTREAM_REVISION}
    if any(contract.get(key) != version for key, version in current.items()):
        raise PdmxPipelineError("pdmx.cached_contract_stale")
    if contract.get("manifest", {}).get("schema_profile") != PDMX_SCHEMA_PROFILE_ID:
        raise PdmxPipelineError("pdmx.cached_schema_profile_stale")
    return fingerprint(contract)


def read_publication(root: Path, *, split: str) -> tuple[CorpusIndex, SplitManifest, dict[str, object]]:
    guard_split(split)
    value = _read_sealed(Path(root) / "publication.json")
    if value.get("version") != PIPELINE_VERSION:
        raise PdmxPipelineError("pdmx.publication_version_invalid")
    if validate_pdmx_contract(root, split=split) != value.get("contract"):
        raise PdmxPipelineError("pdmx.publication_binding_mismatch")
    index = loads_corpus_index(dumps_canonical_json(value["index"]))
    manifest = loads_split_manifest(dumps_canonical_json(value["split_manifest"]))
    for row in manifest.assignments:
        guard_split(row.split)
    validate_split_manifest(manifest, (index,))
    return index, manifest, value["report"]
