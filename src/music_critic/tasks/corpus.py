"""Versioned, portable canonical corpus index and cache contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable

from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalPiece,
    dumps_piece,
    loads_piece,
    validate_or_raise,
)
from music_critic.graph import (
    FEATURE_REGISTRY_VERSION,
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
)
from music_critic.tasks.multisource import TaskAvailability, project_multisource_targets
from music_critic.tasks.ontology import (
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_fingerprint,
)
from music_critic.tasks.encoding import (
    TARGET_ENCODING_REGISTRY_VERSION,
    target_encoding_contract_fingerprint,
)


MULTISOURCE_CORPUS_INDEX_VERSION = "1.0.0"
MULTISOURCE_CACHE_VERSION = "1.0.0"
HOOKTHEORY_CORPUS_ADAPTER_VERSION = "2B.1"


class CorpusContractError(ValueError):
    """Structured failure raised by corpus index/cache operations."""

    def __init__(
        self,
        category: str,
        message: str | None = None,
        *,
        dataset_id: str | None = None,
        piece_id: str | None = None,
    ) -> None:
        if message is None:
            message = category
            category = "corpus.worker_propagated"
        self.category = category
        self.dataset_id = dataset_id
        self.piece_id = piece_id
        super().__init__(f"[{category}] {message}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable_relative_path(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CorpusContractError(
            "corpus_index.portability",
            f"{field_name} must be a non-empty POSIX relative path",
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise CorpusContractError(
            "corpus_index.portability",
            f"{field_name} must be a normalized POSIX relative path",
        )
    return value


@dataclass(frozen=True, slots=True)
class CorpusCacheConfig:
    """Explicit cache root and immutable namespace policy."""

    root: Path
    namespace: str = f"multisource-cache-v{MULTISOURCE_CACHE_VERSION}"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))
        _portable_relative_path(self.namespace, "namespace")


@dataclass(frozen=True, slots=True)
class CorpusIndexHeader:
    index_version: str
    cache_version: str
    dataset_id: str
    adapter_name: str
    adapter_version: str
    adapter_config_fingerprint: str
    canonical_schema_version: str
    graph_schema_version: str
    graph_builder_version: str
    feature_registry_version: str
    target_ontology_version: str
    target_ontology_fingerprint: str
    target_encoding_version: str
    target_encoding_fingerprint: str
    source_identity: str
    source_fingerprint: str
    creation_policy: str
    record_count: int
    index_fingerprint: str

    def __post_init__(self) -> None:
        text_fields = (
            self.dataset_id,
            self.adapter_name,
            self.adapter_version,
            self.source_identity,
            self.creation_policy,
        )
        if not all(isinstance(value, str) and value for value in text_fields):
            raise CorpusContractError(
                "corpus_index.header_invalid",
                "header identity and policy fields must be non-empty strings",
            )
        for name in (
            "adapter_config_fingerprint",
            "target_ontology_fingerprint",
            "target_encoding_fingerprint",
            "source_fingerprint",
            "index_fingerprint",
        ):
            value = getattr(self, name)
            if not _is_sha256(value):
                raise CorpusContractError(
                    "corpus_index.header_invalid",
                    f"{name} must be non-empty lowercase SHA-256",
                )
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 0
        ):
            raise CorpusContractError(
                "corpus_index.header_invalid",
                "record_count must be a non-negative integer",
            )


@dataclass(frozen=True, slots=True)
class IndexedCorpusRecord:
    """Portable accepted-piece record; quarantine never enters this collection."""

    dataset_id: str
    piece_id: str
    source_group_id: str
    lineage_group_id: str
    source_identity: str
    source_relative_path: str
    source_sha256: str
    cache_key: str
    canonical_relative_path: str
    canonical_sha256: str
    target_availability: tuple[TaskAvailability, ...]
    suggested_split: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.source_group_id,
                self.lineage_group_id,
                self.source_identity,
            )
        ):
            raise CorpusContractError(
                "corpus_index.record_invalid",
                "record identity fields must be non-empty strings",
                dataset_id=self.dataset_id,
                piece_id=self.piece_id,
            )
        _portable_relative_path(self.source_relative_path, "source_relative_path")
        _portable_relative_path(
            self.canonical_relative_path, "canonical_relative_path"
        )
        for name in ("source_sha256", "cache_key", "canonical_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise CorpusContractError(
                    "corpus_index.record_invalid",
                    f"{name} must be lowercase SHA-256",
                    dataset_id=self.dataset_id,
                    piece_id=self.piece_id,
                )
        if self.suggested_split is not None and (
            not isinstance(self.suggested_split, str) or not self.suggested_split
        ):
            raise CorpusContractError(
                "corpus_index.record_invalid",
                "suggested_split must be null or a non-empty string",
            )
        task_ids = tuple(item.task_id for item in self.target_availability)
        if task_ids != tuple(sorted(task_ids)) or len(task_ids) != len(set(task_ids)):
            raise CorpusContractError(
                "corpus_index.record_invalid",
                "target availability must be uniquely sorted by task ID",
            )


def _header_core(header: CorpusIndexHeader) -> dict[str, object]:
    value = asdict(header)
    value["index_fingerprint"] = ""
    return value


def _record_dict(record: IndexedCorpusRecord) -> dict[str, object]:
    value = asdict(record)
    value["target_availability"] = [
        asdict(item) for item in record.target_availability
    ]
    return value


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    header: CorpusIndexHeader
    records: tuple[IndexedCorpusRecord, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.records, key=lambda row: (row.dataset_id, row.piece_id)))
        if self.records != ordered:
            raise CorpusContractError(
                "corpus_index.order_invalid",
                "records must be sorted by dataset_id and piece_id",
            )
        keys = tuple((row.dataset_id, row.piece_id) for row in self.records)
        if len(keys) != len(set(keys)):
            clusters: dict[
                tuple[str, str], list[IndexedCorpusRecord]
            ] = {}
            for row in self.records:
                clusters.setdefault(
                    (row.dataset_id, row.piece_id), []
                ).append(row)
            duplicates = tuple(
                {
                    "dataset_id": dataset_id,
                    "piece_id": piece_id,
                    "cluster_size": len(rows),
                    "sources": tuple(
                        sorted(
                            (
                                row.source_identity,
                                row.source_relative_path,
                            )
                            for row in rows
                        )
                    ),
                }
                for (dataset_id, piece_id), rows in sorted(
                    clusters.items()
                )
                if len(rows) > 1
            )
            raise CorpusContractError(
                "corpus_index.duplicate_piece",
                "duplicate dataset/piece identities are rejected; "
                f"clusters={duplicates!r}",
                dataset_id=duplicates[0]["dataset_id"],
                piece_id=duplicates[0]["piece_id"],
            )
        if any(row.dataset_id != self.header.dataset_id for row in self.records):
            raise CorpusContractError(
                "corpus_index.dataset_mismatch",
                "all records must match the header dataset_id",
            )
        if self.header.record_count != len(self.records):
            raise CorpusContractError(
                "corpus_index.count_mismatch",
                "header record_count differs from records",
            )
        actual = _fingerprint(
            {
                "header": _header_core(self.header),
                "records": [_record_dict(row) for row in self.records],
            }
        )
        if self.header.index_fingerprint != actual:
            raise CorpusContractError(
                "corpus_index.fingerprint_mismatch",
                "index fingerprint differs from deterministic content",
            )


@dataclass(frozen=True, slots=True)
class CorpusQuarantineRecord:
    dataset_id: str
    source_identity: str
    source_relative_path: str
    category: str
    message: str

    def __post_init__(self) -> None:
        _portable_relative_path(self.source_relative_path, "source_relative_path")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.source_identity,
                self.category,
                self.message,
            )
        ):
            raise CorpusContractError(
                "corpus_build.quarantine_invalid",
                "quarantine fields must be non-empty strings",
            )


@dataclass(frozen=True, slots=True)
class CorpusBuildReport:
    dataset_id: str
    indexed_count: int
    accepted_count: int
    quarantine: tuple[CorpusQuarantineRecord, ...]
    failure_category_counts: tuple[tuple[str, int], ...]
    unique_source_group_count: int
    unique_lineage_group_count: int
    suggested_split_counts: tuple[tuple[str, int], ...]
    suggested_split_conflict_count: int
    cache_hit_count: int
    cache_miss_count: int
    raw_only_piece_count: int
    target_available_counts: tuple[tuple[str, int], ...]
    target_masked_counts: tuple[tuple[str, int], ...]
    index_fingerprint: str

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantine)

    def __post_init__(self) -> None:
        counts = (
            self.indexed_count,
            self.accepted_count,
            self.unique_source_group_count,
            self.unique_lineage_group_count,
            self.suggested_split_conflict_count,
            self.cache_hit_count,
            self.cache_miss_count,
            self.raw_only_piece_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise CorpusContractError(
                "corpus_build.report_invalid",
                "build report counts must be non-negative integers",
            )
        if (
            self.indexed_count != self.accepted_count
            or self.cache_hit_count + self.cache_miss_count != self.accepted_count
            or self.raw_only_piece_count > self.accepted_count
        ):
            raise CorpusContractError(
                "corpus_build.report_invalid",
                "accepted/cache/raw-only counts are inconsistent",
            )
        if not _is_sha256(self.index_fingerprint):
            raise CorpusContractError(
                "corpus_build.report_invalid",
                "index_fingerprint must be non-empty lowercase SHA-256",
            )


@dataclass(frozen=True, slots=True)
class CanonicalCorpusInput:
    piece: CanonicalPiece
    lineage_group_id: str
    source_identity: str
    source_relative_path: str
    source_sha256: str
    suggested_split: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.piece, CanonicalPiece):
            raise CorpusContractError(
                "corpus_build.input_invalid",
                "piece must be CanonicalPiece",
            )
        if not all(
            isinstance(value, str) and value
            for value in (self.lineage_group_id, self.source_identity)
        ):
            raise CorpusContractError(
                "corpus_build.input_invalid",
                "lineage and source identities must be non-empty strings",
            )
        _portable_relative_path(self.source_relative_path, "source_relative_path")
        if not _is_sha256(self.source_sha256):
            raise CorpusContractError(
                "corpus_build.input_invalid",
                "source_sha256 must be lowercase SHA-256",
            )
        if self.suggested_split is not None and (
            not isinstance(self.suggested_split, str) or not self.suggested_split
        ):
            raise CorpusContractError(
                "corpus_build.input_invalid",
                "suggested_split must be null or a non-empty string",
            )


def corpus_cache_key(
    *,
    source_identity: str,
    source_sha256: str,
    adapter_name: str,
    adapter_version: str,
    adapter_config_fingerprint: str,
) -> str:
    if not all(
        isinstance(value, str) and value
        for value in (source_identity, adapter_name, adapter_version)
    ):
        raise CorpusContractError(
            "corpus_cache.key_invalid",
            "cache-key identity and adapter fields must be non-empty strings",
        )
    if not _is_sha256(source_sha256) or not _is_sha256(
        adapter_config_fingerprint
    ):
        raise CorpusContractError(
            "corpus_cache.key_invalid",
            "cache-key source/config fingerprints must be non-empty lowercase "
            "SHA-256",
        )
    return _fingerprint(
        {
            "cache_version": MULTISOURCE_CACHE_VERSION,
            "source_identity": source_identity,
            "source_sha256": source_sha256,
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "adapter_config_fingerprint": adapter_config_fingerprint,
            "canonical_schema_version": SCHEMA_VERSION,
            "target_ontology_version": TARGET_ONTOLOGY_VERSION,
            "target_ontology_fingerprint": ontology_contract_fingerprint(),
        }
    )


def _safe_cache_path(config: CorpusCacheConfig, relative_path: str) -> Path:
    _portable_relative_path(relative_path, "canonical_relative_path")
    root = (config.root / config.namespace).resolve()
    candidate = (root / PurePosixPath(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CorpusContractError(
            "corpus_cache.path_escape",
            "cache artifact path escapes the configured namespace",
        ) from exc
    return candidate


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_piece(
    config: CorpusCacheConfig,
    *,
    cache_key: str,
    piece: CanonicalPiece,
) -> tuple[str, str, bool]:
    validate_or_raise(piece)
    payload = dumps_piece(piece).encode("utf-8")
    artifact_sha256 = sha256(payload).hexdigest()
    artifact_identity = _fingerprint(
        {"cache_key": cache_key, "canonical_sha256": artifact_sha256}
    )
    relative = (
        f"artifacts/{artifact_identity[:2]}/{artifact_identity}.canonical.json"
    )
    path = _safe_cache_path(config, relative)
    cache_hit = path.exists()
    if cache_hit:
        existing = path.read_bytes()
        if sha256(existing).hexdigest() != artifact_sha256:
            raise CorpusContractError(
                "corpus_cache.existing_artifact_mismatch",
                f"existing cache artifact has unexpected content: {relative}",
                dataset_id=piece.dataset_name,
                piece_id=piece.piece_id,
            )
        loads_piece(existing.decode("utf-8"))
    else:
        _write_atomic(path, payload)
    return relative, artifact_sha256, cache_hit


def make_corpus_index(
    *,
    dataset_id: str,
    adapter_name: str,
    adapter_version: str,
    adapter_config_fingerprint: str,
    source_identity: str,
    source_fingerprint: str,
    creation_policy: str,
    records: Iterable[IndexedCorpusRecord],
) -> CorpusIndex:
    ordered = tuple(sorted(records, key=lambda row: (row.dataset_id, row.piece_id)))
    header_values: dict[str, object] = {
        "index_version": MULTISOURCE_CORPUS_INDEX_VERSION,
        "cache_version": MULTISOURCE_CACHE_VERSION,
        "dataset_id": dataset_id,
        "adapter_name": adapter_name,
        "adapter_version": adapter_version,
        "adapter_config_fingerprint": adapter_config_fingerprint,
        "canonical_schema_version": SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "target_ontology_version": TARGET_ONTOLOGY_VERSION,
        "target_ontology_fingerprint": ontology_contract_fingerprint(),
        "target_encoding_version": TARGET_ENCODING_REGISTRY_VERSION,
        "target_encoding_fingerprint": target_encoding_contract_fingerprint(),
        "source_identity": source_identity,
        "source_fingerprint": source_fingerprint,
        "creation_policy": creation_policy,
        "record_count": len(ordered),
    }
    fingerprint = _fingerprint(
        {
            "header": {**header_values, "index_fingerprint": ""},
            "records": [_record_dict(row) for row in ordered],
        }
    )
    header = CorpusIndexHeader(
        **header_values,
        index_fingerprint=fingerprint,
    )
    return CorpusIndex(header, ordered)


def cache_canonical_corpus(
    inputs: Iterable[CanonicalCorpusInput],
    *,
    cache_config: CorpusCacheConfig,
    dataset_id: str,
    adapter_name: str,
    adapter_version: str,
    adapter_config: Mapping[str, object],
    source_identity: str,
    source_fingerprint: str | Callable[[], str],
    creation_policy: str = "offline_explicit",
    quarantine: Iterable[CorpusQuarantineRecord] = (),
) -> tuple[CorpusIndex, CorpusBuildReport]:
    adapter_config_fingerprint = _fingerprint(dict(adapter_config))
    records: list[IndexedCorpusRecord] = []
    available: Counter[str] = Counter()
    masked: Counter[str] = Counter()
    cache_hits = 0
    cache_misses = 0
    raw_only_pieces = 0
    for item in inputs:
        piece = item.piece
        if piece.dataset_name != dataset_id:
            raise CorpusContractError(
                "corpus_build.dataset_mismatch",
                "canonical piece dataset differs from requested corpus",
                dataset_id=dataset_id,
                piece_id=piece.piece_id,
            )
        cache_key = corpus_cache_key(
            source_identity=item.source_identity,
            source_sha256=item.source_sha256,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            adapter_config_fingerprint=adapter_config_fingerprint,
        )
        relative, canonical_sha, cache_hit = _write_piece(
            cache_config, cache_key=cache_key, piece=piece
        )
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)
        raw_only_pieces += int(not piece.targets)
        availability = project_multisource_targets(
            piece, lineage_group_id=item.lineage_group_id
        ).target_availability
        for row in availability:
            available[row.task_id] += row.available_count
            masked[row.task_id] += row.masked_count
        records.append(
            IndexedCorpusRecord(
                dataset_id=dataset_id,
                piece_id=piece.piece_id,
                source_group_id=piece.source_group_id,
                lineage_group_id=item.lineage_group_id,
                source_identity=item.source_identity,
                source_relative_path=item.source_relative_path,
                source_sha256=item.source_sha256,
                cache_key=cache_key,
                canonical_relative_path=relative,
                canonical_sha256=canonical_sha,
                target_availability=availability,
                suggested_split=item.suggested_split,
            )
        )
    resolved_source_fingerprint = (
        source_fingerprint()
        if callable(source_fingerprint)
        else source_fingerprint
    )
    index = make_corpus_index(
        dataset_id=dataset_id,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        adapter_config_fingerprint=adapter_config_fingerprint,
        source_identity=source_identity,
        source_fingerprint=resolved_source_fingerprint,
        creation_policy=creation_policy,
        records=records,
    )
    quarantined = tuple(
        sorted(
            quarantine,
            key=lambda row: (
                row.dataset_id,
                row.source_identity,
                row.source_relative_path,
                row.category,
            ),
        )
    )
    suggested: Counter[str] = Counter(
        row.suggested_split
        for row in index.records
        if row.suggested_split is not None
    )
    split_by_component: dict[tuple[str, str], set[str]] = {}
    for row in index.records:
        if row.suggested_split is None:
            continue
        for key in (
            ("source", row.source_group_id),
            ("lineage", row.lineage_group_id),
        ):
            split_by_component.setdefault(key, set()).add(row.suggested_split)
    report = CorpusBuildReport(
        dataset_id=dataset_id,
        indexed_count=len(index.records),
        accepted_count=len(index.records),
        quarantine=quarantined,
        failure_category_counts=tuple(
            sorted(Counter(row.category for row in quarantined).items())
        ),
        unique_source_group_count=len(
            {row.source_group_id for row in index.records}
        ),
        unique_lineage_group_count=len(
            {row.lineage_group_id for row in index.records}
        ),
        suggested_split_counts=tuple(sorted(suggested.items())),
        suggested_split_conflict_count=sum(
            len(splits) > 1 for splits in split_by_component.values()
        ),
        cache_hit_count=cache_hits,
        cache_miss_count=cache_misses,
        raw_only_piece_count=raw_only_pieces,
        target_available_counts=tuple(sorted(available.items())),
        target_masked_counts=tuple(sorted(masked.items())),
        index_fingerprint=index.header.index_fingerprint,
    )
    return index, report


def corpus_index_dict(index: CorpusIndex) -> dict[str, object]:
    return {
        "header": asdict(index.header),
        "records": [_record_dict(row) for row in index.records],
    }


def dumps_corpus_index(index: CorpusIndex) -> str:
    return _canonical_json(corpus_index_dict(index)) + "\n"


def dump_corpus_index(index: CorpusIndex, path: str | os.PathLike[str]) -> None:
    _write_atomic(Path(path), dumps_corpus_index(index).encode("utf-8"))


def _availability_from_dict(value: object) -> tuple[TaskAvailability, ...]:
    if not isinstance(value, list):
        raise CorpusContractError(
            "corpus_index.parse_error", "target_availability must be a list"
        )
    try:
        return tuple(TaskAvailability(**row) for row in value)
    except (TypeError, ValueError) as exc:
        raise CorpusContractError(
            "corpus_index.parse_error", f"invalid target availability: {exc}"
        ) from exc


def loads_corpus_index(payload: str, *, require_current: bool = True) -> CorpusIndex:
    try:
        value = json.loads(payload)
        header_value = value["header"]
        record_values = value["records"]
        header = CorpusIndexHeader(**header_value)
        records = tuple(
            IndexedCorpusRecord(
                **{
                    **row,
                    "target_availability": _availability_from_dict(
                        row["target_availability"]
                    ),
                }
            )
            for row in record_values
        )
        index = CorpusIndex(header, records)
    except CorpusContractError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CorpusContractError(
            "corpus_index.parse_error", f"invalid corpus index: {exc}"
        ) from exc
    if require_current:
        validate_current_corpus_index(index)
    return index


def validate_current_corpus_index(index: CorpusIndex) -> None:
    """Require every runtime-affecting index contract to match this checkout."""

    expected = {
        "index_version": MULTISOURCE_CORPUS_INDEX_VERSION,
        "cache_version": MULTISOURCE_CACHE_VERSION,
        "canonical_schema_version": SCHEMA_VERSION,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "target_ontology_version": TARGET_ONTOLOGY_VERSION,
        "target_ontology_fingerprint": ontology_contract_fingerprint(),
        "target_encoding_version": TARGET_ENCODING_REGISTRY_VERSION,
        "target_encoding_fingerprint": target_encoding_contract_fingerprint(),
    }
    for name, current in expected.items():
        if getattr(index.header, name) != current:
            raise CorpusContractError(
                "corpus_index.contract_mismatch",
                f"{name} differs from the current runtime contract",
                dataset_id=index.header.dataset_id,
            )


def load_corpus_index(
    path: str | os.PathLike[str], *, require_current: bool = True
) -> CorpusIndex:
    try:
        payload = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusContractError(
            "corpus_index.unreadable", f"cannot read corpus index: {exc}"
        ) from exc
    return loads_corpus_index(payload, require_current=require_current)


def load_cached_piece(
    record: IndexedCorpusRecord, cache_config: CorpusCacheConfig
) -> CanonicalPiece:
    path = _safe_cache_path(cache_config, record.canonical_relative_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CorpusContractError(
            "corpus_cache.artifact_unreadable",
            f"cannot read canonical artifact: {record.canonical_relative_path}: {exc}",
            dataset_id=record.dataset_id,
            piece_id=record.piece_id,
        ) from exc
    if sha256(payload).hexdigest() != record.canonical_sha256:
        raise CorpusContractError(
            "corpus_cache.artifact_fingerprint_mismatch",
            "canonical artifact SHA-256 differs from index",
            dataset_id=record.dataset_id,
            piece_id=record.piece_id,
        )
    try:
        piece = loads_piece(payload.decode("utf-8"))
        validate_or_raise(piece)
    except (UnicodeError, ValueError) as exc:
        raise CorpusContractError(
            "corpus_cache.artifact_invalid",
            f"canonical artifact is invalid: {exc}",
            dataset_id=record.dataset_id,
            piece_id=record.piece_id,
        ) from exc
    if piece.dataset_name != record.dataset_id or piece.piece_id != record.piece_id:
        raise CorpusContractError(
            "corpus_cache.identity_mismatch",
            "canonical artifact identity differs from index",
            dataset_id=record.dataset_id,
            piece_id=record.piece_id,
        )
    if piece.source_group_id != record.source_group_id:
        raise CorpusContractError(
            "corpus_cache.source_group_mismatch",
            "canonical artifact source_group_id differs from index",
            dataset_id=record.dataset_id,
            piece_id=record.piece_id,
        )
    return piece


def _jsonable_source(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_source(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_source(item) for item in value]
    return value


def _validate_builder_limit(limit: int | None) -> None:
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise CorpusContractError(
            "corpus_build.limit_invalid",
            "limit must be None or a positive non-bool integer",
        )


def build_hooktheory_corpus_cache(
    raw_path: str | os.PathLike[str],
    *,
    cache_config: CorpusCacheConfig,
    dataset_id: str = "hooktheory",
    include_targets: bool = True,
    structure_root: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> tuple[CorpusIndex, CorpusBuildReport]:
    """Stream HookTheory once and build a deterministic offline cache/index."""

    from music_critic.adapters import (
        HookTheoryAdapterConfig,
        HookTheoryAdapterError,
        convert_hooktheory_record,
    )
    from music_critic.adapters._json_stream import iter_jsonl, iter_object_records

    _validate_builder_limit(limit)
    raw = Path(raw_path)
    if raw.name != "4_merged.json":
        raise CorpusContractError(
            "hooktheory.source_invalid",
            "HookTheory source must be named 4_merged.json",
        )
    structure: dict[str, Mapping[str, Any]] = {}
    if structure_root is not None:
        for split in ("train", "val", "test"):
            structure_path = Path(structure_root) / f"HookTheoryStructure.{split}.jsonl"
            if not structure_path.is_file():
                continue
            for _line, row in iter_jsonl(structure_path):
                audio = row.get("audio_path")
                if isinstance(audio, str):
                    clip_id = Path(audio).stem
                    if clip_id in structure:
                        raise CorpusContractError(
                            "hooktheory.structure_duplicate",
                            f"duplicate structure row for {clip_id}",
                        )
                    structure[clip_id] = row
    config = HookTheoryAdapterConfig(
        dataset_name=dataset_id, include_targets=include_targets
    )
    quarantined: list[CorpusQuarantineRecord] = []
    source_rows: list[tuple[str, str]] = []

    def inputs() -> Iterable[CanonicalCorpusInput]:
        for ordinal, (clip_id, record) in enumerate(iter_object_records(raw)):
            if limit is not None and ordinal >= limit:
                break
            relative = f"4_merged.json#{clip_id}"
            row_sha = _fingerprint(_jsonable_source(record))
            source_rows.append((relative, row_sha))
            try:
                piece = convert_hooktheory_record(
                    clip_id,
                    record,
                    config=config,
                    structure_row=structure.get(clip_id),
                    source_path="4_merged.json",
                )
                yield CanonicalCorpusInput(
                    piece=piece,
                    lineage_group_id=piece.source_group_id,
                    source_identity=clip_id,
                    source_relative_path=relative,
                    source_sha256=row_sha,
                    suggested_split=piece.split,
                )
            except HookTheoryAdapterError as exc:
                quarantined.append(
                    CorpusQuarantineRecord(
                        dataset_id=dataset_id,
                        source_identity=clip_id,
                        source_relative_path=relative,
                        category="hooktheory.record_conversion_invalid",
                        message=" ".join(str(exc).split()) or type(exc).__name__,
                    )
                )
    return cache_canonical_corpus(
        inputs(),
        cache_config=cache_config,
        dataset_id=dataset_id,
        adapter_name="hooktheory",
        adapter_version=HOOKTHEORY_CORPUS_ADAPTER_VERSION,
        adapter_config=asdict(config),
        source_identity=raw.name,
        source_fingerprint=lambda: _fingerprint(sorted(source_rows)),
        creation_policy=(
            "offline_full_corpus"
            if limit is None
            else f"offline_bounded_limit:{limit}"
        ),
        quarantine=quarantined,
    )


def build_pop909_cl_corpus_cache(
    root: str | os.PathLike[str],
    *,
    cache_config: CorpusCacheConfig,
    include_targets: bool = True,
    limit: int | None = None,
) -> tuple[CorpusIndex, CorpusBuildReport]:
    """Discover POP909-CL once, then adapt accepted records into the cache."""

    from music_critic.adapters import (
        POP909_CL_ADAPTER_VERSION,
        Pop909ClAdapterConfig,
        Pop909ClQuarantine,
        convert_pop909_cl_file,
        discover_pop909_cl_corpus,
    )

    _validate_builder_limit(limit)
    discovery = discover_pop909_cl_corpus(root)
    config = Pop909ClAdapterConfig(include_targets=include_targets)
    quarantined: list[CorpusQuarantineRecord] = []
    records = discovery.records if limit is None else discovery.records[:limit]

    def inputs() -> Iterable[CanonicalCorpusInput]:
        for record in records:
            result = convert_pop909_cl_file(record, config=config)
            if isinstance(result, Pop909ClQuarantine):
                quarantined.append(
                    CorpusQuarantineRecord(
                        dataset_id=record.dataset_name,
                        source_identity=record.song_id,
                        source_relative_path=record.relative_path,
                        category=result.category,
                        message=result.source_error,
                    )
                )
                continue
            yield CanonicalCorpusInput(
                piece=result.piece,
                lineage_group_id=record.lineage_group_id,
                source_identity=record.song_id,
                source_relative_path=record.relative_path,
                source_sha256=record.sha256,
                suggested_split=None,
            )
    return cache_canonical_corpus(
        inputs(),
        cache_config=cache_config,
        dataset_id=discovery.records[0].dataset_name if discovery.records else "pop909_cl",
        adapter_name="pop909_cl",
        adapter_version=POP909_CL_ADAPTER_VERSION,
        adapter_config=asdict(config),
        source_identity=discovery.corpus_root.name,
        source_fingerprint=discovery.content_fingerprint,
        creation_policy=(
            "offline_full_corpus"
            if limit is None
            else f"offline_bounded_limit:{limit}"
        ),
        quarantine=quarantined,
    )


__all__ = [
    "MULTISOURCE_CACHE_VERSION",
    "MULTISOURCE_CORPUS_INDEX_VERSION",
    "CanonicalCorpusInput",
    "CorpusBuildReport",
    "CorpusCacheConfig",
    "CorpusContractError",
    "CorpusIndex",
    "CorpusIndexHeader",
    "CorpusQuarantineRecord",
    "IndexedCorpusRecord",
    "build_hooktheory_corpus_cache",
    "build_pop909_cl_corpus_cache",
    "cache_canonical_corpus",
    "corpus_cache_key",
    "corpus_index_dict",
    "dump_corpus_index",
    "dumps_corpus_index",
    "load_cached_piece",
    "load_corpus_index",
    "loads_corpus_index",
    "make_corpus_index",
    "validate_current_corpus_index",
]
