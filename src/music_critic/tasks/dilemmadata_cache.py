"""Immutable offline TargetBundle cache for Phase 9B.2B Dilemmadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_critic.adapters.dilemmadata import (
        DilemmadataAccepted,
        DilemmadataCorpusIdentity,
    )
    from music_critic.adapters.dilemmadata_targets import (
        DilemmadataTargetAccepted,
    )
from music_critic.data import dumps_piece
from music_critic.tasks.corpus import (
    CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    CorpusCacheConfig,
    CorpusIndex,
    IndexedCorpusRecord,
    corpus_cache_key,
    load_cached_piece,
    load_corpus_index,
    validate_current_corpus_index,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
    dilemmadata_family_registry_fingerprint,
)
from music_critic.tasks.encoding import (
    dilemmadata_target_encoding_contract_fingerprint,
)
from music_critic.tasks.multisource import (
    TARGET_BUNDLE_CONTRACT_VERSION,
    TargetBundle,
    dumps_target_bundle,
    loads_target_bundle,
    target_bundle_fingerprint,
)


DILEMMADATA_TARGET_CACHE_VERSION = "1.0.0"
DILEMMADATA_TARGET_CACHE_INDEX_VERSION = "1.0.0"
DILEMMADATA_TARGET_CACHE_MANIFEST_VERSION = "1.0.0"
DILEMMADATA_TARGET_CACHE_IDENTITY_VERSION = "1.0.0"
_DILEMMADATA_DATASET_ID = "dilemmadata"


class DilemmadataTargetCacheError(ValueError):
    """Stable failure at the offline target-cache boundary."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        piece_id: str | None = None,
    ) -> None:
        self.category = category
        self.piece_id = piece_id
        super().__init__(f"[{category}] {message}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _portable(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.path_invalid",
            "cache artifact path must be a normalized POSIX relative path",
        )
    return value


@dataclass(frozen=True, slots=True)
class DilemmadataTargetCacheConfig:
    root: Path
    namespace: str = f"dilemmadata-target-cache-v{DILEMMADATA_TARGET_CACHE_VERSION}"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))
        _portable(self.namespace)


@dataclass(frozen=True, slots=True)
class DilemmadataTargetCacheRecord:
    dataset_id: str
    piece_id: str
    source_record_id: str
    raw_index_fingerprint: str
    raw_cache_key: str
    canonical_artifact_sha256: str
    raw_record_binding_sha256: str
    physical_source_sha256: str
    raw_projection_sha256: str
    target_source_fingerprint: str
    metadata_index_fingerprint: str
    raw_alignment_evidence_version: str
    raw_alignment_evidence_fingerprint: str
    target_adapter_version: str
    target_sidecar_version: str
    family_registry_fingerprint: str
    encoding_registry_fingerprint: str
    alignment_registry_fingerprint: str
    target_bundle_contract_version: str
    target_bundle_fingerprint: str
    cache_identity_fingerprint: str
    artifact_relative_path: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.source_record_id,
                self.raw_alignment_evidence_version,
                self.target_adapter_version,
                self.target_sidecar_version,
                self.target_bundle_contract_version,
            )
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.record_invalid",
                "record identity/version fields must be non-empty",
                piece_id=self.piece_id,
            )
        for name in (
            "raw_index_fingerprint",
            "raw_cache_key",
            "canonical_artifact_sha256",
            "raw_record_binding_sha256",
            "physical_source_sha256",
            "raw_projection_sha256",
            "target_source_fingerprint",
            "metadata_index_fingerprint",
            "raw_alignment_evidence_fingerprint",
            "family_registry_fingerprint",
            "encoding_registry_fingerprint",
            "alignment_registry_fingerprint",
            "target_bundle_fingerprint",
            "cache_identity_fingerprint",
            "artifact_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise DilemmadataTargetCacheError(
                    "dilemmadata.target_cache.record_invalid",
                    f"{name} must be lowercase SHA-256",
                    piece_id=self.piece_id,
                )
        _portable(self.artifact_relative_path)
        expected = _cache_identity_fingerprint(self)
        if self.cache_identity_fingerprint != expected:
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.identity_mismatch",
                "cache identity fingerprint differs from record bindings",
                piece_id=self.piece_id,
            )


@dataclass(frozen=True, slots=True)
class DilemmadataTargetCacheIndex:
    index_version: str
    cache_version: str
    dataset_id: str
    raw_index_fingerprint: str
    metadata_index_fingerprint: str
    records: tuple[DilemmadataTargetCacheRecord, ...]
    index_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.index_version != DILEMMADATA_TARGET_CACHE_INDEX_VERSION
            or self.cache_version != DILEMMADATA_TARGET_CACHE_VERSION
            or self.dataset_id != _DILEMMADATA_DATASET_ID
            or not _is_sha256(self.raw_index_fingerprint)
            or not _is_sha256(self.metadata_index_fingerprint)
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.index_incompatible",
                "target-cache index header is incompatible",
            )
        identities = tuple((row.dataset_id, row.piece_id) for row in self.records)
        if identities != tuple(sorted(identities)) or len(identities) != len(
            set(identities)
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.index_order_invalid",
                "target-cache records must be uniquely sorted",
            )
        if any(
            row.dataset_id != self.dataset_id
            or row.raw_index_fingerprint != self.raw_index_fingerprint
            or row.metadata_index_fingerprint != self.metadata_index_fingerprint
            for row in self.records
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.index_binding_mismatch",
                "target-cache record differs from index header",
            )
        if self.index_fingerprint != _target_index_fingerprint(self):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.index_fingerprint_mismatch",
                "target-cache index fingerprint differs from contents",
            )

    def by_identity(self) -> dict[tuple[str, str], DilemmadataTargetCacheRecord]:
        return {(row.dataset_id, row.piece_id): row for row in self.records}


@dataclass(frozen=True, slots=True)
class DilemmadataTargetCacheBuildReport:
    record_count: int
    cache_hit_count: int
    cache_miss_count: int
    index_fingerprint: str
    target_bundle_fingerprint: str


def _alignment_registry_fingerprint() -> str:
    return _fingerprint(
        {
            "alignment_rules_version": DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
            "family_registry_fingerprint": dilemmadata_family_registry_fingerprint(),
        }
    )


def _identity_payload(record: DilemmadataTargetCacheRecord) -> dict[str, object]:
    ignored = {"cache_identity_fingerprint", "artifact_relative_path", "artifact_sha256"}
    return {
        "identity_version": DILEMMADATA_TARGET_CACHE_IDENTITY_VERSION,
        **{
            key: value
            for key, value in asdict(record).items()
            if key not in ignored
        },
    }


def _cache_identity_fingerprint(record: DilemmadataTargetCacheRecord) -> str:
    return _fingerprint(_identity_payload(record))


def _index_payload(
    index: DilemmadataTargetCacheIndex, *, clear: bool = False
) -> dict[str, object]:
    return {
        "cache_version": index.cache_version,
        "dataset_id": index.dataset_id,
        "index_fingerprint": "" if clear else index.index_fingerprint,
        "index_version": index.index_version,
        "metadata_index_fingerprint": index.metadata_index_fingerprint,
        "raw_index_fingerprint": index.raw_index_fingerprint,
        "records": [asdict(row) for row in index.records],
    }


def _target_index_fingerprint(index: DilemmadataTargetCacheIndex) -> str:
    return _fingerprint(_index_payload(index, clear=True))


def _safe_path(config: DilemmadataTargetCacheConfig, relative: str) -> Path:
    _portable(relative)
    root = (config.root / config.namespace).resolve()
    path = (root / PurePosixPath(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.path_escape",
            "target-cache path escapes its namespace",
        ) from exc
    return path


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def dumps_dilemmadata_target_cache_index(
    index: DilemmadataTargetCacheIndex,
) -> str:
    return _canonical_bytes(_index_payload(index)).decode("utf-8") + "\n"


def loads_dilemmadata_target_cache_index(
    payload: str,
) -> DilemmadataTargetCacheIndex:
    try:
        value = json.loads(payload)
        records = tuple(
            DilemmadataTargetCacheRecord(**row) for row in value["records"]
        )
        return DilemmadataTargetCacheIndex(
            **{**value, "records": records}
        )
    except DilemmadataTargetCacheError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.index_parse_invalid",
            f"cannot parse target-cache index: {exc}",
        ) from exc


def dump_dilemmadata_target_cache_index(
    index: DilemmadataTargetCacheIndex, path: str | os.PathLike[str]
) -> None:
    _write_atomic(
        Path(path), dumps_dilemmadata_target_cache_index(index).encode("utf-8")
    )


def load_dilemmadata_target_cache_index(
    path: str | os.PathLike[str],
) -> DilemmadataTargetCacheIndex:
    try:
        return loads_dilemmadata_target_cache_index(
            Path(path).read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.index_unreadable",
            f"cannot read target-cache index: {exc}",
        ) from exc


def _new_record(
    *,
    raw_index: CorpusIndex,
    raw_record: IndexedCorpusRecord,
    accepted: DilemmadataAccepted,
    target: DilemmadataTargetAccepted,
    metadata_fingerprint: str,
) -> DilemmadataTargetCacheRecord:
    from music_critic.adapters.dilemmadata_targets import (
        DILEMMADATA_TARGET_ADAPTER_VERSION,
        DILEMMADATA_TARGET_SIDECAR_VERSION,
    )

    values = dict(
        dataset_id=raw_record.dataset_id,
        piece_id=raw_record.piece_id,
        source_record_id=accepted.record.record_id,
        raw_index_fingerprint=raw_index.header.index_fingerprint,
        raw_cache_key=raw_record.cache_key,
        canonical_artifact_sha256=raw_record.canonical_sha256,
        raw_record_binding_sha256=accepted.record.record_binding_sha256,
        physical_source_sha256=accepted.record.physical_source_sha256,
        raw_projection_sha256=accepted.record.raw_projection_sha256,
        target_source_fingerprint=target.statistics.target_source_sha256,
        metadata_index_fingerprint=metadata_fingerprint,
        raw_alignment_evidence_version=accepted.alignment_evidence.version,
        raw_alignment_evidence_fingerprint=accepted.alignment_evidence.fingerprint,
        target_adapter_version=DILEMMADATA_TARGET_ADAPTER_VERSION,
        target_sidecar_version=DILEMMADATA_TARGET_SIDECAR_VERSION,
        family_registry_fingerprint=dilemmadata_family_registry_fingerprint(),
        encoding_registry_fingerprint=(
            dilemmadata_target_encoding_contract_fingerprint()
        ),
        alignment_registry_fingerprint=_alignment_registry_fingerprint(),
        target_bundle_contract_version=TARGET_BUNDLE_CONTRACT_VERSION,
        target_bundle_fingerprint=target.sidecar_fingerprint,
        cache_identity_fingerprint="",
        artifact_relative_path="placeholder.json",
        artifact_sha256="0" * 64,
    )
    provisional = DilemmadataTargetCacheRecord.__new__(
        DilemmadataTargetCacheRecord
    )
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    identity = _cache_identity_fingerprint(provisional)
    payload = dumps_target_bundle(target.target_bundle).encode("utf-8")
    payload_sha = sha256(payload).hexdigest()
    artifact_identity = _fingerprint(
        {
            "cache_identity_fingerprint": identity,
            "target_bundle_fingerprint": target.sidecar_fingerprint,
            "artifact_sha256": payload_sha,
        }
    )
    values.update(
        cache_identity_fingerprint=identity,
        artifact_relative_path=(
            f"artifacts/{artifact_identity[:2]}/{artifact_identity}.target.json"
        ),
        artifact_sha256=payload_sha,
    )
    return DilemmadataTargetCacheRecord(**values)


def build_dilemmadata_target_cache(
    source_root: str | os.PathLike[str],
    *,
    raw_index: CorpusIndex | str | os.PathLike[str],
    raw_cache_config: CorpusCacheConfig,
    target_cache_config: DilemmadataTargetCacheConfig,
    identity: DilemmadataCorpusIdentity | None = None,
    limit: int | None = None,
) -> tuple[DilemmadataTargetCacheIndex, DilemmadataTargetCacheBuildReport]:
    """Build sidecars only after verifying the exact immutable raw cache."""

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.limit_invalid",
            "limit must be null or a positive integer",
        )
    from music_critic.adapters.dilemmadata import (
        DILEMMADATA_ADAPTER_VERSION,
        DilemmadataAccepted,
        DilemmadataCorpusIdentity,
        convert_dilemmadata_record,
        discover_dilemmadata_corpus,
    )
    from music_critic.adapters.dilemmadata_targets import (
        DilemmadataTargetAccepted,
        build_dilemmadata_target_sidecar,
        load_dilemmadata_target_metadata_index,
    )

    raw = load_corpus_index(raw_index) if not isinstance(raw_index, CorpusIndex) else raw_index
    validate_current_corpus_index(raw)
    if (
        raw.header.dataset_id != _DILEMMADATA_DATASET_ID
        or raw.header.adapter_name != "dilemmadata"
        or raw.header.adapter_version != DILEMMADATA_ADAPTER_VERSION
    ):
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.raw_index_incompatible",
            "target cache requires the current Dilemmadata raw index",
        )
    discovery = discover_dilemmadata_corpus(
        source_root,
        identity=(DilemmadataCorpusIdentity() if identity is None else identity),
        require_valid=True,
    )
    discovered = {record.record_id: record for record in discovery.records}
    metadata = load_dilemmadata_target_metadata_index(
        discovery.root, discovery.records
    )
    raw_records = raw.records if limit is None else raw.records[:limit]
    cache_records: list[DilemmadataTargetCacheRecord] = []
    hit_count = 0
    miss_count = 0
    bundle_fingerprints: list[tuple[str, str]] = []
    for raw_record in raw_records:
        record = discovered.get(raw_record.source_identity)
        if record is None or record.piece_id != raw_record.piece_id:
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.raw_record_missing",
                "raw index record has no exact discovered source record",
                piece_id=raw_record.piece_id,
            )
        expected_raw_key = corpus_cache_key(
            source_identity=record.record_id,
            source_sha256=record.physical_source_sha256,
            adapter_name="dilemmadata",
            adapter_version=DILEMMADATA_ADAPTER_VERSION,
            adapter_config_fingerprint=raw.header.adapter_config_fingerprint,
            cache_input_sha256=record.raw_projection_sha256,
        )
        if expected_raw_key != raw_record.cache_key:
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.raw_projection_stale",
                "discovered raw projection differs from raw cache identity",
                piece_id=raw_record.piece_id,
            )
        cached_piece = load_cached_piece(raw_record, raw_cache_config)
        outcome = convert_dilemmadata_record(record)
        if not isinstance(outcome, DilemmadataAccepted):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.raw_reconstruction_failed",
                "accepted raw index record no longer converts",
                piece_id=raw_record.piece_id,
            )
        if (
            dumps_piece(outcome.piece) != dumps_piece(cached_piece)
            or sha256(dumps_piece(cached_piece).encode("utf-8")).hexdigest()
            != raw_record.canonical_sha256
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.canonical_binding_mismatch",
                "independently reconstructed canonical piece differs from raw cache",
                piece_id=raw_record.piece_id,
            )
        target = build_dilemmadata_target_sidecar(
            outcome, metadata_index=metadata
        )
        if not isinstance(target, DilemmadataTargetAccepted):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.target_conversion_failed",
                "target conversion did not produce an accepted bundle: "
                + ",".join(target.categories),
                piece_id=raw_record.piece_id,
            )
        cache_record = _new_record(
            raw_index=raw,
            raw_record=raw_record,
            accepted=outcome,
            target=target,
            metadata_fingerprint=metadata.fingerprint,
        )
        artifact = _safe_path(target_cache_config, cache_record.artifact_relative_path)
        payload = dumps_target_bundle(target.target_bundle).encode("utf-8")
        if artifact.exists():
            existing = artifact.read_bytes()
            if sha256(existing).hexdigest() != cache_record.artifact_sha256:
                raise DilemmadataTargetCacheError(
                    "dilemmadata.target_cache.existing_artifact_corrupt",
                    "existing immutable artifact differs from expected bytes",
                    piece_id=raw_record.piece_id,
                )
            decoded = loads_target_bundle(existing.decode("utf-8"))
            if target_bundle_fingerprint(decoded) != cache_record.target_bundle_fingerprint:
                raise DilemmadataTargetCacheError(
                    "dilemmadata.target_cache.existing_artifact_forged",
                    "existing target bundle fingerprint differs from index identity",
                    piece_id=raw_record.piece_id,
                )
            hit_count += 1
        else:
            _write_atomic(artifact, payload)
            miss_count += 1
        cache_records.append(cache_record)
        bundle_fingerprints.append((raw_record.piece_id, target.sidecar_fingerprint))
    ordered_records = tuple(
        sorted(cache_records, key=lambda row: (row.dataset_id, row.piece_id))
    )
    header_values = {
        "cache_version": DILEMMADATA_TARGET_CACHE_VERSION,
        "dataset_id": _DILEMMADATA_DATASET_ID,
        "index_fingerprint": "",
        "index_version": DILEMMADATA_TARGET_CACHE_INDEX_VERSION,
        "metadata_index_fingerprint": metadata.fingerprint,
        "raw_index_fingerprint": raw.header.index_fingerprint,
        "records": [asdict(row) for row in ordered_records],
    }
    index = DilemmadataTargetCacheIndex(
        index_version=DILEMMADATA_TARGET_CACHE_INDEX_VERSION,
        cache_version=DILEMMADATA_TARGET_CACHE_VERSION,
        dataset_id=_DILEMMADATA_DATASET_ID,
        raw_index_fingerprint=raw.header.index_fingerprint,
        metadata_index_fingerprint=metadata.fingerprint,
        records=ordered_records,
        index_fingerprint=_fingerprint(header_values),
    )
    return index, DilemmadataTargetCacheBuildReport(
        record_count=len(index.records),
        cache_hit_count=hit_count,
        cache_miss_count=miss_count,
        index_fingerprint=index.index_fingerprint,
        target_bundle_fingerprint=_fingerprint(bundle_fingerprints),
    )


def validate_dilemmadata_target_cache_index(
    index: DilemmadataTargetCacheIndex,
    raw_index: CorpusIndex,
) -> None:
    from music_critic.adapters.dilemmadata import (
        DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION,
    )
    from music_critic.adapters.dilemmadata_targets import (
        DILEMMADATA_TARGET_ADAPTER_VERSION,
        DILEMMADATA_TARGET_SIDECAR_VERSION,
    )

    validate_current_corpus_index(raw_index)
    if index.raw_index_fingerprint != raw_index.header.index_fingerprint:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.raw_index_stale",
            "target-cache index is bound to a different raw index",
        )
    raw_by_id = {(row.dataset_id, row.piece_id): row for row in raw_index.records}
    if set(index.by_identity()) != set(raw_by_id):
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.coverage_mismatch",
            "target-cache index does not cover the exact raw index",
        )
    for identity, row in index.by_identity().items():
        raw = raw_by_id[identity]
        if (
            row.raw_cache_key != raw.cache_key
            or row.canonical_artifact_sha256 != raw.canonical_sha256
            or row.target_adapter_version != DILEMMADATA_TARGET_ADAPTER_VERSION
            or row.target_sidecar_version != DILEMMADATA_TARGET_SIDECAR_VERSION
            or row.raw_alignment_evidence_version
            != DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION
            or row.family_registry_fingerprint
            != dilemmadata_family_registry_fingerprint()
            or row.encoding_registry_fingerprint
            != dilemmadata_target_encoding_contract_fingerprint()
            or row.alignment_registry_fingerprint != _alignment_registry_fingerprint()
            or row.target_bundle_contract_version != TARGET_BUNDLE_CONTRACT_VERSION
        ):
            raise DilemmadataTargetCacheError(
                "dilemmadata.target_cache.contract_stale",
                "target-cache record bindings differ from current contracts",
                piece_id=row.piece_id,
            )


def load_dilemmadata_target_bundle(
    record: DilemmadataTargetCacheRecord,
    config: DilemmadataTargetCacheConfig,
) -> TargetBundle:
    path = _safe_path(config, record.artifact_relative_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.artifact_unreadable",
            f"cannot read target bundle artifact: {exc}",
            piece_id=record.piece_id,
        ) from exc
    if sha256(payload).hexdigest() != record.artifact_sha256:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.artifact_fingerprint_mismatch",
            "target bundle artifact SHA-256 differs from index",
            piece_id=record.piece_id,
        )
    try:
        bundle = loads_target_bundle(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.artifact_invalid",
            f"target bundle artifact is invalid: {exc}",
            piece_id=record.piece_id,
        ) from exc
    if (
        bundle.dataset_id != record.dataset_id
        or bundle.piece_id != record.piece_id
        or target_bundle_fingerprint(bundle) != record.target_bundle_fingerprint
    ):
        raise DilemmadataTargetCacheError(
            "dilemmadata.target_cache.artifact_binding_mismatch",
            "target bundle artifact identity/fingerprint differs from index",
            piece_id=record.piece_id,
        )
    return bundle


def check_dilemmadata_target_cache(
    index: DilemmadataTargetCacheIndex,
    *,
    raw_index: CorpusIndex,
    cache_config: DilemmadataTargetCacheConfig,
) -> dict[str, object]:
    validate_dilemmadata_target_cache_index(index, raw_index)
    bundles = tuple(
        load_dilemmadata_target_bundle(record, cache_config)
        for record in index.records
    )
    return {
        "ready": True,
        "record_count": len(bundles),
        "index_fingerprint": index.index_fingerprint,
        "raw_index_fingerprint": index.raw_index_fingerprint,
        "target_bundle_fingerprint": _fingerprint(
            [
                [bundle.piece_id, target_bundle_fingerprint(bundle)]
                for bundle in bundles
            ]
        ),
    }


def dilemmadata_target_cache_manifest(
    index: DilemmadataTargetCacheIndex,
    report: DilemmadataTargetCacheBuildReport,
) -> dict[str, object]:
    from music_critic.adapters.dilemmadata import (
        DILEMMADATA_ADAPTER_VERSION,
        DILEMMADATA_RAW_PROJECTION_VERSION,
        DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION,
    )
    from music_critic.adapters.dilemmadata_targets import (
        DILEMMADATA_TARGET_ADAPTER_VERSION,
        DILEMMADATA_TARGET_SIDECAR_VERSION,
    )

    return {
        "manifest_version": DILEMMADATA_TARGET_CACHE_MANIFEST_VERSION,
        "cache_version": DILEMMADATA_TARGET_CACHE_VERSION,
        "index_version": DILEMMADATA_TARGET_CACHE_INDEX_VERSION,
        "dataset_id": index.dataset_id,
        "record_count": report.record_count,
        "raw_index_fingerprint": index.raw_index_fingerprint,
        "target_cache_index_fingerprint": index.index_fingerprint,
        "metadata_index_fingerprint": index.metadata_index_fingerprint,
        "target_bundle_fingerprint": report.target_bundle_fingerprint,
        "target_adapter_version": DILEMMADATA_TARGET_ADAPTER_VERSION,
        "target_sidecar_version": DILEMMADATA_TARGET_SIDECAR_VERSION,
        "raw_alignment_evidence_version": (
            DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION
        ),
        "family_registry_fingerprint": dilemmadata_family_registry_fingerprint(),
        "encoding_registry_fingerprint": (
            dilemmadata_target_encoding_contract_fingerprint()
        ),
        "alignment_registry_fingerprint": _alignment_registry_fingerprint(),
        "target_bundle_contract_version": TARGET_BUNDLE_CONTRACT_VERSION,
        "raw_contracts_unchanged": {
            "adapter_version": DILEMMADATA_ADAPTER_VERSION,
            "raw_projection_version": DILEMMADATA_RAW_PROJECTION_VERSION,
            "cache_input_identity_version": CORPUS_CACHE_INPUT_IDENTITY_VERSION,
        },
    }


__all__ = [
    "DILEMMADATA_TARGET_CACHE_IDENTITY_VERSION",
    "DILEMMADATA_TARGET_CACHE_INDEX_VERSION",
    "DILEMMADATA_TARGET_CACHE_MANIFEST_VERSION",
    "DILEMMADATA_TARGET_CACHE_VERSION",
    "DilemmadataTargetCacheBuildReport",
    "DilemmadataTargetCacheConfig",
    "DilemmadataTargetCacheError",
    "DilemmadataTargetCacheIndex",
    "DilemmadataTargetCacheRecord",
    "build_dilemmadata_target_cache",
    "check_dilemmadata_target_cache",
    "dilemmadata_target_cache_manifest",
    "dump_dilemmadata_target_cache_index",
    "dumps_dilemmadata_target_cache_index",
    "load_dilemmadata_target_bundle",
    "load_dilemmadata_target_cache_index",
    "loads_dilemmadata_target_cache_index",
    "validate_dilemmadata_target_cache_index",
]
