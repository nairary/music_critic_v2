"""Lazy indexed datasets, group-safe splits, deterministic mixtures and loaders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from music_critic.tasks.collator import collate_multisource_samples
from music_critic.tasks.corpus import (
    CorpusCacheConfig,
    CorpusContractError,
    CorpusIndex,
    IndexedCorpusRecord,
    load_cached_piece,
    load_corpus_index,
    validate_current_corpus_index,
)
from music_critic.tasks.multisource import (
    GroupAssignment,
    MultiSourceContractError,
    MultiSourceSample,
    prepare_multisource_sample,
    validate_group_assignments,
)


MIXTURE_SAMPLER_VERSION = "1.0.0"
SPLIT_MANIFEST_VERSION = "1.0.0"
DATASET_VIEW_CONTRACT_VERSION = "1.0.0"


class DatasetContractError(ValueError):
    """Structured Phase 5B.2 dataset/split/sampler failure."""

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
            category = "dataset.worker_propagated"
        self.category = category
        self.dataset_id = dataset_id
        self.piece_id = piece_id
        super().__init__(f"[{category}] {message}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_seed(*values: object) -> int:
    return int(_fingerprint(list(values))[:16], 16)


def _record_key(record: IndexedCorpusRecord) -> tuple[str, str]:
    return record.dataset_id, record.piece_id


def _ordered_indices(indices: Sequence[CorpusIndex]) -> tuple[CorpusIndex, ...]:
    ordered = tuple(sorted(indices, key=lambda index: index.header.dataset_id))
    dataset_ids = tuple(index.header.dataset_id for index in ordered)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise DatasetContractError(
            "split_manifest.duplicate_dataset",
            "constituent indices must have unique dataset IDs",
        )
    return ordered


def _all_records(indices: Sequence[CorpusIndex]) -> tuple[IndexedCorpusRecord, ...]:
    ordered_indices = _ordered_indices(indices)
    records = tuple(
        sorted(
            (record for index in ordered_indices for record in index.records),
            key=_record_key,
        )
    )
    keys = tuple(_record_key(record) for record in records)
    if len(keys) != len(set(keys)):
        raise DatasetContractError(
            "split_manifest.duplicate_piece",
            "constituent indices contain duplicate dataset/piece identities",
        )
    return records


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    dataset_id: str
    piece_id: str
    source_group_id: str
    lineage_group_id: str
    split: str
    component_fingerprint: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.source_group_id,
                self.lineage_group_id,
                self.split,
                self.component_fingerprint,
            )
        ):
            raise DatasetContractError(
                "split_manifest.assignment_invalid",
                "split assignment fields must be non-empty strings",
            )
        if (
            len(self.component_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.component_fingerprint
            )
        ):
            raise DatasetContractError(
                "split_manifest.assignment_invalid",
                "component fingerprint must be lowercase SHA-256",
            )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    version: str
    seed: int
    policy: str
    policy_config_fingerprint: str
    index_fingerprints: tuple[tuple[str, str], ...]
    assignments: tuple[SplitAssignment, ...]
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        if self.version != SPLIT_MANIFEST_VERSION:
            raise DatasetContractError(
                "split_manifest.version_mismatch",
                "unsupported split manifest version",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DatasetContractError(
                "split_manifest.seed_invalid", "split seed must be an integer"
            )
        if not self.policy:
            raise DatasetContractError(
                "split_manifest.policy_invalid", "split policy cannot be empty"
            )
        if not _is_sha256(self.policy_config_fingerprint):
            raise DatasetContractError(
                "split_manifest.fingerprint_invalid",
                "policy config fingerprint must be non-empty lowercase SHA-256",
            )
        if self.index_fingerprints != tuple(sorted(self.index_fingerprints)):
            raise DatasetContractError(
                "split_manifest.order_invalid",
                "index fingerprints must use deterministic dataset order",
            )
        dataset_ids: list[str] = []
        for row in self.index_fingerprints:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or not _is_sha256(row[1])
            ):
                raise DatasetContractError(
                    "split_manifest.fingerprint_invalid",
                    "index fingerprints require dataset IDs and non-empty "
                    "lowercase SHA-256 values",
                )
            dataset_ids.append(row[0])
        if len(dataset_ids) != len(set(dataset_ids)):
            raise DatasetContractError(
                "split_manifest.duplicate_dataset",
                "index fingerprints must not repeat a dataset ID",
            )
        ordered = tuple(
            sorted(self.assignments, key=lambda row: (row.dataset_id, row.piece_id))
        )
        if self.assignments != ordered:
            raise DatasetContractError(
                "split_manifest.order_invalid",
                "assignments must use deterministic dataset/piece order",
            )
        keys = tuple((row.dataset_id, row.piece_id) for row in self.assignments)
        if len(keys) != len(set(keys)):
            raise DatasetContractError(
                "split_manifest.duplicate_piece",
                "a piece must have exactly one split assignment",
            )
        if not _is_sha256(self.manifest_fingerprint):
            raise DatasetContractError(
                "split_manifest.fingerprint_invalid",
                "manifest fingerprint must be non-empty lowercase SHA-256",
            )
        actual = _split_manifest_fingerprint(self, clear=True)
        if self.manifest_fingerprint != actual:
            raise DatasetContractError(
                "split_manifest.fingerprint_mismatch",
                "split manifest fingerprint differs from content",
            )


def _split_manifest_dict(
    manifest: SplitManifest, *, clear: bool = False
) -> dict[str, object]:
    value = asdict(manifest)
    value["index_fingerprints"] = [
        list(item) for item in manifest.index_fingerprints
    ]
    value["assignments"] = [asdict(item) for item in manifest.assignments]
    if clear:
        value["manifest_fingerprint"] = ""
    return value


def _split_manifest_fingerprint(
    manifest: SplitManifest, *, clear: bool = False
) -> str:
    return _fingerprint(_split_manifest_dict(manifest, clear=clear))


def dumps_split_manifest(manifest: SplitManifest) -> str:
    return _canonical_json(_split_manifest_dict(manifest)) + "\n"


def loads_split_manifest(payload: str) -> SplitManifest:
    try:
        value = json.loads(payload)
        return SplitManifest(
            version=value["version"],
            seed=value["seed"],
            policy=value["policy"],
            policy_config_fingerprint=value["policy_config_fingerprint"],
            index_fingerprints=tuple(
                (row[0], row[1]) for row in value["index_fingerprints"]
            ),
            assignments=tuple(
                SplitAssignment(**row) for row in value["assignments"]
            ),
            manifest_fingerprint=value["manifest_fingerprint"],
        )
    except DatasetContractError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetContractError(
            "split_manifest.parse_error", f"invalid split manifest: {exc}"
        ) from exc


def dump_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial")
    temporary.write_text(dumps_split_manifest(manifest), encoding="utf-8")
    temporary.replace(target)


def load_split_manifest(path: str | Path) -> SplitManifest:
    try:
        return loads_split_manifest(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetContractError(
            "split_manifest.unreadable", f"cannot read split manifest: {exc}"
        ) from exc


def _components(
    records: tuple[IndexedCorpusRecord, ...],
) -> tuple[tuple[IndexedCorpusRecord, ...], ...]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_source: dict[str, int] = {}
    by_lineage: dict[str, int] = {}
    for index, record in enumerate(records):
        for value, seen in (
            (record.source_group_id, by_source),
            (record.lineage_group_id, by_lineage),
        ):
            if value in seen:
                union(index, seen[value])
            else:
                seen[value] = index
    grouped: dict[int, list[IndexedCorpusRecord]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return tuple(
        sorted(
            (
                tuple(sorted(component, key=_record_key))
                for component in grouped.values()
            ),
            key=lambda component: tuple(map(_record_key, component)),
        )
    )


def _component_fingerprint(
    component: tuple[IndexedCorpusRecord, ...],
) -> str:
    return _fingerprint(
        [
            {
                "dataset_id": row.dataset_id,
                "piece_id": row.piece_id,
                "source_group_id": row.source_group_id,
                "lineage_group_id": row.lineage_group_id,
            }
            for row in component
        ]
    )


def _make_manifest(
    indices: Sequence[CorpusIndex],
    assignments: tuple[SplitAssignment, ...],
    *,
    seed: int,
    policy: str,
    policy_config: Mapping[str, object],
) -> SplitManifest:
    ordered_indices = _ordered_indices(indices)
    kwargs = {
        "version": SPLIT_MANIFEST_VERSION,
        "seed": seed,
        "policy": policy,
        "policy_config_fingerprint": _fingerprint(dict(policy_config)),
        "index_fingerprints": tuple(
            sorted(
                (
                    index.header.dataset_id,
                    index.header.index_fingerprint,
                )
                for index in ordered_indices
            )
        ),
        "assignments": tuple(
            sorted(assignments, key=lambda row: (row.dataset_id, row.piece_id))
        ),
    }
    core = {
        **kwargs,
        "index_fingerprints": [list(row) for row in kwargs["index_fingerprints"]],
        "assignments": [asdict(row) for row in kwargs["assignments"]],
        "manifest_fingerprint": "",
    }
    return SplitManifest(
        **kwargs,
        manifest_fingerprint=_fingerprint(core),
    )


def create_split_manifest(
    indices: Sequence[CorpusIndex],
    split_by_piece: Mapping[tuple[str, str], str],
    *,
    seed: int,
    policy: str = "explicit",
    policy_config: Mapping[str, object] | None = None,
) -> SplitManifest:
    records = _all_records(indices)
    expected = {_record_key(record) for record in records}
    supplied = set(split_by_piece)
    if supplied != expected:
        missing = tuple(sorted(expected - supplied))
        extra = tuple(sorted(supplied - expected))
        raise DatasetContractError(
            "split_manifest.coverage_mismatch",
            f"assignments must cover every piece exactly once; missing={missing!r}, "
            f"extra={extra!r}",
        )
    groups = tuple(
        GroupAssignment(
            dataset_id=row.dataset_id,
            piece_id=row.piece_id,
            source_group_id=row.source_group_id,
            lineage_group_id=row.lineage_group_id,
            split=split_by_piece[_record_key(row)],
        )
        for row in records
    )
    try:
        validate_group_assignments(groups)
    except MultiSourceContractError as exc:
        raise DatasetContractError(
            "split_manifest.group_leakage", str(exc)
        ) from exc
    component_by_key: dict[tuple[str, str], str] = {}
    for component in _components(records):
        fingerprint = _component_fingerprint(component)
        for row in component:
            component_by_key[_record_key(row)] = fingerprint
    assignments = tuple(
        SplitAssignment(
            dataset_id=row.dataset_id,
            piece_id=row.piece_id,
            source_group_id=row.source_group_id,
            lineage_group_id=row.lineage_group_id,
            split=split_by_piece[_record_key(row)],
            component_fingerprint=component_by_key[_record_key(row)],
        )
        for row in records
    )
    return _make_manifest(
        indices,
        assignments,
        seed=seed,
        policy=policy,
        policy_config=policy_config or {},
    )


def _largest_remainder(
    weights: Mapping[str, float], total: int
) -> tuple[tuple[str, int], ...]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise DatasetContractError(
            "mixture_sampler.epoch_size_invalid",
            "quota total must be a non-negative integer",
        )
    if not weights:
        raise DatasetContractError(
            "mixture_sampler.weights_invalid", "weights cannot be empty"
        )
    normalized: dict[str, Decimal] = {}
    for key, value in weights.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise DatasetContractError(
                "mixture_sampler.weights_invalid",
                "weights require non-empty IDs and finite positive values",
            )
        normalized[key] = Decimal(str(value))
    denominator = sum(normalized.values(), Decimal(0))
    exact = {
        key: Decimal(total) * value / denominator
        for key, value in normalized.items()
    }
    quotas = {
        key: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for key, value in exact.items()
    }
    remaining = total - sum(quotas.values())
    order = sorted(
        normalized,
        key=lambda key: (-(exact[key] - Decimal(quotas[key])), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return tuple(sorted(quotas.items()))


def plan_group_hash_split(
    indices: Sequence[CorpusIndex],
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> SplitManifest:
    """Optional, target-blind, input-order-invariant component split planner."""

    records = _all_records(indices)
    components = _components(records)
    quotas = dict(_largest_remainder(ratios, len(components)))
    ordered = sorted(
        components,
        key=lambda component: (
            _fingerprint(
                {
                    "seed": seed,
                    "component": _component_fingerprint(component),
                }
            ),
            tuple(map(_record_key, component)),
        ),
    )
    split_schedule = tuple(
        split for split in sorted(quotas) for _ in range(quotas[split])
    )
    split_by_piece: dict[tuple[str, str], str] = {}
    for component, split in zip(ordered, split_schedule, strict=True):
        for row in component:
            split_by_piece[_record_key(row)] = split
    return create_split_manifest(
        indices,
        split_by_piece,
        seed=seed,
        policy="deterministic_group_hash",
        policy_config={"ratios": dict(sorted(ratios.items()))},
    )


def validate_split_manifest(
    manifest: SplitManifest, indices: Sequence[CorpusIndex]
) -> None:
    records = _all_records(indices)
    ordered_indices = _ordered_indices(indices)
    expected_indices = tuple(
        (index.header.dataset_id, index.header.index_fingerprint)
        for index in ordered_indices
    )
    if manifest.index_fingerprints != expected_indices:
        raise DatasetContractError(
            "split_manifest.index_mismatch",
            "manifest is bound to different corpus index fingerprints",
        )
    split_by_piece = {
        (row.dataset_id, row.piece_id): row.split for row in manifest.assignments
    }
    expected = {_record_key(row) for row in records}
    if set(split_by_piece) != expected:
        raise DatasetContractError(
            "split_manifest.coverage_mismatch",
            "manifest does not cover the exact indexed pieces",
        )
    groups = tuple(
        GroupAssignment(
            dataset_id=row.dataset_id,
            piece_id=row.piece_id,
            source_group_id=row.source_group_id,
            lineage_group_id=row.lineage_group_id,
            split=split_by_piece[_record_key(row)],
        )
        for row in records
    )
    try:
        validate_group_assignments(groups)
    except MultiSourceContractError as exc:
        raise DatasetContractError(
            "split_manifest.group_leakage", str(exc)
        ) from exc
    expected_components = {
        _record_key(row): _component_fingerprint(component)
        for component in _components(records)
        for row in component
    }
    records_by_key = {_record_key(record): record for record in records}
    if any(
        (
            row.component_fingerprint
            != expected_components[(row.dataset_id, row.piece_id)]
            or row.source_group_id
            != records_by_key[(row.dataset_id, row.piece_id)].source_group_id
            or row.lineage_group_id
            != records_by_key[(row.dataset_id, row.piece_id)].lineage_group_id
        )
        for row in manifest.assignments
    ):
        raise DatasetContractError(
            "split_manifest.component_mismatch",
            "manifest component evidence differs from source/lineage closure",
        )


class IndexedMultiSourceDataset(Dataset[MultiSourceSample]):
    """Map-style dataset that reads exactly one canonical artifact per item."""

    def __init__(
        self,
        index: CorpusIndex | str | Path,
        *,
        cache_config: CorpusCacheConfig,
    ) -> None:
        self.index = (
            load_corpus_index(index)
            if isinstance(index, (str, Path))
            else index
        )
        if not isinstance(self.index, CorpusIndex):
            raise DatasetContractError(
                "dataset.index_invalid", "index must be a CorpusIndex or path"
            )
        validate_current_corpus_index(self.index)
        self.cache_config = cache_config

    @property
    def dataset_id(self) -> str:
        return self.index.header.dataset_id

    def __len__(self) -> int:
        return len(self.index.records)

    def __getitem__(self, index: int) -> MultiSourceSample:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self)
        ):
            raise DatasetContractError(
                "dataset.index_out_of_range",
                f"dataset index must lie in [0, {len(self)})",
                dataset_id=self.dataset_id,
            )
        record = self.index.records[index]
        try:
            piece = load_cached_piece(record, self.cache_config)
            if piece.source_group_id != record.source_group_id:
                raise DatasetContractError(
                    "dataset.source_group_mismatch",
                    "canonical piece source_group_id differs from index sidecar",
                    dataset_id=record.dataset_id,
                    piece_id=record.piece_id,
                )
            sample = prepare_multisource_sample(piece)
            expected_identity = (
                record.dataset_id,
                record.piece_id,
                record.source_group_id,
                record.lineage_group_id,
            )
            actual_identity = (
                sample.dataset_id,
                sample.piece_id,
                sample.source_group_id,
                sample.lineage_group_id,
            )
            if actual_identity != expected_identity:
                raise DatasetContractError(
                    "dataset.prepared_identity_mismatch",
                    "prepared sample identity/lineage differs from index sidecar",
                    dataset_id=record.dataset_id,
                    piece_id=record.piece_id,
                )
            if sample.target_availability != record.target_availability:
                raise DatasetContractError(
                    "dataset.target_availability_mismatch",
                    "recomputed target availability differs from index sidecar",
                    dataset_id=record.dataset_id,
                    piece_id=record.piece_id,
                )
            return sample
        except DatasetContractError:
            raise
        except CorpusContractError as exc:
            raise DatasetContractError(
                exc.category,
                str(exc),
                dataset_id=record.dataset_id,
                piece_id=record.piece_id,
            ) from exc
        except Exception as exc:
            raise DatasetContractError(
                "dataset.sample_preparation_failed",
                f"failed to prepare indexed sample: {exc}",
                dataset_id=record.dataset_id,
                piece_id=record.piece_id,
            ) from exc


_DATASET_VIEW_TOKEN = object()


class DatasetView(Dataset[MultiSourceSample]):
    """One validated split view over an indexed corpus."""

    def __init__(
        self,
        dataset: IndexedMultiSourceDataset,
        manifest: SplitManifest,
        *,
        split: str,
        global_index_fingerprints: tuple[tuple[str, str], ...] | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not _DATASET_VIEW_TOKEN:
            raise DatasetContractError(
                "dataset_view.global_validation_required",
                "dataset views must be derived by MultiCorpusDataset after "
                "global split-manifest validation",
            )
        if not isinstance(split, str) or not split:
            raise DatasetContractError(
                "dataset_view.split_invalid", "split must be a non-empty string"
            )
        by_key = {
            (row.dataset_id, row.piece_id): row.split
            for row in manifest.assignments
        }
        self.dataset = dataset
        self.manifest = manifest
        self.split = split
        assert global_index_fingerprints is not None
        self.global_index_fingerprints = global_index_fingerprints
        self.record_indices = tuple(
            index
            for index, record in enumerate(dataset.index.records)
            if by_key[_record_key(record)] == split
        )
        self.selected_record_identities = tuple(
            _record_key(dataset.index.records[index])
            for index in self.record_indices
        )
        self.view_fingerprint = _fingerprint(
            {
                "dataset_view_contract_version": DATASET_VIEW_CONTRACT_VERSION,
                "sampler_version": MIXTURE_SAMPLER_VERSION,
                "split": split,
                "global_manifest_fingerprint": manifest.manifest_fingerprint,
                "constituent_index_fingerprints": [
                    list(row) for row in global_index_fingerprints
                ],
                "dataset_id": self.dataset_id,
                "corpus_index_fingerprint": self.index_fingerprint,
                "selected_record_identities": [
                    list(row) for row in self.selected_record_identities
                ],
            }
        )

    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    @property
    def index_fingerprint(self) -> str:
        return self.dataset.index.header.index_fingerprint

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getitem__(self, index: int) -> MultiSourceSample:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self)
        ):
            raise DatasetContractError(
                "dataset_view.index_out_of_range",
                f"view index must lie in [0, {len(self)})",
                dataset_id=self.dataset_id,
            )
        return self.dataset[self.record_indices[index]]


class MultiCorpusDataset(Dataset[MultiSourceSample]):
    """One globally validated multi-index manifest projected to one split."""

    def __init__(
        self,
        datasets: Sequence[IndexedMultiSourceDataset],
        manifest: SplitManifest,
        *,
        split: str,
    ) -> None:
        if not datasets:
            raise DatasetContractError(
                "multi_corpus.empty", "at least one indexed dataset is required"
            )
        if not isinstance(split, str) or not split:
            raise DatasetContractError(
                "multi_corpus.split_invalid", "split must be a non-empty string"
            )
        ordered_datasets = tuple(
            sorted(datasets, key=lambda dataset: dataset.dataset_id)
        )
        ids = tuple(dataset.dataset_id for dataset in ordered_datasets)
        if len(ids) != len(set(ids)):
            raise DatasetContractError(
                "multi_corpus.duplicate_dataset",
                "constituent dataset IDs must be unique",
            )
        indices = tuple(dataset.index for dataset in ordered_datasets)
        validate_split_manifest(manifest, indices)
        constituent_fingerprints = tuple(
            (index.header.dataset_id, index.header.index_fingerprint)
            for index in indices
        )
        ordered = tuple(
            DatasetView(
                dataset,
                manifest,
                split=split,
                global_index_fingerprints=constituent_fingerprints,
                _token=_DATASET_VIEW_TOKEN,
            )
            for dataset in ordered_datasets
        )
        self.views = ordered
        self.manifest = manifest
        self.split = split
        starts: list[tuple[str, int, int]] = []
        cursor = 0
        for view in ordered:
            starts.append((view.dataset_id, cursor, cursor + len(view)))
            cursor += len(view)
        self.global_ranges = tuple(starts)
        self.constituent_fingerprints = constituent_fingerprints
        self.manifest_fingerprint = manifest.manifest_fingerprint
        self.view_fingerprints = tuple(
            (view.dataset_id, view.view_fingerprint) for view in ordered
        )
        self.composition_fingerprint = _fingerprint(
            {
                "dataset_view_contract_version": DATASET_VIEW_CONTRACT_VERSION,
                "sampler_version": MIXTURE_SAMPLER_VERSION,
                "split": split,
                "global_manifest_fingerprint": manifest.manifest_fingerprint,
                "constituent_index_fingerprints": [
                    list(row) for row in constituent_fingerprints
                ],
                "views": [
                    {
                        "dataset_id": view.dataset_id,
                        "view_fingerprint": view.view_fingerprint,
                        "selected_record_identities": [
                            list(row)
                            for row in view.selected_record_identities
                        ],
                    }
                    for view in ordered
                ],
            }
        )
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> MultiSourceSample:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self)
        ):
            raise DatasetContractError(
                "multi_corpus.index_out_of_range",
                f"global index must lie in [0, {len(self)})",
            )
        for view, (_, start, end) in zip(
            self.views, self.global_ranges, strict=True
        ):
            if index < end:
                return view[index - start]
        raise AssertionError("validated global index did not resolve")

    def record_identity(self, index: int) -> tuple[str, str]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self)
        ):
            raise DatasetContractError(
                "multi_corpus.index_out_of_range",
                f"global index must lie in [0, {len(self)})",
            )
        for view, (_, start, end) in zip(
            self.views, self.global_ranges, strict=True
        ):
            if index < end:
                return view.selected_record_identities[index - start]
        raise AssertionError("validated global index did not resolve")


@dataclass(frozen=True, slots=True)
class MixtureEpochEvidence:
    sampler_version: str
    dataset_view_contract_version: str
    split: str
    manifest_fingerprint: str
    composition_fingerprint: str
    view_fingerprints: tuple[tuple[str, str], ...]
    epoch: int
    epoch_size: int
    seed: int
    requested_weights: tuple[tuple[str, float], ...]
    normalized_weights: tuple[tuple[str, float], ...]
    quotas: tuple[tuple[str, int], ...]
    constituent_fingerprints: tuple[tuple[str, str], ...]
    repeats_after_cycle_exhaustion: tuple[tuple[str, int], ...]
    schedule_fingerprint: str


class DeterministicQuotaSampler(Sampler[int]):
    """Exact largest-remainder dataset quotas with shuffled local cycles."""

    def __init__(
        self,
        dataset: MultiCorpusDataset,
        *,
        weights: Mapping[str, float],
        seed: int,
        epoch_size: int,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DatasetContractError(
                "mixture_sampler.seed_invalid", "sampler seed must be an integer"
            )
        if (
            isinstance(epoch_size, bool)
            or not isinstance(epoch_size, int)
            or epoch_size <= 0
        ):
            raise DatasetContractError(
                "mixture_sampler.epoch_size_invalid",
                "epoch_size must be a positive integer",
            )
        ranges = {dataset_id: (start, end) for dataset_id, start, end in dataset.global_ranges}
        if set(weights) != set(ranges):
            raise DatasetContractError(
                "mixture_sampler.dataset_mismatch",
                "weights must cover constituent dataset IDs exactly",
            )
        if any(start == end for start, end in ranges.values()):
            raise DatasetContractError(
                "mixture_sampler.empty_dataset",
                "positive-weight constituent datasets cannot be empty",
            )
        self.dataset = dataset
        self.weights = dict(sorted(weights.items()))
        self.seed = seed
        self.epoch_size = epoch_size
        self.quotas = _largest_remainder(self.weights, epoch_size)
        self.epoch = 0
        self._ranges = ranges
        self.last_evidence: MixtureEpochEvidence | None = None

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise DatasetContractError(
                "mixture_sampler.epoch_invalid",
                "epoch must be a non-negative integer",
            )
        self.epoch = epoch

    def _local_indices(self, dataset_id: str, count: int) -> list[int]:
        start, end = self._ranges[dataset_id]
        size = end - start
        result: list[int] = []
        cycle = 0
        while len(result) < count:
            generator = torch.Generator()
            generator.manual_seed(
                _stable_seed(
                    MIXTURE_SAMPLER_VERSION,
                    self.seed,
                    self.epoch,
                    dataset_id,
                    cycle,
                )
            )
            permutation = torch.randperm(size, generator=generator).tolist()
            result.extend(start + index for index in permutation)
            cycle += 1
        return result[:count]

    def __iter__(self) -> Iterator[int]:
        pools = {
            dataset_id: iter(self._local_indices(dataset_id, count))
            for dataset_id, count in self.quotas
        }
        labels = [
            dataset_id
            for dataset_id, count in self.quotas
            for _ in range(count)
        ]
        generator = torch.Generator()
        generator.manual_seed(
            _stable_seed(
                MIXTURE_SAMPLER_VERSION,
                self.seed,
                self.epoch,
                "global_schedule",
            )
        )
        if labels:
            order = torch.randperm(len(labels), generator=generator).tolist()
            labels = [labels[index] for index in order]
        schedule = tuple(next(pools[dataset_id]) for dataset_id in labels)
        resolved_schedule = tuple(
            self.dataset.record_identity(index) for index in schedule
        )
        schedule_fingerprint = _fingerprint(
            {
                "sampler_version": MIXTURE_SAMPLER_VERSION,
                "dataset_view_contract_version": DATASET_VIEW_CONTRACT_VERSION,
                "split": self.dataset.split,
                "manifest_fingerprint": self.dataset.manifest_fingerprint,
                "composition_fingerprint": self.dataset.composition_fingerprint,
                "seed": self.seed,
                "epoch": self.epoch,
                "epoch_size": self.epoch_size,
                "requested_weights": [
                    list(row) for row in self.weights.items()
                ],
                "quotas": [list(row) for row in self.quotas],
                "resolved_piece_schedule": [
                    list(row) for row in resolved_schedule
                ],
            }
        )
        self.last_evidence = MixtureEpochEvidence(
            sampler_version=MIXTURE_SAMPLER_VERSION,
            dataset_view_contract_version=DATASET_VIEW_CONTRACT_VERSION,
            split=self.dataset.split,
            manifest_fingerprint=self.dataset.manifest_fingerprint,
            composition_fingerprint=self.dataset.composition_fingerprint,
            view_fingerprints=self.dataset.view_fingerprints,
            epoch=self.epoch,
            epoch_size=self.epoch_size,
            seed=self.seed,
            requested_weights=tuple(self.weights.items()),
            normalized_weights=tuple(
                (
                    dataset_id,
                    weight / sum(self.weights.values()),
                )
                for dataset_id, weight in self.weights.items()
            ),
            quotas=self.quotas,
            constituent_fingerprints=self.dataset.constituent_fingerprints,
            repeats_after_cycle_exhaustion=tuple(
                (
                    dataset_id,
                    max(0, count - (self._ranges[dataset_id][1] - self._ranges[dataset_id][0])),
                )
                for dataset_id, count in self.quotas
            ),
            schedule_fingerprint=schedule_fingerprint,
        )
        return iter(schedule)


def multisource_worker_seed(worker_id: int) -> int:
    """Return the deterministic PyTorch-assigned seed for one worker."""
    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise DatasetContractError(
            "dataloader.worker_id_invalid",
            "worker_id must be a non-negative integer",
        )
    return torch.initial_seed() % (2**32)


def seed_multisource_worker(worker_id: int) -> None:
    """Top-level spawn-picklable worker initializer for Python and torch."""

    seed = multisource_worker_seed(worker_id)
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True, slots=True)
class MultiSourceDataLoaderConfig:
    batch_size: int
    num_workers: int
    seed: int
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    multiprocessing_context: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise DatasetContractError(
                "dataloader.batch_size_invalid", "batch_size must be positive"
            )
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise DatasetContractError(
                "dataloader.num_workers_invalid",
                "num_workers must be non-negative",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DatasetContractError(
                "dataloader.seed_invalid", "loader seed must be an integer"
            )
        if self.num_workers == 0 and (
            self.persistent_workers or self.prefetch_factor is not None
        ):
            raise DatasetContractError(
                "dataloader.worker_option_invalid",
                "persistent workers and prefetch require num_workers > 0",
            )
        if self.prefetch_factor is not None and (
            isinstance(self.prefetch_factor, bool)
            or not isinstance(self.prefetch_factor, int)
            or self.prefetch_factor <= 0
        ):
            raise DatasetContractError(
                "dataloader.prefetch_invalid",
                "prefetch_factor must be a positive integer",
            )
        if self.multiprocessing_context is not None and self.num_workers == 0:
            raise DatasetContractError(
                "dataloader.context_invalid",
                "multiprocessing context requires num_workers > 0",
            )


def make_multisource_dataloader(
    dataset: MultiCorpusDataset,
    *,
    sampler: DeterministicQuotaSampler,
    config: MultiSourceDataLoaderConfig,
) -> DataLoader[Any]:
    if sampler.dataset is not dataset:
        raise DatasetContractError(
            "dataloader.sampler_mismatch",
            "sampler must be bound to the exact composed dataset",
        )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "collate_fn": collate_multisource_samples,
        "worker_init_fn": seed_multisource_worker,
        "generator": generator,
        "persistent_workers": config.persistent_workers,
    }
    if config.prefetch_factor is not None:
        kwargs["prefetch_factor"] = config.prefetch_factor
    if config.multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = config.multiprocessing_context
    return DataLoader(**kwargs)


@dataclass(frozen=True, slots=True)
class DatasetViewReport:
    split: str
    constituent_counts: tuple[tuple[str, int], ...]
    piece_count: int
    source_group_count: int
    lineage_group_count: int
    component_count: int
    index_fingerprints: tuple[tuple[str, str], ...]
    split_manifest_fingerprints: tuple[tuple[str, str], ...]
    available_target_counts: tuple[tuple[str, int], ...]
    masked_target_counts: tuple[tuple[str, int], ...]


def dataset_view_report(dataset: MultiCorpusDataset) -> DatasetViewReport:
    available: Counter[str] = Counter()
    masked: Counter[str] = Counter()
    counts: list[tuple[str, int]] = []
    source_groups: set[str] = set()
    lineage_groups: set[str] = set()
    components: set[str] = set()
    for view in dataset.views:
        counts.append((view.dataset_id, len(view)))
        assignments = {
            (row.dataset_id, row.piece_id): row
            for row in view.manifest.assignments
        }
        for record_index in view.record_indices:
            record = view.dataset.index.records[record_index]
            source_groups.add(record.source_group_id)
            lineage_groups.add(record.lineage_group_id)
            assignment = assignments[_record_key(record)]
            components.add(assignment.component_fingerprint)
            for row in record.target_availability:
                available[row.task_id] += row.available_count
                masked[row.task_id] += row.masked_count
    return DatasetViewReport(
        split=dataset.split,
        constituent_counts=tuple(counts),
        piece_count=len(dataset),
        source_group_count=len(source_groups),
        lineage_group_count=len(lineage_groups),
        component_count=len(components),
        index_fingerprints=dataset.constituent_fingerprints,
        split_manifest_fingerprints=tuple(
            (view.dataset_id, dataset.manifest_fingerprint)
            for view in dataset.views
        ),
        available_target_counts=tuple(sorted(available.items())),
        masked_target_counts=tuple(sorted(masked.items())),
    )


@dataclass(frozen=True, slots=True)
class DataLoaderBenchmark:
    batch_count: int
    sample_count: int
    num_workers: int
    warm_canonical_read_seconds: float
    graph_preparation_seconds: float
    collation_seconds: float
    elapsed_seconds: float
    samples_per_second: float
    node_count: int
    edge_count: int
    target_row_count: int
    schedule_fingerprint: str


def benchmark_multisource_dataloader(
    loader: DataLoader[Any],
    *,
    max_batches: int,
) -> DataLoaderBenchmark:
    if isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0:
        raise DatasetContractError(
            "dataloader.benchmark_invalid", "max_batches must be positive"
        )
    if not isinstance(loader.dataset, MultiCorpusDataset):
        raise DatasetContractError(
            "dataloader.benchmark_invalid",
            "benchmark requires MultiCorpusDataset",
        )
    batch_size = loader.batch_size
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise DatasetContractError(
            "dataloader.benchmark_invalid",
            "benchmark requires a positive fixed batch size",
        )
    schedule = tuple(iter(loader.sampler))[: max_batches * batch_size]
    sampler_evidence = getattr(loader.sampler, "last_evidence", None)
    if not isinstance(sampler_evidence, MixtureEpochEvidence):
        raise DatasetContractError(
            "dataloader.benchmark_invalid",
            "benchmark sampler did not publish deterministic epoch evidence",
        )

    def record_for_global(
        global_index: int,
    ) -> tuple[IndexedCorpusRecord, CorpusCacheConfig]:
        for view, (_, start, end) in zip(
            loader.dataset.views,
            loader.dataset.global_ranges,
            strict=True,
        ):
            if start <= global_index < end:
                local = global_index - start
                record_index = view.record_indices[local]
                return (
                    view.dataset.index.records[record_index],
                    view.dataset.cache_config,
                )
        raise AssertionError("benchmark schedule index did not resolve")

    record_configs = tuple(record_for_global(index) for index in schedule)
    # Prime the operating-system cache outside the measurement.
    for record, cache_config in record_configs:
        load_cached_piece(record, cache_config)
    read_started = perf_counter()
    pieces = tuple(
        load_cached_piece(record, cache_config)
        for record, cache_config in record_configs
    )
    read_seconds = perf_counter() - read_started
    prepare_started = perf_counter()
    samples = tuple(
        prepare_multisource_sample(
            piece, lineage_group_id=record.lineage_group_id
        )
        for piece, (record, _cache_config) in zip(
            pieces, record_configs, strict=True
        )
    )
    preparation_seconds = perf_counter() - prepare_started
    collate_started = perf_counter()
    _direct_batches = tuple(
        collate_multisource_samples(samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    )
    collation_seconds = perf_counter() - collate_started

    started = perf_counter()
    piece_ids: list[str] = []
    batches = 0
    node_count = 0
    edge_count = 0
    target_row_count = 0
    for batch in loader:
        piece_ids.extend(batch.piece_ids)
        node_count += sum(value for _name, value in batch.statistics.node_counts)
        edge_count += sum(value for _name, value in batch.statistics.edge_counts)
        target_row_count += batch.statistics.target_row_count
        batches += 1
        if batches >= max_batches:
            break
    elapsed = perf_counter() - started
    return DataLoaderBenchmark(
        batch_count=batches,
        sample_count=len(piece_ids),
        num_workers=loader.num_workers,
        warm_canonical_read_seconds=read_seconds,
        graph_preparation_seconds=preparation_seconds,
        collation_seconds=collation_seconds,
        elapsed_seconds=elapsed,
        samples_per_second=(len(piece_ids) / elapsed if elapsed else math.inf),
        node_count=node_count,
        edge_count=edge_count,
        target_row_count=target_row_count,
        schedule_fingerprint=sampler_evidence.schedule_fingerprint,
    )


__all__ = [
    "DATASET_VIEW_CONTRACT_VERSION",
    "MIXTURE_SAMPLER_VERSION",
    "SPLIT_MANIFEST_VERSION",
    "DataLoaderBenchmark",
    "DatasetContractError",
    "DatasetView",
    "DatasetViewReport",
    "DeterministicQuotaSampler",
    "IndexedMultiSourceDataset",
    "MixtureEpochEvidence",
    "MultiCorpusDataset",
    "MultiSourceDataLoaderConfig",
    "SplitAssignment",
    "SplitManifest",
    "benchmark_multisource_dataloader",
    "create_split_manifest",
    "dataset_view_report",
    "dump_split_manifest",
    "dumps_split_manifest",
    "load_split_manifest",
    "loads_split_manifest",
    "make_multisource_dataloader",
    "multisource_worker_seed",
    "plan_group_hash_split",
    "seed_multisource_worker",
    "validate_split_manifest",
]
