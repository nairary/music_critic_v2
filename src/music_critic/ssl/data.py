"""Raw-only data boundary for deterministic Phase 7A SSL."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
from torch_geometric.data import Batch

from music_critic.device import (
    resolve_runtime_device,
)
from music_critic.graph import (
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    GraphBuildError,
    GraphContractError,
    build_raw_graph,
    graph_fingerprint,
    validate_raw_graph_batch,
)
from music_critic.data.validation_membership import (
    fixed_validation_membership,
)
from music_critic.ssl.bounded_fixture import (
    build_phase7a_bounded_fixture,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    CorpusContractError,
    DeterministicQuotaSampler,
    MultiCorpusDataset,
    MultiSourceBatch,
    MultiSourceSample,
    dataset_view_report,
    load_cached_piece,
    load_corpus_index,
    load_split_manifest,
    seed_multisource_worker,
    validate_current_corpus_index,
)
from music_critic.training.data import ValidationMembership


class SSLDataError(ValueError):
    """Raised when the target-free SSL data boundary is violated."""


@dataclass(frozen=True, slots=True)
class SSLRawSample:
    """A raw graph plus the identity sidecar used only for deterministic seeds."""

    raw_graph: Any
    raw_graph_fingerprint: str
    dataset_id: str
    piece_id: str

    def __post_init__(self) -> None:
        _validate_identity(self.dataset_id, name="dataset_id")
        _validate_identity(self.piece_id, name="piece_id")
        if (
            not isinstance(self.raw_graph_fingerprint, str)
            or len(self.raw_graph_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.raw_graph_fingerprint
            )
        ):
            raise SSLDataError(
                "ssl.data.sample_raw_graph_fingerprint_invalid"
            )
        try:
            current = graph_fingerprint(self.raw_graph)
        except GraphContractError as exc:
            raise SSLDataError(
                f"ssl.data.sample_raw_graph_invalid:{exc}"
            ) from exc
        if current != self.raw_graph_fingerprint:
            raise SSLDataError(
                "ssl.data.sample_raw_graph_binding_mismatch"
            )


class IndexedSSLRawDataset:
    """Production cache view that never projects or validates target bundles."""

    def __init__(
        self,
        index: object,
        *,
        cache_config: CorpusCacheConfig,
    ) -> None:
        self.index = (
            load_corpus_index(index)
            if isinstance(index, (str, Path))
            else index
        )
        try:
            validate_current_corpus_index(self.index)
        except CorpusContractError as exc:
            raise SSLDataError(
                f"ssl.data.corpus_index_invalid:{exc}"
            ) from exc
        self.cache_config = cache_config

    @property
    def dataset_id(self) -> str:
        return self.index.header.dataset_id

    def __len__(self) -> int:
        return len(self.index.records)

    def __getitem__(self, index: int) -> SSLRawSample:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self)
        ):
            raise SSLDataError("ssl.data.dataset_index_out_of_range")
        record = self.index.records[index]
        try:
            piece = load_cached_piece(record, self.cache_config)
            graph = build_raw_graph(piece, assume_valid=True)
            return SSLRawSample(
                raw_graph=graph,
                raw_graph_fingerprint=graph_fingerprint(graph),
                dataset_id=record.dataset_id,
                piece_id=record.piece_id,
            )
        except (
            CorpusContractError,
            GraphBuildError,
            GraphContractError,
        ) as exc:
            raise SSLDataError(
                f"ssl.data.raw_sample_load_failed:{exc}"
            ) from exc


def _validate_identity(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SSLDataError(f"ssl.data.{name}_invalid")


def _graph_counts(raw_graph_batch: object) -> tuple[int, int]:
    node_count = sum(
        int(raw_graph_batch[node_type].num_nodes)
        for node_type in MANDATORY_NODE_TYPES
    )
    edge_count = sum(
        int(raw_graph_batch[edge_type].edge_index.shape[1])
        for edge_type in MANDATORY_EDGE_TYPES
    )
    return node_count, edge_count


@dataclass(frozen=True, slots=True)
class SSLBatch:
    """An exact raw graph batch with identity and aggregate counts only."""

    raw_graph_batch: Any
    dataset_ids: tuple[str, ...]
    piece_ids: tuple[str, ...]
    sample_count: int
    node_count: int
    edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_ids, tuple) or not isinstance(
            self.piece_ids, tuple
        ):
            raise SSLDataError("ssl.data.identity_collections_must_be_tuples")
        if (
            len(self.dataset_ids) != len(self.piece_ids)
            or len(self.dataset_ids) == 0
        ):
            raise SSLDataError("ssl.data.identity_lengths_invalid")
        for dataset_id in self.dataset_ids:
            _validate_identity(dataset_id, name="dataset_id")
        for piece_id in self.piece_ids:
            _validate_identity(piece_id, name="piece_id")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count != len(self.dataset_ids)
        ):
            raise SSLDataError("ssl.data.sample_count_invalid")
        try:
            validate_raw_graph_batch(
                self.raw_graph_batch,
                sample_count=self.sample_count,
            )
        except GraphContractError as exc:
            raise SSLDataError(
                f"ssl.data.raw_graph_contract_invalid:{exc}"
            ) from exc
        song_entity_ids = self.raw_graph_batch["song"].entity_id
        if any(len(identifiers) != 1 for identifiers in song_entity_ids):
            raise SSLDataError("ssl.data.song_cardinality_invalid")
        expected_piece_ids = tuple(
            identifiers[0] for identifiers in song_entity_ids
        )
        if expected_piece_ids != self.piece_ids:
            raise SSLDataError("ssl.data.piece_identity_graph_mismatch")
        actual_node_count, actual_edge_count = _graph_counts(
            self.raw_graph_batch
        )
        for name, value, expected in (
            ("node_count", self.node_count, actual_node_count),
            ("edge_count", self.edge_count, actual_edge_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise SSLDataError(f"ssl.data.{name}_invalid")


@dataclass(frozen=True, slots=True)
class SSLDataRuntime:
    """Target-free views over the Phase 6C group-safe data runtime."""

    first_train_batch: SSLBatch
    train_loader: Callable[[int], Iterable[SSLBatch]]
    validation_loader: Callable[[], Iterable[SSLBatch]]
    validation_membership: ValidationMembership
    fingerprints: dict[str, object]
    mixture_statistics: dict[str, object]


def strip_multisource_batch(batch: MultiSourceBatch) -> SSLBatch:
    """Drop every supervised sidecar without reading its contents."""

    if not isinstance(batch, MultiSourceBatch):
        raise SSLDataError("ssl.data.multisource_batch_required")
    node_count, edge_count = _graph_counts(batch.raw_graph_batch)
    return SSLBatch(
        raw_graph_batch=batch.raw_graph_batch,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        sample_count=len(batch.dataset_ids),
        node_count=node_count,
        edge_count=edge_count,
    )


def collate_ssl_samples(
    samples: Sequence[SSLRawSample | MultiSourceSample],
) -> SSLBatch:
    """Collate only immutable raw graphs and identity seed sidecars.

    Harmonic targets, annotations, provenance, diagnostics, and lineage are
    deliberately never accessed by this function.
    """

    if isinstance(samples, (str, bytes)) or not isinstance(
        samples, Sequence
    ):
        raise SSLDataError("ssl.data.samples_must_be_a_sequence")
    prepared = tuple(samples)
    if not prepared:
        raise SSLDataError("ssl.data.samples_empty")
    if not all(
        isinstance(sample, (SSLRawSample, MultiSourceSample))
        for sample in prepared
    ):
        raise SSLDataError("ssl.data.sample_type_invalid")
    for sample in prepared:
        try:
            current = graph_fingerprint(sample.raw_graph)
        except GraphContractError as exc:
            raise SSLDataError(
                f"ssl.data.sample_raw_graph_invalid:{exc}"
            ) from exc
        if current != sample.raw_graph_fingerprint:
            raise SSLDataError("ssl.data.sample_raw_graph_binding_mismatch")
    graph_batch = Batch.from_data_list(
        [sample.raw_graph for sample in prepared]
    )
    node_count, edge_count = _graph_counts(graph_batch)
    return SSLBatch(
        raw_graph_batch=graph_batch,
        dataset_ids=tuple(sample.dataset_id for sample in prepared),
        piece_ids=tuple(sample.piece_id for sample in prepared),
        sample_count=len(prepared),
        node_count=node_count,
        edge_count=edge_count,
    )


def validate_ssl_batch(batch: SSLBatch) -> None:
    """Revalidate the complete raw-only graph and identity/count metadata."""

    if not isinstance(batch, SSLBatch):
        raise SSLDataError("ssl.data.ssl_batch_required")
    batch.__post_init__()


def _membership(value: object) -> ValidationMembership:
    return ValidationMembership(
        identities=value.identities,
        membership_fingerprint=value.membership_fingerprint,
        dataset_counts=value.dataset_counts,
        full_view_count=value.full_view_count,
        selected_count=value.selected_count,
        subset_limit=value.subset_limit,
    )


class _FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _data_loader(
    dataset: object,
    sampler: Sampler[int],
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_ssl_samples,
        worker_init_fn=seed_multisource_worker,
        generator=generator,
        persistent_workers=False,
    )


def _bounded_epoch(
    samples: tuple[SSLRawSample, ...],
    *,
    batch_size: int,
    epoch_size: int,
    seed: int,
    epoch: int,
) -> tuple[SSLBatch, ...]:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    order: list[int] = []
    while len(order) < epoch_size:
        order.extend(
            torch.randperm(len(samples), generator=generator).tolist()
        )
    selected = tuple(samples[index] for index in order[:epoch_size])
    return tuple(
        collate_ssl_samples(selected[start : start + batch_size])
        for start in range(0, len(selected), batch_size)
    )


def _bounded_runtime(config: object, seed: int) -> SSLDataRuntime:
    fixture = build_phase7a_bounded_fixture()
    train = fixture.raw_samples("train")
    validation = fixture.raw_samples("validation")
    selection = fixed_validation_membership(
        tuple(
            (sample.dataset_id, sample.piece_id)
            for sample in validation
        ),
        limit=config.validation_epoch_size,
        seed=(config.validation_seed if config.validation_seed >= 0 else seed),
    )
    membership = _membership(selection)

    def train_loader(epoch: int) -> Iterable[SSLBatch]:
        return _bounded_epoch(
            train,
            batch_size=config.batch_size,
            epoch_size=config.epoch_size,
            seed=seed,
            epoch=epoch,
        )

    def validation_loader() -> Iterable[SSLBatch]:
        selected = tuple(
            validation[index] for index in selection.indices
        )
        return tuple(
            collate_ssl_samples(
                selected[start : start + config.batch_size]
            )
            for start in range(0, len(selected), config.batch_size)
        )

    fingerprints = fixture.fingerprint_bundle()
    fingerprints["validation_membership_fingerprint"] = (
        membership.membership_fingerprint
    )
    return SSLDataRuntime(
        first_train_batch=collate_ssl_samples(
            train[: config.batch_size]
        ),
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_membership=membership,
        fingerprints=fingerprints,
        mixture_statistics={
            "bounded_fixture_contract_version": (
                fixture.contract_version
            ),
            "bounded_fixture_policy": fixture.policy,
            "fixture_counts": fixture.count_summary(),
            "train_counts": fixture.count_summary("train"),
            "validation_counts": fixture.count_summary(
                "validation"
            ),
            "composition": fixture.composition_payload(),
            "requested_weights": dict(config.mixture_weights),
            "train_dataset_counts": dict(
                sorted(
                    Counter(
                        item.dataset_id for item in train
                    ).items()
                )
            ),
            "validation_dataset_counts": dict(
                membership.dataset_counts
            ),
            "validation_membership": asdict(membership),
        },
    )


def _selected_groups(
    dataset: MultiCorpusDataset,
) -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    lineages: set[str] = set()
    for view in dataset.views:
        for index in view.record_indices:
            record = view.dataset.index.records[index]
            sources.add(record.source_group_id)
            lineages.add(record.lineage_group_id)
    return sources, lineages


def _corpus_runtime(config: object, seed: int) -> SSLDataRuntime:
    if (
        not config.index_paths
        or len(config.index_paths) != len(config.cache_roots)
        or not config.split_manifest
    ):
        raise SSLDataError("ssl.data.corpus_paths_incomplete")
    indices = tuple(
        load_corpus_index(path) for path in config.index_paths
    )
    indexed = tuple(
        IndexedSSLRawDataset(
            index,
            cache_config=CorpusCacheConfig(Path(cache_root)),
        )
        for index, cache_root in zip(
            indices, config.cache_roots, strict=True
        )
    )
    manifest = load_split_manifest(config.split_manifest)
    train = MultiCorpusDataset(
        indexed, manifest, split=config.train_split
    )
    validation = MultiCorpusDataset(
        indexed, manifest, split=config.validation_split
    )
    train_sources, train_lineages = _selected_groups(train)
    validation_sources, validation_lineages = _selected_groups(
        validation
    )
    if (
        train_sources & validation_sources
        or train_lineages & validation_lineages
    ):
        raise SSLDataError("ssl.data.split_isolation_failed")
    weights = dict(config.mixture_weights)
    selection = fixed_validation_membership(
        tuple(
            validation.record_identity(index)
            for index in range(len(validation))
        ),
        limit=config.validation_epoch_size,
        seed=(config.validation_seed if config.validation_seed >= 0 else seed),
    )
    membership = _membership(selection)

    def train_loader(epoch: int) -> Iterable[SSLBatch]:
        sampler = DeterministicQuotaSampler(
            train,
            weights=weights,
            seed=seed,
            epoch_size=config.epoch_size,
        )
        sampler.set_epoch(epoch)
        return _data_loader(
            train,
            sampler,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=seed,
        )

    def validation_loader() -> Iterable[SSLBatch]:
        return _data_loader(
            validation,
            _FixedIndexSampler(selection.indices),
            batch_size=config.batch_size,
            workers=config.workers,
            seed=seed + 10_000,
        )

    return SSLDataRuntime(
        first_train_batch=next(iter(train_loader(0))),
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_membership=membership,
        fingerprints={
            "kind": "corpus_cache",
            "index_fingerprints": [
                [index.header.dataset_id, index.header.index_fingerprint]
                for index in sorted(
                    indices,
                    key=lambda item: item.header.dataset_id,
                )
            ],
            "split_manifest_fingerprint": (
                manifest.manifest_fingerprint
            ),
            "train_composition_fingerprint": (
                train.composition_fingerprint
            ),
            "validation_composition_fingerprint": (
                validation.composition_fingerprint
            ),
            "validation_membership_fingerprint": (
                membership.membership_fingerprint
            ),
        },
        mixture_statistics={
            "requested_weights": weights,
            "train": asdict(dataset_view_report(train)),
            "validation": asdict(dataset_view_report(validation)),
            "validation_membership": asdict(membership),
        },
    )


def build_ssl_data_runtime(
    config: object,
    *,
    seed: int,
) -> SSLDataRuntime:
    """Build target-free loaders over Phase 6C paths and split contracts."""

    if config.name == "bounded":
        return _bounded_runtime(config, seed)
    if config.name in {"hooktheory", "pop909_cl", "mixed"}:
        return _corpus_runtime(config, seed)
    raise SSLDataError(f"ssl.data.unknown:{config.name}")


def move_ssl_batch(
    batch: SSLBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> SSLBatch:
    """Deep-copy a batch and move only tensor attributes to ``device``."""

    if not isinstance(batch, SSLBatch):
        raise SSLDataError("ssl.data.ssl_batch_required")
    # Re-run the exact source contract before the unchecked device rebuild.
    validate_ssl_batch(batch)
    target_device = resolve_runtime_device(device)
    graph = copy.deepcopy(batch.raw_graph_batch)
    for store in graph.stores:
        for key, value in tuple(store.items()):
            if isinstance(value, Tensor):
                store[key] = value.to(
                    device=target_device,
                    non_blocking=non_blocking,
                )
    moved = object.__new__(SSLBatch)
    for name, value in (
        ("raw_graph_batch", graph),
        ("dataset_ids", batch.dataset_ids),
        ("piece_ids", batch.piece_ids),
        ("sample_count", batch.sample_count),
        ("node_count", batch.node_count),
        ("edge_count", batch.edge_count),
    ):
        object.__setattr__(moved, name, value)
    _validate_moved_batch(moved, source=batch, device=target_device)
    return moved


def _validate_moved_batch(
    batch: SSLBatch,
    *,
    source: SSLBatch,
    device: torch.device,
) -> None:
    if (
        batch.raw_graph_batch is source.raw_graph_batch
        or batch.dataset_ids != source.dataset_ids
        or batch.piece_ids != source.piece_ids
        or batch.sample_count != source.sample_count
        or batch.node_count != source.node_count
        or batch.edge_count != source.edge_count
    ):
        raise SSLDataError("ssl.data.device_transfer_metadata_changed")
    if (
        tuple(batch.raw_graph_batch.node_types) != MANDATORY_NODE_TYPES
        or tuple(batch.raw_graph_batch.edge_types) != MANDATORY_EDGE_TYPES
        or _graph_counts(batch.raw_graph_batch)
        != (batch.node_count, batch.edge_count)
    ):
        raise SSLDataError("ssl.data.device_transfer_structure_changed")
    for store in batch.raw_graph_batch.stores:
        store_key = getattr(store, "_key", None)
        if store_key is None:
            location_prefix = "global"
        elif isinstance(store_key, str):
            location_prefix = f"node:{store_key}"
        else:
            location_prefix = "edge:" + "|".join(store_key)
        for name, value in store.items():
            if isinstance(value, Tensor):
                _require_ssl_tensor_device(
                    value,
                    device=device,
                    location=f"{location_prefix}:{name}",
                )
    for node_type in MANDATORY_NODE_TYPES:
        moved_store = batch.raw_graph_batch[node_type]
        source_store = source.raw_graph_batch[node_type]
        if int(moved_store.num_nodes) != int(source_store.num_nodes):
            raise SSLDataError(
                f"ssl.data.device_transfer_node_shape_changed:{node_type}"
            )
        for name in (
            "x_cat",
            "x_cat_available",
            "x_cont",
            "x_cont_available",
        ):
            moved_value = moved_store[name]
            source_value = source_store[name]
            if (
                moved_value.shape != source_value.shape
                or moved_value.dtype != source_value.dtype
            ):
                raise SSLDataError(
                    f"ssl.data.device_transfer_node_shape_changed:"
                    f"{node_type}.{name}"
                )
    for edge_type in MANDATORY_EDGE_TYPES:
        moved_edges = batch.raw_graph_batch[edge_type].edge_index
        source_edges = source.raw_graph_batch[edge_type].edge_index
        if (
            moved_edges.shape != source_edges.shape
            or moved_edges.dtype != source_edges.dtype
        ):
            raise SSLDataError(
                "ssl.data.device_transfer_edge_shape_changed:"
                + "|".join(edge_type)
            )


def _require_ssl_tensor_device(
    value: Tensor,
    *,
    device: torch.device,
    location: str,
) -> None:
    if value.device != device:
        raise SSLDataError(
            "ssl.data.device_transfer_tensor_mismatch:"
            f"location={location};expected={device};actual={value.device}"
        )


__all__ = [
    "SSLBatch",
    "SSLDataError",
    "SSLDataRuntime",
    "SSLRawSample",
    "IndexedSSLRawDataset",
    "build_ssl_data_runtime",
    "collate_ssl_samples",
    "move_ssl_batch",
    "strip_multisource_batch",
    "validate_ssl_batch",
]
