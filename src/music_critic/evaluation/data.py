"""Fixed train/evaluation split loading and fingerprint evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from music_critic.evaluation.contracts import (
    EvaluationContractError,
    canonical_fingerprint,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    MultiSourceBatch,
    TARGET_ENCODING_REGISTRY_VERSION,
    TARGET_ONTOLOGY_VERSION,
    collate_multisource_samples,
    create_split_manifest,
    load_corpus_index,
    load_split_manifest,
    ontology_contract_fingerprint,
    prepare_multisource_sample,
    seed_multisource_worker,
    target_encoding_contract_fingerprint,
)
from music_critic.training.data import (
    _bounded_samples,
    _hook_piece,
    _pop_piece,
)


@dataclass(frozen=True, slots=True)
class EvaluationDataRuntime:
    train_loader: Callable[[], Iterable[MultiSourceBatch]]
    evaluation_loader: Callable[[], Iterable[MultiSourceBatch]]
    bindings: dict[str, object]
    train_membership: dict[str, object]
    evaluation_membership: dict[str, object]


class _FixedSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _indices(length: int, limit: int) -> tuple[int, ...]:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
    ):
        raise EvaluationContractError(
            "evaluation.data.sample_limit_invalid"
        )
    return tuple(range(length if limit == 0 else min(length, limit)))


def _fixed_validation_indices(
    identities: Sequence[tuple[str, str]],
    *,
    limit: int,
    seed: int,
) -> tuple[int, ...]:
    """Mirror the Phase 6C fixed-validation membership policy exactly."""

    if limit < 0 or limit > len(identities):
        raise EvaluationContractError(
            "evaluation.data.validation_limit_invalid"
        )
    if limit == 0 or limit == len(identities):
        return tuple(range(len(identities)))
    ranked = sorted(
        range(len(identities)),
        key=lambda index: (
            canonical_fingerprint(
                {
                    "policy": "fixed_validation_membership_v1",
                    "seed": seed,
                    "identity": list(identities[index]),
                }
            ),
            identities[index],
        ),
    )
    return tuple(sorted(ranked[:limit]))


def _membership(
    identities: Sequence[tuple[str, str]],
    indices: Sequence[int],
    *,
    split: str,
    seed: int,
    limit: int,
) -> dict[str, object]:
    selected = [list(identities[index]) for index in indices]
    payload = (
        {
            "policy": "fixed_validation_membership_v1",
            "seed": seed,
            "subset_limit": limit,
            "full_view_count": len(identities),
            "selected_identities": selected,
        }
        if split == "validation"
        else {
            "policy": "canonical_fixed_no_replacement_v1",
            "split": split,
            "full_view_count": len(identities),
            "selected_identities": selected,
        }
    )
    return {
        **payload,
        "selected_count": len(selected),
        "membership_fingerprint": canonical_fingerprint(payload),
    }


def _batches(
    samples: Sequence[Any],
    indices: Sequence[int],
    *,
    batch_size: int,
) -> tuple[MultiSourceBatch, ...]:
    selected = tuple(samples[index] for index in indices)
    return tuple(
        collate_multisource_samples(
            selected[start : start + batch_size]
        )
        for start in range(0, len(selected), batch_size)
    )


def _bounded_test_samples() -> tuple[Any, ...]:
    with TemporaryDirectory(prefix="music-critic-phase6d-test-") as tmp:
        root = Path(tmp)
        pieces = (
            _hook_piece("bounded-test-hook", 4),
            _pop_piece(root, "903", (59, 62, 67)),
        )
        return tuple(prepare_multisource_sample(piece) for piece in pieces)


def _bounded_runtime(
    config: Any, split: str, seed: int
) -> EvaluationDataRuntime:
    train, validation = _bounded_samples()
    evaluation = (
        validation if split == "validation" else _bounded_test_samples()
    )
    train_identities = tuple(
        (sample.dataset_id, sample.piece_id) for sample in train
    )
    evaluation_identities = tuple(
        (sample.dataset_id, sample.piece_id) for sample in evaluation
    )
    train_indices = _indices(len(train), config.max_train_samples)
    evaluation_indices = (
        _fixed_validation_indices(
            evaluation_identities,
            limit=config.max_evaluation_samples,
            seed=seed,
        )
        if split == "validation"
        else _indices(len(evaluation), config.max_evaluation_samples)
    )
    train_membership = _membership(
        train_identities,
        train_indices,
        split="train",
        seed=seed,
        limit=config.max_train_samples,
    )
    evaluation_membership = _membership(
        evaluation_identities,
        evaluation_indices,
        split=split,
        seed=seed,
        limit=config.max_evaluation_samples,
    )
    bounded_split_fingerprint = canonical_fingerprint(
        {
            "train": [list(item) for item in train_identities],
            split: [list(item) for item in evaluation_identities],
        }
    )
    base = {
        "kind": "bounded",
        "split_manifest_fingerprint": bounded_split_fingerprint,
        "effective_split_manifest_fingerprint": (
            bounded_split_fingerprint
        ),
        "index_fingerprints": [
            ["bounded", canonical_fingerprint(
                [list(item) for item in train_identities + evaluation_identities]
            )]
        ],
        "cache_fingerprints": [
            [
                "bounded",
                canonical_fingerprint(
                    [list(item) for item in train_identities]
                ),
            ]
        ],
        "train_composition_fingerprint": canonical_fingerprint(
            [list(item) for item in train_identities]
        ),
        "evaluation_composition_fingerprint": canonical_fingerprint(
            [list(item) for item in evaluation_identities]
        ),
        "ontology_version": TARGET_ONTOLOGY_VERSION,
        "ontology_fingerprint": ontology_contract_fingerprint(),
        "encoding_version": TARGET_ENCODING_REGISTRY_VERSION,
        "encoding_fingerprint": target_encoding_contract_fingerprint(),
        "cache_validation": (
            "synthetic fixtures are built in memory; no production cache read"
        ),
    }
    return EvaluationDataRuntime(
        train_loader=lambda: _batches(
            train, train_indices, batch_size=config.batch_size
        ),
        evaluation_loader=lambda: _batches(
            evaluation,
            evaluation_indices,
            batch_size=config.batch_size,
        ),
        bindings={
            **base,
            "train_membership_fingerprint": train_membership[
                "membership_fingerprint"
            ],
            "evaluation_membership_fingerprint": evaluation_membership[
                "membership_fingerprint"
            ],
        },
        train_membership=train_membership,
        evaluation_membership=evaluation_membership,
    )


def _loader(
    dataset: MultiCorpusDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=_FixedSampler(indices),
        num_workers=workers,
        collate_fn=collate_multisource_samples,
        worker_init_fn=seed_multisource_worker,
        generator=generator,
        persistent_workers=False,
    )


def _groups(dataset: MultiCorpusDataset) -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    lineages: set[str] = set()
    for view in dataset.views:
        for record_index in view.record_indices:
            record = view.dataset.index.records[record_index]
            sources.add(record.source_group_id)
            lineages.add(record.lineage_group_id)
    return sources, lineages


def _corpus_runtime(
    config: Any,
    split: str,
    seed: int,
) -> EvaluationDataRuntime:
    if (
        not config.index_paths
        or len(config.index_paths) != len(config.cache_roots)
        or not config.split_manifest
    ):
        raise EvaluationContractError(
            "evaluation.data.corpus_paths_incomplete"
        )
    indices = tuple(load_corpus_index(path) for path in config.index_paths)
    indexed = tuple(
        IndexedMultiSourceDataset(
            index,
            cache_config=CorpusCacheConfig(Path(cache_root)),
        )
        for index, cache_root in zip(
            indices, config.cache_roots, strict=True
        )
    )
    source_manifest = load_split_manifest(config.split_manifest)
    expected_index_fingerprints = tuple(
        sorted(
            (
                index.header.dataset_id,
                index.header.index_fingerprint,
            )
            for index in indices
        )
    )
    manifest = source_manifest
    if source_manifest.index_fingerprints != expected_index_fingerprints:
        supplied = dict(source_manifest.index_fingerprints)
        if any(
            supplied.get(dataset_id) != fingerprint
            for dataset_id, fingerprint in expected_index_fingerprints
        ):
            raise EvaluationContractError(
                "evaluation.data.split_index_binding_mismatch"
            )
        selected_datasets = {
            dataset_id for dataset_id, _ in expected_index_fingerprints
        }
        split_by_piece = {
            (assignment.dataset_id, assignment.piece_id): assignment.split
            for assignment in source_manifest.assignments
            if assignment.dataset_id in selected_datasets
        }
        manifest = create_split_manifest(
            indices,
            split_by_piece,
            seed=source_manifest.seed,
            policy="evaluation_dataset_subset_v1",
            policy_config={
                "source_manifest_fingerprint": (
                    source_manifest.manifest_fingerprint
                ),
                "selected_datasets": sorted(selected_datasets),
            },
        )
    train = MultiCorpusDataset(indexed, manifest, split=config.train_split)
    evaluation = MultiCorpusDataset(indexed, manifest, split=split)
    train_sources, train_lineages = _groups(train)
    eval_sources, eval_lineages = _groups(evaluation)
    if train_sources & eval_sources or train_lineages & eval_lineages:
        raise EvaluationContractError(
            "evaluation.data.split_isolation_failed"
        )
    train_identities = tuple(
        train.record_identity(index) for index in range(len(train))
    )
    evaluation_identities = tuple(
        evaluation.record_identity(index)
        for index in range(len(evaluation))
    )
    train_indices = _indices(len(train), config.max_train_samples)
    evaluation_indices = (
        _fixed_validation_indices(
            evaluation_identities,
            limit=config.max_evaluation_samples,
            seed=seed,
        )
        if split == "validation"
        else _indices(len(evaluation), config.max_evaluation_samples)
    )
    train_membership = _membership(
        train_identities,
        train_indices,
        split=config.train_split,
        seed=seed,
        limit=config.max_train_samples,
    )
    evaluation_membership = _membership(
        evaluation_identities,
        evaluation_indices,
        split=split,
        seed=seed,
        limit=config.max_evaluation_samples,
    )
    bindings = {
        "kind": "corpus_cache",
        "index_fingerprints": [
            [index.header.dataset_id, index.header.index_fingerprint]
            for index in sorted(
                indices, key=lambda item: item.header.dataset_id
            )
        ],
        "cache_fingerprints": [
            [
                index.header.dataset_id,
                canonical_fingerprint(
                    {
                        "cache_version": index.header.cache_version,
                        "records": [
                            [
                                record.piece_id,
                                record.cache_key,
                                record.canonical_sha256,
                            ]
                            for record in index.records
                        ],
                    }
                ),
            ]
            for index in sorted(
                indices, key=lambda item: item.header.dataset_id
            )
        ],
        "split_manifest_fingerprint": (
            source_manifest.manifest_fingerprint
        ),
        "effective_split_manifest_fingerprint": (
            manifest.manifest_fingerprint
        ),
        "train_composition_fingerprint": train.composition_fingerprint,
        "evaluation_composition_fingerprint": (
            evaluation.composition_fingerprint
        ),
        "train_membership_fingerprint": train_membership[
            "membership_fingerprint"
        ],
        "evaluation_membership_fingerprint": evaluation_membership[
            "membership_fingerprint"
        ],
        "ontology_version": TARGET_ONTOLOGY_VERSION,
        "ontology_fingerprint": ontology_contract_fingerprint(),
        "encoding_version": TARGET_ENCODING_REGISTRY_VERSION,
        "encoding_fingerprint": target_encoding_contract_fingerprint(),
        "cache_validation": (
            "every selected canonical artifact is validated against its "
            "index-bound fingerprint during dataset loading"
        ),
    }
    return EvaluationDataRuntime(
        train_loader=lambda: _loader(
            train,
            train_indices,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=seed,
        ),
        evaluation_loader=lambda: _loader(
            evaluation,
            evaluation_indices,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=seed + 10_000,
        ),
        bindings=bindings,
        train_membership=train_membership,
        evaluation_membership=evaluation_membership,
    )


def build_evaluation_data_runtime(
    config: Any,
    *,
    split: str,
    seed: int,
) -> EvaluationDataRuntime:
    if split not in {"validation", "test"}:
        raise EvaluationContractError(
            f"evaluation.data.split_invalid:{split}"
        )
    if config.name == "bounded":
        return _bounded_runtime(config, split, seed)
    if config.name in {"hooktheory", "pop909_cl", "mixed"}:
        return _corpus_runtime(config, split, seed)
    raise EvaluationContractError(
        f"evaluation.data.unknown:{config.name}"
    )


__all__ = [
    "EvaluationDataRuntime",
    "build_evaluation_data_runtime",
]
