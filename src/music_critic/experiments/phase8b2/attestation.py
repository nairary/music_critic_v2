"""Official data attestation and target-free sample-schedule resolution."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from music_critic.evaluation.config import EvaluationDataConfig
from music_critic.evaluation.data import build_evaluation_data_runtime
from music_critic.experiments.phase8b2.contracts import (
    DataBinding,
    Phase8B2ContractError,
    fingerprint,
)
from music_critic.experiments.phase8b2.schedule import (
    SCHEDULE_CONTRACT_VERSION,
    SeedDomains,
)
from music_critic.ssl.data import IndexedSSLRawDataset, build_ssl_data_runtime
from music_critic.tasks import (
    CorpusCacheConfig,
    DeterministicQuotaSampler,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    load_corpus_index,
    load_split_manifest,
)
from music_critic.training.config import DataConfig
from music_critic.training.data import build_data_runtime


DATA_ATTESTATION_CONTRACT_VERSION = "1.1.0"
ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION = "1.1.0"


def _production_metadata_schedule(
    data: Mapping[str, Any],
    *,
    seed: int,
    logical_updates: int,
    optimizer_steps_per_epoch: int,
    raw_only: bool,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    """Use the official quota sampler without materializing cached pieces."""

    indices = tuple(load_corpus_index(path) for path in data["index_paths"])
    dataset_type = IndexedSSLRawDataset if raw_only else IndexedMultiSourceDataset
    datasets = tuple(
        dataset_type(
            index,
            cache_config=CorpusCacheConfig(Path(cache_root)),
        )
        for index, cache_root in zip(
            indices, data["cache_roots"], strict=True
        )
    )
    manifest = load_split_manifest(data["split_manifest"])
    train = MultiCorpusDataset(datasets, manifest, split="train")
    batch_size = int(data["batch_size"])
    slots: list[dict[str, object]] = []
    identities: list[tuple[str, str]] = []
    update = 0
    epoch = 0
    while update < logical_updates:
        sampler = DeterministicQuotaSampler(
            train,
            weights=dict(data["mixture_weights"]),
            seed=seed,
            epoch_size=batch_size * optimizer_steps_per_epoch,
        )
        sampler.set_epoch(epoch)
        order = list(iter(sampler))
        for start in range(0, len(order), batch_size):
            if update >= logical_updates:
                break
            batch_indices = order[start : start + batch_size]
            if len(batch_indices) != batch_size:
                raise Phase8B2ContractError(
                    "phase8b2.schedule.partial_metadata_batch_forbidden"
                )
            for batch_position, index in enumerate(batch_indices):
                dataset_id, piece_id = train.record_identity(index)
                identities.append((dataset_id, piece_id))
                slots.append(
                    {
                        "dataset_id": dataset_id,
                        "piece_id": piece_id,
                        "sample_position": len(slots),
                        "logical_update": update,
                        "batch_position": batch_position,
                    }
                )
            update += 1
        epoch += 1
    return slots, identities


def _expected_matches(
    expected: object, actual: object, *, category: str
) -> None:
    if expected in (None, "", {}, []):
        return
    if expected != actual:
        raise Phase8B2ContractError(
            f"phase8b2.data_attestation.{category}_mismatch"
        )


def _evaluation_data_config(
    data: Mapping[str, Any],
    *,
    validation_samples: int,
    validation_seed: int,
) -> EvaluationDataConfig:
    production = bool(data["index_paths"])
    return EvaluationDataConfig(
        name="mixed" if production else "bounded",
        index_paths=list(data["index_paths"]),
        cache_roots=list(data["cache_roots"]),
        split_manifest=str(data["split_manifest"]),
        batch_size=int(data["batch_size"]),
        workers=int(data["workers"]),
        max_train_samples=0,
        max_evaluation_samples=validation_samples,
        validation_seed=validation_seed,
    )


def _verify_production_path_bindings(data: Mapping[str, Any]) -> None:
    index_paths = tuple(str(path) for path in data["index_paths"])
    cache_roots = tuple(str(path) for path in data["cache_roots"])
    if not index_paths and not cache_roots:
        return
    if (
        not index_paths
        or len(index_paths) != len(cache_roots)
        or not data["split_manifest"]
        or any(not Path(path).is_absolute() for path in index_paths)
        or any(not Path(path).is_absolute() for path in cache_roots)
        or not Path(str(data["split_manifest"])).is_absolute()
    ):
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.production_paths_incomplete"
        )
    dataset_ids: list[str] = []
    for index_path, cache_root in zip(index_paths, cache_roots, strict=True):
        index = load_corpus_index(index_path)
        dataset_ids.append(index.header.dataset_id)
        if index.records:
            record = index.records[0]
            candidate = (
                Path(cache_root)
                / CorpusCacheConfig(Path(cache_root)).namespace
                / record.canonical_relative_path
            )
            if (
                not candidate.is_file()
                or sha256(candidate.read_bytes()).hexdigest()
                != record.canonical_sha256
            ):
                raise Phase8B2ContractError(
                    "phase8b2.data_attestation.index_cache_path_mismatch:"
                    f"{index.header.dataset_id}"
                )
    if len(dataset_ids) != len(set(dataset_ids)):
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.duplicate_dataset_id"
        )


def attest_data_binding(
    data: Mapping[str, Any],
    *,
    validation_samples: int,
    validation_seed: int,
) -> tuple[DataBinding, dict[str, object]]:
    """Derive every identity from official metadata, never caller SHA claims."""

    _verify_production_path_bindings(data)
    config = _evaluation_data_config(
        data,
        validation_samples=validation_samples,
        validation_seed=validation_seed,
    )
    validation = build_evaluation_data_runtime(
        config, split="validation", seed=validation_seed
    )
    test_config = _evaluation_data_config(
        data,
        validation_samples=0,
        validation_seed=validation_seed,
    )
    test = build_evaluation_data_runtime(
        test_config, split="test", seed=validation_seed
    )
    bindings = validation.bindings
    if config.name == "bounded":
        train_ids = tuple(
            tuple(row)
            for row in validation.train_membership["selected_identities"]
        )
        validation_ids = tuple(
            tuple(row)
            for row in validation.evaluation_membership[
                "selected_identities"
            ]
        )
        test_ids = tuple(
            tuple(row)
            for row in test.evaluation_membership["selected_identities"]
        )
        by_dataset: dict[str, list[list[str]]] = {}
        for dataset_id, piece_id in (*train_ids, *validation_ids, *test_ids):
            by_dataset.setdefault(dataset_id, []).append(
                [dataset_id, piece_id]
            )
        index_rows = tuple(
            (dataset_id, fingerprint(sorted(rows)))
            for dataset_id, rows in sorted(by_dataset.items())
        )
        cache_rows = tuple(
            (dataset_id, fingerprint({"bounded_cache": sorted(rows)}))
            for dataset_id, rows in sorted(by_dataset.items())
        )
    else:
        index_rows = tuple(
            (str(dataset_id), str(value))
            for dataset_id, value in bindings["index_fingerprints"]
        )
        cache_rows = tuple(
            (str(dataset_id), str(value))
            for dataset_id, value in bindings["cache_fingerprints"]
        )
    actual_index = dict(index_rows)
    actual_cache = dict(cache_rows)
    _expected_matches(
        dict(data["index_fingerprints"]),
        actual_index,
        category="index_fingerprints",
    )
    _expected_matches(
        dict(data["cache_fingerprints"]),
        actual_cache,
        category="cache_fingerprints",
    )
    split_fingerprint = str(bindings["split_manifest_fingerprint"])
    train_membership = str(bindings["train_membership_fingerprint"])
    validation_membership = str(
        bindings["evaluation_membership_fingerprint"]
    )
    test_membership = str(
        test.bindings["evaluation_membership_fingerprint"]
    )
    for field, actual in (
        ("split_manifest_fingerprint", split_fingerprint),
        ("train_membership_fingerprint", train_membership),
        ("validation_membership_fingerprint", validation_membership),
        ("test_membership_fingerprint", test_membership),
    ):
        _expected_matches(data[field], actual, category=field)
    weights = tuple(
        sorted(
            (str(dataset_id), float(weight))
            for dataset_id, weight in data["mixture_weights"].items()
        )
    )
    if set(dict(weights)) != set(actual_index):
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.mixture_dataset_mismatch"
        )
    binding = DataBinding(
        dataset_indices=index_rows,
        cache_identities=cache_rows,
        split_manifest_fingerprint=split_fingerprint,
        train_membership_fingerprint=train_membership,
        validation_membership_fingerprint=validation_membership,
        test_membership_fingerprint=test_membership,
        mixture_weights=weights,
        workers=int(data["workers"]),
        actual_train_size=int(validation.train_membership["selected_count"]),
        actual_validation_size=int(
            validation.evaluation_membership["selected_count"]
        ),
        actual_test_size=int(test.evaluation_membership["selected_count"]),
        validation_subset_limit=validation_samples,
        fixed_validation_seed=validation_seed,
    )
    evidence = {
        "data_attestation_contract_version": (
            DATA_ATTESTATION_CONTRACT_VERSION
        ),
        "source": "official_evaluation_data_runtime_metadata",
        "data_binding": binding.to_dict(),
        "train_membership": validation.train_membership,
        "validation_membership": validation.evaluation_membership,
        "test_membership": test.evaluation_membership,
        "actual_train_size": validation.train_membership["selected_count"],
        "actual_validation_size": validation.evaluation_membership[
            "selected_count"
        ],
        "actual_test_size": test.evaluation_membership["selected_count"],
        "targets_read_for_schedule": False,
    }
    evidence["fingerprint"] = fingerprint(evidence)
    return binding, evidence


def resolve_actual_ssl_schedule(
    data: Mapping[str, Any],
    *,
    seed: int,
    logical_updates: int,
    optimizer_steps_per_epoch: int,
    validation_samples: int,
    validation_seed: int,
) -> dict[str, object]:
    """Run the official raw-only sampler and record exact update positions."""

    domains = SeedDomains.create(seed)
    batch_size = int(data["batch_size"])
    config = DataConfig(
        name="mixed" if data["index_paths"] else "bounded",
        index_paths=list(data["index_paths"]),
        cache_roots=list(data["cache_roots"]),
        split_manifest=str(data["split_manifest"]),
        batch_size=batch_size,
        workers=int(data["workers"]),
        epoch_size=batch_size * optimizer_steps_per_epoch,
        validation_epoch_size=validation_samples,
        validation_seed=validation_seed,
        mixture_weights=dict(data["mixture_weights"]),
    )
    if data["index_paths"]:
        slots, identities = _production_metadata_schedule(
            data,
            seed=domains.ssl_data_order,
            logical_updates=logical_updates,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            raw_only=True,
        )
        runtime = None
    else:
        runtime = build_ssl_data_runtime(
            OmegaConf.structured(config), seed=domains.ssl_data_order
        )
        slots = []
        identities = []
        update = 0
        epoch = 0
        while update < logical_updates:
            for batch in runtime.train_loader(epoch):
                if update >= logical_updates:
                    break
                current = tuple(
                    zip(batch.dataset_ids, batch.piece_ids, strict=True)
                )
                if len(current) != batch_size:
                    raise Phase8B2ContractError(
                        "phase8b2.schedule.partial_raw_batch_forbidden"
                    )
                for batch_position, (dataset_id, piece_id) in enumerate(current):
                    identities.append((dataset_id, piece_id))
                    slots.append(
                        {
                            "dataset_id": dataset_id,
                            "piece_id": piece_id,
                            "sample_position": len(slots),
                            "logical_update": update,
                            "batch_position": batch_position,
                        }
                    )
                update += 1
            epoch += 1
    schedule_fingerprint = fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_ssl_sample_schedule",
            "identities": [list(row) for row in identities],
        }
    )
    result = {
        "actual_sample_schedule_contract_version": (
            ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION
        ),
        "seed": seed,
        "data_order_seed": domains.ssl_data_order,
        "batch_size": batch_size,
        "logical_updates": logical_updates,
        "slots": slots,
        "sample_schedule_fingerprint": schedule_fingerprint,
        "sampler": "official_ssl_data_runtime",
        "targets_read": False,
        "runtime_data_fingerprints": (
            None if runtime is None else runtime.fingerprints
        ),
        "runtime_data_composition": (
            {"source": "official_metadata_sampler"}
            if runtime is None
            else runtime.mixture_statistics
        ),
        "validation_membership": (
            None if runtime is None else asdict(runtime.validation_membership)
        ),
    }
    result["fingerprint"] = fingerprint(result)
    return result


def resolve_actual_downstream_schedule(
    data: Mapping[str, Any],
    *,
    seed: int,
    logical_updates: int,
    optimizer_steps_per_epoch: int,
    validation_samples: int,
    validation_seed: int,
) -> dict[str, object]:
    """Resolve the exact supervised raw-batch order before optimization."""

    domains = SeedDomains.create(seed)
    batch_size = int(data["batch_size"])
    config = DataConfig(
        name="mixed" if data["index_paths"] else "bounded",
        index_paths=list(data["index_paths"]),
        cache_roots=list(data["cache_roots"]),
        split_manifest=str(data["split_manifest"]),
        batch_size=batch_size,
        workers=int(data["workers"]),
        epoch_size=batch_size * optimizer_steps_per_epoch,
        validation_epoch_size=validation_samples,
        validation_seed=validation_seed,
        mixture_weights=dict(data["mixture_weights"]),
    )
    if data["index_paths"]:
        slots, identities = _production_metadata_schedule(
            data,
            seed=domains.downstream_data_order,
            logical_updates=logical_updates,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            raw_only=False,
        )
        runtime = None
    else:
        runtime = build_data_runtime(config, seed=domains.downstream_data_order)
        slots = []
        identities = []
        update = 0
        epoch = 0
        while update < logical_updates:
            for batch in runtime.train_loader(epoch):
                if update >= logical_updates:
                    break
                current = tuple(
                    zip(batch.dataset_ids, batch.piece_ids, strict=True)
                )
                if len(current) != batch_size:
                    raise Phase8B2ContractError(
                        "phase8b2.schedule.partial_downstream_batch_forbidden"
                    )
                for batch_position, (dataset_id, piece_id) in enumerate(current):
                    identities.append((dataset_id, piece_id))
                    slots.append(
                        {
                            "dataset_id": dataset_id,
                            "piece_id": piece_id,
                            "sample_position": len(slots),
                            "logical_update": update,
                            "batch_position": batch_position,
                        }
                    )
                update += 1
            epoch += 1
    schedule_fingerprint = fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_downstream_sample_schedule",
            "identities": [list(row) for row in identities],
        }
    )
    result = {
        "actual_sample_schedule_contract_version": (
            ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION
        ),
        "kind": "downstream",
        "seed": seed,
        "data_order_seed": domains.downstream_data_order,
        "batch_size": batch_size,
        "logical_updates": logical_updates,
        "slots": slots,
        "sample_schedule_fingerprint": schedule_fingerprint,
        "sampler": "official_training_data_runtime",
        "schedule_targets_read": False,
        "runtime_data_fingerprints": (
            None if runtime is None else runtime.fingerprints
        ),
        "runtime_data_composition": (
            {"source": "official_metadata_sampler"}
            if runtime is None
            else runtime.mixture_statistics
        ),
        "validation_membership": (
            None if runtime is None else asdict(runtime.validation_membership)
        ),
    }
    result["fingerprint"] = fingerprint(result)
    return result


__all__ = [
    "ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION",
    "DATA_ATTESTATION_CONTRACT_VERSION",
    "attest_data_binding",
    "resolve_actual_downstream_schedule",
    "resolve_actual_ssl_schedule",
]
