"""Official data attestation and target-free sample-schedule resolution."""

from __future__ import annotations

from collections import Counter
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
from music_critic.ssl.data import (
    IndexedSSLRawDataset,
    build_ssl_data_runtime,
    load_ssl_eligibility_manifest,
)
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


DATA_ATTESTATION_CONTRACT_VERSION = "1.2.0"
ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION = "1.2.0"
DATA_SEMANTIC_PROJECTION_CONTRACT_VERSION = "1.0.0"


def _sorted_pairs(value: object, *, category: str) -> list[list[object]]:
    if isinstance(value, Mapping):
        rows = list(value.items())
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise Phase8B2ContractError(
            f"phase8b2.data_projection.{category}_invalid"
        )
    normalized: list[list[object]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise Phase8B2ContractError(
                f"phase8b2.data_projection.{category}_invalid"
            )
        normalized.append([str(row[0]), row[1]])
    normalized.sort(key=lambda row: row[0])
    if len({row[0] for row in normalized}) != len(normalized):
        raise Phase8B2ContractError(
            f"phase8b2.data_projection.{category}_duplicate"
        )
    return normalized


def _cache_identity(index: object) -> str:
    return fingerprint(
        {
            "cache_version": index.header.cache_version,
            "records": [
                [record.piece_id, record.cache_key, record.canonical_sha256]
                for record in index.records
            ],
        }
    )


def _production_metadata_identities(
    data: Mapping[str, Any],
) -> dict[str, object] | None:
    raw_index_paths = data.get("index_paths", ())
    raw_cache_roots = data.get("cache_roots", ())
    if not isinstance(raw_index_paths, (list, tuple)) or not isinstance(
        raw_cache_roots, (list, tuple)
    ):
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.production_paths_invalid"
        )
    index_paths = tuple(str(path) for path in raw_index_paths)
    cache_roots = tuple(str(path) for path in raw_cache_roots)
    split_manifest = data.get("split_manifest", "")
    if not index_paths and not cache_roots:
        return None
    if (
        not index_paths
        or len(index_paths) != len(cache_roots)
        or not split_manifest
        or any(not Path(path).is_absolute() for path in index_paths)
        or any(not Path(path).is_absolute() for path in cache_roots)
        or not Path(str(split_manifest)).is_absolute()
    ):
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.production_paths_incomplete"
        )
    indices = []
    dataset_ids: list[str] = []
    for index_path, cache_root in zip(index_paths, cache_roots, strict=True):
        try:
            index = load_corpus_index(index_path)
        except Exception as exc:
            raise Phase8B2ContractError(
                "phase8b2.data_attestation.index_metadata_invalid"
            ) from exc
        indices.append(index)
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
    try:
        manifest = load_split_manifest(split_manifest)
    except Exception as exc:
        raise Phase8B2ContractError(
            "phase8b2.data_attestation.split_manifest_invalid"
        ) from exc
    return {
        "dataset_indices": [
            [index.header.dataset_id, index.header.index_fingerprint]
            for index in sorted(indices, key=lambda row: row.header.dataset_id)
        ],
        "cache_identities": [
            [index.header.dataset_id, _cache_identity(index)]
            for index in sorted(indices, key=lambda row: row.header.dataset_id)
        ],
        "split_manifest_fingerprint": manifest.manifest_fingerprint,
    }


def _dataset_counts_from_composition(
    composition: Mapping[str, Any],
) -> list[list[object]]:
    direct = composition.get("train_dataset_counts")
    if isinstance(direct, Mapping):
        return _sorted_pairs(direct, category="train_dataset_counts")
    train = composition.get("train")
    if isinstance(train, Mapping):
        counts = train.get("constituent_counts")
        return _sorted_pairs(counts, category="train_dataset_counts")
    raise Phase8B2ContractError(
        "phase8b2.data_projection.train_dataset_counts_missing"
    )


def attest_runtime_data_projection(
    data: Mapping[str, Any],
    *,
    data_binding: Mapping[str, Any],
    runtime_fingerprints: Mapping[str, Any],
    runtime_data_composition: Mapping[str, Any],
    validation_membership: Mapping[str, Any],
) -> dict[str, object]:
    """Project metadata and engine evidence into one source-neutral contract."""

    for name, value in (
        ("data", data),
        ("data_binding", data_binding),
        ("runtime_fingerprints", runtime_fingerprints),
        ("runtime_data_composition", runtime_data_composition),
        ("validation_membership", validation_membership),
    ):
        if not isinstance(value, Mapping):
            raise Phase8B2ContractError(
                f"phase8b2.data_projection.{name}_mapping_required"
            )

    path_identities = _production_metadata_identities(data)
    if path_identities is None:
        runtime_identity = str(
            runtime_fingerprints.get(
                "bounded_fixture_fingerprint",
                runtime_fingerprints.get("split_fingerprint", ""),
            )
        )
        split_fingerprint = str(
            runtime_fingerprints.get(
                "split_manifest_fingerprint",
                runtime_fingerprints.get("split_fingerprint", ""),
            )
        )
        train_counts = _dataset_counts_from_composition(
            runtime_data_composition
        )
        dataset_ids = sorted(str(row[0]) for row in train_counts)
        dataset_indices = [
            [
                dataset_id,
                fingerprint(
                    {
                        "kind": "bounded_runtime_index",
                        "dataset_id": dataset_id,
                        "runtime_identity": runtime_identity,
                    }
                ),
            ]
            for dataset_id in dataset_ids
        ]
        cache_identities = [
            [
                dataset_id,
                fingerprint(
                    {
                        "kind": "bounded_runtime_cache",
                        "dataset_id": dataset_id,
                        "runtime_identity": runtime_identity,
                    }
                ),
            ]
            for dataset_id in dataset_ids
        ]
    else:
        dataset_indices = _sorted_pairs(
            runtime_fingerprints.get("index_fingerprints"),
            category="index_fingerprints",
        )
        cache_identities = _sorted_pairs(
            path_identities["cache_identities"],
            category="cache_identities",
        )
        split_fingerprint = str(
            runtime_fingerprints.get("split_manifest_fingerprint", "")
        )
        if dataset_indices != path_identities["dataset_indices"]:
            raise Phase8B2ContractError(
                "phase8b2.data_projection.index_metadata_runtime_mismatch"
            )
        if split_fingerprint != path_identities[
            "split_manifest_fingerprint"
        ]:
            raise Phase8B2ContractError(
                "phase8b2.data_projection.split_metadata_runtime_mismatch"
            )
    if path_identities is not None:
        expected_indices = _sorted_pairs(
            data_binding.get("dataset_indices"),
            category="binding_dataset_indices",
        )
        expected_cache = _sorted_pairs(
            data_binding.get("cache_identities"),
            category="binding_cache_identities",
        )
        if dataset_indices != expected_indices:
            raise Phase8B2ContractError(
                "phase8b2.data_projection.index_identity_mismatch"
            )
        if cache_identities != expected_cache:
            raise Phase8B2ContractError(
                "phase8b2.data_projection.cache_identity_mismatch"
            )
        if split_fingerprint != data_binding.get(
            "split_manifest_fingerprint"
        ):
            raise Phase8B2ContractError(
                "phase8b2.data_projection.split_manifest_mismatch"
            )
    train_counts = _dataset_counts_from_composition(runtime_data_composition)
    try:
        normalized_train_counts = [
            [str(row[0]), int(row[1])] for row in train_counts
        ]
        train_size = sum(int(row[1]) for row in normalized_train_counts)
        validation_values = {
            "full_view_count": int(
                validation_membership.get("full_view_count", -1)
            ),
            "selected_count": int(
                validation_membership.get("selected_count", -1)
            ),
            "subset_limit": int(
                validation_membership.get("subset_limit", -1)
            ),
        }
    except (TypeError, ValueError) as exc:
        raise Phase8B2ContractError(
            "phase8b2.data_projection.count_invalid"
        ) from exc
    if any(row[1] < 0 for row in normalized_train_counts):
        raise Phase8B2ContractError(
            "phase8b2.data_projection.count_invalid"
        )
    train_counts = normalized_train_counts
    validation_counts = _sorted_pairs(
        validation_membership.get("dataset_counts"),
        category="validation_dataset_counts",
    )
    validation = {
        "membership_fingerprint": str(
            validation_membership.get("membership_fingerprint", "")
        ),
        "dataset_counts": validation_counts,
        **validation_values,
    }
    if (
        min(
            validation["full_view_count"],
            validation["selected_count"],
            validation["subset_limit"],
        )
        < 0
    ):
        raise Phase8B2ContractError(
            "phase8b2.data_projection.membership_or_size_mismatch"
        )
    if path_identities is not None and (
        train_size != data_binding.get("actual_train_size")
        or validation["selected_count"]
        != data_binding.get("actual_validation_size")
        or validation["membership_fingerprint"]
        != data_binding.get("validation_membership_fingerprint")
    ):
        raise Phase8B2ContractError(
            "phase8b2.data_projection.membership_or_size_mismatch"
        )
    requested_weights = _sorted_pairs(
        runtime_data_composition.get("requested_weights"),
        category="mixture_weights",
    )
    binding_weights = _sorted_pairs(
        data_binding.get("mixture_weights"),
        category="binding_mixture_weights",
    )
    try:
        requested_weights = [
            [str(dataset_id), float(value)]
            for dataset_id, value in requested_weights
        ]
        binding_weights = [
            [str(dataset_id), float(value)]
            for dataset_id, value in binding_weights
        ]
    except (TypeError, ValueError) as exc:
        raise Phase8B2ContractError(
            "phase8b2.data_projection.mixture_weights_invalid"
        ) from exc
    if requested_weights != binding_weights:
        raise Phase8B2ContractError(
            "phase8b2.data_projection.mixture_weights_mismatch"
        )
    train_composition = {
        "dataset_counts": train_counts,
        "piece_count": train_size,
    }
    train_composition["semantic_fingerprint"] = fingerprint(
        train_composition
    )
    validation["semantic_fingerprint"] = fingerprint(validation)
    result = {
        "data_semantic_projection_contract_version": (
            DATA_SEMANTIC_PROJECTION_CONTRACT_VERSION
        ),
        "dataset_indices": dataset_indices,
        "cache_identities": cache_identities,
        "split_manifest_fingerprint": split_fingerprint,
        "train_composition": train_composition,
        "validation_membership": validation,
        "mixture_weights": requested_weights,
    }
    result["fingerprint"] = fingerprint(result)
    return result


def assert_data_semantic_projection_match(
    expected: object,
    actual: object,
    *,
    stage: str,
) -> None:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise Phase8B2ContractError(
            f"phase8b2.runner.{stage}_data_projection_invalid"
        )
    for field, category in (
        ("dataset_indices", "index_identity_mismatch"),
        ("cache_identities", "cache_identity_mismatch"),
        ("split_manifest_fingerprint", "split_manifest_mismatch"),
        ("train_composition", "train_composition_mismatch"),
        ("validation_membership", "validation_membership_mismatch"),
        ("mixture_weights", "mixture_weights_mismatch"),
    ):
        if expected.get(field) != actual.get(field):
            raise Phase8B2ContractError(
                f"phase8b2.runner.{stage}_{category}"
            )
    if (
        expected.get("data_semantic_projection_contract_version")
        != DATA_SEMANTIC_PROJECTION_CONTRACT_VERSION
        or actual.get("data_semantic_projection_contract_version")
        != DATA_SEMANTIC_PROJECTION_CONTRACT_VERSION
    ):
        raise Phase8B2ContractError(
            f"phase8b2.runner.{stage}_data_projection_version_mismatch"
        )


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
    eligibility_path = str(data.get("ssl_eligibility_manifest", ""))
    included_identities = None
    if eligibility_path:
        included_identities, evidence = load_ssl_eligibility_manifest(
            eligibility_path
        )
        expected_indices = [
            [index.header.dataset_id, index.header.index_fingerprint]
            for index in sorted(indices, key=lambda item: item.header.dataset_id)
        ]
        if (
            evidence.get("split_manifest_fingerprint")
            != manifest.manifest_fingerprint
            or evidence.get("index_fingerprints") != expected_indices
        ):
            raise Phase8B2ContractError(
                "phase8b2.schedule.ssl_eligibility_binding_mismatch"
            )
        expected_population = {
            (row.dataset_id, row.piece_id)
            for row in manifest.assignments
            if row.split in {"train", "validation"}
        }
        observed_population = set(included_identities) | {
            (str(row[0]), str(row[1]))
            for row in evidence["excluded_identities"]
        }
        if observed_population != expected_population:
            raise Phase8B2ContractError(
                "phase8b2.schedule.ssl_eligibility_coverage_mismatch"
            )
    train = MultiCorpusDataset(
        datasets,
        manifest,
        split="train",
        included_identities=included_identities,
    )
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
    _production_metadata_identities(data)


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
    train_dataset_counts = Counter(
        str(row[0]) for row in validation.train_membership[
            "selected_identities"
        ]
    )
    semantic_projection = attest_runtime_data_projection(
        data,
        data_binding=binding.to_dict(),
        runtime_fingerprints=bindings,
        runtime_data_composition={
            "requested_weights": dict(data["mixture_weights"]),
            "train_dataset_counts": dict(sorted(train_dataset_counts.items())),
        },
        validation_membership=validation.evaluation_membership,
    )
    test_identities = tuple(
        tuple(row)
        for row in test.evaluation_membership["selected_identities"]
    )
    test_membership_summary = {
        "split": "test",
        "split_manifest_fingerprint": split_fingerprint,
        "membership_fingerprint": test_membership,
        "full_view_count": int(
            test.evaluation_membership["full_view_count"]
        ),
        "selected_count": int(
            test.evaluation_membership["selected_count"]
        ),
        "dataset_counts": dict(
            sorted(Counter(row[0] for row in test_identities).items())
        ),
    }
    evidence = {
        "data_attestation_contract_version": (
            DATA_ATTESTATION_CONTRACT_VERSION
        ),
        "source": "official_evaluation_data_runtime_metadata",
        "data_binding": binding.to_dict(),
        "data_semantic_projection": semantic_projection,
        "train_membership": validation.train_membership,
        "validation_membership": validation.evaluation_membership,
        "test_membership_summary": test_membership_summary,
        "actual_train_size": validation.train_membership["selected_count"],
        "actual_validation_size": validation.evaluation_membership[
            "selected_count"
        ],
        "actual_test_size": test.evaluation_membership["selected_count"],
        "targets_read_for_schedule": False,
        "test_membership_metadata_resolved": True,
        "test_inference_performed": False,
        "test_targets_accessed": False,
        "test_metrics_accessed": False,
    }
    evidence["fingerprint"] = fingerprint(evidence)
    return binding, evidence


def resolve_actual_ssl_schedule(
    data: Mapping[str, Any],
    *,
    data_semantic_projection: Mapping[str, Any],
    data_binding: Mapping[str, Any],
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
        data_semantic_projection = attest_runtime_data_projection(
            data,
            data_binding=data_binding,
            runtime_fingerprints=runtime.fingerprints,
            runtime_data_composition=runtime.mixture_statistics,
            validation_membership=asdict(runtime.validation_membership),
        )
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
        "sampler": (
            "official_metadata_sampler"
            if data["index_paths"]
            else "official_ssl_data_runtime"
        ),
        "targets_read": False,
        "canonical_payloads_read_for_schedule": False,
        "data_semantic_projection": dict(data_semantic_projection),
    }
    result["fingerprint"] = fingerprint(result)
    return result


def resolve_actual_downstream_schedule(
    data: Mapping[str, Any],
    *,
    data_semantic_projection: Mapping[str, Any],
    data_binding: Mapping[str, Any],
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
        data_semantic_projection = attest_runtime_data_projection(
            data,
            data_binding=data_binding,
            runtime_fingerprints=runtime.fingerprints,
            runtime_data_composition=runtime.mixture_statistics,
            validation_membership=asdict(runtime.validation_membership),
        )
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
        "sampler": (
            "official_metadata_sampler"
            if data["index_paths"]
            else "official_training_data_runtime"
        ),
        "schedule_targets_read": False,
        "canonical_payloads_read_for_schedule": False,
        "data_semantic_projection": dict(data_semantic_projection),
    }
    result["fingerprint"] = fingerprint(result)
    return result


__all__ = [
    "ACTUAL_SAMPLE_SCHEDULE_CONTRACT_VERSION",
    "DATA_ATTESTATION_CONTRACT_VERSION",
    "DATA_SEMANTIC_PROJECTION_CONTRACT_VERSION",
    "assert_data_semantic_projection_match",
    "attest_runtime_data_projection",
    "attest_data_binding",
    "resolve_actual_downstream_schedule",
    "resolve_actual_ssl_schedule",
]
