"""Target-blind planning for the Phase 9C-A dependency DAG."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from music_critic.experiments.phase8b2.schedule import (
    SCHEDULE_CONTRACT_VERSION,
    SeedDomains,
    build_variant_schedule,
    derive_seed,
)
from music_critic.experiments.phase8b2.attestation import (
    resolve_actual_downstream_schedule,
    resolve_actual_ssl_schedule,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    CorpusIndex,
    SplitManifest,
    create_split_manifest,
    dumps_split_manifest,
    load_corpus_index,
    load_split_manifest,
    validate_split_manifest,
)
from music_critic.ssl.data import (
    SSL_ELIGIBILITY_MANIFEST_VERSION,
    SSL_ELIGIBILITY_REQUIRED_POLICIES,
    load_ssl_eligibility_manifest,
)

from .contracts import (
    CLAIM_BOUNDARIES,
    DOWNSTREAM_MODES,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PHASE9C_PLAN_VERSION,
    PHASE9C_PROTOCOL_VERSION,
    PHASE9C_SEED,
    PRIMARY_VARIANTS,
    SSL_PRIMARY_VARIANTS,
    TASK_IDS,
    Phase9CContractError,
    fingerprint,
    locked_test_state,
    resolve_preset,
    validate_protocol,
)
from .sampling import build_source_balanced_schedule


DEFAULT_MIXTURE = {
    "dilemmadata": 1.0 / 3.0,
    "hooktheory": 1.0 / 3.0,
    "pop909_cl": 1.0 / 3.0,
}


def _write_immutable_text(destination: Path, payload: str, *, conflict: str) -> None:
    if destination.exists():
        if not destination.is_file() or destination.read_text(encoding="utf-8") != payload:
            raise Phase9CContractError(conflict)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_text(encoding="utf-8") != payload:
                raise Phase9CContractError(conflict)
    finally:
        temporary.unlink(missing_ok=True)


def _ssl_eligibility_manifest(
    indices: tuple[CorpusIndex, ...],
    cache_roots: tuple[str, ...],
    split_manifest: SplitManifest,
) -> dict[str, object]:
    """Bind a target-blind minimum-note eligibility view without repartitioning."""

    if len(indices) != len(cache_roots):
        raise Phase9CContractError("phase9c.plan.ssl_eligibility_cache_inventory_invalid")
    assignments = {
        (row.dataset_id, row.piece_id): row.split
        for row in split_manifest.assignments
    }
    eligible: list[list[str]] = []
    excluded: list[list[str]] = []
    eligible_counts: Counter[tuple[str, str]] = Counter()
    excluded_counts: Counter[tuple[str, str]] = Counter()
    for index, cache_root in zip(indices, cache_roots, strict=True):
        cache = CorpusCacheConfig(Path(cache_root))
        for record in index.records:
            split = assignments[(record.dataset_id, record.piece_id)]
            if split not in {"train", "validation"}:
                continue
            path = cache.root / cache.namespace / record.canonical_relative_path
            try:
                raw = path.read_bytes()
                if sha256(raw).hexdigest() != record.canonical_sha256:
                    raise Phase9CContractError(
                        "phase9c.plan.ssl_eligibility_cache_sha_mismatch"
                    )
                value = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                raise Phase9CContractError(
                    "phase9c.plan.ssl_eligibility_cache_unreadable"
                ) from exc
            notes = value.get("notes") if isinstance(value, dict) else None
            bars = value.get("bars") if isinstance(value, dict) else None
            if not isinstance(notes, list) or not isinstance(bars, list):
                raise Phase9CContractError(
                    "phase9c.plan.ssl_eligibility_structure_invalid"
                )
            try:
                bar_intervals = tuple(
                    (
                        Fraction(
                            int(bar["start_qn"]["num"]),
                            int(bar["start_qn"]["den"]),
                        ),
                        Fraction(
                            int(bar["start_qn"]["num"]),
                            int(bar["start_qn"]["den"]),
                        )
                        + Fraction(
                            int(bar["duration_qn"]["num"]),
                            int(bar["duration_qn"]["den"]),
                        ),
                    )
                    for bar in bars
                )
                note_onsets = tuple(
                    Fraction(
                        int(note["onset_qn"]["num"]),
                        int(note["onset_qn"]["den"]),
                    )
                    for note in notes
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise Phase9CContractError(
                    "phase9c.plan.ssl_eligibility_structure_invalid"
                ) from exc
            owners = tuple(
                tuple(
                    bar_index
                    for bar_index, (start, end) in enumerate(bar_intervals)
                    if start <= onset < end
                )
                for onset in note_onsets
            )
            if any(len(row) != 1 for row in owners):
                raise Phase9CContractError(
                    "phase9c.plan.ssl_eligibility_structure_invalid"
                )
            occupied_bars = {row[0] for row in owners}
            row = [record.dataset_id, record.piece_id]
            if len(notes) >= 2 and len(occupied_bars) >= 2:
                eligible.append(row)
                eligible_counts[(split, record.dataset_id)] += 1
            else:
                excluded.append(row)
                excluded_counts[(split, record.dataset_id)] += 1
    eligible.sort()
    excluded.sort()
    payload: dict[str, object] = {
        "contract_version": SSL_ELIGIBILITY_MANIFEST_VERSION,
        "criterion": "primary_hierarchy_policy_structural_eligibility",
        "minimum_raw_note_count": 2,
        "minimum_occupied_bar_count": 2,
        "required_policies": list(SSL_ELIGIBILITY_REQUIRED_POLICIES),
        "target_or_provenance_access": False,
        "split_assignments_changed": False,
        "split_manifest_fingerprint": split_manifest.manifest_fingerprint,
        "index_fingerprints": [
            [index.header.dataset_id, index.header.index_fingerprint]
            for index in sorted(indices, key=lambda item: item.header.dataset_id)
        ],
        "eligible_identities": eligible,
        "excluded_identities": excluded,
        "eligible_counts": [
            [split, dataset_id, count]
            for (split, dataset_id), count in sorted(eligible_counts.items())
        ],
        "excluded_counts": [
            [split, dataset_id, count]
            for (split, dataset_id), count in sorted(excluded_counts.items())
        ],
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def compose_ssl_split_manifest(
    indices: tuple[CorpusIndex, ...],
    source_manifests: tuple[SplitManifest, ...],
) -> tuple[SplitManifest, dict[str, object]]:
    """Compose existing assignments without repartitioning any record."""

    if len(indices) != 3 or len(source_manifests) != 2:
        raise Phase9CContractError("phase9c.plan.ssl_split_composition_inventory_invalid")
    index_by_dataset = {index.header.dataset_id: index for index in indices}
    if set(index_by_dataset) != set(DEFAULT_MIXTURE):
        raise Phase9CContractError("phase9c.plan.ssl_dataset_inventory_invalid")

    source_rows = {}
    covered_datasets: set[str] = set()
    source_dataset_sets: set[frozenset[str]] = set()
    for manifest in source_manifests:
        dataset_ids = tuple(row[0] for row in manifest.index_fingerprints)
        if not dataset_ids or covered_datasets.intersection(dataset_ids):
            raise Phase9CContractError("phase9c.plan.ssl_split_composition_overlap")
        try:
            manifest_indices = tuple(index_by_dataset[name] for name in dataset_ids)
        except KeyError as exc:
            raise Phase9CContractError(
                "phase9c.plan.ssl_split_composition_index_missing"
            ) from exc
        validate_split_manifest(manifest, manifest_indices)
        covered_datasets.update(dataset_ids)
        source_dataset_sets.add(frozenset(dataset_ids))
        for row in manifest.assignments:
            source_rows[(row.dataset_id, row.piece_id)] = row
    if covered_datasets != set(index_by_dataset):
        raise Phase9CContractError("phase9c.plan.ssl_split_composition_coverage_invalid")
    if source_dataset_sets != {
        frozenset({"hooktheory", "pop909_cl"}),
        frozenset({"dilemmadata"}),
    }:
        raise Phase9CContractError("phase9c.plan.ssl_split_composition_sources_invalid")

    composed = create_split_manifest(
        tuple(index_by_dataset[name] for name in sorted(index_by_dataset)),
        {key: row.split for key, row in source_rows.items()},
        seed=PHASE9C_SEED,
        policy="compose_existing_assignments",
        policy_config={
            "source_manifest_fingerprints": sorted(
                manifest.manifest_fingerprint for manifest in source_manifests
            )
        },
    )
    composed_rows = {
        (row.dataset_id, row.piece_id): row for row in composed.assignments
    }
    if composed_rows != source_rows:
        raise Phase9CContractError("phase9c.plan.ssl_split_assignment_drift")
    validate_split_manifest(composed, tuple(index_by_dataset.values()))

    dilemmadata_rows = tuple(
        row for row in composed.assignments if row.dataset_id == "dilemmadata"
    )
    held_out = {
        row.piece_id for row in dilemmadata_rows if row.split in {"validation", "test"}
    }
    ssl_train = {row.piece_id for row in dilemmadata_rows if row.split == "train"}
    if not held_out.isdisjoint(ssl_train):
        raise Phase9CContractError("phase9c.plan.dilemmadata_ssl_holdout_leakage")
    evidence = {
        "policy": composed.policy,
        "source_manifest_fingerprints": sorted(
            manifest.manifest_fingerprint for manifest in source_manifests
        ),
        "composed_manifest_fingerprint": composed.manifest_fingerprint,
        "assignment_count": len(composed.assignments),
        "assignments_preserved_exactly": True,
        "validated_against_all_three_indices": True,
        "dilemmadata_validation_test_excluded_from_ssl_train": True,
    }
    return composed, evidence


def materialize_ssl_split_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create or verify the configured common SSL manifest before planning."""

    prepared = dict(config)
    if str(prepared.get("preset", "bounded_acceptance")) == "bounded_acceptance":
        return prepared
    data = dict(prepared.get("data", {}))
    index_paths = tuple(str(path) for path in data.get("ssl_index_paths", ()))
    source_paths = tuple(
        str(path) for path in data.get("ssl_source_split_manifests", ())
    )
    destination_text = str(data.get("ssl_split_manifest", ""))
    if len(index_paths) != 3 or len(source_paths) != 2 or not destination_text:
        raise Phase9CContractError("phase9c.plan.ssl_split_composition_config_invalid")
    indices = tuple(load_corpus_index(path) for path in index_paths)
    manifests = tuple(load_split_manifest(path) for path in source_paths)
    composed, evidence = compose_ssl_split_manifest(indices, manifests)
    payload = dumps_split_manifest(composed)
    destination = Path(destination_text)
    _write_immutable_text(
        destination,
        payload,
        conflict="phase9c.plan.ssl_split_destination_conflict",
    )
    cache_roots = tuple(str(path) for path in data.get("ssl_cache_roots", ()))
    if cache_roots and len(cache_roots) != 3:
        raise Phase9CContractError(
            "phase9c.plan.ssl_eligibility_cache_inventory_invalid"
        )
    data["ssl_split_composition"] = evidence
    if cache_roots:
        eligibility = _ssl_eligibility_manifest(indices, cache_roots, composed)
        eligibility_destination = Path(
            str(
                data.get(
                    "ssl_eligibility_manifest",
                    str(destination) + ".eligibility.json",
                )
            )
        )
        _write_immutable_text(
            eligibility_destination,
            json.dumps(
                eligibility,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            conflict="phase9c.plan.ssl_eligibility_destination_conflict",
        )
        data["ssl_eligibility_manifest"] = str(eligibility_destination)
        data["ssl_eligibility"] = {
            "fingerprint": eligibility["fingerprint"],
            "criterion": eligibility["criterion"],
            "minimum_raw_note_count": eligibility["minimum_raw_note_count"],
            "minimum_occupied_bar_count": eligibility[
                "minimum_occupied_bar_count"
            ],
            "required_policies": eligibility["required_policies"],
            "split_assignments_changed": False,
            "target_or_provenance_access": False,
            "eligible_counts": eligibility["eligible_counts"],
            "excluded_counts": eligibility["excluded_counts"],
        }
    prepared["data"] = data
    return prepared


def _repository_evidence(*, require_clean: bool) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase9CContractError("phase9c.plan.repository_unavailable") from exc
    clean = status == ""
    if require_clean and not clean:
        raise Phase9CContractError("phase9c.plan.production_clean_head_required")
    return {"git_head": head, "clean": clean, "exact_head_required": True}


def _file_sha256(path: str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_identities() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    ssl = {
        dataset_id: tuple(f"{dataset_id}:fixture:{index}" for index in range(4))
        for dataset_id in sorted(DEFAULT_MIXTURE)
    }
    downstream = tuple(f"dilemmadata:train:fixture:{index}" for index in range(6))
    return ssl, downstream


def _production_data(config: Mapping[str, Any]) -> tuple[
    dict[str, tuple[str, ...]], tuple[str, ...], dict[str, object], dict[str, object]
]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise Phase9CContractError("phase9c.plan.production_data_missing")
    index_paths = tuple(str(path) for path in data.get("ssl_index_paths", ()))
    cache_roots = tuple(str(path) for path in data.get("ssl_cache_roots", ()))
    if len(index_paths) != 3 or len(cache_roots) != 3:
        raise Phase9CContractError("phase9c.plan.ssl_three_sources_required")
    ssl_split_path = str(data.get("ssl_split_manifest", ""))
    ssl_eligibility_path = str(data.get("ssl_eligibility_manifest", ""))
    downstream_split_path = str(data.get("downstream_split_manifest", ""))
    downstream_raw_index = str(data.get("downstream_raw_index", ""))
    downstream_raw_cache = str(data.get("downstream_raw_cache_root", ""))
    target_index = str(data.get("target_cache_index", ""))
    target_cache = str(data.get("target_cache_root", ""))
    all_paths = (
        *index_paths,
        *cache_roots,
        ssl_split_path,
        ssl_eligibility_path,
        downstream_split_path,
        downstream_raw_index,
        downstream_raw_cache,
        target_index,
        target_cache,
    )
    missing = sorted(path for path in all_paths if not path or not Path(path).exists())
    if missing:
        raise Phase9CContractError("phase9c.plan.production_path_missing:" + ",".join(missing))

    indices = tuple(load_corpus_index(path) for path in index_paths)
    by_dataset = {index.header.dataset_id: index for index in indices}
    if set(by_dataset) != set(DEFAULT_MIXTURE):
        raise Phase9CContractError("phase9c.plan.ssl_dataset_inventory_invalid")
    ssl_split = load_split_manifest(ssl_split_path)
    validate_split_manifest(ssl_split, indices)
    eligible_identities, eligibility_manifest = load_ssl_eligibility_manifest(
        ssl_eligibility_path
    )
    expected_index_fingerprints = [
        [index.header.dataset_id, index.header.index_fingerprint]
        for index in sorted(indices, key=lambda item: item.header.dataset_id)
    ]
    if (
        eligibility_manifest.get("split_manifest_fingerprint")
        != ssl_split.manifest_fingerprint
        or eligibility_manifest.get("index_fingerprints")
        != expected_index_fingerprints
        or eligibility_manifest.get("fingerprint")
        != data.get("ssl_eligibility", {}).get("fingerprint")
    ):
        raise Phase9CContractError(
            "phase9c.plan.ssl_eligibility_evidence_invalid"
        )
    expected_eligibility_population = {
        (row.dataset_id, row.piece_id)
        for row in ssl_split.assignments
        if row.split in {"train", "validation"}
    }
    observed_eligibility_population = set(eligible_identities) | {
        (str(row[0]), str(row[1]))
        for row in eligibility_manifest["excluded_identities"]
    }
    if observed_eligibility_population != expected_eligibility_population:
        raise Phase9CContractError(
            "phase9c.plan.ssl_eligibility_coverage_invalid"
        )
    composition = data.get("ssl_split_composition")
    if (
        not isinstance(composition, Mapping)
        or composition.get("composed_manifest_fingerprint")
        != ssl_split.manifest_fingerprint
        or composition.get("assignments_preserved_exactly") is not True
        or composition.get("validated_against_all_three_indices") is not True
        or composition.get("dilemmadata_validation_test_excluded_from_ssl_train")
        is not True
    ):
        raise Phase9CContractError("phase9c.plan.ssl_split_composition_evidence_invalid")
    train_keys = {
        (row.dataset_id, row.piece_id)
        for row in ssl_split.assignments
        if row.split == "train"
    }
    ssl_identities = {
        dataset_id: tuple(
            row.piece_id
            for row in index.records
            if (dataset_id, row.piece_id) in train_keys
            and (dataset_id, row.piece_id) in eligible_identities
        )
        for dataset_id, index in sorted(by_dataset.items())
    }
    if any(not rows for rows in ssl_identities.values()):
        raise Phase9CContractError("phase9c.plan.ssl_train_source_empty")

    downstream_index = load_corpus_index(downstream_raw_index)
    if downstream_index.header.dataset_id != "dilemmadata":
        raise Phase9CContractError("phase9c.plan.downstream_dataset_invalid")
    downstream_split = load_split_manifest(downstream_split_path)
    validate_split_manifest(downstream_split, (downstream_index,))
    split_rows: dict[str, list[object]] = {name: [] for name in ("train", "validation", "test")}
    for row in downstream_split.assignments:
        if row.dataset_id == "dilemmadata" and row.split in split_rows:
            split_rows[row.split].append(row)
    record_counts = {name: len(rows) for name, rows in split_rows.items()}
    component_counts = {
        name: len({row.component_fingerprint for row in rows})
        for name, rows in split_rows.items()
    }
    if record_counts != {"train": 577, "validation": 71, "test": 71} or component_counts != {
        "train": 565,
        "validation": 71,
        "test": 71,
    }:
        raise Phase9CContractError("phase9c.plan.dilemmadata_split_counts_invalid")
    downstream_train = tuple(sorted(row.piece_id for row in split_rows["train"]))
    validation_identities = tuple(sorted(row.piece_id for row in split_rows["validation"]))
    test_identities = tuple(sorted(row.piece_id for row in split_rows["test"]))
    data_projection = {
        "kind": "production",
        "ssl_indices": [
            {
                "dataset_id": index.header.dataset_id,
                "index_fingerprint": index.header.index_fingerprint,
                "cache_root": str(Path(cache_roots[index_paths.index(path)]).resolve()),
            }
            for path, index in zip(index_paths, indices, strict=True)
        ],
        "ssl_split_manifest_fingerprint": ssl_split.manifest_fingerprint,
        "ssl_split_composition": dict(composition),
        "ssl_eligibility": dict(data["ssl_eligibility"]),
        "downstream_raw_index_fingerprint": downstream_index.header.index_fingerprint,
        "downstream_target_index_sha256": _file_sha256(target_index),
        "downstream_split_manifest_fingerprint": downstream_split.manifest_fingerprint,
        "downstream_split_record_counts": record_counts,
        "downstream_split_component_counts": component_counts,
        "validation_membership_fingerprint": fingerprint(
            {"split": "validation", "piece_ids": validation_identities}
        ),
        "test_membership_fingerprint": fingerprint(
            {"split": "test", "piece_ids": test_identities}
        ),
        "test_membership_count": len(test_identities),
        "test_identities_serialized": False,
        "target_bundles_loaded_during_planning": False,
    }
    runtime_paths = {
        "ssl_index_paths": list(index_paths),
        "ssl_cache_roots": list(cache_roots),
        "ssl_split_manifest": ssl_split_path,
        "ssl_eligibility_manifest": ssl_eligibility_path,
        "downstream_raw_index": downstream_raw_index,
        "downstream_raw_cache_root": downstream_raw_cache,
        "target_cache_index": target_index,
        "target_cache_root": target_cache,
        "downstream_split_manifest": downstream_split_path,
    }
    return ssl_identities, downstream_train, data_projection, runtime_paths


def _downstream_schedule(identities: tuple[str, ...], count: int, seed: int) -> dict[str, object]:
    schedule = build_source_balanced_schedule(
        {"dilemmadata": identities},
        weights={"dilemmadata": 1.0},
        sample_count=count,
        seed=seed,
    )
    engine_fingerprint = fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_downstream_sample_schedule",
            "identities": [
                [str(row["dataset_id"]), str(row["piece_id"])]
                for row in schedule["slots"]
            ],
        }
    )
    return {**schedule, "engine_schedule_fingerprint": engine_fingerprint}


def _official_schedule_evidence(
    actual: Mapping[str, object],
    *,
    source_sizes: Mapping[str, int],
    weights: Mapping[str, float],
    engine_key: str,
) -> dict[str, object]:
    slots = list(actual["slots"])
    counts = Counter(str(row["dataset_id"]) for row in slots)
    unique = {
        source: len(
            {
                str(row["piece_id"])
                for row in slots
                if row["dataset_id"] == source
            }
        )
        for source in sorted(source_sizes)
    }
    payload = {
        "contract_version": "1.0.0",
        "seed": actual["data_order_seed"],
        "weights": [[source, float(weights[source])] for source in sorted(weights)],
        "slots": slots,
        "dataset_counts": {source: counts[source] for source in sorted(source_sizes)},
        "unique_record_counts": unique,
        "repeat_counts": {
            source: counts[source] - unique[source] for source in sorted(source_sizes)
        },
        "completed_or_entered_cycle_counts": {
            source: (
                0
                if counts[source] == 0
                else (counts[source] + source_sizes[source] - 1) // source_sizes[source]
            )
            for source in sorted(source_sizes)
        },
        "replacement_within_cycle": False,
        "target_or_provenance_access": False,
        "official_sampler": actual["sampler"],
        "engine_schedule_fingerprint": actual["sample_schedule_fingerprint"],
        engine_key: actual["sample_schedule_fingerprint"],
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def build_experiment_plan(config: Mapping[str, Any]) -> dict[str, object]:
    """Resolve the complete immutable one-seed plan without target-bundle access."""

    preset_name = str(config.get("preset", "bounded_acceptance"))
    preset = resolve_preset(
        preset_name,
        ssl_updates=config.get("ssl_updates"),
        downstream_epochs=config.get("downstream_epochs"),
        downstream_steps_per_epoch=config.get("downstream_steps_per_epoch"),
        batch_size=config.get("batch_size"),
        bootstrap_replicates=config.get("bootstrap_replicates"),
    )
    production = preset_name != "bounded_acceptance"
    if production:
        ssl_identities, downstream_ids, data_projection, runtime_paths = _production_data(config)
    else:
        ssl_identities, downstream_ids = _bounded_identities()
        data_projection = {
            "kind": "bounded_synthetic_fixture",
            "downstream_split_record_counts": {"train": 6, "validation": 4, "test": 4},
            "downstream_split_component_counts": {"train": 6, "validation": 4, "test": 4},
            "validation_membership_fingerprint": fingerprint({"fixture": "validation"}),
            "test_membership_fingerprint": fingerprint({"fixture": "test"}),
            "test_membership_count": 4,
            "test_identities_serialized": False,
            "target_bundles_loaded_during_planning": False,
        }
        runtime_paths = {}

    batch_size = preset.batch_size or 1
    ssl_updates = preset.ssl_updates or 1
    downstream_epochs = preset.downstream_epochs or 1
    downstream_steps = preset.downstream_steps_per_epoch or 1
    mixture = dict(config.get("mixture_weights", DEFAULT_MIXTURE))
    if production:
        ssl_data_config = {
            "index_paths": runtime_paths["ssl_index_paths"],
            "cache_roots": runtime_paths["ssl_cache_roots"],
            "split_manifest": runtime_paths["ssl_split_manifest"],
            "ssl_eligibility_manifest": runtime_paths[
                "ssl_eligibility_manifest"
            ],
            "batch_size": batch_size,
            "workers": 0,
            "mixture_weights": mixture,
        }
        actual_ssl = resolve_actual_ssl_schedule(
            ssl_data_config,
            data_semantic_projection=data_projection,
            data_binding={},
            seed=PHASE9C_SEED,
            logical_updates=ssl_updates,
            optimizer_steps_per_epoch=min(ssl_updates, downstream_steps),
            validation_samples=0,
            validation_seed=20260817,
        )
        schedule = _official_schedule_evidence(
            actual_ssl,
            source_sizes={source: len(rows) for source, rows in ssl_identities.items()},
            weights=mixture,
            engine_key="sample_schedule_fingerprint",
        )
        downstream_data_config = {
            "index_paths": [runtime_paths["downstream_raw_index"]],
            "cache_roots": [runtime_paths["downstream_raw_cache_root"]],
            "split_manifest": runtime_paths["downstream_split_manifest"],
            "batch_size": batch_size,
            "workers": 0,
            "mixture_weights": {"dilemmadata": 1.0},
        }
        actual_downstream = resolve_actual_downstream_schedule(
            downstream_data_config,
            data_semantic_projection=data_projection,
            data_binding={},
            seed=PHASE9C_SEED,
            logical_updates=downstream_epochs * downstream_steps,
            optimizer_steps_per_epoch=downstream_steps,
            validation_samples=0,
            validation_seed=20260817,
        )
        downstream_schedule = _official_schedule_evidence(
            actual_downstream,
            source_sizes={"dilemmadata": len(downstream_ids)},
            weights={"dilemmadata": 1.0},
            engine_key="engine_schedule_fingerprint",
        )
    else:
        schedule = build_source_balanced_schedule(
            ssl_identities,
            weights=mixture,
            sample_count=ssl_updates * batch_size,
            seed=derive_seed(PHASE9C_SEED, "phase9c/ssl_data_order"),
        )
        downstream_schedule = _downstream_schedule(
            downstream_ids,
            downstream_epochs * downstream_steps * batch_size,
            derive_seed(PHASE9C_SEED, "phase9c/downstream_data_order"),
        )
    domains = SeedDomains.create(PHASE9C_SEED)
    ssl_variants = tuple(variant for variant in preset.variants if variant != "scratch")
    variant_schedules = {
        variant: build_variant_schedule(
            variant,
            comparison_mode="encoder_forward_matched",
            logical_updates=ssl_updates,
            batch_size=batch_size,
            matched_encoder_forwards_per_update=PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
            sample_identity_schedule=tuple(
                (str(row["dataset_id"]), str(row["piece_id"]))
                for row in schedule["slots"]
            ),
            mask_seed=domains.ssl_mask_planning,
        ).to_dict()
        for variant in ssl_variants
    }
    protocol_payload = {
        "contract_version": PHASE9C_PROTOCOL_VERSION,
        "phase": "9C-A",
        "seed": PHASE9C_SEED,
        "preset": preset.to_dict(),
        "repository": _repository_evidence(require_clean=production),
        "primary_variants": list(PRIMARY_VARIANTS),
        "executed_variants": list(preset.variants),
        "optional_variants_automatic": False,
        "task_ids": list(TASK_IDS),
        "data": data_projection,
        "mixture": {
            "weights": [[key, float(mixture[key])] for key in sorted(mixture)],
            "schedule_fingerprint": schedule["fingerprint"],
            "target_blind": True,
            "validation_and_test_excluded": True,
        },
        "paired_initialization": {
            "initial_encoder_seed": domains.model_initialization,
            "initial_encoder_fingerprint": fingerprint(
                {"domain": "initial_encoder", "seed": domains.model_initialization}
            ),
            "fresh_head_seed": domains.downstream_initialization,
            "fresh_head_fingerprint": fingerprint(
                {"domain": "fresh_heads", "seed": domains.downstream_initialization}
            ),
            "ssl_data_order_seed": domains.ssl_data_order,
            "downstream_data_order_seed": domains.downstream_data_order,
        },
        "compute": {
            "unit": "instrumented_encoder_forward",
            "encoder_forwards_per_logical_update": PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
            "ssl_logical_updates": preset.ssl_updates,
            "downstream_epochs": preset.downstream_epochs,
            "downstream_steps_per_epoch": preset.downstream_steps_per_epoch,
            "batch_size": preset.batch_size,
            "fixed_ssl_budget_no_downstream_early_stopping": True,
            "fixed_downstream_optimizer_updates": downstream_epochs * downstream_steps,
            "downstream_checkpoint_policy": "last_after_fixed_budget",
        },
        "supervised": {
            "reduction": "candidate_rows_mean_then_source_entries_mean_then_fixed_equal_task_sum",
            "fp32_boundary": "encoder_amp_float16_heads_logits_ce_total_fp32",
            "grad_scaler_initial_scale": 16384,
            "scheduler_after_applied_update_only": True,
            "positive_unlabeled_and_open_string_heads": False,
        },
        "selection": {
            "split": "validation",
            "comparison_metric": "mean_task_nll_div_log_class_count",
            "lower_is_better": True,
            "checkpoint_policy": "last_after_fixed_budget",
            "checkpoint_selection_between_epochs": False,
            "validation_compares_final_checkpoints_only": True,
            "fixed_before_results": True,
        },
        "bootstrap": {
            "unit": "component",
            "seed": domains.bootstrap,
            "replicates": preset.bootstrap_replicates,
            "reference": "scratch_same_transfer_mode",
        },
        "test_lock": locked_test_state(),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }
    protocol = {**protocol_payload, "fingerprint": fingerprint(protocol_payload)}
    validate_protocol(protocol)

    ssl_cells = [
        {
            "cell_id": f"ssl/{variant}",
            "variant_id": variant,
            "schedule": variant_schedules[variant],
            "initial_encoder_fingerprint": protocol["paired_initialization"]["initial_encoder_fingerprint"],
        }
        for variant in ssl_variants
    ]
    export_cells = [
        {
            "cell_id": "encoder_export/initial_scratch",
            "variant_id": "scratch",
            "depends_on": None,
            "source_kind": "phase6_hierarchical",
        },
        *[
        {
            "cell_id": f"encoder_export/{variant}",
            "variant_id": variant,
            "depends_on": f"ssl/{variant}",
            "source_kind": (
                "phase7a_ssl"
                if variant in {"phase7a_control", "phase8a_mask_only"}
                else "phase8b_multilevel_ssl"
            ),
        }
        for variant in ssl_variants
        ],
    ]
    downstream_cells = []
    for variant in preset.variants:
        for mode in DOWNSTREAM_MODES:
            downstream_cells.append(
                {
                    "cell_id": f"downstream/{variant}/{mode}",
                    "variant_id": variant,
                    "transfer_mode": (
                        f"scratch_{mode}" if variant == "scratch" else mode
                    ),
                    "engine_transfer_mode": (
                        "supervised_scratch"
                        if variant == "scratch" and mode == "full_finetune"
                        else mode
                    ),
                    "depends_on": (
                        "encoder_export/initial_scratch"
                        if variant == "scratch" and mode == "frozen_probe"
                        else (None if variant == "scratch" else f"encoder_export/{variant}")
                    ),
                    "sample_schedule_fingerprint": downstream_schedule["engine_schedule_fingerprint"],
                    "fresh_head_fingerprint": protocol["paired_initialization"]["fresh_head_fingerprint"],
                    "optimizer_update_budget": downstream_epochs * downstream_steps,
                    "comparison_checkpoint": "last.pt",
                }
            )
    validation_cells = [
        {
            "cell_id": f"validation/{row['variant_id']}/{row['transfer_mode']}",
            "variant_id": row["variant_id"],
            "transfer_mode": row["transfer_mode"],
            "depends_on": row["cell_id"],
            "prior_dependency": "train_priors/dilemmadata",
            "split": "validation",
            "membership_fingerprint": data_projection["validation_membership_fingerprint"],
            "optimizer_update_budget": row["optimizer_update_budget"],
            "comparison_checkpoint": row["comparison_checkpoint"],
        }
        for row in downstream_cells
    ]
    plan_payload = {
        "contract_version": PHASE9C_PLAN_VERSION,
        "protocol": protocol,
        "data_semantic_projection": data_projection,
        "runtime_paths": runtime_paths,
        "ssl_sample_schedule": schedule,
        "downstream_sample_schedule": downstream_schedule,
        "variant_schedules": variant_schedules,
        "ssl_cells": ssl_cells,
        "encoder_export_cells": export_cells,
        "downstream_cells": downstream_cells,
        "train_prior_cells": [
            {
                "cell_id": "train_priors/dilemmadata",
                "split": "train",
                "target_access": "downstream_train_only",
                "depends_on": None,
            }
        ],
        "validation_cells": validation_cells,
        "profile_candidates": list(config.get("profile_batch_candidates", [1, 2, 3, 4, 6, 8])),
        "profile_report_path": str(config.get("profile_report_path", "")),
        "profile_rebuild_config": (
            None
            if not production
            else {
                **dict(config),
                "preset": "rtx_profile",
                "profile_report_path": "",
            }
        ),
        "production_run_requires_explicit_action": True,
        "test_action_implemented": False,
    }
    return {**plan_payload, "fingerprint": fingerprint(plan_payload)}


__all__ = [
    "DEFAULT_MIXTURE",
    "build_experiment_plan",
    "compose_ssl_split_manifest",
    "materialize_ssl_split_manifest",
]
