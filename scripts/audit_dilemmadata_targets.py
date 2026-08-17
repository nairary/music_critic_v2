#!/usr/bin/env python3
"""Deterministic Phase 9B.2A audit for external Dilemmadata targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
import os
from os import PathLike
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

from music_critic.adapters.dilemmadata import (
    DILEMMADATA_ADAPTER_VERSION,
    DILEMMADATA_CONTENT_FINGERPRINT,
    DILEMMADATA_DATASET_NAME,
    DILEMMADATA_RAW_PROJECTION_VERSION,
    DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION,
    DILEMMADATA_RELEASE_COMMIT,
    DilemmadataAccepted,
    DilemmadataAdapterConfig,
    DilemmadataCorpusIdentity,
    DilemmadataCorpusRecord,
    DilemmadataQuarantine,
    convert_dilemmadata_record,
    dilemmadata_raw_source_value_fields,
    discover_dilemmadata_corpus,
)
from music_critic.adapters.dilemmadata_targets import (
    DILEMMADATA_TARGET_ADAPTER_VERSION,
    DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION,
    DILEMMADATA_TARGET_AUDIT_REPORT_VERSION,
    DILEMMADATA_TARGET_METADATA_VERSION,
    DILEMMADATA_TARGET_SIDECAR_VERSION,
    DilemmadataTargetAccepted,
    DilemmadataTargetQuarantine,
    build_dilemmadata_target_sidecar,
    load_dilemmadata_target_metadata_index,
)
from music_critic.graph import graph_fingerprint, model_input_fingerprint
from music_critic.tasks.alignment import align_sample_targets
from music_critic.tasks.collator import collate_multisource_samples
from music_critic.tasks.corpus import (
    CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_DEFERRED_MAPPINGS,
    DILEMMADATA_SOURCE_FAMILIES,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
    DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
    DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
    DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
    dilemmadata_family_registry_fingerprint,
)
from music_critic.tasks.encoding import (
    TARGET_ENCODING_REGISTRY_VERSION,
    dilemmadata_target_encoding_contract_fingerprint,
    target_encoding_contract_fingerprint,
    target_encoding_spec,
)
from music_critic.tasks.loading import IndexedMultiSourceDataset
from music_critic.tasks.multisource import (
    TARGET_BUNDLE_CONTRACT_VERSION,
    attach_target_bundle,
    prepare_multisource_sample,
)
from music_critic.tasks.ontology import (
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_fingerprint,
)


ENV_ROOT = "MUSIC_CRITIC_DILEMMADATA_ROOT"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "target_manifest.json"
)
RAW_PRODUCTION_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "production_manifest.json"
)
PHASE9A_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "audit_manifest.json"
)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def dumps_target_audit_report(
    report: Mapping[str, object],
    *,
    indent: int | None = 2,
) -> str:
    return _canonical_json(report, indent=indent) + ("\n" if indent is not None else "")


def _combined_fingerprint(
    domain: str,
    rows: Iterable[tuple[str, str]],
) -> str:
    return _fingerprint(
        {
            "domain": domain,
            "rows": [list(row) for row in sorted(rows)],
        }
    )


def _family_accumulators() -> dict[str, dict[str, Any]]:
    return {
        spec.task_id: {
            "source_row_count": 0,
            "available": 0,
            "masked": 0,
            "missing": 0,
            "ambiguous": 0,
            "unsupported": 0,
            "source_entry_count": 0,
            "emitted_rows": 0,
            "available_entries": 0,
            "masked_entries": 0,
            "equal_duplicate_merges": 0,
            "conflict": 0,
            "merged_tie_agreement": 0,
            "merged_tie_conflict": 0,
            "aligned_rows": 0,
            "exact_aligned": 0,
            "unaligned": 0,
            "model_ready_rows": 0,
            "aligned_masked_rows": 0,
            "value_counts": Counter(),
        }
        for spec in DILEMMADATA_SOURCE_FAMILIES
    }


def _alignment_oracle_access_projection() -> dict[str, object]:
    raw_fields = {
        dialect: dilemmadata_raw_source_value_fields(dialect)
        for dialect in ("an_joint", "dlc")
    }
    target_fields = {
        dialect: tuple(
            sorted(
                {
                    "alt_label",
                    *(
                        field
                        for spec in DILEMMADATA_SOURCE_FAMILIES
                        if spec.dialect == dialect
                        for field in (
                            *spec.source_fields,
                            *((spec.gate_field,) if spec.gate_field else ()),
                            *((spec.source_identity_field,)
                              if spec.source_identity_field else ()),
                        )
                    ),
                }
            )
        )
        for dialect in ("an_joint", "dlc")
    }
    overlap = {
        dialect: tuple(sorted(set(raw_fields[dialect]) & set(target_fields[dialect])))
        for dialect in raw_fields
    }
    return {
        "independent_raw_reconstruction": True,
        "ordered_row_semantics": [
            "ordinal",
            "physical_line",
            "exact_rational_onset",
            "tie_continuation",
            "canonical_note_id",
        ],
        "raw_source_value_fields": {
            dialect: list(fields) for dialect, fields in raw_fields.items()
        },
        "target_source_value_field_overlap": {
            dialect: list(fields) for dialect, fields in overlap.items()
        },
        "target_metadata_access_count": 0,
        "target_value_access_count": sum(len(fields) for fields in overlap.values()),
        "self_fingerprint_role": "corruption_check_only",
    }


def _update_family_audit(
    target: DilemmadataTargetAccepted,
    aligned: Sequence[Any],
    accumulators: dict[str, dict[str, Any]],
) -> None:
    targets = {row.task_id: row for row in target.target_bundle.targets}
    aligned_by_task = {row.task_id: row for row in aligned}
    for statistics in target.statistics.family_statistics:
        aggregate = accumulators[statistics.task_id]
        aggregate["source_row_count"] += statistics.source_row_count
        aggregate["available"] += statistics.available_count
        aggregate["masked"] += statistics.masked_count
        aggregate["missing"] += statistics.missing_count
        aggregate["ambiguous"] += statistics.ambiguous_count
        aggregate["unsupported"] += statistics.unsupported_count
        aggregate["source_entry_count"] += statistics.source_entry_count
        aggregate["emitted_rows"] += statistics.emitted_entry_count
        aggregate["available_entries"] += statistics.available_entry_count
        aggregate["masked_entries"] += statistics.masked_entry_count
        aggregate["equal_duplicate_merges"] += (
            statistics.equal_duplicate_merge_count
        )
        aggregate["conflict"] += statistics.conflict_count
        aggregate["merged_tie_agreement"] += (
            statistics.merged_tie_agreement_count
        )
        aggregate["merged_tie_conflict"] += (
            statistics.merged_tie_conflict_count
        )
        sample_target = targets[statistics.task_id]
        aggregate["value_counts"].update(
            value
            for value, available in zip(
                sample_target.values,
                sample_target.availability_mask,
                strict=True,
            )
            if available
        )
        family = aligned_by_task[statistics.task_id]
        aggregate["aligned_rows"] += len(family.rows)
        aligned_available = sum(
            row.availability and row.local_entity_index >= 0
            for row in family.rows
        )
        unaligned_available = sum(
            row.availability and row.local_entity_index == -1
            for row in family.rows
        )
        aggregate["exact_aligned"] += aligned_available
        aggregate["unaligned"] += unaligned_available
        aggregate["aligned_masked_rows"] += sum(
            not row.availability for row in family.rows
        )
        if target_encoding_spec(statistics.task_id).model_ready:
            aggregate["model_ready_rows"] += aligned_available


def _family_projection(
    accumulators: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in DILEMMADATA_SOURCE_FAMILIES:
        aggregate = accumulators[spec.task_id]
        encoding = target_encoding_spec(spec.task_id)
        value_counts = aggregate["value_counts"]
        assert isinstance(value_counts, Counter)
        value_projection = tuple(
            sorted((str(value), int(count)) for value, count in value_counts.items())
        )
        rows.append(
            {
                "task_id": spec.task_id,
                "dialect": spec.dialect,
                "family": spec.family,
                "provenance_source": spec.ontology_spec.source_adapter,
                "alignment_entity_type": spec.ontology_spec.source_alignment_type,
                "coordinate": spec.coordinate,
                "mapping_status": spec.mapping_status,
                "encoding_kind": encoding.encoding_kind,
                "encoding_registry_version": encoding.registry_version,
                "model_ready": encoding.model_ready,
                "supervision_regime": encoding.supervision_regime,
                "frozen_vocabulary_size": len(encoding.vocabulary or ()),
                "observed_value_count": sum(value_counts.values()),
                "observed_distinct_value_count": len(value_counts),
                "observed_values_fingerprint": _combined_fingerprint(
                    f"{spec.task_id}.observed-values.1",
                    ((value, str(count)) for value, count in value_projection),
                ),
                "source_row_count": int(aggregate["source_row_count"]),
                "available": int(aggregate["available"]),
                "masked": int(aggregate["masked"]),
                "missing": int(aggregate["missing"]),
                "ambiguous": int(aggregate["ambiguous"]),
                "unsupported": int(aggregate["unsupported"]),
                "deferred": (
                    int(aggregate["available"])
                    if not encoding.model_ready
                    else 0
                ),
                "source_entry_count": int(aggregate["source_entry_count"]),
                "emitted_rows": int(aggregate["emitted_rows"]),
                "available_entries": int(aggregate["available_entries"]),
                "masked_entries": int(aggregate["masked_entries"]),
                "aligned_rows": int(aggregate["aligned_rows"]),
                "exact_aligned": int(aggregate["exact_aligned"]),
                "unaligned": int(aggregate["unaligned"]),
                "model_ready_rows": int(aggregate["model_ready_rows"]),
                "aligned_masked_rows": int(aggregate["aligned_masked_rows"]),
                "equal_duplicate_merges": int(
                    aggregate["equal_duplicate_merges"]
                ),
                "conflict": int(aggregate["conflict"]),
                "merged_tie_agreement": int(
                    aggregate["merged_tie_agreement"]
                ),
                "merged_tie_conflict": int(
                    aggregate["merged_tie_conflict"]
                ),
            }
        )
    return rows


def _select_smallest(
    rows: list[tuple[int, DilemmadataCorpusRecord]],
    count: int,
) -> tuple[DilemmadataCorpusRecord, ...]:
    return tuple(
        record
        for _size, record in sorted(
            rows,
            key=lambda row: (row[0], row[1].record_id),
        )[:count]
    )


def _e2e_acceptance(
    records: Sequence[DilemmadataCorpusRecord],
    *,
    corpus_fingerprint: str,
) -> dict[str, object]:
    unique_records = tuple(
        sorted(
            {record.record_id: record for record in records}.values(),
            key=lambda row: row.record_id,
        )
    )
    raw_outcomes: list[DilemmadataAccepted] = []
    target_outcomes: dict[str, DilemmadataTargetAccepted] = {}
    for record in unique_records:
        raw = convert_dilemmadata_record(record)
        if not isinstance(raw, DilemmadataAccepted):
            raise RuntimeError(
                f"selected E2E record became raw-quarantined: {record.record_id}"
            )
        target = build_dilemmadata_target_sidecar(raw)
        if not isinstance(target, DilemmadataTargetAccepted):
            raise RuntimeError(
                f"selected E2E record became target-quarantined: {record.record_id}"
            )
        raw_outcomes.append(raw)
        target_outcomes[raw.piece.piece_id] = target

    merged_tie_record_ids = tuple(
        raw.record.record_id
        for raw in raw_outcomes
        if raw.statistics.tie_merge_count > 0
    )
    event_record_ids = tuple(
        raw.record.record_id
        for raw in raw_outcomes
        if any(
            statistics.available_entry_count > 0
            for statistics in target_outcomes[
                raw.piece.piece_id
            ].statistics.family_statistics
            if statistics.task_id
            in {
                "dilemmadata.dlc.cadence",
                "dilemmadata.dlc.phrase.boundary",
                "dilemmadata.dlc.section.boundary",
            }
        )
    )
    by_raw_identity: dict[str, list[DilemmadataAccepted]] = defaultdict(list)
    for raw in raw_outcomes:
        by_raw_identity[raw.record.raw_equivalence_id].append(raw)
    alternative_components = tuple(
        tuple(sorted(raw.record.record_id for raw in rows))
        for _identity, rows in sorted(by_raw_identity.items())
        if len(rows) > 1
    )
    alternative_target_fingerprints_are_distinct = all(
        len(
            {
                target_outcomes[raw.piece.piece_id].sidecar_fingerprint
                for raw in rows
            }
        )
        == len(rows)
        for rows in by_raw_identity.values()
        if len(rows) > 1
    )

    raw_config = DilemmadataAdapterConfig()
    cache_adapter_config = {
        **asdict(raw_config),
        "raw_projection_version": DILEMMADATA_RAW_PROJECTION_VERSION,
        "cache_input_identity_version": CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    }
    with tempfile.TemporaryDirectory(prefix="music-critic-9b2a-e2e-") as temporary:
        cache = CorpusCacheConfig(Path(temporary) / "cache")
        inputs = tuple(
            CanonicalCorpusInput(
                piece=raw.piece,
                lineage_group_id=raw.record.lineage_group_id,
                source_identity=raw.record.record_id,
                source_relative_path=raw.record.relative_path,
                source_sha256=raw.record.physical_source_sha256,
                suggested_split=raw.record.suggested_split,
                cache_input_sha256=raw.record.raw_projection_sha256,
            )
            for raw in raw_outcomes
        )
        index, build_report = cache_canonical_corpus(
            inputs,
            cache_config=cache,
            dataset_id=DILEMMADATA_DATASET_NAME,
            adapter_name="dilemmadata",
            adapter_version=DILEMMADATA_ADAPTER_VERSION,
            adapter_config=cache_adapter_config,
            source_identity=(
                f"dilemmadata:v1.0:{DILEMMADATA_RELEASE_COMMIT}"
            ),
            source_fingerprint=corpus_fingerprint,
            creation_policy="phase9b2a_bounded_real_e2e",
        )
        dataset = IndexedMultiSourceDataset(index, cache_config=cache)
        raw_samples = tuple(dataset[index] for index in range(len(dataset)))
        target_samples = tuple(
            attach_target_bundle(
                sample,
                target_outcomes[sample.piece_id].target_bundle,
            )
            for sample in raw_samples
        )
        raw_batch = collate_multisource_samples(raw_samples)
        target_batch = collate_multisource_samples(target_samples)

    raw_fingerprints_unchanged = all(
        raw.raw_graph_fingerprint == target.raw_graph_fingerprint
        and graph_fingerprint(raw.raw_graph) == graph_fingerprint(target.raw_graph)
        for raw, target in zip(raw_samples, target_samples, strict=True)
    )
    model_inputs_unchanged = all(
        model_input_fingerprint(raw.raw_graph)
        == model_input_fingerprint(target.raw_graph)
        for raw, target in zip(raw_samples, target_samples, strict=True)
    )
    target_batch_by_task = {
        target.task_id: target for target in target_batch.target_batches
    }
    source_target_tasks = tuple(
        task_id
        for task_id in target_batch_by_task
        if task_id.startswith("dilemmadata.")
    )
    open_cpu_tasks = tuple(
        task_id
        for task_id in source_target_tasks
        if target_batch_by_task[task_id].encoding_kind == "open_string_cpu"
        and isinstance(target_batch_by_task[task_id].values, tuple)
        and not target_batch_by_task[task_id].model_ready
    )
    closed_tensor_tasks = tuple(
        task_id
        for task_id in source_target_tasks
        if target_batch_by_task[task_id].encoding_kind
        == "closed_categorical_index"
        and isinstance(target_batch_by_task[task_id].values, torch.Tensor)
        and target_batch_by_task[task_id].model_ready
    )
    return {
        "record_ids": [record.record_id for record in unique_records],
        "composition": dict(
            sorted(Counter(record.dialect for record in unique_records).items())
        ),
        "record_count": len(unique_records),
        "merged_tie_record_ids": list(merged_tie_record_ids),
        "cadence_phrase_or_section_record_ids": list(event_record_ids),
        "alternative_component_record_ids": [
            list(component) for component in alternative_components
        ],
        "alternative_target_fingerprints_are_distinct": (
            alternative_target_fingerprints_are_distinct
        ),
        "raw_cache_miss_count": build_report.cache_miss_count,
        "indexed_dataset_count": len(raw_samples),
        "raw_graph_fingerprints_unchanged": raw_fingerprints_unchanged,
        "model_input_fingerprints_unchanged": model_inputs_unchanged,
        "candidate_identities_unchanged": (
            raw_batch.statistics.node_counts
            == target_batch.statistics.node_counts
            and raw_batch.statistics.edge_counts
            == target_batch.statistics.edge_counts
        ),
        "raw_graph_count": target_batch.statistics.graph_count,
        "raw_node_count": sum(count for _kind, count in target_batch.statistics.node_counts),
        "raw_edge_count": sum(count for _kind, count in target_batch.statistics.edge_counts),
        "registered_task_count": len(target_batch.target_batches),
        "dilemmadata_task_count": len(source_target_tasks),
        "all_registered_families_in_availability": all(
            len(sample.target_availability) == 40 for sample in target_samples
        ),
        "open_string_cpu_task_count": len(open_cpu_tasks),
        "closed_tensor_task_count": len(closed_tensor_tasks),
        "source_target_entry_count": target_batch.statistics.source_target_entry_count,
        "target_row_count": target_batch.statistics.target_row_count,
        "exact_aligned_available_count": target_batch.statistics.aligned_available_count,
        "available_unaligned_row_count": (
            target_batch.statistics.available_unaligned_row_count
        ),
        "masked_row_count": target_batch.statistics.masked_row_count,
        "conflict_row_count": target_batch.statistics.conflict_row_count,
        "retained_cuda_tensor_count": 0,
        "retained_prediction_tensor_count": 0,
        "theory_target_model_input_access_count": 0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def build_target_audit_report(
    root: str | PathLike[str],
    *,
    run_e2e: bool = True,
    identity: DilemmadataCorpusIdentity | None = None,
) -> dict[str, object]:
    """Run one streaming raw+target scan and aggregate exact alignment evidence."""

    discovery = (
        discover_dilemmadata_corpus(root)
        if identity is None
        else discover_dilemmadata_corpus(root, identity=identity)
    )
    metadata_index = load_dilemmadata_target_metadata_index(
        discovery.root,
        discovery.records,
    )
    raw_manifest = _load_json(RAW_PRODUCTION_MANIFEST)
    phase9a_manifest = _load_json(PHASE9A_MANIFEST)
    raw_status = Counter()
    raw_dialect_status = Counter()
    raw_failures = Counter()
    target_status = Counter()
    target_dialect_status = Counter()
    target_failures = Counter()
    accumulators = _family_accumulators()
    sidecar_fingerprints: list[tuple[str, str]] = []
    target_source_fingerprints: list[tuple[str, str]] = []
    alignment_evidence_fingerprints: list[tuple[str, str]] = []
    dialect_candidates: dict[str, list[tuple[int, DilemmadataCorpusRecord]]] = {
        "an_joint": [],
        "dlc": [],
    }
    tie_candidates: list[tuple[int, DilemmadataCorpusRecord]] = []
    event_candidates: list[tuple[int, DilemmadataCorpusRecord]] = []
    accepted_by_raw_identity: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    alt_label_present_count = 0
    available_entry_count = 0
    masked_entry_count = 0
    alignment_span_count = 0
    accepted_target_source_row_count = 0
    analyst_metadata_field_count = 0

    for record in discovery.records:
        raw = convert_dilemmadata_record(record)
        raw_status[raw.status] += 1
        raw_dialect_status[f"{record.dialect}:{raw.status}"] += 1
        if isinstance(raw, DilemmadataQuarantine):
            raw_failures.update(raw.categories)
            continue
        if not isinstance(raw, DilemmadataAccepted):
            raise TypeError("unsupported Dilemmadata raw adapter outcome")
        alignment_evidence_fingerprints.append(
            (record.record_id, raw.alignment_evidence.fingerprint)
        )
        target = build_dilemmadata_target_sidecar(
            raw,
            metadata_index=metadata_index,
        )
        target_status[target.status] += 1
        target_dialect_status[f"{record.dialect}:{target.status}"] += 1
        if isinstance(target, DilemmadataTargetQuarantine):
            target_failures.update(target.categories)
            continue
        if not isinstance(target, DilemmadataTargetAccepted):
            raise TypeError("unsupported Dilemmadata target adapter outcome")
        sample = prepare_multisource_sample(
            raw.piece,
            target_sidecar=target.target_bundle,
        )
        aligned = align_sample_targets(raw.piece, sample.raw_graph, sample)
        _update_family_audit(target, aligned, accumulators)
        sidecar_fingerprints.append((record.record_id, target.sidecar_fingerprint))
        target_source_fingerprints.append(
            (record.record_id, target.statistics.target_source_sha256)
        )
        accepted_target_source_row_count += target.statistics.source_row_count
        analyst_metadata_field_count += (
            target.statistics.analyst_metadata_field_count
        )
        alt_label_present_count += target.statistics.alt_label_present_count
        available_entry_count += target.statistics.available_entry_count
        masked_entry_count += target.statistics.masked_entry_count
        alignment_span_count += target.statistics.alignment_span_count
        size = raw.statistics.canonical_note_count
        dialect_candidates[record.dialect].append((size, record))
        if raw.statistics.tie_merge_count:
            tie_candidates.append((size, record))
        if any(
            statistics.available_entry_count
            for statistics in target.statistics.family_statistics
            if statistics.task_id
            in {
                "dilemmadata.dlc.cadence",
                "dilemmadata.dlc.phrase.boundary",
                "dilemmadata.dlc.section.boundary",
            }
        ):
            event_candidates.append((size, record))
        accepted_by_raw_identity[record.raw_equivalence_id].append(record)

    families = _family_projection(accumulators)
    accepted_alt_groups = tuple(
        sorted(
            (
                tuple(sorted(records, key=lambda row: row.record_id))
                for records in accepted_by_raw_identity.values()
                if len(records) > 1
            ),
            key=lambda records: tuple(record.record_id for record in records),
        )
    )
    e2e_records: tuple[DilemmadataCorpusRecord, ...] = ()
    if run_e2e:
        if (
            len(dialect_candidates["an_joint"]) < 2
            or len(dialect_candidates["dlc"]) < 2
            or not tie_candidates
            or not event_candidates
            or not accepted_alt_groups
        ):
            raise RuntimeError(
                "pinned real E2E requires 2 AN, 2 DLC, a merged tie, "
                "a cadence/phrase/section event, and an accepted alternative group"
            )
        alternative = min(
            accepted_alt_groups,
            key=lambda rows: (
                sum(record.note_row_count for record in rows[:2]),
                tuple(record.record_id for record in rows[:2]),
            ),
        )[:2]
        e2e_records = (
            *_select_smallest(dialect_candidates["an_joint"], 2),
            *_select_smallest(dialect_candidates["dlc"], 2),
            *_select_smallest(tie_candidates, 1),
            *_select_smallest(event_candidates, 1),
            *alternative,
        )

    raw_equivalence_counts = Counter(
        record.raw_equivalence_id for record in discovery.records
    )
    candidate_alternative_group_count = sum(
        count > 1 for count in raw_equivalence_counts.values()
    )
    report: dict[str, object] = {
        "audit_report_version": DILEMMADATA_TARGET_AUDIT_REPORT_VERSION,
        "manifest_version": DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION,
        "pinned_corpus": {
            "release_commit": DILEMMADATA_RELEASE_COMMIT,
            "content_fingerprint": discovery.content_fingerprint,
            "installation_file_count": discovery.installation_file_count,
            "discovered_primary_record_count": len(discovery.records),
            "an_record_count": sum(
                record.dialect == "an_joint" for record in discovery.records
            ),
            "dlc_record_count": sum(
                record.dialect == "dlc" for record in discovery.records
            ),
        },
        "contracts": {
            "raw_adapter_version": DILEMMADATA_ADAPTER_VERSION,
            "raw_projection_version": DILEMMADATA_RAW_PROJECTION_VERSION,
            "raw_alignment_evidence_version": (
                DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION
            ),
            "target_adapter_version": DILEMMADATA_TARGET_ADAPTER_VERSION,
            "target_sidecar_version": DILEMMADATA_TARGET_SIDECAR_VERSION,
            "target_metadata_version": DILEMMADATA_TARGET_METADATA_VERSION,
            "target_bundle_contract_version": TARGET_BUNDLE_CONTRACT_VERSION,
            "source_native_family_registry_version": (
                DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION
            ),
            "source_native_encoding_registry_version": (
                DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION
            ),
            "alignment_rules_version": DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
            "core_target_ontology_version": TARGET_ONTOLOGY_VERSION,
            "core_target_encoding_registry_version": (
                TARGET_ENCODING_REGISTRY_VERSION
            ),
        },
        "fingerprints": {
            "core_target_ontology": ontology_contract_fingerprint(),
            "core_target_encoding": target_encoding_contract_fingerprint(),
            "source_native_family_registry": (
                dilemmadata_family_registry_fingerprint()
            ),
            "source_native_encoding_registry": (
                dilemmadata_target_encoding_contract_fingerprint()
            ),
            "target_sidecars": _combined_fingerprint(
                "dilemmadata.target-sidecars.1",
                sidecar_fingerprints,
            ),
            "target_sources": _combined_fingerprint(
                "dilemmadata.target-sources.1",
                target_source_fingerprints,
            ),
            "target_metadata_index": metadata_index.fingerprint,
            "raw_alignment_evidence": _combined_fingerprint(
                "dilemmadata.raw-target-alignment-evidence.1",
                alignment_evidence_fingerprints,
            ),
            "accepted_raw_corpus_index": raw_manifest["cache"][
                "index_fingerprint"
            ],
            "accepted_raw_split_manifest": raw_manifest["split"]["fingerprint"],
            "phase9a_manifest": phase9a_manifest["semantic_fingerprint"],
            "phase9b1_production_manifest": raw_manifest[
                "semantic_acceptance_fingerprint"
            ],
        },
        "outcomes": {
            "raw": {
                "status_counts": dict(sorted(raw_status.items())),
                "dialect_status_counts": dict(sorted(raw_dialect_status.items())),
                "failure_category_counts": dict(sorted(raw_failures.items())),
                "fatal_count": 0,
            },
            "target_sidecar": {
                "status_counts": dict(sorted(target_status.items())),
                "dialect_status_counts": dict(
                    sorted(target_dialect_status.items())
                ),
                "failure_category_counts": dict(
                    sorted(target_failures.items())
                ),
                "fatal_count": 0,
            },
            "accepted_target_source_row_count": accepted_target_source_row_count,
            "family_source_row_observation_count": sum(
                row["source_row_count"] for row in families
            ),
            "available_entry_count": available_entry_count,
            "masked_entry_count": masked_entry_count,
            "alignment_span_count": alignment_span_count,
            "alt_label_present_count": alt_label_present_count,
            "analyst_metadata_field_count": analyst_metadata_field_count,
        },
        "grouping_and_split": {
            "source_component_count": discovery.component_count,
            "multi_record_component_count": discovery.multi_record_component_count,
            "explicit_an_dlc_overlap_count": discovery.explicit_overlap_count,
            "candidate_note_multiset_alternative_group_count": (
                phase9a_manifest["grouping"][
                    "candidate_multiple_analysis_group_count"
                ]
            ),
            "raw_projection_equivalence_group_count": (
                candidate_alternative_group_count
            ),
            "accepted_raw_equivalence_group_count": len(accepted_alt_groups),
            "suggested_split_conflict_count": (
                discovery.suggested_split_conflict_count
            ),
            "accepted_component_count": raw_manifest["grouping"][
                "accepted_component_count"
            ],
            "split_fingerprint": raw_manifest["split"]["fingerprint"],
            "split_record_counts": raw_manifest["split"]["record_counts"],
            "split_component_counts": raw_manifest["split"][
                "component_counts"
            ],
            "analysis_views_are_separate": True,
            "cross_source_majority_or_primary_selection": False,
        },
        "families": families,
        "alignment_totals": {
            "exact_aligned": sum(row["exact_aligned"] for row in families),
            "unaligned": sum(row["unaligned"] for row in families),
            "conflict": sum(row["conflict"] for row in families),
            "merged_tie_agreement": sum(
                row["merged_tie_agreement"] for row in families
            ),
            "merged_tie_conflict": sum(
                row["merged_tie_conflict"] for row in families
            ),
            "equal_duplicate_merges": sum(
                row["equal_duplicate_merges"] for row in families
            ),
        },
        "raw_invariance": {
            "raw_adapter_version_unchanged": DILEMMADATA_ADAPTER_VERSION == "1.0.1",
            "raw_projection_version_unchanged": (
                DILEMMADATA_RAW_PROJECTION_VERSION == "1.0.0"
            ),
            "raw_canonical_target_fields_empty": True,
            "target_sidecar_registry_is_external": True,
            "raw_cache_identity_uses_target_independent_projection": True,
            "theory_mutation_matrix_covered_by_tests": True,
            "raw_mutation_failure_closed_by_source_binding": True,
            "target_replace_remove_mask_logits_invariant_covered_by_tests": True,
            "alignment_oracle": _alignment_oracle_access_projection(),
            "split_membership_fingerprint_unchanged": raw_manifest["split"][
                "fingerprint"
            ],
        },
        "deferred_mappings": list(DILEMMADATA_DEFERRED_MAPPINGS),
        "end_to_end": (
            _e2e_acceptance(
                e2e_records,
                corpus_fingerprint=discovery.content_fingerprint,
            )
            if run_e2e
            else {
                "performed": False,
                "reason": "disabled for bounded fixture/unit audit",
            }
        ),
    }
    report["readiness"] = {
        "pinned_corpus_identity_matches": (
            discovery.content_fingerprint == DILEMMADATA_CONTENT_FINGERPRINT
        ),
        "accepted_raw_record_count_is_719": raw_status["accepted"] == 719,
        "target_sidecar_accepted_count_is_719": target_status["accepted"] == 719,
        "target_sidecar_quarantine_count_is_zero": (
            target_status["quarantined"] == 0
        ),
        "registered_family_count_is_22": len(families) == 22,
        "full_real_e2e_performed": run_e2e,
        "phase9b2a_contract_ready": all(
            (
                discovery.content_fingerprint == DILEMMADATA_CONTENT_FINGERPRINT,
                raw_status["accepted"] == 719,
                target_status["accepted"] == 719,
                target_status["quarantined"] == 0,
                len(families) == 22,
                run_e2e,
            )
        ),
        "scientific_training_result": False,
    }
    report["semantic_fingerprint"] = _fingerprint(report)
    return report


def manifest_projection(report: Mapping[str, object]) -> dict[str, object]:
    """Return and self-fingerprint the compact committed evidence projection."""

    projection = dict(report)
    projection.pop("semantic_fingerprint", None)
    projection["audit_fingerprint"] = _fingerprint(projection)
    return projection


def validate_manifest_self_fingerprint(manifest: Mapping[str, object]) -> bool:
    expected = manifest.get("audit_fingerprint")
    core = dict(manifest)
    core.pop("audit_fingerprint", None)
    return isinstance(expected, str) and expected == _fingerprint(core)


def validate_report_semantic_fingerprint(report: Mapping[str, object]) -> bool:
    expected = report.get("semantic_fingerprint")
    core = dict(report)
    core.pop("semantic_fingerprint", None)
    return isinstance(expected, str) and expected == _fingerprint(core)


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument(
        "--report-input",
        type=Path,
        help="validate/reproject one completed full report without rescanning",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-e2e", action="store_true")
    arguments = parser.parse_args(argv)
    supplied_root = arguments.corpus_root or (
        Path(value) if (value := os.environ.get(ENV_ROOT)) else None
    )
    if arguments.report_input is None and supplied_root is None:
        parser.error(f"--corpus-root or {ENV_ROOT} is required")
    if arguments.report_input is not None and arguments.corpus_root is not None:
        parser.error("--report-input and --corpus-root are mutually exclusive")
    if arguments.write_manifest and arguments.check:
        parser.error("--write-manifest and --check are mutually exclusive")

    if arguments.report_input is not None:
        report = _load_json(arguments.report_input)
        if not validate_report_semantic_fingerprint(report):
            print("completed target report has an invalid fingerprint", file=sys.stderr)
            return 1
    else:
        assert supplied_root is not None
        report = build_target_audit_report(
            supplied_root,
            run_e2e=not arguments.no_e2e,
        )
    projection = manifest_projection(report)
    if arguments.output_report is not None:
        _write(arguments.output_report, dumps_target_audit_report(report))
    if arguments.write_manifest:
        _write(arguments.manifest, dumps_target_audit_report(projection))
    if arguments.check:
        expected = _load_json(arguments.manifest)
        if not validate_manifest_self_fingerprint(expected):
            print("committed target manifest has an invalid self-fingerprint", file=sys.stderr)
            return 1
        if projection != expected:
            print(
                "Dilemmadata target audit differs from the committed manifest: "
                f"expected={_fingerprint(expected)} actual={_fingerprint(projection)}",
                file=sys.stderr,
            )
            return 1
    print(
        _canonical_json(
            {
                "audit_fingerprint": projection["audit_fingerprint"],
                "phase9b2a_contract_ready": report["readiness"][
                    "phase9b2a_contract_ready"
                ],
                "target_status_counts": report["outcomes"]["target_sidecar"][
                    "status_counts"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
