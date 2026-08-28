#!/usr/bin/env python3
"""Deterministic Phase 9E-A Dilemmadata common-harmony audit and check."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from music_critic.adapters.dilemmadata import (
    DilemmadataAccepted,
    DilemmadataAdapterConfig,
    DilemmadataCorpusIdentity,
    DilemmadataCorpusRecord,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
)
from music_critic.adapters.dilemmadata_targets import (
    DilemmadataTargetAccepted,
    build_dilemmadata_target_sidecar,
    load_dilemmadata_target_metadata_index,
)
from music_critic.data import dumps_piece
from music_critic.graph import build_raw_graph, graph_fingerprint, model_input_fingerprint
from music_critic.tasks.dilemmadata_common import (
    ANALYSISGNN_REFERENCE,
    COMMON_BASS_PC_TASK,
    COMMON_INVERSION_TASK,
    COMMON_LOCAL_KEY_TASK,
    COMMON_PITCH_CLASS_SET_TASK,
    COMMON_QUALITY_TASK,
    COMMON_ROOT_PC_TASK,
    DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION,
    DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION,
    DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION,
    DILEMMADATA_COMMON_HARMONIC_REGISTRY,
    DilemmadataCommonAuditFact,
    DilemmadataCommonCollapseEvidence,
    DilemmadataCommonHarmonicAuditManifest,
    DilemmadataCommonHarmonicAuditReport,
    DilemmadataCommonInvariantEvidence,
    DilemmadataCommonOverlapEvidence,
    _bound_supplemental_source_evidence,
    dumps_dilemmadata_common_audit_manifest,
    dumps_dilemmadata_common_audit_report,
    loads_dilemmadata_common_audit_manifest,
    make_dilemmadata_common_audit_manifest,
    make_dilemmadata_common_audit_report,
    map_dilemmadata_common_inversion,
    map_dilemmadata_common_pitch_class,
    map_dilemmadata_common_quality,
    project_dilemmadata_common_harmony,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_FAMILIES,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
)
from music_critic.tasks.multisource import target_bundle_fingerprint


ENV_ROOT = "MUSIC_CRITIC_DILEMMADATA_ROOT"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "common_harmonic_manifest.json"
)
RAW_PRODUCTION_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "production_manifest.json"
)
SOURCE_TARGET_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "target_manifest.json"
)
_MISSING = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
_TRUE = frozenset({"1", "True", "true", "TRUE"})
_FALSE = frozenset({"0", "False", "false", "FALSE"})
_COMMON_SOURCE_TASKS = frozenset(
    source_task
    for family in DILEMMADATA_COMMON_HARMONIC_REGISTRY.families
    for source_task in family.source_task_ids
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


def _combined_fingerprint(domain: str, rows: Iterable[tuple[str, str]]) -> str:
    return _fingerprint({"domain": domain, "rows": [list(row) for row in sorted(rows)]})


def _current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _component_fingerprint(records: Sequence[DilemmadataCorpusRecord]) -> str:
    return _fingerprint(
        [
            {
                "dataset_id": "dilemmadata",
                "piece_id": row.piece_id,
                "source_group_id": row.source_group_id,
                "lineage_group_id": row.lineage_group_id,
            }
            for row in sorted(records, key=lambda item: item.piece_id)
        ]
    )


def _largest_remainder(total: int) -> dict[str, int]:
    ratios = {"test": Decimal("0.1"), "train": Decimal("0.8"), "validation": Decimal("0.1")}
    exact = {key: Decimal(total) * value for key, value in ratios.items()}
    quotas = {
        key: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for key, value in exact.items()
    }
    remaining = total - sum(quotas.values())
    order = sorted(quotas, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _accepted_split_assignments(
    records: Sequence[DilemmadataCorpusRecord],
) -> dict[str, str]:
    components: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    # Discovery source groups are already transitive across explicit overlap,
    # same-input alternatives, and lineage.  Retaining the full source-group
    # token also keeps accepted fragments of a quarantined component atomic.
    for record in records:
        components[record.source_group_id].append(record)
    component_rows = tuple(
        tuple(sorted(rows, key=lambda row: row.piece_id))
        for _group, rows in sorted(components.items())
    )
    quotas = _largest_remainder(len(component_rows))
    schedule = tuple(
        split for split in sorted(quotas) for _index in range(quotas[split])
    )
    ordered = sorted(
        component_rows,
        key=lambda rows: (
            _fingerprint(
                {
                    "component": _component_fingerprint(rows),
                    "seed": 9_001,
                }
            ),
            tuple(row.piece_id for row in rows),
        ),
    )
    return {
        record.piece_id: split
        for rows, split in zip(ordered, schedule, strict=True)
        for record in rows
    }


def _fact(
    name: str,
    fact_value: str | int | bool,
    **dimensions: str,
) -> DilemmadataCommonAuditFact:
    return DilemmadataCommonAuditFact(
        name=name,
        dimensions=tuple(sorted(dimensions.items())),
        value=fact_value,
    )


def _gate_state(raw: str | None) -> bool | None:
    if raw is None or raw.strip() in _MISSING:
        return None
    if raw.strip() in _TRUE:
        return True
    if raw.strip() in _FALSE:
        return False
    return None


def _row_mapping_state(
    task_id: str,
    value: str,
    row: Mapping[str, str],
) -> str:
    if task_id.endswith("chord.quality"):
        return map_dilemmadata_common_quality(task_id, value).state
    if task_id.endswith("chord.inversion"):
        return map_dilemmadata_common_inversion(task_id, value).state
    if task_id.endswith("chord.root"):
        spelling = row.get("a_root") or row.get("root")
        return map_dilemmadata_common_pitch_class(
            task_id, value, source_spelling=spelling if spelling else None
        ).state
    if task_id.endswith("chord.bass"):
        spelling = row.get("a_bass") or row.get("bass_note")
        return map_dilemmadata_common_pitch_class(
            task_id, value, source_spelling=spelling if spelling else None
        ).state
    if task_id.endswith("key.local"):
        if task_id.startswith("dilemmadata.an."):
            return (
                "exact"
                if map_dilemmadata_common_pitch_class(
                    "dilemmadata.an.chord.root", value
                ).state
                == "exact"
                else "unsupported"
            )
        raw_tpc = row.get("localkey_tpc", "").strip()
        if raw_tpc in _MISSING:
            return "ambiguous"
        try:
            int(raw_tpc)
        except ValueError:
            return "invalid"
        return "exact"
    return "unsupported"


def _full_source_mapping_counts(
    records: Sequence[DilemmadataCorpusRecord],
    raw_status: Mapping[str, str],
    splits: Mapping[str, str],
) -> tuple[Counter[tuple[str, str, str, str, str]], Counter[tuple[str, str, str, str]]]:
    states: Counter[tuple[str, str, str, str, str]] = Counter()
    classes: Counter[tuple[str, str, str, str]] = Counter()
    for record in records:
        specs = tuple(
            spec
            for spec in DILEMMADATA_SOURCE_FAMILIES
            if spec.dialect == record.dialect and spec.task_id in _COMMON_SOURCE_TASKS
        )
        try:
            handle = record.path.open("r", encoding="utf-8", newline="")
        except (OSError, UnicodeError):
            for spec in specs:
                states[(spec.task_id, record.dialect, raw_status[record.record_id], "unassigned", "invalid")] += 1
            continue
        split = splits.get(record.piece_id, "unassigned")
        with handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                for spec in specs:
                    if spec.primary_field not in row:
                        state = "unsupported"
                        value = ""
                    else:
                        gate = True
                        if spec.gate_field is not None:
                            gate = _gate_state(row.get(spec.gate_field))
                        value = row.get(spec.primary_field, "").strip()
                        if gate is not True:
                            state = "masked"
                        elif value in _MISSING:
                            state = "missing"
                        else:
                            state = _row_mapping_state(spec.task_id, value, row)
                    states[(spec.task_id, record.dialect, raw_status[record.record_id], split, state)] += 1
                    if value not in _MISSING and state in {"exact", "coarsened"}:
                        classes[(spec.task_id, record.dialect, split, value)] += 1
    return states, classes


def _value_token(value: object) -> str:
    if value is None:
        return "<unavailable>"
    if hasattr(value, "tonic_pc") and hasattr(value, "mode"):
        return f"{value.tonic_pc}:{value.mode}"
    if isinstance(value, tuple):
        return "{" + ",".join(str(item) for item in value) + "}"
    return str(value)


def _common_counts(
    projections: Mapping[str, object],
    records: Mapping[str, DilemmadataCorpusRecord],
    splits: Mapping[str, str],
) -> tuple[
    Counter[tuple[str, str, str, str]],
    Counter[tuple[str, str, str, str]],
]:
    states: Counter[tuple[str, str, str, str]] = Counter()
    classes: Counter[tuple[str, str, str, str]] = Counter()
    for piece_id, projection in projections.items():
        record = records[piece_id]
        split = splits[piece_id]
        for target in projection.targets:
            for entry in target.entries:
                states[(target.task_id, record.dialect, split, entry.state)] += 1
                if entry.state in {"exact", "coarsened"}:
                    classes[(target.task_id, record.dialect, split, _value_token(entry.common_value))] += 1
    return states, classes


def _imbalance_facts(
    facts: list[DilemmadataCommonAuditFact],
    classes: Mapping[tuple[str, str, str, str], int],
    *,
    space: str,
) -> None:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (task, dialect, split, _value), count in classes.items():
        if split == "train" and count > 0:
            grouped[(task, dialect)].append(count)
    for (task, dialect), counts in sorted(grouped.items()):
        facts.append(
            _fact(
                "train_support_min",
                min(counts),
                space=space,
                task=task,
                dialect=dialect,
            )
        )
        facts.append(
            _fact(
                "train_support_max",
                max(counts),
                space=space,
                task=task,
                dialect=dialect,
            )
        )
        facts.append(
            _fact(
                "train_imbalance_ratio_microunits",
                int(round(max(counts) / min(counts) * 1_000_000)),
                space=space,
                task=task,
                dialect=dialect,
            )
        )


def _collapse_table() -> tuple[DilemmadataCommonCollapseEvidence, ...]:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in DILEMMADATA_COMMON_HARMONIC_REGISTRY.quality_mapping_rows:
        if row.common_value is not None:
            grouped[str(row.common_value)].append(
                (row.dialect, row.source_value, row.state)
            )
    return tuple(
        DilemmadataCommonCollapseEvidence(
            common_value=value,
            source_rows=tuple(sorted(rows)),
        )
        for value, rows in sorted(grouped.items())
        if len(rows) > 1
    )


def _parity_rows() -> tuple[tuple[str, str, str, str, str], ...]:
    """Return parity rows keyed by dialect-specific source task and value."""

    rows = tuple(
        sorted(
            (
                row.source_task_id,
                row.source_value,
                row.analysisgnn_reference_value or "<not-applicable>",
                str(row.common_value) if row.common_value is not None else "<masked>",
                row.analysisgnn_agreement,
            )
            for row in (
                *DILEMMADATA_COMMON_HARMONIC_REGISTRY.quality_mapping_rows,
                *DILEMMADATA_COMMON_HARMONIC_REGISTRY.inversion_mapping_rows,
            )
        )
    )
    identities = tuple((task_id, source_value) for task_id, source_value, *_ in rows)
    if identities != tuple(sorted(set(identities))):
        raise RuntimeError(
            "AnalysisGNN parity rows must have unique source-task/value identity"
        )
    return rows


def _projection_target(projection: object, task_id: str) -> object:
    return next(target for target in projection.targets if target.task_id == task_id)


def _target_signature(target: object) -> tuple[tuple[str, str, tuple[str | None, ...]], ...]:
    return tuple(
        sorted(
            (
                entry.state,
                _value_token(entry.common_value),
                entry.source_values,
            )
            for entry in target.entries
        )
    )


def _overlap_evidence(
    records: Sequence[DilemmadataCorpusRecord],
    projections: Mapping[str, object],
) -> tuple[DilemmadataCommonOverlapEvidence, ...]:
    groups: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    for record in records:
        groups[f"source:{record.source_group_id}"].append(record)
        groups[f"alternative:{record.grouping_fingerprint}"].append(record)
    rows: list[DilemmadataCommonOverlapEvidence] = []
    families = (
        COMMON_QUALITY_TASK,
        COMMON_INVERSION_TASK,
        COMMON_ROOT_PC_TASK,
        COMMON_BASS_PC_TASK,
        COMMON_LOCAL_KEY_TASK,
        COMMON_PITCH_CLASS_SET_TASK,
    )
    for component_id, members in sorted(groups.items()):
        unique = tuple(sorted({row.record_id: row for row in members}.values(), key=lambda row: row.record_id))
        if len(unique) < 2:
            continue
        # Source groups are reported when they cross dialects or contain an
        # accepted alternative view; the conservative note-event multiset
        # grouping fingerprint covers all 30 Phase 9A same-input candidate
        # groups independently of the narrower raw-projection equivalence.
        if component_id.startswith("source:") and len({row.dialect for row in unique}) < 2:
            continue
        record_ids = tuple(row.record_id for row in unique)
        available = tuple(
            projections[row.piece_id]
            for row in unique
            if row.piece_id in projections
        )
        for family in families:
            if len(available) < 2:
                rows.append(
                    DilemmadataCommonOverlapEvidence(
                        component_id=component_id,
                        record_ids=record_ids,
                        family=family,
                        comparison="unavailable",
                        left_value=None,
                        right_value=None,
                    )
                )
                continue
            left = _target_signature(_projection_target(available[0], family))
            right = _target_signature(_projection_target(available[1], family))
            common_left = tuple((state, value) for state, value, _source in left)
            common_right = tuple((state, value) for state, value, _source in right)
            source_left = tuple(source for _state, _value, source in left)
            source_right = tuple(source for _state, _value, source in right)
            if common_left == common_right and source_left == source_right:
                comparison = "exact_agreement"
            elif common_left == common_right and family in {
                COMMON_ROOT_PC_TASK,
                COMMON_BASS_PC_TASK,
                COMMON_LOCAL_KEY_TASK,
            }:
                comparison = "enharmonic_only_agreement"
            elif common_left == common_right and any(
                state == "coarsened" for state, _value in common_left
            ):
                comparison = "coarsened_agreement"
            else:
                comparison = "conflict"
            rows.append(
                DilemmadataCommonOverlapEvidence(
                    component_id=component_id,
                    record_ids=record_ids,
                    family=family,
                    comparison=comparison,
                    left_value=_fingerprint(left),
                    right_value=_fingerprint(right),
                )
            )
    return tuple(
        sorted(rows, key=lambda row: (row.component_id, row.family, row.record_ids))
    )


def _invariant(
    name: str,
    before: Iterable[tuple[str, str]],
    after: Iterable[tuple[str, str]],
) -> DilemmadataCommonInvariantEvidence:
    before_fingerprint = _combined_fingerprint(f"{name}.before", before)
    after_fingerprint = _combined_fingerprint(f"{name}.before", after)
    return DilemmadataCommonInvariantEvidence(
        name=name,
        before_fingerprint=before_fingerprint,
        after_fingerprint=after_fingerprint,
        unchanged=before_fingerprint == after_fingerprint,
    )


def build_common_audit_report(
    root: str | os.PathLike[str],
    *,
    identity: DilemmadataCorpusIdentity = DilemmadataCorpusIdentity(),
    raw_config: DilemmadataAdapterConfig = DilemmadataAdapterConfig(),
    base_git_sha: str | None = None,
    run_graph_invariance: bool = True,
) -> DilemmadataCommonHarmonicAuditReport:
    """Build full-source and accepted-runtime common projection evidence."""

    discovery = discover_dilemmadata_corpus(root, identity=identity, require_valid=True)
    metadata_index = load_dilemmadata_target_metadata_index(
        discovery.root,
        discovery.records,
    )
    raw_outcomes: dict[str, object] = {}
    raw_status: dict[str, str] = {}
    accepted_raw: dict[str, DilemmadataAccepted] = {}
    target_outcomes: dict[str, DilemmadataTargetAccepted] = {}
    projections: dict[str, object] = {}
    raw_records_by_piece: dict[str, DilemmadataCorpusRecord] = {}

    canonical_before: list[tuple[str, str]] = []
    canonical_after: list[tuple[str, str]] = []
    graph_before: list[tuple[str, str]] = []
    graph_after: list[tuple[str, str]] = []
    model_before: list[tuple[str, str]] = []
    model_after: list[tuple[str, str]] = []
    target_before: list[tuple[str, str]] = []
    target_after: list[tuple[str, str]] = []
    grouping_before: list[tuple[str, str]] = []
    grouping_after: list[tuple[str, str]] = []

    for record in discovery.records:
        outcome = convert_dilemmadata_record(record, config=raw_config)
        raw_outcomes[record.record_id] = outcome
        raw_status[record.record_id] = outcome.status
        if not isinstance(outcome, DilemmadataAccepted):
            continue
        accepted_raw[record.piece_id] = outcome
        raw_records_by_piece[record.piece_id] = record
        target = build_dilemmadata_target_sidecar(
            outcome,
            metadata_index=metadata_index,
        )
        if not isinstance(target, DilemmadataTargetAccepted):
            raise RuntimeError(
                f"accepted raw record failed target binding: {record.record_id}"
            )
        target_outcomes[record.piece_id] = target
        canonical = sha256(dumps_piece(outcome.piece).encode("utf-8")).hexdigest()
        source_target = target_bundle_fingerprint(target.target_bundle)
        canonical_before.append((record.piece_id, canonical))
        target_before.append((record.piece_id, source_target))
        grouping_before.append((record.piece_id, record.grouping_fingerprint))
        if run_graph_invariance:
            graph = build_raw_graph(outcome.piece)
            graph_before.append((record.piece_id, graph_fingerprint(graph)))
            model_before.append((record.piece_id, model_input_fingerprint(graph)))
        supplemental = _bound_supplemental_source_evidence(outcome, target)
        projection = project_dilemmadata_common_harmony(
            target.target_bundle,
            supplemental_source_evidence=supplemental,
        )
        projections[record.piece_id] = projection
        canonical_after.append(
            (
                record.piece_id,
                sha256(dumps_piece(outcome.piece).encode("utf-8")).hexdigest(),
            )
        )
        target_after.append(
            (record.piece_id, target_bundle_fingerprint(target.target_bundle))
        )
        grouping_after.append((record.piece_id, record.grouping_fingerprint))
        if run_graph_invariance:
            graph_after.append((record.piece_id, graph_fingerprint(graph)))
            model_after.append((record.piece_id, model_input_fingerprint(graph)))

    split_assignments = _accepted_split_assignments(
        tuple(row.record for row in accepted_raw.values())
    )
    full_states, source_classes = _full_source_mapping_counts(
        discovery.records,
        raw_status,
        split_assignments,
    )
    common_states, common_classes = _common_counts(
        projections,
        raw_records_by_piece,
        split_assignments,
    )
    facts: list[DilemmadataCommonAuditFact] = [
        _fact("source_record_count", len(discovery.records)),
        _fact("source_component_count", discovery.component_count),
        _fact("source_explicit_overlap_count", discovery.explicit_overlap_count),
        _fact("source_multi_record_component_count", discovery.multi_record_component_count),
        _fact("accepted_projection_count", len(projections)),
        _fact("quarantined_raw_count", sum(value == "quarantined" for value in raw_status.values())),
        _fact("analysis_view_count", len(target_outcomes)),
        _fact("test_targets_used_for_model_evaluation", False),
    ]
    split_counts = Counter(split_assignments.values())
    for split, count in sorted(split_counts.items()):
        facts.append(_fact("accepted_split_record_count", count, split=split))
    for (task, dialect, status, split, state), count in sorted(full_states.items()):
        facts.append(
            _fact(
                "full_source_mapping_state_count",
                count,
                task=task,
                dialect=dialect,
                raw_status=status,
                split=split,
                state=state,
            )
        )
    for (task, dialect, split, state), count in sorted(common_states.items()):
        facts.append(
            _fact(
                "accepted_common_mapping_state_count",
                count,
                task=task,
                dialect=dialect,
                split=split,
                state=state,
            )
        )
    for (task, dialect, split, value), count in sorted(source_classes.items()):
        facts.append(
            _fact(
                "source_native_class_count",
                count,
                task=task,
                dialect=dialect,
                split=split,
                value=value,
            )
        )
    for (task, dialect, split, value), count in sorted(common_classes.items()):
        facts.append(
            _fact(
                "common_class_count",
                count,
                task=task,
                dialect=dialect,
                split=split,
                value=value,
            )
        )
    _imbalance_facts(facts, source_classes, space="source_native")
    _imbalance_facts(facts, common_classes, space="common")

    invariants = [
        _invariant("canonical_piece", canonical_before, canonical_after),
        _invariant("source_target_bundle", target_before, target_after),
        _invariant("grouping", grouping_before, grouping_after),
    ]
    if run_graph_invariance:
        invariants.extend(
            (
                _invariant("raw_graph", graph_before, graph_after),
                _invariant("model_input", model_before, model_after),
            )
        )
    else:
        facts.append(_fact("graph_invariance_performed", False))

    raw_manifest = json.loads(RAW_PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    source_target_manifest = json.loads(
        SOURCE_TARGET_MANIFEST.read_text(encoding="utf-8")
    )
    projection_fingerprint = _combined_fingerprint(
        "dilemmadata.common.projections.v1",
        (
            (piece_id, projection.projection_fingerprint)
            for piece_id, projection in projections.items()
        ),
    )
    fingerprints = (
        ("analysisgnn_reference", ANALYSISGNN_REFERENCE.fingerprint),
        ("common_projection_combined", projection_fingerprint),
        ("common_registry", DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint),
        ("raw_index", raw_manifest["cache"]["index_fingerprint"]),
        ("source_target_audit", source_target_manifest["audit_fingerprint"]),
        ("source_target_combined", source_target_manifest["fingerprints"]["target_sidecars"]),
        ("split_manifest", raw_manifest["split"]["fingerprint"]),
    )
    source_entry_count = sum(
        target.statistics.available_entry_count + target.statistics.masked_entry_count
        for target in target_outcomes.values()
    )
    source_span_count = sum(
        target.statistics.alignment_span_count for target in target_outcomes.values()
    )
    report = make_dilemmadata_common_audit_report(
        base_git_sha=base_git_sha or _current_git_sha(),
        source_record_count=len(discovery.records),
        source_component_count=discovery.component_count,
        annotation_view_count=len(target_outcomes),
        source_entry_count=source_entry_count,
        source_span_count=source_span_count,
        projection_count=len(projections),
        facts=tuple(facts),
        collapse_table=_collapse_table(),
        analysisgnn_parity=_parity_rows(),
        overlap_evidence=_overlap_evidence(discovery.records, projections),
        invariance_evidence=tuple(invariants),
        fingerprints=fingerprints,
        test_target_access_policy="representation_audit_only_no_model_inference_metrics_selection_or_unlock",
    )
    return report


def manifest_projection(
    report: DilemmadataCommonHarmonicAuditReport,
) -> DilemmadataCommonHarmonicAuditManifest:
    """Create compact source-free evidence from one full report."""

    summary_names = {
        "accepted_projection_count",
        "accepted_split_record_count",
        "analysis_view_count",
        "quarantined_raw_count",
        "source_component_count",
        "source_explicit_overlap_count",
        "source_multi_record_component_count",
        "source_record_count",
        "test_targets_used_for_model_evaluation",
    }
    summary = [fact for fact in report.facts if fact.name in summary_names]
    for fact_name, output_name, states in (
        (
            "accepted_common_mapping_state_count",
            "accepted_common_mapping_state_total",
            ("ambiguous", "coarsened", "exact", "invalid", "masked", "missing", "unsupported"),
        ),
        (
            "full_source_mapping_state_count",
            "full_source_mapping_state_total",
            ("ambiguous", "coarsened", "exact", "invalid", "masked", "missing", "unsupported"),
        ),
    ):
        totals: Counter[str] = Counter()
        for fact in report.facts:
            if fact.name != fact_name or not isinstance(fact.value, int):
                continue
            state = dict(fact.dimensions).get("state")
            if state is not None:
                totals[state] += fact.value
        summary.extend(
            DilemmadataCommonAuditFact(
                name=output_name,
                dimensions=(("state", state),),
                value=totals[state],
            )
            for state in states
        )

    overlap_totals = Counter(row.comparison for row in report.overlap_evidence)
    summary.extend(
        DilemmadataCommonAuditFact(
            name="overlap_comparison_total",
            dimensions=(("comparison", comparison),),
            value=overlap_totals[comparison],
        )
        for comparison in (
            "coarsened_agreement",
            "conflict",
            "enharmonic_only_agreement",
            "exact_agreement",
            "unavailable",
        )
    )
    parity_totals = Counter(row[4] for row in report.analysisgnn_parity)
    summary.extend(
        DilemmadataCommonAuditFact(
            name="analysisgnn_parity_total",
            dimensions=(("agreement", agreement),),
            value=parity_totals[agreement],
        )
        for agreement in ("agree", "diverge", "not_applicable")
    )
    summary.extend(
        DilemmadataCommonAuditFact(
            name="invariant_unchanged",
            dimensions=(("artifact", row.name),),
            value=row.unchanged,
        )
        for row in report.invariance_evidence
    )
    summary.extend(
        DilemmadataCommonAuditFact(
            name="artifact_fingerprint",
            dimensions=(("artifact", name),),
            value=fingerprint,
        )
        for name, fingerprint in report.fingerprints
    )
    summary.extend(
        (
            DilemmadataCommonAuditFact(
                name="source_entry_count", dimensions=(), value=report.source_entry_count
            ),
            DilemmadataCommonAuditFact(
                name="source_span_count", dimensions=(), value=report.source_span_count
            ),
            DilemmadataCommonAuditFact(
                name="mapping_collapse_row_count",
                dimensions=(),
                value=len(report.collapse_table),
            ),
            DilemmadataCommonAuditFact(
                name="analysisgnn_parity_row_count",
                dimensions=(),
                value=len(report.analysisgnn_parity),
            ),
            DilemmadataCommonAuditFact(
                name="overlap_evidence_row_count",
                dimensions=(),
                value=len(report.overlap_evidence),
            ),
            DilemmadataCommonAuditFact(
                name="candidate_same_input_alternative_group_count",
                dimensions=(),
                value=len(
                    {
                        row.component_id
                        for row in report.overlap_evidence
                        if row.component_id.startswith("alternative:")
                    }
                ),
            ),
        )
    )
    summary_tuple = tuple(
        sorted(summary, key=lambda row: (row.name, row.dimensions))
    )
    invalid_common = sum(
        int(fact.value)
        for fact in report.facts
        if fact.name == "accepted_common_mapping_state_count"
        and dict(fact.dimensions).get("state") == "invalid"
        and isinstance(fact.value, int)
    )
    split_counts = {
        dict(fact.dimensions).get("split"): fact.value
        for fact in report.facts
        if fact.name == "accepted_split_record_count"
    }
    parity_totals = Counter(row[4] for row in report.analysisgnn_parity)
    divergences = {
        (task_id, source_value, reference_value, common_value)
        for task_id, source_value, reference_value, common_value, agreement in (
            report.analysisgnn_parity
        )
        if agreement == "diverge"
    }
    ready = (
        report.projection_count == report.annotation_view_count
        and report.projection_count > 0
        and invalid_common == 0
        and all(row.unchanged for row in report.invariance_evidence)
        and set(split_counts) == {"test", "train", "validation"}
        and all(isinstance(value, int) and value > 0 for value in split_counts.values())
        and parity_totals
        == Counter({"agree": 36, "diverge": 2, "not_applicable": 51})
        and divergences
        == {
            (
                "dilemmadata.dlc.chord.quality",
                "+7",
                "augmented triad",
                "augmented seventh chord",
            ),
            (
                "dilemmadata.dlc.chord.quality",
                "+M7",
                "augmented triad",
                "augmented major tetrachord",
            ),
        }
        and all(
            row.state in {"exact", "coarsened", "ambiguous", "unsupported", "invalid"}
            for row in DILEMMADATA_COMMON_HARMONIC_REGISTRY.quality_mapping_rows
        )
    )
    return make_dilemmadata_common_audit_manifest(
        report,
        summary_facts=summary_tuple,
        ready=ready,
    )


def validate_manifest_self_fingerprint(
    value: Mapping[str, object] | DilemmadataCommonHarmonicAuditManifest,
) -> bool:
    try:
        manifest = (
            value
            if isinstance(value, DilemmadataCommonHarmonicAuditManifest)
            else loads_dilemmadata_common_audit_manifest(_canonical_json(value))
        )
    except (TypeError, ValueError):
        return False
    return (
        manifest.registry_fingerprint
        == DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint
        and manifest.analysisgnn_reference_fingerprint
        == ANALYSISGNN_REFERENCE.fingerprint
        and manifest.contract_version
        == DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION
        and manifest.audit_report_version
        == DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION
    )


def _validate_manifest_evidence(
    manifest: DilemmadataCommonHarmonicAuditManifest,
) -> bool:
    """Validate the compact production evidence independently of the corpus."""

    facts = {
        (fact.name, fact.dimensions): fact.value for fact in manifest.summary_facts
    }
    split_counts = {
        dict(dimensions).get("split"): value
        for (name, dimensions), value in facts.items()
        if name == "accepted_split_record_count"
    }
    projection_count = facts.get(("accepted_projection_count", ()))
    analysis_view_count = facts.get(("analysis_view_count", ()))
    invariant_names = {
        "canonical_piece",
        "grouping",
        "model_input",
        "raw_graph",
        "source_target_bundle",
    }
    artifact_names = {
        "analysisgnn_reference",
        "common_projection_combined",
        "common_registry",
        "raw_index",
        "source_target_audit",
        "source_target_combined",
        "split_manifest",
    }
    invariants_valid = all(
        facts.get(("invariant_unchanged", (("artifact", name),))) is True
        for name in invariant_names
    )
    artifact_fingerprints = {
        dict(dimensions).get("artifact"): value
        for (name, dimensions), value in facts.items()
        if name == "artifact_fingerprint"
    }
    artifacts_valid = (
        set(artifact_fingerprints) == artifact_names
        and all(
            isinstance(value, str) and len(value) == 64
            for value in artifact_fingerprints.values()
        )
        and artifact_fingerprints["common_registry"]
        == manifest.registry_fingerprint
        and artifact_fingerprints["analysisgnn_reference"]
        == manifest.analysisgnn_reference_fingerprint
    )
    invalid_total = facts.get(
        (
            "accepted_common_mapping_state_total",
            (("state", "invalid"),),
        )
    )
    test_used = facts.get(("test_targets_used_for_model_evaluation", ()))
    alternative_group_count = facts.get(
        ("candidate_same_input_alternative_group_count", ())
    )
    parity_counts = {
        dict(dimensions).get("agreement"): value
        for (name, dimensions), value in facts.items()
        if name == "analysisgnn_parity_total"
    }
    parity_row_count = facts.get(("analysisgnn_parity_row_count", ()))
    return (
        isinstance(projection_count, int)
        and projection_count > 0
        and projection_count == analysis_view_count
        and set(split_counts) == {"test", "train", "validation"}
        and sum(split_counts.values()) == projection_count
        and invalid_total == 0
        and alternative_group_count == 30
        and parity_row_count == 89
        and parity_counts == {"agree": 36, "diverge": 2, "not_applicable": 51}
        and test_used is False
        and invariants_valid
        and artifacts_valid
    )


def check_committed_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, object]:
    try:
        manifest = loads_dilemmadata_common_audit_manifest(
            path.read_text(encoding="utf-8")
        )
        valid = validate_manifest_self_fingerprint(
            manifest
        ) and _validate_manifest_evidence(manifest)
        ready = valid and manifest.ready
        return {
            "check_version": "1.0.0",
            "manifest_path": path.name,
            "manifest_fingerprint": manifest.manifest_fingerprint,
            "registry_fingerprint": manifest.registry_fingerprint,
            "analysisgnn_reference_fingerprint": (
                manifest.analysisgnn_reference_fingerprint
            ),
            "valid": valid,
            "ready": ready,
        }
    except (OSError, TypeError, ValueError) as exc:
        return {
            "check_version": "1.0.0",
            "manifest_path": path.name,
            "valid": False,
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-git-sha")
    parser.add_argument("--no-graph-invariance", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.check:
        result = check_committed_manifest(arguments.manifest)
        print(_canonical_json(result, indent=2))
        return 0 if result.get("ready") else 1
    root = arguments.root
    if root is None:
        configured = os.environ.get(ENV_ROOT)
        if configured:
            root = Path(configured)
    if root is None:
        raise SystemExit(
            f"provide a corpus root or set {ENV_ROOT}; --check is source-free"
        )
    report = build_common_audit_report(
        root,
        base_git_sha=arguments.base_git_sha,
        run_graph_invariance=not arguments.no_graph_invariance,
    )
    report_payload = dumps_dilemmadata_common_audit_report(report)
    if arguments.output is None:
        sys.stdout.write(report_payload)
    else:
        _write(arguments.output, report_payload)
    if arguments.write_manifest:
        manifest = manifest_projection(report)
        _write(
            arguments.manifest,
            dumps_dilemmadata_common_audit_manifest(manifest, indent=None),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
