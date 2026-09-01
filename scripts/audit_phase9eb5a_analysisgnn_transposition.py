#!/usr/bin/env python3
"""Build or source-free verify the Phase 9E-B5A transposition audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys

from music_critic.adapters.dilemmadata import (
    DILEMMADATA_RECORD_BINDING_VERSION,
    DilemmadataAccepted,
    DilemmadataCorpusIdentity,
    DilemmadataCorpusRecord,
    _bind_record,
    _hash_file as _adapter_hash_file,
    _parse_raw_file,
    _piece_id,
    convert_dilemmadata_record,
)
from music_critic.experiments.analysisgnn.class_balance import (
    observations_from_sidecar,
)
from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_NAMESPACE,
    EXPECTED_PAPER_COUNTS,
    PRODUCTION_TASKS,
    TASK_BY_ID,
    get_vocabulary,
    materialize_target_sidecar,
    sidecar_contract_counts,
)
from music_critic.experiments.analysisgnn.transposition import (
    ABSOLUTE_TASKS,
    SHIFT_PCS,
    SIGNED_BY_SHIFT_PC,
    TRANSPOSITION_AUDIT_SCHEMA,
    PostTranspositionAccumulator,
    RecordShiftEligibility,
    corrected_transposition_profile,
    mapping_summary,
    model_input_collision_fingerprint,
    official_transposition_evidence,
    role_recommendations,
    semantic_mapping_index,
    semantic_mapping_rows,
    transformation_registry,
    transposition_contract,
    valid_shift_for_midi,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = (
    REPO_ROOT
    / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
)
DEFAULT_B2_ROOT = (
    REPO_ROOT / "outputs/phase9eb2/dilemmadata-coverage-remediation-877c168"
)
DEFAULT_B3_ROOT = (
    REPO_ROOT / "outputs/phase9eb3/analysisgnn-multitask-contract-01290f5"
)
DEFAULT_B4_ROOT = (
    REPO_ROOT / "outputs/phase9eb4/analysisgnn-class-balance-671097b"
)
DEFAULT_B4_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb4_class_balance_audit.json"
)
DEFAULT_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb5a_transposition_audit.json"
)
EXPECTED_B3_SEMANTIC_FINGERPRINT = (
    "94a19ed6bbecbbd0497310233c8a8ff4e34311b414124593a7326c759ff07954"
)
EXPECTED_B4_SEMANTIC_FINGERPRINT = (
    "4b1edf9f47815bafa5e197be87b9331a19789142c0625ef4aceda1f87649df4d"
)
REQUIRED_B3_ARTIFACTS = (
    "dataset_manifest.json",
    "entity_registry.jsonl",
    "paper_candidate_manifest.json",
    "paper_candidate_records.jsonl",
    "split_assignments.jsonl",
    "split_summary.json",
    "target_sidecars.jsonl",
    "task_registry.json",
    "vocabularies.json",
)
OUTPUT_ARTIFACTS = (
    "official_transposition_evidence.json",
    "transformation_registry.json",
    "semantic_mapping.jsonl",
    "record_shift_eligibility.jsonl",
    "collision_report.jsonl",
    "post_transposition_balance.json",
    "head_role_recommendations.json",
    "audit_summary.json",
    "AUDIT_REPORT.md",
)
INVARIANT_TASKS = {
    row.task_id
    for row in transformation_registry()
    if row.transformation_kind
    in {
        "relative_label_invariant",
        "structural_label_invariant",
        "boolean_label_invariant",
    }
}


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_short_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def default_output() -> Path:
    return REPO_ROOT / (
        f"outputs/phase9eb5a/analysisgnn-transposition-{_git_short_commit()}"
    )


def _verify_b3(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("semantic_fingerprint") != EXPECTED_B3_SEMANTIC_FINGERPRINT:
        raise RuntimeError("Phase 9E-B3 semantic fingerprint changed before B5A")
    if summary.get("valid") is not True or summary.get("ready") is not True:
        raise RuntimeError("Phase 9E-B3 is not valid/ready")
    declared = summary.get("artifacts")
    if not isinstance(declared, dict):
        raise RuntimeError("Phase 9E-B3 artifact hashes are absent")
    hashes = {name: _hash_file(root / name) for name in REQUIRED_B3_ARTIFACTS}
    if any(declared.get(name) != value for name, value in hashes.items()):
        raise RuntimeError("Phase 9E-B3 artifact drift")
    return summary, hashes


def _verify_b4(
    root: Path, fixture_path: Path
) -> tuple[dict[str, object], dict[str, Mapping[str, object]], dict[str, str]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    observed = fixture.pop("fixture_fingerprint", None)
    if fingerprint(fixture) != observed:
        raise RuntimeError("Phase 9E-B4 fixture fingerprint mismatch")
    fixture["fixture_fingerprint"] = observed
    if fixture.get("semantic_fingerprint") != EXPECTED_B4_SEMANTIC_FINGERPRINT:
        raise RuntimeError("Phase 9E-B4 semantic fingerprint changed before B5A")
    declared = fixture.get("artifact_sha256")
    if not isinstance(declared, dict):
        raise RuntimeError("Phase 9E-B4 artifact hashes are absent")
    hashes = {name: _hash_file(root / name) for name in declared}
    if hashes != declared:
        raise RuntimeError("Phase 9E-B4 artifact drift")
    payload = json.loads((root / "head_balance_summary.json").read_text(encoding="utf-8"))
    heads = payload.get("heads")
    if not isinstance(heads, list) or len(heads) != 20:
        raise RuntimeError("Phase 9E-B4 head summary is invalid")
    return fixture, {str(row["task_id"]): row for row in heads}, hashes


def _selected_train_target_descriptors(
    registry_lines: Iterable[str],
    descriptor_lines: Iterable[str],
    *,
    train_ids: set[str],
    descriptor_decoder: Callable[[str], object] = json.loads,
) -> tuple[set[str], dict[str, dict[str, object]]]:
    """Filter by target-free entity registry before decoding target descriptors."""

    registry_ids: set[str] = set()
    descriptors: dict[str, dict[str, object]] = {}
    for registry_line, descriptor_line in zip_longest(registry_lines, descriptor_lines):
        if registry_line is None or descriptor_line is None:
            raise RuntimeError("B3 entity/target descriptor line counts differ")
        registry = json.loads(registry_line)
        record_id = str(registry["record_id"])
        registry_ids.add(record_id)
        if record_id not in train_ids:
            continue
        descriptor = descriptor_decoder(descriptor_line)
        if not isinstance(descriptor, dict) or descriptor.get("record_id") != record_id:
            raise RuntimeError("B3 entity/target descriptor ordering changed")
        descriptors[record_id] = descriptor
    return registry_ids, descriptors


def _target_free_inputs(
    b3_root: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    assignments = _read_jsonl(b3_root / "split_assignments.jsonl")
    candidate = {
        str(row["record_id"]): row
        for row in _read_jsonl(b3_root / "paper_candidate_records.jsonl")
    }
    if len(assignments) != EXPECTED_PAPER_COUNTS["total"]:
        raise RuntimeError("frozen paper-candidate count changed")
    if any(
        row.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM
        or row.get("assignment_namespace") != ASSIGNMENT_NAMESPACE
        for row in assignments
    ):
        raise RuntimeError("frozen split assignment identity changed")
    assignment_ids = {str(row["record_id"]) for row in assignments}
    if assignment_ids != set(candidate):
        raise RuntimeError("B3 target-free manifests have different coverage")
    train_ids = {
        str(row["record_id"]) for row in assignments if row["split"] == "train"
    }
    with (
        (b3_root / "entity_registry.jsonl").open("r", encoding="utf-8") as registry,
        (b3_root / "target_sidecars.jsonl").open("r", encoding="utf-8") as descriptors,
    ):
        registry_ids, selected = _selected_train_target_descriptors(
            registry, descriptors, train_ids=train_ids
        )
    if registry_ids != assignment_ids or set(selected) != train_ids:
        raise RuntimeError("strict TRAIN descriptor selection changed")
    return assignments, candidate, selected


def _record_path(root: Path, record_id: str) -> tuple[Path, str, str, str | None]:
    prefix, middle, piece = record_id.split(":", 2)
    if prefix == "an":
        return (
            root / f"pitch_arrays/AN/{middle}/{piece}_joint.tsv",
            "an_joint",
            piece.split("-", 1)[0],
            middle,
        )
    if prefix == "dlc":
        return root / f"pitch_arrays/DLC/{middle}/{piece}.tsv", "dlc", middle, None
    raise RuntimeError(f"unknown frozen record ID {record_id!r}")


def _selective_record(
    root: Path, record_row: Mapping[str, object]
) -> DilemmadataCorpusRecord:
    """Read only raw graph source fields for one frozen record."""

    record_id = str(record_row["record_id"])
    path, dialect, collection, suggested_split = _record_path(root, record_id)
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"frozen record path is missing: {record_id}")
    parse = _parse_raw_file(path, dialect)  # type: ignore[arg-type]
    if parse.raw_projection_sha256 != record_row["raw_projection_sha256"]:
        raise RuntimeError(f"raw projection changed for {record_id}")
    if parse.categories:
        raise RuntimeError(f"raw record no longer accepted: {record_id}: {parse.categories}")
    component_id = str(record_row["source_component_id"])
    component_digest = component_id.removeprefix("dilemmadata-component:")
    score_path: Path | None = None
    if dialect == "an_joint":
        candidates = [
            path.with_name(path.name.removesuffix("_joint.tsv") + suffix)
            for suffix in (".mxl", ".musicxml")
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        score_path = existing[0] if len(existing) == 1 else None
    relative = path.relative_to(root.resolve()).as_posix()
    score_relative = (
        None if score_path is None else score_path.relative_to(root.resolve()).as_posix()
    )
    record = DilemmadataCorpusRecord(
        record_id=record_id,
        piece_id=str(record_row["piece_id"]),
        dialect=dialect,  # type: ignore[arg-type]
        path=path,
        relative_path=relative,
        collection=collection,
        piece_name=record_id.split(":", 2)[2],
        suggested_split=suggested_split,
        physical_source_sha256=_adapter_hash_file(path),
        raw_projection_sha256=parse.raw_projection_sha256,
        raw_equivalence_id=str(record_row["raw_equivalence_id"]),
        grouping_fingerprint=parse.grouping_fingerprint,
        source_group_id=component_id,
        lineage_group_id=f"dilemmadata-lineage:{component_digest}",
        source_resolution=parse.source_resolution,
        score_path=score_path,
        score_relative_path=score_relative,
        score_sha256=None if score_path is None else _adapter_hash_file(score_path),
        raw_issue_categories=parse.categories,
        note_row_count=parse.note_row_count,
        tie_continuation_row_count=parse.tie_continuation_row_count,
        zero_duration_row_count=parse.zero_duration_row_count,
        corpus_identity=DilemmadataCorpusIdentity(),
        record_binding_version=DILEMMADATA_RECORD_BINDING_VERSION,
        record_binding_sha256="",
    )
    bound = _bind_record(record)
    if bound.piece_id != _piece_id(record_id, dialect):
        raise RuntimeError(f"piece identity changed for {record_id}")
    return bound


def _load_b2_graph_fingerprints(root: Path) -> dict[str, str]:
    rows = _read_jsonl(root / "record_results.jsonl")
    values = {str(row["record_id"]): str(row["graph_fingerprint"]) for row in rows}
    if len(values) != 1633:
        raise RuntimeError("Phase 9E-B2 graph fingerprint coverage changed")
    return values


def _target_closure(
    observations: object,
    *,
    dialect: str,
    shift_pc: int,
) -> tuple[bool, bool, tuple[str, ...]]:
    index = semantic_mapping_index()
    reasons: set[str] = set()
    round_trip = True
    targets = getattr(observations, "targets")
    unique = {
        (row.task_id, row.class_value)
        for row in targets
        if row.available and not row.masked and row.class_value is not None
    }
    for task_id, value in unique:
        if task_id not in ABSOLUTE_TASKS and task_id != "pitch_class_set":
            continue
        row = index.get((task_id, dialect, value, shift_pc))
        if row is None:
            reasons.add(f"unsupported_source_value:{task_id}")
            round_trip = False
        elif not row.valid:
            reasons.add(f"{row.invalid_reason}:{task_id}")
            round_trip = round_trip and row.round_trip_valid
        else:
            round_trip = round_trip and row.round_trip_valid
    return not reasons, round_trip, tuple(sorted(reasons))


def _raw_train_counts(class_rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], int]:
    return {
        (str(row["task_id"]), str(row["class_value"])): int(
            row["canonical_target_row_count"]
        )
        for row in class_rows
        if row["split"] == "train" and row["task_id"] in TASK_BY_ID
    }


def _invariance_proof(
    accumulator: PostTranspositionAccumulator,
    raw_counts: Mapping[tuple[str, str], int],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    checked = 0
    for task_id in sorted(INVARIANT_TASKS):
        vocabulary = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
        for value in vocabulary:
            checked += 1
            raw = raw_counts.get((task_id, value), 0)
            expected = accumulator.expected_rows[(task_id, value)]
            if abs(expected - raw) > 1e-6:
                mismatches.append(
                    {"task_id": task_id, "class_value": value, "raw": raw, "expected": expected}
                )
    return {
        "checked_class_rows": checked,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
        "quality_17_distribution_identical": not any(
            row["task_id"] == "quality" for row in mismatches
        ),
        "roman_184_distribution_identical": not any(
            row["task_id"] == "roman_numeral" for row in mismatches
        ),
        "note_degree_distribution_identical": not any(
            row["task_id"] == "note_degree" for row in mismatches
        ),
        "phrase_section_negative_examples_created": False,
    }


def _eligibility_summary(
    rows: Sequence[RecordShiftEligibility],
) -> dict[str, object]:
    by_record: defaultdict[str, list[RecordShiftEligibility]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    limited_records: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_record[row.record_id].append(row)
        for reason in row.corrected_invalid_reasons:
            code = reason.split(":", 1)[0]
            reason_counts[code] += 1
            limited_records[code].add(row.record_id)
    valid_counts = {
        record_id: sum(row.corrected_valid for row in record_rows)
        for record_id, record_rows in by_record.items()
    }
    summary: dict[str, object] = {
        "train_record_count": len(by_record),
        "record_shift_row_count": len(rows),
        "records_with_12_valid_shifts": sum(value == 12 for value in valid_counts.values()),
        "records_with_2_to_11_valid_shifts": sum(2 <= value <= 11 for value in valid_counts.values()),
        "identity_only_records": sum(value == 1 for value in valid_counts.values()),
        "records_with_no_valid_non_identity_shift": sum(
            value == 1 for value in valid_counts.values()
        ),
        "minimum_valid_shifts": min(valid_counts.values()),
        "maximum_valid_shifts": max(valid_counts.values()),
        "corrected_valid_variant_count": sum(valid_counts.values()),
        "official_valid_variant_count": sum(row.official_valid for row in rows),
        "official_requested_variant_count": sum(row.official_valid for row in rows),
        "official_materialization_success_attested": False,
        "official_valid_semantics": (
            "requested_interval_before_caught_external_encoder_materialization_exceptions"
        ),
        "invalid_reason_counts": dict(sorted(reason_counts.items())),
        "records_limited_by_reason": {
            key: len(value) for key, value in sorted(limited_records.items())
        },
    }
    summary["fingerprint"] = fingerprint(summary)
    return summary


def _compact_heads(
    official: Sequence[Mapping[str, object]],
    corrected: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    official_by_task = {str(row["task_id"]): row for row in official}
    corrected_by_task = {str(row["task_id"]): row for row in corrected}
    rows: list[dict[str, object]] = []
    for task in PRODUCTION_TASKS:
        official_row = official_by_task[task.task_id]
        corrected_row = corrected_by_task[task.task_id]
        rows.append(
            {
                "task_id": task.task_id,
                "b4_raw_status": corrected_row["b4_raw_status"],
                "official_full_orbit_status": official_row["full_orbit_status"],
                "official_expected_epoch_status": official_row["expected_epoch_status"],
                "corrected_full_orbit_status": corrected_row["full_orbit_status"],
                "corrected_expected_epoch_status": corrected_row["expected_epoch_status"],
                "augmentation_effect": corrected_row["augmentation_effect"],
                "raw_observed_class_count": corrected_row["raw"]["observed_class_count"],
                "corrected_observed_class_count": corrected_row["expected_epoch"][
                    "observed_class_count"
                ],
                "corrected_absent_classes": corrected_row["expected_epoch"][
                    "absent_classes"
                ],
                "corrected_insufficient_classes": corrected_row["expected_epoch"][
                    "insufficient_classes"
                ],
                "raw_majority_share": corrected_row["raw"]["majority_share"],
                "raw_max_to_min_nonzero_ratio": corrected_row["raw"][
                    "max_to_min_nonzero_ratio"
                ],
                "raw_normalized_entropy": corrected_row["raw"][
                    "normalized_entropy"
                ],
                "official_expected_majority_share": official_row["expected_epoch"][
                    "majority_share"
                ],
                "official_expected_max_to_min_nonzero_ratio": official_row[
                    "expected_epoch"
                ]["max_to_min_nonzero_ratio"],
                "official_expected_normalized_entropy": official_row[
                    "expected_epoch"
                ]["normalized_entropy"],
                "corrected_expected_majority_share": corrected_row[
                    "expected_epoch"
                ]["majority_share"],
                "corrected_expected_max_to_min_nonzero_ratio": corrected_row[
                    "expected_epoch"
                ]["max_to_min_nonzero_ratio"],
                "corrected_expected_normalized_entropy": corrected_row[
                    "expected_epoch"
                ]["normalized_entropy"],
                "source_component_count": corrected_row["source_component_count"],
                "transformed_component_support": corrected_row[
                    "transformed_component_support"
                ],
                "component_shift_support": corrected_row["component_shift_support"],
                "classes_created_only_by_augmentation": corrected_row[
                    "classes_created_only_by_augmentation"
                ],
                "classes_remaining_unsupported": corrected_row[
                    "classes_remaining_unsupported"
                ],
            }
        )
    return rows


def _recommendation_summary(payload: Mapping[str, object]) -> dict[str, object]:
    official = payload["official_reproduction"]
    corrected = payload["corrected_v2"]
    return {
        "official_counts": dict(
            sorted(Counter(row["recommendation"] for row in official).items())
        ),
        "corrected_counts": dict(
            sorted(Counter(row["candidate_role"] for row in corrected).items())
        ),
        "official_by_task": {
            row["task_id"]: row["recommendation"] for row in official
        },
        "corrected_by_task": {
            row["task_id"]: row["candidate_role"] for row in corrected
        },
    }


def _report(
    summary: Mapping[str, object],
    heads: Sequence[Mapping[str, object]],
) -> str:
    eligibility = summary["eligibility"]
    leakage = summary["leakage"]
    lines = [
        "# Phase 9E-B5A AnalysisGNN transposition audit",
        "",
        "Official reproduction evidence and the corrected V2 candidate are separate profiles. No augmentation is wired into training in this phase.",
        "",
        "## Frozen access boundary",
        "",
        f"- TRAIN raw/target records: `{summary['target_access']['train_target_records_opened']}` / `{summary['target_access']['train_target_records_opened']}`",
        f"- VALIDATION raw/target records: `{summary['target_access']['validation_raw_records_opened']}` / `0`",
        f"- TEST raw/target records: `{summary['target_access']['test_raw_records_opened']}` / `0`",
        "- TEST target rows loaded: `0`; TEST evaluation: `false`",
        "",
        "## Eligibility and leakage",
        "",
        f"- Full 12-shift records: `{eligibility['records_with_12_valid_shifts']}`",
        f"- Partial 2..11 records: `{eligibility['records_with_2_to_11_valid_shifts']}`",
        f"- Identity-only records: `{eligibility['identity_only_records']}`",
        f"- Corrected collision variants excluded: `{leakage['corrected_collision_variant_count']}`",
        f"- Official collision variants reported: `{leakage['official_collision_variant_count']}`",
        "",
        "## Post-transposition head status",
        "",
        "| head | B4 raw | official orbit | official epoch | corrected orbit | corrected epoch | effect |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in heads:
        lines.append(
            "| {task_id} | {b4_raw_status} | {official_full_orbit_status} | "
            "{official_expected_epoch_status} | {corrected_full_orbit_status} | "
            "{corrected_expected_epoch_status} | {augmentation_effect} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Quality-17, Roman-184, note-degree, phrase, and section remain invariant under joint pitch/key transposition. In particular, transposition creates neither `augmented sixth` quality rows nor new Roman functions or phrase/section negatives.",
            "",
            f"Semantic fingerprint: `{summary['semantic_fingerprint']}`.",
            "",
            "Dataset, split, TEST assignment, vocabularies, raw graph cache, model, heads, losses, sampler, and training configuration were not changed. Candidate head roles remain recommendations only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact_hashes(output: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: _hash_file(output / name) for name in names}


def _compact_fixture(
    *,
    summary: Mapping[str, object],
    heads: Sequence[Mapping[str, object]],
    recommendations: Mapping[str, object],
    output: Path,
) -> dict[str, object]:
    fixture: dict[str, object] = {
        "fixture_schema": "phase9eb5a-analysisgnn-transposition-fixture-v1",
        "schema": TRANSPOSITION_AUDIT_SCHEMA,
        "version": transposition_contract()["version"],
        "artifact_sha256": _artifact_hashes(output, OUTPUT_ARTIFACTS),
        "input_fingerprints": summary["input_fingerprints"],
        "official_evidence_fingerprint": summary["profiles"]["official"]["fingerprint"],
        "corrected_policy_fingerprint": summary["profiles"]["corrected"]["fingerprint"],
        "transformation_registry_fingerprint": summary["transformation_registry_fingerprint"],
        "mapping_summary": summary["mapping_summary"],
        "eligibility_summary": summary["eligibility"],
        "leakage_summary": summary["leakage"],
        "invariance_proof": summary["invariance_proof"],
        "head_count": len(heads),
        "head_summaries": list(heads),
        "recommendation_summary": _recommendation_summary(recommendations),
        "test_lock": summary["test_lock"],
        "semantic_fingerprint": summary["semantic_fingerprint"],
        "valid": summary["valid"],
    }
    fixture["fixture_fingerprint"] = fingerprint(fixture)
    return fixture


def check_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint", None)
    if fingerprint(value) != observed:
        raise RuntimeError("Phase 9E-B5A fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    if value.get("schema") != TRANSPOSITION_AUDIT_SCHEMA or value.get("valid") is not True:
        raise RuntimeError("Phase 9E-B5A fixture is not valid")
    if value.get("head_count") != 20 or len(value.get("head_summaries", [])) != 20:
        raise RuntimeError("Phase 9E-B5A fixture must cover 20 heads")
    if value.get("official_evidence_fingerprint") != official_transposition_evidence()[
        "fingerprint"
    ]:
        raise RuntimeError("official transposition evidence changed")
    if value.get("corrected_policy_fingerprint") != corrected_transposition_profile()[
        "fingerprint"
    ]:
        raise RuntimeError("corrected transposition policy changed")
    lock = value.get("test_lock")
    required = {
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }
    if not isinstance(lock, dict) or any(lock.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("Phase 9E-B5A TEST lock is invalid")
    hashes = value.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(OUTPUT_ARTIFACTS):
        raise RuntimeError("Phase 9E-B5A artifact inventory is incomplete")
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or set(item) - set("0123456789abcdef")
        for item in hashes.values()
    ):
        raise RuntimeError("Phase 9E-B5A artifact SHA-256 is invalid")
    return value


def reseal_derived_entity_counts(
    output: Path,
    fixture: Path,
    *,
    b3_root: Path = DEFAULT_B3_ROOT,
) -> None:
    """Add source-free entity totals from frozen TRAIN descriptor counts.

    This does not open corpus files or target payloads.  It combines the
    already-audited valid-shift manifest with B3's compact per-task state
    counts, then reseals the affected artifact hashes.
    """

    assignments = _read_jsonl(b3_root / "split_assignments.jsonl")
    train_ids = {
        str(row["record_id"]) for row in assignments if row["split"] == "train"
    }
    with (
        (b3_root / "entity_registry.jsonl").open("r", encoding="utf-8") as registry,
        (b3_root / "target_sidecars.jsonl").open("r", encoding="utf-8") as descriptors,
    ):
        _registry_ids, selected = _selected_train_target_descriptors(
            registry, descriptors, train_ids=train_ids
        )
    valid_shifts: Counter[str] = Counter()
    valid_shift_values: defaultdict[str, set[int]] = defaultdict(set)
    for row in _read_jsonl(output / "record_shift_eligibility.jsonl"):
        if row["corrected_valid"]:
            record_id = str(row["record_id"])
            valid_shifts[record_id] += 1
            valid_shift_values[record_id].add(int(row["shift_pc"]))
    corrected_entities: Counter[str] = Counter()
    official_entities: Counter[str] = Counter()
    expected_entities: Counter[str] = Counter()
    corrected_component_shifts: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)
    official_component_shifts: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)
    for record_id, descriptor in selected.items():
        for task_id, states in descriptor["task_states"].items():
            available = int(states.get("available", 0))
            expected_entities[task_id] += available
            corrected_entities[task_id] += available * valid_shifts[record_id]
            official_entities[task_id] += available * 12
            if available:
                component_id = str(descriptor["source_component_id"])
                corrected_component_shifts[task_id].update(
                    (component_id, shift_pc)
                    for shift_pc in valid_shift_values[record_id]
                )
                official_component_shifts[task_id].update(
                    (component_id, shift_pc) for shift_pc in SHIFT_PCS
                )
    post_path = output / "post_transposition_balance.json"
    post = json.loads(post_path.read_text(encoding="utf-8"))
    for profile, totals, component_shifts in (
        (
            "official_reproduction",
            official_entities,
            official_component_shifts,
        ),
        ("corrected_v2", corrected_entities, corrected_component_shifts),
    ):
        for head in post[profile]:
            task_id = str(head["task_id"])
            head["component_shift_support"] = len(component_shifts[task_id])
            head["transformed_component_support"] = head[
                "source_component_count"
            ]
            head["full_orbit"]["variant_entity_count"] = totals[task_id]
            head["full_orbit"]["variant_canonical_row_count"] = head[
                "full_orbit"
            ]["canonical_target_row_count"]
            head["expected_epoch"]["expected_entity_count"] = expected_entities[
                task_id
            ]
            head["expected_epoch"]["expected_canonical_row_count"] = head[
                "expected_epoch"
            ]["canonical_target_row_count"]
            head["sampling_interpretation"] = (
                "exact_uniform_one_draw_per_record_expectation"
                if profile == "corrected_v2"
                else "normalized_equal_requested_view_diagnostic_not_actual_loader_probability"
            )
    summary_path = output / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profiles = {
        "official": official_transposition_evidence(),
        "corrected": corrected_transposition_profile(),
    }
    registry_payload: dict[str, object] = {
        "contract": transposition_contract(),
        "heads": [asdict(row) for row in transformation_registry()],
    }
    registry_payload["fingerprint"] = fingerprint(registry_payload)
    eligibility = dict(summary["eligibility"])
    eligibility.pop("fingerprint", None)
    eligibility["official_requested_variant_count"] = eligibility[
        "official_valid_variant_count"
    ]
    eligibility["official_materialization_success_attested"] = False
    eligibility["official_valid_semantics"] = (
        "requested_interval_before_caught_external_encoder_materialization_exceptions"
    )
    eligibility["fingerprint"] = fingerprint(eligibility)
    recommendations = json.loads(
        (output / "head_role_recommendations.json").read_text(encoding="utf-8")
    )
    compact_heads = _compact_heads(
        post["official_reproduction"], post["corrected_v2"]
    )
    post["comparison"] = compact_heads
    _write_json(post_path, post)
    summary["profiles"] = profiles
    summary["transformation_registry_fingerprint"] = registry_payload["fingerprint"]
    summary["eligibility"] = eligibility
    semantic_payload = {
        "profiles": {
            key: value["fingerprint"] for key, value in profiles.items()
        },
        "registry": registry_payload["fingerprint"],
        "mapping": summary["mapping_summary"],
        "eligibility": eligibility,
        "leakage": summary["leakage"],
        "invariance": summary["invariance_proof"],
        "heads": compact_heads,
        "recommendations": _recommendation_summary(recommendations),
        "test_lock": summary["test_lock"],
    }
    summary["semantic_fingerprint"] = fingerprint(semantic_payload)
    _write_json(output / "official_transposition_evidence.json", profiles["official"])
    _write_json(output / "transformation_registry.json", registry_payload)
    (output / "AUDIT_REPORT.md").write_text(
        _report(summary, compact_heads), encoding="utf-8", newline="\n"
    )
    summary["artifact_sha256"] = _artifact_hashes(
        output, (name for name in OUTPUT_ARTIFACTS if name != "audit_summary.json")
    )
    _write_json(summary_path, summary)
    _write_json(
        fixture,
        _compact_fixture(
            summary=summary,
            heads=compact_heads,
            recommendations=recommendations,
            output=output,
        ),
    )


def build_audit(
    corpus_root: Path,
    *,
    b2_root: Path,
    b3_root: Path,
    b4_root: Path,
    b4_fixture: Path,
    output: Path,
    fixture: Path,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("production output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    b3_summary, b3_hashes = _verify_b3(b3_root)
    b4_value, raw_heads, b4_hashes = _verify_b4(b4_root, b4_fixture)
    assignments, candidates, descriptors = _target_free_inputs(b3_root)
    split_counts = Counter(str(row["split"]) for row in assignments)
    if split_counts != {"train": 1295, "validation": 162, "test": 162}:
        raise RuntimeError(f"frozen split changed: {dict(split_counts)}")
    b2_graphs = _load_b2_graph_fingerprints(b2_root)
    assignment_by_id = {str(row["record_id"]): row for row in assignments}
    heldout_ids = sorted(
        record_id
        for record_id, row in assignment_by_id.items()
        if row["split"] in ("validation", "test")
    )
    heldout_model_inputs: defaultdict[str, list[str]] = defaultdict(list)
    heldout_graphs: defaultdict[str, list[str]] = defaultdict(list)
    heldout_piece_ids: set[str] = set()
    heldout_components: set[str] = set()
    heldout_raw_equivalence: set[str] = set()
    access: Counter[str] = Counter()
    for ordinal, record_id in enumerate(heldout_ids, start=1):
        converted = convert_dilemmadata_record(
            _selective_record(corpus_root, candidates[record_id])
        )
        if not isinstance(converted, DilemmadataAccepted):
            raise RuntimeError(f"frozen held-out raw record failed: {record_id}")
        heldout_model_inputs[
            model_input_collision_fingerprint(converted.piece, shift_pc=0)
        ].append(record_id)
        heldout_graphs[b2_graphs[record_id]].append(record_id)
        heldout_piece_ids.add(str(candidates[record_id]["piece_id"]))
        heldout_components.add(str(candidates[record_id]["source_component_id"]))
        heldout_raw_equivalence.add(str(candidates[record_id]["raw_equivalence_id"]))
        split = str(assignment_by_id[record_id]["split"])
        access[f"{split}_raw_records_opened"] += 1
        if ordinal % 100 == 0:
            print(f"audited {ordinal}/{len(heldout_ids)} held-out raw records", flush=True)

    corrected_accumulator = PostTranspositionAccumulator("corrected_v2")
    official_accumulator = PostTranspositionAccumulator("official_reproduction")
    eligibility_rows: list[RecordShiftEligibility] = []
    collision_rows: list[dict[str, object]] = []
    descriptor_failures: list[str] = []
    train_ids = sorted(
        record_id
        for record_id, row in assignment_by_id.items()
        if row["split"] == "train"
    )
    for ordinal, record_id in enumerate(train_ids, start=1):
        converted = convert_dilemmadata_record(
            _selective_record(corpus_root, candidates[record_id])
        )
        if not isinstance(converted, DilemmadataAccepted):
            raise RuntimeError(f"frozen TRAIN raw record failed: {record_id}")
        sidecar = materialize_target_sidecar(converted)
        contract_counts = sidecar_contract_counts(sidecar)
        descriptor = descriptors[record_id]
        if (
            sidecar["entity_counts"] != descriptor["entity_counts"]
            or sidecar["relation_counts"] != descriptor["relation_counts"]
            or contract_counts["task_states"] != descriptor["task_states"]
            or contract_counts["joint_structural_support"]
            != descriptor["joint_structural_support"]
        ):
            descriptor_failures.append(record_id)
        observations = observations_from_sidecar(sidecar, split="train")
        pitches = tuple(
            note.pitch for note in converted.piece.notes if not note.is_percussion
        )
        corrected_valid_shifts: list[int] = []
        piece_collision = str(candidates[record_id]["piece_id"]) in heldout_piece_ids
        component_collision = (
            str(candidates[record_id]["source_component_id"]) in heldout_components
        )
        provenance_collision = (
            str(candidates[record_id]["raw_equivalence_id"])
            in heldout_raw_equivalence
        )
        exact_graph_collision = b2_graphs[record_id] in heldout_graphs
        for shift_pc in SHIFT_PCS:
            range_valid = valid_shift_for_midi(pitches, shift_pc)
            closed, round_trip, target_reasons = _target_closure(
                observations,
                dialect=observations.dialect,
                shift_pc=shift_pc,
            )
            corrected_collision_reasons: set[str] = set()
            official_collision_reasons: set[str] = set()
            if piece_collision:
                corrected_collision_reasons.add("piece_identity_collision")
                official_collision_reasons.add("piece_identity_collision")
            if component_collision:
                corrected_collision_reasons.add("source_component_collision")
                official_collision_reasons.add("source_component_collision")
            if provenance_collision:
                corrected_collision_reasons.add("source_row_provenance_collision")
                official_collision_reasons.add("source_row_provenance_collision")
            if shift_pc == 0 and exact_graph_collision:
                corrected_collision_reasons.add("exact_graph_fingerprint_collision")
                official_collision_reasons.add("exact_graph_fingerprint_collision")
            corrected_input = model_input_collision_fingerprint(
                converted.piece, shift_pc=shift_pc
            )
            official_input = model_input_collision_fingerprint(
                converted.piece, shift_pc=shift_pc, official_wrap=True
            )
            if corrected_input in heldout_model_inputs:
                corrected_collision_reasons.add("transposition_equivalence_collision")
            if official_input in heldout_model_inputs:
                official_collision_reasons.add("transposition_equivalence_collision")
            corrected_reasons = set(target_reasons)
            if not range_valid:
                corrected_reasons.add("midi_range_violation")
            corrected_reasons.update(corrected_collision_reasons)
            corrected_valid = not corrected_reasons and round_trip
            if not round_trip:
                corrected_reasons.add("non_bijective_mapping")
            if corrected_valid:
                corrected_valid_shifts.append(shift_pc)
            collision_status = "none"
            if corrected_collision_reasons or official_collision_reasons:
                collision_status = "corrected=" + (
                    ",".join(sorted(corrected_collision_reasons)) or "none"
                ) + ";official=" + (
                    ",".join(sorted(official_collision_reasons)) or "none"
                )
                collision_rows.append(
                    {
                        "record_id": record_id,
                        "source_component_id": observations.component_id,
                        "shift_pc": shift_pc,
                        "corrected_collision_reasons": sorted(corrected_collision_reasons),
                        "official_collision_reasons": sorted(official_collision_reasons),
                        "corrected_heldout_matches": heldout_model_inputs.get(
                            corrected_input, []
                        ),
                        "official_heldout_matches": heldout_model_inputs.get(
                            official_input, []
                        ),
                    }
                )
            eligibility_rows.append(
                RecordShiftEligibility(
                    record_id=record_id,
                    source_component_id=observations.component_id,
                    dialect=observations.dialect,
                    shift_pc=shift_pc,
                    signed_semitones=SIGNED_BY_SHIFT_PC[shift_pc],
                    official_valid=True,
                    official_invalid_reasons=(),
                    corrected_valid=corrected_valid,
                    corrected_invalid_reasons=tuple(sorted(corrected_reasons)),
                    graph_range_valid=range_valid,
                    all_targets_closed=closed,
                    round_trip_valid=round_trip,
                    collision_status=collision_status,
                )
            )
        if 0 not in corrected_valid_shifts:
            raise RuntimeError(f"identity shift is invalid for TRAIN record {record_id}")
        corrected_accumulator.add_record(observations, corrected_valid_shifts)
        official_accumulator.add_record(observations, SHIFT_PCS)
        access["train_raw_records_opened"] += 1
        access["train_target_records_opened"] += 1
        access["train_target_rows_loaded"] += converted.record.note_row_count
        if ordinal % 100 == 0:
            print(f"audited {ordinal}/{len(train_ids)} TRAIN records", flush=True)
    if descriptor_failures:
        raise RuntimeError(
            f"TRAIN target descriptors changed: {descriptor_failures[:16]}"
        )

    class_rows = _read_jsonl(b4_root / "class_counts.jsonl")
    raw_counts = _raw_train_counts(class_rows)
    corrected_heads = corrected_accumulator.summarize(raw_heads)
    official_heads = official_accumulator.summarize(raw_heads)
    invariant_proof = _invariance_proof(corrected_accumulator, raw_counts)
    if invariant_proof["mismatch_count"]:
        raise RuntimeError("invariant one-draw distributions changed")
    eligibility = _eligibility_summary(eligibility_rows)
    collision_types = Counter(
        reason
        for row in collision_rows
        for reason in set(row["corrected_collision_reasons"])
    )
    official_collision_types = Counter(
        reason
        for row in collision_rows
        for reason in set(row["official_collision_reasons"])
    )
    leakage: dict[str, object] = {
        "corrected_collision_variant_count": sum(
            bool(row["corrected_collision_reasons"]) for row in collision_rows
        ),
        "official_collision_variant_count": sum(
            bool(row["official_collision_reasons"]) for row in collision_rows
        ),
        "corrected_collision_reason_counts": dict(sorted(collision_types.items())),
        "official_collision_reason_counts": dict(
            sorted(official_collision_types.items())
        ),
        "variants_moved_to_other_split": 0,
        "variants_counted_as_new_source_components": False,
        "raw_test_targets_accessed": False,
    }
    leakage["fingerprint"] = fingerprint(leakage)
    recommendations = role_recommendations(corrected_heads, official_heads)
    compact_heads = _compact_heads(official_heads, corrected_heads)
    registry_payload: dict[str, object] = {
        "contract": transposition_contract(),
        "heads": [asdict(row) for row in transformation_registry()],
    }
    registry_payload["fingerprint"] = fingerprint(registry_payload)
    profiles = {
        "official": official_transposition_evidence(),
        "corrected": corrected_transposition_profile(),
    }
    mapping = mapping_summary()
    test_lock = {
        "test_assignments_seen": True,
        "test_assignment_record_count": split_counts["test"],
        "test_raw_records_opened": access["test_raw_records_opened"],
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }
    semantic_payload = {
        "profiles": {
            key: value["fingerprint"] for key, value in profiles.items()
        },
        "registry": registry_payload["fingerprint"],
        "mapping": mapping,
        "eligibility": eligibility,
        "leakage": leakage,
        "invariance": invariant_proof,
        "heads": compact_heads,
        "recommendations": _recommendation_summary(recommendations),
        "test_lock": test_lock,
    }
    semantic = fingerprint(semantic_payload)
    valid = (
        len(eligibility_rows) == 1295 * 12
        and len(compact_heads) == 20
        and eligibility["identity_only_records"] >= 0
        and invariant_proof["mismatch_count"] == 0
        and test_lock["test_target_records_opened"] == 0
        and test_lock["test_target_rows_loaded"] == 0
        and not descriptor_failures
    )
    summary: dict[str, object] = {
        "schema": TRANSPOSITION_AUDIT_SCHEMA,
        "valid": valid,
        "semantic_fingerprint": semantic,
        "input_fingerprints": {
            "b2_semantic_fingerprint": b3_summary["b2_audit"][
                "semantic_fingerprint"
            ],
            "b3_semantic_fingerprint": b3_summary["semantic_fingerprint"],
            "b3_artifact_sha256": b3_hashes,
            "b4_semantic_fingerprint": b4_value["semantic_fingerprint"],
            "b4_fixture_fingerprint": b4_value["fixture_fingerprint"],
            "b4_artifact_sha256": b4_hashes,
            "split_fingerprint": b3_summary["test_lock"][
                "test_assignment_fingerprint"
            ],
        },
        "profiles": profiles,
        "transformation_registry_fingerprint": registry_payload["fingerprint"],
        "mapping_summary": mapping,
        "eligibility": eligibility,
        "leakage": leakage,
        "invariance_proof": invariant_proof,
        "split_counts": dict(split_counts),
        "target_access": {
            "train_raw_records_opened": access["train_raw_records_opened"],
            "train_target_records_opened": access["train_target_records_opened"],
            "train_target_rows_loaded": access["train_target_rows_loaded"],
            "validation_raw_records_opened": access["validation_raw_records_opened"],
            "validation_target_records_opened": 0,
            "validation_target_rows_loaded": 0,
            "test_raw_records_opened": access["test_raw_records_opened"],
            "test_target_records_opened": 0,
            "test_target_rows_loaded": 0,
        },
        "test_lock": test_lock,
        "production_attempts": [
            {
                "attempt": 1,
                "semantic_result": "not_formed",
                "result": "technical_gate_failure",
                "reason": "compact_descriptor_fingerprint_was_compared_to_full_materialized_sidecar",
                "test_target_records_opened": 0,
                "test_target_rows_loaded": 0,
            },
            {"attempt": 2, "semantic_result": "success", "result": "success"},
        ],
        "dataset_changed": False,
        "split_changed": False,
        "test_assignment_changed": False,
        "vocabularies_changed": False,
        "raw_graph_cache_changed": False,
        "model_changed": False,
        "heads_changed": False,
        "losses_changed": False,
        "sampler_changed": False,
        "training_run": False,
        "inference_run": False,
        "head_roles_frozen": False,
    }

    _write_json(output / "official_transposition_evidence.json", profiles["official"])
    _write_json(output / "transformation_registry.json", registry_payload)
    _write_jsonl(output / "semantic_mapping.jsonl", (asdict(row) for row in semantic_mapping_rows()))
    _write_jsonl(output / "record_shift_eligibility.jsonl", (asdict(row) for row in eligibility_rows))
    _write_jsonl(output / "collision_report.jsonl", collision_rows)
    _write_json(
        output / "post_transposition_balance.json",
        {
            "official_reproduction": list(official_heads),
            "corrected_v2": list(corrected_heads),
            "comparison": compact_heads,
            "invariance_proof": invariant_proof,
        },
    )
    _write_json(output / "head_role_recommendations.json", recommendations)
    (output / "AUDIT_REPORT.md").write_text(
        _report(summary, compact_heads), encoding="utf-8", newline="\n"
    )
    summary["artifact_sha256"] = _artifact_hashes(
        output, (name for name in OUTPUT_ARTIFACTS if name != "audit_summary.json")
    )
    _write_json(output / "audit_summary.json", summary)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        fixture,
        _compact_fixture(
            summary=summary,
            heads=compact_heads,
            recommendations=recommendations,
            output=output,
        ),
    )
    if not valid:
        raise RuntimeError("Phase 9E-B5A audit failed; inspect audit_summary.json")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2_ROOT)
    parser.add_argument("--b3-root", type=Path, default=DEFAULT_B3_ROOT)
    parser.add_argument("--b4-root", type=Path, default=DEFAULT_B4_ROOT)
    parser.add_argument("--b4-fixture", type=Path, default=DEFAULT_B4_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        value = check_fixture(args.fixture)
        print(canonical_json({"valid": value["valid"]}, indent=2))
        return 0
    summary = build_audit(
        args.root,
        b2_root=args.b2_root,
        b3_root=args.b3_root,
        b4_root=args.b4_root,
        b4_fixture=args.b4_fixture,
        output=args.output or default_output(),
        fixture=args.fixture,
    )
    print(
        canonical_json(
            {
                "semantic_fingerprint": summary["semantic_fingerprint"],
                "valid": summary["valid"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
