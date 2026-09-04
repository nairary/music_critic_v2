#!/usr/bin/env python3
"""Build or source-free verify the Phase 9E-B4 class-balance audit."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable
from typing import Iterable, Mapping

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
    CLASS_BALANCE_SCHEMA,
    COMPATIBILITY_QUALITY_TASK,
    PRODUCTION_AUDIT_TASKS,
    ClassBalanceAccumulator,
    JointTupleAccumulator,
    candidate_class_weights,
    class_balance_contract,
    compact_problem_classes,
    joint_observations_from_sidecar,
    observations_from_sidecar,
    project_quality_record,
    quality_focus_summary,
    recommendation_payload,
    recommend_head_trainability,
    roman_numeral_summary,
    semantic_fingerprint,
)
from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_NAMESPACE,
    EXPECTED_PAPER_COUNTS,
    PRODUCTION_TASKS,
    materialize_target_sidecar,
    sidecar_contract_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = (
    REPO_ROOT
    / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
)
DEFAULT_B3_ROOT = (
    REPO_ROOT / "outputs/phase9eb3/analysisgnn-multitask-contract-01290f5"
)
DEFAULT_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb4_class_balance_audit.json"
)
EXPECTED_B3_SEMANTIC_FINGERPRINT = (
    "94a19ed6bbecbbd0497310233c8a8ff4e34311b414124593a7326c759ff07954"
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
    "class_counts.jsonl",
    "head_balance_summary.json",
    "joint_tuple_counts.jsonl",
    "trainability_recommendations.json",
    "candidate_class_weights.json",
    "audit_summary.json",
    "AUDIT_REPORT.md",
)


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
    return REPO_ROOT / f"outputs/phase9eb4/analysisgnn-class-balance-{_git_short_commit()}"


def _verify_b3(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    summary_path = root / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("semantic_fingerprint") != EXPECTED_B3_SEMANTIC_FINGERPRINT:
        raise RuntimeError("Phase 9E-B3 semantic fingerprint changed before B4")
    if summary.get("valid") is not True or summary.get("ready") is not True:
        raise RuntimeError("Phase 9E-B3 audit is not valid/ready")
    declared = summary.get("artifacts")
    if not isinstance(declared, dict):
        raise RuntimeError("Phase 9E-B3 artifact hashes are absent")
    hashes = {name: _hash_file(root / name) for name in REQUIRED_B3_ARTIFACTS}
    differences = {
        name: {"declared": declared.get(name), "observed": value}
        for name, value in hashes.items()
        if declared.get(name) != value
    }
    if differences:
        raise RuntimeError(f"Phase 9E-B3 artifact drift: {differences}")
    return summary, hashes


def _target_free_inputs(
    b3_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    assignments = _read_jsonl(b3_root / "split_assignments.jsonl")
    if len(assignments) != EXPECTED_PAPER_COUNTS["total"]:
        raise RuntimeError("frozen B3 assignment count changed")
    if any(
        row.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM
        or row.get("assignment_namespace") != ASSIGNMENT_NAMESPACE
        for row in assignments
    ):
        raise RuntimeError("frozen B3 assignment identity changed")
    candidate = {
        str(row["record_id"]): row
        for row in _read_jsonl(b3_root / "paper_candidate_records.jsonl")
    }
    assignment_ids = {str(row["record_id"]) for row in assignments}
    if assignment_ids != set(candidate):
        raise RuntimeError("B3 target-free manifests do not cover the same records")
    allowed_ids = {
        str(row["record_id"])
        for row in assignments
        if row["split"] in ("train", "validation")
    }
    with (
        (b3_root / "entity_registry.jsonl").open("r", encoding="utf-8") as registry,
        (b3_root / "target_sidecars.jsonl").open("r", encoding="utf-8") as sidecars,
    ):
        registry_ids, descriptors = _selected_target_descriptors(
            registry, sidecars, allowed_ids=allowed_ids
        )
    if registry_ids != assignment_ids or set(descriptors) != allowed_ids:
        raise RuntimeError("B3 descriptor registry coverage changed")
    return assignments, candidate, descriptors


def _selected_target_descriptors(
    registry_lines: Iterable[str],
    descriptor_lines: Iterable[str],
    *,
    allowed_ids: set[str],
    descriptor_decoder: Callable[[str], object] = json.loads,
) -> tuple[set[str], dict[str, dict[str, object]]]:
    """Decode descriptors only after target-free registry split filtering."""

    descriptors: dict[str, dict[str, object]] = {}
    registry_ids: set[str] = set()
    for entity_line, descriptor_line in zip_longest(registry_lines, descriptor_lines):
        if entity_line is None or descriptor_line is None:
            raise RuntimeError("B3 entity/target descriptor line counts differ")
        entity = json.loads(entity_line)
        record_id = str(entity["record_id"])
        registry_ids.add(record_id)
        # TEST descriptor bytes remain opaque and are never deserialized.
        if record_id not in allowed_ids:
            continue
        descriptor = descriptor_decoder(descriptor_line)
        if not isinstance(descriptor, dict) or descriptor.get("record_id") != record_id:
            raise RuntimeError("B3 entity/target descriptor ordering changed")
        descriptors[record_id] = descriptor
    return registry_ids, descriptors


def _record_path(root: Path, record_id: str) -> tuple[Path, str, str, str | None]:
    prefix, middle, piece = record_id.split(":", 2)
    if prefix == "an":
        relative = f"pitch_arrays/AN/{middle}/{piece}_joint.tsv"
        return root / relative, "an_joint", piece.split("-", 1)[0], middle
    if prefix == "dlc":
        relative = f"pitch_arrays/DLC/{middle}/{piece}.tsv"
        return root / relative, "dlc", middle, None
    raise RuntimeError(f"unknown frozen record ID {record_id!r}")


def _selective_record(
    root: Path, record_row: Mapping[str, object]
) -> DilemmadataCorpusRecord:
    """Open exactly one already-filtered TRAIN/VALIDATION source record.

    This avoids the global discovery path, which would deserialize rows from
    locked TEST files before filtering.  The resulting raw projection is bound
    back to the frozen B3 target-free manifest before target materialization.
    """

    record_id = str(record_row["record_id"])
    path, dialect, collection, suggested_split = _record_path(root, record_id)
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"frozen record path is missing: {record_id}")
    parse = _parse_raw_file(path, dialect)
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
    record = _bind_record(record)
    if record.piece_id != _piece_id(record_id, dialect):
        raise RuntimeError(f"piece identity changed for {record_id}")
    return record


def _artifact_hashes(output: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: _hash_file(output / name) for name in names}


def _candidate_weight_summary(weights: Mapping[str, object]) -> dict[str, object]:
    heads = weights.get("heads")
    if not isinstance(heads, list):
        raise RuntimeError("candidate-weight head rows are invalid")
    policies = Counter(str(row["diagnostic_policy_recommendation"]) for row in heads)
    return {
        "fingerprint": weights["fingerprint"],
        "head_policies": {
            str(row["task_id"]): str(row["diagnostic_policy_recommendation"])
            for row in heads
        },
        "methods": list(class_balance_contract()["weight_methods"]),
        "policy_counts": dict(sorted(policies.items())),
        "train_only": weights["train_only"],
        "validation_counts_used": weights["validation_counts_used"],
        "weighting_policy_frozen": weights["weighting_policy_frozen"],
    }


def _report(summary: Mapping[str, object], heads: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Phase 9E-B4 AnalysisGNN class-balance audit",
        "",
        "This audit measures label support for baseline planning; it does not guarantee model quality.",
        "",
        "## Access boundary",
        "",
        f"- TRAIN target records opened: `{summary['target_access']['train_target_records_opened']}`",
        f"- VALIDATION target records opened: `{summary['target_access']['validation_target_records_opened']}`",
        "- TEST target records opened: `0`",
        "- TEST target rows loaded: `0`",
        "- training/model/inference/evaluation: `false`",
        "",
        "## Production heads",
        "",
        "| head | vocab | TRAIN observed | VALIDATION observed | absent | insufficient | fragile | majority | ratio | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for head in heads:
        tiers = head["support_tier_counts"]
        lines.append(
            "| {task_id} | {vocabulary_size} | {train_observed_class_count} | "
            "{validation_observed_class_count} | {absent} | {insufficient} | {fragile} | "
            "{majority_share:.4f} | {ratio} | {recommendation} |".format(
                **head,
                absent=tiers["absent"],
                insufficient=tiers["insufficient"],
                fragile=tiers["fragile"],
                ratio=(
                    "n/a"
                    if head["max_to_min_nonzero_ratio"] is None
                    else f"{float(head['max_to_min_nonzero_ratio']):.2f}"
                ),
            )
        )
    quality = summary["quality"]
    roman = summary["roman_numeral_184"]
    joint = summary["joint_tuples"]
    lines.extend(
        [
            "",
            "## Focus findings",
            "",
            "- Corrected quality remains 17 classes; compatibility quality is a separate 15-class projection.",
            "- `+7` and `+M7` remain explicit corrected rows and only project to `augmented triad` in compatibility evidence.",
            f"- Roman-184 TRAIN absent classes: `{roman['train_absent_class_count']}`; VALIDATION absent classes: `{roman['validation_absent_class_count']}`.",
            f"- Corrected event TRAIN complete rows: `{joint['corrected_harmonic_event']['train']['row_count']}`.",
            f"- Compatibility note TRAIN rows: `{joint['compatibility_note']['train']['row_count']}`; canonical harmonic rows: `{joint['compatibility_note']['train']['canonical_harmonic_target_rows']}`.",
            f"- Semantic fingerprint: `{summary['semantic_fingerprint']}`.",
            "",
            "Final loss weights and sampling policy remain unfrozen. Dataset, split, vocabularies, masks, raw graphs, model, heads, losses, and training configs were not changed.",
        ]
    )
    assert quality
    return "\n".join(lines) + "\n"


def _compact_fixture(
    *,
    summary: Mapping[str, object],
    heads: Sequence[Mapping[str, object]],
    output: Path,
) -> dict[str, object]:
    artifact_sha256 = _artifact_hashes(output, OUTPUT_ARTIFACTS)
    fixture: dict[str, object] = {
        "artifact_sha256": artifact_sha256,
        "candidate_class_weight_summary": summary["candidate_class_weight_summary"],
        "class_balance_contract": summary["class_balance_contract"],
        "fixture_schema": "phase9eb4-analysisgnn-class-balance-fixture-v1",
        "head_count": len(heads),
        "head_summaries": [
            {
                "majority_share": row["majority_share"],
                "max_to_min_nonzero_ratio": row["max_to_min_nonzero_ratio"],
                "problem_classes": compact_problem_classes(row),
                "recommendation": row["recommendation"],
                "support_tier_counts": row["support_tier_counts"],
                "task_id": row["task_id"],
                "train_observed_class_count": row["train_observed_class_count"],
                "validation_observed_class_count": row["validation_observed_class_count"],
                "vocabulary_size": row["vocabulary_size"],
            }
            for row in heads
        ],
        "input_fingerprints": summary["input_fingerprints"],
        "joint_tuples": summary["joint_tuples"],
        "quality": summary["quality"],
        "recommendation_groups": summary["recommendation_groups"],
        "roman_numeral_184": summary["roman_numeral_184"],
        "schema": CLASS_BALANCE_SCHEMA,
        "semantic_fingerprint": summary["semantic_fingerprint"],
        "split_counts": summary["split_counts"],
        "test_lock": summary["test_lock"],
        "valid": summary["valid"],
    }
    fixture["fixture_fingerprint"] = fingerprint(fixture)
    return fixture


def check_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint", None)
    if fingerprint(value) != observed:
        raise RuntimeError("Phase 9E-B4 fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    if value.get("schema") != CLASS_BALANCE_SCHEMA or value.get("valid") is not True:
        raise RuntimeError("Phase 9E-B4 fixture is not valid")
    contract = value.get("class_balance_contract")
    if not isinstance(contract, dict) or contract != class_balance_contract():
        raise RuntimeError("Phase 9E-B4 threshold/formula contract changed")
    lock = value.get("test_lock")
    required_lock = {
        "test_assignments_seen": True,
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }
    if not isinstance(lock, dict) or any(lock.get(key) != expected for key, expected in required_lock.items()):
        raise RuntimeError("Phase 9E-B4 TEST lock evidence is invalid")
    if value.get("head_count") != 20 or len(value.get("head_summaries", [])) != 20:
        raise RuntimeError("Phase 9E-B4 fixture does not cover 20 heads")
    hashes = value.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(OUTPUT_ARTIFACTS):
        raise RuntimeError("Phase 9E-B4 artifact fingerprint inventory is incomplete")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or set(digest) - set("0123456789abcdef")
        for digest in hashes.values()
    ):
        raise RuntimeError("Phase 9E-B4 artifact SHA-256 is invalid")
    return value


def reseal_derived_artifacts(
    output: Path, fixture: Path, *, b3_root: Path = DEFAULT_B3_ROOT
) -> dict[str, object]:
    """Rebuild recommendations/report hashes from completed semantic counts.

    This path is deliberately source-free: it cannot open the corpus, raw
    records, or target sidecars and therefore cannot weaken the TEST lock.
    """

    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("valid") is not True:
        raise RuntimeError("only a completed valid B4 audit may be resealed")
    _b3_summary, b3_hashes = _verify_b3(b3_root)
    summary["input_fingerprints"]["b3_artifact_sha256"] = b3_hashes
    class_rows = _read_jsonl(output / "class_counts.jsonl")
    joint_rows = _read_jsonl(output / "joint_tuple_counts.jsonl")
    head_payload = json.loads((output / "head_balance_summary.json").read_text(encoding="utf-8"))
    heads = head_payload.get("heads")
    if not isinstance(heads, list) or len(heads) != 20:
        raise RuntimeError("completed B4 head summary is invalid")
    by_task_split: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for task in PRODUCTION_AUDIT_TASKS:
        for split in ("train", "validation"):
            by_task_split[(task.task_id, split)] = [
                row
                for row in class_rows
                if row["task_id"] == task.task_id and row["split"] == split
            ]
    remediated_heads: list[dict[str, object]] = []
    for old in heads:
        head = dict(old)
        task_id = str(head["task_id"])
        train = by_task_split[(task_id, "train")]
        validation = by_task_split[(task_id, "validation")]
        head["train_missing_class_count"] = sum(
            row["support_tier"] == "absent" for row in train
        )
        status, reasons = recommend_head_trainability(
            vocabulary_size=int(head["vocabulary_size"]),
            train_tiers=[str(row["support_tier"]) for row in train],
            validation_tiers=[str(row["validation_tier"]) for row in validation],
            available_train_components=int(head["available_train_component_count"]),
            majority_share=float(head["majority_share"]),
            max_to_min_nonzero_ratio=(
                None
                if head["max_to_min_nonzero_ratio"] is None
                else float(head["max_to_min_nonzero_ratio"])
            ),
            normalized_entropy=float(head["normalized_entropy"]),
        )
        head["recommendation"] = status
        head["recommendation_reasons"] = reasons
        remediated_heads.append(head)
    recommendations = recommendation_payload(remediated_heads)
    production_rows = [row for row in class_rows if row["task_id"] != "quality_compatibility"]
    weights = candidate_class_weights(production_rows, remediated_heads)
    semantic = semantic_fingerprint(
        class_rows=class_rows,
        head_summaries=remediated_heads,
        joint_rows=joint_rows,
        recommendations=recommendations,
        weights=weights,
    )
    summary["recommendation_groups"] = {
        status: [
            str(row["task_id"])
            for row in remediated_heads
            if row["recommendation"] == status
        ]
        for status in (
            "trainable",
            "trainable_with_reweighting",
            "insufficient_support",
            "descriptive_only",
        )
    }
    summary["candidate_class_weight_summary"] = _candidate_weight_summary(weights)
    compatibility_rows = [
        row for row in class_rows if row["task_id"] == "quality_compatibility"
    ]
    summary["quality"] = quality_focus_summary(
        production_rows, compatibility_rows
    )
    summary["semantic_fingerprint"] = semantic
    _write_json(output / "head_balance_summary.json", {"heads": remediated_heads})
    _write_json(output / "trainability_recommendations.json", recommendations)
    _write_json(output / "candidate_class_weights.json", weights)
    (output / "AUDIT_REPORT.md").write_text(
        _report(summary, remediated_heads), encoding="utf-8", newline="\n"
    )
    summary["artifact_sha256"] = _artifact_hashes(
        output,
        (
            "class_counts.jsonl",
            "head_balance_summary.json",
            "joint_tuple_counts.jsonl",
            "trainability_recommendations.json",
            "candidate_class_weights.json",
            "AUDIT_REPORT.md",
        ),
    )
    _write_json(output / "audit_summary.json", summary)
    _write_json(
        fixture,
        _compact_fixture(summary=summary, heads=remediated_heads, output=output),
    )
    return summary


def build_audit(
    corpus_root: Path,
    *,
    b3_root: Path,
    output: Path,
    fixture: Path,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("production output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    b3_summary, b3_hashes = _verify_b3(b3_root)
    assignments, candidate, descriptors = _target_free_inputs(b3_root)
    split_counts = Counter(str(row["split"]) for row in assignments)
    if split_counts != {"train": 1295, "validation": 162, "test": 162}:
        raise RuntimeError(f"frozen split counts changed: {dict(split_counts)}")

    production = ClassBalanceAccumulator()
    compatibility = ClassBalanceAccumulator((COMPATIBILITY_QUALITY_TASK,))
    joint = JointTupleAccumulator()
    access = Counter()
    descriptor_failures: list[str] = []
    # Critical lock boundary: TEST is discarded before record path resolution,
    # raw conversion, target-sidecar materialization, or target counting.
    allowed = [row for row in assignments if row["split"] in ("train", "validation")]
    for ordinal, assignment in enumerate(allowed, start=1):
        split = str(assignment["split"])
        record_id = str(assignment["record_id"])
        record = _selective_record(corpus_root, candidate[record_id])
        converted = convert_dilemmadata_record(record)
        if not isinstance(converted, DilemmadataAccepted):
            raise RuntimeError(f"frozen raw record failed conversion: {record_id}")
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
        record_observations = observations_from_sidecar(sidecar, split=split)  # type: ignore[arg-type]
        production.add_record(record_observations)
        compatibility.add_record(project_quality_record(record_observations))
        joint.add_record(joint_observations_from_sidecar(sidecar, split=split))  # type: ignore[arg-type]
        access[f"{split}_target_records_opened"] += 1
        access[f"{split}_target_rows_loaded"] += record.note_row_count
        if ordinal % 100 == 0:
            print(f"audited {ordinal}/{len(allowed)} TRAIN/VALIDATION records", flush=True)
    if descriptor_failures:
        raise RuntimeError(
            f"expanded target sidecars differ from frozen B3 descriptors: {descriptor_failures[:16]}"
        )

    class_rows = production.class_rows()
    compatibility_rows = compatibility.class_rows()
    all_class_rows = [*class_rows, *compatibility_rows]
    heads = production.head_summaries(class_rows)
    joint_rows = joint.rows()
    joint_summary = joint.summary(joint_rows)
    recommendations = recommendation_payload(heads)
    weights = candidate_class_weights(class_rows, heads)
    roman = roman_numeral_summary(class_rows)
    quality = quality_focus_summary(class_rows, compatibility_rows)
    recommendation_groups = {
        status: [str(row["task_id"]) for row in heads if row["recommendation"] == status]
        for status in (
            "trainable",
            "trainable_with_reweighting",
            "insufficient_support",
            "descriptive_only",
        )
    }
    semantic = semantic_fingerprint(
        class_rows=all_class_rows,
        head_summaries=heads,
        joint_rows=joint_rows,
        recommendations=recommendations,
        weights=weights,
    )
    test_lock = {
        "test_assignments_seen": split_counts["test"] > 0,
        "test_assignment_record_count": split_counts["test"],
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }
    valid = (
        len(heads) == len(PRODUCTION_TASKS) == 20
        and len(class_rows)
        == sum(task.class_count for task in PRODUCTION_TASKS) * 2
        and not descriptor_failures
        and test_lock["test_target_records_opened"] == 0
        and test_lock["test_target_rows_loaded"] == 0
    )
    summary: dict[str, object] = {
        "class_balance_contract": class_balance_contract(),
        "candidate_class_weight_summary": _candidate_weight_summary(weights),
        "dataset_changed": False,
        "graphs_changed": False,
        "head_count": len(heads),
        "input_fingerprints": {
            "b3_artifact_sha256": b3_hashes,
            "b3_semantic_fingerprint": b3_summary["semantic_fingerprint"],
            "corpus_content_fingerprint": b3_summary["snapshot"]["content_fingerprint"],
            "split_fingerprint": b3_summary["test_lock"]["test_assignment_fingerprint"],
        },
        "joint_tuples": joint_summary,
        "loss_or_sampling_policy_frozen": False,
        "model_implemented": False,
        "quality": quality,
        "recommendation_groups": recommendation_groups,
        "roman_numeral_184": roman,
        "schema": CLASS_BALANCE_SCHEMA,
        "semantic_fingerprint": semantic,
        "split_counts": dict(split_counts),
        "target_access": {
            "train_target_records_opened": access["train_target_records_opened"],
            "train_target_rows_loaded": access["train_target_rows_loaded"],
            "validation_target_records_opened": access["validation_target_records_opened"],
            "validation_target_rows_loaded": access["validation_target_rows_loaded"],
        },
        "test_lock": test_lock,
        "training_run": False,
        "valid": valid,
        "validation_inference_run": False,
        "vocabularies_changed": False,
    }

    _write_jsonl(output / "class_counts.jsonl", all_class_rows)
    _write_json(output / "head_balance_summary.json", {"heads": heads})
    _write_jsonl(output / "joint_tuple_counts.jsonl", joint_rows)
    _write_json(output / "trainability_recommendations.json", recommendations)
    _write_json(output / "candidate_class_weights.json", weights)
    report = _report(summary, heads)
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    summary["artifact_sha256"] = _artifact_hashes(
        output,
        (
            "class_counts.jsonl",
            "head_balance_summary.json",
            "joint_tuple_counts.jsonl",
            "trainability_recommendations.json",
            "candidate_class_weights.json",
            "AUDIT_REPORT.md",
        ),
    )
    _write_json(output / "audit_summary.json", summary)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    compact = _compact_fixture(summary=summary, heads=heads, output=output)
    _write_json(fixture, compact)
    if not valid:
        raise RuntimeError("Phase 9E-B4 audit failed; inspect audit_summary.json")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--b3-root", type=Path, default=DEFAULT_B3_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reseal-derived", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        value = check_fixture(args.fixture)
        print(canonical_json({"valid": value["valid"]}, indent=2))
        return 0
    if args.reseal_derived:
        summary = reseal_derived_artifacts(
            args.output or default_output(), args.fixture, b3_root=args.b3_root
        )
        print(
            canonical_json(
                {
                    "semantic_fingerprint": summary["semantic_fingerprint"],
                    "source_free": True,
                    "valid": summary["valid"],
                },
                indent=2,
            )
        )
        return 0
    summary = build_audit(
        args.root,
        b3_root=args.b3_root,
        output=args.output or default_output(),
        fixture=args.fixture,
    )
    print(canonical_json({"semantic_fingerprint": summary["semantic_fingerprint"], "valid": summary["valid"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
