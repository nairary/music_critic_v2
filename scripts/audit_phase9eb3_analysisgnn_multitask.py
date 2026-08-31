#!/usr/bin/env python3
"""Materialize and verify the Phase 9E-B3 AnalysisGNN multi-task contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

from music_critic.adapters import (
    DilemmadataAccepted,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
    load_dilemmadata_target_metadata_index,
)
from music_critic.data import dumps_piece
from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    ANALYSISGNN_COMMIT,
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_NAMESPACE,
    DATASET_MANIFEST_VERSION,
    EXPECTED_FULL_COUNTS,
    EXPECTED_PAPER_COUNTS,
    FULL_RAW_UNIVERSE_ID,
    PAPER_CANDIDATE_UNIVERSE_ID,
    PRODUCTION_REGISTRY_ID,
    PRODUCTION_TASKS,
    TARGET_SIDECAR_VERSION,
    full_raw_manifest,
    materialize_target_sidecar_descriptor,
    metric_contract,
    overlap_exclusions,
    paper_candidate_manifest,
    paper_candidate_records,
    pinned_code_reference_registry,
    production_task_registry,
    read_source_rows,
    source_component_rows,
    split_summary,
    stable_split_assignments,
    test_lock_manifest,
    validate_loaded_registry,
    vocabularies_payload,
    _record_availability_from_rows,
)
from music_critic.graph import build_raw_graph, graph_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO_ROOT
    / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/phase9eb3/analysisgnn-multitask-contract-01290f5"
)
DEFAULT_B2 = (
    REPO_ROOT
    / "outputs/phase9eb2/dilemmadata-coverage-remediation-877c168"
)
DEFAULT_B1 = REPO_ROOT / "outputs/phase9eb1/common-data"
DEFAULT_FIXTURE = (
    REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb3_multitask_contract.json"
)
EXPECTED_SNAPSHOT_FINGERPRINT = (
    "8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312"
)
EXPECTED_B2_SEMANTIC_FINGERPRINT = (
    "831890a6f1b1d6a33c2c213201ca38ff290326160bb32a84d0c6d019cf481218"
)
EXPECTED_B2_SHA256 = {
    "AUDIT_REPORT.md": "315771023e6b28a293e815bd3c90620bbe5e54c73d57cfe130e4b42cca2b13d6",
    "artifact_sha256.json": "586fb71204bed5aba77c97eaadb17110fbebcb37768f8071960356ea9302adae",
    "audit_summary.json": "e0551a3e996199a509b7e3de1c7d1e7c21d83bc885469893eb813bfd7358c6dc",
    "quarantine_records.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "record_results.jsonl": "861968885f83a8b455706ddc8d3940e1c1f3935259fe32b03166039c0730fd66",
    "repair_evidence.jsonl": "82c3fecd80c6d176d78600fc8651c5339b71170b3019b7656fadfb27afb8281c",
    "target_sidecar_smoke.json": "360998198c6025b612662961817b3c62ab8d936739577b8380b8f3bd681f81be",
}
EXPECTED_B1_MANIFEST_SHA256 = (
    "4c35f2075493307b19b69048a606df4b8489a5a2216f8e24e4eb4e0e68650da2"
)
EXPECTED_B1_TREE_FINGERPRINT = (
    "92a49df4f9adb2627c4470b551d18d671e7df495f9165ca8f4d5d7d27ae68585"
)
REQUIRED_ARTIFACTS = (
    "full_raw_manifest.json",
    "paper_candidate_manifest.json",
    "source_components.jsonl",
    "overlap_exclusions.jsonl",
    "paper_candidate_records.jsonl",
    "task_registry.json",
    "pinned_code_reference_registry.json",
    "vocabularies.json",
    "entity_registry.jsonl",
    "target_sidecars.jsonl",
    "split_assignments.jsonl",
    "split_summary.json",
    "metric_contract.json",
    "dataset_manifest.json",
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


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def _tree_snapshot(root: Path) -> dict[str, object]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _hash_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {"file_count": len(rows), "fingerprint": fingerprint(rows)}


def _verify_b2(root: Path) -> tuple[dict[str, str], dict[str, object]]:
    observed = {name: _hash_file(root / name) for name in EXPECTED_B2_SHA256}
    if observed != EXPECTED_B2_SHA256:
        differences = {
            name: {"expected": EXPECTED_B2_SHA256[name], "observed": observed.get(name)}
            for name in EXPECTED_B2_SHA256
            if observed.get(name) != EXPECTED_B2_SHA256[name]
        }
        raise RuntimeError(f"Phase 9E-B2 artifact SHA-256 mismatch: {differences}")
    summary = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("semantic_fingerprint") != EXPECTED_B2_SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Phase 9E-B2 semantic fingerprint mismatch: "
            f"{summary.get('semantic_fingerprint')}"
        )
    if summary.get("outcomes", {}).get("quarantined") != 0:
        raise RuntimeError("Phase 9E-B2 quarantine is not empty")
    return observed, summary


def _metadata_map(root: Path, records) -> dict[str, dict[str, str]]:
    index = load_dilemmadata_target_metadata_index(root, records)
    return {row.record_id: dict(row.fields) for row in index.records}


def _artifact_hashes(output: Path) -> dict[str, str]:
    return {
        path.name: _hash_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"audit_summary.json", "AUDIT_REPORT.md"}
    }


def _contract_fingerprints(output: Path) -> dict[str, str]:
    names = (
        "dataset_manifest.json",
        "full_raw_manifest.json",
        "metric_contract.json",
        "paper_candidate_manifest.json",
        "pinned_code_reference_registry.json",
        "split_summary.json",
        "task_registry.json",
        "vocabularies.json",
    )
    values = {
        name: json.loads((output / name).read_text(encoding="utf-8"))
        for name in names
    }
    return {name: str(value["fingerprint"]) for name, value in values.items()}


def _report(summary: Mapping[str, object]) -> str:
    universe = summary["universes"]
    split = summary["split"]
    tasks = summary["task_inventory"]
    joint = summary["joint_metric"]
    return f"""# Phase 9E-B3 AnalysisGNN multi-task contract audit

- valid: `{str(summary['valid']).lower()}`
- ready: `{str(summary['ready']).lower()}`
- model implemented: `false`
- training run: `false`
- TEST evaluated: `false`
- full raw universe: `{universe['full_raw']['total']}` (AN `{universe['full_raw']['an_joint']}`, DLC `{universe['full_raw']['dlc']}`)
- paper-candidate universe: `{universe['paper_candidate']['total']}` (AN `{universe['paper_candidate']['an_joint']}`, DLC `{universe['paper_candidate']['dlc']}`)
- overlap exclusions: `14`
- paper / pinned-code / production task counts: `{tasks['paper']} / {tasks['pinned_code']} / {tasks['production']}`
- split TRAIN / VALIDATION / TEST: `{split['record_counts']['train']} / {split['record_counts']['validation']} / {split['record_counts']['test']}`
- component leakage failures: `{split['component_leakage_failure_count']}`
- joint structural support TRAIN / VALIDATION: `{joint['train']} / {joint['validation']}`
- TEST assignment frozen: `true`

This is an AnalysisGNN-derived, paper-compatible corrected contract, not an
exact official reproduction.  The pinned implementation has documented
literal, alias, missing-mask, and task-inventory defects; the external cadence
corpus is not available.  No model, training, inference, or TEST metric was run.
"""


def _audit_record_worker(args) -> dict[str, object]:
    record, previous, is_candidate = args
    converted = convert_dilemmadata_record(record)
    if not isinstance(converted, DilemmadataAccepted):
        return {
            "quarantine": {
                "categories": list(converted.categories),
                "record_id": record.record_id,
            },
            "record_id": record.record_id,
        }
    graph = build_raw_graph(converted.piece, assume_valid=True)
    observed_graph = graph_fingerprint(graph)
    canonical_sha = sha256(dumps_piece(converted.piece).encode("utf-8")).hexdigest()
    raw_mismatch = (
        previous is None
        or previous.get("graph_fingerprint") != observed_graph
        or previous.get("canonical_piece_sha256") != canonical_sha
    )
    sidecar = None
    deterministic = True
    if is_candidate:
        sidecar = materialize_target_sidecar_descriptor(converted)
        second = materialize_target_sidecar_descriptor(converted)
        deterministic = sidecar["fingerprint"] == second["fingerprint"]
    return {
        "deterministic": deterministic,
        "dialect": record.dialect,
        "raw_mismatch": raw_mismatch,
        "record_id": record.record_id,
        "sidecar": sidecar,
        "source_component_id": record.source_group_id,
    }


def build_audit(
    root: Path,
    *,
    output: Path,
    b2_root: Path = DEFAULT_B2,
    b1_root: Path = DEFAULT_B1,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    b2_hashes, b2_summary = _verify_b2(b2_root)
    if _hash_file(b1_root / "manifest.json") != EXPECTED_B1_MANIFEST_SHA256:
        raise RuntimeError("Phase 9E-B1 manifest changed before B3 audit")
    b1_before = _tree_snapshot(b1_root)
    if b1_before["fingerprint"] != EXPECTED_B1_TREE_FINGERPRINT:
        raise RuntimeError("Phase 9E-B1 output tree changed before B3 audit")

    discovery = discover_dilemmadata_corpus(root, require_valid=True)
    if discovery.content_fingerprint != EXPECTED_SNAPSHOT_FINGERPRINT:
        raise RuntimeError(
            "Dilemmadata snapshot fingerprint mismatch: expected "
            f"{EXPECTED_SNAPSHOT_FINGERPRINT}, observed {discovery.content_fingerprint}"
        )
    full_manifest = full_raw_manifest(discovery)
    candidate_manifest = paper_candidate_manifest(discovery.records)
    candidate = paper_candidate_records(discovery.records)
    candidate_ids = {row.record_id for row in candidate}
    exclusions = overlap_exclusions(discovery.records)
    metadata = _metadata_map(root, discovery.records)
    components = source_component_rows(discovery.records, metadata=metadata)

    availability: dict[str, dict[str, bool]] = {}
    full_availability_counts = Counter()
    candidate_availability_counts = Counter()
    for record in sorted(discovery.records, key=lambda row: row.record_id):
        values = _record_availability_from_rows(record.dialect, read_source_rows(record))
        availability[record.record_id] = values
        full_availability_counts.update(
            task_id for task_id, available in values.items() if available
        )
        if record.record_id in candidate_ids:
            candidate_availability_counts.update(
                task_id for task_id, available in values.items() if available
            )
    assignments = stable_split_assignments(candidate, availability)
    assignment_by_record = {row.record_id: row for row in assignments}
    split_payload = split_summary(assignments, availability)
    leakage = split_payload["component_leakage"]
    assert isinstance(leakage, dict)
    component_leakage_failures = sum(len(value) for value in leakage.values())

    pinned_registry = pinned_code_reference_registry()
    task_registry = production_task_registry()
    vocabulary_registry = vocabularies_payload()
    validate_loaded_registry(task_registry, vocabulary_registry)
    metric_payload = metric_contract()
    lock_payload = test_lock_manifest(assignments)
    previous_records = {
        str(row["record_id"]): row
        for row in _read_jsonl(b2_root / "record_results.jsonl")
    }
    raw_failures: list[str] = []
    quarantine: list[dict[str, object]] = []
    determinism_failures: list[str] = []
    entity_counts = Counter()
    relation_counts = Counter()
    task_states_by_split: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    joint_support = Counter()
    sidecar_fingerprints: list[list[str]] = []
    entity_registry_rows: list[dict[str, object]] = []

    target_path = output / "target_sidecars.jsonl"
    ordered_records = tuple(sorted(discovery.records, key=lambda row: row.record_id))
    worker_inputs = tuple(
        (record, previous_records.get(record.record_id), record.record_id in candidate_ids)
        for record in ordered_records
    )
    worker_count = int(os.environ.get("MUSIC_CRITIC_PHASE9EB3_WORKERS", "3"))
    if worker_count < 1 or worker_count > 3:
        raise RuntimeError("MUSIC_CRITIC_PHASE9EB3_WORKERS must be in [1, 3]")
    pool = None
    if worker_count == 1:
        results = map(_audit_record_worker, worker_inputs)
    else:
        pool = multiprocessing.get_context("fork").Pool(processes=worker_count)
        results = pool.imap(_audit_record_worker, worker_inputs, chunksize=1)
    with target_path.open("w", encoding="utf-8", newline="\n") as target_handle:
        for ordinal, result in enumerate(results, start=1):
            record_id = str(result["record_id"])
            if result.get("quarantine") is not None:
                quarantine.append(result["quarantine"])
                continue
            if result["raw_mismatch"]:
                raw_failures.append(record_id)
            sidecar = result["sidecar"]
            if sidecar is None:
                continue
            if not result["deterministic"]:
                determinism_failures.append(record_id)
            split_name = assignment_by_record[record_id].split
            joint_support[split_name] += int(sidecar["joint_structural_support"])
            for task_id, states in sidecar["task_states"].items():
                task_states_by_split[split_name][task_id].update(states)
            entity_counts.update(sidecar["entity_counts"])
            relation_counts.update(sidecar["relation_counts"])
            sidecar_fingerprints.append([record_id, str(sidecar["fingerprint"])])
            entity_row = {
                "dialect": result["dialect"],
                "entities_fingerprint": sidecar["entities_fingerprint"],
                "entity_counts": sidecar["entity_counts"],
                "record_id": record_id,
                "relations_fingerprint": sidecar["relations_fingerprint"],
                "relation_counts": sidecar["relation_counts"],
                "sidecar_fingerprint": sidecar["fingerprint"],
                "source_component_id": result["source_component_id"],
                "version": TARGET_SIDECAR_VERSION,
            }
            entity_registry_rows.append(entity_row)
            target_descriptor = {
                **entity_row,
                "encoding": "content-addressed-per-record-sidecar-descriptor-v1",
                "full_payload_api": (
                    "music_critic.experiments.analysisgnn."
                    "materialize_target_sidecar"
                ),
                "joint_structural_support": sidecar["joint_structural_support"],
                "target_states_fingerprint": sidecar["target_states_fingerprint"],
                "task_states": sidecar["task_states"],
            }
            target_handle.write(
                json.dumps(
                    target_descriptor,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            if ordinal % 100 == 0:
                print(f"materialized {ordinal}/{len(discovery.records)} raw records", flush=True)
    if pool is not None:
        pool.close()
        pool.join()

    paper_record_rows = [
        {
            "dialect": row.dialect,
            "piece_id": row.piece_id,
            "raw_equivalence_id": row.raw_equivalence_id,
            "raw_projection_sha256": row.raw_projection_sha256,
            "record_id": row.record_id,
            "source_component_id": row.source_group_id,
            "task_availability": availability[row.record_id],
        }
        for row in candidate
    ]
    assignment_rows = [row.__dict__ if hasattr(row, "__dict__") else {
        "record_id": row.record_id,
        "dialect": row.dialect,
        "source_component_id": row.source_component_id,
        "split": row.split,
        "assignment_algorithm": row.assignment_algorithm,
        "assignment_namespace": row.assignment_namespace,
    } for row in assignments]
    _write_json(output / "full_raw_manifest.json", full_manifest)
    _write_json(output / "paper_candidate_manifest.json", candidate_manifest)
    _write_jsonl(output / "source_components.jsonl", components)
    _write_jsonl(output / "overlap_exclusions.jsonl", exclusions)
    _write_jsonl(output / "paper_candidate_records.jsonl", paper_record_rows)
    _write_json(output / "task_registry.json", task_registry)
    _write_json(output / "pinned_code_reference_registry.json", pinned_registry)
    _write_json(output / "vocabularies.json", vocabulary_registry)
    _write_jsonl(output / "entity_registry.jsonl", entity_registry_rows)
    _write_jsonl(output / "split_assignments.jsonl", assignment_rows)
    _write_json(output / "split_summary.json", split_payload)
    _write_json(output / "metric_contract.json", metric_payload)

    record_splits: dict[str, set[str]] = defaultdict(set)
    raw_equivalence_splits: dict[str, set[str]] = defaultdict(set)
    for record in candidate:
        split_name = assignment_by_record[record.record_id].split
        record_splits[record.record_id].add(split_name)
        raw_equivalence_splits[record.raw_equivalence_id].add(split_name)
    b1_after = _tree_snapshot(b1_root)
    dataset_payload: dict[str, object] = {
        "analysisgnn_external_commit": ANALYSISGNN_COMMIT,
        "artifact_versions": {
            "dataset": DATASET_MANIFEST_VERSION,
            "full_universe": FULL_RAW_UNIVERSE_ID,
            "paper_candidate": PAPER_CANDIDATE_UNIVERSE_ID,
            "production_registry": PRODUCTION_REGISTRY_ID,
        },
        "cadence_external_corpus_available": False,
        "cadence_external_corpus_included": False,
        "entity_registry_fingerprint": fingerprint(entity_registry_rows),
        "full_raw_manifest_fingerprint": full_manifest["fingerprint"],
        "paper_candidate_manifest_fingerprint": candidate_manifest["fingerprint"],
        "raw_graph_fingerprints_unchanged": not raw_failures,
        "raw_graph_target_availability_included": False,
        "sidecar_fingerprint": fingerprint(sidecar_fingerprints),
        "split_fingerprint": split_payload["fingerprint"],
        "target_sidecar_version": TARGET_SIDECAR_VERSION,
        "task_registry_fingerprint": task_registry["fingerprint"],
        "test_lock": lock_payload,
        "vocabularies_fingerprint": vocabulary_registry["fingerprint"],
    }
    dataset_payload["fingerprint"] = fingerprint(dataset_payload)
    _write_json(output / "dataset_manifest.json", dataset_payload)

    valid = not any(
        (
            quarantine,
            raw_failures,
            determinism_failures,
            component_leakage_failures,
            [key for key, values in record_splits.items() if len(values) != 1],
            [key for key, values in raw_equivalence_splits.items() if len(values) != 1],
            [] if joint_support["train"] > 0 else ["zero_train_joint_support"],
            [] if joint_support["validation"] > 0 else ["zero_validation_joint_support"],
            [] if b1_before == b1_after else ["phase9eb1_tree_changed"],
        )
    )
    summary: dict[str, object] = {
        "artifacts": {},
        "b2_audit": {
            "artifact_sha256": b2_hashes,
            "semantic_fingerprint": b2_summary["semantic_fingerprint"],
        },
        "contracts": _contract_fingerprints(output),
        "entity_contract": {
            "entity_counts": dict(entity_counts),
            "relation_counts": dict(relation_counts),
            "shared_harmonic_entity_identity": True,
            "zero_joint_quality_inversion_support_resolved": joint_support["train"] > 0,
        },
        "failures": {
            "determinism": determinism_failures,
            "quarantine": quarantine,
            "raw_graph": raw_failures,
        },
        "joint_metric": {
            "components": [
                "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
            ],
            "test": "not_evaluated",
            "train": joint_support["train"],
            "validation": joint_support["validation"],
        },
        "model_implemented": False,
        "phase9eb1_preservation": {
            "after": b1_after,
            "before": b1_before,
            "manifest_sha256": EXPECTED_B1_MANIFEST_SHA256,
            "split_counts": {"test": 71, "train": 577, "validation": 71},
            "unchanged": b1_before == b1_after,
        },
        "ready": valid,
        "snapshot": {
            "content_fingerprint": discovery.content_fingerprint,
            "installation_byte_count": discovery.installation_byte_count,
            "installation_file_count": discovery.installation_file_count,
            "quarantine_count": len(quarantine),
        },
        "split": {
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "assignment_namespace": ASSIGNMENT_NAMESPACE,
            "component_leakage_failure_count": component_leakage_failures,
            "component_counts": split_payload["component_counts"],
            "component_leakage": leakage,
            "record_counts": split_payload["record_counts"],
        },
        "structural_availability": {
            "full_raw": dict(sorted(full_availability_counts.items())),
            "paper_candidate": dict(sorted(candidate_availability_counts.items())),
            "per_split_entity_states": {
                split_name: {
                    task_id: dict(states)
                    for task_id, states in sorted(task_rows.items())
                }
                for split_name, task_rows in sorted(task_states_by_split.items())
            },
        },
        "task_inventory": {
            "paper": 20,
            "pinned_code": 21,
            "production": len(PRODUCTION_TASKS),
            "production_heads": [row.task_id for row in PRODUCTION_TASKS],
        },
        "test_evaluated": False,
        "test_lock": lock_payload,
        "training_run": False,
        "universes": {
            "full_raw": EXPECTED_FULL_COUNTS,
            "paper_candidate": EXPECTED_PAPER_COUNTS,
            "source_component_count_full": len(components),
            "source_component_count_paper_candidate": len(
                {row.source_group_id for row in candidate}
            ),
        },
        "valid": valid,
    }
    hashes = _artifact_hashes(output)
    summary["artifacts"] = hashes
    summary["semantic_fingerprint"] = fingerprint(
        {key: value for key, value in summary.items() if key not in {"artifacts", "semantic_fingerprint"}}
    )
    _write_json(output / "audit_summary.json", summary)
    (output / "AUDIT_REPORT.md").write_text(_report(summary), encoding="utf-8", newline="\n")
    if not valid:
        raise RuntimeError("Phase 9E-B3 audit failed; inspect audit_summary.json")
    return summary


def reseal_audit_summary(output: Path) -> dict[str, object]:
    """Bind current contract fingerprints after a completed full audit."""

    summary_path = output / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("valid") is not True or summary.get("ready") is not True:
        raise RuntimeError("only a completed valid/ready audit may be resealed")
    summary["contracts"] = _contract_fingerprints(output)
    summary["semantic_fingerprint"] = fingerprint(
        {
            key: value
            for key, value in summary.items()
            if key not in {"artifacts", "semantic_fingerprint"}
        }
    )
    _write_json(summary_path, summary)
    (output / "AUDIT_REPORT.md").write_text(
        _report(summary), encoding="utf-8", newline="\n"
    )
    return summary


def check_fixture(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "phase9eb3-source-free-fixture-v1":
        raise RuntimeError("unknown Phase 9E-B3 source-free fixture")
    expected = value.pop("fingerprint", None)
    if fingerprint(value) != expected:
        raise RuntimeError("Phase 9E-B3 source-free fixture fingerprint mismatch")
    value["fingerprint"] = expected
    if value.get("valid") is not True or value.get("ready") is not True:
        raise RuntimeError("Phase 9E-B3 source-free fixture is not ready")
    if value.get("universes") != {
        "full_raw": EXPECTED_FULL_COUNTS,
        "paper_candidate": EXPECTED_PAPER_COUNTS,
    }:
        raise RuntimeError("Phase 9E-B3 source-free universe counts changed")
    if value.get("task_counts") != {"paper": 20, "pinned_code": 21, "production": 20}:
        raise RuntimeError("Phase 9E-B3 task inventory counts changed")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2)
    parser.add_argument("--b1-root", type=Path, default=DEFAULT_B1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check:
        value = check_fixture(args.fixture)
        print(canonical_json({"ready": value["ready"], "valid": value["valid"]}, indent=2))
        return 0
    configured = args.root or os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
    root = Path(configured) if configured else DEFAULT_ROOT
    summary = build_audit(root, output=args.output, b2_root=args.b2_root, b1_root=args.b1_root)
    print(canonical_json({"ready": summary["ready"], "valid": summary["valid"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
