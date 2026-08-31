#!/usr/bin/env python3
"""Full-corpus Phase 9E-B2 Dilemmadata raw coverage remediation audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

from music_critic.adapters import (
    DILEMMADATA_COVERAGE_REMEDIATION_REPORT_VERSION,
    DILEMMADATA_RELEASE_COMMIT,
    DilemmadataAccepted,
    DilemmadataCorpusRecord,
    DilemmadataQuarantine,
    build_dilemmadata_target_sidecar,
    convert_dilemmadata_record,
    dilemmadata_raw_repair_evidence_payload,
    discover_dilemmadata_corpus,
    load_dilemmadata_target_metadata_index,
    validate_dilemmadata_raw_repair_evidence,
)
from music_critic.data import dumps_piece, loads_piece
from music_critic.graph import (
    build_raw_graph,
    dumps_graph,
    graph_fingerprint,
    validate_raw_graph,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS_AUDIT = (
    REPO_ROOT / "outputs/phase9eb2/dilemmadata-coverage-audit-e607c934"
)
DEFAULT_PHASE9EB1_COMMON = REPO_ROOT / "outputs/phase9eb1/common-data"
PREVIOUS_AUDIT_SHA256 = {
    "AUDIT_REPORT.md": "c283a6b07f42ba2aa373f8d151d219d0a103d627533a7dab668619dc6e3a4fae",
    "audit_summary.json": "58d980f9af2a3ff34495e9d4ba35acff43ad4ab0e4d9000546d2a665b4bf4a57",
    "quarantine_records.jsonl": "99d6ce7c05a09b16f211860e2851a28d2c674d83cae59fb0318630d8c7816dc1",
    "reason_counts.json": "3aabbe9d771945c8cdb6ede2c7966e778bd4ccecc13a0e9ecc91893124df2f16",
    "record_reconciliation.jsonl": "d8e6ffceeb60128addd7963b046e97090accbcacfac5a0a7cab0245a8ffc1441",
    "recoverability_counts.json": "53ead39ec19ef5a6e15ed96484927de091676b017fbc70a97eb1dbb7f22b865e",
    "run_coverage_audit.py": "c33000001bf6c523f95770d136a96f96c7fa0570e11f21a897c07972ccf6a276",
    "source_inventory.jsonl": "8dde503f1b6d110157650e24e6f37e8b38f6d5df71cd94b53b8e33ba6f33e7e6",
    "task_availability.json": "550deea64438b6f0a7c7503c7570dc7647aeb023cf79687efe2e88e6f84c89b3",
}


def _canonical_json(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value, indent=None).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(_canonical_json(row, indent=None) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _acyclic_next_in_track(graph) -> bool:
    edge_type = ("note", "next_in_track", "note")
    edge_index = graph[edge_type].edge_index
    node_count = int(graph["note"].num_nodes)
    mutable: dict[int, list[int]] = defaultdict(list)
    indegree = [0] * node_count
    for source, target in edge_index.t().tolist():
        source_index = int(source)
        target_index = int(target)
        mutable[source_index].append(target_index)
        indegree[target_index] += 1
    queue = [index for index, value in enumerate(indegree) if value == 0]
    cursor = 0
    visited_count = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        visited_count += 1
        for target in mutable.get(node, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited_count == node_count


def _context_gate(graph) -> dict[str, object]:
    note_count = int(graph["note"].num_nodes)
    bar_edges = graph[("note", "belongs_to_bar", "bar")].edge_index
    onset_edges = graph[("note", "in_onset", "onset")].edge_index
    beat_edges = graph[("onset", "belongs_to_beat", "beat")].edge_index
    bar_counts = Counter(int(value) for value in bar_edges[0].tolist())
    onset_by_note: dict[int, list[int]] = defaultdict(list)
    for note, onset in onset_edges.t().tolist():
        onset_by_note[int(note)].append(int(onset))
    beat_counts_by_onset = Counter(int(value) for value in beat_edges[0].tolist())
    return {
        "each_note_has_one_bar": all(bar_counts[index] == 1 for index in range(note_count)),
        "each_note_has_beat_context": all(
            len(onset_by_note[index]) == 1
            and beat_counts_by_onset[onset_by_note[index][0]] == 1
            for index in range(note_count)
        ),
        "note_count": note_count,
    }


def _verify_previous_audit(root: Path) -> dict[str, str]:
    observed = {name: _hash_file(root / name) for name in PREVIOUS_AUDIT_SHA256}
    if observed != PREVIOUS_AUDIT_SHA256:
        raise RuntimeError("previous coverage-audit SHA256 set differs from preflight")
    return observed


def _tree_snapshot(root: Path) -> dict[str, object]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _hash_file(path),
                "size": path.stat().st_size,
            }
        )
    return {"file_count": len(rows), "fingerprint": _fingerprint(rows)}


def _target_smoke_selection(
    records_by_id: Mapping[str, DilemmadataCorpusRecord],
    accepted_record_ids: set[str],
    repair_types_by_id: Mapping[str, tuple[str, ...]],
    previous_reason_by_id: Mapping[str, str],
) -> tuple[str, ...]:
    eligible = [
        records_by_id[record_id]
        for record_id in sorted(accepted_record_ids)
        if record_id in previous_reason_by_id
        and not (
            records_by_id[record_id].dialect == "an_joint"
            and records_by_id[record_id].suggested_split in {"validation", "test"}
        )
    ]
    selected: set[str] = set()
    covered: set[tuple[str, str]] = set()
    for record in eligible:
        key = (record.dialect, previous_reason_by_id[record.record_id])
        if key not in covered:
            covered.add(key)
            selected.add(record.record_id)
    repair_types = {
        repair_type
        for record in eligible
        for repair_type in repair_types_by_id.get(record.record_id, ())
    }
    covered_repairs = {
        repair_type
        for record_id in selected
        for repair_type in repair_types_by_id.get(record_id, ())
    }
    for repair_type in sorted(repair_types - covered_repairs):
        candidate = next(
            record
            for record in eligible
            if repair_type in repair_types_by_id.get(record.record_id, ())
        )
        selected.add(candidate.record_id)
    return tuple(sorted(selected))


def build_remediation_audit(
    root: Path,
    *,
    output_dir: Path,
    previous_audit: Path = DEFAULT_PREVIOUS_AUDIT,
    phase9eb1_common: Path = DEFAULT_PHASE9EB1_COMMON,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output directory must not already contain artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_hashes = _verify_previous_audit(previous_audit)
    previous_quarantine = _read_jsonl(previous_audit / "quarantine_records.jsonl")
    previous_reason_by_id = {
        str(row["record_id"]): str(row["primary_exclusion_reason"])
        for row in previous_quarantine
    }
    previous_reconciliation = _read_jsonl(
        previous_audit / "record_reconciliation.jsonl"
    )
    overlap_ids = {
        str(row["record_id"])
        for row in previous_reconciliation
        if row["intentional_overlap_duplicate_exclusion"] is True
    }
    availability = json.loads(
        (previous_audit / "task_availability.json").read_text(encoding="utf-8")
    )
    task_availability_after = {
        task_id: row["snapshot"] for task_id, row in availability["tasks"].items()
    }
    old_manifest_path = phase9eb1_common / "manifest.json"
    old_manifest_sha256 = _hash_file(old_manifest_path)
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_by_record = {row["record_id"]: row for row in old_manifest["records"]}
    phase9eb1_tree_before = _tree_snapshot(phase9eb1_common)

    discovery = discover_dilemmadata_corpus(root)
    outcomes = Counter()
    dialect_outcomes = Counter()
    quarantine_reasons = Counter()
    repair_event_counts = Counter()
    repair_record_sets: dict[str, set[str]] = defaultdict(set)
    previous_reason_counts = Counter()
    previous_reason_repair_counts = Counter()
    record_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    accepted_record_ids: set[str] = set()
    repair_types_by_id: dict[str, tuple[str, ...]] = {}
    deterministic_failures: list[str] = []
    graph_failures: list[str] = []
    context_failures: list[str] = []
    tie_cycle_failures: list[str] = []
    negative_duration_failures: list[str] = []
    target_leakage_failures: list[str] = []
    repair_evidence_failures: list[str] = []
    preservation_failures: list[str] = []

    for ordinal, record in enumerate(discovery.records, start=1):
        first = convert_dilemmadata_record(record)
        second = convert_dilemmadata_record(record)
        outcomes[first.status] += 1
        dialect_outcomes[(record.dialect, first.status)] += 1
        if isinstance(first, DilemmadataQuarantine):
            assert isinstance(second, DilemmadataQuarantine)
            quarantine_reasons.update(first.categories)
            quarantine_rows.append(
                {
                    "categories": list(first.categories),
                    "dialect": record.dialect,
                    "messages": list(first.messages),
                    "record_id": record.record_id,
                }
            )
            if first != second:
                deterministic_failures.append(record.record_id)
            continue
        assert isinstance(second, DilemmadataAccepted)
        accepted_record_ids.add(record.record_id)
        first_piece = dumps_piece(first.piece)
        second_piece = dumps_piece(second.piece)
        first_graph = build_raw_graph(first.piece, assume_valid=True)
        second_graph = build_raw_graph(second.piece, assume_valid=True)
        validate_raw_graph(first_graph)
        validate_raw_graph(second_graph)
        first_graph_text = dumps_graph(first_graph)
        second_graph_text = dumps_graph(second_graph)
        deterministic = (
            first_piece == second_piece
            and first_graph_text == second_graph_text
            and graph_fingerprint(first_graph) == graph_fingerprint(second_graph)
            and first.repair_evidence == second.repair_evidence
        )
        if not deterministic:
            deterministic_failures.append(record.record_id)
        if first.piece.targets or first.piece.annotations:
            target_leakage_failures.append(record.record_id)
        if any(note.duration_qn.num < 0 for note in first.piece.notes):
            negative_duration_failures.append(record.record_id)
        context = _context_gate(first_graph)
        if not context["each_note_has_one_bar"] or not context[
            "each_note_has_beat_context"
        ]:
            context_failures.append(record.record_id)
        if not _acyclic_next_in_track(first_graph):
            tie_cycle_failures.append(record.record_id)
        if first.repair_evidence is not None:
            if not validate_dilemmadata_raw_repair_evidence(first.repair_evidence):
                repair_evidence_failures.append(record.record_id)
            for repair in first.repair_evidence.repairs:
                repair_event_counts[repair.repair_type] += 1
                repair_record_sets[repair.repair_type].add(record.record_id)
            repair_rows.append(dilemmadata_raw_repair_evidence_payload(first.repair_evidence))
            repair_types_by_id[record.record_id] = tuple(
                sorted({repair.repair_type for repair in first.repair_evidence.repairs})
            )
        previous_reason = previous_reason_by_id.get(record.record_id)
        if previous_reason is not None:
            previous_reason_counts[previous_reason] += 1
            if first.repair_evidence is not None:
                for repair_type in {
                    row.repair_type for row in first.repair_evidence.repairs
                }:
                    previous_reason_repair_counts[(previous_reason, repair_type)] += 1
        preserved = None
        if record.record_id in old_by_record:
            old_row = old_by_record[record.record_id]
            old_piece_path = (
                phase9eb1_common / "records" / str(old_row["piece_id"]) / "piece.json"
            )
            old_piece_text = old_piece_path.read_text(encoding="utf-8")
            old_graph = build_raw_graph(loads_piece(old_piece_text), assume_valid=True)
            preserved = bool(
                first.repair_evidence is None
                and first_piece == old_piece_text
                and first_graph_text == dumps_graph(old_graph)
                and graph_fingerprint(first_graph) == graph_fingerprint(old_graph)
            )
            if not preserved:
                preservation_failures.append(record.record_id)
        record_rows.append(
            {
                "canonical_piece_sha256": sha256(first_piece.encode("utf-8")).hexdigest(),
                "deterministic_two_pass": deterministic,
                "dialect": record.dialect,
                "graph_fingerprint": graph_fingerprint(first_graph),
                "graph_serialization_sha256": sha256(
                    first_graph_text.encode("utf-8")
                ).hexdigest(),
                "intentional_analysisgnn_overlap_exclusion": record.record_id in overlap_ids,
                "piece_id": first.piece.piece_id,
                "preserved_phase9eb1": preserved,
                "previous_exclusion_reason": previous_reason,
                "raw_only": not first.piece.targets and not first.piece.annotations,
                "record_id": record.record_id,
                "repair_evidence_fingerprint": (
                    None
                    if first.repair_evidence is None
                    else first.repair_evidence.fingerprint
                ),
            }
        )
        if ordinal % 100 == 0:
            print(f"audited {ordinal}/{len(discovery.records)}", flush=True)

    records_by_id = {record.record_id: record for record in discovery.records}
    smoke_record_ids = _target_smoke_selection(
        records_by_id,
        accepted_record_ids,
        repair_types_by_id,
        previous_reason_by_id,
    )
    smoke_selection = tuple(
        convert_dilemmadata_record(records_by_id[record_id])
        for record_id in smoke_record_ids
    )
    if not all(isinstance(outcome, DilemmadataAccepted) for outcome in smoke_selection):
        raise RuntimeError("target smoke selection changed raw acceptance on replay")
    metadata_index = load_dilemmadata_target_metadata_index(
        discovery.root,
        tuple(outcome.record for outcome in smoke_selection),
    )
    target_smoke_rows = []
    for outcome in smoke_selection:
        sidecar = build_dilemmadata_target_sidecar(
            outcome,
            metadata_index=metadata_index,
        )
        if not hasattr(sidecar, "target_bundle"):
            target_smoke_rows.append(
                {
                    "accepted": False,
                    "record_id": outcome.record.record_id,
                    "categories": list(sidecar.categories),
                }
            )
            continue
        target_smoke_rows.append(
            {
                "accepted": True,
                "available_entry_count": sidecar.statistics.available_entry_count,
                "dialect": outcome.record.dialect,
                "masked_entry_count": sidecar.statistics.masked_entry_count,
                "record_id": outcome.record.record_id,
                "sidecar_fingerprint": sidecar.sidecar_fingerprint,
                "target_family_count": len(sidecar.target_bundle.targets),
            }
        )

    phase9eb1_tree_after = _tree_snapshot(phase9eb1_common)
    accepted_count = outcomes["accepted"]
    quarantine_count = outcomes["quarantined"]
    gates = {
        "accepted_at_least_900": accepted_count >= 900,
        "all_raw_graphs_valid": not graph_failures,
        "all_repairs_have_valid_evidence": not repair_evidence_failures,
        "all_two_pass_outputs_identical": not deterministic_failures,
        "analysisgnn_overlap_policy_count_is_14": len(overlap_ids) == 14,
        "no_negative_duration": not negative_duration_failures,
        "no_raw_target_or_annotation_leakage": not target_leakage_failures,
        "no_tie_or_next_in_track_cycles": not tie_cycle_failures,
        "notes_have_exact_bar_and_beat_context": not context_failures,
        "phase9eb1_719_exact": (
            len(old_by_record) == 719 and not preservation_failures
        ),
        "phase9eb1_outputs_unchanged_during_audit": (
            phase9eb1_tree_before == phase9eb1_tree_after
            and old_manifest_sha256 == _hash_file(old_manifest_path)
        ),
        "target_sidecar_smoke_passed": bool(target_smoke_rows)
        and all(row["accepted"] for row in target_smoke_rows),
    }
    summary: dict[str, object] = {
        "accepted_raw_ceiling": 1633,
        "analysisgnn_dataset_selection": {
            "excluded_overlap_count": len(overlap_ids),
            "excluded_record_ids": sorted(overlap_ids),
            "paper_candidate_ceiling": accepted_count - len(overlap_ids),
            "raw_adapter_record_count": accepted_count,
        },
        "contracts": {
            "release_commit": DILEMMADATA_RELEASE_COMMIT,
            "report_version": DILEMMADATA_COVERAGE_REMEDIATION_REPORT_VERSION,
        },
        "dialect_outcomes": {
            f"{dialect}:{status}": count
            for (dialect, status), count in sorted(dialect_outcomes.items())
        },
        "gates": gates,
        "outcomes": {
            "accepted": accepted_count,
            "discovered": len(discovery.records),
            "quarantined": quarantine_count,
        },
        "phase9eb1_preservation": {
            "common_manifest_sha256": old_manifest_sha256,
            "common_manifest_fingerprint": old_manifest["manifest_fingerprint"],
            "failure_count": len(preservation_failures),
            "previous_record_count": len(old_by_record),
            "raw_index_fingerprint": old_manifest["raw_index_fingerprint"],
            "source_split_fingerprint": old_manifest["source_split_fingerprint"],
            "split_counts": old_manifest["split_counts"],
            "tree_snapshot": phase9eb1_tree_after,
        },
        "previous_audit_sha256": previous_hashes,
        "previous_reason_counts_after_remediation": dict(
            sorted(previous_reason_counts.items())
        ),
        "previous_reason_repair_record_counts": {
            f"{reason}|{repair_type}": count
            for (reason, repair_type), count in sorted(
                previous_reason_repair_counts.items()
            )
        },
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
        "ready": all(gates.values()),
        "repair_event_counts": dict(sorted(repair_event_counts.items())),
        "repair_record_counts": {
            repair_type: len(record_ids)
            for repair_type, record_ids in sorted(repair_record_sets.items())
        },
        "task_availability_after_remediation": task_availability_after,
        "target_sidecar_smoke": {
            "record_count": len(target_smoke_rows),
            "records": target_smoke_rows,
            "writes_existing_sidecars": False,
        },
    }
    summary["semantic_fingerprint"] = _fingerprint(summary)
    _write_jsonl(output_dir / "record_results.jsonl", record_rows)
    _write_jsonl(output_dir / "repair_evidence.jsonl", repair_rows)
    _write_jsonl(output_dir / "quarantine_records.jsonl", quarantine_rows)
    _write_json(output_dir / "target_sidecar_smoke.json", summary["target_sidecar_smoke"])
    _write_json(output_dir / "audit_summary.json", summary)
    report_lines = [
        "# Phase 9E-B2 Dilemmadata coverage remediation audit",
        "",
        f"- Ready: `{summary['ready']}`",
        f"- Discovered: `{len(discovery.records)}`",
        f"- Accepted raw: `{accepted_count}`",
        f"- Quarantined: `{quarantine_count}`",
        f"- Previous 719 exact preservation failures: `{len(preservation_failures)}`",
        f"- AnalysisGNN overlap policy exclusions: `{len(overlap_ids)}`",
        f"- Paper-selection ceiling: `{accepted_count - len(overlap_ids)}`",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{value}`" for name, value in sorted(gates.items())],
        "",
        "## Repair record counts",
        "",
        *[
            f"- `{name}`: `{len(record_ids)}` records / "
            f"`{repair_event_counts[name]}` events"
            for name, record_ids in sorted(repair_record_sets.items())
        ],
        "",
        "No training, validation, locked-TEST evaluation, model inference, split "
        "replacement, or sidecar replacement was performed.",
    ]
    (output_dir / "AUDIT_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts = {
        path.name: _hash_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    _write_json(output_dir / "artifact_sha256.json", artifacts)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, default=DEFAULT_PREVIOUS_AUDIT)
    parser.add_argument(
        "--phase9eb1-common",
        type=Path,
        default=DEFAULT_PHASE9EB1_COMMON,
    )
    arguments = parser.parse_args()
    root = arguments.root
    if root is None:
        configured = os.environ.get("MUSIC_CRITIC_DILEMMADATA_ROOT")
        if not configured:
            parser.error("--root or MUSIC_CRITIC_DILEMMADATA_ROOT is required")
        root = Path(configured)
    try:
        summary = build_remediation_audit(
            root.resolve(),
            output_dir=arguments.output_dir.resolve(),
            previous_audit=arguments.previous_audit.resolve(),
            phase9eb1_common=arguments.phase9eb1_common.resolve(),
        )
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json({"output_dir": str(arguments.output_dir), "ready": summary["ready"]}))
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
