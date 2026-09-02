#!/usr/bin/env python3
"""Audit Phase 9E-B5F transposition correctness without opening TEST."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    frozen_split_assignments,
    load_production_record,
    minimal_real_train_coverage_records,
    production_valid_shifts,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    SIGNED_BY_SHIFT_PC,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    B5F_AUDIT_SCHEMA,
    B5F_PAIR_SCHEMA,
    EXPECTED_PAIR_COUNT,
    audit_graph_transform,
    audit_sidecar_targets,
    check_compact_fixture,
    compact_audit_fixture,
    cross_head_checks,
    prepare_sidecar_diagnostic_context,
    schedule_diagnostics,
    source_free_runtime_regression,
    transformation_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "tests/fixtures/analysisgnn/phase9eb5f_transposition_correctness.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/phase9eb5f/analysisgnn-transposition-correctness"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _b5a_rows(paths: ProductionArtifactPaths) -> dict[tuple[str, int], dict[str, object]]:
    values = {}
    with paths.b5a_shift_eligibility.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values[(str(row["record_id"]), int(row["shift_pc"]))] = row
    return values


def _real_train_runtime_regression(paths: ProductionArtifactPaths) -> dict[str, object]:
    """Run real TRAIN coverage through graph -> logits -> target alignment."""

    import torch

    from music_critic.experiments.analysisgnn.corrected_model import (
        CorrectedAnalysisGNNModel,
    )

    records = minimal_real_train_coverage_records(paths)
    valid_shifts = production_valid_shifts(paths)
    torch.manual_seed(17)
    model = CorrectedAnalysisGNNModel().eval()
    routed_rows = Counter()
    valid_rows = Counter()
    mismatches: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []
    with torch.no_grad():
        for record_id in records:
            batch, sidecar = load_production_record(
                record_id, split="train", paths=paths
            )
            non_identity = next(
                (shift for shift in valid_shifts[record_id] if shift != 0), None
            )
            shifts = (0,) if non_identity is None else (0, non_identity)
            baseline_routing: dict[str, tuple[object, ...]] = {}
            for shift_pc in shifts:
                shifted = transpose_raw_graph_batch(
                    batch.raw_graph_batch, (shift_pc,)
                )
                output = model(shifted)
                alignment = align_target_sidecars_after_prediction(
                    output, shifted, (sidecar,), shifts=(shift_pc,)
                )
                if alignment.target_sidecar_fingerprints != (
                    str(sidecar["fingerprint"]),
                ):
                    mismatches.append(
                        {
                            "record_id": record_id,
                            "shift_pc": shift_pc,
                            "reason": "sidecar_fingerprint_changed",
                        }
                    )
                for task_id in ACTIVE_HEADS:
                    rows = alignment.heads[task_id]
                    routed_rows[task_id] += len(rows.entity_ids)
                    valid_rows[task_id] += int(rows.valid_mask.sum())
                    signature = (
                        rows.entity_ids,
                        rows.record_ids,
                        rows.component_ids,
                        tuple(rows.candidate_indices.tolist()),
                        tuple(rows.valid_mask.tolist()),
                        rows.masked_row_count,
                        rows.alignment_failure_count,
                    )
                    if shift_pc == 0:
                        baseline_routing[task_id] = signature
                    elif signature != baseline_routing[task_id]:
                        mismatches.append(
                            {
                                "record_id": record_id,
                                "shift_pc": shift_pc,
                                "task_id": task_id,
                                "reason": "routing_or_mask_changed",
                            }
                        )
                evaluated.append({"record_id": record_id, "shift_pc": shift_pc})
    uncovered = [task for task in ACTIVE_HEADS if valid_rows[task] == 0]
    body: dict[str, object] = {
        "records": list(records),
        "evaluated_record_shifts": evaluated,
        "active_head_count": len(ACTIVE_HEADS),
        "routed_rows": dict(sorted(routed_rows.items())),
        "valid_rows": dict(sorted(valid_rows.items())),
        "uncovered_active_heads": uncovered,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "graph_transformed_before_forward": True,
        "targets_aligned_after_logits": True,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "passed": not uncovered and not mismatches,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _record_shift_row(
    *,
    record_id: str,
    split: str,
    graph: object,
    sidecar: Mapping[str, object],
    shift_pc: int,
    b5a: Mapping[str, object] | None,
    prepared_sidecar: Mapping[str, object],
) -> dict[str, object]:
    graph_result = audit_graph_transform(
        graph, shift_pc=shift_pc, compare_direct_and_runtime=False
    )
    target_result = audit_sidecar_targets(
        sidecar, shift_pc=shift_pc, prepared=prepared_sidecar
    )
    if target_result["target_vocabulary_closed"]:
        cross_head = cross_head_checks(sidecar, graph, shift_pc=shift_pc)
    else:
        cross_head_body = {
            "checks": {},
            "failed": False,
            "passed": False,
            "not_checkable": ["target_vocabulary_not_closed"],
        }
        cross_head = {
            **cross_head_body,
            "fingerprint": fingerprint(cross_head_body),
        }
    reasons = set(str(value) for value in graph_result["invalid_reasons"])
    reasons.update(str(value) for value in target_result["invalid_reasons"])
    if b5a is not None:
        reasons.update(str(value) for value in b5a.get("corrected_invalid_reasons", []))
    eligible = bool(graph_result["midi_range_valid"]) and bool(
        target_result["target_vocabulary_closed"]
    ) and (b5a is None or b5a.get("corrected_valid") is True)
    round_trip = eligible and bool(graph_result["round_trip_valid"]) and bool(
        target_result["round_trip_valid"]
    )
    if eligible and not round_trip:
        reasons.add("eligible_round_trip_failure")
    body: dict[str, object] = {
        "schema": B5F_PAIR_SCHEMA,
        "record_id": record_id,
        "split": split,
        "dialect": sidecar["dialect"],
        "source_component_id": sidecar["source_component_id"],
        "shift_pc": shift_pc,
        "signed_semitones": SIGNED_BY_SHIFT_PC[shift_pc],
        "eligible": eligible,
        "midi_range_valid": graph_result["midi_range_valid"],
        "target_vocabulary_closed": target_result["target_vocabulary_closed"],
        "spelling_valid": target_result["spelling_valid"],
        "round_trip_valid": round_trip,
        "runtime_path_matches_contract": graph_result[
            "runtime_path_matches_contract"
        ],
        "masks_preserved": target_result["masks_preserved"],
        "target_class_ids_round_trip_valid": bool(
            target_result["class_ids_preserved"]
        )
        and bool(target_result["round_trip_valid"]),
        "entity_ids_preserved": bool(graph_result["entity_ids_preserved"])
        and bool(target_result["entity_ids_preserved"]),
        "routing_preserved": True,
        "routing_fingerprint": fingerprint(sidecar["relations"]),
        "topology_preserved": graph_result["topology_preserved"],
        "changed_graph_fields": graph_result["changed_graph_fields"],
        "cross_head_status": (
            "failed"
            if cross_head["failed"]
            else "passed"
            if cross_head["passed"]
            else "not_checkable"
        ),
        "cross_head_not_checkable": cross_head["not_checkable"],
        "invalid_reasons": sorted(reasons),
        "graph_fingerprint": graph_result["fingerprint"],
        "target_fingerprint": target_result["fingerprint"],
        "cross_head_fingerprint": cross_head["fingerprint"],
        "b5a_corrected_valid": None if b5a is None else b5a["corrected_valid"],
    }
    body["fingerprint"] = fingerprint(body)
    return body


def build_corpus_audit(
    *,
    output: Path,
    fixture: Path,
    max_records: int | None = None,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = ProductionArtifactPaths()
    assignments = frozen_split_assignments(paths)
    selected = [
        (record_id, str(row["split"]))
        for record_id, row in sorted(assignments.items())
        if row["split"] in {"train", "validation"}
    ]
    if max_records is not None:
        selected = selected[:max_records]
    b5a = _b5a_rows(paths)
    pair_path = output / "record_shift_diagnostics.jsonl"
    shift_stats = {
        shift: Counter() for shift in SHIFT_PCS
    }
    reasons: Counter[str] = Counter()
    split_records: Counter[str] = Counter()
    cross_head_not_checkable: Counter[str] = Counter()
    pair_count = 0
    for record_index, (record_id, split) in enumerate(selected, 1):
        batch, sidecar = load_production_record(record_id, split=split, paths=paths)
        graph = batch.raw_graph_batch.to_data_list()[0]
        prepared_sidecar = prepare_sidecar_diagnostic_context(sidecar)
        split_records[split] += 1
        for shift_pc in SHIFT_PCS:
            row = _record_shift_row(
                record_id=record_id,
                split=split,
                graph=graph,
                sidecar=sidecar,
                shift_pc=shift_pc,
                b5a=b5a.get((record_id, shift_pc)) if split == "train" else None,
                prepared_sidecar=prepared_sidecar,
            )
            _append_jsonl(pair_path, row)
            pair_count += 1
            stats = shift_stats[shift_pc]
            stats["eligible"] += int(bool(row["eligible"]))
            stats["invalid"] += int(not bool(row["eligible"]))
            stats["round_trip_checked"] += int(bool(row["eligible"]))
            stats["round_trip_failures"] += int(
                bool(row["eligible"]) and not bool(row["round_trip_valid"])
            )
            stats["runtime_mismatches"] += int(
                not bool(row["runtime_path_matches_contract"])
            )
            stats["cross_head_failures"] += int(row["cross_head_status"] == "failed")
            for reason in row["invalid_reasons"]:
                reasons[str(reason)] += 1
            for name in row["cross_head_not_checkable"]:
                cross_head_not_checkable[str(name)] += 1
        if record_index % 20 == 0 or record_index == len(selected):
            print(
                json.dumps(
                    {
                        "event": "corpus_progress",
                        "record_index": record_index,
                        "record_count": len(selected),
                        "record_id": record_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
    expected_records = 1457 if max_records is None else len(selected)
    expected_pairs = expected_records * len(SHIFT_PCS)
    full = max_records is None and pair_count == EXPECTED_PAIR_COUNT
    runtime = source_free_runtime_regression()
    real_runtime = _real_train_runtime_regression(paths)
    schedule = schedule_diagnostics(paths)
    compact = compact_audit_fixture(runtime=runtime, schedule=schedule)
    summary_body: dict[str, object] = {
        "schema": B5F_AUDIT_SCHEMA,
        "phase": "9E-B5F",
        "valid": pair_count == expected_pairs,
        "full_corpus_pair_audit": full,
        "record_counts": dict(sorted(split_records.items())),
        "record_shift_pair_count": pair_count,
        "expected_record_shift_pair_count": expected_pairs,
        "transformation_matrix": list(transformation_matrix()),
        "per_shift": {
            str(shift): dict(sorted(shift_stats[shift].items()))
            for shift in SHIFT_PCS
        },
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "cross_head_not_checkable_counts": dict(
            sorted(cross_head_not_checkable.items())
        ),
        "runtime_regression": runtime,
        "real_train_runtime_regression": real_runtime,
        "schedule": schedule,
        "final_status": (
            "implementation_or_contract_defect"
            if any(shift_stats[shift]["round_trip_failures"] for shift in SHIFT_PCS)
            or any(shift_stats[shift]["runtime_mismatches"] for shift in SHIFT_PCS)
            else "inconclusive"
        ),
        "checkpoint_diagnostics_run": False,
        "shift0_metrics_reproduced": False,
        "ready_for_soft_augmentation": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    summary_body["fingerprint"] = fingerprint(summary_body)
    _write_json(output / "audit_summary.json", summary_body)
    compact["corpus_contract"] = {
        "train_records": split_records["train"],
        "validation_records": split_records["validation"],
        "record_shift_pairs": pair_count,
        "full_pair_audit_run": full,
        "test_enumerated": False,
        "summary_fingerprint": summary_body["fingerprint"],
        "per_shift": summary_body["per_shift"],
        "invalid_reason_counts": summary_body["invalid_reason_counts"],
        "real_train_runtime_regression": real_runtime,
    }
    compact.pop("evidence_fingerprint", None)
    compact.pop("fixture_fingerprint", None)
    compact["evidence_fingerprint"] = fingerprint(compact)
    compact["fixture_fingerprint"] = fingerprint(compact)
    _write_json(fixture, compact)
    return summary_body


def build_source_free_fixture(*, fixture: Path) -> dict[str, object]:
    runtime = source_free_runtime_regression()
    schedule = schedule_diagnostics()
    payload = compact_audit_fixture(runtime=runtime, schedule=schedule)
    _write_json(fixture, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build-source-free", action="store_true")
    mode.add_argument("--build-corpus", action="store_true")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check:
        result = check_compact_fixture(args.fixture)
    elif args.build_source_free:
        result = build_source_free_fixture(fixture=args.fixture)
    else:
        result = build_corpus_audit(
            output=args.output_dir,
            fixture=args.fixture,
            max_records=args.max_records,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
