#!/usr/bin/env python3
"""Audit the Phase 9E-B5G directed inverse remediation without TEST access."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Mapping

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    ProductionArtifactPaths,
    frozen_split_assignments,
    load_production_record,
)
from music_critic.experiments.analysisgnn.directed_transposition_diagnostics import (
    B5G_AUDIT_SCHEMA,
    EXPECTED_PAIR_COUNT,
    compact_directed_fixture,
    check_directed_fixture,
    directed_record_shift_row,
    historical_schedule_evidence,
)
from music_critic.experiments.analysisgnn.transposition import SHIFT_PCS
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    prepare_sidecar_diagnostic_context,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5g_directed_inverse.json"
DEFAULT_OUTPUT = ROOT / "outputs/phase9eb5g/directed-inverse"
DEFAULT_B5F_PAIRS = (
    ROOT
    / "outputs/phase9eb5f/analysisgnn-transposition-correctness/record_shift_diagnostics.jsonl"
)
_WORKER_PATHS: ProductionArtifactPaths | None = None
_WORKER_B5A: dict[tuple[str, int], dict[str, object]] = {}
_WORKER_PRIOR: dict[tuple[str, int], dict[str, object]] = {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _b5a_rows(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(str(row["record_id"]), int(row["shift_pc"])): row for row in rows}


def _initialize_worker(b5f_pairs: str | None) -> None:
    global _WORKER_PATHS, _WORKER_B5A, _WORKER_PRIOR
    import torch

    torch.set_num_threads(1)
    _WORKER_PATHS = ProductionArtifactPaths()
    _WORKER_B5A = _b5a_rows(_WORKER_PATHS.b5a_shift_eligibility)
    _WORKER_PRIOR = {}
    if b5f_pairs is not None and Path(b5f_pairs).is_file():
        with Path(b5f_pairs).open("r", encoding="utf-8") as handle:
            _WORKER_PRIOR = {
                (str(row["record_id"]), int(row["shift_pc"])): row
                for line in handle
                if line.strip()
                for row in (json.loads(line),)
            }


def _audit_record_worker(
    item: tuple[str, str],
) -> tuple[str, str, list[dict[str, object]]]:
    if _WORKER_PATHS is None:
        raise RuntimeError("B5G audit worker is not initialized")
    record_id, split = item
    batch, sidecar = load_production_record(
        record_id, split=split, paths=_WORKER_PATHS
    )
    graph = batch.raw_graph_batch.to_data_list()[0]
    prepared = (
        {} if _WORKER_PRIOR else prepare_sidecar_diagnostic_context(sidecar)
    )
    rows = [
        directed_record_shift_row(
            graph=graph,
            sidecar=sidecar,
            split=split,
            shift_pc=shift_pc,
            b5a=(
                _WORKER_B5A.get((record_id, shift_pc))
                if split == "train"
                else None
            ),
            prepared_sidecar=prepared,
            prior_b5f=_WORKER_PRIOR.get((record_id, shift_pc)),
        )
        for shift_pc in SHIFT_PCS
    ]
    return record_id, split, rows


def build_corpus_audit(
    *,
    output: Path,
    fixture: Path,
    b5f_pairs: Path | None = DEFAULT_B5F_PAIRS,
    workers: int = 1,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = ProductionArtifactPaths()
    assignments = frozen_split_assignments(paths)
    records = [
        (record_id, str(row["split"]))
        for record_id, row in sorted(assignments.items())
        if row["split"] in {"train", "validation"}
    ]
    prior_available = b5f_pairs is not None and b5f_pairs.is_file()
    pair_path = output / "record_shift_diagnostics.jsonl"
    per_shift = {shift: Counter() for shift in SHIFT_PCS}
    split_counts: Counter[str] = Counter()
    verified_pairs: dict[tuple[str, int], bool] = {}
    total_failures = Counter()
    pair_count = 0
    if workers <= 0:
        raise ValueError("workers must be positive")
    b5f_locator = str(b5f_pairs) if b5f_pairs is not None else None
    _initialize_worker(b5f_locator)
    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(b5f_locator,),
        )
        if workers > 1
        else None
    )
    audited = (
        map(_audit_record_worker, records)
        if executor is None
        else executor.map(_audit_record_worker, records, chunksize=1)
    )
    for index, (record_id, split, rows) in enumerate(audited, 1):
        split_counts[split] += 1
        for shift_pc, row in zip(SHIFT_PCS, rows, strict=True):
            _append_jsonl(pair_path, row)
            pair_count += 1
            eligible = bool(row["eligible"])
            verified_pairs[(record_id, shift_pc)] = bool(
                row["forward_success_status_equal"]
                and row["canonical_forward_identical"]
            )
            stats = per_shift[shift_pc]
            stats["raw_views"] += 1
            stats["eligible"] += int(eligible)
            stats["round_trip_failures"] += int(
                eligible and not bool(row["raw_round_trip_valid"])
            )
            stats["target_round_trip_failures"] += int(
                eligible and not bool(row["target_round_trip_valid"])
            )
            stats["canonical_forward_mismatches"] += int(
                not bool(row["forward_success_status_equal"])
                or not bool(row["canonical_forward_identical"])
            )
            stats["executable_cross_head_failures"] += int(
                row["cross_head_status"] == "failed"
            )
            stats["cross_head_not_checkable"] += int(
                row["cross_head_status"] == "not_checkable"
            )
        if index % 20 == 0 or index == len(records):
            print(json.dumps({
                "event": "corpus_progress",
                "record_index": index,
                "record_count": len(records),
                "record_id": record_id,
            }, sort_keys=True, separators=(",", ":")), flush=True)
    if executor is not None:
        executor.shutdown()
    for stats in per_shift.values():
        total_failures["round_trip"] += stats["round_trip_failures"]
        total_failures["target_round_trip"] += stats["target_round_trip_failures"]
        total_failures["forward"] += stats["canonical_forward_mismatches"]
        total_failures["cross_head"] += stats["executable_cross_head_failures"]
    history = historical_schedule_evidence(verified_pairs)
    valid = (
        pair_count == EXPECTED_PAIR_COUNT
        and dict(split_counts) == {"train": 1295, "validation": 162}
        and per_shift[6]["eligible"] == 1439
        and not any(total_failures.values())
        and history["all_draws_bound_to_verified_identical_forward_pair"] is True
    )
    body: dict[str, object] = {
        "schema": B5G_AUDIT_SCHEMA,
        "phase": "9E-B5G",
        "valid": valid,
        "full_corpus_pair_audit": True,
        "record_counts": dict(sorted(split_counts.items())),
        "record_shift_pair_count": pair_count,
        "per_shift": {str(k): dict(sorted(v.items())) for k, v in per_shift.items()},
        "shift6_eligible": per_shift[6]["eligible"],
        "round_trip_failure_count": total_failures["round_trip"],
        "target_round_trip_failure_count": total_failures["target_round_trip"],
        "canonical_forward_mismatch_count": total_failures["forward"],
        "executable_cross_head_failure_count": total_failures["cross_head"],
        "historical_schedule": history,
        "historical_b5f_pair_evidence_reused": prior_available,
        "inverse_contract_valid": valid,
        "b5d_c1_forward_training_changed": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    _write_json(output / "audit_summary.json", body)
    compact_corpus = {
        key: body[key]
        for key in (
            "full_corpus_pair_audit",
            "record_counts",
            "record_shift_pair_count",
            "per_shift",
            "shift6_eligible",
            "round_trip_failure_count",
            "target_round_trip_failure_count",
            "canonical_forward_mismatch_count",
            "executable_cross_head_failure_count",
            "fingerprint",
        )
    }
    _write_json(fixture, compact_directed_fixture(corpus=compact_corpus))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--build-source-free", action="store_true")
    mode.add_argument("--build-corpus", action="store_true")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.check:
        result = check_directed_fixture(args.fixture)
    elif args.build_source_free:
        result = compact_directed_fixture()
        _write_json(args.fixture, result)
    else:
        result = build_corpus_audit(
            output=args.output_dir, fixture=args.fixture, workers=args.workers
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
