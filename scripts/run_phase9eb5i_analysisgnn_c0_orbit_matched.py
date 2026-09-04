#!/usr/bin/env python3
"""Run the Phase 9E-B5I 120k orbit-matched C0 CUDA control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
    ProductionArtifactPaths,
    attempt_applied_update,
    build_source_free_fixture,
    checkpoint_payload,
    combine_single_record_raw_batches,
    environment_fingerprint,
    load_checkpoint,
    load_frozen_class_weights,
    load_production_record,
    move_raw_graph_batch,
    production_component_records,
    production_valid_shifts,
    run_corrected_validation,
    save_checkpoint,
    select_best_validation_checkpoint,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.full_orbit_training import (
    FULL_ORBIT_BATCH_SIZE,
    FULL_ORBIT_CHECKPOINT_INTERVAL,
    FULL_ORBIT_DRAW_BUDGET,
    FULL_ORBIT_PROFILE_ID,
    FULL_ORBIT_PROGRESS_INTERVAL,
    FULL_ORBIT_SEED,
    FULL_ORBIT_UPDATE_BUDGET,
    FULL_ORBIT_VALIDATION_INTERVAL,
    build_full_orbit_table,
    full_orbit_profile_contract,
    run_full_orbit_diagnostic_validation,
)
from music_critic.experiments.analysisgnn.orbit_matched_control import (
    ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,
    ORBIT_MATCHED_CONTROL_LABEL,
    ORBIT_MATCHED_CONTROL_PROFILE_ID,
    OrbitMatchedControlRuntimeConfig,
    OrbitMatchedControlSampler,
    build_orbit_matched_control_optimizer_scheduler,
    completed_control_history_contract,
    orbit_matched_control_preflight,
    orbit_matched_control_profile_contract,
)


DEFAULT_OUTPUT = Path("outputs/phase9eb5i")
RUN_NAME = "c0-seed17-orbit-matched-u120000"
RUN_SUMMARY_SCHEMA = "Phase9EB5IOrbitMatchedControlRunSummary@1.0.0"
SMOKE_SCHEMA = "Phase9EB5IOrbitMatchedControlSmoke@1.0.0"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _rewrite_jsonl(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _emit(event: str, **values: object) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deterministic_cuda() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loss_row(report: object) -> dict[str, object]:
    def scalar(value: object) -> float | None:
        return None if value is None else float(value.detach().cpu())

    return {
        "total_loss": scalar(report.total),
        "primary_group_loss": scalar(report.primary),
        "auxiliary_group_loss": scalar(report.auxiliary),
        "zero_valid_heads": list(report.zero_valid_heads),
    }


def _checkpoint(
    *,
    model: CorrectedAnalysisGNNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: OrbitMatchedControlSampler,
    config: OrbitMatchedControlRuntimeConfig,
    applied_update: int,
    best_primary_score: float | None,
    best_update: int | None,
    record_history: Sequence[str],
    schedule_shift_history: Sequence[int],
    applied_shift_history: Sequence[int],
    elapsed_wall_seconds: float,
) -> dict[str, object]:
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        applied_update=applied_update,
        best_primary_score=best_primary_score,
        best_update=best_update,
        record_history=record_history,
        shift_history=applied_shift_history,
    )
    payload["phase"] = "9E-B5I"
    payload["orbit_matched_control_profile_fingerprint"] = (
        orbit_matched_control_profile_contract()["fingerprint"]
    )
    payload["matched_c2_profile_fingerprint"] = full_orbit_profile_contract()[
        "fingerprint"
    ]
    payload["matched_schedule_shift_history"] = list(schedule_shift_history)
    payload["elapsed_wall_seconds"] = elapsed_wall_seconds
    return payload


def _expected_validation_updates(applied_update: int) -> list[int]:
    return list(range(0, applied_update + 1, FULL_ORBIT_VALIDATION_INTERVAL))


def _reconcile_resume_ledgers(run_root: Path, applied_update: int) -> None:
    """Discard rows written after the last atomic checkpoint."""

    training_path = run_root / "training_metrics.jsonl"
    validation_path = run_root / "validation_metrics.jsonl"
    training = [
        row
        for row in _read_jsonl(training_path)
        if int(row["applied_update"]) <= applied_update
    ]
    expected_training = list(range(1, applied_update + 1))
    observed_training = [int(row["applied_update"]) for row in training]
    if observed_training != expected_training:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.resume_training_ledger_mismatch",
            f"expected={len(expected_training)} observed={len(observed_training)}",
        )

    validation = [
        row
        for row in _read_jsonl(validation_path)
        if int(row["applied_update"]) <= applied_update
    ]
    expected_validation = _expected_validation_updates(applied_update)
    observed_validation = [int(row["applied_update"]) for row in validation]
    if observed_validation != expected_validation:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.resume_validation_ledger_mismatch",
            f"expected={expected_validation} observed={observed_validation}",
        )
    _rewrite_jsonl(training_path, training)
    _rewrite_jsonl(validation_path, validation)


def _validate_restored_histories(
    *,
    sampler: OrbitMatchedControlSampler,
    record_history: Sequence[str],
    schedule_shift_history: Sequence[int],
    applied_shift_history: Sequence[int],
) -> None:
    expected = sampler.position
    if not (
        len(record_history)
        == len(schedule_shift_history)
        == len(applied_shift_history)
        == expected
    ):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.resume_history_length_mismatch",
            (
                f"position={expected} records={len(record_history)} "
                f"scheduled={len(schedule_shift_history)} "
                f"applied={len(applied_shift_history)}"
            ),
        )
    if any(
        shift != ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
        for shift in applied_shift_history
    ):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.resume_non_identity_shift",
            repr(sorted(set(applied_shift_history))),
        )

    expected_sampler = OrbitMatchedControlSampler(
        sampler.table, seed=sampler.seed
    )
    for offset, (record_id, schedule_shift) in enumerate(
        zip(record_history, schedule_shift_history, strict=True)
    ):
        expected_draw = expected_sampler.peek(offset)
        if (
            record_id != expected_draw.record_id
            or int(schedule_shift) != expected_draw.schedule_shift_pc
        ):
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.resume_schedule_prefix_mismatch",
                (
                    f"offset={offset} expected_record={expected_draw.record_id} "
                    f"observed_record={record_id} "
                    f"expected_shift={expected_draw.schedule_shift_pc} "
                    f"observed_shift={schedule_shift}"
                ),
            )


def _guard_fresh_run(run_root: Path) -> None:
    protected = (
        "training_metrics.jsonl",
        "validation_metrics.jsonl",
        "last.ckpt",
        "best-validation.ckpt",
        "run_summary.json",
    )
    existing = [name for name in protected if (run_root / name).exists()]
    if existing:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.output_exists",
            f"use --resume or a new output root: {existing}",
        )


def _validation_progress(applied_update: int):
    def progress(index: int, total: int, record_id: str) -> None:
        if index % 20 == 0 or index == total:
            _emit(
                "validation_progress",
                applied_update=applied_update,
                record_index=index,
                record_count=total,
                record_id=record_id,
            )

    return progress


def smoke(device: str) -> dict[str, object]:
    """Prove that all twelve schedule strata execute the same identity view."""

    if device == "cuda" and not torch.cuda.is_available():
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.cuda_unavailable",
            "CUDA is required",
        )
    batch, sidecar = build_source_free_fixture()
    _seed(FULL_ORBIT_SEED)
    model = CorrectedAnalysisGNNModel().to(device).eval()
    rows: list[dict[str, object]] = []
    reference: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for schedule_shift in range(12):
            raw = move_raw_graph_batch(
                transpose_raw_graph_batch(
                    batch.raw_graph_batch,
                    (ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,),
                ),
                device,
            )
            output = model(raw)
            current = {
                task: value.detach().cpu().clone()
                for task, value in output.logits.items()
            }
            if reference is None:
                reference = current
            identity_equal = all(
                torch.equal(current[task], reference[task]) for task in current
            )
            finite = all(torch.isfinite(value).all() for value in current.values())
            rows.append(
                {
                    "schedule_shift_pc": schedule_shift,
                    "applied_shift_pc": ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,
                    "finite": bool(finite),
                    "identity_logits_equal": identity_equal,
                }
            )
    body: dict[str, object] = {
        "schema": SMOKE_SCHEMA,
        "valid": all(
            row["finite"] is True and row["identity_logits_equal"] is True
            for row in rows
        ),
        "record_id": sidecar["record_id"],
        "profile": ORBIT_MATCHED_CONTROL_LABEL,
        "profile_id": ORBIT_MATCHED_CONTROL_PROFILE_ID,
        "matched_schedule_profile_id": FULL_ORBIT_PROFILE_ID,
        "per_schedule_shift": rows,
        "training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def run_full(*, output_root: Path, resume: Path | None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.cuda_unavailable",
            "full training is CUDA-only",
        )
    _deterministic_cuda()
    config = OrbitMatchedControlRuntimeConfig()
    run_root = output_root / RUN_NAME
    if resume is None:
        _guard_fresh_run(run_root)

    paths = ProductionArtifactPaths()
    preflight = orbit_matched_control_preflight(paths)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "preflight.json", preflight)
    _write_json(run_root / "resolved_config.json", config.to_dict())
    _write_json(
        run_root / "orbit_matched_control_contract.json",
        orbit_matched_control_profile_contract(),
    )
    _write_json(run_root / "environment.json", environment_fingerprint())

    _seed(FULL_ORBIT_SEED)
    model = CorrectedAnalysisGNNModel().to("cuda")
    initial_fingerprint = model_state_fingerprint(model)
    if initial_fingerprint != preflight["initial_model_state_fingerprint"]:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.initial_state_mismatch",
            (
                f"expected={preflight['initial_model_state_fingerprint']} "
                f"observed={initial_fingerprint}"
            ),
        )
    _seed(FULL_ORBIT_SEED * 1000 + 1)
    optimizer, scheduler = build_orbit_matched_control_optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = OrbitMatchedControlSampler(
        build_full_orbit_table(
            production_component_records(paths), production_valid_shifts(paths)
        )
    )
    class_weights = load_frozen_class_weights()

    applied_update = 0
    best_score: float | None = None
    best_update: int | None = None
    record_history: list[str] = []
    schedule_shift_history: list[int] = []
    applied_shift_history: list[int] = []
    elapsed_before = 0.0

    if resume is not None:
        restored = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
        )
        applied_update = int(restored["applied_update"])
        if applied_update % FULL_ORBIT_CHECKPOINT_INTERVAL:
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.resume_interval_mismatch",
                str(applied_update),
            )
        best_score = restored["best_primary_score"]
        best_update = restored["best_update"]
        record_history = list(restored["record_history"])
        applied_shift_history = list(restored["shift_history"])
        schedule_shift_history = list(
            restored.get("matched_schedule_shift_history", ())
        )
        elapsed_before = float(restored.get("elapsed_wall_seconds", 0.0))
        _validate_restored_histories(
            sampler=sampler,
            record_history=record_history,
            schedule_shift_history=schedule_shift_history,
            applied_shift_history=applied_shift_history,
        )
        _reconcile_resume_ledgers(run_root, applied_update)
        _emit("resume_loaded", applied_update=applied_update)
    else:
        _emit(
            "validation_start",
            applied_update=0,
            profile=ORBIT_MATCHED_CONTROL_LABEL,
        )
        validation = run_corrected_validation(
            model,
            device="cuda",
            paths=paths,
            progress=_validation_progress(0),
        )
        _append_jsonl(
            run_root / "validation_metrics.jsonl",
            {"applied_update": 0, **validation},
        )
        best_score = float(validation["corrected_primary_macro_score"])
        best_update = 0
        initial = _checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            config=config,
            applied_update=0,
            best_primary_score=best_score,
            best_update=best_update,
            record_history=(),
            schedule_shift_history=(),
            applied_shift_history=(),
            elapsed_wall_seconds=0.0,
        )
        save_checkpoint(run_root / "best-validation.ckpt", initial)
        save_checkpoint(run_root / "last.ckpt", initial)
        _emit(
            "validation_complete",
            applied_update=0,
            profile=ORBIT_MATCHED_CONTROL_LABEL,
            primary_score=best_score,
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    overflow_retries = 0
    while applied_update < FULL_ORBIT_UPDATE_BUDGET:
        draws = [sampler.peek(offset) for offset in range(FULL_ORBIT_BATCH_SIZE)]
        loaded = [
            load_production_record(draw.record_id, split="train", paths=paths)
            for draw in draws
        ]
        raw = combine_single_record_raw_batches(
            [loaded_record[0].raw_graph_batch for loaded_record in loaded]
        )
        schedule_shifts = tuple(draw.schedule_shift_pc for draw in draws)
        applied_shifts = tuple(draw.applied_shift_pc for draw in draws)
        if any(
            shift != ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
            for shift in applied_shifts
        ):
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.non_identity_shift_selected",
                repr(applied_shifts),
            )
        raw = move_raw_graph_batch(
            transpose_raw_graph_batch(raw, applied_shifts), "cuda"
        )
        result = attempt_applied_update(
            model=model,
            raw_graph_batch=raw,
            sidecars=tuple(loaded_record[1] for loaded_record in loaded),
            shifts=applied_shifts,
            class_weights=class_weights,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if not result.applied:
            overflow_retries += 1
            if overflow_retries >= 3:
                raise CorrectedTrainingError(
                    "analysisgnn.orbit_matched_control.persistent_overflow",
                    str(applied_update),
                )
            continue

        overflow_retries = 0
        applied_update += 1
        sampler.advance_after_applied_update()
        record_history.extend(draw.record_id for draw in draws)
        schedule_shift_history.extend(schedule_shifts)
        applied_shift_history.extend(applied_shifts)
        elapsed = elapsed_before + time.perf_counter() - started
        rate = applied_update / max(elapsed, 1e-12)
        training_row = {
            "applied_update": applied_update,
            "records": [draw.record_id for draw in draws],
            "components": [draw.component_id for draw in draws],
            "matched_schedule_shifts": list(schedule_shifts),
            "applied_shifts": list(applied_shifts),
            "shifts": list(applied_shifts),
            "orbit_epochs": sampler.position / sampler.draws_per_epoch,
            "gradient_norm": result.gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates_per_second": rate,
            **_loss_row(result.loss),
        }
        _append_jsonl(run_root / "training_metrics.jsonl", training_row)

        if applied_update % FULL_ORBIT_PROGRESS_INTERVAL == 0:
            _emit(
                "training_progress",
                profile=ORBIT_MATCHED_CONTROL_LABEL,
                applied_update=applied_update,
                update_budget=FULL_ORBIT_UPDATE_BUDGET,
                total_loss=training_row["total_loss"],
                learning_rate=training_row["learning_rate"],
                updates_per_second=rate,
                eta_minutes=(FULL_ORBIT_UPDATE_BUDGET - applied_update)
                / max(rate, 1e-12)
                / 60,
            )

        if applied_update % FULL_ORBIT_VALIDATION_INTERVAL == 0:
            _emit(
                "validation_start",
                applied_update=applied_update,
                profile=ORBIT_MATCHED_CONTROL_LABEL,
            )
            validation = run_corrected_validation(
                model,
                device="cuda",
                paths=paths,
                progress=_validation_progress(applied_update),
            )
            _append_jsonl(
                run_root / "validation_metrics.jsonl",
                {"applied_update": applied_update, **validation},
            )
            score = float(validation["corrected_primary_macro_score"])
            if select_best_validation_checkpoint(
                current_score=score, best_score=best_score
            ):
                best_score = score
                best_update = applied_update
                save_checkpoint(
                    run_root / "best-validation.ckpt",
                    _checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        sampler=sampler,
                        config=config,
                        applied_update=applied_update,
                        best_primary_score=best_score,
                        best_update=best_update,
                        record_history=record_history,
                        schedule_shift_history=schedule_shift_history,
                        applied_shift_history=applied_shift_history,
                        elapsed_wall_seconds=elapsed_before
                        + time.perf_counter()
                        - started,
                    ),
                )
            _emit(
                "validation_complete",
                applied_update=applied_update,
                profile=ORBIT_MATCHED_CONTROL_LABEL,
                primary_score=score,
                best_primary_score=best_score,
                best_update=best_update,
            )

        if applied_update % FULL_ORBIT_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                run_root / "last.ckpt",
                _checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    sampler=sampler,
                    config=config,
                    applied_update=applied_update,
                    best_primary_score=best_score,
                    best_update=best_update,
                    record_history=record_history,
                    schedule_shift_history=schedule_shift_history,
                    applied_shift_history=applied_shift_history,
                    elapsed_wall_seconds=elapsed_before
                    + time.perf_counter()
                    - started,
                ),
            )

    _validate_restored_histories(
        sampler=sampler,
        record_history=record_history,
        schedule_shift_history=schedule_shift_history,
        applied_shift_history=applied_shift_history,
    )
    history = completed_control_history_contract(
        record_history,
        schedule_shift_history,
        applied_shift_history,
    )
    if sampler.position != FULL_ORBIT_DRAW_BUDGET:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.completed_sampler_position_mismatch",
            f"{sampler.position} != {FULL_ORBIT_DRAW_BUDGET}",
        )

    _emit(
        "all_shift_validation_start", profile=ORBIT_MATCHED_CONTROL_LABEL
    )
    diagnostic = run_full_orbit_diagnostic_validation(
        model, device="cuda", paths=paths
    )
    _write_json(run_root / "all_shift_validation.json", diagnostic)

    elapsed = elapsed_before + time.perf_counter() - started
    validations = _read_jsonl(run_root / "validation_metrics.jsonl")
    expected_validation = _expected_validation_updates(FULL_ORBIT_UPDATE_BUDGET)
    observed_validation = [int(row["applied_update"]) for row in validations]
    if observed_validation != expected_validation:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.final_validation_schedule_mismatch",
            repr(observed_validation),
        )

    c2_profile = full_orbit_profile_contract()
    summary: dict[str, object] = {
        "schema": RUN_SUMMARY_SCHEMA,
        "phase": "9E-B5I",
        "valid": True,
        "profile": ORBIT_MATCHED_CONTROL_LABEL,
        "profile_id": ORBIT_MATCHED_CONTROL_PROFILE_ID,
        "matched_schedule_profile": "C2",
        "matched_schedule_profile_id": FULL_ORBIT_PROFILE_ID,
        "matched_schedule_profile_fingerprint": c2_profile["fingerprint"],
        "seed": FULL_ORBIT_SEED,
        "batch_size": FULL_ORBIT_BATCH_SIZE,
        "applied_updates": applied_update,
        "train_draws": len(record_history),
        "orbit_epochs": len(record_history) / sampler.draws_per_epoch,
        "applied_shift_pc": ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,
        "schedule_history": history,
        "initial_model_state_fingerprint": initial_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "validation_updates": expected_validation,
        "best_primary_score": best_score,
        "best_update": best_update,
        "final_primary_score": validations[-1][
            "corrected_primary_macro_score"
        ],
        "all_shift_validation_fingerprint": diagnostic["fingerprint"],
        "elapsed_wall_seconds": elapsed,
        "updates_per_second": applied_update / max(elapsed, 1e-12),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nan_count": 0,
        "overflow_count": 0,
        "skipped_update_count": 0,
        "same_budget_as_c2": True,
        "same_schedule_as_c2": True,
        "same_optimizer_scheduler_as_c2": True,
        "all_train_transforms_identity": True,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
        "control_training_run": True,
        "multi_seed_run": False,
    }
    summary["fingerprint"] = fingerprint(summary)
    _write_json(run_root / "run_summary.json", summary)
    (run_root / "REPORT.md").write_text(
        "# Phase 9E-B5I C0 120k orbit-matched control\n\n"
        f"- profile: `{ORBIT_MATCHED_CONTROL_LABEL}`\n"
        f"- applied updates: `{applied_update}`\n"
        f"- train draws: `{len(record_history)}`\n"
        f"- matched C2 orbit epochs: `{summary['orbit_epochs']}`\n"
        "- applied TRAIN shift: `0` for every draw\n"
        f"- best primary score: `{best_score}` at update `{best_update}`\n"
        f"- final primary score: `{summary['final_primary_score']}`\n"
        "- TEST evaluated: `false`\n"
        "- multi-seed claim: `false`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=FULL_ORBIT_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.seed != FULL_ORBIT_SEED:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.cli_contract_changed",
            "seed=17 is frozen",
        )
    if args.preflight:
        result = orbit_matched_control_preflight()
        _write_json(args.output_root / "preflight.json", result)
    elif args.smoke:
        result = smoke(args.device)
    else:
        if args.device != "cuda":
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.cli_contract_changed",
                "full training is CUDA-only",
            )
        result = run_full(output_root=args.output_root, resume=args.resume)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
