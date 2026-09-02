#!/usr/bin/env python3
"""Run the sealed Phase 9E-B5H C2 full-orbit CUDA experiment."""

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
    FullOrbitRuntimeConfig,
    FullOrbitSampler,
    build_full_orbit_optimizer_scheduler,
    build_full_orbit_table,
    full_orbit_preflight,
    full_orbit_profile_contract,
    run_full_orbit_diagnostic_validation,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    production_component_records,
    production_valid_shifts,
)


DEFAULT_OUTPUT = Path("outputs/phase9eb5h")
RUN_NAME = "c2-seed17-full-orbit-u120000"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True, separators=(",", ":")), flush=True)


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
    scalar = lambda value: None if value is None else float(value.detach().cpu())
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
    sampler: FullOrbitSampler,
    config: FullOrbitRuntimeConfig,
    applied_update: int,
    best_primary_score: float | None,
    best_update: int | None,
    record_history: Sequence[str],
    shift_history: Sequence[int],
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
        shift_history=shift_history,
    )
    payload["phase"] = "9E-B5H"
    payload["full_orbit_profile_fingerprint"] = full_orbit_profile_contract()["fingerprint"]
    payload["elapsed_wall_seconds"] = elapsed_wall_seconds
    return payload


def smoke(device: str) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        raise CorrectedTrainingError("analysisgnn.full_orbit.cuda_unavailable", "CUDA is required")
    batch, sidecar = build_source_free_fixture()
    _seed(FULL_ORBIT_SEED)
    model = CorrectedAnalysisGNNModel().to(device).eval()
    rows = []
    with torch.no_grad():
        for shift in range(12):
            raw = move_raw_graph_batch(
                transpose_raw_graph_batch(batch.raw_graph_batch, (shift,)), device
            )
            output = model(raw)
            rows.append({"shift_pc": shift, "finite": all(torch.isfinite(value).all() for value in output.logits.values())})
    body: dict[str, object] = {
        "schema": "Phase9EB5HFullOrbitSmoke@1.0.0",
        "valid": all(row["finite"] for row in rows),
        "record_id": sidecar["record_id"],
        "profile_id": FULL_ORBIT_PROFILE_ID,
        "per_shift": rows,
        "training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _validation_progress(applied_update: int):
    def progress(index: int, total: int, record_id: str) -> None:
        if index % 20 == 0 or index == total:
            _emit("validation_progress", applied_update=applied_update, record_index=index, record_count=total, record_id=record_id)
    return progress


def run_full(*, output_root: Path, resume: Path | None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise CorrectedTrainingError("analysisgnn.full_orbit.cuda_unavailable", "full training is CUDA-only")
    _deterministic_cuda()
    config = FullOrbitRuntimeConfig()
    run_root = output_root / RUN_NAME
    protected = ("training_metrics.jsonl", "validation_metrics.jsonl", "last.ckpt", "run_summary.json")
    if resume is None and any((run_root / name).exists() for name in protected):
        raise CorrectedTrainingError("analysisgnn.full_orbit.output_exists", "use --resume or a new output root")
    paths = ProductionArtifactPaths()
    preflight = full_orbit_preflight(paths)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "preflight.json", preflight)
    _write_json(run_root / "resolved_config.json", config.to_dict())
    _write_json(run_root / "environment.json", environment_fingerprint())

    _seed(FULL_ORBIT_SEED)
    model = CorrectedAnalysisGNNModel().to("cuda")
    initial_fingerprint = model_state_fingerprint(model)
    _seed(FULL_ORBIT_SEED * 1000 + 1)
    optimizer, scheduler = build_full_orbit_optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = FullOrbitSampler(
        build_full_orbit_table(
            production_component_records(paths), production_valid_shifts(paths)
        )
    )
    class_weights = load_frozen_class_weights()
    applied_update = 0
    best_score: float | None = None
    best_update: int | None = None
    record_history: list[str] = []
    shift_history: list[int] = []
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
            raise CorrectedTrainingError("analysisgnn.full_orbit.resume_interval_mismatch", str(applied_update))
        best_score = restored["best_primary_score"]
        best_update = restored["best_update"]
        record_history = list(restored["record_history"])
        shift_history = list(restored["shift_history"])
        elapsed_before = float(restored.get("elapsed_wall_seconds", 0.0))
        _emit("resume_loaded", applied_update=applied_update)
    else:
        _emit("validation_start", applied_update=0, profile="C2")
        validation = run_corrected_validation(model, device="cuda", paths=paths, progress=_validation_progress(0))
        _append_jsonl(run_root / "validation_metrics.jsonl", {"applied_update": 0, **validation})
        best_score = float(validation["corrected_primary_macro_score"])
        best_update = 0
        initial = _checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            sampler=sampler, config=config, applied_update=0,
            best_primary_score=best_score, best_update=best_update,
            record_history=(), shift_history=(), elapsed_wall_seconds=0.0,
        )
        save_checkpoint(run_root / "best-validation.ckpt", initial)
        save_checkpoint(run_root / "last.ckpt", initial)
        _emit("validation_complete", applied_update=0, profile="C2", primary_score=best_score)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    overflow_retries = 0
    while applied_update < FULL_ORBIT_UPDATE_BUDGET:
        draws = [sampler.peek(offset) for offset in range(FULL_ORBIT_BATCH_SIZE)]
        loaded = [load_production_record(row.record_id, split="train", paths=paths) for row in draws]
        raw = combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
        shifts = tuple(row.shift_pc for row in draws)
        raw = move_raw_graph_batch(transpose_raw_graph_batch(raw, shifts), "cuda")
        result = attempt_applied_update(
            model=model, raw_graph_batch=raw, sidecars=tuple(row[1] for row in loaded),
            shifts=shifts, class_weights=class_weights, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
        )
        if not result.applied:
            overflow_retries += 1
            if overflow_retries >= 3:
                raise CorrectedTrainingError("analysisgnn.full_orbit.persistent_overflow", str(applied_update))
            continue
        overflow_retries = 0
        applied_update += 1
        sampler.advance_after_applied_update()
        record_history.extend(row.record_id for row in draws)
        shift_history.extend(shifts)
        elapsed = elapsed_before + time.perf_counter() - started
        rate = applied_update / max(elapsed, 1e-12)
        training_row = {
            "applied_update": applied_update,
            "records": [row.record_id for row in draws],
            "components": [row.component_id for row in draws],
            "shifts": list(shifts),
            "orbit_epochs": sampler.position / sampler.draws_per_epoch,
            "gradient_norm": result.gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates_per_second": rate,
            **_loss_row(result.loss),
        }
        _append_jsonl(run_root / "training_metrics.jsonl", training_row)
        if applied_update % FULL_ORBIT_PROGRESS_INTERVAL == 0:
            _emit("training_progress", profile="C2", applied_update=applied_update,
                  update_budget=FULL_ORBIT_UPDATE_BUDGET, total_loss=training_row["total_loss"],
                  learning_rate=training_row["learning_rate"], updates_per_second=rate,
                  eta_minutes=(FULL_ORBIT_UPDATE_BUDGET-applied_update)/max(rate,1e-12)/60)
        if applied_update % FULL_ORBIT_VALIDATION_INTERVAL == 0:
            _emit("validation_start", applied_update=applied_update, profile="C2")
            validation = run_corrected_validation(model, device="cuda", paths=paths, progress=_validation_progress(applied_update))
            _append_jsonl(run_root / "validation_metrics.jsonl", {"applied_update": applied_update, **validation})
            score = float(validation["corrected_primary_macro_score"])
            if select_best_validation_checkpoint(current_score=score, best_score=best_score):
                best_score, best_update = score, applied_update
                save_checkpoint(run_root / "best-validation.ckpt", _checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    sampler=sampler, config=config, applied_update=applied_update,
                    best_primary_score=best_score, best_update=best_update,
                    record_history=record_history, shift_history=shift_history,
                    elapsed_wall_seconds=elapsed_before+time.perf_counter()-started,
                ))
            _emit("validation_complete", applied_update=applied_update, profile="C2", primary_score=score, best_primary_score=best_score, best_update=best_update)
        if applied_update % FULL_ORBIT_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(run_root / "last.ckpt", _checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                sampler=sampler, config=config, applied_update=applied_update,
                best_primary_score=best_score, best_update=best_update,
                record_history=record_history, shift_history=shift_history,
                elapsed_wall_seconds=elapsed_before+time.perf_counter()-started,
            ))

    if len(record_history) != FULL_ORBIT_DRAW_BUDGET:
        raise CorrectedTrainingError("analysisgnn.full_orbit.completed_draw_count_mismatch", str(len(record_history)))
    _emit("all_shift_validation_start", profile="C2")
    diagnostic = run_full_orbit_diagnostic_validation(model, device="cuda", paths=paths)
    _write_json(run_root / "all_shift_validation.json", diagnostic)
    elapsed = elapsed_before + time.perf_counter() - started
    validations = _read_jsonl(run_root / "validation_metrics.jsonl")
    summary: dict[str, object] = {
        "schema": "Phase9EB5HFullOrbitRunSummary@1.0.0",
        "phase": "9E-B5H", "valid": True, "profile": "C2",
        "profile_id": FULL_ORBIT_PROFILE_ID, "seed": FULL_ORBIT_SEED,
        "batch_size": FULL_ORBIT_BATCH_SIZE, "applied_updates": applied_update,
        "train_draws": len(record_history), "orbit_epochs": len(record_history)/sampler.draws_per_epoch,
        "initial_model_state_fingerprint": initial_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "best_primary_score": best_score, "best_update": best_update,
        "final_primary_score": validations[-1]["corrected_primary_macro_score"],
        "all_shift_validation_fingerprint": diagnostic["fingerprint"],
        "elapsed_wall_seconds": elapsed, "updates_per_second": applied_update/max(elapsed,1e-12),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nan_count": 0, "overflow_count": 0, "skipped_update_count": 0,
        "test_evaluated": False, "test_targets_used_for_evaluation": False,
        "full_orbit_training_run": True, "multi_seed_run": False,
    }
    summary["fingerprint"] = fingerprint(summary)
    _write_json(run_root / "run_summary.json", summary)
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
        raise CorrectedTrainingError("analysisgnn.full_orbit.cli_contract_changed", "seed=17 is frozen")
    if args.preflight:
        result = full_orbit_preflight()
        _write_json(args.output_root / "preflight.json", result)
    elif args.smoke:
        result = smoke(args.device)
    else:
        if args.device != "cuda":
            raise CorrectedTrainingError("analysisgnn.full_orbit.cli_contract_changed", "full training is CUDA-only")
        result = run_full(output_root=args.output_root, resume=args.resume)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
