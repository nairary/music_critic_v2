#!/usr/bin/env python3
"""Run the paired Phase 9E-B5D 10,000-update CUDA full-training screen."""

from __future__ import annotations

import argparse
import importlib.util
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
    CorrectedComponentSampler,
    CorrectedRuntimeConfig,
    CorrectedTrainingError,
    ProductionArtifactPaths,
    attempt_applied_update,
    build_optimizer_scheduler,
    checkpoint_payload,
    combine_single_record_raw_batches,
    environment_fingerprint,
    load_checkpoint,
    load_frozen_class_weights,
    load_production_record,
    move_raw_graph_batch,
    production_component_records,
    production_valid_shifts,
    record_schedule_fingerprint,
    run_corrected_validation,
    save_checkpoint,
    select_best_validation_checkpoint,
    transpose_raw_graph_batch,
    transposition_schedule_fingerprint,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_BATCH_SIZE,
    FULL_CHECKPOINT_INTERVAL,
    FULL_PROGRESS_INTERVAL,
    FULL_RUN_SUMMARY_SCHEMA,
    FULL_SEED,
    FULL_UPDATE_BUDGET,
    FULL_VALIDATION_INTERVAL,
    FullTrainingContract,
    build_full_comparison,
    full_run_root_name,
    full_runtime_config,
    full_training_contract,
    full_validation_updates,
)


def _load_b5c_smoke():
    path = Path(__file__).with_name("run_phase9eb5c_analysisgnn_corrected.py")
    spec = importlib.util.spec_from_file_location("phase9eb5c_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B5C smoke gate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.smoke


b5c_smoke = _load_b5c_smoke()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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
        json.dumps({"event": event, **values}, sort_keys=True, separators=(",", ":")),
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
    scalar = lambda value: None if value is None else float(value.detach().cpu())
    return {
        "total_loss": scalar(report.total),
        "primary_group_loss": scalar(report.primary),
        "auxiliary_group_loss": scalar(report.auxiliary),
        "zero_valid_heads": list(report.zero_valid_heads),
        "per_head": {
            task: {
                "weighted_ce": scalar(row.weighted_ce),
                "valid_row_count": row.valid_row_count,
                "masked_row_count": row.masked_row_count,
                "unsupported_row_count": row.unsupported_row_count,
            }
            for task, row in report.heads.items()
        },
    }


def _checkpoint(
    *,
    model: CorrectedAnalysisGNNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: CorrectedComponentSampler,
    config: CorrectedRuntimeConfig,
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
        sampler=sampler,
        config=config,
        applied_update=applied_update,
        best_primary_score=best_primary_score,
        best_update=best_update,
        record_history=record_history,
        shift_history=shift_history,
    )
    payload["phase"] = "9E-B5D"
    payload["full_training_contract_fingerprint"] = full_training_contract()[
        "fingerprint"
    ]
    payload["elapsed_wall_seconds"] = elapsed_wall_seconds
    return payload


def _reconcile_resume_ledgers(run_root: Path, applied_update: int) -> None:
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
            "analysisgnn.full.resume_training_ledger_mismatch",
            f"expected={len(expected_training)} observed={len(observed_training)}",
        )
    validation = [
        row
        for row in _read_jsonl(validation_path)
        if int(row["applied_update"]) <= applied_update
    ]
    expected_validation = [
        update for update in full_validation_updates() if update <= applied_update
    ]
    observed_validation = [int(row["applied_update"]) for row in validation]
    if observed_validation != expected_validation:
        raise CorrectedTrainingError(
            "analysisgnn.full.resume_validation_ledger_mismatch",
            f"expected={expected_validation} observed={observed_validation}",
        )
    _rewrite_jsonl(training_path, training)
    _rewrite_jsonl(validation_path, validation)


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
            "analysisgnn.full.output_exists",
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


def preflight(output_root: Path) -> dict[str, object]:
    paths = ProductionArtifactPaths()
    components = production_component_records(paths)
    shifts = production_valid_shifts(paths)
    draw_count = FULL_UPDATE_BUDGET * FULL_BATCH_SIZE
    schedule = record_schedule_fingerprint(
        components, seed=FULL_SEED, draw_count=draw_count
    )
    transposition: dict[str, str] = {}
    for profile in ("C0", "C1"):
        sampler = CorrectedComponentSampler(
            components,
            shifts,
            profile_id=full_runtime_config(profile).profile_id,
        )
        draws = [sampler.peek(offset) for offset in range(draw_count)]
        transposition[profile] = transposition_schedule_fingerprint(draws)
    _seed(FULL_SEED)
    initial = model_state_fingerprint(CorrectedAnalysisGNNModel())
    payload: dict[str, object] = {
        "schema": "Phase9EB5DFullPreflight@1.0.0",
        "valid": True,
        "contract": full_training_contract(),
        "resolved_C0": full_runtime_config("C0").to_dict(),
        "resolved_C1": full_runtime_config("C1").to_dict(),
        "initial_model_state_fingerprint": initial,
        "record_schedule_fingerprints": {"C0": schedule, "C1": schedule},
        "record_schedule_fingerprints_equal": True,
        "transposition_schedule_fingerprints": transposition,
        "transposition_schedule_fingerprints_differ": (
            transposition["C0"] != transposition["C1"]
        ),
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "full_training_run": False,
        "multi_seed_run": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    _write_json(output_root / "preflight.json", payload)
    return payload


def _comparison_if_ready(output_root: Path) -> dict[str, object] | None:
    roots = {
        profile: output_root / full_run_root_name(profile)
        for profile in ("C0", "C1")
    }
    if not all((root / "run_summary.json").is_file() for root in roots.values()):
        return None
    summaries = {
        profile: json.loads(
            (root / "run_summary.json").read_text(encoding="utf-8")
        )
        for profile, root in roots.items()
    }
    validation = {
        profile: _read_jsonl(root / "validation_metrics.jsonl")
        for profile, root in roots.items()
    }
    comparison = build_full_comparison(summaries=summaries, validation=validation)
    destination = output_root / "c0-vs-c1-seed17-full-u10000"
    _write_json(destination / "comparison.json", comparison)
    (destination / "REPORT.md").write_text(
        "# Phase 9E-B5D paired full-training screen\n\n"
        f"- seed: `{FULL_SEED}`\n"
        f"- applied updates: `{FULL_UPDATE_BUDGET}`\n"
        f"- train draws: `{FULL_UPDATE_BUDGET * FULL_BATCH_SIZE}`\n"
        f"- sampler epochs: `{FullTrainingContract().sampler_epochs}`\n"
        "- paired initial state: `true`\n"
        "- paired record schedule: `true`\n"
        "- transposition schedules differ: `true`\n"
        f"- final primary score delta (C1 - C0): "
        f"`{comparison['final_primary_score_delta_C1_minus_C0']}`\n"
        f"- best primary score delta (C1 - C0): "
        f"`{comparison['best_primary_score_delta_C1_minus_C0']}`\n"
        f"- conclusion: {comparison['directional_conclusion']}\n"
        "- TEST evaluated: `false`\n"
        "- multi-seed claim: `false`\n",
        encoding="utf-8",
    )
    for profile, root in roots.items():
        updated = dict(summaries[profile])
        updated["comparison_completed"] = True
        updated.pop("fingerprint", None)
        updated["fingerprint"] = fingerprint(updated)
        _write_json(root / "run_summary.json", updated)
    return comparison


def run_full(
    profile: str,
    *,
    output_root: Path,
    resume: Path | None,
) -> dict[str, object]:
    config = full_runtime_config(profile)
    run_root = output_root / full_run_root_name(profile)
    if resume is None:
        _guard_fresh_run(run_root)
    if not torch.cuda.is_available():
        raise CorrectedTrainingError(
            "analysisgnn.full.cuda_unavailable", "full training is CUDA-only"
        )
    _deterministic_cuda()
    _emit("smoke_start", profile=profile)
    smoke = b5c_smoke("cuda", output_root / "cuda-smoke")
    if smoke.get("valid") is not True or smoke.get("selected_batch_size") != 2:
        raise CorrectedTrainingError(
            "analysisgnn.full.cuda_smoke_gate_failed", repr(smoke.get("selected_batch_size"))
        )
    _emit("smoke_complete", profile=profile, selected_batch_size=2)

    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "resolved_config.json", config.to_dict())
    _write_json(run_root / "full_training_contract.json", full_training_contract())
    _write_json(run_root / "environment.json", environment_fingerprint())

    _seed(FULL_SEED)
    model = CorrectedAnalysisGNNModel().to("cuda")
    initial_fingerprint = model_state_fingerprint(model)
    _seed(FULL_SEED * 1000 + 1)
    optimizer, scheduler = build_optimizer_scheduler(
        model, total_updates=FULL_UPDATE_BUDGET
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    paths = ProductionArtifactPaths()
    sampler = CorrectedComponentSampler(
        production_component_records(paths),
        production_valid_shifts(paths),
        profile_id=config.profile_id,
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
            sampler=sampler,
            config=config,
        )
        applied_update = int(restored["applied_update"])
        if applied_update % FULL_CHECKPOINT_INTERVAL != 0:
            raise CorrectedTrainingError(
                "analysisgnn.full.resume_checkpoint_interval_mismatch",
                str(applied_update),
            )
        best_score = restored["best_primary_score"]
        best_update = restored["best_update"]
        record_history = list(restored["record_history"])
        shift_history = list(restored["shift_history"])
        elapsed_before = float(restored.get("elapsed_wall_seconds", 0.0))
        _reconcile_resume_ledgers(run_root, applied_update)
        _emit("resume_loaded", profile=profile, applied_update=applied_update)
    else:
        _emit("validation_start", profile=profile, applied_update=0)
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
        initial_checkpoint = _checkpoint(
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
            shift_history=(),
            elapsed_wall_seconds=0.0,
        )
        save_checkpoint(run_root / "best-validation.ckpt", initial_checkpoint)
        save_checkpoint(run_root / "last.ckpt", initial_checkpoint)
        _emit(
            "validation_complete",
            profile=profile,
            applied_update=0,
            primary_score=best_score,
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    overflow_retries = 0
    while applied_update < FULL_UPDATE_BUDGET:
        draws = [sampler.peek(offset) for offset in range(FULL_BATCH_SIZE)]
        loaded = [
            load_production_record(row.record_id, split="train", paths=paths)
            for row in draws
        ]
        raw = combine_single_record_raw_batches(
            [row[0].raw_graph_batch for row in loaded]
        )
        shifts = tuple(row.shift_pc for row in draws)
        raw = move_raw_graph_batch(transpose_raw_graph_batch(raw, shifts), "cuda")
        result = attempt_applied_update(
            model=model,
            raw_graph_batch=raw,
            sidecars=tuple(row[1] for row in loaded),
            shifts=shifts,
            class_weights=class_weights,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if not result.applied:
            overflow_retries += 1
            if overflow_retries >= 3:
                raise CorrectedTrainingError(
                    "analysisgnn.full.persistent_overflow", str(applied_update)
                )
            continue
        overflow_retries = 0
        applied_update += 1
        sampler.advance_after_applied_update(FULL_BATCH_SIZE)
        record_history.extend(row.record_id for row in draws)
        shift_history.extend(shifts)
        elapsed = elapsed_before + time.perf_counter() - started
        update_rate = applied_update / max(elapsed, 1e-12)
        row = {
            "applied_update": applied_update,
            "records": [draw.record_id for draw in draws],
            "components": [draw.component_id for draw in draws],
            "shifts": list(shifts),
            "gradient_norm": result.gradient_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates_per_second": update_rate,
            **_loss_row(result.loss),
        }
        _append_jsonl(run_root / "training_metrics.jsonl", row)

        if applied_update % FULL_PROGRESS_INTERVAL == 0:
            _emit(
                "training_progress",
                profile=profile,
                applied_update=applied_update,
                update_budget=FULL_UPDATE_BUDGET,
                total_loss=row["total_loss"],
                learning_rate=row["learning_rate"],
                updates_per_second=update_rate,
                eta_minutes=(FULL_UPDATE_BUDGET - applied_update)
                / max(update_rate, 1e-12)
                / 60,
            )

        if applied_update % FULL_VALIDATION_INTERVAL == 0:
            _emit("validation_start", profile=profile, applied_update=applied_update)
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
                        shift_history=shift_history,
                        elapsed_wall_seconds=elapsed_before
                        + time.perf_counter()
                        - started,
                    ),
                )
            _emit(
                "validation_complete",
                profile=profile,
                applied_update=applied_update,
                primary_score=score,
                best_primary_score=best_score,
                best_update=best_update,
            )

        if applied_update % FULL_CHECKPOINT_INTERVAL == 0:
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
                    shift_history=shift_history,
                    elapsed_wall_seconds=elapsed_before
                    + time.perf_counter()
                    - started,
                ),
            )

    elapsed = elapsed_before + time.perf_counter() - started
    validations = _read_jsonl(run_root / "validation_metrics.jsonl")
    if [int(row["applied_update"]) for row in validations] != list(
        full_validation_updates()
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full.final_validation_schedule_mismatch", profile
        )
    expected_record_fingerprint = record_schedule_fingerprint(
        sampler.component_records,
        seed=FULL_SEED,
        draw_count=FULL_UPDATE_BUDGET * FULL_BATCH_SIZE,
    )
    expected_sampler = CorrectedComponentSampler(
        sampler.component_records,
        sampler.valid_shifts_by_record,
        profile_id=config.profile_id,
    )
    expected_draws = [
        expected_sampler.peek(offset)
        for offset in range(FULL_UPDATE_BUDGET * FULL_BATCH_SIZE)
    ]
    if (
        record_history != [draw.record_id for draw in expected_draws]
        or shift_history != [draw.shift_pc for draw in expected_draws]
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full.completed_schedule_mismatch", profile
        )
    summary: dict[str, object] = {
        "schema": FULL_RUN_SUMMARY_SCHEMA,
        "valid": True,
        "phase": "9E-B5D",
        "profile": profile,
        "profile_id": config.profile_id,
        "seed": FULL_SEED,
        "batch_size": FULL_BATCH_SIZE,
        "applied_updates": applied_update,
        "train_draws": len(record_history),
        "sampler_epochs": len(record_history) / FullTrainingContract().draws_per_epoch,
        "validation_updates": list(full_validation_updates()),
        "initial_model_state_fingerprint": initial_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "record_schedule_fingerprint": expected_record_fingerprint,
        "transposition_schedule_fingerprint": transposition_schedule_fingerprint(
            expected_draws
        ),
        "best_primary_score": best_score,
        "best_update": best_update,
        "final_primary_score": validations[-1]["corrected_primary_macro_score"],
        "elapsed_wall_seconds": elapsed,
        "updates_per_second": applied_update / max(elapsed, 1e-12),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nan_count": 0,
        "overflow_count": 0,
        "skipped_update_count": 0,
        "checkpoint_interval": FULL_CHECKPOINT_INTERVAL,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
        "full_training_run": True,
        "multi_seed_run": False,
        "comparison_completed": False,
    }
    summary["fingerprint"] = fingerprint(summary)
    _write_json(run_root / "run_summary.json", summary)
    (run_root / "REPORT.md").write_text(
        "# Phase 9E-B5D full-training screen\n\n"
        f"- profile: `{profile}`\n"
        f"- applied updates: `{applied_update}`\n"
        f"- train draws: `{len(record_history)}`\n"
        f"- sampler epochs: `{summary['sampler_epochs']}`\n"
        f"- best primary score: `{best_score}` at update `{best_update}`\n"
        f"- final primary score: `{summary['final_primary_score']}`\n"
        "- TEST evaluated: `false`\n"
        "- multi-seed claim: `false`\n",
        encoding="utf-8",
    )
    comparison = _comparison_if_ready(output_root)
    if comparison is not None:
        summary = json.loads(
            (run_root / "run_summary.json").read_text(encoding="utf-8")
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--profile", choices=("C0", "C1"), default="C0")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=FULL_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/phase9eb5d")
    )
    args = parser.parse_args()
    if args.seed != FULL_SEED or args.device != "cuda":
        raise CorrectedTrainingError(
            "analysisgnn.full.cli_contract_changed", "seed=17 and CUDA are frozen"
        )
    if args.preflight:
        result = preflight(args.output_root)
    else:
        result = run_full(
            args.profile, output_root=args.output_root, resume=args.resume
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
