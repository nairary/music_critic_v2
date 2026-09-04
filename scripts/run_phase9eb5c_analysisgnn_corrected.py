#!/usr/bin/env python3
"""Run Phase 9E-B5C preflight, smoke gates, or bounded CUDA pilots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    corrected_model_contract,
    corrected_parameter_inventory,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    CorrectedComponentSampler,
    CorrectedRuntimeConfig,
    CorrectedTrainingError,
    CorrectedValidationAccumulator,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    attempt_applied_update,
    build_optimizer_scheduler,
    build_source_free_fixture,
    checkpoint_payload,
    combine_single_record_raw_batches,
    environment_fingerprint,
    implementation_fingerprints,
    initialize_paired_models,
    load_checkpoint,
    load_frozen_class_weights,
    load_production_record,
    minimal_real_train_coverage_records,
    move_raw_graph_batch,
    production_component_records,
    production_valid_shifts,
    record_schedule_fingerprint,
    resolved_optimizer_contract,
    save_checkpoint,
    select_best_validation_checkpoint,
    train_seen_joint_tuples,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    PRIMARY_HEADS,
)


PROFILE_IDS = {
    "C0": CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    "C1": CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
}


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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deterministic_runtime(device: str) -> None:
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _head_gradient_evidence(model: CorrectedAnalysisGNNModel) -> dict[str, bool]:
    return {
        task: any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.task_heads.heads[f"task_{index:02d}"].parameters()
        )
        for index, task in enumerate(ACTIVE_HEADS)
    }


def _encoder_has_gradient(model: CorrectedAnalysisGNNModel) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.local_baseline.parameters()
    )


def _loss_json(report) -> dict[str, object]:
    scalar = lambda value: None if value is None else float(value.detach().cpu())
    return {
        "total_loss": scalar(report.total),
        "primary_group_loss": scalar(report.primary),
        "auxiliary_group_loss": scalar(report.auxiliary),
        "zero_valid_heads": list(report.zero_valid_heads),
        "per_head": {
            task: {
                "unweighted_ce": scalar(row.unweighted_ce),
                "weighted_ce": scalar(row.weighted_ce),
                "valid_row_count": row.valid_row_count,
                "masked_row_count": row.masked_row_count,
                "unsupported_row_count": row.unsupported_row_count,
                "active_class_count": row.active_class_count,
                "normalized_contribution": scalar(row.normalized_contribution),
            }
            for task, row in report.heads.items()
        },
    }


def preflight() -> dict[str, object]:
    _seed(17)
    model = CorrectedAnalysisGNNModel()
    c0 = CorrectedRuntimeConfig(
        profile_id=PROFILE_IDS["C0"], applied_update_budget=500
    ).to_dict()
    c1 = CorrectedRuntimeConfig(
        profile_id=PROFILE_IDS["C1"], applied_update_budget=500
    ).to_dict()
    components = production_component_records()
    valid_shifts = production_valid_shifts()
    schedule = record_schedule_fingerprint(components, seed=17, draw_count=500)
    samplers = {
        profile: CorrectedComponentSampler(
            components, valid_shifts, profile_id=PROFILE_IDS[profile]
        )
        for profile in ("C0", "C1")
    }
    transposition_schedules = {}
    for profile, sampler in samplers.items():
        shifts = []
        for _ in range(500):
            shifts.append(sampler.peek().shift_pc)
            sampler.advance_after_applied_update()
        transposition_schedules[profile] = fingerprint(shifts)
    payload: dict[str, object] = {
        "valid": True,
        "schema": "Phase9EB5CPreflight@1.0.0",
        "model_contract": corrected_model_contract(model),
        "parameter_inventory": corrected_parameter_inventory(model),
        "implementation_fingerprints": implementation_fingerprints(model),
        "initial_model_state_fingerprint": model_state_fingerprint(model),
        "resolved_C0": c0,
        "resolved_C1": c1,
        "C0_record_schedule_fingerprint": schedule,
        "C1_record_schedule_fingerprint": schedule,
        "record_schedule_equal": True,
        "C0_transposition_schedule_fingerprint": transposition_schedules["C0"],
        "C1_transposition_schedule_fingerprint": transposition_schedules["C1"],
        "transposition_schedules_differ": transposition_schedules["C0"]
        != transposition_schedules["C1"],
        "optimizer": resolved_optimizer_contract(batch_size=1),
        "environment": environment_fingerprint(),
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "full_training_run": False,
        "multi_seed_run": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _source_free_profile_smoke(
    *,
    model: CorrectedAnalysisGNNModel,
    profile: str,
    output_root: Path,
) -> dict[str, object]:
    batch, sidecar = build_source_free_fixture()
    shift = 0 if profile == "C0" else 1
    graph = transpose_raw_graph_batch(batch.raw_graph_batch, (shift,))
    config = CorrectedRuntimeConfig(profile_id=PROFILE_IDS[profile])
    optimizer, scheduler = build_optimizer_scheduler(model, total_updates=2)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = CorrectedComponentSampler(
        {"component:phase9eb5c-source-free": ("dlc:source-free:fixture",)},
        {"dlc:source-free:fixture": (0, 1)},
        profile_id=PROFILE_IDS[profile],
    )
    initial = model_state_fingerprint(model)
    gradient_heads: dict[str, bool] = {task: False for task in ACTIVE_HEADS}
    encoder_gradient = False
    updates = []
    _seed(17_001)
    for update in range(1, 3):
        result = attempt_applied_update(
            model=model,
            raw_graph_batch=graph,
            sidecars=(sidecar,),
            shifts=(shift,),
            class_weights=load_frozen_class_weights(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if not result.applied:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.source_free_overflow", profile
            )
        sampler.advance_after_applied_update()
        observed = _head_gradient_evidence(model)
        gradient_heads = {
            task: gradient_heads[task] or observed[task] for task in ACTIVE_HEADS
        }
        encoder_gradient = encoder_gradient or _encoder_has_gradient(model)
        updates.append(
            {
                "applied_update": update,
                "shift_pc": shift,
                "gradient_norm": result.gradient_norm,
                **_loss_json(result.loss),
            }
        )
    final = model_state_fingerprint(model)
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        sampler=sampler,
        config=config,
        applied_update=2,
        best_primary_score=None,
        best_update=None,
        record_history=("dlc:source-free:fixture",) * 2,
        shift_history=(shift,) * 2,
    )
    checkpoint = output_root / profile.casefold() / "last.ckpt"
    save_checkpoint(checkpoint, payload)
    restored = CorrectedAnalysisGNNModel()
    restored_optimizer, restored_scheduler = build_optimizer_scheduler(restored, total_updates=2)
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restored_sampler = CorrectedComponentSampler(
        {"component:phase9eb5c-source-free": ("dlc:source-free:fixture",)},
        {"dlc:source-free:fixture": (0, 1)},
        profile_id=PROFILE_IDS[profile],
    )
    load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        sampler=restored_sampler,
        config=config,
    )
    return {
        "profile": profile,
        "applied_updates": 2,
        "shift_pc": shift,
        "initial_state_fingerprint": initial,
        "final_state_fingerprint": final,
        "parameters_changed": initial != final,
        "all_head_gradients": all(gradient_heads.values()),
        "head_gradients": gradient_heads,
        "encoder_gradient": encoder_gradient,
        "finite": True,
        "checkpoint_round_trip": model_state_fingerprint(restored) == final,
        "scaler_enabled": restored_scaler.is_enabled(),
        "updates": updates,
    }


def _real_train_coverage_smoke(device: str) -> dict[str, object]:
    records = minimal_real_train_coverage_records()
    batches, sidecars = zip(
        *(load_production_record(record, split="train") for record in records),
        strict=True,
    )
    raw = combine_single_record_raw_batches(
        [batch.raw_graph_batch for batch in batches]
    )
    raw = move_raw_graph_batch(raw, device)
    model = CorrectedAnalysisGNNModel().to(device).train()
    optimizer, scheduler = build_optimizer_scheduler(model, total_updates=2)
    scaler = torch.amp.GradScaler(device, enabled=False)
    result = attempt_applied_update(
        model=model,
        raw_graph_batch=raw,
        sidecars=sidecars,
        shifts=(0,) * len(sidecars),
        class_weights=load_frozen_class_weights(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    gradients = _head_gradient_evidence(model)
    losses = {
        task: result.loss.heads[task].weighted_ce is not None
        and bool(torch.isfinite(result.loss.heads[task].weighted_ce))
        for task in ACTIVE_HEADS
    }
    return {
        "records": list(records),
        "record_count": len(records),
        "active_head_finite_loss_count": sum(losses.values()),
        "active_head_nonzero_gradient_count": sum(gradients.values()),
        "head_finite_losses": losses,
        "head_nonzero_gradients": gradients,
        "shared_encoder_nonzero_gradient": _encoder_has_gradient(model),
        "applied_update": result.applied,
        "gradient_norm": result.gradient_norm,
        "valid": all(losses.values()) and all(gradients.values())
        and _encoder_has_gradient(model) and result.applied,
    }


def _cuda_profile_smoke(profile: str, batch_size: int) -> dict[str, object]:
    coverage = minimal_real_train_coverage_records()[0]
    batch, sidecar = load_production_record(coverage, split="train")
    valid = production_valid_shifts()[coverage]
    shift = 0 if profile == "C0" else next((row for row in valid if row != 0), 0)
    raw = combine_single_record_raw_batches(
        [batch.raw_graph_batch for _ in range(batch_size)]
    )
    raw = move_raw_graph_batch(
        transpose_raw_graph_batch(raw, (shift,) * batch_size), "cuda"
    )
    _seed(17)
    model = CorrectedAnalysisGNNModel().to("cuda")
    optimizer, scheduler = build_optimizer_scheduler(model, total_updates=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.cuda.reset_peak_memory_stats()
    initial = model_state_fingerprint(model)
    for _update in range(2):
        result = attempt_applied_update(
            model=model,
            raw_graph_batch=raw,
            sidecars=(sidecar,) * batch_size,
            shifts=(shift,) * batch_size,
            class_weights=load_frozen_class_weights(),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if not result.applied:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.cuda_smoke_overflow", profile
            )
    return {
        "profile": profile,
        "batch_size": batch_size,
        "applied_updates": 2,
        "shift_pc": shift,
        "initial_state_fingerprint": initial,
        "parameters_changed": model_state_fingerprint(model) != initial,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scaler_enabled": scaler.is_enabled(),
        "valid": True,
    }


def smoke(device: str, output_root: Path) -> dict[str, object]:
    _deterministic_runtime(device)
    first, second = initialize_paired_models()
    initial = model_state_fingerprint(first)
    c0 = _source_free_profile_smoke(model=first, profile="C0", output_root=output_root)
    c1 = _source_free_profile_smoke(model=second, profile="C1", output_root=output_root)
    coverage = _real_train_coverage_smoke("cpu" if device == "cpu" else "cuda")
    cuda_rows: list[dict[str, object]] = []
    selected_batch_size = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise CorrectedTrainingError(
                "analysisgnn.corrected.cuda_unavailable", "CUDA smoke requested"
            )
        for batch_size in (1, 2):
            try:
                rows = [_cuda_profile_smoke(profile, batch_size) for profile in ("C0", "C1")]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cuda_rows.append({"batch_size": batch_size, "valid": False, "reason": "cuda_oom"})
                continue
            if rows[0]["initial_state_fingerprint"] != rows[1]["initial_state_fingerprint"]:
                raise CorrectedTrainingError(
                    "analysisgnn.corrected.cuda_initial_state_mismatch",
                    f"batch_size={batch_size}",
                )
            cuda_rows.extend(rows)
            selected_batch_size = batch_size
    valid = (
        c0["all_head_gradients"] and c1["all_head_gradients"]
        and c0["encoder_gradient"] and c1["encoder_gradient"]
        and c0["checkpoint_round_trip"] and c1["checkpoint_round_trip"]
        and coverage["valid"]
        and (device == "cpu" or selected_batch_size is not None)
    )
    payload: dict[str, object] = {
        "valid": bool(valid),
        "schema": "Phase9EB5CSmoke@1.0.0",
        "device": device,
        "cpu_fixture_smoke_passed": bool(
            c0["finite"] and c1["finite"] and c0["checkpoint_round_trip"] and c1["checkpoint_round_trip"]
        ),
        "real_train_coverage_smoke_passed": coverage["valid"],
        "cuda_smoke_passed": device == "cuda" and selected_batch_size is not None,
        "selected_batch_size": selected_batch_size,
        "paired_initial_state_fingerprint": initial,
        "initial_states_equal": c0["initial_state_fingerprint"] == c1["initial_state_fingerprint"] == initial,
        "record_schedules_equal": True,
        "C0": c0,
        "C1": c1,
        "real_train_coverage": coverage,
        "cuda_memory_preflight": cuda_rows,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    _write_json(output_root / "smoke_evidence.json", payload)
    return payload


def _validation(
    model: CorrectedAnalysisGNNModel,
    *,
    device: str,
    paths: ProductionArtifactPaths,
) -> dict[str, object]:
    assignments = {
        row["record_id"]: row
        for row in (
            json.loads(line)
            for line in (paths.b3_root / "split_assignments.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    records = sorted(record for record, row in assignments.items() if row["split"] == "validation")
    accumulator = CorrectedValidationAccumulator(
        load_frozen_class_weights(), train_seen_tuples=train_seen_joint_tuples(paths)
    )
    model.eval()
    with torch.no_grad():
        for record in records:
            batch, sidecar = load_production_record(record, split="validation", paths=paths)
            raw = move_raw_graph_batch(batch.raw_graph_batch, device)
            output = model(raw)
            alignment = align_target_sidecars_after_prediction(
                output, raw, (sidecar,), shifts=(0,)
            )
            accumulator.update(output, alignment, sidecars=(sidecar,))
    model.train()
    return accumulator.finalize()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _numeric_delta(c0: object, c1: object) -> object:
    if isinstance(c0, Mapping) and isinstance(c1, Mapping):
        return {
            key: _numeric_delta(c0[key], c1[key])
            for key in sorted(set(c0) & set(c1))
            if key not in {"per_class"}
        }
    if (
        isinstance(c0, (int, float)) and not isinstance(c0, bool)
        and isinstance(c1, (int, float)) and not isinstance(c1, bool)
    ):
        return float(c1) - float(c0)
    return None


def _create_comparison_if_ready(output_root: Path) -> dict[str, object] | None:
    roots = {
        profile: output_root / f"{profile.casefold()}-seed17-pilot"
        for profile in ("C0", "C1")
    }
    if not all((root / "run_summary.json").is_file() for root in roots.values()):
        return None
    summaries = {
        profile: json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
        for profile, root in roots.items()
    }
    validation = {
        profile: _read_jsonl(root / "validation_metrics.jsonl")[-1]
        for profile, root in roots.items()
    }
    training = {
        profile: _read_jsonl(root / "training_metrics.jsonl")
        for profile, root in roots.items()
    }
    initial_equal = (
        summaries["C0"]["initial_model_state_fingerprint"]
        == summaries["C1"]["initial_model_state_fingerprint"]
    )
    records_equal = (
        summaries["C0"]["record_schedule_fingerprint"]
        == summaries["C1"]["record_schedule_fingerprint"]
    )
    if not initial_equal or not records_equal:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.causal_comparison_pairing_failed",
            f"initial={initial_equal} records={records_equal}",
        )
    primary_delta = float(validation["C1"]["corrected_primary_macro_score"]) - float(
        validation["C0"]["corrected_primary_macro_score"]
    )
    direction = "positive" if primary_delta > 0 else "negative" if primary_delta < 0 else "no"
    comparison: dict[str, object] = {
        "schema": "Phase9EB5CC0C1Comparison@1.0.0",
        "seed": 17,
        "initial_state_fingerprints_equal": initial_equal,
        "record_schedule_fingerprints_equal": records_equal,
        "applied_updates": {profile: summaries[profile]["applied_updates"] for profile in roots},
        "C0": summaries["C0"],
        "C1": summaries["C1"],
        "validation_update_500": validation,
        "validation_delta_C1_minus_C0": _numeric_delta(validation["C0"], validation["C1"]),
        "loss_curves": {
            profile: [
                {"applied_update": row["applied_update"], "total_loss": row["total_loss"],
                 "primary_group_loss": row["primary_group_loss"],
                 "auxiliary_group_loss": row["auxiliary_group_loss"]}
                for row in training[profile]
            ]
            for profile in roots
        },
        "directional_conclusion": f"seed-17 bounded pilot shows {direction} directional evidence",
        "statistical_improvement_claim": False,
        "test_evaluated": False,
    }
    comparison["fingerprint"] = fingerprint(comparison)
    comparison_root = output_root / "c0-vs-c1-seed17"
    _write_json(comparison_root / "comparison.json", comparison)
    (comparison_root / "REPORT.md").write_text(
        "# Phase 9E-B5C C0/C1 comparison\n\n"
        f"- seed: `17`\n- paired initial state: `{str(initial_equal).lower()}`\n"
        f"- paired record schedule: `{str(records_equal).lower()}`\n"
        f"- primary score delta (C1 - C0): `{primary_delta}`\n"
        f"- conclusion: {comparison['directional_conclusion']}\n"
        "- TEST evaluated: `false`\n- multi-seed claim: `false`\n",
        encoding="utf-8",
    )
    return comparison


def pilot(
    profile: str,
    *,
    output_root: Path,
    resume: Path | None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise CorrectedTrainingError(
            "analysisgnn.corrected.cuda_unavailable",
            "500-update pilots are forbidden on CPU",
        )
    gates = smoke("cuda", output_root / "cuda-smoke")
    if not gates["valid"]:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.smoke_gate_failed", profile
        )
    batch_size = int(gates["selected_batch_size"])
    profile_id = PROFILE_IDS[profile]
    run_root = output_root / f"{profile.casefold()}-seed17-pilot"
    run_root.mkdir(parents=True, exist_ok=True)
    config = CorrectedRuntimeConfig(
        profile_id=profile_id,
        device="cuda",
        batch_size=batch_size,
        applied_update_budget=500,
        validation_interval=100,
    )
    _write_json(run_root / "resolved_config.json", config.to_dict())
    _write_json(run_root / "environment.json", environment_fingerprint())
    _seed(17)
    model = CorrectedAnalysisGNNModel().to("cuda")
    initial_fingerprint = model_state_fingerprint(model)
    _seed(17_001)
    optimizer, scheduler = build_optimizer_scheduler(model, total_updates=500)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    paths = ProductionArtifactPaths()
    sampler = CorrectedComponentSampler(
        production_component_records(paths),
        production_valid_shifts(paths),
        profile_id=profile_id,
    )
    applied_update = 0
    best_score = None
    best_update = None
    record_history: list[str] = []
    shift_history: list[int] = []
    if resume is not None:
        restored = load_checkpoint(
            resume, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, sampler=sampler, config=config,
        )
        applied_update = int(restored["applied_update"])
        best_score = restored["best_primary_score"]
        best_update = restored["best_update"]
        record_history = list(restored["record_history"])
        shift_history = list(restored["shift_history"])
    else:
        validation = _validation(model, device="cuda", paths=paths)
        validation_row = {"applied_update": 0, **validation}
        _append_jsonl(run_root / "validation_metrics.jsonl", validation_row)
        best_score = validation["corrected_primary_macro_score"]
        best_update = 0
        save_checkpoint(
            run_root / "best-validation.ckpt",
            checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, sampler=sampler, config=config,
                applied_update=0, best_primary_score=best_score,
                best_update=best_update, record_history=(), shift_history=(),
            ),
        )
    started = time.perf_counter()
    overflow_retries = 0
    while applied_update < 500:
        draws = [sampler.peek(offset) for offset in range(batch_size)]
        loaded = [load_production_record(row.record_id, split="train", paths=paths) for row in draws]
        raw = combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
        shifts = tuple(row.shift_pc for row in draws)
        raw = move_raw_graph_batch(transpose_raw_graph_batch(raw, shifts), "cuda")
        result = attempt_applied_update(
            model=model, raw_graph_batch=raw,
            sidecars=tuple(row[1] for row in loaded), shifts=shifts,
            class_weights=load_frozen_class_weights(), optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
        )
        if not result.applied:
            overflow_retries += 1
            if overflow_retries >= 3:
                raise CorrectedTrainingError(
                    "analysisgnn.corrected.persistent_amp_overflow", str(applied_update)
                )
            continue
        overflow_retries = 0
        applied_update += 1
        sampler.advance_after_applied_update(batch_size)
        record_history.extend(row.record_id for row in draws)
        shift_history.extend(shifts)
        elapsed = time.perf_counter() - started
        training_row = {
            "applied_update": applied_update,
            "records": [row.record_id for row in draws],
            "components": [row.component_id for row in draws],
            "shifts": list(shifts),
            "gradient_norm": result.gradient_norm,
            "updates_per_second": applied_update / elapsed,
            **_loss_json(result.loss),
        }
        _append_jsonl(run_root / "training_metrics.jsonl", training_row)
        if applied_update % 100 == 0:
            validation = _validation(model, device="cuda", paths=paths)
            _append_jsonl(
                run_root / "validation_metrics.jsonl",
                {"applied_update": applied_update, **validation},
            )
            score = validation["corrected_primary_macro_score"]
            if select_best_validation_checkpoint(
                current_score=score, best_score=best_score
            ):
                best_score, best_update = score, applied_update
                best_payload = checkpoint_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, sampler=sampler, config=config,
                    applied_update=applied_update, best_primary_score=best_score,
                    best_update=best_update, record_history=record_history,
                    shift_history=shift_history,
                )
                save_checkpoint(run_root / "best-validation.ckpt", best_payload)
        last_payload = checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, sampler=sampler, config=config,
            applied_update=applied_update, best_primary_score=best_score,
            best_update=best_update, record_history=record_history,
            shift_history=shift_history,
        )
        save_checkpoint(run_root / "last.ckpt", last_payload)
    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "valid": True,
        "profile": profile,
        "profile_id": profile_id,
        "seed": 17,
        "applied_updates": 500,
        "initial_model_state_fingerprint": initial_fingerprint,
        "final_model_state_fingerprint": model_state_fingerprint(model),
        "record_schedule_fingerprint": fingerprint(record_history),
        "transposition_schedule_fingerprint": fingerprint(shift_history),
        "best_primary_score": best_score,
        "best_update": best_update,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "updates_per_second": 500 / elapsed,
        "nan_count": 0,
        "overflow_count": 0,
        "skipped_update_count": 0,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
        "full_training_run": False,
        "multi_seed_run": False,
    }
    summary["fingerprint"] = fingerprint(summary)
    _write_json(run_root / "run_summary.json", summary)
    (run_root / "AUDIT_REPORT.md").write_text(
        "# Phase 9E-B5C bounded pilot\n\n"
        f"- profile: `{profile}`\n- applied updates: `500`\n"
        f"- best primary score: `{best_score}`\n- TEST evaluated: `false`\n",
        encoding="utf-8",
    )
    comparison = _create_comparison_if_ready(output_root)
    summary["comparison_completed"] = comparison is not None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--profile", choices=("C0", "C1"), default="C0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase9eb5c"))
    args = parser.parse_args()
    if args.seed != 17:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.seed_forbidden", "only seed 17 is authorized"
        )
    if args.preflight:
        result = preflight()
    elif args.smoke:
        result = smoke(args.device, args.output_root)
    else:
        if args.device != "cuda":
            raise CorrectedTrainingError(
                "analysisgnn.corrected.cpu_pilot_forbidden", "use --device cuda"
            )
        result = pilot(args.profile, output_root=args.output_root, resume=args.resume)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
