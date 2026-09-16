#!/usr/bin/env python3
"""Registered synthetic prefetch and serialized-resume gate on one runtime.

All heavy imports are inside functions. CUDA launch requires inherited thread
limits before Torch/NumPy import; spawned workers import this file without
running main, initializing a model, or touching CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
WORKER_COUNTS = (0, 1, 2, 4)
MEASURED_DRAWS = 6
WARMUP_DRAWS = 2
RESUME_UPDATES = 24
RESUME_DRAWS = 48


def _check_thread_environment(device):
    values = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if device == "cuda" and any(value != "1" for value in values.values()):
        raise ValueError("prefetch_fixture_gate.cuda_requires_inherited_single_thread_limits:" +
                         ",".join(name for name, value in values.items() if value != "1"))
    return values


def _configure_parent_threads(torch):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _resume_accounting(result):
    """Count completed own synthetic branches; never infer partial progress."""
    if result.get("worker_counts") != list(WORKER_COUNTS) or type(result.get("passed")) is not bool:
        raise ValueError("prefetch_fixture_gate.resume_result_invalid")
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != len(WORKER_COUNTS):
        raise ValueError("prefetch_fixture_gate.resume_arm_coverage_invalid")
    observed, updates = set(), 0
    for row in rows:
        worker_count = row.get("worker_count")
        if type(worker_count) is not int or worker_count not in WORKER_COUNTS or worker_count in observed:
            raise ValueError("prefetch_fixture_gate.resume_arm_coverage_invalid")
        observed.add(worker_count)
        for branch in ("uninterrupted", "resumed"):
            committed = row.get(branch, {}).get("committed")
            if type(committed) is not int or committed != MEASURED_DRAWS // 2:
                raise ValueError("prefetch_fixture_gate.resume_draw_count_invalid")
            updates += committed
        if worker_count:
            pending = row.get("resumed", {}).get("pending_at_resume")
            if type(pending) is not int or pending <= 0:
                raise ValueError("prefetch_fixture_gate.resume_pending_not_exercised")
    if updates != RESUME_UPDATES:
        raise ValueError("prefetch_fixture_gate.resume_update_budget_mismatch")
    return {"planned_updates": RESUME_UPDATES, "planned_record_draws": RESUME_DRAWS,
            "actual_updates": updates, "actual_record_draws": 2 * updates,
            "scope": "supplemental_synthetic_resume_excludes_throughput_windows"}


def run_registered_fixture_gate(*, experiment_id, parent_experiment_id, output_root,
                                device="cuda", repository_root=None):
    if device not in ("cpu", "cuda"):
        raise ValueError("prefetch_fixture_gate.invalid_device")
    inherited_threads = _check_thread_environment(device)  # Before all heavy imports.
    from music_critic.experiments.analysisgnn import performance_replay as existing
    from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchConfig
    from music_critic.experiments.analysisgnn.performance_prefetch_replay import (
        build_prefetch_schedule, protocol_bindings, run_fixture_resume_gate, run_prefetch_replay,
    )
    import torch

    if not parent_experiment_id:
        raise ValueError("prefetch_fixture_gate.terminal_parent_required")
    config = PrefetchConfig()
    _, descriptor = build_prefetch_schedule(mode="fixture", measured_draws=MEASURED_DRAWS,
                                           warmup_draws=WARMUP_DRAWS, reserved_bytes=2_000_000)
    bindings = protocol_bindings(mode="fixture", descriptor=descriptor, reservation_manifest=None,
                                config=config, worker_counts=WORKER_COUNTS)
    protocol = {**bindings, "mode": "fixture", "device": device, "repeats": 1,
        "batch_size": 2, "batch_sizes": [2], "measured_draws": MEASURED_DRAWS,
        "warmup_draws": WARMUP_DRAWS, "update_strategy": "optimized", "quality_claims": False,
        "canonical_checkpoint_input_used": False, "own_fixture_checkpoint_roundtrip": True,
        "supplemental_resume_budget": {"planned_updates": RESUME_UPDATES, "planned_record_draws": RESUME_DRAWS},
        "question": "Does ordered prefetch preserve exact synthetic training and serialized resume on this runtime?",
        "hypothesis": "Direct worker zero and spawn workers preserve full model, Adam, logits and RNG states across resume.",
        "decision_use": "Gate runtime compatibility before TRAIN sizing or production throughput; no RAM or musical-quality claim."}
    output = Path(output_root)

    def execute():
        if device == "cuda":
            existing._assert_idle_gpu()
            if not torch.cuda.is_available():
                raise RuntimeError("prefetch_fixture_gate.cuda_unavailable")
        throughput = run_prefetch_replay(output, experiment_id=experiment_id, mode="fixture", device=device,
            measured_draws=MEASURED_DRAWS, warmup_draws=WARMUP_DRAWS, repeats=1,
            worker_counts=WORKER_COUNTS, config=config, descriptor=descriptor)
        existing._write(output / "throughput-summary.json", throughput)
        environment = {"inherited_thread_environment": inherited_threads,
            "parent_torch_threads": torch.get_num_threads(),
            "parent_torch_interop_threads": torch.get_num_interop_threads(),
            "worker_torch_thread_policy": {"intra_op": 1, "inter_op": 1}}
        supplemental = {**protocol["supplemental_resume_budget"], "actual_updates": 0, "actual_record_draws": 0,
                        "scope": "supplemental_synthetic_resume_excludes_throughput_windows"}
        result = {**throughput, "resume_result": None, "resume_equivalence_checked": False,
            "canonical_checkpoint_input_used": False, "own_fixture_checkpoint_input_used": False,
            "fixture_gate_environment": environment, "supplemental_resume_budget": supplemental}
        if throughput.get("equivalence_passed") is not True:
            result.update(resume_status="not_run_throughput_equivalence_failed", equivalence_passed=False)
            existing._write(output / "summary.json", result)
            return result
        result.update(completed=False, resume_status="running", equivalence_passed=False)
        existing._write(output / "summary.json", result)
        try:
            gc.collect()  # Outside every measured window, before the separate fixture gate.
            if device == "cuda":
                existing._assert_idle_gpu()
            resume = run_fixture_resume_gate(output / "own-checkpoints", worker_counts=WORKER_COUNTS,
                measured_draws=MEASURED_DRAWS, resume_after_updates=1, device=device)
            supplemental = _resume_accounting(resume)
            result.update(completed=True, resume_status="completed", resume_result=resume,
                resume_equivalence_checked=True, own_fixture_checkpoint_input_used=True, checkpoint_input_used=True,
                supplemental_resume_budget=supplemental,
                equivalence_passed=throughput["equivalence_passed"] and resume["passed"],
                actual_attempted_updates=throughput["actual_attempted_updates"] + supplemental["actual_updates"],
                actual_applied_updates=throughput["actual_applied_updates"] + supplemental["actual_updates"])
            result["actual_training_draws_including_warmup_and_resume"] = 2 * result["actual_applied_updates"]
            existing._write(output / "summary.json", result)
            return result
        except BaseException as exc:
            result.update(completed=False, resume_status="failed", equivalence_passed=False,
                supplemental_resume_budget={**supplemental, "actual_updates": None, "actual_record_draws": None},
                actual_attempted_updates=None, actual_applied_updates=None,
                completed_throughput_attempted_updates=throughput["actual_attempted_updates"],
                completed_throughput_applied_updates=throughput["actual_applied_updates"],
                actual_training_draws_including_warmup_and_resume=None,
                resume_failure={"type": type(exc).__name__, "message": str(exc)})
            existing._write(output / "summary.json", result)
            raise

    return run_registered_replay(experiment_id=experiment_id, parent_experiment_id=parent_experiment_id,
        output_root=output, protocol=protocol, execute=execute, repository_root=repository_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        _check_thread_environment(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    import torch
    _configure_parent_threads(torch)  # Once, in a fresh parent, before model work.
    result = run_registered_fixture_gate(experiment_id=args.experiment_id,
        parent_experiment_id=args.parent_experiment_id, output_root=args.output_root, device=args.device)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
