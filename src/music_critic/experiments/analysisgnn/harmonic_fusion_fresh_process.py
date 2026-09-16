"""Synthetic process-boundary resume gate, separate from scientific dispatch.

All inputs are generated fixtures. Each arm performs six uninterrupted
updates and three prefix updates; a new Python process restores and finishes
the remaining three. CUDA execution belongs to a registered coordinator slot.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .contracts import fingerprint
from .corrected_model import CorrectedAnalysisGNNModel
from .corrected_training import build_source_free_fixture, load_frozen_class_weights
from .full_orbit_training import FullOrbitPair
from .harmonic_fusion_resume import (
    create_training_payload, load_training_checkpoint, prepare_training_resume_runtime,
    preserve_caller_rng, save_training_checkpoint,
)
from .harmonic_fusion_run import _ProgressReceipt, _manifest, _write_bytes_once, _write_json_once
from .harmonic_fusion_training import FusionTrainingConfig, FusionTrainingSession, evaluate_validation
from .performance_replay import recursive_state_fingerprint


SCHEMA = "HarmonicFusionFreshProcessGate@1.0.0"
WORKER_SCHEMA = "HarmonicFusionFreshProcessWorker@1.0.0"
_MODULE_NAMES = ("harmonic_fusion_fresh_process.py", "harmonic_fusion_training.py",
                 "harmonic_fusion_resume.py", "harmonic_logit_fusion.py", "corrected_model.py",
                 "corrected_training.py", "full_orbit_training.py")


def _environment(device):
    """Observe process policy without querying or initializing a CUDA device."""
    return {"python": sys.version.split()[0], "executable": sys.executable, "device": device,
            "packages": {name: version(name) for name in ("torch", "numpy", "torch-geometric")},
            "torch_cuda_build": torch.version.cuda, "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "native_threads": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def _recipe(reads):
    config = FusionTrainingConfig(total_updates=6, warmup_updates=2,
                                  validation_interval=3, checkpoint_interval=3,
                                  validation_mode="required")
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"])
                  for shift in range(12))
    bindings = {"source_kind": "fresh_process_generated_fixture_only",
                "class_weight_fingerprint": weights.payload_fingerprint,
                "sidecar_fingerprint": sidecar["fingerprint"],
                "source_code": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in _MODULE_NAMES}}

    def reader(record_id, *, split):
        if split not in reads or record_id != sidecar["record_id"]:
            raise ValueError("fusion_fresh_process.fixture_boundary")
        reads[split] += 1
        return batch.raw_graph_batch, deepcopy(sidecar)

    def validate(model, *, split):
        return evaluate_validation(model, records=(sidecar["record_id"],), reader=reader,
                                   class_weights=weights, split=split)

    return config, bindings, table, reader, weights, validate


def _failure(root, progress, exc, *, schema):
    try:
        failure = progress.fail(exc)
        receipt = {"schema_version": schema, "passed": False, "process_id": os.getpid(),
                   "failure": failure, "progress": progress.snapshot(),
                   "real_data_accessed": False, "test_accessed": False,
                   "test_inputs_accessed": False, "test_targets_accessed": False,
                   "test_metrics_computed": False, "fresh_process_resume": True,
                   "unreported_child_work_is_unknown": True}
        _write_json_once(root / "failure-receipt.json", receipt)
        _manifest(root, filename="failure-artifact-manifest.json")
    except BaseException as storage_error:
        exc.add_note(f"Failure receipt incomplete: {type(storage_error).__name__}: {storage_error}")


def run_resume_worker(output_root, *, checkpoint, expected_sha256, expected_size_bytes,
                      enabled: bool, device="cpu", expected_cuda_device_count=None):
    """Restore one fixed synthetic update-three checkpoint in this new process."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    reads = {"train": 0, "validation": 0}
    progress = _ProgressReceipt(root, reads)
    arm = "fusion" if enabled else "control"
    progress.begin(arm, "fresh_process_resumed", 3)
    try:
        config, bindings, table, reader, weights, validate = _recipe(reads)
        readiness = prepare_training_resume_runtime(device, expected_cuda_device_count=expected_cuda_device_count)
        environment = _environment(device)
        progress.phase("strict_restore")
        session = load_training_checkpoint(checkpoint, expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes, config=config, bindings=bindings,
            table=table, device=device, expected_enabled=enabled)
        progress.attach(session)  # Exact expected cursor three is mandatory.
        initial_validation_updates = [row["update"] for row in session.validations]
        session.run_until(6, reader, weights, validate=validate, progress=progress.event)
        progress.phase("checkpoint_save")
        artifact = save_training_checkpoint(session, root / "final.pt")
        state_fingerprint = recursive_state_fingerprint(create_training_payload(session))
        progress.finish()
        result = {"schema_version": WORKER_SCHEMA, "passed": True, "device": device,
                  "process_id": os.getpid(), "runtime_readiness": readiness,
                  "environment": environment, "environment_fingerprint": fingerprint(environment),
                  "arm": arm, "source_checkpoint_sha256": expected_sha256,
                  "source_checkpoint_size_bytes": expected_size_bytes,
                  "initial_validation_updates": initial_validation_updates,
                  "final_validation_updates": [row["update"] for row in session.validations],
                  "final_checkpoint": artifact, "full_state_fingerprint": state_fingerprint,
                  "progress": progress.snapshot(), "actual_updates": progress.snapshot()["applied_updates"],
                  "actual_draws": reads["train"], "real_data_accessed": False, "test_accessed": False,
                  "test_inputs_accessed": False, "test_targets_accessed": False,
                  "test_metrics_computed": False, "fresh_process_resume": True}
        content = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _manifest(root, pending_files={"report.json": content})
        _write_bytes_once(root / "report.json", content)
        return result
    except BaseException as exc:
        _failure(root, progress, exc, schema=WORKER_SCHEMA)
        raise


def _read_child_receipt(root):
    path = root / "report.json"
    if not path.exists():
        path = root / "failure-receipt.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _merge_child_progress(progress, reads, child):
    """Retain confirmed child work, including a failed child's successful prefix."""
    if child is None:
        return
    if child.get("schema_version") != WORKER_SCHEMA:
        raise ValueError("fusion_fresh_process.child_schema_mismatch")
    child_progress = child["progress"]
    trajectories = child_progress["trajectories"]
    if (len(trajectories) != 1 or trajectories[0]["arm"] != progress.current["arm"]
            or trajectories[0]["trajectory"] != "fresh_process_resumed"
            or trajectories[0]["start_update"] != 3):
        raise ValueError("fusion_fresh_process.child_trajectory_mismatch")
    progress.current.update(deepcopy(trajectories[0]))
    for split in reads:
        reads[split] += child_progress["reader_calls"][split]
    progress.phase("fresh_process_child_receipt", "observed")


def run_fresh_process_gate(output_root, *, device="cpu", expected_cuda_device_count=None,
                           experiment_record=None):
    """Run both fixture arms; compare whole state across actual Python processes."""
    if device == "cuda" and experiment_record is None:
        raise ValueError("fusion_fresh_process.planned_experiment_record_required")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    reads = {"train": 0, "validation": 0}
    progress = _ProgressReceipt(root, reads)
    started = time.perf_counter()
    try:
        config, bindings, table, reader, weights, validate = _recipe(reads)
        readiness = prepare_training_resume_runtime(device, expected_cuda_device_count=expected_cuda_device_count)
        environment = _environment(device)
        with preserve_caller_rng():
            torch.manual_seed(17)
            base = CorrectedAnalysisGNNModel()
        rows = []
        for enabled in (False, True):
            arm = "fusion" if enabled else "control"

            def fresh():
                return FusionTrainingSession(base=base, enabled=enabled, config=config,
                                             bindings=bindings, table=table, device=device)

            progress.begin(arm, "uninterrupted", 0)
            full = fresh()
            progress.attach(full)
            full.run_until(6, reader, weights, validate=validate, progress=progress.event)
            expected_fingerprint = recursive_state_fingerprint(create_training_payload(full))
            progress.finish()
            del full
            progress.begin(arm, "prefix", 0)
            prefix = fresh()
            progress.attach(prefix)
            prefix.run_until(3, reader, weights, validate=validate, progress=progress.event)
            prefix_path = root / f"{arm}-update-000003.pt"
            prefix_artifact = save_training_checkpoint(prefix, prefix_path)
            progress.finish()
            del prefix

            progress.begin(arm, "fresh_process_resumed", 3)
            child_root = root / f"{arm}-fresh-process"
            entrypoint = Path(__file__).resolve().parents[4] / "scripts" / "run_harmonic_fusion_fresh_process_gate.py"
            command = [sys.executable, str(entrypoint), "--smoke", "--device", device,
                       "--output-root", str(child_root), "--worker-checkpoint", str(prefix_path),
                       "--checkpoint-sha256", prefix_artifact["sha256"],
                       "--checkpoint-size-bytes", str(prefix_artifact["size_bytes"]), "--arm", arm]
            if device == "cuda":
                command += ["--expected-cuda-device-count", str(expected_cuda_device_count),
                            "--experiment-record", str(Path(experiment_record).resolve())]
            _write_json_once(root / f"{arm}-worker-command.json", command)
            progress.phase("fresh_process_start")
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
            _write_bytes_once(root / f"{arm}-worker.stdout.txt", completed.stdout.encode())
            _write_bytes_once(root / f"{arm}-worker.stderr.txt", completed.stderr.encode())
            child = _read_child_receipt(child_root)
            _merge_child_progress(progress, reads, child)
            if completed.returncode != 0 or child is None or child.get("passed") is not True:
                raise ValueError(f"fusion_fresh_process.child_failed: exit={completed.returncode}; root={child_root}")
            if child["process_id"] == os.getpid() or child["arm"] != arm:
                raise ValueError("fusion_fresh_process.process_boundary_not_verified")
            if child["environment"] != environment or child["environment_fingerprint"] != fingerprint(environment):
                raise ValueError("fusion_fresh_process.runtime_policy_mismatch")
            evidence = child["final_checkpoint"]
            restored = load_training_checkpoint(child_root / "final.pt", expected_sha256=evidence["sha256"],
                expected_size_bytes=evidence["size_bytes"], config=config, bindings=bindings,
                table=table, device=device, expected_enabled=enabled)
            observed = recursive_state_fingerprint(create_training_payload(restored))
            if observed != expected_fingerprint or observed != child["full_state_fingerprint"]:
                raise ValueError("fusion_fresh_process.trajectory_mismatch")
            if child["initial_validation_updates"] != [0, 3] or child["final_validation_updates"] != [0, 3, 6]:
                raise ValueError("fusion_fresh_process.validation_history_mismatch")
            progress.finish()
            child_manifest = (child_root / "artifact-manifest.json").read_bytes()
            rows.append({"arm": arm, "worker_process_id": child["process_id"],
                         "whole_state_equal": True, "full_state_fingerprint": observed,
                         "worker_runtime_readiness": child["runtime_readiness"],
                         "worker_artifact_manifest": {"sha256": sha256(child_manifest).hexdigest(),
                                                      "size_bytes": len(child_manifest)},
                         "prefix_checkpoint": prefix_artifact, "resumed_checkpoint": evidence})
            del restored
        accounting = progress.snapshot()
        if (accounting["attempted_updates"] != 24 or accounting["applied_updates"] != 24
                or accounting["attempted_draws"] != 48 or accounting["applied_draws"] != 48
                or accounting["reader_calls"] != {"train": 48, "validation": 12}):
            raise ValueError("fusion_fresh_process.accounting_mismatch")
        result = {"schema_version": SCHEMA, "passed": True, "device": device,
                  "parent_process_id": os.getpid(), "runtime_readiness": readiness,
                  "environment": environment, "environment_fingerprint": fingerprint(environment),
                  "arms": rows, "progress": accounting, "config": config.to_dict(),
                  "bindings": bindings, "wall_seconds": time.perf_counter() - started,
                  "actual_updates": accounting["applied_updates"], "actual_draws": reads["train"],
                  "real_data_accessed": False, "test_accessed": False,
                  "test_inputs_accessed": False, "test_targets_accessed": False,
                  "test_metrics_computed": False, "fresh_process_resume": True,
                  "claim": "synthetic_fresh_process_resume_only_not_CUDA_evidence_when_device_is_cpu"}
        content = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _manifest(root, pending_files={"report.json": content})
        _write_bytes_once(root / "report.json", content)
        return result
    except BaseException as exc:
        _failure(root, progress, exc, schema=SCHEMA)
        raise


__all__ = ["prepare_training_resume_runtime", "run_fresh_process_gate", "run_resume_worker"]
