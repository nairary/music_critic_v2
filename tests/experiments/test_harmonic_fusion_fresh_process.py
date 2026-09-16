"""Explicit runtime readiness and actual new-process synthetic CPU resume."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_resume as resume
from music_critic.experiments.analysisgnn.contracts import fingerprint
from tests.experiments.test_harmonic_fusion_cuda_rng import _same


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts" / "run_harmonic_fusion_fresh_process_gate.py"


def _deny(*args, **kwargs):
    pytest.fail("unexpected CUDA runtime access or checkpoint state application")


def _native_environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_cpu_runtime_readiness_never_queries_or_initializes_cuda(monkeypatch):
    for name in ("is_available", "is_initialized", "device_count", "init", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, _deny)
    observed = resume.prepare_training_resume_runtime("cpu")
    assert observed["cuda_runtime_accessed"] is False
    assert observed["verified_cuda_ordinals"] == []
    assert observed["saved_rng_applied"] is False


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.0])
def test_cuda_expected_count_required_before_any_runtime_access(monkeypatch, count):
    monkeypatch.setattr(torch.cuda, "is_available", _deny)
    monkeypatch.setattr(torch.cuda, "init", _deny)
    with pytest.raises(resume.FusionResumeError, match="runtime_expected_cuda_count_required"):
        resume.prepare_training_resume_runtime("cuda", expected_cuda_device_count=count)


@pytest.mark.parametrize("available,count,category", [(False, 1, "runtime_cuda_unavailable"),
                                                      (True, 2, "runtime_cuda_inventory_mismatch")])
def test_missing_or_excess_cuda_devices_fail_before_init(monkeypatch, available, count, category):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "init", _deny)
    with pytest.raises(resume.FusionResumeError, match=category):
        resume.prepare_training_resume_runtime("cuda", expected_cuda_device_count=1)


@pytest.mark.parametrize("already_initialized", [False, True])
def test_explicit_cuda_initialization_verifies_all_ordinals_without_saved_state(monkeypatch,
                                                                             already_initialized):
    original_generator = torch.Generator
    runtime = {"initialized": already_initialized, "init_calls": 0}
    states = [torch.tensor([2, 4, 6], dtype=torch.uint8), torch.tensor([3, 5, 7], dtype=torch.uint8)]
    verified = []
    before_cpu = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone())

    def initialize():
        runtime["initialized"] = True
        runtime["init_calls"] += 1
        random.random(), np.random.random(), torch.rand(3)

    class FakeGenerator:
        def __init__(self, device):
            self.ordinal = int(device.split(":")[1])

        def set_state(self, value):
            assert torch.equal(value, states[self.ordinal])
            verified.append(self.ordinal)

    def generator(*args, **kwargs):
        device = str(kwargs.get("device", "cpu"))
        return FakeGenerator(device) if device.startswith("cuda:") else original_generator(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: runtime["initialized"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "init", initialize)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in states])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda actual: _same(actual, states))
    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    result = resume.prepare_training_resume_runtime("cuda", expected_cuda_device_count=2)
    assert result["cuda_initialized_by_call"] is (not already_initialized)
    assert result["verified_cuda_ordinals"] == verified == [0, 1]
    assert result["saved_rng_applied"] is False
    assert runtime == {"initialized": True, "init_calls": int(not already_initialized)}
    assert random.getstate() == before_cpu[0]
    now_np = np.random.get_state()
    assert now_np[0] == before_cpu[1][0] and np.array_equal(now_np[1], before_cpu[1][1])
    assert now_np[2:] == before_cpu[1][2:]
    assert torch.equal(torch.get_rng_state(), before_cpu[2])


def test_initialization_failure_never_reports_readiness(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", _deny)
    with pytest.raises(resume.FusionResumeError, match="runtime_cuda_initialization_failed"):
        resume.prepare_training_resume_runtime("cuda", expected_cuda_device_count=1)


def test_device_inventory_drift_during_initialization_is_rejected(monkeypatch):
    runtime = {"initialized": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if runtime["initialized"] else 1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: runtime["initialized"])
    monkeypatch.setattr(torch.cuda, "init", lambda: runtime.update(initialized=True))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", _deny)
    with pytest.raises(resume.FusionResumeError, match="runtime_cuda_inventory_mismatch"):
        resume.prepare_training_resume_runtime("cuda", expected_cuda_device_count=1)


def test_cpu_rejects_cuda_reservation_and_unknown_device():
    with pytest.raises(resume.FusionResumeError, match="runtime_cuda_count_not_applicable"):
        resume.prepare_training_resume_runtime("cpu", expected_cuda_device_count=1)
    with pytest.raises(resume.FusionResumeError, match="runtime_device_invalid"):
        resume.prepare_training_resume_runtime("mps")


def test_true_fresh_process_cpu_resume_for_both_arms(tmp_path):
    output = tmp_path / "new-process-gate"
    result = subprocess.run([sys.executable, str(ENTRYPOINT), "--smoke", "--device", "cpu",
                             "--output-root", str(output)], cwd=ROOT, env=_native_environment(),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is True and report["device"] == "cpu"
    assert report["fresh_process_resume"] is True
    assert report["environment_fingerprint"] == fingerprint(report["environment"])
    driver_pid = report["parent_process_id"]
    assert driver_pid != os.getpid()
    assert {row["arm"] for row in report["arms"]} == {"control", "fusion"}
    worker_pids = {row["worker_process_id"] for row in report["arms"]}
    assert len(worker_pids) == 2 and driver_pid not in worker_pids and os.getpid() not in worker_pids
    for row in report["arms"]:
        assert row["whole_state_equal"] is True
        assert row["worker_runtime_readiness"]["cuda_runtime_accessed"] is False
        child_root = output / f"{row['arm']}-fresh-process"
        child = json.loads((child_root / "report.json").read_text())
        child_manifest = (child_root / "artifact-manifest.json").read_bytes()
        assert sha256(child_manifest).hexdigest() == row["worker_artifact_manifest"]["sha256"]
        assert len(child_manifest) == row["worker_artifact_manifest"]["size_bytes"]
        assert child["initial_validation_updates"] == [0, 3]
        assert child["final_validation_updates"] == [0, 3, 6]
        assert child["environment_fingerprint"] == report["environment_fingerprint"]
        content = (child_root / "final.pt").read_bytes()
        assert sha256(content).hexdigest() == row["resumed_checkpoint"]["sha256"]
        assert len(content) == row["resumed_checkpoint"]["size_bytes"]
    assert report["progress"]["applied_updates"] == report["progress"]["attempted_updates"] == 24
    assert report["progress"]["applied_draws"] == report["progress"]["attempted_draws"] == 48
    assert report["progress"]["reader_calls"] == {"train": 48, "validation": 12}
    assert report["real_data_accessed"] is report["test_accessed"] is False
    assert report["actual_updates"] == 24 and report["actual_draws"] == 48
    assert all(report[key] is False for key in (
        "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed"))
    assert not (output / "failure-receipt.json").exists()


def test_worker_rejects_bad_checkpoint_without_training_or_success_publication(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    output = tmp_path / "failed-worker"
    result = subprocess.run([sys.executable, str(ENTRYPOINT), "--smoke", "--output-root", str(output),
        "--worker-checkpoint", str(checkpoint), "--checkpoint-sha256", "0" * 64,
        "--checkpoint-size-bytes", str(checkpoint.stat().st_size), "--arm", "control"],
        cwd=ROOT, env=_native_environment(), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "artifact.sha256_mismatch" in result.stderr
    receipt = json.loads((output / "failure-receipt.json").read_text())
    assert receipt["passed"] is False
    assert receipt["progress"]["attempted_updates"] == receipt["progress"]["applied_updates"] == 0
    assert receipt["progress"]["reader_calls"] == {"train": 0, "validation": 0}
    assert not (output / "report.json").exists() and not (output / "final.pt").exists()
    assert checkpoint.read_bytes() == b"not a checkpoint"


def test_cuda_cli_requires_registered_gate_before_runtime_or_output(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ENTRYPOINT), "--smoke", "--device", "cuda",
        "--expected-cuda-device-count", "1", "--output-root", str(output)], cwd=ROOT,
        env=_native_environment(), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "planned --experiment-record" in result.stderr
    assert not output.exists()
