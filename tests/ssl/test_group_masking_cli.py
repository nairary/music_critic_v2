"""Direct CLI contract and pre-execution gates; synthetic CPU only."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import torch

import music_critic.ssl.group_masking_cli as cli
from music_critic.ssl.group_masking_cli import main
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ssl_group/random_vs_beat.json"


def test_direct_cpu_fixture_cli(tmp_path):
    output = tmp_path / "result"
    command = [sys.executable, "scripts/run_group_masking_pilot.py", "--config", str(CONFIG),
               "--mode", "cpu-fixture", "--output", str(output)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "result.json").read_text())
    execution = [json.loads(line) for line in (output / "execution.jsonl").read_text().splitlines()]
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert report["status"] == "completed" and report["quality_evidence"] is False
    assert report["TEST_access"] is False and report["VALIDATION_access"] is False
    assert report["device"] == "cpu" and report["peak_allocated_bytes"] is None
    assert len(execution) == report["actual_updates_including_resume_replays"] == 8
    assert report["actual_encoder_forwards_including_resume_replays"] == 24
    assert report["actual_ssl_draws_including_resume_replays"] == 16
    assert report["actual_theory_draws_including_resume_replays"] == 16
    assert report["attempted_updates"] == report["observed_completed_updates"] == 8
    assert report["observed_encoder_forwards"] == 24
    assert report["optimizer_outcomes"] == {"applied": 8, "not_applied": 0, "unknown": 0}
    by_attempt = {}
    for observation in attempts:
        by_attempt.setdefault(observation["execution_attempt_id"], []).append(observation)
    assert set(by_attempt) == {observation["execution_attempt_id"] for observation in execution}
    assert len(by_attempt) == 8
    for events in by_attempt.values():
        first, final = events[0]["receipt"], events[-1]["receipt"]
        assert first["status"] == "started"
        assert first["observed_encoder_forwards"] == 0
        assert first["optimizer_outcome"] == "not_applied"
        assert final["status"] == "completed"
        assert final["observed_encoder_forwards"] == 3
        assert final["optimizer_outcome"] == "applied"
        assert final["confirmed_updates"] - final["completed_updates_before"] == 1
    plan = json.loads((output / "plan.json").read_text())
    assert plan["wall_budget"] == {"seconds": 600, "enforcement": "pre_step_elapsed_time_check",
                                   "hard_timeout": "external_coordinator_resource_wrapper_required"}
    assert report["source"]["kind"] == "synthetic"
    starts = {arm["initial_fingerprint"] for arm in report["arms"].values()}
    assert len(starts) == 1 and all(arm["resume_exact"] for arm in report["arms"].values())


@pytest.mark.parametrize("mode,source,reason", [("cuda-fixture", False, "requires_bound_C2"),
    ("cuda-fixture", True, "planned_record_required"), ("cpu-fixture", True, "planned_record_required")])
def test_checkpoint_cuda_preflight_before_output_or_loader(tmp_path, mode, source, reason):
    output = tmp_path / "must-not-exist"
    args = ["--config", str(CONFIG), "--mode", mode, "--output", str(output)]
    if source:
        args += ["--source-checkpoint", str(tmp_path / "nonexistent.pt")]
    with pytest.raises(ValueError, match=reason):
        main(args)
    assert not output.exists()


def _cpu_args(output):
    return ["--config", str(CONFIG), "--mode", "cpu-fixture", "--output", str(output)]


def test_interrupted_adam_retains_started_receipt_and_unknown_outcome(tmp_path, monkeypatch):
    output = tmp_path / "interrupted-adam"
    original = torch.optim.Adam.step
    applied = []

    def step_then_interrupt(optimizer, *args, **kwargs):
        # The started and pre-Adam receipts must be on disk before Adam runs.
        before = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
        assert before[0]["receipt"]["status"] == "started"
        assert before[-1]["receipt"]["optimizer_outcome"] == "unknown"
        assert before[-1]["receipt"]["observed_encoder_forwards"] == 3
        result = original(optimizer, *args, **kwargs)
        applied.append(result)
        raise KeyboardInterrupt("injected after actual Adam update")

    monkeypatch.setattr(torch.optim.Adam, "step", step_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after actual Adam"):
        main(_cpu_args(output))
    assert len(applied) == 1
    failure = json.loads((output / "failure.json").read_text())
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    assert failure["type"] == "KeyboardInterrupt"
    assert failure["attempted_updates"] == 1
    assert failure["observed_completed_updates"] == failure["recorded_successful_updates"] == 0
    assert failure["observed_encoder_forwards"] == 3
    assert failure["optimizer_outcomes"] == {"applied": 0, "not_applied": 0, "unknown": 1}
    assert attempts[-1] == failure["last_attempt"]
    receipt = failure["last_attempt"]["receipt"]
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "optimizer_step"
    assert receipt["optimizer_outcome"] == "unknown"
    assert receipt["error_type"] == "KeyboardInterrupt"
    assert (output / "execution.jsonl").read_text() == ""
    assert not (output / "result.json").exists()


def test_failure_before_forward_reports_no_applied_optimizer(tmp_path, monkeypatch):
    output = tmp_path / "failed-forward"

    def fail_forward(*args, **kwargs):
        raise RuntimeError("injected before encoder forward")

    monkeypatch.setattr(TheoryRetentionHarness, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="before encoder forward"):
        main(_cpu_args(output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["observed_completed_updates"] == failure["observed_encoder_forwards"] == 0
    assert failure["optimizer_outcomes"] == {"applied": 0, "not_applied": 1, "unknown": 0}
    assert failure["last_attempt"]["receipt"]["status"] == "failed"


def test_post_step_validation_failure_reports_applied_but_unconfirmed(tmp_path, monkeypatch):
    output = tmp_path / "failed-post-step-validation"
    original = torch.optim.Adam.step

    def corrupt_slot_after_step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        next(iter(optimizer.state.values()))["exp_avg_sq"].fill_(float("inf"))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", corrupt_slot_after_step)
    with pytest.raises(ValueError, match="nonfinite_optimizer_state"):
        main(_cpu_args(output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["observed_completed_updates"] == failure["recorded_successful_updates"] == 0
    assert failure["observed_encoder_forwards"] == 3
    assert failure["optimizer_outcomes"] == {"applied": 1, "not_applied": 0, "unknown": 0}
    assert failure["last_attempt"]["receipt"]["stage"] == "post_step_validation"


def test_receipt_failure_after_confirmation_preserves_confirmed_update(tmp_path, monkeypatch):
    output = tmp_path / "failed-completion-receipt"
    original = cli._jsonl_append_durable
    injected = []

    def fail_completion_receipt(path, value):
        if (Path(path).name == "attempts.jsonl" and value["receipt"]["status"] == "completed"
                and not injected):
            injected.append(True)
            raise OSError("injected completion receipt write failure")
        return original(path, value)

    monkeypatch.setattr(cli, "_jsonl_append_durable", fail_completion_receipt)
    with pytest.raises(OSError, match="completion receipt"):
        main(_cpu_args(output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["observed_completed_updates"] == 1
    assert failure["recorded_successful_updates"] == 0
    assert failure["observed_encoder_forwards"] == 3
    assert failure["optimizer_outcomes"] == {"applied": 1, "not_applied": 0, "unknown": 0}
    assert failure["last_attempt"]["receipt"]["confirmed_updates"] == 1


def test_successful_row_write_failure_keeps_confirmed_count_separate(tmp_path, monkeypatch):
    output = tmp_path / "failed-execution-write"
    original = cli._jsonl_append_durable

    def fail_execution_write(path, value):
        if Path(path).name == "execution.jsonl":
            raise OSError("injected execution row write failure")
        return original(path, value)

    monkeypatch.setattr(cli, "_jsonl_append_durable", fail_execution_write)
    with pytest.raises(OSError, match="execution row"):
        main(_cpu_args(output))
    failure = json.loads((output / "failure.json").read_text())
    assert failure["observed_completed_updates"] == 1
    assert failure["recorded_successful_updates"] == 0
    assert failure["optimizer_outcomes"] == {"applied": 1, "not_applied": 0, "unknown": 0}
    assert failure["last_attempt"]["receipt"]["status"] == "completed"
    assert (output / "execution.jsonl").read_text() == ""
