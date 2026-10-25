"""Request the existing trainer's graceful stop at an announced validation step.

An external operator control: it never edits frozen trainer source/configs or
signals the supervisor. The trainer finishes the current update's validation
and checkpoint before honoring SIGTERM. Existing cloud export/VM stop follows.
Linux pidfds prevent signaling a reused PID. Only standard-library modules used.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import time


def require(value, reason):
    if not value:
        raise ValueError("stop_at_update." + reason)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def recent_events(path):
    """Ignore an unfinished last JSONL write; retain complete trailing events."""
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        data = stream.read()
    lines = data.split(b"\n")
    if size > 65536:
        lines = lines[1:]
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def trigger(events, target):
    require(type(target) is int and 0 < target <= 80000 and target % 5000 == 0,
            "validation_boundary_required")
    updates = [r["update"] for r in events
               if r.get("event") in ("training", "validation_progress", "validation")]
    require(all(type(u) is int for u in updates), "invalid_update")
    require(not updates or max(updates) <= target, "requested_boundary_already_passed")
    return target in updates


def validate_identity(actual_command, expected_command, actual_cwd, expected_cwd):
    require(actual_command == expected_command, "trainer_command_changed")
    require(Path(actual_cwd) == Path(expected_cwd), "trainer_cwd_changed")
    require("music_critic.experiments.analysisgnn.temporal_slices.campaign" in actual_command,
            "campaign_required")


def verify_result(out, target):
    report = read(out / "report.json")
    require(report.get("final_update") == target and report.get("interrupted") is True,
            "terminal_update_mismatch")
    rows = [r for r in report.get("validations", []) if r.get("update") == target]
    require(len(rows) == 1, "validation_missing")
    latest = read(out / "latest-checkpoint.json")
    require(latest.get("update") == target, "checkpoint_update_mismatch")
    checkpoint = Path(latest["path"])
    require(checkpoint.parent == out and checkpoint.is_file() and not checkpoint.is_symlink(),
            "checkpoint_path")
    h = sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    require(h.hexdigest() == latest["sha256"] and checkpoint.stat().st_size == latest["size_bytes"],
            "checkpoint_digest")
    return dict(final_update=target, validation=rows[0], checkpoint=latest,
                trainer_report_sha256=sha256((out / "report.json").read_bytes()).hexdigest())


def run(base, target, *, hours=8):
    base = Path(base).resolve(strict=True)
    require(base.parent == Path("/srv/music-critic/experiments/c3_pair_20260917")
            and base.name in ("control", "attention"), "owned_arm_required")
    out = base / "evidence/train/run"
    receipt = base / "evidence/stop-at-60k"
    receipt.mkdir(exist_ok=False)
    cfg = read(base / "evidence/launch/train-supervisor.json")
    started = read(base / "evidence/train/supervisor/started.json")
    pid = started["child_pid"]
    require(type(pid) is int and pid > 1, "trainer_pid")
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "pidfd_required")
    fd = os.pidfd_open(pid)
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        validate_identity(command, cfg["command"], os.readlink(f"/proc/{pid}/cwd"), cfg["cwd"])
        # Refuse late installation; the target must still be strictly in the future.
        events = recent_events(out / "events.jsonl")
        require(not trigger(events, target), "install_before_boundary")
        write(receipt / "armed.json", dict(target_update=target, child_pid=pid,
            command=command, time=time.time(), signal="SIGTERM to trainer only",
            validation_finishes_before_exit=True,
            source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
        print("ARMED", base.name, target, flush=True)
        deadline = time.monotonic() + hours * 3600
        while not trigger(recent_events(out / "events.jsonl"), target):
            require(time.monotonic() < deadline, "watch_timeout")
            signal.pidfd_send_signal(fd, 0)
            time.sleep(0.25)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        write(receipt / "signal-sent.json", dict(target_update=target, child_pid=pid,
              time=time.time(), signal="SIGTERM", training_source_changed=False))
        # The existing supervisor remains alive to finalize/export/stop the VM.
        deadline = time.monotonic() + 1800
        while not (out / "report.json").exists():
            require(time.monotonic() < deadline, "terminal_report_timeout")
            time.sleep(0.25)
        result = verify_result(out, target)
        write(receipt / "verified.json", dict(time=time.time(), **result))
        print("VERIFIED", base.name, target, flush=True)
    except Exception as exc:
        write(receipt / "failure.json", dict(time=time.time(), type=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target-update", type=int, required=True)
    args = parser.parse_args()
    run(args.base, args.target_update)
