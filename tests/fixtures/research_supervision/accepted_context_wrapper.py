#!/usr/bin/env python3
"""External Linux resource supervisor for the registered synthetic context suite.

Uses only the standard library. The coordinator must own an exclusive GPU slot.
The driver-memory check is sampled whole-device usage, not a hard CUDA allocation
limit or Torch's allocated-memory metric. The runner's durable progress journal
is retained even when terminating its process group prevents a Python receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


MAX_JOURNAL_BYTES = 1024 * 1024  # Covers the runner's 200 x (4096 + newline) cap.
POLL_SECONDS = 0.25
QUERY_TIMEOUT_SECONDS = 1.0
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


class ResourceStop(Exception):
    """An external limit or monitor failure stopped the child."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def query_gpu_memory(gpu_uuid: str, executable: str, timeout: float) -> int:
    """Return sampled whole-device MiB, refusing missing or ambiguous evidence."""
    try:
        result = subprocess.run(
            [executable, f"--id={gpu_uuid}", "--query-gpu=uuid,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceStop("gpu_monitor_failed", f"{type(exc).__name__}: {exc}") from exc
    rows = result.stdout.strip().splitlines()
    values = rows[0].split(",") if len(rows) == 1 else []
    if (len(values) != 2 or values[0].strip() != gpu_uuid
            or not re.fullmatch(r"[0-9]+", values[1].strip())):
        raise ResourceStop("gpu_monitor_failed", "GPU identity or integer memory.used unavailable")
    return int(values[1].strip())


def recover_progress(path: Path) -> dict:
    """Recover only the complete, valid, monotonic JSONL prefix; never infer work."""
    result = {
        "path": str(path), "sha256": None, "size_bytes": None,
        "journal_partial": False, "valid_prefix_records": 0,
        "last_valid_record": None, "accounting": None,
        "scope": "last durable valid prefix; completed is a lower bound; incomplete attempts remain unknown",
        "issue": None,
    }
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JOURNAL_BYTES + 1)
        result["size_bytes"] = path.stat().st_size
    except OSError as exc:
        result.update(journal_partial=True, issue=f"{type(exc).__name__}: {exc}")
        return result
    if len(raw) > MAX_JOURNAL_BYTES:
        result.update(journal_partial=True, issue="journal_exceeds_bounded_recovery")
        return result
    result["sha256"] = sha256(raw).hexdigest()
    if not raw:
        result.update(journal_partial=True, issue="empty_journal_no_accounting")
        return result
    lines = raw.split(b"\n")
    if lines[-1]:
        result.update(journal_partial=True, issue="incomplete_final_line")
    lines.pop()  # A record is durable evidence only after its complete newline.
    previous = None
    counters = ("forwards_started", "forwards_completed", "forwards_completion_unknown",
                "views_recorded", "applied_updates", "budget", "sequence")
    for line in lines:
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != "ContextSuiteProgress@1.0.0":
                raise ValueError("schema")
            if any(type(row.get(key)) is not int for key in counters):
                raise ValueError("noninteger_counter")
            started, completed = row["forwards_started"], row["forwards_completed"]
            if not (row["budget"] == 20 and 0 <= completed <= started <= 20
                    and row["forwards_completion_unknown"] == started - completed
                    and 0 <= row["views_recorded"] <= completed and row["applied_updates"] == 0
                    and isinstance(row.get("event"), str) and isinstance(row.get("stage"), str)
                    and all(key in row for key in ("pair_id", "side", "view"))):
                raise ValueError("accounting_invariant")
            if previous is None:
                if row["sequence"] not in (0, 1):
                    raise ValueError("initial_sequence")
            elif (row["sequence"] != previous["sequence"] + 1
                  or any(row[key] < previous[key] for key in
                         ("forwards_started", "forwards_completed", "views_recorded"))):
                raise ValueError("nonmonotonic_prefix")
        except (ValueError, TypeError, UnicodeError) as exc:
            result.update(journal_partial=True, issue=f"invalid_record_after_prefix: {exc}")
            break
        previous = row
        result["valid_prefix_records"] += 1
    if previous is not None:
        result["last_valid_record"] = previous
        result["accounting"] = {key: previous[key] for key in (
            "forwards_started", "forwards_completed", "forwards_completion_unknown",
            "views_recorded", "applied_updates", "budget",
            "stage", "pair_id", "side", "view",
        )}
    return result


def _require_ownership_context() -> None:
    # A foreign reaper could invalidate the waitid -> killpg ownership interval.
    # This supervisor is a standalone, single-threaded process with no SIGCHLD
    # callback; using it in an embedding with its own child reaper is unsupported.
    if (threading.current_thread() is not threading.main_thread()
            or threading.active_count() != 1
            or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
        raise ResourceStop("child_ownership_context_unavailable",
                           "one main thread and default SIGCHLD disposition are required")


def _require_nonreaping_runtime() -> None:
    names = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT",
             "CLD_EXITED", "CLD_KILLED", "CLD_DUMPED")
    if (not sys.platform.startswith("linux") or not callable(getattr(os, "waitid", None))
            or any(not hasattr(os, name) for name in names)):
        raise ResourceStop("nonreaping_observer_unavailable",
                           "Linux with native os.waitid/WNOWAIT is required; no fallback")
    _require_ownership_context()


def _observe_child_exit(child: subprocess.Popen) -> int | None:
    """Observe our unreaped leader; its PID cannot be reused while retained."""
    _require_ownership_context()
    if child.returncode is not None:
        raise ResourceStop("child_ownership_lost", "leader was already reaped or returncode was modified")
    try:
        info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except OSError as exc:
        raise ResourceStop("child_ownership_lost", f"non-reaping child observation failed: {exc}") from exc
    if info is None:
        return None
    if info.si_pid != child.pid:
        raise ResourceStop("child_ownership_lost", "waitid returned a different child identity")
    if info.si_code == os.CLD_EXITED:
        return info.si_status
    if info.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
        return -info.si_status
    raise ResourceStop("child_ownership_lost", "waitid returned an unsupported child status")


def _cleanup_owned_group(child: subprocess.Popen, lifecycle: dict) -> int:
    """Signal exactly once while PID ownership is held, then perform final reap."""
    _observe_child_exit(child)  # Revalidate immediately before numeric group use.
    try:
        if os.getpgid(child.pid) != child.pid:
            raise ResourceStop("child_ownership_lost", "leader no longer owns its original process group")
    except OSError as exc:
        raise ResourceStop("child_ownership_lost", f"owned leader/group unavailable: {exc}") from exc
    lifecycle["ownership_confirmed_before_cleanup"] = True
    lifecycle["group_cleanup_attempted"] = True
    try:
        os.killpg(child.pid, signal.SIGKILL)
        lifecycle["group_signal_sent"] = True
    except ProcessLookupError:
        # Still-unreaped leader identity was validated above. No group remains.
        lifecycle["group_already_absent"] = True
    # This is the first and final consuming wait. There is no numeric signal
    # operation after it, including error/timeout paths and normal leader exit.
    try:
        returncode = child.wait(timeout=2)
    except subprocess.TimeoutExpired as exc:
        raise ResourceStop("termination_unconfirmed", "group cleanup requested; final reap exceeded 2 seconds") from exc
    lifecycle["leader_reaped_after_cleanup"] = True
    return returncode


def _finite_bound(value: float, upper: float, label: str) -> None:
    if not math.isfinite(value) or not 0 < value <= upper:
        raise ValueError(f"{label} must be positive and at most {upper}")


def supervise(*, command: list[str], gpu_uuid: str, progress: Path, receipt: Path,
              stdout: Path, stderr: Path, wall_seconds: float = 600,
              memory_mib: int = 4096, nvidia_smi: str = "nvidia-smi",
              poll_seconds: float = POLL_SECONDS) -> int:
    """Run one fresh queue item; return nonzero for any failure or forced stop."""
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise ValueError("POSIX process-group supervision is required")
    if not command or not GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError("an explicit command and full physical GPU UUID are required")
    _finite_bound(wall_seconds, 600, "wall_seconds")
    _finite_bound(poll_seconds, POLL_SECONDS, "poll_seconds")
    if type(memory_mib) is not int or not 0 < memory_mib <= 4096:
        raise ValueError("memory_mib must be an integer in [1, 4096]")
    paths = [path.resolve() for path in (progress, receipt, stdout, stderr)]
    if len(set(paths)) != len(paths):
        raise ValueError("progress, receipt, stdout and stderr must be distinct paths")
    if any(os.path.lexists(path) for path in paths):
        raise FileExistsError("all four evidence paths must be fresh")
    progress, receipt, stdout, stderr = paths
    # Exclusive opens prevent clobbering an earlier attempt even after precheck.
    with receipt.open("x", encoding="utf-8") as receipt_stream, stdout.open("xb") as out, stderr.open("xb") as err:
        start = time.monotonic()
        deadline = start + wall_seconds
        child = None
        child_started = None
        samples = 0
        peak_mib = None
        returncode = None
        received_signal = None
        status, reason, detail = "failed", "supervisor_failed", None
        old_handlers = {}
        lifecycle = {"strategy": "Linux waitid WNOWAIT; group cleanup before final reap",
                     "ownership_lost": False,
                     "ownership_confirmed_before_cleanup": False,
                     "group_cleanup_attempted": False, "group_signal_sent": False,
                     "group_already_absent": False, "leader_reaped_after_cleanup": False}

        def remember_signal(signum, _frame):
            nonlocal received_signal
            received_signal = signum

        def sample():
            nonlocal samples, peak_mib
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResourceStop("wall_timeout", "external wall deadline reached")
            try:
                used = query_gpu_memory(gpu_uuid, nvidia_smi, min(QUERY_TIMEOUT_SECONDS, remaining))
            except ResourceStop:
                if time.monotonic() >= deadline:
                    raise ResourceStop("wall_timeout", "external wall deadline reached during monitor query") from None
                raise
            samples += 1
            peak_mib = used if peak_mib is None else max(peak_mib, used)
            if time.monotonic() >= deadline:
                raise ResourceStop("wall_timeout", "external wall deadline reached during monitor query")
            if used > memory_mib:
                raise ResourceStop("gpu_memory_ceiling", f"sampled {used} MiB exceeds {memory_mib} MiB")

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            _require_nonreaping_runtime()
            # Reinstall default to clear inherited autoreap disposition/flags.
            # Ignore/custom handlers are rejected before any child is launched.
            old_handlers[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, remember_signal)
            sample()  # Fail closed before the child or checkpoint can be read.
            if received_signal is not None:
                raise ResourceStop("supervisor_signal", f"received signal {received_signal}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            child_started = time.monotonic()
            while True:
                if received_signal is not None:
                    raise ResourceStop("supervisor_signal", f"received signal {received_signal}")
                if time.monotonic() >= deadline:
                    raise ResourceStop("wall_timeout", "external wall deadline reached")
                returncode = _observe_child_exit(child)
                if returncode is not None:
                    status = "completed" if returncode == 0 else "failed"
                    reason = "child_exit"
                    break
                sample()
                time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
        except ResourceStop as exc:
            reason, detail = exc.reason, str(exc)
            lifecycle["ownership_lost"] = reason in ("child_ownership_lost", "child_ownership_context_unavailable")
            status = "aborted" if reason in ("wall_timeout", "gpu_memory_ceiling", "supervisor_signal") else "failed"
        except (Exception, KeyboardInterrupt) as exc:
            reason = "supervisor_interrupted" if isinstance(exc, KeyboardInterrupt) else "supervisor_failed"
            status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
            detail = f"{type(exc).__name__}: {exc}"
        finally:
            # SIGKILL deliberately avoids extra CUDA work/grace past the deadline.
            # Recovery uses durable journal evidence instead of requiring Python
            # exception cleanup. Detached descendants are outside this contract.
            if child is not None and not lifecycle["ownership_lost"]:
                try:
                    returncode = _cleanup_owned_group(child, lifecycle)
                except ResourceStop as exc:
                    status, reason, detail = "failed", exc.reason, str(exc)
                    lifecycle["ownership_lost"] = exc.reason in ("child_ownership_lost", "child_ownership_context_unavailable")
                    returncode = None
                except Exception as exc:
                    status, reason, detail = "failed", "cleanup_failed", f"{type(exc).__name__}: {exc}"
                    returncode = None
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            ended = time.monotonic()
            progress_evidence = recover_progress(progress)
            accounting = progress_evidence["accounting"]
            if status == "completed" and (
                    progress_evidence["journal_partial"] or accounting is None
                    or any(accounting[key] != 20 for key in
                           ("forwards_started", "forwards_completed", "views_recorded"))
                    or progress_evidence["last_valid_record"]["event"] != "result_written"):
                status, reason = "failed", "progress_incomplete"
                detail = "child exited zero without a durable result_written journal and all 20 forwards/views"
            payload = {
                "schema": "ContextSuiteResourceReceipt@1.0.0",
                "status": status, "termination_reason": reason, "detail": detail,
                "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
                "command": command, "working_directory": str(Path.cwd()),
                "wrapper_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "gpu_uuid": gpu_uuid, "cuda_visible_devices": gpu_uuid,
                "cpu_thread_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "operator_wall_limit_seconds": wall_seconds,
                "controller_wall_seconds": ended - start,
                "child_wall_seconds": ended - child_started if child_started is not None else None,
                "child_pid": child.pid if child is not None else None,
                "child_returncode": returncode,
                "child_lifecycle": lifecycle,
                "gpu_driver_memory_ceiling_mib": memory_mib,
                "gpu_driver_memory_peak_sampled_mib": peak_mib,
                "gpu_memory_samples": samples,
                "monitor_poll_seconds": poll_seconds,
                "monitor_query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
                "gpu_memory_scope": "sampled whole-device NVIDIA memory.used; includes driver/reserved memory; not Torch allocated or a hard allocation cap",
                "limit_scope": "coordinator-exclusive physical GPU; owned unreaped leader required for group SIGKILL; OS scheduling/termination may overrun; no detached descendants or foreign reapers",
                "stdout_path": str(stdout), "stderr_path": str(stderr),
                "progress": progress_evidence,
            }
            json.dump(payload, receipt_stream, sort_keys=True, indent=2, allow_nan=False)
            receipt_stream.write("\n")
            receipt_stream.flush()
            os.fsync(receipt_stream.fileno())
    return 0 if status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(command=command, gpu_uuid=args.gpu_uuid, progress=args.progress,
                     receipt=args.receipt, stdout=args.stdout, stderr=args.stderr,
                     wall_seconds=args.wall_seconds, memory_mib=args.memory_mib,
                     nvidia_smi=args.nvidia_smi)


if __name__ == "__main__":
    raise SystemExit(main())
