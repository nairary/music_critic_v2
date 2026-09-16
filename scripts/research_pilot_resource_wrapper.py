"""External resource supervision; zero exit is process evidence, not scientific acceptance.

Reuse the exact native Linux ownership implementation already accepted by
EXP-CTX-C2-LINUX-001. Task-specific durable journals remain in the task output
directory and must be validated independently before accepting an experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

OWNERSHIP_SHA256 = "65416b6f35a227cceebab10ec0bcf99699e35dff590bc9652eb3ac41c40e878b"


def load_ownership(path: Path):
    if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != OWNERSHIP_SHA256:
        raise ValueError("unaccepted ownership implementation")
    spec = importlib.util.spec_from_file_location("accepted_context_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_memory(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            number, unit = value.split()
            if unit != "kB" or int(number) < 0 or name in values:
                raise ValueError("invalid host memory evidence")
            values[name] = int(number)
    if set(values) != {"MemAvailable", "SwapTotal", "SwapFree"}:
        raise ValueError("incomplete host memory evidence")
    if values["SwapFree"] > values["SwapTotal"]:
        raise ValueError("invalid swap evidence")
    return {"available_kib": values["MemAvailable"],
            "swap_used_kib": values["SwapTotal"] - values["SwapFree"]}


def supervise(*, ownership, command: list[str], gpu_uuid: str, receipt: Path,
              stdout: Path, stderr: Path, wall_seconds: float, memory_mib: int,
              minimum_available_mib: int = 8192, max_swap_growth_mib: int = 256,
              poll_seconds: float = 0.5, nvidia_smi: str = "nvidia-smi") -> int:
    if (not command or not ownership.GPU_UUID.fullmatch(gpu_uuid)
            or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 86400
            or type(memory_mib) is not int or not 0 < memory_mib <= 16384
            or type(minimum_available_mib) is not int or minimum_available_mib < 8192
            or type(max_swap_growth_mib) is not int or not 0 <= max_swap_growth_mib <= 256
            or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 1):
        raise ValueError("invalid explicit resource bounds")
    paths = tuple(p.resolve() for p in (receipt, stdout, stderr))
    if len(set(paths)) != 3 or any(os.path.lexists(p) for p in paths):
        raise FileExistsError("three distinct new evidence paths are required")
    receipt, stdout, stderr = paths
    ownership._require_nonreaping_runtime()
    lifecycle = {"strategy": "accepted Linux waitid WNOWAIT; cleanup before reap",
                 "ownership_lost": False, "ownership_confirmed_before_cleanup": False,
                 "group_cleanup_attempted": False, "group_signal_sent": False,
                 "group_already_absent": False, "leader_reaped_after_cleanup": False}
    start = time.monotonic()
    deadline = start + wall_seconds
    child = None
    returncode = None
    process_status, reason, detail = "failed", "supervisor_failed", None
    received_signal = None
    handlers = {}
    samples, gpu_peak, min_available, initial_swap, peak_swap = 0, None, None, None, None

    def remember_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum

    def sample():
        nonlocal samples, gpu_peak, min_available, initial_swap, peak_swap
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ownership.ResourceStop("wall_timeout", "external deadline")
        used = ownership.query_gpu_memory(gpu_uuid, nvidia_smi, min(1.0, remaining))
        mem = host_memory()
        samples += 1
        gpu_peak = used if gpu_peak is None else max(gpu_peak, used)
        min_available = mem["available_kib"] if min_available is None else min(min_available, mem["available_kib"])
        initial_swap = mem["swap_used_kib"] if initial_swap is None else initial_swap
        peak_swap = mem["swap_used_kib"] if peak_swap is None else max(peak_swap, mem["swap_used_kib"])
        if time.monotonic() >= deadline:
            raise ownership.ResourceStop("wall_timeout", "external deadline during monitor query")
        if used > memory_mib:
            raise ownership.ResourceStop("gpu_memory_ceiling", f"sampled {used} MiB")
        if mem["available_kib"] < minimum_available_mib * 1024:
            raise ownership.ResourceStop("host_memory_floor", "insufficient available host memory")
        if mem["swap_used_kib"] - initial_swap > max_swap_growth_mib * 1024:
            raise ownership.ResourceStop("swap_growth", "host swap growth exceeded bound")

    with receipt.open("x") as record, stdout.open("xb") as out, stderr.open("xb") as err:
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            handlers[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, remember_signal)
            sample()
            if received_signal is not None:
                raise ownership.ResourceStop("supervisor_signal", str(received_signal))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            child = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            while True:
                if received_signal is not None:
                    raise ownership.ResourceStop("supervisor_signal", str(received_signal))
                if time.monotonic() >= deadline:
                    raise ownership.ResourceStop("wall_timeout", "external deadline")
                returncode = ownership._observe_child_exit(child)
                if returncode is not None:
                    process_status = "exited_zero" if returncode == 0 else "failed"
                    reason = "child_exit"
                    break
                sample()
                time.sleep(min(poll_seconds, max(0, deadline-time.monotonic())))
        except ownership.ResourceStop as exc:
            reason, detail = exc.reason, str(exc)
            lifecycle["ownership_lost"] = reason in {"child_ownership_lost", "child_ownership_context_unavailable"}
            process_status = "aborted" if reason in {"wall_timeout", "gpu_memory_ceiling", "host_memory_floor", "swap_growth", "supervisor_signal"} else "failed"
        except (Exception, KeyboardInterrupt) as exc:
            process_status = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
            reason, detail = "supervisor_exception", f"{type(exc).__name__}: {exc}"
        finally:
            if child is not None and not lifecycle["ownership_lost"]:
                try:
                    returncode = ownership._cleanup_owned_group(child, lifecycle)
                except ownership.ResourceStop as exc:
                    process_status, reason, detail = "failed", exc.reason, str(exc)
                    lifecycle["ownership_lost"] = exc.reason in {"child_ownership_lost", "child_ownership_context_unavailable"}
                    returncode = None
                except Exception as exc:
                    process_status, reason, detail = "failed", "cleanup_failed", str(exc)
                    returncode = None
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            payload = {"schema": "ResearchPilotResourceReceipt@1.0.0", "process_status": process_status,
                       "experiment_acceptance": "requires_independent_task_artifact_validation",
                       "reason": reason, "detail": detail, "command": command, "cwd": str(Path.cwd()),
                       "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
                       "wall_seconds": time.monotonic()-start, "wall_limit_seconds": wall_seconds,
                       "child_returncode": returncode, "child_pid": child.pid if child else None,
                       "ownership_source_sha256": OWNERSHIP_SHA256, "child_lifecycle": lifecycle,
                       "wrapper_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "gpu_uuid": gpu_uuid, "gpu_memory_ceiling_mib": memory_mib,
                       "sampled_gpu_memory_peak_mib": gpu_peak, "sampled_host_available_min_kib": min_available,
                       "host_available_floor_mib": minimum_available_mib, "initial_swap_used_kib": initial_swap,
                       "sampled_peak_swap_used_kib": peak_swap, "max_swap_growth_mib": max_swap_growth_mib,
                       "resource_samples": samples, "poll_seconds": poll_seconds,
                       "limits_scope": "sampled whole-device GPU and host memory; OS scheduling can overrun; ordinary inherited-group descendants only",
                       "task_accounting": "read durable task journals; missing completion is unknown, never zero work"}
            json.dump(payload, record, sort_keys=True, indent=2, allow_nan=False)
            record.write("\n"); record.flush(); os.fsync(record.fileno())
    return 0 if process_status == "exited_zero" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ownership-wrapper", type=Path, required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--wall-seconds", type=float, required=True)
    p.add_argument("--memory-mib", type=int, required=True)
    for name in ("receipt", "stdout", "stderr"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    return supervise(ownership=load_ownership(a.ownership_wrapper), command=command,
                     gpu_uuid=a.gpu_uuid, wall_seconds=a.wall_seconds, memory_mib=a.memory_mib,
                     receipt=a.receipt, stdout=a.stdout, stderr=a.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
