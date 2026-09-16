"""Synthetic resource-controller mechanics: never query a physical GPU."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/context_diagnostics_resource_wrapper.py"
SPEC = importlib.util.spec_from_file_location("context_resource_wrapper", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
NATIVE_WAITID = sys.platform.startswith("linux") and callable(getattr(os, "waitid", None))
requires_native_waitid = pytest.mark.skipif(
    not NATIVE_WAITID, reason="requires Linux native waitid/WNOWAIT; local mocks do not establish kernel acceptance")


def _row(sequence=1, *, started=3, completed=2, **changes):
    return dict(schema="ContextSuiteProgress@1.0.0", sequence=sequence,
                event="forward_started", stage="predict", pair_id="pair-1",
                side="right", view="full", forwards_started=started,
                forwards_completed=completed, forwards_completion_unknown=started-completed,
                views_recorded=completed, applied_updates=0, budget=20, **changes)


def _paths(tmp_path):
    return {name: tmp_path / f"{name}.jsonl" for name in ("progress", "receipt", "stdout", "stderr")}


def _child(paths, *, sleep=0, code=0, complete=None):
    if complete is None:
        complete = sleep == 0 and code == 0
    record = _row(started=20, completed=20) if complete else _row()
    if complete:
        record.update(event="result_written", stage="result_written")
    row = json.dumps(record) + "\n"
    # Fixture-only process. Durable evidence precedes its simulated unfinished work.
    return [sys.executable, "-c",
            f"import os,time; f=open({str(paths['progress'])!r},'x'); "
            f"f.write({row!r}); f.flush(); os.fsync(f.fileno()); "
            f"print(os.environ['CUDA_VISIBLE_DEVICES']); time.sleep({sleep!r}); raise SystemExit({code})"]


@pytest.mark.parametrize("raw", ["", "N/A", f"{UUID}, N/A", f"{UUID}, -1",
                               "GPU-wrong, 12", f"{UUID}, 12\n{UUID}, 13"])
def test_gpu_monitor_rejects_missing_or_ambiguous_evidence(monkeypatch, raw):
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, raw, ""))
    with pytest.raises(wrapper.ResourceStop, match="identity or integer"):
        wrapper.query_gpu_memory(UUID, "unused", 1)


def test_gpu_query_binds_uuid_and_timeout(monkeypatch):
    def fake(args, **kwargs):
        assert args == ["fake-smi", f"--id={UUID}", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]
        assert kwargs["timeout"] == 0.125
        return subprocess.CompletedProcess(args, 0, f"{UUID}, 4096\n", "")
    monkeypatch.setattr(wrapper.subprocess, "run", fake)
    assert wrapper.query_gpu_memory(UUID, "fake-smi", 0.125) == 4096


@pytest.mark.parametrize("error", [FileNotFoundError("fake missing"), subprocess.TimeoutExpired("fake", 1),
                                  subprocess.CalledProcessError(1, "fake")])
def test_gpu_monitor_errors_fail_closed(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(wrapper.subprocess, "run", fail)
    with pytest.raises(wrapper.ResourceStop) as caught:
        wrapper.query_gpu_memory(UUID, "unused", 1)
    assert caught.value.reason == "gpu_monitor_failed"


def test_progress_recovers_only_complete_prefix_and_unknown_attempt(tmp_path):
    path = tmp_path / "progress.jsonl"
    raw = (json.dumps(_row()) + "\n" + '{"sequence":2').encode()
    path.write_bytes(raw)
    recovered = wrapper.recover_progress(path)
    assert recovered["sha256"] == sha256(raw).hexdigest()
    assert recovered["size_bytes"] == len(raw)
    assert recovered["journal_partial"]
    assert recovered["valid_prefix_records"] == 1
    assert recovered["accounting"]["forwards_started"] == 3
    assert recovered["accounting"]["forwards_completed"] == 2
    assert recovered["accounting"]["forwards_completion_unknown"] == 1
    assert recovered["accounting"]["stage"] == "predict"


@pytest.mark.parametrize("second", [_row(sequence=3), _row(sequence=2, started=2, completed=1),
                                    dict(_row(sequence=2), applied_updates=1),
                                    dict(_row(sequence=2), forwards_completion_unknown=0)])
def test_progress_rejects_invalid_suffix_without_losing_valid_prefix(tmp_path, second):
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(second) + "\n")
    recovered = wrapper.recover_progress(path)
    assert recovered["journal_partial"]
    assert recovered["valid_prefix_records"] == 1
    assert recovered["accounting"]["forwards_completed"] == 2


@pytest.mark.parametrize("case", ["missing", "empty", "large", "no_newline"])
def test_missing_or_unbounded_journal_does_not_invent_zero_updates(tmp_path, case):
    path = tmp_path / "progress.jsonl"
    if case != "missing":
        path.write_bytes({"empty": b"", "large": b"x" * (wrapper.MAX_JOURNAL_BYTES + 1),
                          "no_newline": json.dumps(_row()).encode()}[case])
    recovered = wrapper.recover_progress(path)
    assert recovered["journal_partial"]
    assert recovered["accounting"] is None


@pytest.mark.parametrize("exit_code", [0, 3])
@requires_native_waitid
def test_child_exit_receipt_and_explicit_device_mapping(tmp_path, monkeypatch, exit_code):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: 17)
    code = wrapper.supervise(command=_child(paths, code=exit_code), gpu_uuid=UUID,
                             poll_seconds=0.01, **paths)
    receipt = json.loads(paths["receipt"].read_text())
    assert code == (0 if exit_code == 0 else 1)
    assert receipt["child_returncode"] == exit_code
    assert receipt["termination_reason"] == "child_exit"
    assert receipt["gpu_driver_memory_peak_sampled_mib"] == 17
    assert receipt["progress"]["accounting"]["forwards_completion_unknown"] == (0 if exit_code == 0 else 1)
    assert UUID in paths["stdout"].read_text()


@requires_native_waitid
def test_wall_timeout_retains_completed_lower_bound(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: 17)
    started = time.monotonic()
    assert wrapper.supervise(command=_child(paths, sleep=60), gpu_uuid=UUID,
                              wall_seconds=0.3, poll_seconds=0.01, **paths) == 1
    assert time.monotonic() - started < 3
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["status"] == "aborted"
    assert receipt["termination_reason"] == "wall_timeout"
    assert receipt["child_returncode"] < 0
    assert receipt["progress"]["accounting"]["forwards_completed"] == 2
    assert receipt["progress"]["accounting"]["forwards_completion_unknown"] == 1
    assert receipt["progress"]["accounting"]["applied_updates"] == 0


@pytest.mark.parametrize("state", ["incomplete_forward", "suite_only"])
@requires_native_waitid
def test_zero_exit_with_incomplete_journal_is_not_gate_completion(tmp_path, monkeypatch, state):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: 17)
    command = _child(paths, complete=state == "suite_only")
    if state == "suite_only":
        command[-1] = command[-1].replace("result_written", "suite_completed")
    assert wrapper.supervise(command=command, gpu_uuid=UUID,
                              poll_seconds=0.01, **paths) == 1
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["child_returncode"] == 0
    assert receipt["status"] == "failed"
    assert receipt["termination_reason"] == "progress_incomplete"
    assert receipt["progress"]["accounting"]["forwards_completion_unknown"] == (1 if state == "incomplete_forward" else 0)


@pytest.mark.parametrize("mode", ["memory", "monitor"])
def test_resource_failure_before_child_launch(tmp_path, monkeypatch, mode):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "_require_nonreaping_runtime", lambda: None)
    def monitor(*args):
        if mode == "monitor":
            raise wrapper.ResourceStop("gpu_monitor_failed", "injected unavailable")
        return 4097
    monkeypatch.setattr(wrapper, "query_gpu_memory", monitor)
    assert wrapper.supervise(command=_child(paths), gpu_uuid=UUID, **paths) == 1
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["child_pid"] is None
    assert not paths["progress"].exists()
    assert receipt["progress"]["accounting"] is None
    assert receipt["termination_reason"] == {"memory": "gpu_memory_ceiling", "monitor": "gpu_monitor_failed"}[mode]


@pytest.mark.parametrize("mode", ["memory", "monitor"])
@requires_native_waitid
def test_runtime_resource_failure_kills_child_and_recovers_progress(tmp_path, monkeypatch, mode):
    paths = _paths(tmp_path)
    def monitor(*args):
        if paths["progress"].exists():
            if mode == "monitor":
                raise wrapper.ResourceStop("gpu_monitor_failed", "injected after progress")
            return 4097
        return 5
    monkeypatch.setattr(wrapper, "query_gpu_memory", monitor)
    assert wrapper.supervise(command=_child(paths, sleep=60), gpu_uuid=UUID,
                              poll_seconds=0.01, **paths) == 1
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["child_returncode"] < 0
    assert receipt["progress"]["accounting"]["forwards_completion_unknown"] == 1
    assert receipt["termination_reason"] == {"memory": "gpu_memory_ceiling", "monitor": "gpu_monitor_failed"}[mode]


@requires_native_waitid
def test_supervisor_signal_kills_child_and_restores_handler(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: 1)
    command = _child(paths, sleep=60)
    command[-1] = command[-1].replace("time.sleep(60)",
                                      "os.kill(os.getppid(), 15); time.sleep(60)")
    handler = signal.getsignal(signal.SIGTERM)
    assert wrapper.supervise(command=command, gpu_uuid=UUID,
                              poll_seconds=0.01, **paths) == 1
    assert signal.getsignal(signal.SIGTERM) == handler
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["termination_reason"] == "supervisor_signal"
    assert receipt["child_returncode"] < 0
    assert receipt["progress"]["accounting"]["forwards_completed"] == 2


def test_query_timeout_at_deadline_is_wall_timeout(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "_require_nonreaping_runtime", lambda: None)
    def timeout(*args):
        time.sleep(args[-1] + 0.01)
        raise wrapper.ResourceStop("gpu_monitor_failed", "injected query timeout")
    monkeypatch.setattr(wrapper, "query_gpu_memory", timeout)
    assert wrapper.supervise(command=_child(paths), gpu_uuid=UUID,
                              wall_seconds=0.01, **paths) == 1
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["termination_reason"] == "wall_timeout"
    assert receipt["child_pid"] is None


@pytest.mark.parametrize("values", [{"wall_seconds": 601}, {"wall_seconds": float("nan")},
                                    {"memory_mib": 4097}, {"memory_mib": True},
                                    {"gpu_uuid": "0"}])
def test_invalid_operator_bounds_fail_before_evidence_or_gpu_access(tmp_path, monkeypatch, values):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: pytest.fail("must not query"))
    args = dict(command=_child(paths), gpu_uuid=UUID, **paths)
    args.update(values)
    with pytest.raises(ValueError):
        wrapper.supervise(**args)
    assert all(not path.exists() for path in paths.values())


def test_fresh_paths_prevent_overwriting_evidence(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    paths["receipt"].write_text("immutable")
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *a: pytest.fail("must not query"))
    with pytest.raises(FileExistsError):
        wrapper.supervise(command=_child(paths), gpu_uuid=UUID, **paths)
    assert paths["receipt"].read_text() == "immutable"


@requires_native_waitid
def test_cli_uses_only_fake_nvidia_smi_subprocess(tmp_path):
    paths = _paths(tmp_path)
    fake = tmp_path / "fake-nvidia-smi"
    fake.write_text(f"#!{sys.executable}\nprint({(UUID + ', 42')!r})\n")
    fake.chmod(0o755)
    args = ["--gpu-uuid", UUID, "--nvidia-smi", str(fake), "--wall-seconds", "5"]
    for name, path in paths.items():
        args.extend([f"--{name}", str(path)])
    assert wrapper.main(args + ["--"] + _child(paths)) == 0
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["gpu_driver_memory_peak_sampled_mib"] == 42
    assert receipt["gpu_memory_samples"] >= 1


def _fake_owned_child(monkeypatch, *, exit_code=0, outcomes=None):
    """No real PID is signalled: every lifecycle OS boundary is a local fake."""
    calls = []
    monkeypatch.setattr(wrapper.threading, "active_count", lambda: 1)
    monkeypatch.setattr(wrapper.signal, "getsignal", lambda signum: signal.SIG_DFL)
    child = SimpleNamespace(pid=987654321, returncode=None)
    info = SimpleNamespace(si_pid=child.pid, si_code=1, si_status=exit_code)
    pending = list(outcomes or [info])
    for name, value in (("CLD_EXITED", 1), ("CLD_KILLED", 2), ("CLD_DUMPED", 3)):
        monkeypatch.setattr(wrapper.os, name, value, raising=False)

    def waitid(idtype, pid, flags):
        calls.append(("waitid", idtype, pid, flags))
        assert idtype == wrapper.os.P_PID and pid == child.pid
        assert flags == wrapper.os.WEXITED | wrapper.os.WNOHANG | wrapper.os.WNOWAIT
        value = pending.pop(0) if pending else info
        if isinstance(value, BaseException):
            raise value
        return value

    def getpgid(pid):
        calls.append(("getpgid", pid))
        assert pid == child.pid
        return pid

    def killpg(pid, signum):
        calls.append(("killpg", pid, signum))
        assert child.returncode is None, "no numeric group signal after reap"
        assert (pid, signum) == (child.pid, signal.SIGKILL)

    def wait(timeout):
        calls.append(("wait", timeout))
        assert any(call[0] == "killpg" for call in calls), "reap must follow group cleanup"
        child.returncode = exit_code
        return exit_code

    child.wait = wait
    child.poll = lambda: pytest.fail("consuming Popen.poll before group cleanup")
    monkeypatch.setattr(wrapper.os, "waitid", waitid, raising=False)
    monkeypatch.setattr(wrapper.os, "getpgid", getpgid)
    monkeypatch.setattr(wrapper.os, "killpg", killpg)
    return child, calls


@pytest.mark.parametrize("exit_code", [0, 3])
def test_mock_lifecycle_retains_leader_through_group_cleanup_then_reaps_once(monkeypatch, exit_code):
    child, calls = _fake_owned_child(monkeypatch, exit_code=exit_code)
    assert wrapper._observe_child_exit(child) == exit_code
    assert child.returncode is None  # Observing exit did not reap or change Popen state.
    lifecycle = {}
    assert wrapper._cleanup_owned_group(child, lifecycle) == exit_code
    assert [call[0] for call in calls] == ["waitid", "waitid", "getpgid", "killpg", "wait"]
    assert lifecycle["ownership_confirmed_before_cleanup"]
    assert lifecycle["leader_reaped_after_cleanup"]
    assert child.returncode == exit_code


@pytest.mark.parametrize("loss", ["already_reaped", "echild", "identity_changed", "group_changed"])
def test_mock_lost_or_reused_pid_never_receives_group_signal(monkeypatch, loss):
    outcomes = [ChildProcessError("injected external reap")] if loss == "echild" else None
    child, calls = _fake_owned_child(monkeypatch, outcomes=outcomes)
    if loss == "already_reaped":
        child.returncode = 0
    elif loss == "identity_changed":
        monkeypatch.setattr(wrapper.os, "waitid", lambda *args: SimpleNamespace(si_pid=child.pid + 1))
    elif loss == "group_changed":
        monkeypatch.setattr(wrapper.os, "getpgid", lambda pid: pid + 1)
    with pytest.raises(wrapper.ResourceStop) as caught:
        wrapper._cleanup_owned_group(child, {})
    assert caught.value.reason == "child_ownership_lost"
    assert not any(call[0] in ("killpg", "wait") for call in calls)


def test_mock_observation_loss_is_terminal_without_retrying_reused_numeric_pid(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    child, calls = _fake_owned_child(monkeypatch, outcomes=[ChildProcessError("injected PID reuse")])
    monkeypatch.setattr(wrapper, "_require_nonreaping_runtime", lambda: None)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *args: 1)
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *args, **kwargs: child)
    assert wrapper.supervise(command=["synthetic-not-executed"], gpu_uuid=UUID, **paths) == 1
    assert [call[0] for call in calls] == ["waitid"]
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["termination_reason"] == "child_ownership_lost"
    assert receipt["child_returncode"] is None
    assert receipt["child_lifecycle"]["ownership_lost"]
    assert not receipt["child_lifecycle"]["group_cleanup_attempted"]


@pytest.mark.parametrize("disposition", [signal.SIG_IGN, lambda signum, frame: None])
def test_sigchld_autoreap_or_foreign_handler_rejected_before_launch(monkeypatch, disposition):
    monkeypatch.setattr(wrapper.signal, "getsignal", lambda signum: disposition)
    with pytest.raises(wrapper.ResourceStop, match="default SIGCHLD"):
        wrapper._require_ownership_context()


def test_foreign_reaper_thread_rejected(monkeypatch):
    monkeypatch.setattr(wrapper.threading, "active_count", lambda: 2)
    with pytest.raises(wrapper.ResourceStop, match="one main thread"):
        wrapper._require_ownership_context()


def test_unavailable_nonreaping_runtime_fails_before_child_or_gpu(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(wrapper.os, "waitid", None, raising=False)
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *args: pytest.fail("must not query GPU"))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    assert wrapper.supervise(command=["not-executed"], gpu_uuid=UUID, **paths) == 1
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["termination_reason"] == "nonreaping_observer_unavailable"
    assert receipt["child_pid"] is None


@requires_native_waitid
def test_native_normal_leader_exit_cleans_ordinary_descendant_before_reap(tmp_path, monkeypatch):
    """EOF verifies our descendant stopped; teardown never signals a saved PID."""
    paths = _paths(tmp_path)
    fifo = tmp_path / "descendant-alive.fifo"
    ready = tmp_path / "descendant-ready"
    os.mkfifo(fifo)
    read_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    descendant = (f"import os,time; f=os.open({str(fifo)!r},os.O_WRONLY); "
                  f"os.write(f,b'ready'); open({str(ready)!r},'w').close(); time.sleep(5)")
    command = _child(paths)
    command[-1] = ("import subprocess,time; "
                   f"subprocess.Popen([{sys.executable!r},'-c',{descendant!r}]); "
                   f"\nwhile not __import__('pathlib').Path({str(ready)!r}).exists(): time.sleep(.005)\n"
                   + command[-1])
    monkeypatch.setattr(wrapper, "query_gpu_memory", lambda *args: 1)
    try:
        started = time.monotonic()
        assert wrapper.supervise(command=command, gpu_uuid=UUID, wall_seconds=5,
                                  poll_seconds=0.01, **paths) == 0
        # Prove EOF happens before the descendant's natural five-second exit.
        assert time.monotonic() - started < 3
        assert os.read(read_fd, 5) == b"ready"
        deadline = time.monotonic() + 0.3
        while True:
            try:
                assert os.read(read_fd, 1) == b""  # EOF: descendant's held writer closed.
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    pytest.fail("ordinary descendant remained alive after normal leader exit")
                time.sleep(0.01)
        assert time.monotonic() - started < 3
        receipt = json.loads(paths["receipt"].read_text())
        assert receipt["child_returncode"] == 0
        assert receipt["child_lifecycle"]["leader_reaped_after_cleanup"]
    finally:
        os.close(read_fd)  # Child has a 5-second TTL; never signal a possibly reused PID.
