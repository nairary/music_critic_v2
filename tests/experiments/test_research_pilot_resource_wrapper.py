"""Only synthetic children and fake monitor readings; no GPU or checkpoint access."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("research_supervisor", ROOT / "scripts/research_pilot_resource_wrapper.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
OWNERSHIP = ROOT / "tests/fixtures/research_supervision/accepted_context_wrapper.py"
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def test_unaccepted_ownership_is_not_imported(tmp_path):
    p = tmp_path / "unsafe.py"
    p.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="unaccepted"):
        wrapper.load_ownership(p)


def test_host_memory_contract(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemAvailable: 12345 kB\nSwapTotal: 1000 kB\nSwapFree: 990 kB\n")
    assert wrapper.host_memory(p) == {"available_kib": 12345, "swap_used_kib": 10}
    p.write_text("MemAvailable: 1 kB\nSwapTotal: 1 kB\nSwapFree: 2 kB\n")
    with pytest.raises(ValueError, match="swap"):
        wrapper.host_memory(p)


@pytest.mark.parametrize("text", ["", "MemAvailable: 1 MB\n", "MemAvailable: -1 kB\n"])
def test_bad_host_memory_fails_closed(tmp_path, text):
    p = tmp_path / "meminfo"; p.write_text(text)
    with pytest.raises(ValueError):
        wrapper.host_memory(p)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="native Linux ownership acceptance")
@pytest.mark.parametrize("case", ["success", "nonzero", "timeout", "gpu", "ram", "swap", "query_error", "memory_error"])
def test_native_resource_supervision(tmp_path, monkeypatch, case):
    owner = wrapper.load_ownership(OWNERSHIP)
    calls = {"n": 0}

    def memory():
        calls["n"] += 1
        if case == "memory_error":
            raise OSError("fake meminfo read error")
        return {"available_kib": (1 if case == "ram" else 32*1024*1024),
                "swap_used_kib": 1024*1024 if case == "swap" and calls["n"] > 1 else 0}

    def gpu(*_args):
        if case == "query_error":
            raise owner.ResourceStop("gpu_monitor_failed", "fake error")
        return 4097 if case == "gpu" else 100

    monkeypatch.setattr(wrapper, "host_memory", memory)
    monkeypatch.setattr(owner, "query_gpu_memory", gpu)
    sleep = 5 if case in {"timeout", "swap"} else 0
    exit_code = 7 if case == "nonzero" else 0
    cmd = [sys.executable, "-c", f"import time; time.sleep({sleep}); raise SystemExit({exit_code})"]
    rc = wrapper.supervise(ownership=owner, command=cmd, gpu_uuid=UUID,
                           receipt=tmp_path/"receipt.json", stdout=tmp_path/"out", stderr=tmp_path/"err",
                           wall_seconds=0.3 if case == "timeout" else (43200 if case == "success" else 3),
                           memory_mib=4096, poll_seconds=0.01)
    report = json.loads((tmp_path/"receipt.json").read_text())
    assert (rc == 0) == (case == "success")
    assert report["experiment_acceptance"] == "requires_independent_task_artifact_validation"
    if report["child_pid"] is not None:
        assert report["child_lifecycle"]["ownership_confirmed_before_cleanup"]
        assert report["child_lifecycle"]["leader_reaped_after_cleanup"]
    if case == "timeout":
        assert report["reason"] == "wall_timeout"
    if case == "swap":
        assert report["reason"] == "swap_growth"
