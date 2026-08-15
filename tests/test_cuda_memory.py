from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import music_critic.cuda_memory as cuda_memory
from music_critic.cuda_memory import (
    CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION,
    CudaMemoryStatisticsLifecycleError,
    initialize_cuda_memory_statistics,
)
from music_critic.device import RuntimeDeviceError


class _DeviceContext:
    def __init__(
        self,
        index: int,
        *,
        state: dict[str, object],
        events: list[str],
    ) -> None:
        self.index = index
        self.state = state
        self.events = events
        self.previous: int | None = None

    def __enter__(self) -> None:
        self.previous = int(self.state["current"])
        self.events.append(f"enter:{self.index}")
        self.state["current"] = self.index

    def __exit__(self, *_args: object) -> None:
        self.events.append(f"exit:{self.index}")
        self.state["current"] = self.previous


def _install_lifecycle_oracle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_current: int = 1,
    initialized: bool = False,
) -> tuple[dict[str, object], list[str]]:
    state: dict[str, object] = {
        "current": initial_current,
        "initialized": initialized,
    }
    events: list[str] = []

    def _resolve(device: str | torch.device) -> int:
        resolved = torch.device(device)
        assert resolved.type == "cuda"
        assert type(resolved.index) is int
        events.append(f"resolve:{resolved.index}")
        return resolved.index

    def _is_initialized() -> bool:
        value = bool(state["initialized"])
        events.append(f"initialized:{str(value).lower()}")
        return value

    def _device(index: int) -> _DeviceContext:
        assert type(index) is int
        return _DeviceContext(index, state=state, events=events)

    def _init() -> None:
        events.append(f"init:{state['current']}")
        state["initialized"] = True

    def _reset(index: int) -> None:
        assert type(index) is int
        events.append(f"reset:{index}")
        if state["initialized"] is not True:
            raise RuntimeError("Invalid device argument")
        assert state["current"] == index

    monkeypatch.setattr(cuda_memory, "resolve_cuda_device_index", _resolve)
    monkeypatch.setattr(torch.cuda, "is_initialized", _is_initialized)
    monkeypatch.setattr(torch.cuda, "device", _device)
    monkeypatch.setattr(torch.cuda, "init", _init)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _reset)
    return state, events


def test_uninitialized_indexed_reset_reproduces_hardware_failure_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _ = _install_lifecycle_oracle(monkeypatch)

    with pytest.raises(RuntimeError, match="Invalid device argument"):
        torch.cuda.reset_peak_memory_stats(0)

    assert state["initialized"] is False


def test_lifecycle_orders_resolve_context_init_then_integer_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, events = _install_lifecycle_oracle(monkeypatch)

    evidence = initialize_cuda_memory_statistics(torch.device("cuda:0"))

    assert evidence.to_dict() == {
        "contract_version": "1.0.0",
        "logical_device_index": 0,
        "initialized_before": False,
        "initialized_after": True,
    }
    assert events == [
        "resolve:0",
        "initialized:false",
        "enter:0",
        "init:0",
        "initialized:true",
        "reset:0",
        "exit:0",
    ]
    assert state["current"] == 1


def test_lifecycle_is_repeatable_and_does_not_change_model_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, events = _install_lifecycle_oracle(monkeypatch)
    model = torch.nn.Linear(3, 2)
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    first = initialize_cuda_memory_statistics(torch.device("cuda:0"))
    second = initialize_cuda_memory_statistics(torch.device("cuda:0"))

    assert first.initialized_before is False
    assert second.initialized_before is True
    assert second.initialized_after is True
    assert events.count("init:0") == 2
    assert events.count("reset:0") == 2
    assert state["current"] == 1
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


def test_lifecycle_keeps_cuda_zero_and_one_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, events = _install_lifecycle_oracle(
        monkeypatch,
        initial_current=0,
    )

    zero = initialize_cuda_memory_statistics(torch.device("cuda:0"))
    one = initialize_cuda_memory_statistics(torch.device("cuda:1"))

    assert zero.logical_device_index == 0
    assert one.logical_device_index == 1
    assert [event for event in events if event.startswith("reset:")] == [
        "reset:0",
        "reset:1",
    ]
    assert state["current"] == 0


def test_lifecycle_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeDeviceError) as captured:
        initialize_cuda_memory_statistics(torch.device("cpu"))

    assert (
        captured.value.category
        == "runtime.device.cuda_operation_requires_cuda"
    )


@pytest.mark.parametrize("device", ("cuda:0", torch.device("cuda")))
def test_lifecycle_requires_concrete_torch_cuda_device(
    device: str | torch.device,
) -> None:
    with pytest.raises(RuntimeDeviceError) as captured:
        initialize_cuda_memory_statistics(device)  # type: ignore[arg-type]

    assert (
        captured.value.category
        == "runtime.device.cuda_concrete_device_required"
    )


@pytest.mark.parametrize(
    ("failing_operation", "category"),
    (
        (
            "init",
            "runtime.cuda_memory_statistics.initialization_failed",
        ),
        ("reset", "runtime.cuda_memory_statistics.reset_failed"),
    ),
)
def test_lifecycle_failures_have_distinct_structured_categories(
    monkeypatch: pytest.MonkeyPatch,
    failing_operation: str,
    category: str,
) -> None:
    state, _ = _install_lifecycle_oracle(monkeypatch)

    def _fail() -> None:
        raise RuntimeError("initialization failed")

    def _reset_fail(_index: int) -> None:
        raise RuntimeError("reset failed")

    if failing_operation == "init":
        monkeypatch.setattr(torch.cuda, "init", _fail)
    else:
        monkeypatch.setattr(
            torch.cuda,
            "reset_peak_memory_stats",
            _reset_fail,
        )

    with pytest.raises(CudaMemoryStatisticsLifecycleError) as captured:
        initialize_cuda_memory_statistics(torch.device("cuda:0"))

    assert captured.value.category == category
    assert captured.value.logical_device_index == 0
    assert captured.value.initialized_before is False
    assert state["current"] == 1


def test_lifecycle_source_contains_no_dummy_allocation_or_global_set_device() -> None:
    source = Path("src/music_critic/cuda_memory.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "torch.empty",
        "torch.zeros",
        "torch.ones",
        "torch.tensor",
        "torch.cuda.set_device",
        "reset_peak_memory_stats()",
    ):
        assert forbidden not in source
    assert CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION == "1.0.0"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fresh-process lifecycle integration requires CUDA",
)
def test_fresh_process_initializes_before_indexed_reset() -> None:
    code = """
import json
import torch
from music_critic.cuda_memory import initialize_cuda_memory_statistics

before = torch.cuda.is_initialized()
evidence = initialize_cuda_memory_statistics(torch.device("cuda:0"))
print(json.dumps({
    "initialized_before_process": before,
    "evidence": evidence.to_dict(),
    "peak_allocated": torch.cuda.max_memory_allocated(0),
    "peak_reserved": torch.cuda.max_memory_reserved(0),
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "initialized_before_process": False,
        "evidence": {
            "contract_version": "1.0.0",
            "logical_device_index": 0,
            "initialized_before": False,
            "initialized_after": True,
        },
        "peak_allocated": 0,
        "peak_reserved": 0,
    }
