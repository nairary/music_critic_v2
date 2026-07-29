from __future__ import annotations

import pytest
import torch

from music_critic.device import (
    DEVICE_TRANSFER_CONTRACT_VERSION,
    RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION,
    RuntimeDeviceError,
    resolve_runtime_device,
)


def _make_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_runtime_device_resolution_contract_starts_at_version_1() -> None:
    assert RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION == "1.0.0"
    assert DEVICE_TRANSFER_CONTRACT_VERSION == "1.0.1"


def test_cpu_resolves_to_canonical_cpu() -> None:
    assert resolve_runtime_device("cpu") == torch.device("cpu")
    assert resolve_runtime_device(torch.device("cpu")) == torch.device(
        "cpu"
    )
    assert resolve_runtime_device(torch.device("cpu:0")) == torch.device(
        "cpu"
    )


def test_explicit_cuda_zero_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail(
            "explicit CUDA index must not query the current device"
        ),
    )

    assert resolve_runtime_device("cuda:0") == torch.device("cuda:0")


@pytest.mark.parametrize("current_device", (0, 1))
def test_abstract_cuda_resolves_to_mocked_current_device(
    monkeypatch: pytest.MonkeyPatch,
    current_device: int,
) -> None:
    _make_cuda_available(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: current_device,
    )

    resolved = resolve_runtime_device(torch.device("cuda"))

    assert resolved == torch.device("cuda", current_device)
    assert resolved.index == current_device


def test_distinct_explicit_cuda_indices_are_not_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch)

    zero = resolve_runtime_device("cuda:0")
    one = resolve_runtime_device("cuda:1")

    assert zero != one


@pytest.mark.parametrize(
    "requested",
    ("cuda", torch.device("cuda"), "cuda:0"),
)
def test_unavailable_cuda_request_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    requested: str | torch.device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_runtime_device(requested)

    assert captured.value.category == "runtime.device.cuda_unavailable"
    assert captured.value.requested_device == str(requested)
    assert str(captured.value) == (
        "runtime.device.cuda_unavailable:"
        f"requested={requested}"
    )
