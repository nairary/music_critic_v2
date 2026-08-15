from __future__ import annotations

import pytest
import torch

from music_critic.device import (
    CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION,
    DEVICE_TRANSFER_CONTRACT_VERSION,
    RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION,
    RuntimeDeviceError,
    resolve_cuda_device_index,
    resolve_runtime_device,
)


def _make_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: device_count,
    )


def test_runtime_device_resolution_contract_has_index_validation_patch() -> None:
    assert RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION == "1.0.2"
    assert DEVICE_TRANSFER_CONTRACT_VERSION == "1.0.2"
    assert CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION == "1.0.0"


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
    _make_cuda_available(monkeypatch, device_count=1)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail(
            "explicit CUDA index must not query the current device"
        ),
    )

    assert resolve_runtime_device("cuda:0") == torch.device("cuda:0")


@pytest.mark.parametrize(
    ("current_device", "device_count"),
    ((0, 1), (1, 2)),
)
def test_abstract_cuda_resolves_to_mocked_current_device(
    monkeypatch: pytest.MonkeyPatch,
    current_device: int,
    device_count: int,
) -> None:
    _make_cuda_available(
        monkeypatch,
        device_count=device_count,
    )
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
    _make_cuda_available(monkeypatch, device_count=2)

    zero = resolve_runtime_device("cuda:0")
    one = resolve_runtime_device("cuda:1")

    assert zero != one


def test_cuda_only_helper_returns_distinct_logical_integer_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch, device_count=2)

    assert resolve_cuda_device_index("cuda:0") == 0
    assert resolve_cuda_device_index(torch.device("cuda:1")) == 1


def test_cuda_only_helper_resolves_abstract_current_logical_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch, device_count=2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert resolve_cuda_device_index("cuda") == 1


def test_cuda_only_helper_rejects_cpu_with_stable_category() -> None:
    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_cuda_device_index("cpu")

    assert (
        captured.value.category
        == "runtime.device.cuda_operation_requires_cuda"
    )
    assert str(captured.value) == (
        "runtime.device.cuda_operation_requires_cuda:requested=cpu"
    )


def test_explicit_cuda_index_is_rejected_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch, device_count=1)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail(
            "explicit CUDA index must not query the current device"
        ),
    )

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_runtime_device("cuda:1")

    assert (
        captured.value.category
        == "runtime.device.cuda_index_out_of_range"
    )
    assert captured.value.requested_device == "cuda:1"
    assert captured.value.visible_device_count == 1
    assert captured.value.resolved_index is None
    assert str(captured.value) == (
        "runtime.device.cuda_index_out_of_range:"
        "requested=cuda:1;visible_device_count=1"
    )


def test_cuda_only_helper_preserves_out_of_range_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch, device_count=1)

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_cuda_device_index("cuda:1")

    assert (
        captured.value.category
        == "runtime.device.cuda_index_out_of_range"
    )


def test_current_cuda_index_is_checked_against_visible_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_cuda_available(monkeypatch, device_count=1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_runtime_device("cuda")

    assert (
        captured.value.category
        == "runtime.device.cuda_index_out_of_range"
    )
    assert captured.value.requested_device == "cuda"
    assert captured.value.visible_device_count == 1
    assert captured.value.resolved_index == 1
    assert str(captured.value) == (
        "runtime.device.cuda_index_out_of_range:"
        "requested=cuda;visible_device_count=1;resolved_index=1"
    )


@pytest.mark.parametrize("device_count", (True, -1, "1"))
def test_invalid_visible_device_count_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    device_count: object,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: device_count,
    )

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_runtime_device("cuda:0")

    assert (
        captured.value.category
        == "runtime.device.cuda_device_count_invalid"
    )
    assert captured.value.requested_device == "cuda:0"
    assert captured.value.visible_device_count is None


@pytest.mark.parametrize(
    ("probe", "category"),
    (
        ("availability", "runtime.device.cuda_availability_probe_failed"),
        ("count", "runtime.device.cuda_device_count_invalid"),
    ),
)
def test_cuda_runtime_probe_errors_are_structured(
    monkeypatch: pytest.MonkeyPatch,
    probe: str,
    category: str,
) -> None:
    if probe == "availability":
        monkeypatch.setattr(
            torch.cuda,
            "is_available",
            lambda: (_ for _ in ()).throw(RuntimeError("raw")),
        )
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "device_count",
            lambda: (_ for _ in ()).throw(RuntimeError("raw")),
        )

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_cuda_device_index("cuda:0")

    assert captured.value.category == category
    assert captured.value.requested_device == "cuda:0"


@pytest.mark.parametrize(
    ("requested", "category"),
    (
        ("mps", "runtime.device.type_unsupported"),
        ("xpu", "runtime.device.type_unsupported"),
        ("meta", "runtime.device.type_unsupported"),
        ("cuda:not-an-index", "runtime.device.request_invalid"),
        ("", "runtime.device.request_invalid"),
    ),
)
def test_unsupported_and_malformed_devices_are_structured(
    requested: str,
    category: str,
) -> None:
    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_runtime_device(requested)

    assert captured.value.category == category
    assert captured.value.requested_device == requested


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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="logical CUDA-index integration requires a CUDA runner",
)
def test_real_cuda_statistics_and_name_accept_explicit_integer_index() -> None:
    cuda_device_index = resolve_cuda_device_index("cuda:0")
    assert cuda_device_index == 0

    torch.cuda.reset_peak_memory_stats(cuda_device_index)
    allocation = torch.empty(
        4096,
        dtype=torch.float32,
        device=torch.device("cuda", cuda_device_index),
    )
    torch.cuda.synchronize(cuda_device_index)

    assert torch.cuda.max_memory_allocated(cuda_device_index) > 0
    assert torch.cuda.max_memory_reserved(cuda_device_index) > 0
    assert torch.cuda.get_device_name(cuda_device_index)
    del allocation


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="logical CUDA-index integration requires a CUDA runner",
)
def test_real_cuda_rejects_first_out_of_range_logical_index() -> None:
    visible_device_count = torch.cuda.device_count()
    assert visible_device_count >= 1
    invalid_index = 1 if visible_device_count == 1 else visible_device_count

    with pytest.raises(RuntimeDeviceError) as captured:
        resolve_cuda_device_index(f"cuda:{invalid_index}")

    assert (
        captured.value.category
        == "runtime.device.cuda_index_out_of_range"
    )
    if visible_device_count == 1:
        assert captured.value.requested_device == "cuda:1"
