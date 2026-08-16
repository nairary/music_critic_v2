"""Canonical runtime-device resolution shared by execution boundaries."""

from __future__ import annotations

import torch


RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION = "1.0.2"
DEVICE_TRANSFER_CONTRACT_VERSION = "1.0.2"
CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION = "1.0.0"


class RuntimeDeviceError(ValueError):
    """A structured failure to resolve one concrete runtime device."""

    def __init__(
        self,
        category: str,
        *,
        requested_device: str,
        visible_device_count: int | None = None,
        resolved_index: int | None = None,
    ) -> None:
        self.category = category
        self.requested_device = requested_device
        self.visible_device_count = visible_device_count
        self.resolved_index = resolved_index
        evidence = [f"requested={requested_device}"]
        if visible_device_count is not None:
            evidence.append(
                f"visible_device_count={visible_device_count}"
            )
        if resolved_index is not None:
            evidence.append(f"resolved_index={resolved_index}")
        super().__init__(f"{category}:" + ";".join(evidence))


def resolve_runtime_device(
    device: str | torch.device,
) -> torch.device:
    """Resolve ``cpu`` or ``cuda`` to an exactly comparable runtime device."""

    if not isinstance(device, (str, torch.device)):
        raise RuntimeDeviceError(
            "runtime.device.request_invalid",
            requested_device=type(device).__name__,
        )
    try:
        requested = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeDeviceError(
            "runtime.device.request_invalid",
            requested_device=str(device),
        ) from exc
    if requested.type == "cpu":
        return torch.device("cpu")
    if requested.type != "cuda":
        raise RuntimeDeviceError(
            "runtime.device.type_unsupported",
            requested_device=str(requested),
        )
    try:
        cuda_available = torch.cuda.is_available()
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeDeviceError(
            "runtime.device.cuda_availability_probe_failed",
            requested_device=str(requested),
        ) from exc
    if not isinstance(cuda_available, bool):
        raise RuntimeDeviceError(
            "runtime.device.cuda_availability_probe_invalid",
            requested_device=str(requested),
        )
    if not cuda_available:
        raise RuntimeDeviceError(
            "runtime.device.cuda_unavailable",
            requested_device=str(requested),
        )
    try:
        visible_device_count = torch.cuda.device_count()
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeDeviceError(
            "runtime.device.cuda_device_count_invalid",
            requested_device=str(requested),
        ) from exc
    if (
        isinstance(visible_device_count, bool)
        or not isinstance(visible_device_count, int)
        or visible_device_count < 0
    ):
        raise RuntimeDeviceError(
            "runtime.device.cuda_device_count_invalid",
            requested_device=str(requested),
        )
    resolved_from_current = requested.index is None
    if resolved_from_current:
        try:
            resolved_index = torch.cuda.current_device()
        except (AssertionError, RuntimeError) as exc:
            raise RuntimeDeviceError(
                "runtime.device.cuda_current_device_invalid",
                requested_device=str(requested),
                visible_device_count=visible_device_count,
            ) from exc
        if isinstance(resolved_index, bool) or not isinstance(
            resolved_index,
            int,
        ):
            raise RuntimeDeviceError(
                "runtime.device.cuda_current_device_invalid",
                requested_device=str(requested),
                visible_device_count=visible_device_count,
            )
    else:
        resolved_index = requested.index
    if not 0 <= resolved_index < visible_device_count:
        raise RuntimeDeviceError(
            "runtime.device.cuda_index_out_of_range",
            requested_device=str(requested),
            visible_device_count=visible_device_count,
            resolved_index=(
                resolved_index if resolved_from_current else None
            ),
        )
    return torch.device("cuda", resolved_index)


def resolve_cuda_device_index(
    device: str | torch.device,
) -> int:
    """Resolve a CUDA-only operation to one logical integer device index.

    PyTorch tensor placement accepts ``torch.device`` while some CUDA runtime
    APIs require the logical integer index.  Reusing the canonical resolver
    keeps explicit indices, the current device, and ``CUDA_VISIBLE_DEVICES``
    validation on one boundary.
    """

    resolved = resolve_runtime_device(device)
    if resolved.type != "cuda":
        raise RuntimeDeviceError(
            "runtime.device.cuda_operation_requires_cuda",
            requested_device=str(resolved),
        )
    index = resolved.index
    if isinstance(index, bool) or not isinstance(index, int):
        raise RuntimeDeviceError(
            "runtime.device.cuda_logical_index_invalid",
            requested_device=str(resolved),
        )
    return index


__all__ = [
    "CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION",
    "DEVICE_TRANSFER_CONTRACT_VERSION",
    "RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION",
    "RuntimeDeviceError",
    "resolve_cuda_device_index",
    "resolve_runtime_device",
]
