"""Canonical runtime-device resolution shared by execution boundaries."""

from __future__ import annotations

import torch


RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION = "1.0.0"
DEVICE_TRANSFER_CONTRACT_VERSION = "1.0.1"


class RuntimeDeviceError(ValueError):
    """A structured failure to resolve one concrete runtime device."""

    def __init__(
        self,
        category: str,
        *,
        requested_device: str,
    ) -> None:
        self.category = category
        self.requested_device = requested_device
        super().__init__(
            f"{category}:requested={requested_device}"
        )


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
    if not torch.cuda.is_available():
        raise RuntimeDeviceError(
            "runtime.device.cuda_unavailable",
            requested_device=str(requested),
        )
    if requested.index is not None:
        return torch.device("cuda", requested.index)
    current_device = torch.cuda.current_device()
    if (
        isinstance(current_device, bool)
        or not isinstance(current_device, int)
        or current_device < 0
    ):
        raise RuntimeDeviceError(
            "runtime.device.cuda_current_device_invalid",
            requested_device=str(requested),
        )
    return torch.device("cuda", current_device)


__all__ = [
    "DEVICE_TRANSFER_CONTRACT_VERSION",
    "RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION",
    "RuntimeDeviceError",
    "resolve_runtime_device",
]
