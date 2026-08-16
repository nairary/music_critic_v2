"""Explicit CUDA memory-statistics lifecycle and evidence boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from music_critic.device import RuntimeDeviceError, resolve_cuda_device_index


CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class CudaMemoryStatisticsLifecycleEvidence:
    """Evidence that indexed memory statistics were initialized and reset."""

    contract_version: str
    logical_device_index: int
    initialized_before: bool
    initialized_after: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CudaMemoryStatisticsLifecycleError(RuntimeError):
    """Structured CUDA initialization or indexed-reset failure."""

    def __init__(
        self,
        category: str,
        *,
        requested_device: str,
        logical_device_index: int,
        initialized_before: bool,
        initialized_after: bool | None,
    ) -> None:
        self.category = category
        self.requested_device = requested_device
        self.logical_device_index = logical_device_index
        self.initialized_before = initialized_before
        self.initialized_after = initialized_after
        super().__init__(
            f"{category}:requested={requested_device};"
            f"logical_device_index={logical_device_index};"
            f"initialized_before={str(initialized_before).lower()};"
            "initialized_after="
            + (
                "unknown"
                if initialized_after is None
                else str(initialized_after).lower()
            )
        )


def initialize_cuda_memory_statistics(
    device: torch.device,
) -> CudaMemoryStatisticsLifecycleEvidence:
    """Initialize one explicit CUDA device before indexed peak-stat reset.

    The scoped device context restores the prior current device.  ``init`` is
    intentionally retained even when the runtime is already initialized: the
    public API is idempotent, and keeping one sequence avoids lifecycle races
    in fresh worker processes.  No tensor allocation is performed here.
    """

    if not isinstance(device, torch.device) or (
        device.type == "cuda" and device.index is None
    ):
        raise RuntimeDeviceError(
            "runtime.device.cuda_concrete_device_required",
            requested_device=str(device),
        )
    logical_device_index = resolve_cuda_device_index(device)
    requested_device = str(device)
    try:
        initialized_before = torch.cuda.is_initialized()
    except Exception as exc:
        raise CudaMemoryStatisticsLifecycleError(
            "runtime.cuda_memory_statistics.initialization_failed",
            requested_device=requested_device,
            logical_device_index=logical_device_index,
            initialized_before=False,
            initialized_after=None,
        ) from exc
    try:
        with torch.cuda.device(logical_device_index):
            try:
                torch.cuda.init()
                initialized_after = torch.cuda.is_initialized()
            except Exception as exc:
                raise CudaMemoryStatisticsLifecycleError(
                    "runtime.cuda_memory_statistics.initialization_failed",
                    requested_device=requested_device,
                    logical_device_index=logical_device_index,
                    initialized_before=initialized_before,
                    initialized_after=None,
                ) from exc
            if initialized_after is not True:
                raise CudaMemoryStatisticsLifecycleError(
                    "runtime.cuda_memory_statistics.initialization_failed",
                    requested_device=requested_device,
                    logical_device_index=logical_device_index,
                    initialized_before=initialized_before,
                    initialized_after=initialized_after,
                )
            try:
                torch.cuda.reset_peak_memory_stats(logical_device_index)
            except Exception as exc:
                raise CudaMemoryStatisticsLifecycleError(
                    "runtime.cuda_memory_statistics.reset_failed",
                    requested_device=requested_device,
                    logical_device_index=logical_device_index,
                    initialized_before=initialized_before,
                    initialized_after=initialized_after,
                ) from exc
    except CudaMemoryStatisticsLifecycleError:
        raise
    except Exception as exc:
        raise CudaMemoryStatisticsLifecycleError(
            "runtime.cuda_memory_statistics.initialization_failed",
            requested_device=requested_device,
            logical_device_index=logical_device_index,
            initialized_before=initialized_before,
            initialized_after=None,
        ) from exc
    return CudaMemoryStatisticsLifecycleEvidence(
        contract_version=(
            CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION
        ),
        logical_device_index=logical_device_index,
        initialized_before=initialized_before,
        initialized_after=initialized_after,
    )


__all__ = [
    "CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION",
    "CudaMemoryStatisticsLifecycleError",
    "CudaMemoryStatisticsLifecycleEvidence",
    "initialize_cuda_memory_statistics",
]
