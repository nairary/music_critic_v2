"""Failure-atomic deterministic runtime for repeatable CUDA evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

import torch
from torch import Tensor


DETERMINISTIC_CUDA_EVIDENCE_RUNTIME_CONTRACT_VERSION = "1.0.0"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")
DEFAULT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class DeterministicCudaEvidenceRuntimeError(RuntimeError):
    """Raised when the deterministic evidence runtime cannot be applied."""


def _validate_cublas_workspace_config() -> str | None:
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        configured is not None
        and configured not in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
    ):
        raise DeterministicCudaEvidenceRuntimeError(
            "ssl.evidence_runtime.cublas_workspace_config_invalid"
        )
    return configured


@contextmanager
def deterministic_cuda_evidence_runtime() -> Iterator[None]:
    """Apply and failure-atomically restore exact-replay runtime state.

    This context configures execution only for deterministic CUDA evidence.
    It does not seed the caller, suppress nondeterministic-operation errors, or
    make a cross-process/cross-backend bit-identity promise.
    """

    previous_workspace = _validate_cublas_workspace_config()
    previous_workspace_present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cpu_rng = torch.get_rng_state().clone()
    previous_cuda_rng: tuple[Tensor, ...] | None = None

    try:
        if previous_workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                DEFAULT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
            )
        if torch.cuda.is_available():
            previous_cuda_rng = tuple(
                value.clone() for value in torch.cuda.get_rng_state_all()
            )
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        # Restore every captured surface even when the evidence body raises.
        # None of these operations intentionally consumes random values.
        torch.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(list(previous_cuda_rng))
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = (
            previous_cudnn_deterministic
        )
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        if previous_workspace_present:
            assert previous_workspace is not None
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


__all__ = [
    "DEFAULT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG",
    "DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS",
    "DETERMINISTIC_CUDA_EVIDENCE_RUNTIME_CONTRACT_VERSION",
    "DeterministicCudaEvidenceRuntimeError",
    "deterministic_cuda_evidence_runtime",
]
