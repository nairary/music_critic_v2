from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_torch_deterministic_backend_state():
    """Prevent one test's runtime flags from authorizing a later test."""

    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_workspace_present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
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
