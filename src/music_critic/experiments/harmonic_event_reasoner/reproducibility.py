"""Opt-in fail-closed execution settings for the isolated H/member pilot.

Seeding remains the caller's responsibility. Deterministic execution is an
engineering gate on one pinned runtime, not a cross-device/version guarantee.
For CUDA, set CUBLAS_WORKSPACE_CONFIG in the launching process environment,
before any CUDA initialization. This module never modifies that environment.
"""
from __future__ import annotations

from contextlib import contextmanager
import os


@contextmanager
def deterministic_runtime(device, *, enabled=False):
    """Yield effective settings and restore process flags, even on body failure.

    The historical default is observational only: no flags or RNGs change.
    Enabling asks PyTorch to use deterministic kernels or raise, without
    suppressing unsupported-operation errors. This is process-global and must
    not be used concurrently by threads with different runtime policies.
    """
    if type(enabled) is not bool:
        raise ValueError('deterministic runtime enabled must be boolean')
    cuda = str(device).split(':', 1)[0] == 'cuda'
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if enabled and cuda and workspace not in {':4096:8', ':16:8'}:
        raise ValueError('deterministic CUDA requires launch-provided CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    import torch

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    initialized = torch.cuda.is_initialized() if cuda else False
    try:
        if enabled:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        yield dict(
            schema='HMemberRuntime@1', requested=enabled, device=str(device),
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cublas_workspace_config=workspace,
            cuda_initialized_on_entry=initialized,
        )
    finally:
        if enabled:
            torch.backends.cudnn.benchmark = previous[2]
            torch.backends.cudnn.deterministic = previous[3]
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
