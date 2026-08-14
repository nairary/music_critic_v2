from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PREPARED = "tests/ssl/test_hierarchical_prepared_binding.py"
_ACCEPTANCE = "tests/ssl/test_phase8a_cuda_amp_acceptance.py"
_SUBPROCESS_SENTINEL = "MUSIC_CRITIC_PHASE8A_ORDER_SUBPROCESS"


def _run_fresh_pytest(*targets: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    environment[_SUBPROCESS_SENTINEL] = "1"
    return subprocess.run(
        (sys.executable, "-m", "pytest", "-q", *targets),
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900.0,
    )


@pytest.mark.skipif(
    os.environ.get(_SUBPROCESS_SENTINEL) == "1"
    or not torch.cuda.is_available(),
    reason="Phase 8A order independence requires actual CUDA",
)
@pytest.mark.parametrize(
    "invocations",
    [
        (
            (
                _PREPARED
                + "::test_optional_cuda_prepared_policy_parity",
            ),
        ),
        ((_PREPARED, _ACCEPTANCE),),
        (
            ("tests/ssl",),
            (_PREPARED, _ACCEPTANCE),
        ),
        ((_ACCEPTANCE, _PREPARED),),
    ],
    ids=(
        "isolated-five-policy",
        "targeted-two-files",
        "full-ssl-then-targeted",
        "reverse-modules",
    ),
)
def test_cuda_acceptance_is_fresh_process_and_order_independent(
    invocations: tuple[tuple[str, ...], ...],
) -> None:
    for targets in invocations:
        completed = _run_fresh_pytest(*targets)
        assert completed.returncode == 0, (
            f"targets={targets!r}\nstdout:\n{completed.stdout}"
            f"\nstderr:\n{completed.stderr}"
        )
