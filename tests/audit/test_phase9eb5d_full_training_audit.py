from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_phase9eb5d_source_free_audit_passes() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_phase9eb5d_analysisgnn_full_training.py"),
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["valid"] is True
    assert all(result["checks"].values())
    assert result["ready_for_paired_cuda_full_training"] is True
    assert result["c0_full_training_completed"] is False
    assert result["c1_full_training_completed"] is False
    assert result["comparison_completed"] is False
    assert result["test_evaluated"] is False
    assert result["multi_seed_run"] is False
