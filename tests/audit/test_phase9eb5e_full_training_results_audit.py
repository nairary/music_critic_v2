from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_phase9eb5e_source_free_results_audit_passes() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/audit_phase9eb5e_analysisgnn_full_training_results.py"
            ),
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
    assert result["selected_profile"] == "C0"
    assert result["C1_status"] == "experimental_deferred"
    assert result["final_primary_scores"] == {
        "C0": 0.3548871111124754,
        "C1": 0.2715279571712017,
    }
    assert result["corrected_joint_accuracy"] == {
        "C0": 0.11430474921480918,
        "C1": 0.01408584753021795,
    }
    assert result["unseen_tuple_joint_accuracy"] == {"C0": 0.0, "C1": 0.0}
    assert result["test_evaluated"] is False
    assert result["multi_seed_run"] is False
