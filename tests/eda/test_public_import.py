from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_eda_import_has_no_heavy_renderer_or_legacy_imports() -> None:
    code = """
import json
import sys
import music_critic.eda

forbidden = (
    "torch",
    "torch_geometric",
    "hydra",
    "mido",
    "pretty_midi",
    "partitura",
    "src",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == []
