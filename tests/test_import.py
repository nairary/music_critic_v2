from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import music_critic


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_and_version() -> None:
    assert isinstance(music_critic.__version__, str)
    assert music_critic.__version__


def test_public_import_has_no_heavy_or_legacy_imports() -> None:
    code = """
import json
import sys
import music_critic

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
    assert result.stdout.strip() == "[]"


def test_data_layer_does_not_import_renderer_or_mido() -> None:
    code = """
import json
import sys
import music_critic.data

loaded = sorted(
    name for name in sys.modules
    if name == "mido" or name.startswith("mido.")
    or name == "music_critic.exporters" or name.startswith("music_critic.exporters.")
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
    assert result.stdout.strip() == "[]"
