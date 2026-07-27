from __future__ import annotations

from pathlib import Path

import pytest

from music_critic.training.config import DataConfig
from music_critic.training.data import build_data_runtime


@pytest.fixture(scope="session")
def bounded_runtime():
    return build_data_runtime(DataConfig(), seed=42)


@pytest.fixture
def bounded_batch(bounded_runtime):
    return bounded_runtime.first_train_batch


@pytest.fixture
def phase6c_output(tmp_path: Path) -> Path:
    return tmp_path / "phase6c"
