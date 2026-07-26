from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from music_critic.tasks import (
    collate_multisource_samples,
    prepare_multisource_sample,
)
from tests.tasks.test_multisource_contract import _hook_piece, _pop_piece


@pytest.fixture
def mixed_batch(tmp_path: Path):
    hook_piece = _hook_piece()
    pop_piece = _pop_piece(tmp_path / "pop")
    raw_piece = replace(hook_piece, annotations=(), targets=())
    return collate_multisource_samples(
        tuple(
            prepare_multisource_sample(piece)
            for piece in (hook_piece, pop_piece, raw_piece)
        )
    )
