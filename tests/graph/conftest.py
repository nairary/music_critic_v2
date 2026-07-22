from __future__ import annotations

from pathlib import Path

import pytest

from music_critic.data import CanonicalPiece, load_piece


@pytest.fixture
def canonical_piece() -> CanonicalPiece:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "canonical_piece_v2.json"
    return load_piece(path)
