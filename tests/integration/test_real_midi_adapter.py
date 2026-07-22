from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.smoke_midi_adapter import _discover, _select_paths
from music_critic.adapters import MidiAdapterConfig, MidiAdapterError, load_midi_piece
from music_critic.data import dumps_piece, loads_piece, validate_piece


REPO_ROOT = Path(__file__).resolve().parents[2]
POP909_CL_ROOT = REPO_ROOT / "data" / "pop909-cl" / "POP909_processed" / "POP909_processed"
PDMX_ROOT = REPO_ROOT / "data" / "pdmx" / "mid"
RUN_REAL_MIDI = os.environ.get("MUSIC_CRITIC_RUN_REAL_MIDI_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_MIDI,
    reason="set MUSIC_CRITIC_RUN_REAL_MIDI_TESTS=1 to run local real-MIDI tests",
)


@pytest.mark.parametrize(
    ("dataset_name", "root"),
    (("pop909_cl_unsafe_complete_file", POP909_CL_ROOT), ("pdmx", PDMX_ROOT)),
)
def test_real_midi_spread_sample(dataset_name: str, root: Path) -> None:
    assert root.is_dir(), f"required {dataset_name} MIDI root is missing: {root}"
    discovered = _discover(root)
    assert discovered is not None
    assert len(discovered) >= 20, (
        f"{dataset_name} requires at least 20 MIDI files, found {len(discovered)}"
    )
    selected = _select_paths(discovered, 20, "spread")
    assert len(selected) == 20
    assert len(set(selected)) == 20

    totals = {
        "converted": 0,
        "notes": 0,
        "tracks": 0,
        "quality_flags": 0,
        "validation_warnings": 0,
    }
    config = MidiAdapterConfig(dataset_name=dataset_name)
    for path in selected:
        relative_path = path.relative_to(root).as_posix()
        try:
            piece = load_midi_piece(str(path), config=config)
        except MidiAdapterError as exc:
            pytest.fail(
                f"{dataset_name} | {relative_path} | "
                f"{' '.join(str(exc).split())[:300]}"
            )
        except Exception as exc:
            pytest.fail(
                f"{dataset_name} | {relative_path} | unexpected "
                f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
            )

        report = validate_piece(piece)
        assert report.errors == (), (
            f"{dataset_name} | {relative_path} | "
            f"{len(report.errors)} canonical validation error(s)"
        )
        assert loads_piece(dumps_piece(piece)) == piece, (
            f"{dataset_name} | {relative_path} | serialization round-trip mismatch"
        )
        totals["converted"] += 1
        totals["notes"] += len(piece.notes)
        totals["tracks"] += len(piece.tracks)
        totals["quality_flags"] += len(piece.quality_flags)
        totals["validation_warnings"] += len(report.warnings)

    assert totals["converted"] == 20
    assert totals["notes"] >= 0
    assert totals["tracks"] >= 0
    assert totals["quality_flags"] >= 0
    assert totals["validation_warnings"] >= 0
