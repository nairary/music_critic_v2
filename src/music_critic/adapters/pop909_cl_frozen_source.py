"""Hash-bound score-only reader for a separately guarded frozen evaluation.

The caller must establish membership in its sealed TRAIN/VALIDATION manifest
before resolving a record. This module independently denies other splits before
file access, never discovers a corpus, and never extracts chord evidence. It
does not enable real execution in the existing synthetic-only evaluator.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

import mido

from music_critic.data import CanonicalPiece

from .midi import MidiAdapterConfig, MidiAdapterError, load_midi_piece
from .pop909_cl import (
    POP909_CL_DATASET_NAME,
    POP909_CL_EXPECTED_MISSING_TARGET_IDS,
    POP909_CL_QUARANTINE_IDS,
    Pop909ClConversionError,
    Pop909ClCorpusRecord,
    inspect_pop909_cl_instruments,
    pop909_cl_piece_id,
    pop909_cl_raw_input_group_id,
    pop909_lineage_group_id,
    project_pop909_cl_score_bytes,
)


def _fail(category: str, message: str, song_id: str | None = None) -> None:
    raise Pop909ClConversionError(
        message, category=f"pop909_frozen.{category}", song_id=song_id
    )


def read_raw_source(
    record: Pop909ClCorpusRecord,
    *,
    split: str,
    expected_projection_sha256: str,
) -> CanonicalPiece:
    """Read one verified source and convert only its channel-0 score projection.

    Source bytes are read once; the digest and MIDI parser consume those same
    bytes. A temporary score-only file bridges the existing path-only generic
    adapter. No source/cache file is written and no target extraction is called.
    Instrument checks preserve the production adapter's missing-target rules.
    Record/group/split membership remains the calling manifest's responsibility.
    """
    if isinstance(split, str) and split.lower() == "test":
        _fail("test_access_forbidden", "TEST source access is forbidden")
    if split not in ("train", "validation"):
        _fail("split_invalid", "explicit train or validation split required")
    return _read_bound_score(record, split=split,
                             expected_projection_sha256=expected_projection_sha256)


def _read_bound_score(record: Pop909ClCorpusRecord, *, split: str,
                      expected_projection_sha256: str) -> CanonicalPiece:
    """Shared score conversion; the calling entrypoint owns its access guard.

    The original public reader still forbids TEST before reaching this helper.
    The separately registered C3 full-corpus entrypoint admits only its explicitly
    authorized, complete POP909-CL population, retaining original split identity.
    """
    if not isinstance(record, Pop909ClCorpusRecord):
        _fail("record_invalid", "Pop909ClCorpusRecord required")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (record.sha256, expected_projection_sha256)
    ):
        _fail("fingerprint_invalid", "source and projection SHA-256 required", record.song_id)
    if record.dataset_name != POP909_CL_DATASET_NAME:
        _fail("record_identity_mismatch", "POP909-CL dataset identity required", record.song_id)
    piece_id = pop909_cl_piece_id(record.song_id)
    if record.song_id != record.song_id.strip() or record.lineage_group_id != pop909_lineage_group_id(record.song_id):
        _fail("record_identity_mismatch", "song and lineage identity differ", record.song_id)
    expected_group = pop909_cl_raw_input_group_id(expected_projection_sha256)
    if record.source_group_id != expected_group:
        _fail("source_group_mismatch", "source group must bind the score projection", record.song_id)
    relative = PurePosixPath(record.relative_path)
    if not record.relative_path or relative.is_absolute() or ".." in relative.parts or "\\" in record.relative_path:
        _fail("record_path_invalid", "portable source-relative path required", record.song_id)
    if record.song_id in POP909_CL_QUARANTINE_IDS:
        _fail("quarantined_record", "pinned quarantine is not an evaluation input", record.song_id)
    try:
        payload = record.path.read_bytes()
    except (OSError, ValueError) as exc:
        raise Pop909ClConversionError(
            "cannot read the bound POP909-CL source",
            category="pop909_frozen.source_unreadable", song_id=record.song_id,
        ) from exc
    if sha256(payload).hexdigest() != record.sha256:
        _fail("source_fingerprint_mismatch", "source bytes differ from their binding", record.song_id)
    try:
        midi = mido.MidiFile(file=BytesIO(payload))
    except (EOFError, KeyError, OSError, ValueError) as exc:
        raise Pop909ClConversionError(
            "bound POP909-CL source is not readable MIDI",
            category="pop909_frozen.midi_unreadable", song_id=record.song_id,
        ) from exc
    resolution = inspect_pop909_cl_instruments(midi)
    failures = set(resolution.failure_categories)
    if "missing_chord_instrument" in failures:
        if record.song_id not in POP909_CL_EXPECTED_MISSING_TARGET_IDS:
            _fail("missing_chord_unexpected", "unexpected missing chord instrument", record.song_id)
        failures.remove("missing_chord_instrument")
    if failures:
        _fail("instrument_invalid", f"unsafe instrument contract: {sorted(failures)}", record.song_id)
    projection = project_pop909_cl_score_bytes(midi, resolution)
    if sha256(projection).hexdigest() != expected_projection_sha256:
        _fail("projection_fingerprint_mismatch", "score projection differs from its binding", record.song_id)
    try:
        with TemporaryDirectory(prefix="pop909-frozen-score-") as directory:
            projected_path = Path(directory) / "score.mid"
            projected_path.write_bytes(projection)
            piece = load_midi_piece(
                projected_path,
                config=MidiAdapterConfig(
                    dataset_name=POP909_CL_DATASET_NAME,
                    source_group_id=expected_group,
                    split=split,
                    piece_id=piece_id,
                ),
            )
    except MidiAdapterError as exc:
        raise Pop909ClConversionError(
            "score projection failed the generic MIDI contract",
            category="pop909_frozen.score_conversion_failed", song_id=record.song_id,
        ) from exc
    return replace(piece, source_path=record.relative_path)


__all__ = ["read_raw_source"]
