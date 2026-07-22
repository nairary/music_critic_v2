#!/usr/bin/env python3
"""Deterministic read-only POP909-CL evidence audit.

This is a Phase 4A audit boundary, not the production adapter. It identifies
the documented score/channel-0 and chord/channel-1 instruments, projects only
the score into the existing generic MIDI adapter, and preserves embedded chord
blocks as target-bearing evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from os import PathLike
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import mido

from music_critic.adapters import MidiAdapterConfig, MidiAdapterError, load_midi_piece
from music_critic.data import dumps_piece, loads_piece, validate_piece


AUDIT_SCHEMA_VERSION = "3.0.0"
CORPUS_ID = "pop909_cl"
UPSTREAM_REPOSITORY = "https://github.com/AndyWeasley2004/POP909-CL-Dataset"
UPSTREAM_COMMIT = "be9094392903c471a930519e1c0bacf8b6be5d62"
UPSTREAM_PAPER = "https://arxiv.org/abs/2510.06528"
UPSTREAM_LICENSE = "MIT"
UPSTREAM_LICENSE_SHA256 = "fe6064d631bdf4ce46028ef3aa7bc4eac285b8a1000c46682795f26448d29288"
PINNED_CONTENT_FINGERPRINT = "b34f07d9a2678abdb6f0dcf5db1c3aec3f35caca813f1fac80c0717cfc8e0c65"
EXPECTED_SONG_IDS = tuple(f"{number:03d}" for number in range(1, 910))
DEFAULT_ROUND_TRIP_SAMPLE_SIZE = 16
SCORE_CHANNEL = 0
CHORD_CHANNEL = 1
EXPECTED_MISSING_CHORD_TARGET_IDS = frozenset({"367", "658"})
QUARANTINED_SCORE_IDS = frozenset({"172"})
RAW_BLOCK_PROVENANCE_ID = "pop909_cl.raw_chord_block"
NORMALIZED_TARGET_PROVENANCE_ID = "pop909_cl.upstream_normalized_target"
IMPLICIT_N_PROVENANCE_ID = "pop909_cl.upstream_implicit_n"
_SONG_ID_RE = re.compile(r"^[0-9]{3}$")
_MIDI_SUFFIXES = {".mid", ".midi"}
_T = TypeVar("_T")

_TRIADS = (
    ("M", frozenset({0, 4, 7})),
    ("m", frozenset({0, 3, 7})),
    ("o", frozenset({0, 3, 6})),
    ("+", frozenset({0, 4, 8})),
    ("sus2", frozenset({0, 2, 7})),
    ("sus4", frozenset({0, 5, 7})),
)
_SEVENTHS = (
    ("D7", frozenset({0, 4, 7, 10})),
    ("M7", frozenset({0, 4, 7, 11})),
    ("m7", frozenset({0, 3, 7, 10})),
    ("/o7", frozenset({0, 3, 6, 10})),
    ("o7", frozenset({0, 3, 6, 9})),
    ("mM7", frozenset({0, 3, 7, 11})),
    ("+7", frozenset({0, 4, 8, 10})),
)
_QUALITY_PATTERNS = (*_SEVENTHS, *_TRIADS)
_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class Pop909ClAuditError(ValueError):
    """Raised for an invalid invocation or violated CL audit precondition."""


@dataclass(frozen=True, slots=True)
class Pop909ClAsset:
    song_id: str
    path: Path
    relative_to_root: str
    relative_to_corpus: str


@dataclass(frozen=True, slots=True)
class Pop909ClDiscovery:
    root: Path
    corpus_root: Path
    assets: tuple[Pop909ClAsset, ...]
    duplicate_song_ids: tuple[tuple[str, tuple[str, ...]], ...]
    unexpected_midi_paths: tuple[str, ...]
    missing_song_ids: tuple[str, ...]
    installation_files: tuple[Path, ...]
    noise_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class InstrumentContract:
    tracks: tuple[dict[str, Any], ...]
    score_track_indices: tuple[int, ...]
    chord_track_indices: tuple[int, ...]
    metadata_track_indices: tuple[int, ...]
    unexpected_track_indices: tuple[int, ...]
    failures: tuple[dict[str, Any], ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def ensure_output_outside_root(root: Path, output: Path) -> None:
    root_resolved = root.resolve()
    output_resolved = output.resolve(strict=False)
    if output_resolved == root_resolved or output_resolved.is_relative_to(root_resolved):
        raise Pop909ClAuditError(
            f"output must be outside dataset root: {output_resolved} is under {root_resolved}"
        )


def _safe_files(root: Path) -> Iterator[Path]:
    resolved_root = root.resolve()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        for filename in sorted(filenames):
            candidate = directory_path / filename
            if candidate.is_symlink() and not _is_within(candidate, resolved_root):
                continue
            if candidate.is_file() and _is_within(candidate, resolved_root):
                yield candidate


def _is_noise(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _logical_song_id(path: Path) -> str | None:
    stripped = path.stem.strip()
    return stripped if _SONG_ID_RE.fullmatch(stripped) else None


def _find_corpus_root(root: Path) -> Path:
    candidates = (
        root,
        root / "POP909_processed",
        root / "POP909_processed" / "POP909_processed",
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        direct_midis = [
            path
            for path in candidate.iterdir()
            if path.is_file()
            and path.suffix.lower() in _MIDI_SUFFIXES
            and not _is_noise(path)
        ]
        if direct_midis:
            return candidate.resolve()
    raise Pop909ClAuditError(
        "could not find POP909_processed MIDI files at root or supported nested layout"
    )


def discover_pop909_cl(root: str | PathLike[str]) -> Pop909ClDiscovery:
    """Discover the extracted or upstream POP909_processed layout."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise Pop909ClAuditError(f"dataset root is not a directory: {root_path}")
    root_path = root_path.resolve()
    corpus_root = _find_corpus_root(root_path)
    installation_files = tuple(
        sorted(_safe_files(root_path), key=lambda path: path.relative_to(root_path).as_posix())
    )
    noise_files = tuple(path for path in installation_files if _is_noise(path))
    midi_paths = sorted(
        (
            path
            for path in corpus_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in _MIDI_SUFFIXES
            and not _is_noise(path)
        ),
        key=lambda path: path.name,
    )
    grouped: dict[str, list[Path]] = defaultdict(list)
    unexpected: list[str] = []
    for path in midi_paths:
        song_id = _logical_song_id(path)
        if song_id is None:
            unexpected.append(path.relative_to(root_path).as_posix())
        else:
            grouped[song_id].append(path)
    assets: list[Pop909ClAsset] = []
    duplicates: list[tuple[str, tuple[str, ...]]] = []
    for song_id, paths in sorted(grouped.items()):
        ordered = sorted(paths, key=lambda path: path.name)
        if len(ordered) > 1:
            duplicates.append(
                (song_id, tuple(path.relative_to(root_path).as_posix() for path in ordered))
            )
        path = ordered[0]
        assets.append(
            Pop909ClAsset(
                song_id=song_id,
                path=path,
                relative_to_root=path.relative_to(root_path).as_posix(),
                relative_to_corpus=path.relative_to(corpus_root).as_posix(),
            )
        )
    missing = tuple(sorted(set(EXPECTED_SONG_IDS) - set(grouped)))
    return Pop909ClDiscovery(
        root=root_path,
        corpus_root=corpus_root,
        assets=tuple(assets),
        duplicate_song_ids=tuple(duplicates),
        unexpected_midi_paths=tuple(sorted(unexpected)),
        missing_song_ids=missing,
        installation_files=installation_files,
        noise_files=noise_files,
    )


def propose_source_group_id(song_id: str) -> str:
    normalized = song_id.strip()
    if not _SONG_ID_RE.fullmatch(normalized):
        raise Pop909ClAuditError(f"invalid POP909-CL song identifier: {song_id!r}")
    return f"pop909-cl:{normalized}"


def propose_lineage_group_id(song_id: str) -> str:
    normalized = song_id.strip()
    if not _SONG_ID_RE.fullmatch(normalized):
        raise Pop909ClAuditError(f"invalid POP909 lineage song identifier: {song_id!r}")
    return f"pop909-lineage:{normalized}"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(rows: Iterable[tuple[str, str]]) -> str:
    digest = sha256()
    for relative, checksum in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _select_spread(items: Sequence[_T], limit: int | None) -> list[_T]:
    if limit is None or limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    denominator = limit - 1
    return [items[(index * last + denominator - 1) // denominator] for index in range(limit)]


def _quantile(values: Sequence[int | float], probability: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _distribution(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0] if ordered else None,
        "median": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "maximum": ordered[-1] if ordered else None,
    }


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda item: str(item))}


def inspect_instrument_contract(midi: mido.MidiFile) -> InstrumentContract:
    """Resolve documented instruments by channel-bearing events, not names/order."""

    tracks: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        event_channels: set[int] = set()
        note_on_channels: set[int] = set()
        names: list[str] = []
        programs: list[dict[str, int]] = []
        global_meta_events: list[dict[str, Any]] = []
        positive_notes = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "track_name":
                names.append(message.name)
            channel = getattr(message, "channel", None)
            if isinstance(channel, int):
                event_channels.add(channel)
            if message.type == "program_change":
                programs.append({"tick": absolute_tick, "channel": message.channel, "program": message.program})
            if message.type in {"set_tempo", "time_signature", "key_signature"}:
                event: dict[str, Any] = {"tick": absolute_tick, "type": message.type}
                if message.type == "set_tempo":
                    event["tempo_us_per_qn"] = int(message.tempo)
                elif message.type == "time_signature":
                    event["numerator"] = int(message.numerator)
                    event["denominator"] = int(message.denominator)
                else:
                    event["key"] = message.key
                global_meta_events.append(event)
            if message.type == "note_on" and message.velocity > 0:
                positive_notes += 1
                note_on_channels.add(message.channel)
        tracks.append({
            "source_track_index": track_index,
            "names": names,
            "event_channels": sorted(event_channels),
            "note_on_channels": sorted(note_on_channels),
            "program_changes": programs,
            "global_meta_events": global_meta_events,
            "positive_note_on_count": positive_notes,
            "end_tick": absolute_tick,
        })

    score = tuple(
        row["source_track_index"]
        for row in tracks
        if row["event_channels"] == [SCORE_CHANNEL] and row["positive_note_on_count"] > 0
    )
    chord = tuple(
        row["source_track_index"]
        for row in tracks
        if row["event_channels"] == [CHORD_CHANNEL]
    )
    metadata = tuple(
        row["source_track_index"] for row in tracks if not row["event_channels"]
    )
    unexpected = tuple(
        row["source_track_index"]
        for row in tracks
        if row["event_channels"] not in ([], [SCORE_CHANNEL], [CHORD_CHANNEL])
        or (
            row["event_channels"] == [SCORE_CHANNEL]
            and row["positive_note_on_count"] == 0
        )
    )
    failures: list[dict[str, Any]] = []
    for role, indices in (("score", score), ("chord", chord)):
        if not indices:
            failures.append({"category": f"missing_{role}_instrument", "track_indices": []})
        elif len(indices) > 1:
            failures.append({"category": f"ambiguous_{role}_instrument", "track_indices": list(indices)})
    if unexpected:
        failures.append({"category": "unexpected_or_mixed_channel_instrument", "track_indices": list(unexpected)})
    excluded_meta = tuple(
        index
        for index in (*chord, *unexpected)
        if tracks[index]["global_meta_events"]
    )
    if excluded_meta:
        failures.append({
            "category": "required_global_meta_on_excluded_instrument",
            "track_indices": list(excluded_meta),
        })
    if len(midi.tracks) < 2:
        failures.append({"category": "unexpected_track_configuration", "track_indices": list(range(len(midi.tracks)))})
    return InstrumentContract(
        tracks=tuple(tracks),
        score_track_indices=score,
        chord_track_indices=chord,
        metadata_track_indices=metadata,
        unexpected_track_indices=unexpected,
        failures=tuple(failures),
    )


def project_score_midi(midi: mido.MidiFile, contract: InstrumentContract) -> mido.MidiFile:
    """Return a score-only MIDI; no channel-1 instrument enters raw content."""

    if len(contract.score_track_indices) != 1:
        raise Pop909ClAuditError("score projection requires exactly one score instrument")
    if contract.unexpected_track_indices:
        raise Pop909ClAuditError("score projection rejects unexpected or mixed-channel instruments")
    selected = set(contract.metadata_track_indices) | {contract.score_track_indices[0]}
    excluded_global_meta = [
        row["source_track_index"]
        for row in contract.tracks
        if row["source_track_index"] not in selected and row["global_meta_events"]
    ]
    if excluded_global_meta:
        raise Pop909ClAuditError(
            "score projection rejects required global meta-events on an excluded instrument"
        )
    projected = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    for track_index, track in enumerate(midi.tracks):
        if track_index not in selected:
            continue
        copied = mido.MidiTrack()
        copied.extend(message.copy() for message in track)
        projected.tracks.append(copied)
    return projected


def project_score_midi_bytes(midi: mido.MidiFile, contract: InstrumentContract) -> bytes:
    buffer = BytesIO()
    project_score_midi(midi, contract).save(file=buffer)
    return buffer.getvalue()


def _raw_block_provenance() -> dict[str, Any]:
    return {
        "provenance_id": RAW_BLOCK_PROVENANCE_ID,
        "source": "human",
        "details": ["human_corrected", "expert_reviewed"],
        "confidence": None,
    }


def _normalizer_provenance(*, output: str) -> dict[str, Any]:
    gap_derivation = output in {
        "upstream_no_chord_gap_inference",
        "trailing_coverage_classification",
    }
    return {
        "provenance_id": IMPLICIT_N_PROVENANCE_ID
        if gap_derivation
        else NORMALIZED_TARGET_PROVENANCE_ID,
        "source": "derived",
        "confidence": None,
        "derivation_chain": [
            {
                "source": "human",
                "entity": "POP909-CL channel-1 raw chord-block evidence",
                "details": ["human_corrected", "expert_reviewed"],
            },
            {
                "source": "derived",
                "method": "POP909-CL process_pop909.py:process_pop909 gap-event construction"
                if gap_derivation
                else "POP909-CL process_pop909.py:get_chord_quality",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "semantics": (
                    "emit leading N before the first chord and internal N between chord blocks; emit no trailing N after the final chord"
                    if gap_derivation
                    else "ascending candidate roots; exact sevenths before exact triads"
                ),
            },
            {
                "source": "derived",
                "method": f"music_critic.phase_4a.{output}",
                "semantics": "candidate-preserving exact-tick evidence audit",
            },
        ],
    }


def _target_field(
    *,
    available: bool,
    value: Any,
    source: str,
    provenance_ref: str,
    candidate_values: Sequence[Any] | None = None,
) -> dict[str, Any]:
    row = {
        "available": available,
        "value": value if available else None,
        "source": source if available else None,
        "provenance": provenance_ref if available else None,
    }
    if candidate_values is not None:
        row["candidate_values"] = list(candidate_values)
    return row


def _pair_chord_notes(
    track: mido.MidiTrack,
    *,
    source_track_index: int,
    source_path: str,
    source_sha256: str,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    open_notes: dict[int, deque[tuple[int, int, int]]] = defaultdict(deque)
    paired: list[dict[str, int]] = []
    unmatched_events: list[dict[str, Any]] = []
    chord_note_event_ordinal = 0
    tick = 0
    for message in track:
        tick += int(message.time)
        if getattr(message, "channel", None) != CHORD_CHANNEL:
            continue
        is_on = message.type == "note_on" and message.velocity > 0
        is_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if not (is_on or is_off):
            continue
        ordinal = chord_note_event_ordinal
        chord_note_event_ordinal += 1
        if is_on:
            open_notes[int(message.note)].append((tick, int(message.velocity), ordinal))
        elif is_off:
            queue = open_notes[int(message.note)]
            if not queue:
                unmatched_events.append({
                    "anomaly_id": f"unmatched_note_off:{ordinal}",
                    "category": "unmatched_note_off",
                    "tick": tick,
                    "pitch": int(message.note),
                    "velocity": int(message.velocity),
                    "channel": int(message.channel),
                    "message_type": message.type,
                    "ordinal": ordinal,
                    "source_track_index": source_track_index,
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                })
                continue
            onset, velocity, note_ordinal = queue.popleft()
            paired.append({
                "onset_tick": onset,
                "end_tick": tick,
                "pitch": int(message.note),
                "velocity": velocity,
                "ordinal": note_ordinal,
            })
    dangling_events = [
        {
            "anomaly_id": f"dangling_note_on:{ordinal}",
            "category": "dangling_note_on",
            "tick": onset,
            "pitch": pitch,
            "velocity": velocity,
            "channel": CHORD_CHANNEL,
            "message_type": "note_on",
            "ordinal": ordinal,
            "source_track_index": source_track_index,
            "source_path": source_path,
            "source_sha256": source_sha256,
        }
        for pitch, queue in open_notes.items()
        for onset, velocity, ordinal in queue
    ]
    events = sorted(
        [*unmatched_events, *dangling_events],
        key=lambda row: (row["tick"], row["ordinal"], row["category"], row["pitch"]),
    )
    paired.sort(key=lambda row: (row["onset_tick"], row["pitch"], row["end_tick"], row["ordinal"]))
    return paired, {
        "unmatched_note_off": len(unmatched_events),
        "dangling_note_on": len(dangling_events),
        "events": events,
    }


def chord_normalization(pitch_classes: Iterable[int], bass_pc: int) -> dict[str, Any]:
    """Audit the pinned upstream pitch-class normalization without discarding ambiguity."""

    pcs = tuple(sorted(set(int(value) % 12 for value in pitch_classes)))
    candidates: list[dict[str, Any]] = []
    for root_pc in pcs:
        degrees = frozenset((pitch_class - root_pc) % 12 for pitch_class in pcs)
        for quality, pattern in _QUALITY_PATTERNS:
            if degrees == pattern:
                candidates.append({
                    "root_pc": root_pc,
                    "root": _PITCH_CLASS_NAMES[root_pc],
                    "quality": quality,
                    "bass_pc": bass_pc,
                    "bass": _PITCH_CLASS_NAMES[bass_pc],
                    "inversion_semitones": (bass_pc - root_pc) % 12,
                })
    selected = candidates[0] if candidates else None
    ambiguous = len(candidates) > 1
    root_values = sorted({candidate["root"] for candidate in candidates})
    quality_values = sorted({candidate["quality"] for candidate in candidates})
    inversion_values = sorted({candidate["inversion_semitones"] for candidate in candidates})
    return {
        "status": "unsupported" if not candidates else ("ambiguous" if ambiguous else "supported"),
        "selected_by_upstream_order": selected,
        "candidates": candidates,
        "provenance": {
            "source": "derived",
            "provenance_ref": NORMALIZED_TARGET_PROVENANCE_ID,
        },
        "target_fields": {
            "root": _target_field(
                available=bool(candidates) and not ambiguous,
                value=selected["root"] if selected else None,
                source="derived",
                provenance_ref=NORMALIZED_TARGET_PROVENANCE_ID,
                candidate_values=root_values,
            ),
            "quality": _target_field(
                available=bool(candidates) and len(quality_values) == 1,
                value=quality_values[0] if len(quality_values) == 1 else None,
                source="derived",
                provenance_ref=NORMALIZED_TARGET_PROVENANCE_ID,
                candidate_values=quality_values,
            ),
            "inversion": _target_field(
                available=bool(candidates) and not ambiguous,
                value=selected["inversion_semitones"] if selected else None,
                source="derived",
                provenance_ref=NORMALIZED_TARGET_PROVENANCE_ID,
                candidate_values=inversion_values,
            ),
            "bass": _target_field(
                available=True,
                value=_PITCH_CLASS_NAMES[bass_pc],
                source="human",
                provenance_ref=RAW_BLOCK_PROVENANCE_ID,
            ),
        },
    }


def extract_chord_blocks(
    midi: mido.MidiFile,
    contract: InstrumentContract,
    *,
    source_path: str,
    source_sha256: str,
    score_duration_tick: int,
) -> dict[str, Any]:
    """Extract lossless onset-grouped chord evidence from the channel-1 instrument."""

    if len(contract.chord_track_indices) != 1:
        return {
            "status": "unavailable",
            "reason": "chord instrument is missing or ambiguous",
            "task_availability": {
                task: False
                for task in ("boundary", "bass", "root", "quality", "inversion", "no_chord")
            },
            "blocks": [],
            "implicit_n_gaps": [],
            "trailing_unannotated_span": None,
            "pairing_diagnostics": {
                "unmatched_note_off": 0,
                "dangling_note_on": 0,
                "events": [],
            },
        }
    track_index = contract.chord_track_indices[0]
    paired, pairing = _pair_chord_notes(
        midi.tracks[track_index],
        source_track_index=track_index,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    for note in paired:
        grouped[note["onset_tick"]].append(note)
    track_names = contract.tracks[track_index]["names"]
    blocks: list[dict[str, Any]] = []
    for onset, notes in sorted(grouped.items()):
        pitches = sorted(note["pitch"] for note in notes)
        pitch_classes = sorted({pitch % 12 for pitch in pitches})
        lowest = min(pitches)
        bass_pc = lowest % 12
        raw_provenance = _raw_block_provenance()
        normalization = chord_normalization(pitch_classes, bass_pc)
        blocks.append({
            "onset_tick": onset,
            "end_tick": max(note["end_tick"] for note in notes),
            "note_end_ticks": sorted(note["end_tick"] for note in notes),
            "midi_pitch_multiset": pitches,
            "pitch_class_set": pitch_classes,
            "lowest_source_pitch": lowest,
            "bass_pitch_class": bass_pc,
            "source_track_index": track_index,
            "source_instrument_channel": CHORD_CHANNEL,
            "source_track_names": track_names,
            "source_path": source_path,
            "source_sha256": source_sha256,
            "provenance": raw_provenance,
            "target_fields": {
                "boundary": _target_field(
                    available=True,
                    value={
                        "onset_tick": onset,
                        "end_tick": max(note["end_tick"] for note in notes),
                    },
                    source="human",
                    provenance_ref=RAW_BLOCK_PROVENANCE_ID,
                ),
                **normalization["target_fields"],
            },
            "normalization": normalization,
            "pairing_anomaly_ids": [],
        })
    gaps, trailing_unannotated, overlap_count = chord_span_diagnostics(
        blocks, score_duration_tick
    )
    spans: list[dict[str, Any]] = [*gaps]
    if trailing_unannotated is not None:
        spans.append(trailing_unannotated)
    for event in pairing["events"]:
        tick = int(event["tick"])
        affected_blocks = [
            block
            for block in blocks
            if int(block["onset_tick"]) <= tick <= int(block["end_tick"])
        ]
        affected_spans = [
            span
            for span in spans
            if int(span["start_tick"]) <= tick < int(span["end_tick"])
        ]
        event["affected_block_onsets"] = [
            int(block["onset_tick"]) for block in affected_blocks
        ]
        event["affected_span_ids"] = [span["span_id"] for span in affected_spans]
        event["affected_interval"] = {
            "start_tick": tick,
            "end_tick": score_duration_tick
            if event["category"] == "dangling_note_on"
            else tick,
            "basis": "open_note_to_score_end"
            if event["category"] == "dangling_note_on"
            else "unmatched_point_event",
        }
        for block in affected_blocks:
            block["pairing_anomaly_ids"].append(event["anomaly_id"])
        for span in affected_spans:
            span["pairing_anomaly_ids"].append(event["anomaly_id"])
    repeated_pitch_blocks = sum(
        len(block["midi_pitch_multiset"]) != len(set(block["midi_pitch_multiset"]))
        for block in blocks
    )
    mixed_end_blocks = sum(len(set(block["note_end_ticks"])) > 1 for block in blocks)
    return {
        "status": "available",
        "task_availability": {
            task: True
            for task in ("boundary", "bass", "root", "quality", "inversion", "no_chord")
        },
        "ppqn": midi.ticks_per_beat,
        "block_count": len(blocks),
        "blocks": blocks,
        "implicit_n_gaps": gaps,
        "implicit_n_gap_count": len(gaps),
        "trailing_unannotated_span": trailing_unannotated,
        "trailing_unannotated_span_count": int(trailing_unannotated is not None),
        "overlap_count": overlap_count,
        "duplicate_block_onset_count": 0,
        "repeated_pitch_at_onset_block_count": repeated_pitch_blocks,
        "mixed_note_end_tick_block_count": mixed_end_blocks,
        "pairing_diagnostics": pairing,
    }


def chord_span_diagnostics(
    blocks: Sequence[Mapping[str, Any]],
    score_duration_tick: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    """Return upstream-compatible N gaps, masked trailing coverage, and overlaps."""

    gaps: list[dict[str, Any]] = []
    covered_until = 0
    overlap_count = 0
    gap_index = 0
    for block in blocks:
        onset = int(block["onset_tick"])
        end = int(block["end_tick"])
        if onset > covered_until:
            gap_kind = "leading_no_chord" if covered_until == 0 else "internal_no_chord"
            gaps.append({
                "span_id": f"implicit_n:{gap_index}",
                "kind": gap_kind,
                "start_tick": covered_until,
                "end_tick": onset,
                "pairing_anomaly_ids": [],
                "target_field": _target_field(
                    available=True,
                    value="N",
                    source="derived",
                    provenance_ref=IMPLICIT_N_PROVENANCE_ID,
                ),
            })
            gap_index += 1
        elif onset < covered_until:
            overlap_count += 1
        covered_until = max(covered_until, end)
    trailing: dict[str, Any] | None = None
    if covered_until < score_duration_tick:
        trailing = {
            "span_id": "trailing_unannotated:0",
            "kind": "trailing_unannotated",
            "start_tick": covered_until,
            "end_tick": score_duration_tick,
            "pairing_anomaly_ids": [],
            "target_field": _target_field(
                available=False,
                value=None,
                source="derived",
                provenance_ref=IMPLICIT_N_PROVENANCE_ID,
            ),
        }
    return gaps, trailing, overlap_count


def _score_duration_tick(contract: InstrumentContract) -> int:
    selected = set(contract.metadata_track_indices) | set(contract.score_track_indices)
    return max(
        (contract.tracks[index]["end_tick"] for index in selected),
        default=0,
    )


def analyze_meter_boundaries(midi: mido.MidiFile) -> list[dict[str, Any]]:
    events: list[tuple[int, int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            if message.type == "time_signature":
                events.append((tick, track_index, message_index, message.numerator, message.denominator))
    active_numerator = 4
    active_denominator = 4
    region_start = 0
    rows: list[dict[str, Any]] = []
    for tick, track_index, message_index, numerator, denominator in sorted(events):
        bar_length = Fraction(
            midi.ticks_per_beat * active_numerator * 4,
            active_denominator,
        )
        elapsed = Fraction(tick - region_start)
        quotient = elapsed // bar_length if bar_length else 0
        previous = Fraction(region_start) + quotient * bar_length
        remainder = elapsed - quotient * bar_length
        next_boundary = previous if remainder == 0 else previous + bar_length
        rows.append({
            "tick": tick,
            "source_track_index": track_index,
            "message_index": message_index,
            "previous_meter": f"{active_numerator}/{active_denominator}",
            "new_meter": f"{numerator}/{denominator}",
            "active_bar_length_ticks": int(bar_length) if bar_length.denominator == 1 else f"{bar_length.numerator}/{bar_length.denominator}",
            "previous_bar_boundary_tick": int(previous) if previous.denominator == 1 else f"{previous.numerator}/{previous.denominator}",
            "next_bar_boundary_tick": int(next_boundary) if next_boundary.denominator == 1 else f"{next_boundary.numerator}/{next_boundary.denominator}",
            "offset_inside_bar_ticks": int(remainder) if remainder.denominator == 1 else f"{remainder.numerator}/{remainder.denominator}",
            "on_expected_boundary": remainder == 0,
        })
        active_numerator = numerator
        active_denominator = denominator
        region_start = tick
    return rows


def _exception_category(exc: BaseException) -> tuple[str, str]:
    exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
    message = str(exc).lower()
    if isinstance(exc, MidiAdapterError):
        if "inside a bar" in message:
            return exception_type, "midi_adapter.meter_change_inside_bar"
        if "corrupted or unreadable" in message:
            return exception_type, "midi_adapter.corrupt_or_unreadable"
        if "canonical validation failed" in message:
            return exception_type, "midi_adapter.canonical_validation"
        return exception_type, "midi_adapter.other"
    return exception_type, "unexpected_exception"


def _normalized_error(exc: BaseException, actual_path: Path, display_path: str) -> str:
    return " ".join(str(exc).replace(str(actual_path), display_path).split())[:500]


def _git_identity(root: Path) -> dict[str, str] | None:
    for candidate in (root, *root.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            commit = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            remote = subprocess.run(
                ["git", "-C", str(candidate), "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        return {"commit": commit, "remote": remote}
    return None


def _ancestor_file(root: Path, name: str) -> Path | None:
    for candidate in (root, *root.parents):
        path = candidate / name
        if path.is_file():
            return path
    return None


def _upstream_comparison(
    local: Pop909ClDiscovery,
    local_hashes: Mapping[str, str],
    upstream_root: str | PathLike[str] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if upstream_root is None:
        return ({
            "performed": False,
            "expected_repository": UPSTREAM_REPOSITORY,
            "expected_commit": UPSTREAM_COMMIT,
        }, {})
    upstream = discover_pop909_cl(upstream_root)
    upstream_rows: dict[str, dict[str, Any]] = {}
    for asset in upstream.assets:
        upstream_rows[asset.song_id] = {
            "relative_to_supplied_root": asset.relative_to_root,
            "relative_to_corpus_root": asset.relative_to_corpus,
            "sha256": _hash_file(asset.path),
        }
    exact = 0
    mismatches: list[dict[str, Any]] = []
    for asset in local.assets:
        candidate = upstream_rows.get(asset.song_id)
        matched = candidate is not None and candidate["sha256"] == local_hashes[asset.song_id]
        exact += matched
        if not matched:
            mismatches.append({
                "song_id": asset.song_id,
                "local_path": asset.relative_to_root,
                "local_sha256": local_hashes[asset.song_id],
                "upstream": candidate,
            })
    local_ids = {asset.song_id for asset in local.assets}
    upstream_ids = set(upstream_rows)
    git = _git_identity(upstream.corpus_root)
    license_path = _ancestor_file(upstream.corpus_root, "LICENSE")
    observed_license_sha256 = _hash_file(license_path) if license_path else None
    upstream_fingerprint = _fingerprint(
        (asset.relative_to_corpus, upstream_rows[asset.song_id]["sha256"])
        for asset in upstream.assets
    )
    comparison = {
        "performed": True,
        "repository": UPSTREAM_REPOSITORY,
        "expected_commit": UPSTREAM_COMMIT,
        "observed_git": git,
        "license": UPSTREAM_LICENSE,
        "license_sha256": UPSTREAM_LICENSE_SHA256,
        "observed_license_sha256": observed_license_sha256,
        "upstream_content_fingerprint": upstream_fingerprint,
        "local_song_count": len(local.assets),
        "upstream_song_count": len(upstream.assets),
        "exact_content_matches": exact,
        "content_mismatches": len(mismatches),
        "mismatch_sample": mismatches[:10],
        "local_only_song_ids": sorted(local_ids - upstream_ids),
        "upstream_only_song_ids": sorted(upstream_ids - local_ids),
        "provenance_confirmed": (
            git is not None
            and git["commit"] == UPSTREAM_COMMIT
            and observed_license_sha256 == UPSTREAM_LICENSE_SHA256
            and upstream_fingerprint == PINNED_CONTENT_FINGERPRINT
            and exact == len(local.assets) == len(upstream.assets)
            and not mismatches
            and local_ids == upstream_ids
            and not upstream.missing_song_ids
            and not upstream.duplicate_song_ids
            and not upstream.unexpected_midi_paths
        ),
    }
    return comparison, upstream_rows


def _piece_result(piece: Any) -> tuple[dict[str, int], int, int]:
    validation = validate_piece(piece)
    if validation.errors:
        raise RuntimeError(f"canonical validation returned {len(validation.errors)} errors")
    warnings = Counter(flag.code for flag in piece.quality_flags)
    warnings.update(issue.code for issue in validation.warnings)
    return _counter(warnings), len(piece.notes), len(piece.tracks)


def _golden_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["song_id"]: row for row in rows}
    reasons: dict[str, set[str]] = defaultdict(set)
    for song_id, reason in (
        ("001", "ordinary_score_and_chord_contract"),
        ("043", "filename_whitespace"),
        ("172", "mid_bar_meter_change"),
        ("367", "missing_chord_instrument"),
        ("658", "score_track_name_conflicts_with_channel_evidence"),
    ):
        if song_id in by_id:
            reasons[song_id].add(reason)
    converted = [row for row in rows if row["score_projection"]["status"] == "converted"]
    for field, reason in (
        ("block_count", "maximum_chord_blocks"),
        ("unsupported_block_count", "unsupported_chord_shape"),
        ("ambiguous_block_count", "ambiguous_chord_shape"),
        ("implicit_n_gap_count", "implicit_no_chord_gaps"),
        ("overlap_count", "overlapping_chord_blocks"),
        ("score_warning_count", "maximum_score_warning_count"),
    ):
        candidates = [row for row in converted if row.get(field, 0)]
        if candidates:
            selected = max(candidates, key=lambda row: (row[field], row["song_id"]))
            reasons[selected["song_id"]].add(reason)
    trailing_candidates = [
        row
        for row in converted
        if row.get("chord_annotations", {}).get("trailing_unannotated_span")
    ]
    if trailing_candidates:
        selected = max(
            trailing_candidates,
            key=lambda row: (
                row["chord_annotations"]["trailing_unannotated_span"]["end_tick"]
                - row["chord_annotations"]["trailing_unannotated_span"]["start_tick"],
                row["song_id"],
            ),
        )
        reasons[selected["song_id"]].add("maximum_trailing_unannotated_coverage")
    result: list[dict[str, Any]] = []
    for song_id in sorted(reasons):
        row = by_id[song_id]
        result.append({
            "song_id": song_id,
            "source_group_id": row["source_group_id"],
            "lineage_group_id": row["lineage_group_id"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "reasons": sorted(reasons[song_id]),
            "expected": {
                "instrument_failure_categories": [item["category"] for item in row["instrument_contract"]["failures"]],
                "score_projection_status": row["score_projection"]["status"],
                "score_note_count": row["score_projection"].get("canonical_note_count"),
                "score_warning_codes": row["score_projection"].get("warning_codes", {}),
                "block_count": row.get("block_count", 0),
                "unsupported_block_count": row.get("unsupported_block_count", 0),
                "ambiguous_block_count": row.get("ambiguous_block_count", 0),
                "implicit_n_gap_count": row.get("implicit_n_gap_count", 0),
                "trailing_unannotated_span_count": row.get(
                    "trailing_unannotated_span_count", 0
                ),
                "overlap_count": row.get("overlap_count", 0),
                "meter_boundary_evidence": row["meter_boundary_evidence"],
            },
        })
    return result


def build_report(
    root: str | PathLike[str],
    *,
    upstream_root: str | PathLike[str] | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Run the streaming POP909-CL audit and return deterministic JSON data."""

    discovery = discover_pop909_cl(root)
    selected = _select_spread(discovery.assets, sample_size)
    content_hashes = {asset.song_id: _hash_file(asset.path) for asset in discovery.assets}
    content_fingerprint = _fingerprint(
        (asset.relative_to_corpus, content_hashes[asset.song_id]) for asset in discovery.assets
    )
    installation_rows = [
        (path.relative_to(discovery.root).as_posix(), _hash_file(path))
        for path in discovery.installation_files
    ]
    installation_fingerprint = _fingerprint(installation_rows)
    upstream_comparison, upstream_rows = _upstream_comparison(
        discovery, content_hashes, upstream_root
    )

    structure_failures: Counter[str] = Counter()
    score_warning_counts: Counter[str] = Counter()
    score_warning_files: Counter[str] = Counter()
    unsafe_warning_counts: Counter[str] = Counter()
    unsafe_warning_files: Counter[str] = Counter()
    score_failure_categories: Counter[str] = Counter()
    unsafe_failure_categories: Counter[str] = Counter()
    chord_shapes: Counter[str] = Counter()
    normalized_labels: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    basses: Counter[str] = Counter()
    inversions: Counter[int] = Counter()
    block_statuses: Counter[str] = Counter()
    track_shapes: Counter[str] = Counter()
    track_names: Counter[str] = Counter()
    global_meta_counts: Counter[str] = Counter()
    global_meta_locations: Counter[str] = Counter()
    midi_types: Counter[int] = Counter()
    ppqns: Counter[int] = Counter()
    score_rows: list[dict[str, Any]] = []
    score_note_counts: list[int] = []
    unsafe_note_counts: list[int] = []
    score_warning_totals: list[int] = []
    block_counts: list[int] = []
    gap_counts: list[int] = []
    overlap_counts: list[int] = []
    trailing_unannotated_durations: list[int] = []
    task_available: Counter[str] = Counter()
    task_unavailable: Counter[str] = Counter()
    files_with_available_chord_targets = 0
    files_with_unavailable_chord_targets = 0
    pairing_anomaly_events: list[dict[str, Any]] = []
    round_trip_ids = {
        asset.song_id
        for asset in _select_spread(selected, min(DEFAULT_ROUND_TRIP_SAMPLE_SIZE, len(selected)))
    }
    round_trips: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="music-critic-pop909-cl-audit-") as temp_dir:
        projected_path = Path(temp_dir) / "score_projection.mid"
        for asset in selected:
            checksum = content_hashes[asset.song_id]
            upstream = upstream_rows.get(asset.song_id)
            row: dict[str, Any] = {
                "song_id": asset.song_id,
                "source_group_id": propose_source_group_id(asset.song_id),
                "lineage_group_id": propose_lineage_group_id(asset.song_id),
                "relative_path": asset.relative_to_root,
                "relative_to_corpus_root": asset.relative_to_corpus,
                "sha256": checksum,
                "upstream_path": upstream["relative_to_supplied_root"] if upstream else None,
                "upstream_sha256": upstream["sha256"] if upstream else None,
                "upstream_exact_match": upstream is not None and upstream["sha256"] == checksum,
            }
            try:
                midi = mido.MidiFile(filename=asset.path)
                midi_types[int(midi.type)] += 1
                ppqns[int(midi.ticks_per_beat)] += 1
                contract = inspect_instrument_contract(midi)
                for track in contract.tracks:
                    track_shapes[str(track["event_channels"])] += 1
                    track_names.update(track["names"])
                    for event in track["global_meta_events"]:
                        global_meta_counts[event["type"]] += 1
                        global_meta_locations[
                            f"{event['type']}@track:{track['source_track_index']}"
                        ] += 1
                for failure in contract.failures:
                    structure_failures[failure["category"]] += 1
                row["instrument_contract"] = {
                    "score_track_indices": list(contract.score_track_indices),
                    "chord_track_indices": list(contract.chord_track_indices),
                    "metadata_track_indices": list(contract.metadata_track_indices),
                    "unexpected_track_indices": list(contract.unexpected_track_indices),
                    "tracks": list(contract.tracks),
                    "failures": list(contract.failures),
                }
                score_duration = _score_duration_tick(contract)
                chord = extract_chord_blocks(
                    midi,
                    contract,
                    source_path=asset.relative_to_root,
                    source_sha256=checksum,
                    score_duration_tick=score_duration,
                )
                row["chord_annotations"] = chord
                blocks = chord["blocks"]
                if chord["status"] == "available":
                    files_with_available_chord_targets += 1
                else:
                    files_with_unavailable_chord_targets += 1
                supported = 0
                ambiguous = 0
                unsupported = 0
                for block in blocks:
                    shape = ",".join(str(value) for value in block["pitch_class_set"])
                    chord_shapes[shape] += 1
                    status = block["normalization"]["status"]
                    block_statuses[status] += 1
                    supported += status in {"supported", "ambiguous"}
                    ambiguous += status == "ambiguous"
                    unsupported += status == "unsupported"
                    for task, target in block["target_fields"].items():
                        (task_available if target["available"] else task_unavailable)[task] += 1
                    selected_normalization = block["normalization"]["selected_by_upstream_order"]
                    if selected_normalization is not None:
                        label = (
                            f"{selected_normalization['root']}:{selected_normalization['quality']}"
                            f"/{selected_normalization['bass']}"
                        )
                        normalized_labels[label] += 1
                        roots[selected_normalization["root"]] += 1
                        qualities[selected_normalization["quality"]] += 1
                        basses[selected_normalization["bass"]] += 1
                        inversions[selected_normalization["inversion_semitones"]] += 1
                task_available["no_chord"] += chord.get("implicit_n_gap_count", 0)
                trailing_span = chord.get("trailing_unannotated_span")
                if trailing_span is not None:
                    task_unavailable["no_chord"] += 1
                    trailing_unannotated_durations.append(
                        int(trailing_span["end_tick"]) - int(trailing_span["start_tick"])
                    )
                pairing_anomaly_events.extend(
                    chord.get("pairing_diagnostics", {}).get("events", [])
                )
                row.update({
                    "block_count": len(blocks),
                    "supported_block_count": supported,
                    "ambiguous_block_count": ambiguous,
                    "unsupported_block_count": unsupported,
                    "implicit_n_gap_count": chord.get("implicit_n_gap_count", 0),
                    "trailing_unannotated_span_count": chord.get(
                        "trailing_unannotated_span_count", 0
                    ),
                    "overlap_count": chord.get("overlap_count", 0),
                })
                block_counts.append(len(blocks))
                gap_counts.append(row["implicit_n_gap_count"])
                overlap_counts.append(row["overlap_count"])
                row["meter_boundary_evidence"] = analyze_meter_boundaries(midi)

                try:
                    projected_bytes = project_score_midi_bytes(midi, contract)
                    projected_path.write_bytes(projected_bytes)
                    piece = load_midi_piece(
                        projected_path,
                        config=MidiAdapterConfig(
                            dataset_name=CORPUS_ID,
                            source_group_id=propose_source_group_id(asset.song_id),
                        ),
                    )
                    warnings, note_count, track_count = _piece_result(piece)
                    warning_total = sum(warnings.values())
                    score_warning_counts.update(warnings)
                    score_warning_files.update(warnings.keys())
                    score_warning_totals.append(warning_total)
                    score_note_counts.append(note_count)
                    row["score_projection"] = {
                        "status": "converted",
                        "acceptance": "accepted",
                        "sha256": sha256(projected_bytes).hexdigest(),
                        "canonical_note_count": note_count,
                        "canonical_track_count": track_count,
                        "warning_codes": warnings,
                        "warning_count": warning_total,
                    }
                    row["score_warning_count"] = warning_total
                    if asset.song_id in round_trip_ids:
                        round_trips.append({
                            "song_id": asset.song_id,
                            "equal": loads_piece(dumps_piece(piece)) == piece,
                        })
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    exception_type, category = _exception_category(exc)
                    score_failure_categories[category] += 1
                    row["score_projection"] = {
                        "status": "failed",
                        "acceptance": "quarantined"
                        if asset.song_id in QUARANTINED_SCORE_IDS
                        else "fatal_failure",
                        "exception_type": exception_type,
                        "category": category,
                        "message": _normalized_error(exc, projected_path, f"score_projection:{asset.song_id}"),
                    }
                    row["score_warning_count"] = 0

                try:
                    unsafe_piece = load_midi_piece(
                        asset.path,
                        config=MidiAdapterConfig(
                            dataset_name=f"{CORPUS_ID}_unsafe_complete_file",
                            source_group_id=propose_source_group_id(asset.song_id),
                        ),
                    )
                    warnings, note_count, track_count = _piece_result(unsafe_piece)
                    unsafe_warning_counts.update(warnings)
                    unsafe_warning_files.update(warnings.keys())
                    unsafe_note_counts.append(note_count)
                    row["unsafe_complete_file_generic"] = {
                        "status": "converted",
                        "canonical_note_count": note_count,
                        "canonical_track_count": track_count,
                        "warning_codes": warnings,
                        "warning_count": sum(warnings.values()),
                        "raw_safe": not bool(contract.chord_track_indices),
                    }
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    exception_type, category = _exception_category(exc)
                    unsafe_failure_categories[category] += 1
                    row["unsafe_complete_file_generic"] = {
                        "status": "failed",
                        "exception_type": exception_type,
                        "category": category,
                        "message": _normalized_error(exc, asset.path, asset.relative_to_root),
                        "raw_safe": False,
                    }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                exception_type, category = _exception_category(exc)
                structure_failures["midi_parse_failure"] += 1
                row.setdefault("instrument_contract", {"failures": []})
                row["instrument_contract"]["failures"].append({
                    "category": "midi_parse_failure",
                    "exception_type": exception_type,
                    "message": _normalized_error(exc, asset.path, asset.relative_to_root),
                })
                row.setdefault("chord_annotations", {"status": "unavailable", "blocks": []})
                row.setdefault("meter_boundary_evidence", [])
                row.setdefault("block_count", 0)
                row.setdefault("unsupported_block_count", 0)
                row.setdefault("ambiguous_block_count", 0)
                row.setdefault("implicit_n_gap_count", 0)
                row.setdefault("trailing_unannotated_span_count", 0)
                row.setdefault("overlap_count", 0)
                row.setdefault("score_warning_count", 0)
                row.setdefault("score_projection", {
                    "status": "failed",
                    "acceptance": "fatal_failure",
                    "exception_type": exception_type,
                    "category": category,
                    "message": _normalized_error(exc, asset.path, asset.relative_to_root),
                })
                row.setdefault("unsafe_complete_file_generic", {
                    "status": "failed",
                    "exception_type": exception_type,
                    "category": category,
                    "message": _normalized_error(exc, asset.path, asset.relative_to_root),
                    "raw_safe": False,
                })
            score_rows.append(row)

    exact_file_rows = []
    for asset in discovery.assets:
        upstream = upstream_rows.get(asset.song_id)
        exact_file_rows.append({
            "song_id": asset.song_id,
            "logical_id": asset.song_id,
            "relative_to_supplied_root": asset.relative_to_root,
            "relative_to_corpus_root": asset.relative_to_corpus,
            "sha256": content_hashes[asset.song_id],
            "upstream_relative_to_supplied_root": upstream["relative_to_supplied_root"] if upstream else None,
            "upstream_relative_to_corpus_root": upstream["relative_to_corpus_root"] if upstream else None,
            "upstream_sha256": upstream["sha256"] if upstream else None,
            "upstream_exact_match": upstream is not None and upstream["sha256"] == content_hashes[asset.song_id],
        })

    score_failures = [row for row in score_rows if row["score_projection"]["status"] == "failed"]
    unsafe_failures = [row for row in score_rows if row["unsafe_complete_file_generic"]["status"] == "failed"]
    instrument_failure_rows = []
    for row in score_rows:
        if not row["instrument_contract"]["failures"]:
            continue
        expected_missing_target = (
            row["song_id"] in EXPECTED_MISSING_CHORD_TARGET_IDS
            and [failure["category"] for failure in row["instrument_contract"]["failures"]]
            == ["missing_chord_instrument"]
        )
        instrument_failure_rows.append({
            "song_id": row["song_id"],
            "relative_path": row["relative_path"],
            "classification": "expected_masked_target_unavailability"
            if expected_missing_target
            else "fatal_structure_failure",
            "fatal": not expected_missing_target,
            "failures": row["instrument_contract"]["failures"],
        })
    fatal_instrument_failure_rows = [row for row in instrument_failure_rows if row["fatal"]]
    expected_missing_observed = {
        row["song_id"]
        for row in instrument_failure_rows
        if row["classification"] == "expected_masked_target_unavailability"
    }
    quarantined_score_failures = [
        row for row in score_failures if row["song_id"] in QUARANTINED_SCORE_IDS
    ]
    fatal_score_failures = [
        row for row in score_failures if row["song_id"] not in QUARANTINED_SCORE_IDS
    ]
    evidence_violations: list[str] = []
    if discovery.missing_song_ids:
        evidence_violations.append("missing_song_ids")
    if discovery.duplicate_song_ids:
        evidence_violations.append("duplicate_song_ids")
    if discovery.unexpected_midi_paths:
        evidence_violations.append("unexpected_midi_paths")
    if content_fingerprint != PINNED_CONTENT_FINGERPRINT:
        evidence_violations.append("pinned_content_fingerprint_mismatch")
    if upstream_root is not None and not upstream_comparison["provenance_confirmed"]:
        evidence_violations.append("upstream_content_or_commit_mismatch")
    if fatal_instrument_failure_rows:
        evidence_violations.append("fatal_instrument_contract_failures")
    if expected_missing_observed != EXPECTED_MISSING_CHORD_TARGET_IDS:
        evidence_violations.append("expected_missing_chord_target_set_mismatch")
    if fatal_score_failures:
        evidence_violations.append("fatal_score_projection_conversion_failures")
    if {row["song_id"] for row in quarantined_score_failures} != QUARANTINED_SCORE_IDS:
        evidence_violations.append("documented_quarantine_set_mismatch")
    if not all(row["equal"] for row in round_trips):
        evidence_violations.append("serialization_round_trip_failures")

    total_blocks = sum(block_counts)
    sorted_pairing_anomaly_events = sorted(
        pairing_anomaly_events,
        key=lambda row: (
            row["source_path"], row["tick"], row["ordinal"], row["category"]
        ),
    )
    pairing_anomaly_evidence_sha256 = sha256(
        json.dumps(
            sorted_pairing_anomaly_events,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "corpus_identity": {
            "corpus_id": CORPUS_ID,
            "dataset_name": "POP909-CL",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "paper": UPSTREAM_PAPER,
            "license": UPSTREAM_LICENSE,
            "license_sha256": UPSTREAM_LICENSE_SHA256,
            "root_basename": discovery.root.name,
            "corpus_root_relative": discovery.corpus_root.relative_to(discovery.root).as_posix()
            if discovery.corpus_root != discovery.root
            else ".",
            "discovered_song_count": len(discovery.assets),
            "selected_song_count": len(selected),
            "sample_size": sample_size,
            "corpus_midi_file_count": len(discovery.assets),
            "corpus_content_fingerprint": content_fingerprint,
            "installation_file_count": len(discovery.installation_files),
            "installation_noise_file_count": len(discovery.noise_files),
            "installation_fingerprint": installation_fingerprint,
            "missing_song_ids": list(discovery.missing_song_ids),
            "duplicate_song_ids": [
                {"song_id": song_id, "paths": list(paths)}
                for song_id, paths in discovery.duplicate_song_ids
            ],
            "unexpected_midi_paths": list(discovery.unexpected_midi_paths),
            "installation_noise_sample": [
                path.relative_to(discovery.root).as_posix()
                for path in discovery.noise_files[:10]
            ],
            "files": exact_file_rows,
        },
        "upstream_comparison": upstream_comparison,
        "instrument_contract": {
            "documented_score_channel": SCORE_CHANNEL,
            "documented_chord_channel": CHORD_CHANNEL,
            "selection_policy": "unique channel-bearing instrument; names and track order are corroborating evidence only",
            "failure_counts": _counter(structure_failures),
            "failures": instrument_failure_rows,
            "fatal_failure_counts": _counter(Counter(
                failure["category"]
                for row in fatal_instrument_failure_rows
                for failure in row["failures"]
            )),
            "expected_masked_target_unavailability_song_ids": sorted(
                expected_missing_observed
            ),
            "track_channel_shapes": _counter(track_shapes),
            "track_names": _counter(track_names),
            "global_meta_event_counts": _counter(global_meta_counts),
            "global_meta_event_locations": _counter(global_meta_locations),
            "midi_type_distribution": _counter(midi_types),
            "ppqn_distribution": _counter(ppqns),
        },
        "score_only_crosswalk": {
            "attempted": len(selected),
            "converted": len(selected) - len(score_failures),
            "failed": len(score_failures),
            "quarantined": len(quarantined_score_failures),
            "fatal_failed": len(fatal_score_failures),
            "quarantined_song_ids": [row["song_id"] for row in quarantined_score_failures],
            "phase_4b_mvp_policy": {
                "policy": "retain_documented_quarantine",
                "accepted_song_count": len(selected) - len(score_failures),
                "quarantined_song_ids": [
                    row["song_id"] for row in quarantined_score_failures
                ],
                "reason": "midi_adapter.meter_change_inside_bar",
            },
            "failures_by_category": _counter(score_failure_categories),
            "failures": [
                {"song_id": row["song_id"], "relative_path": row["relative_path"], **row["score_projection"]}
                for row in score_failures
            ],
            "warnings_by_code": _counter(score_warning_counts),
            "files_affected_by_warning_code": _counter(score_warning_files),
            "warning_count_distribution": _distribution(score_warning_totals),
            "canonical_note_count_distribution": _distribution(score_note_counts),
            "serialization_round_trip_sample": round_trips,
        },
        "unsafe_complete_file_generic_diagnostics": {
            "production_safe": False,
            "reason": "channel-1 chord annotation notes become canonical musical notes and raw graph observations",
            "attempted": len(selected),
            "converted": len(selected) - len(unsafe_failures),
            "failed": len(unsafe_failures),
            "failures_by_category": _counter(unsafe_failure_categories),
            "warnings_by_code": _counter(unsafe_warning_counts),
            "files_affected_by_warning_code": _counter(unsafe_warning_files),
            "canonical_note_count_distribution": _distribution(unsafe_note_counts),
        },
        "chord_annotation_inventory": {
            "raw_block_provenance": _raw_block_provenance(),
            "normalized_target_provenance": _normalizer_provenance(
                output="upstream_chord_normalization"
            ),
            "implicit_n_provenance": _normalizer_provenance(
                output="upstream_no_chord_gap_inference"
            ),
            "qualification": "curated labels preserve raw MIDI evidence; unsupported and ambiguous normalizations remain explicit",
            "time_coordinate": "exact MIDI ticks with per-file PPQN",
            "no_chord_policy": "only upstream-compatible leading/internal positive-duration gaps are derived N; trailing uncovered time is masked/unannotated",
            "files_with_available_chord_targets": files_with_available_chord_targets,
            "files_with_unavailable_chord_targets": files_with_unavailable_chord_targets,
            "task_mask_counts": {
                task: {
                    "available": task_available[task],
                    "unavailable": task_unavailable[task],
                }
                for task in ("boundary", "bass", "root", "quality", "inversion", "no_chord")
            },
            "total_blocks": total_blocks,
            "block_count_distribution": _distribution(block_counts),
            "normalization_status_counts": _counter(block_statuses),
            "selected_normalization_coverage": (
                (total_blocks - block_statuses["unsupported"]) / total_blocks if total_blocks else None
            ),
            "unambiguous_normalization_coverage": (
                block_statuses["supported"] / total_blocks if total_blocks else None
            ),
            "pitch_class_set_vocabulary": _counter(chord_shapes),
            "normalized_label_vocabulary": _counter(normalized_labels),
            "roots": _counter(roots),
            "qualities": _counter(qualities),
            "bass_pitch_classes": _counter(basses),
            "inversion_semitones": _counter(inversions),
            "implicit_n_gap_count": sum(gap_counts),
            "implicit_n_gap_count_distribution": _distribution(gap_counts),
            "trailing_unannotated_span_count": len(trailing_unannotated_durations),
            "trailing_unannotated_duration_tick_distribution": _distribution(
                trailing_unannotated_durations
            ),
            "overlap_count": sum(overlap_counts),
            "overlap_count_distribution": _distribution(overlap_counts),
            "duplicate_block_onset_count": sum(
                row["chord_annotations"].get("duplicate_block_onset_count", 0) for row in score_rows
            ),
            "repeated_pitch_at_onset_block_count": sum(
                row["chord_annotations"].get("repeated_pitch_at_onset_block_count", 0) for row in score_rows
            ),
            "mixed_note_end_tick_block_count": sum(
                row["chord_annotations"].get("mixed_note_end_tick_block_count", 0) for row in score_rows
            ),
            "pairing_diagnostics": {
                "unmatched_note_off": sum(
                    row["chord_annotations"].get("pairing_diagnostics", {}).get("unmatched_note_off", 0)
                    for row in score_rows
                ),
                "dangling_note_on": sum(
                    row["chord_annotations"].get("pairing_diagnostics", {}).get("dangling_note_on", 0)
                    for row in score_rows
                ),
            },
            "pairing_anomaly_events": sorted_pairing_anomaly_events,
            "pairing_anomaly_evidence_sha256": pairing_anomaly_evidence_sha256,
        },
        "grouping": {
            "source_group_policy": "pop909-cl:<song-id> contains all CL evidence for one song",
            "lineage_group_policy": "pop909-lineage:<song-id> binds POP909-CL and original POP909 when both are used",
            "final_splits_assigned": False,
        },
        "per_file": score_rows,
        "golden_evidence": _golden_cases(score_rows),
        "strict": {
            "evidence_contract_ready": not evidence_violations,
            "production_adapter_ready": False,
            "evidence_violations": evidence_violations,
            "production_blockers": [
                "phase_4b_production_adapter_not_implemented",
            ],
        },
    }
    return report


def dumps_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_report(report), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--sample-size", type=_positive_int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        ensure_output_outside_root(args.root, args.output)
        report = build_report(
            args.root,
            upstream_root=args.upstream_root,
            sample_size=args.sample_size,
        )
        write_report(report, args.output)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"POP909-CL audit failed: {type(exc).__name__}: {' '.join(str(exc).split())[:500]}", file=sys.stderr)
        return 2
    if args.strict and not report["strict"]["evidence_contract_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
