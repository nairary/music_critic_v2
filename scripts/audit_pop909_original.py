#!/usr/bin/env python3
"""Deterministic, read-only original-POP909 lineage evidence audit.

This module is deliberately an audit boundary, not a production dataset adapter.
It supports both the official song-directory layout and flattened/processed MIDI
mirrors, retaining missing evidence and per-file failures in deterministic JSON.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from os import PathLike
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import mido

from music_critic.adapters import MidiAdapterConfig, MidiAdapterError, load_midi_piece
from music_critic.data import dumps_piece, loads_piece, validate_piece


AUDIT_SCHEMA_VERSION = "1.0.0"
OFFICIAL_REPOSITORY = "https://github.com/music-x-lab/POP909-Dataset"
OFFICIAL_PAPER = "https://arxiv.org/abs/2008.07142"
ALIGNMENT_TOLERANCE_SECONDS = Decimal("0.1")
DEFAULT_ROUND_TRIP_SAMPLE_SIZE = 16
ANNOTATION_FAMILIES = (
    "beat_audio",
    "beat_midi",
    "chord_audio",
    "chord_midi",
    "key_audio",
)
EXPECTED_ROLE_NAMES = {
    "MELODY": "melody",
    "BRIDGE": "secondary_melody",
    "PIANO": "accompaniment",
}
_MIDI_SUFFIXES = {".mid", ".midi"}
_SONG_ID_RE = re.compile(r"^[0-9]{3}$")
_VERSION_RE = re.compile(r"^(?P<song>[0-9]{3})-v(?P<version>[0-9]+)$", re.IGNORECASE)
_CHORD_RE = re.compile(
    r"^(?P<root>[A-G](?:b|#)?):(?P<body>[A-Za-z0-9]+(?:\([^()]*\))?)(?:/(?P<bass>[b#]*[0-9]+))?$"
)
_CHORD_BODY_RE = re.compile(r"^(?P<quality>[A-Za-z0-9]+)(?:\((?P<mods>[^()]*)\))?$")
_DEGREE_RE = re.compile(r"^(?P<alter>[b#]*)(?P<degree>[0-9]+)$")
_KEY_RE = re.compile(r"^(?P<tonic>[A-G](?:b|#)?):(?P<mode>maj|min)$")
_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
_DEGREE_SEMITONES = (0, 2, 4, 5, 7, 9, 11)
_T = TypeVar("_T")


class Pop909AuditError(ValueError):
    """Raised for an invalid audit invocation or malformed evidence line."""


@dataclass(frozen=True, slots=True)
class SongAssets:
    song_id: str
    primary_midi: Path
    alternatives: tuple[Path, ...]
    annotations: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class Discovery:
    root: Path
    corpus_root: Path
    layout: str
    songs: tuple[SongAssets, ...]
    files: tuple[Path, ...]
    unexpected: tuple[Path, ...]
    duplicate_song_ids: tuple[tuple[str, tuple[Path, ...]], ...]
    duplicate_version_ids: tuple[tuple[str, tuple[Path, ...]], ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def ensure_output_outside_root(root: Path, output: Path) -> None:
    """Reject every output path that resolves inside the dataset root."""

    root_resolved = root.resolve()
    output_resolved = output.resolve(strict=False)
    if output_resolved == root_resolved or output_resolved.is_relative_to(root_resolved):
        raise Pop909AuditError(
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


def _find_corpus_root(root: Path) -> tuple[Path, str]:
    candidates = (root, root / "POP909")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        numeric_dirs = [
            path for path in candidate.iterdir() if path.is_dir() and _SONG_ID_RE.fullmatch(path.name)
        ]
        if numeric_dirs:
            return candidate, "official_song_directories"
    return root, "processed_flat_midi"


def _normalized_song_id(path: Path) -> str | None:
    stem = path.stem.strip()
    return stem if _SONG_ID_RE.fullmatch(stem) else None


def discover_dataset(root: str | PathLike[str]) -> Discovery:
    """Discover POP909 assets deterministically without mutating ``root``."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise Pop909AuditError(f"dataset root is not a directory: {root_path}")
    root_path = root_path.resolve()
    corpus_root, layout = _find_corpus_root(root_path)
    all_files = tuple(sorted(_safe_files(root_path), key=lambda path: _relative(path, root_path)))
    unexpected: list[Path] = []
    songs: list[SongAssets] = []
    primary_by_id: dict[str, list[Path]] = defaultdict(list)
    versions_by_id: dict[str, list[Path]] = defaultdict(list)

    if layout == "official_song_directories":
        for song_dir in sorted(
            (path for path in corpus_root.iterdir() if path.is_dir() and _SONG_ID_RE.fullmatch(path.name)),
            key=lambda path: path.name,
        ):
            song_id = song_dir.name
            expected_primary = song_dir / f"{song_id}.mid"
            primary_candidates = sorted(
                (
                    path
                    for path in song_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in _MIDI_SUFFIXES
                    and _normalized_song_id(path) == song_id
                ),
                key=lambda path: path.name,
            )
            if expected_primary.is_file():
                primary = expected_primary
            elif primary_candidates:
                primary = primary_candidates[0]
            else:
                primary = expected_primary
            for candidate in primary_candidates:
                primary_by_id[song_id].append(candidate)

            alternatives: list[Path] = []
            versions_dir = song_dir / "versions"
            if versions_dir.is_dir():
                for candidate in sorted(versions_dir.iterdir(), key=lambda path: path.name):
                    if not candidate.is_file() or candidate.suffix.lower() not in _MIDI_SUFFIXES:
                        unexpected.append(candidate)
                        continue
                    alternatives.append(candidate)
                    match = _VERSION_RE.fullmatch(candidate.stem.strip())
                    version_id = (
                        f"{match.group('song').zfill(3)}-v{int(match.group('version'))}"
                        if match
                        else candidate.stem.strip().lower()
                    )
                    versions_by_id[version_id].append(candidate)

            annotations = tuple(
                (family, song_dir / f"{family}.txt") for family in ANNOTATION_FAMILIES
            )
            songs.append(
                SongAssets(
                    song_id=song_id,
                    primary_midi=primary,
                    alternatives=tuple(alternatives),
                    annotations=annotations,
                )
            )
    else:
        midi_candidates = [
            path
            for path in all_files
            if path.suffix.lower() in _MIDI_SUFFIXES
            and "__MACOSX" not in path.parts
            and not path.name.startswith("._")
        ]
        grouped: dict[str, list[Path]] = defaultdict(list)
        for candidate in midi_candidates:
            song_id = _normalized_song_id(candidate)
            if song_id is None:
                unexpected.append(candidate)
                continue
            grouped[song_id].append(candidate)
            primary_by_id[song_id].append(candidate)
        for song_id, paths in sorted(grouped.items()):
            ordered = sorted(paths, key=lambda path: _relative(path, root_path))
            songs.append(
                SongAssets(
                    song_id=song_id,
                    primary_midi=ordered[0],
                    alternatives=(),
                    annotations=(),
                )
            )

    known_paths = {
        path
        for song in songs
        for path in (
            song.primary_midi,
            *song.alternatives,
            *(path for _, path in song.annotations),
        )
        if path.is_file()
    }
    for path in all_files:
        relative = _relative(path, root_path)
        lower_name = path.name.lower()
        if path in known_paths:
            continue
        if (
            lower_name.startswith("readme")
            or lower_name.startswith("license")
            or path.suffix.lower() in {".md", ".xlsx"}
        ):
            continue
        unexpected.append(path)

    duplicate_songs = tuple(
        (song_id, tuple(sorted(paths, key=lambda path: _relative(path, root_path))))
        for song_id, paths in sorted(primary_by_id.items())
        if len(paths) > 1
    )
    duplicate_versions = tuple(
        (version_id, tuple(sorted(paths, key=lambda path: _relative(path, root_path))))
        for version_id, paths in sorted(versions_by_id.items())
        if len(paths) > 1
    )
    return Discovery(
        root=root_path,
        corpus_root=corpus_root,
        layout=layout,
        songs=tuple(songs),
        files=all_files,
        unexpected=tuple(sorted(set(unexpected), key=lambda path: _relative(path, root_path))),
        duplicate_song_ids=duplicate_songs,
        duplicate_version_ids=duplicate_versions,
    )


def propose_source_group_id(song_id: str) -> str:
    """Return the stable group ID shared by every asset/version of one song."""

    normalized = song_id.strip()
    if not _SONG_ID_RE.fullmatch(normalized):
        raise Pop909AuditError(f"invalid POP909 song identifier: {song_id!r}")
    return f"pop909-original:{normalized}"


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_kind(path: Path, discovery: Discovery) -> str:
    relative = _relative(path, discovery.root)
    lower = path.name.lower()
    if "__MACOSX" in path.parts or path.name.startswith("._"):
        return "appledouble_metadata"
    if path.suffix.lower() in _MIDI_SUFFIXES:
        if "/versions/" in f"/{relative}":
            return "alternative_midi"
        return "primary_or_processed_midi"
    if path.stem in ANNOTATION_FAMILIES and path.suffix.lower() == ".txt":
        return f"annotation.{path.stem}"
    if lower.startswith("readme"):
        return "readme"
    if lower.startswith("license"):
        return "license"
    if lower == "index.xlsx":
        return "index"
    return "unexpected"


def _git_identity(discovery: Discovery) -> dict[str, Any] | None:
    candidates = (discovery.root, discovery.root.parent)
    for candidate in candidates:
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
            continue
        return {"commit": commit, "remote": remote}
    return None


def _select_spread(items: Sequence[_T], limit: int | None) -> list[_T]:
    if limit is None or limit >= len(items):
        return list(items)
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    denominator = limit - 1
    return [items[(index * last + denominator - 1) // denominator] for index in range(limit)]


def _quantile(sorted_values: Sequence[float | int], probability: float) -> float | int | None:
    if not sorted_values:
        return None
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    index = math.ceil(probability * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


def summarize_distribution(values: Iterable[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
    }


def _decimal(token: str, *, family: str, line_number: int, field: str) -> Decimal:
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise Pop909AuditError(
            f"{family} line {line_number}: invalid {field} decimal {token!r}"
        ) from exc
    if not value.is_finite():
        raise Pop909AuditError(
            f"{family} line {line_number}: non-finite {field} decimal {token!r}"
        )
    return value


def parse_annotation_line(family: str, line: str, line_number: int) -> dict[str, Any]:
    """Parse one official annotation line without float coercion."""

    if family not in ANNOTATION_FAMILIES:
        raise Pop909AuditError(f"unsupported annotation family: {family}")
    stripped = line.strip()
    if not stripped:
        raise Pop909AuditError(f"{family} line {line_number}: empty line")
    columns = stripped.split()
    expected = 2 if family == "beat_audio" else 3
    if len(columns) != expected:
        raise Pop909AuditError(
            f"{family} line {line_number}: expected {expected} columns, found {len(columns)}"
        )
    if family == "beat_audio":
        return {
            "time_seconds": _decimal(columns[0], family=family, line_number=line_number, field="time"),
            "beat_order": _decimal(columns[1], family=family, line_number=line_number, field="beat_order"),
        }
    if family == "beat_midi":
        return {
            "time_seconds": _decimal(columns[0], family=family, line_number=line_number, field="time"),
            "downbeat_simple": _decimal(columns[1], family=family, line_number=line_number, field="downbeat_simple"),
            "downbeat_compound": _decimal(columns[2], family=family, line_number=line_number, field="downbeat_compound"),
        }
    record = {
        "start_seconds": _decimal(columns[0], family=family, line_number=line_number, field="start"),
        "end_seconds": _decimal(columns[1], family=family, line_number=line_number, field="end"),
        "label": columns[2],
    }
    if record["end_seconds"] < record["start_seconds"]:
        raise Pop909AuditError(f"{family} line {line_number}: end precedes start")
    return record


def parse_chord_label(label: str) -> dict[str, Any]:
    """Parse the observed Harte-like POP909 chord vocabulary losslessly."""

    if label == "N":
        return {
            "raw": label,
            "status": "no_chord",
            "root": None,
            "root_pc": None,
            "quality": None,
            "extensions": [],
            "alterations": [],
            "suspensions": [],
            "bass": None,
            "bass_pc": None,
            "lossless": True,
        }
    match = _CHORD_RE.fullmatch(label)
    if match is None:
        return {"raw": label, "status": "unparsed", "lossless": False}
    body = _CHORD_BODY_RE.fullmatch(match.group("body"))
    if body is None:
        return {"raw": label, "status": "unparsed", "lossless": False}
    quality = body.group("quality")
    modifiers = [] if body.group("mods") is None else body.group("mods").split(",")
    if any(not token or _DEGREE_RE.fullmatch(token) is None for token in modifiers):
        return {"raw": label, "status": "unparsed", "lossless": False}
    bass = match.group("bass")
    if bass is not None and _DEGREE_RE.fullmatch(bass) is None:
        return {"raw": label, "status": "unparsed", "lossless": False}
    root = match.group("root")
    root_pc = _PITCH_CLASS.get(root)
    bass_pc = _bass_pitch_class(root_pc, bass)
    return {
        "raw": label,
        "status": "parsed",
        "root": root,
        "root_pc": root_pc,
        "quality": quality,
        "extensions": [token for token in modifiers if not token.startswith(("b", "#"))],
        "alterations": [token for token in modifiers if token.startswith(("b", "#"))],
        "suspensions": [quality] if quality.startswith("sus") else [],
        "bass": bass,
        "bass_pc": bass_pc,
        "lossless": True,
    }


def _bass_pitch_class(root_pc: int | None, bass: str | None) -> int | None:
    if root_pc is None or bass is None:
        return None
    match = _DEGREE_RE.fullmatch(bass)
    if match is None:
        return None
    degree = int(match.group("degree"))
    if degree <= 0:
        return None
    natural = _DEGREE_SEMITONES[(degree - 1) % 7] + 12 * ((degree - 1) // 7)
    alteration = match.group("alter").count("#") - match.group("alter").count("b")
    return (root_pc + natural + alteration) % 12


def parse_key_label(label: str) -> dict[str, Any]:
    match = _KEY_RE.fullmatch(label)
    if match is None:
        return {"raw": label, "status": "unparsed", "lossless": False}
    tonic = match.group("tonic")
    return {
        "raw": label,
        "status": "parsed",
        "tonic": tonic,
        "tonic_pc": _PITCH_CLASS.get(tonic),
        "mode": "major" if match.group("mode") == "maj" else "minor",
        "lossless": True,
    }


@dataclass(slots=True)
class _TimingAccumulator:
    signed_errors: list[float]
    unmatched: int = 0
    outside_duration: int = 0

    @classmethod
    def create(cls) -> _TimingAccumulator:
        return cls(signed_errors=[])

    def add(self, signed_error: Fraction, *, matched: bool, outside: bool) -> None:
        self.signed_errors.append(float(signed_error))
        if not matched:
            self.unmatched += 1
        if outside:
            self.outside_duration += 1

    def summary(self) -> dict[str, Any]:
        signed = sorted(self.signed_errors)
        absolute = sorted(abs(value) for value in signed)
        return {
            "count": len(signed),
            "absolute_seconds": {
                "median": _quantile(absolute, 0.5),
                "p95": _quantile(absolute, 0.95),
                "maximum": absolute[-1] if absolute else None,
            },
            "signed_seconds": {
                "minimum": signed[0] if signed else None,
                "p05": _quantile(signed, 0.05),
                "median": _quantile(signed, 0.5),
                "p95": _quantile(signed, 0.95),
                "maximum": signed[-1] if signed else None,
                "mean": (sum(signed) / len(signed)) if signed else None,
            },
            "unmatched_over_100ms": self.unmatched,
            "outside_piece_duration": self.outside_duration,
        }


def _fraction_from_decimal(value: Decimal) -> Fraction:
    return Fraction(value)


def _tempo_map(midi_file: Any) -> tuple[list[tuple[int, int]], int]:
    located: list[tuple[int, int, int, int]] = []
    source_end_tick = 0
    for track_index, track in enumerate(midi_file.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            source_end_tick = max(source_end_tick, tick)
            if message.type == "set_tempo":
                located.append((tick, track_index, message_index, int(message.tempo)))
    first_by_tick: dict[int, int] = {}
    for tick, _track_index, _message_index, tempo in sorted(located):
        first_by_tick.setdefault(tick, tempo)
    if 0 not in first_by_tick:
        first_by_tick[0] = 500_000
    return sorted(first_by_tick.items()), source_end_tick


def _seconds_at_tick(tick: Fraction, tempo: Sequence[tuple[int, int]], ppqn: int) -> Fraction:
    if tick < 0:
        return Fraction(0)
    elapsed = Fraction(0)
    cursor = 0
    active = tempo[0][1]
    for event_tick, value in tempo[1:]:
        if tick <= event_tick:
            break
        elapsed += Fraction((event_tick - cursor) * active, ppqn * 1_000_000)
        cursor = event_tick
        active = value
    elapsed += Fraction(tick - cursor) * Fraction(active, ppqn * 1_000_000)
    return elapsed


def _nearest(reference: Sequence[Fraction], observed: Fraction) -> tuple[Fraction, Fraction]:
    if not reference:
        return observed, Fraction(0)
    index = bisect_left(reference, observed)
    candidates: list[Fraction] = []
    if index < len(reference):
        candidates.append(reference[index])
    if index:
        candidates.append(reference[index - 1])
    selected = min(candidates, key=lambda value: (abs(observed - value), value))
    return observed - selected, selected


def compare_timings(
    observed_seconds: Iterable[Decimal],
    canonical_seconds: Sequence[Fraction],
    *,
    duration_seconds: Fraction,
    tolerance_seconds: Decimal = ALIGNMENT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Compare observed seconds to nearest exact canonical points."""

    accumulator = _TimingAccumulator.create()
    tolerance = _fraction_from_decimal(tolerance_seconds)
    for value in observed_seconds:
        observed = _fraction_from_decimal(value)
        signed, _ = _nearest(canonical_seconds, observed)
        accumulator.add(
            signed,
            matched=abs(signed) <= tolerance,
            outside=observed < 0 or observed > duration_seconds,
        )
    return accumulator.summary()


def _raw_midi_evidence(path: Path) -> dict[str, Any]:
    midi_file = mido.MidiFile(filename=path)
    track_rows: list[dict[str, Any]] = []
    tempo_count = 0
    meter_values: list[str] = []
    tempo_values: list[int] = []
    for track_index, track in enumerate(midi_file.tracks):
        tick = 0
        names: list[str] = []
        channels: set[int] = set()
        programs: set[int] = set()
        note_count = 0
        for message in track:
            tick += int(message.time)
            if message.type == "track_name":
                names.append(message.name)
            elif message.type == "set_tempo":
                tempo_count += 1
                tempo_values.append(int(message.tempo))
            elif message.type == "time_signature":
                meter_values.append(f"{message.numerator}/{message.denominator}")
            channel = getattr(message, "channel", None)
            if isinstance(channel, int):
                channels.add(channel)
            if message.type == "program_change":
                programs.add(int(message.program))
            if message.type == "note_on" and message.velocity > 0:
                note_count += 1
        track_rows.append(
            {
                "source_track_index": track_index,
                "names": names,
                "channels": sorted(channels),
                "programs": sorted(programs),
                "drum_flags": sorted({channel == 9 for channel in channels}),
                "note_on_count": note_count,
                "end_tick": tick,
                "empty_of_note_ons": note_count == 0,
            }
        )
    tempo, source_end_tick = _tempo_map(midi_file)
    duration_seconds = _seconds_at_tick(Fraction(source_end_tick), tempo, midi_file.ticks_per_beat)
    return {
        "midi_type": int(midi_file.type),
        "ppqn": int(midi_file.ticks_per_beat),
        "source_track_count": len(midi_file.tracks),
        "source_note_on_count": sum(row["note_on_count"] for row in track_rows),
        "source_duration_ticks": source_end_tick,
        "source_duration_seconds": float(duration_seconds),
        "tempo_event_count": tempo_count,
        "tempo_values_us_per_qn": tempo_values,
        "meter_event_count": len(meter_values),
        "meter_values": meter_values,
        "tracks": track_rows,
        "_tempo_map": tempo,
        "_duration_seconds_fraction": duration_seconds,
    }


def _exception_category(exc: BaseException) -> tuple[str, str]:
    exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
    message = str(exc).lower()
    if isinstance(exc, MidiAdapterError):
        patterns = (
            ("corrupted or unreadable", "midi_adapter.corrupt_or_unreadable"),
            ("midi type 2", "midi_adapter.type_2"),
            ("smpte/non-ppqn", "midi_adapter.smpte"),
            ("invalid meter", "midi_adapter.invalid_meter"),
            ("inside a bar", "midi_adapter.meter_change_inside_bar"),
            ("metric-grid safety", "midi_adapter.metric_grid_safety"),
            ("canonical validation failed", "midi_adapter.canonical_validation"),
        )
        for needle, category in patterns:
            if needle in message:
                return exception_type, category
        return exception_type, "midi_adapter.other"
    return exception_type, "unexpected_exception"


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda value: str(value))}


def _short_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _annotation_times(family: str, records: Sequence[Mapping[str, Any]]) -> list[Decimal]:
    if family.startswith("beat_"):
        return [record["time_seconds"] for record in records]
    return [
        value
        for record in records
        for value in (record["start_seconds"], record["end_seconds"])
    ]


def _record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(f"{key}={record[key]}" for key in sorted(record))


def _read_annotation_file(
    family: str,
    path: Path,
    *,
    relative_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    delimiters: Counter[str] = Counter()
    try:
        handle = path.open("r", encoding="utf-8", errors="strict", newline=None)
    except (OSError, UnicodeError) as exc:
        return [], {
            "path": relative_path,
            "status": "failed",
            "line_count": 0,
        }, [{
            "path": relative_path,
            "line": None,
            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": _short_error(exc),
        }]
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                failures.append({
                    "path": relative_path,
                    "line": line_number,
                    "exception_type": f"{Pop909AuditError.__module__}.{Pop909AuditError.__name__}",
                    "message": f"{family} line {line_number}: empty line",
                })
                continue
            delimiters["tab" if "\t" in line else "whitespace"] += 1
            try:
                records.append(parse_annotation_line(family, line, line_number))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                failures.append({
                    "path": relative_path,
                    "line": line_number,
                    "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "message": _short_error(exc),
                })

    duplicate_count = sum(count - 1 for count in Counter(_record_key(row) for row in records).values())
    if family.startswith("beat_"):
        starts = [row["time_seconds"] for row in records]
        ends = starts
    else:
        starts = [row["start_seconds"] for row in records]
        ends = [row["end_seconds"] for row in records]
    non_monotonic = sum(current < previous for previous, current in zip(starts, starts[1:]))
    overlap_count = 0
    gap_count = 0
    if not family.startswith("beat_"):
        for previous_end, current_start in zip(ends, starts[1:]):
            overlap_count += current_start < previous_end
            gap_count += current_start > previous_end
    labels = [row["label"] for row in records if "label" in row]
    return records, {
        "path": relative_path,
        "status": "ok" if not failures else "partial",
        "encoding": "utf-8",
        "delimiter_lines": _counter(delimiters),
        "line_count": len(records) + len(failures),
        "parsed_count": len(records),
        "failure_count": len(failures),
        "duplicate_record_count": duplicate_count,
        "non_monotonic_start_count": non_monotonic,
        "overlap_count": overlap_count,
        "gap_count": gap_count,
        "minimum_seconds": float(min(starts)) if starts else None,
        "maximum_seconds": float(max(ends)) if ends else None,
        "special_tokens": _counter(Counter(label for label in labels if label in {"N", ""})),
    }, failures


def _add_nearest_comparison(
    accumulator: _TimingAccumulator,
    observed_values: Iterable[Decimal],
    canonical_seconds: Sequence[Fraction],
    duration_seconds: Fraction,
) -> float:
    maximum = 0.0
    tolerance = _fraction_from_decimal(ALIGNMENT_TOLERANCE_SECONDS)
    for value in observed_values:
        observed = _fraction_from_decimal(value)
        signed, _ = _nearest(canonical_seconds, observed)
        accumulator.add(
            signed,
            matched=abs(signed) <= tolerance,
            outside=observed < 0 or observed > duration_seconds,
        )
        maximum = max(maximum, float(abs(signed)))
    return maximum


def _add_view_comparison(
    accumulator: _TimingAccumulator,
    left: Sequence[Decimal],
    right: Sequence[Decimal],
) -> tuple[int, int]:
    left_values = [_fraction_from_decimal(value) for value in left]
    right_values = [_fraction_from_decimal(value) for value in right]
    tolerance = _fraction_from_decimal(ALIGNMENT_TOLERANCE_SECONDS)
    i = 0
    j = 0
    left_unmatched = 0
    right_unmatched = 0
    while i < len(left_values) and j < len(right_values):
        signed = left_values[i] - right_values[j]
        if abs(signed) <= tolerance:
            accumulator.add(signed, matched=True, outside=False)
            i += 1
            j += 1
        elif left_values[i] < right_values[j]:
            left_unmatched += 1
            i += 1
        else:
            right_unmatched += 1
            j += 1
    left_unmatched += len(left_values) - i
    right_unmatched += len(right_values) - j
    return left_unmatched, right_unmatched


def _role_evidence(raw: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    found: dict[str, list[int]] = defaultdict(list)
    observed_names: list[dict[str, Any]] = []
    for row in raw["tracks"]:
        for name in row["names"]:
            normalized = name.strip().upper()
            observed_names.append({
                "source_track_index": row["source_track_index"],
                "raw_name": name,
                "normalized_name": normalized,
            })
            if normalized in EXPECTED_ROLE_NAMES:
                found[normalized].append(row["source_track_index"])
    stable = all(len(found[name]) == 1 for name in EXPECTED_ROLE_NAMES)
    return stable, {
        "observed_names": observed_names,
        "resolved": {
            EXPECTED_ROLE_NAMES[name]: indices[0]
            for name, indices in sorted(found.items())
            if len(indices) == 1
        },
        "missing": sorted(name for name in EXPECTED_ROLE_NAMES if not found[name]),
        "ambiguous": {
            name: indices for name, indices in sorted(found.items()) if len(indices) > 1
        },
    }


def _corpus_fingerprint(file_rows: Sequence[Mapping[str, Any]]) -> str:
    digest = sha256()
    for row in file_rows:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _midi_inventory_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "converted"]
    warning_counts = [row["warning_count"] for row in successful]
    note_counts = [row["canonical_note_count"] for row in successful]
    track_counts = [row["canonical_track_count"] for row in successful]
    source_track_counts = [row["source_track_count"] for row in successful]
    durations_qn = [row["duration_qn_float"] for row in successful]
    durations_seconds = [row["source_duration_seconds"] for row in successful]
    return {
        "warnings_per_file": summarize_distribution(warning_counts),
        "canonical_note_count": summarize_distribution(note_counts),
        "canonical_track_count": summarize_distribution(track_counts),
        "source_track_count": summarize_distribution(source_track_counts),
        "duration_qn": summarize_distribution(durations_qn),
        "duration_seconds": summarize_distribution(durations_seconds),
    }


def _select_golden_cases(
    midi_rows: Sequence[Mapping[str, Any]],
    songs: Mapping[str, SongAssets],
    file_hashes: Mapping[str, str],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    converted = [row for row in midi_rows if row["status"] == "converted"]
    reasons_by_song: dict[str, set[str]] = defaultdict(set)
    if not converted:
        return []

    role_rows = [row for row in converted if row["official_roles_resolved"]]
    if role_rows:
        reasons_by_song[min(role_rows, key=lambda row: row["song_id"])["song_id"]].add(
            "ordinary_official_role_tracks"
        )
    for field, reason in (
        ("tempo_event_count", "tempo_changes"),
        ("key_annotation_count", "key_changes"),
        ("no_chord_count", "no_chord_regions"),
        ("alternative_version_count", "alternative_versions"),
        ("source_track_count", "unusual_track_configuration"),
        ("maximum_alignment_error_seconds", "alignment_extreme"),
    ):
        candidate = max(converted, key=lambda row: (row.get(field, 0), row["song_id"]))
        if candidate.get(field, 0):
            reasons_by_song[candidate["song_id"]].add(reason)

    complex_rows = [row for row in converted if row.get("complex_chord_count", 0)]
    if complex_rows:
        candidate = max(
            complex_rows,
            key=lambda row: (row["complex_chord_count"], row["song_id"]),
        )
        reasons_by_song[candidate["song_id"]].add("complex_or_rare_chord_labels")

    warning_codes = sorted({code for row in converted for code in row["warning_codes"]})
    for code in warning_codes:
        candidate = max(
            converted,
            key=lambda row: (row["warning_codes"].get(code, 0), row["song_id"]),
        )
        if candidate["warning_codes"].get(code, 0):
            reasons_by_song[candidate["song_id"]].add(f"warning:{code}")

    cases: list[dict[str, Any]] = []
    rows_by_song = {row["song_id"]: row for row in converted}
    for song_id in sorted(reasons_by_song):
        row = rows_by_song[song_id]
        song = songs[song_id]
        relative = _relative(song.primary_midi, root)
        cases.append({
            "song_id": song_id,
            "source_group_id": propose_source_group_id(song_id),
            "primary_midi": relative,
            "sha256": file_hashes[relative],
            "reasons": sorted(reasons_by_song[song_id]),
            "expected": {
                "canonical_note_count": row["canonical_note_count"],
                "canonical_track_count": row["canonical_track_count"],
                "source_track_count": row["source_track_count"],
                "warning_codes": row["warning_codes"],
                "tempo_event_count": row["tempo_event_count"],
                "alternative_version_count": row["alternative_version_count"],
                "key_annotation_count": row.get("key_annotation_count", 0),
                "no_chord_count": row.get("no_chord_count", 0),
            },
        })
    return cases


def _compare_chord_views(
    audio_records: Sequence[Mapping[str, Any]],
    midi_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    tolerance = ALIGNMENT_TOLERANCE_SECONDS
    i = 0
    j = 0
    matched = 0
    labels_equal = 0
    labels_different = 0
    while i < len(audio_records) and j < len(midi_records):
        audio = audio_records[i]
        midi = midi_records[j]
        start_error = audio["start_seconds"] - midi["start_seconds"]
        end_error = audio["end_seconds"] - midi["end_seconds"]
        if abs(start_error) <= tolerance and abs(end_error) <= tolerance:
            matched += 1
            if audio["label"] == midi["label"]:
                labels_equal += 1
            else:
                labels_different += 1
            i += 1
            j += 1
        elif audio["start_seconds"] < midi["start_seconds"]:
            i += 1
        else:
            j += 1
    return {
        "matched_segments_with_100ms_endpoint_tolerance": matched,
        "matched_labels_equal": labels_equal,
        "matched_labels_different": labels_different,
        "audio_unmatched_segments": len(audio_records) - matched,
        "midi_unmatched_segments": len(midi_records) - matched,
    }


def _complex_chord(parsed: Mapping[str, Any]) -> bool:
    if parsed.get("status") != "parsed":
        return False
    return bool(
        parsed.get("extensions")
        or parsed.get("alterations")
        or parsed.get("suspensions")
        or parsed.get("bass")
        or parsed.get("quality") not in {"maj", "min"}
    )


def build_report(
    root: str | PathLike[str],
    *,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Build the complete deterministic JSON-compatible audit report."""

    discovery = discover_dataset(root)
    selected_songs = _select_spread(discovery.songs, sample_size)
    selected_song_ids = {song.song_id for song in selected_songs}
    file_rows: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    hash_duplicates: dict[str, list[str]] = defaultdict(list)
    for path in discovery.files:
        relative = _relative(path, discovery.root)
        digest = _hash_file(path)
        file_hashes[relative] = digest
        hash_duplicates[digest].append(relative)
        file_rows.append({
            "path": relative,
            "kind": _file_kind(path, discovery),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        })

    missing_assets: list[dict[str, str]] = []
    for song in discovery.songs:
        if not song.primary_midi.is_file():
            missing_assets.append({
                "song_id": song.song_id,
                "kind": "primary_midi",
                "path": _relative(song.primary_midi, discovery.root),
            })
        if discovery.layout == "official_song_directories":
            for family, path in song.annotations:
                if not path.is_file():
                    missing_assets.append({
                        "song_id": song.song_id,
                        "kind": f"annotation.{family}",
                        "path": _relative(path, discovery.root),
                    })

    midi_rows: list[dict[str, Any]] = []
    midi_failures: list[dict[str, Any]] = []
    failure_types: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    warning_files: Counter[str] = Counter()
    midi_types: Counter[int] = Counter()
    ppqns: Counter[int] = Counter()
    tempo_event_counts: Counter[int] = Counter()
    meter_event_counts: Counter[int] = Counter()
    tempo_values: Counter[int] = Counter()
    meter_values: Counter[str] = Counter()
    track_names: Counter[str] = Counter()
    source_track_indices: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    programs: Counter[int] = Counter()
    drum_flags: Counter[bool] = Counter()
    raw_empty_tracks = 0
    role_primary_attempted = 0
    role_primary_resolved = 0
    role_primary_exceptions: list[dict[str, Any]] = []
    role_alternative_resolved = 0
    role_alternative_attempted = 0
    role_alternative_failures: list[dict[str, Any]] = []

    annotation_file_rows: list[dict[str, Any]] = []
    annotation_failures: list[dict[str, Any]] = []
    annotation_family_files: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    annotation_family_records: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    annotation_duplicates: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    annotation_non_monotonic: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    annotation_overlaps: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    annotation_gaps: dict[str, int] = {family: 0 for family in ANNOTATION_FAMILIES}
    chord_labels_by_view: dict[str, Counter[str]] = {
        "audio": Counter(),
        "midi": Counter(),
    }
    key_labels: Counter[str] = Counter()
    chord_roots: Counter[str] = Counter()
    chord_basses: Counter[str] = Counter()
    chord_qualities: Counter[str] = Counter()
    chord_extensions: Counter[str] = Counter()
    chord_alterations: Counter[str] = Counter()
    chord_suspensions: Counter[str] = Counter()
    chord_parse_failures: Counter[str] = Counter()
    chord_parsed = 0
    chord_total = 0
    key_parse_failures: Counter[str] = Counter()
    key_parsed = 0
    key_total = 0
    timing_by_family = {family: _TimingAccumulator.create() for family in ANNOTATION_FAMILIES}
    beat_view_timing = _TimingAccumulator.create()
    chord_view_timing = _TimingAccumulator.create()
    beat_view_left_unmatched = 0
    beat_view_right_unmatched = 0
    chord_view_left_unmatched = 0
    chord_view_right_unmatched = 0
    chord_view_labels = Counter()
    tempo_change_timing = _TimingAccumulator.create()

    round_trip_songs = {
        song.song_id
        for song in _select_spread(selected_songs, min(DEFAULT_ROUND_TRIP_SAMPLE_SIZE, len(selected_songs)))
    }
    round_trip_results: list[dict[str, Any]] = []

    for song in selected_songs:
        relative_midi = _relative(song.primary_midi, discovery.root)
        row: dict[str, Any] = {
            "song_id": song.song_id,
            "source_group_id": propose_source_group_id(song.song_id),
            "path": relative_midi,
            "status": "failed",
            "alternative_version_count": len(song.alternatives),
            "key_annotation_count": 0,
            "no_chord_count": 0,
            "complex_chord_count": 0,
            "maximum_alignment_error_seconds": 0.0,
        }
        raw: dict[str, Any] | None = None
        piece = None
        validation = None
        try:
            raw = _raw_midi_evidence(song.primary_midi)
            row.update({
                "midi_type": raw["midi_type"],
                "ppqn": raw["ppqn"],
                "source_track_count": raw["source_track_count"],
                "source_note_on_count": raw["source_note_on_count"],
                "source_duration_ticks": raw["source_duration_ticks"],
                "source_duration_seconds": raw["source_duration_seconds"],
                "raw_tempo_event_count": raw["tempo_event_count"],
                "raw_meter_event_count": raw["meter_event_count"],
                "raw_tracks": raw["tracks"],
            })
            midi_types[raw["midi_type"]] += 1
            ppqns[raw["ppqn"]] += 1
            tempo_values.update(raw["tempo_values_us_per_qn"])
            meter_values.update(raw["meter_values"])
            resolved, role = _role_evidence(raw)
            role_primary_attempted += 1
            role_primary_resolved += resolved
            row["official_roles_resolved"] = resolved
            row["role_evidence"] = role
            if not resolved:
                role_primary_exceptions.append({
                    "song_id": song.song_id,
                    "path": relative_midi,
                    **role,
                })
            for track in raw["tracks"]:
                raw_empty_tracks += track["empty_of_note_ons"]
                source_track_indices[track["source_track_index"]] += 1
                for name in track["names"]:
                    track_names[name] += 1
                channels.update(track["channels"])
                programs.update(track["programs"])
                drum_flags.update(track["drum_flags"])

            piece = load_midi_piece(
                str(song.primary_midi),
                config=MidiAdapterConfig(
                    dataset_name="pop909_original",
                    source_group_id=propose_source_group_id(song.song_id),
                ),
            )
            validation = validate_piece(piece)
            if validation.errors:
                raise RuntimeError(
                    f"adapter returned {len(validation.errors)} canonical validation errors"
                )
            row.update({
                "status": "converted",
                "canonical_track_count": len(piece.tracks),
                "canonical_note_count": len(piece.notes),
                "duration_qn": f"{piece.duration_qn.num}/{piece.duration_qn.den}",
                "duration_qn_float": float(piece.duration_qn.to_fraction()),
                "tempo_event_count": len(piece.tempo_events),
                "meter_event_count": len(piece.meter_events),
            })
            codes = Counter(flag.code for flag in piece.quality_flags)
            codes.update(issue.code for issue in validation.warnings)
            row["warning_codes"] = _counter(codes)
            row["warning_count"] = sum(codes.values())
            warning_counts.update(codes)
            warning_files.update(codes.keys())
            tempo_event_counts[len(piece.tempo_events)] += 1
            meter_event_counts[len(piece.meter_events)] += 1

            if song.song_id in round_trip_songs:
                round_trip_results.append({
                    "song_id": song.song_id,
                    "path": relative_midi,
                    "equal": loads_piece(dumps_piece(piece)) == piece,
                })
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            exception_type, category = _exception_category(exc)
            failure_types[exception_type] += 1
            failure_categories[category] += 1
            failure = {
                "song_id": song.song_id,
                "path": relative_midi,
                "exception_type": exception_type,
                "category": category,
                "message": _short_error(exc),
            }
            midi_failures.append(failure)
            row["failure"] = failure
        midi_rows.append(row)

        records_by_family: dict[str, list[dict[str, Any]]] = {}
        for family, annotation_path in song.annotations:
            if not annotation_path.is_file():
                continue
            relative_annotation = _relative(annotation_path, discovery.root)
            records, file_summary, failures = _read_annotation_file(
                family,
                annotation_path,
                relative_path=relative_annotation,
            )
            records_by_family[family] = records
            annotation_file_rows.append(file_summary)
            annotation_failures.extend(failures)
            annotation_family_files[family] += 1
            annotation_family_records[family] += len(records)
            annotation_duplicates[family] += file_summary["duplicate_record_count"]
            annotation_non_monotonic[family] += file_summary["non_monotonic_start_count"]
            annotation_overlaps[family] += file_summary["overlap_count"]
            annotation_gaps[family] += file_summary["gap_count"]

            if family.startswith("chord_"):
                view = family.removeprefix("chord_")
                for record in records:
                    label = record["label"]
                    chord_labels_by_view[view][label] += 1
                    chord_total += 1
                    if label == "N":
                        row["no_chord_count"] += 1
                    parsed = parse_chord_label(label)
                    if parsed["lossless"]:
                        chord_parsed += 1
                        if parsed["root"] is not None:
                            chord_roots[parsed["root"]] += 1
                        if parsed["bass"] is not None:
                            chord_basses[parsed["bass"]] += 1
                        if parsed["quality"] is not None:
                            chord_qualities[parsed["quality"]] += 1
                        chord_extensions.update(parsed["extensions"])
                        chord_alterations.update(parsed["alterations"])
                        chord_suspensions.update(parsed["suspensions"])
                        row["complex_chord_count"] += _complex_chord(parsed)
                    else:
                        chord_parse_failures[label] += 1
            elif family == "key_audio":
                row["key_annotation_count"] = len(records)
                for record in records:
                    label = record["label"]
                    key_labels[label] += 1
                    key_total += 1
                    parsed = parse_key_label(label)
                    if parsed["lossless"]:
                        key_parsed += 1
                    else:
                        key_parse_failures[label] += 1

            if raw is not None and piece is not None:
                canonical_seconds = sorted({
                    _seconds_at_tick(
                        beat.start_qn.to_fraction() * raw["ppqn"],
                        raw["_tempo_map"],
                        raw["ppqn"],
                    )
                    for beat in piece.beats
                })
                maximum = _add_nearest_comparison(
                    timing_by_family[family],
                    _annotation_times(family, records),
                    canonical_seconds,
                    raw["_duration_seconds_fraction"],
                )
                row["maximum_alignment_error_seconds"] = max(
                    row["maximum_alignment_error_seconds"], maximum
                )
                if len(piece.tempo_events) > 1:
                    _add_nearest_comparison(
                        tempo_change_timing,
                        _annotation_times(family, records),
                        canonical_seconds,
                        raw["_duration_seconds_fraction"],
                    )

        if "beat_audio" in records_by_family and "beat_midi" in records_by_family:
            left, right = _add_view_comparison(
                beat_view_timing,
                _annotation_times("beat_audio", records_by_family["beat_audio"]),
                _annotation_times("beat_midi", records_by_family["beat_midi"]),
            )
            beat_view_left_unmatched += left
            beat_view_right_unmatched += right
        if "chord_audio" in records_by_family and "chord_midi" in records_by_family:
            left, right = _add_view_comparison(
                chord_view_timing,
                _annotation_times("chord_audio", records_by_family["chord_audio"]),
                _annotation_times("chord_midi", records_by_family["chord_midi"]),
            )
            chord_view_left_unmatched += left
            chord_view_right_unmatched += right
            chord_view_labels.update(
                _compare_chord_views(
                    records_by_family["chord_audio"],
                    records_by_family["chord_midi"],
                )
            )

        for alternative in song.alternatives:
            role_alternative_attempted += 1
            try:
                alternative_raw = _raw_midi_evidence(alternative)
                resolved, role = _role_evidence(alternative_raw)
                role_alternative_resolved += resolved
                if not resolved:
                    role_alternative_failures.append({
                        "song_id": song.song_id,
                        "path": _relative(alternative, discovery.root),
                        "kind": "unresolved_role_mapping",
                        **role,
                    })
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                role_alternative_failures.append({
                    "song_id": song.song_id,
                    "path": _relative(alternative, discovery.root),
                    "kind": "midi_parse_failure",
                    "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
                    "message": _short_error(exc),
                })

    chord_combined = chord_labels_by_view["audio"] + chord_labels_by_view["midi"]
    exact_duplicate_hashes = [
        {"sha256": digest, "paths": sorted(paths)}
        for digest, paths in sorted(hash_duplicates.items())
        if len(paths) > 1
    ]
    songs_by_id = {song.song_id: song for song in selected_songs}
    golden_cases = _select_golden_cases(
        midi_rows,
        songs_by_id,
        file_hashes,
        root=discovery.root,
    )

    beat_view_summary = beat_view_timing.summary()
    beat_view_summary.update({
        "audio_unmatched": beat_view_left_unmatched,
        "midi_unmatched": beat_view_right_unmatched,
    })
    chord_view_summary = chord_view_timing.summary()
    chord_view_summary.update({
        "audio_boundary_unmatched": chord_view_left_unmatched,
        "midi_boundary_unmatched": chord_view_right_unmatched,
        "segment_label_comparison": _counter(chord_view_labels),
    })

    full_official_assets = discovery.layout == "official_song_directories"
    role_stable = (
        role_primary_resolved == role_primary_attempted == len(selected_songs)
        and role_alternative_resolved == role_alternative_attempted
    )
    strict_violations: list[str] = []
    if missing_assets:
        strict_violations.append("missing_contract_assets")
    if discovery.duplicate_song_ids:
        strict_violations.append("duplicate_song_ids")
    if discovery.duplicate_version_ids:
        strict_violations.append("duplicate_version_ids")
    if midi_failures:
        strict_violations.append("primary_midi_conversion_failures")
    if annotation_failures:
        strict_violations.append("annotation_parser_failures")
    if not all(result["equal"] for result in round_trip_results):
        strict_violations.append("serialization_round_trip_failures")
    if full_official_assets and not role_stable:
        strict_violations.append("unstable_official_track_roles")
    if not full_official_assets:
        strict_violations.append("not_official_full_annotation_layout")

    top_level_counts = Counter(
        _relative(path, discovery.root).split("/", 1)[0] for path in discovery.files
    )
    git_identity = _git_identity(discovery)
    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "evidence_policy": {
            "local_measurement": "all corpus facts are measured from --root",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_paper": OFFICIAL_PAPER,
            "alignment_tolerance_seconds": str(ALIGNMENT_TOLERANCE_SECONDS),
            "annotation_interval_boundary": "source_undocumented; Phase 4B proposal is half-open [start,end)",
            "seconds_mapping": "Decimal source seconds; exact Fraction tempo integration; no rounding to beats",
        },
        "identity": {
            "corpus_id": "pop909_original",
            "layout": discovery.layout,
            "root_basename": discovery.root.name,
            "corpus_root_relative": _relative(discovery.corpus_root, discovery.root)
            if discovery.corpus_root != discovery.root
            else ".",
            "git": git_identity,
            "song_count": len(discovery.songs),
            "selected_song_count": len(selected_songs),
            "sample_size": sample_size,
            "primary_midi_count": sum(song.primary_midi.is_file() for song in discovery.songs),
            "alternative_midi_count": sum(len(song.alternatives) for song in discovery.songs),
            "annotation_file_counts": {
                family: sum(path.is_file() for song in discovery.songs for name, path in song.annotations if name == family)
                for family in ANNOTATION_FAMILIES
            },
            "file_count": len(file_rows),
            "corpus_fingerprint_sha256": _corpus_fingerprint(file_rows),
        },
        "discovery": {
            "top_level_file_membership_counts": _counter(top_level_counts),
            "song_ids": [song.song_id for song in discovery.songs],
            "source_groups": [
                {"song_id": song.song_id, "source_group_id": propose_source_group_id(song.song_id)}
                for song in discovery.songs
            ],
            "missing_assets": missing_assets,
            "unexpected_assets": [_relative(path, discovery.root) for path in discovery.unexpected],
            "duplicate_song_ids": [
                {"song_id": song_id, "paths": [_relative(path, discovery.root) for path in paths]}
                for song_id, paths in discovery.duplicate_song_ids
            ],
            "duplicate_version_ids": [
                {"version_id": version_id, "paths": [_relative(path, discovery.root) for path in paths]}
                for version_id, paths in discovery.duplicate_version_ids
            ],
            "duplicate_file_hashes": exact_duplicate_hashes,
            "files": file_rows,
        },
        "generic_midi_crosswalk": {
            "discovered": len(discovery.songs),
            "attempted": len(selected_songs),
            "converted": sum(row["status"] == "converted" for row in midi_rows),
            "failed": len(midi_failures),
            "failures_by_exception_type": _counter(failure_types),
            "failures_by_category": _counter(failure_categories),
            "failures": midi_failures,
            "warnings_by_code": _counter(warning_counts),
            "files_affected_by_warning_code": _counter(warning_files),
            "warning_counts_per_file": [
                {"song_id": row["song_id"], "path": row["path"], "count": row.get("warning_count", 0)}
                for row in midi_rows
            ],
            "distributions": _midi_inventory_summary(midi_rows),
            "midi_type_distribution": _counter(midi_types),
            "ppqn_distribution": _counter(ppqns),
            "tempo_event_count_distribution": _counter(tempo_event_counts),
            "tempo_value_distribution_us_per_qn": _counter(tempo_values),
            "meter_event_count_distribution": _counter(meter_event_counts),
            "meter_value_distribution": _counter(meter_values),
            "track_evidence": {
                "track_names": _counter(track_names),
                "source_track_indices": _counter(source_track_indices),
                "channels": _counter(channels),
                "programs": _counter(programs),
                "drum_flags": _counter(drum_flags),
                "raw_empty_track_count": raw_empty_tracks,
            },
            "serialization_round_trip_sample": round_trip_results,
            "per_file": midi_rows,
            "warning_interpretation": {
                "unit": "event/entity-level warning occurrences, not failed files",
                "overlap_code": "OVERLAPPING_SAME_PITCH_NOTES is emitted once for each later overlapping same-pitch note on one canonical track",
                "processed_variant_risk": "a flattened processed mirror may contain a synthesized chords track and is not role-equivalent to official MELODY/BRIDGE/PIANO MIDI",
                "quality_rule": "warnings remain diagnostics; only explicit conversion failures are failed files",
            },
        },
        "track_role_evidence": {
            "documented_mapping": EXPECTED_ROLE_NAMES,
            "evidence_field": "case-normalized MIDI track_name; source track order is corroborating, not sole evidence",
            "primary_attempted": role_primary_attempted,
            "primary_resolved": role_primary_resolved,
            "alternative_attempted": role_alternative_attempted,
            "alternative_resolved": role_alternative_resolved,
            "stable_for_every_audited_primary_and_version": role_stable,
            "primary_exceptions": role_primary_exceptions,
            "alternative_exceptions": role_alternative_failures,
        },
        "annotations": {
            "families": {
                family: {
                    "files": annotation_family_files[family],
                    "records": annotation_family_records[family],
                    "duplicate_records": annotation_duplicates[family],
                    "non_monotonic_starts": annotation_non_monotonic[family],
                    "overlaps": annotation_overlaps[family],
                    "gaps": annotation_gaps[family],
                    "time_coordinate": "seconds",
                    "columns": {
                        "beat_audio": ["time_seconds", "beat_order"],
                        "beat_midi": ["time_seconds", "downbeat_simple", "downbeat_compound"],
                        "chord_audio": ["start_seconds", "end_seconds", "label"],
                        "chord_midi": ["start_seconds", "end_seconds", "label"],
                        "key_audio": ["start_seconds", "end_seconds", "label"],
                    }[family],
                }
                for family in ANNOTATION_FAMILIES
            },
            "file_summaries": annotation_file_rows,
            "parser_failure_count": len(annotation_failures),
            "parser_failures": annotation_failures,
            "canonical_beat_alignment": {
                family: timing_by_family[family].summary() for family in ANNOTATION_FAMILIES
            },
            "audio_midi_view_alignment": {
                "beats": beat_view_summary,
                "chords": chord_view_summary,
            },
            "tempo_change_piece_alignment": tempo_change_timing.summary(),
        },
        "vocabularies": {
            "chords": {
                "audio_labels": _counter(chord_labels_by_view["audio"]),
                "midi_labels": _counter(chord_labels_by_view["midi"]),
                "combined_labels": _counter(chord_combined),
                "total": chord_total,
                "losslessly_parsed": chord_parsed,
                "lossless_coverage": chord_parsed / chord_total if chord_total else None,
                "unparsed_labels": _counter(chord_parse_failures),
                "no_chord_tokens": {"N": chord_combined["N"]},
                "roots": _counter(chord_roots),
                "bass_intervals": _counter(chord_basses),
                "qualities": _counter(chord_qualities),
                "extensions": _counter(chord_extensions),
                "alterations": _counter(chord_alterations),
                "suspensions": _counter(chord_suspensions),
            },
            "keys": {
                "labels": _counter(key_labels),
                "total": key_total,
                "losslessly_parsed": key_parsed,
                "lossless_coverage": key_parsed / key_total if key_total else None,
                "unparsed_labels": _counter(key_parse_failures),
                "pieces_with_key_changes": sum(row.get("key_annotation_count", 0) > 1 for row in midi_rows),
            },
        },
        "grouping": {
            "policy": "one pop909-original:<three-digit-song-id> group for primary MIDI, every annotation, and every alternative version",
            "final_splits_assigned": False,
            "deterministic_split_interface": "split_source_groups(group_ids, seed, ratios) -> mapping[group_id, split]; sort groups before seeded assignment",
            "duplicate_song_identifier_count": len(discovery.duplicate_song_ids),
            "duplicate_version_identifier_count": len(discovery.duplicate_version_ids),
            "duplicate_content_group_count": len(exact_duplicate_hashes),
        },
        "provenance_assessment": {
            "midi_arrangements": {
                "source": "human",
                "confidence": None,
                "evidence": "professional arranger/reviewer process; primary files are final qualified versions",
            },
            "tempo_curve": {
                "source": "human",
                "confidence": None,
                "evidence": "paper describes manually labeled tempo curves used for audio alignment",
            },
            "beat_audio": {"source": "algorithm", "confidence": None},
            "beat_midi": {"source": "algorithm", "confidence": None},
            "chord_audio": {"source": "algorithm", "confidence": None},
            "chord_midi": {"source": "algorithm", "confidence": None},
            "key_audio": {"source": "algorithm", "confidence": None},
            "role_targets": {
                "source": "dataset",
                "confidence": None,
                "policy": "available only when exact documented track-name evidence resolves uniquely",
            },
        },
        "golden_evidence": golden_cases,
        "strict": {
            "contract_ready": not strict_violations,
            "violations": strict_violations,
        },
    }
    return report


def dumps_report(report: Mapping[str, Any]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_report(report), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=_positive_int)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        ensure_output_outside_root(args.root, args.output)
        report = build_report(args.root, sample_size=args.sample_size)
        write_report(report, args.output)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"POP909 audit failed: {type(exc).__name__}: {_short_error(exc)}", file=sys.stderr)
        return 2
    if args.strict and not report["strict"]["contract_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
