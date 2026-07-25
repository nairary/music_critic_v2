"""Production POP909-CL adapter with a strict raw/target boundary."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from os import PathLike
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator, Literal, Sequence

import mido

from music_critic.adapters.midi import (
    MidiAdapterConfig,
    MidiAdapterError,
    load_midi_piece,
)
from music_critic.data import (
    AnnotationSpan,
    CanonicalPiece,
    ProvenanceRecord,
    QualityFlag,
    RationalTime,
    TargetArray,
    ValidationReport,
    validate_piece,
)


POP909_CL_ADAPTER_VERSION = "1.0.0"
POP909_CL_CORPUS_MANIFEST_VERSION = "1.0.0"
POP909_CL_DATASET_NAME = "pop909_cl"
POP909_CL_UPSTREAM_REPOSITORY = (
    "https://github.com/AndyWeasley2004/POP909-CL-Dataset"
)
POP909_CL_UPSTREAM_COMMIT = "be9094392903c471a930519e1c0bacf8b6be5d62"
POP909_CL_UPSTREAM_LICENSE = "MIT"
POP909_CL_UPSTREAM_LICENSE_SHA256 = (
    "fe6064d631bdf4ce46028ef3aa7bc4eac285b8a1000c46682795f26448d29288"
)
POP909_CL_CONTENT_FINGERPRINT = (
    "b34f07d9a2678abdb6f0dcf5db1c3aec3f35caca813f1fac80c0717cfc8e0c65"
)
POP909_CL_ANOMALY_FINGERPRINT = (
    "d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051"
)
POP909_CL_EXPECTED_SONG_IDS = tuple(f"{value:03d}" for value in range(1, 910))
POP909_CL_EXPECTED_MISSING_TARGET_IDS = frozenset({"367", "658"})
POP909_CL_QUARANTINE_IDS = frozenset({"172"})

POP909_CL_TASK_BOUNDARY = "pop909_cl.chord.boundary"
POP909_CL_TASK_ROOT = "pop909_cl.chord.root"
POP909_CL_TASK_QUALITY = "pop909_cl.chord.quality"
POP909_CL_TASK_BASS = "pop909_cl.chord.bass"
POP909_CL_TASK_INVERSION = "pop909_cl.chord.inversion"
POP909_CL_TASK_NO_CHORD = "pop909_cl.chord.no_chord"
POP909_CL_TASKS = (
    POP909_CL_TASK_BASS,
    POP909_CL_TASK_BOUNDARY,
    POP909_CL_TASK_INVERSION,
    POP909_CL_TASK_NO_CHORD,
    POP909_CL_TASK_QUALITY,
    POP909_CL_TASK_ROOT,
)

_SCORE_CHANNEL = 0
_CHORD_CHANNEL = 1
_SONG_ID_RE = re.compile(r"^[0-9]{3}$")
_MIDI_SUFFIXES = frozenset({".mid", ".midi"})
_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
_QUALITIES = (
    "M",
    "m",
    "o",
    "+",
    "sus2",
    "sus4",
    "D7",
    "M7",
    "m7",
    "/o7",
    "o7",
    "mM7",
    "+7",
)
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


class Pop909ClAdapterError(Exception):
    """Base class for structured POP909-CL production failures."""

    category: str
    song_id: str | None

    def __init__(
        self,
        message: str,
        *,
        category: str,
        song_id: str | None = None,
    ) -> None:
        self.category = category
        self.song_id = song_id
        super().__init__(message)


class Pop909ClCorpusIdentityError(Pop909ClAdapterError):
    """Raised when corpus discovery does not match its pinned identity."""

    discovery: Pop909ClCorpusDiscovery

    def __init__(self, discovery: Pop909ClCorpusDiscovery) -> None:
        self.discovery = discovery
        categories = ", ".join(issue.category for issue in discovery.issues)
        super().__init__(
            f"POP909-CL corpus identity failed: {categories}",
            category="pop909_cl.corpus_identity",
        )


class Pop909ClConversionError(Pop909ClAdapterError):
    """Raised when one source cannot be safely routed or converted."""


@dataclass(frozen=True, slots=True)
class Pop909ClCorpusIdentity:
    expected_song_ids: tuple[str, ...] = POP909_CL_EXPECTED_SONG_IDS
    expected_content_fingerprint: str = POP909_CL_CONTENT_FINGERPRINT


@dataclass(frozen=True, slots=True)
class Pop909ClAdapterConfig:
    include_targets: bool = True


@dataclass(frozen=True, slots=True)
class Pop909ClCorpusIssue:
    category: str
    song_id: str | None
    paths: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class Pop909ClCorpusRecord:
    song_id: str
    path: Path
    relative_path: str
    corpus_relative_path: str
    sha256: str
    source_group_id: str
    lineage_group_id: str
    dataset_name: str = POP909_CL_DATASET_NAME
    upstream_repository: str = POP909_CL_UPSTREAM_REPOSITORY
    upstream_commit: str = POP909_CL_UPSTREAM_COMMIT
    upstream_license: str = POP909_CL_UPSTREAM_LICENSE
    upstream_license_sha256: str = POP909_CL_UPSTREAM_LICENSE_SHA256


@dataclass(frozen=True, slots=True)
class Pop909ClCorpusDiscovery:
    root: Path
    corpus_root: Path
    records: tuple[Pop909ClCorpusRecord, ...]
    content_fingerprint: str
    issues: tuple[Pop909ClCorpusIssue, ...]
    noise_paths: tuple[str, ...]
    manifest_version: str = POP909_CL_CORPUS_MANIFEST_VERSION

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class Pop909ClTrackEvidence:
    source_track_index: int
    names: tuple[str, ...]
    event_channels: tuple[int, ...]
    note_on_channels: tuple[int, ...]
    positive_note_on_count: int
    end_tick: int
    global_meta_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pop909ClInstrumentResolution:
    tracks: tuple[Pop909ClTrackEvidence, ...]
    score_track_indices: tuple[int, ...]
    chord_track_indices: tuple[int, ...]
    metadata_track_indices: tuple[int, ...]
    unexpected_track_indices: tuple[int, ...]
    failure_categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pop909ClChordCandidate:
    root_pc: int
    root: str
    quality: str
    bass_pc: int
    bass: str
    inversion_semitones: int


@dataclass(frozen=True, slots=True)
class Pop909ClPairingAnomaly:
    anomaly_id: str
    category: Literal["dangling_note_on", "unmatched_note_off"]
    tick: int
    pitch: int
    velocity: int
    channel: int
    message_type: str
    ordinal: int
    source_track_index: int
    source_path: str
    source_sha256: str
    affected_block_onsets: tuple[int, ...]
    affected_span_ids: tuple[str, ...]
    affected_interval_start_tick: int
    affected_interval_end_tick: int
    affected_interval_basis: str


@dataclass(frozen=True, slots=True)
class Pop909ClChordBlock:
    block_id: str
    onset_tick: int
    end_tick: int
    ppqn: int
    midi_pitch_multiset: tuple[int, ...]
    note_end_ticks: tuple[int, ...]
    pitch_class_set: tuple[int, ...]
    lowest_source_pitch: int
    bass_pitch_class: int
    source_track_index: int
    source_channel: int
    source_track_names: tuple[str, ...]
    source_path: str
    source_sha256: str
    repeated_pitch: bool
    mixed_end_ticks: bool
    overlaps_previous: bool
    normalization_status: Literal["supported", "ambiguous", "unsupported"]
    candidates: tuple[Pop909ClChordCandidate, ...]
    selected_candidate: Pop909ClChordCandidate | None
    root_available: bool
    quality_available: bool
    inversion_available: bool
    pairing_anomaly_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pop909ClCoverageSpan:
    span_id: str
    kind: Literal[
        "leading_no_chord",
        "internal_no_chord",
        "trailing_unannotated",
        "missing_chord_targets",
    ]
    start_tick: int
    end_tick: int
    ppqn: int
    available: bool
    value: str | None
    pairing_anomaly_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pop909ClChordEvidence:
    status: Literal["available", "expected_missing"]
    ppqn: int
    blocks: tuple[Pop909ClChordBlock, ...]
    no_chord_spans: tuple[Pop909ClCoverageSpan, ...]
    trailing_spans: tuple[Pop909ClCoverageSpan, ...]
    pairing_anomalies: tuple[Pop909ClPairingAnomaly, ...]
    repeated_pitch_block_count: int
    mixed_end_block_count: int
    overlap_count: int


@dataclass(frozen=True, slots=True)
class Pop909ClAccepted:
    status: Literal["accepted"]
    record: Pop909ClCorpusRecord
    piece: CanonicalPiece
    chord_evidence: Pop909ClChordEvidence
    instrument_resolution: Pop909ClInstrumentResolution
    score_projection_sha256: str
    validation_report: ValidationReport


@dataclass(frozen=True, slots=True)
class Pop909ClExpectedTargetAbsence:
    status: Literal["accepted_missing_targets"]
    record: Pop909ClCorpusRecord
    piece: CanonicalPiece
    chord_evidence: Pop909ClChordEvidence
    instrument_resolution: Pop909ClInstrumentResolution
    score_projection_sha256: str
    validation_report: ValidationReport


@dataclass(frozen=True, slots=True)
class Pop909ClQuarantine:
    status: Literal["quarantined"]
    record: Pop909ClCorpusRecord
    category: str
    source_error_type: str
    source_error: str
    instrument_resolution: Pop909ClInstrumentResolution
    chord_evidence: Pop909ClChordEvidence
    score_projection_sha256: str


Pop909ClConversionResult = (
    Pop909ClAccepted | Pop909ClExpectedTargetAbsence | Pop909ClQuarantine
)


def pop909_cl_source_group_id(song_id: str) -> str:
    normalized = song_id.strip()
    if not _SONG_ID_RE.fullmatch(normalized):
        raise Pop909ClAdapterError(
            f"invalid POP909-CL song ID: {song_id!r}",
            category="pop909_cl.song_id_invalid",
            song_id=None,
        )
    return f"pop909-cl:{normalized}"


def pop909_lineage_group_id(song_id: str) -> str:
    normalized = song_id.strip()
    if not _SONG_ID_RE.fullmatch(normalized):
        raise Pop909ClAdapterError(
            f"invalid POP909 lineage song ID: {song_id!r}",
            category="pop909_cl.song_id_invalid",
            song_id=None,
        )
    return f"pop909-lineage:{normalized}"


def _is_noise(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _safe_files(root: Path) -> Iterator[Path]:
    resolved_root = root.resolve()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        directory_path = Path(directory)
        for filename in sorted(filenames):
            candidate = directory_path / filename
            try:
                candidate.resolve().relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            if candidate.is_file():
                yield candidate.resolve()


def _find_corpus_root(root: Path) -> Path:
    candidates = (
        root,
        root / "POP909_processed",
        root / "POP909_processed" / "POP909_processed",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(
            path.is_file()
            and path.suffix.lower() in _MIDI_SUFFIXES
            and not _is_noise(path)
            for path in candidate.iterdir()
        ):
            return candidate.resolve()
    raise Pop909ClAdapterError(
        "could not find direct or nested POP909_processed corpus",
        category="pop909_cl.corpus_root_missing",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_fingerprint(rows: Iterable[tuple[str, str]]) -> str:
    digest = sha256()
    for relative_path, checksum in sorted(rows):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def discover_pop909_cl_corpus(
    root: str | PathLike[str],
    *,
    identity: Pop909ClCorpusIdentity = Pop909ClCorpusIdentity(),
    require_valid: bool = True,
) -> Pop909ClCorpusDiscovery:
    """Discover and fingerprint the direct or nested POP909_processed corpus."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise Pop909ClAdapterError(
            f"POP909-CL root is not a directory: {root_path}",
            category="pop909_cl.root_invalid",
        )
    root_path = root_path.resolve()
    corpus_root = _find_corpus_root(root_path)
    files = tuple(
        sorted(
            _safe_files(root_path),
            key=lambda path: path.relative_to(root_path).as_posix(),
        )
    )
    noise = tuple(
        path.relative_to(root_path).as_posix() for path in files if _is_noise(path)
    )
    midi_paths = tuple(
        sorted(
            (
                path
                for path in corpus_root.iterdir()
                if path.is_file()
                and path.suffix.lower() in _MIDI_SUFFIXES
                and not _is_noise(path)
            ),
            key=lambda path: path.name,
        )
    )
    grouped: dict[str, list[Path]] = defaultdict(list)
    malformed: list[Path] = []
    for path in midi_paths:
        candidate = path.stem.strip()
        if _SONG_ID_RE.fullmatch(candidate):
            grouped[candidate].append(path)
        else:
            malformed.append(path)

    expected = tuple(identity.expected_song_ids)
    expected_set = set(expected)
    issues: list[Pop909ClCorpusIssue] = []
    if len(expected_set) != len(expected) or any(
        _SONG_ID_RE.fullmatch(value) is None for value in expected
    ):
        raise Pop909ClAdapterError(
            "expected_song_ids must contain unique three-digit IDs",
            category="pop909_cl.identity_config_invalid",
        )
    for path in malformed:
        relative = path.relative_to(root_path).as_posix()
        issues.append(
            Pop909ClCorpusIssue(
                category="malformed_midi_id",
                song_id=None,
                paths=(relative,),
                message=f"MIDI filename does not trim to three digits: {relative}",
            )
        )
    for song_id, paths in sorted(grouped.items()):
        if len(paths) > 1:
            relatives = tuple(
                path.relative_to(root_path).as_posix()
                for path in sorted(paths, key=lambda item: item.name)
            )
            issues.append(
                Pop909ClCorpusIssue(
                    category="duplicate_song_id",
                    song_id=song_id,
                    paths=relatives,
                    message=f"multiple MIDI files resolve to song {song_id}",
                )
            )
        if song_id not in expected_set:
            issues.append(
                Pop909ClCorpusIssue(
                    category="unexpected_song_id",
                    song_id=song_id,
                    paths=tuple(
                        path.relative_to(root_path).as_posix() for path in paths
                    ),
                    message=f"unexpected POP909-CL song ID {song_id}",
                )
            )
    for song_id in sorted(expected_set - set(grouped)):
        issues.append(
            Pop909ClCorpusIssue(
                category="missing_song_id",
                song_id=song_id,
                paths=(),
                message=f"missing POP909-CL song ID {song_id}",
            )
        )

    records: list[Pop909ClCorpusRecord] = []
    for song_id in sorted(expected_set & set(grouped)):
        paths = sorted(grouped[song_id], key=lambda item: item.name)
        if len(paths) != 1:
            continue
        path = paths[0]
        records.append(
            Pop909ClCorpusRecord(
                song_id=song_id,
                path=path.resolve(),
                relative_path=path.relative_to(root_path).as_posix(),
                corpus_relative_path=path.relative_to(corpus_root).as_posix(),
                sha256=_file_sha256(path),
                source_group_id=pop909_cl_source_group_id(song_id),
                lineage_group_id=pop909_lineage_group_id(song_id),
            )
        )
    fingerprint = _content_fingerprint(
        (record.corpus_relative_path, record.sha256) for record in records
    )
    if fingerprint != identity.expected_content_fingerprint:
        issues.append(
            Pop909ClCorpusIssue(
                category="content_fingerprint_mismatch",
                song_id=None,
                paths=(),
                message=(
                    f"expected {identity.expected_content_fingerprint}, "
                    f"observed {fingerprint}"
                ),
            )
        )
    discovery = Pop909ClCorpusDiscovery(
        root=root_path,
        corpus_root=corpus_root,
        records=tuple(records),
        content_fingerprint=fingerprint,
        issues=tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.category,
                    item.song_id or "",
                    item.paths,
                    item.message,
                ),
            )
        ),
        noise_paths=tuple(sorted(noise)),
    )
    if require_valid and not discovery.is_valid:
        raise Pop909ClCorpusIdentityError(discovery)
    return discovery


def inspect_pop909_cl_instruments(midi: mido.MidiFile) -> Pop909ClInstrumentResolution:
    """Resolve instruments only from channel-bearing MIDI events."""

    tracks: list[Pop909ClTrackEvidence] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        event_channels: set[int] = set()
        note_channels: set[int] = set()
        names: list[str] = []
        meta_types: list[str] = []
        note_count = 0
        for message in track:
            tick += int(message.time)
            if message.type == "track_name":
                names.append(message.name)
            channel = getattr(message, "channel", None)
            if isinstance(channel, int):
                event_channels.add(channel)
            if message.type == "note_on" and message.velocity > 0:
                note_channels.add(message.channel)
                note_count += 1
            if message.type in {"set_tempo", "time_signature", "key_signature"}:
                meta_types.append(message.type)
        tracks.append(
            Pop909ClTrackEvidence(
                source_track_index=track_index,
                names=tuple(names),
                event_channels=tuple(sorted(event_channels)),
                note_on_channels=tuple(sorted(note_channels)),
                positive_note_on_count=note_count,
                end_tick=tick,
                global_meta_types=tuple(meta_types),
            )
        )
    score = tuple(
        row.source_track_index
        for row in tracks
        if row.event_channels == (_SCORE_CHANNEL,) and row.positive_note_on_count > 0
    )
    chord = tuple(
        row.source_track_index
        for row in tracks
        if row.event_channels == (_CHORD_CHANNEL,)
    )
    metadata = tuple(
        row.source_track_index for row in tracks if not row.event_channels
    )
    unexpected = tuple(
        row.source_track_index
        for row in tracks
        if row.event_channels
        not in ((), (_SCORE_CHANNEL,), (_CHORD_CHANNEL,))
        or (
            row.event_channels == (_SCORE_CHANNEL,)
            and row.positive_note_on_count == 0
        )
    )
    failures: list[str] = []
    if len(score) == 0:
        failures.append("missing_score_instrument")
    elif len(score) > 1:
        failures.append("multiple_score_instruments")
    if len(chord) == 0:
        failures.append("missing_chord_instrument")
    elif len(chord) > 1:
        failures.append("multiple_chord_instruments")
    if unexpected:
        failures.append("mixed_or_unexpected_channel_instrument")
    selected = set(metadata) | set(score)
    if any(
        row.source_track_index not in selected and row.global_meta_types
        for row in tracks
    ):
        failures.append("required_global_meta_on_excluded_instrument")
    return Pop909ClInstrumentResolution(
        tracks=tuple(tracks),
        score_track_indices=score,
        chord_track_indices=chord,
        metadata_track_indices=metadata,
        unexpected_track_indices=unexpected,
        failure_categories=tuple(failures),
    )


def project_pop909_cl_score_bytes(
    midi: mido.MidiFile,
    resolution: Pop909ClInstrumentResolution,
) -> bytes:
    """Serialize the leakage-safe channel-0 plus conductor/meta projection."""

    if len(resolution.score_track_indices) != 1:
        raise Pop909ClConversionError(
            "score projection requires exactly one channel-0 score instrument",
            category="pop909_cl.instrument.score_not_unique",
        )
    fatal = set(resolution.failure_categories) - {"missing_chord_instrument"}
    if fatal:
        raise Pop909ClConversionError(
            f"unsafe POP909-CL instrument contract: {sorted(fatal)}",
            category="pop909_cl.instrument.invalid",
        )
    selected = set(resolution.metadata_track_indices) | {
        resolution.score_track_indices[0]
    }
    projected = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    for track_index, track in enumerate(midi.tracks):
        if track_index not in selected:
            continue
        copied = mido.MidiTrack()
        copied.extend(message.copy() for message in track)
        projected.tracks.append(copied)
    buffer = BytesIO()
    projected.save(file=buffer)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class _PairedChordNote:
    onset_tick: int
    end_tick: int
    pitch: int
    velocity: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _RawPairingAnomaly:
    anomaly_id: str
    category: Literal["dangling_note_on", "unmatched_note_off"]
    tick: int
    pitch: int
    velocity: int
    message_type: str
    ordinal: int


def _pair_chord_notes(
    track: mido.MidiTrack,
) -> tuple[tuple[_PairedChordNote, ...], tuple[_RawPairingAnomaly, ...]]:
    open_notes: dict[int, deque[tuple[int, int, int]]] = defaultdict(deque)
    paired: list[_PairedChordNote] = []
    anomalies: list[_RawPairingAnomaly] = []
    tick = 0
    ordinal = 0
    for message in track:
        tick += int(message.time)
        if getattr(message, "channel", None) != _CHORD_CHANNEL:
            continue
        is_on = message.type == "note_on" and message.velocity > 0
        is_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if not (is_on or is_off):
            continue
        event_ordinal = ordinal
        ordinal += 1
        pitch = int(message.note)
        if is_on:
            open_notes[pitch].append(
                (tick, int(message.velocity), event_ordinal)
            )
        elif not open_notes[pitch]:
            anomalies.append(
                _RawPairingAnomaly(
                    anomaly_id=f"unmatched_note_off:{event_ordinal}",
                    category="unmatched_note_off",
                    tick=tick,
                    pitch=pitch,
                    velocity=int(message.velocity),
                    message_type=message.type,
                    ordinal=event_ordinal,
                )
            )
        else:
            onset, velocity, note_ordinal = open_notes[pitch].popleft()
            paired.append(
                _PairedChordNote(
                    onset_tick=onset,
                    end_tick=tick,
                    pitch=pitch,
                    velocity=velocity,
                    ordinal=note_ordinal,
                )
            )
    for pitch, queue in open_notes.items():
        for onset, velocity, note_ordinal in queue:
            anomalies.append(
                _RawPairingAnomaly(
                    anomaly_id=f"dangling_note_on:{note_ordinal}",
                    category="dangling_note_on",
                    tick=onset,
                    pitch=pitch,
                    velocity=velocity,
                    message_type="note_on",
                    ordinal=note_ordinal,
                )
            )
    return (
        tuple(
            sorted(
                paired,
                key=lambda item: (
                    item.onset_tick,
                    item.pitch,
                    item.end_tick,
                    item.ordinal,
                ),
            )
        ),
        tuple(
            sorted(
                anomalies,
                key=lambda item: (
                    item.tick,
                    item.ordinal,
                    item.category,
                    item.pitch,
                ),
            )
        ),
    )


def _normalization_candidates(
    pitch_classes: Iterable[int],
    bass_pc: int,
) -> tuple[Pop909ClChordCandidate, ...]:
    pcs = tuple(sorted(set(int(value) % 12 for value in pitch_classes)))
    candidates: list[Pop909ClChordCandidate] = []
    for root_pc in pcs:
        degrees = frozenset((pitch_class - root_pc) % 12 for pitch_class in pcs)
        for quality, pattern in _QUALITY_PATTERNS:
            if degrees == pattern:
                candidates.append(
                    Pop909ClChordCandidate(
                        root_pc=root_pc,
                        root=_PITCH_CLASS_NAMES[root_pc],
                        quality=quality,
                        bass_pc=bass_pc,
                        bass=_PITCH_CLASS_NAMES[bass_pc],
                        inversion_semitones=(bass_pc - root_pc) % 12,
                    )
                )
    return tuple(candidates)


def _score_duration_tick(
    resolution: Pop909ClInstrumentResolution,
) -> int:
    selected = set(resolution.metadata_track_indices) | set(
        resolution.score_track_indices
    )
    return max(
        (
            resolution.tracks[index].end_tick
            for index in selected
        ),
        default=0,
    )


def _extract_chord_evidence(
    midi: mido.MidiFile,
    resolution: Pop909ClInstrumentResolution,
    record: Pop909ClCorpusRecord,
) -> Pop909ClChordEvidence:
    ppqn = int(midi.ticks_per_beat)
    score_end = _score_duration_tick(resolution)
    if not resolution.chord_track_indices:
        missing = Pop909ClCoverageSpan(
            span_id="missing_chord_targets:0",
            kind="missing_chord_targets",
            start_tick=0,
            end_tick=score_end,
            ppqn=ppqn,
            available=False,
            value=None,
        )
        return Pop909ClChordEvidence(
            status="expected_missing",
            ppqn=ppqn,
            blocks=(),
            no_chord_spans=(),
            trailing_spans=(missing,),
            pairing_anomalies=(),
            repeated_pitch_block_count=0,
            mixed_end_block_count=0,
            overlap_count=0,
        )
    track_index = resolution.chord_track_indices[0]
    paired, raw_anomalies = _pair_chord_notes(midi.tracks[track_index])
    grouped: dict[int, list[_PairedChordNote]] = defaultdict(list)
    for note in paired:
        grouped[note.onset_tick].append(note)

    preliminary: list[Pop909ClChordBlock] = []
    covered_until = 0
    overlap_count = 0
    for block_index, (onset, notes) in enumerate(sorted(grouped.items())):
        pitches = tuple(sorted(note.pitch for note in notes))
        note_ends = tuple(sorted(note.end_tick for note in notes))
        pitch_classes = tuple(sorted({pitch % 12 for pitch in pitches}))
        lowest = min(pitches)
        candidates = _normalization_candidates(pitch_classes, lowest % 12)
        selected_candidate = candidates[0] if candidates else None
        status: Literal["supported", "ambiguous", "unsupported"]
        if not candidates:
            status = "unsupported"
        elif len(candidates) > 1:
            status = "ambiguous"
        else:
            status = "supported"
        overlaps = onset < covered_until
        overlap_count += int(overlaps)
        end_tick = max(note_ends)
        preliminary.append(
            Pop909ClChordBlock(
                block_id=f"block:{block_index:06d}",
                onset_tick=onset,
                end_tick=end_tick,
                ppqn=ppqn,
                midi_pitch_multiset=pitches,
                note_end_ticks=note_ends,
                pitch_class_set=pitch_classes,
                lowest_source_pitch=lowest,
                bass_pitch_class=lowest % 12,
                source_track_index=track_index,
                source_channel=_CHORD_CHANNEL,
                source_track_names=resolution.tracks[track_index].names,
                source_path=record.relative_path,
                source_sha256=record.sha256,
                repeated_pitch=len(pitches) != len(set(pitches)),
                mixed_end_ticks=len(set(note_ends)) > 1,
                overlaps_previous=overlaps,
                normalization_status=status,
                candidates=candidates,
                selected_candidate=selected_candidate,
                root_available=len(candidates) == 1,
                quality_available=(
                    bool(candidates)
                    and len({candidate.quality for candidate in candidates}) == 1
                ),
                inversion_available=len(candidates) == 1,
                pairing_anomaly_ids=(),
            )
        )
        covered_until = max(covered_until, end_tick)

    gaps: list[Pop909ClCoverageSpan] = []
    trailing: list[Pop909ClCoverageSpan] = []
    covered_until = 0
    for block in preliminary:
        if block.onset_tick > covered_until:
            gaps.append(
                Pop909ClCoverageSpan(
                    span_id=f"implicit_n:{len(gaps)}",
                    kind=(
                        "leading_no_chord"
                        if covered_until == 0
                        else "internal_no_chord"
                    ),
                    start_tick=covered_until,
                    end_tick=block.onset_tick,
                    ppqn=ppqn,
                    available=True,
                    value="N",
                )
            )
        covered_until = max(covered_until, block.end_tick)
    if covered_until < score_end:
        trailing.append(
            Pop909ClCoverageSpan(
                span_id="trailing_unannotated:0",
                kind="trailing_unannotated",
                start_tick=covered_until,
                end_tick=score_end,
                ppqn=ppqn,
                available=False,
                value=None,
            )
        )

    blocks = list(preliminary)
    spans = [*gaps, *trailing]
    anomalies: list[Pop909ClPairingAnomaly] = []
    for anomaly in raw_anomalies:
        affected_blocks = tuple(
            block.onset_tick
            for block in blocks
            if block.onset_tick <= anomaly.tick <= block.end_tick
        )
        affected_spans = tuple(
            span.span_id
            for span in spans
            if span.start_tick <= anomaly.tick < span.end_tick
        )
        interval_end = (
            score_end if anomaly.category == "dangling_note_on" else anomaly.tick
        )
        anomalies.append(
            Pop909ClPairingAnomaly(
                anomaly_id=anomaly.anomaly_id,
                category=anomaly.category,
                tick=anomaly.tick,
                pitch=anomaly.pitch,
                velocity=anomaly.velocity,
                channel=_CHORD_CHANNEL,
                message_type=anomaly.message_type,
                ordinal=anomaly.ordinal,
                source_track_index=track_index,
                source_path=record.relative_path,
                source_sha256=record.sha256,
                affected_block_onsets=affected_blocks,
                affected_span_ids=affected_spans,
                affected_interval_start_tick=anomaly.tick,
                affected_interval_end_tick=interval_end,
                affected_interval_basis=(
                    "open_note_to_score_end"
                    if anomaly.category == "dangling_note_on"
                    else "unmatched_point_event"
                ),
            )
        )
    anomaly_ids_by_block: dict[int, list[str]] = defaultdict(list)
    anomaly_ids_by_span: dict[str, list[str]] = defaultdict(list)
    for anomaly in anomalies:
        for onset in anomaly.affected_block_onsets:
            anomaly_ids_by_block[onset].append(anomaly.anomaly_id)
        for span_id in anomaly.affected_span_ids:
            anomaly_ids_by_span[span_id].append(anomaly.anomaly_id)
    blocks = [
        replace(
            block,
            pairing_anomaly_ids=tuple(anomaly_ids_by_block[block.onset_tick]),
        )
        for block in blocks
    ]
    gaps = [
        replace(
            span,
            pairing_anomaly_ids=tuple(anomaly_ids_by_span[span.span_id]),
        )
        for span in gaps
    ]
    trailing = [
        replace(
            span,
            pairing_anomaly_ids=tuple(anomaly_ids_by_span[span.span_id]),
        )
        for span in trailing
    ]
    return Pop909ClChordEvidence(
        status="available",
        ppqn=ppqn,
        blocks=tuple(blocks),
        no_chord_spans=tuple(gaps),
        trailing_spans=tuple(trailing),
        pairing_anomalies=tuple(anomalies),
        repeated_pitch_block_count=sum(block.repeated_pitch for block in blocks),
        mixed_end_block_count=sum(block.mixed_end_ticks for block in blocks),
        overlap_count=overlap_count,
    )


def _json_detail(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_dict(candidate: Pop909ClChordCandidate) -> dict[str, Any]:
    return {
        "bass": candidate.bass,
        "bass_pc": candidate.bass_pc,
        "inversion_semitones": candidate.inversion_semitones,
        "quality": candidate.quality,
        "root": candidate.root,
        "root_pc": candidate.root_pc,
    }


def _canonical_provenance_order(
    records: Iterable[ProvenanceRecord],
) -> tuple[ProvenanceRecord, ...]:
    by_id = {record.provenance_id: record for record in records}
    remaining = set(by_id)
    emitted: set[str] = set()
    ordered: list[ProvenanceRecord] = []
    while remaining:
        ready = sorted(
            identifier
            for identifier in remaining
            if all(parent in emitted or parent not in by_id for parent in by_id[identifier].parents)
        )
        if not ready:
            raise Pop909ClConversionError(
                "POP909-CL provenance graph contains a cycle",
                category="pop909_cl.provenance_cycle",
            )
        identifier = ready[0]
        ordered.append(by_id[identifier])
        emitted.add(identifier)
        remaining.remove(identifier)
    return tuple(ordered)


def _target(
    *,
    target_id: str,
    task: str,
    entity_ids: Sequence[str],
    values: Sequence[str | None],
    mask: Sequence[bool],
    sources: Sequence[str | None],
    provenance: Sequence[str | None],
    class_labels: tuple[str, ...],
) -> TargetArray:
    length = len(entity_ids)
    return TargetArray(
        target_id=target_id,
        task=task,
        annotation_view_id="pop909_cl.channel_1",
        alignment_type="annotation_span",
        entity_ids=tuple(entity_ids),
        value_type="categorical",
        class_labels=class_labels,
        values=tuple(values),
        mask=tuple(mask),
        confidence=(None,) * length,
        source=tuple(sources),  # type: ignore[arg-type]
        provenance=tuple(provenance),
    )


def _attach_targets(
    piece: CanonicalPiece,
    record: Pop909ClCorpusRecord,
    evidence: Pop909ClChordEvidence,
) -> CanonicalPiece:
    file_provenance_id = "prov:pop909-cl-file"
    records: list[ProvenanceRecord] = [
        *piece.provenance,
        ProvenanceRecord(
            provenance_id=file_provenance_id,
            kind="source",
            source="pop909_cl",
            record_id=record.song_id,
            uri=(
                f"{POP909_CL_UPSTREAM_REPOSITORY}/blob/"
                f"{POP909_CL_UPSTREAM_COMMIT}/POP909_processed/"
                f"{record.corpus_relative_path}"
            ),
            version=POP909_CL_UPSTREAM_COMMIT,
            checksum_sha256=record.sha256,
            created_at=None,
            parents=(),
            details=(
                ("adapter_version", POP909_CL_ADAPTER_VERSION),
                ("dataset_name", POP909_CL_DATASET_NAME),
                ("license", POP909_CL_UPSTREAM_LICENSE),
                ("license_sha256", POP909_CL_UPSTREAM_LICENSE_SHA256),
                ("lineage_group_id", record.lineage_group_id),
                ("relative_path", record.relative_path),
                ("source_group_id", record.source_group_id),
                ("upstream_repository", POP909_CL_UPSTREAM_REPOSITORY),
            ),
        ),
    ]
    annotations: list[AnnotationSpan] = []
    block_span_ids: list[str] = []
    roots: list[str | None] = []
    qualities: list[str | None] = []
    basses: list[str | None] = []
    inversions: list[str | None] = []
    root_masks: list[bool] = []
    quality_masks: list[bool] = []
    inversion_masks: list[bool] = []
    raw_provenance_ids: list[str] = []
    normalized_provenance_ids: list[str | None] = []

    for index, block in enumerate(evidence.blocks):
        span_id = f"span:pop909-cl-chord-{index:06d}"
        raw_id = f"prov:pop909-cl-block-{index:06d}"
        normalized_id = f"prov:pop909-cl-normalized-{index:06d}"
        exact_start_qn = RationalTime(block.onset_tick, block.ppqn)
        exact_end_qn = RationalTime(block.end_tick, block.ppqn)
        aligned_start_qn = min(exact_start_qn, piece.duration_qn)
        aligned_end_qn = min(exact_end_qn, piece.duration_qn)
        block_span_ids.append(span_id)
        raw_provenance_ids.append(raw_id)
        normalized_provenance_ids.append(
            normalized_id
            if (
                block.root_available
                or block.quality_available
                or block.inversion_available
            )
            else None
        )
        annotations.append(
            AnnotationSpan(
                annotation_id=span_id,
                annotation_type="pop909_cl.chord",
                layer="target_alignment",
                start_qn=aligned_start_qn,
                end_qn=aligned_end_qn,
                track_id=None,
                value=None,
                provenance_id=raw_id,
            )
        )
        records.append(
            ProvenanceRecord(
                provenance_id=raw_id,
                kind="annotation",
                source="human",
                record_id=block.block_id,
                uri=None,
                version=None,
                checksum_sha256=None,
                created_at=None,
                parents=(file_provenance_id,),
                details=(
                    ("bass_pitch_class", block.bass_pitch_class),
                    ("end_tick", block.end_tick),
                    ("expert_reviewed", True),
                    ("human_corrected", True),
                    ("midi_pitch_multiset", _json_detail(block.midi_pitch_multiset)),
                    ("note_end_ticks", _json_detail(block.note_end_ticks)),
                    ("onset_tick", block.onset_tick),
                    ("pitch_class_set", _json_detail(block.pitch_class_set)),
                    ("ppqn", block.ppqn),
                    ("source_channel", block.source_channel),
                    ("source_track_index", block.source_track_index),
                    (
                        "target_alignment_clipped_to_raw_duration",
                        exact_start_qn != aligned_start_qn
                        or exact_end_qn != aligned_end_qn,
                    ),
                ),
            )
        )
        if normalized_provenance_ids[-1] is not None:
            records.append(
                ProvenanceRecord(
                    provenance_id=normalized_id,
                    kind="derivation",
                    source="pop909_cl_upstream_normalizer",
                    record_id=block.block_id,
                    uri=(
                        f"{POP909_CL_UPSTREAM_REPOSITORY}/blob/"
                        f"{POP909_CL_UPSTREAM_COMMIT}/process_pop909.py"
                    ),
                    version=POP909_CL_UPSTREAM_COMMIT,
                    checksum_sha256=None,
                    created_at=None,
                    parents=(raw_id,),
                    details=(
                        (
                            "candidates",
                            _json_detail(
                                tuple(
                                    _candidate_dict(candidate)
                                    for candidate in block.candidates
                                )
                            ),
                        ),
                        (
                            "method",
                            "get_chord_quality_exact_sevenths_then_triads_ascending_roots",
                        ),
                        ("normalization_status", block.normalization_status),
                    ),
                )
            )
        selected = block.selected_candidate
        roots.append(selected.root if block.root_available and selected else None)
        qualities.append(
            (
                next(iter({candidate.quality for candidate in block.candidates}))
                if block.quality_available
                else None
            )
        )
        basses.append(_PITCH_CLASS_NAMES[block.bass_pitch_class])
        inversions.append(
            (
                str(selected.inversion_semitones)
                if block.inversion_available and selected
                else None
            )
        )
        root_masks.append(block.root_available)
        quality_masks.append(block.quality_available)
        inversion_masks.append(block.inversion_available)

    no_chord_span_ids: list[str] = []
    no_chord_values: list[str | None] = []
    no_chord_masks: list[bool] = []
    no_chord_sources: list[str | None] = []
    no_chord_provenance: list[str | None] = []
    all_coverage = (*evidence.no_chord_spans, *evidence.trailing_spans)
    for index, span in enumerate(all_coverage):
        span_id = f"span:pop909-cl-coverage-{index:06d}"
        no_chord_span_ids.append(span_id)
        derivation_id = (
            f"prov:pop909-cl-gap-{index:06d}" if span.available else None
        )
        annotations.append(
            AnnotationSpan(
                annotation_id=span_id,
                annotation_type="pop909_cl.chord",
                layer="target_alignment",
                start_qn=RationalTime(span.start_tick, span.ppqn),
                end_qn=RationalTime(span.end_tick, span.ppqn),
                track_id=None,
                value=None,
                provenance_id=derivation_id,
            )
        )
        if derivation_id is not None:
            parent_ids = tuple(
                raw_provenance_ids[block_index]
                for block_index, block in enumerate(evidence.blocks)
                if block.end_tick == span.start_tick
                or block.onset_tick == span.end_tick
            ) or (file_provenance_id,)
            records.append(
                ProvenanceRecord(
                    provenance_id=derivation_id,
                    kind="derivation",
                    source="pop909_cl_gap_rule",
                    record_id=span.span_id,
                    uri=(
                        f"{POP909_CL_UPSTREAM_REPOSITORY}/blob/"
                        f"{POP909_CL_UPSTREAM_COMMIT}/process_pop909.py"
                    ),
                    version=POP909_CL_UPSTREAM_COMMIT,
                    checksum_sha256=None,
                    created_at=None,
                    parents=parent_ids,
                    details=(
                        (
                            "method",
                            "leading_or_internal_positive_duration_gap_only",
                        ),
                        ("span_kind", span.kind),
                    ),
                )
            )
        no_chord_values.append(span.value if span.available else None)
        no_chord_masks.append(span.available)
        no_chord_sources.append("derived" if span.available else None)
        no_chord_provenance.append(derivation_id)

    if evidence.status == "expected_missing":
        unavailable_id = no_chord_span_ids[0]
        block_span_ids = [unavailable_id]
        roots = [None]
        qualities = [None]
        basses = [None]
        inversions = [None]
        root_masks = [False]
        quality_masks = [False]
        inversion_masks = [False]
        raw_provenance_ids = []
        normalized_provenance_ids = [None]

    def masked_sources(mask: Sequence[bool], source: str) -> tuple[str | None, ...]:
        return tuple(source if available else None for available in mask)

    def normalized_refs(mask: Sequence[bool]) -> tuple[str | None, ...]:
        return tuple(
            reference if available else None
            for available, reference in zip(mask, normalized_provenance_ids)
        )

    if evidence.status == "expected_missing":
        boundary_mask = (False,)
        boundary_values: tuple[str | None, ...] = (None,)
        boundary_provenance: tuple[str | None, ...] = (None,)
        bass_mask = (False,)
        bass_provenance: tuple[str | None, ...] = (None,)
    else:
        boundary_mask = (True,) * len(block_span_ids)
        boundary_values = ("present",) * len(block_span_ids)
        boundary_provenance = tuple(raw_provenance_ids)
        bass_mask = (True,) * len(block_span_ids)
        bass_provenance = tuple(raw_provenance_ids)

    targets = [
        _target(
            target_id="target:pop909-cl-bass",
            task=POP909_CL_TASK_BASS,
            entity_ids=block_span_ids,
            values=basses,
            mask=bass_mask,
            sources=masked_sources(bass_mask, "human"),
            provenance=bass_provenance,
            class_labels=_PITCH_CLASS_NAMES,
        ),
        _target(
            target_id="target:pop909-cl-boundary",
            task=POP909_CL_TASK_BOUNDARY,
            entity_ids=block_span_ids,
            values=boundary_values,
            mask=boundary_mask,
            sources=masked_sources(boundary_mask, "human"),
            provenance=boundary_provenance,
            class_labels=("present",),
        ),
        _target(
            target_id="target:pop909-cl-inversion",
            task=POP909_CL_TASK_INVERSION,
            entity_ids=block_span_ids,
            values=inversions,
            mask=inversion_masks,
            sources=masked_sources(inversion_masks, "derived"),
            provenance=normalized_refs(inversion_masks),
            class_labels=tuple(str(value) for value in range(12)),
        ),
        _target(
            target_id="target:pop909-cl-no-chord",
            task=POP909_CL_TASK_NO_CHORD,
            entity_ids=no_chord_span_ids,
            values=no_chord_values,
            mask=no_chord_masks,
            sources=no_chord_sources,
            provenance=no_chord_provenance,
            class_labels=("N",),
        ),
        _target(
            target_id="target:pop909-cl-quality",
            task=POP909_CL_TASK_QUALITY,
            entity_ids=block_span_ids,
            values=qualities,
            mask=quality_masks,
            sources=masked_sources(quality_masks, "derived"),
            provenance=normalized_refs(quality_masks),
            class_labels=_QUALITIES,
        ),
        _target(
            target_id="target:pop909-cl-root",
            task=POP909_CL_TASK_ROOT,
            entity_ids=block_span_ids,
            values=roots,
            mask=root_masks,
            sources=masked_sources(root_masks, "derived"),
            provenance=normalized_refs(root_masks),
            class_labels=_PITCH_CLASS_NAMES,
        ),
    ]
    flags = list(piece.quality_flags)
    if evidence.status == "expected_missing":
        flags.append(
            QualityFlag(
                code="pop909_cl.expected_missing_chord_targets",
                severity="warning",
                message=(
                    f"Song {record.song_id} has no channel-1 chord instrument; "
                    "all POP909-CL chord target families are explicitly masked."
                ),
                entity_ids=(piece.piece_id,),
                provenance_id=None,
            )
        )
    return replace(
        piece,
        annotations=tuple(
            sorted(
                annotations,
                key=lambda item: (
                    item.start_qn,
                    item.end_qn,
                    item.annotation_id,
                ),
            )
        ),
        targets=tuple(
            sorted(
                targets,
                key=lambda item: (
                    item.task,
                    item.annotation_view_id is not None,
                    item.annotation_view_id or "",
                    item.target_id,
                ),
            )
        ),
        provenance=_canonical_provenance_order(records),
        quality_flags=tuple(
            sorted(flags, key=lambda item: (item.code, item.entity_ids, item.message))
        ),
    )


def _midi_error_category(exc: MidiAdapterError) -> str:
    message = str(exc).lower()
    if "inside a bar" in message:
        return "midi_adapter.meter_change_inside_bar"
    if "corrupted or unreadable" in message:
        return "midi_adapter.corrupt_or_unreadable"
    if "canonical validation failed" in message:
        return "midi_adapter.canonical_validation"
    return "midi_adapter.other"


def _convert_score_projection(
    projection: bytes,
    record: Pop909ClCorpusRecord,
) -> CanonicalPiece:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="music-critic-pop909-cl-",
            suffix=".mid",
            delete=False,
        ) as handle:
            handle.write(projection)
            temporary_path = handle.name
        piece = load_midi_piece(
            temporary_path,
            config=MidiAdapterConfig(
                dataset_name=POP909_CL_DATASET_NAME,
                source_group_id=record.source_group_id,
                split=None,
            ),
        )
    except MidiAdapterError as exc:
        normalized = str(exc)
        if temporary_path is not None:
            normalized = normalized.replace(temporary_path, record.relative_path)
        raise MidiAdapterError(
            normalized,
            validation_report=exc.validation_report,
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                Path(temporary_path).unlink()
            except FileNotFoundError:
                pass
    return replace(piece, source_path=record.relative_path)


def convert_pop909_cl_file(
    record: Pop909ClCorpusRecord,
    *,
    config: Pop909ClAdapterConfig = Pop909ClAdapterConfig(),
) -> Pop909ClConversionResult:
    """Convert one discovered POP909-CL MIDI into an accepted or quarantine result."""

    try:
        payload = record.path.read_bytes()
    except (OSError, ValueError) as exc:
        raise Pop909ClConversionError(
            f"{record.relative_path}: unreadable MIDI source: {exc}",
            category="pop909_cl.midi_unreadable",
            song_id=record.song_id,
        ) from exc
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != record.sha256:
        raise Pop909ClConversionError(
            f"{record.relative_path}: file SHA-256 changed after discovery",
            category="pop909_cl.file_fingerprint_mismatch",
            song_id=record.song_id,
        )
    try:
        midi = mido.MidiFile(file=BytesIO(payload))
    except (EOFError, KeyError, OSError, ValueError) as exc:
        raise Pop909ClConversionError(
            f"{record.relative_path}: unreadable MIDI: {exc}",
            category="pop909_cl.midi_unreadable",
            song_id=record.song_id,
        ) from exc
    resolution = inspect_pop909_cl_instruments(midi)
    failures = set(resolution.failure_categories)
    if "missing_chord_instrument" in failures:
        if record.song_id not in POP909_CL_EXPECTED_MISSING_TARGET_IDS:
            raise Pop909ClConversionError(
                f"{record.relative_path}: unexpected missing chord instrument",
                category="pop909_cl.instrument.missing_chord_unexpected",
                song_id=record.song_id,
            )
        failures.remove("missing_chord_instrument")
    if failures:
        raise Pop909ClConversionError(
            f"{record.relative_path}: instrument contract failed: {sorted(failures)}",
            category="pop909_cl.instrument.invalid",
            song_id=record.song_id,
        )
    projection = project_pop909_cl_score_bytes(midi, resolution)
    projection_sha256 = sha256(projection).hexdigest()
    evidence = _extract_chord_evidence(midi, resolution, record)
    try:
        piece = _convert_score_projection(projection, record)
    except MidiAdapterError as exc:
        category = _midi_error_category(exc)
        if (
            record.song_id in POP909_CL_QUARANTINE_IDS
            and category == "midi_adapter.meter_change_inside_bar"
        ):
            normalized_error = " ".join(
                str(exc).replace(str(record.path), record.relative_path).split()
            )
            return Pop909ClQuarantine(
                status="quarantined",
                record=record,
                category=category,
                source_error_type=f"{type(exc).__module__}.{type(exc).__name__}",
                source_error=normalized_error,
                instrument_resolution=resolution,
                chord_evidence=evidence,
                score_projection_sha256=projection_sha256,
            )
        raise Pop909ClConversionError(
            f"{record.relative_path}: unexpected score conversion failure: {exc}",
            category=f"pop909_cl.conversion.{category}",
            song_id=record.song_id,
        ) from exc
    if record.song_id in POP909_CL_QUARANTINE_IDS:
        raise Pop909ClConversionError(
            f"{record.relative_path}: pinned quarantine unexpectedly converted",
            category="pop909_cl.quarantine_expected_failure_missing",
            song_id=record.song_id,
        )
    if config.include_targets:
        piece = _attach_targets(piece, record, evidence)
    report = validate_piece(piece)
    if report.errors:
        raise Pop909ClConversionError(
            f"{record.relative_path}: canonical validation failed",
            category="pop909_cl.canonical_validation",
            song_id=record.song_id,
        )
    if evidence.status == "expected_missing":
        return Pop909ClExpectedTargetAbsence(
            status="accepted_missing_targets",
            record=record,
            piece=piece,
            chord_evidence=evidence,
            instrument_resolution=resolution,
            score_projection_sha256=projection_sha256,
            validation_report=report,
        )
    return Pop909ClAccepted(
        status="accepted",
        record=record,
        piece=piece,
        chord_evidence=evidence,
        instrument_resolution=resolution,
        score_projection_sha256=projection_sha256,
        validation_report=report,
    )


def iter_pop909_cl_corpus(
    root: str | PathLike[str],
    *,
    config: Pop909ClAdapterConfig = Pop909ClAdapterConfig(),
    identity: Pop909ClCorpusIdentity = Pop909ClCorpusIdentity(),
) -> Iterator[Pop909ClConversionResult]:
    """Validate corpus identity once, then convert one MIDI at a time."""

    discovery = discover_pop909_cl_corpus(root, identity=identity)
    for record in discovery.records:
        yield convert_pop909_cl_file(record, config=config)


__all__ = [
    "POP909_CL_ADAPTER_VERSION",
    "POP909_CL_ANOMALY_FINGERPRINT",
    "POP909_CL_CONTENT_FINGERPRINT",
    "POP909_CL_CORPUS_MANIFEST_VERSION",
    "POP909_CL_DATASET_NAME",
    "POP909_CL_EXPECTED_MISSING_TARGET_IDS",
    "POP909_CL_EXPECTED_SONG_IDS",
    "POP909_CL_QUARANTINE_IDS",
    "POP909_CL_TASK_BASS",
    "POP909_CL_TASK_BOUNDARY",
    "POP909_CL_TASK_INVERSION",
    "POP909_CL_TASK_NO_CHORD",
    "POP909_CL_TASK_QUALITY",
    "POP909_CL_TASK_ROOT",
    "POP909_CL_TASKS",
    "POP909_CL_UPSTREAM_COMMIT",
    "POP909_CL_UPSTREAM_LICENSE",
    "POP909_CL_UPSTREAM_LICENSE_SHA256",
    "POP909_CL_UPSTREAM_REPOSITORY",
    "Pop909ClAccepted",
    "Pop909ClAdapterConfig",
    "Pop909ClAdapterError",
    "Pop909ClChordBlock",
    "Pop909ClChordCandidate",
    "Pop909ClChordEvidence",
    "Pop909ClConversionError",
    "Pop909ClConversionResult",
    "Pop909ClCorpusDiscovery",
    "Pop909ClCorpusIdentity",
    "Pop909ClCorpusIdentityError",
    "Pop909ClCorpusIssue",
    "Pop909ClCorpusRecord",
    "Pop909ClCoverageSpan",
    "Pop909ClExpectedTargetAbsence",
    "Pop909ClInstrumentResolution",
    "Pop909ClPairingAnomaly",
    "Pop909ClQuarantine",
    "Pop909ClTrackEvidence",
    "convert_pop909_cl_file",
    "discover_pop909_cl_corpus",
    "inspect_pop909_cl_instruments",
    "iter_pop909_cl_corpus",
    "pop909_cl_source_group_id",
    "pop909_lineage_group_id",
    "project_pop909_cl_score_bytes",
]
