"""Isolated raw candidate / annotated next-voice-note contract.

This module neither changes a graph nor reads a corpus. Source voice annotations
are an explicit target-only sidecar; CanonicalNote.voice is never promoted to
gold automatically. See VOICE_CONTINUATION_CONTRACT.md for coverage limits.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Literal

from music_critic.data.schema import CanonicalPiece, ProvenanceRecord
from music_critic.data.timing import RationalTime


VOICE_CONTINUATION_VERSION = "voice-continuation-candidates-targets@1.0.0"
AnnotationSource = Literal["human", "dataset", "synthetic"]
TargetReason = Literal[
    "available", "missing_annotation", "ambiguous_source_onset",
    "ambiguous_successor_onset", "incomplete_annotation_interval",
    "no_observed_successor", "successor_outside_candidates",
]


def _identifier(value: str, name: str) -> None:
    if (
        not isinstance(value, str) or not value or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty trimmed identifier")


@dataclass(frozen=True, slots=True)
class VoiceCandidatePolicy:
    """Predeclared raw-only horizon and output/memory bounds, not tuned defaults."""

    max_onset_delta_qn: RationalTime = RationalTime(8)
    max_candidates_per_note: int = 16
    max_notes_per_piece: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.max_onset_delta_qn, RationalTime):
            raise TypeError("max_onset_delta_qn must use exact RationalTime")
        if self.max_onset_delta_qn <= RationalTime(0):
            raise ValueError("max_onset_delta_qn must be positive")
        for name in ("max_candidates_per_note", "max_notes_per_piece"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class RawVoiceNote:
    note_id: str
    onset_qn: RationalTime
    pitch: int
    duration_qn: RationalTime


@dataclass(frozen=True, slots=True)
class VoiceCandidates:
    piece_id: str
    policy: VoiceCandidatePolicy
    notes: tuple[RawVoiceNote, ...]
    # One row per notes[i]. Each entry is a destination note ID.
    destination_ids: tuple[tuple[str, ...], ...]
    version: str = VOICE_CONTINUATION_VERSION


def build_voice_candidates(
    piece: CanonicalPiece, policy: VoiceCandidatePolicy = VoiceCandidatePolicy(),
) -> VoiceCandidates:
    """All notes are eligible, across tracks; only later onsets are candidates.

    A strictly later onset can overlap the source note's sustained duration.
    Sorting/truncation use raw onset, pitch, duration, and ID only. Complexity is
    O(N log N + N K), and no quadratic all-pairs allocation is made.
    """
    _identifier(piece.piece_id, "piece_id")
    if len(piece.notes) > policy.max_notes_per_piece:
        raise ValueError("piece exceeds max_notes_per_piece; no silent truncation")
    raw = tuple(sorted(
        (RawVoiceNote(note.note_id, note.onset_qn, note.pitch, note.duration_qn)
         for note in piece.notes),
        key=lambda note: (note.onset_qn, note.pitch, note.duration_qn, note.note_id),
    ))
    if len({note.note_id for note in raw}) != len(raw):
        raise ValueError("duplicate note IDs in piece")
    for note in raw:
        _identifier(note.note_id, "note_id")
        if not note.note_id.startswith("note:"):
            raise ValueError("note_id must have canonical note: prefix")
        if not isinstance(note.onset_qn, RationalTime) or not isinstance(
            note.duration_qn, RationalTime
        ):
            raise TypeError("candidate timing must use exact RationalTime")
        if note.onset_qn < RationalTime(0) or note.duration_qn < RationalTime(0):
            raise ValueError("candidate timing must be nonnegative")
        if type(note.pitch) is not int or not 0 <= note.pitch <= 127:
            raise ValueError("candidate pitch must be a MIDI integer")
    onsets = tuple(note.onset_qn for note in raw)
    destinations = []
    for note in raw:
        start = bisect_right(onsets, note.onset_qn)
        end = min(
            bisect_right(onsets, note.onset_qn + policy.max_onset_delta_qn),
            start + policy.max_candidates_per_note,
        )
        destinations.append(tuple(target.note_id for target in raw[start:end]))
    return VoiceCandidates(piece.piece_id, policy, raw, tuple(destinations))


@dataclass(frozen=True, slots=True)
class VoiceAnnotation:
    note_id: str
    # Explicit source scope plus source voice identifier; never track == voice.
    scoped_voice_id: tuple[str, str] | None
    source: AnnotationSource | None
    provenance_id: str | None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.note_id, "annotation note_id")
        if self.scoped_voice_id is None:
            if any(value is not None for value in (
                self.source, self.provenance_id, self.confidence,
            )):
                raise ValueError("missing annotation requires null evidence/confidence")
            return
        if type(self.scoped_voice_id) is not tuple or len(self.scoped_voice_id) != 2:
            raise ValueError("scoped_voice_id requires (source scope, source voice)")
        for value in self.scoped_voice_id:
            _identifier(value, "scoped_voice_id")
        if self.source not in ("human", "dataset", "synthetic"):
            raise ValueError("voice targets require source annotations, not heuristics")
        _identifier(self.provenance_id, "provenance_id")
        if self.confidence is not None and (
            type(self.confidence) not in (float, int)
            or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be null or finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class VoiceAnnotationView:
    piece_id: str
    annotation_view_id: str
    annotations: tuple[VoiceAnnotation, ...]
    provenance: tuple[ProvenanceRecord, ...]


@dataclass(frozen=True, slots=True)
class VoiceContinuationTarget:
    source_note_id: str
    next_note_id: str | None
    candidate_index: int | None
    mask: bool
    confidence: float | None
    # References retain every annotation used to establish the next-note claim.
    evidence: tuple[VoiceAnnotation, ...]
    reason: TargetReason


@dataclass(frozen=True, slots=True)
class VoiceContinuationTargets:
    piece_id: str
    annotation_view_id: str
    candidates: VoiceCandidates
    targets: tuple[VoiceContinuationTarget, ...]
    provenance: tuple[ProvenanceRecord, ...]
    version: str = VOICE_CONTINUATION_VERSION


def map_voice_continuation_targets(
    candidates: VoiceCandidates, view: VoiceAnnotationView,
) -> VoiceContinuationTargets:
    """Map the unique next strictly later source-annotated voice note.

    Missing evidence anywhere from source onset through the proposed next
    onset masks the claim: an unknown note could belong to the same voice.
    Simultaneous notes within a source voice are ambiguous, never pitch-ranked
    gold. End-of-observation and raw candidate misses do not become negatives.
    """
    if candidates.version != VOICE_CONTINUATION_VERSION:
        raise ValueError("unsupported voice candidate version")
    if view.piece_id != candidates.piece_id:
        raise ValueError("annotation piece_id does not match candidate piece_id")
    _identifier(view.annotation_view_id, "annotation_view_id")
    notes = candidates.notes
    note_ids = {note.note_id for note in notes}
    if len(notes) != len(note_ids) or len(notes) != len(candidates.destination_ids):
        raise ValueError("malformed candidate rows")
    # Candidate objects are immutable but publicly constructible. Fail closed
    # on forged topology before aligning any target.
    keys = [(n.onset_qn, n.pitch, n.duration_qn, n.note_id) for n in notes]
    if keys != sorted(keys) or len(notes) > candidates.policy.max_notes_per_piece:
        raise ValueError("malformed candidate ordering or note bound")
    onsets = tuple(note.onset_qn for note in notes)
    for note, row in zip(notes, candidates.destination_ids, strict=True):
        start = bisect_right(onsets, note.onset_qn)
        stop = min(
            bisect_right(onsets, note.onset_qn + candidates.policy.max_onset_delta_qn),
            start + candidates.policy.max_candidates_per_note,
        )
        if row != tuple(target.note_id for target in notes[start:stop]):
            raise ValueError("candidate rows violate raw policy")
    annotations = {annotation.note_id: annotation for annotation in view.annotations}
    if len(annotations) != len(view.annotations):
        raise ValueError("duplicate voice annotations")
    if not annotations.keys() <= note_ids:
        raise ValueError("annotation refers to a note outside this piece")
    provenance_ids = {record.provenance_id for record in view.provenance}
    if len(provenance_ids) != len(view.provenance):
        raise ValueError("duplicate annotation provenance")
    for annotation in view.annotations:
        if annotation.scoped_voice_id is not None:
            if annotation.provenance_id not in provenance_ids:
                raise ValueError("annotation provenance reference is unresolved")
    voices: dict[tuple[str, str], list[RawVoiceNote]] = {}
    unknown_onsets = []
    for note in notes:
        annotation = annotations.get(note.note_id)
        if annotation is None or annotation.scoped_voice_id is None:
            unknown_onsets.append(note.onset_qn)
        else:
            voices.setdefault(annotation.scoped_voice_id, []).append(note)

    targets = []
    for note, destinations in zip(notes, candidates.destination_ids, strict=True):
        reason: TargetReason = "missing_annotation"
        annotation = annotations.get(note.note_id)
        successor = None
        if annotation is not None and annotation.scoped_voice_id is not None:
            voice_notes = voices[annotation.scoped_voice_id]
            same_time = [other for other in voice_notes if other.onset_qn == note.onset_qn]
            future = [other for other in voice_notes if other.onset_qn > note.onset_qn]
            if len(same_time) != 1:
                reason = "ambiguous_source_onset"
            elif not future:
                reason = "no_observed_successor"
            elif len([other for other in future if other.onset_qn == future[0].onset_qn]) != 1:
                reason = "ambiguous_successor_onset"
            elif any(note.onset_qn <= time <= future[0].onset_qn for time in unknown_onsets):
                reason = "incomplete_annotation_interval"
            elif future[0].note_id not in destinations:
                reason = "successor_outside_candidates"
            else:
                reason = "available"
                successor = future[0]
        if successor is None:
            targets.append(VoiceContinuationTarget(
                note.note_id, None, None, False, None, (), reason,
            ))
            continue
        evidence = tuple(
            annotations[other.note_id] for other in notes
            if note.onset_qn <= other.onset_qn <= successor.onset_qn
        )
        confidence = (
            None if any(item.confidence is None for item in evidence)
            else min(item.confidence for item in evidence)
        )
        targets.append(VoiceContinuationTarget(
            note.note_id, successor.note_id, destinations.index(successor.note_id),
            True, confidence, evidence, "available",
        ))
    return VoiceContinuationTargets(
        candidates.piece_id, view.annotation_view_id, candidates, tuple(targets),
        view.provenance,
    )
