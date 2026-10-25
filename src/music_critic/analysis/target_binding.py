"""Exact source-row bindings for external note-relation supervision.

This module does not infer musical roles, change raw music, locate corpora, or
read files. Callers authenticate source artifacts and split metadata first.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence,
    DilemmadataSourceRowBinding,
    _alignment_evidence_fingerprint,
)
from music_critic.data import CanonicalPiece, RationalTime, dumps_piece


class TargetBindingError(ValueError):
    """The source packet cannot be bound to the authenticated raw piece."""


@dataclass(frozen=True, slots=True)
class SourceEventBinding:
    """One source segment, including a possible tied-note continuation.

    ``binding_available`` asserts exact usable lineage, not musical correctness.
    A source score Part is intentionally distinct from a canonical raw track.
    """

    source_event_id: str
    canonical_note_id: str | None
    source_row_number: int
    source_start_qn: str
    source_end_qn: str
    canonical_start_qn: str
    canonical_end_qn: str
    canonical_track_id: str | None
    source_part_id: str | None
    source_staff: int | None
    source_voice: int | None
    tie_continuation: bool
    binding_available: bool
    mask_reason: str | None


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TargetBindingError(reason)


def _text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction(value: object, field: str) -> Fraction:
    _require(isinstance(value, str), f"{field}: exact rational string required")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise TargetBindingError(f"{field}: invalid rational") from exc
    _require(_text(result) == value, f"{field}: reduced rational required")
    return result


def _optional_int(value: object, field: str) -> int | None:
    if value in (None, ""):
        return None
    _require(type(value) in (str, int), f"{field}: integer required")
    try:
        result = int(value)
    except ValueError as exc:
        raise TargetBindingError(f"{field}: integer required") from exc
    _require(str(result) == str(value), f"{field}: canonical integer required")
    return result


def _evidence(payload: Mapping) -> DilemmadataRawTargetAlignmentEvidence:
    """Read the existing adapter's JSON representation without reopening TSVs."""
    try:
        fields = dict(payload)
        fields["raw_source_fields"] = tuple(fields["raw_source_fields"])
        fields["rows"] = tuple(
            DilemmadataSourceRowBinding(
                **{**row, "onset_qn": RationalTime(**row["onset_qn"])}
            ) for row in fields["rows"]
        )
        return DilemmadataRawTargetAlignmentEvidence(**fields)
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetBindingError("malformed raw alignment evidence") from exc


def bind_source_events(
    piece: CanonicalPiece,
    alignment_payload: Mapping,
    source_packet: Mapping,
    *,
    record_id: str,
    source_sha256: str,
    current_split: str,
    source_to_canonical_offset_qn: str = "0/1",
) -> tuple[SourceEventBinding, ...]:
    """Join an authenticated T04 DLC packet by exact original TSV line identity.

    Source bytes, parent artifact SHA/size, packet SHA, and current split
    metadata must already have been checked by the caller. This function then
    checks internal record/source identity, the adapter alignment fingerprint,
    canonical piece SHA, and note geometry. No nearest-note fallback exists.

    The explicit offset supports canonical pickup repair; all times remain in
    exact quarter notes. Repair-masked rows stay mapped but unavailable. A tied
    continuation names the original canonical note and keeps its own subspan.
    """
    _require(current_split == "train" and source_packet.get("current_split") == "train",
             "current TRAIN membership required")
    _require(isinstance(record_id, str) and record_id.startswith("dlc:"), "DLC record required")
    _require(isinstance(source_sha256, str) and len(source_sha256) == 64
             and all(c in "0123456789abcdef" for c in source_sha256), "source SHA invalid")
    source_version_id = f"{record_id}@sha256:{source_sha256}"
    _require(source_packet.get("source_version_id") == source_version_id, "source version mismatch")
    _require(source_packet.get("source_table", {}).get("sha256") == source_sha256, "source SHA mismatch")
    _require(not piece.targets and not piece.annotations, "raw-only piece required")
    evidence = _evidence(alignment_payload)
    _require(evidence.record_id == record_id and evidence.piece_id == piece.piece_id, "record/piece identity mismatch")
    _require(evidence.version in ("1.1.0", "1.2.0"), "unsupported alignment version")
    _require(evidence.canonical_piece_sha256 == sha256(dumps_piece(piece).encode()).hexdigest(), "canonical piece SHA mismatch")
    _require(evidence.fingerprint == _alignment_evidence_fingerprint(evidence), "alignment fingerprint mismatch")
    _require((evidence.version == "1.2.0") == (evidence.raw_repair_evidence_fingerprint is not None), "repair evidence mismatch")
    _require(tuple(row.ordinal for row in evidence.rows) == tuple(range(len(evidence.rows))), "invalid source row ordinals")
    _require(all(row.line == row.ordinal + 2 and type(row.tie_continuation) is bool
                 and row.repair_mask_scope in ("none", "note", "all") for row in evidence.rows), "invalid source row binding")
    _require(evidence.version == "1.2.0" or all(row.repair_mask_scope == "none" for row in evidence.rows), "unexpected repair mask")
    repaired_origin_verified = evidence.version == "1.2.0" and any(
        provenance.kind == "conversion"
        and provenance.source == "music_critic.dilemmadata_adapter"
        and dict(provenance.details).get("raw_repair_evidence_fingerprint")
        == evidence.raw_repair_evidence_fingerprint
        for provenance in piece.provenance
    )
    rows = {row.line: row for row in evidence.rows}
    notes = {note.note_id: note for note in piece.notes}
    _require(len(notes) == len(piece.notes), "duplicate canonical note ID")
    offset = _fraction(source_to_canonical_offset_qn, "source offset")
    events = source_packet.get("events")
    _require(isinstance(events, list) and bool(events), "nonempty source events required")
    result, seen_ids, seen_lines = [], set(), set()
    for event in events:
        _require(isinstance(event, Mapping), "source event mapping required")
        event_id, line = event.get("id"), event.get("source_row_number")
        _require(isinstance(event_id, str) and bool(event_id.strip()) and event_id not in seen_ids, "duplicate or invalid source event ID")
        _require(type(line) is int and line in rows and line not in seen_lines, "duplicate or missing source line")
        seen_ids.add(event_id)
        seen_lines.add(line)
        _require(event.get("source_version_id") == source_version_id
                 and event.get("source_row_id") == f"{record_id}#line={line}", "source event lineage mismatch")
        _require(event.get("time_unit") == "quarter_note", "quarter-note source clock required")
        start = _fraction(event.get("start"), "source start")
        duration = _fraction(event.get("duration"), "source duration")
        end = _fraction(event.get("end"), "source end")
        _require(duration >= 0 and end == start + duration, "source duration/end mismatch")
        row = rows[line]
        canonical_start, canonical_end = start + offset, end + offset
        _require(canonical_start == Fraction(row.onset_qn.num, row.onset_qn.den), "source/canonical onset mismatch")
        _require(type(event.get("is_note_onset")) is bool
                 and event["is_note_onset"] != row.tie_continuation, "tie-onset semantics mismatch")
        staff, voice = (_optional_int(event.get(key), key) for key in ("source_staff", "source_voice"))
        note_id = row.canonical_note_id
        track_id = None
        remediated_continuation_at_attack = False
        remediated_continuation_voice_change = False
        remediated_continuation_staff_change = False
        if note_id is None:
            _require(row.repair_mask_scope == "note", "unexplained missing canonical note")
        else:
            _require(note_id in notes, "canonical note missing")
            note = notes[note_id]
            _require(type(event.get("pitch")) is int and event["pitch"] == note.pitch, "source/canonical pitch mismatch")
            remediated_continuation_voice_change = (
                row.tie_continuation and repaired_origin_verified and staff == note.staff
                and voice != note.voice
            )
            # The raw adapter keeps the attack's staff even when its recorded
            # tie repair joins a continuation written on another staff. Keep
            # that exact row accounting, but never expose its labels as known.
            remediated_continuation_staff_change = (
                row.tie_continuation and repaired_origin_verified and staff != note.staff
            )
            _require((staff, voice) == (note.staff, note.voice)
                     or remediated_continuation_voice_change or remediated_continuation_staff_change,
                     "source/canonical staff/voice mismatch")
            note_start = Fraction(note.onset_qn.num, note.onset_qn.den)
            note_end = note_start + Fraction(note.duration_qn.num, note.duration_qn.den)
            _require(note_start <= canonical_start <= canonical_end <= note_end, "source segment outside canonical note")
            remediated_continuation_at_attack = (
                row.tie_continuation and canonical_start == note_start and repaired_origin_verified
            )
            _require((canonical_start > note_start or remediated_continuation_at_attack)
                     if row.tie_continuation else (canonical_start == note_start),
                     "source attack/continuation placement mismatch")
            track_id = note.track_id
        source_part = event.get("part_id")
        _require(source_part is None or isinstance(source_part, str) and bool(source_part.strip()), "source part ID invalid")
        available = (note_id is not None and row.repair_mask_scope == "none"
                     and not remediated_continuation_at_attack and not remediated_continuation_voice_change
                     and not remediated_continuation_staff_change)
        mask_reason = ("remediated_continuation_at_canonical_attack" if remediated_continuation_at_attack
                       else "remediated_continuation_staff_change" if remediated_continuation_staff_change
                       else "remediated_continuation_voice_change" if remediated_continuation_voice_change
                       else None if available else f"raw_repair_mask:{row.repair_mask_scope}")
        result.append(SourceEventBinding(
            source_event_id=event_id, canonical_note_id=note_id, source_row_number=line,
            source_start_qn=_text(start), source_end_qn=_text(end),
            canonical_start_qn=_text(canonical_start), canonical_end_qn=_text(canonical_end),
            canonical_track_id=track_id, source_part_id=source_part,
            source_staff=staff, source_voice=voice, tie_continuation=row.tie_continuation,
            binding_available=available,
            mask_reason=mask_reason,
        ))
    return tuple(result)
