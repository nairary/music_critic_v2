"""Exact, deterministic note edits for a bounded intervention pilot.

Compatibility is technical. Even ``compatible`` never certifies musical
independence or lower quality. Context IDs are declared by the author; this
module cannot discover missing musical dependencies. Rebuild derived graphs
from the returned piece and review the complete selected combination.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal

from music_critic.data.schema import CanonicalPiece, ProvenanceRecord, QualityFlag
from music_critic.data.serialization import (
    canonical_json_sha256, dumps_canonical_json, piece_to_dict,
)
from music_critic.data.timing import RationalTime
from music_critic.data.validation import CanonicalValidationError, validate_or_raise


VERSION = "MusicInterventions@1.0.0"
Status = Literal["compatible", "semantic_revalidation", "conflict"]


class InterventionError(ValueError):
    """An invalid or conflicting edit set, with no partially applied result."""


def _identifier(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InterventionError(f"{field} must be a nonempty trimmed string")
    if any(ord(char) < 32 for char in value):
        raise InterventionError(f"{field} cannot contain control characters")


@dataclass(frozen=True, slots=True)
class NotePatch:
    note_id: str
    pitch: int | None = None
    onset_qn: RationalTime | None = None
    duration_qn: RationalTime | None = None
    delete: bool = False

    def __post_init__(self) -> None:
        _identifier(self.note_id, "note_id")
        if type(self.delete) is not bool:
            raise InterventionError("delete must be bool")
        if self.pitch is not None and (
            type(self.pitch) is not int or not 0 <= self.pitch <= 127
        ):
            raise InterventionError("pitch must be an integer in 0..127")
        for field in ("onset_qn", "duration_qn"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, RationalTime):
                raise InterventionError(f"{field} must be exact RationalTime")
        fields = self.written_fields
        if self.delete and fields != frozenset({"delete"}):
            raise InterventionError("delete cannot also change note fields")
        if not fields:
            raise InterventionError("empty note patch")

    @property
    def written_fields(self) -> frozenset[str]:
        fields = {
            name for name in ("pitch", "onset_qn", "duration_qn")
            if getattr(self, name) is not None
        }
        if self.delete:
            fields.add("delete")
        return frozenset(fields)


@dataclass(frozen=True, slots=True)
class Intervention:
    id: str
    patches: tuple[NotePatch, ...]
    context_note_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.id, "intervention id")
        if not isinstance(self.patches, tuple) or not self.patches:
            raise InterventionError("patches must be a nonempty tuple")
        if not all(isinstance(patch, NotePatch) for patch in self.patches):
            raise InterventionError("patches must contain only NotePatch")
        ids = [patch.note_id for patch in self.patches]
        if len(set(ids)) != len(ids):
            raise InterventionError("duplicate note IDs within intervention")
        if not isinstance(self.context_note_ids, tuple):
            raise InterventionError("context_note_ids must be a tuple")
        for note_id in self.context_note_ids:
            _identifier(note_id, "context note ID")
        if len(set(self.context_note_ids)) != len(self.context_note_ids):
            raise InterventionError("duplicate context note IDs")


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    status: Status
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InterventionResult:
    piece: CanonicalPiece
    compatibility: CompatibilityReport
    applied_ids: tuple[str, ...]


def check_compatibility(left: Intervention, right: Intervention) -> CompatibilityReport:
    """Symmetric pairwise check; semantic revalidation is not a hard conflict."""
    if not isinstance(left, Intervention) or not isinstance(right, Intervention):
        raise InterventionError("pair must contain Intervention values")
    conflicts: list[str] = []
    if left.id == right.id:
        conflicts.append(f"duplicate intervention ID: {left.id}")
    left_patches = {patch.note_id: patch for patch in left.patches}
    right_patches = {patch.note_id: patch for patch in right.patches}
    for note_id in sorted(left_patches.keys() & right_patches.keys()):
        a, b = left_patches[note_id], right_patches[note_id]
        if a.delete or b.delete:
            conflicts.append(f"delete/edit conflict: {note_id}")
        else:
            overlap = a.written_fields & b.written_fields
            if overlap:
                conflicts.append(f"same-field conflict: {note_id}: {','.join(sorted(overlap))}")
    if conflicts:
        return CompatibilityReport("conflict", tuple(sorted(conflicts)))
    affected_context = (
        (set(left.context_note_ids) & right_patches.keys())
        | (set(right.context_note_ids) & left_patches.keys())
    )
    if affected_context:
        return CompatibilityReport("semantic_revalidation", tuple(
            f"edited context note: {note_id}" for note_id in sorted(affected_context)
        ))
    return CompatibilityReport("compatible")


def _plan_payload(interventions: tuple[Intervention, ...]) -> list[dict]:
    def rational(value: RationalTime | None) -> list[int] | None:
        return None if value is None else [value.num, value.den]

    return [{
        "id": item.id,
        "context_note_ids": sorted(item.context_note_ids),
        "patches": [{
            "note_id": patch.note_id, "pitch": patch.pitch,
            "onset_qn": rational(patch.onset_qn),
            "duration_qn": rational(patch.duration_qn), "delete": patch.delete,
        } for patch in sorted(item.patches, key=lambda patch: patch.note_id)],
    } for item in interventions]


def apply_interventions(
    piece: CanonicalPiece, interventions: tuple[Intervention, ...],
) -> InterventionResult:
    """Materialize one edit set against one source, or fail without mutation.

The result keeps source grouping/split and note identity, gives the derived
piece a new ID, and discards targets, target-alignment spans and quality flags.
Observation spans remain source observations, not re-inferred annotations.
All source note timing fields are cleared on a temporal edit; spelling is
cleared on a pitch edit. The provenance binds source bytes and the entire plan.
"""
    if not isinstance(interventions, tuple) or not interventions:
        raise InterventionError("interventions must be a nonempty tuple")
    if not all(isinstance(item, Intervention) for item in interventions):
        raise InterventionError("interventions must contain Intervention values")
    try:
        validate_or_raise(piece)
    except CanonicalValidationError as exc:
        raise InterventionError(f"invalid source piece: {exc}") from exc
    ordered = tuple(sorted(interventions, key=lambda item: item.id))
    reports = tuple(check_compatibility(a, b) for a, b in combinations(ordered, 2))
    conflicts = sorted({reason for report in reports if report.status == "conflict"
                        for reason in report.reasons})
    if conflicts:
        raise InterventionError("; ".join(conflicts))
    reasons = tuple(sorted({reason for report in reports for reason in report.reasons}))
    compatibility = CompatibilityReport(
        "semantic_revalidation" if reasons else "compatible", reasons,
    )
    notes_by_id = {note.note_id: note for note in piece.notes}
    merged: dict[str, dict] = {}
    for item in ordered:
        unknown_context = set(item.context_note_ids) - notes_by_id.keys()
        if unknown_context:
            raise InterventionError(f"unknown context note IDs: {sorted(unknown_context)}")
        for patch in item.patches:
            if patch.note_id not in notes_by_id:
                raise InterventionError(f"unknown edited note ID: {patch.note_id}")
            values = {field: getattr(patch, field) for field in patch.written_fields}
            if not patch.delete and all(
                getattr(notes_by_id[patch.note_id], field) == value
                for field, value in values.items()
            ):
                raise InterventionError(f"unchanged note patch: {patch.note_id}")
            merged.setdefault(patch.note_id, {}).update(values)

    source_hash = canonical_json_sha256(piece_to_dict(piece))
    plan = _plan_payload(ordered)
    digest = canonical_json_sha256({"source_sha256": source_hash, "plan": plan, "version": VERSION})
    result_id = f"piece:intervention-{digest}"
    provenance_id = f"prov:intervention-{digest}"
    if provenance_id in {item.provenance_id for item in piece.provenance}:
        raise InterventionError("derived provenance ID collision")
    changed_notes = []
    for note in piece.notes:
        values = dict(merged.get(note.note_id, {}))
        if values.pop("delete", False):
            continue
        if values:
            if "pitch" in values:
                values.update(spelling_step=None, spelling_alter=None)
            if "onset_qn" in values or "duration_qn" in values:
                values.update(source_onset_ticks=None, source_duration_ticks=None,
                              source_onset_seconds=None, source_duration_seconds=None)
            values["provenance_id"] = provenance_id
        changed_notes.append(replace(note, **values) if values else note)
    if not changed_notes:
        raise InterventionError("interventions cannot delete every note in the pilot")
    track_order = {track.track_id: index for index, track in enumerate(piece.tracks)}
    changed_notes.sort(key=lambda note: (
        note.onset_qn, track_order[note.track_id], note.pitch, note.duration_qn, note.note_id,
    ))
    provenance = ProvenanceRecord(
        provenance_id=provenance_id, kind="synthetic", source=VERSION,
        record_id=result_id, uri=None, version=VERSION,
        checksum_sha256=source_hash, created_at=None,
        parents=tuple(item.provenance_id for item in piece.provenance),
        details=tuple(sorted((("source_piece_id", piece.piece_id), ("source_sha256", source_hash),
                 ("plan_json", dumps_canonical_json(plan)),
                 ("compatibility", compatibility.status),
                 ("quality_claim", "none"),
                 ("dropped_target_count", len(piece.targets)),
                 ("dropped_quality_flag_count", len(piece.quality_flags))))),
    )
    result = replace(
        piece, piece_id=result_id, notes=tuple(changed_notes), targets=(),
        annotations=tuple(item for item in piece.annotations if item.layer == "observation"),
        provenance=(*piece.provenance, provenance),
        quality_flags=(QualityFlag(
            code="intervention.synthetic", severity="info",
            message=f"Deterministic note edits; {compatibility.status}; no quality label.",
            entity_ids=(result_id,), provenance_id=provenance_id,
        ),),
    )
    try:
        validate_or_raise(result)
    except CanonicalValidationError as exc:
        raise InterventionError(f"invalid intervention result: {exc}") from exc
    return InterventionResult(result, compatibility, tuple(item.id for item in ordered))
