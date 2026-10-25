"""Source-bound, TRAIN-only weak note-role targets with exact subspan evidence.

The TargetBundle is a note-participation projection. The assignments remain a
separate event-level artifact: relation roles may change during a sustained
canonical note, and several source fragments may bind one merged tied note.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable

from music_critic.data import CanonicalPiece, ProvenanceRecord, RationalTime
from music_critic.tasks.multisource import (
    TARGET_BUNDLE_CONTRACT_VERSION,
    SampleTarget,
    TargetBundle,
)
from music_critic.tasks.relational_registry import (
    RELATIONAL_FLAGS,
    RELATIONAL_TARGET_FAMILIES,
    RELATIONAL_TARGET_REGISTRY_ID,
)


RELATIONAL_ASSIGNMENT_CONTRACT_VERSION = "1.0.0"


class RelationalTargetError(ValueError):
    """Malformed or inadmissible note-role assignment."""


@dataclass(frozen=True, slots=True)
class RelationalTargetAssignment:
    """One source-supported role on an exact interval of one canonical note."""

    assignment_id: str
    dataset_id: str
    piece_id: str
    note_id: str
    flag: str
    start_qn: RationalTime
    end_qn: RationalTime
    relation_id: str
    source_event_ids: tuple[str, ...]
    related_note_ids: tuple[str, ...]
    source_version_id: str
    annotation_origin: str
    annotation_status: str
    admitted_weak: bool
    model_requested: str
    reasoning_effort: str

    def __post_init__(self) -> None:
        for name in (
            "assignment_id", "dataset_id", "piece_id", "note_id", "flag",
            "relation_id", "source_version_id", "annotation_origin",
            "annotation_status", "model_requested", "reasoning_effort",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise RelationalTargetError(f"{name} must be a non-empty stripped string")
        if self.flag not in RELATIONAL_FLAGS:
            raise RelationalTargetError(f"unknown relational flag {self.flag!r}")
        if not isinstance(self.admitted_weak, bool):
            raise RelationalTargetError("admitted_weak must be a boolean")
        if not all(isinstance(t, RationalTime) for t in (self.start_qn, self.end_qn)):
            raise RelationalTargetError("assignment timing must use RationalTime")
        if not self.start_qn < self.end_qn:
            raise RelationalTargetError("assignment requires a nonempty exact interval")
        for name in ("source_event_ids", "related_note_ids"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(not isinstance(v, str) or not v or v != v.strip() for v in values)
                or len(values) != len(set(values))
            ):
                raise RelationalTargetError(f"{name} must contain unique exact IDs")
        if not self.source_event_ids:
            raise RelationalTargetError("assignment requires source_event_ids")
        if self.admitted_weak and self.annotation_status not in {
            "admitted_weak", "astra_inference", "source_supported",
        }:
            raise RelationalTargetError("unaccepted annotation status cannot be admitted")


def _ordered_assignments(
    assignments: Iterable[RelationalTargetAssignment],
) -> tuple[RelationalTargetAssignment, ...]:
    by_id: dict[str, RelationalTargetAssignment] = {}
    for assignment in assignments:
        if not isinstance(assignment, RelationalTargetAssignment):
            raise RelationalTargetError("assignments must be typed records")
        if assignment.assignment_id in by_id:
            raise RelationalTargetError("duplicate assignment_id")
        by_id[assignment.assignment_id] = assignment
    return tuple(by_id[key] for key in sorted(by_id))


def build_relational_target_bundle(
    piece: CanonicalPiece,
    assignments: Iterable[RelationalTargetAssignment],
    *,
    analysis_view_id: str,
) -> TargetBundle:
    """Project admitted role participation onto every note without negatives.

    Only canonical TRAIN pieces are admitted. This check supplements the
    exporter's corpus-native and global split admission; it cannot establish
    the provenance or musical correctness of an input assignment by itself.
    """

    if piece.split != "train":
        raise RelationalTargetError("relational weak targets require a TRAIN canonical piece")
    ordered = _ordered_assignments(assignments)
    notes = {note.note_id: note for note in piece.notes}
    if len(notes) != len(piece.notes):
        raise RelationalTargetError("canonical note IDs must be unique")
    grouped: dict[tuple[str, str], list[RelationalTargetAssignment]] = {}
    for assignment in ordered:
        if (assignment.dataset_id, assignment.piece_id) != (
            piece.dataset_name, piece.piece_id,
        ):
            raise RelationalTargetError("assignment belongs to a different canonical piece")
        if assignment.note_id not in notes or not set(assignment.related_note_ids) <= notes.keys():
            raise RelationalTargetError("assignment references an absent canonical note")
        note = notes[assignment.note_id]
        if (
            assignment.start_qn < note.onset_qn
            or assignment.end_qn > note.onset_qn + note.duration_qn
        ):
            raise RelationalTargetError("assignment interval lies outside its canonical note")
        if assignment.admitted_weak:
            grouped.setdefault((assignment.flag, assignment.note_id), []).append(assignment)
    provenance = []
    targets = []
    entity_ids = tuple(notes)
    for spec in RELATIONAL_TARGET_FAMILIES:
        flag = spec.task_id.rsplit(".", 1)[1]
        values = []
        sources = []
        provenance_ids = []
        masks = []
        for note_id in entity_ids:
            accepted = grouped.get((flag, note_id), [])
            if not accepted:
                values.append(None)
                sources.append(None)
                provenance_ids.append(None)
                masks.append(False)
                continue
            payload = dumps_relational_assignments(accepted)
            fingerprint = sha256(payload.encode("utf-8")).hexdigest()
            provenance_id = f"prov:relational-target:{fingerprint}"
            provenance.append(ProvenanceRecord(
                provenance_id=provenance_id,
                kind="annotation",
                source="pseudo_label",
                record_id=note_id,
                uri=None,
                version=RELATIONAL_ASSIGNMENT_CONTRACT_VERSION,
                checksum_sha256=fingerprint,
                created_at=None,
                parents=(),
                details=(
                    ("assignment_evidence_json", payload),
                    ("projection", "note_participates_in_admitted_relation"),
                    ("flag", flag),
                    ("confidence", None),
                ),
            ))
            values.append("present")
            sources.append("pseudo_label")
            provenance_ids.append(provenance_id)
            masks.append(True)
        targets.append(SampleTarget(
            task_id=spec.task_id,
            annotation_view_id=None,
            alignment_type="note",
            entity_ids=entity_ids,
            values=tuple(values),
            availability_mask=tuple(masks),
            confidence=(None,) * len(entity_ids),
            source=tuple(sources),
            provenance_ids=tuple(provenance_ids),
        ))
    return TargetBundle(
        contract_version=TARGET_BUNDLE_CONTRACT_VERSION,
        registry_extension_ids=(RELATIONAL_TARGET_REGISTRY_ID,),
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
        analysis_view_id=analysis_view_id,
        alignment_spans=(),
        targets=tuple(targets),
        provenance=tuple(provenance),
        diagnostics=(),
    )


def dumps_relational_assignments(
    assignments: Iterable[RelationalTargetAssignment], *, indent: int | None = None,
) -> str:
    """Serialize exact per-event evidence separately from note participation."""

    return json.dumps(
        {
            "contract_version": RELATIONAL_ASSIGNMENT_CONTRACT_VERSION,
            "assignments": [asdict(item) for item in _ordered_assignments(assignments)],
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def loads_relational_assignments(payload: str) -> tuple[RelationalTargetAssignment, ...]:
    """Decode a versioned assignment sidecar without rounding exact timing."""

    try:
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or set(value) != {"contract_version", "assignments"}
            or value["contract_version"] != RELATIONAL_ASSIGNMENT_CONTRACT_VERSION
            or not isinstance(value["assignments"], list)
        ):
            raise RelationalTargetError("assignment sidecar root is invalid")
        decoded = []
        for raw in value["assignments"]:
            raw = dict(raw)
            for name in ("start_qn", "end_qn"):
                raw[name] = RationalTime(**raw[name])
            for name in ("source_event_ids", "related_note_ids"):
                if not isinstance(raw[name], list):
                    raise RelationalTargetError(f"{name} must be a JSON list")
                raw[name] = tuple(raw[name])
            decoded.append(RelationalTargetAssignment(**raw))
        return _ordered_assignments(decoded)
    except (ValueError, TypeError, KeyError) as exc:
        raise RelationalTargetError(f"relational assignment parse invalid: {exc}") from exc
