"""Opt-in note-role targets, separate from the historical core ontology."""

from types import MappingProxyType

from music_critic.tasks.ontology import NOTE_IDENTITY_ALIGNMENT, TargetFamilySpec


RELATIONAL_TARGET_REGISTRY_VERSION = "1.0.0"
RELATIONAL_TARGET_REGISTRY_ID = "music_critic.relations@1.0.0"
RELATIONAL_TARGET_ENCODING_REGISTRY_VERSION = "1.0.0"
RELATIONAL_FLAGS = (
    "held_common_tone",
    "resolving_chordal_seventh",
    "resolution_arrival",
    "anticipation",
    "anticipated_arrival",
    "contrary_ascending",
    "contrary_descending",
    "octave_upper",
    "octave_lower",
)


def _spec(flag: str) -> TargetFamilySpec:
    return TargetFamilySpec(
        task_id=f"theory.relations.{flag}",
        registry_version=RELATIONAL_TARGET_REGISTRY_VERSION,
        semantic_description=(
            f"A canonical note participates as {flag} in an admitted weak "
            "musical relation; exact subspans remain in the assignment sidecar."
        ),
        canonical_dtype="str",
        value_type="categorical",
        vocabulary=("absent", "present"),
        open_vocabulary=None,
        target_entity="canonical note",
        granularity="canonical_note_identity",
        source_alignment_type="note",
        time_unit="exact rational quarter notes",
        interval_semantics=(
            "note-level participation projection; not a claim that the role "
            "holds throughout the note; sidecar retains exact half-open subspans"
        ),
        supervision_context="source_bound_relational_weak_labels",
        source_adapter="music_critic.tasks.relational_targets@1.0.0",
        annotation_view_id=None,
        missing_value_semantics="unannotated, rejected and ambiguous are unknown",
        availability_mask_required=True,
        provenance_required=True,
        confidence_policy="null; no calibrated confidence from model annotation",
        supervision_objective="positive_unlabeled_event_detection",
        negative_example_policy=(
            "absent is reserved and never emitted; no unannotated note or "
            "missing flag is a negative example"
        ),
        alignment_policy=NOTE_IDENTITY_ALIGNMENT,
        cross_source_sharing="forbidden",
    )


RELATIONAL_TARGET_FAMILIES = tuple(
    sorted((_spec(flag) for flag in RELATIONAL_FLAGS), key=lambda spec: spec.task_id)
)
RELATIONAL_TARGET_FAMILY_BY_ID = MappingProxyType(
    {spec.task_id: spec for spec in RELATIONAL_TARGET_FAMILIES}
)
