"""Versioned source-native target ontology and conservative crosswalk registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal

from music_critic.adapters.pop909_cl import (
    POP909_CL_TARGET_SEMANTICS_VERSION,
)

TARGET_ONTOLOGY_VERSION = "1.0.1"

CrosswalkStatus = Literal[
    "exact_shared",
    "derived_lossless_subset",
    "source_specific",
    "deferred",
    "incompatible",
]
CrossSourceSharing = Literal["forbidden", "conditional", "allowed"]


@dataclass(frozen=True, slots=True)
class CandidateAlignmentRule:
    """Machine-readable temporal semantics for one raw graph node type."""

    node_type: Literal["song", "track", "bar", "beat", "onset", "note"]
    geometry: Literal["identity", "point", "anchor"]
    time_reference: Literal["entity_id", "start_qn"]
    match_rule: Literal[
        "exact_entity_id",
        "half_open_containment",
        "exact_event_time",
    ]


@dataclass(frozen=True, slots=True)
class AlignmentPolicy:
    """How a source target may later be aligned without entering raw graph stores."""

    policy_id: str
    candidate_node_types: tuple[str, ...]
    candidate_rules: tuple[CandidateAlignmentRule, ...]
    multi_span_resolution: Literal["merge_equal_mask_conflicts"]
    conflict_diagnostic_code: Literal["multisource.alignment_conflict"]
    node_type_routing: Literal["explicit_per_entry"]
    unmatched_event_policy: Literal[
        "not_applicable",
        "retain_source_mask_alignment",
        "retain_event_mask_index_no_snap",
    ]
    ownership: str
    boundary_behavior: str
    empty_family_behavior: str
    ambiguous_behavior: str
    unsupported_behavior: str
    storage: Literal["sidecar_only"] = "sidecar_only"

    def __post_init__(self) -> None:
        if not self.policy_id or not self.candidate_node_types:
            raise ValueError("alignment policy requires an ID and candidate nodes")
        if not set(self.candidate_node_types) <= {
            "song",
            "track",
            "bar",
            "beat",
            "onset",
            "note",
        }:
            raise ValueError("alignment policy names an unknown raw node type")
        if tuple(rule.node_type for rule in self.candidate_rules) != (
            self.candidate_node_types
        ):
            raise ValueError("candidate rules must match candidate node ordering")
        if len(self.candidate_node_types) != len(set(self.candidate_node_types)):
            raise ValueError("candidate node types must be unique")


@dataclass(frozen=True, slots=True)
class TargetFamilySpec:
    """Declarative contract for one stable production-adapter task."""

    task_id: str
    registry_version: str
    semantic_description: str
    canonical_dtype: str
    value_type: Literal["categorical", "multi_label"]
    vocabulary: tuple[str, ...] | None
    open_vocabulary: str | None
    target_entity: str
    granularity: str
    source_alignment_type: Literal["note", "annotation_span"]
    time_unit: str
    interval_semantics: str
    supervision_context: str
    source_adapter: str
    annotation_view_id: str | None
    missing_value_semantics: str
    availability_mask_required: bool
    provenance_required: bool
    confidence_policy: str
    supervision_objective: str
    negative_example_policy: str
    alignment_policy: AlignmentPolicy
    cross_source_sharing: CrossSourceSharing

    def __post_init__(self) -> None:
        if (
            not isinstance(self.registry_version, str)
            or not self.registry_version.strip()
            or self.registry_version != self.registry_version.strip()
        ):
            raise ValueError("target spec registry version must be a non-empty string")
        if not self.task_id or "." not in self.task_id:
            raise ValueError("task_id must be a non-empty dotted stable identifier")
        if self.value_type == "multi_label" and not self.vocabulary:
            raise ValueError("multi-label target families require a closed vocabulary")
        if self.vocabulary is not None:
            if not self.vocabulary or len(self.vocabulary) != len(set(self.vocabulary)):
                raise ValueError("target vocabulary must be non-empty and unique")
            if self.open_vocabulary is not None:
                raise ValueError("a target cannot have closed and open vocabularies")
        elif not self.open_vocabulary:
            raise ValueError("open target vocabularies require a semantic description")
        if not self.availability_mask_required or not self.provenance_required:
            raise ValueError("Phase 5A target families require masks and provenance")
        if (
            self.source_alignment_type == "note"
            and self.alignment_policy.policy_id != "note_identity_v1"
        ):
            raise ValueError("note targets require note-identity alignment")


@dataclass(frozen=True, slots=True)
class CrosswalkSpec:
    """Classification of a source-native target or a potential cross-source pair."""

    crosswalk_id: str
    left_task_id: str | None
    right_task_id: str | None
    status: CrosswalkStatus
    prerequisites: tuple[str, ...]
    algorithm: str | None
    unavailable_policy: str
    provenance_policy: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.crosswalk_id:
            raise ValueError("crosswalk_id must be non-empty")
        if self.status not in {
            "exact_shared",
            "derived_lossless_subset",
            "source_specific",
            "deferred",
            "incompatible",
        }:
            raise ValueError("crosswalk status is unsupported")
        if self.status == "derived_lossless_subset":
            if not self.prerequisites or not self.algorithm:
                raise ValueError(
                    "derived_lossless_subset requires prerequisites and an algorithm"
                )
        elif self.algorithm is not None:
            raise ValueError("only accepted lossless subset mappings define algorithms")


NOTE_IDENTITY_ALIGNMENT = AlignmentPolicy(
    policy_id="note_identity_v1",
    candidate_node_types=("note",),
    candidate_rules=(
        CandidateAlignmentRule(
            node_type="note",
            geometry="identity",
            time_reference="entity_id",
            match_rule="exact_entity_id",
        ),
    ),
    multi_span_resolution="merge_equal_mask_conflicts",
    conflict_diagnostic_code="multisource.alignment_conflict",
    node_type_routing="explicit_per_entry",
    unmatched_event_policy="not_applicable",
    ownership="exact canonical entity_id identity",
    boundary_behavior="not_applicable",
    empty_family_behavior="emit an empty sidecar family with zero entries",
    ambiguous_behavior="mask the entry and retain source diagnostics",
    unsupported_behavior="mask the entry and retain source diagnostics",
)
REGION_SPAN_ALIGNMENT = AlignmentPolicy(
    policy_id="half_open_anchor_span_v1",
    candidate_node_types=("onset", "beat", "bar"),
    candidate_rules=(
        CandidateAlignmentRule(
            node_type="onset",
            geometry="point",
            time_reference="start_qn",
            match_rule="half_open_containment",
        ),
        CandidateAlignmentRule(
            node_type="beat",
            geometry="anchor",
            time_reference="start_qn",
            match_rule="half_open_containment",
        ),
        CandidateAlignmentRule(
            node_type="bar",
            geometry="anchor",
            time_reference="start_qn",
            match_rule="half_open_containment",
        ),
    ),
    multi_span_resolution="merge_equal_mask_conflicts",
    conflict_diagnostic_code="multisource.alignment_conflict",
    node_type_routing="explicit_per_entry",
    unmatched_event_policy="retain_source_mask_alignment",
    ownership=(
        "onset point time and beat/bar start anchors use exact half-open "
        "containment: span.start_qn <= candidate_time < span.end_qn"
    ),
    boundary_behavior=(
        "a candidate exactly at span end belongs to the following half-open span; "
        "the terminal piece boundary belongs only to the final raw interval"
    ),
    empty_family_behavior="emit an empty sidecar family with zero entries",
    ambiguous_behavior=(
        "equal available values for one typed candidate merge deterministically; "
        "conflicting available values are masked with a diagnostic"
    ),
    unsupported_behavior="mask entries; never synthesize a negative label",
)
BOUNDARY_EVENT_ALIGNMENT = AlignmentPolicy(
    policy_id="span_start_boundary_v1",
    candidate_node_types=("onset", "beat", "bar"),
    candidate_rules=(
        CandidateAlignmentRule(
            node_type="onset",
            geometry="point",
            time_reference="start_qn",
            match_rule="exact_event_time",
        ),
        CandidateAlignmentRule(
            node_type="beat",
            geometry="anchor",
            time_reference="start_qn",
            match_rule="exact_event_time",
        ),
        CandidateAlignmentRule(
            node_type="bar",
            geometry="anchor",
            time_reference="start_qn",
            match_rule="exact_event_time",
        ),
    ),
    multi_span_resolution="merge_equal_mask_conflicts",
    conflict_diagnostic_code="multisource.alignment_conflict",
    node_type_routing="explicit_per_entry",
    unmatched_event_policy="retain_event_mask_index_no_snap",
    ownership=(
        "the exact span start is the boundary event; Phase 5B may choose only an "
        "exact-time raw candidate or keep the event unaligned"
    ),
    boundary_behavior=(
        "no nearest-neighbor snapping and no implicit node-type priority; every "
        "aligned index carries its raw node type"
    ),
    empty_family_behavior="emit an empty sidecar family with zero entries",
    ambiguous_behavior="mask the boundary-to-node index and retain the event",
    unsupported_behavior="mask the boundary-to-node index and retain the event",
)
COVERAGE_SPAN_ALIGNMENT = AlignmentPolicy(
    policy_id="coverage_span_v1",
    candidate_node_types=("onset", "beat", "bar"),
    candidate_rules=REGION_SPAN_ALIGNMENT.candidate_rules,
    multi_span_resolution="merge_equal_mask_conflicts",
    conflict_diagnostic_code="multisource.alignment_conflict",
    node_type_routing="explicit_per_entry",
    unmatched_event_policy="retain_source_mask_alignment",
    ownership=(
        "onset point time and beat/bar start anchors use exact half-open "
        "containment in an explicitly available coverage span"
    ),
    boundary_behavior=(
        "leading/internal N spans are available; trailing uncovered regions remain "
        "masked and are not relabeled N"
    ),
    empty_family_behavior="emit an empty sidecar family with zero entries",
    ambiguous_behavior="mask the entry; ambiguity is not no-chord",
    unsupported_behavior="mask the entry; unsupported is not no-chord",
)


def _spec(
    task_id: str,
    *,
    description: str,
    value_type: Literal["categorical", "multi_label"],
    vocabulary: tuple[str, ...] | None,
    open_vocabulary: str | None = None,
    entity: str,
    granularity: str,
    interval: str,
    context: str,
    adapter: str,
    view: str | None,
    missing: str,
    alignment: AlignmentPolicy,
    supervision_objective: str = "masked_source_native_classification",
    negative_example_policy: str = "only explicit available source labels are negative",
) -> TargetFamilySpec:
    return TargetFamilySpec(
        task_id=task_id,
        registry_version=TARGET_ONTOLOGY_VERSION,
        semantic_description=description,
        canonical_dtype=(
            "tuple[str, ...]" if value_type == "multi_label" else "str"
        ),
        value_type=value_type,
        vocabulary=vocabulary,
        open_vocabulary=open_vocabulary,
        target_entity=entity,
        granularity=granularity,
        source_alignment_type=(
            "note"
            if alignment.policy_id == NOTE_IDENTITY_ALIGNMENT.policy_id
            else "annotation_span"
        ),
        time_unit="exact rational quarter notes",
        interval_semantics=interval,
        supervision_context=context,
        source_adapter=adapter,
        annotation_view_id=view,
        missing_value_semantics=missing,
        availability_mask_required=True,
        provenance_required=True,
        confidence_policy=(
            "nullable numeric confidence; null means not supplied, never zero or one"
        ),
        supervision_objective=supervision_objective,
        negative_example_policy=negative_example_policy,
        alignment_policy=alignment,
        cross_source_sharing="forbidden",
    )


_HT_CONTEXT = "melody_conditioned_harmony"
_POP_CONTEXT = "score_conditioned_harmony_recognition"
_HT_ADAPTER = "music_critic.adapters.hooktheory"
# ``source_adapter`` identifies immutable target-extraction semantics, not the
# current runtime/corpus-identity adapter release.
_POP_ADAPTER = (
    "music_critic.adapters.pop909_cl@"
    f"{POP909_CL_TARGET_SEMANTICS_VERSION}"
)
_PC_NAMES = (
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


TARGET_FAMILIES = tuple(
    sorted(
        (
            _spec(
                "theory.melody.scale_degree",
                description="HookTheory melody scale degree in the active local key.",
                value_type="categorical",
                vocabulary=tuple(
                    f"{accidental}{degree}"
                    for degree in range(1, 8)
                    for accidental in ("", "b", "#", "bb", "##")
                ),
                entity="canonical note",
                granularity="note",
                interval="exact note identity; note timing remains canonical",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="invalid, resting, or unresolved notes do not become negatives",
                alignment=NOTE_IDENTITY_ALIGNMENT,
            ),
            _spec(
                "theory.local_key.tonic_pc",
                description="HookTheory local-key tonic as an absolute pitch class.",
                value_type="categorical",
                vocabulary=tuple(str(value) for value in range(12)),
                entity="target-alignment key span",
                granularity="local_key_region",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="unresolved tonic is unavailable, not a tonic class",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "theory.local_key.mode",
                description="HookTheory normalized local-key mode.",
                value_type="categorical",
                vocabulary=None,
                open_vocabulary="normalized mode string accepted by the adapter",
                entity="target-alignment key span",
                granularity="local_key_region",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="unresolved mode is unavailable, not an unknown-mode label",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "theory.chord.presence",
                description="Whether the HookTheory chord span is a chord or an explicit rest.",
                value_type="categorical",
                vocabulary=("false", "true"),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="absence of a chord annotation is unavailable, not false",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "theory.chord.root_degree",
                description="HookTheory functional chord-root degree relative to local key.",
                value_type="categorical",
                vocabulary=tuple(str(value) for value in range(7)) + ("bVII",),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="rests, zero/invalid roots, and unresolved function are unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "theory.chord.extent",
                description="HookTheory source chord extent token.",
                value_type="categorical",
                vocabulary=("5", "7", "9", "11", "13"),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="rests and unsupported extents are unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "theory.chord.inversion",
                description="HookTheory ordinal inversion index.",
                value_type="categorical",
                vocabulary=("0", "1", "2", "3"),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="rests and unsupported ordinal inversions are unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            *(
                _spec(
                    f"theory.chord.{name}",
                    description=f"HookTheory source-native chord {name} tokens.",
                    value_type="multi_label",
                    vocabulary=labels,
                    entity="target-alignment chord span",
                    granularity="chord_span",
                    interval="half-open [start_qn, end_qn)",
                    context=_HT_CONTEXT,
                    adapter=_HT_ADAPTER,
                    view=None,
                    missing=f"rests and invalid {name} fields are unavailable",
                    alignment=REGION_SPAN_ALIGNMENT,
                )
                for name, labels in (
                    ("adds", ("4", "6", "9")),
                    ("omits", ("3", "5")),
                    ("alterations", ("b5", "#5", "b9", "#9", "#11", "b13")),
                    ("suspensions", ("2", "4")),
                )
            ),
            _spec(
                "theory.chord.borrowed",
                description="HookTheory source-native borrowed-chord status.",
                value_type="categorical",
                vocabulary=None,
                open_vocabulary="none, mode:<mode>, pcset:<pcs>, or preserved unknown:<text>",
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn)",
                context=_HT_CONTEXT,
                adapter=_HT_ADAPTER,
                view=None,
                missing="rests and structurally invalid borrowed values are unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "pop909_cl.chord.boundary",
                description="Directly observed POP909-CL channel-1 chord-block start.",
                value_type="categorical",
                vocabulary=("present",),
                entity="target-alignment chord span start",
                granularity="boundary_event",
                interval="point event at exact chord-span start",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="missing chord instrument is unavailable, not no boundary",
                alignment=BOUNDARY_EVENT_ALIGNMENT,
                supervision_objective="positive_unlabeled_event_detection",
                negative_example_policy=(
                    "no absent class in source-native evidence; non-boundary raw "
                    "candidates are unlabeled, not negative"
                ),
            ),
            _spec(
                "pop909_cl.chord.root",
                description="Derived absolute pitch-class root of a POP909-CL chord block.",
                value_type="categorical",
                vocabulary=_PC_NAMES,
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn), clipped only for raw duration",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="ambiguous or unsupported normalization remains unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "pop909_cl.chord.quality",
                description="Derived POP909-CL exact pitch-class-set quality.",
                value_type="categorical",
                vocabulary=(
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
                ),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn), clipped only for raw duration",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="unsupported or quality-disagreeing ambiguity is unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "pop909_cl.chord.bass",
                description="Directly observed lowest-note pitch class of a POP909-CL block.",
                value_type="categorical",
                vocabulary=_PC_NAMES,
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn), clipped only for raw duration",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="missing chord instrument is unavailable; bass has its own mask",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "pop909_cl.chord.inversion",
                description="Derived semitone distance from root to observed bass.",
                value_type="categorical",
                vocabulary=tuple(str(value) for value in range(12)),
                entity="target-alignment chord span",
                granularity="chord_span",
                interval="half-open [start_qn, end_qn), clipped only for raw duration",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="ambiguous or unsupported root leaves inversion unavailable",
                alignment=REGION_SPAN_ALIGNMENT,
            ),
            _spec(
                "pop909_cl.chord.no_chord",
                description=(
                    "Derived positive-only leading/internal POP909-CL no-chord "
                    "coverage span."
                ),
                value_type="categorical",
                vocabulary=("N",),
                entity="target-alignment coverage span",
                granularity="coverage_span",
                interval="half-open [start_qn, end_qn)",
                context=_POP_CONTEXT,
                adapter=_POP_ADAPTER,
                view="pop909_cl.channel_1",
                missing="trailing uncovered and missing-instrument spans stay unavailable",
                alignment=COVERAGE_SPAN_ALIGNMENT,
                supervision_objective=(
                    "positive_unlabeled_coverage_detection"
                ),
                negative_example_policy=(
                    "only explicit available N coverage spans are positive; "
                    "chord spans, uncovered candidates, and absent annotations "
                    "are unlabeled, not negative"
                ),
            ),
        ),
        key=lambda item: item.task_id,
    )
)

TARGET_FAMILY_BY_ID = MappingProxyType(
    {spec.task_id: spec for spec in TARGET_FAMILIES}
)
if len(TARGET_FAMILY_BY_ID) != len(TARGET_FAMILIES):
    raise RuntimeError("target ontology contains duplicate stable task IDs")


_PAIR_CROSSWALKS = (
    CrosswalkSpec(
        crosswalk_id="hooktheory_root_degree__pop909_cl_absolute_root",
        left_task_id="theory.chord.root_degree",
        right_task_id="pop909_cl.chord.root",
        status="incompatible",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="no automatic mapping; preserve both native masks",
        provenance_policy="source-native provenance remains unchanged",
        rationale=(
            "functional degree and absolute pitch-class root are different semantics; "
            "applied and borrowed harmony are unresolved"
        ),
    ),
    CrosswalkSpec(
        crosswalk_id="hooktheory_extent__pop909_cl_quality",
        left_task_id="theory.chord.extent",
        right_task_id="pop909_cl.chord.quality",
        status="incompatible",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="no automatic mapping; preserve both native masks",
        provenance_policy="source-native provenance remains unchanged",
        rationale="extent does not losslessly determine pitch-class-set quality",
    ),
    CrosswalkSpec(
        crosswalk_id="hooktheory_ordinal_inversion__pop909_cl_semitones",
        left_task_id="theory.chord.inversion",
        right_task_id="pop909_cl.chord.inversion",
        status="incompatible",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="no automatic mapping; bass and inversion masks stay independent",
        provenance_policy="source-native provenance remains unchanged",
        rationale="ordinal inversion and root-to-bass semitone distance are not equivalent",
    ),
    CrosswalkSpec(
        crosswalk_id="hooktheory_presence__pop909_cl_boundary",
        left_task_id="theory.chord.presence",
        right_task_id="pop909_cl.chord.boundary",
        status="incompatible",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="absence of annotation never becomes a negative boundary label",
        provenance_policy="source-native provenance remains unchanged",
        rationale="span presence and a chord-block boundary event answer different questions",
    ),
    CrosswalkSpec(
        crosswalk_id="hooktheory_presence__pop909_cl_no_chord",
        left_task_id="theory.chord.presence",
        right_task_id="pop909_cl.chord.no_chord",
        status="incompatible",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="missing, rest, trailing masked, and N remain distinct",
        provenance_policy="source-native provenance remains unchanged",
        rationale="HookTheory rest/presence semantics do not prove a POP909-CL gap-rule N",
    ),
    CrosswalkSpec(
        crosswalk_id="future_absolute_root_renderer",
        left_task_id="theory.chord.root_degree",
        right_task_id="pop909_cl.chord.root",
        status="deferred",
        prerequisites=(
            "versioned applied-harmony semantics",
            "versioned borrowed-chord semantics",
            "lossless local-key/root-degree crosswalk",
        ),
        algorithm=None,
        unavailable_policy="mask every unsupported or ambiguous HookTheory chord",
        provenance_policy="future derivation must reference all source target provenance",
        rationale="current HookTheory fields do not support a proven lossless absolute root",
    ),
    CrosswalkSpec(
        crosswalk_id="future_pitch_class_set_renderer",
        left_task_id="theory.chord.extent",
        right_task_id="pop909_cl.chord.quality",
        status="deferred",
        prerequisites=(
            "versioned source-chord renderer",
            "applied and borrowed semantics",
            "decoration handling",
        ),
        algorithm=None,
        unavailable_policy="mask unsupported or multi-candidate renderings",
        provenance_policy="future derived target must identify renderer and source parents",
        rationale="Phase 5A does not synthesize or infer target-derived chord tones",
    ),
)

_PAIRED_TASKS = {
    task_id
    for item in _PAIR_CROSSWALKS
    for task_id in (item.left_task_id, item.right_task_id)
    if task_id is not None
}
_SOURCE_SPECIFIC = tuple(
    CrosswalkSpec(
        crosswalk_id=f"source_specific__{spec.task_id.replace('.', '_')}",
        left_task_id=spec.task_id if spec.source_adapter == _HT_ADAPTER else None,
        right_task_id=spec.task_id if spec.source_adapter == _POP_ADAPTER else None,
        status="source_specific",
        prerequisites=(),
        algorithm=None,
        unavailable_policy="absent from the other source; never emit a negative label",
        provenance_policy="retain the source-native target provenance",
        rationale="no current production target has proven equivalent semantics",
    )
    for spec in TARGET_FAMILIES
    if spec.task_id not in _PAIRED_TASKS
)

CROSSWALKS = tuple(
    sorted((*_PAIR_CROSSWALKS, *_SOURCE_SPECIFIC), key=lambda item: item.crosswalk_id)
)
CROSSWALK_BY_ID = MappingProxyType(
    {item.crosswalk_id: item for item in CROSSWALKS}
)
if len(CROSSWALK_BY_ID) != len(CROSSWALKS):
    raise RuntimeError("target crosswalk registry contains duplicate IDs")


def ontology_contract_dict() -> dict[str, object]:
    """Return the complete deterministic registry/crosswalk mapping."""

    return json.loads(
        json.dumps(
            {
                "ontology_version": TARGET_ONTOLOGY_VERSION,
                "target_families": [asdict(item) for item in TARGET_FAMILIES],
                "crosswalks": [asdict(item) for item in CROSSWALKS],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def dumps_ontology_contract(*, indent: int | None = None) -> str:
    """Serialize the registry deterministically."""

    return json.dumps(
        ontology_contract_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def ontology_contract_fingerprint() -> str:
    """Return the SHA-256 of canonical compact registry serialization."""

    return sha256(dumps_ontology_contract().encode("utf-8")).hexdigest()


__all__ = [
    "AlignmentPolicy",
    "CandidateAlignmentRule",
    "CROSSWALKS",
    "CROSSWALK_BY_ID",
    "CrosswalkSpec",
    "CrosswalkStatus",
    "TARGET_FAMILIES",
    "TARGET_FAMILY_BY_ID",
    "TARGET_ONTOLOGY_VERSION",
    "TargetFamilySpec",
    "dumps_ontology_contract",
    "ontology_contract_dict",
    "ontology_contract_fingerprint",
]
