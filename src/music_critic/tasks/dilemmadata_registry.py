"""Source-native Phase 9B.2A Dilemmadata target-family registry.

This registry is deliberately separate from the accepted HookTheory/POP909-CL
ontology.  Its tasks can be attached as an explicit sample sidecar extension
without changing raw canonical cache or split identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal

from music_critic.tasks.ontology import (
    BOUNDARY_EVENT_ALIGNMENT,
    NOTE_IDENTITY_ALIGNMENT,
    REGION_SPAN_ALIGNMENT,
    AlignmentPolicy,
    TargetFamilySpec,
)


DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION = "1.0.0"
DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION = "1.0.0"
DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION = "1.0.0"
DILEMMADATA_TARGET_ADAPTER_SOURCE = (
    "music_critic.adapters.dilemmadata_targets@1.0.0"
)

DilemmadataDialect = Literal["an_joint", "dlc"]
DilemmadataTargetCoordinate = Literal[
    "global_span",
    "annotation_run",
    "exact_onset_event",
    "canonical_note_identity",
]
DilemmadataEncodingMode = Literal[
    "closed_categorical",
    "open_string_cpu",
    "positive_unlabeled",
]


@dataclass(frozen=True, slots=True)
class DilemmadataSourceFamilySpec:
    """Extraction evidence for one dialect-specific target family."""

    task_id: str
    dialect: DilemmadataDialect
    family: str
    primary_field: str
    source_fields: tuple[str, ...]
    gate_field: str | None
    source_identity_field: str | None
    coordinate: DilemmadataTargetCoordinate
    encoding_mode: DilemmadataEncodingMode
    vocabulary: tuple[str, ...] | None
    mapping_status: Literal["source_specific", "deferred_crosswalk"]
    ontology_spec: TargetFamilySpec

    def __post_init__(self) -> None:
        if self.task_id != self.ontology_spec.task_id:
            raise ValueError("Dilemmadata source family task differs from ontology spec")
        if self.encoding_mode in {"closed_categorical", "positive_unlabeled"}:
            if not self.vocabulary:
                raise ValueError("closed Dilemmadata family requires a vocabulary")
        elif self.vocabulary is not None:
            raise ValueError("open Dilemmadata family cannot define a vocabulary")
        if self.coordinate == "canonical_note_identity":
            expected = NOTE_IDENTITY_ALIGNMENT.policy_id
        elif self.coordinate == "exact_onset_event":
            expected = BOUNDARY_EVENT_ALIGNMENT.policy_id
        else:
            expected = REGION_SPAN_ALIGNMENT.policy_id
        if self.ontology_spec.alignment_policy.policy_id != expected:
            raise ValueError("Dilemmadata source coordinate/alignment mismatch")


_AN_QUALITY = (
    "Augmented Fourth",
    "Diminished Fifth",
    "French augmented sixth chord",
    "French augmented sixth chord in first inversion",
    "French augmented sixth chord in root position",
    "French augmented sixth chord in third inversion",
    "German augmented sixth chord",
    "German augmented sixth chord in root position",
    "German augmented sixth chord in second inversion",
    "German augmented sixth chord in third inversion",
    "Italian augmented sixth chord",
    "Italian augmented sixth chord in root position",
    "Italian augmented sixth chord in second inversion",
    "Kumoi pentachord",
    "Major Second",
    "Major Seventh",
    "Major Sixth",
    "Major Third",
    "Minor Sixth",
    "Minor Third",
    "Perfect Fifth",
    "Perfect Fourth",
    "augmented major tetrachord",
    "augmented seventh chord",
    "augmented triad",
    "diminished seventh chord",
    "diminished triad",
    "diminished-major ninth chord",
    "dominant seventh chord",
    "dominant-ninth",
    "enharmonic equivalent to diminished triad",
    "enharmonic equivalent to half-diminished seventh chord",
    "enharmonic equivalent to major triad",
    "enharmonic equivalent to minor seventh chord",
    "enharmonic equivalent to minor triad",
    "enharmonic to dominant seventh chord",
    "flat-ninth pentachord",
    "half-diminished seventh chord",
    "incomplete dominant-seventh chord",
    "incomplete half-diminished seventh chord",
    "incomplete major-seventh chord",
    "incomplete minor-seventh chord",
    "lydian tetrachord",
    "major seventh chord",
    "major triad",
    "major-minor tetramirror",
    "major-ninth chord",
    "major-second major tetrachord",
    "major-second minor tetrachord",
    "minor seventh chord",
    "minor triad",
    "minor trichord",
    "minor-augmented tetrachord",
    "minor-diminished ninth chord",
    "minor-ninth chord",
    "note",
    "perfect-fourth diminished tetrachord",
    "perfect-fourth major tetrachord",
    "perfect-fourth minor tetrachord",
    "phrygian tetrachord",
    "quartal tetramirror",
    "quartal trichord",
    "whole-tone tetramirror",
    "whole-tone trichord",
)
_DLC_QUALITY = (
    "%7",
    "+",
    "+7",
    "+M7",
    "Fr",
    "Ger",
    "It",
    "M",
    "MM7",
    "Mm7",
    "m",
    "mM7",
    "mm7",
    "o",
    "o7",
)
_AN_INVERSION = ("0", "1", "2", "3")
_DLC_INVERSION = ("2", "43", "6", "64", "65", "7")
_DLC_CADENCE = ("DC", "EC", "HC", "IAC", "PAC", "PC")


def _ontology(
    task_id: str,
    *,
    dialect: DilemmadataDialect,
    family: str,
    coordinate: DilemmadataTargetCoordinate,
    vocabulary: tuple[str, ...] | None,
    positive_unlabeled: bool = False,
) -> TargetFamilySpec:
    alignment: AlignmentPolicy
    if coordinate == "canonical_note_identity":
        alignment = NOTE_IDENTITY_ALIGNMENT
        source_alignment_type = "note"
    elif coordinate == "exact_onset_event":
        alignment = BOUNDARY_EVENT_ALIGNMENT
        source_alignment_type = "annotation_span"
    else:
        alignment = REGION_SPAN_ALIGNMENT
        source_alignment_type = "annotation_span"
    context = f"dilemmadata_{'an' if dialect == 'an_joint' else 'dlc'}_source_native_theory"
    return TargetFamilySpec(
        task_id=task_id,
        registry_version=DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
        semantic_description=(
            f"Dilemmadata {dialect} source-native {family}; no cross-source normalization."
        ),
        canonical_dtype="str",
        value_type="categorical",
        vocabulary=vocabulary,
        open_vocabulary=(
            None
            if vocabulary is not None
            else "lossless source string; no runtime vocabulary construction"
        ),
        target_entity=(
            "canonical note"
            if coordinate == "canonical_note_identity"
            else "target-only exact alignment span"
        ),
        granularity=coordinate,
        source_alignment_type=source_alignment_type,
        time_unit="exact rational quarter notes",
        interval_semantics=(
            "exact canonical note identity"
            if coordinate == "canonical_note_identity"
            else (
                "point event at exact source onset"
                if coordinate == "exact_onset_event"
                else "exact half-open run; terminal unproved end remains unaligned"
            )
        ),
        supervision_context=context,
        source_adapter=DILEMMADATA_TARGET_ADAPTER_SOURCE,
        annotation_view_id=None,
        missing_value_semantics="missing, masked, unsupported, and ambiguous are not negatives",
        availability_mask_required=True,
        provenance_required=True,
        confidence_policy="null; the release supplies no calibrated numeric confidence",
        supervision_objective=(
            "positive_unlabeled_event_detection"
            if positive_unlabeled
            else (
                "deferred_open_vocabulary"
                if vocabulary is None
                else "masked_source_native_classification"
            )
        ),
        negative_example_policy=(
            "no absent class; unannotated candidates are unlabeled"
            if positive_unlabeled
            else "only explicit available source labels are classes"
        ),
        alignment_policy=alignment,
        cross_source_sharing="forbidden",
    )


def _source(
    task_id: str,
    *,
    dialect: DilemmadataDialect,
    family: str,
    primary: str,
    fields: tuple[str, ...],
    gate: str | None,
    identity: str | None,
    coordinate: DilemmadataTargetCoordinate,
    mode: DilemmadataEncodingMode,
    vocabulary: tuple[str, ...] | None = None,
    mapping: Literal["source_specific", "deferred_crosswalk"] = "source_specific",
) -> DilemmadataSourceFamilySpec:
    return DilemmadataSourceFamilySpec(
        task_id=task_id,
        dialect=dialect,
        family=family,
        primary_field=primary,
        source_fields=fields,
        gate_field=gate,
        source_identity_field=identity,
        coordinate=coordinate,
        encoding_mode=mode,
        vocabulary=vocabulary,
        mapping_status=mapping,
        ontology_spec=_ontology(
            task_id,
            dialect=dialect,
            family=family,
            coordinate=coordinate,
            vocabulary=vocabulary,
            positive_unlabeled=mode == "positive_unlabeled",
        ),
    )


_AN = "an_joint"
_DLC = "dlc"
_AN_ID = "a_annotationNumber"
_DLC_ID = "unfolded_harmony_index"

DILEMMADATA_SOURCE_FAMILIES = tuple(
    sorted(
        (
            _source("dilemmadata.an.key.local", dialect=_AN, family="local_key", primary="a_localKey", fields=("a_localKey",), gate=None, identity=_AN_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.an.chord.boundary", dialect=_AN, family="chord_boundary", primary="a_isOnset", fields=("a_isOnset",), gate="valid_chord_label", identity=None, coordinate="exact_onset_event", mode="positive_unlabeled", vocabulary=("present",)),
            _source("dilemmadata.an.harmony.roman_numeral", dialect=_AN, family="roman_numeral", primary="a_romanNumeral", fields=("a_romanNumeral", "a_simpleNumeral", "a_degree1", "a_degree2"), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.an.chord.root", dialect=_AN, family="chord_root", primary="a_root", fields=("a_root", "a_degree1"), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="open_string_cpu", mapping="deferred_crosswalk"),
            _source("dilemmadata.an.chord.quality", dialect=_AN, family="chord_quality", primary="a_quality", fields=("a_quality",), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="closed_categorical", vocabulary=_AN_QUALITY),
            _source("dilemmadata.an.chord.bass", dialect=_AN, family="bass", primary="a_bass", fields=("a_bass",), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="open_string_cpu", mapping="deferred_crosswalk"),
            _source("dilemmadata.an.chord.inversion", dialect=_AN, family="inversion", primary="a_inversion", fields=("a_inversion",), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="closed_categorical", vocabulary=_AN_INVERSION),
            _source("dilemmadata.an.harmony.applied", dialect=_AN, family="applied_secondary_harmony", primary="a_degree2", fields=("a_degree2", "a_tonicizedKey"), gate="valid_chord_label", identity=_AN_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.an.note.scale_degree", dialect=_AN, family="note_degree", primary="note_degree", fields=("note_degree",), gate="valid_chord_label", identity=None, coordinate="canonical_note_identity", mode="open_string_cpu"),
            _source("dilemmadata.dlc.key.global", dialect=_DLC, family="global_key", primary="globalkey", fields=("globalkey", "globalkey_tpc", "globalkey_mode"), gate=None, identity=None, coordinate="global_span", mode="open_string_cpu"),
            _source("dilemmadata.dlc.key.local", dialect=_DLC, family="local_key", primary="localkey", fields=("localkey", "localkey_tpc", "localkey_mode", "localkey_is_minor"), gate=None, identity=_DLC_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.dlc.chord.boundary", dialect=_DLC, family="chord_boundary", primary="a_isOnset", fields=("a_isOnset",), gate="valid_chord_label", identity=None, coordinate="exact_onset_event", mode="positive_unlabeled", vocabulary=("present",)),
            _source("dilemmadata.dlc.harmony.roman_numeral", dialect=_DLC, family="roman_numeral", primary="label", fields=("label", "numeral", "relativeroot", "a_simpleNumeral", "a_degree1", "a_degree2"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.dlc.chord.root", dialect=_DLC, family="chord_root", primary="root_tpc", fields=("root", "root_tpc", "a_root", "a_degree1"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="open_string_cpu", mapping="deferred_crosswalk"),
            _source("dilemmadata.dlc.chord.quality", dialect=_DLC, family="chord_quality", primary="chord_type", fields=("chord_type", "a_quality"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="closed_categorical", vocabulary=_DLC_QUALITY),
            _source("dilemmadata.dlc.chord.bass", dialect=_DLC, family="bass", primary="bass_note_tpc", fields=("bass_note", "bass_note_tpc", "a_bass"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="open_string_cpu", mapping="deferred_crosswalk"),
            _source("dilemmadata.dlc.chord.inversion", dialect=_DLC, family="inversion", primary="figbass", fields=("figbass", "a_inversion"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="closed_categorical", vocabulary=_DLC_INVERSION),
            _source("dilemmadata.dlc.harmony.applied", dialect=_DLC, family="applied_secondary_harmony", primary="relativeroot", fields=("relativeroot", "relativeroot_resolved", "applied_to_numeral", "a_degree2"), gate="valid_chord_label", identity=_DLC_ID, coordinate="annotation_run", mode="open_string_cpu"),
            _source("dilemmadata.dlc.cadence", dialect=_DLC, family="cadence", primary="cadence_type", fields=("cadence", "cadence_type", "cadence_subtype"), gate="valid_cadence_label", identity=None, coordinate="exact_onset_event", mode="positive_unlabeled", vocabulary=_DLC_CADENCE),
            _source("dilemmadata.dlc.phrase.boundary", dialect=_DLC, family="phrase_boundary", primary="a_phraseend", fields=("phraseend", "a_phraseend"), gate="valid_phrase_label", identity=None, coordinate="exact_onset_event", mode="positive_unlabeled", vocabulary=("present",)),
            _source("dilemmadata.dlc.section.boundary", dialect=_DLC, family="section_boundary", primary="section_start", fields=("section_start",), gate="valid_section_start_label", identity=None, coordinate="exact_onset_event", mode="positive_unlabeled", vocabulary=("present",)),
            _source("dilemmadata.dlc.note.scale_degree", dialect=_DLC, family="note_degree", primary="note_degree", fields=("note_degree",), gate="valid_chord_label", identity=None, coordinate="canonical_note_identity", mode="open_string_cpu"),
        ),
        key=lambda item: item.task_id,
    )
)

DILEMMADATA_TARGET_FAMILIES = tuple(
    item.ontology_spec for item in DILEMMADATA_SOURCE_FAMILIES
)
DILEMMADATA_SOURCE_FAMILY_BY_TASK = MappingProxyType(
    {item.task_id: item for item in DILEMMADATA_SOURCE_FAMILIES}
)
DILEMMADATA_TARGET_FAMILY_BY_ID = MappingProxyType(
    {item.task_id: item for item in DILEMMADATA_TARGET_FAMILIES}
)
if len(DILEMMADATA_TARGET_FAMILY_BY_ID) != len(DILEMMADATA_TARGET_FAMILIES):
    raise RuntimeError("Dilemmadata target registry contains duplicate task IDs")

DILEMMADATA_TASK_IDS_BY_DIALECT = MappingProxyType(
    {
        dialect: tuple(
            item.task_id
            for item in DILEMMADATA_SOURCE_FAMILIES
            if item.dialect == dialect
        )
        for dialect in (_AN, _DLC)
    }
)

DILEMMADATA_DEFERRED_MAPPINGS = (
    "borrowed_harmony_unavailable",
    "staff_voice_to_semantic_role_incompatible",
    "tonal_region_alias_deferred",
    "an_dlc_crosswalk_deferred",
    "hooktheory_pop909_crosswalk_deferred",
    "root_crosswalk_deferred",
    "bass_crosswalk_deferred",
    "chord_quality_crosswalk_deferred",
)


def dilemmadata_family_registry_dict() -> dict[str, object]:
    return {
        "alignment_rules_version": DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
        "encoding_registry_version": DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
        "family_registry_version": DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
        "families": [asdict(item) for item in DILEMMADATA_SOURCE_FAMILIES],
        "deferred_mappings": list(DILEMMADATA_DEFERRED_MAPPINGS),
    }


def dumps_dilemmadata_family_registry(*, indent: int | None = None) -> str:
    return json.dumps(
        dilemmadata_family_registry_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def dilemmadata_family_registry_fingerprint() -> str:
    return sha256(dumps_dilemmadata_family_registry().encode("utf-8")).hexdigest()


__all__ = [
    "DILEMMADATA_DEFERRED_MAPPINGS",
    "DILEMMADATA_SOURCE_FAMILIES",
    "DILEMMADATA_SOURCE_FAMILY_BY_TASK",
    "DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION",
    "DILEMMADATA_TARGET_ADAPTER_SOURCE",
    "DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION",
    "DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION",
    "DILEMMADATA_TARGET_FAMILIES",
    "DILEMMADATA_TARGET_FAMILY_BY_ID",
    "DILEMMADATA_TASK_IDS_BY_DIALECT",
    "DilemmadataSourceFamilySpec",
    "dilemmadata_family_registry_dict",
    "dilemmadata_family_registry_fingerprint",
    "dumps_dilemmadata_family_registry",
]
