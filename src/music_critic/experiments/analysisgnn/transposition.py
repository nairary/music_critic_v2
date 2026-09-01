"""Phase 9E-B5A AnalysisGNN transposition policy and audit primitives.

This module deliberately separates the pinned public AnalysisGNN behaviour
from the corrected Music Critic V2 policy.  It does not import AnalysisGNN,
does not mutate canonical pieces or cached graphs, and contains no model,
training, or inference path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any, Literal

import torch

from music_critic.data import CanonicalPiece
from music_critic.experiments.analysisgnn.class_balance import (
    EntityTargetObservation,
    RecordTargetObservations,
    recommend_head_trainability,
    train_support_tier,
)
from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    ANALYSISGNN_REPOSITORY,
    fingerprint,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    PRODUCTION_TASKS,
    TASK_BY_ID,
    get_vocabulary,
)


TRANSPOSITION_AUDIT_SCHEMA = "DilemmadataAnalysisGNNTranspositionAudit@1.0.0"
TRANSPOSITION_CONTRACT_VERSION = "analysisgnn-transposition-contract-v1"
OFFICIAL_PROFILE_ID = "analysisgnn-official-transposition-e115182-v1"
CORRECTED_PROFILE_ID = "music-critic-v2-closed-transposition-v1"
SHIFT_PCS = tuple(range(12))
SIGNED_SEMITONES = (0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1)
SIGNED_BY_SHIFT_PC = dict(zip(SHIFT_PCS, SIGNED_SEMITONES, strict=True))
OFFICIAL_INTERVALS = (
    "P1",
    "m2",
    "M2",
    "m3",
    "M3",
    "P4",
    "A4",
    "P5",
    "m6",
    "M6",
    "m7",
    "M7",
)
OFFICIAL_SEMITONES = dict(zip(OFFICIAL_INTERVALS, SHIFT_PCS, strict=True))

TRANSFORMATION_KINDS = (
    "absolute_pitch_transpose",
    "pitch_class_set_transpose",
    "relative_label_invariant",
    "structural_label_invariant",
    "boolean_label_invariant",
    "recompute_from_transposed_dependencies",
    "unsupported_or_ambiguous",
)
MAPPING_INVALID_REASONS = (
    "target_oov",
    "ambiguous_spelling",
    "missing_context",
    "unsupported_source_value",
    "non_bijective_mapping",
)
ABSOLUTE_TASKS = ("local_key", "tonicized_key", "root", "bass")
PITCH_CLASS_SET_TASKS = ("pitch_class_set",)
RELATIVE_TASKS = (
    "primary_degree",
    "secondary_degree",
    "quality",
    "inversion",
    "roman_numeral",
    "note_degree",
)
STRUCTURAL_TASKS = (
    "harmonic_rhythm",
    "cadence",
    "phrase",
    "section",
    "metrical_strength",
)
BOOLEAN_TASKS = ("pedal", "chord_tone", "is_root", "is_bass")
PITCH_DEPENDENT_GRAPH_FEATURES = (
    "note.pitch",
    "note.pitch_class",
    "note.octave",
)
RECOMPUTED_GRAPH_FEATURES = ("note.track_relative_pitch",)
INVARIANT_GRAPH_FIELDS = (
    "node_counts",
    "edge_counts",
    "node_entity_ids",
    "edge_topology",
    "onset_times",
    "durations",
    "measure_positions",
    "beat_positions",
    "meter",
    "tempo",
    "track_part_staff_voice_ownership",
    "tie_structure",
    "grace_structure",
    "repair_evidence",
    "source_provenance",
    "feature_availability_masks_except_recomputed_track_relative_pitch",
    "target_availability_masks",
)

OFFICIAL_EVIDENCE_FILES = (
    {
        "path": "analysisgnn/utils/globals.py",
        "sha256": "205886a94409dba5c9a41c393be3b8714163b0b1f828221ef19fc7b2973b86da",
        "symbols": ["TRANSPOSITIONKEYS", "INTERVALCLASSES"],
    },
    {
        "path": "analysisgnn/utils/chord_representations.py",
        "sha256": "49be2e51e5d89f28e989b3c5045730938ce26996d23d54b74c1f597bac4adabc",
        "symbols": [
            "_getTranspositions",
            "TransposeKey",
            "TransposePitch",
            "TransposePcSet",
            "FeatureRepresentationTI",
            "OutputRepresentation",
            "NoteDegree49",
        ],
    },
    {
        "path": "analysisgnn/utils/dcl_tsv_utils.py",
        "sha256": "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
        "symbols": ["create_graph_from_df", "create_labels_dlc"],
    },
    {
        "path": "analysisgnn/utils/music.py",
        "sha256": "6366f6ec55905d845b18816b58659064a36acc4242084ba8dc00abef1bc573a7",
        "symbols": ["transpose_note_array", "PitchEncoder", "KeySignatureEncoder"],
    },
    {
        "path": "analysisgnn/data/datasets/dlc.py",
        "sha256": "3144af37692c708916f4d90924bc8b3dd63beceb2859013c6ef3b9db62853e36",
        "symbols": ["DLCGraphDataset", "DLCplusGraphDataset"],
    },
    {
        "path": "analysisgnn/data/datamodules/analysis.py",
        "sha256": "97546aad356e7632eb33714dddc930a758d6a8a6e5c04e1592b2b1d0f7797de4",
        "symbols": ["AnalysisDataModule.setup", "AnalysisDataModule.train_dataloader"],
    },
    {
        "path": "analysisgnn/train/train_analysisgnn.py",
        "sha256": "aac75dad00c3f637e96ae85b2bf8fd1b37f015f171d7f49afabdfb77c2d85a45",
        "symbols": ["--use_transpositions", "AnalysisDataModule"],
    },
)

_STEPS = "CDEFGAB"
_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SIGNED_DIATONIC_STEPS = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    -5: -3,
    -4: -2,
    -3: -2,
    -2: -1,
    -1: -1,
}
_OFFICIAL_DIATONIC_STEPS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
                             7: 4, 8: 5, 9: 5, 10: 6, 11: 6}
_OFFICIAL_KEY_FIFTH_DELTAS = {0: 0, 1: 7, 2: 2, 3: -3, 4: 4, 5: -1,
                              6: 8, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}


class AnalysisGNNTranspositionError(ValueError):
    """Raised when a transposition would violate the frozen policy."""


@dataclass(frozen=True, slots=True)
class TransformationSpec:
    task_id: str
    source_task_id: str
    entity_type: str
    transformation_kind: str
    input_dependencies: tuple[str, ...]
    output_vocabulary: str
    mask_policy: str
    official_behavior: str
    corrected_behavior: str
    evidence_reference: str


@dataclass(frozen=True, slots=True)
class SemanticMappingRow:
    source_task_id: str
    dialect: str
    source_semantic_value: str
    shift_pc: int
    target_semantic_value: str | None
    target_class_id: int | None
    spelling_strategy: str
    valid: bool
    invalid_reason: str | None
    round_trip_valid: bool
    composition_scope: str


@dataclass(frozen=True, slots=True)
class AugmentedGraphIdentity:
    source_record_id: str
    source_graph_fingerprint: str
    transposition_profile_id: str
    shift_pc: int
    signed_semitones: int

    @property
    def fingerprint(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class RecordShiftEligibility:
    record_id: str
    source_component_id: str
    dialect: str
    shift_pc: int
    signed_semitones: int
    official_valid: bool
    official_invalid_reasons: tuple[str, ...]
    corrected_valid: bool
    corrected_invalid_reasons: tuple[str, ...]
    graph_range_valid: bool
    all_targets_closed: bool
    round_trip_valid: bool
    collision_status: str


def official_transposition_evidence() -> dict[str, object]:
    """Return independently pinned public behaviour without correcting it."""

    payload: dict[str, object] = {
        "profile_id": OFFICIAL_PROFILE_ID,
        "repository": ANALYSISGNN_REPOSITORY,
        "commit": ANALYSISGNN_COMMIT,
        "evidence_files": list(OFFICIAL_EVIDENCE_FILES),
        "intervals": list(OFFICIAL_INTERVALS),
        "identity_interval": "P1",
        "identity_probability": (
            "not_parameterized; one materialized P1 view among successfully built views "
            "before the view-level train/validation split"
        ),
        "sampling_distribution": (
            "dataset-index and neighbor-loader driven; no per-record transform draw"
        ),
        "sampling_unit": "materialized_graph_view_then_neighbor_subgraph",
        "pitch_transform": {
            "dlc_graph_path": "(pitch + semitones) % 128",
            "cadence_helper_path": "(pitch + semitones) % 127",
            "midi_range_fail_closed": False,
            "octave_folding": "implicit modulo wrap defect",
        },
        "label_transform": {
            "absolute": "music21 interval transposition then class lookup",
            "pitch_class_set": "cyclic semitone shift then sorted tuple",
            "invariant": "FeatureRepresentationTI copy",
            "oov": "OutputRepresentation routes to final class index",
        },
        "enharmonic_spelling": "music21 positive named interval; path dependent",
        "rng": {
            "view_split": "sklearn train_test_split(random_state=0)",
            "loader": "implicit framework/GraphMuse RNG; no augmentation seed argument",
        },
        "split_behavior": {
            "test": "identity-only for hard-coded test_pieces",
            "train_validation": (
                "materialized transposed graph indices are split 90/10 after augmentation"
            ),
            "source_group_safe": False,
            "validation_identity_only": False,
        },
        "epoch_behavior": "same materialized views; loader sampling/order may vary",
        "audit_projection_semantics": (
            "official full-orbit/expected tables normalize the 12 requested views; "
            "they are not an attestation of runtime view success or loader probability"
        ),
        "official_valid_field_semantics": (
            "requested interval under pinned dataset policy before caught external "
            "encoder/materialization exceptions"
        ),
        "runtime_dependency_added": False,
        "known_defects": [
            "modulo pitch wrap",
            "inconsistent modulo 127/128 paths",
            "view-level TRAIN/VALIDATION leakage",
            "OOV-to-last-class fallback",
            "augmentation RNG not independently specified",
        ],
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def corrected_transposition_profile() -> dict[str, object]:
    payload: dict[str, object] = {
        "profile_id": CORRECTED_PROFILE_ID,
        "scope": "TRAIN_only_on_the_fly_view",
        "shift_pc": list(SHIFT_PCS),
        "signed_semitones": list(SIGNED_SEMITONES),
        "tritone_sign": "+6",
        "identity_shift_pc": 0,
        "sampling_distribution": "uniform_over_record_valid_shifts_including_identity",
        "sampling_unit": "one_shift_draw_per_source_record_per_epoch",
        "seed_contract": "sha256(profile, supplied_seed, epoch, source_record_id)",
        "validation_behavior": "identity_only",
        "test_behavior": "identity_only_without_target_access",
        "midi_range": "reject_entire_record_shift_if_any_non_drum_note_leaves_0_127",
        "octave_folding": False,
        "mask_policy": "preserve_exactly",
        "spelling_policy": (
            "source spelling plus signed diatonic interval; shift_pc=6 uses an "
            "involutive within-vocabulary pitch-class pairing"
        ),
        "vocabulary_policy": "semantic_decode_transform_lookup_fail_closed",
        "collision_policy": "exclude_variant_without_changing_split",
        "variants_are_independent_components": False,
        "runtime_dependency_added": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _spec(
    task_id: str,
    kind: str,
    dependencies: tuple[str, ...],
    official: str,
    corrected: str,
) -> TransformationSpec:
    task = TASK_BY_ID[task_id]
    return TransformationSpec(
        task_id=task_id,
        source_task_id=task.task_family,
        entity_type=task.entity_type,
        transformation_kind=kind,
        input_dependencies=dependencies,
        output_vocabulary=task.vocabulary_id,
        mask_policy="available/masked state and missing reason preserved exactly",
        official_behavior=official,
        corrected_behavior=corrected,
        evidence_reference=(
            f"{ANALYSISGNN_REPOSITORY}@{ANALYSISGNN_COMMIT}; "
            "Phase 9E-B3 source-native target contract"
        ),
    )


def transformation_registry() -> tuple[TransformationSpec, ...]:
    rows: list[TransformationSpec] = []
    for task in PRODUCTION_TASKS:
        task_id = task.task_id
        if task_id in {"local_key", "tonicized_key"}:
            rows.append(_spec(task_id, "absolute_pitch_transpose", (task_id,),
                              "music21 key transposition; OOV becomes final class",
                              "mode-preserving source spelling transform with closure"))
        elif task_id in {"root", "bass"}:
            rows.append(_spec(task_id, "absolute_pitch_transpose", (task_id, "local_key"),
                              "music21 pitch transposition; OOV becomes final class",
                              "source spelling transform with vocabulary closure"))
        elif task_id == "pitch_class_set":
            rows.append(_spec(task_id, "pitch_class_set_transpose", (task_id,),
                              "cyclic pc-set shift followed by class lookup",
                              "cyclic pc-set shift; reject target OOV"))
        elif task_id in RELATIVE_TASKS:
            official = (
                "NoteDegree49 inherits OutputRepresentationTI"
                if task_id == "note_degree"
                else "pinned transposition-invariant output representation"
            )
            rows.append(_spec(task_id, "relative_label_invariant", (task_id,), official,
                              "relative function is invariant when pitch and key move together"))
        elif task_id in STRUCTURAL_TASKS:
            rows.append(_spec(task_id, "structural_label_invariant", (task_id,),
                              "structural label unchanged or dataset-specific recomputation",
                              "structural semantics and masks remain invariant"))
        elif task_id in BOOLEAN_TASKS:
            rows.append(_spec(task_id, "boolean_label_invariant", (task_id,),
                              "boolean/structural relation remains unchanged",
                              "joint pitch/chord transposition preserves boolean relation"))
        else:  # pragma: no cover - registry completeness guard
            rows.append(_spec(task_id, "unsupported_or_ambiguous", (task_id,),
                              "not established", "fail closed"))
    if len(rows) != 20 or {row.task_id for row in rows} != set(TASK_BY_ID):
        raise AnalysisGNNTranspositionError("transformation registry must cover 20 heads")
    return tuple(rows)


def transposition_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": TRANSPOSITION_AUDIT_SCHEMA,
        "version": TRANSPOSITION_CONTRACT_VERSION,
        "official_profile_fingerprint": official_transposition_evidence()["fingerprint"],
        "corrected_profile_fingerprint": corrected_transposition_profile()["fingerprint"],
        "transformation_kinds": list(TRANSFORMATION_KINDS),
        "mapping_invalid_reasons": list(MAPPING_INVALID_REASONS),
        "transformation_registry": [asdict(row) for row in transformation_registry()],
        "pitch_dependent_graph_features": list(PITCH_DEPENDENT_GRAPH_FEATURES),
        "recomputed_graph_features": list(RECOMPUTED_GRAPH_FEATURES),
        "invariant_graph_fields": list(INVARIANT_GRAPH_FIELDS),
        "identity_must_be_valid": True,
        "round_trip_required_for_valid_mapping": True,
        "composition_guaranteed_for": ["pitch_class_set_when_intermediate_and_result_are_in_vocabulary"],
        "variants_are_independent_records": False,
        "variants_are_independent_components": False,
        "b4_thresholds_reused": True,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _parse_spelling(value: str) -> tuple[str, int, bool] | None:
    if not value or value[0].upper() not in _STEPS:
        return None
    suffix = value[1:]
    if any(char not in "#b" for char in suffix) or ("#" in suffix and "b" in suffix):
        return None
    alter = suffix.count("#") - suffix.count("b")
    return value[0].upper(), alter, value[0].islower()


def _render_spelling(step: str, alter: int, minor: bool) -> str:
    letter = step.lower() if minor else step
    return letter + ("#" * alter if alter >= 0 else "b" * -alter)


def _spelling_pc(value: str) -> int | None:
    parsed = _parse_spelling(value)
    if parsed is None:
        return None
    step, alter, _minor = parsed
    return (_NATURAL_PC[step] + alter) % 12


def _alter_for_pc(step: str, pitch_class: int) -> int:
    value = (pitch_class - _NATURAL_PC[step]) % 12
    return value - 12 if value > 6 else value


def _spelling_rank(value: str) -> tuple[int, int, int, str]:
    parsed = _parse_spelling(value)
    if parsed is None:
        return (999, 999, 999, value)
    step, alter, _minor = parsed
    direction = 0 if alter == 0 else (1 if alter > 0 else 2)
    return (abs(alter), direction, _STEPS.index(step), value)


def _tritone_partner(value: str, vocabulary: Sequence[str]) -> str | None:
    parsed = _parse_spelling(value)
    if parsed is None:
        return None
    _step, _alter, minor = parsed
    pc = _spelling_pc(value)
    assert pc is not None
    target_pc = (pc + 6) % 12
    source_group = sorted(
        (item for item in vocabulary if _spelling_pc(item) == pc and item[0].islower() == minor),
        key=_spelling_rank,
    )
    target_group = sorted(
        (item for item in vocabulary if _spelling_pc(item) == target_pc and item[0].islower() == minor),
        key=_spelling_rank,
    )
    try:
        ordinal = source_group.index(value)
    except ValueError:
        return None
    return target_group[ordinal] if ordinal < len(target_group) else None


def _corrected_spelling_target(
    value: str, shift_pc: int, vocabulary: Sequence[str]
) -> tuple[str | None, str]:
    if shift_pc == 0:
        return value, "identity"
    if shift_pc == 6:
        return _tritone_partner(value, vocabulary), "involutive_pc_rank_tritone"
    parsed = _parse_spelling(value)
    if parsed is None:
        return None, "unsupported_source_spelling"
    step, _alter, minor = parsed
    signed = SIGNED_BY_SHIFT_PC[shift_pc]
    step_delta = _SIGNED_DIATONIC_STEPS[signed]
    target_step = _STEPS[(_STEPS.index(step) + step_delta) % 7]
    target_pc = ((_spelling_pc(value) or 0) + shift_pc) % 12
    target = _render_spelling(target_step, _alter_for_pc(target_step, target_pc), minor)
    return target, "source_spelling_signed_diatonic_interval"


def _official_spelling_target(value: str, shift_pc: int) -> str | None:
    parsed = _parse_spelling(value)
    if parsed is None:
        return None
    step, _alter, minor = parsed
    target_step = _STEPS[(_STEPS.index(step) + _OFFICIAL_DIATONIC_STEPS[shift_pc]) % 7]
    target_pc = ((_spelling_pc(value) or 0) + shift_pc) % 12
    return _render_spelling(target_step, _alter_for_pc(target_step, target_pc), minor)


def _pcset_target(value: str, shift_pc: int) -> str | None:
    try:
        values = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError:
        return None
    if not values or any(item < 0 or item > 11 for item in values):
        return None
    return ",".join(
        str(item) for item in sorted((item + shift_pc) % 12 for item in values)
    )


@lru_cache(maxsize=1)
def semantic_mapping_rows() -> tuple[SemanticMappingRow, ...]:
    rows: list[SemanticMappingRow] = []
    for task_id in (*ABSOLUTE_TASKS, *PITCH_CLASS_SET_TASKS):
        task = TASK_BY_ID[task_id]
        vocabulary = get_vocabulary(task.vocabulary_id)
        class_id = {value: index for index, value in enumerate(vocabulary.labels)}
        for dialect in task.source_dialects:
            for source in vocabulary.labels:
                for shift_pc in SHIFT_PCS:
                    if task_id in PITCH_CLASS_SET_TASKS:
                        target = _pcset_target(source, shift_pc)
                        strategy = "cyclic_pitch_class_set"
                    else:
                        target, strategy = _corrected_spelling_target(
                            source, shift_pc, vocabulary.labels
                        )
                    if target is None:
                        invalid = (
                            "non_bijective_mapping" if shift_pc == 6 else "unsupported_source_value"
                        )
                    elif target not in class_id:
                        invalid = "target_oov"
                    else:
                        invalid = None
                    round_trip = False
                    if invalid is None and target is not None:
                        inverse = (-shift_pc) % 12
                        if task_id in PITCH_CLASS_SET_TASKS:
                            back = _pcset_target(target, inverse)
                        else:
                            back, _ = _corrected_spelling_target(
                                target, inverse, vocabulary.labels
                            )
                        round_trip = back == source
                        if not round_trip:
                            invalid = "non_bijective_mapping"
                    rows.append(
                        SemanticMappingRow(
                            source_task_id=task_id,
                            dialect=dialect,
                            source_semantic_value=source,
                            shift_pc=shift_pc,
                            target_semantic_value=target,
                            target_class_id=class_id.get(target) if invalid is None else None,
                            spelling_strategy=f"{dialect}:{strategy}",
                            valid=invalid is None,
                            invalid_reason=invalid,
                            round_trip_valid=round_trip,
                            composition_scope=(
                                "closed_group_action_where_vocabulary_contains_each_pcset"
                                if task_id in PITCH_CLASS_SET_TASKS
                                else "diagnostic_only_no_global_spelling_group_claim"
                            ),
                        )
                    )
    if any(row.shift_pc == 0 and not row.valid for row in rows):
        raise AnalysisGNNTranspositionError("identity semantic mapping must always be valid")
    return tuple(rows)


@lru_cache(maxsize=1)
def semantic_mapping_index() -> dict[tuple[str, str, str, int], SemanticMappingRow]:
    return {
        (row.source_task_id, row.dialect, row.source_semantic_value, row.shift_pc): row
        for row in semantic_mapping_rows()
    }


def transform_semantic_value(
    task_id: str,
    value: str,
    *,
    shift_pc: int,
    dialect: str,
    profile: Literal["corrected_v2", "official_reproduction"] = "corrected_v2",
) -> str:
    if shift_pc not in SHIFT_PCS:
        raise AnalysisGNNTranspositionError("shift_pc must be in 0..11")
    if task_id in RELATIVE_TASKS or task_id in STRUCTURAL_TASKS or task_id in BOOLEAN_TASKS:
        return value
    if profile == "corrected_v2":
        row = semantic_mapping_index().get((task_id, dialect, value, shift_pc))
        if row is None or not row.valid or row.target_semantic_value is None:
            reason = "unsupported_source_value" if row is None else row.invalid_reason
            raise AnalysisGNNTranspositionError(
                f"{task_id} {value!r} shift {shift_pc} is not closed: {reason}"
            )
        return row.target_semantic_value
    if profile != "official_reproduction":
        raise AnalysisGNNTranspositionError("unknown transposition profile")
    vocabulary = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
    if task_id in ABSOLUTE_TASKS:
        target = _official_spelling_target(value, shift_pc)
    elif task_id in PITCH_CLASS_SET_TASKS:
        target = _pcset_target(value, shift_pc)
    else:
        target = value
    if target is None or target not in vocabulary:
        # This is intentionally not corrected: pinned OutputRepresentation
        # routes an unknown semantic value to its final output class.
        return vocabulary[-1]
    return target


def transpose_record_observations(
    record: RecordTargetObservations,
    *,
    shift_pc: int,
    profile: Literal["corrected_v2", "official_reproduction"] = "corrected_v2",
    variant_source_rows: bool = False,
) -> RecordTargetObservations:
    targets: list[EntityTargetObservation] = []
    for row in record.targets:
        value = row.class_value
        transformed = (
            None
            if value is None
            else transform_semantic_value(
                row.task_id,
                value,
                shift_pc=shift_pc,
                dialect=record.dialect,
                profile=profile,
            )
        )
        source_row = row.source_row_id
        if source_row is not None and variant_source_rows and shift_pc != 0:
            source_row = f"{source_row}@shift_pc={shift_pc:02d}"
        targets.append(
            EntityTargetObservation(
                task_id=row.task_id,
                entity_id=row.entity_id,
                source_row_id=source_row,
                class_value=transformed,
                available=row.available,
                masked=row.masked,
            )
        )
    return RecordTargetObservations(
        record.record_id,
        record.component_id,
        record.dialect,
        record.split,
        tuple(targets),
    )


def valid_shift_for_midi(pitches: Iterable[int], shift_pc: int) -> bool:
    signed = SIGNED_BY_SHIFT_PC[shift_pc]
    return all(0 <= int(pitch) + signed <= 127 for pitch in pitches)


def select_record_shift(
    record_id: str,
    valid_shifts: Sequence[int],
    *,
    seed: int,
    epoch: int,
) -> int:
    ordered = tuple(sorted(set(valid_shifts)))
    if not ordered or any(shift not in SHIFT_PCS for shift in ordered):
        raise AnalysisGNNTranspositionError("valid shifts must be a non-empty subset of 0..11")
    digest = fingerprint(
        {
            "epoch": epoch,
            "profile_id": CORRECTED_PROFILE_ID,
            "record_id": record_id,
            "seed": seed,
        }
    )
    return ordered[int(digest, 16) % len(ordered)]


def _feature_index(names: Sequence[str], name: str) -> int | None:
    try:
        return tuple(names).index(name)
    except ValueError:
        return None


def transpose_raw_graph_view(graph: Any, *, shift_pc: int) -> Any:
    """Return a detached on-the-fly graph view with allowlisted feature edits."""

    if shift_pc not in SHIFT_PCS:
        raise AnalysisGNNTranspositionError("shift_pc must be in 0..11")
    view = copy.deepcopy(graph)
    if shift_pc == 0:
        return view
    note = view["note"]
    pitch_i = _feature_index(note.cat_feature_names, "pitch")
    pc_i = _feature_index(note.cat_feature_names, "pitch_class")
    octave_i = _feature_index(note.cat_feature_names, "octave")
    percussion_i = _feature_index(note.cat_feature_names, "is_percussion")
    if None in (pitch_i, pc_i, octave_i, percussion_i):
        raise AnalysisGNNTranspositionError("raw graph lacks required pitch features")
    pitch_i, pc_i, octave_i, percussion_i = (
        int(pitch_i), int(pc_i), int(octave_i), int(percussion_i)
    )
    non_drum = note.x_cat[:, percussion_i].eq(0)
    shifted = note.x_cat[:, pitch_i].clone()
    shifted[non_drum] += SIGNED_BY_SHIFT_PC[shift_pc]
    if shifted[non_drum].numel() and (
        shifted[non_drum].min().item() < 0 or shifted[non_drum].max().item() > 127
    ):
        raise AnalysisGNNTranspositionError("midi_range_violation")
    note.x_cat[:, pitch_i] = shifted
    note.x_cat[non_drum, pc_i] = shifted[non_drum].remainder(12)
    note.x_cat[non_drum, octave_i] = torch.div(
        shifted[non_drum], 12, rounding_mode="floor"
    )
    relative_i = _feature_index(note.cont_feature_names, "track_relative_pitch")
    relation = ("track", "contains_note", "note")
    if relative_i is not None and relation in view.edge_types:
        edge = view[relation].edge_index
        note.x_cont[:, relative_i] = 0.0
        note.x_cont_available[:, relative_i] = False
        for track_index in edge[0].unique(sorted=True).tolist():
            indices = edge[1, edge[0].eq(track_index)]
            values = shifted[indices].to(torch.float64)
            if values.numel() == 0:
                continue
            mean = values.mean()
            std = torch.sqrt(((values - mean) ** 2).mean())
            if std.item() > 0:
                note.x_cont[indices, relative_i] = ((values - mean) / std).to(
                    note.x_cont.dtype
                )
                note.x_cont_available[indices, relative_i] = True
    return view


def graph_changed_fields(before: Any, after: Any) -> tuple[str, ...]:
    """Return deterministic tensor-field differences for invariant tests."""

    changed: list[str] = []
    for node_type in sorted(before.node_types):
        left = before[node_type]
        right = after[node_type]
        for key in sorted(set(left.keys()) | set(right.keys())):
            a, b = left[key], right[key]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if not torch.equal(a, b):
                    if key in {"x_cat", "x_cont", "x_cont_available"}:
                        names = (
                            left.cat_feature_names
                            if key == "x_cat"
                            else left.cont_feature_names
                        )
                        for index, name in enumerate(names):
                            left_tensor = a[:, index]
                            right_tensor = b[:, index]
                            if not torch.equal(left_tensor, right_tensor):
                                changed.append(f"{node_type}.{name}")
                    else:
                        changed.append(f"{node_type}.{key}")
            elif a != b:
                changed.append(f"{node_type}.{key}")
    for edge_type in sorted(before.edge_types):
        if not torch.equal(before[edge_type].edge_index, after[edge_type].edge_index):
            changed.append("edge:" + "|".join(edge_type))
    return tuple(sorted(set(changed)))


def _time_payload(value: Any) -> tuple[int, int]:
    return int(value.num), int(value.den)


def model_input_collision_fingerprint(
    piece: CanonicalPiece,
    *,
    shift_pc: int,
    official_wrap: bool = False,
) -> str:
    """Hash a target-free, identity-free projection of raw graph inputs.

    This projection is used only for collision detection.  It does not replace
    the canonical raw graph fingerprint and is never a cache identity.
    """

    signed = shift_pc if official_wrap else SIGNED_BY_SHIFT_PC[shift_pc]
    track_index = {track.track_id: index for index, track in enumerate(piece.tracks)}
    notes = []
    for note in piece.notes:
        pitch = note.pitch
        if not note.is_percussion:
            pitch = (pitch + signed) % 128 if official_wrap else pitch + signed
        notes.append(
            [
                track_index[note.track_id],
                pitch,
                _time_payload(note.onset_qn),
                _time_payload(note.duration_qn),
                note.velocity,
                note.channel,
                note.program,
                note.is_percussion,
                note.is_grace,
                note.staff,
                note.voice,
            ]
        )
    payload = {
        "version": "phase9eb5a-target-free-model-input-collision-v1",
        "duration_qn": _time_payload(piece.duration_qn),
        "tracks": [
            [track.program, track.channel, track.is_percussion]
            for track in piece.tracks
        ],
        "notes": notes,
        "bars": [
            [_time_payload(bar.start_qn), _time_payload(bar.duration_qn)]
            for bar in piece.bars
        ],
        "beats": [
            [_time_payload(beat.start_qn), _time_payload(beat.duration_qn), beat.strength]
            for beat in piece.beats
        ],
        "tempo": [
            [_time_payload(row.onset_qn), row.microseconds_per_quarter]
            for row in piece.tempo_events
        ],
        "meter": [
            [_time_payload(row.onset_qn), row.numerator, row.denominator]
            for row in piece.meter_events
        ],
    }
    return fingerprint(payload)


def mapping_composition_summary(
    rows: Sequence[SemanticMappingRow] | None = None,
) -> dict[str, object]:
    source = tuple(rows or semantic_mapping_rows())
    index = {
        (row.source_task_id, row.dialect, row.source_semantic_value, row.shift_pc): row
        for row in source
    }
    promised_checked = 0
    promised_failures: list[dict[str, object]] = []
    diagnostic_checked = 0
    diagnostic_failures = 0
    for row in source:
        if not row.valid or row.target_semantic_value is None:
            continue
        for second in SHIFT_PCS:
            intermediate = index.get(
                (row.source_task_id, row.dialect, row.target_semantic_value, second)
            )
            direct = index.get(
                (
                    row.source_task_id,
                    row.dialect,
                    row.source_semantic_value,
                    (row.shift_pc + second) % 12,
                )
            )
            if not intermediate or not direct or not intermediate.valid or not direct.valid:
                continue
            equal = intermediate.target_semantic_value == direct.target_semantic_value
            if row.source_task_id == "pitch_class_set":
                promised_checked += 1
                if not equal:
                    promised_failures.append(
                        {
                            "dialect": row.dialect,
                            "source": row.source_semantic_value,
                            "first": row.shift_pc,
                            "second": second,
                        }
                    )
            else:
                diagnostic_checked += 1
                diagnostic_failures += int(not equal)
    return {
        "promised_scope": "pitch_class_set_closed_rows",
        "promised_checked": promised_checked,
        "promised_failure_count": len(promised_failures),
        "promised_failures": promised_failures,
        "spelling_diagnostic_checked": diagnostic_checked,
        "spelling_diagnostic_failure_count": diagnostic_failures,
    }


class PostTranspositionAccumulator:
    """Exact full-orbit and analytical one-draw sufficient statistics."""

    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.full_rows: Counter[tuple[str, str]] = Counter()
        self.full_entities: Counter[tuple[str, str]] = Counter()
        self.expected_rows: Counter[tuple[str, str]] = Counter()
        self.expected_entities: Counter[tuple[str, str]] = Counter()
        self.records: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self.components: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        self.component_views: defaultdict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
        self.variant_counts: Counter[str] = Counter()
        self.source_components: defaultdict[str, set[str]] = defaultdict(set)

    def add_record(
        self,
        record: RecordTargetObservations,
        valid_shifts: Sequence[int],
    ) -> None:
        ordered = tuple(sorted(valid_shifts))
        if not ordered or 0 not in ordered:
            raise AnalysisGNNTranspositionError("every record orbit must include identity")
        probability = 1.0 / len(ordered)
        self.variant_counts["record_variants"] += len(ordered)
        entity_counts: Counter[tuple[str, str]] = Counter()
        canonical_rows: set[tuple[str, str, str]] = set()
        for observation in record.targets:
            if not observation.available or observation.masked or observation.class_value is None:
                continue
            self.source_components[observation.task_id].add(record.component_id)
            source_key = (observation.task_id, observation.class_value)
            entity_counts[source_key] += 1
            canonical_rows.add(
                (
                    observation.task_id,
                    observation.source_row_id or observation.entity_id,
                    observation.class_value,
                )
            )
        row_counts = Counter((task_id, value) for task_id, _row, value in canonical_rows)
        for shift_pc in ordered:
            transformed_values = {
                source_key: transform_semantic_value(
                    source_key[0],
                    source_key[1],
                    shift_pc=shift_pc,
                    dialect=record.dialect,
                    profile=self.profile,  # type: ignore[arg-type]
                )
                for source_key in entity_counts
            }
            for source_key, count in entity_counts.items():
                key = (source_key[0], transformed_values[source_key])
                self.full_entities[key] += count
                self.expected_entities[key] += count * probability
                self.records[key].add(record.record_id)
                self.components[key].add(record.component_id)
                self.component_views[key].add((record.component_id, shift_pc))
            for source_key, count in row_counts.items():
                key = (source_key[0], transformed_values[source_key])
                self.full_rows[key] += count
                self.expected_rows[key] += count * probability

    def summarize(
        self,
        raw_heads: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        output: list[dict[str, object]] = []
        for task in PRODUCTION_TASKS:
            vocabulary = get_vocabulary(task.vocabulary_id).labels
            full = [self.full_rows[(task.task_id, value)] for value in vocabulary]
            expected = [self.expected_rows[(task.task_id, value)] for value in vocabulary]
            full_nonzero = [value for value in full if value > 0]
            expected_nonzero = [value for value in expected if value > 0]
            full_total = sum(full)
            expected_total = sum(expected)
            full_tiers = [
                train_support_tier(
                    count,
                    len(self.components[(task.task_id, value)]),
                )
                for value, count in zip(vocabulary, full, strict=True)
            ]
            expected_tiers = [
                train_support_tier(
                    count,
                    len(self.components[(task.task_id, value)]),
                )
                for value, count in zip(vocabulary, expected, strict=True)
            ]
            raw = raw_heads[task.task_id]
            validation_unobservable = set(
                raw.get(
                    "validation_unobservable_classes",
                    raw.get("problem_classes", {}).get(
                        "validation_unobservable_classes", []
                    ),
                )
            )
            validation_fragile = set(
                raw.get(
                    "validation_fragile_classes",
                    raw.get("problem_classes", {}).get(
                        "validation_fragile_classes", []
                    ),
                )
            )
            validation_tiers = tuple(
                "unobservable"
                if value in validation_unobservable
                else "fragile_validation"
                if value in validation_fragile
                else "observable"
                for value in vocabulary
            )
            def stats(counts: Sequence[float], nonzero: Sequence[float], total: float) -> dict[str, object]:
                entropy = (
                    -sum((count / total) * math.log(count / total) for count in nonzero)
                    if total else 0.0
                )
                return {
                    "observed_class_count": len(nonzero),
                    "absent_classes": [
                        value for value, count in zip(vocabulary, counts, strict=True) if count == 0
                    ],
                    "insufficient_classes": [
                        value
                        for value, tier in zip(
                            vocabulary,
                            full_tiers if counts is full else expected_tiers,
                            strict=True,
                        )
                        if tier == "insufficient"
                    ],
                    "majority_share": round(max(nonzero) / total, 12) if nonzero else 0.0,
                    "max_to_min_nonzero_ratio": (
                        round(max(nonzero) / min(nonzero), 12) if nonzero else None
                    ),
                    "normalized_entropy": (
                        round(entropy / math.log(len(vocabulary)), 12)
                        if total and len(vocabulary) > 1 else 0.0
                    ),
                    "canonical_target_row_count": round(total, 12),
                }
            full_stats = stats(full, full_nonzero, full_total)
            expected_stats = stats(expected, expected_nonzero, expected_total)
            full_stats["variant_entity_count"] = self.full_entity_count(
                task.task_id
            )
            full_stats["variant_canonical_row_count"] = full_stats[
                "canonical_target_row_count"
            ]
            expected_stats["expected_entity_count"] = self.expected_entity_count(
                task.task_id
            )
            expected_stats["expected_canonical_row_count"] = expected_stats[
                "canonical_target_row_count"
            ]
            source_components = len(self.source_components[task.task_id])
            full_status, _ = recommend_head_trainability(
                vocabulary_size=len(vocabulary),
                train_tiers=full_tiers,
                validation_tiers=validation_tiers,
                available_train_components=source_components,
                majority_share=float(full_stats["majority_share"]),
                max_to_min_nonzero_ratio=full_stats["max_to_min_nonzero_ratio"],  # type: ignore[arg-type]
                normalized_entropy=float(full_stats["normalized_entropy"]),
            )
            expected_status, _ = recommend_head_trainability(
                vocabulary_size=len(vocabulary),
                train_tiers=expected_tiers,
                validation_tiers=validation_tiers,
                available_train_components=source_components,
                majority_share=float(expected_stats["majority_share"]),
                max_to_min_nonzero_ratio=expected_stats["max_to_min_nonzero_ratio"],  # type: ignore[arg-type]
                normalized_entropy=float(expected_stats["normalized_entropy"]),
            )
            raw_observed = int(raw["train_observed_class_count"])
            observed = int(full_stats["observed_class_count"])
            kind = next(row.transformation_kind for row in transformation_registry() if row.task_id == task.task_id)
            if kind in {"relative_label_invariant", "structural_label_invariant", "boolean_label_invariant"}:
                effect = "unchanged_invariant"
            elif observed > raw_observed and not full_stats["absent_classes"]:
                effect = "coverage_recovered"
            elif observed > raw_observed:
                effect = "partially_improved"
            elif float(full_stats["normalized_entropy"]) > float(raw["normalized_entropy"]):
                effect = "balance_improved"
            else:
                effect = "not_sufficient"
            output.append(
                {
                    "task_id": task.task_id,
                    "vocabulary_size": len(vocabulary),
                    "b4_raw_status": raw["recommendation"],
                    "raw": {
                        "observed_class_count": raw_observed,
                        "absent_classes": list(
                            raw.get(
                                "absent_classes",
                                raw.get("problem_classes", {}).get(
                                    "absent_classes", []
                                ),
                            )
                        ),
                        "majority_share": raw["majority_share"],
                        "max_to_min_nonzero_ratio": raw["max_to_min_nonzero_ratio"],
                        "normalized_entropy": raw["normalized_entropy"],
                    },
                    "full_orbit": full_stats,
                    "expected_epoch": expected_stats,
                    "full_orbit_status": full_status,
                    "expected_epoch_status": expected_status,
                    "augmentation_effect": effect,
                    "source_component_count": source_components,
                    "transformed_component_support": len(
                        set().union(
                            *(self.components[(task.task_id, value)] for value in vocabulary)
                        )
                    ),
                    "component_shift_support": len(
                        set().union(
                            *(self.component_views[(task.task_id, value)] for value in vocabulary)
                        )
                    ),
                    "variant_count": self.variant_counts["record_variants"],
                    "sampling_interpretation": (
                        "exact_uniform_one_draw_per_record_expectation"
                        if self.profile == "corrected_v2"
                        else "normalized_equal_requested_view_diagnostic_not_actual_loader_probability"
                    ),
                    "classes_created_only_by_augmentation": [
                        value
                        for value, count in zip(vocabulary, full, strict=True)
                        if count > 0
                        and value
                        in raw.get(
                            "absent_classes",
                            raw.get("problem_classes", {}).get(
                                "absent_classes", []
                            ),
                        )
                    ],
                    "classes_remaining_unsupported": list(full_stats["absent_classes"]),
                }
            )
        return tuple(output)

    def full_entity_count(self, task_id: str) -> int:
        return sum(
            count
            for (observed_task, _value), count in self.full_entities.items()
            if observed_task == task_id
        )

    def expected_entity_count(self, task_id: str) -> float:
        return round(
            sum(
                count
                for (observed_task, _value), count in self.expected_entities.items()
                if observed_task == task_id
            ),
            12,
        )


def role_recommendations(
    corrected_rows: Sequence[Mapping[str, object]],
    official_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    corrected_by_task = {str(row["task_id"]): row for row in corrected_rows}
    official_by_task = {str(row["task_id"]): row for row in official_rows}
    official: list[dict[str, object]] = []
    corrected: list[dict[str, object]] = []
    for task in PRODUCTION_TASKS:
        task_id = task.task_id
        official_status = str(official_by_task[task_id]["expected_epoch_status"])
        if task_id in ABSOLUTE_TASKS and official_by_task[task_id]["classes_remaining_unsupported"]:
            official_role = "official_semantic_ambiguity"
        elif official_status == "descriptive_only":
            official_role = "official_unobservable"
        elif official_status in {"insufficient_support", "trainable_with_reweighting"}:
            official_role = "official_sparse"
        else:
            official_role = "official_trainable_as_pinned"
        official.append({"task_id": task_id, "recommendation": official_role})
        row = corrected_by_task[task_id]
        status = str(row["expected_epoch_status"])
        if status == "trainable":
            role = "primary_candidate"
        elif status == "trainable_with_reweighting":
            role = "auxiliary_candidate"
        elif row["augmentation_effect"] == "unchanged_invariant" and row["b4_raw_status"] == "descriptive_only":
            role = "derived_metric_candidate"
        elif status == "insufficient_support" or row["b4_raw_status"] == "insufficient_support":
            role = "requires_policy_decision"
        else:
            role = "deferred_candidate"
        corrected.append({"task_id": task_id, "candidate_role": role})
    payload: dict[str, object] = {
        "official_reproduction": official,
        "corrected_v2": corrected,
        "roles_frozen": False,
        "losses_changed": False,
        "model_changed": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def mapping_summary(rows: Sequence[SemanticMappingRow] | None = None) -> dict[str, object]:
    values = tuple(rows or semantic_mapping_rows())
    reasons = Counter(row.invalid_reason for row in values if not row.valid)
    by_task: dict[str, dict[str, object]] = {}
    for task_id in (*ABSOLUTE_TASKS, *PITCH_CLASS_SET_TASKS):
        selected = [row for row in values if row.source_task_id == task_id]
        by_task[task_id] = {
            "row_count": len(selected),
            "valid_count": sum(row.valid for row in selected),
            "invalid_count": sum(not row.valid for row in selected),
            "round_trip_failure_count": sum(
                row.valid and not row.round_trip_valid for row in selected
            ),
        }
    return {
        "rows": len(values),
        "valid": sum(row.valid for row in values),
        "invalid": sum(not row.valid for row in values),
        "invalid_reason_counts": {
            str(key): value for key, value in sorted(reasons.items()) if key is not None
        },
        "by_task": by_task,
        "composition": mapping_composition_summary(values),
    }


__all__ = [
    "ABSOLUTE_TASKS",
    "AugmentedGraphIdentity",
    "CORRECTED_PROFILE_ID",
    "INVARIANT_GRAPH_FIELDS",
    "MAPPING_INVALID_REASONS",
    "OFFICIAL_INTERVALS",
    "OFFICIAL_PROFILE_ID",
    "PITCH_DEPENDENT_GRAPH_FEATURES",
    "PostTranspositionAccumulator",
    "RECOMPUTED_GRAPH_FEATURES",
    "RecordShiftEligibility",
    "SEMANTIC_MAPPING_ROWS",
    "SHIFT_PCS",
    "SIGNED_BY_SHIFT_PC",
    "SIGNED_SEMITONES",
    "SemanticMappingRow",
    "TRANSPOSITION_AUDIT_SCHEMA",
    "TRANSPOSITION_CONTRACT_VERSION",
    "TransformationSpec",
    "AnalysisGNNTranspositionError",
    "corrected_transposition_profile",
    "graph_changed_fields",
    "mapping_composition_summary",
    "mapping_summary",
    "model_input_collision_fingerprint",
    "official_transposition_evidence",
    "role_recommendations",
    "select_record_shift",
    "semantic_mapping_index",
    "semantic_mapping_rows",
    "transformation_registry",
    "transform_semantic_value",
    "transpose_raw_graph_view",
    "transpose_record_observations",
    "transposition_contract",
    "valid_shift_for_midi",
]


# Public immutable alias for callers that need the complete deterministic table.
SEMANTIC_MAPPING_ROWS = semantic_mapping_rows()
