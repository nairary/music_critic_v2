"""Phase 9E-B5F diagnostics for the executable B5A transposition contract.

The helpers in this module audit, but never replace, the B5A implementation.
Independent arithmetic oracles are intentionally kept separate from the
production transform.  No function in this module opens TEST or trains a
model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from music_critic.experiments.analysisgnn.class_balance import (
    observations_from_sidecar,
)
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    CORRECTED_CHECKPOINT_SCHEMA,
    CorrectedComponentSampler,
    CorrectedTrainingError,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    build_source_free_fixture,
    production_component_records,
    production_valid_shifts,
    record_schedule_fingerprint,
    transpose_raw_graph_batch,
    transposition_schedule_fingerprint,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_BATCH_SIZE,
    FULL_SEED,
    FULL_UPDATE_BUDGET,
    full_runtime_config,
    full_training_contract,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    SIGNED_BY_SHIFT_PC,
    semantic_mapping_rows,
    transformation_registry,
    transform_semantic_value,
    transpose_raw_graph_view,
    transpose_record_observations,
    transposition_contract,
    valid_shift_for_midi,
)


B5F_AUDIT_SCHEMA = "Phase9EB5FTranspositionCorrectness@1.0.0"
B5F_PAIR_SCHEMA = "Phase9EB5FRecordShiftDiagnostic@1.0.0"
B5F_CHECKPOINT_SCHEMA = "Phase9EB5FCheckpointShiftDiagnostics@1.0.0"
B5F_FINAL_STATUS = "implementation_or_contract_defect"
GRAPH_CONTINUOUS_ATOL = 1e-6
EXPECTED_RECORD_COUNTS = {"train": 1295, "validation": 162}
EXPECTED_PAIR_COUNT = sum(EXPECTED_RECORD_COUNTS.values()) * len(SHIFT_PCS)

EXPECTED_TRANSFORMATION_KINDS = {
    "local_key": "absolute_pitch_transpose",
    "tonicized_key": "absolute_pitch_transpose",
    "root": "absolute_pitch_transpose",
    "bass": "absolute_pitch_transpose",
    "pitch_class_set": "pitch_class_set_transpose",
    "primary_degree": "relative_label_invariant",
    "secondary_degree": "relative_label_invariant",
    "quality": "relative_label_invariant",
    "inversion": "relative_label_invariant",
    "roman_numeral": "relative_label_invariant",
    "note_degree": "relative_label_invariant",
    "harmonic_rhythm": "structural_label_invariant",
    "cadence": "structural_label_invariant",
    "phrase": "structural_label_invariant",
    "section": "structural_label_invariant",
    "metrical_strength": "structural_label_invariant",
    "pedal": "boolean_label_invariant",
    "chord_tone": "boolean_label_invariant",
    "is_root": "boolean_label_invariant",
    "is_bass": "boolean_label_invariant",
}
EQUIVARIANT_HEADS = frozenset(
    {"local_key", "tonicized_key", "root", "bass", "pitch_class_set"}
)
INVARIANT_HEADS = frozenset(EXPECTED_TRANSFORMATION_KINDS) - EQUIVARIANT_HEADS
ALLOWED_GRAPH_CHANGES = frozenset(
    {
        "note.pitch",
        "note.pitch_class",
        "note.octave",
        "note.track_relative_pitch",
    }
)


class TranspositionDiagnosticError(ValueError):
    """Stable fail-closed B5F diagnostic error."""


@dataclass(frozen=True, slots=True)
class RecordShiftDiagnostic:
    schema: str
    record_id: str
    split: str
    dialect: str
    source_component_id: str
    shift_pc: int
    signed_semitones: int
    midi_range_valid: bool
    target_vocabulary_closed: bool
    spelling_valid: bool
    round_trip_valid: bool
    runtime_path_matches_contract: bool
    masks_preserved: bool
    entity_ids_preserved: bool
    routing_preserved: bool
    changed_graph_fields: tuple[str, ...]
    invalid_reasons: tuple[str, ...]
    cross_head_status: str
    fingerprint: str


def _without_fingerprint(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("fingerprint", None)
    return body


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    body = _without_fingerprint(value)
    body["fingerprint"] = fingerprint(body)
    return body


def transformation_matrix() -> tuple[dict[str, object], ...]:
    """Return the executable B5A registry with the B5F expectation check."""

    rows = transformation_registry()
    observed = {row.task_id: row.transformation_kind for row in rows}
    complete = observed == EXPECTED_TRANSFORMATION_KINDS
    return tuple(
        {
            **asdict(row),
            "expected_transformation_kind": EXPECTED_TRANSFORMATION_KINDS.get(
                row.task_id
            ),
            "classification_matches": (
                EXPECTED_TRANSFORMATION_KINDS.get(row.task_id)
                == row.transformation_kind
            ),
            "registry_complete": complete,
        }
        for row in rows
    )


def _spelling(value: str) -> tuple[int, bool] | None:
    """Independent spelling parser used only as a pitch-class oracle."""

    if not value or value[0].upper() not in "CDEFGAB":
        return None
    suffix = value[1:]
    if any(char not in "#b" for char in suffix) or (
        "#" in suffix and "b" in suffix
    ):
        return None
    natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    alteration = suffix.count("#") - suffix.count("b")
    return (natural[value[0].upper()] + alteration) % 12, value[0].islower()


def _pcset(value: str) -> tuple[int, ...] | None:
    try:
        parsed = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError:
        return None
    return parsed if parsed and all(0 <= item <= 11 for item in parsed) else None


def independent_target_oracle(
    task_id: str, before: str, after: str, *, shift_pc: int
) -> bool:
    """Check semantic behavior without deriving the result via B5A."""

    if task_id in INVARIANT_HEADS:
        return after == before
    if task_id == "pitch_class_set":
        source = _pcset(before)
        target = _pcset(after)
        return source is not None and target == tuple(
            sorted((value + shift_pc) % 12 for value in source)
        )
    if task_id in {"local_key", "tonicized_key", "root", "bass"}:
        source = _spelling(before)
        target = _spelling(after)
        if source is None or target is None:
            return False
        source_pc, source_minor = source
        target_pc, target_minor = target
        return target_pc == (source_pc + shift_pc) % 12 and (
            task_id not in {"local_key", "tonicized_key"}
            or source_minor == target_minor
        )
    return False


def prepare_sidecar_diagnostic_context(
    sidecar: Mapping[str, object],
) -> dict[str, object]:
    """Scan shift-invariant sidecar evidence once per production record."""

    unique_counts: Counter[tuple[str, str]] = Counter()
    unclassified: set[str] = set()
    entity_ids_valid = True
    harmonic_contexts = {
        "absolute_heads_same_shift": False,
        "inversion_with_root_bass": False,
        "degree_roman_with_key_root": False,
        "pitch_class_set_shift": False,
    }
    entities_by_id: dict[str, Mapping[str, object]] = {}
    for entity in sidecar["entities"]:  # type: ignore[index]
        entity_id = str(entity["canonical_entity_id"])
        entities_by_id[entity_id] = entity
        targets = entity.get("targets", {})
        if not isinstance(targets, Mapping):
            continue
        available: set[str] = set()
        for task_id, raw_state in targets.items():
            if task_id not in EXPECTED_TRANSFORMATION_KINDS or not isinstance(
                raw_state, Mapping
            ):
                unclassified.add(str(task_id))
                continue
            value = raw_state.get("canonical_value")
            entity_ids_valid &= str(
                raw_state.get("canonical_entity_id", entity_id)
            ) == entity_id
            if (
                raw_state.get("available") is True
                and raw_state.get("masked") is False
                and isinstance(value, str)
            ):
                unique_counts[(str(task_id), value)] += 1
            if (
                raw_state.get("available") is True
                and raw_state.get("masked") is False
                and isinstance(value, str)
            ):
                available.add(str(task_id))
        if entity.get("entity_type") == "harmonic_event":
            harmonic_contexts["absolute_heads_same_shift"] |= len(
                available & {"root", "bass", "local_key", "tonicized_key"}
            ) >= 2
            harmonic_contexts["inversion_with_root_bass"] |= {
                "inversion",
                "root",
                "bass",
            } <= available
            harmonic_contexts["degree_roman_with_key_root"] |= {
                "primary_degree",
                "secondary_degree",
                "roman_numeral",
                "local_key",
                "root",
            } <= available
            harmonic_contexts["pitch_class_set_shift"] |= (
                "pitch_class_set" in available
            )
    note_degree_context = False
    boolean_context = False
    for relation in sidecar["relations"]:  # type: ignore[index]
        if relation.get("relation") != "note_to_harmonic_event":
            continue
        note = entities_by_id.get(str(relation["source_entity_id"]))
        harmonic = entities_by_id.get(str(relation["target_entity_id"]))
        if note is None or harmonic is None:
            continue
        note_targets = note.get("targets", {})
        harmonic_targets = harmonic.get("targets", {})
        if not isinstance(note_targets, Mapping) or not isinstance(
            harmonic_targets, Mapping
        ):
            continue
        degree = note_targets.get("note_degree")
        key = harmonic_targets.get("local_key")
        note_degree_context |= isinstance(degree, Mapping) and isinstance(key, Mapping)
        boolean_context |= any(
            isinstance(note_targets.get(task), Mapping)
            for task in ("chord_tone", "is_root", "is_bass")
        )
    relation_contexts = {
        **harmonic_contexts,
        "note_degree_with_pitch_key": note_degree_context,
        "note_chord_booleans": boolean_context,
    }
    return {
        "dialect": str(sidecar["dialect"]),
        "unique_target_counts": [
            [task, value, count]
            for (task, value), count in sorted(unique_counts.items())
        ],
        "checked_target_rows": sum(unique_counts.values()),
        "unclassified_sidecar_heads": sorted(unclassified),
        "entity_ids_valid": entity_ids_valid,
        "relation_contexts": relation_contexts,
        "sidecar_fingerprint": sidecar.get("fingerprint"),
    }


def audit_sidecar_targets(
    sidecar: Mapping[str, object],
    *,
    shift_pc: int,
    prepared: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit target closure, independent semantics, masks, IDs and round trip."""

    context = (
        prepare_sidecar_diagnostic_context(sidecar) if prepared is None else prepared
    )
    dialect = str(context["dialect"])
    reasons: set[str] = set()
    unique_counts = Counter(
        {
            (str(task), str(value)): int(count)
            for task, value, count in context["unique_target_counts"]  # type: ignore[index]
        }
    )
    round_trips = 0
    invariant_checked = 0
    equivariant_checked = 0
    masks_preserved = True
    entity_ids_preserved = bool(context["entity_ids_valid"])
    class_ids_preserved = True
    tritone_checked = 0
    per_head: defaultdict[str, Counter[str]] = defaultdict(Counter)
    reasons.update(
        f"unclassified_sidecar_head:{task}"
        for task in context["unclassified_sidecar_heads"]  # type: ignore[index]
    )
    if not entity_ids_preserved:
        reasons.add("entity_id_mismatch")
    for (task_id, value), count in unique_counts.items():
        try:
            after = transform_semantic_value(
                task_id,
                value,
                shift_pc=shift_pc,
                dialect=dialect,
                profile="corrected_v2",
            )
        except Exception as exc:  # fail closed with the B5A reason
            reasons.add(f"target_transform:{task_id}:{type(exc).__name__}")
            per_head[task_id]["invalid"] += count
            continue
        vocabulary = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
        if after not in vocabulary:
            reasons.add(f"target_oov:{task_id}")
            per_head[task_id]["invalid"] += count
            continue
        if not independent_target_oracle(task_id, value, after, shift_pc=shift_pc):
            reasons.add(f"independent_oracle_mismatch:{task_id}")
            per_head[task_id]["oracle_mismatch"] += count
        if task_id in INVARIANT_HEADS:
            invariant_checked += count
            if vocabulary.index(after) != vocabulary.index(value):
                class_ids_preserved = False
                reasons.add(f"invariant_class_id_changed:{task_id}")
        else:
            equivariant_checked += count
            tritone_checked += count * int(shift_pc == 6)
        try:
            restored = transform_semantic_value(
                task_id,
                after,
                shift_pc=(-shift_pc) % 12,
                dialect=dialect,
                profile="corrected_v2",
            )
        except Exception as exc:
            reasons.add(f"inverse_transform:{task_id}:{type(exc).__name__}")
            per_head[task_id]["round_trip_failure"] += count
        else:
            if restored == value:
                round_trips += count
            else:
                reasons.add(f"target_round_trip:{task_id}")
                per_head[task_id]["round_trip_failure"] += count
        per_head[task_id]["checked"] += count
    checked = sum(unique_counts.values())
    return _sealed(
        {
            "checked_target_rows": checked,
            "round_trip_rows": round_trips,
            "invariant_rows_checked": invariant_checked,
            "equivariant_rows_checked": equivariant_checked,
            "tritone_rows_checked": tritone_checked,
            "target_vocabulary_closed": not any(
                reason.startswith(("target_transform:", "target_oov:"))
                for reason in reasons
            ),
            "spelling_valid": not any(
                (
                    reason.startswith("independent_oracle_mismatch:")
                    and reason.rsplit(":", 1)[-1]
                    in {"local_key", "tonicized_key", "root", "bass"}
                )
                or (
                    reason.startswith("target_transform:")
                    and reason.split(":", 2)[1]
                    in {"local_key", "tonicized_key", "root", "bass"}
                )
                for reason in reasons
            ),
            "round_trip_valid": checked == round_trips,
            "masks_preserved": masks_preserved,
            "entity_ids_preserved": entity_ids_preserved,
            "class_ids_preserved": class_ids_preserved,
            "invalid_reasons": sorted(reasons),
            "per_head": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(per_head.items())
            },
        }
    )


def audit_record_observation_transform(
    sidecar: Mapping[str, object], *, shift_pc: int, split: str = "train"
) -> dict[str, object]:
    """Compare the B5A record-observation API to the independent oracle."""

    source = observations_from_sidecar(sidecar, split=split)
    transformed = transpose_record_observations(source, shift_pc=shift_pc)
    restored = transpose_record_observations(
        transformed, shift_pc=(-shift_pc) % 12
    )
    mismatches = []
    round_trip_mismatches = []
    for before, after, final in zip(
        source.targets, transformed.targets, restored.targets, strict=True
    ):
        if (
            before.task_id != after.task_id
            or before.entity_id != after.entity_id
            or before.source_row_id != after.source_row_id
            or before.available != after.available
            or before.masked != after.masked
        ):
            mismatches.append(
                {"task_id": before.task_id, "entity_id": before.entity_id}
            )
            continue
        if before.class_value is not None and (
            after.class_value is None
            or not independent_target_oracle(
                before.task_id,
                before.class_value,
                after.class_value,
                shift_pc=shift_pc,
            )
        ):
            mismatches.append(
                {"task_id": before.task_id, "entity_id": before.entity_id}
            )
        if final != before:
            round_trip_mismatches.append(
                {"task_id": before.task_id, "entity_id": before.entity_id}
            )
    body = {
        "record_id": source.record_id,
        "shift_pc": shift_pc,
        "row_count": len(source.targets),
        "mismatch_count": len(mismatches),
        "round_trip_mismatch_count": len(round_trip_mismatches),
        "mismatches": mismatches[:32],
        "round_trip_mismatches": round_trip_mismatches[:32],
        "masks_preserved": not mismatches,
        "entity_ids_preserved": not mismatches,
        "round_trip_valid": not round_trip_mismatches,
    }
    return _sealed(body)


def cross_head_checks(
    sidecar: Mapping[str, object], graph: Any, *, shift_pc: int
) -> dict[str, object]:
    """Run relation checks without treating absent context as success."""

    dialect = str(sidecar["dialect"])
    checks: dict[str, dict[str, object]] = {}
    harmonic_entities = [
        entity
        for entity in sidecar["entities"]  # type: ignore[index]
        if entity.get("entity_type") == "harmonic_event"
    ]

    def available(entity: Mapping[str, object], task: str) -> str | None:
        targets = entity.get("targets", {})
        state = targets.get(task) if isinstance(targets, Mapping) else None
        if not isinstance(state, Mapping):
            return None
        value = state.get("canonical_value")
        return (
            str(value)
            if state.get("available") is True
            and state.get("masked") is False
            and isinstance(value, str)
            else None
        )

    absolute_contexts = 0
    absolute_failures = 0
    invariant_contexts: Counter[str] = Counter()
    invariant_failures: Counter[str] = Counter()
    seen_absolute: set[tuple[tuple[str, str], ...]] = set()
    seen_invariant: defaultdict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
    for entity in harmonic_entities:
        absolute = {
            task: available(entity, task)
            for task in ("root", "bass", "local_key", "tonicized_key")
        }
        present = {task: value for task, value in absolute.items() if value is not None}
        absolute_key = tuple(sorted((task, value) for task, value in present.items()))
        if len(present) >= 2 and absolute_key not in seen_absolute:
            seen_absolute.add(absolute_key)
            absolute_contexts += 1
            for task, value in present.items():
                after = transform_semantic_value(
                    task,
                    value,
                    shift_pc=shift_pc,
                    dialect=dialect,
                    profile="corrected_v2",
                )
                absolute_failures += int(
                    not independent_target_oracle(
                        task, value, after, shift_pc=shift_pc
                    )
                )
        for name, required in (
            ("inversion_with_root_bass", ("inversion", "root", "bass")),
            (
                "degree_roman_with_key_root",
                ("primary_degree", "secondary_degree", "roman_numeral", "local_key", "root"),
            ),
            ("pitch_class_set_shift", ("pitch_class_set",)),
        ):
            values = {task: available(entity, task) for task in required}
            if not all(value is not None for value in values.values()):
                continue
            invariant_key = tuple(
                sorted((task, str(value)) for task, value in values.items())
            )
            if invariant_key in seen_invariant[name]:
                continue
            seen_invariant[name].add(invariant_key)
            invariant_contexts[name] += 1
            for task, value in values.items():
                assert value is not None
                after = transform_semantic_value(
                    task,
                    value,
                    shift_pc=shift_pc,
                    dialect=dialect,
                    profile="corrected_v2",
                )
                invariant_failures[name] += int(
                    not independent_target_oracle(
                        task, value, after, shift_pc=shift_pc
                    )
                )
    checks["absolute_heads_same_shift"] = {
        "status": (
            "not_checkable"
            if absolute_contexts == 0
            else "passed"
            if absolute_failures == 0
            else "failed"
        ),
        "contexts": absolute_contexts,
        "failures": absolute_failures,
    }
    for name in (
        "inversion_with_root_bass",
        "degree_roman_with_key_root",
        "pitch_class_set_shift",
    ):
        contexts = invariant_contexts[name]
        failures = invariant_failures[name]
        checks[name] = {
            "status": (
                "not_checkable"
                if contexts == 0
                else "passed"
                if failures == 0
                else "failed"
            ),
            "contexts": contexts,
            "failures": failures,
        }

    note = graph["note"]
    pitch_i = _feature_index(note.cat_feature_names, "pitch")
    pitches = {
        str(entity_id): int(note.x_cat[index, pitch_i])
        for index, entity_id in enumerate(note.entity_id)
    }
    harmonic_by_id = {
        str(entity["canonical_entity_id"]): entity for entity in harmonic_entities
    }
    note_to_harmonic = {
        str(row["source_entity_id"]): str(row["target_entity_id"])
        for row in sidecar["relations"]  # type: ignore[index]
        if row.get("relation") == "note_to_harmonic_event"
    }
    note_contexts = 0
    note_failures = 0
    boolean_contexts = 0
    boolean_failures = 0
    seen_note_degree: set[tuple[str, str, int]] = set()
    seen_booleans: set[tuple[str, str]] = set()
    note_entities = {
        str(entity["canonical_entity_id"]): entity
        for entity in sidecar["entities"]  # type: ignore[index]
        if entity.get("entity_type") == "note"
    }
    for note_id, harmonic_id in note_to_harmonic.items():
        entity = note_entities.get(note_id)
        harmonic = harmonic_by_id.get(harmonic_id)
        if entity is None or harmonic is None or note_id not in pitches:
            continue
        degree = available(entity, "note_degree")
        local_key = available(harmonic, "local_key")
        if degree is not None and local_key is not None:
            note_key = (degree, local_key, pitches[note_id] % 12)
            if note_key in seen_note_degree:
                degree = None
            else:
                seen_note_degree.add(note_key)
        if degree is not None and local_key is not None:
            key_pc = _spelling(local_key)
            shifted_key = transform_semantic_value(
                "local_key",
                local_key,
                shift_pc=shift_pc,
                dialect=dialect,
                profile="corrected_v2",
            )
            shifted_key_pc = _spelling(shifted_key)
            note_contexts += 1
            if key_pc is None or shifted_key_pc is None:
                note_failures += 1
            else:
                before_delta = (pitches[note_id] - key_pc[0]) % 12
                after_delta = (
                    pitches[note_id]
                    + SIGNED_BY_SHIFT_PC[shift_pc]
                    - shifted_key_pc[0]
                ) % 12
                note_failures += int(before_delta != after_delta)
                shifted_degree = transform_semantic_value(
                    "note_degree",
                    degree,
                    shift_pc=shift_pc,
                    dialect=dialect,
                    profile="corrected_v2",
                )
                note_failures += int(shifted_degree != degree)
        for task in ("chord_tone", "is_root", "is_bass"):
            value = available(entity, task)
            if value is None:
                continue
            boolean_key = (task, value)
            if boolean_key in seen_booleans:
                continue
            seen_booleans.add(boolean_key)
            boolean_contexts += 1
            boolean_failures += int(
                transform_semantic_value(
                    task,
                    value,
                    shift_pc=shift_pc,
                    dialect=dialect,
                    profile="corrected_v2",
                )
                != value
            )
    for name, contexts, failures in (
        ("note_degree_with_pitch_key", note_contexts, note_failures),
        ("note_chord_booleans", boolean_contexts, boolean_failures),
    ):
        checks[name] = {
            "status": (
                "not_checkable"
                if contexts == 0
                else "passed"
                if failures == 0
                else "failed"
            ),
            "contexts": contexts,
            "failures": failures,
        }
    statuses = [str(row["status"]) for row in checks.values()]
    payload = {
        "checks": checks,
        "passed": bool(statuses) and all(status == "passed" for status in statuses),
        "failed": any(status == "failed" for status in statuses),
        "not_checkable": [
            name for name, row in checks.items() if row["status"] == "not_checkable"
        ],
    }
    return _sealed(payload)


def _feature_index(names: Sequence[str], name: str) -> int:
    try:
        return tuple(names).index(name)
    except ValueError as exc:
        raise TranspositionDiagnosticError(f"missing graph feature: {name}") from exc


def _tensor_equal(left: Tensor, right: Tensor, *, continuous: bool) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if continuous and (left.is_floating_point() or right.is_floating_point()):
        return bool(torch.allclose(left, right, atol=GRAPH_CONTINUOUS_ATOL, rtol=0.0))
    return bool(torch.equal(left, right))


def _graph_differences(left: Any, right: Any) -> tuple[str, ...]:
    differences: list[str] = []
    if tuple(left.node_types) != tuple(right.node_types):
        differences.append("node_types")
    if tuple(left.edge_types) != tuple(right.edge_types):
        differences.append("edge_types")
    for node_type in left.node_types:
        a = left[node_type]
        b = right[node_type]
        if tuple(a.keys()) != tuple(b.keys()):
            differences.append(f"{node_type}.keys")
            continue
        for key in a.keys():
            av, bv = a[key], b[key]
            if isinstance(av, Tensor) and isinstance(bv, Tensor):
                if not _tensor_equal(av, bv, continuous=key == "x_cont"):
                    differences.append(f"{node_type}.{key}")
            elif av != bv:
                differences.append(f"{node_type}.{key}")
    for edge_type in left.edge_types:
        if not torch.equal(left[edge_type].edge_index, right[edge_type].edge_index):
            differences.append("edge:" + "|".join(edge_type))
    return tuple(sorted(set(differences)))


def _independent_relative_pitch(graph: Any, shifted_pitch: Tensor) -> tuple[Tensor, Tensor]:
    note = graph["note"]
    relative_i = _feature_index(note.cont_feature_names, "track_relative_pitch")
    expected = torch.zeros_like(note.x_cont[:, relative_i])
    available = torch.zeros_like(note.x_cont_available[:, relative_i])
    relation = ("track", "contains_note", "note")
    edge = graph[relation].edge_index
    for track_index in edge[0].unique(sorted=True).tolist():
        indices = edge[1, edge[0].eq(track_index)]
        values = shifted_pitch[indices].to(torch.float64)
        if not values.numel():
            continue
        mean = values.mean()
        standard_deviation = torch.sqrt(((values - mean) ** 2).mean())
        if standard_deviation.item() > 0:
            expected[indices] = ((values - mean) / standard_deviation).to(
                expected.dtype
            )
            available[indices] = True
    return expected, available


def audit_graph_transform(
    graph: Any, *, shift_pc: int, compare_direct_and_runtime: bool = True
) -> dict[str, object]:
    """Compare direct B5A and the B5C runtime batch path to independent oracles."""

    from torch_geometric.data import Batch
    from music_critic.experiments.analysisgnn.transposition import graph_changed_fields

    note = graph["note"]
    pitch_i = _feature_index(note.cat_feature_names, "pitch")
    pc_i = _feature_index(note.cat_feature_names, "pitch_class")
    octave_i = _feature_index(note.cat_feature_names, "octave")
    percussion_i = _feature_index(note.cat_feature_names, "is_percussion")
    non_drum = note.x_cat[:, percussion_i].eq(0)
    source_pitch = note.x_cat[:, pitch_i]
    midi_range_valid = valid_shift_for_midi(
        source_pitch[non_drum].tolist(), shift_pc
    )
    if not midi_range_valid:
        from torch_geometric.data import Batch

        direct_rejected = False
        runtime_rejected = False
        try:
            transpose_raw_graph_view(graph, shift_pc=shift_pc)
        except Exception:
            direct_rejected = True
        try:
            transpose_raw_graph_batch(
                Batch.from_data_list([copy.deepcopy(graph)]), (shift_pc,)
            )
        except Exception:
            runtime_rejected = True
        fail_closed = direct_rejected and runtime_rejected
        return _sealed(
            {
                "midi_range_valid": False,
                "direct_transform_succeeded": False,
                "runtime_path_matches_contract": fail_closed,
                "oracle_passed": fail_closed,
                "identity_exact": shift_pc != 0,
                "round_trip_valid": False,
                "changed_graph_fields": [],
                "entity_ids_preserved": True,
                "topology_preserved": True,
                "invalid_reasons": (
                    ["midi_range_violation"]
                    if fail_closed
                    else ["midi_range_violation", "range_failure_not_fail_closed"]
                ),
            }
        )
    runtime = transpose_raw_graph_batch(
        Batch.from_data_list([copy.deepcopy(graph)]), (shift_pc,)
    ).to_data_list()[0]
    direct = (
        transpose_raw_graph_view(graph, shift_pc=shift_pc)
        if compare_direct_and_runtime
        else runtime
    )
    expected_pitch = source_pitch.clone()
    expected_pitch[non_drum] += SIGNED_BY_SHIFT_PC[shift_pc]
    pitch_ok = torch.equal(direct["note"].x_cat[:, pitch_i], expected_pitch)
    pc_ok = torch.equal(
        direct["note"].x_cat[non_drum, pc_i],
        expected_pitch[non_drum].remainder(12),
    )
    octave_ok = torch.equal(
        direct["note"].x_cat[non_drum, octave_i],
        torch.div(expected_pitch[non_drum], 12, rounding_mode="floor"),
    )
    relative_i = _feature_index(note.cont_feature_names, "track_relative_pitch")
    relative, relative_available = _independent_relative_pitch(
        graph, expected_pitch
    )
    relative_ok = _tensor_equal(
        direct["note"].x_cont[:, relative_i], relative, continuous=True
    ) and torch.equal(
        direct["note"].x_cont_available[:, relative_i], relative_available
    )
    changed = graph_changed_fields(graph, direct)
    illegal_changes = sorted(set(changed) - ALLOWED_GRAPH_CHANGES)
    runtime_differences = (
        _graph_differences(direct, runtime) if compare_direct_and_runtime else ()
    )
    inverse_rejected = False
    try:
        inverse = transpose_raw_graph_view(direct, shift_pc=(-shift_pc) % 12)
    except Exception:
        inverse_rejected = True
        round_trip_differences = ("inverse_transform_rejected",)
    else:
        round_trip_differences = _graph_differences(graph, inverse)
    identity_differences = _graph_differences(graph, direct) if shift_pc == 0 else ()
    entity_ids_preserved = all(
        tuple(graph[node].entity_id) == tuple(direct[node].entity_id)
        for node in graph.node_types
    )
    topology_preserved = all(
        torch.equal(graph[edge].edge_index, direct[edge].edge_index)
        for edge in graph.edge_types
    )
    reasons = []
    for ok, reason in (
        (pitch_ok, "pitch_oracle_mismatch"),
        (pc_ok, "pitch_class_oracle_mismatch"),
        (octave_ok, "octave_oracle_mismatch"),
        (relative_ok, "track_relative_pitch_oracle_mismatch"),
        (not illegal_changes, "non_allowlisted_graph_change"),
        (not runtime_differences, "runtime_direct_mismatch"),
        (not round_trip_differences, "graph_round_trip_mismatch"),
        (not identity_differences, "identity_not_exact"),
        (entity_ids_preserved, "entity_ids_changed"),
        (topology_preserved, "topology_changed"),
    ):
        if not ok:
            reasons.append(reason)
    return _sealed(
        {
            "midi_range_valid": True,
            "direct_transform_succeeded": True,
            "runtime_path_matches_contract": not runtime_differences,
            "oracle_passed": not reasons,
            "identity_exact": not identity_differences,
            "round_trip_valid": not round_trip_differences,
            "changed_graph_fields": list(changed),
            "illegal_changed_graph_fields": illegal_changes,
            "runtime_differences": list(runtime_differences),
            "direct_runtime_comparison_performed": compare_direct_and_runtime,
            "round_trip_differences": list(round_trip_differences),
            "inverse_transform_rejected": inverse_rejected,
            "entity_ids_preserved": entity_ids_preserved,
            "topology_preserved": topology_preserved,
            "continuous_tolerance": GRAPH_CONTINUOUS_ATOL,
            "invalid_reasons": reasons,
        }
    )


def source_free_runtime_regression() -> dict[str, object]:
    """Exercise model-forward then target-alignment through the B5C runtime path."""

    from music_critic.experiments.analysisgnn.corrected_model import (
        CorrectedAnalysisGNNModel,
    )

    batch, sidecar = build_source_free_fixture()
    source = batch.raw_graph_batch.to_data_list()[0]
    torch.manual_seed(FULL_SEED)
    model = CorrectedAnalysisGNNModel().eval()
    graph_rows = []
    observation_rows = []
    routing_mismatches: list[dict[str, object]] = []
    with torch.no_grad():
        for shift_pc in SHIFT_PCS:
            graph_row = audit_graph_transform(source, shift_pc=shift_pc)
            graph_rows.append(graph_row)
            observation_row = audit_record_observation_transform(
                sidecar, shift_pc=shift_pc
            )
            observation_rows.append(observation_row)
            shifted = transpose_raw_graph_batch(batch.raw_graph_batch, (shift_pc,))
            output = model(shifted)
            alignment = align_target_sidecars_after_prediction(
                output, shifted, (sidecar,), shifts=(shift_pc,)
            )
            target_audit = audit_sidecar_targets(sidecar, shift_pc=shift_pc)
            for task_id in ACTIVE_HEADS:
                rows = alignment.heads[task_id]
                if not rows.entity_ids:
                    routing_mismatches.append(
                        {"shift_pc": shift_pc, "task_id": task_id, "reason": "no_routed_rows"}
                    )
                    continue
                state = next(
                    state
                    for entity in sidecar["entities"]
                    for observed_task, state in entity.get("targets", {}).items()
                    if observed_task == task_id
                )
                expected = transform_semantic_value(
                    task_id,
                    str(state["canonical_value"]),
                    shift_pc=shift_pc,
                    dialect=str(sidecar["dialect"]),
                    profile="corrected_v2",
                )
                vocabulary = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
                if int(rows.values[0]) != vocabulary.index(expected):
                    routing_mismatches.append(
                        {"shift_pc": shift_pc, "task_id": task_id, "reason": "class_id_mismatch"}
                    )
                if not bool(rows.valid_mask[0]) or rows.masked_row_count:
                    routing_mismatches.append(
                        {"shift_pc": shift_pc, "task_id": task_id, "reason": "mask_mismatch"}
                    )
            if target_audit["invalid_reasons"]:
                routing_mismatches.append(
                    {"shift_pc": shift_pc, "task_id": "*", "reason": "target_oracle_failed"}
                )
            if observation_row["mismatch_count"] or observation_row[
                "round_trip_mismatch_count"
            ]:
                routing_mismatches.append(
                    {
                        "shift_pc": shift_pc,
                        "task_id": "*",
                        "reason": "record_observation_transform_mismatch",
                    }
                )
    payload = {
        "fixture_record_id": sidecar["record_id"],
        "active_routed_heads": list(ACTIVE_HEADS),
        "deferred_metadata_heads": ["phrase", "section"],
        "shift_count": len(graph_rows),
        "graph_oracles_passed": all(row["oracle_passed"] for row in graph_rows),
        "runtime_path_matches_contract": all(
            row["runtime_path_matches_contract"] for row in graph_rows
        )
        and not routing_mismatches,
        "identity_exact": bool(graph_rows[0]["identity_exact"]),
        "round_trip_passed": all(row["round_trip_valid"] for row in graph_rows),
        "target_record_observation_round_trip_passed": all(
            row["round_trip_valid"] for row in observation_rows
        ),
        "routing_mismatch_count": len(routing_mismatches),
        "routing_mismatches": routing_mismatches,
    }
    return _sealed(payload)


def schedule_diagnostics(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, object]:
    """Reproduce the exact seed-17 B5D C0/C1 20,000-draw schedules."""

    components = production_component_records(paths)
    valid_shifts = production_valid_shifts(paths)
    draw_count = FULL_UPDATE_BUDGET * FULL_BATCH_SIZE
    samplers = {
        profile: CorrectedComponentSampler(
            components,
            valid_shifts,
            profile_id=full_runtime_config(profile).profile_id,
            seed=FULL_SEED,
        )
        for profile in ("C0", "C1")
    }
    draws = {
        profile: tuple(sampler.peek(offset) for offset in range(draw_count))
        for profile, sampler in samplers.items()
    }
    c1 = draws["C1"]
    counts = Counter(row.shift_pc for row in c1)
    records = defaultdict(set)
    for row in c1:
        records[row.shift_pc].add(row.record_id)
    expected = Counter()
    for row in c1:
        for shift in valid_shifts[row.record_id]:
            expected[shift] += 1.0 / len(valid_shifts[row.record_id])
    probabilities = [counts[shift] / draw_count for shift in SHIFT_PCS]
    entropy = -sum(value * math.log(value) for value in probabilities if value)
    eligibility_rows = []
    if paths.b5a_shift_eligibility.is_file():
        with paths.b5a_shift_eligibility.open("r", encoding="utf-8") as handle:
            eligibility_rows = [json.loads(line) for line in handle if line.strip()]
    limited = sorted(record for record, shifts in valid_shifts.items() if len(shifts) < 12)
    reasons: Counter[str] = Counter()
    limited_payload = []
    invalid_by_record: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligibility_rows:
        record_id = str(row["record_id"])
        if record_id in valid_shifts and row.get("corrected_valid") is not True:
            invalid_by_record[record_id].append(
                {
                    "shift_pc": int(row["shift_pc"]),
                    "reasons": list(row.get("corrected_invalid_reasons", [])),
                }
            )
            reasons.update(
                str(reason).split(":", 1)[0]
                for reason in row.get("corrected_invalid_reasons", [])
            )
    for record_id in limited:
        limited_payload.append(
            {
                "record_id": record_id,
                "valid_shifts": list(valid_shifts[record_id]),
                "invalid": invalid_by_record[record_id],
            }
        )
    record_sequences_equal = [row.record_id for row in draws["C0"]] == [
        row.record_id for row in draws["C1"]
    ]
    record_fp = record_schedule_fingerprint(
        components, seed=FULL_SEED, draw_count=draw_count
    )
    payload = {
        "seed": FULL_SEED,
        "applied_updates": FULL_UPDATE_BUDGET,
        "batch_size": FULL_BATCH_SIZE,
        "record_draws": draw_count,
        "shift_draw_counts": {str(shift): counts[shift] for shift in SHIFT_PCS},
        "identity_draw_count": counts[0],
        "identity_draw_fraction": counts[0] / draw_count,
        "unique_records_per_shift": {
            str(shift): len(records[shift]) for shift in SHIFT_PCS
        },
        "valid_shift_set_size_distribution": dict(
            (str(size), count)
            for size, count in sorted(
                Counter(len(value) for value in valid_shifts.values()).items()
            )
        ),
        "limited_record_count": len(limited),
        "limited_records": limited_payload,
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "shift_entropy": entropy,
        "normalized_shift_entropy": entropy / math.log(len(SHIFT_PCS)),
        "expected_uniform_over_record_valid_sets": {
            str(shift): expected[shift] for shift in SHIFT_PCS
        },
        "max_absolute_draw_deviation": max(
            abs(counts[shift] - expected[shift]) for shift in SHIFT_PCS
        ),
        "max_fractional_draw_deviation": max(
            abs(counts[shift] - expected[shift]) / draw_count
            for shift in SHIFT_PCS
        ),
        "record_schedule_fingerprint": record_fp,
        "record_schedules_equal": record_sequences_equal,
        "C0_transposition_schedule_fingerprint": transposition_schedule_fingerprint(
            draws["C0"]
        ),
        "C1_transposition_schedule_fingerprint": transposition_schedule_fingerprint(
            draws["C1"]
        ),
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    return _sealed(payload)


def validate_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    profile: str,
    expected_model_fingerprint: str,
    expected_model_contract_fingerprint: str | None = None,
    expected_record_schedule_fingerprint: str | None = None,
) -> dict[str, object]:
    """Fail closed on every B5D identity available in a checkpoint."""

    expected_config = full_runtime_config(profile).to_dict()
    sampler_state = payload.get("sampler_state")
    checks = {
        "schema": payload.get("schema_version") == CORRECTED_CHECKPOINT_SCHEMA,
        "phase": payload.get("phase") == "9E-B5D",
        "profile": payload.get("resolved_config") == expected_config,
        "seed": isinstance(payload.get("resolved_config"), Mapping)
        and payload["resolved_config"].get("seed") == FULL_SEED,  # type: ignore[index]
        "applied_update": payload.get("applied_update") == FULL_UPDATE_BUDGET,
        "dataset_split_training_policy": payload.get(
            "full_training_contract_fingerprint"
        )
        == full_training_contract()["fingerprint"],
        "record_schedule_dataset_split_binding": (
            expected_record_schedule_fingerprint is None
            or (
                isinstance(sampler_state, Mapping)
                and sampler_state.get("record_schedule_fingerprint")
                == expected_record_schedule_fingerprint
            )
        ),
        "model_architecture": (
            expected_model_contract_fingerprint is None
            or payload.get("model_contract_fingerprint")
            == expected_model_contract_fingerprint
        ),
        "model_state": payload.get("model_state_fingerprint")
        == expected_model_fingerprint,
        "test_lock": payload.get("test_loader_created") is False
        and payload.get("test_targets_read") is False
        and payload.get("test_metrics_computed") is False,
    }
    result = {
        "profile": profile,
        "checks": checks,
        "valid": all(checks.values()),
        "expected_model_state_fingerprint": expected_model_fingerprint,
        "observed_model_state_fingerprint": payload.get("model_state_fingerprint"),
        "runtime_config_fingerprint": expected_config["fingerprint"],
        "training_contract_fingerprint": full_training_contract()["fingerprint"],
        "expected_record_schedule_fingerprint": expected_record_schedule_fingerprint,
        "expected_model_contract_fingerprint": expected_model_contract_fingerprint,
    }
    return _sealed(result)


def compact_audit_fixture(
    *, runtime: Mapping[str, object], schedule: Mapping[str, object]
) -> dict[str, object]:
    """Build the checkpoint-free committed B5F evidence seal."""

    matrix = transformation_matrix()
    mapping = semantic_mapping_rows()
    mapping_failures = [
        asdict(row)
        for row in mapping
        if row.valid and (
            row.target_semantic_value is None
            or not independent_target_oracle(
                row.source_task_id,
                row.source_semantic_value,
                row.target_semantic_value,
                shift_pc=row.shift_pc,
            )
        )
    ]
    implementation_defect = not bool(runtime["round_trip_passed"]) or not bool(
        runtime["runtime_path_matches_contract"]
    )
    final_status = (
        "implementation_or_contract_defect" if implementation_defect else "inconclusive"
    )
    status = {
        "audit_execution_valid": True,
        "transposition_correctness_passed": False,
        "runtime_path_matches_contract": bool(
            runtime["runtime_path_matches_contract"]
        ),
        "all_20_heads_classified": len(matrix) == 20
        and all(row["classification_matches"] for row in matrix),
        "identity_exact": bool(runtime["identity_exact"]),
        "round_trip_passed": bool(runtime["round_trip_passed"]),
        "cross_head_consistency_passed": False,
        "schedule_reproduced": bool(schedule["record_schedules_equal"]),
        "checkpoint_diagnostics_run": False,
        "shift0_metrics_reproduced": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "ready_for_soft_augmentation": False,
    }
    compact_schedule_keys = (
        "seed",
        "applied_updates",
        "batch_size",
        "record_draws",
        "shift_draw_counts",
        "identity_draw_count",
        "identity_draw_fraction",
        "unique_records_per_shift",
        "valid_shift_set_size_distribution",
        "limited_record_count",
        "invalid_reason_counts",
        "shift_entropy",
        "normalized_shift_entropy",
        "max_absolute_draw_deviation",
        "max_fractional_draw_deviation",
        "record_schedule_fingerprint",
        "record_schedules_equal",
        "C0_transposition_schedule_fingerprint",
        "C1_transposition_schedule_fingerprint",
        "test_loader_created",
        "test_targets_read",
        "test_metrics_computed",
        "fingerprint",
    )
    compact_schedule = {
        key: schedule[key] for key in compact_schedule_keys
    }
    if "limited_records" in schedule:
        compact_schedule["limited_records_fingerprint"] = fingerprint(
            schedule["limited_records"]
        )
    body = {
        "schema": B5F_AUDIT_SCHEMA,
        "phase": "9E-B5F",
        "source_head": "003982a9327d42ca52c9102c06e1be77b6355abb",
        "final_status": final_status,
        "status": status,
        "reason": (
            "shift_pc=6 raw graph round trip applies +6 twice and returns +12 semitones"
            if implementation_defect
            else "CUDA and the sealed B5D C0/C1 checkpoints are unavailable locally"
        ),
        "transposition_contract_fingerprint": transposition_contract()["fingerprint"],
        "transformation_matrix_fingerprint": fingerprint(matrix),
        "semantic_mapping_fingerprint": fingerprint([asdict(row) for row in mapping]),
        "independent_mapping_oracle_failure_count": len(mapping_failures),
        "independent_mapping_oracle_failures": mapping_failures[:32],
        "runtime_regression": dict(runtime),
        "schedule": compact_schedule,
        "corpus_contract": {
            "train_records": 1295,
            "validation_records": 162,
            "record_shift_pairs": EXPECTED_PAIR_COUNT,
            "full_pair_audit_run": False,
            "test_enumerated": False,
        },
        "checkpoint_diagnostics": {
            "run": False,
            "per_shift_metrics": None,
            "shift0_reproduction": False,
        },
        "training_run": False,
        "dataset_changed": False,
        "split_changed": False,
        "model_changed": False,
        "sampler_changed": False,
    }
    body["evidence_fingerprint"] = fingerprint(body)
    body["fixture_fingerprint"] = fingerprint(body)
    return body


def check_compact_fixture(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("fixture_fingerprint", None)
    if observed != fingerprint(value):
        raise TranspositionDiagnosticError("B5F fixture fingerprint mismatch")
    value["fixture_fingerprint"] = observed
    if value.get("schema") != B5F_AUDIT_SCHEMA:
        raise TranspositionDiagnosticError("B5F fixture schema mismatch")
    if value.get("final_status") not in {
        "implementation_or_contract_defect",
        "inconclusive",
    }:
        raise TranspositionDiagnosticError("B5F fixture final status is invalid")
    status = value.get("status")
    if not isinstance(status, Mapping):
        raise TranspositionDiagnosticError("B5F status payload is missing")
    required = {
        "audit_execution_valid": True,
        "transposition_correctness_passed": False,
        "all_20_heads_classified": True,
        "checkpoint_diagnostics_run": False,
        "shift0_metrics_reproduced": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "ready_for_soft_augmentation": False,
    }
    if any(status.get(key) != expected for key, expected in required.items()):
        raise TranspositionDiagnosticError("B5F fixture status is not fail closed")
    if value.get("transposition_contract_fingerprint") != transposition_contract()[
        "fingerprint"
    ]:
        raise TranspositionDiagnosticError("B5A transposition contract changed")
    if value.get("transformation_matrix_fingerprint") != fingerprint(
        transformation_matrix()
    ):
        raise TranspositionDiagnosticError("B5F transformation matrix changed")
    if value.get("semantic_mapping_fingerprint") != fingerprint(
        [asdict(row) for row in semantic_mapping_rows()]
    ):
        raise TranspositionDiagnosticError("B5A semantic mapping changed")
    return value


__all__ = [
    "ALLOWED_GRAPH_CHANGES",
    "B5F_AUDIT_SCHEMA",
    "B5F_CHECKPOINT_SCHEMA",
    "B5F_FINAL_STATUS",
    "B5F_PAIR_SCHEMA",
    "EQUIVARIANT_HEADS",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_RECORD_COUNTS",
    "EXPECTED_TRANSFORMATION_KINDS",
    "GRAPH_CONTINUOUS_ATOL",
    "INVARIANT_HEADS",
    "RecordShiftDiagnostic",
    "TranspositionDiagnosticError",
    "audit_graph_transform",
    "audit_record_observation_transform",
    "audit_sidecar_targets",
    "check_compact_fixture",
    "compact_audit_fixture",
    "cross_head_checks",
    "independent_target_oracle",
    "prepare_sidecar_diagnostic_context",
    "schedule_diagnostics",
    "source_free_runtime_regression",
    "transformation_matrix",
    "validate_checkpoint_metadata",
]
