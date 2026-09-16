"""Checked, separate native raw/expanded-target reads for the local control.

The coordinator's registered binding audit supplies the expected cache bytes.
This reader neither creates production caches nor reconstructs the source corpus.
It verifies selected identities and the same frozen joins after raw processing.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Mapping

import torch

from music_critic.data import CanonicalPiece, RationalTime, piece_from_dict
from music_critic.experiments.context_diagnostics import (
    FileBinding, Member, TargetRow, fingerprint, guard_split, require_sha,
)
from music_critic.experiments.analysisgnn.corrected_model import corrected_task_head_specs
from music_critic.experiments.analysisgnn.corrected_training import align_target_sidecars_after_prediction
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary


MAX_RAW_BYTES = 32 * 1024 * 1024
MAX_TARGET_BYTES = 64 * 1024 * 1024
MAX_ENTITIES = 20_000
MAX_RELATIONS = 60_000


def _fail(reason: str) -> None:
    raise ValueError(f"context_local_exports.{reason}")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def checked_json(binding: FileBinding, *, max_bytes: int = MAX_TARGET_BYTES):
    def invalid_constant(_value):
        _fail("nonfinite_json")
    return json.loads(binding.read(max_bytes=max_bytes), object_pairs_hook=_object_pairs,
                      parse_constant=invalid_constant, parse_float=_finite_float)


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        _fail("nonfinite_json")
    return result


def resolve_binding(payload: Mapping, root: Path) -> FileBinding:
    if not isinstance(payload, dict) or set(payload) != {"locator", "sha256", "size_bytes"}:
        _fail("explicit_file_binding_required")
    locator = payload["locator"]
    if not isinstance(locator, str) or not locator or "\\" in locator:
        _fail("portable_locator_required")
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(p in {"", ".", ".."} for p in locator.split("/")) or ":" in locator:
        _fail("portable_locator_required")
    base = root.resolve()
    path = (base / locator).resolve()
    if not path.is_relative_to(base):
        _fail("locator_escaped_input_root")
    require_sha(payload["sha256"])
    size = payload["size_bytes"]
    if type(size) is not int or size <= 0:
        _fail("positive_size_required")
    return FileBinding(str(path), payload["sha256"], size)


@dataclass(frozen=True)
class SelectedExport:
    record_id: str
    member: Member
    dialect: str
    raw: FileBinding
    targets: FileBinding
    metadata: FileBinding


def read_raw(export: SelectedExport) -> CanonicalPiece:
    """No expanded target read; stamp split only from verified membership."""
    guard_split(export.member.split)
    metadata = checked_json(export.metadata, max_bytes=16 * 1024)
    if metadata != {"record_id": export.record_id, "split": export.member.split,
                    "piece_sha256": export.raw.sha256, "sidecar_sha256": export.targets.sha256}:
        _fail("native_cache_metadata_mismatch")
    piece = piece_from_dict(checked_json(export.raw, max_bytes=MAX_RAW_BYTES))
    if (piece.piece_id != export.member.piece_id or piece.source_group_id != export.member.component_id
            or piece.dataset_name != "dilemmadata" or piece.split not in {None, export.member.split}):
        _fail("raw_official_identity_mismatch")
    if piece.annotations or piece.targets:
        _fail("raw_export_contains_semantic_targets")
    if len(piece.notes) > 16384:
        _fail("note_budget_exceeded")
    return replace(piece, split=export.member.split)


def _exact_time(value: object) -> RationalTime:
    if (not isinstance(value, dict) or set(value) != {"num", "den"}
            or type(value["num"]) is not int or type(value["den"]) is not int):
        _fail("exact_canonical_time_required")
    time = RationalTime(value["num"], value["den"])
    if (time.num, time.den) != (value["num"], value["den"]):
        _fail("unreduced_time")
    return time


def read_targets(export: SelectedExport) -> dict:
    """Called only after raw features and, for VALIDATION, frozen predictions."""
    guard_split(export.member.split)
    sidecar = checked_json(export.targets)
    if not isinstance(sidecar, dict) or sidecar.get("schema_version") != "analysisgnn-source-native-target-sidecar-v1":
        _fail("expanded_native_sidecar_required")
    if (sidecar.get("record_id"), sidecar.get("source_component_id"), sidecar.get("dialect")) != (
            export.record_id, export.member.component_id, export.dialect):
        _fail("target_official_identity_mismatch")
    if sidecar.get("fingerprint") != fingerprint({k: v for k, v in sidecar.items() if k != "fingerprint"}):
        _fail("native_sidecar_fingerprint_mismatch")
    entities, relations = sidecar.get("entities"), sidecar.get("relations")
    if (not isinstance(entities, list) or len(entities) > MAX_ENTITIES
            or not isinstance(relations, list) or len(relations) > MAX_RELATIONS):
        _fail("expanded_sidecar_budget")
    seen = set()
    for entity in entities:
        if not isinstance(entity, dict):
            _fail("invalid_target_entity")
        identity = entity.get("canonical_entity_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            _fail("duplicate_or_missing_target_entity")
        seen.add(identity)
        if entity.get("entity_type") not in {"harmonic_event", "onset", "note"}:
            _fail("unsupported_target_entity")
        _exact_time(entity.get("onset_qn"))
        if not isinstance(entity.get("source_entity_id"), str) or not entity["source_entity_id"]:
            _fail("source_entity_provenance_required")
        targets = entity.get("targets")
        if not isinstance(targets, dict):
            _fail("independent_task_states_required")
        for task, state in targets.items():
            if task not in TASK_BY_ID or TASK_BY_ID[task].prediction_level != entity["entity_type"]:
                _fail("target_level_or_task_mismatch")
            if not isinstance(state, dict):
                _fail("invalid_target_state")
            if (type(state.get("available")) is not bool or type(state.get("masked")) is not bool
                    or state["available"] == state["masked"]):
                _fail("independent_boolean_masks_required")
            if (state.get("canonical_entity_id") != identity
                    or state.get("source_entity_id") != entity["source_entity_id"]
                    or not isinstance(state.get("provenance"), dict) or not state["provenance"]):
                _fail("target_state_provenance_mismatch")
            if state["available"]:
                if not isinstance(state.get("canonical_value"), str) or state.get("missing_reason") is not None:
                    _fail("invalid_available_state")
            elif (state.get("canonical_value") is not None
                  or not isinstance(state.get("missing_reason"), str) or not state["missing_reason"]):
                _fail("masked_gold_must_be_null_with_reason")
    seen_relations = set()
    for relation in relations:
        if (not isinstance(relation, dict)
                or set(relation) != {"relation", "source_entity_id", "target_entity_id"}
                or not all(isinstance(v, str) and v for v in relation.values())):
            _fail("invalid_relation")
        key = (relation["relation"], relation["source_entity_id"])
        if key in seen_relations:
            _fail("ambiguous_duplicate_join")
        seen_relations.add(key)
    if sidecar.get("entity_counts") != dict(Counter(e["entity_type"] for e in entities)):
        _fail("entity_count_mismatch")
    if sidecar.get("relation_counts") != dict(Counter(r["relation"] for r in relations)):
        _fail("relation_count_mismatch")
    return sidecar


def training_candidate_layout(view):
    """Only raw candidate indices for the owning aligner, without logits or model.

    TRAIN does not need a C2 forward: alignment consumes only these indices.
    They are built from the same frozen routing and raw graph as VALIDATION.
    """
    rows = []
    for spec in corrected_task_head_specs():
        size = view.graph[spec.node_types[0]].num_nodes
        rows.append(SimpleNamespace(task_id=spec.task_id, allowed_node_types=spec.node_types,
                                    global_entity_indices=torch.arange(size, dtype=torch.long),
                                    sample_indices=torch.zeros(size, dtype=torch.long)))
    return SimpleNamespace(predictions=tuple(rows))


def aligned_target_rows(view, sidecar: dict, tasks: tuple[str, ...], local_candidate_ids: dict[str, set[str]]):
    """Use the production mask/join result; retain source observations separately."""
    output = view.output if view.output is not None else training_candidate_layout(view)
    alignment = align_target_sidecars_after_prediction(output, view.batch, [sidecar])
    entities = {row["canonical_entity_id"]: row for row in sidecar["entities"]}
    output_by_task = {p.task_id: p for p in output.predictions}
    result = {}
    for task in tasks:
        head, prediction = alignment.heads[task], output_by_task[task]
        raw_ids = list(view.graph[prediction.allowed_node_types[0]].entity_id)
        diagnostics = {d.entity_id: d.category for d in alignment.diagnostics if d.task_id == task}
        rows = []
        for i, identity in enumerate(head.entity_ids):
            entity, state = entities[identity], entities[identity]["targets"][task]
            candidate_position = int(head.candidate_indices[i])
            aligned = diagnostics.get(identity) != "exact_alignment_failed"
            candidate_id = (raw_ids[int(prediction.global_entity_indices[candidate_position])]
                            if aligned else "unaligned:" + identity)
            available = bool(head.valid_mask[i])
            reason = None if available else (diagnostics.get(identity) or state.get("missing_reason") or "frozen_mask_unavailable")
            if available and candidate_id not in local_candidate_ids[task]:
                available, reason = False, "raw_candidate_at_piece_end"
            row = TargetRow(identity, candidate_id, int(head.values[i]) if available else None,
                            available, reason, json.dumps({"native_provenance": state["provenance"],
                                "source_entity_id": entity["source_entity_id"],
                                "sidecar_fingerprint": sidecar["fingerprint"]}, sort_keys=True, ensure_ascii=False), None)
            row.validate(len(get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels))
            rows.append(row)
        result[task] = tuple(rows)
    return result, {"target_sidecar_fingerprint": sidecar["fingerprint"],
                    "alignment_diagnostics": [vars(d) if hasattr(d, "__dict__") else
                        {name: getattr(d, name) for name in ("task_id", "sample_index", "entity_id", "category", "message")}
                        for d in alignment.diagnostics]}
