"""Fusion-bound POP evaluation with the unchanged frozen metric arithmetic.

Model-only exports must come from separately validated training checkpoints;
this evaluator never treats a model snapshot as a training-resume artifact.
The old corrected loader, synthetic evaluator and transfer API are unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import platform
import re
from time import perf_counter
from typing import Callable, Iterable

import torch

from music_critic.experiments.pop909_cl_frozen.contracts import (
    VIEW_IDS, canonical_json, evaluation_protocol, fingerprint, guard_execution,
)
from music_critic.experiments.pop909_cl_frozen.evaluation import (
    EvaluationPiece, SourceTargets, _predict_piece, _summarize, align_event_start,
)
from music_critic.experiments.pop909_cl_frozen.projection import quality_gold_class_id
from music_critic.experiments.pop909_cl_frozen import transfer

from .corrected_model import model_state_fingerprint
from .harmonic_fusion_checkpoint import (
    FUSION_CHECKPOINT_EVIDENCE_SCHEMA, FUSION_CHECKPOINT_SCHEMA,
    LoadedFusionCheckpoint, load_fusion_checkpoint,
)
from .harmonic_logit_fusion import HarmonicFusionModel, fusion_model_contract


PROTOCOL_SCHEMA = "HarmonicFusionPopEvaluationProtocol@1.0.0"
SYNTHETIC_REPORT_SCHEMA = "HarmonicFusionPopSyntheticReport@1.0.0"
TRANSFER_PREFLIGHT_SCHEMA = "HarmonicFusionPopTransferPreflight@1.0.0"
TRANSFER_REPORT_SCHEMA = "HarmonicFusionPopTransferReport@1.0.0"


def fusion_evaluation_protocol(*, synthetic: bool = False) -> dict:
    frozen = evaluation_protocol()
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "frozen_metric_protocol": frozen,
        "prediction_source": "enabled_base_plus_fusion_all_18_ordered_heads",
        "checkpoint_schema": FUSION_CHECKPOINT_SCHEMA,
        "checkpoint_evidence_schema": FUSION_CHECKPOINT_EVIDENCE_SCHEMA,
        "checkpoint_loading": "sha_size_bound_complete_fusion_cpu_weights_only_strict",
        "execution_policy": ("synthetic_only_test_denied_before_readers" if synthetic
                             else "explicit_registered_train_validation_cpu_only"),
        "selection": "dilemmadata_validation_only_no_pop_tuning_or_checkpoint_selection",
        "claim": "pop_development_transfer_diagnostic_not_new_independent_zero_shot_evidence",
        "training_resume": False,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _checked_model_binding(model: HarmonicFusionModel, evidence: dict | None) -> tuple[dict, str, dict | None]:
    if not isinstance(model, HarmonicFusionModel) or model.fusion is None:
        raise ValueError("harmonic_fusion_eval.enabled_fusion_model_required")
    if (any(module.training for module in model.modules())
            or any(p.device.type != "cpu" for p in model.parameters())):
        raise ValueError("harmonic_fusion_eval.cpu_eval_model_required")
    contract = fusion_model_contract(model)
    state = model_state_fingerprint(model)
    if evidence is not None:
        expected = {
            "schema_version": FUSION_CHECKPOINT_EVIDENCE_SCHEMA,
            "checkpoint_schema": FUSION_CHECKPOINT_SCHEMA,
            "model_contract": contract,
            "model_contract_fingerprint": contract["fingerprint"],
            "model_state_fingerprint": state,
            "contract_verification": "full_payload_and_current_fingerprint",
            "load_policy": "cpu_weights_only_strict_model_state",
            "optimizer_restored": False, "checkpoint_rng_applied": False,
            "caller_rng_preserved": True, "model_training": False,
            "parameters_require_grad": False,
        }
        if (not isinstance(evidence, dict)
                or canonical_json({key: evidence.get(key) for key in expected}) != canonical_json(expected)
                or not isinstance(evidence.get("artifact_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", evidence["artifact_sha256"])
                or type(evidence.get("artifact_size_bytes")) is not int or evidence["artifact_size_bytes"] <= 0
                or any(p.requires_grad for p in model.parameters())):
            raise ValueError("harmonic_fusion_eval.checkpoint_evidence_binding_mismatch")
        evidence = deepcopy(evidence)
    return contract, state, evidence


def _evaluate_fusion_core(model: HarmonicFusionModel, inputs: Iterable[EvaluationPiece], *,
                          checkpoint_evidence: dict | None = None) -> dict:
    """Own model binding and orchestration; reuse exact WS4 metric functions.

The old core hardcodes the corrected-model contract. We deliberately do not
impersonate that model or patch its contract/predict method to reuse the core.
Both views, candidate gathering, projection, rational alignment, exclusions,
confusions and denominator arithmetic use its existing functions unchanged.
"""
    contract, state, evidence = _checked_model_binding(model, checkpoint_evidence)
    per_view_rows = {view_id: [] for view_id in VIEW_IDS}
    per_view_bindings = {view_id: [] for view_id in VIEW_IDS}
    piece_rows = []
    seen = set()
    for item in inputs:
        piece = item.piece
        if piece.split is not None and piece.split.lower() == "test":
            raise ValueError("harmonic_fusion_eval.test_access_forbidden")
        if piece.piece_id in seen or not item.group_id:
            raise ValueError("harmonic_fusion_eval.unique_piece_and_group_identity_required")
        seen.add(piece.piece_id)
        predictions = {}
        for view_id in VIEW_IDS:
            predictions[view_id], binding = _predict_piece(model, piece, view_id)
            per_view_bindings[view_id].append({"piece_id": piece.piece_id, "group_id": item.group_id, **binding})
        targets = item.read_targets()
        if not isinstance(targets, SourceTargets):
            raise ValueError("harmonic_fusion_eval.source_targets_required")
        events = sorted(targets.events, key=lambda event: (event.start_qn, event.event_id))
        piece_rows.append({
            "piece_id": piece.piece_id, "group_id": item.group_id,
            "source_status": targets.source_status, "event_count": len(events),
            "no_chord_span_count": targets.no_chord_span_count,
            "trailing_unannotated_span_count": targets.trailing_unannotated_span_count,
            "target_fingerprint": fingerprint([asdict(event) for event in events]),
        })
        for event in events:
            index, alignment_reason = align_event_start(event.start_qn, piece.beats)
            for view_id in VIEW_IDS:
                row = {"piece_id": piece.piece_id, "group_id": item.group_id, "event_id": event.event_id,
                       "start_qn": asdict(event.start_qn),
                       "beat_id": None if index is None else piece.beats[index].beat_id,
                       "alignment_reason": alignment_reason}
                for task in ("root", "quality"):
                    gold = event.root_pc if task == "root" else (
                        quality_gold_class_id(event.quality) if event.quality_available else None)
                    label_reason = getattr(event, f"{task}_reason")
                    if getattr(event, f"{task}_available") and gold is None:
                        label_reason = "unsupported"
                    row[task] = {"gold": gold,
                                 "prediction": None if index is None else predictions[view_id][task][index],
                                 "reason": label_reason or alignment_reason, "label_reason": label_reason,
                                 "source_label": event.root_pc if task == "root" else event.quality}
                per_view_rows[view_id].append(row)
    if (model_state_fingerprint(model) != state or fusion_model_contract(model) != contract
            or any(module.training for module in model.modules())
            or (evidence is not None and any(p.requires_grad for p in model.parameters()))):
        raise ValueError("harmonic_fusion_eval.model_mutated_during_evaluation")
    piece_rows.sort(key=lambda row: row["piece_id"])
    views = {}
    for view_id in VIEW_IDS:
        rows = sorted(per_view_rows[view_id], key=lambda row: (row["piece_id"], row["event_id"]))
        bindings = sorted(per_view_bindings[view_id], key=lambda row: row["piece_id"])
        views[view_id] = {"role": "primary" if view_id == "native" else "secondary",
                          "input_bindings": bindings, "view_fingerprint": fingerprint(bindings),
                          **_summarize(rows, piece_rows), "event_observations": rows}
    return {"model": {"contract": contract, "state_fingerprint": state,
                      "checkpoint_evidence": evidence}, "pieces": piece_rows, "views": views}


def evaluate_fusion_synthetic(model: HarmonicFusionModel, inputs: Iterable[EvaluationPiece], *,
                              split: str, checkpoint_evidence: dict | None = None) -> dict:
    """Source-free acceptance API; rejects non-synthetic splits before iteration."""
    guard_execution(split)
    core = _evaluate_fusion_core(model, inputs, checkpoint_evidence=checkpoint_evidence)
    payload = {
        "schema_version": SYNTHETIC_REPORT_SCHEMA, "protocol": fusion_evaluation_protocol(synthetic=True),
        "split": split,
        "provenance": {"kind": "caller_declared_synthetic_fixture", "scientific_quality_evidence": False},
        "access_observations": {"synthetic_piece_count": len(core["pieces"]),
                                "target_reader_calls": len(core["pieces"]),
                                "real_data_execution_enabled": False, "test_inputs_accessed": False,
                                "test_targets_accessed": False, "test_metrics_accessed": False},
        **core,
    }
    return {**payload, "report_fingerprint": fingerprint(payload)}


def evaluate_fusion_from_readers(*, split: str, checkpoint_reader: Callable[[], LoadedFusionCheckpoint],
                                 source_reader: Callable[[], Iterable[EvaluationPiece]]) -> dict:
    guard_execution(split)
    loaded = checkpoint_reader()
    if not isinstance(loaded, LoadedFusionCheckpoint) or not isinstance(loaded.evidence, dict):
        raise ValueError("harmonic_fusion_eval.checked_checkpoint_required")
    # Validate before even invoking the source reader, not only before iteration.
    _checked_model_binding(loaded.model, loaded.evidence)
    return evaluate_fusion_synthetic(loaded.model, source_reader(), split=split,
                                     checkpoint_evidence=loaded.evidence)


def fusion_transfer_preflight(request: transfer.TransferRequest) -> dict:
    """Reuse exact global membership validation; bind a distinct fusion protocol."""
    payload = transfer.preflight(request)
    payload.pop("binding_fingerprint")
    payload.update(schema_version=TRANSFER_PREFLIGHT_SCHEMA, protocol=fusion_evaluation_protocol())
    return {**payload, "binding_fingerprint": fingerprint(payload)}


def evaluate_fusion_transfer(request: transfer.TransferRequest, *, experiment_record: Path) -> dict:
    """Registered frozen TRAIN/VALIDATION diagnostic with strict fusion weights.

The planned record's ``pop909_transfer_binding_fingerprint`` must match the
new fusion preflight. A baseline transfer record cannot authorize this path.
Source construction/join reuses the hash-bound raw channel-0 reader unchanged.
"""
    transfer.guard_transfer(request.split)
    binding = fusion_transfer_preflight(request)
    record = transfer._registered_record(experiment_record, binding)
    started = perf_counter()
    loaded = load_fusion_checkpoint(request.checkpoint.path, expected_sha256=request.checkpoint.sha256,
                                    expected_size_bytes=request.checkpoint.size_bytes)
    core = _evaluate_fusion_core(loaded.model, transfer._inputs(request, binding),
                                 checkpoint_evidence=loaded.evidence)
    payload = {
        "schema_version": TRANSFER_REPORT_SCHEMA, "protocol": fusion_evaluation_protocol(), "split": request.split,
        "binding": binding, "experiment_id": record["experiment_id"],
        "planned_record_fingerprint": record["record_fingerprint"],
        "provenance": {"kind": "explicit_indexed_pop909_cl_source", "music_quality_evidence": False,
                       "independent_zero_shot_evidence": False},
        "access_observations": {"source_piece_count": len(core["pieces"]),
                                "target_reader_calls": len(core["pieces"]),
                                "validation_accessed": request.split == "validation",
                                "test_inputs_accessed": False, "test_targets_accessed": False,
                                "test_metrics_computed": False, "test_used_for_selection": False},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "threads": torch.get_num_threads(),
                    "wall_seconds": perf_counter() - started, "applied_updates": 0},
        **core,
    }
    payload["scientific_fingerprint"] = fingerprint({key: value for key, value in payload.items() if key != "runtime"})
    return {**payload, "report_fingerprint": fingerprint(payload)}


__all__ = [
    "fusion_evaluation_protocol", "evaluate_fusion_synthetic", "evaluate_fusion_from_readers",
    "fusion_transfer_preflight", "evaluate_fusion_transfer",
]
