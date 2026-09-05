"""Candidate-first, exact source-event evaluation with independent masks.

The execution gate deliberately admits synthetic inputs only in this wave.
No source discovery, cache reader, training resume, or real-data loader exists.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence
import re

import torch
from torch_geometric.data import Batch

from music_critic.adapters.pop909_cl import Pop909ClChordEvidence
from music_critic.data import CanonicalBeat, CanonicalPiece, RationalTime
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, corrected_model_contract, model_state_fingerprint,
)
from music_critic.graph import validate_raw_graph_batch
from music_critic.tasks import ENTITY_NODE_TYPE_TO_CODE

from .contracts import REPORT_SCHEMA, VIEW_IDS, canonical_json, evaluation_protocol, fingerprint, guard_execution
from .checkpoint import CHECKPOINT_EVIDENCE_SCHEMA
from .projection import (
    predict_quality_ids, predict_root_ids, quality_gold_class_id, quality_subset_ids,
    quality_class_labels,
)
from .views import build_view

_UNAVAILABLE = ("missing", "ambiguous", "unsupported")


@dataclass(frozen=True)
class SourceEvent:
    event_id: str
    start_qn: RationalTime
    root_pc: int | None = None
    quality: str | None = None
    root_available: bool = False
    quality_available: bool = False
    root_reason: str | None = "missing"
    quality_reason: str | None = "missing"

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id or not isinstance(self.start_qn, RationalTime):
            raise ValueError("pop909_frozen.exact_source_event_required")
        for task in ("root", "quality"):
            available = getattr(self, f"{task}_available")
            reason = getattr(self, f"{task}_reason")
            if type(available) is not bool or (available and reason is not None) or (
                not available and reason not in _UNAVAILABLE
            ):
                raise ValueError("pop909_frozen.invalid_independent_mask_reason")
        if self.root_available and (type(self.root_pc) is not int or not 0 <= self.root_pc < 12):
            raise ValueError("pop909_frozen.invalid_gold_root")
        if self.quality_available and not isinstance(self.quality, str):
            raise ValueError("pop909_frozen.invalid_gold_quality")
        if (not self.root_available and self.root_pc is not None) or (
            not self.quality_available and self.quality is not None
        ):
            raise ValueError("pop909_frozen.masked_values_must_be_null")


@dataclass(frozen=True)
class SourceTargets:
    events: tuple[SourceEvent, ...] = ()
    source_status: str = "available"
    no_chord_span_count: int = 0
    trailing_unannotated_span_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or any(not isinstance(event, SourceEvent) for event in self.events):
            raise ValueError("pop909_frozen.source_event_tuple_required")
        if self.source_status not in {"available", "expected_missing"}:
            raise ValueError("pop909_frozen.invalid_source_status")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("pop909_frozen.duplicate_source_event")
        if self.source_status == "expected_missing" and self.events:
            raise ValueError("pop909_frozen.missing_source_with_events")
        if any(type(n) is not int or n < 0 for n in (
            self.no_chord_span_count, self.trailing_unannotated_span_count
        )):
            raise ValueError("pop909_frozen.invalid_span_count")


def targets_from_chord_evidence(evidence: Pop909ClChordEvidence) -> SourceTargets:
    """Use original block onset/PPQN, never clipped alignment or gold notes."""
    events = []
    for block in evidence.blocks:
        root = block.selected_candidate.root_pc if block.root_available and block.selected_candidate else None
        qualities = {candidate.quality for candidate in block.candidates}
        if block.quality_available and len(qualities) != 1:
            raise ValueError("pop909_frozen.inconsistent_source_quality_mask")
        if block.root_available and root is None:
            raise ValueError("pop909_frozen.inconsistent_source_root_mask")
        reason = block.normalization_status
        if reason not in _UNAVAILABLE:
            reason = "missing"
        events.append(SourceEvent(
            event_id=block.block_id,
            start_qn=RationalTime(block.onset_tick, block.ppqn),
            root_pc=root,
            quality=next(iter(qualities)) if block.quality_available else None,
            root_available=block.root_available,
            quality_available=block.quality_available,
            root_reason=None if block.root_available else reason,
            quality_reason=None if block.quality_available else reason,
        ))
    return SourceTargets(
        events=tuple(events), source_status=evidence.status,
        no_chord_span_count=sum(span.kind in {"leading_no_chord", "internal_no_chord"} for span in evidence.no_chord_spans),
        trailing_unannotated_span_count=sum(
            span.kind == "trailing_unannotated" for span in evidence.trailing_spans
        ),
    )


@dataclass(frozen=True)
class EvaluationPiece:
    piece: CanonicalPiece
    group_id: str
    read_targets: Callable[[], SourceTargets]


def align_event_start(start: RationalTime, beats: Sequence[CanonicalBeat]) -> tuple[int | None, str | None]:
    if not isinstance(start, RationalTime):
        raise ValueError("pop909_frozen.exact_source_event_required")
    matches = [index for index, beat in enumerate(beats)
               if beat.start_qn <= start < beat.start_qn + beat.duration_qn]
    if len(matches) == 1:
        return matches[0], None
    return None, "missing_containing_beat" if not matches else "ambiguous_containing_beat"


def _ratio(numerator: int | float, denominator: int, reason: str) -> dict:
    return {"value": numerator / denominator if denominator else None,
            "reason": None if denominator else reason}


def _task_metrics(pairs: list[tuple[int, int]], class_count: int, allowed_gold: Sequence[int]) -> dict:
    confusion = [[0] * class_count for _ in range(class_count)]
    for gold, predicted in pairs:
        confusion[gold][predicted] += 1
    supports = [sum(row) for row in confusion]
    predictions = [sum(row[column] for row in confusion) for column in range(class_count)]
    included = [index for index, support in enumerate(supports) if support]
    per_class = []
    for index in range(class_count):
        tp = confusion[index][index]
        fp, fn = predictions[index] - tp, supports[index] - tp
        per_class.append({
            "class_id": index, "support": supports[index], "predicted_count": predictions[index],
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn, "no_true_or_predicted_support"),
        })
    return {
        "eligible_events": len(pairs),
        "accuracy": _ratio(sum(confusion[i][i] for i in range(class_count)), len(pairs), "no_eligible_events"),
        "observed_true_class_macro_f1": _ratio(
            sum(per_class[i]["f1"]["value"] for i in included), len(included), "no_observed_true_classes"),
        "included_class_ids": included,
        "allowed_gold_class_ids": list(allowed_gold),
        "confusion_class_ids": list(range(class_count)),
        "confusion": confusion,
        "per_class": per_class,
        "outside_subset_predictions": sum(predictions[i] for i in range(class_count) if i not in allowed_gold),
    }


def _summarize(rows: list[dict], pieces: list[dict]) -> dict:
    tasks = {}
    for task, class_count, gold_ids in (("root", 12, tuple(range(12))), ("quality", len(quality_class_labels()), quality_subset_ids())):
        eligible = [row for row in rows if row[task]["reason"] is None]
        metrics = _task_metrics([(row[task]["gold"], row[task]["prediction"]) for row in eligible], class_count, gold_ids)
        exclusions = {reason: sum(row[task]["reason"] == reason for row in rows) for reason in (
            "missing", "ambiguous", "unsupported", "missing_containing_beat", "ambiguous_containing_beat"
        )}
        metrics["coverage"] = {
            "eligible_events": len(eligible), "total_events": len(rows),
            "eligible_pieces": len({row["piece_id"] for row in eligible}), "total_pieces": len(pieces),
            "eligible_groups": len({row["group_id"] for row in eligible}),
            "total_groups": len({piece["group_id"] for piece in pieces}),
            "event_fraction": _ratio(len(eligible), len(rows), "no_source_chord_events"),
            "piece_fraction": _ratio(len({row["piece_id"] for row in eligible}), len(pieces), "no_pieces"),
            "group_fraction": _ratio(len({row["group_id"] for row in eligible}), len({p["group_id"] for p in pieces}), "no_groups"),
            "excluded_events": exclusions,
            "unaligned_events": sum(row["alignment_reason"] is not None for row in rows),
            "otherwise_label_eligible_unaligned_events": exclusions["missing_containing_beat"] + exclusions["ambiguous_containing_beat"],
            "label_unavailable": {
                reason: {"events": sum(row[task]["label_reason"] == reason for row in rows),
                         "pieces": len({row["piece_id"] for row in rows if row[task]["label_reason"] == reason}),
                         "groups": len({row["group_id"] for row in rows if row[task]["label_reason"] == reason})}
                for reason in _UNAVAILABLE
            },
            "alignment_unavailable": {
                reason: {"events": sum(row["alignment_reason"] == reason for row in rows),
                         "pieces": len({row["piece_id"] for row in rows if row["alignment_reason"] == reason}),
                         "groups": len({row["group_id"] for row in rows if row["alignment_reason"] == reason})}
                for reason in ("missing_containing_beat", "ambiguous_containing_beat")
            },
        }
        tasks[task] = metrics
    joint = [row for row in rows if row["root"]["reason"] is None and row["quality"]["reason"] is None]
    grouped = defaultdict(list)
    for row in rows:
        if row["beat_id"] is not None:
            grouped[(row["piece_id"], row["beat_id"])].append(row)
    multiple = [group for group in grouped.values() if len(group) > 1]
    conflicts = {task: sum(len({r[task]["source_label"] for r in group if r[task]["source_label"] is not None}) > 1 for group in multiple)
                 for task in ("root", "quality")}
    return {
        "tasks": tasks,
        "root_quality_joint_accuracy": {
            **_ratio(sum(all(row[t]["gold"] == row[t]["prediction"] for t in ("root", "quality")) for row in joint),
                     len(joint), "no_jointly_eligible_events"),
            "eligible_events": len(joint),
        },
        "coverage": {
            "source_chord_events": len(rows), "pieces": len(pieces),
            "groups": len({piece["group_id"] for piece in pieces}),
            "missing_target_pieces": sum(piece["source_status"] == "expected_missing" for piece in pieces),
            "missing_target_groups": len({piece["group_id"] for piece in pieces if piece["source_status"] == "expected_missing"}),
            "pieces_without_chord_events": sum(piece["event_count"] == 0 for piece in pieces),
            "no_chord_spans": sum(piece["no_chord_span_count"] for piece in pieces),
            "trailing_unannotated_spans": sum(piece["trailing_unannotated_span_count"] for piece in pieces),
            "alignment_failures": dict(sorted(Counter(row["alignment_reason"] for row in rows if row["alignment_reason"]).items())),
        },
        "resolution": {
            "multiple_events_same_beat": len(multiple),
            "events_in_shared_beats": sum(len(group) for group in multiple),
            "conflicting_label_beats": conflicts,
            "policy": "keep_each_event_one_prediction_per_beat_no_conflict_resolution",
        },
    }


def _predict_piece(model: CorrectedAnalysisGNNModel, piece: CanonicalPiece, view_id: str) -> tuple[dict, dict]:
    view = build_view(piece, view_id)
    batch = Batch.from_data_list([view.graph])
    validate_raw_graph_batch(batch, sample_count=1)
    with torch.inference_mode():
        output = model.predict(batch)
    task_rows = {}
    binding = {
        "graph_fingerprint": view.graph_fingerprint,
        "model_input_fingerprint": view.model_input_fingerprint,
        "view_fingerprint": view.view_fingerprint,
        "candidate_ids": list(view.graph["beat"].entity_id),
        "logits_fingerprints": {},
    }
    for task in ("root", "quality"):
        candidates = [row for row in output.predictions if row.task_id == task]
        if len(candidates) != 1:
            raise ValueError("pop909_frozen.missing_or_duplicate_prediction_head")
        row = candidates[0]
        indices = row.global_entity_indices.cpu().tolist()
        if (row.allowed_node_types != ("beat",) or
            not torch.all(row.candidate_node_type_codes == ENTITY_NODE_TYPE_TO_CODE["beat"]) or
            not torch.all(row.sample_indices == 0) or
            sorted(indices) != list(range(len(piece.beats)))):
            raise ValueError("pop909_frozen.incompatible_beat_candidates")
        predicted = (predict_root_ids(row.logits) if task == "root" else predict_quality_ids(row.logits)).cpu().tolist()
        task_rows[task] = dict(zip(indices, predicted, strict=True))
        ordered_logits = row.logits[torch.argsort(row.global_entity_indices)].detach().cpu()
        binding["logits_fingerprints"][task] = model_state_fingerprint({task: ordered_logits})
    return task_rows, binding


def evaluate_frozen(model: CorrectedAnalysisGNNModel, inputs: Iterable[EvaluationPiece], *, split: str,
                    checkpoint_evidence: dict | None = None) -> dict:
    """Evaluate declared synthetic input providers; targets open after predict.

    Iterating ``inputs`` itself may read sources, so the gate must precede it.
    Only eval-mode CPU models are accepted; this function never changes mode.
    """
    guard_execution(split)
    if any(module.training for module in model.modules()) or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("pop909_frozen.cpu_eval_model_required")
    initial_state = model_state_fingerprint(model)
    model_contract = corrected_model_contract(model)
    if checkpoint_evidence is not None:
        expected = {
            "schema_version": CHECKPOINT_EVIDENCE_SCHEMA,
            "model_contract_fingerprint": model_contract["fingerprint"],
            "model_contract": model_contract,
            "model_state_fingerprint": initial_state,
            "load_policy": "cpu_weights_only_strict_model_state",
            "optimizer_restored": False, "checkpoint_rng_applied": False,
            "caller_rng_preserved": True, "model_training": False, "parameters_require_grad": False,
        }
        if (not isinstance(checkpoint_evidence, dict) or
            canonical_json({key: checkpoint_evidence.get(key) for key in expected}) != canonical_json(expected) or
            not isinstance(checkpoint_evidence.get("artifact_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", checkpoint_evidence["artifact_sha256"]) or
            type(checkpoint_evidence.get("artifact_size_bytes")) is not int or checkpoint_evidence["artifact_size_bytes"] <= 0):
            raise ValueError("pop909_frozen.checkpoint_evidence_binding_mismatch")
        checkpoint_evidence = deepcopy(checkpoint_evidence)
    per_view_rows = {view_id: [] for view_id in VIEW_IDS}
    per_view_bindings = {view_id: [] for view_id in VIEW_IDS}
    piece_rows = []
    seen = set()
    for item in inputs:
        piece = item.piece
        if piece.split is not None and piece.split.lower() == "test":
            raise ValueError("pop909_frozen.test_access_forbidden")
        if piece.piece_id in seen or not item.group_id:
            raise ValueError("pop909_frozen.unique_piece_and_group_identity_required")
        seen.add(piece.piece_id)
        predictions = {}
        for view_id in VIEW_IDS:
            predictions[view_id], binding = _predict_piece(model, piece, view_id)
            per_view_bindings[view_id].append({"piece_id": piece.piece_id, "group_id": item.group_id, **binding})
        targets = item.read_targets()
        if not isinstance(targets, SourceTargets):
            raise ValueError("pop909_frozen.source_targets_required")
        ordered_events = sorted(targets.events, key=lambda e: (e.start_qn, e.event_id))
        piece_rows.append({"piece_id": piece.piece_id, "group_id": item.group_id,
                           "source_status": targets.source_status, "event_count": len(targets.events),
                           "no_chord_span_count": targets.no_chord_span_count,
                           "trailing_unannotated_span_count": targets.trailing_unannotated_span_count,
                           "target_fingerprint": fingerprint([asdict(e) for e in ordered_events])})
        for event in ordered_events:
            index, alignment_reason = align_event_start(event.start_qn, piece.beats)
            for view_id in VIEW_IDS:
                row = {"piece_id": piece.piece_id, "group_id": item.group_id, "event_id": event.event_id,
                       "start_qn": asdict(event.start_qn), "beat_id": None if index is None else piece.beats[index].beat_id,
                       "alignment_reason": alignment_reason}
                for task in ("root", "quality"):
                    gold = event.root_pc if task == "root" else (
                        quality_gold_class_id(event.quality) if event.quality_available else None)
                    label_reason = getattr(event, f"{task}_reason")
                    if getattr(event, f"{task}_available") and gold is None:
                        label_reason = "unsupported"
                    row[task] = {"gold": gold, "prediction": None if index is None else predictions[view_id][task][index],
                                 "reason": label_reason or alignment_reason,
                                 "label_reason": label_reason,
                                 "source_label": event.root_pc if task == "root" else event.quality}
                per_view_rows[view_id].append(row)
    if model_state_fingerprint(model) != initial_state:
        raise ValueError("pop909_frozen.model_mutated_during_evaluation")
    piece_rows.sort(key=lambda row: row["piece_id"])
    views = {}
    for view_id in VIEW_IDS:
        rows = sorted(per_view_rows[view_id], key=lambda row: (row["piece_id"], row["event_id"]))
        bindings = sorted(per_view_bindings[view_id], key=lambda row: row["piece_id"])
        views[view_id] = {"role": "primary" if view_id == "native" else "secondary",
                          "input_bindings": bindings, "view_fingerprint": fingerprint(bindings),
                          **_summarize(rows, piece_rows), "event_observations": rows}
    payload = {
        "schema_version": REPORT_SCHEMA, "protocol": evaluation_protocol(),
        "split": split, "provenance": {"kind": "caller_declared_synthetic_fixture", "scientific_quality_evidence": False},
        "access_observations": {"synthetic_piece_count": len(piece_rows), "target_reader_calls": len(piece_rows),
                                "real_data_execution_enabled": False, "test_inputs_accessed": False,
                                "test_targets_accessed": False, "test_metrics_accessed": False},
        "model": {"contract": model_contract, "state_fingerprint": initial_state,
                  "checkpoint_evidence": checkpoint_evidence},
        "pieces": piece_rows, "views": views,
    }
    return {**payload, "report_fingerprint": fingerprint(payload)}


def evaluate_from_readers(*, split: str, checkpoint_reader: Callable, source_reader: Callable) -> dict:
    """Future execution boundary: fail before even calling either reader."""
    guard_execution(split)
    loaded = checkpoint_reader()
    return evaluate_frozen(loaded.model, source_reader(), split=split, checkpoint_evidence=loaded.evidence)
