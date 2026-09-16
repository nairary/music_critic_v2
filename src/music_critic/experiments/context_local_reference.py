"""Frozen native C2 reference for a bounded TRAIN-local / VALIDATION comparison.

The dispatcher owns source, split, selection and protocol gates before loading
this adapter. Here only a SHA-bound model is loaded, with no resume state. Raw
TRAIN views can be built for later target joins; prediction is VALIDATION-only.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.data import CanonicalPiece
from music_critic.experiments.analysisgnn.corrected_model import CorrectedModelOutput
from music_critic.experiments.analysisgnn.corrected_training import move_raw_graph_batch
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.context_diagnostics import (
    FileBinding, fingerprint, guard_split, raw_candidates, require_sha,
)
from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
from music_critic.graph import (
    MANDATORY_NODE_TYPES, build_raw_graph, model_input_fingerprint, validate_raw_graph_batch,
)
from music_critic.tasks import ENTITY_NODE_TYPE_TO_CODE


TASK_LEVELS = {"local_key": "beat", "root": "beat", "quality": "beat",
               "primary_degree": "beat", "cadence": "onset", "metrical_strength": "note"}
ALLOWED_TASKS = tuple(TASK_LEVELS)
MAX_FORWARDS = 32
MAX_NOTES = 16384
MAX_CANDIDATES_PER_LEVEL = 16384


@dataclass(frozen=True)
class ReferencePiece:
    graph: HeteroData
    batch: Batch
    output: CorrectedModelOutput | None
    predictions: dict[str, dict[str, int]]
    logits: dict[str, dict[str, list[float]]]
    input_fingerprint: str
    evidence: dict


def ordered_classes(tasks: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if (not isinstance(tasks, tuple) or not tasks or len(set(tasks)) != len(tasks)
            or any(task not in TASK_LEVELS for task in tasks)):
        raise ValueError("context_local_reference.fixed_unique_tasks_required")
    result = {task: tuple(get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels) for task in tasks}
    if any(len(result[task]) != TASK_BY_ID[task].class_count for task in tasks):
        raise ValueError("context_local_reference.class_registry_mismatch")
    return result


def _all_candidate_starts(piece: CanonicalPiece, level: str) -> dict:
    if level == "beat":
        rows = [(beat.beat_id, beat.start_qn) for beat in piece.beats]
    elif level == "note":
        rows = [(note.note_id, note.onset_qn) for note in piece.notes]
    else:
        rows = [(f"onset:{time.num}_{time.den}", time)
                for time in sorted({note.onset_qn for note in piece.notes})]
    if len({identity for identity, _ in rows}) != len(rows):
        raise ValueError("context_local_reference.duplicate_raw_candidate")
    return dict(rows)


def build_reference_view(piece: CanonicalPiece, tasks: tuple[str, ...]) -> ReferencePiece:
    """Build target-free native graph/layout for TRAIN or VALIDATION, without weights."""
    guard_split(piece.split)
    classes = ordered_classes(tasks)
    raw = replace(piece, annotations=(), targets=())
    if len(raw.notes) > MAX_NOTES:
        raise ValueError("context_local_reference.note_budget")
    if len(raw.beats) > MAX_CANDIDATES_PER_LEVEL:
        raise ValueError("context_local_reference.candidate_budget")
    levels = tuple(dict.fromkeys(TASK_LEVELS[task] for task in tasks))
    all_starts = {level: _all_candidate_starts(raw, level) for level in levels}
    if any(len(rows) > MAX_CANDIDATES_PER_LEVEL for rows in all_starts.values()):
        raise ValueError("context_local_reference.candidate_budget")
    eligible = {level: tuple(candidate.candidate_id for candidate in raw_candidates(raw, level))
                for level in levels}
    graph = build_raw_graph(raw)
    if set(graph.node_types) != set(MANDATORY_NODE_TYPES):
        raise ValueError("context_local_reference.raw_only_graph_required")
    excluded = {}
    for level in levels:
        identities = tuple(graph[level].entity_id)
        if (len(set(identities)) != len(identities) or len(identities) != graph[level].num_nodes
                or set(identities) != set(all_starts[level])):
            raise ValueError("context_local_reference.graph_candidate_identity_mismatch")
        extra = sorted(set(identities) - set(eligible[level]))
        if any(all_starts[level][identity] != raw.duration_qn for identity in extra):
            raise ValueError("context_local_reference.unexplained_graph_only_candidate")
        excluded[level] = extra
    batch = Batch.from_data_list([graph])
    validate_raw_graph_batch(batch, sample_count=1)
    evidence = {
        "schema": "ContextNativeReferenceView@1.0.0", "piece_id": raw.piece_id,
        "split": raw.split, "view": "native_full_raw", "training_updates": 0,
        "ordered_classes": classes, "class_registry_sha256": fingerprint(classes),
        "candidate_ids_by_level": eligible, "excluded_endpoint_candidates": excluded,
        "endpoint_policy": "graph/output retain exact endpoint candidates; local maps use starts strictly before piece end",
        "annotations_or_targets_used": False,
    }
    return ReferencePiece(graph, batch, None, {}, {}, model_input_fingerprint(graph), evidence)


def _prediction_maps(view: ReferencePiece, output: CorrectedModelOutput,
                     tasks: tuple[str, ...]) -> tuple[dict, dict]:
    classes = ordered_classes(tasks)
    available = {}
    for prediction in output.predictions:
        if prediction.task_id in available:
            raise ValueError("context_local_reference.duplicate_prediction_task")
        available[prediction.task_id] = prediction
    predictions, logits = {}, {}
    for task in tasks:
        if task not in available:
            raise ValueError("context_local_reference.missing_prediction_task")
        prediction = available[task]
        level = TASK_LEVELS[task]
        identities = tuple(view.graph[level].entity_id)
        count = len(identities)
        code = ENTITY_NODE_TYPE_TO_CODE[level]
        indices = prediction.global_entity_indices.detach().cpu().tolist()
        expected_counts = [0] * len(MANDATORY_NODE_TYPES)
        expected_offsets = [-1] * len(MANDATORY_NODE_TYPES)
        expected_counts[code], expected_offsets[code] = count, 0
        if (prediction.allowed_node_types != (level,)
                or prediction.logits.dtype != torch.float32
                or tuple(prediction.logits.shape) != (count, len(classes[task]))
                or prediction.global_entity_indices.dtype != torch.long
                or sorted(indices) != list(range(count))
                or prediction.sample_indices.detach().cpu().tolist() != [0] * count
                or prediction.candidate_node_type_codes.detach().cpu().tolist() != [code] * count
                or prediction.candidate_counts_by_node_type.detach().cpu().tolist() != expected_counts
                or prediction.candidate_offsets_by_node_type.detach().cpu().tolist() != expected_offsets):
            raise ValueError("context_local_reference.prediction_candidate_or_class_mismatch")
        values = prediction.logits.detach().cpu()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("context_local_reference.nonfinite_logits")
        by_id = {identities[index]: values[row] for row, index in enumerate(indices)}
        candidate_ids = view.evidence["candidate_ids_by_level"][level]
        logits[task] = {identity: by_id[identity].tolist() for identity in candidate_ids}
        predictions[task] = {identity: int(by_id[identity].argmax()) for identity in candidate_ids}
    return predictions, logits


class FrozenC2Reference:
    """One native forward per VALIDATION piece, at most 32 durably announced attempts.

    The callback must persist its own receipts. A started attempt is announced
    before predict; only a returned predict plus successful CUDA sync completes
    it. Failed start-callback persistence also leaves the announced attempt
    unknown in memory; this does not prove that a physical forward began.
    A callback/prediction/sync/mapping failure makes this instance terminal.
    """

    def __init__(self, checkpoint: FileBinding, expected_model_contract_sha256: str,
                 expected_model_state_sha256: str, device: str = "cpu",
                 progress: Callable[[str, dict], None] | None = None):
        require_sha(checkpoint.sha256)
        require_sha(expected_model_contract_sha256)
        require_sha(expected_model_state_sha256)
        if type(checkpoint.size_bytes) is not int or not 0 < checkpoint.size_bytes <= 64 * 1024 * 1024:
            raise ValueError("context_local_reference.checkpoint_budget")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("context_local_reference.explicit_cpu_or_cuda0_required")
        if progress is not None and not callable(progress):
            raise ValueError("context_local_reference.progress_callback_required")
        loaded = load_model_checkpoint(checkpoint.path, expected_sha256=checkpoint.sha256,
                                       expected_size_bytes=checkpoint.size_bytes)
        if (loaded.evidence["model_contract_fingerprint"] != expected_model_contract_sha256
                or loaded.evidence["model_state_fingerprint"] != expected_model_state_sha256):
            raise ValueError("context_local_reference.expected_model_binding_mismatch")
        if device == "cuda:0" and not torch.cuda.is_available():
            raise ValueError("context_local_reference.cuda_unavailable")
        self._model = loaded.model.to(device)
        self.checkpoint_evidence = deepcopy(loaded.evidence)
        self.device = device
        self.progress = progress
        self.forwards_started = 0
        self.forwards_completed = 0
        self.failure_stage: str | None = None
        self._failed = False
        self._seen_piece_ids: set[str] = set()

    @property
    def forwards_unknown(self) -> int:
        return self.forwards_started - self.forwards_completed

    @property
    def forwards_completion_unknown(self) -> int:
        return self.forwards_unknown

    def _emit(self, event: str, piece_id: str, stage: str, **detail) -> None:
        if self.progress is not None:
            self.progress(event, {"piece_id": piece_id, "stage": stage,
                                 "forwards_started": self.forwards_started,
                                 "forwards_completed": self.forwards_completed,
                                 "forwards_completion_unknown": self.forwards_unknown,
                                 "applied_updates": 0, "forward_budget": MAX_FORWARDS, **detail})

    def predict(self, piece: CanonicalPiece, tasks: tuple[str, ...]) -> ReferencePiece:
        if piece.split != "validation":
            raise ValueError("context_local_reference.VALIDATION_only_no_TRAIN_or_TEST_forward")
        if self._failed:
            raise ValueError("context_local_reference.failed_instance_requires_new_attempt")
        if piece.piece_id in self._seen_piece_ids:
            raise ValueError("context_local_reference.one_forward_per_piece")
        if self.forwards_started >= MAX_FORWARDS:
            raise ValueError("context_local_reference.forward_budget")
        stage = "raw_view"
        try:
            view = build_reference_view(piece, tasks)
            stage = "frozen_model_check"
            if any(module.training for module in self._model.modules()) or any(
                    parameter.requires_grad for parameter in self._model.parameters()):
                raise ValueError("context_local_reference.frozen_eval_model_required")
            stage = "device_transfer"
            batch = move_raw_graph_batch(view.batch, self.device)
            self._seen_piece_ids.add(piece.piece_id)
            self.forwards_started += 1
            stage = "predict"
            self._emit("forward_started", piece.piece_id, stage)
            with torch.inference_mode():
                output = self._model.predict(batch)
                if self.device != "cpu":
                    stage = "cuda_synchronize"
                    torch.cuda.synchronize(0)
            self.forwards_completed += 1
            stage = "prediction_mapping"
            self._emit("forward_completed", piece.piece_id, stage)
            predictions, logits = _prediction_maps(view, output, tasks)
            evidence = {**view.evidence, "checkpoint": deepcopy(self.checkpoint_evidence),
                        "device": self.device, "forwards_started": self.forwards_started,
                        "forwards_completed": self.forwards_completed,
                        "forwards_completion_unknown": self.forwards_unknown}
            return replace(view, batch=batch, output=output, predictions=predictions,
                           logits=logits, evidence=evidence)
        except BaseException as exc:
            self._failed = True
            self.failure_stage = stage
            try:
                self._emit("reference_failed", piece.piece_id, stage,
                           error_type=type(exc).__name__, error=str(exc)[:1000])
            except BaseException:
                pass  # Preserve the original failure and all in-memory counters.
            raise
