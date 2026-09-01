"""Deterministic Phase 9E-B5C training primitives and bounded trainer.

Prediction is intentionally completed before an expanded B3 sidecar is
aligned.  The module contains an explicit TEST lock and only exposes TRAIN and
VALIDATION execution paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import copy
from pathlib import Path
import platform
import random
import sys
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CORRECTED_MODEL_ID,
    CORRECTED_MODEL_SCHEMA,
    CorrectedAnalysisGNNModel,
    CorrectedModelOutput,
    corrected_model_contract,
    corrected_parameter_inventory,
    corrected_routing_contract,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    CORRECTED_V2_METRIC_ID,
    PAPER_DEFINED_JOINT_COMPONENTS,
    PAPER_TEXT_COMPATIBILITY_METRIC_ID,
    TASK_BY_ID,
    get_vocabulary,
    project_quality_for_analysisgnn,
)
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SPLIT_ASSIGNMENTS_SHA256,
    CORRECTED_TEST_LOCK_FINGERPRINT,
    DEFERRED_HEADS,
    GROUP_WEIGHTS,
    PRIMARY_HEADS,
    aggregate_corrected_losses,
    class_weight_contract,
    component_balanced_record_draw,
    corrected_loss_contract,
    corrected_metric_contract,
    corrected_optimizer_envelope,
    masked_weighted_cross_entropy,
    validate_class_weight_payload,
)
from music_critic.experiments.analysisgnn.transposition import (
    select_record_shift,
    transpose_raw_graph_view,
    transform_semantic_value,
)
from music_critic.models.heads import TaskPrediction


CORRECTED_TRAINING_SCHEMA = "CorrectedAnalysisGNNTraining@1.0.0"
CORRECTED_CHECKPOINT_SCHEMA = "CorrectedAnalysisGNNCheckpoint@1.0.0"
CORRECTED_ALIGNMENT_SCHEMA = "CorrectedAnalysisGNNTargetAlignment@1.0.0"
CORRECTED_RUNTIME_SCHEMA = "CorrectedAnalysisGNNRuntimeConfig@1.0.0"
ACTIVE_HEADS = (*PRIMARY_HEADS, *AUXILIARY_HEADS)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_B5B_FIXTURE = _REPO_ROOT / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json"


class CorrectedTrainingError(ValueError):
    """Stable structured training failure."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.message = message


def require_non_test_split(split: str) -> Literal["train", "validation"]:
    normalized = split.casefold()
    if normalized == "test":
        raise CorrectedTrainingError(
            "analysisgnn.corrected.test_lock",
            "TEST loaders, targets, losses, metrics, and checkpoint selection are disabled",
        )
    if normalized not in {"train", "validation"}:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.split_invalid", split
        )
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CorrectedRuntimeConfig:
    profile_id: str
    seed: int = 17
    device: str = "cpu"
    batch_size: int = 1
    applied_update_budget: int = 2
    validation_interval: int = 100
    mixed_precision: str = "fp32_baseline"
    test_enabled: bool = False
    early_stopping: bool = False
    schema_version: str = CORRECTED_RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        if self.profile_id not in {
            CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
            CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
        }:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.profile_invalid", self.profile_id
            )
        if self.test_enabled:
            require_non_test_split("test")
        if self.seed != 17 or self.batch_size not in {1, 2}:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.runtime_contract_invalid",
                "seed must be 17 and batch size must be one or two",
            )
        if self.device not in {"cpu", "cuda"}:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.device_invalid", self.device
            )
        if self.mixed_precision != "fp32_baseline" or self.early_stopping:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.optimizer_envelope_changed",
                "B5B freezes FP32 baseline and disables early stopping",
            )
        if self.applied_update_budget <= 0 or self.validation_interval <= 0:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.budget_invalid", "budgets must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        body = asdict(self)
        body["transposition_enabled"] = (
            self.profile_id == CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID
        )
        body["validation_transposition"] = "identity"
        body["test_loader_created"] = False
        body["rng_domains"] = {
            "model_initialization_torch_seed": self.seed,
            "dropout_torch_seed": self.seed * 1000 + 1,
            "loader_worker_seed_base": self.seed * 1000 + 2,
            "record_sampling": "sha256_B5B_component_sampler_domain",
            "transposition": "sha256_B5A_record_epoch_profile_domain",
            "loader_workers": 0,
        }
        body["fingerprint"] = fingerprint(body)
        return body


def resolved_optimizer_contract(
    *, batch_size: int
) -> dict[str, object]:
    if batch_size not in {1, 2}:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.batch_size_invalid", str(batch_size)
        )
    frozen = corrected_optimizer_envelope()
    payload: dict[str, object] = {
        "source_fingerprint": frozen["fingerprint"],
        "source_contract": frozen["source_contract"],
        "model_identity": CORRECTED_MODEL_ID,
        "model_implementation_status": "implemented",
        "optimizer": frozen["optimizer"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "warmup_applied_updates": frozen["warmup_applied_updates"],
        "scheduler": frozen["scheduler"],
        "gradient_clip_norm": frozen["gradient_clip_norm"],
        "mixed_precision": frozen["mixed_precision"],
        "batch_size": batch_size,
        "batch_window_policy": "full_production_graph_no_windows",
        "applied_update_semantics": "successful_optimizer_steps_only",
        "persistent_overflow_policy": "retry_same_draw_then_fail_closed",
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


@dataclass(frozen=True, slots=True)
class FrozenClassWeights:
    values: Mapping[str, Tensor]
    supported: Mapping[str, Tensor]
    payload_fingerprint: str


def load_frozen_class_weights(
    path: str | os.PathLike[str] = _B5B_FIXTURE,
) -> FrozenClassWeights:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = fixture["class_weight_payload"]
    validate_class_weight_payload(payload)
    values: dict[str, Tensor] = {}
    supported: dict[str, Tensor] = {}
    for head in payload["heads"]:
        task_id = head["task_id"]
        if task_id in DEFERRED_HEADS:
            continue
        rows = head["classes"]
        supported[task_id] = torch.tensor(
            [row["weight"] is not None for row in rows], dtype=torch.bool
        )
        values[task_id] = torch.tensor(
            [1.0 if row["weight"] is None else row["weight"] for row in rows],
            dtype=torch.float32,
        )
    if tuple(values) != ACTIVE_HEADS:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.class_weight_inventory_changed",
            "B5B class weights do not match the 18 active heads",
        )
    return FrozenClassWeights(values, supported, payload["fingerprint"])


@dataclass(frozen=True, slots=True)
class AlignmentDiagnostic:
    task_id: str
    sample_index: int
    entity_id: str
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class AlignedHeadTargets:
    task_id: str
    candidate_indices: Tensor
    values: Tensor
    valid_mask: Tensor
    sample_indices: Tensor
    entity_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    masked_row_count: int
    alignment_failure_count: int

    def to(self, device: torch.device | str) -> "AlignedHeadTargets":
        return AlignedHeadTargets(
            task_id=self.task_id,
            candidate_indices=self.candidate_indices.to(device),
            values=self.values.to(device),
            valid_mask=self.valid_mask.to(device),
            sample_indices=self.sample_indices.to(device),
            entity_ids=self.entity_ids,
            record_ids=self.record_ids,
            component_ids=self.component_ids,
            masked_row_count=self.masked_row_count,
            alignment_failure_count=self.alignment_failure_count,
        )


@dataclass(frozen=True, slots=True)
class CorrectedTargetAlignment:
    schema_version: str
    heads: Mapping[str, AlignedHeadTargets]
    diagnostics: tuple[AlignmentDiagnostic, ...]
    target_sidecar_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CORRECTED_ALIGNMENT_SCHEMA or tuple(self.heads) != ACTIVE_HEADS:
            raise ValueError("analysisgnn.corrected.alignment_contract_invalid")


def _time_to_onset_id(value: object) -> str:
    if not isinstance(value, Mapping):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.onset_time_invalid", repr(value)
        )
    try:
        numerator = int(value["num"])
        denominator = int(value["den"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.onset_time_invalid", repr(value)
        ) from exc
    if denominator <= 0 or math.gcd(numerator, denominator) != 1:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.onset_time_noncanonical", repr(value)
        )
    return f"onset:{numerator}_{denominator}"


def _candidate_lookup(
    prediction: TaskPrediction, raw_graph_batch: object, node_type: str
) -> dict[tuple[int, str], int]:
    store = raw_graph_batch[node_type]  # type: ignore[index]
    entity_ids = store.entity_id
    lookup: dict[tuple[int, str], int] = {}
    for candidate_index, (sample, global_index) in enumerate(
        zip(
            prediction.sample_indices.detach().cpu().tolist(),
            prediction.global_entity_indices.detach().cpu().tolist(),
            strict=True,
        )
    ):
        pointer = int(store.ptr[sample])
        local_index = int(global_index) - pointer
        identifiers = entity_ids[sample]
        if not 0 <= local_index < len(identifiers):
            raise CorrectedTrainingError(
                "analysisgnn.corrected.candidate_metadata_mismatch",
                f"{node_type} sample={sample} index={global_index}",
            )
        lookup[(sample, identifiers[local_index])] = candidate_index
    return lookup


def align_target_sidecars_after_prediction(
    output: CorrectedModelOutput,
    raw_graph_batch: object,
    sidecars: Sequence[Mapping[str, object]],
    *,
    shifts: Sequence[int] | None = None,
) -> CorrectedTargetAlignment:
    """Exact B3 entity alignment, called only after raw-only logits exist."""

    if len(sidecars) == 0:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.empty_batch", "at least one sidecar is required"
        )
    shifts = tuple(0 for _ in sidecars) if shifts is None else tuple(shifts)
    if len(shifts) != len(sidecars):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.shift_batch_mismatch", "one shift per sidecar required"
        )
    predictions = {row.task_id: row for row in output.predictions}
    diagnostics: list[AlignmentDiagnostic] = []
    per_task: dict[str, dict[str, list[Any]]] = {
        task: defaultdict(list) for task in ACTIVE_HEADS
    }
    routes = {
        task: predictions[task].allowed_node_types[0] for task in ACTIVE_HEADS
    }
    lookups = {
        task: _candidate_lookup(predictions[task], raw_graph_batch, routes[task])
        for task in ACTIVE_HEADS
    }

    for sample_index, (sidecar, shift) in enumerate(zip(sidecars, shifts, strict=True)):
        if sidecar.get("schema_version") != "analysisgnn-source-native-target-sidecar-v1":
            raise CorrectedTrainingError(
                "analysisgnn.corrected.sidecar_schema_invalid", str(sidecar.get("schema_version"))
            )
        dialect = str(sidecar["dialect"])
        record_id = str(sidecar["record_id"])
        component = str(sidecar["source_component_id"])
        harmonic_to_beat = {
            str(row["source_entity_id"]): str(row["target_entity_id"])
            for row in sidecar["relations"]  # type: ignore[index]
            if row.get("relation") == "harmonic_event_to_beat"
        }
        for entity in sidecar["entities"]:  # type: ignore[index]
            entity_type = str(entity["entity_type"])
            targets = entity.get("targets", {})
            if not isinstance(targets, Mapping):
                continue
            canonical_entity_id = str(entity["canonical_entity_id"])
            if entity_type == "harmonic_event":
                raw_entity_id = harmonic_to_beat.get(canonical_entity_id)
            elif entity_type == "onset":
                raw_entity_id = _time_to_onset_id(entity.get("onset_qn"))
            elif entity_type == "note":
                raw_entity_id = canonical_entity_id
            else:
                continue
            for task_id, state in targets.items():
                if task_id not in per_task or TASK_BY_ID[task_id].prediction_level != entity_type:
                    continue
                values = per_task[task_id]
                values["entity_ids"].append(canonical_entity_id)
                values["record_ids"].append(record_id)
                values["component_ids"].append(component)
                values["sample_indices"].append(sample_index)
                values["masked"].append(bool(state.get("masked", True)))
                candidate = None if raw_entity_id is None else lookups[task_id].get((sample_index, raw_entity_id))
                available = state.get("available") is True and state.get("masked") is False
                label = state.get("canonical_value")
                class_id = 0
                if available and isinstance(label, str):
                    if shift:
                        label = transform_semantic_value(
                            task_id, label, shift_pc=shift, dialect=dialect, profile="corrected_v2"
                        )
                    vocabulary = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
                    try:
                        class_id = vocabulary.index(label)
                    except ValueError:
                        available = False
                        diagnostics.append(
                            AlignmentDiagnostic(task_id, sample_index, canonical_entity_id, "unknown_target", str(label))
                        )
                elif available:
                    available = False
                    diagnostics.append(
                        AlignmentDiagnostic(task_id, sample_index, canonical_entity_id, "missing_canonical_value", "available row lacks a canonical value")
                    )
                if candidate is None:
                    available = False
                    candidate = 0
                    diagnostics.append(
                        AlignmentDiagnostic(task_id, sample_index, canonical_entity_id, "exact_alignment_failed", f"raw_entity_id={raw_entity_id!r}")
                    )
                values["candidate_indices"].append(candidate)
                values["values"].append(class_id)
                values["valid"].append(available)

    heads: dict[str, AlignedHeadTargets] = {}
    for task_id in ACTIVE_HEADS:
        rows = per_task[task_id]
        row_count = len(rows["values"])
        valid = torch.tensor(rows["valid"], dtype=torch.bool)
        heads[task_id] = AlignedHeadTargets(
            task_id=task_id,
            candidate_indices=torch.tensor(rows["candidate_indices"], dtype=torch.long),
            values=torch.tensor(rows["values"], dtype=torch.long),
            valid_mask=valid,
            sample_indices=torch.tensor(rows["sample_indices"], dtype=torch.long),
            entity_ids=tuple(rows["entity_ids"]),
            record_ids=tuple(rows["record_ids"]),
            component_ids=tuple(rows["component_ids"]),
            masked_row_count=sum(rows["masked"]),
            alignment_failure_count=sum(
                row.task_id == task_id and row.category == "exact_alignment_failed"
                for row in diagnostics
            ),
        )
        if any(tensor.shape != (row_count,) for tensor in (
            heads[task_id].candidate_indices,
            heads[task_id].values,
            heads[task_id].valid_mask,
            heads[task_id].sample_indices,
        )):
            raise AssertionError("aligned target row tensors differ")
    return CorrectedTargetAlignment(
        schema_version=CORRECTED_ALIGNMENT_SCHEMA,
        heads=heads,
        diagnostics=tuple(diagnostics),
        target_sidecar_fingerprints=tuple(str(row["fingerprint"]) for row in sidecars),
    )


@dataclass(frozen=True, slots=True)
class HeadLossReport:
    task_id: str
    unweighted_ce: Tensor | None
    weighted_ce: Tensor | None
    valid_row_count: int
    masked_row_count: int
    unsupported_row_count: int
    active_class_count: int
    normalized_contribution: Tensor | None


@dataclass(frozen=True, slots=True)
class CorrectedLossReport:
    total: Tensor | None
    primary: Tensor | None
    auxiliary: Tensor | None
    heads: Mapping[str, HeadLossReport]
    zero_valid_heads: tuple[str, ...]
    fp32_boundary: bool


def corrected_supervised_loss(
    output: CorrectedModelOutput,
    alignment: CorrectedTargetAlignment,
    class_weights: FrozenClassWeights,
) -> CorrectedLossReport:
    predictions = output.logits
    weighted: dict[str, Tensor | None] = {}
    reports: dict[str, HeadLossReport] = {}
    for task_id in ACTIVE_HEADS:
        rows = alignment.heads[task_id].to(predictions[task_id].device)
        logits = predictions[task_id].index_select(0, rows.candidate_indices)
        supported_classes = class_weights.supported[task_id].to(logits.device)
        supported_rows = supported_classes.index_select(0, rows.values)
        loss_mask = rows.valid_mask & supported_rows
        unsupported_count = int((rows.valid_mask & ~supported_rows).sum().item())
        with torch.amp.autocast(logits.device.type, enabled=False):
            logits = logits.float()
            unweighted = masked_weighted_cross_entropy(
                logits, rows.values, rows.valid_mask
            )
            value = masked_weighted_cross_entropy(
                logits,
                rows.values,
                loss_mask,
                class_weights=class_weights.values[task_id].to(logits.device),
            )
        weighted[task_id] = value
        reports[task_id] = HeadLossReport(
            task_id=task_id,
            unweighted_ce=unweighted,
            weighted_ce=value,
            valid_row_count=int(loss_mask.sum().item()),
            masked_row_count=int((~rows.valid_mask).sum().item()),
            unsupported_row_count=unsupported_count,
            active_class_count=int(supported_classes.sum().item()),
            normalized_contribution=None,
        )
    aggregated = aggregate_corrected_losses(weighted)
    contributions: dict[str, Tensor | None] = {}
    primary_count = sum(weighted[row] is not None for row in PRIMARY_HEADS)
    auxiliary_count = sum(weighted[row] is not None for row in AUXILIARY_HEADS)
    for task_id in ACTIVE_HEADS:
        value = weighted[task_id]
        denominator = primary_count if task_id in PRIMARY_HEADS else auxiliary_count
        scale = 1.0 if task_id in PRIMARY_HEADS else GROUP_WEIGHTS["auxiliary"]
        contributions[task_id] = None if value is None else value * scale / denominator
        reports[task_id] = replace(
            reports[task_id],
            normalized_contribution=contributions[task_id],
        )
    tensors = [row for row in weighted.values() if row is not None]
    tensors.extend(value for value in (aggregated.total, aggregated.primary, aggregated.auxiliary) if value is not None)
    return CorrectedLossReport(
        total=aggregated.total,
        primary=aggregated.primary,
        auxiliary=aggregated.auxiliary,
        heads=reports,
        zero_valid_heads=aggregated.zero_valid_heads,
        fp32_boundary=all(row.dtype == torch.float32 for row in tensors),
    )


def gradient_norm(model: nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().float().pow(2).sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def transpose_raw_graph_batch(raw_graph_batch: object, shifts: Sequence[int]) -> object:
    """Create detached B5A views from a production-collated raw batch."""

    from torch_geometric.data import Batch
    from music_critic.graph import validate_raw_graph_batch

    graphs = raw_graph_batch.to_data_list()  # type: ignore[attr-defined]
    if len(graphs) != len(shifts):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.graph_shift_batch_mismatch",
            f"graphs={len(graphs)} shifts={len(shifts)}",
        )
    views = [
        transpose_raw_graph_view(graph, shift_pc=int(shift))
        for graph, shift in zip(graphs, shifts, strict=True)
    ]
    batch = Batch.from_data_list(views)
    validate_raw_graph_batch(batch, sample_count=len(views))
    return batch


def combine_single_record_raw_batches(batches: Sequence[object]) -> object:
    """Combine records already validated by the production collator."""

    from torch_geometric.data import Batch
    from music_critic.graph import validate_raw_graph_batch

    graphs = []
    for batch in batches:
        rows = batch.to_data_list()  # type: ignore[attr-defined]
        if len(rows) != 1:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.single_record_batch_required", str(len(rows))
            )
        graphs.append(rows[0])
    combined = Batch.from_data_list(graphs)
    validate_raw_graph_batch(combined, sample_count=len(graphs))
    return combined


def move_raw_graph_batch(raw_graph_batch: object, device: torch.device | str) -> object:
    """Move tensor stores while preserving tuple-valued production metadata."""

    from music_critic.graph import validate_raw_graph_batch

    target = torch.device(device)
    moved = copy.deepcopy(raw_graph_batch)
    for store in moved.stores:  # type: ignore[attr-defined]
        for key, value in tuple(store.items()):
            if isinstance(value, Tensor):
                store[key] = value.to(target)
    validate_raw_graph_batch(moved, sample_count=int(moved.num_graphs))
    return moved


@dataclass(frozen=True, slots=True)
class AppliedUpdateResult:
    applied: bool
    loss: CorrectedLossReport
    gradient_norm: float
    finite_logits: bool
    finite_gradients: bool
    scaler_scale_before: float
    scaler_scale_after: float


def attempt_applied_update(
    *,
    model: CorrectedAnalysisGNNModel,
    raw_graph_batch: object,
    sidecars: Sequence[Mapping[str, object]],
    shifts: Sequence[int],
    class_weights: FrozenClassWeights,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float = 1.0,
) -> AppliedUpdateResult:
    """Attempt one update; callers advance the sampler only when applied."""

    optimizer.zero_grad(set_to_none=True)
    output = model(raw_graph_batch)
    alignment = align_target_sidecars_after_prediction(
        output, raw_graph_batch, sidecars, shifts=shifts
    )
    loss = corrected_supervised_loss(output, alignment, class_weights)
    if loss.total is None or not loss.fp32_boundary:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.no_finite_supervision", repr(loss.zero_valid_heads)
        )
    finite_logits = all(bool(torch.isfinite(value).all()) for value in output.logits.values())
    if not finite_logits or not bool(torch.isfinite(loss.total)):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.non_finite_forward", "logits or loss are non-finite"
        )
    scale_before = float(scaler.get_scale())
    scaler.scale(loss.total).backward()
    scaler.unscale_(optimizer)
    finite_gradients = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    norm = gradient_norm(model)
    if not finite_gradients or not math.isfinite(norm):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.non_finite_gradient", str(norm)
        )
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    applied = scale_after >= scale_before
    if applied:
        scheduler.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return AppliedUpdateResult(
        applied=applied,
        loss=loss,
        gradient_norm=norm,
        finite_logits=finite_logits,
        finite_gradients=finite_gradients,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
    )


@dataclass(frozen=True, slots=True)
class ScheduledDraw:
    epoch: int
    draw_index: int
    component_id: str
    record_id: str
    shift_pc: int


class CorrectedComponentSampler:
    """Stateful wrapper around the frozen domain-separated B5B draw APIs."""

    draws_per_epoch = 1295

    def __init__(
        self,
        component_records: Mapping[str, Sequence[str]],
        valid_shifts_by_record: Mapping[str, Sequence[int]],
        *,
        profile_id: str,
        seed: int = 17,
        position: int = 0,
    ) -> None:
        self.component_records = {
            key: tuple(value) for key, value in component_records.items()
        }
        self.valid_shifts_by_record = {
            key: tuple(value) for key, value in valid_shifts_by_record.items()
        }
        self.profile_id = profile_id
        self.seed = seed
        self.position = position

    def peek(self, offset: int = 0) -> ScheduledDraw:
        if offset < 0:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.sampler_offset_invalid", str(offset)
            )
        epoch, draw_index = divmod(self.position + offset, self.draws_per_epoch)
        component, record = component_balanced_record_draw(
            self.component_records,
            seed=self.seed,
            epoch=epoch,
            draw_index=draw_index,
        )
        if self.profile_id == CORRECTED_NO_TRANSPOSITION_PROFILE_ID:
            shift = 0
        elif self.profile_id == CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID:
            valid = self.valid_shifts_by_record.get(record, ())
            if not valid:
                raise CorrectedTrainingError(
                    "analysisgnn.corrected.record_has_no_safe_shift", record
                )
            shift = select_record_shift(record, valid, seed=self.seed, epoch=epoch)
        else:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.profile_invalid", self.profile_id
            )
        return ScheduledDraw(epoch, draw_index, component, record, shift)

    def advance_after_applied_update(self, draws: int = 1) -> None:
        if draws <= 0:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.sampler_advance_invalid", str(draws)
            )
        self.position += draws

    def state_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "seed": self.seed,
            "profile_id": self.profile_id,
            "record_schedule_fingerprint": record_schedule_fingerprint(
                self.component_records, seed=self.seed, draw_count=max(self.position, 1)
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("seed") != self.seed or state.get("profile_id") != self.profile_id:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.sampler_resume_mismatch", repr(state)
            )
        self.position = int(state["position"])


def record_schedule_fingerprint(
    component_records: Mapping[str, Sequence[str]], *, seed: int, draw_count: int
) -> str:
    rows = []
    for position in range(draw_count):
        epoch, draw_index = divmod(position, 1295)
        component, record = component_balanced_record_draw(
            component_records, seed=seed, epoch=epoch, draw_index=draw_index
        )
        rows.append([epoch, draw_index, component, record])
    return fingerprint(rows)


def transposition_schedule_fingerprint(draws: Sequence[ScheduledDraw]) -> str:
    return fingerprint([[row.record_id, row.epoch, row.shift_pc] for row in draws])


def initialize_paired_models(seed: int = 17) -> tuple[CorrectedAnalysisGNNModel, CorrectedAnalysisGNNModel]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    first = CorrectedAnalysisGNNModel()
    state = {key: value.detach().clone() for key, value in first.state_dict().items()}
    second = CorrectedAnalysisGNNModel()
    second.load_state_dict(state)
    if model_state_fingerprint(first) != model_state_fingerprint(second):
        raise AssertionError("paired corrected models differ at initialization")
    return first, second


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def build_optimizer_scheduler(
    model: CorrectedAnalysisGNNModel, *, total_updates: int
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    envelope = corrected_optimizer_envelope()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(envelope["learning_rate"]),
        weight_decay=float(envelope["weight_decay"]),
    )
    warmup = int(envelope["warmup_applied_updates"])

    def multiplier(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def environment_fingerprint() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch", "torch-geometric"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def checkpoint_payload(
    *,
    model: CorrectedAnalysisGNNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: CorrectedComponentSampler,
    config: CorrectedRuntimeConfig,
    applied_update: int,
    best_primary_score: float | None,
    best_update: int | None,
    record_history: Sequence[str],
    shift_history: Sequence[int],
) -> dict[str, object]:
    return {
        "schema_version": CORRECTED_CHECKPOINT_SCHEMA,
        "resolved_config": config.to_dict(),
        "model_contract_fingerprint": corrected_model_contract(model)["fingerprint"],
        "model_state": model.state_dict(),
        "model_state_fingerprint": model_state_fingerprint(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "amp_scaler_state": scaler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "rng_state": capture_rng_state(),
        "applied_update": applied_update,
        "epoch": sampler.position // sampler.draws_per_epoch,
        "draw_position": sampler.position % sampler.draws_per_epoch,
        "sampler_position": sampler.position,
        "best_primary_score": best_primary_score,
        "best_update": best_update,
        "record_history": list(record_history),
        "shift_history": list(shift_history),
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "environment": environment_fingerprint(),
    }


def save_checkpoint(path: str | os.PathLike[str], payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: CorrectedAnalysisGNNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: CorrectedComponentSampler,
    config: CorrectedRuntimeConfig,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CORRECTED_CHECKPOINT_SCHEMA:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.checkpoint_schema_invalid", str(payload.get("schema_version"))
        )
    if payload.get("resolved_config") != config.to_dict():
        raise CorrectedTrainingError(
            "analysisgnn.corrected.resume_config_mismatch", "immutable resolved config changed"
        )
    model.load_state_dict(payload["model_state"])
    if model_state_fingerprint(model) != payload["model_state_fingerprint"]:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.checkpoint_model_hash_mismatch", str(path)
        )
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["amp_scaler_state"])
    sampler.load_state_dict(payload["sampler_state"])
    restore_rng_state(payload["rng_state"])
    return payload


def _confusion(predicted: Tensor, target: Tensor, classes: int) -> Tensor:
    values = target * classes + predicted
    return torch.bincount(values, minlength=classes * classes).reshape(classes, classes)


def per_head_validation_metrics(
    output: CorrectedModelOutput,
    alignment: CorrectedTargetAlignment,
    class_weights: FrozenClassWeights,
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for task_id in ACTIVE_HEADS:
        prediction = output.logits[task_id]
        rows = alignment.heads[task_id].to(prediction.device)
        valid = rows.valid_mask
        logits = prediction.index_select(0, rows.candidate_indices)[valid]
        targets = rows.values[valid]
        class_count = TASK_BY_ID[task_id].class_count
        if targets.numel() == 0:
            metrics[task_id] = {
                "masked_cross_entropy": None,
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1_observed_validation_classes": None,
                "per_class": [],
                "support": 0,
                "train_supported_class_count": int(class_weights.supported[task_id].sum()),
                "train_absent_class_count": int((~class_weights.supported[task_id]).sum()),
                "validation_only_class_count": 0,
                "record_support": 0,
                "component_support": 0,
            }
            continue
        predicted = logits.argmax(dim=1)
        matrix = _confusion(predicted, targets, class_count).cpu()
        true_support = matrix.sum(1)
        predicted_support = matrix.sum(0)
        diagonal = matrix.diag()
        precision = diagonal / predicted_support.clamp_min(1)
        recall = diagonal / true_support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)
        observed = true_support > 0
        train_supported = class_weights.supported[task_id]
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist()
        metrics[task_id] = {
            "masked_cross_entropy": float(F.cross_entropy(logits.float(), targets).detach()),
            "accuracy": float((predicted == targets).float().mean()),
            "balanced_accuracy": float(recall[observed].mean()),
            "macro_f1_observed_validation_classes": float(f1[observed].mean()),
            "per_class": [
                {
                    "class_id": index,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(true_support[index]),
                }
                for index in range(class_count)
            ],
            "support": int(targets.numel()),
            "train_supported_class_count": int(train_supported.sum()),
            "train_absent_class_count": int((~train_supported).sum()),
            "validation_only_class_count": int((observed & ~train_supported).sum()),
            "record_support": len({rows.record_ids[index] for index in valid_indices}),
            "component_support": len({rows.component_ids[index] for index in valid_indices}),
        }
    return metrics


def corrected_primary_macro_score(metrics: Mapping[str, Mapping[str, object]]) -> float | None:
    values = [
        metrics[task]["macro_f1_observed_validation_classes"]
        for task in PRIMARY_HEADS
        if metrics[task]["macro_f1_observed_validation_classes"] is not None
    ]
    return None if not values else sum(float(value) for value in values) / len(values)


def select_best_validation_checkpoint(
    *, current_score: float | None, best_score: float | None
) -> bool:
    return current_score is not None and (best_score is None or current_score > best_score)


def joint_metric_contract_evidence() -> dict[str, object]:
    return {
        "corrected": {
            "metric_id": CORRECTED_V2_METRIC_ID,
            "unit": "harmonic_event",
            "quality_classes": 17,
            "components": list(PAPER_DEFINED_JOINT_COMPONENTS),
        },
        "paper": {
            "metric_id": PAPER_TEXT_COMPATIBILITY_METRIC_ID,
            "unit": "note",
            "quality_classes": 15,
            "components": list(PAPER_DEFINED_JOINT_COMPONENTS),
        },
        "direct_roman_numeral": {
            "task_id": "roman_numeral",
            "classes": 184,
            "derived_joint_correctness_separate": True,
        },
    }


class CorrectedValidationAccumulator:
    """Exact confusion/loss and joint-metric accumulation across VALIDATION."""

    def __init__(
        self,
        class_weights: FrozenClassWeights,
        *,
        train_seen_tuples: set[tuple[str, ...]] | None = None,
    ) -> None:
        self.class_weights = class_weights
        self.confusions = {
            task: torch.zeros(
                (TASK_BY_ID[task].class_count, TASK_BY_ID[task].class_count),
                dtype=torch.long,
            )
            for task in ACTIVE_HEADS
        }
        self.ce_sums = {task: 0.0 for task in ACTIVE_HEADS}
        self.supports = {task: 0 for task in ACTIVE_HEADS}
        self.components = {task: set() for task in ACTIVE_HEADS}
        self.records = {task: set() for task in ACTIVE_HEADS}
        self.train_seen_tuples = train_seen_tuples or set()
        self.corrected_total = 0
        self.corrected_correct = 0
        self.seen_total = 0
        self.seen_correct = 0
        self.unseen_total = 0
        self.unseen_correct = 0
        self.unseen_tuples: set[tuple[str, ...]] = set()
        self.paper_total = 0
        self.paper_correct = 0

    def update(
        self,
        output: CorrectedModelOutput,
        alignment: CorrectedTargetAlignment,
        *,
        sidecars: Sequence[Mapping[str, object]] = (),
    ) -> None:
        joint_rows: dict[str, dict[tuple[int, str], tuple[int, int]]] = {}
        for task in ACTIVE_HEADS:
            logits = output.logits[task]
            rows = alignment.heads[task].to(logits.device)
            selected = logits.index_select(0, rows.candidate_indices)
            valid_indices = torch.nonzero(rows.valid_mask, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                if task in PAPER_DEFINED_JOINT_COMPONENTS:
                    joint_rows[task] = {}
                continue
            valid_logits = selected.index_select(0, valid_indices).float()
            valid_targets = rows.values.index_select(0, valid_indices)
            predicted = valid_logits.argmax(1)
            self.confusions[task] += _confusion(
                predicted, valid_targets, TASK_BY_ID[task].class_count
            ).cpu()
            self.ce_sums[task] += float(
                F.cross_entropy(valid_logits, valid_targets, reduction="sum").detach()
            )
            self.supports[task] += int(valid_targets.numel())
            cpu_indices = valid_indices.cpu().tolist()
            self.components[task].update(rows.component_ids[index] for index in cpu_indices)
            self.records[task].update(rows.record_ids[index] for index in cpu_indices)
            if task in PAPER_DEFINED_JOINT_COMPONENTS:
                joint_rows[task] = {
                    (int(rows.sample_indices[index]), rows.entity_ids[index]): (
                        int(predicted[offset]), int(valid_targets[offset])
                    )
                    for offset, index in enumerate(cpu_indices)
                }
        if not all(task in joint_rows for task in PAPER_DEFINED_JOINT_COMPONENTS):
            return
        keys = set.intersection(
            *(set(joint_rows[task]) for task in PAPER_DEFINED_JOINT_COMPONENTS)
        )
        corrected_by_key: dict[tuple[int, str], tuple[bool, tuple[str, ...], tuple[str, ...]]] = {}
        for key in keys:
            predicted_labels = tuple(
                get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[joint_rows[task][key][0]]
                for task in PAPER_DEFINED_JOINT_COMPONENTS
            )
            target_labels = tuple(
                get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[joint_rows[task][key][1]]
                for task in PAPER_DEFINED_JOINT_COMPONENTS
            )
            correct = predicted_labels == target_labels
            corrected_by_key[key] = (correct, predicted_labels, target_labels)
            self.corrected_total += 1
            self.corrected_correct += int(correct)
            if target_labels in self.train_seen_tuples:
                self.seen_total += 1
                self.seen_correct += int(correct)
            else:
                self.unseen_total += 1
                self.unseen_correct += int(correct)
                self.unseen_tuples.add(target_labels)
        for sample_index, sidecar in enumerate(sidecars):
            note_to_harmonic = {
                str(row["source_entity_id"]): str(row["target_entity_id"])
                for row in sidecar["relations"]  # type: ignore[index]
                if row.get("relation") == "note_to_harmonic_event"
            }
            for harmonic_id in note_to_harmonic.values():
                row = corrected_by_key.get((sample_index, harmonic_id))
                if row is None:
                    continue
                _corrected, predicted_labels, target_labels = row
                predicted_paper = list(predicted_labels)
                target_paper = list(target_labels)
                quality_index = PAPER_DEFINED_JOINT_COMPONENTS.index("quality")
                predicted_paper[quality_index] = project_quality_for_analysisgnn(
                    predicted_paper[quality_index]
                )
                target_paper[quality_index] = project_quality_for_analysisgnn(
                    target_paper[quality_index]
                )
                self.paper_total += 1
                self.paper_correct += int(tuple(predicted_paper) == tuple(target_paper))

    def finalize(self) -> dict[str, object]:
        heads: dict[str, dict[str, object]] = {}
        for task in ACTIVE_HEADS:
            matrix = self.confusions[task]
            true_support = matrix.sum(1)
            predicted_support = matrix.sum(0)
            diagonal = matrix.diag()
            precision = diagonal / predicted_support.clamp_min(1)
            recall = diagonal / true_support.clamp_min(1)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float32).eps)
            observed = true_support > 0
            supported = self.class_weights.supported[task]
            total = self.supports[task]
            heads[task] = {
                "masked_cross_entropy": None if total == 0 else self.ce_sums[task] / total,
                "accuracy": None if total == 0 else int(diagonal.sum()) / total,
                "balanced_accuracy": None if not observed.any() else float(recall[observed].mean()),
                "macro_f1_observed_validation_classes": None if not observed.any() else float(f1[observed].mean()),
                "per_class": [
                    {"class_id": index, "precision": float(precision[index]),
                     "recall": float(recall[index]), "f1": float(f1[index]),
                     "support": int(true_support[index])}
                    for index in range(matrix.shape[0])
                ],
                "support": total,
                "train_supported_class_count": int(supported.sum()),
                "train_absent_class_count": int((~supported).sum()),
                "validation_only_class_count": int((observed & ~supported).sum()),
                "record_support": len(self.records[task]),
                "component_support": len(self.components[task]),
            }
        primary_score = corrected_primary_macro_score(heads)

        def ratio(correct: int, total: int) -> float | None:
            return None if total == 0 else correct / total

        return {
            "per_head": heads,
            "corrected_primary_macro_score": primary_score,
            CORRECTED_V2_METRIC_ID: ratio(self.corrected_correct, self.corrected_total),
            "corrected_joint_support": self.corrected_total,
            "seen_tuple_joint_accuracy": ratio(self.seen_correct, self.seen_total),
            "seen_tuple_support": self.seen_total,
            "unseen_tuple_joint_accuracy": ratio(self.unseen_correct, self.unseen_total),
            "unseen_tuple_support": self.unseen_total,
            "unseen_tuple_count": len(self.unseen_tuples),
            PAPER_TEXT_COMPATIBILITY_METRIC_ID: ratio(self.paper_correct, self.paper_total),
            "paper_note_joint_support": self.paper_total,
            "direct_roman_numeral_accuracy": heads["roman_numeral"]["accuracy"],
            "direct_roman_numeral_macro_f1": heads["roman_numeral"]["macro_f1_observed_validation_classes"],
            "derived_joint_correctness_separate": True,
        }


def implementation_fingerprints(model: CorrectedAnalysisGNNModel) -> dict[str, str]:
    contracts = {
        "model_architecture": corrected_model_contract(model),
        "parameter_inventory": corrected_parameter_inventory(model),
        "active_head_inventory": [asdict(row) for row in model.task_specs],
        "routing_contract": corrected_routing_contract(),
        "loss_implementation": corrected_loss_contract(),
        "class_weight_contract": class_weight_contract(),
        "optimizer_envelope": resolved_optimizer_contract(batch_size=1),
        "metric_implementation": corrected_metric_contract(),
        "joint_metric_implementation": joint_metric_contract_evidence(),
        "test_lock": CORRECTED_TEST_LOCK_FINGERPRINT,
    }
    return {key: fingerprint(value) for key, value in contracts.items()}


def build_source_free_fixture() -> tuple[object, dict[str, object]]:
    """Build one tiny production-format raw graph and synthetic all-head sidecar."""

    from music_critic.data import (
        SCHEMA_VERSION,
        CanonicalBar,
        CanonicalBeat,
        CanonicalNote,
        CanonicalPiece,
        CanonicalTrack,
        KeySignatureEvent,
        MeterEvent,
        PieceMetadata,
        ProvenanceRecord,
        RationalTime,
        TempoEvent,
    )
    from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample

    zero, one, two = RationalTime(0), RationalTime(1), RationalTime(2)
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION,
        piece_id="piece:phase9eb5c-source-free",
        dataset_name="phase9eb5c-source-free",
        source_group_id="component:phase9eb5c-source-free",
        split="train",
        source_path=None,
        source_resolution=None,
        duration_qn=two,
        metadata=PieceMetadata(
            source_format="synthetic", title="Phase 9E-B5C fixture",
            creators=None, collection=None, movement_title=None,
            movement_number=None, genres=None, copyright=None, language=None,
        ),
        tracks=(CanonicalTrack(
            track_id="track:fixture", source_track_index=0, name="fixture",
            instrument_name=None, program=0, channel=0, is_percussion=False,
            provenance_id=None,
        ),),
        notes=(
            CanonicalNote(
                note_id="note:fixture-0", track_id="track:fixture", pitch=60,
                onset_qn=zero, duration_qn=one, velocity=64, channel=0, program=0,
                is_percussion=False, is_grace=False, spelling_step="C",
                spelling_alter=0, staff=None, voice=1, articulations=None,
                dynamic=None, source_onset_ticks=None, source_duration_ticks=None,
                source_onset_seconds=None, source_duration_seconds=None,
                provenance_id=None,
            ),
            CanonicalNote(
                note_id="note:fixture-1", track_id="track:fixture", pitch=64,
                onset_qn=one, duration_qn=one, velocity=64, channel=0, program=0,
                is_percussion=False, is_grace=False, spelling_step="E",
                spelling_alter=0, staff=None, voice=1, articulations=None,
                dynamic=None, source_onset_ticks=None, source_duration_ticks=None,
                source_onset_seconds=None, source_duration_seconds=None,
                provenance_id=None,
            ),
        ),
        bars=(CanonicalBar(
            bar_id="bar:fixture-0", index=0, start_qn=zero, duration_qn=two,
            meter_event_id="meter:fixture-0", metric_offset_qn=zero,
            is_pickup=False, is_incomplete=True, display_number="1",
            provenance_id=None,
        ),),
        beats=(
            CanonicalBeat(
                beat_id="beat:fixture-0", bar_id="bar:fixture-0",
                meter_event_id="meter:fixture-0", index_in_bar=0, start_qn=zero,
                duration_qn=one, position_in_bar_qn=zero, is_downbeat=True,
                strength=1.0, provenance_id=None,
            ),
            CanonicalBeat(
                beat_id="beat:fixture-1", bar_id="bar:fixture-0",
                meter_event_id="meter:fixture-0", index_in_bar=1, start_qn=one,
                duration_qn=one, position_in_bar_qn=one, is_downbeat=False,
                strength=0.5, provenance_id=None,
            ),
        ),
        tempo_events=(TempoEvent(
            tempo_event_id="tempo:fixture-0", onset_qn=zero,
            microseconds_per_quarter=500_000, provenance_id=None,
        ),),
        meter_events=(MeterEvent(
            meter_event_id="meter:fixture-0", onset_qn=zero, numerator=4,
            denominator=4, provenance_id=None,
        ),),
        key_signature_events=(KeySignatureEvent(
            key_signature_event_id="keysig:fixture-0", onset_qn=zero, fifths=0,
            mode="major", raw_value=None, provenance_id=None,
        ),),
        annotations=(), targets=(), provenance=(ProvenanceRecord(
            provenance_id="prov:phase9eb5c-source-free", kind="synthetic",
            source="Phase 9E-B5C source-free fixture", record_id=None, uri=None,
            version="1", checksum_sha256=None, created_at=None, parents=(),
            details=(),
        ),), quality_flags=(),
    )
    sample = prepare_multisource_sample(piece)
    batch = collate_multisource_samples((sample,))
    weights = load_frozen_class_weights()
    labels = {
        task: get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[
            int(torch.nonzero(weights.supported[task], as_tuple=False)[0])
        ]
        for task in ACTIVE_HEADS
    }

    def state(task: str, entity_id: str) -> dict[str, object]:
        return {
            "available": True, "masked": False, "missing_reason": None,
            "source_value": labels[task], "canonical_value": labels[task],
            "source_entity_id": f"source:{entity_id}",
            "canonical_entity_id": entity_id,
            "provenance": {"source_free_fixture": True},
        }

    harmonic_id, onset_id, note_id = (
        "harmonic-event:fixture-0", "onset-sidecar:fixture-0", "note:fixture-0"
    )
    by_level = {
        level: [task for task in ACTIVE_HEADS if TASK_BY_ID[task].prediction_level == level]
        for level in ("harmonic_event", "onset", "note")
    }
    semantic: dict[str, object] = {
        "schema_version": "analysisgnn-source-native-target-sidecar-v1",
        "record_id": "dlc:source-free:fixture",
        "dialect": "dlc",
        "source_component_id": "component:phase9eb5c-source-free",
        "entities": [
            {"canonical_entity_id": harmonic_id, "entity_type": "harmonic_event",
             "onset_qn": {"num": 0, "den": 1},
             "targets": {task: state(task, harmonic_id) for task in by_level["harmonic_event"]}},
            {"canonical_entity_id": onset_id, "entity_type": "onset",
             "onset_qn": {"num": 0, "den": 1},
             "targets": {task: state(task, onset_id) for task in by_level["onset"]}},
            {"canonical_entity_id": note_id, "entity_type": "note",
             "onset_qn": {"num": 0, "den": 1},
             "targets": {task: state(task, note_id) for task in by_level["note"]}},
        ],
        "relations": [{
            "relation": "harmonic_event_to_beat",
            "source_entity_id": harmonic_id,
            "target_entity_id": "beat:fixture-0",
        }],
    }
    semantic["fingerprint"] = fingerprint(semantic)
    return batch, semantic


@dataclass(frozen=True, slots=True)
class ProductionArtifactPaths:
    corpus_root: Path = _REPO_ROOT / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
    b2_source_inventory: Path = _REPO_ROOT / "outputs/phase9eb2/dilemmadata-coverage-audit-e607c934/source_inventory.jsonl"
    b3_root: Path = _REPO_ROOT / "outputs/phase9eb3/analysisgnn-multitask-contract-01290f5"
    b4_joint_tuple_counts: Path = _REPO_ROOT / "outputs/phase9eb4/analysisgnn-class-balance-671097b/joint_tuple_counts.jsonl"
    b5a_shift_eligibility: Path = _REPO_ROOT / "outputs/phase9eb5a/analysisgnn-transposition-a6a2796/record_shift_eligibility.jsonl"
    cache_root: Path = _REPO_ROOT / "outputs/phase9eb5c/production-cache"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def frozen_split_assignments(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, dict[str, object]]:
    source = paths.b3_root / "split_assignments.jsonl"
    if _sha256_file(source) != CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.split_assignment_hash_mismatch", str(source)
        )
    rows = _jsonl(source)
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    components = {
        split: len({row["source_component_id"] for row in rows if row["split"] == split})
        for split in counts
    }
    if counts != {"train": 1295, "validation": 162, "test": 162} or components != {
        "train": 1209, "validation": 147, "test": 151,
    }:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.split_count_mismatch", f"records={counts} components={components}"
        )
    return {str(row["record_id"]): row for row in rows}


def production_component_records(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in frozen_split_assignments(paths).values():
        if row["split"] == "train":
            grouped[str(row["source_component_id"])].append(str(row["record_id"]))
    result = {key: tuple(sorted(value)) for key, value in sorted(grouped.items())}
    sizes = {size: sum(len(value) == size for value in result.values()) for size in (1, 2)}
    if len(result) != 1209 or sizes != {1: 1123, 2: 86}:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.component_sampler_input_mismatch", repr(sizes)
        )
    return result


def production_valid_shifts(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, tuple[int, ...]]:
    train = {
        record_id for record_id, row in frozen_split_assignments(paths).items()
        if row["split"] == "train"
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in _jsonl(paths.b5a_shift_eligibility):
        record_id = str(row["record_id"])
        if record_id in train and row.get("corrected_valid") is True:
            grouped[record_id].append(int(row["shift_pc"]))
    result = {key: tuple(sorted(value)) for key, value in grouped.items()}
    if set(result) != train or any(0 not in shifts for shifts in result.values()):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.shift_eligibility_incomplete", "TRAIN records lack a closed identity shift"
        )
    return result


def train_seen_joint_tuples(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> set[tuple[str, ...]]:
    values = set()
    for row in _jsonl(paths.b4_joint_tuple_counts):
        if row.get("split") != "train" or row.get("mode") != "corrected_harmonic_event":
            continue
        target = row["tuple"]
        values.add(tuple(str(target[task]) for task in PAPER_DEFINED_JOINT_COMPONENTS))
    if not values:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.train_joint_tuple_inventory_empty", str(paths.b4_joint_tuple_counts)
        )
    return values


def minimal_real_train_coverage_records(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> tuple[str, ...]:
    assignments = frozen_split_assignments(paths)
    candidates = []
    for row in _jsonl(paths.b3_root / "target_sidecars.jsonl"):
        record_id = str(row["record_id"])
        if assignments[record_id]["split"] != "train":
            continue
        covered = {
            task for task in ACTIVE_HEADS
            if int(row["task_states"].get(task, {}).get("available", 0)) > 0
        }
        size = sum(int(value) for value in row["entity_counts"].values())
        candidates.append((size, record_id, covered))
    uncovered = set(ACTIVE_HEADS)
    chosen: list[str] = []
    while uncovered:
        ranked = sorted(
            candidates,
            key=lambda row: (-len(row[2] & uncovered), row[0], row[1]),
        )
        if not ranked or not (ranked[0][2] & uncovered):
            raise CorrectedTrainingError(
                "analysisgnn.corrected.real_train_coverage_incomplete", repr(sorted(uncovered))
            )
        _size, record_id, covered = ranked[0]
        chosen.append(record_id)
        uncovered -= covered
        candidates = [row for row in candidates if row[1] != record_id]
    return tuple(chosen)


def _resolve_selected_corpus_record(
    record_id: str,
    *,
    split: str,
    paths: ProductionArtifactPaths,
) -> object:
    """Rebuild one B2-bound record without corpus-wide discovery or TEST I/O."""

    required_split = require_non_test_split(split)
    assignment = frozen_split_assignments(paths).get(record_id)
    if assignment is None or assignment["split"] != required_split:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.record_split_mismatch", f"{record_id}:{assignment}"
        )
    # Import adapter internals only after the split gate.  They reconstruct the
    # exact public DilemmadataCorpusRecord binding for this selected record.
    import csv
    from music_critic.adapters import dilemmadata as adapter
    from music_critic.adapters.dilemmadata import (
        DilemmadataCorpusIdentity,
        DilemmadataCorpusRecord,
    )

    inventory = {
        str(row["record_id"]): row for row in _jsonl(paths.b2_source_inventory)
    }[record_id]
    paper = {
        str(row["record_id"]): row
        for row in _jsonl(paths.b3_root / "paper_candidate_records.jsonl")
    }[record_id]
    source_path = paths.corpus_root / str(inventory["annotation_path"])
    dialect = str(inventory["dialect"])
    parsed = adapter._parse_raw_file(source_path, dialect)
    source_fingerprints = inventory["source_fingerprints"]
    try:
        physical_annotation_sha256 = _sha256_file(source_path)
    except OSError as exc:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_source_fingerprint_mismatch", record_id
        ) from exc
    source_checks = {
        "physical_annotation_sha256": physical_annotation_sha256,
        "raw_projection_sha256": parsed.raw_projection_sha256,
        "grouping_sha256": parsed.grouping_fingerprint,
    }
    expected_source_checks = {
        key: str(source_fingerprints[key]) for key in source_checks
    }
    if source_checks != expected_source_checks:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_source_fingerprint_mismatch", record_id
        )
    if parsed.source_resolution != int(inventory["source_resolution"]):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_source_resolution_mismatch", record_id
        )
    suggested_split: str | None
    if dialect == "an_joint":
        suggested_split = Path(str(inventory["annotation_path"])).parent.name
    else:
        metadata_path = paths.corpus_root / "processing/DLC/distant_listening_corpus.metadata.tsv"
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            metadata = {
                ((row.get("corpus") or "").strip(), (row.get("piece") or "").strip()):
                ((row.get("split") or "").strip() or None)
                for row in csv.DictReader(handle, delimiter="\t")
            }
        suggested_split = metadata[
            (str(inventory["collection"]), str(inventory["source_piece_id"]))
        ]
    component = str(inventory["source_group_id"])
    score_relative = inventory.get("score_path")
    score_path = None if score_relative is None else paths.corpus_root / str(score_relative)
    if score_path is not None:
        try:
            score_sha256 = _sha256_file(score_path)
        except OSError as exc:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.b2_score_fingerprint_mismatch", record_id
            ) from exc
        if score_sha256 != source_fingerprints["score_sha256"]:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.b2_score_fingerprint_mismatch", record_id
            )
    elif source_fingerprints["score_sha256"] is not None:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_score_fingerprint_mismatch", record_id
        )
    record = DilemmadataCorpusRecord(
        record_id=record_id,
        piece_id=str(paper["piece_id"]),
        dialect=dialect,  # type: ignore[arg-type]
        path=source_path,
        relative_path=str(inventory["annotation_path"]),
        collection=str(inventory["collection"]),
        piece_name=str(inventory["source_piece_id"]),
        suggested_split=suggested_split,
        physical_source_sha256=str(source_fingerprints["physical_annotation_sha256"]),
        raw_projection_sha256=str(source_fingerprints["raw_projection_sha256"]),
        raw_equivalence_id=str(paper["raw_equivalence_id"]),
        grouping_fingerprint=str(source_fingerprints["grouping_sha256"]),
        source_group_id=component,
        lineage_group_id=component.replace("dilemmadata-component:", "dilemmadata-lineage:"),
        source_resolution=int(inventory["source_resolution"]),
        score_path=score_path,
        score_relative_path=None if score_relative is None else str(score_relative),
        score_sha256=source_fingerprints["score_sha256"],
        raw_issue_categories=parsed.categories,
        note_row_count=parsed.note_row_count,
        tie_continuation_row_count=parsed.tie_continuation_row_count,
        zero_duration_row_count=parsed.zero_duration_row_count,
        corpus_identity=DilemmadataCorpusIdentity(),
        record_binding_version=adapter.DILEMMADATA_RECORD_BINDING_VERSION,
        record_binding_sha256="",
    )
    return _bind_portable_production_record(
        record,
        expected_binding_sha256=str(source_fingerprints["record_binding_sha256"]),
        paths=paths,
        adapter=adapter,
    )


def _bind_portable_production_record(
    record: object,
    *,
    expected_binding_sha256: str,
    paths: ProductionArtifactPaths,
    adapter: object,
) -> object:
    """Validate the historical B2 seal while returning a valid local binding."""

    bound = adapter._bind_record(record)
    if not adapter.validate_dilemmadata_record_binding(bound):
        raise CorrectedTrainingError(
            "analysisgnn.corrected.local_record_binding_invalid", bound.record_id
        )
    if bound.record_binding_sha256 == expected_binding_sha256:
        return bound

    summary_path = paths.b2_source_inventory.parent / "audit_summary.json"
    try:
        snapshot = json.loads(summary_path.read_text(encoding="utf-8"))["snapshot"]
        historical_root = Path(str(snapshot["actual_path"]))
        identity = bound.corpus_identity
        snapshot_valid = (
            historical_root.is_absolute()
            and snapshot["content_fingerprint"] == identity.content_fingerprint
            and int(snapshot["file_count"]) == identity.installation_file_count
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_audit_snapshot_invalid", str(summary_path)
        ) from exc
    if not snapshot_valid:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_audit_snapshot_invalid", str(summary_path)
        )

    historical = replace(
        bound,
        path=historical_root / bound.relative_path,
        score_path=(
            None
            if bound.score_relative_path is None
            else historical_root / bound.score_relative_path
        ),
        record_binding_sha256="",
    )
    historical = adapter._bind_record(historical)
    if historical.record_binding_sha256 != expected_binding_sha256:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.b2_record_binding_mismatch", bound.record_id
        )
    return bound


def load_production_record(
    record_id: str,
    *,
    split: str,
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> tuple[object, dict[str, object]]:
    """Load/cache one canonical piece and expanded sidecar after a split gate."""

    required_split = require_non_test_split(split)
    assignment = frozen_split_assignments(paths).get(record_id)
    if assignment is None or assignment["split"] != required_split:
        raise CorrectedTrainingError(
            "analysisgnn.corrected.record_split_mismatch", record_id
        )
    from music_critic.data import dumps_piece, loads_piece
    from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample

    token = sha256(record_id.encode("utf-8")).hexdigest()[:24]
    record_root = paths.cache_root / token
    piece_path, sidecar_path, metadata_path = (
        record_root / "piece.json", record_root / "target_sidecar.json", record_root / "metadata.json"
    )
    if piece_path.is_file() and sidecar_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != {
            "record_id": record_id,
            "split": required_split,
            "piece_sha256": _sha256_file(piece_path),
            "sidecar_sha256": _sha256_file(sidecar_path),
        }:
            raise CorrectedTrainingError(
                "analysisgnn.corrected.production_cache_binding_invalid", record_id
            )
        piece = loads_piece(piece_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    else:
        from music_critic.adapters.dilemmadata import DilemmadataAccepted, convert_dilemmadata_record
        from music_critic.experiments.analysisgnn.multitask_contract import materialize_target_sidecar

        record = _resolve_selected_corpus_record(record_id, split=required_split, paths=paths)
        accepted = convert_dilemmadata_record(record)  # type: ignore[arg-type]
        if not isinstance(accepted, DilemmadataAccepted):
            raise CorrectedTrainingError(
                "analysisgnn.corrected.production_record_quarantined", f"{record_id}:{accepted}"
            )
        piece = accepted.piece
        sidecar = materialize_target_sidecar(accepted)
        record_root.mkdir(parents=True, exist_ok=True)
        temporary_piece = piece_path.with_suffix(".json.tmp")
        temporary_sidecar = sidecar_path.with_suffix(".json.tmp")
        temporary_piece.write_text(dumps_piece(piece), encoding="utf-8")
        temporary_sidecar.write_text(json.dumps(sidecar, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary_piece.replace(piece_path)
        temporary_sidecar.replace(sidecar_path)
        metadata = {
            "record_id": record_id,
            "split": required_split,
            "piece_sha256": _sha256_file(piece_path),
            "sidecar_sha256": _sha256_file(sidecar_path),
        }
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        temporary_metadata.replace(metadata_path)
    sample = prepare_multisource_sample(piece)
    batch = collate_multisource_samples((sample,))
    return batch, sidecar


__all__ = [
    "ACTIVE_HEADS",
    "CORRECTED_CHECKPOINT_SCHEMA",
    "CORRECTED_TRAINING_SCHEMA",
    "AlignedHeadTargets",
    "AlignmentDiagnostic",
    "AppliedUpdateResult",
    "CorrectedComponentSampler",
    "CorrectedLossReport",
    "CorrectedRuntimeConfig",
    "CorrectedTargetAlignment",
    "CorrectedTrainingError",
    "CorrectedValidationAccumulator",
    "FrozenClassWeights",
    "ProductionArtifactPaths",
    "ScheduledDraw",
    "align_target_sidecars_after_prediction",
    "attempt_applied_update",
    "build_optimizer_scheduler",
    "build_source_free_fixture",
    "capture_rng_state",
    "checkpoint_payload",
    "combine_single_record_raw_batches",
    "corrected_primary_macro_score",
    "corrected_supervised_loss",
    "environment_fingerprint",
    "gradient_norm",
    "implementation_fingerprints",
    "initialize_paired_models",
    "joint_metric_contract_evidence",
    "load_checkpoint",
    "load_frozen_class_weights",
    "load_production_record",
    "model_state_fingerprint",
    "per_head_validation_metrics",
    "minimal_real_train_coverage_records",
    "move_raw_graph_batch",
    "production_component_records",
    "production_valid_shifts",
    "record_schedule_fingerprint",
    "require_non_test_split",
    "resolved_optimizer_contract",
    "restore_rng_state",
    "save_checkpoint",
    "select_best_validation_checkpoint",
    "transposition_schedule_fingerprint",
    "transpose_raw_graph_batch",
    "train_seen_joint_tuples",
]
