"""Opt-in, raw-prediction-only linear residual fusion of harmonic beat heads.

This is a separate ablation, not the official AnalysisGNN attention fusion.
The frozen corrected model, its default prediction path and contracts are not
modified. There is no label, loss, checkpoint or dataset reader in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    CorrectedModelOutput,
    corrected_model_contract,
    corrected_task_head_specs,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    PRODUCTION_REGISTRY_ID,
    PRODUCTION_REGISTRY_SCHEMA_VERSION,
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.contracts import TASK_PREDICTION_CONTRACT_VERSION
from music_critic.models.heads import TaskPrediction
from music_critic.models.hierarchy import ContextualEncoderOutput
from music_critic.tasks import ENTITY_NODE_TYPE_TO_CODE


FUSION_MODEL_SCHEMA = "HarmonicFusionModel@1.0.0"
FUSION_CONTRACT_SCHEMA = "HarmonicLogitFusionContract@1.0.0"


class HarmonicLogitFusionError(ValueError):
    """A prediction, parameter or wrapper violates the isolated fusion contract."""

    def __init__(self, category: str, message: str = "") -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


def _fail(reason: str, detail: str = "") -> None:
    raise HarmonicLogitFusionError(f"harmonic_fusion.{reason}", detail)


def _specs():
    specs = corrected_task_head_specs()
    harmonic = tuple(row for row in specs if row.node_types == ("beat",))
    if len(specs) != 18 or len(harmonic) != 12:
        _fail("registry_inventory_mismatch")
    return specs, harmonic


def _validate_predictions(predictions: tuple[TaskPrediction, ...]) -> torch.device:
    specs, harmonic = _specs()
    if (not isinstance(predictions, tuple) or len(predictions) != len(specs)
            or any(not isinstance(row, TaskPrediction) for row in predictions)
            or tuple(row.task_id for row in predictions) != tuple(row.task_id for row in specs)):
        _fail("task_inventory_mismatch", "exact ordered 18-task tuple required")
    device: torch.device | None = None
    aligned: TaskPrediction | None = None
    harmonic_ids = {row.task_id for row in harmonic}
    row_fields = ("sample_indices", "global_entity_indices", "candidate_node_type_codes")
    count_fields = ("candidate_offsets_by_node_type", "candidate_counts_by_node_type")
    for prediction, spec in zip(predictions, specs, strict=True):
        task = spec.task_id
        logits = prediction.logits
        if (not isinstance(logits, Tensor) or logits.layout != torch.strided
                or logits.ndim != 2 or logits.shape[1] != spec.output_dim
                or logits.dtype != torch.float32):
            _fail("logits_shape_or_dtype", task)
        if device is None:
            device = logits.device
        if logits.device != device:
            _fail("prediction_device_mismatch", task)
        if not bool(torch.isfinite(logits).all()):
            _fail("non_finite_logits", task)
        if (prediction.contract_version != TASK_PREDICTION_CONTRACT_VERSION
                or prediction.allowed_node_types != spec.node_types
                or prediction.source_adapter != spec.source_adapter):
            _fail("head_route_mismatch", task)
        count = logits.shape[0]
        for name in (*row_fields, *count_fields):
            value = getattr(prediction, name)
            expected_shape = (count,) if name in row_fields else (len(MANDATORY_NODE_TYPES),)
            if (not isinstance(value, Tensor) or value.layout != torch.strided
                    or value.dtype != torch.long or value.shape != expected_shape
                    or value.device != device):
                _fail("candidate_metadata_invalid", f"{task}:{name}")
        code = ENTITY_NODE_TYPE_TO_CODE[spec.node_types[0]]
        if (bool((prediction.sample_indices < 0).any())
                or bool((prediction.global_entity_indices < 0).any())
                or bool((prediction.global_entity_indices >= count).any())
                or bool((prediction.candidate_node_type_codes != code).any())):
            _fail("candidate_identity_invalid", task)
        # A singleton raw head enumerates the whole node store. Indices may
        # be permuted together across heads, but cannot alias across samples.
        if torch.unique(prediction.global_entity_indices).numel() != count:
            _fail("duplicate_candidate", task)
        offsets = torch.full_like(prediction.candidate_offsets_by_node_type, -1)
        counts = torch.zeros_like(prediction.candidate_counts_by_node_type)
        offsets[code] = 0
        counts[code] = count
        if (not torch.equal(prediction.candidate_offsets_by_node_type, offsets)
                or not torch.equal(prediction.candidate_counts_by_node_type, counts)):
            _fail("candidate_mapping_invalid", task)
        if task in harmonic_ids:
            if aligned is None:
                aligned = prediction
            elif any(not torch.equal(getattr(prediction, name), getattr(aligned, name))
                     for name in (*row_fields, *count_fields)):
                _fail("harmonic_candidate_alignment_mismatch", task)
    assert device is not None
    return device


class HarmonicLogitFusion(nn.Module):
    """Twelve simultaneous bias-free corrections from the other eleven logits."""

    def __init__(self) -> None:
        super().__init__()
        specs, harmonic = _specs()
        self.task_order = tuple(row.task_id for row in specs)
        self.harmonic_task_ids = tuple(row.task_id for row in harmonic)
        self.class_counts = {row.task_id: row.output_dim for row in harmonic}
        total_width = sum(self.class_counts.values())
        # Explicit CPU/FP32 zeros avoid random initialization and caller defaults.
        self.weights = nn.ParameterDict({
            row.task_id: nn.Parameter(torch.zeros(
                (row.output_dim, total_width - row.output_dim),
                dtype=torch.float32, device="cpu",
            )) for row in harmonic
        })

    def _validate_weights(self, device: torch.device | None = None) -> None:
        specs, harmonic = _specs()
        expected_ids = tuple(row.task_id for row in harmonic)
        expected_counts = {row.task_id: row.output_dim for row in harmonic}
        if (self.task_order != tuple(row.task_id for row in specs)
                or self.harmonic_task_ids != expected_ids
                or self.class_counts != expected_counts
                or set(self.weights) != set(expected_ids)):
            _fail("fusion_inventory_mismatch")
        total_width = sum(expected_counts.values())
        parameter_ids = set()
        for task, width in expected_counts.items():
            weight = self.weights[task]
            if (not isinstance(weight, nn.Parameter) or weight.layout != torch.strided
                    or weight.shape != (width, total_width - width)
                    or weight.dtype != torch.float32):
                _fail("weight_shape_or_dtype", task)
            if id(weight) in parameter_ids:
                _fail("weight_alias", task)
            parameter_ids.add(id(weight))
            if device is None:
                device = weight.device
            if weight.device != device:
                _fail("weight_device_mismatch", task)
            if not bool(torch.isfinite(weight).all()):
                _fail("non_finite_weight", task)

    def forward(self, predictions: tuple[TaskPrediction, ...]) -> tuple[TaskPrediction, ...]:
        device = _validate_predictions(predictions)
        self._validate_weights(device)
        original = {row.task_id: row.logits for row in predictions}
        result = []
        with torch.autocast(device_type=device.type, enabled=False):
            for prediction in predictions:
                task = prediction.task_id
                if task not in self.weights:
                    result.append(prediction)
                    continue
                context = torch.cat(tuple(original[other] for other in self.harmonic_task_ids
                                          if other != task), dim=1)
                logits = original[task] + F.linear(context, self.weights[task])
                if not bool(torch.isfinite(logits).all()):
                    _fail("non_finite_fused_logits", task)
                result.append(replace(prediction, logits=logits))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class HarmonicFusionOutput:
    schema_version: str
    encoder: ContextualEncoderOutput
    predictions: tuple[TaskPrediction, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FUSION_MODEL_SCHEMA:
            _fail("output_schema_mismatch")
        if not isinstance(self.encoder, ContextualEncoderOutput):
            _fail("encoder_output_invalid")
        _validate_predictions(self.predictions)

    @property
    def logits(self) -> Mapping[str, Tensor]:
        return {row.task_id: row.logits for row in self.predictions}


class HarmonicFusionModel(nn.Module):
    """Compose an existing corrected model without changing its state or modes.

    Disabled prediction directly returns the base output. The wrapper retains
    normal ``base.*`` state nesting and never impersonates the old model schema.
    New fusion weights start on CPU; move the complete wrapper for other devices.
    """

    def __init__(self, base: CorrectedAnalysisGNNModel, *, enabled: bool = True) -> None:
        super().__init__()
        if not isinstance(base, CorrectedAnalysisGNNModel):
            _fail("base_model_type_invalid")
        if type(enabled) is not bool:
            _fail("enabled_bool_required")
        self.base = base
        self.enabled = enabled
        self.fusion = HarmonicLogitFusion() if enabled else None
        # Do not call self.train(): it would overwrite heterogeneous base modes.
        self.training = base.training
        if self.fusion is not None:
            self.fusion.train(base.training)

    def predict(self, raw_graph_batch: object, *, return_layers: bool = False
                ) -> CorrectedModelOutput | HarmonicFusionOutput:
        if (type(self.enabled) is not bool
                or self.enabled != (self.fusion is not None)):
            _fail("enabled_state_mismatch")
        output = self.base.predict(raw_graph_batch, return_layers=return_layers)
        if self.fusion is None:
            return output
        return HarmonicFusionOutput(
            schema_version=FUSION_MODEL_SCHEMA,
            encoder=output.encoder,
            predictions=self.fusion(output.predictions),
        )

    def forward(self, raw_graph_batch: object) -> CorrectedModelOutput | HarmonicFusionOutput:
        return self.predict(raw_graph_batch)


def fusion_model_contract(model: HarmonicFusionModel) -> dict[str, object]:
    """Fresh state-independent metadata, stable across eval and parameter freeze."""

    if not isinstance(model, HarmonicFusionModel):
        _fail("wrapper_type_invalid")
    if (type(model.enabled) is not bool
            or model.enabled != (model.fusion is not None)):
        _fail("enabled_state_mismatch")
    if not isinstance(model.base, CorrectedAnalysisGNNModel):
        _fail("base_model_type_invalid")
    if any(parameter.dtype != torch.float32 for parameter in model.base.parameters()):
        _fail("base_parameter_dtype_invalid")
    if model.fusion is not None:
        model.fusion._validate_weights()
    specs, harmonic = _specs()
    registry = {
        "registry_id": PRODUCTION_REGISTRY_ID,
        "registry_schema_version": PRODUCTION_REGISTRY_SCHEMA_VERSION,
        "tasks": [{
            "task_id": row.task_id,
            "node_type": row.node_types[0],
            "vocabulary_id": TASK_BY_ID[row.task_id].vocabulary_id,
            "class_count": row.output_dim,
            "class_labels": list(get_vocabulary(TASK_BY_ID[row.task_id].vocabulary_id).labels),
        } for row in specs],
    }
    registry["fingerprint"] = fingerprint(registry)
    base_count = sum(parameter.numel() for parameter in model.base.parameters())
    fusion_count = (0 if model.fusion is None
                    else sum(parameter.numel() for parameter in model.fusion.parameters()))
    payload = {
        "schema_version": FUSION_CONTRACT_SCHEMA,
        "model_schema": FUSION_MODEL_SCHEMA,
        "base_model_contract": corrected_model_contract(model.base),
        "enabled": model.enabled,
        "task_order": [row.task_id for row in specs],
        "harmonic_task_order": [row.task_id for row in harmonic],
        "harmonic_widths": {row.task_id: row.output_dim for row in harmonic},
        "class_registry": registry,
        "operation": "simultaneous_z_h_plus_W_h_concat_original_other_11_logits",
        "bias": False,
        "initialization": "zero_fp32_parameters_without_rng_consumption",
        "precision": "finite_fp32_logits_and_weights_autocast_disabled_for_fusion",
        "alignment": "exact_ordered_sample_global_entity_node_type_and_candidate_mapping",
        "input": "raw_predictions_only_no_targets_masks_confidence_or_gold_structure",
        "gradient": "differentiable_cross_task_no_detach",
        "other_heads": "same_prediction_objects",
        "parameter_counts": {"base": base_count, "fusion": fusion_count,
                             "total": base_count + fusion_count},
        "official_attention_fusion_reproduction": False,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


__all__ = [
    "FUSION_MODEL_SCHEMA", "FUSION_CONTRACT_SCHEMA", "HarmonicLogitFusionError",
    "HarmonicLogitFusion", "HarmonicFusionOutput", "HarmonicFusionModel",
    "fusion_model_contract",
]
