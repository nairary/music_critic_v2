"""Corrected raw-only 18-head AnalysisGNN-derived V2 baseline.

This module deliberately reuses the production hierarchical encoder, onset
BiGRU decoder, and source-native MLP head implementation.  Target sidecars are
not accepted anywhere on the prediction path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    DEFERRED_HEADS,
    PRIMARY_HEADS,
    corrected_head_roles,
)
from music_critic.models.encoder import LocalHeterogeneousEncoder
from music_critic.models.heads import SourceNativeTaskHeads, TaskPrediction
from music_critic.models.hierarchy import (
    ContextualEncoderOutput,
    HierarchicalContextEncoder,
    extract_hierarchy_ownership,
)
from music_critic.models.hierarchy_contracts import HierarchicalBaselineConfig
from music_critic.models.onset_bigru import OnsetBiGRUDecoder


CORRECTED_MODEL_ID = "music-critic-v2-corrected-analysisgnn-18head-v1"
CORRECTED_MODEL_SCHEMA = "CorrectedAnalysisGNNModel@1.0.0"
CORRECTED_ROUTING_VERSION = "analysisgnn-corrected-entity-routing-v1"
CORRECTED_PARAMETER_INVENTORY_VERSION = "analysisgnn-corrected-parameter-inventory-v1"


class CorrectedAnalysisGNNModelError(ValueError):
    """Stable structured failure at the corrected-model boundary."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.message = message


@dataclass(frozen=True, slots=True)
class CorrectedAnalysisGNNConfig:
    hidden_dim: int = 128
    local_gnn_layers: int = 3
    transformer_layers: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    residual: bool = True

    def __post_init__(self) -> None:
        if (
            self.hidden_dim,
            self.local_gnn_layers,
            self.transformer_layers,
            self.attention_heads,
            self.ffn_multiplier,
            self.dropout,
            self.residual,
        ) != (128, 3, 2, 4, 4, 0.1, True):
            raise CorrectedAnalysisGNNModelError(
                "analysisgnn.corrected.fixed_architecture_required",
                "the Phase 9E-B5C architecture has no tunable model fields",
            )

    def hierarchy_config(self) -> HierarchicalBaselineConfig:
        return HierarchicalBaselineConfig(
            hidden_dim=self.hidden_dim,
            local_gnn_layers=self.local_gnn_layers,
            transformer_layers=self.transformer_layers,
            attention_heads=self.attention_heads,
            ffn_multiplier=self.ffn_multiplier,
            dropout=self.dropout,
            local_residual=self.residual,
        )


@dataclass(frozen=True, slots=True)
class CorrectedTaskHeadSpec:
    """Duck-typed contract consumed by ``SourceNativeTaskHeads``."""

    task_id: str
    source_adapter: str
    encoding_kind: Literal["closed_categorical_index"]
    output_dim: int
    node_types: tuple[Literal["note", "onset", "beat"], ...]
    supervision_regime: Literal["fully_supervised"] = "fully_supervised"


@dataclass(frozen=True, slots=True)
class CorrectedModelOutput:
    schema_version: str
    encoder: ContextualEncoderOutput
    predictions: tuple[TaskPrediction, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CORRECTED_MODEL_SCHEMA:
            raise ValueError("analysisgnn.corrected.output_schema_incompatible")
        if len(self.predictions) != 18 or any(
            row.logits.dtype != torch.float32 for row in self.predictions
        ):
            raise ValueError("analysisgnn.corrected.fp32_18_head_output_required")

    @property
    def logits(self) -> Mapping[str, Tensor]:
        return {row.task_id: row.logits for row in self.predictions}


_NODE_TYPE_BY_LEVEL = {
    "harmonic_event": "beat",
    "onset": "onset",
    "note": "note",
}


def corrected_task_head_specs() -> tuple[CorrectedTaskHeadSpec, ...]:
    """Return the fixed 18-head registry and its raw-entity routing."""

    roles = {row.task_id: row for row in corrected_head_roles()}
    specs = []
    for task_id in (*PRIMARY_HEADS, *AUXILIARY_HEADS):
        task = TASK_BY_ID[task_id]
        try:
            node_type = _NODE_TYPE_BY_LEVEL[task.prediction_level]
        except KeyError as exc:  # pragma: no cover - protects future registry drift
            raise CorrectedAnalysisGNNModelError(
                "analysisgnn.corrected.prediction_level_unsupported",
                f"{task_id} has level {task.prediction_level}",
            ) from exc
        if roles[task_id].role == "deferred":  # pragma: no cover
            raise AssertionError("deferred task entered active model specs")
        specs.append(
            CorrectedTaskHeadSpec(
                task_id=task_id,
                source_adapter="dilemmadata.analysisgnn.b3-sidecar",
                encoding_kind="closed_categorical_index",
                output_dim=task.class_count,
                node_types=(node_type,),  # type: ignore[arg-type]
            )
        )
    if len(specs) != 18 or len({row.task_id for row in specs}) != 18:
        raise CorrectedAnalysisGNNModelError(
            "analysisgnn.corrected.head_inventory_invalid",
            "exactly 18 unique active heads are required",
        )
    return tuple(specs)


class _LocalEncoderContainer(nn.Module):
    def __init__(self, config: CorrectedAnalysisGNNConfig) -> None:
        super().__init__()
        self.encoder = LocalHeterogeneousEncoder(
            hidden_dim=config.hidden_dim,
            gnn_layers=config.local_gnn_layers,
            dropout=config.dropout,
            residual=config.residual,
            use_message_passing=True,
        )


class CorrectedAnalysisGNNModel(nn.Module):
    """Music Critic V2 corrected AnalysisGNN-derived multi-task baseline."""

    def __init__(
        self,
        config: CorrectedAnalysisGNNConfig = CorrectedAnalysisGNNConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.task_specs = corrected_task_head_specs()
        self.local_baseline = _LocalEncoderContainer(config)
        self.context_encoder = HierarchicalContextEncoder(config.hierarchy_config())
        self.sequence_decoder = OnsetBiGRUDecoder(config.hidden_dim, config.dropout)
        self.task_heads = SourceNativeTaskHeads(
            self.task_specs,  # type: ignore[arg-type]
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
            force_float32=True,
        )
        if self.task_heads.node_type_embeddings:
            raise CorrectedAnalysisGNNModelError(
                "analysisgnn.corrected.single_entity_head_required",
                "active heads must not fuse multiple entity types",
            )

    def encode(
        self, raw_graph_batch: object, *, return_layers: bool = False
    ) -> ContextualEncoderOutput:
        ownership = extract_hierarchy_ownership(raw_graph_batch)
        local = self.local_baseline.encoder(
            raw_graph_batch, return_layers=return_layers
        )
        encoded = self.context_encoder._forward_with_extracted_ownership(
            local, ownership
        )
        return replace(
            encoded,
            fused=self.sequence_decoder(encoded.fused, raw_graph_batch),
        )

    def predict(
        self, raw_graph_batch: object, *, return_layers: bool = False
    ) -> CorrectedModelOutput:
        encoded = self.encode(raw_graph_batch, return_layers=return_layers)
        predictions = self.task_heads(encoded.fused)
        return CorrectedModelOutput(
            schema_version=CORRECTED_MODEL_SCHEMA,
            encoder=encoded,
            predictions=predictions,
        )

    def forward(self, raw_graph_batch: object) -> CorrectedModelOutput:
        return self.predict(raw_graph_batch)

    def request_logits(self, task_id: str, raw_graph_batch: object) -> Tensor:
        """Explicit single-head API with a stable deferred-head rejection."""

        if task_id in DEFERRED_HEADS:
            raise CorrectedAnalysisGNNModelError(
                "analysisgnn.corrected.deferred_head_logits_forbidden",
                f"{task_id} is registry metadata only",
            )
        output = self.predict(raw_graph_batch)
        try:
            return output.logits[task_id]
        except KeyError as exc:
            raise CorrectedAnalysisGNNModelError(
                "analysisgnn.corrected.unknown_head", task_id
            ) from exc


def corrected_model_contract(
    model: CorrectedAnalysisGNNModel | None = None,
) -> dict[str, object]:
    model = CorrectedAnalysisGNNModel() if model is None else model
    specs = [asdict(row) for row in model.task_specs]
    roles = [asdict(row) for row in corrected_head_roles()]
    payload: dict[str, object] = {
        "schema_version": CORRECTED_MODEL_SCHEMA,
        "model_id": CORRECTED_MODEL_ID,
        "config": asdict(model.config),
        "encoder_input": "production_raw_graph_only",
        "local_encoder": "music_critic.models.encoder.LocalHeterogeneousEncoder",
        "hierarchical_encoder": "music_critic.models.hierarchy.HierarchicalContextEncoder",
        "decoder": {
            "implementation": "music_critic.models.onset_bigru.OnsetBiGRUDecoder",
            "input_dim": 128,
            "output_dim": 128,
            "layers": 1,
            "bidirectional": True,
            "dropout": 0.1,
        },
        "head_mlp": ["Linear(128,128)", "GELU", "Dropout(0.1)", "Linear(128,C)"],
        "head_specs": specs,
        "registry_roles": roles,
        "active_head_count": 18,
        "primary_head_count": 8,
        "auxiliary_head_count": 10,
        "deferred_head_count": 2,
        "staff_included": False,
        "logit_fusion": False,
        "target_join": "post_prediction_only",
        "precision_boundary": "encoder_autocast_allowed_heads_logits_and_losses_fp32",
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def corrected_routing_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "version": CORRECTED_ROUTING_VERSION,
        "head_routes": {
            row.task_id: row.node_types[0] for row in corrected_task_head_specs()
        },
        "harmonic_event_alignment": "exact_harmonic_event_to_beat_relation",
        "onset_alignment": "exact_rational_onset_qn_to_onset:{num}_{den}",
        "note_alignment": "exact_canonical_note_id",
        "alignment_failure": "mask_row_and_emit_diagnostic",
        "heuristic_alignment": False,
        "target_aware_graph_construction": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def model_state_fingerprint(model_or_state: nn.Module | Mapping[str, Tensor]) -> str:
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    digest = sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def corrected_parameter_inventory(model: CorrectedAnalysisGNNModel) -> dict[str, object]:
    groups = {
        "encoder": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(("local_baseline.", "context_encoder."))
        ),
        "onset_bigru_decoder": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("sequence_decoder.")
        ),
    }
    head_counts: dict[str, int] = {}
    for index, spec in enumerate(model.task_specs):
        head_counts[spec.task_id] = sum(
            parameter.numel()
            for parameter in model.task_heads.heads[f"task_{index:02d}"].parameters()
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    payload: dict[str, object] = {
        "version": CORRECTED_PARAMETER_INVENTORY_VERSION,
        "groups": groups,
        "heads": head_counts,
        "primary_heads": sum(head_counts[row] for row in PRIMARY_HEADS),
        "auxiliary_heads": sum(head_counts[row] for row in AUXILIARY_HEADS),
        "trainable": trainable,
        "frozen": frozen,
        "total": trainable + frozen,
    }
    accounted = groups["encoder"] + groups["onset_bigru_decoder"] + sum(head_counts.values())
    if accounted != payload["total"] or frozen != 0:
        raise CorrectedAnalysisGNNModelError(
            "analysisgnn.corrected.parameter_inventory_mismatch",
            f"accounted={accounted} total={payload['total']} frozen={frozen}",
        )
    payload["fingerprint"] = fingerprint(payload)
    return payload


__all__ = [
    "CORRECTED_MODEL_ID",
    "CORRECTED_MODEL_SCHEMA",
    "CorrectedAnalysisGNNConfig",
    "CorrectedAnalysisGNNModel",
    "CorrectedAnalysisGNNModelError",
    "CorrectedModelOutput",
    "CorrectedTaskHeadSpec",
    "corrected_model_contract",
    "corrected_parameter_inventory",
    "corrected_routing_contract",
    "corrected_task_head_specs",
    "model_state_fingerprint",
]
