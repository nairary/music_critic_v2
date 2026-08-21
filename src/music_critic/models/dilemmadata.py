"""Phase 9B.2B candidate-first Dilemmadata hierarchical supervision."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
import json
import math
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from music_critic.models.encoder import LocalHeterogeneousEncoder
from music_critic.models.heads import (
    RoutingOperationCounts,
    SourceNativeTaskHeads,
    TaskPrediction,
    TaskSupervision,
    join_task_supervision,
    routing_operation_counts,
)
from music_critic.models.hierarchy import (
    ContextualEncoderOutput,
    HierarchicalContextEncoder,
    extract_hierarchy_ownership,
)
from music_critic.models.hierarchy_contracts import HierarchicalBaselineConfig
from music_critic.models.onset_bigru import (
    DilemmadataDecoderConfig,
    OnsetBiGRUDecoder,
    ONSET_BIGRU_DECODER_CONTRACT_VERSION,
)
from music_critic.tasks import (
    BatchTarget,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
    DILEMMADATA_TARGET_ENCODING_BY_TASK,
    MultiSourceBatch,
)


DILEMMADATA_MODEL_CONTRACT_VERSION = "1.2.0"
DILEMMADATA_HEAD_CONTRACT_VERSION = "1.0.0"
DILEMMADATA_LOSS_CONTRACT_VERSION = "1.0.0"
DILEMMADATA_FP32_HEAD_LOSS_BOUNDARY_VERSION = "1.0.0"
DILEMMADATA_ENCODER_TRANSFER_VERSION = "1.0.0"
DILEMMADATA_CLASS_WEIGHT_CONTRACT_VERSION = "1.0.0"

DILEMMADATA_ACTIVE_TASK_IDS = (
    "dilemmadata.an.chord.inversion",
    "dilemmadata.an.chord.quality",
    "dilemmadata.dlc.chord.inversion",
    "dilemmadata.dlc.chord.quality",
)
DILEMMADATA_PU_TASK_IDS = (
    "dilemmadata.an.chord.boundary",
    "dilemmadata.dlc.cadence",
    "dilemmadata.dlc.chord.boundary",
    "dilemmadata.dlc.phrase.boundary",
    "dilemmadata.dlc.section.boundary",
)
DILEMMADATA_OPEN_TASK_IDS = tuple(
    sorted(
        task_id
        for task_id, spec in DILEMMADATA_TARGET_ENCODING_BY_TASK.items()
        if spec.encoding_kind == "open_string_cpu"
    )
)
if len(DILEMMADATA_OPEN_TASK_IDS) != 13:
    raise RuntimeError("Phase 9B.2B requires exactly 13 deferred open tasks")


@dataclass(frozen=True, slots=True)
class DilemmadataHierarchicalConfig:
    hidden_dim: int = 128
    local_gnn_layers: int = 3
    transformer_layers: int = 2
    attention_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.1
    residual: bool = True
    task_hidden_dim: int | None = None
    decoder: DilemmadataDecoderConfig = field(
        default_factory=DilemmadataDecoderConfig
    )
    task_weights: tuple[tuple[str, float], ...] = tuple(
        (task_id, 1.0) for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    )

    def __post_init__(self) -> None:
        if isinstance(self.decoder, Mapping):
            object.__setattr__(
                self, "decoder", DilemmadataDecoderConfig(**self.decoder)
            )
        elif not isinstance(self.decoder, DilemmadataDecoderConfig):
            raise ValueError("dilemmadata.model.decoder_config_invalid")
        hierarchy = self.hierarchy_config()
        del hierarchy
        if self.task_hidden_dim is not None and (
            isinstance(self.task_hidden_dim, bool)
            or not isinstance(self.task_hidden_dim, int)
            or self.task_hidden_dim <= 0
        ):
            raise ValueError("dilemmadata.model.task_hidden_dim_invalid")
        task_ids = tuple(task_id for task_id, _ in self.task_weights)
        if task_ids != DILEMMADATA_ACTIVE_TASK_IDS:
            raise ValueError("dilemmadata.model.fixed_task_inventory_required")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
            for _, weight in self.task_weights
        ):
            raise ValueError("dilemmadata.model.task_weight_invalid")

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


class DilemmadataModelContractError(ValueError):
    """Stable failure for malformed or state-inconsistent model contracts."""


@dataclass(frozen=True, slots=True)
class DilemmadataTaskHeadSpec:
    task_id: str
    source_adapter: str
    encoding_kind: Literal["closed_categorical_index"]
    output_dim: int
    node_types: tuple[str, ...]
    supervision_regime: Literal["fully_supervised"] = "fully_supervised"

    def __post_init__(self) -> None:
        if self.task_id not in DILEMMADATA_ACTIVE_TASK_IDS:
            raise ValueError("dilemmadata.head.task_not_active")
        family = DILEMMADATA_SOURCE_FAMILY_BY_TASK[self.task_id].ontology_spec
        encoding = DILEMMADATA_TARGET_ENCODING_BY_TASK[self.task_id]
        if (
            self.source_adapter != family.source_adapter
            or self.encoding_kind != "closed_categorical_index"
            or encoding.supervision_regime != "fully_supervised"
            or self.output_dim != len(encoding.vocabulary or ())
            or self.node_types != family.alignment_policy.candidate_node_types
        ):
            raise ValueError("dilemmadata.head.registry_mismatch")


def dilemmadata_task_head_specs() -> tuple[DilemmadataTaskHeadSpec, ...]:
    return tuple(
        DilemmadataTaskHeadSpec(
            task_id=task_id,
            source_adapter=(
                DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
                .ontology_spec.source_adapter
            ),
            encoding_kind="closed_categorical_index",
            output_dim=len(
                DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary or ()
            ),
            node_types=(
                DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
                .ontology_spec.alignment_policy.candidate_node_types
            ),
        )
        for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    )


@dataclass(frozen=True, slots=True)
class DilemmadataSourceEntryTaskLoss:
    task_id: str
    fixed_weight: float
    expanded_row_count: int
    effective_source_entry_count: int
    entry_sample_indices: Tensor
    entry_source_indices: Tensor
    entry_row_counts: Tensor
    entry_mean_losses: Tensor
    mean_loss: Tensor

    def __post_init__(self) -> None:
        count = self.entry_mean_losses.shape[0]
        if (
            self.task_id not in DILEMMADATA_ACTIVE_TASK_IDS
            or count != self.effective_source_entry_count
            or self.expanded_row_count < count
            or self.mean_loss.ndim != 0
            or any(
                value.ndim != 1 or value.shape[0] != count
                for value in (
                    self.entry_sample_indices,
                    self.entry_source_indices,
                    self.entry_row_counts,
                    self.entry_mean_losses,
                )
            )
            or any(
                value.dtype != torch.long
                for value in (
                    self.entry_sample_indices,
                    self.entry_source_indices,
                    self.entry_row_counts,
                )
            )
        ):
            raise ValueError("dilemmadata.loss.source_entry_report_invalid")


@dataclass(frozen=True, slots=True)
class DilemmadataLossReport:
    contract_version: str
    reduction: str
    task_losses: tuple[DilemmadataSourceEntryTaskLoss, ...]
    total_loss: Tensor | None

    def __post_init__(self) -> None:
        if (
            self.contract_version != DILEMMADATA_LOSS_CONTRACT_VERSION
            or self.reduction
            != "candidate_rows_mean_per_source_entry_then_entries_mean_per_task_fixed_weight_sum"
            or (self.total_loss is None) != (not self.task_losses)
        ):
            raise ValueError("dilemmadata.loss.report_invalid")


def aggregate_dilemmadata_source_entry_losses(
    supervisions: tuple[TaskSupervision, ...],
    *,
    task_weights: Mapping[str, float],
) -> DilemmadataLossReport:
    """Reduce rows -> exact source entries -> tasks without active renormalization."""

    task_losses: list[DilemmadataSourceEntryTaskLoss] = []
    for supervision in supervisions:
        if supervision.task_id not in DILEMMADATA_ACTIVE_TASK_IDS:
            raise ValueError("dilemmadata.loss.non_active_supervision")
        if supervision.per_row_loss.numel() == 0:
            continue
        keys = torch.stack(
            (supervision.sample_indices, supervision.source_entry_indices), dim=1
        )
        unique, inverse, counts = torch.unique(
            keys,
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        sums = torch.zeros(
            unique.shape[0],
            dtype=supervision.per_row_loss.dtype,
            device=supervision.per_row_loss.device,
        )
        sums.index_add_(0, inverse, supervision.per_row_loss)
        entry_means = sums / counts.to(sums.dtype)
        weight = float(task_weights[supervision.task_id])
        task_losses.append(
            DilemmadataSourceEntryTaskLoss(
                task_id=supervision.task_id,
                fixed_weight=weight,
                expanded_row_count=supervision.per_row_loss.shape[0],
                effective_source_entry_count=unique.shape[0],
                entry_sample_indices=unique[:, 0],
                entry_source_indices=unique[:, 1],
                entry_row_counts=counts,
                entry_mean_losses=entry_means,
                mean_loss=entry_means.mean(),
            )
        )
    total = (
        None
        if not task_losses
        else torch.stack(
            [item.mean_loss * item.fixed_weight for item in task_losses]
        ).sum()
    )
    return DilemmadataLossReport(
        contract_version=DILEMMADATA_LOSS_CONTRACT_VERSION,
        reduction=(
            "candidate_rows_mean_per_source_entry_then_entries_mean_per_task_fixed_weight_sum"
        ),
        task_losses=tuple(task_losses),
        total_loss=total,
    )


class _LocalEncoderContainer(nn.Module):
    """Preserve accepted encoder-export state names without legacy heads."""

    def __init__(self, config: DilemmadataHierarchicalConfig) -> None:
        super().__init__()
        self.encoder = LocalHeterogeneousEncoder(
            hidden_dim=config.hidden_dim,
            gnn_layers=config.local_gnn_layers,
            dropout=config.dropout,
            residual=config.residual,
            use_message_passing=True,
        )


@dataclass(frozen=True, slots=True)
class DilemmadataHierarchicalOutput:
    model_contract_version: str
    encoder: ContextualEncoderOutput
    predictions: tuple[TaskPrediction, ...]
    supervisions: tuple[TaskSupervision, ...]
    harmonic_loss: DilemmadataLossReport
    routing_operations: RoutingOperationCounts
    reconstruction: tuple[()] = ()
    reconstruction_loss: None = None

    def __post_init__(self) -> None:
        if self.model_contract_version != DILEMMADATA_MODEL_CONTRACT_VERSION:
            raise ValueError("dilemmadata.model.output_version_incompatible")
        tensors = [prediction.logits for prediction in self.predictions]
        tensors.extend(row.per_row_loss for row in self.supervisions)
        tensors.extend(
            value
            for row in self.harmonic_loss.task_losses
            for value in (row.entry_mean_losses, row.mean_loss)
        )
        if self.harmonic_loss.total_loss is not None:
            tensors.append(self.harmonic_loss.total_loss)
        if any(value.dtype != torch.float32 for value in tensors):
            raise ValueError("dilemmadata.model.fp32_head_loss_boundary_violated")


class DilemmadataHierarchicalModel(nn.Module):
    """Four separate Dilemmadata heads over raw-only hierarchical embeddings."""

    def __init__(
        self,
        config: DilemmadataHierarchicalConfig = DilemmadataHierarchicalConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.task_specs = dilemmadata_task_head_specs()
        self.local_baseline = _LocalEncoderContainer(config)
        self.context_encoder = HierarchicalContextEncoder(config.hierarchy_config())
        self.task_heads = SourceNativeTaskHeads(
            self.task_specs,  # type: ignore[arg-type]
            config.hidden_dim,
            config.task_hidden_dim or config.hidden_dim,
            config.dropout,
            force_float32=True,
        )
        self.sequence_decoder = (
            OnsetBiGRUDecoder(config.hidden_dim, config.dropout)
            if config.decoder.kind == "onset_bigru"
            else None
        )

    def encode(
        self,
        raw_graph_batch: object,
        *,
        return_layers: bool = False,
        feature_overlay=None,
    ) -> ContextualEncoderOutput:
        ownership = extract_hierarchy_ownership(raw_graph_batch)
        local = self.local_baseline.encoder(
            raw_graph_batch,
            return_layers=return_layers,
            feature_overlay=feature_overlay,
        )
        return self.context_encoder._forward_with_extracted_ownership(
            local, ownership
        )

    def predict(
        self,
        raw_graph_batch: object,
        *,
        return_layers: bool = False,
        feature_overlay=None,
    ) -> tuple[ContextualEncoderOutput, tuple[TaskPrediction, ...]]:
        encoded = self.encode(
            raw_graph_batch,
            return_layers=return_layers,
            feature_overlay=feature_overlay,
        )
        if self.sequence_decoder is not None:
            encoded = replace(
                encoded,
                fused=self.sequence_decoder(encoded.fused, raw_graph_batch),
            )
        return encoded, self.task_heads(encoded.fused)

    def forward(
        self,
        batch: MultiSourceBatch,
        *,
        return_layers: bool = False,
        include_reconstruction: bool = False,
        class_weights: Mapping[str, Tensor] | None = None,
    ) -> DilemmadataHierarchicalOutput:
        if include_reconstruction:
            raise ValueError("dilemmadata.model.reconstruction_forbidden")
        encoded, predictions = self.predict(
            batch.raw_graph_batch, return_layers=return_layers
        )
        return self.supervise(
            encoded,
            predictions,
            batch.target_batches,
            class_weights=class_weights,
        )

    def supervise(
        self,
        encoded: ContextualEncoderOutput,
        predictions: tuple[TaskPrediction, ...],
        target_batches: tuple[BatchTarget, ...],
        *,
        class_weights: Mapping[str, Tensor] | None = None,
    ) -> DilemmadataHierarchicalOutput:
        """Join targets after raw prediction without repeating the encoder/heads."""

        if any(row.logits.dtype != torch.float32 for row in predictions):
            raise ValueError("dilemmadata.model.fp32_head_logits_required")
        device_type = predictions[0].logits.device.type
        with torch.amp.autocast(device_type, enabled=False):
            supervisions = join_task_supervision(
                predictions,
                target_batches,
                categorical_class_weights=class_weights,
            )
            loss = aggregate_dilemmadata_source_entry_losses(
                supervisions, task_weights=dict(self.config.task_weights)
            )
        return DilemmadataHierarchicalOutput(
            model_contract_version=DILEMMADATA_MODEL_CONTRACT_VERSION,
            encoder=encoded,
            predictions=predictions,
            supervisions=supervisions,
            harmonic_loss=loss,
            routing_operations=routing_operation_counts(
                self.task_specs, supervisions  # type: ignore[arg-type]
            ),
        )


def dilemmadata_model_contract_dict(
    model: DilemmadataHierarchicalModel,
) -> dict[str, object]:
    config = asdict(model.config)
    decoder = config.pop("decoder")
    contract = {
        "model_contract_version": DILEMMADATA_MODEL_CONTRACT_VERSION,
        "head_contract_version": DILEMMADATA_HEAD_CONTRACT_VERSION,
        "loss_contract_version": DILEMMADATA_LOSS_CONTRACT_VERSION,
        "fp32_head_loss_boundary_version": (
            DILEMMADATA_FP32_HEAD_LOSS_BOUNDARY_VERSION
        ),
        "config": config,
        "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
        "pu_tasks_without_heads": list(DILEMMADATA_PU_TASK_IDS),
        "open_tasks_without_heads": list(DILEMMADATA_OPEN_TASK_IDS),
        "head_specs": [asdict(spec) for spec in model.task_specs],
        "prediction_input": "raw_graph_hierarchical_encoder_output_only",
        "target_join": "typed_post_prediction_supervise_only",
        "amp_precision_boundary": (
            "encoder_autocast_allowed_heads_logits_ce_source_entry_total_fp32"
        ),
    }
    if model.config.decoder.kind == "onset_bigru":
        contract.update(
            {
                "decoder_contract_version": ONSET_BIGRU_DECODER_CONTRACT_VERSION,
                "decoder": decoder,
                "sequence_input": "encoded.fused.embeddings.onset_raw_rows_only",
                "owner_context": "raw_onset_to_beat_and_onset_to_bar_mean_pool",
            }
        )
    return contract


def dilemmadata_config_from_model_contract(
    model_contract: object,
    model_state: object,
) -> DilemmadataHierarchicalConfig:
    """Reconstruct a typed config without guessing a decoder from tensors."""

    if not isinstance(model_contract, Mapping):
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.mapping_required"
        )
    if (
        model_contract.get("model_contract_version")
        != DILEMMADATA_MODEL_CONTRACT_VERSION
    ):
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.version_incompatible"
        )
    raw_config = model_contract.get("config")
    expected_config_fields = {
        row.name for row in fields(DilemmadataHierarchicalConfig)
    } - {"decoder"}
    if (
        not isinstance(raw_config, Mapping)
        or set(raw_config) != expected_config_fields
    ):
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.config_fields_invalid"
        )
    if not isinstance(model_state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in model_state.items()
    ):
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.state_invalid"
        )
    has_decoder = "decoder" in model_contract
    decoder_value = model_contract.get("decoder")
    decoder_version = model_contract.get("decoder_contract_version")
    has_sequence_tensors = any(
        name.startswith("sequence_decoder.") for name in model_state
    )
    if not has_decoder:
        if any(
            name in model_contract
            for name in (
                "decoder_contract_version",
                "sequence_input",
                "owner_context",
            )
        ):
            raise DilemmadataModelContractError(
                "dilemmadata.model_contract.decoder_fields_inconsistent"
            )
        if has_sequence_tensors:
            raise DilemmadataModelContractError(
                "dilemmadata.model_contract.mlp_state_has_sequence_decoder"
            )
        decoder = DilemmadataDecoderConfig(kind="mlp")
    else:
        if decoder_version != ONSET_BIGRU_DECODER_CONTRACT_VERSION:
            raise DilemmadataModelContractError(
                "dilemmadata.model_contract.decoder_version_incompatible"
            )
        if (
            not isinstance(decoder_value, Mapping)
            or set(decoder_value) != {"kind"}
            or decoder_value.get("kind") != "onset_bigru"
            or model_contract.get("sequence_input")
            != "encoded.fused.embeddings.onset_raw_rows_only"
            or model_contract.get("owner_context")
            != "raw_onset_to_beat_and_onset_to_bar_mean_pool"
        ):
            raise DilemmadataModelContractError(
                "dilemmadata.model_contract.decoder_invalid"
            )
        if not has_sequence_tensors:
            raise DilemmadataModelContractError(
                "dilemmadata.model_contract.bigru_state_missing_sequence_decoder"
            )
        decoder = DilemmadataDecoderConfig(kind="onset_bigru")
    config = dict(raw_config)
    task_weights = config.get("task_weights")
    if not isinstance(task_weights, (list, tuple)) or any(
        not isinstance(row, (list, tuple)) or len(row) != 2
        for row in task_weights
    ):
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.task_weights_invalid"
        )
    config["task_weights"] = tuple(tuple(row) for row in task_weights)
    config["decoder"] = decoder
    try:
        return DilemmadataHierarchicalConfig(**config)
    except (TypeError, ValueError) as exc:
        raise DilemmadataModelContractError(
            "dilemmadata.model_contract.config_invalid"
        ) from exc


def dilemmadata_model_contract_fingerprint(
    model: DilemmadataHierarchicalModel,
) -> str:
    return sha256(
        json.dumps(
            dilemmadata_model_contract_dict(model),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


_ENCODER_PREFIXES = (
    "local_baseline.encoder.",
    "context_encoder.pooling.",
    "context_encoder.transformer.",
    "context_encoder.fusion.",
)


@dataclass(frozen=True, slots=True)
class DilemmadataEncoderTransferReport:
    contract_version: str
    source_kind: str
    source_checkpoint_sha256: str
    transfer_mode: Literal["frozen_probe", "full_finetune"]
    loaded_tensors: tuple[str, ...]
    unloaded_fresh_tensors: tuple[str, ...]
    optimizer_parameter_names: tuple[str, ...]
    encoder_frozen: bool
    supervised_heads_transferred: bool
    ssl_heads_transferred: bool


def _tensor_state_fingerprint(state: Mapping[str, Tensor]) -> str:
    digest = sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def dilemmadata_fresh_supervised_fingerprint(
    model: DilemmadataHierarchicalModel,
) -> str:
    """Fingerprint decoder/head tensors which encoder transfer must preserve."""

    return _tensor_state_fingerprint(
        {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith(_ENCODER_PREFIXES)
        }
    )


def load_dilemmadata_encoder_state(
    model: DilemmadataHierarchicalModel,
    source: object,
    *,
    source_kind: Literal[
        "phase7a_ssl", "phase8b_multilevel_ssl", "phase6_hierarchical"
    ],
    source_checkpoint_sha256: str,
    transfer_mode: Literal["frozen_probe", "full_finetune"],
) -> DilemmadataEncoderTransferReport:
    """Failure-atomically transfer only encoder tensors from a versioned export."""

    from music_critic.ssl.transfer import (
        EncoderTransferError,
        validate_pretrained_encoder_export_structure,
    )

    try:
        raw_state = validate_pretrained_encoder_export_structure(source)
    except EncoderTransferError as exc:
        raise ValueError(str(exc)) from exc
    if not _is_sha256_text(source_checkpoint_sha256):
        raise ValueError("dilemmadata.transfer.source_sha256_invalid")
    expected_state = model.state_dict()
    expected_names = tuple(
        sorted(
            name
            for name in expected_state
            if name.startswith(_ENCODER_PREFIXES)
        )
    )
    actual_names = tuple(sorted(raw_state))
    if actual_names != expected_names:
        raise ValueError(
            "dilemmadata.transfer.encoder_keys_incompatible:"
            f"missing={sorted(set(expected_names)-set(actual_names))},"
            f"unexpected={sorted(set(actual_names)-set(expected_names))}"
        )
    for name in expected_names:
        value = raw_state[name]
        expected = expected_state[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"dilemmadata.transfer.tensor_incompatible:{name}")
    original = copy.deepcopy(expected_state)
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    merged = {
        name: (
            raw_state[name].detach().to(device=value.device, dtype=value.dtype)
            if name in raw_state
            else value
        )
        for name, value in original.items()
    }
    try:
        model.load_state_dict(merged, strict=True)
        if any(
            not torch.equal(model.state_dict()[name], original[name])
            for name in set(original) - set(expected_names)
        ):
            raise ValueError("dilemmadata.transfer.fresh_tensor_changed")
    except Exception as exc:
        model.load_state_dict(original, strict=True)
        raise ValueError(f"dilemmadata.transfer.application_failed:{exc}") from exc
    named_parameters = dict(model.named_parameters())
    loaded_parameters = tuple(name for name in expected_names if name in named_parameters)
    if transfer_mode == "frozen_probe":
        for name in loaded_parameters:
            named_parameters[name].requires_grad_(False)
    optimizer_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if transfer_mode == "frozen_probe" and set(loaded_parameters) & set(
        optimizer_names
    ):
        model.load_state_dict(original, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        raise ValueError("dilemmadata.transfer.frozen_encoder_in_optimizer")
    return DilemmadataEncoderTransferReport(
        contract_version=DILEMMADATA_ENCODER_TRANSFER_VERSION,
        source_kind=source_kind,
        source_checkpoint_sha256=source_checkpoint_sha256,
        transfer_mode=transfer_mode,
        loaded_tensors=expected_names,
        unloaded_fresh_tensors=tuple(sorted(set(original) - set(expected_names))),
        optimizer_parameter_names=optimizer_names,
        encoder_frozen=transfer_mode == "frozen_probe",
        supervised_heads_transferred=False,
        ssl_heads_transferred=False,
    )


def _is_sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def class_weight_artifact(
    counts: Mapping[str, tuple[int, ...]],
    *,
    policy: Literal[
        "unweighted",
        "inverse_frequency",
        "inverse_sqrt_frequency",
        "inverse_sqrt_frequency_supported",
    ],
    train_membership_fingerprint: str,
) -> dict[str, object]:
    """Create a train-only fingerprinted optional class-weight ablation."""

    if not _is_sha256_text(train_membership_fingerprint) or policy not in {
        "unweighted",
        "inverse_frequency",
        "inverse_sqrt_frequency",
        "inverse_sqrt_frequency_supported",
    } or set(counts) != set(DILEMMADATA_ACTIVE_TASK_IDS):
        raise ValueError("dilemmadata.class_weights.config_invalid")
    weights: dict[str, list[float]] = {}
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        values = counts[task_id]
        if len(values) != len(
            DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary or ()
        ) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("dilemmadata.class_weights.counts_invalid")
        if policy == "unweighted":
            weights[task_id] = [1.0 for _ in values]
        else:
            if (
                policy != "inverse_sqrt_frequency_supported"
                and any(value == 0 for value in values)
            ):
                raise ValueError("dilemmadata.class_weights.zero_train_support")
            power = -1.0 if policy == "inverse_frequency" else -0.5
            raw = [0.0 if value == 0 else float(value) ** power for value in values]
            if policy == "inverse_sqrt_frequency_supported":
                scale = sum(values) / sum(
                    count * weight
                    for count, weight in zip(values, raw, strict=True)
                )
                weights[task_id] = [weight * scale for weight in raw]
            else:
                weights[task_id] = raw
    artifact = {
        "contract_version": DILEMMADATA_CLASS_WEIGHT_CONTRACT_VERSION,
        "policy": policy,
        "source_split": "train_only",
        "train_membership_fingerprint": train_membership_fingerprint,
        "class_counts": {key: list(counts[key]) for key in sorted(counts)},
        "weights": weights,
    }
    return {
        **artifact,
        "fingerprint": sha256(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def class_weight_tensors(
    artifact: Mapping[str, object], *, device: torch.device
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Validate a fingerprinted train-only artifact and materialize FP32 CE weights."""

    payload = dict(artifact)
    observed = payload.pop("fingerprint", None)
    expected = sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if (
        observed != expected
        or payload.get("contract_version")
        != DILEMMADATA_CLASS_WEIGHT_CONTRACT_VERSION
        or payload.get("policy")
        not in {
            "unweighted",
            "inverse_frequency",
            "inverse_sqrt_frequency",
            "inverse_sqrt_frequency_supported",
        }
        or payload.get("source_split") != "train_only"
        or not _is_sha256_text(payload.get("train_membership_fingerprint"))
        or set(payload.get("class_counts", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
        or set(payload.get("weights", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise ValueError("dilemmadata.class_weights.artifact_invalid")
    tensors: dict[str, Tensor] = {}
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        counts = payload["class_counts"][task_id]
        weights = payload["weights"][task_id]
        expected_width = len(
            DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary or ()
        )
        if (
            not isinstance(counts, list)
            or not isinstance(weights, list)
            or len(counts) != expected_width
            or len(weights) != expected_width
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in weights
            )
        ):
            raise ValueError("dilemmadata.class_weights.artifact_invalid")
        tensors[task_id] = torch.tensor(weights, dtype=torch.float32, device=device)
    try:
        expected_artifact = class_weight_artifact(
            {
                task_id: tuple(payload["class_counts"][task_id])
                for task_id in DILEMMADATA_ACTIVE_TASK_IDS
            },
            policy=payload["policy"],
            train_membership_fingerprint=payload["train_membership_fingerprint"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("dilemmadata.class_weights.artifact_invalid") from exc
    if dict(artifact) != expected_artifact:
        raise ValueError("dilemmadata.class_weights.artifact_invalid")
    evidence = {
        "contract_version": DILEMMADATA_CLASS_WEIGHT_CONTRACT_VERSION,
        "policy": payload["policy"],
        "source_split": "train_only",
        "train_membership_fingerprint": payload["train_membership_fingerprint"],
        "artifact_fingerprint": observed,
        "class_counts": payload["class_counts"],
        "weights": payload["weights"],
    }
    return tensors, evidence


__all__ = [
    "DILEMMADATA_ACTIVE_TASK_IDS",
    "DILEMMADATA_CLASS_WEIGHT_CONTRACT_VERSION",
    "DILEMMADATA_ENCODER_TRANSFER_VERSION",
    "DILEMMADATA_FP32_HEAD_LOSS_BOUNDARY_VERSION",
    "DILEMMADATA_HEAD_CONTRACT_VERSION",
    "DILEMMADATA_LOSS_CONTRACT_VERSION",
    "DILEMMADATA_MODEL_CONTRACT_VERSION",
    "DILEMMADATA_OPEN_TASK_IDS",
    "DILEMMADATA_PU_TASK_IDS",
    "DilemmadataEncoderTransferReport",
    "DilemmadataHierarchicalConfig",
    "DilemmadataHierarchicalModel",
    "DilemmadataHierarchicalOutput",
    "DilemmadataLossReport",
    "DilemmadataModelContractError",
    "DilemmadataSourceEntryTaskLoss",
    "DilemmadataTaskHeadSpec",
    "aggregate_dilemmadata_source_entry_losses",
    "class_weight_artifact",
    "class_weight_tensors",
    "dilemmadata_model_contract_dict",
    "dilemmadata_config_from_model_contract",
    "dilemmadata_model_contract_fingerprint",
    "dilemmadata_fresh_supervised_fingerprint",
    "dilemmadata_task_head_specs",
    "load_dilemmadata_encoder_state",
]
