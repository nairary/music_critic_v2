"""Trainable feature-only and local-HeteroGNN Phase 6A baselines."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from music_critic.models.contracts import (
    MODEL_CONTRACT_VERSION,
    LocalBaselineConfig,
    active_task_head_specs,
)
from music_critic.models.encoder import (
    LocalHeterogeneousEncoder,
    MultiScaleEncoderOutput,
)
from music_critic.models.heads import (
    BaselineLossReport,
    RoutingOperationCounts,
    SourceNativeTaskHeads,
    TaskPrediction,
    TaskSupervision,
    aggregate_task_losses,
    join_task_supervision,
    routing_operation_counts,
)
from music_critic.models.reconstruction import (
    RawReconstructionHeads,
    ReconstructionOutput,
    reconstruction_loss,
)
from music_critic.tasks import MultiSourceBatch


@dataclass(frozen=True, slots=True)
class BaselineOutput:
    """Complete inspectable Phase 6A forward result."""

    model_contract_version: str
    encoder: MultiScaleEncoderOutput
    predictions: tuple[TaskPrediction, ...]
    supervisions: tuple[TaskSupervision, ...]
    harmonic_loss: BaselineLossReport
    routing_operations: RoutingOperationCounts
    reconstruction: tuple[ReconstructionOutput, ...]
    reconstruction_loss: Tensor | None

    def __post_init__(self) -> None:
        if self.model_contract_version != MODEL_CONTRACT_VERSION:
            raise ValueError("baseline output model contract is incompatible")


class LocalHeterogeneousBaseline(nn.Module):
    """One controlled implementation for both required Phase 6A variants."""

    def __init__(self, config: LocalBaselineConfig = LocalBaselineConfig()) -> None:
        super().__init__()
        self.config = config
        self.task_specs = active_task_head_specs()
        task_hidden_dim = config.task_hidden_dim or config.hidden_dim
        self.encoder = LocalHeterogeneousEncoder(
            hidden_dim=config.hidden_dim,
            gnn_layers=config.gnn_layers,
            dropout=config.dropout,
            residual=config.residual,
            use_message_passing=config.variant == "local_gnn",
        )
        self.task_heads = SourceNativeTaskHeads(
            self.task_specs,
            config.hidden_dim,
            task_hidden_dim,
            config.dropout,
        )
        self.reconstruction_heads = RawReconstructionHeads(config.hidden_dim)

    def encode(
        self, raw_graph_batch: object, *, return_layers: bool = False
    ) -> MultiScaleEncoderOutput:
        """Encode a raw graph without any target or source metadata."""

        return self.encoder(raw_graph_batch, return_layers=return_layers)

    def forward(
        self,
        batch: MultiSourceBatch,
        *,
        return_layers: bool = False,
        include_reconstruction: bool = True,
    ) -> BaselineOutput:
        encoded = self.encode(
            batch.raw_graph_batch, return_layers=return_layers
        )
        predictions = self.task_heads(encoded.final_output)
        supervisions = join_task_supervision(
            predictions, batch.target_batches
        )
        task_weights = dict(self.config.task_weights)
        harmonic = aggregate_task_losses(
            supervisions, task_weights=task_weights
        )
        reconstruction = (
            self.reconstruction_heads(
                encoded.final_output, batch.raw_graph_batch
            )
            if include_reconstruction
            else ()
        )
        return BaselineOutput(
            model_contract_version=MODEL_CONTRACT_VERSION,
            encoder=encoded,
            predictions=predictions,
            supervisions=supervisions,
            harmonic_loss=harmonic,
            routing_operations=routing_operation_counts(
                self.task_specs, supervisions
            ),
            reconstruction=reconstruction,
            reconstruction_loss=reconstruction_loss(reconstruction),
        )

    def predict(
        self,
        raw_graph_batch: object,
        *,
        return_layers: bool = False,
    ) -> tuple[MultiScaleEncoderOutput, tuple[TaskPrediction, ...]]:
        """Emit raw-graph candidate logits without any target sidecar."""

        encoded = self.encode(
            raw_graph_batch, return_layers=return_layers
        )
        return encoded, self.task_heads(encoded.final_output)


__all__ = [
    "BaselineOutput",
    "LocalHeterogeneousBaseline",
]
