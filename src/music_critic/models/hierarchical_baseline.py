"""Phase 6B local-GNN plus deterministic hierarchy baseline."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from music_critic.models.baseline import LocalHeterogeneousBaseline
from music_critic.models.heads import (
    BaselineLossReport,
    RoutingOperationCounts,
    TaskPrediction,
    TaskSupervision,
    aggregate_task_losses,
    join_task_supervision,
    routing_operation_counts,
)
from music_critic.models.hierarchy import (
    ContextualEncoderOutput,
    HierarchicalContextEncoder,
    extract_hierarchy_ownership,
)
from music_critic.models.hierarchy_contracts import (
    HIERARCHICAL_MODEL_CONTRACT_VERSION,
    HierarchicalBaselineConfig,
)
from music_critic.models.reconstruction import (
    ReconstructionOutput,
    reconstruction_loss,
)
from music_critic.tasks import MultiSourceBatch


@dataclass(frozen=True, slots=True)
class HierarchicalBaselineOutput:
    """Candidate-first Phase 6B output with all local/coarse/fused evidence."""

    model_contract_version: str
    encoder: ContextualEncoderOutput
    predictions: tuple[TaskPrediction, ...]
    supervisions: tuple[TaskSupervision, ...]
    harmonic_loss: BaselineLossReport
    routing_operations: RoutingOperationCounts
    reconstruction: tuple[ReconstructionOutput, ...]
    reconstruction_loss: Tensor | None

    def __post_init__(self) -> None:
        if (
            self.model_contract_version
            != HIERARCHICAL_MODEL_CONTRACT_VERSION
        ):
            raise ValueError(
                "hierarchical baseline output contract is incompatible"
            )


class HierarchicalHeterogeneousBaseline(nn.Module):
    """Add coarse context beside an unchanged Phase 6A local baseline."""

    def __init__(
        self,
        config: HierarchicalBaselineConfig = HierarchicalBaselineConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.local_baseline = LocalHeterogeneousBaseline(
            config.local_config()
        )
        self.context_encoder = HierarchicalContextEncoder(config)

    @property
    def task_specs(self):
        return self.local_baseline.task_specs

    def encode(
        self,
        raw_graph_batch: object,
        *,
        return_layers: bool = False,
        feature_overlay=None,
    ) -> ContextualEncoderOutput:
        # Extract before Phase 6A validation so every malformed ownership path
        # has a structured Phase 6B error. The internal handoff validates local
        # row/device consistency without scanning the raw relations again.
        ownership = extract_hierarchy_ownership(raw_graph_batch)
        local = self.local_baseline.encode(
            raw_graph_batch,
            return_layers=return_layers,
            feature_overlay=feature_overlay,
        )
        return self.context_encoder._forward_with_extracted_ownership(
            local, ownership
        )

    def _encode_prepared(
        self,
        raw_graph_batch: object,
        *,
        prepared_input_token: object,
        return_layers: bool = False,
        feature_overlay=None,
    ) -> ContextualEncoderOutput:
        """Internal hierarchy entry gated before raw ownership is read."""

        from music_critic.ssl.masking import (
            _verify_prepared_input_token,
        )

        _verify_prepared_input_token(
            raw_graph_batch,
            prepared_input_token,
        )
        ownership = extract_hierarchy_ownership(raw_graph_batch)
        local = self.local_baseline._encode_prepared(
            raw_graph_batch,
            prepared_input_token=prepared_input_token,
            return_layers=return_layers,
            feature_overlay=feature_overlay,
        )
        return self.context_encoder._forward_with_extracted_ownership(
            local, ownership
        )

    def forward(
        self,
        batch: MultiSourceBatch,
        *,
        return_layers: bool = False,
        include_reconstruction: bool = True,
    ) -> HierarchicalBaselineOutput:
        encoded = self.encode(
            batch.raw_graph_batch, return_layers=return_layers
        )
        predictions = self.local_baseline.task_heads(encoded.fused)
        supervisions = join_task_supervision(
            predictions, batch.target_batches
        )
        harmonic = aggregate_task_losses(
            supervisions,
            task_weights=dict(self.config.task_weights),
        )
        reconstruction = (
            self.local_baseline.reconstruction_heads(
                encoded.fused, batch.raw_graph_batch
            )
            if include_reconstruction
            else ()
        )
        return HierarchicalBaselineOutput(
            model_contract_version=HIERARCHICAL_MODEL_CONTRACT_VERSION,
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
        feature_overlay=None,
    ) -> tuple[ContextualEncoderOutput, tuple[TaskPrediction, ...]]:
        encoded = self.encode(
            raw_graph_batch,
            return_layers=return_layers,
            feature_overlay=feature_overlay,
        )
        return encoded, self.local_baseline.task_heads(encoded.fused)


__all__ = [
    "HierarchicalBaselineOutput",
    "HierarchicalHeterogeneousBaseline",
]
