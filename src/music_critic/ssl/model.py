"""Shared-full-target deterministic masked hierarchical SSL baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from music_critic.models import (
    ContextualEncoderOutput,
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    hierarchical_checkpoint_metadata,
)
from music_critic.ssl.contracts import (
    MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
    MASK_PLAN_CONTRACT_VERSION,
    MASK_POLICY_VERSION,
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    SSL_CONTRACT_VERSION,
    MaskPlan,
    is_sha256,
)
from music_critic.ssl.data import SSLBatch
from music_critic.ssl.decoder import (
    DECODER_REMASK_CONTRACT_VERSION,
    REPRESENTATION_DECODER_CONTRACT_VERSION,
    DecoderRemaskPlan,
    RepresentationDecoder,
    build_decoder_remask_plan,
)
from music_critic.ssl.field_registry import (
    MASKABLE_FIELD_REGISTRY_FINGERPRINT,
    MASKABLE_FIELD_REGISTRY_VERSION,
)
from music_critic.ssl.masking import (
    PreparedMaskBinding,
    validate_prepared_mask_binding,
)
from music_critic.ssl.objective import (
    ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
    COSINE_EPSILON,
    COSINE_FORMULA,
    COSINE_REDUCTION,
    LATENT_PROJECTOR_PREDICTOR_CONTRACT_VERSION,
    MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION,
    REPRESENTATION_LOSS_CONTRACT_VERSION,
    SSL_OBJECTIVE_CONTRACT_VERSION,
    ZERO_NORM_POLICY,
    AntiCollapseDiagnostics,
    LatentProjectorPredictor,
    MultiViewRepresentationLoss,
    RepresentationLoss,
    SSLObjectiveLoss,
    SSLObjectiveWeights,
    anti_collapse_diagnostics,
    combine_ssl_losses,
    multi_view_representation_loss,
    representation_cosine_loss,
)
from music_critic.ssl.views import (
    BoundFeatureMaskOverlay,
    FeatureMaskOverlay,
)


SSL_MODEL_CONTRACT_VERSION = "1.2.0"
SSL_MODEL_OUTPUT_CONTRACT_VERSION = "1.2.0"
SSL_REPRESENTATION_TARGET_CONTRACT_VERSION = "1.0.0"
TARGET_MODE = "shared_stop_gradient_full_view"
DECODER_CONTEXT_MODE = (
    "online_owner_track_bar_song_temporal_neighbors"
)

_TRACK_NOTE_EDGE = ("track", "contains_note", "note")
_BAR_NOTE_EDGE = ("bar", "contains_note", "note")
_NOTE_NEXT_EDGE = ("note", "next_in_track", "note")
_NOTE_PREVIOUS_EDGE = ("note", "previous_in_track", "note")


def _rate(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class MaskedGraphSSLConfig:
    """Complete Phase 7A objective configuration beside Phase 6B architecture."""

    mask_rate: float = 0.30
    decoder_views: int = 3
    decoder_remask_probability: float = 0.20
    decoder_hidden_dim: int | None = None
    projector_hidden_dim: int | None = None
    note_weight: float = 1.0
    bar_weight: float = 1.0
    song_weight: float = 1.0
    cosine_epsilon: float = COSINE_EPSILON

    def __post_init__(self) -> None:
        _rate(self.mask_rate, name="mask_rate")
        _positive_int(self.decoder_views, name="decoder_views")
        _rate(
            self.decoder_remask_probability,
            name="decoder_remask_probability",
        )
        for name, value in (
            ("decoder_hidden_dim", self.decoder_hidden_dim),
            ("projector_hidden_dim", self.projector_hidden_dim),
        ):
            if value is not None:
                _positive_int(value, name=name)
        SSLObjectiveWeights(
            note_weight=self.note_weight,
            bar_weight=self.bar_weight,
            song_weight=self.song_weight,
        )
        if self.cosine_epsilon != COSINE_EPSILON:
            raise ValueError(
                "Phase 7A cosine epsilon is contract-fixed at 1e-8"
            )

    def weights(self) -> SSLObjectiveWeights:
        return SSLObjectiveWeights(
            note_weight=self.note_weight,
            bar_weight=self.bar_weight,
            song_weight=self.song_weight,
        )


@dataclass(frozen=True, slots=True)
class RepresentationTargets:
    """Detached full-view targets at the required note, bar, and song levels."""

    contract_version: str
    note: Tensor
    bar: Tensor
    song: Tensor

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != SSL_REPRESENTATION_TARGET_CONTRACT_VERSION
        ):
            raise ValueError("SSL representation target version is incompatible")
        for name, value in (
            ("note", self.note),
            ("bar", self.bar),
            ("song", self.song),
        ):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 2
                or value.requires_grad
            ):
                raise ValueError(
                    f"{name} target must be a detached rank-two tensor"
                )


@dataclass(frozen=True, slots=True)
class LatentPrediction:
    """Inspectable online prediction and detached target for one coarse level."""

    level: str
    prediction: Tensor
    target: Tensor
    loss: RepresentationLoss
    diagnostics: AntiCollapseDiagnostics

    def __post_init__(self) -> None:
        if self.level not in {"bar", "song"}:
            raise ValueError("latent prediction level must be bar or song")
        if self.target.requires_grad:
            raise ValueError("latent target must be stop-gradient")


@dataclass(frozen=True, slots=True)
class SSLForwardOutput:
    """Complete inspectable output for one masked encoder view."""

    contract_version: str
    prepared_mask_binding_fingerprint: str
    mask_plans: tuple[MaskPlan, ...]
    feature_overlay: FeatureMaskOverlay
    online_encoder: ContextualEncoderOutput
    targets: RepresentationTargets
    selected_global_note_indices: Tensor
    decoder_remask_plans: tuple[tuple[DecoderRemaskPlan, ...], ...]
    decoder_predictions: tuple[Tensor, ...]
    note_loss: MultiViewRepresentationLoss
    note_diagnostics: AntiCollapseDiagnostics
    bar_latent: LatentPrediction
    song_latent: LatentPrediction
    objective: SSLObjectiveLoss

    def __post_init__(self) -> None:
        if self.contract_version != SSL_MODEL_OUTPUT_CONTRACT_VERSION:
            raise ValueError("SSL model output contract is incompatible")
        if (
            not is_sha256(
                self.prepared_mask_binding_fingerprint
            )
        ):
            raise ValueError(
                "prepared mask binding fingerprint is incompatible"
            )
        if len(self.decoder_predictions) != len(
            self.decoder_remask_plans
        ):
            raise ValueError("decoder prediction/view counts differ")
        if self.selected_global_note_indices.dtype != torch.long:
            raise ValueError("selected note indices must use torch.long")


class MaskedGraphSSLModel(nn.Module):
    """GraphMAE2-inspired, non-EMA Phase 7A hierarchical baseline."""

    def __init__(
        self,
        encoder_config: HierarchicalBaselineConfig = (
            HierarchicalBaselineConfig()
        ),
        ssl_config: MaskedGraphSSLConfig = MaskedGraphSSLConfig(),
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.ssl_config = ssl_config
        self.encoder = HierarchicalHeterogeneousBaseline(encoder_config)
        hidden_dim = encoder_config.hidden_dim
        self.feature_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder = RepresentationDecoder(
            hidden_dim,
            ssl_config.decoder_hidden_dim,
        )
        self.bar_projector_predictor = LatentProjectorPredictor(
            hidden_dim,
            ssl_config.projector_hidden_dim,
        )
        self.song_projector_predictor = LatentProjectorPredictor(
            hidden_dim,
            ssl_config.projector_hidden_dim,
        )

    def ssl_contract_metadata(self) -> dict[str, object]:
        """Return the complete checkpoint and report compatibility binding."""

        return {
            "ssl_contract_version": SSL_CONTRACT_VERSION,
            "ssl_model_contract_version": SSL_MODEL_CONTRACT_VERSION,
            "ssl_model_output_contract_version": (
                SSL_MODEL_OUTPUT_CONTRACT_VERSION
            ),
            "representation_target_contract_version": (
                SSL_REPRESENTATION_TARGET_CONTRACT_VERSION
            ),
            "target_mode": TARGET_MODE,
            "decoder_context_mode": DECODER_CONTEXT_MODE,
            "mask_plan_contract_version": MASK_PLAN_CONTRACT_VERSION,
            "mask_policy_version": MASK_POLICY_VERSION,
            "prepared_mask_binding_contract_version": (
                PREPARED_MASK_BINDING_CONTRACT_VERSION
            ),
            "masked_feature_overlay_contract_version": (
                MASKED_FEATURE_OVERLAY_CONTRACT_VERSION
            ),
            "maskable_field_registry_version": (
                MASKABLE_FIELD_REGISTRY_VERSION
            ),
            "maskable_field_registry_fingerprint": (
                MASKABLE_FIELD_REGISTRY_FINGERPRINT
            ),
            "decoder_remask_contract_version": (
                DECODER_REMASK_CONTRACT_VERSION
            ),
            "representation_decoder_contract_version": (
                REPRESENTATION_DECODER_CONTRACT_VERSION
            ),
            "representation_loss_contract_version": (
                REPRESENTATION_LOSS_CONTRACT_VERSION
            ),
            "multi_view_loss_contract_version": (
                MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION
            ),
            "latent_projector_predictor_contract_version": (
                LATENT_PROJECTOR_PREDICTOR_CONTRACT_VERSION
            ),
            "ssl_objective_contract_version": (
                SSL_OBJECTIVE_CONTRACT_VERSION
            ),
            "anti_collapse_diagnostics_contract_version": (
                ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
            ),
            "cosine_formula": COSINE_FORMULA,
            "cosine_epsilon": COSINE_EPSILON,
            "cosine_reduction": COSINE_REDUCTION,
            "zero_norm_policy": ZERO_NORM_POLICY,
            "ssl_config": asdict(self.ssl_config),
            "encoder_contract": hierarchical_checkpoint_metadata(
                self.encoder
            ),
            "ema_target_encoder": False,
        }

    def _full_view_targets(
        self,
        graph: object,
        *,
        prepared_input_token: object,
    ) -> RepresentationTargets:
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                encoded = self.encoder._encode_prepared(
                    graph,
                    prepared_input_token=prepared_input_token,
                )
                targets = RepresentationTargets(
                    contract_version=(
                        SSL_REPRESENTATION_TARGET_CONTRACT_VERSION
                    ),
                    note=encoded.fused.embeddings["note"].detach(),
                    bar=encoded.fused.embeddings["bar"].detach(),
                    song=encoded.fused.embeddings["song"].detach(),
                )
        finally:
            self.encoder.train(was_training)
        return targets

    def _decode_views(
        self,
        selected_online: Tensor,
        selected_context: Tensor,
        plans: tuple[MaskPlan, ...],
    ) -> tuple[
        tuple[Tensor, ...],
        tuple[tuple[DecoderRemaskPlan, ...], ...],
    ]:
        predictions: list[Tensor] = []
        plans_by_view: list[tuple[DecoderRemaskPlan, ...]] = []
        row_counts = tuple(
            len(plan.selected_local_node_indices) for plan in plans
        )
        for view_index in range(self.ssl_config.decoder_views):
            view_plans = tuple(
                build_decoder_remask_plan(
                    plan,
                    decoder_view_index=view_index,
                    remask_probability=(
                        self.ssl_config.decoder_remask_probability
                    ),
                )
                for plan in plans
            )
            cursor = 0
            sample_predictions = []
            for row_count, decoder_plan in zip(
                row_counts, view_plans, strict=True
            ):
                rows = selected_online[cursor : cursor + row_count]
                context = selected_context[
                    cursor : cursor + row_count
                ]
                sample_predictions.append(
                    self.decoder(
                        rows,
                        decoder_plan,
                        context=context,
                    )
                )
                cursor += row_count
            if cursor != int(selected_online.shape[0]):
                raise ValueError(
                    "decoder compact-row partition is inconsistent"
                )
            predictions.append(
                torch.cat(sample_predictions, dim=0)
                if sample_predictions
                else selected_online.new_empty(
                    (0, selected_online.shape[1])
                )
            )
            plans_by_view.append(view_plans)
        return tuple(predictions), tuple(plans_by_view)

    @staticmethod
    def _owner_context(
        graph: object,
        *,
        edge_type: tuple[str, str, str],
        owner_embeddings: Tensor,
        note_count: int,
    ) -> tuple[Tensor, Tensor]:
        owners = torch.full(
            (note_count,),
            -1,
            dtype=torch.long,
            device=owner_embeddings.device,
        )
        source, target = graph[edge_type].edge_index
        if source.device != owner_embeddings.device:
            raise ValueError(
                "prepared decoder ownership and embeddings use "
                "different devices"
            )
        owners.index_copy_(0, target, source)
        available = owners >= 0
        safe_owners = owners.clamp_min(0)
        context = owner_embeddings.index_select(
            0,
            safe_owners,
        )
        context = torch.where(
            available.unsqueeze(-1),
            context,
            torch.zeros_like(context),
        )
        return context, available

    def _selected_decoder_context(
        self,
        batch: SSLBatch,
        online: ContextualEncoderOutput,
        selected_global_note_indices: Tensor,
    ) -> Tensor:
        graph = batch.raw_graph_batch
        note = online.fused.embeddings["note"]
        note_count = int(note.shape[0])
        track, track_available = self._owner_context(
            graph,
            edge_type=_TRACK_NOTE_EDGE,
            owner_embeddings=online.fused.embeddings["track"],
            note_count=note_count,
        )
        bar, bar_available = self._owner_context(
            graph,
            edge_type=_BAR_NOTE_EDGE,
            owner_embeddings=online.fused.embeddings["bar"],
            note_count=note_count,
        )
        song_membership = online.fused.batch_membership["note"]
        song = online.fused.embeddings["song"].index_select(
            0, song_membership
        )
        neighbor_sum = torch.zeros_like(note)
        neighbor_count = torch.zeros(
            note_count,
            dtype=note.dtype,
            device=note.device,
        )
        for edge_type in (_NOTE_NEXT_EDGE, _NOTE_PREVIOUS_EDGE):
            source, target = graph[edge_type].edge_index
            if source.device != note.device:
                raise ValueError(
                    "prepared decoder neighbors and embeddings use "
                    "different devices"
                )
            neighbor_sum.index_add_(
                0, target, note.index_select(0, source)
            )
            neighbor_count.index_add_(
                0,
                target,
                torch.ones_like(target, dtype=note.dtype),
            )
        context = track + bar + song + neighbor_sum
        contribution_count = (
            track_available.to(note.dtype)
            + bar_available.to(note.dtype)
            + 1.0
            + neighbor_count
        ).clamp_min(1.0)
        context = context / contribution_count.unsqueeze(-1)
        return context.index_select(
            0,
            selected_global_note_indices,
        )

    def forward(
        self,
        batch: SSLBatch,
        *,
        prepared_mask_binding: PreparedMaskBinding,
    ) -> SSLForwardOutput:
        if not isinstance(batch, SSLBatch):
            raise TypeError("MaskedGraphSSLModel requires a raw-only SSLBatch")
        target_input_token = validate_prepared_mask_binding(
            batch,
            prepared_mask_binding,
            expected_mask_rate=self.ssl_config.mask_rate,
        )
        plans = prepared_mask_binding.mask_plans
        full_targets = self._full_view_targets(
            batch.raw_graph_batch,
            prepared_input_token=target_input_token,
        )
        feature_overlay = prepared_mask_binding.feature_overlay
        bound_overlay: BoundFeatureMaskOverlay = feature_overlay.bind(
            self.feature_mask_token
        )
        online_input_token = validate_prepared_mask_binding(
            batch,
            prepared_mask_binding,
            expected_mask_rate=self.ssl_config.mask_rate,
        )
        online = self.encoder._encode_prepared(
            batch.raw_graph_batch,
            prepared_input_token=online_input_token,
            feature_overlay=bound_overlay,
        )
        selected_indices = (
            prepared_mask_binding.selected_global_note_indices_tensor
        )
        selected_online = online.fused.embeddings["note"].index_select(
            0,
            selected_indices,
        )
        selected_target = full_targets.note.index_select(
            0,
            selected_indices,
        ).detach()
        selected_context = self._selected_decoder_context(
            batch,
            online,
            selected_indices,
        )
        decoder_predictions, decoder_plans = self._decode_views(
            selected_online,
            selected_context,
            plans,
        )
        note_loss = multi_view_representation_loss(
            decoder_predictions,
            selected_target,
            component="note_reconstruction",
        )
        mean_note_prediction = torch.stack(
            decoder_predictions, dim=0
        ).mean(dim=0)
        note_diagnostics = anti_collapse_diagnostics(
            selected_target, mean_note_prediction
        )

        bar_prediction, bar_target = self.bar_projector_predictor(
            online.fused.embeddings["bar"],
            full_targets.bar,
        )
        bar_loss = representation_cosine_loss(
            bar_prediction,
            bar_target,
            component="bar_latent",
        )
        bar_latent = LatentPrediction(
            level="bar",
            prediction=bar_prediction,
            target=bar_target,
            loss=bar_loss,
            diagnostics=anti_collapse_diagnostics(
                bar_target, bar_prediction
            ),
        )
        song_prediction, song_target = self.song_projector_predictor(
            online.fused.embeddings["song"],
            full_targets.song,
        )
        song_loss = representation_cosine_loss(
            song_prediction,
            song_target,
            component="song_latent",
        )
        song_latent = LatentPrediction(
            level="song",
            prediction=song_prediction,
            target=song_target,
            loss=song_loss,
            diagnostics=anti_collapse_diagnostics(
                song_target, song_prediction
            ),
        )
        objective = combine_ssl_losses(
            note_loss,
            bar_loss,
            song_loss,
            weights=self.ssl_config.weights(),
        )
        return SSLForwardOutput(
            contract_version=SSL_MODEL_OUTPUT_CONTRACT_VERSION,
            prepared_mask_binding_fingerprint=(
                prepared_mask_binding.fingerprint
            ),
            mask_plans=plans,
            feature_overlay=feature_overlay,
            online_encoder=online,
            targets=full_targets,
            selected_global_note_indices=selected_indices,
            decoder_remask_plans=decoder_plans,
            decoder_predictions=decoder_predictions,
            note_loss=note_loss,
            note_diagnostics=note_diagnostics,
            bar_latent=bar_latent,
            song_latent=song_latent,
            objective=objective,
        )


def build_ssl_model(
    model_config: object,
    ssl_config: object,
) -> MaskedGraphSSLModel:
    """Build the only accepted Phase 7A hierarchical SSL architecture."""

    if getattr(model_config, "name", None) != "hierarchical":
        raise ValueError("Phase 7A supports only model=hierarchical")
    encoder_config = HierarchicalBaselineConfig(
        hidden_dim=int(model_config.hidden_dim),
        local_gnn_layers=int(model_config.local_gnn_layers),
        transformer_layers=int(model_config.transformer_layers),
        attention_heads=int(model_config.attention_heads),
        ffn_multiplier=int(model_config.ffn_multiplier),
        dropout=float(model_config.dropout),
        local_residual=bool(model_config.residual),
    )
    objective_config = MaskedGraphSSLConfig(
        mask_rate=float(ssl_config.mask_rate),
        decoder_views=int(ssl_config.decoder_views),
        decoder_remask_probability=float(
            ssl_config.decoder_remask_prob
        ),
        decoder_hidden_dim=int(ssl_config.decoder_hidden_dim),
        projector_hidden_dim=int(ssl_config.projector_hidden_dim),
        note_weight=float(ssl_config.note_weight),
        bar_weight=float(ssl_config.bar_weight),
        song_weight=float(ssl_config.song_weight),
        cosine_epsilon=float(ssl_config.epsilon),
    )
    return MaskedGraphSSLModel(encoder_config, objective_config)


__all__ = [
    "SSL_MODEL_CONTRACT_VERSION",
    "SSL_MODEL_OUTPUT_CONTRACT_VERSION",
    "SSL_REPRESENTATION_TARGET_CONTRACT_VERSION",
    "TARGET_MODE",
    "DECODER_CONTEXT_MODE",
    "LatentPrediction",
    "MaskedGraphSSLConfig",
    "MaskedGraphSSLModel",
    "RepresentationTargets",
    "SSLForwardOutput",
    "build_ssl_model",
]
