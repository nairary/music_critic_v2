"""Versioned representation objectives and linear anti-collapse diagnostics.

Phase 7A reconstructs stop-gradient representations.  It does not define a
masked-note likelihood, perplexity, pseudo-log-likelihood, critic, or quality
score.  The loss contract is ordinary row-wise cosine error:

``sum_i (1 - cosine(prediction_i, stopgrad(target_i))) / row_count``

with ``eps=1e-8``.  Rows containing zero vectors remain in both the numerator
and denominator; they are reported rather than silently filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from music_critic.ssl.decoder import DecoderRemaskPlan


REPRESENTATION_LOSS_CONTRACT_VERSION = "1.0.0"
ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION = "1.0.0"
MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION = "1.0.0"
LATENT_PROJECTOR_PREDICTOR_CONTRACT_VERSION = "1.0.0"
SSL_OBJECTIVE_CONTRACT_VERSION = "1.0.0"

COSINE_EPSILON = 1e-8
COSINE_FORMULA = "one_minus_cosine"
COSINE_REDUCTION = "sum_count_mean"
ZERO_NORM_POLICY = "count_and_retain"

_NO_ELIGIBLE_ROWS = "no_eligible_rows"
_FEWER_THAN_TWO_ROWS = "fewer_than_two_rows"


def _validate_component(component: object) -> str:
    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a non-empty string")
    if component != component.strip():
        raise ValueError("component must already be trimmed")
    return component


def _validate_representation_pair(
    prediction: object,
    target: object,
) -> tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must be rank-two tensors")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match exactly")
    if prediction.shape[1] == 0:
        raise ValueError("representation width must be positive")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must use floating-point dtypes")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target dtypes must match")
    if prediction.device != target.device:
        raise ValueError("prediction and target devices must match")
    return prediction, target


@dataclass(frozen=True, slots=True)
class RepresentationLoss:
    """One component's exact row-sum/count/mean cosine-loss report."""

    contract_version: str
    component: str
    numerator: Tensor
    denominator: int
    mean: Tensor | None
    unavailable_reason: str | None
    zero_norm_count: int

    def __post_init__(self) -> None:
        if self.contract_version != REPRESENTATION_LOSS_CONTRACT_VERSION:
            raise ValueError("representation loss contract version is incompatible")
        _validate_component(self.component)
        if (
            not isinstance(self.numerator, Tensor)
            or self.numerator.ndim != 0
            or not self.numerator.is_floating_point()
        ):
            raise ValueError("representation loss numerator must be a float scalar")
        if (
            isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator < 0
        ):
            raise ValueError("representation loss denominator must be non-negative")
        if (
            isinstance(self.zero_norm_count, bool)
            or not isinstance(self.zero_norm_count, int)
            or not 0 <= self.zero_norm_count <= self.denominator
        ):
            raise ValueError("zero_norm_count must lie within the row count")
        if self.denominator == 0:
            if self.mean is not None or self.unavailable_reason != _NO_ELIGIBLE_ROWS:
                raise ValueError(
                    "empty representation loss must have an explicit unavailable state"
                )
        elif (
            not isinstance(self.mean, Tensor)
            or self.mean.ndim != 0
            or self.mean.device != self.numerator.device
            or self.mean.dtype != self.numerator.dtype
            or self.unavailable_reason is not None
        ):
            raise ValueError(
                "available representation loss requires a compatible scalar mean"
            )

    @property
    def count(self) -> int:
        """Alias used by metrics code that names the denominator ``count``."""

        return self.denominator

    @property
    def loss_sum(self) -> Tensor:
        return self.numerator

    @property
    def loss_mean(self) -> Tensor | None:
        return self.mean

    @property
    def available(self) -> bool:
        return self.mean is not None


def representation_cosine_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    component: str,
) -> RepresentationLoss:
    """Compute versioned ``1-cosine`` while retaining every eligible row."""

    prediction, target = _validate_representation_pair(prediction, target)
    component = _validate_component(component)
    row_count = int(prediction.shape[0])
    # ``sum`` keeps the empty numerator differentiably connected to prediction
    # while the public mean remains unavailable instead of a fake zero result.
    if row_count == 0:
        return RepresentationLoss(
            contract_version=REPRESENTATION_LOSS_CONTRACT_VERSION,
            component=component,
            numerator=prediction.sum(),
            denominator=0,
            mean=None,
            unavailable_reason=_NO_ELIGIBLE_ROWS,
            zero_norm_count=0,
        )

    detached_target = target.detach()
    prediction_norms = torch.linalg.vector_norm(prediction, dim=-1)
    target_norms = torch.linalg.vector_norm(detached_target, dim=-1)
    zero_rows = (prediction_norms == 0) | (target_norms == 0)
    # PyTorch's cosine implementation clamps each norm by ``eps``.  A zero
    # vector therefore has cosine zero, contributes loss one, and is counted.
    per_row = 1.0 - F.cosine_similarity(
        prediction,
        detached_target,
        dim=-1,
        eps=COSINE_EPSILON,
    )
    numerator = per_row.sum()
    return RepresentationLoss(
        contract_version=REPRESENTATION_LOSS_CONTRACT_VERSION,
        component=component,
        numerator=numerator,
        denominator=row_count,
        mean=numerator / row_count,
        unavailable_reason=None,
        zero_norm_count=int(zero_rows.count_nonzero().item()),
    )


@dataclass(frozen=True, slots=True)
class AntiCollapseDiagnostics:
    """Detached O(ND) sufficient-statistic diagnostics for one row set."""

    contract_version: str
    row_count: int
    embedding_dim: int
    target_embedding_variance: Tensor | None
    prediction_embedding_variance: Tensor | None
    target_mean_norm: Tensor | None
    prediction_mean_norm: Tensor | None
    target_zero_norm_count: int
    prediction_zero_norm_count: int
    target_mean_off_diagonal_cosine: Tensor | None
    prediction_mean_off_diagonal_cosine: Tensor | None
    unavailable_reason: str | None
    pairwise_unavailable_reason: str | None
    pairwise_policy: str = "exact_linear_normalized_sum"

    def __post_init__(self) -> None:
        if self.contract_version != ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION:
            raise ValueError("anti-collapse contract version is incompatible")
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count < 0
            or isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, int)
            or self.embedding_dim <= 0
        ):
            raise ValueError("anti-collapse row/dimension counts are invalid")
        for name, count in (
            ("target_zero_norm_count", self.target_zero_norm_count),
            ("prediction_zero_norm_count", self.prediction_zero_norm_count),
        ):
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= self.row_count
            ):
                raise ValueError(f"{name} must lie within row_count")
        regular = (
            self.target_embedding_variance,
            self.prediction_embedding_variance,
            self.target_mean_norm,
            self.prediction_mean_norm,
        )
        pairwise = (
            self.target_mean_off_diagonal_cosine,
            self.prediction_mean_off_diagonal_cosine,
        )
        if self.row_count == 0:
            if (
                any(value is not None for value in (*regular, *pairwise))
                or self.unavailable_reason != _NO_ELIGIBLE_ROWS
                or self.pairwise_unavailable_reason != _NO_ELIGIBLE_ROWS
            ):
                raise ValueError("empty diagnostics must be explicitly unavailable")
        else:
            if (
                any(
                    not isinstance(value, Tensor) or value.ndim != 0
                    for value in regular
                )
                or self.unavailable_reason is not None
            ):
                raise ValueError("non-empty diagnostics require scalar statistics")
            if self.row_count == 1:
                if (
                    any(value is not None for value in pairwise)
                    or self.pairwise_unavailable_reason
                    != _FEWER_THAN_TWO_ROWS
                ):
                    raise ValueError(
                        "singleton pairwise cosine must be explicitly unavailable"
                    )
            elif (
                any(
                    not isinstance(value, Tensor) or value.ndim != 0
                    for value in pairwise
                )
                or self.pairwise_unavailable_reason is not None
            ):
                raise ValueError(
                    "multi-row diagnostics require pairwise scalar statistics"
                )


def _mean_off_diagonal_cosine(values: Tensor, norms: Tensor) -> Tensor:
    """Exact ordered-pair mean from O(ND) normalized-vector statistics."""

    normalized = values / norms.clamp_min(COSINE_EPSILON).unsqueeze(-1)
    normalized_sum = normalized.sum(dim=0)
    diagonal_sum = normalized.square().sum()
    ordered_pair_sum = normalized_sum.square().sum() - diagonal_sum
    row_count = int(values.shape[0])
    return ordered_pair_sum / (row_count * (row_count - 1))


def anti_collapse_diagnostics(
    target: Tensor,
    prediction: Tensor,
) -> AntiCollapseDiagnostics:
    """Report variance, norms, zeros, and pairwise cosine without N x N data."""

    prediction, target = _validate_representation_pair(prediction, target)
    detached_target = target.detach()
    detached_prediction = prediction.detach()
    row_count, embedding_dim = (
        int(prediction.shape[0]),
        int(prediction.shape[1]),
    )
    if row_count == 0:
        return AntiCollapseDiagnostics(
            contract_version=ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
            row_count=0,
            embedding_dim=embedding_dim,
            target_embedding_variance=None,
            prediction_embedding_variance=None,
            target_mean_norm=None,
            prediction_mean_norm=None,
            target_zero_norm_count=0,
            prediction_zero_norm_count=0,
            target_mean_off_diagonal_cosine=None,
            prediction_mean_off_diagonal_cosine=None,
            unavailable_reason=_NO_ELIGIBLE_ROWS,
            pairwise_unavailable_reason=_NO_ELIGIBLE_ROWS,
        )

    target_norms = torch.linalg.vector_norm(detached_target, dim=-1)
    prediction_norms = torch.linalg.vector_norm(detached_prediction, dim=-1)
    pairwise_reason = (
        _FEWER_THAN_TWO_ROWS if row_count == 1 else None
    )
    return AntiCollapseDiagnostics(
        contract_version=ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
        row_count=row_count,
        embedding_dim=embedding_dim,
        target_embedding_variance=detached_target.var(
            dim=0, unbiased=False
        ).mean(),
        prediction_embedding_variance=detached_prediction.var(
            dim=0, unbiased=False
        ).mean(),
        target_mean_norm=target_norms.mean(),
        prediction_mean_norm=prediction_norms.mean(),
        target_zero_norm_count=int(
            (target_norms == 0).count_nonzero().item()
        ),
        prediction_zero_norm_count=int(
            (prediction_norms == 0).count_nonzero().item()
        ),
        target_mean_off_diagonal_cosine=(
            None
            if row_count == 1
            else _mean_off_diagonal_cosine(
                detached_target, target_norms
            )
        ),
        prediction_mean_off_diagonal_cosine=(
            None
            if row_count == 1
            else _mean_off_diagonal_cosine(
                detached_prediction, prediction_norms
            )
        ),
        unavailable_reason=None,
        pairwise_unavailable_reason=pairwise_reason,
    )


@dataclass(frozen=True, slots=True)
class DecoderViewRepresentationLoss:
    """One separately reportable decoder-view reconstruction loss."""

    decoder_view_index: int
    decoder_plan_fingerprint: str | None
    decoder_view_seed: int | None
    loss: RepresentationLoss

    def __post_init__(self) -> None:
        if (
            isinstance(self.decoder_view_index, bool)
            or not isinstance(self.decoder_view_index, int)
            or self.decoder_view_index < 0
        ):
            raise ValueError("decoder_view_index must be non-negative")
        if (self.decoder_plan_fingerprint is None) != (
            self.decoder_view_seed is None
        ):
            raise ValueError("decoder plan fingerprint and seed availability differ")
        if self.decoder_plan_fingerprint is not None and (
            len(self.decoder_plan_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.decoder_plan_fingerprint
            )
        ):
            raise ValueError("decoder plan fingerprint must be SHA-256")
        if self.decoder_view_seed is not None and (
            isinstance(self.decoder_view_seed, bool)
            or not isinstance(self.decoder_view_seed, int)
            or not 0 <= self.decoder_view_seed < (1 << 64)
        ):
            raise ValueError("decoder view seed must be uint64")


@dataclass(frozen=True, slots=True)
class MultiViewRepresentationLoss:
    """Per-view reports plus their exact combined row mean."""

    contract_version: str
    component: str
    view_losses: tuple[DecoderViewRepresentationLoss, ...]
    numerator: Tensor
    denominator: int
    mean: Tensor | None
    unavailable_reason: str | None
    zero_norm_count: int

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION
        ):
            raise ValueError("multi-view loss contract version is incompatible")
        _validate_component(self.component)
        if not self.view_losses:
            raise ValueError("multi-view loss requires at least one decoder view")
        if tuple(
            view.decoder_view_index for view in self.view_losses
        ) != tuple(range(len(self.view_losses))):
            raise ValueError("decoder view losses must be contiguous and ordered")
        if any(
            view.loss.component != self.component for view in self.view_losses
        ):
            raise ValueError("decoder view loss components differ")
        if (
            not isinstance(self.numerator, Tensor)
            or self.numerator.ndim != 0
            or isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator < 0
            or isinstance(self.zero_norm_count, bool)
            or not isinstance(self.zero_norm_count, int)
            or not 0 <= self.zero_norm_count <= self.denominator
        ):
            raise ValueError("multi-view aggregate fields are inconsistent")
        if self.denominator == 0:
            if self.mean is not None or self.unavailable_reason != _NO_ELIGIBLE_ROWS:
                raise ValueError("empty multi-view loss must be unavailable")
        elif (
            not isinstance(self.mean, Tensor)
            or self.mean.ndim != 0
            or self.unavailable_reason is not None
        ):
            raise ValueError("available multi-view loss requires a scalar mean")

    @property
    def count(self) -> int:
        return self.denominator

    @property
    def loss_sum(self) -> Tensor:
        return self.numerator

    @property
    def loss_mean(self) -> Tensor | None:
        return self.mean

    @property
    def available(self) -> bool:
        return self.mean is not None


def multi_view_representation_loss(
    predictions: Sequence[Tensor],
    target: Tensor,
    *,
    component: str,
    plans: Sequence[DecoderRemaskPlan] | None = None,
) -> MultiViewRepresentationLoss:
    """Report every decoder view and the mean across all view/row pairs."""

    predictions = tuple(predictions)
    if not predictions:
        raise ValueError("at least one decoder-view prediction is required")
    if plans is not None:
        plans = tuple(plans)
        if len(plans) != len(predictions):
            raise ValueError("decoder plans and predictions must have equal length")
        if tuple(plan.decoder_view_index for plan in plans) != tuple(
            range(len(plans))
        ):
            raise ValueError("decoder plans must be contiguous and ordered")
    component = _validate_component(component)
    view_losses = []
    for view_index, prediction in enumerate(predictions):
        loss = representation_cosine_loss(
            prediction,
            target,
            component=component,
        )
        plan = None if plans is None else plans[view_index]
        view_losses.append(
            DecoderViewRepresentationLoss(
                decoder_view_index=view_index,
                decoder_plan_fingerprint=(
                    None if plan is None else plan.fingerprint
                ),
                decoder_view_seed=(
                    None if plan is None else plan.stable_seed
                ),
                loss=loss,
            )
        )
    numerator = torch.stack(
        [view.loss.numerator for view in view_losses]
    ).sum()
    denominator = sum(view.loss.denominator for view in view_losses)
    return MultiViewRepresentationLoss(
        contract_version=MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION,
        component=component,
        view_losses=tuple(view_losses),
        numerator=numerator,
        denominator=denominator,
        mean=(
            None
            if denominator == 0
            else numerator / denominator
        ),
        unavailable_reason=(
            _NO_ELIGIBLE_ROWS if denominator == 0 else None
        ),
        zero_norm_count=sum(
            view.loss.zero_norm_count for view in view_losses
        ),
    )


class LatentProjectorPredictor(nn.Module):
    """Shared projector plus online predictor for bar/song latent objectives."""

    contract_version = LATENT_PROJECTOR_PREDICTOR_CONTRACT_VERSION

    def __init__(
        self,
        hidden_dim: int,
        projector_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if projector_hidden_dim is None:
            projector_hidden_dim = hidden_dim
        if (
            isinstance(projector_hidden_dim, bool)
            or not isinstance(projector_hidden_dim, int)
            or projector_hidden_dim <= 0
        ):
            raise ValueError("projector_hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(projector_hidden_dim),
            nn.Linear(projector_hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(projector_hidden_dim),
            nn.Linear(projector_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        online_latents: Tensor,
        full_view_latents: Tensor,
    ) -> tuple[Tensor, Tensor]:
        online_latents, full_view_latents = _validate_representation_pair(
            online_latents,
            full_view_latents,
        )
        if int(online_latents.shape[1]) != self.hidden_dim:
            raise ValueError("latent hidden dimension is incompatible with projector")
        prediction = self.predictor(self.projector(online_latents))
        # No graph from the full-view encoder or this target-side projector
        # invocation survives.  The same projector parameters remain trainable
        # through the online invocation above.
        with torch.no_grad():
            target = self.projector(full_view_latents.detach()).detach()
        return prediction, target


LatentPredictionHead = LatentProjectorPredictor


@dataclass(frozen=True, slots=True)
class SSLObjectiveWeights:
    """Independent Phase 7A note/bar/song objective weights."""

    note_weight: float = 1.0
    bar_weight: float = 0.25
    song_weight: float = 0.25

    def __post_init__(self) -> None:
        for name, value in (
            ("note_weight", self.note_weight),
            ("bar_weight", self.bar_weight),
            ("song_weight", self.song_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if not any(
            value > 0
            for value in (
                self.note_weight,
                self.bar_weight,
                self.song_weight,
            )
        ):
            raise ValueError("at least one SSL objective weight must be positive")


@dataclass(frozen=True, slots=True)
class SSLObjectiveLoss:
    """Weighted note/bar/song objective with strict unavailable propagation."""

    contract_version: str
    weights: SSLObjectiveWeights
    note_reconstruction: MultiViewRepresentationLoss
    bar_latent: RepresentationLoss
    song_latent: RepresentationLoss
    total_loss: Tensor | None
    unavailable_components: tuple[tuple[str, str], ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if self.contract_version != SSL_OBJECTIVE_CONTRACT_VERSION:
            raise ValueError("SSL objective contract version is incompatible")
        if self.note_reconstruction.component != "note_reconstruction":
            raise ValueError("note reconstruction component name is incompatible")
        if self.bar_latent.component != "bar_latent":
            raise ValueError("bar latent component name is incompatible")
        if self.song_latent.component != "song_latent":
            raise ValueError("song latent component name is incompatible")
        if self.unavailable_components:
            if self.total_loss is not None or self.unavailable_reason is None:
                raise ValueError(
                    "required unavailable components make total SSL loss unavailable"
                )
        elif (
            not isinstance(self.total_loss, Tensor)
            or self.total_loss.ndim != 0
            or self.unavailable_reason is not None
        ):
            raise ValueError("complete SSL objective requires a scalar total loss")

    @property
    def available(self) -> bool:
        return self.total_loss is not None


def combine_ssl_losses(
    note_reconstruction: MultiViewRepresentationLoss,
    bar_latent: RepresentationLoss,
    song_latent: RepresentationLoss,
    *,
    weights: SSLObjectiveWeights = SSLObjectiveWeights(),
) -> SSLObjectiveLoss:
    """Apply ``w_note*L_note + w_bar*L_bar + w_song*L_song`` exactly.

    A positively weighted unavailable component makes the total unavailable;
    its weight is never silently redistributed to the remaining components.
    """

    if not isinstance(note_reconstruction, MultiViewRepresentationLoss):
        raise TypeError("note_reconstruction must be a multi-view loss")
    if not isinstance(bar_latent, RepresentationLoss) or not isinstance(
        song_latent, RepresentationLoss
    ):
        raise TypeError("bar_latent and song_latent must be representation losses")
    if not isinstance(weights, SSLObjectiveWeights):
        raise TypeError("weights must be SSLObjectiveWeights")
    components = (
        (
            "note_reconstruction",
            float(weights.note_weight),
            note_reconstruction,
        ),
        ("bar_latent", float(weights.bar_weight), bar_latent),
        ("song_latent", float(weights.song_weight), song_latent),
    )
    unavailable = tuple(
        (
            name,
            loss.unavailable_reason or _NO_ELIGIBLE_ROWS,
        )
        for name, weight, loss in components
        if weight > 0 and loss.mean is None
    )
    if unavailable:
        return SSLObjectiveLoss(
            contract_version=SSL_OBJECTIVE_CONTRACT_VERSION,
            weights=weights,
            note_reconstruction=note_reconstruction,
            bar_latent=bar_latent,
            song_latent=song_latent,
            total_loss=None,
            unavailable_components=unavailable,
            unavailable_reason="required_component_unavailable",
        )
    weighted_means = [
        loss.mean * weight
        for _name, weight, loss in components
        if weight > 0 and loss.mean is not None
    ]
    # SSLObjectiveWeights guarantees at least one positive weight, and the
    # unavailable branch above guarantees every corresponding mean exists.
    total_loss = torch.stack(weighted_means).sum()
    return SSLObjectiveLoss(
        contract_version=SSL_OBJECTIVE_CONTRACT_VERSION,
        weights=weights,
        note_reconstruction=note_reconstruction,
        bar_latent=bar_latent,
        song_latent=song_latent,
        total_loss=total_loss,
        unavailable_components=(),
        unavailable_reason=None,
    )


__all__ = [
    "ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION",
    "COSINE_EPSILON",
    "COSINE_FORMULA",
    "COSINE_REDUCTION",
    "LATENT_PROJECTOR_PREDICTOR_CONTRACT_VERSION",
    "MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION",
    "REPRESENTATION_LOSS_CONTRACT_VERSION",
    "SSL_OBJECTIVE_CONTRACT_VERSION",
    "ZERO_NORM_POLICY",
    "AntiCollapseDiagnostics",
    "DecoderViewRepresentationLoss",
    "LatentPredictionHead",
    "LatentProjectorPredictor",
    "MultiViewRepresentationLoss",
    "RepresentationLoss",
    "SSLObjectiveLoss",
    "SSLObjectiveWeights",
    "anti_collapse_diagnostics",
    "combine_ssl_losses",
    "multi_view_representation_loss",
    "representation_cosine_loss",
]
