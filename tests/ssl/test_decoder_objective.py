from __future__ import annotations

from collections.abc import Iterable
import math

import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from music_critic.ssl.contracts import (
    MASK_POLICY_VERSION,
    UNIFORM_NOTE_MASK_POLICY,
    CollateralFeatureMask,
    MaskPlan,
    canonical_sha256,
)
from music_critic.ssl.decoder import (
    RepresentationDecoder,
    apply_decoder_remask,
    build_decoder_remask_plan,
    build_decoder_remask_plans,
    selected_global_node_indices,
)
from music_critic.ssl.field_registry import NOTE_PITCH_GROUP
from music_critic.ssl.objective import (
    COSINE_EPSILON,
    LatentProjectorPredictor,
    SSLObjectiveWeights,
    anti_collapse_diagnostics,
    combine_ssl_losses,
    multi_view_representation_loss,
    representation_cosine_loss,
)


def _mask_plan(row_count: int) -> MaskPlan:
    seed_sha256 = canonical_sha256(
        {
            "fixture": "decoder-objective",
            "row_count": row_count,
        }
    )
    peer_collateral = CollateralFeatureMask(
        reason=NOTE_PITCH_GROUP.peer_note_collateral_reason,
        node_type="note",
        local_node_indices=(),
        features=NOTE_PITCH_GROUP.peer_note_collateral_fields,
    )
    track_collateral = CollateralFeatureMask(
        reason=NOTE_PITCH_GROUP.collateral_reason,
        node_type="track",
        local_node_indices=(0,) if row_count else (),
        features=NOTE_PITCH_GROUP.collateral_fields,
    )
    return MaskPlan.create(
        mask_policy=UNIFORM_NOTE_MASK_POLICY,
        mask_policy_version=MASK_POLICY_VERSION,
        dataset_id="fixture",
        piece_id=f"piece-{row_count}",
        stage="train",
        epoch=3,
        encoder_view_index=0,
        selected_node_type="note",
        selected_local_node_indices=tuple(range(row_count)),
        primary_feature_group="note_pitch_group",
        collateral_feature_masks=(
            peer_collateral,
            track_collateral,
        ),
        requested_mask_rate=1.0 if row_count else 0.0,
        maskable_node_count=row_count,
        realized_mask_rate=1.0 if row_count else 0.0,
        global_seed=19,
        stable_seed=int(seed_sha256[:16], 16),
        stable_seed_sha256=seed_sha256,
    )


def _has_nonzero_gradient(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


def _dense_off_diagonal_cosine(values: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(values, dim=-1, eps=COSINE_EPSILON)
    matrix = normalized @ normalized.T
    mask = ~torch.eye(
        values.shape[0],
        dtype=torch.bool,
        device=values.device,
    )
    return matrix[mask].mean()


def test_decoder_view_seeds_are_distinct_and_bit_exact() -> None:
    mask_plan = _mask_plan(64)
    first = build_decoder_remask_plans(
        mask_plan,
        decoder_views=3,
        remask_probability=0.2,
    )
    second = build_decoder_remask_plans(
        mask_plan,
        decoder_views=3,
        remask_probability=0.2,
    )

    assert first == second
    assert tuple(plan.decoder_view_index for plan in first) == (0, 1, 2)
    assert len({plan.stable_seed for plan in first}) == 3
    assert len({plan.stable_seed_sha256 for plan in first}) == 3
    assert len({plan.fingerprint for plan in first}) == 3
    assert all(plan.mask_plan_fingerprint == mask_plan.fingerprint for plan in first)
    assert all(plan.selected_row_count == mask_plan.selected_count for plan in first)
    assert all(
        set(plan.remasked_positions) <= set(range(mask_plan.selected_count))
        for plan in first
    )

    latents = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    token = torch.full((4,), -17.0)
    original = latents.clone()
    remasked = apply_decoder_remask(latents, first[0], token)
    remasked_positions = torch.tensor(
        first[0].remasked_positions,
        dtype=torch.long,
    )
    visible_positions = torch.tensor(
        sorted(set(range(64)) - set(first[0].remasked_positions)),
        dtype=torch.long,
    )
    assert torch.equal(latents, original)
    assert remasked.data_ptr() != latents.data_ptr()
    assert torch.equal(
        remasked.index_select(0, remasked_positions),
        token.expand(remasked_positions.shape[0], -1),
    )
    assert torch.equal(
        remasked.index_select(0, visible_positions),
        latents.index_select(0, visible_positions),
    )


def test_fully_remasked_rows_remain_context_conditioned() -> None:
    torch.manual_seed(7)
    plan = build_decoder_remask_plan(
        _mask_plan(2),
        decoder_view_index=0,
        remask_probability=1.0,
    )
    decoder = RepresentationDecoder(4, 8).eval()
    latents = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    )
    context = torch.tensor(
        [[0.5, 0.0, -0.5, 1.0], [-1.0, 0.5, 1.0, 0.0]]
    )

    prediction = decoder(latents, plan, context=context)

    assert plan.remasked_positions == (0, 1)
    assert not torch.equal(prediction[0], prediction[1])


def test_decoder_casts_parameter_token_for_autocast_latents() -> None:
    plan = build_decoder_remask_plan(
        _mask_plan(2),
        decoder_view_index=0,
        remask_probability=1.0,
    )
    decoder = RepresentationDecoder(hidden_dim=4).eval()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = decoder(
            torch.zeros(2, 4, dtype=torch.bfloat16),
            plan,
        )

    assert prediction.shape == (2, 4)
    assert prediction.dtype == torch.bfloat16


def test_simple_masking_mode_is_one_view_with_no_latent_remask() -> None:
    mask_plan = _mask_plan(7)
    plans = build_decoder_remask_plans(
        mask_plan,
        decoder_views=1,
        remask_probability=0.0,
    )
    assert len(plans) == 1
    assert plans[0].remasked_positions == ()
    assert plans[0].realized_remask_rate == 0.0

    torch.manual_seed(5)
    decoder = RepresentationDecoder(hidden_dim=6, decoder_hidden_dim=9).eval()
    latents = torch.randn(7, 6)
    assert torch.equal(
        decoder(latents, plans[0]),
        decoder(latents),
    )


def test_decoder_probability_boundaries_and_batch_offsets() -> None:
    mask_plan = _mask_plan(5)
    none = build_decoder_remask_plan(
        mask_plan,
        decoder_view_index=0,
        remask_probability=0.0,
    )
    every = build_decoder_remask_plan(
        mask_plan,
        decoder_view_index=0,
        remask_probability=1.0,
    )
    assert none.remasked_positions == ()
    assert every.remasked_positions == (0, 1, 2, 3, 4)
    assert every.realized_remask_rate == 1.0

    small = _mask_plan(2)
    assert selected_global_node_indices(
        (small, mask_plan),
        torch.tensor([0, 2, 7], dtype=torch.long),
    ).tolist() == list(range(7))


@pytest.mark.parametrize(
    "probability",
    (-0.01, 1.01, math.nan, math.inf, -math.inf),
)
def test_decoder_rejects_invalid_remask_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_decoder_remask_plan(
            _mask_plan(3),
            decoder_view_index=0,
            remask_probability=probability,
        )


@pytest.mark.parametrize("probability", (True, "0.2"))
def test_decoder_rejects_non_real_remask_probability(
    probability: object,
) -> None:
    with pytest.raises(TypeError, match="real number"):
        build_decoder_remask_plan(
            _mask_plan(3),
            decoder_view_index=0,
            remask_probability=probability,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("decoder_views", (0, -1, True))
def test_decoder_rejects_invalid_view_count(decoder_views: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_decoder_remask_plans(
            _mask_plan(3),
            decoder_views=decoder_views,  # type: ignore[arg-type]
            remask_probability=0.2,
        )


def test_decoder_rejects_wrong_compact_rows_and_mask_token() -> None:
    plan = build_decoder_remask_plan(
        _mask_plan(3),
        decoder_view_index=0,
        remask_probability=1.0,
    )
    with pytest.raises(ValueError, match="compact rows"):
        apply_decoder_remask(torch.zeros(4, 5), plan, torch.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        apply_decoder_remask(torch.zeros(3, 5), plan, torch.zeros(4))
    with pytest.raises(ValueError, match="dtype"):
        apply_decoder_remask(
            torch.zeros(3, 5),
            plan,
            torch.zeros(5, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="batch_offsets"):
        selected_global_node_indices((_mask_plan(2),), (0, 1, 2))
    with pytest.raises(ValueError, match="outside"):
        selected_global_node_indices((_mask_plan(2),), (0, 1))


def test_cosine_loss_zero_singleton_and_empty_states_are_explicit() -> None:
    zero = representation_cosine_loss(
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        component="bar_latent",
    )
    assert zero.count == 2
    assert zero.zero_norm_count == 2
    assert zero.unavailable_reason is None
    assert zero.mean is not None and torch.equal(zero.mean, torch.tensor(1.0))
    assert torch.equal(zero.numerator, torch.tensor(2.0))
    assert torch.isfinite(zero.numerator)

    singleton = representation_cosine_loss(
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[3.0, 4.0]]),
        component="song_latent",
    )
    assert singleton.count == 1
    assert singleton.zero_norm_count == 0
    assert singleton.mean is not None
    assert singleton.mean.item() == pytest.approx(0.0, abs=1e-7)

    empty = representation_cosine_loss(
        torch.empty(0, 4),
        torch.empty(0, 4),
        component="note_reconstruction",
    )
    assert empty.count == 0
    assert empty.mean is None
    assert empty.unavailable_reason == "no_eligible_rows"
    assert empty.zero_norm_count == 0
    assert torch.equal(empty.numerator, torch.tensor(0.0))
    assert torch.isfinite(empty.numerator)

    empty_diagnostics = anti_collapse_diagnostics(
        torch.empty(0, 4),
        torch.empty(0, 4),
    )
    assert empty_diagnostics.unavailable_reason == "no_eligible_rows"
    assert (
        empty_diagnostics.pairwise_unavailable_reason
        == "no_eligible_rows"
    )
    assert empty_diagnostics.target_embedding_variance is None
    assert empty_diagnostics.target_mean_off_diagonal_cosine is None

    singleton_diagnostics = anti_collapse_diagnostics(
        torch.zeros(1, 4),
        torch.ones(1, 4),
    )
    assert singleton_diagnostics.unavailable_reason is None
    assert (
        singleton_diagnostics.pairwise_unavailable_reason
        == "fewer_than_two_rows"
    )
    assert singleton_diagnostics.target_zero_norm_count == 1
    assert singleton_diagnostics.prediction_zero_norm_count == 0
    assert singleton_diagnostics.target_embedding_variance is not None
    assert singleton_diagnostics.target_embedding_variance.item() == 0.0
    assert singleton_diagnostics.target_mean_off_diagonal_cosine is None


def test_target_is_detached_and_prediction_receives_gradient() -> None:
    prediction = torch.tensor(
        [[0.2, 0.7, -0.4], [0.9, -0.1, 0.3]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[-0.3, 0.4, 0.8], [0.1, 0.5, -0.7]],
        requires_grad=True,
    )
    report = representation_cosine_loss(
        prediction,
        target,
        component="bar_latent",
    )
    assert report.mean is not None
    report.mean.backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad)
    assert target.grad is None


def test_decoder_projector_and_predictor_receive_gradients() -> None:
    torch.manual_seed(23)
    mask_plan = _mask_plan(5)
    no_remask = build_decoder_remask_plan(
        mask_plan,
        decoder_view_index=0,
        remask_probability=0.0,
    )
    decoder = RepresentationDecoder(hidden_dim=8, decoder_hidden_dim=12)
    online_notes = torch.randn(5, 8, requires_grad=True)
    note_target = torch.randn(5, 8, requires_grad=True)
    note_prediction = decoder(online_notes, no_remask)
    note_loss = representation_cosine_loss(
        note_prediction,
        note_target,
        component="note_reconstruction",
    )
    assert note_loss.mean is not None
    note_loss.mean.backward()
    assert online_notes.grad is not None
    assert torch.count_nonzero(online_notes.grad)
    assert note_target.grad is None
    assert _has_nonzero_gradient(decoder.network.parameters())

    decoder.zero_grad(set_to_none=True)
    all_remasked = build_decoder_remask_plan(
        mask_plan,
        decoder_view_index=1,
        remask_probability=1.0,
    )
    remasked_loss = representation_cosine_loss(
        decoder(online_notes.detach(), all_remasked),
        note_target.detach(),
        component="note_reconstruction",
    )
    assert remasked_loss.mean is not None
    remasked_loss.mean.backward()
    assert decoder.mask_token.grad is not None
    assert torch.count_nonzero(decoder.mask_token.grad)

    latent_head = LatentProjectorPredictor(
        hidden_dim=8,
        projector_hidden_dim=11,
    )
    online_bars = torch.randn(4, 8, requires_grad=True)
    full_view_bars = torch.randn(4, 8, requires_grad=True)
    latent_prediction, latent_target = latent_head(
        online_bars,
        full_view_bars,
    )
    assert latent_prediction.requires_grad
    assert not latent_target.requires_grad
    latent_loss = representation_cosine_loss(
        latent_prediction,
        latent_target,
        component="bar_latent",
    )
    assert latent_loss.mean is not None
    latent_loss.mean.backward()
    assert online_bars.grad is not None
    assert torch.count_nonzero(online_bars.grad)
    assert full_view_bars.grad is None
    assert _has_nonzero_gradient(latent_head.projector.parameters())
    assert _has_nonzero_gradient(latent_head.predictor.parameters())


def test_multiview_mean_and_weighted_unavailable_state() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    predictions = (
        target.clone().requires_grad_(),
        -target.clone().requires_grad_(),
    )
    note = multi_view_representation_loss(
        predictions,
        target,
        component="note_reconstruction",
    )
    assert len(note.view_losses) == 2
    assert note.count == 4
    assert note.mean is not None
    assert note.mean.item() == pytest.approx(1.0)

    bar = representation_cosine_loss(
        target,
        target,
        component="bar_latent",
    )
    song = representation_cosine_loss(
        target[:1],
        -target[:1],
        component="song_latent",
    )
    combined = combine_ssl_losses(
        note,
        bar,
        song,
        weights=SSLObjectiveWeights(
            note_weight=2.0,
            bar_weight=3.0,
            song_weight=4.0,
        ),
    )
    assert combined.total_loss is not None
    assert combined.total_loss.item() == pytest.approx(10.0)
    assert combined.unavailable_components == ()

    unavailable_note = multi_view_representation_loss(
        (torch.empty(0, 2),),
        torch.empty(0, 2),
        component="note_reconstruction",
    )
    unavailable = combine_ssl_losses(
        unavailable_note,
        bar,
        song,
    )
    assert unavailable.total_loss is None
    assert unavailable.unavailable_reason == "required_component_unavailable"
    assert unavailable.unavailable_components == (
        ("note_reconstruction", "no_eligible_rows"),
    )


class _RejectPairwiseSquare(TorchDispatchMode):
    """Fail if production diagnostics materialize an N by N tensor."""

    def __init__(self, row_count: int) -> None:
        super().__init__()
        self.row_count = row_count
        self.maximum_tensor_elements = 0

    def _inspect(self, value: object) -> None:
        if isinstance(value, torch.Tensor):
            self.maximum_tensor_elements = max(
                self.maximum_tensor_elements,
                value.numel(),
            )
            if (
                value.ndim >= 2
                and tuple(value.shape[-2:])
                == (self.row_count, self.row_count)
            ):
                raise AssertionError("diagnostics materialized an N x N tensor")
        elif isinstance(value, (tuple, list)):
            for item in value:
                self._inspect(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._inspect(item)

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        del types
        result = func(*args, **(kwargs or {}))
        self._inspect(result)
        return result


def test_anti_collapse_matches_dense_reference_without_n_by_n_path() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    prediction = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_target_pairwise = _dense_off_diagonal_cosine(target)
    expected_prediction_pairwise = _dense_off_diagonal_cosine(prediction)
    diagnostics = anti_collapse_diagnostics(target, prediction)

    assert diagnostics.row_count == 4
    assert diagnostics.embedding_dim == 3
    assert diagnostics.target_zero_norm_count == 1
    assert diagnostics.prediction_zero_norm_count == 1
    assert torch.allclose(
        diagnostics.target_embedding_variance,
        target.var(dim=0, unbiased=False).mean(),
    )
    assert torch.allclose(
        diagnostics.prediction_embedding_variance,
        prediction.var(dim=0, unbiased=False).mean(),
    )
    assert torch.allclose(
        diagnostics.target_mean_norm,
        torch.linalg.vector_norm(target, dim=-1).mean(),
    )
    assert torch.allclose(
        diagnostics.prediction_mean_norm,
        torch.linalg.vector_norm(prediction, dim=-1).mean(),
    )
    assert torch.allclose(
        diagnostics.target_mean_off_diagonal_cosine,
        expected_target_pairwise,
        atol=1e-7,
    )
    assert torch.allclose(
        diagnostics.prediction_mean_off_diagonal_cosine,
        expected_prediction_pairwise,
        atol=1e-7,
    )

    row_count = 257
    generator = torch.Generator().manual_seed(31)
    large_target = torch.randn(row_count, 7, generator=generator)
    large_prediction = torch.randn(row_count, 7, generator=generator)
    guard = _RejectPairwiseSquare(row_count)
    with guard:
        large = anti_collapse_diagnostics(
            large_target,
            large_prediction,
        )
    assert large.target_mean_off_diagonal_cosine is not None
    assert large.prediction_mean_off_diagonal_cosine is not None
    assert guard.maximum_tensor_elements <= row_count * 7
