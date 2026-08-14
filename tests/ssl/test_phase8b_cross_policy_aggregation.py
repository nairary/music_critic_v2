from __future__ import annotations

import pytest
import torch

from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
)
from music_critic.ssl.multilevel import (
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    ONSET_LATENT,
    PHASE7A_BAR_LATENT,
    PHASE7A_NOTE_RECONSTRUCTION,
    PHASE7A_SONG_LATENT,
    PHASE8B_FAMILY_LOSS_CONTRACT_VERSION,
    PHASE8B_OBJECTIVE_FAMILIES,
    PHASE8B_SCHEDULED_VIEW_AGGREGATION,
    TRACK_LATENT,
    Phase8BFamilyLoss,
    Phase8BObjectiveAccumulator,
    Phase8BObjectiveConfig,
    aggregate_phase8b_family_loss_views,
)
from music_critic.ssl.phase8b_acceptance import (
    phase8b_cross_policy_manual_oracle,
)


_POLICIES = (
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)


def _row(
    family: str,
    numerator: float,
    denominator: int,
    config: Phase8BObjectiveConfig,
) -> Phase8BFamilyLoss:
    value = torch.tensor(numerator, dtype=torch.float64, requires_grad=True)
    return Phase8BFamilyLoss(
        contract_version=PHASE8B_FAMILY_LOSS_CONTRACT_VERSION,
        family=family,
        numerator=value,
        eligible_denominator=denominator,
        mean_loss=value / denominator,
        available=True,
        unavailable_reason=None,
        configured_weight=config.weight(family),
        active=True,
        zero_norm_count=0,
    )


def _equal_config() -> Phase8BObjectiveConfig:
    return Phase8BObjectiveConfig.for_mode("multilevel_equal_weight")


def _oracle_rows(config: Phase8BObjectiveConfig):
    return (
        (
            ONSET_PITCH_DESCENDANTS,
            (_row(ONSET_LATENT, 3.0, 2, config),),
        ),
        (
            BEAT_PITCH_DESCENDANTS,
            (_row(BEAT_LATENT, 5.0, 4, config),),
        ),
        (
            CONTIGUOUS_BAR_PITCH_SPAN,
            (_row(HIERARCHY_BAR_LATENT, 6.0, 3, config),),
        ),
        (
            TRACK_BAR_PITCH_SPAN,
            (
                _row(HIERARCHY_BAR_LATENT, 15.0, 5, config),
                _row(TRACK_LATENT, 9.0, 6, config),
            ),
        ),
    )


def _family(batch, family: str):
    return next(row for row in batch.families if row.family == family)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_independent_manual_oracle_applies_repeated_bar_weight_once() -> None:
    config = _equal_config()
    rows = _oracle_rows(config)
    batch = aggregate_phase8b_family_loss_views(
        rows, objective_config=config
    )
    oracle = phase8b_cross_policy_manual_oracle()

    bar = _family(batch, HIERARCHY_BAR_LATENT)
    assert bar.numerator is not None
    assert float(bar.numerator.detach()) == 21.0
    assert bar.eligible_denominator == 8
    assert float(bar.mean_loss.detach()) == 2.625
    assert bar.family_view_pass_count == 2
    assert bar.applied_family_weight_count == 1
    assert batch.total_loss is not None
    assert float(batch.total_loss.detach()) == 6.875
    assert float(batch.total_loss.detach()) == oracle["family_global_total"]
    assert oracle["old_policy_pass_average"] == 2.3125
    assert float(batch.total_loss.detach()) != oracle["old_policy_pass_average"]

    batch.total_loss.backward()
    gradients = {
        (policy, row.family): float(row.numerator.grad.detach())
        for policy, family_rows in rows
        for row in family_rows
        if row.numerator is not None
    }
    assert gradients[(CONTIGUOUS_BAR_PITCH_SPAN, HIERARCHY_BAR_LATENT)] == (
        1.0 / 8.0
    )
    assert gradients[(TRACK_BAR_PITCH_SPAN, HIERARCHY_BAR_LATENT)] == (
        1.0 / 8.0
    )


def test_reporting_matches_optimizer_formula_with_one_packed_materialization() -> None:
    config = _equal_config()
    batch = aggregate_phase8b_family_loss_views(
        _oracle_rows(config), objective_config=config
    )
    accumulator = Phase8BObjectiveAccumulator(config)
    accumulator.update_batch(batch)
    report = accumulator.finalize()

    assert report["aggregation"] == PHASE8B_SCHEDULED_VIEW_AGGREGATION
    assert report["optimizer_total_loss"] == pytest.approx(6.875)
    assert report["reported_total_loss"] == pytest.approx(6.875)
    assert report["optimizer_reported_total_consistency"]["consistent"]
    assert report["family_denominators"][HIERARCHY_BAR_LATENT] == 8
    assert report["family_means"][HIERARCHY_BAR_LATENT] == pytest.approx(2.625)
    assert report["family_view_pass_counts"][HIERARCHY_BAR_LATENT] == 2
    assert report["applied_family_weight_count"][HIERARCHY_BAR_LATENT] == 1
    assert report["family_view_pass_count"] == 5
    assert report["eligible_prediction_row_count"] == 20
    assert report["packed_host_materialization_count"] == 1
    assert report["maximum_packed_d2h_transfers_per_cpu_batch"] <= 1
    assert report["retained_cuda_tensor_count"] == 0
    assert report["retained_prediction_tensor_count"] == 0
    assert not _contains_tensor(report)


def test_policy_order_and_unavailable_family_do_not_rescale_other_means() -> None:
    config = _equal_config()
    rows = _oracle_rows(config)
    forward = aggregate_phase8b_family_loss_views(rows, objective_config=config)
    reverse = aggregate_phase8b_family_loss_views(
        tuple(reversed(rows)), objective_config=config
    )
    assert forward.total_loss is not None and reverse.total_loss is not None
    assert torch.equal(forward.total_loss, reverse.total_loss)
    for family in PHASE8B_OBJECTIVE_FAMILIES:
        left = _family(forward, family)
        right = _family(reverse, family)
        assert left.eligible_denominator == right.eligible_denominator
        if left.mean_loss is not None:
            assert right.mean_loss is not None
            assert torch.equal(left.mean_loss, right.mean_loss)

    without_beat = aggregate_phase8b_family_loss_views(
        tuple(
            (policy, () if policy == BEAT_PITCH_DESCENDANTS else family_rows)
            for policy, family_rows in rows
        ),
        objective_config=config,
    )
    beat = _family(without_beat, BEAT_LATENT)
    assert beat.numerator is None
    assert beat.eligible_denominator == 0
    assert beat.mean_loss is None
    assert beat.applied_family_weight_count == 0
    assert without_beat.total_loss is not None
    assert float(without_beat.total_loss.detach()) == 5.625
    for family in (ONSET_LATENT, HIERARCHY_BAR_LATENT, TRACK_LATENT):
        assert torch.equal(
            _family(without_beat, family).mean_loss,
            _family(forward, family).mean_loss,
        )


def test_bar_eligible_row_change_is_isolated_and_single_policy_is_unchanged() -> None:
    config = _equal_config()
    baseline_rows = _oracle_rows(config)
    baseline = aggregate_phase8b_family_loss_views(
        baseline_rows, objective_config=config
    )
    changed_rows = (*baseline_rows[:-1], (
        TRACK_BAR_PITCH_SPAN,
        (
            _row(HIERARCHY_BAR_LATENT, 20.0, 7, config),
            _row(TRACK_LATENT, 9.0, 6, config),
        ),
    ))
    changed = aggregate_phase8b_family_loss_views(
        changed_rows, objective_config=config
    )
    baseline_bar = _family(baseline, HIERARCHY_BAR_LATENT)
    changed_bar = _family(changed, HIERARCHY_BAR_LATENT)
    assert baseline_bar.numerator is not None
    assert changed_bar.numerator is not None
    assert float(baseline_bar.numerator.detach()) == 21.0
    assert baseline_bar.eligible_denominator == 8
    assert float(changed_bar.numerator.detach()) == 26.0
    assert changed_bar.eligible_denominator == 10
    assert float(changed_bar.mean_loss.detach()) == 2.6
    for family in (ONSET_LATENT, BEAT_LATENT, TRACK_LATENT):
        assert torch.equal(
            _family(changed, family).mean_loss,
            _family(baseline, family).mean_loss,
        )

    onset_only = Phase8BObjectiveConfig.for_mode("onset_only")
    onset_row = _row(ONSET_LATENT, 3.0, 2, onset_only)
    single = aggregate_phase8b_family_loss_views(
        ((ONSET_PITCH_DESCENDANTS, (onset_row,)),),
        objective_config=onset_only,
    )
    assert single.total_loss is not None
    assert torch.equal(single.total_loss, onset_row.mean_loss)


def test_mask_only_old_families_use_family_global_rows_across_four_views() -> None:
    config = Phase8BObjectiveConfig.for_mode("phase7a_control")
    values = (
        ((2.0, 1), (3.0, 1), (5.0, 1)),
        ((4.0, 2), (8.0, 2), (12.0, 3)),
        ((9.0, 3), (15.0, 3), (7.0, 1)),
        ((16.0, 4), (24.0, 4), (18.0, 3)),
    )
    families = (
        PHASE7A_NOTE_RECONSTRUCTION,
        PHASE7A_BAR_LATENT,
        PHASE7A_SONG_LATENT,
    )
    rows = tuple(
        (
            policy,
            tuple(
                _row(family, numerator, denominator, config)
                for family, (numerator, denominator) in zip(
                    families, pass_values, strict=True
                )
            ),
        )
        for policy, pass_values in zip(_POLICIES, values, strict=True)
    )
    batch = aggregate_phase8b_family_loss_views(rows, objective_config=config)
    assert float(
        _family(batch, PHASE7A_NOTE_RECONSTRUCTION).mean_loss.detach()
    ) == 3.1
    assert float(_family(batch, PHASE7A_BAR_LATENT).mean_loss.detach()) == 5.0
    assert float(_family(batch, PHASE7A_SONG_LATENT).mean_loss.detach()) == 5.25
    assert batch.total_loss is not None
    assert float(batch.total_loss.detach()) == 13.35
    assert float(batch.total_loss.detach()) != 12.75
    for family in families:
        row = _family(batch, family)
        assert row.family_view_pass_count == 4
        assert row.applied_family_weight_count == 1
