from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    DEFERRED_HEADS,
    GROUP_WEIGHTS,
    OFFICIAL_TRAINING_PROFILE_ID,
    PRIMARY_HEADS,
    aggregate_corrected_losses,
    build_training_profiles,
    component_balanced_record_draw,
    component_sampler_contract,
    corrected_head_roles,
    corrected_loss_contract,
    corrected_masked_multitask_loss,
    corrected_metric_contract,
    corrected_profile_comparison,
    corrected_sampler_draw,
    head_role_contract,
    masked_weighted_cross_entropy,
    stop_gate_contract,
    validate_class_weight_payload,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures/analysisgnn/phase9eb5b_training_policy.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _profiles():
    payload = _fixture()["class_weight_payload"]
    assert isinstance(payload, dict)
    return build_training_profiles(payload)


def test_three_profiles_are_separate_and_official_is_not_corrected() -> None:
    profiles = _profiles()
    assert set(profiles) == {"O", "C0", "C1"}
    assert profiles["O"].profile_id == OFFICIAL_TRAINING_PROFILE_ID
    assert profiles["C0"].profile_id == CORRECTED_NO_TRANSPOSITION_PROFILE_ID
    assert profiles["C1"].profile_id == CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID
    assert profiles["O"].runnable is False
    assert profiles["O"].reproduction_status == "partial_contract_only"
    assert profiles["O"].head_roles["head_count"] == 21
    assert profiles["C0"].head_roles["head_count"] == 20
    assert profiles["O"].dataset_contract != profiles["C0"].dataset_contract


def test_c0_c1_only_substantive_difference_is_transposition() -> None:
    profiles = _profiles()
    comparison = corrected_profile_comparison(profiles["C0"], profiles["C1"])
    assert comparison["identity_differences"] == ["profile_id"]
    assert comparison["substantive_difference_domains"] == [
        "transposition_policy"
    ]
    assert comparison["other_difference_paths"] == []
    assert comparison["only_transposition_differs"] is True
    assert profiles["C0"].loss_aggregation == profiles["C1"].loss_aggregation
    assert profiles["C0"].sampler_policy == profiles["C1"].sampler_policy
    assert profiles["C0"].validation_metrics == profiles["C1"].validation_metrics
    assert profiles["C0"].optimizer_training_budget == profiles[
        "C1"
    ].optimizer_training_budget


def test_corrected_head_roles_cover_registry_exactly_once_as_8_10_2() -> None:
    roles = corrected_head_roles()
    assert len(roles) == len({row.task_id for row in roles}) == 20
    assert tuple(row.task_id for row in roles if row.role == "primary") == PRIMARY_HEADS
    assert tuple(row.task_id for row in roles if row.role == "auxiliary") == AUXILIARY_HEADS
    assert tuple(row.task_id for row in roles if row.role == "deferred") == DEFERRED_HEADS
    assert head_role_contract()["role_counts"] == {
        "primary": 8,
        "auxiliary": 10,
        "deferred": 2,
    }


def test_deferred_roles_and_corrected_vocabulary_sizes_are_frozen() -> None:
    by_task = {row.task_id: row for row in corrected_head_roles()}
    for task in DEFERRED_HEADS:
        row = by_task[task]
        assert row.loss_active is False
        assert row.metric_reportable is False
        assert row.deferred_reason == "missing_negative_supervision"
    assert by_task["quality"].class_count == 17
    assert by_task["roman_numeral"].class_count == 184
    assert "staff" not in by_task


def test_masked_rows_do_not_influence_per_head_loss() -> None:
    logits = torch.tensor([[3.0, -1.0], [-50.0, 50.0], [0.5, 0.2]])
    targets = torch.tensor([0, 0, 1])
    mask = torch.tensor([True, False, True])
    first = masked_weighted_cross_entropy(logits, targets, mask)
    changed = logits.clone()
    changed[1] = torch.tensor([1000.0, -1000.0])
    second = masked_weighted_cross_entropy(changed, targets, mask)
    assert first is not None and second is not None
    assert torch.equal(first, second)


def test_zero_valid_head_is_excluded_from_group_denominator_and_logged() -> None:
    logits = {
        "quality": torch.tensor([[1.0, 0.0]]),
        "inversion": torch.tensor([[1.0, 0.0]]),
    }
    targets = {task: torch.tensor([0]) for task in logits}
    masks = {
        "quality": torch.tensor([True]),
        "inversion": torch.tensor([False]),
    }
    result = corrected_masked_multitask_loss(logits, targets, masks)
    quality = masked_weighted_cross_entropy(
        logits["quality"], targets["quality"], masks["quality"]
    )
    assert quality is not None
    assert result.total is not None and torch.equal(result.total, quality)
    assert result.zero_valid_heads == ("inversion",)


def test_repeating_rows_within_one_head_does_not_change_intertask_weight() -> None:
    quality = masked_weighted_cross_entropy(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([True]),
    )
    repeated = masked_weighted_cross_entropy(
        torch.tensor([[2.0, 0.0]]).repeat(100, 1),
        torch.tensor([0]).repeat(100),
        torch.tensor([True]).repeat(100),
    )
    cadence = masked_weighted_cross_entropy(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([1]),
        torch.tensor([True]),
    )
    assert quality is not None and repeated is not None and cadence is not None
    one = aggregate_corrected_losses({"quality": quality, "cadence": cadence})
    many = aggregate_corrected_losses({"quality": repeated, "cadence": cadence})
    assert one.total is not None and many.total is not None
    assert torch.allclose(one.total, many.total)


def test_deferred_heads_cannot_change_total_and_aggregation_is_deterministic() -> None:
    primary = torch.tensor(2.0)
    auxiliary = torch.tensor(4.0)
    first = aggregate_corrected_losses(
        {"quality": primary, "cadence": auxiliary, "phrase": None}
    )
    second = aggregate_corrected_losses(
        {"cadence": auxiliary, "section": None, "quality": primary}
    )
    assert first.total is not None and second.total is not None
    assert first.total.item() == pytest.approx(2.0 + 0.25 * 4.0)
    assert torch.equal(first.total, second.total)
    assert GROUP_WEIGHTS == {"primary": 1.0, "auxiliary": 0.25, "deferred": 0.0}
    assert corrected_loss_contract()["group_weights"] == GROUP_WEIGHTS


def test_class_weights_are_full_train_only_and_zero_count_is_null() -> None:
    payload = _fixture()["class_weight_payload"]
    assert isinstance(payload, dict)
    validate_class_weight_payload(payload)
    contract = payload["contract"]
    assert contract["source_splits"] == ["train"]
    assert contract["validation_used"] is False
    assert contract["test_used"] is False
    assert contract["augmented_view_multiplier_used"] is False
    heads = {row["task_id"]: row for row in payload["heads"]}
    quality = {
        row["class_value"]: row for row in heads["quality"]["classes"]
    }
    assert len(quality) == 17
    assert quality["augmented sixth"]["train_canonical_target_row_count"] == 0
    assert quality["augmented sixth"]["weight"] is None
    assert quality["augmented sixth"]["train_supported"] is False
    assert len(heads["roman_numeral"]["classes"]) == 184
    assert all(head["supported_weight_mean"] == pytest.approx(1.0) for head in heads.values())
    supported_weights = [
        row["weight"]
        for head in heads.values()
        for row in head["classes"]
        if row["weight"] is not None
    ]
    assert min(supported_weights) >= 0.25
    assert max(supported_weights) <= 4.0


def test_component_balanced_draw_is_seeded_and_epoch_dependent() -> None:
    components = {"c0": ("r0", "r1"), "c1": ("r2",), "c2": ("r3",)}
    first = [
        component_balanced_record_draw(
            components, seed=17, epoch=0, draw_index=index
        )
        for index in range(20)
    ]
    assert first == [
        component_balanced_record_draw(
            components, seed=17, epoch=0, draw_index=index
        )
        for index in range(20)
    ]
    changed = [
        component_balanced_record_draw(
            components, seed=17, epoch=1, draw_index=index
        )
        for index in range(20)
    ]
    assert changed != first
    assert component_sampler_contract()["test_sampling"] == "disabled_loader_not_created"


def test_c0_c1_sampler_record_order_matches_and_c1_uses_only_valid_shifts() -> None:
    components = {"c0": ("r0", "r1"), "c1": ("r2",)}
    valid = {"r0": (0, 2), "r1": (0, 5, 11), "r2": (0, 6)}
    for draw_index in range(12):
        c0 = corrected_sampler_draw(
            components,
            profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
            valid_shifts_by_record=valid,
            seed=17,
            epoch=3,
            draw_index=draw_index,
        )
        c1 = corrected_sampler_draw(
            components,
            profile_id=CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
            valid_shifts_by_record=valid,
            seed=17,
            epoch=3,
            draw_index=draw_index,
        )
        assert c0[:2] == c1[:2]
        assert c0[2] == 0
        assert c1[2] in valid[c1[1]]


def test_metric_contract_separates_selection_joint_and_roman_metrics() -> None:
    contract = corrected_metric_contract()
    selection = contract["primary_model_selection"]
    assert selection["metric_id"] == "corrected_primary_macro_score"
    assert tuple(selection["heads"]) == PRIMARY_HEADS
    assert selection["auxiliary_included"] is False
    assert selection["deferred_included"] is False
    joints = contract["joint_metrics"]
    assert {row["unit"] for row in joints} == {"harmonic_event", "note"}
    assert joints[0]["metric_id"] != joints[1]["metric_id"]
    assert contract["roman_numeral"]["derived_harmonic_correctness_separate"] is True


def test_profiles_keep_test_closed_and_model_unknowns_explicit() -> None:
    profiles = _profiles()
    for key in ("C0", "C1"):
        profile = profiles[key]
        assert profile.test_lock_state["loader_created"] is False
        assert profile.test_lock_state["targets_read"] is False
        assert profile.test_lock_state["metrics_computed"] is False
        assert profile.optimizer_training_budget["parameter_budget"] is None
        assert profile.optimizer_training_budget["batch_window_policy"] is None
        assert profile.runnable is False
    gates = stop_gate_contract()
    assert gates["test_loader_created"] is False
    assert gates["test_targets_read"] is False
    assert gates["checkpoint_selection_uses_test"] is False
