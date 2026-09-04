from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
)
from music_critic.experiments.analysisgnn.full_orbit_training import (
    FULL_ORBIT_DRAW_BUDGET,
    FULL_ORBIT_PROFILE_ID,
    FULL_ORBIT_UPDATE_BUDGET,
    FullOrbitRuntimeConfig,
    FullOrbitSampler,
    build_full_orbit_optimizer_scheduler,
    build_full_orbit_table,
    full_orbit_profile_contract,
)
from music_critic.experiments.analysisgnn.orbit_matched_control import (
    ORBIT_MATCHED_CONTROL_LABEL,
    ORBIT_MATCHED_CONTROL_PROFILE_ID,
    OrbitMatchedControlRuntimeConfig,
    OrbitMatchedControlSampler,
    build_orbit_matched_control_optimizer_scheduler,
    completed_control_history_contract,
    orbit_matched_control_profile_contract,
)
from music_critic.experiments.analysisgnn.training_policy import (
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
)


def _small_table():
    return build_full_orbit_table(
        {"component-a": ("record-a",), "component-b": ("record-b",)},
        {"record-a": (0, 1, 6), "record-b": (0, 5, 11)},
    )


def test_control_runtime_matches_c2_budget_optimizer_and_scheduler() -> None:
    control = OrbitMatchedControlRuntimeConfig()
    c2 = FullOrbitRuntimeConfig()
    assert control.profile_id == ORBIT_MATCHED_CONTROL_PROFILE_ID
    assert control.semantic_base_profile_id == CORRECTED_NO_TRANSPOSITION_PROFILE_ID
    assert control.matched_schedule_profile_id == FULL_ORBIT_PROFILE_ID
    assert control.applied_update_budget == c2.applied_update_budget == 120_000
    assert control.train_draw_budget == c2.train_draw_budget == 240_000
    assert control.train_draw_budget == FULL_ORBIT_DRAW_BUDGET
    assert control.applied_update_budget == FULL_ORBIT_UPDATE_BUDGET
    assert control.batch_size == c2.batch_size == 2
    assert control.warmup_applied_updates == c2.warmup_applied_updates == 6_000
    assert control.validation_interval == c2.validation_interval == 5_000
    assert control.applied_shift_pc == 0
    assert control.test_enabled is False

    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    control_optimizer, control_scheduler = (
        build_orbit_matched_control_optimizer_scheduler(first)  # type: ignore[arg-type]
    )
    c2_optimizer, c2_scheduler = build_full_orbit_optimizer_scheduler(
        second  # type: ignore[arg-type]
    )
    assert control_optimizer.defaults == c2_optimizer.defaults
    for step in (0, 1, 5_999, 6_000, 60_000, 119_999, 120_000):
        assert control_scheduler.lr_lambdas[0](step) == pytest.approx(
            c2_scheduler.lr_lambdas[0](step), abs=0.0
        )


def test_control_sampler_reuses_exact_c2_order_but_applies_identity() -> None:
    table = _small_table()
    source = FullOrbitSampler(table)
    control = OrbitMatchedControlSampler(table)
    for offset in range(len(table) * 2 + 3):
        c2_draw = source.peek(offset)
        c0_draw = control.peek(offset)
        assert (
            c0_draw.orbit_epoch,
            c0_draw.orbit_index,
            c0_draw.component_id,
            c0_draw.record_id,
            c0_draw.schedule_shift_pc,
        ) == (
            c2_draw.orbit_epoch,
            c2_draw.orbit_index,
            c2_draw.component_id,
            c2_draw.record_id,
            c2_draw.shift_pc,
        )
        assert c0_draw.applied_shift_pc == 0

    control.advance_after_applied_update()
    state = control.state_dict()
    restored = OrbitMatchedControlSampler(table)
    restored.load_state_dict(state)
    assert restored.position == 2
    assert restored.peek() == control.peek()
    assert state["profile_id"] == ORBIT_MATCHED_CONTROL_PROFILE_ID
    assert state["matched_schedule_profile_id"] == FULL_ORBIT_PROFILE_ID

    tampered = dict(state)
    tampered["applied_shift_pc"] = 1
    with pytest.raises(CorrectedTrainingError):
        restored.load_state_dict(tampered)


def test_control_profile_is_fresh_test_locked_and_c2_matched() -> None:
    profile = orbit_matched_control_profile_contract()
    c2 = full_orbit_profile_contract()
    assert profile["profile"] == ORBIT_MATCHED_CONTROL_LABEL
    assert profile["comparison_profile_id"] == FULL_ORBIT_PROFILE_ID
    assert profile["comparison_profile_fingerprint"] == c2["fingerprint"]
    assert profile["record_multiplicity"] == "exactly_equal_to_C2"
    assert profile["scheduled_shift_applied"] is False
    assert profile["applied_shift_pc"] == 0
    assert profile["from_scratch"] is True
    assert profile["resume_from_c0_10k_checkpoint"] is False
    assert profile["resume_from_c2_checkpoint"] is False
    assert profile["c2_implementation_changed"] is False
    assert profile["test_enabled"] is False
    assert profile["control_training_run"] is False


def test_completed_history_rejects_any_non_identity_application() -> None:
    records = ("a", "b", "a", "b", "a", "b")
    schedule_shifts = (0, 1, 6, 11, 5, 2)
    applied_shifts = (0, 0, 0, 0, 0, 0)
    history = completed_control_history_contract(
        records,
        schedule_shifts,
        applied_shifts,
        expected_draws=len(records),
    )
    assert history["draw_count"] == len(records)
    assert history["scheduled_shift_distribution"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "5": 1,
        "6": 1,
        "11": 1,
    }
    assert history["applied_shift_distribution"] == {"0": len(records)}
    assert history["all_applied_shifts_are_identity"] is True

    with pytest.raises(
        CorrectedTrainingError,
        match="analysisgnn.orbit_matched_control.non_identity_shift_applied",
    ):
        completed_control_history_contract(
            records,
            schedule_shifts,
            (0, 0, 1, 0, 0, 0),
            expected_draws=len(records),
        )


def test_resume_history_requires_exact_c2_schedule_prefix() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_phase9eb5i_analysisgnn_c0_orbit_matched.py"
    )
    spec = importlib.util.spec_from_file_location("phase9eb5i_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sampler = OrbitMatchedControlSampler(_small_table())
    expected = [sampler.peek(offset) for offset in range(4)]
    sampler.advance_after_applied_update(4)
    records = [draw.record_id for draw in expected]
    schedule_shifts = [draw.schedule_shift_pc for draw in expected]
    module._validate_restored_histories(
        sampler=sampler,
        record_history=records,
        schedule_shift_history=schedule_shifts,
        applied_shift_history=[0, 0, 0, 0],
    )

    tampered_records = list(records)
    tampered_records[2] = "record-not-in-prefix"
    with pytest.raises(
        CorrectedTrainingError,
        match="analysisgnn.orbit_matched_control.resume_schedule_prefix_mismatch",
    ):
        module._validate_restored_histories(
            sampler=sampler,
            record_history=tampered_records,
            schedule_shift_history=schedule_shifts,
            applied_shift_history=[0, 0, 0, 0],
        )


def test_runner_cpu_smoke_maps_every_schedule_stratum_to_identity() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_phase9eb5i_analysisgnn_c0_orbit_matched.py"
    )
    spec = importlib.util.spec_from_file_location("phase9eb5i_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    smoke = module.smoke("cpu")
    assert smoke["valid"] is True
    assert smoke["profile_id"] == ORBIT_MATCHED_CONTROL_PROFILE_ID
    rows = smoke["per_schedule_shift"]
    assert [row["schedule_shift_pc"] for row in rows] == list(range(12))
    assert {row["applied_shift_pc"] for row in rows} == {0}
    assert all(row["finite"] is True for row in rows)
    assert all(row["identity_logits_equal"] is True for row in rows)
