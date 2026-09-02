from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
)
from music_critic.experiments.analysisgnn.full_orbit_training import (
    EXPECTED_VALID_SHIFT_DISTRIBUTION,
    FULL_ORBIT_DRAW_BUDGET,
    FULL_ORBIT_PROFILE_ID,
    FULL_ORBIT_UPDATE_BUDGET,
    FULL_ORBIT_WARMUP_UPDATES,
    FullOrbitRuntimeConfig,
    FullOrbitSampler,
    build_full_orbit_optimizer_scheduler,
    build_full_orbit_table,
    check_full_orbit_fixture,
    full_orbit_profile_contract,
    full_orbit_table_contract,
)


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/analysisgnn/phase9eb5h_full_orbit_profile.json"
)


def _table():
    """Build a source-free table with the exact sealed production orbit shape."""

    component_records: dict[str, tuple[str, ...]] = {}
    valid_shifts: dict[str, tuple[int, ...]] = {}
    record_index = 0
    for shift_count, frequency in sorted(EXPECTED_VALID_SHIFT_DISTRIBUTION.items()):
        for _ in range(frequency):
            record_id = f"record-{record_index:04d}"
            component_records[f"component-{record_index:04d}"] = (record_id,)
            valid_shifts[record_id] = tuple(range(shift_count))
            record_index += 1
    return build_full_orbit_table(component_records, valid_shifts)


def test_c2_orbit_has_exact_eligible_pairs_distribution_and_identity() -> None:
    sealed = check_full_orbit_fixture(FIXTURE)["orbit_table"]
    table = _table()
    contract = full_orbit_table_contract(table)
    assert len(table) == 15_389
    assert contract["base_train_records"] == 1_295
    assert contract["nominal_record_shift_pairs"] == 15_540
    assert contract["excluded_train_pairs"] == 151
    assert contract["identity_pairs"] == 1_295
    assert contract["identity_fraction"] == 1_295 / 15_389
    assert Counter(Counter(row.record_id for row in table).values()) == Counter(
        EXPECTED_VALID_SHIFT_DISTRIBUTION
    )
    assert sealed["table_fingerprint"] == (
        "133983af065f28faab2258e8e2a1de057c87e34cdf214e494fa19a1e76987661"
    )
    assert sealed["source_split_changed"] is False


def test_full_orbit_epoch_has_no_replacement_or_omissions() -> None:
    table = _table()
    sampler = FullOrbitSampler(table)
    epoch = [sampler.peek(offset) for offset in range(len(table))]
    assert len({(row.record_id, row.shift_pc) for row in epoch}) == len(table)
    assert {(row.record_id, row.shift_pc) for row in epoch} == {
        (row.record_id, row.shift_pc) for row in table
    }
    assert sum(row.shift_pc == 0 for row in epoch) == 1_295
    assert sampler.peek(len(table)).orbit_epoch == 1


def test_partial_final_epoch_is_explicit_and_deterministic() -> None:
    sealed = check_full_orbit_fixture(FIXTURE)
    table = _table()
    a = FullOrbitSampler(table)
    b = FullOrbitSampler(table)
    start = (FULL_ORBIT_DRAW_BUDGET // len(table)) * len(table)
    count = FULL_ORBIT_DRAW_BUDGET % len(table)
    assert count == sealed["partial_final_epoch_draw_count"] == 9_165
    first = [a.peek(start + index) for index in range(count)]
    second = [b.peek(start + index) for index in range(count)]
    assert first == second
    assert len({(row.record_id, row.shift_pc) for row in first}) == count


def test_c2_budget_scheduler_and_test_lock_are_sealed() -> None:
    config = FullOrbitRuntimeConfig()
    profile = full_orbit_profile_contract()
    assert config.profile_id == FULL_ORBIT_PROFILE_ID
    assert config.applied_update_budget == FULL_ORBIT_UPDATE_BUDGET == 120_000
    assert config.train_draw_budget == FULL_ORBIT_DRAW_BUDGET == 240_000
    assert config.warmup_applied_updates == FULL_ORBIT_WARMUP_UPDATES == 6_000
    assert config.test_enabled is False
    assert profile["orbit_epochs_at_budget"] == 240_000 / 15_389
    assert profile["from_scratch"] is True
    assert profile["resume_from_c1_checkpoint"] is False
    model = CorrectedAnalysisGNNModel()
    optimizer, scheduler = build_full_orbit_optimizer_scheduler(model)
    assert optimizer.defaults["lr"] == 0.005
    assert scheduler.lr_lambdas[0](0) == 1 / 6_000
    assert scheduler.lr_lambdas[0](120_000) == 0.0
