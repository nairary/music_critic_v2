from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
    build_source_free_fixture,
)
from music_critic.experiments.analysisgnn import validation_eligibility_repair as repair


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_free_identity_is_always_validation_eligible() -> None:
    batch, sidecar = build_source_free_fixture()
    shifts = repair.valid_shifts_for_validation_record(
        batch.raw_graph_batch,
        sidecar,
    )
    assert shifts
    assert shifts == tuple(sorted(set(shifts)))
    assert 0 in shifts
    assert set(shifts) <= set(range(12))


def test_validation_eligibility_contract_is_complete_and_test_locked() -> None:
    eligibility = {
        f"validation-{index:03d}": (0, 1, 6)
        for index in range(162)
    }
    contract = repair.validation_eligibility_repair_contract(eligibility)
    assert contract["record_count"] == 162
    assert contract["identity_record_count"] == 162
    assert contract["eligible_record_shift_pairs"] == 486
    assert contract["per_shift_record_count"]["0"] == 162
    assert contract["per_shift_record_count"]["1"] == 162
    assert contract["per_shift_record_count"]["6"] == 162
    assert contract["train_b5a_eligibility_used"] is False
    assert contract["training_or_optimizer_step_executed"] is False
    assert contract["test_loader_created"] is False
    assert contract["test_targets_read"] is False
    assert contract["test_metrics_computed"] is False


def test_validation_eligibility_contract_rejects_missing_identity() -> None:
    eligibility = {
        f"validation-{index:03d}": (0, 1)
        for index in range(162)
    }
    eligibility["validation-017"] = (1,)
    with pytest.raises(
        CorrectedTrainingError,
        match="analysisgnn.full_orbit.validation_eligibility_invalid",
    ):
        repair.validation_eligibility_repair_contract(eligibility)


def test_validation_derivation_reads_validation_only(monkeypatch) -> None:
    assignments = {
        **{
            f"validation-{index:03d}": {"split": "validation"}
            for index in range(162)
        },
        "train-sentinel": {"split": "train"},
        "test-sentinel": {"split": "test"},
    }
    loaded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        repair,
        "frozen_split_assignments",
        lambda _paths: assignments,
    )

    def fake_load(record_id: str, *, split: str, paths: object):
        loaded.append((record_id, split))
        return (
            SimpleNamespace(raw_graph_batch=object()),
            {"record_id": record_id},
        )

    monkeypatch.setattr(repair, "load_production_record", fake_load)
    monkeypatch.setattr(
        repair,
        "valid_shifts_for_validation_record",
        lambda _raw, _sidecar: (0, 2),
    )

    result = repair.validation_valid_shifts()
    assert len(result) == 162
    assert set(result.values()) == {(0, 2)}
    assert len(loaded) == 162
    assert all(split == "validation" for _, split in loaded)
    assert not any(record_id.endswith("sentinel") for record_id, _ in loaded)


def test_c2_finalizer_wrapper_replaces_only_diagnostic_entrypoint() -> None:
    wrapper = _load_script(
        REPO_ROOT / "scripts/run_phase9eb5j_finalize_c2.py",
        "phase9eb5j_c2_wrapper_test",
    )
    runner = wrapper._patched_runner()
    assert (
        runner.run_full_orbit_diagnostic_validation
        is repair.run_repaired_full_orbit_diagnostic_validation
    )
    assert runner.FULL_ORBIT_UPDATE_BUDGET == 120_000
    assert runner.FULL_ORBIT_DRAW_BUDGET == 240_000


def test_c0_wrapper_replaces_only_diagnostic_entrypoint() -> None:
    wrapper = _load_script(
        REPO_ROOT / "scripts/run_phase9eb5j_analysisgnn_c0_orbit_matched.py",
        "phase9eb5j_c0_wrapper_test",
    )
    runner = wrapper._patched_runner()
    assert (
        runner.run_full_orbit_diagnostic_validation
        is repair.run_repaired_full_orbit_diagnostic_validation
    )
    assert runner.FULL_ORBIT_UPDATE_BUDGET == 120_000
    assert runner.FULL_ORBIT_DRAW_BUDGET == 240_000
