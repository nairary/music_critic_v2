from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from music_critic.experiments.analysisgnn import corrected_training
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
    run_corrected_validation,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_COMPARISON_SCHEMA,
    FULL_RUN_SUMMARY_SCHEMA,
    FullTrainingContract,
    build_full_comparison,
    full_run_root_name,
    full_runtime_config,
    full_training_contract,
    full_validation_updates,
)

RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/run_phase9eb5d_analysisgnn_full.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("phase9eb5d_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _summary(profile: str) -> dict[str, object]:
    validation = _validation(profile)
    payload: dict[str, object] = {
        "schema": FULL_RUN_SUMMARY_SCHEMA,
        "valid": True,
        "phase": "9E-B5D",
        "profile": profile,
        "profile_id": FullTrainingContract().to_dict()["profiles"][profile],
        "seed": 17,
        "batch_size": 2,
        "applied_updates": 10_000,
        "train_draws": 20_000,
        "validation_updates": list(full_validation_updates()),
        "initial_model_state_fingerprint": "paired-initial",
        "record_schedule_fingerprint": "paired-records",
        "transposition_schedule_fingerprint": f"shifts-{profile}",
        "best_primary_score": 0.2 if profile == "C0" else 0.24,
        "best_update": 9_500 if profile == "C0" else 10_000,
        "final_primary_score": validation[-1]["corrected_primary_macro_score"],
        "nan_count": 0,
        "overflow_count": 0,
        "skipped_update_count": 0,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
        "full_training_run": True,
        "multi_seed_run": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _validation(profile: str) -> list[dict[str, object]]:
    offset = 0.01 if profile == "C1" else 0.0
    return [
        {
            "applied_update": update,
            "corrected_primary_macro_score": 0.1 + update / 100_000 + offset,
        }
        for update in full_validation_updates()
    ]


def test_full_contract_freezes_the_approved_10000_update_screen() -> None:
    contract = full_training_contract()
    assert contract["phase"] == "9E-B5D"
    assert contract["seed"] == 17
    assert contract["batch_size"] == 2
    assert contract["applied_update_budget"] == 10_000
    assert contract["train_draw_budget"] == 20_000
    assert contract["draws_per_epoch"] == 1_295
    assert contract["sampler_epochs"] == pytest.approx(20_000 / 1_295)
    assert contract["warmup_applied_updates"] == 500
    assert contract["scheduler"] == "linear_warmup_then_cosine_decay"
    assert contract["validation_interval"] == 500
    assert contract["checkpoint_interval"] == 100
    assert contract["validation_updates"] == list(range(0, 10_001, 500))
    assert len(full_validation_updates()) == 21
    assert contract["early_stopping"] is False
    assert contract["test_enabled"] is False
    assert contract["profile_o_enabled"] is False
    assert contract["multi_seed_claim"] is False


def test_full_runtime_configs_are_a_paired_cuda_contract() -> None:
    c0 = full_runtime_config("C0").to_dict()
    c1 = full_runtime_config("C1").to_dict()
    for config in (c0, c1):
        assert config["seed"] == 17
        assert config["device"] == "cuda"
        assert config["batch_size"] == 2
        assert config["applied_update_budget"] == 10_000
        assert config["validation_interval"] == 500
        assert config["test_enabled"] is False
        assert config["early_stopping"] is False
    assert c0["transposition_enabled"] is False
    assert c1["transposition_enabled"] is True
    assert full_run_root_name("C0") == "c0-seed17-full-u10000"
    assert full_run_root_name("C1") == "c1-seed17-full-u10000"
    with pytest.raises(CorrectedTrainingError, match="profile_invalid"):
        full_runtime_config("O")


def test_comparison_requires_exact_pairing_and_reports_both_deltas() -> None:
    summaries = {profile: _summary(profile) for profile in ("C0", "C1")}
    validation = {profile: _validation(profile) for profile in ("C0", "C1")}
    comparison = build_full_comparison(
        summaries=summaries, validation=validation
    )
    assert comparison["schema"] == FULL_COMPARISON_SCHEMA
    assert comparison["applied_updates"] == 10_000
    assert comparison["train_draws"] == 20_000
    assert comparison["initial_state_fingerprints_equal"] is True
    assert comparison["record_schedule_fingerprints_equal"] is True
    assert comparison["transposition_schedule_fingerprints_differ"] is True
    assert comparison["final_primary_score_delta_C1_minus_C0"] == pytest.approx(
        0.01
    )
    assert comparison["best_primary_score_delta_C1_minus_C0"] == pytest.approx(
        0.04
    )
    assert comparison["statistical_improvement_claim"] is False
    assert comparison["test_evaluated"] is False


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    [
        ("C1", "initial_model_state_fingerprint", "other", "causal_pairing_failed"),
        ("C1", "record_schedule_fingerprint", "other", "causal_pairing_failed"),
        ("C1", "transposition_schedule_fingerprint", "shifts-C0", "causal_pairing_failed"),
        ("C1", "test_evaluated", True, "summary_invalid"),
        ("C1", "best_primary_score", None, "best_checkpoint_invalid"),
    ],
)
def test_comparison_fails_closed_on_invalid_evidence(
    target: str, key: str, value: object, message: str
) -> None:
    summaries = {profile: _summary(profile) for profile in ("C0", "C1")}
    summaries[target][key] = value
    summaries[target].pop("fingerprint")
    summaries[target]["fingerprint"] = fingerprint(summaries[target])
    with pytest.raises(CorrectedTrainingError, match=message):
        build_full_comparison(
            summaries=summaries,
            validation={profile: _validation(profile) for profile in ("C0", "C1")},
        )


def test_comparison_rejects_incomplete_validation_schedule() -> None:
    validation = {profile: _validation(profile) for profile in ("C0", "C1")}
    validation["C1"].pop()
    with pytest.raises(CorrectedTrainingError, match="validation_schedule_mismatch"):
        build_full_comparison(
            summaries={profile: _summary(profile) for profile in ("C0", "C1")},
            validation=validation,
        )


def test_resume_reconciliation_truncates_only_rows_after_checkpoint(tmp_path) -> None:
    training = [
        {"applied_update": update} for update in range(1, 206)
    ]
    validation = [
        {"applied_update": 0},
        {"applied_update": 500},
    ]
    runner._rewrite_jsonl(tmp_path / "training_metrics.jsonl", training)
    runner._rewrite_jsonl(tmp_path / "validation_metrics.jsonl", validation)
    runner._reconcile_resume_ledgers(tmp_path, 200)
    assert [
        row["applied_update"]
        for row in runner._read_jsonl(tmp_path / "training_metrics.jsonl")
    ] == list(range(1, 201))
    assert runner._read_jsonl(tmp_path / "validation_metrics.jsonl") == [
        {"applied_update": 0}
    ]


def test_resume_and_fresh_output_guards_fail_closed(tmp_path) -> None:
    runner._rewrite_jsonl(
        tmp_path / "training_metrics.jsonl",
        [{"applied_update": 1}, {"applied_update": 3}],
    )
    runner._rewrite_jsonl(
        tmp_path / "validation_metrics.jsonl", [{"applied_update": 0}]
    )
    with pytest.raises(CorrectedTrainingError, match="training_ledger_mismatch"):
        runner._reconcile_resume_ledgers(tmp_path, 3)
    with pytest.raises(CorrectedTrainingError, match="output_exists"):
        runner._guard_fresh_run(tmp_path)


def test_validation_wrapper_rejects_non_frozen_record_count_before_loading(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        corrected_training,
        "frozen_split_assignments",
        lambda _paths: {"only": {"split": "validation"}},
    )
    with pytest.raises(CorrectedTrainingError, match="validation_record_count_mismatch"):
        run_corrected_validation(object(), device="cpu")  # type: ignore[arg-type]


def test_contract_is_stable_under_copy() -> None:
    assert FullTrainingContract().to_dict() == copy.deepcopy(full_training_contract())
    json.dumps(full_training_contract(), sort_keys=True)
