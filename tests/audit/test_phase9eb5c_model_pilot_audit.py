from __future__ import annotations

import json
from pathlib import Path

from music_critic.experiments.analysisgnn.contracts import fingerprint
from scripts import audit_phase9eb5c_analysisgnn_model_pilot as audit_script


def test_source_free_model_pilot_audit_is_valid_and_deterministic() -> None:
    first = audit_script.audit()
    second = audit_script.audit()
    assert first == second
    assert first["valid"] is True
    assert first["model_implemented"] is True
    assert first["trainer_implemented"] is True
    assert first["cpu_smoke_passed"] is True
    assert first["real_train_coverage_smoke_passed"] is True
    assert first["cuda_smoke_passed"] is False
    assert first["ready_for_cuda_pilot"] is True
    assert first["pilot_not_run_reason"] == "cuda_unavailable"


def test_fixture_is_self_fingerprinted_and_does_not_fabricate_pilots() -> None:
    fixture = json.loads(audit_script.FIXTURE.read_text(encoding="utf-8"))
    observed = fixture.pop("fixture_fingerprint")
    assert fingerprint(fixture) == observed
    status = fixture["status"]
    assert status["c0_pilot_completed"] is False
    assert status["c1_pilot_completed"] is False
    assert status["comparison_completed"] is False
    assert fixture["pilot_summaries"] == {"C0": None, "C1": None}
    assert fixture["comparison"] is None


def test_test_lock_and_training_scope_are_explicit() -> None:
    result = audit_script.audit()
    assert result["test_evaluated"] is False
    assert result["test_targets_used_for_evaluation"] is False
    assert result["full_training_run"] is False
    assert result["multi_seed_run"] is False
    assert result["checks"]["test_closed"] is True
    assert result["checks"]["no_full_or_multiseed"] is True


def test_fixture_contains_exact_coverage_and_cuda_commands() -> None:
    fixture = json.loads(audit_script.FIXTURE.read_text(encoding="utf-8"))
    coverage = fixture["smoke_evidence"]["real_train_coverage"]
    assert coverage["finite_loss_head_count"] == 18
    assert coverage["nonzero_gradient_head_count"] == 18
    assert coverage["shared_encoder_nonzero_gradient"] is True
    assert len(fixture["cuda_commands"]) == 3
    assert all("CUBLAS_WORKSPACE_CONFIG=:4096:8" in row for row in fixture["cuda_commands"])


def test_fixture_is_at_the_required_source_free_path() -> None:
    assert audit_script.FIXTURE == (
        Path(__file__).parents[1]
        / "fixtures/analysisgnn/phase9eb5c_model_pilot.json"
    )
    assert audit_script.FIXTURE.is_file()
