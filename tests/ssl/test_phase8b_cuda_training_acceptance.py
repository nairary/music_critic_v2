from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from music_critic.ssl.multilevel import PHASE8B_NEW_OBJECTIVE_FAMILIES
from music_critic.ssl.phase8b_cuda_acceptance import (
    Phase8BCudaAcceptanceError,
    _compare_precision_reports,
    _official_command,
    _require_exact_head,
    _validate_training_report,
)


def _group(*, active: bool) -> dict[str, object]:
    return {
        "parameter_count": 12,
        "with_gradient_count": 12 if active else 0,
        "finite_gradient_count": 12 if active else 0,
        "nonzero_gradient_count": 12 if active else 0,
        "changed_parameter_count": 12 if active else 0,
    }


def _report() -> dict[str, object]:
    groups = {
        family: _group(active=family == "onset_latent")
        for family in PHASE8B_NEW_OBJECTIVE_FAMILIES
    }
    groups["online_encoder"] = _group(active=True)
    return {
        "mechanics_acceptance": {"passed": True},
        "accounting": {
            "optimizer_step_attempt_count": 12,
            "optimizer_step_applied_count": 11,
            "optimizer_step_skipped_count": 1,
            "optimizer_step_count": 11,
        },
        "loss_decreased": True,
        "initial": {
            "total_ssl_loss": 1.1,
            "input_batch_fingerprints": ["a" * 64],
            "prepared_binding_fingerprints": ["b" * 64],
            "prepared_objective_binding_fingerprints": ["f" * 64],
            "objective": {
                "family_denominators": {"onset_latent": 10},
                "family_view_pass_counts": {"onset_latent": 1},
            },
        },
        "final": {
            "total_ssl_loss": 0.4,
            "input_batch_fingerprints": ["a" * 64],
            "prepared_binding_fingerprints": ["b" * 64],
            "prepared_objective_binding_fingerprints": ["f" * 64],
            "objective": {
                "family_denominators": {"onset_latent": 10},
                "family_view_pass_counts": {"onset_latent": 1},
            },
        },
        "input_fixture_fingerprint": "c" * 64,
        "model_state_fingerprints": {
            "initial": "d" * 64,
            "final": "e" * 64,
            "changed": True,
        },
        "resolved_mask_policies": ["onset_pitch_descendants"],
        "optimizer_parameter_coverage": {
            "all_trainable_parameters_present_exactly_once": True,
        },
        "gradient_coverage": {
            "acceptance": {"passed": True},
            "groups": groups,
        },
        "cuda_peak_memory": {
            "available": True,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
        },
    }


def test_official_command_covers_exact_cuda_amp_mode_and_artifact_path() -> None:
    command = _official_command(
        Path("/tmp/phase8b-rtx/onset"),
        mode="onset_only",
        amp=True,
        seed=42,
        steps=12,
    )
    assert "+phase8b_objective=onset_only" in command
    assert "+phase8b_masking=onset_only" in command
    assert "device=cuda" in command
    assert "device.amp=true" in command
    assert "experiment.steps=12" in command


def test_runner_cli_requires_exact_expected_head() -> None:
    script = Path("scripts/accept_phase8b_cuda_amp_training.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--expected-head", required=True)' in script
    assert "expected_head=arguments.expected_head" in script
    _require_exact_head(expected_head="a" * 40, actual_head="a" * 40)
    with pytest.raises(
        Phase8BCudaAcceptanceError,
        match="phase8b.cuda.exact_head_mismatch",
    ):
        _require_exact_head(expected_head="a" * 40, actual_head="b" * 40)


def test_training_report_validation_fails_closed_on_zero_update_bug() -> None:
    report = _report()
    assert not _validate_training_report(
        report, mode="onset_only", expected_steps=12
    )
    broken = deepcopy(report)
    broken["accounting"]["optimizer_step_applied_count"] = 0
    broken["accounting"]["optimizer_step_skipped_count"] = 12
    broken["loss_decreased"] = False
    broken["model_state_fingerprints"]["changed"] = False
    broken["gradient_coverage"]["groups"]["online_encoder"] = _group(
        active=False
    )
    broken["gradient_coverage"]["groups"]["onset_latent"] = _group(
        active=False
    )
    failures = _validate_training_report(
        broken, mode="onset_only", expected_steps=12
    )
    assert "no_optimizer_step_applied" in failures
    assert "bounded_loss_did_not_decrease" in failures
    assert "model_parameters_unchanged" in failures
    assert "online_encoder_gradient_or_update_invalid" in failures
    assert "active_head_invalid:onset_latent" in failures


def test_fp32_amp_parity_uses_structural_equality_and_numeric_tolerance() -> None:
    fp32 = _report()
    amp = deepcopy(fp32)
    amp["initial"]["total_ssl_loss"] = 1.105
    evidence = _compare_precision_reports(
        fp32,
        amp,
        relative_tolerance=0.02,
        absolute_tolerance=0.02,
    )
    assert evidence["passed"]
    assert not evidence["bit_exact_required"]
    final_diverged = deepcopy(amp)
    final_diverged["final"]["total_ssl_loss"] = 0.5
    assert not _compare_precision_reports(
        fp32,
        final_diverged,
        relative_tolerance=0.02,
        absolute_tolerance=0.02,
    )["passed"]
    amp["initial"]["objective"]["family_denominators"] = {
        "onset_latent": 9
    }
    assert not _compare_precision_reports(
        fp32,
        amp,
        relative_tolerance=0.02,
        absolute_tolerance=0.02,
    )["passed"]
