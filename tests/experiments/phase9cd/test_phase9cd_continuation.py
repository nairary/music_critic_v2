from __future__ import annotations

from pathlib import Path

import pytest
import torch

from music_critic.experiments.phase9cc.runner import _write
from music_critic.experiments.phase9cc.training import model_state_fingerprint
from music_critic.experiments.phase9cc_continuation.contracts import (
    file_sha256,
    fingerprint,
)
from music_critic.experiments.phase9cc_continuation.training import _restore_parent
from music_critic.experiments.phase9cd import contracts as phase9cd_contracts
from music_critic.experiments.phase9cd.contracts import CELLS, MILESTONES
from music_critic.experiments.phase9cd.runner import _delta
from music_critic.training.checkpoint import capture_rng_state
from music_critic.training.models import model_contract_metadata
from tests.models.test_onset_bigru_decoder import _model


def _parent_checkpoint(path: Path):
    model = _model("onset_bigru")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    payload = {
        "metadata": {
            "training_checkpoint_version": "1.0.0",
            "model_contract": model_contract_metadata(model),
            "resume_boundary": "epoch_only",
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "next_epoch": 1,
    }
    torch.save(payload, path)
    return payload


def test_real_bigru_parent_restores_full_state_and_rejects_cross_kind(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.pt"
    payload = _parent_checkpoint(checkpoint)
    binding = {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "model_state_fingerprint": model_state_fingerprint(payload["model_state"]),
        "attempted_updates": 4,
        "skipped_updates": 0,
    }
    plan = {
        "protocol": {
            "parent_binding": {"kind": "phase9cb"},
            "schedule": {"start_applied_update": 4},
        }
    }
    cell = {"cell_id": "scratch_onset_bigru", "parent_checkpoint": binding}
    model = _model("onset_bigru")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    restored = _restore_parent(
        plan,
        cell,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
    )
    assert model_state_fingerprint(model) == binding["model_state_fingerprint"]
    assert restored["optimizer_state"] == payload["optimizer_state"]

    wrong_model = _model("mlp")
    with pytest.raises(ValueError, match="resume.model_binding_invalid"):
        _restore_parent(
            plan,
            cell,
            model=wrong_model,
            optimizer=torch.optim.AdamW(wrong_model.parameters(), lr=0.0003),
            scheduler=None,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
        )


def test_phase9cd_inventory_metrics_and_rtx_interface() -> None:
    assert CELLS == ("scratch_onset_bigru", "ssl_onset_bigru")
    assert MILESTONES == (3000, 6000, 9000, 12000, 15000)
    delta = _delta(
        {
            "mean_normalized_nll": 0.4,
            "mean_macro_f1": 0.3,
            "mean_balanced_accuracy": 0.2,
            "mean_accuracy": 0.5,
            "mean_prediction_entropy": 0.6,
        },
        {
            "mean_normalized_nll": 0.5,
            "mean_macro_f1": 0.2,
            "mean_balanced_accuracy": 0.1,
            "mean_accuracy": 0.4,
            "mean_prediction_entropy": 0.7,
        },
    )
    assert delta["mean_normalized_nll"] == pytest.approx(-0.1)
    assert delta["mean_prediction_entropy"] == pytest.approx(-0.1)
    script = Path("scripts/run_phase9cc_rtx3090_convergence.sh").read_text()
    assert "scratch_onset_bigru,ssl_onset_bigru" in script
    assert "3000,6000,9000,12000,15000" in script
    assert "--mlp-reference-root" in script


def test_sealed_mlp_reference_verification_does_not_rebuild_old_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "protocol": {
            "git_head": "m" * 40,
            "test_lock": {
                "test_inference": False,
                "test_targets_accessed": False,
                "test_metrics_accessed": False,
                "test_unlock": False,
            },
        }
    }
    report = {"test_access": False}
    report = {**report, "fingerprint": fingerprint(report)}
    _write(tmp_path / "continuation_plan.json", plan)
    _write(tmp_path / "convergence_report.json", report)
    files = {
        name: file_sha256(tmp_path / name)
        for name in ("continuation_plan.json", "convergence_report.json")
    }
    manifest = {"contract_version": "1.0.0", "files": files, "file_count": 2}
    manifest = {**manifest, "fingerprint": fingerprint(manifest)}
    _write(tmp_path / "manifest.json", manifest)
    (tmp_path / "payload.sha256").write_text(
        f"{file_sha256(tmp_path / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(phase9cd_contracts, "MLP_SHA", "m" * 40)
    monkeypatch.setattr(
        phase9cd_contracts, "MLP_REPORT_FINGERPRINT", report["fingerprint"]
    )
    monkeypatch.setattr(
        phase9cd_contracts, "MLP_MANIFEST_FINGERPRINT", manifest["fingerprint"]
    )
    verified = phase9cd_contracts.verify_sealed_mlp_reference(tmp_path)
    assert verified["report"] == report
