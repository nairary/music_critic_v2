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
from music_critic.experiments.phase9cc_continuation.runner import (
    _bound_parent_model_fingerprint,
)
from music_critic.experiments.phase9cd import contracts as phase9cd_contracts
from music_critic.experiments.phase9cd.contracts import CELLS, MILESTONES
from music_critic.experiments.phase9cd import runner as phase9cd_runner
from music_critic.experiments.phase9cd.runner import _delta, _milestone_transitions
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
    assert _bound_parent_model_fingerprint(payload, cell) == binding[
        "model_state_fingerprint"
    ]

    tampered_binding = {
        **cell,
        "parent_checkpoint": {
            **binding,
            "model_state_fingerprint": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="preflight.model_fingerprint_mismatch"):
        _bound_parent_model_fingerprint(payload, tampered_binding)

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
    assert _milestone_transitions(MILESTONES) == [
        (3000, 6000),
        (6000, 9000),
        (9000, 12000),
        (12000, 15000),
        (3000, 15000),
    ]
    with pytest.raises(ValueError, match="milestones_invalid"):
        _milestone_transitions((3000, 6000, 6000, 15000))


def _completed_failed_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "failed"
    milestones = (3, 4, 5, 6, 7)
    cells = ("scratch_onset_bigru", "ssl_onset_bigru")
    for cell_id in cells:
        directory = root / "cells" / cell_id
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "milestones").mkdir()
        training = {
            "complete": True,
            "applied_updates": 7,
            "attempted_updates": 7,
            "skipped_updates": 0,
            "sample_schedule_position": 7,
        }
        training = {**training, "fingerprint": fingerprint(training)}
        _write(directory / "training_report.json", training)
        (directory / "train_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
        rows = []
        for update in milestones:
            checkpoint = directory / "checkpoints" / f"update-{update}.pt"
            torch.save({"model_state": {}}, checkpoint)
            report = {
                "membership_fingerprint": "v" * 64,
                "aggregate": {},
                "tasks": {},
            }
            report = {**report, "fingerprint": fingerprint(report)}
            report_path = directory / "milestones" / f"update-{update}.json"
            _write(report_path, report)
            rows.append(
                {
                    "update": update,
                    "checkpoint_source": "continuation",
                    "checkpoint_path": f"checkpoints/update-{update}.pt",
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "validation_report_path": f"milestones/update-{update}.json",
                    "validation_report_sha256": file_sha256(report_path),
                    "validation_report_fingerprint": report["fingerprint"],
                    "validation_membership_fingerprint": "v" * 64,
                }
            )
        validation = {"milestones": rows}
        validation = {**validation, "fingerprint": fingerprint(validation)}
        _write(directory / "validation_milestones.json", validation)
    protocol = {
        "git_head": "t" * 40,
        "fingerprint": "q",
        "parent_binding": {"manifest_fingerprint": "m"},
        "schedule": {
            "validation_milestones": list(milestones),
            "target_applied_update": 7,
        },
    }
    _write(
        root / "continuation_plan.json",
        {
            "fingerprint": "p",
            "protocol": protocol,
            "cells": [{"cell_id": cell_id} for cell_id in cells],
        },
    )
    failure = tmp_path / "historical.log"
    failure.write_text(
        "ValueError: zip() argument 2 is shorter than argument 1\n",
        encoding="utf-8",
    )
    return root, failure


def test_recovery_finalizes_completed_fixture_without_training_or_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, failure = _completed_failed_root(tmp_path)
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_TRAINING_SHA", "t" * 40)
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_PLAN_FINGERPRINT", "p")
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_PROTOCOL_FINGERPRINT", "q")
    monkeypatch.setattr(phase9cd_runner, "PARENT_MANIFEST_FINGERPRINT", "m")
    monkeypatch.setattr(phase9cd_runner, "MILESTONES", (3, 4, 5, 6, 7))
    monkeypatch.setattr(phase9cd_runner, "TARGET_UPDATE", 7)
    monkeypatch.setattr(phase9cd_runner, "_git_head", lambda: "f" * 40, raising=False)
    monkeypatch.setattr(phase9cd_contracts, "_git_head", lambda: "f" * 40)
    calls = {"training": 0, "validation": 0, "aggregate": 0}

    def aggregate(output, plan):
        calls["aggregate"] += 1
        report = {"fingerprint": "r", "test_access": False}
        _write(output / "bigru_convergence_report.json", report)
        _write(output / "decoder_comparison_report.json", report)
        return report

    monkeypatch.setattr(phase9cd_runner, "aggregate", aggregate)
    monkeypatch.setattr(
        phase9cd_runner,
        "verify_bundle",
        lambda output, expected_sha=None: {"status": "verified", "sha": expected_sha},
    )
    result = phase9cd_runner.recover_finalize(
        root,
        finalizer_sha="f" * 40,
        training_sha="t" * 40,
        historical_failure_log=failure,
    )
    assert result["status"] == "verified"
    assert calls == {"training": 0, "validation": 0, "aggregate": 1}
    provenance = phase9cd_runner._read(root / "finalization_provenance.json")
    assert provenance["training_reexecuted"] is False
    assert provenance["validation_reexecuted"] is False
    assert (root / "manifest.json").is_file()


@pytest.mark.parametrize("mutation", ("missing_milestone", "missing_checkpoint", "wrong_training_sha"))
def test_recovery_rejects_incomplete_or_mismatched_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, failure = _completed_failed_root(tmp_path)
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_TRAINING_SHA", "t" * 40)
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_PLAN_FINGERPRINT", "p")
    monkeypatch.setattr(phase9cd_runner, "RECOVERY_PROTOCOL_FINGERPRINT", "q")
    monkeypatch.setattr(phase9cd_runner, "PARENT_MANIFEST_FINGERPRINT", "m")
    monkeypatch.setattr(phase9cd_runner, "MILESTONES", (3, 4, 5, 6, 7))
    monkeypatch.setattr(phase9cd_runner, "TARGET_UPDATE", 7)
    monkeypatch.setattr(phase9cd_contracts, "_git_head", lambda: "f" * 40)
    if mutation == "missing_milestone":
        validation_path = root / "cells/scratch_onset_bigru/validation_milestones.json"
        validation = phase9cd_runner._read(validation_path)
        validation["milestones"].pop()
        unsigned = dict(validation)
        unsigned.pop("fingerprint")
        validation["fingerprint"] = fingerprint(unsigned)
        _write(validation_path, validation)
    elif mutation == "missing_checkpoint":
        (root / "cells/scratch_onset_bigru/checkpoints/update-7.pt").unlink()
    training_sha = "x" * 40 if mutation == "wrong_training_sha" else "t" * 40
    with pytest.raises(ValueError):
        phase9cd_runner.recover_finalize(
            root,
            finalizer_sha="f" * 40,
            training_sha=training_sha,
            historical_failure_log=failure,
        )


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
