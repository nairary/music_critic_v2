from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest
import torch

from music_critic.experiments.dilemmadata import supervised_smoke as smoke
from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    DILEMMADATA_OPEN_TASK_IDS,
    DILEMMADATA_PU_TASK_IDS,
    DilemmadataHierarchicalModel,
    dilemmadata_model_contract_dict,
)
from music_critic.training.checkpoint import (
    TrainingCheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_phase9b2c_rtx3090_supervised_smoke.sh"
HEAD = "a" * 40


def _record(piece: str, dialect: str, component: str) -> dict[str, object]:
    return {
        "dataset_id": "dilemmadata",
        "piece_id": piece,
        "component_fingerprint": component * 64,
        "source_group_id": f"source:{piece}",
        "lineage_group_id": f"lineage:{piece}",
        "dialect": dialect,
        "target_cache_binding": {
            "cache_identity_fingerprint": "1" * 64,
            "target_bundle_fingerprint": "2" * 64,
            "artifact_sha256": "3" * 64,
            "raw_cache_key": "4" * 64,
            "canonical_artifact_sha256": "5" * 64,
        },
    }


def _memberships() -> tuple[dict[str, object], dict[str, object]]:
    train = smoke._with_fingerprint(
        {
            "contract_version": smoke.DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
            "split": "train",
            "selection_policy": "lexicographic_minimum_train_target_coverage_v1",
            "selection_may_read_labels": True,
            "replacement": False,
            "split_manifest_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
            "required_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
            "covered_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
            "dataset_counts": {"dilemmadata": 2},
            "dialect_counts": {"an": 1, "dlc": 1},
            "records": [
                _record("piece:train-an", "an", "a"),
                _record("piece:train-dlc", "dlc", "b"),
            ],
            "target_artifact_access": {
                "allowed_splits": ["train"],
                "observed_splits": ["train"],
                "artifact_read_count": 2,
                "test_target_accessed": False,
            },
        }
    )
    validation = smoke._with_fingerprint(
        {
            "contract_version": smoke.DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
            "split": "validation",
            "selection_policy": "seed17_identity_component_rank_v1",
            "selection_may_read_labels": False,
            "replacement": False,
            "requested_limit": 2,
            "split_manifest_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
            "dataset_counts": {"dilemmadata": 2},
            "dialect_counts": {"an": 1, "dlc": 1},
            "records": [
                _record("piece:validation-an", "an", "c"),
                _record("piece:validation-dlc", "dlc", "d"),
            ],
            "target_artifact_access_during_selection": {
                "artifact_read_count": 0,
                "validation_labels_read": False,
                "test_target_accessed": False,
            },
        }
    )
    return train, validation


def _evaluation(
    train: dict[str, object], validation: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    priors = smoke._with_fingerprint(
        {
            "contract_version": "1.0.0",
            "source_split": "train_only",
            "train_membership_fingerprint": train["fingerprint"],
            "tasks": {
                task_id: {
                    "class_counts": [1, 1],
                    "class_probabilities": [0.5, 0.5],
                    "majority_class_id": 0,
                    "source_entry_count": 2,
                }
                for task_id in DILEMMADATA_ACTIVE_TASK_IDS
            },
        }
    )
    tasks = {}
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        tasks[task_id] = {
            "available": True,
            "undefined_reason": None,
            "source_entry_count": 2,
            "expanded_row_count": 2,
            "nll": 0.7,
            "top1_accuracy": 0.5,
            "macro_f1": 0.5,
            "balanced_accuracy": 0.5,
            "top3_accuracy": 1.0 if task_id.endswith(".quality") else None,
            "record_metrics": {"piece": {"available": True}},
            "component_metrics": {"component": {"available": True}},
        }
    report = smoke._with_fingerprint(
        {
            "contract_version": "1.0.0",
            "split": "validation",
            "membership_fingerprint": validation["fingerprint"],
            "validation_only_default": True,
            "test_unlock_fingerprint": None,
            "train_prior_fingerprint": priors["fingerprint"],
            "tasks": tasks,
            "counts": {
                "source_entry_count": 8,
                "expanded_row_count": 8,
                "record_count": 2,
                "component_count": 2,
                "dataset_counts": {"dilemmadata": 2},
            },
            "entry_predictions": [],
        }
    )
    return priors, report


def _base_report(
    train: dict[str, object],
    validation: dict[str, object],
    priors: dict[str, object],
    evaluation: dict[str, object],
) -> dict[str, object]:
    groups = {"raw_encoder": True, **{task: True for task in DILEMMADATA_ACTIVE_TASK_IDS}}
    changes = {
        group: {
            "changed": True,
            "changed_parameter_count": 1,
            "changed_parameters": [f"{group}.weight"],
        }
        for group in groups
    }
    curve = [
        {
            "step": step,
            "total_loss": 4.0 - step * 0.01,
            "task_losses": {task: 1.0 for task in DILEMMADATA_ACTIVE_TASK_IDS},
            "gradient_norm_before_clip": 1.0,
            "gradients": {"all_gradients_finite": True},
            "amp_scale_before": 65536.0,
            "amp_scale_after": 65536.0,
            "optimizer_step_applied": True,
            "learning_rate_after": 3e-4,
        }
        for step in range(10)
    ]
    training_config = smoke._training_config(10)
    return {
        "contract_version": smoke.DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
        "phase": smoke.DILEMMADATA_SUPERVISED_SMOKE_PHASE,
        "expected_head": HEAD,
        "evidence_kind": "bounded_executable_mechanics_not_scientific_quality",
        "hardware": {
            "accelerator": smoke.DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME,
            "logical_cuda_index": 0,
            "device": "cuda:0",
            "cuda_available": True,
            "cuda_device_count": 1,
            "total_memory_bytes": 24 * 1024**3,
            "amp_enabled": True,
            "amp_dtype": "float16",
            "grad_scaler_enabled": True,
            "cpu_fallback": False,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
        },
        "bindings": {
            "raw_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
            "target_cache_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_INDEX_FINGERPRINT,
            "split_manifest_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
        },
        "model_contract_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT,
        "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
        "excluded_supervision": {
            "positive_unlabeled_task_ids": list(DILEMMADATA_PU_TASK_IDS),
            "positive_unlabeled_ce_heads": 0,
            "positive_unlabeled_ce_losses": 0,
            "open_string_task_ids": list(DILEMMADATA_OPEN_TASK_IDS),
            "open_string_heads": 0,
            "open_string_losses": 0,
        },
        "training_config": training_config,
        "train_membership_fingerprint": train["fingerprint"],
        "validation_membership_fingerprint": validation["fingerprint"],
        "candidate_first": {
            "prediction_completed_before_target_join": True,
            "prediction_equal_after_target_join": True,
            "initial": {},
            "target_mutation": {
                "verified": True,
                "mutated_target_row_count": 4,
                "candidate_identity_fingerprint_before": "1" * 64,
                "candidate_identity_fingerprint_after": "1" * 64,
                "raw_only_logits_fingerprint_before": "2" * 64,
                "raw_only_logits_fingerprint_after": "2" * 64,
            },
        },
        "source_entry_reduction": {
            "verified": True,
            "reduction": "candidate_rows_mean_per_source_entry_then_entries_mean_per_task_fixed_weight_sum",
            "tasks": {task: {} for task in DILEMMADATA_ACTIVE_TASK_IDS},
        },
        "optimization": {
            "attempted_update_count": 10,
            "applied_update_count": 10,
            "skipped_update_count": 0,
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "aggregate_nonzero_gradient_by_group": groups,
            "parameter_changes": changes,
            "initial_loss": 4.0,
            "minimum_loss": 3.91,
            "final_loss": 3.91,
            "curve": curve,
        },
        "checkpoint": {
            "artifact": "checkpoint.pt",
            "sha256": "0" * 64,
            "model_contract_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT,
            "raw_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
            "target_cache_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_INDEX_FINGERPRINT,
            "split_manifest_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
            "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
            "seed": 17,
            "train_membership_fingerprint": train["fingerprint"],
            "cuda_device": "cuda:0",
            "amp_dtype": "float16",
            "grad_scaler_state_present": True,
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "scratch_loaded_encoder_tensors": [],
            "scratch_supervised_heads_transferred": False,
            "scratch_ssl_heads_transferred": False,
            "reload_failure_atomic_contract": "training_checkpoint@1.0.0",
            "reload_bit_exact_raw_only_logits": True,
            "final_raw_only_logits_fingerprint": "3" * 64,
            "reloaded_raw_only_logits_fingerprint": "3" * 64,
        },
        "validation": {
            "artifact": "validation_report.json",
            "report_fingerprint": evaluation["fingerprint"],
            "official_evaluator": True,
            "split": "validation",
            "selection_uses_labels": False,
            "replacement": False,
            "train_only_baseline_fingerprint": priors["fingerprint"],
            "test_split_accessed": False,
            "test_targets_accessed": False,
            "test_metrics_computed": False,
            "test_unlock_used": False,
        },
        "runtime_access": {
            "accepted_inputs": [
                "raw_index",
                "raw_cache",
                "target_index",
                "target_cache",
                "split_manifest",
            ],
            "source_tsv_path_accepted": False,
            "raw_adapter_called": False,
            "alignment_oracle_called": False,
            "worker_count": 0,
            "adapter_modules_loaded_for_fail_closed_guards": [],
            "source_access_guard": {
                "guarded_functions": list(smoke._GUARDED_SOURCE_FUNCTIONS),
                "forbidden_call_count": 0,
            },
        },
        "claim_boundaries": {
            "bounded_mechanics_only": True,
            "scratch_vs_ssl_comparison": False,
            "representation_quality_claim": False,
            "calibration_or_significance_claim": False,
            "long_training_executed": False,
            "test_split_opened": False,
            "phase9c_started": False,
            "pdmx_started": False,
            "phase10_started": False,
            "legacy_used": False,
        },
        "git_preflight": {
            "expected_head": HEAD,
            "actual_head": HEAD,
            "worktree_clean_excluding_output_root": True,
        },
        "lifecycle": {
            "tracked_prediction_tensor_count": 12,
            "retained_prediction_tensor_count": 0,
            "allocated_bytes_after_cleanup": 0,
            "retained_cuda_tensor_count": 0,
        },
    }


def _checkpoint(
    root: Path, report: dict[str, object]
) -> None:
    model = DilemmadataHierarchicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    data_fingerprints = {
        **report["bindings"],
        "train_membership_fingerprint": report["train_membership_fingerprint"],
        "validation_membership_fingerprint": report[
            "validation_membership_fingerprint"
        ],
    }
    payload = {
        "metadata": {
            "training_checkpoint_version": "1.0.0",
            "model_contract": dilemmadata_model_contract_dict(model),
            "resolved_config_fingerprint": smoke._fingerprint(
                report["training_config"]
            ),
            "data_fingerprints": data_fingerprints,
            "data_fingerprint": smoke._fingerprint(data_fingerprints),
            "resume_boundary": "epoch_only",
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "next_epoch": 0,
        "best_validation_loss": None,
        "committed_metric_rows": 0,
        "rng_state": {
            "python": [],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        },
    }
    torch.save(payload, root / "checkpoint.pt")
    digest = sha256((root / "checkpoint.pt").read_bytes()).hexdigest()
    (root / "checkpoint.pt.sha256").write_text(digest + "\n", encoding="utf-8")
    report["checkpoint"]["sha256"] = digest


def _valid_evidence(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    train, validation = _memberships()
    priors, evaluation = _evaluation(train, validation)
    report = _base_report(train, validation, priors, evaluation)
    _checkpoint(root, report)
    report = smoke._with_fingerprint(report)
    for name, value in (
        ("run_report.json", report),
        ("train_membership.json", train),
        ("train_priors.json", priors),
        ("validation_membership.json", validation),
        ("validation_report.json", evaluation),
    ):
        (root / name).write_bytes(smoke._canonical_bytes(value) + b"\n")
    (root / "execution.log").write_text("synthetic CPU contract evidence\n")
    smoke.seal_evidence_directory(root, expected_head=HEAD)
    smoke.verify_evidence_directory(
        root, expected_head=HEAD, require_current_hardware=False
    )
    return root, report


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        (lambda value: value["hardware"].update(device="cpu"), "hardware"),
        (
            lambda value: value["optimization"].update(applied_update_count=0),
            "optimization",
        ),
        (
            lambda value: value["optimization"]["curve"][0].update(total_loss=float("inf")),
            "json_invalid",
        ),
        (
            lambda value: value["optimization"]["curve"][0]["gradients"].update(
                all_gradients_finite=False
            ),
            "optimization_curve",
        ),
        (
            lambda value: value["checkpoint"].update(reload_bit_exact_raw_only_logits=False),
            "checkpoint",
        ),
        (
            lambda value: value["bindings"].update(target_cache_index_fingerprint="f" * 64),
            "report_contract",
        ),
        (
            lambda value: value["bindings"].update(split_manifest_fingerprint="e" * 64),
            "report_contract",
        ),
        (
            lambda value: value["validation"].update(test_unlock_used=True),
            "validation",
        ),
        (
            lambda value: value.update(active_task_ids=list(DILEMMADATA_ACTIVE_TASK_IDS[:-1])),
            "report_contract",
        ),
    ),
)
def test_report_rejects_wrong_hardware_updates_nonfinite_and_bindings(
    mutation, category: str
) -> None:
    train, validation = _memberships()
    priors, evaluation = _evaluation(train, validation)
    report = _base_report(train, validation, priors, evaluation)
    mutation(report)
    if category == "json_invalid":
        with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match=category):
            smoke._with_fingerprint(report)
        return
    forged = smoke._with_fingerprint(report)
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match=category):
        smoke.validate_smoke_report(forged)


def test_membership_rejects_validation_label_selection_and_test_access() -> None:
    _, validation = _memberships()
    payload = dict(validation)
    payload.pop("fingerprint")
    payload["selection_may_read_labels"] = True
    payload["target_artifact_access_during_selection"] = {
        "artifact_read_count": 1,
        "validation_labels_read": True,
        "test_target_accessed": True,
    }
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="validation_membership"):
        smoke._validate_membership(smoke._with_fingerprint(payload), split="validation")


def test_source_access_guard_is_fail_closed_and_restored() -> None:
    from music_critic.adapters import dilemmadata as raw_adapter

    original = raw_adapter.discover_dilemmadata_corpus
    evidence, restore = smoke._install_source_access_guards()
    try:
        with pytest.raises(
            smoke.DilemmadataSupervisedSmokeError,
            match="source_or_oracle_access_forbidden",
        ):
            raw_adapter.discover_dilemmadata_corpus(Path("unused"))
        assert evidence["forbidden_call_count"] == 1
        assert len(evidence["guarded_functions"]) == 4
    finally:
        restore()
    assert raw_adapter.discover_dilemmadata_corpus is original


def test_missing_and_malformed_input_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="raw_index"):
        smoke._require_regular(
            tmp_path / "missing.json",
            directory=False,
            category="dilemmadata.smoke.raw_index_invalid",
        )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="raw_index"):
        smoke._require_regular(
            directory,
            directory=False,
            category="dilemmadata.smoke.raw_index_invalid",
        )


def test_forged_resealed_artifact_fails_semantic_verification(tmp_path: Path) -> None:
    root, _ = _valid_evidence(tmp_path / "evidence")
    report = json.loads((root / "run_report.json").read_text())
    report.pop("fingerprint")
    report["optimization"]["applied_update_count"] = 0
    (root / "run_report.json").write_bytes(
        smoke._canonical_bytes(smoke._with_fingerprint(report)) + b"\n"
    )
    (root / "artifact_manifest.json").unlink()
    smoke.seal_evidence_directory(root, expected_head=HEAD)
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="optimization"):
        smoke.verify_evidence_directory(
            root, expected_head=HEAD, require_current_hardware=False
        )


def test_bundle_rejects_incomplete_and_unsafe_tar(tmp_path: Path) -> None:
    root, _ = _valid_evidence(tmp_path / "evidence")
    tar_path = tmp_path / "evidence.tar"
    sidecar = tmp_path / "evidence.tar.sha256"
    smoke.pack_evidence_bundle(
        root,
        tar_path=tar_path,
        sidecar_path=sidecar,
        expected_head=HEAD,
        require_current_hardware=False,
    )
    smoke.verify_evidence_bundle(
        tar_path,
        sidecar,
        expected_head=HEAD,
        require_current_hardware=False,
    )

    incomplete = tmp_path / "incomplete.tar"
    with tarfile.open(incomplete, "w") as archive:
        archive.add(root / "run_report.json", arcname="phase9b2c-evidence/run_report.json")
    incomplete_sidecar = tmp_path / "incomplete.sha256"
    incomplete_sidecar.write_text(sha256(incomplete.read_bytes()).hexdigest() + "\n")
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="inventory"):
        smoke.verify_evidence_bundle(
            incomplete,
            incomplete_sidecar,
            expected_head=HEAD,
            require_current_hardware=False,
        )

    unsafe = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe, "w") as archive:
        for path in sorted(root.iterdir()):
            if path.name == "execution.log":
                info = tarfile.TarInfo("phase9b2c-evidence/execution.log")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)
            else:
                archive.add(path, arcname=f"phase9b2c-evidence/{path.name}")
    unsafe_sidecar = tmp_path / "unsafe.sha256"
    unsafe_sidecar.write_text(sha256(unsafe.read_bytes()).hexdigest() + "\n")
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="unsafe"):
        smoke.verify_evidence_bundle(
            unsafe,
            unsafe_sidecar,
            expected_head=HEAD,
            require_current_hardware=False,
        )


def test_pack_output_collision_and_failure_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _valid_evidence(tmp_path / "evidence")
    collision = tmp_path / "collision.tar"
    collision.write_bytes(b"existing")
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="collision"):
        smoke.pack_evidence_bundle(
            root,
            tar_path=collision,
            sidecar_path=tmp_path / "collision.sha256",
            expected_head=HEAD,
            require_current_hardware=False,
        )
    tar_path = tmp_path / "atomic.tar"
    monkeypatch.setattr(
        smoke,
        "_write_text_atomic",
        lambda path, value: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(OSError, match="injected"):
        smoke.pack_evidence_bundle(
            root,
            tar_path=tar_path,
            sidecar_path=tmp_path / "atomic.sha256",
            expected_head=HEAD,
            require_current_hardware=False,
        )
    assert not tar_path.exists()


def test_checkpoint_target_binding_mismatch_is_failure_atomic(tmp_path: Path) -> None:
    model = DilemmadataHierarchicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    config = smoke._training_config(10)
    data = {"target_cache_index_fingerprint": "1" * 64}
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        committed_metric_rows=0,
        resolved_config=config,
        data_fingerprints=data,
    )
    destination = DilemmadataHierarchicalModel()
    destination_optimizer = torch.optim.AdamW(destination.parameters(), lr=3e-4)
    destination_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        destination_optimizer, T_max=10
    )
    destination_scaler = torch.amp.GradScaler("cpu", enabled=True)
    before = {name: value.clone() for name, value in destination.state_dict().items()}
    with pytest.raises(TrainingCheckpointError, match="metadata_mismatch"):
        load_training_checkpoint(
            path,
            destination,
            destination_optimizer,
            scheduler=destination_scheduler,
            scaler=destination_scaler,
            maximum_next_epoch=0,
            resolved_config=config,
            data_fingerprints={"target_cache_index_fingerprint": "2" * 64},
        )
    assert all(
        torch.equal(destination.state_dict()[name], value)
        for name, value in before.items()
    )


def test_wrong_head_missing_path_and_shell_output_collision(tmp_path: Path) -> None:
    actual = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="head_mismatch"):
        smoke.validate_git_preflight(
            REPO_ROOT,
            expected_head=("0" * 40 if actual != "0" * 40 else "1" * 40),
            allowed_output_root=tmp_path,
        )
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match="bundle_invalid"):
        smoke.verify_evidence_bundle(
            tmp_path / "missing.tar",
            tmp_path / "missing.sha256",
            expected_head=HEAD,
            require_current_hardware=False,
        )

    output = tmp_path / "output"
    output.mkdir()
    (output / "fixed-run").mkdir()
    command = (
        "bash",
        str(RUNNER),
        "--expected-head",
        actual,
        "--raw-index",
        "missing",
        "--raw-cache-root",
        "missing",
        "--target-index",
        "missing",
        "--target-cache-root",
        "missing",
        "--split-manifest",
        "missing",
        "--output-root",
        str(output),
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PHASE9B2C_RUN_ID": "fixed-run"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "output collision" in completed.stderr


def test_runner_contract_and_optional_exact_rtx3090_probe() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    assert "--expected-head" in text
    assert "cpu" not in text.lower()
    assert "execution.log" in text
    assert "verify_phase9b2c_rtx3090_supervised_smoke.py" in text
    if not torch.cuda.is_available():
        pytest.skip("exact RTX 3090 CUDA hardware is not available")
    if torch.cuda.get_device_properties(0).name != smoke.DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME:
        pytest.skip("CUDA exists, but device 0 is not an RTX 3090")
    device, evidence = smoke._cuda_preflight()
    assert device == torch.device("cuda:0")
    assert evidence["amp_dtype"] == "float16"
    assert evidence["grad_scaler_enabled"] is True
