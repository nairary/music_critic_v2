from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "value",
    (
        torch.tensor(1.25, dtype=torch.float32),
        torch.tensor(1.25, dtype=torch.float16),
        torch.tensor(7, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    ),
)
def test_tensor_fingerprint_supports_scalar_and_empty_tensors(
    value: torch.Tensor,
) -> None:
    fingerprint = smoke._tensor_fingerprint(value)
    assert len(fingerprint) == 64
    assert fingerprint == smoke._tensor_fingerprint(value.clone())


def test_tensor_fingerprint_supports_non_contiguous_tensor() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not value.is_contiguous()
    assert smoke._tensor_fingerprint(value) == smoke._tensor_fingerprint(
        value.contiguous()
    )


def test_tensor_fingerprint_preserves_vector_and_matrix_contract() -> None:
    assert smoke._tensor_fingerprint(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    ) == "6c7930bbde426d083d789de06c6bc1ca6a52f03471d847da684b0d7b52570db1"
    assert smoke._tensor_fingerprint(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    ) == "34bf5eb7a09595f77d3e81877e042c9e930e497357428ea9c77396b51c3315f7"


def test_tensor_fingerprint_separates_scalar_values_dtypes_and_shapes() -> None:
    fingerprints = {
        smoke._tensor_fingerprint(torch.tensor(1.0, dtype=torch.float32)),
        smoke._tensor_fingerprint(torch.tensor(2.0, dtype=torch.float32)),
        smoke._tensor_fingerprint(torch.tensor(1.0, dtype=torch.float16)),
        smoke._tensor_fingerprint(torch.tensor([1.0], dtype=torch.float32)),
    }
    assert len(fingerprints) == 4


def test_supervision_loss_evidence_supports_scalar_total_loss() -> None:
    output = SimpleNamespace(
        supervisions=(),
        harmonic_loss=SimpleNamespace(
            total_loss=torch.tensor(3.5, dtype=torch.float16)
        ),
    )
    evidence = smoke._supervision_loss_evidence(output)
    assert evidence["fingerprint"] == smoke._fingerprint([])
    assert evidence["total_loss_fingerprint"] == smoke._tensor_fingerprint(
        output.harmonic_loss.total_loss
    )


def _replay_diagnostic() -> dict[str, object]:
    return {
        "contract_version": smoke.DILEMMADATA_CUDA_REPLAY_DIAGNOSTIC_VERSION,
        "purpose": "independent_cuda_amp_replay_not_target_leakage",
        "candidate_identities_exact": True,
        "all_logits_finite": True,
        "comparison_dtype": "float32",
        "absolute_tolerance": smoke.DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE,
        "relative_tolerance": smoke.DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE,
        "minimum_cosine_similarity": (
            smoke.DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY
        ),
        "tasks": [
            {
                "task_id": task_id,
                "max_absolute_difference_fp32": 0.001,
                "max_relative_difference_fp32": 0.001,
                "cosine_similarity_fp32": 0.999999,
                "within_elementwise_tolerance": True,
            }
            for task_id in DILEMMADATA_ACTIVE_TASK_IDS
        ],
    }


def _predictions(*, delta: float = 0.0) -> tuple[SimpleNamespace, ...]:
    rows = []
    for index, task_id in enumerate(DILEMMADATA_ACTIVE_TASK_IDS):
        rows.append(
            SimpleNamespace(
                contract_version="fixture@1",
                task_id=task_id,
                source_adapter=task_id.split(".")[1],
                allowed_node_types=("note",),
                candidate_node_type_codes=torch.tensor([0, 0]),
                global_entity_indices=torch.tensor([index, index + 1]),
                sample_indices=torch.tensor([0, 0]),
                candidate_offsets_by_node_type=torch.tensor([0, 2]),
                candidate_counts_by_node_type=torch.tensor([2, 0]),
                logits=torch.tensor(
                    [[1.0 + delta, 2.0], [3.0, 4.0 - delta]],
                    dtype=torch.float32,
                ),
            )
        )
    return tuple(rows)


def test_small_independent_replay_difference_is_not_target_leakage() -> None:
    evidence = smoke._prediction_replay_diagnostic(
        _predictions(), _predictions(delta=0.001)
    )
    assert evidence["purpose"] == "independent_cuda_amp_replay_not_target_leakage"
    assert all(row["within_elementwise_tolerance"] for row in evidence["tasks"])


def test_target_join_prediction_mutation_is_rejected() -> None:
    predictions = _predictions()
    snapshot = smoke._prediction_snapshot(predictions)
    predictions[0].logits.add_(0.001)
    with pytest.raises(
        smoke.DilemmadataSupervisedSmokeError,
        match="target_join_changed_raw_predictions",
    ):
        smoke._assert_prediction_snapshot(predictions, snapshot)


def test_target_dependent_prediction_replacement_is_rejected() -> None:
    predictions = _predictions()
    snapshot = smoke._prediction_snapshot(predictions)
    target_dependent_replay = _predictions(delta=0.001)
    with pytest.raises(
        smoke.DilemmadataSupervisedSmokeError,
        match="target_join_replaced_prediction_object",
    ):
        smoke._assert_prediction_snapshot(target_dependent_replay, snapshot)


def test_candidate_identity_mutation_is_rejected() -> None:
    reference = _predictions()
    replay = list(_predictions())
    replay[0].global_entity_indices[0] += 1
    with pytest.raises(
        smoke.DilemmadataSupervisedSmokeError,
        match="cuda_replay_candidate_identity_mismatch",
    ):
        smoke._prediction_replay_diagnostic(reference, tuple(replay))


@pytest.mark.parametrize("failure", ("nan", "inf", "tolerance"))
def test_cuda_replay_rejects_nonfinite_and_excessive_difference(
    failure: str,
) -> None:
    reference = _predictions()
    replay = list(_predictions())
    replay[0].logits[0, 0] = {
        "nan": float("nan"),
        "inf": float("inf"),
        "tolerance": 2.0,
    }[failure]
    category = "non_finite" if failure != "tolerance" else "tolerance_exceeded"
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError, match=category):
        smoke._prediction_replay_diagnostic(reference, tuple(replay))


def _target_semantics(observed_index: str) -> dict[str, object]:
    return smoke._with_fingerprint(
        {
            "policy": "stable_semantics_plus_observed_physical_index_v1",
            "record_count": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT,
            "raw_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
            "metadata_index_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT,
            "aggregate_target_bundle_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT,
            "observed_target_cache_index_fingerprint": observed_index,
            "known_observed_physical_index_fingerprints": list(
                smoke.DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS
            ),
            "target_index_role": "exact_run_resume_evaluation_binding_not_universal_semantic_identity",
            "contract_versions": dict(smoke._TARGET_CONTRACT_VERSIONS),
            "source_free_full_validation": {
                "index_self_fingerprint_verified": True,
                "index_record_count_verified": 719,
                "artifact_sha256_verified_count": 719,
                "target_bundle_fingerprint_verified_count": 719,
            },
        }
    )


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
    observed_index = smoke.DILEMMADATA_SUPERVISED_SMOKE_LOCAL_TARGET_INDEX_FINGERPRINT
    target_semantics = _target_semantics(observed_index)
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
            "observed_target_cache_index_fingerprint": observed_index,
            "split_manifest_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
            "target_semantic_projection_fingerprint": target_semantics["fingerprint"],
        },
        "target_semantic_validation": target_semantics,
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
            "target_columns_read_only_after_raw_prediction": True,
            "prediction_object_exact_after_target_joins": True,
            "initial": {},
            "target_mutation": {
                "verified": True,
                "raw_prediction_call_count": 1,
                "same_prediction_object_for_both_joins": True,
                "mutated_target_row_count": 4,
                "candidate_identity_fingerprint_before": "1" * 64,
                "candidate_identity_fingerprint_after": "1" * 64,
                "raw_only_logits_fingerprint_before": "2" * 64,
                "raw_only_logits_fingerprint_after": "2" * 64,
                "tensor_storage_and_values_exact_after_original_join": True,
                "tensor_storage_and_values_exact_after_mutated_join": True,
                "original_target_fingerprint": "4" * 64,
                "mutated_target_fingerprint": "5" * 64,
                "original_supervision_loss": {"fingerprint": "6" * 64},
                "mutated_supervision_loss": {"fingerprint": "7" * 64},
            },
        },
        "cuda_replay_diagnostic": _replay_diagnostic(),
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
            "observed_target_cache_index_fingerprint": observed_index,
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
            "reload_model_state": {
                "model_state_tensors_exact": True,
                "tensor_count": 1,
                "state_fingerprint": "8" * 64,
            },
            "reload_logits_bounded_cuda_replay": True,
            "reload_cuda_replay_diagnostic": _replay_diagnostic(),
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
            "observed_target_cache_index_fingerprint": observed_index,
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
    state_rows = [
        {"name": name, "fingerprint": smoke._tensor_fingerprint(value)}
        for name, value in payload["model_state"].items()
    ]
    report["checkpoint"]["reload_model_state"] = {
        "model_state_tensors_exact": True,
        "tensor_count": len(state_rows),
        "state_fingerprint": smoke._fingerprint(state_rows),
    }


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
            lambda value: value["checkpoint"].update(reload_logits_bounded_cuda_replay=False),
            "checkpoint",
        ),
        (
            lambda value: value["bindings"].update(
                observed_target_cache_index_fingerprint="f" * 64
            ),
            "target_semantic",
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
            lambda value: value["validation"].update(
                observed_target_cache_index_fingerprint="f" * 64
            ),
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


@pytest.mark.parametrize(
    "observed_index",
    smoke.DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS,
)
def test_report_accepts_both_observed_indexes_with_same_semantics(
    observed_index: str,
) -> None:
    train, validation = _memberships()
    priors, evaluation = _evaluation(train, validation)
    report = _base_report(train, validation, priors, evaluation)
    semantics = _target_semantics(observed_index)
    report["target_semantic_validation"] = semantics
    report["bindings"].update(
        observed_target_cache_index_fingerprint=observed_index,
        target_semantic_projection_fingerprint=semantics["fingerprint"],
    )
    report["checkpoint"]["observed_target_cache_index_fingerprint"] = observed_index
    report["validation"]["observed_target_cache_index_fingerprint"] = observed_index
    validated = smoke.validate_smoke_report(smoke._with_fingerprint(report))
    assert validated["bindings"]["observed_target_cache_index_fingerprint"] == (
        observed_index
    )


def _production_objects(observed_index: str):
    raw = SimpleNamespace(
        header=SimpleNamespace(
            index_fingerprint=smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT
        )
    )
    target = SimpleNamespace(
        index_fingerprint=observed_index,
        raw_index_fingerprint=smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
        metadata_index_fingerprint=smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT,
        records=(None,) * smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT,
    )
    split = SimpleNamespace(
        manifest_fingerprint=smoke.DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT,
        index_fingerprints=(
            (
                "dilemmadata",
                smoke.DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
            ),
        ),
    )
    return raw, target, split


def test_production_policy_accepts_any_self_consistent_physical_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def checked(index, *, raw_index, cache_config):
        del raw_index, cache_config
        return {
            "ready": True,
            "record_count": len(index.records),
            "index_fingerprint": index.index_fingerprint,
            "raw_index_fingerprint": index.raw_index_fingerprint,
            "target_bundle_fingerprint": smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT,
        }

    monkeypatch.setattr(smoke, "check_dilemmadata_target_cache", checked)
    for observed_index in (
        *smoke.DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS,
        "f" * 64,
    ):
        raw, target, split = _production_objects(observed_index)
        bindings, semantics = smoke._validate_production_bindings(
            raw,
            target,
            smoke.DilemmadataTargetCacheConfig(tmp_path),
            split,
        )
        assert bindings["observed_target_cache_index_fingerprint"] == observed_index
        assert semantics["aggregate_target_bundle_fingerprint"] == (
            smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT
        )


@pytest.mark.parametrize("mutation", ("raw", "metadata", "bundle"))
def test_production_policy_rejects_semantic_mutation(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, target, split = _production_objects("f" * 64)
    if mutation == "raw":
        raw.header.index_fingerprint = "e" * 64
    elif mutation == "metadata":
        target.metadata_index_fingerprint = "e" * 64

    def checked(index, *, raw_index, cache_config):
        del raw_index, cache_config
        return {
            "ready": True,
            "record_count": len(index.records),
            "index_fingerprint": index.index_fingerprint,
            "raw_index_fingerprint": index.raw_index_fingerprint,
            "target_bundle_fingerprint": (
                "e" * 64
                if mutation == "bundle"
                else smoke.DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT
            ),
        }

    monkeypatch.setattr(smoke, "check_dilemmadata_target_cache", checked)
    with pytest.raises(smoke.DilemmadataSupervisedSmokeError):
        smoke._validate_production_bindings(
            raw,
            target,
            smoke.DilemmadataTargetCacheConfig(tmp_path),
            split,
        )


def test_production_policy_rejects_artifact_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw, target, split = _production_objects("f" * 64)

    def corrupt(*args, **kwargs):
        del args, kwargs
        raise smoke.DilemmadataTargetCacheError(
            "dilemmadata.target_cache.artifact_fingerprint_mismatch",
            "corrupt artifact",
        )

    monkeypatch.setattr(smoke, "check_dilemmadata_target_cache", corrupt)
    with pytest.raises(
        smoke.DilemmadataSupervisedSmokeError,
        match="target_semantic_validation_failed",
    ):
        smoke._validate_production_bindings(
            raw,
            target,
            smoke.DilemmadataTargetCacheConfig(tmp_path),
            split,
        )


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


def test_checkpoint_observed_target_index_mismatch_is_failure_atomic(
    tmp_path: Path,
) -> None:
    model = DilemmadataHierarchicalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    config = smoke._training_config(10)
    data = {"observed_target_cache_index_fingerprint": "1" * 64}
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
            data_fingerprints={"observed_target_cache_index_fingerprint": "2" * 64},
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
