from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from music_critic.experiments.phase8b2.schedule import (
    RawDownstreamSampleSchedule,
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cc.contracts import (
    PHASE9CC_CELLS,
    PHASE9CC_TASKS,
    file_sha256,
    fingerprint,
)
from music_critic.experiments.phase9cc import contracts
from music_critic.evaluation.dilemmadata import evaluate_dilemmadata_model
from music_critic.evaluation.dilemmadata_run import _model as evaluation_model
from music_critic.experiments.phase9cc import runner
from music_critic.experiments.phase9cc import training
from music_critic.experiments.phase9cc.training import run_cell_training
from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    class_weight_artifact,
)
from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK
from music_critic.tasks import (
    create_split_manifest,
    dumps_corpus_index,
    dumps_dilemmadata_target_cache_index,
    dumps_split_manifest,
)
from music_critic.training import engine as training_engine
from music_critic.training.data import build_data_runtime
from music_critic.training.data import DataRuntime, ValidationMembership
from music_critic.training.models import build_baseline_model
from tests.models.test_dilemmadata_heads import _batch
from tests.tasks.test_dilemmadata_target_cache import _build


def _runtime() -> DataRuntime:
    batch = _batch()
    identities = tuple(zip(batch.dataset_ids, batch.piece_ids, strict=True))
    membership = ValidationMembership(
        identities=identities,
        membership_fingerprint="v" * 64,
        dataset_counts={"dilemmadata": len(identities)},
        full_view_count=len(identities),
        selected_count=len(identities),
        subset_limit=0,
    )
    return DataRuntime(
        first_train_batch=batch,
        train_loader=lambda epoch: tuple(batch for _ in range(12)),
        validation_loader=lambda: (batch,),
        validation_membership=membership,
        fingerprints={
            "kind": "production_format_bounded_fixture",
            "raw_index_fingerprint": "r" * 64,
            "target_cache_index_fingerprint": "t" * 64,
            "validation_membership_fingerprint": "v" * 64,
        },
        mixture_statistics={"requested_weights": {"dilemmadata": 1.0}},
    )


def _bounded_plan(tmp_path: Path) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    batch = _batch()
    one_batch = tuple(zip(batch.dataset_ids, batch.piece_ids, strict=True))
    identities = one_batch * 12
    weights = class_weight_artifact(
        {
            task_id: tuple(
                index + 1
                for index, _ in enumerate(
                    DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary
                )
            )
            for task_id in DILEMMADATA_ACTIVE_TASK_IDS
        },
        policy="inverse_sqrt_frequency_supported",
        train_membership_fingerprint="a" * 64,
    )
    weights_path = tmp_path / "class_weights.json"
    weights_path.write_text(json.dumps(weights), encoding="utf-8")
    ssl_checkpoint = tmp_path / "ssl-last.pt"
    ssl_checkpoint.write_bytes(b"bounded-ssl-checkpoint")
    model_config = {
        "name": "hierarchical",
        "hidden_dim": 16,
        "local_gnn_layers": 1,
        "transformer_layers": 1,
        "attention_heads": 4,
        "ffn_multiplier": 2,
        "dropout": 0.1,
        "residual": True,
        "decoder": {"kind": "mlp"},
    }
    torch.manual_seed(91)
    source_model = build_baseline_model(
        OmegaConf.create(model_config),
        task_weights={task: 1.0 for task in PHASE9CC_TASKS},
        dilemmadata=True,
    )
    encoder_export = tmp_path / "encoder.pt"
    encoder_prefixes = (
        "local_baseline.encoder.",
        "context_encoder.pooling.",
        "context_encoder.transformer.",
        "context_encoder.fusion.",
    )
    torch.save(
        {
            "encoder_state": {
                name: value.detach().clone()
                for name, value in source_model.state_dict().items()
                if name.startswith(encoder_prefixes)
            }
        },
        encoder_export,
    )
    bound_files = {}
    for name in ("raw", "target", "split", "priors"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"fixture": name}), encoding="utf-8")
        bound_files[name] = path
    (tmp_path / "raw-cache").mkdir()
    (tmp_path / "target-cache").mkdir()
    schedule = {
        "epochs": 1,
        "optimizer_steps_per_epoch": 12,
        "required_applied_updates": 12,
        "batch_size": 2,
        "epoch_size": 24,
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "scheduler": "none",
        "amp": "disabled_for_bounded_cpu_fixture",
        "telemetry_interval_applied": 2,
        "checkpoint_interval_applied": 4,
        "validation_milestones": [0, 4, 8, 12],
        "maximum_consecutive_skips": 2,
        "downstream_initialization_seed": 1701,
        "downstream_data_order_seed": 1702,
        "sample_schedule_fingerprint": (
            raw_downstream_sample_schedule_fingerprint(identities)
        ),
        "sample_count": len(identities),
        "identity_contract_version": "1.2.0",
        "resume_boundary": "applied_update_mid_epoch",
    }
    schedule["fingerprint"] = fingerprint(schedule)
    bindings = {
        "raw_index": {
            "path": str(bound_files["raw"]),
            "sha256": file_sha256(bound_files["raw"]),
        },
        "target_index": {
            "path": str(bound_files["target"]),
            "sha256": file_sha256(bound_files["target"]),
        },
        "split_manifest": {
            "path": str(bound_files["split"]),
            "sha256": file_sha256(bound_files["split"]),
        },
        "class_weight_artifact": {
            "path": str(weights_path),
            "sha256": file_sha256(weights_path),
        },
        "train_priors": {
            "path": str(bound_files["priors"]),
            "sha256": file_sha256(bound_files["priors"]),
        },
        "raw_cache_root": str(tmp_path / "raw-cache"),
        "target_cache_root": str(tmp_path / "target-cache"),
        "ssl_checkpoint": {
            "path": str(ssl_checkpoint),
            "sha256": file_sha256(ssl_checkpoint),
            "source_kind": "phase8b_multilevel_ssl",
            "encoder_export_path": str(encoder_export),
            "encoder_export_sha256": file_sha256(encoder_export),
        },
    }
    protocol = {
        "contract_version": "1.0.0",
        "phase": "9C-C",
        "hypothesis": "bounded",
        "seed": 17,
        "git_head": "a" * 40,
        "phase9cb_base_sha": "786d0dd9320545f2eee50b6d59e609e72d96da49",
        "cells": list(PHASE9CC_CELLS),
        "tasks": list(PHASE9CC_TASKS),
        "schedule": schedule,
        "model": model_config,
        "bindings": bindings,
        "validation_membership": {
            "split": "validation",
            "identities": [list(value) for value in one_batch],
            "membership_fingerprint": "v" * 64,
            "evaluation_membership_fingerprint": "e" * 64,
            "selected_count": len(one_batch),
        },
        "test_lock": {
            "test_inference": False,
            "test_targets_accessed": False,
            "test_metrics_accessed": False,
            "test_unlock": False,
        },
        "claim_boundary": "bounded_regression",
        "bounded_test_protocol": True,
        "bounded_schedule_identities": [list(value) for value in identities],
    }
    protocol["fingerprint"] = fingerprint(protocol)
    cells = [
        {
            "cell_id": cell_id,
            "encoder_initialization": "ssl" if cell_id == "ssl_mlp" else "scratch",
            "decoder_kind": "mlp",
            "transfer_mode": "full_finetune" if cell_id == "ssl_mlp" else "supervised_scratch",
            "schedule_fingerprint": schedule["sample_schedule_fingerprint"],
            "validation_milestones": [0, 4, 8, 12],
            "comparison_checkpoint_policy": "fixed_update_milestones",
        }
        for cell_id in PHASE9CC_CELLS
    ]
    plan = {
        "contract_version": "1.0.0",
        "protocol": protocol,
        "cells": cells,
        "production_started": False,
    }
    plan["fingerprint"] = fingerprint(plan)
    return plan, identities


def _install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    identities: tuple[tuple[str, str], ...],
) -> None:
    monkeypatch.setattr(
        training_engine, "build_data_runtime", lambda config, seed: _runtime()
    )
    monkeypatch.setattr(training, "_schedule", lambda plan: identities)


class _View:
    def __init__(self, prefix: str, count: int):
        self.prefix = prefix
        self.count = count

    def __len__(self) -> int:
        return self.count

    def record_identity(self, index: int) -> tuple[str, str]:
        return "dilemmadata", f"{self.prefix}-{index}"


def test_production_plan_is_exact_two_cell_9000_update_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {}
    for name in (
        "ssl_checkpoint",
        "ssl_encoder_export",
        "raw_index",
        "target_index",
        "split_manifest",
        "class_weight_artifact",
        "train_priors",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    raw_root = tmp_path / "raw-cache"
    target_root = tmp_path / "target-cache"
    raw_root.mkdir()
    target_root.mkdir()
    monkeypatch.setattr(contracts, "_validate_encoder_export", lambda path: None)
    monkeypatch.setattr(contracts, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(contracts, "_base_is_ancestor", lambda: True)
    monkeypatch.setattr(
        contracts, "_validate_weight_artifacts", lambda *args: None
    )
    monkeypatch.setattr(
        contracts,
        "build_corpus_data_views",
        lambda config: SimpleNamespace(
            train=_View("train", 7),
            validation=_View("validation", 3),
            manifest=SimpleNamespace(manifest_fingerprint="m" * 64),
        ),
    )
    def schedule_builder(
        dataset,
        *,
        weights,
        seed,
        first_epoch,
        epochs,
        steps_per_epoch,
        batch_size,
    ):
        del weights
        identities = tuple(
            dataset.record_identity(index % len(dataset))
            for index in range(epochs * steps_per_epoch * batch_size)
        )
        return RawDownstreamSampleSchedule(
            seed=seed,
            first_epoch=first_epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            identities=identities,
            fingerprint=raw_downstream_sample_schedule_fingerprint(identities),
        )

    monkeypatch.setattr(
        contracts, "build_raw_downstream_sample_schedule", schedule_builder
    )
    config = {
        **{name: str(path) for name, path in files.items()},
        "ssl_checkpoint_sha256": file_sha256(files["ssl_checkpoint"]),
        "ssl_encoder_export_sha256": file_sha256(files["ssl_encoder_export"]),
        "ssl_source_kind": "phase8b_multilevel_ssl",
        "raw_cache_root": str(raw_root),
        "target_cache_root": str(target_root),
        "git_head": "a" * 40,
    }
    config.update(
        {
            f"{name}_sha256": file_sha256(files[name])
            for name in (
                "raw_index",
                "target_index",
                "split_manifest",
                "class_weight_artifact",
                "train_priors",
            )
        }
    )
    plan = contracts.build_plan(config)
    schedule = plan["protocol"]["schedule"]
    assert tuple(cell["cell_id"] for cell in plan["cells"]) == PHASE9CC_CELLS
    assert {cell["decoder_kind"] for cell in plan["cells"]} == {"mlp"}
    assert schedule["epochs"] == 1
    assert schedule["optimizer_steps_per_epoch"] == 9000
    assert schedule["required_applied_updates"] == 9000
    assert schedule["telemetry_interval_applied"] == 100
    assert schedule["checkpoint_interval_applied"] == 1000
    assert schedule["validation_milestones"] == [0, 1000, 3000, 6000, 9000]
    assert plan["protocol"]["test_lock"] == {
        "test_inference": False,
        "test_targets_accessed": False,
        "test_metrics_accessed": False,
        "test_unlock": False,
    }


def test_bounded_production_format_fixture_uses_real_quota_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _corpus,
        raw_cache,
        raw_index,
        target_cache,
        target_index,
        _report,
    ) = _build(tmp_path / "fixture")
    assignments = {
        (record.dataset_id, record.piece_id): "train"
        for record in raw_index.records
    }
    split = create_split_manifest((raw_index,), assignments, seed=17)
    paths = {
        "raw_index": tmp_path / "raw.index.json",
        "target_index": tmp_path / "target.index.json",
        "split_manifest": tmp_path / "split.json",
        "class_weight_artifact": tmp_path / "weights.json",
        "train_priors": tmp_path / "priors.json",
        "ssl_checkpoint": tmp_path / "ssl.pt",
        "ssl_encoder_export": tmp_path / "encoder.pt",
    }
    paths["raw_index"].write_text(
        dumps_corpus_index(raw_index), encoding="utf-8"
    )
    paths["target_index"].write_text(
        dumps_dilemmadata_target_cache_index(target_index), encoding="utf-8"
    )
    paths["split_manifest"].write_text(
        dumps_split_manifest(split), encoding="utf-8"
    )
    paths["class_weight_artifact"].write_text("{}", encoding="utf-8")
    paths["train_priors"].write_text("{}", encoding="utf-8")
    paths["ssl_checkpoint"].write_bytes(b"ssl")
    torch.save(
        {
            "metadata": {
                "encoder_export_contract_version": "1.0.0",
                "hierarchical_encoder_contract": {"fixture": True},
                "parameter_names": ["local_baseline.encoder.fixture"],
            },
            "encoder_state": {
                "local_baseline.encoder.fixture": torch.zeros(1)
            },
        },
        paths["ssl_encoder_export"],
    )
    monkeypatch.setattr(
        contracts, "_validate_weight_artifacts", lambda *args: None
    )
    monkeypatch.setattr(contracts, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(contracts, "_base_is_ancestor", lambda: True)
    config = {
        **{name: str(path) for name, path in paths.items()},
        **{
            f"{name}_sha256": file_sha256(paths[name])
            for name in (
                "raw_index",
                "target_index",
                "split_manifest",
                "class_weight_artifact",
                "train_priors",
            )
        },
        "ssl_checkpoint_sha256": file_sha256(paths["ssl_checkpoint"]),
        "ssl_encoder_export_sha256": file_sha256(
            paths["ssl_encoder_export"]
        ),
        "ssl_source_kind": "phase8b_multilevel_ssl",
        "raw_cache_root": str(raw_cache.root),
        "target_cache_root": str(target_cache.root),
        "git_head": "a" * 40,
    }
    plan = contracts.build_plan(
        config,
        _bounded_protocol={
            "updates": 12,
            "telemetry_interval": 2,
            "checkpoint_interval": 4,
            "milestones": (0, 4, 8, 12),
            "batch_size": 1,
            "hidden_dim": 16,
            "local_gnn_layers": 1,
            "transformer_layers": 1,
            "attention_heads": 4,
            "ffn_multiplier": 2,
            "dropout": 0.0,
        },
    )
    training_cfg = training.training_config(
        plan, plan["cells"][0], tmp_path / "unused", device="cpu"
    )
    runtime = build_data_runtime(
        OmegaConf.create(training_cfg["data"]),
        seed=plan["protocol"]["schedule"]["downstream_data_order_seed"],
    )
    observed = tuple(
        identity
        for batch in runtime.train_loader(0)
        for identity in zip(batch.dataset_ids, batch.piece_ids, strict=True)
    )
    planned = tuple(
        tuple(value)
        for value in plan["protocol"]["bounded_schedule_identities"]
    )
    assert len(observed) == len(planned) == 12
    assert observed == planned
    assert raw_downstream_sample_schedule_fingerprint(observed) == plan[
        "protocol"
    ]["schedule"]["sample_schedule_fingerprint"]


def test_continuous_update_telemetry_and_resume_are_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, identities = _bounded_plan(tmp_path)
    _install_runtime(monkeypatch, identities)
    scratch = plan["cells"][0]
    uninterrupted = tmp_path / "uninterrupted"
    uninterrupted_report = run_cell_training(
        plan, scratch, uninterrupted, action="run", device="cpu"
    )
    assert uninterrupted_report["applied_updates"] == 12
    assert uninterrupted_report["attempted_updates"] == 12
    assert uninterrupted_report["telemetry_row_count"] == 6
    rows = runner._rows(uninterrupted / "train_telemetry.jsonl")
    assert [row["applied_updates"] for row in rows] == [2, 4, 6, 8, 10, 12]
    assert all(row["window_applied_updates"] == 2 for row in rows)
    assert set(rows[-1]["mean_task_losses"]) == set(PHASE9CC_TASKS)

    interrupted = tmp_path / "interrupted"
    partial = run_cell_training(
        plan,
        scratch,
        interrupted,
        action="run",
        device="cpu",
        stop_after_applied=6,
    )
    assert partial["complete"] is False
    resumed = run_cell_training(
        plan, scratch, interrupted, action="resume", device="cpu"
    )
    assert resumed["complete"] is True
    assert resumed["final_model_state_fingerprint"] == uninterrupted_report[
        "final_model_state_fingerprint"
    ]
    assert runner._rows(interrupted / "train_telemetry.jsonl") == rows
    for update in (0, 4, 8, 12):
        left = torch.load(
            uninterrupted / "checkpoints" / f"update-{update}.pt",
            map_location="cpu",
            weights_only=True,
        )
        right = torch.load(
            interrupted / "checkpoints" / f"update-{update}.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert left["model_state_fingerprint"] == right["model_state_fingerprint"]
    batch = _batch()
    components = {
        identity: f"component-{index}"
        for index, identity in enumerate(
            zip(batch.dataset_ids, batch.piece_ids, strict=True)
        )
    }
    evaluations = []
    for directory in (uninterrupted, interrupted):
        model = evaluation_model(
            directory / "checkpoints" / "update-12.pt",
            torch.device("cpu"),
        )
        evaluations.append(
            evaluate_dilemmadata_model(
                model,
                (batch,),
                component_by_identity=components,
                membership_fingerprint="e" * 64,
            )
        )
    assert evaluations[0] == evaluations[1]


def test_telemetry_and_validation_boundaries_do_not_change_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, identities = _bounded_plan(tmp_path)
    _install_runtime(monkeypatch, identities)
    scratch = plan["cells"][0]
    with_telemetry = run_cell_training(
        plan, scratch, tmp_path / "with", action="run", device="cpu"
    )
    without_telemetry = run_cell_training(
        plan,
        scratch,
        tmp_path / "without",
        action="run",
        device="cpu",
        telemetry_enabled=False,
    )
    assert with_telemetry["final_model_state_fingerprint"] == without_telemetry[
        "final_model_state_fingerprint"
    ]
    checkpoint = torch.load(
        tmp_path / "with" / "checkpoints" / "update-8.pt",
        map_location="cpu",
        weights_only=True,
    )
    before_rng = copy.deepcopy(checkpoint["rng_state"])
    before_model = checkpoint["model_state_fingerprint"]
    assert training.model_state_fingerprint(checkpoint["model_state"]) == before_model
    assert checkpoint["rng_state"]["python"] == before_rng["python"]
    assert torch.equal(
        checkpoint["rng_state"]["torch_cpu"], before_rng["torch_cpu"]
    )
    interrupted = tmp_path / "validation-interrupted"
    run_cell_training(
        plan,
        scratch,
        interrupted,
        action="run",
        device="cpu",
        stop_after_applied=4,
    )
    milestone_path = interrupted / "checkpoints" / "update-4.pt"
    model = evaluation_model(milestone_path, torch.device("cpu"))
    batch = _batch()
    evaluate_dilemmadata_model(
        model,
        (batch,),
        component_by_identity={
            identity: f"component-{index}"
            for index, identity in enumerate(
                zip(batch.dataset_ids, batch.piece_ids, strict=True)
            )
        },
        membership_fingerprint="e" * 64,
    )
    resumed = run_cell_training(
        plan, scratch, interrupted, action="resume", device="cpu"
    )
    assert resumed["final_model_state_fingerprint"] == with_telemetry[
        "final_model_state_fingerprint"
    ]


def test_skipped_attempt_is_not_applied_and_persistent_skip_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, identities = _bounded_plan(tmp_path)
    _install_runtime(monkeypatch, identities)
    scratch = plan["cells"][0]
    original = training_engine._optimize_batch
    calls = {"count": 0}

    def skip_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return None, {"optimizer_step_applied": False}, True
        return original(*args, **kwargs)

    monkeypatch.setattr(training_engine, "_optimize_batch", skip_once)
    report = run_cell_training(
        plan, scratch, tmp_path / "skip-once", action="run", device="cpu"
    )
    assert report["applied_updates"] == 12
    assert report["attempted_updates"] == 13
    assert report["skipped_updates"] == 1
    assert report["actual_sample_schedule_fingerprint"] == plan["protocol"][
        "schedule"
    ]["sample_schedule_fingerprint"]

    monkeypatch.setattr(
        training_engine,
        "_optimize_batch",
        lambda *args, **kwargs: (
            None,
            {"optimizer_step_applied": False},
            True,
        ),
    )
    with pytest.raises(ValueError, match="persistent_amp_overflow"):
        run_cell_training(
            plan,
            scratch,
            tmp_path / "persistent-skip",
            action="run",
            device="cpu",
        )


def _metrics(update: int) -> dict[str, object]:
    tasks = {
        task: {
            "normalized_nll": 1.0 - update / 100.0,
            "macro_f1": update / 100.0,
            "balanced_accuracy": update / 100.0,
            "accuracy": update / 100.0,
            "true_class_support": [1],
            "predicted_class_distribution": [1.0],
            "prediction_entropy": 0.0,
        }
        for task in PHASE9CC_TASKS
    }
    return {
        "tasks": tasks,
        "aggregate": {
            "task_count": 4,
            "mean_normalized_nll": 1.0 - update / 100.0,
            "mean_macro_f1": update / 100.0,
            "mean_balanced_accuracy": update / 100.0,
            "mean_accuracy": update / 100.0,
            "mean_prediction_entropy": 0.0,
        },
    }


def _complete_bundle(
    root: Path,
    plan: dict[str, object],
    source_roots: dict[str, Path],
) -> None:
    runner._write(root / "experiment_plan.json", plan)
    runner._write(root / "protocol.json", plan["protocol"])
    for cell in plan["cells"]:
        cell_id = cell["cell_id"]
        source = source_roots[cell_id]
        destination = root / "cells" / cell_id
        shutil.copytree(source, destination)
        milestone_rows = []
        for update in (0, 4, 8, 12):
            checkpoint_path = destination / "checkpoints" / f"update-{update}.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            report_payload = {
                "contract_version": "fixture",
                "split": "validation",
                "membership_fingerprint": "e" * 64,
                **_metrics(update),
            }
            report_payload["fingerprint"] = fingerprint(report_payload)
            report_path = destination / "milestones" / f"update-{update}.json"
            runner._write(report_path, report_payload)
            milestone_rows.append(
                {
                    "update": update,
                    "checkpoint_path": f"checkpoints/update-{update}.pt",
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "model_state_fingerprint": checkpoint[
                        "model_state_fingerprint"
                    ],
                    "checkpoint_declared_model_state_fingerprint": checkpoint[
                        "model_state_fingerprint"
                    ],
                    "validation_report_path": f"milestones/update-{update}.json",
                    "validation_report_sha256": file_sha256(report_path),
                    "validation_report_fingerprint": report_payload["fingerprint"],
                    "validation_membership_fingerprint": "e" * 64,
                    **_metrics(update),
                }
            )
        milestones = {
            "contract_version": "1.0.0",
            "cell_id": cell_id,
            "split": "validation",
            "milestones": milestone_rows,
            "test_access": False,
        }
        milestones["fingerprint"] = fingerprint(milestones)
        runner._write(destination / "validation_milestones.json", milestones)
    runner.aggregate(root, plan)


def test_two_cell_pairing_bundle_verifier_and_fail_closed_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, identities = _bounded_plan(tmp_path)
    _install_runtime(monkeypatch, identities)
    source_roots = {}
    for cell in plan["cells"]:
        directory = tmp_path / f"source-{cell['cell_id']}"
        report = run_cell_training(
            plan, cell, directory, action="run", device="cpu"
        )
        assert report["actual_sample_schedule_fingerprint"] == plan[
            "protocol"
        ]["schedule"]["sample_schedule_fingerprint"]
        source_roots[cell["cell_id"]] = directory
    scratch_report = runner._read(
        source_roots["scratch_mlp"] / "training_report.json"
    )
    ssl_report = runner._read(source_roots["ssl_mlp"] / "training_report.json")
    assert scratch_report["fresh_supervised_initialization_fingerprint"] == (
        ssl_report["fresh_supervised_initialization_fingerprint"]
    )
    bundle = tmp_path / "bundle"
    _complete_bundle(bundle, plan, source_roots)
    assert runner.verify_bundle(bundle, expected_sha="a" * 40)["status"] == "verified"
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase9cc.run",
            "verify",
            "--output-root",
            str(bundle),
            "--expected-sha",
            "a" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    assert '"status": "verified"' in cli.stdout

    missing = tmp_path / "missing"
    shutil.copytree(bundle, missing)
    (missing / "cells" / "scratch_mlp" / "milestones" / "update-8.json").unlink()
    with pytest.raises(ValueError, match="unreadable|hash_invalid"):
        runner.verify_bundle(missing, expected_sha="a" * 40)

    crossed = tmp_path / "crossed"
    shutil.copytree(bundle, crossed)
    shutil.copyfile(
        crossed / "cells" / "scratch_mlp" / "checkpoints" / "update-4.pt",
        crossed / "cells" / "ssl_mlp" / "checkpoints" / "update-4.pt",
    )
    with pytest.raises(ValueError, match="checkpoint_binding_invalid"):
        runner.verify_bundle(crossed, expected_sha="a" * 40)

    corrupted = tmp_path / "corrupted"
    shutil.copytree(bundle, corrupted)
    path = corrupted / "cells" / "scratch_mlp" / "checkpoints" / "update-4.pt"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint_unreadable"):
        runner.verify_bundle(corrupted, expected_sha="a" * 40)

    test_access = tmp_path / "test-access"
    shutil.copytree(bundle, test_access)
    report_path = test_access / "convergence_report.json"
    report = runner._read(report_path)
    report["test_access"] = True
    runner._write(report_path, report)
    with pytest.raises(ValueError, match="convergence_claim_invalid"):
        runner.verify_bundle(test_access, expected_sha="a" * 40)


def test_phase9cb_mlp_optimizer_step_default_contract_remains_compatible() -> None:
    import inspect

    signature = inspect.signature(training_engine._optimize_batch)
    assert signature.parameters["collect_update_metric"].default is False
