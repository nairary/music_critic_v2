from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch

from music_critic.evaluation.dilemmadata import evaluate_dilemmadata_model
from music_critic.evaluation.dilemmadata_run import _model as evaluation_model
from music_critic.experiments.phase9cc import runner as parent_runner
from music_critic.experiments.phase9cc import training as parent_training
from music_critic.experiments.phase9cc.contracts import PHASE9CC_CELLS, file_sha256
from music_critic.experiments.phase9cc_continuation import contracts
from music_critic.experiments.phase9cc_continuation import runner
from music_critic.experiments.phase9cc_continuation import training
from music_critic.training import engine as training_engine
from music_critic.training.data import DataRuntime, ValidationMembership
from tests.experiments.phase9cc.test_phase9cc_convergence import (
    _batch,
    _bounded_plan,
    _complete_bundle,
)


def _full_batch():
    batch = _batch()
    targets = {target.task_id: target for target in batch.target_batches}
    donor_by_missing = {
        "dilemmadata.an.chord.quality": "dilemmadata.an.chord.inversion",
        "dilemmadata.dlc.chord.inversion": "dilemmadata.dlc.chord.quality",
    }
    replacements = {}
    for missing, donor_name in donor_by_missing.items():
        target = targets[missing]
        donor = targets[donor_name]
        replacements[missing] = replace(
            target,
            values=torch.zeros_like(donor.values),
            availability_mask=donor.availability_mask.clone(),
            entity_indices=donor.entity_indices.clone(),
            entity_index_mask=donor.entity_index_mask.clone(),
            entity_node_type_codes=donor.entity_node_type_codes.clone(),
            entity_node_types=donor.entity_node_types,
            sample_indices=donor.sample_indices.clone(),
            source_entry_indices=donor.source_entry_indices.clone(),
            source_entry_counts_by_sample=(
                donor.source_entry_counts_by_sample.clone()
            ),
            entry_count=donor.entry_count,
            source_entry_count=donor.source_entry_count,
            provenance_cpu=donor.provenance_cpu,
            diagnostics_cpu=donor.diagnostics_cpu,
        )
    result = copy.deepcopy(batch)
    object.__setattr__(
        result,
        "target_batches",
        tuple(
            replacements.get(target.task_id, target)
            for target in result.target_batches
        ),
    )
    return result


def _runtime(batch_count: int) -> DataRuntime:
    batch = _full_batch()
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
        train_loader=lambda epoch: tuple(batch for _ in range(batch_count)),
        validation_loader=lambda: (batch,),
        validation_membership=membership,
        fingerprints={
            "kind": "production_format_bounded_continuation_fixture",
            "raw_index_fingerprint": "r" * 64,
            "target_cache_index_fingerprint": "t" * 64,
            "validation_membership_fingerprint": "v" * 64,
        },
        mixture_statistics={"requested_weights": {"dilemmadata": 1.0}},
    )


def _config(parent_plan: dict[str, object], path: Path) -> Path:
    bindings = parent_plan["protocol"]["bindings"]
    config = {
        name: binding["path"]
        for name, binding in bindings.items()
        if name
        in {
            "raw_index",
            "target_index",
            "split_manifest",
            "class_weight_artifact",
            "train_priors",
        }
    }
    config.update(
        {
            f"{name}_sha256": bindings[name]["sha256"]
            for name in (
                "raw_index",
                "target_index",
                "split_manifest",
                "class_weight_artifact",
                "train_priors",
            )
        }
    )
    ssl = bindings["ssl_checkpoint"]
    config.update(
        {
            "ssl_checkpoint": ssl["path"],
            "ssl_checkpoint_sha256": ssl["sha256"],
            "ssl_encoder_export": ssl["encoder_export_path"],
            "ssl_encoder_export_sha256": ssl["encoder_export_sha256"],
            "ssl_source_kind": ssl["source_kind"],
            "raw_cache_root": bindings["raw_cache_root"],
            "target_cache_root": bindings["target_cache_root"],
            "learning_rate": parent_plan["protocol"]["schedule"][
                "learning_rate"
            ],
            "git_head": "b" * 40,
        }
    )
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _evaluate(checkpoint: Path) -> dict[str, object]:
    batch = _full_batch()
    identities = tuple(zip(batch.dataset_ids, batch.piece_ids, strict=True))
    return evaluate_dilemmadata_model(
        evaluation_model(checkpoint, torch.device("cpu")),
        (batch,),
        component_by_identity={
            identity: f"component-{index}"
            for index, identity in enumerate(identities)
        },
        membership_fingerprint="e" * 64,
    )


def _replace_parent_final_metrics(parent: Path) -> None:
    for cell_id in PHASE9CC_CELLS:
        directory = parent / "cells" / cell_id
        checkpoint = directory / "checkpoints" / "update-12.pt"
        report = _evaluate(checkpoint)
        report_path = directory / "milestones" / "update-12.json"
        parent_runner._write(report_path, report)
        milestones = parent_runner._read(
            directory / "validation_milestones.json"
        )
        row = next(
            value for value in milestones["milestones"] if value["update"] == 12
        )
        row.update(
            {
                "validation_report_sha256": file_sha256(report_path),
                "validation_report_fingerprint": report["fingerprint"],
                "validation_membership_fingerprint": "e" * 64,
                "aggregate": report["aggregate"],
                "tasks": report["tasks"],
            }
        )
        unsigned = dict(milestones)
        unsigned.pop("fingerprint")
        milestones["fingerprint"] = parent_runner.fingerprint(unsigned)
        parent_runner._write(directory / "validation_milestones.json", milestones)


def _parent_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], tuple[tuple[str, str], ...], Path]:
    inputs = tmp_path / "parent-inputs"
    inputs.mkdir()
    parent_plan, identities = _bounded_plan(inputs)
    monkeypatch.setattr(
        training_engine,
        "build_data_runtime",
        lambda config, seed: _runtime(12),
    )
    monkeypatch.setattr(parent_training, "_schedule", lambda plan: identities)
    source_roots = {}
    for cell in parent_plan["cells"]:
        directory = tmp_path / f"parent-source-{cell['cell_id']}"
        parent_training.run_cell_training(
            parent_plan, cell, directory, action="run", device="cpu"
        )
        source_roots[cell["cell_id"]] = directory
    parent = tmp_path / "parent-bundle"
    _complete_bundle(parent, parent_plan, source_roots)
    _replace_parent_final_metrics(parent)
    parent_runner.aggregate(parent, parent_plan)
    config_path = _config(parent_plan, tmp_path / "parent-config.json")
    return parent, parent_plan, identities, config_path


def _continuation_plan(
    parent: Path,
    parent_plan: dict[str, object],
    identities: tuple[tuple[str, str], ...],
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    manifest = parent_runner._read(parent / "manifest.json")
    hashes = {
        cell_id: file_sha256(
            parent / "cells" / cell_id / "checkpoints" / "update-12.pt"
        )
        for cell_id in PHASE9CC_CELLS
    }
    monkeypatch.setattr(contracts, "_git_head", lambda: "b" * 40)
    monkeypatch.setattr(contracts, "_git_branch", lambda: "phase/test")
    return contracts.build_continuation_plan(
        parent,
        config_path,
        start_update=12,
        target_update=24,
        validation_milestones=(12, 18, 24),
        _bounded_protocol={
            "telemetry_interval": 2,
            "checkpoint_interval": 6,
            "maximum_consecutive_skips": 2,
            "schedule_identities": identities + identities,
            "parent_manifest_fingerprint": manifest["fingerprint"],
            "parent_git_sha": parent_plan["protocol"]["git_head"],
            "parent_git_branch": "phase/test-parent",
            "parent_checkpoint_sha256": hashes,
        },
    )


def _install_continuation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        training_engine,
        "build_data_runtime",
        lambda config, seed: _runtime(24),
    )


def _install_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    def run_evaluation(plan, checkpoint, report_path, *, device):
        del plan, device
        if report_path.is_file():
            return parent_runner._read(report_path)
        report = _evaluate(checkpoint)
        parent_runner._write(report_path, report)
        report_path.with_suffix(".log").write_text("bounded evaluation\n")
        return report

    monkeypatch.setattr(runner, "_run_evaluation", run_evaluation)


def test_bounded_exact_continuation_preflight_resume_and_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, parent_plan, identities, config_path = _parent_bundle(
        tmp_path, monkeypatch
    )
    plan = _continuation_plan(
        parent, parent_plan, identities, config_path, monkeypatch
    )
    _install_continuation_runtime(monkeypatch)
    _install_evaluator(monkeypatch)
    parent_hashes_before = {
        str(path.relative_to(parent)): file_sha256(path)
        for path in parent.rglob("*")
        if path.is_file()
    }

    uninterrupted = tmp_path / "continuation"
    result = runner.execute(uninterrupted, plan, device="cpu")
    assert result["status"] == "execution_complete_pending_manifest"
    (uninterrupted / "execution.log").write_text("bounded execution\n")
    verified = runner.finalize(uninterrupted, expected_sha="b" * 40)
    assert verified["status"] == "verified"
    assert verified["start_applied_update"] == 12
    assert verified["final_applied_update"] == 24
    assert parent_hashes_before == {
        str(path.relative_to(parent)): file_sha256(path)
        for path in parent.rglob("*")
        if path.is_file()
    }

    extension_config = _config(plan, tmp_path / "extension-config.json")
    extension_manifest = parent_runner._read(uninterrupted / "manifest.json")
    extension_hashes = {
        cell_id: file_sha256(
            uninterrupted
            / "cells"
            / cell_id
            / "checkpoints"
            / "update-24.pt"
        )
        for cell_id in PHASE9CC_CELLS
    }
    extended_identities = identities + identities + identities[:8]
    extension_plan = contracts.build_continuation_plan(
        uninterrupted,
        extension_config,
        start_update=24,
        target_update=28,
        validation_milestones=(24, 26, 28),
        _bounded_protocol={
            "telemetry_interval": 2,
            "checkpoint_interval": 2,
            "maximum_consecutive_skips": 2,
            "schedule_identities": extended_identities,
            "parent_manifest_fingerprint": extension_manifest["fingerprint"],
            "parent_git_sha": plan["protocol"]["git_head"],
            "parent_git_branch": "phase/test-parent-continuation",
            "parent_checkpoint_sha256": extension_hashes,
        },
    )
    assert extension_plan["protocol"]["continuation_generation"] == 2
    monkeypatch.setattr(
        training_engine,
        "build_data_runtime",
        lambda config, seed: _runtime(28),
    )
    extension_root = tmp_path / "extension"
    extension_result = runner.execute(extension_root, extension_plan, device="cpu")
    assert extension_result["status"] == "execution_complete_pending_manifest"
    (extension_root / "execution.log").write_text("bounded extension\n")
    extension_verified = runner.finalize(extension_root, expected_sha="b" * 40)
    assert extension_verified["start_applied_update"] == 24
    assert extension_verified["final_applied_update"] == 28
    for cell_id in PHASE9CC_CELLS:
        extension_training = parent_runner._read(
            extension_root / "cells" / cell_id / "training_report.json"
        )
        assert extension_training["restore_mode"] == (
            "model_optimizer_scaler_scheduler_rng_sampler"
        )
        assert extension_training["skipped_updates"] == 0
        assert [
            row["applied_updates"]
            for row in parent_runner._rows(
                extension_root / "cells" / cell_id / "train_telemetry.jsonl"
            )
        ] == [26, 28]
    for cell_id in PHASE9CC_CELLS:
        directory = uninterrupted / "cells" / cell_id
        report = parent_runner._read(directory / "training_report.json")
        assert report["encoder_export_reloaded"] is False
        assert report["applied_updates"] == 24
        assert [
            row["applied_updates"]
            for row in parent_runner._rows(directory / "train_telemetry.jsonl")
        ] == [14, 16, 18, 20, 22, 24]
        assert parent_runner._read(directory / "preflight_evidence.json")[
            "passed"
        ] is True
        preflight = parent_runner._read(directory / "preflight_evidence.json")
        assert preflight["logit_replay"]["candidate_identities_exact"] is True
        assert preflight["metric_comparison"]["within_tolerance"] is True
        assert len(preflight["next_train_sample_identities"]) == 2
        assert [
            row["update"]
            for row in parent_runner._read(
                directory / "validation_milestones.json"
            )["milestones"]
        ] == [12, 18, 24]

    scratch = plan["cells"][0]
    interrupted = tmp_path / "interrupted-scratch"
    partial = training.run_cell_training(
        plan,
        scratch,
        interrupted,
        action="run",
        device="cpu",
        stop_after_applied=18,
    )
    assert partial["complete"] is False
    resumed = training.run_cell_training(
        plan, scratch, interrupted, action="resume", device="cpu"
    )
    full = parent_runner._read(
        uninterrupted / "cells" / "scratch_mlp" / "training_report.json"
    )
    assert resumed["final_model_state_fingerprint"] == full[
        "final_model_state_fingerprint"
    ]
    assert parent_runner._rows(interrupted / "train_telemetry.jsonl") == (
        parent_runner._rows(
            uninterrupted / "cells" / "scratch_mlp" / "train_telemetry.jsonl"
        )
    )

    original_optimize = training_engine._optimize_batch
    calls = {"count": 0}

    def skip_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return None, {"optimizer_step_applied": False}, True
        return original_optimize(*args, **kwargs)

    monkeypatch.setattr(training_engine, "_optimize_batch", skip_once)
    skipped = training.run_cell_training(
        plan, scratch, tmp_path / "skip-once", action="run", device="cpu"
    )
    assert skipped["applied_updates"] == 24
    assert skipped["attempted_updates"] == 25
    assert skipped["skipped_updates"] == 1
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
        training.run_cell_training(
            plan, scratch, tmp_path / "persistent-skip", action="run", device="cpu"
        )


def test_continuation_rejects_prefix_checkpoint_and_bundle_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, parent_plan, identities, config_path = _parent_bundle(
        tmp_path, monkeypatch
    )
    valid = _continuation_plan(
        parent, parent_plan, identities, config_path, monkeypatch
    )
    parent_sha_config = tmp_path / "parent-sha-config.json"
    parent_sha_value = json.loads(config_path.read_text(encoding="utf-8"))
    parent_sha_value["git_head"] = parent_plan["protocol"]["git_head"]
    parent_sha_config.write_text(
        json.dumps(parent_sha_value), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="config_scientific_binding_mismatch"):
        _continuation_plan(
            parent,
            parent_plan,
            identities,
            parent_sha_config,
            monkeypatch,
        )
    manifest = parent_runner._read(parent / "manifest.json")
    hashes = {
        cell_id: file_sha256(
            parent / "cells" / cell_id / "checkpoints" / "update-12.pt"
        )
        for cell_id in PHASE9CC_CELLS
    }
    with pytest.raises(ValueError, match="schedule_prefix_mismatch"):
        contracts.build_continuation_plan(
            parent,
            config_path,
            start_update=12,
            target_update=24,
            validation_milestones=(12, 18, 24),
            _bounded_protocol={
                "telemetry_interval": 2,
                "checkpoint_interval": 6,
                "schedule_identities": tuple(reversed(identities)) + identities,
                "parent_manifest_fingerprint": manifest["fingerprint"],
                "parent_git_sha": parent_plan["protocol"]["git_head"],
                "parent_checkpoint_sha256": hashes,
            },
        )

    wrong_hashes = copy.deepcopy(hashes)
    wrong_hashes["scratch_mlp"] = "0" * 64
    with pytest.raises(ValueError, match="parent_checkpoint_invalid"):
        contracts.build_continuation_plan(
            parent,
            config_path,
            start_update=12,
            target_update=24,
            validation_milestones=(12, 18, 24),
            _bounded_protocol={
                "telemetry_interval": 2,
                "checkpoint_interval": 6,
                "schedule_identities": identities + identities,
                "parent_manifest_fingerprint": manifest["fingerprint"],
                "parent_git_sha": parent_plan["protocol"]["git_head"],
                "parent_checkpoint_sha256": wrong_hashes,
            },
        )

    _install_continuation_runtime(monkeypatch)
    _install_evaluator(monkeypatch)
    bundle = tmp_path / "bundle"
    runner.execute(bundle, valid, device="cpu")
    (bundle / "execution.log").write_text("bounded execution\n")
    runner.finalize(bundle, expected_sha="b" * 40)
    missing = tmp_path / "missing-checkpoint"
    shutil.copytree(bundle, missing)
    (missing / "cells" / "scratch_mlp" / "checkpoints" / "update-18.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint_inventory_invalid"):
        runner.verify_bundle(missing, expected_sha="b" * 40)
    telemetry_path = bundle / "cells" / "scratch_mlp" / "train_telemetry.jsonl"
    telemetry = parent_runner._rows(telemetry_path)
    telemetry.append(copy.deepcopy(telemetry[-1]))
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="telemetry_invalid"):
        runner.verify_bundle(bundle, expected_sha="b" * 40)


def test_both_preflights_finish_before_first_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "protocol": {
            "bounded_test_protocol": True,
            "git_head": "b" * 40,
            "parent_binding": {},
        },
        "cells": [{"cell_id": value} for value in PHASE9CC_CELLS],
    }
    events = []
    monkeypatch.setattr(
        runner,
        "_preflight_cell",
        lambda root, plan, cell, device: events.append(
            f"preflight:{cell['cell_id']}"
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_cell_training",
        lambda plan, cell, directory, action, device: (
            events.append(f"train:{cell['cell_id']}")
            or {"complete": True}
        ),
    )
    monkeypatch.setattr(runner, "evaluate_milestones", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner, "aggregate", lambda root, plan: {"fingerprint": "f" * 64}
    )
    runner.execute(tmp_path / "unused", plan, device="cpu")
    assert events == [
        "preflight:scratch_mlp",
        "preflight:ssl_mlp",
        "train:scratch_mlp",
        "train:ssl_mlp",
    ]
