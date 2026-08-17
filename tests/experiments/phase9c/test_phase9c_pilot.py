from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile

import pytest

from music_critic.experiments.phase9c import (
    OPTIONAL_VARIANTS,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PRIMARY_VARIANTS,
    Phase9CContractError,
    build_experiment_plan,
    build_source_balanced_schedule,
    component_bootstrap_primary_delta,
    execute_experiment,
    primary_validation_summary,
    resolve_preset,
    safe_extract_members,
    select_checkpoint,
    verify_bundle,
)
from music_critic.experiments.phase9c.artifacts import read_json
from music_critic.experiments.phase9c.contracts import validate_test_lock


def test_variant_registry_and_presets_keep_optional_ablations_explicit() -> None:
    bounded = resolve_preset("bounded_acceptance")
    primary = resolve_preset("one_seed_primary_pilot")
    full = resolve_preset("one_seed_full_ablation")
    assert tuple(bounded.variants) == PRIMARY_VARIANTS
    assert tuple(primary.variants) == PRIMARY_VARIANTS
    assert not set(primary.variants) & set(OPTIONAL_VARIANTS)
    assert set(full.variants) == set(PRIMARY_VARIANTS) | set(OPTIONAL_VARIANTS)
    assert primary.bootstrap_replicates >= 1000
    assert not primary.production_budget_resolved


def test_paired_initialization_raw_schedule_mixture_and_compute() -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    protocol = plan["protocol"]
    assert protocol["seed"] == 17
    assert protocol["mixture"]["target_blind"] is True
    assert protocol["mixture"]["validation_and_test_excluded"] is True
    schedules = list(plan["variant_schedules"].values())
    assert {row["encoder_forward_count"] for row in schedules} == {
        PHASE9C_ENCODER_FORWARDS_PER_UPDATE
    }
    assert len({row["sample_schedule_fingerprint"] for row in schedules}) == 1
    assert len({cell["initial_encoder_fingerprint"] for cell in plan["ssl_cells"]}) == 1
    assert len({cell["fresh_head_fingerprint"] for cell in plan["downstream_cells"]}) == 1
    assert plan["ssl_sample_schedule"]["dataset_counts"] == {
        "dilemmadata": 1,
        "hooktheory": 1,
        "pop909_cl": 0,
    } or sum(plan["ssl_sample_schedule"]["dataset_counts"].values()) == 2


def test_source_balanced_cycles_are_deterministic_and_no_replacement() -> None:
    identities = {"a": ("a0", "a1"), "b": ("b0", "b1")}
    first = build_source_balanced_schedule(
        identities, weights={"a": 1.0, "b": 1.0}, sample_count=12, seed=17
    )
    second = build_source_balanced_schedule(
        identities, weights={"a": 1.0, "b": 1.0}, sample_count=12, seed=17
    )
    assert first == second
    assert first["dataset_counts"] == {"a": 6, "b": 6}
    assert first["repeat_counts"] == {"a": 4, "b": 4}
    assert first["replacement_within_cycle"] is False
    by_source_cycle: dict[tuple[str, int], list[str]] = {}
    for row in first["slots"]:
        by_source_cycle.setdefault(
            (row["dataset_id"], row["cycle_index"]), []
        ).append(row["piece_id"])
    assert all(len(rows) == len(set(rows)) for rows in by_source_cycle.values())


def _validation_report(offset: float = 0.0) -> dict[str, object]:
    tasks = {}
    entries = []
    task_ids = (
        "dilemmadata.an.chord.inversion",
        "dilemmadata.an.chord.quality",
        "dilemmadata.dlc.chord.inversion",
        "dilemmadata.dlc.chord.quality",
    )
    for task_index, task_id in enumerate(task_ids):
        class_count = 4 if task_id.endswith("inversion") else 8
        nll = 1.0 + offset + task_index / 10
        tasks[task_id] = {
            "available": True,
            "class_count": class_count,
            "nll": nll,
            "macro_f1": 0.5 - offset,
        }
        for component in range(3):
            entries.append(
                {
                    "task_id": task_id,
                    "dataset_id": "dilemmadata",
                    "piece_id": f"piece-{component}",
                    "component_fingerprint": f"component-{component}",
                    "source_entry_index": task_index,
                    "label": 0,
                    "log_probabilities": [-nll] + [-nll - 1] * (class_count - 1),
                }
            )
    return {"tasks": tasks, "entry_predictions": entries}


def test_primary_score_tie_breakers_and_component_bootstrap() -> None:
    better = primary_validation_summary(_validation_report(0.0))
    worse = primary_validation_summary(_validation_report(0.1))
    assert better["primary_score"] < worse["primary_score"]
    rows = [
        {
            "validation_summary": worse,
            "epoch": 0,
            "checkpoint_identity": "b",
        },
        {
            "validation_summary": better,
            "epoch": 1,
            "checkpoint_identity": "a",
        },
    ]
    selected = select_checkpoint(rows)
    assert selected["selected_checkpoint_identity"] == "a"
    bootstrap = component_bootstrap_primary_delta(
        _validation_report(0.0),
        _validation_report(0.1),
        seed=17,
        replicates=100,
    )
    assert bootstrap["unit"] == "component"
    assert bootstrap["component_count"] == 3
    assert bootstrap["observed_delta"] > 0
    assert "optimization-seed" in bootstrap["interpretation"]


def test_test_lock_and_plan_never_serialize_test_identities() -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    validate_test_lock(plan["protocol"]["test_lock"])
    projection = plan["data_semantic_projection"]
    assert projection["test_identities_serialized"] is False
    assert projection["target_bundles_loaded_during_planning"] is False
    damaged = copy.deepcopy(plan["protocol"]["test_lock"])
    damaged["test_inference"] = True
    with pytest.raises(Phase9CContractError, match="test_lock.invalid"):
        validate_test_lock(damaged)


def test_bounded_dag_resume_aggregate_select_verify_and_transfer(tmp_path: Path) -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    root = tmp_path / "bundle"
    stopped = execute_experiment(root, plan, action="run", fail_after_cell=5)
    assert stopped["status"] == "stopped"
    result = execute_experiment(root, plan, action="resume")
    assert result["status"] == "complete"
    assert result["production_pilot_executed"] is False
    assert verify_bundle(root)["status"] == "verified"

    for variant in PRIMARY_VARIANTS[1:]:
        report = read_json(root / "cells" / "ssl" / variant / "training_report.json")
        assert report["actual_encoder_forward_count"] == 12
        assert report["applied_optimizer_updates"] == 1
        assert report["retained_prediction_tensor_count"] == 0
        assert report["lifecycle_allocated_growth_bytes"] == 0
    frozen = read_json(
        root / "cells" / "downstream" / "phase7a_control" / "frozen_probe" / "training_report.json"
    )
    tuned = read_json(
        root / "cells" / "downstream" / "phase7a_control" / "full_finetune" / "training_report.json"
    )
    assert frozen["frozen_encoder_bit_exact"] is True
    assert frozen["fresh_optimizer"] is frozen["fresh_scheduler"] is frozen["fresh_scaler"] is True
    assert tuned["full_finetune_finite_encoder_gradients"] is True
    assert tuned["full_finetune_encoder_changed"] is True
    assert tuned["head_logits_dtype"] == tuned["ce_dtype"] == tuned["total_loss_dtype"] == "float32"

    aggregate = execute_experiment(root, plan, action="aggregate")
    selected = execute_experiment(root, plan, action="select")
    assert aggregate["status"] == "aggregated"
    assert selected["status"] == "selected"
    comparison = read_json(root / "final_comparison_report.json")
    assert comparison["test_access"] is False
    assert {row["transfer_mode"] for row in comparison["rows"]} >= {
        "frozen_probe",
        "full_finetune",
        "scratch_frozen_probe",
        "scratch_full_finetune",
    }


def test_artifact_corruption_and_unsafe_tar_are_rejected(tmp_path: Path) -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    root = tmp_path / "bundle"
    execute_experiment(root, plan, action="run")
    (root / "comparison_table.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(Phase9CContractError, match="artifact_corruption"):
        verify_bundle(root)

    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../escape")
        payload = b"bad"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(Phase9CContractError, match="unsafe_tar_member"):
        safe_extract_members(archive)


def test_production_budget_must_be_explicit() -> None:
    preset = resolve_preset("one_seed_primary_pilot")
    assert preset.production_budget_resolved is False
