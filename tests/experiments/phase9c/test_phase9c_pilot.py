from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
from omegaconf import OmegaConf

from music_critic.experiments.phase9c import (
    OPTIONAL_VARIANTS,
    PHASE9C_ENCODER_FORWARDS_PER_UPDATE,
    PRIMARY_VARIANTS,
    Phase9CContractError,
    build_experiment_plan,
    build_source_balanced_schedule,
    compose_ssl_split_manifest,
    component_bootstrap_primary_delta,
    execute_experiment,
    materialize_ssl_split_manifest,
    primary_validation_summary,
    profile_experiment,
    resolve_preset,
    safe_extract_members,
    verify_bundle,
)
from music_critic.experiments.phase9c.artifacts import read_json
from music_critic.experiments.phase9c.contracts import (
    fingerprint,
    validate_test_lock,
)
from music_critic.evaluation import DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION
from music_critic.experiments.phase9c.runner import (
    _downstream_command,
    _ssl_command,
    _validation_command,
)
from music_critic.ssl.data import IndexedSSLRawDataset, load_ssl_eligibility_manifest
from music_critic.tasks import (
    CorpusCacheConfig,
    IndexedCorpusRecord,
    create_split_manifest,
    dumps_corpus_index,
    dumps_split_manifest,
    load_split_manifest,
    make_corpus_index,
    MultiCorpusDataset,
    validate_split_manifest,
)
from music_critic.models import DILEMMADATA_ACTIVE_TASK_IDS
from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK
from music_critic.training.engine import _validate_config


def _fixture_index(dataset_id: str, splits: tuple[str, ...]):
    digest = sha256(dataset_id.encode()).hexdigest()
    records = tuple(
        IndexedCorpusRecord(
            dataset_id=dataset_id,
            piece_id=f"{dataset_id}-{ordinal}",
            source_group_id=f"{dataset_id}-source-{ordinal}",
            lineage_group_id=f"{dataset_id}-lineage-{ordinal}",
            source_identity=f"{dataset_id}-identity-{ordinal}",
            source_relative_path=f"source/{ordinal}.json",
            source_sha256=sha256(f"source-{dataset_id}-{ordinal}".encode()).hexdigest(),
            cache_key=sha256(f"cache-{dataset_id}-{ordinal}".encode()).hexdigest(),
            canonical_relative_path=f"canonical/{ordinal}.json",
            canonical_sha256=sha256(f"canonical-{dataset_id}-{ordinal}".encode()).hexdigest(),
            target_availability=(),
            suggested_split=split,
        )
        for ordinal, split in enumerate(splits)
    )
    return make_corpus_index(
        dataset_id=dataset_id,
        adapter_name="phase9c_fixture",
        adapter_version="1.0.0",
        adapter_config_fingerprint=digest,
        source_identity=f"{dataset_id}-fixture",
        source_fingerprint=digest,
        creation_policy="phase9c_manifest_composition_test",
        records=records,
    )


def _manifest(indices, *, seed: int):
    return create_split_manifest(
        indices,
        {
            (row.dataset_id, row.piece_id): row.suggested_split
            for index in indices
            for row in index.records
        },
        seed=seed,
        policy="existing_fixture_assignment",
    )


def _cached_fixture_index(
    root: Path,
    dataset_id: str,
    rows: tuple[tuple[str, int], ...],
):
    base = _fixture_index(dataset_id, tuple(split for split, _ in rows))
    cache = CorpusCacheConfig(root)
    records = []
    for record, (_split, note_count) in zip(base.records, rows, strict=True):
        value = {
            "dataset_name": dataset_id,
            "piece_id": record.piece_id,
            "bars": [
                {
                    "start_qn": {"num": 0, "den": 1},
                    "duration_qn": {"num": 4, "den": 1},
                },
                {
                    "start_qn": {"num": 4, "den": 1},
                    "duration_qn": {"num": 4, "den": 1},
                },
            ],
            "notes": [
                {
                    "onset_qn": {
                        "num": 0 if ordinal % 2 == 0 else 4,
                        "den": 1,
                    }
                }
                for ordinal in range(note_count)
            ],
        }
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        path = cache.root / cache.namespace / record.canonical_relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        records.append(replace(record, canonical_sha256=sha256(raw).hexdigest()))
    return make_corpus_index(
        dataset_id=dataset_id,
        adapter_name=base.header.adapter_name,
        adapter_version=base.header.adapter_version,
        adapter_config_fingerprint=base.header.adapter_config_fingerprint,
        source_identity=base.header.source_identity,
        source_fingerprint=base.header.source_fingerprint,
        creation_policy=base.header.creation_policy,
        records=tuple(records),
    )


def test_train_only_class_weight_worker_emits_zero_safe_artifact(
    tmp_path: Path,
) -> None:
    prior_payload = {
        "contract_version": DILEMMADATA_TRAIN_PRIOR_CONTRACT_VERSION,
        "source_split": "train_only",
        "train_membership_fingerprint": "a" * 64,
        "tasks": {
            task_id: {
                "class_counts": [
                    0 if index == 0 else index
                    for index, _ in enumerate(
                        DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary
                    )
                ]
            }
            for task_id in DILEMMADATA_ACTIVE_TASK_IDS
        },
    }
    priors = {**prior_payload, "fingerprint": fingerprint(prior_payload)}
    source = tmp_path / "train_priors.json"
    output = tmp_path / "class_weights.json"
    source.write_text(json.dumps(priors), encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase9c.worker",
            "build-class-weights",
            "--train-priors",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["policy"] == "inverse_sqrt_frequency_supported"
    assert artifact["source_split"] == "train_only"
    assert all(
        weights[0] == 0.0 for weights in artifact["weights"].values()
    )

    priors["source_split"] = "validation"
    source.write_text(json.dumps(priors), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase9c.worker",
            "build-class-weights",
            "--train-priors",
            str(source),
            "--output",
            str(tmp_path / "rejected.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "phase9c.class_weights.train_priors_invalid" in rejected.stderr


def test_ssl_manifest_composition_preserves_assignments_and_holdouts(tmp_path: Path) -> None:
    hook = _fixture_index("hooktheory", ("train", "validation"))
    pop = _fixture_index("pop909_cl", ("train", "test"))
    dilemma = _fixture_index("dilemmadata", ("train", "validation", "test"))
    base = _manifest((hook, pop), seed=11)
    downstream = _manifest((dilemma,), seed=13)

    composed, evidence = compose_ssl_split_manifest(
        (hook, pop, dilemma), (base, downstream)
    )
    validate_split_manifest(composed, (hook, pop, dilemma))
    assert evidence["assignments_preserved_exactly"] is True
    assert evidence["dilemmadata_validation_test_excluded_from_ssl_train"] is True
    assert {
        (row.dataset_id, row.piece_id): row
        for row in composed.assignments
    } == {
        (row.dataset_id, row.piece_id): row
        for manifest in (base, downstream)
        for row in manifest.assignments
    }

    index_paths = []
    for index in (hook, pop, dilemma):
        path = tmp_path / f"{index.header.dataset_id}.index.json"
        path.write_text(dumps_corpus_index(index), encoding="utf-8")
        index_paths.append(str(path))
    source_paths = []
    for name, manifest in (("base", base), ("dilemma", downstream)):
        path = tmp_path / f"{name}.split.json"
        path.write_text(dumps_split_manifest(manifest), encoding="utf-8")
        source_paths.append(str(path))
    destination = tmp_path / "all-three.split.json"
    config = materialize_ssl_split_manifest(
        {
            "preset": "rtx_profile",
            "data": {
                "ssl_index_paths": index_paths,
                "ssl_source_split_manifests": source_paths,
                "ssl_split_manifest": str(destination),
            },
        }
    )
    first_bytes = destination.read_bytes()
    materialize_ssl_split_manifest(config)
    assert destination.read_bytes() == first_bytes
    assert load_split_manifest(destination) == composed
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(Phase9CContractError, match="destination_conflict"):
        materialize_ssl_split_manifest(config)


def test_ssl_eligibility_excludes_zero_note_record_without_repartitioning(
    tmp_path: Path,
) -> None:
    cache_roots = {
        dataset_id: tmp_path / f"{dataset_id}-cache"
        for dataset_id in ("hooktheory", "pop909_cl", "dilemmadata")
    }
    hook = _cached_fixture_index(
        cache_roots["hooktheory"],
        "hooktheory",
        (("train", 2), ("validation", 0), ("validation", 2)),
    )
    pop = _cached_fixture_index(
        cache_roots["pop909_cl"],
        "pop909_cl",
        (("train", 2), ("validation", 2)),
    )
    dilemma = _cached_fixture_index(
        cache_roots["dilemmadata"],
        "dilemmadata",
        (("train", 2), ("validation", 2), ("test", 2)),
    )
    source_manifests = (
        _manifest((hook, pop), seed=11),
        _manifest((dilemma,), seed=13),
    )
    index_paths = []
    for index in (hook, pop, dilemma):
        path = tmp_path / f"{index.header.dataset_id}.index.json"
        path.write_text(dumps_corpus_index(index), encoding="utf-8")
        index_paths.append(str(path))
    source_paths = []
    for ordinal, manifest in enumerate(source_manifests):
        path = tmp_path / f"source-{ordinal}.split.json"
        path.write_text(dumps_split_manifest(manifest), encoding="utf-8")
        source_paths.append(str(path))
    destination = tmp_path / "all-three.split.json"
    prepared = materialize_ssl_split_manifest(
        {
            "preset": "rtx_profile",
            "data": {
                "ssl_index_paths": index_paths,
                "ssl_cache_roots": [
                    str(cache_roots[index.header.dataset_id])
                    for index in (hook, pop, dilemma)
                ],
                "ssl_source_split_manifests": source_paths,
                "ssl_split_manifest": str(destination),
            },
        }
    )
    identities, evidence = load_ssl_eligibility_manifest(
        prepared["data"]["ssl_eligibility_manifest"]
    )
    empty_identity = ("hooktheory", hook.records[1].piece_id)
    assert empty_identity not in identities
    assert ("hooktheory", hook.records[2].piece_id) in identities
    assert evidence["split_assignments_changed"] is False
    assert evidence["target_or_provenance_access"] is False
    composed = load_split_manifest(destination)
    assert {
        (row.dataset_id, row.piece_id): row.split
        for row in composed.assignments
    } == {
        (row.dataset_id, row.piece_id): row.split
        for manifest in source_manifests
        for row in manifest.assignments
    }
    indexed = tuple(
        IndexedSSLRawDataset(
            index,
            cache_config=CorpusCacheConfig(cache_roots[index.header.dataset_id]),
        )
        for index in (hook, pop, dilemma)
    )
    validation = MultiCorpusDataset(
        indexed,
        composed,
        split="validation",
        included_identities=identities,
    )
    assert empty_identity not in {
        validation.record_identity(index) for index in range(len(validation))
    }


def test_production_profile_and_run_validation_command_uses_fixed_budget_last(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    engine = root / "cells" / "downstream" / "variant" / "full_finetune" / "engine"
    engine.mkdir(parents=True)
    (engine / "last.pt").write_bytes(b"fixed-budget")
    (engine / "training_report.json").write_text(
        json.dumps(
            {
                "optimizer_step_attempt_count": 20,
                "optimizer_step_applied_count": 20,
                "optimizer_step_skipped_count": 0,
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "runtime_paths": {
            "downstream_raw_index": "/raw.index.json",
            "downstream_raw_cache_root": "/raw-cache",
            "target_cache_index": "/target.index.json",
            "target_cache_root": "/target-cache",
            "downstream_split_manifest": "/dilemmadata.split.json",
        },
        "protocol": {"preset": {"batch_size": 2}},
    }
    cell = {
        "cell_id": "validation/variant/full_finetune",
        "depends_on": "downstream/variant/full_finetune",
        "optimizer_update_budget": 20,
        "comparison_checkpoint": "last.pt",
    }
    command = _validation_command(root, plan, cell, tmp_path / "staging")
    checkpoint = command[command.index("--checkpoint") + 1]
    assert checkpoint.endswith("/engine/last.pt")
    assert "best.pt" not in command

    damaged = json.loads((engine / "training_report.json").read_text(encoding="utf-8"))
    damaged["optimizer_step_applied_count"] = 19
    (engine / "training_report.json").write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(Phase9CContractError, match="fixed_budget_binding_invalid"):
        _validation_command(root, plan, cell, tmp_path / "staging")


def test_generated_ssl_command_composes_with_official_hydra_cli(tmp_path: Path) -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    plan["runtime_paths"] = {
        "ssl_index_paths": ["/indices/hook.json", "/indices/pop.json", "/indices/dilemma.json"],
        "ssl_cache_roots": ["/cache/hook", "/cache/pop", "/cache/dilemma"],
        "ssl_split_manifest": "/splits/all-three.json",
        "ssl_eligibility_manifest": "/splits/all-three.eligibility.json",
    }
    generated = _ssl_command(
        plan,
        plan["ssl_cells"][0],
        tmp_path / "staging",
    )
    command = [*generated[:3], "--cfg", "job", *generated[3:]]
    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert any(value.startswith("+data.mixture_weights=") for value in generated)
    assert (
        "data.ssl_eligibility_manifest=/splits/all-three.eligibility.json"
        in generated
    )
    assert "dilemmadata: 0.3333333333333333" in result.stdout
    assert "hooktheory: 0.3333333333333333" in result.stdout
    assert "pop909_cl: 0.3333333333333333" in result.stdout


def test_generated_scratch_commands_compose_and_satisfy_transfer_contract(
    tmp_path: Path,
) -> None:
    plan = build_experiment_plan({"preset": "bounded_acceptance"})
    plan["runtime_paths"].update(
        {
            "downstream_raw_index": "/indices/dilemmadata.json",
            "downstream_raw_cache_root": "/cache/dilemmadata",
            "target_cache_index": "/indices/dilemmadata-target.json",
            "target_cache_root": "/cache/dilemmadata-target",
            "downstream_split_manifest": "/splits/dilemmadata.json",
        }
    )
    root = tmp_path / "candidate"
    scratch_export = (
        root
        / "cells"
        / "encoder_export"
        / "initial_scratch"
        / "engine"
        / "encoder.pt"
    )
    scratch_export.parent.mkdir(parents=True)
    scratch_export.write_bytes(b"paired-untrained-encoder")

    cells = {
        str(cell["transfer_mode"]): cell
        for cell in plan["downstream_cells"]
        if cell["variant_id"] == "scratch"
    }

    def compose_and_validate(
        cell: dict[str, object], name: str
    ) -> dict[str, object]:
        generated = _downstream_command(root, plan, cell, tmp_path / name)
        command = [*generated[:3], "--cfg", "job", *generated[3:]]
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        config = OmegaConf.to_container(
            OmegaConf.create(result.stdout), resolve=True
        )
        assert isinstance(config, dict)
        _validate_config(config)
        return config

    frozen = compose_and_validate(cells["scratch_frozen_probe"], "frozen")
    assert frozen["transfer"]["mode"] == "frozen_probe"
    assert frozen["transfer"]["encoder_export_path"] == str(scratch_export)
    assert frozen["transfer"]["source_kind"] == "phase6_hierarchical"

    full = compose_and_validate(cells["scratch_full_finetune"], "full")
    assert full["transfer"]["mode"] == "supervised_scratch"
    assert full["transfer"]["encoder_export_path"] == ""
    assert full["transfer"]["encoder_export_sha256"] == ""
    assert full["transfer"]["source_ssl_checkpoint_sha256"] == ""


def test_profile_cli_fails_closed_when_no_candidate_passes(tmp_path: Path) -> None:
    root = tmp_path / "failed-profile"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase9c.run",
            "profile",
            "--preset",
            "bounded_acceptance",
            "--profile-batch-candidates",
            "16",
            "--output-root",
            str(root),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    report = read_json(root / "profile_report.json")
    assert report["status"] == "no_candidate_passed"
    assert report["results"][0]["status"] == "oom"
    assert report["production_started"] is False
    assert "phase9c.rtx.profile.complete" not in result.stdout


def test_failed_production_profile_preserves_candidate_root_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_experiment_plan(
        {"preset": "bounded_acceptance", "profile_batch_candidates": [3]}
    )
    plan["data_semantic_projection"] = {"kind": "production"}
    plan["profile_rebuild_config"] = {"preset": "rtx_profile"}

    def fail_candidate(command, **_kwargs):
        candidate_root = Path(command[-2])
        candidate_root.mkdir(parents=True, exist_ok=True)
        (candidate_root / "failure.txt").write_text("preserved", encoding="utf-8")
        return subprocess.CompletedProcess(command, 42, "", "failed")

    monkeypatch.setattr(
        "music_critic.experiments.phase9c.runner.subprocess.run",
        fail_candidate,
    )
    root = tmp_path / "profile"
    report = profile_experiment(root, plan)
    candidate_root = root / ".profile" / "candidate-3"
    assert report["status"] == "no_candidate_passed"
    assert report["results"][0]["candidate_root"] == str(candidate_root)
    assert report["results"][0]["candidate_root_preserved"] is True
    assert (candidate_root / "failure.txt").read_text(encoding="utf-8") == "preserved"
    assert (candidate_root / "profile_subprocess_stderr.log").read_text(
        encoding="utf-8"
    ) == "failed"
    assert read_json(candidate_root / "profile_subprocess.json")["returncode"] == 42
    assert read_json(root / "profile_report.json")["status"] == "no_candidate_passed"


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


def test_primary_score_and_component_bootstrap() -> None:
    better = primary_validation_summary(_validation_report(0.0))
    worse = primary_validation_summary(_validation_report(0.1))
    assert better["primary_score"] < worse["primary_score"]
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
    assert plan["protocol"]["selection"]["checkpoint_policy"] == "last_after_fixed_budget"
    assert plan["protocol"]["selection"]["checkpoint_selection_between_epochs"] is False
    root = tmp_path / "bundle"
    profile = execute_experiment(root, plan, action="profile")
    assert profile["production_started"] is False
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
    binding_report = read_json(root / "selection_report.json")
    assert comparison["test_access"] is False
    assert binding_report["checkpoint_selection_between_epochs"] is False
    assert {
        row["checkpoint_binding"]["filename"]
        for row in binding_report["configurations"]
    } == {"last.pt"}
    assert {
        row["checkpoint_binding"]["applied_optimizer_updates"]
        for row in binding_report["configurations"]
    } == {1}
    assert {row["transfer_mode"] for row in comparison["rows"]} >= {
        "frozen_probe",
        "full_finetune",
        "scratch_frozen_probe",
        "scratch_full_finetune",
    }

    damaged_path = (
        root / "cells" / "downstream" / "scratch" / "full_finetune"
        / "training_report.json"
    )
    damaged = read_json(damaged_path)
    damaged["applied_optimizer_updates"] = 0
    damaged_path.write_text(json.dumps(damaged, sort_keys=True), encoding="utf-8")
    with pytest.raises(Phase9CContractError, match="fixed_budget_binding_invalid"):
        execute_experiment(root, plan, action="aggregate")


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
