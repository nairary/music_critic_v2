from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from music_critic.experiments.phase8b2.attestation import (
    assert_data_semantic_projection_match,
)
from music_critic.experiments.phase8b2.contracts import Phase8B2ContractError
from music_critic.experiments.phase8b2.orchestrator import _ssl_runtime_binding
from music_critic.tasks import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
    create_split_manifest,
    dumps_corpus_index,
    dumps_split_manifest,
    load_corpus_index,
    project_multisource_targets,
)
from music_critic.training.data import _hook_piece, _pop_piece


def _cache_dataset(
    root: Path,
    *,
    dataset_id: str,
    pieces: tuple[object, ...],
) -> tuple[Path, CorpusCacheConfig, object]:
    cache = CorpusCacheConfig(root / f"{dataset_id}-cache")
    inputs = []
    for ordinal, piece in enumerate(pieces):
        source = f"{dataset_id}-source-{ordinal}".encode("utf-8")
        lineage_group_id = project_multisource_targets(
            piece
        ).lineage_group_id
        inputs.append(
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=lineage_group_id,
                source_identity=f"{dataset_id}-source-{ordinal}",
                source_relative_path=f"sources/{ordinal}.json",
                source_sha256=sha256(source).hexdigest(),
                suggested_split=("train", "validation", "test")[ordinal],
            )
        )
    index, _ = cache_canonical_corpus(
        inputs,
        cache_config=cache,
        dataset_id=dataset_id,
        adapter_name=f"{dataset_id}_production_path_fixture",
        adapter_version="1.0.0",
        adapter_config={"fixture": True},
        source_identity=f"{dataset_id}-production-path-fixture",
        source_fingerprint=sha256(dataset_id.encode("utf-8")).hexdigest(),
        creation_policy="phase8b2_production_path_regression",
    )
    index_path = root / f"{dataset_id}.index.json"
    index_path.write_text(dumps_corpus_index(index), encoding="utf-8")
    return index_path, cache, index


def _production_fixture(root: Path) -> tuple[list[Path], list[Path], Path]:
    hook_pieces = tuple(
        _hook_piece(f"production-hook-{ordinal}", ordinal + 1)
        for ordinal in range(3)
    )
    pop_root = root / "pop-midi"
    pop_root.mkdir(parents=True)
    pop_pieces = tuple(
        replace(
            _pop_piece(
                pop_root,
                f"{ordinal + 101:03d}",
                (60 + ordinal, 64 + ordinal, 67 + ordinal),
            ),
            source_group_id=f"pop909-production-source-{ordinal}",
        )
        for ordinal in range(3)
    )
    hook_path, hook_cache, hook_index = _cache_dataset(
        root,
        dataset_id="hooktheory",
        pieces=hook_pieces,
    )
    pop_path, pop_cache, pop_index = _cache_dataset(
        root,
        dataset_id="pop909_cl",
        pieces=pop_pieces,
    )
    assignments = {}
    for index in (hook_index, pop_index):
        for ordinal, record in enumerate(index.records):
            assignments[(record.dataset_id, record.piece_id)] = (
                "train",
                "validation",
                "test",
            )[ordinal]
    manifest = create_split_manifest(
        (hook_index, pop_index),
        assignments,
        seed=20260815,
        policy="phase8b2_production_path_regression",
    )
    manifest_path = root / "split-manifest.json"
    manifest_path.write_text(
        dumps_split_manifest(manifest), encoding="utf-8"
    )
    return (
        [hook_path, pop_path],
        [hook_cache.root, pop_cache.root],
        manifest_path,
    )


def _hydra_list(values: list[Path]) -> str:
    return "[" + ",".join(str(value.resolve()) for value in values) + "]"


def _run_production_cli(
    output_root: Path,
    *,
    index_paths: list[Path],
    cache_roots: list[Path],
    split_manifest: Path,
    action: str = "run",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase8b2.run",
            f"action={action}",
            f"output_root={output_root.resolve()}",
            "comparison=bounded_acceptance",
            "comparison.variants=[phase7a_control]",
            "comparison.seeds=[17]",
            "comparison.ssl_optimizer_steps=1",
            "comparison.downstream_optimizer_steps=1",
            "comparison.optimizer_steps_per_epoch=1",
            "comparison.bootstrap_replicates=2",
            f"data.index_paths={_hydra_list(index_paths)}",
            f"data.cache_roots={_hydra_list(cache_roots)}",
            f"data.split_manifest={split_manifest.resolve()}",
            *extra,
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_published_real_corpus_one_seed_smoke_is_bounded_and_test_locked(
    tmp_path: Path,
) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    result = _run_production_cli(
        tmp_path / "published-bounded-plan",
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        action="plan",
        extra=(
            "device.name=cuda:0",
            "device.amp=true",
            "device.amp_dtype=float16",
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["protocol"]["seeds"] == [17]
    assert plan["protocol"]["variants"] == ["phase7a_control"]
    assert plan["protocol"]["runtime_execution_config"][
        "ssl_attempted_logical_updates"
    ] == 1
    assert plan["protocol"]["runtime_execution_config"][
        "downstream_attempted_logical_updates"
    ] == 1
    assert plan["protocol"]["amp_device_config"] == {
        "amp": True,
        "amp_dtype": "float16",
        "name": "cuda:0",
        "non_blocking": False,
    }
    assert plan["claims"][
        "bounded_acceptance_is_scientific_superiority_evidence"
    ] is False
    assert plan["protocol"]["test_unlock_state"]["unlocked"] is False
    assert plan["protocol"]["test_unlock_state"][
        "test_inference_performed"
    ] is False
    assert plan["protocol"]["test_unlock_state"][
        "test_targets_accessed"
    ] is False
    assert plan["protocol"]["test_unlock_state"][
        "test_metrics_accessed"
    ] is False
    assert "selected_identities" not in plan["data_attestation"][
        "test_membership_summary"
    ]


def test_production_pilot_keeps_three_seed_minimum_gate() -> None:
    configured = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase8b2.run",
            "--cfg",
            "job",
            "comparison=production_pilot",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert "name: production_pilot" in configured.stdout
    assert "minimum_production_seeds: 3" in configured.stdout
    assert "seeds:\n  - 17\n  - 29\n  - 43" in configured.stdout

    one_seed = subprocess.run(
        [
            sys.executable,
            "-m",
            "music_critic.experiments.phase8b2.run",
            "action=plan",
            "comparison=production_pilot",
            "comparison.seeds=[17]",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert one_seed.returncode != 0
    assert "phase8b2.config.production_seed_minimum_not_met" in (
        one_seed.stdout + one_seed.stderr
    )


def test_production_format_cli_run_completes_source_neutral_attestation(
    tmp_path: Path,
) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    output_root = tmp_path / "production-mini-dag"

    result = _run_production_cli(
        output_root,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["status"] == "complete"
    report = json.loads(
        (output_root / "final_bundle" / "final_comparison_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["executed_cell_count"] == 8
    assert report["expected_cell_count"] == 8
    assert report["verified_runtime_binding_cell_count"] == 8
    assert report["checkpoint_to_evaluation_verified_cell_count"] == 3
    assert report["test_inference_performed"] is False
    assert report["test_targets_accessed"] is False
    assert report["test_metrics_accessed"] is False

    schedule = json.loads(
        (output_root / "actual_sample_schedule.json").read_text(
            encoding="utf-8"
        )
    )
    expected_projection = schedule["ssl"][0]["data_semantic_projection"]
    assert schedule["downstream"][0]["data_semantic_projection"] == (
        expected_projection
    )
    assert schedule["ssl"][0]["canonical_payloads_read_for_schedule"] is False
    assert schedule["downstream"][0][
        "canonical_payloads_read_for_schedule"
    ] is False
    assert expected_projection["train_composition"]["dataset_counts"] == [
        ["hooktheory", 1],
        ["pop909_cl", 1],
    ]
    manifests = tuple((output_root / "cells").rglob("cell_manifest.json"))
    scientific = []
    for path in manifests:
        manifest_payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = manifest_payload.get("runtime_binding_evidence", {})
        if evidence.get("verified") is True:
            scientific.append(manifest_payload)
            if path.relative_to(output_root / "cells").parts[0] in {
                "ssl",
                "downstream",
            }:
                assert evidence[
                    "data_semantic_projection_fingerprint"
                ] == expected_projection["fingerprint"]
    assert len(scientific) == 8

    serialized_plan = (output_root / "plan.json").read_text(encoding="utf-8")
    test_summary = json.loads(serialized_plan)["data_attestation"][
        "test_membership_summary"
    ]
    assert "selected_identities" not in test_summary
    assert test_summary["selected_count"] == 2
    for index_path in index_paths:
        test_piece_id = load_corpus_index(index_path).records[2].piece_id
        assert test_piece_id not in serialized_plan

    immutable = tuple(
        sorted((output_root / "final_bundle").glob("*"))
    )
    before = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in immutable
    }
    rerun = _run_production_cli(
        output_root,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
    )
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert json.loads(rerun.stdout)["resumed"] is True
    assert {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in immutable
    } == before

    plan = json.loads((output_root / "plan.json").read_text(encoding="utf-8"))
    ssl_cell = plan["ssl_cells"][0]
    published_ssl = output_root / "cells" / Path(*ssl_cell["cell_id"].split("/"))
    tampered_ssl = tmp_path / "tampered-ssl-runtime"
    shutil.copytree(published_ssl, tampered_ssl)
    report_path = tampered_ssl / "engine" / "training_report.json"
    runtime_report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_report["observed_ssl_sample_schedule_fingerprint"] = "0" * 64
    report_path.write_text(json.dumps(runtime_report), encoding="utf-8")
    with pytest.raises(
        Phase8B2ContractError,
        match="phase8b2.runner.ssl_sample_schedule_mismatch",
    ):
        _ssl_runtime_binding(plan, ssl_cell, tampered_ssl)

    malformed_ssl = tmp_path / "malformed-ssl-runtime"
    shutil.copytree(published_ssl, malformed_ssl)
    report_path = malformed_ssl / "engine" / "training_report.json"
    runtime_report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_report.pop("accounting")
    report_path.write_text(json.dumps(runtime_report), encoding="utf-8")
    with pytest.raises(
        Phase8B2ContractError,
        match="phase8b2.runner.ssl_runtime_evidence_malformed",
    ):
        _ssl_runtime_binding(plan, ssl_cell, malformed_ssl)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 8B.2A production-path acceptance requires a CUDA runner",
)
def test_production_format_cli_runs_cuda_amp_with_integer_vram_boundary(
    tmp_path: Path,
) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    output_root = tmp_path / "production-cuda-mini-dag"

    result = _run_production_cli(
        output_root,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        extra=(
            "device.name=cuda:0",
            "device.amp=true",
            "device.amp_dtype=float16",
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "complete"
    report = json.loads(
        (output_root / "final_bundle" / "final_comparison_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["executed_cell_count"] == 8
    assert report["expected_cell_count"] == 8
    assert report["verified_runtime_binding_cell_count"] == 8
    assert report["checkpoint_to_evaluation_verified_cell_count"] == 3
    assert report["test_inference_performed"] is False
    assert report["test_targets_accessed"] is False
    assert report["test_metrics_accessed"] is False

    plan = json.loads(
        (output_root / "plan.json").read_text(encoding="utf-8")
    )
    assert plan["protocol"]["seeds"] == [17]
    assert [row["variant_id"] for row in plan["ssl_cells"]] == [
        "phase7a_control"
    ]
    assert plan["ssl_cells"][0]["schedule"]["logical_updates"] == 1
    assert {
        row["transfer_mode"] for row in plan["downstream_cells"]
    } == {"frozen_probe", "full_finetune", "supervised_scratch"}
    assert len(plan["downstream_cells"]) == 3
    assert len(plan["evaluation_cells"]) == 3

    compute = json.loads(
        (output_root / "final_bundle" / "compute_accounting.json")
        .read_text(encoding="utf-8")
    )
    assert len(compute["cells"]) == 1
    cuda_peak = compute["cells"][0]["cuda_peak_memory"]
    assert cuda_peak["available"] is True
    assert cuda_peak["cuda_logical_device_index"] == 0
    assert cuda_peak["peak_allocated_bytes"] > 0
    assert cuda_peak["peak_reserved_bytes"] > 0

    run_manifest = json.loads(
        (output_root / "final_bundle" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["environment"]["device"] == "cuda:0"
    assert run_manifest["environment"]["cuda_logical_device_index"] == 0


@pytest.mark.parametrize(
    ("field", "category"),
    (
        ("dataset_indices", "ssl_index_identity_mismatch"),
        ("cache_identities", "ssl_cache_identity_mismatch"),
        ("split_manifest_fingerprint", "ssl_split_manifest_mismatch"),
        ("train_composition", "ssl_train_composition_mismatch"),
        ("validation_membership", "ssl_validation_membership_mismatch"),
    ),
)
def test_semantic_projection_mismatches_have_stable_categories(
    field: str,
    category: str,
) -> None:
    projection = {
        "data_semantic_projection_contract_version": "1.0.0",
        "dataset_indices": [["hooktheory", "a" * 64]],
        "cache_identities": [["hooktheory", "b" * 64]],
        "split_manifest_fingerprint": "c" * 64,
        "train_composition": {
            "dataset_counts": [["hooktheory", 1]],
            "piece_count": 1,
            "semantic_fingerprint": "d" * 64,
        },
        "validation_membership": {
            "membership_fingerprint": "e" * 64,
            "dataset_counts": [["hooktheory", 1]],
            "full_view_count": 1,
            "selected_count": 1,
            "subset_limit": 0,
            "semantic_fingerprint": "f" * 64,
        },
        "mixture_weights": [["hooktheory", 1.0]],
        "fingerprint": "0" * 64,
    }
    changed = deepcopy(projection)
    if isinstance(changed[field], str):
        changed[field] = "1" * 64
    elif isinstance(changed[field], list):
        changed[field][0] = ["changed", "1" * 64]
    else:
        changed[field]["semantic_fingerprint"] = "1" * 64
    with pytest.raises(Phase8B2ContractError, match=category):
        assert_data_semantic_projection_match(
            projection, changed, stage="ssl"
        )


def _assert_plan_failure(
    root: Path,
    *,
    index_paths: list[Path],
    cache_roots: list[Path],
    split_manifest: Path,
    category: str,
    extra: tuple[str, ...] = (),
) -> None:
    result = _run_production_cli(
        root / "unused-output",
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=split_manifest,
        action="plan",
        extra=extra,
    )
    assert result.returncode != 0
    assert category in result.stdout + result.stderr


def test_changed_on_disk_index_fingerprint_fails_closed(
    tmp_path: Path,
) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    payload = json.loads(index_paths[0].read_text(encoding="utf-8"))
    payload["header"]["index_fingerprint"] = "0" * 64
    index_paths[0].write_text(json.dumps(payload), encoding="utf-8")
    _assert_plan_failure(
        tmp_path,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        category="phase8b2.data_attestation.index_metadata_invalid",
    )


def test_wrong_cache_root_fails_closed(tmp_path: Path) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    _assert_plan_failure(
        tmp_path,
        index_paths=index_paths,
        cache_roots=list(reversed(cache_roots)),
        split_manifest=manifest,
        category="phase8b2.data_attestation.index_cache_path_mismatch",
    )


def test_changed_cache_artifact_sha_fails_closed(tmp_path: Path) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    index = load_corpus_index(index_paths[0])
    artifact = (
        cache_roots[0]
        / CorpusCacheConfig(cache_roots[0]).namespace
        / index.records[0].canonical_relative_path
    )
    artifact.write_bytes(artifact.read_bytes() + b"stale")
    _assert_plan_failure(
        tmp_path,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        category="phase8b2.data_attestation.index_cache_path_mismatch",
    )


def test_changed_split_manifest_fails_closed(tmp_path: Path) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["manifest_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _assert_plan_failure(
        tmp_path,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        category="phase8b2.data_attestation.split_manifest_invalid",
    )


def test_changed_validation_membership_expectation_fails_closed(
    tmp_path: Path,
) -> None:
    index_paths, cache_roots, manifest = _production_fixture(tmp_path)
    _assert_plan_failure(
        tmp_path,
        index_paths=index_paths,
        cache_roots=cache_roots,
        split_manifest=manifest,
        extra=("data.validation_membership_fingerprint=" + "0" * 64,),
        category=(
            "phase8b2.data_attestation."
            "validation_membership_fingerprint_mismatch"
        ),
    )
