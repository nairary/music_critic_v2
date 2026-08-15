from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def _run_cli(
    output_root: Path,
    *,
    action: str,
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
            "comparison.bootstrap_replicates=20",
            *extra,
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bounded_cli_matrix_is_resumable_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matrix"
    planned = _run_cli(output_root, action="plan")
    assert planned.returncode == 0, planned.stderr
    planned_result = json.loads(planned.stdout)
    assert planned_result["summary"]["ssl_cell_count"] == 8
    assert planned_result["summary"]["downstream_cell_count"] == 18
    assert planned_result["summary"]["evaluation_cell_count"] == 18
    assert planned_result["summary"][
        "ssl_encoder_forward_budget_per_cell"
    ] == 24
    assert not output_root.exists()

    interrupted = _run_cli(
        output_root,
        action="run",
        extra=("fail_after_cell=1",),
    )
    assert interrupted.returncode != 0
    assert "injected_interruption_after_published_cell" in (
        interrupted.stdout + interrupted.stderr
    )
    assert len(list((output_root / "cells" / "preflight").rglob(
        "cell_manifest.json"
    ))) == 8
    assert len(list((output_root / "cells" / "ssl").rglob(
        "cell_manifest.json"
    ))) == 1

    resumed = _run_cli(output_root, action="resume")
    assert resumed.returncode == 0, resumed.stderr
    resumed_result = json.loads(resumed.stdout)
    assert resumed_result["status"] == "complete"
    assert resumed_result["resumed"] is False

    final = output_root / "final_bundle"
    report = _json(final / "final_comparison_report.json")
    selection = _json(final / "validation_selection.json")
    schedule = _json(final / "actual_sample_schedule.json")
    run_manifest = _json(final / "run_manifest.json")
    assert report["executed_cell_count"] == 52
    assert report["expected_cell_count"] == 52
    assert report["preflight_cell_count"] == 8
    assert report["verified_runtime_binding_cell_count"] == 52
    assert report["checkpoint_to_evaluation_verified_cell_count"] == 18
    assert report["ssl_compute_totals"] == {
        "attempted_updates": 16,
        "applied_updates": 16,
        "skipped_updates": 0,
        "raw_samples": 32,
        "encoder_forwards": 192,
    }
    assert report["test_membership_metadata_resolved"] is True
    assert report["test_inference_performed"] is False
    assert report["test_targets_accessed"] is False
    assert report["test_metrics_accessed"] is False
    assert report["validation_only_selection"] is True
    assert report["pdmx_evidence"] is False
    assert selection["selected_count"] == 2
    assert selection["complete_paired_seed_evidence"] is True
    assert selection["declared_seeds"] == [17, 29]
    assert len(selection["selected_checkpoints"]) == 2
    assert run_manifest["artifact_contract_version"] == "1.2.1"
    assert len(run_manifest["cells"]) == 60

    aggregated = _run_cli(output_root, action="aggregate")
    assert aggregated.returncode == 0, aggregated.stderr
    assert json.loads(aggregated.stdout)["status"] == "aggregated"
    selected = _run_cli(output_root, action="select")
    assert selected.returncode == 0, selected.stderr
    selected_result = json.loads(selected.stdout)
    assert selected_result["status"] == "selected"
    assert selected_result["selection"]["fingerprint"] == selection[
        "fingerprint"
    ]

    expected_by_seed = {
        row["seed"]: [
            (slot["dataset_id"], slot["piece_id"])
            for slot in row["slots"]
        ]
        for row in schedule["ssl"]
    }
    observed_by_seed: dict[int, set[tuple[tuple[str, str], ...]]] = {}
    for path in sorted((output_root / "cells" / "ssl").rglob(
        "training_report.json"
    )):
        cell_report = _json(path)
        seed = int(path.relative_to(output_root / "cells" / "ssl").parts[1])
        identities = tuple(
            tuple(row) for row in cell_report["observed_sample_identities"]
        )
        observed_by_seed.setdefault(seed, set()).add(identities)
        assert cell_report["actual_sample_schedule_verified"] is True
        assert cell_report["accounting"][
            "optimizer_step_applied_count"
        ] == 2
        assert cell_report["accounting"][
            "optimizer_step_skipped_count"
        ] == 0
        assert cell_report["accounting"]["encoder_forward_count"] == 24
    assert set(observed_by_seed) == {17, 29}
    assert all(len(values) == 1 for values in observed_by_seed.values())
    assert {
        seed: list(next(iter(values)))
        for seed, values in observed_by_seed.items()
    } == expected_by_seed

    evaluation_manifests = list(
        (output_root / "cells" / "evaluation").rglob(
            "cell_manifest.json"
        )
    )
    assert len(evaluation_manifests) == 18
    for path in evaluation_manifests:
        evidence = _json(path)["runtime_binding_evidence"]
        assert evidence["checkpoint_to_evaluation_data_verified"] is True
        assert evidence["membership_mismatch"] is False

    immutable_files = (
        final / "run_manifest.json",
        final / "statistical_summary.json",
        final / "validation_selection.json",
        final / "final_comparison_report.json",
    )
    before = {path.name: _sha256(path) for path in immutable_files}
    rerun = _run_cli(
        output_root,
        action="run",
        extra=(
            "comparison.variants=[multilevel_equal,onset_latent,"
            "phase8a_mask_only,phase7a_control]",
            "comparison.seeds=[29,17]",
        ),
    )
    assert rerun.returncode == 0, rerun.stderr
    rerun_result = json.loads(rerun.stdout)
    assert rerun_result["status"] == "complete"
    assert rerun_result["resumed"] is True
    assert {path.name: _sha256(path) for path in immutable_files} == before

    final_report_path = final / "final_comparison_report.json"
    original_final_report = final_report_path.read_bytes()
    final_report_path.write_bytes(original_final_report + b"\n")
    stale_final = _run_cli(output_root, action="run")
    assert stale_final.returncode != 0
    assert "final_bundle_stale_or_incomplete" in (
        stale_final.stdout + stale_final.stderr
    )
    final_report_path.write_bytes(original_final_report)

    first_preflight = sorted(
        (output_root / "cells" / "preflight").rglob("stdout.log")
    )[0]
    first_preflight.write_text(
        first_preflight.read_text(encoding="utf-8") + "stale\n",
        encoding="utf-8",
    )
    stale = _run_cli(output_root, action="resume")
    assert stale.returncode != 0
    assert "stale_or_incomplete_cell" in stale.stdout + stale.stderr


def test_cli_rejects_incomplete_cell_staging_without_overwrite(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "incomplete"
    staging = (
        output_root
        / ".staging"
        / "preflight__encoder_forward_matched__17__multilevel_equal"
    )
    staging.mkdir(parents=True)
    sentinel = staging / "operator-note.txt"
    sentinel.write_text("inspect me", encoding="utf-8")

    result = _run_cli(output_root, action="run")

    assert result.returncode != 0
    assert "incomplete_staging_requires_inspection" in (
        result.stdout + result.stderr
    )
    assert sentinel.read_text(encoding="utf-8") == "inspect me"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 8B.2A CUDA matrix acceptance requires a CUDA runner",
)
def test_bounded_cli_matrix_runs_on_explicit_cuda_amp_with_vram_evidence(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "cuda-matrix"
    result = _run_cli(
        output_root,
        action="run",
        extra=(
            "device.name=cuda:0",
            "device.amp=true",
            "device.amp_dtype=float16",
        ),
    )
    assert result.returncode == 0, result.stderr
    compute = _json(
        output_root / "final_bundle" / "compute_accounting.json"
    )
    assert all(
        row["cuda_peak_memory"]["available"] is True
        and row["cuda_peak_memory"]["peak_allocated_bytes"] > 0
        for row in compute["cells"]
    )
    manifest = _json(output_root / "final_bundle" / "run_manifest.json")
    assert manifest["environment"]["device"] == "cuda:0"
