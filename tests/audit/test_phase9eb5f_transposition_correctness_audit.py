from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music_critic.experiments.analysisgnn.corrected_training import (
    build_source_free_fixture,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    B5F_AUDIT_SCHEMA,
    TranspositionDiagnosticError,
    check_compact_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5f_transposition_correctness.json"
AUDIT = ROOT / "scripts/audit_phase9eb5f_analysisgnn_transposition_correctness.py"
RUNNER = ROOT / "scripts/run_phase9eb5f_analysisgnn_shift_diagnostics.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("phase9eb5f_audit", AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_free_fixture_is_sealed_and_fail_closed() -> None:
    value = check_compact_fixture(FIXTURE)
    assert value["schema"] == B5F_AUDIT_SCHEMA
    assert value["final_status"] == "implementation_or_contract_defect"
    assert value["independent_mapping_oracle_failure_count"] == 0
    assert value["runtime_regression"]["runtime_path_matches_contract"] is True
    assert value["runtime_regression"]["round_trip_passed"] is False
    assert value["schedule"]["record_draws"] == 20_000
    assert value["checkpoint_diagnostics"]["run"] is False


def test_check_path_does_not_call_production_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _audit_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("source-free --check called a production builder")

    monkeypatch.setattr(module, "build_corpus_audit", forbidden)
    monkeypatch.setattr(module, "build_source_free_fixture", forbidden)
    value = module.check_compact_fixture(FIXTURE)
    assert value["status"]["audit_execution_valid"] is True


def test_pair_audit_executes_cross_head_relations() -> None:
    module = _audit_module()
    batch, sidecar = build_source_free_fixture()
    graph = batch.raw_graph_batch.to_data_list()[0]
    prepared = module.prepare_sidecar_diagnostic_context(sidecar)
    row = module._record_shift_row(
        record_id="dlc:source-free:fixture",
        split="train",
        graph=graph,
        sidecar=sidecar,
        shift_pc=1,
        b5a=None,
        prepared_sidecar=prepared,
    )
    assert row["cross_head_status"] == "not_checkable"
    assert "note_degree_with_pitch_key" in row["cross_head_not_checkable"]
    assert "target_vocabulary_not_closed" not in row["cross_head_not_checkable"]


def test_fixture_tampering_fails_closed(tmp_path: Path) -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["status"]["ready_for_soft_augmentation"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TranspositionDiagnosticError, match="fingerprint"):
        check_compact_fixture(path)


def test_audit_cli_check_passes_without_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--check", "--fixture", str(FIXTURE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["status"]["checkpoint_diagnostics_run"] is False
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_runner_refuses_missing_cuda_before_checkpoint_io(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--c0-checkpoint",
            str(tmp_path / "missing-c0.ckpt"),
            "--c1-checkpoint",
            str(tmp_path / "missing-c1.ckpt"),
            "--device",
            "cuda",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "analysisgnn.b5f.cuda_unavailable" in result.stderr
    assert not (tmp_path / "out").exists()


def test_checkpoint_runner_smoke_exercises_all_profiles_shifts_and_heads(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--smoke",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["valid"] is True
    assert value["smoke_checkpoint_diagnostics_run"] is True
    assert value["checkpoint_diagnostics_run"] is False
    assert set(value["profiles"]) == {"C0", "C1"}
    assert all(len(profile["per_shift"]) == 12 for profile in value["profiles"].values())
    assert all(
        row["head_count"] == 18
        for profile in value["profiles"].values()
        for row in profile["per_shift"].values()
    )
    assert (tmp_path / "smoke_checkpoint_shift_diagnostics.json").is_file()


def test_fixture_contains_all_required_status_fields() -> None:
    status = check_compact_fixture(FIXTURE)["status"]
    assert set(status) == {
        "audit_execution_valid",
        "transposition_correctness_passed",
        "runtime_path_matches_contract",
        "all_20_heads_classified",
        "identity_exact",
        "round_trip_passed",
        "cross_head_consistency_passed",
        "schedule_reproduced",
        "checkpoint_diagnostics_run",
        "shift0_metrics_reproduced",
        "test_loader_created",
        "test_targets_read",
        "test_metrics_computed",
        "ready_for_soft_augmentation",
    }
