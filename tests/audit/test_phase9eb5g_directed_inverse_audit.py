from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music_critic.experiments.analysisgnn.directed_transposition_diagnostics import (
    DirectedTranspositionDiagnosticError,
    check_directed_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5g_directed_inverse.json"
AUDIT = ROOT / "scripts/audit_phase9eb5g_directed_inverse.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("phase9eb5g_audit", AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_corpus_and_historical_forward_evidence_is_sealed() -> None:
    value = check_directed_fixture(FIXTURE)
    corpus = value["corpus_audit"]
    assert corpus["full_corpus_pair_audit"] is True
    assert corpus["record_shift_pair_count"] == 17_484
    assert corpus["shift6_eligible"] == 1_439
    assert corpus["round_trip_failure_count"] == 0
    assert corpus["target_round_trip_failure_count"] == 0
    assert corpus["canonical_forward_mismatch_count"] == 0
    assert corpus["executable_cross_head_failure_count"] == 0
    history = value["historical_schedule"]
    assert history["draw_count"] == 20_000
    assert history["all_draws_bound_to_verified_identical_forward_pair"] is True
    assert history["inverse_api_used_by_historical_training"] is False


def test_old_schedule_fingerprints_are_unchanged() -> None:
    history = check_directed_fixture(FIXTURE)["historical_schedule"]
    assert history["record_schedule_fingerprint"] == "67f4401806f2d5419bb849449aef811fd54dfbca62588c5a1543dbbe6c1b63f8"
    assert history["C0_transposition_schedule_fingerprint"] == "af937f0ece2ffc459a093b5d8a19be815c4159653b545059eee723c3bc71bb2b"
    assert history["C1_transposition_schedule_fingerprint"] == "745aef3bf213228635bbd4926a5f9d61f4dc26a425434b3757535eeccae4ef4a"


def test_check_path_does_not_rebuild_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _audit_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("--check rebuilt production corpus")

    monkeypatch.setattr(module, "build_corpus_audit", forbidden)
    assert module.check_directed_fixture(FIXTURE)["inverse_contract_valid"] is True


def test_tampering_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["inverse_contract_valid"] = False
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DirectedTranspositionDiagnosticError, match="fingerprint"):
        check_directed_fixture(path)


def test_audit_cli_check_is_source_free() -> None:
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--check", "--fixture", str(FIXTURE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["test_metrics_computed"] is False
