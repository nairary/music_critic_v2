from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
)
from music_critic.experiments.analysisgnn.full_orbit_training import (
    FULL_ORBIT_PROFILE_ID,
    check_full_orbit_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5h_full_orbit_profile.json"
AUDIT = ROOT / "scripts/audit_phase9eb5h_full_orbit_profile.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("phase9eb5h_audit", AUDIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_fixture_seals_ready_but_untrained_profile() -> None:
    value = check_full_orbit_fixture(FIXTURE)
    assert value["profile"]["profile_id"] == FULL_ORBIT_PROFILE_ID
    assert value["inverse_contract_valid"] is True
    assert value["full_orbit_profile_valid"] is True
    assert value["ready_for_full_orbit_training"] is True
    assert value["full_orbit_training_run"] is False
    assert value["test_loader_created"] is False
    assert value["test_targets_read"] is False
    assert value["test_metrics_computed"] is False


def test_split_profiles_and_model_contract_are_unchanged() -> None:
    value = check_full_orbit_fixture(FIXTURE)
    assert value["c0_profile_id"] == "music-critic-v2-corrected-no-transposition-v1"
    assert value["c1_profile_id"] == "music-critic-v2-corrected-safe-transposition-v1"
    assert value["c0_c1_profile_fingerprints"] == {
        "music-critic-v2-corrected-no-transposition-v1": "b811b9b422ee12c7d1c723bca2c97bd88fdf3dbb8da4babd23d89abea6b75333",
        "music-critic-v2-corrected-safe-transposition-v1": "21933d56970ebf30368d91d53feb2853465675249b745cd0a3147129e1cc167d",
    }
    assert value["orbit_table"]["source_split_changed"] is False
    torch_seed = 17
    import torch
    torch.manual_seed(torch_seed)
    model = CorrectedAnalysisGNNModel()
    assert value["initial_model_state_fingerprint"] == model_state_fingerprint(model)
    assert value["model_contract_fingerprint"] == corrected_model_contract(model)["fingerprint"]


def test_check_path_does_not_build_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _audit_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("--check called production preflight")

    monkeypatch.setattr(module, "full_orbit_preflight", forbidden)
    assert module.check_full_orbit_fixture(FIXTURE)["full_orbit_profile_valid"] is True


def test_tampering_fails_closed(tmp_path: Path) -> None:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["ready_for_full_orbit_training"] = False
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CorrectedTrainingError, match="fixture_fingerprint_mismatch"):
        check_full_orbit_fixture(path)


def test_audit_cli_check_is_source_free_and_reproducible() -> None:
    command = [sys.executable, str(AUDIT), "--check", "--fixture", str(FIXTURE)]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)
