from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_phase9eb5b_analysisgnn_training_policy as audit_script
from music_critic.experiments.analysisgnn.contracts import fingerprint


def test_source_free_fixture_is_canonical_deterministic_and_valid() -> None:
    first = audit_script.check_fixture()
    second = audit_script.check_fixture()
    assert first == second
    assert first["valid"] is True
    assert first["ready_for_model_implementation"] is True
    assert first["training_run"] is False
    assert first["validation_inference_run"] is False
    assert first["test_evaluated"] is False
    assert first["test_targets_used_for_evaluation"] is False


def test_source_free_check_never_calls_production_builder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("source-free check must not open B3/B4 production outputs")

    monkeypatch.setattr(audit_script, "build_fixture", forbidden)
    assert audit_script.main(["--check"]) == 0
    output = capsys.readouterr().out
    assert '"valid": true' in output
    assert '"training_run": false' in output
    assert '"test_targets_used_for_evaluation": false' in output


def test_fixture_self_fingerprints_reproduce_without_manual_values() -> None:
    fixture = audit_script.check_fixture()
    fixture_fingerprint = fixture.pop("fixture_fingerprint")
    assert fingerprint(fixture) == fixture_fingerprint
    semantic_fingerprint = fixture.pop("audit_semantic_fingerprint")
    assert fingerprint(fixture) == semantic_fingerprint


def test_all_required_contract_and_profile_fingerprints_are_present() -> None:
    fixture = audit_script.check_fixture()
    fingerprints = fixture["fingerprints"]
    assert set(fingerprints) == {
        "head_roles",
        "loss",
        "class_weights",
        "sampler",
        "metrics",
        "profile_O",
        "profile_C0",
        "profile_C1",
        "combined",
    }
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in fingerprints.values()
    )
    assert len(fixture["audit_semantic_fingerprint"]) == 64
    assert len(fixture["fixture_fingerprint"]) == 64


def test_fixture_freezes_profiles_roles_loss_and_test_lock() -> None:
    fixture = audit_script.check_fixture()
    assert fixture["profile_ids"] == {
        "O": "analysisgnn-official-reproduction-e115182-v1",
        "C0": "music-critic-v2-corrected-no-transposition-v1",
        "C1": "music-critic-v2-corrected-safe-transposition-v1",
    }
    assert fixture["contracts"]["head_roles"]["role_counts"] == {
        "primary": 8,
        "auxiliary": 10,
        "deferred": 2,
    }
    assert fixture["contracts"]["loss"]["group_weights"] == {
        "primary": 1.0,
        "auxiliary": 0.25,
        "deferred": 0.0,
    }
    assert fixture["corrected_profile_comparison"][
        "only_transposition_differs"
    ] is True
    assert fixture["component_sampling_evidence"]["test_draw_count"] == 0


def test_full_class_weight_payload_contains_quality17_and_roman184() -> None:
    fixture = audit_script.check_fixture()
    payload = fixture["class_weight_payload"]
    heads = {row["task_id"]: row for row in payload["heads"]}
    assert len(heads) == 20
    assert len(heads["quality"]["classes"]) == 17
    assert len(heads["roman_numeral"]["classes"]) == 184
    augmented_sixth = next(
        row
        for row in heads["quality"]["classes"]
        if row["class_value"] == "augmented sixth"
    )
    assert augmented_sixth["weight"] is None
    assert augmented_sixth["support_status"] == "unsupported"


def test_fixture_is_committed_at_the_source_free_path() -> None:
    assert audit_script.DEFAULT_FIXTURE == (
        Path(__file__).parents[1]
        / "fixtures/analysisgnn/phase9eb5b_training_policy.json"
    )
    assert audit_script.DEFAULT_FIXTURE.is_file()
