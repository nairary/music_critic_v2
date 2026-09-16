"""Required validation cannot disappear across exceptions or strict resume."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_resume as resume
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from tests.experiments.test_harmonic_fusion_training import fixture, _session


def _validate(model, *, split):
    assert split == "validation"
    return {"corrected_primary_macro_score": 0.5}


def _roundtrip(session, tmp_path):
    path = tmp_path / "pending.pt"
    binding = resume.save_training_checkpoint(session, path)
    return resume.load_training_checkpoint(path, expected_sha256=binding["sha256"],
        expected_size_bytes=binding["size_bytes"], config=session.config,
        bindings=session.bindings, table=session.sampler.table)


@pytest.mark.parametrize("boundary", [0, 3, 6])
def test_callback_failure_can_save_and_recover_same_pending_model(fixture, tmp_path, boundary):
    session = _session(fixture, validation_mode="required")
    full = type(session)(base=session.model.base, enabled=session.model.enabled,
                         config=session.config, bindings=session.bindings,
                         table=session.sampler.table)
    full.run_until(6, fixture[4], fixture[2], validate=_validate)
    def failing(model, *, split):
        if session.applied_updates == boundary:
            torch.rand(17)  # A failed callback cannot disturb dropout state.
            raise RuntimeError("injected_validation_failure")
        return _validate(model, split=split)
    with pytest.raises(RuntimeError, match="injected_validation_failure"):
        session.run_until(6, fixture[4], fixture[2], validate=failing)
    assert session.applied_updates == boundary
    assert session.pending_validation == boundary
    with pytest.raises(ValueError, match="validation_pending"):
        _ = session.best_update
    with pytest.raises(ValueError, match="validation_pending"):
        session.step(fixture[4], fixture[2])
    loaded = _roundtrip(session, tmp_path)
    assert loaded.pending_validation == boundary
    events = []
    loaded.run_until(6, fixture[4], fixture[2], validate=_validate, progress=events.append)
    assert events[0] == {"stage": "validation", "status": "started", "applied_updates": boundary}
    assert [row["update"] for row in loaded.validations] == [0, 3, 6]
    assert loaded.pending_validation is None and loaded.best_update == 0
    assert recursive_state_fingerprint(resume.create_training_payload(full)) == recursive_state_fingerprint(
        resume.create_training_payload(loaded))


def test_disabling_callback_between_calls_and_after_resume_fails_before_updates(fixture, tmp_path):
    session = _session(fixture, validation_mode="required")
    session.run_until(2, fixture[4], fixture[2], validate=_validate)
    loaded = _roundtrip(session, tmp_path)
    for current in (session, loaded):
        with pytest.raises(ValueError, match="validation_callback_required"):
            current.run_until(6, fixture[4], fixture[2])
        assert current.applied_updates == 2
    loaded.run_until(6, fixture[4], fixture[2], validate=_validate)
    assert [row["update"] for row in loaded.validations] == [0, 3, 6]


@pytest.mark.parametrize("missing", [0, 3])
def test_earlier_missing_validation_rejected_on_save_best_and_restricted_load(fixture, tmp_path, missing):
    session = _session(fixture, validation_mode="required")
    session.run_until(6, fixture[4], fixture[2], validate=_validate)
    payload = resume.create_training_payload(session)
    payload["validations"] = [row for row in payload["validations"] if row["update"] != missing]
    session.validations = deepcopy(payload["validations"])
    with pytest.raises(ValueError, match="validation_schedule_incomplete"):
        _ = session.best_update
    with pytest.raises(ValueError, match="validation_schedule_incomplete"):
        resume.create_training_payload(session)
    path = tmp_path / "forged.pt"
    torch.save(payload, path)
    data = path.read_bytes()
    with pytest.raises(ValueError, match="validation_schedule_incomplete"):
        resume.load_training_checkpoint(path, expected_sha256=sha256(data).hexdigest(),
            expected_size_bytes=len(data), config=session.config,
            bindings=session.bindings, table=session.sampler.table)


def test_synthetic_mode_is_explicit_and_cannot_acquire_scientific_history(fixture, tmp_path):
    session = _session(fixture, validation_mode="synthetic_disabled")
    session.run_until(3, fixture[4], fixture[2])
    assert session.pending_validation is None and session.best_update is None
    loaded = _roundtrip(session, tmp_path)
    with pytest.raises(ValueError, match="validation_disabled"):
        loaded.run_until(6, fixture[4], fixture[2], validate=_validate)
    payload = resume.create_training_payload(loaded)
    payload["validations"] = [{"update": 0, "metrics": _validate(None, split="validation")}]
    loaded.validations = payload["validations"]
    with pytest.raises(ValueError, match="validation_disabled"):
        resume.create_training_payload(loaded)


def test_pending_marker_cannot_be_removed_or_forged(fixture, tmp_path):
    session = _session(fixture, validation_mode="required")
    payload = resume.create_training_payload(session)
    assert payload["pending_validation"] == 0
    payload["pending_validation"] = None
    path = tmp_path / "forged-pending.pt"
    torch.save(payload, path)
    data = path.read_bytes()
    with pytest.raises(ValueError, match="pending_validation_mismatch"):
        resume.load_training_checkpoint(path, expected_sha256=sha256(data).hexdigest(),
            expected_size_bytes=len(data), config=session.config,
            bindings=session.bindings, table=session.sampler.table)


def test_nondivisible_final_boundary_is_required(fixture):
    session = _session(fixture, validation_mode="required")
    session.config = replace(session.config, validation_interval=4)
    events = []
    session.run_until(6, fixture[4], fixture[2], validate=_validate, progress=events.append)
    assert [row["update"] for row in session.validations] == [0, 4, 6]
    assert sum(row["stage"] == "update" and row["status"] == "completed" for row in events) == 6


@pytest.mark.parametrize("score", [True, 1, 0, "0.5", float("inf"), float("nan")])
def test_malformed_callback_score_leaves_validation_pending(fixture, score):
    session = _session(fixture, validation_mode="required")
    with pytest.raises((ValueError, TypeError)):
        session.run_until(0, fixture[4], fixture[2],
                          validate=lambda *args, **kwargs: {"corrected_primary_macro_score": score})
    assert session.pending_validation == 0 and session.validations == []
    assert resume.create_training_payload(session)["pending_validation"] == 0
