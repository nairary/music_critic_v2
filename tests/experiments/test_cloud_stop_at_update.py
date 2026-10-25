from hashlib import sha256
import json

import pytest

from scripts.stop_cloud_training_at_update import recent_events, trigger, validate_identity, verify_result


@pytest.mark.parametrize("event", ["training", "validation_progress", "validation"])
def test_stops_during_target_iteration(event):
    assert not trigger([dict(event="training", update=59900)], 60000)
    assert trigger([dict(event=event, update=60000)], 60000)


@pytest.mark.parametrize("target", [0, 59999, 85000, True])
def test_rejects_non_validation_boundaries(target):
    with pytest.raises(ValueError):
        trigger([], target)


def test_does_not_claim_missed_boundary():
    with pytest.raises(ValueError, match="already_passed"):
        trigger([dict(event="training", update=60100)], 60000)


def test_partial_journal_write_does_not_trigger(tmp_path):
    path = tmp_path / "events"
    path.write_text('{"event":"training","update":59900}\n{"event":"training","update":600')
    assert not trigger(recent_events(path), 60000)
    with path.open("a") as f:
        f.write('00}\n')
    assert trigger(recent_events(path), 60000)


def test_wrong_process_or_cwd_rejected():
    cmd = ["python", "-m", "music_critic.experiments.analysisgnn.temporal_slices.campaign", "--config", "owned.json"]
    validate_identity(cmd, cmd, "/owned/source", "/owned/source")
    with pytest.raises(ValueError):
        validate_identity(cmd + ["extra"], cmd, "/owned/source", "/owned/source")
    with pytest.raises(ValueError):
        validate_identity(cmd, cmd, "/other/source", "/owned/source")


def fixture_result(out):
    checkpoint = out / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    pin = dict(update=60000, path=str(checkpoint), sha256=sha256(checkpoint.read_bytes()).hexdigest(), size_bytes=checkpoint.stat().st_size)
    (out / "latest-checkpoint.json").write_text(json.dumps(pin))
    report = dict(final_update=60000, interrupted=True, validations=[dict(update=60000, primary_score=0.5)])
    (out / "report.json").write_text(json.dumps(report))
    return checkpoint, report


def test_requires_validation_and_durable_checkpoint(tmp_path):
    checkpoint, report = fixture_result(tmp_path)
    assert verify_result(tmp_path, 60000)["final_update"] == 60000
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        verify_result(tmp_path, 60000)


@pytest.mark.parametrize("change", [dict(final_update=60001), dict(interrupted=False), dict(validations=[])])
def test_wrong_terminal_state_rejected(tmp_path, change):
    _, report = fixture_result(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps(report | change))
    with pytest.raises(ValueError):
        verify_result(tmp_path, 60000)
