"""Synthetic GPU-guard dispatch only; no CUDA, corpus, model or worker execution."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from music_critic.experiments.analysisgnn import performance_prefetch_replay as replay


def _run(root, **kwargs):
    return replay.run_prefetch_replay(root, experiment_id="SYNTHETIC-DESKTOP-BOUNDARY",
        mode="fixture", device="cuda", measured_draws=2, warmup_draws=0,
        repeats=1, worker_counts=(0,), **kwargs)


def _forbidden(*args, **kwargs):
    pytest.fail("guard dispatch reached forbidden CUDA/model/source work")


@pytest.mark.parametrize("invalid", [False, 0, "approved", {}, []])
def test_noncallable_guard_fails_before_output_schedule_source_or_model(tmp_path, monkeypatch, invalid):
    for name in ("build_prefetch_schedule", "load_production_prefetch_inputs", "_components"):
        monkeypatch.setattr(replay, name, _forbidden)
    root = tmp_path / "unused"
    with pytest.raises(ValueError, match="gpu_idle_check_must_be_callable"):
        _run(root, gpu_idle_check=invalid)
    assert not root.exists()


def test_default_uses_unchanged_guard_and_rejects_nautilus(tmp_path, monkeypatch):
    queries = []
    def query(argv, **kwargs):
        queries.append(argv)
        return SimpleNamespace(stdout="299037, /usr/bin/nautilus\n")
    monkeypatch.setattr(replay.existing.subprocess, "run", query)
    monkeypatch.setattr(replay.torch.cuda, "is_available", _forbidden)
    monkeypatch.setattr(replay, "_components", _forbidden)
    with pytest.raises(RuntimeError, match="performance.other_gpu_compute_process_active"):
        _run(tmp_path / "default")
    assert queries == [["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"]]
    failure = json.loads((tmp_path / "default/failure.json").read_text())
    assert failure["completed"] is False
    assert not (tmp_path / "default/summary.json").exists()


def test_explicit_callback_replaces_first_guard_and_preserves_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.existing, "_assert_idle_gpu", _forbidden)
    monkeypatch.setattr(replay.torch.cuda, "is_available", _forbidden)
    error = RuntimeError("synthetic desktop policy rejection")
    calls = []
    def reject():
        calls.append("callback")
        raise error
    with pytest.raises(RuntimeError) as caught:
        _run(tmp_path / "rejected", gpu_idle_check=reject)
    assert caught.value is error and calls == ["callback"]
    failure = json.loads((tmp_path / "rejected/failure.json").read_text())
    assert failure == {"completed": False, "error": str(error), "error_type": "RuntimeError"}
    assert not (tmp_path / "rejected/summary.json").exists()


def test_same_falsey_callable_is_used_before_runtime_and_before_arm(tmp_path, monkeypatch):
    """Stop at the second guard: model/runtime stand-ins never run CUDA."""
    events = []
    error = RuntimeError("synthetic arm guard rejection")
    class Guard:
        def __bool__(self):
            return False
        def __call__(self):
            events.append("guard")
            if events.count("guard") == 2:
                raise error
    monkeypatch.setattr(replay.existing, "_assert_idle_gpu", _forbidden)
    monkeypatch.setattr(replay.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(replay.existing, "_preserve_runtime", lambda *_: nullcontext())
    monkeypatch.setattr(replay, "FixturePreparer", lambda: object())
    monkeypatch.setattr(replay, "_prewarm", lambda *_: events.append("prewarm"))
    def components(*_):
        events.append("components")
        return (None,) * 6
    monkeypatch.setattr(replay, "_components", components)
    monkeypatch.setattr(replay.existing, "_snapshot", lambda *_: {})
    monkeypatch.setattr(replay, "BatchStream", _forbidden)
    with pytest.raises(RuntimeError) as caught:
        _run(tmp_path / "second", gpu_idle_check=Guard())
    assert caught.value is error
    assert events == ["guard", "prewarm", "components", "guard"]
    assert not (tmp_path / "second/r00-workers0-b2").exists()
    assert json.loads((tmp_path / "second/failure.json").read_text())["completed"] is False
