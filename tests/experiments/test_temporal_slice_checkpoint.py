from copy import deepcopy
from hashlib import sha256
import random

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import corrected_training as old
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.training_rng import capture_safe_rng
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint, _scaler
from music_critic.experiments.analysisgnn.temporal_slices import TemporalSliceModel, build_temporal_batch
from music_critic.experiments.analysisgnn.temporal_slices.checkpoint import (
    TemporalCheckpointError, create_temporal_payload, load_temporal_checkpoint,
    save_temporal_checkpoint, temporal_model_contract,
)
from music_critic.experiments.analysisgnn.temporal_slices.training import attempt_temporal_update


CONFIG = {"batch_size": 2, "total_updates": 80000, "scheduler_total_updates": 120000, "seed": 17}
BINDINGS = {"data_fingerprint": "fixture-only", "source_fingerprint": "fixture-source-v1",
            "target_registry": "frozen-current", "environment_admission": "source-free-cpu"}
TABLE = (FullOrbitPair("fixture-a", 0, "component-a"), FullOrbitPair("fixture-b", 2, "component-b"))


@pytest.fixture
def fixture(monkeypatch):
    import music_critic.tasks as tasks
    captured = []
    original = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        captured.append(piece)
        return original(piece, *args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_multisource_sample", capture)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    _, sidecar = old.build_source_free_fixture()
    graph = build_temporal_batch([captured[0], captured[0]])
    yield graph, [sidecar, sidecar], old.load_frozen_class_weights()
    torch.set_num_threads(threads)


def components():
    model = TemporalSliceModel()
    optimizer, scheduler = old.build_optimizer_scheduler(model, total_updates=120000)
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler,
            "scaler": _scaler()[0]}


def progress(update=0):
    return {"applied_updates": update, "draw_cursor": 2 * update, "validation_history": [],
            "best_score": None, "best_update": None, "pending_validation": None, "elapsed_seconds": 1.5}


def advance(trainer, fixture, sampler):
    graph, sidecars, weights = fixture
    result = attempt_temporal_update(**trainer, graph=graph, sidecars=sidecars, shifts=[0, 0], class_weights=weights)
    assert result.applied
    sampler.advance_after_applied_update(2)
    return result


def save(path, trainer, sampler, update=0, **kw):
    return save_temporal_checkpoint(path, **trainer, config=CONFIG, bindings=BINDINGS,
                                    progress=kw.pop("progress", progress(update)), sampler_state=sampler.state_dict(), **kw)


def load(path, evidence, trainer, sampler=None, **kw):
    return load_temporal_checkpoint(path, **trainer, config=kw.pop("config", CONFIG),
                                    bindings=kw.pop("bindings", BINDINGS), sampler=sampler,
                                    expected_sha256=evidence["sha256"], expected_size_bytes=evidence["size_bytes"], **kw)


def write_payload(path, payload):
    torch.save(payload, path)
    content = path.read_bytes()
    return {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}


def reseal(payload):
    payload["state_fingerprint"] = recursive_state_fingerprint({k: v for k, v in payload.items() if k != "state_fingerprint"})


def state(trainer):
    return recursive_state_fingerprint({k: v.state_dict() for k, v in trainer.items()})


def test_exact_next_update_rng_sampler_and_all_states(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    advance(trainer, fixture, sampler)
    # Preserve intentionally mixed module modes as well as root mode.
    trainer["model"].temporal_layers[0].dropout.eval()
    before_rng = recursive_state_fingerprint(capture_safe_rng())
    path = tmp_path / "update-1.ckpt"
    evidence = save(path, trainer, sampler, 1)
    assert recursive_state_fingerprint(capture_safe_rng()) == before_rng
    assert torch.load(path, weights_only=True)["progress"]["applied_updates"] == 1
    expected_draws = [sampler.peek(0), sampler.peek(1)]
    expected_random = (random.random(), np.random.rand(), torch.rand(4))
    expected = advance(trainer, fixture, sampler)
    final_state = state(trainer)
    replacement, resumed_sampler = components(), FullOrbitSampler(TABLE)
    restored = load(path, evidence, replacement, resumed_sampler)
    assert restored["progress"] == progress(1)
    assert [resumed_sampler.peek(0), resumed_sampler.peek(1)] == expected_draws
    assert not replacement["model"].temporal_layers[0].dropout.training
    actual_random = (random.random(), np.random.rand(), torch.rand(4))
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
    repeated = advance(replacement, fixture, resumed_sampler)
    assert expected.loss.total.item() == repeated.loss.total.item()
    assert state(replacement) == final_state
    assert resumed_sampler.state_dict() == sampler.state_dict()


def test_immutable_publication_and_no_temporary_leaks(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    path = tmp_path / "initial.ckpt"
    evidence = save(path, trainer, sampler)
    with pytest.raises(FileExistsError):
        save(path, trainer, sampler)
    assert sha256(path.read_bytes()).hexdigest() == evidence["sha256"]
    assert list(tmp_path.iterdir()) == [path]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change,match", [("config", "config_binding"), ("bindings", "data_source_environment_binding"),
                                          ("runtime", "runtime_binding"), ("extra_field", "schema_required"),
                                          ("model", "model_contract"), ("old_schema", "schema_required")])
def test_fail_closed_bindings_leave_caller_untouched(tmp_path, fixture, change, match):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    path = tmp_path / "state.ckpt"
    evidence = save(path, trainer, sampler)
    payload = torch.load(path, weights_only=True)
    if change in ("config", "bindings"):
        kwargs = {change: {**(CONFIG if change == "config" else BINDINGS), "unexpected": True}}
    else:
        kwargs = {}
        if change == "runtime":
            payload["runtime"]["threads"] += 1
        elif change == "extra_field":
            payload["other"] = "no"
        elif change == "model":
            payload["model_contract"]["graph_schema"] = "coarse-beat-model"
        else:
            payload["schema_version"] = "GroupedMMoEAttentionCheckpoint@1.0.0"
        reseal(payload)
        path = tmp_path / "invalid.ckpt"
        evidence = write_payload(path, payload)
    replacement = components()
    before, rng = state(replacement), recursive_state_fingerprint(capture_safe_rng())
    with pytest.raises(TemporalCheckpointError, match=match):
        load(path, evidence, replacement, sampler, **kwargs)
    assert state(replacement) == before
    assert recursive_state_fingerprint(capture_safe_rng()) == rng
    assert sampler.position == 0


@pytest.mark.parametrize("bad", ["cursor", "sampler_fingerprint", "sampler_binding", "scheduler", "optimizer", "rng"])
def test_reject_inconsistent_resume_components(tmp_path, fixture, bad):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    advance(trainer, fixture, sampler)
    path = tmp_path / "state.ckpt"
    save(path, trainer, sampler, 1)
    payload = torch.load(path, weights_only=True)
    if bad == "cursor":
        payload["progress"]["draw_cursor"] = 4  # Prefetched reservation is not committed.
    elif bad == "sampler_fingerprint":
        payload["sampler_state"]["fingerprint"] = "0" * 64
    elif bad == "sampler_binding":
        payload["sampler_state"]["table_fingerprint"] = "0" * 64
        payload["sampler_state"]["fingerprint"] = fingerprint({k: v for k, v in payload["sampler_state"].items() if k != "fingerprint"})
    elif bad == "scheduler":
        payload["scheduler_state"]["last_epoch"] += 1
    elif bad == "optimizer":
        first = next(iter(payload["optimizer_state"]["state"].values()))
        first["exp_avg_sq"].fill_(-1)
    else:
        payload["rng_state"]["torch_cpu"] = torch.zeros(2, dtype=torch.uint8)
    reseal(payload)
    badpath = tmp_path / "invalid.ckpt"
    evidence = write_payload(badpath, payload)
    replacement, replacement_sampler = components(), FullOrbitSampler(TABLE)
    before, rng = state(replacement), recursive_state_fingerprint(capture_safe_rng())
    with pytest.raises(ValueError):
        load(badpath, evidence, replacement, replacement_sampler)
    assert state(replacement) == before
    assert recursive_state_fingerprint(capture_safe_rng()) == rng
    assert replacement_sampler.position == 0


def test_external_byte_hash_and_internal_state_seal(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    path = tmp_path / "state.ckpt"
    evidence = save(path, trainer, sampler)
    with pytest.raises(ValueError, match="sha256"):
        load(path, {**evidence, "sha256": "0" * 64}, components())
    payload = torch.load(path, weights_only=True)
    payload["progress"]["elapsed_seconds"] = 99.
    path = tmp_path / "unsealed.ckpt"
    evidence = write_payload(path, payload)
    with pytest.raises(TemporalCheckpointError, match="state_fingerprint"):
        load(path, evidence, components())


def test_pending_validation_and_inspection_without_rng_restore(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    advance(trainer, fixture, sampler)
    p = progress(1)
    p["pending_validation"] = 1
    p["validation_history"] = [{"update": 0, "metrics": {"corrected_primary_macro_score": .1}}]
    p["best_update"], p["best_score"] = 0, .1
    path = tmp_path / "pending.ckpt"
    evidence = save(path, trainer, sampler, 1, progress=p)
    replacement = components()
    rng = recursive_state_fingerprint(capture_safe_rng())
    result = load(path, evidence, replacement, restore_rng=False)
    assert result["progress"] == p
    assert recursive_state_fingerprint(capture_safe_rng()) == rng


def test_restore_failure_rolls_back_every_component(tmp_path, fixture, monkeypatch):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    advance(trainer, fixture, sampler)
    path = tmp_path / "state.ckpt"
    evidence = save(path, trainer, sampler, 1)
    replacement, replacement_sampler = components(), FullOrbitSampler(TABLE)
    original = replacement["scaler"].load_state_dict
    calls = []
    def fail_once(value):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("synthetic restoration failure")
        return original(value)
    monkeypatch.setattr(replacement["scaler"], "load_state_dict", fail_once)
    before, rng = state(replacement), recursive_state_fingerprint(capture_safe_rng())
    with pytest.raises(RuntimeError, match="synthetic restoration"):
        load(path, evidence, replacement, replacement_sampler)
    assert state(replacement) == before
    assert recursive_state_fingerprint(capture_safe_rng()) == rng
    assert replacement_sampler.position == 0


def test_model_only_or_c2_model_cannot_be_saved(fixture):
    with pytest.raises(TemporalCheckpointError, match="temporal_model_required"):
        temporal_model_contract(CorrectedAnalysisGNNModel())
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    payload = create_temporal_payload(**trainer, config=CONFIG, bindings=BINDINGS,
                                     progress=progress(), sampler_state=sampler.state_dict())
    assert payload["model_state_fingerprint"] == model_state_fingerprint(trainer["model"])
    assert "cache" not in payload and "worker_queue" not in payload


@pytest.mark.skipif(not hasattr(torch.amp, "GradScaler"),
                   reason="PyTorch 2.2 has no CPU GradScaler; native temporal campaign uses disabled FP32 scaler")
def test_enabled_amp_scaler_growth_resumes_exactly(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    trainer["scaler"] = torch.amp.GradScaler("cpu", enabled=True, init_scale=2., growth_interval=2)
    advance(trainer, fixture, sampler)
    assert trainer["scaler"].state_dict()["_growth_tracker"] == 1
    path = tmp_path / "amp.ckpt"
    evidence = save(path, trainer, sampler, 1)
    result = advance(trainer, fixture, sampler)
    expected_state = state(trainer)
    assert trainer["scaler"].get_scale() == 4.
    resumed = components()
    resumed["scaler"] = torch.amp.GradScaler("cpu", enabled=True, init_scale=2., growth_interval=2)
    restored_sampler = FullOrbitSampler(TABLE)
    load(path, evidence, resumed, restored_sampler)
    actual = advance(resumed, fixture, restored_sampler)
    assert actual.loss.total.item() == result.loss.total.item()
    assert state(resumed) == expected_state


def test_restricted_loading_has_no_unsafe_fallback(tmp_path, fixture):
    trainer, sampler = components(), FullOrbitSampler(TABLE)
    payload = create_temporal_payload(**trainer, config=CONFIG, bindings=BINDINGS,
                                     progress=progress(), sampler_state=sampler.state_dict())
    # Filesystem Path is safe to construct here but is not a weights-only type.
    payload["bindings"]["untrusted_object"] = tmp_path
    path = tmp_path / "unsafe-object.ckpt"
    evidence = write_payload(path, payload)
    with pytest.raises(TemporalCheckpointError, match="restricted_load_failed"):
        load(path, evidence, components())


def test_campaign_actual_environment_bindings_are_checkpoint_safe(tmp_path, fixture):
    """Native gate regression: JSON accepted TorchVersion but safe save did not."""
    from music_critic.experiments.analysisgnn.temporal_slices import campaign
    from music_critic.experiments.analysisgnn.temporal_slices.checkpoint import _snapshot
    raw = old.environment_fingerprint()
    assert isinstance(raw["torch"], str)
    if type(raw["torch"]) is not str:
        with pytest.raises(TemporalCheckpointError, match="unsupported_or_nonfinite_state"):
            _snapshot(raw)
    environment = campaign.campaign_environment()
    assert type(environment["torch"]) is str
    assert environment["torch"] == str(torch.__version__)
    assert {k: v for k, v in environment.items() if k in raw} == raw
    bindings = {"source": "fixture-source", "data": "fixture-data", "environment": environment,
                "scientific_config": fingerprint(campaign.scientific_config())}
    parts = campaign.components("cpu")
    trainer = dict(zip(("model", "optimizer", "scheduler", "scaler"), parts[:4]))
    sampler = FullOrbitSampler(TABLE)
    advance(trainer, fixture, sampler)
    path = tmp_path / "actual-runtime-environment.ckpt"
    pin = save_temporal_checkpoint(path, **trainer, config=campaign.scientific_config(),
                                   bindings=bindings, progress=progress(1), sampler_state=sampler.state_dict())
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert type(saved["bindings"]["environment"]["torch"]) is str
    expected = advance(trainer, fixture, sampler)
    expected_state = state(trainer)
    parts = campaign.components("cpu")
    resumed = dict(zip(("model", "optimizer", "scheduler", "scaler"), parts[:4]))
    restored_sampler = FullOrbitSampler(TABLE)
    restored = load_temporal_checkpoint(path, expected_sha256=pin["sha256"], expected_size_bytes=pin["size_bytes"],
                                       **resumed, config=campaign.scientific_config(), bindings=bindings,
                                       sampler=restored_sampler)
    assert restored["progress"]["applied_updates"] == 1
    repeated = advance(resumed, fixture, restored_sampler)
    assert expected.loss.total.item() == repeated.loss.total.item()
    assert state(resumed) == expected_state
