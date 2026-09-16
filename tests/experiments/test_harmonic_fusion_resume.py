"""Synthetic full-state persistence and hostile-input acceptance, CPU only."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import random

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_resume as resume
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    build_source_free_fixture, load_frozen_class_weights,
)
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair
from music_critic.experiments.analysisgnn.harmonic_fusion_checkpoint import create_model_payload
from music_critic.experiments.analysisgnn.harmonic_fusion_training import (
    FusionTrainingConfig, FusionTrainingSession,
)


@pytest.fixture(scope="module")
def fixture():
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"])
                  for shift in (0, 1, 2))
    config = FusionTrainingConfig(total_updates=4, warmup_updates=2,
                                  validation_interval=2, checkpoint_interval=1,
                                  validation_mode="synthetic_disabled")
    with resume.preserve_caller_rng():
        torch.manual_seed(507)
        base = CorrectedAnalysisGNNModel()
    bindings = {"source": "synthetic-fixture-only", "data_fingerprint": fingerprint(sidecar),
                "source_model_state_fingerprint": model_state_fingerprint(base),
                "class_weight_fingerprint": weights.payload_fingerprint}

    def reader(record_id, *, split):
        assert split == "train" and record_id == sidecar["record_id"]
        # Exercise all three RNG domains, not only dropout.
        random.random(), np.random.random(), torch.rand(1)
        return batch.raw_graph_batch, deepcopy(sidecar)

    def session(enabled=True):
        return FusionTrainingSession(base=base, enabled=enabled, config=config,
                                     bindings=bindings, table=table)

    return session, config, bindings, table, reader, weights


@pytest.fixture(scope="module")
def progressed_payload(fixture):
    session, _, _, _, reader, weights = fixture
    trainer = session()
    trainer.run_until(2, reader, weights)
    return resume.create_training_payload(trainer)


def _save(tmp_path, payload):
    path = tmp_path / "synthetic-trainer.pt"
    torch.save(payload, path)
    content = path.read_bytes()
    return path, {"expected_sha256": sha256(content).hexdigest(), "expected_size_bytes": len(content)}


def _load(path, hashes, fixture, **overrides):
    _, config, bindings, table, _, _ = fixture
    kwargs = {"config": config, "bindings": bindings, "table": table}
    kwargs.update(overrides)
    return resume.load_training_checkpoint(path, **hashes, **kwargs)


def _assert_same(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same(a, b)
    else:
        assert left == right


def _deny(*args, **kwargs):
    raise AssertionError("state application forbidden before complete validation")


@pytest.mark.parametrize("enabled", [False, True])
def test_mid_orbit_resume_exactly_matches_uninterrupted_complete_training(tmp_path, fixture, enabled):
    make_session, _, _, _, reader, weights = fixture
    uninterrupted = make_session(enabled)
    interrupted = make_session(enabled)
    before = resume.capture_safe_rng()
    uninterrupted.run_until(4, reader, weights)
    interrupted.run_until(2, reader, weights)
    assert interrupted.sampler.position == 4  # Nonzero position in second orbit.
    path = tmp_path / "update2.pt"
    evidence = resume.save_training_checkpoint(interrupted, path)
    loaded = _load(path, {"expected_sha256": evidence["sha256"],
                          "expected_size_bytes": evidence["size_bytes"]}, fixture,
                   expected_enabled=enabled)
    assert loaded is not interrupted
    assert not ({id(p) for p in loaded.model.parameters()} & {id(p) for p in interrupted.model.parameters()})
    loaded.run_until(4, reader, weights)
    _assert_same(resume.create_training_payload(loaded), resume.create_training_payload(uninterrupted))
    _assert_same(before, resume.capture_safe_rng())
    if enabled:
        assert any(bool(parameter.ne(0).any()) for parameter in loaded.model.fusion.parameters())
    assert any(value == 0 for value in loaded.optimizer_steps.values())  # Unused encoder heads.
    path.unlink()


def test_snapshot_detached_immutable_save_and_initial_state_roundtrip(tmp_path, fixture):
    trainer = fixture[0]()
    payload = resume.create_training_payload(trainer)
    path = tmp_path / "initial.pt"
    evidence = resume.save_training_checkpoint(trainer, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        resume.save_training_checkpoint(trainer, path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".initial.pt.*"))
    loaded = _load(path, {"expected_sha256": evidence["sha256"],
                          "expected_size_bytes": evidence["size_bytes"]}, fixture)
    assert loaded.applied_updates == 0 and loaded.optimizer.state_dict()["state"] == {}
    with torch.no_grad():
        next(trainer.model.parameters()).add_(1)
    assert model_state_fingerprint(payload["model_state"]) == payload["model_state_fingerprint"]
    assert model_state_fingerprint(trainer.model) != payload["model_state_fingerprint"]
    path.unlink()


@pytest.mark.parametrize("mutation,category", [
    ("envelope", "fusion_resume.envelope_keys_mismatch"),
    ("model_shape", "checkpoint.tensor_shape_mismatch"),
    ("model_dtype", "checkpoint.tensor_dtype_mismatch"),
    ("model_nan", "checkpoint.tensor_nonfinite"),
    ("model_missing_fusion", "checkpoint.state_keys_mismatch"),
    ("model_fingerprint", "fusion_resume.model_state_fingerprint_mismatch"),
    ("classes", "fusion_resume.model_contract_mismatch"),
    ("optimizer_names", "fusion_resume.optimizer_contract_mismatch"),
    ("optimizer_param_order", "fusion_resume.optimizer_groups_mismatch"),
    ("optimizer_lr", "fusion_resume.optimizer_groups_mismatch"),
    ("optimizer_missing_slot", "fusion_resume.optimizer_slots_invalid"),
    ("optimizer_missing_state", "fusion_resume.optimizer_state_inventory_mismatch"),
    ("optimizer_shape", "fusion_resume.optimizer_tensor_invalid"),
    ("optimizer_dtype", "fusion_resume.optimizer_tensor_invalid"),
    ("optimizer_nonfinite", "fusion_resume.optimizer_tensor_invalid"),
    ("optimizer_negative_variance", "fusion_resume.optimizer_tensor_invalid"),
    ("optimizer_step", "fusion_resume.optimizer_step_invalid"),
    ("optimizer_counter", "fusion_resume.optimizer_step_invalid"),
    ("scheduler_epoch", "fusion_resume.scheduler_mismatch"),
    ("scheduler_step", "fusion_resume.scheduler_mismatch"),
    ("scheduler_lr", "fusion_resume.scheduler_mismatch"),
    ("scaler", "fusion_resume.scaler_mismatch"),
    ("sampler_position", "fusion_resume.sampler_mismatch"),
    ("sampler_fingerprint", "fusion_resume.sampler_mismatch"),
    ("history_draw", "fusion_resume.history_mismatch"),
    ("history_length", "fusion_resume.history_invalid"),
    ("source_state", "fusion_resume.source_model_state_mismatch"),
    ("rng_python", "fusion_resume.rng_python_invalid"),
    ("rng_numpy", "fusion_resume.rng_numpy_invalid"),
    ("rng_torch", "fusion_resume.rng_invalid"),
    ("validation", "fusion_resume.validations_invalid"),
])
def test_corrupt_full_state_rejected_before_any_application(tmp_path, fixture, progressed_payload,
                                                         monkeypatch, mutation, category):
    payload = deepcopy(progressed_payload)
    name = next(key for key in payload["model_state"] if key.startswith("fusion."))
    optimizer = payload["optimizer_state"]
    key = next(iter(optimizer["state"]))
    state = optimizer["state"][key]
    if mutation == "envelope":
        payload["extra"] = True
    elif mutation.startswith("model_"):
        if mutation == "model_shape":
            payload["model_state"][name] = torch.zeros(1)
        elif mutation == "model_dtype":
            payload["model_state"][name] = payload["model_state"][name].double()
        elif mutation == "model_nan":
            payload["model_state"][name][0, 0] = float("nan")
        elif mutation == "model_missing_fusion":
            del payload["model_state"][name]
        else:
            payload["model_state_fingerprint"] = "0" * 64
    elif mutation == "classes":
        contract = payload["model_contract"]
        registry = contract["class_registry"]
        registry["tasks"][0]["class_labels"].reverse()
        registry["fingerprint"] = fingerprint({k: v for k, v in registry.items() if k != "fingerprint"})
        contract["fingerprint"] = fingerprint({k: v for k, v in contract.items() if k != "fingerprint"})
        payload["model_contract_fingerprint"] = contract["fingerprint"]
    elif mutation == "optimizer_names":
        payload["optimizer_contract"]["ordered_parameter_names"].reverse()
    elif mutation == "optimizer_param_order":
        optimizer["param_groups"][0]["params"].reverse()
    elif mutation == "optimizer_lr":
        optimizer["param_groups"][0]["lr"] *= 2
    elif mutation == "optimizer_missing_slot":
        del state["exp_avg_sq"]
    elif mutation == "optimizer_missing_state":
        del optimizer["state"][key]
    elif mutation == "optimizer_shape":
        state["exp_avg"] = torch.zeros(1)
    elif mutation == "optimizer_dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif mutation == "optimizer_nonfinite":
        state["exp_avg"].flatten()[0] = float("inf")
    elif mutation == "optimizer_negative_variance":
        state["exp_avg_sq"].flatten()[0] = -1
    elif mutation == "optimizer_step":
        state["step"] -= 1
    elif mutation == "optimizer_counter":
        parameter_name = payload["optimizer_contract"]["ordered_parameter_names"][key]
        payload["optimizer_steps"][parameter_name] -= 1
    elif mutation.startswith("scheduler_"):
        field = {"scheduler_epoch": "last_epoch", "scheduler_step": "_step_count", "scheduler_lr": "_last_lr"}[mutation]
        payload["scheduler_state"][field] = [0.2] if mutation == "scheduler_lr" else 99
    elif mutation == "scaler":
        payload["scaler_state"] = {"scale": 1024.0}
    elif mutation == "sampler_position":
        payload["sampler_state"]["position"] += 1
    elif mutation == "sampler_fingerprint":
        payload["sampler_state"]["fingerprint"] = "0" * 64
    elif mutation == "history_draw":
        payload["history"][0]["draws"][0]["shift_pc"] = 11
    elif mutation == "history_length":
        payload["history"].pop()
    elif mutation == "source_state":
        payload["source_model_state"] = "0" * 64
    elif mutation == "rng_python":
        payload["rng_state"]["python"]["state"][-1] = True
    elif mutation == "rng_numpy":
        payload["rng_state"]["numpy"]["keys"] = torch.zeros(624, dtype=torch.float32)
    elif mutation == "rng_torch":
        payload["rng_state"]["torch_cpu"] = torch.zeros(12, dtype=torch.uint8)
    else:
        payload["validations"] = [{"update": 1, "metrics": {"corrected_primary_macro_score": 0.2}}]
    path, hashes = _save(tmp_path, payload)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", _deny)
    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", _deny)
    before = resume.capture_safe_rng()
    with pytest.raises(resume.FusionResumeError) as error:
        _load(path, hashes, fixture)
    assert error.value.category == category
    _assert_same(before, resume.capture_safe_rng())
    path.unlink()


@pytest.mark.parametrize("changed,category", [
    ("sha", "artifact.sha256_mismatch"),
    ("size", "artifact.size_mismatch"),
    ("config", "fusion_resume.config_mismatch"),
    ("bindings", "fusion_resume.bindings_mismatch"),
    ("table", "fusion_resume.sampler_mismatch"),
    ("arm", "fusion_resume.arm_mismatch"),
])
def test_resume_requires_expected_artifact_config_data_and_arm(tmp_path, fixture, progressed_payload,
                                                            changed, category):
    path, hashes = _save(tmp_path, progressed_payload)
    overrides = {}
    if changed == "sha":
        hashes["expected_sha256"] = "0" * 64
    elif changed == "size":
        hashes["expected_size_bytes"] -= 1
    elif changed == "config":
        overrides["config"] = replace(fixture[1], peak_lr=0.0002)
    elif changed == "bindings":
        overrides["bindings"] = dict(fixture[2], source="another-synthetic-source")
    elif changed == "table":
        overrides["table"] = fixture[3][:2]
    else:
        overrides["expected_enabled"] = False
    with pytest.raises(resume.FusionResumeError) as error:
        _load(path, hashes, fixture, **overrides)
    assert error.value.category == category
    path.unlink()


def test_restricted_same_checked_bytes_only_and_rng_failure_atomic(tmp_path, fixture, progressed_payload, monkeypatch):
    payload = dict(progressed_payload, unsafe_numpy=np.random.get_state())
    path, hashes = _save(tmp_path, payload)
    original = torch.load
    calls = []

    def checked_load(source, **kwargs):
        assert sha256(source.getvalue()).hexdigest() == hashes["expected_sha256"]
        assert len(source.getvalue()) == hashes["expected_size_bytes"]
        calls.append(kwargs)
        random.random(), np.random.random(), torch.rand(1)
        return original(source, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    before = resume.capture_safe_rng()
    with pytest.raises(resume.FusionResumeError, match="restricted_load_failed"):
        _load(path, hashes, fixture)
    _assert_same(before, resume.capture_safe_rng())
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    path.unlink()


def test_model_only_is_never_training_resume(tmp_path, fixture):
    path, hashes = _save(tmp_path, create_model_payload(fixture[0]().model))
    with pytest.raises(resume.FusionResumeError, match="schema_invalid"):
        _load(path, hashes, fixture)
    path.unlink()


def test_rng_restore_validates_entire_snapshot_before_mutation():
    before = resume.capture_safe_rng()
    changed = deepcopy(before)
    changed["python"]["state"][0] += 1
    changed["torch_cpu"] = torch.zeros(5, dtype=torch.uint8)
    with pytest.raises(resume.FusionResumeError, match="rng_invalid"):
        resume.restore_safe_rng(changed)
    _assert_same(before, resume.capture_safe_rng())


def test_late_application_failure_preserves_caller_live_trainer_and_artifact(tmp_path, fixture,
                                                                         progressed_payload, monkeypatch):
    live = fixture[0]()
    before_live = resume.create_training_payload(live)
    before_rng = resume.capture_safe_rng()
    path, hashes = _save(tmp_path, progressed_payload)

    def fail_optimizer(self, state):
        random.random(), np.random.random(), torch.rand(1)
        raise RuntimeError("synthetic late optimizer application failure")

    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", fail_optimizer)
    with pytest.raises(RuntimeError, match="late optimizer application"):
        _load(path, hashes, fixture)
    _assert_same(before_live, resume.create_training_payload(live))
    _assert_same(before_rng, resume.capture_safe_rng())
    assert sha256(path.read_bytes()).hexdigest() == hashes["expected_sha256"]
    path.unlink()


def test_failed_serialization_never_publishes_or_replaces_checkpoint(tmp_path, fixture, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("synthetic serialization failure")

    monkeypatch.setattr(torch, "save", fail_save)
    target = tmp_path / "unpublished.pt"
    with pytest.raises(OSError, match="serialization failure"):
        resume.save_training_checkpoint(fixture[0](), target)
    assert not target.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation,category", [("base", "initial_source_state_mismatch"),
                                               ("fusion", "initial_fusion_nonzero")])
def test_initial_state_cannot_forge_fresh_continuation(tmp_path, fixture, mutation, category):
    payload = resume.create_training_payload(fixture[0]())
    name = next(name for name in payload["model_state"] if name.startswith(mutation + "."))
    payload["model_state"][name].add_(0.01)
    payload["model_state_fingerprint"] = model_state_fingerprint(payload["model_state"])
    path, hashes = _save(tmp_path, payload)
    with pytest.raises(resume.FusionResumeError, match=category):
        _load(path, hashes, fixture)
    path.unlink()


def test_snapshot_and_restore_python_and_numpy_cached_gaussians():
    with resume.preserve_caller_rng():
        random.seed(1234)
        np.random.seed(1234)
        random.gauss(0, 1)
        np.random.normal()
        state = resume.capture_safe_rng()
        expected = (random.gauss(0, 1), np.random.normal(), torch.rand(3))
        resume.restore_safe_rng(state)
        actual = (random.gauss(0, 1), np.random.normal(), torch.rand(3))
        _assert_same(expected, actual)
