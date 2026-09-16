"""CUDA resume guards exercised with CPU tensors and fake CUDA interfaces only.

These tests establish rejection/order/atomicity of the contract. They are not
evidence that CUDA training or CUDA interrupted continuation was executed.
"""

from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_resume as resume
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.harmonic_fusion_training import (
    FusionTrainingConfig, FusionTrainingSession,
)


def _deny(*args, **kwargs):
    pytest.fail("unexpected runtime access or state application")


def _same(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _same(first, second)
    else:
        assert left == right


@pytest.fixture
def cpu_rng(monkeypatch):
    # No CUDA availability query or initialization is needed to make a fixture.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _deny)
    return resume.capture_safe_rng()


def _fake_inventory(monkeypatch, *, count=2, initialized=True, fail_ordinal=None):
    """Only fresh fake CUDA generators accept the test's three-byte states."""
    original_generator = torch.Generator
    checked = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _deny)

    class FakeGenerator:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        def set_state(self, state):
            checked.append((self.ordinal, state.clone()))
            if self.ordinal == fail_ordinal or state.shape != (3,):
                raise RuntimeError("synthetic CUDA generator rejected state")

    def generator(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        if str(device).startswith("cuda:"):
            return FakeGenerator(int(str(device).split(":")[1]))
        assert str(device) == "cpu"
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", generator)
    return checked


def _cuda_states(count):
    return [torch.tensor([index + 1, 10, 20], dtype=torch.uint8, device="cpu")
            for index in range(count)]


@pytest.mark.parametrize("mutation,category", [
    ("empty", "rng_cuda_inventory_missing"),
    ("missing", "rng_invalid"),
    ("tuple", "rng_cuda_invalid"),
    ("none", "rng_cuda_invalid"),
])
def test_cuda_missing_inventory_fails_before_any_runtime_or_generator_access(cpu_rng, monkeypatch,
                                                                          mutation, category):
    supplied = deepcopy(cpu_rng)
    if mutation == "missing":
        del supplied["torch_cuda"]
    elif mutation == "tuple":
        supplied["torch_cuda"] = ()
    elif mutation == "none":
        supplied["torch_cuda"] = None
    monkeypatch.setattr(torch.cuda, "is_initialized", _deny)
    monkeypatch.setattr(torch.cuda, "device_count", _deny)
    monkeypatch.setattr(torch, "Generator", _deny)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    with pytest.raises(resume.FusionResumeError, match=category):
        resume.restore_safe_rng(supplied, device_type="cuda")


@pytest.mark.parametrize("saved_count,runtime_count,initialized", [
    (1, 2, True), (3, 2, True), (1, 0, True), (1, True, True), (2, 2, False),
])
def test_cuda_inventory_must_match_complete_initialized_runtime(cpu_rng, monkeypatch,
                                                              saved_count, runtime_count, initialized):
    supplied = dict(cpu_rng, torch_cuda=_cuda_states(saved_count))
    _fake_inventory(monkeypatch, count=runtime_count, initialized=initialized)
    monkeypatch.setattr(torch, "Generator", _deny)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    with pytest.raises(resume.FusionResumeError, match="rng_cuda_runtime_mismatch"):
        resume.restore_safe_rng(supplied, device_type="cuda")


@pytest.mark.parametrize("mutation", ["dtype", "rank", "empty", "not_tensor"])
def test_cuda_malformed_tensor_rejected_before_any_rng_application(cpu_rng, monkeypatch, mutation):
    supplied = dict(cpu_rng, torch_cuda=_cuda_states(2))
    if mutation == "dtype":
        supplied["torch_cuda"][1] = torch.zeros(3, dtype=torch.float32)
    elif mutation == "rank":
        supplied["torch_cuda"][1] = torch.zeros((1, 3), dtype=torch.uint8)
    elif mutation == "empty":
        supplied["torch_cuda"][1] = torch.zeros(0, dtype=torch.uint8)
    else:
        supplied["torch_cuda"][1] = [1, 10, 20]
    checked = _fake_inventory(monkeypatch)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    with pytest.raises(resume.FusionResumeError, match="rng_torch_invalid"):
        resume.restore_safe_rng(supplied, device_type="cuda")
    assert not checked


def test_all_cuda_ordinals_validate_in_order_before_any_live_state_application(cpu_rng, monkeypatch):
    supplied = dict(cpu_rng, torch_cuda=_cuda_states(2))
    checked = _fake_inventory(monkeypatch)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    resume._validate_safe_rng(supplied, device_type="cuda")
    assert [ordinal for ordinal, _ in checked] == [0, 1]
    for (_, actual), expected in zip(checked, supplied["torch_cuda"], strict=True):
        assert torch.equal(actual, expected)


def test_later_cuda_generator_rejection_never_applies_earlier_rng(cpu_rng, monkeypatch):
    supplied = dict(cpu_rng, torch_cuda=_cuda_states(2))
    checked = _fake_inventory(monkeypatch, fail_ordinal=1)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    with pytest.raises(resume.FusionResumeError, match="rng_cuda_invalid"):
        resume.restore_safe_rng(supplied, device_type="cuda")
    assert [ordinal for ordinal, _ in checked] == [0, 1]


def test_complete_fake_cuda_inventory_restores_only_after_all_validation(cpu_rng, monkeypatch):
    supplied = dict(cpu_rng, torch_cuda=_cuda_states(2))
    checked = _fake_inventory(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: _cuda_states(2))
    applied = []

    def apply(state):
        assert [ordinal for ordinal, _ in checked] == [0, 1]
        applied.append(state)

    monkeypatch.setattr(resume, "_apply_rng", apply)
    resume.restore_safe_rng(supplied, device_type="cuda")
    assert applied == [supplied]


def test_cpu_empty_cuda_inventory_remains_valid_without_cuda_access(cpu_rng, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", _deny)
    monkeypatch.setattr(torch.cuda, "device_count", _deny)
    resume._validate_safe_rng(cpu_rng, device_type="cpu")


def test_device_type_contract_rejects_unknown_value(cpu_rng):
    with pytest.raises(resume.FusionResumeError, match="rng_device_type_invalid"):
        resume._validate_safe_rng(cpu_rng, device_type="mps")


@pytest.fixture(scope="module")
def initial_training():
    config = FusionTrainingConfig(total_updates=2, warmup_updates=1,
                                  validation_interval=1, checkpoint_interval=1)
    table = (FullOrbitPair("synthetic:cuda-rng-negative", 0, "synthetic:component"),)
    bindings = {"source": "synthetic-negative-contract-test"}
    with resume.preserve_caller_rng():
        trainer = FusionTrainingSession(base=CorrectedAnalysisGNNModel(), enabled=True,
                                        config=config, table=table, bindings=bindings)
    trainer.run_until(0, _deny, None,
                      validate=lambda *args, **kwargs: {"corrected_primary_macro_score": 0.5})
    return trainer, resume.create_training_payload(trainer)


def test_cuda_checkpoint_empty_rng_is_rejected_before_constructor_or_any_state_load(
        tmp_path, initial_training, monkeypatch):
    reference, initial_payload = initial_training
    payload = deepcopy(initial_payload)
    payload["device_type"] = "cuda"
    payload["rng_state"]["torch_cuda"] = []
    path = tmp_path / "synthetic-missing-cuda-rng.pt"
    torch.save(payload, path)
    content = path.read_bytes()
    monkeypatch.setattr(resume, "_fresh_session", _deny)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", _deny)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _deny)
    monkeypatch.setattr(torch, "Generator", _deny)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", _deny)
    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", _deny)
    monkeypatch.setattr(FullOrbitSampler, "load_state_dict", _deny)
    monkeypatch.setattr(resume, "_apply_rng", _deny)
    before = resume.capture_safe_rng()
    with pytest.raises(resume.FusionResumeError, match="rng_cuda_inventory_missing"):
        resume.load_training_checkpoint(path, expected_sha256=sha256(content).hexdigest(),
                                        expected_size_bytes=len(content), config=reference.config,
                                        bindings=reference.bindings, table=reference.sampler.table,
                                        device="cuda")
    _same(before, resume.capture_safe_rng())
    assert path.read_bytes() == content
