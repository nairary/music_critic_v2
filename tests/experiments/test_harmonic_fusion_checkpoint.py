"""Synthetic model-only persistence; no historical weights or corpus access."""

from copy import deepcopy
from hashlib import sha256
import random

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_checkpoint as checkpoint
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import build_source_free_fixture
from music_critic.experiments.analysisgnn.harmonic_logit_fusion import HarmonicFusionModel
from music_critic.experiments.pop909_cl_frozen import checkpoint as old_checkpoint


@pytest.fixture(scope="module")
def synthetic_payload():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(207)
        model = HarmonicFusionModel(CorrectedAnalysisGNNModel())
        # Round-trip real nonzero fusion tensors, not only zero initialization.
        with torch.no_grad():
            for parameter in model.fusion.parameters():
                parameter.fill_(0.003)
        return checkpoint.create_model_payload(model)


def _save(tmp_path, payload):
    path = tmp_path / "synthetic-fusion.pt"
    torch.save(payload, path)
    content = path.read_bytes()
    return path, {"expected_sha256": sha256(content).hexdigest(), "expected_size_bytes": len(content)}


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_same(expected):
    actual = _rng_snapshot()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0] and np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def _deny(*args, **kwargs):
    raise AssertionError("forbidden state mutation or training-state operation")


def test_restricted_roundtrip_freezes_complete_model_and_preserves_rng(tmp_path, synthetic_payload, monkeypatch):
    path, expected = _save(tmp_path, synthetic_payload)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", _deny)
    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", _deny)
    original_load = torch.load
    calls = []

    def tracked_load(source, **kwargs):
        assert sha256(source.getvalue()).hexdigest() == expected["expected_sha256"]
        calls.append(kwargs)
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    before = _rng_snapshot()
    loaded = checkpoint.load_fusion_checkpoint(path, **expected)
    _assert_rng_same(before)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert all(not module.training for module in loaded.model.modules())
    assert all(not p.requires_grad and p.device.type == "cpu" for p in loaded.model.parameters())
    assert model_state_fingerprint(loaded.model) == synthetic_payload["model_state_fingerprint"]
    assert any(name.startswith("base.") for name in loaded.model.state_dict())
    assert any(name.startswith("fusion.") for name in loaded.model.state_dict())
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, synthetic_payload["model_state"][name])
    assert loaded.evidence["model_contract"] == synthetic_payload["model_contract"]
    assert loaded.evidence["artifact_sha256"] == expected["expected_sha256"]
    assert loaded.evidence["optimizer_restored"] is False
    assert loaded.evidence["checkpoint_rng_applied"] is False


@pytest.mark.parametrize("field,value,category", [
    ("expected_sha256", "0" * 64, "artifact.sha256_mismatch"),
    ("expected_sha256", "missing", "artifact.expected_sha256_invalid"),
    ("expected_size_bytes", 1, "artifact.size_mismatch"),
    ("expected_size_bytes", True, "artifact.expected_size_invalid"),
])
def test_wrong_artifact_rejected_before_parse(tmp_path, synthetic_payload, monkeypatch, field, value, category):
    path, expected = _save(tmp_path, synthetic_payload)
    expected[field] = value
    monkeypatch.setattr(torch, "load", _deny)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.HarmonicFusionCheckpointError) as error:
        checkpoint.load_fusion_checkpoint(path, **expected)
    assert error.value.category == category
    _assert_rng_same(before)


@pytest.mark.parametrize("mutation,category", [
    ("schema", "checkpoint.schema_invalid"),
    ("extra_field", "checkpoint.envelope_keys_mismatch"),
    ("missing_field", "checkpoint.envelope_keys_mismatch"),
    ("contract_fingerprint", "checkpoint.model_contract_mismatch"),
    ("contract_payload", "checkpoint.model_contract_mismatch"),
    ("missing_base_key", "checkpoint.state_keys_mismatch"),
    ("missing_fusion_key", "checkpoint.state_keys_mismatch"),
    ("extra_key", "checkpoint.state_keys_mismatch"),
    ("shape", "checkpoint.tensor_shape_mismatch"),
    ("dtype", "checkpoint.tensor_dtype_mismatch"),
    ("nonfinite", "checkpoint.tensor_nonfinite"),
    ("state_fingerprint", "checkpoint.state_fingerprint_mismatch"),
])
def test_invalid_envelope_rejected_before_state_load(tmp_path, synthetic_payload, monkeypatch, mutation, category):
    payload = dict(synthetic_payload)
    payload["model_state"] = dict(payload["model_state"])
    key = next(name for name in payload["model_state"] if name.startswith("fusion."))
    tensor = payload["model_state"][key]
    if mutation == "schema":
        payload["schema_version"] = old_checkpoint.MODEL_ONLY_CHECKPOINT_SCHEMA
    elif mutation == "extra_field":
        payload["optimizer_state"] = {}
    elif mutation == "missing_field":
        del payload["model_contract"]
    elif mutation == "contract_fingerprint":
        payload["model_contract_fingerprint"] = "0" * 64
    elif mutation == "contract_payload":
        payload["model_contract"] = deepcopy(payload["model_contract"])
        payload["model_contract"]["unexpected_mode"] = True
    elif mutation == "missing_base_key":
        del payload["model_state"][next(name for name in payload["model_state"] if name.startswith("base."))]
    elif mutation == "missing_fusion_key":
        del payload["model_state"][key]
    elif mutation == "extra_key":
        payload["model_state"]["unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        payload["model_state"][key] = tensor.reshape(-1)[:1]
    elif mutation == "dtype":
        payload["model_state"][key] = tensor.double()
    elif mutation == "nonfinite":
        payload["model_state"][key] = tensor.clone()
        payload["model_state"][key].reshape(-1)[0] = float("inf")
    else:
        payload["model_state_fingerprint"] = "0" * 64
    path, expected = _save(tmp_path, payload)
    monkeypatch.setattr(HarmonicFusionModel, "load_state_dict", _deny)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.HarmonicFusionCheckpointError) as error:
        checkpoint.load_fusion_checkpoint(path, **expected)
    assert error.value.category == category
    _assert_rng_same(before)


def test_old_strict_loader_rejects_fusion_schema(tmp_path, synthetic_payload):
    path, expected = _save(tmp_path, synthetic_payload)
    with pytest.raises(old_checkpoint.FrozenCheckpointError) as error:
        old_checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.schema_invalid"


@pytest.mark.parametrize("mutation", ["vocabulary_order", "disabled"])
def test_self_consistent_different_contract_is_rejected(tmp_path, synthetic_payload, monkeypatch, mutation):
    payload = dict(synthetic_payload)
    contract = deepcopy(payload["model_contract"])
    if mutation == "vocabulary_order":
        registry = contract["class_registry"]
        root = next(row for row in registry["tasks"] if row["task_id"] == "root")
        root["class_labels"][0], root["class_labels"][1] = root["class_labels"][1], root["class_labels"][0]
        registry["fingerprint"] = fingerprint({key: value for key, value in registry.items() if key != "fingerprint"})
    else:
        contract["enabled"] = False
    contract["fingerprint"] = fingerprint({key: value for key, value in contract.items() if key != "fingerprint"})
    payload["model_contract"] = contract
    payload["model_contract_fingerprint"] = contract["fingerprint"]
    path, expected = _save(tmp_path, payload)
    monkeypatch.setattr(HarmonicFusionModel, "load_state_dict", _deny)
    with pytest.raises(checkpoint.HarmonicFusionCheckpointError) as error:
        checkpoint.load_fusion_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.model_contract_mismatch"


def test_unrestricted_pickle_is_never_a_fallback(tmp_path, synthetic_payload, monkeypatch):
    payload = dict(synthetic_payload, forbidden_numpy_state=np.random.get_state())
    path, expected = _save(tmp_path, payload)
    original_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs)
        random.random(), np.random.random(), torch.rand(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.HarmonicFusionCheckpointError) as error:
        checkpoint.load_fusion_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.restricted_load_failed"
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    _assert_rng_same(before)


def test_constructor_failure_preserves_rng(tmp_path, synthetic_payload, monkeypatch):
    path, expected = _save(tmp_path, synthetic_payload)

    def fail_constructor(*args, **kwargs):
        random.random(), np.random.random(), torch.rand(3)
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(checkpoint, "HarmonicFusionModel", fail_constructor)
    before = _rng_snapshot()
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        checkpoint.load_fusion_checkpoint(path, **expected)
    _assert_rng_same(before)


def test_disabled_export_rejected_and_snapshot_detached():
    model = HarmonicFusionModel(CorrectedAnalysisGNNModel(), enabled=False)
    with pytest.raises(checkpoint.HarmonicFusionCheckpointError, match="enabled_model_required"):
        checkpoint.create_model_payload(model)
    model = HarmonicFusionModel(model.base, enabled=True)
    before = _rng_snapshot()
    payload = checkpoint.create_model_payload(model)
    _assert_rng_same(before)
    with torch.no_grad():
        next(model.fusion.parameters()).add_(1)
    assert model_state_fingerprint(payload["model_state"]) == payload["model_state_fingerprint"]
    assert model_state_fingerprint(model) != payload["model_state_fingerprint"]


def test_external_optimizer_state_restoration_reproduces_next_synthetic_step(tmp_path):
    # This unit test saves optimizer state separately in memory. The production
    # loader remains model-only; no scheduler/sampler/dropout resume is claimed.
    batch, _ = build_source_free_fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(208)
        model = HarmonicFusionModel(CorrectedAnalysisGNNModel()).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def step(module, opt):
        opt.zero_grad(set_to_none=True)
        loss = sum(logits.square().mean() for logits in module.predict(batch.raw_graph_batch).logits.values())
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())
        opt.step()
        return loss.detach()

    for _ in range(2):
        step(model, optimizer)
    assert any(bool(p.ne(0).any()) for p in model.fusion.parameters())
    path, expected = _save(tmp_path, checkpoint.create_model_payload(model))
    external_optimizer_state = deepcopy(optimizer.state_dict())
    restored = checkpoint.load_fusion_checkpoint(path, **expected).model
    restored.requires_grad_(True)
    second_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=0.01)
    second_optimizer.load_state_dict(external_optimizer_state)
    assert torch.equal(step(model, optimizer), step(restored, second_optimizer))
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
