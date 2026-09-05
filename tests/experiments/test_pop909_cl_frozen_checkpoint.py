from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import random

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn import corrected_training
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    model_state_fingerprint,
)
from music_critic.experiments.pop909_cl_frozen import checkpoint


@pytest.fixture(scope="module")
def synthetic_payload():
    # Only randomly initialized synthetic weights; never a training artifact.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1904)
        model = CorrectedAnalysisGNNModel()
        return checkpoint.create_model_only_payload(model)


def _save(tmp_path, payload):
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    content = path.read_bytes()
    return path, {"expected_sha256": sha256(content).hexdigest(), "expected_size_bytes": len(content)}


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_same(expected):
    actual = _rng_snapshot()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def _forbidden(*args, **kwargs):
    raise AssertionError("resume, optimizer or saved RNG API must not be called")


def test_restricted_strict_synthetic_roundtrip_preserves_rng(
    tmp_path, synthetic_payload, monkeypatch
):
    path, expected = _save(tmp_path, synthetic_payload)
    for name in ("load_checkpoint", "restore_rng_state", "capture_rng_state", "build_optimizer_scheduler"):
        monkeypatch.setattr(corrected_training, name, _forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", _forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", _forbidden)
    original_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    before = _rng_snapshot()
    loaded = checkpoint.load_model_checkpoint(path, **expected)
    _assert_rng_same(before)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert all(not module.training for module in loaded.model.modules())
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in loaded.model.parameters())
    assert model_state_fingerprint(loaded.model) == synthetic_payload["model_state_fingerprint"]
    for key, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, synthetic_payload["model_state"][key])
    assert loaded.evidence["artifact_sha256"] == expected["expected_sha256"]
    assert loaded.evidence["artifact_size_bytes"] == expected["expected_size_bytes"]
    assert loaded.evidence["contract_verification"] == "full_payload_and_current_fingerprint"
    assert loaded.evidence["checkpoint_rng_applied"] is False


@pytest.mark.parametrize("field,value,category", [
    ("expected_sha256", "0" * 64, "artifact.sha256_mismatch"),
    ("expected_sha256", "unknown", "artifact.expected_sha256_invalid"),
    ("expected_size_bytes", 1, "artifact.size_mismatch"),
    ("expected_size_bytes", 0, "artifact.expected_size_invalid"),
    ("expected_size_bytes", True, "artifact.expected_size_invalid"),
])
def test_artifact_rejected_before_deserialization(
    tmp_path, synthetic_payload, monkeypatch, field, value, category
):
    path, expected = _save(tmp_path, synthetic_payload)
    expected[field] = value
    monkeypatch.setattr(torch, "load", _forbidden)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.FrozenCheckpointError) as error:
        checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == category
    _assert_rng_same(before)


@pytest.mark.parametrize("mutation,category", [
    ("schema", "checkpoint.schema_invalid"),
    ("envelope_keys", "checkpoint.envelope_keys_mismatch"),
    ("contract_fingerprint", "checkpoint.model_contract_mismatch"),
    ("contract_payload", "checkpoint.model_contract_mismatch"),
    ("missing_state_key", "checkpoint.state_keys_mismatch"),
    ("extra_state_key", "checkpoint.state_keys_mismatch"),
    ("shape", "checkpoint.tensor_shape_mismatch"),
    ("dtype", "checkpoint.tensor_dtype_mismatch"),
    ("nonfinite", "checkpoint.tensor_nonfinite"),
    ("state_fingerprint", "checkpoint.state_fingerprint_mismatch"),
])
def test_invalid_checkpoint_rejected_and_rng_preserved(
    tmp_path, synthetic_payload, mutation, category
):
    payload = dict(synthetic_payload)
    payload["model_state"] = dict(payload["model_state"])
    first_key = next(iter(payload["model_state"]))
    tensor = payload["model_state"][first_key]
    if mutation == "schema":
        payload["schema_version"] = "unknown"
    elif mutation == "envelope_keys":
        payload["optimizer_state"] = {}
    elif mutation == "contract_fingerprint":
        payload["model_contract_fingerprint"] = "0" * 64
    elif mutation == "contract_payload":
        payload["model_contract"] = deepcopy(payload["model_contract"])
        payload["model_contract"]["active_head_count"] = 19
    elif mutation == "missing_state_key":
        del payload["model_state"][first_key]
    elif mutation == "extra_state_key":
        payload["model_state"]["extra"] = torch.ones(1)
    elif mutation == "shape":
        payload["model_state"][first_key] = tensor.reshape(-1)[:1]
    elif mutation == "dtype":
        payload["model_state"][first_key] = tensor.double()
    elif mutation == "nonfinite":
        payload["model_state"][first_key] = tensor.clone()
        payload["model_state"][first_key].reshape(-1)[0] = float("nan")
    elif mutation == "state_fingerprint":
        payload["model_state_fingerprint"] = "0" * 64
    path, expected = _save(tmp_path, payload)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.FrozenCheckpointError) as error:
        checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == category
    _assert_rng_same(before)


def test_constructor_failure_restores_all_caller_rng(tmp_path, synthetic_payload, monkeypatch):
    path, expected = _save(tmp_path, synthetic_payload)

    def failing_constructor():
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(checkpoint, "CorrectedAnalysisGNNModel", failing_constructor)
    before = _rng_snapshot()
    with pytest.raises(RuntimeError, match="synthetic constructor failure"):
        checkpoint.load_model_checkpoint(path, **expected)
    _assert_rng_same(before)


def test_restricted_load_failure_never_retries_unsafe(tmp_path, synthetic_payload, monkeypatch):
    payload = dict(synthetic_payload)
    payload["schema_version"] = checkpoint.CORRECTED_TRAINING_CHECKPOINT_SCHEMA
    payload["rng_state"] = {"numpy": np.random.get_state()}
    path, expected = _save(tmp_path, payload)
    original_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        random.random()
        np.random.random()
        torch.rand(3)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.FrozenCheckpointError) as error:
        checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.restricted_load_failed"
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    _assert_rng_same(before)


def test_fabricated_restricted_compatible_historical_envelope_ignores_resume(
    tmp_path, synthetic_payload, monkeypatch
):
    payload = dict(synthetic_payload)
    payload["schema_version"] = checkpoint.CORRECTED_TRAINING_CHECKPOINT_SCHEMA
    del payload["model_contract"]
    payload.update(
        optimizer_state={"poison": "must be ignored"},
        scheduler_state={"poison": "must be ignored"},
        amp_scaler_state={"poison": "must be ignored"},
        sampler_state={"poison": "must be ignored"},
        rng_state={"python": "invalid", "numpy": "invalid", "torch_cpu": torch.zeros(1)},
        resolved_config={"synthetic_only": True},
    )
    path, expected = _save(tmp_path, payload)
    for name in ("load_checkpoint", "restore_rng_state", "build_optimizer_scheduler"):
        monkeypatch.setattr(corrected_training, name, _forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", _forbidden)
    before = _rng_snapshot()
    loaded = checkpoint.load_model_checkpoint(path, **expected)
    _assert_rng_same(before)
    assert loaded.evidence["historical_contract_payload_available"] is False
    assert loaded.evidence["contract_verification"] == "reconstructed_current_contract_fingerprint_only"
    assert model_state_fingerprint(loaded.model) == synthetic_payload["model_state_fingerprint"]


def test_payload_is_a_detached_snapshot():
    model = CorrectedAnalysisGNNModel()
    payload = checkpoint.create_model_only_payload(model)
    original_fingerprint = payload["model_state_fingerprint"]
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert model_state_fingerprint(payload["model_state"]) == original_fingerprint
    assert model_state_fingerprint(model) != original_fingerprint


def test_historical_optional_contract_payload_must_also_match(tmp_path, synthetic_payload):
    payload = dict(synthetic_payload)
    payload["schema_version"] = checkpoint.CORRECTED_TRAINING_CHECKPOINT_SCHEMA
    payload["model_contract"] = deepcopy(payload["model_contract"])
    payload["model_contract"]["active_head_count"] = 19
    path, expected = _save(tmp_path, payload)
    with pytest.raises(checkpoint.FrozenCheckpointError) as error:
        checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.model_contract_mismatch"


@pytest.mark.parametrize("schema", [
    checkpoint.MODEL_ONLY_CHECKPOINT_SCHEMA,
    checkpoint.CORRECTED_TRAINING_CHECKPOINT_SCHEMA,
])
@pytest.mark.parametrize("key_path,value", [
    (("active_head_count",), 18.0),
    (("decoder", "layers"), True),
    (("config", "residual"), 1),
])
@pytest.mark.parametrize("recompute_embedded_fingerprint", [False, True])
def test_model_contract_rejects_python_equal_numeric_and_boolean_drift(
    tmp_path, synthetic_payload, schema, key_path, value, recompute_embedded_fingerprint
):
    payload = dict(synthetic_payload)
    payload["schema_version"] = schema
    contract = deepcopy(payload["model_contract"])
    payload["model_contract"] = contract
    target = contract
    for key in key_path[:-1]:
        target = target[key]
    assert target[key_path[-1]] == value
    assert type(target[key_path[-1]]) is not type(value)
    target[key_path[-1]] = value
    if recompute_embedded_fingerprint:
        contract["fingerprint"] = fingerprint(
            {key: value for key, value in contract.items() if key != "fingerprint"}
        )
    path, expected = _save(tmp_path, payload)
    before = _rng_snapshot()
    with pytest.raises(checkpoint.FrozenCheckpointError) as error:
        checkpoint.load_model_checkpoint(path, **expected)
    assert error.value.category == "checkpoint.model_contract_mismatch"
    assert (
        "complete canonical corrected contract differs"
        if recompute_embedded_fingerprint
        else "embedded model contract fingerprint does not match payload"
    ) in error.value.message
    _assert_rng_same(before)


def test_caller_default_dtype_does_not_change_loaded_architecture(tmp_path, synthetic_payload):
    path, expected = _save(tmp_path, synthetic_payload)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        loaded = checkpoint.load_model_checkpoint(path, **expected)
        assert torch.get_default_dtype() == torch.float64
        assert all(parameter.dtype == torch.float32 for parameter in loaded.model.parameters())
    finally:
        torch.set_default_dtype(original_dtype)
