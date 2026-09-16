from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import pickle
import random
import zipfile

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, corrected_model_contract, model_state_fingerprint,
)
from music_critic.experiments.pop909_cl_frozen import checkpoint, checkpoint_export as export


def _touch(path):
    Path(path).write_text("executed")
    return None


class _Executable:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _touch, (self.path,)


@pytest.fixture(scope="module")
def envelope():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(917)
        model = CorrectedAnalysisGNNModel()
    return {
        "schema_version": checkpoint.CORRECTED_TRAINING_CHECKPOINT_SCHEMA,
        "resolved_config": {"seed": 17, "nested": {"optimizer_state": "a nested key"}, "flags": [True, None, 0.5]},
        "model_contract_fingerprint": corrected_model_contract(model)["fingerprint"],
        "model_state": model.state_dict(),
        "model_state_fingerprint": model_state_fingerprint(model),
        "optimizer_state": {},
        "rng_state": {"numpy": np.random.get_state()},
        "applied_update": 95000,
    }


def _bytes(envelope, protocol=2):
    buffer = BytesIO()
    torch.save(envelope, buffer, pickle_protocol=protocol)
    return buffer.getvalue()


def _save(tmp_path, content):
    source = tmp_path / "source.pt"
    source.write_bytes(content)
    return source, {"expected_sha256": sha256(content).hexdigest(), "expected_size_bytes": len(content)}


def _rewrite(content, transform):
    output = BytesIO()
    with zipfile.ZipFile(BytesIO(content)) as source, zipfile.ZipFile(output, "w") as target:
        for item in source.infolist():
            target.writestr(item.filename, transform(item.filename, source.read(item)))
    return output.getvalue()


def _forbidden(*args, **kwargs):
    raise AssertionError("forbidden deserializer/training/global API")


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng(before):
    after = _rng()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_numpy_training_envelope_exports_identical_strict_model_only(tmp_path, envelope, monkeypatch):
    source_bytes = _bytes(envelope)
    source, binding = _save(tmp_path, source_bytes)
    with pytest.raises(checkpoint.FrozenCheckpointError, match="restricted_load_failed"):
        checkpoint.load_model_checkpoint(source, **binding)
    real_load = torch.load
    calls = []

    def restricted(*args, **kwargs):
        calls.append(kwargs.copy())
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", restricted)
    monkeypatch.setattr(torch.serialization, "add_safe_globals", _forbidden, raising=False)
    monkeypatch.setattr(torch.serialization, "safe_globals", _forbidden, raising=False)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", _forbidden)
    monkeypatch.setattr(pickle, "load", _forbidden)
    monkeypatch.setattr(pickle, "loads", _forbidden)
    before = _rng()
    output = tmp_path / "model-only.pt"
    result = export.export_model_only_checkpoint(source, output, **binding)
    _assert_rng(before)
    assert len(calls) == 2
    assert source.read_bytes() == source_bytes
    assert result["source_sha256"] == binding["expected_sha256"]
    assert result["export_sha256"] == sha256(output.read_bytes()).hexdigest()
    assert result["selection_update_fields_extracted"] is False
    assert result["extraction"]["tail_interpreted"] is False
    assert result["extraction"]["source_model_state_fingerprint"] == envelope["model_state_fingerprint"]
    payload = real_load(output, weights_only=True, map_location="cpu")
    assert set(payload) == checkpoint._MODEL_ONLY_FIELDS
    assert payload["model_state_fingerprint"] == envelope["model_state_fingerprint"]
    for name, value in payload["model_state"].items():
        assert torch.equal(value, envelope["model_state"][name])
    reduced, evidence = export._reduce_archive(source_bytes)
    with zipfile.ZipFile(BytesIO(source_bytes)) as a, zipfile.ZipFile(BytesIO(reduced)) as b:
        assert a.namelist() == b.namelist()
        for name in evidence["copied_members"]:
            assert a.read(name) == b.read(name)


def test_postprefix_executable_is_never_interpreted_or_called(tmp_path, envelope):
    marker = tmp_path / "must-not-exist"
    source_envelope = dict(envelope, optimizer_state=_Executable(marker))
    source, binding = _save(tmp_path, _bytes(source_envelope))
    result = export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
    assert not marker.exists()
    assert result["extraction"]["prefix_globals"] == sorted(export._GLOBALS)
    assert "test_pop909" not in str(result["extraction"]["prefix_globals"])


def test_shared_flattened_storage_and_nonzero_offsets_preserve_all_weights(tmp_path, envelope):
    # CUDA GRU flattening can produce contiguous views into one larger
    # allocation. The checkpoint must preserve values, not storage ownership.
    flat = torch.cat([torch.zeros(5), *(value.reshape(-1) for value in envelope["model_state"].values())])
    state = OrderedDict()
    offset = 5
    for name, value in envelope["model_state"].items():
        state[name] = flat[offset:offset + value.numel()].view(value.shape)
        offset += value.numel()
    state._metadata = envelope["model_state"]._metadata
    assert model_state_fingerprint(state) == envelope["model_state_fingerprint"]
    source, binding = _save(tmp_path, _bytes(dict(envelope, model_state=state)))
    result = export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
    assert len(result["extraction"]["referenced_storage_sizes"]) == 1
    assert result["export_verification"]["model_state_fingerprint"] == envelope["model_state_fingerprint"]


def test_prefix_executable_rejected_before_any_torch_load(tmp_path, envelope, monkeypatch):
    marker = tmp_path / "must-not-exist"
    altered = dict(envelope, resolved_config={"bad": _Executable(marker)})
    source, binding = _save(tmp_path, _bytes(altered))
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises(checkpoint.FrozenCheckpointError, match="forbidden_global"):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
    assert not marker.exists()


@pytest.mark.parametrize("change", ["order", "missing", "duplicate", "protocol", "wrong_schema", "non_primitive_config"])
def test_noncanonical_prefix_rejected(tmp_path, envelope, monkeypatch, change):
    altered = dict(envelope)
    protocol = 2
    if change == "order":
        altered = {key: envelope[key] for key in ["resolved_config", "schema_version", *list(envelope)[2:]]}
    elif change == "missing":
        del altered["resolved_config"]
    elif change == "wrong_schema":
        altered["schema_version"] = "other"
    elif change == "non_primitive_config":
        altered["resolved_config"] = {"tensor": torch.zeros(1)}
    elif change == "duplicate":
        altered["resolved_config"] = {"alpha": 1, "omega": 2}
    elif change == "protocol":
        protocol = 4
    content = _bytes(altered, protocol)
    if change == "duplicate":
        # A nested duplicate dictionary key is invalid even if ignored by model loading.
        content = _rewrite(content, lambda name, value: value.replace(b"omega", b"alpha", 1) if name.endswith("data.pkl") else value)
    source, binding = _save(tmp_path, content)
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises(checkpoint.FrozenCheckpointError):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
    assert not (tmp_path / "export.pt").exists()


@pytest.mark.parametrize("field", ["model_contract_fingerprint", "model_state_fingerprint"])
def test_wrong_fingerprint_rejected_by_unchanged_loader(tmp_path, envelope, field):
    source, binding = _save(tmp_path, _bytes(dict(envelope, **{field: "0" * 64})))
    before = _rng()
    with pytest.raises(checkpoint.FrozenCheckpointError, match="mismatch"):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
    _assert_rng(before)
    assert not (tmp_path / "export.pt").exists()


def test_changed_storage_bytes_fail_original_state_fingerprint(tmp_path, envelope):
    changed = False

    def mutate(name, value):
        nonlocal changed
        if "/data/" in name and value and not changed:
            changed = True
            return bytes([value[0] ^ 1]) + value[1:]
        return value

    source, binding = _save(tmp_path, _rewrite(_bytes(envelope), mutate))
    with pytest.raises(checkpoint.FrozenCheckpointError, match="state_fingerprint_mismatch"):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)


def test_short_storage_rejected_statically(tmp_path, envelope, monkeypatch):
    content = _rewrite(_bytes(envelope), lambda name, value: value[:-1] if name.endswith("/data/0") else value)
    source, binding = _save(tmp_path, content)
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises(checkpoint.FrozenCheckpointError, match="storage_reference"):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)


@pytest.mark.parametrize("change", ["sha", "size", "huge", "existing"])
def test_artifact_guards_precede_deserialization(tmp_path, envelope, monkeypatch, change):
    source, binding = _save(tmp_path, _bytes(envelope))
    destination = tmp_path / "export.pt"
    if change == "sha":
        binding["expected_sha256"] = "0" * 64
    elif change == "size":
        binding["expected_size_bytes"] -= 1
    elif change == "huge":
        binding["expected_size_bytes"] = export.MAX_ARCHIVE_BYTES + 1
    else:
        destination.write_bytes(b"immutable")
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises((checkpoint.FrozenCheckpointError, FileExistsError)):
        export.export_model_only_checkpoint(source, destination, **binding)
    if change == "existing":
        assert destination.read_bytes() == b"immutable"


@pytest.mark.parametrize("bad_name", ["../outside", "/absolute/data.pkl", "archive/../data.pkl", "archive/extra.pkl"])
def test_archive_paths_and_extra_payload_rejected(bad_name, envelope):
    content = BytesIO(_bytes(envelope))
    with zipfile.ZipFile(content, "a") as archive:
        archive.writestr(bad_name, b"payload")
    with pytest.raises(checkpoint.FrozenCheckpointError):
        export._reduce_archive(content.getvalue())


def test_no_boundary_or_truncated_pickle_is_rejected(envelope):
    content = _rewrite(_bytes(envelope), lambda name, value: value[:32] if name.endswith("data.pkl") else value)
    with pytest.raises(checkpoint.FrozenCheckpointError):
        export._reduce_archive(content)


def test_primitive_shared_dag_is_bounded_and_cycles_are_rejected():
    dag = export._Container("list", [1])
    for _ in range(40):
        dag = export._Container("list", [dag, dag])
    assert export._primitive(dag)
    cycle = export._Container("list", [])
    cycle.items.append(cycle)
    assert not export._primitive(cycle)
    deep = export._Container("list", [])
    for _ in range(70):
        deep = export._Container("list", [deep])
    assert not export._primitive(deep)


def test_cyclic_source_config_rejected_without_torch_load(tmp_path, envelope, monkeypatch):
    cycle = []
    cycle.append(cycle)
    source, binding = _save(tmp_path, _bytes(dict(envelope, resolved_config={"cycle": cycle})))
    monkeypatch.setattr(torch, "load", _forbidden)
    with pytest.raises(checkpoint.FrozenCheckpointError, match="root_field_types"):
        export.export_model_only_checkpoint(source, tmp_path / "export.pt", **binding)
