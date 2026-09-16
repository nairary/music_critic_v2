"""Metadata adapter and CLI mechanics on synthetic pins, never scientific evidence."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music_critic.experiments.analysisgnn import harmonic_fusion_data as frozen_data
from music_critic.experiments.analysisgnn import harmonic_fusion_prepare as prepare
from music_critic.experiments.analysisgnn.contracts import fingerprint
from tests.experiments.test_harmonic_fusion_data import source  # Real metadata-validator fixture; all bytes synthetic.


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/prepare_harmonic_fusion_inputs.py"


def _file(path, content):
    payload = content if isinstance(content, bytes) else json.dumps(content, ensure_ascii=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _pin(name):
    return {"path": "/not-opened/synthetic/" + name, "sha256": "a" * 64, "size_bytes": 17}


def _row(rid="synthetic:train", split="train"):
    return {"record_id": rid, "split": split, "source_component_id": "synthetic-component:" + split,
            "raw": _pin(rid + "/piece.json"), "sidecar": _pin(rid + "/sidecar.json"),
            "metadata": _pin(rid + "/metadata.json")}


def _inputs(tmp_path, rows=None, metadata=None):
    rows = [_row("an:test:synthetic", "train"), _row("synthetic:validation", "validation")] if rows is None else rows
    index = rows if isinstance(rows, bytes) else b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    metadata = {key: _pin(key) for key in prepare._METADATA_KEYS} if metadata is None else metadata
    return {"cache_index": _file(tmp_path / "cache-index.jsonl", index),
            "metadata_pins": _file(tmp_path / "metadata-pins.json", metadata), "output": tmp_path / "result.json"}


@pytest.fixture
def synthetic_validator(monkeypatch):
    """Only adapter wiring is mocked; never labels synthetic pins production-ready."""
    calls = []
    class Validator:
        def __init__(self, manifest):
            calls.append(deepcopy(manifest))
            self.bindings = {"synthetic_metadata_validator_only": True}
            self.manifest_fingerprint = fingerprint(manifest)
            self.validation_record_ids = tuple(row["record_id"] for row in manifest["records"] if row["split"] == "validation")
        def reader(self, *_args, **_kwargs):
            pytest.fail("preparation must never call a payload reader")
        def verify_bindings(self):
            pytest.fail("preparation must never hash warm payloads")
    monkeypatch.setattr(prepare, "FrozenFusionData", Validator)
    return calls


def _cli_main():
    spec = importlib.util.spec_from_file_location("fusion_prepare_cli_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _argv(inputs):
    args = ["data-manifest"]
    for role in ("cache_index", "metadata_pins"):
        flag = "--" + role.replace("_", "-")
        pin = inputs[role]
        args.extend([flag, pin["path"], flag + "-sha256", pin["sha256"], flag + "-size-bytes", str(pin["size_bytes"])])
    return [*args, "--output", str(inputs["output"])]


def test_raw_to_piece_mapping_is_lossless_and_output_is_canonical_once(tmp_path, synthetic_validator):
    inputs = _inputs(tmp_path)
    report = prepare.prepare_data_manifest(**inputs)
    payload = inputs["output"].read_bytes()
    manifest = json.loads(payload)
    original = [json.loads(row) for row in Path(inputs["cache_index"]["path"]).read_bytes().splitlines()]
    expected = [{("piece" if key == "raw" else key): value for key, value in row.items()} for row in original]
    assert manifest["records"] == sorted(expected, key=lambda row: row["record_id"])
    assert manifest["schema_version"] == frozen_data.DATA_SCHEMA
    assert payload == json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    assert report["output"] == {"path": str(inputs["output"]), "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
    assert report["split_counts"] == {"train": 1, "validation": 1}
    assert report["warm_payloads_verified"] is False and report["scientific_launch_ready"] is False
    assert report["warm_payload_files_opened"] == report["graph_builds"] == report["checkpoint_files_opened"] == 0
    assert len(synthetic_validator) == 1
    with pytest.raises(ValueError, match="output_already_exists"):
        prepare.prepare_data_manifest(**inputs)
    assert inputs["output"].read_bytes() == payload


def test_actual_cli_validates_synthetic_frozen_metadata_without_payload_access(source, tmp_path, monkeypatch, capsys):
    rows = [{("raw" if key == "piece" else key): value for key, value in row.items()} for row in source["records"]]
    inputs = _inputs(tmp_path / "cli", rows=rows, metadata={key: source[key] for key in prepare._METADATA_KEYS})
    original = frozen_data.read_bound_bytes
    def no_payload(binding):
        assert "/exports/" not in binding["path"]
        return original(binding)
    monkeypatch.setattr(frozen_data, "read_bound_bytes", no_payload)
    monkeypatch.setattr(frozen_data, "loads_piece", lambda *_: pytest.fail("raw decode forbidden"))
    monkeypatch.setattr(frozen_data.FrozenFusionData, "verify_bindings", lambda *_: pytest.fail("payload hashes forbidden"))
    assert _cli_main()(_argv(inputs)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metadata_validated"] is True
    assert report["declared_records"] == 163 and report["validation_records"] == 162
    assert report["split_counts"] == {"train": 1, "validation": 162}
    assert report["data_bindings"]["class_weight_fingerprint"] == frozen_data.FROZEN_CLASS_WEIGHT_FINGERPRINT
    emitted = json.loads(inputs["output"].read_bytes())
    assert frozen_data.FrozenFusionData(emitted).manifest_fingerprint == report["data_manifest_fingerprint"]


@pytest.mark.parametrize("rows", [[_row(split="test")], [_row(), _row()], b"\n  \n",
    [{**_row(), "unexpected": "field"}], [{**_row(), "record_id": 4}], [None],
    b'{"split":"train","split":"test"}\n', b'{"split":"train","record_id":NaN}\n'])
def test_bad_or_test_index_rejected_before_metadata_file_io(tmp_path, rows, monkeypatch, synthetic_validator):
    inputs = _inputs(tmp_path, rows=rows)
    original = prepare.read_bound_bytes
    calls = []
    def read(binding):
        assert binding["path"] == inputs["cache_index"]["path"]
        calls.append(binding)
        return original(binding)
    monkeypatch.setattr(prepare, "read_bound_bytes", read)
    with pytest.raises(ValueError):
        prepare.prepare_data_manifest(**inputs)
    assert len(calls) == 1 and synthetic_validator == [] and not inputs["output"].exists()


@pytest.mark.parametrize("field,value", [("size_bytes", True), ("size_bytes", 0), ("sha256", "bad"),
                                       ("path", "relative"), ("path", "/tmp/../escape")])
def test_invalid_record_pin_rejected_before_metadata_read(tmp_path, field, value, monkeypatch):
    row = _row(); row["raw"][field] = value
    inputs = _inputs(tmp_path, rows=[row])
    original = prepare.read_bound_bytes
    def read(binding):
        assert binding["path"] == inputs["cache_index"]["path"]
        return original(binding)
    monkeypatch.setattr(prepare, "read_bound_bytes", read)
    with pytest.raises(ValueError):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("role", ["cache_index", "metadata_pins"])
@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size_bytes", 1)])
def test_compact_input_bytes_must_match_caller_pins(tmp_path, role, field, value, synthetic_validator):
    inputs = _inputs(tmp_path)
    inputs[role][field] = value
    with pytest.raises(ValueError, match="artifact_.*mismatch"):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists() and synthetic_validator == []


@pytest.mark.parametrize("mutation", ["missing_tuple", "extra", "bad_pin", "duplicate", "nan", "not_object"])
def test_exact_four_metadata_pins_required_without_defaults(tmp_path, mutation, synthetic_validator):
    metadata = {key: _pin(key) for key in prepare._METADATA_KEYS}
    if mutation == "missing_tuple":
        metadata.pop("joint_tuple_counts")
    elif mutation == "extra":
        metadata["guessed_tuple_directory"] = "/not-allowed"
    elif mutation == "bad_pin":
        metadata["joint_tuple_counts"] = None
    elif mutation == "duplicate":
        metadata = b'{"joint_tuple_counts":1,"joint_tuple_counts":2}'
    elif mutation == "nan":
        metadata = b'{"joint_tuple_counts":NaN}'
    else:
        metadata = []
    inputs = _inputs(tmp_path, metadata=metadata)
    with pytest.raises(ValueError):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists() and synthetic_validator == []


def test_frozen_validator_failure_cannot_publish_manifest(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    def reject(_):
        raise ValueError("synthetic frozen contract rejection")
    monkeypatch.setattr(prepare, "FrozenFusionData", reject)
    with pytest.raises(ValueError, match="frozen contract"):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists()


def test_input_drift_during_metadata_validation_prevents_publication(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    def drift(_):
        Path(inputs["cache_index"]["path"]).write_text("changed after initial hash")
        return object()
    monkeypatch.setattr(prepare, "FrozenFusionData", drift)
    with pytest.raises(ValueError, match="artifact_.*mismatch"):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists()


def test_failed_fsync_leaves_no_output_or_partial_file(tmp_path, monkeypatch, synthetic_validator):
    inputs = _inputs(tmp_path)
    def fail(_):
        raise OSError("synthetic fsync failure")
    monkeypatch.setattr(prepare.os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failure"):
        prepare.prepare_data_manifest(**inputs)
    assert not inputs["output"].exists()
    assert not list(tmp_path.glob(".fusion-manifest-*.partial"))


def test_competing_output_cannot_be_overwritten(tmp_path, monkeypatch, synthetic_validator):
    inputs = _inputs(tmp_path)
    original = prepare.os.link
    def race(source, destination):
        Path(destination).write_bytes(b"other writer")
        return original(source, destination)
    monkeypatch.setattr(prepare.os, "link", race)
    with pytest.raises(FileExistsError):
        prepare.prepare_data_manifest(**inputs)
    assert inputs["output"].read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".fusion-manifest-*.partial"))


def test_output_parent_symlink_rejected(tmp_path, synthetic_validator):
    inputs = _inputs(tmp_path)
    link = tmp_path / "parent-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    inputs["output"] = link / "new.json"
    with pytest.raises(ValueError, match="regular_output_path_required"):
        prepare.prepare_data_manifest(**inputs)
    assert synthetic_validator == []


def test_cli_help_and_test_rejection_in_fresh_process(tmp_path):
    help_result = subprocess.run([sys.executable, str(CLI), "data-manifest", "--help"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and "--metadata-pins-sha256" in help_result.stdout
    inputs = _inputs(tmp_path, rows=[_row(split="test")])
    Path(inputs["metadata_pins"]["path"]).unlink()
    result = subprocess.run([sys.executable, str(CLI), *_argv(inputs)], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "cache_index_split_forbidden" in result.stderr
    assert "artifact_unavailable" not in result.stderr
    assert not inputs["output"].exists()


def test_cli_requires_all_byte_pin_arguments_before_import(tmp_path, capsys):
    with pytest.raises(SystemExit) as failure:
        _cli_main()(["data-manifest", "--cache-index", "missing", "--output", str(tmp_path / "manifest")])
    assert failure.value.code == 2 and "--metadata-pins" in capsys.readouterr().err
