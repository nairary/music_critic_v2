"""Source-only compatibility fixtures; no CUDA execution or checkpoint reads."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_gate_reuse as reuse
from music_critic.experiments.analysisgnn.harmonic_fusion_pilot import artifact_pin


OLD_RESUME = b'''from __future__ import annotations

def restore():
    return "unchanged"

__all__ = ["restore"]
'''
HELPER = '''def prepare_training_resume_runtime(device: str, *, expected_cuda_device_count: int | None = None) -> dict[str, object]:
    return {"device_type": device}

'''
NEW_RESUME = OLD_RESUME.decode().replace('__all__ = ["restore"]',
    HELPER + '__all__ = ["restore", "prepare_training_resume_runtime"]').encode()


def _row(name, content):
    return {"path": name, "sha256": sha256(content).hexdigest(), "size_bytes": len(content)}


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _seal(fixture):
    _write(fixture.manifest_path, json.dumps(fixture.manifest, sort_keys=True).encode())
    fixture.pin = artifact_pin(fixture.manifest_path)
    fixture.record["repository"]["final_reviewed_source_manifest_sha256"] = fixture.pin["sha256"]
    fixture.report["bindings"]["source_code"] = {
        name: next(row["sha256"] for row in fixture.manifest["files"] if row["path"] == reuse._PACKAGE + name)
        for name in reuse._REPORT_MODULES}


@pytest.fixture
def source(tmp_path, monkeypatch):
    root, old = tmp_path / "current", tmp_path / "history"
    contents = {name: b"# unchanged source fixture\n" for name in reuse._REQUIRED}
    contents[reuse._RESUME] = OLD_RESUME
    contents["src/another_historical_dependency.py"] = b"# non-core historical dependency\n"
    rows = [_row(name, content) for name, content in sorted(contents.items())]
    # Irrelevant old documents can be omitted from the deployment package;
    # their manifest rows must still be structurally unambiguous and safe.
    rows.append(_row("docs/not-packaged.md", b"historical document"))
    for name, content in contents.items():
        _write(root / name, NEW_RESUME if name == reuse._RESUME else content)
    _write(root / (reuse._PACKAGE + "harmonic_fusion_pilot.py"), b"# new dispatcher\n")
    _write(old / reuse._RESUME, OLD_RESUME)
    fixture = SimpleNamespace(root=root, historical_root=old, manifest_path=old / "deployment-manifest.json",
        manifest={"files": rows, "own_patch_sha256": "a" * 64},
        record={"experiment_id": "FIXTURE-HISTORICAL-GATE", "record_fingerprint": "b" * 64,
            "repository": {"execution_diff_sha256": "a" * 64}},
        report={"fingerprint": "c" * 64, "bindings": {"source_code": {}}})
    _seal(fixture)

    def denied(*args, **kwargs):
        pytest.fail("source compatibility must not load checkpoint tensors or initialize CUDA")

    monkeypatch.setattr(torch, "load", denied)
    monkeypatch.setattr(torch.cuda, "init", denied)
    monkeypatch.setattr(torch.cuda, "is_available", denied)
    return fixture


def _verify(fixture, **kwargs):
    arguments = {"record": fixture.record, "report": fixture.report, "manifest_pin": fixture.pin,
                 "repository_root": fixture.root, "current_source_manifest_sha256": "d" * 64}
    arguments.update(kwargs)
    return reuse.verify_historical_gate_source(**arguments)


def test_unchanged_historical_closure_with_uncalled_addition_can_be_reused(source):
    record_before, report_before = deepcopy(source.record), deepcopy(source.report)
    result = _verify(source)
    assert result["compatible_historical_synthetic_mechanics"] is True
    assert result["runtime_closure_files"] == len(reuse._REQUIRED) + 1
    assert result["byte_identical_runtime_files"] == result["runtime_closure_files"] - 1
    assert result["addition_policy"] == "uncalled_runtime_preparation_only_exact_historical_ast_projection"
    assert result["historical_resume_sha256"] == sha256(OLD_RESUME).hexdigest()
    assert result["current_resume_sha256"] == sha256(NEW_RESUME).hexdigest()
    assert result["new_source_paths"] == [reuse._PACKAGE + "harmonic_fusion_pilot.py"]
    assert result["historical_source_manifest_sha256"] == source.pin["sha256"]
    assert result["current_source_manifest_sha256"] == "d" * 64
    assert result["checkpoint_tensor_reads"] == 0
    assert all(result[key] is False for key in (
        "current_source_cuda_execution_claim", "fresh_process_resume_claim", "real_train_reference_claim"))
    assert source.record == record_before and source.report == report_before


def test_identical_resume_needs_no_projection_exception(source):
    (source.root / reuse._RESUME).write_bytes(OLD_RESUME)
    result = _verify(source)
    assert result["addition_policy"] == "historical_resume_byte_identical"
    assert result["historical_resume_ast_fingerprint"] is None
    assert result["byte_identical_runtime_files"] == result["runtime_closure_files"]


@pytest.mark.parametrize("field", ["record_manifest", "record_patch", "report_module", "current_manifest"])
def test_source_proof_is_bound_to_record_report_and_current_manifest(source, field):
    kwargs = {}
    if field == "record_manifest":
        source.record["repository"]["final_reviewed_source_manifest_sha256"] = "e" * 64
    elif field == "record_patch":
        source.record["repository"]["execution_diff_sha256"] = "e" * 64
    elif field == "report_module":
        source.report["bindings"]["source_code"]["harmonic_fusion_training.py"] = "e" * 64
    else:
        kwargs["current_source_manifest_sha256"] = None
    with pytest.raises(ValueError, match="binding_mismatch"):
        _verify(source, **kwargs)


@pytest.mark.parametrize("name", [
    "pyproject.toml", "scripts/run_harmonic_fusion_pilot.py",
    "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json",
    "src/another_historical_dependency.py", reuse._PACKAGE + "corrected_training.py",
])
def test_every_historical_runtime_dependency_is_byte_verified(source, name):
    (source.root / name).write_bytes(b"changed dependency\n")
    with pytest.raises(ValueError, match="source_bytes_mismatch"):
        _verify(source)


@pytest.mark.parametrize("mutation", ["empty", "missing_noncore", "missing_required", "unreviewed_new_source"])
def test_manifest_cannot_omit_runtime_inventory_or_admit_unreviewed_additions(source, mutation):
    if mutation == "empty":
        source.manifest["files"] = []
    elif mutation == "unreviewed_new_source":
        _write(source.root / "src/unreviewed.py", b"# omitted source\n")
    else:
        removed = "src/another_historical_dependency.py" if mutation == "missing_noncore" else "pyproject.toml"
        source.manifest["files"] = [row for row in source.manifest["files"] if row["path"] != removed]
    if mutation != "unreviewed_new_source":
        _write(source.manifest_path, json.dumps(source.manifest).encode())
        source.pin = artifact_pin(source.manifest_path)
        source.record["repository"]["final_reviewed_source_manifest_sha256"] = source.pin["sha256"]
    with pytest.raises(ValueError, match="historical_runtime_closure_incomplete"):
        _verify(source)


@pytest.mark.parametrize("mutation", ["absolute", "parent", "dot", "backslash", "duplicate", "hash", "boolsize", "negative_size"])
def test_all_manifest_rows_are_checked_even_unread_documents(source, mutation):
    row = source.manifest["files"][-1]
    if mutation == "absolute": row["path"] = "/outside.md"
    elif mutation == "parent": row["path"] = "docs/../outside.md"
    elif mutation == "dot": row["path"] = "."
    elif mutation == "backslash": row["path"] = "docs\\outside.md"
    elif mutation == "duplicate": source.manifest["files"].append(deepcopy(row))
    elif mutation == "hash": row["sha256"] = "unknown"
    elif mutation == "boolsize": row["size_bytes"] = True
    else: row["size_bytes"] = -1
    _seal(source)
    with pytest.raises(ValueError, match="manifest_row_invalid"):
        _verify(source)


@pytest.mark.parametrize("historical", [False, True])
def test_resume_source_cannot_be_symlinked_or_changed(source, historical):
    root = source.historical_root if historical else source.root
    path = root / reuse._RESUME
    copy = path.with_suffix(".copy")
    path.rename(copy)
    path.symlink_to(copy.name)
    with pytest.raises(ValueError, match="regular_source_file_required"):
        _verify(source)


def test_historical_resume_bytes_must_match_manifest(source):
    (source.historical_root / reuse._RESUME).write_bytes(OLD_RESUME + b"# mutation\n")
    with pytest.raises(ValueError, match="source_bytes_mismatch"):
        _verify(source)


@pytest.mark.parametrize("mutation,reason", [
    ("old_body", "historical_resume_semantics_changed"),
    ("top_level", "historical_resume_semantics_changed"),
    ("second_function", "historical_resume_semantics_changed"),
    ("duplicate_helper", "single_runtime_helper_required"),
    ("decorator", "runtime_helper_import_side_effect"),
    ("default", "runtime_helper_import_side_effect"),
    ("duplicate_export", "single_runtime_export_required"),
])
def test_resume_exception_is_one_inert_definition_without_other_semantic_changes(source, mutation, reason):
    text = NEW_RESUME.decode()
    if mutation == "old_body": text = text.replace('return "unchanged"', 'return "changed"')
    elif mutation == "top_level": text += '\nrestore = lambda: "changed"\n'
    elif mutation == "second_function": text += '\ndef another():\n    pass\n'
    elif mutation == "duplicate_helper": text += "\n" + HELPER
    elif mutation == "decorator": text = text.replace("def prepare_training_resume_runtime", "@restore()\ndef prepare_training_resume_runtime")
    elif mutation == "default": text = text.replace("= None) ->", "= restore()) ->")
    else: text = text.replace('["restore", "prepare_training_resume_runtime"]',
                             '["restore", "prepare_training_resume_runtime", "prepare_training_resume_runtime"]')
    (source.root / reuse._RESUME).write_text(text)
    with pytest.raises(ValueError, match=reason):
        _verify(source)


@pytest.mark.parametrize("wildcard", [False, True])
def test_even_historically_bound_imports_must_not_observe_the_addition(source, wildcard):
    name = "src/another_historical_dependency.py"
    code = (b"from music_critic.experiments.analysisgnn.harmonic_fusion_resume import *\n" if wildcard
            else b"from music_critic.experiments.analysisgnn.harmonic_fusion_resume import prepare_training_resume_runtime\n")
    (source.root / name).write_bytes(code)
    source.manifest["files"] = [_row(name, code) if row["path"] == name else row for row in source.manifest["files"]]
    _seal(source)
    with pytest.raises(ValueError, match="historical_resume_wildcard_import" if wildcard
                       else "historical_code_references_new_helper"):
        _verify(source)
