"""Verify reuse of historical CUDA mechanics without relabeling its execution.

The caller verifies canonical terminal record/report artifacts and the whole
current source manifest. This narrower check binds the historical source
manifest to that record and proves the historical gate code is unchanged,
apart from one uncalled, additive runtime-preparation function. It performs no
model loading, runtime initialization, data preparation, or experiment import.
"""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re

from .contracts import fingerprint


_PACKAGE = "src/music_critic/experiments/analysisgnn/"
_RESUME = _PACKAGE + "harmonic_fusion_resume.py"
_ADDITION = "prepare_training_resume_runtime"
_REPORT_MODULES = ("harmonic_fusion_run.py", "harmonic_fusion_training.py", "harmonic_fusion_resume.py")
_NEW_SOURCE_FILES = frozenset(_PACKAGE + "harmonic_fusion_" + name + ".py" for name in (
    "data", "fresh_process", "pilot", "pilot_evaluation", "prepare", "reference", "gate_reuse", "dispatch"))
_REQUIRED = frozenset({"pyproject.toml", "scripts/run_harmonic_fusion_pilot.py",
    "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json", *(_PACKAGE + name for name in (
        "contracts.py", "corrected_model.py", "corrected_training.py", "full_orbit_training.py",
        "performance_replay.py", "harmonic_logit_fusion.py", "harmonic_fusion_checkpoint.py",
        "harmonic_fusion_evaluation.py", *_REPORT_MODULES))})


def _fail(reason):
    raise ValueError("fusion_gate_reuse." + reason)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _regular_bytes(path, row):
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        _fail("regular_source_file_required")
    content = path.read_bytes()
    if len(content) != row["size_bytes"] or sha256(content).hexdigest() != row["sha256"]:
        _fail("source_bytes_mismatch")
    return content


def _manifest_rows(manifest):
    if (type(manifest) is not dict or set(manifest) != {"files", "own_patch_sha256"}
            or not _sha(manifest["own_patch_sha256"]) or type(manifest["files"]) is not list):
        _fail("historical_manifest_invalid")
    rows = {}
    for row in manifest["files"]:
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            _fail("manifest_row_invalid")
        name = row["path"]
        if (type(name) is not str or not name or name == "." or "\\" in name or "\x00" in name
                or PurePosixPath(name).is_absolute() or ":" in name
                or ".." in PurePosixPath(name).parts or PurePosixPath(name).as_posix() != name
                or name in rows or not _sha(row["sha256"])
                or type(row["size_bytes"]) is not int or row["size_bytes"] < 0):
            _fail("manifest_row_invalid")
        rows[name] = row
    if not _REQUIRED <= rows.keys():
        _fail("historical_runtime_closure_incomplete")
    return rows


def _resume_projection(historical, current):
    try:
        old_tree, new_tree = ast.parse(historical), ast.parse(current)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError("fusion_gate_reuse.resume_source_invalid") from exc
    additions = [node for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == _ADDITION]
    if len(additions) != 1 or not isinstance(additions[0], ast.FunctionDef):
        _fail("single_runtime_helper_required")
    addition = additions[0]
    args = addition.args
    if (addition.decorator_list or args.posonlyargs or args.vararg is not None or args.kwarg is not None
            or [arg.arg for arg in args.args] != ["device"] or args.defaults
            or [arg.arg for arg in args.kwonlyargs] != ["expected_cuda_device_count"]
            or len(args.kw_defaults) != 1 or not isinstance(args.kw_defaults[0], ast.Constant)
            or args.kw_defaults[0].value is not None):
        _fail("runtime_helper_import_side_effect")
    if not any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names) for node in old_tree.body):
        _fail("deferred_annotations_required")
    new_tree.body.remove(addition)
    exports = [node for node in new_tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)]
    if len(exports) != 1 or not isinstance(exports[0].value, (ast.List, ast.Tuple)):
        _fail("single_runtime_export_required")
    entries = exports[0].value.elts
    added_exports = [node for node in entries if isinstance(node, ast.Constant) and node.value == _ADDITION]
    if len(added_exports) != 1:
        _fail("single_runtime_export_required")
    entries.remove(added_exports[0])
    old_dump, new_dump = (ast.dump(tree, include_attributes=False) for tree in (old_tree, new_tree))
    if old_dump != new_dump:
        _fail("historical_resume_semantics_changed")
    return sha256(old_dump.encode()).hexdigest()


def verify_historical_gate_source(*, record, report, manifest_pin, repository_root,
                                  current_source_manifest_sha256):
    """Return a source compatibility attestation, never a new CUDA run claim.

    ``manifest_pin`` names the historical deployment manifest. Its adjacent
    ``src/.../harmonic_fusion_resume.py`` supplies the only historical source
    bytes needed; other historical files need not be packaged beside it.
    """
    from .harmonic_fusion_pilot import checked_json

    repository = record.get("repository", {})
    if (not _sha(current_source_manifest_sha256)
            or type(manifest_pin) is not dict
            or manifest_pin.get("sha256") != repository.get("final_reviewed_source_manifest_sha256")):
        _fail("historical_manifest_record_binding_mismatch")
    manifest = checked_json(manifest_pin)
    rows = _manifest_rows(manifest)
    if manifest["own_patch_sha256"] != repository.get("execution_diff_sha256"):
        _fail("historical_patch_record_binding_mismatch")
    expected_report_sources = {name: rows[_PACKAGE + name]["sha256"] for name in _REPORT_MODULES}
    if report.get("bindings", {}).get("source_code") != expected_report_sources:
        _fail("historical_report_source_binding_mismatch")
    root = Path(repository_root).absolute()
    historical_sources = {name for name in rows if name.startswith("src/")}
    if any(not name.endswith(".py") for name in historical_sources):
        _fail("historical_source_type_invalid")
    current_sources = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py")}
    if (historical_sources - current_sources
            or current_sources - historical_sources - _NEW_SOURCE_FILES
            or historical_sources & _NEW_SOURCE_FILES):
        _fail("historical_runtime_closure_incomplete")
    selected = sorted(historical_sources | _REQUIRED)
    unchanged = []
    old_resume = None
    current_resume = None
    for name in selected:
        path = root / name
        if name == _RESUME:
            old_resume = _regular_bytes(Path(manifest_pin["path"]).parent / name, rows[name])
            if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
                _fail("regular_source_file_required")
            current_resume = path.read_bytes()
        else:
            content = _regular_bytes(path, rows[name])
            if name.endswith(".py") and _ADDITION.encode() in content:
                _fail("historical_code_references_new_helper")
            if name.endswith(".py"):
                # __all__ gains one name; old wildcard imports could observe it.
                for node in ast.walk(ast.parse(content)):
                    if (isinstance(node, ast.ImportFrom) and node.module
                            and node.module.endswith("harmonic_fusion_resume")
                            and any(alias.name == "*" for alias in node.names)):
                        _fail("historical_resume_wildcard_import")
            unchanged.append(name)
    if old_resume == current_resume:
        projection = None
        addition_policy = "historical_resume_byte_identical"
    else:
        projection = _resume_projection(old_resume, current_resume)
        addition_policy = "uncalled_runtime_preparation_only_exact_historical_ast_projection"
    payload = {"schema_version": "HarmonicFusionHistoricalGateSourceCompatibility@1.0.0",
        "historical_experiment_id": record["experiment_id"],
        "historical_record_fingerprint": record["record_fingerprint"],
        "historical_report_fingerprint": report["fingerprint"],
        "historical_source_manifest_sha256": manifest_pin["sha256"],
        "historical_patch_sha256": manifest["own_patch_sha256"],
        "current_source_manifest_sha256": current_source_manifest_sha256,
        "historical_source_files": len(historical_sources), "runtime_closure_files": len(selected),
        "byte_identical_runtime_files": len(unchanged) + int(old_resume == current_resume),
        "runtime_closure_fingerprint": fingerprint([rows[name] for name in selected]),
        "historical_resume_sha256": rows[_RESUME]["sha256"],
        "current_resume_sha256": sha256(current_resume).hexdigest(),
        "historical_resume_ast_fingerprint": projection, "addition_policy": addition_policy,
        "new_source_paths": sorted(current_sources - historical_sources),
        "compatible_historical_synthetic_mechanics": True,
        "current_source_cuda_execution_claim": False, "fresh_process_resume_claim": False,
        "real_train_reference_claim": False, "checkpoint_tensor_reads": 0,
        "policy": "reuse_reported_historical_mechanics_only_current_real_reference_required"}
    return {**payload, "fingerprint": fingerprint(payload)}


__all__ = ["verify_historical_gate_source"]
