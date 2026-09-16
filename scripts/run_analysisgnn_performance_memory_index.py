#!/usr/bin/env python3
"""Register and stream a frozen TRAIN size index; no GPU/model/update execution.

Production reads only existing warm canonical/sidecar entries after the frozen
B3 membership gate. This is a data audit, not evidence of GPU memory capacity.
Synthetic source injection exists only in the Python test API, not the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import gc
from hashlib import sha256
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Callable, Protocol

import torch
import torch_geometric

from music_critic.data import SCHEMA_VERSION
from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn import performance_data as pd
from music_critic.experiments.analysisgnn.performance_evidence import (
    _artifact_rows, _git, _now, _transition, _validated_parent_ids, source_manifest,
)
from music_critic.experiments.analysisgnn.performance_memory import (
    MEMORY_GATE_LIMITS, PERFORMANCE_MEMORY_SCHEMA, TrainRecordMetadata,
    build_size_index_row, select_extreme_b2_cases,
)
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, file_sha256, render_registry,
)


INDEX_RUN_SCHEMA = "AnalysisGNNPerformanceMemoryIndexRun@1.0.0"
PRODUCTION_TRAIN_RECORDS = 1295


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"performance_memory_index.{message}")


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(_jsonable(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class IndexSource(Protocol):
    """In-memory protocol for explicit synthetic tests and the production adapter."""

    def prebind(self) -> dict: ...
    def read(self, record_id: str) -> tuple[pd.PreparedRecord, TrainRecordMetadata, bytes]: ...
    def finalize(self, initial: dict) -> dict: ...


class ProductionIndexSource:
    def __init__(self, paths: ProductionArtifactPaths, *, split: str = "train") -> None:
        _require(split == "train", "train_only")  # Before loader or metadata I/O.
        self.loader = pd.FrozenProductionRecordLoader(paths=paths, split="train")
        # Only the concrete, source-bound implementation is accepted here.
        _require(type(self.loader) is pd.FrozenProductionRecordLoader, "concrete_loader_required")
        payload, split_binding = pd._read_stable(self.loader._split.path)
        _require(split_binding.digest == self.loader._split.digest, "split_hash_mismatch")
        assignments = tuple(json.loads(line) for line in payload.splitlines() if line.strip())
        self.components = {row["record_id"]: row["source_component_id"]
                           for row in assignments if row["split"] == "train"}
        _require(len(self.components) == PRODUCTION_TRAIN_RECORDS
                 and set(self.components) == set(self.loader._train_ids), "train_membership_mismatch")
        self._initial = None

    def prebind(self) -> dict:
        # Read/hash warm bytes for the complete TRAIN assignment before building
        # any graph. Missing entries fail, never trigger conversion/cache writes.
        for record_id in sorted(self.components):
            self.loader.check_binding(record_id, split="train")
        bindings = self.loader.validate_all_bindings()
        _require(set(bindings["record_bindings"]) == set(self.components), "prebind_membership_mismatch")
        self._initial = {
            "record_ids": sorted(self.components), "components": dict(sorted(self.components.items())),
            "bindings": bindings,
        }
        return _jsonable(self._initial)

    def read(self, record_id: str) -> tuple[pd.PreparedRecord, TrainRecordMetadata, bytes]:
        _require(self._initial is not None, "prebind_required")
        _require(record_id in self.components, "record_not_train")
        # _load is the existing get() implementation plus its original source
        # bytes. It preserves CanonicalPiece.split=None and every frozen check.
        prepared, sidecar_bytes = self.loader._load(record_id, split="train")
        _require(prepared.binding_seal == self._initial["bindings"]["record_bindings"][record_id],
                 "record_binding_changed")
        canonical, sidecar, metadata_file = self.loader._records[record_id]
        metadata = TrainRecordMetadata(
            record_id=record_id, source_component_id=self.components[record_id], split="train",
            canonical_sha256=canonical.digest, sidecar_sha256=sidecar.digest,
            metadata_sha256=metadata_file.digest, binding_seal=prepared.binding_seal,
            source_seal=self.loader.source_seal, code_seal=self.loader.code_seal,
            split_sha256=self.loader._split.digest,
            source_versions=(("canonical_schema", SCHEMA_VERSION),
                             ("memory_index", PERFORMANCE_MEMORY_SCHEMA),
                             ("performance_data", pd.PERFORMANCE_DATA_SCHEMA)),
        )
        return prepared, metadata, sidecar_bytes

    def finalize(self, initial: dict) -> dict:
        _require(self._initial == initial, "initial_binding_mismatch")
        final = self.loader.validate_all_bindings(expected=initial["bindings"])
        return {**initial, "bindings": final}


def _environment() -> dict:
    # Build-version text is safe to inspect; no CUDA availability/device calls.
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "torch": str(torch.__version__), "pyg": str(torch_geometric.__version__),
            "torch_cuda_build": torch.version.cuda, "cpu_count": os.cpu_count(),
            "installed_packages": sorted((item.metadata.get("Name", "unknown"), item.version)
                                         for item in distributions()),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED")}}


def _stream_index(source: IndexSource, output: Path, *, expected_count: int) -> dict:
    rows = []  # Compact scalar index only; no retained PreparedRecord/graph/sidecar.
    with (output / "size-index.jsonl").open("xb") as stream:
        initial = _jsonable(source.prebind())
        ids = initial.get("record_ids")
        _require(isinstance(ids, list) and len(ids) == expected_count
                 and all(isinstance(item, str) and item.strip() for item in ids)
                 and len(set(ids)) == len(ids), "complete_train_membership_required")
        ids = sorted(ids)
        initial_bytes = canonical_json_bytes(initial)
        _write_json(output / "source-bindings-initial.json", initial)
        for record_id in ids:
            prepared = metadata = sidecar_bytes = None
            try:
                prepared, metadata, sidecar_bytes = source.read(record_id)
                _require(metadata.record_id == record_id, "returned_record_mismatch")
                row = build_size_index_row(
                    prepared, metadata=metadata, cache_representation="source_json_bytes",
                    compact_sidecar_bytes=sidecar_bytes,
                )
                stream.write(canonical_json_bytes(_jsonable(row.to_dict())) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
            finally:
                del prepared, metadata, sidecar_bytes
                gc.collect()  # Release graph cycles between records, outside any timed training.
        final = _jsonable(source.finalize(_jsonable(initial)))
        _write_json(output / "source-bindings-final.json", final)
        _require(canonical_json_bytes(final) == initial_bytes, "source_bindings_changed")
    selection = select_extreme_b2_cases(rows, expected_train_ids=ids)
    _write_json(output / "extreme-b2-cases.json", asdict(selection))
    return {
        "schema_version": INDEX_RUN_SCHEMA, "indexed_records": len(rows),
        "indexed_components": selection.indexed_component_count,
        "size_index_sha256": file_sha256(output / "size-index.jsonl"),
        "supplied_index_fingerprint": selection.supplied_index_fingerprint,
        "membership_fingerprint": selection.expected_train_ids_fingerprint,
        "source_bindings_fingerprint": sha256(canonical_json_bytes(final)).hexdigest(),
        "selected_record_ids": [row.record_id for row in selection.selected_rows],
        "ordered_b2_case_count": len(selection.ordered_b2_cases),
        "extreme_cases_sha256": file_sha256(output / "extreme-b2-cases.json"),
        "limits": list(MEMORY_GATE_LIMITS),
    }


def run_registered_memory_index(
    *, experiment_id: str, output_root: Path, parent_experiment_id: str,
    paths: ProductionArtifactPaths | None = None, split: str = "train",
    repository_root: Path | None = None, command: list[str] | None = None,
    synthetic: bool = False, synthetic_source_factory: Callable[[], IndexSource] | None = None,
    synthetic_expected_count: int | None = None,
) -> dict:
    """Plan before loader startup, preserve partial failures, never resume/overwrite.

    Synthetic injection is explicit and cannot be mixed with production paths.
    No I/O-audit attestation is invented: data-access observations remain unknown
    even on completion, alongside a closed policy and sealed guarded source.
    """
    _require(split == "train", "train_only")
    _require(isinstance(experiment_id, str)
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id) is not None,
             "invalid_experiment_id")
    if synthetic:
        _require(paths is None and synthetic_source_factory is not None
                 and type(synthetic_expected_count) is int and synthetic_expected_count > 0,
                 "explicit_synthetic_source_required")
        expected_count = synthetic_expected_count
    else:
        _require(paths is not None and synthetic_source_factory is None and synthetic_expected_count is None,
                 "production_paths_required")
        expected_count = PRODUCTION_TRAIN_RECORDS
    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    output = Path(output_root).resolve()
    records = root / "docs/experiments/records"
    record_path = records / f"{experiment_id}.json"
    _require(not record_path.exists() and not record_path.is_symlink() and not output.exists(),
             "fresh_experiment_and_output_required")
    _require(output != root and not (root in output.parents and not (
        (root / "outputs") in output.parents or (root / "artifacts/experiments") in output.parents)),
             "output_must_be_ignored_or_external")
    _require(parent_experiment_id is not None, "terminal_parent_required")
    parents = _validated_parent_ids(records, experiment_id, parent_experiment_id)
    manifest, environment = source_manifest(root), _jsonable(_environment())
    config = {"schema_version": INDEX_RUN_SCHEMA, "mode": "synthetic" if synthetic else "production",
              "split": split, "expected_train_records": expected_count,
              "cache_representation": "source_json_bytes", "prebind_complete_train": True,
              "paths": None if paths is None else {field.name: str(getattr(paths, field.name))
                                                   for field in fields(ProductionArtifactPaths)}}
    protocol = {key: value for key, value in config.items() if key != "paths"}
    protocol.update(config_fingerprint=sha256(canonical_json_bytes(config)).hexdigest(),
                    environment_fingerprint=sha256(canonical_json_bytes(environment)).hexdigest())
    policy = {name: False for name in ("validation_allowed", "test_inputs_allowed", "test_targets_allowed",
                                      "test_metrics_allowed", "test_selection_allowed")}
    policy["source"] = "docs/PERFORMANCE_REPLAY.md; TRAIN-only memory index CLI"
    unknown_access = {key: None for key in DATA_ACCESS_FIELDS}
    state = {"git_commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current"),
             "dirty": bool(_git(root, "status", "--porcelain")),
             "diff_sha256": sha256(_git(root, "diff", "--binary", "HEAD").encode()).hexdigest(),
             "source_manifest_fingerprint": manifest["fingerprint"]}
    record = {
        "schema_version": "1.0.0", "experiment_id": experiment_id, "parent_ids": parents,
        "kind": "data_audit", "phase": "9E-P1", "status": "planned",
        "title": "Frozen TRAIN graph size index for a future B2 memory gate",
        "question": "What are the observed TRAIN graph/cache sizes and deterministic extreme B2 cases?",
        "hypothesis": "A complete sealed index identifies observed size extrema without reading excluded payloads.",
        "decision_use": "Select finite cases for a separate registered GPU memory gate; no capacity acceptance.",
        "repository": state, "seeds": [], "protocol": protocol,
        "budget": {"planned": expected_count, "actual": None, "unit": "TRAIN_record_size_rows"},
        "data_access": {**unknown_access, "declared_policy": policy,
                        "access_evidence_status": "pending_primary_evidence"},
        "results": {}, "validity": {"valid": None},
        "claims": {"supported": [], "not_supported": ["GPU memory capacity", "universal OOM safety",
                    "prefetch reservation safety", "B4/B8 readiness", "musical quality"],
                   "caveats": list(MEMORY_GATE_LIMITS)},
        "artifacts": [], "import_provenance": {"mode": "synthetic_test" if synthetic else "local_execution"},
        "created_at": _now(), "started_at": None,
    }
    lifecycle = [_transition(record_path, record)]  # Durable planned before loader factory or corpus I/O.
    owned_output = False
    error = None
    result = {}
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        owned_output = True
        record.update(status="running", started_at=_now())
        lifecycle.append(_transition(record_path, record))
        _write_json(output / "lifecycle-start.json", lifecycle)
        _write_json(output / "resolved-config.json", config)
        _write_json(output / "environment.json", environment)
        _write_json(output / "source-manifest-initial.json", manifest)
        source = (synthetic_source_factory() if synthetic else ProductionIndexSource(paths, split=split))
        result = _stream_index(source, output, expected_count=expected_count)
        final_manifest = source_manifest(root)
        _write_json(output / "source-manifest-final.json", final_manifest)
        _require(final_manifest == manifest, "source_changed_during_execution")
        result.update(experiment_id=experiment_id, mode=protocol["mode"],
                      config_fingerprint=protocol["config_fingerprint"],
                      environment_fingerprint=protocol["environment_fingerprint"])
        _write_json(output / "summary.json", result)
        record.update(status="completed", validity={"valid": True}, results=result)
        record["budget"]["actual"] = result["indexed_records"]
        record["claims"]["supported"] = ["Complete supplied TRAIN size index and deterministic observed extreme cases."]
    except BaseException as exc:
        error = exc
        record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      validity={"valid": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        # Retain even startup/prebind failures. Do not inspect a raced foreign output.
        try:
            if owned_output:
                partial = output / "size-index.jsonl"
                if partial.is_file():
                    with partial.open("rb") as stream:
                        complete_rows = sum(1 for line in stream if line.endswith(b"\n") and line.strip())
                    record["budget"]["actual"] = complete_rows
                _write_json(output / "execution-evidence.json", {
                    "protocol": protocol, "repository": state, "lifecycle": lifecycle,
                    "argv": list(command if command is not None else sys.argv),
                    "failure": None if error is None else repr(error),
                    "access_observation_limits": "No independent I/O audit; observations remain unknown. Split metadata is permitted.",
                })
                record["artifacts"] = _artifact_rows(root, output, experiment_id)
        except BaseException as exc:
            record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          validity={"valid": False, "reason": f"evidence_finalization_failed:{type(exc).__name__}: {exc}"})
            record["artifacts"] = []
            record["claims"]["supported"] = []
            if error is None:
                error = exc
        record["ended_at"] = _now()
        record["budget"]["wall_seconds"] = time.perf_counter() - started
        _transition(record_path, record)
        (root / "docs/EXPERIMENT_LEDGER.md").write_text(render_registry(records), encoding="utf-8")
    if error is not None:
        raise error
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    for field in fields(ProductionArtifactPaths):
        parser.add_argument("--" + field.name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args(argv)
    paths = ProductionArtifactPaths(**{field.name: getattr(args, field.name) for field in fields(ProductionArtifactPaths)})
    result = run_registered_memory_index(
        experiment_id=args.experiment_id, parent_experiment_id=args.parent_experiment_id,
        output_root=args.output_root, paths=paths, split=args.split,
        command=[sys.executable, __file__, *(sys.argv[1:] if argv is None else argv)],
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
