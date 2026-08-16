"""Immutable artifacts, provenance, and aggregate-bundle validation."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping

import torch

from music_critic.cuda_memory import (
    CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION,
)
from music_critic.device import (
    CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION,
    resolve_cuda_device_index,
)
from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_ARTIFACT_CONTRACT_VERSION,
    Phase8B2ContractError,
    canonical_json_bytes,
    fingerprint,
)


REQUIRED_ARTIFACTS = (
    "comparison_protocol.json",
    "actual_sample_schedule.json",
    "run_manifest.json",
    "ssl_training_metrics.jsonl",
    "ssl_checkpoint_evidence.json",
    "transfer_evidence.json",
    "downstream_metrics.json",
    "piece_statistics.json",
    "validation_selection.json",
    "statistical_summary.json",
    "compute_accounting.json",
    "final_comparison_report.json",
)
OPTIONAL_ARTIFACTS = ("test_metrics.json",)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: str | Path, value: object) -> str:
    """Create one immutable JSON artifact atomically."""

    destination = Path(path)
    if destination.exists():
        raise Phase8B2ContractError(
            f"phase8b2.artifact.already_exists:{destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value, pretty=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(destination)


def write_jsonl_once(path: str | Path, rows: Iterable[object]) -> str:
    destination = Path(path)
    if destination.exists():
        raise Phase8B2ContractError(
            f"phase8b2.artifact.already_exists:{destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in rows:
                stream.write(canonical_json_bytes(row))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(destination)


def _command(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise Phase8B2ContractError(
            "phase8b2.repository.git_command_failed:"
            + arguments[0]
        )
    return process.stdout.strip()


def repository_evidence(
    repository: str | Path, *, require_clean: bool = True
) -> dict[str, object]:
    """Capture exact git identity and reject a dirty scientific run."""

    root = Path(repository).resolve()
    exact_sha = _command(root, "rev-parse", "HEAD")
    status = _command(root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    if require_clean and dirty:
        raise Phase8B2ContractError(
            "phase8b2.repository.dirty_worktree_forbidden"
        )
    return {
        "git_sha": exact_sha,
        "dirty": dirty,
        "dirty_status_fingerprint": fingerprint(status.splitlines()),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_evidence(device: torch.device) -> dict[str, object]:
    """Record environment facts without claiming unavailable CUDA evidence."""

    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_geometric": _package_version("torch-geometric"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_runtime_device_index_contract_version": (
            CUDA_RUNTIME_DEVICE_INDEX_CONTRACT_VERSION
        ),
        "cuda_memory_statistics_lifecycle_contract_version": (
            CUDA_MEMORY_STATISTICS_LIFECYCLE_CONTRACT_VERSION
        ),
        "cuda_logical_device_index": None,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise Phase8B2ContractError(
                "phase8b2.environment.cuda_evidence_unavailable"
            )
        cuda_device_index = resolve_cuda_device_index(device)
        result["cuda_logical_device_index"] = cuda_device_index
        result["cuda_device_name"] = torch.cuda.get_device_name(
            cuda_device_index
        )
    else:
        result["cuda_device_name"] = None
    return result


def manifest_payload(
    *,
    protocol_fingerprint: str,
    repository: Mapping[str, object],
    environment: Mapping[str, object],
    artifact_sha256: Mapping[str, str],
    cells: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = sorted(
        (dict(cell) for cell in cells),
        key=lambda row: str(row["cell_id"]),
    )
    identities = [row["cell_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise Phase8B2ContractError(
            "phase8b2.manifest.duplicate_cell"
        )
    return {
        "artifact_contract_version": PHASE8B2_ARTIFACT_CONTRACT_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "repository": dict(repository),
        "environment": dict(environment),
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "cells": rows,
    }


def write_complete_artifact_bundle(
    output_directory: str | Path,
    *,
    protocol_fingerprint: str,
    repository: Mapping[str, object],
    environment: Mapping[str, object],
    cells: Iterable[Mapping[str, object]],
    json_artifacts: Mapping[str, object],
    ssl_metric_rows: Iterable[object],
    allow_dirty_repository: bool = False,
) -> dict[str, object]:
    """Create a complete immutable comparison bundle and final manifest."""

    output = Path(output_directory)
    if output.exists():
        raise Phase8B2ContractError(
            "phase8b2.artifact.new_output_directory_required"
        )
    expected_json = set(REQUIRED_ARTIFACTS) - {
        "run_manifest.json",
        "ssl_training_metrics.jsonl",
    }
    supplied = set(json_artifacts)
    if supplied != expected_json and supplied != (
        expected_json | set(OPTIONAL_ARTIFACTS)
    ):
        missing = sorted(expected_json - supplied)
        unexpected = sorted(supplied - expected_json - set(OPTIONAL_ARTIFACTS))
        raise Phase8B2ContractError(
            "phase8b2.artifact.complete_payload_invalid:"
            f"missing={missing},unexpected={unexpected}"
        )
    if (
        (repository.get("dirty") is not False and not allow_dirty_repository)
        or not repository.get("git_sha")
    ):
        raise Phase8B2ContractError(
            "phase8b2.artifact.clean_git_identity_required"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    hashes: dict[str, str] = {}
    try:
        for name in sorted(json_artifacts):
            hashes[name] = write_json_once(staging / name, json_artifacts[name])
        hashes["ssl_training_metrics.jsonl"] = write_jsonl_once(
            staging / "ssl_training_metrics.jsonl", ssl_metric_rows
        )
        manifest = manifest_payload(
            protocol_fingerprint=protocol_fingerprint,
            repository=repository,
            environment=environment,
            artifact_sha256=hashes,
            cells=cells,
        )
        manifest_sha = write_json_once(staging / "run_manifest.json", manifest)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "artifact_contract_version": PHASE8B2_ARTIFACT_CONTRACT_VERSION,
        "output_directory": str(output.resolve()),
        "artifact_sha256": {**hashes, "run_manifest.json": manifest_sha},
        "manifest_fingerprint": fingerprint(manifest),
        "complete": True,
    }


def validate_aggregate_bundles(
    bundles: Iterable[Mapping[str, object]],
    *,
    allow_test_metrics: bool = False,
) -> dict[str, object]:
    """Reject incompatible, duplicate, incomplete, or stale input bundles."""

    rows = tuple(dict(bundle) for bundle in bundles)
    if not rows:
        raise Phase8B2ContractError("phase8b2.aggregate.input_empty")
    required = {
        "protocol_fingerprint",
        "comparison_mode",
        "data_binding_fingerprint",
        "initial_encoder_fingerprint",
        "seed",
        "cell_id",
        "complete",
        "artifact_fingerprint",
        "recomputed_artifact_fingerprint",
        "test_membership_metadata_resolved",
        "test_inference_performed",
        "test_targets_accessed",
        "test_metrics_accessed",
    }
    for row in rows:
        if not required <= set(row):
            raise Phase8B2ContractError(
                "phase8b2.aggregate.bundle_fields_missing"
            )
        test_fields = (
            "test_membership_metadata_resolved",
            "test_inference_performed",
            "test_targets_accessed",
            "test_metrics_accessed",
        )
        if any(not isinstance(row[field], bool) for field in test_fields):
            raise Phase8B2ContractError(
                "phase8b2.aggregate.test_access_evidence_invalid"
            )
        if row["test_membership_metadata_resolved"] is not True:
            raise Phase8B2ContractError(
                "phase8b2.aggregate.test_membership_metadata_missing"
            )
        if row["complete"] is not True:
            raise Phase8B2ContractError(
                "phase8b2.aggregate.incomplete_run"
            )
        if row["artifact_fingerprint"] != row[
            "recomputed_artifact_fingerprint"
        ]:
            raise Phase8B2ContractError(
                "phase8b2.aggregate.stale_artifact"
            )
        if (
            any(
                row[field]
                for field in test_fields[1:]
            )
            and (
                not allow_test_metrics
                or row.get("test_authorization_consumed") is not True
            )
        ):
            raise Phase8B2ContractError(
                "phase8b2.aggregate.unauthorized_test_access"
            )
    for field, category in (
        ("protocol_fingerprint", "protocol_fingerprint_mismatch"),
        ("comparison_mode", "comparison_mode_mixed"),
        ("data_binding_fingerprint", "data_binding_mismatch"),
    ):
        if len({row[field] for row in rows}) != 1:
            raise Phase8B2ContractError(
                f"phase8b2.aggregate.{category}"
            )
    initial_by_seed: dict[int, str] = {}
    for row in rows:
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise Phase8B2ContractError(
                "phase8b2.aggregate.seed_invalid"
            )
        initial = row["initial_encoder_fingerprint"]
        if seed in initial_by_seed and initial_by_seed[seed] != initial:
            raise Phase8B2ContractError(
                "phase8b2.aggregate.initial_encoder_mismatch"
            )
        initial_by_seed[seed] = str(initial)
    cell_ids = [row["cell_id"] for row in rows]
    if len(cell_ids) != len(set(cell_ids)):
        raise Phase8B2ContractError(
            "phase8b2.aggregate.duplicate_cell"
        )
    return {
        "artifact_contract_version": PHASE8B2_ARTIFACT_CONTRACT_VERSION,
        "protocol_fingerprint": rows[0]["protocol_fingerprint"],
        "comparison_mode": rows[0]["comparison_mode"],
        "data_binding_fingerprint": rows[0]["data_binding_fingerprint"],
        "initial_encoder_fingerprints_by_seed": {
            str(seed): value for seed, value in sorted(initial_by_seed.items())
        },
        "cell_count": len(rows),
        "cell_ids": sorted(str(value) for value in cell_ids),
    }


def read_json(path: str | Path) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8B2ContractError(
            f"phase8b2.artifact.unreadable:{Path(path).name}"
        ) from exc


__all__ = [
    "OPTIONAL_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
    "environment_evidence",
    "file_sha256",
    "manifest_payload",
    "read_json",
    "repository_evidence",
    "validate_aggregate_bundles",
    "write_json_once",
    "write_complete_artifact_bundle",
    "write_jsonl_once",
]
