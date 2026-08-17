"""Executable, source-free Phase 9B.2C RTX 3090 supervised smoke."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from hashlib import sha256
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tarfile
import tempfile
from typing import Mapping, Sequence
import weakref

import torch

from music_critic.cuda_memory import initialize_cuda_memory_statistics
from music_critic.evaluation import (
    build_dilemmadata_train_priors,
    evaluate_dilemmadata_model,
)
from music_critic.models import (
    DILEMMADATA_ACTIVE_TASK_IDS,
    DILEMMADATA_OPEN_TASK_IDS,
    DILEMMADATA_PU_TASK_IDS,
    DilemmadataHierarchicalModel,
    dilemmadata_model_contract_fingerprint,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    DilemmadataTargetCacheConfig,
    DilemmadataTargetCacheError,
    IndexedMultiSourceDataset,
    check_dilemmadata_target_cache,
    collate_multisource_samples,
    load_corpus_index,
    load_dilemmadata_target_bundle,
    load_dilemmadata_target_cache_index,
    load_split_manifest,
)
from music_critic.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from music_critic.training.device import move_multisource_batch


DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION = "1.2.0"
DILEMMADATA_SUPERVISED_SMOKE_BUNDLE_VERSION = "1.2.0"
DILEMMADATA_CUDA_REPLAY_DIAGNOSTIC_VERSION = "1.0.0"
DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE = 0.005
DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE = 0.005
DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY = 0.9999
DILEMMADATA_SUPERVISED_SMOKE_PHASE = "9B.2C"
DILEMMADATA_SUPERVISED_SMOKE_SEED = 17
DILEMMADATA_SUPERVISED_SMOKE_UPDATES = 10
DILEMMADATA_SUPERVISED_SMOKE_LEARNING_RATE = 3e-4
DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT = (
    "c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e"
)
DILEMMADATA_SUPERVISED_SMOKE_LOCAL_TARGET_INDEX_FINGERPRINT = (
    "76feee8d128cc3c5dd1a5b261599df89ef241baa21d82b3c24202a11218beea4"
)
# Backward-compatible observed-value alias. It is never an acceptance allowlist.
DILEMMADATA_SUPERVISED_SMOKE_TARGET_INDEX_FINGERPRINT = (
    DILEMMADATA_SUPERVISED_SMOKE_LOCAL_TARGET_INDEX_FINGERPRINT
)
DILEMMADATA_SUPERVISED_SMOKE_RTX_TARGET_INDEX_FINGERPRINT = (
    "02fcf7eb03adda2962ade7223924e0fe44483e4900097bd33f50bf93b68d862a"
)
DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS = (
    DILEMMADATA_SUPERVISED_SMOKE_LOCAL_TARGET_INDEX_FINGERPRINT,
    DILEMMADATA_SUPERVISED_SMOKE_RTX_TARGET_INDEX_FINGERPRINT,
)
DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT = (
    "41e15e1d2edb1c52ad3ca90acf782bec7c26bfb042fea51dc805d6f86b52d0a7"
)
DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT = 719
DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT = (
    "939ad5b871db28fefd76e47d56243ac2109a8bb01d57c6391f424ae943159072"
)
DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT = (
    "58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e"
)
DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT = (
    "69a1ab3e6f5deb98a8bcfa26af7a3177b345ad157d164a3cf72e0273a0c58c81"
)
DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME = "NVIDIA GeForce RTX 3090"

_TARGET_CONTRACT_VERSIONS = {
    "target_cache": "1.0.0",
    "target_cache_index": "1.0.0",
    "target_cache_identity": "1.0.0",
    "target_adapter": "1.1.0",
    "target_sidecar": "1.0.0",
    "raw_alignment_evidence": "1.1.0",
    "source_native_family_registry": "1.0.0",
    "target_encoding_registry": "1.0.0",
    "target_alignment_rules": "1.0.0",
    "target_bundle": "1.0.0",
}

_ENCODER_PREFIXES = (
    "local_baseline.encoder.",
    "context_encoder.pooling.",
    "context_encoder.transformer.",
    "context_encoder.fusion.",
)
_JSON_ARTIFACTS = (
    "run_report.json",
    "train_membership.json",
    "train_priors.json",
    "validation_membership.json",
    "validation_report.json",
)
_EVIDENCE_ARTIFACTS = tuple(
    sorted(
        (*_JSON_ARTIFACTS, "checkpoint.pt", "checkpoint.pt.sha256", "execution.log")
    )
)
_SEALED_ARTIFACTS = tuple(sorted((*_EVIDENCE_ARTIFACTS, "artifact_manifest.json")))
_FORBIDDEN_RUNTIME_MODULES = (
    "music_critic.adapters.dilemmadata",
    "music_critic.adapters.dilemmadata_targets",
)
_GUARDED_SOURCE_FUNCTIONS = (
    "music_critic.adapters.dilemmadata.convert_dilemmadata_record",
    "music_critic.adapters.dilemmadata.discover_dilemmadata_corpus",
    "music_critic.adapters.dilemmadata.reconstruct_dilemmadata_alignment_evidence",
    "music_critic.adapters.dilemmadata_targets.build_dilemmadata_target_sidecar",
)
_MAX_BUNDLE_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_BUNDLE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024


class DilemmadataSupervisedSmokeError(ValueError):
    """Stable fail-closed Phase 9B.2C boundary."""

    def __init__(self, category: str, message: str | None = None) -> None:
        self.category = category
        super().__init__(f"[{category}] {message or category}")


def _install_source_access_guards() -> tuple[dict[str, object], object]:
    guarded = {
        "music_critic.adapters.dilemmadata": (
            "convert_dilemmadata_record",
            "discover_dilemmadata_corpus",
            "reconstruct_dilemmadata_alignment_evidence",
        ),
        "music_critic.adapters.dilemmadata_targets": (
            "build_dilemmadata_target_sidecar",
        ),
    }
    originals = {}
    evidence: dict[str, object] = {
        "guarded_functions": [],
        "forbidden_call_count": 0,
    }

    def forbidden(*args, **kwargs):
        del args, kwargs
        evidence["forbidden_call_count"] = int(evidence["forbidden_call_count"]) + 1
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.source_or_oracle_access_forbidden"
        )

    for module_name, names in guarded.items():
        module = importlib.import_module(module_name)
        for name in names:
            if not hasattr(module, name):
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.source_guard_target_missing",
                    f"{module_name}.{name}",
                )
            originals[(module, name)] = getattr(module, name)
            setattr(module, name, forbidden)
            evidence["guarded_functions"].append(f"{module_name}.{name}")
    evidence["guarded_functions"] = sorted(evidence["guarded_functions"])

    def restore() -> None:
        for (module, name), value in originals.items():
            setattr(module, name, value)

    return evidence, restore


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.json_invalid", str(exc)
        ) from exc


def _fingerprint(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_git_sha(value: object, category: str) -> str:
    if not _is_git_sha(value):
        raise DilemmadataSupervisedSmokeError(category)
    return str(value)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _with_fingerprint(value: dict[str, object]) -> dict[str, object]:
    return {**value, "fingerprint": _fingerprint(value)}


def _validate_fingerprinted(
    value: object, *, category: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DilemmadataSupervisedSmokeError(category)
    payload = dict(value)
    fingerprint = payload.pop("fingerprint", None)
    if fingerprint != _fingerprint(payload):
        raise DilemmadataSupervisedSmokeError(category)
    return payload


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path, category: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DilemmadataSupervisedSmokeError(category, str(exc)) from exc
    if not isinstance(value, dict):
        raise DilemmadataSupervisedSmokeError(category)
    return value


def _require_regular(path: Path, *, directory: bool, category: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DilemmadataSupervisedSmokeError(category, str(exc)) from exc
    if path.is_symlink() or (resolved.is_dir() if directory else resolved.is_file()) is False:
        raise DilemmadataSupervisedSmokeError(category)
    return resolved


def _git_output(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo_root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.git_unavailable", str(exc)
        ) from exc
    return completed.stdout.strip()


def validate_git_preflight(
    repo_root: Path, *, expected_head: str, allowed_output_root: Path
) -> dict[str, object]:
    """Require exact HEAD and a clean tree excluding only the output root."""

    _require_git_sha(expected_head, "dilemmadata.smoke.expected_head_invalid")
    root = Path(_git_output(repo_root, "rev-parse", "--show-toplevel")).resolve()
    head = _git_output(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.git_head_mismatch", f"expected={expected_head},actual={head}"
        )
    output = allowed_output_root.resolve()
    command = ["status", "--porcelain=v1", "--untracked-files=all", "--", "."]
    try:
        relative = output.relative_to(root).as_posix()
    except ValueError:
        relative = None
    if relative not in (None, "."):
        command.extend((f":(exclude){relative}", f":(exclude){relative}/**"))
    dirty = _git_output(root, *command)
    if dirty:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.worktree_dirty", dirty
        )
    return {
        "repo_root": str(root),
        "expected_head": expected_head,
        "actual_head": head,
        "worktree_clean_excluding_output_root": True,
        "allowed_output_root": str(output),
    }


def _cuda_preflight() -> tuple[torch.device, dict[str, object]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.cuda_unavailable"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    if properties.name != DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.gpu_mismatch", properties.name
        )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    if not scaler.is_enabled():
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.grad_scaler_disabled"
        )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        probe = torch.ones((8, 8), device=device) @ torch.ones(
            (8, 8), device=device
        )
    if probe.dtype != torch.float16 or probe.device != device:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.amp_float16_unavailable"
        )
    del probe, scaler
    torch.cuda.synchronize(device)
    return device, {
        "accelerator": properties.name,
        "logical_cuda_index": 0,
        "device": "cuda:0",
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(properties.total_memory),
        "amp_enabled": True,
        "amp_dtype": "float16",
        "grad_scaler_enabled": True,
        "cpu_fallback": False,
    }


def _dialect(source_record_id: str) -> str:
    prefix = source_record_id.split(":", 1)[0]
    if prefix not in {"an", "dlc"}:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.dialect_invalid", source_record_id
        )
    return prefix


def _membership_record(assignment: object, target_record: object) -> dict[str, object]:
    return {
        "dataset_id": assignment.dataset_id,
        "piece_id": assignment.piece_id,
        "component_fingerprint": assignment.component_fingerprint,
        "source_group_id": assignment.source_group_id,
        "lineage_group_id": assignment.lineage_group_id,
        "dialect": _dialect(target_record.source_record_id),
        "target_cache_binding": {
            "cache_identity_fingerprint": target_record.cache_identity_fingerprint,
            "target_bundle_fingerprint": target_record.target_bundle_fingerprint,
            "artifact_sha256": target_record.artifact_sha256,
            "raw_cache_key": target_record.raw_cache_key,
            "canonical_artifact_sha256": target_record.canonical_artifact_sha256,
        },
    }


def _train_membership(
    raw_index: object,
    target_index: object,
    target_config: DilemmadataTargetCacheConfig,
    split_manifest: object,
) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    raw_identities = {(row.dataset_id, row.piece_id) for row in raw_index.records}
    target_by_identity = target_index.by_identity()
    assignments = tuple(
        row
        for row in split_manifest.assignments
        if row.dataset_id == "dilemmadata" and row.split == "train"
    )
    required_by_dialect = {
        "an": frozenset(task for task in DILEMMADATA_ACTIVE_TASK_IDS if ".an." in task),
        "dlc": frozenset(task for task in DILEMMADATA_ACTIVE_TASK_IDS if ".dlc." in task),
    }
    coverage: dict[tuple[str, str], frozenset[str]] = {}
    reads: list[dict[str, object]] = []
    assignment_by_identity = {
        (row.dataset_id, row.piece_id): row for row in assignments
    }
    for identity in sorted(assignment_by_identity):
        if identity not in raw_identities or identity not in target_by_identity:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.train_binding_missing", ":".join(identity)
            )
        target_record = target_by_identity[identity]
        bundle = load_dilemmadata_target_bundle(target_record, target_config)
        available = frozenset(
            target.task_id
            for target in bundle.targets
            if target.task_id in DILEMMADATA_ACTIVE_TASK_IDS
            and any(target.availability_mask)
        )
        coverage[identity] = available
        reads.append(
            {
                "dataset_id": identity[0],
                "piece_id": identity[1],
                "split": "train",
                "artifact_sha256": target_record.artifact_sha256,
            }
        )
    selected: set[tuple[str, str]] = set()
    for dialect, required in required_by_dialect.items():
        candidates = tuple(
            identity
            for identity in sorted(coverage)
            if _dialect(target_by_identity[identity].source_record_id) == dialect
        )
        complete = next(
            (identity for identity in candidates if required <= coverage[identity]),
            None,
        )
        if complete is not None:
            selected.add(complete)
            continue
        for task_id in sorted(required):
            candidate = next(
                (identity for identity in candidates if task_id in coverage[identity]),
                None,
            )
            if candidate is None:
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.train_supervision_incomplete", task_id
                )
            selected.add(candidate)
    identities = tuple(sorted(selected))
    union = frozenset().union(*(coverage[identity] for identity in identities))
    if union != frozenset(DILEMMADATA_ACTIVE_TASK_IDS):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.train_supervision_incomplete"
        )
    records = tuple(
        _membership_record(assignment_by_identity[identity], target_by_identity[identity])
        for identity in identities
    )
    payload = {
        "contract_version": DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
        "split": "train",
        "selection_policy": "lexicographic_minimum_train_target_coverage_v1",
        "selection_may_read_labels": True,
        "replacement": False,
        "split_manifest_fingerprint": split_manifest.manifest_fingerprint,
        "required_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
        "covered_task_ids": sorted(union),
        "dataset_counts": {"dilemmadata": len(records)},
        "dialect_counts": {
            dialect: sum(row["dialect"] == dialect for row in records)
            for dialect in ("an", "dlc")
        },
        "records": list(records),
        "target_artifact_access": {
            "allowed_splits": ["train"],
            "observed_splits": sorted({str(row["split"]) for row in reads}),
            "artifact_read_count": len(reads),
            "test_target_accessed": False,
        },
    }
    return _with_fingerprint(payload), identities


def _validation_membership(
    target_index: object,
    split_manifest: object,
    *,
    limit: int,
) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.validation_limit_invalid"
        )
    target_by_identity = target_index.by_identity()
    candidates = tuple(
        row
        for row in split_manifest.assignments
        if row.dataset_id == "dilemmadata" and row.split == "validation"
    )
    ranked = tuple(
        sorted(
            candidates,
            key=lambda row: (
                _fingerprint(
                    {
                        "seed": DILEMMADATA_SUPERVISED_SMOKE_SEED,
                        "dataset_id": row.dataset_id,
                        "piece_id": row.piece_id,
                        "component_fingerprint": row.component_fingerprint,
                    }
                ),
                row.piece_id,
            ),
        )
    )
    selected: list[object] = []
    for dialect in ("an", "dlc"):
        row = next(
            (
                item
                for item in ranked
                if (item.dataset_id, item.piece_id) in target_by_identity
                and _dialect(
                    target_by_identity[(item.dataset_id, item.piece_id)].source_record_id
                )
                == dialect
            ),
            None,
        )
        if row is None:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.validation_dialect_missing", dialect
            )
        selected.append(row)
    for row in ranked:
        if len(selected) >= min(limit, len(ranked)):
            break
        if row not in selected:
            selected.append(row)
    selected = sorted(selected, key=lambda row: (row.dataset_id, row.piece_id))
    identities = tuple((row.dataset_id, row.piece_id) for row in selected)
    records = tuple(
        _membership_record(row, target_by_identity[(row.dataset_id, row.piece_id)])
        for row in selected
    )
    payload = {
        "contract_version": DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
        "split": "validation",
        "selection_policy": "seed17_identity_component_rank_v1",
        "selection_may_read_labels": False,
        "replacement": False,
        "requested_limit": limit,
        "split_manifest_fingerprint": split_manifest.manifest_fingerprint,
        "dataset_counts": {"dilemmadata": len(records)},
        "dialect_counts": {
            dialect: sum(row["dialect"] == dialect for row in records)
            for dialect in ("an", "dlc")
        },
        "records": list(records),
        "target_artifact_access_during_selection": {
            "artifact_read_count": 0,
            "validation_labels_read": False,
            "test_target_accessed": False,
        },
    }
    return _with_fingerprint(payload), identities


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _prediction_evidence(predictions: Sequence[object]) -> dict[str, object]:
    identities = []
    logits = []
    for prediction in predictions:
        identities.append(
            {
                "task_id": prediction.task_id,
                "candidate_node_type_codes": _tensor_fingerprint(
                    prediction.candidate_node_type_codes
                ),
                "global_entity_indices": _tensor_fingerprint(
                    prediction.global_entity_indices
                ),
                "sample_indices": _tensor_fingerprint(prediction.sample_indices),
                "candidate_offsets_by_node_type": _tensor_fingerprint(
                    prediction.candidate_offsets_by_node_type
                ),
                "candidate_counts_by_node_type": _tensor_fingerprint(
                    prediction.candidate_counts_by_node_type
                ),
                "candidate_count": int(prediction.logits.shape[0]),
            }
        )
        logits.append(
            {
                "task_id": prediction.task_id,
                "fingerprint": _tensor_fingerprint(prediction.logits),
            }
        )
    return {
        "candidate_identity_fingerprint": _fingerprint(identities),
        "raw_only_logits_fingerprint": _fingerprint(logits),
        "candidate_counts": {
            prediction.task_id: int(prediction.logits.shape[0])
            for prediction in predictions
        },
    }


_PREDICTION_TENSOR_FIELDS = (
    "candidate_node_type_codes",
    "global_entity_indices",
    "sample_indices",
    "candidate_offsets_by_node_type",
    "candidate_counts_by_node_type",
    "logits",
)


def _prediction_snapshot(predictions: Sequence[object]) -> dict[str, object]:
    return {
        "sequence": predictions,
        "rows": tuple(
            {
                "row": prediction,
                "metadata": (
                    prediction.contract_version,
                    prediction.task_id,
                    prediction.source_adapter,
                    prediction.allowed_node_types,
                ),
                "tensors": {
                    name: {
                        "tensor": tensor,
                        "storage_data_ptr": tensor.untyped_storage().data_ptr(),
                        "data_ptr": tensor.data_ptr(),
                        "storage_offset": tensor.storage_offset(),
                        "stride": tensor.stride(),
                        "value": tensor.detach().clone(),
                    }
                    for name in _PREDICTION_TENSOR_FIELDS
                    for tensor in (getattr(prediction, name),)
                },
            }
            for prediction in predictions
        ),
        "evidence": _prediction_evidence(predictions),
    }


def _assert_prediction_snapshot(
    predictions: Sequence[object], snapshot: Mapping[str, object]
) -> dict[str, object]:
    rows = snapshot["rows"]
    if predictions is not snapshot["sequence"] or len(predictions) != len(rows):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_join_replaced_prediction_object"
        )
    for prediction, row in zip(predictions, rows, strict=True):
        metadata = (
            prediction.contract_version,
            prediction.task_id,
            prediction.source_adapter,
            prediction.allowed_node_types,
        )
        if prediction is not row["row"] or metadata != row["metadata"]:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.target_join_changed_candidate_identities"
            )
        for name in _PREDICTION_TENSOR_FIELDS:
            tensor = getattr(prediction, name)
            expected = row["tensors"][name]
            if (
                tensor is not expected["tensor"]
                or tensor.untyped_storage().data_ptr()
                != expected["storage_data_ptr"]
                or tensor.data_ptr() != expected["data_ptr"]
                or tensor.storage_offset() != expected["storage_offset"]
                or tensor.stride() != expected["stride"]
            ):
                category = (
                    "dilemmadata.smoke.target_join_changed_raw_predictions"
                    if name == "logits"
                    else "dilemmadata.smoke.target_join_changed_candidate_identities"
                )
                raise DilemmadataSupervisedSmokeError(category)
            if not torch.equal(tensor, expected["value"]):
                category = (
                    "dilemmadata.smoke.target_join_changed_raw_predictions"
                    if name == "logits"
                    else "dilemmadata.smoke.target_join_changed_candidate_identities"
                )
                raise DilemmadataSupervisedSmokeError(category)
    evidence = _prediction_evidence(predictions)
    if evidence != snapshot["evidence"]:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_join_changed_raw_predictions"
        )
    return evidence


def _supervision_loss_evidence(output: object) -> dict[str, object]:
    rows = [
        {
            "task_id": row.task_id,
            "target_row_indices": _tensor_fingerprint(row.target_row_indices),
            "candidate_indices": _tensor_fingerprint(row.candidate_indices),
            "per_row_loss": _tensor_fingerprint(row.per_row_loss),
        }
        for row in output.supervisions
    ]
    total = output.harmonic_loss.total_loss
    return {
        "fingerprint": _fingerprint(rows),
        "total_loss_fingerprint": None
        if total is None
        else _tensor_fingerprint(total),
    }


def _prediction_replay_diagnostic(
    reference: Sequence[object], replay: Sequence[object]
) -> dict[str, object]:
    reference_identity = _prediction_evidence(reference)[
        "candidate_identity_fingerprint"
    ]
    replay_identity = _prediction_evidence(replay)[
        "candidate_identity_fingerprint"
    ]
    if len(reference) != len(replay) or reference_identity != replay_identity:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.cuda_replay_candidate_identity_mismatch"
        )
    tasks = []
    for left, right in zip(reference, replay, strict=True):
        if left.task_id != right.task_id or left.logits.shape != right.logits.shape:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.cuda_replay_candidate_identity_mismatch"
            )
        left_fp32 = left.logits.detach().float()
        right_fp32 = right.logits.detach().float()
        if not bool(torch.isfinite(left_fp32).all()) or not bool(
            torch.isfinite(right_fp32).all()
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.cuda_replay_non_finite"
            )
        difference = (left_fp32 - right_fp32).abs()
        max_absolute = float(difference.max()) if difference.numel() else 0.0
        denominator = torch.maximum(
            torch.maximum(left_fp32.abs(), right_fp32.abs()),
            torch.full_like(left_fp32, 1e-12),
        )
        max_relative = (
            float((difference / denominator).max()) if difference.numel() else 0.0
        )
        if left_fp32.numel():
            left_flat = left_fp32.reshape(-1)
            right_flat = right_fp32.reshape(-1)
            norms = float(torch.linalg.vector_norm(left_flat)) * float(
                torch.linalg.vector_norm(right_flat)
            )
            cosine = (
                1.0
                if norms == 0.0 and torch.equal(left_flat, right_flat)
                else float(torch.dot(left_flat, right_flat)) / norms
                if norms > 0.0
                else 0.0
            )
        else:
            cosine = 1.0
        within_elementwise = bool(
            torch.all(
                difference
                <= DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE
                + DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE
                * left_fp32.abs()
            )
        )
        if (
            not within_elementwise
            or cosine < DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.cuda_replay_tolerance_exceeded",
                (
                    f"task={left.task_id},max_abs={max_absolute},"
                    f"max_rel={max_relative},cosine={cosine}"
                ),
            )
        tasks.append(
            {
                "task_id": left.task_id,
                "max_absolute_difference_fp32": max_absolute,
                "max_relative_difference_fp32": max_relative,
                "cosine_similarity_fp32": cosine,
                "within_elementwise_tolerance": True,
            }
        )
    return {
        "contract_version": DILEMMADATA_CUDA_REPLAY_DIAGNOSTIC_VERSION,
        "purpose": "independent_cuda_amp_replay_not_target_leakage",
        "candidate_identities_exact": True,
        "all_logits_finite": True,
        "comparison_dtype": "float32",
        "absolute_tolerance": DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE,
        "relative_tolerance": DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE,
        "minimum_cosine_similarity": DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY,
        "tasks": tasks,
    }


def _mutate_targets(batch: object, model: DilemmadataHierarchicalModel) -> tuple[object, int]:
    output_dims = {spec.task_id: spec.output_dim for spec in model.task_specs}
    targets = []
    mutated = 0
    for target in batch.target_batches:
        if target.task_id not in output_dims:
            targets.append(target)
            continue
        values = target.values.clone()
        mask = target.availability_mask
        mutated += int(mask.sum().item())
        values[mask] = (values[mask] + 1) % output_dims[target.task_id]
        targets.append(replace(target, values=values))
    if mutated == 0:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_mutation_empty"
        )
    return replace(batch, target_batches=tuple(targets)), mutated


def _target_supervision_fingerprint(target_batches: Sequence[object]) -> str:
    rows = []
    for target in target_batches:
        if target.task_id not in DILEMMADATA_ACTIVE_TASK_IDS:
            continue
        rows.append(
            {
                "task_id": target.task_id,
                "values": _tensor_fingerprint(target.values),
                "availability_mask": _tensor_fingerprint(
                    target.availability_mask
                ),
                "entity_indices": _tensor_fingerprint(target.entity_indices),
                "sample_indices": _tensor_fingerprint(target.sample_indices),
                "source_entry_indices": _tensor_fingerprint(
                    target.source_entry_indices
                ),
            }
        )
    return _fingerprint(rows)


def _verify_source_entry_reduction(output: object) -> dict[str, object]:
    if (
        output.harmonic_loss.reduction
        != "candidate_rows_mean_per_source_entry_then_entries_mean_per_task_fixed_weight_sum"
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.source_entry_reduction_mismatch"
        )
    report_by_task = {
        item.task_id: item for item in output.harmonic_loss.task_losses
    }
    if set(report_by_task) != set(DILEMMADATA_ACTIVE_TASK_IDS):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.four_task_supervision_incomplete"
        )
    evidence = {}
    for supervision in output.supervisions:
        keys = tuple(
            zip(
                supervision.sample_indices.detach().cpu().tolist(),
                supervision.source_entry_indices.detach().cpu().tolist(),
                strict=True,
            )
        )
        grouped: dict[tuple[int, int], list[float]] = {}
        for key, loss in zip(
            keys,
            supervision.per_row_loss.detach().float().cpu().tolist(),
            strict=True,
        ):
            grouped.setdefault(key, []).append(float(loss))
        means = [sum(grouped[key]) / len(grouped[key]) for key in sorted(grouped)]
        expected = sum(means) / len(means)
        actual = float(report_by_task[supervision.task_id].mean_loss.detach())
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.source_entry_reduction_mismatch",
                supervision.task_id,
            )
        evidence[supervision.task_id] = {
            "expanded_row_count": len(keys),
            "effective_source_entry_count": len(grouped),
            "independent_mean_loss": expected,
            "reported_mean_loss": actual,
        }
    return {
        "verified": True,
        "reduction": output.harmonic_loss.reduction,
        "tasks": evidence,
    }


def _train_prior_rows(output: object, batch: object) -> list[dict[str, object]]:
    predictions = {row.task_id: row for row in output.predictions}
    targets = {row.task_id: row for row in batch.target_batches}
    rows: list[dict[str, object]] = []
    for supervision in output.supervisions:
        target = targets[supervision.task_id]
        prediction = predictions[supervision.task_id]
        labels = target.values.index_select(
            0, supervision.target_row_indices
        ).detach().cpu()
        sample_indices = supervision.sample_indices.detach().cpu().tolist()
        source_indices = supervision.source_entry_indices.detach().cpu().tolist()
        by_key: dict[tuple[int, int], int] = {}
        for position, key in enumerate(
            zip(sample_indices, source_indices, strict=True)
        ):
            label = int(labels[position])
            previous = by_key.setdefault(key, label)
            if previous != label:
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.train_source_entry_label_conflict"
                )
        class_count = int(prediction.logits.shape[1])
        for (sample_index, source_index), label in sorted(by_key.items()):
            rows.append(
                {
                    "task_id": supervision.task_id,
                    "dataset_id": batch.dataset_ids[sample_index],
                    "piece_id": batch.piece_ids[sample_index],
                    "source_entry_index": source_index,
                    "label": label,
                    "log_probabilities": [0.0 for _ in range(class_count)],
                }
            )
    return rows


def _parameter_groups(
    model: DilemmadataHierarchicalModel,
) -> dict[str, tuple[str, ...]]:
    names = tuple(name for name, _ in model.named_parameters())
    groups = {
        "raw_encoder": tuple(
            name for name in names if name.startswith(_ENCODER_PREFIXES)
        )
    }
    for index, task_id in enumerate(DILEMMADATA_ACTIVE_TASK_IDS):
        prefix = f"task_heads.heads.task_{index:02d}."
        groups[task_id] = tuple(name for name in names if name.startswith(prefix))
    if not groups["raw_encoder"] or any(
        not groups[task_id] for task_id in DILEMMADATA_ACTIVE_TASK_IDS
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.parameter_inventory_invalid"
        )
    return groups


def _snapshot_parameters(
    model: DilemmadataHierarchicalModel, groups: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, torch.Tensor]]:
    parameters = dict(model.named_parameters())
    return {
        group: {
            name: parameters[name].detach().cpu().clone()
            for name in names
        }
        for group, names in groups.items()
    }


def _gradient_evidence(
    model: DilemmadataHierarchicalModel,
    groups: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    non_finite = []
    gradient_parameter_count = 0
    nonzero_by_group = {}
    for group, names in groups.items():
        nonzero = []
        for name in names:
            gradient = parameters[name].grad
            if gradient is None:
                continue
            gradient_parameter_count += 1
            if not bool(torch.isfinite(gradient).all()):
                non_finite.append(name)
            elif bool(torch.count_nonzero(gradient)):
                nonzero.append(name)
        nonzero_by_group[group] = nonzero
    if non_finite:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.gradient_non_finite", ",".join(non_finite)
        )
    return {
        "all_gradients_finite": True,
        "gradient_parameter_count": gradient_parameter_count,
        "non_finite_parameters": [],
        "nonzero_parameter_count_by_group": {
            group: len(names) for group, names in nonzero_by_group.items()
        },
        "nonzero_parameters_by_group": nonzero_by_group,
    }


def _parameter_change_evidence(
    model: DilemmadataHierarchicalModel,
    before: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    parameters = dict(model.named_parameters())
    result = {}
    for group, values in before.items():
        changed = [
            name
            for name, original in values.items()
            if not torch.equal(parameters[name].detach().cpu(), original)
        ]
        result[group] = {
            "changed": bool(changed),
            "changed_parameter_count": len(changed),
            "changed_parameters": changed,
        }
    return result


def _model_state_reload_evidence(
    reference: DilemmadataHierarchicalModel,
    reloaded: DilemmadataHierarchicalModel,
) -> dict[str, object]:
    left = reference.state_dict()
    right = reloaded.state_dict()
    if tuple(left) != tuple(right):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_model_state_inventory_mismatch"
        )
    fingerprints = []
    for name in left:
        if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.checkpoint_model_state_mismatch", name
            )
        fingerprints.append(
            {"name": name, "fingerprint": _tensor_fingerprint(left[name])}
        )
    return {
        "model_state_tensors_exact": True,
        "tensor_count": len(fingerprints),
        "state_fingerprint": _fingerprint(fingerprints),
    }


def _selected_batch(
    dataset: IndexedMultiSourceDataset,
    identities: Sequence[tuple[str, str]],
) -> object:
    by_identity = {
        (row.dataset_id, row.piece_id): index
        for index, row in enumerate(dataset.index.records)
    }
    try:
        samples = tuple(dataset[by_identity[identity]] for identity in identities)
    except KeyError as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.selected_raw_record_missing", str(exc)
        ) from exc
    return collate_multisource_samples(samples)


def _selected_batches(
    dataset: IndexedMultiSourceDataset,
    identities: Sequence[tuple[str, str]],
    *,
    batch_size: int = 2,
) -> tuple[object, ...]:
    return tuple(
        _selected_batch(dataset, identities[start : start + batch_size])
        for start in range(0, len(identities), batch_size)
    )


def _training_config(updates: int) -> dict[str, object]:
    return {
        "phase": DILEMMADATA_SUPERVISED_SMOKE_PHASE,
        "preset": "supervised_scratch",
        "seed": DILEMMADATA_SUPERVISED_SMOKE_SEED,
        "device": "cuda:0",
        "amp": {"enabled": True, "dtype": "float16", "grad_scaler": True},
        "optimizer": {
            "name": "adamw",
            "learning_rate": DILEMMADATA_SUPERVISED_SMOKE_LEARNING_RATE,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": {"name": "cosine_annealing", "t_max": updates},
        "updates": updates,
        "objective": {
            "reconstruction_weight": 0.0,
            "task_weights": {
                task_id: 1.0 for task_id in DILEMMADATA_ACTIVE_TASK_IDS
            },
            "active_task_renormalization": False,
        },
        "transfer": {
            "mode": "supervised_scratch",
            "loaded_encoder_tensors": [],
            "supervised_heads_transferred": False,
            "ssl_heads_transferred": False,
        },
    }


def _cuda_execution(
    *,
    output_dir: Path,
    device: torch.device,
    hardware: dict[str, object],
    dataset: IndexedMultiSourceDataset,
    train_identities: tuple[tuple[str, str], ...],
    validation_identities: tuple[tuple[str, str], ...],
    train_membership: dict[str, object],
    validation_membership: dict[str, object],
    bindings: dict[str, object],
    target_semantic_validation: dict[str, object],
    expected_head: str,
    updates: int,
) -> tuple[dict[str, object], list[weakref.ReferenceType[torch.Tensor]]]:
    random.seed(DILEMMADATA_SUPERVISED_SMOKE_SEED)
    torch.manual_seed(DILEMMADATA_SUPERVISED_SMOKE_SEED)
    torch.cuda.manual_seed_all(DILEMMADATA_SUPERVISED_SMOKE_SEED)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    initialize_cuda_memory_statistics(device)

    cpu_train_batch = _selected_batch(dataset, train_identities)
    cpu_validation_batches = _selected_batches(dataset, validation_identities)
    batch = move_multisource_batch(cpu_train_batch, device)
    model = DilemmadataHierarchicalModel().to(device)
    model_fingerprint = dilemmadata_model_contract_fingerprint(model)
    if model_fingerprint != DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.model_fingerprint_mismatch", model_fingerprint
        )
    if tuple(spec.task_id for spec in model.task_specs) != DILEMMADATA_ACTIVE_TASK_IDS:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.head_inventory_mismatch"
        )
    groups = _parameter_groups(model)
    before = _snapshot_parameters(model, groups)
    tracked: list[weakref.ReferenceType[torch.Tensor]] = []

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        encoded, raw_predictions = model.predict(batch.raw_graph_batch)
    tracked.extend(weakref.ref(row.logits) for row in raw_predictions)
    prediction_snapshot = _prediction_snapshot(raw_predictions)
    initial_prediction = prediction_snapshot["evidence"]
    original_target_fingerprint = _target_supervision_fingerprint(
        batch.target_batches
    )
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        initial_output = model.supervise(
            encoded, raw_predictions, batch.target_batches
        )
    post_original_join = _assert_prediction_snapshot(
        initial_output.predictions, prediction_snapshot
    )
    original_supervision = _supervision_loss_evidence(initial_output)
    if post_original_join != initial_prediction:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_join_changed_raw_predictions"
        )
    reduction = _verify_source_entry_reduction(initial_output)
    prior_rows = _train_prior_rows(initial_output, batch)
    train_priors = build_dilemmadata_train_priors(
        prior_rows,
        train_membership_fingerprint=str(train_membership["fingerprint"]),
    )
    _write_json_atomic(output_dir / "train_priors.json", train_priors)

    mutated_cpu, mutated_rows = _mutate_targets(cpu_train_batch, model)
    mutated_batch = move_multisource_batch(mutated_cpu, device)
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        mutated_output = model.supervise(
            encoded, raw_predictions, mutated_batch.target_batches
        )
    post_mutated_join = _assert_prediction_snapshot(
        mutated_output.predictions, prediction_snapshot
    )
    mutated_target_fingerprint = _target_supervision_fingerprint(
        mutated_batch.target_batches
    )
    mutated_supervision = _supervision_loss_evidence(mutated_output)
    if (
        post_mutated_join != initial_prediction
        or original_target_fingerprint == mutated_target_fingerprint
        or original_supervision == mutated_supervision
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_mutation_evidence_invalid"
        )
    target_invariance = {
        "verified": True,
        "raw_prediction_call_count": 1,
        "same_prediction_object_for_both_joins": True,
        "mutated_target_row_count": mutated_rows,
        "candidate_identity_fingerprint_before": initial_prediction[
            "candidate_identity_fingerprint"
        ],
        "candidate_identity_fingerprint_after": post_mutated_join[
            "candidate_identity_fingerprint"
        ],
        "raw_only_logits_fingerprint_before": initial_prediction[
            "raw_only_logits_fingerprint"
        ],
        "raw_only_logits_fingerprint_after": post_mutated_join[
            "raw_only_logits_fingerprint"
        ],
        "tensor_storage_and_values_exact_after_original_join": True,
        "tensor_storage_and_values_exact_after_mutated_join": True,
        "original_target_fingerprint": original_target_fingerprint,
        "mutated_target_fingerprint": mutated_target_fingerprint,
        "original_supervision_loss": original_supervision,
        "mutated_supervision_loss": mutated_supervision,
    }
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        _, replay_predictions = model.predict(batch.raw_graph_batch)
    tracked.extend(weakref.ref(row.logits) for row in replay_predictions)
    cuda_replay = _prediction_replay_diagnostic(
        raw_predictions, replay_predictions
    )
    del mutated_output, mutated_batch, mutated_cpu, replay_predictions

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DILEMMADATA_SUPERVISED_SMOKE_LEARNING_RATE,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    curve = []
    attempted = applied = skipped = 0
    aggregate_nonzero = {group: False for group in groups}
    for step in range(updates):
        attempted += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            output = model(batch)
            loss = output.harmonic_loss.total_loss
        tracked.extend(weakref.ref(row.logits) for row in output.predictions)
        if loss is None or not bool(torch.isfinite(loss)):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.loss_non_finite", f"step={step}"
            )
        task_ids = tuple(item.task_id for item in output.harmonic_loss.task_losses)
        if task_ids != DILEMMADATA_ACTIVE_TASK_IDS:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.four_task_supervision_incomplete",
                f"step={step},tasks={task_ids}",
            )
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = _gradient_evidence(model, groups)
        for group, count in gradients["nonzero_parameter_count_by_group"].items():
            aggregate_nonzero[group] = aggregate_nonzero[group] or int(count) > 0
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if not bool(torch.isfinite(norm)):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.gradient_norm_non_finite"
            )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        was_skipped = scale_after < scale_before
        if was_skipped:
            skipped += 1
        else:
            applied += 1
            scheduler.step()
        curve.append(
            {
                "step": step,
                "total_loss": float(loss.detach()),
                "task_losses": {
                    item.task_id: float(item.mean_loss.detach())
                    for item in output.harmonic_loss.task_losses
                },
                "gradient_norm_before_clip": float(norm.detach()),
                "gradients": gradients,
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "optimizer_step_applied": not was_skipped,
                "learning_rate_after": float(optimizer.param_groups[0]["lr"]),
            }
        )
    if applied < 1:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.zero_applied_updates"
        )
    if not all(aggregate_nonzero.values()):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.gradient_coverage_incomplete",
            str(aggregate_nonzero),
        )
    changes = _parameter_change_evidence(model, before)
    if not all(bool(value["changed"]) for value in changes.values()):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.parameter_change_incomplete", str(changes)
        )

    model.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        _, final_predictions = model.predict(batch.raw_graph_batch)
    tracked.extend(weakref.ref(row.logits) for row in final_predictions)
    final_prediction = _prediction_evidence(final_predictions)

    training_config = _training_config(updates)
    data_fingerprints = {
        **bindings,
        "train_membership_fingerprint": train_membership["fingerprint"],
        "validation_membership_fingerprint": validation_membership["fingerprint"],
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        committed_metric_rows=0,
        resolved_config=training_config,
        data_fingerprints=data_fingerprints,
    )
    checkpoint_sha = _sha256_file(checkpoint_path)
    _write_text_atomic(output_dir / "checkpoint.pt.sha256", checkpoint_sha + "\n")

    reloaded = DilemmadataHierarchicalModel().to(device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=DILEMMADATA_SUPERVISED_SMOKE_LEARNING_RATE,
        weight_decay=0.0,
    )
    reloaded_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        reloaded_optimizer, T_max=updates
    )
    reloaded_scaler = torch.amp.GradScaler("cuda", enabled=True)
    load_training_checkpoint(
        checkpoint_path,
        reloaded,
        reloaded_optimizer,
        scheduler=reloaded_scheduler,
        scaler=reloaded_scaler,
        maximum_next_epoch=0,
        resolved_config=training_config,
        data_fingerprints=data_fingerprints,
    )
    reload_model_state = _model_state_reload_evidence(model, reloaded)
    reloaded.eval()
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.float16
    ):
        _, reload_predictions = reloaded.predict(batch.raw_graph_batch)
    tracked.extend(weakref.ref(row.logits) for row in reload_predictions)
    reload_prediction = _prediction_evidence(reload_predictions)
    checkpoint_replay = _prediction_replay_diagnostic(
        final_predictions, reload_predictions
    )

    component_by_identity = {
        (str(row["dataset_id"]), str(row["piece_id"])): str(
            row["component_fingerprint"]
        )
        for row in validation_membership["records"]
    }
    validation_batches = tuple(
        move_multisource_batch(cpu_batch, device)
        for cpu_batch in cpu_validation_batches
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        validation_report = evaluate_dilemmadata_model(
            reloaded,
            validation_batches,
            component_by_identity=component_by_identity,
            split="validation",
            membership_fingerprint=str(validation_membership["fingerprint"]),
            train_priors=train_priors,
            test_unlock=None,
        )
    _write_json_atomic(output_dir / "validation_report.json", validation_report)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated <= 0 or peak_reserved <= 0:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.vram_evidence_missing"
        )

    report = {
        "contract_version": DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
        "phase": DILEMMADATA_SUPERVISED_SMOKE_PHASE,
        "expected_head": expected_head,
        "evidence_kind": "bounded_executable_mechanics_not_scientific_quality",
        "hardware": {
            **hardware,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "bindings": bindings,
        "target_semantic_validation": target_semantic_validation,
        "model_contract_fingerprint": model_fingerprint,
        "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
        "excluded_supervision": {
            "positive_unlabeled_task_ids": list(DILEMMADATA_PU_TASK_IDS),
            "positive_unlabeled_ce_heads": 0,
            "positive_unlabeled_ce_losses": 0,
            "open_string_task_ids": list(DILEMMADATA_OPEN_TASK_IDS),
            "open_string_heads": 0,
            "open_string_losses": 0,
        },
        "training_config": training_config,
        "train_membership_fingerprint": train_membership["fingerprint"],
        "validation_membership_fingerprint": validation_membership["fingerprint"],
        "candidate_first": {
            "prediction_completed_before_target_join": True,
            "target_columns_read_only_after_raw_prediction": True,
            "prediction_object_exact_after_target_joins": True,
            "initial": initial_prediction,
            "target_mutation": target_invariance,
        },
        "cuda_replay_diagnostic": cuda_replay,
        "source_entry_reduction": reduction,
        "optimization": {
            "attempted_update_count": attempted,
            "applied_update_count": applied,
            "skipped_update_count": skipped,
            "all_losses_finite": all(_finite(row["total_loss"]) for row in curve),
            "all_gradients_finite": all(
                row["gradients"]["all_gradients_finite"] for row in curve
            ),
            "aggregate_nonzero_gradient_by_group": aggregate_nonzero,
            "parameter_changes": changes,
            "initial_loss": curve[0]["total_loss"],
            "minimum_loss": min(float(row["total_loss"]) for row in curve),
            "final_loss": curve[-1]["total_loss"],
            "curve": curve,
        },
        "checkpoint": {
            "artifact": "checkpoint.pt",
            "sha256": checkpoint_sha,
            "model_contract_fingerprint": model_fingerprint,
            "raw_index_fingerprint": bindings["raw_index_fingerprint"],
            "observed_target_cache_index_fingerprint": bindings[
                "observed_target_cache_index_fingerprint"
            ],
            "split_manifest_fingerprint": bindings["split_manifest_fingerprint"],
            "active_task_ids": list(DILEMMADATA_ACTIVE_TASK_IDS),
            "seed": DILEMMADATA_SUPERVISED_SMOKE_SEED,
            "train_membership_fingerprint": train_membership["fingerprint"],
            "cuda_device": "cuda:0",
            "amp_dtype": "float16",
            "grad_scaler_state_present": bool(scaler.state_dict()),
            "optimizer_state_present": bool(optimizer.state_dict()["state"]),
            "scheduler_state_present": bool(scheduler.state_dict()),
            "scratch_loaded_encoder_tensors": [],
            "scratch_supervised_heads_transferred": False,
            "scratch_ssl_heads_transferred": False,
            "reload_failure_atomic_contract": "training_checkpoint@1.0.0",
            "reload_model_state": reload_model_state,
            "reload_logits_bounded_cuda_replay": True,
            "reload_cuda_replay_diagnostic": checkpoint_replay,
            "final_raw_only_logits_fingerprint": final_prediction[
                "raw_only_logits_fingerprint"
            ],
            "reloaded_raw_only_logits_fingerprint": reload_prediction[
                "raw_only_logits_fingerprint"
            ],
        },
        "validation": {
            "artifact": "validation_report.json",
            "report_fingerprint": validation_report["fingerprint"],
            "official_evaluator": True,
            "split": "validation",
            "selection_uses_labels": False,
            "replacement": False,
            "train_only_baseline_fingerprint": train_priors["fingerprint"],
            "observed_target_cache_index_fingerprint": bindings[
                "observed_target_cache_index_fingerprint"
            ],
            "test_split_accessed": False,
            "test_targets_accessed": False,
            "test_metrics_computed": False,
            "test_unlock_used": False,
        },
        "runtime_access": {
            "accepted_inputs": [
                "raw_index",
                "raw_cache",
                "target_index",
                "target_cache",
                "split_manifest",
            ],
            "source_tsv_path_accepted": False,
            "raw_adapter_called": False,
            "alignment_oracle_called": False,
            "worker_count": 0,
            "adapter_modules_loaded_for_fail_closed_guards": [
                name for name in _FORBIDDEN_RUNTIME_MODULES if name in sys.modules
            ],
        },
        "claim_boundaries": {
            "bounded_mechanics_only": True,
            "scratch_vs_ssl_comparison": False,
            "representation_quality_claim": False,
            "calibration_or_significance_claim": False,
            "long_training_executed": False,
            "test_split_opened": False,
            "phase9c_started": False,
            "pdmx_started": False,
            "phase10_started": False,
            "legacy_used": False,
        },
    }
    return _with_fingerprint(report), tracked


def _validate_production_bindings(
    raw_index: object,
    target_index: object,
    target_cache_config: DilemmadataTargetCacheConfig,
    split_manifest: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate stable semantics, then retain the exact physical index binding."""

    bindings = {
        "raw_index_fingerprint": raw_index.header.index_fingerprint,
        "observed_target_cache_index_fingerprint": target_index.index_fingerprint,
        "split_manifest_fingerprint": split_manifest.manifest_fingerprint,
    }
    if (
        bindings["raw_index_fingerprint"]
        != DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT
        or bindings["split_manifest_fingerprint"]
        != DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT
        or not _is_sha256(bindings["observed_target_cache_index_fingerprint"])
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.production_fingerprint_mismatch",
            f"actual={bindings}",
        )
    if target_index.raw_index_fingerprint != raw_index.header.index_fingerprint:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_raw_binding_mismatch"
        )
    split_indices = dict(split_manifest.index_fingerprints)
    if split_indices.get("dilemmadata") != raw_index.header.index_fingerprint:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.split_raw_binding_mismatch"
        )
    try:
        checked = check_dilemmadata_target_cache(
            target_index,
            raw_index=raw_index,
            cache_config=target_cache_config,
        )
    except DilemmadataTargetCacheError as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_semantic_validation_failed",
            f"{exc.category}:{exc}",
        ) from exc
    expected_semantics = {
        "record_count": DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT,
        "raw_index_fingerprint": DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT,
        "metadata_index_fingerprint": (
            DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT
        ),
        "aggregate_target_bundle_fingerprint": (
            DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT
        ),
    }
    observed_semantics = {
        "record_count": checked.get("record_count"),
        "raw_index_fingerprint": checked.get("raw_index_fingerprint"),
        "metadata_index_fingerprint": target_index.metadata_index_fingerprint,
        "aggregate_target_bundle_fingerprint": checked.get(
            "target_bundle_fingerprint"
        ),
    }
    if observed_semantics != expected_semantics:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_semantic_projection_mismatch",
            f"expected={expected_semantics},actual={observed_semantics}",
        )
    semantic_validation = _with_fingerprint(
        {
            "policy": "stable_semantics_plus_observed_physical_index_v1",
            **observed_semantics,
            "observed_target_cache_index_fingerprint": target_index.index_fingerprint,
            "known_observed_physical_index_fingerprints": list(
                DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS
            ),
            "target_index_role": (
                "exact_run_resume_evaluation_binding_not_universal_semantic_identity"
            ),
            "contract_versions": dict(_TARGET_CONTRACT_VERSIONS),
            "source_free_full_validation": {
                "index_self_fingerprint_verified": True,
                "index_record_count_verified": len(target_index.records),
                "artifact_sha256_verified_count": len(target_index.records),
                "target_bundle_fingerprint_verified_count": len(target_index.records),
            },
        }
    )
    bindings["target_semantic_projection_fingerprint"] = semantic_validation[
        "fingerprint"
    ]
    return bindings, semantic_validation


def _run_supervised_smoke_guarded(
    *,
    repo_root: Path,
    expected_head: str,
    raw_index_path: Path,
    raw_cache_root: Path,
    target_index_path: Path,
    target_cache_root: Path,
    split_manifest_path: Path,
    output_root: Path,
    output_dir: Path,
    updates: int = DILEMMADATA_SUPERVISED_SMOKE_UPDATES,
    validation_limit: int = 8,
    source_guard_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Execute the official production-cache-only bounded CUDA smoke."""

    if isinstance(updates, bool) or not isinstance(updates, int) or not 10 <= updates <= 20:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.update_budget_invalid"
        )
    output_root = output_root.resolve()
    try:
        output_dir = output_dir.resolve(strict=True)
        output_dir.relative_to(output_root)
    except (OSError, ValueError) as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.output_path_invalid", str(exc)
        ) from exc
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.output_path_invalid"
        )
    existing = tuple(sorted(path.name for path in output_dir.iterdir()))
    if existing not in ((), ("execution.log",)):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.output_collision", str(existing)
        )
    git = validate_git_preflight(
        repo_root,
        expected_head=expected_head,
        allowed_output_root=output_root,
    )
    raw_index_path = _require_regular(
        raw_index_path, directory=False, category="dilemmadata.smoke.raw_index_invalid"
    )
    raw_cache_root = _require_regular(
        raw_cache_root, directory=True, category="dilemmadata.smoke.raw_cache_invalid"
    )
    target_index_path = _require_regular(
        target_index_path,
        directory=False,
        category="dilemmadata.smoke.target_index_invalid",
    )
    target_cache_root = _require_regular(
        target_cache_root,
        directory=True,
        category="dilemmadata.smoke.target_cache_invalid",
    )
    split_manifest_path = _require_regular(
        split_manifest_path,
        directory=False,
        category="dilemmadata.smoke.split_manifest_invalid",
    )
    device, hardware = _cuda_preflight()
    raw_index = load_corpus_index(raw_index_path)
    target_index = load_dilemmadata_target_cache_index(target_index_path)
    split_manifest = load_split_manifest(split_manifest_path)
    target_config = DilemmadataTargetCacheConfig(target_cache_root)
    bindings, target_semantic_validation = _validate_production_bindings(
        raw_index, target_index, target_config, split_manifest
    )
    train_membership, train_identities = _train_membership(
        raw_index, target_index, target_config, split_manifest
    )
    validation_membership, validation_identities = _validation_membership(
        target_index, split_manifest, limit=validation_limit
    )
    if set(train_identities) & set(validation_identities):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.train_validation_overlap"
        )
    _write_json_atomic(output_dir / "train_membership.json", train_membership)
    _write_json_atomic(
        output_dir / "validation_membership.json", validation_membership
    )
    dataset = IndexedMultiSourceDataset(
        raw_index,
        cache_config=CorpusCacheConfig(raw_cache_root),
        target_cache_index=target_index,
        target_cache_config=target_config,
        require_target_sidecars=True,
    )
    report, tracked = _cuda_execution(
        output_dir=output_dir,
        device=device,
        hardware=hardware,
        dataset=dataset,
        train_identities=train_identities,
        validation_identities=validation_identities,
        train_membership=train_membership,
        validation_membership=validation_membership,
        bindings=bindings,
        target_semantic_validation=target_semantic_validation,
        expected_head=expected_head,
        updates=updates,
    )
    del dataset
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    retained_predictions = sum(reference() is not None for reference in tracked)
    allocated_end = int(torch.cuda.memory_allocated(0))
    if retained_predictions or allocated_end:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.cuda_tensor_retention",
            f"predictions={retained_predictions},allocated={allocated_end}",
        )
    payload = dict(report)
    payload.pop("fingerprint")
    payload["git_preflight"] = git
    payload["lifecycle"] = {
        "tracked_prediction_tensor_count": len(tracked),
        "retained_prediction_tensor_count": retained_predictions,
        "allocated_bytes_after_cleanup": allocated_end,
        "retained_cuda_tensor_count": 0,
    }
    payload["runtime_access"]["source_access_guard"] = dict(
        source_guard_evidence
    )
    report = _with_fingerprint(payload)
    validate_smoke_report(report)
    _write_json_atomic(output_dir / "run_report.json", report)
    return report


def run_supervised_smoke(
    *,
    repo_root: Path,
    expected_head: str,
    raw_index_path: Path,
    raw_cache_root: Path,
    target_index_path: Path,
    target_cache_root: Path,
    split_manifest_path: Path,
    output_root: Path,
    output_dir: Path,
    updates: int = DILEMMADATA_SUPERVISED_SMOKE_UPDATES,
    validation_limit: int = 8,
) -> dict[str, object]:
    guards, restore = _install_source_access_guards()
    try:
        return _run_supervised_smoke_guarded(
            repo_root=repo_root,
            expected_head=expected_head,
            raw_index_path=raw_index_path,
            raw_cache_root=raw_cache_root,
            target_index_path=target_index_path,
            target_cache_root=target_cache_root,
            split_manifest_path=split_manifest_path,
            output_root=output_root,
            output_dir=output_dir,
            updates=updates,
            validation_limit=validation_limit,
            source_guard_evidence=guards,
        )
    finally:
        restore()


def _validate_membership(
    value: object, *, split: str
) -> dict[str, object]:
    payload = _validate_fingerprinted(
        value, category="dilemmadata.smoke.membership_invalid"
    )
    records = payload.get("records")
    if (
        payload.get("contract_version")
        != DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION
        or payload.get("split") != split
        or payload.get("replacement") is not False
        or payload.get("split_manifest_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT
        or not isinstance(records, list)
        or len(records) < 2
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.membership_invalid"
        )
    identities = []
    dialects = set()
    for row in records:
        if not isinstance(row, dict):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.membership_invalid"
            )
        identity = (row.get("dataset_id"), row.get("piece_id"))
        binding = row.get("target_cache_binding")
        if (
            identity[0] != "dilemmadata"
            or not isinstance(identity[1], str)
            or not _is_sha256(row.get("component_fingerprint"))
            or row.get("dialect") not in {"an", "dlc"}
            or not isinstance(row.get("source_group_id"), str)
            or not row["source_group_id"]
            or not isinstance(row.get("lineage_group_id"), str)
            or not row["lineage_group_id"]
            or not isinstance(binding, dict)
            or not all(
                _is_sha256(binding.get(name))
                for name in (
                    "cache_identity_fingerprint",
                    "target_bundle_fingerprint",
                    "artifact_sha256",
                    "raw_cache_key",
                    "canonical_artifact_sha256",
                )
            )
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.membership_invalid"
            )
        identities.append(identity)
        dialects.add(row["dialect"])
    dialect_counts = {
        name: sum(row["dialect"] == name for row in records)
        for name in ("an", "dlc")
    }
    if (
        identities != sorted(identities)
        or len(identities) != len(set(identities))
        or dialects != {"an", "dlc"}
        or payload.get("dataset_counts") != {"dilemmadata": len(records)}
        or payload.get("dialect_counts") != dialect_counts
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.membership_dialect_or_replacement_invalid"
        )
    if split == "train":
        access = payload.get("target_artifact_access")
        if (
            payload.get("selection_may_read_labels") is not True
            or payload.get("selection_policy")
            != "lexicographic_minimum_train_target_coverage_v1"
            or payload.get("required_task_ids")
            != list(DILEMMADATA_ACTIVE_TASK_IDS)
            or payload.get("covered_task_ids") != list(DILEMMADATA_ACTIVE_TASK_IDS)
            or not isinstance(access, dict)
            or access.get("observed_splits") != ["train"]
            or not isinstance(access.get("artifact_read_count"), int)
            or access["artifact_read_count"] < len(records)
            or access.get("test_target_accessed") is not False
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.train_membership_invalid"
            )
    else:
        access = payload.get("target_artifact_access_during_selection")
        if (
            payload.get("selection_may_read_labels") is not False
            or payload.get("selection_policy")
            != "seed17_identity_component_rank_v1"
            or payload.get("requested_limit") != len(records)
            or not isinstance(access, dict)
            or access.get("artifact_read_count") != 0
            or access.get("validation_labels_read") is not False
            or access.get("test_target_accessed") is not False
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.validation_membership_invalid"
            )
    return payload


def _validate_evaluation_artifacts(
    train_priors: object,
    validation_report: object,
    *,
    train_membership_fingerprint: str,
    validation_membership_fingerprint: str,
) -> None:
    priors = _validate_fingerprinted(
        train_priors, category="dilemmadata.smoke.train_priors_invalid"
    )
    if (
        priors.get("source_split") != "train_only"
        or priors.get("train_membership_fingerprint")
        != train_membership_fingerprint
        or set(priors.get("tasks", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.train_priors_invalid"
        )
    evaluation = _validate_fingerprinted(
        validation_report,
        category="dilemmadata.smoke.validation_report_invalid",
    )
    if (
        evaluation.get("split") != "validation"
        or evaluation.get("membership_fingerprint")
        != validation_membership_fingerprint
        or evaluation.get("test_unlock_fingerprint") is not None
        or evaluation.get("train_prior_fingerprint")
        != train_priors.get("fingerprint")
        or set(evaluation.get("tasks", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.validation_report_invalid"
        )
    for task_id in DILEMMADATA_ACTIVE_TASK_IDS:
        metrics = evaluation["tasks"][task_id]
        if not isinstance(metrics, dict) or "undefined_reason" not in metrics:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.validation_metrics_incomplete", task_id
            )
        if metrics.get("available") is True:
            for name in (
                "nll",
                "top1_accuracy",
                "macro_f1",
                "balanced_accuracy",
            ):
                if not _finite(metrics.get(name)):
                    raise DilemmadataSupervisedSmokeError(
                        "dilemmadata.smoke.validation_metrics_non_finite",
                        f"{task_id}:{name}",
                    )
            if task_id.endswith(".quality") and not _finite(
                metrics.get("top3_accuracy")
            ):
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.validation_metrics_non_finite",
                    f"{task_id}:top3_accuracy",
                )
        if not isinstance(metrics.get("record_metrics"), dict) or not isinstance(
            metrics.get("component_metrics"), dict
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.validation_projection_missing", task_id
            )


def _validate_target_semantic_validation(
    value: object, *, observed_target_index_fingerprint: str
) -> dict[str, object]:
    payload = _validate_fingerprinted(
        value, category="dilemmadata.smoke.target_semantic_evidence_invalid"
    )
    full = payload.get("source_free_full_validation")
    if (
        payload.get("policy")
        != "stable_semantics_plus_observed_physical_index_v1"
        or payload.get("record_count")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT
        or payload.get("raw_index_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT
        or payload.get("metadata_index_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT
        or payload.get("aggregate_target_bundle_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT
        or payload.get("observed_target_cache_index_fingerprint")
        != observed_target_index_fingerprint
        or payload.get("known_observed_physical_index_fingerprints")
        != list(DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS)
        or payload.get("target_index_role")
        != "exact_run_resume_evaluation_binding_not_universal_semantic_identity"
        or payload.get("contract_versions") != _TARGET_CONTRACT_VERSIONS
        or not isinstance(full, dict)
        or full.get("index_self_fingerprint_verified") is not True
        or full.get("index_record_count_verified")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT
        or full.get("artifact_sha256_verified_count")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT
        or full.get("target_bundle_fingerprint_verified_count")
        != DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_semantic_evidence_invalid"
        )
    return payload


def _validate_cuda_replay_diagnostic(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.cuda_replay_evidence_invalid"
        )
    tasks = value.get("tasks")
    if (
        value.get("contract_version")
        != DILEMMADATA_CUDA_REPLAY_DIAGNOSTIC_VERSION
        or value.get("purpose")
        != "independent_cuda_amp_replay_not_target_leakage"
        or value.get("candidate_identities_exact") is not True
        or value.get("all_logits_finite") is not True
        or value.get("comparison_dtype") != "float32"
        or value.get("absolute_tolerance")
        != DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE
        or value.get("relative_tolerance")
        != DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE
        or value.get("minimum_cosine_similarity")
        != DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY
        or not isinstance(tasks, list)
        or [row.get("task_id") for row in tasks if isinstance(row, dict)]
        != list(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.cuda_replay_evidence_invalid"
        )
    for row in tasks:
        if (
            not isinstance(row, dict)
            or row.get("within_elementwise_tolerance") is not True
            or not _finite(row.get("max_absolute_difference_fp32"))
            or float(row["max_absolute_difference_fp32"]) < 0.0
            or not _finite(row.get("max_relative_difference_fp32"))
            or float(row["max_relative_difference_fp32"]) < 0.0
            or not _finite(row.get("cosine_similarity_fp32"))
            or float(row["cosine_similarity_fp32"])
            < DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.cuda_replay_evidence_invalid"
            )
    return value


def validate_smoke_report(report: object) -> dict[str, object]:
    payload = _validate_fingerprinted(
        report, category="dilemmadata.smoke.report_fingerprint_mismatch"
    )
    bindings = payload.get("bindings")
    hardware = payload.get("hardware")
    optimization = payload.get("optimization")
    checkpoint = payload.get("checkpoint")
    validation = payload.get("validation")
    lifecycle = payload.get("lifecycle")
    runtime = payload.get("runtime_access")
    claims = payload.get("claim_boundaries")
    excluded = payload.get("excluded_supervision")
    target_semantics = payload.get("target_semantic_validation")
    cuda_replay = payload.get("cuda_replay_diagnostic")
    configured_updates = (
        optimization.get("attempted_update_count", -1)
        if isinstance(optimization, dict)
        else -1
    )
    observed_target_index = (
        bindings.get("observed_target_cache_index_fingerprint")
        if isinstance(bindings, dict)
        else None
    )
    if (
        payload.get("contract_version")
        != DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION
        or payload.get("phase") != DILEMMADATA_SUPERVISED_SMOKE_PHASE
        or not _is_git_sha(payload.get("expected_head"))
        or payload.get("active_task_ids") != list(DILEMMADATA_ACTIVE_TASK_IDS)
        or payload.get("model_contract_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT
        or not isinstance(bindings, dict)
        or set(bindings)
        != {
            "raw_index_fingerprint",
            "observed_target_cache_index_fingerprint",
            "split_manifest_fingerprint",
            "target_semantic_projection_fingerprint",
        }
        or bindings.get("raw_index_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT
        or not _is_sha256(observed_target_index)
        or bindings.get("split_manifest_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT
        or payload.get("training_config") != _training_config(configured_updates)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.report_contract_mismatch"
        )
    validated_target_semantics = _validate_target_semantic_validation(
        target_semantics,
        observed_target_index_fingerprint=str(observed_target_index),
    )
    _validate_cuda_replay_diagnostic(cuda_replay)
    if (
        bindings["target_semantic_projection_fingerprint"]
        != _fingerprint(validated_target_semantics)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.target_semantic_binding_mismatch"
        )
    if (
        not isinstance(hardware, dict)
        or hardware.get("accelerator") != DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME
        or hardware.get("logical_cuda_index") != 0
        or hardware.get("device") != "cuda:0"
        or hardware.get("amp_enabled") is not True
        or hardware.get("amp_dtype") != "float16"
        or hardware.get("grad_scaler_enabled") is not True
        or hardware.get("cpu_fallback") is not False
        or not isinstance(hardware.get("peak_allocated_bytes"), int)
        or hardware["peak_allocated_bytes"] <= 0
        or not isinstance(hardware.get("peak_reserved_bytes"), int)
        or hardware["peak_reserved_bytes"] <= 0
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.hardware_evidence_invalid"
        )
    if not isinstance(optimization, dict):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.optimization_evidence_invalid"
        )
    attempted = optimization.get("attempted_update_count")
    applied = optimization.get("applied_update_count")
    skipped = optimization.get("skipped_update_count")
    curve = optimization.get("curve")
    groups = optimization.get("aggregate_nonzero_gradient_by_group")
    changes = optimization.get("parameter_changes")
    expected_groups = {"raw_encoder", *DILEMMADATA_ACTIVE_TASK_IDS}
    if (
        not isinstance(attempted, int)
        or not 10 <= attempted <= 20
        or not isinstance(applied, int)
        or applied < 1
        or not isinstance(skipped, int)
        or applied + skipped != attempted
        or optimization.get("all_losses_finite") is not True
        or optimization.get("all_gradients_finite") is not True
        or not isinstance(curve, list)
        or len(curve) != attempted
        or not isinstance(groups, dict)
        or set(groups) != expected_groups
        or not all(value is True for value in groups.values())
        or not isinstance(changes, dict)
        or set(changes) != expected_groups
        or not all(
            isinstance(row, dict) and row.get("changed") is True
            for row in changes.values()
        )
        or sum(
            row.get("optimizer_step_applied") is True
            for row in curve
            if isinstance(row, dict)
        )
        != applied
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.optimization_evidence_invalid"
        )
    for row in curve:
        if (
            not isinstance(row, dict)
            or not _finite(row.get("total_loss"))
            or not isinstance(row.get("task_losses"), dict)
            or set(row["task_losses"]) != set(DILEMMADATA_ACTIVE_TASK_IDS)
            or not all(_finite(value) for value in row["task_losses"].values())
            or not _finite(row.get("gradient_norm_before_clip"))
            or row.get("gradients", {}).get("all_gradients_finite") is not True
            or not _finite(row.get("amp_scale_before"))
            or not _finite(row.get("amp_scale_after"))
            or not isinstance(row.get("optimizer_step_applied"), bool)
            or not _finite(row.get("learning_rate_after"))
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.optimization_curve_invalid"
            )
    if (
        not _finite(optimization.get("initial_loss"))
        or not _finite(optimization.get("minimum_loss"))
        or not _finite(optimization.get("final_loss"))
        or optimization["initial_loss"] != curve[0]["total_loss"]
        or optimization["final_loss"] != curve[-1]["total_loss"]
        or optimization["minimum_loss"]
        != min(float(row["total_loss"]) for row in curve)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.optimization_summary_invalid"
        )
    if (
        not isinstance(excluded, dict)
        or excluded.get("positive_unlabeled_task_ids") != list(DILEMMADATA_PU_TASK_IDS)
        or excluded.get("positive_unlabeled_ce_heads") != 0
        or excluded.get("positive_unlabeled_ce_losses") != 0
        or excluded.get("open_string_task_ids") != list(DILEMMADATA_OPEN_TASK_IDS)
        or excluded.get("open_string_heads") != 0
        or excluded.get("open_string_losses") != 0
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.excluded_supervision_invalid"
        )
    candidate = payload.get("candidate_first")
    reduction = payload.get("source_entry_reduction")
    target_mutation = (
        candidate.get("target_mutation") if isinstance(candidate, dict) else None
    )
    if (
        not isinstance(candidate, dict)
        or candidate.get("prediction_completed_before_target_join") is not True
        or candidate.get("target_columns_read_only_after_raw_prediction") is not True
        or candidate.get("prediction_object_exact_after_target_joins") is not True
        or not isinstance(target_mutation, dict)
        or target_mutation.get("verified") is not True
        or target_mutation.get("raw_prediction_call_count") != 1
        or target_mutation.get("same_prediction_object_for_both_joins") is not True
        or target_mutation.get(
            "tensor_storage_and_values_exact_after_original_join"
        ) is not True
        or target_mutation.get(
            "tensor_storage_and_values_exact_after_mutated_join"
        ) is not True
        or target_mutation.get(
            "candidate_identity_fingerprint_before"
        )
        != target_mutation.get(
            "candidate_identity_fingerprint_after"
        )
        or target_mutation.get("raw_only_logits_fingerprint_before")
        != target_mutation.get("raw_only_logits_fingerprint_after")
        or not _is_sha256(target_mutation.get("original_target_fingerprint"))
        or not _is_sha256(target_mutation.get("mutated_target_fingerprint"))
        or target_mutation.get("original_target_fingerprint")
        == target_mutation.get("mutated_target_fingerprint")
        or not isinstance(target_mutation.get("original_supervision_loss"), dict)
        or not isinstance(target_mutation.get("mutated_supervision_loss"), dict)
        or target_mutation.get("original_supervision_loss")
        == target_mutation.get("mutated_supervision_loss")
        or not isinstance(reduction, dict)
        or reduction.get("verified") is not True
        or set(reduction.get("tasks", {})) != set(DILEMMADATA_ACTIVE_TASK_IDS)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.candidate_or_reduction_evidence_invalid"
        )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("model_contract_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT
        or checkpoint.get("raw_index_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT
        or checkpoint.get("observed_target_cache_index_fingerprint")
        != observed_target_index
        or checkpoint.get("split_manifest_fingerprint")
        != DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT
        or checkpoint.get("active_task_ids") != list(DILEMMADATA_ACTIVE_TASK_IDS)
        or checkpoint.get("seed") != DILEMMADATA_SUPERVISED_SMOKE_SEED
        or checkpoint.get("cuda_device") != "cuda:0"
        or checkpoint.get("amp_dtype") != "float16"
        or checkpoint.get("grad_scaler_state_present") is not True
        or checkpoint.get("optimizer_state_present") is not True
        or checkpoint.get("scheduler_state_present") is not True
        or checkpoint.get("scratch_loaded_encoder_tensors") != []
        or checkpoint.get("scratch_supervised_heads_transferred") is not False
        or checkpoint.get("scratch_ssl_heads_transferred") is not False
        or checkpoint.get("reload_logits_bounded_cuda_replay") is not True
        or not isinstance(checkpoint.get("reload_model_state"), dict)
        or checkpoint["reload_model_state"].get("model_state_tensors_exact")
        is not True
        or not isinstance(checkpoint["reload_model_state"].get("tensor_count"), int)
        or checkpoint["reload_model_state"]["tensor_count"] <= 0
        or not _is_sha256(
            checkpoint["reload_model_state"].get("state_fingerprint")
        )
        or not _is_sha256(checkpoint.get("sha256"))
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_evidence_invalid"
        )
    _validate_cuda_replay_diagnostic(
        checkpoint.get("reload_cuda_replay_diagnostic")
    )
    if (
        not isinstance(validation, dict)
        or validation.get("split") != "validation"
        or validation.get("official_evaluator") is not True
        or validation.get("selection_uses_labels") is not False
        or validation.get("replacement") is not False
        or validation.get("observed_target_cache_index_fingerprint")
        != observed_target_index
        or any(
            validation.get(name) is not False
            for name in (
                "test_split_accessed",
                "test_targets_accessed",
                "test_metrics_computed",
                "test_unlock_used",
            )
        )
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.validation_evidence_invalid"
        )
    if (
        not isinstance(lifecycle, dict)
        or lifecycle.get("retained_prediction_tensor_count") != 0
        or lifecycle.get("allocated_bytes_after_cleanup") != 0
        or lifecycle.get("retained_cuda_tensor_count") != 0
        or not isinstance(runtime, dict)
        or runtime.get("source_tsv_path_accepted") is not False
        or runtime.get("raw_adapter_called") is not False
        or runtime.get("alignment_oracle_called") is not False
        or not isinstance(runtime.get("source_access_guard"), dict)
        or runtime["source_access_guard"].get("forbidden_call_count") != 0
        or runtime["source_access_guard"].get("guarded_functions")
        != list(_GUARDED_SOURCE_FUNCTIONS)
        or not isinstance(claims, dict)
        or claims.get("bounded_mechanics_only") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "scratch_vs_ssl_comparison",
                "representation_quality_claim",
                "calibration_or_significance_claim",
                "long_training_executed",
                "test_split_opened",
                "phase9c_started",
                "pdmx_started",
                "phase10_started",
                "legacy_used",
            )
        )
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.lifecycle_or_claim_evidence_invalid"
        )
    return payload


def _validate_checkpoint(
    evidence_dir: Path,
    report: Mapping[str, object],
) -> None:
    checkpoint_path = evidence_dir / "checkpoint.pt"
    digest = _sha256_file(checkpoint_path)
    sidecar = (evidence_dir / "checkpoint.pt.sha256").read_text(
        encoding="utf-8"
    )
    if sidecar != digest + "\n" or report["checkpoint"]["sha256"] != digest:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_sha_mismatch"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_unreadable", str(exc)
        ) from exc
    expected_keys = {
        "metadata",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "next_epoch",
        "best_validation_loss",
        "committed_metric_rows",
        "rng_state",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_fields_invalid"
        )
    metadata = payload.get("metadata")
    training_config = report.get("training_config")
    bindings = report.get("bindings")
    data_fingerprints = {
        **bindings,
        "train_membership_fingerprint": report["train_membership_fingerprint"],
        "validation_membership_fingerprint": report[
            "validation_membership_fingerprint"
        ],
    }
    if (
        not isinstance(metadata, dict)
        or metadata.get("training_checkpoint_version") != "1.0.0"
        or metadata.get("resolved_config_fingerprint") != _fingerprint(training_config)
        or metadata.get("data_fingerprints") != data_fingerprints
        or metadata.get("data_fingerprint") != _fingerprint(data_fingerprints)
        or metadata.get("model_contract", {}).get("active_task_ids")
        != list(DILEMMADATA_ACTIVE_TASK_IDS)
        or metadata.get("model_contract", {}).get("pu_tasks_without_heads")
        != list(DILEMMADATA_PU_TASK_IDS)
        or metadata.get("model_contract", {}).get("open_tasks_without_heads")
        != list(DILEMMADATA_OPEN_TASK_IDS)
        or payload.get("next_epoch") != 0
        or payload.get("best_validation_loss") is not None
        or payload.get("committed_metric_rows") != 0
        or not isinstance(payload.get("optimizer_state"), dict)
        or not payload["optimizer_state"].get("state")
        or not isinstance(payload.get("scheduler_state"), dict)
        or not isinstance(payload.get("scaler_state"), dict)
        or not payload["scaler_state"]
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_binding_invalid"
        )
    model = DilemmadataHierarchicalModel()
    if dilemmadata_model_contract_fingerprint(model) != report[
        "model_contract_fingerprint"
    ]:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_model_contract_invalid"
        )
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except Exception as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_model_state_invalid", str(exc)
        ) from exc
    model_state_rows = [
        {"name": name, "fingerprint": _tensor_fingerprint(value)}
        for name, value in payload["model_state"].items()
    ]
    reload_model_state = report["checkpoint"]["reload_model_state"]
    if (
        reload_model_state["tensor_count"] != len(model_state_rows)
        or reload_model_state["state_fingerprint"]
        != _fingerprint(model_state_rows)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_model_state_evidence_mismatch"
        )
    head_roots = {
        name.split(".", 3)[2]
        for name in payload["model_state"]
        if name.startswith("task_heads.heads.")
    }
    if head_roots != {f"task_{index:02d}" for index in range(4)}:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.checkpoint_head_inventory_invalid"
        )


def _safe_evidence_files(root: Path, expected: Sequence[str]) -> dict[str, Path]:
    try:
        if root.is_symlink() or not root.is_dir():
            raise OSError("evidence root is not a regular directory")
        entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.evidence_root_invalid", str(exc)
        ) from exc
    if tuple(path.name for path in entries) != tuple(sorted(expected)):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.evidence_inventory_invalid",
            str(tuple(path.name for path in entries)),
        )
    result = {}
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.evidence_special_file", path.name
            )
        result[path.name] = path
    return result


def seal_evidence_directory(evidence_dir: Path, *, expected_head: str) -> dict[str, object]:
    """Seal the closed evidence inventory after execution.log is final."""

    _require_git_sha(expected_head, "dilemmadata.smoke.expected_head_invalid")
    files = _safe_evidence_files(evidence_dir, _EVIDENCE_ARTIFACTS)
    manifest = {
        "bundle_version": DILEMMADATA_SUPERVISED_SMOKE_BUNDLE_VERSION,
        "contract_version": DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION,
        "phase": DILEMMADATA_SUPERVISED_SMOKE_PHASE,
        "expected_head": expected_head,
        "artifact_count": len(files),
        "artifacts": {
            name: {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(files.items())
        },
    }
    sealed = _with_fingerprint(manifest)
    _write_json_atomic(evidence_dir / "artifact_manifest.json", sealed)
    return sealed


def verify_evidence_directory(
    evidence_dir: Path,
    *,
    expected_head: str,
    require_current_hardware: bool = True,
) -> dict[str, object]:
    """Independently verify a sealed source-free evidence directory."""

    _require_git_sha(expected_head, "dilemmadata.smoke.expected_head_invalid")
    files = _safe_evidence_files(evidence_dir, _SEALED_ARTIFACTS)
    manifest = _load_json(
        files["artifact_manifest.json"],
        "dilemmadata.smoke.artifact_manifest_unreadable",
    )
    manifest_payload = _validate_fingerprinted(
        manifest, category="dilemmadata.smoke.artifact_manifest_invalid"
    )
    artifacts = manifest_payload.get("artifacts")
    if (
        manifest_payload.get("bundle_version")
        != DILEMMADATA_SUPERVISED_SMOKE_BUNDLE_VERSION
        or manifest_payload.get("contract_version")
        != DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION
        or manifest_payload.get("phase") != DILEMMADATA_SUPERVISED_SMOKE_PHASE
        or manifest_payload.get("expected_head") != expected_head
        or manifest_payload.get("artifact_count") != len(_EVIDENCE_ARTIFACTS)
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(_EVIDENCE_ARTIFACTS)
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.artifact_manifest_invalid"
        )
    for name in _EVIDENCE_ARTIFACTS:
        row = artifacts[name]
        if (
            not isinstance(row, dict)
            or row.get("sha256") != _sha256_file(files[name])
            or row.get("size_bytes") != files[name].stat().st_size
        ):
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.artifact_sha_mismatch", name
            )
    report = _load_json(
        files["run_report.json"], "dilemmadata.smoke.report_unreadable"
    )
    validate_smoke_report(report)
    if report.get("expected_head") != expected_head:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.report_head_mismatch"
        )
    train_membership = _load_json(
        files["train_membership.json"],
        "dilemmadata.smoke.train_membership_unreadable",
    )
    validation_membership = _load_json(
        files["validation_membership.json"],
        "dilemmadata.smoke.validation_membership_unreadable",
    )
    _validate_membership(train_membership, split="train")
    _validate_membership(validation_membership, split="validation")
    if (
        report.get("train_membership_fingerprint")
        != train_membership.get("fingerprint")
        or report.get("validation_membership_fingerprint")
        != validation_membership.get("fingerprint")
        or report["checkpoint"].get("train_membership_fingerprint")
        != train_membership.get("fingerprint")
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.membership_binding_mismatch"
        )
    train_ids = {
        (row["dataset_id"], row["piece_id"])
        for row in train_membership["records"]
    }
    validation_ids = {
        (row["dataset_id"], row["piece_id"])
        for row in validation_membership["records"]
    }
    if train_ids & validation_ids:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.train_validation_overlap"
        )
    train_priors = _load_json(
        files["train_priors.json"], "dilemmadata.smoke.train_priors_unreadable"
    )
    validation_report = _load_json(
        files["validation_report.json"],
        "dilemmadata.smoke.validation_report_unreadable",
    )
    _validate_evaluation_artifacts(
        train_priors,
        validation_report,
        train_membership_fingerprint=str(train_membership["fingerprint"]),
        validation_membership_fingerprint=str(validation_membership["fingerprint"]),
    )
    if (
        report["validation"].get("report_fingerprint")
        != validation_report.get("fingerprint")
        or report["validation"].get("train_only_baseline_fingerprint")
        != train_priors.get("fingerprint")
    ):
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.validation_binding_mismatch"
        )
    _validate_checkpoint(evidence_dir, report)
    if require_current_hardware:
        _, hardware = _cuda_preflight()
        if hardware["accelerator"] != report["hardware"]["accelerator"]:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.verifier_hardware_mismatch"
            )
    return manifest


def pack_evidence_bundle(
    evidence_dir: Path,
    *,
    tar_path: Path,
    sidecar_path: Path,
    expected_head: str,
    require_current_hardware: bool = True,
) -> str:
    """Create a deterministic regular-file-only tar and SHA-256 sidecar."""

    verify_evidence_directory(
        evidence_dir,
        expected_head=expected_head,
        require_current_hardware=require_current_hardware,
    )
    if tar_path.exists() or sidecar_path.exists():
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.output_collision"
        )
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{tar_path.name}.", suffix=".partial", dir=tar_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(evidence_dir.iterdir(), key=lambda item: item.name):
                data = path.read_bytes()
                information = tarfile.TarInfo(f"phase9b2c-evidence/{path.name}")
                information.size = len(data)
                information.mode = 0o644
                information.uid = information.gid = 0
                information.uname = information.gname = ""
                information.mtime = 0
                archive.addfile(information, io.BytesIO(data))
        os.replace(temporary, tar_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = _sha256_file(tar_path)
    try:
        _write_text_atomic(sidecar_path, digest + "\n")
    except BaseException:
        tar_path.unlink(missing_ok=True)
        raise
    return digest


def verify_evidence_bundle(
    tar_path: Path,
    sidecar_path: Path,
    *,
    expected_head: str,
    require_current_hardware: bool = True,
) -> dict[str, object]:
    """Verify sidecar, safe tar inventory and extracted sealed evidence."""

    tar_path = _require_regular(
        tar_path, directory=False, category="dilemmadata.smoke.bundle_invalid"
    )
    sidecar_path = _require_regular(
        sidecar_path, directory=False, category="dilemmadata.smoke.sidecar_invalid"
    )
    sidecar = sidecar_path.read_text(encoding="ascii")
    if sidecar != _sha256_file(tar_path) + "\n":
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.bundle_sha_mismatch"
        )
    try:
        archive = tarfile.open(tar_path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise DilemmadataSupervisedSmokeError(
            "dilemmadata.smoke.bundle_unreadable", str(exc)
        ) from exc
    with archive, tempfile.TemporaryDirectory(prefix="phase9b2c-verify-") as name:
        root = Path(name) / "phase9b2c-evidence"
        root.mkdir()
        members = archive.getmembers()
        expected_names = {
            f"phase9b2c-evidence/{artifact}" for artifact in _SEALED_ARTIFACTS
        }
        observed = {member.name for member in members}
        if len(observed) != len(members) or observed != expected_names:
            raise DilemmadataSupervisedSmokeError(
                "dilemmadata.smoke.bundle_inventory_invalid"
            )
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or member.name != path.as_posix()
                or len(path.parts) != 2
                or path.parts[0] != "phase9b2c-evidence"
                or path.parts[1] in {"", ".", ".."}
                or member.size < 0
                or member.size > _MAX_BUNDLE_FILE_BYTES
            ):
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.bundle_unsafe_member", member.name
                )
            total += member.size
            if total > _MAX_BUNDLE_TOTAL_BYTES:
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.bundle_size_limit"
                )
            source = archive.extractfile(member)
            if source is None:
                raise DilemmadataSupervisedSmokeError(
                    "dilemmadata.smoke.bundle_member_unreadable", member.name
                )
            destination = root / path.name
            with destination.open("xb") as handle:
                copied = 0
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > member.size:
                        raise DilemmadataSupervisedSmokeError(
                            "dilemmadata.smoke.bundle_member_size_mismatch"
                        )
                    handle.write(block)
                if copied != member.size:
                    raise DilemmadataSupervisedSmokeError(
                        "dilemmadata.smoke.bundle_member_size_mismatch"
                    )
        return verify_evidence_directory(
            root,
            expected_head=expected_head,
            require_current_hardware=require_current_hardware,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--expected-head", required=True)
    run.add_argument("--raw-index", type=Path, required=True)
    run.add_argument("--raw-cache-root", type=Path, required=True)
    run.add_argument("--target-index", type=Path, required=True)
    run.add_argument("--target-cache-root", type=Path, required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--updates", type=int, default=DILEMMADATA_SUPERVISED_SMOKE_UPDATES)
    run.add_argument("--validation-limit", type=int, default=8)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--evidence-dir", type=Path, required=True)
    seal.add_argument("--expected-head", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--evidence-dir", type=Path, required=True)
    pack.add_argument("--tar", type=Path, required=True)
    pack.add_argument("--sidecar", type=Path, required=True)
    pack.add_argument("--expected-head", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            result = run_supervised_smoke(
                repo_root=arguments.repo_root,
                expected_head=arguments.expected_head,
                raw_index_path=arguments.raw_index,
                raw_cache_root=arguments.raw_cache_root,
                target_index_path=arguments.target_index,
                target_cache_root=arguments.target_cache_root,
                split_manifest_path=arguments.split_manifest,
                output_root=arguments.output_root,
                output_dir=arguments.output_dir,
                updates=arguments.updates,
                validation_limit=arguments.validation_limit,
            )
            print(result["fingerprint"])
        elif arguments.command == "seal":
            result = seal_evidence_directory(
                arguments.evidence_dir, expected_head=arguments.expected_head
            )
            print(result["fingerprint"])
        else:
            print(
                pack_evidence_bundle(
                    arguments.evidence_dir,
                    tar_path=arguments.tar,
                    sidecar_path=arguments.sidecar,
                    expected_head=arguments.expected_head,
                    require_current_hardware=True,
                )
            )
    except DilemmadataSupervisedSmokeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE",
    "DILEMMADATA_CUDA_REPLAY_DIAGNOSTIC_VERSION",
    "DILEMMADATA_CUDA_REPLAY_MINIMUM_COSINE_SIMILARITY",
    "DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE",
    "DILEMMADATA_SUPERVISED_SMOKE_BUNDLE_VERSION",
    "DILEMMADATA_SUPERVISED_SMOKE_CONTRACT_VERSION",
    "DILEMMADATA_SUPERVISED_SMOKE_GPU_NAME",
    "DILEMMADATA_SUPERVISED_SMOKE_LEARNING_RATE",
    "DILEMMADATA_SUPERVISED_SMOKE_LOCAL_TARGET_INDEX_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_MODEL_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_PHASE",
    "DILEMMADATA_SUPERVISED_SMOKE_RAW_INDEX_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_SEED",
    "DILEMMADATA_SUPERVISED_SMOKE_SPLIT_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_OBSERVED_TARGET_INDEX_FINGERPRINTS",
    "DILEMMADATA_SUPERVISED_SMOKE_RTX_TARGET_INDEX_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_TARGET_BUNDLE_AGGREGATE_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_TARGET_INDEX_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_TARGET_METADATA_FINGERPRINT",
    "DILEMMADATA_SUPERVISED_SMOKE_TARGET_RECORD_COUNT",
    "DILEMMADATA_SUPERVISED_SMOKE_UPDATES",
    "DilemmadataSupervisedSmokeError",
    "pack_evidence_bundle",
    "run_supervised_smoke",
    "seal_evidence_directory",
    "validate_git_preflight",
    "validate_smoke_report",
    "verify_evidence_bundle",
    "verify_evidence_directory",
]
