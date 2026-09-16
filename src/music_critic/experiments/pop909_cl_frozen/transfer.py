"""Explicit TRAIN/VALIDATION transfer; the original synthetic API stays closed.

Preflight reads only bound index/split metadata. MIDI and checkpoint bytes are
opened only by execution after a matching planned experiment is verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import platform
import re
from time import perf_counter

import torch

from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig, Pop909ClCorpusRecord, Pop909ClQuarantine,
    convert_pop909_cl_file, pop909_cl_piece_id, pop909_cl_raw_input_group_id,
    pop909_lineage_group_id,
)
from music_critic.adapters.pop909_cl_frozen_source import read_raw_source
from music_critic.experiments.registry import load_record
from music_critic.tasks import loads_corpus_index, loads_split_manifest, validate_split_manifest

from .checkpoint import load_model_checkpoint
from .contracts import canonical_json, evaluation_protocol, fingerprint
from .evaluation import EvaluationPiece, _evaluate_core, targets_from_chord_evidence

REQUEST_SCHEMA = "Pop909ClTransferRequest@1.0.0"
PREFLIGHT_SCHEMA = "Pop909ClTransferPreflight@1.0.0"
REPORT_SCHEMA = "Pop909ClTransferReport@1.0.0"


def guard_transfer(split: str) -> None:
    if isinstance(split, str) and split.lower() == "test":
        raise ValueError("pop909_transfer.test_access_forbidden")
    if split not in ("train", "validation"):
        raise ValueError("pop909_transfer.train_or_validation_required")


def transfer_protocol() -> dict:
    original = evaluation_protocol()
    payload = {
        "schema_version": "Pop909ClTransferProtocol@1.0.0",
        "based_on_synthetic_protocol_fingerprint": original["fingerprint"],
        "metric_policy": {key: value for key, value in original.items()
                          if key not in {"schema_version", "fingerprint", "execution_policy"}},
        "execution_policy": "explicit_registered_train_validation_cpu_only",
        "membership": "explicit_allowlist_bound_to_complete_validated_global_split",
        "selection": "no_training_no_checkpoint_selection_no_pop_tuning",
        "raw_source": "hash_bound_channel0_projection_before_chord_normalization",
        "target_reader": "after_both_views_recheck_same_source_and_projection",
        "corpus_discovery": False,
        "model_selection": False,
        "claim": "harmonic_domain_transfer_not_music_quality_or_all_18_tasks",
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (not isinstance(self.path, Path) or not self.path.is_absolute()
                or not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
                or type(self.size_bytes) is not int or self.size_bytes <= 0):
            raise ValueError("pop909_transfer.invalid_file_binding")

    def identity(self) -> dict:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    def read(self) -> bytes:
        payload = self.path.read_bytes()
        if len(payload) != self.size_bytes or sha256(payload).hexdigest() != self.sha256:
            raise ValueError("pop909_transfer.artifact_binding_mismatch")
        return payload


@dataclass(frozen=True)
class TransferRequest:
    split: str
    indices: tuple[BoundFile, ...]
    split_manifest: BoundFile
    checkpoint: BoundFile
    source_root: Path
    piece_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        guard_transfer(self.split)
        if (not isinstance(self.indices, tuple) or not self.indices
                or any(not isinstance(item, BoundFile) for item in self.indices)
                or not isinstance(self.split_manifest, BoundFile)
                or not isinstance(self.checkpoint, BoundFile)
                or not isinstance(self.source_root, Path) or not self.source_root.is_absolute()
                or not isinstance(self.piece_ids, tuple) or not self.piece_ids
                or any(not _nonempty(item) for item in self.piece_ids)
                or len(set(self.piece_ids)) != len(self.piece_ids)):
            raise ValueError("pop909_transfer.invalid_explicit_request")


def _unique_json(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("pop909_transfer.duplicate_json_key")
        result[key] = value
    return result


def load_request(path: Path, *, split: str) -> TransferRequest:
    guard_transfer(split)  # Even the request file is unopened for TEST.
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
    fields = {"schema_version", "split", "indices", "split_manifest", "checkpoint", "source_root", "piece_ids"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema_version"] != REQUEST_SCHEMA or value["split"] != split):
        raise ValueError("pop909_transfer.request_schema_or_split_mismatch")

    def bound(item):
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise ValueError("pop909_transfer.invalid_file_binding")
        return BoundFile(Path(item["path"]), item["sha256"], item["size_bytes"])

    if not isinstance(value["indices"], list) or not isinstance(value["piece_ids"], list):
        raise ValueError("pop909_transfer.request_lists_required")
    return TransferRequest(split, tuple(bound(item) for item in value["indices"]),
                           bound(value["split_manifest"]), bound(value["checkpoint"]),
                           Path(value["source_root"]), tuple(value["piece_ids"]))


def preflight(request: TransferRequest) -> dict:
    """Validate complete split closure without MIDI/cache/checkpoint reads."""
    guard_transfer(request.split)
    indices = tuple(loads_corpus_index(item.read().decode("utf-8")) for item in request.indices)
    split = loads_split_manifest(request.split_manifest.read().decode("utf-8"))
    validate_split_manifest(split, indices)
    pop = [index for index in indices if index.header.dataset_id == "pop909_cl"]
    if len(pop) != 1:
        raise ValueError("pop909_transfer.one_pop_index_required")
    records = {row.piece_id: row for row in pop[0].records}
    assignments = {row.piece_id: row for row in split.assignments if row.dataset_id == "pop909_cl"}
    selected = []
    for piece_id in sorted(request.piece_ids):
        if piece_id not in records or assignments[piece_id].split != request.split:
            raise ValueError("pop909_transfer.selected_piece_not_in_requested_split")
        row, assignment = records[piece_id], assignments[piece_id]
        path = PurePosixPath(row.source_relative_path)
        projection_sha = row.source_group_id.removeprefix("pop909-cl-score:")
        if (path.is_absolute() or ".." in path.parts or "\\" in row.source_relative_path
                or not re.fullmatch(r"[0-9]{3}", row.source_identity)
                or row.source_identity == "172"
                or row.piece_id != pop909_cl_piece_id(row.source_identity)
                or row.source_group_id != pop909_cl_raw_input_group_id(projection_sha)
                or row.lineage_group_id != pop909_lineage_group_id(row.source_identity)):
            raise ValueError("pop909_transfer.invalid_pop_source_identity")
        selected.append({
            "piece_id": piece_id, "song_id": row.source_identity,
            "source_relative_path": row.source_relative_path, "source_sha256": row.source_sha256,
            "score_projection_sha256": projection_sha, "source_group_id": row.source_group_id,
            "lineage_group_id": row.lineage_group_id, "component_fingerprint": assignment.component_fingerprint,
        })
    payload = {
        "schema_version": PREFLIGHT_SCHEMA, "protocol": transfer_protocol(), "split": request.split,
        "indices": sorted((item.identity() for item in request.indices), key=lambda item: item["sha256"]),
        "split_artifact": request.split_manifest.identity(), "split_fingerprint": split.manifest_fingerprint,
        "index_fingerprints": dict(split.index_fingerprints), "checkpoint": request.checkpoint.identity(),
        "selected_records": selected, "selected_piece_count": len(selected),
        "selected_component_count": len({row["component_fingerprint"] for row in selected}),
        "available_split_piece_count": sum(row.split == request.split for row in assignments.values()),
        "access": "index_and_split_metadata_only_no_source_cache_target_or_checkpoint_bytes",
        "checkpoint_bytes_verified": False, "source_bytes_verified": False,
        "overlap_with_dilemmadata": "not_a_cross_corpus_duplicate_audit",
    }
    return {**payload, "binding_fingerprint": fingerprint(payload)}


def _registered_record(path: Path, binding: dict) -> dict:
    record = load_record(path)
    if (record["status"] != "planned"
            or record.get("protocol", {}).get("pop909_transfer_binding_fingerprint") != binding["binding_fingerprint"]):
        raise ValueError("pop909_transfer.matching_planned_record_required")
    policy = record["data_access"].get("declared_policy", {})
    if (any(policy.get(key) is not False for key in (
            "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed"))
            or (binding["split"] == "validation" and policy.get("validation_allowed") is not True)):
        raise ValueError("pop909_transfer.registered_access_policy_mismatch")
    return record


def _inputs(request: TransferRequest, binding: dict):
    root = request.source_root.resolve()
    for row in binding["selected_records"]:
        path = (root / row["source_relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("pop909_transfer.source_path_escapes_root")
        record = Pop909ClCorpusRecord(
            song_id=row["song_id"], path=path, relative_path=row["source_relative_path"],
            corpus_relative_path=Path(row["source_relative_path"]).name,
            sha256=row["source_sha256"], source_group_id=row["source_group_id"],
            lineage_group_id=row["lineage_group_id"],
        )
        projection_sha = row["score_projection_sha256"]
        piece = read_raw_source(record, split=request.split, expected_projection_sha256=projection_sha)

        def read_targets(record=record, projection_sha=projection_sha):
            result = convert_pop909_cl_file(record, config=Pop909ClAdapterConfig(include_targets=False))
            if isinstance(result, Pop909ClQuarantine) or result.score_projection_sha256 != projection_sha:
                raise ValueError("pop909_transfer.target_source_projection_mismatch")
            return targets_from_chord_evidence(result.chord_evidence)

        yield EvaluationPiece(piece, row["component_fingerprint"], read_targets)


def evaluate_transfer(request: TransferRequest, *, experiment_record: Path) -> dict:
    guard_transfer(request.split)
    binding = preflight(request)
    record = _registered_record(experiment_record, binding)
    started = perf_counter()
    loaded = load_model_checkpoint(request.checkpoint.path, expected_sha256=request.checkpoint.sha256,
                                   expected_size_bytes=request.checkpoint.size_bytes)
    core = _evaluate_core(loaded.model, _inputs(request, binding), checkpoint_evidence=loaded.evidence)
    payload = {
        "schema_version": REPORT_SCHEMA, "protocol": transfer_protocol(), "split": request.split,
        "binding": binding, "experiment_id": record["experiment_id"],
        "planned_record_fingerprint": record["record_fingerprint"],
        "provenance": {"kind": "explicit_indexed_pop909_cl_source", "music_quality_evidence": False},
        "access_observations": {"source_piece_count": len(core["pieces"]),
                                "target_reader_calls": len(core["pieces"]),
                                "validation_accessed": request.split == "validation",
                                "test_inputs_accessed": False, "test_targets_accessed": False,
                                "test_metrics_computed": False, "test_used_for_selection": False},
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "threads": torch.get_num_threads(),
                    "wall_seconds": perf_counter() - started, "applied_updates": 0},
        **core,
    }
    # Scientific output identity excludes variable wall time, retained separately.
    payload["scientific_fingerprint"] = fingerprint({key: value for key, value in payload.items() if key != "runtime"})
    return {**payload, "report_fingerprint": fingerprint(payload)}
