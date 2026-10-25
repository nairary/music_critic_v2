"""Frozen, sequential Dilemmadata source binding for the bounded WS2 pilot.

Preparation is an explicit real-data operation. Metadata-only preflight never
opens a score, annotation, cache or checkpoint. Runtime reconstruction follows
the corrected reference resolver and adapter; no prepared/corpus cache is used.
The caller registers preparation/execution and supplies its resource wrapper.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.group_masking_pilot import (
    GroupPilotConfig, PilotBatch, POLICIES, bind_train_batch, paired_mask_bindings,
)
from music_critic.ssl.hierarchical_masking import HierarchyMaskPolicyConfig
from music_critic.ssl.masking import prepare_hierarchy_mask_binding

REQUEST_SCHEMA = "WS2GroupMaskingScientificDataRequest@1.0.0"
BINDING_SCHEMA = "WS2GroupMaskingScientificDataBinding@1.0.0"
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024
_METADATA = frozenset({"source_inventory", "quarantine_records", "audit_summary",
                       "split_assignments", "paper_candidate_records", "dlc_metadata",
                       "class_weights"})


def _fail(code: str) -> None:
    raise ValueError("group_scientific_data." + code)


def guard_split(split: str) -> None:
    if isinstance(split, str) and split.casefold() == "test":
        _fail("test_access_forbidden")
    if split not in {"train", "validation"}:
        _fail("train_or_validation_required")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _json(payload: bytes):
    return json.loads(payload, object_pairs_hook=_unique,
                      parse_constant=lambda _: _fail("nonfinite_json"))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(value):
    return (isinstance(value, str) and value != "" and value == value.strip()
            and not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _positive(value, maximum):
    return type(value) is int and 0 < value <= maximum


def _stat(path: Path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        _fail("regular_file_required")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self):
        if (not isinstance(self.path, Path) or not self.path.is_absolute()
                or not _sha(self.sha256)
                or not _positive(self.size_bytes, MAX_METADATA_BYTES)):
            _fail("invalid_artifact_binding")

    def to_dict(self):
        return {"path": str(self.path), "sha256": self.sha256, "size_bytes": self.size_bytes}

    def read(self):
        before = _stat(self.path)
        if before[2] != self.size_bytes:
            _fail("artifact_size_mismatch")
        # Avoid following an exchanged final-path symlink, and bound the read.
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            payload = stream.read(self.size_bytes + 1)
            info = os.fstat(stream.fileno())
        after = _stat(self.path)
        if (before != after or (info.st_dev, info.st_ino) != before[:2]
                or len(payload) != self.size_bytes
                or sha256(payload).hexdigest() != self.sha256):
            _fail("artifact_binding_mismatch")
        return payload


def _bound(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail("invalid_artifact_binding")
    if not isinstance(value["path"], str):
        _fail("invalid_artifact_binding")
    return BoundFile(Path(value["path"]), value["sha256"], value["size_bytes"])


@dataclass(frozen=True)
class ScientificDataRequest:
    """An opaque, deep-copied request; all paths and hashes are caller supplied."""
    payload: dict

    def to_dict(self):
        return deepcopy(self.payload)


def load_data_request(path: str | Path, *, split: str = "train") -> ScientificDataRequest:
    guard_split(split)  # Even request bytes remain inaccessible for TEST.
    path = Path(path)
    if _stat(path)[2] > MAX_REQUEST_BYTES:
        _fail("request_too_large")
    with path.open("rb") as stream:
        value = _json(stream.read(MAX_REQUEST_BYTES + 1))
    request = ScientificDataRequest(value)
    _request(request)
    return request


def _request(request):
    if not isinstance(request, ScientificDataRequest):
        _fail("request_required")
    value = request.to_dict()
    if (not isinstance(value, dict) or set(value) != {
            "schema", "source_root", "metadata", "source_sizes", "pilot_schedule",
            "recovery_schedule", "validation_limits"} or value["schema"] != REQUEST_SCHEMA
            or not isinstance(value["source_root"], str)
            or not Path(value["source_root"]).is_absolute()
            or not isinstance(value["metadata"], dict) or set(value["metadata"]) != _METADATA
            or not isinstance(value["source_sizes"], dict)):
        _fail("request_schema_invalid")
    from music_critic.ssl.group_masking_scientific_eval import ValidationResourceLimits
    limits = value["validation_limits"]
    if not isinstance(limits, dict) or set(limits) != {
            "max_notes_per_piece", "max_nodes_per_piece", "max_edges_per_piece"}:
        _fail("validation_limits_required")
    ValidationResourceLimits(**limits)
    files = {key: _bound(item) for key, item in value["metadata"].items()}
    inventory_root = files["source_inventory"].path.parent
    b3_root = files["split_assignments"].path.parent
    expected_paths = {
        "source_inventory": inventory_root / "source_inventory.jsonl",
        "quarantine_records": inventory_root / "quarantine_records.jsonl",
        "audit_summary": inventory_root / "audit_summary.json",
        "split_assignments": b3_root / "split_assignments.jsonl",
        "paper_candidate_records": b3_root / "paper_candidate_records.jsonl",
        "dlc_metadata": Path(value["source_root"]) / "processing/DLC/distant_listening_corpus.metadata.tsv",
    }
    if any(files[name].path != path for name, path in expected_paths.items()):
        _fail("reference_metadata_layout_mismatch")
    for rows, keys in ((value["pilot_schedule"], {"ssl", "theory"}),
                       (value["recovery_schedule"], {"theory"})):
        if not isinstance(rows, list) or not rows or len(rows) > 1000:
            _fail("schedule_invalid")
        for row in rows:
            if not isinstance(row, dict) or set(row) != keys:
                _fail("schedule_invalid")
            for ids in row.values():
                if (not isinstance(ids, list) or not ids or len(ids) > 2
                        or any(not _identity(item) for item in ids)):
                    _fail("schedule_invalid")
    for record, sizes in value["source_sizes"].items():
        if (not _identity(record) or not isinstance(sizes, dict)
                or set(sizes) != {"annotation_size_bytes", "score_size_bytes"}
                or not _positive(sizes["annotation_size_bytes"], MAX_SOURCE_BYTES)
                or (sizes["score_size_bytes"] is not None
                    and not _positive(sizes["score_size_bytes"], MAX_SOURCE_BYTES))):
            _fail("source_size_binding_invalid")
    return value, files


def _rows(payload):
    rows = [_json(line) for line in payload.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        _fail("metadata_row_invalid")
    return rows


def _indexed(rows):
    result = {}
    for row in rows:
        record = row.get("record_id")
        if not _identity(record) or record in result:
            _fail("duplicate_or_invalid_record_id")
        result[record] = row
    return result


def _source_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in {".", ".."} for part in relative.split("/"))):
        _fail("source_path_invalid")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        _fail("source_path_escapes_root")
    return path


def preflight_data(request: ScientificDataRequest, pilot_config: GroupPilotConfig,
                   recovery_config: GroupPilotConfig, *, split: str = "train") -> dict:
    """Read complete bound metadata; never inspect selected source bytes."""
    guard_split(split)
    value, files = _request(request)
    if (len(value["pilot_schedule"]) != pilot_config.updates
            or len(value["recovery_schedule"]) != recovery_config.updates
            or pilot_config.lambda_ssl <= 0 or pilot_config.lambda_theory <= 0
            or recovery_config.lambda_ssl != 0 or recovery_config.lambda_theory <= 0):
        _fail("stage_config_or_budget_mismatch")
    for rows, config in ((value["pilot_schedule"], pilot_config),
                         (value["recovery_schedule"], recovery_config)):
        if any(len(ids) != config.max_samples_per_batch for row in rows for ids in row.values()):
            _fail("fixed_batch_size_mismatch")
    payloads = {name: item.read() for name, item in files.items()}
    if files["split_assignments"].sha256 != reference.CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
        _fail("frozen_split_sha_mismatch")
    assignments = _indexed(_rows(payloads["split_assignments"]))
    inventory = _indexed(_rows(payloads["source_inventory"]))
    paper = _indexed(_rows(payloads["paper_candidate_records"]))
    if set(assignments) != set(paper) or not set(assignments) <= set(inventory):
        _fail("complete_split_record_closure_mismatch")
    component_splits, piece_components, raw_components = {}, {}, {}
    for record, row in assignments.items():
        component, name = row.get("source_component_id"), row.get("split")
        candidate, source = paper[record], inventory[record]
        piece, raw = candidate.get("piece_id"), candidate.get("raw_equivalence_id")
        if (not _identity(component) or name not in {"train", "validation", "test"}
                or not _identity(piece) or not _identity(raw)
                or source.get("source_group_id") != component
                or candidate.get("source_component_id") != component):
            _fail("split_component_metadata_mismatch")
        if component in component_splits and component_splits[component] != name:
            _fail("component_cross_split_leakage")
        component_splits[component] = name
        for identity, mapping in ((piece, piece_components), (raw, raw_components)):
            if identity in mapping and mapping[identity] != component:
                _fail("identity_component_closure_mismatch")
            mapping[identity] = component
    validation = tuple(sorted(record for record, row in assignments.items() if row["split"] == "validation"))
    if not validation or not any(row["split"] == "train" for row in assignments.values()):
        _fail("train_validation_membership_empty")
    train = {item for rows in (value["pilot_schedule"], value["recovery_schedule"])
             for row in rows for ids in row.values() for item in ids}
    if any(record not in assignments or assignments[record]["split"] != "train" for record in train):
        _fail("scheduled_record_not_train")
    selected = train | set(validation)
    if set(value["source_sizes"]) != selected:
        _fail("selected_source_sizes_incomplete_or_extra")
    root = Path(value["source_root"])
    for record in selected:
        source, sizes = inventory[record], value["source_sizes"][record]
        _source_path(root, source.get("annotation_path"))
        score = source.get("score_path")
        if score is not None:
            _source_path(root, score)
        hashes = source.get("source_fingerprints", {})
        if (source.get("dialect") not in {"an_joint", "dlc"}
                or not all(_sha(hashes.get(key)) for key in (
                    "physical_annotation_sha256", "raw_projection_sha256", "grouping_sha256", "record_binding_sha256"))
                or (score is None) != (sizes["score_size_bytes"] is None)
                or (score is None and hashes.get("score_sha256") is not None)
                or (score is not None and not _sha(hashes.get("score_sha256")))):
            _fail("source_identity_binding_invalid")
    payload = {"schema": "WS2GroupMaskingScientificPreflight@1.0.0",
               "request_fingerprint": fingerprint(value),
               "pilot_config": pilot_config.contract(), "recovery_config": recovery_config.contract(),
               "validation_limits": value["validation_limits"],
               "metadata": {name: {"sha256": item.sha256, "size_bytes": item.size_bytes}
                            for name, item in files.items()},
               "split_record_counts": {name: sum(row["split"] == name for row in assignments.values())
                                       for name in ("train", "validation", "test")},
               "split_component_counts": {name: sum(split_name == name for split_name in component_splits.values())
                                          for name in ("train", "validation", "test")},
               "train_record_ids": sorted(train), "validation_record_ids": list(validation),
               "validation_membership": [["dilemmadata", paper[record]["piece_id"]] for record in validation],
               "source_bytes_verified": False, "test_source_access": False,
               "data_path": "reference_sequential_source_reconstruction_no_cache_no_transposition"}
    return {**payload, "fingerprint": fingerprint(payload)}


class BoundScientificData:
    """No graph retention, no sampling, no fallback and no TEST reader."""

    def __init__(self, request, pilot_config, recovery_config, *, binding=None, on_receipt=None):
        self.request = ScientificDataRequest(request.to_dict())
        self.configs = {"pilot": pilot_config, "recovery": recovery_config}
        self.preflight = preflight_data(self.request, pilot_config, recovery_config)
        self._value, self._files = _request(self.request)
        from music_critic.ssl.group_masking_scientific_eval import ValidationResourceLimits
        self.validation_resource_limits = ValidationResourceLimits(**self._value["validation_limits"])
        self._metadata_stats = {key: _stat(item.path) for key, item in self._files.items()}
        self._assignments = _indexed(_rows(self._files["split_assignments"].read()))
        self._inventory = _indexed(_rows(self._files["source_inventory"].read()))
        self._paper = _indexed(_rows(self._files["paper_candidate_records"].read()))
        self._paths = reference.ProductionArtifactPaths(
            corpus_root=Path(self._value["source_root"]),
            b2_source_inventory=self._files["source_inventory"].path,
            b3_root=self._files["split_assignments"].path.parent,
        )
        self._weights = reference.load_frozen_class_weights(self._files["class_weights"].path)
        self._verify_metadata()
        # The reference helper memoizes quarantine metadata. This single-process
        # sequential provider must not inherit a prior request's cached content.
        reference._b2_historical_raw_issue_categories.cache_clear()
        self.on_receipt = on_receipt
        self.access_observations = {"source_file_verifications": 0, "verified_source_bytes": 0,
                                    "raw_records_loaded": 0, "target_sidecars_materialized": 0,
                                    "test_source_reads": 0}
        self._records = {}
        self._binding = None
        if binding is not None:
            self._accept_binding(binding)

    @property
    def binding(self):
        if self._binding is None:
            _fail("prepared_binding_required")
        return deepcopy(self._binding)

    @property
    def validation_membership(self):
        return tuple(tuple(item) for item in self.preflight["validation_membership"])

    @property
    def validation_bindings(self):
        return tuple(self.binding["validation_batch_bindings"])

    @property
    def train_seen_tuples(self):
        # Full TRAIN target enumeration is outside this bounded data schedule.
        return None

    def schedule_bindings(self, stage):
        self._stage(stage)
        return tuple(self.binding["schedule_bindings"][stage])

    def _stage(self, stage):
        if stage not in self.configs:
            _fail("stage_invalid")

    def _verify_metadata(self):
        for name, item in self._files.items():
            if _stat(item.path) != self._metadata_stats[name]:
                item.read()
                self._metadata_stats[name] = _stat(item.path)

    def _receipt(self, payload):
        if self.on_receipt is not None:
            self.on_receipt({"schema": "WS2GroupMaskingDataReceipt@1.0.0", **payload,
                             "access_observations": deepcopy(self.access_observations)})

    def _source_files(self, record):
        source = self._inventory[record]
        sizes = self._value["source_sizes"][record]
        hashes = source["source_fingerprints"]
        result = [BoundFile(_source_path(self._paths.corpus_root, source["annotation_path"]),
                           hashes["physical_annotation_sha256"], sizes["annotation_size_bytes"])]
        if source["score_path"] is not None:
            result.append(BoundFile(_source_path(self._paths.corpus_root, source["score_path"]),
                                    hashes["score_sha256"], sizes["score_size_bytes"]))
        return result

    def _check_sources(self, files):
        for item in files:
            item.read()
            self.access_observations["source_file_verifications"] += 1
            self.access_observations["verified_source_bytes"] += item.size_bytes

    def _record(self, record_id, *, split, targets):
        guard_split(split)
        if (record_id not in self._assignments or self._assignments[record_id]["split"] != split
                or record_id not in self._value["source_sizes"]):
            _fail("record_not_in_bound_membership")
        self._receipt({"status": "started", "record_id": record_id, "split": split, "targets": targets})
        try:
            self._verify_metadata()
            files = self._source_files(record_id)
            self._check_sources(files)
            from music_critic.adapters.dilemmadata import DilemmadataAccepted, convert_dilemmadata_record
            from music_critic.experiments.analysisgnn.multitask_contract import LEGACY_TARGET_SIDECAR_VERSION, materialize_target_sidecar
            record = reference._resolve_selected_corpus_record(record_id, split=split, paths=self._paths)
            accepted = convert_dilemmadata_record(record)
            if not isinstance(accepted, DilemmadataAccepted):
                _fail("reference_record_quarantined")
            graph = build_raw_graph(accepted.piece)
            sample = SSLRawSample(graph, graph_fingerprint(graph), "dilemmadata", accepted.piece.piece_id)
            if sample.piece_id != self._paper[record_id]["piece_id"]:
                _fail("record_piece_identity_mismatch")
            self.access_observations["raw_records_loaded"] += 1
            sidecar = materialize_target_sidecar(accepted, target_sidecar_version=LEGACY_TARGET_SIDECAR_VERSION) if targets else None
            if targets:
                self.access_observations["target_sidecars_materialized"] += 1
                if (sidecar.get("record_id") != record_id
                        or sidecar.get("source_component_id") != self._assignments[record_id]["source_component_id"]
                        or sidecar.get("fingerprint") != fingerprint({key: val for key, val in sidecar.items() if key != "fingerprint"})):
                    _fail("sidecar_identity_or_fingerprint_mismatch")
            self._check_sources(files)
            self._verify_metadata()
            evidence = {"raw_graph_fingerprint": sample.raw_graph_fingerprint,
                        "sidecar_fingerprint": None if sidecar is None else fingerprint(sidecar)}
            previous = self._records.get(record_id)
            if previous is not None and (previous["raw_graph_fingerprint"] != evidence["raw_graph_fingerprint"]
                    or (targets and previous.get("sidecar_fingerprint") is not None
                        and previous["sidecar_fingerprint"] != evidence["sidecar_fingerprint"])):
                _fail("record_content_binding_mismatch")
            if self._binding is not None and (previous is None
                    or (targets and previous.get("sidecar_fingerprint") is None)):
                _fail("record_content_binding_missing")
            if previous is None or targets:
                self._records[record_id] = evidence
            self._receipt({"status": "completed", "record_id": record_id, "split": split, "targets": targets})
            return sample, sidecar
        except BaseException as exc:
            try:
                self._receipt({"status": "failed", "record_id": record_id, "split": split,
                               "targets": targets, "exception_type": type(exc).__name__, "reason": str(exc)})
            except BaseException:
                pass
            raise

    def _batch(self, row, *, split):
        ssl_samples = [self._record(record, split=split, targets=False)[0] for record in row.get("ssl", ())]
        theory = [self._record(record, split=split, targets=True) for record in row.get("theory", ())]
        return PilotBatch(collate_ssl_samples(ssl_samples) if ssl_samples else None,
                          collate_ssl_samples([item[0] for item in theory]) if theory else None,
                          tuple(item[1] for item in theory), self._weights if theory else None)

    def load_update(self, stage: str, index: int, *, split: str = "train") -> PilotBatch:
        guard_split(split)
        if split != "train":
            _fail("updates_train_only")
        self._stage(stage)
        rows = self._value[stage + "_schedule"]
        if type(index) is not int or not 0 <= index < len(rows):
            _fail("update_index_invalid")
        batch = self._batch(rows[index], split="train")
        actual = bind_train_batch(batch, self.configs[stage])
        if stage == "pilot":
            paired_mask_bindings(batch.ssl, self.configs[stage], update=index)
        if self._binding is not None and actual != self.schedule_bindings(stage)[index]:
            _fail("update_content_binding_mismatch")
        return batch

    def validation_batches(self, *, split: str = "validation") -> Iterator[PilotBatch]:
        guard_split(split)  # Executed immediately, before returning the iterator.
        if split != "validation":
            _fail("validation_only")
        from music_critic.ssl.group_masking_scientific_eval import validation_batch_binding, validation_policy_bindings

        def iterate():
            for index, record in enumerate(self.preflight["validation_record_ids"]):
                batch = self._batch({"ssl": [record], "theory": [record]}, split="validation")
                actual = validation_batch_binding(batch, self.configs["pilot"], resource_limits=self.validation_resource_limits)
                validation_policy_bindings(batch.ssl, self.configs["pilot"])
                if self._binding is not None and actual != self.validation_bindings[index]:
                    _fail("validation_content_binding_mismatch")
                yield batch
        return iterate()

    def _accept_binding(self, binding):
        value = deepcopy(binding)
        if (not isinstance(value, dict) or set(value) != {"schema", "preflight", "records", "schedule_bindings",
                "validation_batch_bindings", "paired_eligibility", "fingerprint"}
                or value["schema"] != BINDING_SCHEMA
                or value["fingerprint"] != fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
                or value["preflight"] != self.preflight
                or not isinstance(value["records"], dict) or set(value["records"]) != set(self._value["source_sizes"])
                or not isinstance(value["schedule_bindings"], dict) or set(value["schedule_bindings"]) != set(self.configs)):
            _fail("prepared_binding_invalid")
        for stage in self.configs:
            hashes = value["schedule_bindings"][stage]
            if not isinstance(hashes, list) or len(hashes) != self.configs[stage].updates or not all(_sha(item) for item in hashes):
                _fail("prepared_schedule_invalid")
        hashes = value["validation_batch_bindings"]
        if not isinstance(hashes, list) or len(hashes) != len(self.validation_membership) or not all(_sha(item) for item in hashes):
            _fail("prepared_validation_invalid")
        for row in value["records"].values():
            if (not isinstance(row, dict) or set(row) != {"raw_graph_fingerprint", "sidecar_fingerprint"}
                    or not _sha(row["raw_graph_fingerprint"])
                    or (row["sidecar_fingerprint"] is not None and not _sha(row["sidecar_fingerprint"]))):
                _fail("prepared_record_invalid")
        expected = {"policies": list(POLICIES), "fallback": False,
                    "pilot_updates_checked": self.configs["pilot"].updates,
                    "validation_batches_checked": len(self.validation_membership)}
        if value["paired_eligibility"] != expected:
            _fail("prepared_eligibility_invalid")
        self._records = deepcopy(value["records"])
        self._binding = value


def prepare_scientific_data(request, pilot_config, recovery_config, *, on_receipt=None):
    """Compile actual content/mask bindings before either arm can update.

    This opens only scheduled TRAIN plus the complete VALIDATION membership.
    The returned JSON binding must be published immutably by the caller.
    """
    from music_critic.ssl.group_masking_scientific_eval import validation_batch_binding
    data = BoundScientificData(request, pilot_config, recovery_config, on_receipt=on_receipt)
    schedules = {}
    for stage, config in data.configs.items():
        schedules[stage] = [bind_train_batch(data.load_update(stage, index), config)
                            for index in range(config.updates)]
    validation = [validation_batch_binding(batch, pilot_config, resource_limits=data.validation_resource_limits)
                  for batch in data.validation_batches()]
    payload = {"schema": BINDING_SCHEMA, "preflight": data.preflight,
               "records": deepcopy(data._records), "schedule_bindings": schedules,
               "validation_batch_bindings": validation,
               "paired_eligibility": {"policies": list(POLICIES), "fallback": False,
                    "pilot_updates_checked": pilot_config.updates,
                    "validation_batches_checked": len(data.validation_membership)}}
    data._accept_binding({**payload, "fingerprint": fingerprint(payload)})
    return data


def load_bound_scientific_data(request, binding_payload, pilot_config, recovery_config, *, on_receipt=None):
    """Recheck metadata at resume/startup; each yielded batch rechecks its sources."""
    return BoundScientificData(request, pilot_config, recovery_config,
                               binding=binding_payload, on_receipt=on_receipt)
