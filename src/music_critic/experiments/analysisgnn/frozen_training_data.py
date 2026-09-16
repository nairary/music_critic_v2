"""Shared strict read-only Dilemma warm exports for TRAIN/VALIDATION campaigns.

The coordinator supplies a SHA/size-bound manifest; this is not a corpus
locator or a preprocessing command. Missing warm exports fail closed. Existing
canonical deserialization, raw preparation/collation, B5B class weights and
historical full-orbit sampling remain authoritative. No cache is materialized,
no source is discovered, and no graph or decoded record is retained.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat

from music_critic.data import loads_piece
from music_critic.adapters.dilemmadata import _piece_id
from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample
from . import corrected_training as reference
from .contracts import fingerprint
from .full_orbit_training import build_full_orbit_table, full_orbit_table_contract
from .multitask_contract import PAPER_DEFINED_JOINT_COMPONENTS, TARGET_SIDECAR_VERSION
from .performance_c2_train_schedule import _train_records


DATA_SCHEMA = "HarmonicFusionDilemmaData@1.0.0"
_METADATA = ("split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts")
_FILES = ("piece", "sidecar", "metadata")
# Independently pinned by the checked-in B5A input, B5B weight and B5H orbit
# fixtures respectively. A caller-provided self-consistent hash cannot change
# those historical scientific contracts.
FROZEN_JOINT_TUPLE_SHA256 = "32cd9a4396bdb68c2077b68ca0efd0cd31d400e7c3385ca000d9b1206e38686b"
FROZEN_CLASS_WEIGHT_FINGERPRINT = "87266d8a60574b16a6e66d4081b11144f680f55e6f5a7b203ff722f64d817ee8"
FROZEN_ORBIT_TABLE_FINGERPRINT = "133983af065f28faab2258e8e2a1de057c87e34cdf214e494fa19a1e76987661"


def _require(condition, reason):
    if not condition:
        raise ValueError("fusion_data." + reason)


def _pairs(rows):
    result = {}
    for key, value in rows:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _json(payload):
    def invalid(_):
        raise ValueError("fusion_data.nonfinite_json")
    return json.loads(payload, object_pairs_hook=_pairs, parse_constant=invalid)


def _jsonl(payload):
    return [_json(line) for line in payload.splitlines() if line.strip()]


def _binding(value):
    _require(type(value) is dict and set(value) == {"path", "sha256", "size_bytes"}, "artifact_fields")
    _require(type(value["path"]) is str and value["path"] and "\x00" not in value["path"], "artifact_path")
    path = Path(value["path"])
    _require(path.is_absolute() and ".." not in path.parts, "absolute_regular_path_required")
    _require(type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "artifact_sha256")
    _require(type(value["size_bytes"]) is int and value["size_bytes"] > 0, "artifact_size")
    return deepcopy(value)


def _identity(path):
    try:
        _require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink_forbidden")
        value = path.stat()
    except OSError as exc:
        raise ValueError("fusion_data.artifact_unavailable") from exc
    _require(stat.S_ISREG(value.st_mode), "regular_file_required")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_bound_bytes(binding):
    """Hash exact explicit regular-file bytes before returning them to a parser."""
    binding = _binding(binding)
    path = Path(binding["path"])
    before = _identity(path)
    _require(before[2] == binding["size_bytes"], "artifact_size_mismatch")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise ValueError("fusion_data.artifact_unavailable") from exc
    _require(before == _identity(path), "artifact_changed_during_read")
    _require(len(payload) == binding["size_bytes"] and sha256(payload).hexdigest() == binding["sha256"],
             "artifact_hash_mismatch")
    return payload


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


class FrozenTrainingData:
    """Metadata-authenticated reader for explicit TRAIN/VALIDATION warm files.

    Manifest fields: ``schema_version``, four metadata bindings named in
    ``_METADATA``, and ``records``. Each record has ``record_id``, ``split``,
    ``source_component_id``, plus ``piece``, ``sidecar``, ``metadata`` bindings.
    Every file binding contains absolute ``path``, lowercase ``sha256`` and
    positive ``size_bytes``. Paths are operational locators and are excluded
    from ``manifest_fingerprint``. Full VALIDATION must be declared. TRAIN may
    contain only the fixed scheduled subset; ``require_train_records`` verifies
    that subset before the dispatcher launches either arm.

    Construction reads only pinned metadata. ``verify_bindings`` authenticates
    every declared warm file at startup/resume without decoding graph payloads.
    ``reader`` checks split before I/O and rehashes its three files each call.
    The manifest pins exported cache bytes; rederiving them from original TSV
    sources remains the coordinator's separate provenance/export obligation.
    """

    def __init__(self, manifest: dict, *, allowed_splits=("train", "validation")):
        _require(type(allowed_splits) in (tuple, list) and tuple(allowed_splits) == ("train", "validation"),
                 "train_validation_only")
        _require(type(manifest) is dict and set(manifest) == {"schema_version", "records", *_METADATA},
                 "manifest_fields")
        _require(manifest["schema_version"] == DATA_SCHEMA, "manifest_schema")
        _require(type(manifest["records"]) is list and manifest["records"], "record_manifest_required")
        self._manifest = deepcopy(manifest)
        self._records = {}
        # Structural TEST rejection happens before even metadata access.
        for row in self._manifest["records"]:
            _require(type(row) is dict and set(row) == {"record_id", "split", "source_component_id", *_FILES},
                     "record_fields")
            _require(row["split"] in ("train", "validation"), "record_split_forbidden")
            _require(all(type(row[key]) is str and row[key] for key in ("record_id", "source_component_id")),
                     "record_identity")
            _require(row["record_id"] not in self._records, "duplicate_record")
            for key in _FILES:
                row[key] = _binding(row[key])
            self._records[row["record_id"]] = row
        self._manifest["records"] = [self._records[key] for key in sorted(self._records)]
        for key in _METADATA:
            self._manifest[key] = _binding(self._manifest[key])
        _require(self._manifest["split_assignments"]["sha256"] == reference.CORRECTED_SPLIT_ASSIGNMENTS_SHA256,
                 "frozen_split_hash_mismatch")
        _require(self._manifest["joint_tuple_counts"]["sha256"] == FROZEN_JOINT_TUPLE_SHA256,
                 "frozen_joint_tuple_hash_mismatch")
        payloads, self._metadata_identities = {}, {}
        for key in _METADATA:
            path = Path(self._manifest[key]["path"])
            identity = _identity(path)
            payloads[key] = read_bound_bytes(self._manifest[key])
            _require(identity == _identity(path), "metadata_changed")
            self._metadata_identities[key] = identity
        split_rows = _jsonl(payloads["split_assignments"])
        self._assignments = self._validate_assignments(split_rows)
        self.validation_record_ids = tuple(sorted(key for key, row in self._assignments.items()
                                                   if row["split"] == "validation"))
        for rid, row in self._records.items():
            assignment = self._assignments.get(rid)
            _require(assignment is not None and all(row[key] == assignment[key]
                     for key in ("split", "source_component_id")), "record_assignment_mismatch")
        _require(set(self.validation_record_ids) <= set(self._records), "complete_validation_required")
        train = _train_records(split_rows, _jsonl(payloads["shift_eligibility"]))
        components = {}
        for row in train:
            components.setdefault(row["component_id"], []).append(row["record_id"])
        self.table = build_full_orbit_table(components, {row["record_id"]: row["valid_shifts"] for row in train})
        orbit = full_orbit_table_contract(self.table)
        _require(orbit["table_fingerprint"] == FROZEN_ORBIT_TABLE_FINGERPRINT, "frozen_orbit_table_mismatch")
        # Reuse the exact frozen implementation; rehash after its own path read.
        # The independent payload fingerprint disallows caller-invented weights.
        _json(payloads["class_weights"])  # Also reject duplicate/nonfinite JSON before the legacy parser.
        self.class_weights = reference.load_frozen_class_weights(self._manifest["class_weights"]["path"])
        read_bound_bytes(self._manifest["class_weights"])
        _require(self.class_weights.payload_fingerprint == FROZEN_CLASS_WEIGHT_FINGERPRINT,
                 "frozen_class_weights_mismatch")
        tuples = set()
        for row in _jsonl(payloads["joint_tuple_counts"]):
            if row.get("split") == "train" and row.get("mode") == "corrected_harmonic_event":
                target = row.get("tuple")
                _require(type(target) is dict and all(type(target.get(task)) is str
                         for task in PAPER_DEFINED_JOINT_COMPONENTS), "joint_tuple_fields")
                tuples.add(tuple(target[task] for task in PAPER_DEFINED_JOINT_COMPONENTS))
        _require(bool(tuples), "train_joint_tuple_inventory_empty")
        self.train_seen_tuples = frozenset(tuples)
        self.manifest_fingerprint = fingerprint(_portable(self._manifest))
        self._bindings = {"data_manifest_fingerprint": self.manifest_fingerprint,
            "class_weight_fingerprint": self.class_weights.payload_fingerprint,
            "split_sha256": self._manifest["split_assignments"]["sha256"],
            "orbit_table_fingerprint": orbit["table_fingerprint"],
            "native_train_joint_tuple_sha256": self._manifest["joint_tuple_counts"]["sha256"],
            "transposition": "historical_C2_stale_track_pitch_summaries",
            "preparation": "original_sequential_canonical_singleton_no_retained_cache"}
        self._check_metadata()

    @staticmethod
    def _validate_assignments(rows):
        assignments, components = {}, {}
        for row in rows:
            _require(type(row) is dict and row.get("split") in ("train", "validation", "test"), "split_row")
            rid, component = row.get("record_id"), row.get("source_component_id")
            _require(type(rid) is str and rid and type(component) is str and component, "split_identity")
            _require(rid not in assignments, "duplicate_split_record")
            _require(components.get(component, row["split"]) == row["split"], "cross_split_component")
            assignments[rid], components[component] = row, row["split"]
        counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
        component_counts = {split: sum(value == split for value in components.values()) for split in counts}
        _require(counts == {"train": 1295, "validation": 162, "test": 162}
                 and component_counts == {"train": 1209, "validation": 147, "test": 151}, "frozen_split_counts")
        return assignments

    @property
    def assignments(self):
        return deepcopy(self._assignments)

    @property
    def bindings(self):
        return deepcopy(self._bindings)

    def _gate(self, record_id, split):
        _require(split in ("train", "validation"), "requested_split_forbidden")
        _require(type(record_id) is str, "record_identity")
        assignment = self._assignments.get(record_id)
        _require(assignment is not None and assignment["split"] == split, "requested_record_split_mismatch")
        _require(record_id in self._records, "warm_export_not_declared")
        return self._records[record_id]

    def require_train_records(self, record_ids):
        ids = tuple(record_ids)
        _require(bool(ids), "scheduled_train_records_required")
        for record_id in ids:
            self._gate(record_id, "train")
        return tuple(sorted(set(ids)))

    def _check_metadata(self):
        for key in _METADATA:
            _require(_identity(Path(self._manifest[key]["path"])) == self._metadata_identities[key],
                     "metadata_changed")

    def verify_bindings(self):
        """Full metadata/export byte check for startup or resume, no graph I/O."""
        self._check_metadata()
        for key in _METADATA:
            read_bound_bytes(self._manifest[key])
        for rid, row in self._records.items():
            self._gate(rid, row["split"])
            for key in _FILES:
                read_bound_bytes(row[key])
        self._check_metadata()
        return {"schema_version": DATA_SCHEMA, "manifest_fingerprint": self.manifest_fingerprint,
                "verified_records": len(self._records), "record_files_verified": 3 * len(self._records),
                "test_payloads_accessed": False, "graph_builds": 0}

    def reader(self, record_id, *, split):
        row = self._gate(record_id, split)  # No payload or metadata access before the split gate.
        self._check_metadata()
        payloads = {key: read_bound_bytes(row[key]) for key in _FILES}
        metadata = _json(payloads["metadata"])
        _require(metadata == {"record_id": record_id, "split": split,
                 "piece_sha256": row["piece"]["sha256"], "sidecar_sha256": row["sidecar"]["sha256"]},
                 "cache_metadata_mismatch")
        sidecar = _json(payloads["sidecar"])
        _require(type(sidecar) is dict and sidecar.get("record_id") == record_id
                 and sidecar.get("source_component_id") == row["source_component_id"], "sidecar_identity_mismatch")
        _require(sidecar.get("schema_version") == TARGET_SIDECAR_VERSION
                 and sidecar.get("fingerprint") == fingerprint({key: value for key, value in sidecar.items()
                                                               if key != "fingerprint"}), "sidecar_contract_mismatch")
        _require(sidecar.get("dialect") in ("an_joint", "dlc"), "sidecar_dialect")
        piece = loads_piece(payloads["piece"].decode("utf-8"))
        _require(piece.dataset_name == "dilemmadata" and piece.split in (None, split)
                 and piece.source_group_id == row["source_component_id"]
                 and piece.piece_id == _piece_id(record_id, sidecar["dialect"]), "canonical_identity_mismatch")
        sample = prepare_multisource_sample(piece)
        batch = collate_multisource_samples((sample,)).raw_graph_batch
        self._check_metadata()
        # Detect source replacement during preparation as well as during reads.
        for key in _FILES:
            read_bound_bytes(row[key])
        return batch, sidecar
