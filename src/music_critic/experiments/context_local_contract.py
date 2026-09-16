"""Frozen B3 metadata admission for a bounded, label-blind local comparison.

This module reads eight explicit metadata artifacts. It has no corpus reader,
checkpoint loader, source discovery, or target-value path. TEST record IDs are
checked only to establish the full frozen split. Runtime hash/count overrides
are deliberately absent; synthetic tests may patch module constants in memory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from music_critic.experiments.analysisgnn.contracts import fingerprint as b3_fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    ASSIGNMENT_ALGORITHM, ASSIGNMENT_NAMESPACE, FULL_RAW_UNIVERSE_ID,
    PAPER_CANDIDATE_UNIVERSE_ID, SOURCE_COMPONENT_VERSION,
    production_task_registry, validate_loaded_registry, vocabularies_payload,
)
from music_critic.experiments.context_diagnostics import FileBinding, fingerprint


EXPECTED_METADATA_SHA256 = {
    "full_raw_manifest.json": "421cb23a4f1cdac444dc9df244f54fdf625d0a19ffb1368d973890435843b389",
    "paper_candidate_manifest.json": "ae806363358bae8e82931581b11ade48f9d9fa9a4883fb18e15e4425f69e1dfb",
    "source_components.jsonl": "909e84531cc1c70fab14dd24dfe1c9525c5634e5edd46a6f88faf13cfc755778",
    "overlap_exclusions.jsonl": "20a02da59821eb71f5f37e0cc05bee8d1993b0eb6ef8807ab144325e1b865496",
    "paper_candidate_records.jsonl": "42d688619c5699e404ec545076339547bb38d16976c6e4605437e8c539f972c1",
    "split_assignments.jsonl": "056de0942918976981401db5017f6e567bc8949bae91ae395f0bd669adbfd51a",
    "task_registry.json": "232e1b45542334755018f1b710741532924eee3facb77a14656fa3c1c8ce2b42",
    "vocabularies.json": "3895c52d46fc980250484850afabc953d74108d9a766a409aa4e158926523deb",
}
EXPECTED_FULL_COUNTS = {"an_joint": 353, "dlc": 1280, "total": 1633}
EXPECTED_PAPER_COUNTS = {"an_joint": 353, "dlc": 1266, "total": 1619}
EXPECTED_SPLIT_COUNTS = {"train": 1295, "validation": 162, "test": 162}
EXPECTED_COMPONENT_COUNTS = {"train": 1209, "validation": 147, "test": 151}
EXPECTED_EXCLUSION_COUNT = 14
SELECTED_RECORD_COUNTS = {"train": 128, "validation": 32}
SELECTION_SEED = 17
SELECTION_NAMESPACE = "music-critic-v2.context-local-record-selection-v1"
MAX_METADATA_FILE_BYTES = 8 * 1024 * 1024
MAX_METADATA_TOTAL_BYTES = 32 * 1024 * 1024


def _fail(reason):
    raise ValueError("context_local_metadata." + reason)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class MetadataSelection:
    assignments: Mapping[str, Mapping[str, str]]
    selected: tuple[Mapping[str, str], ...]
    fingerprint: str
    evidence: Mapping[str, object]
    record_to_piece: Mapping[str, str]

    @property
    def selected_records(self) -> Mapping[str, Mapping[str, str]]:
        return MappingProxyType({row["record_id"]: row for row in self.selected})

    def to_dict(self) -> dict:
        """Return a detached JSON-serializable representation."""
        return _thaw({"assignments": self.assignments, "selected": self.selected,
                      "fingerprint": self.fingerprint, "evidence": self.evidence,
                      "record_to_piece": self.record_to_piece})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _parse(raw):
    def invalid_constant(_):
        _fail("nonfinite_json")
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        _fail("metadata_object_required")
    return value


def _identity(value):
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        _fail("nonempty_identity_required")
    return value


def _unique_ids(values):
    if not isinstance(values, list):
        _fail("record_id_list_required")
    identities = [_identity(value) for value in values]
    if len(identities) != len(set(identities)):
        _fail("duplicate_record_id")
    return set(identities)


def _rows_by_id(rows, field):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        _fail("metadata_row_list_required")
    result = {}
    for row in rows:
        key = _identity(row.get(field))
        if key in result:
            _fail("duplicate_" + field)
        result[key] = row
    return result


def _check_fingerprint(payload):
    if payload.get("fingerprint") != b3_fingerprint({k: v for k, v in payload.items() if k != "fingerprint"}):
        _fail("internal_fingerprint_mismatch")


def _check_manifest(payload, universe, counts):
    _check_fingerprint(payload)
    if payload.get("universe_id") != universe or payload.get("counts") != counts:
        _fail("universe_or_count_mismatch")
    ids = _unique_ids(payload.get("record_ids"))
    if len(ids) != counts["total"]:
        _fail("manifest_record_count_mismatch")
    return ids


def selection_rank(split: str, record_id: str) -> str:
    """Fixed seed-17 SHA rank of identity only; no labels or availability."""
    if split not in SELECTED_RECORD_COUNTS:
        _fail("selection_TRAIN_or_VALIDATION_only")
    _identity(record_id)
    return fingerprint([SELECTION_NAMESPACE, SELECTION_SEED, split, record_id])


def load_authoritative_metadata(bindings: Mapping[str, FileBinding]) -> MetadataSelection:
    """Verify full official bytes/closure, then select exactly 128/32 records.

    Paths must already have been resolved inside the caller's declared evidence
    root. Sizes are supplied explicitly and checked against the SHA-bound bytes.
    The function never opens a path obtained from metadata contents.
    """
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_METADATA_SHA256):
        _fail("exact_eight_metadata_bindings_required")
    for name, expected in EXPECTED_METADATA_SHA256.items():
        binding = bindings[name]
        if not isinstance(binding, FileBinding) or binding.sha256 != expected:
            _fail("official_metadata_hash_required")
        if (not isinstance(binding.path, str) or not Path(binding.path).is_absolute() or
                type(binding.size_bytes) is not int or not 0 < binding.size_bytes <= MAX_METADATA_FILE_BYTES):
            _fail("absolute_bounded_metadata_binding_required")
    if sum(binding.size_bytes for binding in bindings.values()) > MAX_METADATA_TOTAL_BYTES:
        _fail("total_metadata_budget_exceeded")
    payloads = {}
    for name in sorted(bindings):
        raw = bindings[name].read(max_bytes=MAX_METADATA_FILE_BYTES)
        payloads[name] = [_parse(line) for line in raw.splitlines() if line.strip()] if name.endswith(".jsonl") else _parse(raw)

    full_ids = _check_manifest(payloads["full_raw_manifest.json"], FULL_RAW_UNIVERSE_ID, EXPECTED_FULL_COUNTS)
    paper_ids = _check_manifest(payloads["paper_candidate_manifest.json"], PAPER_CANDIDATE_UNIVERSE_ID, EXPECTED_PAPER_COUNTS)
    if not paper_ids <= full_ids or len(full_ids - paper_ids) != EXPECTED_EXCLUSION_COUNT:
        _fail("paper_universe_exclusions_mismatch")
    records = _rows_by_id(payloads["paper_candidate_records.jsonl"], "record_id")
    if set(records) != paper_ids:
        _fail("paper_record_mapping_membership_mismatch")
    pieces = {}
    dialects = {}
    for record_id, row in records.items():
        dialect = row.get("dialect")
        if dialect not in ("an_joint", "dlc"):
            _fail("unknown_record_dialect")
        piece_id = _identity(row.get("piece_id"))
        label = "an" if dialect == "an_joint" else "dlc"
        expected_piece_id = f"piece:dilemmadata-{label}-{sha256(record_id.encode('utf-8')).hexdigest()[:24]}"
        if piece_id != expected_piece_id or piece_id in pieces.values():
            _fail("record_piece_mapping_mismatch")
        pieces[record_id] = piece_id
        dialects[record_id] = dialect
        _identity(row.get("source_component_id"))
    if {**dict(Counter(dialects.values())), "total": len(records)} != EXPECTED_PAPER_COUNTS:
        _fail("paper_dialect_count_mismatch")

    components = _rows_by_id(payloads["source_components.jsonl"], "source_component_id")
    record_components = {}
    for component_id, row in components.items():
        if row.get("version") != SOURCE_COMPONENT_VERSION:
            _fail("source_component_version_mismatch")
        members = _unique_ids(row.get("record_ids"))
        if not members:
            _fail("empty_source_component")
        evidence = _rows_by_id(row.get("source_evidence", []), "record_id")
        if set(evidence) != members:
            _fail("source_component_evidence_membership_mismatch")
        for record_id in members:
            if record_id in record_components:
                _fail("record_in_multiple_components")
            record_components[record_id] = component_id
    if set(record_components) != full_ids or len(components) != sum(EXPECTED_COMPONENT_COUNTS.values()):
        _fail("full_source_component_membership_mismatch")
    if any(records[record_id]["source_component_id"] != record_components[record_id] for record_id in records):
        _fail("paper_source_component_mismatch")

    exclusions = _rows_by_id(payloads["overlap_exclusions.jsonl"], "record_id")
    if set(exclusions) != full_ids - paper_ids:
        _fail("excluded_record_membership_mismatch")
    for record_id, row in exclusions.items():
        peer = row.get("matched_an_record_id")
        if (row.get("dialect") != "dlc" or peer not in records or records[peer]["dialect"] != "an_joint" or
                row.get("source_component_id") != record_components[record_id] or
                record_components[record_id] != record_components[peer]):
            _fail("excluded_peer_component_mismatch")
        dialects[record_id] = "dlc"
    if {**dict(Counter(dialects.values())), "total": len(dialects)} != EXPECTED_FULL_COUNTS:
        _fail("full_dialect_count_mismatch")

    assignments = _rows_by_id(payloads["split_assignments.jsonl"], "record_id")
    if set(assignments) != paper_ids:
        _fail("official_split_manifest_membership_mismatch")
    component_splits = {}
    for record_id, row in assignments.items():
        if set(row) != {"record_id", "dialect", "source_component_id", "split", "assignment_algorithm", "assignment_namespace"}:
            _fail("official_split_row_schema_mismatch")
        if row["assignment_algorithm"] != ASSIGNMENT_ALGORITHM or row["assignment_namespace"] != ASSIGNMENT_NAMESPACE:
            _fail("official_split_namespace_mismatch")
        if row["split"] not in EXPECTED_SPLIT_COUNTS:
            _fail("unknown_official_split")
        if row["dialect"] != dialects[record_id] or row["source_component_id"] != record_components[record_id]:
            _fail("official_split_record_identity_mismatch")
        component = row["source_component_id"]
        if component_splits.setdefault(component, row["split"]) != row["split"]:
            _fail("official_component_crosses_split")
    if dict(Counter(row["split"] for row in assignments.values())) != EXPECTED_SPLIT_COUNTS:
        _fail("official_split_record_count_mismatch")
    if dict(Counter(component_splits.values())) != EXPECTED_COMPONENT_COUNTS:
        _fail("official_split_component_count_mismatch")

    tasks, vocabularies = payloads["task_registry.json"], payloads["vocabularies.json"]
    _check_fingerprint(tasks)
    _check_fingerprint(vocabularies)
    validate_loaded_registry(tasks, vocabularies)
    if (tasks["fingerprint"] != production_task_registry()["fingerprint"] or
            vocabularies["fingerprint"] != vocabularies_payload()["fingerprint"]):
        _fail("runtime_registry_drift")

    selected = []
    for split, count in SELECTED_RECORD_COUNTS.items():
        ranked = sorted((record_id for record_id, row in assignments.items() if row["split"] == split),
                        key=lambda record_id: (selection_rank(split, record_id), record_id))
        if len(ranked) < count:
            _fail("insufficient_official_members")
        selected.extend({"record_id": record_id, "component_id": record_components[record_id], "split": split}
                        for record_id in ranked[:count])
    evidence = {
        "schema_version": "ContextLocalMetadataSelection@1.0.0",
        "source_metadata": {name: {"sha256": binding.sha256, "size_bytes": binding.size_bytes} for name, binding in sorted(bindings.items())},
        "full_raw_counts": EXPECTED_FULL_COUNTS, "paper_candidate_counts": EXPECTED_PAPER_COUNTS,
        "official_record_counts": EXPECTED_SPLIT_COUNTS, "official_component_counts": EXPECTED_COMPONENT_COUNTS,
        "excluded_duplicate_records": sorted(exclusions),
        "selection": {"algorithm": "sha256_rank_then_record_id_first_n_per_official_split-v1",
                      "hash_payload": "compact_utf8_json([namespace,seed,split,record_id]); no newline",
                      "namespace": SELECTION_NAMESPACE, "seed": SELECTION_SEED,
                      "unit": "record", "requested_counts": SELECTED_RECORD_COUNTS,
                      "selected_record_ids": [row["record_id"] for row in selected],
                      "selected_component_counts": {split: len({row["component_id"] for row in selected if row["split"] == split})
                                                    for split in SELECTED_RECORD_COUNTS},
                      "uses_target_values_or_availability": False},
        "access_scope": {"test_identities_checked": True, "test_raw_or_targets_opened": False,
                         "source_payloads_opened": False, "checkpoint_opened": False},
        "export_lineage_boundary": "metadata_and_cache_hashes_do_not_prove_independent_corpus_to_cache_regeneration; coordinator_audit_and_selected_export_bindings_are_required",
        "upstream_registry_fingerprints": {"tasks": tasks["fingerprint"], "vocabularies": vocabularies["fingerprint"]},
    }
    content = {"assignments": assignments, "selected": selected, "record_to_piece": pieces, "evidence": evidence}
    return MetadataSelection(_freeze(assignments), _freeze(selected), fingerprint(content), _freeze(evidence), _freeze(pieces))


__all__ = ["MetadataSelection", "load_authoritative_metadata", "selection_rank"]
