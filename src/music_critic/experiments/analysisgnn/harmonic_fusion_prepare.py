"""Metadata-only preparation of the explicit bounded fusion data manifest.

This adapts the coordinator's pinned compact index, not the underlying corpus.
No raw/sidecar payload reader, graph preparation, checkpoint or GPU operation
is called. Native B4 tuple and deployed weight locators are required inputs;
there is no guessed-path fallback or source-manifest generation here.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from .harmonic_fusion_data import DATA_SCHEMA, FrozenFusionData, _binding, _json, read_bound_bytes


_METADATA_KEYS = frozenset(("split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts"))
_INDEX_FIELDS = frozenset(("record_id", "split", "source_component_id", "raw", "sidecar", "metadata"))


def _require(condition, reason):
    if not condition:
        raise ValueError("fusion_prepare." + reason)


def _records(payload):
    rows = [_json(line) for line in payload.splitlines() if line.strip()]
    _require(bool(rows), "cache_index_empty")
    # Finish this entire compact-index gate before reading metadata-pins or
    # any referenced metadata file. Record names are not split authority:
    # historical names containing ':test:' may correctly belong to TRAIN.
    for row in rows:
        _require(type(row) is dict, "cache_index_row_object")
        _require(row.get("split") in ("train", "validation"), "cache_index_split_forbidden")
    seen, result = set(), []
    for row in rows:
        _require(set(row) == _INDEX_FIELDS, "cache_index_row_fields")
        rid = row["record_id"]
        _require(type(rid) is str and bool(rid) and type(row["source_component_id"]) is str
                 and bool(row["source_component_id"]), "cache_index_identity")
        _require(rid not in seen, "cache_index_duplicate_record")
        seen.add(rid)
        result.append({"record_id": rid, "split": row["split"],
            "source_component_id": row["source_component_id"], "piece": _binding(row["raw"]),
            "sidecar": _binding(row["sidecar"]), "metadata": _binding(row["metadata"])})
    return sorted(result, key=lambda row: row["record_id"])


def _output_path(value):
    path = Path(value).absolute()
    _require(".." not in path.parts and not any(part.is_symlink() for part in (path, *path.parents)),
             "regular_output_path_required")
    _require(not path.exists(), "output_already_exists")
    return path


def _publish_once(path, payload):
    """Publish complete, fsynced bytes with exclusive atomic namespace creation."""
    path = _output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".fusion-manifest-", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Never replaces an earlier manifest, including a competing writer.
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data_manifest(*, cache_index: dict, metadata_pins: dict, output: str | Path) -> dict:
    """Authenticate compact inputs, check frozen metadata, publish one manifest.

    ``cache_index`` and ``metadata_pins`` are explicit ``path/sha256/size_bytes``
    pins of the two input files. The latter file contains exactly four named
    metadata file pins. The returned report describes metadata validation only;
    the dispatcher still has to authenticate actual warm exports and execute
    separately registered source/runtime/reference gates.
    """
    destination = _output_path(output)
    cache_index = _binding(cache_index)
    records = _records(read_bound_bytes(cache_index))
    # Do not even validate/open this input before the index's TEST/shape gate.
    metadata_pins = _binding(metadata_pins)
    metadata = _json(read_bound_bytes(metadata_pins))
    _require(type(metadata) is dict and set(metadata) == _METADATA_KEYS, "exact_metadata_pins_required")
    metadata = {key: _binding(metadata[key]) for key in sorted(_METADATA_KEYS)}
    manifest = {"schema_version": DATA_SCHEMA, **metadata, "records": records}
    data = FrozenFusionData(manifest)  # Metadata only: never call reader or verify_bindings here.
    # Preserve the pinned input identity through validation rather than letting
    # an input change silently between the initial read and publication.
    read_bound_bytes(cache_index)
    read_bound_bytes(metadata_pins)
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8") + b"\n"
    report = {"schema_version": "HarmonicFusionDataManifestPreparation@1.0.0",
        "output": {"path": str(destination), "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)},
        "input_pins": {"cache_index": cache_index, "metadata_pins": metadata_pins},
        "data_bindings": data.bindings, "data_manifest_fingerprint": data.manifest_fingerprint,
        "declared_records": len(records), "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "validation_records": len(data.validation_record_ids), "metadata_validated": True,
        "warm_payloads_verified": False, "warm_payload_files_opened": 0,
        "graph_builds": 0, "model_forwards": 0, "checkpoint_files_opened": 0,
        "scientific_launch_ready": False}
    # Validate all report values before committing output, even if an injected
    # metadata validator returned a malformed value during a test.
    json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True)
    _publish_once(destination, payload)
    return report
