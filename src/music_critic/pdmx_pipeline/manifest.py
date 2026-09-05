"""Explicit source evidence and pre-window grouping for the PDMX pipeline.

Only metadata is retained at corpus scope. There is deliberately no discovery
function: a caller supplies every source identity, hash and split assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import json
import math
import re
from typing import Mapping

from music_critic.data.serialization import canonical_json_sha256

MANIFEST_VERSION = "1.0.0"
SELECTION_VERSION = "1.0.0"


class PdmxPipelineError(ValueError):
    def __init__(self, category: str, message: str = "") -> None:
        self.category = category
        super().__init__(f"{category}: {message}")


def fingerprint(value: object) -> str:
    return canonical_json_sha256(value)


def require_hash(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PdmxPipelineError("pdmx.manifest_invalid", f"{name} requires SHA-256")


def require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PdmxPipelineError("pdmx.manifest_invalid", f"invalid {name}")


def guard_split(split: str) -> None:
    # Must precede source/cache callbacks, including a cache-index reader.
    if split not in ("train", "validation"):
        raise PdmxPipelineError("pdmx.split_locked", "only explicit TRAIN/VALIDATION allowed")


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PdmxPipelineError("pdmx.path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise PdmxPipelineError("pdmx.path_invalid")
    if str(path) != value or any(ord(c) < 32 for c in value):
        raise PdmxPipelineError("pdmx.path_invalid")
    return value


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative_path(relative)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PdmxPipelineError("pdmx.path_escape") from exc
    return path


def strict_json(payload: str) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise PdmxPipelineError("pdmx.json_duplicate_key", key)
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise PdmxPipelineError("pdmx.json_nonfinite", value)

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    relative_path: str
    sha256: str
    license_id: str | None = None
    license_evidence: str | None = None
    rating: float | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        require_text(self.source_id, "source_id")
        relative_path(self.relative_path)
        require_hash(self.sha256, "source hash")
        for name in ("license_id", "license_evidence"):
            if getattr(self, name) is not None:
                require_text(getattr(self, name), name)
        if type(self.synthetic) is not bool:
            raise PdmxPipelineError("pdmx.manifest_invalid", "synthetic must be boolean")
        if self.rating is not None and (
            isinstance(self.rating, bool) or not isinstance(self.rating, (int, float))
            or not math.isfinite(self.rating)
        ):
            raise PdmxPipelineError("pdmx.manifest_invalid", "rating must be finite")


@dataclass(frozen=True, slots=True)
class LineageLink:
    left: str
    right: str
    evidence: str

    def __post_init__(self) -> None:
        for name in ("left", "right", "evidence"):
            require_text(getattr(self, name), name)
        if self.left == self.right:
            raise PdmxPipelineError("pdmx.manifest_invalid", "self lineage link")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    records: tuple[SourceRecord, ...]
    schema_profile: str
    mode: str
    release_identity: str | None = None
    release_sha256: str | None = None
    license_artifact_sha256: str | None = None
    lineage_links: tuple[LineageLink, ...] = ()
    cross_corpus_links: str = "unknown"
    version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.version != MANIFEST_VERSION or self.mode not in ("fixture", "production"):
            raise PdmxPipelineError("pdmx.manifest_version_or_mode_invalid")
        require_text(self.schema_profile, "schema_profile")
        if not isinstance(self.records, tuple) or not all(type(r) is SourceRecord for r in self.records):
            raise PdmxPipelineError("pdmx.manifest_invalid", "records must be immutable SourceRecords")
        ids = [r.source_id for r in self.records]
        paths = [r.relative_path for r in self.records]
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise PdmxPipelineError("pdmx.manifest_duplicate_source")
        if self.release_identity is not None:
            require_text(self.release_identity, "release identity")
        for name in ("release_sha256", "license_artifact_sha256"):
            if getattr(self, name) is not None:
                require_hash(getattr(self, name), name)
        if not isinstance(self.lineage_links, tuple) or not all(type(r) is LineageLink for r in self.lineage_links):
            raise PdmxPipelineError("pdmx.manifest_invalid", "lineage links must be immutable")
        for link in self.lineage_links:
            if link.left not in ids or link.right not in ids:
                raise PdmxPipelineError("pdmx.lineage_unknown_source")
        if self.cross_corpus_links != "unknown":
            raise PdmxPipelineError("pdmx.cross_corpus_binding_unsupported")
        if self.mode == "fixture" and any(not r.synthetic for r in self.records):
            raise PdmxPipelineError("pdmx.fixture_source_not_declared")
        if self.mode == "production":
            if any(r.synthetic for r in self.records) or not all((
                self.release_identity, self.release_sha256, self.license_artifact_sha256,
            )):
                raise PdmxPipelineError("pdmx.production_bindings_missing")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["records"] = [asdict(r) for r in sorted(self.records, key=lambda r: r.source_id)]
        value["lineage_links"] = sorted(
            ({"left": min(l.left, l.right), "right": max(l.left, l.right), "evidence": l.evidence}
             for l in self.lineage_links), key=lambda row: (row["left"], row["right"], row["evidence"]),
        )
        return value

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    allowed_licenses: tuple[str, ...]
    min_rating: float | None = None
    require_license_evidence: bool = True
    max_records: int | None = None
    version: str = SELECTION_VERSION

    def __post_init__(self) -> None:
        if self.version != SELECTION_VERSION:
            raise PdmxPipelineError("pdmx.selection_version_invalid")
        if not isinstance(self.allowed_licenses, tuple) or not self.allowed_licenses:
            raise PdmxPipelineError("pdmx.selection_invalid", "explicit license allowlist required")
        for item in self.allowed_licenses:
            require_text(item, "allowed license")
        if len(set(self.allowed_licenses)) != len(self.allowed_licenses):
            raise PdmxPipelineError("pdmx.selection_invalid", "duplicate allowed license")
        if type(self.require_license_evidence) is not bool:
            raise PdmxPipelineError("pdmx.selection_invalid")
        if self.min_rating is not None and (
            isinstance(self.min_rating, bool) or not isinstance(self.min_rating, (int, float))
            or not math.isfinite(self.min_rating)
        ):
            raise PdmxPipelineError("pdmx.selection_invalid", "finite rating threshold required")
        if self.max_records is not None and (type(self.max_records) is not int or self.max_records < 1):
            raise PdmxPipelineError("pdmx.selection_invalid", "positive record budget required")


@dataclass(frozen=True, slots=True)
class SelectionPlan:
    records: tuple[SourceRecord, ...]
    exclusions: tuple[tuple[str, str], ...]
    components: tuple[tuple[str, str], ...]
    splits: tuple[tuple[str, str], ...]
    fingerprint: str


def plan_selection(
    manifest: SourceManifest, selection: SelectionConfig, split_by_source: Mapping[str, str],
) -> SelectionPlan:
    """Validate all supplied splits and group all evidence *before* filtering."""
    for split in split_by_source.values():
        guard_split(split)
    rows = sorted(manifest.records, key=lambda r: r.source_id)
    ids = {r.source_id for r in rows}
    if set(split_by_source) != ids:
        raise PdmxPipelineError("pdmx.split_coverage_mismatch")
    parent = {key: key for key in ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    by_hash: dict[str, str] = {}
    for row in rows:
        union(row.source_id, by_hash.setdefault(row.sha256, row.source_id))
    for link in manifest.lineage_links:
        union(link.left, link.right)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(find(row.source_id), []).append(row.source_id)
    components: dict[str, str] = {}
    for members in groups.values():
        if len({split_by_source[s] for s in members}) != 1:
            raise PdmxPipelineError("pdmx.split_component_crossing")
        identity = "pdmx-component:" + fingerprint(sorted(members))
        components.update((source_id, identity) for source_id in members)
    accepted: list[SourceRecord] = []
    exclusions: list[tuple[str, str]] = []
    for row in rows:
        reason = None
        if row.license_id not in selection.allowed_licenses:
            reason = "pdmx.filter_license"
        elif selection.require_license_evidence and row.license_evidence is None:
            reason = "pdmx.filter_license_evidence_missing"
        elif selection.min_rating is not None and row.rating is None:
            reason = "pdmx.filter_rating_missing"
        elif selection.min_rating is not None and row.rating < selection.min_rating:
            reason = "pdmx.filter_rating"
        elif selection.max_records is not None and len(accepted) >= selection.max_records:
            reason = "pdmx.filter_record_budget"
        if reason is None:
            accepted.append(row)
        else:
            exclusions.append((row.source_id, reason))
    binding = {"manifest": manifest.fingerprint, "selection": asdict(selection),
               "splits": sorted(split_by_source.items()), "components": sorted(components.items())}
    return SelectionPlan(tuple(accepted), tuple(exclusions), tuple(sorted(components.items())),
                         tuple(sorted(split_by_source.items())), fingerprint(binding))


def source_manifest_from_dict(value: dict[str, object]) -> SourceManifest:
    value = dict(value)
    value["records"] = tuple(SourceRecord(**row) for row in value["records"])
    value["lineage_links"] = tuple(LineageLink(**row) for row in value.get("lineage_links", ()))
    return SourceManifest(**value)


def load_source_manifest(path: Path) -> SourceManifest:
    return source_manifest_from_dict(strict_json(path.read_text(encoding="utf-8")))
