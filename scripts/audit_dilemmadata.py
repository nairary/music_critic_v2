#!/usr/bin/env python3
"""Streaming, deterministic Phase 9A audit for Dilemmadata v1.0.

This module is evidence tooling, not a production dataset adapter.  It reads
the installed release without importing its processing code, keeps paths
corpus-relative, and separates score-derived observations from annotation and
derived-target columns.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
import json
import os
from os import PathLike
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


AUDIT_SCHEMA_VERSION = "1.0.0"
CORPUS_ID = "dilemmadata"
CORPUS_NAME = "Dilemmadata: A symbolic dataset for music research"
RELEASE_VERSION = "v1.0"
RELEASE_COMMIT = "d60ee75b4a9495e932a4a7be39381578be17e222"
UPSTREAM_REPOSITORY = "https://github.com/johentsch/dilemmadata"
LICENSE_ID = "CC-BY-NC-SA-4.0"
ENV_ROOT = "MUSIC_CRITIC_DILEMMADATA_ROOT"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "dilemmadata"
    / "audit_manifest.json"
)
VOCABULARY_LIMIT = 256

RELEASE_PRODUCERS = (
    {
        "name": "Hentschel, Johannes",
        "orcid": "0000-0002-1986-9545",
        "affiliation": "Anton Bruckner University",
    },
    {
        "name": "Karystinaios, Emmanouil",
        "orcid": "0000-0001-9354-8953",
        "affiliation": "Johannes Kepler University",
    },
)
SOURCE_LINEAGE = (
    {
        "name": "AugmentedNet",
        "repository": "https://github.com/johentsch/AugmentedNet",
        "release_gitlink_commit": "ec6cfe78fe252098ecdedd96bb300ad131830cc6",
    },
    {
        "name": "Distant Listening Corpus",
        "repository": "https://github.com/DCMLab/distant_listening_corpus",
        "release_gitlink_commit": "3a3152b5ee2448359497bc02794cd39b937d4118",
    },
)

MISSING_TOKENS = frozenset({"", "<NA>", "NA", "NaN", "nan", "None", "null"})
TRUE_TOKENS = frozenset({"1", "True", "true", "TRUE"})
FALSE_TOKENS = frozenset({"0", "False", "false", "FALSE"})

AN_REQUIRED_FIELDS = (
    "onset_div",
    "duration_div",
    "s_offset_frac",
    "s_duration_frac",
    "s_midi",
    "s_isOnset",
    "s_step",
    "s_alter",
    "ks_fifths",
    "ts_beats",
    "ts_beat_type",
    "s_part_id",
    "s_voice_id",
)
DLC_REQUIRED_FIELDS = (
    "onset_div",
    "duration_div",
    "quarterbeats_playthrough",
    "duration",
    "pitch",
    "is_note_onset",
    "step",
    "alter",
    "ks_fifths",
    "ts_beats",
    "ts_beat_type",
    "staff",
    "voice",
)

# These sets are justified by the release's own processing boundary.  AN uses
# score-prefixed s_* fields plus exact timing/meter columns.  DLC's
# make_pitch_array() constructs the columns through quarterbeats_playthrough
# from the score note/measure facets before make_labeled_pitch_array() merges
# annotations beginning at section_start/unfolded_harmony_index.
AN_RAW_FIELDS = frozenset(
    {
        "onset_div",
        "duration_div",
        "j_offset",
        "s_offset_frac",
        "s_duration",
        "s_duration_frac",
        "s_measure",
        "ks_fifths",
        "ts_beats",
        "ts_beat_type",
        "measureNumberWithSuffix",
        "onset_beat",
        "s_note",
        "s_midi",
        "s_isOnset",
        "s_step",
        "s_alter",
        "mn_onset",
        "s_beat_float",
        "is_downbeat",
        "downbeat",
        "s_part_id",
        "s_voice_id",
    }
)
DLC_RAW_FIELDS = frozenset(
    {
        "onset_div",
        "duration_div",
        "onset_beat",
        "pitch",
        "tpc",
        "step",
        "alter",
        "beat_float",
        "ts_beats",
        "ts_beat_type",
        "staff",
        "voice",
        "duration",
        "is_note_onset",
        "ks_fifths",
        "mc",
        "mc_playthrough",
        "mn",
        "mn_playthrough",
        "octave",
        "quarterbeats_playthrough",
    }
)


@dataclass(frozen=True, slots=True)
class TargetFamilySpec:
    family: str
    an_primary: str | None
    dlc_primary: str | None
    an_source_fields: tuple[str, ...]
    dlc_source_fields: tuple[str, ...]
    coordinate: str
    positive_only: bool = False
    an_gate: str | None = None
    dlc_gate: str | None = None
    vocabulary_kind: str = "open"
    mapping_status: str = "source_specific"


TARGET_FAMILIES = (
    TargetFamilySpec(
        "global_key", None, "globalkey", (), ("globalkey", "globalkey_tpc", "globalkey_mode"),
        "piece_or_repeated_note_sidecar", vocabulary_kind="open", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "local_key", "a_localKey", "localkey", ("a_localKey",),
        ("localkey", "localkey_tpc", "localkey_mode", "localkey_is_minor"),
        "annotation_run_sidecar", vocabulary_kind="open", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "tonal_region", "a_localKey", "localkey", ("a_localKey",),
        ("localkey", "localkey_tpc", "localkey_mode"), "annotation_run_sidecar",
        vocabulary_kind="open", mapping_status="deferred"
    ),
    TargetFamilySpec(
        "chord_boundary", "a_isOnset", "a_isOnset", ("a_isOnset",), ("a_isOnset",),
        "exact_observed_note_onset_point", positive_only=True, an_gate="valid_chord_label",
        dlc_gate="valid_chord_label", vocabulary_kind="positive_unlabeled",
        mapping_status="derived_lossless_subset"
    ),
    TargetFamilySpec(
        "roman_numeral", "a_romanNumeral", "label",
        ("a_romanNumeral", "a_simpleNumeral", "a_degree1", "a_degree2"),
        ("label", "numeral", "relativeroot", "a_simpleNumeral", "a_degree1", "a_degree2"),
        "annotation_run_sidecar", an_gate="valid_chord_label", dlc_gate="valid_chord_label",
        vocabulary_kind="open", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "chord_root", "a_root", "root_tpc", ("a_root", "a_degree1"),
        ("root", "root_tpc", "a_root", "a_degree1"), "annotation_run_sidecar",
        an_gate="valid_chord_label", dlc_gate="valid_chord_label",
        vocabulary_kind="open", mapping_status="deferred"
    ),
    TargetFamilySpec(
        "chord_quality", "a_quality", "chord_type", ("a_quality",),
        ("chord_type", "a_quality"), "annotation_run_sidecar", an_gate="valid_chord_label",
        dlc_gate="valid_chord_label", vocabulary_kind="closed_observed",
        mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "bass", "a_bass", "bass_note_tpc", ("a_bass",),
        ("bass_note", "bass_note_tpc", "a_bass"), "annotation_run_sidecar",
        an_gate="valid_chord_label", dlc_gate="valid_chord_label",
        vocabulary_kind="open", mapping_status="deferred"
    ),
    TargetFamilySpec(
        "inversion", "a_inversion", "figbass", ("a_inversion",),
        ("figbass", "a_inversion"), "annotation_run_sidecar", an_gate="valid_chord_label",
        dlc_gate="valid_chord_label", vocabulary_kind="closed_observed",
        mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "applied_secondary_harmony", "a_degree2", "relativeroot",
        ("a_degree2", "a_tonicizedKey"),
        ("relativeroot", "relativeroot_resolved", "applied_to_numeral", "a_degree2"),
        "annotation_run_sidecar", an_gate="valid_chord_label", dlc_gate="valid_chord_label",
        vocabulary_kind="open", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "borrowed_harmony", None, None, (), (), "unavailable", vocabulary_kind="unavailable",
        mapping_status="deferred"
    ),
    TargetFamilySpec(
        "cadence", None, "cadence_type", (), ("cadence", "cadence_type", "cadence_subtype"),
        "exact_observed_note_onset_point", positive_only=True, dlc_gate="valid_cadence_label",
        vocabulary_kind="closed_observed", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "phrase_boundary", None, "a_phraseend", (), ("phraseend", "a_phraseend"),
        "exact_observed_note_onset_point", positive_only=True, dlc_gate="valid_phrase_label",
        vocabulary_kind="positive_unlabeled", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "section_boundary", None, "section_start", (), ("section_start",),
        "exact_observed_note_onset_point", positive_only=True,
        dlc_gate="valid_section_start_label", vocabulary_kind="positive_unlabeled",
        mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "note_degree", "note_degree", "note_degree", ("note_degree",), ("note_degree",),
        "exact_source_note_row", an_gate="valid_chord_label", dlc_gate="valid_chord_label",
        vocabulary_kind="open", mapping_status="source_specific"
    ),
    TargetFamilySpec(
        "voice_role", None, None, (), (), "unavailable; staff/voice are optional observations",
        vocabulary_kind="unavailable", mapping_status="incompatible"
    ),
)

PROFILE_FIELDS = frozenset(
    set(AN_RAW_FIELDS)
    | set(DLC_RAW_FIELDS)
    | {field for spec in TARGET_FAMILIES for field in spec.an_source_fields}
    | {field for spec in TARGET_FAMILIES for field in spec.dlc_source_fields}
    | {
        "alt_label",
        "valid_chord_label",
        "valid_cadence_label",
        "valid_phrase_label",
        "valid_section_start_label",
        "unfolded_harmony_index",
        "a_annotationNumber",
    }
)


class DilemmadataAuditError(ValueError):
    """Raised for an invalid audit invocation or unsafe source layout."""


@dataclass(frozen=True, slots=True)
class RecordAsset:
    record_id: str
    dialect: str
    path: Path
    relative_path: str
    collection: str
    piece_name: str
    suggested_split: str | None
    score_path: Path | None
    score_relative_path: str | None
    slices_path: Path | None
    slices_relative_path: str | None


@dataclass(frozen=True, slots=True)
class Discovery:
    root: Path
    records: tuple[RecordAsset, ...]
    installation_files: tuple[Path, ...]
    auxiliary_tsv_paths: tuple[str, ...]
    unexpected_primary_paths: tuple[str, ...]


class CappedVocabulary:
    def __init__(self, limit: int = VOCABULARY_LIMIT) -> None:
        self.limit = limit
        self.values: set[str] = set()
        self.truncated = False

    def add(self, value: str) -> None:
        if value in self.values:
            return
        if len(self.values) < self.limit:
            self.values.add(value)
            return
        self.truncated = True
        largest = max(self.values)
        if value < largest:
            self.values.remove(largest)
            self.values.add(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": "open_vocabulary" if self.truncated else "observed_closed_vocabulary",
            "distinct_value_count": None if self.truncated else len(self.values),
            "distinct_value_count_lower_bound": self.limit + 1 if self.truncated else len(self.values),
            "values": sorted(self.values),
            "values_are_lexicographically_bounded": self.truncated,
        }


class FieldProfile:
    def __init__(self) -> None:
        self.files_present = 0
        self.rows_seen = 0
        self.available = 0
        self.missing = 0
        self.vocabulary = CappedVocabulary()

    def observe(self, value: str) -> None:
        self.rows_seen += 1
        normalized = value.strip()
        if normalized in MISSING_TOKENS:
            self.missing += 1
        else:
            self.available += 1
            self.vocabulary.add(normalized)

    def to_dict(self, *, record_count: int) -> dict[str, Any]:
        return {
            "files_present": self.files_present,
            "files_absent": record_count - self.files_present,
            "rows_seen": self.rows_seen,
            "available": self.available,
            "missing_or_null": self.missing,
            "bounded_observed_lexical_types": sorted(
                {_lexical_type(value) for value in self.vocabulary.values}
            ),
            "vocabulary": self.vocabulary.to_dict(),
        }


class TargetAggregate:
    def __init__(self, spec: TargetFamilySpec, dialect: str) -> None:
        self.spec = spec
        self.dialect = dialect
        self.rows = 0
        self.available = 0
        self.masked = 0
        self.missing = 0
        self.ambiguous = 0
        self.unsupported = 0
        self.source_entries = 0
        self.distinct_source_entries = 0
        self.records_with_available = 0
        self.records_without_available = 0
        self.vocabulary = CappedVocabulary()

    def to_dict(self) -> dict[str, Any]:
        fields = self.spec.an_source_fields if self.dialect == "an_joint" else self.spec.dlc_source_fields
        primary = self.spec.an_primary if self.dialect == "an_joint" else self.spec.dlc_primary
        return {
            "raw_source_fields": list(fields),
            "primary_count_field": primary,
            "coordinate_system": self.spec.coordinate,
            "mapping_status": self.spec.mapping_status,
            "vocabulary_kind": self.spec.vocabulary_kind,
            "rows_examined": self.rows,
            "available": self.available,
            "masked": self.masked,
            "missing": self.missing,
            "ambiguous": self.ambiguous,
            "unsupported": self.unsupported,
            "source_entries_after_note_row_deduplication": self.distinct_source_entries,
            "records_with_available": self.records_with_available,
            "records_without_available": self.records_without_available,
            "vocabulary": self.vocabulary.to_dict(),
        }


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if a < b:
            self.parent[b] = a
        else:
            self.parent[a] = b


class MultisetFingerprint:
    """Order-independent fixed-memory fingerprint of a multiset of digests."""

    _MODULUS = 1 << 256

    def __init__(self, domain: bytes) -> None:
        self.domain = domain
        self.count = 0
        self.total = 0
        self.squared_total = 0
        self.xor = 0

    def add(self, value: bytes) -> None:
        integer = int.from_bytes(value, "big")
        self.count += 1
        self.total = (self.total + integer) % self._MODULUS
        self.squared_total = (
            self.squared_total + integer * integer
        ) % self._MODULUS
        self.xor ^= integer

    def hexdigest(self) -> str:
        digest = sha256(self.domain)
        digest.update(self.count.to_bytes(8, "big"))
        digest.update(self.total.to_bytes(32, "big"))
        digest.update(self.squared_total.to_bytes(32, "big"))
        digest.update(self.xor.to_bytes(32, "big"))
        return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if isinstance(parsed, bool) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return sha256(_compact_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_files(root: Path) -> Iterator[Path]:
    resolved = root.resolve()
    for directory, dirnames, filenames in os.walk(resolved, followlinks=False):
        directory_path = Path(directory)
        dirnames[:] = sorted(name for name in dirnames if name != ".git")
        for name in sorted(filenames):
            path = directory_path / name
            try:
                path.resolve().relative_to(resolved)
            except (OSError, ValueError):
                continue
            if path.is_file():
                yield path


def ensure_output_outside_root(root: Path, output: Path) -> None:
    source = root.resolve()
    destination = output.resolve(strict=False)
    if destination == source or destination.is_relative_to(source):
        raise DilemmadataAuditError("output must be outside the dataset root")


def resolve_root(root: str | PathLike[str] | None) -> Path:
    supplied = str(root) if root is not None else os.environ.get(ENV_ROOT)
    if not supplied:
        raise DilemmadataAuditError(f"dataset root is required via --root or {ENV_ROOT}")
    path = Path(supplied).resolve()
    if not path.is_dir():
        raise DilemmadataAuditError(f"dataset root is not a directory: {path}")
    required = (path / "README.md", path / ".zenodo.json", path / "pitch_arrays")
    missing = [item.name for item in required if not item.exists()]
    if missing:
        raise DilemmadataAuditError(f"unsupported Dilemmadata layout; missing: {', '.join(missing)}")
    return path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def discover_dilemmadata(root: str | PathLike[str]) -> Discovery:
    root_path = resolve_root(root)
    files = tuple(sorted(_safe_files(root_path), key=lambda p: _relative(p, root_path)))
    records: list[RecordAsset] = []
    primary: set[Path] = set()

    an_root = root_path / "pitch_arrays" / "AN"
    if an_root.is_dir():
        for path in sorted(an_root.rglob("*_joint.tsv")):
            split = path.parent.name
            piece = path.name[: -len("_joint.tsv")]
            score_candidates = [
                candidate
                for suffix in (".mxl", ".musicxml")
                if (candidate := path.parent / f"{piece}{suffix}").is_file()
            ]
            score = score_candidates[0] if len(score_candidates) == 1 else None
            slices = path.parent / f"{piece}_slices.tsv"
            records.append(
                RecordAsset(
                    record_id=f"an:{split}:{piece}", dialect="an_joint", path=path,
                    relative_path=_relative(path, root_path), collection=piece.split("-", 1)[0],
                    piece_name=piece, suggested_split=split, score_path=score,
                    score_relative_path=_relative(score, root_path) if score else None,
                    slices_path=slices if slices.is_file() else None,
                    slices_relative_path=_relative(slices, root_path) if slices.is_file() else None,
                )
            )
            primary.add(path.resolve())

    dlc_root = root_path / "pitch_arrays" / "DLC"
    if dlc_root.is_dir():
        for path in sorted(dlc_root.rglob("*.tsv")):
            rel = path.relative_to(dlc_root)
            collection = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            piece = path.stem
            records.append(
                RecordAsset(
                    record_id=f"dlc:{collection}:{piece}", dialect="dlc", path=path,
                    relative_path=_relative(path, root_path), collection=collection,
                    piece_name=piece, suggested_split=None, score_path=None,
                    score_relative_path=None, slices_path=None, slices_relative_path=None,
                )
            )
            primary.add(path.resolve())

    all_pitch_tsv = {
        path.resolve() for path in (root_path / "pitch_arrays").rglob("*.tsv")
    }
    auxiliary = tuple(sorted(_relative(path, root_path) for path in all_pitch_tsv - primary))
    unexpected = tuple(
        path for path in auxiliary
        if not (path.endswith("_slices.tsv") or path.endswith("dataset_summary.tsv"))
    )
    return Discovery(
        root=root_path,
        records=tuple(sorted(records, key=lambda item: item.record_id)),
        installation_files=files,
        auxiliary_tsv_paths=auxiliary,
        unexpected_primary_paths=unexpected,
    )


def _bool_value(value: str) -> bool | None:
    normalized = value.strip()
    if normalized in TRUE_TOKENS:
        return True
    if normalized in FALSE_TOKENS:
        return False
    return None


def _lexical_type(value: str) -> str:
    if value.lower() in {"true", "false"}:
        return "boolean"
    try:
        int(value)
    except ValueError:
        try:
            Decimal(value)
        except InvalidOperation:
            try:
                Fraction(value)
            except (ValueError, ZeroDivisionError):
                return "string"
            return "fraction"
        return "decimal"
    return "integer"


def _ratio(div_value: str, musical_value: Fraction) -> Fraction | None:
    if musical_value == 0:
        return None
    return Fraction(int(div_value), 1) / musical_value


def _note_hash(onset: Fraction, duration: Fraction, pitch: int) -> bytes:
    return sha256(_compact_bytes([
        [onset.numerator, onset.denominator],
        [duration.numerator, duration.denominator],
        pitch,
    ])).digest()


def _target_primary(spec: TargetFamilySpec, dialect: str) -> str | None:
    return spec.an_primary if dialect == "an_joint" else spec.dlc_primary


def _target_gate(spec: TargetFamilySpec, dialect: str) -> str | None:
    return spec.an_gate if dialect == "an_joint" else spec.dlc_gate


def _source_entry_key(
    spec: TargetFamilySpec,
    dialect: str,
    row: Mapping[str, str],
    onset: Fraction | None,
    value: str,
    ordinal: int,
) -> tuple[str, ...]:
    if spec.coordinate == "piece_or_repeated_note_sidecar":
        return (spec.family, value)
    if spec.coordinate == "exact_source_note_row":
        return (spec.family, "row", str(ordinal))
    if spec.coordinate == "exact_observed_note_onset_point":
        if onset is None:
            return (spec.family, "unresolved", value)
        return (
            spec.family,
            "onset",
            str(onset.numerator),
            str(onset.denominator),
            value,
        )
    identity_field = "a_annotationNumber" if dialect == "an_joint" else "unfolded_harmony_index"
    identity = row.get(identity_field, "").strip()
    if identity not in MISSING_TOKENS:
        return (spec.family, identity, value)
    if onset is None:
        return (spec.family, "unresolved", value)
    return (spec.family, str(onset.numerator), str(onset.denominator), value)


def _scan_record(
    asset: RecordAsset,
    *,
    profiles: dict[tuple[str, str], FieldProfile],
    target_aggregates: dict[tuple[str, str], TargetAggregate],
) -> dict[str, Any]:
    required = AN_REQUIRED_FIELDS if asset.dialect == "an_joint" else DLC_REQUIRED_FIELDS
    raw_fields = AN_RAW_FIELDS if asset.dialect == "an_joint" else DLC_RAW_FIELDS
    errors: Counter[str] = Counter()
    error_examples: dict[str, dict[str, Any]] = {}
    note_fingerprint = MultisetFingerprint(
        b"dilemmadata.midi-compatible-note-multiset.2\0"
    )
    raw_digest = sha256(f"dilemmadata.raw-observation.{asset.dialect}.1\0".encode())
    target_digest = sha256(f"dilemmadata.target-sidecar.{asset.dialect}.1\0".encode())
    row_count = 0
    last_onset: Fraction | None = None
    onset_order_violations = 0
    zero_duration_rows = 0
    tie_continuation_rows = 0
    pitch_spelling_mismatches = 0
    resolution_values: set[int] = set()
    resolution_mismatches = 0
    voice_keys: set[tuple[str, ...]] = set()
    meter_values: set[tuple[str, str]] = set()
    family_events: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    family_available: Counter[str] = Counter()
    family_rows: dict[str, Counter[str]] = defaultdict(Counter)

    with asset.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = next(reader)
        except StopIteration:
            header = []
            errors["empty_file"] += 1
        if len(header) != len(set(header)):
            errors["duplicate_header_field"] += 1
        missing_required = sorted(set(required) - set(header))
        if missing_required:
            errors["missing_required_field"] += len(missing_required)
        indices = {name: index for index, name in enumerate(header)}
        profile_indices = [
            (index, field)
            for index, field in enumerate(header)
            if field in PROFILE_FIELDS
        ]
        for _, field in profile_indices:
            profiles[(asset.dialect, field)].files_present += 1
        raw_indices = [(index, field) for index, field in enumerate(header) if field in raw_fields]
        target_indices = [(index, field) for index, field in enumerate(header) if field not in raw_fields]

        for ordinal, values in enumerate(reader, start=2):
            row_count += 1
            if len(values) != len(header):
                errors["row_width_mismatch"] += 1
                error_examples.setdefault("row_width_mismatch", {
                    "relative_path": asset.relative_path,
                    "line": ordinal,
                    "expected_columns": len(header),
                    "actual_columns": len(values),
                })
                continue
            row = {name: values[index] for name, index in indices.items()}
            for index, field in profile_indices:
                profiles[(asset.dialect, field)].observe(values[index])
            raw_digest.update(_compact_bytes([[field, values[index]] for index, field in raw_indices]))
            target_digest.update(_compact_bytes([[field, values[index]] for index, field in target_indices]))

            onset: Fraction | None = None
            duration: Fraction | None = None
            try:
                if asset.dialect == "an_joint":
                    onset = Fraction(row["s_offset_frac"])
                    duration = Fraction(row["s_duration_frac"])
                    pitch = int(row["s_midi"])
                    onset_ratio = _ratio(row["onset_div"], onset)
                    duration_ratio = _ratio(row["duration_div"], duration)
                    tie_value = _bool_value(row["s_isOnset"])
                    voice_keys.add((row["s_part_id"], row["s_voice_id"]))
                    step = row["s_step"].strip()
                    alter = row["s_alter"].strip()
                else:
                    onset = Fraction(row["quarterbeats_playthrough"])
                    duration = Fraction(row["duration"]) * 4
                    pitch = int(row["pitch"])
                    onset_ratio = _ratio(row["onset_div"], onset)
                    duration_ratio_raw = _ratio(row["duration_div"], Fraction(row["duration"]))
                    duration_ratio = duration_ratio_raw / 4 if duration_ratio_raw is not None else None
                    tie_value = _bool_value(row["is_note_onset"])
                    voice_keys.add((row["staff"], row["voice"]))
                    step = row["step"].strip()
                    alter = row["alter"].strip()
                if onset < 0:
                    raise ValueError("negative onset")
                if duration < 0:
                    raise ValueError("negative duration")
                if not 0 <= pitch <= 127:
                    raise ValueError("pitch outside MIDI range")
                if duration == 0:
                    zero_duration_rows += 1
                if last_onset is not None and onset < last_onset:
                    onset_order_violations += 1
                last_onset = onset
                if tie_value is False:
                    tie_continuation_rows += 1
                elif tie_value is None:
                    errors["malformed_tie_flag"] += 1
                ratios = [ratio for ratio in (onset_ratio, duration_ratio) if ratio is not None]
                for ratio in ratios:
                    if ratio.denominator != 1 or ratio <= 0:
                        resolution_mismatches += 1
                    else:
                        resolution_values.add(ratio.numerator)
                if len(set(ratios)) > 1:
                    resolution_mismatches += 1
                note_fingerprint.add(_note_hash(onset, duration, pitch))
                if step and alter not in MISSING_TOKENS:
                    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}.get(step)
                    if pc is None or (pc + int(alter)) % 12 != pitch % 12:
                        pitch_spelling_mismatches += 1
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                category = "malformed_raw_note"
                errors[category] += 1
                error_examples.setdefault(category, {
                    "relative_path": asset.relative_path,
                    "line": ordinal,
                    "error_type": type(exc).__name__,
                    "message": " ".join(str(exc).split())[:200],
                })

            meter_values.add((row.get("ts_beats", ""), row.get("ts_beat_type", "")))
            alt_present = row.get("alt_label", "").strip() not in MISSING_TOKENS
            for spec in TARGET_FAMILIES:
                aggregate = target_aggregates[(asset.dialect, spec.family)]
                aggregate.rows += 1
                family_rows[spec.family]["rows"] += 1
                primary = _target_primary(spec, asset.dialect)
                gate_field = _target_gate(spec, asset.dialect)
                if primary is None or primary not in indices:
                    aggregate.masked += 1
                    family_rows[spec.family]["masked"] += 1
                    continue
                gate = True
                if gate_field is not None:
                    gate_raw = row.get(gate_field, "")
                    gate_value = _bool_value(gate_raw)
                    if gate_value is None and gate_raw.strip() not in MISSING_TOKENS:
                        aggregate.unsupported += 1
                    gate = gate_value is True
                value = row.get(primary, "").strip()
                if not gate:
                    aggregate.masked += 1
                    family_rows[spec.family]["masked"] += 1
                    continue
                if spec.positive_only:
                    boolean = _bool_value(value)
                    if boolean is True or (boolean is None and value not in MISSING_TOKENS):
                        available_value = value
                    else:
                        aggregate.masked += 1
                        family_rows[spec.family]["masked"] += 1
                        continue
                elif value in MISSING_TOKENS:
                    aggregate.missing += 1
                    family_rows[spec.family]["missing"] += 1
                    continue
                else:
                    available_value = value
                aggregate.available += 1
                aggregate.source_entries += 1
                aggregate.vocabulary.add(available_value)
                family_available[spec.family] += 1
                family_rows[spec.family]["available"] += 1
                key = _source_entry_key(
                    spec, asset.dialect, row, onset, available_value, row_count
                )
                family_events[spec.family].add(key)
                if alt_present and spec.family in {
                    "global_key", "local_key", "tonal_region", "roman_numeral",
                    "chord_root", "chord_quality", "bass", "inversion",
                    "applied_secondary_harmony",
                }:
                    aggregate.ambiguous += 1
                    family_rows[spec.family]["ambiguous"] += 1

    if len(resolution_values) > 1:
        errors["inconsistent_source_resolution"] += 1
    if resolution_mismatches:
        errors["division_coordinate_mismatch"] += resolution_mismatches
    if onset_order_violations:
        errors["nonmonotonic_note_order"] += onset_order_violations

    for spec in TARGET_FAMILIES:
        aggregate = target_aggregates[(asset.dialect, spec.family)]
        distinct = len(family_events[spec.family])
        aggregate.distinct_source_entries += distinct
        if family_available[spec.family]:
            aggregate.records_with_available += 1
        else:
            aggregate.records_without_available += 1

    score_sha = _hash_file(asset.score_path) if asset.score_path is not None else None
    quarantine_categories = sorted(errors)
    return {
        "record_id": asset.record_id,
        "dialect": asset.dialect,
        "relative_path": asset.relative_path,
        "collection": asset.collection,
        "piece_name": asset.piece_name,
        "suggested_split": asset.suggested_split,
        "score_relative_path": asset.score_relative_path,
        "score_sha256": score_sha,
        "slices_relative_path": asset.slices_relative_path,
        "file_sha256": _hash_file(asset.path),
        "header": header,
        "header_sha256": sha256("\t".join(header).encode("utf-8")).hexdigest(),
        "row_count": row_count,
        "raw_observation_fingerprint": raw_digest.hexdigest(),
        "midi_compatible_note_projection_fingerprint": note_fingerprint.hexdigest(),
        "target_sidecar_fingerprint": target_digest.hexdigest(),
        "source_resolution_candidates": sorted(resolution_values),
        "voice_identity_count": len(voice_keys),
        "meter_value_count": len(meter_values),
        "zero_duration_row_count": zero_duration_rows,
        "tie_continuation_row_count": tie_continuation_rows,
        "pitch_spelling_mismatch_count": pitch_spelling_mismatches,
        "target_family_counts": {
            family: dict(sorted(counts.items())) for family, counts in sorted(family_rows.items())
        },
        "quarantine_categories": quarantine_categories,
        "error_counts": dict(sorted(errors.items())),
        "error_examples": [error_examples[key] for key in sorted(error_examples)],
        "raw_compatible_note_projection": not quarantine_categories,
    }


def _inventory(root: Path, paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative_path = _relative(path, root)
        rows.append({
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
            "suffix": path.suffix.lower() if path.suffix else "[no_extension]",
        })
    return rows, _fingerprint([[row["relative_path"], row["size_bytes"], row["sha256"]] for row in rows])


def _compare_upstream(root: Path, upstream_root: Path | None) -> dict[str, Any]:
    if upstream_root is None:
        return {
            "performed": False,
            "exact_match": None,
            "upstream_commit": None,
            "matching_file_count": 0,
            "mismatches": [],
            "local_only": [],
            "upstream_only": [],
        }
    upstream = upstream_root.resolve()
    if not upstream.is_dir():
        raise DilemmadataAuditError(f"upstream root is not a directory: {upstream}")
    local = {_relative(path, root): _hash_file(path) for path in _safe_files(root)}
    remote = {_relative(path, upstream): _hash_file(path) for path in _safe_files(upstream)}
    common = sorted(set(local) & set(remote))
    mismatches = [path for path in common if local[path] != remote[path]]
    commit = None
    git_dir = upstream / ".git"
    if git_dir.exists():
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    return {
        "performed": True,
        "exact_match": not mismatches and set(local) == set(remote) and commit == RELEASE_COMMIT,
        "upstream_commit": commit,
        "matching_file_count": len(common) - len(mismatches),
        "mismatches": mismatches,
        "local_only": sorted(set(local) - set(remote)),
        "upstream_only": sorted(set(remote) - set(local)),
    }


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _metadata_evidence(root: Path, assets: Sequence[RecordAsset]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata: dict[str, dict[str, Any]] = {asset.record_id: {} for asset in assets}
    an_by_piece = {asset.piece_name: asset for asset in assets if asset.dialect == "an_joint"}
    dlc_by_key = {
        (asset.collection, asset.piece_name): asset for asset in assets if asset.dialect == "dlc"
    }

    an_summary = _read_tsv_rows(root / "pitch_arrays" / "AN" / "dataset_summary.tsv")
    for row in an_summary:
        asset = an_by_piece.get(row.get("file", ""))
        if asset is None:
            continue
        metadata[asset.record_id] = {
            "composer": row.get("s_composer") or row.get("a_composer") or None,
            "title": row.get("s_title") or row.get("a_title") or None,
            "movement": row.get("s_movementName") or None,
            "analyst": row.get("a_analyst") or None,
            "proofreader": row.get("a_proofreader") or None,
        }

    dlc_metadata_path = root / "processing" / "DLC" / "distant_listening_corpus.metadata.tsv"
    dlc_summary_rows = _read_tsv_rows(dlc_metadata_path)
    duplicate_metadata_keys: Counter[tuple[str, str]] = Counter()
    for row in dlc_summary_rows:
        key = (row.get("corpus", ""), row.get("piece", ""))
        duplicate_metadata_keys[key] += 1
        asset = dlc_by_key.get(key)
        if asset is None or metadata[asset.record_id]:
            continue
        split = (row.get("split") or "").strip() or None
        metadata[asset.record_id] = {
            "composer": row.get("composer") or row.get("composer_text") or None,
            "title": row.get("workTitle") or row.get("title_text") or None,
            "movement": row.get("movementTitle") or row.get("movementNumber") or None,
            "annotators": row.get("annotators") or None,
            "reviewers": row.get("reviewers") or None,
            "source_version": row.get("last_modified") or None,
            "source_url": row.get("last_modified_url") or None,
            "suggested_split": split,
        }

    creators = {
        row.get("composer") for row in metadata.values() if row.get("composer")
    }
    work_keys = {
        (row.get("composer"), row.get("title"), row.get("movement"))
        for row in metadata.values()
        if row.get("composer") or row.get("title") or row.get("movement")
    }
    evidence = {
        "an_summary_row_count": len(an_summary),
        "dlc_metadata_row_count": len(dlc_summary_rows),
        "records_with_metadata": sum(bool(row) for row in metadata.values()),
        "records_without_metadata": sum(not row for row in metadata.values()),
        "distinct_nonempty_creator_strings": len(creators),
        "distinct_nonempty_work_metadata_keys": len(work_keys),
        "duplicate_dlc_metadata_key_count": sum(count - 1 for count in duplicate_metadata_keys.values() if count > 1),
    }
    return evidence, metadata


def _grouping(
    root: Path,
    assets: Sequence[RecordAsset],
    scans: Sequence[dict[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in scans}
    uf = UnionFind(by_id)
    edge_counts: Counter[str] = Counter()

    def connect_groups(key_name: str, rows: Iterable[tuple[str, str | None]]) -> None:
        grouped: dict[str, list[str]] = defaultdict(list)
        for record_id, key in rows:
            if key:
                grouped[key].append(record_id)
        for members in grouped.values():
            ordered = sorted(set(members))
            for other in ordered[1:]:
                uf.union(ordered[0], other)
                edge_counts[key_name] += 1

    connect_groups(
        "midi_compatible_projection",
        ((row["record_id"], row["midi_compatible_note_projection_fingerprint"]) for row in scans),
    )
    connect_groups(
        "an_score_content",
        ((row["record_id"], row["score_sha256"]) for row in scans if row["dialect"] == "an_joint"),
    )

    an_lookup = {asset.piece_name: asset.record_id for asset in assets if asset.dialect == "an_joint"}
    dlc_lookup = {
        (asset.collection, asset.piece_name): asset.record_id for asset in assets if asset.dialect == "dlc"
    }
    explicit_links: list[dict[str, str]] = []
    for row in _read_tsv_rows(root / "processing" / "merged_summary.tsv"):
        an_id = an_lookup.get((row.get("id_v100") or "").strip())
        dlc_id = dlc_lookup.get(((row.get("corpus_dlc") or "").strip(), (row.get("piece") or "").strip()))
        if an_id and dlc_id:
            uf.union(an_id, dlc_id)
            edge_counts["merged_summary_explicit_overlap"] += 1
            explicit_links.append({"an_record_id": an_id, "dlc_record_id": dlc_id})

    components: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(by_id):
        components[uf.find(record_id)].append(record_id)
    component_rows = []
    split_conflicts = []
    alternative_clusters = []
    for members in sorted(components.values(), key=lambda values: (len(values), values)):
        component_id = f"dilemmadata-component:{_fingerprint(members)}"
        splits = sorted({
            split
            for record_id in members
            if (split := (by_id[record_id].get("suggested_split") or metadata.get(record_id, {}).get("suggested_split")))
        })
        target_fingerprints = sorted({by_id[record_id]["target_sidecar_fingerprint"] for record_id in members})
        raw_fingerprints = sorted({by_id[record_id]["midi_compatible_note_projection_fingerprint"] for record_id in members})
        row = {
            "component_id": component_id,
            "record_ids": members,
            "suggested_splits": splits,
            "raw_projection_fingerprints": raw_fingerprints,
            "target_sidecar_fingerprint_count": len(target_fingerprints),
        }
        if len(members) > 1:
            component_rows.append(row)
        if len(splits) > 1:
            split_conflicts.append(row)
        if len(raw_fingerprints) == 1 and len(target_fingerprints) > 1 and len(members) > 1:
            alternative_clusters.append(row)

    equivalent_groups = defaultdict(list)
    for row in scans:
        equivalent_groups[row["midi_compatible_note_projection_fingerprint"]].append(row["record_id"])
    duplicate_clusters = [sorted(values) for values in equivalent_groups.values() if len(values) > 1]
    return {
        "policy": "transitive closure over exact MIDI-compatible note projection, exact AN score bytes, and explicit merged-summary overlap",
        "identity_non_goal": "composer/title similarity alone never joins a split component",
        "edge_counts": dict(sorted(edge_counts.items())),
        "component_count": len(components),
        "multi_record_component_count": len(component_rows),
        "multi_record_components": component_rows,
        "exact_equivalent_input_cluster_count": len(duplicate_clusters),
        "exact_equivalent_input_record_count": sum(len(values) for values in duplicate_clusters),
        "alternative_analysis_cluster_count": len(alternative_clusters),
        "alternative_analysis_record_count": sum(len(row["record_ids"]) for row in alternative_clusters),
        "alternative_analysis_clusters": alternative_clusters,
        "explicit_cross_source_overlap_count": len(explicit_links),
        "explicit_cross_source_overlaps": sorted(explicit_links, key=lambda row: (row["an_record_id"], row["dlc_record_id"])),
        "suggested_split_conflict_count": len(split_conflicts),
        "suggested_split_conflicts": split_conflicts,
        "final_splits_assigned": False,
    }


def _header_shapes(scans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    headers: dict[tuple[str, str], Sequence[str]] = {}
    for row in scans:
        key = (str(row["dialect"]), str(row["header_sha256"]))
        grouped[key].append(str(row["relative_path"]))
        headers[key] = row["header"]
    result = []
    for key, paths in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        result.append({
            "dialect": key[0],
            "header_sha256": key[1],
            "file_count": len(paths),
            "column_count": len(headers[key]),
            "columns": list(headers[key]),
            "representative_path": sorted(paths)[0],
        })
    return result


def _manifest_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    inventory = report["corpus_identity"]["inventory"]
    return {
        "audit_schema_version": report["audit_schema_version"],
        "semantic_fingerprint": report["semantic_fingerprint"],
        "corpus": {
            "release_version": report["corpus_identity"]["release_version"],
            "release_commit": report["corpus_identity"]["release_commit"],
            "installation_file_count": inventory["file_count"],
            "installation_byte_count": inventory["byte_count"],
            "content_fingerprint": inventory["content_fingerprint"],
            "upstream_exact_match": report["corpus_identity"]["upstream_comparison"]["exact_match"],
        },
        "records": report["record_inventory"],
        "formats": {
            "header_shape_count": len(report["formats"]["header_shapes"]),
            "header_shape_fingerprint": _fingerprint(report["formats"]["header_shapes"]),
        },
        "raw_projection": report["raw_musical_representation"]["acceptance_counts"],
        "targets": {
            family: {
                dialect: {
                    key: values[key]
                    for key in (
                        "available", "masked", "missing", "ambiguous", "unsupported",
                        "source_entries_after_note_row_deduplication", "records_with_available",
                    )
                }
                for dialect, values in family_values["by_dialect"].items()
            }
            for family, family_values in report["target_inventory"].items()
        },
        "grouping": {
            key: report["grouping"][key]
            for key in (
                "component_count", "multi_record_component_count",
                "exact_equivalent_input_cluster_count", "exact_equivalent_input_record_count",
                "alternative_analysis_cluster_count", "alternative_analysis_record_count",
                "explicit_cross_source_overlap_count", "suggested_split_conflict_count",
            )
        },
        "quarantine": report["quarantine"],
        "readiness": report["readiness"],
    }


def build_report(
    root: str | PathLike[str],
    *,
    upstream_root: str | PathLike[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    root_path = resolve_root(root)
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise DilemmadataAuditError("limit must be a positive integer or null")
    discovery = discover_dilemmadata(root_path)
    selected = discovery.records if limit is None else discovery.records[:limit]
    inventory_rows, content_fingerprint = _inventory(root_path, discovery.installation_files)
    suffix_counts = Counter(row["suffix"] for row in inventory_rows)
    profiles: dict[tuple[str, str], FieldProfile] = defaultdict(FieldProfile)
    target_aggregates = {
        (dialect, spec.family): TargetAggregate(spec, dialect)
        for dialect in ("an_joint", "dlc")
        for spec in TARGET_FAMILIES
    }
    scans = [
        _scan_record(asset, profiles=profiles, target_aggregates=target_aggregates)
        for asset in selected
    ]
    metadata_evidence, metadata = _metadata_evidence(root_path, selected)
    grouping = _grouping(root_path, selected, scans, metadata)
    record_counts = Counter(row["dialect"] for row in scans)
    row_counts = Counter()
    for row in scans:
        row_counts[row["dialect"]] += row["row_count"]
    quarantine_counts = Counter(
        category for row in scans for category in row["quarantine_categories"]
    )
    quarantined_records = [row for row in scans if row["quarantine_categories"]]
    raw_ready_count = sum(bool(row["raw_compatible_note_projection"]) for row in scans)

    target_inventory: dict[str, Any] = {}
    for spec in TARGET_FAMILIES:
        by_dialect = {
            dialect: target_aggregates[(dialect, spec.family)].to_dict()
            for dialect in ("an_joint", "dlc")
        }
        target_inventory[spec.family] = {
            "semantic_definition_status": "source-native evidence retained; normalization is not inferred by Phase 9A",
            "by_dialect": by_dialect,
            "cross_source_mapping": spec.mapping_status,
            "provenance": "release pitch-array column plus release processing/specification evidence",
            "confidence": "no calibrated numeric confidence field observed",
            "exact_alignment": spec.coordinate,
        }

    strict_violations: list[str] = []
    upstream = _compare_upstream(
        root_path, Path(upstream_root) if upstream_root is not None else None
    )
    if upstream["performed"] and not upstream["exact_match"]:
        strict_violations.append("upstream_release_content_mismatch")
    if discovery.unexpected_primary_paths:
        strict_violations.append("unexpected_primary_tsv_layout")
    if quarantined_records:
        strict_violations.append("primary_record_quarantines_present")
    if limit is not None:
        strict_violations.append("bounded_limit_is_not_complete_evidence")

    evidence_warnings = []
    production_blockers = ["phase_9b_production_adapter_not_implemented"]
    if grouping["suggested_split_conflict_count"]:
        evidence_warnings.append("suggested_split_components_require_group_aware_reassignment")
        production_blockers.append("release_suggested_splits_conflict_with_transitive_source_groups")

    report: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "corpus_identity": {
            "corpus_id": CORPUS_ID,
            "official_name": CORPUS_NAME,
            "source_repository": UPSTREAM_REPOSITORY,
            "release_version": RELEASE_VERSION,
            "release_commit": RELEASE_COMMIT,
            "publication_date": "2026-02-04",
            "producers": list(RELEASE_PRODUCERS),
            "source_lineage": list(SOURCE_LINEAGE),
            "license_identifier": LICENSE_ID,
            "license_text_file_present": any(row["relative_path"].lower().startswith("license") for row in inventory_rows),
            "citation_evidence": {
                "dataset_metadata": ".zenodo.json",
                "dataset_specific_citation_file_present": any(
                    Path(row["relative_path"]).name.lower() in {"citation.cff", "citation.bib"}
                    for row in inventory_rows
                ),
                "upstream_citations_are_described_in": "README.md",
            },
            "installation_kind": "exact release snapshot with processed pitch arrays and processing source; not a nested git checkout",
            "nested_git_checkout": (root_path / ".git").exists(),
            "upstream_comparison": upstream,
            "inventory": {
                "file_count": len(inventory_rows),
                "byte_count": sum(row["size_bytes"] for row in inventory_rows),
                "suffix_counts": dict(sorted(suffix_counts.items())),
                "content_fingerprint": content_fingerprint,
                "files": inventory_rows,
            },
        },
        "record_inventory": {
            "discovered_primary_record_count": len(discovery.records),
            "selected_primary_record_count": len(selected),
            "an_joint_record_count": record_counts["an_joint"],
            "dlc_record_count": record_counts["dlc"],
            "an_score_file_count": sum(asset.score_path is not None for asset in selected if asset.dialect == "an_joint"),
            "an_slices_file_count": sum(asset.slices_path is not None for asset in selected if asset.dialect == "an_joint"),
            "auxiliary_pitch_array_tsv_count": len(discovery.auxiliary_tsv_paths),
            "unexpected_primary_path_count": len(discovery.unexpected_primary_paths),
            "note_row_count": sum(row["row_count"] for row in scans),
            "note_rows_by_dialect": dict(sorted(row_counts.items())),
            "collection_count": len({asset.collection for asset in selected}),
            "metadata": metadata_evidence,
            "limit": limit,
        },
        "formats": {
            "encoding": "UTF-8 strict",
            "delimiter": "tab",
            "compression": {
                "primary_tsv": "none",
                "mxl_score_assets": "ZIP container",
                "musicxml_score_assets": "uncompressed XML",
            },
            "primary_dialects": ["an_joint", "dlc"],
            "derived_non_primary_formats": ["an_slices"],
            "header_shapes": _header_shapes(scans),
            "field_profiles": {
                dialect: {
                    field: profiles[(dialect, field)].to_dict(record_count=record_counts[dialect])
                    for observed_dialect, field in sorted(profiles)
                    if observed_dialect == dialect
                }
                for dialect in ("an_joint", "dlc")
            },
        },
        "raw_musical_representation": {
            "classification": "score-derived symbolic note arrays, not raw MIDI",
            "pitch": "MIDI-compatible absolute pitch; pitch class is derivable",
            "spelling": "score-derived step/alter is optional metadata and is not required by the MIDI-compatible projection",
            "timing": {
                "an_joint": "s_offset_frac and s_duration_frac are exact quarter-note fractions; onset_div/duration_div share a per-record proportional resolution",
                "dlc": "quarterbeats_playthrough is exact qn; duration is a whole-note fraction and converts exactly by multiplying by four; divisions corroborate a per-record resolution",
                "float_fields": "onset_beat/beat_float/s_beat_float are diagnostic only and are not exact alignment coordinates",
            },
            "voice_and_part": "AN part/voice and DLC staff/voice are optional score observations; no semantic role is inferred or required",
            "meter": "per-note numerator/denominator and measure coordinates are observable processed score evidence; exact silent meter-event reconstruction still needs Phase 9B validation",
            "tempo": "not present; a canonical adapter may only use the existing explicit provenance-bearing default-tempo rule",
            "ties": "is-onset fields expose tied continuations; exact Phase 9B note identity must handle them without invented durations",
            "zero_duration": "source-zero durations are raw grace-note candidates; Phase 9B must map them to is_grace=true only under an explicit tested rule and quarantine contradictory evidence",
            "rests_articulations_dynamics": "not represented as primary note rows",
            "polyphonic_score_like_sequence": True,
            "target_independent_note_projection": True,
            "canonical_piece_without_theory": "possible for exact note evidence, subject to Phase 9B meter/bar/tie validation and structured quarantine",
            "acceptance_counts": {
                "raw_compatible_note_projection_records": raw_ready_count,
                "raw_projection_quarantine_records": len(scans) - raw_ready_count,
                "zero_duration_rows": sum(row["zero_duration_row_count"] for row in scans),
                "records_with_zero_duration_rows": sum(
                    row["zero_duration_row_count"] > 0 for row in scans
                ),
                "tie_continuation_rows": sum(row["tie_continuation_row_count"] for row in scans),
                "records_with_tie_continuations": sum(
                    row["tie_continuation_row_count"] > 0 for row in scans
                ),
                "pitch_spelling_mismatches": sum(row["pitch_spelling_mismatch_count"] for row in scans),
                "records_with_one_source_resolution": sum(len(row["source_resolution_candidates"]) == 1 for row in scans),
                "records_with_multiple_source_resolutions": sum(len(row["source_resolution_candidates"]) > 1 for row in scans),
            },
        },
        "target_inventory": target_inventory,
        "grouping": grouping,
        "alignment_contract": {
            "note_identity": "exact source row plus dialect-specific tie handling; no nearest-neighbour or float match",
            "onset_point": "exact Fraction coordinate only",
            "spans": "retain source label identity/run as a target sidecar; do not infer an end beyond the next exact evidenced boundary",
            "boundaries": "available only at exact observed note onset; unmatched or between-note source events stay unaligned/quarantined",
            "voice_identity": "optional source part/staff/voice tuple; never semantic role and never required for raw-compatible graph topology",
            "forbidden": [
                "float snapping", "priority-based overlap choice", "missing-to-negative coercion",
                "target-derived notes/nodes/edges/features", "target fields in graph or model-input fingerprints",
            ],
        },
        "leakage_contract": {
            "raw_input_fields": {
                "an_joint": sorted(AN_RAW_FIELDS),
                "dlc": sorted(DLC_RAW_FIELDS),
            },
            "target_only_fields": {
                "an_joint": sorted({field for spec in TARGET_FAMILIES for field in spec.an_source_fields}),
                "dlc": sorted({field for spec in TARGET_FAMILIES for field in spec.dlc_source_fields}),
            },
            "grouping_identity": "external transitive component sidecar",
            "provenance_diagnostics_confidence": "sidecars only",
            "required_phase9b_mutation_tests": [
                "delete_replace_reorder_theory_annotations_preserves_raw_canonical_projection",
                "delete_replace_reorder_theory_annotations_preserves_graph_stores",
                "delete_replace_reorder_theory_annotations_preserves_graph_fingerprint",
                "delete_replace_reorder_theory_annotations_preserves_model_input_fingerprint",
                "alternative_analyses_share_one_split_component",
                "target_derived_pitch_arrays_never_create_notes_without_separate_score_provenance",
            ],
        },
        "quarantine": {
            "record_count": len(quarantined_records),
            "category_counts": dict(sorted(quarantine_counts.items())),
            "records": [
                {
                    "record_id": row["record_id"],
                    "relative_path": row["relative_path"],
                    "categories": row["quarantine_categories"],
                    "error_counts": row["error_counts"],
                    "examples": row["error_examples"],
                }
                for row in quarantined_records
            ],
        },
        "representative_records": {
            "selection_policy": "first lexicographic record for each dialect/header shape plus every quarantine record",
            "records": [
                {
                    "record_id": row["record_id"],
                    "relative_path": row["relative_path"],
                    "dialect": row["dialect"],
                    "row_count": row["row_count"],
                    "header_sha256": row["header_sha256"],
                    "quarantine_categories": row["quarantine_categories"],
                }
                for row in sorted(
                    {
                        (row["dialect"], row["header_sha256"]): row for row in reversed(scans)
                    }.values(),
                    key=lambda row: row["record_id"],
                )
            ] + [
                {
                    "record_id": row["record_id"],
                    "relative_path": row["relative_path"],
                    "dialect": row["dialect"],
                    "row_count": row["row_count"],
                    "header_sha256": row["header_sha256"],
                    "quarantine_categories": row["quarantine_categories"],
                }
                for row in quarantined_records
            ],
        },
        "per_record": scans,
        "readiness": {
            "evidence_contract_ready": not strict_violations,
            "production_adapter_ready": False,
            "evidence_violations": strict_violations,
            "evidence_warnings": evidence_warnings,
            "production_blockers": production_blockers,
        },
        "runtime_diagnostics": {
            "absolute_paths_included": False,
            "timings_included": False,
            "platform_dependent_values_in_semantic_fingerprint": False,
        },
    }
    semantic_payload = dict(report)
    semantic_payload.pop("runtime_diagnostics")
    report["semantic_fingerprint"] = _fingerprint(semantic_payload)
    return report


def manifest_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact committed acceptance projection."""

    return _manifest_projection(report)


def dumps_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_report(report), encoding="utf-8", newline="\n")


def check_manifest(report: Mapping[str, Any], manifest_path: Path) -> tuple[bool, str | None]:
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DilemmadataAuditError(f"cannot read audit manifest {manifest_path}: {exc}") from exc
    actual = manifest_projection(report)
    if actual == expected:
        return True, None
    return False, f"manifest mismatch: expected {_fingerprint(expected)}, actual {_fingerprint(actual)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help=f"dataset root; defaults to {ENV_ROOT}")
    parser.add_argument("--upstream-root", type=Path, help="optional clean v1.0 checkout for byte comparison")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--check", nargs="?", const=DEFAULT_MANIFEST, type=Path, metavar="MANIFEST",
        help="compare the compact projection with MANIFEST (default: committed audit manifest)",
    )
    args = parser.parse_args(argv)
    try:
        root = resolve_root(args.root)
        ensure_output_outside_root(root, args.output)
        report = build_report(
            root, upstream_root=args.upstream_root, limit=args.limit
        )
        write_report(report, args.output)
        if args.check is not None:
            matches, message = check_manifest(report, args.check)
            if not matches:
                print(message, file=sys.stderr)
                return 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        message = " ".join(str(exc).split())[:500]
        print(f"Dilemmadata audit failed: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
