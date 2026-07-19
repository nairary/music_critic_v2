#!/usr/bin/env python3
"""Read-only, deterministic audit of local HookTheory and legacy V1 artifacts.

This is audit tooling, not a production adapter.  It deliberately lives outside
``src/music_critic`` and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


REPORT_SCHEMA_VERSION = "hooktheory_legacy_audit_v1"
BOUNDED_EXAMPLES = 12
RAW_FIELDS = (
    "beat",
    "duration",
    "sd",
    "octave",
    "isRest",
    "root",
    "type",
    "inversion",
    "applied",
    "adds",
    "omits",
    "alterations",
    "suspensions",
    "borrowed",
    "alternate",
    "pedal",
    "beatUnit",
    "numBeats",
)
PRIMARY_HOOKTHEORY_NAMES = (
    "Hooktheory.json",
    "Hooktheory_Train_Segments.json",
    "Hooktheory_Valid_Segments.json",
    "Hooktheory_Test_Segments.json",
    "HookTheoryKey.train.jsonl",
    "HookTheoryKey.val.jsonl",
    "HookTheoryKey.test.jsonl",
    "HookTheoryStructure.train.jsonl",
    "HookTheoryStructure.val.jsonl",
    "HookTheoryStructure.test.jsonl",
    "FIELDS_DECODE.txt",
)
LEGACY_SOURCES = (
    "src/data/preprocess_hooktheory.py",
    "src/data/canonicalize_hooktheory.py",
    "src/data/build_preprocess_song_timelines.py",
    "src/data/encode_teacher_features.py",
    "src/data/render_encoded_song_to_midi.py",
    "src/dataloader/theory_helpers.py",
    "src/dataloader/hooktheory_dataset.py",
    "src/dataloader/utils_graph.py",
    "tests/test_canonicalize_hooktheory.py",
    "docs/hooktheory_processed.txt",
    "docs/hooktheory_selected_field_types_documentation.txt",
    "docs/FIELDS_DECODE.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def bounded_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): bounded_value(value[key]) for key in sorted(value)[:12]}
    if isinstance(value, list):
        return [bounded_value(item) for item in value[:12]]
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    return value


def stable_value_key(value: Any) -> str:
    return json.dumps(bounded_value(value), ensure_ascii=False, sort_keys=True)


@dataclass
class FieldProfile:
    present: int = 0
    missing: int = 0
    types: Counter[str] = field(default_factory=Counter)
    values: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, *, clip_id: str, value: Any, present: bool = True) -> None:
        if not present:
            self.missing += 1
            return
        self.present += 1
        self.types[json_type(value)] += 1
        self.values[stable_value_key(value)] += 1
        if len(self.examples) < BOUNDED_EXAMPLES:
            self.examples.append({"clip_id": clip_id, "value": bounded_value(value)})

    def to_dict(self) -> dict[str, Any]:
        domains = sorted(self.values.items(), key=lambda item: (-item[1], item[0]))
        return {
            "present": self.present,
            "missing": self.missing,
            "runtime_types": dict(sorted(self.types.items())),
            "value_domain": [
                {"value": json.loads(value), "count": count}
                for value, count in domains[:BOUNDED_EXAMPLES]
            ],
            "distinct_bounded_values": len(self.values),
            "examples": self.examples,
        }


class IncrementalJSON:
    """Small incremental JSON value reader for very large top-level objects."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.handle = path.open("r", encoding="utf-8")
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def close(self) -> None:
        self.handle.close()

    def _fill(self) -> bool:
        if self.eof:
            return False
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return False

    def skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def peek(self) -> str:
        self.skip_space()
        if self.position >= len(self.buffer):
            return ""
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"expected {expected!r}, found {actual!r}")
        self.position += 1

    def value(self) -> Any:
        self.skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if not self._fill():
                    raise
            else:
                self.position = end
                return value


def iter_top_level_object(path: Path) -> Iterator[tuple[str, Any]]:
    """Yield a JSON object or the legacy object-fragment form entry by entry."""

    stream = IncrementalJSON(path)
    try:
        has_braces = stream.peek() == "{"
        if has_braces:
            stream.consume("{")
        while True:
            marker = stream.peek()
            if not marker or (has_braces and marker == "}"):
                if marker == "}":
                    stream.consume("}")
                break
            key = stream.value()
            if not isinstance(key, str):
                raise ValueError(f"top-level key is not a string in {path}")
            stream.consume(":")
            yield key, stream.value()
            marker = stream.peek()
            if marker == ",":
                stream.consume(",")
                continue
            if has_braces and marker == "}":
                continue
            if not marker:
                break
            raise ValueError(f"unexpected top-level delimiter {marker!r} in {path}")
    finally:
        stream.close()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield line_number, row


def normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return "val" if normalized == "valid" else normalized


def detect_structure(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() != ".json":
        return "text" if path.suffix.lower() in {".txt", ".md"} else "regular_file"
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096).lstrip()
    if prefix.startswith("{"):
        return "json_object"
    if prefix.startswith("["):
        return "json_array"
    if prefix.startswith('"'):
        return "json_object_fragment"
    return "json_unknown"


def relative_to_root(path: Path, root: Path, prefix: str) -> str:
    return f"{prefix}/{path.relative_to(root).as_posix()}"


def count_json_records(path: Path, structure: str) -> int | None:
    if structure in {"json_object", "json_object_fragment"}:
        return sum(1 for _ in iter_top_level_object(path))
    if structure == "jsonl":
        return sum(1 for _ in iter_jsonl(path))
    if structure == "json_array":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return len(value)
    return None


def inventory_file(path: Path, root: Path, prefix: str, *, count: bool) -> dict[str, Any]:
    structure = detect_structure(path)
    split = None
    lower_name = path.name.lower()
    for token, normalized in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        if token in lower_name:
            split = normalized
            break
    name = path.name.lower()
    if name.startswith("hooktheorystructure."):
        role, identifiers = "structural", ["audio_path", "ori_uid"]
    elif "key." in name:
        role, identifiers = "structural_key_label", ["audio_path", "ori_uid"]
    elif "segments" in name:
        role, identifiers = "split_manifest", ["top_level_key", "audio_tag"]
    elif "encoded" in path.as_posix() or "teacher_encoded" in name:
        role, identifiers = "encoded", ["top_level_key", "song_id", "meta.ori_uid"]
    elif "canonical" in path.as_posix():
        role, identifiers = "canonicalized", ["top_level_key", "song_id", "meta.ori_uid"]
    elif "processed" in path.as_posix() or "timeline" in name:
        role, identifiers = "processed", ["top_level_key", "song_id", "meta.ori_uid"]
    elif path == find_raw_source(root) if prefix == "data/HookTheory" else False:
        role, identifiers = "raw_legacy", ["top_level_key", "hash"]
    elif name == "hooktheory.json":
        role, identifiers = "selected_alternate_schema", ["top_level_key", "hooktheory.id"]
    else:
        role, identifiers = "documentation", []
    return {
        "path": relative_to_root(path, root, prefix),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "detected_structure": structure,
        "record_count": count_json_records(path, structure) if count else None,
        "split": split,
        "role": role,
        "identifier_fields": identifiers,
    }


def find_raw_source(hooktheory_root: Path) -> Path:
    candidates: list[Path] = []
    for entry in sorted(hooktheory_root.iterdir(), key=lambda item: item.name.lower()):
        if entry.name.lower() == "hooktheory_raw.json":
            if entry.is_file():
                candidates.append(entry)
            elif entry.is_dir():
                candidates.extend(sorted(entry.glob("*.json")))
    if not candidates:
        raise FileNotFoundError("Hooktheory_Raw.json file or directory was not found")
    preferred = [path for path in candidates if "merged" in path.name.lower()]
    return preferred[0] if preferred else candidates[0]


def classify_borrowed(value: Any) -> str:
    if value is None:
        return "borrowed_null"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "borrowed_empty_string"
        if stripped.startswith("[") and stripped.endswith("]"):
            return "borrowed_stringified_list"
        if stripped.lower().replace("-", "_").replace(" ", "_") in {
            "major", "minor", "dorian", "phrygian", "lydian", "mixolydian",
            "locrian", "harmonic_minor", "harmonicminor", "phrygian_dominant",
            "phrygiandominant",
        }:
            return "borrowed_mode_string"
        return "borrowed_unknown_string"
    if isinstance(value, list):
        return "borrowed_pitch_class_list"
    return "borrowed_unexpected_type"


def add_candidate(candidates: defaultdict[str, set[str]], tag: str, clip_id: str) -> None:
    candidates[tag].add(clip_id)


def inspect_event_fields(
    profiles: dict[str, FieldProfile],
    candidates: defaultdict[str, set[str]],
    clip_id: str,
    payload: dict[str, Any],
) -> None:
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    chords = payload.get("chords") if isinstance(payload.get("chords"), list) else []
    meters = payload.get("meters") if isinstance(payload.get("meters"), list) else []
    for event in notes:
        if not isinstance(event, dict):
            continue
        for name in ("beat", "duration", "sd", "octave", "isRest"):
            profiles[name].observe(clip_id=clip_id, value=event.get(name), present=name in event)
        if isinstance(event.get("beat"), float) or isinstance(event.get("duration"), float):
            add_candidate(candidates, "fractional_timing", clip_id)
        if event.get("beat") is None:
            add_candidate(candidates, "missing_beat", clip_id)
        if event.get("octave") is None:
            add_candidate(candidates, "missing_octave", clip_id)
        if isinstance(event.get("sd"), str) and event["sd"].startswith(("b", "#")):
            add_candidate(candidates, "accidental_scale_degree", clip_id)
        if event.get("isRest") is True:
            add_candidate(candidates, "melody_rest", clip_id)
    for event in chords:
        if not isinstance(event, dict):
            continue
        for name in (
            "beat", "duration", "root", "type", "inversion", "applied", "adds",
            "omits", "alterations", "suspensions", "borrowed", "alternate", "pedal",
        ):
            profiles[name].observe(clip_id=clip_id, value=event.get(name), present=name in event)
        root = event.get("root")
        if root == 0 and event.get("isRest") is True:
            add_candidate(candidates, "root_zero_rest", clip_id)
        if root == 0 and event.get("isRest") is not True:
            add_candidate(candidates, "root_zero_non_rest", clip_id)
        if root == 8:
            add_candidate(candidates, "root_eight_bvii", clip_id)
        if isinstance(root, int) and root not in range(0, 9):
            add_candidate(candidates, "out_of_domain_root", clip_id)
        if event.get("type") in {9, 11, 13}:
            add_candidate(candidates, "extended_chord_type", clip_id)
        if event.get("type") == 7:
            add_candidate(candidates, "seventh_chord_type", clip_id)
        if event.get("inversion") not in {None, 0}:
            add_candidate(candidates, "inversion", clip_id)
        if event.get("applied") not in {None, 0}:
            add_candidate(candidates, "applied_raw_deferred", clip_id)
        for name in ("adds", "omits", "alterations", "suspensions"):
            if event.get(name):
                add_candidate(candidates, f"nonempty_{name}", clip_id)
        add_candidate(candidates, classify_borrowed(event.get("borrowed")), clip_id)
    for event in meters:
        if not isinstance(event, dict):
            continue
        for name in ("beat", "beatUnit", "numBeats"):
            profiles[name].observe(clip_id=clip_id, value=event.get(name), present=name in event)


def audit_raw(raw_path: Path, candidate_limit: int) -> tuple[dict[str, Any], dict[str, str], dict[str, list[str]]]:
    profiles = {name: FieldProfile() for name in RAW_FIELDS}
    region_profiles = {name: FieldProfile() for name in ("keys", "tempos", "meters")}
    split_counts: Counter[str] = Counter()
    list_counts: Counter[str] = Counter()
    candidates: defaultdict[str, set[str]] = defaultdict(set)
    clip_splits: dict[str, str] = {}
    missing_json = 0
    unexpected_json = 0
    for top_key, record in iter_top_level_object(raw_path):
        if not isinstance(record, dict):
            unexpected_json += 1
            add_candidate(candidates, "unexpected_record_type", top_key)
            continue
        clip_id = str(record.get("hash", top_key))
        split = normalize_split(record.get("split")) or "unknown"
        split_counts[split] += 1
        clip_splits[clip_id] = split
        payload = record.get("json")
        if payload is None:
            missing_json += 1
            add_candidate(candidates, "missing_json_payload", clip_id)
            continue
        if not isinstance(payload, dict):
            unexpected_json += 1
            add_candidate(candidates, "unexpected_json_payload_type", clip_id)
            continue
        for plural in ("notes", "chords", "keys", "tempos", "meters"):
            value = payload.get(plural)
            if isinstance(value, list):
                list_counts[plural] += len(value)
            elif value is not None:
                add_candidate(candidates, f"unexpected_{plural}_type", clip_id)
            if plural in region_profiles:
                region_profiles[plural].observe(
                    clip_id=clip_id, value=value, present=plural in payload
                )
        keys = payload.get("keys") if isinstance(payload.get("keys"), list) else []
        scales = [item.get("scale") for item in keys if isinstance(item, dict)]
        if "major" in scales:
            add_candidate(candidates, "ordinary_major", clip_id)
        if any(scale not in {None, "major"} for scale in scales):
            add_candidate(candidates, "minor_or_modal", clip_id)
        if len(keys) > 1:
            add_candidate(candidates, "multiple_key_regions", clip_id)
        if isinstance(payload.get("tempos"), list) and len(payload["tempos"]) > 1:
            add_candidate(candidates, "multiple_tempo_regions", clip_id)
        if isinstance(payload.get("meters"), list) and len(payload["meters"]) > 1:
            add_candidate(candidates, "multiple_meter_regions", clip_id)
        inspect_event_fields(profiles, candidates, clip_id, payload)
    bounded_candidates = {
        tag: sorted(ids)[:candidate_limit] for tag, ids in sorted(candidates.items())
    }
    return (
        {
            "record_counts_by_split": dict(sorted(split_counts.items())),
            "event_counts": dict(sorted(list_counts.items())),
            "records_missing_json": missing_json,
            "records_with_unexpected_json_type": unexpected_json,
            "field_profiles": {name: profiles[name].to_dict() for name in RAW_FIELDS},
            "region_profiles": {
                name: region_profiles[name].to_dict() for name in region_profiles
            },
        },
        clip_splits,
        bounded_candidates,
    )


def load_object_ids(path: Path) -> tuple[set[str], Counter[str]]:
    ids: set[str] = set()
    splits: Counter[str] = Counter()
    for key, record in iter_top_level_object(path):
        ids.add(key)
        if isinstance(record, dict):
            meta = record.get("meta")
            split = normalize_split(meta.get("split")) if isinstance(meta, dict) else None
            splits[split or "unknown"] += 1
    return ids, splits


def audit_structures(hooktheory_root: Path, symbolic_splits: dict[str, str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    labels = FieldProfile()
    timestamps = {name: FieldProfile() for name in ("segment_start", "segment_end", "duration")}
    for split in rows_by_split:
        path = hooktheory_root / f"HookTheoryStructure.{split}.jsonl"
        for line_number, row in iter_jsonl(path):
            audio_path = row.get("audio_path")
            clip_id = Path(audio_path).stem if isinstance(audio_path, str) and audio_path else None
            compact = {
                "line_number": line_number,
                "clip_id": clip_id,
                "ori_uid": row.get("ori_uid"),
                "label": bounded_value(row.get("label")),
                "duration": row.get("duration"),
                "segment_start": row.get("segment_start"),
                "segment_end": row.get("segment_end"),
            }
            rows_by_split[split].append(compact)
            locator = clip_id or f"line:{line_number}"
            labels.observe(clip_id=locator, value=row.get("label"), present="label" in row)
            for name, profile in timestamps.items():
                profile.observe(clip_id=locator, value=row.get(name), present=name in row)

    joins: dict[str, Any] = {}
    uid_splits: defaultdict[str, set[str]] = defaultdict(set)
    uid_clips: defaultdict[str, set[str]] = defaultdict(set)
    missing_ori_uid = 0
    matched_rows: list[str] = []
    unmatched_rows: list[str] = []
    duplicate_matches: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        symbolic_ids = {clip for clip, value in symbolic_splits.items() if value == split}
        structure_ids = {row["clip_id"] for row in rows if row["clip_id"]}
        per_clip = Counter(row["clip_id"] for row in rows if row["clip_id"])
        duplicates = {clip: count for clip, count in per_clip.items() if count > 1}
        duplicate_matches.update({f"{split}:{clip}": count for clip, count in duplicates.items()})
        intersection = symbolic_ids & structure_ids
        joins[split] = {
            "symbolic_clips": len(symbolic_ids),
            "structure_rows": len(rows),
            "structure_clip_ids": len(structure_ids),
            "matched_clip_ids": len(intersection),
            "unmatched_symbolic_clips": len(symbolic_ids - structure_ids),
            "unmatched_structure_clip_ids": len(structure_ids - symbolic_ids),
            "duplicate_structure_clip_ids": len(duplicates),
        }
        matched_rows.extend(f"{split}:{clip}" for clip in sorted(intersection))
        unmatched_rows.extend(f"{split}:{clip}" for clip in sorted(structure_ids - symbolic_ids))
        for row in rows:
            uid = row["ori_uid"]
            if uid is None or uid == "":
                missing_ori_uid += 1
                continue
            uid_text = str(uid)
            uid_splits[uid_text].add(split)
            if row["clip_id"]:
                uid_clips[uid_text].add(row["clip_id"])

    leakage = {
        uid: sorted(splits) for uid, splits in sorted(uid_splits.items()) if len(splits) > 1
    }
    shared = {
        uid: sorted(clips) for uid, clips in sorted(uid_clips.items()) if len(clips) > 1
    }
    candidates = {
        "matched_structure": matched_rows[:BOUNDED_EXAMPLES],
        "unmatched_structure": unmatched_rows[:BOUNDED_EXAMPLES],
        "multiple_clips_same_ori_uid": [
            {"ori_uid": uid, "clip_ids": clips[:BOUNDED_EXAMPLES]}
            for uid, clips in list(shared.items())[:BOUNDED_EXAMPLES]
        ],
    }
    return {
        "join_by_split": joins,
        "missing_ori_uid_rows": missing_ori_uid,
        "duplicate_structure_matches": dict(sorted(duplicate_matches.items())),
        "ori_uid_split_leakage": leakage,
        "ori_uid_groups_with_multiple_clips": len(shared),
        "structure_label_profile": labels.to_dict(),
        "section_timestamp_profiles": {
            name: profile.to_dict() for name, profile in timestamps.items()
        },
    }, candidates


def legacy_inventory(legacy_root: Path) -> list[dict[str, Any]]:
    inventory = []
    for relative in LEGACY_SOURCES:
        path = legacy_root / relative
        discovered = path
        if not path.exists() and relative.startswith("docs/"):
            discovered = legacy_root / "docs/music_critic_v1" / Path(relative).name
        inventory.append({
            "requested_path": relative,
            "discovered_path": discovered.relative_to(legacy_root).as_posix() if discovered.exists() else None,
            "exists": discovered.is_file(),
            "sha256": sha256_file(discovered) if discovered.is_file() else None,
        })
    return inventory


def discover_inventory(hooktheory_root: Path, htcanon_root: Path, raw_path: Path) -> list[dict[str, Any]]:
    paths: list[tuple[Path, Path, str, bool]] = []
    for name in PRIMARY_HOOKTHEORY_NAMES:
        path = hooktheory_root / name
        if path.is_file():
            paths.append((path, hooktheory_root, "data/HookTheory", True))
    if raw_path not in [item[0] for item in paths]:
        paths.append((raw_path, hooktheory_root, "data/HookTheory", True))
    processed_root = htcanon_root / "HK_processed"
    expected = (
        processed_root / "hooktheory_processed.json",
        processed_root / "hooktheory_processed_structured_only.json",
        processed_root / "original_songs_timeline.json",
        processed_root / "canonical_full/hooktheory_canonical.json",
        processed_root / "canonical_structured_only/hooktheory_canonical.json",
        processed_root / "encoded_full/teacher_encoded.json",
        processed_root / "encoded_structured_only/teacher_encoded.json",
        htcanon_root / "encoded_full/teacher_encoded.json",
    )
    for path in expected:
        if path.is_file():
            paths.append((path, htcanon_root, "data/HTCanon", True))
    return [inventory_file(path, root, prefix, count=count) for path, root, prefix, count in paths]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    hooktheory_root = args.hooktheory_root.resolve()
    htcanon_root = args.htcanon_root.resolve()
    legacy_root = args.legacy_root.resolve()
    for label, root in (
        ("HookTheory", hooktheory_root), ("HTCanon", htcanon_root), ("legacy", legacy_root)
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{label} root is not a directory: {root}")
    raw_path = find_raw_source(hooktheory_root)
    inventory = discover_inventory(hooktheory_root, htcanon_root, raw_path)
    raw_audit, raw_splits, raw_candidates = audit_raw(raw_path, args.candidate_limit)
    processed_path = htcanon_root / "HK_processed/hooktheory_processed.json"
    canonical_path = htcanon_root / "HK_processed/canonical_full/hooktheory_canonical.json"
    processed_ids, processed_splits = load_object_ids(processed_path)
    canonical_ids, canonical_splits = load_object_ids(canonical_path)
    structure_audit, structure_candidates = audit_structures(hooktheory_root, raw_splits)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "inputs": {
            "hooktheory_root": "data/HookTheory",
            "htcanon_root": "data/HTCanon",
            "legacy_commit_expected": "2d8281f31cc9ad9c8fecaf332da0c61e0e949415",
        },
        "source_inventory": inventory,
        "legacy_source_inventory": legacy_inventory(legacy_root),
        "raw_audit": raw_audit,
        "processed_outputs": {
            "processed_record_count": len(processed_ids),
            "processed_counts_by_split": dict(sorted(processed_splits.items())),
            "canonical_record_count": len(canonical_ids),
            "canonical_counts_by_split": dict(sorted(canonical_splits.items())),
            "raw_to_processed": {
                "raw_ids": len(raw_splits),
                "matched": len(set(raw_splits) & processed_ids),
                "raw_only": len(set(raw_splits) - processed_ids),
                "processed_only": len(processed_ids - set(raw_splits)),
            },
            "processed_to_canonical": {
                "matched": len(processed_ids & canonical_ids),
                "processed_only": len(processed_ids - canonical_ids),
                "canonical_only": len(canonical_ids - processed_ids),
            },
        },
        "structure_audit": structure_audit,
        "golden_candidate_clip_ids": raw_candidates,
        "structure_candidates": structure_candidates,
        "confirmed_contract": {
            "symbolic_onset": "Fraction(raw_decimal_string) - 1",
            "derived_pitch_formula": "72 + 12 * octave + tonic_pc + sd_chromatic_offset",
            "derived_pitch_provenance": "hooktheory_sd_octave_to_midi_v1",
            "root_mapping": {"1..7": "0..6", "8": "bVII", "0": "rest_or_empty_never_tonic"},
            "structure_join": "normalized_split + audio_path_stem",
            "section_alignment_status": "unresolved_audio_seconds",
            "applied_harmony": "raw_evidence_only_deferred",
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hooktheory-root", type=Path, required=True)
    parser.add_argument("--htcanon-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=100)
    args = parser.parse_args()
    if args.candidate_limit < 1:
        parser.error("--candidate-limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = build_report(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
