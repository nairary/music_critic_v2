"""Admit byte-bound authored pairs without regenerating intervention recipes.

Only a separate, source-bound direction review enables weak ranking. Neither
technical compatibility nor edit count supplies a musical preference label.
All author/reviewer information stays on the target side of the interface.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from music_critic.data import CanonicalPiece, RationalTime, dumps_piece, loads_piece
from music_critic.experiments.music_interventions import Intervention, NotePatch, apply_interventions
from .contracts import ASPECTS
from .targets import PairTargets, VerifiedPatch, verify_actual_diff

MANIFEST_SCHEMA = "AuthoredPairManifest@1.0.0"
TARGET_VERSION = "AuthoredGlobalPairTargets@1.0.0"
REVIEW_SCHEMA = "AuthoredDirectionReview@1.0.0"
GENRES = frozenset(("classical", "jazz", "rock", "hiphop", "electronic"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AuthoredPair:
    pair_id: str
    work_id: str
    component_id: str
    split: str
    original: CanonicalPiece
    variant: CanonicalPiece
    targets: PairTargets
    edit_count: int
    genre: str
    operator: str
    source_id: str


@dataclass(frozen=True)
class AuthoredDataset:
    pairs: tuple[AuthoredPair, ...]
    fingerprint: str
    source_summary: Mapping[str, object]
    split_assignments: Mapping[str, str]
    omissions: tuple[Mapping[str, object], ...]


def _known(value, name):
    if (not isinstance(value, str) or value != value.strip() or not value
            or value.lower() in {"unknown", "none", "null"}):
        raise ValueError("authored known " + name + " required")
    return value


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("authored duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("authored nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _binding(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size_bytes"}:
        raise ValueError("authored file binding fields invalid")
    path = binding["path"]
    if (not isinstance(path, str) or not path or "\\" in path
            or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts):
        raise ValueError("authored unsafe relative path")
    if not isinstance(binding["sha256"], str) or not _SHA256.fullmatch(binding["sha256"]):
        raise ValueError("authored file SHA256 required")
    if type(binding["size_bytes"]) is not int or binding["size_bytes"] <= 0:
        raise ValueError("authored positive file size required")


def _read(root, binding):
    _binding(binding)
    path = (root / binding["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("authored symlink escapes payload root")
    raw = path.read_bytes()
    if len(raw) != binding["size_bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("authored file integrity mismatch: " + binding["path"])
    return raw


def _preflight(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("authored manifest schema invalid")
    if manifest.get("target_version") != TARGET_VERSION:
        raise ValueError("authored target version mismatch")
    for key in ("source_id", "split_id"):
        _known(manifest.get(key), key)
    rows = manifest.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("authored nonempty pair manifest required")
    ids, split_keys = set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("authored pair descriptor required")
        # Check every row before opening any referenced payload, including rows
        # that callers would later filter out. Final campaign reserve is closed.
        if row.get("split") not in {"train", "head_dev"}:
            raise ValueError("authored final_holdout or unknown split forbidden")
        if row.get("source_scope") not in {"train", "external_development"}:
            raise ValueError("authored corpus heldout or unknown source scope forbidden")
        for key in ("pair_id", "work_id", "component_id", "variant_id"):
            _known(row.get(key), key)
        if row["pair_id"] in ids:
            raise ValueError("authored duplicate pair ID")
        ids.add(row["pair_id"])
        if row.get("genre") not in GENRES:
            raise ValueError("authored unknown genre")
        if type(row.get("edit_count")) is not int or not 1 <= row["edit_count"] <= 4:
            raise ValueError("authored edit count must be1..4")
        for key in ("original", "variant", "plan"):
            _binding(row.get(key))
        if row.get("direction_review") is not None:
            _binding(row["direction_review"])
        for identity in (("work", row["work_id"]), ("component", row["component_id"]),
                         ("original", row["original"]["sha256"]),
                         ("variant", row["variant"]["sha256"])):
            if identity in split_keys and split_keys[identity] != row["split"]:
                raise ValueError("authored cross-split identity overlap")
            split_keys[identity] = row["split"]
    return rows


def _patch(raw):
    if not isinstance(raw, dict) or "note_id" not in raw:
        raise ValueError("authored note patch required")
    if set(raw) - {"note_id", "pitch", "onset_qn", "duration_qn", "delete"}:
        raise ValueError("authored unexpected patch field")
    values = dict(raw)
    for key in ("onset_qn", "duration_qn"):
        if key in values and values[key] is not None:
            value = values[key]
            if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
                raise ValueError("authored timing requires integer rational pair")
            values[key] = RationalTime(*value)
    return NotePatch(**values)


def _selection(plan, row):
    if not isinstance(plan, dict) or not isinstance(plan.get("variants"), list):
        raise ValueError("authored plan variants required")
    variants = {}
    for item in plan["variants"]:
        identifier = _known(item.get("id"), "authored variant ID")
        if identifier in variants:
            raise ValueError("authored duplicate variant ID")
        variants[identifier] = item
    combos = {}
    for item in plan.get("combinations", []):
        identifier = _known(item.get("id"), "combination ID")
        if identifier in combos or identifier in variants:
            raise ValueError("authored duplicate combination ID")
        combos[identifier] = item
    selected = row["variant_id"]
    if selected in variants:
        parents = (selected,)
    elif selected in combos:
        raw = combos[selected].get("parents")
        if not isinstance(raw, list) or any(not isinstance(p, str) for p in raw):
            raise ValueError("authored combination parents required")
        parents = tuple(raw)
    else:
        raise ValueError("authored selected variant absent")
    if len(parents) != row["edit_count"] or len(set(parents)) != len(parents) or any(p not in variants for p in parents):
        raise ValueError("authored parent/edit count mismatch")
    aspects = {variants[p].get("aspect") for p in parents}
    if len(aspects) != 1 or not aspects <= set(ASPECTS):
        raise ValueError("authored same known aspect required")
    interventions = tuple(Intervention(p, tuple(_patch(patch) for patch in variants[p]["patches"]),
                                       tuple(variants[p].get("context_note_ids", ()))) for p in parents)
    return parents, aspects.pop(), interventions


def _merged_patches(interventions):
    merged = {}
    for intervention in interventions:
        for patch in intervention.patches:
            values = merged.setdefault(patch.note_id, {})
            for key in patch.written_fields:
                if key in values:
                    raise ValueError("authored overlapping patch write")
                values[key] = getattr(patch, key)
    return tuple(NotePatch(note_id, **values) for note_id, values in sorted(merged.items()))


def _direction(review, row, parents, aspect):
    if review is None:
        return 0, "missing_direction_review"
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("authored review schema mismatch")
    expected = {"work_id": row["work_id"], "aspect": aspect,
                "original_sha256": row["original"]["sha256"], "variant_sha256": row["variant"]["sha256"],
                "plan_sha256": row["plan"]["sha256"],
                "scope": "single" if len(parents) == 1 else "joint_combination"}
    if any(review.get(key) != value for key, value in expected.items()) or review.get("parents") != list(parents):
        raise ValueError("authored direction review binding/scope mismatch")
    if review.get("evidence_kind") != "weak_author_intent" or review.get("human_preference") != "unknown":
        raise ValueError("authored weak intent must stay separate from human preference")
    for key in ("reviewer", "rationale"):
        _known(review.get(key), "review " + key)
    if not isinstance(review.get("uncertainty"), str) or not review["uncertainty"].strip():
        raise ValueError("authored review uncertainty required")
    direction = review.get("direction")
    if type(direction) is not int or direction not in {-1, 0, 1}:
        raise ValueError("authored review direction invalid")
    if review.get("status") == "accepted" and direction in {-1, 1}:
        return direction, None
    if review.get("status") == "unknown" and direction == 0:
        return 0, "unknown_author_direction"
    raise ValueError("authored direction availability/status mismatch")


def _observation(piece):
    # Ignore IDs and annotation/provenance when guarding against aliased source
    # copies. This is identity grouping, never a learned predictor feature.
    rational = lambda q: (q.num, q.den)
    groups = defaultdict(list)
    for n in piece.notes:
        groups[n.track_id].append((n.pitch, rational(n.onset_qn), rational(n.duration_qn),
                                  n.velocity, n.channel, n.program, n.is_percussion, n.is_grace))
    tracks = [(t.program, t.channel, t.is_percussion, sorted(groups[t.track_id], key=_digest)) for t in piece.tracks]
    return _digest({"tracks": sorted(tracks, key=_digest), "duration": rational(piece.duration_qn),
                    "bars": [(rational(b.start_qn), rational(b.duration_qn), rational(b.metric_offset_qn), b.is_pickup, b.is_incomplete) for b in piece.bars],
                    "beats": [(rational(b.start_qn), rational(b.duration_qn), rational(b.position_in_bar_qn), b.is_downbeat, b.strength) for b in piece.beats],
                    "meter": [(rational(m.onset_qn), m.numerator, m.denominator) for m in piece.meter_events],
                    "tempo": [(rational(t.onset_qn), t.microseconds_per_quarter) for t in piece.tempo_events],
                    "key": [(rational(k.onset_qn), k.fifths, k.mode) for k in piece.key_signature_events]})


def load_authored_dataset(manifest_path: str | Path, manifest_sha256: str) -> AuthoredDataset:
    """Load bound train/dev pairs. Final reserve cannot enter this loader.

    Payload paths are relative to the manifest directory and cannot escape it.
    All rows and declared split memberships are checked before payload access.
    Unknown direction is retained as an unavailable target and omission reason.
    """
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256) or sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("authored manifest integrity mismatch")
    manifest = _json(raw)
    rows = _preflight(manifest)
    pairs, omissions, assignments, split_keys = [], [], {}, {}
    for row in rows:
        original = loads_piece(_read(path.parent, row["original"]).decode())
        variant = loads_piece(_read(path.parent, row["variant"]).decode())
        if original.split in {"validation", "test"} or variant.split in {"validation", "test"}:
            raise ValueError("authored embedded corpus heldout split forbidden")
        if original.piece_id == variant.piece_id:
            raise ValueError("authored variant must have a distinct immutable identity")
        plan = _json(_read(path.parent, row["plan"]))
        parents, aspect, interventions = _selection(plan, row)
        result = apply_interventions(original, interventions)
        if dumps_piece(result.piece) != dumps_piece(variant):
            raise ValueError("authored materialized variant differs from exact replay")
        proof = VerifiedPatch(original.piece_id, variant.piece_id, row["work_id"],
                              _merged_patches(interventions), (aspect,), "authored_note_patch_replay")
        diff = verify_actual_diff(original, variant, proof)
        if not diff.changed_note_ids and not diff.deleted_note_ids:
            raise ValueError("authored pair requires a nonzero actual diff")
        review = _json(_read(path.parent, row["direction_review"])) if row.get("direction_review") else None
        direction, unavailable_reason = _direction(review, row, parents, aspect)
        mask = tuple(name == aspect and direction != 0 for name in ASPECTS)
        provenance = {"target_version": TARGET_VERSION, "source_id": manifest["source_id"],
                      "split_id": manifest["split_id"], "source_scope": row["source_scope"],
                      "manifest_sha256": manifest_sha256, "plan": dict(row["plan"]),
                      "original": dict(row["original"]), "variant": dict(row["variant"]),
                      "parents": parents, "aspect": aspect, "edit_count": len(parents),
                      "direction_review": review, "direction_review_binding": row.get("direction_review"),
                      "human_preference": "unknown", "quality_preference": "unknown",
                      "label_kind": "weak_author_intent", "mechanical_compatibility": result.compatibility.status,
                      "changed_note_ids": diff.changed_note_ids, "deleted_note_ids": diff.deleted_note_ids,
                      "exhaustive_diff": True}
        targets = PairTargets(row["work_id"], (original.piece_id, variant.piece_id), mask,
                              tuple(direction if available else 0 for available in mask), ({}, {}), provenance)
        for piece in (original, variant):
            for identity in (("piece", piece.piece_id), ("source_group", piece.source_group_id), ("raw", _observation(piece))):
                if identity in split_keys and split_keys[identity] != row["split"]:
                    raise ValueError("authored raw/lineage cross-split overlap")
                split_keys[identity] = row["split"]
        if unavailable_reason:
            omissions.append({"pair_id": row["pair_id"], "work_id": row["work_id"], "aspect": aspect,
                              "edit_count": len(parents), "reason": unavailable_reason, "pair_retained": True})
        original = replace(original, annotations=(), targets=(), quality_flags=())
        variant = replace(variant, annotations=(), targets=(), quality_flags=())
        pairs.append(AuthoredPair(row["pair_id"], row["work_id"], row["component_id"], row["split"],
                                 original, variant, targets, len(parents), row["genre"], row["variant_id"], manifest["source_id"]))
        assignments[row["work_id"]] = row["split"]
    summary = {"schema": MANIFEST_SCHEMA, "source_id": manifest["source_id"], "split_id": manifest["split_id"],
               "target_version": TARGET_VERSION, "manifest_sha256": manifest_sha256,
               "pairs": len(pairs), "works": len(assignments), "splits": dict(Counter(p.split for p in pairs)),
               "available_aspect_targets": {aspect: sum(p.targets.pair_mask[i] for p in pairs) for i, aspect in enumerate(ASPECTS)},
               "claim": "Reviewed weak authored aspect directions; human preference unknown; no local targets."}
    return AuthoredDataset(tuple(pairs), _digest({"manifest_sha256": manifest_sha256, "target_version": TARGET_VERSION}),
                           summary, assignments, tuple(omissions))
