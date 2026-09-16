"""Bounded context diagnostics; additive to the frozen corrected evaluator.

The local control is a TRAIN-only standardized nearest-centroid classifier.
It has deliberately limited capacity and is not a compute-matched C2 ablation.
Targets are read after raw candidates/features (and validation C2 predictions).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Callable, Sequence

from music_critic.data import CanonicalPiece, RationalTime


def fingerprint(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require_sha(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("context_diagnostics.sha256_required")


def guard_split(split: str) -> None:
    if split not in {"train", "validation"}:
        raise ValueError("context_diagnostics.TRAINVAL_only_TEST_closed")


@dataclass(frozen=True)
class FileBinding:
    path: str
    sha256: str
    size_bytes: int

    def read(self, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
        require_sha(self.sha256)
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= max_bytes:
            raise ValueError("context_diagnostics.file_budget")
        with Path(self.path).open("rb") as handle:
            raw = handle.read(self.size_bytes + 1)
        if len(raw) != self.size_bytes or sha256(raw).hexdigest() != self.sha256:
            raise ValueError("context_diagnostics.file_binding_mismatch")
        return raw


@dataclass(frozen=True)
class Member:
    piece_id: str
    component_id: str
    split: str


def validate_membership(members: Sequence[Member]) -> tuple[Member, ...]:
    """Validate full declared membership before any raw/checkpoint callback.

    Its upstream official split bytes must additionally be bound by the caller;
    this function does not discover duplicate works from musical content.
    """
    seen: set[str] = set()
    components: dict[str, str] = {}
    for member in members:
        guard_split(member.split)
        if not member.piece_id or not member.component_id or member.piece_id in seen:
            raise ValueError("context_diagnostics.duplicate_or_missing_piece_identity")
        seen.add(member.piece_id)
        if components.setdefault(member.component_id, member.split) != member.split:
            raise ValueError("context_diagnostics.component_crosses_split")
    if not members:
        raise ValueError("context_diagnostics.empty_membership")
    return tuple(sorted(members, key=lambda member: member.piece_id))


@dataclass(frozen=True)
class ModelBinding:
    checkpoint_sha256: str
    model_contract_sha256: str
    model_state_sha256: str
    class_sha256: str
    feature_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            require_sha(value)


FEATURE_CONTRACT = {
    "schema": "ContextLocalFeatures@1.0.0",
    "window": "[raw_candidate_start,min(start+1_qn,piece_end)); no gold boundaries",
    "chroma": "12 duration-weighted bins normalized by total sounding duration",
    "interval": "7 undirected interval-class bins weighted by simultaneous overlap",
    "rhythm": ["sounding_note_count", "attacks_in_window", "held_at_left_edge",
               "sum_overlap_qn_per_window_qn", "mean_clipped_duration_qn"],
    "timing": "exact rational comparisons and accumulation; float only at feature output",
    "excluded": ["labels", "gold_boundaries", "key_signature", "staff", "voice",
                 "track_id", "velocity", "program", "channel", "absolute_position"],
}
LOCAL_FEATURE_SHA256 = fingerprint(FEATURE_CONTRACT)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    start: RationalTime
    end: RationalTime


def raw_candidates(piece: CanonicalPiece, level: str) -> tuple[Candidate, ...]:
    guard_split(piece.split)
    if level == "beat":
        starts = [(beat.beat_id, beat.start_qn) for beat in piece.beats]
    elif level == "note":
        starts = [(note.note_id, note.onset_qn) for note in piece.notes]
    elif level == "onset":
        starts = [(f"onset:{time.num}_{time.den}", time)
                  for time in sorted({note.onset_qn for note in piece.notes})]
    else:
        raise ValueError("context_diagnostics.raw_level_required")
    if len({row[0] for row in starts}) != len(starts):
        raise ValueError("context_diagnostics.duplicate_raw_candidate")
    return tuple(Candidate(identity, start, min(start + RationalTime(1), piece.duration_qn))
                 for identity, start in sorted(starts) if start < piece.duration_qn)


def local_features(piece: CanonicalPiece, candidate: Candidate) -> tuple[float, ...]:
    """Only simultaneous raw evidence in a fixed exact one-quarter-note window."""
    if not isinstance(candidate.start, RationalTime) or not isinstance(candidate.end, RationalTime):
        raise ValueError("context_diagnostics.exact_timing_required")
    if not RationalTime(0) <= candidate.start < candidate.end <= piece.duration_qn:
        raise ValueError("context_diagnostics.invalid_raw_window")
    notes = []
    for note in piece.notes:
        if note.is_percussion or note.is_grace:
            continue
        left = max(candidate.start, note.onset_qn)
        right = min(candidate.end, note.onset_qn + note.duration_qn)
        if left < right:
            notes.append((note.pitch, note.onset_qn, left, right))
    notes.sort()
    if len(notes) > 256:
        raise ValueError("context_diagnostics.local_polyphony_budget")
    chroma = [Fraction(0) for _ in range(12)]
    intervals = [Fraction(0) for _ in range(7)]
    for pitch, _, left, right in notes:
        chroma[pitch % 12] += (right - left).to_fraction()
    for index, (pitch, _, left, right) in enumerate(notes):
        for other, _, other_left, other_right in notes[index + 1:]:
            overlap = min(right, other_right) - max(left, other_left)
            if overlap > RationalTime(0):
                interval = abs(pitch - other) % 12
                intervals[min(interval, 12 - interval)] += overlap.to_fraction()
    total = sum(chroma)
    interval_total = sum(intervals)
    length = (candidate.end - candidate.start).to_fraction()
    rhythm = [len(notes), sum(candidate.start <= row[1] < candidate.end for row in notes),
              sum(row[1] < candidate.start for row in notes), total / length,
              total / len(notes) if notes else Fraction(0)]
    return tuple(float(value / total) if total else 0.0 for value in chroma) + tuple(
        float(value / interval_total) if interval_total else 0.0 for value in intervals
    ) + tuple(float(value) for value in rhythm)


@dataclass(frozen=True)
class TargetRow:
    observation_id: str
    candidate_id: str
    value: int | None
    available: bool
    reason: str | None
    provenance: str
    confidence: float | None = None

    def validate(self, class_count: int) -> None:
        if not self.observation_id or not self.candidate_id or not self.provenance:
            raise ValueError("context_diagnostics.target_provenance_required")
        if type(self.available) is not bool:
            raise ValueError("context_diagnostics.bool_mask_required")
        if self.available:
            if type(self.value) is not int or not 0 <= self.value < class_count or self.reason is not None:
                raise ValueError("context_diagnostics.invalid_available_target")
        elif self.value is not None or not self.reason:
            raise ValueError("context_diagnostics.masked_value_must_be_null")
        if self.confidence is not None and (
            type(self.confidence) not in {int, float} or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("context_diagnostics.invalid_confidence")


@dataclass(frozen=True)
class LabeledFeature:
    member: Member
    observation_id: str
    features: tuple[float, ...]
    target: TargetRow


@dataclass(frozen=True)
class LocalControl:
    classes: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    centroids: tuple[tuple[float, ...] | None, ...]
    train_support: tuple[int, ...]
    training_fingerprint: str

    def predict(self, features: tuple[float, ...]) -> int:
        if len(features) != len(self.means) or not all(math.isfinite(x) for x in features):
            raise ValueError("context_diagnostics.invalid_feature_vector")
        point = [(x - mean) / scale for x, mean, scale in zip(features, self.means, self.scales)]
        distances = [sum((x - center) ** 2 for x, center in zip(point, centroid))
                     if centroid is not None else math.inf for centroid in self.centroids]
        return min(range(len(distances)), key=lambda index: (distances[index], index))


def fit_local_control(rows: Sequence[LabeledFeature], classes: Sequence[str]) -> LocalControl:
    if len(classes) < 2 or len(set(classes)) != len(classes) or not all(isinstance(x, str) and x for x in classes):
        raise ValueError("context_diagnostics.ordered_classes_required")
    ordered = sorted(rows, key=lambda row: (row.member.piece_id, row.observation_id))
    if len({(row.member.piece_id, row.observation_id) for row in ordered}) != len(ordered):
        raise ValueError("context_diagnostics.duplicate_training_observation")
    for row in ordered:
        if row.member.split != "train":
            raise ValueError("context_diagnostics.fit_TRAIN_only")
        row.target.validate(len(classes))
    eligible = [row for row in ordered if row.target.available]
    if not eligible:
        raise ValueError("context_diagnostics.no_available_TRAIN_targets")
    width = len(eligible[0].features)
    if not width or any(len(row.features) != width or not all(math.isfinite(x) for x in row.features) for row in eligible):
        raise ValueError("context_diagnostics.invalid_feature_vector")
    means = tuple(math.fsum(row.features[i] for row in eligible) / len(eligible) for i in range(width))
    scales = tuple(math.sqrt(math.fsum((row.features[i] - means[i]) ** 2 for row in eligible) / len(eligible)) or 1.0
                   for i in range(width))
    centroids = []
    support = []
    for class_id in range(len(classes)):
        group = [row for row in eligible if row.target.value == class_id]
        support.append(len(group))
        centroids.append(tuple(math.fsum((row.features[i] - means[i]) / scales[i] for row in group) / len(group)
                               for i in range(width)) if group else None)
    return LocalControl(tuple(classes), means, scales, tuple(centroids), tuple(support),
                        fingerprint([asdict(row) for row in eligible]))


def classification_metrics(pairs: Sequence[tuple[int, int]], classes: Sequence[str]) -> dict:
    count = len(classes)
    matrix = [[0] * count for _ in range(count)]
    for gold, prediction in pairs:
        if any(type(value) is not int or not 0 <= value < count for value in (gold, prediction)):
            raise ValueError("context_diagnostics.class_out_of_registry")
        matrix[gold][prediction] += 1
    per_class = []
    for index, label in enumerate(classes):
        support, predicted, tp = sum(matrix[index]), sum(row[index] for row in matrix), matrix[index][index]
        per_class.append({"id": index, "label": label, "support": support, "predicted": predicted, "tp": tp,
                          "fp": predicted - tp, "fn": support - tp,
                          "precision": tp / predicted if predicted else None,
                          "recall": tp / support if support else None,
                          "f1": 2 * tp / (support + predicted) if support + predicted else None})
    included = [row["f1"] for row in per_class if row["support"]]
    return {"confusion_gold_rows_prediction_columns": matrix, "classes": per_class,
            "eligible_observations": len(pairs),
            "accuracy": sum(matrix[i][i] for i in range(count)) / len(pairs) if pairs else None,
            "observed_class_macro_f1": math.fsum(included) / len(included) if included else None,
            "undefined": "null when denominator zero; absent gold classes excluded from macro"}


def run_context_diagnostics(*, members: Sequence[Member], membership_sha256: str,
                            classes: Sequence[str], level: str, binding: ModelBinding,
                            raw_reader: Callable[[Member], CanonicalPiece],
                            target_reader: Callable[[Member], Sequence[TargetRow]],
                            reference_predictor: Callable[[CanonicalPiece, tuple[Candidate, ...]], dict[str, int]],
                            max_pieces: int = 160, max_notes: int = 5000,
                            max_candidates: int = 5000) -> dict:
    """The caller verifies source bytes and model bindings before this adapter.

    All members are explicit; no corpus discovery, checkpoint selection or TEST
    path. Separate source events sharing a raw candidate remain observations.
    """
    require_sha(membership_sha256)
    ordered = validate_membership(members)
    if not 0 < max_pieces <= 160 or not 0 < max_notes <= 5000 or not 0 < max_candidates <= 5000:
        raise ValueError("context_diagnostics.fixed_budget_exceeded")
    if len(ordered) > max_pieces:
        raise ValueError("context_diagnostics.piece_budget")
    if fingerprint(list(classes)) != binding.class_sha256:
        raise ValueError("context_diagnostics.class_order_binding_mismatch")
    train = []
    validation = []
    coverage = Counter()
    reference_calls = 0
    for member in sorted(ordered, key=lambda m: (m.split != "train", m.piece_id)):
        piece = raw_reader(member)
        if (piece.piece_id, piece.source_group_id, piece.split) != (member.piece_id, member.component_id, member.split):
            raise ValueError("context_diagnostics.raw_membership_mismatch")
        if len(piece.notes) > max_notes:
            raise ValueError("context_diagnostics.note_budget")
        piece = replace(piece, annotations=(), targets=())
        candidates = raw_candidates(piece, level)
        if len(candidates) > max_candidates:
            raise ValueError("context_diagnostics.candidate_budget")
        features = {candidate.candidate_id: local_features(piece, candidate) for candidate in candidates}
        predictions = reference_predictor(piece, candidates) if member.split == "validation" else None
        reference_calls += predictions is not None
        if predictions is not None and (set(predictions) != set(features) or any(
            type(value) is not int or not 0 <= value < len(classes) for value in predictions.values()
        )):
            raise ValueError("context_diagnostics.reference_candidate_mismatch")
        targets = tuple(target_reader(member))
        if len(targets) > 20000 or len({row.observation_id for row in targets}) != len(targets):
            raise ValueError("context_diagnostics.target_budget_or_duplicate")
        for target in targets:
            target.validate(len(classes))
            coverage[f"{member.split}_source_observations"] += 1
            if not target.available:
                coverage[f"{member.split}_masked:{target.reason}"] += 1
            elif target.candidate_id not in features:
                coverage[f"{member.split}_unaligned"] += 1
            else:
                row = LabeledFeature(member, target.observation_id, features[target.candidate_id], target)
                if member.split == "train":
                    train.append(row)
                else:
                    validation.append((row, predictions[target.candidate_id]))
    control = fit_local_control(train, classes)
    observations = [{"piece_id": row.member.piece_id, "component_id": row.member.component_id,
                     "observation_id": row.observation_id, "candidate_id": row.target.candidate_id,
                     "gold": row.target.value, "local": control.predict(row.features), "reference": prediction,
                     "target_provenance": row.target.provenance, "target_confidence": row.target.confidence}
                    for row, prediction in validation]
    report = {"schema": "ContextDiagnosticsReport@1.0.0", "membership_sha256": membership_sha256,
              "selected_membership_sha256": fingerprint([asdict(member) for member in ordered]),
              "model_binding": asdict(binding), "local_feature_sha256": LOCAL_FEATURE_SHA256,
              "local_model": asdict(control), "candidate_level": level,
              "coverage": dict(sorted(coverage.items())), "observations": observations,
              "local": classification_metrics([(r["gold"], r["local"]) for r in observations], classes),
              "reference": classification_metrics([(r["gold"], r["reference"]) for r in observations], classes),
              "reference_forward_pieces": reference_calls, "test_access": False,
              "claim_boundary": "local descriptive control; context sensitivity is not correctness; no capacity matching"}
    report["report_fingerprint"] = fingerprint(report)
    return report


def restrict_note_context(piece: CanonicalPiece, start: RationalTime, end: RationalTime) -> CanonicalPiece:
    """Retain only notes overlapping the focus, with original attacks/durations.

    This is an out-of-distribution C2 intervention, never a trained local model.
    All derived graph features must be rebuilt from the returned raw piece.
    Sustaining notes retain their own pre-focus attack and post-focus release;
    this intentionally prevents an artificial attack at the left boundary.
    It is note-inventory restriction, not an exact receptive-field cutoff.
    No gold boundaries, annotations, targets or voice assignments are required.
    """
    if not isinstance(start, RationalTime) or not isinstance(end, RationalTime):
        raise ValueError("context_diagnostics.exact_timing_required")
    if not RationalTime(0) <= start < end <= piece.duration_qn:
        raise ValueError("context_diagnostics.invalid_raw_window")
    notes = []
    for note in piece.notes:
        left, right = max(start, note.onset_qn), min(end, note.onset_qn + note.duration_qn)
        if left < right:
            notes.append(note)
    return replace(piece, notes=tuple(notes), annotations=(), targets=())
