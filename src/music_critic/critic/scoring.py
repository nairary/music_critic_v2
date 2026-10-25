"""Versioned TRAIN-relative scales, with explicit support for comparisons.

Percentiles describe a reference distribution, never calibrated musical quality.
No predictor, target sidecar or corpus access belongs in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping

from music_critic.critic.contracts import ASPECTS, OUTPUT_SCHEMA, CriticOutput

SCORE_POLICY_VERSION = "CriticAspectScorePolicy@1.0.0"


def _scores(values: Mapping[str, float | None]) -> dict[str, float | None]:
    if set(values) - set(ASPECTS):
        raise ValueError("critic.score.unknown_aspect")
    result = {aspect: values.get(aspect) for aspect in ASPECTS}
    for value in result.values():
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("critic.score.nonfinite")
    return result


def _weights(values: Mapping[str, float] | None) -> dict[str, float]:
    result = {aspect: 1.0 for aspect in ASPECTS} if values is None else dict(values)
    if set(result) - set(ASPECTS):
        raise ValueError("critic.score.unknown_aspect")
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("critic.score.weights_invalid")
    if not result or sum(result.values()) <= 0:
        raise ValueError("critic.score.weights_invalid")
    return {aspect: float(result.get(aspect, 0.0)) for aspect in ASPECTS}


def _scope(horizon: str, support_policy: str) -> None:
    if not horizon or not support_policy:
        raise ValueError("critic.score.scope_required")


@dataclass(frozen=True)
class ReferenceRow:
    work_id: str
    scores: Mapping[str, float | None]
    kind: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.work_id or not self.kind:
            raise ValueError("critic.reference.identity_required")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("critic.reference.weight_invalid")
        object.__setattr__(self, "scores", MappingProxyType(_scores(self.scores)))


@dataclass(frozen=True)
class TrainReferenceBank:
    reference_id: str
    rows: tuple[ReferenceRow, ...]
    horizon: str
    support_policy: str
    type_weights: Mapping[str, float]
    predictor_id: str = ""
    split: str = "train"
    version: str = SCORE_POLICY_VERSION

    def __post_init__(self) -> None:
        _scope(self.horizon, self.support_policy)
        if not self.reference_id or self.split != "train" or self.version != SCORE_POLICY_VERSION:
            raise ValueError("critic.reference.binding_invalid")
        if not self.predictor_id or self.predictor_id.lower() in {"unknown", "none", "null"}:
            raise ValueError("critic.reference.predictor_required")
        weights = dict(self.type_weights)
        if (not weights or any(not k or isinstance(v, bool) or not math.isfinite(v)
                               or v < 0 for k, v in weights.items())
                or sum(weights.values()) <= 0):
            raise ValueError("critic.reference.type_weights_invalid")
        if any(row.kind not in weights for row in self.rows):
            raise ValueError("critic.reference.unbound_type")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "type_weights", MappingProxyType(weights))

    @property
    def fingerprint(self) -> str:
        body = {"reference_id": self.reference_id, "version": self.version,
                "split": self.split, "horizon": self.horizon,
                "predictor_id": self.predictor_id,
                "support_policy": self.support_policy, "type_weights": dict(self.type_weights),
                "rows": [{"work_id": r.work_id, "scores": dict(r.scores),
                          "kind": r.kind, "weight": r.weight} for r in self.rows]}
        return sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()).hexdigest()

    def percentile(self, aspect: str, score: float) -> float | None:
        if aspect not in ASPECTS or not math.isfinite(score):
            raise ValueError("critic.reference.query_invalid")
        by_work: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in self.rows:
            value = row.scores[aspect]
            type_weight = self.type_weights[row.kind]
            if value is not None and type_weight > 0:
                # Log weights avoid overflow/underflow before per-work normalization.
                by_work[row.work_id].append((value, math.log(row.weight) + math.log(type_weight)))
        if not by_work:
            return None
        work_cdfs = []
        for logs in by_work.values():
            maximum = max(weight for _, weight in logs)
            values = [(value, math.exp(weight - maximum)) for value, weight in logs]
            total = sum(weight for _, weight in values)
            below = sum(weight for value, weight in values if value < score)
            tied = sum(weight for value, weight in values if value == score)
            work_cdfs.append((below + 0.5 * tied) / total)
        return sum(work_cdfs) / len(work_cdfs)


@dataclass(frozen=True)
class ScoredAspects:
    raw_scores: Mapping[str, float | None]
    percentiles: Mapping[str, float | None]
    overall: float | None
    weighted_coverage: float
    weights: Mapping[str, float]
    horizon: str
    support_policy: str
    reference_fingerprint: str | None
    unavailable_reasons: Mapping[str, str]
    predictor_id: str | None = None
    version: str = SCORE_POLICY_VERSION


@dataclass(frozen=True)
class CommonSupportComparison:
    left_overall: float | None
    right_overall: float | None
    common_aspects: tuple[str, ...]
    weighted_coverage: float
    unavailable_reason: str | None


class AspectScorePolicy:
    def from_output(
        self, outputs: CriticOutput, reference: TrainReferenceBank | None = None,
        weights: Mapping[str, float] | None = None, *, sample_index: int = 0,
        predictor_id: str | None = None,
    ) -> ScoredAspects:
        if outputs.schema != OUTPUT_SCHEMA:
            raise ValueError("critic.score.output_schema_mismatch")
        if type(sample_index) is not int or not 0 <= sample_index < len(outputs.samples):
            raise ValueError("critic.score.sample_index_invalid")
        sample = outputs.samples[sample_index]
        if sample.raw_scores.shape != (len(ASPECTS),) or sample.availability.shape != (len(ASPECTS),):
            raise ValueError("critic.score.output_shape_invalid")
        raw = {aspect: float(sample.raw_scores[i].detach().cpu())
               if bool(sample.availability[i]) else None for i, aspect in enumerate(ASPECTS)}
        horizon = f"{sample.horizon.numerator}/{sample.horizon.denominator}"
        return self.apply(raw, reference, weights, horizon=horizon,
                          support_policy=sample.support_policy, predictor_id=predictor_id)

    def apply(
        self, outputs: Mapping[str, float | None], reference: TrainReferenceBank | None = None,
        weights: Mapping[str, float] | None = None, *, horizon: str, support_policy: str,
        predictor_id: str | None = None,
    ) -> ScoredAspects:
        _scope(horizon, support_policy)
        raw = _scores(outputs)
        resolved = _weights(weights)
        if reference is not None and predictor_id != reference.predictor_id:
            raise ValueError("critic.score.reference_predictor_mismatch")
        if reference is not None and (reference.horizon != horizon
                                      or reference.support_policy != support_policy):
            raise ValueError("critic.score.reference_scope_mismatch")
        percentiles: dict[str, float | None] = {}
        reasons = {}
        for aspect, score in raw.items():
            if score is None:
                percentiles[aspect] = None
                reasons[aspect] = "aspect_unavailable"
            elif reference is None:
                percentiles[aspect] = None
                reasons[aspect] = "reference_unavailable"
            else:
                percentiles[aspect] = reference.percentile(aspect, score)
                if percentiles[aspect] is None:
                    reasons[aspect] = "reference_aspect_unavailable"
        normalized = {a: w / max(resolved.values()) for a, w in resolved.items()}
        used = sum(normalized[a] for a in ASPECTS if percentiles[a] is not None)
        overall = (sum(normalized[a] * percentiles[a] for a in ASPECTS
                       if percentiles[a] is not None) / used if used else None)
        return ScoredAspects(
            MappingProxyType(raw), MappingProxyType(percentiles), overall,
            used / sum(normalized.values()), MappingProxyType(resolved), horizon,
            support_policy, reference.fingerprint if reference is not None else None,
            MappingProxyType(reasons), predictor_id,
        )

    def compare(self, left: ScoredAspects, right: ScoredAspects) -> CommonSupportComparison:
        if (left.version, left.horizon, left.support_policy, left.reference_fingerprint, left.predictor_id,
            dict(left.weights)) != (right.version, right.horizon, right.support_policy,
                                   right.reference_fingerprint, right.predictor_id, dict(right.weights)):
            raise ValueError("critic.score.comparison_binding_mismatch")
        common = tuple(a for a in ASPECTS if left.weights[a] > 0
                       and left.percentiles[a] is not None and right.percentiles[a] is not None)
        normalized = {a: w / max(left.weights.values()) for a, w in left.weights.items()}
        mass = sum(normalized[a] for a in common)
        if not mass:
            return CommonSupportComparison(None, None, (), 0.0, "no_common_reference_support")
        return CommonSupportComparison(
            sum(normalized[a] * left.percentiles[a] for a in common) / mass,
            sum(normalized[a] * right.percentiles[a] for a in common) / mass,
            common, mass / sum(normalized.values()), None,
        )
