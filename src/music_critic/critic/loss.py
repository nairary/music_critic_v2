"""Masked weak ranking and intervention-presence loss, without quality claims."""

from dataclasses import dataclass
from collections import defaultdict
from fractions import Fraction
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import ASPECTS, CriticOutput, SamplePrediction
from .targets import PairTargets


@dataclass(frozen=True)
class LossResult:
    total: Tensor | None
    pair: Tensor | None
    local: Tensor | None
    pair_support: int
    local_support: int
    skip_update: bool


class CriticLoss:
    def __init__(self, lambda_local: float = 0.25):
        if type(lambda_local) not in (int, float) or not math.isfinite(lambda_local) or lambda_local < 0:
            raise ValueError("lambda_local must be finite and nonnegative")
        self.lambda_local = float(lambda_local)

    def __call__(self, output_pair, target_sidecar) -> LossResult:
        pair_groups, local_groups = defaultdict(list), defaultdict(list)
        pair_support = local_support = 0
        for a, b, target in _aligned_pairs(output_pair, target_sidecar):
            for i, aspect in enumerate(ASPECTS):
                if target.pair_mask[i] and a.availability[i] and b.availability[i]:
                    difference = a.raw_scores[i] - b.raw_scores[i]
                    _require_finite(difference)
                    pair_groups[target.work_id, aspect].append(F.softplus(-target.preference[i] * difference))
                    pair_support += 1
            for prediction, local_targets in zip((a, b), target.local_targets):
                for _namespace, predicted, labels, indices in _observed_local(
                        prediction, local_targets, allow_missing=self.lambda_local == 0):
                    for i in indices:
                        value, logit = labels.values[i], predicted.logits[i]
                        local_groups[target.work_id, predicted.aspect, predicted.candidate_type, value].append(
                            F.binary_cross_entropy_with_logits(logit, logit.new_tensor(value)))
                        local_support += 1
        pair = _macro_work_aspect({key: torch.stack(terms).mean() for key, terms in pair_groups.items()})
        type_groups = defaultdict(list)
        for (work, aspect, kind, _class), terms in local_groups.items():
            type_groups[work, aspect, kind].append(torch.stack(terms).mean())
        work_groups = defaultdict(list)
        for (work, aspect, _kind), classes in type_groups.items():
            work_groups[work, aspect].append(torch.stack(classes).mean())
        local = _macro_work_aspect({key: torch.stack(kinds).mean() for key, kinds in work_groups.items()})
        terms = ([pair] if pair is not None else []) + (
            [self.lambda_local * local] if local is not None and self.lambda_local > 0 else [])
        total = torch.stack(terms).sum() if terms else None
        return LossResult(total, pair, local, pair_support, local_support, total is None)


def _macro_work_aspect(groups) -> Tensor | None:
    """Equal works inside each supported aspect, then equal supported aspects."""
    aspects = defaultdict(list)
    for (_work, aspect), value in groups.items():
        aspects[aspect].append(value)
    return torch.stack([torch.stack(values).mean() for values in aspects.values()]).mean() if aspects else None


def _validate_vector(values: Tensor, availability: Tensor, size: int) -> None:
    if values.shape != (size,) or availability.shape != (size,):
        raise ValueError("prediction vector shape mismatch")
    if availability.dtype != torch.bool:
        raise ValueError("prediction availability must be boolean")
    if not values.is_floating_point() or values.device != availability.device:
        raise ValueError("prediction requires floating scores and colocated availability")


def _require_finite(value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("observed prediction must be finite")


def _aligned_pairs(output_pair, target_sidecar):
    left, right = output_pair
    left = left.samples if isinstance(left, CriticOutput) else (left,)
    right = right.samples if isinstance(right, CriticOutput) else (right,)
    targets = (target_sidecar,) if isinstance(target_sidecar, PairTargets) else tuple(target_sidecar)
    if not len(left) == len(right) == len(targets):
        raise ValueError("prediction/target batch lengths disagree")
    for a, b, target in zip(left, right, targets):
        if not isinstance(a, SamplePrediction) or not isinstance(b, SamplePrediction):
            raise ValueError("sample predictions required")
        for prediction in (a, b):
            _validate_vector(prediction.raw_scores, prediction.availability, len(ASPECTS))
        if (a.sample_id, b.sample_id) != target.sample_ids:
            raise ValueError("prediction/target sample identities disagree")
        if a.horizon != b.horizon or a.support_policy != b.support_policy:
            raise ValueError("pair prediction scope mismatch")
        yield a, b, target


def _observed_local(prediction, local_targets, *, allow_missing=False):
    for namespace, labels in local_targets.items():
        if namespace not in prediction.local:
            if any(labels.mask) and not allow_missing:
                raise ValueError("observed local target has no prediction namespace")
            continue
        predicted = prediction.local[namespace]
        _validate_vector(predicted.logits, predicted.availability, len(predicted.candidates))
        if namespace != f"{predicted.aspect}/{predicted.candidate_type}" or predicted.aspect not in ASPECTS:
            raise ValueError("local prediction namespace disagrees")
        if predicted.candidates != labels.candidates:
            raise ValueError("local prediction/target candidates disagree")
        indices = tuple(i for i, observed in enumerate(labels.mask) if observed and predicted.availability[i])
        for i in indices:
            _require_finite(predicted.logits[i])
        yield namespace, predicted, labels, indices


@dataclass(frozen=True)
class LocalMetric:
    precision: float | None
    recall: float | None
    interval_iou: float | None
    observed_support: int
    positive_support: int
    predicted_support: int
    work_support: int
    precision_work_support: int
    recall_work_support: int
    interval_work_support: int


@dataclass(frozen=True)
class LocalEvaluation:
    per_namespace: Mapping[str, LocalMetric]
    logit_threshold: float
    claim: str = "synthetic_intervention_presence"


@torch.no_grad()
def evaluate_local_interventions(output_pair, target_sidecar, *, logit_threshold: float = 0.0) -> LocalEvaluation:
    """Work-macro detection metrics; interval IoU is only for raw slices/beats.

    Undefined precision/recall and empty interval unions remain unavailable.
    Interval intersections/unions are measured per view before summing per work,
    so time shared by different examples is never collapsed into one interval.
    """
    if not math.isfinite(logit_threshold):
        raise ValueError("finite local threshold required")
    groups = defaultdict(lambda: [0, 0, 0, 0, Fraction(0), Fraction(0)])
    namespaces = set()
    for left, right, target in _aligned_pairs(output_pair, target_sidecar):
        for prediction, labels_by_type in zip((left, right), target.local_targets):
            namespaces.update(labels_by_type)
            for namespace, predicted, labels, indices in _observed_local(prediction, labels_by_type):
                if not indices:
                    continue
                counts = groups[target.work_id, namespace]
                actual, detected = [], []
                for i in indices:
                    truth, guess = bool(labels.values[i]), bool(predicted.logits[i] >= logit_threshold)
                    counts[0] += int(truth and guess)
                    counts[1] += int(not truth and guess)
                    counts[2] += int(truth and not guess)
                    counts[3] += 1
                    candidate = labels.candidates[i]
                    if truth:
                        actual.append((candidate.start, candidate.end))
                    if guess:
                        detected.append((candidate.start, candidate.end))
                if namespace in {"harmony/slice", "rhythm/beat"}:
                    intersection, union = _interval_overlap(actual, detected)
                    counts[4] += intersection
                    counts[5] += union
    result = {}
    mean = lambda values: sum(values) / len(values) if values else None
    for namespace in sorted(namespaces):
        rows = [counts for (_work, key), counts in groups.items() if key == namespace]
        precision = [tp / (tp + fp) for tp, fp, _fn, _n, _i, _u in rows if tp + fp]
        recall = [tp / (tp + fn) for tp, _fp, fn, _n, _i, _u in rows if tp + fn]
        overlap = [float(i / u) for _tp, _fp, _fn, _n, i, u in rows if u]
        result[namespace] = LocalMetric(mean(precision), mean(recall), mean(overlap),
            sum(row[3] for row in rows), sum(row[0] + row[2] for row in rows),
            sum(row[0] + row[1] for row in rows), len(rows), len(precision), len(recall), len(overlap))
    return LocalEvaluation(result, float(logit_threshold))


def _interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _interval_overlap(actual, detected):
    actual, detected = _interval_union(actual), _interval_union(detected)
    i = j = 0
    intersection = Fraction(0)
    while i < len(actual) and j < len(detected):
        a, b = actual[i], detected[j]
        intersection += max(Fraction(0), min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    union = sum((end - start for start, end in (*actual, *detected)), Fraction(0)) - intersection
    return intersection, union
