"""Dependency-free, mask-aware metrics for oracle-H membership queries.

Primary AUC averages supported H within each work, then supported works. Unknown
labels never enter classification scores, but their notes remain retrieval
candidates. Retrieval ties use input candidate order, never target labels.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
import math
from numbers import Real
from typing import Any

__all__ = ["score_metrics", "choose_threshold", "choose_threshold_exact", "choose_threshold_fpr"]
_EPSILON = 1e-12


def _validate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list or tuple of query mappings")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    families: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a mapping")
        for key in ("work", "h_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        identity = row["work"], row["h_id"]
        if identity in seen:
            raise ValueError(f"duplicate (work, h_id): {identity!r}")
        seen.add(identity)
        family = row.get("family")
        if family is not None:
            if not isinstance(family, str) or not family.strip():
                raise ValueError("family must be a nonempty string when supplied")
            if families.setdefault(row["work"], family) != family:
                raise ValueError("a work cannot belong to conflicting families")
        for key in ("scores", "y", "mask"):
            if not isinstance(row.get(key), (list, tuple)):
                raise ValueError(f"{key} must be a list or tuple")
        scores, labels, mask = row["scores"], row["y"], row["mask"]
        if len(scores) != len(labels) or len(scores) != len(mask):
            raise ValueError("scores, y and mask must have equal lengths")
        if any(not isinstance(m, bool) for m in mask):
            raise ValueError("mask entries must be booleans")
        if any(not isinstance(y, (int, bool)) or y not in (0, 1) for y in labels):
            raise ValueError("y entries must be boolean or integer 0/1")
        if any(
            isinstance(s, bool) or not isinstance(s, Real)
            or not math.isfinite(s) or not 0 <= s <= 1 for s in scores
        ):
            raise ValueError("scores must be finite probabilities in [0, 1]")
        result.append({
            "work": row["work"], "h_id": row["h_id"], "family": family,
            "scores": [float(s) for s in scores],
            "y": [int(y) for y in labels], "mask": list(mask),
        })
    return result


def _average(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _divide(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney AUC with exact score ties; O(n log n), O(n) storage."""
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    pairs = sorted(zip(scores, labels))
    negatives_below = 0
    wins = 0.0
    start = 0
    while start < len(pairs):
        stop = start + 1
        while stop < len(pairs) and pairs[stop][0] == pairs[start][0]:
            stop += 1
        tied_positive = sum(y for _, y in pairs[start:stop])
        tied_negative = stop - start - tied_positive
        wins += tied_positive * (negatives_below + 0.5 * tied_negative)
        negatives_below += tied_negative
        start = stop
    return wins / (positive * negative)


def _classification(pairs: Sequence[tuple[float, int]], threshold: float) -> dict[str, Any]:
    tp = sum(y == 1 and score >= threshold for score, y in pairs)
    fp = sum(y == 0 and score >= threshold for score, y in pairs)
    fn = sum(y == 1 and score < threshold for score, y in pairs)
    tn = len(pairs) - tp - fp - fn

    def class_metrics(correct: int, incorrect: int, missed: int) -> dict[str, Any]:
        return {
            "support": correct + missed, "predicted": correct + incorrect,
            "precision": _divide(correct, correct + incorrect),
            "recall": _divide(correct, correct + missed),
            "f1": _divide(2 * correct, 2 * correct + incorrect + missed),
        }

    positive = class_metrics(tp, fp, fn)
    negative = class_metrics(tn, fn, fp)
    return {
        "known": len(pairs), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        # Fixed two-class F1 counts an undefined class as zero. Without known
        # labels the whole metric is unsupported, rather than zero.
        "macro_f1": ((positive["f1"] or 0.0) + (negative["f1"] or 0.0)) / 2
        if pairs else None,
        "macro_f1_policy": "fixed_classes_0_1_undefined_class_zero",
        "precision": positive["precision"], "recall": positive["recall"],
        "fpr": _divide(fp, fp + tn),
        "brier": _average([(score - y) ** 2 for score, y in pairs]),
        "bce": _average([
            -math.log(max(_EPSILON, min(1 - _EPSILON, score if y else 1 - score)))
            for score, y in pairs
        ]),
        "bce_probability_clip": _EPSILON,
        "per_class": {"0": negative, "1": positive},
    }


def _summarize(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    aucs: list[float] = []
    recalls: dict[int, list[float]] = {1: [], 5: []}
    candidate_count = positive_count = eligible_h = 0
    for row in rows:
        scores, labels, mask = row["scores"], row["y"], row["mask"]
        known = [(score, y) for score, y, m in zip(scores, labels, mask) if m]
        pairs.extend(known)
        candidate_count += len(scores)
        positives = sum(y for _, y in known)
        positive_count += positives
        auc = _auc([score for score, _ in known], [y for _, y in known])
        if auc is not None:
            aucs.append(auc)
            eligible_h += 1
        if positives:
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            for k in recalls:
                recovered = sum(mask[index] and labels[index] == 1 for index in order[:k])
                recalls[k].append(recovered / positives)
    return {
        "primary_auc": _average(aucs), "eligible_h": eligible_h,
        "coverage": {
            "total_h": len(rows), "candidates": candidate_count,
            "known": len(pairs), "positive": positive_count,
            "negative": len(pairs) - positive_count,
            "unknown": candidate_count - len(pairs),
        },
        "global": _classification(pairs, threshold),
        "retrieval": {
            "recall_at_1": _average(recalls[1]), "recall_at_5": _average(recalls[5]),
            "eligible_h": len(recalls[1]), "tie_policy": "stable_input_candidate_order",
            "candidate_policy": "all_candidates_including_unknown",
        },
    }


def score_metrics(rows: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate membership probabilities without treating unknowns as negatives.

    Every row is one unique ``(work, h_id)`` query containing equally sized
    ``scores``, ``y`` and boolean ``mask`` lists. ``family`` is optional metadata.
    AUC and main Recall@k are H-macro within work followed by work-macro. The
    query-macro Recall@k is reported separately. Classification aggregates all
    known candidates, with per-work metrics alongside it. Unsupported ratios
    are ``None``; fixed-two-class F1's zero-division convention is explicit.
    """
    if (
        isinstance(threshold, bool) or not isinstance(threshold, Real)
        or not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and in [0, 1]")
    checked = _validate(rows)
    by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checked:
        by_work[row["work"]].append(row)
    per_work = {work: _summarize(group, threshold) for work, group in sorted(by_work.items())}
    for work, group in by_work.items():
        families = [row["family"] for row in group if row["family"] is not None]
        per_work[work]["family"] = families[0] if families else None
    result = _summarize(checked, threshold)
    eligible_work_scores = [
        value["primary_auc"] for value in per_work.values() if value["primary_auc"] is not None
    ]
    result["primary_auc"] = _average(eligible_work_scores)
    result["eligible_works"] = len(eligible_work_scores)
    result["coverage"]["total_works"] = len(per_work)
    retrieval = result["retrieval"]
    for k in (1, 5):
        key = f"recall_at_{k}"
        retrieval[f"query_macro_{key}"] = retrieval[key]
        retrieval[key] = _average([
            value["retrieval"][key] for value in per_work.values()
            if value["retrieval"][key] is not None
        ])
    retrieval["eligible_works"] = sum(
        value["retrieval"]["eligible_h"] > 0 for value in per_work.values()
    )
    result.update({
        "threshold": float(threshold),
        "primary_aggregation": "mean_H_within_work_then_mean_supported_works",
        "retrieval_aggregation": "mean_positive_H_within_work_then_mean_supported_works",
        "per_work": per_work,
        "controls": {"source_only_primary_auc": 0.5 if eligible_work_scores else None},
    })
    return result


def choose_threshold(selection_rows: Sequence[Mapping[str, Any]]) -> float:
    """Choose on inner-fit selection rows only, never exploratory dev or TEST.

    Maximize known-target global fixed-two-class macro F1 on 0.1,...,0.9.
    Ties prefer the grid point nearest 0.5, then the lower point. The caller
    owns split admission: row contents cannot establish which split was read.
    Raise for no known support, rather than silently selecting a threshold.
    """
    checked = _validate(selection_rows)
    pairs = [
        (score, y) for row in checked
        for score, y, mask in zip(row["scores"], row["y"], row["mask"]) if mask
    ]
    if not pairs:
        raise ValueError("threshold selection requires known labels in inner-fit selection")
    tick = max(
        range(1, 10),
        key=lambda value: (_classification(pairs, value / 10)["macro_f1"], -abs(value - 5), -value),
    )
    return tick / 10


def _selection_pairs(selection_rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, int]]:
    checked = _validate(selection_rows)
    pairs = sorted(
        (score, y) for row in checked
        for score, y, mask in zip(row["scores"], row["y"], row["mask"]) if mask
    )
    if not pairs:
        raise ValueError("threshold selection requires known labels in inner-fit selection")
    return pairs


def _threshold_confusions(
    pairs: Sequence[tuple[float, int]],
    min_threshold: float = 0.0, max_threshold: float = 1.0,
) -> Iterator[tuple[float, int, int, int, int]]:
    """Yield attainable boundary, TP, FP, TN, FN for nonempty sorted pairs."""
    tp = fp = 0
    fn = sum(y for _, y in pairs)
    tn = len(pairs) - fn
    maximum = pairs[-1][0]
    candidates = {min_threshold, max_threshold}
    candidates.update(tick / 10 for tick in range(1, 10)
                      if min_threshold <= tick / 10 <= max_threshold)
    candidates.update(score for score, _ in pairs if min_threshold <= score <= max_threshold)
    if maximum < 1.0:
        all_negative = math.nextafter(maximum, 1.0)
        if min_threshold <= all_negative <= max_threshold:
            candidates.add(all_negative)
    stop = len(pairs)
    for threshold in sorted(candidates, reverse=True):
        # Move every score meeting the inclusive threshold before scoring;
        # equal scores cannot be split, including those at either endpoint.
        while stop and pairs[stop - 1][0] >= threshold:
            positive = pairs[stop - 1][1]
            tp, fn = tp + positive, fn - positive
            fp, tn = fp + 1 - positive, tn - 1 + positive
            stop -= 1
        yield threshold, tp, fp, tn, fn


def choose_threshold_exact(
    selection_rows: Sequence[Mapping[str, Any]], *,
    min_threshold: float = 0.0, max_threshold: float = 1.0,
) -> float:
    """Maximize empirical known-target global macro F1 in O(n log n).

    This opt-in policy leaves :func:`choose_threshold` unchanged. The caller
    admits inner-fit selection data; row contents cannot establish a split.
    Unknown labels contribute neither an objective term nor a score boundary.

    With the inclusive ``score >= threshold`` rule, the candidate thresholds
    are each distinct known score and historical grid point 0.1,...,0.9 in the
    requested inclusive bounds, both endpoints, and (if within bounds and the
    maximum score is below 1) its next representable float towards 1. Nesting
    the historical grid preserves its tie preference on an unchanged plateau.
    These cover every attainable prediction
    partition within the bounds. The extra boundary represents all-negative prediction;
    it is impossible when a known score is exactly 1, so no out-of-range
    sentinel is returned. Defaults cover [0, 1]. Restricting the bounds, e.g.
    to [0.1, 0.9], isolates grid resolution from the historical range limit.

    Objective ties use exact integer cross-products, then prefer the candidate
    nearest 0.5 (float distance), then the lower candidate. Thus a returned
    threshold is an empirical boundary, historical grid point or endpoint.
    As for the historical policy, missing known support raises ValueError.
    """
    for name, value in (("min_threshold", min_threshold), ("max_threshold", max_threshold)):
        if (
            isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold must not exceed max_threshold")
    pairs = _selection_pairs(selection_rows)
    best_threshold: float | None = None
    best_numerator, best_denominator = -1, 1
    for threshold, tp, fp, tn, fn in _threshold_confusions(pairs, float(min_threshold), float(max_threshold)):
        # macro F1 = TP/(2TP+FP+FN) + TN/(2TN+FP+FN), with
        # undefined class terms zero. Compare fractions without roundoff.
        positive_denominator = 2 * tp + fp + fn or 1
        negative_denominator = 2 * tn + fp + fn or 1
        numerator = tp * negative_denominator + tn * positive_denominator
        denominator = positive_denominator * negative_denominator
        objective_difference = numerator * best_denominator - best_numerator * denominator
        if objective_difference > 0 or (
            objective_difference == 0 and best_threshold is not None
            and (abs(threshold - 0.5), threshold)
            < (abs(best_threshold - 0.5), best_threshold)
        ):
            best_threshold = threshold
            best_numerator, best_denominator = numerator, denominator

    assert best_threshold is not None
    return best_threshold


def choose_threshold_fpr(
    selection_rows: Sequence[Mapping[str, Any]], max_fpr: float = 0.1,
) -> float:
    """Maximize known-positive recall subject to empirical FPR <= max_fpr.

    The caller admits inner-fit selection rows and declares the error limit
    before dev evaluation; this selection constraint gives no dev guarantee.
    Require known positives and negatives, excluding unknowns completely.
    Sweep the same attainable [0, 1] boundaries as ``choose_threshold_exact``
    in O(n log n). Recall ties prefer the candidate nearest 0.5 (float
    distance), then the lower candidate, without a secondary FPR objective.
    Raise explicitly if inclusive score==1 negatives make the limit impossible;
    an invalid threshold above 1 is never used to manufacture feasibility.
    """
    if (
        isinstance(max_fpr, bool) or not isinstance(max_fpr, Real)
        or not math.isfinite(max_fpr) or not 0 <= max_fpr <= 1
    ):
        raise ValueError("max_fpr must be finite and in [0, 1]")
    pairs = _selection_pairs(selection_rows)
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("FPR threshold selection requires both known positive and negative labels")
    feasible = (
        (tp, -abs(threshold - 0.5), -threshold, threshold)
        for threshold, tp, fp, _, _ in _threshold_confusions(pairs)
        if fp / negatives <= max_fpr
    )
    best = max(feasible, default=None)
    if best is None:
        raise ValueError("no attainable threshold in [0, 1] satisfies max_fpr; known negative score==1")
    return best[-1]
