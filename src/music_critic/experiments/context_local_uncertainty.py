"""Paired, bounded component bootstrap of already eligible VALIDATION rows.

This module performs no inference, target loading, model selection or event-level
resampling. Official membership (including components without labels) is supplied
by the caller. Intervals describe this finite diagnostic, not model superiority.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random

from music_critic.experiments.context_diagnostics import (
    Member, classification_metrics, fingerprint, validate_membership,
)


MAX_VALIDATION_PIECES = 32
MAX_OBSERVATIONS_PER_PIECE = 20_000
MAX_OBSERVATIONS = MAX_VALIDATION_PIECES * MAX_OBSERVATIONS_PER_PIECE
MAX_CLASSES = 256
MAX_DRAWS = 200
_ARMS = ("local", "reference", "reference_minus_local")
_METRICS = ("accuracy", "observed_class_macro_f1")
_CLASS_METRICS = ("precision", "recall", "f1")


def _fail(reason: str) -> None:
    raise ValueError(f"context_local_uncertainty.{reason}")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True, slots=True)
class _Observation:
    piece_id: str
    component_id: str
    observation_id: str
    candidate_id: str
    gold: int
    local: int
    reference: int
    target_provenance: str
    target_confidence: float | int | None


class _Pairs(Sequence):
    """Read-only view: reuse the existing metric oracle without copying events."""
    def __init__(self, rows: Sequence[_Observation], arm: str):
        self.rows, self.arm = rows, arm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [(row.gold, getattr(row, self.arm)) for row in self.rows[index]]
        row = self.rows[index]
        return row.gold, getattr(row, self.arm)

    def __iter__(self):
        return ((row.gold, getattr(row, self.arm)) for row in self.rows)


def _observation_digest(rows: Sequence[_Observation]) -> str:
    digest = sha256()
    for row in rows:
        digest.update(json.dumps(asdict(row), sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _delta(reference: float | None, local: float | None) -> float | None:
    return None if reference is None or local is None else reference - local


def _from_counts(counts: Counter, size: int, fixed_classes: Sequence[int]) -> tuple[dict, list[dict]]:
    support, predicted, true = [0] * size, [0] * size, [0] * size
    for (gold, prediction), count in counts.items():
        support[gold] += count
        predicted[prediction] += count
        if gold == prediction:
            true[gold] += count
    total = sum(support)
    # No per-class comparison without gold in this resample, even if the model
    # predicts that class. Point confusion statistics still retain false positives.
    per_class = [{
        "precision": true[i] / predicted[i] if support[i] and predicted[i] else None,
        "recall": true[i] / support[i] if support[i] else None,
        "f1": 2 * true[i] / (support[i] + predicted[i]) if support[i] else None,
    } for i in range(size)]
    macro = (math.fsum(per_class[i]["f1"] for i in fixed_classes) / len(fixed_classes)
             if fixed_classes and all(support[i] for i in fixed_classes) else None)
    return {"accuracy": sum(true) / total if total else None,
            "observed_class_macro_f1": macro}, per_class


def _quantile(values: Sequence[float], numerator: int) -> float:
    """Linear interpolation at numerator/40: 1/40 and 39/40 are 95% endpoints."""
    quotient, remainder = divmod((len(values) - 1) * numerator, 40)
    if not remainder:
        return values[quotient]
    return values[quotient] + (values[quotient + 1] - values[quotient]) * remainder / 40


def _interval(point: float | None, values: Sequence[float | None], groups: int) -> dict:
    defined = sorted(value for value in values if value is not None)
    if point is None:
        reason = "point_metric_undefined"
    elif groups < 2:
        reason = "fewer_than_two_eligible_components"
    elif len(defined) < 2:
        reason = "fewer_than_two_defined_draws"
    else:
        reason = None
    return {"point_estimate": point, "lower": _quantile(defined, 1) if reason is None else None,
            "upper": _quantile(defined, 39) if reason is None else None,
            "level": 0.95, "eligible_component_support": groups,
            "defined_draws": len(defined), "undefined_draws": len(values) - len(defined),
            "conditional_on_defined_draws": len(defined) != len(values),
            "undefined_reason": reason}


def paired_component_report(observations: Sequence[Mapping], classes: Sequence[str],
                            validation_members: Sequence[Member], *, seed: int = 17,
                            draws: int = 200) -> dict:
    """Report reference (C2) minus local on exactly the same eligible rows.

    Resample N official validation components with replacement N times per draw;
    apply each multiplicity to both arms' component confusion counts. Within a
    sampled component all source observations keep their original weight. The
    estimand remains pooled observation accuracy/F1, not mean component accuracy.

    The macro class set is fixed by positive gold support in the point estimate.
    A draw missing any of these classes has macro=null in both arms. Percentile
    intervals use defined draws only and explicitly expose that conditioning.
    At least two eligible independent components and two defined draws are needed;
    class intervals additionally require gold support in two distinct components.
    Input rows must already be eligible: masked/unknown values are rejected, never
    converted to negatives or silently removed. No member means no sampling frame.
    """
    if (not isinstance(classes, Sequence) or isinstance(classes, (str, bytes))
            or not 2 <= len(classes) <= MAX_CLASSES
            or any(not _text(value) for value in classes) or len(set(classes)) != len(classes)):
        _fail("invalid_classes")
    if type(seed) is not int or not 0 <= seed < 2**32:
        _fail("invalid_seed")
    if type(draws) is not int or not 1 <= draws <= MAX_DRAWS:
        _fail("draw_budget")
    if (not isinstance(validation_members, Sequence) or not validation_members
            or len(validation_members) > MAX_VALIDATION_PIECES):
        _fail("membership_budget")
    for member in validation_members:
        if (not isinstance(member, Member) or member.split != "validation"
                or not _text(member.piece_id) or not _text(member.component_id)):
            _fail("official_VALIDATION_members_required")
    members = validate_membership(validation_members)
    membership = {member.piece_id: member.component_id for member in members}
    if not isinstance(observations, Sequence) or len(observations) > MAX_OBSERVATIONS:
        _fail("observation_budget")
    rows, seen, piece_counts = [], set(), Counter()
    size = len(classes)
    for value in observations:
        if not isinstance(value, Mapping):
            _fail("observation_mapping_required")
        for key in ("piece_id", "component_id", "observation_id", "candidate_id", "target_provenance"):
            if not _text(value.get(key)):
                _fail(f"invalid_{key}")
        piece_id = value["piece_id"]
        if (piece_id not in membership or value["component_id"] != membership[piece_id]
                or value.get("split", "validation") != "validation"):
            _fail("observation_membership_mismatch")
        identity = piece_id, value["observation_id"]
        if identity in seen:
            _fail("duplicate_observation")
        seen.add(identity)
        piece_counts[piece_id] += 1
        if piece_counts[piece_id] > MAX_OBSERVATIONS_PER_PIECE:
            _fail("per_piece_observation_budget")
        for key in ("gold", "local", "reference"):
            if type(value.get(key)) is not int or not 0 <= value[key] < size:
                _fail(f"invalid_{key}")
        if (value.get("available", True) is not True or value.get("masked", False) is not False
                or value.get("reason") is not None):
            _fail("eligible_observations_only")
        confidence = value.get("target_confidence")
        if confidence is not None and (type(confidence) not in (int, float)
                                       or not 0 <= confidence <= 1 or not math.isfinite(confidence)):
            _fail("invalid_target_confidence")
        rows.append(_Observation(*(value[key] for key in (
            "piece_id", "component_id", "observation_id", "candidate_id", "gold", "local",
            "reference", "target_provenance")), confidence))
    rows.sort(key=lambda row: (row.piece_id, row.observation_id))
    component_ids = sorted(set(membership.values()))
    counts = {component: {arm: Counter() for arm in _ARMS[:2]} for component in component_ids}
    gold_records, gold_components = [set() for _ in classes], [set() for _ in classes]
    prediction_records = {arm: [set() for _ in classes] for arm in _ARMS[:2]}
    prediction_components = {arm: [set() for _ in classes] for arm in _ARMS[:2]}
    for row in rows:
        gold_records[row.gold].add(row.piece_id)
        gold_components[row.gold].add(row.component_id)
        for arm in _ARMS[:2]:
            prediction = getattr(row, arm)
            counts[row.component_id][arm][row.gold, prediction] += 1
            prediction_records[arm][prediction].add(row.piece_id)
            prediction_components[arm][prediction].add(row.component_id)
    eligible_groups = sum(bool(counts[component]["local"]) for component in component_ids)
    fixed_classes = [i for i in range(size) if gold_records[i]]
    point = {arm: classification_metrics(_Pairs(rows, arm), classes) for arm in _ARMS[:2]}
    point["reference_minus_local"] = {
        metric: _delta(point["reference"][metric], point["local"][metric]) for metric in _METRICS}
    for arm in _ARMS[:2]:
        for i, row in enumerate(point[arm]["classes"]):
            row.update(gold_record_support=len(gold_records[i]), gold_component_support=len(gold_components[i]),
                       prediction_record_support=len(prediction_records[arm][i]),
                       prediction_component_support=len(prediction_components[arm][i]))
    endpoint_values = {arm: {metric: [] for metric in _METRICS} for arm in _ARMS}
    class_values = [{arm: {metric: [] for metric in _CLASS_METRICS} for arm in _ARMS} for _ in classes]
    rng, draw_rows = random.Random(seed), []
    for draw in range(draws):
        multiplicities = Counter(rng.randrange(len(component_ids)) for _ in component_ids)
        combined = {arm: Counter() for arm in _ARMS[:2]}
        for index, multiplier in sorted(multiplicities.items()):
            for arm in _ARMS[:2]:
                for key, count in counts[component_ids[index]][arm].items():
                    combined[arm][key] += multiplier * count
        metrics, per_class = {}, {}
        for arm in _ARMS[:2]:
            metrics[arm], per_class[arm] = _from_counts(combined[arm], size, fixed_classes)
        metrics["reference_minus_local"] = {
            metric: _delta(metrics["reference"][metric], metrics["local"][metric]) for metric in _METRICS}
        for arm in _ARMS:
            for metric in _METRICS:
                endpoint_values[arm][metric].append(metrics[arm][metric])
        for i in range(size):
            for metric in _CLASS_METRICS:
                for arm in _ARMS[:2]:
                    class_values[i][arm][metric].append(per_class[arm][i][metric])
                class_values[i]["reference_minus_local"][metric].append(
                    _delta(per_class["reference"][i][metric], per_class["local"][i][metric]))
        draw_rows.append({"draw": draw, "component_multiplicities": [multiplicities[i] for i in range(len(component_ids))],
                          "eligible_observations": sum(combined["local"].values()), **metrics})
    class_intervals = []
    for i, label in enumerate(classes):
        class_point = {arm: {metric: point[arm]["classes"][i][metric] if gold_records[i] else None
                            for metric in _CLASS_METRICS} for arm in _ARMS[:2]}
        class_point["reference_minus_local"] = {
            metric: _delta(class_point["reference"][metric], class_point["local"][metric])
            for metric in _CLASS_METRICS}
        class_intervals.append({"id": i, "label": label, "gold_record_support": len(gold_records[i]),
                                "gold_component_support": len(gold_components[i]),
                                "intervals": {arm: {metric: _interval(class_point[arm][metric],
                                    class_values[i][arm][metric], len(gold_components[i]))
                                    for metric in _CLASS_METRICS} for arm in _ARMS}})
    report = {
        "schema": "ContextPairedComponentReport@1.0.0", "classes": list(classes),
        "class_sha256": fingerprint(list(classes)),
        "validation_membership_sha256": fingerprint([asdict(member) for member in members]),
        "observations_sha256": _observation_digest(rows),
        "observation_hash_contract": "sorted normalized eligible rows as UTF-8 canonical JSON lines, including provenance/confidence",
        "support": {"official_validation_records": len(members), "official_validation_components": len(component_ids),
                    "eligible_records": len(piece_counts), "eligible_components": eligible_groups,
                    "eligible_observations": len(rows),
                    "unknown_target_confidence": sum(row.target_confidence is None for row in rows)},
        "members": [{**asdict(member), "eligible_observations": piece_counts[member.piece_id]} for member in members],
        "component_order": component_ids,
        "components": [{"component_id": component, "piece_ids": sorted(p for p, c in membership.items() if c == component),
                        "confusion_counts": {arm: [[gold, pred, count] for (gold, pred), count in sorted(counts[component][arm].items())]
                                             for arm in _ARMS[:2]}} for component in component_ids],
        "point": point,
        "bootstrap": {"seed": seed, "requested_draws": draws, "sampling_unit": "official_validation_component",
                      "paired": True, "fixed_macro_gold_class_ids": fixed_classes,
                      "estimand": "pooled eligible source observations; component sampling preserves within-component weights",
                      "interval_method": "95% percentile, linear interpolation at (n-1)*q, defined draws only",
                      "undefined_policy": "null for no eligible rows or missing any fixed macro gold class; no zero fill; per-class gold required",
                      "intervals": {arm: {metric: _interval(point[arm][metric], endpoint_values[arm][metric], eligible_groups)
                                           for metric in _METRICS} for arm in _ARMS},
                      "per_class": class_intervals, "draws": draw_rows},
        "claim_boundary": "descriptive paired C2-minus-local diagnostic on available labels; conditional intervals may omit undefined draws; no event resampling, model selection, significance or general superiority claim",
        "test_access": False,
    }
    report["report_fingerprint"] = fingerprint(report)
    return report
