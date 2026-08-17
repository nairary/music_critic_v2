"""Primary score, tie-breaks, and component bootstrap for Phase 9C-A."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean
from typing import Mapping

from .contracts import PHASE9C_SELECTION_VERSION, Phase9CContractError, TASK_IDS, fingerprint


BOOTSTRAP_CONTRACT_VERSION = "1.0.0"


def primary_validation_summary(report: Mapping[str, object]) -> dict[str, object]:
    tasks = report.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASK_IDS):
        raise Phase9CContractError("phase9c.selection.task_inventory_invalid")
    normalized: list[float] = []
    macro_f1: list[float] = []
    nlls: list[float] = []
    per_task = {}
    for task_id in TASK_IDS:
        row = tasks[task_id]
        if not isinstance(row, Mapping) or row.get("available") is not True:
            raise Phase9CContractError(f"phase9c.selection.task_unavailable:{task_id}")
        class_count = int(row["class_count"])
        nll = float(row["nll"])
        if class_count <= 1 or not math.isfinite(nll):
            raise Phase9CContractError("phase9c.selection.metric_invalid")
        score = nll / math.log(class_count)
        normalized.append(score)
        nlls.append(nll)
        macro_f1.append(float(row["macro_f1"]))
        per_task[task_id] = {
            "nll": nll,
            "class_count": class_count,
            "normalized_nll": score,
            "macro_f1": float(row["macro_f1"]),
        }
    payload = {
        "contract_version": PHASE9C_SELECTION_VERSION,
        "primary_metric": "mean_task_nll_div_log_class_count",
        "lower_is_better": True,
        "primary_score": mean(normalized),
        "mean_macro_f1": mean(macro_f1),
        "mean_task_nll": mean(nlls),
        "tasks": per_task,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _entry_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["task_id"],
        row["dataset_id"],
        row["piece_id"],
        row["source_entry_index"],
        row["label"],
    )


def component_bootstrap_primary_delta(
    scratch: Mapping[str, object],
    variant: Mapping[str, object],
    *,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    if isinstance(replicates, bool) or replicates <= 0:
        raise Phase9CContractError("phase9c.bootstrap.replicates_invalid")
    left = scratch.get("entry_predictions")
    right = variant.get("entry_predictions")
    if not isinstance(left, list) or not isinstance(right, list):
        raise Phase9CContractError("phase9c.bootstrap.predictions_missing")
    left_by_id = {_entry_identity(row): row for row in left}
    right_by_id = {_entry_identity(row): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise Phase9CContractError("phase9c.bootstrap.pairing_mismatch")
    by_component_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    class_counts: dict[str, int] = {}
    for key in sorted(left_by_id):
        lrow, rrow = left_by_id[key], right_by_id[key]
        component = str(lrow["component_fingerprint"])
        if component != rrow["component_fingerprint"]:
            raise Phase9CContractError("phase9c.bootstrap.component_mismatch")
        task_id = str(lrow["task_id"])
        label = int(lrow["label"])
        lprobs = lrow["log_probabilities"]
        rprobs = rrow["log_probabilities"]
        if len(lprobs) != len(rprobs) or len(lprobs) <= 1:
            raise Phase9CContractError("phase9c.bootstrap.class_count_mismatch")
        class_counts.setdefault(task_id, len(lprobs))
        if class_counts[task_id] != len(lprobs):
            raise Phase9CContractError("phase9c.bootstrap.class_count_changed")
        delta = (-float(rprobs[label]) + float(lprobs[label])) / math.log(len(lprobs))
        by_component_task[(component, task_id)].append(delta)
    components = sorted({key[0] for key in by_component_task})
    if not components:
        raise Phase9CContractError("phase9c.bootstrap.components_empty")

    def statistic(sampled: Sequence[str]) -> float:
        task_values: dict[str, list[float]] = defaultdict(list)
        for component in sampled:
            for task_id in TASK_IDS:
                rows = by_component_task.get((component, task_id))
                if rows:
                    task_values[task_id].extend(rows)
        if set(task_values) != set(TASK_IDS):
            raise Phase9CContractError("phase9c.bootstrap.task_unavailable")
        return mean(mean(task_values[task]) for task in TASK_IDS)

    observed = statistic(components)
    generator = random.Random(seed)
    draws = sorted(
        statistic([generator.choice(components) for _ in components])
        for _ in range(replicates)
    )
    payload = {
        "contract_version": BOOTSTRAP_CONTRACT_VERSION,
        "unit": "component",
        "metric": "primary_normalized_nll_delta_variant_minus_scratch",
        "lower_is_better": True,
        "seed": seed,
        "replicates": replicates,
        "component_count": len(components),
        "observed_delta": observed,
        "confidence_interval_95": [
            draws[int(0.025 * (replicates - 1))],
            draws[int(0.975 * (replicates - 1))],
        ],
        "interpretation": (
            "validation-sample uncertainty only; optimization-seed uncertainty "
            "is not measured"
        ),
    }
    return {**payload, "fingerprint": fingerprint(payload)}


__all__ = [
    "BOOTSTRAP_CONTRACT_VERSION",
    "component_bootstrap_primary_delta",
    "primary_validation_summary",
]
