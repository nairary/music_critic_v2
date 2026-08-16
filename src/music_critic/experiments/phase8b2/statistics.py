"""Deterministic paired piece-level summaries for repeated seeds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Any

from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_STATISTICS_CONTRACT_VERSION,
    Phase8B2ContractError,
    fingerprint,
)
from music_critic.experiments.phase8b2.schedule import derive_seed


@dataclass(frozen=True, slots=True)
class PieceMetric:
    dataset_id: str
    piece_id: str
    seed: int
    variant_id: str
    transfer_mode: str
    metric_id: str
    value: float

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.dataset_id,
                self.piece_id,
                self.variant_id,
                self.transfer_mode,
                self.metric_id,
            )
        ) or isinstance(self.seed, bool) or not isinstance(self.seed, int) or (
            not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
            or not math.isfinite(self.value)
        ):
            raise Phase8B2ContractError(
                "phase8b2.statistics.piece_metric_invalid"
            )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    coordinate = (len(ordered) - 1) * probability
    low = int(math.floor(coordinate))
    high = int(math.ceil(coordinate))
    if low == high:
        return ordered[low]
    weight = coordinate - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def paired_piece_bootstrap(
    differences: dict[str, float],
    *,
    bootstrap_seed: int,
    replicates: int,
) -> dict[str, object]:
    """Bootstrap independent piece IDs, never target rows."""

    if len(differences) < 2:
        return {
            "available": False,
            "unavailable": {
                "category": "insufficient_independent_pieces",
                "reason": "paired piece-level bootstrap requires at least two pieces",
            },
            "piece_count": len(differences),
            "confidence_interval_95": None,
        }
    if replicates < 2:
        raise Phase8B2ContractError(
            "phase8b2.statistics.bootstrap_replicates_invalid"
        )
    piece_ids = sorted(differences)
    generator = random.Random(bootstrap_seed)
    distribution = []
    for _ in range(replicates):
        sampled = [
            differences[piece_ids[generator.randrange(len(piece_ids))]]
            for _ in piece_ids
        ]
        distribution.append(mean(sampled))
    return {
        "available": True,
        "unavailable": None,
        "piece_count": len(piece_ids),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": replicates,
        "point_estimate_mean_delta": mean(differences.values()),
        "confidence_interval_95": {
            "lower": _quantile(distribution, 0.025),
            "upper": _quantile(distribution, 0.975),
        },
        "retained_bootstrap_draws": 0,
    }


def aggregate_paired_piece_metrics(
    records: Iterable[PieceMetric],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    minimum_scientific_seeds: int = 3,
) -> dict[str, object]:
    """Aggregate cells and paired deltas against scratch and Phase 7A."""

    rows = tuple(records)
    if not rows:
        raise Phase8B2ContractError(
            "phase8b2.statistics.records_empty"
        )
    keys = [
        (
            row.dataset_id,
            row.piece_id,
            row.seed,
            row.variant_id,
            row.transfer_mode,
            row.metric_id,
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise Phase8B2ContractError(
            "phase8b2.statistics.duplicate_piece_metric"
        )
    cells: dict[
        tuple[str, str, str, str], list[PieceMetric]
    ] = defaultdict(list)
    for row in rows:
        cells[
            (
                row.dataset_id,
                row.variant_id,
                row.transfer_mode,
                row.metric_id,
            )
        ].append(row)
    cell_summaries = []
    for key in sorted(cells):
        dataset_id, variant_id, transfer_mode, metric_id = key
        by_seed: dict[int, list[float]] = defaultdict(list)
        for row in cells[key]:
            by_seed[row.seed].append(float(row.value))
        seed_rows = [
            {
                "seed": seed,
                "piece_count": len(values),
                "mean": mean(values),
                "median": median(values),
            }
            for seed, values in sorted(by_seed.items())
        ]
        seed_means = [float(row["mean"]) for row in seed_rows]
        scientific_seed_minimum_met = (
            len(seed_rows) >= minimum_scientific_seeds
        )
        cell_summaries.append(
            {
                "dataset_id": dataset_id,
                "variant_id": variant_id,
                "transfer_mode": transfer_mode,
                "metric_id": metric_id,
                "seed_count": len(seed_rows),
                "per_seed": seed_rows,
                "mean_of_seed_means": mean(seed_means),
                "median_of_seed_means": median(seed_means),
                "between_seed_standard_deviation": (
                    stdev(seed_means) if len(seed_means) >= 2 else None
                ),
                "between_seed_sd_unavailable": (
                    None
                    if len(seed_means) >= 2
                    else {
                        "category": "insufficient_seeds",
                        "reason": "between-seed SD requires at least two seeds",
                    }
                ),
                "scientific_seed_minimum_met": scientific_seed_minimum_met,
                "scientific_seed_unavailable": (
                    None
                    if scientific_seed_minimum_met
                    else {
                        "category": "insufficient_seeds",
                        "reason": (
                            "scientific summaries require at least "
                            f"{minimum_scientific_seeds} paired seeds"
                        ),
                    }
                ),
            }
        )
    lookup = {
        (
            row.dataset_id,
            row.piece_id,
            row.seed,
            row.variant_id,
            row.transfer_mode,
            row.metric_id,
        ): float(row.value)
        for row in rows
    }
    comparisons = []
    candidates = sorted(
        {
            (row.dataset_id, row.variant_id, row.transfer_mode, row.metric_id)
            for row in rows
        }
    )
    for dataset_id, variant_id, transfer_mode, metric_id in candidates:
        references = (
            ("supervised_scratch", "supervised_scratch"),
            ("phase7a_control", transfer_mode),
        )
        for reference_variant, reference_mode in references:
            if variant_id == reference_variant and transfer_mode == reference_mode:
                continue
            current = {
                (row.seed, row.piece_id): float(row.value)
                for row in rows
                if (
                    row.dataset_id,
                    row.variant_id,
                    row.transfer_mode,
                    row.metric_id,
                )
                == (dataset_id, variant_id, transfer_mode, metric_id)
            }
            paired = {
                identity: value
                - lookup[
                    (
                        dataset_id,
                        identity[1],
                        identity[0],
                        reference_variant,
                        reference_mode,
                        metric_id,
                    )
                ]
                for identity, value in current.items()
                if (
                    dataset_id,
                    identity[1],
                    identity[0],
                    reference_variant,
                    reference_mode,
                    metric_id,
                )
                in lookup
            }
            piece_differences: dict[str, list[float]] = defaultdict(list)
            for (_, piece_id), value in paired.items():
                piece_differences[piece_id].append(value)
            collapsed = {
                piece_id: mean(values)
                for piece_id, values in piece_differences.items()
            }
            seed = derive_seed(
                bootstrap_seed,
                "paired_piece_bootstrap",
                dataset_id,
                variant_id,
                transfer_mode,
                metric_id,
                reference_variant,
                reference_mode,
            )
            paired_seed_count = len({identity[0] for identity in paired})
            seed_minimum_met = paired_seed_count >= minimum_scientific_seeds
            bootstrap = paired_piece_bootstrap(
                collapsed,
                bootstrap_seed=seed,
                replicates=bootstrap_replicates,
            )
            scientific_unavailable = []
            if not seed_minimum_met:
                scientific_unavailable.append(
                    {
                        "category": "insufficient_seeds",
                        "reason": (
                            "paired comparison requires at least "
                            f"{minimum_scientific_seeds} seeds"
                        ),
                    }
                )
            if not bootstrap["available"]:
                scientific_unavailable.append(bootstrap["unavailable"])
            comparisons.append(
                {
                    "dataset_id": dataset_id,
                    "variant_id": variant_id,
                    "transfer_mode": transfer_mode,
                    "metric_id": metric_id,
                    "reference_variant_id": reference_variant,
                    "reference_transfer_mode": reference_mode,
                    "paired_seed_piece_count": len(paired),
                    "paired_seed_count": paired_seed_count,
                    "scientific_seed_minimum_met": seed_minimum_met,
                    "scientific_summary_available": not scientific_unavailable,
                    "scientific_unavailable": scientific_unavailable or None,
                    "bootstrap": bootstrap,
                }
            )
    artifact = {
        "statistics_contract_version": PHASE8B2_STATISTICS_CONTRACT_VERSION,
        "statistical_unit": "independent_piece",
        "row_level_resampling_forbidden": True,
        "minimum_scientific_seeds": minimum_scientific_seeds,
        "bounded_results_are_scientific_evidence": False,
        "cell_summaries": cell_summaries,
        "paired_comparisons": comparisons,
        "significance_tests": {
            "performed": False,
            "reason": "Phase 8B.2A bounded acceptance does not claim significance",
        },
    }
    artifact["fingerprint"] = fingerprint(artifact)
    return artifact


def _merge_sufficient(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    if not values:
        raise Phase8B2ContractError(
            "phase8b2.statistics.sufficient_rows_empty"
        )
    kind = values[0].get("kind")
    labels = values[0].get("class_labels")
    if any(
        row.get("kind") != kind or row.get("class_labels") != labels
        for row in values
    ):
        raise Phase8B2ContractError(
            "phase8b2.statistics.sufficient_identity_mismatch"
        )
    class_count = len(labels)
    if kind == "closed_categorical":
        confusion = [[0] * class_count for _ in range(class_count)]
        nll_sum = 0.0
        eligible = 0
        for row in values:
            for truth in range(class_count):
                for predicted in range(class_count):
                    confusion[truth][predicted] += int(
                        row["confusion_counts"][truth][predicted]
                    )
            nll_sum += float(row["nll_sum"])
            eligible += int(row["eligible_count"])
        f1_values = []
        for index in range(class_count):
            tp = confusion[index][index]
            fp = sum(row[index] for row in confusion) - tp
            fn = sum(confusion[index]) - tp
            denominator = 2 * tp + fp + fn
            if denominator:
                f1_values.append(2 * tp / denominator)
        return {
            "kind": kind,
            "macro_f1": None if not f1_values else mean(f1_values),
            "nll": None if eligible == 0 else nll_sum / eligible,
            "eligible_count": eligible,
        }
    if kind == "closed_multilabel":
        tp = [0] * class_count
        fp = [0] * class_count
        fn = [0] * class_count
        bce_sum = 0.0
        eligible_labels = 0
        for row in values:
            for index in range(class_count):
                tp[index] += int(row["tp"][index])
                fp[index] += int(row["fp"][index])
                fn[index] += int(row["fn"][index])
            bce_sum += float(row["bce_nll_sum"])
            eligible_labels += int(row["eligible_label_count"])
        f1_values = []
        for index in range(class_count):
            denominator = 2 * tp[index] + fp[index] + fn[index]
            if denominator:
                f1_values.append(2 * tp[index] / denominator)
        return {
            "kind": kind,
            "macro_f1": None if not f1_values else mean(f1_values),
            "nll": (
                None
                if eligible_labels == 0
                else bce_sum / eligible_labels
            ),
            "eligible_count": eligible_labels,
        }
    raise Phase8B2ContractError(
        "phase8b2.statistics.sufficient_kind_invalid"
    )


def _dataset_endpoint(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row["statistics"])
    metrics = [_merge_sufficient(values) for values in by_task.values()]
    f1 = [float(row["macro_f1"]) for row in metrics if row["macro_f1"] is not None]
    nll = [float(row["nll"]) for row in metrics if row["nll"] is not None]
    if not f1 or not nll:
        raise Phase8B2ContractError(
            "phase8b2.statistics.dataset_endpoint_unavailable"
        )
    return {"macro_f1": mean(f1), "nll": mean(nll)}


def aggregate_piece_sufficient_statistics(
    records: Iterable[Mapping[str, Any]],
    *,
    declared_seeds: Iterable[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
    minimum_scientific_seeds: int = 3,
) -> dict[str, object]:
    """Recompute corpus endpoints after every independent-piece resample."""

    rows = [dict(row) for row in records]
    seeds = tuple(sorted(declared_seeds))
    if not rows or not seeds or bootstrap_replicates < 2:
        raise Phase8B2ContractError(
            "phase8b2.statistics.sufficient_input_invalid"
        )
    identities = [
        (
            row.get("dataset_id"),
            row.get("piece_id"),
            row.get("seed"),
            row.get("variant_id"),
            row.get("transfer_mode"),
            row.get("task_id"),
            row.get("encoding_kind"),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise Phase8B2ContractError(
            "phase8b2.statistics.duplicate_piece_sufficient_row"
        )
    configurations = sorted(
        {
            (str(row["variant_id"]), str(row["transfer_mode"]))
            for row in rows
        }
    )
    datasets = sorted({str(row["dataset_id"]) for row in rows})
    cell_summaries = []
    for variant_id, transfer_mode in configurations:
        per_seed = []
        for seed in seeds:
            endpoints = {}
            for dataset_id in datasets:
                selected = [
                    row
                    for row in rows
                    if (
                        row["variant_id"],
                        row["transfer_mode"],
                        row["seed"],
                        row["dataset_id"],
                    )
                    == (variant_id, transfer_mode, seed, dataset_id)
                ]
                if selected:
                    endpoints[dataset_id] = _dataset_endpoint(selected)
            if set(endpoints) != set(datasets):
                raise Phase8B2ContractError(
                    "phase8b2.statistics.paired_seed_evidence_incomplete"
                )
            per_seed.append({"seed": seed, "dataset_endpoints": endpoints})
        cell_summaries.append(
            {
                "configuration_id": f"{variant_id}/{transfer_mode}",
                "variant_id": variant_id,
                "transfer_mode": transfer_mode,
                "seed_count": len(per_seed),
                "per_seed": per_seed,
                "aggregated_dataset_endpoints": {
                    dataset_id: {
                        "macro_f1": mean(
                            row["dataset_endpoints"][dataset_id]["macro_f1"]
                            for row in per_seed
                        ),
                        "nll": mean(
                            row["dataset_endpoints"][dataset_id]["nll"]
                            for row in per_seed
                        ),
                    }
                    for dataset_id in datasets
                },
            }
        )
    comparisons = []
    lookup = {
        (str(row["variant_id"]), str(row["transfer_mode"])): row
        for row in cell_summaries
    }
    for variant_id, transfer_mode in configurations:
        references = [
            ("supervised_scratch", "supervised_scratch"),
            ("phase7a_control", transfer_mode),
        ]
        for reference in references:
            if reference == (variant_id, transfer_mode) or reference not in lookup:
                continue
            for dataset_id in datasets:
                generator = random.Random(
                    derive_seed(
                        bootstrap_seed,
                        "piece_sufficient_bootstrap",
                        variant_id,
                        transfer_mode,
                        reference[0],
                        reference[1],
                        dataset_id,
                    )
                )
                distribution = []
                point_deltas = []
                common_piece_count = 0
                paired_rows: list[
                    tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]
                ] = []
                for seed in seeds:
                    current_rows = [
                        row
                        for row in rows
                        if (
                            row["variant_id"],
                            row["transfer_mode"],
                            row["seed"],
                            row["dataset_id"],
                        )
                        == (variant_id, transfer_mode, seed, dataset_id)
                    ]
                    reference_rows = [
                        row
                        for row in rows
                        if (
                            row["variant_id"],
                            row["transfer_mode"],
                            row["seed"],
                            row["dataset_id"],
                        )
                        == (reference[0], reference[1], seed, dataset_id)
                    ]
                    pieces = sorted(
                        {str(row["piece_id"]) for row in current_rows}
                        & {str(row["piece_id"]) for row in reference_rows}
                    )
                    if len(pieces) < 2:
                        continue
                    common_piece_count = max(common_piece_count, len(pieces))
                    paired_rows.append((current_rows, reference_rows, pieces))
                    point_deltas.append(
                        _dataset_endpoint(current_rows)["macro_f1"]
                        - _dataset_endpoint(reference_rows)["macro_f1"]
                    )
                for _ in range(bootstrap_replicates):
                    seed_deltas = []
                    for current_rows, reference_rows, pieces in paired_rows:
                        sampled = [
                            pieces[generator.randrange(len(pieces))]
                            for _ in pieces
                        ]
                        current_sample = [
                            row
                            for piece_id in sampled
                            for row in current_rows
                            if row["piece_id"] == piece_id
                        ]
                        reference_sample = [
                            row
                            for piece_id in sampled
                            for row in reference_rows
                            if row["piece_id"] == piece_id
                        ]
                        seed_deltas.append(
                            _dataset_endpoint(current_sample)["macro_f1"]
                            - _dataset_endpoint(reference_sample)["macro_f1"]
                        )
                    if seed_deltas:
                        distribution.append(mean(seed_deltas))
                paired_seed_count = len(paired_rows)
                available = (
                    bool(distribution)
                    and paired_seed_count >= minimum_scientific_seeds
                )
                comparisons.append(
                    {
                        "dataset_id": dataset_id,
                        "configuration_id": f"{variant_id}/{transfer_mode}",
                        "reference_configuration_id": (
                            f"{reference[0]}/{reference[1]}"
                        ),
                        "paired_seed_count": paired_seed_count,
                        "common_piece_count": common_piece_count,
                        "available": available,
                        "unavailable": (
                            None
                            if available
                            else {
                                "category": "insufficient_pieces_or_seeds",
                                "reason": (
                                    "piece bootstrap requires at least two "
                                    "paired pieces and "
                                    f"{minimum_scientific_seeds} paired seeds"
                                ),
                            }
                        ),
                        "point_estimate_mean_delta": (
                            None if not point_deltas else mean(point_deltas)
                        ),
                        "confidence_interval_95": (
                            None
                            if not distribution
                            else {
                                "lower": _quantile(distribution, 0.025),
                                "upper": _quantile(distribution, 0.975),
                            }
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                        "retained_bootstrap_draws": 0,
                    }
                )
    artifact = {
        "statistics_contract_version": PHASE8B2_STATISTICS_CONTRACT_VERSION,
        "statistical_unit": "independent_piece",
        "endpoint_recomputed_from_sufficient_statistics_after_each_resample": True,
        "per_piece_macro_f1_averaging_forbidden": True,
        "average_precision_scope": "descriptive_corpus_metric_only",
        "declared_seeds": list(seeds),
        "minimum_scientific_seeds": minimum_scientific_seeds,
        "bounded_results_are_scientific_evidence": False,
        "cell_summaries": cell_summaries,
        "paired_comparisons": comparisons,
    }
    artifact["fingerprint"] = fingerprint(artifact)
    return artifact


__all__ = [
    "PieceMetric",
    "aggregate_paired_piece_metrics",
    "aggregate_piece_sufficient_statistics",
    "paired_piece_bootstrap",
]
