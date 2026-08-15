"""Deterministic paired piece-level summaries for repeated seeds."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import mean, median, stdev
from typing import Iterable

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
    cells: dict[tuple[str, str, str, str], list[PieceMetric]] = defaultdict(list)
    for row in rows:
        cells[(row.dataset_id, row.variant_id, row.transfer_mode, row.metric_id)].append(row)
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


__all__ = [
    "PieceMetric",
    "aggregate_paired_piece_metrics",
    "paired_piece_bootstrap",
]
