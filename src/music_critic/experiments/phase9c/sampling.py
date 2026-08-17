"""Target-blind deterministic source-balanced schedules for Phase 9C-A."""

from __future__ import annotations

from collections import Counter
import random
from typing import Mapping, Sequence

from music_critic.experiments.phase8b2.schedule import derive_seed

from .contracts import Phase9CContractError, fingerprint


SAMPLER_CONTRACT_VERSION = "1.0.0"


def _validate_weights(weights: Mapping[str, float], sources: Sequence[str]) -> None:
    if set(weights) != set(sources) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in weights.values()
    ):
        raise Phase9CContractError("phase9c.sampler.weights_invalid")


def build_source_balanced_schedule(
    identities: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float],
    sample_count: int,
    seed: int,
) -> dict[str, object]:
    """Build weighted source choices with deterministic shuffled no-replacement cycles."""

    sources = tuple(sorted(identities))
    if not sources or isinstance(sample_count, bool) or sample_count <= 0:
        raise Phase9CContractError("phase9c.sampler.arguments_invalid")
    _validate_weights(weights, sources)
    normalized = {source: float(weights[source]) / sum(weights.values()) for source in sources}
    for source in sources:
        rows = tuple(identities[source])
        if not rows or len(rows) != len(set(rows)) or any(
            not isinstance(row, str) or not row for row in rows
        ):
            raise Phase9CContractError("phase9c.sampler.identities_invalid")

    cycles = {source: 0 for source in sources}
    positions = {source: 0 for source in sources}
    shuffled: dict[str, list[str]] = {}

    def refill(source: str) -> None:
        rows = list(identities[source])
        domain_seed = derive_seed(seed, "phase9c/source_cycle", source, cycles[source])
        random.Random(domain_seed).shuffle(rows)
        shuffled[source] = rows
        positions[source] = 0
        cycles[source] += 1

    for source in sources:
        refill(source)
    counts = Counter()
    unique: dict[str, set[str]] = {source: set() for source in sources}
    rows: list[dict[str, object]] = []
    for index in range(sample_count):
        # Weighted fair queue: choose the source furthest below its ideal prefix.
        source = min(
            sources,
            key=lambda name: (
                counts[name] - normalized[name] * (index + 1),
                derive_seed(seed, "phase9c/source_tie", index, name),
                name,
            ),
        )
        if positions[source] == len(shuffled[source]):
            refill(source)
        piece_id = shuffled[source][positions[source]]
        cycle_index = cycles[source] - 1
        positions[source] += 1
        counts[source] += 1
        unique[source].add(piece_id)
        rows.append(
            {
                "position": index,
                "dataset_id": source,
                "piece_id": piece_id,
                "cycle_index": cycle_index,
            }
        )
    projection = {
        "contract_version": SAMPLER_CONTRACT_VERSION,
        "seed": seed,
        "weights": [[source, float(weights[source])] for source in sources],
        "normalized_weights": [[source, normalized[source]] for source in sources],
        "slots": rows,
        "dataset_counts": {source: counts[source] for source in sources},
        "unique_record_counts": {source: len(unique[source]) for source in sources},
        "repeat_counts": {
            source: counts[source] - len(unique[source]) for source in sources
        },
        "completed_or_entered_cycle_counts": {source: cycles[source] for source in sources},
        "replacement_within_cycle": False,
        "target_or_provenance_access": False,
    }
    return {**projection, "fingerprint": fingerprint(projection)}


__all__ = ["SAMPLER_CONTRACT_VERSION", "build_source_balanced_schedule"]
