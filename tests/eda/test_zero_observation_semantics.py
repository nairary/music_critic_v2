from __future__ import annotations

from dataclasses import replace

import pytest

from music_critic.eda import ComputationStatus, CorpusId, EDAContractError


def test_unobserved_unknown_population_does_not_become_a_zero_count(
    raw_reports,
) -> None:
    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == "accepted_records"
    )
    coverage = replace(
        metric.coverage,
        denominator=10,
        observed_count=0,
        unknown_count=10,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
    )
    fabricated_zero = replace(metric.count, value=0, denominator=10)
    with pytest.raises(EDAContractError, match="empty_summary"):
        replace(metric, coverage=coverage, count=fabricated_zero)

    unsummarized = replace(metric, coverage=coverage, count=None)
    assert unsummarized.count is None
    assert unsummarized.coverage.unknown_count == 10
