from __future__ import annotations

from dataclasses import replace

import pytest

from music_critic.eda import (
    APPROVED_RAW_GRAPH_CONTRACT,
    CategoryCount,
    ComputationStatus,
    CorpusId,
    EDAContractError,
    GraphEvidence,
    NumericDistribution,
    ObservationUnit,
    RawCorpusEDA,
    UnitCount,
)


def _raw_report_with_one_observed_graph(
    raw_reports,
    *,
    node_total: int,
    reported_size: int,
) -> RawCorpusEDA:
    source = raw_reports[CorpusId.PDMX]
    graph_metric_ids = {
        "graph_node_counts",
        "graph_edge_counts",
        "graph_size_distribution",
    }
    graph_metrics = {
        metric.metric_id: metric
        for metric in source.semantic_payload.metrics
        if metric.metric_id in graph_metric_ids
    }
    coverage = replace(
        graph_metrics["graph_node_counts"].coverage,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
        provenance=("exact-graph-fixture",),
    )
    node_count = UnitCount(
        name="graph_node_counts",
        observation_unit=ObservationUnit.GRAPH_NODE,
        value=node_total,
        denominator=1,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=coverage.split_scope,
        evidence_scope=coverage.evidence_scope,
        provenance=coverage.provenance,
    )
    replacements = {
        "graph_node_counts": replace(
            graph_metrics["graph_node_counts"],
            coverage=coverage,
            categories=(CategoryCount("native", node_count),),
        ),
        "graph_edge_counts": replace(
            graph_metrics["graph_edge_counts"],
            coverage=coverage,
            categories=(),
        ),
        "graph_size_distribution": replace(
            graph_metrics["graph_size_distribution"],
            coverage=coverage,
            numeric=NumericDistribution(
                measurement_unit="nodes_plus_edges_per_record",
                minimum=reported_size,
                maximum=reported_size,
                mean=reported_size,
            ),
        ),
    }
    metrics = tuple(
        replacements.get(metric.metric_id, metric)
        for metric in source.semantic_payload.metrics
    )
    return RawCorpusEDA(
        envelope=replace(
            source.envelope,
            observation_units=(
                ObservationUnit.RECORD,
                ObservationUnit.GRAPH_NODE,
            ),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            metrics=metrics,
            graph_evidence=GraphEvidence(
                status=ComputationStatus.OBSERVED,
                target_free=True,
                **dict(APPROVED_RAW_GRAPH_CONTRACT),
            ),
        ),
    )


def test_graph_mean_does_not_lose_one_above_float_exact_integer_range(
    raw_reports,
) -> None:
    with pytest.raises(EDAContractError, match="size_mean_mismatch"):
        _raw_report_with_one_observed_graph(
            raw_reports,
            node_total=2**53,
            reported_size=2**53 + 1,
        )


def test_graph_mean_accepts_exact_arbitrarily_large_integer(raw_reports) -> None:
    huge_exact_total = 10**400
    report = _raw_report_with_one_observed_graph(
        raw_reports,
        node_total=huge_exact_total,
        reported_size=huge_exact_total,
    )
    size_metric = next(
        metric
        for metric in report.semantic_payload.metrics
        if metric.metric_id == "graph_size_distribution"
    )
    assert size_metric.numeric is not None
    assert size_metric.numeric.mean == huge_exact_total
