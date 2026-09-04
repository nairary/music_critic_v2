from __future__ import annotations

from dataclasses import replace
import json

import pytest

from music_critic.eda import (
    CompletenessStatus,
    ComputationStatus,
    CorpusId,
    EDAContractError,
    EDAReasonCode,
    EvidenceScope,
    ExtensionRow,
    GraphEvidence,
    MetricCoverage,
    ObservationUnit,
    RawCorpusEDA,
    SourceExtension,
    SplitScope,
    SupervisionEDA,
    TestTargetLockEvidence as TargetLockEvidence,
    UnavailableReason,
    UnitCount,
    dumps_report,
    loads_report,
    report_dict,
)


PROVENANCE = ("source-extension-fixture",)


def _coverage(
    *,
    unit: ObservationUnit = ObservationUnit.RECORD,
    denominator: int | None = 2,
    observed_count: int | None = 2,
    unknown_count: int | None = 0,
    split: SplitScope = SplitScope.UNSPLIT,
    evidence: EvidenceScope = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = PROVENANCE,
    status: ComputationStatus = ComputationStatus.OBSERVED,
    reason_code: str | None = None,
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=unit,
        denominator=denominator,
        observed_count=observed_count,
        unknown_count=unknown_count,
        split_scope=split,
        evidence_scope=evidence,
        provenance=provenance,
        status=status,
        reason_code=reason_code,
    )


def _extension(
    row: ExtensionRow,
    *,
    corpus: CorpusId = CorpusId.PDMX,
    namespace: str = "pdmx.source_metric",
    split: SplitScope = SplitScope.UNSPLIT,
    evidence: EvidenceScope = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = PROVENANCE,
    target_free: bool = True,
) -> SourceExtension:
    return SourceExtension(
        corpus=corpus,
        namespace=namespace,
        schema_name="SourceMetric",
        schema_version="1.0.0",
        split_scope=split,
        evidence_scope=evidence,
        provenance=provenance,
        rows=(row,),
        target_free=target_free,
    )


def test_extension_row_requires_keyword_only_metric_coverage() -> None:
    with pytest.raises(TypeError, match="coverage"):
        ExtensionRow("tempo", {"source_tempo_mean": 100.0})  # type: ignore[call-arg]

    row = ExtensionRow(
        "tempo",
        {"source_tempo_mean": 100.0},
        coverage=_coverage(denominator=3, observed_count=2, unknown_count=1),
    )
    assert row.coverage.observation_unit == ObservationUnit.RECORD
    assert row.coverage.denominator == 3
    assert row.coverage.observed_count == 2
    assert row.coverage.unknown_count == 1


@pytest.mark.parametrize(
    "payload",
    (
        {"source_tempo_mean": 120.0},
        {"source_tempo": {"minimum": 80, "maximum": 120, "mean": 100}},
        {"source_mode_distribution": {"major": 0.6, "minor": 0.4}},
        {"source_modes": ["major", "minor", "major"]},
    ),
)
def test_extension_aggregate_payload_is_bound_to_explicit_coverage(
    raw_reports,
    payload,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = _extension(
        ExtensionRow("source_metric", payload, coverage=_coverage())
    )
    report = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    restored = loads_report(dumps_report(report))
    serialized_row = report_dict(restored)["semantic_payload"]["extensions"][0][
        "rows"
    ][0]
    assert serialized_row["coverage"] == {
        "denominator": 2,
        "evidence_scope": "fixture",
        "observation_unit": "record",
        "observed_count": 2,
        "provenance": ["source-extension-fixture"],
        "reason_code": None,
        "split_scope": "unsplit",
        "status": "observed",
        "unknown_count": 0,
    }


def test_extension_integer_category_map_requires_typed_counts() -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            "source_modes",
            {"source_modes": {"major": 6, "minor": 4}},
            coverage=_coverage(denominator=10, observed_count=10),
        )


def test_extension_count_must_bind_coverage_and_observed_population() -> None:
    coverage = _coverage(denominator=3, observed_count=2, unknown_count=1)
    count = UnitCount(
        name="source_rows",
        observation_unit=ObservationUnit.RECORD,
        value=2,
        denominator=3,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=PROVENANCE,
    )
    ExtensionRow("source_rows", {}, (count,), coverage=coverage)

    with pytest.raises(EDAContractError, match="count_coverage_mismatch"):
        ExtensionRow(
            "source_rows",
            {},
            (replace(count, denominator=2),),
            coverage=coverage,
        )
    with pytest.raises(EDAContractError, match="count_coverage_mismatch"):
        ExtensionRow(
            "source_rows",
            {},
            (replace(count, provenance=("other",)),),
            coverage=coverage,
        )
    with pytest.raises(EDAContractError, match="count_observed_exceeded"):
        ExtensionRow(
            "source_rows",
            {},
            (replace(count, value=3),),
            coverage=coverage,
        )


def test_non_observed_and_zero_observation_extension_metrics_do_not_fabricate_summaries() -> None:
    unavailable = _coverage(
        denominator=2,
        observed_count=None,
        unknown_count=None,
        status=ComputationStatus.NOT_COMPUTED,
        reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
    )
    ExtensionRow("not_computed", {}, coverage=unavailable)
    with pytest.raises(EDAContractError, match="unavailable_summary"):
        ExtensionRow("not_computed", {"mean": 0}, coverage=unavailable)

    unknown_population = _coverage(
        denominator=2,
        observed_count=0,
        unknown_count=2,
    )
    ExtensionRow("unknown_population", {}, coverage=unknown_population)
    with pytest.raises(EDAContractError, match="empty_summary"):
        ExtensionRow("unknown_population", {"mean": 0}, coverage=unknown_population)

    known_empty = _coverage(
        denominator=0,
        observed_count=0,
        unknown_count=0,
    )
    zero = UnitCount(
        name="source_rows",
        observation_unit=ObservationUnit.RECORD,
        value=0,
        denominator=0,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=PROVENANCE,
    )
    ExtensionRow("known_empty", {}, (zero,), coverage=known_empty)
    with pytest.raises(EDAContractError, match="empty_summary"):
        ExtensionRow(
            "unknown_population",
            {},
            (replace(zero, denominator=2),),
            coverage=unknown_population,
        )


def test_extension_coverage_binds_source_extension_scope_and_provenance() -> None:
    row = ExtensionRow("metric", {"value": "source"}, coverage=_coverage())
    with pytest.raises(EDAContractError, match="coverage_scope_mismatch"):
        _extension(row, split=SplitScope.TRAIN)
    with pytest.raises(EDAContractError, match="coverage_scope_mismatch"):
        _extension(row, provenance=("other",))


def test_known_work_coverage_requires_versioned_work_identity() -> None:
    row = ExtensionRow(
        "work_metric",
        {"source_summary": "known"},
        coverage=_coverage(unit=ObservationUnit.LOGICAL_WORK),
    )
    with pytest.raises(EDAContractError, match="work_identity_unproven"):
        _extension(row)


def test_extension_coverage_unit_is_part_of_envelope_used_units(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    row = ExtensionRow(
        "source_file_metric",
        {"source_summary": "known"},
        coverage=_coverage(unit=ObservationUnit.SOURCE_FILE),
    )
    extension = _extension(row)
    with pytest.raises(EDAContractError, match="observation_units.mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            observation_units=(
                *source.envelope.observation_units,
                ObservationUnit.SOURCE_FILE,
            ),
        ),
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    assert ObservationUnit.SOURCE_FILE in report.envelope.observation_units


def test_extension_coverage_is_required_by_strict_deserializer(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = _extension(
        ExtensionRow("source_metric", {"value": "source"}, coverage=_coverage())
    )
    report = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    payload = report_dict(report)
    del payload["semantic_payload"]["extensions"][0]["rows"][0]["coverage"]
    with pytest.raises(EDAContractError, match="fields_invalid"):
        loads_report(json.dumps(payload))


def test_extension_row_schema_is_stable_across_splits(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]

    def extension_for(
        split: SplitScope,
        *,
        coverage_unit: ObservationUnit = ObservationUnit.RECORD,
        count_unit: ObservationUnit = ObservationUnit.EVENT,
    ) -> SourceExtension:
        coverage = _coverage(unit=coverage_unit, split=split)
        count = UnitCount(
            name="events_seen",
            observation_unit=count_unit,
            value=1,
            denominator=2,
            denominator_unit=coverage_unit,
            split_scope=split,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=PROVENANCE,
        )
        return _extension(
            ExtensionRow("event_metric", {}, (count,), coverage=coverage),
            split=split,
        )

    train = extension_for(SplitScope.TRAIN)
    validation_unit_change = extension_for(
        SplitScope.VALIDATION,
        coverage_unit=ObservationUnit.SOURCE_FILE,
    )
    with pytest.raises(EDAContractError, match="schema_mismatch"):
        replace(
            source.semantic_payload,
            extensions=(train, validation_unit_change),
        )

    validation_count_change = extension_for(
        SplitScope.VALIDATION,
        count_unit=ObservationUnit.NOTE,
    )
    with pytest.raises(EDAContractError, match="schema_mismatch"):
        replace(
            source.semantic_payload,
            extensions=(train, validation_count_change),
        )


def test_non_computed_raw_report_may_enumerate_non_observed_extension_metric(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    metrics = tuple(
        replace(
            metric,
            coverage=replace(
                metric.coverage,
                status=ComputationStatus.NOT_COMPUTED,
                observed_count=None,
                unknown_count=None,
                reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
            ),
            count=None,
            numeric=None,
            categories=(),
        )
        for metric in source.semantic_payload.metrics
    )
    row = ExtensionRow(
        "source_metric",
        {},
        coverage=_coverage(
            observed_count=None,
            unknown_count=None,
            status=ComputationStatus.NOT_COMPUTED,
            reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
        ),
    )
    extension = _extension(row)
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            completeness_status=CompletenessStatus.NOT_COMPUTED,
            unavailable_reasons=(
                UnavailableReason(
                    code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
                    status=ComputationStatus.NOT_COMPUTED,
                    provenance=PROVENANCE,
                ),
            ),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            metrics=metrics,
            graph_evidence=GraphEvidence(
                status=ComputationStatus.NOT_COMPUTED,
                target_free=None,
                reason_code="eda.graph_not_computed",
            ),
            extensions=(extension,),
        ),
    )
    assert report.semantic_payload.extensions[0].rows[0].coverage.status == (
        ComputationStatus.NOT_COMPUTED
    )


def test_non_computed_supervision_report_does_not_require_observed_test_guard(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    tasks = tuple(
        replace(
            task,
            status=ComputationStatus.NOT_COMPUTED,
            availability=None,
            class_support=(),
            empty_multilabel_available_count=None,
            projection_availability=(),
            projections=(),
            reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
        )
        for task in source.semantic_payload.tasks
    )
    existing_lock = source.semantic_payload.test_lock
    locked = TargetLockEvidence.not_executed(
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=existing_lock.test_assignment_count.provenance,
        reason_code=EDAReasonCode.TEST_TARGETS_LOCKED.value,
        assignment_manifest_fingerprint=existing_lock.assignment_manifest_fingerprint,
        test_assignment_denominator=existing_lock.test_assignment_count.denominator,
    )
    extension_provenance = source.semantic_payload.extensions[0].provenance
    row = ExtensionRow(
        "source_metric",
        {},
        coverage=_coverage(
            unit=ObservationUnit.TARGET_ROW,
            denominator=1,
            observed_count=None,
            unknown_count=None,
            split=SplitScope.TRAIN,
            provenance=extension_provenance,
            status=ComputationStatus.NOT_COMPUTED,
            reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
        ),
    )
    extension = replace(source.semantic_payload.extensions[0], rows=(row,))
    report = SupervisionEDA(
        envelope=replace(
            source.envelope,
            completeness_status=CompletenessStatus.NOT_COMPUTED,
            observation_units=(
                ObservationUnit.SPLIT_ASSIGNMENT,
                ObservationUnit.TARGET_ACCESS_ATTEMPT,
                ObservationUnit.RECORD,
                ObservationUnit.TARGET_ROW,
            ),
            unavailable_reasons=(
                UnavailableReason(
                    code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
                    status=ComputationStatus.NOT_COMPUTED,
                    provenance=PROVENANCE,
                ),
            ),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            tasks=tasks,
            test_lock=locked,
            extensions=(extension,),
        ),
    )
    assert report.semantic_payload.test_lock.test_assignment_count.status == (
        ComputationStatus.LOCKED
    )
