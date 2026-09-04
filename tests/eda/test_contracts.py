from __future__ import annotations

from dataclasses import replace
import math

import pytest

from music_critic.data.serialization import canonical_json_sha256
from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    AvailabilityCounts,
    ComputationStatus,
    CorpusId,
    EDA_CAPABILITIES,
    EDAContractError,
    EDAReasonCode,
    EvidenceScope,
    ExecutionMode,
    ExtensionRow,
    GraphEvidence,
    InputManifestRef,
    LabelValueType,
    MetricCoverage,
    NumericDistribution,
    ObservationUnit,
    ProjectionEvidence,
    RawCorpusEDA,
    SourceExtension,
    SourceValueIdentity,
    SourceValueKind,
    SplitScope,
    StructuredWarning,
    SupervisionEDA,
    TaskFamilyEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    capability_registry_dict,
    capability_registry_fingerprint,
    report_dict,
    sum_unit_counts,
)


def _extension_coverage(
    *,
    split_scope: SplitScope = SplitScope.UNSPLIT,
    evidence_scope: EvidenceScope = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = ("fixture",),
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=ObservationUnit.RECORD,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        split_scope=split_scope,
        evidence_scope=evidence_scope,
        provenance=provenance,
    )


def test_minimal_synthetic_raw_and_supervision_reports_cover_capability_matrix(
    raw_reports, supervision_reports
) -> None:
    assert set(raw_reports) == set(CorpusId)
    assert set(supervision_reports) == {
        CorpusId.DILEMMADATA,
        CorpusId.HOOKTHEORY,
        CorpusId.POP909_CL,
    }
    assert all(isinstance(report, RawCorpusEDA) for report in raw_reports.values())
    assert all(
        isinstance(report, SupervisionEDA) for report in supervision_reports.values()
    )
    assert {
        corpus.value: (row.raw_corpus_eda, row.supervision_eda)
        for corpus, row in EDA_CAPABILITIES.items()
    } == {
        "dilemmadata": (True, True),
        "hooktheory": (True, True),
        "pdmx": (True, False),
        "pop909_cl": (True, True),
    }
    registry = capability_registry_dict()
    assert registry["schema_name"] == "MultiSourceEDACapabilityRegistry"
    assert capability_registry_fingerprint() == canonical_json_sha256(registry)


def test_pdmx_rejects_supervision_instead_of_emitting_placeholder(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    with pytest.raises(EDAContractError, match="supervision_forbidden"):
        SupervisionEDA(
            envelope=replace(source.envelope, corpus=CorpusId.PDMX),
            semantic_payload=source.semantic_payload,
        )


def test_unknown_and_not_computed_are_null_not_zero_or_missing() -> None:
    unknown = MetricCoverage(
        observation_unit=ObservationUnit.RECORD,
        denominator=None,
        observed_count=None,
        unknown_count=None,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.UNKNOWN,
        provenance=("unknown-source",),
        status=ComputationStatus.UNKNOWN,
        reason_code="source.status_unknown",
    )
    assert unknown.status == ComputationStatus.UNKNOWN
    assert unknown.denominator is None
    assert unknown.observed_count is None
    assert unknown.unknown_count is None
    with pytest.raises(EDAContractError, match="unavailable_counts_not_null"):
        replace(unknown, observed_count=0, unknown_count=0)


def test_availability_states_are_exhaustive_and_bool_is_not_an_integer() -> None:
    with pytest.raises(EDAContractError, match="denominator_mismatch"):
        AvailabilityCounts(
            observation_unit=ObservationUnit.TARGET_ROW,
            denominator=4,
            available=1,
            masked=1,
            missing=1,
            unsupported=0,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("fixture",),
        )
    with pytest.raises(EDAContractError, match="non-negative integer"):
        UnitCount(
            name="bad-bool",
            observation_unit=ObservationUnit.RECORD,
            value=True,
            denominator=1,
            denominator_unit=ObservationUnit.RECORD,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("fixture",),
        )


def test_unit_aggregation_rejects_implicit_unit_and_scope_mixing() -> None:
    common = dict(
        value=1,
        denominator=2,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
    )
    records = UnitCount(
        name="records", observation_unit=ObservationUnit.RECORD, **common
    )
    notes = UnitCount(name="notes", observation_unit=ObservationUnit.NOTE, **common)
    with pytest.raises(EDAContractError, match="unit_or_scope_mismatch"):
        sum_unit_counts("invalid-total", (records, notes))


def test_same_surface_tokens_in_different_dialects_have_distinct_identity() -> None:
    left = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id="dilemmadata.an.chord.inversion",
        dialect="an_joint",
        source_value="2",
        value_kind=SourceValueKind.SCALAR,
    )
    right = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id="dilemmadata.dlc.chord.inversion",
        dialect="dlc",
        source_value="2",
        value_kind=SourceValueKind.SCALAR,
    )
    assert left.source_value == right.source_value
    assert left.identity != right.identity


def test_class_support_cannot_include_non_available_rows(supervision_reports) -> None:
    support = supervision_reports[CorpusId.POP909_CL].semantic_payload.tasks[0].class_support[0]
    with pytest.raises(EDAContractError, match="not_available_only"):
        replace(support, available_only=False)


def test_projection_requires_the_exact_existing_registry(supervision_reports) -> None:
    task = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[0]
    source_value = task.class_support[0].source_value
    approved = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    fake = replace(
        approved,
        fingerprint="0" * 64,
    )
    with pytest.raises(EDAContractError, match="registry_unapproved"):
        ProjectionEvidence(
            source_value=source_value,
            mapping_registry=fake,
            common_task_identity="dilemmadata.common.chord.quality",
            native_state="available",
            mapping_state="exact",
            projected_value="major",
            provenance=("fake",),
        )


def test_fixture_scope_cannot_claim_production_execution(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="evidence.execution_mismatch"):
        replace(
            source.envelope,
            evidence_scope=EvidenceScope.PRODUCTION,
            execution_mode=ExecutionMode.SYNTHETIC_FIXTURE,
        )


def test_synthetic_fixture_cannot_be_consistently_relabeled_production(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    metrics = []
    for metric in source.semantic_payload.metrics:
        count = (
            None
            if metric.count is None
            else replace(metric.count, evidence_scope=EvidenceScope.PRODUCTION)
        )
        categories = tuple(
            replace(
                category,
                count=replace(
                    category.count,
                    evidence_scope=EvidenceScope.PRODUCTION,
                ),
            )
            for category in metric.categories
        )
        metrics.append(
            replace(
                metric,
                coverage=replace(
                    metric.coverage,
                    evidence_scope=EvidenceScope.PRODUCTION,
                ),
                count=count,
                categories=categories,
            )
        )
    with pytest.raises(EDAContractError, match="fixture_as_production"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                evidence_scope=EvidenceScope.PRODUCTION,
                execution_mode=ExecutionMode.PRODUCTION_SCAN,
            ),
            semantic_payload=replace(
                source.semantic_payload,
                metrics=tuple(metrics),
            ),
        )


def test_production_attestation_preserves_source_domain_words(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    provenance = ("pdmx-production-scan",)
    metrics = []
    for metric in source.semantic_payload.metrics:
        coverage = replace(
            metric.coverage,
            evidence_scope=EvidenceScope.PRODUCTION,
            provenance=provenance,
            reason_code=(
                None
                if metric.coverage.status == ComputationStatus.OBSERVED
                else EDAReasonCode.METRIC_NOT_COMPUTED.value
            ),
        )
        count = (
            None
            if metric.count is None
            else replace(
                metric.count,
                evidence_scope=EvidenceScope.PRODUCTION,
                provenance=provenance,
            )
        )
        categories = tuple(
            replace(
                category,
                count=replace(
                    category.count,
                    evidence_scope=EvidenceScope.PRODUCTION,
                    provenance=provenance,
                ),
            )
            for category in metric.categories
        )
        metrics.append(
            replace(
                metric,
                coverage=coverage,
                count=count,
                categories=categories,
            )
        )
    manifest = replace(
        source.envelope.input_manifests[0],
        identity=VersionedIdentity("pdmx.raw_manifest", "1.0.0", "c" * 64),
        repository_relative_path="manifests/pdmx.json",
    )
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.generation_metadata",
        schema_name="GenerationMetadata",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.PRODUCTION,
        provenance=provenance,
        rows=(
            ExtensionRow(
                row_id="source-method",
                payload={"title": "Replay", "generation_method": "synthetic"},
                coverage=_extension_coverage(
                    evidence_scope=EvidenceScope.PRODUCTION,
                    provenance=provenance,
                ),
            ),
        ),
        target_free=True,
    )
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            source_identity=VersionedIdentity("pdmx.release", "1.0.0", "a" * 64),
            producer_identity=VersionedIdentity(
                "pdmx.eda_adapter", "1.0.0", "b" * 64
            ),
            evidence_scope=EvidenceScope.PRODUCTION,
            execution_mode=ExecutionMode.PRODUCTION_SCAN,
            input_manifests=(manifest,),
            unavailable_reasons=(
                UnavailableReason(
                    code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
                    status=ComputationStatus.NOT_COMPUTED,
                    provenance=provenance,
                ),
            ),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            metrics=tuple(metrics),
            graph_evidence=replace(
                source.semantic_payload.graph_evidence,
                reason_code=EDAReasonCode.METRIC_NOT_COMPUTED.value,
            ),
            extensions=(extension,),
        ),
    )
    assert report.semantic_payload.extensions[0].rows[0].payload["title"] == "Replay"
    assert (
        report.semantic_payload.extensions[0].rows[0].payload["generation_method"]
        == "synthetic"
    )


def test_same_population_count_cannot_exceed_observed_coverage(raw_reports) -> None:
    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == "invalid_records"
    )
    coverage = replace(
        metric.coverage,
        denominator=10,
        observed_count=2,
        unknown_count=8,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
    )
    count = UnitCount(
        name=metric.metric_id,
        observation_unit=ObservationUnit.RECORD,
        value=9,
        denominator=10,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=coverage.split_scope,
        evidence_scope=coverage.evidence_scope,
        provenance=coverage.provenance,
    )
    with pytest.raises(EDAContractError, match="count_coverage_exceeded"):
        replace(metric, coverage=coverage, count=count)


def test_raw_report_rejects_target_derived_extension_fields(raw_reports) -> None:
    source = raw_reports[CorpusId.HOOKTHEORY]
    extension = SourceExtension(
        corpus=CorpusId.HOOKTHEORY,
        namespace="hooktheory.bad_raw_extension",
        schema_name="BadRawExtension",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                row_id="bad",
                payload={"labels": ["major"]},
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    with pytest.raises(EDAContractError, match="semantic_field.forbidden"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )


def test_source_extensions_cannot_shadow_common_fields_or_hide_bare_counts() -> None:
    for payload in (
        {"accepted_records": 999},
        {"acceptedRecords": 999},
        {"accepted-records": 999},
        {"accepted records": 999},
        {"duration": 123},
        {"nested": {"semantic_payload": {}}},
        {"availability": {"available": 3}},
        {"class_distribution": {"major": 7}},
        {"target_availability": {"available": 7}},
        {"coverage": {"known": 7}},
        {"cooccurrence": {"major": 7}},
        {"envelope_schema_name": "shadow"},
        {"envelopeSchemaVersion": "shadow"},
        {"version_policy": "shadow"},
        {"versionPolicy": "shadow"},
    ):
        with pytest.raises(EDAContractError, match="common_field_collision"):
            ExtensionRow(
                row_id="shadow",
                payload=payload,
                coverage=_extension_coverage(),
            )
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            row_id="count",
            payload={"native_event_count": 3},
            coverage=_extension_coverage(),
        )

    source_native = ExtensionRow(
        row_id="composer",
        payload={
            "composer": {
                "category": "source-credit",
                "mean": "not-a-common-metric",
                "name": "Alice",
                "payload": {"value": "credited", "status": "verified"},
                "provenance": "source-metadata",
            }
        },
        coverage=_extension_coverage(),
    )
    assert source_native.payload["composer"]["name"] == "Alice"


@pytest.mark.parametrize(
    "name",
    ("accepted_records", "availability", "classSupport"),
)
def test_extension_typed_counts_cannot_shadow_common_fields(name) -> None:
    count = UnitCount(
        name=name,
        observation_unit=ObservationUnit.RECORD,
        value=1,
        denominator=1,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-extension",),
    )
    with pytest.raises(EDAContractError, match="common_field_collision"):
        ExtensionRow(
            row_id="shadow",
            payload={},
            counts=(count,),
            coverage=_extension_coverage(provenance=("source-extension",)),
        )


def test_graph_metrics_require_complete_target_free_attestation() -> None:
    with pytest.raises(EDAContractError, match="target_free_unproven"):
        GraphEvidence(status=ComputationStatus.OBSERVED, target_free=True)


def test_available_empty_multilabel_is_not_missing_or_a_label_occurrence(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    task = TaskFamilyEvidence(
        corpus=source.corpus,
        source_task_id="theory.chord.pitch_classes",
        dialect=source.dialect,
        annotation_namespace=source.annotation_namespace,
        vocabulary=source.vocabulary,
        label_granularity="source_entry",
        label_value_type=LabelValueType.MULTI_LABEL,
        observation_unit=ObservationUnit.TARGET_ROW,
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("empty-multilabel-fixture",),
        status=ComputationStatus.OBSERVED,
        availability=AvailabilityCounts(
            observation_unit=ObservationUnit.TARGET_ROW,
            denominator=1,
            available=1,
            masked=0,
            missing=0,
            unsupported=0,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("empty-multilabel-fixture",),
        ),
        class_support=(),
        empty_multilabel_available_count=UnitCount(
            name="empty_multilabel_available_count",
            observation_unit=ObservationUnit.TARGET_ROW,
            value=1,
            denominator=1,
            denominator_unit=ObservationUnit.TARGET_ROW,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("empty-multilabel-fixture",),
        ),
    )
    assert task.availability.available == 1
    assert task.availability.missing == 0
    assert task.class_support == ()
    assert task.empty_multilabel_available_count is not None
    assert task.empty_multilabel_available_count.value == 1


def test_work_support_is_not_fabricated_from_filename(supervision_reports) -> None:
    task = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    work = task.class_support[0].unique_work_count
    assert work.status == ComputationStatus.NOT_APPLICABLE
    assert work.value is None
    assert work.reason_code == "work_identity.unproven"


def test_manifest_paths_are_repository_relative() -> None:
    ref = VersionedIdentity("fixture", "1.0.0", "1" * 64)
    with pytest.raises(EDAContractError, match="absolute filesystem path"):
        InputManifestRef(
            role="raw_projection",
            identity=ref,
            target_free=True,
            repository_relative_path="/tmp/corpus.json",
        )


def test_non_finite_numeric_values_fail_closed() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(EDAContractError, match="must be finite"):
            NumericDistribution(
                measurement_unit="quarter_note",
                minimum=0,
                maximum=value,
                mean=0,
            )


def test_semantic_warning_changes_fingerprint_but_operational_data_does_not(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    first = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            operational_metadata={
                "hostname": "host-a",
                "pid": 1,
                "wall_clock_seconds": 0.1,
            },
        ),
        semantic_payload=source.semantic_payload,
    )
    second = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            operational_metadata={
                "hostname": "host-b",
                "pid": 99,
                "wall_clock_seconds": 9.9,
            },
        ),
        semantic_payload=source.semantic_payload,
    )
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert report_dict(first) != report_dict(second)
    warning = StructuredWarning(
        code="fixture.semantic_warning",
        message="semantic warning",
        provenance=("fixture",),
    )
    changed = RawCorpusEDA(
        envelope=replace(first.envelope, warnings=(warning,)),
        semantic_payload=first.semantic_payload,
    )
    assert changed.semantic_fingerprint != first.semantic_fingerprint
