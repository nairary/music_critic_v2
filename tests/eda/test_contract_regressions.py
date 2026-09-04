from __future__ import annotations

from dataclasses import replace

import pytest

import music_critic.eda as eda_api
import music_critic.eda.contracts as contract_module
from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    APPROVED_RAW_GRAPH_CONTRACT,
    AvailabilityCounts,
    CategoryCount,
    ClassSupport,
    ComputationStatus,
    CorpusId,
    EDAContractError,
    EDAReasonCode,
    EvidenceScope,
    ExtensionRow,
    GraphEvidence,
    InputManifestRef,
    InvariantEvidence,
    InvariantStatus,
    LabelValueType,
    MetricCoverage,
    NumericDistribution,
    ObservationUnit,
    ProjectionEvidence,
    QuantilePoint,
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
    dumps_report,
    loads_report,
    sum_unit_counts,
)


def _extension_coverage(
    *,
    observation_unit: ObservationUnit = ObservationUnit.RECORD,
    denominator: int | None = 1,
    split_scope: SplitScope = SplitScope.UNSPLIT,
    evidence_scope: EvidenceScope = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = ("fixture",),
    status: ComputationStatus = ComputationStatus.OBSERVED,
    reason_code: str | None = None,
) -> MetricCoverage:
    observed = status == ComputationStatus.OBSERVED
    return MetricCoverage(
        observation_unit=observation_unit,
        denominator=denominator,
        observed_count=denominator if observed else None,
        unknown_count=0 if observed else None,
        split_scope=split_scope,
        evidence_scope=evidence_scope,
        provenance=provenance,
        status=status,
        reason_code=reason_code,
    )


@pytest.mark.parametrize("scope", (EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE))
def test_observed_evidence_cannot_claim_unknown_or_unavailable_scope(scope) -> None:
    with pytest.raises(EDAContractError):
        MetricCoverage(
            observation_unit=ObservationUnit.RECORD,
            denominator=1,
            observed_count=1,
            unknown_count=0,
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=scope,
            provenance=("fixture",),
            status=ComputationStatus.OBSERVED,
        )

    with pytest.raises(EDAContractError):
        UnitCount(
            name="records",
            observation_unit=ObservationUnit.RECORD,
            value=1,
            denominator=1,
            denominator_unit=ObservationUnit.RECORD,
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=scope,
            provenance=("fixture",),
            status=ComputationStatus.OBSERVED,
        )


@pytest.mark.parametrize("scope", (EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE))
def test_availability_partition_cannot_claim_unknown_or_unavailable_scope(
    scope,
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    assert source.availability is not None
    with pytest.raises(EDAContractError):
        replace(source.availability, evidence_scope=scope)


def test_operational_metadata_rejects_semantic_smuggling(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="operational"):
        replace(
            source.envelope,
            operational_metadata={"semantic_model_version": "must-be-fingerprinted"},
        )


def test_scalar_source_value_cannot_encode_missing_as_a_class() -> None:
    for value in (None, ""):
        with pytest.raises(EDAContractError, match="source_value"):
            SourceValueIdentity(
                corpus=CorpusId.DILEMMADATA,
                source_task_id="dilemmadata.an.chord.quality",
                dialect="an_joint",
                source_value=value,
                value_kind=SourceValueKind.SCALAR,
            )


def test_multilabel_identity_is_set_ordered_and_rejects_invalid_members() -> None:
    first = SourceValueIdentity(
        corpus=CorpusId.HOOKTHEORY,
        source_task_id="theory.chord.pitch_classes",
        dialect="hooktheory-v2b1",
        source_value=("4", "0", "7"),
        value_kind=SourceValueKind.MULTI_LABEL,
    )
    reordered = SourceValueIdentity(
        corpus=CorpusId.HOOKTHEORY,
        source_task_id="theory.chord.pitch_classes",
        dialect="hooktheory-v2b1",
        source_value=("7", "4", "0"),
        value_kind=SourceValueKind.MULTI_LABEL,
    )
    assert first.identity == reordered.identity
    assert first.source_value == reordered.source_value

    for value in (
        ("0", "0"),
        ("0", None),
        ("",),
        (" ",),
        ("0 ",),
        (True,),
        (1,),
    ):
        with pytest.raises(EDAContractError, match="multilabel"):
            SourceValueIdentity(
                corpus=CorpusId.HOOKTHEORY,
                source_task_id="theory.chord.pitch_classes",
                dialect="hooktheory-v2b1",
                source_value=value,
                value_kind=SourceValueKind.MULTI_LABEL,
            )


def _multilabel_support_row(
    *,
    label: str,
    occurrences: int,
    provenance: tuple[str, ...],
) -> ClassSupport:
    return ClassSupport(
        source_value=SourceValueIdentity(
            corpus=CorpusId.HOOKTHEORY,
            source_task_id="theory.chord.pitch_classes",
            dialect="hooktheory-v2b1",
            source_value=label,
            value_kind=SourceValueKind.SCALAR,
        ),
        occurrence_count=UnitCount(
            name="occurrence_count",
            observation_unit=ObservationUnit.LABEL_OCCURRENCE,
            value=occurrences,
            denominator=3,
            denominator_unit=ObservationUnit.TARGET_ROW,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=provenance,
        ),
        unique_record_count=UnitCount(
            name="unique_record_count",
            observation_unit=ObservationUnit.RECORD,
            value=min(occurrences, 2),
            denominator=2,
            denominator_unit=ObservationUnit.RECORD,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=provenance,
        ),
        unique_work_count=UnitCount(
            name="unique_work_count",
            observation_unit=ObservationUnit.LOGICAL_WORK,
            value=None,
            denominator=None,
            denominator_unit=ObservationUnit.LOGICAL_WORK,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=provenance,
            status=ComputationStatus.NOT_APPLICABLE,
            reason_code="work_identity.unproven",
        ),
    )


def _nonempty_multilabel_task() -> TaskFamilyEvidence:
    provenance = ("multilabel-fixture",)
    support = tuple(
        _multilabel_support_row(
            label=label,
            occurrences=occurrences,
            provenance=provenance,
        )
        for label, occurrences in (("0", 2), ("4", 1), ("7", 2))
    )
    return TaskFamilyEvidence(
        corpus=CorpusId.HOOKTHEORY,
        source_task_id="theory.chord.pitch_classes",
        dialect="hooktheory-v2b1",
        annotation_namespace="hooktheory.theory",
        vocabulary=VersionedIdentity(
            "theory.chord.pitch_classes.vocabulary", "1.0.0", "3" * 64
        ),
        label_granularity="source_entry",
        label_value_type=LabelValueType.MULTI_LABEL,
        observation_unit=ObservationUnit.TARGET_ROW,
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=provenance,
        status=ComputationStatus.OBSERVED,
        availability=AvailabilityCounts(
            observation_unit=ObservationUnit.TARGET_ROW,
            denominator=3,
            available=3,
            masked=0,
            missing=0,
            unsupported=0,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=provenance,
        ),
        class_support=support,
        empty_multilabel_available_count=UnitCount(
            name="empty_multilabel_available_count",
            observation_unit=ObservationUnit.TARGET_ROW,
            value=1,
            denominator=3,
            denominator_unit=ObservationUnit.TARGET_ROW,
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=provenance,
        ),
    )


def test_nonempty_multilabel_support_uses_scalar_labels_and_round_trips(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    task = _nonempty_multilabel_task()
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, tasks=(task,)),
    )
    restored = loads_report(dumps_report(report))
    restored_task = restored.semantic_payload.tasks[0]
    assert all(
        item.source_value.value_kind == SourceValueKind.SCALAR
        for item in restored_task.class_support
    )
    assert sum(
        item.occurrence_count.value or 0 for item in restored_task.class_support
    ) == 5


def test_multilabel_class_support_rejects_set_identity_and_row_overcount() -> None:
    task = _nonempty_multilabel_task()
    first = task.class_support[0]
    set_identity = SourceValueIdentity(
        corpus=CorpusId.HOOKTHEORY,
        source_task_id=task.source_task_id,
        dialect=task.dialect,
        source_value=("0", "4"),
        value_kind=SourceValueKind.MULTI_LABEL,
    )
    with pytest.raises(EDAContractError, match="class_identity_invalid"):
        replace(task, class_support=(replace(first, source_value=set_identity),))

    overcount = replace(
        first,
        occurrence_count=replace(first.occurrence_count, value=3),
    )
    with pytest.raises(EDAContractError, match="class_occurrence_exceeded"):
        replace(task, class_support=(overcount, *task.class_support[1:]))


def test_class_support_rejects_mixed_denominator_units(supervision_reports) -> None:
    support = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[
        0
    ].class_support[0]
    invalid = replace(
        support.unique_record_count,
        denominator=100,
        denominator_unit=ObservationUnit.OPTIMIZER_UPDATE,
    )
    with pytest.raises(EDAContractError, match="class_support"):
        replace(support, unique_record_count=invalid)


@pytest.mark.parametrize(
    ("field", "name"),
    (
        ("occurrence_count", "test_class_occurrence"),
        ("unique_record_count", "occurrence_count"),
        ("unique_work_count", "unique_record_count"),
    ),
)
def test_class_support_count_names_are_field_bound(
    supervision_reports,
    field,
    name,
) -> None:
    support = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[
        0
    ].class_support[0]
    changed = replace(getattr(support, field), name=name)
    with pytest.raises(EDAContractError, match="name_invalid"):
        replace(support, **{field: changed})


def test_class_support_rejects_impossible_cardinality_and_provenance(
    supervision_reports,
) -> None:
    support = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[
        0
    ].class_support[0]
    assert support.occurrence_count.value == 1

    too_many_records = replace(
        support.unique_record_count,
        value=2,
        denominator=2,
    )
    with pytest.raises(EDAContractError, match="class_support"):
        replace(support, unique_record_count=too_many_records)

    unrelated_provenance = replace(
        support.unique_record_count,
        provenance=("unrelated-source",),
    )
    with pytest.raises(EDAContractError, match="class_support"):
        replace(support, unique_record_count=unrelated_provenance)

    work_support = supervision_reports[
        CorpusId.DILEMMADATA
    ].semantic_payload.tasks[0].class_support[0]
    too_many_works = replace(
        work_support.unique_work_count,
        value=2,
        denominator=2,
    )
    with pytest.raises(EDAContractError, match="class_support"):
        replace(work_support, unique_work_count=too_many_works)


def test_numeric_quantiles_must_lie_inside_distribution_bounds() -> None:
    with pytest.raises(EDAContractError, match="quantile"):
        NumericDistribution(
            measurement_unit="quarter_note",
            minimum=0,
            maximum=1,
            mean=0.5,
            quantiles=(QuantilePoint(1, 2, 99),),
        )


def test_projection_binds_existing_task_and_mapping_row() -> None:
    source_value = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id="dilemmadata.an.chord.quality",
        dialect="an_joint",
        source_value="major triad",
        value_kind=SourceValueKind.SCALAR,
    )
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))

    with pytest.raises(EDAContractError, match="projection"):
        ProjectionEvidence(
            source_value=source_value,
            mapping_registry=registry,
            common_task_identity="invented.common.task",
            native_state="available",
            mapping_state="exact",
            projected_value="major triad",
            provenance=("fixture",),
        )

    with pytest.raises(EDAContractError, match="projection"):
        ProjectionEvidence(
            source_value=source_value,
            mapping_registry=registry,
            common_task_identity="dilemmadata.common.chord.quality",
            native_state="available",
            mapping_state="exact",
            projected_value="minor triad",
            provenance=("fixture",),
        )


def test_duplicate_projection_rows_are_rejected(supervision_reports) -> None:
    task = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[0]
    projection = task.projections[0]
    with pytest.raises(EDAContractError, match="projection"):
        replace(task, projections=(projection, projection))


def test_report_projection_source_is_natively_available(supervision_reports) -> None:
    task = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[0]
    projection = task.projections[0]
    with pytest.raises(EDAContractError, match="native_state_invalid"):
        replace(
            projection,
            native_state="masked",
            mapping_state="masked",
            projected_value=None,
        )


def test_all_empty_multilabel_rows_cannot_have_label_occurrences(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    assert source.availability is not None
    assert source.availability.available == 1
    with pytest.raises(EDAContractError, match="multilabel"):
        replace(
            source,
            label_value_type=LabelValueType.MULTI_LABEL,
            empty_multilabel_available_count=UnitCount(
                name="empty_multilabel_available_count",
                observation_unit=ObservationUnit.TARGET_ROW,
                value=1,
                denominator=1,
                denominator_unit=ObservationUnit.TARGET_ROW,
                split_scope=source.split_scope,
                evidence_scope=source.evidence_scope,
                provenance=source.provenance,
            ),
        )


def test_supervision_extension_cannot_carry_test_evidence(supervision_reports) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    with pytest.raises(EDAContractError):
        extension = SourceExtension(
            corpus=CorpusId.HOOKTHEORY,
            namespace="hooktheory.test_leak",
            schema_name="TestLeak",
            schema_version="1.0.0",
            split_scope=SplitScope.TEST,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("fixture",),
            rows=(
                ExtensionRow(
                    row_id="leak",
                    payload={"classes": ["secret-test-class"]},
                    coverage=_extension_coverage(split_scope=SplitScope.TEST),
                ),
            ),
            target_free=False,
        )
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )

    for selector, payload_field in (
        ("partition", "classes"),
        ("scope", "coverage"),
        ("fold", "cooccurrence"),
    ):
        with pytest.raises(
            EDAContractError,
            match="common_field_collision|test_lock",
        ):
            hidden = SourceExtension(
                corpus=CorpusId.HOOKTHEORY,
                namespace=f"hooktheory.hidden_{selector}",
                schema_name="HiddenTestLeak",
                schema_version="1.0.0",
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=("fixture",),
                rows=(
                    ExtensionRow(
                        row_id="leak",
                        payload={selector: "test", payload_field: ["secret"]},
                        coverage=_extension_coverage(split_scope=SplitScope.TRAIN),
                    ),
                ),
                target_free=False,
            )
            SupervisionEDA(
                envelope=source.envelope,
                semantic_payload=replace(
                    source.semantic_payload,
                    extensions=(hidden,),
                ),
            )

    payload_leak = SourceExtension(
        corpus=CorpusId.HOOKTHEORY,
        namespace="hooktheory.test_payload_leak",
        schema_name="TestPayloadLeak",
        schema_version="1.0.0",
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                row_id="leak",
                payload={"split": "test", "classes": ["secret-test-class"]},
                coverage=_extension_coverage(split_scope=SplitScope.TRAIN),
            ),
        ),
        target_free=False,
    )
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(payload_leak,),
            ),
        )


def test_raw_extension_rejects_target_derived_aliases(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="forbidden"):
        extension = SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.target_leak",
            schema_name="TargetLeak",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("fixture",),
            rows=(
                ExtensionRow(
                    row_id="leak",
                    payload={"target_values": ["secret-target"]},
                    coverage=_extension_coverage(),
                ),
            ),
            target_free=True,
        )
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )

    target_manifest = InputManifestRef(
        role="gold_target_projection",
        identity=VersionedIdentity("fixture.target", "1.0.0", "f" * 64),
        target_free=True,
    )
    with pytest.raises(EDAContractError, match="target_manifest_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, input_manifests=(target_manifest,)),
            semantic_payload=source.semantic_payload,
        )


def test_raw_inventory_partition_must_match_discovered_records(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    metrics = []
    for metric in source.semantic_payload.metrics:
        if metric.metric_id == "quarantined_records":
            assert metric.count is not None
            metric = replace(metric, count=replace(metric.count, value=1))
        metrics.append(metric)
    with pytest.raises(EDAContractError, match="inventory_mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, metrics=tuple(metrics)),
        )


def test_projection_lookup_tables_are_deeply_immutable() -> None:
    task_id = "dilemmadata.an.chord.inversion"
    before = dict(contract_module._INVERSION_PROJECTION[task_id])
    with pytest.raises(TypeError):
        contract_module._INVERSION_PROJECTION[task_id]["0"] = "third"
    assert dict(contract_module._INVERSION_PROJECTION[task_id]) == before


def test_raw_metric_catalog_value_types_are_public_adapter_api() -> None:
    assert "MetricSummaryKind" in eda_api.__all__
    assert "RawMetricSpec" in eda_api.__all__
    spec = eda_api.RAW_METRIC_CATALOG["accepted_records"]
    assert isinstance(spec, eda_api.RawMetricSpec)
    assert spec.summary_kind is eda_api.MetricSummaryKind.COUNT


@pytest.mark.parametrize(
    "forbidden_unit",
    (
        ObservationUnit.TARGET_ACCESS_ATTEMPT,
        ObservationUnit.TARGET_ROW,
        ObservationUnit.LABEL_OCCURRENCE,
        ObservationUnit.AUGMENTED_PAIR,
        ObservationUnit.SAMPLER_PRESENTATION,
        ObservationUnit.OPTIMIZER_UPDATE,
    ),
)
@pytest.mark.parametrize("field", ("observation_unit", "denominator_unit"))
def test_raw_extensions_reject_target_and_training_units(
    raw_reports,
    forbidden_unit,
    field,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    units = {
        "observation_unit": ObservationUnit.RECORD,
        "denominator_unit": ObservationUnit.RECORD,
    }
    units[field] = forbidden_unit
    count = UnitCount(
        name="source_rows_seen",
        value=1,
        denominator=1,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-structure",),
        **units,
    )
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.source_structure",
        schema_name="SourceStructure",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-structure",),
        rows=(
            ExtensionRow(
                "structure",
                {"kind": "source"},
                (count,),
                coverage=_extension_coverage(
                    observation_unit=units["denominator_unit"],
                    provenance=("source-structure",),
                ),
            ),
        ),
        target_free=True,
    )
    with pytest.raises(EDAContractError, match="observation_unit_forbidden"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                observation_units=(
                    *source.envelope.observation_units,
                    forbidden_unit,
                ),
            ),
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )


def test_unproven_task_work_population_must_remain_null(supervision_reports) -> None:
    task = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    assert task.work_identity is None
    support = task.class_support[0]
    changed = replace(
        support,
        unique_work_count=replace(support.unique_work_count, denominator=123),
    )
    rows = tuple(changed if item is support else item for item in task.class_support)
    with pytest.raises(EDAContractError, match="work_identity_unproven"):
        replace(task, class_support=rows)


def test_unproven_extension_work_denominator_must_remain_null() -> None:
    count = UnitCount(
        name="records_per_work",
        observation_unit=ObservationUnit.RECORD,
        value=1,
        denominator=123,
        denominator_unit=ObservationUnit.LOGICAL_WORK,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-structure",),
    )
    with pytest.raises(EDAContractError, match="work_identity_unproven"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.work_summary",
            schema_name="WorkSummary",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-structure",),
            rows=(
                ExtensionRow(
                    "work",
                    {"kind": "source"},
                    (count,),
                    coverage=_extension_coverage(
                        observation_unit=ObservationUnit.LOGICAL_WORK,
                        denominator=123,
                        provenance=("source-structure",),
                    ),
                ),
            ),
            target_free=True,
        )


def test_class_support_requires_nonzero_record_for_occurrences(
    supervision_reports,
) -> None:
    support = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[
        0
    ].class_support[0]
    with pytest.raises(EDAContractError, match="record_cardinality_mismatch"):
        replace(
            support,
            occurrence_count=replace(support.occurrence_count, value=3),
            unique_record_count=replace(support.unique_record_count, value=0),
        )


def test_observed_work_support_requires_nonzero_work_for_records(
    supervision_reports,
) -> None:
    support = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[
        0
    ].class_support[0]
    with pytest.raises(EDAContractError, match="work_cardinality_mismatch"):
        replace(
            support,
            unique_work_count=replace(support.unique_work_count, value=0),
        )


@pytest.mark.parametrize(
    "identity",
    ("machine_name", "hostname", "pid", "timestamp", "wall_clock_duration"),
)
def test_operational_aliases_cannot_enter_typed_identity_channels(
    raw_reports,
    identity,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="semantic_alias_forbidden"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                source_identity=replace(
                    source.envelope.source_identity,
                    identity=identity,
                ),
            ),
            semantic_payload=source.semantic_payload,
        )


@pytest.mark.parametrize(
    "provenance",
    (
        "host_myserver",
        "machine_node_a",
        "generated_on_host_node_a",
        "runner_machine_node_a",
    ),
)
def test_host_and_machine_attestations_cannot_enter_provenance(
    raw_reports,
    provenance,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    warning = StructuredWarning(
        code="source.warning",
        message="source warning",
        provenance=(provenance,),
    )
    with pytest.raises(EDAContractError, match="semantic_alias_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, warnings=(warning,)),
            semantic_payload=source.semantic_payload,
        )


def test_drum_machine_extension_identities_remain_source_domain(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    provenance = ("drum-machine-source",)
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.drum_machine",
        schema_name="DrumMachineSummary",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=provenance,
        rows=(
            ExtensionRow(
                "device",
                {"drum_machine": "Roland TR-808"},
                coverage=_extension_coverage(provenance=provenance),
            ),
        ),
        target_free=True,
    )
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            producer_identity=replace(
                source.envelope.producer_identity,
                identity="pdmx.drum_machine_adapter",
            ),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            extensions=(extension,),
        ),
    )
    assert report.semantic_payload.extensions[0].schema_name == "DrumMachineSummary"


@pytest.mark.parametrize(
    "selector",
    (
        "heldout",
        "held_out",
        "holdout",
        "hold-out",
        "heldouttarget",
        "holdoutrow",
        "testlabel",
        "testrecord",
        "testrecords",
        "heldoutsample",
        "holdoutitem",
        "testingexamples",
        "tests_target_distribution",
    ),
)
def test_supervision_extensions_reject_held_out_test_aliases(
    supervision_reports,
    selector,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = SourceExtension(
        corpus=CorpusId.HOOKTHEORY,
        namespace="hooktheory.native_histogram",
        schema_name="NativeHistogram",
        schema_version="1.0.0",
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-supervision",),
        rows=(
            ExtensionRow(
                "partition",
                {"split_selector": selector, "native_histogram": {"major": "present"}},
                coverage=_extension_coverage(
                    split_scope=SplitScope.TRAIN,
                    provenance=("source-supervision",),
                ),
            ),
        ),
        target_free=False,
    )
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )


def test_raw_extension_preserves_established_target_free_source_terms(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.DILEMMADATA]
    extension = SourceExtension(
        corpus=CorpusId.DILEMMADATA,
        namespace="dilemmadata.raw_projection",
        schema_name="RawProjectionEvidence",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("target_independent_raw_projection",),
        rows=(
            ExtensionRow(
                "pitch_class_summary",
                {
                    "raw_projection_version": "1.0.0",
                    "pitch_class_histogram": {"C": 0.5},
                    "title": "Gold",
                    "category": "Theory",
                },
                coverage=_extension_coverage(
                    provenance=("target_independent_raw_projection",),
                ),
            ),
        ),
        target_free=True,
    )
    report = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    assert report.semantic_payload.extensions[0].namespace == (
        "dilemmadata.raw_projection"
    )


@pytest.mark.parametrize(
    "payload",
    (
        {"category": {"target_labels": ["C"]}},
        {"title": "supervision targets"},
        {"name": "theory_annotation"},
        {"target_independent": False},
        {"targetIndependent": "false"},
    ),
)
def test_raw_source_literals_cannot_hide_explicit_target_semantics(
    raw_reports,
    payload,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="forbidden|collision"):
        extension = SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.source_metadata",
            schema_name="SourceMetadata",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-metadata",),
            rows=(
                ExtensionRow(
                    "source",
                    payload,
                    coverage=_extension_coverage(
                        provenance=("source-metadata",),
                    ),
                ),
            ),
            target_free=True,
        )
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    "identity",
    (
        "pdmx.gold_target_labels",
        "pdmx.supervision_sidecar",
        "pdmx.answerkey",
    ),
)
def test_raw_split_assignment_manifest_remains_target_free(
    raw_reports,
    identity,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    manifest = replace(
        source.envelope.input_manifests[0],
        role="split_assignment",
        identity=replace(
            source.envelope.input_manifests[0].identity,
            identity=identity,
        ),
        repository_relative_path="targets/gold_labels.json",
    )
    with pytest.raises(EDAContractError, match="target_manifest_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, input_manifests=(manifest,)),
            semantic_payload=source.semantic_payload,
        )


def test_supervision_split_assignment_allows_test_names_but_not_targets(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    split_manifest = next(
        item
        for item in source.envelope.input_manifests
        if item.role == "split_assignment"
    )
    test_named = replace(
        split_manifest,
        identity=replace(
            split_manifest.identity,
            identity="hooktheory.test_split_assignment",
        ),
        repository_relative_path="manifests/test_split_assignment.json",
    )
    retained = tuple(
        test_named if item is split_manifest else item
        for item in source.envelope.input_manifests
    )
    report = SupervisionEDA(
        envelope=replace(source.envelope, input_manifests=retained),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.input_manifests[0].target_free is True

    target_named = replace(
        split_manifest,
        identity=replace(
            split_manifest.identity,
            identity="hooktheory.gold_target_labels",
        ),
        repository_relative_path="targets/gold_labels.json",
    )
    retained = tuple(
        target_named if item is split_manifest else item
        for item in source.envelope.input_manifests
    )
    with pytest.raises(EDAContractError, match="assignment_manifest_target_forbidden"):
        SupervisionEDA(
            envelope=replace(source.envelope, input_manifests=retained),
            semantic_payload=source.semantic_payload,
        )


def test_manifest_relative_path_may_contain_operational_surface_word(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    manifest = replace(
        source.envelope.input_manifests[0],
        repository_relative_path="manifests/run_timestamp.json",
    )
    report = RawCorpusEDA(
        envelope=replace(source.envelope, input_manifests=(manifest,)),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.input_manifests[0].repository_relative_path == (
        "manifests/run_timestamp.json"
    )


@pytest.mark.parametrize(
    "marker",
    (
        "not_targetfree",
        "nontargetfree",
        "targetfreefalse",
        "targetfreeno",
        "targetfreeunproven",
        "not_targetindependent",
        "targetindependentfalse",
    ),
)
def test_compact_negated_target_free_attestations_are_rejected(
    raw_reports,
    marker,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                source_identity=replace(
                    source.envelope.source_identity,
                    identity=f"pdmx.{marker}.release",
                ),
            ),
            semantic_payload=source.semantic_payload,
        )


def test_contract_collection_surfaces_reject_one_shot_iterables(
    raw_reports,
    supervision_reports,
) -> None:
    raw = raw_reports[CorpusId.PDMX]
    supervision = supervision_reports[CorpusId.HOOKTHEORY]
    metric = raw.semantic_payload.metrics[0]
    numeric_metric = next(
        item for item in raw.semantic_payload.metrics if item.numeric is not None
    )
    assert numeric_metric.numeric is not None
    extension = supervision.semantic_payload.extensions[0]
    row = extension.rows[0]
    task = supervision.semantic_payload.tasks[0]
    count = task.class_support[0].occurrence_count

    builders = (
        lambda: replace(
            raw.envelope,
            observation_units=(item for item in raw.envelope.observation_units),
        ),
        lambda: replace(
            raw.envelope,
            input_manifests=(item for item in raw.envelope.input_manifests),
        ),
        lambda: replace(raw.envelope, invariants=(item for item in ())),
        lambda: replace(raw.envelope, warnings=(item for item in ())),
        lambda: replace(
            raw.envelope,
            unavailable_reasons=(item for item in raw.envelope.unavailable_reasons),
        ),
        lambda: replace(metric, categories=(item for item in metric.categories)),
        lambda: replace(
            numeric_metric.numeric,
            quantiles=(item for item in numeric_metric.numeric.quantiles),
        ),
        lambda: replace(row, counts=(item for item in row.counts)),
        lambda: replace(extension, rows=(item for item in extension.rows)),
        lambda: replace(
            raw.semantic_payload,
            metrics=(item for item in raw.semantic_payload.metrics),
        ),
        lambda: replace(
            raw.semantic_payload,
            extensions=(item for item in raw.semantic_payload.extensions),
        ),
        lambda: replace(task, class_support=(item for item in task.class_support)),
        lambda: replace(
            task,
            projection_availability=(
                item for item in task.projection_availability
            ),
        ),
        lambda: replace(task, projections=(item for item in task.projections)),
        lambda: replace(
            supervision.semantic_payload,
            tasks=(item for item in supervision.semantic_payload.tasks),
        ),
        lambda: replace(
            supervision.semantic_payload,
            extensions=(item for item in supervision.semantic_payload.extensions),
        ),
        lambda: sum_unit_counts("sum", (item for item in (count,))),
    )
    for build in builders:
        with pytest.raises(EDAContractError, match="collection.type_invalid"):
            build()


@pytest.mark.parametrize(
    "field",
    (
        "targetrecord",
        "labelrecord",
        "supervisionsample",
        "annotationitem",
        "targetclassdistribution",
        "goldlabelsupport",
        "theorytargetmetadata",
        "annotationtargetrows",
        "labelclassmatrix",
        "supervisiontargetdataset",
        "targetlabelrecords",
        "classlabelcounts",
        "targetloader",
        "labelloader",
        "supervisionreader",
        "theoryfile",
        "annotationfile",
        "targetaccess",
        "targetopen",
        "targetpath",
        "targetfilepath",
        "targeturi",
        "targeturl",
    ),
)
def test_raw_extensions_reject_compact_target_compounds(raw_reports, field) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(
        EDAContractError,
        match="target_field_forbidden|untyped_count",
    ):
        extension = SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.source_summary",
            schema_name="SourceSummary",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-summary",),
            rows=(
                ExtensionRow(
                    "summary",
                    {field: "present"},
                    coverage=_extension_coverage(
                        provenance=("source-summary",),
                    ),
                ),
            ),
            target_free=True,
        )
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    "field",
    (
        "heldouttargetdistribution",
        "holdoutclasscoverage",
        "testtargetrows",
        "testclasssupport",
        "testlabelcooccurrence",
        "testingtargetdataset",
        "testsetclassdistribution",
        "heldouttargetrecords",
        "holdoutlabelsupport",
        "testpath",
        "testfilepath",
        "testfile",
        "testuri",
        "testurl",
        "heldoutpath",
        "holdoutfile",
    ),
)
def test_supervision_extensions_reject_compact_test_compounds(
    supervision_reports,
    field,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "summary",
                {field: "present"},
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"title": "Test Drive"},
        {"name": "Test"},
        {"description": "held-out composition"},
        {"category": "testing music"},
        {"source_value": "test"},
    ),
)
def test_supervision_extensions_preserve_source_domain_test_words(
    supervision_reports,
    payload,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "source_literal",
                payload,
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    assert dict(report.semantic_payload.extensions[0].rows[0].payload) == payload


@pytest.mark.parametrize(
    "payload",
    (
        {"title": "test target distribution"},
        {"category": {"heldout_labels": ["C"]}},
    ),
)
def test_supervision_source_literals_cannot_hide_test_semantics(
    supervision_reports,
    payload,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "source_literal",
                payload,
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    "marker",
    (
        "fixturedata",
        "fixturesource",
        "syntheticdata",
        "replaydata",
        "boundedresult",
        "fixtureadapter",
        "production_not_run",
        "pdmx.fixtureeda",
        "pdmx.syntheticreport",
        "pdmx.boundedanalysis",
        "pdmx.replayaudit",
        "pdmx.fixturesourcev2",
    ),
)
def test_production_attestation_rejects_compact_nonproduction_markers(marker) -> None:
    with pytest.raises(EDAContractError, match="fixture_as_production"):
        contract_module._reject_nonproduction_attestation(marker, path="$.identity")


@pytest.mark.parametrize("path", (".", "manifests/\x00bad.json", "manifests/\nbad.json"))
def test_manifest_paths_must_be_normalized_repository_file_paths(path) -> None:
    with pytest.raises(EDAContractError, match="path_invalid"):
        InputManifestRef(
            role="raw",
            identity=VersionedIdentity("pdmx.raw", "1.0.0", "a" * 64),
            target_free=True,
            repository_relative_path=path,
        )


@pytest.mark.parametrize(
    ("metric_id", "minimum", "maximum", "mean"),
    (
        ("duration", -10, -1, -5),
        ("notes", -4, -2, -3),
        ("tempo", -120, -10, -60),
        ("tempo", 0, 0, 0),
        ("pitch_range", 999, 1000, 999.5),
    ),
)
def test_raw_numeric_metrics_enforce_physical_domains(
    raw_reports,
    metric_id,
    minimum,
    maximum,
    mean,
) -> None:
    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == metric_id
    )
    spec = eda_api.RAW_METRIC_CATALOG[metric_id]
    with pytest.raises(EDAContractError, match="numeric_domain_invalid"):
        replace(
            metric,
            coverage=replace(
                metric.coverage,
                denominator=1,
                observed_count=1,
                unknown_count=0,
                status=ComputationStatus.OBSERVED,
                reason_code=None,
            ),
            numeric=NumericDistribution(
                measurement_unit=spec.measurement_unit,
                minimum=minimum,
                maximum=maximum,
                mean=mean,
            ),
        )


@pytest.mark.parametrize(
    "namespace",
    ("pdmx.", "pdmx..x", "pdmx. x", "pdmx.\x00x", "other.source"),
)
def test_source_extension_namespace_requires_normalized_owned_components(
    namespace,
) -> None:
    with pytest.raises(EDAContractError, match="namespace_invalid"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace=namespace,
            schema_name="SourceMetadata",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-metadata",),
            rows=(),
            target_free=True,
        )


def test_graph_metric_statuses_exactly_bind_graph_attestation(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="metric_status_mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                graph_evidence=replace(
                    source.semantic_payload.graph_evidence,
                    status=ComputationStatus.UNKNOWN,
                ),
            ),
        )

    graph_metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "graph_node_counts"
    )
    locked_metric = replace(
        graph_metric,
        coverage=replace(
            graph_metric.coverage,
            status=ComputationStatus.LOCKED,
            reason_code="eda.graph.locked",
        ),
    )
    metrics = tuple(
        locked_metric if item is graph_metric else item
        for item in source.semantic_payload.metrics
    )
    with pytest.raises(EDAContractError, match="metric_status_mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, metrics=metrics),
        )


@pytest.mark.parametrize(
    ("channel", "value"),
    (
        ("source_identity", "pdmx.cooccurrence"),
        ("manifest_role", "cooccurrence"),
        ("manifest_identity", "pdmx.projection"),
        ("manifest_path", "manifests/projection.json"),
        ("extension_namespace", "pdmx.cooccurrence_matrix"),
        ("payload", {"source_metric": "co-occurrence"}),
        ("domain_payload", {"category": "co-occurrence"}),
    ),
)
def test_raw_typed_channels_reject_projection_and_cooccurrence_semantics(
    raw_reports,
    channel,
    value,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    envelope = source.envelope
    payload = source.semantic_payload
    if channel == "source_identity":
        envelope = replace(
            envelope,
            source_identity=replace(envelope.source_identity, identity=value),
        )
    elif channel.startswith("manifest_"):
        manifest = envelope.input_manifests[0]
        if channel == "manifest_role":
            manifest = replace(manifest, role=value)
        elif channel == "manifest_identity":
            manifest = replace(
                manifest,
                identity=replace(manifest.identity, identity=value),
            )
        else:
            manifest = replace(manifest, repository_relative_path=value)
        envelope = replace(envelope, input_manifests=(manifest,))
    else:
        extension = SourceExtension(
            corpus=CorpusId.PDMX,
            namespace=(
                value if channel == "extension_namespace" else "pdmx.source_summary"
            ),
            schema_name="SourceSummary",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-summary",),
            rows=(
                ExtensionRow(
                    "summary",
                    value if channel in {"payload", "domain_payload"} else {},
                    coverage=_extension_coverage(
                        provenance=("source-summary",),
                    ),
                ),
            ),
            target_free=True,
        )
        payload = replace(payload, extensions=(extension,))
    with pytest.raises(EDAContractError, match="target_field_forbidden|forbidden"):
        RawCorpusEDA(envelope=envelope, semantic_payload=payload)


def test_repository_tests_fixture_path_does_not_mean_test_split(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    target_manifest = next(
        item for item in source.envelope.input_manifests if not item.target_free
    )
    fixture_manifest = replace(
        target_manifest,
        repository_relative_path="tests/fixtures/hooktheory/targets.json",
    )
    manifests = tuple(
        fixture_manifest if item is target_manifest else item
        for item in source.envelope.input_manifests
    )
    report = SupervisionEDA(
        envelope=replace(source.envelope, input_manifests=manifests),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.input_manifests[1].repository_relative_path.startswith(
        "tests/fixtures/"
    )


@pytest.mark.parametrize(
    "provenance",
    (
        "test_targets/labels.json",
        "tests_target_distribution",
        "test-target-reader",
        "test-loader",
        "testloader",
        "test-reader",
        "testreader",
        "test-opened",
        "test-read",
        "test-access",
        "test-used",
    ),
)
def test_test_lock_provenance_cannot_claim_target_access(
    supervision_reports,
    provenance,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    existing = source.semantic_payload.test_lock
    lock = type(existing).from_guard(
        test_assignment_count=existing.test_assignment_count.value,
        assignment_manifest_fingerprint=existing.assignment_manifest_fingerprint,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=(provenance,),
    )
    with pytest.raises(EDAContractError, match="provenance_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, test_lock=lock),
        )


def test_test_lock_guard_provenance_may_name_the_lock(supervision_reports) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    existing = source.semantic_payload.test_lock
    lock = type(existing).from_guard(
        test_assignment_count=existing.test_assignment_count.value,
        assignment_manifest_fingerprint=existing.assignment_manifest_fingerprint,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("test-lock-guard",),
    )
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, test_lock=lock),
    )
    assert report.semantic_payload.test_lock.test_targets_read is False


@pytest.mark.parametrize(
    "reason_code",
    ("read_test_targets", "testloader", "testreader", "testopened", "testaccess"),
)
def test_test_lock_reason_cannot_claim_target_read(
    supervision_reports,
    reason_code,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    existing = source.semantic_payload.test_lock
    lock = type(existing).not_executed(
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("test-lock-guard",),
        reason_code=reason_code,
        assignment_manifest_fingerprint=existing.assignment_manifest_fingerprint,
        test_assignment_denominator=existing.test_assignment_count.denominator,
    )
    with pytest.raises(EDAContractError, match="reason_code_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, test_lock=lock),
        )


def test_extension_payload_cannot_encode_unapproved_common_projection() -> None:
    with pytest.raises(EDAContractError, match="common_field_collision"):
        ExtensionRow(
            "projection",
            {
                "common_task_identity": "invented.common.chord",
                "mapping_registry": {
                    "identity": "invented.registry",
                    "version": "1.0.0",
                    "fingerprint": "a" * 64,
                },
                "native_state": "available",
                "mapping_state": "exact",
                "projected_value": "major",
            },
            coverage=_extension_coverage(),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"available": 1, "masked": 0, "missing": 0, "unsupported": 0},
        {
            "exact": 1,
            "coarsened": 0,
            "ambiguous": 0,
            "unsupported": 0,
            "invalid": 0,
            "missing": 0,
            "masked": 0,
        },
    ),
)
def test_extension_payload_cannot_shadow_common_state_partitions(payload) -> None:
    with pytest.raises(EDAContractError, match="common_field_collision"):
        ExtensionRow("partition", payload, coverage=_extension_coverage())


@pytest.mark.parametrize(
    "identity",
    (
        "hooktheory.testfold",
        "hooktheory.testingfold",
        "hooktheory.holdoutfold",
        "hooktheory.heldoutfold",
        "hooktheory.testsfold",
    ),
)
def test_supervision_typed_identity_rejects_compact_test_folds(
    supervision_reports,
    identity,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    with pytest.raises(EDAContractError, match="task_test_field_forbidden"):
        SupervisionEDA(
            envelope=replace(
                source.envelope,
                source_identity=replace(
                    source.envelope.source_identity,
                    identity=identity,
                ),
            ),
            semantic_payload=source.semantic_payload,
        )


@pytest.mark.parametrize("version", ("01.0.0", "1.00.0", "1.0.00", "00.00.00"))
def test_semantic_versions_reject_leading_zero_components(version) -> None:
    with pytest.raises(EDAContractError, match="version.invalid"):
        VersionedIdentity("source.release", version, "a" * 64)


@pytest.mark.parametrize(
    "identity",
    (
        "HookTheory.release",
        "HookTheoryEDAAdapter",
        "hook_theory.release",
        "hook-theory-release",
    ),
)
def test_hooktheory_corpus_name_is_not_target_derived(
    raw_reports,
    identity,
) -> None:
    source = raw_reports[CorpusId.HOOKTHEORY]
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            source_identity=replace(source.envelope.source_identity, identity=identity),
        ),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.corpus == CorpusId.HOOKTHEORY


def test_hooktheory_name_does_not_exempt_theory_label_identity(raw_reports) -> None:
    source = raw_reports[CorpusId.HOOKTHEORY]
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                source_identity=replace(
                    source.envelope.source_identity,
                    identity="HookTheory.theory_labels",
                ),
            ),
            semantic_payload=source.semantic_payload,
        )


def _raw_report_with_extension_payload(raw_reports, payload) -> RawCorpusEDA:
    source = raw_reports[CorpusId.PDMX]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.regression_probe",
        schema_name="RegressionProbe",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-regression-probe",),
        rows=(
            ExtensionRow(
                "probe",
                payload,
                coverage=_extension_coverage(
                    provenance=("source-regression-probe",),
                ),
            ),
        ),
        target_free=True,
    )
    return RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )


@pytest.mark.parametrize(
    "payload",
    (
        {"ground": {"truth": {"values": ["C"]}}},
        {"answer": {"key": {"values": ["C"]}}},
    ),
)
def test_nested_raw_target_aliases_are_rejected(raw_reports, payload) -> None:
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        _raw_report_with_extension_payload(raw_reports, payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"hostnames": ["node-a"]},
        {"pids": [42]},
        {"process_ids": [42]},
        {"machine_names": ["node-a"]},
        {"machine_identifier": "node-a"},
        {"machine_identifiers": ["node-a"]},
        {"host_identifier": "node-a"},
        {"host_identifiers": ["node-a"]},
        {"process_identifier": "42"},
        {"process_identifiers": ["42"]},
        {"run_timestamps": ["2026-09-03T12:00:00Z"]},
        {"runtimes": [1.0]},
        {"host_names": ["node-a"]},
        {"machine_ids": ["node-a"]},
        {"processids": [42]},
        {"run_times": [1.0]},
        {"runtimestamps": ["2026-09-03T12:00:00Z"]},
        {"execution_times": [1.0]},
        {"processing_times": [1.0]},
        {"wallclocks": [1.0]},
        {"process": {"id": 42}},
        {"machine": {"name": "node-a"}},
        {"run": {"time": "2026-09-03T12:00:00Z"}},
        {"wall": {"clock": {"seconds": 1.2}}},
    ),
)
def test_operational_plural_and_nested_aliases_are_rejected(
    raw_reports,
    payload,
) -> None:
    with pytest.raises(EDAContractError, match="operational_metadata"):
        _raw_report_with_extension_payload(raw_reports, payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"title": "V /ii"},
        {"title": "AC/DC"},
        {"title": "Love/Hate"},
        {"record_label": "Decca Records"},
        {"publisher_label": "Blue Note Records"},
        {"title": "Gold Label Records"},
        {"parser": {"runtime_complexity": "O(n log n)"}},
        {"event": {"timestamp": "source-beat-1"}},
        {"drum_machine": "Roland TR-808"},
        {"instrument": {"drum_machine": True}},
        {"machine_readable_format": "MIDI"},
    ),
)
def test_source_domain_and_static_runtime_literals_are_preserved(
    raw_reports,
    payload,
) -> None:
    report = _raw_report_with_extension_payload(raw_reports, payload)
    assert dict(report.semantic_payload.extensions[0].rows[0].payload) == payload


def test_processing_timestamp_is_not_a_source_event_timestamp(raw_reports) -> None:
    with pytest.raises(EDAContractError, match="forbidden|operational"):
        _raw_report_with_extension_payload(
            raw_reports,
            {
                "event": {
                    "processing": {"timestamp": "2026-09-03T12:00:00Z"}
                }
            },
        )


@pytest.mark.parametrize(
    "title",
    (
        "Imported from /home/alice/private/song.mid",
        "Imported from /root",
        "Imported from /workspace",
        "Imported from /secret",
        "Imported-from-/secret",
        "Imported_from_/secret",
        "Imported from //server/share/song.mid",
        "Imported from ///secret",
        r"Imported from C:\\Users\\alice\\song.mid",
        r"Imported from \secret",
        r"Imported from \\secret",
        r"Imported_from_\secret",
        r"Imported from \\\\server\\share\\song.mid",
    ),
)
def test_domain_literals_still_reject_embedded_absolute_paths(
    raw_reports,
    title,
) -> None:
    with pytest.raises(EDAContractError, match="absolute_path_forbidden"):
        _raw_report_with_extension_payload(raw_reports, {"title": title})


@pytest.mark.parametrize(
    "identity",
    ("pdmx.self_supervised_corpus", "SelfSupervisedRawAdapter"),
)
def test_self_supervised_raw_identities_are_target_free(raw_reports, identity) -> None:
    source = raw_reports[CorpusId.PDMX]
    report = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            source_identity=replace(source.envelope.source_identity, identity=identity),
        ),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.source_identity.identity == identity


@pytest.mark.parametrize(
    "identity",
    (
        "pdmx.labeled_corpus",
        "pdmx.labelled_corpus",
        "pdmx.annotated_corpus",
        "pdmx.supervised_corpus",
        "pdmx.semi_supervised_corpus",
    ),
)
def test_raw_identities_reject_positive_supervision_morphology(
    raw_reports,
    identity,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                source_identity=replace(
                    source.envelope.source_identity,
                    identity=identity,
                ),
            ),
            semantic_payload=source.semantic_payload,
        )


def test_structural_control_characters_fail_but_domain_prose_remains_utf8() -> None:
    with pytest.raises(EDAContractError, match="control_character"):
        VersionedIdentity("pdmx.ra\u200bw", "1.0.0", "a" * 64)
    with pytest.raises(EDAContractError, match="key_invalid"):
        ExtensionRow(
            "probe",
            {"source\u200bfield": "value"},
            coverage=_extension_coverage(),
        )
    with pytest.raises(EDAContractError, match="control_character"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.probe",
            schema_name="Probe",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source\u200bprobe",),
            rows=(),
            target_free=True,
        )

    count = UnitCount(
        name="domain_category",
        observation_unit=ObservationUnit.RECORD,
        value=1,
        denominator=1,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("source-probe",),
    )
    assert CategoryCount("engineer 👩‍💻", count).category == "engineer 👩‍💻"
    assert "\n" in StructuredWarning(
        code="source.prose",
        message="first line\nsecond\tline 👩‍💻",
        provenance=("source-probe",),
    ).message
    assert UnavailableReason(
        code="source.unavailable",
        status=ComputationStatus.NOT_COMPUTED,
        provenance=("source-probe",),
        detail="first line\nsecond\tline 👩‍💻",
    ).detail is not None


@pytest.mark.parametrize(
    "payload",
    (
        {
            "duration_summary": {
                "minimum": 1,
                "maximum": 4,
                "mean": 2.5,
                "measurement_unit": "quarter_note",
            }
        },
        {"notes_summary": {"minimum": 1, "maximum": 4, "mean": 2.5}},
        {"raw_duration": 4},
        {"tempo_statistics": {"minimum": 80, "maximum": 120}},
        {"graph_size_summary": {"minimum": 2, "maximum": 9}},
        {"coverage_override": {"observed_total": 8, "unknown_total": 2}},
        {
            "availability_override": {
                "available_total": 8,
                "masked_total": 1,
                "missing_total": 1,
                "unsupported_total": 0,
            }
        },
        {
            "common_task_identity_override": "invented.common.task",
            "mapping_registry_override": {
                "identity": "invented.registry",
                "version": "1.0.0",
                "fingerprint": "a" * 64,
            },
            "native_state_override": "available",
            "mapping_state_override": "exact",
            "projected_value_override": "major",
        },
    ),
)
def test_extensions_cannot_shadow_common_fields_through_aliases(payload) -> None:
    with pytest.raises(EDAContractError, match="common_field_collision"):
        ExtensionRow("shadow", payload, coverage=_extension_coverage())


@pytest.mark.parametrize(
    "payload",
    (
        {"source_duration_summary": {"mean": 4.0, "unit": "quarter_note"}},
        {"audio_duration_seconds": 3.5},
        {"probability_distribution": {"major": 1, "minor": 0}},
        {"normalized_distribution": {"major": 1, "minor": 0}},
        {"confidence_histogram": {"high": 1, "low": 0}},
        {"time_signature": {"numerator": 4, "denominator": 4, "clocks_per_click": 24}},
        {"ratio": {"numerator": 1, "denominator": 2, "reduced": True}},
        {"canonical_pitch_id": 60},
        {"logical_track_id": "track-1"},
    ),
)
def test_source_qualified_measurements_and_ratios_are_not_common_counts(
    payload,
) -> None:
    row = ExtensionRow(
        "source-measurement",
        payload,
        coverage=_extension_coverage(),
    )
    assert dict(row.payload) == payload


@pytest.mark.parametrize(
    "payload",
    (
        {"confidence_histogram": {"high": 999, "low": 999}},
        {"probability_distribution": {"major": 2.0, "minor": 1.0}},
        {"normalized_distribution": {"major": 1.5, "minor": 0.5}},
        {"weight_histogram": {"high": 9, "low": 1}},
    ),
)
def test_probability_like_extension_containers_cannot_hide_bucket_counts(
    payload,
) -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            "source-measurement",
            payload,
            coverage=_extension_coverage(denominator=10),
        )


def test_ratio_auxiliary_fields_cannot_hide_untyped_counts() -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            "ratio",
            {
                "ratio": {
                    "numerator": 1,
                    "denominator": 2,
                    "record_count": 99,
                }
            },
            coverage=_extension_coverage(),
        )


def test_raw_multivalued_reason_codes_and_empty_occurrence_metrics(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    reason_metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "reason_codes"
    )
    observed_coverage = replace(
        reason_metric.coverage,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
    )

    def category(value: str) -> CategoryCount:
        return CategoryCount(
            value,
            UnitCount(
                name="reason_codes",
                observation_unit=ObservationUnit.RECORD,
                value=1,
                denominator=1,
                denominator_unit=ObservationUnit.RECORD,
                split_scope=SplitScope.UNSPLIT,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=observed_coverage.provenance,
            ),
        )

    multivalued = replace(
        reason_metric,
        coverage=observed_coverage,
        categories=(category("parse"), category("oversize")),
    )
    empty_multivalued = replace(
        reason_metric,
        coverage=observed_coverage,
        categories=(),
    )
    instrument_metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "instruments"
    )
    empty_occurrences = replace(
        instrument_metric,
        coverage=replace(
            instrument_metric.coverage,
            denominator=1,
            observed_count=1,
            unknown_count=0,
            status=ComputationStatus.OBSERVED,
            reason_code=None,
        ),
        categories=(),
    )
    assert len(multivalued.categories) == 2
    assert empty_multivalued.categories == ()
    assert empty_occurrences.categories == ()

    single_valued = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "conversion_outcomes"
    )
    with pytest.raises(EDAContractError, match="summary_arity"):
        replace(
            single_valued,
            coverage=replace(
                single_valued.coverage,
                denominator=1,
                observed_count=1,
                unknown_count=0,
                status=ComputationStatus.OBSERVED,
                reason_code=None,
            ),
            categories=(),
        )


def test_known_empty_count_requires_an_explicit_typed_zero(raw_reports) -> None:
    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == "accepted_records"
    )
    coverage = replace(
        metric.coverage,
        denominator=0,
        observed_count=0,
        unknown_count=0,
    )
    zero = replace(metric.count, value=0, denominator=0)
    assert replace(metric, coverage=coverage, count=zero).count == zero
    with pytest.raises(EDAContractError, match="explicit typed zero"):
        replace(metric, coverage=coverage, count=None)


def test_numeric_endpoints_discrete_bounds_and_singletons_are_consistent(
    raw_reports,
) -> None:
    with pytest.raises(EDAContractError, match="endpoint_mismatch"):
        NumericDistribution(
            measurement_unit="quarter_note",
            minimum=0,
            maximum=4,
            mean=2,
            quantiles=(QuantilePoint(0, 1, 1), QuantilePoint(1, 1, 4)),
        )

    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == "notes"
    )
    coverage = replace(
        metric.coverage,
        denominator=2,
        observed_count=2,
        unknown_count=0,
    )
    with pytest.raises(EDAContractError, match="numeric_domain_invalid"):
        replace(
            metric,
            coverage=coverage,
            numeric=NumericDistribution(
                measurement_unit="notes_per_record",
                minimum=0.5,
                maximum=2.5,
                mean=1.5,
            ),
        )
    with pytest.raises(EDAContractError, match="singleton_summary_invalid"):
        replace(
            metric,
            coverage=replace(
                coverage,
                denominator=1,
                observed_count=1,
            ),
            numeric=NumericDistribution(
                measurement_unit="notes_per_record",
                minimum=1,
                maximum=2,
                mean=1.5,
            ),
        )
    fractional = replace(
        metric,
        coverage=coverage,
        numeric=NumericDistribution(
            measurement_unit="notes_per_record",
            minimum=0,
            maximum=1,
            mean=0.5,
            quantiles=(QuantilePoint(1, 2, 0.5),),
        ),
    )
    assert fractional.numeric is not None
    assert fractional.numeric.mean == 0.5


def test_numeric_mean_and_extrema_are_feasible_for_observed_sample(
    raw_reports,
) -> None:
    metric = next(
        item
        for item in raw_reports[CorpusId.PDMX].semantic_payload.metrics
        if item.metric_id == "duration"
    )
    with pytest.raises(EDAContractError, match="sample_extrema_mismatch"):
        replace(
            metric,
            coverage=replace(
                metric.coverage,
                denominator=2,
                observed_count=2,
                unknown_count=0,
            ),
            numeric=NumericDistribution(
                measurement_unit="quarter_note",
                minimum=0,
                maximum=2,
                mean=0,
            ),
        )


def test_canonical_raw_and_test_lock_reason_codes_are_allowed(
    raw_reports,
    supervision_reports,
) -> None:
    raw = raw_reports[CorpusId.PDMX]
    reason = UnavailableReason(
        code=EDAReasonCode.TARGET_FREE_UNPROVEN,
        status=ComputationStatus.NOT_COMPUTED,
        provenance=("raw-contract",),
    )
    density = next(
        item for item in raw.semantic_payload.metrics if item.metric_id == "density"
    )
    changed_density = replace(
        density,
        coverage=replace(
            density.coverage,
            reason_code=EDAReasonCode.TARGET_FREE_UNPROVEN,
        ),
    )
    metrics = tuple(
        changed_density if item is density else item
        for item in raw.semantic_payload.metrics
    )
    invariant = InvariantEvidence(
        code="raw.graph_target_free",
        status=InvariantStatus.NOT_COMPUTED,
        provenance=("raw-contract",),
        reason_code=EDAReasonCode.TARGET_FREE_UNPROVEN,
    )
    raw_report = RawCorpusEDA(
        envelope=replace(
            raw.envelope,
            invariants=(invariant,),
            unavailable_reasons=(
                *raw.envelope.unavailable_reasons,
                reason,
            ),
        ),
        semantic_payload=replace(raw.semantic_payload, metrics=metrics),
    )
    assert raw_report.envelope.invariants == (invariant,)

    supervision = supervision_reports[CorpusId.HOOKTHEORY]
    lock_reason = UnavailableReason(
        code=EDAReasonCode.TEST_TARGETS_LOCKED,
        status=ComputationStatus.LOCKED,
        provenance=("target-access-guard",),
    )
    lock_invariant = InvariantEvidence(
        code="supervision.partition_lock",
        status=InvariantStatus.NOT_COMPUTED,
        provenance=("target-access-guard",),
        reason_code=EDAReasonCode.TEST_TARGETS_LOCKED,
    )
    report = SupervisionEDA(
        envelope=replace(
            supervision.envelope,
            invariants=(lock_invariant,),
            unavailable_reasons=(lock_reason,),
        ),
        semantic_payload=supervision.semantic_payload,
    )
    assert report.envelope.invariants == (lock_invariant,)


def _task_for_split(task: TaskFamilyEvidence, split: SplitScope) -> TaskFamilyEvidence:
    availability = (
        None
        if task.availability is None
        else replace(task.availability, split_scope=split)
    )
    support = tuple(
        replace(
            row,
            occurrence_count=replace(row.occurrence_count, split_scope=split),
            unique_record_count=replace(row.unique_record_count, split_scope=split),
            unique_work_count=replace(row.unique_work_count, split_scope=split),
        )
        for row in task.class_support
    )
    return replace(
        task,
        split_scope=split,
        availability=availability,
        class_support=support,
    )


def test_task_schema_identity_is_stable_across_train_validation(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    train = source.semantic_payload.tasks[0]
    validation = _task_for_split(train, SplitScope.VALIDATION)
    envelope = replace(source.envelope, split_scope=SplitScope.TRAIN_VALIDATION)
    report = SupervisionEDA(
        envelope=envelope,
        semantic_payload=replace(
            source.semantic_payload,
            tasks=(train, validation),
        ),
    )
    assert len(report.semantic_payload.tasks) == 2

    mismatched = replace(
        validation,
        vocabulary=replace(
            validation.vocabulary,
            identity="theory.chord.inversion.validation_vocabulary",
        ),
    )
    with pytest.raises(EDAContractError, match="task_schema_mismatch"):
        replace(source.semantic_payload, tasks=(train, mismatched))


@pytest.mark.parametrize(
    "payload",
    (
        {"held": {"out": {"labels": ["C"]}}},
        {"hold": {"out": {"target_rows": ["x"]}}},
    ),
)
def test_nested_test_target_aliases_are_rejected(
    supervision_reports,
    payload,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "probe",
                payload,
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )


@pytest.mark.parametrize("value", ("   ", "\t", "\n", " major "))
def test_scalar_source_values_reject_blank_or_padded_strings(value: str) -> None:
    with pytest.raises(EDAContractError, match="missing_is_not_class"):
        SourceValueIdentity(
            corpus=CorpusId.HOOKTHEORY,
            source_task_id="theory.chord.quality",
            dialect="hooktheory-v2b1",
            source_value=value,
            value_kind=SourceValueKind.SCALAR,
        )


def test_extension_schema_identity_is_stable_across_splits(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    train = source.semantic_payload.extensions[0]
    validation_rows = tuple(
        replace(
            row,
            coverage=replace(
                row.coverage,
                split_scope=SplitScope.VALIDATION,
            ),
        )
        for row in train.rows
    )
    validation = replace(
        train,
        split_scope=SplitScope.VALIDATION,
        schema_name="DifferentValidationSchema",
        schema_version="2.0.0",
        rows=validation_rows,
    )
    with pytest.raises(EDAContractError, match="schema_mismatch"):
        replace(source.semantic_payload, extensions=(train, validation))


def test_graph_metric_coverages_bind_one_population_and_lineage(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    metrics = tuple(
        replace(
            metric,
            coverage=replace(metric.coverage, split_scope=SplitScope.TRAIN),
        )
        if metric.metric_id == "graph_node_counts"
        else metric
        for metric in source.semantic_payload.metrics
    )
    with pytest.raises(EDAContractError, match="metric_coverage_mismatch"):
        RawCorpusEDA(
            envelope=replace(source.envelope, split_scope=SplitScope.ALL),
            semantic_payload=replace(source.semantic_payload, metrics=metrics),
        )


@pytest.mark.parametrize(
    "channel",
    ("source_identity", "task_provenance", "extension_payload"),
)
@pytest.mark.parametrize("alias", ("tests", "testsets"))
def test_bare_plural_test_aliases_are_rejected_in_typed_channels(
    supervision_reports,
    channel,
    alias,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    envelope = source.envelope
    payload = source.semantic_payload
    if channel == "source_identity":
        envelope = replace(
            envelope,
            source_identity=replace(envelope.source_identity, identity=alias),
        )
    elif channel == "task_provenance":
        task = replace(payload.tasks[0], provenance=(alias,))
        payload = replace(payload, tasks=(task,))
    else:
        extension = replace(
            payload.extensions[0],
            rows=(
                ExtensionRow(
                    "probe",
                    {alias: [3, 2]},
                    coverage=payload.extensions[0].rows[0].coverage,
                ),
            ),
        )
        payload = replace(payload, extensions=(extension,))
    with pytest.raises(EDAContractError, match="test_lock"):
        SupervisionEDA(envelope=envelope, semantic_payload=payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"canonical_work_id": "filename-stem"},
        {"logical_work_id": "filename-stem"},
        {"work_ids": ["filename-stem"]},
        {"work": {"identifier": "filename-stem"}},
        {"work": {"identity": "filename-stem"}},
        {"work_uuid": "filename-stem"},
        {"workid": "filename-stem"},
        {"workids": ["filename-stem"]},
        {"workidentifier": "filename-stem"},
        {"workidentifiers": ["filename-stem"]},
        {"workidentities": ["filename-stem"]},
        {"workuuids": ["filename-stem"]},
        {"canonical_id": "filename-stem"},
        {"canonicalids": ["filename-stem"]},
        {"canonical_identifier": "filename-stem"},
        {"logical_id": "filename-stem"},
        {"logicalids": ["filename-stem"]},
        {"logical_identifier": "filename-stem"},
        {"work_key": "filename-stem"},
    ),
)
def test_extension_work_ids_require_versioned_identity(payload) -> None:
    with pytest.raises(EDAContractError, match="work_identity_unproven"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.work_probe",
            schema_name="WorkProbe",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("source-work-probe",),
            rows=(
                ExtensionRow(
                    "work",
                    payload,
                    coverage=_extension_coverage(
                        provenance=("source-work-probe",),
                    ),
                ),
            ),
            target_free=True,
        )


def test_raw_inventory_metrics_require_one_coverage_partition(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    values = {
        "discovered_records": (8, 10, 0),
        "accepted_records": (5, 5, 5),
        "quarantined_records": (3, 3, 7),
    }
    metrics = []
    for metric in source.semantic_payload.metrics:
        if metric.metric_id in values:
            value, observed, unknown = values[metric.metric_id]
            assert metric.count is not None
            metric = replace(
                metric,
                coverage=replace(
                    metric.coverage,
                    denominator=10,
                    observed_count=observed,
                    unknown_count=unknown,
                    provenance=("inventory",),
                ),
                count=replace(
                    metric.count,
                    value=value,
                    denominator=10,
                    provenance=("inventory",),
                ),
            )
        metrics.append(metric)
    with pytest.raises(EDAContractError, match="inventory_mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                metrics=tuple(metrics),
            ),
        )


def test_discovered_inventory_count_equals_observed_coverage(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    metrics = tuple(
        replace(metric, count=replace(metric.count, value=0))
        if metric.metric_id
        in {"discovered_records", "accepted_records", "quarantined_records"}
        and metric.count is not None
        else metric
        for metric in source.semantic_payload.metrics
    )
    with pytest.raises(EDAContractError, match="inventory_mismatch"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, metrics=metrics),
        )


def test_graph_size_mean_matches_node_plus_edge_occurrences(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    graph_ids = {
        "graph_node_counts",
        "graph_edge_counts",
        "graph_size_distribution",
    }
    graph_metrics = {
        metric.metric_id: metric
        for metric in source.semantic_payload.metrics
        if metric.metric_id in graph_ids
    }
    coverage = replace(
        graph_metrics["graph_node_counts"].coverage,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
        provenance=("graph-fixture",),
    )

    def graph_category(metric_id: str, unit: ObservationUnit, value: int):
        return CategoryCount(
            "native",
            UnitCount(
                name=metric_id,
                observation_unit=unit,
                value=value,
                denominator=1,
                denominator_unit=ObservationUnit.RECORD,
                split_scope=SplitScope.UNSPLIT,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=coverage.provenance,
            ),
        )

    replacements = {
        "graph_node_counts": replace(
            graph_metrics["graph_node_counts"],
            coverage=coverage,
            categories=(
                graph_category(
                    "graph_node_counts",
                    ObservationUnit.GRAPH_NODE,
                    2,
                ),
            ),
        ),
        "graph_edge_counts": replace(
            graph_metrics["graph_edge_counts"],
            coverage=coverage,
            categories=(
                graph_category(
                    "graph_edge_counts",
                    ObservationUnit.GRAPH_EDGE,
                    3,
                ),
            ),
        ),
        "graph_size_distribution": replace(
            graph_metrics["graph_size_distribution"],
            coverage=coverage,
            numeric=NumericDistribution(
                measurement_unit="nodes_plus_edges_per_record",
                minimum=999,
                maximum=999,
                mean=999,
            ),
        ),
    }
    metrics = tuple(replacements.get(metric.metric_id, metric) for metric in source.semantic_payload.metrics)
    graph_evidence = GraphEvidence(
        status=ComputationStatus.OBSERVED,
        target_free=True,
        **dict(APPROVED_RAW_GRAPH_CONTRACT),
    )
    with pytest.raises(EDAContractError, match="size_mean_mismatch"):
        RawCorpusEDA(
            envelope=replace(
                source.envelope,
                observation_units=(
                    ObservationUnit.RECORD,
                    ObservationUnit.GRAPH_NODE,
                    ObservationUnit.GRAPH_EDGE,
                ),
            ),
            semantic_payload=replace(
                source.semantic_payload,
                metrics=metrics,
                graph_evidence=graph_evidence,
            ),
        )


def test_graph_size_extrema_are_feasible_for_aggregate_total(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    graph_ids = {
        "graph_node_counts",
        "graph_edge_counts",
        "graph_size_distribution",
    }
    graph_metrics = {
        metric.metric_id: metric
        for metric in source.semantic_payload.metrics
        if metric.metric_id in graph_ids
    }
    coverage = replace(
        graph_metrics["graph_node_counts"].coverage,
        denominator=2,
        observed_count=2,
        unknown_count=0,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
        provenance=("graph-fixture",),
    )
    base = 2**53
    rounded_mean = float(base)
    node_total = 2 * base + 1
    replacements = {
        "graph_node_counts": replace(
            graph_metrics["graph_node_counts"],
            coverage=coverage,
            categories=(
                CategoryCount(
                    "native",
                    UnitCount(
                        name="graph_node_counts",
                        observation_unit=ObservationUnit.GRAPH_NODE,
                        value=node_total,
                        denominator=2,
                        denominator_unit=ObservationUnit.RECORD,
                        split_scope=coverage.split_scope,
                        evidence_scope=coverage.evidence_scope,
                        provenance=coverage.provenance,
                    ),
                ),
            ),
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
                minimum=base,
                maximum=base + 2,
                mean=rounded_mean,
            ),
        ),
    }
    metrics = tuple(
        replacements.get(metric.metric_id, metric)
        for metric in source.semantic_payload.metrics
    )
    graph_evidence = GraphEvidence(
        status=ComputationStatus.OBSERVED,
        target_free=True,
        **dict(APPROVED_RAW_GRAPH_CONTRACT),
    )
    with pytest.raises(EDAContractError, match="size_extrema_mismatch"):
        RawCorpusEDA(
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
                graph_evidence=graph_evidence,
            ),
        )
