from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    APPROVED_RAW_GRAPH_CONTRACT,
    CategoryCount,
    CompletenessStatus,
    ComputationStatus,
    CorpusId,
    EDAContractError,
    EDAReasonCode,
    EvidenceScope,
    ExecutionMode,
    ExtensionRow,
    GraphEvidence,
    InvariantEvidence,
    InvariantStatus,
    LabelValueType,
    MetricCoverage,
    NumericDistribution,
    ObservationUnit,
    ProjectionAvailabilityCounts,
    ProjectionEvidence,
    ProjectionMappingState,
    QuantilePoint,
    RawCorpusEDA,
    RawCorpusEDAPayload,
    SourceExtension,
    SourceValueIdentity,
    SourceValueKind,
    SplitScope,
    SupervisionEDA,
    SupervisionEDAPayload,
    StructuredWarning,
    TestTargetLockEvidence as TargetLockEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    dumps_report,
    loads_report,
    report_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _extension_coverage(
    *,
    observation_unit: ObservationUnit = ObservationUnit.RECORD,
    denominator: int = 1,
    split_scope: SplitScope = SplitScope.UNSPLIT,
    evidence_scope: EvidenceScope = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = ("fixture",),
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=observation_unit,
        denominator=denominator,
        observed_count=denominator,
        unknown_count=0,
        split_scope=split_scope,
        evidence_scope=evidence_scope,
        provenance=provenance,
    )


def test_approved_raw_graph_contract_is_exported_accepted_and_file_bound() -> None:
    expected_files = {
        "graph_schema": "src/music_critic/graph/relations.py",
        "graph_builder": "src/music_critic/graph/builder.py",
        "feature_registry": "src/music_critic/graph/feature_registry.py",
        "validator": "src/music_critic/graph/validation.py",
    }
    assert set(APPROVED_RAW_GRAPH_CONTRACT) == set(expected_files)

    evidence = GraphEvidence(
        status=ComputationStatus.OBSERVED,
        target_free=True,
        **dict(APPROVED_RAW_GRAPH_CONTRACT),
    )
    for field, relative_path in expected_files.items():
        identity = APPROVED_RAW_GRAPH_CONTRACT[field]
        assert getattr(evidence, field) == identity
        assert identity.fingerprint == hashlib.sha256(
            (REPO_ROOT / relative_path).read_bytes()
        ).hexdigest()


def test_observed_graph_evidence_rejects_forged_or_untyped_contract_identity() -> None:
    approved = dict(APPROVED_RAW_GRAPH_CONTRACT)
    forged = replace(approved["validator"], fingerprint="0" * 64)

    with pytest.raises(EDAContractError, match="contract_unapproved"):
        GraphEvidence(
            status=ComputationStatus.OBSERVED,
            target_free=True,
            **(approved | {"validator": forged}),
        )

    with pytest.raises(EDAContractError, match="identity_invalid"):
        GraphEvidence(
            status=ComputationStatus.OBSERVED,
            target_free=True,
            **(approved | {"validator": "music_critic.graph.validate_raw_graph"}),
        )


def test_projection_availability_round_trip_keeps_native_missing_independent(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.DILEMMADATA]
    task = source.semantic_payload.tasks[0]
    native = task.availability
    projection = task.projection_availability[0]
    assert native is not None
    assert native.missing == 1

    dynamic_projection = replace(
        projection,
        missing=0,
        unsupported=projection.unsupported + projection.missing,
    )
    changed_task = replace(
        task,
        projection_availability=(dynamic_projection,),
    )
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, tasks=(changed_task,)),
    )

    payload = dumps_report(report)
    restored = loads_report(payload)
    assert isinstance(restored, SupervisionEDA)
    restored_task = restored.semantic_payload.tasks[0]
    restored_projection = restored_task.projection_availability[0]
    assert isinstance(restored_projection, ProjectionAvailabilityCounts)
    assert restored_task.availability is not None
    assert restored_task.availability.missing == 1
    assert restored_projection.missing == 0
    assert report_dict(restored) == report_dict(report)
    assert dumps_report(restored) == payload


def test_projection_availability_must_bind_the_native_task(supervision_reports) -> None:
    task = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[0]
    projection = task.projection_availability[0]
    independently_valid_other_task = replace(
        projection,
        source_task_id="dilemmadata.dlc.chord.quality",
        dialect="dlc",
    )

    with pytest.raises(EDAContractError, match="availability_binding_mismatch"):
        replace(
            task,
            projection_availability=(independently_valid_other_task,),
        )


def test_projection_row_requires_projection_availability_aggregate(
    supervision_reports,
) -> None:
    task = supervision_reports[CorpusId.DILEMMADATA].semantic_payload.tasks[0]
    assert task.projections
    with pytest.raises(EDAContractError, match="availability_binding_missing"):
        replace(task, projection_availability=())


@pytest.mark.parametrize(
    ("source_task_id", "source_value", "common_task_identity"),
    (
        (
            "dilemmadata.an.chord.quality",
            "major triad",
            "dilemmadata.common.chord.quality",
        ),
        (
            "dilemmadata.an.chord.inversion",
            "0",
            "dilemmadata.common.chord.inversion",
        ),
    ),
)
@pytest.mark.parametrize(
    "mapping_state",
    (ProjectionMappingState.UNSUPPORTED, ProjectionMappingState.MISSING),
)
def test_known_static_projection_cannot_be_recast_as_missing_or_unsupported(
    source_task_id: str,
    source_value: str,
    common_task_identity: str,
    mapping_state: ProjectionMappingState,
) -> None:
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    identity = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id=source_task_id,
        dialect="an_joint",
        source_value=source_value,
        value_kind=SourceValueKind.SCALAR,
    )

    with pytest.raises(EDAContractError, match="projection"):
        ProjectionEvidence(
            source_value=identity,
            mapping_registry=registry,
            common_task_identity=common_task_identity,
            native_state="available",
            mapping_state=mapping_state,
            projected_value=None,
            provenance=("approved-static-registry",),
        )


@pytest.mark.parametrize(
    ("source_value", "projected_value"),
    (("C", False), ("C#", True)),
)
def test_root_pitch_class_projection_rejects_boolean_values(
    source_value: str,
    projected_value: bool,
) -> None:
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    identity = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id="dilemmadata.an.chord.root",
        dialect="an_joint",
        source_value=source_value,
        value_kind=SourceValueKind.SCALAR,
    )

    with pytest.raises(EDAContractError, match="value_type_invalid"):
        ProjectionEvidence(
            source_value=identity,
            mapping_registry=registry,
            common_task_identity="dilemmadata.common.chord.root_pc",
            native_state="available",
            mapping_state=ProjectionMappingState.EXACT,
            projected_value=projected_value,
            provenance=("approved-static-registry",),
        )


@pytest.mark.parametrize("bad_mode", ([], {"mode": "major"}))
def test_local_key_projection_rejects_unhashable_mode_with_contract_error(
    bad_mode,
) -> None:
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    identity = SourceValueIdentity(
        corpus=CorpusId.DILEMMADATA,
        source_task_id="dilemmadata.an.key.local",
        dialect="an_joint",
        source_value="C major",
        value_kind=SourceValueKind.SCALAR,
    )
    with pytest.raises(EDAContractError, match="projection"):
        ProjectionEvidence(
            source_value=identity,
            mapping_registry=registry,
            common_task_identity="dilemmadata.common.key.local",
            native_state="available",
            mapping_state=ProjectionMappingState.EXACT,
            projected_value={"mode": bad_mode, "tonic_pc": 0},
            provenance=("approved-static-registry",),
        )


def _empty_multilabel_task(supervision_reports):
    source = supervision_reports[CorpusId.HOOKTHEORY].semantic_payload.tasks[0]
    assert source.availability is not None
    count = UnitCount(
        name="empty_multilabel_available_count",
        observation_unit=ObservationUnit.TARGET_ROW,
        value=source.availability.available,
        denominator=source.availability.available,
        denominator_unit=ObservationUnit.TARGET_ROW,
        split_scope=source.split_scope,
        evidence_scope=source.evidence_scope,
        provenance=source.provenance,
    )
    task = replace(
        source,
        label_value_type=LabelValueType.MULTI_LABEL,
        class_support=(),
        empty_multilabel_available_count=count,
    )
    return task, count


def test_empty_multilabel_count_requires_a_unit_count(supervision_reports) -> None:
    task, _ = _empty_multilabel_task(supervision_reports)
    with pytest.raises(EDAContractError, match="count_type_invalid"):
        replace(task, empty_multilabel_available_count=1)


@pytest.mark.parametrize(
    "changes",
    (
        {"name": "empty_rows"},
        {"observation_unit": ObservationUnit.RECORD},
        {"denominator_unit": ObservationUnit.RECORD},
        {"denominator": 2},
        {"split_scope": SplitScope.VALIDATION},
        {"evidence_scope": EvidenceScope.BOUNDED},
        {"provenance": ("different-provenance",)},
    ),
)
def test_empty_multilabel_count_requires_exact_task_binding(
    supervision_reports,
    changes: dict[str, object],
) -> None:
    task, count = _empty_multilabel_task(supervision_reports)
    wrong = replace(count, **changes)
    with pytest.raises(EDAContractError, match="count_binding_mismatch"):
        replace(task, empty_multilabel_available_count=wrong)


def _raw_extension_with_count() -> SourceExtension:
    count = UnitCount(
        name="records_seen",
        observation_unit=ObservationUnit.RECORD,
        value=1,
        denominator=1,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("synthetic-fixture",),
    )
    return SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.safe_summary",
        schema_name="SafeSummary",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("synthetic-fixture",),
        rows=(
            ExtensionRow(
                row_id="summary",
                payload={"kind": "synthetic"},
                counts=(count,),
                coverage=_extension_coverage(
                    provenance=("synthetic-fixture",),
                ),
            ),
        ),
        target_free=True,
    )


@pytest.mark.parametrize(
    "leak_location",
    ("namespace", "schema_name", "provenance", "count_name", "count_provenance"),
)
def test_raw_extension_rejects_target_semantics_outside_payload(
    raw_reports,
    leak_location: str,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = _raw_extension_with_count()
    if leak_location == "namespace":
        extension = replace(extension, namespace="pdmx.target_summary")
    elif leak_location == "schema_name":
        extension = replace(extension, schema_name="GoldSummary")
    elif leak_location == "provenance":
        row = extension.rows[0]
        provenance = ("label-source",)
        extension = replace(
            extension,
            provenance=provenance,
            rows=(
                replace(
                    row,
                    coverage=replace(row.coverage, provenance=provenance),
                    counts=tuple(
                        replace(count, provenance=provenance)
                        for count in row.counts
                    ),
                ),
            ),
        )
    else:
        row = extension.rows[0]
        count = row.counts[0]
        if leak_location == "count_name":
            count = replace(count, name="target_count")
            row = replace(row, counts=(count,))
        else:
            provenance = ("annotation-source",)
            count = replace(count, provenance=provenance)
            row = replace(
                row,
                coverage=replace(row.coverage, provenance=provenance),
                counts=(count,),
            )
        extension = replace(
            extension,
            provenance=(
                provenance
                if leak_location == "count_provenance"
                else extension.provenance
            ),
            rows=(row,),
        )

    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                extensions=(extension,),
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "event-count",
        "eventCount",
        "n_records",
        "total_records",
        "recordcount",
        "numrecords",
        "totalrecords",
        "numberofrows",
        "record_total",
        "recordtotal",
        "event_total",
        "label_total",
        "sample_size",
        "dataset_size",
        "vocabulary_size",
        "class_cardinality",
        "label_cardinality",
        "num_examples",
        "number_of_examples",
        "n_examples",
        "entrycount",
        "n_entries",
        "number_of_entries",
        "total_entries",
        "entry_total",
        "segmentcount",
        "nsegments",
        "number_of_segments",
        "occurrences",
        "frequency",
        "frequencies",
        "class_frequency",
        "event_frequency",
        "row_frequency",
        "n_observations",
        "incidences",
        "cardinality",
        "row_cardinality",
        "item_cardinality",
        "tally",
        "record_tally",
        "record_quantity",
        "population_total",
        "event_cardinality",
        "work_cardinality",
        "file_cardinality",
        "collection_cardinality",
        "set_cardinality",
        "vocab_size",
        "vocab_cardinality",
        "record_population_size",
        "unique_record_total",
        "accepted_record_total",
        "recordset_size",
        "total_population",
        "size_of_population",
        "population_length",
        "dataset_length",
        "collection_length",
        "vocab_length",
        "total_dataset",
        "dataset_total",
        "total_collection",
        "collection_total",
        "total_vocab",
        "vocab_total",
        "total_vocabulary",
        "vocabulary_total",
        "attempts",
        "target_access_attempts",
        "descriptor_resolution_attempts",
        "collisions",
        "raw_identity_collisions",
        "identity_collisions",
        "datasets",
        "collections",
        "corpora",
        "vocabularies",
        "nobs",
        "population",
        "dataset_population",
        "sample_population",
        "event_multiplicity",
    ),
)
def test_extension_payload_rejects_untyped_count_aliases(field: str) -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            row_id="untyped",
            payload={field: 1},
            coverage=_extension_coverage(),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"records": 10},
        {"files": 4},
        {"events": 99},
        {"support_by_label": {"major": 7}},
        {"buckets": {"ok": 9, "bad": 1}},
    ),
)
def test_extension_payload_rejects_untyped_population_counts(payload) -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            row_id="untyped",
            payload=payload,
            coverage=_extension_coverage(),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"catalog_number": "BWV 846"},
        {"movement_number": "I"},
        {"track_number": 1},
        {"take_number": 2},
    ),
)
def test_source_native_ordinals_are_not_misclassified_as_counts(payload) -> None:
    row = ExtensionRow(
        row_id="ordinal",
        payload=payload,
        coverage=_extension_coverage(),
    )
    assert dict(row.payload) == payload


@pytest.mark.parametrize(
    "payload",
    (
        {"frequency_hz": 440.0},
        {"sampling_frequency": 44100},
        {"distribution": {"mean": 0.5, "stddev": 0.1}},
        {"confidence_distribution": {"p50": 0.8}},
        {"support_probability": 0.9},
        {"metric_position": {"numerator": 1, "denominator": 4}},
        {"exact_time": {"num": 1, "denominator": 4}},
        {"tempo_ratio": {"numerator": 3, "denominator": 2}},
        {"item_number": 7},
        {"item_no": 7},
        {"duration_total": 4.5},
        {"file_size_bytes": 4096},
        {"graph_size": 12},
        {"dataset_size_bytes": 8192},
        {"dataset_size": {"value": 8192, "unit": "bytes"}},
        {"frequency_response_hz": [20, 1000, 20000]},
        {"sampling_frequency_hz": [44100, 48000]},
        {"audio_frequency_hz": {"low": 20, "high": 20000}},
        {"pitch_support_midi_note_numbers": [0, 4, 7]},
        {"observation": {"observed": 440.0, "expected": 442.0}},
    ),
)
def test_source_measurements_and_exact_ratios_are_not_counts(payload) -> None:
    row = ExtensionRow(
        row_id="measurement",
        payload=payload,
        coverage=_extension_coverage(),
    )
    assert set(row.payload) == set(payload)


def test_typed_numeric_distribution_is_valid_extension_measurement() -> None:
    distribution = NumericDistribution(
        measurement_unit="beats_per_minute",
        minimum=80,
        maximum=120,
        mean=100.0,
        quantiles=(QuantilePoint(1, 2, 100),),
    )
    row = ExtensionRow(
        row_id="tempo_summary",
        payload={"source_summary": distribution},
        coverage=_extension_coverage(),
    )
    assert row.payload["source_summary"]["quantiles"][0]["numerator"] == 1


def test_exact_ratio_cannot_hide_an_untyped_count_in_value() -> None:
    with pytest.raises(EDAContractError, match="untyped_count"):
        ExtensionRow(
            row_id="hidden",
            payload={
                "ratio": {
                    "numerator": 1,
                    "denominator": 2,
                    "value": {"native_event_count": 99},
                }
            },
            coverage=_extension_coverage(),
        )


@pytest.mark.parametrize(
    "work_unit",
    (ObservationUnit.LOGICAL_WORK, ObservationUnit.CANONICAL_WORK),
)
def test_observed_extension_work_count_requires_work_identity(work_unit) -> None:
    count = UnitCount(
        name="works_seen",
        observation_unit=work_unit,
        value=1,
        denominator=1,
        denominator_unit=work_unit,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("synthetic-fixture",),
    )
    with pytest.raises(EDAContractError, match="work_identity_unproven"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.work_summary",
            schema_name="WorkSummary",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("synthetic-fixture",),
            rows=(
                ExtensionRow(
                    row_id="works",
                    payload={"kind": "synthetic"},
                    counts=(count,),
                    coverage=_extension_coverage(
                        observation_unit=work_unit,
                        provenance=("synthetic-fixture",),
                    ),
                ),
            ),
            target_free=True,
        )


def test_same_extension_namespace_is_allowed_per_train_validation_split(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    train_extension = source.semantic_payload.extensions[0]
    validation_extension = replace(
        train_extension,
        split_scope=SplitScope.VALIDATION,
        rows=tuple(
            replace(
                row,
                coverage=replace(
                    row.coverage,
                    split_scope=SplitScope.VALIDATION,
                ),
            )
            for row in train_extension.rows
        ),
    )
    envelope = replace(source.envelope, split_scope=SplitScope.TRAIN_VALIDATION)
    first = SupervisionEDA(
        envelope=envelope,
        semantic_payload=replace(
            source.semantic_payload,
            extensions=(train_extension, validation_extension),
        ),
    )
    second = SupervisionEDA(
        envelope=envelope,
        semantic_payload=replace(
            source.semantic_payload,
            extensions=(validation_extension, train_extension),
        ),
    )

    assert [item.split_scope for item in first.semantic_payload.extensions] == [
        SplitScope.TRAIN,
        SplitScope.VALIDATION,
    ]
    assert dumps_report(first) == dumps_report(second)
    restored = loads_report(dumps_report(first))
    assert report_dict(restored) == report_dict(first)


def test_directly_built_reports_have_strict_round_trips(
    raw_reports,
    supervision_reports,
) -> None:
    raw_source = raw_reports[CorpusId.PDMX]
    supervision_source = supervision_reports[CorpusId.HOOKTHEORY]
    reports = (
        RawCorpusEDA(
            envelope=raw_source.envelope,
            semantic_payload=raw_source.semantic_payload,
        ),
        SupervisionEDA(
            envelope=supervision_source.envelope,
            semantic_payload=supervision_source.semantic_payload,
        ),
    )

    for report in reports:
        payload = dumps_report(report)
        restored = loads_report(payload)
        assert type(restored) is type(report)
        assert report_dict(restored) == report_dict(report)
        assert dumps_report(restored) == payload


def test_direct_constructors_reject_nested_type_misuse_cleanly(
    raw_reports,
    supervision_reports,
) -> None:
    raw = raw_reports[CorpusId.PDMX]
    supervision = supervision_reports[CorpusId.HOOKTHEORY]

    with pytest.raises(EDAContractError, match="type_invalid"):
        RawCorpusEDA(
            envelope={"schema_name": "RawCorpusEDA"},
            semantic_payload=raw.semantic_payload,
        )

    with pytest.raises(EDAContractError, match="type_invalid"):
        SupervisionEDAPayload(
            tasks=({"source_task_id": "not-a-task-object"},),
            test_lock=supervision.semantic_payload.test_lock,
        )

    with pytest.raises(EDAContractError, match="type_invalid"):
        RawCorpusEDAPayload(
            metrics=raw.semantic_payload.metrics,
            graph_evidence={"status": "not_computed"},
        )


@pytest.mark.parametrize(
    ("scope", "completeness", "task_status"),
    (
        (EvidenceScope.UNKNOWN, CompletenessStatus.UNKNOWN, ComputationStatus.UNKNOWN),
        (
            EvidenceScope.UNAVAILABLE,
            CompletenessStatus.UNAVAILABLE,
            ComputationStatus.NOT_COMPUTED,
        ),
    ),
)
def test_manifest_free_non_evidence_reports_are_truthful_and_round_trip(
    raw_reports,
    supervision_reports,
    scope,
    completeness,
    task_status,
) -> None:
    reason = UnavailableReason(
        code=f"fixture.{scope.value}",
        status=task_status,
        provenance=("synthetic-non-evidence",),
    )
    raw_source = raw_reports[CorpusId.PDMX]
    raw_metrics = tuple(
        replace(
            metric,
            coverage=replace(
                metric.coverage,
                denominator=None,
                observed_count=None,
                unknown_count=None,
                evidence_scope=scope,
                status=task_status,
                reason_code=reason.code,
            ),
            count=None,
            numeric=None,
            categories=(),
        )
        for metric in raw_source.semantic_payload.metrics
    )
    raw = RawCorpusEDA(
        envelope=replace(
            raw_source.envelope,
            evidence_scope=scope,
            execution_mode=ExecutionMode.NOT_EXECUTED,
            completeness_status=completeness,
            input_manifests=(),
            observation_units=(ObservationUnit.RECORD,),
            unavailable_reasons=(reason,),
        ),
        semantic_payload=RawCorpusEDAPayload(
            metrics=raw_metrics,
            graph_evidence=GraphEvidence(
                status=task_status,
                target_free=None,
                reason_code=reason.code,
            ),
        ),
    )

    supervision_source = supervision_reports[CorpusId.HOOKTHEORY]
    source_task = supervision_source.semantic_payload.tasks[0]
    task = replace(
        source_task,
        evidence_scope=scope,
        status=task_status,
        availability=None,
        class_support=(),
        projection_availability=(),
        projections=(),
        reason_code=reason.code,
    )
    lock = TargetLockEvidence.not_executed(
        evidence_scope=scope,
        provenance=("synthetic-non-evidence",),
        reason_code=reason.code,
    )
    supervision = SupervisionEDA(
        envelope=replace(
            supervision_source.envelope,
            evidence_scope=scope,
            execution_mode=ExecutionMode.NOT_EXECUTED,
            completeness_status=completeness,
            input_manifests=(),
            observation_units=(
                ObservationUnit.RECORD,
                ObservationUnit.SPLIT_ASSIGNMENT,
                ObservationUnit.TARGET_ACCESS_ATTEMPT,
                ObservationUnit.TARGET_ROW,
            ),
            unavailable_reasons=(reason,),
        ),
        semantic_payload=SupervisionEDAPayload(
            tasks=(task,),
            test_lock=lock,
        ),
    )

    for report in (raw, supervision):
        restored = loads_report(dumps_report(report))
        assert report_dict(restored) == report_dict(report)


@pytest.mark.parametrize(
    ("payload", "provenance"),
    (
        ({"source_path": "/home/user/corpus/raw.mid"}, ("fixture",)),
        ({"files": {"/home/user/corpus/raw.mid": "ok"}}, ("fixture",)),
        ({"machine_name": "private-host"}, ("fixture",)),
        ({"hostName": "private-host"}, ("fixture",)),
        ({"time_stamp": "2026-09-03T12:00:00Z"}, ("fixture",)),
        ({"wallclock_seconds": 1.5}, ("fixture",)),
        ({"durationSeconds": 1.5}, ("fixture",)),
        ({"duration-seconds": 1.5}, ("fixture",)),
        ({"duration seconds": 1.5}, ("fixture",)),
        ({"run_duration": 1.5}, ("fixture",)),
        ({"runDurationSeconds": 1.5}, ("fixture",)),
        ({"execution_time": 1.5}, ("fixture",)),
        ({"processing_seconds": 1.5}, ("fixture",)),
        ({"elapsedseconds": 1.5}, ("fixture",)),
        ({"elapsedmilliseconds": 1.5}, ("fixture",)),
        ({"wallclockmilliseconds": 1.5}, ("fixture",)),
        ({"processduration": 1.5}, ("fixture",)),
        ({"processtime": 1.5}, ("fixture",)),
        ({"processingduration": 1.5}, ("fixture",)),
        ({"executionduration": 1.5}, ("fixture",)),
        ({"rundurationms": 1.5}, ("fixture",)),
        ({"runelapsedseconds": 1.5}, ("fixture",)),
        ({"runtimehours": 1.5}, ("fixture",)),
        ({"hostidentifier": "private-host"}, ("fixture",)),
        ({"machinehostname": "private-host"}, ("fixture",)),
        ({"runstartedat": "2026-09-03T12:00:00Z"}, ("fixture",)),
        ({"runendtime": "2026-09-03T12:00:00Z"}, ("fixture",)),
        ({"absolutePath": "fixtures/example.mid"}, ("fixture",)),
        ({"source_ref": r"from \\server\share\file.mid"}, ("fixture",)),
        ({"source_ref": r"source=\\server\share\file.mid"}, ("fixture",)),
        ({"elapsed_ms": 123}, ("fixture",)),
        ({"kind": "raw"}, ("/home/user/corpus/raw.mid",)),
    ),
)
def test_operational_aliases_and_absolute_paths_cannot_enter_semantics(
    raw_reports,
    payload,
    provenance,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.source_summary",
        schema_name="SourceSummary",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=provenance,
        rows=(
            ExtensionRow(
                "source",
                payload,
                coverage=_extension_coverage(provenance=provenance),
            ),
        ),
        target_free=True,
    )
    with pytest.raises(EDAContractError, match="operational_metadata"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    "message",
    (
        "source was read from /home/user/private.mid",
        "source:/home/user/private.mid",
        "path[/tmp/private.mid]",
        "failed->/var/data/private.mid",
    ),
)
def test_absolute_path_in_warning_is_not_semantic(raw_reports, message) -> None:
    source = raw_reports[CorpusId.PDMX]
    warning = StructuredWarning(
        code="fixture.source_warning",
        message=message,
        provenance=("fixture",),
    )
    with pytest.raises(EDAContractError, match="absolute_path_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, warnings=(warning,)),
            semantic_payload=source.semantic_payload,
        )


def test_relative_source_path_and_source_timestamp_are_valid_semantics(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.harmonic_path",
        schema_name="HarmonicPath",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "path",
                {
                    "harmonic_path": ["I", "V"],
                    "secondary_dominant": "V/ii",
                    "source_event_timestamp": "measure-1-beat-1",
                    "source_path": "fixtures/example.mid",
                    "source_url": "https://example.invalid/example.mid",
                },
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    report = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    assert loads_report(dumps_report(report)).semantic_fingerprint == (
        report.semantic_fingerprint
    )


def test_extension_payload_is_deeply_frozen_after_fingerprinting(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    caller_values = ["raw"]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.nested_summary",
        schema_name="NestedSummary",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "nested",
                {"nested": {"values": caller_values}},
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    report = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    fingerprint = report.semantic_fingerprint
    caller_values.append("gold_target")
    stored_nested = report.semantic_payload.extensions[0].rows[0].payload["nested"]
    assert stored_nested["values"] == ("raw",)
    with pytest.raises(AttributeError):
        stored_nested["values"].append("gold_target")
    with pytest.raises(TypeError):
        stored_nested["other"] = "gold_target"
    assert report.semantic_fingerprint == fingerprint
    assert report_dict(report)["semantic_payload"]["extensions"][0]["rows"][0][
        "payload"
    ]["nested"]["values"] == ["raw"]


def test_projected_values_are_deeply_frozen_after_fingerprinting(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.DILEMMADATA]
    task = source.semantic_payload.tasks[0]
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    caller_value = [0, 4, 7]
    projection = ProjectionEvidence(
        source_value=task.class_support[0].source_value,
        mapping_registry=registry,
        common_task_identity="dilemmadata.common.chord.pitch_class_set",
        native_state="available",
        mapping_state=ProjectionMappingState.EXACT,
        projected_value=caller_value,
        provenance=("fixture",),
    )
    projection_availability = replace(
        task.projection_availability[0],
        common_task_identity="dilemmadata.common.chord.pitch_class_set",
    )
    changed_task = replace(
        task,
        projections=(*task.projections, projection),
        projection_availability=(
            *task.projection_availability,
            projection_availability,
        ),
    )
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, tasks=(changed_task,)),
    )
    fingerprint = report.semantic_fingerprint
    caller_value.append(11)
    frozen = next(
        item.projected_value
        for item in report.semantic_payload.tasks[0].projections
        if item.common_task_identity
        == "dilemmadata.common.chord.pitch_class_set"
    )
    assert frozen == (0, 4, 7)
    with pytest.raises(AttributeError):
        frozen.append(11)
    assert report.semantic_fingerprint == fingerprint
    assert loads_report(dumps_report(report)).semantic_fingerprint == fingerprint


@pytest.mark.parametrize("field", ("row_id", "work_identity"))
def test_raw_extension_rejects_target_tokens_in_all_identity_fields(
    raw_reports,
    field,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.summary",
        schema_name="Summary",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "summary",
                {"kind": "raw"},
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    if field == "row_id":
        extension = replace(
            extension,
            rows=(replace(extension.rows[0], row_id="gold_target_labels"),),
        )
    else:
        extension = replace(
            extension,
            work_identity=VersionedIdentity(
                "pdmx.gold_target_work", "1.0.0", "1" * 64
            ),
        )
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize("field", ("identity", "repository_relative_path"))
def test_raw_manifest_rejects_disguised_target_binding(raw_reports, field) -> None:
    source = raw_reports[CorpusId.PDMX]
    manifest = source.envelope.input_manifests[0]
    if field == "identity":
        manifest = replace(
            manifest,
            identity=replace(manifest.identity, identity="pdmx.gold_labels"),
        )
    else:
        manifest = replace(
            manifest,
            repository_relative_path="targets/gold.json",
        )
    with pytest.raises(EDAContractError, match="target_manifest_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, input_manifests=(manifest,)),
            semantic_payload=source.semantic_payload,
        )


def test_raw_count_summary_name_is_bound_to_metric_id(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "accepted_records"
    )
    assert metric.count is not None
    with pytest.raises(EDAContractError, match="count_binding_mismatch"):
        replace(metric, count=replace(metric.count, name="gold_label_count"))


def test_raw_category_count_name_is_deterministically_bound_to_metric_id(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "conversion_outcomes"
    )
    coverage = replace(
        metric.coverage,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        status=ComputationStatus.OBSERVED,
        reason_code=None,
    )
    count = UnitCount(
        name="unbound_name",
        observation_unit=ObservationUnit.RECORD,
        value=1,
        denominator=1,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=coverage.provenance,
    )
    with pytest.raises(EDAContractError, match="count_binding_mismatch"):
        replace(
            metric,
            coverage=coverage,
            categories=(CategoryCount("accepted", count),),
        )


def test_raw_common_metric_rejects_target_provenance(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    metric = next(
        item
        for item in source.semantic_payload.metrics
        if item.metric_id == "accepted_records"
    )
    assert metric.count is not None
    changed = replace(
        metric,
        coverage=replace(metric.coverage, provenance=("target-sidecar",)),
        count=replace(metric.count, provenance=("target-sidecar",)),
    )
    metrics = tuple(
        changed if item.metric_id == changed.metric_id else item
        for item in source.semantic_payload.metrics
    )
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, metrics=metrics),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"TARGETLabels": ["x"]},
        {"ground_truth": ["C"]},
        {"groundTruth": ["C"]},
        {"answer_key": ["C"]},
    ),
)
def test_raw_extension_rejects_target_aliases(raw_reports, payload) -> None:
    source = raw_reports[CorpusId.PDMX]
    extension = SourceExtension(
        corpus=CorpusId.PDMX,
        namespace="pdmx.alias_probe",
        schema_name="AliasProbe",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "probe",
                payload,
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


def test_graph_unavailable_accepts_canonical_target_free_unproven_reason(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    graph_metric_ids = {
        "graph_node_counts",
        "graph_edge_counts",
        "graph_size_distribution",
    }
    metrics = tuple(
        replace(
            metric,
            coverage=replace(
                metric.coverage,
                reason_code=EDAReasonCode.TARGET_FREE_UNPROVEN,
            ),
        )
        if metric.metric_id in graph_metric_ids
        else metric
        for metric in source.semantic_payload.metrics
    )
    reason = UnavailableReason(
        code=EDAReasonCode.TARGET_FREE_UNPROVEN,
        status=ComputationStatus.NOT_COMPUTED,
        provenance=("raw-contract",),
    )
    report = RawCorpusEDA(
        envelope=replace(source.envelope, unavailable_reasons=(reason,)),
        semantic_payload=replace(
            source.semantic_payload,
            metrics=metrics,
            graph_evidence=replace(
                source.semantic_payload.graph_evidence,
                reason_code=EDAReasonCode.TARGET_FREE_UNPROVEN,
            ),
        ),
    )
    assert report.semantic_payload.graph_evidence.reason_code == (
        EDAReasonCode.TARGET_FREE_UNPROVEN
    )


def test_supervision_extension_rejects_acronym_pascal_test_alias(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "probe",
                {"TESTTargets": [1]},
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    with pytest.raises(EDAContractError, match="extension_test_field_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


@pytest.mark.parametrize(
    ("collection", "row"),
    (
        (
            "warnings",
            StructuredWarning(
                code="gold.class_distribution",
                message="gold labels summary",
                provenance=("target-sidecar",),
            ),
        ),
        (
            "invariants",
            InvariantEvidence(
                code="raw.supervision_absent",
                status=InvariantStatus.PASSED,
                provenance=("fixture",),
            ),
        ),
        (
            "unavailable_reasons",
            UnavailableReason(
                code="raw.metric_unavailable",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=("gold-label-index",),
                detail="class support was not scanned",
            ),
        ),
    ),
)
def test_raw_envelope_evidence_rejects_target_derived_channels(
    raw_reports,
    collection,
    row,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        RawCorpusEDA(
            envelope=replace(source.envelope, **{collection: (row,)}),
            semantic_payload=source.semantic_payload,
        )


def test_raw_unavailable_reason_preserves_typed_target_free_attestation(
    raw_reports,
) -> None:
    source = raw_reports[CorpusId.PDMX]
    reason = UnavailableReason(
        code=EDAReasonCode.TARGET_FREE_UNPROVEN,
        status=ComputationStatus.NOT_COMPUTED,
        provenance=("raw-contract",),
        detail="raw graph contract was not established",
    )
    report = RawCorpusEDA(
        envelope=replace(source.envelope, unavailable_reasons=(reason,)),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.unavailable_reasons == (reason,)


@pytest.mark.parametrize(
    ("collection", "row"),
    (
        (
            "warnings",
            StructuredWarning(
                code="test.class_distribution",
                message="TEST class support summary",
                provenance=("test-targets",),
            ),
        ),
        (
            "invariants",
            InvariantEvidence(
                code="test.targets_absent",
                status=InvariantStatus.PASSED,
                provenance=("fixture",),
            ),
        ),
        (
            "unavailable_reasons",
            UnavailableReason(
                code="source.metric_unavailable",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=("fixture",),
                detail="TEST target support was not emitted",
            ),
        ),
    ),
)
def test_supervision_envelope_evidence_rejects_test_channels(
    supervision_reports,
    collection,
    row,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    with pytest.raises(EDAContractError, match="envelope_test_field_forbidden"):
        SupervisionEDA(
            envelope=replace(source.envelope, **{collection: (row,)}),
            semantic_payload=source.semantic_payload,
        )


def test_supervision_unavailable_reason_preserves_typed_test_lock(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    reason = UnavailableReason(
        code=EDAReasonCode.TEST_TARGETS_LOCKED,
        status=ComputationStatus.LOCKED,
        provenance=("target-access-guard",),
        detail="held behind the assignment gate",
    )
    report = SupervisionEDA(
        envelope=replace(source.envelope, unavailable_reasons=(reason,)),
        semantic_payload=source.semantic_payload,
    )
    assert report.envelope.unavailable_reasons == (reason,)


def test_supervision_extension_rows_require_observed_test_gate(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    task = replace(
        source.semantic_payload.tasks[0],
        status=ComputationStatus.NOT_COMPUTED,
        availability=None,
        class_support=(),
        projection_availability=(),
        projections=(),
        empty_multilabel_available_count=None,
        reason_code="eda.metric_not_computed",
    )
    existing_lock = source.semantic_payload.test_lock
    locked = TargetLockEvidence.not_executed(
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("target-access-guard",),
        reason_code=EDAReasonCode.TEST_TARGETS_LOCKED,
        assignment_manifest_fingerprint=(
            existing_lock.assignment_manifest_fingerprint
        ),
        test_assignment_denominator=(
            existing_lock.test_assignment_count.denominator
        ),
    )
    with pytest.raises(EDAContractError, match="observed_evidence_without_guard"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(
                source.semantic_payload,
                tasks=(task,),
                test_lock=locked,
            ),
        )


@pytest.mark.parametrize("field", ("namespace", "schema_name", "row_id", "work_identity"))
def test_supervision_extension_rejects_test_tokens_in_identity_fields(
    supervision_reports,
    field,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = source.semantic_payload.extensions[0]
    if field == "namespace":
        extension = replace(extension, namespace="hooktheory.test_summary")
    elif field == "schema_name":
        extension = replace(extension, schema_name="TestSummary")
    elif field == "row_id":
        extension = replace(
            extension,
            rows=(replace(extension.rows[0], row_id="test_classes"),),
        )
    else:
        extension = replace(
            extension,
            work_identity=VersionedIdentity(
                "hooktheory.test_work", "1.0.0", "2" * 64
            ),
        )
    with pytest.raises(EDAContractError, match="extension_test_field_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
        )


def test_supervision_manifest_and_task_provenance_reject_test_tokens(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    target_manifest = next(
        item for item in source.envelope.input_manifests if not item.target_free
    )
    hidden = replace(
        target_manifest,
        identity=replace(target_manifest.identity, identity="hooktheory.test_targets"),
    )
    retained = tuple(
        hidden if item is target_manifest else item
        for item in source.envelope.input_manifests
    )
    with pytest.raises(EDAContractError, match="manifest_test_field_forbidden"):
        SupervisionEDA(
            envelope=replace(source.envelope, input_manifests=retained),
            semantic_payload=source.semantic_payload,
        )

    task = replace(
        source.semantic_payload.tasks[0], provenance=("test-target-source",)
    )
    with pytest.raises(EDAContractError, match="task_test_field_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, tasks=(task,)),
        )


def test_plural_tests_token_is_locked_in_manifests_and_task_provenance(
    supervision_reports,
) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    target_manifest = next(
        item for item in source.envelope.input_manifests if not item.target_free
    )
    hidden = replace(
        target_manifest,
        role="tests_target_sidecar",
        identity=replace(
            target_manifest.identity,
            identity="hooktheory.tests_targets",
        ),
        repository_relative_path="targets/tests_labels.json",
    )
    retained = tuple(
        hidden if item is target_manifest else item
        for item in source.envelope.input_manifests
    )
    with pytest.raises(EDAContractError, match="manifest_test_field_forbidden"):
        SupervisionEDA(
            envelope=replace(source.envelope, input_manifests=retained),
            semantic_payload=source.semantic_payload,
        )

    task = replace(
        source.semantic_payload.tasks[0],
        provenance=("tests_target_distribution",),
    )
    with pytest.raises(EDAContractError, match="task_test_field_forbidden"):
        SupervisionEDA(
            envelope=source.envelope,
            semantic_payload=replace(source.semantic_payload, tasks=(task,)),
        )


def test_non_test_extension_scope_metadata_remains_allowed(supervision_reports) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    extension = replace(
        source.semantic_payload.extensions[0],
        rows=(
            ExtensionRow(
                "training_scope",
                {"partition": "train", "scope": "source-release"},
                coverage=source.semantic_payload.extensions[0].rows[0].coverage,
            ),
        ),
    )
    report = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(extension,)),
    )
    assert report.semantic_payload.extensions[0].rows[0].payload["partition"] == "train"
