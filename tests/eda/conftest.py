from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pytest

from music_critic.data.serialization import canonical_json_sha256
from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    AvailabilityCounts,
    CategoryCount,
    ClassSupport,
    CompletenessStatus,
    ComputationStatus,
    CorpusId,
    EvidenceScope,
    ExecutionMode,
    ExtensionRow,
    GraphEvidence,
    InputManifestRef,
    LabelValueType,
    MetricCoverage,
    NumericDistribution,
    ObservationUnit,
    ProjectionAvailabilityCounts,
    ProjectionEvidence,
    ProjectionMappingState,
    RAW_CORPUS_EDA_SCHEMA_NAME,
    RAW_CORPUS_EDA_SCHEMA_VERSION,
    RAW_METRIC_CATALOG,
    RawCorpusEDA,
    RawCorpusEDAPayload,
    RawMetricEvidence,
    ReportEnvelope,
    ReportKind,
    SourceExtension,
    SourceValueIdentity,
    SourceValueKind,
    SplitScope,
    SUPERVISION_EDA_SCHEMA_NAME,
    SUPERVISION_EDA_SCHEMA_VERSION,
    SupervisionEDA,
    SupervisionEDAPayload,
    TaskFamilyEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    load_supervision_train_validation_only,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/multisource/eda_contract_cases.json"


def identity(name: str, value: object) -> VersionedIdentity:
    return VersionedIdentity(
        identity=name,
        version="1.0.0",
        fingerprint=canonical_json_sha256(value),
    )


def coverage(
    *,
    observed: bool,
    denominator: int | None,
    scope: EvidenceScope = EvidenceScope.FIXTURE,
    split: SplitScope = SplitScope.UNSPLIT,
    provenance: tuple[str, ...] = ("synthetic-fixture",),
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=ObservationUnit.RECORD,
        denominator=denominator,
        observed_count=denominator if observed else None,
        unknown_count=0 if observed else None,
        split_scope=split,
        evidence_scope=scope,
        provenance=provenance,
        status=(ComputationStatus.OBSERVED if observed else ComputationStatus.NOT_COMPUTED),
        reason_code=None if observed else "fixture.metric_not_computed",
    )


def raw_report(case: dict[str, Any], *, operational: dict[str, object] | None = None) -> RawCorpusEDA:
    corpus = CorpusId(case["corpus"])
    records = case["records"]
    denominator = len(records)
    count_values = {
        "accepted_records": denominator,
        "discovered_records": denominator,
        "quarantined_records": 0,
    }
    numeric_values = {
        "duration": [row["duration_qn"] for row in records],
        "notes": [row["note_count"] for row in records],
    }
    metrics: list[RawMetricEvidence] = []
    for metric_id, spec in RAW_METRIC_CATALOG.items():
        if metric_id in count_values:
            metric_coverage = coverage(observed=True, denominator=denominator)
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=metric_coverage,
                    count=UnitCount(
                        name=metric_id,
                        observation_unit=spec.value_unit,
                        value=count_values[metric_id],
                        denominator=denominator,
                        denominator_unit=ObservationUnit.RECORD,
                        split_scope=SplitScope.UNSPLIT,
                        evidence_scope=EvidenceScope.FIXTURE,
                        provenance=("synthetic-fixture",),
                    ),
                )
            )
        elif metric_id in numeric_values:
            values = numeric_values[metric_id]
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=coverage(observed=True, denominator=denominator),
                    numeric=NumericDistribution(
                        measurement_unit=spec.measurement_unit,
                        minimum=min(values),
                        maximum=max(values),
                        mean=sum(values) / len(values),
                    ),
                )
            )
        else:
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=coverage(observed=False, denominator=None),
                )
            )
    source = identity(f"{corpus.value}.synthetic_release", case)
    producer = identity(f"{corpus.value}.synthetic_eda_adapter", {"adapter": corpus.value})
    manifest = InputManifestRef(
        role="raw_projection",
        identity=identity(f"{corpus.value}.synthetic_raw_manifest", records),
        target_free=True,
        repository_relative_path="tests/fixtures/multisource/eda_contract_cases.json",
    )
    envelope = ReportEnvelope(
        schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
        schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.RAW_CORPUS,
        corpus=corpus,
        source_identity=source,
        producer_identity=producer,
        repository_commit="659bde251fee6c836564a45aea854b6abcac9fe0",
        evidence_scope=EvidenceScope.FIXTURE,
        execution_mode=ExecutionMode.SYNTHETIC_FIXTURE,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.UNSPLIT,
        observation_units=(ObservationUnit.RECORD,),
        input_manifests=(manifest,),
        unavailable_reasons=(
            UnavailableReason(
                code="fixture.graph_not_built",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=("synthetic-fixture",),
            ),
        ),
        operational_metadata=operational or {},
    )
    return RawCorpusEDA(
        envelope=envelope,
        semantic_payload=RawCorpusEDAPayload(
            metrics=tuple(reversed(metrics)),
            graph_evidence=GraphEvidence(
                status=ComputationStatus.NOT_COMPUTED,
                target_free=None,
                reason_code="fixture.graph_not_built",
            ),
        ),
    )


def supervision_report(
    case: dict[str, Any], *, operational: dict[str, object] | None = None
) -> SupervisionEDA:
    corpus = CorpusId(case["corpus"])
    task_data = case["task"]
    assert task_data is not None
    assignment_fingerprint = canonical_json_sha256(
        {
            "corpus": corpus.value,
            "assignments": [
                {"record_id": "fixture-train", "split": "train"},
                {"record_id": "fixture-test", "split": "test"},
            ],
        }
    )
    _, test_lock = load_supervision_train_validation_only(
        corpus,
        (
            {
                "assignment_manifest_fingerprint": assignment_fingerprint,
                "corpus": corpus.value,
                "record_id": "fixture-train",
                "split": "train",
                "target_free": True,
            },
            {"split": "test", "target_path": "/must-not-be-read"},
        ),
        resolve_descriptor=lambda record_id, split: (record_id, split.value),
        load_target=lambda descriptor, split: descriptor,
    )
    state_counts = Counter(task_data["states"])
    available_values = Counter(task_data["available_values"])
    available = state_counts["available"]
    record_denominator = len(case["records"])
    support: list[ClassSupport] = []
    for value, count in reversed(sorted(available_values.items())):
        source_value = SourceValueIdentity(
            corpus=corpus,
            source_task_id=task_data["source_task_id"],
            dialect=task_data["dialect"],
            source_value=value,
            value_kind=SourceValueKind.SCALAR,
        )
        if task_data["work_identity_proven"]:
            work_count = UnitCount(
                name="unique_work_count",
                observation_unit=ObservationUnit.LOGICAL_WORK,
                value=1,
                denominator=record_denominator,
                denominator_unit=ObservationUnit.LOGICAL_WORK,
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=("synthetic-target-fixture",),
            )
        else:
            work_count = UnitCount(
                name="unique_work_count",
                observation_unit=ObservationUnit.LOGICAL_WORK,
                value=None,
                denominator=None,
                denominator_unit=ObservationUnit.LOGICAL_WORK,
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=("synthetic-target-fixture",),
                status=ComputationStatus.NOT_APPLICABLE,
                reason_code="work_identity.unproven",
            )
        support.append(
            ClassSupport(
                source_value=source_value,
                occurrence_count=UnitCount(
                    name="occurrence_count",
                    observation_unit=ObservationUnit.LABEL_OCCURRENCE,
                    value=count,
                    denominator=available,
                    denominator_unit=ObservationUnit.TARGET_ROW,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.FIXTURE,
                    provenance=("synthetic-target-fixture",),
                ),
                unique_record_count=UnitCount(
                    name="unique_record_count",
                    observation_unit=ObservationUnit.RECORD,
                    value=1,
                    denominator=record_denominator,
                    denominator_unit=ObservationUnit.RECORD,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.FIXTURE,
                    provenance=("synthetic-target-fixture",),
                ),
                unique_work_count=work_count,
            )
        )
    projections: tuple[ProjectionEvidence, ...] = ()
    projection_availability: tuple[ProjectionAvailabilityCounts, ...] = ()
    if corpus == CorpusId.DILEMMADATA:
        registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
        projections = tuple(
            ProjectionEvidence(
                source_value=item.source_value,
                mapping_registry=registry,
                common_task_identity="dilemmadata.common.chord.quality",
                native_state="available",
                mapping_state=ProjectionMappingState.EXACT,
                projected_value=item.source_value.source_value,
                provenance=("existing-dilemmadata-common-registry",),
            )
            for item in support
        )
        projection_availability = (
            ProjectionAvailabilityCounts(
                corpus=corpus,
                source_task_id=task_data["source_task_id"],
                dialect=task_data["dialect"],
                mapping_registry=registry,
                common_task_identity="dilemmadata.common.chord.quality",
                observation_unit=ObservationUnit.TARGET_ROW,
                denominator=len(task_data["states"]),
                exact=available,
                coarsened=0,
                ambiguous=0,
                unsupported=state_counts["unsupported"],
                invalid=0,
                missing=state_counts["missing"],
                masked=state_counts["masked"],
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.FIXTURE,
                provenance=("existing-dilemmadata-common-registry",),
            ),
        )
    task = TaskFamilyEvidence(
        corpus=corpus,
        source_task_id=task_data["source_task_id"],
        dialect=task_data["dialect"],
        annotation_namespace=task_data["annotation_namespace"],
        vocabulary=identity(
            f"{task_data['source_task_id']}.vocabulary", task_data["vocabulary"]
        ),
        label_granularity="source_entry",
        label_value_type=LabelValueType.CATEGORICAL,
        observation_unit=ObservationUnit.TARGET_ROW,
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("synthetic-target-fixture",),
        status=ComputationStatus.OBSERVED,
        availability=AvailabilityCounts(
            observation_unit=ObservationUnit.TARGET_ROW,
            denominator=len(task_data["states"]),
            available=state_counts["available"],
            masked=state_counts["masked"],
            missing=state_counts["missing"],
            unsupported=state_counts["unsupported"],
            split_scope=SplitScope.TRAIN,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("synthetic-target-fixture",),
        ),
        work_identity=(
            identity(f"{corpus.value}.synthetic_work_identity", case["records"])
            if task_data["work_identity_proven"]
            else None
        ),
        class_support=tuple(support),
        projection_availability=projection_availability,
        projections=projections,
    )
    producer = identity(f"{corpus.value}.synthetic_eda_adapter", {"adapter": corpus.value})
    extension = SourceExtension(
        corpus=corpus,
        namespace=f"{corpus.value}.fixture_supervision",
        schema_name="SyntheticSupervisionExtension",
        schema_version="1.0.0",
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("synthetic-target-fixture",),
        rows=(
            ExtensionRow(
                row_id="dialect",
                payload={"dialect_detail": task_data["dialect"]},
                coverage=MetricCoverage(
                    observation_unit=ObservationUnit.TARGET_ROW,
                    denominator=len(task_data["states"]),
                    observed_count=len(task_data["states"]),
                    unknown_count=0,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.FIXTURE,
                    provenance=("synthetic-target-fixture",),
                ),
            ),
        ),
        target_free=False,
    )
    envelope = ReportEnvelope(
        schema_name=SUPERVISION_EDA_SCHEMA_NAME,
        schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.SUPERVISION,
        corpus=corpus,
        source_identity=identity(f"{corpus.value}.synthetic_release", case),
        producer_identity=producer,
        repository_commit="659bde251fee6c836564a45aea854b6abcac9fe0",
        evidence_scope=EvidenceScope.FIXTURE,
        execution_mode=ExecutionMode.SYNTHETIC_FIXTURE,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.TRAIN,
        observation_units=(
            ObservationUnit.SPLIT_ASSIGNMENT,
            ObservationUnit.TARGET_ACCESS_ATTEMPT,
            ObservationUnit.TARGET_ROW,
            ObservationUnit.LABEL_OCCURRENCE,
            ObservationUnit.RECORD,
            ObservationUnit.LOGICAL_WORK,
        ),
        input_manifests=(
            InputManifestRef(
                role="split_assignment",
                identity=VersionedIdentity(
                    identity=f"{corpus.value}.synthetic_split_assignment",
                    version="1.0.0",
                    fingerprint=assignment_fingerprint,
                ),
                target_free=True,
                repository_relative_path="tests/fixtures/multisource/eda_contract_cases.json",
            ),
            InputManifestRef(
                role="target_sidecar",
                identity=identity(
                    f"{corpus.value}.synthetic_target_manifest", task_data
                ),
                target_free=False,
                repository_relative_path="tests/fixtures/multisource/eda_contract_cases.json",
            ),
        ),
        operational_metadata=operational or {},
    )
    return SupervisionEDA(
        envelope=envelope,
        semantic_payload=SupervisionEDAPayload(
            tasks=(task,),
            test_lock=test_lock,
            extensions=(extension,),
        ),
    )


@pytest.fixture(scope="session")
def eda_cases() -> list[dict[str, Any]]:
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert value["schema"] == "MultiSourceEDASyntheticInput@1.0.0"
    return value["corpora"]


@pytest.fixture()
def raw_reports(eda_cases: list[dict[str, Any]]) -> dict[CorpusId, RawCorpusEDA]:
    return {CorpusId(case["corpus"]): raw_report(case) for case in eda_cases}


@pytest.fixture()
def supervision_reports(
    eda_cases: list[dict[str, Any]],
) -> dict[CorpusId, SupervisionEDA]:
    return {
        CorpusId(case["corpus"]): supervision_report(case)
        for case in eda_cases
        if case["task"] is not None
    }
