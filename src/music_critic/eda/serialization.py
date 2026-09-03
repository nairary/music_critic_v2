"""Strict deterministic serialization for multi-source EDA reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, TypeVar

from music_critic.eda.contracts import (
    AvailabilityCounts,
    CategoryCount,
    ClassSupport,
    EDAContractError,
    ExtensionRow,
    GraphEvidence,
    InputManifestRef,
    InvariantEvidence,
    NumericDistribution,
    ProjectionAvailabilityCounts,
    ProjectionEvidence,
    QuantilePoint,
    RAW_CORPUS_EDA_SCHEMA_NAME,
    RawCorpusEDA,
    RawCorpusEDAPayload,
    RawMetricEvidence,
    ReportEnvelope,
    SourceExtension,
    SourceValueIdentity,
    StructuredWarning,
    SUPERVISION_EDA_SCHEMA_NAME,
    SupervisionEDA,
    SupervisionEDAPayload,
    TaskFamilyEvidence,
    TestTargetLockEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    dumps_report,
    report_dict,
)


Report = RawCorpusEDA | SupervisionEDA
T = TypeVar("T")


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EDAContractError(
            "eda.serialization.object_invalid", "expected a JSON object", path=path
        )
    return value


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise EDAContractError(
            "eda.serialization.array_invalid", "expected a JSON array", path=path
        )
    return value


def _exact(
    value: object, fields: tuple[str, ...], path: str
) -> Mapping[str, Any]:
    mapping = _object(value, path)
    expected = set(fields)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise EDAContractError(
            "eda.serialization.fields_invalid",
            f"missing={missing!r}, unknown={unknown!r}",
            path=path,
        )
    return mapping


def _rows(value: object, decoder: Any, path: str) -> tuple[Any, ...]:
    return tuple(
        decoder(item, f"{path}[{index}]")
        for index, item in enumerate(_array(value, path))
    )


def _identity(value: object, path: str) -> VersionedIdentity:
    row = _exact(value, ("identity", "version", "fingerprint"), path)
    return VersionedIdentity(**row)


def _manifest(value: object, path: str) -> InputManifestRef:
    row = _exact(
        value,
        ("role", "identity", "target_free", "repository_relative_path"),
        path,
    )
    return InputManifestRef(
        role=row["role"],
        identity=_identity(row["identity"], f"{path}.identity"),
        target_free=row["target_free"],
        repository_relative_path=row["repository_relative_path"],
    )


def _unit_count(value: object, path: str) -> UnitCount:
    row = _exact(
        value,
        (
            "name",
            "observation_unit",
            "value",
            "denominator",
            "denominator_unit",
            "split_scope",
            "evidence_scope",
            "provenance",
            "status",
            "reason_code",
        ),
        path,
    )
    return UnitCount(
        **{key: item for key, item in row.items() if key != "provenance"},
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _coverage(value: object, path: str):
    from music_critic.eda.contracts import MetricCoverage

    row = _exact(
        value,
        (
            "observation_unit",
            "denominator",
            "observed_count",
            "unknown_count",
            "split_scope",
            "evidence_scope",
            "provenance",
            "status",
            "reason_code",
        ),
        path,
    )
    return MetricCoverage(
        **{key: item for key, item in row.items() if key != "provenance"},
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _quantile(value: object, path: str) -> QuantilePoint:
    return QuantilePoint(**_exact(value, ("numerator", "denominator", "value"), path))


def _numeric(value: object, path: str) -> NumericDistribution:
    row = _exact(
        value,
        (
            "measurement_unit",
            "minimum",
            "maximum",
            "mean",
            "quantiles",
            "float_policy",
            "quantile_policy",
        ),
        path,
    )
    return NumericDistribution(
        **{key: item for key, item in row.items() if key != "quantiles"},
        quantiles=_rows(row["quantiles"], _quantile, f"{path}.quantiles"),
    )


def _category(value: object, path: str) -> CategoryCount:
    row = _exact(value, ("category", "count"), path)
    return CategoryCount(
        category=row["category"], count=_unit_count(row["count"], f"{path}.count")
    )


def _raw_metric(value: object, path: str) -> RawMetricEvidence:
    row = _exact(
        value, ("metric_id", "coverage", "count", "numeric", "categories"), path
    )
    return RawMetricEvidence(
        metric_id=row["metric_id"],
        coverage=_coverage(row["coverage"], f"{path}.coverage"),
        count=None if row["count"] is None else _unit_count(row["count"], f"{path}.count"),
        numeric=None if row["numeric"] is None else _numeric(row["numeric"], f"{path}.numeric"),
        categories=_rows(row["categories"], _category, f"{path}.categories"),
    )


def _graph(value: object, path: str) -> GraphEvidence:
    row = _exact(
        value,
        (
            "status",
            "target_free",
            "graph_schema",
            "graph_builder",
            "feature_registry",
            "validator",
            "reason_code",
        ),
        path,
    )
    return GraphEvidence(
        status=row["status"],
        target_free=row["target_free"],
        graph_schema=None if row["graph_schema"] is None else _identity(row["graph_schema"], f"{path}.graph_schema"),
        graph_builder=None if row["graph_builder"] is None else _identity(row["graph_builder"], f"{path}.graph_builder"),
        feature_registry=None if row["feature_registry"] is None else _identity(row["feature_registry"], f"{path}.feature_registry"),
        validator=None if row["validator"] is None else _identity(row["validator"], f"{path}.validator"),
        reason_code=row["reason_code"],
    )


def _invariant(value: object, path: str) -> InvariantEvidence:
    row = _exact(value, ("code", "status", "provenance", "reason_code"), path)
    return InvariantEvidence(
        code=row["code"],
        status=row["status"],
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
        reason_code=row["reason_code"],
    )


def _warning(value: object, path: str) -> StructuredWarning:
    row = _exact(value, ("code", "message", "provenance"), path)
    return StructuredWarning(
        code=row["code"],
        message=row["message"],
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _reason(value: object, path: str) -> UnavailableReason:
    row = _exact(value, ("code", "status", "provenance", "detail"), path)
    return UnavailableReason(
        code=row["code"],
        status=row["status"],
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
        detail=row["detail"],
    )


def _envelope(value: object, path: str = "$.envelope") -> ReportEnvelope:
    row = _exact(
        value,
        (
            "schema_name",
            "schema_version",
            "report_kind",
            "corpus",
            "source_identity",
            "producer_identity",
            "repository_commit",
            "evidence_scope",
            "execution_mode",
            "completeness_status",
            "split_scope",
            "observation_units",
            "input_manifests",
            "invariants",
            "warnings",
            "unavailable_reasons",
            "operational_metadata",
            "envelope_schema_name",
            "envelope_schema_version",
            "version_policy",
        ),
        path,
    )
    return ReportEnvelope(
        **{
            key: item
            for key, item in row.items()
            if key
            not in {
                "source_identity",
                "producer_identity",
                "observation_units",
                "input_manifests",
                "invariants",
                "warnings",
                "unavailable_reasons",
                "operational_metadata",
            }
        },
        source_identity=_identity(row["source_identity"], f"{path}.source_identity"),
        producer_identity=_identity(row["producer_identity"], f"{path}.producer_identity"),
        observation_units=tuple(_array(row["observation_units"], f"{path}.observation_units")),
        input_manifests=_rows(row["input_manifests"], _manifest, f"{path}.input_manifests"),
        invariants=_rows(row["invariants"], _invariant, f"{path}.invariants"),
        warnings=_rows(row["warnings"], _warning, f"{path}.warnings"),
        unavailable_reasons=_rows(row["unavailable_reasons"], _reason, f"{path}.unavailable_reasons"),
        operational_metadata=_object(row["operational_metadata"], f"{path}.operational_metadata"),
    )


def _extension_row(value: object, path: str) -> ExtensionRow:
    row = _exact(value, ("row_id", "payload", "counts", "coverage"), path)
    return ExtensionRow(
        row_id=row["row_id"],
        payload=_object(row["payload"], f"{path}.payload"),
        counts=_rows(row["counts"], _unit_count, f"{path}.counts"),
        coverage=_coverage(row["coverage"], f"{path}.coverage"),
    )


def _extension(value: object, path: str) -> SourceExtension:
    row = _exact(
        value,
        (
            "corpus",
            "namespace",
            "schema_name",
            "schema_version",
            "split_scope",
            "evidence_scope",
            "provenance",
            "rows",
            "target_free",
            "work_identity",
            "extension_contract_version",
            "extension_fingerprint",
        ),
        path,
    )
    extension = SourceExtension(
        corpus=row["corpus"],
        namespace=row["namespace"],
        schema_name=row["schema_name"],
        schema_version=row["schema_version"],
        split_scope=row["split_scope"],
        evidence_scope=row["evidence_scope"],
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
        rows=_rows(row["rows"], _extension_row, f"{path}.rows"),
        target_free=row["target_free"],
        work_identity=(
            None
            if row["work_identity"] is None
            else _identity(row["work_identity"], f"{path}.work_identity")
        ),
        extension_contract_version=row["extension_contract_version"],
    )
    if row["extension_fingerprint"] != extension.extension_fingerprint:
        raise EDAContractError(
            "eda.extension.fingerprint_mismatch",
            "serialized extension fingerprint differs from semantic rows",
            path=f"{path}.extension_fingerprint",
        )
    return extension


def _availability(value: object, path: str) -> AvailabilityCounts:
    row = _exact(
        value,
        (
            "observation_unit",
            "denominator",
            "available",
            "masked",
            "missing",
            "unsupported",
            "split_scope",
            "evidence_scope",
            "provenance",
        ),
        path,
    )
    return AvailabilityCounts(
        **{key: item for key, item in row.items() if key != "provenance"},
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _source_value(value: object, path: str) -> SourceValueIdentity:
    row = _exact(
        value,
        (
            "corpus",
            "source_task_id",
            "dialect",
            "source_value",
            "value_kind",
            "identity_version",
            "identity",
        ),
        path,
    )
    identity = SourceValueIdentity(
        **{key: item for key, item in row.items() if key != "identity"}
    )
    if row["identity"] != identity.identity:
        raise EDAContractError(
            "eda.source_value.identity_mismatch",
            "serialized source-value identity differs from its composite fields",
            path=f"{path}.identity",
        )
    return identity


def _class_support(value: object, path: str) -> ClassSupport:
    row = _exact(
        value,
        (
            "source_value",
            "occurrence_count",
            "unique_record_count",
            "unique_work_count",
            "available_only",
        ),
        path,
    )
    return ClassSupport(
        source_value=_source_value(row["source_value"], f"{path}.source_value"),
        occurrence_count=_unit_count(row["occurrence_count"], f"{path}.occurrence_count"),
        unique_record_count=_unit_count(row["unique_record_count"], f"{path}.unique_record_count"),
        unique_work_count=_unit_count(row["unique_work_count"], f"{path}.unique_work_count"),
        available_only=row["available_only"],
    )


def _projection(value: object, path: str) -> ProjectionEvidence:
    row = _exact(
        value,
        (
            "source_value",
            "mapping_registry",
            "common_task_identity",
            "native_state",
            "mapping_state",
            "projected_value",
            "provenance",
        ),
        path,
    )
    return ProjectionEvidence(
        source_value=_source_value(row["source_value"], f"{path}.source_value"),
        mapping_registry=_identity(row["mapping_registry"], f"{path}.mapping_registry"),
        common_task_identity=row["common_task_identity"],
        native_state=row["native_state"],
        mapping_state=row["mapping_state"],
        projected_value=row["projected_value"],
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _projection_availability(
    value: object, path: str
) -> ProjectionAvailabilityCounts:
    row = _exact(
        value,
        (
            "corpus",
            "source_task_id",
            "dialect",
            "mapping_registry",
            "common_task_identity",
            "observation_unit",
            "denominator",
            "exact",
            "coarsened",
            "ambiguous",
            "unsupported",
            "invalid",
            "missing",
            "masked",
            "split_scope",
            "evidence_scope",
            "provenance",
        ),
        path,
    )
    return ProjectionAvailabilityCounts(
        **{
            key: item
            for key, item in row.items()
            if key not in {"mapping_registry", "provenance"}
        },
        mapping_registry=_identity(
            row["mapping_registry"], f"{path}.mapping_registry"
        ),
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
    )


def _task(value: object, path: str) -> TaskFamilyEvidence:
    row = _exact(
        value,
        (
            "corpus",
            "source_task_id",
            "dialect",
            "annotation_namespace",
            "vocabulary",
            "label_granularity",
            "label_value_type",
            "observation_unit",
            "split_scope",
            "evidence_scope",
            "provenance",
            "status",
            "availability",
            "work_identity",
            "class_support",
            "empty_multilabel_available_count",
            "projection_availability",
            "projections",
            "reason_code",
        ),
        path,
    )
    return TaskFamilyEvidence(
        **{
            key: item
            for key, item in row.items()
            if key
            not in {
                "vocabulary",
                "provenance",
                "availability",
                "work_identity",
                "class_support",
                "empty_multilabel_available_count",
                "projection_availability",
                "projections",
            }
        },
        vocabulary=_identity(row["vocabulary"], f"{path}.vocabulary"),
        provenance=tuple(_array(row["provenance"], f"{path}.provenance")),
        availability=None if row["availability"] is None else _availability(row["availability"], f"{path}.availability"),
        work_identity=None if row["work_identity"] is None else _identity(row["work_identity"], f"{path}.work_identity"),
        class_support=_rows(row["class_support"], _class_support, f"{path}.class_support"),
        empty_multilabel_available_count=(
            None
            if row["empty_multilabel_available_count"] is None
            else _unit_count(
                row["empty_multilabel_available_count"],
                f"{path}.empty_multilabel_available_count",
            )
        ),
        projection_availability=_rows(
            row["projection_availability"],
            _projection_availability,
            f"{path}.projection_availability",
        ),
        projections=_rows(row["projections"], _projection, f"{path}.projections"),
    )


def _test_lock(value: object, path: str) -> TestTargetLockEvidence:
    row = _exact(
        value,
        (
            "test_assignment_count",
            "assignment_manifest_fingerprint",
            "test_descriptor_resolution_count",
            "test_target_loader_call_count",
            "test_target_records_opened",
            "test_target_rows_loaded",
            "assignment_gate_before_descriptor_resolution",
            "assignment_gate_before_target_open",
            "test_targets_read",
            "test_targets_used_for_eda",
            "test_targets_used_for_model_evaluation",
            "test_class_distributions_emitted",
            "test_coverage_emitted",
            "test_cooccurrence_emitted",
            "contract_version",
            "gate_order",
        ),
        path,
    )
    count_fields = {
        "test_assignment_count",
        "test_descriptor_resolution_count",
        "test_target_loader_call_count",
        "test_target_records_opened",
        "test_target_rows_loaded",
    }
    return TestTargetLockEvidence(
        **{key: item for key, item in row.items() if key not in count_fields},
        **{
            key: _unit_count(row[key], f"{path}.{key}")
            for key in count_fields
        },
    )


def report_from_dict(value: object) -> Report:
    """Decode, cross-validate, and fingerprint-check one exact report object."""

    row = _exact(value, ("envelope", "semantic_payload", "semantic_fingerprint"), "$")
    envelope = _envelope(row["envelope"])
    payload = _object(row["semantic_payload"], "$.semantic_payload")
    if envelope.schema_name == RAW_CORPUS_EDA_SCHEMA_NAME:
        raw = _exact(payload, ("metrics", "graph_evidence", "extensions"), "$.semantic_payload")
        report: Report = RawCorpusEDA(
            envelope=envelope,
            semantic_payload=RawCorpusEDAPayload(
                metrics=_rows(raw["metrics"], _raw_metric, "$.semantic_payload.metrics"),
                graph_evidence=_graph(raw["graph_evidence"], "$.semantic_payload.graph_evidence"),
                extensions=_rows(raw["extensions"], _extension, "$.semantic_payload.extensions"),
            ),
        )
    elif envelope.schema_name == SUPERVISION_EDA_SCHEMA_NAME:
        supervision = _exact(payload, ("tasks", "test_lock", "extensions"), "$.semantic_payload")
        report = SupervisionEDA(
            envelope=envelope,
            semantic_payload=SupervisionEDAPayload(
                tasks=_rows(supervision["tasks"], _task, "$.semantic_payload.tasks"),
                test_lock=_test_lock(supervision["test_lock"], "$.semantic_payload.test_lock"),
                extensions=_rows(supervision["extensions"], _extension, "$.semantic_payload.extensions"),
            ),
        )
    else:
        raise EDAContractError(
            "eda.serialization.schema_unknown",
            f"unsupported report schema {envelope.schema_name!r}",
            path="$.envelope.schema_name",
        )
    if row["semantic_fingerprint"] != report.semantic_fingerprint:
        raise EDAContractError(
            "eda.fingerprint.mismatch",
            "serialized semantic fingerprint differs from canonical report semantics",
            path="$.semantic_fingerprint",
        )
    return report


def loads_report(payload: str | bytes | bytearray) -> Report:
    """Decode strict UTF-8 JSON; reject non-standard non-finite constants."""

    try:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
    except UnicodeDecodeError as exc:
        raise EDAContractError(
            "eda.serialization.utf8_invalid", "report bytes must be valid UTF-8"
        ) from exc
    if not isinstance(text, str):
        raise EDAContractError(
            "eda.serialization.payload_invalid", "report payload must be text or bytes"
        )

    def reject_constant(value: str) -> None:
        raise EDAContractError(
            "eda.serialization.non_finite", f"non-standard JSON constant {value!r}"
        )

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise EDAContractError(
                    "eda.serialization.duplicate_key",
                    f"duplicate JSON object key {key!r}",
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise EDAContractError(
            "eda.serialization.json_invalid", str(exc)
        ) from exc
    return report_from_dict(value)


def canonical_report_bytes(report: Report, *, indent: int | None = 2) -> bytes:
    """Encode one report with exactly one terminal newline."""

    return (dumps_report(report, indent=indent) + "\n").encode("utf-8")


def dump_report(report: Report, path: str | Path, *, indent: int | None = 2) -> None:
    """Failure-atomically write one deterministic UTF-8 report."""

    encoded = canonical_report_bytes(report, indent=indent)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_report(path: str | Path) -> Report:
    return loads_report(Path(path).read_bytes())


__all__ = [
    "Report",
    "canonical_report_bytes",
    "dump_report",
    "dumps_report",
    "load_report",
    "loads_report",
    "report_dict",
    "report_from_dict",
]
