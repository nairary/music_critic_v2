from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from music_critic.eda import (
    CorpusId,
    EDAContractError,
    EvidenceScope,
    ExtensionRow,
    MetricCoverage,
    ObservationUnit,
    RawCorpusEDA,
    SourceExtension,
    StructuredWarning,
    SplitScope,
    SupervisionEDA,
    canonical_report_bytes,
    dumps_report,
    loads_report,
    report_dict,
    report_fingerprint,
)


def _extension_coverage() -> MetricCoverage:
    return MetricCoverage(
        observation_unit=ObservationUnit.RECORD,
        denominator=1,
        observed_count=1,
        unknown_count=0,
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
    )


def test_raw_and_supervision_reports_round_trip_strict_canonical_json(
    raw_reports, supervision_reports
) -> None:
    reports = [*raw_reports.values(), *supervision_reports.values()]
    for report in reports:
        payload = dumps_report(report)
        restored = loads_report(payload)
        assert type(restored) is type(report)
        assert report_dict(restored) == report_dict(report)
        assert dumps_report(restored) == payload
        assert report.semantic_fingerprint == report_fingerprint(report)
        assert canonical_report_bytes(report).endswith(b"\n")
        assert not canonical_report_bytes(report).endswith(b"\n\n")


def test_semantically_unordered_rows_do_not_change_fingerprint(raw_reports) -> None:
    source = raw_reports[CorpusId.DILEMMADATA]
    reordered = RawCorpusEDA(
        envelope=replace(
            source.envelope,
            input_manifests=tuple(reversed(source.envelope.input_manifests)),
            observation_units=tuple(reversed(source.envelope.observation_units)),
        ),
        semantic_payload=replace(
            source.semantic_payload,
            metrics=tuple(reversed(source.semantic_payload.metrics)),
        ),
    )
    assert reordered.semantic_fingerprint == source.semantic_fingerprint
    assert dumps_report(reordered) == dumps_report(source)


def test_warning_ties_are_order_independent(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    warnings = (
        StructuredWarning("same.code", "same message", ("source-a",)),
        StructuredWarning("same.code", "same message", ("source-b",)),
    )
    first = RawCorpusEDA(
        envelope=replace(source.envelope, warnings=warnings),
        semantic_payload=source.semantic_payload,
    )
    second = RawCorpusEDA(
        envelope=replace(source.envelope, warnings=tuple(reversed(warnings))),
        semantic_payload=source.semantic_payload,
    )
    assert dumps_report(first) == dumps_report(second)
    assert first.semantic_fingerprint == second.semantic_fingerprint


def test_mapping_insertion_order_and_unicode_are_canonical(raw_reports) -> None:
    source = raw_reports[CorpusId.POP909_CL]
    left = SourceExtension(
        corpus=CorpusId.POP909_CL,
        namespace="pop909_cl.synthetic_unicode",
        schema_name="SyntheticUnicode",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "row",
                {"β": 2, "á": 1},
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    right = SourceExtension(
        corpus=CorpusId.POP909_CL,
        namespace="pop909_cl.synthetic_unicode",
        schema_name="SyntheticUnicode",
        schema_version="1.0.0",
        split_scope=SplitScope.UNSPLIT,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        rows=(
            ExtensionRow(
                "row",
                {"á": 1, "β": 2},
                coverage=_extension_coverage(),
            ),
        ),
        target_free=True,
    )
    assert left.extension_fingerprint == right.extension_fingerprint
    first = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(left,)),
    )
    second = RawCorpusEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, extensions=(right,)),
    )
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert "β" in dumps_report(first)
    assert "\\u03b2" not in dumps_report(first)


def test_semantic_change_changes_fingerprint(supervision_reports) -> None:
    source = supervision_reports[CorpusId.HOOKTHEORY]
    task = source.semantic_payload.tasks[0]
    changed_task = replace(task, label_granularity="annotation_event")
    changed = SupervisionEDA(
        envelope=source.envelope,
        semantic_payload=replace(source.semantic_payload, tasks=(changed_task,)),
    )
    assert changed.semantic_fingerprint != source.semantic_fingerprint


def test_tampered_stored_fingerprint_is_rejected(raw_reports) -> None:
    value = report_dict(raw_reports[CorpusId.PDMX])
    value["semantic_fingerprint"] = "0" * 64
    with pytest.raises(EDAContractError, match="fingerprint.mismatch"):
        loads_report(json.dumps(value, ensure_ascii=False))


def test_nested_non_finite_values_are_rejected_before_fingerprinting(raw_reports) -> None:
    source = raw_reports[CorpusId.PDMX]
    with pytest.raises(EDAContractError, match="non_finite"):
        replace(
            source.envelope,
            operational_metadata={"nested": {"bad": math.nan}},
        )
    with pytest.raises(EDAContractError, match="non_finite"):
        SourceExtension(
            corpus=CorpusId.PDMX,
            namespace="pdmx.nonfinite",
            schema_name="NonFinite",
            schema_version="1.0.0",
            split_scope=SplitScope.UNSPLIT,
            evidence_scope=EvidenceScope.FIXTURE,
            provenance=("fixture",),
            rows=(
                ExtensionRow(
                    "bad",
                    {"value": math.inf},
                    coverage=_extension_coverage(),
                ),
            ),
            target_free=True,
        )


def test_non_standard_json_infinity_is_rejected(raw_reports) -> None:
    payload = dumps_report(raw_reports[CorpusId.PDMX])
    malformed = payload.replace('"semantic_fingerprint"', '"bad":NaN,"semantic_fingerprint"')
    with pytest.raises(EDAContractError, match="non_finite"):
        loads_report(malformed)


def test_duplicate_json_object_keys_are_rejected(raw_reports) -> None:
    payload = dumps_report(raw_reports[CorpusId.PDMX])
    marker = '"semantic_fingerprint":'
    assert payload.count(marker) == 1
    malformed = payload.replace(
        marker,
        f'{marker}"{"0" * 64}",{marker}',
        1,
    )
    with pytest.raises(EDAContractError, match="duplicate"):
        loads_report(malformed)


@pytest.mark.parametrize(
    ("location", "surrogate"),
    (
        ("operational_value", "\ud800"),
        ("semantic_value", "\ud801"),
        ("semantic_key", "\ud802"),
    ),
)
def test_report_loading_rejects_lone_unicode_surrogates(
    raw_reports,
    location,
    surrogate,
) -> None:
    payload = report_dict(raw_reports[CorpusId.PDMX])
    if location == "operational_value":
        payload["envelope"]["operational_metadata"] = {"hostname": surrogate}
    elif location == "semantic_value":
        payload["envelope"]["source_identity"]["identity"] = surrogate
    else:
        payload["semantic_payload"]["extensions"] = [
            {
                "corpus": "pdmx",
                "namespace": "pdmx.unicode_probe",
                "schema_name": "UnicodeProbe",
                "schema_version": "1.0.0",
                "split_scope": "unsplit",
                "evidence_scope": "fixture",
                "provenance": ["fixture"],
                "rows": [
                        {
                            "row_id": "probe",
                            "payload": {surrogate: "raw"},
                            "counts": [],
                            "coverage": {
                                "observation_unit": "record",
                                "denominator": 1,
                                "observed_count": 1,
                                "unknown_count": 0,
                                "split_scope": "unsplit",
                                "evidence_scope": "fixture",
                                "provenance": ["fixture"],
                                "status": "observed",
                                "reason_code": None,
                            },
                        }
                ],
                "target_free": True,
                "work_identity": None,
                "extension_contract_version": "1.0.0",
                "extension_fingerprint": "0" * 64,
            }
        ]
    encoded = json.dumps(payload, ensure_ascii=True)
    with pytest.raises(EDAContractError, match="utf8_invalid"):
        loads_report(encoded)


def test_report_loading_rejects_surrogate_in_extension_provenance(
    supervision_reports,
) -> None:
    payload = report_dict(supervision_reports[CorpusId.HOOKTHEORY])
    payload["semantic_payload"]["extensions"][0]["provenance"][0] = "\ud800"
    with pytest.raises(EDAContractError, match="utf8_invalid"):
        loads_report(json.dumps(payload, ensure_ascii=True))


@pytest.mark.parametrize("bad_metric_id", ({}, []))
def test_malformed_metric_id_raises_contract_error(raw_reports, bad_metric_id) -> None:
    payload = report_dict(raw_reports[CorpusId.PDMX])
    payload["semantic_payload"]["metrics"][0]["metric_id"] = bad_metric_id
    with pytest.raises(EDAContractError, match="identity.invalid"):
        loads_report(json.dumps(payload))
