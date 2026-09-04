from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from music_critic.eda import (
    EDAContractError,
    EvidenceScope,
    ObservationUnit,
    RawCorpusEDA,
    SplitScope,
    UnitCount,
    VersionedIdentity,
    canonical_report_bytes,
    dump_report,
    dumps_report,
    report_dict,
    report_fingerprint,
    sum_unit_counts,
)


@pytest.mark.parametrize(
    "operation",
    (
        report_dict,
        report_fingerprint,
        dumps_report,
        canonical_report_bytes,
    ),
)
def test_public_report_serializers_reject_unvalidated_top_level_objects(
    operation,
) -> None:
    with pytest.raises(EDAContractError, match="eda.report.type_invalid"):
        operation({})


def test_file_serializer_rejects_unvalidated_report_without_leaving_output(
    tmp_path,
) -> None:
    destination = tmp_path / "not-a-report.json"
    with pytest.raises(EDAContractError, match="eda.report.type_invalid"):
        dump_report({}, destination)
    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_root_report_subclass_cannot_add_unfingerprinted_wire_fields(
    raw_reports,
) -> None:
    @dataclass(frozen=True, slots=True)
    class ExtendedRawCorpusEDA(RawCorpusEDA):
        extra: str = "not-in-the-base-schema"

    source = raw_reports[next(iter(raw_reports))]
    with pytest.raises(EDAContractError, match="eda.raw.type_invalid"):
        ExtendedRawCorpusEDA(
            envelope=source.envelope,
            semantic_payload=source.semantic_payload,
        )


def test_nested_schema_subclass_is_rejected_before_report_fingerprinting(
    raw_reports,
) -> None:
    @dataclass(frozen=True, slots=True)
    class ExtendedVersionedIdentity(VersionedIdentity):
        extra: str = "not-in-the-base-schema"

    source = raw_reports[next(iter(raw_reports))]
    identity = source.envelope.source_identity
    extended_identity = ExtendedVersionedIdentity(
        identity=identity.identity,
        version=identity.version,
        fingerprint=identity.fingerprint,
    )
    with pytest.raises(EDAContractError, match="schema_type_invalid"):
        RawCorpusEDA(
            envelope=replace(source.envelope, source_identity=extended_identity),
            semantic_payload=source.semantic_payload,
        )


def _count(*, name: str, value: int, denominator: int) -> UnitCount:
    return UnitCount(
        name=name,
        observation_unit=ObservationUnit.RECORD,
        value=value,
        denominator=denominator,
        denominator_unit=ObservationUnit.RECORD,
        split_scope=SplitScope.TRAIN,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("sum-fixture",),
    )


def test_unit_count_sum_preserves_the_shared_population_denominator() -> None:
    total = sum_unit_counts(
        "partition_total",
        (
            _count(name="accepted", value=3, denominator=10),
            _count(name="quarantined", value=4, denominator=10),
        ),
    )
    assert total.value == 7
    assert total.denominator == 10


def test_unit_count_sum_rejects_distinct_denominators() -> None:
    with pytest.raises(EDAContractError, match="unit_or_scope_mismatch"):
        sum_unit_counts(
            "invalid_total",
            (
                _count(name="accepted", value=3, denominator=10),
                _count(name="quarantined", value=4, denominator=11),
            ),
        )
