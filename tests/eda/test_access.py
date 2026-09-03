from __future__ import annotations

from dataclasses import replace

import pytest

from music_critic.eda import (
    CorpusId,
    EDAContractError,
    EvidenceScope,
    ObservationUnit,
    SplitScope,
    SupervisionEDA,
    SupervisionTargetAccessGuard,
    TestTargetLockEvidence as TargetLockEvidence,
    UnitCount,
    load_supervision_train_validation_only,
)


def _assignment(record_id: str, split: str) -> dict[str, object]:
    return {
        "corpus": "hooktheory",
        "record_id": record_id,
        "split": split,
        "assignment_manifest_fingerprint": "a" * 64,
        "target_free": True,
    }


def test_test_target_loader_and_descriptor_resolver_are_never_invoked() -> None:
    descriptor_calls: list[tuple[str, SplitScope]] = []
    loader_calls: list[tuple[str, SplitScope]] = []

    def resolve(record_id: str, split: SplitScope) -> str:
        descriptor_calls.append((record_id, split))
        return f"descriptor:{record_id}"

    def load(descriptor: str, split: SplitScope) -> str:
        loader_calls.append((descriptor, split))
        if split == SplitScope.TEST:
            raise AssertionError("TEST target loader was called")
        return descriptor

    assignments = [
        _assignment("train-1", "train"),
        {
            "split": "test",
            # Deliberately no record ID: the TEST branch must stop at split.
            "target_path": "/must/not/be/resolved.json",
        },
        _assignment("validation-1", "validation"),
    ]
    loaded, evidence = load_supervision_train_validation_only(
        CorpusId.HOOKTHEORY,
        assignments,
        resolve_descriptor=resolve,
        load_target=load,
    )
    assert loaded == ("descriptor:train-1", "descriptor:validation-1")
    assert descriptor_calls == [
        ("train-1", SplitScope.TRAIN),
        ("validation-1", SplitScope.VALIDATION),
    ]
    assert loader_calls == [
        ("descriptor:train-1", SplitScope.TRAIN),
        ("descriptor:validation-1", SplitScope.VALIDATION),
    ]
    assert evidence.test_assignment_count.value == 1
    assert evidence.test_descriptor_resolution_count.value == 0
    assert evidence.test_target_loader_call_count.value == 0
    assert evidence.test_targets_read is False
    assert evidence.test_targets_used_for_eda is False
    assert evidence.test_targets_used_for_model_evaluation is False

    typed_counts = (
        evidence.test_assignment_count,
        evidence.test_descriptor_resolution_count,
        evidence.test_target_loader_call_count,
        evidence.test_target_records_opened,
        evidence.test_target_rows_loaded,
    )
    assert all(count.denominator == 1 for count in typed_counts)
    assert all(
        count.denominator_unit == ObservationUnit.SPLIT_ASSIGNMENT
        for count in typed_counts
    )
    assert all(count.split_scope == SplitScope.TEST for count in typed_counts)
    assert all(count.evidence_scope == EvidenceScope.FIXTURE for count in typed_counts)
    assert [count.observation_unit.value for count in typed_counts] == [
        "split_assignment",
        "target_access_attempt",
        "target_access_attempt",
        "record",
        "target_row",
    ]


def test_complete_assignment_preflight_happens_before_first_callback() -> None:
    calls: list[str] = []
    assignments = [
        _assignment("train-1", "train"),
        {"split": "mystery"},
    ]
    with pytest.raises(EDAContractError, match="assignment_split_invalid"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            assignments,
            resolve_descriptor=lambda record_id, split: calls.append(record_id),
            load_target=lambda descriptor, split: descriptor,
        )
    assert calls == []


def test_invalid_utf8_record_id_fails_before_callbacks() -> None:
    descriptor_calls: list[str] = []
    loader_calls: list[str] = []
    row = _assignment("\ud800", "train")
    with pytest.raises(EDAContractError, match="utf8_invalid"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (row,),
            resolve_descriptor=lambda record_id, split: (
                descriptor_calls.append(record_id) or record_id
            ),
            load_target=lambda descriptor, split: (
                loader_calls.append(descriptor) or descriptor
            ),
        )
    assert descriptor_calls == []
    assert loader_calls == []


def test_invalid_utf8_guard_provenance_fails_at_construction() -> None:
    with pytest.raises(EDAContractError, match="utf8_invalid"):
        SupervisionTargetAccessGuard(
            corpus=CorpusId.HOOKTHEORY,
            provenance=("\ud800",),
        )


def test_duplicate_allowed_assignments_fail_before_callbacks() -> None:
    calls: list[str] = []
    assignment = _assignment("train-1", "train")
    with pytest.raises(EDAContractError, match="assignment_duplicate"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (assignment, dict(assignment)),
            resolve_descriptor=lambda record_id, split: calls.append(record_id),
            load_target=lambda descriptor, split: descriptor,
        )
    assert calls == []


def test_record_assigned_to_train_and_validation_fails_before_callbacks() -> None:
    descriptor_calls: list[str] = []
    loader_calls: list[str] = []
    with pytest.raises(EDAContractError, match="assignment_duplicate"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (
                _assignment("cross-split-record", "train"),
                _assignment("cross-split-record", "validation"),
            ),
            resolve_descriptor=lambda record_id, split: (
                descriptor_calls.append(record_id) or record_id
            ),
            load_target=lambda descriptor, split: (
                loader_calls.append(descriptor) or descriptor
            ),
        )
    assert descriptor_calls == []
    assert loader_calls == []


def test_pdmx_has_no_target_guard_or_loader_path() -> None:
    with pytest.raises(EDAContractError, match="supervision_forbidden"):
        SupervisionTargetAccessGuard(corpus=CorpusId.PDMX)


def test_test_lock_evidence_rejects_any_access_or_use_flag() -> None:
    clean = TargetLockEvidence.from_guard(
        test_assignment_count=3,
        assignment_manifest_fingerprint="a" * 64,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
    )
    for field in (
        "test_target_loader_call_count",
        "test_target_rows_loaded",
    ):
        count = getattr(clean, field)
        with pytest.raises(EDAContractError, match="count_binding_invalid"):
            replace(clean, **{field: replace(count, value=1)})
    for field, value in (
        ("test_targets_read", True),
        ("test_targets_used_for_eda", True),
        ("test_targets_used_for_model_evaluation", True),
        ("test_class_distributions_emitted", True),
        ("test_coverage_emitted", True),
        ("test_cooccurrence_emitted", True),
    ):
        with pytest.raises(EDAContractError, match="test_lock.violation"):
            replace(clean, **{field: value})
    with pytest.raises(EDAContractError, match="gate_missing"):
        replace(clean, assignment_gate_before_target_open=1)

    unavailable = UnitCount(
        name="test_assignment_count",
        observation_unit=ObservationUnit.SPLIT_ASSIGNMENT,
        value=None,
        denominator=3,
        denominator_unit=ObservationUnit.SPLIT_ASSIGNMENT,
        split_scope=SplitScope.TEST,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=("fixture",),
        status="not_computed",
        reason_code="fixture.not_computed",
    )
    with pytest.raises(EDAContractError, match="count_binding_invalid"):
        replace(clean, test_assignment_count=unavailable)


def test_supervision_validator_rejects_test_task_rows(supervision_reports) -> None:
    report = supervision_reports[CorpusId.POP909_CL]
    task = report.semantic_payload.tasks[0]
    with pytest.raises(EDAContractError, match="supervision_split_forbidden"):
        replace(task, split_scope=SplitScope.TEST)


def test_test_lock_count_scope_must_match_supervision_report(
    supervision_reports,
) -> None:
    report = supervision_reports[CorpusId.HOOKTHEORY]
    lock = TargetLockEvidence.from_guard(
        test_assignment_count=1,
        assignment_manifest_fingerprint=(
            report.semantic_payload.test_lock.assignment_manifest_fingerprint
        ),
        evidence_scope=EvidenceScope.BOUNDED,
        provenance=("bounded-spy",),
    )
    with pytest.raises(EDAContractError, match="evidence_scope_mismatch"):
        SupervisionEDA(
            envelope=report.envelope,
            semantic_payload=replace(report.semantic_payload, test_lock=lock),
        )


def test_target_fields_are_rejected_before_train_loader_invocation() -> None:
    calls: list[str] = []
    row = _assignment("train-1", "train")
    row["sidecar_path"] = "targets/train-1.json"
    with pytest.raises(EDAContractError, match="target_field_forbidden"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (row,),
            resolve_descriptor=lambda record_id, split: calls.append(record_id),
            load_target=lambda descriptor, split: descriptor,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_descriptor", "targets/train-1.json"),
        ("class_values", ["secret-class"]),
        ("unexpected_metadata", "not-part-of-the-target-free-contract"),
    ),
)
def test_train_validation_assignment_rejects_every_undeclared_field_before_callbacks(
    field: str,
    value: object,
) -> None:
    descriptor_calls: list[str] = []
    loader_calls: list[str] = []
    row = _assignment("train-1", "train")
    row[field] = value

    with pytest.raises(EDAContractError):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (row,),
            resolve_descriptor=lambda record_id, split: (
                descriptor_calls.append(record_id) or record_id
            ),
            load_target=lambda descriptor, split: (
                loader_calls.append(descriptor) or descriptor
            ),
        )

    assert descriptor_calls == []
    assert loader_calls == []


def test_assignment_manifest_identity_is_consistent_before_callbacks() -> None:
    calls: list[str] = []
    first = _assignment("train-1", "train")
    second = _assignment("validation-1", "validation")
    second["assignment_manifest_fingerprint"] = "b" * 64
    with pytest.raises(EDAContractError, match="assignment_manifest_mismatch"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (first, second),
            resolve_descriptor=lambda record_id, split: calls.append(record_id),
            load_target=lambda descriptor, split: descriptor,
        )
    assert calls == []


@pytest.mark.parametrize("record_id", ("te\x00st-target", "te\u200bst-target"))
def test_structural_record_id_controls_fail_before_callbacks(record_id: str) -> None:
    descriptor_calls: list[str] = []
    loader_calls: list[str] = []

    with pytest.raises(EDAContractError, match="control_character"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (_assignment(record_id, "train"),),
            resolve_descriptor=lambda value, split: (
                descriptor_calls.append(value) or value
            ),
            load_target=lambda descriptor, split: (
                loader_calls.append(descriptor) or descriptor
            ),
        )

    assert descriptor_calls == []
    assert loader_calls == []


def test_structural_guard_provenance_controls_fail_at_construction() -> None:
    with pytest.raises(EDAContractError, match="control_character"):
        SupervisionTargetAccessGuard(
            corpus=CorpusId.HOOKTHEORY,
            provenance=("te\u200bst_targets_loaded",),
        )


def test_non_string_assignment_keys_fail_cleanly_before_callbacks() -> None:
    descriptor_calls: list[str] = []
    loader_calls: list[str] = []
    row = _assignment("train-1", "train")
    row[1] = "bad-key"  # type: ignore[index]

    with pytest.raises(EDAContractError, match="assignment_field_invalid"):
        load_supervision_train_validation_only(
            CorpusId.HOOKTHEORY,
            (row,),
            resolve_descriptor=lambda value, split: (
                descriptor_calls.append(value) or value
            ),
            load_target=lambda descriptor, split: (
                loader_calls.append(descriptor) or descriptor
            ),
        )

    assert descriptor_calls == []
    assert loader_calls == []
