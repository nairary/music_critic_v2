from __future__ import annotations

from dataclasses import replace

import pytest

from music_critic.eda import (
    CorpusId,
    EDAContractError,
    ObservationUnit,
    SplitScope,
    SupervisionEDA,
)


def test_task_family_keeps_one_work_unit_across_splits(
    supervision_reports,
) -> None:
    report = supervision_reports[CorpusId.POP909_CL]
    train = report.semantic_payload.tasks[0]
    assert train.class_support
    validation_support = tuple(
        replace(
            support,
            occurrence_count=replace(
                support.occurrence_count,
                split_scope=SplitScope.VALIDATION,
            ),
            unique_record_count=replace(
                support.unique_record_count,
                split_scope=SplitScope.VALIDATION,
            ),
            unique_work_count=replace(
                support.unique_work_count,
                observation_unit=ObservationUnit.CANONICAL_WORK,
                denominator_unit=ObservationUnit.CANONICAL_WORK,
                split_scope=SplitScope.VALIDATION,
            ),
        )
        for support in train.class_support
    )
    validation = replace(
        train,
        split_scope=SplitScope.VALIDATION,
        availability=replace(
            train.availability,
            split_scope=SplitScope.VALIDATION,
        ),
        class_support=validation_support,
    )

    with pytest.raises(EDAContractError, match="task_schema_mismatch"):
        replace(report.semantic_payload, tasks=(train, validation))


@pytest.mark.parametrize("label_granularity", ("test_target_row", "heldout_labels"))
def test_task_label_granularity_cannot_smuggle_test_scope(
    supervision_reports,
    label_granularity,
) -> None:
    report = supervision_reports[CorpusId.HOOKTHEORY]
    task = replace(
        report.semantic_payload.tasks[0],
        label_granularity=label_granularity,
    )
    with pytest.raises(EDAContractError, match="task_test_field_forbidden"):
        SupervisionEDA(
            envelope=report.envelope,
            semantic_payload=replace(report.semantic_payload, tasks=(task,)),
        )
