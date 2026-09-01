from __future__ import annotations

import copy

import pytest

from music_critic.experiments.analysisgnn.class_balance import (
    CLASS_BALANCE_SCHEMA,
    AuditTaskSpec,
    ClassBalanceAccumulator,
    EntityTargetObservation,
    JointTupleAccumulator,
    JointTupleObservation,
    RecordTargetObservations,
    candidate_class_weights,
    class_balance_contract,
    joint_observations_from_sidecar,
    load_train_validation_only,
    observations_from_sidecar,
    project_quality_record,
    recommend_head_trainability,
    semantic_fingerprint,
    train_support_tier,
    validation_support_tier,
)
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    COMPATIBILITY_QUALITY_VOCABULARY_ID,
    CORRECTED_QUALITY_VOCABULARY_ID,
    get_vocabulary,
)


TASK = AuditTaskSpec("task", "fixture-v1", "note", ("a", "b", "zero"))


def _target(
    entity: str,
    value: str | None,
    row: str | None,
    *,
    available: bool = True,
    masked: bool = False,
) -> EntityTargetObservation:
    return EntityTargetObservation("task", entity, row, value, available, masked)


def _record(
    record_id: str,
    component: str,
    dialect: str,
    split: str,
    *targets: EntityTargetObservation,
) -> RecordTargetObservations:
    return RecordTargetObservations(
        record_id,
        component,
        dialect,  # type: ignore[arg-type]
        split,  # type: ignore[arg-type]
        tuple(targets),
    )


def _rows(accumulator: ClassBalanceAccumulator):
    return {
        (row["split"], row["class_value"]): row
        for row in accumulator.class_rows()
    }


def test_masked_and_missing_never_become_classes_and_zero_class_is_emitted() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    audit.add_record(
        _record(
            "r1",
            "c1",
            "an_joint",
            "train",
            _target("e1", None, None, available=False, masked=True),
            _target("e2", None, None, available=False, masked=False),
        )
    )
    rows = _rows(audit)
    assert rows[("train", "zero")]["entity_count"] == 0
    assert rows[("train", "zero")]["support_tier"] == "absent"
    summary = audit.head_summaries(audit.class_rows())[0]
    assert summary["state_counts"]["train"] == {
        "available_count": 0,
        "masked_count": 1,
        "absent_count": 1,
    }


def test_entity_and_canonical_row_support_diverge_under_broadcast() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    audit.add_record(
        _record(
            "r1",
            "c1",
            "dlc",
            "train",
            _target("e1", "a", "source:1"),
            _target("e2", "a", "source:1"),
            _target("e3", "a", "source:2"),
        )
    )
    row = _rows(audit)[("train", "a")]
    assert row["entity_count"] == 3
    assert row["canonical_target_row_count"] == 2
    assert row["broadcast_factor"] == 1.5


def test_record_component_and_dialect_support_are_independent_and_deterministic() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    for row in (
        _record("r1", "c1", "an_joint", "train", _target("e1", "a", "s1")),
        _record("r2", "c1", "dlc", "train", _target("e2", "a", "s2")),
        _record("r3", "c2", "dlc", "train", _target("e3", "a", "s3")),
    ):
        audit.add_record(row)
    value = _rows(audit)[("train", "a")]
    assert value["record_count"] == 3
    assert value["component_count"] == 2
    assert value["an_record_count"] == value["an_component_count"] == 1
    assert value["dlc_record_count"] == 2
    assert value["dlc_component_count"] == 2
    assert value["dialect_support"] == "shared_an_dlc"


def test_concentration_and_effective_component_count_use_canonical_rows() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    for index in range(3):
        audit.add_record(
            _record(
                f"r{index}",
                "c1" if index < 2 else "c2",
                "dlc",
                "train",
                _target(f"e{index}", "a", f"s{index}"),
            )
        )
    value = _rows(audit)[("train", "a")]
    assert value["largest_record_share"] == pytest.approx(1 / 3)
    assert value["largest_component_share"] == pytest.approx(2 / 3)
    assert value["top_5_components_share"] == 1.0
    assert value["effective_component_count"] == pytest.approx(9 / 5)


@pytest.mark.parametrize(
    ("rows", "components", "expected"),
    [
        (0, 0, "absent"),
        (19, 3, "insufficient"),
        (20, 2, "insufficient"),
        (20, 3, "fragile"),
        (100, 10, "usable"),
        (1000, 49, "usable"),
        (1000, 50, "broad"),
    ],
)
def test_train_support_tier_boundaries(rows: int, components: int, expected: str) -> None:
    assert train_support_tier(rows, components) == expected


@pytest.mark.parametrize(
    ("rows", "components", "expected"),
    [
        (0, 0, "unobservable"),
        (9, 2, "fragile_validation"),
        (10, 1, "fragile_validation"),
        (10, 2, "observable"),
    ],
)
def test_validation_support_tier_boundaries(
    rows: int, components: int, expected: str
) -> None:
    assert validation_support_tier(rows, components) == expected


def test_head_recommendation_uses_frozen_priority() -> None:
    status, reasons = recommend_head_trainability(
        vocabulary_size=4,
        train_tiers=("absent", "usable", "usable", "usable"),
        validation_tiers=("unobservable", "observable", "observable", "observable"),
        available_train_components=19,
        majority_share=0.9,
        max_to_min_nonzero_ratio=100,
        normalized_entropy=0.1,
    )
    assert status == "descriptive_only"
    assert "fewer_than_20_available_train_components" in reasons
    status, _ = recommend_head_trainability(
        vocabulary_size=4,
        train_tiers=("insufficient", "usable", "usable", "usable"),
        validation_tiers=("observable",) * 4,
        available_train_components=20,
        majority_share=0.4,
        max_to_min_nonzero_ratio=2,
        normalized_entropy=0.9,
    )
    assert status == "insufficient_support"
    status, _ = recommend_head_trainability(
        vocabulary_size=2,
        train_tiers=("fragile", "usable"),
        validation_tiers=("observable", "observable"),
        available_train_components=20,
        majority_share=0.5,
        max_to_min_nonzero_ratio=1,
        normalized_entropy=1.0,
    )
    assert status == "trainable_with_reweighting"
    status, _ = recommend_head_trainability(
        vocabulary_size=2,
        train_tiers=("usable", "broad"),
        validation_tiers=("fragile_validation", "observable"),
        available_train_components=20,
        majority_share=0.75,
        max_to_min_nonzero_ratio=3,
        normalized_entropy=0.8,
    )
    assert status == "trainable_with_reweighting"
    status, _ = recommend_head_trainability(
        vocabulary_size=2,
        train_tiers=("usable", "broad"),
        validation_tiers=("observable", "observable"),
        available_train_components=20,
        majority_share=0.5,
        max_to_min_nonzero_ratio=1,
        normalized_entropy=1.0,
    )
    assert status == "trainable"


def test_candidate_weights_use_train_only_and_unsupported_is_null() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    for index in range(20):
        audit.add_record(
            _record(
                f"r{index}",
                f"c{index}",
                "dlc",
                "train",
                _target(f"e{index}", "a" if index < 15 else "b", f"s{index}"),
            )
        )
    for index in range(10):
        audit.add_record(
            _record(
                f"v{index}",
                f"vc{index}",
                "dlc",
                "validation",
                _target(f"ve{index}", "a", f"vs{index}"),
            )
        )
    rows = audit.class_rows()
    summaries = audit.head_summaries(rows)
    first = candidate_class_weights(rows, summaries)
    mutated = copy.deepcopy(rows)
    for row in mutated:
        if row["split"] == "validation":
            row["canonical_target_row_count"] = 999999
    second = candidate_class_weights(mutated, summaries)
    assert first == second
    for vector in first["heads"][0]["vectors"].values():
        assert vector[2]["status"] == "unsupported"
        assert vector[2]["weight"] is None
        nonzero = [row["weight"] for row in vector if row["weight"] is not None]
        assert sum(nonzero) / len(nonzero) == pytest.approx(1.0)


def test_quality_compatibility_projection_collapses_only_two_corrected_classes() -> None:
    record = RecordTargetObservations(
        "r",
        "c",
        "dlc",
        "train",
        (
            EntityTargetObservation("quality", "e1", "s1", "augmented seventh chord", True, False),
            EntityTargetObservation("quality", "e2", "s2", "augmented major tetrachord", True, False),
            EntityTargetObservation("quality", "e3", "s3", "major triad", True, False),
        ),
    )
    projected = project_quality_record(record)
    assert [row.class_value for row in projected.targets] == [
        "augmented triad",
        "augmented triad",
        "major triad",
    ]
    assert get_vocabulary(CORRECTED_QUALITY_VOCABULARY_ID).class_count == 17
    assert get_vocabulary(COMPATIBILITY_QUALITY_VOCABULARY_ID).class_count == 15


def test_roman_vocabulary_stays_184_and_repairs_literal() -> None:
    vocabulary = get_vocabulary("analysisgnn.roman-numeral-corrected-v1")
    assert vocabulary.class_count == 184
    assert "none" not in vocabulary.labels
    assert "#VIIbvio7" not in vocabulary.labels
    assert "#VII" in vocabulary.labels
    assert "bvio7" in vocabulary.labels


def _joint_sidecar() -> dict[str, object]:
    values = {
        "local_key": "C",
        "primary_degree": "1",
        "secondary_degree": "5",
        "quality": "augmented seventh chord",
        "inversion": "0",
    }
    targets = {
        task: {
            "available": True,
            "masked": False,
            "canonical_value": value,
            "provenance": {"source_field": task, "source_row_ordinal": 1},
        }
        for task, value in values.items()
    }
    return {
        "record_id": "r",
        "source_component_id": "c",
        "dialect": "dlc",
        "entities": [
            {"canonical_entity_id": "h1", "entity_type": "harmonic_event", "targets": targets},
            {"canonical_entity_id": "n1", "entity_type": "note", "targets": {}},
            {"canonical_entity_id": "n2", "entity_type": "note", "targets": {}},
        ],
        "relations": [
            {"relation": "note_to_harmonic_event", "source_entity_id": "n1", "target_entity_id": "h1"},
            {"relation": "note_to_harmonic_event", "source_entity_id": "n2", "target_entity_id": "h1"},
        ],
    }


def test_joint_tuple_counts_separate_event_note_and_canonical_units() -> None:
    observations = joint_observations_from_sidecar(_joint_sidecar(), split="train")
    assert sum(row.mode == "corrected_harmonic_event" for row in observations) == 1
    assert sum(row.mode == "compatibility_note" for row in observations) == 2
    assert {row.values[3] for row in observations if row.mode == "compatibility_note"} == {
        "augmented triad"
    }
    audit = JointTupleAccumulator()
    audit.add_record(observations)
    rows = audit.rows()
    event = next(row for row in rows if row["mode"] == "corrected_harmonic_event")
    note = next(row for row in rows if row["mode"] == "compatibility_note")
    assert event["row_count"] == event["canonical_harmonic_target_row_count"] == 1
    assert note["row_count"] == 2
    assert note["canonical_harmonic_target_row_count"] == 1


def test_unseen_validation_tuple_is_relative_to_train() -> None:
    train = JointTupleObservation(
        "corrected_harmonic_event", "train", "r", "c", "dlc", "h1", "h1",
        ("C", "1", "5", "major triad", "0"),
    )
    validation = JointTupleObservation(
        "corrected_harmonic_event", "validation", "v", "vc", "dlc", "h2", "h2",
        ("d", "2", "5", "minor triad", "1"),
    )
    audit = JointTupleAccumulator()
    audit.add_record((train,))
    audit.add_record((validation,))
    summary = audit.summary(audit.rows())["corrected_harmonic_event"]
    assert summary["validation_tuples_unseen_in_train"] == [
        {
            "local_key": "d",
            "primary_degree": "2",
            "secondary_degree": "5",
            "quality": "minor triad",
            "inversion": "1",
        }
    ]


def test_test_target_loader_is_never_called() -> None:
    calls: list[str] = []
    assignments = [
        {"record_id": "r", "split": "train"},
        {"record_id": "v", "split": "validation"},
        {"record_id": "t", "split": "test"},
    ]

    def loader(record_id: str, split: str) -> str:
        if split == "test":
            raise AssertionError("TEST target loader must not run")
        calls.append(record_id)
        return record_id

    loaded, lock = load_train_validation_only(assignments, loader)
    assert loaded == calls == ["r", "v"]
    assert lock == {
        "test_assignments_seen": True,
        "test_assignment_record_count": 1,
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }


def test_serialization_order_and_fingerprint_are_deterministic() -> None:
    audit = ClassBalanceAccumulator((TASK,))
    audit.add_record(_record("r", "c", "dlc", "train", _target("e", "b", "s")))
    rows = audit.class_rows()
    assert [(row["split"], row["class_id"]) for row in rows] == [
        ("train", 0), ("train", 1), ("train", 2),
        ("validation", 0), ("validation", 1), ("validation", 2),
    ]
    summaries = audit.head_summaries(rows)
    weights = candidate_class_weights(rows, summaries)
    recommendations = {"heads": []}
    first = semantic_fingerprint(
        class_rows=rows,
        head_summaries=summaries,
        joint_rows=[],
        recommendations=recommendations,
        weights=weights,
    )
    second = semantic_fingerprint(
        class_rows=rows,
        head_summaries=summaries,
        joint_rows=[],
        recommendations=recommendations,
        weights=weights,
    )
    assert first == second == fingerprint(
        {
            "class_balance_contract": class_balance_contract()["fingerprint"],
            "class_rows": rows,
            "head_summaries": summaries,
            "joint_rows": [],
            "recommendations": recommendations,
            "weights": weights,
        }
    )
    assert class_balance_contract()["schema"] == CLASS_BALANCE_SCHEMA


def test_sidecar_projection_uses_task_registry_entity_types() -> None:
    sidecar = _joint_sidecar()
    # Fill all production harmonic targets as masked so the extractor can
    # project the registry-defined harmonic entity without inventing a class.
    harmonic = sidecar["entities"][0]  # type: ignore[index]
    for task_id in (
        "tonicized_key", "root", "bass", "roman_numeral", "pitch_class_set",
        "harmonic_rhythm", "pedal",
    ):
        harmonic["targets"][task_id] = {  # type: ignore[index]
            "available": False,
            "masked": True,
            "canonical_value": None,
            "provenance": {},
        }
    record = observations_from_sidecar(sidecar, split="train")
    harmonic_task_ids = {row.task_id for row in record.targets if row.entity_id == "h1"}
    assert {"local_key", "quality", "roman_numeral"} <= harmonic_task_ids
    assert "note_degree" not in harmonic_task_ids
