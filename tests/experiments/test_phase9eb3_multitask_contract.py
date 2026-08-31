from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_critic.experiments.analysisgnn.contracts import (
    EXPECTED_RECORD_COUNT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_FINGERPRINT,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    ANALYSISGNN_COMMIT,
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_NAMESPACE,
    EXPECTED_FULL_COUNTS,
    EXPECTED_PAPER_COUNTS,
    OVERLAP_AN_PEERS,
    OVERLAP_EXCLUSION_NAMES,
    PINNED_CODE_HEADS,
    PRODUCTION_TASKS,
    VOCABULARIES,
    AnalysisGNNMultitaskContractError,
    get_task,
    get_entity_mappings,
    get_vocabulary,
    materialize_target_sidecar,
    materialize_target_sidecar_descriptor,
    metric_contract,
    pinned_code_reference_registry,
    production_task_registry,
    require_test_evaluation_unlock,
    load_split_assignments,
    sidecar_contract_counts,
    stable_split_assignments,
    test_lock_manifest as build_test_lock_manifest,
    validate_loaded_registry,
    validate_split,
    vocabularies_payload,
)
from tests.adapters.test_dilemmadata import CORPUS, _accepted


def _synthetic_records() -> tuple[SimpleNamespace, ...]:
    rows = []
    for index in range(1619):
        # A deterministic 112-record set of two-dialect components exercises the
        # component boundary while retaining the frozen corpus size.
        component = index - 1 if index % 29 == 1 else index
        rows.append(
            SimpleNamespace(
                record_id=f"record:{index:04d}",
                dialect="an_joint" if index % 5 == 0 else "dlc",
                source_group_id=f"component:{component:04d}",
            )
        )
    return tuple(rows)


def _availability(records) -> dict[str, dict[str, bool]]:
    return {
        row.record_id: {
            task.task_id: (task.task_id != "cadence" or row.dialect == "dlc")
            for task in PRODUCTION_TASKS
        }
        for row in records
    }


def test_universe_and_exclusion_locks_are_exact() -> None:
    assert EXPECTED_FULL_COUNTS == {"an_joint": 353, "dlc": 1280, "total": 1633}
    assert EXPECTED_PAPER_COUNTS == {"an_joint": 353, "dlc": 1266, "total": 1619}
    assert len(OVERLAP_EXCLUSION_NAMES) == 14
    assert len(set(OVERLAP_EXCLUSION_NAMES)) == 14
    assert set(OVERLAP_EXCLUSION_NAMES) == set(OVERLAP_AN_PEERS)
    assert "monteverdi_madrigals_5-04d" not in OVERLAP_EXCLUSION_NAMES
    assert len(set(OVERLAP_AN_PEERS.values())) == 14


def test_task_inventory_and_alias_evidence_are_explicit() -> None:
    assert len(PINNED_CODE_HEADS) == 21
    assert len(PRODUCTION_TASKS) == 20
    assert len({row.task_id for row in PRODUCTION_TASKS}) == 20
    reference = pinned_code_reference_registry()
    assert reference["head_count"] == 21
    by_code = {row["task_name_in_code"]: row for row in reference["rows"]}
    assert by_code["organ_point"]["canonical_task_id"] == "pedal"
    assert by_code["organ_point"]["status"] == "alias_normalized"
    assert by_code["downbeat"]["canonical_task_id"] == "metrical_strength"
    assert by_code["staff"]["status"] == "code_only"
    assert reference["external_commit"] == ANALYSISGNN_COMMIT


def test_vocabularies_are_unique_contiguous_and_match_tasks() -> None:
    assert len({row.vocabulary_id for row in VOCABULARIES}) == len(VOCABULARIES)
    payload = vocabularies_payload()
    registry = production_task_registry()
    validate_loaded_registry(registry, payload)
    for task in PRODUCTION_TASKS:
        vocabulary = get_vocabulary(task.vocabulary_id)
        assert task.class_count == len(vocabulary.labels)
        assert len(vocabulary.labels) == len(set(vocabulary.labels))


def test_quality_vocabulary_repairs_missing_and_dlc_conflation() -> None:
    vocabulary = get_vocabulary("analysisgnn.quality-corrected-v1")
    assert len(vocabulary.labels) == 17
    assert "None" not in vocabulary.labels
    assert "augmented seventh chord" in vocabulary.labels
    assert "augmented major tetrachord" in vocabulary.labels
    assert vocabulary.normalize("+7") == "augmented seventh chord"
    assert vocabulary.normalize("+M7") == "augmented major tetrachord"
    assert vocabulary.normalize("augmented seventh") == "augmented seventh chord"


def test_roman_numeral_vocabulary_repairs_concatenated_literal() -> None:
    vocabulary = get_vocabulary("analysisgnn.roman-numeral-corrected-v1")
    assert len(vocabulary.labels) == 184
    assert "none" not in vocabulary.labels
    assert "#VII" in vocabulary.labels
    assert "bvio7" in vocabulary.labels
    assert "#VIIbvio7" not in vocabulary.labels


def test_unknown_task_and_vocabulary_fail_closed() -> None:
    with pytest.raises(AnalysisGNNMultitaskContractError, match="unknown task"):
        get_task("not-a-task")
    with pytest.raises(AnalysisGNNMultitaskContractError, match="unknown vocabulary"):
        get_vocabulary("not-a-vocabulary")


def test_registry_rejects_length_and_duplicate_class_ids() -> None:
    registry = production_task_registry()
    vocabularies = vocabularies_payload()
    first = vocabularies["vocabularies"][0]
    first["classes"][1]["class_id"] = 0
    with pytest.raises(AnalysisGNNMultitaskContractError, match="class ID"):
        validate_loaded_registry(registry, vocabularies)

    vocabularies = vocabularies_payload()
    registry["tasks"][0]["class_count"] += 1
    with pytest.raises(AnalysisGNNMultitaskContractError, match="length mismatch"):
        validate_loaded_registry(registry, vocabularies)


def test_split_is_component_safe_deterministic_and_order_independent() -> None:
    records = _synthetic_records()
    availability = _availability(records)
    first = stable_split_assignments(records, availability)
    second = stable_split_assignments(tuple(reversed(records)), availability)
    assert first == second
    assert Counter(row.split for row in first) == {
        "train": 1295,
        "validation": 162,
        "test": 162,
    }
    assert all(row.assignment_algorithm == ASSIGNMENT_ALGORITHM for row in first)
    assert all(row.assignment_namespace == ASSIGNMENT_NAMESPACE for row in first)
    validate_split(first)
    component_splits: dict[str, set[str]] = {}
    for row in first:
        component_splits.setdefault(row.source_component_id, set()).add(row.split)
    assert all(len(values) == 1 for values in component_splits.values())


def test_split_does_not_call_python_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _synthetic_records()
    availability = _availability(records)

    def forbidden_hash(_value: object) -> int:
        raise AssertionError("Python hash() must not participate in assignment")

    monkeypatch.setattr("builtins.hash", forbidden_hash)
    assignments = stable_split_assignments(records, availability)
    assert len(assignments) == 1619


def test_split_rejects_duplicate_records_and_component_overlap() -> None:
    records = _synthetic_records()
    availability = _availability(records)
    with pytest.raises(AnalysisGNNMultitaskContractError, match="duplicate record"):
        stable_split_assignments((*records, records[0]), availability)
    assignments = list(stable_split_assignments(records, availability))
    duplicate_component = replace(
        assignments[0],
        record_id="record:duplicate-component",
        split="test" if assignments[0].split != "test" else "train",
    )
    with pytest.raises(AnalysisGNNMultitaskContractError, match="component overlap"):
        validate_split((*assignments, duplicate_component))


@pytest.mark.parametrize("record_id", ["an:training:same", "dlc:demo:same"])
def test_shared_entity_materialization_is_deterministic(record_id: str) -> None:
    accepted = _accepted(CORPUS, record_id)
    first = materialize_target_sidecar(accepted)
    second = materialize_target_sidecar(accepted)
    assert first == second
    harmonic = [row for row in first["entities"] if row["entity_type"] == "harmonic_event"]
    assert harmonic
    for row in harmonic:
        entity_id = row["canonical_entity_id"]
        for task_id in (
            "local_key", "primary_degree", "secondary_degree", "quality", "inversion",
            "roman_numeral", "pitch_class_set", "harmonic_rhythm",
        ):
            assert row["targets"][task_id]["canonical_entity_id"] == entity_id
    assert all(
        row["targets"]["quality"]["canonical_entity_id"]
        == row["targets"]["inversion"]["canonical_entity_id"]
        for row in harmonic
    )
    descriptor = materialize_target_sidecar_descriptor(accepted)
    assert descriptor == materialize_target_sidecar_descriptor(accepted)
    assert descriptor["entity_counts"] == first["entity_counts"]
    assert descriptor["relation_counts"] == first["relation_counts"]


def test_missing_is_masked_not_class_zero_and_masks_are_independent() -> None:
    sidecar = materialize_target_sidecar(_accepted(CORPUS, "dlc:demo:same"))
    harmonic = [row for row in sidecar["entities"] if row["entity_type"] == "harmonic_event"]
    secondary = [row["targets"]["secondary_degree"] for row in harmonic]
    assert any(state["masked"] is True for state in secondary)
    assert all(state["canonical_value"] is None for state in secondary if state["masked"])
    all_states = [state for row in sidecar["entities"] for state in row["targets"].values()]
    assert any(state["available"] is True for state in all_states)
    assert any(state["available"] is False for state in all_states)


def test_note_and_cross_level_entity_mappings_are_valid() -> None:
    sidecar = materialize_target_sidecar(_accepted(CORPUS, "dlc:demo:same"))
    entities = {row["canonical_entity_id"]: row for row in sidecar["entities"]}
    note_ids = {entity_id for entity_id, row in entities.items() if row["entity_type"] == "note"}
    relations = sidecar["relations"]
    assert note_ids
    assert {row["source_entity_id"] for row in relations if row["relation"] == "note_to_onset"} == note_ids
    assert {row["source_entity_id"] for row in relations if row["relation"] == "note_to_harmonic_event"} == note_ids
    assert any(row["relation"] == "onset_to_beat" for row in relations)
    assert any(row["relation"] == "beat_to_measure" for row in relations)
    assert get_entity_mappings(sidecar) == tuple(relations)
    first_note = next(iter(note_ids))
    assert all(
        row["source_entity_id"] == first_note
        for row in get_entity_mappings(sidecar, entity_id=first_note)
    )


def test_absent_optional_families_do_not_exclude_a_record() -> None:
    sidecar = materialize_target_sidecar(_accepted(CORPUS, "an:training:same"))
    assert sidecar["record_id"] == "an:training:same"
    onset = [row for row in sidecar["entities"] if row["entity_type"] == "onset"]
    for row in onset:
        for task_id in ("cadence", "phrase", "section"):
            assert row["targets"][task_id]["available"] is False
            assert row["targets"][task_id]["missing_reason"] == "unsupported_dialect"


def test_repair_evidence_is_target_only_not_an_entity_or_task() -> None:
    sidecar = materialize_target_sidecar(_accepted(CORPUS, "an:training:same"))
    assert "repair_evidence_fingerprint" in sidecar
    assert "repair_evidence" not in {row.task_id for row in PRODUCTION_TASKS}
    assert all(row["entity_type"] != "repair" for row in sidecar["entities"])


def test_joint_metric_contract_requires_one_shared_harmonic_entity() -> None:
    contract = metric_contract()
    joint = contract["joint_metrics"][0]
    assert joint["components"] == [
        "local_key", "primary_degree", "secondary_degree", "quality", "inversion"
    ]
    assert joint["entity_type"] == "harmonic_event"
    assert joint["undefined_payload"] == {
        "accuracy": None,
        "available": False,
        "support": 0,
        "undefined_reason": "no rows satisfy the joint component contract",
    }


def test_test_lock_requires_explicit_authorization() -> None:
    records = _synthetic_records()
    assignments = stable_split_assignments(records, _availability(records))
    lock = build_test_lock_manifest(assignments)
    assert lock["test_assignment_frozen"] is True
    assert lock["test_metrics_computed"] is False
    assert lock["test_targets_used_for_model_evaluation"] is False
    with pytest.raises(AnalysisGNNMultitaskContractError, match="TEST evaluation is locked"):
        require_test_evaluation_unlock()
    require_test_evaluation_unlock(explicit_allow=True)


def test_phase9eb1_negative_pilot_contract_is_unchanged() -> None:
    assert EXPECTED_RECORD_COUNT == 719
    assert EXPECTED_SPLIT_COUNTS == {"train": 577, "validation": 71, "test": 71}
    assert EXPECTED_SPLIT_FINGERPRINT == (
        "58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e"
    )


def test_split_loader_checks_sha_manifest_and_component_overlap(tmp_path: Path) -> None:
    records = _synthetic_records()
    assignments = stable_split_assignments(records, _availability(records))
    path = tmp_path / "split.jsonl"
    payload = "".join(
        json.dumps(
            {
                "assignment_algorithm": row.assignment_algorithm,
                "assignment_namespace": row.assignment_namespace,
                "dialect": row.dialect,
                "record_id": row.record_id,
                "source_component_id": row.source_component_id,
                "split": row.split,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in assignments
    )
    path.write_text(payload, encoding="utf-8")
    digest = sha256(payload.encode("utf-8")).hexdigest()
    loaded = load_split_assignments(
        path,
        manifest_record_ids=(row.record_id for row in records),
        expected_sha256=digest,
    )
    assert loaded == assignments
    with pytest.raises(AnalysisGNNMultitaskContractError, match="fingerprint mismatch"):
        load_split_assignments(
            path,
            manifest_record_ids=(row.record_id for row in records),
            expected_sha256="0" * 64,
        )
    with pytest.raises(AnalysisGNNMultitaskContractError, match="outside manifest"):
        load_split_assignments(
            path,
            manifest_record_ids=(row.record_id for row in records[:-1]),
            expected_sha256=digest,
        )
