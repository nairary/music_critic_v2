from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_critic.adapters import dilemmadata_eda as module
from music_critic.adapters.dilemmadata_eda import (
    DILEMMADATA_EDA_ADAPTER_IDENTITY,
    DILEMMADATA_EDA_CONTRACT_BASE,
    DILEMMADATA_RAW_EXTENSION_NAMESPACE,
    DILEMMADATA_SOURCE_IDENTITY,
    DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE,
    DilemmadataEDAAdapter,
    DilemmadataEDARequest,
    dilemmadata_surface_value_identities,
)
from music_critic.eda import (
    ComputationStatus,
    CorpusId,
    EDAAdapterRegistry,
    EDAContractError,
    EvidenceScope,
    ExecutionMode,
    ObservationUnit,
    ProjectionMappingState,
    RAW_METRIC_CATALOG,
    SplitScope,
    canonical_report_bytes,
    load_report,
    report_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def eda_request() -> DilemmadataEDARequest:
    return DilemmadataEDARequest(
        repository_root=ROOT,
        repository_commit=DILEMMADATA_EDA_CONTRACT_BASE,
    )


@pytest.fixture()
def registry() -> EDAAdapterRegistry:
    value = EDAAdapterRegistry()
    value.register(DilemmadataEDAAdapter())
    return value


def _task(report, task_id: str, split: SplitScope):
    return next(
        row
        for row in report.semantic_payload.tasks
        if row.source_task_id == task_id and row.split_scope == split
    )


def _extension_row(report, split: SplitScope, row_id: str):
    extension = next(
        row
        for row in report.semantic_payload.extensions
        if row.split_scope == split
    )
    return next(row for row in extension.rows if row.row_id == row_id)


def _count_values(row) -> dict[str, int]:
    return {count.name: count.value for count in row.counts}


def test_registration_and_externally_bound_identities() -> None:
    adapter = DilemmadataEDAAdapter()
    assert adapter.corpus == CorpusId.DILEMMADATA
    assert adapter.extension_namespaces == (
        DILEMMADATA_RAW_EXTENSION_NAMESPACE,
        DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE,
    )
    assert DILEMMADATA_EDA_ADAPTER_IDENTITY.identity == (
        "music_critic.adapters.dilemmadata_eda"
    )
    assert DILEMMADATA_EDA_ADAPTER_IDENTITY.version == "1.0.1"
    assert DILEMMADATA_EDA_ADAPTER_IDENTITY.fingerprint == (
        "efc89198d4a2e644e746ea7fe173ce60ae7ab7b9b646cca75396a76c78ec96c4"
    )
    assert DILEMMADATA_SOURCE_IDENTITY.fingerprint == (
        "8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312"
    )


def test_raw_manifest_replay_is_partial_target_free_and_inventory_exact(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_raw(CorpusId.DILEMMADATA, eda_request)
    assert report.envelope.evidence_scope == EvidenceScope.MANIFEST_REPLAY
    assert report.envelope.execution_mode == ExecutionMode.MANIFEST_REPLAY
    assert report.envelope.split_scope == SplitScope.ALL
    assert report.envelope.observation_units == (
        ObservationUnit.CANONICAL_WORK,
        ObservationUnit.RECORD,
    )
    assert len(report.semantic_payload.metrics) == len(RAW_METRIC_CATALOG) == 32
    assert all(item.target_free for item in report.envelope.input_manifests)

    metrics = {row.metric_id: row for row in report.semantic_payload.metrics}
    assert metrics["discovered_records"].count.value == 1633
    assert metrics["accepted_records"].count.value == 719
    assert metrics["quarantined_records"].count.value == 914
    assert metrics["duplicate_candidates"].count.value == 60
    assert {
        row.category: row.count.value
        for row in metrics["conversion_outcomes"].categories
    } == {"accepted": 719, "quarantined": 914}
    assert metrics["graph_node_counts"].coverage.status == (
        ComputationStatus.NOT_COMPUTED
    )
    assert metrics["graph_node_counts"].categories == ()
    assert report.semantic_payload.graph_evidence.target_free is None

    extension = report.semantic_payload.extensions[0]
    assert extension.namespace == DILEMMADATA_RAW_EXTENSION_NAMESPACE
    assert extension.target_free is True
    rows = {row.row_id: row for row in extension.rows}
    assert {count.name: count.value for count in rows["dialect_inventory"].counts} == {
        "an_joint_records": 353,
        "dlc_records": 1280,
    }
    assert {
        count.name: count.value
        for count in rows["dialect_conversion_outcomes"].counts
    } == {
        "an_joint_accepted": 108,
        "an_joint_quarantined": 245,
        "dlc_accepted": 611,
        "dlc_quarantined": 669,
    }
    assert _count_values(rows["paper_candidate_inventory"]) == {
        "an_joint_records": 353,
        "dlc_records": 1266,
        "paper_records": 1619,
        "selection_exclusions": 14,
    }
    assert _count_values(rows["paper_candidate_split"]) == {
        "test_records": 162,
        "train_records": 1295,
        "validation_records": 162,
    }
    assert _count_values(rows["paper_candidate_component_split"]) == {
        "test_components": 151,
        "train_components": 1209,
        "validation_components": 147,
    }
    assert _count_values(rows["common_subset_inventory"]) == {
        "an_joint_records": 108,
        "dlc_records": 611,
        "outside_subset_records": 914,
        "subset_records": 719,
    }
    assert _count_values(rows["common_subset_split"]) == {
        "test_records": 71,
        "train_records": 577,
        "validation_records": 71,
    }
    assert _count_values(rows["common_subset_component_split"]) == {
        "test_components": 71,
        "train_components": 565,
        "validation_components": 71,
    }
    assert rows["paper_candidate_component_split"].coverage.observation_unit == (
        ObservationUnit.CANONICAL_WORK
    )
    assert extension.work_identity.identity == (
        "dilemmadata.phase9eb3.source_component"
    )
    assert report_fingerprint(report) == report.semantic_fingerprint


def test_supervision_exposes_all_native_families_and_marks_unreplayed_splits(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    assert report.envelope.evidence_scope == EvidenceScope.MANIFEST_REPLAY
    assert report.envelope.execution_mode == ExecutionMode.MANIFEST_REPLAY
    assert report.envelope.split_scope == SplitScope.TRAIN_VALIDATION
    assert len(report.semantic_payload.tasks) == 44
    observed = [
        row
        for row in report.semantic_payload.tasks
        if row.status == ComputationStatus.OBSERVED
    ]
    assert len(observed) == 8
    assert all(
        row.availability is None and not row.class_support
        for row in report.semantic_payload.tasks
        if row not in observed
    )
    assert all(
        row.reason_code
        == "dilemmadata.manifest.family_split_distribution_not_replayed"
        for row in report.semantic_payload.tasks
        if row not in observed
    )

    an_inversion = _task(
        report, "dilemmadata.an.chord.inversion", SplitScope.TRAIN
    )
    dlc_inversion = _task(
        report, "dilemmadata.dlc.chord.inversion", SplitScope.TRAIN
    )
    assert (
        an_inversion.availability.available,
        an_inversion.availability.masked,
        an_inversion.availability.missing,
        an_inversion.availability.unsupported,
    ) == (2, 1, 0, 0)
    assert (
        dlc_inversion.availability.available,
        dlc_inversion.availability.masked,
        dlc_inversion.availability.missing,
        dlc_inversion.availability.unsupported,
    ) == (3, 0, 0, 0)
    assert sum(row.occurrence_count.value for row in an_inversion.class_support) == 2
    assert all(row.available_only for row in observed for row in row.class_support)

    dlc_quality_validation = _task(
        report, "dilemmadata.dlc.chord.quality", SplitScope.VALIDATION
    )
    assert (
        dlc_quality_validation.availability.available,
        dlc_quality_validation.availability.masked,
        dlc_quality_validation.availability.missing,
        dlc_quality_validation.availability.unsupported,
    ) == (1, 0, 1, 0)
    assert all(
        support.unique_work_count.status == ComputationStatus.OBSERVED
        and support.unique_work_count.observation_unit == ObservationUnit.CANONICAL_WORK
        for row in observed
        for support in row.class_support
    )


def test_an_and_dlc_inversion_surface_values_are_never_merged(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    fixture = json.loads(
        (ROOT / "tests/fixtures/dilemmadata/eda_supervision_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    identities = dilemmadata_surface_value_identities(fixture)
    assert [(row.dialect, row.source_value) for row in identities] == [
        ("an_joint", "2"),
        ("dlc", "2"),
        ("dlc", "42"),
    ]
    assert len({row.identity for row in identities}) == 3

    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    an = _task(report, "dilemmadata.an.chord.inversion", SplitScope.TRAIN)
    dlc = _task(report, "dilemmadata.dlc.chord.inversion", SplitScope.TRAIN)
    an_projection = next(
        row for row in an.projections if row.source_value.source_value == "2"
    )
    dlc_projection = next(
        row for row in dlc.projections if row.source_value.source_value == "2"
    )
    assert an_projection.projected_value == "second"
    assert dlc_projection.projected_value == "third"
    assert an_projection.source_value.identity != dlc_projection.source_value.identity
    counts = dlc.projection_availability[0]
    assert counts.exact == 3
    assert counts.unsupported == 0
    assert module._projection_for(
        "dilemmadata.dlc.chord.inversion", "dlc", "42"
    ) == (ProjectionMappingState.EXACT, "third")
    alias = _extension_row(
        report, SplitScope.TRAIN, "dlc_42_surface_resolution"
    )
    assert alias.payload == {
        "normalized_native_label": "2",
        "ordinal_label": "third",
        "resolution": "source_alias_then_exact_approved_registry",
        "surface_spelling": "42",
    }
    assert _count_values(alias) == {"resolved_surface_rows": 1}


def test_only_exact_approved_projection_registry_is_used(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    observed = [
        row
        for row in report.semantic_payload.tasks
        if row.status == ComputationStatus.OBSERVED
    ]
    registries = {
        projection.mapping_registry
        for row in observed
        for projection in row.projections
    }
    assert len(registries) == 1
    registry_identity = registries.pop()
    assert registry_identity.identity == "music_critic.dilemmadata.common_harmonic"
    assert registry_identity.fingerprint == (
        "bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e"
    )
    an_quality = _task(
        report, "dilemmadata.an.chord.quality", SplitScope.TRAIN
    )
    assert {row.mapping_state for row in an_quality.projections} == {
        ProjectionMappingState.EXACT
    }
    assert all(row.projection_availability for row in observed)


def test_test_gate_calls_spies_only_for_train_validation() -> None:
    descriptor_calls: list[tuple[str, SplitScope]] = []
    loader_calls: list[tuple[str, SplitScope]] = []

    def descriptor_spy(record_id: str, split: SplitScope) -> None:
        assert split != SplitScope.TEST
        descriptor_calls.append((record_id, split))

    def loader_spy(record_id: str, split: SplitScope) -> None:
        assert split != SplitScope.TEST
        loader_calls.append((record_id, split))

    report = DilemmadataEDAAdapter().build_supervision_eda(
        DilemmadataEDARequest(
            repository_root=ROOT,
            repository_commit=DILEMMADATA_EDA_CONTRACT_BASE,
            descriptor_observer=descriptor_spy,
            target_loader_observer=loader_spy,
        )
    )
    assert len(descriptor_calls) == len(loader_calls) == 9
    assert descriptor_calls == loader_calls
    assert {split for _, split in descriptor_calls} == {
        SplitScope.TRAIN,
        SplitScope.VALIDATION,
    }
    lock = report.semantic_payload.test_lock
    assert lock.test_assignment_count.value == 162
    assert lock.test_descriptor_resolution_count.value == 0
    assert lock.test_target_loader_call_count.value == 0
    assert lock.test_target_records_opened.value == 0
    assert lock.test_target_rows_loaded.value == 0
    assert lock.test_targets_read is False
    assert lock.test_targets_used_for_eda is False
    assert lock.test_targets_used_for_model_evaluation is False
    assert lock.test_class_distributions_emitted is False
    assert lock.test_coverage_emitted is False
    assert lock.test_cooccurrence_emitted is False


def test_target_bearing_evidence_is_loaded_only_after_descriptor_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original = module._load_json
    target_bearing = {
        "eda_supervision_fixture.json",
        "target_manifest.json",
        "common_harmonic_manifest.json",
        "phase9eb3_multitask_contract.json",
        "phase9eb4_class_balance_audit.json",
        "phase9eb5a_transposition_audit.json",
        "phase9eb5b_training_policy.json",
        "phase9eb5e_full_training_results.json",
        "phase9eb5h_full_orbit_profile.json",
    }

    def observed_load(path: Path):
        if path.name in target_bearing:
            events.append(f"target_open:{path.name}")
        return original(path)

    monkeypatch.setattr(module, "_load_json", observed_load)
    report = DilemmadataEDAAdapter().build_supervision_eda(
        DilemmadataEDARequest(
            repository_root=ROOT,
            repository_commit=DILEMMADATA_EDA_CONTRACT_BASE,
            descriptor_observer=lambda _record, _split: events.append("descriptor"),
        )
    )
    assert report.semantic_payload.test_lock.test_assignment_count.value == 162
    assert events[0] == "descriptor"
    assert events.count("target_open:eda_supervision_fixture.json") == 1
    opened = {
        event.removeprefix("target_open:")
        for event in events
        if event.startswith("target_open:")
    }
    assert opened == target_bearing


@pytest.mark.parametrize(
    "field",
    ["canonical_work_id", "source_group_id", "lineage_ids"],
)
def test_cross_split_identity_leakage_fails_for_all_identity_planes(field: str) -> None:
    left = {
        "canonical_work_id": "work-a",
        "source_group_id": "group-a",
        "lineage_ids": ["lineage-a"],
        "split": "train",
    }
    right = {
        "canonical_work_id": "work-a" if field == "canonical_work_id" else "work-b",
        "source_group_id": "group-a" if field == "source_group_id" else "group-b",
        "lineage_ids": ["lineage-a" if field == "lineage_ids" else "lineage-b"],
        "split": "validation",
    }
    with pytest.raises(EDAContractError) as raised:
        module._assert_split_atomic((left, right), field)
    assert raised.value.category == "dilemmadata.eda.identity_leakage"


def test_fixture_imbalance_and_cooccurrence_are_split_and_dialect_explicit(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    an_train = _extension_row(
        report, SplitScope.TRAIN, "an_quality_inversion_cooccurrence"
    )
    dlc_train = _extension_row(
        report, SplitScope.TRAIN, "dlc_quality_inversion_cooccurrence"
    )
    combined_train = _extension_row(
        report, SplitScope.TRAIN, "combined_quality_inversion_cooccurrence"
    )
    assert an_train.counts[0].value == 2
    assert dlc_train.counts[0].value == 3
    assert combined_train.counts[0].value == 5
    assert combined_train.coverage.denominator == 6

    imbalance = _extension_row(
        report, SplitScope.TRAIN, "combined_identity_preserving_quality_imbalance"
    )
    assert imbalance.payload["aggregation_policy"] == (
        "corpus_task_dialect_source_value_identity"
    )
    assert imbalance.payload["majority_share"] == pytest.approx(1 / 3)
    assert imbalance.payload["max_to_min_nonzero_ratio"] == 2.0
    assert len(report.semantic_payload.extensions) == 2
    assert all(extension.extension_fingerprint for extension in report.semantic_payload.extensions)


def test_corrected_joint_and_compatibility_support_keep_distinct_units(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    expected = {
        SplitScope.TRAIN: {
            "records": 1295,
            "components": 1209,
            "joint": (98715, 1170, 1091),
            "compatibility": (187548, 98438, 1170, 1091),
        },
        SplitScope.VALIDATION: {
            "records": 162,
            "components": 147,
            "joint": (10507, 149, 134),
            "compatibility": (20465, 10477, 149, 134),
        },
    }
    for split, values in expected.items():
        records = _extension_row(report, split, "corrected_population_records")
        components = _extension_row(
            report, split, "corrected_population_components"
        )
        joint = _extension_row(report, split, "corrected_joint_cooccurrence")
        compatibility = _extension_row(
            report, split, "compatibility_joint_support"
        )
        assert _count_values(records) == {
            "paper_candidate_records": values["records"]
        }
        assert _count_values(components) == {
            "source_components": values["components"]
        }
        assert components.coverage.observation_unit == ObservationUnit.CANONICAL_WORK
        assert _count_values(joint) == {
            "joint_events": values["joint"][0],
            "participating_components": values["joint"][2],
            "participating_records": values["joint"][1],
        }
        assert joint.coverage.observation_unit == ObservationUnit.EVENT
        assert _count_values(compatibility) == {
            "canonical_harmonic_rows": values["compatibility"][1],
            "compatible_notes": values["compatibility"][0],
            "participating_components": values["compatibility"][3],
            "participating_records": values["compatibility"][2],
        }
        assert compatibility.coverage.observation_unit == ObservationUnit.NOTE


def test_quality_support_preserves_rows_events_pieces_and_concentration(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    expected = {
        SplitScope.TRAIN: {
            "source_rows": (245, 52, 193),
            "records": (40, 6, 34),
            "components": (38, 6, 34),
            "effective": 16.837307152875,
        },
        SplitScope.VALIDATION: {
            "source_rows": (77, 4, 73),
            "records": (6, 1, 5),
            "components": (6, 1, 5),
            "effective": 3.029637199796,
        },
    }
    for split, values in expected.items():
        source_rows = _extension_row(
            report, split, "quality_augmented_seventh_chord_source_rows"
        )
        events = _extension_row(
            report, split, "quality_augmented_seventh_chord_events"
        )
        records = _extension_row(
            report, split, "quality_augmented_seventh_chord_records"
        )
        components = _extension_row(
            report, split, "quality_augmented_seventh_chord_components"
        )
        assert _count_values(source_rows) == {
            "an_source_rows": values["source_rows"][1],
            "canonical_source_rows": values["source_rows"][0],
            "dlc_source_rows": values["source_rows"][2],
        }
        assert source_rows.coverage.observation_unit == ObservationUnit.TARGET_ROW
        assert _count_values(events) == {
            "harmonic_events": values["source_rows"][0]
        }
        assert events.coverage.observation_unit == ObservationUnit.EVENT
        assert _count_values(records) == {
            "an_records": values["records"][1],
            "dlc_records": values["records"][2],
            "supporting_records": values["records"][0],
        }
        assert _count_values(components) == {
            "an_components": values["components"][1],
            "dlc_components": values["components"][2],
            "supporting_components": values["components"][0],
        }
        assert components.coverage.observation_unit == ObservationUnit.CANONICAL_WORK
        assert components.payload["dialect_support"] == "shared_an_dlc"
        assert components.payload["effective_support"] == {
            "measurement_unit": "source_component",
            "value": values["effective"],
        }

    rare_train = _extension_row(
        report, SplitScope.TRAIN, "quality_augmented_major_tetrachord_components"
    )
    rare_validation = _extension_row(
        report,
        SplitScope.VALIDATION,
        "quality_augmented_major_tetrachord_components",
    )
    assert _count_values(rare_train)["supporting_components"] == 24
    assert _count_values(rare_validation)["supporting_components"] == 4


def test_head_roles_concentration_and_train_validation_shift_are_advisory(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    roles = _extension_row(report, SplitScope.TRAIN, "advisory_head_roles")
    assert len(roles.payload["primary_heads"]) == 8
    assert len(roles.payload["auxiliary_heads"]) == 10
    assert roles.payload["deferred_heads"] == ("phrase", "section")
    assert roles.payload["interpretation"] == "training_policy_not_data_truth"

    roman_train = _extension_row(
        report, SplitScope.TRAIN, "head_roman_numeral_vocabulary_coverage"
    )
    roman_validation = _extension_row(
        report,
        SplitScope.VALIDATION,
        "head_roman_numeral_vocabulary_coverage",
    )
    assert roman_train.coverage.denominator == roman_validation.coverage.denominator == 184
    assert _count_values(roman_train) == {"observed_vocabulary_classes": 178}
    assert _count_values(roman_validation) == {
        "observed_vocabulary_classes": 113
    }
    assert roman_train.payload["majority_share"] == pytest.approx(0.208140111387)

    shift = _extension_row(report, SplitScope.VALIDATION, "roman_184_split_shift")
    assert shift.payload["absent_vocabulary_fraction"] == pytest.approx(71 / 184)
    assert shift.payload["validation_only_labels"] == (
        "vii%9",
        "N+7",
        "bV+7",
        "#v7",
    )


def test_safe_shift_orbit_and_profile_exposure_separates_historical_c2_snapshot(
    registry: EDAAdapterRegistry, eda_request: DilemmadataEDARequest
) -> None:
    report = registry.build_supervision(CorpusId.DILEMMADATA, eda_request)
    coverage = _extension_row(
        report, SplitScope.TRAIN, "safe_shift_record_coverage"
    )
    mappings = _extension_row(
        report, SplitScope.TRAIN, "safe_shift_mapping_rows"
    )
    orbit = _extension_row(report, SplitScope.TRAIN, "full_orbit_pairs")
    assert _count_values(coverage) == {
        "base_records": 1295,
        "full_orbit_records": 1231,
        "limited_orbit_records": 64,
    }
    assert _count_values(mappings) == {
        "invalid_mapping_rows": 452,
        "valid_mapping_rows": 5956,
    }
    assert mappings.coverage.denominator == 6408
    assert _count_values(orbit) == {
        "eligible_pairs": 15389,
        "excluded_pairs": 151,
        "identity_pairs": 1295,
    }
    assert orbit.coverage.denominator == 15540
    assert orbit.coverage.observation_unit == ObservationUnit.AUGMENTED_PAIR
    assert orbit.payload["variants_are_independent_musical_works"] is False

    for profile in ("c0", "c1"):
        presentations = _extension_row(
            report,
            SplitScope.TRAIN,
            f"profile_{profile}_presentation_exposure",
        )
        updates = _extension_row(
            report, SplitScope.TRAIN, f"profile_{profile}_update_exposure"
        )
        assert _count_values(presentations) == {
            "configured_presentations": 20000,
            "observed_presentations": 20000,
        }
        assert _count_values(updates) == {
            "configured_updates": 10000,
            "observed_updates": 10000,
        }
    c2_presentations = _extension_row(
        report, SplitScope.TRAIN, "profile_c2_presentation_exposure"
    )
    c2_updates = _extension_row(
        report, SplitScope.TRAIN, "profile_c2_update_exposure"
    )
    assert _count_values(c2_presentations) == {
        "configured_presentations_at_b5h_snapshot": 240000,
        "observed_presentations_at_b5h_snapshot": 0,
    }
    assert _count_values(c2_updates) == {
        "configured_updates_at_b5h_snapshot": 120000,
        "observed_updates_at_b5h_snapshot": 0,
    }
    expected_snapshot_state = {
        "current_run_state_included": False,
        "current_run_state_source": "docs/EXPERIMENT_LEDGER.md",
        "execution_state_at_snapshot": "configured_untrained",
        "run_state_scope": "historical_b5h_planning_snapshot",
        "snapshot_evidence_fingerprint": (
            "28a77c929c9e5b006ce6b37d226428814cf503bcc06e15626aa52d4756c25df6"
        ),
        "snapshot_phase": "9E-B5H",
    }
    for row in (c2_presentations, c2_updates):
        assert "execution_state" not in row.payload
        assert {
            key: row.payload[key] for key in expected_snapshot_state
        } == expected_snapshot_state

    manifest_roles = {row.role for row in report.envelope.input_manifests}
    assert "historical_full_orbit_planning_snapshot" in manifest_roles
    assert "full_orbit_profile" not in manifest_roles
    warnings = {row.code: row.message for row in report.envelope.warnings}
    assert "dilemmadata.b5h_historical_planning_snapshot" in warnings
    assert "docs/EXPERIMENT_LEDGER.md" in warnings[
        "dilemmadata.b5h_historical_planning_snapshot"
    ]
    validation = _extension_row(
        report, SplitScope.VALIDATION, "identity_only_validation_policy"
    )
    assert _count_values(validation) == {"identity_records": 162}


def test_reports_round_trip_with_canonical_fingerprints(
    registry: EDAAdapterRegistry,
    eda_request: DilemmadataEDARequest,
    tmp_path: Path,
) -> None:
    for name, report in (
        ("raw", registry.build_raw(CorpusId.DILEMMADATA, eda_request)),
        ("supervision", registry.build_supervision(CorpusId.DILEMMADATA, eda_request)),
    ):
        path = tmp_path / f"dilemmadata-{name}.json"
        path.write_bytes(canonical_report_bytes(report))
        restored = load_report(path)
        assert restored == report
        assert report_fingerprint(restored) == report.semantic_fingerprint
        assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("constant_name", "report_kind"),
    (
        ("_AUDIT_MANIFEST_SHA256", "raw"),
        ("_PRODUCTION_MANIFEST_SHA256", "raw"),
        ("_POPULATION_MANIFEST_SHA256", "raw"),
        ("_SPLIT_MANIFEST_SHA256", "supervision"),
        ("_SUPERVISION_FIXTURE_SHA256", "supervision"),
        ("_TARGET_MANIFEST_SHA256", "supervision"),
        ("_COMMON_MANIFEST_SHA256", "supervision"),
        ("_B3_MANIFEST_SHA256", "supervision"),
        ("_B4_MANIFEST_SHA256", "supervision"),
        ("_B5A_MANIFEST_SHA256", "supervision"),
        ("_B5B_MANIFEST_SHA256", "supervision"),
        ("_B5E_MANIFEST_SHA256", "supervision"),
        ("_B5H_MANIFEST_SHA256", "supervision"),
    ),
)
def test_bound_manifest_file_drift_fails_closed(
    constant_name: str,
    report_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    eda_request: DilemmadataEDARequest,
) -> None:
    monkeypatch.setattr(module, constant_name, "0" * 64)
    with pytest.raises(EDAContractError) as raised:
        adapter = DilemmadataEDAAdapter()
        if report_kind == "raw":
            adapter.build_raw_eda(eda_request)
        else:
            adapter.build_supervision_eda(eda_request)
    assert raised.value.category == "dilemmadata.eda.manifest_fingerprint_mismatch"


def test_split_manifest_semantic_drift_fails_before_callbacks(
    monkeypatch: pytest.MonkeyPatch, eda_request: DilemmadataEDARequest
) -> None:
    callbacks: list[str] = []
    monkeypatch.setattr(module, "_SPLIT_MANIFEST_FINGERPRINT", "0" * 64)
    request = DilemmadataEDARequest(
        repository_root=eda_request.repository_root,
        repository_commit=eda_request.repository_commit,
        descriptor_observer=lambda _record, _split: callbacks.append("descriptor"),
        target_loader_observer=lambda _record, _split: callbacks.append("loader"),
    )
    with pytest.raises(EDAContractError) as raised:
        DilemmadataEDAAdapter().build_supervision_eda(request)
    assert raised.value.category == (
        "dilemmadata.eda.split_manifest_fingerprint_mismatch"
    )
    assert callbacks == []


@pytest.mark.parametrize(
    "drift",
    (
        "unexpected_top_level",
        "missing_top_level",
        "fingerprint_policy",
        "semantic_fingerprint_type",
        "locked_assignment_bool",
        "locked_assignment_float",
        "assignments_container",
        "assignment_extra_field",
        "assignment_missing_field",
        "assignment_fingerprint_type",
        "assignment_corpus_type",
        "assignment_record_id_type",
        "assignment_split_type",
        "assignment_target_free_type",
        "locked_assignment_extra_field",
        "locked_assignment_split_type",
    ),
)
def test_split_manifest_shape_drift_fails_before_callbacks_and_target_reads(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
    eda_request: DilemmadataEDARequest,
) -> None:
    manifest = json.loads(
        (ROOT / "tests/fixtures/dilemmadata/eda_split_assignments.json").read_text()
    )
    if drift == "unexpected_top_level":
        manifest["unexpected_top_level"] = "tampered"
    elif drift == "missing_top_level":
        del manifest["fingerprint_policy"]
    elif drift == "fingerprint_policy":
        manifest["fingerprint_policy"] = "tampered"
    elif drift == "semantic_fingerprint_type":
        manifest["semantic_fingerprint"] = 1
    elif drift == "locked_assignment_bool":
        manifest["locked_assignment_count"] = True
    elif drift == "locked_assignment_float":
        manifest["locked_assignment_count"] = 162.0
    elif drift == "assignments_container":
        manifest["assignments"] = tuple(manifest["assignments"])
    elif drift == "assignment_extra_field":
        manifest["assignments"][0]["unexpected"] = "tampered"
    elif drift == "assignment_missing_field":
        del manifest["assignments"][0]["target_free"]
    elif drift == "assignment_fingerprint_type":
        manifest["assignments"][0]["assignment_manifest_fingerprint"] = 1
    elif drift == "assignment_corpus_type":
        manifest["assignments"][0]["corpus"] = True
    elif drift == "assignment_record_id_type":
        manifest["assignments"][0]["record_id"] = 1.0
    elif drift == "assignment_split_type":
        manifest["assignments"][0]["split"] = True
    elif drift == "assignment_target_free_type":
        manifest["assignments"][0]["target_free"] = 1
    elif drift == "locked_assignment_extra_field":
        manifest["assignments"][-1]["target_free"] = True
    elif drift == "locked_assignment_split_type":
        manifest["assignments"][-1]["split"] = 1
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(drift)

    events: list[str] = []
    original = module._load_json

    def observed_load(path: Path):
        if path.name == "eda_split_assignments.json":
            return manifest
        events.append(f"target_open:{path.name}")
        return original(path)

    monkeypatch.setattr(module, "_load_json", observed_load)
    request = DilemmadataEDARequest(
        repository_root=eda_request.repository_root,
        repository_commit=eda_request.repository_commit,
        descriptor_observer=lambda _record, _split: events.append("descriptor"),
        target_loader_observer=lambda _record, _split: events.append("loader"),
    )
    with pytest.raises(EDAContractError) as raised:
        DilemmadataEDAAdapter().build_supervision_eda(request)
    assert raised.value.category == "dilemmadata.eda.split_manifest_invalid"
    assert events == []


def test_split_manifest_exact_file_drift_fails_before_callbacks_and_target_reads(
    tmp_path: Path,
    eda_request: DilemmadataEDARequest,
) -> None:
    source = ROOT / "tests/fixtures/dilemmadata/eda_split_assignments.json"
    target = tmp_path / "tests/fixtures/dilemmadata/eda_split_assignments.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes() + b" ")
    callbacks: list[str] = []
    request = DilemmadataEDARequest(
        repository_root=tmp_path,
        repository_commit=eda_request.repository_commit,
        descriptor_observer=lambda _record, _split: callbacks.append("descriptor"),
        target_loader_observer=lambda _record, _split: callbacks.append("loader"),
    )
    with pytest.raises(EDAContractError) as raised:
        DilemmadataEDAAdapter().build_supervision_eda(request)
    assert raised.value.category == "dilemmadata.eda.manifest_fingerprint_mismatch"
    assert callbacks == []
