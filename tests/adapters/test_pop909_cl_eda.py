from __future__ import annotations

import json
from pathlib import Path

import pytest

import music_critic.adapters.pop909_cl_eda as pop909_cl_eda_module
from music_critic.adapters.pop909_cl_eda import (
    EDA_CONTRACT_SHA,
    POP909_CL_EDA_ADAPTER_VERSION,
    POP909_CL_EDA_SUPERVISION_FIXTURE_SHA256,
    POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256,
    POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256,
    POP909_CL_RAW_EXTENSION_NAMESPACE,
    Pop909ClEDAAdapter,
    Pop909ClPhase4EvidenceRequest,
    Pop909ClRawEDARequest,
    Pop909ClSupervisionEDARequest,
    Pop909ClUnavailableSupervisionEDARequest,
    dumps_pop909_cl_phase4_evidence,
    pop909_cl_phase4_evidence_dict,
    replay_pop909_cl_phase4_evidence,
    validate_pop909_cl_identity_splits,
)
from music_critic.eda import (
    ComputationStatus,
    CompletenessStatus,
    CorpusId,
    EDAAdapterRegistry,
    EDAContractError,
    EvidenceScope,
    ExecutionMode,
    ObservationUnit,
    SplitScope,
    dumps_report,
    loads_report,
    report_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_MANIFEST = REPO_ROOT / "tests/fixtures/pop909_cl/eda_raw_manifest.json"
SPLIT_MANIFEST = (
    REPO_ROOT / "tests/fixtures/pop909_cl/eda_split_assignments.json"
)
SUPERVISION_FIXTURE = (
    REPO_ROOT / "tests/fixtures/pop909_cl/eda_supervision_fixture.json"
)
AUDIT_MANIFEST = REPO_ROOT / "tests/fixtures/pop909_cl/audit_manifest.json"
PRODUCTION_MANIFEST = (
    REPO_ROOT / "tests/fixtures/pop909_cl/production_manifest.json"
)


def _registry() -> tuple[EDAAdapterRegistry, Pop909ClEDAAdapter]:
    adapter = Pop909ClEDAAdapter()
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    return registry, adapter


def _raw_report():
    registry, _ = _registry()
    return registry.build_raw(
        CorpusId.POP909_CL,
        Pop909ClRawEDARequest(RAW_MANIFEST, repository_commit=EDA_CONTRACT_SHA),
    )


def _supervision_report(**kwargs):
    registry, _ = _registry()
    return registry.build_supervision(
        CorpusId.POP909_CL,
        Pop909ClSupervisionEDARequest(
            SPLIT_MANIFEST,
            SUPERVISION_FIXTURE,
            repository_commit=EDA_CONTRACT_SHA,
            **kwargs,
        ),
    )


def _unavailable_supervision_report():
    registry, _ = _registry()
    return registry.build_supervision(
        CorpusId.POP909_CL,
        Pop909ClUnavailableSupervisionEDARequest(
            repository_commit=EDA_CONTRACT_SHA
        ),
    )


def _phase4_replay():
    return replay_pop909_cl_phase4_evidence(
        Pop909ClPhase4EvidenceRequest(
            RAW_MANIFEST,
            AUDIT_MANIFEST,
            PRODUCTION_MANIFEST,
            repository_commit=EDA_CONTRACT_SHA,
        )
    )


def test_adapter_registration_and_identity_are_source_owned() -> None:
    registry, adapter = _registry()
    assert registry.registrations() == (
        (
            CorpusId.POP909_CL,
            adapter.adapter_identity,
            (POP909_CL_RAW_EXTENSION_NAMESPACE,),
        ),
    )
    assert adapter.adapter_identity.identity == "music_critic.adapters.pop909_cl_eda"
    assert adapter.adapter_identity.version == POP909_CL_EDA_ADAPTER_VERSION
    assert len(adapter.adapter_identity.fingerprint) == 64


@pytest.mark.parametrize(
    "request_type,args",
    (
        (Pop909ClRawEDARequest, (RAW_MANIFEST,)),
        (
            Pop909ClSupervisionEDARequest,
            (SPLIT_MANIFEST, SUPERVISION_FIXTURE),
        ),
        (Pop909ClUnavailableSupervisionEDARequest, ()),
        (
            Pop909ClPhase4EvidenceRequest,
            (RAW_MANIFEST, AUDIT_MANIFEST, PRODUCTION_MANIFEST),
        ),
    ),
)
def test_every_request_requires_explicit_repository_commit(
    request_type, args: tuple[object, ...]
) -> None:
    with pytest.raises(TypeError, match="repository_commit"):
        request_type(*args)


def test_raw_manifest_replay_preserves_909_908_and_172() -> None:
    report = _raw_report()
    metrics = {
        row.metric_id: row for row in report.semantic_payload.metrics
    }
    assert report.envelope.evidence_scope == EvidenceScope.MANIFEST_REPLAY
    assert report.envelope.split_scope == SplitScope.ALL
    assert metrics["discovered_records"].count.value == 909
    assert metrics["accepted_records"].count.value == 908
    assert metrics["quarantined_records"].count.value == 1
    assert {
        row.category: row.count.value
        for row in metrics["conversion_outcomes"].categories
    } == {"accepted": 908, "quarantined": 1}
    assert {
        row.category: row.count.value
        for row in metrics["reason_codes"].categories
    } == {"midi_adapter.meter_change_inside_bar": 1}
    assert report.envelope.source_identity.identity == "pop909_cl.release"
    assert report.envelope.source_identity.version == "2.0.0"


def test_raw_identity_evidence_keeps_543_and_553_split_atomic() -> None:
    report = _raw_report()
    metrics = {row.metric_id: row for row in report.semantic_payload.metrics}
    assert metrics["duplicate_candidates"].count.value == 2
    assert metrics["duplicate_candidates"].coverage.denominator == 908
    collision = metrics["cross_split_raw_identity_collisions"]
    assert collision.count.value == 0
    assert collision.count.observation_unit == ObservationUnit.RAW_IDENTITY_COLLISION
    extension = report.semantic_payload.extensions[0]
    rows = {row.row_id: row for row in extension.rows}
    duplicate = rows["raw_equivalence_cluster"]
    assert duplicate.payload["member_record_ids"] == (
        "piece:pop909-cl-543",
        "piece:pop909-cl-553",
    )
    assert duplicate.payload["member_splits"] == ("train", "train")
    assert duplicate.counts[0].value == 2
    identity = rows["identity_split_safety"]
    assert {count.name: count.value for count in identity.counts} == {
        "lineage_split_collisions": 0,
        "source_group_split_collisions": 0,
    }


def test_raw_graph_and_unreplayed_metrics_remain_non_observed() -> None:
    report = _raw_report()
    metrics = {row.metric_id: row for row in report.semantic_payload.metrics}
    assert report.semantic_payload.graph_evidence.status == ComputationStatus.NOT_COMPUTED
    assert report.semantic_payload.graph_evidence.target_free is None
    for metric_id in (
        "graph_edge_counts",
        "graph_node_counts",
        "graph_size_distribution",
    ):
        assert metrics[metric_id].coverage.status == ComputationStatus.NOT_COMPUTED
        assert metrics[metric_id].count is None
        assert metrics[metric_id].numeric is None
        assert metrics[metric_id].categories == ()
    assert metrics["notes"].coverage.status == ComputationStatus.NOT_COMPUTED
    assert metrics["notes"].numeric is None


def test_raw_extensions_use_typed_counts_and_literal_outlier_scope() -> None:
    report = _raw_report()
    extension = report.semantic_payload.extensions[0]
    rows = {row.row_id: row for row in extension.rows}
    assert extension.namespace == POP909_CL_RAW_EXTENSION_NAMESPACE
    assert extension.target_free is True
    assert len(extension.extension_fingerprint) == 64
    assert rows["installation_noise"].coverage.observation_unit == (
        ObservationUnit.SOURCE_FILE
    )
    assert rows["installation_noise"].counts[0].value == 910
    warning_counts = {
        count.name: count.value
        for count in rows["score_warning_occurrences"].counts
    }
    assert warning_counts["same_pitch_overlap_warning_occurrences"] == 123439
    assert warning_counts["warning_occurrences"] == 126163
    meter = rows["meter_change_offsets"]
    assert meter.coverage.observed_count == 1
    assert meter.coverage.unknown_count == 908
    assert meter.payload["song_id"] == "172"


def test_raw_replay_includes_only_sound_phase4_structural_counts() -> None:
    report = _raw_report()
    rows = {
        row.row_id: row for row in report.semantic_payload.extensions[0].rows
    }
    container = rows["midi_container"]
    assert container.payload == {"midi_file_type": 1, "ppqn": 480}
    assert {count.name: count.value for count in container.counts} == {
        "empty_conductor_tracks": 909,
        "midi_type_1_records": 909,
        "ppqn_480_records": 909,
    }
    assert {
        count.observation_unit for count in container.counts
    } == {ObservationUnit.RECORD, ObservationUnit.TRACK}
    metadata = rows["global_metadata_events"]
    assert metadata.payload == {"source_location": "conductor_track_0"}
    assert {count.name: count.value for count in metadata.counts} == {
        "key_signature_event_occurrences": 1065,
        "meter_event_occurrences": 911,
        "tempo_event_occurrences": 909,
    }
    assert {count.observation_unit for count in metadata.counts} == {
        ObservationUnit.EVENT,
        ObservationUnit.METER_EVENT,
        ObservationUnit.TEMPO_EVENT,
    }
    split_population = rows["split_assignment_population"]
    assert split_population.coverage.denominator == 908
    assert {count.name: count.value for count in split_population.counts} == {
        "test_assignment_rows": 106,
        "train_assignment_rows": 701,
        "validation_assignment_rows": 101,
    }
    notes = {
        row.metric_id: row for row in report.semantic_payload.metrics
    }["notes"]
    assert notes.coverage.status == ComputationStatus.NOT_COMPUTED
    assert notes.numeric is None
    note_extension = rows["score_note_distribution"]
    assert note_extension.payload == {
        "maximum": 4233,
        "mean_status": "not_computed",
        "median": 1655,
        "minimum": 175,
        "p95": 2403,
    }
    assert note_extension.coverage.observed_count == 908
    assert note_extension.coverage.unknown_count == 1
    warning_extension = rows["score_warning_occurrences"]
    warning_counts = {
        count.name: count.value for count in warning_extension.counts
    }
    assert warning_counts["warning_occurrences"] == 126163
    assert warning_counts["empty_track_warning_occurrences"] == 908
    assert warning_counts["same_pitch_overlap_warning_occurrences"] == 123439
    assert warning_counts["same_pitch_overlap_warning_affected_records"] == 907
    assert warning_extension.payload["minimum_per_converted_record"] == 3
    assert warning_extension.payload["median_per_converted_record"] == 123
    assert warning_extension.payload["p95_per_converted_record"] == 282
    assert warning_extension.payload["maximum_per_converted_record"] == 966


def test_phase4_replay_binds_exact_accepted_manifest_bytes() -> None:
    replay = _phase4_replay()
    assert replay.audit_manifest_identity.fingerprint == (
        POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256
    )
    assert replay.production_manifest_identity.fingerprint == (
        POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256
    )
    assert replay.logical_record_count == 909
    assert replay.accepted_record_count == 908
    assert replay.quarantined_record_ids == ("172",)
    assert replay.accepted_missing_target_record_ids == ("367", "658")
    assert replay.source_records_with_chord_instrument == 907
    assert replay.accepted_records_with_chord_evidence == 906
    assert len(replay.semantic_fingerprint) == 64
    assert replay.semantic_fingerprint == _phase4_replay().semantic_fingerprint
    assert pop909_cl_phase4_evidence_dict(replay)["schema"] == (
        "Pop909ClPhase4EvidenceReplay@1.0.0"
    )
    changed_commit = replay_pop909_cl_phase4_evidence(
        Pop909ClPhase4EvidenceRequest(
            RAW_MANIFEST,
            AUDIT_MANIFEST,
            PRODUCTION_MANIFEST,
            repository_commit="f" * 40,
        )
    )
    assert changed_commit.semantic_fingerprint != replay.semantic_fingerprint
    serialized = dumps_pop909_cl_phase4_evidence(replay, indent=2)
    decoded = json.loads(serialized)
    assert decoded["semantic_fingerprint"] == replay.semantic_fingerprint
    assert decoded["repository_commit"] == EDA_CONTRACT_SHA
    assert dumps_pop909_cl_phase4_evidence(replay) == (
        dumps_pop909_cl_phase4_evidence(_phase4_replay())
    )


def test_phase4_replay_preserves_native_availability_states() -> None:
    replay = _phase4_replay()
    rows = {row.task_id: row for row in replay.target_rows}
    expected = {
        "pop909_cl.chord.bass": (116055, 116055, 0, 0, 0),
        "pop909_cl.chord.boundary": (116055, 116055, 0, 0, 0),
        "pop909_cl.chord.inversion": (116055, 109668, 5801, 0, 586),
        "pop909_cl.chord.no_chord": (1098, 947, 151, 0, 0),
        "pop909_cl.chord.quality": (116055, 109800, 5669, 0, 586),
        "pop909_cl.chord.root": (116055, 109668, 5801, 0, 586),
    }
    assert {
        task_id: (
            row.denominator,
            row.available,
            row.masked,
            row.missing,
            row.unsupported,
        )
        for task_id, row in rows.items()
    } == expected
    assert rows["pop909_cl.chord.bass"].accepted_available_record_support == 906
    assert (
        rows["pop909_cl.chord.boundary"].accepted_available_record_support
        == 906
    )
    assert all(row.accepted_missing_record_count == 2 for row in rows.values())
    assert all(
        row.accepted_available_record_support is None
        and row.record_support_status == ComputationStatus.NOT_COMPUTED
        for task_id, row in rows.items()
        if task_id
        not in {"pop909_cl.chord.bass", "pop909_cl.chord.boundary"}
    )


def test_phase4_replay_keeps_trailing_and_partial_raw_distributions_literal() -> None:
    replay = _phase4_replay()
    assert replay.chord_block_count == 116055
    assert (
        replay.chord_block_observed_record_count,
        replay.chord_block_minimum_per_record,
        replay.chord_block_median_per_record,
        replay.chord_block_p95_per_record,
        replay.chord_block_maximum_per_record,
    ) == (909, 0, 124, 185, 278)
    assert replay.leading_internal_no_chord_span_count == 947
    assert replay.trailing_uncovered_span_count == 151
    assert (
        replay.trailing_duration_minimum_ticks,
        replay.trailing_duration_median_ticks,
        replay.trailing_duration_p95_ticks,
        replay.trailing_duration_maximum_ticks,
    ) == (1, 401, 3361, 12861)
    assert (
        replay.score_note_observed_record_count,
        replay.score_note_unknown_record_count,
        replay.score_note_minimum,
        replay.score_note_median,
        replay.score_note_p95,
        replay.score_note_maximum,
    ) == (908, 1, 175, 1655, 2403, 4233)
    assert (
        replay.score_warning_observed_record_count,
        replay.score_warning_unknown_record_count,
        replay.score_warning_occurrence_count,
        replay.score_warning_minimum_per_record,
        replay.score_warning_median_per_record,
        replay.score_warning_p95_per_record,
        replay.score_warning_maximum_per_record,
    ) == (908, 1, 126163, 3, 123, 282, 966)
    assert replay.raw_pitch_class_set_count == 261
    assert replay.selected_root_quality_bass_label_count == 340
    assert replay.duplicate_block_onset_count == 0
    assert replay.class_concentration_status == ComputationStatus.NOT_COMPUTED
    assert replay.cooccurrence_status == ComputationStatus.NOT_COMPUTED
    assert replay.train_validation_shift_status == ComputationStatus.NOT_COMPUTED
    assert replay.canonical_work_identity_status == ComputationStatus.NOT_COMPUTED


@pytest.mark.parametrize(
    "source_path,field_path,expected_error",
    (
        (
            AUDIT_MANIFEST,
            ("aggregates", "chord_annotation_inventory", "total_blocks"),
            "phase4_audit_manifest_drift",
        ),
        (
            PRODUCTION_MANIFEST,
            ("expected", "chord_blocks"),
            "phase4_production_manifest_drift",
        ),
    ),
)
def test_phase4_replay_rejects_manifest_byte_drift(
    tmp_path: Path,
    source_path: Path,
    field_path: tuple[str, ...],
    expected_error: str,
) -> None:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    cursor = payload
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] += 1
    changed = tmp_path / source_path.name
    changed.write_text(json.dumps(payload), encoding="utf-8")
    request = Pop909ClPhase4EvidenceRequest(
        RAW_MANIFEST,
        changed if source_path == AUDIT_MANIFEST else AUDIT_MANIFEST,
        changed if source_path == PRODUCTION_MANIFEST else PRODUCTION_MANIFEST,
        repository_commit=EDA_CONTRACT_SHA,
    )
    with pytest.raises(EDAContractError, match=expected_error):
        replay_pop909_cl_phase4_evidence(request)


def test_raw_and_supervision_reports_round_trip_canonically() -> None:
    for report in (
        _raw_report(),
        _supervision_report(),
        _unavailable_supervision_report(),
    ):
        payload = dumps_report(report)
        restored = loads_report(payload)
        assert restored == report
        assert dumps_report(restored) == payload
        assert report_fingerprint(report) == report.semantic_fingerprint


def test_raw_manifest_fails_closed_when_accepted_inventory_changes(
    tmp_path: Path,
) -> None:
    payload = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    payload["inventory"]["accepted_records"] = 909
    path = tmp_path / "bad_raw_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="manifest_mismatch"):
        registry.build_raw(
            CorpusId.POP909_CL,
            Pop909ClRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA),
        )


def test_raw_manifest_rejects_a_changed_evidence_basis(tmp_path: Path) -> None:
    payload = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    payload["evidence_basis"] = ["unverified_scan"]
    path = tmp_path / "bad_evidence_basis.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="manifest_mismatch"):
        registry.build_raw(
            CorpusId.POP909_CL,
            Pop909ClRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA),
        )


def test_raw_manifest_cannot_hide_an_unapproved_field(tmp_path: Path) -> None:
    payload = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    payload["target_distribution"] = {"C": 909}
    path = tmp_path / "raw_with_hidden_field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="manifest_fields_invalid"):
        registry.build_raw(
            CorpusId.POP909_CL,
            Pop909ClRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA),
        )


def test_supervision_is_fixture_only_native_and_split_specific() -> None:
    report = _supervision_report()
    tasks = report.semantic_payload.tasks
    assert report.envelope.evidence_scope == EvidenceScope.FIXTURE
    assert report.envelope.split_scope == SplitScope.TRAIN_VALIDATION
    assert len(tasks) == 12
    assert {task.split_scope for task in tasks} == {
        SplitScope.TRAIN,
        SplitScope.VALIDATION,
    }
    assert {task.source_task_id for task in tasks} == {
        "pop909_cl.chord.bass",
        "pop909_cl.chord.boundary",
        "pop909_cl.chord.inversion",
        "pop909_cl.chord.no_chord",
        "pop909_cl.chord.quality",
        "pop909_cl.chord.root",
    }
    assert all(task.projections == () for task in tasks)
    assert all(task.projection_availability == () for task in tasks)
    assert all(task.work_identity is None for task in tasks)
    assert all(
        support.unique_work_count.status == ComputationStatus.NOT_APPLICABLE
        for task in tasks
        for support in task.class_support
    )


def test_unavailable_production_supervision_cannot_masquerade_as_fixture() -> None:
    report = _unavailable_supervision_report()
    assert report.envelope.evidence_scope == EvidenceScope.UNKNOWN
    assert report.envelope.execution_mode == ExecutionMode.NOT_EXECUTED
    assert report.envelope.completeness_status == CompletenessStatus.UNKNOWN
    assert report.envelope.input_manifests == ()
    assert len(report.semantic_payload.tasks) == 12
    assert all(
        task.status == ComputationStatus.NOT_COMPUTED
        and task.availability is None
        and task.class_support == ()
        and task.work_identity is None
        for task in report.semantic_payload.tasks
    )
    lock = report.semantic_payload.test_lock
    assert lock.assignment_manifest_fingerprint is None
    assert lock.test_assignment_count.status == ComputationStatus.LOCKED
    assert lock.test_assignment_count.value is None
    assert all(
        count.value is None
        for count in (
            lock.test_descriptor_resolution_count,
            lock.test_target_loader_call_count,
            lock.test_target_records_opened,
            lock.test_target_rows_loaded,
        )
    )
    reason_codes = {
        reason.code for reason in report.envelope.unavailable_reasons
    }
    assert {
        "pop909_cl.supervision.class_concentration_not_computed",
        "pop909_cl.supervision.cooccurrence_not_computed",
        "pop909_cl.supervision.train_validation_shift_not_computed",
        "pop909_cl.supervision.split_rows_not_computed",
        "eda.work_identity_unproven",
    } <= reason_codes


def test_supervision_availability_and_classes_exclude_non_available_rows() -> None:
    report = _supervision_report()
    by_key = {
        (task.source_task_id, task.split_scope): task
        for task in report.semantic_payload.tasks
    }
    train_root = by_key[("pop909_cl.chord.root", SplitScope.TRAIN)]
    assert (
        train_root.availability.available,
        train_root.availability.masked,
        train_root.availability.missing,
        train_root.availability.unsupported,
    ) == (1, 1, 0, 0)
    assert [row.source_value.source_value for row in train_root.class_support] == ["C"]
    train_no_chord = by_key[("pop909_cl.chord.no_chord", SplitScope.TRAIN)]
    assert (
        train_no_chord.availability.available,
        train_no_chord.availability.missing,
    ) == (1, 1)
    assert [row.source_value.source_value for row in train_no_chord.class_support] == [
        "N"
    ]
    validation_quality = by_key[
        ("pop909_cl.chord.quality", SplitScope.VALIDATION)
    ]
    assert validation_quality.availability.unsupported == 1
    assert validation_quality.class_support == ()


def test_target_fixture_is_opened_only_after_guard_preflight_and_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    original_load_json = pop909_cl_eda_module._load_json
    original_guard = pop909_cl_eda_module.load_supervision_train_validation_only

    def observed_load_json(path: Path):
        events.append(("json_read", Path(path).name))
        return original_load_json(path)

    def observed_guard(*args, **kwargs):
        events.append(("guard_enter", ""))
        return original_guard(*args, **kwargs)

    monkeypatch.setattr(pop909_cl_eda_module, "_load_json", observed_load_json)
    monkeypatch.setattr(
        pop909_cl_eda_module,
        "load_supervision_train_validation_only",
        observed_guard,
    )

    report = _supervision_report(
        descriptor_observer=lambda record_id, split: events.append(
            ("descriptor", f"{split.value}:{record_id}")
        ),
        loader_observer=lambda record_id, split: events.append(
            ("loader", f"{split.value}:{record_id}")
        ),
    )

    fixture_read = ("json_read", SUPERVISION_FIXTURE.name)
    assert events[0] == ("json_read", SPLIT_MANIFEST.name)
    assert events[1] == ("guard_enter", "")
    assert events.index(fixture_read) > next(
        index for index, event in enumerate(events) if event[0] == "descriptor"
    )
    assert events.index(fixture_read) > next(
        index for index, event in enumerate(events) if event[0] == "loader"
    )
    assert events.count(fixture_read) == 1
    assert all(
        not detail.startswith("test:")
        for kind, detail in events
        if kind in {"descriptor", "loader"}
    )
    assert report.semantic_payload.test_lock.test_target_records_opened.value == 0


def test_target_fixture_remains_unopened_when_assignment_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    payload["assignments"][0]["split"] = "heldout"
    bad_split = tmp_path / "bad_split.json"
    bad_split.write_text(json.dumps(payload), encoding="utf-8")
    opened: list[str] = []
    original_load_json = pop909_cl_eda_module._load_json

    def observed_load_json(path: Path):
        opened.append(Path(path).name)
        return original_load_json(path)

    monkeypatch.setattr(pop909_cl_eda_module, "_load_json", observed_load_json)
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="assignment_split_invalid"):
        registry.build_supervision(
            CorpusId.POP909_CL,
            Pop909ClSupervisionEDARequest(
                bad_split,
                SUPERVISION_FIXTURE,
                repository_commit=EDA_CONTRACT_SHA,
            ),
        )
    assert opened == [bad_split.name]


def test_supervision_fixture_exact_bytes_are_pinned_inside_loader(
    tmp_path: Path,
) -> None:
    assert (
        pop909_cl_eda_module._file_sha256(SUPERVISION_FIXTURE)
        == POP909_CL_EDA_SUPERVISION_FIXTURE_SHA256
    )
    changed = tmp_path / SUPERVISION_FIXTURE.name
    changed.write_bytes(SUPERVISION_FIXTURE.read_bytes() + b" ")
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="supervision_fixture_drift"):
        registry.build_supervision(
            CorpusId.POP909_CL,
            Pop909ClSupervisionEDARequest(
                SPLIT_MANIFEST,
                changed,
                repository_commit=EDA_CONTRACT_SHA,
            ),
        )


def test_supervision_guard_spies_never_receive_test() -> None:
    descriptor_calls: list[tuple[str, SplitScope]] = []
    loader_calls: list[tuple[str, SplitScope]] = []

    def descriptor_spy(record_id: str, split: SplitScope) -> None:
        if split == SplitScope.TEST:
            raise AssertionError("TEST descriptor resolution attempted")
        descriptor_calls.append((record_id, split))

    def loader_spy(record_id: str, split: SplitScope) -> None:
        if split == SplitScope.TEST:
            raise AssertionError("TEST loader attempted")
        loader_calls.append((record_id, split))

    report = _supervision_report(
        descriptor_observer=descriptor_spy,
        loader_observer=loader_spy,
    )
    assert len(descriptor_calls) == len(loader_calls) == 3
    assert all(split != SplitScope.TEST for _, split in descriptor_calls + loader_calls)
    lock = report.semantic_payload.test_lock
    assert lock.test_assignment_count.value == 1
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


def test_source_identity_leakage_fails_before_callbacks(tmp_path: Path) -> None:
    payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    payload["identity_rows"][2]["source_group_id"] = payload["identity_rows"][0][
        "source_group_id"
    ]
    path = tmp_path / "leaking_split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[tuple[str, SplitScope]] = []
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="identity_leakage"):
        registry.build_supervision(
            CorpusId.POP909_CL,
            Pop909ClSupervisionEDARequest(
                path,
                SUPERVISION_FIXTURE,
                repository_commit=EDA_CONTRACT_SHA,
                descriptor_observer=lambda record_id, split: calls.append(
                    (record_id, split)
                ),
                loader_observer=lambda record_id, split: calls.append(
                    (record_id, split)
                ),
            ),
        )
    assert calls == []


def test_supervision_fixture_rejects_unapproved_crosswalk_field(
    tmp_path: Path,
) -> None:
    payload = json.loads(SUPERVISION_FIXTURE.read_text(encoding="utf-8"))
    payload["records"]["fixture-pop-train-a"]["rows"][0]["common_projection"] = (
        "unapproved"
    )
    path = tmp_path / "fixture_with_crosswalk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry, _ = _registry()
    with pytest.raises(EDAContractError, match="manifest_fields_invalid"):
        registry.build_supervision(
            CorpusId.POP909_CL,
            Pop909ClSupervisionEDARequest(
                SPLIT_MANIFEST,
                path,
                repository_commit=EDA_CONTRACT_SHA,
            ),
        )


def _identity_rows() -> list[dict[str, object]]:
    payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return payload["identity_rows"]


def test_identity_validator_accepts_shared_source_group_within_train() -> None:
    validate_pop909_cl_identity_splits(_identity_rows())


@pytest.mark.parametrize(
    "field_name,shared_value",
    (
        ("source_group_id", "pop909-cl-score:shared"),
        ("lineage_id", "pop909-lineage:shared"),
        ("canonical_work_id", "canonical-work:shared"),
    ),
)
def test_identity_validator_rejects_every_cross_split_identity_family(
    field_name: str, shared_value: str
) -> None:
    rows = _identity_rows()
    rows[0][field_name] = shared_value
    rows[2][field_name] = shared_value
    with pytest.raises(EDAContractError, match="identity_leakage"):
        validate_pop909_cl_identity_splits(rows)


def test_identity_validator_closes_transitive_components() -> None:
    rows = _identity_rows()
    rows[0]["source_group_id"] = "pop909-cl-score:bridge"
    rows[1]["source_group_id"] = "pop909-cl-score:bridge"
    rows[1]["lineage_id"] = "pop909-lineage:bridge"
    rows[2]["lineage_id"] = "pop909-lineage:bridge"
    with pytest.raises(EDAContractError, match="identity_leakage"):
        validate_pop909_cl_identity_splits(rows)


def test_identity_validator_rejects_duplicate_record_ids() -> None:
    rows = _identity_rows()
    rows[1]["record_id"] = rows[0]["record_id"]
    with pytest.raises(EDAContractError, match="record_identity_duplicate"):
        validate_pop909_cl_identity_splits(rows)


def test_supervision_fixture_has_no_held_out_record_or_rows() -> None:
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(SUPERVISION_FIXTURE.read_text(encoding="utf-8"))
    held_out_rows = [row for row in split["assignments"] if row["split"] == "test"]
    assert held_out_rows == [{"split": "test"}]
    assert all("test" not in record_id for record_id in fixture["records"])


def test_adapter_rejects_wrong_request_types() -> None:
    adapter = Pop909ClEDAAdapter()
    with pytest.raises(EDAContractError, match="request_invalid"):
        adapter.build_raw_eda(object())
    with pytest.raises(EDAContractError, match="request_invalid"):
        adapter.build_supervision_eda(object())
