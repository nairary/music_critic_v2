from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import pytest

import music_critic.adapters.hooktheory_eda as hooktheory_eda
from music_critic.adapters.hooktheory_eda import (
    EDA_CONTRACT_SHA,
    HOOKTHEORY_EDA_ADAPTER_IDENTITY,
    HOOKTHEORY_RAW_EXTENSION_NAMESPACE,
    HOOKTHEORY_SOURCE_TASKS,
    HookTheoryEDAAdapter,
    HookTheoryProductionStatusEDARequest,
    HookTheoryRawEDARequest,
    HookTheorySupervisionEDARequest,
    hooktheory_vocabulary_identity,
)
from music_critic.eda import (
    CompletenessStatus,
    ComputationStatus,
    CorpusId,
    EDAAdapterRegistry,
    EDAContractError,
    EvidenceScope,
    ExecutionMode,
    LabelValueType,
    RAW_METRIC_CATALOG,
    SourceValueKind,
    SplitScope,
    canonical_report_bytes,
    dumps_report,
    loads_report,
    report_dict,
    report_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapter() -> HookTheoryEDAAdapter:
    return HookTheoryEDAAdapter()


@pytest.fixture(scope="module")
def raw_report(adapter: HookTheoryEDAAdapter):
    return adapter.build_raw_eda(
        HookTheoryRawEDARequest(
            REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
        )
    )


@pytest.fixture(scope="module")
def supervision_report(adapter: HookTheoryEDAAdapter):
    return adapter.build_supervision_eda(
        HookTheorySupervisionEDARequest(
            REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
        )
    )


def test_adapter_identity_and_bounded_raw_inventory_are_exact(
    adapter: HookTheoryEDAAdapter, raw_report
) -> None:
    assert adapter.corpus == CorpusId.HOOKTHEORY
    assert adapter.adapter_identity == HOOKTHEORY_EDA_ADAPTER_IDENTITY
    assert adapter.adapter_identity.fingerprint == (
        "58562fe45afbbfb2890dc28e5c2be4e8396157da4c45d63ddf0e8c6c9e80f26c"
    )
    assert adapter.extension_namespaces == (HOOKTHEORY_RAW_EXTENSION_NAMESPACE,)

    envelope = raw_report.envelope
    assert envelope.corpus == CorpusId.HOOKTHEORY
    assert envelope.evidence_scope == EvidenceScope.BOUNDED
    assert envelope.execution_mode == ExecutionMode.BOUNDED_SCAN
    assert envelope.completeness_status == CompletenessStatus.PARTIAL
    assert envelope.split_scope == SplitScope.ALL
    assert len(envelope.input_manifests) == 1
    assert envelope.input_manifests[0].target_free is True

    metrics = {row.metric_id: row for row in raw_report.semantic_payload.metrics}
    assert tuple(metrics) == tuple(RAW_METRIC_CATALOG)
    assert metrics["discovered_records"].count.value == 19
    assert metrics["accepted_records"].count.value == 18
    assert metrics["quarantined_records"].count.value == 1
    assert metrics["invalid_records"].count.value == 1
    assert metrics["empty_records"].count.value == 9
    assert metrics["oversize_records"].count.value == 0
    assert metrics["duration"].coverage.denominator == 19
    assert metrics["duration"].coverage.observed_count == 18
    assert metrics["duration"].coverage.unknown_count == 1

    extension = raw_report.semantic_payload.extensions[0]
    assert extension.namespace == HOOKTHEORY_RAW_EXTENSION_NAMESPACE
    assert extension.target_free is True
    split_counts = {
        count.name: count.value for row in extension.rows for count in row.counts
    }
    assert split_counts == {"test_case_count": 1, "train_case_count": 18}


def test_shared_registry_dispatches_both_hooktheory_capabilities(
    adapter: HookTheoryEDAAdapter,
) -> None:
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    raw = registry.build_raw(
        CorpusId.HOOKTHEORY,
        HookTheoryRawEDARequest(
            REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
        ),
    )
    supervision = registry.build_supervision(
        CorpusId.HOOKTHEORY,
        HookTheorySupervisionEDARequest(
            REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
        ),
    )
    assert raw.envelope.producer_identity == HOOKTHEORY_EDA_ADAPTER_IDENTITY
    assert (
        supervision.envelope.producer_identity
        == HOOKTHEORY_EDA_ADAPTER_IDENTITY
    )


def test_bounded_raw_graph_and_unattested_metrics_stay_not_computed(
    raw_report,
) -> None:
    metrics = {row.metric_id: row for row in raw_report.semantic_payload.metrics}
    for metric_id in (
        "cross_split_raw_identity_collisions",
        "duplicate_candidates",
        "graph_edge_counts",
        "graph_node_counts",
        "graph_size_distribution",
        "pitch_range",
        "tempo",
        "version_candidates",
    ):
        row = metrics[metric_id]
        assert row.coverage.status == ComputationStatus.NOT_COMPUTED
        assert row.count is None
        assert row.numeric is None
        assert row.categories == ()

    graph = raw_report.semantic_payload.graph_evidence
    assert graph.status == ComputationStatus.NOT_COMPUTED
    assert graph.target_free is None
    assert graph.reason_code == "eda.target_free_unproven"


def test_supervision_emits_only_native_train_and_validation_task_rows(
    supervision_report,
) -> None:
    envelope = supervision_report.envelope
    assert envelope.evidence_scope == EvidenceScope.BOUNDED
    assert envelope.execution_mode == ExecutionMode.BOUNDED_SCAN
    assert envelope.split_scope == SplitScope.TRAIN_VALIDATION
    assert [(row.role, row.target_free) for row in envelope.input_manifests] == [
        ("bounded_rows", False),
        ("split_assignment", True),
    ]

    tasks = supervision_report.semantic_payload.tasks
    expected_pairs = set(
        product(
            HOOKTHEORY_SOURCE_TASKS,
            (SplitScope.TRAIN, SplitScope.VALIDATION),
        )
    )
    assert {(row.source_task_id, row.split_scope) for row in tasks} == expected_pairs
    assert len(tasks) == 24
    for task in tasks:
        assert task.corpus == CorpusId.HOOKTHEORY
        assert task.dialect == "hooktheory.theorytab"
        assert task.projection_availability == ()
        assert task.projections == ()
        if task.split_scope == SplitScope.TRAIN:
            assert task.status == ComputationStatus.OBSERVED
            assert task.availability is not None
            assert task.work_identity is None
            assert all(
                item.source_value.value_kind == SourceValueKind.SCALAR
                for item in task.class_support
            )
            assert all(
                item.unique_work_count.status
                == ComputationStatus.NOT_APPLICABLE
                for item in task.class_support
            )
        else:
            assert task.status == ComputationStatus.NOT_COMPUTED
            assert task.availability is None
            assert task.class_support == ()
            assert task.reason_code == "hooktheory.validation_rows_unavailable"


def test_open_vocabulary_values_keep_hooktheory_identity(
    supervision_report,
) -> None:
    train = {
        row.source_task_id: row
        for row in supervision_report.semantic_payload.tasks
        if row.split_scope == SplitScope.TRAIN
    }
    borrowed = train["theory.chord.borrowed"]
    modes = train["theory.local_key.mode"]
    assert {row.source_value.source_value for row in borrowed.class_support} == {
        "mode:major",
        "mode:minor",
        "none",
        "pcset:1,2,4,6,8,9,11",
        "unknown:super:2",
    }
    assert {row.source_value.source_value for row in modes.class_support} == {
        "dorian",
        "major",
        "minor",
    }
    for task in (borrowed, modes):
        assert task.vocabulary == hooktheory_vocabulary_identity(
            task.source_task_id
        )
        assert all(
            row.source_value.source_task_id == task.source_task_id
            and row.source_value.dialect == "hooktheory.theorytab"
            for row in task.class_support
        )


def test_multilabel_empty_sets_and_availability_states_remain_separate(
    supervision_report,
) -> None:
    train = {
        row.source_task_id: row
        for row in supervision_report.semantic_payload.tasks
        if row.split_scope == SplitScope.TRAIN
    }
    expected = {
        "theory.chord.adds": (15, {"4": 1}),
        "theory.chord.alterations": (14, {"#5": 1, "b5": 1}),
        "theory.chord.omits": (15, {"5": 1}),
        "theory.chord.suspensions": (14, {"4": 2}),
    }
    for task_id, (empty_count, support) in expected.items():
        task = train[task_id]
        assert task.label_value_type == LabelValueType.MULTI_LABEL
        assert task.availability.denominator == 17
        assert (
            task.availability.available,
            task.availability.masked,
            task.availability.missing,
            task.availability.unsupported,
        ) == (16, 1, 0, 0)
        assert task.empty_multilabel_available_count.value == empty_count
        assert {
            row.source_value.source_value: row.occurrence_count.value
            for row in task.class_support
        } == support
        assert "" not in support

    root = train["theory.chord.root_degree"]
    assert (
        root.availability.available,
        root.availability.masked,
        root.availability.missing,
        root.availability.unsupported,
    ) == (15, 1, 0, 1)


def test_shared_preflight_never_resolves_or_opens_test_targets(
    adapter: HookTheoryEDAAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    descriptor_calls: list[tuple[str, SplitScope]] = []
    loader_calls: list[tuple[str, SplitScope]] = []
    opened: list[Path] = []
    shared_guard = hooktheory_eda.load_supervision_train_validation_only
    original_read_bytes = Path.read_bytes

    def guarded(*args, **kwargs):
        calls.append("shared_guard")
        return shared_guard(*args, **kwargs)

    def observed_read_bytes(path: Path) -> bytes:
        opened.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(
        hooktheory_eda, "load_supervision_train_validation_only", guarded
    )
    monkeypatch.setattr(Path, "read_bytes", observed_read_bytes)
    report = adapter.build_supervision_eda(
        HookTheorySupervisionEDARequest(
            REPO_ROOT,
            repository_commit=EDA_CONTRACT_SHA,
            descriptor_observer=lambda record_id, split: descriptor_calls.append(
                (record_id, split)
            ),
            target_loader_observer=lambda record_id, split: loader_calls.append(
                (record_id, split)
            ),
        )
    )

    assert calls == ["shared_guard"]
    assert descriptor_calls == loader_calls
    assert len(descriptor_calls) == 17
    assert all(split == SplitScope.TRAIN for _, split in descriptor_calls)
    test_fixture = (
        REPO_ROOT / "tests/fixtures/hooktheory/cases/root_zero_rest.json"
    ).resolve()
    assert test_fixture not in opened

    lock = report.semantic_payload.test_lock
    assert lock.assignment_gate_before_descriptor_resolution is True
    assert lock.assignment_gate_before_target_open is True
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


def test_production_status_is_unknown_without_invented_zeros(
    adapter: HookTheoryEDAAdapter,
) -> None:
    request = HookTheoryProductionStatusEDARequest(repository_commit="f" * 40)
    raw = adapter.build_raw_eda(request)
    supervision = adapter.build_supervision_eda(request)

    for report in (raw, supervision):
        assert report.envelope.evidence_scope == EvidenceScope.UNKNOWN
        assert report.envelope.execution_mode == ExecutionMode.NOT_EXECUTED
        assert report.envelope.completeness_status == CompletenessStatus.UNKNOWN
        assert report.envelope.input_manifests == ()

    for metric in raw.semantic_payload.metrics:
        assert metric.coverage.status == ComputationStatus.UNKNOWN
        assert metric.coverage.denominator is None
        assert metric.coverage.observed_count is None
        assert metric.coverage.unknown_count is None
        assert metric.count is None
        assert metric.numeric is None
        assert metric.categories == ()
    assert raw.semantic_payload.graph_evidence.status == ComputationStatus.UNKNOWN
    assert raw.semantic_payload.graph_evidence.target_free is None

    assert len(supervision.semantic_payload.tasks) == 24
    for task in supervision.semantic_payload.tasks:
        assert task.status == ComputationStatus.UNKNOWN
        assert task.availability is None
        assert task.class_support == ()
        assert task.empty_multilabel_available_count is None
        assert task.projection_availability == ()
        assert task.projections == ()
    lock = supervision.semantic_payload.test_lock
    assert lock.assignment_manifest_fingerprint is None
    for count in (
        lock.test_assignment_count,
        lock.test_descriptor_resolution_count,
        lock.test_target_loader_call_count,
        lock.test_target_records_opened,
        lock.test_target_rows_loaded,
    ):
        assert count.value is None
        assert count.denominator is None
        assert count.status == ComputationStatus.LOCKED


@pytest.mark.parametrize(
    ("request_type", "args"),
    (
        (HookTheoryRawEDARequest, (REPO_ROOT,)),
        (HookTheorySupervisionEDARequest, (REPO_ROOT,)),
        (HookTheoryProductionStatusEDARequest, ()),
    ),
)
def test_repository_commit_cannot_be_omitted(request_type, args) -> None:
    with pytest.raises(TypeError, match="repository_commit"):
        request_type(*args)


@pytest.mark.parametrize(
    ("relative_path", "request_kind"),
    (
        ("tests/fixtures/hooktheory/eda_raw_bounded_manifest.json", "raw"),
        ("tests/fixtures/hooktheory/eda_split_assignments.json", "split"),
        (
            "tests/fixtures/hooktheory/eda_supervision_manifest.json",
            "supervision",
        ),
    ),
)
def test_bound_manifest_drift_fails_closed(
    adapter: HookTheoryEDAAdapter,
    tmp_path: Path,
    relative_path: str,
    request_kind: str,
) -> None:
    source = REPO_ROOT / relative_path
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(
        EDAContractError, match="hooktheory.eda.manifest_fingerprint_mismatch"
    ):
        if request_kind == "raw":
            adapter.build_raw_eda(
                HookTheoryRawEDARequest(
                    REPO_ROOT,
                    repository_commit=EDA_CONTRACT_SHA,
                    manifest_path=changed,
                )
            )
        elif request_kind == "split":
            adapter.build_supervision_eda(
                HookTheorySupervisionEDARequest(
                    REPO_ROOT,
                    repository_commit=EDA_CONTRACT_SHA,
                    split_manifest_path=changed,
                )
            )
        else:
            adapter.build_supervision_eda(
                HookTheorySupervisionEDARequest(
                    REPO_ROOT,
                    repository_commit=EDA_CONTRACT_SHA,
                    supervision_manifest_path=changed,
                )
            )


@pytest.mark.parametrize(
    "field_path",
    (
        ("split_counts", "test"),
        ("inventory", "conversion_outcomes", "quarantined"),
        ("inventory", "reason_codes", "hooktheory.missing_json_payload"),
    ),
)
def test_nested_boolean_cannot_satisfy_integer_manifest_pin(
    adapter: HookTheoryEDAAdapter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    source = REPO_ROOT / "tests/fixtures/hooktheory/eda_raw_bounded_manifest.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    cursor = value
    for field in field_path[:-1]:
        cursor = cursor[field]
    cursor[field_path[-1]] = True
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    changed = tmp_path / "raw-manifest-with-nested-bool.json"
    changed.write_bytes(encoded)
    monkeypatch.setattr(
        hooktheory_eda, "_RAW_MANIFEST_SHA256", sha256(encoded).hexdigest()
    )

    with pytest.raises(EDAContractError, match="hooktheory.eda.manifest_mismatch"):
        adapter.build_raw_eda(
            HookTheoryRawEDARequest(
                REPO_ROOT,
                repository_commit=EDA_CONTRACT_SHA,
                manifest_path=changed,
            )
        )


def test_reports_round_trip_and_fingerprints_are_deterministic(
    adapter: HookTheoryEDAAdapter, raw_report, supervision_report
) -> None:
    rebuilt = (
        adapter.build_raw_eda(
            HookTheoryRawEDARequest(
                REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
            )
        ),
        adapter.build_supervision_eda(
            HookTheorySupervisionEDARequest(
                REPO_ROOT, repository_commit=EDA_CONTRACT_SHA
            )
        ),
    )
    for report, second in zip(
        (raw_report, supervision_report), rebuilt, strict=True
    ):
        payload = dumps_report(report)
        restored = loads_report(payload)
        assert report_dict(restored) == report_dict(report)
        assert dumps_report(restored) == payload
        assert report.semantic_fingerprint == second.semantic_fingerprint
        assert report.semantic_fingerprint == report_fingerprint(report)
        assert canonical_report_bytes(report).endswith(b"\n")

    changed_commit = adapter.build_raw_eda(
        HookTheoryRawEDARequest(REPO_ROOT, repository_commit="f" * 40)
    )
    assert changed_commit.semantic_fingerprint != raw_report.semantic_fingerprint


def test_adapter_rejects_foreign_request_types(
    adapter: HookTheoryEDAAdapter,
) -> None:
    with pytest.raises(EDAContractError, match="hooktheory.eda.request_invalid"):
        adapter.build_raw_eda(object())
    with pytest.raises(EDAContractError, match="hooktheory.eda.request_invalid"):
        adapter.build_supervision_eda(object())
