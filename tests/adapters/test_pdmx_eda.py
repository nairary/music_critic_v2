from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

import music_critic.adapters.pdmx_eda as pdmx_eda_module
from music_critic.adapters.pdmx_eda import (
    EDA_CONTRACT_SHA,
    PDMX_EDA_ADAPTER_VERSION,
    PDMX_RAW_EXTENSION_NAMESPACE,
    PDMXEDAAdapter,
    PDMXRawEDARequest,
)
from music_critic.eda import (
    ComputationStatus,
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
MANIFEST = REPO_ROOT / "tests/fixtures/pdmx/eda_raw_manifest.json"


def _report(*, repository_commit: str = EDA_CONTRACT_SHA):
    adapter = PDMXEDAAdapter()
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    return registry.build_raw(
        CorpusId.PDMX,
        PDMXRawEDARequest(MANIFEST, repository_commit=repository_commit),
    )


def _metric(report, metric_id: str):
    return next(
        row for row in report.semantic_payload.metrics if row.metric_id == metric_id
    )


def _extension_rows(report):
    extension = report.semantic_payload.extensions[0]
    return extension, {row.row_id: row for row in extension.rows}


def test_import_and_registry_use_raw_only_pdmx_capability() -> None:
    adapter = PDMXEDAAdapter()
    assert adapter.corpus is CorpusId.PDMX
    assert adapter.adapter_identity.identity == "music_critic.adapters.pdmx_eda"
    assert adapter.adapter_identity.version == PDMX_EDA_ADAPTER_VERSION
    assert adapter.extension_namespaces == (PDMX_RAW_EXTENSION_NAMESPACE,)
    assert not hasattr(adapter, "build_supervision_eda")

    registry = EDAAdapterRegistry()
    registry.register(adapter)
    assert (
        registry.build_raw(
            CorpusId.PDMX,
            PDMXRawEDARequest(MANIFEST, repository_commit=EDA_CONTRACT_SHA),
        ).envelope.corpus
        is CorpusId.PDMX
    )
    with pytest.raises(EDAContractError, match="supervision_forbidden"):
        registry.build_supervision(CorpusId.PDMX, object())


def test_report_scope_and_manifest_binding_are_literal() -> None:
    report = _report()
    envelope = report.envelope
    assert envelope.repository_commit == EDA_CONTRACT_SHA
    assert envelope.evidence_scope is EvidenceScope.MANIFEST_REPLAY
    assert envelope.execution_mode is ExecutionMode.MANIFEST_REPLAY
    assert envelope.split_scope is SplitScope.ALL
    assert envelope.completeness_status.value == "partial"
    assert envelope.input_manifests[0].repository_relative_path == (
        "tests/fixtures/pdmx/eda_raw_manifest.json"
    )
    assert envelope.input_manifests[0].target_free is True
    manifest_sha = sha256(MANIFEST.read_bytes()).hexdigest()
    assert envelope.input_manifests[0].identity.fingerprint == manifest_sha
    assert envelope.source_identity.fingerprint == manifest_sha
    assert set(envelope.observation_units) == {
        ObservationUnit.EVENT,
        ObservationUnit.NOTE,
        ObservationUnit.RECORD,
        ObservationUnit.SOURCE_FILE,
        ObservationUnit.TRACK,
    }


def test_full_tree_discovery_and_bounded_conversion_are_not_conflated() -> None:
    report = _report()
    discovered = _metric(report, "discovered_records")
    assert discovered.coverage.denominator == 254_035
    assert discovered.coverage.observed_count == 254_035
    assert discovered.count is not None
    assert discovered.count.value == 254_035

    accepted = _metric(report, "accepted_records")
    quarantined = _metric(report, "quarantined_records")
    assert accepted.coverage.status is ComputationStatus.NOT_COMPUTED
    assert quarantined.coverage.status is ComputationStatus.NOT_COMPUTED
    assert accepted.count is None
    assert quarantined.count is None

    outcomes = _metric(report, "conversion_outcomes")
    assert outcomes.coverage.denominator == 100
    assert outcomes.coverage.observed_count == 100
    assert {row.category: row.count.value for row in outcomes.categories} == {
        "converted": 99,
        "quarantined": 1,
    }


def test_failure_reason_preserves_the_only_historical_rejection() -> None:
    reasons = _metric(_report(), "reason_codes")
    values = {row.category: row.count.value for row in reasons.categories}
    assert values["midi_adapter.meter_change_inside_bar"] == 1
    assert values["midi_adapter.unreadable_or_corrupt"] == 0
    assert values["midi_adapter.type_2"] == 0
    assert values["midi_adapter.smpte_or_non_ppqn"] == 0
    assert sum(values.values()) == 1


def test_extension_preserves_funnel_units_and_outlier() -> None:
    extension, rows = _extension_rows(_report())
    assert extension.namespace == PDMX_RAW_EXTENSION_NAMESPACE
    assert extension.target_free is True
    assert extension.work_identity is None

    funnel = rows["bounded_conversion_funnel"]
    counts = {row.name: row for row in funnel.counts}
    assert counts["attempted_sample_records"].value == 100
    assert counts["converted_sample_records"].value == 99
    assert counts["failed_sample_records"].value == 1
    assert counts["note_occurrences"].value == 47_459
    assert counts["note_occurrences"].observation_unit is ObservationUnit.NOTE
    assert counts["track_occurrences"].value == 246
    assert counts["track_occurrences"].observation_unit is ObservationUnit.TRACK
    assert counts["warning_occurrences"].value == 378
    assert counts["warning_occurrences"].observation_unit is ObservationUnit.EVENT
    assert all(row.denominator == 100 for row in counts.values())
    assert all(row.denominator_unit is ObservationUnit.RECORD for row in counts.values())

    outlier = rows["conversion_outlier"]
    assert outlier.coverage.observed_count == 1
    assert outlier.coverage.unknown_count == 99
    assert outlier.payload["event_tick"] == 8970
    assert outlier.payload["active_meter"] == {"numerator": 75, "denominator": 4}
    assert str(outlier.payload["relative_source_path"]).startswith("2/31/")


def test_scale_estimators_remain_sample_ratios() -> None:
    _, rows = _extension_rows(_report())
    estimator = rows["resource_scale_estimators"]
    assert estimator.coverage.denominator == 100
    assert estimator.coverage.observed_count == 99
    assert estimator.coverage.unknown_count == 1
    assert estimator.payload["conversion_fraction"] == {
        "numerator": 99,
        "denominator": 100,
    }
    assert estimator.payload["notes_per_converted_record"] == {
        "numerator": 47_459,
        "denominator": 99,
    }
    assert estimator.payload["interpretation"] == (
        "spread_sample_ratios_not_population_totals"
    )


def test_release_license_identity_and_leakage_gaps_are_explicit() -> None:
    report = _report()
    _, rows = _extension_rows(report)
    for row_id in (
        "source_release_pin",
        "source_license_artifact",
        "source_identity_and_leakage",
    ):
        row = rows[row_id]
        assert row.coverage.status is ComputationStatus.NOT_COMPUTED
        assert row.payload == {}
        assert row.counts == ()
    reason_codes = {row.code for row in report.envelope.unavailable_reasons}
    assert {
        "pdmx.raw.phase10_artifacts_absent",
        "pdmx.raw.release_pin_unproven",
        "pdmx.raw.license_artifact_unproven",
        "pdmx.raw.identity_scan_not_run",
        "pdmx.raw.production_metric_not_computed",
        "eda.target_free_unproven",
        "eda.work_identity_unproven",
    } <= reason_codes


def test_musical_and_graph_distributions_are_not_fabricated() -> None:
    report = _report()
    for metric_id in (
        "duration",
        "notes",
        "onsets",
        "tracks",
        "density",
        "polyphony",
        "pitch_range",
        "tempo",
        "meter",
        "instruments",
        "percussion_presence",
        "graph_node_counts",
        "graph_edge_counts",
        "graph_size_distribution",
        "duplicate_candidates",
        "cross_split_raw_identity_collisions",
    ):
        metric = _metric(report, metric_id)
        assert metric.coverage.status is ComputationStatus.NOT_COMPUTED
        assert metric.count is None
        assert metric.numeric is None
        assert metric.categories == ()
    assert report.semantic_payload.graph_evidence.status is ComputationStatus.NOT_COMPUTED
    assert report.semantic_payload.graph_evidence.target_free is None


def test_report_round_trip_and_repository_commit_are_fingerprinted() -> None:
    report = _report()
    assert loads_report(dumps_report(report)) == report
    assert report_fingerprint(report) == report.semantic_fingerprint
    other = _report(repository_commit="f" * 40)
    assert other.semantic_fingerprint != report.semantic_fingerprint
    assert other.semantic_payload == report.semantic_payload


def test_adapter_rejects_wrong_request_type() -> None:
    with pytest.raises(EDAContractError, match="request_invalid"):
        PDMXEDAAdapter().build_raw_eda(object())


def test_request_requires_explicit_repository_commit() -> None:
    with pytest.raises(TypeError, match="repository_commit"):
        PDMXRawEDARequest(MANIFEST)  # type: ignore[call-arg]


def test_manifest_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["discovery"]["files_seen"] = 254_034
    path = tmp_path / "pdmx-drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EDAContractError, match="manifest_mismatch"):
        PDMXEDAAdapter().build_raw_eda(
            PDMXRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA)
        )


def test_manifest_unknown_field_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "pdmx-extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EDAContractError, match="manifest_fields_invalid"):
        PDMXEDAAdapter().build_raw_eda(
            PDMXRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA)
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("bounded_midi_diagnostic", "attempted", 100.0),
        ("scan_policy", "production_scan_run", 0),
    ),
)
def test_manifest_rejects_json_type_coercion(
    tmp_path: Path, section: str, field: str, replacement: object
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload[section][field] = replacement
    path = tmp_path / f"pdmx-type-drift-{section}-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EDAContractError, match="manifest_mismatch"):
        PDMXEDAAdapter().build_raw_eda(
            PDMXRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA)
        )


def test_manifest_duplicate_key_fails_before_report_construction(tmp_path: Path) -> None:
    text = MANIFEST.read_text(encoding="utf-8").replace(
        '"corpus": "pdmx",',
        '"corpus": "pdmx",\n  "corpus": "pdmx",',
        1,
    )
    path = tmp_path / "pdmx-duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EDAContractError, match="manifest_duplicate_key"):
        PDMXEDAAdapter().build_raw_eda(
            PDMXRawEDARequest(path, repository_commit=EDA_CONTRACT_SHA)
        )


def test_tracked_evidence_input_hashes_match_the_foundation() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in payload["evidence_inputs"]:
        path = REPO_ROOT / row["repository_path"]
        assert sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_tracked_evidence_input_drift_fails_closed(monkeypatch) -> None:
    adapter = PDMXEDAAdapter()
    file_sha256 = pdmx_eda_module._file_sha256

    def drift_status_hash(path: Path) -> str:
        if path == REPO_ROOT / "docs/STATUS.md":
            return "0" * 64
        return file_sha256(path)

    monkeypatch.setattr(pdmx_eda_module, "_file_sha256", drift_status_hash)
    with pytest.raises(EDAContractError, match="evidence_input_drift"):
        adapter.build_raw_eda(
            PDMXRawEDARequest(MANIFEST, repository_commit=EDA_CONTRACT_SHA)
        )


def test_scan_policy_proves_this_task_replayed_without_corpus_access() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["scan_policy"] == {
        "production_scan_run": False,
        "bounded_scan_run_for_this_task": False,
        "corpus_files_opened_for_this_task": False,
        "midi_conversion_run_for_this_task": False,
        "graph_build_run": False,
        "duplicate_scan_run": False,
        "domain_gap_run": False,
    }
