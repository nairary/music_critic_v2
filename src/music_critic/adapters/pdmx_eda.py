"""PDMX raw EDA over tracked, target-free Phase 2A.1 evidence.

This adapter deliberately does not implement Phase 10 ingestion.  It replays
the compact facts already committed by the generic-MIDI acceptance work and
keeps every unavailable corpus-wide statistic explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path

from music_critic.eda import (
    CategoryCount,
    CompletenessStatus,
    ComputationStatus,
    CorpusId,
    EDAContractError,
    EvidenceScope,
    ExecutionMode,
    ExtensionRow,
    GraphEvidence,
    InputManifestRef,
    InvariantEvidence,
    InvariantStatus,
    MetricCoverage,
    ObservationUnit,
    RAW_CORPUS_EDA_SCHEMA_NAME,
    RAW_CORPUS_EDA_SCHEMA_VERSION,
    RAW_METRIC_CATALOG,
    RawCorpusEDA,
    RawCorpusEDAPayload,
    RawMetricEvidence,
    ReportEnvelope,
    ReportKind,
    SourceExtension,
    SplitScope,
    StructuredWarning,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
)


EDA_CONTRACT_SHA = "65eb32fb948efde0fa117d7d27d19d8f16fa25b4"
PDMX_EDA_ADAPTER_VERSION = "1.0.0"
PDMX_RAW_EDA_MANIFEST_SCHEMA = "PDMXRawEDAManifest@1.0.0"
PDMX_RAW_EXTENSION_NAMESPACE = "pdmx.raw_manifest"
PDMX_RAW_EXTENSION_SCHEMA = "PDMXRawManifestExtension"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROVENANCE = ("pdmx.phase2a1.midi_diagnostic_replay",)
_DISCOVERED_SOURCE_FILES = 254_035
_SAMPLE_RECORDS = 100
_CONVERTED_SAMPLE_RECORDS = 99

_EVIDENCE_INPUT_HASHES = {
    "docs/STATUS.md": "4837aedb4b574767447564fcbf8464b3a315de09290f9767ee25f973a9d2432c",
    "scripts/smoke_midi_adapter.py": "61d5cfb778f3ba8d26ee8edddfcae7bdf02011e436455538e90c45ca401ddd6b",
    "tests/integration/test_real_midi_adapter.py": "a569c8060d4bb9bd697db3a4a16fbf7c73e60ac9d9ab02a89ef090f5d49d884f",
}


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _adapter_identity() -> VersionedIdentity:
    return VersionedIdentity(
        identity="music_critic.adapters.pdmx_eda",
        version=PDMX_EDA_ADAPTER_VERSION,
        fingerprint=_file_sha256(Path(__file__)),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise EDAContractError(
                "pdmx.eda.manifest_duplicate_key", f"duplicate JSON key {key!r}"
            )
        decoded[key] = value
    return decoded


def _load_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EDAContractError(
            "pdmx.eda.manifest_invalid",
            f"cannot load PDMX EDA manifest {path.name!r}: {exc}",
        ) from exc
    if not isinstance(decoded, dict):
        raise EDAContractError(
            "pdmx.eda.manifest_invalid", "PDMX EDA manifest root must be an object"
        )
    return decoded, sha256(raw).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EDAContractError(
            "pdmx.eda.manifest_invalid", f"{name} must be an object"
        )
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise EDAContractError(
            "pdmx.eda.manifest_invalid", f"{name} must be an array"
        )
    return value


def _require_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise EDAContractError(
            "pdmx.eda.manifest_fields_invalid",
            f"{name} fields must be exactly {sorted(expected)!r}",
        )


def _json_values_match_exactly(value: object, expected: object) -> bool:
    """Compare decoded JSON without Python's bool/int/float coercions."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        return expected.keys() == value.keys() and all(
            _json_values_match_exactly(value[key], expected_item)
            for key, expected_item in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(value, list)
        return len(value) == len(expected) and all(
            _json_values_match_exactly(observed_item, expected_item)
            for observed_item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _expect(value: object, expected: object, name: str) -> None:
    if not _json_values_match_exactly(value, expected):
        raise EDAContractError(
            "pdmx.eda.manifest_mismatch",
            f"{name} must equal the tracked value {expected!r}",
        )


def _repository_relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class PDMXRawEDARequest:
    """Request a replay of the tracked Phase 2A.1 PDMX raw evidence."""

    manifest_path: str | Path
    repository_commit: str


def _coverage(
    *,
    denominator: int | None,
    observed_count: int | None,
    unknown_count: int | None,
    unit: ObservationUnit = ObservationUnit.RECORD,
    status: ComputationStatus = ComputationStatus.OBSERVED,
    reason_code: str | None = None,
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=unit,
        denominator=denominator,
        observed_count=observed_count,
        unknown_count=unknown_count,
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_PROVENANCE,
        status=status,
        reason_code=reason_code,
    )


def _count(
    name: str,
    value: int,
    *,
    denominator: int,
    observation_unit: ObservationUnit,
    denominator_unit: ObservationUnit,
) -> UnitCount:
    return UnitCount(
        name=name,
        observation_unit=observation_unit,
        value=value,
        denominator=denominator,
        denominator_unit=denominator_unit,
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_PROVENANCE,
    )


def _not_computed_coverage(
    *, denominator: int | None, reason_code: str
) -> MetricCoverage:
    return _coverage(
        denominator=denominator,
        observed_count=None,
        unknown_count=None,
        status=ComputationStatus.NOT_COMPUTED,
        reason_code=reason_code,
    )


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    _require_keys(
        manifest,
        {
            "artifact_state",
            "bounded_midi_diagnostic",
            "corpus",
            "discovery",
            "evidence_basis",
            "evidence_inputs",
            "graph_state",
            "identity_state",
            "resource_estimators",
            "scan_policy",
            "schema",
            "source_state",
        },
        "manifest",
    )
    _expect(manifest.get("schema"), PDMX_RAW_EDA_MANIFEST_SCHEMA, "schema")
    _expect(manifest.get("corpus"), "pdmx", "corpus")
    _expect(
        manifest.get("evidence_basis"),
        [
            "phase_2a1_repository_status",
            "phase_2a1_midi_smoke_contract",
            "phase_10_not_started",
        ],
        "evidence_basis",
    )

    evidence_inputs = _sequence(manifest.get("evidence_inputs"), "evidence_inputs")
    if len(evidence_inputs) != len(_EVIDENCE_INPUT_HASHES):
        raise EDAContractError(
            "pdmx.eda.manifest_mismatch", "evidence input inventory changed"
        )
    observed_hashes: dict[str, str] = {}
    for index, value in enumerate(evidence_inputs):
        row = _mapping(value, f"evidence_inputs[{index}]")
        _require_keys(row, {"repository_path", "sha256"}, "evidence input")
        path = row.get("repository_path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise EDAContractError(
                "pdmx.eda.manifest_invalid", "evidence input fields must be strings"
            )
        observed_hashes[path] = digest
    _expect(observed_hashes, _EVIDENCE_INPUT_HASHES, "evidence input hashes")
    for repository_path, expected_hash in _EVIDENCE_INPUT_HASHES.items():
        evidence_path = _REPO_ROOT / repository_path
        try:
            observed_hash = _file_sha256(evidence_path)
        except OSError as exc:
            raise EDAContractError(
                "pdmx.eda.evidence_input_unavailable",
                f"cannot read tracked evidence input {repository_path!r}: {exc}",
            ) from exc
        if observed_hash != expected_hash:
            raise EDAContractError(
                "pdmx.eda.evidence_input_drift",
                f"tracked evidence input {repository_path!r} changed",
            )

    source_state = _mapping(manifest.get("source_state"), "source_state")
    _expect(
        source_state,
        {
            "dataset_role": "public_domain_raw_symbolic_ssl_candidate",
            "paper_url": "https://arxiv.org/abs/2409.10831",
            "release_pin_state": "not_tracked",
            "license_artifact_state": "not_tracked",
            "provenance_state": "local_tree_location_only",
        },
        "source_state",
    )
    _expect(
        manifest.get("artifact_state"),
        {
            "phase10_adapter": "absent",
            "phase10_manifest": "absent",
            "phase10_cache": "absent",
            "phase10_graph_projection": "absent",
        },
        "artifact_state",
    )
    _expect(
        manifest.get("identity_state"),
        {
            "release_identity": "unproven",
            "artifact_version_identity": "unproven",
            "work_identity": "unproven",
            "duplicate_analysis": "not_computed",
            "leakage_analysis": "not_computed",
        },
        "identity_state",
    )
    _expect(
        manifest.get("discovery"),
        {
            "files_seen": _DISCOVERED_SOURCE_FILES,
            "tree_shape": "complete_branched_midi_tree",
        },
        "discovery",
    )

    diagnostic = _mapping(
        manifest.get("bounded_midi_diagnostic"), "bounded_midi_diagnostic"
    )
    expected_diagnostic_keys = {
        "attempted",
        "converted",
        "failed",
        "failure_reasons",
        "notes",
        "outlier",
        "sample_mode",
        "selected_max_depth",
        "selected_min_depth",
        "selected_parent_dirs",
        "tracks",
        "type_0",
        "type_1",
        "warnings",
    }
    _require_keys(diagnostic, expected_diagnostic_keys, "bounded_midi_diagnostic")
    expected_scalars = {
        "sample_mode": "spread",
        "attempted": 100,
        "converted": 99,
        "failed": 1,
        "warnings": 378,
        "notes": 47_459,
        "tracks": 246,
        "type_0": 0,
        "type_1": 99,
        "selected_parent_dirs": 100,
        "selected_min_depth": 3,
        "selected_max_depth": 3,
    }
    for name, expected in expected_scalars.items():
        _expect(diagnostic.get(name), expected, f"bounded_midi_diagnostic.{name}")
    _expect(
        diagnostic.get("failure_reasons"),
        {
            "midi_adapter.meter_change_inside_bar": 1,
            "midi_adapter.unreadable_or_corrupt": 0,
            "midi_adapter.type_2": 0,
            "midi_adapter.smpte_or_non_ppqn": 0,
            "midi_adapter.invalid_meter": 0,
            "midi_adapter.metric_grid_safety": 0,
            "canonical.validation": 0,
            "canonical.serialization_round_trip": 0,
            "adapter.unexpected_exception": 0,
        },
        "bounded_midi_diagnostic.failure_reasons",
    )
    _expect(
        diagnostic.get("outlier"),
        {
            "relative_source_path": (
                "2/31/QmcmH3b8xr1N9KSEu5zS4HG7f6Beq1fENiy3bdZ9D3FXrE.mid"
            ),
            "event_tick": 8970,
            "active_meter": {"numerator": 75, "denominator": 4},
        },
        "bounded_midi_diagnostic.outlier",
    )
    _expect(
        manifest.get("resource_estimators"),
        {
            "conversion_fraction": {"numerator": 99, "denominator": 100},
            "notes_per_converted_record": {
                "numerator": 47_459,
                "denominator": 99,
            },
            "tracks_per_converted_record": {"numerator": 246, "denominator": 99},
            "warning_occurrences_per_converted_record": {
                "numerator": 378,
                "denominator": 99,
            },
        },
        "resource_estimators",
    )
    _expect(
        manifest.get("graph_state"),
        {"status": "not_computed", "reason": "eda.target_free_unproven"},
        "graph_state",
    )
    _expect(
        manifest.get("scan_policy"),
        {
            "production_scan_run": False,
            "bounded_scan_run_for_this_task": False,
            "corpus_files_opened_for_this_task": False,
            "midi_conversion_run_for_this_task": False,
            "graph_build_run": False,
            "duplicate_scan_run": False,
            "domain_gap_run": False,
        },
        "scan_policy",
    )


def _raw_metrics(manifest: Mapping[str, object]) -> tuple[RawMetricEvidence, ...]:
    diagnostic = _mapping(
        manifest["bounded_midi_diagnostic"], "bounded_midi_diagnostic"
    )
    failure_reasons = _mapping(diagnostic["failure_reasons"], "failure_reasons")
    discovery_coverage = _coverage(
        denominator=_DISCOVERED_SOURCE_FILES,
        observed_count=_DISCOVERED_SOURCE_FILES,
        unknown_count=0,
    )
    sample_coverage = _coverage(
        denominator=_SAMPLE_RECORDS,
        observed_count=_SAMPLE_RECORDS,
        unknown_count=0,
    )
    graph_coverage = _not_computed_coverage(
        denominator=_DISCOVERED_SOURCE_FILES,
        reason_code="eda.target_free_unproven",
    )
    production_unavailable = _not_computed_coverage(
        denominator=_DISCOVERED_SOURCE_FILES,
        reason_code="pdmx.raw.production_metric_not_computed",
    )
    metrics: list[RawMetricEvidence] = []
    for metric_id in RAW_METRIC_CATALOG:
        if metric_id == "discovered_records":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=discovery_coverage,
                    count=_count(
                        metric_id,
                        _DISCOVERED_SOURCE_FILES,
                        denominator=_DISCOVERED_SOURCE_FILES,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                    ),
                )
            )
        elif metric_id == "conversion_outcomes":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=sample_coverage,
                    categories=(
                        CategoryCount(
                            category="converted",
                            count=_count(
                                metric_id,
                                int(diagnostic["converted"]),
                                denominator=_SAMPLE_RECORDS,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                            ),
                        ),
                        CategoryCount(
                            category="quarantined",
                            count=_count(
                                metric_id,
                                int(diagnostic["failed"]),
                                denominator=_SAMPLE_RECORDS,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                            ),
                        ),
                    ),
                )
            )
        elif metric_id == "reason_codes":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=sample_coverage,
                    categories=tuple(
                        CategoryCount(
                            category=category,
                            count=_count(
                                metric_id,
                                int(value),
                                denominator=_SAMPLE_RECORDS,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                            ),
                        )
                        for category, value in failure_reasons.items()
                    ),
                )
            )
        elif metric_id in {
            "graph_edge_counts",
            "graph_node_counts",
            "graph_size_distribution",
        }:
            metrics.append(
                RawMetricEvidence(metric_id=metric_id, coverage=graph_coverage)
            )
        else:
            metrics.append(
                RawMetricEvidence(metric_id=metric_id, coverage=production_unavailable)
            )
    return tuple(metrics)


def _raw_extension(manifest: Mapping[str, object]) -> SourceExtension:
    diagnostic = _mapping(
        manifest["bounded_midi_diagnostic"], "bounded_midi_diagnostic"
    )
    estimators = _mapping(manifest["resource_estimators"], "resource_estimators")
    outlier = _mapping(diagnostic["outlier"], "outlier")
    rows = (
        ExtensionRow(
            row_id="bounded_conversion_funnel",
            payload={
                "selection_policy": "deterministic_spread",
                "historical_scope": "phase_2a1_diagnostic",
            },
            counts=(
                _count(
                    "attempted_sample_records",
                    int(diagnostic["attempted"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "converted_sample_records",
                    int(diagnostic["converted"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "failed_sample_records",
                    int(diagnostic["failed"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "midi_type_0_records",
                    int(diagnostic["type_0"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "midi_type_1_records",
                    int(diagnostic["type_1"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "note_occurrences",
                    int(diagnostic["notes"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.NOTE,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "track_occurrences",
                    int(diagnostic["tracks"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.TRACK,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "warning_occurrences",
                    int(diagnostic["warnings"]),
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=_SAMPLE_RECORDS,
                observed_count=_SAMPLE_RECORDS,
                unknown_count=0,
            ),
        ),
        ExtensionRow(
            row_id="conversion_outlier",
            payload={
                "relative_source_path": outlier["relative_source_path"],
                "event_tick": outlier["event_tick"],
                "active_meter": outlier["active_meter"],
                "reason": "midi_adapter.meter_change_inside_bar",
            },
            counts=(
                _count(
                    "meter_change_rejection_records",
                    1,
                    denominator=_SAMPLE_RECORDS,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=_SAMPLE_RECORDS,
                observed_count=1,
                unknown_count=_SAMPLE_RECORDS - 1,
            ),
        ),
        ExtensionRow(
            row_id="discovery_inventory",
            payload={"tree_shape": "complete_branched_midi_tree"},
            counts=(
                _count(
                    "source_files_seen",
                    _DISCOVERED_SOURCE_FILES,
                    denominator=_DISCOVERED_SOURCE_FILES,
                    observation_unit=ObservationUnit.SOURCE_FILE,
                    denominator_unit=ObservationUnit.SOURCE_FILE,
                ),
            ),
            coverage=_coverage(
                denominator=_DISCOVERED_SOURCE_FILES,
                observed_count=_DISCOVERED_SOURCE_FILES,
                unknown_count=0,
                unit=ObservationUnit.SOURCE_FILE,
            ),
        ),
        ExtensionRow(
            row_id="resource_scale_estimators",
            payload={
                "conversion_fraction": estimators["conversion_fraction"],
                "notes_per_converted_record": estimators[
                    "notes_per_converted_record"
                ],
                "tracks_per_converted_record": estimators[
                    "tracks_per_converted_record"
                ],
                "warning_occurrences_per_converted_record": estimators[
                    "warning_occurrences_per_converted_record"
                ],
                "interpretation": "spread_sample_ratios_not_population_totals",
            },
            coverage=_coverage(
                denominator=_SAMPLE_RECORDS,
                observed_count=_CONVERTED_SAMPLE_RECORDS,
                unknown_count=_SAMPLE_RECORDS - _CONVERTED_SAMPLE_RECORDS,
            ),
        ),
        ExtensionRow(
            row_id="source_release_pin",
            payload={},
            coverage=_not_computed_coverage(
                denominator=None, reason_code="pdmx.raw.release_pin_unproven"
            ),
        ),
        ExtensionRow(
            row_id="source_license_artifact",
            payload={},
            coverage=_not_computed_coverage(
                denominator=None, reason_code="pdmx.raw.license_artifact_unproven"
            ),
        ),
        ExtensionRow(
            row_id="source_identity_and_leakage",
            payload={},
            coverage=_not_computed_coverage(
                denominator=_DISCOVERED_SOURCE_FILES,
                reason_code="pdmx.raw.identity_scan_not_run",
            ),
        ),
    )
    return SourceExtension(
        corpus=CorpusId.PDMX,
        namespace=PDMX_RAW_EXTENSION_NAMESPACE,
        schema_name=PDMX_RAW_EXTENSION_SCHEMA,
        schema_version="1.0.0",
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_PROVENANCE,
        rows=rows,
        target_free=True,
    )


def _build_report(
    request: PDMXRawEDARequest, adapter_identity: VersionedIdentity
) -> RawCorpusEDA:
    manifest_path = Path(request.manifest_path)
    manifest, manifest_fingerprint = _load_json(manifest_path)
    _validate_manifest(manifest)
    source_identity = VersionedIdentity(
        identity="pdmx.phase2a1_local_midi_snapshot_evidence",
        version="1.0.0",
        fingerprint=manifest_fingerprint,
    )
    envelope = ReportEnvelope(
        schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
        schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.RAW_CORPUS,
        corpus=CorpusId.PDMX,
        source_identity=source_identity,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        execution_mode=ExecutionMode.MANIFEST_REPLAY,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.ALL,
        observation_units=(
            ObservationUnit.EVENT,
            ObservationUnit.NOTE,
            ObservationUnit.RECORD,
            ObservationUnit.SOURCE_FILE,
            ObservationUnit.TRACK,
        ),
        input_manifests=(
            InputManifestRef(
                role="raw_projection",
                identity=VersionedIdentity(
                    identity="pdmx.eda.raw_manifest",
                    version="1.0.0",
                    fingerprint=manifest_fingerprint,
                ),
                target_free=True,
                repository_relative_path=_repository_relative_path(manifest_path),
            ),
        ),
        invariants=(
            InvariantEvidence(
                code="pdmx.raw.historical_diagnostic_immutable",
                status=InvariantStatus.PASSED,
                provenance=_PROVENANCE,
            ),
            InvariantEvidence(
                code="pdmx.raw.phase10_deferred",
                status=InvariantStatus.PASSED,
                provenance=_PROVENANCE,
            ),
            InvariantEvidence(
                code="pdmx.raw.production_scan_absent",
                status=InvariantStatus.PASSED,
                provenance=_PROVENANCE,
            ),
            InvariantEvidence(
                code="pdmx.raw.work_identity",
                status=InvariantStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                reason_code="eda.work_identity_unproven",
            ),
        ),
        warnings=(
            StructuredWarning(
                code="pdmx.raw.bounded_sample_only",
                message=(
                    "The historical 100-file spread diagnostic is bounded evidence "
                    "and does not describe production corpus distributions."
                ),
                provenance=_PROVENANCE,
            ),
        ),
        unavailable_reasons=(
            UnavailableReason(
                code="pdmx.raw.phase10_artifacts_absent",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail=(
                    "The Phase 10 adapter, manifest, cache, and graph artifacts "
                    "have not been implemented."
                ),
            ),
            UnavailableReason(
                code="pdmx.raw.release_pin_unproven",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail="No versioned upstream release pin is tracked in this repository.",
            ),
            UnavailableReason(
                code="pdmx.raw.license_artifact_unproven",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail="No repository license artifact is bound to the local PDMX tree.",
            ),
            UnavailableReason(
                code="pdmx.raw.identity_scan_not_run",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail=(
                    "Artifact, version, work, duplicate, and split-collision identities "
                    "have not been audited for the production tree."
                ),
            ),
            UnavailableReason(
                code="pdmx.raw.production_metric_not_computed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail=(
                    "Duration, onset, track, density, polyphony, pitch, tempo, meter, "
                    "instrument, percussion, and graph-size distributions were not scanned."
                ),
            ),
            UnavailableReason(
                code="eda.target_free_unproven",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_PROVENANCE,
                detail="Approved raw graph-contract aggregation was not executed.",
            ),
            UnavailableReason(
                code="eda.work_identity_unproven",
                status=ComputationStatus.NOT_APPLICABLE,
                provenance=_PROVENANCE,
                detail="The historical MIDI diagnostic did not define a versioned work identity.",
            ),
        ),
    )
    return RawCorpusEDA(
        envelope=envelope,
        semantic_payload=RawCorpusEDAPayload(
            metrics=_raw_metrics(manifest),
            graph_evidence=GraphEvidence(
                status=ComputationStatus.NOT_COMPUTED,
                target_free=None,
                reason_code="eda.target_free_unproven",
            ),
            extensions=(_raw_extension(manifest),),
        ),
    )


@dataclass(frozen=True, slots=True)
class PDMXEDAAdapter:
    """Raw-only PDMX EDA adapter; Phase 10 ingestion remains out of scope."""

    corpus: CorpusId = CorpusId.PDMX
    adapter_identity: VersionedIdentity = field(default_factory=_adapter_identity)
    extension_namespaces: tuple[str, ...] = (PDMX_RAW_EXTENSION_NAMESPACE,)

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        if type(request) is not PDMXRawEDARequest:
            raise EDAContractError(
                "pdmx.eda.request_invalid", "raw EDA requires PDMXRawEDARequest"
            )
        return _build_report(request, self.adapter_identity)


__all__ = [
    "EDA_CONTRACT_SHA",
    "PDMX_EDA_ADAPTER_VERSION",
    "PDMX_RAW_EDA_MANIFEST_SCHEMA",
    "PDMX_RAW_EXTENSION_NAMESPACE",
    "PDMXEDAAdapter",
    "PDMXRawEDARequest",
]
