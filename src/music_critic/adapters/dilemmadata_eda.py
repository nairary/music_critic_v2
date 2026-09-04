"""Dilemmadata implementation of the frozen multi-source EDA contract.

Both paths replay committed compact evidence and never discover or scan a
corpus.  Supervision evidence is opened only after the shared pre-open TEST
gate.  The source-native surface cases remain deliberately small; the
corpus-scale B3/B4/B5 evidence is reported separately in typed extensions so
that a fixture probe is never presented as a production distribution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from music_critic.data.serialization import canonical_json_sha256
from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    AvailabilityCounts,
    AvailabilityState,
    CategoryCount,
    ClassSupport,
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
    LabelValueType,
    MetricCoverage,
    ObservationUnit,
    ProjectionAvailabilityCounts,
    ProjectionEvidence,
    ProjectionMappingState,
    RAW_CORPUS_EDA_SCHEMA_NAME,
    RAW_CORPUS_EDA_SCHEMA_VERSION,
    RAW_METRIC_CATALOG,
    RawCorpusEDA,
    RawCorpusEDAPayload,
    RawMetricEvidence,
    ReportEnvelope,
    ReportKind,
    SUPERVISION_EDA_SCHEMA_NAME,
    SUPERVISION_EDA_SCHEMA_VERSION,
    SourceExtension,
    SourceValueIdentity,
    SourceValueKind,
    SplitScope,
    StructuredWarning,
    SupervisionEDA,
    SupervisionEDAPayload,
    TaskFamilyEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    load_supervision_train_validation_only,
)
from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_FAMILIES,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
    DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
)


DILEMMADATA_EDA_ADAPTER_VERSION = "1.0.1"
DILEMMADATA_EDA_CONTRACT_BASE = "65eb32fb948efde0fa117d7d27d19d8f16fa25b4"
DILEMMADATA_RAW_EXTENSION_NAMESPACE = "dilemmadata.raw_inventory"
DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE = (
    "dilemmadata.supervision_diagnostics"
)

_AUDIT_MANIFEST_PATH = "tests/fixtures/dilemmadata/audit_manifest.json"
_PRODUCTION_MANIFEST_PATH = "tests/fixtures/dilemmadata/production_manifest.json"
_POPULATION_MANIFEST_PATH = (
    "tests/fixtures/dilemmadata/eda_population_manifest.json"
)
_SPLIT_MANIFEST_PATH = "tests/fixtures/dilemmadata/eda_split_assignments.json"
_SUPERVISION_FIXTURE_PATH = (
    "tests/fixtures/dilemmadata/eda_supervision_fixture.json"
)
_TARGET_MANIFEST_PATH = "tests/fixtures/dilemmadata/target_manifest.json"
_COMMON_MANIFEST_PATH = (
    "tests/fixtures/dilemmadata/common_harmonic_manifest.json"
)
_B3_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb3_multitask_contract.json"
)
_B4_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb4_class_balance_audit.json"
)
_B5A_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb5a_transposition_audit.json"
)
_B5B_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json"
)
_B5E_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb5e_full_training_results.json"
)
_B5H_MANIFEST_PATH = (
    "tests/fixtures/analysisgnn/phase9eb5h_full_orbit_profile.json"
)
_AUDIT_MANIFEST_SHA256 = (
    "c321a75064abec81e1357690c256e16a16af2eb8d4a3e50e3cbb624b2c3d52aa"
)
_PRODUCTION_MANIFEST_SHA256 = (
    "58e0c28ea7f88dcea3d3e1e453b11d1cfd44635dcc8af4b91f04250c026bcbac"
)
_POPULATION_MANIFEST_SHA256 = (
    "62b9e87eea6e1c4f6bd3612b8457e4c20834e986b8a48b822e2f2644f9c2047b"
)
_SPLIT_MANIFEST_SHA256 = (
    "dd34263ec9dde70a134a6b987114e4d8db027cc27d744cb91179672caa958ea3"
)
_SPLIT_MANIFEST_FINGERPRINT = (
    "17a9191d6260fb100548164d39bf95773eff44e58f8693c6e7d73412676abaa9"
)
_SUPERVISION_FIXTURE_SHA256 = (
    "ff28fa99b6a577cc970e75b07d9196e1bb7f17a0b3db5b065d6211aaf25b86a7"
)
_TARGET_MANIFEST_SHA256 = (
    "f13a8017ee6d3618a9a177c387618c9e1631abdab01bbecf044f2eeb45ac0318"
)
_COMMON_MANIFEST_SHA256 = (
    "ef899a8a8f77f5387d0e952b1eaf94fa32cf2a9dea494bfcb35b4729f1cb40e2"
)
_B3_MANIFEST_SHA256 = (
    "a32c0dbf6d9a6c55da31a1296a736101d5cb2408f0a9d75d8a42007d7223f806"
)
_B4_MANIFEST_SHA256 = (
    "fb9b41a0e6c985f9753f5609374bea0b168a9b131f21acd5cb9e8d343fb359c1"
)
_B5A_MANIFEST_SHA256 = (
    "3d6625381f170d1419bb0cacf1f6bb6f8c21bb641a11c8d0adc299f1803d734d"
)
_B5B_MANIFEST_SHA256 = (
    "12b8b812e138af45e0ca2f7926bf2de3a53872368f165b3ead24b6aa9140dd34"
)
_B5E_MANIFEST_SHA256 = (
    "1d573a158666a9b258641a80a44b803b68bcffa514ccc5abbf38d84553097470"
)
_B5H_MANIFEST_SHA256 = (
    "69fddbf6aab4c1e49940343463cdeada05eefd2fee3aef0b5e487cda7cdaf74d"
)
_SOURCE_CONTENT_FINGERPRINT = (
    "8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312"
)
_SOURCE_RELEASE_COMMIT = "d60ee75b4a9495e932a4a7be39381578be17e222"
_AUDIT_SEMANTIC_FINGERPRINT = (
    "ce7e13b04c0299c48e5f33db36ab98948d11ea2df0d81cf438042633746112ed"
)
_PRODUCTION_SEMANTIC_FINGERPRINT = (
    "92187b3b10e27662536870b4fce9d683065a32bc20bf970184a2a7b33727287a"
)
_POPULATION_SEMANTIC_FINGERPRINT = (
    "45f03a1d0efc7dc68967502a69a11f7b640d177c494b97b59a8655ae21a4189a"
)
_TARGET_AUDIT_FINGERPRINT = (
    "a971ff0daf8d5a442beaa3365ec8c43ca9368f07baab4a1102927977f6ebdd05"
)
_COMMON_MANIFEST_FINGERPRINT = (
    "4ce7b657d2003d2ce3aadcfe9de9e39c7f9a49b69e985a745a399ef02e056294"
)
_B3_SEMANTIC_FINGERPRINT = (
    "94a19ed6bbecbbd0497310233c8a8ff4e34311b414124593a7326c759ff07954"
)
_B4_SEMANTIC_FINGERPRINT = (
    "4b1edf9f47815bafa5e197be87b9331a19789142c0625ef4aceda1f87649df4d"
)
_B5A_SEMANTIC_FINGERPRINT = (
    "b8aba86430fe2c87b250a5d1d1adc7557eed41ac54f24ae6cff32fd8bc815644"
)
_B5B_SEMANTIC_FINGERPRINT = (
    "6639f35d84770f639e9a2179df2122e2e3d5aa5af49964fbf4222e93a0474bd9"
)
_B5E_EVIDENCE_FINGERPRINT = (
    "ad9f4f5f25851aeaddd0193abe1cfd9db2d0693a6bc1863f96e3d2044cbab8fe"
)
_B5H_EVIDENCE_FINGERPRINT = (
    "28a77c929c9e5b006ce6b37d226428814cf503bcc06e15626aa52d4756c25df6"
)

_SPLIT_MANIFEST_SCHEMA = "DilemmadataEDASplitAssignments@1.0.0"
_SPLIT_MANIFEST_FINGERPRINT_POLICY = (
    "canonical_projection_without_bound_fingerprint_v1"
)
_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "assignments",
        "fingerprint_policy",
        "locked_assignment_count",
        "schema",
        "semantic_fingerprint",
    }
)
_RETAINED_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_manifest_fingerprint",
        "corpus",
        "record_id",
        "split",
        "target_free",
    }
)
_LOCKED_ASSIGNMENT_FIELDS = frozenset({"split"})

_ADAPTER_CONTRACT = {
    "adapter_identity": "music_critic.adapters.dilemmadata_eda",
    "adapter_version": DILEMMADATA_EDA_ADAPTER_VERSION,
    "contract_base": DILEMMADATA_EDA_CONTRACT_BASE,
    "raw_mode": "tracked_compact_manifest_replay_only",
    "supervision_mode": "guarded_train_validation_manifest_replay",
}
DILEMMADATA_EDA_ADAPTER_IDENTITY = VersionedIdentity(
    identity=_ADAPTER_CONTRACT["adapter_identity"],
    version=DILEMMADATA_EDA_ADAPTER_VERSION,
    fingerprint=canonical_json_sha256(_ADAPTER_CONTRACT),
)
DILEMMADATA_SOURCE_IDENTITY = VersionedIdentity(
    identity="johentsch.dilemmadata.release",
    version="1.0.0",
    fingerprint=_SOURCE_CONTENT_FINGERPRINT,
)

_RAW_PROVENANCE = (
    "dilemmadata-audit-manifest-1.1.0",
    "dilemmadata-eda-population-manifest-1.0.0",
    "dilemmadata-production-manifest-1.1.0",
)
_SURFACE_PROVENANCE = ("dilemmadata-eda-surface-cases-1.0.0",)
_NATIVE_CATALOG_PROVENANCE = ("dilemmadata-native-family-manifest-1.1.0",)
_AGGREGATE_PROVENANCE = (
    "phase9eb3-corrected-multitask-manifest-2.0.0",
    "phase9eb4-class-balance-manifest-1.0.0",
    "phase9eb5a-transposition-manifest-1.0.0",
    "phase9eb5b-policy-manifest-1.0.0",
    "phase9eb5e-training-result-manifest-1.0.0",
    "phase9eb5h-full-orbit-manifest-1.0.0",
)
_SUPERVISION_PROVENANCE = (
    "dilemmadata-common-harmonic-manifest-1.0.0",
    *_SURFACE_PROVENANCE,
    *_NATIVE_CATALOG_PROVENANCE,
    *_AGGREGATE_PROVENANCE,
)
_ACCESS_PROVENANCE = ("dilemmadata-eda-split-gate-1.0.0",)
_OBSERVED_TASKS = frozenset(
    {
        "dilemmadata.an.chord.inversion",
        "dilemmadata.an.chord.quality",
        "dilemmadata.dlc.chord.inversion",
        "dilemmadata.dlc.chord.quality",
    }
)
_COMMON_TASK_BY_SOURCE = {
    "dilemmadata.an.chord.inversion": "dilemmadata.common.chord.inversion",
    "dilemmadata.an.chord.quality": "dilemmadata.common.chord.quality",
    "dilemmadata.dlc.chord.inversion": "dilemmadata.common.chord.inversion",
    "dilemmadata.dlc.chord.quality": "dilemmadata.common.chord.quality",
}
_QUALITY_PROJECTION = {
    ("an_joint", "major triad"): (ProjectionMappingState.EXACT, "major triad"),
    ("an_joint", "minor triad"): (ProjectionMappingState.EXACT, "minor triad"),
    ("dlc", "+7"): (ProjectionMappingState.EXACT, "augmented seventh chord"),
    ("dlc", "M"): (ProjectionMappingState.EXACT, "major triad"),
    ("dlc", "m"): (ProjectionMappingState.EXACT, "minor triad"),
}
_INVERSION_PROJECTION = {
    ("an_joint", "2"): (ProjectionMappingState.EXACT, "second"),
    ("dlc", "2"): (ProjectionMappingState.EXACT, "third"),
    ("dlc", "42"): (ProjectionMappingState.EXACT, "third"),
    ("dlc", "43"): (ProjectionMappingState.EXACT, "second"),
}

_SOURCE_COMPONENT_IDENTITY = VersionedIdentity(
    identity="dilemmadata.phase9eb3.source_component",
    version="1.0.0",
    fingerprint="909e84531cc1c70fab14dd24dfe1c9525c5634e5edd46a6f88faf13cfc755778",
)

Observer = Callable[[str, SplitScope], None]


@dataclass(frozen=True, slots=True)
class DilemmadataEDARequest:
    """Operational request; no local path is placed in semantic evidence."""

    repository_root: Path
    repository_commit: str
    descriptor_observer: Observer | None = None
    target_loader_observer: Observer | None = None


@dataclass(frozen=True, slots=True)
class _TargetDescriptor:
    record_id: str
    split: SplitScope
    manifest_path: Path


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EDAContractError(
                "dilemmadata.eda.manifest_duplicate_key",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EDAContractError(
            "dilemmadata.eda.manifest_invalid",
            f"cannot load tracked EDA manifest {path.name!r}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise EDAContractError(
            "dilemmadata.eda.manifest_invalid", "EDA manifest root must be an object"
        )
    return value


def _sha256_file(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EDAContractError(
            "dilemmadata.eda.manifest_unavailable",
            f"cannot read tracked EDA manifest {path.name!r}",
        ) from exc


def _request_root(request: object) -> tuple[DilemmadataEDARequest, Path]:
    if type(request) is not DilemmadataEDARequest:
        raise EDAContractError(
            "dilemmadata.eda.request_invalid",
            "Dilemmadata EDA requires DilemmadataEDARequest",
        )
    root = Path(request.repository_root).resolve()
    if not root.is_dir():
        raise EDAContractError(
            "dilemmadata.eda.repository_root_invalid",
            "repository_root must identify the checked-out repository",
        )
    if any(
        observer is not None and not callable(observer)
        for observer in (
            request.descriptor_observer,
            request.target_loader_observer,
        )
    ):
        raise EDAContractError(
            "dilemmadata.eda.request_invalid",
            "descriptor and loader observers must be callable or null",
        )
    return request, root


def _tracked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise EDAContractError(
            "dilemmadata.eda.path_invalid", "tracked input escaped repository_root"
        )
    return path


def _verify_hash(path: Path, expected: str) -> None:
    if _sha256_file(path) != expected:
        raise EDAContractError(
            "dilemmadata.eda.manifest_fingerprint_mismatch",
            f"tracked manifest {path.name!r} differs from the adapter binding",
        )


def _coverage(
    *,
    denominator: int | None,
    split: SplitScope,
    scope: EvidenceScope,
    provenance: tuple[str, ...],
    observed: bool,
    unit: ObservationUnit = ObservationUnit.RECORD,
    reason: str | None = None,
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=unit,
        denominator=denominator,
        observed_count=denominator if observed else None,
        unknown_count=0 if observed else None,
        split_scope=split,
        evidence_scope=scope,
        provenance=provenance,
        status=(ComputationStatus.OBSERVED if observed else ComputationStatus.NOT_COMPUTED),
        reason_code=None if observed else reason,
    )


def _count(
    name: str,
    value: int,
    denominator: int,
    *,
    unit: ObservationUnit,
    denominator_unit: ObservationUnit,
    split: SplitScope,
    scope: EvidenceScope,
    provenance: tuple[str, ...],
) -> UnitCount:
    return UnitCount(
        name=name,
        observation_unit=unit,
        value=value,
        denominator=denominator,
        denominator_unit=denominator_unit,
        split_scope=split,
        evidence_scope=scope,
        provenance=provenance,
    )


def _manifest_ref(
    *, role: str, identity: str, version: str, fingerprint: str, target_free: bool,
    path: str,
) -> InputManifestRef:
    return InputManifestRef(
        role=role,
        identity=VersionedIdentity(
            identity=identity,
            version=version,
            fingerprint=fingerprint,
        ),
        target_free=target_free,
        repository_relative_path=path,
    )


def _build_raw_metrics(
    audit: Mapping[str, object], production: Mapping[str, object]
) -> tuple[RawMetricEvidence, ...]:
    records = _require_mapping(audit, "records")
    grouping = _require_mapping(audit, "grouping")
    raw_projection = _require_mapping(audit, "raw_projection")
    outcomes = _require_mapping(production, "outcomes")
    discovered = _require_int(outcomes, "discovered_count")
    statuses = _require_mapping(outcomes, "status_counts")
    accepted = _require_int(statuses, "accepted")
    quarantined = _require_int(statuses, "quarantined")
    if (discovered, accepted, quarantined) != (1633, 719, 914):
        raise EDAContractError(
            "dilemmadata.eda.inventory_mismatch",
            "tracked production inventory differs from the pinned 1633/719/914 result",
        )
    if accepted + quarantined != discovered:
        raise EDAContractError(
            "dilemmadata.eda.inventory_mismatch", "raw inventory is not conserved"
        )
    if _require_int(records, "an_joint_record_count") + _require_int(
        records, "dlc_record_count"
    ) != discovered:
        raise EDAContractError(
            "dilemmadata.eda.dialect_inventory_mismatch",
            "AN and DLC record inventories do not sum to the combined inventory",
        )

    common_coverage = _coverage(
        denominator=discovered,
        split=SplitScope.ALL,
        scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_RAW_PROVENANCE,
        observed=True,
    )
    count_values = {
        "accepted_records": accepted,
        "discovered_records": discovered,
        "duplicate_candidates": _require_int(
            grouping, "midi_note_event_multiset_equivalent_record_count"
        ),
        "quarantined_records": quarantined,
    }
    categories: dict[str, tuple[tuple[str, int], ...]] = {
        "conversion_outcomes": (("accepted", accepted), ("quarantined", quarantined)),
        "parse_outcomes": (("raw_projection_parsed", discovered),),
        "reason_codes": tuple(
            sorted(
                (
                    str(category),
                    _require_nonnegative_int(value, f"failure category {category}"),
                )
                for category, value in _require_mapping(
                    outcomes, "failure_category_counts"
                ).items()
            )
        ),
    }
    metrics: list[RawMetricEvidence] = []
    graph_reason = "dilemmadata.graph_distribution_not_in_compact_manifests"
    default_reason = "dilemmadata.metric_not_in_compact_manifests"
    for metric_id, spec in RAW_METRIC_CATALOG.items():
        if metric_id in count_values:
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=common_coverage,
                    count=_count(
                        metric_id,
                        count_values[metric_id],
                        discovered,
                        unit=spec.value_unit,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        scope=EvidenceScope.MANIFEST_REPLAY,
                        provenance=_RAW_PROVENANCE,
                    ),
                )
            )
        elif metric_id in categories:
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=common_coverage,
                    categories=tuple(
                        CategoryCount(
                            category=category,
                            count=_count(
                                metric_id,
                                value,
                                discovered,
                                unit=spec.value_unit,
                                denominator_unit=ObservationUnit.RECORD,
                                split=SplitScope.ALL,
                                scope=EvidenceScope.MANIFEST_REPLAY,
                                provenance=_RAW_PROVENANCE,
                            ),
                        )
                        for category, value in categories[metric_id]
                    ),
                )
            )
        else:
            reason = graph_reason if metric_id in {
                "graph_edge_counts",
                "graph_node_counts",
                "graph_size_distribution",
            } else default_reason
            denominator = accepted if metric_id.startswith("graph_") else discovered
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=_coverage(
                        denominator=denominator,
                        split=SplitScope.ALL,
                        scope=EvidenceScope.MANIFEST_REPLAY,
                        provenance=_RAW_PROVENANCE,
                        observed=False,
                        reason=reason,
                    ),
                )
            )
    if _require_int(raw_projection, "raw_compatible_note_projection_records") != discovered:
        raise EDAContractError(
            "dilemmadata.eda.raw_projection_mismatch",
            "raw projection coverage differs from discovered records",
        )
    return tuple(metrics)


def _raw_extension(
    audit: Mapping[str, object],
    production: Mapping[str, object],
    population: Mapping[str, object],
) -> SourceExtension:
    records = _require_mapping(audit, "records")
    grouping = _require_mapping(audit, "grouping")
    raw_projection = _require_mapping(audit, "raw_projection")
    outcomes = _require_mapping(production, "outcomes")
    dialect_status = _require_mapping(outcomes, "dialect_status_counts")
    denominator = _require_int(outcomes, "discovered_count")
    paper = _require_mapping(population, "paper_candidate")
    common_projection = _require_mapping(population, "common_projection_subset")

    def row(
        row_id: str,
        values: Sequence[tuple[str, int]],
        *,
        population_size: int = denominator,
        unit: ObservationUnit = ObservationUnit.RECORD,
    ) -> ExtensionRow:
        coverage = _coverage(
            denominator=population_size,
            split=SplitScope.ALL,
            scope=EvidenceScope.MANIFEST_REPLAY,
            provenance=_RAW_PROVENANCE,
            observed=True,
            unit=unit,
        )
        return ExtensionRow(
            row_id=row_id,
            payload={},
            counts=tuple(
                _count(
                    name,
                    value,
                    population_size,
                    unit=unit,
                    denominator_unit=unit,
                    split=SplitScope.ALL,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=_RAW_PROVENANCE,
                )
                for name, value in values
            ),
            coverage=coverage,
        )

    rows = (
        row(
            "dialect_inventory",
            (
                ("an_joint_records", _require_int(records, "an_joint_record_count")),
                ("dlc_records", _require_int(records, "dlc_record_count")),
            ),
        ),
        row(
            "dialect_conversion_outcomes",
            tuple(
                (key.replace(":", "_"), _require_nonnegative_int(value, key))
                for key, value in sorted(dialect_status.items())
            ),
        ),
        row(
            "candidate_group_membership",
            ((
                "candidate_group_records",
                _require_int(
                    grouping, "midi_note_event_multiset_equivalent_record_count"
                ),
            ),),
        ),
        row(
            "source_resolution_coverage",
            (
                (
                    "multiple_resolution_records",
                    _require_int(
                        raw_projection, "records_with_multiple_source_resolutions"
                    ),
                ),
                (
                    "single_resolution_records",
                    _require_int(raw_projection, "records_with_one_source_resolution"),
                ),
            ),
        ),
        row(
            "source_irregularity_presence",
            (
                (
                    "tie_continuation_records",
                    _require_int(raw_projection, "records_with_tie_continuations"),
                ),
                (
                    "zero_duration_records",
                    _require_int(raw_projection, "records_with_zero_duration_rows"),
                ),
            ),
        ),
        row(
            "paper_candidate_inventory",
            (
                ("paper_records", _require_int(paper, "record_count")),
                ("selection_exclusions", _require_int(paper, "selection_exclusion_count")),
                ("an_joint_records", _require_int(paper, "an_joint_record_count")),
                ("dlc_records", _require_int(paper, "dlc_record_count")),
            ),
        ),
        row(
            "paper_candidate_split",
            tuple(
                (f"{split_name}_records", _require_nonnegative_int(value, split_name))
                for split_name, value in sorted(
                    _require_mapping(paper, "split_record_counts").items()
                )
            ),
            population_size=_require_int(paper, "record_count"),
        ),
        row(
            "paper_candidate_component_split",
            tuple(
                (f"{split_name}_components", _require_nonnegative_int(value, split_name))
                for split_name, value in sorted(
                    _require_mapping(paper, "split_component_counts").items()
                )
            ),
            population_size=_require_int(paper, "canonical_component_count"),
            unit=ObservationUnit.CANONICAL_WORK,
        ),
        row(
            "common_subset_inventory",
            (
                ("subset_records", _require_int(common_projection, "record_count")),
                (
                    "outside_subset_records",
                    denominator - _require_int(common_projection, "record_count"),
                ),
                (
                    "an_joint_records",
                    _require_int(common_projection, "an_joint_record_count"),
                ),
                ("dlc_records", _require_int(common_projection, "dlc_record_count")),
            ),
        ),
        row(
            "common_subset_split",
            tuple(
                (f"{split_name}_records", _require_nonnegative_int(value, split_name))
                for split_name, value in sorted(
                    _require_mapping(common_projection, "split_record_counts").items()
                )
            ),
            population_size=_require_int(common_projection, "record_count"),
        ),
        row(
            "common_subset_component_split",
            tuple(
                (f"{split_name}_components", _require_nonnegative_int(value, split_name))
                for split_name, value in sorted(
                    _require_mapping(common_projection, "split_component_counts").items()
                )
            ),
            population_size=_require_int(
                common_projection, "canonical_component_count"
            ),
            unit=ObservationUnit.CANONICAL_WORK,
        ),
    )
    return SourceExtension(
        corpus=CorpusId.DILEMMADATA,
        namespace=DILEMMADATA_RAW_EXTENSION_NAMESPACE,
        schema_name="DilemmadataRawManifestReplay",
        schema_version="1.0.0",
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_RAW_PROVENANCE,
        rows=rows,
        target_free=True,
        work_identity=_SOURCE_COMPONENT_IDENTITY,
    )


def _require_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise EDAContractError(
            "dilemmadata.eda.manifest_invalid", f"{key!r} must be an object"
        )
    return item


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EDAContractError(
            "dilemmadata.eda.manifest_invalid",
            f"{label!r} must be a non-negative integer",
        )
    return value


def _require_int(value: Mapping[str, object], key: str) -> int:
    return _require_nonnegative_int(value.get(key), key)


def _validate_population_manifest(value: Mapping[str, object]) -> None:
    projected = {key: item for key, item in value.items() if key != "semantic_fingerprint"}
    if (
        value.get("schema") != "DilemmadataEDAPopulationManifest@1.0.0"
        or value.get("target_free") is not True
        or value.get("semantic_fingerprint") != _POPULATION_SEMANTIC_FINGERPRINT
        or canonical_json_sha256(projected) != _POPULATION_SEMANTIC_FINGERPRINT
    ):
        raise EDAContractError(
            "dilemmadata.eda.population_manifest_invalid",
            "target-free population projection differs from its semantic binding",
        )
    source = _require_mapping(value, "source_identity")
    if source != {
        "content_fingerprint": _SOURCE_CONTENT_FINGERPRINT,
        "release_commit": _SOURCE_RELEASE_COMMIT,
    }:
        raise EDAContractError(
            "dilemmadata.eda.population_manifest_invalid",
            "population projection differs from the pinned source release",
        )
    full = _require_mapping(value, "full_raw")
    paper = _require_mapping(value, "paper_candidate")
    common_projection = _require_mapping(value, "common_projection_subset")
    if {
        "record_count": _require_int(full, "record_count"),
        "an_joint_record_count": _require_int(full, "an_joint_record_count"),
        "dlc_record_count": _require_int(full, "dlc_record_count"),
        "canonical_component_count": _require_int(full, "canonical_component_count"),
    } != {
        "record_count": 1633,
        "an_joint_record_count": 353,
        "dlc_record_count": 1280,
        "canonical_component_count": 1507,
    }:
        raise EDAContractError(
            "dilemmadata.eda.population_manifest_invalid",
            "full source population facts changed",
        )
    if {
        "record_count": _require_int(paper, "record_count"),
        "an_joint_record_count": _require_int(paper, "an_joint_record_count"),
        "dlc_record_count": _require_int(paper, "dlc_record_count"),
        "selection_exclusion_count": _require_int(paper, "selection_exclusion_count"),
        "canonical_component_count": _require_int(paper, "canonical_component_count"),
    } != {
        "record_count": 1619,
        "an_joint_record_count": 353,
        "dlc_record_count": 1266,
        "selection_exclusion_count": 14,
        "canonical_component_count": 1507,
    }:
        raise EDAContractError(
            "dilemmadata.eda.population_manifest_invalid",
            "corrected paper-candidate population facts changed",
        )
    if {
        "record_count": _require_int(common_projection, "record_count"),
        "an_joint_record_count": _require_int(
            common_projection, "an_joint_record_count"
        ),
        "dlc_record_count": _require_int(common_projection, "dlc_record_count"),
        "canonical_component_count": _require_int(
            common_projection, "canonical_component_count"
        ),
    } != {
        "record_count": 719,
        "an_joint_record_count": 108,
        "dlc_record_count": 611,
        "canonical_component_count": 707,
    }:
        raise EDAContractError(
            "dilemmadata.eda.population_manifest_invalid",
            "legacy accepted subset facts changed",
        )
    expected_splits = (
        (
            paper,
            {"train": 1295, "validation": 162, "test": 162},
            {"train": 1209, "validation": 147, "test": 151},
        ),
        (
            common_projection,
            {"train": 577, "validation": 71, "test": 71},
            {"train": 565, "validation": 71, "test": 71},
        ),
    )
    for population, record_expected, component_expected in expected_splits:
        if _require_mapping(population, "split_record_counts") != record_expected or (
            _require_mapping(population, "split_component_counts") != component_expected
        ):
            raise EDAContractError(
                "dilemmadata.eda.population_manifest_invalid",
                "population split facts changed",
            )


def _load_bound_manifest(root: Path, path: str, expected_sha256: str) -> Mapping[str, object]:
    resolved = _tracked_path(root, path)
    _verify_hash(resolved, expected_sha256)
    return _load_json(resolved)


def _validate_supervision_manifests(
    manifests: Mapping[str, Mapping[str, object]],
) -> None:
    target = manifests["native"]
    common = manifests["common"]
    b3 = manifests["b3"]
    b4 = manifests["b4"]
    b5a = manifests["b5a"]
    b5b = manifests["b5b"]
    b5e = manifests["b5e"]
    b5h = manifests["b5h"]
    b4_test_lock = _require_mapping(b4, "test_lock")
    b5a_test_lock = _require_mapping(b5a, "test_lock")
    families = target.get("families")
    if (
        target.get("audit_fingerprint") != _TARGET_AUDIT_FINGERPRINT
        or not isinstance(families, list)
        or len(families) != 22
    ):
        raise EDAContractError(
            "dilemmadata.eda.native_manifest_invalid",
            "native-family manifest differs from the pinned 22-family audit",
        )
    if (
        common.get("manifest_fingerprint") != _COMMON_MANIFEST_FINGERPRINT
        or common.get("registry_fingerprint")
        != next(iter(APPROVED_PROJECTION_REGISTRIES.values())).fingerprint
        or common.get("ready") is not True
    ):
        raise EDAContractError(
            "dilemmadata.eda.common_manifest_invalid",
            "common-projection manifest differs from the approved registry binding",
        )
    if (
        b3.get("semantic_fingerprint") != _B3_SEMANTIC_FINGERPRINT
        or b3.get("valid") is not True
        or b3.get("ready") is not True
        or b3.get("training_run") is not False
        or b3.get("test_targets_used_for_evaluation") is not False
    ):
        raise EDAContractError(
            "dilemmadata.eda.b3_manifest_invalid",
            "B3 multitask evidence differs from the pinned source-free contract",
        )
    if (
        b4.get("semantic_fingerprint") != _B4_SEMANTIC_FINGERPRINT
        or b4.get("valid") is not True
        or b4.get("head_count") != 20
        or b4_test_lock.get("test_assignment_record_count") != 162
        or b4_test_lock.get("test_target_records_opened") != 0
        or b4_test_lock.get("test_target_rows_loaded") != 0
        or b4_test_lock.get("test_targets_counted") is not False
        or b4_test_lock.get("test_targets_used_for_decisions") is not False
    ):
        raise EDAContractError(
            "dilemmadata.eda.b4_manifest_invalid",
            "B4 class-balance evidence differs from the pinned audit",
        )
    if (
        b5a.get("semantic_fingerprint") != _B5A_SEMANTIC_FINGERPRINT
        or b5a.get("valid") is not True
        or b5a.get("head_count") != 20
        or b5a_test_lock.get("test_assignment_record_count") != 162
        or b5a_test_lock.get("test_target_records_opened") != 0
        or b5a_test_lock.get("test_target_rows_loaded") != 0
        or b5a_test_lock.get("test_targets_counted") is not False
        or b5a_test_lock.get("test_targets_used_for_decisions") is not False
    ):
        raise EDAContractError(
            "dilemmadata.eda.b5a_manifest_invalid",
            "B5A transposition evidence differs from the pinned audit",
        )
    if (
        b5b.get("audit_semantic_fingerprint") != _B5B_SEMANTIC_FINGERPRINT
        or b5b.get("valid") is not True
        or b5b.get("training_run") is not False
        or b5b.get("test_targets_used_for_evaluation") is not False
    ):
        raise EDAContractError(
            "dilemmadata.eda.b5b_manifest_invalid",
            "B5B training-policy evidence differs from the pinned audit",
        )
    if (
        b5e.get("evidence_fingerprint") != _B5E_EVIDENCE_FINGERPRINT
        or b5e.get("valid") is not True
        or _require_mapping(b5e, "decision").get("test_evaluated") is not False
        or any(
            not isinstance(run, Mapping)
            or run.get("test_evaluated") is not False
            or run.get("test_targets_used_for_evaluation") is not False
            for run in _require_mapping(b5e, "run_summaries").values()
        )
    ):
        raise EDAContractError(
            "dilemmadata.eda.b5e_manifest_invalid",
            "B5E observed profile evidence differs from the pinned result",
        )
    if (
        b5h.get("evidence_fingerprint") != _B5H_EVIDENCE_FINGERPRINT
        or b5h.get("full_orbit_profile_valid") is not True
        or b5h.get("full_orbit_training_run") is not False
        or b5h.get("test_targets_read") is not False
    ):
        raise EDAContractError(
            "dilemmadata.eda.b5h_manifest_invalid",
            "B5H full-orbit evidence differs from the pinned profile",
        )
    b5a_inputs = _require_mapping(b5a, "input_fingerprints")
    b5b_inputs = _require_mapping(b5b, "input_fingerprints")
    if (
        b5a_inputs.get("b3_semantic_fingerprint") != _B3_SEMANTIC_FINGERPRINT
        or b5a_inputs.get("b4_semantic_fingerprint") != _B4_SEMANTIC_FINGERPRINT
        or b5b_inputs != {
            "b3_semantic": _B3_SEMANTIC_FINGERPRINT,
            "b4_semantic": _B4_SEMANTIC_FINGERPRINT,
            "b5a_semantic": _B5A_SEMANTIC_FINGERPRINT,
        }
    ):
        raise EDAContractError(
            "dilemmadata.eda.manifest_lineage_invalid",
            "B3/B4/B5 evidence lineage is inconsistent",
        )


def _split_manifest_error(detail: str) -> EDAContractError:
    return EDAContractError("dilemmadata.eda.split_manifest_invalid", detail)


def _validate_split_manifest_shape(value: Mapping[str, object]) -> None:
    if type(value) is not dict or set(value) != _SPLIT_MANIFEST_FIELDS:
        raise _split_manifest_error(
            "split manifest must have the exact frozen top-level fields"
        )
    if type(value["schema"]) is not str or (
        value["schema"] != _SPLIT_MANIFEST_SCHEMA
    ):
        raise _split_manifest_error("split manifest schema is invalid")
    if type(value["fingerprint_policy"]) is not str or (
        value["fingerprint_policy"] != _SPLIT_MANIFEST_FINGERPRINT_POLICY
    ):
        raise _split_manifest_error("split manifest fingerprint policy is invalid")
    if type(value["semantic_fingerprint"]) is not str:
        raise _split_manifest_error("split semantic fingerprint must be a string")
    locked = value["locked_assignment_count"]
    if type(locked) is not int or locked != 162:
        raise _split_manifest_error(
            "locked_assignment_count must be the exact integer 162"
        )

    assignments = value["assignments"]
    if type(assignments) is not list or not assignments:
        raise _split_manifest_error("split assignments must be a non-empty JSON array")

    split_counts: Counter[str] = Counter()
    retained_record_ids: set[str] = set()
    for index, row in enumerate(assignments):
        if type(row) is not dict:
            raise _split_manifest_error(
                f"split assignment {index} must be a JSON object"
            )
        split = row.get("split")
        if type(split) is not str or split not in {
            SplitScope.TRAIN.value,
            SplitScope.VALIDATION.value,
            SplitScope.TEST.value,
        }:
            raise _split_manifest_error(
                f"split assignment {index} has an invalid string split"
            )
        split_counts[split] += 1
        if split == SplitScope.TEST.value:
            if set(row) != _LOCKED_ASSIGNMENT_FIELDS:
                raise _split_manifest_error(
                    "the TEST assignment template must contain only split"
                )
            continue
        if set(row) != _RETAINED_ASSIGNMENT_FIELDS:
            raise _split_manifest_error(
                f"retained split assignment {index} fields differ from the contract"
            )
        if type(row["assignment_manifest_fingerprint"]) is not str:
            raise _split_manifest_error(
                f"retained assignment {index} fingerprint must be a string"
            )
        if type(row["corpus"]) is not str or row["corpus"] != CorpusId.DILEMMADATA.value:
            raise _split_manifest_error(
                f"retained assignment {index} corpus must be dilemmadata"
            )
        record_id = row["record_id"]
        if type(record_id) is not str or not record_id or record_id != record_id.strip():
            raise _split_manifest_error(
                f"retained assignment {index} record_id must be a non-empty string"
            )
        if record_id in retained_record_ids:
            raise _split_manifest_error("retained assignment record IDs must be unique")
        retained_record_ids.add(record_id)
        if type(row["target_free"]) is not bool or row["target_free"] is not True:
            raise _split_manifest_error(
                f"retained assignment {index} target_free must be JSON true"
            )

    if split_counts != Counter({"train": 6, "validation": 3, "test": 1}):
        raise _split_manifest_error(
            "split assignment surface counts must remain 6/3/1"
        )


def _split_manifest_fingerprint(value: Mapping[str, object]) -> str:
    _validate_split_manifest_shape(value)
    assignments = value["assignments"]
    assert type(assignments) is list
    projected: list[dict[str, object]] = []
    for row in assignments:
        assert type(row) is dict
        projected.append(
            {
                key: item
                for key, item in row.items()
                if key != "assignment_manifest_fingerprint"
            }
        )
    return canonical_json_sha256(
        {
            "assignments": projected,
            "fingerprint_policy": value["fingerprint_policy"],
            "locked_assignment_count": value["locked_assignment_count"],
            "schema": value["schema"],
        }
    )


def _validate_split_manifest(value: Mapping[str, object]) -> None:
    fingerprint = _split_manifest_fingerprint(value)
    if (
        value["semantic_fingerprint"] != _SPLIT_MANIFEST_FINGERPRINT
        or fingerprint != _SPLIT_MANIFEST_FINGERPRINT
    ):
        raise EDAContractError(
            "dilemmadata.eda.split_manifest_fingerprint_mismatch",
            "target-free split assignment does not match its semantic binding",
        )
    assignments = value["assignments"]
    assert type(assignments) is list
    if any(
        row.get("split") != SplitScope.TEST.value
        and row["assignment_manifest_fingerprint"] != _SPLIT_MANIFEST_FINGERPRINT
        for row in assignments
    ):
        raise EDAContractError(
            "dilemmadata.eda.split_manifest_fingerprint_mismatch",
            "retained assignment differs from the split manifest fingerprint",
        )


def _expanded_split_assignments(
    value: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    _validate_split_manifest_shape(value)
    assignments = value.get("assignments")
    locked = _require_nonnegative_int(
        value.get("locked_assignment_count"), "locked_assignment_count"
    )
    if not isinstance(assignments, list) or not assignments:
        raise EDAContractError(
            "dilemmadata.eda.split_manifest_invalid",
            "split manifest requires assignments",
        )
    locked_templates = [
        row
        for row in assignments
        if isinstance(row, Mapping) and row.get("split") == SplitScope.TEST.value
    ]
    if len(locked_templates) != 1 or set(locked_templates[0]) != {"split"} or locked <= 0:
        raise EDAContractError(
            "dilemmadata.eda.split_manifest_invalid",
            "split manifest requires one identity-redacted locked assignment template",
        )
    retained = [row for row in assignments if row is not locked_templates[0]]
    return tuple((*retained, *({"split": SplitScope.TEST.value} for _ in range(locked))))


def _validate_target_fixture(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if value.get("schema") != "DilemmadataEDASupervisionFixture@1.0.0" or (
        value.get("evidence_scope") != EvidenceScope.MANIFEST_REPLAY.value
    ):
        raise EDAContractError(
            "dilemmadata.eda.supervision_fixture_invalid",
            "unsupported Dilemmadata supervision fixture",
        )
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise EDAContractError(
            "dilemmadata.eda.supervision_fixture_invalid",
            "supervision fixture requires records",
        )
    result: list[Mapping[str, object]] = []
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise EDAContractError(
                "dilemmadata.eda.supervision_fixture_invalid",
                "fixture record must be an object",
            )
        record_id = record.get("record_id")
        dialect = record.get("dialect")
        split = record.get("split")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in record_ids
            or dialect not in {"an_joint", "dlc"}
            or split not in {SplitScope.TRAIN.value, SplitScope.VALIDATION.value}
        ):
            raise EDAContractError(
                "dilemmadata.eda.supervision_fixture_invalid",
                "fixture record identity/dialect/split is invalid",
            )
        record_ids.add(record_id)
        for identity_key in ("canonical_work_id", "source_group_id"):
            identity = record.get(identity_key)
            if not isinstance(identity, str) or not identity:
                raise EDAContractError(
                    "dilemmadata.eda.supervision_fixture_invalid",
                    f"fixture record requires {identity_key}",
                )
        lineage = record.get("lineage_ids")
        if (
            not isinstance(lineage, list)
            or not lineage
            or any(not isinstance(item, str) or not item for item in lineage)
            or len(lineage) != len(set(lineage))
        ):
            raise EDAContractError(
                "dilemmadata.eda.supervision_fixture_invalid",
                "fixture lineage IDs must be non-empty and unique",
            )
        tasks = record.get("tasks")
        expected = {
            task_id
            for task_id in _OBSERVED_TASKS
            if (task_id.startswith("dilemmadata.an.")) == (dialect == "an_joint")
        }
        if not isinstance(tasks, Mapping) or set(tasks) != expected:
            raise EDAContractError(
                "dilemmadata.eda.supervision_fixture_invalid",
                "fixture record must carry the two expected dialect task rows",
            )
        for task_id, task_value in tasks.items():
            if not isinstance(task_value, Mapping):
                raise EDAContractError(
                    "dilemmadata.eda.supervision_fixture_invalid",
                    "fixture task row must be an object",
                )
            try:
                state = AvailabilityState(task_value.get("state"))
            except (TypeError, ValueError) as exc:
                raise EDAContractError(
                    "dilemmadata.eda.supervision_fixture_invalid",
                    "fixture task state is invalid",
                ) from exc
            source_value = task_value.get("source_value")
            source_surface_value = task_value.get("source_surface_value")
            if state == AvailabilityState.AVAILABLE:
                family = DILEMMADATA_SOURCE_FAMILY_BY_TASK[str(task_id)]
                if (
                    not isinstance(source_value, str)
                    or not source_value
                    or family.vocabulary is None
                    or source_value not in family.vocabulary
                ):
                    raise EDAContractError(
                        "dilemmadata.eda.supervision_fixture_invalid",
                        "available fixture value is outside the frozen native vocabulary",
                    )
                if source_surface_value is not None and (
                    task_id != "dilemmadata.dlc.chord.inversion"
                    or source_surface_value != "42"
                    or source_value != "2"
                ):
                    raise EDAContractError(
                        "dilemmadata.eda.supervision_fixture_invalid",
                        "only the frozen DLC 42-to-2 surface normalization is permitted",
                    )
            elif state in {AvailabilityState.MASKED, AvailabilityState.MISSING} and (
                source_value is not None or source_surface_value is not None
            ):
                raise EDAContractError(
                    "dilemmadata.eda.supervision_fixture_invalid",
                    "masked/missing fixture rows cannot fabricate a class value",
                )
        result.append(record)
    return tuple(result)


def _work_identity(value: Mapping[str, object]) -> VersionedIdentity:
    contract = _require_mapping(value, "work_identity_contract")
    if set(contract) != {
        "canonical_field", "identity", "lineage_field", "source_group_field", "version"
    }:
        raise EDAContractError(
            "dilemmadata.eda.work_identity_contract_invalid",
            "fixture work identity contract fields changed",
        )
    identity = contract.get("identity")
    version = contract.get("version")
    if not isinstance(identity, str) or not isinstance(version, str):
        raise EDAContractError(
            "dilemmadata.eda.work_identity_contract_invalid",
            "fixture work identity contract is malformed",
        )
    return VersionedIdentity(
        identity=identity,
        version=version,
        fingerprint=canonical_json_sha256(contract),
    )


def _assert_split_atomic(records: Sequence[Mapping[str, object]], field: str) -> None:
    assigned: dict[str, str] = {}
    for record in records:
        values: Sequence[object]
        raw_value = record[field]
        values = raw_value if isinstance(raw_value, list) else (raw_value,)
        split = str(record["split"])
        for value in values:
            assert isinstance(value, str)
            previous = assigned.setdefault(value, split)
            if previous != split:
                raise EDAContractError(
                    "dilemmadata.eda.identity_leakage",
                    f"{field} {value!r} crosses TRAIN/VALIDATION",
                )


def _vocabulary_identity(task_id: str) -> VersionedIdentity:
    family = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
    payload = {
        "encoding_mode": family.encoding_mode,
        "registry_version": DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
        "source_task_id": task_id,
        "vocabulary": family.vocabulary,
    }
    return VersionedIdentity(
        identity=f"{task_id}.source_vocabulary",
        version=DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
        fingerprint=canonical_json_sha256(payload),
    )


def _projection_for(
    task_id: str, dialect: str, source_value: str
) -> tuple[ProjectionMappingState, object | None]:
    table = _QUALITY_PROJECTION if task_id.endswith(".quality") else _INVERSION_PROJECTION
    return table.get(
        (dialect, source_value),
        (ProjectionMappingState.UNSUPPORTED, None),
    )


def _build_task(
    *,
    family: Any,
    split: SplitScope,
    records: Sequence[Mapping[str, object]],
    work_identity: VersionedIdentity,
) -> TaskFamilyEvidence:
    task_provenance = (
        _SURFACE_PROVENANCE
        if family.task_id in _OBSERVED_TASKS
        else _NATIVE_CATALOG_PROVENANCE
    )
    common = {
        "corpus": CorpusId.DILEMMADATA,
        "source_task_id": family.task_id,
        "dialect": family.dialect,
        "annotation_namespace": f"music_critic.dilemmadata.source_native.{family.dialect}",
        "vocabulary": _vocabulary_identity(family.task_id),
        "label_granularity": family.coordinate,
        "label_value_type": LabelValueType.CATEGORICAL,
        "observation_unit": ObservationUnit.TARGET_ROW,
        "split_scope": split,
        "evidence_scope": EvidenceScope.MANIFEST_REPLAY,
        "provenance": task_provenance,
        "work_identity": work_identity,
    }
    if family.task_id not in _OBSERVED_TASKS:
        return TaskFamilyEvidence(
            **common,
            status=ComputationStatus.NOT_COMPUTED,
            availability=None,
            reason_code="dilemmadata.manifest.family_split_distribution_not_replayed",
        )

    rows = [
        record
        for record in records
        if record["dialect"] == family.dialect and record["split"] == split.value
    ]
    states: Counter[AvailabilityState] = Counter()
    value_rows: dict[str, list[Mapping[str, object]]] = {}
    for record in rows:
        task = _require_mapping(_require_mapping(record, "tasks"), family.task_id)
        state = AvailabilityState(task["state"])
        states[state] += 1
        if state == AvailabilityState.AVAILABLE:
            source_value = task["source_value"]
            assert isinstance(source_value, str)
            value_rows.setdefault(source_value, []).append(record)

    available = states[AvailabilityState.AVAILABLE]
    record_denominator = len(rows)
    work_denominator = len({str(record["canonical_work_id"]) for record in rows})
    support: list[ClassSupport] = []
    for source_value, matching in sorted(value_rows.items()):
        identity = SourceValueIdentity(
            corpus=CorpusId.DILEMMADATA,
            source_task_id=family.task_id,
            dialect=family.dialect,
            source_value=source_value,
            value_kind=SourceValueKind.SCALAR,
        )
        support.append(
            ClassSupport(
                source_value=identity,
                occurrence_count=_count(
                    "occurrence_count",
                    len(matching),
                    available,
                    unit=ObservationUnit.LABEL_OCCURRENCE,
                    denominator_unit=ObservationUnit.TARGET_ROW,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=task_provenance,
                ),
                unique_record_count=_count(
                    "unique_record_count",
                    len({str(row["record_id"]) for row in matching}),
                    record_denominator,
                    unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=task_provenance,
                ),
                unique_work_count=_count(
                    "unique_work_count",
                    len({str(row["canonical_work_id"]) for row in matching}),
                    work_denominator,
                    unit=ObservationUnit.CANONICAL_WORK,
                    denominator_unit=ObservationUnit.CANONICAL_WORK,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=task_provenance,
                ),
            )
        )

    projection_states: Counter[ProjectionMappingState] = Counter()
    for record in rows:
        task = _require_mapping(_require_mapping(record, "tasks"), family.task_id)
        state = AvailabilityState(task["state"])
        if state == AvailabilityState.AVAILABLE:
            source_value = task["source_value"]
            assert isinstance(source_value, str)
            mapping_state, _ = _projection_for(
                family.task_id, family.dialect, source_value
            )
        else:
            mapping_state = {
                AvailabilityState.MASKED: ProjectionMappingState.MASKED,
                AvailabilityState.MISSING: ProjectionMappingState.MISSING,
                AvailabilityState.UNSUPPORTED: ProjectionMappingState.UNSUPPORTED,
            }[state]
        projection_states[mapping_state] += 1

    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    common_task = _COMMON_TASK_BY_SOURCE[family.task_id]
    projections = []
    for item in support:
        source_value = item.source_value.source_value
        assert isinstance(source_value, str)
        mapping_state, projected = _projection_for(
            family.task_id, family.dialect, source_value
        )
        projections.append(
            ProjectionEvidence(
                source_value=item.source_value,
                mapping_registry=registry,
                common_task_identity=common_task,
                native_state=AvailabilityState.AVAILABLE,
                mapping_state=mapping_state,
                projected_value=projected,
                provenance=("dilemmadata-common-harmonic-registry-1.0.0",),
            )
        )

    return TaskFamilyEvidence(
        **common,
        status=ComputationStatus.OBSERVED,
        availability=AvailabilityCounts(
            observation_unit=ObservationUnit.TARGET_ROW,
            denominator=record_denominator,
            available=available,
            masked=states[AvailabilityState.MASKED],
            missing=states[AvailabilityState.MISSING],
            unsupported=states[AvailabilityState.UNSUPPORTED],
            split_scope=split,
            evidence_scope=EvidenceScope.MANIFEST_REPLAY,
            provenance=task_provenance,
        ),
        class_support=tuple(support),
        projection_availability=(
            ProjectionAvailabilityCounts(
                corpus=CorpusId.DILEMMADATA,
                source_task_id=family.task_id,
                dialect=family.dialect,
                mapping_registry=registry,
                common_task_identity=common_task,
                observation_unit=ObservationUnit.TARGET_ROW,
                denominator=record_denominator,
                exact=projection_states[ProjectionMappingState.EXACT],
                coarsened=projection_states[ProjectionMappingState.COARSENED],
                ambiguous=projection_states[ProjectionMappingState.AMBIGUOUS],
                unsupported=projection_states[ProjectionMappingState.UNSUPPORTED],
                invalid=projection_states[ProjectionMappingState.INVALID],
                missing=projection_states[ProjectionMappingState.MISSING],
                masked=projection_states[ProjectionMappingState.MASKED],
                split_scope=split,
                evidence_scope=EvidenceScope.MANIFEST_REPLAY,
                provenance=("dilemmadata-common-harmonic-registry-1.0.0",),
            ),
        ),
        projections=tuple(projections),
    )


def _imbalance_payload(values: Sequence[tuple[str, str, str]]) -> dict[str, object]:
    counts = Counter(values)
    supports = sorted(counts.values())
    if not supports:
        return {"calculation_state": "no_native_values"}
    return {
        "aggregation_policy": "corpus_task_dialect_source_value_identity",
        "majority_share": max(supports) / sum(supports),
        "max_to_min_nonzero_ratio": max(supports) / min(supports),
    }


def _diagnostic_extension(
    split: SplitScope,
    records: Sequence[Mapping[str, object]],
    manifests: Mapping[str, Mapping[str, object]],
) -> SourceExtension:
    selected = [record for record in records if record["split"] == split.value]

    def dialect_rows(dialect: str) -> list[Mapping[str, object]]:
        return [record for record in selected if record["dialect"] == dialect]

    def available_values(
        subset: Sequence[Mapping[str, object]], task_id: str
    ) -> list[tuple[str, str, str]]:
        values: list[tuple[str, str, str]] = []
        for record in subset:
            task = _require_mapping(_require_mapping(record, "tasks"), task_id)
            if task["state"] == AvailabilityState.AVAILABLE.value:
                source_value = task["source_value"]
                assert isinstance(source_value, str)
                values.append((task_id, str(record["dialect"]), source_value))
        return values

    def row_coverage(denominator: int) -> MetricCoverage:
        return _coverage(
            denominator=denominator,
            split=split,
            scope=EvidenceScope.MANIFEST_REPLAY,
            provenance=_SUPERVISION_PROVENANCE,
            observed=True,
            unit=ObservationUnit.TARGET_ROW,
        )

    rows: list[ExtensionRow] = []
    for dialect, prefix in (("an_joint", "an"), ("dlc", "dlc")):
        subset = dialect_rows(dialect)
        quality_task = f"dilemmadata.{prefix}.chord.quality"
        inversion_task = f"dilemmadata.{prefix}.chord.inversion"
        for suffix, task_id in (("quality", quality_task), ("inversion", inversion_task)):
            rows.append(
                ExtensionRow(
                    row_id=f"{prefix}_{suffix}_imbalance",
                    payload=_imbalance_payload(available_values(subset, task_id)),
                    coverage=row_coverage(len(subset)),
                )
            )
        paired = sum(
            _require_mapping(_require_mapping(record, "tasks"), quality_task)["state"]
            == AvailabilityState.AVAILABLE.value
            and _require_mapping(
                _require_mapping(record, "tasks"), inversion_task
            )["state"]
            == AvailabilityState.AVAILABLE.value
            for record in subset
        )
        rows.append(
            ExtensionRow(
                row_id=f"{prefix}_quality_inversion_cooccurrence",
                payload={},
                counts=(
                    _count(
                        "paired_native_rows",
                        paired,
                        len(subset),
                        unit=ObservationUnit.TARGET_ROW,
                        denominator_unit=ObservationUnit.TARGET_ROW,
                        split=split,
                        scope=EvidenceScope.MANIFEST_REPLAY,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                ),
                coverage=row_coverage(len(subset)),
            )
        )

    combined_quality_values = [
        value
        for dialect, prefix in (("an_joint", "an"), ("dlc", "dlc"))
        for value in available_values(
            dialect_rows(dialect), f"dilemmadata.{prefix}.chord.quality"
        )
    ]
    rows.append(
        ExtensionRow(
            row_id="combined_identity_preserving_quality_imbalance",
            payload=_imbalance_payload(combined_quality_values),
            coverage=row_coverage(len(selected)),
        )
    )
    paired_combined = sum(
        1
        for record in selected
        if all(
            _require_mapping(_require_mapping(record, "tasks"), task_id)["state"]
            == AvailabilityState.AVAILABLE.value
            for task_id in (
                (
                    "dilemmadata.an.chord.quality"
                    if record["dialect"] == "an_joint"
                    else "dilemmadata.dlc.chord.quality"
                ),
                (
                    "dilemmadata.an.chord.inversion"
                    if record["dialect"] == "an_joint"
                    else "dilemmadata.dlc.chord.inversion"
                ),
            )
        )
    )
    rows.append(
        ExtensionRow(
            row_id="combined_quality_inversion_cooccurrence",
            payload={},
            counts=(
                _count(
                    "paired_native_rows",
                    paired_combined,
                    len(selected),
                    unit=ObservationUnit.TARGET_ROW,
                    denominator_unit=ObservationUnit.TARGET_ROW,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=_SUPERVISION_PROVENANCE,
                ),
            ),
            coverage=row_coverage(len(selected)),
        )
    )

    b3 = manifests["b3"]
    b4 = manifests["b4"]
    b5a = manifests["b5a"]
    b5b = manifests["b5b"]
    b5e = manifests["b5e"]
    b5h = manifests["b5h"]
    split_name = split.value
    b3_split = _require_mapping(b3, "split")
    split_records = _require_int(
        _require_mapping(b3_split, "record_counts"), split_name
    )
    split_components = _require_int(
        _require_mapping(b3_split, "component_counts"), split_name
    )

    def typed_row(
        row_id: str,
        *,
        denominator: int,
        unit: ObservationUnit,
        values: Sequence[tuple[str, int, ObservationUnit]],
        payload: Mapping[str, object] | None = None,
    ) -> None:
        coverage = _coverage(
            denominator=denominator,
            split=split,
            scope=EvidenceScope.MANIFEST_REPLAY,
            provenance=_SUPERVISION_PROVENANCE,
            observed=True,
            unit=unit,
        )
        rows.append(
            ExtensionRow(
                row_id=row_id,
                payload={} if payload is None else payload,
                counts=tuple(
                    _count(
                        name,
                        value,
                        denominator,
                        unit=value_unit,
                        denominator_unit=unit,
                        split=split,
                        scope=EvidenceScope.MANIFEST_REPLAY,
                        provenance=_SUPERVISION_PROVENANCE,
                    )
                    for name, value, value_unit in values
                ),
                coverage=coverage,
            )
        )

    typed_row(
        "corrected_population_records",
        denominator=split_records,
        unit=ObservationUnit.RECORD,
        values=(("paper_candidate_records", split_records, ObservationUnit.RECORD),),
    )
    typed_row(
        "corrected_population_components",
        denominator=split_components,
        unit=ObservationUnit.CANONICAL_WORK,
        values=(("source_components", split_components, ObservationUnit.CANONICAL_WORK),),
    )

    joint = _require_mapping(
        _require_mapping(b4, "joint_tuples"), "corrected_harmonic_event"
    )
    joint_split = _require_mapping(joint, split_name)
    joint_events = _require_int(joint_split, "row_count")
    typed_row(
        "corrected_joint_cooccurrence",
        denominator=joint_events,
        unit=ObservationUnit.EVENT,
        values=(
            ("joint_events", joint_events, ObservationUnit.EVENT),
            (
                "participating_records",
                _require_int(joint_split, "record_count"),
                ObservationUnit.RECORD,
            ),
            (
                "participating_components",
                _require_int(joint_split, "component_count"),
                ObservationUnit.CANONICAL_WORK,
            ),
        ),
        payload={
            "joint_components": [
                "local_key",
                "primary_degree",
                "secondary_degree",
                "quality",
                "inversion",
            ],
            "top_ten_share": _require_mapping(joint_split, "top_coverage").get("top_10"),
            "top_twenty_share": _require_mapping(joint_split, "top_coverage").get("top_20"),
        },
    )

    compatibility = _require_mapping(
        _require_mapping(b4, "joint_tuples"), "compatibility_note"
    )
    compatibility_split = _require_mapping(compatibility, split_name)
    compatibility_notes = _require_int(compatibility_split, "row_count")
    typed_row(
        "compatibility_joint_support",
        denominator=compatibility_notes,
        unit=ObservationUnit.NOTE,
        values=(
            ("compatible_notes", compatibility_notes, ObservationUnit.NOTE),
            (
                "canonical_harmonic_rows",
                _require_int(
                    compatibility_split, "canonical_harmonic_target_rows"
                ),
                ObservationUnit.EVENT,
            ),
            (
                "participating_records",
                _require_int(compatibility_split, "record_count"),
                ObservationUnit.RECORD,
            ),
            (
                "participating_components",
                _require_int(compatibility_split, "component_count"),
                ObservationUnit.CANONICAL_WORK,
            ),
        ),
        payload={
            "quality_space": "analysisgnn-quality-15-compat-e115182-v1",
            "semantic_role": "paper_text_compatibility_only",
        },
    )

    head_summaries = b4.get("head_summaries")
    if not isinstance(head_summaries, list) or len(head_summaries) != 20:
        raise EDAContractError(
            "dilemmadata.eda.b4_manifest_invalid",
            "B4 requires twenty head summaries",
        )
    for head in head_summaries:
        if not isinstance(head, Mapping):
            raise EDAContractError(
                "dilemmadata.eda.b4_manifest_invalid", "B4 head summary must be an object"
            )
        head_id = head.get("task_id")
        if not isinstance(head_id, str) or not head_id:
            raise EDAContractError(
                "dilemmadata.eda.b4_manifest_invalid", "B4 head identity is invalid"
            )
        vocabulary_size = _require_int(head, "vocabulary_size")
        observed_classes = _require_int(
            head, f"{split_name}_observed_class_count"
        )
        payload: dict[str, object] = {
            "head": head_id,
            "recommendation": head.get("recommendation"),
        }
        if split == SplitScope.TRAIN:
            payload["majority_share"] = head.get("majority_share")
            payload["max_to_min_nonzero_ratio"] = head.get(
                "max_to_min_nonzero_ratio"
            )
        rows.append(
            ExtensionRow(
                row_id=f"head_{head_id}_vocabulary_coverage",
                payload=payload,
                counts=(
                    _count(
                        "observed_vocabulary_classes",
                        observed_classes,
                        vocabulary_size,
                        unit=ObservationUnit.LABEL_OCCURRENCE,
                        denominator_unit=ObservationUnit.LABEL_OCCURRENCE,
                        split=split,
                        scope=EvidenceScope.MANIFEST_REPLAY,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                ),
                coverage=_coverage(
                    denominator=vocabulary_size,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=_SUPERVISION_PROVENANCE,
                    observed=True,
                    unit=ObservationUnit.LABEL_OCCURRENCE,
                ),
            )
        )

    focus_classes = _require_mapping(
        _require_mapping(
            _require_mapping(b4, "quality"), "corrected_quality_17"
        ),
        "focus_classes",
    )
    focus_slugs = {
        "augmented major tetrachord": "augmented_major_tetrachord",
        "augmented seventh chord": "augmented_seventh_chord",
        "augmented triad": "augmented_triad",
    }
    if set(focus_classes) != set(focus_slugs):
        raise EDAContractError(
            "dilemmadata.eda.b4_manifest_invalid",
            "B4 corrected-quality focus classes changed",
        )
    for class_label, slug in sorted(focus_slugs.items()):
        class_split = _require_mapping(
            _require_mapping(focus_classes, class_label), split_name
        )
        common_payload = {
            "class_label": class_label,
            "dialect_support": class_split.get("dialect_support"),
        }
        source_rows = _require_int(class_split, "canonical_target_row_count")
        typed_row(
            f"quality_{slug}_source_rows",
            denominator=source_rows,
            unit=ObservationUnit.TARGET_ROW,
            values=(
                ("canonical_source_rows", source_rows, ObservationUnit.TARGET_ROW),
                (
                    "an_source_rows",
                    _require_int(class_split, "an_target_row_count"),
                    ObservationUnit.TARGET_ROW,
                ),
                (
                    "dlc_source_rows",
                    _require_int(class_split, "dlc_target_row_count"),
                    ObservationUnit.TARGET_ROW,
                ),
            ),
            payload=common_payload,
        )
        events = _require_int(class_split, "entity_count")
        typed_row(
            f"quality_{slug}_events",
            denominator=events,
            unit=ObservationUnit.EVENT,
            values=(("harmonic_events", events, ObservationUnit.EVENT),),
            payload=common_payload,
        )
        class_records = _require_int(class_split, "record_count")
        typed_row(
            f"quality_{slug}_records",
            denominator=class_records,
            unit=ObservationUnit.RECORD,
            values=(
                ("supporting_records", class_records, ObservationUnit.RECORD),
                (
                    "an_records",
                    _require_int(class_split, "an_record_count"),
                    ObservationUnit.RECORD,
                ),
                (
                    "dlc_records",
                    _require_int(class_split, "dlc_record_count"),
                    ObservationUnit.RECORD,
                ),
            ),
            payload=common_payload,
        )
        components = _require_int(class_split, "component_count")
        typed_row(
            f"quality_{slug}_components",
            denominator=components,
            unit=ObservationUnit.CANONICAL_WORK,
            values=(
                ("supporting_components", components, ObservationUnit.CANONICAL_WORK),
                (
                    "an_components",
                    _require_int(class_split, "an_component_count"),
                    ObservationUnit.CANONICAL_WORK,
                ),
                (
                    "dlc_components",
                    _require_int(class_split, "dlc_component_count"),
                    ObservationUnit.CANONICAL_WORK,
                ),
            ),
            payload={
                **common_payload,
                "effective_support": {
                    "measurement_unit": "source_component",
                    "value": class_split.get("effective_component_count"),
                },
                "largest_component_share": class_split.get("largest_component_share"),
                "top_five_component_share": class_split.get("top_5_components_share"),
            },
        )

    roman = _require_mapping(b4, "roman_numeral_184")
    if split == SplitScope.TRAIN:
        roman_payload: Mapping[str, object] = {
            "absent_vocabulary_fraction": _require_int(roman, "train_absent_class_count") / 184,
            "top_ten_share": _require_mapping(roman, "top_coverage").get("top_10"),
            "top_twenty_share": _require_mapping(roman, "top_coverage").get("top_20"),
            "top_fifty_share": _require_mapping(roman, "top_coverage").get("top_50"),
        }
    else:
        validation_only = roman.get("validation_classes_absent_in_train")
        if not isinstance(validation_only, list):
            raise EDAContractError(
                "dilemmadata.eda.b4_manifest_invalid",
                "B4 Roman validation shift evidence is malformed",
            )
        roman_payload = {
            "absent_vocabulary_fraction": (
                _require_int(roman, "validation_absent_class_count") / 184
            ),
            "validation_only_labels": validation_only,
        }
    rows.append(
        ExtensionRow(
            row_id="roman_184_split_shift",
            payload=roman_payload,
            coverage=_coverage(
                denominator=split_records,
                split=split,
                scope=EvidenceScope.MANIFEST_REPLAY,
                provenance=_SUPERVISION_PROVENANCE,
                observed=True,
                unit=ObservationUnit.RECORD,
            ),
        )
    )

    head_roles = _require_mapping(_require_mapping(b5b, "contracts"), "head_roles")
    role_rows = head_roles.get("roles")
    if not isinstance(role_rows, list) or len(role_rows) != 20:
        raise EDAContractError(
            "dilemmadata.eda.b5b_manifest_invalid", "B5B head roles are malformed"
        )
    by_role = {"primary": [], "auxiliary": [], "deferred": []}
    deferred_reasons: dict[str, object] = {}
    for role in role_rows:
        if not isinstance(role, Mapping) or role.get("role") not in by_role:
            raise EDAContractError(
                "dilemmadata.eda.b5b_manifest_invalid", "B5B head role row is invalid"
            )
        task_id = role.get("task_id")
        if not isinstance(task_id, str):
            raise EDAContractError(
                "dilemmadata.eda.b5b_manifest_invalid", "B5B head role identity is invalid"
            )
        by_role[str(role["role"])].append(task_id)
        if role["role"] == "deferred":
            deferred_reasons[task_id] = role.get("deferred_reason")
    rows.append(
        ExtensionRow(
            row_id="advisory_head_roles",
            payload={
                "primary_heads": sorted(by_role["primary"]),
                "auxiliary_heads": sorted(by_role["auxiliary"]),
                "deferred_heads": sorted(by_role["deferred"]),
                "deferred_reasons": deferred_reasons,
                "interpretation": "training_policy_not_data_truth",
            },
            coverage=_coverage(
                denominator=split_records,
                split=split,
                scope=EvidenceScope.MANIFEST_REPLAY,
                provenance=_SUPERVISION_PROVENANCE,
                observed=True,
                unit=ObservationUnit.RECORD,
            ),
        )
    )

    if split == SplitScope.TRAIN:
        eligibility = _require_mapping(b5a, "eligibility_summary")
        train_records = _require_int(eligibility, "train_record_count")
        typed_row(
            "safe_shift_record_coverage",
            denominator=train_records,
            unit=ObservationUnit.RECORD,
            values=(
                ("base_records", train_records, ObservationUnit.RECORD),
                (
                    "full_orbit_records",
                    _require_int(eligibility, "records_with_12_valid_shifts"),
                    ObservationUnit.RECORD,
                ),
                (
                    "limited_orbit_records",
                    _require_int(eligibility, "records_with_2_to_11_valid_shifts"),
                    ObservationUnit.RECORD,
                ),
            ),
        )
        mapping = _require_mapping(b5a, "mapping_summary")
        typed_row(
            "safe_shift_mapping_rows",
            denominator=_require_int(mapping, "rows"),
            unit=ObservationUnit.TARGET_ROW,
            values=(
                ("valid_mapping_rows", _require_int(mapping, "valid"), ObservationUnit.TARGET_ROW),
                (
                    "invalid_mapping_rows",
                    _require_int(mapping, "invalid"),
                    ObservationUnit.TARGET_ROW,
                ),
            ),
        )
        orbit = _require_mapping(b5h, "orbit_table")
        nominal_pairs = _require_int(orbit, "nominal_record_shift_pairs")
        typed_row(
            "full_orbit_pairs",
            denominator=nominal_pairs,
            unit=ObservationUnit.AUGMENTED_PAIR,
            values=(
                (
                    "eligible_pairs",
                    _require_int(orbit, "eligible_train_pairs"),
                    ObservationUnit.AUGMENTED_PAIR,
                ),
                (
                    "excluded_pairs",
                    _require_int(orbit, "excluded_train_pairs"),
                    ObservationUnit.AUGMENTED_PAIR,
                ),
                (
                    "identity_pairs",
                    _require_int(orbit, "identity_pairs"),
                    ObservationUnit.AUGMENTED_PAIR,
                ),
            ),
            payload={
                "variants_are_independent_musical_works": orbit.get(
                    "variants_are_independent_musical_works"
                ),
                "pair_order": "stable_record_id_shift_pc_then_epoch_permutation",
            },
        )
        decision = _require_mapping(b5e, "decision")
        run_summaries = _require_mapping(b5e, "run_summaries")
        profiles = _require_mapping(b5b, "profiles")
        for profile_key in ("C0", "C1"):
            run = _require_mapping(run_summaries, profile_key)
            profile = _require_mapping(profiles, profile_key)
            configured_updates = _require_int(
                _require_mapping(
                    _require_mapping(profile, "optimizer_training_budget"),
                    "budgets",
                ),
                "main_optimizer_updates",
            )
            observed_updates = _require_int(run, "applied_updates")
            batch_size = _require_int(run, "batch_size")
            configured_presentations = configured_updates * batch_size
            observed_presentations = _require_int(run, "train_draws")
            if (
                observed_updates > configured_updates
                or observed_presentations > configured_presentations
                or observed_presentations != observed_updates * batch_size
            ):
                raise EDAContractError(
                    "dilemmadata.eda.b5e_manifest_invalid",
                    "B5E observed exposure exceeds or differs from B5B configuration",
                )
            typed_row(
                f"profile_{profile_key.lower()}_presentation_exposure",
                denominator=configured_presentations,
                unit=ObservationUnit.SAMPLER_PRESENTATION,
                values=(
                    (
                        "configured_presentations",
                        configured_presentations,
                        ObservationUnit.SAMPLER_PRESENTATION,
                    ),
                    (
                        "observed_presentations",
                        observed_presentations,
                        ObservationUnit.SAMPLER_PRESENTATION,
                    ),
                ),
                payload={
                    "profile_id": run.get("profile_id"),
                    "execution_state": "completed_seed_17_screen",
                    "selection_state": (
                        "selected_baseline"
                        if decision.get("selected_profile") == profile_key
                        else decision.get(f"{profile_key}_status")
                    ),
                },
            )
            typed_row(
                f"profile_{profile_key.lower()}_update_exposure",
                denominator=configured_updates,
                unit=ObservationUnit.OPTIMIZER_UPDATE,
                values=(
                    (
                        "configured_updates",
                        configured_updates,
                        ObservationUnit.OPTIMIZER_UPDATE,
                    ),
                    (
                        "observed_updates",
                        observed_updates,
                        ObservationUnit.OPTIMIZER_UPDATE,
                    ),
                ),
                payload={
                    "profile_id": run.get("profile_id"),
                    "execution_state": "completed_seed_17_screen",
                },
            )
        c2_profile = _require_mapping(b5h, "profile")
        c2_runtime = _require_mapping(c2_profile, "runtime")
        c2_presentations = _require_int(c2_runtime, "train_draw_budget")
        typed_row(
            "profile_c2_presentation_exposure",
            denominator=c2_presentations,
            unit=ObservationUnit.SAMPLER_PRESENTATION,
            values=(
                (
                    "configured_presentations_at_b5h_snapshot",
                    c2_presentations,
                    ObservationUnit.SAMPLER_PRESENTATION,
                ),
                (
                    "observed_presentations_at_b5h_snapshot",
                    0,
                    ObservationUnit.SAMPLER_PRESENTATION,
                ),
            ),
            payload={
                "profile_id": c2_profile.get("profile_id"),
                "snapshot_phase": "9E-B5H",
                "run_state_scope": "historical_b5h_planning_snapshot",
                "execution_state_at_snapshot": "configured_untrained",
                "current_run_state_included": False,
                "current_run_state_source": "docs/EXPERIMENT_LEDGER.md",
                "snapshot_evidence_fingerprint": _B5H_EVIDENCE_FINGERPRINT,
                "dataset_semantics": c2_profile.get("dataset_semantics"),
            },
        )
        c2_updates = _require_int(c2_runtime, "applied_update_budget")
        typed_row(
            "profile_c2_update_exposure",
            denominator=c2_updates,
            unit=ObservationUnit.OPTIMIZER_UPDATE,
            values=(
                (
                    "configured_updates_at_b5h_snapshot",
                    c2_updates,
                    ObservationUnit.OPTIMIZER_UPDATE,
                ),
                (
                    "observed_updates_at_b5h_snapshot",
                    0,
                    ObservationUnit.OPTIMIZER_UPDATE,
                ),
            ),
            payload={
                "profile_id": c2_profile.get("profile_id"),
                "snapshot_phase": "9E-B5H",
                "run_state_scope": "historical_b5h_planning_snapshot",
                "execution_state_at_snapshot": "configured_untrained",
                "current_run_state_included": False,
                "current_run_state_source": "docs/EXPERIMENT_LEDGER.md",
                "snapshot_evidence_fingerprint": _B5H_EVIDENCE_FINGERPRINT,
            },
        )
        rows.append(
            ExtensionRow(
                row_id="official_profile_contract",
                payload={
                    "profile_id": _require_mapping(
                        _require_mapping(b5b, "profiles"), "O"
                    ).get("profile_id"),
                    "execution_state": "partial_contract_non_runnable",
                },
                coverage=_coverage(
                    denominator=train_records,
                    split=split,
                    scope=EvidenceScope.MANIFEST_REPLAY,
                    provenance=_SUPERVISION_PROVENANCE,
                    observed=True,
                    unit=ObservationUnit.RECORD,
                ),
            )
        )
        typed_row(
            "dlc_42_surface_resolution",
            denominator=1,
            unit=ObservationUnit.TARGET_ROW,
            values=(("resolved_surface_rows", 1, ObservationUnit.TARGET_ROW),),
            payload={
                "surface_spelling": "42",
                "normalized_native_label": "2",
                "ordinal_label": "third",
                "resolution": "source_alias_then_exact_approved_registry",
            },
        )
    else:
        typed_row(
            "identity_only_validation_policy",
            denominator=split_records,
            unit=ObservationUnit.RECORD,
            values=(("identity_records", split_records, ObservationUnit.RECORD),),
            payload={"augmentation_state": "disabled_identity_only"},
        )

    return SourceExtension(
        corpus=CorpusId.DILEMMADATA,
        namespace=DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE,
        schema_name="DilemmadataSupervisionDiagnostics",
        schema_version="2.0.0",
        split_scope=split,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=_SUPERVISION_PROVENANCE,
        rows=tuple(rows),
        target_free=False,
        work_identity=_SOURCE_COMPONENT_IDENTITY,
    )


def dilemmadata_surface_value_identities(
    fixture: Mapping[str, object],
) -> tuple[SourceValueIdentity, ...]:
    """Return the three explicit AN/DLC inversion surface identities."""

    cases = fixture.get("surface_identity_cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise EDAContractError(
            "dilemmadata.eda.surface_identity_fixture_invalid",
            "expected exactly three surface identity cases",
        )
    expected = (
        ("an_joint", "dilemmadata.an.chord.inversion", "2"),
        ("dlc", "dilemmadata.dlc.chord.inversion", "2"),
        ("dlc", "dilemmadata.dlc.chord.inversion", "42"),
    )
    result = []
    for case, expected_case in zip(cases, expected, strict=True):
        if not isinstance(case, Mapping):
            raise EDAContractError(
                "dilemmadata.eda.surface_identity_fixture_invalid",
                "surface identity case must be an object",
            )
        actual = (
            case.get("dialect"),
            case.get("source_task_id"),
            case.get("source_value"),
        )
        if actual != expected_case:
            raise EDAContractError(
                "dilemmadata.eda.surface_identity_fixture_invalid",
                "surface identity cases must preserve AN 2 and DLC 2/42 exactly",
            )
        result.append(
            SourceValueIdentity(
                corpus=CorpusId.DILEMMADATA,
                source_task_id=expected_case[1],
                dialect=expected_case[0],
                source_value=case.get("source_value"),
                value_kind=SourceValueKind.SCALAR,
            )
        )
    identities = tuple(result)
    if len({item.identity for item in identities}) != len(identities):
        raise EDAContractError(
            "dilemmadata.eda.surface_identity_collision",
            "AN 2 and DLC 2/42 must remain distinct source identities",
        )
    return identities


class DilemmadataEDAAdapter:
    """Source-owned raw and supervision adapter under the frozen EDA API."""

    corpus = CorpusId.DILEMMADATA
    adapter_identity = DILEMMADATA_EDA_ADAPTER_IDENTITY
    extension_namespaces = (
        DILEMMADATA_RAW_EXTENSION_NAMESPACE,
        DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE,
    )

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        request, root = _request_root(request)
        audit_path = _tracked_path(root, _AUDIT_MANIFEST_PATH)
        production_path = _tracked_path(root, _PRODUCTION_MANIFEST_PATH)
        population_path = _tracked_path(root, _POPULATION_MANIFEST_PATH)
        _verify_hash(audit_path, _AUDIT_MANIFEST_SHA256)
        _verify_hash(production_path, _PRODUCTION_MANIFEST_SHA256)
        _verify_hash(population_path, _POPULATION_MANIFEST_SHA256)
        audit = _load_json(audit_path)
        production = _load_json(production_path)
        population = _load_json(population_path)
        _validate_population_manifest(population)
        corpus = _require_mapping(audit, "corpus")
        if (
            corpus.get("content_fingerprint") != _SOURCE_CONTENT_FINGERPRINT
            or corpus.get("release_commit") != _SOURCE_RELEASE_COMMIT
            or audit.get("semantic_fingerprint") != _AUDIT_SEMANTIC_FINGERPRINT
            or production.get("semantic_acceptance_fingerprint")
            != _PRODUCTION_SEMANTIC_FINGERPRINT
        ):
            raise EDAContractError(
                "dilemmadata.eda.source_binding_mismatch",
                "compact manifests differ from the pinned Dilemmadata release evidence",
            )
        metrics = _build_raw_metrics(audit, production)
        extension = _raw_extension(audit, production, population)
        return RawCorpusEDA(
            envelope=ReportEnvelope(
                schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
                schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
                report_kind=ReportKind.RAW_CORPUS,
                corpus=CorpusId.DILEMMADATA,
                source_identity=DILEMMADATA_SOURCE_IDENTITY,
                producer_identity=self.adapter_identity,
                repository_commit=request.repository_commit,
                evidence_scope=EvidenceScope.MANIFEST_REPLAY,
                execution_mode=ExecutionMode.MANIFEST_REPLAY,
                completeness_status=CompletenessStatus.PARTIAL,
                split_scope=SplitScope.ALL,
                observation_units=(
                    ObservationUnit.CANONICAL_WORK,
                    ObservationUnit.RECORD,
                ),
                input_manifests=(
                    _manifest_ref(
                        role="raw_audit",
                        identity="dilemmadata.audit_manifest",
                        version="1.1.0",
                        fingerprint=_AUDIT_MANIFEST_SHA256,
                        target_free=True,
                        path=_AUDIT_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="raw_acceptance",
                        identity="dilemmadata.production_manifest",
                        version="1.1.0",
                        fingerprint=_PRODUCTION_MANIFEST_SHA256,
                        target_free=True,
                        path=_PRODUCTION_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="population_inventory",
                        identity="dilemmadata.eda_population_manifest",
                        version="1.0.0",
                        fingerprint=_POPULATION_MANIFEST_SHA256,
                        target_free=True,
                        path=_POPULATION_MANIFEST_PATH,
                    ),
                ),
                invariants=(
                    InvariantEvidence(
                        code="dilemmadata.dialect_inventory_conservation",
                        status=InvariantStatus.PASSED,
                        provenance=_RAW_PROVENANCE,
                    ),
                    InvariantEvidence(
                        code="dilemmadata.inventory_conservation",
                        status=InvariantStatus.PASSED,
                        provenance=_RAW_PROVENANCE,
                    ),
                    InvariantEvidence(
                        code="dilemmadata.production_identity_leakage_recheck",
                        status=InvariantStatus.NOT_COMPUTED,
                        provenance=_RAW_PROVENANCE,
                        reason_code="dilemmadata.identity_domain_scan_not_run",
                    ),
                ),
                warnings=(
                    StructuredWarning(
                        code="dilemmadata.compact_manifest_replay_only",
                        message=(
                            "Tracked compact evidence was replayed; no source corpus "
                            "traversal or identity-domain rescan was performed."
                        ),
                        provenance=_RAW_PROVENANCE,
                    ),
                ),
                unavailable_reasons=(
                    UnavailableReason(
                        code="dilemmadata.graph_distribution_not_in_compact_manifests",
                        status=ComputationStatus.NOT_COMPUTED,
                        provenance=_RAW_PROVENANCE,
                        detail="Per-record approved graph distributions are absent.",
                    ),
                    UnavailableReason(
                        code="dilemmadata.metric_not_in_compact_manifests",
                        status=ComputationStatus.NOT_COMPUTED,
                        provenance=_RAW_PROVENANCE,
                        detail=(
                            "Per-record values needed for the remaining common metrics "
                            "are absent."
                        ),
                    ),
                ),
            ),
            semantic_payload=RawCorpusEDAPayload(
                metrics=metrics,
                graph_evidence=GraphEvidence(
                    status=ComputationStatus.NOT_COMPUTED,
                    target_free=None,
                    reason_code="dilemmadata.graph_distribution_not_in_compact_manifests",
                ),
                extensions=(extension,),
            ),
        )

    def build_supervision_eda(self, request: object) -> SupervisionEDA:
        request, root = _request_root(request)
        split_path = _tracked_path(root, _SPLIT_MANIFEST_PATH)
        fixture_path = _tracked_path(root, _SUPERVISION_FIXTURE_PATH)
        _verify_hash(split_path, _SPLIT_MANIFEST_SHA256)
        split_manifest = _load_json(split_path)
        _validate_split_manifest(split_manifest)
        assignments = _expanded_split_assignments(split_manifest)

        fixture_cache: dict[str, object] = {}

        def resolve_descriptor(record_id: str, split: SplitScope) -> _TargetDescriptor:
            if request.descriptor_observer is not None:
                request.descriptor_observer(record_id, split)
            return _TargetDescriptor(record_id, split, fixture_path)

        def load_target(
            descriptor: _TargetDescriptor, split: SplitScope
        ) -> Mapping[str, object]:
            if request.target_loader_observer is not None:
                request.target_loader_observer(descriptor.record_id, split)
            if not fixture_cache:
                _verify_hash(descriptor.manifest_path, _SUPERVISION_FIXTURE_SHA256)
                fixture_cache.update(_load_json(descriptor.manifest_path))
                fixture_cache["validated_records"] = _validate_target_fixture(
                    fixture_cache
                )
            records = fixture_cache["validated_records"]
            assert isinstance(records, tuple)
            matches = [
                row
                for row in records
                if row["record_id"] == descriptor.record_id
            ]
            if len(matches) != 1 or matches[0]["split"] != split.value:
                raise EDAContractError(
                    "dilemmadata.eda.target_descriptor_mismatch",
                    "guarded descriptor does not match exactly one fixture row",
                )
            return matches[0]

        loaded, test_lock = load_supervision_train_validation_only(
            CorpusId.DILEMMADATA,
            assignments,
            resolve_descriptor=resolve_descriptor,
            load_target=load_target,
            evidence_scope=EvidenceScope.MANIFEST_REPLAY,
            provenance=_ACCESS_PROVENANCE,
        )
        if not fixture_cache:
            raise EDAContractError(
                "dilemmadata.eda.target_fixture_not_loaded",
                "guarded TRAIN/VALIDATION target population is empty",
            )
        records = tuple(loaded)
        expected_records = fixture_cache["validated_records"]
        assert isinstance(expected_records, tuple)
        if {str(row["record_id"]) for row in records} != {
            str(row["record_id"]) for row in expected_records
        }:
            raise EDAContractError(
                "dilemmadata.eda.assignment_target_mismatch",
                "assignment and target fixture record sets differ",
            )
        for field in ("canonical_work_id", "source_group_id", "lineage_ids"):
            _assert_split_atomic(records, field)
        work_identity = _work_identity(fixture_cache)
        surface_identities = dilemmadata_surface_value_identities(fixture_cache)
        assert len(surface_identities) == 3

        manifests = {
            "native": _load_bound_manifest(
                root, _TARGET_MANIFEST_PATH, _TARGET_MANIFEST_SHA256
            ),
            "common": _load_bound_manifest(
                root, _COMMON_MANIFEST_PATH, _COMMON_MANIFEST_SHA256
            ),
            "b3": _load_bound_manifest(root, _B3_MANIFEST_PATH, _B3_MANIFEST_SHA256),
            "b4": _load_bound_manifest(root, _B4_MANIFEST_PATH, _B4_MANIFEST_SHA256),
            "b5a": _load_bound_manifest(
                root, _B5A_MANIFEST_PATH, _B5A_MANIFEST_SHA256
            ),
            "b5b": _load_bound_manifest(
                root, _B5B_MANIFEST_PATH, _B5B_MANIFEST_SHA256
            ),
            "b5e": _load_bound_manifest(
                root, _B5E_MANIFEST_PATH, _B5E_MANIFEST_SHA256
            ),
            "b5h": _load_bound_manifest(
                root, _B5H_MANIFEST_PATH, _B5H_MANIFEST_SHA256
            ),
        }
        _validate_supervision_manifests(manifests)

        tasks = tuple(
            _build_task(
                family=family,
                split=split,
                records=records,
                work_identity=work_identity,
            )
            for family in DILEMMADATA_SOURCE_FAMILIES
            for split in (SplitScope.TRAIN, SplitScope.VALIDATION)
        )
        return SupervisionEDA(
            envelope=ReportEnvelope(
                schema_name=SUPERVISION_EDA_SCHEMA_NAME,
                schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
                report_kind=ReportKind.SUPERVISION,
                corpus=CorpusId.DILEMMADATA,
                source_identity=DILEMMADATA_SOURCE_IDENTITY,
                producer_identity=self.adapter_identity,
                repository_commit=request.repository_commit,
                evidence_scope=EvidenceScope.MANIFEST_REPLAY,
                execution_mode=ExecutionMode.MANIFEST_REPLAY,
                completeness_status=CompletenessStatus.PARTIAL,
                split_scope=SplitScope.TRAIN_VALIDATION,
                observation_units=(
                    ObservationUnit.AUGMENTED_PAIR,
                    ObservationUnit.CANONICAL_WORK,
                    ObservationUnit.EVENT,
                    ObservationUnit.LABEL_OCCURRENCE,
                    ObservationUnit.NOTE,
                    ObservationUnit.OPTIMIZER_UPDATE,
                    ObservationUnit.RECORD,
                    ObservationUnit.SAMPLER_PRESENTATION,
                    ObservationUnit.SPLIT_ASSIGNMENT,
                    ObservationUnit.TARGET_ACCESS_ATTEMPT,
                    ObservationUnit.TARGET_ROW,
                ),
                input_manifests=(
                    _manifest_ref(
                        role="split_assignment",
                        identity="dilemmadata.eda_split_assignment",
                        version="1.0.0",
                        fingerprint=_SPLIT_MANIFEST_FINGERPRINT,
                        target_free=True,
                        path=_SPLIT_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="native_surface_cases",
                        identity="dilemmadata.eda_surface_cases",
                        version="1.0.0",
                        fingerprint=_SUPERVISION_FIXTURE_SHA256,
                        target_free=False,
                        path=_SUPERVISION_FIXTURE_PATH,
                    ),
                    _manifest_ref(
                        role="native_family_manifest",
                        identity="dilemmadata.native_family_manifest",
                        version="1.1.0",
                        fingerprint=_TARGET_MANIFEST_SHA256,
                        target_free=False,
                        path=_TARGET_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="common_harmonic_manifest",
                        identity="dilemmadata.common_harmonic_manifest",
                        version="1.0.0",
                        fingerprint=_COMMON_MANIFEST_SHA256,
                        target_free=False,
                        path=_COMMON_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="corrected_multitask_contract",
                        identity="dilemmadata.phase9eb3_multitask_contract",
                        version="2.0.0",
                        fingerprint=_B3_MANIFEST_SHA256,
                        target_free=False,
                        path=_B3_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="class_balance_audit",
                        identity="dilemmadata.phase9eb4_class_balance_audit",
                        version="1.0.0",
                        fingerprint=_B4_MANIFEST_SHA256,
                        target_free=False,
                        path=_B4_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="safe_transposition_audit",
                        identity="dilemmadata.phase9eb5a_transposition_audit",
                        version="1.0.0",
                        fingerprint=_B5A_MANIFEST_SHA256,
                        target_free=False,
                        path=_B5A_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="training_policy",
                        identity="dilemmadata.phase9eb5b_training_policy",
                        version="1.0.0",
                        fingerprint=_B5B_MANIFEST_SHA256,
                        target_free=False,
                        path=_B5B_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="training_result",
                        identity="dilemmadata.phase9eb5e_training_result",
                        version="1.0.0",
                        fingerprint=_B5E_MANIFEST_SHA256,
                        target_free=False,
                        path=_B5E_MANIFEST_PATH,
                    ),
                    _manifest_ref(
                        role="historical_full_orbit_planning_snapshot",
                        identity="dilemmadata.phase9eb5h_full_orbit_profile",
                        version="1.0.0",
                        fingerprint=_B5H_MANIFEST_SHA256,
                        target_free=False,
                        path=_B5H_MANIFEST_PATH,
                    ),
                ),
                invariants=(
                    InvariantEvidence(
                        code="dilemmadata.canonical_work_split_atomic",
                        status=InvariantStatus.PASSED,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    InvariantEvidence(
                        code="dilemmadata.lineage_split_atomic",
                        status=InvariantStatus.PASSED,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    InvariantEvidence(
                        code="dilemmadata.source_group_split_atomic",
                        status=InvariantStatus.PASSED,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    InvariantEvidence(
                        code="dilemmadata.surface_value_identity_separation",
                        status=InvariantStatus.PASSED,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                ),
                warnings=(
                    StructuredWarning(
                        code="dilemmadata.compact_manifest_replay",
                        message=(
                            "Corpus-scale findings replay committed aggregate evidence; the "
                            "native task rows are nine guarded surface-semantic cases and are "
                            "not a corpus distribution."
                        ),
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    StructuredWarning(
                        code="dilemmadata.dlc_42_surface_alias",
                        message=(
                            "DLC surface spelling 42 normalizes to native DLC value 2 and then "
                            "maps exactly to third inversion; AN value 2 remains second."
                        ),
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    StructuredWarning(
                        code="dilemmadata.b5h_historical_planning_snapshot",
                        message=(
                            "The B5H C2 exposure rows replay a historical planning "
                            "snapshot only. docs/EXPERIMENT_LEDGER.md is authoritative "
                            "for current experiment run-state."
                        ),
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                ),
                unavailable_reasons=(
                    UnavailableReason(
                        code="dilemmadata.fixture.other_native_families_not_replayed",
                        status=ComputationStatus.NOT_COMPUTED,
                        provenance=_SUPERVISION_PROVENANCE,
                        detail=(
                            "Only dialect-specific chord quality and inversion have fixture "
                            "case distributions; every other native family remains explicit and "
                            "uncomputed at split level."
                        ),
                    ),
                ),
            ),
            semantic_payload=SupervisionEDAPayload(
                tasks=tasks,
                test_lock=test_lock,
                extensions=(
                    _diagnostic_extension(SplitScope.TRAIN, records, manifests),
                    _diagnostic_extension(SplitScope.VALIDATION, records, manifests),
                ),
            ),
        )


__all__ = [
    "DILEMMADATA_EDA_ADAPTER_IDENTITY",
    "DILEMMADATA_EDA_ADAPTER_VERSION",
    "DILEMMADATA_EDA_CONTRACT_BASE",
    "DILEMMADATA_RAW_EXTENSION_NAMESPACE",
    "DILEMMADATA_SOURCE_IDENTITY",
    "DILEMMADATA_SUPERVISION_EXTENSION_NAMESPACE",
    "DilemmadataEDAAdapter",
    "DilemmadataEDARequest",
    "dilemmadata_surface_value_identities",
]
