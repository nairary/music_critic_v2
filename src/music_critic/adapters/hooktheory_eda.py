"""HookTheory implementation of the frozen multi-source EDA contract.

The observed raw path reads only a compact target-free summary of the 19
tracked Phase 2B golden cases.  The observed supervision path passes the full
split assignment inventory through the shared pre-open guard before it opens
any source-native annotation fixture.  No path in this module discovers or
scans the production corpus.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from music_critic.adapters.hooktheory import (
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
)
from music_critic.data import CanonicalPiece, TargetArray
from music_critic.data.serialization import canonical_json_sha256
from music_critic.eda import (
    AvailabilityCounts,
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
    NumericDistribution,
    ObservationUnit,
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
    TestTargetLockEvidence,
    UnavailableReason,
    UnitCount,
    VersionedIdentity,
    load_supervision_train_validation_only,
)


EDA_CONTRACT_SHA = "65eb32fb948efde0fa117d7d27d19d8f16fa25b4"
HOOKTHEORY_EDA_ADAPTER_VERSION = "1.0.0"
HOOKTHEORY_RAW_BOUNDED_SCHEMA = "HookTheoryRawBoundedManifest@1.0.0"
HOOKTHEORY_SPLIT_SCHEMA = "HookTheoryEDASplitAssignments@1.0.0"
HOOKTHEORY_SUPERVISION_MANIFEST_SCHEMA = (
    "HookTheoryEDASupervisionManifest@1.0.0"
)
HOOKTHEORY_RAW_EXTENSION_NAMESPACE = "hooktheory.raw_cases"

_RAW_MANIFEST_RELATIVE = (
    "tests/fixtures/hooktheory/eda_raw_bounded_manifest.json"
)
_SPLIT_MANIFEST_RELATIVE = (
    "tests/fixtures/hooktheory/eda_split_assignments.json"
)
_SUPERVISION_MANIFEST_RELATIVE = (
    "tests/fixtures/hooktheory/eda_supervision_manifest.json"
)
_RAW_MANIFEST_SHA256 = (
    "3b7246de6662ded92a177ebc7530506875c82d2ec585c926a87861baf60cdafb"
)
_SPLIT_MANIFEST_SHA256 = (
    "7302b00e5b946d9b021f484da9276dfba8bb87672db32297c592f6c48db200be"
)
_SUPERVISION_MANIFEST_SHA256 = (
    "2ebf774e712b54ba5b2cec763dee3de7b624103f4d27d8e4266490e8d64e5745"
)

_SOURCE_IDENTITY = VersionedIdentity(
    identity="map.hooktheory.raw_release",
    version="1.0.0",
    fingerprint=(
        "8ab601050d0b8c8752c3b6bf190d63edefa5fce07735ce823bca6a3922dff833"
    ),
)
_ADAPTER_CONTRACT = {
    "adapter": "music_critic.adapters.hooktheory_eda",
    "contract_base": EDA_CONTRACT_SHA,
    "manifest_value_comparison": "recursive_exact_json_types",
    "production_mode": "status_only_without_scan",
    "raw_mode": "tracked_bounded_summary",
    "repository_commit_policy": "required_explicit",
    "supervision_mode": "guarded_bounded_source_excerpts",
    "version": HOOKTHEORY_EDA_ADAPTER_VERSION,
}
HOOKTHEORY_EDA_ADAPTER_IDENTITY = VersionedIdentity(
    identity="music_critic.adapters.hooktheory_eda",
    version=HOOKTHEORY_EDA_ADAPTER_VERSION,
    fingerprint=canonical_json_sha256(_ADAPTER_CONTRACT),
)

_RAW_PROVENANCE = ("hooktheory.eda.phase_2b_bounded_summary",)
_RAW_EXTENSION_PROVENANCE = ("hooktheory.eda.phase_2b_case_splits",)
_SUPERVISION_PROVENANCE = ("hooktheory.eda.phase_2b_bounded_rows",)
_GUARD_PROVENANCE = ("hooktheory.eda.split_gate",)
_PRODUCTION_STATUS_PROVENANCE = ("hooktheory.eda.production_status",)

_SCALE_DEGREES = tuple(
    f"{accidental}{degree}"
    for degree in range(1, 8)
    for accidental in ("", "b", "#", "bb", "##")
)
_TASK_SPECS: Mapping[str, tuple[str, str, LabelValueType, tuple[str, ...] | None]] = {
    "theory.chord.adds": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.MULTI_LABEL,
        ("4", "6", "9"),
    ),
    "theory.chord.alterations": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.MULTI_LABEL,
        ("b5", "#5", "b9", "#9", "#11", "b13"),
    ),
    "theory.chord.borrowed": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.CATEGORICAL,
        None,
    ),
    "theory.chord.extent": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.CATEGORICAL,
        ("5", "7", "9", "11", "13"),
    ),
    "theory.chord.inversion": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.CATEGORICAL,
        ("0", "1", "2", "3"),
    ),
    "theory.chord.omits": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.MULTI_LABEL,
        ("3", "5"),
    ),
    "theory.chord.presence": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.CATEGORICAL,
        ("false", "true"),
    ),
    "theory.chord.root_degree": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.CATEGORICAL,
        ("0", "1", "2", "3", "4", "5", "6", "bVII"),
    ),
    "theory.chord.suspensions": (
        "hooktheory.chord",
        "chord_region",
        LabelValueType.MULTI_LABEL,
        ("2", "4"),
    ),
    "theory.local_key.mode": (
        "hooktheory.local_key",
        "key_region",
        LabelValueType.CATEGORICAL,
        None,
    ),
    "theory.local_key.tonic_pc": (
        "hooktheory.local_key",
        "key_region",
        LabelValueType.CATEGORICAL,
        tuple(str(value) for value in range(12)),
    ),
    "theory.melody.scale_degree": (
        "hooktheory.melody",
        "note",
        LabelValueType.CATEGORICAL,
        _SCALE_DEGREES,
    ),
}
HOOKTHEORY_SOURCE_TASKS = tuple(sorted(_TASK_SPECS))
_DIALECT = "hooktheory.theorytab"


Observer = Callable[[str, SplitScope], None]


@dataclass(frozen=True, slots=True)
class HookTheoryRawEDARequest:
    repository_root: Path
    repository_commit: str
    manifest_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class HookTheorySupervisionEDARequest:
    repository_root: Path
    repository_commit: str
    split_manifest_path: str | Path | None = None
    supervision_manifest_path: str | Path | None = None
    descriptor_observer: Observer | None = field(
        default=None, repr=False, compare=False
    )
    target_loader_observer: Observer | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class HookTheoryProductionStatusEDARequest:
    """Request truthful non-evidence when no production scan is available."""

    repository_commit: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EDAContractError(
                "hooktheory.eda.manifest_duplicate_key",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=Decimal,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid",
            f"cannot load HookTheory EDA input {path.name!r}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", "EDA input root must be an object"
        )
    return value, sha256(raw).hexdigest()


def _load_bound_json(
    path: Path, expected_fingerprint: str
) -> tuple[dict[str, object], str]:
    value, fingerprint = _load_json(path)
    if fingerprint != expected_fingerprint:
        raise EDAContractError(
            "hooktheory.eda.manifest_fingerprint_mismatch",
            f"tracked EDA input {path.name!r} differs from its adapter binding",
        )
    return value, fingerprint


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", f"{name} must be an object"
        )
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", f"{name} must be an array"
        )
    return value


def _require_fields(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise EDAContractError(
            "hooktheory.eda.manifest_fields_invalid",
            f"{name} fields must be exactly {sorted(expected)!r}",
        )


def _exact_json_equal(value: object, expected: object) -> bool:
    """Compare parsed JSON without Python's bool/int equality coercion."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact_json_equal(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _expect(value: object, expected: object, name: str) -> None:
    if not _exact_json_equal(value, expected):
        raise EDAContractError(
            "hooktheory.eda.manifest_mismatch",
            f"{name} must equal the pinned value {expected!r}",
        )


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid",
            f"{name} must be a non-negative integer",
        )
    return value


def _request_root(value: Path) -> Path:
    try:
        root = Path(value).resolve()
    except (TypeError, ValueError) as exc:
        raise EDAContractError(
            "hooktheory.eda.repository_root_invalid",
            "repository_root must be path-like",
        ) from exc
    if not root.is_dir():
        raise EDAContractError(
            "hooktheory.eda.repository_root_invalid",
            "repository_root must identify the checked-out repository",
        )
    return root


def _input_path(root: Path, value: str | Path | None, default: str) -> Path:
    path = Path(default if value is None else value)
    path = path if path.is_absolute() else root / path
    return path.resolve()


def _repository_relative(root: Path, path: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _fraction(value: object, name: str) -> Fraction:
    if not isinstance(value, str):
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", f"{name} must be an exact fraction"
        )
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", f"{name} is not an exact fraction"
        ) from exc
    if result < 0:
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", f"{name} must be non-negative"
        )
    return result


def _number(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _numeric_distribution(
    values: Sequence[object], *, measurement_unit: str, exact: bool
) -> NumericDistribution:
    parsed = (
        [_fraction(value, measurement_unit) for value in values]
        if exact
        else [Fraction(_integer(value, measurement_unit)) for value in values]
    )
    if not parsed:
        raise EDAContractError(
            "hooktheory.eda.manifest_invalid", "numeric distribution cannot be empty"
        )
    total = sum(parsed, Fraction())
    return NumericDistribution(
        measurement_unit=measurement_unit,
        minimum=_number(min(parsed)),
        maximum=_number(max(parsed)),
        mean=_number(total / len(parsed)),
    )


def _coverage(
    *,
    denominator: int | None,
    observed_count: int | None,
    unknown_count: int | None,
    split: SplitScope,
    evidence: EvidenceScope,
    provenance: tuple[str, ...],
    status: ComputationStatus = ComputationStatus.OBSERVED,
    reason: str | None = None,
    unit: ObservationUnit = ObservationUnit.RECORD,
) -> MetricCoverage:
    return MetricCoverage(
        observation_unit=unit,
        denominator=denominator,
        observed_count=observed_count,
        unknown_count=unknown_count,
        split_scope=split,
        evidence_scope=evidence,
        provenance=provenance,
        status=status,
        reason_code=reason,
    )


def _count(
    name: str,
    value: int,
    *,
    denominator: int,
    observation_unit: ObservationUnit,
    denominator_unit: ObservationUnit,
    split: SplitScope,
    evidence: EvidenceScope,
    provenance: tuple[str, ...],
) -> UnitCount:
    return UnitCount(
        name=name,
        observation_unit=observation_unit,
        value=value,
        denominator=denominator,
        denominator_unit=denominator_unit,
        split_scope=split,
        evidence_scope=evidence,
        provenance=provenance,
    )


def _validate_raw_manifest(manifest: Mapping[str, object]) -> None:
    _require_fields(
        manifest,
        {
            "categorical_values",
            "contract_base",
            "corpus",
            "evidence_basis",
            "inventory",
            "schema",
            "selection",
            "source_record_fingerprints",
            "source_release",
            "split_counts",
            "structural_values",
            "uncomputed_metrics",
        },
        "raw bounded manifest",
    )
    _expect(manifest["schema"], HOOKTHEORY_RAW_BOUNDED_SCHEMA, "schema")
    _expect(manifest["corpus"], CorpusId.HOOKTHEORY.value, "corpus")
    _expect(manifest["contract_base"], EDA_CONTRACT_SHA, "contract_base")
    _expect(
        manifest["evidence_basis"],
        ["phase_2b_golden_cases", "phase_2b_adapter_conversion"],
        "evidence_basis",
    )
    source = _mapping(manifest["source_release"], "source_release")
    _require_fields(source, {"fingerprint", "identity", "version"}, "source_release")
    _expect(source["identity"], _SOURCE_IDENTITY.identity, "source identity")
    _expect(source["version"], _SOURCE_IDENTITY.version, "source version")
    _expect(source["fingerprint"], _SOURCE_IDENTITY.fingerprint, "source fingerprint")
    selection = _mapping(manifest["selection"], "selection")
    _require_fields(
        selection,
        {"case_count", "converted_excerpt_count", "production_scan_run"},
        "selection",
    )
    _expect(selection["case_count"], 19, "selection.case_count")
    _expect(
        selection["converted_excerpt_count"],
        18,
        "selection.converted_excerpt_count",
    )
    _expect(selection["production_scan_run"], False, "selection.production_scan_run")
    splits = _mapping(manifest["split_counts"], "split_counts")
    _expect(dict(splits), {"train": 18, "test": 1}, "split_counts")
    inventory = _mapping(manifest["inventory"], "inventory")
    _require_fields(
        inventory,
        {
            "accepted_records",
            "conversion_outcomes",
            "discovered_records",
            "empty_records",
            "invalid_records",
            "oversize_records",
            "parse_outcomes",
            "quarantined_records",
            "reason_codes",
        },
        "inventory",
    )
    for name, expected in (
        ("discovered_records", 19),
        ("accepted_records", 18),
        ("quarantined_records", 1),
        ("invalid_records", 1),
        ("empty_records", 9),
        ("oversize_records", 0),
    ):
        _expect(inventory[name], expected, f"inventory.{name}")
    _expect(
        dict(_mapping(inventory["parse_outcomes"], "parse_outcomes")),
        {"parsed": 19},
        "parse_outcomes",
    )
    _expect(
        dict(_mapping(inventory["conversion_outcomes"], "conversion_outcomes")),
        {"accepted": 18, "quarantined": 1},
        "conversion_outcomes",
    )
    _expect(
        dict(_mapping(inventory["reason_codes"], "reason_codes")),
        {"hooktheory.missing_json_payload": 1},
        "reason_codes",
    )
    fingerprints = _mapping(
        manifest["source_record_fingerprints"], "source_record_fingerprints"
    )
    if len(fingerprints) != 19 or len(set(fingerprints.values())) != 19 or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in fingerprints.values()
    ):
        raise EDAContractError(
            "hooktheory.eda.manifest_mismatch",
            "bounded case source fingerprints must contain 19 distinct SHA-256 values",
        )
    structural = _mapping(manifest["structural_values"], "structural_values")
    expected_structural = {
        "bars",
        "beats",
        "density",
        "duration",
        "meter_changes",
        "notes",
        "onsets",
        "parts",
        "polyphony",
        "tempo_changes",
        "tracks",
    }
    _require_fields(structural, expected_structural, "structural_values")
    for name in expected_structural:
        values = _sequence(structural[name], f"structural_values.{name}")
        if len(values) != 18:
            raise EDAContractError(
                "hooktheory.eda.manifest_mismatch",
                f"{name} must contain one value per converted excerpt",
            )
    expected_uncomputed = {
        "cross_split_raw_identity_collisions",
        "duplicate_candidates",
        "graph_edge_counts",
        "graph_node_counts",
        "graph_size_distribution",
        "pitch_range",
        "tempo",
        "version_candidates",
    }
    if set(_sequence(manifest["uncomputed_metrics"], "uncomputed_metrics")) != (
        expected_uncomputed
    ):
        raise EDAContractError(
            "hooktheory.eda.manifest_mismatch", "uncomputed metric set changed"
        )


def _raw_metrics(manifest: Mapping[str, object]) -> tuple[RawMetricEvidence, ...]:
    inventory = _mapping(manifest["inventory"], "inventory")
    structural = _mapping(manifest["structural_values"], "structural_values")
    categorical = _mapping(manifest["categorical_values"], "categorical_values")
    inventory_coverage = _coverage(
        denominator=19,
        observed_count=19,
        unknown_count=0,
        split=SplitScope.ALL,
        evidence=EvidenceScope.BOUNDED,
        provenance=_RAW_PROVENANCE,
    )
    converted_coverage = _coverage(
        denominator=19,
        observed_count=18,
        unknown_count=1,
        split=SplitScope.ALL,
        evidence=EvidenceScope.BOUNDED,
        provenance=_RAW_PROVENANCE,
    )
    graph_coverage = _coverage(
        denominator=19,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        evidence=EvidenceScope.BOUNDED,
        provenance=_RAW_PROVENANCE,
        status=ComputationStatus.NOT_COMPUTED,
        reason="eda.target_free_unproven",
    )
    generic_uncomputed = _coverage(
        denominator=19,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        evidence=EvidenceScope.BOUNDED,
        provenance=_RAW_PROVENANCE,
        status=ComputationStatus.NOT_COMPUTED,
        reason="hooktheory.raw.metric_not_computed",
    )
    numeric_exact = {"density", "duration"}
    rows: list[RawMetricEvidence] = []
    for metric_id, spec in RAW_METRIC_CATALOG.items():
        if metric_id in {
            "accepted_records",
            "discovered_records",
            "quarantined_records",
        }:
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    count=_count(
                        metric_id,
                        _integer(inventory[metric_id], metric_id),
                        denominator=19,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        evidence=EvidenceScope.BOUNDED,
                        provenance=_RAW_PROVENANCE,
                    ),
                )
            )
        elif metric_id in {"empty_records", "oversize_records"}:
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=converted_coverage,
                    count=_count(
                        metric_id,
                        _integer(inventory[metric_id], metric_id),
                        denominator=19,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        evidence=EvidenceScope.BOUNDED,
                        provenance=_RAW_PROVENANCE,
                    ),
                )
            )
        elif metric_id == "invalid_records":
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    count=_count(
                        metric_id,
                        _integer(inventory[metric_id], metric_id),
                        denominator=19,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        evidence=EvidenceScope.BOUNDED,
                        provenance=_RAW_PROVENANCE,
                    ),
                )
            )
        elif metric_id in {"conversion_outcomes", "parse_outcomes", "reason_codes"}:
            values = _mapping(inventory[metric_id], metric_id)
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    categories=tuple(
                        CategoryCount(
                            category=category,
                            count=_count(
                                metric_id,
                                _integer(value, f"{metric_id}.{category}"),
                                denominator=19,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                                split=SplitScope.ALL,
                                evidence=EvidenceScope.BOUNDED,
                                provenance=_RAW_PROVENANCE,
                            ),
                        )
                        for category, value in sorted(values.items())
                    ),
                )
            )
        elif metric_id in structural:
            values = _sequence(structural[metric_id], metric_id)
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=converted_coverage,
                    numeric=_numeric_distribution(
                        values,
                        measurement_unit=spec.measurement_unit or metric_id,
                        exact=metric_id in numeric_exact,
                    ),
                )
            )
        elif metric_id in {"instruments", "meter", "percussion_presence", "programs"}:
            values = _mapping(categorical[metric_id], metric_id)
            rows.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=converted_coverage,
                    categories=tuple(
                        CategoryCount(
                            category=category,
                            count=_count(
                                metric_id,
                                _integer(value, f"{metric_id}.{category}"),
                                denominator=19,
                                observation_unit=spec.value_unit
                                or ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                                split=SplitScope.ALL,
                                evidence=EvidenceScope.BOUNDED,
                                provenance=_RAW_PROVENANCE,
                            ),
                        )
                        for category, value in sorted(values.items())
                    ),
                )
            )
        elif metric_id in {
            "graph_edge_counts",
            "graph_node_counts",
            "graph_size_distribution",
        }:
            rows.append(RawMetricEvidence(metric_id=metric_id, coverage=graph_coverage))
        else:
            rows.append(
                RawMetricEvidence(metric_id=metric_id, coverage=generic_uncomputed)
            )
    return tuple(rows)


def _raw_extension(manifest: Mapping[str, object]) -> SourceExtension:
    splits = _mapping(manifest["split_counts"], "split_counts")
    coverage = _coverage(
        denominator=19,
        observed_count=19,
        unknown_count=0,
        split=SplitScope.ALL,
        evidence=EvidenceScope.BOUNDED,
        provenance=_RAW_EXTENSION_PROVENANCE,
    )
    return SourceExtension(
        corpus=CorpusId.HOOKTHEORY,
        namespace=HOOKTHEORY_RAW_EXTENSION_NAMESPACE,
        schema_name="HookTheoryBoundedCaseSplitEvidence",
        schema_version="1.0.0",
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.BOUNDED,
        provenance=_RAW_EXTENSION_PROVENANCE,
        rows=(
            ExtensionRow(
                row_id="case_split_inventory",
                payload={},
                counts=(
                    _count(
                        "train_case_count",
                        _integer(splits["train"], "split_counts.train"),
                        denominator=19,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        evidence=EvidenceScope.BOUNDED,
                        provenance=_RAW_EXTENSION_PROVENANCE,
                    ),
                    _count(
                        "test_case_count",
                        _integer(splits["test"], "split_counts.test"),
                        denominator=19,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        split=SplitScope.ALL,
                        evidence=EvidenceScope.BOUNDED,
                        provenance=_RAW_EXTENSION_PROVENANCE,
                    ),
                ),
                coverage=coverage,
            ),
        ),
        target_free=True,
    )


def _build_bounded_raw(
    request: HookTheoryRawEDARequest,
    adapter_identity: VersionedIdentity,
) -> RawCorpusEDA:
    root = _request_root(request.repository_root)
    path = _input_path(root, request.manifest_path, _RAW_MANIFEST_RELATIVE)
    manifest, fingerprint = _load_bound_json(path, _RAW_MANIFEST_SHA256)
    _validate_raw_manifest(manifest)
    envelope = ReportEnvelope(
        schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
        schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.RAW_CORPUS,
        corpus=CorpusId.HOOKTHEORY,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.BOUNDED,
        execution_mode=ExecutionMode.BOUNDED_SCAN,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.ALL,
        observation_units=(
            ObservationUnit.INSTRUMENT,
            ObservationUnit.METER_EVENT,
            ObservationUnit.RECORD,
        ),
        input_manifests=(
            InputManifestRef(
                role="raw_summary",
                identity=VersionedIdentity(
                    identity="hooktheory.eda.raw_bounded_manifest",
                    version="1.0.0",
                    fingerprint=fingerprint,
                ),
                target_free=True,
                repository_relative_path=_repository_relative(root, path),
            ),
        ),
        invariants=(
            InvariantEvidence(
                code="hooktheory.raw.bounded_inventory_partition",
                status=InvariantStatus.PASSED,
                provenance=_RAW_PROVENANCE,
            ),
            InvariantEvidence(
                code="hooktheory.raw.graph_attestation",
                status=InvariantStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                reason_code="eda.target_free_unproven",
            ),
        ),
        warnings=(
            StructuredWarning(
                code="hooktheory.raw.bounded_scope",
                message=(
                    "Statistics cover selected Phase 2B excerpts and are not a "
                    "corpus-wide distribution."
                ),
                provenance=_RAW_PROVENANCE,
            ),
        ),
        unavailable_reasons=(
            UnavailableReason(
                code="eda.target_free_unproven",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                detail="No exact approved graph attestation exists for these excerpts.",
            ),
            UnavailableReason(
                code="hooktheory.raw.metrics_not_computed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                detail=(
                    "The compact summary omits pitch, tempo, duplicate, version, "
                    "and cross-split identity distributions."
                ),
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


def hooktheory_vocabulary_identity(task_id: str) -> VersionedIdentity:
    try:
        _, _, value_type, values = _TASK_SPECS[task_id]
    except KeyError as exc:
        raise EDAContractError(
            "hooktheory.eda.task_unknown", f"unknown HookTheory task {task_id!r}"
        ) from exc
    descriptor = {
        "dialect": _DIALECT,
        "openness": "open" if values is None else "closed",
        "task_id": task_id,
        "value_type": value_type.value,
        "values": None if values is None else list(values),
    }
    return VersionedIdentity(
        identity=f"hooktheory.{task_id}.vocabulary",
        version="1.0.0",
        fingerprint=canonical_json_sha256(descriptor),
    )


def _validate_split_manifest(manifest: Mapping[str, object]) -> Sequence[object]:
    _require_fields(manifest, {"assignments", "corpus", "schema"}, "split manifest")
    _expect(manifest["schema"], HOOKTHEORY_SPLIT_SCHEMA, "split schema")
    _expect(manifest["corpus"], CorpusId.HOOKTHEORY.value, "split corpus")
    assignments = _sequence(manifest["assignments"], "assignments")
    retained: set[str] = set()
    test_rows = 0
    for index, value in enumerate(assignments):
        row = _mapping(value, f"assignments[{index}]")
        if row.get("split") == SplitScope.TEST.value:
            if set(row) != {"split"}:
                raise EDAContractError(
                    "hooktheory.eda.assignment_invalid",
                    "the held-out assignment must remain split-only",
                )
            test_rows += 1
            continue
        _require_fields(row, {"record_id", "split"}, f"assignments[{index}]")
        if row["split"] != SplitScope.TRAIN.value or not isinstance(
            row["record_id"], str
        ):
            raise EDAContractError(
                "hooktheory.eda.assignment_invalid",
                "bounded retained assignments must be TRAIN records",
            )
        retained.add(row["record_id"])
    if len(assignments) != 18 or len(retained) != 17 or test_rows != 1:
        raise EDAContractError(
            "hooktheory.eda.assignment_mismatch",
            "bounded supervision requires 17 TRAIN rows and one split-only held-out row",
        )
    return assignments


def _validate_supervision_manifest(
    manifest: Mapping[str, object], retained_ids: set[str]
) -> Mapping[str, object]:
    _require_fields(
        manifest, {"corpus", "records", "schema"}, "supervision manifest"
    )
    _expect(
        manifest["schema"],
        HOOKTHEORY_SUPERVISION_MANIFEST_SCHEMA,
        "supervision schema",
    )
    _expect(manifest["corpus"], CorpusId.HOOKTHEORY.value, "supervision corpus")
    records = _mapping(manifest["records"], "supervision records")
    if set(records) != retained_ids:
        raise EDAContractError(
            "hooktheory.eda.supervision_assignment_mismatch",
            "supervision records must exactly match retained assignments",
        )
    for record_id, raw_descriptor in records.items():
        descriptor = _mapping(raw_descriptor, f"records.{record_id}")
        _require_fields(descriptor, {"path", "sha256", "split"}, f"records.{record_id}")
        _expect(descriptor["split"], SplitScope.TRAIN.value, f"records.{record_id}.split")
        _expect(
            descriptor["path"],
            f"tests/fixtures/hooktheory/cases/{record_id}.json",
            f"records.{record_id}.path",
        )
        fingerprint = descriptor["sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise EDAContractError(
                "hooktheory.eda.manifest_invalid",
                f"records.{record_id}.sha256 must be SHA-256",
            )
    return records


def _excerpt_piece(
    root: Path,
    case_id: str,
    descriptor: Mapping[str, object],
) -> CanonicalPiece:
    path = _input_path(root, str(descriptor["path"]), "")
    case, fingerprint = _load_json(path)
    _expect(fingerprint, descriptor["sha256"], f"{case_id} fixture fingerprint")
    _expect(case.get("case_id"), case_id, f"{case_id}.case_id")
    reference = _mapping(case.get("source_reference"), f"{case_id}.source_reference")
    _expect(reference.get("split"), SplitScope.TRAIN.value, f"{case_id}.split")
    clip_id = reference.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise EDAContractError(
            "hooktheory.eda.fixture_invalid", f"{case_id} has no clip identity"
        )
    excerpt = _mapping(case.get("raw_excerpt"), f"{case_id}.raw_excerpt")
    if excerpt.get("json_present") is False:
        raise EDAContractError(
            "hooktheory.eda.fixture_invalid",
            "the supervision universe cannot include an unconverted excerpt",
        )
    regions = _mapping(excerpt.get("regions"), f"{case_id}.regions")
    payload: dict[str, object] = dict(regions)
    for name in ("notes", "chords"):
        events = _sequence(excerpt.get(name), f"{case_id}.{name}")
        payload[name] = [
            _mapping(event, f"{case_id}.{name}")["value"] for event in events
        ]
    end_beat = Decimal(2)
    for name in ("notes", "chords"):
        for raw in payload[name]:  # type: ignore[index]
            event = _mapping(raw, f"{case_id}.{name}")
            beat, duration = event.get("beat"), event.get("duration")
            if (
                isinstance(beat, (int, Decimal))
                and not isinstance(beat, bool)
                and isinstance(duration, (int, Decimal))
                and not isinstance(duration, bool)
            ):
                end_beat = max(end_beat, Decimal(beat) + Decimal(duration))
    for name in ("keys", "tempos", "meters"):
        for raw in _sequence(payload.get(name, ()), f"{case_id}.{name}"):
            region = _mapping(raw, f"{case_id}.{name}")
            beat = region.get("beat")
            if isinstance(beat, (int, Decimal)) and not isinstance(beat, bool):
                end_beat = max(end_beat, Decimal(beat) + 1)
    payload["endBeat"] = end_beat
    structure_value = case.get("structure_excerpt")
    structure: Mapping[str, object] | None = None
    if structure_value is not None:
        source = _mapping(structure_value, f"{case_id}.structure_excerpt")
        structure = {
            "audio_path": f"audio/{clip_id}.mp3",
            "duration": source["duration"],
            "label": source["labels"],
            "ori_uid": source["ori_uid"],
            "segment_end": source["segment_end"],
            "segment_start": source["segment_start"],
        }
    try:
        return convert_hooktheory_record(
            clip_id,
            {"hash": clip_id, "split": SplitScope.TRAIN.value, "json": payload},
            config=HookTheoryAdapterConfig(
                dataset_name="hooktheory-eda-bounded",
                include_targets=True,
            ),
            structure_row=structure,
            source_path="4_merged.json",
        )
    except HookTheoryAdapterError as exc:
        raise EDAContractError(
            "hooktheory.eda.fixture_conversion_failed", str(exc)
        ) from exc


def _source_state(
    piece: CanonicalPiece,
    target: TargetArray,
    index: int,
    presence_by_entity: Mapping[str, str],
) -> str:
    if target.mask[index]:
        return "available"
    entity_id = target.entity_ids[index]
    if target.task.startswith("theory.chord.") and (
        target.task != "theory.chord.presence"
    ) and presence_by_entity.get(entity_id) == "false":
        return "masked"
    return "unsupported"


def _task_rows(pieces: Sequence[CanonicalPiece]) -> tuple[TaskFamilyEvidence, ...]:
    rows_by_task: dict[
        str, list[tuple[str, str, object | None]]
    ] = defaultdict(list)
    records = {piece.piece_id for piece in pieces}
    for piece in pieces:
        targets = {target.task: target for target in piece.targets}
        if set(targets) != set(HOOKTHEORY_SOURCE_TASKS):
            raise EDAContractError(
                "hooktheory.eda.task_inventory_mismatch",
                "converted excerpt does not expose the 12 HookTheory families",
            )
        presence = targets["theory.chord.presence"]
        presence_by_entity = {
            entity_id: str(value)
            for entity_id, value, available in zip(
                presence.entity_ids, presence.values, presence.mask, strict=True
            )
            if available and value is not None
        }
        for task_id, target in targets.items():
            spec = _TASK_SPECS[task_id]
            if target.value_type != spec[2].value:
                raise EDAContractError(
                    "hooktheory.eda.task_schema_mismatch",
                    f"{task_id} value type differs from its source contract",
                )
            expected_labels = spec[3]
            if target.class_labels != expected_labels:
                raise EDAContractError(
                    "hooktheory.eda.vocabulary_mismatch",
                    f"{task_id} vocabulary differs from its source contract",
                )
            for index, value in enumerate(target.values):
                rows_by_task[task_id].append(
                    (
                        piece.piece_id,
                        _source_state(piece, target, index, presence_by_entity),
                        value,
                    )
                )

    result: list[TaskFamilyEvidence] = []
    for task_id in HOOKTHEORY_SOURCE_TASKS:
        namespace, granularity, value_type, vocabulary = _TASK_SPECS[task_id]
        task_rows = rows_by_task[task_id]
        states = Counter(state for _, state, _ in task_rows)
        class_occurrences: Counter[str] = Counter()
        class_records: dict[str, set[str]] = defaultdict(set)
        empty_multilabel = 0
        for record_id, state, raw_value in task_rows:
            if state != "available":
                if raw_value is not None:
                    raise EDAContractError(
                        "hooktheory.eda.availability_value_mismatch",
                        "non-available source rows must not expose a class value",
                    )
                continue
            if value_type == LabelValueType.CATEGORICAL:
                if not isinstance(raw_value, str) or not raw_value:
                    raise EDAContractError(
                        "hooktheory.eda.source_value_invalid",
                        f"{task_id} available value must be a non-empty string",
                    )
                values = (raw_value,)
            else:
                if not isinstance(raw_value, tuple) or any(
                    not isinstance(value, str) or not value for value in raw_value
                ):
                    raise EDAContractError(
                        "hooktheory.eda.source_value_invalid",
                        f"{task_id} available value must be a string tuple",
                    )
                if not raw_value:
                    empty_multilabel += 1
                    continue
                values = raw_value
            for value in values:
                if vocabulary is not None and value not in vocabulary:
                    raise EDAContractError(
                        "hooktheory.eda.vocabulary_mismatch",
                        f"{task_id} source value {value!r} is outside its vocabulary",
                    )
                class_occurrences[value] += 1
                class_records[value].add(record_id)
        available = states["available"]
        support = tuple(
            ClassSupport(
                source_value=SourceValueIdentity(
                    corpus=CorpusId.HOOKTHEORY,
                    source_task_id=task_id,
                    dialect=_DIALECT,
                    source_value=value,
                    value_kind=SourceValueKind.SCALAR,
                ),
                occurrence_count=UnitCount(
                    name="occurrence_count",
                    observation_unit=ObservationUnit.LABEL_OCCURRENCE,
                    value=count,
                    denominator=available,
                    denominator_unit=ObservationUnit.TARGET_ROW,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.BOUNDED,
                    provenance=_SUPERVISION_PROVENANCE,
                ),
                unique_record_count=UnitCount(
                    name="unique_record_count",
                    observation_unit=ObservationUnit.RECORD,
                    value=len(class_records[value]),
                    denominator=len(records),
                    denominator_unit=ObservationUnit.RECORD,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.BOUNDED,
                    provenance=_SUPERVISION_PROVENANCE,
                ),
                unique_work_count=UnitCount(
                    name="unique_work_count",
                    observation_unit=ObservationUnit.LOGICAL_WORK,
                    value=None,
                    denominator=None,
                    denominator_unit=ObservationUnit.LOGICAL_WORK,
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.BOUNDED,
                    provenance=_SUPERVISION_PROVENANCE,
                    status=ComputationStatus.NOT_APPLICABLE,
                    reason_code="eda.work_identity_unproven",
                ),
            )
            for value, count in sorted(class_occurrences.items())
        )
        empty_count = None
        if value_type == LabelValueType.MULTI_LABEL:
            empty_count = UnitCount(
                name="empty_multilabel_available_count",
                observation_unit=ObservationUnit.TARGET_ROW,
                value=empty_multilabel,
                denominator=available,
                denominator_unit=ObservationUnit.TARGET_ROW,
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.BOUNDED,
                provenance=_SUPERVISION_PROVENANCE,
            )
        result.append(
            TaskFamilyEvidence(
                corpus=CorpusId.HOOKTHEORY,
                source_task_id=task_id,
                dialect=_DIALECT,
                annotation_namespace=namespace,
                vocabulary=hooktheory_vocabulary_identity(task_id),
                label_granularity=granularity,
                label_value_type=value_type,
                observation_unit=ObservationUnit.TARGET_ROW,
                split_scope=SplitScope.TRAIN,
                evidence_scope=EvidenceScope.BOUNDED,
                provenance=_SUPERVISION_PROVENANCE,
                status=ComputationStatus.OBSERVED,
                availability=AvailabilityCounts(
                    observation_unit=ObservationUnit.TARGET_ROW,
                    denominator=len(task_rows),
                    available=available,
                    masked=states["masked"],
                    missing=states["missing"],
                    unsupported=states["unsupported"],
                    split_scope=SplitScope.TRAIN,
                    evidence_scope=EvidenceScope.BOUNDED,
                    provenance=_SUPERVISION_PROVENANCE,
                ),
                work_identity=None,
                class_support=support,
                empty_multilabel_available_count=empty_count,
                projection_availability=(),
                projections=(),
            )
        )
        result.append(
            TaskFamilyEvidence(
                corpus=CorpusId.HOOKTHEORY,
                source_task_id=task_id,
                dialect=_DIALECT,
                annotation_namespace=namespace,
                vocabulary=hooktheory_vocabulary_identity(task_id),
                label_granularity=granularity,
                label_value_type=value_type,
                observation_unit=ObservationUnit.TARGET_ROW,
                split_scope=SplitScope.VALIDATION,
                evidence_scope=EvidenceScope.BOUNDED,
                provenance=_SUPERVISION_PROVENANCE,
                status=ComputationStatus.NOT_COMPUTED,
                availability=None,
                reason_code="hooktheory.validation_rows_unavailable",
            )
        )
    return tuple(result)


def _build_bounded_supervision(
    request: HookTheorySupervisionEDARequest,
    adapter_identity: VersionedIdentity,
) -> SupervisionEDA:
    root = _request_root(request.repository_root)
    if any(
        observer is not None and not callable(observer)
        for observer in (request.descriptor_observer, request.target_loader_observer)
    ):
        raise EDAContractError(
            "hooktheory.eda.request_invalid", "observers must be callable or null"
        )
    split_path = _input_path(
        root, request.split_manifest_path, _SPLIT_MANIFEST_RELATIVE
    )
    supervision_path = _input_path(
        root,
        request.supervision_manifest_path,
        _SUPERVISION_MANIFEST_RELATIVE,
    )
    split_manifest, split_fingerprint = _load_bound_json(
        split_path, _SPLIT_MANIFEST_SHA256
    )
    assignments_input = _validate_split_manifest(split_manifest)
    projected: list[dict[str, object]] = []
    retained_ids: set[str] = set()
    for value in assignments_input:
        row = _mapping(value, "assignment")
        if row.get("split") == SplitScope.TEST.value:
            projected.append({"split": SplitScope.TEST.value})
            continue
        record_id = row["record_id"]
        assert isinstance(record_id, str)
        retained_ids.add(record_id)
        projected.append(
            {
                "assignment_manifest_fingerprint": split_fingerprint,
                "corpus": CorpusId.HOOKTHEORY.value,
                "record_id": record_id,
                "split": row["split"],
                "target_free": True,
            }
        )

    manifest_cache: dict[str, Mapping[str, object]] = {}

    def resolve_descriptor(record_id: str, split: SplitScope) -> str:
        if request.descriptor_observer is not None:
            request.descriptor_observer(record_id, split)
        return record_id

    def load_piece(record_id: str, split: SplitScope) -> CanonicalPiece:
        if request.target_loader_observer is not None:
            request.target_loader_observer(record_id, split)
        if split != SplitScope.TRAIN:
            raise EDAContractError(
                "hooktheory.eda.split_invalid",
                "bounded source excerpts contain TRAIN rows only",
            )
        if not manifest_cache:
            target_manifest, target_fingerprint = _load_bound_json(
                supervision_path, _SUPERVISION_MANIFEST_SHA256
            )
            _expect(
                target_fingerprint,
                _SUPERVISION_MANIFEST_SHA256,
                "supervision manifest fingerprint",
            )
            manifest_cache.update(
                _validate_supervision_manifest(target_manifest, retained_ids)
            )
        try:
            descriptor = _mapping(manifest_cache[record_id], record_id)
        except KeyError as exc:
            raise EDAContractError(
                "hooktheory.eda.fixture_missing",
                f"missing bounded source excerpt {record_id!r}",
            ) from exc
        return _excerpt_piece(root, record_id, descriptor)

    loaded, test_lock = load_supervision_train_validation_only(
        CorpusId.HOOKTHEORY,
        tuple(projected),
        resolve_descriptor=resolve_descriptor,
        load_target=load_piece,
        evidence_scope=EvidenceScope.BOUNDED,
        provenance=_GUARD_PROVENANCE,
    )
    if len(loaded) != 17:
        raise EDAContractError(
            "hooktheory.eda.loaded_count_mismatch",
            "bounded supervision must load exactly 17 TRAIN excerpts",
        )
    tasks = _task_rows(loaded)
    envelope = ReportEnvelope(
        schema_name=SUPERVISION_EDA_SCHEMA_NAME,
        schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.SUPERVISION,
        corpus=CorpusId.HOOKTHEORY,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.BOUNDED,
        execution_mode=ExecutionMode.BOUNDED_SCAN,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.TRAIN_VALIDATION,
        observation_units=(
            ObservationUnit.LABEL_OCCURRENCE,
            ObservationUnit.LOGICAL_WORK,
            ObservationUnit.RECORD,
            ObservationUnit.SPLIT_ASSIGNMENT,
            ObservationUnit.TARGET_ACCESS_ATTEMPT,
            ObservationUnit.TARGET_ROW,
        ),
        input_manifests=(
            InputManifestRef(
                role="split_assignment",
                identity=VersionedIdentity(
                    identity="hooktheory.eda.bounded_split_assignment",
                    version="1.0.0",
                    fingerprint=split_fingerprint,
                ),
                target_free=True,
                repository_relative_path=_repository_relative(root, split_path),
            ),
            InputManifestRef(
                role="bounded_rows",
                identity=VersionedIdentity(
                    identity="hooktheory.eda.bounded_source_rows",
                    version="1.0.0",
                    fingerprint=_SUPERVISION_MANIFEST_SHA256,
                ),
                target_free=False,
                repository_relative_path=_repository_relative(root, supervision_path),
            ),
        ),
        invariants=(
            InvariantEvidence(
                code="hooktheory.supervision.record_identity_unique",
                status=InvariantStatus.PASSED,
                provenance=_SUPERVISION_PROVENANCE,
            ),
            InvariantEvidence(
                code="hooktheory.supervision.split_gate_enforced",
                status=InvariantStatus.PASSED,
                provenance=_GUARD_PROVENANCE,
            ),
            InvariantEvidence(
                code="hooktheory.supervision.work_identity",
                status=InvariantStatus.NOT_COMPUTED,
                provenance=_SUPERVISION_PROVENANCE,
                reason_code="eda.work_identity_unproven",
            ),
        ),
        warnings=(
            StructuredWarning(
                code="hooktheory.supervision.bounded_scope",
                message=(
                    "Class support covers 17 retained TRAIN excerpts; VALIDATION "
                    "and corpus-wide distributions were not computed."
                ),
                provenance=_SUPERVISION_PROVENANCE,
            ),
        ),
        unavailable_reasons=(
            UnavailableReason(
                code="eda.work_identity_unproven",
                status=ComputationStatus.NOT_APPLICABLE,
                provenance=_SUPERVISION_PROVENANCE,
                detail=(
                    "Partial source-group values do not prove a complete logical-work "
                    "identity contract."
                ),
            ),
            UnavailableReason(
                code="hooktheory.production_availability_untracked",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_SUPERVISION_PROVENANCE,
                detail=(
                    "No tracked machine-readable production availability table is "
                    "bound for TRAIN or VALIDATION."
                ),
            ),
        ),
    )
    return SupervisionEDA(
        envelope=envelope,
        semantic_payload=SupervisionEDAPayload(
            tasks=tasks,
            test_lock=test_lock,
            extensions=(),
        ),
    )


def _unknown_raw(
    request: HookTheoryProductionStatusEDARequest,
    adapter_identity: VersionedIdentity,
) -> RawCorpusEDA:
    coverage = _coverage(
        denominator=None,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        evidence=EvidenceScope.UNKNOWN,
        provenance=_PRODUCTION_STATUS_PROVENANCE,
        status=ComputationStatus.UNKNOWN,
        reason="eda.production_not_run",
    )
    graph_coverage = _coverage(
        denominator=None,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        evidence=EvidenceScope.UNKNOWN,
        provenance=_PRODUCTION_STATUS_PROVENANCE,
        status=ComputationStatus.UNKNOWN,
        reason="eda.target_free_unproven",
    )
    metrics = tuple(
        RawMetricEvidence(
            metric_id=metric_id,
            coverage=(
                graph_coverage
                if metric_id
                in {
                    "graph_edge_counts",
                    "graph_node_counts",
                    "graph_size_distribution",
                }
                else coverage
            ),
        )
        for metric_id in RAW_METRIC_CATALOG
    )
    envelope = ReportEnvelope(
        schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
        schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.RAW_CORPUS,
        corpus=CorpusId.HOOKTHEORY,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.UNKNOWN,
        execution_mode=ExecutionMode.NOT_EXECUTED,
        completeness_status=CompletenessStatus.UNKNOWN,
        split_scope=SplitScope.ALL,
        observation_units=(ObservationUnit.RECORD,),
        input_manifests=(),
        unavailable_reasons=(
            UnavailableReason(
                code="eda.production_not_run",
                status=ComputationStatus.UNKNOWN,
                provenance=_PRODUCTION_STATUS_PROVENANCE,
                detail="No production corpus scan was executed for this source task.",
            ),
            UnavailableReason(
                code="eda.target_free_unproven",
                status=ComputationStatus.UNKNOWN,
                provenance=_PRODUCTION_STATUS_PROVENANCE,
                detail="No production graph attestation was computed.",
            ),
        ),
    )
    return RawCorpusEDA(
        envelope=envelope,
        semantic_payload=RawCorpusEDAPayload(
            metrics=metrics,
            graph_evidence=GraphEvidence(
                status=ComputationStatus.UNKNOWN,
                target_free=None,
                reason_code="eda.target_free_unproven",
            ),
        ),
    )


def _unknown_supervision(
    request: HookTheoryProductionStatusEDARequest,
    adapter_identity: VersionedIdentity,
) -> SupervisionEDA:
    tasks = tuple(
        TaskFamilyEvidence(
            corpus=CorpusId.HOOKTHEORY,
            source_task_id=task_id,
            dialect=_DIALECT,
            annotation_namespace=_TASK_SPECS[task_id][0],
            vocabulary=hooktheory_vocabulary_identity(task_id),
            label_granularity=_TASK_SPECS[task_id][1],
            label_value_type=_TASK_SPECS[task_id][2],
            observation_unit=ObservationUnit.TARGET_ROW,
            split_scope=split,
            evidence_scope=EvidenceScope.UNKNOWN,
            provenance=_PRODUCTION_STATUS_PROVENANCE,
            status=ComputationStatus.UNKNOWN,
            availability=None,
            reason_code="eda.production_not_run",
        )
        for task_id in HOOKTHEORY_SOURCE_TASKS
        for split in (SplitScope.TRAIN, SplitScope.VALIDATION)
    )
    lock = TestTargetLockEvidence.not_executed(
        evidence_scope=EvidenceScope.UNKNOWN,
        provenance=_PRODUCTION_STATUS_PROVENANCE,
        reason_code="eda.test_targets_locked",
    )
    envelope = ReportEnvelope(
        schema_name=SUPERVISION_EDA_SCHEMA_NAME,
        schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.SUPERVISION,
        corpus=CorpusId.HOOKTHEORY,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.UNKNOWN,
        execution_mode=ExecutionMode.NOT_EXECUTED,
        completeness_status=CompletenessStatus.UNKNOWN,
        split_scope=SplitScope.TRAIN_VALIDATION,
        observation_units=(
            ObservationUnit.RECORD,
            ObservationUnit.SPLIT_ASSIGNMENT,
            ObservationUnit.TARGET_ACCESS_ATTEMPT,
            ObservationUnit.TARGET_ROW,
        ),
        input_manifests=(),
        unavailable_reasons=(
            UnavailableReason(
                code="eda.production_not_run",
                status=ComputationStatus.UNKNOWN,
                provenance=_PRODUCTION_STATUS_PROVENANCE,
                detail=(
                    "No tracked machine-readable production availability or "
                    "split distribution was executed for this source task."
                ),
            ),
        ),
    )
    return SupervisionEDA(
        envelope=envelope,
        semantic_payload=SupervisionEDAPayload(tasks=tasks, test_lock=lock),
    )


@dataclass(frozen=True, slots=True)
class HookTheoryEDAAdapter:
    corpus: CorpusId = CorpusId.HOOKTHEORY
    adapter_identity: VersionedIdentity = HOOKTHEORY_EDA_ADAPTER_IDENTITY
    extension_namespaces: tuple[str, ...] = (HOOKTHEORY_RAW_EXTENSION_NAMESPACE,)

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        if type(request) is HookTheoryRawEDARequest:
            return _build_bounded_raw(request, self.adapter_identity)
        if type(request) is HookTheoryProductionStatusEDARequest:
            return _unknown_raw(request, self.adapter_identity)
        raise EDAContractError(
            "hooktheory.eda.request_invalid",
            "raw EDA requires a bounded or production-status HookTheory request",
        )

    def build_supervision_eda(self, request: object) -> SupervisionEDA:
        if type(request) is HookTheorySupervisionEDARequest:
            return _build_bounded_supervision(request, self.adapter_identity)
        if type(request) is HookTheoryProductionStatusEDARequest:
            return _unknown_supervision(request, self.adapter_identity)
        raise EDAContractError(
            "hooktheory.eda.request_invalid",
            "supervision EDA requires a bounded or production-status HookTheory request",
        )


__all__ = [
    "EDA_CONTRACT_SHA",
    "HOOKTHEORY_EDA_ADAPTER_IDENTITY",
    "HOOKTHEORY_EDA_ADAPTER_VERSION",
    "HOOKTHEORY_RAW_BOUNDED_SCHEMA",
    "HOOKTHEORY_RAW_EXTENSION_NAMESPACE",
    "HOOKTHEORY_SOURCE_TASKS",
    "HOOKTHEORY_SPLIT_SCHEMA",
    "HOOKTHEORY_SUPERVISION_MANIFEST_SCHEMA",
    "HookTheoryEDAAdapter",
    "HookTheoryProductionStatusEDARequest",
    "HookTheoryRawEDARequest",
    "HookTheorySupervisionEDARequest",
    "hooktheory_vocabulary_identity",
]
