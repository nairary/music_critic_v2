"""Source-neutral contracts for deterministic multi-source EDA evidence.

The module is intentionally standard-library only.  It defines report shapes
and cross-field validation, but it never discovers a corpus, opens a target
sidecar, builds a graph, or imports a source adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from fractions import Fraction
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias
import unicodedata

from music_critic.data.serialization import (
    canonical_json_sha256,
    dumps_canonical_json,
)


EDA_ENVELOPE_SCHEMA_NAME = "MultiSourceEDAEnvelope"
EDA_ENVELOPE_SCHEMA_VERSION = "1.0.0"
RAW_CORPUS_EDA_SCHEMA_NAME = "RawCorpusEDA"
RAW_CORPUS_EDA_SCHEMA_VERSION = "1.0.0"
SUPERVISION_EDA_SCHEMA_NAME = "SupervisionEDA"
SUPERVISION_EDA_SCHEMA_VERSION = "1.0.0"
EDA_CAPABILITY_REGISTRY_SCHEMA_NAME = "MultiSourceEDACapabilityRegistry"
EDA_CAPABILITY_REGISTRY_SCHEMA_VERSION = "1.0.0"
EDA_TEST_TARGET_LOCK_VERSION = "1.0.0"
EDA_ADAPTER_REGISTRY_VERSION = "1.0.0"
EDA_SOURCE_VALUE_IDENTITY_VERSION = "1.0.0"
EDA_SOURCE_EXTENSION_VERSION = "1.0.0"
EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE = "split_assignment"
EDA_SCHEMA_VERSION_POLICY = "semantic_versioning_fail_closed_v1"
EDA_FLOAT_POLICY = "finite_binary64_json_shortest_roundtrip_v1"
EDA_QUANTILE_POLICY = "r7_linear_interpolation_sorted_finite_v1"

RAW_CORPUS_EDA_SCHEMA = (
    f"{RAW_CORPUS_EDA_SCHEMA_NAME}@{RAW_CORPUS_EDA_SCHEMA_VERSION}"
)
SUPERVISION_EDA_SCHEMA = (
    f"{SUPERVISION_EDA_SCHEMA_NAME}@{SUPERVISION_EDA_SCHEMA_VERSION}"
)
EDA_CAPABILITY_REGISTRY_SCHEMA = (
    f"{EDA_CAPABILITY_REGISTRY_SCHEMA_NAME}@"
    f"{EDA_CAPABILITY_REGISTRY_SCHEMA_VERSION}"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_VERSION_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_EXTENSION_NAMESPACE_RE = re.compile(
    r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)+$"
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
_SCHEMA_DATACLASS_TYPES: frozenset[type[object]] = frozenset()


class EDAContractError(ValueError):
    """A stable, fail-closed multi-source EDA contract violation."""

    def __init__(self, category: str, message: str, *, path: str = "$") -> None:
        self.category = category
        self.path = path
        super().__init__(f"[{category}] {path}: {message}")


class CorpusId(StrEnum):
    DILEMMADATA = "dilemmadata"
    HOOKTHEORY = "hooktheory"
    PDMX = "pdmx"
    POP909_CL = "pop909_cl"


class ReportKind(StrEnum):
    RAW_CORPUS = "raw_corpus_eda"
    SUPERVISION = "supervision_eda"


class EvidenceScope(StrEnum):
    FIXTURE = "fixture"
    MANIFEST_REPLAY = "manifest_replay"
    BOUNDED = "bounded"
    PRODUCTION = "production"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


class ExecutionMode(StrEnum):
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    MANIFEST_REPLAY = "manifest_replay"
    BOUNDED_SCAN = "bounded_scan"
    PRODUCTION_SCAN = "production_scan"
    NOT_EXECUTED = "not_executed"


class CompletenessStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_COMPUTED = "not_computed"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class SplitScope(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TRAIN_VALIDATION = "train_validation"
    TEST = "test"
    ALL = "all"
    UNSPLIT = "unsplit"
    UNKNOWN = "unknown"


class ObservationUnit(StrEnum):
    SPLIT_ASSIGNMENT = "split_assignment"
    TARGET_ACCESS_ATTEMPT = "target_access_attempt"
    SOURCE_FILE = "source_file"
    RECORD = "record"
    LOGICAL_WORK = "logical_work"
    CANONICAL_WORK = "canonical_work"
    EXCERPT = "excerpt"
    EVENT = "event"
    ONSET = "onset"
    NOTE = "note"
    BAR = "bar"
    BEAT = "beat"
    TRACK = "track"
    PART = "part"
    TEMPO_EVENT = "tempo_event"
    METER_EVENT = "meter_event"
    INSTRUMENT = "instrument"
    PROGRAM = "program"
    TARGET_ROW = "target_row"
    LABEL_OCCURRENCE = "label_occurrence"
    AUGMENTED_PAIR = "augmented_pair"
    SAMPLER_PRESENTATION = "sampler_presentation"
    OPTIMIZER_UPDATE = "optimizer_update"
    GRAPH_NODE = "graph_node"
    GRAPH_EDGE = "graph_edge"
    RAW_IDENTITY_COLLISION = "raw_identity_collision"


_RAW_ALLOWED_OBSERVATION_UNITS = frozenset(
    {
        ObservationUnit.SPLIT_ASSIGNMENT,
        ObservationUnit.SOURCE_FILE,
        ObservationUnit.RECORD,
        ObservationUnit.LOGICAL_WORK,
        ObservationUnit.CANONICAL_WORK,
        ObservationUnit.EXCERPT,
        ObservationUnit.EVENT,
        ObservationUnit.ONSET,
        ObservationUnit.NOTE,
        ObservationUnit.BAR,
        ObservationUnit.BEAT,
        ObservationUnit.TRACK,
        ObservationUnit.PART,
        ObservationUnit.TEMPO_EVENT,
        ObservationUnit.METER_EVENT,
        ObservationUnit.INSTRUMENT,
        ObservationUnit.PROGRAM,
        ObservationUnit.GRAPH_NODE,
        ObservationUnit.GRAPH_EDGE,
        ObservationUnit.RAW_IDENTITY_COLLISION,
    }
)


class ComputationStatus(StrEnum):
    OBSERVED = "observed"
    UNKNOWN = "unknown"
    NOT_COMPUTED = "not_computed"
    NOT_APPLICABLE = "not_applicable"
    LOCKED = "locked"


class AvailabilityState(StrEnum):
    AVAILABLE = "available"
    MASKED = "masked"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


class LabelValueType(StrEnum):
    CATEGORICAL = "categorical"
    MULTI_LABEL = "multi_label"


class SourceValueKind(StrEnum):
    SCALAR = "scalar"
    MULTI_LABEL = "multi_label"
    EMPTY_MULTI_LABEL = "empty_multi_label"


class ProjectionMappingState(StrEnum):
    EXACT = "exact"
    COARSENED = "coarsened"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    MISSING = "missing"
    MASKED = "masked"


class MetricSummaryKind(StrEnum):
    COUNT = "count"
    NUMERIC = "numeric_distribution"
    CATEGORICAL = "categorical_distribution"


class InvariantStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_COMPUTED = "not_computed"


class EDAReasonCode(StrEnum):
    MANIFEST_UNAVAILABLE = "eda.manifest_unavailable"
    METRIC_NOT_COMPUTED = "eda.metric_not_computed"
    NOT_APPLICABLE = "eda.not_applicable"
    PRODUCTION_NOT_RUN = "eda.production_not_run"
    SOURCE_UNAVAILABLE = "eda.source_unavailable"
    TARGET_FREE_UNPROVEN = "eda.target_free_unproven"
    TEST_TARGETS_LOCKED = "eda.test_targets_locked"
    WORK_IDENTITY_UNPROVEN = "eda.work_identity_unproven"


_SCOPE_TO_EXECUTION = {
    EvidenceScope.FIXTURE: ExecutionMode.SYNTHETIC_FIXTURE,
    EvidenceScope.MANIFEST_REPLAY: ExecutionMode.MANIFEST_REPLAY,
    EvidenceScope.BOUNDED: ExecutionMode.BOUNDED_SCAN,
    EvidenceScope.PRODUCTION: ExecutionMode.PRODUCTION_SCAN,
    EvidenceScope.UNKNOWN: ExecutionMode.NOT_EXECUTED,
    EvidenceScope.UNAVAILABLE: ExecutionMode.NOT_EXECUTED,
}

_OPERATIONAL_KEYS = frozenset(
    {
        "absolute_path",
        "cwd",
        "duration_seconds",
        "finished_at",
        "hostname",
        "host_name",
        "pid",
        "started_at",
        "timestamp",
        "wall_clock_duration",
        "wall_clock_seconds",
    }
)
_OPERATIONAL_STRING_KEYS = frozenset(
    {
        "absolute_path",
        "cwd",
        "finished_at",
        "hostname",
        "host_name",
        "started_at",
        "timestamp",
    }
)
_OPERATIONAL_DURATION_KEYS = frozenset(
    {"duration_seconds", "wall_clock_duration", "wall_clock_seconds"}
)
_RAW_TARGET_DERIVED_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "class_distribution",
        "class_support",
        "cooccurrence",
        "label",
        "labels",
        "projection",
        "projections",
        "supervision",
        "target",
        "target_availability",
        "target_rows",
        "targets",
        "theory",
    }
)
_RAW_TARGET_TOKENS = frozenset(
    {
        "annotation",
        "annotations",
        "annotated",
        "class",
        "classes",
        "cooccurrence",
        "gold",
        "label",
        "labeled",
        "labelled",
        "labels",
        "projection",
        "projections",
        "supervision",
        "supervised",
        "target",
        "targets",
        "theory",
    }
)
_RAW_TARGET_COMPOUND_ALIASES = frozenset(
    {
        "answer_key",
        "answer_keys",
        "answerkey",
        "answerkeys",
        "co_occurrence",
        "cooccurrence",
        "ground_truth",
        "ground_truths",
        "groundtruth",
        "groundtruths",
    }
)
_RAW_NEGATED_TARGET_FREE_ALIASES = frozenset(
    {
        "non_target_free",
        "not_target_free",
        "target_free_disabled",
        "target_free_false",
        "target_free_missing",
        "target_free_no",
        "target_free_unknown",
        "target_free_unproven",
        "target_not_free",
    }
)
_RAW_NEGATED_TARGET_INDEPENDENT_ALIASES = frozenset(
    {
        "non_target_independent",
        "not_target_independent",
        "target_independent_disabled",
        "target_independent_false",
        "target_independent_missing",
        "target_independent_no",
        "target_independent_unknown",
        "target_independent_unproven",
        "target_not_independent",
    }
)
_RAW_MANIFEST_TARGET_TOKENS = _RAW_TARGET_TOKENS
_TEST_SCOPE_TOKENS = frozenset({"test", "testing", "testset"})
_TEST_SCOPE_COMPOUND_ALIASES = frozenset(
    {
        "held_out",
        "held_outs",
        "heldout",
        "heldouts",
        "hold_out",
        "hold_outs",
        "holdout",
        "holdouts",
    }
)
_TEST_COMPACT_PREFIXES = frozenset(
    {
        "heldout",
        "heldouts",
        "holdout",
        "holdouts",
        "test",
        "testing",
        "tests",
        "testset",
        "testsets",
    }
)
_TEST_COMPACT_SUFFIXES = frozenset(
    {
        "access",
        "class",
        "classes",
        "cooccurrence",
        "coverage",
        "data",
        "dataset",
        "distribution",
        "descriptor",
        "example",
        "examples",
        "fold",
        "folds",
        "file",
        "filepath",
        "files",
        "item",
        "items",
        "label",
        "labels",
        "histogram",
        "manifest",
        "load",
        "loaded",
        "loader",
        "open",
        "opened",
        "partition",
        "path",
        "read",
        "reader",
        "record",
        "records",
        "row",
        "rows",
        "sample",
        "samples",
        "resolution",
        "resolver",
        "set",
        "sidecar",
        "split",
        "support",
        "target",
        "targets",
        "uri",
        "url",
        "use",
        "used",
    }
)
_TEST_DOMAIN_SEMANTIC_SUFFIXES = frozenset(
    {
        "class",
        "classes",
        "cooccurrence",
        "coverage",
        "data",
        "dataset",
        "distribution",
        "histogram",
        "label",
        "labels",
        "record",
        "records",
        "row",
        "rows",
        "support",
        "target",
        "targets",
    }
)
_RAW_TARGET_COMPACT_PREFIXES = frozenset(
    {
        "annotation",
        "annotations",
        "annotated",
        "class",
        "classes",
        "gold",
        "label",
        "labeled",
        "labelled",
        "labels",
        "projection",
        "projections",
        "supervision",
        "supervised",
        "target",
        "targets",
        "theory",
    }
)
_RAW_TARGET_COMPACT_SUFFIXES = frozenset(
    {
        "access",
        "annotation",
        "annotations",
        "count",
        "counts",
        "data",
        "dataset",
        "descriptor",
        "distribution",
        "file",
        "filepath",
        "files",
        "histogram",
        "label",
        "labels",
        "manifest",
        "matrix",
        "metadata",
        "item",
        "items",
        "record",
        "records",
        "load",
        "loaded",
        "loader",
        "open",
        "opened",
        "path",
        "read",
        "reader",
        "resolution",
        "resolver",
        "row",
        "rows",
        "sample",
        "samples",
        "set",
        "sets",
        "sidecar",
        "support",
        "target",
        "targets",
        "uri",
        "url",
        "use",
        "used",
        "value",
        "values",
    }
)
_TARGET_COMPACT_BRIDGES = frozenset(
    {
        "annotation",
        "annotations",
        "class",
        "classes",
        "gold",
        "label",
        "labels",
        "projection",
        "projections",
        "supervision",
        "target",
        "targets",
        "theory",
    }
)
_TEST_ACCESS_ACTIONS = frozenset(
    {
        "access",
        "descriptor",
        "load",
        "loaded",
        "loader",
        "open",
        "opened",
        "read",
        "reader",
        "resolution",
        "resolver",
        "use",
        "used",
    }
)
_NON_PRODUCTION_EVIDENCE_TOKENS = frozenset(
    {
        "bounded",
        "fixture",
        "fixtures",
        "not_executed",
        "replay",
        "synthetic",
    }
)
_NON_PRODUCTION_COMPACT_MARKERS = frozenset(
    {
        "boundedscan",
        "fixturescan",
        "manifestreplay",
        "notexecuted",
        "notrun",
        "syntheticfixture",
    }
)
_NON_PRODUCTION_COMPACT_PREFIXES = frozenset(
    {"bounded", "fixture", "fixtures", "replay", "synthetic"}
)
_NON_PRODUCTION_COMPACT_SUFFIXES = frozenset(
    {
        "adapter",
        "analysis",
        "audit",
        "data",
        "eda",
        "evidence",
        "manifest",
        "report",
        "result",
        "scan",
        "source",
    }
)
_PRODUCTION_ATTESTATION_FIELDS = frozenset(
    {
        "annotation_namespace",
        "code",
        "common_task_identity",
        "dialect",
        "identity",
        "input_manifests",
        "label_granularity",
        "mapping_registry",
        "metric_id",
        "name",
        "namespace",
        "producer_identity",
        "provenance",
        "reason_code",
        "repository_relative_path",
        "role",
        "row_id",
        "schema_name",
        "source_identity",
        "source_task_id",
        "vocabulary",
        "work_identity",
    }
)
_PRODUCTION_DOMAIN_FIELDS = frozenset(
    {"category", "detail", "message", "payload", "projected_value", "source_value"}
)
_OPERATIONAL_ATTESTATION_FIELDS = _PRODUCTION_ATTESTATION_FIELDS - {
    "input_manifests",
    "repository_relative_path"
}
_SOURCE_DOMAIN_VALUE_FIELDS = frozenset(
    {
        "categories",
        "category",
        "description",
        "detail",
        "display_name",
        "message",
        "name",
        "names",
        "native_value",
        "source_value",
        "source_values",
        "title",
        "titles",
    }
)
_COMMON_REPORT_KEYS = frozenset(
    {
        "envelope_schema_name",
        "envelope_schema_version",
        "schema_name",
        "schema_version",
        "report_kind",
        "corpus",
        "source_identity",
        "producer_identity",
        "repository_commit",
        "evidence_scope",
        "execution_mode",
        "completeness_status",
        "split_scope",
        "observation_units",
        "input_manifests",
        "semantic_payload",
        "invariants",
        "warnings",
        "unavailable_reasons",
        "operational_metadata",
        "semantic_fingerprint",
        "version_policy",
    }
)
_COMMON_SEMANTIC_KEYS = frozenset(
    {
        "availability",
        "categories",
        "class_distribution",
        "class_support",
        "common_task_identity",
        "cooccurrence",
        "count",
        "counts",
        "coverage",
        "empty_multilabel_available_count",
        "extension_contract_version",
        "extension_fingerprint",
        "extensions",
        "graph_evidence",
        "metric_id",
        "metrics",
        "mapping_registry",
        "mapping_state",
        "namespace",
        "native_state",
        "numeric",
        "projection_availability",
        "projections",
        "projected_value",
        "row_id",
        "rows",
        "tasks",
        "target_availability",
        "target_free",
        "target_independent",
        "test_lock",
        "work_identity",
    }
)
_COUNT_POPULATION_PLURALS = frozenset(
    {
        "assignments",
        "attempts",
        "bars",
        "beats",
        "buckets",
        "candidates",
        "classes",
        "collections",
        "components",
        "collisions",
        "corpora",
        "datasets",
        "edges",
        "entries",
        "errors",
        "events",
        "examples",
        "excerpts",
        "files",
        "incidences",
        "instruments",
        "items",
        "labels",
        "measures",
        "movements",
        "nodes",
        "notes",
        "observations",
        "occurrences",
        "onsets",
        "pairs",
        "parts",
        "pieces",
        "phrases",
        "presentations",
        "programs",
        "records",
        "rows",
        "samples",
        "sections",
        "segments",
        "sources",
        "spans",
        "targets",
        "tokens",
        "tracks",
        "updates",
        "versions",
        "vocabularies",
        "warnings",
        "works",
    }
)
_COUNT_POPULATION_COMPACT = _COUNT_POPULATION_PLURALS | frozenset(
    {
        "assignment",
        "attempt",
        "bar",
        "beat",
        "bucket",
        "candidate",
        "class",
        "collection",
        "component",
        "collision",
        "corpus",
        "dataset",
        "edge",
        "entry",
        "error",
        "event",
        "example",
        "excerpt",
        "file",
        "incidence",
        "instrument",
        "item",
        "label",
        "measure",
        "movement",
        "node",
        "note",
        "observation",
        "occurrence",
        "onset",
        "pair",
        "part",
        "piece",
        "phrase",
        "presentation",
        "program",
        "record",
        "row",
        "sample",
        "section",
        "segment",
        "source",
        "span",
        "target",
        "token",
        "track",
        "update",
        "version",
        "vocabulary",
        "warning",
        "work",
    }
)
_COUNT_CONTAINER_TOKENS = frozenset(
    {"buckets", "distribution", "frequencies", "frequency", "histogram", "support"}
)
_COUNT_AGGREGATE_COMPACT = frozenset(
    {
        "classcardinality",
        "datasetcardinality",
        "datasetsize",
        "labelcardinality",
        "populationcardinality",
        "populationtotal",
        "populationsize",
        "samplecardinality",
        "samplesize",
        "vocabularycardinality",
        "vocabularysize",
        "vocabcardinality",
        "vocabsize",
    }
)
_COUNT_AGGREGATE_ENTITIES = frozenset(
    {"collection", "dataset", "population", "recordset", "vocab", "vocabulary"}
)


def _enum(value: Any, enum_type: type[StrEnum], name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise EDAContractError(
            "eda.enum.invalid", f"{name} has unsupported value {value!r}"
        ) from exc


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EDAContractError(
            "eda.identity.invalid", f"{name} must be a non-empty stripped string"
        )
    if value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
        raise EDAContractError(
            "eda.identity.absolute_path",
            f"{name} must not be an absolute filesystem path",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EDAContractError(
            "eda.serialization.utf8_invalid",
            f"{name} must contain valid UTF-8 scalar text",
        ) from exc
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise EDAContractError(
            "eda.identity.control_character",
            f"{name} must not contain control or format characters",
        )
    return value


def _domain_text(value: object, name: str) -> str:
    """Validate opaque source/prose text without treating formatting as structure."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise EDAContractError(
            "eda.identity.invalid", f"{name} must be a non-empty stripped string"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EDAContractError(
            "eda.serialization.utf8_invalid",
            f"{name} must contain valid UTF-8 scalar text",
        ) from exc
    return value


def _version(value: object, name: str) -> str:
    value = _identifier(value, name)
    if _VERSION_RE.fullmatch(value) is None:
        raise EDAContractError(
            "eda.version.invalid", f"{name} must use MAJOR.MINOR.PATCH"
        )
    return value


def source_extension_namespace_is_valid(
    value: object,
    corpus: CorpusId,
) -> bool:
    """Return whether a namespace is normalized and owned by ``corpus``."""

    return (
        isinstance(value, str)
        and _EXTENSION_NAMESPACE_RE.fullmatch(value) is not None
        and value.startswith(f"{corpus.value}.")
    )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EDAContractError(
            "eda.fingerprint.invalid", f"{name} must be lowercase SHA-256"
        )
    return value


def _integer(value: object, name: str, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EDAContractError(
            "eda.count.invalid", f"{name} must be a non-negative integer"
        )
    return value


def _sorted_unique_strings(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise EDAContractError(
            "eda.identity.invalid", f"{name} must be a sequence of strings"
        )
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
        raise EDAContractError(
            "eda.identity.invalid", f"{name} must contain non-empty stripped strings"
        )
    for value in values:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EDAContractError(
                "eda.serialization.utf8_invalid",
                f"{name} must contain valid UTF-8 scalar text",
            ) from exc
        if any(
            unicodedata.category(character) in {"Cc", "Cf"}
            for character in value
        ):
            raise EDAContractError(
                "eda.identity.control_character",
                f"{name} must not contain control or format characters",
            )
    if len(values) != len(set(values)):
        raise EDAContractError(
            "eda.identity.duplicate", f"{name} must not contain duplicates"
        )
    return tuple(sorted(values))


def _tuple_collection(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise EDAContractError(
            "eda.collection.type_invalid",
            f"{name} must be a tuple or list, not a one-shot iterable",
        )
    return tuple(value)


def _jsonable(value: object, *, path: str = "$") -> JsonValue:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EDAContractError(
                "eda.serialization.utf8_invalid",
                "JSON strings must contain valid UTF-8 scalar text",
                path=path,
            ) from exc
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EDAContractError(
                "eda.serialization.non_finite",
                "NaN and Infinity are forbidden",
                path=path,
            )
        return 0.0 if value == 0.0 else value
    if is_dataclass(value) and not isinstance(value, type):
        if type(value) not in _SCHEMA_DATACLASS_TYPES:
            raise EDAContractError(
                "eda.serialization.schema_type_invalid",
                "only exact registered EDA schema dataclass types are serializable",
                path=path,
            )
        result: dict[str, JsonValue] = {}
        for item in fields(value):
            result[item.name] = _jsonable(
                getattr(value, item.name), path=f"{path}.{item.name}"
            )
        return result
    if isinstance(value, Mapping):
        result = {}
        if any(not isinstance(key, str) for key in value):
            raise EDAContractError(
                "eda.serialization.key_invalid",
                "JSON mapping keys must be strings",
                path=path,
            )
        for key in value:
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise EDAContractError(
                    "eda.serialization.utf8_invalid",
                    "JSON mapping keys must contain valid UTF-8 scalar text",
                    path=path,
                ) from exc
            if any(
                unicodedata.category(character) in {"Cc", "Cf"}
                for character in key
            ):
                raise EDAContractError(
                    "eda.serialization.key_invalid",
                    "JSON mapping keys must not contain control or format characters",
                    path=path,
                )
        for key in sorted(value):
            result[key] = _jsonable(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _jsonable(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise EDAContractError(
        "eda.serialization.type_invalid",
        f"unsupported semantic value type {type(value).__name__}",
        path=path,
    )


def _freeze_json(value: JsonValue) -> object:
    """Recursively freeze validated JSON so report fingerprints cannot go stale."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EDAContractError(
            "eda.serialization.object_invalid",
            "expected a JSON mapping",
            path=path,
        )
    encoded = _jsonable(value, path=path)
    if not isinstance(encoded, dict):  # pragma: no cover - guarded above.
        raise EDAContractError(
            "eda.serialization.object_invalid", "expected a JSON mapping", path=path
        )
    frozen = _freeze_json(encoded)
    assert isinstance(frozen, Mapping)
    return frozen


def _reject_keys(
    value: object,
    forbidden: frozenset[str],
    *,
    path: str = "$",
    ancestor_fields: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        ancestor_tokens = frozenset(
            token
            for field_name in ancestor_fields
            for token in field_name.split("_")
        )
        for key, item in value.items():
            normalized_key = _normalized_field_name(key)
            source_event_timestamp = (
                forbidden == _OPERATIONAL_KEYS
                and normalized_key == "timestamp"
                and "event" in ancestor_tokens
                and not ancestor_tokens
                & {"execution", "processing", "run", "runtime", "wall", "wallclock"}
            )
            if key in forbidden and not source_event_timestamp:
                raise EDAContractError(
                    "eda.semantic_field.forbidden",
                    f"field {key!r} is forbidden in this semantic scope",
                    path=f"{path}.{key}",
                )
            _reject_keys(
                item,
                forbidden,
                path=f"{path}.{key}",
                ancestor_fields=(*ancestor_fields, normalized_key),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_keys(
                item,
                forbidden,
                path=f"{path}[{index}]",
                ancestor_fields=ancestor_fields,
            )


def _normalized_field_name(key: str) -> str:
    key = "".join(
        character
        for character in key
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    key = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return "_".join(
        token for token in re.split(r"[^a-z0-9]+", key.lower()) if token
    )


def _normalized_key_tokens(key: str) -> frozenset[str]:
    normalized = _normalized_field_name(key)
    return frozenset(normalized.split("_")) if normalized else frozenset()


def _compact_prefixed_semantics(
    token: str,
    *,
    prefixes: frozenset[str],
    suffixes: frozenset[str],
) -> bool:
    """Recognize separator-free semantic compounds without substring guessing."""

    def _matches_suffix(remainder: str) -> bool:
        if remainder in suffixes:
            return True
        return any(
            remainder.startswith(bridge)
            and _matches_suffix(remainder[len(bridge) :])
            for bridge in sorted(_TARGET_COMPACT_BRIDGES, key=len, reverse=True)
        )

    for prefix in sorted(prefixes, key=len, reverse=True):
        if not token.startswith(prefix):
            continue
        if _matches_suffix(token[len(prefix) :]):
            return True
    return False


def _reject_text_tokens(
    value: str | None,
    forbidden: frozenset[str],
    *,
    category: str,
    label: str,
    path: str = "$",
    forbidden_compounds: frozenset[str] = frozenset(),
) -> None:
    if forbidden in {_RAW_TARGET_TOKENS, _RAW_MANIFEST_TARGET_TOKENS}:
        forbidden_compounds = (
            forbidden_compounds | _RAW_TARGET_COMPOUND_ALIASES
        )
    if forbidden == _TEST_SCOPE_TOKENS:
        forbidden_compounds = forbidden_compounds | _TEST_SCOPE_COMPOUND_ALIASES
    normalized = None if value is None else _normalized_field_name(value)
    disallowed_target_free = False
    plural_tests_forbidden = False
    if normalized is not None and forbidden in {
        _RAW_TARGET_TOKENS,
        _RAW_MANIFEST_TARGET_TOKENS,
    }:
        original_padded = f"_{normalized}_"
        original_compact = normalized.replace("_", "")
        disallowed_target_free = any(
            f"_{alias}_" in original_padded
            or alias.replace("_", "") in original_compact
            for alias in (
                _RAW_NEGATED_TARGET_FREE_ALIASES
                | _RAW_NEGATED_TARGET_INDEPENDENT_ALIASES
            )
        )
        normalized = (
            f"_{normalized}_"
            .replace("_hook_theory_", "_hooktheory_")
            .replace("_un_annotated_", "_unannotated_")
            .replace("_un_labeled_", "_unlabeled_")
            .replace("_un_labelled_", "_unlabelled_")
            .replace("_un_supervised_", "_unsupervised_")
            .replace("_self_supervised_", "_selfsupervised_")
            .replace("_target_free_", "_")
            .replace("_target_independent_", "_")
            .replace("_raw_projections_", "_")
            .replace("_raw_projection_", "_")
            .strip("_")
        )
        normalized = "_".join(
            token.replace("targetindependent", "")
            .replace("targetfree", "")
            .replace("rawprojections", "")
            .replace("rawprojection", "")
            for token in normalized.split("_")
            if token
        )
        normalized = (
            f"_{normalized}_"
            .replace("_pitch_classes_", "_pitchclasses_")
            .replace("_pitch_class_", "_pitchclass_")
            .strip("_")
        )
    if normalized is not None and forbidden == _TEST_SCOPE_TOKENS:
        normalized_tokens = frozenset(normalized.split("_"))
        plural_tests_forbidden = bool(normalized_tokens & {"tests", "testsets"})
    padded = "" if normalized is None else f"_{normalized}_"
    compact_tokens = () if normalized is None else tuple(normalized.split("_"))
    compact_forbidden = False
    if forbidden in {_RAW_TARGET_TOKENS, _RAW_MANIFEST_TARGET_TOKENS}:
        compact_forbidden = any(
            _compact_prefixed_semantics(
                token,
                prefixes=_RAW_TARGET_COMPACT_PREFIXES,
                suffixes=_RAW_TARGET_COMPACT_SUFFIXES,
            )
            or any(alias.replace("_", "") in token for alias in forbidden_compounds)
            for token in compact_tokens
        )
    elif forbidden == _TEST_SCOPE_TOKENS:
        compact_forbidden = any(
            _compact_prefixed_semantics(
                token,
                prefixes=_TEST_COMPACT_PREFIXES,
                suffixes=_TEST_COMPACT_SUFFIXES,
            )
            for token in compact_tokens
        )
    if value is not None and (
        (frozenset(normalized.split("_")) if normalized else frozenset()) & forbidden
        or any(f"_{alias}_" in padded for alias in forbidden_compounds)
        or compact_forbidden
        or disallowed_target_free
        or plural_tests_forbidden
    ):
        raise EDAContractError(
            category,
            f"{label} contains forbidden semantic tokens",
            path=path,
        )


def _looks_like_absolute_path(
    value: str,
    *,
    scan_embedded: bool = True,
) -> bool:
    embedded_windows_or_home = (
        re.search(r"(?:^|[^A-Za-z0-9_])(?:~[/\\]|[A-Za-z]:[\\/])", value)
        is not None
        or re.search(
            r"(?:^|[^A-Za-z0-9\\])\\+[^\\/\s]+",
            value,
        )
        is not None
        or re.search(
            r"(?:^|[^A-Za-z0-9_\\])\\{2,}[^\\/\s]+[\\/]",
            value,
        )
        is not None
    )
    embedded_posix = (
        re.search(
            r"(?:^|[^A-Za-z0-9:/.])/{2,}[^\s\]\[(){}<>]+",
            value,
        )
        is not None
    )
    for match in re.finditer(
        r"(?:^|[^A-Za-z0-9/.])(/(?!/)[^\s\]\[(){}<>]+)",
        value,
    ):
        components = tuple(
            component.rstrip(".,;:!?")
            for component in match.group(1).split("/")[1:]
            if component
        )
        known_root = bool(components) and components[0].lower() in {
            "data",
            "etc",
            "home",
            "mnt",
            "opt",
            "srv",
            "tmp",
            "users",
            "usr",
            "var",
        }
        non_harmonic_absolute_token = bool(components) and not all(
            re.fullmatch(r"[ivx]+", component, re.IGNORECASE)
            for component in components
        )
        if known_root or non_harmonic_absolute_token:
            embedded_posix = True
            break
    return (
        value.startswith(('/', '\\', '~/'))
        or _WINDOWS_ABSOLUTE_RE.match(value) is not None
        or embedded_windows_or_home
        or embedded_posix
        or (
            scan_embedded
            and re.search(r"(?:^|[^A-Za-z0-9/.])/(?!/)\S+", value)
            is not None
        )
        or "file://" in value.lower()
    )


def _compact_operational_alias(token: str) -> bool:
    if token in {
        "absolutepath",
        "durationseconds",
        "elapsedtime",
        "executiontime",
        "finishedat",
        "hostname",
        "machinename",
        "pid",
        "processid",
        "processingseconds",
        "processingtime",
        "runduration",
        "rundurationseconds",
        "runtimestamp",
        "runtime",
        "startedat",
        "wallclock",
        "wallclockduration",
        "wallclockseconds",
        "wallclocktime",
    }:
        return True
    if token.endswith("s") and _compact_operational_alias(token[:-1]):
        return True
    timed_suffixes = {
        "duration",
        "durationhours",
        "durationmilliseconds",
        "durationminutes",
        "durationms",
        "durationseconds",
        "elapsed",
        "elapsedhours",
        "elapsedmilliseconds",
        "elapsedminutes",
        "elapsedms",
        "elapsedseconds",
        "hours",
        "milliseconds",
        "minutes",
        "ms",
        "seconds",
        "time",
    }
    if any(
        token == f"{prefix}{suffix}"
        for prefix in ("elapsed", "execution", "process", "processing", "runtime", "wallclock")
        for suffix in timed_suffixes
    ):
        return True
    if any(
        token == f"run{suffix}"
        for suffix in (
            *timed_suffixes,
            "endtime",
            "finishedat",
            "startedat",
            "starttime",
            "timestamp",
        )
    ):
        return True
    return any(
        token == f"{prefix}{suffix}"
        for prefix in ("host", "machine")
        for suffix in ("host", "hostname", "id", "identifier", "name")
    )


def _operational_alias(tokens: frozenset[str]) -> bool:
    if {"runtime", "complexity"} <= tokens and not tokens & {
        "duration",
        "elapsed",
        "hours",
        "milliseconds",
        "minutes",
        "ms",
        "seconds",
        "time",
        "wallclock",
    }:
        return False
    if any(_compact_operational_alias(token) for token in tokens):
        return True
    plural_map = {
        "clocks": "clock",
        "durations": "duration",
        "identifiers": "identifier",
        "ids": "id",
        "names": "name",
        "times": "time",
        "timestamps": "timestamp",
    }
    operational_qualifiers = {
        "execution",
        "host",
        "machine",
        "process",
        "processing",
        "run",
        "runtime",
        "wall",
        "wallclock",
    }
    if len(tokens) > 1 and tokens & operational_qualifiers:
        singularized = frozenset(plural_map.get(token, token) for token in tokens)
        if singularized != tokens and _operational_alias(singularized):
            return True
    if tokens in {
        frozenset({"absolute", "path"}),
        frozenset({"cwd"}),
        frozenset({"duration", "seconds"}),
    }:
        return True
    if tokens & {"hostname", "machinename", "pid", "processid"} or tokens == {
        "host"
    }:
        return True
    if "host" in tokens and "name" in tokens:
        return True
    if "host" in tokens and "identifier" in tokens:
        return True
    if "machine" in tokens and tokens & {
        "name",
        "host",
        "id",
        "identifier",
        "node",
    }:
        return True
    if "process" in tokens and tokens & {"id", "identifier"}:
        return True
    if tokens == {"timestamp"} or (
        "timestamp" in tokens and tokens & {"run", "wall", "execution"}
    ):
        return True
    if tokens == {"time", "stamp"} or (
        {"time", "stamp"}.issubset(tokens)
        and tokens & {"run", "wall", "execution"}
    ):
        return True
    if tokens & {"elapsed", "runtime"}:
        return True
    if tokens & {"execution", "processing", "run"} and tokens & {
        "duration",
        "milliseconds",
        "ms",
        "seconds",
        "time",
    }:
        return True
    if "wall" in tokens and tokens & {"clock", "duration", "seconds", "time"}:
        return True
    if "wallclock" in tokens:
        return True
    if tokens & {"started", "finished"} and "at" in tokens:
        return True
    return False


def _reject_operational_semantics(
    value: object,
    *,
    path: str = "$",
    ancestor_tokens: frozenset[str] = frozenset(),
    scan_embedded_paths: bool = True,
) -> None:
    """Keep machine/run-local aliases and absolute paths out of semantics."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            tokens = _normalized_key_tokens(key)
            path_tokens = ancestor_tokens | tokens
            typed_manifest_path = (
                key == "repository_relative_path"
                and ".input_manifests[" in path
                and path.startswith("$.envelope.")
            )
            source_event_timestamp = (
                key == "timestamp"
                and "event" in ancestor_tokens
                and not ancestor_tokens
                & {"execution", "processing", "run", "runtime", "wall", "wallclock"}
            )
            if _looks_like_absolute_path(key, scan_embedded=scan_embedded_paths):
                raise EDAContractError(
                    "eda.operational_metadata.absolute_path_forbidden",
                    "absolute filesystem paths are forbidden in semantic evidence",
                    path=f"{path}.{key}",
                )
            if not typed_manifest_path and not source_event_timestamp and (
                _operational_alias(tokens) or _operational_alias(path_tokens)
            ):
                raise EDAContractError(
                    "eda.operational_metadata.semantic_alias_forbidden",
                    f"operational field alias {key!r} belongs in operational_metadata",
                    path=f"{path}.{key}",
                )
            _reject_operational_semantics(
                item,
                path=f"{path}.{key}",
                ancestor_tokens=path_tokens,
                scan_embedded_paths=(
                    scan_embedded_paths
                    and _normalized_field_name(key)
                    not in (_SOURCE_DOMAIN_VALUE_FIELDS - {"detail", "message"})
                ),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_operational_semantics(
                item,
                path=f"{path}[{index}]",
                ancestor_tokens=ancestor_tokens,
                scan_embedded_paths=scan_embedded_paths,
            )
    elif isinstance(value, str) and _looks_like_absolute_path(
        value,
        scan_embedded=scan_embedded_paths,
    ):
        raise EDAContractError(
            "eda.operational_metadata.absolute_path_forbidden",
            "absolute filesystem paths are forbidden in semantic evidence",
            path=path,
        )


def _reject_operational_attestation_value(value: object, *, path: str) -> None:
    """Reject run-local aliases inside typed identities and provenance."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_operational_attestation_value(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_operational_attestation_value(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        tokens = _normalized_key_tokens(value)
        if _operational_alias(tokens) or "host" in tokens:
            raise EDAContractError(
                "eda.operational_metadata.semantic_alias_forbidden",
                "typed semantic identity/provenance contains an operational alias",
                path=path,
            )


def _reject_operational_attestation_markers(
    value: object, *, path: str = "$"
) -> None:
    """Inspect typed channels while preserving arbitrary source-domain strings."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _PRODUCTION_DOMAIN_FIELDS:
                continue
            if key in _OPERATIONAL_ATTESTATION_FIELDS:
                _reject_operational_attestation_value(item, path=item_path)
            else:
                _reject_operational_attestation_markers(item, path=item_path)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_operational_attestation_markers(
                item, path=f"{path}[{index}]"
            )


def _raw_domain_literal_has_explicit_target_semantics(value: str) -> bool:
    """Distinguish opaque source labels/titles from target-bearing phrases."""

    normalized = _normalized_field_name(value)
    padded = f"_{normalized}_"
    compact = normalized.replace("_", "")
    negated_attestations = (
        _RAW_NEGATED_TARGET_FREE_ALIASES
        | _RAW_NEGATED_TARGET_INDEPENDENT_ALIASES
    )
    if any(
        f"_{alias}_" in padded or alias.replace("_", "") in compact
        for alias in negated_attestations
    ):
        return True
    normalized = (
        padded.replace("_hook_theory_", "_hooktheory_")
        .replace("_un_annotated_", "_unannotated_")
        .replace("_un_labeled_", "_unlabeled_")
        .replace("_un_labelled_", "_unlabelled_")
        .replace("_un_supervised_", "_unsupervised_")
        .replace("_self_supervised_", "_selfsupervised_")
        .replace("_target_free_", "_")
        .replace("_target_independent_", "_")
        .replace("_raw_projections_", "_")
        .replace("_raw_projection_", "_")
        .replace("_pitch_classes_", "_pitchclasses_")
        .replace("_pitch_class_", "_pitchclass_")
        .strip("_")
    )
    transformed_tokens = tuple(
        token.replace("targetindependent", "")
        .replace("targetfree", "")
        .replace("rawprojections", "")
        .replace("rawprojection", "")
        for token in normalized.split("_")
        if token
    )
    normalized = "_".join(token for token in transformed_tokens if token)
    padded = f"_{normalized}_"
    compact_tokens = tuple(normalized.split("_")) if normalized else ()
    record_company_label_indexes = {
        index
        for index, token in enumerate(compact_tokens)
        if token == "label"
        and (
            (index > 0 and compact_tokens[index - 1] in {"record", "records"})
            or (
                index + 1 < len(compact_tokens)
                and compact_tokens[index + 1] in {"record", "records"}
            )
        )
    }
    if record_company_label_indexes:
        compact_tokens = tuple(
            token
            for index, token in enumerate(compact_tokens)
            if index not in record_company_label_indexes
        )
    tokens = frozenset(compact_tokens)
    if any(
        f"_{alias}_" in padded or alias.replace("_", "") in normalized.replace("_", "")
        for alias in _RAW_TARGET_COMPOUND_ALIASES
    ):
        return True
    if any(
        _compact_prefixed_semantics(
            token,
            prefixes=_RAW_TARGET_COMPACT_PREFIXES,
            suffixes=_RAW_TARGET_COMPACT_SUFFIXES,
        )
        for token in compact_tokens
    ):
        return True
    target_tokens = tokens & _RAW_TARGET_TOKENS
    explicit_context = {
        "data",
        "dataset",
        "distribution",
        "manifest",
        "matrix",
        "metadata",
        "row",
        "rows",
        "set",
        "sets",
        "sidecar",
        "support",
        "value",
        "values",
    }
    return len(target_tokens) > 1 or bool(target_tokens and tokens & explicit_context)


def _reject_raw_target_fields(
    value: object,
    *,
    path: str = "$",
    domain_literal: bool = False,
    ancestor_fields: tuple[str, ...] = (),
) -> None:
    """Reject target/supervision semantics from source-owned raw extensions."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_field_name(key)
            field_path = "_".join((*ancestor_fields, normalized_key))
            source_credit_field = normalized_key in {
                "publisher_label",
                "record_label",
            }
            if not source_credit_field:
                _reject_text_tokens(
                    field_path,
                    _RAW_TARGET_TOKENS,
                    forbidden_compounds=_RAW_TARGET_COMPOUND_ALIASES,
                    category="eda.raw.target_field_forbidden",
                    label=f"target-derived raw extension field {key!r}",
                    path=f"{path}.{key}",
                )
            _reject_raw_target_fields(
                item,
                path=f"{path}.{key}",
                domain_literal=(
                    not isinstance(item, Mapping)
                    and (
                        normalized_key in _SOURCE_DOMAIN_VALUE_FIELDS
                        or source_credit_field
                    )
                ),
                ancestor_fields=(*ancestor_fields, normalized_key),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_raw_target_fields(
                item,
                path=f"{path}[{index}]",
                domain_literal=domain_literal and not isinstance(item, Mapping),
                ancestor_fields=ancestor_fields,
            )
    elif isinstance(value, str):
        if domain_literal:
            if _raw_domain_literal_has_explicit_target_semantics(value):
                raise EDAContractError(
                    "eda.raw.target_field_forbidden",
                    "source-domain value encodes explicit target semantics",
                    path=path,
                )
            return
        _reject_text_tokens(
            value,
            _RAW_TARGET_TOKENS,
            forbidden_compounds=_RAW_TARGET_COMPOUND_ALIASES,
            category="eda.raw.target_field_forbidden",
            label="target-derived raw extension value",
            path=path,
        )


def _test_domain_literal_has_explicit_semantics(value: str) -> bool:
    normalized = _normalized_field_name(value)
    tokens = frozenset(normalized.split("_")) if normalized else frozenset()
    padded = f"_{normalized}_"
    scope_present = bool(tokens & _TEST_SCOPE_TOKENS) or "tests" in tokens or any(
        f"_{alias}_" in padded for alias in _TEST_SCOPE_COMPOUND_ALIASES
    )
    explicit_context = _TEST_DOMAIN_SEMANTIC_SUFFIXES | _TARGET_COMPACT_BRIDGES
    if scope_present and bool(tokens & explicit_context):
        return True
    return any(
        _compact_prefixed_semantics(
            token,
            prefixes=_TEST_COMPACT_PREFIXES,
            suffixes=_TEST_DOMAIN_SEMANTIC_SUFFIXES,
        )
        for token in tokens
    )


def _test_lock_provenance_claims_access(value: str) -> bool:
    if _test_domain_literal_has_explicit_semantics(value):
        return True
    normalized = _normalized_field_name(value)
    tokens = frozenset(normalized.split("_")) if normalized else frozenset()
    padded = f"_{normalized}_"
    scope_present = bool(tokens & _TEST_SCOPE_TOKENS) or "tests" in tokens or any(
        f"_{alias}_" in padded for alias in _TEST_SCOPE_COMPOUND_ALIASES
    )
    if scope_present and bool(tokens & _TEST_ACCESS_ACTIONS):
        return True
    return any(
        _compact_prefixed_semantics(
            token,
            prefixes=_TEST_COMPACT_PREFIXES,
            suffixes=_TEST_ACCESS_ACTIONS,
        )
        for token in tokens
    )


def _reject_test_supervision_fields(
    value: object,
    *,
    path: str = "$",
    domain_literal: bool = False,
    ancestor_fields: tuple[str, ...] = (),
) -> None:
    """Reject any TEST-scoped semantic payload inside supervision extensions."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = _normalized_field_name(key)
            field_path = "_".join((*ancestor_fields, normalized_key))
            _reject_text_tokens(
                field_path,
                _TEST_SCOPE_TOKENS,
                category="eda.test_lock.extension_test_field_forbidden",
                label=f"extension TEST field {key!r}",
                path=f"{path}.{key}",
            )
            _reject_test_supervision_fields(
                item,
                path=f"{path}.{key}",
                domain_literal=(
                    not isinstance(item, Mapping)
                    and normalized_key in _SOURCE_DOMAIN_VALUE_FIELDS
                ),
                ancestor_fields=(*ancestor_fields, normalized_key),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_test_supervision_fields(
                item,
                path=f"{path}[{index}]",
                domain_literal=domain_literal and not isinstance(item, Mapping),
                ancestor_fields=ancestor_fields,
            )
    elif isinstance(value, str):
        if domain_literal:
            if _test_domain_literal_has_explicit_semantics(value):
                raise EDAContractError(
                    "eda.test_lock.extension_test_value_forbidden",
                    "source-domain value encodes explicit TEST semantics",
                    path=path,
                )
            return
        _reject_text_tokens(
            value,
            _TEST_SCOPE_TOKENS,
            category="eda.test_lock.extension_test_value_forbidden",
            label="supervision extension payload TEST selector/value",
            path=path,
        )


def _reject_nonproduction_attestation(value: object, *, path: str) -> None:
    """Reject non-production markers inside one typed attestation channel."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonproduction_attestation(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_nonproduction_attestation(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        normalized = _normalized_field_name(value)
        compact = normalized.replace("_", "")
        components = tuple(normalized.split("_"))
        compact_marker = any(
            marker in compact for marker in _NON_PRODUCTION_COMPACT_MARKERS
        ) or any(
            re.fullmatch(f"{prefix}{suffix}(?:v[0-9]+)?", component) is not None
            for component in components
            for prefix in _NON_PRODUCTION_COMPACT_PREFIXES
            for suffix in _NON_PRODUCTION_COMPACT_SUFFIXES
        )
        if compact_marker:
            raise EDAContractError(
                "eda.evidence.fixture_as_production",
                "production evidence attestation contains a non-production marker",
                path=path,
            )
        _reject_text_tokens(
            value,
            _NON_PRODUCTION_EVIDENCE_TOKENS,
            category="eda.evidence.fixture_as_production",
            label="production evidence attestation",
            path=path,
        )


def _reject_nonproduction_markers(value: object, *, path: str = "$") -> None:
    """Prevent relabeling evidence as production without policing domain values."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in _PRODUCTION_DOMAIN_FIELDS:
                continue
            if key in _PRODUCTION_ATTESTATION_FIELDS:
                _reject_nonproduction_attestation(item, path=item_path)
            else:
                _reject_nonproduction_markers(item, path=item_path)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_nonproduction_markers(item, path=f"{path}[{index}]")


def _validate_operational_metadata(value: Mapping[str, JsonValue]) -> None:
    unknown = set(value) - _OPERATIONAL_KEYS
    if unknown:
        raise EDAContractError(
            "eda.operational_metadata.field_forbidden",
            f"only frozen operational keys may be excluded from semantics: {sorted(unknown)!r}",
            path="$.operational_metadata",
        )
    for key, item in value.items():
        if key in _OPERATIONAL_STRING_KEYS:
            if not isinstance(item, str) or not item:
                raise EDAContractError(
                    "eda.operational_metadata.value_invalid",
                    f"{key} must be a non-empty string",
                    path=f"$.operational_metadata.{key}",
                )
        elif key == "pid":
            _integer(item, "operational pid")
        elif key in _OPERATIONAL_DURATION_KEYS:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item < 0
            ):
                raise EDAContractError(
                    "eda.operational_metadata.value_invalid",
                    f"{key} must be a finite non-negative number",
                    path=f"$.operational_metadata.{key}",
                )


def _contains_integer_leaf(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, Mapping):
        return any(_contains_integer_leaf(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_integer_leaf(item) for item in value)
    return False


def _contains_numeric_leaf(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_numeric_leaf(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_numeric_leaf(item) for item in value)
    return False


def _is_unit_interval_numeric_tree(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value) and 0 <= value <= 1
    if isinstance(value, Mapping):
        return bool(value) and all(
            _is_unit_interval_numeric_tree(item) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_is_unit_interval_numeric_tree(item) for item in value)
    return False


def _is_nonnegative_integer_tree(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, Mapping):
        return bool(value) and all(
            _is_nonnegative_integer_tree(item) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return bool(value) and all(
            _is_nonnegative_integer_tree(item) for item in value
        )
    return False


def _is_measurement_summary(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if {
        "measurement_unit",
        "minimum",
        "maximum",
        "mean",
    }.issubset(value):
        return True
    summary_fields = {
        "max",
        "maximum",
        "mean",
        "median",
        "min",
        "minimum",
        "p25",
        "p50",
        "p75",
        "q1",
        "q3",
        "std",
        "stddev",
        "variance",
    }
    return all(_normalized_field_name(key) in summary_fields for key in value)


def _is_exact_ratio_mapping(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = set(value)
    if "num" in keys and "numerator" in keys:
        return False
    numerator_key = "numerator" if "numerator" in keys else "num"
    if numerator_key not in keys or "denominator" not in keys:
        return False
    numerator = value[numerator_key]
    denominator = value["denominator"]
    scalar_value = value.get("value")
    return (
        isinstance(numerator, int)
        and not isinstance(numerator, bool)
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
        and denominator != 0
        and (
            "value" not in value
            or (
                isinstance(scalar_value, (int, float))
                and not isinstance(scalar_value, bool)
                and (not isinstance(scalar_value, float) or math.isfinite(scalar_value))
            )
        )
    )


def _looks_like_untyped_count_field(key: str, value: object) -> bool:
    normalized = _normalized_field_name(key)
    ordered_tokens = tuple(normalized.split("_")) if normalized else ()
    tokens = frozenset(ordered_tokens)
    compact = normalized.replace("_", "")
    physical_size_units = {
        "b",
        "byte",
        "bytes",
        "gb",
        "gib",
        "kb",
        "kib",
        "mb",
        "mib",
        "tb",
        "tib",
    }
    if "size" in tokens and tokens & physical_size_units:
        return False
    if (
        "size" in tokens
        and isinstance(value, Mapping)
        and set(value) == {"unit", "value"}
        and isinstance(value["unit"], str)
        and _normalized_field_name(value["unit"]) in physical_size_units
        and isinstance(value["value"], (int, float))
        and not isinstance(value["value"], bool)
        and (
            not isinstance(value["value"], float)
            or math.isfinite(value["value"])
        )
    ):
        return False
    if tokens & {"count", "counts", "denominator"}:
        return True
    if (
        compact.endswith(("count", "counts"))
        and compact not in {"account", "discount"}
    ):
        return True
    if compact in _COUNT_AGGREGATE_COMPACT:
        return True
    if any(
        compact in {
            f"{population}count",
            f"{population}counts",
            f"count{population}",
            f"{population}total",
        }
        for population in _COUNT_POPULATION_COMPACT
    ):
        return True
    for prefix in ("n", "num", "numberof", "total"):
        if compact.startswith(prefix) and compact[len(prefix) :] in (
            _COUNT_POPULATION_COMPACT
        ):
            return True
    numeric_value = isinstance(value, (int, float)) and not isinstance(value, bool)
    if numeric_value and tokens & {
        "cardinality",
        "multiplicity",
        "population",
        "quantity",
        "tally",
    }:
        return True
    integer_value = isinstance(value, int) and not isinstance(value, bool)
    if integer_value and normalized in {
        "ambiguous",
        "available",
        "coarsened",
        "exact",
        "invalid",
        "masked",
        "missing",
        "not_applicable",
        "not_computed",
        "observed",
        "locked",
        "unsupported",
        "unknown",
    }:
        return True
    if numeric_value and compact == "nobs":
        return True
    if numeric_value and "total" in tokens and tokens & _COUNT_POPULATION_COMPACT:
        return True
    if numeric_value and tokens & _COUNT_AGGREGATE_ENTITIES and tokens & {
        "length",
        "size",
        "total",
    }:
        return True
    if numeric_value and any(
        compact in {
            f"{entity}length",
            f"{entity}total",
            f"sizeof{entity}",
            f"total{entity}",
        }
        for entity in _COUNT_AGGREGATE_ENTITIES
    ):
        return True
    if numeric_value and compact.endswith("size") and any(
        population in compact[: -len("size")]
        for population in (
            "collection",
            "dataset",
            "entries",
            "entry",
            "items",
            "population",
            "records",
            "recordset",
            "samples",
            "vocab",
            "vocabulary",
        )
    ):
        return True
    if numeric_value and (
        (len(ordered_tokens) > 1 and ordered_tokens[0] == "n")
        or compact.startswith("num")
        or compact.startswith("numberof")
    ):
        return True
    if normalized in {"n", "num", "total"}:
        return True
    ordinal_suffix = (
        len(ordered_tokens) > 1
        and ordered_tokens[-1] in {"num", "number"}
    )
    if not ordinal_suffix and tokens & {"n", "num", "number", "total"} and (
        tokens & _COUNT_POPULATION_PLURALS
    ):
        return True
    if tokens & _COUNT_POPULATION_PLURALS and isinstance(value, (int, float)) and (
        not isinstance(value, bool)
    ):
        return True
    if numeric_value and tokens & {"frequency", "frequencies"} and not tokens & {
        "audio",
        "clock",
        "hertz",
        "hz",
        "sampling",
    }:
        return True
    count_container = (
        bool(tokens & _COUNT_CONTAINER_TOKENS)
        and isinstance(value, (Mapping, tuple, list))
        and not _is_measurement_summary(value)
        and not tokens & {"audio", "hertz", "hz", "midi", "note", "sampling"}
    )
    if count_container:
        if tokens & {"confidence", "normalized", "probabilities", "probability"}:
            return (
                _contains_numeric_leaf(value)
                and not _is_unit_interval_numeric_tree(value)
            )
        return _contains_integer_leaf(value)
    safe_integer_mapping_tokens = {
        "audio",
        "clock",
        "code",
        "codes",
        "hertz",
        "hz",
        "id",
        "ids",
        "identifier",
        "identifiers",
        "index",
        "indices",
        "midi",
        "note",
        "number",
        "ordinal",
        "pitch",
        "process",
        "program",
        "sampling",
        "signature",
    }
    return (
        isinstance(value, Mapping)
        and _is_nonnegative_integer_tree(value)
        and not _is_measurement_summary(value)
        and not _is_exact_ratio_mapping(value)
        and not tokens & safe_integer_mapping_tokens
    )


def _reject_untyped_count_fields(value: object, *, path: str = "$") -> None:
    if _is_exact_ratio_mapping(value):
        assert isinstance(value, Mapping)
        for key, item in value.items():
            if key not in {"denominator", "num", "numerator", "value"}:
                if _looks_like_untyped_count_field(key, item):
                    raise EDAContractError(
                        "eda.extension.untyped_count",
                        "extension counts must use UnitCount rather than bare payload fields",
                        path=f"{path}.{key}",
                    )
                _reject_untyped_count_fields(item, path=f"{path}.{key}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _looks_like_untyped_count_field(key, item):
                raise EDAContractError(
                    "eda.extension.untyped_count",
                    "extension counts must use UnitCount rather than bare payload fields",
                    path=f"{path}.{key}",
                )
            _reject_untyped_count_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_untyped_count_fields(item, path=f"{path}[{index}]")


def _reject_extension_common_fields(value: object, *, path: str = "$") -> None:
    """Prevent a namespaced extension from shadowing any shared schema field."""

    if isinstance(value, Mapping):
        normalized_keys = {_normalized_field_name(key) for key in value}
        common_partitions = (
            {"available", "masked", "missing", "unsupported"},
            {
                "ambiguous",
                "coarsened",
                "exact",
                "invalid",
                "masked",
                "missing",
                "unsupported",
            },
        )
        if any(signature <= normalized_keys for signature in common_partitions):
            raise EDAContractError(
                "eda.extension.common_field_collision",
                "extension mapping collides with a shared availability partition",
                path=path,
            )
        for key, item in value.items():
            if _is_extension_common_field(key):
                raise EDAContractError(
                    "eda.extension.common_field_collision",
                    f"extension field {key!r} collides with the shared EDA schema",
                    path=f"{path}.{key}",
                )
            _reject_extension_common_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_extension_common_fields(item, path=f"{path}[{index}]")


def _is_extension_common_field(value: str) -> bool:
    reserved = (
        _COMMON_REPORT_KEYS
        | _COMMON_SEMANTIC_KEYS
        | frozenset(RAW_METRIC_CATALOG)
    )
    normalized = _normalized_field_name(value)
    normalized_reserved = {_normalized_field_name(item) for item in reserved}
    explicit_shadow_aliases = {
        alias
        for item in normalized_reserved
        for suffix in ("override", "replacement", "shadow")
        for alias in (f"{item}_{suffix}", f"{suffix}_{item}")
    }
    raw_metric_bases = {
        metric[: -len("_distribution")]
        if metric.endswith("_distribution")
        else metric
        for metric in (_normalized_field_name(item) for item in RAW_METRIC_CATALOG)
    }
    raw_metric_aliases = {
        alias
        for metric in raw_metric_bases
        for alias in (
            f"common_{metric}",
            f"raw_{metric}",
            f"{metric}_distribution",
            f"{metric}_statistics",
            f"{metric}_stats",
            f"{metric}_summary",
        )
    }
    compact = normalized.replace("_", "")
    attestation_alias = re.search(
        r"(?:^|_)target_(?:free|independent)(?:_|$)", normalized
    ) is not None or any(
        compact == f"{prefix}{base}{suffix}"
        for prefix in ("", "extension", "is", "raw", "report", "source")
        for base in ("targetfree", "targetindependent")
        for suffix in (
            "",
            "claim",
            "disabled",
            "enabled",
            "false",
            "flag",
            "missing",
            "no",
            "state",
            "status",
            "true",
            "unknown",
            "unproven",
            "value",
            "yes",
        )
    )
    return (
        attestation_alias
        or normalized in normalized_reserved
        or normalized in explicit_shadow_aliases
        or normalized in raw_metric_aliases
        or compact
        in {
            item.replace("_", "")
            for item in (
                normalized_reserved | explicit_shadow_aliases | raw_metric_aliases
            )
        }
    )


def _relative_path(value: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and any(
        unicodedata.category(character) in {"Cc", "Cf"} for character in value
    ):
        raise EDAContractError(
            "eda.manifest.path_invalid",
            "manifest paths must be normalized repository-relative POSIX paths",
        )
    value = _identifier(value, "repository_relative_path")
    path = PurePosixPath(value)
    if (
        value == "."
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value != path.as_posix()
    ):
        raise EDAContractError(
            "eda.manifest.path_invalid",
            "manifest paths must be normalized repository-relative POSIX paths",
        )
    return value


def _split_within(child: SplitScope, parent: SplitScope) -> bool:
    if parent == SplitScope.ALL:
        return True
    if parent == SplitScope.TRAIN_VALIDATION:
        return child in {
            SplitScope.TRAIN,
            SplitScope.VALIDATION,
            SplitScope.TRAIN_VALIDATION,
        }
    return child == parent


@dataclass(frozen=True, slots=True)
class VersionedIdentity:
    identity: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        _identifier(self.identity, "identity")
        _version(self.version, "identity version")
        _sha256(self.fingerprint, "identity fingerprint")


@dataclass(frozen=True, slots=True)
class InputManifestRef:
    role: str
    identity: VersionedIdentity
    target_free: bool
    repository_relative_path: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.role, "manifest role")
        if not isinstance(self.identity, VersionedIdentity):
            raise EDAContractError(
                "eda.manifest.identity_invalid",
                "manifest identity must be a VersionedIdentity",
            )
        if not isinstance(self.target_free, bool):
            raise EDAContractError(
                "eda.manifest.target_free_invalid", "target_free must be boolean"
            )
        object.__setattr__(
            self,
            "repository_relative_path",
            _relative_path(self.repository_relative_path),
        )


@dataclass(frozen=True, slots=True)
class UnitCount:
    """One typed count with an explicit comparison population."""

    name: str
    observation_unit: ObservationUnit
    value: int | None
    denominator: int | None
    denominator_unit: ObservationUnit
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]
    status: ComputationStatus = ComputationStatus.OBSERVED
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "count name")
        object.__setattr__(
            self,
            "observation_unit",
            _enum(self.observation_unit, ObservationUnit, "observation_unit"),
        )
        object.__setattr__(
            self,
            "denominator_unit",
            _enum(self.denominator_unit, ObservationUnit, "denominator_unit"),
        )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ComputationStatus, "status")
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "count provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "every count requires provenance"
            )
        if self.evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE} and (
            self.status == ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "unknown/unavailable evidence cannot contain observed counts",
            )
        if self.status == ComputationStatus.OBSERVED:
            value = _integer(self.value, "count value")
            denominator = _integer(self.denominator, "count denominator")
            assert value is not None and denominator is not None
            if (
                self.observation_unit == self.denominator_unit
                and value > denominator
            ):
                raise EDAContractError(
                    "eda.count.denominator_exceeded",
                    "a count cannot exceed a denominator expressed in the same unit",
                )
            if self.reason_code is not None:
                raise EDAContractError(
                    "eda.reason.unexpected",
                    "an observed count cannot carry an unavailable reason",
                )
        else:
            if self.value is not None:
                raise EDAContractError(
                    "eda.count.unavailable_not_null",
                    "unknown/not-computed/not-applicable/locked counts must be null",
                )
            _integer(self.denominator, "known unavailable denominator", nullable=True)
            _identifier(self.reason_code, "count reason_code")


def sum_unit_counts(name: str, counts: tuple[UnitCount, ...]) -> UnitCount:
    """Add compatible observed counts and reject implicit unit/scope mixing."""

    counts_input = _tuple_collection(counts, "counts")
    if not counts_input:
        raise EDAContractError(
            "eda.count.empty_aggregation", "at least one count is required"
        )
    if any(not isinstance(item, UnitCount) for item in counts_input):
        raise EDAContractError(
            "eda.count.type_invalid", "counts must contain UnitCount objects"
        )
    first = counts_input[0]
    signature = (
        first.observation_unit,
        first.denominator,
        first.denominator_unit,
        first.split_scope,
        first.evidence_scope,
        first.provenance,
        first.status,
    )
    if first.status != ComputationStatus.OBSERVED or any(
        (
            item.observation_unit,
            item.denominator,
            item.denominator_unit,
            item.split_scope,
            item.evidence_scope,
            item.provenance,
            item.status,
        )
        != signature
        for item in counts_input
    ):
        raise EDAContractError(
            "eda.count.unit_or_scope_mismatch",
            "counts may be summed only with an identical denominator, units, scopes, provenance, and status",
        )
    return UnitCount(
        name=name,
        observation_unit=first.observation_unit,
        value=sum(item.value or 0 for item in counts_input),
        denominator=first.denominator,
        denominator_unit=first.denominator_unit,
        split_scope=first.split_scope,
        evidence_scope=first.evidence_scope,
        provenance=first.provenance,
    )


@dataclass(frozen=True, slots=True)
class MetricCoverage:
    observation_unit: ObservationUnit
    denominator: int | None
    observed_count: int | None
    unknown_count: int | None
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]
    status: ComputationStatus = ComputationStatus.OBSERVED
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_unit",
            _enum(self.observation_unit, ObservationUnit, "observation_unit"),
        )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ComputationStatus, "status")
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "metric provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "every metric requires provenance"
            )
        if self.evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE} and (
            self.status == ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "unknown/unavailable evidence cannot contain observed metric coverage",
            )
        if self.status == ComputationStatus.OBSERVED:
            denominator = _integer(self.denominator, "metric denominator")
            observed = _integer(self.observed_count, "metric observed_count")
            unknown = _integer(self.unknown_count, "metric unknown_count")
            assert denominator is not None and observed is not None and unknown is not None
            if observed + unknown != denominator:
                raise EDAContractError(
                    "eda.metric.coverage_mismatch",
                    "observed_count + unknown_count must equal denominator",
                )
            if self.reason_code is not None:
                raise EDAContractError(
                    "eda.reason.unexpected",
                    "a computed metric cannot carry an unavailable reason",
                )
        else:
            if self.observed_count is not None or self.unknown_count is not None:
                raise EDAContractError(
                    "eda.metric.unavailable_counts_not_null",
                    "non-observed metric coverage must not fabricate zero counts",
                )
            _integer(self.denominator, "known unavailable denominator", nullable=True)
            _identifier(self.reason_code, "metric reason_code")


@dataclass(frozen=True, slots=True)
class QuantilePoint:
    numerator: int
    denominator: int
    value: int | float

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or not isinstance(self.numerator, int)
            or isinstance(self.denominator, bool)
            or not isinstance(self.denominator, int)
            or self.denominator <= 0
            or self.numerator < 0
            or self.numerator > self.denominator
            or math.gcd(self.numerator, self.denominator) != 1
        ):
            raise EDAContractError(
                "eda.quantile.probability_invalid",
                "quantile probability must be one reduced fraction in [0, 1]",
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise EDAContractError(
                "eda.quantile.value_invalid", "quantile value must be numeric"
            )
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise EDAContractError(
                "eda.serialization.non_finite", "quantile values must be finite"
            )

    @property
    def probability(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    measurement_unit: str
    minimum: int | float
    maximum: int | float
    mean: int | float
    quantiles: tuple[QuantilePoint, ...] = ()
    float_policy: str = EDA_FLOAT_POLICY
    quantile_policy: str = EDA_QUANTILE_POLICY

    def __post_init__(self) -> None:
        quantiles = _tuple_collection(self.quantiles, "quantiles")
        _identifier(self.measurement_unit, "measurement_unit")
        if self.float_policy != EDA_FLOAT_POLICY:
            raise EDAContractError(
                "eda.float_policy.invalid", "numeric summaries use the frozen float policy"
            )
        if self.quantile_policy != EDA_QUANTILE_POLICY:
            raise EDAContractError(
                "eda.quantile_policy.invalid",
                "numeric summaries use the frozen quantile policy",
            )
        values = (self.minimum, self.maximum, self.mean)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            for value in values
        ):
            raise EDAContractError(
                "eda.numeric.value_invalid", "numeric summary values must be finite"
            )
        if not self.minimum <= self.mean <= self.maximum:
            raise EDAContractError(
                "eda.numeric.order_invalid", "minimum <= mean <= maximum is required"
            )
        if any(not isinstance(item, QuantilePoint) for item in quantiles):
            raise EDAContractError(
                "eda.quantile.type_invalid",
                "quantiles must contain QuantilePoint objects",
            )
        ordered = tuple(sorted(quantiles, key=lambda item: item.probability))
        if len({item.probability for item in ordered}) != len(ordered):
            raise EDAContractError(
                "eda.quantile.duplicate", "quantile probabilities must be unique"
            )
        if any(
            left.value > right.value for left, right in zip(ordered, ordered[1:])
        ):
            raise EDAContractError(
                "eda.quantile.order_invalid", "quantile values must be nondecreasing"
            )
        if any(not self.minimum <= item.value <= self.maximum for item in ordered):
            raise EDAContractError(
                "eda.quantile.range_invalid",
                "every quantile value must lie within minimum and maximum",
            )
        if any(
            (item.numerator == 0 and item.value != self.minimum)
            or (
                item.numerator == item.denominator
                and item.value != self.maximum
            )
            for item in ordered
        ):
            raise EDAContractError(
                "eda.quantile.endpoint_mismatch",
                "R7 endpoint quantiles must equal the distribution bounds",
            )
        object.__setattr__(self, "quantiles", ordered)


@dataclass(frozen=True, slots=True)
class CategoryCount:
    category: str
    count: UnitCount

    def __post_init__(self) -> None:
        _domain_text(self.category, "category")
        if not isinstance(self.count, UnitCount):
            raise EDAContractError(
                "eda.raw_metric.count_type_invalid",
                "category count must be a UnitCount",
            )


@dataclass(frozen=True, slots=True)
class RawMetricSpec:
    metric_id: str
    summary_kind: MetricSummaryKind
    population_unit: ObservationUnit
    value_unit: ObservationUnit | None = None
    measurement_unit: str | None = None
    multi_valued: bool = False

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "raw metric ID")
        object.__setattr__(
            self,
            "summary_kind",
            _enum(self.summary_kind, MetricSummaryKind, "summary_kind"),
        )
        object.__setattr__(
            self,
            "population_unit",
            _enum(self.population_unit, ObservationUnit, "population_unit"),
        )
        if self.value_unit is not None:
            object.__setattr__(
                self,
                "value_unit",
                _enum(self.value_unit, ObservationUnit, "value_unit"),
            )
        if self.summary_kind == MetricSummaryKind.NUMERIC:
            _identifier(self.measurement_unit, "numeric measurement_unit")
            if self.value_unit is not None:
                raise EDAContractError(
                    "eda.raw_metric.spec_invalid",
                    "numeric metrics use measurement_unit rather than value_unit",
                )
        elif self.measurement_unit is not None or self.value_unit is None:
            raise EDAContractError(
                "eda.raw_metric.spec_invalid",
                "count/categorical metrics require value_unit only",
            )
        if not isinstance(self.multi_valued, bool) or (
            self.multi_valued
            and self.summary_kind != MetricSummaryKind.CATEGORICAL
        ):
            raise EDAContractError(
                "eda.raw_metric.spec_invalid",
                "multi_valued is valid only for categorical metrics",
            )


_RAW_METRIC_SPECS = (
    RawMetricSpec("bars", "numeric_distribution", "record", measurement_unit="bars_per_record"),
    RawMetricSpec("beats", "numeric_distribution", "record", measurement_unit="beats_per_record"),
    RawMetricSpec("conversion_outcomes", "categorical_distribution", "record", "record"),
    RawMetricSpec("cross_split_raw_identity_collisions", "count", "record", "raw_identity_collision"),
    RawMetricSpec("density", "numeric_distribution", "record", measurement_unit="pitched_notes_per_quarter_note"),
    RawMetricSpec("duplicate_candidates", "count", "record", "record"),
    RawMetricSpec("duration", "numeric_distribution", "record", measurement_unit="quarter_note"),
    RawMetricSpec("empty_records", "count", "record", "record"),
    RawMetricSpec("graph_edge_counts", "categorical_distribution", "record", "graph_edge"),
    RawMetricSpec("graph_node_counts", "categorical_distribution", "record", "graph_node"),
    RawMetricSpec("graph_size_distribution", "numeric_distribution", "record", measurement_unit="nodes_plus_edges_per_record"),
    RawMetricSpec("instruments", "categorical_distribution", "record", "instrument"),
    RawMetricSpec("invalid_records", "count", "record", "record"),
    RawMetricSpec("meter", "categorical_distribution", "record", "meter_event"),
    RawMetricSpec("meter_changes", "numeric_distribution", "record", measurement_unit="meter_changes_per_record"),
    RawMetricSpec("notes", "numeric_distribution", "record", measurement_unit="notes_per_record"),
    RawMetricSpec("onsets", "numeric_distribution", "record", measurement_unit="onsets_per_record"),
    RawMetricSpec("oversize_records", "count", "record", "record"),
    RawMetricSpec("parse_outcomes", "categorical_distribution", "record", "record"),
    RawMetricSpec("parts", "numeric_distribution", "record", measurement_unit="parts_per_record"),
    RawMetricSpec("percussion_presence", "categorical_distribution", "record", "record"),
    RawMetricSpec("pitch_range", "numeric_distribution", "record", measurement_unit="midi_note_number"),
    RawMetricSpec("polyphony", "numeric_distribution", "record", measurement_unit="simultaneous_note_count"),
    RawMetricSpec("programs", "categorical_distribution", "record", "program"),
    RawMetricSpec("quarantined_records", "count", "record", "record"),
    RawMetricSpec(
        "reason_codes",
        "categorical_distribution",
        "record",
        "record",
        multi_valued=True,
    ),
    RawMetricSpec("accepted_records", "count", "record", "record"),
    RawMetricSpec("discovered_records", "count", "record", "record"),
    RawMetricSpec("tempo", "numeric_distribution", "record", measurement_unit="beats_per_minute"),
    RawMetricSpec("tempo_changes", "numeric_distribution", "record", measurement_unit="tempo_changes_per_record"),
    RawMetricSpec("tracks", "numeric_distribution", "record", measurement_unit="tracks_per_record"),
    RawMetricSpec("version_candidates", "count", "record", "record"),
)
RAW_METRIC_CATALOG = MappingProxyType(
    {item.metric_id: item for item in sorted(_RAW_METRIC_SPECS, key=lambda item: item.metric_id)}
)
_INTEGRAL_RAW_NUMERIC_METRICS = frozenset(
    {
        "bars",
        "beats",
        "graph_size_distribution",
        "meter_changes",
        "notes",
        "onsets",
        "parts",
        "pitch_range",
        "polyphony",
        "tempo_changes",
        "tracks",
    }
)


@dataclass(frozen=True, slots=True)
class RawMetricEvidence:
    metric_id: str
    coverage: MetricCoverage
    count: UnitCount | None = None
    numeric: NumericDistribution | None = None
    categories: tuple[CategoryCount, ...] = ()

    def __post_init__(self) -> None:
        categories_input = _tuple_collection(self.categories, "raw metric categories")
        _identifier(self.metric_id, "raw metric ID")
        if not isinstance(self.coverage, MetricCoverage):
            raise EDAContractError(
                "eda.raw_metric.coverage_type_invalid",
                "raw metric coverage must be MetricCoverage",
            )
        if self.count is not None and not isinstance(self.count, UnitCount):
            raise EDAContractError(
                "eda.raw_metric.count_type_invalid",
                "raw metric count must be UnitCount or null",
            )
        if self.numeric is not None and not isinstance(
            self.numeric, NumericDistribution
        ):
            raise EDAContractError(
                "eda.raw_metric.numeric_type_invalid",
                "raw metric numeric summary must be NumericDistribution or null",
            )
        if any(not isinstance(item, CategoryCount) for item in categories_input):
            raise EDAContractError(
                "eda.raw_metric.category_type_invalid",
                "raw metric categories must contain CategoryCount objects",
            )
        try:
            spec = RAW_METRIC_CATALOG[self.metric_id]
        except KeyError as exc:
            raise EDAContractError(
                "eda.raw_metric.unknown", f"unknown common raw metric {self.metric_id!r}"
            ) from exc
        if self.coverage.observation_unit != spec.population_unit:
            raise EDAContractError(
                "eda.raw_metric.observation_unit_mismatch",
                f"{self.metric_id} requires population unit {spec.population_unit.value}",
            )
        categories = tuple(sorted(categories_input, key=lambda item: item.category))
        if len({item.category for item in categories}) != len(categories):
            raise EDAContractError(
                "eda.raw_metric.category_duplicate", "metric categories must be unique"
            )
        object.__setattr__(self, "categories", categories)
        summaries = int(self.count is not None) + int(self.numeric is not None) + int(bool(categories))
        if self.coverage.status != ComputationStatus.OBSERVED:
            if summaries:
                raise EDAContractError(
                    "eda.raw_metric.unavailable_summary",
                    "an unavailable metric cannot contain a numeric/count/category summary",
                )
            return
        assert self.coverage.observed_count is not None
        if self.coverage.observed_count == 0:
            known_empty_population = self.coverage.denominator == 0
            if spec.summary_kind == MetricSummaryKind.COUNT and known_empty_population:
                if self.count is None or summaries != 1:
                    raise EDAContractError(
                        "eda.raw_metric.empty_summary",
                        "a known-empty count metric requires one explicit typed zero",
                    )
                self._validate_count(spec, self.count)
            elif summaries:
                raise EDAContractError(
                    "eda.raw_metric.empty_summary",
                    "a metric with zero observed rows must not summarize an unknown population",
                )
            return
        empty_multi_occurrence_distribution = (
            spec.summary_kind == MetricSummaryKind.CATEGORICAL
            and not categories
            and (
                spec.multi_valued
                or spec.value_unit != spec.population_unit
            )
        )
        if empty_multi_occurrence_distribution:
            return
        if summaries != 1:
            raise EDAContractError(
                "eda.raw_metric.summary_arity",
                "a computed non-empty metric requires exactly one summary",
            )
        if spec.summary_kind == MetricSummaryKind.COUNT:
            if self.count is None:
                raise EDAContractError(
                    "eda.raw_metric.summary_kind_mismatch", "count summary required"
                )
            self._validate_count(spec, self.count)
        elif spec.summary_kind == MetricSummaryKind.NUMERIC:
            if self.numeric is None or self.numeric.measurement_unit != spec.measurement_unit:
                raise EDAContractError(
                    "eda.raw_metric.summary_kind_mismatch",
                    f"numeric summary in {spec.measurement_unit!r} required",
                )
            if self.numeric.minimum < 0:
                raise EDAContractError(
                    "eda.raw_metric.numeric_domain_invalid",
                    f"{self.metric_id} cannot contain negative source measurements",
                )
            if self.metric_id == "tempo" and self.numeric.minimum <= 0:
                raise EDAContractError(
                    "eda.raw_metric.numeric_domain_invalid",
                    "observed tempo values must be strictly positive",
                )
            if self.metric_id in _INTEGRAL_RAW_NUMERIC_METRICS and any(
                isinstance(value, float) and not value.is_integer()
                for value in (self.numeric.minimum, self.numeric.maximum)
            ):
                raise EDAContractError(
                    "eda.raw_metric.numeric_domain_invalid",
                    f"{self.metric_id} bounds must be integer-valued",
                )
            if self.metric_id == "pitch_range" and self.numeric.maximum > 127:
                raise EDAContractError(
                    "eda.raw_metric.numeric_domain_invalid",
                    "MIDI pitch values must lie in the inclusive range 0..127",
                )
            if self.coverage.observed_count == 1 and any(
                value != self.numeric.minimum
                for value in (
                    self.numeric.maximum,
                    self.numeric.mean,
                    *(item.value for item in self.numeric.quantiles),
                )
            ):
                raise EDAContractError(
                    "eda.raw_metric.singleton_summary_invalid",
                    "one observed scalar requires identical bounds, mean, and quantiles",
                )
            observed_count = self.coverage.observed_count
            mean_total = Fraction(self.numeric.mean) * observed_count
            minimum_total = (
                Fraction(self.numeric.maximum)
                + (observed_count - 1) * Fraction(self.numeric.minimum)
            )
            maximum_total = (
                Fraction(self.numeric.minimum)
                + (observed_count - 1) * Fraction(self.numeric.maximum)
            )
            mean_tolerance = (
                Fraction(math.ulp(self.numeric.mean)) * observed_count
                if isinstance(self.numeric.mean, float)
                else Fraction(0)
            )
            if (
                mean_total + mean_tolerance < minimum_total
                or mean_total - mean_tolerance > maximum_total
            ):
                raise EDAContractError(
                    "eda.raw_metric.sample_extrema_mismatch",
                    "numeric bounds and mean are impossible for the observed sample size",
                )
        else:
            if not categories:
                raise EDAContractError(
                    "eda.raw_metric.summary_kind_mismatch", "category summary required"
                )
            for category in categories:
                self._validate_count(spec, category.count)
            if spec.value_unit == spec.population_unit and not spec.multi_valued:
                total = sum(category.count.value or 0 for category in categories)
                if total != self.coverage.observed_count:
                    raise EDAContractError(
                        "eda.raw_metric.category_coverage_mismatch",
                        "single-valued category counts must equal observed_count",
                    )

    def _validate_count(self, spec: RawMetricSpec, count: UnitCount) -> None:
        if (
            count.name != self.metric_id
            or count.observation_unit != spec.value_unit
            or count.denominator_unit != spec.population_unit
            or count.denominator != self.coverage.denominator
            or count.split_scope != self.coverage.split_scope
            or count.evidence_scope != self.coverage.evidence_scope
            or count.provenance != self.coverage.provenance
            or count.status != ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.raw_metric.count_binding_mismatch",
                "metric counts must bind the catalog units and exact coverage scope",
            )
        if (
            spec.value_unit == spec.population_unit
            and count.value is not None
            and self.coverage.observed_count is not None
            and count.value > self.coverage.observed_count
        ):
            raise EDAContractError(
                "eda.raw_metric.count_coverage_exceeded",
                "same-population metric counts cannot exceed observed_count",
            )


APPROVED_RAW_GRAPH_CONTRACT = MappingProxyType(
    {
        "graph_schema": VersionedIdentity(
            identity="music_critic.graph.raw_schema",
            version="1.0.0",
            fingerprint="e0be8d4c522147036418501b230411ac5fc2eafa5284bab44bbc3e6ee3059fc8",
        ),
        "graph_builder": VersionedIdentity(
            identity="music_critic.graph.build_raw_graph",
            version="1.0.0",
            fingerprint="ccf423169631d4bb12295b92b4403625902eb1ded9478165f2ebc23d836fab65",
        ),
        "feature_registry": VersionedIdentity(
            identity="music_critic.graph.raw_feature_registry",
            version="1.0.0",
            fingerprint="a041e2c4a221bc0bc722ff3015423230b9e5d5cf56a6efbc4dc71aab351df6f7",
        ),
        "validator": VersionedIdentity(
            identity="music_critic.graph.validate_raw_graph",
            version="1.0.0",
            fingerprint="8de80cbf5929507da727293751aaba723d4256a5bc65aa0309b968873ffafa99",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    status: ComputationStatus
    target_free: bool | None
    graph_schema: VersionedIdentity | None = None
    graph_builder: VersionedIdentity | None = None
    feature_registry: VersionedIdentity | None = None
    validator: VersionedIdentity | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, ComputationStatus, "graph status")
        )
        identities = (
            self.graph_schema,
            self.graph_builder,
            self.feature_registry,
            self.validator,
        )
        if self.target_free is not None and not isinstance(self.target_free, bool):
            raise EDAContractError(
                "eda.graph.target_free_invalid", "target_free must be boolean or null"
            )
        if any(
            item is not None and not isinstance(item, VersionedIdentity)
            for item in identities
        ):
            raise EDAContractError(
                "eda.graph.identity_invalid",
                "graph contract bindings must be VersionedIdentity objects",
            )
        if self.status == ComputationStatus.OBSERVED:
            if self.target_free is not True or any(item is None for item in identities):
                raise EDAContractError(
                    "eda.graph.target_free_unproven",
                    "observed graph metrics require target_free=true and four contract bindings",
                )
            observed = {
                "graph_schema": self.graph_schema,
                "graph_builder": self.graph_builder,
                "feature_registry": self.feature_registry,
                "validator": self.validator,
            }
            if observed != dict(APPROVED_RAW_GRAPH_CONTRACT):
                raise EDAContractError(
                    "eda.graph.contract_unapproved",
                    "observed graph evidence must bind the exact approved raw graph contract",
                )
            if self.reason_code is not None:
                raise EDAContractError(
                    "eda.reason.unexpected", "observed graph evidence has no unavailable reason"
                )
        else:
            if self.target_free is True:
                raise EDAContractError(
                    "eda.graph.unavailable_claims_target_free",
                    "unavailable graph evidence cannot assert a completed target-free proof",
                )
            if any(item is not None for item in identities):
                raise EDAContractError(
                    "eda.graph.unavailable_contract_present",
                    "non-observed graph evidence cannot carry completed contract bindings",
                )
            _identifier(self.reason_code, "graph reason_code")


@dataclass(frozen=True, slots=True)
class InvariantEvidence:
    code: str
    status: InvariantStatus
    provenance: tuple[str, ...]
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.code, "invariant code")
        object.__setattr__(
            self, "status", _enum(self.status, InvariantStatus, "invariant status")
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "invariant provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "every invariant requires provenance"
            )
        if self.status == InvariantStatus.PASSED:
            if self.reason_code is not None:
                raise EDAContractError(
                    "eda.reason.unexpected", "passed invariant cannot have a reason"
                )
        else:
            _identifier(self.reason_code, "invariant reason_code")


@dataclass(frozen=True, slots=True)
class StructuredWarning:
    code: str
    message: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.code, "warning code")
        _domain_text(self.message, "warning message")
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "warning provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "every warning requires provenance"
            )


@dataclass(frozen=True, slots=True)
class UnavailableReason:
    code: str
    status: ComputationStatus
    provenance: tuple[str, ...]
    detail: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.code, "unavailable reason code")
        object.__setattr__(
            self, "status", _enum(self.status, ComputationStatus, "reason status")
        )
        if self.status == ComputationStatus.OBSERVED:
            raise EDAContractError(
                "eda.reason.status_invalid", "unavailable reason cannot be observed"
            )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "reason provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "every unavailable reason requires provenance"
            )
        if self.detail is not None:
            _domain_text(self.detail, "reason detail")


@dataclass(frozen=True, slots=True)
class ReportEnvelope:
    schema_name: str
    schema_version: str
    report_kind: ReportKind
    corpus: CorpusId
    source_identity: VersionedIdentity
    producer_identity: VersionedIdentity
    repository_commit: str
    evidence_scope: EvidenceScope
    execution_mode: ExecutionMode
    completeness_status: CompletenessStatus
    split_scope: SplitScope
    observation_units: tuple[ObservationUnit, ...]
    input_manifests: tuple[InputManifestRef, ...]
    invariants: tuple[InvariantEvidence, ...] = ()
    warnings: tuple[StructuredWarning, ...] = ()
    unavailable_reasons: tuple[UnavailableReason, ...] = ()
    operational_metadata: Mapping[str, object] = field(default_factory=dict)
    envelope_schema_name: str = EDA_ENVELOPE_SCHEMA_NAME
    envelope_schema_version: str = EDA_ENVELOPE_SCHEMA_VERSION
    version_policy: str = EDA_SCHEMA_VERSION_POLICY

    def __post_init__(self) -> None:
        observation_units = _tuple_collection(
            self.observation_units, "observation_units"
        )
        input_manifests = _tuple_collection(self.input_manifests, "input_manifests")
        invariants_input = _tuple_collection(self.invariants, "invariants")
        warnings_input = _tuple_collection(self.warnings, "warnings")
        reasons_input = _tuple_collection(
            self.unavailable_reasons, "unavailable_reasons"
        )
        _identifier(self.schema_name, "schema_name")
        _version(self.schema_version, "schema_version")
        if not isinstance(self.source_identity, VersionedIdentity) or not isinstance(
            self.producer_identity, VersionedIdentity
        ):
            raise EDAContractError(
                "eda.envelope.identity_invalid",
                "source_identity and producer_identity must be VersionedIdentity objects",
            )
        object.__setattr__(
            self, "report_kind", _enum(self.report_kind, ReportKind, "report_kind")
        )
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        if not isinstance(self.repository_commit, str) or _GIT_SHA_RE.fullmatch(
            self.repository_commit
        ) is None:
            raise EDAContractError(
                "eda.repository_commit.invalid",
                "repository_commit must be a lowercase 40- or 64-hex Git object ID",
            )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        object.__setattr__(
            self,
            "execution_mode",
            _enum(self.execution_mode, ExecutionMode, "execution_mode"),
        )
        object.__setattr__(
            self,
            "completeness_status",
            _enum(self.completeness_status, CompletenessStatus, "completeness_status"),
        )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        units_unsorted = tuple(
            _enum(item, ObservationUnit, "observation unit")
            for item in observation_units
        )
        if len(units_unsorted) != len(set(units_unsorted)):
            raise EDAContractError(
                "eda.observation_units.duplicate",
                "observation units must not be duplicated",
            )
        units = tuple(sorted(units_unsorted, key=lambda item: item.value))
        if not units:
            raise EDAContractError(
                "eda.observation_units.empty", "the envelope must name every used unit"
            )
        object.__setattr__(self, "observation_units", units)
        if any(not isinstance(item, InputManifestRef) for item in input_manifests):
            raise EDAContractError(
                "eda.manifest.type_invalid",
                "input_manifests must contain InputManifestRef objects",
            )
        manifests = tuple(
            sorted(input_manifests, key=lambda item: (item.role, item.identity.identity))
        )
        if not manifests and self.evidence_scope not in {
            EvidenceScope.UNKNOWN,
            EvidenceScope.UNAVAILABLE,
        }:
            raise EDAContractError(
                "eda.manifest.empty",
                "observed-scope reports require at least one input manifest binding",
            )
        if len({(item.role, item.identity.identity) for item in manifests}) != len(manifests):
            raise EDAContractError(
                "eda.manifest.duplicate", "manifest role/identity pairs must be unique"
            )
        object.__setattr__(self, "input_manifests", manifests)
        typed_collections = (
            (invariants_input, InvariantEvidence, "invariants"),
            (warnings_input, StructuredWarning, "warnings"),
            (reasons_input, UnavailableReason, "unavailable_reasons"),
        )
        for values, expected_type, name in typed_collections:
            if any(not isinstance(item, expected_type) for item in values):
                raise EDAContractError(
                    "eda.envelope.row_type_invalid",
                    f"{name} contains an invalid row type",
                )
        invariants = tuple(sorted(invariants_input, key=lambda item: item.code))
        warnings = tuple(
            sorted(
                warnings_input,
                key=lambda item: (item.code, item.message, item.provenance),
            )
        )
        reasons = tuple(sorted(reasons_input, key=lambda item: item.code))
        if len({item.code for item in invariants}) != len(invariants):
            raise EDAContractError(
                "eda.invariant.duplicate", "invariant codes must be unique"
            )
        if len({item.code for item in reasons}) != len(reasons):
            raise EDAContractError(
                "eda.reason.duplicate", "unavailable reason codes must be unique"
            )
        if len(warnings) != len(
            {(item.code, item.message, item.provenance) for item in warnings}
        ):
            raise EDAContractError(
                "eda.warning.duplicate", "identical warning rows are not allowed"
            )
        object.__setattr__(self, "invariants", invariants)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "unavailable_reasons", reasons)
        operational_metadata = _mapping(
            self.operational_metadata, path="$.operational_metadata"
        )
        _validate_operational_metadata(operational_metadata)
        object.__setattr__(self, "operational_metadata", operational_metadata)
        if self.envelope_schema_name != EDA_ENVELOPE_SCHEMA_NAME or (
            self.envelope_schema_version != EDA_ENVELOPE_SCHEMA_VERSION
        ):
            raise EDAContractError(
                "eda.envelope.version_invalid", "unsupported common envelope schema"
            )
        if self.version_policy != EDA_SCHEMA_VERSION_POLICY:
            raise EDAContractError(
                "eda.version_policy.invalid", "unsupported schema versioning policy"
            )
        expected_mode = _SCOPE_TO_EXECUTION[self.evidence_scope]
        if self.execution_mode != expected_mode:
            raise EDAContractError(
                "eda.evidence.execution_mismatch",
                f"{self.evidence_scope.value} evidence requires {expected_mode.value}",
            )
        if self.evidence_scope == EvidenceScope.FIXTURE and (
            self.execution_mode == ExecutionMode.PRODUCTION_SCAN
        ):
            raise EDAContractError(
                "eda.evidence.fixture_as_production",
                "fixture evidence cannot be declared production evidence",
            )
        if self.evidence_scope == EvidenceScope.UNAVAILABLE and (
            self.completeness_status != CompletenessStatus.UNAVAILABLE
        ):
            raise EDAContractError(
                "eda.evidence.unavailable_completeness_mismatch",
                "unavailable evidence requires unavailable completeness",
            )
        if self.evidence_scope == EvidenceScope.UNKNOWN and (
            self.completeness_status != CompletenessStatus.UNKNOWN
        ):
            raise EDAContractError(
                "eda.evidence.unknown_completeness_mismatch",
                "unknown evidence requires unknown completeness",
            )
        if self.completeness_status in {
            CompletenessStatus.NOT_COMPUTED,
            CompletenessStatus.UNAVAILABLE,
            CompletenessStatus.UNKNOWN,
        } and not self.unavailable_reasons:
            raise EDAContractError(
                "eda.reason.missing",
                "non-computed/unavailable/unknown reports require a structured reason",
            )


@dataclass(frozen=True, slots=True)
class ExtensionRow:
    row_id: str
    payload: Mapping[str, object]
    counts: tuple[UnitCount, ...] = ()
    coverage: MetricCoverage = field(kw_only=True)

    def __post_init__(self) -> None:
        counts_input = _tuple_collection(self.counts, "extension counts")
        _identifier(self.row_id, "extension row_id")
        encoded = _mapping(self.payload, path=f"$.extensions[{self.row_id}]")
        _reject_extension_common_fields(
            encoded, path=f"$.extensions[{self.row_id}].payload"
        )
        _reject_keys(encoded, _OPERATIONAL_KEYS)
        _reject_untyped_count_fields(encoded)
        object.__setattr__(self, "payload", encoded)
        if not isinstance(self.coverage, MetricCoverage):
            raise EDAContractError(
                "eda.extension.coverage_type_invalid",
                "every extension metric requires MetricCoverage",
                path=f"$.extensions[{self.row_id}].coverage",
            )
        if any(not isinstance(item, UnitCount) for item in counts_input):
            raise EDAContractError(
                "eda.extension.count_type_invalid",
                "extension counts must contain UnitCount objects",
            )
        counts = tuple(sorted(counts_input, key=lambda item: item.name))
        if len({item.name for item in counts}) != len(counts):
            raise EDAContractError(
                "eda.extension.count_duplicate",
                "extension count names must be unique within one row",
            )
        for index, count in enumerate(counts):
            if _is_extension_common_field(count.name):
                raise EDAContractError(
                    "eda.extension.common_field_collision",
                    f"extension count {count.name!r} collides with the shared EDA schema",
                    path=f"$.extensions[{self.row_id}].counts[{index}].name",
                )
            if (
                count.split_scope != self.coverage.split_scope
                or count.evidence_scope != self.coverage.evidence_scope
                or count.provenance != self.coverage.provenance
                or count.status != ComputationStatus.OBSERVED
            ):
                raise EDAContractError(
                    "eda.extension.count_coverage_mismatch",
                    "extension counts must be observed and bind the metric coverage scopes and provenance",
                    path=f"$.extensions[{self.row_id}].counts[{index}]",
                )
            if (
                count.denominator != self.coverage.denominator
                or count.denominator_unit != self.coverage.observation_unit
            ):
                raise EDAContractError(
                    "eda.extension.count_coverage_mismatch",
                    "known extension count denominators must bind the metric coverage population",
                    path=f"$.extensions[{self.row_id}].counts[{index}]",
                )
            if self.coverage.status != ComputationStatus.OBSERVED:
                raise EDAContractError(
                    "eda.extension.count_coverage_mismatch",
                    "observed extension counts require observed metric coverage",
                    path=f"$.extensions[{self.row_id}].counts[{index}]",
                )
            if (
                count.observation_unit == self.coverage.observation_unit
                and self.coverage.observed_count is not None
                and count.value is not None
                and count.value > self.coverage.observed_count
            ):
                raise EDAContractError(
                    "eda.extension.count_observed_exceeded",
                    "same-population extension counts cannot exceed observed_count",
                    path=f"$.extensions[{self.row_id}].counts[{index}]",
                )
        if self.coverage.status != ComputationStatus.OBSERVED and (encoded or counts):
            raise EDAContractError(
                "eda.extension.unavailable_summary",
                "a non-observed extension metric cannot contain payload or counts",
                path=f"$.extensions[{self.row_id}]",
            )
        if self.coverage.observed_count == 0:
            if encoded:
                raise EDAContractError(
                    "eda.extension.empty_summary",
                    "an extension metric with zero observed rows cannot contain payload",
                    path=f"$.extensions[{self.row_id}].payload",
                )
            if counts and (
                self.coverage.denominator != 0
                or self.coverage.unknown_count != 0
                or any(count.value != 0 for count in counts)
            ):
                raise EDAContractError(
                    "eda.extension.empty_summary",
                    "typed zero counts are valid only for a known-empty 0/0 coverage population",
                    path=f"$.extensions[{self.row_id}].counts",
                )
        object.__setattr__(self, "counts", counts)


def _contains_work_identity_field(
    value: object,
    *,
    ancestor_tokens: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_tokens = _normalized_key_tokens(key)
            tokens = ancestor_tokens | key_tokens
            compact_key = _normalized_field_name(key).replace("_", "")
            if compact_key in {
                "canonicalid",
                "canonicalids",
                "canonicalidentifier",
                "canonicalidentifiers",
                "canonicalworkid",
                "canonicalworkids",
                "canonicalworkidentifier",
                "canonicalworkidentifiers",
                "logicalid",
                "logicalids",
                "logicalidentifier",
                "logicalidentifiers",
                "logicalworkid",
                "logicalworkids",
                "logicalworkidentifier",
                "logicalworkidentifiers",
                "workid",
                "workids",
                "workidentity",
                "workidentities",
                "workidentifier",
                "workidentifiers",
                "workkey",
                "workkeys",
                "workuid",
                "workuids",
                "workuuid",
                "workuuids",
            }:
                return True
            identity_tokens = {
                "id",
                "ids",
                "identity",
                "identities",
                "identifier",
                "identifiers",
                "uid",
                "uids",
                "uuid",
                "uuids",
            }
            if tokens & {"work", "works"} and tokens & (
                identity_tokens | {"key", "keys"}
            ):
                return True
            if _contains_work_identity_field(item, ancestor_tokens=tokens):
                return True
    elif isinstance(value, (tuple, list)):
        return any(
            _contains_work_identity_field(item, ancestor_tokens=ancestor_tokens)
            for item in value
        )
    return False


@dataclass(frozen=True, slots=True)
class SourceExtension:
    corpus: CorpusId
    namespace: str
    schema_name: str
    schema_version: str
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]
    rows: tuple[ExtensionRow, ...]
    target_free: bool
    work_identity: VersionedIdentity | None = None
    extension_contract_version: str = EDA_SOURCE_EXTENSION_VERSION
    extension_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        rows_input = _tuple_collection(self.rows, "extension rows")
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        if not source_extension_namespace_is_valid(self.namespace, self.corpus):
            raise EDAContractError(
                "eda.extension.namespace_invalid",
                "extension namespace must be normalized and begin with '<corpus>.'",
            )
        _identifier(self.schema_name, "extension schema_name")
        _version(self.schema_version, "extension schema_version")
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "extension provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "source extensions require provenance"
            )
        if self.extension_contract_version != EDA_SOURCE_EXTENSION_VERSION:
            raise EDAContractError(
                "eda.extension.version_invalid", "unsupported source extension contract"
            )
        if not isinstance(self.target_free, bool):
            raise EDAContractError(
                "eda.extension.target_free_invalid", "target_free must be boolean"
            )
        if self.work_identity is not None and not isinstance(
            self.work_identity, VersionedIdentity
        ):
            raise EDAContractError(
                "eda.extension.work_identity_invalid",
                "extension work_identity must be a VersionedIdentity",
            )
        if any(not isinstance(item, ExtensionRow) for item in rows_input):
            raise EDAContractError(
                "eda.extension.row_type_invalid",
                "extension rows must contain ExtensionRow objects",
            )
        rows = tuple(sorted(rows_input, key=lambda item: item.row_id))
        if len({item.row_id for item in rows}) != len(rows):
            raise EDAContractError(
                "eda.extension.row_duplicate", "extension row IDs must be unique"
            )
        for index, row in enumerate(rows):
            if (
                row.coverage.split_scope != self.split_scope
                or row.coverage.evidence_scope != self.evidence_scope
                or row.coverage.provenance != self.provenance
            ):
                raise EDAContractError(
                    "eda.extension.coverage_scope_mismatch",
                    "extension metric coverage must bind the extension split, evidence scope, and provenance",
                    path=f"$.extensions.rows[{index}].coverage",
                )
        if self.evidence_scope in {
            EvidenceScope.UNKNOWN,
            EvidenceScope.UNAVAILABLE,
        } and any(
            row.coverage.status == ComputationStatus.OBSERVED for row in rows
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "unknown/unavailable extensions cannot contain observed metric rows",
            )
        work_units = {ObservationUnit.LOGICAL_WORK, ObservationUnit.CANONICAL_WORK}
        if self.work_identity is None and any(
            (
                row.coverage.observation_unit in work_units
                and row.coverage.denominator is not None
            )
            or any(
                (
                    count.observation_unit in work_units
                    or count.denominator_unit in work_units
                )
                and (count.value is not None or count.denominator is not None)
                for count in row.counts
            )
            for row in rows
        ):
            raise EDAContractError(
                "eda.extension.work_identity_unproven",
                "known extension work values or populations require a versioned work identity",
            )
        if self.work_identity is None and any(
            _contains_work_identity_field(row.payload) for row in rows
        ):
            raise EDAContractError(
                "eda.extension.work_identity_unproven",
                "extension work-ID fields require a versioned work identity",
            )
        object.__setattr__(self, "rows", rows)
        payload = {
            "corpus": self.corpus.value,
            "evidence_scope": self.evidence_scope.value,
            "extension_contract_version": self.extension_contract_version,
            "namespace": self.namespace,
            "provenance": self.provenance,
            "rows": _jsonable(self.rows),
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "split_scope": self.split_scope.value,
            "target_free": self.target_free,
            "work_identity": _jsonable(self.work_identity),
        }
        object.__setattr__(self, "extension_fingerprint", canonical_json_sha256(payload))


def _validate_extension_schemas_across_splits(
    extensions: tuple[SourceExtension, ...],
) -> None:
    extension_schemas: dict[str, tuple[object, ...]] = {}
    row_units: dict[tuple[str, str], ObservationUnit] = {}
    observed_count_schemas: dict[
        tuple[str, str], tuple[tuple[str, ObservationUnit, ObservationUnit], ...]
    ] = {}
    for extension in extensions:
        schema = (
            extension.schema_name,
            extension.schema_version,
            extension.work_identity,
            extension.target_free,
        )
        previous = extension_schemas.setdefault(extension.namespace, schema)
        if previous != schema:
            raise EDAContractError(
                "eda.extension.schema_mismatch",
                "one extension namespace must keep one schema identity across splits",
            )
        for row in extension.rows:
            key = (extension.namespace, row.row_id)
            previous_unit = row_units.setdefault(
                key, row.coverage.observation_unit
            )
            if previous_unit != row.coverage.observation_unit:
                raise EDAContractError(
                    "eda.extension.schema_mismatch",
                    "one extension metric row must keep one coverage observation unit across splits",
                )
            if row.coverage.status == ComputationStatus.OBSERVED:
                count_schema = tuple(
                    (
                        count.name,
                        count.observation_unit,
                        count.denominator_unit,
                    )
                    for count in row.counts
                )
                previous_counts = observed_count_schemas.setdefault(key, count_schema)
                if previous_counts != count_schema:
                    raise EDAContractError(
                        "eda.extension.schema_mismatch",
                        "one extension metric row must keep one typed-count schema across splits",
                    )


@dataclass(frozen=True, slots=True)
class RawCorpusEDAPayload:
    metrics: tuple[RawMetricEvidence, ...]
    graph_evidence: GraphEvidence
    extensions: tuple[SourceExtension, ...] = ()

    def __post_init__(self) -> None:
        metrics_input = _tuple_collection(self.metrics, "raw metrics")
        extensions_input = _tuple_collection(self.extensions, "raw extensions")
        if any(not isinstance(item, RawMetricEvidence) for item in metrics_input):
            raise EDAContractError(
                "eda.raw_metric.type_invalid",
                "raw metrics must contain RawMetricEvidence objects",
            )
        if not isinstance(self.graph_evidence, GraphEvidence):
            raise EDAContractError(
                "eda.graph.type_invalid",
                "graph_evidence must be a GraphEvidence object",
            )
        if any(not isinstance(item, SourceExtension) for item in extensions_input):
            raise EDAContractError(
                "eda.extension.type_invalid",
                "extensions must contain SourceExtension objects",
            )
        metrics = tuple(sorted(metrics_input, key=lambda item: item.metric_id))
        if tuple(item.metric_id for item in metrics) != tuple(RAW_METRIC_CATALOG):
            raise EDAContractError(
                "eda.raw_metric.catalog_incomplete",
                "raw reports require exactly one row for every common metric ID",
            )
        object.__setattr__(self, "metrics", metrics)
        extensions = tuple(
            sorted(
                extensions_input,
                key=lambda item: (item.namespace, item.split_scope.value),
            )
        )
        extension_keys = {
            (item.namespace, item.split_scope) for item in extensions
        }
        if len(extension_keys) != len(extensions):
            raise EDAContractError(
                "eda.extension.namespace_duplicate",
                "extension namespace/split pairs must be unique",
            )
        _validate_extension_schemas_across_splits(extensions)
        object.__setattr__(self, "extensions", extensions)


@dataclass(frozen=True, slots=True)
class AvailabilityCounts:
    observation_unit: ObservationUnit
    denominator: int
    available: int
    masked: int
    missing: int
    unsupported: int
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_unit",
            _enum(self.observation_unit, ObservationUnit, "availability observation_unit"),
        )
        if self.observation_unit != ObservationUnit.TARGET_ROW:
            raise EDAContractError(
                "eda.availability.unit_invalid",
                "native supervision availability is counted in target rows",
            )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        if self.evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE}:
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "availability counts require observed evidence",
            )
        values = tuple(
            _integer(value, name)
            for value, name in (
                (self.denominator, "availability denominator"),
                (self.available, "available"),
                (self.masked, "masked"),
                (self.missing, "missing"),
                (self.unsupported, "unsupported"),
            )
        )
        if sum(value or 0 for value in values[1:]) != values[0]:
            raise EDAContractError(
                "eda.availability.denominator_mismatch",
                "available + masked + missing + unsupported must equal denominator",
            )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "availability provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "availability counts require provenance"
            )


@dataclass(frozen=True, slots=True)
class SourceValueIdentity:
    corpus: CorpusId
    source_task_id: str
    dialect: str
    source_value: object
    value_kind: SourceValueKind
    identity_version: str = EDA_SOURCE_VALUE_IDENTITY_VERSION
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        _identifier(self.source_task_id, "source_task_id")
        _identifier(self.dialect, "dialect")
        object.__setattr__(
            self, "value_kind", _enum(self.value_kind, SourceValueKind, "value_kind")
        )
        if self.identity_version != EDA_SOURCE_VALUE_IDENTITY_VERSION:
            raise EDAContractError(
                "eda.source_value.version_invalid", "unsupported source-value identity"
            )
        value = _jsonable(self.source_value, path="$.source_value")
        if self.value_kind == SourceValueKind.SCALAR:
            if value is None or (
                isinstance(value, str)
                and (not value or value != value.strip())
            ):
                raise EDAContractError(
                    "eda.source_value.missing_is_not_class",
                    "missing/null/blank source values are availability states, never classes",
                )
            if isinstance(value, (list, dict)):
                raise EDAContractError(
                    "eda.source_value.kind_mismatch",
                    "scalar source value cannot be a collection",
                )
        if self.value_kind in {SourceValueKind.MULTI_LABEL, SourceValueKind.EMPTY_MULTI_LABEL}:
            if not isinstance(value, list):
                raise EDAContractError(
                    "eda.source_value.kind_mismatch", "multilabel source value must be a list/tuple"
                )
            if self.value_kind == SourceValueKind.EMPTY_MULTI_LABEL and value:
                raise EDAContractError(
                    "eda.source_value.empty_multilabel_invalid",
                    "empty multilabel identity must contain zero labels",
                )
            if self.value_kind == SourceValueKind.MULTI_LABEL and not value:
                raise EDAContractError(
                    "eda.source_value.multilabel_empty",
                    "use empty_multi_label for an available empty label set",
                )
            if self.value_kind == SourceValueKind.MULTI_LABEL and any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in value
            ):
                raise EDAContractError(
                    "eda.source_value.multilabel_member_invalid",
                    "multilabel identities require non-empty, stripped string members",
                )
            rendered = tuple(dumps_canonical_json(item) for item in value)
            if len(rendered) != len(set(rendered)):
                raise EDAContractError(
                    "eda.source_value.multilabel_duplicate",
                    "multilabel identities cannot contain duplicate members",
                )
            value = [
                item
                for _, item in sorted(
                    zip(rendered, value, strict=True), key=lambda pair: pair[0]
                )
            ]
        normalized_value: object = tuple(value) if isinstance(value, list) else value
        object.__setattr__(self, "source_value", normalized_value)
        identity_payload = {
            "corpus": self.corpus.value,
            "dialect": self.dialect,
            "identity_version": self.identity_version,
            "source_task_id": self.source_task_id,
            "source_value": _jsonable(normalized_value),
            "value_kind": self.value_kind.value,
        }
        object.__setattr__(
            self,
            "identity",
            f"source-value:{canonical_json_sha256(identity_payload)}",
        )


@dataclass(frozen=True, slots=True)
class ClassSupport:
    source_value: SourceValueIdentity
    occurrence_count: UnitCount
    unique_record_count: UnitCount
    unique_work_count: UnitCount
    available_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_value, SourceValueIdentity) or any(
            not isinstance(item, UnitCount)
            for item in (
                self.occurrence_count,
                self.unique_record_count,
                self.unique_work_count,
            )
        ):
            raise EDAContractError(
                "eda.class_support.type_invalid",
                "class support requires one SourceValueIdentity and three UnitCount objects",
            )
        if self.available_only is not True:
            raise EDAContractError(
                "eda.class_support.not_available_only",
                "class support may be computed only from available native rows",
            )
        if (
            self.occurrence_count.name != "occurrence_count"
            or self.unique_record_count.name != "unique_record_count"
            or self.unique_work_count.name != "unique_work_count"
        ):
            raise EDAContractError(
                "eda.class_support.name_invalid",
                "class-support UnitCount names must match their typed fields",
            )
        if (
            self.occurrence_count.observation_unit != ObservationUnit.LABEL_OCCURRENCE
            or self.occurrence_count.denominator_unit != ObservationUnit.TARGET_ROW
            or self.unique_record_count.observation_unit != ObservationUnit.RECORD
            or self.unique_record_count.denominator_unit != ObservationUnit.RECORD
            or self.unique_work_count.observation_unit
            not in {ObservationUnit.LOGICAL_WORK, ObservationUnit.CANONICAL_WORK}
            or self.unique_work_count.denominator_unit
            != self.unique_work_count.observation_unit
        ):
            raise EDAContractError(
                "eda.class_support.unit_invalid",
                "class support requires occurrence, unique-record, and explicit work units",
            )
        counts = (self.occurrence_count, self.unique_record_count, self.unique_work_count)
        if any(
            item.split_scope != self.occurrence_count.split_scope
            or item.evidence_scope != self.occurrence_count.evidence_scope
            or item.provenance != self.occurrence_count.provenance
            for item in counts
        ):
            raise EDAContractError(
                "eda.class_support.scope_mismatch",
                "class-support scopes and provenance must agree",
            )
        if self.occurrence_count.status != ComputationStatus.OBSERVED or (
            self.unique_record_count.status != ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.class_support.required_count_unavailable",
                "occurrence and unique-record support must be observed",
            )
        occurrence = self.occurrence_count.value
        unique_records = self.unique_record_count.value
        assert occurrence is not None and unique_records is not None
        if unique_records > occurrence:
            raise EDAContractError(
                "eda.class_support.unique_record_exceeded",
                "unique-record support cannot exceed label occurrences",
            )
        if (occurrence == 0) != (unique_records == 0):
            raise EDAContractError(
                "eda.class_support.record_cardinality_mismatch",
                "label occurrences and unique-record support must be zero together",
            )
        if self.unique_work_count.status == ComputationStatus.OBSERVED:
            unique_works = self.unique_work_count.value
            assert unique_works is not None
            if unique_works > unique_records:
                raise EDAContractError(
                    "eda.class_support.unique_work_exceeded",
                    "unique-work support cannot exceed unique-record support",
                )
            if (unique_records == 0) != (unique_works == 0):
                raise EDAContractError(
                    "eda.class_support.work_cardinality_mismatch",
                    "unique-record and unique-work support must be zero together",
                )


DILEMMADATA_COMMON_REGISTRY_ID = "music_critic.dilemmadata.common_harmonic"
DILEMMADATA_COMMON_REGISTRY_VERSION = "1.0.0"
DILEMMADATA_COMMON_REGISTRY_FINGERPRINT = (
    "bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e"
)
APPROVED_PROJECTION_REGISTRIES = MappingProxyType(
    {
        DILEMMADATA_COMMON_REGISTRY_ID: VersionedIdentity(
            identity=DILEMMADATA_COMMON_REGISTRY_ID,
            version=DILEMMADATA_COMMON_REGISTRY_VERSION,
            fingerprint=DILEMMADATA_COMMON_REGISTRY_FINGERPRINT,
        )
    }
)

_COMMON_QUALITY_TASK = "dilemmadata.common.chord.quality"
_COMMON_INVERSION_TASK = "dilemmadata.common.chord.inversion"
_COMMON_ROOT_PC_TASK = "dilemmadata.common.chord.root_pc"
_COMMON_BASS_PC_TASK = "dilemmadata.common.chord.bass_pc"
_COMMON_LOCAL_KEY_TASK = "dilemmadata.common.key.local"
_COMMON_PITCH_CLASS_SET_TASK = "dilemmadata.common.chord.pitch_class_set"
_AN_QUALITY_TASK = "dilemmadata.an.chord.quality"
_DLC_QUALITY_TASK = "dilemmadata.dlc.chord.quality"
_AN_INVERSION_TASK = "dilemmadata.an.chord.inversion"
_DLC_INVERSION_TASK = "dilemmadata.dlc.chord.inversion"
_AN_ROOT_TASK = "dilemmadata.an.chord.root"
_DLC_ROOT_TASK = "dilemmadata.dlc.chord.root"
_AN_BASS_TASK = "dilemmadata.an.chord.bass"
_DLC_BASS_TASK = "dilemmadata.dlc.chord.bass"
_AN_LOCAL_KEY_TASK = "dilemmadata.an.key.local"
_DLC_LOCAL_KEY_TASK = "dilemmadata.dlc.key.local"

DILEMMADATA_COMMON_TASK_IDS = (
    _COMMON_BASS_PC_TASK,
    _COMMON_INVERSION_TASK,
    _COMMON_PITCH_CLASS_SET_TASK,
    _COMMON_QUALITY_TASK,
    _COMMON_ROOT_PC_TASK,
    _COMMON_LOCAL_KEY_TASK,
)
_DILEMMADATA_COMMON_SOURCES = MappingProxyType(
    {
        _COMMON_QUALITY_TASK: frozenset({_AN_QUALITY_TASK, _DLC_QUALITY_TASK}),
        _COMMON_INVERSION_TASK: frozenset({_AN_INVERSION_TASK, _DLC_INVERSION_TASK}),
        _COMMON_ROOT_PC_TASK: frozenset({_AN_ROOT_TASK, _DLC_ROOT_TASK}),
        _COMMON_BASS_PC_TASK: frozenset({_AN_BASS_TASK, _DLC_BASS_TASK}),
        _COMMON_LOCAL_KEY_TASK: frozenset({_AN_LOCAL_KEY_TASK, _DLC_LOCAL_KEY_TASK}),
        _COMMON_PITCH_CLASS_SET_TASK: frozenset(
            {_AN_QUALITY_TASK, _AN_ROOT_TASK, _DLC_QUALITY_TASK, _DLC_ROOT_TASK}
        ),
    }
)
_AN_QUALITY_VALUES = frozenset(
    {
        "Augmented Fourth", "Diminished Fifth", "French augmented sixth chord",
        "French augmented sixth chord in first inversion",
        "French augmented sixth chord in root position",
        "French augmented sixth chord in third inversion",
        "German augmented sixth chord",
        "German augmented sixth chord in root position",
        "German augmented sixth chord in second inversion",
        "German augmented sixth chord in third inversion",
        "Italian augmented sixth chord",
        "Italian augmented sixth chord in root position",
        "Italian augmented sixth chord in second inversion", "Kumoi pentachord",
        "Major Second", "Major Seventh", "Major Sixth", "Major Third",
        "Minor Sixth", "Minor Third", "Perfect Fifth", "Perfect Fourth",
        "augmented major tetrachord", "augmented seventh chord", "augmented triad",
        "diminished seventh chord", "diminished triad",
        "diminished-major ninth chord", "dominant seventh chord", "dominant-ninth",
        "enharmonic equivalent to diminished triad",
        "enharmonic equivalent to half-diminished seventh chord",
        "enharmonic equivalent to major triad",
        "enharmonic equivalent to minor seventh chord",
        "enharmonic equivalent to minor triad",
        "enharmonic to dominant seventh chord", "flat-ninth pentachord",
        "half-diminished seventh chord", "incomplete dominant-seventh chord",
        "incomplete half-diminished seventh chord", "incomplete major-seventh chord",
        "incomplete minor-seventh chord", "lydian tetrachord", "major seventh chord",
        "major triad", "major-minor tetramirror", "major-ninth chord",
        "major-second major tetrachord", "major-second minor tetrachord",
        "minor seventh chord", "minor triad", "minor trichord",
        "minor-augmented tetrachord", "minor-diminished ninth chord",
        "minor-ninth chord", "note", "perfect-fourth diminished tetrachord",
        "perfect-fourth major tetrachord", "perfect-fourth minor tetrachord",
        "phrygian tetrachord", "quartal tetramirror", "quartal trichord",
        "whole-tone tetramirror", "whole-tone trichord",
    }
)
_QUALITY_COLLAPSES = MappingProxyType(
    {
        "French augmented sixth chord in first inversion": "French augmented sixth chord",
        "French augmented sixth chord in root position": "French augmented sixth chord",
        "French augmented sixth chord in third inversion": "French augmented sixth chord",
        "German augmented sixth chord in root position": "German augmented sixth chord",
        "German augmented sixth chord in second inversion": "German augmented sixth chord",
        "German augmented sixth chord in third inversion": "German augmented sixth chord",
        "Italian augmented sixth chord in root position": "Italian augmented sixth chord",
        "Italian augmented sixth chord in second inversion": "Italian augmented sixth chord",
        "enharmonic equivalent to diminished triad": "diminished triad",
        "enharmonic equivalent to half-diminished seventh chord": "half-diminished seventh chord",
        "enharmonic equivalent to major triad": "major triad",
        "enharmonic equivalent to minor seventh chord": "minor seventh chord",
        "enharmonic equivalent to minor triad": "minor triad",
        "enharmonic to dominant seventh chord": "dominant seventh chord",
    }
)
_DLC_QUALITY_PROJECTION = MappingProxyType(
    {
        "%7": "half-diminished seventh chord", "+": "augmented triad",
        "+7": "augmented seventh chord", "+M7": "augmented major tetrachord",
        "Fr": "French augmented sixth chord", "Ger": "German augmented sixth chord",
        "It": "Italian augmented sixth chord", "M": "major triad",
        "MM7": "major seventh chord", "Mm7": "dominant seventh chord",
        "m": "minor triad", "mM7": "minor-augmented tetrachord",
        "mm7": "minor seventh chord", "o": "diminished triad",
        "o7": "diminished seventh chord",
    }
)
_INVERSION_PROJECTION = MappingProxyType(
    {
        _AN_INVERSION_TASK: MappingProxyType(
            {"0": "root", "1": "first", "2": "second", "3": "third"}
        ),
        _DLC_INVERSION_TASK: MappingProxyType({
            "2": "third", "43": "second", "6": "first", "64": "second",
            "65": "first", "7": "root",
        }),
    }
)
_PITCH_RE = re.compile(r"^([A-Ga-g])((?:#{1,3}|-{1,3}|b{1,3})?)$")
_NATURAL_PC = MappingProxyType(
    {"A": 9, "B": 11, "C": 0, "D": 2, "E": 4, "F": 5, "G": 7}
)


def _expected_pitch_class(source_task_id: str, source_value: object) -> int | None:
    if not isinstance(source_value, str) or not source_value:
        return None
    if source_task_id in {_AN_ROOT_TASK, _AN_BASS_TASK}:
        match = _PITCH_RE.fullmatch(source_value)
        if match is None:
            return None
        step, accidental = match.groups()
        alteration = accidental.count("#") - accidental.count("-") - accidental.count("b")
        return (_NATURAL_PC[step.upper()] + alteration) % 12
    try:
        tpc = int(source_value)
    except ValueError:
        return None
    if str(tpc) != source_value and source_value != f"+{tpc}":
        return None
    return (7 * tpc) % 12


def _validate_approved_projection_row(
    source: SourceValueIdentity,
    common_task: str,
    mapping_state: ProjectionMappingState,
    projected: JsonValue | None,
) -> None:
    allowed_sources = _DILEMMADATA_COMMON_SOURCES.get(common_task)
    if allowed_sources is None or source.source_task_id not in allowed_sources:
        raise EDAContractError(
            "eda.projection.registry_row_unknown",
            "source/common task pair is absent from the approved registry",
        )
    expected_dialect = "an_joint" if source.source_task_id.startswith("dilemmadata.an.") else "dlc"
    if source.dialect != expected_dialect:
        raise EDAContractError(
            "eda.projection.registry_row_unknown",
            "source dialect does not match the approved registry row",
        )
    source_value = source.source_value
    expected: object | None = None
    expected_state: ProjectionMappingState | None = None
    if common_task == _COMMON_QUALITY_TASK:
        if not isinstance(source_value, str) or not source_value:
            expected_state = ProjectionMappingState.INVALID
        elif source.source_task_id == _AN_QUALITY_TASK and source_value in _AN_QUALITY_VALUES:
            expected = _QUALITY_COLLAPSES.get(source_value, source_value)
            expected_state = (
                ProjectionMappingState.COARSENED
                if source_value in _QUALITY_COLLAPSES
                else ProjectionMappingState.EXACT
            )
        elif source.source_task_id == _DLC_QUALITY_TASK and source_value in _DLC_QUALITY_PROJECTION:
            expected = _DLC_QUALITY_PROJECTION[source_value]
            expected_state = ProjectionMappingState.EXACT
        else:
            expected_state = ProjectionMappingState.UNSUPPORTED
    elif common_task == _COMMON_INVERSION_TASK:
        if not isinstance(source_value, str) or not source_value:
            expected_state = ProjectionMappingState.INVALID
        else:
            expected = _INVERSION_PROJECTION[source.source_task_id].get(source_value)
            expected_state = (
                ProjectionMappingState.EXACT
                if expected is not None
                else ProjectionMappingState.UNSUPPORTED
            )
    elif common_task in {_COMMON_ROOT_PC_TASK, _COMMON_BASS_PC_TASK}:
        expected = _expected_pitch_class(source.source_task_id, source_value)
        if expected is not None:
            if isinstance(projected, bool):
                raise EDAContractError(
                    "eda.projection.value_type_invalid",
                    "pitch-class projection must be an integer, never boolean",
                )
            expected_state = ProjectionMappingState.EXACT
        elif not isinstance(source_value, str) or not source_value:
            expected_state = ProjectionMappingState.INVALID
        elif source.source_task_id in {_AN_ROOT_TASK, _AN_BASS_TASK}:
            expected_state = ProjectionMappingState.UNSUPPORTED
        else:
            expected_state = ProjectionMappingState.INVALID
    elif common_task == _COMMON_LOCAL_KEY_TASK:
        if (
            isinstance(projected, dict)
            and set(projected) == {"mode", "tonic_pc"}
            and isinstance(projected["mode"], str)
            and projected["mode"] in {"major", "minor", "unknown", "other"}
            and isinstance(projected["tonic_pc"], int)
            and not isinstance(projected["tonic_pc"], bool)
            and 0 <= projected["tonic_pc"] <= 11
        ):
            expected = projected
            expected_state = ProjectionMappingState.EXACT
        elif projected is None and mapping_state in {
            ProjectionMappingState.AMBIGUOUS,
            ProjectionMappingState.INVALID,
        }:
            expected_state = mapping_state
    elif common_task == _COMMON_PITCH_CLASS_SET_TASK:
        if (
            isinstance(projected, list)
            and projected
            and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 11 for item in projected)
            and projected == sorted(set(projected))
        ):
            expected = projected
            expected_state = mapping_state
            if mapping_state not in {
                ProjectionMappingState.EXACT,
                ProjectionMappingState.COARSENED,
            }:
                expected_state = None
        elif projected is None and mapping_state in {
            ProjectionMappingState.AMBIGUOUS,
            ProjectionMappingState.UNSUPPORTED,
            ProjectionMappingState.INVALID,
        }:
            expected_state = mapping_state
    if expected is None or projected != _jsonable(expected) or mapping_state != expected_state:
        if expected is None and projected is None and mapping_state == expected_state:
            return
        raise EDAContractError(
            "eda.projection.registry_row_unknown",
            "source value, mapping state, or projected value is absent from the approved registry",
        )


@dataclass(frozen=True, slots=True)
class ProjectionEvidence:
    source_value: SourceValueIdentity
    mapping_registry: VersionedIdentity
    common_task_identity: str
    native_state: AvailabilityState
    mapping_state: ProjectionMappingState
    projected_value: object | None
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_value, SourceValueIdentity):
            raise EDAContractError(
                "eda.projection.source_value_invalid",
                "projection source_value must be a SourceValueIdentity",
            )
        if not isinstance(self.mapping_registry, VersionedIdentity):
            raise EDAContractError(
                "eda.projection.registry_invalid",
                "projection mapping_registry must be a VersionedIdentity",
            )
        _identifier(self.common_task_identity, "common_task_identity")
        object.__setattr__(
            self, "native_state", _enum(self.native_state, AvailabilityState, "native_state")
        )
        object.__setattr__(
            self,
            "mapping_state",
            _enum(self.mapping_state, ProjectionMappingState, "mapping_state"),
        )
        approved = APPROVED_PROJECTION_REGISTRIES.get(self.mapping_registry.identity)
        if approved != self.mapping_registry:
            raise EDAContractError(
                "eda.projection.registry_unapproved",
                "projection requires an exact existing approved registry identity/fingerprint",
            )
        if self.source_value.corpus != CorpusId.DILEMMADATA:
            raise EDAContractError(
                "eda.projection.registry_corpus_mismatch",
                "the current approved registry applies only to Dilemmadata",
            )
        projected = None if self.projected_value is None else _jsonable(
            self.projected_value, path="$.projected_value"
        )
        if self.native_state != AvailabilityState.AVAILABLE:
            raise EDAContractError(
                "eda.projection.native_state_invalid",
                "projection rows describe available native class support only",
            )
        if self.mapping_state in {
            ProjectionMappingState.MISSING,
            ProjectionMappingState.MASKED,
        }:
            raise EDAContractError(
                "eda.projection.native_state_mismatch",
                "missing/masked are aggregate projection availability states, not class rows",
            )
        _validate_approved_projection_row(
            self.source_value,
            self.common_task_identity,
            self.mapping_state,
            projected,
        )
        if self.mapping_state in {
            ProjectionMappingState.EXACT,
            ProjectionMappingState.COARSENED,
        }:
            if projected is None:
                raise EDAContractError(
                    "eda.projection.available_value_missing",
                    "exact/coarsened projection requires available native and projected values",
                )
        elif projected is not None:
            raise EDAContractError(
                "eda.projection.unavailable_value_present",
                "non-supervising projection states must not expose a projected value",
            )
        object.__setattr__(
            self,
            "projected_value",
            None if projected is None else _freeze_json(projected),
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "projection provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "projection evidence requires provenance"
            )


@dataclass(frozen=True, slots=True)
class ProjectionAvailabilityCounts:
    """A projection-state partition kept separate from native availability."""

    corpus: CorpusId
    source_task_id: str
    dialect: str
    mapping_registry: VersionedIdentity
    common_task_identity: str
    observation_unit: ObservationUnit
    denominator: int
    exact: int
    coarsened: int
    ambiguous: int
    unsupported: int
    invalid: int
    missing: int
    masked: int
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        _identifier(self.source_task_id, "projection availability source_task_id")
        _identifier(self.dialect, "projection availability dialect")
        _identifier(self.common_task_identity, "projection availability common_task_identity")
        if not isinstance(self.mapping_registry, VersionedIdentity):
            raise EDAContractError(
                "eda.projection.registry_invalid",
                "projection availability registry must be a VersionedIdentity",
            )
        approved = APPROVED_PROJECTION_REGISTRIES.get(self.mapping_registry.identity)
        if approved != self.mapping_registry:
            raise EDAContractError(
                "eda.projection.registry_unapproved",
                "projection availability requires an exact approved registry binding",
            )
        if self.corpus != CorpusId.DILEMMADATA:
            raise EDAContractError(
                "eda.projection.registry_corpus_mismatch",
                "the current approved registry applies only to Dilemmadata",
            )
        allowed_sources = _DILEMMADATA_COMMON_SOURCES.get(self.common_task_identity)
        expected_dialect = (
            "an_joint" if self.source_task_id.startswith("dilemmadata.an.") else "dlc"
        )
        if (
            allowed_sources is None
            or self.source_task_id not in allowed_sources
            or self.dialect != expected_dialect
        ):
            raise EDAContractError(
                "eda.projection.registry_row_unknown",
                "projection availability source/task/dialect is absent from the approved registry",
            )
        object.__setattr__(
            self,
            "observation_unit",
            _enum(self.observation_unit, ObservationUnit, "projection observation_unit"),
        )
        if self.observation_unit != ObservationUnit.TARGET_ROW:
            raise EDAContractError(
                "eda.projection.availability_unit_invalid",
                "projection availability is counted in target rows",
            )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        if self.split_scope not in {SplitScope.TRAIN, SplitScope.VALIDATION}:
            raise EDAContractError(
                "eda.test_lock.supervision_split_forbidden",
                "projection availability may contain only TRAIN or VALIDATION",
            )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        if self.evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE}:
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "projection availability requires observed evidence",
            )
        counts = tuple(
            _integer(value, name)
            for value, name in (
                (self.denominator, "projection availability denominator"),
                (self.exact, "projection exact"),
                (self.coarsened, "projection coarsened"),
                (self.ambiguous, "projection ambiguous"),
                (self.unsupported, "projection unsupported"),
                (self.invalid, "projection invalid"),
                (self.missing, "projection missing"),
                (self.masked, "projection masked"),
            )
        )
        if sum(value or 0 for value in counts[1:]) != counts[0]:
            raise EDAContractError(
                "eda.projection.availability_denominator_mismatch",
                "all projection states must partition the target-row denominator",
            )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "projection availability provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing",
                "projection availability requires provenance",
            )


@dataclass(frozen=True, slots=True)
class TaskFamilyEvidence:
    corpus: CorpusId
    source_task_id: str
    dialect: str
    annotation_namespace: str
    vocabulary: VersionedIdentity
    label_granularity: str
    label_value_type: LabelValueType
    observation_unit: ObservationUnit
    split_scope: SplitScope
    evidence_scope: EvidenceScope
    provenance: tuple[str, ...]
    status: ComputationStatus
    availability: AvailabilityCounts | None
    work_identity: VersionedIdentity | None = None
    class_support: tuple[ClassSupport, ...] = ()
    empty_multilabel_available_count: UnitCount | None = None
    projection_availability: tuple[ProjectionAvailabilityCounts, ...] = ()
    projections: tuple[ProjectionEvidence, ...] = ()
    reason_code: str | None = None

    def __post_init__(self) -> None:
        class_support_input = _tuple_collection(
            self.class_support, "class_support"
        )
        projection_availability_input = _tuple_collection(
            self.projection_availability, "projection_availability"
        )
        projections_input = _tuple_collection(self.projections, "projections")
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        _identifier(self.source_task_id, "source_task_id")
        _identifier(self.dialect, "dialect")
        _identifier(self.annotation_namespace, "annotation_namespace")
        if not isinstance(self.vocabulary, VersionedIdentity):
            raise EDAContractError(
                "eda.supervision.vocabulary_invalid",
                "task vocabulary must be a VersionedIdentity",
            )
        _identifier(self.label_granularity, "label_granularity")
        object.__setattr__(
            self,
            "label_value_type",
            _enum(self.label_value_type, LabelValueType, "label_value_type"),
        )
        object.__setattr__(
            self,
            "observation_unit",
            _enum(self.observation_unit, ObservationUnit, "observation_unit"),
        )
        if self.observation_unit != ObservationUnit.TARGET_ROW:
            raise EDAContractError(
                "eda.supervision.observation_unit_invalid",
                "native task availability uses target_row observation units",
            )
        object.__setattr__(
            self, "split_scope", _enum(self.split_scope, SplitScope, "split_scope")
        )
        if self.split_scope not in {SplitScope.TRAIN, SplitScope.VALIDATION}:
            raise EDAContractError(
                "eda.test_lock.supervision_split_forbidden",
                "supervision task evidence may contain only TRAIN or VALIDATION",
            )
        object.__setattr__(
            self,
            "evidence_scope",
            _enum(self.evidence_scope, EvidenceScope, "evidence_scope"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ComputationStatus, "status")
        )
        object.__setattr__(
            self,
            "provenance",
            _sorted_unique_strings(self.provenance, "task provenance"),
        )
        if not self.provenance:
            raise EDAContractError(
                "eda.provenance.missing", "task evidence requires provenance"
            )
        if self.availability is not None and not isinstance(
            self.availability, AvailabilityCounts
        ):
            raise EDAContractError(
                "eda.availability.type_invalid",
                "task availability must be AvailabilityCounts or null",
            )
        if self.work_identity is not None and not isinstance(
            self.work_identity, VersionedIdentity
        ):
            raise EDAContractError(
                "eda.class_support.work_identity_invalid",
                "task work_identity must be a VersionedIdentity",
            )
        if self.evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE} and (
            self.status == ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "unknown/unavailable evidence cannot contain observed supervision",
            )
        if any(not isinstance(item, ClassSupport) for item in class_support_input):
            raise EDAContractError(
                "eda.class_support.type_invalid",
                "class_support must contain ClassSupport objects",
            )
        support = tuple(
            sorted(class_support_input, key=lambda item: item.source_value.identity)
        )
        if len({item.source_value.identity for item in support}) != len(support):
            raise EDAContractError(
                "eda.class_support.duplicate", "class support identities must be unique"
            )
        if any(
            not isinstance(item, ProjectionAvailabilityCounts)
            for item in projection_availability_input
        ):
            raise EDAContractError(
                "eda.projection.availability_type_invalid",
                "projection_availability must contain ProjectionAvailabilityCounts objects",
            )
        projection_availability = tuple(
            sorted(
                projection_availability_input,
                key=lambda item: (
                    item.mapping_registry.identity,
                    item.common_task_identity,
                ),
            )
        )
        projection_availability_keys = tuple(
            (item.mapping_registry.identity, item.common_task_identity)
            for item in projection_availability
        )
        if len(projection_availability_keys) != len(set(projection_availability_keys)):
            raise EDAContractError(
                "eda.projection.availability_duplicate",
                "projection availability rows must be unique per registry/common task",
            )
        if any(not isinstance(item, ProjectionEvidence) for item in projections_input):
            raise EDAContractError(
                "eda.projection.type_invalid",
                "projections must contain ProjectionEvidence objects",
            )
        projections = tuple(
            sorted(
                projections_input,
                key=lambda item: (
                    item.source_value.identity,
                    item.mapping_registry.identity,
                    item.common_task_identity,
                ),
            )
        )
        projection_keys = tuple(
            (
                item.source_value.identity,
                item.mapping_registry.identity,
                item.common_task_identity,
            )
            for item in projections
        )
        if len(projection_keys) != len(set(projection_keys)):
            raise EDAContractError(
                "eda.projection.duplicate", "projection evidence rows must be unique"
            )
        object.__setattr__(self, "class_support", support)
        object.__setattr__(self, "projection_availability", projection_availability)
        object.__setattr__(self, "projections", projections)
        if self.status == ComputationStatus.OBSERVED:
            if self.availability is None:
                raise EDAContractError(
                    "eda.availability.missing", "observed task requires availability counts"
                )
            if (
                self.availability.observation_unit != self.observation_unit
                or self.availability.split_scope != self.split_scope
                or self.availability.evidence_scope != self.evidence_scope
            ):
                raise EDAContractError(
                    "eda.availability.binding_mismatch",
                    "task and availability units/scopes must agree",
                )
            for item in support:
                if (
                    item.source_value.corpus != self.corpus
                    or item.source_value.source_task_id != self.source_task_id
                    or item.source_value.dialect != self.dialect
                    or item.occurrence_count.denominator != self.availability.available
                    or item.occurrence_count.split_scope != self.split_scope
                    or item.occurrence_count.evidence_scope != self.evidence_scope
                ):
                    raise EDAContractError(
                        "eda.class_support.binding_mismatch",
                        "class support must bind the exact native task and available-row denominator",
                    )
            observed_work_support = any(
                item.unique_work_count.status == ComputationStatus.OBSERVED
                for item in support
            )
            work_unit_pairs = {
                (
                    item.unique_work_count.observation_unit,
                    item.unique_work_count.denominator_unit,
                )
                for item in support
            }
            if len(work_unit_pairs) > 1:
                raise EDAContractError(
                    "eda.class_support.work_unit_mixed",
                    "all class-support rows in one task must use one work unit",
                )
            record_denominators = {
                item.unique_record_count.denominator for item in support
            }
            support_provenance = {
                item.occurrence_count.provenance for item in support
            }
            work_populations = {
                (
                    item.unique_work_count.observation_unit,
                    item.unique_work_count.denominator,
                    item.unique_work_count.status,
                    item.unique_work_count.reason_code,
                )
                for item in support
            }
            if (
                len(record_denominators) > 1
                or len(work_populations) > 1
                or len(support_provenance) > 1
            ):
                raise EDAContractError(
                    "eda.class_support.population_mixed",
                    "all class-support rows must share record and work comparison populations",
                )
            if observed_work_support and self.work_identity is None:
                raise EDAContractError(
                    "eda.class_support.work_identity_unproven",
                    "observed unique-work support requires a versioned work identity",
                )
            if self.work_identity is None and any(
                item.unique_work_count.status != ComputationStatus.NOT_APPLICABLE
                or item.unique_work_count.value is not None
                or item.unique_work_count.denominator is not None
                for item in support
            ):
                raise EDAContractError(
                    "eda.class_support.work_identity_unproven",
                    "without a work identity every work count and denominator must be not_applicable and null",
                )
            support_by_identity = {
                item.source_value.identity: item for item in support
            }
            for item in projections:
                if item.native_state != AvailabilityState.AVAILABLE or (
                    item.source_value.identity not in support_by_identity
                ):
                    raise EDAContractError(
                        "eda.projection.source_support_missing",
                        "projection source must be an available identity in native class support",
                    )
            availability_by_key = {
                (item.mapping_registry.identity, item.common_task_identity): item
                for item in projection_availability
            }
            projection_keys_present = {
                (item.mapping_registry.identity, item.common_task_identity)
                for item in projections
            }
            if not projection_keys_present.issubset(availability_by_key):
                raise EDAContractError(
                    "eda.projection.availability_binding_missing",
                    "every emitted class-mapping family requires a separate projection availability partition",
                )
            for projection_counts in availability_by_key.values():
                if (
                    projection_counts.corpus != self.corpus
                    or projection_counts.source_task_id != self.source_task_id
                    or projection_counts.dialect != self.dialect
                    or projection_counts.observation_unit != self.observation_unit
                    or projection_counts.denominator != self.availability.denominator
                    or projection_counts.split_scope != self.split_scope
                    or projection_counts.evidence_scope != self.evidence_scope
                ):
                    raise EDAContractError(
                        "eda.projection.availability_binding_mismatch",
                        "projection and native availability must share the exact task, denominator, and evidence scope",
                    )
            if self.label_value_type == LabelValueType.CATEGORICAL:
                if self.empty_multilabel_available_count is not None:
                    raise EDAContractError(
                        "eda.multilabel.count_unexpected",
                        "categorical task cannot contain an empty-multilabel count",
                    )
                if any(
                    item.source_value.value_kind != SourceValueKind.SCALAR
                    for item in support
                ):
                    raise EDAContractError(
                        "eda.source_value.kind_mismatch",
                        "categorical class support requires scalar source identities",
                    )
                total = sum(item.occurrence_count.value or 0 for item in support)
                if total != self.availability.available:
                    raise EDAContractError(
                        "eda.class_support.available_mismatch",
                        "categorical occurrence support must equal available rows",
                    )
            else:
                empty_count = self.empty_multilabel_available_count
                if not isinstance(empty_count, UnitCount):
                    raise EDAContractError(
                        "eda.multilabel.count_type_invalid",
                        "multilabel task requires a typed empty-multilabel UnitCount",
                    )
                if (
                    empty_count.name != "empty_multilabel_available_count"
                    or empty_count.observation_unit != ObservationUnit.TARGET_ROW
                    or empty_count.denominator_unit != ObservationUnit.TARGET_ROW
                    or empty_count.denominator != self.availability.available
                    or empty_count.split_scope != self.split_scope
                    or empty_count.evidence_scope != self.evidence_scope
                    or empty_count.provenance != self.provenance
                    or empty_count.status != ComputationStatus.OBSERVED
                    or empty_count.value is None
                ):
                    raise EDAContractError(
                        "eda.multilabel.count_binding_mismatch",
                        "empty-multilabel count must bind the task's available-row denominator and exact scope",
                    )
                empty = empty_count.value
                if empty > self.availability.available:
                    raise EDAContractError(
                        "eda.multilabel.empty_count_invalid",
                        "available empty multilabel rows cannot exceed all available rows",
                    )
                if any(
                    item.source_value.value_kind != SourceValueKind.SCALAR
                    or not isinstance(item.source_value.source_value, str)
                    or not item.source_value.source_value
                    or item.source_value.source_value
                    != item.source_value.source_value.strip()
                    for item in support
                ):
                    raise EDAContractError(
                        "eda.multilabel.class_identity_invalid",
                        "multilabel class support uses one non-empty scalar string identity per vocabulary label; empty sets are counted separately",
                    )
                occurrence_total = sum(
                    item.occurrence_count.value or 0 for item in support
                )
                nonempty_rows = self.availability.available - empty
                if any(
                    (item.occurrence_count.value or 0) > nonempty_rows
                    for item in support
                ):
                    raise EDAContractError(
                        "eda.multilabel.class_occurrence_exceeded",
                        "one vocabulary label cannot occur more than once per non-empty available row",
                    )
                if occurrence_total < nonempty_rows or (
                    nonempty_rows == 0 and occurrence_total != 0
                ):
                    raise EDAContractError(
                        "eda.multilabel.occurrence_count_invalid",
                        "each non-empty available multilabel row requires at least one occurrence",
                    )
            if self.reason_code is not None:
                raise EDAContractError(
                    "eda.reason.unexpected", "observed task cannot have unavailable reason"
                )
        else:
            if (
                self.availability is not None
                or support
                or projection_availability
                or projections
            ):
                raise EDAContractError(
                    "eda.supervision.unavailable_payload_present",
                    "non-observed task must not fabricate availability or class/projection support",
                )
            if self.empty_multilabel_available_count is not None:
                raise EDAContractError(
                    "eda.multilabel.unavailable_count_present",
                    "non-observed task must not encode empty multilabel as zero",
                )
            _identifier(self.reason_code, "task reason_code")


@dataclass(frozen=True, slots=True)
class TestTargetLockEvidence:
    test_assignment_count: UnitCount
    assignment_manifest_fingerprint: str | None
    test_descriptor_resolution_count: UnitCount
    test_target_loader_call_count: UnitCount
    test_target_records_opened: UnitCount
    test_target_rows_loaded: UnitCount
    assignment_gate_before_descriptor_resolution: bool = True
    assignment_gate_before_target_open: bool = True
    test_targets_read: bool = False
    test_targets_used_for_eda: bool = False
    test_targets_used_for_model_evaluation: bool = False
    test_class_distributions_emitted: bool = False
    test_coverage_emitted: bool = False
    test_cooccurrence_emitted: bool = False
    contract_version: str = EDA_TEST_TARGET_LOCK_VERSION
    gate_order: str = "split_assignment_before_descriptor_resolution_or_target_open"

    @classmethod
    def from_guard(
        cls,
        *,
        test_assignment_count: int,
        assignment_manifest_fingerprint: str,
        evidence_scope: EvidenceScope,
        provenance: tuple[str, ...],
    ) -> "TestTargetLockEvidence":
        """Build the five typed TEST audit counters for a completed split gate."""

        common = {
            "denominator": test_assignment_count,
            "denominator_unit": ObservationUnit.SPLIT_ASSIGNMENT,
            "split_scope": SplitScope.TEST,
            "evidence_scope": evidence_scope,
            "provenance": provenance,
        }
        return cls(
            test_assignment_count=UnitCount(
                name="test_assignment_count",
                observation_unit=ObservationUnit.SPLIT_ASSIGNMENT,
                value=test_assignment_count,
                **common,
            ),
            assignment_manifest_fingerprint=assignment_manifest_fingerprint,
            test_descriptor_resolution_count=UnitCount(
                name="test_descriptor_resolution_count",
                observation_unit=ObservationUnit.TARGET_ACCESS_ATTEMPT,
                value=0,
                **common,
            ),
            test_target_loader_call_count=UnitCount(
                name="test_target_loader_call_count",
                observation_unit=ObservationUnit.TARGET_ACCESS_ATTEMPT,
                value=0,
                **common,
            ),
            test_target_records_opened=UnitCount(
                name="test_target_records_opened",
                observation_unit=ObservationUnit.RECORD,
                value=0,
                **common,
            ),
            test_target_rows_loaded=UnitCount(
                name="test_target_rows_loaded",
                observation_unit=ObservationUnit.TARGET_ROW,
                value=0,
                **common,
            ),
        )

    @classmethod
    def not_executed(
        cls,
        *,
        evidence_scope: EvidenceScope,
        provenance: tuple[str, ...],
        reason_code: str,
        assignment_manifest_fingerprint: str | None = None,
        test_assignment_denominator: int | None = None,
    ) -> "TestTargetLockEvidence":
        """Represent a locked/non-executed gate without fabricating zero counts."""

        common = {
            "value": None,
            "denominator": test_assignment_denominator,
            "denominator_unit": ObservationUnit.SPLIT_ASSIGNMENT,
            "split_scope": SplitScope.TEST,
            "evidence_scope": evidence_scope,
            "provenance": provenance,
            "status": ComputationStatus.LOCKED,
            "reason_code": reason_code,
        }
        return cls(
            test_assignment_count=UnitCount(
                name="test_assignment_count",
                observation_unit=ObservationUnit.SPLIT_ASSIGNMENT,
                **common,
            ),
            assignment_manifest_fingerprint=assignment_manifest_fingerprint,
            test_descriptor_resolution_count=UnitCount(
                name="test_descriptor_resolution_count",
                observation_unit=ObservationUnit.TARGET_ACCESS_ATTEMPT,
                **common,
            ),
            test_target_loader_call_count=UnitCount(
                name="test_target_loader_call_count",
                observation_unit=ObservationUnit.TARGET_ACCESS_ATTEMPT,
                **common,
            ),
            test_target_records_opened=UnitCount(
                name="test_target_records_opened",
                observation_unit=ObservationUnit.RECORD,
                **common,
            ),
            test_target_rows_loaded=UnitCount(
                name="test_target_rows_loaded",
                observation_unit=ObservationUnit.TARGET_ROW,
                **common,
            ),
        )

    def __post_init__(self) -> None:
        typed_counts = (
            self.test_assignment_count,
            self.test_descriptor_resolution_count,
            self.test_target_loader_call_count,
            self.test_target_records_opened,
            self.test_target_rows_loaded,
        )
        if any(not isinstance(item, UnitCount) for item in typed_counts):
            raise EDAContractError(
                "eda.test_lock.count_type_invalid",
                "every TEST-lock counter must be a UnitCount",
            )
        statuses = {item.status for item in typed_counts}
        if len(statuses) != 1 or statuses not in (
            {ComputationStatus.OBSERVED},
            {ComputationStatus.LOCKED},
        ):
            raise EDAContractError(
                "eda.test_lock.count_binding_invalid",
                "TEST-lock counters must share observed or locked status",
            )
        observed = statuses == {ComputationStatus.OBSERVED}
        if observed:
            _sha256(
                self.assignment_manifest_fingerprint,
                "TEST-lock assignment manifest fingerprint",
            )
            if any(item.value is None for item in typed_counts):
                raise EDAContractError(
                    "eda.test_lock.count_binding_invalid",
                    "observed TEST-lock counters require non-null values",
                )
        else:
            if self.assignment_manifest_fingerprint is not None:
                _sha256(
                    self.assignment_manifest_fingerprint,
                    "TEST-lock assignment manifest fingerprint",
                )
            if any(item.value is not None for item in typed_counts):
                raise EDAContractError(
                    "eda.test_lock.count_binding_invalid",
                    "locked TEST-lock counters must remain null",
                )
        expected_names = (
            "test_assignment_count",
            "test_descriptor_resolution_count",
            "test_target_loader_call_count",
            "test_target_records_opened",
            "test_target_rows_loaded",
        )
        assignment_denominator = (
            self.test_assignment_count.value
            if observed
            else self.test_assignment_count.denominator
        )
        for index, (count, expected_name) in enumerate(
            zip(typed_counts, expected_names, strict=True)
        ):
            expected_unit = {
                "test_assignment_count": ObservationUnit.SPLIT_ASSIGNMENT,
                "test_descriptor_resolution_count": ObservationUnit.TARGET_ACCESS_ATTEMPT,
                "test_target_loader_call_count": ObservationUnit.TARGET_ACCESS_ATTEMPT,
                "test_target_records_opened": ObservationUnit.RECORD,
                "test_target_rows_loaded": ObservationUnit.TARGET_ROW,
            }[expected_name]
            if (
                count.name != expected_name
                or count.observation_unit != expected_unit
                or count.denominator != assignment_denominator
                or count.denominator_unit != ObservationUnit.SPLIT_ASSIGNMENT
                or count.split_scope != SplitScope.TEST
                or count.evidence_scope != self.test_assignment_count.evidence_scope
                or count.provenance != self.test_assignment_count.provenance
                or count.reason_code != self.test_assignment_count.reason_code
                or count.status not in statuses
                or (
                    observed
                    and index == 0
                    and count.value != assignment_denominator
                )
                or (observed and index > 0 and count.value != 0)
            ):
                raise EDAContractError(
                    "eda.test_lock.count_binding_invalid",
                    "TEST-lock counts must bind one TEST-assignment denominator, evidence scope, and provenance",
                )
        if self.contract_version != EDA_TEST_TARGET_LOCK_VERSION:
            raise EDAContractError(
                "eda.test_lock.version_invalid", "unsupported TEST-target lock contract"
            )
        if self.gate_order != "split_assignment_before_descriptor_resolution_or_target_open":
            raise EDAContractError(
                "eda.test_lock.order_invalid", "TEST gate must precede descriptor/path/target access"
            )
        if self.assignment_gate_before_descriptor_resolution is not True or (
            self.assignment_gate_before_target_open is not True
        ):
            raise EDAContractError(
                "eda.test_lock.gate_missing", "both pre-access TEST gates are mandatory"
            )
        false_fields = (
            self.test_targets_read,
            self.test_targets_used_for_eda,
            self.test_targets_used_for_model_evaluation,
            self.test_class_distributions_emitted,
            self.test_coverage_emitted,
            self.test_cooccurrence_emitted,
        )
        if any(value is not False for value in false_fields):
            raise EDAContractError(
                "eda.test_lock.violation",
                "TEST supervision access/distributions/coverage/co-occurrence must remain zero/false",
            )


@dataclass(frozen=True, slots=True)
class SupervisionEDAPayload:
    tasks: tuple[TaskFamilyEvidence, ...]
    test_lock: TestTargetLockEvidence
    extensions: tuple[SourceExtension, ...] = ()

    def __post_init__(self) -> None:
        tasks_input = _tuple_collection(self.tasks, "supervision tasks")
        extensions_input = _tuple_collection(
            self.extensions, "supervision extensions"
        )
        if any(not isinstance(item, TaskFamilyEvidence) for item in tasks_input):
            raise EDAContractError(
                "eda.supervision.task_type_invalid",
                "tasks must contain TaskFamilyEvidence objects",
            )
        if not isinstance(self.test_lock, TestTargetLockEvidence):
            raise EDAContractError(
                "eda.test_lock.type_invalid",
                "test_lock must be TestTargetLockEvidence",
            )
        if any(not isinstance(item, SourceExtension) for item in extensions_input):
            raise EDAContractError(
                "eda.extension.type_invalid",
                "extensions must contain SourceExtension objects",
            )
        tasks = tuple(
            sorted(
                tasks_input,
                key=lambda item: (
                    item.corpus.value,
                    item.source_task_id,
                    item.dialect,
                    item.split_scope.value,
                ),
            )
        )
        keys = tuple(
            (item.corpus, item.source_task_id, item.dialect, item.split_scope)
            for item in tasks
        )
        if len(keys) != len(set(keys)):
            raise EDAContractError(
                "eda.supervision.task_duplicate", "task/dialect/split rows must be unique"
            )
        family_schemas: dict[tuple[CorpusId, str, str], tuple[object, ...]] = {}
        family_work_units: dict[
            tuple[CorpusId, str, str], ObservationUnit
        ] = {}
        for item in tasks:
            family_key = (item.corpus, item.source_task_id, item.dialect)
            family_schema = (
                item.annotation_namespace,
                item.vocabulary,
                item.label_granularity,
                item.label_value_type,
                item.observation_unit,
                item.work_identity,
            )
            previous = family_schemas.setdefault(family_key, family_schema)
            if previous != family_schema:
                raise EDAContractError(
                    "eda.supervision.task_schema_mismatch",
                    "a source task/dialect must keep one schema identity across splits",
                )
            if item.class_support:
                work_unit = item.class_support[0].unique_work_count.observation_unit
                previous_work_unit = family_work_units.setdefault(
                    family_key, work_unit
                )
                if previous_work_unit != work_unit:
                    raise EDAContractError(
                        "eda.supervision.task_schema_mismatch",
                        "a source task/dialect must keep one class-support work unit across splits",
                    )
        if not tasks:
            raise EDAContractError(
                "eda.supervision.tasks_empty", "a supervision report requires task evidence"
            )
        object.__setattr__(self, "tasks", tasks)
        extensions = tuple(
            sorted(
                extensions_input,
                key=lambda item: (item.namespace, item.split_scope.value),
            )
        )
        extension_keys = {
            (item.namespace, item.split_scope) for item in extensions
        }
        if len(extension_keys) != len(extensions):
            raise EDAContractError(
                "eda.extension.namespace_duplicate",
                "extension namespace/split pairs must be unique",
            )
        _validate_extension_schemas_across_splits(extensions)
        object.__setattr__(self, "extensions", extensions)


@dataclass(frozen=True, slots=True)
class CorpusEDACapability:
    corpus: CorpusId
    raw_corpus_eda: bool
    supervision_eda: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus", _enum(self.corpus, CorpusId, "corpus"))
        if self.raw_corpus_eda is not True:
            raise EDAContractError(
                "eda.capability.raw_required", "all four registered corpora support RawCorpusEDA"
            )
        expected_supervision = self.corpus != CorpusId.PDMX
        if self.supervision_eda is not expected_supervision:
            raise EDAContractError(
                "eda.capability.matrix_invalid",
                "supervision capability must be true for the three labeled corpora and false for PDMX",
            )


EDA_CAPABILITIES = MappingProxyType(
    {
        corpus: CorpusEDACapability(
            corpus=corpus,
            raw_corpus_eda=True,
            supervision_eda=corpus != CorpusId.PDMX,
        )
        for corpus in CorpusId
    }
)


def corpus_eda_capability(corpus: CorpusId | str) -> CorpusEDACapability:
    """Return the frozen capability row for one supported corpus."""

    normalized = _enum(corpus, CorpusId, "corpus")
    assert isinstance(normalized, CorpusId)
    return EDA_CAPABILITIES[normalized]


def capability_registry_dict() -> dict[str, object]:
    return {
        "schema_name": EDA_CAPABILITY_REGISTRY_SCHEMA_NAME,
        "schema_version": EDA_CAPABILITY_REGISTRY_SCHEMA_VERSION,
        "capabilities": [
            _jsonable(EDA_CAPABILITIES[corpus]) for corpus in sorted(CorpusId, key=lambda item: item.value)
        ],
    }


def capability_registry_fingerprint() -> str:
    return canonical_json_sha256(capability_registry_dict())


def _envelope_semantic_dict(envelope: ReportEnvelope) -> dict[str, JsonValue]:
    value = _jsonable(envelope)
    assert isinstance(value, dict)
    value.pop("operational_metadata")
    return value


def raw_report_semantic_dict(report: "RawCorpusEDA") -> dict[str, object]:
    return {
        "envelope": _envelope_semantic_dict(report.envelope),
        "semantic_payload": _jsonable(report.semantic_payload),
    }


def supervision_report_semantic_dict(report: "SupervisionEDA") -> dict[str, object]:
    return {
        "envelope": _envelope_semantic_dict(report.envelope),
        "semantic_payload": _jsonable(report.semantic_payload),
    }


def _used_raw_units(payload: RawCorpusEDAPayload) -> set[ObservationUnit]:
    units: set[ObservationUnit] = set()
    for metric in payload.metrics:
        units.add(metric.coverage.observation_unit)
        if metric.count is not None:
            units.update((metric.count.observation_unit, metric.count.denominator_unit))
        for category in metric.categories:
            units.update(
                (category.count.observation_unit, category.count.denominator_unit)
            )
    for extension in payload.extensions:
        for row in extension.rows:
            units.add(row.coverage.observation_unit)
            for count in row.counts:
                units.update((count.observation_unit, count.denominator_unit))
    return units


def _used_supervision_units(payload: SupervisionEDAPayload) -> set[ObservationUnit]:
    units: set[ObservationUnit] = set()
    for task in payload.tasks:
        units.add(task.observation_unit)
        for support in task.class_support:
            for count in (
                support.occurrence_count,
                support.unique_record_count,
                support.unique_work_count,
            ):
                units.update((count.observation_unit, count.denominator_unit))
        if task.empty_multilabel_available_count is not None:
            count = task.empty_multilabel_available_count
            units.update((count.observation_unit, count.denominator_unit))
    for extension in payload.extensions:
        for row in extension.rows:
            units.add(row.coverage.observation_unit)
            for count in row.counts:
                units.update((count.observation_unit, count.denominator_unit))
    for count in (
        payload.test_lock.test_assignment_count,
        payload.test_lock.test_descriptor_resolution_count,
        payload.test_lock.test_target_loader_call_count,
        payload.test_lock.test_target_records_opened,
        payload.test_lock.test_target_rows_loaded,
    ):
        units.update((count.observation_unit, count.denominator_unit))
    return units


@dataclass(frozen=True, slots=True)
class RawCorpusEDA:
    envelope: ReportEnvelope
    semantic_payload: RawCorpusEDAPayload
    semantic_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not RawCorpusEDA:
            raise EDAContractError(
                "eda.raw.type_invalid",
                "RawCorpusEDA is a closed wire schema and cannot be subclassed",
            )
        if not isinstance(self.envelope, ReportEnvelope) or not isinstance(
            self.semantic_payload, RawCorpusEDAPayload
        ):
            raise EDAContractError(
                "eda.raw.type_invalid",
                "raw report requires ReportEnvelope and RawCorpusEDAPayload",
            )
        if (
            self.envelope.schema_name != RAW_CORPUS_EDA_SCHEMA_NAME
            or self.envelope.schema_version != RAW_CORPUS_EDA_SCHEMA_VERSION
            or self.envelope.report_kind != ReportKind.RAW_CORPUS
        ):
            raise EDAContractError(
                "eda.raw.schema_invalid", "raw report must use RawCorpusEDA@1.0.0"
            )
        capability = corpus_eda_capability(self.envelope.corpus)
        if not capability.raw_corpus_eda:
            raise EDAContractError(
                "eda.capability.raw_forbidden", "corpus does not support raw EDA"
            )
        forbidden_units = set(self.envelope.observation_units) - set(
            _RAW_ALLOWED_OBSERVATION_UNITS
        )
        if forbidden_units:
            raise EDAContractError(
                "eda.raw.observation_unit_forbidden",
                "raw evidence cannot use target-access, target-row, label, or training units: "
                f"{sorted(item.value for item in forbidden_units)!r}",
            )
        if any(not manifest.target_free for manifest in self.envelope.input_manifests):
            raise EDAContractError(
                "eda.raw.manifest_target_bearing",
                "RawCorpusEDA accepts only explicitly target-free manifest projections",
            )
        for field_name, value in (
            ("source_identity", self.envelope.source_identity.identity),
            ("producer_identity", self.envelope.producer_identity.identity),
        ):
            _reject_text_tokens(
                value,
                _RAW_MANIFEST_TARGET_TOKENS,
                category="eda.raw.target_field_forbidden",
                label=f"raw envelope {field_name}",
            )
        for manifest_index, manifest in enumerate(self.envelope.input_manifests):
            fields = (
                ("role", manifest.role),
                ("identity", manifest.identity.identity),
                ("repository_relative_path", manifest.repository_relative_path),
            )
            for field_name, value in fields:
                _reject_text_tokens(
                    value,
                    _RAW_MANIFEST_TARGET_TOKENS,
                    category="eda.raw.target_manifest_forbidden",
                    label=f"raw manifest {field_name}",
                    path=f"$.envelope.input_manifests[{manifest_index}].{field_name}",
                )
        for invariant in self.envelope.invariants:
            for field_name, value in (
                ("code", invariant.code),
                ("reason_code", invariant.reason_code),
                *(("provenance", item) for item in invariant.provenance),
            ):
                if (
                    field_name == "reason_code"
                    and value == EDAReasonCode.TARGET_FREE_UNPROVEN.value
                ):
                    continue
                _reject_text_tokens(
                    value,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label=f"raw invariant {field_name}",
                )
        for warning in self.envelope.warnings:
            for field_name, value in (
                ("code", warning.code),
                ("message", warning.message),
                *(("provenance", item) for item in warning.provenance),
            ):
                _reject_text_tokens(
                    value,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label=f"raw warning {field_name}",
                )
        for reason in self.envelope.unavailable_reasons:
            fields = [
                ("detail", reason.detail),
                *(("provenance", item) for item in reason.provenance),
            ]
            if reason.code != EDAReasonCode.TARGET_FREE_UNPROVEN.value:
                fields.append(("code", reason.code))
            for field_name, value in fields:
                _reject_text_tokens(
                    value,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label=f"raw unavailable reason {field_name}",
                )
        for extension in self.semantic_payload.extensions:
            if extension.corpus != self.envelope.corpus or not extension.target_free:
                raise EDAContractError(
                    "eda.raw.extension_invalid",
                    "raw extensions must match the corpus and be explicitly target-free",
                )
            if (
                extension.evidence_scope != self.envelope.evidence_scope
                or not _split_within(extension.split_scope, self.envelope.split_scope)
            ):
                raise EDAContractError(
                    "eda.raw.extension_scope_mismatch",
                    "raw extensions must explicitly bind the report scopes",
                )
            for field_name, value in (
                ("namespace", extension.namespace),
                ("schema_name", extension.schema_name),
                *(('provenance', value) for value in extension.provenance),
            ):
                _reject_text_tokens(
                    value,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label=f"raw extension {field_name}",
                )
            if extension.work_identity is not None:
                _reject_text_tokens(
                    extension.work_identity.identity,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label="raw extension work_identity",
                )
            for row in extension.rows:
                _reject_text_tokens(
                    row.row_id,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label="raw extension row_id",
                )
                if row.coverage.observation_unit not in _RAW_ALLOWED_OBSERVATION_UNITS:
                    raise EDAContractError(
                        "eda.raw.observation_unit_forbidden",
                        "raw extension metric coverage may use only source-structural units",
                    )
                for field_name, value in (
                    ("coverage.reason_code", row.coverage.reason_code),
                    *(
                        ("coverage.provenance", item)
                        for item in row.coverage.provenance
                    ),
                ):
                    if (
                        field_name == "coverage.reason_code"
                        and value == EDAReasonCode.TARGET_FREE_UNPROVEN.value
                    ):
                        continue
                    _reject_text_tokens(
                        value,
                        _RAW_TARGET_TOKENS,
                        category="eda.raw.target_field_forbidden",
                        label=f"raw extension {field_name}",
                    )
                _reject_keys(row.payload, _RAW_TARGET_DERIVED_KEYS)
                _reject_raw_target_fields(row.payload)
                for count in row.counts:
                    if (
                        count.observation_unit not in _RAW_ALLOWED_OBSERVATION_UNITS
                        or count.denominator_unit not in _RAW_ALLOWED_OBSERVATION_UNITS
                    ):
                        raise EDAContractError(
                            "eda.raw.observation_unit_forbidden",
                            "raw extension counts may use only source-structural units",
                        )
                    for field_name, value in (
                        ("count.name", count.name),
                        *(("count.provenance", value) for value in count.provenance),
                    ):
                        _reject_text_tokens(
                            value,
                            _RAW_TARGET_TOKENS,
                            category="eda.raw.target_field_forbidden",
                            label=f"raw extension {field_name}",
                        )
                    if (
                        count.reason_code
                        != EDAReasonCode.TARGET_FREE_UNPROVEN.value
                    ):
                        _reject_text_tokens(
                            count.reason_code,
                            _RAW_TARGET_TOKENS,
                            category="eda.raw.target_field_forbidden",
                            label="raw extension count reason_code",
                        )
                    if (
                        not _split_within(count.split_scope, extension.split_scope)
                        or count.evidence_scope != extension.evidence_scope
                    ):
                        raise EDAContractError(
                            "eda.raw.extension_count_scope_mismatch",
                            "raw extension counts must bind the report scopes",
                        )
        for metric in self.semantic_payload.metrics:
            metric_counts = (
                *((metric.count,) if metric.count is not None else ()),
                *(category.count for category in metric.categories),
            )
            metric_text = [
                ("coverage.reason_code", metric.coverage.reason_code),
                *(("coverage.provenance", value) for value in metric.coverage.provenance),
                *(("category", category.category) for category in metric.categories),
            ]
            for count in metric_counts:
                metric_text.extend(
                    (
                        ("count.name", count.name),
                        ("count.reason_code", count.reason_code),
                        *(("count.provenance", value) for value in count.provenance),
                    )
                )
            for field_name, value in metric_text:
                if (
                    field_name.endswith("reason_code")
                    and value == EDAReasonCode.TARGET_FREE_UNPROVEN.value
                ):
                    continue
                _reject_text_tokens(
                    value,
                    _RAW_TARGET_TOKENS,
                    category="eda.raw.target_field_forbidden",
                    label=f"raw metric {field_name}",
                )
            if not _split_within(metric.coverage.split_scope, self.envelope.split_scope):
                raise EDAContractError(
                    "eda.raw.split_mismatch", "metric split lies outside report split scope"
                )
            if metric.coverage.evidence_scope != self.envelope.evidence_scope:
                raise EDAContractError(
                    "eda.raw.evidence_scope_mismatch",
                    "metric evidence scope must match the report",
                )
        graph_reason = self.semantic_payload.graph_evidence.reason_code
        if graph_reason not in {
            EDAReasonCode.TARGET_FREE_UNPROVEN.value,
            "eda.graph.target_free_unproven",
        }:
            _reject_text_tokens(
                graph_reason,
                _RAW_TARGET_TOKENS,
                category="eda.raw.target_field_forbidden",
                label="raw graph reason_code",
            )
        non_evidence_report = self.envelope.completeness_status in {
            CompletenessStatus.NOT_COMPUTED,
            CompletenessStatus.UNAVAILABLE,
            CompletenessStatus.UNKNOWN,
        }
        if non_evidence_report and (
            any(
                metric.coverage.status == ComputationStatus.OBSERVED
                for metric in self.semantic_payload.metrics
            )
            or self.semantic_payload.graph_evidence.status == ComputationStatus.OBSERVED
            or any(
                row.coverage.status == ComputationStatus.OBSERVED
                for extension in self.semantic_payload.extensions
                for row in extension.rows
            )
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "a non-computed/unavailable/unknown raw report cannot contain observed values",
            )
        inventory = {
            metric.metric_id: metric
            for metric in self.semantic_payload.metrics
            if metric.metric_id
            in {"discovered_records", "accepted_records", "quarantined_records"}
        }
        if all(item.count is not None for item in inventory.values()):
            discovered_metric = inventory["discovered_records"]
            accepted_metric = inventory["accepted_records"]
            quarantined_metric = inventory["quarantined_records"]
            discovered = discovered_metric.count
            accepted = accepted_metric.count
            quarantined = quarantined_metric.count
            assert discovered is not None and accepted is not None and quarantined is not None
            if all(
                item.status == ComputationStatus.OBSERVED
                for item in (discovered, accepted, quarantined)
            ):
                signature = (
                    discovered.denominator,
                    discovered.denominator_unit,
                    discovered.split_scope,
                    discovered.evidence_scope,
                    discovered.provenance,
                )
                coverage = discovered_metric.coverage
                if any(
                    (
                        item.denominator,
                        item.denominator_unit,
                        item.split_scope,
                        item.evidence_scope,
                        item.provenance,
                    )
                    != signature
                    for item in (accepted, quarantined)
                ) or discovered.value != coverage.observed_count or any(
                    item.coverage != coverage
                    for item in (accepted_metric, quarantined_metric)
                ) or discovered.value != (accepted.value or 0) + (quarantined.value or 0):
                    raise EDAContractError(
                        "eda.raw.inventory_mismatch",
                        "discovered_records must equal accepted_records + quarantined_records under one coverage and denominator",
                    )
        graph_metrics = {
            metric.metric_id: metric
            for metric in self.semantic_payload.metrics
            if metric.metric_id
            in {"graph_node_counts", "graph_edge_counts", "graph_size_distribution"}
        }
        if any(
            metric.coverage.status != self.semantic_payload.graph_evidence.status
            for metric in graph_metrics.values()
        ):
            raise EDAContractError(
                "eda.graph.metric_status_mismatch",
                "all graph metrics must exactly match the target-free graph evidence status",
            )
        graph_coverages = tuple(
            graph_metrics[metric_id].coverage
            for metric_id in (
                "graph_node_counts",
                "graph_edge_counts",
                "graph_size_distribution",
            )
        )
        if any(coverage != graph_coverages[0] for coverage in graph_coverages[1:]):
            raise EDAContractError(
                "eda.graph.metric_coverage_mismatch",
                "all graph metrics must bind one population, split, scope, and provenance",
            )
        if self.semantic_payload.graph_evidence.status == ComputationStatus.OBSERVED:
            observed_count = graph_coverages[0].observed_count
            assert observed_count is not None
            if observed_count:
                node_total = sum(
                    category.count.value or 0
                    for category in graph_metrics["graph_node_counts"].categories
                )
                edge_total = sum(
                    category.count.value or 0
                    for category in graph_metrics["graph_edge_counts"].categories
                )
                size_summary = graph_metrics["graph_size_distribution"].numeric
                assert size_summary is not None
                aggregate_size = node_total + edge_total
                expected_mean = Fraction(aggregate_size, observed_count)
                actual_mean = size_summary.mean
                if isinstance(actual_mean, int):
                    mean_matches = Fraction(actual_mean, 1) == expected_mean
                else:
                    try:
                        mean_matches = actual_mean == float(expected_mean)
                    except OverflowError:
                        mean_matches = False
                if not mean_matches:
                    raise EDAContractError(
                        "eda.graph.size_mean_mismatch",
                        "graph-size mean must equal aggregate node plus edge occurrences per observed record",
                    )
                minimum_possible = (
                    int(size_summary.maximum)
                    + (observed_count - 1) * int(size_summary.minimum)
                )
                maximum_possible = (
                    int(size_summary.minimum)
                    + (observed_count - 1) * int(size_summary.maximum)
                )
                if not minimum_possible <= aggregate_size <= maximum_possible:
                    raise EDAContractError(
                        "eda.graph.size_extrema_mismatch",
                        "graph-size extrema are impossible for the aggregate node and edge total",
                    )
        used_units = _used_raw_units(self.semantic_payload)
        if set(self.envelope.observation_units) != used_units:
            raise EDAContractError(
                "eda.observation_units.mismatch",
                "envelope observation_units must equal the units used by raw metrics",
            )
        semantic = raw_report_semantic_dict(self)
        _reject_keys(semantic, _OPERATIONAL_KEYS)
        _reject_operational_semantics(semantic)
        _reject_operational_attestation_markers(semantic)
        _reject_keys(semantic["semantic_payload"], _RAW_TARGET_DERIVED_KEYS)
        if self.envelope.evidence_scope == EvidenceScope.PRODUCTION:
            _reject_nonproduction_markers(semantic)
        object.__setattr__(self, "semantic_fingerprint", canonical_json_sha256(semantic))


@dataclass(frozen=True, slots=True)
class SupervisionEDA:
    envelope: ReportEnvelope
    semantic_payload: SupervisionEDAPayload
    semantic_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not SupervisionEDA:
            raise EDAContractError(
                "eda.supervision.type_invalid",
                "SupervisionEDA is a closed wire schema and cannot be subclassed",
            )
        if not isinstance(self.envelope, ReportEnvelope) or not isinstance(
            self.semantic_payload, SupervisionEDAPayload
        ):
            raise EDAContractError(
                "eda.supervision.type_invalid",
                "supervision report requires ReportEnvelope and SupervisionEDAPayload",
            )
        if (
            self.envelope.schema_name != SUPERVISION_EDA_SCHEMA_NAME
            or self.envelope.schema_version != SUPERVISION_EDA_SCHEMA_VERSION
            or self.envelope.report_kind != ReportKind.SUPERVISION
        ):
            raise EDAContractError(
                "eda.supervision.schema_invalid",
                "supervision report must use SupervisionEDA@1.0.0",
            )
        capability = corpus_eda_capability(self.envelope.corpus)
        if not capability.supervision_eda:
            raise EDAContractError(
                "eda.capability.supervision_forbidden",
                "PDMX is raw/SSL-only and cannot emit a placeholder supervision report",
            )
        if self.envelope.split_scope not in {
            SplitScope.TRAIN,
            SplitScope.VALIDATION,
            SplitScope.TRAIN_VALIDATION,
        }:
            raise EDAContractError(
                "eda.test_lock.supervision_report_split_forbidden",
                "SupervisionEDA may contain TRAIN/VALIDATION only",
            )
        for field_name, value in (
            ("source_identity", self.envelope.source_identity.identity),
            ("producer_identity", self.envelope.producer_identity.identity),
        ):
            _reject_text_tokens(
                value,
                _TEST_SCOPE_TOKENS,
                category="eda.test_lock.task_test_field_forbidden",
                label=f"supervision envelope {field_name}",
            )
        for invariant in self.envelope.invariants:
            for field_name, value in (
                ("code", invariant.code),
                ("reason_code", invariant.reason_code),
                *(("provenance", item) for item in invariant.provenance),
            ):
                if (
                    field_name == "reason_code"
                    and value == EDAReasonCode.TEST_TARGETS_LOCKED.value
                ):
                    continue
                _reject_text_tokens(
                    value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.envelope_test_field_forbidden",
                    label=f"supervision invariant {field_name}",
                )
        for warning in self.envelope.warnings:
            for field_name, value in (
                ("code", warning.code),
                ("message", warning.message),
                *(("provenance", item) for item in warning.provenance),
            ):
                _reject_text_tokens(
                    value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.envelope_test_field_forbidden",
                    label=f"supervision warning {field_name}",
                )
        for reason in self.envelope.unavailable_reasons:
            fields = [
                ("detail", reason.detail),
                *(("provenance", item) for item in reason.provenance),
            ]
            if reason.code != EDAReasonCode.TEST_TARGETS_LOCKED.value:
                fields.append(("code", reason.code))
            for field_name, value in fields:
                _reject_text_tokens(
                    value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.envelope_test_field_forbidden",
                    label=f"supervision unavailable reason {field_name}",
                )
        if (
            self.semantic_payload.test_lock.test_assignment_count.evidence_scope
            != self.envelope.evidence_scope
        ):
            raise EDAContractError(
                "eda.test_lock.evidence_scope_mismatch",
                "TEST-lock counters must bind the report evidence scope",
            )
        lock_count = self.semantic_payload.test_lock.test_assignment_count
        for provenance in lock_count.provenance:
            if _test_lock_provenance_claims_access(provenance):
                raise EDAContractError(
                    "eda.test_lock.provenance_forbidden",
                    "TEST-lock provenance cannot attest access to TEST targets",
                    path="$.semantic_payload.test_lock.test_assignment_count.provenance",
                )
        if (
            lock_count.reason_code is not None
            and lock_count.reason_code != EDAReasonCode.TEST_TARGETS_LOCKED.value
        ):
            if _test_lock_provenance_claims_access(lock_count.reason_code):
                raise EDAContractError(
                    "eda.test_lock.reason_code_forbidden",
                    "TEST-lock reason_code cannot claim TEST-target access",
                    path=(
                        "$.semantic_payload.test_lock.test_assignment_count.reason_code"
                    ),
                )
        assignment_manifests = tuple(
            manifest
            for manifest in self.envelope.input_manifests
            if manifest.role == EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE
        )
        if self.envelope.input_manifests:
            if (
                len(assignment_manifests) != 1
                or not assignment_manifests[0].target_free
                or assignment_manifests[0].identity.fingerprint
                != self.semantic_payload.test_lock.assignment_manifest_fingerprint
            ):
                raise EDAContractError(
                    "eda.test_lock.assignment_manifest_unbound",
                    "SupervisionEDA requires one target-free split-assignment manifest matching lock evidence",
                )
            if not any(
                not manifest.target_free
                for manifest in self.envelope.input_manifests
                if manifest.role != EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE
            ):
                raise EDAContractError(
                    "eda.supervision.target_manifest_unbound",
                    "SupervisionEDA must bind at least one target-bearing input manifest",
                )
        elif (
            self.semantic_payload.test_lock.assignment_manifest_fingerprint is not None
            or self.semantic_payload.test_lock.test_assignment_count.status
            != ComputationStatus.LOCKED
        ):
            raise EDAContractError(
                "eda.test_lock.assignment_manifest_unbound",
                "a manifest-free unavailable report requires a null, locked TEST attestation",
            )
        for manifest_index, manifest in enumerate(self.envelope.input_manifests):
            fields = [("role", manifest.role)]
            if manifest.role != EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE:
                fields.extend(
                    (
                        ("identity", manifest.identity.identity),
                        ("repository_relative_path", manifest.repository_relative_path),
                    )
                )
            for field_name, value in fields:
                if (
                    field_name == "role"
                    and isinstance(value, str)
                    and _normalized_field_name(value) in {"tests", "testsets"}
                ):
                    raise EDAContractError(
                        "eda.test_lock.manifest_test_field_forbidden",
                        "supervision manifest role cannot select plural TEST scope",
                        path=(
                            f"$.envelope.input_manifests[{manifest_index}].{field_name}"
                        ),
                    )
                checked_value = (
                    value.removeprefix("tests/")
                    if field_name == "repository_relative_path"
                    and isinstance(value, str)
                    else value
                )
                _reject_text_tokens(
                    checked_value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.manifest_test_field_forbidden",
                    label=f"supervision manifest {field_name}",
                    path=f"$.envelope.input_manifests[{manifest_index}].{field_name}",
                )
            if manifest.role == EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE:
                for field_name, value in (
                    ("identity", manifest.identity.identity),
                    ("repository_relative_path", manifest.repository_relative_path),
                ):
                    _reject_text_tokens(
                        value,
                        _RAW_MANIFEST_TARGET_TOKENS,
                        category="eda.test_lock.assignment_manifest_target_forbidden",
                        label=f"target-free split-assignment manifest {field_name}",
                        path=(
                            f"$.envelope.input_manifests[{manifest_index}].{field_name}"
                        ),
                    )
        for task in self.semantic_payload.tasks:
            if task.corpus != self.envelope.corpus:
                raise EDAContractError(
                    "eda.supervision.corpus_mismatch", "task corpus differs from report corpus"
                )
            if not _split_within(task.split_scope, self.envelope.split_scope):
                raise EDAContractError(
                    "eda.supervision.split_mismatch", "task split lies outside report split"
                )
            if task.evidence_scope != self.envelope.evidence_scope:
                raise EDAContractError(
                    "eda.supervision.evidence_scope_mismatch",
                    "task evidence scope must match report evidence scope",
                )
            for field_name, value in (
                ("source_task_id", task.source_task_id),
                ("dialect", task.dialect),
                ("annotation_namespace", task.annotation_namespace),
                ("label_granularity", task.label_granularity),
                ("vocabulary", task.vocabulary.identity),
                ("reason_code", task.reason_code),
                *(("provenance", value) for value in task.provenance),
            ):
                if (
                    field_name == "reason_code"
                    and value == EDAReasonCode.TEST_TARGETS_LOCKED.value
                ):
                    continue
                _reject_text_tokens(
                    value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.task_test_field_forbidden",
                    label=f"supervision task {field_name}",
                )
            if task.work_identity is not None:
                _reject_text_tokens(
                    task.work_identity.identity,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.task_test_field_forbidden",
                    label="supervision task work_identity",
                )
            nested_provenance: list[tuple[str, ...]] = []
            if task.availability is not None:
                nested_provenance.append(task.availability.provenance)
            if task.empty_multilabel_available_count is not None:
                nested_provenance.append(
                    task.empty_multilabel_available_count.provenance
                )
            nested_provenance.extend(
                support.occurrence_count.provenance
                for support in task.class_support
            )
            nested_provenance.extend(
                projection.provenance for projection in task.projections
            )
            nested_provenance.extend(
                projection.provenance
                for projection in task.projection_availability
            )
            for provenance in nested_provenance:
                for value in provenance:
                    _reject_text_tokens(
                        value,
                        _TEST_SCOPE_TOKENS,
                        category="eda.test_lock.task_test_field_forbidden",
                        label="supervision task nested provenance",
                    )
            for support in task.class_support:
                if (
                    support.unique_work_count.reason_code
                    != EDAReasonCode.TEST_TARGETS_LOCKED.value
                ):
                    _reject_text_tokens(
                        support.unique_work_count.reason_code,
                        _TEST_SCOPE_TOKENS,
                        category="eda.test_lock.task_test_field_forbidden",
                        label="supervision class-support work reason_code",
                    )
        observed_supervision_evidence = any(
            task.status == ComputationStatus.OBSERVED
            for task in self.semantic_payload.tasks
        ) or any(
            row.coverage.status == ComputationStatus.OBSERVED
            for extension in self.semantic_payload.extensions
            for row in extension.rows
        )
        if observed_supervision_evidence and (
            self.semantic_payload.test_lock.test_assignment_count.status
            != ComputationStatus.OBSERVED
        ):
            raise EDAContractError(
                "eda.test_lock.observed_evidence_without_guard",
                "observed supervision tasks or extension rows require an observed TEST-gate attestation",
            )
        for extension in self.semantic_payload.extensions:
            if extension.corpus != self.envelope.corpus:
                raise EDAContractError(
                    "eda.supervision.extension_corpus_mismatch",
                    "supervision extension corpus differs from report corpus",
                )
            if (
                extension.split_scope not in {SplitScope.TRAIN, SplitScope.VALIDATION}
                or not _split_within(extension.split_scope, self.envelope.split_scope)
                or extension.evidence_scope != self.envelope.evidence_scope
            ):
                raise EDAContractError(
                    "eda.test_lock.extension_scope_forbidden",
                    "supervision extensions must explicitly bind TRAIN or VALIDATION in the report evidence scope",
                )
            for field_name, value in (
                ("namespace", extension.namespace),
                ("schema_name", extension.schema_name),
                *(("provenance", value) for value in extension.provenance),
            ):
                _reject_text_tokens(
                    value,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.extension_test_field_forbidden",
                    label=f"supervision extension {field_name}",
                )
            if extension.work_identity is not None:
                _reject_text_tokens(
                    extension.work_identity.identity,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.extension_test_field_forbidden",
                    label="supervision extension work_identity",
                )
            for row in extension.rows:
                _reject_text_tokens(
                    row.row_id,
                    _TEST_SCOPE_TOKENS,
                    category="eda.test_lock.extension_test_field_forbidden",
                    label="supervision extension row_id",
                )
                for field_name, value in (
                    ("coverage.reason_code", row.coverage.reason_code),
                    *(
                        ("coverage.provenance", item)
                        for item in row.coverage.provenance
                    ),
                ):
                    if (
                        field_name == "coverage.reason_code"
                        and value == EDAReasonCode.TEST_TARGETS_LOCKED.value
                    ):
                        continue
                    _reject_text_tokens(
                        value,
                        _TEST_SCOPE_TOKENS,
                        category="eda.test_lock.extension_test_field_forbidden",
                        label=f"supervision extension {field_name}",
                    )
                _reject_test_supervision_fields(row.payload)
                for count in row.counts:
                    for field_name, value in (
                        ("count.name", count.name),
                        *(("count.provenance", value) for value in count.provenance),
                        ("count.reason_code", count.reason_code),
                    ):
                        if (
                            field_name == "count.reason_code"
                            and value == EDAReasonCode.TEST_TARGETS_LOCKED.value
                        ):
                            continue
                        _reject_text_tokens(
                            value,
                            _TEST_SCOPE_TOKENS,
                            category="eda.test_lock.extension_test_field_forbidden",
                            label=f"supervision extension {field_name}",
                        )
                    if (
                        not _split_within(count.split_scope, extension.split_scope)
                        or count.evidence_scope != extension.evidence_scope
                    ):
                        raise EDAContractError(
                            "eda.supervision.extension_count_scope_mismatch",
                            "supervision extension counts must bind the report scopes",
                        )
        non_evidence_report = self.envelope.completeness_status in {
            CompletenessStatus.NOT_COMPUTED,
            CompletenessStatus.UNAVAILABLE,
            CompletenessStatus.UNKNOWN,
        }
        if non_evidence_report and (
            any(
                task.status == ComputationStatus.OBSERVED
                for task in self.semantic_payload.tasks
            )
            or any(
                row.coverage.status == ComputationStatus.OBSERVED
                for extension in self.semantic_payload.extensions
                for row in extension.rows
            )
        ):
            raise EDAContractError(
                "eda.evidence.observed_without_evidence",
                "a non-computed/unavailable/unknown supervision report cannot contain observed values",
            )
        used_units = _used_supervision_units(self.semantic_payload)
        if set(self.envelope.observation_units) != used_units:
            raise EDAContractError(
                "eda.observation_units.mismatch",
                "envelope observation_units must equal the units used by supervision evidence",
            )
        semantic = supervision_report_semantic_dict(self)
        _reject_keys(semantic, _OPERATIONAL_KEYS)
        _reject_operational_semantics(semantic)
        _reject_operational_attestation_markers(semantic)
        if self.envelope.evidence_scope == EvidenceScope.PRODUCTION:
            _reject_nonproduction_markers(semantic)
        object.__setattr__(self, "semantic_fingerprint", canonical_json_sha256(semantic))


_SCHEMA_DATACLASS_TYPES = frozenset(
    {
        AvailabilityCounts,
        CategoryCount,
        ClassSupport,
        CorpusEDACapability,
        ExtensionRow,
        GraphEvidence,
        InputManifestRef,
        InvariantEvidence,
        MetricCoverage,
        NumericDistribution,
        ProjectionAvailabilityCounts,
        ProjectionEvidence,
        QuantilePoint,
        RawCorpusEDA,
        RawCorpusEDAPayload,
        RawMetricEvidence,
        RawMetricSpec,
        ReportEnvelope,
        SourceExtension,
        SourceValueIdentity,
        StructuredWarning,
        SupervisionEDA,
        SupervisionEDAPayload,
        TaskFamilyEvidence,
        TestTargetLockEvidence,
        UnavailableReason,
        UnitCount,
        VersionedIdentity,
    }
)


def _validated_report(
    report: object,
) -> RawCorpusEDA | SupervisionEDA:
    if type(report) not in {RawCorpusEDA, SupervisionEDA}:
        raise EDAContractError(
            "eda.report.type_invalid", "expected RawCorpusEDA or SupervisionEDA"
        )
    return report


def report_dict(report: RawCorpusEDA | SupervisionEDA) -> dict[str, object]:
    """Return a canonical JSON-safe report including operational metadata."""

    value = _jsonable(_validated_report(report))
    assert isinstance(value, dict)
    return value


def report_fingerprint(report: RawCorpusEDA | SupervisionEDA) -> str:
    """Recompute and return the semantic SHA-256 (never hash the hash field)."""

    validated = _validated_report(report)
    if isinstance(validated, RawCorpusEDA):
        value = raw_report_semantic_dict(validated)
    else:
        value = supervision_report_semantic_dict(validated)
    return canonical_json_sha256(value)


def dumps_report(
    report: RawCorpusEDA | SupervisionEDA, *, indent: int | None = None
) -> str:
    """Serialize a validated report with stable UTF-8 JSON and no newline."""

    validated = _validated_report(report)
    if validated.semantic_fingerprint != report_fingerprint(validated):
        raise EDAContractError(
            "eda.fingerprint.mismatch", "report semantic fingerprint is stale"
        )
    return dumps_canonical_json(report_dict(validated), indent=indent)


__all__ = [
    "APPROVED_RAW_GRAPH_CONTRACT",
    "APPROVED_PROJECTION_REGISTRIES",
    "AvailabilityCounts",
    "AvailabilityState",
    "CategoryCount",
    "ClassSupport",
    "CompletenessStatus",
    "ComputationStatus",
    "CorpusEDACapability",
    "CorpusId",
    "DILEMMADATA_COMMON_REGISTRY_FINGERPRINT",
    "DILEMMADATA_COMMON_REGISTRY_ID",
    "DILEMMADATA_COMMON_REGISTRY_VERSION",
    "DILEMMADATA_COMMON_TASK_IDS",
    "EDA_ADAPTER_REGISTRY_VERSION",
    "EDA_CAPABILITIES",
    "EDA_CAPABILITY_REGISTRY_SCHEMA",
    "EDA_CAPABILITY_REGISTRY_SCHEMA_NAME",
    "EDA_CAPABILITY_REGISTRY_SCHEMA_VERSION",
    "EDA_ENVELOPE_SCHEMA_NAME",
    "EDA_ENVELOPE_SCHEMA_VERSION",
    "EDA_FLOAT_POLICY",
    "EDA_QUANTILE_POLICY",
    "EDAReasonCode",
    "EDA_SCHEMA_VERSION_POLICY",
    "EDA_SOURCE_EXTENSION_VERSION",
    "EDA_SOURCE_VALUE_IDENTITY_VERSION",
    "EDA_SPLIT_ASSIGNMENT_MANIFEST_ROLE",
    "EDA_TEST_TARGET_LOCK_VERSION",
    "EDAContractError",
    "EvidenceScope",
    "ExecutionMode",
    "ExtensionRow",
    "GraphEvidence",
    "InputManifestRef",
    "InvariantEvidence",
    "InvariantStatus",
    "LabelValueType",
    "MetricCoverage",
    "MetricSummaryKind",
    "NumericDistribution",
    "ObservationUnit",
    "ProjectionEvidence",
    "ProjectionAvailabilityCounts",
    "ProjectionMappingState",
    "QuantilePoint",
    "RAW_CORPUS_EDA_SCHEMA",
    "RAW_CORPUS_EDA_SCHEMA_NAME",
    "RAW_CORPUS_EDA_SCHEMA_VERSION",
    "RAW_METRIC_CATALOG",
    "RawCorpusEDA",
    "RawCorpusEDAPayload",
    "RawMetricEvidence",
    "RawMetricSpec",
    "ReportEnvelope",
    "ReportKind",
    "SUPERVISION_EDA_SCHEMA",
    "SUPERVISION_EDA_SCHEMA_NAME",
    "SUPERVISION_EDA_SCHEMA_VERSION",
    "SourceExtension",
    "SourceValueIdentity",
    "SourceValueKind",
    "SplitScope",
    "StructuredWarning",
    "SupervisionEDA",
    "SupervisionEDAPayload",
    "TaskFamilyEvidence",
    "TestTargetLockEvidence",
    "UnavailableReason",
    "UnitCount",
    "VersionedIdentity",
    "capability_registry_dict",
    "capability_registry_fingerprint",
    "corpus_eda_capability",
    "dumps_report",
    "raw_report_semantic_dict",
    "report_dict",
    "report_fingerprint",
    "sum_unit_counts",
    "supervision_report_semantic_dict",
]
