"""POP909-CL EDA adapter over tracked Phase 4A/4B evidence.

The raw path replays a target-free projection of already accepted manifests.
The source-specific Phase 4 replay validates the exact historical audit and
production-manifest bytes before exposing their pre-split aggregate evidence.
Formal TRAIN/VALIDATION supervision remains either fixture-scoped or explicitly
not computed: the tracked evidence contains no split-by-class rows, so this
module never promotes all-corpus or synthetic values to a split claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from music_critic.adapters.pop909_cl import (
    POP909_CL_ADAPTER_VERSION,
    POP909_CL_CONTENT_FINGERPRINT,
    POP909_CL_DATASET_NAME,
    POP909_CL_TASKS,
)
from music_critic.data.serialization import canonical_json_sha256, dumps_canonical_json
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
from music_critic.tasks.ontology import (
    TARGET_FAMILY_BY_ID,
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_fingerprint,
)


EDA_CONTRACT_SHA = "65eb32fb948efde0fa117d7d27d19d8f16fa25b4"
POP909_CL_EDA_ADAPTER_VERSION = "1.0.0"
POP909_CL_RAW_EDA_MANIFEST_SCHEMA = "Pop909ClRawEDAManifest@1.0.0"
POP909_CL_EDA_SPLIT_SCHEMA = "Pop909ClEDASplitAssignments@1.0.0"
POP909_CL_EDA_SUPERVISION_FIXTURE_SCHEMA = (
    "Pop909ClEDASupervisionFixture@1.0.0"
)
POP909_CL_EDA_SUPERVISION_FIXTURE_SHA256 = (
    "6babf2150d4f3799dd5201af3e649e7e3eae33c7f08eb70f350ee27cb4f2318e"
)
POP909_CL_PHASE4_EVIDENCE_SCHEMA = "Pop909ClPhase4EvidenceReplay@1.0.0"
POP909_CL_RAW_EXTENSION_NAMESPACE = "pop909_cl.raw_manifest"
POP909_CL_RAW_EXTENSION_SCHEMA = "Pop909ClRawManifestExtension"

POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256 = (
    "46e7254f8a451f64a009d54cceec5a16703eb3ca80b88984127a643c73f9105a"
)
POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256 = (
    "bc9c4118c72cb39bc1393fd2d250db577835a837fc53f8fe8c1238c7b13a8031"
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RAW_PROVENANCE = ("pop909_cl.eda.raw_manifest_replay",)
_SUPERVISION_PROVENANCE = ("pop909_cl.eda.supervision_fixture",)
_GUARD_PROVENANCE = ("pop909_cl.eda.supervision_guard",)
_UNAVAILABLE_SUPERVISION_PROVENANCE = (
    "pop909_cl.eda.split_supervision_unavailable",
)
_SOURCE_IDENTITY = VersionedIdentity(
    identity="pop909_cl.release",
    version=POP909_CL_ADAPTER_VERSION,
    fingerprint=POP909_CL_CONTENT_FINGERPRINT,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _adapter_identity() -> VersionedIdentity:
    return VersionedIdentity(
        identity="music_critic.adapters.pop909_cl_eda",
        version=POP909_CL_EDA_ADAPTER_VERSION,
        fingerprint=_file_sha256(Path(__file__)),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EDAContractError(
                "pop909_cl.eda.manifest_duplicate_key",
                f"duplicate JSON key {key!r}",
            )
        result[key] = value
    return result


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
            "pop909_cl.eda.manifest_invalid",
            f"cannot load EDA manifest {path.name!r}: {exc}",
        ) from exc
    if not isinstance(decoded, dict):
        raise EDAContractError(
            "pop909_cl.eda.manifest_invalid", "EDA manifest root must be an object"
        )
    return decoded, sha256(raw).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EDAContractError(
            "pop909_cl.eda.manifest_invalid", f"{name} must be an object"
        )
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise EDAContractError(
            "pop909_cl.eda.manifest_invalid", f"{name} must be an array"
        )
    return value


def _expect(value: object, expected: object, name: str) -> None:
    if value != expected or isinstance(value, bool) != isinstance(expected, bool):
        raise EDAContractError(
            "pop909_cl.eda.manifest_mismatch",
            f"{name} must equal the accepted value {expected!r}",
        )


def _require_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise EDAContractError(
            "pop909_cl.eda.manifest_fields_invalid",
            f"{name} fields must be exactly {sorted(expected)!r}",
        )


def _repository_relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Pop909ClRawEDARequest:
    """Request a target-free replay of the tracked POP909-CL EDA manifest."""

    manifest_path: str | Path
    repository_commit: str


@dataclass(frozen=True, slots=True)
class Pop909ClSupervisionEDARequest:
    """Request fixture-only native supervision evidence.

    The observer hooks exist for TEST-gate regression spies.  They receive only
    retained TRAIN/VALIDATION rows because the shared guard owns dispatch.
    """

    split_manifest_path: str | Path
    supervision_fixture_path: str | Path
    repository_commit: str
    descriptor_observer: Callable[[str, SplitScope], None] | None = field(
        default=None, repr=False, compare=False
    )
    loader_observer: Callable[[str, SplitScope], None] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class Pop909ClUnavailableSupervisionEDARequest:
    """Request an honest formal report for unavailable split supervision.

    This report does not open a split manifest or any target-bearing artifact.
    It exists so production-readiness consumers do not mistake the bounded
    fixture report for corpus-wide TRAIN/VALIDATION evidence.
    """

    repository_commit: str


@dataclass(frozen=True, slots=True)
class Pop909ClPhase4EvidenceRequest:
    """Request exact replay of the target-free projection and Phase 4 manifests."""

    raw_manifest_path: str | Path
    audit_manifest_path: str | Path
    production_manifest_path: str | Path
    repository_commit: str


@dataclass(frozen=True, slots=True)
class Pop909ClPhase4TaskAvailability:
    """Pre-split source-native target-row inventory from Phase 4 evidence."""

    task_id: str
    denominator: int
    available: int
    masked: int
    missing: int
    unsupported: int
    accepted_available_record_support: int | None = None
    accepted_missing_record_count: int = 2
    record_support_status: ComputationStatus = ComputationStatus.NOT_COMPUTED

    def __post_init__(self) -> None:
        if self.task_id not in POP909_CL_TASKS:
            raise EDAContractError(
                "pop909_cl.eda.phase4_task_invalid",
                f"unknown Phase 4 source-native task {self.task_id!r}",
            )
        values = (
            self.denominator,
            self.available,
            self.masked,
            self.missing,
            self.unsupported,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise EDAContractError(
                "pop909_cl.eda.phase4_count_invalid",
                "Phase 4 availability counts must be non-negative integers",
            )
        if sum(values[1:]) != self.denominator:
            raise EDAContractError(
                "pop909_cl.eda.phase4_availability_invalid",
                "Phase 4 availability states must partition the denominator",
            )
        if self.accepted_missing_record_count != 2:
            raise EDAContractError(
                "pop909_cl.eda.phase4_missing_record_support_invalid",
                "all six target families require the two accepted missing records",
            )
        exact_record_support = self.task_id in {
            "pop909_cl.chord.bass",
            "pop909_cl.chord.boundary",
        }
        if exact_record_support:
            if (
                self.accepted_available_record_support != 906
                or self.record_support_status != ComputationStatus.OBSERVED
            ):
                raise EDAContractError(
                    "pop909_cl.eda.phase4_record_support_invalid",
                    "boundary and bass require exact 906-record accepted support",
                )
        elif (
            self.accepted_available_record_support is not None
            or self.record_support_status != ComputationStatus.NOT_COMPUTED
        ):
            raise EDAContractError(
                "pop909_cl.eda.phase4_record_support_unproven",
                "other per-task accepted record support is not tracked",
            )


@dataclass(frozen=True, slots=True)
class Pop909ClPhase4EvidenceReplay:
    """Validated, fingerprinted pre-split evidence replay.

    This is intentionally not ``SupervisionEDA``: Phase 4 predates the frozen
    split, and its corpus-wide aggregates include the later quarantined record.
    Recasting them as TRAIN/VALIDATION rows would violate the common TEST lock.
    """

    source_identity: VersionedIdentity
    producer_identity: VersionedIdentity
    repository_commit: str
    raw_manifest_identity: VersionedIdentity
    audit_manifest_identity: VersionedIdentity
    production_manifest_identity: VersionedIdentity
    logical_record_count: int
    accepted_record_count: int
    quarantined_record_ids: tuple[str, ...]
    accepted_missing_target_record_ids: tuple[str, ...]
    source_records_with_chord_instrument: int
    accepted_records_with_chord_evidence: int
    target_rows: tuple[Pop909ClPhase4TaskAvailability, ...]
    chord_block_count: int
    ambiguous_block_count: int
    unsupported_block_count: int
    trailing_uncovered_span_count: int
    leading_internal_no_chord_span_count: int
    overlap_count: int
    repeated_pitch_block_count: int
    mixed_end_block_count: int
    pairing_anomaly_count: int
    duplicate_block_onset_count: int
    raw_pitch_class_set_count: int
    selected_root_quality_bass_label_count: int
    chord_block_observed_record_count: int
    chord_block_minimum_per_record: int
    chord_block_median_per_record: int
    chord_block_p95_per_record: int
    chord_block_maximum_per_record: int
    midi_type_1_record_count: int
    ppqn_480_record_count: int
    empty_conductor_track_count: int
    tempo_event_count: int
    meter_event_count: int
    key_signature_event_count: int
    score_note_observed_record_count: int
    score_note_unknown_record_count: int
    score_note_minimum: int
    score_note_median: int
    score_note_p95: int
    score_note_maximum: int
    score_warning_observed_record_count: int
    score_warning_unknown_record_count: int
    score_warning_occurrence_count: int
    score_warning_minimum_per_record: int
    score_warning_median_per_record: int
    score_warning_p95_per_record: int
    score_warning_maximum_per_record: int
    trailing_duration_minimum_ticks: int
    trailing_duration_median_ticks: int
    trailing_duration_p95_ticks: int
    trailing_duration_maximum_ticks: int
    class_concentration_status: ComputationStatus
    cooccurrence_status: ComputationStatus
    train_validation_shift_status: ComputationStatus
    canonical_work_identity_status: ComputationStatus
    semantic_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_identity != _SOURCE_IDENTITY:
            raise EDAContractError(
                "pop909_cl.eda.phase4_source_invalid",
                "Phase 4 replay source identity differs from the accepted release",
            )
        if self.producer_identity != _adapter_identity():
            raise EDAContractError(
                "pop909_cl.eda.phase4_producer_invalid",
                "Phase 4 replay producer identity is invalid",
            )
        if not self.repository_commit:
            raise EDAContractError(
                "pop909_cl.eda.phase4_repository_commit_invalid",
                "Phase 4 replay repository commit cannot be empty",
            )
        expected_manifest_identities = (
            (
                self.raw_manifest_identity.identity,
                self.raw_manifest_identity.version,
                "pop909_cl.eda.raw_manifest",
                "1.0.0",
            ),
            (
                self.audit_manifest_identity.identity,
                self.audit_manifest_identity.version,
                "pop909_cl.phase_4a.audit_manifest",
                "3.0.0",
            ),
            (
                self.production_manifest_identity.identity,
                self.production_manifest_identity.version,
                "pop909_cl.phase_4b.production_manifest",
                "2.0.0",
            ),
        )
        if any(
            (identity, version) != (expected_identity, expected_version)
            for identity, version, expected_identity, expected_version in (
                expected_manifest_identities
            )
        ):
            raise EDAContractError(
                "pop909_cl.eda.phase4_manifest_identity_invalid",
                "Phase 4 replay manifest identities or versions are invalid",
            )
        if (
            self.audit_manifest_identity.fingerprint
            != POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256
            or self.production_manifest_identity.fingerprint
            != POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256
        ):
            raise EDAContractError(
                "pop909_cl.eda.phase4_manifest_drift",
                "Phase 4 replay manifest fingerprints differ from accepted bytes",
            )
        if self.logical_record_count != 909 or self.accepted_record_count != 908:
            raise EDAContractError(
                "pop909_cl.eda.phase4_inventory_invalid",
                "Phase 4 replay requires the accepted 909/908 inventory",
            )
        if self.quarantined_record_ids != ("172",):
            raise EDAContractError(
                "pop909_cl.eda.phase4_quarantine_invalid",
                "Phase 4 replay requires only record 172 to remain quarantined",
            )
        if self.accepted_missing_target_record_ids != ("367", "658"):
            raise EDAContractError(
                "pop909_cl.eda.phase4_missing_target_invalid",
                "Phase 4 replay requires the two accepted missing-target records",
            )
        if self.source_records_with_chord_instrument != 907:
            raise EDAContractError(
                "pop909_cl.eda.phase4_instrument_invalid",
                "Phase 4 replay requires 907 source chord instruments",
            )
        if self.accepted_records_with_chord_evidence != 906:
            raise EDAContractError(
                "pop909_cl.eda.phase4_record_support_invalid",
                "Phase 4 replay requires 906 accepted records with chord evidence",
            )
        expected_counts = {
            "chord_block_count": 116055,
            "ambiguous_block_count": 5801,
            "unsupported_block_count": 586,
            "trailing_uncovered_span_count": 151,
            "leading_internal_no_chord_span_count": 947,
            "overlap_count": 691,
            "repeated_pitch_block_count": 87,
            "mixed_end_block_count": 313,
            "pairing_anomaly_count": 8,
            "duplicate_block_onset_count": 0,
            "raw_pitch_class_set_count": 261,
            "selected_root_quality_bass_label_count": 340,
            "chord_block_observed_record_count": 909,
            "chord_block_minimum_per_record": 0,
            "chord_block_median_per_record": 124,
            "chord_block_p95_per_record": 185,
            "chord_block_maximum_per_record": 278,
            "midi_type_1_record_count": 909,
            "ppqn_480_record_count": 909,
            "empty_conductor_track_count": 909,
            "tempo_event_count": 909,
            "meter_event_count": 911,
            "key_signature_event_count": 1065,
            "score_note_observed_record_count": 908,
            "score_note_unknown_record_count": 1,
            "score_note_minimum": 175,
            "score_note_median": 1655,
            "score_note_p95": 2403,
            "score_note_maximum": 4233,
            "score_warning_observed_record_count": 908,
            "score_warning_unknown_record_count": 1,
            "score_warning_occurrence_count": 126163,
            "score_warning_minimum_per_record": 3,
            "score_warning_median_per_record": 123,
            "score_warning_p95_per_record": 282,
            "score_warning_maximum_per_record": 966,
            "trailing_duration_minimum_ticks": 1,
            "trailing_duration_median_ticks": 401,
            "trailing_duration_p95_ticks": 3361,
            "trailing_duration_maximum_ticks": 12861,
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise EDAContractError(
                    "pop909_cl.eda.phase4_count_mismatch",
                    f"{field_name} differs from accepted Phase 4 evidence",
                )
        if tuple(row.task_id for row in self.target_rows) != tuple(
            sorted(POP909_CL_TASKS)
        ):
            raise EDAContractError(
                "pop909_cl.eda.phase4_tasks_incomplete",
                "Phase 4 replay requires exactly the six sorted task families",
            )
        for status in (
            self.class_concentration_status,
            self.cooccurrence_status,
            self.train_validation_shift_status,
            self.canonical_work_identity_status,
        ):
            if status != ComputationStatus.NOT_COMPUTED:
                raise EDAContractError(
                    "pop909_cl.eda.phase4_analysis_unproven",
                    "untracked split/class/work analyses must remain not_computed",
                )
        object.__setattr__(
            self,
            "semantic_fingerprint",
            canonical_json_sha256(pop909_cl_phase4_evidence_dict(self)),
        )


def pop909_cl_phase4_evidence_dict(
    evidence: Pop909ClPhase4EvidenceReplay,
) -> dict[str, object]:
    """Return the canonical semantic projection for a Phase 4 replay."""

    def identity_dict(identity: VersionedIdentity) -> dict[str, str]:
        return {
            "identity": identity.identity,
            "version": identity.version,
            "fingerprint": identity.fingerprint,
        }

    return {
        "schema": POP909_CL_PHASE4_EVIDENCE_SCHEMA,
        "source_identity": identity_dict(evidence.source_identity),
        "producer_identity": identity_dict(evidence.producer_identity),
        "repository_commit": evidence.repository_commit,
        "input_manifests": {
            "raw": identity_dict(evidence.raw_manifest_identity),
            "phase_4a_audit": identity_dict(evidence.audit_manifest_identity),
            "phase_4b_production": identity_dict(
                evidence.production_manifest_identity
            ),
        },
        "record_inventory": {
            "logical_records": evidence.logical_record_count,
            "accepted_records": evidence.accepted_record_count,
            "quarantined_record_ids": list(evidence.quarantined_record_ids),
            "accepted_missing_target_record_ids": list(
                evidence.accepted_missing_target_record_ids
            ),
            "source_records_with_chord_instrument": (
                evidence.source_records_with_chord_instrument
            ),
            "accepted_records_with_chord_evidence": (
                evidence.accepted_records_with_chord_evidence
            ),
        },
        "raw_structure": {
            "midi_type_1_records": evidence.midi_type_1_record_count,
            "ppqn_480_records": evidence.ppqn_480_record_count,
            "empty_conductor_tracks": evidence.empty_conductor_track_count,
            "tempo_events": evidence.tempo_event_count,
            "meter_events": evidence.meter_event_count,
            "key_signature_events": evidence.key_signature_event_count,
            "score_note_distribution": {
                "observed_records": evidence.score_note_observed_record_count,
                "unknown_records": evidence.score_note_unknown_record_count,
                "minimum": evidence.score_note_minimum,
                "median": evidence.score_note_median,
                "p95": evidence.score_note_p95,
                "maximum": evidence.score_note_maximum,
                "mean_status": ComputationStatus.NOT_COMPUTED.value,
            },
            "score_warning_distribution": {
                "observed_records": evidence.score_warning_observed_record_count,
                "unknown_records": evidence.score_warning_unknown_record_count,
                "occurrences": evidence.score_warning_occurrence_count,
                "minimum_per_record": evidence.score_warning_minimum_per_record,
                "median_per_record": evidence.score_warning_median_per_record,
                "p95_per_record": evidence.score_warning_p95_per_record,
                "maximum_per_record": evidence.score_warning_maximum_per_record,
                "mean_status": ComputationStatus.NOT_COMPUTED.value,
            },
        },
        "source_supervision": {
            "chord_blocks": evidence.chord_block_count,
            "ambiguous_blocks": evidence.ambiguous_block_count,
            "unsupported_blocks": evidence.unsupported_block_count,
            "leading_internal_no_chord_spans": (
                evidence.leading_internal_no_chord_span_count
            ),
            "trailing_uncovered_spans": evidence.trailing_uncovered_span_count,
            "trailing_uncovered_duration_ticks": {
                "minimum": evidence.trailing_duration_minimum_ticks,
                "median": evidence.trailing_duration_median_ticks,
                "p95": evidence.trailing_duration_p95_ticks,
                "maximum": evidence.trailing_duration_maximum_ticks,
                "mean_status": ComputationStatus.NOT_COMPUTED.value,
            },
            "overlaps": evidence.overlap_count,
            "repeated_pitch_blocks": evidence.repeated_pitch_block_count,
            "mixed_end_blocks": evidence.mixed_end_block_count,
            "pairing_anomalies": evidence.pairing_anomaly_count,
            "duplicate_block_onsets": evidence.duplicate_block_onset_count,
            "raw_pitch_class_sets": evidence.raw_pitch_class_set_count,
            "selected_root_quality_bass_labels": (
                evidence.selected_root_quality_bass_label_count
            ),
            "block_count_distribution": {
                "observation_unit": ObservationUnit.RECORD.value,
                "observed_records": evidence.chord_block_observed_record_count,
                "minimum": evidence.chord_block_minimum_per_record,
                "median": evidence.chord_block_median_per_record,
                "p95": evidence.chord_block_p95_per_record,
                "maximum": evidence.chord_block_maximum_per_record,
                "mean_status": ComputationStatus.NOT_COMPUTED.value,
            },
            "task_rows": [
                {
                    "task_id": row.task_id,
                    "observation_unit": ObservationUnit.TARGET_ROW.value,
                    "denominator": row.denominator,
                    "available": row.available,
                    "masked": row.masked,
                    "missing": row.missing,
                    "unsupported": row.unsupported,
                    "accepted_record_support": {
                        "observation_unit": ObservationUnit.RECORD.value,
                        "denominator": evidence.accepted_record_count,
                        "available": row.accepted_available_record_support,
                        "available_status": row.record_support_status.value,
                        "missing": row.accepted_missing_record_count,
                        "missing_status": ComputationStatus.OBSERVED.value,
                    },
                }
                for row in evidence.target_rows
            ],
        },
        "unavailable_analyses": {
            "canonical_work_identity": (
                evidence.canonical_work_identity_status.value
            ),
            "class_concentration": evidence.class_concentration_status.value,
            "cooccurrence": evidence.cooccurrence_status.value,
            "train_validation_shift": (
                evidence.train_validation_shift_status.value
            ),
            "reason": "tracked_split_by_class_rows_unavailable",
        },
    }


def dumps_pop909_cl_phase4_evidence(
    evidence: Pop909ClPhase4EvidenceReplay, *, indent: int | None = None
) -> str:
    """Serialize validated Phase 4 evidence with its semantic fingerprint."""

    payload = pop909_cl_phase4_evidence_dict(evidence)
    payload["semantic_fingerprint"] = evidence.semantic_fingerprint
    return dumps_canonical_json(payload, indent=indent)


def replay_pop909_cl_phase4_evidence(
    request: Pop909ClPhase4EvidenceRequest,
) -> Pop909ClPhase4EvidenceReplay:
    """Validate and replay the exact tracked Phase 4 aggregate evidence."""

    if type(request) is not Pop909ClPhase4EvidenceRequest:
        raise EDAContractError(
            "pop909_cl.eda.request_invalid",
            "Phase 4 replay requires Pop909ClPhase4EvidenceRequest",
        )
    raw_manifest, raw_fingerprint = _load_json(Path(request.raw_manifest_path))
    audit_manifest, audit_fingerprint = _load_json(
        Path(request.audit_manifest_path)
    )
    production_manifest, production_fingerprint = _load_json(
        Path(request.production_manifest_path)
    )
    if audit_fingerprint != POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256:
        raise EDAContractError(
            "pop909_cl.eda.phase4_audit_manifest_drift",
            "Phase 4A audit manifest bytes differ from the accepted artifact",
        )
    if production_fingerprint != POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256:
        raise EDAContractError(
            "pop909_cl.eda.phase4_production_manifest_drift",
            "Phase 4B production manifest bytes differ from the accepted artifact",
        )
    _validate_raw_manifest(raw_manifest)

    audit_corpus = _mapping(audit_manifest.get("corpus"), "audit corpus")
    audit_upstream = _mapping(audit_manifest.get("upstream"), "audit upstream")
    audit_aggregates = _mapping(
        audit_manifest.get("aggregates"), "audit aggregates"
    )
    audit_chords = _mapping(
        audit_aggregates.get("chord_annotation_inventory"),
        "audit chord inventory",
    )
    audit_masks = _mapping(
        audit_chords.get("task_mask_counts"), "audit task masks"
    )
    audit_score_crosswalk = _mapping(
        audit_aggregates.get("score_only_crosswalk"),
        "audit score-only crosswalk",
    )
    production_corpus = _mapping(
        production_manifest.get("corpus"), "production corpus"
    )
    production_expected = _mapping(
        production_manifest.get("expected"), "production expected"
    )
    raw_structure = _mapping(
        raw_manifest.get("raw_structure"), "raw structure"
    )
    raw_outliers = _mapping(raw_manifest.get("raw_outliers"), "raw outliers")
    note_distribution = _mapping(
        raw_structure.get("score_note_distribution"), "score note distribution"
    )
    warning_distribution = _mapping(
        raw_outliers.get("score_warning_distribution_per_converted_record"),
        "score warning distribution",
    )

    _expect(audit_manifest.get("audit_schema_version"), "3.0.0", "audit schema")
    _expect(
        production_manifest.get("acceptance_schema_version"),
        "2.0.0",
        "production acceptance schema",
    )
    for observed, name in (
        (audit_corpus.get("corpus_content_fingerprint"), "audit source"),
        (production_corpus.get("content_fingerprint"), "production source"),
    ):
        _expect(observed, _SOURCE_IDENTITY.fingerprint, f"{name} fingerprint")
    _expect(
        audit_upstream.get("commit"),
        "be9094392903c471a930519e1c0bacf8b6be5d62",
        "audit upstream commit",
    )
    _expect(
        production_corpus.get("upstream_commit"),
        audit_upstream.get("commit"),
        "production upstream commit",
    )

    expected_pairs = (
        ("logical_files", 909),
        ("accepted", 908),
        ("quarantine_count", 1),
        ("chord_instruments", 907),
        ("chord_blocks", 116055),
        ("ambiguous_blocks", 5801),
        ("unsupported_blocks", 586),
        ("boundary_available", 116055),
        ("bass_available", 116055),
        ("root_available", 109668),
        ("quality_available", 109800),
        ("inversion_available", 109668),
        ("derived_n_spans", 947),
        ("trailing_masked_spans", 151),
    )
    for field_name, expected in expected_pairs:
        _expect(
            production_expected.get(field_name),
            expected,
            f"production expected.{field_name}",
        )
    _expect(
        production_expected.get("quarantine_song_ids"),
        ["172"],
        "production quarantine identities",
    )
    _expect(
        production_expected.get("missing_target_song_ids"),
        ["367", "658"],
        "production missing-target identities",
    )
    for field_name, expected in (
        ("total_blocks", 116055),
        ("ambiguous", 5801),
        ("unsupported", 586),
        ("implicit_n_gap_count", 947),
        ("trailing_unannotated_span_count", 151),
        ("overlap_count", 691),
        ("duplicate_block_onset_count", 0),
        ("repeated_pitch_at_onset_block_count", 87),
        ("mixed_note_end_tick_block_count", 313),
    ):
        if field_name in {"ambiguous", "unsupported"}:
            statuses = _mapping(
                audit_chords.get("normalization_status_counts"),
                "normalization statuses",
            )
            observed = statuses.get(field_name)
        else:
            observed = audit_chords.get(field_name)
        _expect(observed, expected, f"audit chord inventory.{field_name}")
    for task_name, available, unavailable in (
        ("boundary", 116055, 0),
        ("bass", 116055, 0),
        ("root", 109668, 6387),
        ("quality", 109800, 6255),
        ("inversion", 109668, 6387),
        ("no_chord", 947, 151),
    ):
        mask = _mapping(audit_masks.get(task_name), f"audit mask {task_name}")
        _expect(mask.get("available"), available, f"{task_name}.available")
        _expect(mask.get("unavailable"), unavailable, f"{task_name}.unavailable")

    _expect(
        _mapping(
            audit_score_crosswalk.get("warnings_by_code"),
            "audit warning occurrences",
        ),
        _mapping(
            raw_outliers.get("score_warning_occurrences_by_code"),
            "raw warning occurrences",
        ),
        "raw/audit warning occurrences",
    )
    _expect(
        _mapping(
            audit_score_crosswalk.get("files_affected_by_warning_code"),
            "audit warning affected files",
        ),
        _mapping(
            raw_outliers.get("score_warning_affected_records_by_code"),
            "raw warning affected records",
        ),
        "raw/audit warning affected records",
    )

    pairing = _mapping(
        audit_chords.get("pairing_diagnostics"), "pairing diagnostics"
    )
    pairing_count = int(pairing.get("dangling_note_on", -1)) + int(
        pairing.get("unmatched_note_off", -1)
    )
    _expect(pairing_count, 8, "pairing anomaly count")

    block_denominator = int(production_expected["chord_blocks"])
    no_chord_denominator = (
        int(production_expected["derived_n_spans"])
        + int(production_expected["trailing_masked_spans"])
    )
    task_rows = (
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.bass",
            denominator=block_denominator,
            available=116055,
            masked=0,
            missing=0,
            unsupported=0,
            accepted_available_record_support=906,
            record_support_status=ComputationStatus.OBSERVED,
        ),
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.boundary",
            denominator=block_denominator,
            available=116055,
            masked=0,
            missing=0,
            unsupported=0,
            accepted_available_record_support=906,
            record_support_status=ComputationStatus.OBSERVED,
        ),
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.inversion",
            denominator=block_denominator,
            available=109668,
            masked=5801,
            missing=0,
            unsupported=586,
        ),
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.no_chord",
            denominator=no_chord_denominator,
            available=947,
            masked=151,
            missing=0,
            unsupported=0,
        ),
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.quality",
            denominator=block_denominator,
            available=109800,
            masked=5669,
            missing=0,
            unsupported=586,
        ),
        Pop909ClPhase4TaskAvailability(
            task_id="pop909_cl.chord.root",
            denominator=block_denominator,
            available=109668,
            masked=5801,
            missing=0,
            unsupported=586,
        ),
    )
    return Pop909ClPhase4EvidenceReplay(
        source_identity=_SOURCE_IDENTITY,
        producer_identity=_adapter_identity(),
        repository_commit=request.repository_commit,
        raw_manifest_identity=VersionedIdentity(
            identity="pop909_cl.eda.raw_manifest",
            version="1.0.0",
            fingerprint=raw_fingerprint,
        ),
        audit_manifest_identity=VersionedIdentity(
            identity="pop909_cl.phase_4a.audit_manifest",
            version="3.0.0",
            fingerprint=audit_fingerprint,
        ),
        production_manifest_identity=VersionedIdentity(
            identity="pop909_cl.phase_4b.production_manifest",
            version="2.0.0",
            fingerprint=production_fingerprint,
        ),
        logical_record_count=909,
        accepted_record_count=908,
        quarantined_record_ids=("172",),
        accepted_missing_target_record_ids=("367", "658"),
        source_records_with_chord_instrument=907,
        accepted_records_with_chord_evidence=906,
        target_rows=task_rows,
        chord_block_count=116055,
        ambiguous_block_count=5801,
        unsupported_block_count=586,
        trailing_uncovered_span_count=151,
        leading_internal_no_chord_span_count=947,
        overlap_count=691,
        repeated_pitch_block_count=87,
        mixed_end_block_count=313,
        pairing_anomaly_count=pairing_count,
        duplicate_block_onset_count=0,
        raw_pitch_class_set_count=261,
        selected_root_quality_bass_label_count=340,
        chord_block_observed_record_count=909,
        chord_block_minimum_per_record=0,
        chord_block_median_per_record=124,
        chord_block_p95_per_record=185,
        chord_block_maximum_per_record=278,
        midi_type_1_record_count=int(raw_structure["midi_type_1_records"]),
        ppqn_480_record_count=int(raw_structure["ppqn_480_records"]),
        empty_conductor_track_count=int(raw_structure["empty_conductor_tracks"]),
        tempo_event_count=int(raw_structure["tempo_events"]),
        meter_event_count=int(raw_structure["meter_events"]),
        key_signature_event_count=int(raw_structure["key_signature_events"]),
        score_note_observed_record_count=int(
            note_distribution["observed_records"]
        ),
        score_note_unknown_record_count=int(note_distribution["unknown_records"]),
        score_note_minimum=int(note_distribution["minimum"]),
        score_note_median=int(note_distribution["median"]),
        score_note_p95=int(note_distribution["p95"]),
        score_note_maximum=int(note_distribution["maximum"]),
        score_warning_observed_record_count=int(
            raw_outliers["score_warning_observed_records"]
        ),
        score_warning_unknown_record_count=int(warning_distribution["unknown_records"]),
        score_warning_occurrence_count=int(
            raw_outliers["score_warning_occurrences"]
        ),
        score_warning_minimum_per_record=int(warning_distribution["minimum"]),
        score_warning_median_per_record=int(warning_distribution["median"]),
        score_warning_p95_per_record=int(warning_distribution["p95"]),
        score_warning_maximum_per_record=int(warning_distribution["maximum"]),
        trailing_duration_minimum_ticks=1,
        trailing_duration_median_ticks=401,
        trailing_duration_p95_ticks=3361,
        trailing_duration_maximum_ticks=12861,
        class_concentration_status=ComputationStatus.NOT_COMPUTED,
        cooccurrence_status=ComputationStatus.NOT_COMPUTED,
        train_validation_shift_status=ComputationStatus.NOT_COMPUTED,
        canonical_work_identity_status=ComputationStatus.NOT_COMPUTED,
    )


def validate_pop909_cl_identity_splits(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Reject leakage through record, canonical-work, source-group or lineage.

    Shared identities close transitively: if A shares a source group with B
    and B shares lineage with C, all three records form one split component.
    A null canonical-work identity remains unknown and creates no equivalence.
    """

    if not rows:
        raise EDAContractError(
            "pop909_cl.eda.identity_plan_empty", "identity rows cannot be empty"
        )
    parent: dict[str, str] = {}
    split_by_record: dict[str, SplitScope] = {}
    identity_owner: dict[tuple[str, str], str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    normalized: list[tuple[str, SplitScope, Mapping[str, object]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EDAContractError(
                "pop909_cl.eda.identity_row_invalid",
                f"identity row {index} must be an object",
            )
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise EDAContractError(
                "pop909_cl.eda.identity_row_invalid",
                f"identity row {index} requires record_id",
            )
        try:
            split = SplitScope(row.get("split"))
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "pop909_cl.eda.identity_row_invalid",
                f"identity row {index} has an invalid split",
            ) from exc
        if split not in {SplitScope.TRAIN, SplitScope.VALIDATION}:
            raise EDAContractError(
                "pop909_cl.eda.identity_row_invalid",
                "identity evidence is restricted to TRAIN/VALIDATION",
            )
        if record_id in split_by_record:
            raise EDAContractError(
                "pop909_cl.eda.record_identity_duplicate",
                f"record identity occurs more than once: {record_id}",
            )
        split_by_record[record_id] = split
        find(record_id)
        normalized.append((record_id, split, row))

    for record_id, _, row in normalized:
        for field_name in ("source_group_id", "lineage_id", "canonical_work_id"):
            identity = row.get(field_name)
            if identity is None and field_name == "canonical_work_id":
                continue
            if not isinstance(identity, str) or not identity:
                raise EDAContractError(
                    "pop909_cl.eda.identity_row_invalid",
                    f"{field_name} must be non-empty or canonical_work_id must be null",
                )
            key = (field_name, identity)
            previous = identity_owner.setdefault(key, record_id)
            union(record_id, previous)

    component_splits: dict[str, set[SplitScope]] = defaultdict(set)
    component_records: dict[str, list[str]] = defaultdict(list)
    for record_id, split in split_by_record.items():
        component = find(record_id)
        component_splits[component].add(split)
        component_records[component].append(record_id)
    leaking = [
        tuple(sorted(component_records[component]))
        for component, splits in component_splits.items()
        if len(splits) > 1
    ]
    if leaking:
        raise EDAContractError(
            "pop909_cl.eda.identity_leakage",
            f"identity component crosses TRAIN/VALIDATION: {sorted(leaking)!r}",
        )


def _coverage(
    *,
    denominator: int | None,
    observed_count: int | None,
    unknown_count: int | None,
    split: SplitScope,
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
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
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
    split: SplitScope = SplitScope.ALL,
    evidence_scope: EvidenceScope = EvidenceScope.MANIFEST_REPLAY,
    provenance: tuple[str, ...] = _RAW_PROVENANCE,
) -> UnitCount:
    return UnitCount(
        name=name,
        observation_unit=observation_unit,
        value=value,
        denominator=denominator,
        denominator_unit=denominator_unit,
        split_scope=split,
        evidence_scope=evidence_scope,
        provenance=provenance,
    )


def _raw_metrics(manifest: Mapping[str, object]) -> tuple[RawMetricEvidence, ...]:
    inventory = _mapping(manifest["inventory"], "inventory")
    identity = _mapping(manifest["identity_evidence"], "identity_evidence")
    inventory_coverage = _coverage(
        denominator=909,
        observed_count=909,
        unknown_count=0,
        split=SplitScope.ALL,
        provenance=_RAW_PROVENANCE,
    )
    accepted_coverage = _coverage(
        denominator=908,
        observed_count=908,
        unknown_count=0,
        split=SplitScope.ALL,
        provenance=("pop909_cl.accepted_joint_split",),
    )
    graph_coverage = _coverage(
        denominator=908,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        provenance=_RAW_PROVENANCE,
        status=ComputationStatus.NOT_COMPUTED,
        reason="eda.target_free_unproven",
    )
    not_replayed = _coverage(
        denominator=909,
        observed_count=None,
        unknown_count=None,
        split=SplitScope.ALL,
        provenance=_RAW_PROVENANCE,
        status=ComputationStatus.NOT_COMPUTED,
        reason="pop909_cl.raw.metric_not_replayed",
    )
    metrics: list[RawMetricEvidence] = []
    for metric_id, spec in RAW_METRIC_CATALOG.items():
        if metric_id in {
            "accepted_records",
            "discovered_records",
            "quarantined_records",
        }:
            value = int(inventory[metric_id])
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    count=_count(
                        metric_id,
                        value,
                        denominator=909,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                    ),
                )
            )
        elif metric_id == "conversion_outcomes":
            outcomes = _mapping(
                inventory["conversion_outcomes"], "conversion_outcomes"
            )
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    categories=tuple(
                        CategoryCount(
                            category=category,
                            count=_count(
                                metric_id,
                                int(value),
                                denominator=909,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                            ),
                        )
                        for category, value in outcomes.items()
                    ),
                )
            )
        elif metric_id == "instruments":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=inventory_coverage,
                    categories=(
                        CategoryCount(
                            category="channel_0_score",
                            count=_count(
                                metric_id,
                                int(inventory["unique_channel_0_score_instruments"]),
                                denominator=909,
                                observation_unit=ObservationUnit.INSTRUMENT,
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
                    coverage=inventory_coverage,
                    categories=(
                        CategoryCount(
                            category=str(inventory["quarantine_reason"]),
                            count=_count(
                                metric_id,
                                1,
                                denominator=909,
                                observation_unit=ObservationUnit.RECORD,
                                denominator_unit=ObservationUnit.RECORD,
                            ),
                        ),
                    ),
                )
            )
        elif metric_id == "duplicate_candidates":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=accepted_coverage,
                    count=_count(
                        metric_id,
                        int(identity["raw_input_duplicate_record_count"]),
                        denominator=908,
                        observation_unit=ObservationUnit.RECORD,
                        denominator_unit=ObservationUnit.RECORD,
                        provenance=("pop909_cl.accepted_joint_split",),
                    ),
                )
            )
        elif metric_id == "cross_split_raw_identity_collisions":
            metrics.append(
                RawMetricEvidence(
                    metric_id=metric_id,
                    coverage=accepted_coverage,
                    count=_count(
                        metric_id,
                        int(identity["cross_split_raw_identity_collision_count"]),
                        denominator=908,
                        observation_unit=ObservationUnit.RAW_IDENTITY_COLLISION,
                        denominator_unit=ObservationUnit.RECORD,
                        provenance=("pop909_cl.accepted_joint_split",),
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
                RawMetricEvidence(metric_id=metric_id, coverage=not_replayed)
            )
    return tuple(metrics)


def _raw_extension(manifest: Mapping[str, object]) -> SourceExtension:
    identity = _mapping(manifest["identity_evidence"], "identity_evidence")
    cluster = _mapping(identity["raw_equivalence_cluster"], "raw_equivalence_cluster")
    split_counts = _mapping(
        identity["split_assignment_counts"], "split_assignment_counts"
    )
    outliers = _mapping(manifest["raw_outliers"], "raw_outliers")
    raw_structure = _mapping(manifest["raw_structure"], "raw_structure")
    note_distribution = _mapping(
        raw_structure["score_note_distribution"], "score_note_distribution"
    )
    warning_occurrences = _mapping(
        outliers["score_warning_occurrences_by_code"],
        "score_warning_occurrences_by_code",
    )
    warning_affected_records = _mapping(
        outliers["score_warning_affected_records_by_code"],
        "score_warning_affected_records_by_code",
    )
    warning_distribution = _mapping(
        outliers["score_warning_distribution_per_converted_record"],
        "score_warning_distribution_per_converted_record",
    )
    warning_names = {
        "EMPTY_TRACK": "empty_track_warning",
        "INCOMPLETE_FINAL_BAR": "incomplete_final_bar_warning",
        "OVERLAPPING_SAME_PITCH_NOTES": "same_pitch_overlap_warning",
        "PIECE_TRAILING_SILENCE": "piece_trailing_silence_warning",
    }
    extension_provenance = _RAW_PROVENANCE
    rows = (
        ExtensionRow(
            row_id="global_metadata_events",
            payload={"source_location": "conductor_track_0"},
            counts=(
                _count(
                    "key_signature_event_occurrences",
                    int(raw_structure["key_signature_events"]),
                    denominator=909,
                    observation_unit=ObservationUnit.EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "meter_event_occurrences",
                    int(raw_structure["meter_events"]),
                    denominator=909,
                    observation_unit=ObservationUnit.METER_EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "tempo_event_occurrences",
                    int(raw_structure["tempo_events"]),
                    denominator=909,
                    observation_unit=ObservationUnit.TEMPO_EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=909,
                observed_count=909,
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="identity_split_safety",
            payload={"assessment": "source_group_and_lineage_split_atomic"},
            counts=(
                _count(
                    "lineage_split_collisions",
                    int(identity["cross_split_lineage_collision_count"]),
                    denominator=908,
                    observation_unit=ObservationUnit.RAW_IDENTITY_COLLISION,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "source_group_split_collisions",
                    int(identity["cross_split_source_group_collision_count"]),
                    denominator=908,
                    observation_unit=ObservationUnit.RAW_IDENTITY_COLLISION,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=908,
                observed_count=908,
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="installation_noise",
            payload={"classification": "appledouble_installation_noise"},
            counts=(
                _count(
                    "appledouble_source_files",
                    int(outliers["appledouble_noise_files"]),
                    denominator=int(outliers["installation_source_files"]),
                    observation_unit=ObservationUnit.SOURCE_FILE,
                    denominator_unit=ObservationUnit.SOURCE_FILE,
                ),
            ),
            coverage=_coverage(
                denominator=int(outliers["installation_source_files"]),
                observed_count=int(outliers["installation_source_files"]),
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
                unit=ObservationUnit.SOURCE_FILE,
            ),
        ),
        ExtensionRow(
            row_id="meter_change_offsets",
            payload={
                "song_id": _mapping(
                    outliers["meter_change_offsets"], "meter_change_offsets"
                )["song_id"],
                "offset_measurements": (
                    {
                        "measurement_unit": _mapping(
                            outliers["meter_change_offsets"],
                            "meter_change_offsets",
                        )["unit"],
                        "values": _mapping(
                            outliers["meter_change_offsets"],
                            "meter_change_offsets",
                        )["values"],
                    },
                ),
            },
            coverage=_coverage(
                denominator=909,
                observed_count=1,
                unknown_count=908,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="midi_container",
            payload={"midi_file_type": 1, "ppqn": 480},
            counts=(
                _count(
                    "midi_type_1_records",
                    int(raw_structure["midi_type_1_records"]),
                    denominator=909,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "ppqn_480_records",
                    int(raw_structure["ppqn_480_records"]),
                    denominator=909,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "empty_conductor_tracks",
                    int(raw_structure["empty_conductor_tracks"]),
                    denominator=909,
                    observation_unit=ObservationUnit.TRACK,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=909,
                observed_count=909,
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="raw_equivalence_cluster",
            payload={
                "member_record_ids": cluster["member_record_ids"],
                "member_splits": cluster["member_splits"],
                "source_group_id": cluster["source_group_id"],
            },
            counts=(
                _count(
                    "clustered_records",
                    int(identity["raw_input_duplicate_record_count"]),
                    denominator=908,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=908,
                observed_count=908,
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="score_note_distribution",
            payload={
                "minimum": note_distribution["minimum"],
                "median": note_distribution["median"],
                "p95": note_distribution["p95"],
                "maximum": note_distribution["maximum"],
                "mean_status": note_distribution["mean_status"],
            },
            counts=(
                _count(
                    "observed_score_note_records",
                    int(note_distribution["observed_records"]),
                    denominator=909,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
                _count(
                    "unknown_score_note_records",
                    int(note_distribution["unknown_records"]),
                    denominator=909,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=909,
                observed_count=int(note_distribution["observed_records"]),
                unknown_count=int(note_distribution["unknown_records"]),
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="score_warning_occurrences",
            payload={
                "classification": "event_level_not_file_level",
                "minimum_per_converted_record": warning_distribution["minimum"],
                "median_per_converted_record": warning_distribution["median"],
                "p95_per_converted_record": warning_distribution["p95"],
                "maximum_per_converted_record": warning_distribution["maximum"],
                "mean_status": warning_distribution["mean_status"],
            },
            counts=tuple(
                _count(
                    f"{warning_names[warning_code]}_occurrences",
                    int(value),
                    denominator=909,
                    observation_unit=ObservationUnit.EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                )
                for warning_code, value in warning_occurrences.items()
            )
            + tuple(
                _count(
                    f"{warning_names[warning_code]}_affected_records",
                    int(value),
                    denominator=909,
                    observation_unit=ObservationUnit.RECORD,
                    denominator_unit=ObservationUnit.RECORD,
                )
                for warning_code, value in warning_affected_records.items()
            )
            + (
                _count(
                    "warning_occurrences",
                    int(outliers["score_warning_occurrences"]),
                    denominator=909,
                    observation_unit=ObservationUnit.EVENT,
                    denominator_unit=ObservationUnit.RECORD,
                ),
            ),
            coverage=_coverage(
                denominator=909,
                observed_count=int(outliers["score_warning_observed_records"]),
                unknown_count=1,
                split=SplitScope.ALL,
                provenance=extension_provenance,
            ),
        ),
        ExtensionRow(
            row_id="split_assignment_population",
            payload={"assignment_policy": "deterministic_seed_42_joint_split"},
            counts=tuple(
                _count(
                    f"{split_name}_assignment_rows",
                    int(split_counts[split_name]),
                    denominator=908,
                    observation_unit=ObservationUnit.SPLIT_ASSIGNMENT,
                    denominator_unit=ObservationUnit.SPLIT_ASSIGNMENT,
                )
                for split_name in ("train", "validation", "test")
            ),
            coverage=_coverage(
                denominator=908,
                observed_count=908,
                unknown_count=0,
                split=SplitScope.ALL,
                provenance=extension_provenance,
                unit=ObservationUnit.SPLIT_ASSIGNMENT,
            ),
        ),
    )
    return SourceExtension(
        corpus=CorpusId.POP909_CL,
        namespace=POP909_CL_RAW_EXTENSION_NAMESPACE,
        schema_name=POP909_CL_RAW_EXTENSION_SCHEMA,
        schema_version="1.0.0",
        split_scope=SplitScope.ALL,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        provenance=extension_provenance,
        rows=rows,
        target_free=True,
    )


def _validate_raw_manifest(manifest: Mapping[str, object]) -> None:
    _require_keys(
        manifest,
        {
            "corpus",
            "evidence_basis",
            "graph_evidence",
            "identity_evidence",
            "inventory",
            "raw_outliers",
            "raw_structure",
            "scan_policy",
            "schema",
            "source_release",
        },
        "raw manifest",
    )
    _expect(manifest.get("schema"), POP909_CL_RAW_EDA_MANIFEST_SCHEMA, "schema")
    _expect(manifest.get("corpus"), POP909_CL_DATASET_NAME, "corpus")
    source = _mapping(manifest.get("source_release"), "source_release")
    _require_keys(source, {"fingerprint", "identity", "version"}, "source_release")
    _expect(source.get("identity"), _SOURCE_IDENTITY.identity, "source identity")
    _expect(source.get("version"), _SOURCE_IDENTITY.version, "source version")
    _expect(
        source.get("fingerprint"),
        _SOURCE_IDENTITY.fingerprint,
        "source fingerprint",
    )
    _expect(
        manifest.get("evidence_basis"),
        [
            "phase_4a_raw_audit",
            "phase_4b_raw_acceptance",
            "phase_6c_split_audit",
        ],
        "evidence_basis",
    )
    inventory = _mapping(manifest.get("inventory"), "inventory")
    _require_keys(
        inventory,
        {
            "accepted_records",
            "conversion_outcomes",
            "discovered_records",
            "quarantine_reason",
            "quarantine_song_ids",
            "quarantined_records",
            "unique_channel_0_score_instruments",
        },
        "inventory",
    )
    for field_name, expected in (
        ("discovered_records", 909),
        ("accepted_records", 908),
        ("quarantined_records", 1),
        ("unique_channel_0_score_instruments", 909),
        ("quarantine_song_ids", ["172"]),
        ("quarantine_reason", "midi_adapter.meter_change_inside_bar"),
    ):
        _expect(inventory.get(field_name), expected, f"inventory.{field_name}")
    _expect(
        inventory.get("conversion_outcomes"),
        {"accepted": 908, "quarantined": 1},
        "inventory.conversion_outcomes",
    )
    identity = _mapping(manifest.get("identity_evidence"), "identity_evidence")
    _require_keys(
        identity,
        {
            "accepted_assignment_count",
            "assignment_manifest_file_sha256",
            "assignment_manifest_fingerprint",
            "canonical_work_identity_status",
            "cross_split_lineage_collision_count",
            "cross_split_raw_identity_collision_count",
            "cross_split_source_group_collision_count",
            "raw_equivalence_cluster",
            "raw_input_duplicate_cluster_count",
            "raw_input_duplicate_record_count",
            "raw_input_equivalence_group_count",
            "record_piece_id_count",
            "split_assignment_counts",
        },
        "identity_evidence",
    )
    for field_name, expected in (
        ("accepted_assignment_count", 908),
        (
            "assignment_manifest_fingerprint",
            "b0546316acb225bb95439dab78fab95232b0a7a758316b69b85dc87f733c384d",
        ),
        (
            "assignment_manifest_file_sha256",
            "a5b49cd7f48f87c66ed6656a223e576629373158b9f64b783c47d65e512e5385",
        ),
        ("record_piece_id_count", 908),
        ("raw_input_equivalence_group_count", 907),
        ("raw_input_duplicate_record_count", 2),
        ("raw_input_duplicate_cluster_count", 1),
        ("cross_split_raw_identity_collision_count", 0),
        ("cross_split_source_group_collision_count", 0),
        ("cross_split_lineage_collision_count", 0),
        ("canonical_work_identity_status", "not_computed"),
    ):
        _expect(identity.get(field_name), expected, f"identity_evidence.{field_name}")
    cluster = _mapping(identity.get("raw_equivalence_cluster"), "raw cluster")
    _require_keys(
        cluster,
        {"member_record_ids", "member_splits", "source_group_id"},
        "raw_equivalence_cluster",
    )
    _expect(
        cluster.get("member_record_ids"),
        ["piece:pop909-cl-543", "piece:pop909-cl-553"],
        "raw cluster members",
    )
    _expect(cluster.get("member_splits"), ["train", "train"], "raw cluster splits")
    _expect(
        cluster.get("source_group_id"),
        "pop909-cl-score:4585134e3f7a70c105a3bb678a04ab2bc4522c04e11183f6fd6c59046be25286",
        "raw cluster source group",
    )
    split_counts = _mapping(
        identity.get("split_assignment_counts"), "split assignment counts"
    )
    _require_keys(
        split_counts, {"test", "train", "validation"}, "split assignment counts"
    )
    _expect(
        split_counts,
        {"train": 701, "validation": 101, "test": 106},
        "split assignment counts",
    )
    outliers = _mapping(manifest.get("raw_outliers"), "raw_outliers")
    _require_keys(
        outliers,
        {
            "appledouble_noise_files",
            "installation_source_files",
            "meter_change_offsets",
            "same_pitch_overlap_warning_occurrences",
            "score_warning_affected_records_by_code",
            "score_warning_distribution_per_converted_record",
            "score_warning_observed_records",
            "score_warning_occurrences_by_code",
            "score_warning_occurrences",
        },
        "raw_outliers",
    )
    meter = _mapping(outliers.get("meter_change_offsets"), "meter_change_offsets")
    _require_keys(meter, {"song_id", "unit", "values"}, "meter_change_offsets")
    for field_name, expected in (
        ("appledouble_noise_files", 910),
        ("installation_source_files", 1819),
        ("same_pitch_overlap_warning_occurrences", 123439),
        ("score_warning_observed_records", 908),
        ("score_warning_occurrences", 126163),
    ):
        _expect(outliers.get(field_name), expected, f"raw_outliers.{field_name}")
    _expect(
        meter,
        {"song_id": "172", "unit": "tick", "values": [600, 480]},
        "meter_change_offsets",
    )
    _expect(
        _mapping(
            outliers.get("score_warning_occurrences_by_code"),
            "score_warning_occurrences_by_code",
        ),
        {
            "EMPTY_TRACK": 908,
            "INCOMPLETE_FINAL_BAR": 908,
            "OVERLAPPING_SAME_PITCH_NOTES": 123439,
            "PIECE_TRAILING_SILENCE": 908,
        },
        "score_warning_occurrences_by_code",
    )
    _expect(
        _mapping(
            outliers.get("score_warning_affected_records_by_code"),
            "score_warning_affected_records_by_code",
        ),
        {
            "EMPTY_TRACK": 908,
            "INCOMPLETE_FINAL_BAR": 908,
            "OVERLAPPING_SAME_PITCH_NOTES": 907,
            "PIECE_TRAILING_SILENCE": 908,
        },
        "score_warning_affected_records_by_code",
    )
    _expect(
        _mapping(
            outliers.get("score_warning_distribution_per_converted_record"),
            "score_warning_distribution_per_converted_record",
        ),
        {
            "observed_records": 908,
            "unknown_records": 1,
            "minimum": 3,
            "median": 123,
            "p95": 282,
            "maximum": 966,
            "mean_status": "not_computed",
        },
        "score_warning_distribution_per_converted_record",
    )
    raw_structure = _mapping(manifest.get("raw_structure"), "raw_structure")
    _require_keys(
        raw_structure,
        {
            "key_signature_events",
            "meter_events",
            "midi_type_1_records",
            "empty_conductor_tracks",
            "ppqn_480_records",
            "score_note_distribution",
            "tempo_events",
        },
        "raw_structure",
    )
    for field_name, expected in (
        ("midi_type_1_records", 909),
        ("ppqn_480_records", 909),
        ("empty_conductor_tracks", 909),
        ("tempo_events", 909),
        ("meter_events", 911),
        ("key_signature_events", 1065),
    ):
        _expect(
            raw_structure.get(field_name), expected, f"raw_structure.{field_name}"
        )
    note_distribution = _mapping(
        raw_structure.get("score_note_distribution"), "score_note_distribution"
    )
    _expect(
        note_distribution,
        {
            "observed_records": 908,
            "unknown_records": 1,
            "minimum": 175,
            "median": 1655,
            "p95": 2403,
            "maximum": 4233,
            "mean_status": "not_computed",
        },
        "score_note_distribution",
    )
    graph = _mapping(manifest.get("graph_evidence"), "graph_evidence")
    _require_keys(graph, {"reason", "status"}, "graph_evidence")
    _expect(
        graph,
        {"status": "not_computed", "reason": "eda.target_free_unproven"},
        "graph_evidence",
    )
    scan_policy = _mapping(manifest.get("scan_policy"), "scan_policy")
    _expect(
        scan_policy,
        {
            "production_scan_run": False,
            "corpus_files_opened": False,
            "midi_conversion_run": False,
            "graph_build_run": False,
        },
        "scan_policy",
    )


def _build_raw_report(
    request: Pop909ClRawEDARequest, adapter_identity: VersionedIdentity
) -> RawCorpusEDA:
    manifest_path = Path(request.manifest_path)
    manifest, manifest_fingerprint = _load_json(manifest_path)
    _validate_raw_manifest(manifest)
    identity = _mapping(manifest["identity_evidence"], "identity_evidence")
    graph = _mapping(manifest["graph_evidence"], "graph_evidence")
    _expect(graph.get("status"), "not_computed", "graph status")
    _expect(graph.get("reason"), "eda.target_free_unproven", "graph reason")
    envelope = ReportEnvelope(
        schema_name=RAW_CORPUS_EDA_SCHEMA_NAME,
        schema_version=RAW_CORPUS_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.RAW_CORPUS,
        corpus=CorpusId.POP909_CL,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.MANIFEST_REPLAY,
        execution_mode=ExecutionMode.MANIFEST_REPLAY,
        completeness_status=CompletenessStatus.PARTIAL,
        split_scope=SplitScope.ALL,
        observation_units=(
            ObservationUnit.EVENT,
            ObservationUnit.INSTRUMENT,
            ObservationUnit.METER_EVENT,
            ObservationUnit.RAW_IDENTITY_COLLISION,
            ObservationUnit.RECORD,
            ObservationUnit.SOURCE_FILE,
            ObservationUnit.SPLIT_ASSIGNMENT,
            ObservationUnit.TEMPO_EVENT,
            ObservationUnit.TRACK,
        ),
        input_manifests=(
            InputManifestRef(
                role="raw_projection",
                identity=VersionedIdentity(
                    identity="pop909_cl.eda.raw_manifest",
                    version="1.0.0",
                    fingerprint=manifest_fingerprint,
                ),
                target_free=True,
                repository_relative_path=_repository_relative_path(manifest_path),
            ),
            InputManifestRef(
                role="split_assignment",
                identity=VersionedIdentity(
                    identity="pop909_cl.accepted_joint_split",
                    version="1.0.0",
                    fingerprint=str(identity["assignment_manifest_fingerprint"]),
                ),
                target_free=True,
            ),
        ),
        invariants=(
            InvariantEvidence(
                code="pop909_cl.raw.inventory_partition",
                status=InvariantStatus.PASSED,
                provenance=_RAW_PROVENANCE,
            ),
            InvariantEvidence(
                code="pop909_cl.raw.lineage_split_atomic",
                status=InvariantStatus.PASSED,
                provenance=("pop909_cl.accepted_joint_split",),
            ),
            InvariantEvidence(
                code="pop909_cl.raw.record_identity_unique",
                status=InvariantStatus.PASSED,
                provenance=_RAW_PROVENANCE,
            ),
            InvariantEvidence(
                code="pop909_cl.raw.source_group_split_atomic",
                status=InvariantStatus.PASSED,
                provenance=("pop909_cl.accepted_joint_split",),
            ),
            InvariantEvidence(
                code="pop909_cl.raw.work_identity",
                status=InvariantStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                reason_code="eda.work_identity_unproven",
            ),
        ),
        warnings=(
            StructuredWarning(
                code="pop909_cl.raw.warning_density_high",
                message=(
                    "Score warnings are event-level diagnostics and must not be "
                    "interpreted as failed-record cardinality."
                ),
                provenance=_RAW_PROVENANCE,
            ),
        ),
        unavailable_reasons=(
            UnavailableReason(
                code="pop909_cl.raw.metrics_not_replayed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                detail=(
                    "The accepted aggregate manifests do not contain every common "
                    "numeric distribution."
                ),
            ),
            UnavailableReason(
                code="eda.target_free_unproven",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_RAW_PROVENANCE,
                detail=(
                    "Aggregate graph distributions are absent from the accepted "
                    "raw evidence manifest."
                ),
            ),
            UnavailableReason(
                code="eda.work_identity_unproven",
                status=ComputationStatus.NOT_APPLICABLE,
                provenance=_RAW_PROVENANCE,
                detail=(
                    "Record, score-equivalence, and lineage identities do not prove "
                    "a canonical work ontology."
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


def _vocabulary_identity(task_id: str) -> VersionedIdentity:
    spec = TARGET_FAMILY_BY_ID[task_id]
    return VersionedIdentity(
        identity=f"{task_id}.vocabulary",
        version=TARGET_ONTOLOGY_VERSION,
        fingerprint=canonical_json_sha256(
            {
                "task_id": task_id,
                "value_type": spec.value_type,
                "vocabulary": list(spec.vocabulary or ()),
            }
        ),
    )


def _split_manifest(split_path: Path) -> tuple[dict[str, object], str]:
    """Load and validate only the target-free assignment/identity manifest."""

    split_manifest, split_fingerprint = _load_json(split_path)
    _require_keys(
        split_manifest,
        {"assignments", "corpus", "identity_rows", "schema"},
        "split manifest",
    )
    _expect(split_manifest.get("schema"), POP909_CL_EDA_SPLIT_SCHEMA, "split schema")
    _expect(split_manifest.get("corpus"), POP909_CL_DATASET_NAME, "split corpus")
    return split_manifest, split_fingerprint


def _validate_supervision_fixture(
    fixture: Mapping[str, object], retained_assignment_ids: set[object]
) -> Mapping[str, object]:
    """Validate target-bearing fixture content inside an allowed loader call."""

    _require_keys(
        fixture,
        {
            "corpus",
            "ontology_fingerprint",
            "ontology_version",
            "records",
            "schema",
        },
        "supervision fixture",
    )
    _expect(
        fixture.get("schema"),
        POP909_CL_EDA_SUPERVISION_FIXTURE_SCHEMA,
        "supervision fixture schema",
    )
    _expect(fixture.get("corpus"), POP909_CL_DATASET_NAME, "fixture corpus")
    _expect(
        fixture.get("ontology_version"),
        TARGET_ONTOLOGY_VERSION,
        "fixture ontology version",
    )
    _expect(
        fixture.get("ontology_fingerprint"),
        ontology_contract_fingerprint(),
        "fixture ontology fingerprint",
    )
    records = _mapping(fixture.get("records"), "fixture records")
    if set(records) != retained_assignment_ids:
        raise EDAContractError(
            "pop909_cl.eda.fixture_assignment_mismatch",
            "fixture record IDs must match retained assignments exactly",
        )
    for record_id, record_object in records.items():
        record = _mapping(record_object, f"fixture record {record_id}")
        _require_keys(record, {"rows", "split"}, f"fixture record {record_id}")
        for index, row_object in enumerate(
            _sequence(record.get("rows"), f"fixture rows {record_id}")
        ):
            row = _mapping(row_object, f"fixture row {record_id}[{index}]")
            _require_keys(
                row,
                {"state", "task_id", "value"},
                f"fixture row {record_id}[{index}]",
            )
    return records


def _build_task_rows(
    loaded_records: Sequence[Mapping[str, object]],
) -> tuple[TaskFamilyEvidence, ...]:
    by_split_task: dict[
        tuple[SplitScope, str], list[tuple[str, Mapping[str, object]]]
    ] = defaultdict(list)
    record_populations: dict[SplitScope, set[str]] = defaultdict(set)
    for record in loaded_records:
        _require_keys(record, {"record_id", "rows", "split"}, "loaded record")
        record_id = record["record_id"]
        split = SplitScope(record["split"])
        assert isinstance(record_id, str)
        record_populations[split].add(record_id)
        rows = _sequence(record["rows"], f"rows for {record_id}")
        task_ids: list[str] = []
        for row_object in rows:
            row = _mapping(row_object, f"row for {record_id}")
            _require_keys(row, {"state", "task_id", "value"}, "fixture row")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or task_id not in POP909_CL_TASKS:
                raise EDAContractError(
                    "pop909_cl.eda.fixture_task_invalid",
                    f"unknown source-native task {task_id!r}",
                )
            task_ids.append(task_id)
            by_split_task[(split, task_id)].append((record_id, row))
        if tuple(sorted(task_ids)) != tuple(sorted(POP909_CL_TASKS)):
            raise EDAContractError(
                "pop909_cl.eda.fixture_task_incomplete",
                f"{record_id} must contain exactly the six source-native tasks",
            )

    if not record_populations[SplitScope.TRAIN] or not record_populations[
        SplitScope.VALIDATION
    ]:
        raise EDAContractError(
            "pop909_cl.eda.fixture_split_incomplete",
            "fixture evidence requires both TRAIN and VALIDATION records",
        )

    result: list[TaskFamilyEvidence] = []
    for split in (SplitScope.TRAIN, SplitScope.VALIDATION):
        record_denominator = len(record_populations[split])
        for task_id in POP909_CL_TASKS:
            rows = by_split_task[(split, task_id)]
            spec = TARGET_FAMILY_BY_ID[task_id]
            state_counts: Counter[str] = Counter()
            class_records: dict[str, set[str]] = defaultdict(set)
            class_occurrences: Counter[str] = Counter()
            for record_id, row in rows:
                state = row.get("state")
                if state not in {"available", "masked", "missing", "unsupported"}:
                    raise EDAContractError(
                        "pop909_cl.eda.fixture_state_invalid",
                        f"invalid availability state {state!r}",
                    )
                state_counts[str(state)] += 1
                value = row.get("value")
                if state == "available":
                    if (
                        not isinstance(value, str)
                        or spec.vocabulary is None
                        or value not in spec.vocabulary
                    ):
                        raise EDAContractError(
                            "pop909_cl.eda.fixture_value_invalid",
                            f"{task_id} has an invalid available value {value!r}",
                        )
                    class_occurrences[value] += 1
                    class_records[value].add(record_id)
                elif value is not None:
                    raise EDAContractError(
                        "pop909_cl.eda.fixture_value_invalid",
                        "non-available fixture rows require null values",
                    )
            available = state_counts["available"]
            support = tuple(
                ClassSupport(
                    source_value=SourceValueIdentity(
                        corpus=CorpusId.POP909_CL,
                        source_task_id=task_id,
                        dialect="pop909_cl.channel_1",
                        source_value=value,
                        value_kind=SourceValueKind.SCALAR,
                    ),
                    occurrence_count=UnitCount(
                        name="occurrence_count",
                        observation_unit=ObservationUnit.LABEL_OCCURRENCE,
                        value=count,
                        denominator=available,
                        denominator_unit=ObservationUnit.TARGET_ROW,
                        split_scope=split,
                        evidence_scope=EvidenceScope.FIXTURE,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    unique_record_count=UnitCount(
                        name="unique_record_count",
                        observation_unit=ObservationUnit.RECORD,
                        value=len(class_records[value]),
                        denominator=record_denominator,
                        denominator_unit=ObservationUnit.RECORD,
                        split_scope=split,
                        evidence_scope=EvidenceScope.FIXTURE,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    unique_work_count=UnitCount(
                        name="unique_work_count",
                        observation_unit=ObservationUnit.LOGICAL_WORK,
                        value=None,
                        denominator=None,
                        denominator_unit=ObservationUnit.LOGICAL_WORK,
                        split_scope=split,
                        evidence_scope=EvidenceScope.FIXTURE,
                        provenance=_SUPERVISION_PROVENANCE,
                        status=ComputationStatus.NOT_APPLICABLE,
                        reason_code="eda.work_identity_unproven",
                    ),
                )
                for value, count in sorted(class_occurrences.items())
            )
            result.append(
                TaskFamilyEvidence(
                    corpus=CorpusId.POP909_CL,
                    source_task_id=task_id,
                    dialect="pop909_cl.channel_1",
                    annotation_namespace="pop909_cl.chord",
                    vocabulary=_vocabulary_identity(task_id),
                    label_granularity=spec.granularity,
                    label_value_type=LabelValueType.CATEGORICAL,
                    observation_unit=ObservationUnit.TARGET_ROW,
                    split_scope=split,
                    evidence_scope=EvidenceScope.FIXTURE,
                    provenance=_SUPERVISION_PROVENANCE,
                    status=ComputationStatus.OBSERVED,
                    availability=AvailabilityCounts(
                        observation_unit=ObservationUnit.TARGET_ROW,
                        denominator=len(rows),
                        available=available,
                        masked=state_counts["masked"],
                        missing=state_counts["missing"],
                        unsupported=state_counts["unsupported"],
                        split_scope=split,
                        evidence_scope=EvidenceScope.FIXTURE,
                        provenance=_SUPERVISION_PROVENANCE,
                    ),
                    work_identity=None,
                    class_support=support,
                    projection_availability=(),
                    projections=(),
                )
            )
    return tuple(result)


def _build_supervision_report(
    request: Pop909ClSupervisionEDARequest,
    adapter_identity: VersionedIdentity,
) -> SupervisionEDA:
    split_path = Path(request.split_manifest_path)
    fixture_path = Path(request.supervision_fixture_path)
    split_manifest, split_fingerprint = _split_manifest(split_path)
    assignments_input = _sequence(split_manifest.get("assignments"), "assignments")
    identities_input = _sequence(split_manifest.get("identity_rows"), "identity_rows")
    identity_rows = tuple(
        _mapping(value, f"identity_rows[{index}]")
        for index, value in enumerate(identities_input)
    )
    for row in identity_rows:
        _require_keys(
            row,
            {
                "canonical_work_id",
                "lineage_id",
                "record_id",
                "source_group_id",
                "split",
            },
            "identity row",
        )
    validate_pop909_cl_identity_splits(identity_rows)
    identity_ids = {row["record_id"] for row in identity_rows}
    projected_assignments: list[dict[str, object]] = []
    retained_assignment_ids: set[object] = set()
    for index, value in enumerate(assignments_input):
        assignment = _mapping(value, f"assignments[{index}]")
        if assignment.get("split") == SplitScope.TEST.value:
            projected_assignments.append({"split": SplitScope.TEST.value})
            continue
        if set(assignment) != {"record_id", "split"}:
            raise EDAContractError(
                "pop909_cl.eda.assignment_invalid",
                "retained fixture assignments require exactly record_id and split",
            )
        record_id = assignment["record_id"]
        retained_assignment_ids.add(record_id)
        projected_assignments.append(
            {
                "assignment_manifest_fingerprint": split_fingerprint,
                "corpus": CorpusId.POP909_CL.value,
                "record_id": record_id,
                "split": assignment["split"],
                "target_free": True,
            }
        )
    if retained_assignment_ids != identity_ids:
        raise EDAContractError(
            "pop909_cl.eda.identity_assignment_mismatch",
            "identity rows must match retained split assignments exactly",
        )
    fixture_records: Mapping[str, object] | None = None
    fixture_fingerprint: str | None = None

    def resolve_descriptor(record_id: str, split: SplitScope) -> str:
        if request.descriptor_observer is not None:
            request.descriptor_observer(record_id, split)
        return record_id

    def load_record(record_id: str, split: SplitScope) -> Mapping[str, object]:
        nonlocal fixture_fingerprint, fixture_records
        if request.loader_observer is not None:
            request.loader_observer(record_id, split)
        if fixture_records is None:
            fixture, observed_fingerprint = _load_json(fixture_path)
            fixture_records = _validate_supervision_fixture(
                fixture, retained_assignment_ids
            )
            if observed_fingerprint != POP909_CL_EDA_SUPERVISION_FIXTURE_SHA256:
                raise EDAContractError(
                    "pop909_cl.eda.supervision_fixture_drift",
                    "supervision fixture bytes differ from the accepted artifact",
                )
            fixture_fingerprint = observed_fingerprint
        try:
            fixture_record = fixture_records[record_id]
        except KeyError as exc:
            raise EDAContractError(
                "pop909_cl.eda.fixture_record_missing",
                f"missing fixture record {record_id}",
            ) from exc
        source = _mapping(fixture_record, f"fixture record {record_id}")
        _require_keys(source, {"rows", "split"}, f"fixture record {record_id}")
        if source.get("split") != split.value:
            raise EDAContractError(
                "pop909_cl.eda.fixture_split_mismatch",
                f"fixture record {record_id} disagrees with assignment split",
            )
        return {"record_id": record_id, **source}

    loaded, test_lock = load_supervision_train_validation_only(
        CorpusId.POP909_CL,
        tuple(projected_assignments),
        resolve_descriptor=resolve_descriptor,
        load_target=load_record,
        evidence_scope=EvidenceScope.FIXTURE,
        provenance=_GUARD_PROVENANCE,
    )
    if fixture_fingerprint is None:
        raise EDAContractError(
            "pop909_cl.eda.fixture_not_loaded",
            "allowed supervision assignments did not load the target fixture",
        )
    task_rows = _build_task_rows(loaded)
    envelope = ReportEnvelope(
        schema_name=SUPERVISION_EDA_SCHEMA_NAME,
        schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.SUPERVISION,
        corpus=CorpusId.POP909_CL,
        source_identity=_SOURCE_IDENTITY,
        producer_identity=adapter_identity,
        repository_commit=request.repository_commit,
        evidence_scope=EvidenceScope.FIXTURE,
        execution_mode=ExecutionMode.SYNTHETIC_FIXTURE,
        completeness_status=CompletenessStatus.COMPLETE,
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
                    identity="pop909_cl.eda.fixture_split_assignment",
                    version="1.0.0",
                    fingerprint=split_fingerprint,
                ),
                target_free=True,
                repository_relative_path=_repository_relative_path(split_path),
            ),
            InputManifestRef(
                role="fixture_rows",
                identity=VersionedIdentity(
                    identity="pop909_cl.eda.supervision_fixture",
                    version="1.0.0",
                    fingerprint=fixture_fingerprint,
                ),
                target_free=False,
                repository_relative_path=_repository_relative_path(fixture_path),
            ),
        ),
        invariants=(
            InvariantEvidence(
                code="pop909_cl.supervision.lineage_split_atomic",
                status=InvariantStatus.PASSED,
                provenance=_SUPERVISION_PROVENANCE,
            ),
            InvariantEvidence(
                code="pop909_cl.supervision.record_identity_unique",
                status=InvariantStatus.PASSED,
                provenance=_SUPERVISION_PROVENANCE,
            ),
            InvariantEvidence(
                code="pop909_cl.supervision.source_group_split_atomic",
                status=InvariantStatus.PASSED,
                provenance=_SUPERVISION_PROVENANCE,
            ),
            InvariantEvidence(
                code="pop909_cl.supervision.work_identity",
                status=InvariantStatus.NOT_COMPUTED,
                provenance=_SUPERVISION_PROVENANCE,
                reason_code="eda.work_identity_unproven",
            ),
        ),
        warnings=(
            StructuredWarning(
                code="pop909_cl.supervision.fixture_scope",
                message=(
                    "Class support is complete only for the synthetic fixture and "
                    "must not be presented as corpus-wide evidence."
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
                    "Source-group and lineage identities are leakage constraints, "
                    "not a proven canonical work ontology."
                ),
            ),
        ),
    )
    return SupervisionEDA(
        envelope=envelope,
        semantic_payload=SupervisionEDAPayload(
            tasks=task_rows,
            test_lock=test_lock,
            extensions=(),
        ),
    )


def _build_unavailable_supervision_report(
    request: Pop909ClUnavailableSupervisionEDARequest,
    adapter_identity: VersionedIdentity,
) -> SupervisionEDA:
    reason_code = "pop909_cl.supervision.split_rows_not_computed"
    tasks = tuple(
        TaskFamilyEvidence(
            corpus=CorpusId.POP909_CL,
            source_task_id=task_id,
            dialect="pop909_cl.channel_1",
            annotation_namespace="pop909_cl.chord",
            vocabulary=_vocabulary_identity(task_id),
            label_granularity=TARGET_FAMILY_BY_ID[task_id].granularity,
            label_value_type=LabelValueType.CATEGORICAL,
            observation_unit=ObservationUnit.TARGET_ROW,
            split_scope=split,
            evidence_scope=EvidenceScope.UNKNOWN,
            provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
            status=ComputationStatus.NOT_COMPUTED,
            availability=None,
            reason_code=reason_code,
        )
        for split in (SplitScope.TRAIN, SplitScope.VALIDATION)
        for task_id in POP909_CL_TASKS
    )
    test_lock = TestTargetLockEvidence.not_executed(
        evidence_scope=EvidenceScope.UNKNOWN,
        provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
        reason_code="eda.test_targets_locked",
    )
    envelope = ReportEnvelope(
        schema_name=SUPERVISION_EDA_SCHEMA_NAME,
        schema_version=SUPERVISION_EDA_SCHEMA_VERSION,
        report_kind=ReportKind.SUPERVISION,
        corpus=CorpusId.POP909_CL,
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
        invariants=(),
        warnings=(),
        unavailable_reasons=(
            UnavailableReason(
                code=reason_code,
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
                detail=(
                    "Tracked Phase 4 aggregates predate the split and cannot be "
                    "recast as TRAIN or VALIDATION target rows."
                ),
            ),
            UnavailableReason(
                code="pop909_cl.supervision.class_concentration_not_computed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
                detail=(
                    "No tracked split-by-class occurrence and record-support rows "
                    "are available."
                ),
            ),
            UnavailableReason(
                code="pop909_cl.supervision.cooccurrence_not_computed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
                detail=(
                    "No tracked split-specific task co-occurrence rows are "
                    "available."
                ),
            ),
            UnavailableReason(
                code="pop909_cl.supervision.train_validation_shift_not_computed",
                status=ComputationStatus.NOT_COMPUTED,
                provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
                detail=(
                    "No tracked TRAIN and VALIDATION class distributions exist for "
                    "a shift comparison."
                ),
            ),
            UnavailableReason(
                code="eda.work_identity_unproven",
                status=ComputationStatus.NOT_APPLICABLE,
                provenance=_UNAVAILABLE_SUPERVISION_PROVENANCE,
                detail=(
                    "Record, score-equivalence, and lineage identities do not prove "
                    "a canonical work ontology."
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


@dataclass(frozen=True, slots=True)
class Pop909ClEDAAdapter:
    """Source-owned raw and supervision adapter registered through EDA gates."""

    corpus: CorpusId = CorpusId.POP909_CL
    adapter_identity: VersionedIdentity = field(default_factory=_adapter_identity)
    extension_namespaces: tuple[str, ...] = (POP909_CL_RAW_EXTENSION_NAMESPACE,)

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        if type(request) is not Pop909ClRawEDARequest:
            raise EDAContractError(
                "pop909_cl.eda.request_invalid",
                "raw EDA requires Pop909ClRawEDARequest",
            )
        return _build_raw_report(request, self.adapter_identity)

    def build_supervision_eda(self, request: object) -> SupervisionEDA:
        if type(request) is Pop909ClSupervisionEDARequest:
            return _build_supervision_report(request, self.adapter_identity)
        if type(request) is Pop909ClUnavailableSupervisionEDARequest:
            return _build_unavailable_supervision_report(
                request, self.adapter_identity
            )
        raise EDAContractError(
            "pop909_cl.eda.request_invalid",
            "supervision EDA requires a POP909-CL supervision request",
        )


__all__ = [
    "EDA_CONTRACT_SHA",
    "POP909_CL_EDA_ADAPTER_VERSION",
    "POP909_CL_EDA_SPLIT_SCHEMA",
    "POP909_CL_EDA_SUPERVISION_FIXTURE_SCHEMA",
    "POP909_CL_EDA_SUPERVISION_FIXTURE_SHA256",
    "POP909_CL_PHASE4_AUDIT_MANIFEST_SHA256",
    "POP909_CL_PHASE4_EVIDENCE_SCHEMA",
    "POP909_CL_PHASE4_PRODUCTION_MANIFEST_SHA256",
    "POP909_CL_RAW_EDA_MANIFEST_SCHEMA",
    "POP909_CL_RAW_EXTENSION_NAMESPACE",
    "Pop909ClEDAAdapter",
    "Pop909ClPhase4EvidenceReplay",
    "Pop909ClPhase4EvidenceRequest",
    "Pop909ClPhase4TaskAvailability",
    "Pop909ClRawEDARequest",
    "Pop909ClSupervisionEDARequest",
    "Pop909ClUnavailableSupervisionEDARequest",
    "dumps_pop909_cl_phase4_evidence",
    "pop909_cl_phase4_evidence_dict",
    "replay_pop909_cl_phase4_evidence",
    "validate_pop909_cl_identity_splits",
]
