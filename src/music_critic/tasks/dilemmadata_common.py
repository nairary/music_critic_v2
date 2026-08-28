"""Phase 9E-A target-only common harmonic projection for Dilemmadata.

The contracts in this module deliberately consume an already validated
``TargetBundle``.  They neither inspect nor alter a :class:`CanonicalPiece`, a
raw graph, cache identity, grouping, or split assignment.  The projection is a
separate, SHA-bound sidecar and keeps every source-native value and mapping
state auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
)
from music_critic.tasks.multisource import (
    TARGET_BUNDLE_CONTRACT_VERSION,
    SampleTarget,
    TargetBundle,
    target_bundle_fingerprint,
)


DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION = "1.0.0"
DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION = "1.0.0"
DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION = "1.0.0"
DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION = "1.0.0"
DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION = "1.0.0"
ANALYSISGNN_REFERENCE_MAPPING_VERSION = "1.0.1"
COMMON_QUALITY_TEMPLATE_VERSION = "1.0.0"

ANALYSISGNN_REPOSITORY = "https://github.com/manoskary/analysisgnn"
ANALYSISGNN_REFERENCE_COMMIT = "e115182fb29b74bdcb6bf3547ed427d967580947"
ANALYSISGNN_LICENSE_SPDX = "MIT"
ANALYSISGNN_REFERENCE_FILES = (
    (
        "LICENSE",
        "429692666c4b76c1d69fa738571b2ba1f1a2e6c3c2e6d460d347210b7f58695c",
    ),
    (
        "analysisgnn/utils/chord_representations.py",
        "49be2e51e5d89f28e989b3c5045730938ce26996d23d54b74c1f597bac4adabc",
    ),
    (
        "analysisgnn/utils/dcl_tsv_utils.py",
        "26a9a3fd5628dc063dab9258eb1f6011d1901d32802791f0560e0e2d96ad8a2f",
    ),
    (
        "analysisgnn/utils/globals.py",
        "205886a94409dba5c9a41c393be3b8714163b0b1f828221ef19fc7b2973b86da",
    ),
)

COMMON_QUALITY_TASK = "dilemmadata.common.chord.quality"
COMMON_INVERSION_TASK = "dilemmadata.common.chord.inversion"
COMMON_ROOT_PC_TASK = "dilemmadata.common.chord.root_pc"
COMMON_BASS_PC_TASK = "dilemmadata.common.chord.bass_pc"
COMMON_LOCAL_KEY_TASK = "dilemmadata.common.key.local"
COMMON_PITCH_CLASS_SET_TASK = "dilemmadata.common.chord.pitch_class_set"

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

COMMON_TASK_IDS = (
    COMMON_BASS_PC_TASK,
    COMMON_INVERSION_TASK,
    COMMON_PITCH_CLASS_SET_TASK,
    COMMON_QUALITY_TASK,
    COMMON_ROOT_PC_TASK,
    COMMON_LOCAL_KEY_TASK,
)

MappingState = Literal[
    "exact",
    "coarsened",
    "ambiguous",
    "unsupported",
    "invalid",
]
ProjectionState = Literal[
    "exact",
    "coarsened",
    "ambiguous",
    "unsupported",
    "invalid",
    "missing",
    "masked",
]
ReferenceAgreement = Literal["agree", "diverge", "not_applicable"]
CommonMode = Literal["major", "minor", "unknown", "other"]
CommonValue: TypeAlias = str | int | tuple[int, ...] | None

_SUPERVISION_STATES = frozenset({"exact", "coarsened"})
_MAPPING_STATES = frozenset(
    {"exact", "coarsened", "ambiguous", "unsupported", "invalid"}
)
_PROJECTION_STATES = frozenset({*_MAPPING_STATES, "missing", "masked"})
_REFERENCE_AGREEMENTS = frozenset({"agree", "diverge", "not_applicable"})
_COMMON_MODES = frozenset({"major", "minor", "unknown", "other"})
_OVERLAP_COMPARISONS = frozenset(
    {
        "exact_agreement",
        "enharmonic_only_agreement",
        "coarsened_agreement",
        "conflict",
        "unavailable",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_PITCH_RE = re.compile(r"^([A-Ga-g])((?:#{1,3}|-{1,3}|b{1,3})?)$")


class DilemmadataCommonProjectionError(ValueError):
    """Stable failure at the Phase 9E-A common-projection boundary."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(f"[{category}] {message}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _validate_identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.identifier_invalid",
            f"{name} must be a non-empty stripped string",
        )


@dataclass(frozen=True, slots=True)
class AnalysisGNNReferenceMapping:
    """Pinned, source-attributed AnalysisGNN mapping evidence."""

    contract_version: str
    repository: str
    commit_sha: str
    license_spdx: str
    files: tuple[tuple[str, str], ...]
    dlc_quality_rows: tuple[tuple[str, str], ...]
    inversion_rows: tuple[tuple[str, str, str], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != ANALYSISGNN_REFERENCE_MAPPING_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_version_invalid",
                "AnalysisGNN reference mapping version is incompatible",
            )
        if self.repository != ANALYSISGNN_REPOSITORY or self.commit_sha != (
            ANALYSISGNN_REFERENCE_COMMIT
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_binding_invalid",
                "AnalysisGNN reference must use the pinned repository and commit",
            )
        if self.license_spdx != ANALYSISGNN_LICENSE_SPDX:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_license_invalid",
                "AnalysisGNN reference license differs from the audited MIT license",
            )
        if self.files != tuple(sorted(self.files)) or any(
            not path or not _is_sha256(digest) for path, digest in self.files
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_files_invalid",
                "AnalysisGNN file evidence must be sorted and SHA-256 bound",
            )
        if self.dlc_quality_rows != tuple(sorted(self.dlc_quality_rows)):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_rows_invalid",
                "AnalysisGNN quality rows must use deterministic source-value order",
            )
        inversion_keys = tuple(
            (source_task_id, source_value)
            for source_task_id, source_value, _reference_value in self.inversion_rows
        )
        if (
            self.inversion_rows != tuple(sorted(self.inversion_rows))
            or inversion_keys != tuple(sorted(set(inversion_keys)))
            or any(
                source_task_id not in {_AN_INVERSION_TASK, _DLC_INVERSION_TASK}
                or not source_value
                or reference_value not in {"root", "first", "second", "third"}
                for source_task_id, source_value, reference_value in self.inversion_rows
            )
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_rows_invalid",
                "AnalysisGNN inversion rows must use unique, deterministic "
                "source-task/value identity",
            )
        expected = _fingerprint(_analysisgnn_payload(self, clear=True))
        if self.fingerprint != expected:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_fingerprint_invalid",
                "AnalysisGNN reference fingerprint differs from its evidence",
            )


def _analysisgnn_payload(
    mapping: AnalysisGNNReferenceMapping, *, clear: bool = False
) -> dict[str, object]:
    value = asdict(mapping)
    value["fingerprint"] = "" if clear else mapping.fingerprint
    return value


_ANALYSISGNN_DLC_QUALITY = {
    "%7": "half-diminished seventh chord",
    "+": "augmented triad",
    "+7": "augmented triad",
    "+M7": "augmented triad",
    "Fr": "French augmented sixth chord",
    "Ger": "German augmented sixth chord",
    "It": "Italian augmented sixth chord",
    "M": "major triad",
    "MM7": "major seventh chord",
    "Mm7": "dominant seventh chord",
    "m": "minor triad",
    "mM7": "minor-augmented tetrachord",
    "mm7": "minor seventh chord",
    "o": "diminished triad",
    "o7": "diminished seventh chord",
}

_ANALYSISGNN_INVERSION_BY_SOURCE = {
    _AN_INVERSION_TASK: {
        "0": "root",
        "1": "first",
        "2": "second",
        "3": "third",
    },
    _DLC_INVERSION_TASK: {
        "2": "third",
        "43": "second",
        "6": "first",
        "64": "second",
        "65": "first",
        "7": "root",
    },
}


def _make_analysisgnn_reference() -> AnalysisGNNReferenceMapping:
    values = {
        "contract_version": ANALYSISGNN_REFERENCE_MAPPING_VERSION,
        "repository": ANALYSISGNN_REPOSITORY,
        "commit_sha": ANALYSISGNN_REFERENCE_COMMIT,
        "license_spdx": ANALYSISGNN_LICENSE_SPDX,
        "files": tuple(sorted(ANALYSISGNN_REFERENCE_FILES)),
        "dlc_quality_rows": tuple(sorted(_ANALYSISGNN_DLC_QUALITY.items())),
        "inversion_rows": tuple(
            sorted(
                (source_task_id, source_value, reference_value)
                for source_task_id, mapping in (
                    _ANALYSISGNN_INVERSION_BY_SOURCE.items()
                )
                for source_value, reference_value in mapping.items()
            )
        ),
    }
    return AnalysisGNNReferenceMapping(
        **values,
        fingerprint=_fingerprint({**values, "fingerprint": ""}),
    )


ANALYSISGNN_REFERENCE = _make_analysisgnn_reference()
_ANALYSISGNN_INVERSION_BY_KEY = MappingProxyType(
    {
        (source_task_id, source_value): reference_value
        for source_task_id, source_value, reference_value in (
            ANALYSISGNN_REFERENCE.inversion_rows
        )
    }
)


@dataclass(frozen=True, slots=True)
class DilemmadataCommonMappingEvidence:
    """One explicit source-label to common-label mapping decision."""

    contract_version: str
    evidence_id: str
    dialect: Literal["an_joint", "dlc"]
    source_task_id: str
    source_value: str
    common_task_id: str
    state: MappingState
    common_value: CommonValue
    information_loss: tuple[str, ...]
    diagnostic_code: str | None
    analysisgnn_reference_value: str | None
    analysisgnn_agreement: ReferenceAgreement
    rationale: str

    def __post_init__(self) -> None:
        if self.contract_version != DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_version_invalid",
                "mapping evidence version is incompatible",
            )
        for name in (
            "evidence_id",
            "source_task_id",
            "source_value",
            "common_task_id",
            "rationale",
        ):
            _validate_identifier(getattr(self, name), name)
        if self.state not in _MAPPING_STATES:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_state_invalid",
                "mapping state is outside the frozen Phase 9E-A vocabulary",
            )
        if self.analysisgnn_agreement not in _REFERENCE_AGREEMENTS:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_parity_invalid",
                "AnalysisGNN agreement is outside the frozen vocabulary",
            )
        if self.state in _SUPERVISION_STATES and self.common_value is None:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_value_missing",
                "exact/coarsened mappings require a common value",
            )
        if self.state not in _SUPERVISION_STATES and self.common_value is not None:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_value_invalid",
                "ambiguous/unsupported/invalid mappings cannot expose a value",
            )
        if self.state == "coarsened" and not self.information_loss:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_loss_missing",
                "coarsened mappings must name the information loss",
            )
        if self.state != "coarsened" and self.information_loss and not (
            self.common_task_id in {COMMON_ROOT_PC_TASK, COMMON_BASS_PC_TASK}
            and self.state == "exact"
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_loss_invalid",
                "information loss is reserved for coarsening or exact pitch-class reduction",
            )
        if self.state in {"ambiguous", "unsupported", "invalid"} and not (
            isinstance(self.diagnostic_code, str) and self.diagnostic_code
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.mapping_diagnostic_missing",
                "non-supervising mapping states require a stable diagnostic",
            )
        if self.analysisgnn_agreement == "not_applicable":
            if self.analysisgnn_reference_value is not None:
                raise DilemmadataCommonProjectionError(
                    "dilemmadata.common.analysisgnn_parity_invalid",
                    "not-applicable AnalysisGNN rows cannot define a reference value",
                )
        elif self.analysisgnn_reference_value is None:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.analysisgnn_parity_invalid",
                "AnalysisGNN agreement/divergence requires a reference value",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonQualityTemplate:
    quality: str
    intervals: tuple[int, ...]
    template_version: str = COMMON_QUALITY_TEMPLATE_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.quality, "quality")
        if self.template_version != COMMON_QUALITY_TEMPLATE_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.template_version_invalid",
                "quality template version is incompatible",
            )
        if (
            not self.intervals
            or self.intervals[0] != 0
            or self.intervals != tuple(sorted(set(self.intervals)))
            or any(
                isinstance(interval, bool)
                or not isinstance(interval, int)
                or not 0 <= interval <= 11
                for interval in self.intervals
            )
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.template_invalid",
                "quality template must be a unique sorted pitch-class interval set",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonFamilySpec:
    task_id: str
    source_task_ids: tuple[str, ...]
    value_kind: Literal[
        "categorical",
        "pitch_class",
        "factorized_local_key",
        "pitch_class_set",
    ]
    vocabulary: tuple[str, ...] | None
    value_fields: tuple[str, ...]
    sharing_status: Literal["shared", "not_shared", "deferred"]
    mapping_policy: str
    model_ready: bool

    def __post_init__(self) -> None:
        _validate_identifier(self.task_id, "task_id")
        if (
            not self.source_task_ids
            or self.source_task_ids != tuple(sorted(set(self.source_task_ids)))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.family_sources_invalid",
                "common family source tasks must be non-empty, unique, and sorted",
            )
        if self.vocabulary is not None and self.vocabulary != tuple(
            sorted(set(self.vocabulary))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.family_vocabulary_invalid",
                "common vocabulary must be unique and sorted",
            )
        if not self.value_fields or self.value_fields != tuple(
            sorted(set(self.value_fields))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.family_fields_invalid",
                "common value fields must be non-empty, unique, and sorted",
            )
        _validate_identifier(self.mapping_policy, "mapping_policy")


def _normal_quality(value: str) -> str:
    return value


_QUALITY_COLLAPSES = {
    "French augmented sixth chord in first inversion": (
        "French augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "French augmented sixth chord in root position": (
        "French augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "French augmented sixth chord in third inversion": (
        "French augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "German augmented sixth chord in root position": (
        "German augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "German augmented sixth chord in second inversion": (
        "German augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "German augmented sixth chord in third inversion": (
        "German augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "Italian augmented sixth chord in root position": (
        "Italian augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "Italian augmented sixth chord in second inversion": (
        "Italian augmented sixth chord",
        "source_quality_embeds_inversion",
    ),
    "enharmonic equivalent to diminished triad": (
        "diminished triad",
        "enharmonic_relation_removed",
    ),
    "enharmonic equivalent to half-diminished seventh chord": (
        "half-diminished seventh chord",
        "enharmonic_relation_removed",
    ),
    "enharmonic equivalent to major triad": (
        "major triad",
        "enharmonic_relation_removed",
    ),
    "enharmonic equivalent to minor seventh chord": (
        "minor seventh chord",
        "enharmonic_relation_removed",
    ),
    "enharmonic equivalent to minor triad": (
        "minor triad",
        "enharmonic_relation_removed",
    ),
    "enharmonic to dominant seventh chord": (
        "dominant seventh chord",
        "enharmonic_relation_removed",
    ),
}

_DLC_PRECISE_QUALITY = {
    "%7": "half-diminished seventh chord",
    "+": "augmented triad",
    "+7": "augmented seventh chord",
    "+M7": "augmented major tetrachord",
    "Fr": "French augmented sixth chord",
    "Ger": "German augmented sixth chord",
    "It": "Italian augmented sixth chord",
    "M": "major triad",
    "MM7": "major seventh chord",
    "Mm7": "dominant seventh chord",
    "m": "minor triad",
    "mM7": "minor-augmented tetrachord",
    "mm7": "minor seventh chord",
    "o": "diminished triad",
    "o7": "diminished seventh chord",
}

_QUALITY_TEMPLATES = tuple(
    DilemmadataCommonQualityTemplate(quality, intervals)
    for quality, intervals in sorted(
        {
            "augmented triad": (0, 4, 8),
            "diminished seventh chord": (0, 3, 6, 9),
            "diminished triad": (0, 3, 6),
            "dominant seventh chord": (0, 4, 7, 10),
            "half-diminished seventh chord": (0, 3, 6, 10),
            "major seventh chord": (0, 4, 7, 11),
            "major triad": (0, 4, 7),
            "minor seventh chord": (0, 3, 7, 10),
            "minor triad": (0, 3, 7),
            "note": (0,),
        }.items()
    )
)
_QUALITY_TEMPLATE_BY_VALUE = MappingProxyType(
    {row.quality: row for row in _QUALITY_TEMPLATES}
)


def _quality_evidence_rows() -> tuple[DilemmadataCommonMappingEvidence, ...]:
    rows: list[DilemmadataCommonMappingEvidence] = []
    an_vocabulary = DILEMMADATA_SOURCE_FAMILY_BY_TASK[_AN_QUALITY_TASK].vocabulary
    dlc_vocabulary = DILEMMADATA_SOURCE_FAMILY_BY_TASK[_DLC_QUALITY_TASK].vocabulary
    assert an_vocabulary is not None and dlc_vocabulary is not None
    for source_value in an_vocabulary:
        if source_value in _QUALITY_COLLAPSES:
            common_value, loss = _QUALITY_COLLAPSES[source_value]
            state: MappingState = "coarsened"
            losses = (loss,)
            rationale = (
                "The common quality removes source spelling/inversion detail while "
                "preserving the evidenced harmonic quality."
            )
        else:
            common_value = _normal_quality(source_value)
            state = "exact"
            losses = ()
            rationale = "The common label preserves the source quality semantics."
        reference = source_value if source_value in {
            value for value in _ANALYSISGNN_DLC_QUALITY.values()
        } else None
        agreement: ReferenceAgreement = (
            "agree"
            if reference is not None and common_value == reference
            else "not_applicable"
        )
        rows.append(
            DilemmadataCommonMappingEvidence(
                contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
                evidence_id=(
                    "mapping:dilemmadata-common:an-quality:"
                    + sha256(source_value.encode("utf-8")).hexdigest()[:16]
                ),
                dialect="an_joint",
                source_task_id=_AN_QUALITY_TASK,
                source_value=source_value,
                common_task_id=COMMON_QUALITY_TASK,
                state=state,
                common_value=common_value,
                information_loss=losses,
                diagnostic_code=None,
                analysisgnn_reference_value=reference,
                analysisgnn_agreement=agreement,
                rationale=rationale,
            )
        )
    for source_value in dlc_vocabulary:
        common_value = _DLC_PRECISE_QUALITY[source_value]
        reference = _ANALYSISGNN_DLC_QUALITY[source_value]
        agreement = "agree" if common_value == reference else "diverge"
        rows.append(
            DilemmadataCommonMappingEvidence(
                contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
                evidence_id=(
                    "mapping:dilemmadata-common:dlc-quality:"
                    + sha256(source_value.encode("utf-8")).hexdigest()[:16]
                ),
                dialect="dlc",
                source_task_id=_DLC_QUALITY_TASK,
                source_value=source_value,
                common_task_id=COMMON_QUALITY_TASK,
                state="exact",
                common_value=common_value,
                information_loss=(),
                diagnostic_code=None,
                analysisgnn_reference_value=reference,
                analysisgnn_agreement=agreement,
                rationale=(
                    "The common label preserves the documented DLC chord type; "
                    "the +7/+M7 divergences reject AnalysisGNN's acknowledged "
                    "augmented-triad collapse."
                ),
            )
        )
    expected = {
        (_AN_QUALITY_TASK, value) for value in an_vocabulary
    } | {(_DLC_QUALITY_TASK, value) for value in dlc_vocabulary}
    observed = {(row.source_task_id, row.source_value) for row in rows}
    if expected != observed:
        raise RuntimeError("common quality registry does not cover every source row")
    return tuple(sorted(rows, key=lambda row: (row.source_task_id, row.source_value)))


_INVERSION_BY_SOURCE = {
    _AN_INVERSION_TASK: {"0": "root", "1": "first", "2": "second", "3": "third"},
    _DLC_INVERSION_TASK: {
        "2": "third",
        "43": "second",
        "6": "first",
        "64": "second",
        "65": "first",
        "7": "root",
    },
}


def _inversion_evidence_rows() -> tuple[DilemmadataCommonMappingEvidence, ...]:
    rows: list[DilemmadataCommonMappingEvidence] = []
    for task_id, mapping in sorted(_INVERSION_BY_SOURCE.items()):
        dialect: Literal["an_joint", "dlc"] = (
            "an_joint" if task_id == _AN_INVERSION_TASK else "dlc"
        )
        vocabulary = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id].vocabulary
        assert vocabulary is not None and set(vocabulary) == set(mapping)
        for source_value, common_value in sorted(mapping.items()):
            reference = _ANALYSISGNN_INVERSION_BY_KEY[(task_id, source_value)]
            rows.append(
                DilemmadataCommonMappingEvidence(
                    contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
                    evidence_id=(
                        f"mapping:dilemmadata-common:{dialect}-inversion:"
                        + sha256(source_value.encode("utf-8")).hexdigest()[:16]
                    ),
                    dialect=dialect,
                    source_task_id=task_id,
                    source_value=source_value,
                    common_task_id=COMMON_INVERSION_TASK,
                    state="exact",
                    common_value=common_value,
                    information_loss=(),
                    diagnostic_code=None,
                    analysisgnn_reference_value=reference,
                    analysisgnn_agreement=(
                        "agree" if common_value == reference else "diverge"
                    ),
                    rationale=(
                        "The source ordinal or figured bass determines one common "
                        "ordinal; projection still validates chord cardinality."
                    ),
                )
            )
    return tuple(sorted(rows, key=lambda row: (row.source_task_id, row.source_value)))


_QUALITY_MAPPING_ROWS = _quality_evidence_rows()
_INVERSION_MAPPING_ROWS = _inversion_evidence_rows()
_QUALITY_MAPPING_BY_KEY = MappingProxyType(
    {(row.source_task_id, row.source_value): row for row in _QUALITY_MAPPING_ROWS}
)
_INVERSION_MAPPING_BY_KEY = MappingProxyType(
    {(row.source_task_id, row.source_value): row for row in _INVERSION_MAPPING_ROWS}
)
_COMMON_QUALITY_VOCABULARY = tuple(
    sorted(
        {
            str(row.common_value)
            for row in _QUALITY_MAPPING_ROWS
            if row.common_value is not None
        }
    )
)


_COMMON_FAMILIES = tuple(
    sorted(
        (
            DilemmadataCommonFamilySpec(
                task_id=COMMON_QUALITY_TASK,
                source_task_ids=tuple(sorted((_AN_QUALITY_TASK, _DLC_QUALITY_TASK))),
                value_kind="categorical",
                vocabulary=_COMMON_QUALITY_VOCABULARY,
                value_fields=("quality",),
                sharing_status="shared",
                mapping_policy="explicit_quality_rows_v1",
                model_ready=True,
            ),
            DilemmadataCommonFamilySpec(
                task_id=COMMON_INVERSION_TASK,
                source_task_ids=tuple(
                    sorted((_AN_INVERSION_TASK, _DLC_INVERSION_TASK))
                ),
                value_kind="categorical",
                vocabulary=("first", "root", "second", "third"),
                value_fields=("inversion",),
                sharing_status="shared",
                mapping_policy="ordinal_with_cardinality_validation_v1",
                model_ready=True,
            ),
            DilemmadataCommonFamilySpec(
                task_id=COMMON_ROOT_PC_TASK,
                source_task_ids=tuple(sorted((_AN_ROOT_TASK, _DLC_ROOT_TASK))),
                value_kind="pitch_class",
                vocabulary=None,
                value_fields=("pitch_class",),
                sharing_status="shared",
                mapping_policy="spelling_or_tpc_to_pitch_class_v1",
                model_ready=True,
            ),
            DilemmadataCommonFamilySpec(
                task_id=COMMON_BASS_PC_TASK,
                source_task_ids=tuple(sorted((_AN_BASS_TASK, _DLC_BASS_TASK))),
                value_kind="pitch_class",
                vocabulary=None,
                value_fields=("pitch_class",),
                sharing_status="shared",
                mapping_policy="spelling_or_tpc_to_pitch_class_v1",
                model_ready=True,
            ),
            DilemmadataCommonFamilySpec(
                task_id=COMMON_LOCAL_KEY_TASK,
                source_task_ids=tuple(
                    sorted((_AN_LOCAL_KEY_TASK, _DLC_LOCAL_KEY_TASK))
                ),
                value_kind="factorized_local_key",
                vocabulary=None,
                value_fields=("mode", "tonic_pc"),
                sharing_status="shared",
                mapping_policy="factorized_spelling_or_tpc_plus_mode_v1",
                model_ready=True,
            ),
            DilemmadataCommonFamilySpec(
                task_id=COMMON_PITCH_CLASS_SET_TASK,
                source_task_ids=tuple(
                    sorted((_AN_QUALITY_TASK, _AN_ROOT_TASK, _DLC_QUALITY_TASK, _DLC_ROOT_TASK))
                ),
                value_kind="pitch_class_set",
                vocabulary=None,
                value_fields=("pitch_classes",),
                sharing_status="shared",
                mapping_policy="mapped_root_plus_proven_quality_template_v1",
                model_ready=True,
            ),
        ),
        key=lambda row: row.task_id,
    )
)


@dataclass(frozen=True, slots=True)
class DilemmadataCommonHarmonicRegistry:
    """Frozen common-family, mapping, and reference registry."""

    contract_version: str
    families: tuple[DilemmadataCommonFamilySpec, ...]
    quality_mapping_rows: tuple[DilemmadataCommonMappingEvidence, ...]
    inversion_mapping_rows: tuple[DilemmadataCommonMappingEvidence, ...]
    quality_templates: tuple[DilemmadataCommonQualityTemplate, ...]
    audit_only_or_deferred_families: tuple[tuple[str, str], ...]
    analysisgnn_reference: AnalysisGNNReferenceMapping
    fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_version_invalid",
                "common harmonic registry version is incompatible",
            )
        task_ids = tuple(row.task_id for row in self.families)
        if task_ids != tuple(sorted(set(task_ids))) or task_ids != tuple(
            sorted(COMMON_TASK_IDS)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_families_invalid",
                "common registry must contain the exact sorted MVP family inventory",
            )
        quality_keys = tuple(
            (row.source_task_id, row.source_value) for row in self.quality_mapping_rows
        )
        inversion_keys = tuple(
            (row.source_task_id, row.source_value)
            for row in self.inversion_mapping_rows
        )
        if quality_keys != tuple(sorted(set(quality_keys))) or inversion_keys != tuple(
            sorted(set(inversion_keys))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_rows_invalid",
                "mapping rows must be unique and deterministically sorted",
            )
        if self.quality_templates != tuple(
            sorted(self.quality_templates, key=lambda row: row.quality)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_templates_invalid",
                "quality templates must be deterministically sorted",
            )
        if self.audit_only_or_deferred_families != tuple(
            sorted(self.audit_only_or_deferred_families)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_deferred_invalid",
                "deferred family inventory must be sorted",
            )
        expected = _fingerprint(_registry_payload(self, clear=True))
        if self.fingerprint != expected:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_fingerprint_invalid",
                "common registry fingerprint differs from its contents",
            )


def _registry_payload(
    registry: DilemmadataCommonHarmonicRegistry, *, clear: bool = False
) -> dict[str, object]:
    value = asdict(registry)
    value["fingerprint"] = "" if clear else registry.fingerprint
    return value


def _make_registry() -> DilemmadataCommonHarmonicRegistry:
    values = {
        "contract_version": DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION,
        "families": _COMMON_FAMILIES,
        "quality_mapping_rows": _QUALITY_MAPPING_ROWS,
        "inversion_mapping_rows": _INVERSION_MAPPING_ROWS,
        "quality_templates": _QUALITY_TEMPLATES,
        "audit_only_or_deferred_families": tuple(
            sorted(
                (
                    ("borrowed_harmony", "not_shared:not_provided_by_dilemmadata"),
                    ("cadence", "not_shared:dlc_only"),
                    ("global_key", "not_shared:dlc_only"),
                    ("note_scale_degree", "deferred:tonal_spelling_crosswalk"),
                    ("phrase", "not_shared:dlc_only"),
                    ("roman_numeral", "deferred:syntax_coverage_audit"),
                    ("section", "not_shared:dlc_only"),
                    ("secondary_harmony", "deferred:semantic_crosswalk"),
                    ("semantic_voice_role", "not_shared:must_not_infer_from_staff_voice"),
                )
            )
        ),
        "analysisgnn_reference": ANALYSISGNN_REFERENCE,
    }
    serializable = {
        "contract_version": values["contract_version"],
        "families": [asdict(row) for row in values["families"]],
        "quality_mapping_rows": [asdict(row) for row in values["quality_mapping_rows"]],
        "inversion_mapping_rows": [asdict(row) for row in values["inversion_mapping_rows"]],
        "quality_templates": [asdict(row) for row in values["quality_templates"]],
        "audit_only_or_deferred_families": [list(row) for row in values["audit_only_or_deferred_families"]],
        "analysisgnn_reference": asdict(values["analysisgnn_reference"]),
        "fingerprint": "",
    }
    return DilemmadataCommonHarmonicRegistry(
        **values,
        fingerprint=_fingerprint(serializable),
    )


DILEMMADATA_COMMON_HARMONIC_REGISTRY = _make_registry()
DILEMMADATA_COMMON_FAMILY_BY_TASK = MappingProxyType(
    {row.task_id: row for row in DILEMMADATA_COMMON_HARMONIC_REGISTRY.families}
)


def dilemmadata_common_registry_dict() -> dict[str, object]:
    return _registry_payload(DILEMMADATA_COMMON_HARMONIC_REGISTRY)


def dumps_dilemmadata_common_registry(*, indent: int | None = None) -> str:
    return json.dumps(
        dilemmadata_common_registry_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def dilemmadata_common_registry_fingerprint() -> str:
    return DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint


@dataclass(frozen=True, slots=True)
class DilemmadataCommonLocalKeyValue:
    tonic_pc: int
    mode: CommonMode

    def __post_init__(self) -> None:
        if (
            isinstance(self.tonic_pc, bool)
            or not isinstance(self.tonic_pc, int)
            or not 0 <= self.tonic_pc <= 11
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.local_key_invalid",
                "local-key tonic pitch class must be an integer in [0, 11]",
            )
        if self.mode not in _COMMON_MODES:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.local_key_invalid",
                "local-key mode is outside the frozen Phase 9E-A vocabulary",
            )


ProjectionValue: TypeAlias = (
    str | int | tuple[int, ...] | DilemmadataCommonLocalKeyValue | None
)


@dataclass(frozen=True, slots=True)
class DilemmadataCommonTargetEntry:
    """One target-sidecar entry with explicit mapping and field masks."""

    entity_id: str
    source_task_ids: tuple[str, ...]
    source_values: tuple[str | None, ...]
    state: ProjectionState
    common_value: ProjectionValue
    field_availability: tuple[tuple[str, bool], ...]
    information_loss: tuple[str, ...]
    diagnostic_code: str | None
    mapping_evidence_ids: tuple[str, ...]
    source_provenance_ids: tuple[str, ...]
    dependency_entity_ids: tuple[str, ...]
    supplemental_source_fields: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.entity_id, "entity_id")
        if self.state not in _PROJECTION_STATES:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_state_invalid",
                "entry state is outside the frozen Phase 9E-A vocabulary",
            )
        if (
            not self.source_task_ids
            or self.source_task_ids != tuple(sorted(set(self.source_task_ids)))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_sources_invalid",
                "entry source tasks must be non-empty, unique, and sorted",
            )
        if len(self.source_values) != len(self.source_task_ids):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_sources_invalid",
                "entry source values must correspond to source task IDs",
            )
        field_names = tuple(name for name, _available in self.field_availability)
        if field_names != tuple(sorted(set(field_names))) or not all(
            isinstance(available, bool)
            for _name, available in self.field_availability
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_fields_invalid",
                "entry field masks must be unique, sorted booleans",
            )
        available = self.state in _SUPERVISION_STATES
        if available != (self.common_value is not None):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_value_invalid",
                "only exact/coarsened entries may expose a common value",
            )
        if available and not any(value for _name, value in self.field_availability):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_fields_invalid",
                "an available entry must expose at least one available value field",
            )
        if not available and any(value for _name, value in self.field_availability):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_fields_invalid",
                "masked mapping states cannot expose available value fields",
            )
        if self.state == "coarsened" and not self.information_loss:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_loss_missing",
                "coarsened entries must record their information loss",
            )
        if self.state in {"ambiguous", "unsupported", "invalid"} and not (
            isinstance(self.diagnostic_code, str) and self.diagnostic_code
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_diagnostic_missing",
                "non-supervising mapping outcomes require a diagnostic",
            )
        if self.mapping_evidence_ids != tuple(sorted(set(self.mapping_evidence_ids))):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_evidence_invalid",
                "entry mapping evidence IDs must be unique and sorted",
            )
        if self.source_provenance_ids != tuple(
            sorted(set(self.source_provenance_ids))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_provenance_invalid",
                "entry source provenance IDs must be unique and sorted",
            )
        if self.dependency_entity_ids != tuple(
            sorted(set(self.dependency_entity_ids))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_dependencies_invalid",
                "entry dependency IDs must be unique and sorted",
            )
        if self.supplemental_source_fields != tuple(
            sorted(self.supplemental_source_fields)
        ) or len(tuple(key for key, _value in self.supplemental_source_fields)) != len(
            {key for key, _value in self.supplemental_source_fields}
        ) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in self.supplemental_source_fields
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.entry_supplemental_invalid",
                "supplemental source fields must be uniquely sorted",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonTarget:
    task_id: str
    alignment_type: str
    value_fields: tuple[str, ...]
    entries: tuple[DilemmadataCommonTargetEntry, ...]

    def __post_init__(self) -> None:
        if self.task_id not in DILEMMADATA_COMMON_FAMILY_BY_TASK:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.target_task_invalid",
                "common target task is absent from the registry",
            )
        _validate_identifier(self.alignment_type, "alignment_type")
        expected_fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[
            self.task_id
        ].value_fields
        if self.value_fields != expected_fields:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.target_fields_invalid",
                "common target fields differ from the registry",
            )
        identities = tuple(entry.entity_id for entry in self.entries)
        if identities != tuple(sorted(identities)) or len(identities) != len(
            set(identities)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.target_order_invalid",
                "common target entries must be uniquely sorted by entity ID",
            )
        if any(
            tuple(name for name, _available in entry.field_availability)
            != self.value_fields
            for entry in self.entries
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.target_fields_invalid",
                "common target entry masks differ from family fields",
            )
        family = DILEMMADATA_COMMON_FAMILY_BY_TASK[self.task_id]
        for entry in self.entries:
            if entry.state not in _SUPERVISION_STATES:
                continue
            value = entry.common_value
            if family.value_kind == "categorical":
                if not isinstance(value, str) or value not in (family.vocabulary or ()):
                    raise DilemmadataCommonProjectionError(
                        "dilemmadata.common.target_value_invalid",
                        "categorical common value is absent from the frozen vocabulary",
                    )
            elif family.value_kind == "pitch_class":
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 11:
                    raise DilemmadataCommonProjectionError(
                        "dilemmadata.common.target_value_invalid",
                        "pitch-class common value must be an integer in [0, 11]",
                    )
            elif family.value_kind == "factorized_local_key":
                if not isinstance(value, DilemmadataCommonLocalKeyValue):
                    raise DilemmadataCommonProjectionError(
                        "dilemmadata.common.target_value_invalid",
                        "local-key common value must be factorized",
                    )
            elif family.value_kind == "pitch_class_set":
                if (
                    not isinstance(value, tuple)
                    or not value
                    or value != tuple(sorted(set(value)))
                    or any(
                        isinstance(pc, bool)
                        or not isinstance(pc, int)
                        or not 0 <= pc <= 11
                        for pc in value
                    )
                ):
                    raise DilemmadataCommonProjectionError(
                        "dilemmadata.common.target_value_invalid",
                        "pitch-class-set value must be a non-empty sorted unique tuple",
                    )


def _projection_value_payload(value: ProjectionValue) -> object:
    return asdict(value) if isinstance(value, DilemmadataCommonLocalKeyValue) else value


@dataclass(frozen=True, slots=True)
class DilemmadataCommonHarmonicProjection:
    """Versioned immutable common target sidecar bound to one TargetBundle."""

    contract_version: str
    dataset_id: str
    piece_id: str
    analysis_view_id: str
    source_target_bundle_contract_version: str
    source_target_bundle_fingerprint: str
    common_registry_fingerprint: str
    targets: tuple[DilemmadataCommonTarget, ...]
    projection_fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.projection_version_invalid",
                "common projection version is incompatible",
            )
        for name in ("dataset_id", "piece_id", "analysis_view_id"):
            _validate_identifier(getattr(self, name), name)
        if self.source_target_bundle_contract_version != TARGET_BUNDLE_CONTRACT_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.source_bundle_version_invalid",
                "common projection requires TargetBundle@1.0.0",
            )
        if not _is_sha256(self.source_target_bundle_fingerprint):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.source_bundle_fingerprint_invalid",
                "source target bundle fingerprint must be SHA-256",
            )
        if self.common_registry_fingerprint != (
            DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.registry_binding_invalid",
                "projection is bound to a different common registry",
            )
        tasks = tuple(target.task_id for target in self.targets)
        if tasks != tuple(sorted(COMMON_TASK_IDS)):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.projection_tasks_invalid",
                "projection must contain the exact sorted common MVP inventory",
            )
        if self.projection_fingerprint != _fingerprint(
            _projection_payload(self, clear=True)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.projection_fingerprint_invalid",
                "projection fingerprint differs from its contents",
            )


def _entry_payload(entry: DilemmadataCommonTargetEntry) -> dict[str, object]:
    return {
        "common_value": _projection_value_payload(entry.common_value),
        "dependency_entity_ids": list(entry.dependency_entity_ids),
        "diagnostic_code": entry.diagnostic_code,
        "entity_id": entry.entity_id,
        "field_availability": [list(row) for row in entry.field_availability],
        "information_loss": list(entry.information_loss),
        "mapping_evidence_ids": list(entry.mapping_evidence_ids),
        "source_provenance_ids": list(entry.source_provenance_ids),
        "source_task_ids": list(entry.source_task_ids),
        "source_values": list(entry.source_values),
        "state": entry.state,
        "supplemental_source_fields": [
            list(row) for row in entry.supplemental_source_fields
        ],
    }


def _projection_payload(
    projection: DilemmadataCommonHarmonicProjection, *, clear: bool = False
) -> dict[str, object]:
    return {
        "analysis_view_id": projection.analysis_view_id,
        "common_registry_fingerprint": projection.common_registry_fingerprint,
        "contract_version": projection.contract_version,
        "dataset_id": projection.dataset_id,
        "piece_id": projection.piece_id,
        "projection_fingerprint": "" if clear else projection.projection_fingerprint,
        "source_target_bundle_contract_version": (
            projection.source_target_bundle_contract_version
        ),
        "source_target_bundle_fingerprint": (
            projection.source_target_bundle_fingerprint
        ),
        "targets": [
            {
                "alignment_type": target.alignment_type,
                "entries": [_entry_payload(entry) for entry in target.entries],
                "task_id": target.task_id,
                "value_fields": list(target.value_fields),
            }
            for target in projection.targets
        ],
    }


def _dynamic_evidence_id(policy: str, task_id: str, source_value: str) -> str:
    return (
        "mapping:dilemmadata-common:dynamic:"
        + sha256(f"{policy}\0{task_id}\0{source_value}".encode("utf-8")).hexdigest()[:24]
    )


def map_dilemmadata_common_quality(
    source_task_id: str, source_value: object
) -> DilemmadataCommonMappingEvidence:
    """Return the frozen explicit quality mapping row, or an invalid row."""

    dialect: Literal["an_joint", "dlc"] = (
        "an_joint" if source_task_id == _AN_QUALITY_TASK else "dlc"
    )
    if source_task_id not in {_AN_QUALITY_TASK, _DLC_QUALITY_TASK}:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_task_invalid",
            "quality mapping requires an AN or DLC source quality task",
        )
    if not isinstance(source_value, str) or not source_value:
        rendered = repr(source_value)
        return DilemmadataCommonMappingEvidence(
            contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
            evidence_id=_dynamic_evidence_id("quality_invalid_v1", source_task_id, rendered),
            dialect=dialect,
            source_task_id=source_task_id,
            source_value=rendered,
            common_task_id=COMMON_QUALITY_TASK,
            state="invalid",
            common_value=None,
            information_loss=(),
            diagnostic_code="dilemmadata.common.quality_source_value_invalid",
            analysisgnn_reference_value=None,
            analysisgnn_agreement="not_applicable",
            rationale="The value is not a non-empty source-contract string.",
        )
    row = _QUALITY_MAPPING_BY_KEY.get((source_task_id, source_value))
    if row is not None:
        return row
    return DilemmadataCommonMappingEvidence(
        contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
        evidence_id=_dynamic_evidence_id("quality_unsupported_v1", source_task_id, source_value),
        dialect=dialect,
        source_task_id=source_task_id,
        source_value=source_value,
        common_task_id=COMMON_QUALITY_TASK,
        state="unsupported",
        common_value=None,
        information_loss=(),
        diagnostic_code="dilemmadata.common.quality_unsupported",
        analysisgnn_reference_value=None,
        analysisgnn_agreement="not_applicable",
        rationale="The value is outside the frozen source vocabulary and mapping table.",
    )


def map_dilemmadata_common_inversion(
    source_task_id: str, source_value: object
) -> DilemmadataCommonMappingEvidence:
    """Map a source ordinal/figured-bass value before cardinality validation."""

    dialect: Literal["an_joint", "dlc"] = (
        "an_joint" if source_task_id == _AN_INVERSION_TASK else "dlc"
    )
    if source_task_id not in {_AN_INVERSION_TASK, _DLC_INVERSION_TASK}:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_task_invalid",
            "inversion mapping requires an AN or DLC source inversion task",
        )
    if not isinstance(source_value, str) or not source_value:
        rendered = repr(source_value)
        return DilemmadataCommonMappingEvidence(
            contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
            evidence_id=_dynamic_evidence_id("inversion_invalid_v1", source_task_id, rendered),
            dialect=dialect,
            source_task_id=source_task_id,
            source_value=rendered,
            common_task_id=COMMON_INVERSION_TASK,
            state="invalid",
            common_value=None,
            information_loss=(),
            diagnostic_code="dilemmadata.common.inversion_source_value_invalid",
            analysisgnn_reference_value=None,
            analysisgnn_agreement="not_applicable",
            rationale="The value is not a non-empty source-contract string.",
        )
    row = _INVERSION_MAPPING_BY_KEY.get((source_task_id, source_value))
    if row is not None:
        return row
    return DilemmadataCommonMappingEvidence(
        contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
        evidence_id=_dynamic_evidence_id("inversion_unsupported_v1", source_task_id, source_value),
        dialect=dialect,
        source_task_id=source_task_id,
        source_value=source_value,
        common_task_id=COMMON_INVERSION_TASK,
        state="unsupported",
        common_value=None,
        information_loss=(),
        diagnostic_code="dilemmadata.common.inversion_unsupported",
        analysisgnn_reference_value=None,
        analysisgnn_agreement="not_applicable",
        rationale="The value is outside the frozen source inversion vocabulary.",
    )


_NATURAL_PC = MappingProxyType(
    {"A": 9, "B": 11, "C": 0, "D": 2, "E": 4, "F": 5, "G": 7}
)


def _spelling_pitch_class(value: str) -> int | None:
    match = _PITCH_RE.fullmatch(value)
    if match is None:
        return None
    step, accidental = match.groups()
    alteration = accidental.count("#") - accidental.count("-") - accidental.count("b")
    return (_NATURAL_PC[step.upper()] + alteration) % 12


def map_dilemmadata_common_pitch_class(
    source_task_id: str,
    source_value: object,
    *,
    source_spelling: str | None = None,
) -> DilemmadataCommonMappingEvidence:
    """Map AN pitch spelling or DLC line-of-fifths TPC to chromatic PC."""

    task_map = {
        _AN_ROOT_TASK: ("an_joint", COMMON_ROOT_PC_TASK),
        _DLC_ROOT_TASK: ("dlc", COMMON_ROOT_PC_TASK),
        _AN_BASS_TASK: ("an_joint", COMMON_BASS_PC_TASK),
        _DLC_BASS_TASK: ("dlc", COMMON_BASS_PC_TASK),
    }
    if source_task_id not in task_map:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_task_invalid",
            "pitch-class mapping requires an AN/DLC root or bass task",
        )
    dialect, common_task = task_map[source_task_id]
    rendered = source_value if isinstance(source_value, str) else repr(source_value)
    invalid = not isinstance(source_value, str) or not source_value
    pitch_class: int | None = None
    diagnostic: str | None = None
    rationale: str
    if not invalid and dialect == "an_joint":
        pitch_class = _spelling_pitch_class(source_value)
        if pitch_class is None:
            diagnostic = "dilemmadata.common.pitch_spelling_unsupported"
        rationale = "The exact AN pitch spelling is reduced to chromatic pitch class."
    elif not invalid:
        try:
            tpc = int(source_value)
            if str(tpc) != source_value and source_value not in {f"+{tpc}"}:
                raise ValueError
            pitch_class = (7 * tpc) % 12
        except ValueError:
            diagnostic = "dilemmadata.common.tonal_pitch_class_invalid"
        rationale = (
            "The DLC tonal-pitch-class fifth coordinate is converted exactly by "
            "(7 * tpc) mod 12."
        )
    else:
        diagnostic = "dilemmadata.common.pitch_source_value_invalid"
        rationale = "The pitch source value is not a non-empty source-contract string."
    if pitch_class is not None and source_spelling is not None:
        spelling_pc = _spelling_pitch_class(source_spelling)
        if spelling_pc is None or spelling_pc != pitch_class:
            pitch_class = None
            diagnostic = "dilemmadata.common.pitch_tpc_spelling_conflict"
            rationale = "TPC and retained source spelling do not identify the same pitch class."
    state: MappingState
    if pitch_class is not None:
        state = "exact"
    elif invalid or diagnostic in {
        "dilemmadata.common.tonal_pitch_class_invalid",
        "dilemmadata.common.pitch_tpc_spelling_conflict",
    }:
        state = "invalid"
    else:
        state = "unsupported"
    return DilemmadataCommonMappingEvidence(
        contract_version=DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
        evidence_id=_dynamic_evidence_id(
            "pitch_class_v1", source_task_id, f"{rendered}\0{source_spelling or ''}"
        ),
        dialect=dialect,  # type: ignore[arg-type]
        source_task_id=source_task_id,
        source_value=rendered,
        common_task_id=common_task,
        state=state,
        common_value=pitch_class,
        information_loss=("enharmonic_spelling_removed",) if pitch_class is not None else (),
        diagnostic_code=diagnostic,
        analysisgnn_reference_value=None,
        analysisgnn_agreement="not_applicable",
        rationale=rationale,
    )


def _source_target(bundle: TargetBundle, task_id: str) -> SampleTarget:
    try:
        return next(row for row in bundle.targets if row.task_id == task_id)
    except StopIteration as exc:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_family_missing",
            f"source bundle does not contain required family {task_id!r}",
        ) from exc


def _span_keys(bundle: TargetBundle) -> dict[str, tuple[object, ...]]:
    return {
        span.annotation_id: (
            "span",
            span.start_qn.num,
            span.start_qn.den,
            span.end_qn.num,
            span.end_qn.den,
        )
        for span in bundle.alignment_spans
    }


def _alignment_key(
    span_keys: Mapping[str, tuple[object, ...]], entity_id: str
) -> tuple[object, ...]:
    return span_keys.get(entity_id, ("entity", entity_id))


def _source_provenance(target: SampleTarget, index: int) -> tuple[str, ...]:
    value = target.provenance_ids[index]
    return () if value is None else (value,)


def _masked_entry(
    target: SampleTarget,
    index: int,
    *,
    common_fields: tuple[str, ...],
    state: ProjectionState = "masked",
    diagnostic: str | None = None,
) -> DilemmadataCommonTargetEntry:
    return DilemmadataCommonTargetEntry(
        entity_id=target.entity_ids[index],
        source_task_ids=(target.task_id,),
        source_values=(None,),
        state=state,
        common_value=None,
        field_availability=tuple((field, False) for field in common_fields),
        information_loss=(),
        diagnostic_code=diagnostic,
        mapping_evidence_ids=(),
        source_provenance_ids=(),
        dependency_entity_ids=(),
    )


def _mapped_entry(
    target: SampleTarget,
    index: int,
    evidence: DilemmadataCommonMappingEvidence,
    *,
    common_fields: tuple[str, ...],
) -> DilemmadataCommonTargetEntry:
    available = evidence.state in _SUPERVISION_STATES
    return DilemmadataCommonTargetEntry(
        entity_id=target.entity_ids[index],
        source_task_ids=(target.task_id,),
        source_values=(str(target.values[index]),),
        state=evidence.state,
        common_value=evidence.common_value if available else None,
        field_availability=tuple((field, available) for field in common_fields),
        information_loss=evidence.information_loss,
        diagnostic_code=evidence.diagnostic_code,
        mapping_evidence_ids=(evidence.evidence_id,),
        source_provenance_ids=_source_provenance(target, index),
        dependency_entity_ids=(),
    )


SupplementalSourceEvidence: TypeAlias = Mapping[
    tuple[str, str], Mapping[str, str]
]


def _target_from_entries(
    task_id: str,
    alignment_type: str,
    entries: list[DilemmadataCommonTargetEntry],
) -> DilemmadataCommonTarget:
    return DilemmadataCommonTarget(
        task_id=task_id,
        alignment_type=alignment_type,
        value_fields=DILEMMADATA_COMMON_FAMILY_BY_TASK[task_id].value_fields,
        entries=tuple(sorted(entries, key=lambda row: row.entity_id)),
    )


def _quality_target(
    bundle: TargetBundle, source_task_id: str
) -> DilemmadataCommonTarget:
    source = _source_target(bundle, source_task_id)
    fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[COMMON_QUALITY_TASK].value_fields
    entries: list[DilemmadataCommonTargetEntry] = []
    for index, available in enumerate(source.availability_mask):
        if not available:
            entries.append(_masked_entry(source, index, common_fields=fields))
            continue
        evidence = map_dilemmadata_common_quality(source.task_id, source.values[index])
        entries.append(_mapped_entry(source, index, evidence, common_fields=fields))
    return _target_from_entries(COMMON_QUALITY_TASK, source.alignment_type, entries)


def _pitch_target(
    bundle: TargetBundle,
    source_task_id: str,
    common_task_id: str,
    supplemental: SupplementalSourceEvidence,
) -> DilemmadataCommonTarget:
    source = _source_target(bundle, source_task_id)
    fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[common_task_id].value_fields
    entries: list[DilemmadataCommonTargetEntry] = []
    spelling_keys = (
        ("a_root", "root", "source_spelling")
        if common_task_id == COMMON_ROOT_PC_TASK
        else ("a_bass", "bass_note", "source_spelling")
    )
    for index, available in enumerate(source.availability_mask):
        if not available:
            entries.append(_masked_entry(source, index, common_fields=fields))
            continue
        auxiliary = supplemental.get((source.task_id, source.entity_ids[index]), {})
        spelling = next(
            (
                auxiliary[key]
                for key in spelling_keys
                if isinstance(auxiliary.get(key), str) and auxiliary[key]
            ),
            None,
        )
        if source.task_id in {_AN_ROOT_TASK, _AN_BASS_TASK}:
            spelling = str(source.values[index])
        evidence = map_dilemmadata_common_pitch_class(
            source.task_id,
            source.values[index],
            source_spelling=spelling,
        )
        entries.append(
            replace(
                _mapped_entry(source, index, evidence, common_fields=fields),
                supplemental_source_fields=tuple(sorted(auxiliary.items())),
            )
        )
    return _target_from_entries(common_task_id, source.alignment_type, entries)


def _normalized_mode(value: str | None) -> CommonMode | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"major", "maj"}:
        return "major"
    if normalized in {"minor", "min"}:
        return "minor"
    if normalized in {"unknown", "none", "na", "<na>"}:
        return "unknown"
    return "other"


def _local_key_target(
    bundle: TargetBundle,
    source_task_id: str,
    supplemental: SupplementalSourceEvidence,
) -> DilemmadataCommonTarget:
    source = _source_target(bundle, source_task_id)
    fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[COMMON_LOCAL_KEY_TASK].value_fields
    entries: list[DilemmadataCommonTargetEntry] = []
    for index, available in enumerate(source.availability_mask):
        if not available:
            entries.append(_masked_entry(source, index, common_fields=fields))
            continue
        entity_id = source.entity_ids[index]
        raw_value = str(source.values[index])
        auxiliary = supplemental.get((source.task_id, entity_id), {})
        spelling = next(
            (
                auxiliary[key]
                for key in ("a_localKey", "localkey_spelling", "source_spelling")
                if isinstance(auxiliary.get(key), str) and auxiliary[key]
            ),
            None,
        )
        if spelling is None and _spelling_pitch_class(raw_value) is not None:
            spelling = raw_value
        spelling_pc = _spelling_pitch_class(spelling) if spelling is not None else None
        tpc_pc: int | None = None
        raw_tpc = auxiliary.get("localkey_tpc")
        if isinstance(raw_tpc, str) and raw_tpc:
            try:
                tpc_pc = (7 * int(raw_tpc)) % 12
            except ValueError:
                entries.append(
                    DilemmadataCommonTargetEntry(
                        entity_id=entity_id,
                        source_task_ids=(source.task_id,),
                        source_values=(raw_value,),
                        state="invalid",
                        common_value=None,
                        field_availability=tuple((field, False) for field in fields),
                        information_loss=(),
                        diagnostic_code="dilemmadata.common.local_key_tpc_invalid",
                        mapping_evidence_ids=(
                            _dynamic_evidence_id("local_key_invalid_tpc_v1", source.task_id, raw_tpc),
                        ),
                        source_provenance_ids=_source_provenance(source, index),
                        dependency_entity_ids=(),
                    )
                )
                continue
        if spelling_pc is not None and tpc_pc is not None and spelling_pc != tpc_pc:
            entries.append(
                DilemmadataCommonTargetEntry(
                    entity_id=entity_id,
                    source_task_ids=(source.task_id,),
                    source_values=(raw_value,),
                    state="invalid",
                    common_value=None,
                    field_availability=tuple((field, False) for field in fields),
                    information_loss=(),
                    diagnostic_code="dilemmadata.common.local_key_tpc_spelling_conflict",
                    mapping_evidence_ids=(
                        _dynamic_evidence_id("local_key_conflict_v1", source.task_id, raw_value),
                    ),
                    source_provenance_ids=_source_provenance(source, index),
                    dependency_entity_ids=(),
                )
            )
            continue
        tonic_pc = spelling_pc if spelling_pc is not None else tpc_pc
        if tonic_pc is None:
            entries.append(
                DilemmadataCommonTargetEntry(
                    entity_id=entity_id,
                    source_task_ids=(source.task_id,),
                    source_values=(raw_value,),
                    state="ambiguous",
                    common_value=None,
                    field_availability=tuple((field, False) for field in fields),
                    information_loss=(),
                    diagnostic_code="dilemmadata.common.local_key_tonic_ambiguous",
                    mapping_evidence_ids=(
                        _dynamic_evidence_id("local_key_ambiguous_v1", source.task_id, raw_value),
                    ),
                    source_provenance_ids=_source_provenance(source, index),
                    dependency_entity_ids=(),
                )
            )
            continue
        mode = _normalized_mode(auxiliary.get("localkey_mode"))
        raw_is_minor = auxiliary.get("localkey_is_minor")
        if mode is None and isinstance(raw_is_minor, str) and raw_is_minor:
            normalized_minor = raw_is_minor.strip().lower()
            if normalized_minor in {"true", "1"}:
                mode = "minor"
            elif normalized_minor in {"false", "0"}:
                mode = "major"
            else:
                mode = "other"
        if mode is None and spelling is not None:
            mode = "minor" if spelling[0].islower() else "major"
        if mode is None:
            mode = "unknown"
        mode_available = mode != "unknown"
        evidence_id = _dynamic_evidence_id(
            "factorized_local_key_v1",
            source.task_id,
            f"{raw_value}\0{raw_tpc or ''}\0{mode}",
        )
        entries.append(
            DilemmadataCommonTargetEntry(
                entity_id=entity_id,
                source_task_ids=(source.task_id,),
                source_values=(raw_value,),
                state="exact",
                common_value=DilemmadataCommonLocalKeyValue(tonic_pc, mode),
                field_availability=(("mode", mode_available), ("tonic_pc", True)),
                information_loss=("enharmonic_spelling_removed",),
                diagnostic_code=(
                    None
                    if mode_available
                    else "dilemmadata.common.local_key_mode_unknown"
                ),
                mapping_evidence_ids=(evidence_id,),
                source_provenance_ids=_source_provenance(source, index),
                dependency_entity_ids=(),
                supplemental_source_fields=tuple(sorted(auxiliary.items())),
            )
        )
    return _target_from_entries(COMMON_LOCAL_KEY_TASK, source.alignment_type, entries)


def _entries_by_entity(
    target: DilemmadataCommonTarget,
) -> dict[str, DilemmadataCommonTargetEntry]:
    return {entry.entity_id: entry for entry in target.entries}


def _entries_by_alignment(
    target: DilemmadataCommonTarget,
    span_keys: Mapping[str, tuple[object, ...]],
) -> dict[tuple[object, ...], tuple[DilemmadataCommonTargetEntry, ...]]:
    grouped: dict[tuple[object, ...], list[DilemmadataCommonTargetEntry]] = {}
    for entry in target.entries:
        grouped.setdefault(_alignment_key(span_keys, entry.entity_id), []).append(entry)
    return {
        key: tuple(sorted(values, key=lambda row: row.entity_id))
        for key, values in grouped.items()
    }


def _inversion_target(
    bundle: TargetBundle,
    source_task_id: str,
    quality: DilemmadataCommonTarget,
) -> DilemmadataCommonTarget:
    source = _source_target(bundle, source_task_id)
    fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[COMMON_INVERSION_TASK].value_fields
    span_keys = _span_keys(bundle)
    quality_by_alignment = _entries_by_alignment(quality, span_keys)
    entries: list[DilemmadataCommonTargetEntry] = []
    for index, available in enumerate(source.availability_mask):
        if not available:
            entries.append(_masked_entry(source, index, common_fields=fields))
            continue
        evidence = map_dilemmadata_common_inversion(source.task_id, source.values[index])
        if evidence.state not in _SUPERVISION_STATES:
            entries.append(_mapped_entry(source, index, evidence, common_fields=fields))
            continue
        aligned_quality = quality_by_alignment.get(
            _alignment_key(span_keys, source.entity_ids[index]), ()
        )
        cardinality: int | None = None
        dependency_ids: tuple[str, ...] = ()
        if len(aligned_quality) == 1:
            quality_entry = aligned_quality[0]
            dependency_ids = (quality_entry.entity_id,)
            if isinstance(quality_entry.common_value, str):
                template = _QUALITY_TEMPLATE_BY_VALUE.get(quality_entry.common_value)
                cardinality = len(template.intervals) if template is not None else None
        ordinal = {"root": 0, "first": 1, "second": 2, "third": 3}[
            str(evidence.common_value)
        ]
        source_figure_cardinality = None
        if source.task_id == _DLC_INVERSION_TASK:
            source_figure_cardinality = (
                3 if str(source.values[index]) in {"6", "64"} else 4
            )
        inconsistent = (
            cardinality is not None and ordinal >= cardinality
        ) or (
            cardinality is not None
            and source_figure_cardinality is not None
            and cardinality != source_figure_cardinality
        )
        if inconsistent:
            entries.append(
                DilemmadataCommonTargetEntry(
                    entity_id=source.entity_ids[index],
                    source_task_ids=(source.task_id,),
                    source_values=(str(source.values[index]),),
                    state="ambiguous",
                    common_value=None,
                    field_availability=(("inversion", False),),
                    information_loss=(),
                    diagnostic_code="dilemmadata.common.inversion_cardinality_inconsistent",
                    mapping_evidence_ids=(evidence.evidence_id,),
                    source_provenance_ids=_source_provenance(source, index),
                    dependency_entity_ids=dependency_ids,
                )
            )
            continue
        mapped = _mapped_entry(source, index, evidence, common_fields=fields)
        entries.append(replace(mapped, dependency_entity_ids=dependency_ids))
    return _target_from_entries(COMMON_INVERSION_TASK, source.alignment_type, entries)


def _pitch_class_set_target(
    bundle: TargetBundle,
    root_source_task_id: str,
    root: DilemmadataCommonTarget,
    quality: DilemmadataCommonTarget,
) -> DilemmadataCommonTarget:
    source_root = _source_target(bundle, root_source_task_id)
    root_by_entity = _entries_by_entity(root)
    span_keys = _span_keys(bundle)
    quality_by_alignment = _entries_by_alignment(quality, span_keys)
    fields = DILEMMADATA_COMMON_FAMILY_BY_TASK[
        COMMON_PITCH_CLASS_SET_TASK
    ].value_fields
    entries: list[DilemmadataCommonTargetEntry] = []
    for index, entity_id in enumerate(source_root.entity_ids):
        root_entry = root_by_entity[entity_id]
        aligned_quality = quality_by_alignment.get(
            _alignment_key(span_keys, entity_id), ()
        )
        source_map = {root_source_task_id: root_entry.source_values[0]}
        source_provenance = set(root_entry.source_provenance_ids)
        evidence_ids = set(root_entry.mapping_evidence_ids)
        dependency_ids = {entity_id}
        if len(aligned_quality) != 1:
            source_tasks = tuple(sorted(source_map))
            entries.append(
                DilemmadataCommonTargetEntry(
                    entity_id=entity_id,
                    source_task_ids=source_tasks,
                    source_values=tuple(source_map[task] for task in source_tasks),
                    state="ambiguous" if len(aligned_quality) > 1 else "missing",
                    common_value=None,
                    field_availability=(("pitch_classes", False),),
                    information_loss=(),
                    diagnostic_code=(
                        "dilemmadata.common.pitch_class_set_quality_alignment_ambiguous"
                        if len(aligned_quality) > 1
                        else None
                    ),
                    mapping_evidence_ids=tuple(sorted(evidence_ids)),
                    source_provenance_ids=tuple(sorted(source_provenance)),
                    dependency_entity_ids=tuple(sorted(dependency_ids)),
                )
            )
            continue
        quality_entry = aligned_quality[0]
        quality_task_id = quality_entry.source_task_ids[0]
        source_map[quality_task_id] = quality_entry.source_values[0]
        source_provenance.update(quality_entry.source_provenance_ids)
        evidence_ids.update(quality_entry.mapping_evidence_ids)
        dependency_ids.add(quality_entry.entity_id)
        source_tasks = tuple(sorted(source_map))
        common_state: ProjectionState
        common_value: tuple[int, ...] | None = None
        diagnostic: str | None = None
        losses = set(root_entry.information_loss) | set(quality_entry.information_loss)
        if root_entry.state not in _SUPERVISION_STATES:
            common_state = root_entry.state
            diagnostic = root_entry.diagnostic_code
        elif quality_entry.state not in _SUPERVISION_STATES:
            common_state = quality_entry.state
            diagnostic = quality_entry.diagnostic_code
        elif not isinstance(root_entry.common_value, int) or not isinstance(
            quality_entry.common_value, str
        ):
            common_state = "invalid"
            diagnostic = "dilemmadata.common.pitch_class_set_dependency_invalid"
        else:
            template = _QUALITY_TEMPLATE_BY_VALUE.get(quality_entry.common_value)
            if template is None:
                common_state = "unsupported"
                diagnostic = "dilemmadata.common.pitch_class_set_template_unavailable"
            else:
                common_value = tuple(
                    sorted(
                        {
                            (root_entry.common_value + interval) % 12
                            for interval in template.intervals
                        }
                    )
                )
                evidence_ids.add(
                    f"template:dilemmadata-common:{template.template_version}:{template.quality}"
                )
                common_state = (
                    "coarsened"
                    if "coarsened" in {root_entry.state, quality_entry.state}
                    else "exact"
                )
        entries.append(
            DilemmadataCommonTargetEntry(
                entity_id=entity_id,
                source_task_ids=source_tasks,
                source_values=tuple(source_map[task] for task in source_tasks),
                state=common_state,
                common_value=common_value,
                field_availability=(("pitch_classes", common_value is not None),),
                information_loss=tuple(sorted(losses)),
                diagnostic_code=diagnostic,
                mapping_evidence_ids=tuple(sorted(evidence_ids)),
                source_provenance_ids=tuple(sorted(source_provenance)),
                dependency_entity_ids=tuple(sorted(dependency_ids)),
            )
        )
    return _target_from_entries(
        COMMON_PITCH_CLASS_SET_TASK,
        source_root.alignment_type,
        entries,
    )


def project_dilemmadata_common_harmony(
    source_bundle: TargetBundle,
    *,
    supplemental_source_evidence: SupplementalSourceEvidence | None = None,
) -> DilemmadataCommonHarmonicProjection:
    """Create a target-only common projection from an accepted source bundle.

    ``supplemental_source_evidence`` is optional, target-only evidence keyed by
    ``(source_task_id, entity_id)``.  It can retain DLC TPC spelling and mode
    fields that the immutable source-native bundle did not expose as its
    primary value.  It is never consulted for raw input, graph, grouping, or
    split construction.
    """

    if not isinstance(source_bundle, TargetBundle):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_bundle_invalid",
            "source_bundle must be a validated TargetBundle",
        )
    if source_bundle.contract_version != TARGET_BUNDLE_CONTRACT_VERSION:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_bundle_version_invalid",
            "common projection requires TargetBundle@1.0.0",
        )
    supplemental: SupplementalSourceEvidence = supplemental_source_evidence or {}
    if not isinstance(supplemental, Mapping):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.supplemental_evidence_invalid",
            "supplemental source evidence must be a mapping",
        )
    task_ids = {target.task_id for target in source_bundle.targets}
    has_an = any(task.startswith("dilemmadata.an.") for task in task_ids)
    has_dlc = any(task.startswith("dilemmadata.dlc.") for task in task_ids)
    if has_an == has_dlc:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.dialect_invalid",
            "source bundle must contain exactly one AN or DLC annotation view",
        )
    if has_an:
        quality_task = _AN_QUALITY_TASK
        inversion_task = _AN_INVERSION_TASK
        root_task = _AN_ROOT_TASK
        bass_task = _AN_BASS_TASK
        local_key_task = _AN_LOCAL_KEY_TASK
    else:
        quality_task = _DLC_QUALITY_TASK
        inversion_task = _DLC_INVERSION_TASK
        root_task = _DLC_ROOT_TASK
        bass_task = _DLC_BASS_TASK
        local_key_task = _DLC_LOCAL_KEY_TASK

    quality = _quality_target(source_bundle, quality_task)
    root = _pitch_target(
        source_bundle,
        root_task,
        COMMON_ROOT_PC_TASK,
        supplemental,
    )
    bass = _pitch_target(
        source_bundle,
        bass_task,
        COMMON_BASS_PC_TASK,
        supplemental,
    )
    local_key = _local_key_target(source_bundle, local_key_task, supplemental)
    inversion = _inversion_target(source_bundle, inversion_task, quality)
    pitch_class_set = _pitch_class_set_target(
        source_bundle,
        root_task,
        root,
        quality,
    )
    targets = tuple(
        sorted(
            (quality, inversion, root, bass, local_key, pitch_class_set),
            key=lambda target: target.task_id,
        )
    )
    values = {
        "contract_version": DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION,
        "dataset_id": source_bundle.dataset_id,
        "piece_id": source_bundle.piece_id,
        "analysis_view_id": source_bundle.analysis_view_id,
        "source_target_bundle_contract_version": source_bundle.contract_version,
        "source_target_bundle_fingerprint": target_bundle_fingerprint(source_bundle),
        "common_registry_fingerprint": DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint,
        "targets": targets,
    }
    unsigned = {
        "analysis_view_id": values["analysis_view_id"],
        "common_registry_fingerprint": values["common_registry_fingerprint"],
        "contract_version": values["contract_version"],
        "dataset_id": values["dataset_id"],
        "piece_id": values["piece_id"],
        "projection_fingerprint": "",
        "source_target_bundle_contract_version": values[
            "source_target_bundle_contract_version"
        ],
        "source_target_bundle_fingerprint": values[
            "source_target_bundle_fingerprint"
        ],
        "targets": [
            {
                "alignment_type": target.alignment_type,
                "entries": [_entry_payload(entry) for entry in target.entries],
                "task_id": target.task_id,
                "value_fields": list(target.value_fields),
            }
            for target in targets
        ],
    }
    return DilemmadataCommonHarmonicProjection(
        **values,
        projection_fingerprint=_fingerprint(unsigned),
    )


def common_projection_fingerprint(
    projection: DilemmadataCommonHarmonicProjection,
) -> str:
    if not isinstance(projection, DilemmadataCommonHarmonicProjection):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.projection_invalid",
            "projection must be DilemmadataCommonHarmonicProjection",
        )
    return _fingerprint(_projection_payload(projection, clear=True))


def dumps_dilemmadata_common_projection(
    projection: DilemmadataCommonHarmonicProjection,
    *,
    indent: int | None = None,
) -> str:
    payload = _projection_payload(projection)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    ) + ("\n" if indent is not None else "")


def loads_dilemmadata_common_projection(
    payload: str,
) -> DilemmadataCommonHarmonicProjection:
    """Strictly deserialize and revalidate one common projection."""

    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "analysis_view_id",
            "common_registry_fingerprint",
            "contract_version",
            "dataset_id",
            "piece_id",
            "projection_fingerprint",
            "source_target_bundle_contract_version",
            "source_target_bundle_fingerprint",
            "targets",
        }:
            raise ValueError("projection object has missing or unknown fields")
        targets: list[DilemmadataCommonTarget] = []
        for target_value in value["targets"]:
            if not isinstance(target_value, dict) or set(target_value) != {
                "alignment_type",
                "entries",
                "task_id",
                "value_fields",
            }:
                raise ValueError("common target has missing or unknown fields")
            entries: list[DilemmadataCommonTargetEntry] = []
            for entry_value in target_value["entries"]:
                if not isinstance(entry_value, dict) or set(entry_value) != {
                    "common_value",
                    "dependency_entity_ids",
                    "diagnostic_code",
                    "entity_id",
                    "field_availability",
                    "information_loss",
                    "mapping_evidence_ids",
                    "source_provenance_ids",
                    "source_task_ids",
                    "source_values",
                    "state",
                    "supplemental_source_fields",
                }:
                    raise ValueError("common entry has missing or unknown fields")
                common_value = entry_value["common_value"]
                if isinstance(common_value, dict):
                    if set(common_value) != {"mode", "tonic_pc"}:
                        raise ValueError("factorized local-key value has invalid fields")
                    common_value = DilemmadataCommonLocalKeyValue(**common_value)
                elif isinstance(common_value, list):
                    common_value = tuple(common_value)
                entries.append(
                    DilemmadataCommonTargetEntry(
                        entity_id=entry_value["entity_id"],
                        source_task_ids=tuple(entry_value["source_task_ids"]),
                        source_values=tuple(entry_value["source_values"]),
                        state=entry_value["state"],
                        common_value=common_value,
                        field_availability=tuple(
                            (row[0], row[1])
                            for row in entry_value["field_availability"]
                        ),
                        information_loss=tuple(entry_value["information_loss"]),
                        diagnostic_code=entry_value["diagnostic_code"],
                        mapping_evidence_ids=tuple(entry_value["mapping_evidence_ids"]),
                        source_provenance_ids=tuple(entry_value["source_provenance_ids"]),
                        dependency_entity_ids=tuple(entry_value["dependency_entity_ids"]),
                        supplemental_source_fields=tuple(
                            (row[0], row[1])
                            for row in entry_value["supplemental_source_fields"]
                        ),
                    )
                )
            targets.append(
                DilemmadataCommonTarget(
                    task_id=target_value["task_id"],
                    alignment_type=target_value["alignment_type"],
                    value_fields=tuple(target_value["value_fields"]),
                    entries=tuple(entries),
                )
            )
        return DilemmadataCommonHarmonicProjection(
            contract_version=value["contract_version"],
            dataset_id=value["dataset_id"],
            piece_id=value["piece_id"],
            analysis_view_id=value["analysis_view_id"],
            source_target_bundle_contract_version=value[
                "source_target_bundle_contract_version"
            ],
            source_target_bundle_fingerprint=value[
                "source_target_bundle_fingerprint"
            ],
            common_registry_fingerprint=value["common_registry_fingerprint"],
            targets=tuple(targets),
            projection_fingerprint=value["projection_fingerprint"],
        )
    except DilemmadataCommonProjectionError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.projection_parse_invalid",
            f"cannot parse common projection: {exc}",
        ) from exc


def _bound_supplemental_source_evidence(
    raw_accepted: object,
    target_accepted: object,
) -> dict[tuple[str, str], dict[str, str]]:
    """Reconstruct target-only auxiliary fields under the accepted row binding."""

    # These helpers are private to the producer module but are invoked only at
    # this explicit version-bound adapter boundary.  The resulting evidence is
    # independently checked against the immutable emitted target entry count.
    from music_critic.adapters import dilemmadata_targets as target_adapter

    parsed = target_adapter._read_target_rows(  # type: ignore[attr-defined]
        raw_accepted.record,
        raw_accepted.alignment_evidence,
    )
    if not isinstance(parsed, tuple) or len(parsed) != 4:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.supplemental_source_unavailable",
            "bound target rows could not be reconstructed",
        )
    rows = parsed[0]
    bundle = target_accepted.target_bundle
    evidence: dict[tuple[str, str], dict[str, str]] = {}
    for task_id in (
        _AN_ROOT_TASK,
        _AN_BASS_TASK,
        _AN_LOCAL_KEY_TASK,
        _DLC_ROOT_TASK,
        _DLC_BASS_TASK,
        _DLC_LOCAL_KEY_TASK,
    ):
        if task_id not in {target.task_id for target in bundle.targets}:
            continue
        spec = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id]
        family_rows = tuple(target_adapter._family_row(spec, row) for row in rows)  # type: ignore[attr-defined]
        if spec.coordinate not in {"global_span", "annotation_run"}:
            continue
        emitted, _equal_merges, _conflicts = target_adapter._emit_span_entries(  # type: ignore[attr-defined]
            spec,
            family_rows,
            raw_accepted.piece,
        )
        source_target = _source_target(bundle, task_id)
        if len(emitted) != len(source_target.entity_ids):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.supplemental_alignment_mismatch",
                f"supplemental rows differ from source family {task_id!r}",
            )
        for emitted_entry, entity_id in zip(
            emitted, source_target.entity_ids, strict=True
        ):
            fields: dict[str, str] = {}
            for field in spec.source_fields:
                values = tuple(
                    sorted(
                        {
                            rows[row_index].values.get(field, "").strip()
                            for row_index in emitted_entry.source_rows
                            if rows[row_index].values.get(field, "").strip()
                            not in target_adapter._MISSING  # type: ignore[attr-defined]
                        }
                    )
                )
                if len(values) == 1:
                    fields[field] = values[0]
                elif len(values) > 1:
                    fields[f"conflict.{field}"] = json.dumps(
                        list(values),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
            evidence[(task_id, entity_id)] = fields
    return evidence


def build_dilemmadata_common_harmonic_projection(
    raw_accepted: object,
    target_accepted: object,
) -> DilemmadataCommonHarmonicProjection:
    """Build a projection only after the existing raw/target binding passes.

    This is the production-safe public builder.  The pure
    :func:`project_dilemmadata_common_harmony` function remains useful for
    source-free replay and synthetic contract tests.
    """

    from music_critic.adapters.dilemmadata import (
        DilemmadataAccepted,
        validate_dilemmadata_alignment_evidence,
        validate_dilemmadata_record_binding,
    )
    from music_critic.adapters.dilemmadata_targets import (
        DilemmadataTargetAccepted,
        build_dilemmadata_target_sidecar,
    )

    if not isinstance(raw_accepted, DilemmadataAccepted) or not isinstance(
        target_accepted, DilemmadataTargetAccepted
    ):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.accepted_inputs_invalid",
            "bound projection requires accepted raw and target outcomes",
        )
    if (
        raw_accepted.record != target_accepted.record
        or raw_accepted.piece.piece_id != target_accepted.piece_id
        or raw_accepted.piece.dataset_name
        != target_accepted.target_bundle.dataset_id
        or raw_accepted.piece.targets
        or raw_accepted.piece.annotations
    ):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.raw_target_binding_mismatch",
            "raw and target accepted outcomes do not identify the same raw-only piece",
        )
    if not validate_dilemmadata_record_binding(raw_accepted.record) or not (
        validate_dilemmadata_alignment_evidence(
            raw_accepted.record,
            raw_accepted.piece,
            raw_accepted.alignment_evidence,
        )
    ):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.raw_target_binding_mismatch",
            "raw record/alignment binding no longer validates",
        )
    source_fingerprint = target_bundle_fingerprint(target_accepted.target_bundle)
    if source_fingerprint != target_accepted.sidecar_fingerprint:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_bundle_fingerprint_mismatch",
            "accepted target fingerprint differs from its bundle",
        )
    rebuilt = build_dilemmadata_target_sidecar(raw_accepted)
    if not isinstance(rebuilt, DilemmadataTargetAccepted) or (
        rebuilt.sidecar_fingerprint != target_accepted.sidecar_fingerprint
    ):
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_replay_mismatch",
            "current bound source does not reproduce the accepted target bundle",
        )
    supplemental = _bound_supplemental_source_evidence(
        raw_accepted,
        target_accepted,
    )
    projection = project_dilemmadata_common_harmony(
        target_accepted.target_bundle,
        supplemental_source_evidence=supplemental,
    )
    if target_bundle_fingerprint(target_accepted.target_bundle) != source_fingerprint:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.source_bundle_mutated",
            "common projection mutated the source-native TargetBundle",
        )
    return projection


AuditScalar: TypeAlias = str | int | bool


@dataclass(frozen=True, slots=True)
class DilemmadataCommonAuditFact:
    name: str
    dimensions: tuple[tuple[str, str], ...]
    value: AuditScalar

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "audit fact name")
        if self.dimensions != tuple(sorted(self.dimensions)) or len(
            tuple(key for key, _value in self.dimensions)
        ) != len({key for key, _value in self.dimensions}):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_fact_invalid",
                "audit fact dimensions must be uniquely sorted",
            )
        if isinstance(self.value, bool):
            return
        if isinstance(self.value, int):
            if self.value < 0:
                raise DilemmadataCommonProjectionError(
                    "dilemmadata.common.audit_fact_invalid",
                    "integer audit facts must be non-negative",
                )
            return
        _validate_identifier(self.value, "audit fact value")


@dataclass(frozen=True, slots=True)
class DilemmadataCommonCollapseEvidence:
    common_value: str
    source_rows: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.common_value, "collapse common value")
        if len(self.source_rows) < 2 or self.source_rows != tuple(
            sorted(set(self.source_rows))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.collapse_invalid",
                "collapse evidence requires at least two unique sorted source rows",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonOverlapEvidence:
    component_id: str
    record_ids: tuple[str, ...]
    family: str
    comparison: Literal[
        "exact_agreement",
        "enharmonic_only_agreement",
        "coarsened_agreement",
        "conflict",
        "unavailable",
    ]
    left_value: str | None
    right_value: str | None

    def __post_init__(self) -> None:
        _validate_identifier(self.component_id, "overlap component")
        _validate_identifier(self.family, "overlap family")
        if self.comparison not in _OVERLAP_COMPARISONS:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.overlap_invalid",
                "overlap comparison is outside the frozen vocabulary",
            )
        if len(self.record_ids) < 2 or self.record_ids != tuple(
            sorted(set(self.record_ids))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.overlap_invalid",
                "overlap evidence requires at least two sorted record IDs",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonInvariantEvidence:
    name: str
    before_fingerprint: str
    after_fingerprint: str
    unchanged: bool

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "invariant name")
        if not _is_sha256(self.before_fingerprint) or not _is_sha256(
            self.after_fingerprint
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.invariant_invalid",
                "invariant evidence requires SHA-256 fingerprints",
            )
        if self.unchanged != (self.before_fingerprint == self.after_fingerprint):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.invariant_invalid",
                "invariant unchanged flag differs from fingerprint equality",
            )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonHarmonicAuditReport:
    contract_version: str
    base_git_sha: str
    source_record_count: int
    source_component_count: int
    annotation_view_count: int
    source_entry_count: int
    source_span_count: int
    projection_count: int
    facts: tuple[DilemmadataCommonAuditFact, ...]
    collapse_table: tuple[DilemmadataCommonCollapseEvidence, ...]
    analysisgnn_parity: tuple[tuple[str, str, str, str, str], ...]
    overlap_evidence: tuple[DilemmadataCommonOverlapEvidence, ...]
    invariance_evidence: tuple[DilemmadataCommonInvariantEvidence, ...]
    fingerprints: tuple[tuple[str, str], ...]
    test_target_access_policy: str
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_report_version_invalid",
                "common audit report version is incompatible",
            )
        if not isinstance(self.base_git_sha, str) or _GIT_SHA_RE.fullmatch(
            self.base_git_sha
        ) is None:
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_base_invalid",
                "audit base Git SHA must be a lowercase 40- or 64-hex object ID",
            )
        counts = (
            self.source_record_count,
            self.source_component_count,
            self.annotation_view_count,
            self.source_entry_count,
            self.source_span_count,
            self.projection_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_counts_invalid",
                "audit counts must be non-negative integers",
            )
        if self.facts != tuple(sorted(self.facts, key=lambda row: (row.name, row.dimensions))):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_order_invalid",
                "audit facts must be deterministically sorted",
            )
        if self.collapse_table != tuple(
            sorted(self.collapse_table, key=lambda row: row.common_value)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_order_invalid",
                "collapse table must be deterministically sorted",
            )
        if self.analysisgnn_parity != tuple(sorted(self.analysisgnn_parity)):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_order_invalid",
                "AnalysisGNN parity rows must be deterministically sorted",
            )
        if self.overlap_evidence != tuple(
            sorted(
                self.overlap_evidence,
                key=lambda row: (row.component_id, row.family, row.record_ids),
            )
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_order_invalid",
                "overlap evidence must be deterministically sorted",
            )
        if self.invariance_evidence != tuple(
            sorted(self.invariance_evidence, key=lambda row: row.name)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_order_invalid",
                "invariance evidence must be deterministically sorted",
            )
        if self.fingerprints != tuple(sorted(self.fingerprints)) or any(
            not name or not _is_sha256(value) for name, value in self.fingerprints
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_fingerprints_invalid",
                "audit fingerprints must be named, sorted SHA-256 rows",
            )
        _validate_identifier(self.test_target_access_policy, "test target access policy")
        if self.semantic_fingerprint != _fingerprint(
            _audit_report_payload(self, clear=True)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_fingerprint_invalid",
                "audit semantic fingerprint differs from report contents",
            )


def _audit_report_payload(
    report: DilemmadataCommonHarmonicAuditReport, *, clear: bool = False
) -> dict[str, object]:
    value = asdict(report)
    value["semantic_fingerprint"] = "" if clear else report.semantic_fingerprint
    return value


def make_dilemmadata_common_audit_report(
    *,
    base_git_sha: str,
    source_record_count: int,
    source_component_count: int,
    annotation_view_count: int,
    source_entry_count: int,
    source_span_count: int,
    projection_count: int,
    facts: tuple[DilemmadataCommonAuditFact, ...],
    collapse_table: tuple[DilemmadataCommonCollapseEvidence, ...],
    analysisgnn_parity: tuple[tuple[str, str, str, str, str], ...],
    overlap_evidence: tuple[DilemmadataCommonOverlapEvidence, ...],
    invariance_evidence: tuple[DilemmadataCommonInvariantEvidence, ...],
    fingerprints: tuple[tuple[str, str], ...],
    test_target_access_policy: str,
) -> DilemmadataCommonHarmonicAuditReport:
    values = {
        "contract_version": DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION,
        "base_git_sha": base_git_sha,
        "source_record_count": source_record_count,
        "source_component_count": source_component_count,
        "annotation_view_count": annotation_view_count,
        "source_entry_count": source_entry_count,
        "source_span_count": source_span_count,
        "projection_count": projection_count,
        "facts": tuple(sorted(facts, key=lambda row: (row.name, row.dimensions))),
        "collapse_table": tuple(sorted(collapse_table, key=lambda row: row.common_value)),
        "analysisgnn_parity": tuple(sorted(analysisgnn_parity)),
        "overlap_evidence": tuple(
            sorted(
                overlap_evidence,
                key=lambda row: (row.component_id, row.family, row.record_ids),
            )
        ),
        "invariance_evidence": tuple(
            sorted(invariance_evidence, key=lambda row: row.name)
        ),
        "fingerprints": tuple(sorted(fingerprints)),
        "test_target_access_policy": test_target_access_policy,
    }
    return DilemmadataCommonHarmonicAuditReport(
        **values,
        semantic_fingerprint=_fingerprint(
            {
                **{
                    key: (
                        [asdict(row) for row in value]
                        if key
                        in {
                            "facts",
                            "collapse_table",
                            "overlap_evidence",
                            "invariance_evidence",
                        }
                        else value
                    )
                    for key, value in values.items()
                },
                "semantic_fingerprint": "",
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class DilemmadataCommonHarmonicAuditManifest:
    contract_version: str
    audit_report_version: str
    report_semantic_fingerprint: str
    registry_fingerprint: str
    analysisgnn_reference_fingerprint: str
    summary_facts: tuple[DilemmadataCommonAuditFact, ...]
    ready: bool
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION
            or self.audit_report_version
            != DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_manifest_version_invalid",
                "common audit manifest versions are incompatible",
            )
        if not all(
            _is_sha256(value)
            for value in (
                self.report_semantic_fingerprint,
                self.registry_fingerprint,
                self.analysisgnn_reference_fingerprint,
            )
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_manifest_binding_invalid",
                "manifest bindings must be SHA-256",
            )
        if self.summary_facts != tuple(
            sorted(self.summary_facts, key=lambda row: (row.name, row.dimensions))
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_manifest_order_invalid",
                "manifest summary facts must be deterministically sorted",
            )
        if self.manifest_fingerprint != _fingerprint(
            _audit_manifest_payload(self, clear=True)
        ):
            raise DilemmadataCommonProjectionError(
                "dilemmadata.common.audit_manifest_fingerprint_invalid",
                "manifest fingerprint differs from its contents",
            )


def _audit_manifest_payload(
    manifest: DilemmadataCommonHarmonicAuditManifest, *, clear: bool = False
) -> dict[str, object]:
    value = asdict(manifest)
    value["manifest_fingerprint"] = "" if clear else manifest.manifest_fingerprint
    return value


def make_dilemmadata_common_audit_manifest(
    report: DilemmadataCommonHarmonicAuditReport,
    *,
    summary_facts: tuple[DilemmadataCommonAuditFact, ...],
    ready: bool,
) -> DilemmadataCommonHarmonicAuditManifest:
    values = {
        "contract_version": DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION,
        "audit_report_version": DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION,
        "report_semantic_fingerprint": report.semantic_fingerprint,
        "registry_fingerprint": DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint,
        "analysisgnn_reference_fingerprint": ANALYSISGNN_REFERENCE.fingerprint,
        "summary_facts": tuple(
            sorted(summary_facts, key=lambda row: (row.name, row.dimensions))
        ),
        "ready": ready,
    }
    serializable = {
        **values,
        "summary_facts": [asdict(row) for row in values["summary_facts"]],
        "manifest_fingerprint": "",
    }
    return DilemmadataCommonHarmonicAuditManifest(
        **values,
        manifest_fingerprint=_fingerprint(serializable),
    )


def dumps_dilemmadata_common_audit_report(
    report: DilemmadataCommonHarmonicAuditReport, *, indent: int | None = 2
) -> str:
    return json.dumps(
        _audit_report_payload(report),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    ) + ("\n" if indent is not None else "")


def dumps_dilemmadata_common_audit_manifest(
    manifest: DilemmadataCommonHarmonicAuditManifest, *, indent: int | None = 2
) -> str:
    return json.dumps(
        _audit_manifest_payload(manifest),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    ) + ("\n" if indent is not None else "")


def loads_dilemmadata_common_audit_manifest(
    payload: str,
) -> DilemmadataCommonHarmonicAuditManifest:
    try:
        value = json.loads(payload)
        expected = {
            "analysisgnn_reference_fingerprint",
            "audit_report_version",
            "contract_version",
            "manifest_fingerprint",
            "ready",
            "registry_fingerprint",
            "report_semantic_fingerprint",
            "summary_facts",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("audit manifest has missing or unknown fields")
        facts = tuple(
            DilemmadataCommonAuditFact(
                name=row["name"],
                dimensions=tuple((item[0], item[1]) for item in row["dimensions"]),
                value=row["value"],
            )
            for row in value["summary_facts"]
        )
        return DilemmadataCommonHarmonicAuditManifest(
            contract_version=value["contract_version"],
            audit_report_version=value["audit_report_version"],
            report_semantic_fingerprint=value["report_semantic_fingerprint"],
            registry_fingerprint=value["registry_fingerprint"],
            analysisgnn_reference_fingerprint=value[
                "analysisgnn_reference_fingerprint"
            ],
            summary_facts=facts,
            ready=value["ready"],
            manifest_fingerprint=value["manifest_fingerprint"],
        )
    except DilemmadataCommonProjectionError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DilemmadataCommonProjectionError(
            "dilemmadata.common.audit_manifest_parse_invalid",
            f"cannot parse common audit manifest: {exc}",
        ) from exc


__all__ = [
    "ANALYSISGNN_REFERENCE",
    "ANALYSISGNN_REFERENCE_COMMIT",
    "ANALYSISGNN_REFERENCE_MAPPING_VERSION",
    "ANALYSISGNN_REPOSITORY",
    "AnalysisGNNReferenceMapping",
    "COMMON_BASS_PC_TASK",
    "COMMON_INVERSION_TASK",
    "COMMON_LOCAL_KEY_TASK",
    "COMMON_PITCH_CLASS_SET_TASK",
    "COMMON_QUALITY_TASK",
    "COMMON_ROOT_PC_TASK",
    "DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION",
    "DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION",
    "DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION",
    "DILEMMADATA_COMMON_HARMONIC_REGISTRY",
    "DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION",
    "DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION",
    "DilemmadataCommonAuditFact",
    "DilemmadataCommonCollapseEvidence",
    "DilemmadataCommonFamilySpec",
    "DilemmadataCommonHarmonicAuditManifest",
    "DilemmadataCommonHarmonicAuditReport",
    "DilemmadataCommonHarmonicProjection",
    "DilemmadataCommonHarmonicRegistry",
    "DilemmadataCommonInvariantEvidence",
    "DilemmadataCommonLocalKeyValue",
    "DilemmadataCommonMappingEvidence",
    "DilemmadataCommonOverlapEvidence",
    "DilemmadataCommonProjectionError",
    "DilemmadataCommonQualityTemplate",
    "DilemmadataCommonTarget",
    "DilemmadataCommonTargetEntry",
    "build_dilemmadata_common_harmonic_projection",
    "common_projection_fingerprint",
    "dilemmadata_common_registry_dict",
    "dilemmadata_common_registry_fingerprint",
    "dumps_dilemmadata_common_audit_manifest",
    "dumps_dilemmadata_common_audit_report",
    "dumps_dilemmadata_common_projection",
    "dumps_dilemmadata_common_registry",
    "loads_dilemmadata_common_audit_manifest",
    "loads_dilemmadata_common_projection",
    "make_dilemmadata_common_audit_manifest",
    "make_dilemmadata_common_audit_report",
    "map_dilemmadata_common_inversion",
    "map_dilemmadata_common_pitch_class",
    "map_dilemmadata_common_quality",
    "project_dilemmadata_common_harmony",
]
