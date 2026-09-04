"""Pre-open TEST-target guard for supervision EDA.

The split gate deliberately accepts only target-free assignment metadata.  It
validates the complete TRAIN/VALIDATION plan before invoking either a target
descriptor resolver or a target loader, and it never offers an unlock path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata
from typing import Generic, TypeVar

from music_critic.eda.contracts import (
    CorpusId,
    EDAContractError,
    EDA_TEST_TARGET_LOCK_VERSION,
    EvidenceScope,
    SplitScope,
    TestTargetLockEvidence,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_manifest_fingerprint",
        "corpus",
        "record_id",
        "split",
        "target_free",
    }
)

DescriptorT = TypeVar("DescriptorT")
LoadedT = TypeVar("LoadedT")


def _require_utf8_scalar(value: str, *, label: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EDAContractError(
            "eda.serialization.utf8_invalid",
            f"{label} must contain valid UTF-8 scalar text",
            path=path,
        ) from exc
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise EDAContractError(
            "eda.identity.control_character",
            f"{label} must not contain control or format characters",
            path=path,
        )


@dataclass(frozen=True, slots=True)
class TargetFreeSplitAssignment:
    corpus: CorpusId
    record_id: str
    split: SplitScope
    assignment_manifest_fingerprint: str
    target_free: bool = True

    def __post_init__(self) -> None:
        try:
            corpus = CorpusId(self.corpus)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.test_lock.assignment_corpus_invalid",
                f"unknown corpus {self.corpus!r}",
            ) from exc
        object.__setattr__(self, "corpus", corpus)
        if not isinstance(self.record_id, str) or not self.record_id.strip() or (
            self.record_id != self.record_id.strip()
        ):
            raise EDAContractError(
                "eda.test_lock.assignment_record_invalid",
                "TRAIN/VALIDATION assignment requires a non-empty record_id",
            )
        _require_utf8_scalar(
            self.record_id,
            label="assignment record_id",
            path="$.assignment.record_id",
        )
        try:
            split = SplitScope(self.split)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.test_lock.assignment_split_invalid",
                f"unknown split {self.split!r}",
            ) from exc
        if split not in {SplitScope.TRAIN, SplitScope.VALIDATION}:
            raise EDAContractError(
                "eda.test_lock.assignment_split_invalid",
                "only TRAIN/VALIDATION assignments reach target resolution",
            )
        object.__setattr__(self, "split", split)
        if (
            not isinstance(self.assignment_manifest_fingerprint, str)
            or _SHA256_RE.fullmatch(self.assignment_manifest_fingerprint) is None
        ):
            raise EDAContractError(
                "eda.test_lock.assignment_fingerprint_invalid",
                "assignment manifest fingerprint must be lowercase SHA-256",
            )
        if self.target_free is not True:
            raise EDAContractError(
                "eda.test_lock.assignment_not_target_free",
                "the pre-open gate accepts only target-free assignment metadata",
            )


@dataclass(frozen=True, slots=True)
class SupervisionTargetAccessGuard(Generic[DescriptorT, LoadedT]):
    """Resolve/load TRAIN and VALIDATION supervision after a hard TEST gate."""

    corpus: CorpusId
    evidence_scope: EvidenceScope = EvidenceScope.FIXTURE
    provenance: tuple[str, ...] = ("supervision-target-access-guard",)
    contract_version: str = EDA_TEST_TARGET_LOCK_VERSION

    def __post_init__(self) -> None:
        try:
            corpus = CorpusId(self.corpus)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.test_lock.guard_corpus_invalid", f"unknown corpus {self.corpus!r}"
            ) from exc
        if corpus == CorpusId.PDMX:
            raise EDAContractError(
                "eda.capability.supervision_forbidden",
                "PDMX has no supervision EDA loader path",
            )
        object.__setattr__(self, "corpus", corpus)
        try:
            evidence_scope = EvidenceScope(self.evidence_scope)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.test_lock.evidence_scope_invalid",
                f"unknown evidence scope {self.evidence_scope!r}",
            ) from exc
        if evidence_scope in {EvidenceScope.UNKNOWN, EvidenceScope.UNAVAILABLE}:
            raise EDAContractError(
                "eda.test_lock.evidence_scope_invalid",
                "observed TEST-lock counters require an observed evidence scope",
            )
        object.__setattr__(self, "evidence_scope", evidence_scope)
        if (
            not isinstance(self.provenance, (tuple, list))
            or not self.provenance
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in self.provenance
            )
            or len(self.provenance) != len(set(self.provenance))
        ):
            raise EDAContractError(
                "eda.test_lock.provenance_invalid",
                "TEST-lock provenance must be a non-empty unique string sequence",
            )
        for index, value in enumerate(self.provenance):
            _require_utf8_scalar(
                value,
                label="TEST-lock provenance",
                path=f"$.guard.provenance[{index}]",
            )
        object.__setattr__(self, "provenance", tuple(sorted(self.provenance)))
        if self.contract_version != EDA_TEST_TARGET_LOCK_VERSION:
            raise EDAContractError(
                "eda.test_lock.version_invalid", "unsupported TEST-target guard version"
            )

    def load_train_validation(
        self,
        assignments: Sequence[Mapping[str, object]],
        *,
        resolve_descriptor: Callable[[str, SplitScope], DescriptorT],
        load_target: Callable[[DescriptorT, SplitScope], LoadedT],
    ) -> tuple[tuple[LoadedT, ...], TestTargetLockEvidence]:
        """Load allowed targets only after all assignment rows pass preflight.

        TEST rows are counted from their split token and then discarded before
        any other key—including a record ID or a path-like descriptor—is read.
        Unknown/malformed rows fail during the first pass, before any callback.
        """

        allowed: list[TargetFreeSplitAssignment] = []
        test_assignment_count = 0
        for index, row in enumerate(assignments):
            if not isinstance(row, Mapping):
                raise EDAContractError(
                    "eda.test_lock.assignment_type_invalid",
                    "assignment must be a mapping",
                    path=f"$.assignments[{index}]",
                )
            split_value = row.get("split")
            if split_value == SplitScope.TEST or split_value == SplitScope.TEST.value:
                test_assignment_count += 1
                continue
            if split_value not in {
                SplitScope.TRAIN,
                SplitScope.TRAIN.value,
                SplitScope.VALIDATION,
                SplitScope.VALIDATION.value,
            }:
                raise EDAContractError(
                    "eda.test_lock.assignment_split_invalid",
                    f"unknown or non-EDA split {split_value!r}",
                    path=f"$.assignments[{index}].split",
                )
            invalid_keys = tuple(key for key in row if not isinstance(key, str))
            if invalid_keys:
                raise EDAContractError(
                    "eda.test_lock.assignment_field_invalid",
                    "target-free assignment field names must be strings",
                    path=f"$.assignments[{index}]",
                )
            unknown = set(row) - _ALLOWED_ASSIGNMENT_FIELDS
            if unknown:
                raise EDAContractError(
                    "eda.test_lock.assignment_target_field_forbidden",
                    f"target-free assignment contains unapproved fields {sorted(unknown)!r}",
                    path=f"$.assignments[{index}]",
                )
            try:
                assignment = TargetFreeSplitAssignment(
                    corpus=row["corpus"],  # type: ignore[arg-type]
                    record_id=row["record_id"],  # type: ignore[arg-type]
                    split=split_value,  # type: ignore[arg-type]
                    assignment_manifest_fingerprint=row[
                        "assignment_manifest_fingerprint"
                    ],  # type: ignore[arg-type]
                    target_free=row.get("target_free", False),  # type: ignore[arg-type]
                )
            except KeyError as exc:
                raise EDAContractError(
                    "eda.test_lock.assignment_field_missing",
                    f"missing target-free assignment field {exc.args[0]!r}",
                    path=f"$.assignments[{index}]",
                ) from exc
            if assignment.corpus != self.corpus:
                raise EDAContractError(
                    "eda.test_lock.assignment_corpus_mismatch",
                    "assignment corpus differs from guard corpus",
                    path=f"$.assignments[{index}].corpus",
                )
            allowed.append(assignment)

        assignment_fingerprints = {
            item.assignment_manifest_fingerprint for item in allowed
        }
        if len(assignment_fingerprints) > 1:
            raise EDAContractError(
                "eda.test_lock.assignment_manifest_mismatch",
                "all TRAIN/VALIDATION rows must bind one assignment manifest",
            )
        if not allowed:
            raise EDAContractError(
                "eda.test_lock.allowed_assignment_empty",
                "supervision target access requires at least one TRAIN/VALIDATION assignment",
            )
        allowed_keys = tuple((item.corpus, item.record_id) for item in allowed)
        if len(allowed_keys) != len(set(allowed_keys)):
            raise EDAContractError(
                "eda.test_lock.assignment_duplicate",
                "one record ID cannot belong to more than one TRAIN/VALIDATION assignment",
            )

        loaded: list[LoadedT] = []
        for assignment in allowed:
            descriptor = resolve_descriptor(assignment.record_id, assignment.split)
            loaded.append(load_target(descriptor, assignment.split))

        return tuple(loaded), TestTargetLockEvidence.from_guard(
            test_assignment_count=test_assignment_count,
            assignment_manifest_fingerprint=next(iter(assignment_fingerprints)),
            evidence_scope=self.evidence_scope,
            provenance=self.provenance,
        )


def load_supervision_train_validation_only(
    corpus: CorpusId | str,
    assignments: Sequence[Mapping[str, object]],
    *,
    resolve_descriptor: Callable[[str, SplitScope], DescriptorT],
    load_target: Callable[[DescriptorT, SplitScope], LoadedT],
    evidence_scope: EvidenceScope | str = EvidenceScope.FIXTURE,
    provenance: tuple[str, ...] = ("supervision-target-access-guard",),
) -> tuple[tuple[LoadedT, ...], TestTargetLockEvidence]:
    """Functional façade for the immutable no-unlock access guard."""

    return SupervisionTargetAccessGuard[DescriptorT, LoadedT](
        corpus=corpus,  # type: ignore[arg-type]
        evidence_scope=evidence_scope,  # type: ignore[arg-type]
        provenance=provenance,
    ).load_train_validation(
        assignments,
        resolve_descriptor=resolve_descriptor,
        load_target=load_target,
    )


__all__ = [
    "SupervisionTargetAccessGuard",
    "TargetFreeSplitAssignment",
    "load_supervision_train_validation_only",
]
