"""Versioned tensor/CPU encodings for the source-native target ontology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal, TypeAlias

from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
    DILEMMADATA_TARGET_FAMILIES,
)
from music_critic.tasks.ontology import TARGET_FAMILIES, TargetFamilySpec


TARGET_ENCODING_REGISTRY_VERSION = "1.0.0"

EncodingKind: TypeAlias = Literal[
    "closed_categorical_index",
    "closed_multilabel",
    "open_string_cpu",
]
EncodingDType: TypeAlias = Literal["torch.long", "torch.bool", "cpu.str"]
SupervisionRegime: TypeAlias = Literal[
    "fully_supervised",
    "positive_unlabeled",
    "deferred_open_vocabulary",
]


@dataclass(frozen=True, slots=True)
class TargetEncodingSpec:
    """One stable, source-native target encoding without semantic crosswalks."""

    task_id: str
    registry_version: str
    encoding_kind: EncodingKind
    dtype: EncodingDType
    shape: str
    vocabulary: tuple[str, ...] | None
    unavailable_sentinel: int | bool | None
    model_ready: bool
    deferred_reason: str | None
    supervision_regime: SupervisionRegime

    def __post_init__(self) -> None:
        if self.registry_version not in {
            TARGET_ENCODING_REGISTRY_VERSION,
            DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
        }:
            raise ValueError("target encoding registry version is inconsistent")
        if not self.task_id:
            raise ValueError("target encoding task ID must be non-empty")
        if self.model_ready == (self.deferred_reason is not None):
            raise ValueError(
                "model-ready encodings have no deferred reason and deferred "
                "encodings require one"
            )
        if self.supervision_regime == "deferred_open_vocabulary":
            if self.model_ready:
                raise ValueError(
                    "deferred open-vocabulary supervision cannot be model-ready"
                )
        elif not self.model_ready:
            raise ValueError(
                "non-deferred supervision regimes require model-ready encodings"
            )
        if self.encoding_kind == "closed_categorical_index":
            if (
                self.dtype != "torch.long"
                or not self.vocabulary
                or self.unavailable_sentinel != -1
                or self.shape != "[N]"
            ):
                raise ValueError("closed categorical encoding contract is invalid")
        elif self.encoding_kind == "closed_multilabel":
            if (
                self.dtype != "torch.bool"
                or not self.vocabulary
                or self.unavailable_sentinel is not False
                or self.shape != "[N, C]"
            ):
                raise ValueError("closed multilabel encoding contract is invalid")
        elif (
            self.dtype != "cpu.str"
            or self.vocabulary is not None
            or self.unavailable_sentinel is not None
            or self.shape != "[N] CPU"
            or self.model_ready
            or self.supervision_regime != "deferred_open_vocabulary"
        ):
            raise ValueError("open string encoding contract is invalid")


def _encoding_spec(
    task: TargetFamilySpec,
    *,
    registry_version: str = TARGET_ENCODING_REGISTRY_VERSION,
) -> TargetEncodingSpec:
    if task.vocabulary is None:
        return TargetEncodingSpec(
            task_id=task.task_id,
            registry_version=registry_version,
            encoding_kind="open_string_cpu",
            dtype="cpu.str",
            shape="[N] CPU",
            vocabulary=None,
            unavailable_sentinel=None,
            model_ready=False,
            deferred_reason=(
                "open source vocabulary is preserved losslessly; no dynamic "
                "batch/worker vocabulary or numeric IDs"
            ),
            supervision_regime="deferred_open_vocabulary",
        )
    if task.value_type == "multi_label":
        return TargetEncodingSpec(
            task_id=task.task_id,
            registry_version=registry_version,
            encoding_kind="closed_multilabel",
            dtype="torch.bool",
            shape="[N, C]",
            vocabulary=task.vocabulary,
            unavailable_sentinel=False,
            model_ready=True,
            deferred_reason=None,
            supervision_regime="fully_supervised",
        )
    return TargetEncodingSpec(
        task_id=task.task_id,
        registry_version=registry_version,
        encoding_kind="closed_categorical_index",
        dtype="torch.long",
        shape="[N]",
        vocabulary=task.vocabulary,
        unavailable_sentinel=-1,
        model_ready=True,
        deferred_reason=None,
        supervision_regime=(
            "positive_unlabeled"
            if task.supervision_objective
            in {
                "positive_unlabeled_event_detection",
                "positive_unlabeled_coverage_detection",
            }
            else "fully_supervised"
        ),
    )


TARGET_ENCODINGS = tuple(_encoding_spec(task) for task in TARGET_FAMILIES)
TARGET_ENCODING_BY_TASK = MappingProxyType(
    {spec.task_id: spec for spec in TARGET_ENCODINGS}
)
if len(TARGET_ENCODING_BY_TASK) != len(TARGET_ENCODINGS):
    raise RuntimeError("target encoding registry contains duplicate task IDs")

DILEMMADATA_TARGET_ENCODINGS = tuple(
    _encoding_spec(
        task,
        registry_version=DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
    )
    for task in DILEMMADATA_TARGET_FAMILIES
)
DILEMMADATA_TARGET_ENCODING_BY_TASK = MappingProxyType(
    {spec.task_id: spec for spec in DILEMMADATA_TARGET_ENCODINGS}
)


def target_encoding_spec(task_id: str) -> TargetEncodingSpec:
    """Resolve a core or explicitly registered Dilemmadata encoding."""

    spec = TARGET_ENCODING_BY_TASK.get(task_id)
    if spec is None:
        spec = DILEMMADATA_TARGET_ENCODING_BY_TASK.get(task_id)
    if spec is None:
        raise KeyError(task_id)
    return spec


def target_encoding_contract_dict() -> dict[str, object]:
    """Return the deterministic target encoding registry mapping."""

    return {
        "target_encoding_registry_version": TARGET_ENCODING_REGISTRY_VERSION,
        "encodings": [
            {
                **asdict(spec),
                "vocabulary": (
                    list(spec.vocabulary)
                    if spec.vocabulary is not None
                    else None
                ),
            }
            for spec in TARGET_ENCODINGS
        ],
    }


def dumps_target_encoding_contract(*, indent: int | None = None) -> str:
    """Serialize the target encoding registry deterministically."""

    return json.dumps(
        target_encoding_contract_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def target_encoding_contract_fingerprint() -> str:
    """Return SHA-256 of canonical compact target encoding serialization."""

    return sha256(dumps_target_encoding_contract().encode("utf-8")).hexdigest()


def dilemmadata_target_encoding_contract_dict() -> dict[str, object]:
    """Return the separate source-native Dilemmadata encoding registry."""

    return {
        "target_encoding_registry_version": (
            DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION
        ),
        "encodings": [
            {
                **asdict(spec),
                "vocabulary": (
                    list(spec.vocabulary)
                    if spec.vocabulary is not None
                    else None
                ),
            }
            for spec in DILEMMADATA_TARGET_ENCODINGS
        ],
    }


def dumps_dilemmadata_target_encoding_contract(
    *,
    indent: int | None = None,
) -> str:
    """Serialize the Dilemmadata encoding registry deterministically."""

    return json.dumps(
        dilemmadata_target_encoding_contract_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def dilemmadata_target_encoding_contract_fingerprint() -> str:
    """Return SHA-256 of the source-native encoding contract."""

    return sha256(
        dumps_dilemmadata_target_encoding_contract().encode("utf-8")
    ).hexdigest()


__all__ = [
    "EncodingDType",
    "EncodingKind",
    "SupervisionRegime",
    "TARGET_ENCODINGS",
    "TARGET_ENCODING_BY_TASK",
    "TARGET_ENCODING_REGISTRY_VERSION",
    "DILEMMADATA_TARGET_ENCODINGS",
    "DILEMMADATA_TARGET_ENCODING_BY_TASK",
    "TargetEncodingSpec",
    "dilemmadata_target_encoding_contract_dict",
    "dilemmadata_target_encoding_contract_fingerprint",
    "dumps_dilemmadata_target_encoding_contract",
    "dumps_target_encoding_contract",
    "target_encoding_contract_dict",
    "target_encoding_contract_fingerprint",
    "target_encoding_spec",
]
