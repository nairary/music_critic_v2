"""Versioned, target-blind contracts for Phase 7A masked graph SSL."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Literal


SSL_CONTRACT_VERSION = "1.1.0"
MASK_PLAN_CONTRACT_VERSION = "1.0.0"
MASK_POLICY_VERSION = "1.0.0"
MASKED_FEATURE_OVERLAY_CONTRACT_VERSION = "1.0.0"
PREPARED_MASK_BINDING_CONTRACT_VERSION = "1.0.0"

UNIFORM_NOTE_MASK_POLICY = "uniform_note_without_replacement"

MaskStage = Literal["train", "validation"]
FeatureKind = Literal["categorical", "continuous"]

_SHA256_HEX_LENGTH = 64
_MAX_GLOBAL_SEED = (1 << 63) - 1


class SSLContractError(ValueError):
    """Raised when a Phase 7A masking or view contract is invalid."""


def canonical_sha256(value: object) -> str:
    """Return SHA-256 over compact, sorted, finite JSON."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SSLContractError("SSL contract payload is not finite JSON") from exc
    return sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase SHA-256 hexadecimal digest."""

    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_global_seed(value: object) -> int:
    """Validate the public deterministic seed domain."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_GLOBAL_SEED
    ):
        raise SSLContractError(
            f"global_seed must be an integer in [0, {_MAX_GLOBAL_SEED}]"
        )
    return value


def validate_non_negative_integer(value: object, *, name: str) -> int:
    """Validate a non-negative integer without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SSLContractError(f"{name} must be a non-negative integer")
    return value


def validate_mask_rate(value: object) -> float:
    """Validate a finite mask rate in the closed unit interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise SSLContractError("requested_mask_rate must be finite and in [0, 1]")
    return float(value)


def _validate_identity(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SSLContractError(
            f"{name} must be a non-empty trimmed string without control characters"
        )
    return value


def _validate_indices(
    values: tuple[int, ...],
    *,
    name: str,
    upper_bound: int | None = None,
) -> None:
    if not isinstance(values, tuple):
        raise SSLContractError(f"{name} must be a tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise SSLContractError(f"{name} must contain non-negative integers")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise SSLContractError(f"{name} must be uniquely sorted")
    if upper_bound is not None and any(value >= upper_bound for value in values):
        raise SSLContractError(f"{name} contains an out-of-range local index")


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    """Dataset/piece identity used only by deterministic sidecar construction."""

    dataset_id: str
    piece_id: str

    def __post_init__(self) -> None:
        _validate_identity(self.dataset_id, name="dataset_id")
        _validate_identity(self.piece_id, name="piece_id")

    def to_dict(self) -> dict[str, str]:
        return {"dataset_id": self.dataset_id, "piece_id": self.piece_id}


@dataclass(frozen=True, slots=True)
class StableSeed:
    """A portable integer seed bound to its complete SHA-256 derivation."""

    value: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or not 0 <= self.value < (1 << 64)
        ):
            raise SSLContractError("stable seed value must be an unsigned 64-bit integer")
        if not is_sha256(self.sha256):
            raise SSLContractError("stable seed digest must be lowercase SHA-256")
        if self.value != int(self.sha256[:16], 16):
            raise SSLContractError("stable seed value differs from its SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MaskedFeature:
    """One named raw-registry feature whose value and availability are hidden."""

    node_type: str
    kind: FeatureKind
    feature_name: str
    mask_availability: bool = True

    def __post_init__(self) -> None:
        _validate_identity(self.node_type, name="masked feature node_type")
        if self.kind not in {"categorical", "continuous"}:
            raise SSLContractError("masked feature kind is invalid")
        _validate_identity(self.feature_name, name="masked feature name")
        if self.mask_availability is not True:
            raise SSLContractError(
                "Phase 7A masked fields must hide their availability contribution"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_type": self.node_type,
            "kind": self.kind,
            "feature_name": self.feature_name,
            "mask_availability": self.mask_availability,
        }


@dataclass(frozen=True, slots=True)
class CollateralFeatureMask:
    """Non-target feature masks needed to close a primary-field shortcut."""

    reason: str
    node_type: str
    local_node_indices: tuple[int, ...]
    features: tuple[MaskedFeature, ...]

    def __post_init__(self) -> None:
        _validate_identity(self.reason, name="collateral mask reason")
        _validate_identity(self.node_type, name="collateral mask node_type")
        _validate_indices(
            self.local_node_indices,
            name="collateral local_node_indices",
        )
        if (
            not isinstance(self.features, tuple)
            or not self.features
            or not all(isinstance(field, MaskedFeature) for field in self.features)
        ):
            raise SSLContractError("collateral feature mask must name at least one field")
        identities = tuple(
            (field.node_type, field.kind, field.feature_name) for field in self.features
        )
        if len(identities) != len(set(identities)):
            raise SSLContractError("collateral features must be unique")
        if any(field.node_type != self.node_type for field in self.features):
            raise SSLContractError(
                "collateral feature node types must match the collateral store"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "node_type": self.node_type,
            "local_node_indices": list(self.local_node_indices),
            "features": [field.to_dict() for field in self.features],
        }


def _mask_plan_payload_values(
    *,
    contract_version: str,
    mask_policy: str,
    mask_policy_version: str,
    dataset_id: str,
    piece_id: str,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    selected_node_type: str,
    selected_local_node_indices: tuple[int, ...],
    primary_feature_group: str,
    collateral_feature_masks: tuple[CollateralFeatureMask, ...],
    requested_mask_rate: float,
    maskable_node_count: int,
    realized_mask_rate: float,
    global_seed: int,
    stable_seed: int,
    stable_seed_sha256: str,
) -> dict[str, object]:
    return {
        "contract_version": contract_version,
        "mask_policy": mask_policy,
        "mask_policy_version": mask_policy_version,
        "sample_identity": {
            "dataset_id": dataset_id,
            "piece_id": piece_id,
        },
        "stage": stage,
        "epoch": epoch,
        "encoder_view_index": encoder_view_index,
        "selected_node_type": selected_node_type,
        "selected_local_node_indices": list(selected_local_node_indices),
        "primary_feature_group": primary_feature_group,
        "collateral_feature_masks": [
            mask.to_dict() for mask in collateral_feature_masks
        ],
        "requested_mask_rate": requested_mask_rate,
        "maskable_node_count": maskable_node_count,
        "realized_mask_rate": realized_mask_rate,
        "global_seed": global_seed,
        "stable_seed": stable_seed,
        "stable_seed_sha256": stable_seed_sha256,
    }


def _mask_plan_payload(plan: MaskPlan) -> dict[str, object]:
    return _mask_plan_payload_values(
        contract_version=plan.contract_version,
        mask_policy=plan.mask_policy,
        mask_policy_version=plan.mask_policy_version,
        dataset_id=plan.dataset_id,
        piece_id=plan.piece_id,
        stage=plan.stage,
        epoch=plan.epoch,
        encoder_view_index=plan.encoder_view_index,
        selected_node_type=plan.selected_node_type,
        selected_local_node_indices=plan.selected_local_node_indices,
        primary_feature_group=plan.primary_feature_group,
        collateral_feature_masks=plan.collateral_feature_masks,
        requested_mask_rate=plan.requested_mask_rate,
        maskable_node_count=plan.maskable_node_count,
        realized_mask_rate=plan.realized_mask_rate,
        global_seed=plan.global_seed,
        stable_seed=plan.stable_seed,
        stable_seed_sha256=plan.stable_seed_sha256,
    )


@dataclass(frozen=True, slots=True)
class MaskPlan:
    """One immutable, versioned, per-sample encoder mask plan."""

    contract_version: str
    mask_policy: str
    mask_policy_version: str
    dataset_id: str
    piece_id: str
    stage: MaskStage
    epoch: int
    encoder_view_index: int
    selected_node_type: str
    selected_local_node_indices: tuple[int, ...]
    primary_feature_group: str
    collateral_feature_masks: tuple[CollateralFeatureMask, ...]
    requested_mask_rate: float
    maskable_node_count: int
    realized_mask_rate: float
    global_seed: int
    stable_seed: int
    stable_seed_sha256: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        mask_policy: str,
        mask_policy_version: str,
        dataset_id: str,
        piece_id: str,
        stage: MaskStage,
        epoch: int,
        encoder_view_index: int,
        selected_node_type: str,
        selected_local_node_indices: tuple[int, ...],
        primary_feature_group: str,
        collateral_feature_masks: tuple[CollateralFeatureMask, ...],
        requested_mask_rate: float,
        maskable_node_count: int,
        realized_mask_rate: float,
        global_seed: int,
        stable_seed: int,
        stable_seed_sha256: str,
    ) -> MaskPlan:
        """Create a plan with its fingerprint bound before validation."""

        values = {
            "contract_version": MASK_PLAN_CONTRACT_VERSION,
            "mask_policy": mask_policy,
            "mask_policy_version": mask_policy_version,
            "dataset_id": dataset_id,
            "piece_id": piece_id,
            "stage": stage,
            "epoch": epoch,
            "encoder_view_index": encoder_view_index,
            "selected_node_type": selected_node_type,
            "selected_local_node_indices": selected_local_node_indices,
            "primary_feature_group": primary_feature_group,
            "collateral_feature_masks": collateral_feature_masks,
            "requested_mask_rate": requested_mask_rate,
            "maskable_node_count": maskable_node_count,
            "realized_mask_rate": realized_mask_rate,
            "global_seed": global_seed,
            "stable_seed": stable_seed,
            "stable_seed_sha256": stable_seed_sha256,
        }
        fingerprint = canonical_sha256(_mask_plan_payload_values(**values))
        return cls(**values, fingerprint=fingerprint)

    def __post_init__(self) -> None:
        if self.contract_version != MASK_PLAN_CONTRACT_VERSION:
            raise SSLContractError("mask plan contract version is incompatible")
        if self.mask_policy != UNIFORM_NOTE_MASK_POLICY:
            raise SSLContractError("mask plan policy is not supported by Phase 7A")
        if self.mask_policy_version != MASK_POLICY_VERSION:
            raise SSLContractError("mask policy version is incompatible")
        SampleIdentity(self.dataset_id, self.piece_id)
        if self.stage not in {"train", "validation"}:
            raise SSLContractError("mask stage must be train or validation")
        validate_non_negative_integer(self.epoch, name="epoch")
        if self.stage == "validation" and self.epoch != 0:
            raise SSLContractError("validation mask plans must use canonical epoch zero")
        validate_non_negative_integer(
            self.encoder_view_index,
            name="encoder_view_index",
        )
        if self.selected_node_type != "note":
            raise SSLContractError("Phase 7A selects note nodes only")
        validate_non_negative_integer(
            self.maskable_node_count,
            name="maskable_node_count",
        )
        _validate_indices(
            self.selected_local_node_indices,
            name="selected_local_node_indices",
            upper_bound=self.maskable_node_count,
        )
        if self.primary_feature_group != "note_pitch_group":
            raise SSLContractError("Phase 7A supports only note_pitch_group")
        if not isinstance(self.collateral_feature_masks, tuple):
            raise SSLContractError("collateral_feature_masks must be a tuple")
        if not all(
            isinstance(mask, CollateralFeatureMask)
            for mask in self.collateral_feature_masks
        ):
            raise SSLContractError(
                "collateral_feature_masks must contain collateral mask contracts"
            )
        expected_collateral_keys = (
            ("note", "owner_track_peer_relative_pitch"),
            ("track", "owner_track_pitch_statistics"),
        )
        collateral_keys = tuple(
            (mask.node_type, mask.reason)
            for mask in self.collateral_feature_masks
        )
        if collateral_keys != expected_collateral_keys:
            raise SSLContractError(
                "note_pitch_group requires peer-relative-note and "
                "owner-track-statistics collateral masks"
            )
        rate = validate_mask_rate(self.requested_mask_rate)
        if not isinstance(self.requested_mask_rate, float) or rate != (
            self.requested_mask_rate
        ):
            raise SSLContractError("requested mask rate must use canonical float form")
        if (
            isinstance(self.realized_mask_rate, bool)
            or not isinstance(self.realized_mask_rate, float)
            or not math.isfinite(self.realized_mask_rate)
            or not 0 <= self.realized_mask_rate <= 1
        ):
            raise SSLContractError("realized mask rate must be a finite float in [0, 1]")
        expected_rate = (
            len(self.selected_local_node_indices) / self.maskable_node_count
            if self.maskable_node_count
            else 0.0
        )
        if self.realized_mask_rate != expected_rate:
            raise SSLContractError(
                "realized mask rate differs from selected/maskable node counts"
            )
        if rate == 0 and self.selected_local_node_indices:
            raise SSLContractError("zero requested mask rate selected nodes")
        if rate > 0 and self.maskable_node_count and not self.selected_local_node_indices:
            raise SSLContractError(
                "positive requested mask rate produced an empty non-empty-sample mask"
            )
        if rate == 1 and len(self.selected_local_node_indices) != self.maskable_node_count:
            raise SSLContractError("unit mask rate must select every maskable node")
        validate_global_seed(self.global_seed)
        StableSeed(self.stable_seed, self.stable_seed_sha256)
        if not is_sha256(self.fingerprint):
            raise SSLContractError("mask plan fingerprint must be lowercase SHA-256")
        expected_fingerprint = canonical_sha256(_mask_plan_payload(self))
        if self.fingerprint != expected_fingerprint:
            raise SSLContractError("mask plan fingerprint differs from its content")

    @property
    def sample_identity(self) -> tuple[str, str]:
        """Return the required ``(dataset_id, piece_id)`` sidecar identity."""

        return self.dataset_id, self.piece_id

    @property
    def sample_identity_record(self) -> SampleIdentity:
        """Return the validated structured form used by seed derivation."""

        return SampleIdentity(self.dataset_id, self.piece_id)

    @property
    def selected_count(self) -> int:
        return len(self.selected_local_node_indices)

    @property
    def collateral_node_count(self) -> int:
        return sum(len(mask.local_node_indices) for mask in self.collateral_feature_masks)

    def to_dict(self) -> dict[str, object]:
        payload = _mask_plan_payload(self)
        payload["fingerprint"] = self.fingerprint
        return payload


def mask_plan_fingerprint(plan: MaskPlan) -> str:
    """Recompute a plan fingerprint from its semantic contents."""

    return canonical_sha256(_mask_plan_payload(plan))


__all__ = [
    "MASKED_FEATURE_OVERLAY_CONTRACT_VERSION",
    "MASK_PLAN_CONTRACT_VERSION",
    "MASK_POLICY_VERSION",
    "PREPARED_MASK_BINDING_CONTRACT_VERSION",
    "SSL_CONTRACT_VERSION",
    "UNIFORM_NOTE_MASK_POLICY",
    "CollateralFeatureMask",
    "FeatureKind",
    "MaskPlan",
    "MaskStage",
    "MaskedFeature",
    "SSLContractError",
    "SampleIdentity",
    "StableSeed",
    "canonical_sha256",
    "is_sha256",
    "mask_plan_fingerprint",
    "validate_global_seed",
    "validate_mask_rate",
    "validate_non_negative_integer",
]
