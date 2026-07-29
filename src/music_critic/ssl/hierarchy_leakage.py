"""Fail-closed Phase 8A audit of raw pitch-revealing feature columns."""

from __future__ import annotations

from dataclasses import dataclass

from music_critic.graph import RAW_FEATURE_REGISTRY
from music_critic.models.checkpoint import feature_registry_fingerprint
from music_critic.ssl.contracts import (
    SSLContractError,
    canonical_sha256,
    is_sha256,
)
from music_critic.ssl.field_registry import NOTE_PITCH_GROUP


PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION = "1.0.0"
AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT = (
    "567a5fdbb0d132010af4716c5988686c2bdf998cf6f1b2eec897f8af3ca8c0e2"
)


def _identity(field: object) -> tuple[str, str, str]:
    return (
        getattr(field, "node_type"),
        getattr(field, "kind"),
        getattr(field, "feature_name", getattr(field, "name", None)),
    )


def _payload(audit: Phase8APitchLeakageAudit) -> dict[str, object]:
    return {
        "contract_version": audit.contract_version,
        "raw_feature_registry_version": audit.raw_feature_registry_version,
        "raw_feature_registry_fingerprint": (
            audit.raw_feature_registry_fingerprint
        ),
        "classified_raw_feature_count": (
            audit.classified_raw_feature_count
        ),
        "primary_note_pitch_fields": [
            list(value) for value in audit.primary_note_pitch_fields
        ],
        "peer_note_collateral_fields": [
            list(value) for value in audit.peer_note_collateral_fields
        ],
        "owner_track_collateral_fields": [
            list(value) for value in audit.owner_track_collateral_fields
        ],
        "visible_raw_fields": [
            list(value) for value in audit.visible_raw_fields
        ],
        "topology_boundary": (
            "same_track_simultaneous_note_order_can_expose_relative_"
            "pitch_rank_but_not_an_exact_duplicated_pitch_value"
        ),
    }


@dataclass(frozen=True, slots=True)
class Phase8APitchLeakageAudit:
    """Pinned exhaustive classification beside the unchanged Phase 7A registry."""

    contract_version: str
    raw_feature_registry_version: str
    raw_feature_registry_fingerprint: str
    classified_raw_feature_count: int
    primary_note_pitch_fields: tuple[tuple[str, str, str], ...]
    peer_note_collateral_fields: tuple[tuple[str, str, str], ...]
    owner_track_collateral_fields: tuple[tuple[str, str, str], ...]
    visible_raw_fields: tuple[tuple[str, str, str], ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION
            or self.raw_feature_registry_version
            != RAW_FEATURE_REGISTRY.version
            or self.raw_feature_registry_fingerprint
            != AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT
            or self.classified_raw_feature_count
            != len(RAW_FEATURE_REGISTRY.specs)
        ):
            raise SSLContractError(
                "phase8a.leakage_audit.registry_incompatible"
            )
        expected_primary = tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.primary_fields
        )
        expected_peer = tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.peer_note_collateral_fields
        )
        expected_track = tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.collateral_fields
        )
        hidden_identities = set(
            (*expected_primary, *expected_track)
        )
        expected_visible = tuple(
            _identity(field)
            for field in RAW_FEATURE_REGISTRY.specs
            if _identity(field) not in hidden_identities
        )
        if (
            self.primary_note_pitch_fields != expected_primary
            or self.peer_note_collateral_fields != expected_peer
            or self.owner_track_collateral_fields != expected_track
            or self.visible_raw_fields != expected_visible
            or len(
                {
                    *self.primary_note_pitch_fields,
                    *self.owner_track_collateral_fields,
                    *self.visible_raw_fields,
                }
            )
            != self.classified_raw_feature_count
        ):
            raise SSLContractError(
                "phase8a.leakage_audit.closure_incompatible"
            )
        if (
            not is_sha256(self.fingerprint)
            or self.fingerprint != canonical_sha256(_payload(self))
        ):
            raise SSLContractError(
                "phase8a.leakage_audit.fingerprint_incompatible"
            )

    def to_dict(self) -> dict[str, object]:
        payload = _payload(self)
        payload["fingerprint"] = self.fingerprint
        return payload


def build_phase8a_pitch_leakage_audit() -> Phase8APitchLeakageAudit:
    """Build the pinned audit, failing if any raw-registry contract changed."""

    current = feature_registry_fingerprint()
    if current != AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT:
        raise SSLContractError(
            "phase8a.leakage_audit.raw_registry_changed"
        )
    values = {
        "contract_version": (
            PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION
        ),
        "raw_feature_registry_version": RAW_FEATURE_REGISTRY.version,
        "raw_feature_registry_fingerprint": current,
        "classified_raw_feature_count": len(
            RAW_FEATURE_REGISTRY.specs
        ),
        "primary_note_pitch_fields": tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.primary_fields
        ),
        "peer_note_collateral_fields": tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.peer_note_collateral_fields
        ),
        "owner_track_collateral_fields": tuple(
            _identity(field)
            for field in NOTE_PITCH_GROUP.collateral_fields
        ),
        "visible_raw_fields": tuple(
            _identity(field)
            for field in RAW_FEATURE_REGISTRY.specs
            if _identity(field)
            not in {
                *(
                    _identity(primary)
                    for primary in NOTE_PITCH_GROUP.primary_fields
                ),
                *(
                    _identity(collateral)
                    for collateral in NOTE_PITCH_GROUP.collateral_fields
                ),
            }
        ),
    }
    provisional = object.__new__(Phase8APitchLeakageAudit)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "fingerprint", "0" * 64)
    return Phase8APitchLeakageAudit(
        **values,
        fingerprint=canonical_sha256(_payload(provisional)),
    )


PHASE8A_PITCH_LEAKAGE_AUDIT = build_phase8a_pitch_leakage_audit()
PHASE8A_PITCH_LEAKAGE_AUDIT_FINGERPRINT = (
    PHASE8A_PITCH_LEAKAGE_AUDIT.fingerprint
)


__all__ = [
    "AUDITED_RAW_FEATURE_REGISTRY_FINGERPRINT",
    "PHASE8A_PITCH_LEAKAGE_AUDIT",
    "PHASE8A_PITCH_LEAKAGE_AUDIT_CONTRACT_VERSION",
    "PHASE8A_PITCH_LEAKAGE_AUDIT_FINGERPRINT",
    "Phase8APitchLeakageAudit",
    "build_phase8a_pitch_leakage_audit",
]
