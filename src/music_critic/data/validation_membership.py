"""Backward-compatible fixed-validation membership shared across phases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any


FIXED_VALIDATION_MEMBERSHIP_CONTRACT_VERSION = "1.0.0"
FIXED_VALIDATION_MEMBERSHIP_POLICY = "fixed_validation_membership_v1"


class ValidationMembershipContractError(ValueError):
    """Invalid input to the fixed validation membership contract."""


@dataclass(frozen=True, slots=True)
class FixedValidationMembership:
    """One canonical no-replacement selection and its legacy fingerprint."""

    indices: tuple[int, ...]
    identities: tuple[tuple[str, str], ...]
    membership_payload: dict[str, Any]
    membership_fingerprint: str
    dataset_counts: dict[str, int]
    full_view_count: int
    selected_count: int
    subset_limit: int

    def evidence(self) -> dict[str, Any]:
        return {
            "fixed_validation_membership_contract_version": (
                FIXED_VALIDATION_MEMBERSHIP_CONTRACT_VERSION
            ),
            **self.membership_payload,
            "selected_count": self.selected_count,
            "dataset_counts": self.dataset_counts,
            "membership_fingerprint": self.membership_fingerprint,
        }


def _legacy_phase6c_json_bytes(value: object) -> bytes:
    """Preserve the exact compact, no-newline Phase 6C byte encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _legacy_phase6c_fingerprint(value: object) -> str:
    return sha256(_legacy_phase6c_json_bytes(value)).hexdigest()


def _validate(
    identities: Sequence[tuple[str, str]],
    *,
    limit: int,
    seed: int,
) -> tuple[tuple[str, str], ...]:
    if isinstance(identities, (str, bytes)) or not isinstance(
        identities, Sequence
    ):
        raise ValidationMembershipContractError(
            "validation_membership.identities_invalid"
        )
    resolved = tuple(identities)
    if any(
        not isinstance(identity, tuple)
        or len(identity) != 2
        or any(
            not isinstance(value, str) or not value
            for value in identity
        )
        for identity in resolved
    ):
        raise ValidationMembershipContractError(
            "validation_membership.identity_invalid"
        )
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
        or limit > len(resolved)
    ):
        raise ValidationMembershipContractError(
            "validation_membership.limit_invalid"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationMembershipContractError(
            "validation_membership.seed_invalid"
        )
    return resolved


def fixed_validation_membership(
    identities: Sequence[tuple[str, str]],
    *,
    limit: int,
    seed: int,
) -> FixedValidationMembership:
    """Select and fingerprint exactly as existing Phase 6C checkpoints did."""

    resolved = _validate(identities, limit=limit, seed=seed)
    if limit == 0 or limit == len(resolved):
        indices = tuple(range(len(resolved)))
    else:
        ranked = sorted(
            range(len(resolved)),
            key=lambda index: (
                _legacy_phase6c_fingerprint(
                    {
                        "policy": FIXED_VALIDATION_MEMBERSHIP_POLICY,
                        "seed": seed,
                        "identity": list(resolved[index]),
                    }
                ),
                resolved[index],
            ),
        )
        # Membership is selected by hash but emitted in canonical view order.
        indices = tuple(sorted(ranked[:limit]))
    selected = tuple(resolved[index] for index in indices)
    payload = {
        "policy": FIXED_VALIDATION_MEMBERSHIP_POLICY,
        "seed": seed,
        "subset_limit": limit,
        "full_view_count": len(resolved),
        "selected_identities": [list(item) for item in selected],
    }
    counts = Counter(dataset_id for dataset_id, _piece_id in selected)
    return FixedValidationMembership(
        indices=indices,
        identities=selected,
        membership_payload=payload,
        membership_fingerprint=_legacy_phase6c_fingerprint(payload),
        dataset_counts=dict(sorted(counts.items())),
        full_view_count=len(resolved),
        selected_count=len(selected),
        subset_limit=limit,
    )


__all__ = [
    "FIXED_VALIDATION_MEMBERSHIP_CONTRACT_VERSION",
    "FIXED_VALIDATION_MEMBERSHIP_POLICY",
    "FixedValidationMembership",
    "ValidationMembershipContractError",
    "fixed_validation_membership",
]
