from __future__ import annotations

from hashlib import sha256
import json

import pytest

from music_critic.data.validation_membership import (
    FIXED_VALIDATION_MEMBERSHIP_CONTRACT_VERSION,
    ValidationMembershipContractError,
    fixed_validation_membership,
)
from music_critic.evaluation import data as evaluation_data
from music_critic.training import data as training_data


_MIXED_IDENTITIES = (
    ("hooktheory", "h-02"),
    ("pop909_cl", "p-03"),
    ("hooktheory", "h-01"),
    ("pop909_cl", "p-01"),
    ("hooktheory", "h-03"),
    ("pop909_cl", "p-02"),
)

# These bytes and digests were captured from the pre-Phase-6D Phase 6C
# implementation. They are deliberately literal and are never derived through
# the contract under test.
_LEGACY_ORACLES = (
    (
        0,
        42,
        (0, 1, 2, 3, 4, 5),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":42,"selected_identities":[["hooktheory","h-02"],'
        b'["pop909_cl","p-03"],["hooktheory","h-01"],'
        b'["pop909_cl","p-01"],["hooktheory","h-03"],'
        b'["pop909_cl","p-02"]],"subset_limit":0}',
        "50fce65f51d4ea85c198b6c15600de0f373daafecbe461ae31ada138e4851a5a",
    ),
    (
        1,
        42,
        (5,),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":42,"selected_identities":[["pop909_cl","p-02"]],'
        b'"subset_limit":1}',
        "1046bcb10290fdba6171f4f3072a5da0db150a15f5b1bc005b3bb69826acb4eb",
    ),
    (
        3,
        42,
        (1, 4, 5),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":42,"selected_identities":[["pop909_cl","p-03"],'
        b'["hooktheory","h-03"],["pop909_cl","p-02"]],'
        b'"subset_limit":3}',
        "29804882e03cb33841c3d3a8d5c5be04da021bdb029c486d67bc3b326109be66",
    ),
    (
        6,
        42,
        (0, 1, 2, 3, 4, 5),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":42,"selected_identities":[["hooktheory","h-02"],'
        b'["pop909_cl","p-03"],["hooktheory","h-01"],'
        b'["pop909_cl","p-01"],["hooktheory","h-03"],'
        b'["pop909_cl","p-02"]],"subset_limit":6}',
        "6e5057a510b836ec6121ed8f19f89b58e3065791975a1eb0009fa2eaa64a9238",
    ),
    (
        1,
        7,
        (1,),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":7,"selected_identities":[["pop909_cl","p-03"]],'
        b'"subset_limit":1}',
        "956eec926372e395790981ec6081e652cf221bf20da3ddd4002dbe17785dace8",
    ),
    (
        3,
        7,
        (1, 2, 3),
        b'{"full_view_count":6,"policy":"fixed_validation_membership_v1",'
        b'"seed":7,"selected_identities":[["pop909_cl","p-03"],'
        b'["hooktheory","h-01"],["pop909_cl","p-01"]],'
        b'"subset_limit":3}',
        "ff1d4e278d2e1310c4503a7578edfdce7442ca1a4734ef4d55f9ad4e380b8744",
    ),
)


@pytest.mark.parametrize(
    ("limit", "seed", "indices", "payload_bytes", "fingerprint"),
    _LEGACY_ORACLES,
)
def test_fixed_validation_matches_hard_coded_phase6c_oracles(
    limit: int,
    seed: int,
    indices: tuple[int, ...],
    payload_bytes: bytes,
    fingerprint: str,
) -> None:
    result = fixed_validation_membership(
        _MIXED_IDENTITIES, limit=limit, seed=seed
    )
    actual_bytes = json.dumps(
        result.membership_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert result.indices == indices
    assert result.identities == tuple(
        _MIXED_IDENTITIES[index] for index in indices
    )
    assert actual_bytes == payload_bytes
    assert not actual_bytes.endswith(b"\n")
    assert sha256(actual_bytes).hexdigest() == fingerprint
    assert result.membership_fingerprint == fingerprint


def test_unicode_identity_oracle_preserves_utf8_without_ascii_escaping() -> None:
    identities = (
        ("hooktheory", "песня-α"),
        ("pop909_cl", "曲-02"),
        ("hooktheory", "song-03"),
        ("pop909_cl", "song-04"),
    )
    result = fixed_validation_membership(identities, limit=2, seed=1)
    expected = (
        '{"full_view_count":4,"policy":"fixed_validation_membership_v1",'
        '"seed":1,"selected_identities":[["hooktheory","песня-α"],'
        '["pop909_cl","song-04"]],"subset_limit":2}'
    ).encode()

    assert result.indices == (0, 3)
    assert json.dumps(
        result.membership_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode() == expected
    assert result.membership_fingerprint == (
        "7d56b18d7414532f438e27b8393794a5d4417cf16d0f31a74ed9b53b3b956b59"
    )


def test_training_and_evaluation_import_one_shared_contract() -> None:
    assert training_data.fixed_validation_membership is (
        evaluation_data.fixed_validation_membership
    )
    training = training_data.fixed_validation_membership(
        _MIXED_IDENTITIES, limit=3, seed=42
    )
    evaluation = evaluation_data.fixed_validation_membership(
        _MIXED_IDENTITIES, limit=3, seed=42
    )

    assert training.indices == evaluation.indices
    assert training.identities == evaluation.identities
    assert training.membership_payload == evaluation.membership_payload
    assert training.membership_fingerprint == (
        evaluation.membership_fingerprint
    )
    assert training.dataset_counts == {
        "hooktheory": 1,
        "pop909_cl": 2,
    }
    assert (
        FIXED_VALIDATION_MEMBERSHIP_CONTRACT_VERSION == "1.0.0"
    )


def test_seed_and_subset_limit_change_membership_contract() -> None:
    base = fixed_validation_membership(
        _MIXED_IDENTITIES, limit=3, seed=42
    )
    other_seed = fixed_validation_membership(
        _MIXED_IDENTITIES, limit=3, seed=99
    )
    other_limit = fixed_validation_membership(
        _MIXED_IDENTITIES, limit=1, seed=42
    )

    assert base.membership_fingerprint != (
        other_seed.membership_fingerprint
    )
    assert base.membership_fingerprint != (
        other_limit.membership_fingerprint
    )
    assert base.indices != other_seed.indices
    assert base.indices != other_limit.indices


@pytest.mark.parametrize("limit", [True, -1, 7, 1.5, "1"])
def test_invalid_limit_is_rejected_strictly(limit: object) -> None:
    with pytest.raises(
        ValidationMembershipContractError,
        match="validation_membership.limit_invalid",
    ):
        fixed_validation_membership(
            _MIXED_IDENTITIES, limit=limit, seed=42  # type: ignore[arg-type]
        )
