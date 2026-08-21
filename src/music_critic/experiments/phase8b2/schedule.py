"""Paired, launch-order-independent schedules and compute budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable, Sequence

from music_critic.tasks import DeterministicQuotaSampler, MultiCorpusDataset

from music_critic.experiments.phase8b2.contracts import (
    ComparisonMode,
    Phase8B2ContractError,
    encoder_forwards_per_policy_view,
    fingerprint,
    natural_policy_views,
    variant_modes,
)


SEED_DOMAIN_CONTRACT_VERSION = "1.1.0"
SCHEDULE_CONTRACT_VERSION = "1.2.0"
_POLICIES = {
    "phase7a_control": ("independent_note_pitch",),
    "phase8a_mask_only": (
        "onset_pitch_descendants",
        "beat_pitch_descendants",
        "contiguous_bar_pitch_span",
        "track_bar_pitch_span",
    ),
    "onset_latent": ("onset_pitch_descendants",),
    "beat_latent": ("beat_pitch_descendants",),
    "hierarchy_bar_latent": ("contiguous_bar_pitch_span",),
    "track_latent": ("track_bar_pitch_span",),
    "multilevel_equal": (
        "onset_pitch_descendants",
        "beat_pitch_descendants",
        "contiguous_bar_pitch_span",
        "track_bar_pitch_span",
    ),
}


@dataclass(frozen=True, slots=True)
class RawDownstreamEpochSchedule:
    """One exact sampler epoch shared by planning and training runtime."""

    seed: int
    epoch: int
    epoch_size: int
    batch_size: int
    sampler: DeterministicQuotaSampler
    indices: tuple[int, ...]
    identities: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RawDownstreamSampleSchedule:
    """Exact fixed-update identity schedule over one or more epochs."""

    seed: int
    first_epoch: int
    epochs: int
    steps_per_epoch: int
    batch_size: int
    identities: tuple[tuple[str, str], ...]
    fingerprint: str


def raw_downstream_sample_schedule_fingerprint(
    identities: Sequence[Sequence[str]],
) -> str:
    """Fingerprint normalized identities through the single schedule contract."""

    normalized = tuple(tuple(identity) for identity in identities)
    if any(
        len(identity) != 2
        or not all(isinstance(value, str) and value for value in identity)
        for identity in normalized
    ):
        raise Phase8B2ContractError(
            "phase8b2.schedule.sample_identity_invalid"
        )
    return fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_downstream_sample_schedule",
            "identities": [list(identity) for identity in normalized],
        }
    )


def build_raw_downstream_epoch_schedule(
    dataset: MultiCorpusDataset,
    *,
    weights: dict[str, float],
    seed: int,
    epoch: int,
    epoch_size: int,
    batch_size: int,
) -> RawDownstreamEpochSchedule:
    """Build one exact quota-sampler epoch and its normalized identities."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or epoch_size % batch_size
    ):
        raise Phase8B2ContractError(
            "phase8b2.schedule.partial_batch_forbidden"
        )
    sampler = DeterministicQuotaSampler(
        dataset,
        weights=weights,
        seed=seed,
        epoch_size=epoch_size,
    )
    sampler.set_epoch(epoch)
    indices = tuple(iter(sampler))
    identities = tuple(dataset.record_identity(index) for index in indices)
    return RawDownstreamEpochSchedule(
        seed=seed,
        epoch=epoch,
        epoch_size=epoch_size,
        batch_size=batch_size,
        sampler=sampler,
        indices=indices,
        identities=identities,
    )


def build_raw_downstream_sample_schedule(
    dataset: MultiCorpusDataset,
    *,
    weights: dict[str, float],
    seed: int,
    first_epoch: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
) -> RawDownstreamSampleSchedule:
    """Build the exact fixed-update schedule used by a runtime configuration."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (epochs, steps_per_epoch, batch_size)
    ) or (
        isinstance(first_epoch, bool)
        or not isinstance(first_epoch, int)
        or first_epoch < 0
    ):
        raise Phase8B2ContractError("phase8b2.schedule.budget_invalid")
    epoch_size = steps_per_epoch * batch_size
    identities = tuple(
        identity
        for epoch in range(first_epoch, first_epoch + epochs)
        for identity in build_raw_downstream_epoch_schedule(
            dataset,
            weights=weights,
            seed=seed,
            epoch=epoch,
            epoch_size=epoch_size,
            batch_size=batch_size,
        ).identities
    )
    return RawDownstreamSampleSchedule(
        seed=seed,
        first_epoch=first_epoch,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        identities=identities,
        fingerprint=raw_downstream_sample_schedule_fingerprint(identities),
    )


def derive_seed(base_seed: int, domain: str, *coordinates: object) -> int:
    """Derive a stable nonnegative torch-compatible seed by named domain."""

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
        or not isinstance(domain, str)
        or not domain
    ):
        raise Phase8B2ContractError(
            "phase8b2.seed.domain_arguments_invalid"
        )
    payload = {
        "contract_version": SEED_DOMAIN_CONTRACT_VERSION,
        "base_seed": base_seed,
        "domain": domain,
        "coordinates": list(coordinates),
    }
    digest = sha256(
        fingerprint(payload).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@dataclass(frozen=True, slots=True)
class SeedDomains:
    """Independent domains shared by every paired comparison cell."""

    base_seed: int
    model_initialization: int
    ssl_data_order: int
    ssl_mask_planning: int
    downstream_initialization: int
    downstream_data_order: int
    bootstrap: int

    @classmethod
    def create(cls, base_seed: int) -> SeedDomains:
        return cls(
            base_seed=base_seed,
            model_initialization=derive_seed(base_seed, "model_initialization"),
            ssl_data_order=derive_seed(base_seed, "ssl_data_order"),
            ssl_mask_planning=derive_seed(base_seed, "ssl_mask_planning"),
            downstream_initialization=derive_seed(
                base_seed, "downstream_initialization"
            ),
            downstream_data_order=derive_seed(
                base_seed, "downstream_data_order"
            ),
            bootstrap=derive_seed(base_seed, "piece_bootstrap"),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "contract_version": SEED_DOMAIN_CONTRACT_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class PolicyView:
    index: int
    policy: str
    seed_domain: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VariantSchedule:
    """One variant's fixed per-logical-update view schedule."""

    contract_version: str
    comparison_mode: ComparisonMode
    variant_id: str
    objective_mode: str
    masking_mode: str
    logical_updates: int
    batch_size: int
    policy_views: tuple[PolicyView, ...]
    encoder_forwards_per_policy_view: int
    sample_identity_schedule: tuple[tuple[str, str], ...]
    sample_schedule_fingerprint: str
    fingerprint: str

    @property
    def policy_views_per_update(self) -> int:
        return len(self.policy_views)

    @property
    def policy_view_exposure_count(self) -> int:
        return self.logical_updates * self.policy_views_per_update

    @property
    def encoder_forward_count(self) -> int:
        return (
            self.policy_view_exposure_count
            * self.encoder_forwards_per_policy_view
        )

    @property
    def raw_sample_exposures(self) -> int:
        return len(self.sample_identity_schedule)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "comparison_mode": self.comparison_mode,
            "variant_id": self.variant_id,
            "objective_mode": self.objective_mode,
            "masking_mode": self.masking_mode,
            "logical_updates": self.logical_updates,
            "batch_size": self.batch_size,
            "policy_views": [view.to_dict() for view in self.policy_views],
            "policy_views_per_update": self.policy_views_per_update,
            "policy_view_exposure_count": self.policy_view_exposure_count,
            "encoder_forwards_per_policy_view": (
                self.encoder_forwards_per_policy_view
            ),
            "encoder_forward_count": self.encoder_forward_count,
            "raw_sample_exposures": self.raw_sample_exposures,
            "sample_schedule_fingerprint": (
                self.sample_schedule_fingerprint
            ),
            "fingerprint": self.fingerprint,
        }


def _views(
    variant_id: str,
    *,
    comparison_mode: ComparisonMode,
    matched_encoder_forwards_per_update: int,
    mask_seed: int,
) -> tuple[PolicyView, ...]:
    try:
        natural = _POLICIES[variant_id]
    except KeyError as exc:
        variant_modes(variant_id)
        raise AssertionError("unreachable") from exc
    if comparison_mode == "natural_schedule":
        policies = natural
    elif comparison_mode == "encoder_forward_matched":
        forwards_per_view = encoder_forwards_per_policy_view(variant_id)
        if (
            isinstance(matched_encoder_forwards_per_update, bool)
            or not isinstance(matched_encoder_forwards_per_update, int)
            or matched_encoder_forwards_per_update <= 0
            or matched_encoder_forwards_per_update % forwards_per_view != 0
            or matched_encoder_forwards_per_update // forwards_per_view
            < len(natural)
        ):
            raise Phase8B2ContractError(
                "phase8b2.schedule.compute_budget_unmatchable"
            )
        matched_views_per_update = (
            matched_encoder_forwards_per_update // forwards_per_view
        )
        policies = tuple(
            natural[index % len(natural)]
            for index in range(matched_views_per_update)
        )
    else:
        raise Phase8B2ContractError(
            "phase8b2.schedule.comparison_mode_invalid"
        )
    return tuple(
        PolicyView(
            index=index,
            policy=policy,
            seed_domain=f"phase8b2/mask/{variant_id}/view/{index}",
            seed=derive_seed(
                mask_seed,
                f"phase8b2/mask/{variant_id}/view/{index}",
            ),
        )
        for index, policy in enumerate(policies)
    )


def build_variant_schedule(
    variant_id: str,
    *,
    comparison_mode: ComparisonMode,
    logical_updates: int,
    batch_size: int,
    matched_encoder_forwards_per_update: int,
    sample_identity_schedule: Sequence[tuple[str, str]],
    mask_seed: int,
) -> VariantSchedule:
    """Build one schedule without consulting launch order or supervision."""

    if (
        isinstance(logical_updates, bool)
        or not isinstance(logical_updates, int)
        or logical_updates <= 0
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise Phase8B2ContractError(
            "phase8b2.schedule.budget_invalid"
        )
    identities = tuple(sample_identity_schedule)
    if len(identities) != logical_updates * batch_size:
        raise Phase8B2ContractError(
            "phase8b2.schedule.sample_exposure_count_mismatch"
        )
    if any(
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(piece_id, str)
        or not piece_id
        for dataset_id, piece_id in identities
    ):
        raise Phase8B2ContractError(
            "phase8b2.schedule.sample_identity_invalid"
        )
    objective_mode, masking_mode = variant_modes(variant_id)
    views = _views(
        variant_id,
        comparison_mode=comparison_mode,
        matched_encoder_forwards_per_update=(
            matched_encoder_forwards_per_update
        ),
        mask_seed=mask_seed,
    )
    sample_fingerprint = fingerprint(
        {
            "contract_version": SCHEDULE_CONTRACT_VERSION,
            "kind": "raw_ssl_sample_schedule",
            "identities": [list(row) for row in identities],
        }
    )
    payload = {
        "contract_version": SCHEDULE_CONTRACT_VERSION,
        "comparison_mode": comparison_mode,
        "variant_id": variant_id,
        "objective_mode": objective_mode,
        "masking_mode": masking_mode,
        "logical_updates": logical_updates,
        "batch_size": batch_size,
        "policy_views": [view.to_dict() for view in views],
        "encoder_forwards_per_policy_view": (
            encoder_forwards_per_policy_view(variant_id)
        ),
        "sample_schedule_fingerprint": sample_fingerprint,
    }
    return VariantSchedule(
        contract_version=SCHEDULE_CONTRACT_VERSION,
        comparison_mode=comparison_mode,
        variant_id=variant_id,
        objective_mode=objective_mode,
        masking_mode=masking_mode,
        logical_updates=logical_updates,
        batch_size=batch_size,
        policy_views=views,
        encoder_forwards_per_policy_view=(
            encoder_forwards_per_policy_view(variant_id)
        ),
        sample_identity_schedule=identities,
        sample_schedule_fingerprint=sample_fingerprint,
        fingerprint=fingerprint(payload),
    )


def validate_paired_schedules(
    schedules: Iterable[VariantSchedule],
) -> dict[str, object]:
    """Fail closed unless primary budgets and paired samples are compatible."""

    rows = tuple(schedules)
    if not rows:
        raise Phase8B2ContractError(
            "phase8b2.schedule.matrix_empty"
        )
    if len({row.variant_id for row in rows}) != len(rows):
        raise Phase8B2ContractError(
            "phase8b2.schedule.duplicate_variant"
        )
    fields = {
        "comparison_mode": {row.comparison_mode for row in rows},
        "logical_updates": {row.logical_updates for row in rows},
        "batch_size": {row.batch_size for row in rows},
        "raw_sample_exposures": {row.raw_sample_exposures for row in rows},
        "sample_schedule_fingerprint": {
            row.sample_schedule_fingerprint for row in rows
        },
    }
    mismatched = sorted(name for name, values in fields.items() if len(values) != 1)
    if mismatched:
        raise Phase8B2ContractError(
            "phase8b2.schedule.paired_binding_mismatch:"
            + ",".join(mismatched)
        )
    mode = rows[0].comparison_mode
    matched = mode == "encoder_forward_matched"
    forward_counts = {row.encoder_forward_count for row in rows}
    if matched and len(forward_counts) != 1:
        raise Phase8B2ContractError(
            "phase8b2.schedule.encoder_forwards_not_matched"
        )
    return {
        "contract_version": SCHEDULE_CONTRACT_VERSION,
        "comparison_mode": mode,
        "primary_compute_matched": matched,
        "variant_count": len(rows),
        "logical_updates": rows[0].logical_updates,
        "raw_sample_exposures": rows[0].raw_sample_exposures,
        "sample_schedule_fingerprint": rows[0].sample_schedule_fingerprint,
        "encoder_forward_counts": {
            row.variant_id: row.encoder_forward_count for row in rows
        },
        "policy_view_counts": {
            row.variant_id: row.policy_view_exposure_count for row in rows
        },
    }


def natural_view_count(variant_id: str) -> int:
    """Cross-check the public contract against the concrete policy table."""

    value = len(_POLICIES[variant_id])
    if value != natural_policy_views(variant_id):
        raise RuntimeError("phase8b2.schedule.internal_policy_count_mismatch")
    return value


__all__ = [
    "RawDownstreamEpochSchedule",
    "RawDownstreamSampleSchedule",
    "SCHEDULE_CONTRACT_VERSION",
    "SEED_DOMAIN_CONTRACT_VERSION",
    "PolicyView",
    "SeedDomains",
    "VariantSchedule",
    "build_raw_downstream_epoch_schedule",
    "build_raw_downstream_sample_schedule",
    "build_variant_schedule",
    "derive_seed",
    "natural_view_count",
    "raw_downstream_sample_schedule_fingerprint",
    "validate_paired_schedules",
]
