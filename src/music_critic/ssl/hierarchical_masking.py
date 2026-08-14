"""Deterministic, target-blind Phase 8A hierarchy-aware mask planning.

The planners in this module only resolve raw containment and ownership
relations.  They do not inspect feature values, targets, annotations,
provenance, diagnostics, or global RNG state.  The existing Phase 7A
independent-note planner is used directly for the control policy so its
portable plan and fingerprint remain bit-exact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Literal

from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from music_critic.graph import validate_raw_graph, validate_raw_graph_batch
from music_critic.ssl.contracts import (
    CollateralFeatureMask,
    MaskPlan,
    MaskStage,
    SSLContractError,
    SampleIdentity,
    StableSeed,
    canonical_sha256,
    is_sha256,
    mask_plan_fingerprint,
    validate_global_seed,
    validate_mask_rate,
    validate_non_negative_integer,
)
from music_critic.ssl.field_registry import (
    MASKABLE_FIELD_REGISTRY_FINGERPRINT,
    MASKABLE_FIELD_REGISTRY_VERSION,
    NOTE_PITCH_GROUP,
    NOTE_PITCH_GROUP_NAME,
    SSL_MASKABLE_FIELD_REGISTRY,
)
from music_critic.ssl.masking import (
    build_batched_mask_plans,
    build_mask_plan,
    derive_stable_seed,
)


HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION = "1.2.0"
HIERARCHY_MASK_POLICY_VERSION = "1.2.0"
HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION = "1.2.0"
HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION = "1.0.0"
HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION = "1.2.0"
HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION = "1.0.0"
HIERARCHY_PREPARED_BINDING_PROFILE_VERSION = "1.2.0"
MAX_SPAN_BARS = 8
MAX_SPAN_SELECTION_POOL_SIZE = 8
MAX_SPAN_BUDGET_ERROR_SLACK = 8
DEFAULT_SPAN_SELECTION_POOL_SIZE = 4
DEFAULT_SPAN_BUDGET_ERROR_SLACK = 1
SPAN_SELECTION_METHOD = "bounded_near_optimal_seed_rank_v2"
SPAN_POOL_MEMBERSHIP_RANK_METHOD = (
    "stable_seed_sha256_pool_membership_v1"
)
SPAN_FINAL_CHOICE_RANK_METHOD = (
    "stable_seed_sha256_final_choice_v1"
)

INDEPENDENT_NOTE_PITCH = "independent_note_pitch"
ONSET_PITCH_DESCENDANTS = "onset_pitch_descendants"
BEAT_PITCH_DESCENDANTS = "beat_pitch_descendants"
CONTIGUOUS_BAR_PITCH_SPAN = "contiguous_bar_pitch_span"
TRACK_BAR_PITCH_SPAN = "track_bar_pitch_span"

HierarchyMaskPolicy = Literal[
    "independent_note_pitch",
    "onset_pitch_descendants",
    "beat_pitch_descendants",
    "contiguous_bar_pitch_span",
    "track_bar_pitch_span",
]

HIERARCHY_MASK_POLICIES: tuple[HierarchyMaskPolicy, ...] = (
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)

HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT = canonical_sha256(
    {
        "hierarchical_mask_plan_contract_version": (
            HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
        ),
        "hierarchy_mask_policy_version": (
            HIERARCHY_MASK_POLICY_VERSION
        ),
        "policy_config_contract_version": (
            HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
        ),
        "policy_mixture_contract_version": (
            HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
        ),
        "selection_evidence_contract_version": (
            HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION
        ),
        "unavailable_reason_contract_version": (
            HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION
        ),
        "prepared_binding_profile_version": (
            HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
        ),
        "policies": list(HIERARCHY_MASK_POLICIES),
        "descendant_paths": {
            ONSET_PITCH_DESCENDANTS: [
                "onset|starts_note|note",
            ],
            BEAT_PITCH_DESCENDANTS: [
                "beat|contains_onset|onset",
                "onset|starts_note|note",
            ],
            CONTIGUOUS_BAR_PITCH_SPAN: [
                "bar|contains_onset|onset",
                "onset|starts_note|note",
            ],
            TRACK_BAR_PITCH_SPAN: [
                "track|contains_note|note",
                "bar|contains_onset|onset",
                "onset|starts_note|note",
            ],
        },
        "descendant_semantics": "start_anchored_v1",
        "primary_feature_group": NOTE_PITCH_GROUP_NAME,
        "max_span_bars": MAX_SPAN_BARS,
        "unit_order": "splitmix64_fisher_yates_v1",
        "budget_crossing": "closest_valid_before_or_after_v1",
        "span_selection": SPAN_SELECTION_METHOD,
        "span_pool_membership_rank": (
            SPAN_POOL_MEMBERSHIP_RANK_METHOD
        ),
        "span_final_choice_rank": SPAN_FINAL_CHOICE_RANK_METHOD,
        "span_rank_collision_fallback": (
            "track_start_end_descendants_v1"
        ),
        "max_span_selection_pool_size": MAX_SPAN_SELECTION_POOL_SIZE,
        "max_span_budget_error_slack": MAX_SPAN_BUDGET_ERROR_SLACK,
        "default_span_selection_pool_size": (
            DEFAULT_SPAN_SELECTION_POOL_SIZE
        ),
        "default_span_budget_error_slack": (
            DEFAULT_SPAN_BUDGET_ERROR_SLACK
        ),
        "mixture_resolution": "stable_seed_weighted_cumulative_v1",
    }
)

_POLICY_UNIT_NODE_TYPE: dict[str, str] = {
    INDEPENDENT_NOTE_PITCH: "note",
    ONSET_PITCH_DESCENDANTS: "onset",
    BEAT_PITCH_DESCENDANTS: "beat",
    CONTIGUOUS_BAR_PITCH_SPAN: "bar",
    TRACK_BAR_PITCH_SPAN: "bar",
}

_ONSET_STARTS_NOTE = ("onset", "starts_note", "note")
_BEAT_CONTAINS_ONSET = ("beat", "contains_onset", "onset")
_BAR_CONTAINS_BEAT = ("bar", "contains_beat", "beat")
_BAR_CONTAINS_ONSET = ("bar", "contains_onset", "onset")
_BAR_CONTAINS_NOTE = ("bar", "contains_note", "note")
_TRACK_CONTAINS_NOTE = ("track", "contains_note", "note")
_BAR_NEXT = ("bar", "next_bar", "bar")

_UNAVAILABLE_CODES = {
    "policy_disabled",
    "zero_requested_mask_rate",
    "fewer_than_two_pitched_notes",
    "would_mask_all_pitched_notes",
    "no_nonempty_hierarchy_units",
    "no_valid_unit_selection",
    "no_valid_span",
    "no_valid_track_span",
}


class HierarchyMaskContractError(SSLContractError):
    """Structured Phase 8A hierarchy-contract failure."""


class HierarchyMaskUnavailableError(HierarchyMaskContractError):
    """Raised when a requested prepared mixture has no available plan."""

    def __init__(
        self,
        resolutions: Sequence[HierarchyMaskResolution],
    ) -> None:
        self.resolutions = tuple(resolutions)
        unavailable = [
            {
                "sample_identity": list(resolution.sample_identity),
                "eligibility": [
                    eligibility.to_dict()
                    for eligibility in resolution.eligibility
                ],
            }
            for resolution in self.resolutions
            if resolution.plan is None
        ]
        super().__init__(
            "phase8a.hierarchy.no_available_policy:"
            + canonical_sha256(unavailable)
        )


def _canonical_epoch(stage: MaskStage, epoch: int) -> int:
    validate_non_negative_integer(epoch, name="epoch")
    if stage not in {"train", "validation"}:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.stage_invalid"
        )
    return epoch if stage == "train" else 0


def _validate_policy(value: object) -> HierarchyMaskPolicy:
    if value not in HIERARCHY_MASK_POLICIES:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.policy_unknown"
        )
    return value  # type: ignore[return-value]


def _indices(
    values: object,
    *,
    name: str,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise HierarchyMaskContractError(
            f"phase8a.hierarchy.{name}_not_tuple"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in values
    ):
        raise HierarchyMaskContractError(
            f"phase8a.hierarchy.{name}_invalid"
        )
    if any(left >= right for left, right in zip(values, values[1:])):
        raise HierarchyMaskContractError(
            f"phase8a.hierarchy.{name}_not_unique_sorted"
        )
    if upper_bound is not None and any(
        value >= upper_bound for value in values
    ):
        raise HierarchyMaskContractError(
            f"phase8a.hierarchy.{name}_out_of_range"
        )
    return values


@dataclass(frozen=True, slots=True)
class HierarchyMaskUnavailableReason:
    """Portable, structured explanation for one ineligible policy."""

    contract_version: str
    policy: HierarchyMaskPolicy
    code: str
    candidate_count: int
    pitched_note_count: int
    requested_hidden_note_count: int

    @classmethod
    def create(
        cls,
        *,
        policy: HierarchyMaskPolicy,
        code: str,
        candidate_count: int,
        pitched_note_count: int,
        requested_hidden_note_count: int,
    ) -> HierarchyMaskUnavailableReason:
        return cls(
            contract_version=(
                HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION
            ),
            policy=policy,
            code=code,
            candidate_count=candidate_count,
            pitched_note_count=pitched_note_count,
            requested_hidden_note_count=requested_hidden_note_count,
        )

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.unavailable_version_incompatible"
            )
        _validate_policy(self.policy)
        if self.code not in _UNAVAILABLE_CODES:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.unavailable_code_invalid"
            )
        for name, value in (
            ("candidate_count", self.candidate_count),
            ("pitched_note_count", self.pitched_note_count),
            (
                "requested_hidden_note_count",
                self.requested_hidden_note_count,
            ),
        ):
            validate_non_negative_integer(value, name=name)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "policy": self.policy,
            "code": self.code,
            "candidate_count": self.candidate_count,
            "pitched_note_count": self.pitched_note_count,
            "requested_hidden_note_count": (
                self.requested_hidden_note_count
            ),
        }


def _policy_config_payload(
    config: HierarchyMaskPolicyConfig,
) -> dict[str, object]:
    return {
        "contract_version": config.contract_version,
        "policy_version": config.policy_version,
        "policy_weights": [
            {
                "policy": policy,
                "weight": weight,
            }
            for policy, weight in config.policy_weights
        ],
        "min_span_bars": config.min_span_bars,
        "max_span_bars": config.max_span_bars,
        "span_selection_pool_size": (
            config.span_selection_pool_size
        ),
        "span_budget_error_slack": (
            config.span_budget_error_slack
        ),
        "renormalization": (
            "positive_weights_over_piece_eligible_policies"
        ),
        "resolution": "stable_seed_weighted_cumulative_v1",
    }


def hierarchy_policy_config_fingerprint(
    config: HierarchyMaskPolicyConfig,
) -> str:
    """Recompute configuration evidence without mutating the object."""

    validate_hierarchy_policy_config(config)
    return canonical_sha256(_policy_config_payload(config))


def validate_hierarchy_policy_config(
    config: object,
) -> HierarchyMaskPolicyConfig:
    """Return an exact canonical configuration or fail closed."""

    if type(config) is not HierarchyMaskPolicyConfig:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.policy_config_type_invalid"
        )
    canonical = HierarchyMaskPolicyConfig(
        contract_version=config.contract_version,
        policy_version=config.policy_version,
        policy_weights=config.policy_weights,
        min_span_bars=config.min_span_bars,
        max_span_bars=config.max_span_bars,
        span_selection_pool_size=config.span_selection_pool_size,
        span_budget_error_slack=config.span_budget_error_slack,
    )
    if config != canonical:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.policy_config_non_canonical"
        )
    return config


@dataclass(frozen=True, slots=True)
class HierarchyMaskPolicyConfig:
    """Versioned enabling, weighting, and bounded-span configuration."""

    contract_version: str = HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
    policy_version: str = HIERARCHY_MASK_POLICY_VERSION
    policy_weights: tuple[tuple[str, float], ...] = (
        (INDEPENDENT_NOTE_PITCH, 1.0),
        (ONSET_PITCH_DESCENDANTS, 1.0),
        (BEAT_PITCH_DESCENDANTS, 1.0),
        (CONTIGUOUS_BAR_PITCH_SPAN, 1.0),
        (TRACK_BAR_PITCH_SPAN, 1.0),
    )
    min_span_bars: int = 1
    max_span_bars: int = 2
    span_selection_pool_size: int = DEFAULT_SPAN_SELECTION_POOL_SIZE
    span_budget_error_slack: int = DEFAULT_SPAN_BUDGET_ERROR_SLACK
    fingerprint: str = field(init=False)

    @classmethod
    def create(
        cls,
        *,
        weights: Mapping[str, float],
        min_span_bars: int = 1,
        max_span_bars: int = 2,
        span_selection_pool_size: int = (
            DEFAULT_SPAN_SELECTION_POOL_SIZE
        ),
        span_budget_error_slack: int = (
            DEFAULT_SPAN_BUDGET_ERROR_SLACK
        ),
    ) -> HierarchyMaskPolicyConfig:
        if not isinstance(weights, Mapping):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weights_not_mapping"
            )
        unknown = set(weights) - set(HIERARCHY_MASK_POLICIES)
        if unknown:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weights_unknown_policy"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in weights.values()
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weight_invalid"
            )
        return cls(
            policy_weights=tuple(
                (policy, float(weights.get(policy, 0.0)))
                for policy in HIERARCHY_MASK_POLICIES
            ),
            min_span_bars=min_span_bars,
            max_span_bars=max_span_bars,
            span_selection_pool_size=span_selection_pool_size,
            span_budget_error_slack=span_budget_error_slack,
        )

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
            or self.policy_version != HIERARCHY_MASK_POLICY_VERSION
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_config_version_incompatible"
            )
        if (
            type(self.policy_weights) is not tuple
            or not all(
                type(item) is tuple and len(item) == 2
                for item in self.policy_weights
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weights_not_immutable_tuple"
            )
        if tuple(
            policy for policy, _ in self.policy_weights
        ) != HIERARCHY_MASK_POLICIES:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weights_order_invalid"
            )
        for policy, weight in self.policy_weights:
            _validate_policy(policy)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, float)
                or not math.isfinite(weight)
                or weight < 0.0
                or (
                    weight == 0.0
                    and math.copysign(1.0, weight) < 0.0
                )
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.policy_weight_invalid"
                )
        if not any(weight > 0.0 for _, weight in self.policy_weights):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.all_policies_disabled"
            )
        positive_weights = tuple(
            Fraction(weight)
            for _, weight in self.policy_weights
            if weight > 0.0
        )
        positive_total = sum(
            positive_weights,
            start=Fraction(0),
        )
        if any(
            float(weight / positive_total) <= 0.0
            for weight in positive_weights
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.policy_weight_not_normalizable"
            )
        for name, value in (
            ("min_span_bars", self.min_span_bars),
            ("max_span_bars", self.max_span_bars),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise HierarchyMaskContractError(
                    f"phase8a.hierarchy.{name}_invalid"
                )
        if self.min_span_bars > self.max_span_bars:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.span_bounds_reversed"
            )
        if self.max_span_bars > MAX_SPAN_BARS:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.max_span_exceeds_contract_bound"
            )
        if (
            isinstance(self.span_selection_pool_size, bool)
            or not isinstance(self.span_selection_pool_size, int)
            or self.span_selection_pool_size <= 0
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.span_selection_pool_size_invalid"
            )
        if (
            self.span_selection_pool_size
            > MAX_SPAN_SELECTION_POOL_SIZE
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy."
                "span_selection_pool_size_exceeds_contract_bound"
            )
        if (
            isinstance(self.span_budget_error_slack, bool)
            or not isinstance(self.span_budget_error_slack, int)
            or self.span_budget_error_slack < 0
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.span_budget_error_slack_invalid"
            )
        if (
            self.span_budget_error_slack
            > MAX_SPAN_BUDGET_ERROR_SLACK
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy."
                "span_budget_error_slack_exceeds_contract_bound"
            )
        payload = _policy_config_payload(self)
        object.__setattr__(self, "fingerprint", canonical_sha256(payload))

    def weight(self, policy: HierarchyMaskPolicy) -> float:
        _validate_policy(policy)
        return dict(self.policy_weights)[policy]

    def enabled_policies(self) -> tuple[HierarchyMaskPolicy, ...]:
        return tuple(
            policy
            for policy, weight in self.policy_weights
            if weight > 0.0
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        payload = _policy_config_payload(self)
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SelectedHierarchyUnits:
    """Selected units and exact start-anchored descendant evidence."""

    contract_version: str
    policy: HierarchyMaskPolicy
    unit_node_type: str
    selected_local_unit_indices: tuple[int, ...]
    span_start_bar_index: int | None
    span_end_bar_index: int | None
    span_length_bars: int | None
    selected_local_track_index: int | None
    selected_local_note_descendants: tuple[int, ...]
    total_valid_candidate_count: int
    span_best_budget_error: int | None
    span_tolerance_candidate_count: int | None
    span_admissible_pool_count: int | None
    span_configured_pool_size_limit: int | None
    span_configured_budget_error_slack: int | None
    span_selected_budget_error: int | None
    span_selected_descendant_count: int | None
    span_realized_mask_rate: float | None
    span_selection_method: str | None
    span_pool_membership_rank_method: str | None
    span_final_choice_rank_method: str | None

    @classmethod
    def create(
        cls,
        *,
        policy: HierarchyMaskPolicy,
        selected_local_unit_indices: tuple[int, ...],
        selected_local_note_descendants: tuple[int, ...],
        total_valid_candidate_count: int,
        span_start_bar_index: int | None = None,
        span_end_bar_index: int | None = None,
        span_length_bars: int | None = None,
        selected_local_track_index: int | None = None,
        span_best_budget_error: int | None = None,
        span_tolerance_candidate_count: int | None = None,
        span_admissible_pool_count: int | None = None,
        span_configured_pool_size_limit: int | None = None,
        span_configured_budget_error_slack: int | None = None,
        span_selected_budget_error: int | None = None,
        span_selected_descendant_count: int | None = None,
        span_realized_mask_rate: float | None = None,
        span_selection_method: str | None = None,
        span_pool_membership_rank_method: str | None = None,
        span_final_choice_rank_method: str | None = None,
    ) -> SelectedHierarchyUnits:
        resolved_policy = _validate_policy(policy)
        return cls(
            contract_version=(
                HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION
            ),
            policy=resolved_policy,
            unit_node_type=_POLICY_UNIT_NODE_TYPE[resolved_policy],
            selected_local_unit_indices=(
                selected_local_unit_indices
            ),
            span_start_bar_index=span_start_bar_index,
            span_end_bar_index=span_end_bar_index,
            span_length_bars=span_length_bars,
            selected_local_track_index=selected_local_track_index,
            selected_local_note_descendants=(
                selected_local_note_descendants
            ),
            total_valid_candidate_count=total_valid_candidate_count,
            span_best_budget_error=span_best_budget_error,
            span_tolerance_candidate_count=(
                span_tolerance_candidate_count
            ),
            span_admissible_pool_count=(
                span_admissible_pool_count
            ),
            span_configured_pool_size_limit=(
                span_configured_pool_size_limit
            ),
            span_configured_budget_error_slack=(
                span_configured_budget_error_slack
            ),
            span_selected_budget_error=(
                span_selected_budget_error
            ),
            span_selected_descendant_count=(
                span_selected_descendant_count
            ),
            span_realized_mask_rate=span_realized_mask_rate,
            span_selection_method=span_selection_method,
            span_pool_membership_rank_method=(
                span_pool_membership_rank_method
            ),
            span_final_choice_rank_method=(
                span_final_choice_rank_method
            ),
        )

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.selection_version_incompatible"
            )
        policy = _validate_policy(self.policy)
        if self.unit_node_type != _POLICY_UNIT_NODE_TYPE[policy]:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.selection_unit_type_invalid"
            )
        _indices(
            self.selected_local_unit_indices,
            name="selected_unit_indices",
        )
        _indices(
            self.selected_local_note_descendants,
            name="selected_note_descendants",
        )
        validate_non_negative_integer(
            self.total_valid_candidate_count,
            name="total_valid_candidate_count",
        )
        if bool(self.selected_local_unit_indices) != bool(
            self.selected_local_note_descendants
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.selection_units_descendants_mismatch"
            )
        if self.selected_local_unit_indices:
            if self.total_valid_candidate_count < 1:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_candidate_count_empty"
                )
            if (
                policy
                in {
                    ONSET_PITCH_DESCENDANTS,
                    BEAT_PITCH_DESCENDANTS,
                }
                and self.total_valid_candidate_count
                < len(self.selected_local_unit_indices)
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_candidate_count_small"
                )
        is_span = policy in {
            CONTIGUOUS_BAR_PITCH_SPAN,
            TRACK_BAR_PITCH_SPAN,
        }
        span_values = (
            self.span_start_bar_index,
            self.span_end_bar_index,
            self.span_length_bars,
        )
        empty_unavailable_evidence = (
            not self.selected_local_unit_indices
            and not self.selected_local_note_descendants
            and all(value is None for value in span_values)
            and self.selected_local_track_index is None
        )
        span_selection_values = (
            self.span_best_budget_error,
            self.span_tolerance_candidate_count,
            self.span_admissible_pool_count,
            self.span_configured_pool_size_limit,
            self.span_configured_budget_error_slack,
            self.span_selected_budget_error,
            self.span_selected_descendant_count,
            self.span_realized_mask_rate,
            self.span_selection_method,
            self.span_pool_membership_rank_method,
            self.span_final_choice_rank_method,
        )
        if is_span:
            for name, value, lower, upper in (
                (
                    "span_configured_pool_size_limit",
                    self.span_configured_pool_size_limit,
                    1,
                    MAX_SPAN_SELECTION_POOL_SIZE,
                ),
                (
                    "span_configured_budget_error_slack",
                    self.span_configured_budget_error_slack,
                    0,
                    MAX_SPAN_BUDGET_ERROR_SLACK,
                ),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not lower <= value <= upper
                ):
                    raise HierarchyMaskContractError(
                        f"phase8a.hierarchy.selection_{name}_invalid"
                    )
            if self.span_selection_method != SPAN_SELECTION_METHOD:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy."
                    "selection_span_method_incompatible"
                )
            if (
                self.span_pool_membership_rank_method
                != SPAN_POOL_MEMBERSHIP_RANK_METHOD
                or self.span_final_choice_rank_method
                != SPAN_FINAL_CHOICE_RANK_METHOD
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy."
                    "selection_span_rank_method_incompatible"
                )
            for name, value in (
                (
                    "span_tolerance_candidate_count",
                    self.span_tolerance_candidate_count,
                ),
                (
                    "span_admissible_pool_count",
                    self.span_admissible_pool_count,
                ),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise HierarchyMaskContractError(
                        f"phase8a.hierarchy.selection_{name}_invalid"
                    )
            tolerance_count = self.span_tolerance_candidate_count
            pool_count = self.span_admissible_pool_count
            pool_limit = self.span_configured_pool_size_limit
            assert tolerance_count is not None
            assert pool_count is not None
            assert pool_limit is not None
            if (
                tolerance_count > self.total_valid_candidate_count
                or pool_count != min(tolerance_count, pool_limit)
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_span_pool_count_invalid"
                )
            if self.total_valid_candidate_count == 0:
                if (
                    self.span_best_budget_error is not None
                    or tolerance_count != 0
                    or pool_count != 0
                ):
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy."
                        "selection_empty_span_pool_invalid"
                    )
            elif (
                isinstance(self.span_best_budget_error, bool)
                or not isinstance(self.span_best_budget_error, int)
                or self.span_best_budget_error < 0
                or tolerance_count < 1
                or pool_count < 1
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_span_best_error_invalid"
                )
        elif any(value is not None for value in span_selection_values):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.non_span_has_selection_evidence"
            )
        if is_span and not empty_unavailable_evidence:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in span_values
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_span_invalid"
                )
            start = self.span_start_bar_index
            end = self.span_end_bar_index
            length = self.span_length_bars
            assert start is not None and end is not None
            assert length is not None
            if end < start or length != end - start + 1:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_span_not_contiguous"
                )
            if self.selected_local_unit_indices != tuple(
                range(start, end + 1)
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.selection_span_units_invalid"
                )
            best_error = self.span_best_budget_error
            selected_error = self.span_selected_budget_error
            selected_descendant_count = (
                self.span_selected_descendant_count
            )
            slack = self.span_configured_budget_error_slack
            realized_rate = self.span_realized_mask_rate
            assert best_error is not None
            assert slack is not None
            if (
                isinstance(selected_error, bool)
                or not isinstance(selected_error, int)
                or selected_error < best_error
                or selected_error > best_error + slack
                or isinstance(selected_descendant_count, bool)
                or not isinstance(selected_descendant_count, int)
                or selected_descendant_count
                != len(self.selected_local_note_descendants)
                or isinstance(realized_rate, bool)
                or not isinstance(realized_rate, float)
                or not math.isfinite(realized_rate)
                or not 0.0 < realized_rate < 1.0
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy."
                    "selection_selected_span_evidence_invalid"
                )
        elif any(value is not None for value in span_values):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.non_span_has_span_evidence"
            )
        if is_span and empty_unavailable_evidence and any(
            value is not None
            for value in (
                self.span_selected_budget_error,
                self.span_selected_descendant_count,
                self.span_realized_mask_rate,
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy."
                "selection_unavailable_has_selected_span_evidence"
            )
        if (
            policy == TRACK_BAR_PITCH_SPAN
            and not empty_unavailable_evidence
        ):
            if (
                isinstance(self.selected_local_track_index, bool)
                or not isinstance(
                    self.selected_local_track_index,
                    int,
                )
                or self.selected_local_track_index < 0
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.track_span_owner_invalid"
                )
        elif self.selected_local_track_index is not None:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.non_track_span_has_owner"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "policy": self.policy,
            "unit_node_type": self.unit_node_type,
            "selected_local_unit_indices": list(
                self.selected_local_unit_indices
            ),
            "span_start_bar_index": self.span_start_bar_index,
            "span_end_bar_index": self.span_end_bar_index,
            "span_length_bars": self.span_length_bars,
            "selected_local_track_index": (
                self.selected_local_track_index
            ),
            "selected_local_note_descendants": list(
                self.selected_local_note_descendants
            ),
            "total_valid_candidate_count": (
                self.total_valid_candidate_count
            ),
            "span_best_budget_error": self.span_best_budget_error,
            "span_tolerance_candidate_count": (
                self.span_tolerance_candidate_count
            ),
            "span_admissible_pool_count": (
                self.span_admissible_pool_count
            ),
            "span_configured_pool_size_limit": (
                self.span_configured_pool_size_limit
            ),
            "span_configured_budget_error_slack": (
                self.span_configured_budget_error_slack
            ),
            "span_selected_budget_error": (
                self.span_selected_budget_error
            ),
            "span_selected_descendant_count": (
                self.span_selected_descendant_count
            ),
            "span_realized_mask_rate": (
                self.span_realized_mask_rate
            ),
            "span_selection_method": self.span_selection_method,
            "span_pool_membership_rank_method": (
                self.span_pool_membership_rank_method
            ),
            "span_final_choice_rank_method": (
                self.span_final_choice_rank_method
            ),
            "span_selected_candidate_canonical_identity": (
                None
                if self.span_start_bar_index is None
                else {
                    "start": self.span_start_bar_index,
                    "end": self.span_end_bar_index,
                    "track": self.selected_local_track_index,
                    "descendants": list(
                        self.selected_local_note_descendants
                    ),
                }
            ),
            "descendant_semantics": "start_anchored_v1",
        }

    @property
    def candidate_count(self) -> int:
        """Compatibility alias for internal callers; not serialized."""

        return self.total_valid_candidate_count


def _hierarchical_plan_payload(
    plan: HierarchicalMaskPlan,
) -> dict[str, object]:
    return {
        "contract_version": plan.contract_version,
        "policy_version": plan.policy_version,
        "sample_identity": {
            "dataset_id": plan.dataset_id,
            "piece_id": plan.piece_id,
        },
        "stage": plan.stage,
        "epoch": plan.epoch,
        "encoder_view_index": plan.encoder_view_index,
        "global_seed": plan.global_seed,
        "stable_seed": plan.stable_seed,
        "stable_seed_sha256": plan.stable_seed_sha256,
        "requested_mask_rate": plan.requested_mask_rate,
        "requested_hidden_note_count": (
            plan.requested_hidden_note_count
        ),
        "resolved_policy": plan.resolved_policy,
        "policy_configuration_fingerprint": (
            plan.policy_configuration_fingerprint
        ),
        "policy_configuration": (
            plan.policy_configuration.to_dict()
        ),
        "relevant_structure_fingerprint": (
            plan.relevant_structure_fingerprint
        ),
        "selection": plan.selection.to_dict(),
        "primary_feature_group": plan.primary_feature_group,
        "collateral_feature_masks": [
            mask.to_dict() for mask in plan.collateral_feature_masks
        ],
        "pitched_note_count": plan.pitched_note_count,
        "primary_masked_count": plan.primary_masked_count,
        "visible_pitched_note_count": plan.visible_pitched_note_count,
        "realized_mask_rate": plan.realized_mask_rate,
        "collateral_peer_note_count": (
            plan.collateral_peer_note_count
        ),
        "collateral_owner_track_count": (
            plan.collateral_owner_track_count
        ),
        "available": plan.available,
        "unavailable_reason": (
            None
            if plan.unavailable_reason is None
            else plan.unavailable_reason.to_dict()
        ),
    }


@dataclass(frozen=True, slots=True)
class HierarchicalMaskPlan:
    """Portable Phase 8A plan for one hierarchy-aware pitch-only view."""

    contract_version: str
    policy_version: str
    dataset_id: str
    piece_id: str
    stage: MaskStage
    epoch: int
    encoder_view_index: int
    global_seed: int
    stable_seed: int
    stable_seed_sha256: str
    requested_mask_rate: float
    requested_hidden_note_count: int
    resolved_policy: HierarchyMaskPolicy
    policy_configuration_fingerprint: str
    policy_configuration: HierarchyMaskPolicyConfig = field(
        repr=False,
        compare=True,
    )
    relevant_structure_fingerprint: str
    selection: SelectedHierarchyUnits
    primary_feature_group: str
    collateral_feature_masks: tuple[CollateralFeatureMask, ...]
    pitched_note_count: int
    primary_masked_count: int
    visible_pitched_note_count: int
    realized_mask_rate: float
    collateral_peer_note_count: int
    collateral_owner_track_count: int
    available: bool
    unavailable_reason: HierarchyMaskUnavailableReason | None
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        piece_id: str,
        stage: MaskStage,
        epoch: int,
        encoder_view_index: int,
        global_seed: int,
        stable_seed: StableSeed,
        requested_mask_rate: float,
        requested_hidden_note_count: int,
        resolved_policy: HierarchyMaskPolicy,
        policy_configuration: HierarchyMaskPolicyConfig,
        relevant_structure_fingerprint: str,
        selection: SelectedHierarchyUnits,
        collateral_feature_masks: tuple[
            CollateralFeatureMask, ...
        ],
        pitched_note_count: int,
        available: bool,
        unavailable_reason: (
            HierarchyMaskUnavailableReason | None
        ) = None,
    ) -> HierarchicalMaskPlan:
        resolved_policy = _validate_policy(resolved_policy)
        config = validate_hierarchy_policy_config(
            policy_configuration
        )
        if type(selection) is not SelectedHierarchyUnits:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_selection_type_invalid"
            )
        if (
            type(collateral_feature_masks) is not tuple
            or not all(
                type(mask) is CollateralFeatureMask
                for mask in collateral_feature_masks
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_collateral_type_invalid"
            )
        if type(available) is not bool:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_availability_type_invalid"
            )
        if (
            unavailable_reason is not None
            and type(unavailable_reason)
            is not HierarchyMaskUnavailableReason
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_unavailable_reason_type_invalid"
            )
        primary_count = len(
            selection.selected_local_note_descendants
        )
        visible_count = pitched_note_count - primary_count
        peer_count = 0
        track_count = 0
        for mask in collateral_feature_masks:
            if mask.node_type == "note":
                peer_count += len(mask.local_node_indices)
            elif mask.node_type == "track":
                track_count += len(mask.local_node_indices)
        values = {
            "contract_version": (
                HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
            ),
            "policy_version": HIERARCHY_MASK_POLICY_VERSION,
            "dataset_id": dataset_id,
            "piece_id": piece_id,
            "stage": stage,
            "epoch": epoch,
            "encoder_view_index": encoder_view_index,
            "global_seed": global_seed,
            "stable_seed": stable_seed.value,
            "stable_seed_sha256": stable_seed.sha256,
            "requested_mask_rate": requested_mask_rate,
            "requested_hidden_note_count": (
                requested_hidden_note_count
            ),
            "resolved_policy": resolved_policy,
            "policy_configuration_fingerprint": (
                config.fingerprint
            ),
            "policy_configuration": config,
            "relevant_structure_fingerprint": (
                relevant_structure_fingerprint
            ),
            "selection": selection,
            "primary_feature_group": NOTE_PITCH_GROUP_NAME,
            "collateral_feature_masks": collateral_feature_masks,
            "pitched_note_count": pitched_note_count,
            "primary_masked_count": primary_count,
            "visible_pitched_note_count": visible_count,
            "realized_mask_rate": (
                primary_count / pitched_note_count
                if pitched_note_count
                else 0.0
            ),
            "collateral_peer_note_count": peer_count,
            "collateral_owner_track_count": track_count,
            "available": available,
            "unavailable_reason": unavailable_reason,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "fingerprint", "0" * 64)
        return cls(
            **values,
            fingerprint=canonical_sha256(
                _hierarchical_plan_payload(provisional)
            ),
        )

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
            or self.policy_version != HIERARCHY_MASK_POLICY_VERSION
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_version_incompatible"
            )
        if (
            type(self.selection) is not SelectedHierarchyUnits
            or type(self.collateral_feature_masks) is not tuple
            or not all(
                type(mask) is CollateralFeatureMask
                for mask in self.collateral_feature_masks
            )
            or type(self.available) is not bool
            or (
                self.unavailable_reason is not None
                and type(self.unavailable_reason)
                is not HierarchyMaskUnavailableReason
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_portable_type_invalid"
            )
        SampleIdentity(self.dataset_id, self.piece_id)
        canonical_epoch = _canonical_epoch(self.stage, self.epoch)
        if canonical_epoch != self.epoch:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.validation_epoch_not_canonical"
            )
        validate_non_negative_integer(
            self.encoder_view_index,
            name="encoder_view_index",
        )
        validate_global_seed(self.global_seed)
        StableSeed(self.stable_seed, self.stable_seed_sha256)
        rate = validate_mask_rate(self.requested_mask_rate)
        if (
            not isinstance(self.requested_mask_rate, float)
            or rate != self.requested_mask_rate
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.requested_rate_not_canonical"
            )
        validate_non_negative_integer(
            self.requested_hidden_note_count,
            name="requested_hidden_note_count",
        )
        _validate_policy(self.resolved_policy)
        if self.resolved_policy == INDEPENDENT_NOTE_PITCH:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.control_must_use_phase7a_mask_plan"
            )
        for value, name in (
            (
                self.policy_configuration_fingerprint,
                "policy_configuration_fingerprint",
            ),
            (
                self.relevant_structure_fingerprint,
                "relevant_structure_fingerprint",
            ),
        ):
            if not is_sha256(value):
                raise HierarchyMaskContractError(
                    f"phase8a.hierarchy.{name}_invalid"
                )
        config = validate_hierarchy_policy_config(
            self.policy_configuration
        )
        if (
            config.fingerprint
            != self.policy_configuration_fingerprint
            or config.weight(self.resolved_policy) <= 0.0
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_policy_config_mismatch"
            )
        if (
            self.resolved_policy
            in {
                CONTIGUOUS_BAR_PITCH_SPAN,
                TRACK_BAR_PITCH_SPAN,
            }
            and self.selection.span_length_bars is not None
            and not (
                config.min_span_bars
                <= self.selection.span_length_bars
                <= config.max_span_bars
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_span_outside_config"
            )
        if self.resolved_policy in {
            CONTIGUOUS_BAR_PITCH_SPAN,
            TRACK_BAR_PITCH_SPAN,
        } and (
            self.selection.span_configured_pool_size_limit
            != config.span_selection_pool_size
            or self.selection.span_configured_budget_error_slack
            != config.span_budget_error_slack
            or self.selection.span_selection_method
            != SPAN_SELECTION_METHOD
            or self.selection.span_pool_membership_rank_method
            != SPAN_POOL_MEMBERSHIP_RANK_METHOD
            or self.selection.span_final_choice_rank_method
            != SPAN_FINAL_CHOICE_RANK_METHOD
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy."
                "plan_span_selection_config_mismatch"
            )
        expected_seed = derive_stable_seed(
            namespace="music_critic.ssl.hierarchy_mask.plan.v2",
            global_seed=self.global_seed,
            dataset_id=self.dataset_id,
            piece_id=self.piece_id,
            epoch=self.epoch,
            view_index=self.encoder_view_index,
            extra={
                "stage": self.stage,
                "policy": self.resolved_policy,
                "policy_version": HIERARCHY_MASK_POLICY_VERSION,
                "policy_configuration_fingerprint": (
                    self.policy_configuration_fingerprint
                ),
                "relevant_structure_fingerprint": (
                    self.relevant_structure_fingerprint
                ),
                "maskable_field_registry_version": (
                    MASKABLE_FIELD_REGISTRY_VERSION
                ),
                "maskable_field_registry_fingerprint": (
                    MASKABLE_FIELD_REGISTRY_FINGERPRINT
                ),
            },
        )
        if (
            self.stable_seed != expected_seed.value
            or self.stable_seed_sha256 != expected_seed.sha256
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_seed_non_canonical"
            )
        if (
            self.selection.policy != self.resolved_policy
            or self.primary_feature_group != NOTE_PITCH_GROUP_NAME
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_selection_incompatible"
            )
        validate_non_negative_integer(
            self.pitched_note_count,
            name="pitched_note_count",
        )
        expected_requested_count = _target_count(
            self.pitched_note_count,
            rate,
        )
        if (
            self.requested_hidden_note_count
            != expected_requested_count
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.requested_count_inconsistent"
            )
        for name, value in (
            ("primary_masked_count", self.primary_masked_count),
            (
                "visible_pitched_note_count",
                self.visible_pitched_note_count,
            ),
            (
                "collateral_peer_note_count",
                self.collateral_peer_note_count,
            ),
            (
                "collateral_owner_track_count",
                self.collateral_owner_track_count,
            ),
        ):
            validate_non_negative_integer(value, name=name)
        selected = _indices(
            self.selection.selected_local_note_descendants,
            name="selected_note_descendants",
            upper_bound=self.pitched_note_count,
        )
        if (
            self.primary_masked_count != len(selected)
            or self.visible_pitched_note_count
            != self.pitched_note_count - len(selected)
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_counts_inconsistent"
            )
        expected_rate = (
            len(selected) / self.pitched_note_count
            if self.pitched_note_count
            else 0.0
        )
        if (
            isinstance(self.realized_mask_rate, bool)
            or not isinstance(self.realized_mask_rate, float)
            or not math.isfinite(self.realized_mask_rate)
            or self.realized_mask_rate != expected_rate
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.realized_rate_inconsistent"
            )
        if self.resolved_policy in {
            CONTIGUOUS_BAR_PITCH_SPAN,
            TRACK_BAR_PITCH_SPAN,
        } and self.available and (
            self.selection.span_selected_budget_error
            != abs(
                len(selected) - self.requested_hidden_note_count
            )
            or self.selection.span_selected_descendant_count
            != len(selected)
            or self.selection.span_realized_mask_rate
            != self.realized_mask_rate
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy."
                "plan_span_selection_counts_inconsistent"
            )
        if self.available:
            if (
                self.unavailable_reason is not None
                or not selected
                or not self.selection.selected_local_unit_indices
                or self.selection.candidate_count < 1
                or self.visible_pitched_note_count < 1
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.available_plan_invalid"
                )
            if len(self.collateral_feature_masks) != 2:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.collateral_families_incomplete"
                )
        else:
            reason = self.unavailable_reason
            if (
                reason is None
                or selected
                or self.selection.selected_local_unit_indices
                or self.collateral_feature_masks
                or reason.policy != self.resolved_policy
                or reason.candidate_count
                != self.selection.candidate_count
                or reason.pitched_note_count
                != self.pitched_note_count
                or reason.requested_hidden_note_count
                != self.requested_hidden_note_count
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.unavailable_plan_invalid"
                )
        collateral_by_key = {
            (mask.node_type, mask.reason): mask
            for mask in self.collateral_feature_masks
        }
        if self.available:
            expected_keys = {
                (
                    "note",
                    NOTE_PITCH_GROUP.peer_note_collateral_reason,
                ),
                ("track", NOTE_PITCH_GROUP.collateral_reason),
            }
            if set(collateral_by_key) != expected_keys:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.collateral_keys_invalid"
                )
            peer = collateral_by_key[
                (
                    "note",
                    NOTE_PITCH_GROUP.peer_note_collateral_reason,
                )
            ]
            owner = collateral_by_key[
                ("track", NOTE_PITCH_GROUP.collateral_reason)
            ]
            if (
                peer.features
                != NOTE_PITCH_GROUP.peer_note_collateral_fields
                or owner.features
                != NOTE_PITCH_GROUP.collateral_fields
                or len(peer.local_node_indices)
                != self.collateral_peer_note_count
                or len(owner.local_node_indices)
                != self.collateral_owner_track_count
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.collateral_counts_invalid"
                )
        if not is_sha256(self.fingerprint):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_fingerprint_invalid"
            )
        if self.fingerprint != canonical_sha256(
            _hierarchical_plan_payload(self)
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.plan_fingerprint_inconsistent"
            )

    @property
    def sample_identity(self) -> tuple[str, str]:
        return self.dataset_id, self.piece_id

    @property
    def mask_policy(self) -> str:
        return self.resolved_policy

    @property
    def mask_policy_version(self) -> str:
        return self.policy_version

    @property
    def selected_node_type(self) -> str:
        """Compatibility surface: reconstructed compact rows are notes."""

        return "note"

    @property
    def selected_unit_node_type(self) -> str:
        return self.selection.unit_node_type

    @property
    def selected_local_unit_indices(self) -> tuple[int, ...]:
        return self.selection.selected_local_unit_indices

    @property
    def selected_local_note_indices(self) -> tuple[int, ...]:
        return self.selection.selected_local_note_descendants

    @property
    def selected_local_node_indices(self) -> tuple[int, ...]:
        """Phase 7A overlay/decoder compatibility alias."""

        return self.selection.selected_local_note_descendants

    @property
    def maskable_node_count(self) -> int:
        return self.pitched_note_count

    @property
    def selected_count(self) -> int:
        return self.primary_masked_count

    @property
    def span_start_bar_index(self) -> int | None:
        return self.selection.span_start_bar_index

    @property
    def span_end_bar_index(self) -> int | None:
        return self.selection.span_end_bar_index

    @property
    def span_length_bars(self) -> int | None:
        return self.selection.span_length_bars

    @property
    def selected_local_track_index(self) -> int | None:
        return self.selection.selected_local_track_index

    @property
    def collateral_node_count(self) -> int:
        return (
            self.collateral_peer_note_count
            + self.collateral_owner_track_count
        )

    def to_dict(self) -> dict[str, object]:
        payload = _hierarchical_plan_payload(self)
        payload["fingerprint"] = self.fingerprint
        return payload


def hierarchical_mask_plan_fingerprint(
    plan: HierarchicalMaskPlan,
) -> str:
    if not isinstance(plan, HierarchicalMaskPlan):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.plan_type_invalid"
        )
    return canonical_sha256(_hierarchical_plan_payload(plan))


@dataclass(frozen=True, slots=True)
class HierarchyPolicyEligibility:
    """One policy's deterministic eligibility for one raw sample."""

    policy: HierarchyMaskPolicy
    configured_weight: float
    eligible: bool
    unavailable_reason: HierarchyMaskUnavailableReason | None

    def __post_init__(self) -> None:
        policy = _validate_policy(self.policy)
        if (
            isinstance(self.configured_weight, bool)
            or not isinstance(self.configured_weight, float)
            or not math.isfinite(self.configured_weight)
            or self.configured_weight < 0.0
            or (
                self.configured_weight == 0.0
                and math.copysign(
                    1.0,
                    self.configured_weight,
                )
                < 0.0
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_weight_invalid"
            )
        if type(self.eligible) is not bool:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_flag_invalid"
            )
        if (
            self.unavailable_reason is not None
            and (
                type(self.unavailable_reason)
                is not HierarchyMaskUnavailableReason
                or self.unavailable_reason.policy != policy
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_reason_invalid"
            )
        if self.eligible != (self.unavailable_reason is None):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_reason_inconsistent"
            )
        is_disabled = (
            self.unavailable_reason is not None
            and self.unavailable_reason.code == "policy_disabled"
        )
        if is_disabled != (self.configured_weight == 0.0):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_disabled_inconsistent"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "configured_weight": self.configured_weight,
            "eligible": self.eligible,
            "unavailable_reason": (
                None
                if self.unavailable_reason is None
                else self.unavailable_reason.to_dict()
            ),
        }


def _resolution_payload(
    resolution: HierarchyMaskResolution,
) -> dict[str, object]:
    return {
        "contract_version": resolution.contract_version,
        "sample_identity": {
            "dataset_id": resolution.dataset_id,
            "piece_id": resolution.piece_id,
        },
        "stage": resolution.stage,
        "epoch": resolution.epoch,
        "encoder_view_index": resolution.encoder_view_index,
        "global_seed": resolution.global_seed,
        "requested_mask_rate": resolution.requested_mask_rate,
        "policy_configuration_fingerprint": (
            resolution.policy_configuration_fingerprint
        ),
        "relevant_structure_fingerprint": (
            resolution.relevant_structure_fingerprint
        ),
        "policy_configuration": (
            resolution.policy_configuration.to_dict()
        ),
        "eligibility": [
            item.to_dict() for item in resolution.eligibility
        ],
        "eligible_normalized_weights": [
            {"policy": policy, "weight": weight}
            for policy, weight in (
                resolution.eligible_normalized_weights
            )
        ],
        "resolved_policy": resolution.resolved_policy,
        "plan_fingerprint": (
            None
            if resolution.plan is None
            else resolution.plan.fingerprint
        ),
        "stable_seed": resolution.stable_seed,
        "stable_seed_sha256": resolution.stable_seed_sha256,
    }


@dataclass(frozen=True, slots=True)
class HierarchyMaskResolution:
    """Explicit deterministic mixture renormalization and resolution."""

    contract_version: str
    dataset_id: str
    piece_id: str
    stage: MaskStage
    epoch: int
    encoder_view_index: int
    global_seed: int
    requested_mask_rate: float
    policy_configuration_fingerprint: str
    relevant_structure_fingerprint: str
    policy_configuration: HierarchyMaskPolicyConfig = field(
        repr=False,
        compare=True,
    )
    eligibility: tuple[HierarchyPolicyEligibility, ...]
    eligible_normalized_weights: tuple[tuple[str, float], ...]
    resolved_policy: HierarchyMaskPolicy | None
    plan: MaskPlan | HierarchicalMaskPlan | None = field(
        repr=False,
        compare=True,
    )
    stable_seed: int
    stable_seed_sha256: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        piece_id: str,
        stage: MaskStage,
        epoch: int,
        encoder_view_index: int,
        global_seed: int,
        requested_mask_rate: float,
        relevant_structure_fingerprint: str,
        config: HierarchyMaskPolicyConfig,
        eligibility: tuple[HierarchyPolicyEligibility, ...],
        eligible_normalized_weights: tuple[
            tuple[str, float], ...
        ],
        resolved_policy: HierarchyMaskPolicy | None,
        plan: MaskPlan | HierarchicalMaskPlan | None,
        stable_seed: StableSeed,
    ) -> HierarchyMaskResolution:
        canonical_config = validate_hierarchy_policy_config(config)
        if (
            type(eligibility) is not tuple
            or not all(
                type(item) is HierarchyPolicyEligibility
                for item in eligibility
            )
            or type(eligible_normalized_weights) is not tuple
            or not all(
                type(item) is tuple and len(item) == 2
                for item in eligible_normalized_weights
            )
            or type(stable_seed) is not StableSeed
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_create_type_invalid"
            )
        values = {
            "contract_version": (
                HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
            ),
            "dataset_id": dataset_id,
            "piece_id": piece_id,
            "stage": stage,
            "epoch": epoch,
            "encoder_view_index": encoder_view_index,
            "global_seed": global_seed,
            "requested_mask_rate": requested_mask_rate,
            "policy_configuration_fingerprint": (
                canonical_config.fingerprint
            ),
            "relevant_structure_fingerprint": (
                relevant_structure_fingerprint
            ),
            "policy_configuration": canonical_config,
            "eligibility": eligibility,
            "eligible_normalized_weights": (
                eligible_normalized_weights
            ),
            "resolved_policy": resolved_policy,
            "plan": plan,
            "stable_seed": stable_seed.value,
            "stable_seed_sha256": stable_seed.sha256,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "fingerprint", "0" * 64)
        return cls(
            **values,
            fingerprint=canonical_sha256(
                _resolution_payload(provisional)
            ),
        )

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_version_incompatible"
            )
        SampleIdentity(self.dataset_id, self.piece_id)
        if _canonical_epoch(self.stage, self.epoch) != self.epoch:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_epoch_not_canonical"
            )
        validate_non_negative_integer(
            self.encoder_view_index,
            name="encoder_view_index",
        )
        validate_global_seed(self.global_seed)
        rate = validate_mask_rate(self.requested_mask_rate)
        if (
            type(self.requested_mask_rate) is not float
            or rate != self.requested_mask_rate
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_rate_not_canonical"
            )
        if not is_sha256(self.policy_configuration_fingerprint):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_config_fingerprint_invalid"
            )
        if not is_sha256(self.relevant_structure_fingerprint):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_structure_fingerprint_invalid"
            )
        config = validate_hierarchy_policy_config(
            self.policy_configuration
        )
        if (
            self.policy_configuration_fingerprint
            != config.fingerprint
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_config_fingerprint_mismatch"
            )
        if (
            type(self.eligibility) is not tuple
            or len(self.eligibility) != len(
                HIERARCHY_MASK_POLICIES
            )
            or not all(
                type(item) is HierarchyPolicyEligibility
                for item in self.eligibility
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_type_invalid"
            )
        if tuple(
            item.policy for item in self.eligibility
        ) != HIERARCHY_MASK_POLICIES:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.eligibility_order_invalid"
            )
        for item, (policy, configured_weight) in zip(
            self.eligibility,
            config.policy_weights,
            strict=True,
        ):
            if (
                item.policy != policy
                or item.configured_weight != configured_weight
                or (
                    item.unavailable_reason is not None
                    and item.unavailable_reason.policy != policy
                )
                or (
                    configured_weight == 0.0
                    and (
                        item.eligible
                        or item.unavailable_reason is None
                        or item.unavailable_reason.code
                        != "policy_disabled"
                    )
                )
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.eligibility_config_mismatch"
                )
        eligible = tuple(
            item.policy
            for item in self.eligibility
            if item.eligible and item.configured_weight > 0.0
        )
        if (
            type(self.eligible_normalized_weights) is not tuple
            or not all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                for item in self.eligible_normalized_weights
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.normalized_weight_type_invalid"
            )
        if tuple(
            policy
            for policy, _ in self.eligible_normalized_weights
        ) != eligible:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.normalized_weight_order_invalid"
            )
        eligible_weights = {
            item.policy: Fraction(item.configured_weight)
            for item in self.eligibility
            if item.policy in eligible
        }
        total = sum(
            eligible_weights.values(),
            start=Fraction(0),
        )
        expected_normalized_weights = tuple(
            (
                policy,
                float(eligible_weights[policy] / total),
            )
            for policy in eligible
        )
        if (
            self.eligible_normalized_weights
            != expected_normalized_weights
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.normalized_weights_non_canonical"
            )
        normalized_values = tuple(
            weight for _, weight in self.eligible_normalized_weights
        )
        if normalized_values and (
            any(
                not isinstance(weight, float)
                or not math.isfinite(weight)
                or weight <= 0.0
                for weight in normalized_values
            )
            or not math.isclose(
                sum(normalized_values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.normalized_weights_invalid"
            )
        expected_seed = derive_stable_seed(
            namespace="music_critic.ssl.hierarchy_mask.mixture.v1",
            global_seed=self.global_seed,
            dataset_id=self.dataset_id,
            piece_id=self.piece_id,
            epoch=self.epoch,
            view_index=self.encoder_view_index,
            extra={
                "stage": self.stage,
                "policy_configuration_fingerprint": (
                    self.policy_configuration_fingerprint
                ),
                "eligible_policies": list(eligible),
                "relevant_structure_fingerprint": (
                    self.relevant_structure_fingerprint
                ),
            },
        )
        if (
            self.stable_seed != expected_seed.value
            or self.stable_seed_sha256 != expected_seed.sha256
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.mixture_seed_non_canonical"
            )
        seed = StableSeed(self.stable_seed, self.stable_seed_sha256)
        expected_resolved_policy: HierarchyMaskPolicy | None = None
        if eligible:
            point = Fraction(seed.value, 1 << 64) * total
            cumulative = Fraction(0)
            expected_resolved_policy = eligible[-1]
            for policy in eligible:
                cumulative += eligible_weights[policy]
                if point < cumulative:
                    expected_resolved_policy = policy
                    break
        if self.resolved_policy != expected_resolved_policy:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.resolved_policy_non_deterministic"
            )
        if self.plan is None:
            if (
                self.resolved_policy is not None
                or self.eligible_normalized_weights
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.empty_resolution_invalid"
                )
        else:
            plan = self.plan
            if (
                self.resolved_policy not in eligible
                or plan.sample_identity != self.sample_identity
                or plan.stage != self.stage
                or plan.epoch != self.epoch
                or plan.encoder_view_index
                != self.encoder_view_index
                or plan.global_seed != self.global_seed
                or plan.requested_mask_rate
                != self.requested_mask_rate
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.resolved_plan_context_mismatch"
                )
            if self.resolved_policy == INDEPENDENT_NOTE_PITCH:
                if (
                    type(plan) is not MaskPlan
                    or plan.fingerprint
                    != mask_plan_fingerprint(plan)
                ):
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.control_plan_invalid"
                    )
            elif (
                type(plan) is not HierarchicalMaskPlan
                or not plan.available
                or plan.resolved_policy
                != self.resolved_policy
                or plan.policy_configuration_fingerprint
                != self.policy_configuration_fingerprint
                or plan.policy_configuration
                != self.policy_configuration
                or plan.relevant_structure_fingerprint
                != self.relevant_structure_fingerprint
                or plan.fingerprint
                != hierarchical_mask_plan_fingerprint(plan)
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.resolved_hierarchy_plan_invalid"
                )
        if (
            not is_sha256(self.fingerprint)
            or self.fingerprint
            != canonical_sha256(_resolution_payload(self))
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.resolution_fingerprint_invalid"
            )

    @property
    def sample_identity(self) -> tuple[str, str]:
        return self.dataset_id, self.piece_id

    @property
    def eligible_policies(self) -> tuple[HierarchyMaskPolicy, ...]:
        return tuple(
            policy
            for policy, _ in self.eligible_normalized_weights
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        payload = _resolution_payload(self)
        payload["fingerprint"] = self.fingerprint
        return payload


def hierarchy_mask_resolution_fingerprint(
    resolution: HierarchyMaskResolution,
) -> str:
    """Recompute mixture evidence without mutating the object."""

    if type(resolution) is not HierarchyMaskResolution:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.resolution_type_invalid"
        )
    return canonical_sha256(_resolution_payload(resolution))


@dataclass(frozen=True, slots=True)
class _HierarchyIndex:
    note_count: int
    track_count: int
    bar_count: int
    beat_count: int
    onset_count: int
    onset_notes: tuple[tuple[int, ...], ...]
    beat_onsets: tuple[tuple[int, ...], ...]
    bar_onsets: tuple[tuple[int, ...], ...]
    note_owner_track: tuple[int, ...]
    structure_fingerprint: str


def _ptr(
    graph: HeteroData,
    node_type: str,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(graph, Batch):
        tensor = graph[node_type].ptr
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu":
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.planning_requires_cpu_graph"
            )
        values = tuple(int(value) for value in tensor.detach().tolist())
    else:
        values = (0, int(graph[node_type].num_nodes))
    if (
        len(values) != sample_count + 1
        or values[0] != 0
        or values[-1] != int(graph[node_type].num_nodes)
        or any(left > right for left, right in zip(values, values[1:]))
    ):
        raise HierarchyMaskContractError(
            f"phase8a.hierarchy.{node_type}_ptr_invalid"
        )
    return values


def _edge_rows(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    edge_index = graph[edge_type].edge_index
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.device.type != "cpu"
        or edge_index.ndim != 2
        or int(edge_index.shape[0]) != 2
    ):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.relation_invalid:"
            + "|".join(edge_type)
        )
    rows = edge_index.detach().tolist()
    return tuple(int(value) for value in rows[0]), tuple(
        int(value) for value in rows[1]
    )


def _node_membership(
    ptr: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        sample_index
        for sample_index, (start, end) in enumerate(
            zip(ptr, ptr[1:])
        )
        for _ in range(start, end)
    )


def _build_hierarchy_indices(
    graph: HeteroData,
) -> tuple[_HierarchyIndex, ...]:
    sample_count = int(graph.num_graphs) if isinstance(graph, Batch) else 1
    try:
        if isinstance(graph, Batch):
            validate_raw_graph_batch(graph, sample_count=sample_count)
        else:
            validate_raw_graph(graph)
    except Exception as exc:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.raw_graph_invalid:"
            f"{type(exc).__name__}"
        ) from exc
    node_types = ("track", "bar", "beat", "onset", "note")
    ptr = {
        node_type: _ptr(graph, node_type, sample_count)
        for node_type in node_types
    }
    memberships = {
        node_type: _node_membership(values)
        for node_type, values in ptr.items()
    }
    onset_notes: list[list[int]] = [
        [] for _ in memberships["onset"]
    ]
    beat_onsets: list[list[int]] = [
        [] for _ in memberships["beat"]
    ]
    bar_onsets: list[list[int]] = [
        [] for _ in memberships["bar"]
    ]
    note_owner_track = [-1 for _ in memberships["note"]]
    note_owner_onset = [-1 for _ in memberships["note"]]
    onset_owner_beat = [-1 for _ in memberships["onset"]]
    beat_owner_bar = [-1 for _ in memberships["beat"]]
    onset_owner_bar = [-1 for _ in memberships["onset"]]
    note_owner_bar = [-1 for _ in memberships["note"]]

    relation_targets: tuple[
        tuple[
            tuple[str, str, str],
            list[list[int]] | list[int],
        ],
        ...,
    ] = (
        (_ONSET_STARTS_NOTE, onset_notes),
        (_BEAT_CONTAINS_ONSET, beat_onsets),
        (_BAR_CONTAINS_ONSET, bar_onsets),
        (_TRACK_CONTAINS_NOTE, note_owner_track),
    )
    for edge_type, target_container in relation_targets:
        source_type, _, target_type = edge_type
        sources, targets = _edge_rows(graph, edge_type)
        if len(sources) != len(targets):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.relation_row_mismatch"
            )
        source_count = len(memberships[source_type])
        target_count = len(memberships[target_type])
        seen_pairs: set[tuple[int, int]] = set()
        for source, target in zip(sources, targets, strict=True):
            if (
                not 0 <= source < source_count
                or not 0 <= target < target_count
                or memberships[source_type][source]
                != memberships[target_type][target]
            ):
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.cross_sample_or_out_of_range:"
                    + "|".join(edge_type)
                )
            pair = (source, target)
            if pair in seen_pairs:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.relation_duplicate:"
                    + "|".join(edge_type)
                )
            seen_pairs.add(pair)
            if edge_type == _TRACK_CONTAINS_NOTE:
                owners = target_container
                assert isinstance(owners[target], int)
                if owners[target] != -1:
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.note_owner_duplicate"
                    )
                owners[target] = source
            else:
                descendants = target_container
                row = descendants[source]
                assert isinstance(row, list)
                row.append(target)
            if edge_type == _ONSET_STARTS_NOTE:
                if note_owner_onset[target] != -1:
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.note_onset_owner_duplicate"
                    )
                note_owner_onset[target] = source
            elif edge_type == _BEAT_CONTAINS_ONSET:
                if onset_owner_beat[target] != -1:
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.onset_beat_owner_duplicate"
                    )
                onset_owner_beat[target] = source
            elif edge_type == _BAR_CONTAINS_ONSET:
                if onset_owner_bar[target] != -1:
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.onset_bar_owner_duplicate"
                    )
                onset_owner_bar[target] = source
    if any(owner < 0 for owner in note_owner_track):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.note_owner_missing"
        )
    if any(owner < 0 for owner in note_owner_onset):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.note_onset_owner_missing"
        )
    if any(owner < 0 for owner in onset_owner_beat):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.onset_beat_owner_missing"
        )
    if any(owner < 0 for owner in onset_owner_bar):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.onset_bar_owner_missing"
        )
    if any(not descendants for descendants in onset_notes):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.empty_onset_node"
        )

    beat_bar_sources, beat_bar_targets = _edge_rows(
        graph,
        _BAR_CONTAINS_BEAT,
    )
    for source, target in zip(
        beat_bar_sources,
        beat_bar_targets,
        strict=True,
    ):
        if (
            not 0 <= source < len(memberships["bar"])
            or not 0 <= target < len(memberships["beat"])
            or memberships["bar"][source]
            != memberships["beat"][target]
            or beat_owner_bar[target] != -1
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.beat_bar_owner_invalid"
            )
        beat_owner_bar[target] = source
    if any(owner < 0 for owner in beat_owner_bar):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.beat_bar_owner_missing"
        )
    for onset, beat in enumerate(onset_owner_beat):
        if onset_owner_bar[onset] != beat_owner_bar[beat]:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.onset_beat_bar_composition_mismatch"
            )

    bar_sources, bar_targets = _edge_rows(
        graph,
        _BAR_CONTAINS_NOTE,
    )
    seen_bar_note: set[tuple[int, int]] = set()
    for source, target in zip(
        bar_sources, bar_targets, strict=True
    ):
        if (
            not 0 <= source < len(memberships["bar"])
            or not 0 <= target < len(memberships["note"])
            or memberships["bar"][source]
            != memberships["note"][target]
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.bar_note_cross_sample_or_range"
            )
        pair = (source, target)
        if pair in seen_bar_note or note_owner_bar[target] != -1:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.note_bar_owner_duplicate"
            )
        seen_bar_note.add(pair)
        note_owner_bar[target] = source
    if any(owner < 0 for owner in note_owner_bar):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.note_bar_owner_missing"
        )
    for note, onset in enumerate(note_owner_onset):
        if note_owner_bar[note] != onset_owner_bar[onset]:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.start_bar_composition_mismatch"
            )

    next_sources, next_targets = _edge_rows(graph, _BAR_NEXT)
    next_by_bar = [-1] * len(memberships["bar"])
    for source, target in zip(
        next_sources,
        next_targets,
        strict=True,
    ):
        if (
            not 0 <= source < len(next_by_bar)
            or not 0 <= target < len(next_by_bar)
            or memberships["bar"][source]
            != memberships["bar"][target]
            or next_by_bar[source] != -1
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.bar_next_chain_invalid"
            )
        next_by_bar[source] = target
    for sample_index in range(sample_count):
        start = ptr["bar"][sample_index]
        end = ptr["bar"][sample_index + 1]
        for source in range(start, end):
            expected_target = source + 1 if source + 1 < end else -1
            if next_by_bar[source] != expected_target:
                raise HierarchyMaskContractError(
                    "phase8a.hierarchy.bar_next_chain_invalid"
                )

    result = []
    for sample_index in range(sample_count):
        starts = {
            node_type: ptr[node_type][sample_index]
            for node_type in node_types
        }
        ends = {
            node_type: ptr[node_type][sample_index + 1]
            for node_type in node_types
        }
        local_onset_note_lists = [
            []
            for _ in range(ends["onset"] - starts["onset"])
        ]
        for global_note in range(starts["note"], ends["note"]):
            local_onset_note_lists[
                note_owner_onset[global_note] - starts["onset"]
            ].append(global_note - starts["note"])
        local_onset_notes = tuple(
            tuple(values) for values in local_onset_note_lists
        )
        local_beat_onset_lists = [
            [] for _ in range(ends["beat"] - starts["beat"])
        ]
        local_bar_onset_lists = [
            [] for _ in range(ends["bar"] - starts["bar"])
        ]
        for global_onset in range(
            starts["onset"], ends["onset"]
        ):
            local_onset = global_onset - starts["onset"]
            local_beat_onset_lists[
                onset_owner_beat[global_onset] - starts["beat"]
            ].append(local_onset)
            local_bar_onset_lists[
                onset_owner_bar[global_onset] - starts["bar"]
            ].append(local_onset)
        local_beat_onsets = tuple(
            tuple(values) for values in local_beat_onset_lists
        )
        local_bar_onsets = tuple(
            tuple(values) for values in local_bar_onset_lists
        )
        local_note_owners = tuple(
            note_owner_track[note_index] - starts["track"]
            for note_index in range(starts["note"], ends["note"])
        )
        counts = {
            node_type: ends[node_type] - starts[node_type]
            for node_type in node_types
        }
        if any(
            not 0 <= owner < counts["track"]
            for owner in local_note_owners
        ):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.local_note_owner_invalid"
            )
        structure = {
            "contract": HIERARCHY_MASK_POLICY_VERSION,
            "counts": counts,
            "onset_starts_note": [
                list(values) for values in local_onset_notes
            ],
            "beat_contains_onset": [
                list(values) for values in local_beat_onsets
            ],
            "bar_contains_onset": [
                list(values) for values in local_bar_onsets
            ],
            "note_owner_track": list(local_note_owners),
            "note_owner_onset": [
                note_owner_onset[note_index] - starts["onset"]
                for note_index in range(
                    starts["note"], ends["note"]
                )
            ],
            "onset_owner_beat": [
                onset_owner_beat[onset_index] - starts["beat"]
                for onset_index in range(
                    starts["onset"], ends["onset"]
                )
            ],
            "beat_owner_bar": [
                beat_owner_bar[beat_index] - starts["bar"]
                for beat_index in range(
                    starts["beat"], ends["beat"]
                )
            ],
            "onset_owner_bar": [
                onset_owner_bar[onset_index] - starts["bar"]
                for onset_index in range(
                    starts["onset"], ends["onset"]
                )
            ],
        }
        result.append(
            _HierarchyIndex(
                note_count=counts["note"],
                track_count=counts["track"],
                bar_count=counts["bar"],
                beat_count=counts["beat"],
                onset_count=counts["onset"],
                onset_notes=local_onset_notes,
                beat_onsets=local_beat_onsets,
                bar_onsets=local_bar_onsets,
                note_owner_track=local_note_owners,
                structure_fingerprint=canonical_sha256(structure),
            )
        )
    return tuple(result)


def _target_count(note_count: int, rate: float) -> int:
    if note_count == 0 or rate == 0.0:
        return 0
    if rate == 1.0:
        return note_count
    return max(1, int(math.floor(note_count * rate)))


def _plan_seed(
    *,
    policy: HierarchyMaskPolicy,
    config: HierarchyMaskPolicyConfig,
    identity: SampleIdentity,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    index: _HierarchyIndex,
) -> StableSeed:
    return derive_stable_seed(
        namespace="music_critic.ssl.hierarchy_mask.plan.v2",
        global_seed=global_seed,
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        epoch=epoch,
        view_index=encoder_view_index,
        extra={
            "stage": stage,
            "policy": policy,
            "policy_version": HIERARCHY_MASK_POLICY_VERSION,
            "policy_configuration_fingerprint": config.fingerprint,
            "relevant_structure_fingerprint": (
                index.structure_fingerprint
            ),
            "maskable_field_registry_version": (
                MASKABLE_FIELD_REGISTRY_VERSION
            ),
            "maskable_field_registry_fingerprint": (
                MASKABLE_FIELD_REGISTRY_FINGERPRINT
            ),
        },
    )


def _score(seed: StableSeed, *, purpose: str, value: object) -> str:
    return canonical_sha256(
        {
            "stable_seed_sha256": seed.sha256,
            "purpose": purpose,
            "value": value,
        }
    )


def _descendants_for_onsets(
    index: _HierarchyIndex,
    onsets: Sequence[int],
) -> tuple[int, ...]:
    descendants = tuple(
        note
        for onset in onsets
        for note in index.onset_notes[onset]
    )
    if any(
        left >= right
        for left, right in zip(
            descendants,
            descendants[1:],
        )
    ):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.descendants_not_unique_canonical"
        )
    return descendants


def _unit_candidates(
    policy: HierarchyMaskPolicy,
    index: _HierarchyIndex,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if policy == ONSET_PITCH_DESCENDANTS:
        return tuple(
            (unit, descendants)
            for unit, descendants in enumerate(index.onset_notes)
            if descendants
        )
    if policy == BEAT_PITCH_DESCENDANTS:
        candidates = []
        for unit, onsets in enumerate(index.beat_onsets):
            descendants = _descendants_for_onsets(index, onsets)
            if descendants:
                candidates.append((unit, descendants))
        return tuple(candidates)
    raise AssertionError("unit candidates require onset or beat policy")


def _select_units(
    *,
    candidates: tuple[tuple[int, tuple[int, ...]], ...],
    note_count: int,
    target_count: int,
    seed: StableSeed,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    ordered_list = list(candidates)
    state = seed.value
    for position in range(len(ordered_list) - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        mixed = state
        mixed = (
            (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9
        ) & ((1 << 64) - 1)
        mixed = (
            (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
        ) & ((1 << 64) - 1)
        mixed ^= mixed >> 31
        swap = mixed % (position + 1)
        ordered_list[position], ordered_list[swap] = (
            ordered_list[swap],
            ordered_list[position],
        )
    ordered = tuple(ordered_list)
    selected_units: list[int] = []
    selected_notes: set[int] = set()

    def emitted(
        units: Sequence[int],
        notes: set[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        unit_set = set(units)
        maximum_unit = max(
            (unit for unit, _ in candidates),
            default=-1,
        )
        return (
            tuple(
                unit
                for unit in range(maximum_unit + 1)
                if unit in unit_set
            ),
            tuple(
                note
                for note in range(note_count)
                if note in notes
            ),
    )

    for unit, descendants in ordered:
        new_notes = tuple(
            note for note in descendants if note not in selected_notes
        )
        if len(new_notes) != len(descendants):
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.unit_descendant_overlap"
            )
        if not new_notes:
            continue
        after_count = len(selected_notes) + len(new_notes)
        if after_count >= note_count:
            if selected_notes:
                return emitted(selected_units, selected_notes)
            continue
        if after_count >= target_count:
            after_units_list = [*selected_units, unit]
            after_notes_set = set(selected_notes)
            after_notes_set.update(new_notes)
            if not selected_notes:
                return emitted(after_units_list, after_notes_set)
            before_distance = abs(len(selected_notes) - target_count)
            after_distance = abs(
                len(after_notes_set) - target_count
            )
            if before_distance < after_distance:
                return emitted(selected_units, selected_notes)
            if after_distance < before_distance:
                return emitted(after_units_list, after_notes_set)
            before_units = emitted(
                selected_units,
                selected_notes,
            )[0]
            after_units = emitted(
                after_units_list,
                after_notes_set,
            )[0]
            tie = _score(
                seed,
                purpose="hierarchy_budget_crossing_tie",
                value={
                    "before_units": list(before_units),
                    "after_units": list(after_units),
                },
            )
            return (
                emitted(selected_units, selected_notes)
                if int(tie, 16) % 2 == 0
                else emitted(after_units_list, after_notes_set)
            )
        selected_units.append(unit)
        selected_notes.update(new_notes)
    return (
        emitted(selected_units, selected_notes)
        if selected_notes
        else None
    )


def _bar_note_descendants(
    index: _HierarchyIndex,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _descendants_for_onsets(index, onsets)
        for onsets in index.bar_onsets
    )


def _bar_span_candidates(
    *,
    index: _HierarchyIndex,
    config: HierarchyMaskPolicyConfig,
) -> tuple[
    tuple[int, int, int | None, tuple[int, ...]], ...
]:
    by_bar = _bar_note_descendants(index)
    candidates = []
    for start in range(index.bar_count):
        descendants: list[int] = []
        for length in range(1, config.max_span_bars + 1):
            end = start + length - 1
            if end >= index.bar_count:
                break
            descendants.extend(by_bar[end])
            if length < config.min_span_bars:
                continue
            if descendants and len(descendants) < index.note_count:
                candidates.append(
                    (
                        start,
                        end,
                        None,
                        tuple(descendants),
                    )
                )
    return tuple(candidates)


def _track_bar_span_candidates(
    *,
    index: _HierarchyIndex,
    config: HierarchyMaskPolicyConfig,
) -> tuple[
    tuple[int, int, int | None, tuple[int, ...]], ...
]:
    note_bar = [-1] * index.note_count
    for bar, onsets in enumerate(index.bar_onsets):
        for onset in onsets:
            for note in index.onset_notes[onset]:
                if note_bar[note] != -1:
                    raise HierarchyMaskContractError(
                        "phase8a.hierarchy.note_start_bar_duplicate"
                    )
                note_bar[note] = bar
    if any(bar < 0 for bar in note_bar):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.note_start_bar_missing"
        )
    cell_notes: dict[tuple[int, int], list[int]] = {}
    for note, (track, bar) in enumerate(
        zip(index.note_owner_track, note_bar, strict=True)
    ):
        cell_notes.setdefault((track, bar), []).append(note)

    candidate_keys: dict[tuple[int, int, int], None] = {}
    for track, occupied_bar in cell_notes:
        for length in range(
            config.min_span_bars,
            config.max_span_bars + 1,
        ):
            lowest_start = max(0, occupied_bar - length + 1)
            highest_start = min(
                occupied_bar,
                index.bar_count - length,
            )
            for start in range(lowest_start, highest_start + 1):
                candidate_keys.setdefault(
                    (track, start, start + length - 1),
                    None,
                )
    candidates = []
    for track, start, end in candidate_keys:
        descendants = tuple(
            note
            for bar in range(start, end + 1)
            for note in cell_notes.get((track, bar), ())
        )
        if descendants and len(descendants) < index.note_count:
            candidates.append((start, end, track, descendants))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class _SpanSelection:
    candidate: tuple[int, int, int | None, tuple[int, ...]]
    retained_pool_candidates: tuple[
        tuple[int, int, int | None, tuple[int, ...]], ...
    ]
    best_budget_error: int
    tolerance_candidate_count: int
    admissible_pool_count: int
    selected_budget_error: int


def _span_candidate_identity(
    candidate: tuple[int, int, int | None, tuple[int, ...]],
) -> dict[str, object]:
    return {
        "start": candidate[0],
        "end": candidate[1],
        "track": candidate[2],
        "descendants": list(candidate[3]),
    }


def _span_canonical_candidate_key(
    candidate: tuple[int, int, int | None, tuple[int, ...]],
) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        candidate[2] if candidate[2] is not None else -1,
        candidate[0],
        candidate[1],
        candidate[3],
    )


def _span_candidate_rank(
    *,
    candidate: tuple[int, int, int | None, tuple[int, ...]],
    rank_method: str,
    seed: StableSeed,
    identity: SampleIdentity,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    policy: HierarchyMaskPolicy,
    config: HierarchyMaskPolicyConfig,
) -> str:
    """Return a domain-separated rank binding the full plan identity."""

    return canonical_sha256(
        {
            "rank_method": rank_method,
            "stable_seed_sha256": seed.sha256,
            "sample_identity": {
                "dataset_id": identity.dataset_id,
                "piece_id": identity.piece_id,
            },
            "stage": stage,
            "canonical_epoch": epoch,
            "encoder_view_index": encoder_view_index,
            "global_seed": global_seed,
            "policy": policy,
            "policy_version": HIERARCHY_MASK_POLICY_VERSION,
            "policy_configuration_fingerprint": config.fingerprint,
            "candidate_canonical_identity": (
                _span_candidate_identity(candidate)
            ),
        }
    )


def _select_span(
    *,
    candidates: tuple[
        tuple[int, int, int | None, tuple[int, ...]], ...
    ],
    target_count: int,
    seed: StableSeed,
    config: HierarchyMaskPolicyConfig,
    identity: SampleIdentity,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    policy: HierarchyMaskPolicy,
) -> _SpanSelection | None:
    if not candidates:
        return None
    best_error = min(
        abs(len(candidate[3]) - target_count)
        for candidate in candidates
    )
    maximum_error = best_error + config.span_budget_error_slack
    pool: list[
        tuple[
            str,
            tuple[int, int, int, tuple[int, ...]],
            int,
            tuple[int, int, int | None, tuple[int, ...]],
        ]
    ] = []
    tolerance_candidate_count = 0
    for candidate in candidates:
        candidate_error = abs(len(candidate[3]) - target_count)
        if candidate_error > maximum_error:
            continue
        tolerance_candidate_count += 1
        canonical_key = _span_canonical_candidate_key(candidate)
        membership_rank = _span_candidate_rank(
            candidate=candidate,
            rank_method=SPAN_POOL_MEMBERSHIP_RANK_METHOD,
            seed=seed,
            identity=identity,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            policy=policy,
            config=config,
        )
        ranked_key = (membership_rank, canonical_key)
        insert_at = 0
        while (
            insert_at < len(pool)
            and (pool[insert_at][0], pool[insert_at][1])
            < ranked_key
        ):
            insert_at += 1
        if (
            insert_at >= config.span_selection_pool_size
            and len(pool) >= config.span_selection_pool_size
        ):
            continue
        pool.insert(
            insert_at,
            (
                membership_rank,
                canonical_key,
                candidate_error,
                candidate,
            ),
        )
        if len(pool) > config.span_selection_pool_size:
            pool.pop()
    if not pool:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.span_pool_empty_after_best_candidate"
        )
    _, _, selected_error, selected_candidate = min(
        pool,
        key=lambda item: (
            _span_candidate_rank(
                candidate=item[3],
                rank_method=SPAN_FINAL_CHOICE_RANK_METHOD,
                seed=seed,
                identity=identity,
                stage=stage,
                epoch=epoch,
                encoder_view_index=encoder_view_index,
                global_seed=global_seed,
                policy=policy,
                config=config,
            ),
            item[1],
        ),
    )
    return _SpanSelection(
        candidate=selected_candidate,
        retained_pool_candidates=tuple(item[3] for item in pool),
        best_budget_error=best_error,
        tolerance_candidate_count=tolerance_candidate_count,
        admissible_pool_count=len(pool),
        selected_budget_error=selected_error,
    )


def _collateral_masks(
    index: _HierarchyIndex,
    selected_notes: tuple[int, ...],
) -> tuple[CollateralFeatureMask, ...]:
    selected_set = set(selected_notes)
    selected_owner_set = {
        index.note_owner_track[note]
        for note in selected_notes
    }
    owner_tracks = tuple(
        track
        for track in range(index.track_count)
        if track in selected_owner_set
    )
    owner_set = set(owner_tracks)
    peers = tuple(
        note
        for note, owner in enumerate(index.note_owner_track)
        if note not in selected_set and owner in owner_set
    )
    return (
        CollateralFeatureMask(
            reason=NOTE_PITCH_GROUP.peer_note_collateral_reason,
            node_type="note",
            local_node_indices=peers,
            features=NOTE_PITCH_GROUP.peer_note_collateral_fields,
        ),
        CollateralFeatureMask(
            reason=NOTE_PITCH_GROUP.collateral_reason,
            node_type="track",
            local_node_indices=owner_tracks,
            features=NOTE_PITCH_GROUP.collateral_fields,
        ),
    )


def _unavailable_plan(
    *,
    policy: HierarchyMaskPolicy,
    code: str,
    candidate_count: int,
    index: _HierarchyIndex,
    identity: SampleIdentity,
    config: HierarchyMaskPolicyConfig,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    rate: float,
    target_count: int,
    seed: StableSeed,
    span_selection: _SpanSelection | None = None,
) -> HierarchicalMaskPlan:
    reason = HierarchyMaskUnavailableReason.create(
        policy=policy,
        code=code,
        candidate_count=candidate_count,
        pitched_note_count=index.note_count,
        requested_hidden_note_count=target_count,
    )
    return HierarchicalMaskPlan.create(
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        stable_seed=seed,
        requested_mask_rate=rate,
        requested_hidden_note_count=target_count,
        resolved_policy=policy,
        policy_configuration=config,
        relevant_structure_fingerprint=index.structure_fingerprint,
        selection=SelectedHierarchyUnits.create(
            policy=policy,
            selected_local_unit_indices=(),
            selected_local_note_descendants=(),
            total_valid_candidate_count=candidate_count,
            span_best_budget_error=(
                None
                if span_selection is None
                else span_selection.best_budget_error
            ),
            span_tolerance_candidate_count=(
                (
                    0
                    if span_selection is None
                    else span_selection.tolerance_candidate_count
                )
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_admissible_pool_count=(
                (
                    0
                    if span_selection is None
                    else span_selection.admissible_pool_count
                )
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_configured_pool_size_limit=(
                config.span_selection_pool_size
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_configured_budget_error_slack=(
                config.span_budget_error_slack
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_selection_method=(
                SPAN_SELECTION_METHOD
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_pool_membership_rank_method=(
                SPAN_POOL_MEMBERSHIP_RANK_METHOD
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
            span_final_choice_rank_method=(
                SPAN_FINAL_CHOICE_RANK_METHOD
                if policy
                in {
                    CONTIGUOUS_BAR_PITCH_SPAN,
                    TRACK_BAR_PITCH_SPAN,
                }
                else None
            ),
        ),
        collateral_feature_masks=(),
        pitched_note_count=index.note_count,
        available=False,
        unavailable_reason=reason,
    )


def _hierarchical_plan_from_index(
    *,
    policy: HierarchyMaskPolicy,
    index: _HierarchyIndex,
    identity: SampleIdentity,
    config: HierarchyMaskPolicyConfig,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    requested_mask_rate: float,
) -> HierarchicalMaskPlan:
    if policy == INDEPENDENT_NOTE_PITCH:
        raise AssertionError("control policy uses the Phase 7A planner")
    SSL_MASKABLE_FIELD_REGISTRY.resolve_group(NOTE_PITCH_GROUP_NAME)
    seed = _plan_seed(
        policy=policy,
        config=config,
        identity=identity,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        index=index,
    )
    target_count = _target_count(
        index.note_count,
        requested_mask_rate,
    )
    if requested_mask_rate == 0.0:
        zero_rate_span_selection: _SpanSelection | None = None
        if policy in {
            CONTIGUOUS_BAR_PITCH_SPAN,
            TRACK_BAR_PITCH_SPAN,
        }:
            zero_rate_candidates = (
                _bar_span_candidates(index=index, config=config)
                if policy == CONTIGUOUS_BAR_PITCH_SPAN
                else _track_bar_span_candidates(
                    index=index,
                    config=config,
                )
            )
            zero_rate_candidate_count = len(
                zero_rate_candidates
            )
            zero_rate_span_selection = _select_span(
                candidates=zero_rate_candidates,
                target_count=target_count,
                seed=seed,
                config=config,
                identity=identity,
                stage=stage,
                epoch=epoch,
                encoder_view_index=encoder_view_index,
                global_seed=global_seed,
                policy=policy,
            )
        else:
            zero_rate_candidate_count = len(
                _unit_candidates(policy, index)
            )
        return _unavailable_plan(
            policy=policy,
            code="zero_requested_mask_rate",
            candidate_count=zero_rate_candidate_count,
            index=index,
            identity=identity,
            config=config,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            rate=requested_mask_rate,
            target_count=target_count,
            seed=seed,
            span_selection=zero_rate_span_selection,
        )
    if (
        index.note_count == 0
        and policy
        in {
            ONSET_PITCH_DESCENDANTS,
            BEAT_PITCH_DESCENDANTS,
        }
    ):
        return _unavailable_plan(
            policy=policy,
            code="no_nonempty_hierarchy_units",
            candidate_count=0,
            index=index,
            identity=identity,
            config=config,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            rate=requested_mask_rate,
            target_count=target_count,
            seed=seed,
        )
    if index.note_count < 2:
        return _unavailable_plan(
            policy=policy,
            code="fewer_than_two_pitched_notes",
            candidate_count=0,
            index=index,
            identity=identity,
            config=config,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            rate=requested_mask_rate,
            target_count=target_count,
            seed=seed,
        )

    selection: SelectedHierarchyUnits | None = None
    candidate_count = 0
    unavailable_code = "no_valid_unit_selection"
    if policy in {
        ONSET_PITCH_DESCENDANTS,
        BEAT_PITCH_DESCENDANTS,
    }:
        candidates = _unit_candidates(policy, index)
        candidate_count = len(candidates)
        if not candidates:
            unavailable_code = "no_nonempty_hierarchy_units"
        chosen = _select_units(
            candidates=candidates,
            note_count=index.note_count,
            target_count=target_count,
            seed=seed,
        )
        if chosen is not None:
            units, notes = chosen
            selection = SelectedHierarchyUnits.create(
                policy=policy,
                selected_local_unit_indices=units,
                selected_local_note_descendants=notes,
                total_valid_candidate_count=candidate_count,
            )
    else:
        candidates = (
            _bar_span_candidates(index=index, config=config)
            if policy == CONTIGUOUS_BAR_PITCH_SPAN
            else _track_bar_span_candidates(
                index=index,
                config=config,
            )
        )
        candidate_count = len(candidates)
        unavailable_code = (
            "no_valid_span"
            if policy == CONTIGUOUS_BAR_PITCH_SPAN
            else "no_valid_track_span"
        )
        chosen_span = _select_span(
            candidates=candidates,
            target_count=target_count,
            seed=seed,
            config=config,
            identity=identity,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            policy=policy,
        )
        if chosen_span is not None:
            start, end, track, notes = chosen_span.candidate
            selection = SelectedHierarchyUnits.create(
                policy=policy,
                selected_local_unit_indices=tuple(
                    range(start, end + 1)
                ),
                span_start_bar_index=start,
                span_end_bar_index=end,
                span_length_bars=end - start + 1,
                selected_local_track_index=track,
                selected_local_note_descendants=notes,
                total_valid_candidate_count=candidate_count,
                span_best_budget_error=(
                    chosen_span.best_budget_error
                ),
                span_tolerance_candidate_count=(
                    chosen_span.tolerance_candidate_count
                ),
                span_admissible_pool_count=(
                    chosen_span.admissible_pool_count
                ),
                span_configured_pool_size_limit=(
                    config.span_selection_pool_size
                ),
                span_configured_budget_error_slack=(
                    config.span_budget_error_slack
                ),
                span_selected_budget_error=(
                    chosen_span.selected_budget_error
                ),
                span_selected_descendant_count=len(notes),
                span_realized_mask_rate=(
                    len(notes) / index.note_count
                ),
                span_selection_method=SPAN_SELECTION_METHOD,
                span_pool_membership_rank_method=(
                    SPAN_POOL_MEMBERSHIP_RANK_METHOD
                ),
                span_final_choice_rank_method=(
                    SPAN_FINAL_CHOICE_RANK_METHOD
                ),
            )
    if selection is None:
        return _unavailable_plan(
            policy=policy,
            code=unavailable_code,
            candidate_count=candidate_count,
            index=index,
            identity=identity,
            config=config,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            rate=requested_mask_rate,
            target_count=target_count,
            seed=seed,
        )
    collateral = _collateral_masks(
        index,
        selection.selected_local_note_descendants,
    )
    return HierarchicalMaskPlan.create(
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        stable_seed=seed,
        requested_mask_rate=requested_mask_rate,
        requested_hidden_note_count=target_count,
        resolved_policy=policy,
        policy_configuration=config,
        relevant_structure_fingerprint=index.structure_fingerprint,
        selection=selection,
        collateral_feature_masks=collateral,
        pitched_note_count=index.note_count,
        available=True,
    )


def validate_hierarchical_mask_plans_against_graph(
    graph: HeteroData,
    plans: tuple[object, ...],
) -> None:
    """Rebuild every hierarchy plan from its exact bound raw structure.

    This is a preparation-time CPU check.  It deliberately reconstructs the
    canonical plan instead of trusting portable descendant or candidate
    evidence supplied by a caller.
    """

    if type(plans) is not tuple:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.graph_plan_collection_invalid"
        )
    indices = _build_hierarchy_indices(graph)
    if len(plans) != len(indices):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.graph_plan_count_mismatch"
        )
    for index, plan in zip(indices, plans, strict=True):
        if type(plan) is MaskPlan:
            continue
        if type(plan) is not HierarchicalMaskPlan:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.graph_plan_type_invalid"
            )
        expected = _hierarchical_plan_from_index(
            policy=plan.resolved_policy,
            index=index,
            identity=SampleIdentity(
                plan.dataset_id,
                plan.piece_id,
            ),
            config=plan.policy_configuration,
            stage=plan.stage,
            epoch=plan.epoch,
            encoder_view_index=plan.encoder_view_index,
            global_seed=plan.global_seed,
            requested_mask_rate=plan.requested_mask_rate,
        )
        if plan != expected:
            raise HierarchyMaskContractError(
                "phase8a.hierarchy.graph_plan_non_canonical"
            )


def build_hierarchy_mask_plan(
    graph: HeteroData,
    *,
    dataset_id: str,
    piece_id: str,
    policy: HierarchyMaskPolicy,
    global_seed: int,
    epoch: int,
    encoder_view_index: int = 0,
    requested_mask_rate: float = 0.30,
    stage: MaskStage = "train",
    policy_config: HierarchyMaskPolicyConfig = (
        HierarchyMaskPolicyConfig()
    ),
) -> MaskPlan | HierarchicalMaskPlan:
    """Build one explicit policy; control returns the exact Phase 7A plan."""

    if isinstance(graph, Batch):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.single_planner_received_batch"
        )
    resolved_policy = _validate_policy(policy)
    config = validate_hierarchy_policy_config(policy_config)
    if config.weight(resolved_policy) == 0.0:
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.policy_disabled:"
            + resolved_policy
        )
    identity = SampleIdentity(dataset_id, piece_id)
    validate_global_seed(global_seed)
    canonical_epoch = _canonical_epoch(stage, epoch)
    validate_non_negative_integer(
        encoder_view_index,
        name="encoder_view_index",
    )
    rate = validate_mask_rate(requested_mask_rate)
    if resolved_policy == INDEPENDENT_NOTE_PITCH:
        return build_mask_plan(
            graph,
            dataset_id=dataset_id,
            piece_id=piece_id,
            global_seed=global_seed,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            requested_mask_rate=rate,
            stage=stage,
        )
    index = _build_hierarchy_indices(graph)[0]
    return _hierarchical_plan_from_index(
        policy=resolved_policy,
        index=index,
        identity=identity,
        config=config,
        stage=stage,
        epoch=canonical_epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        requested_mask_rate=rate,
    )


def _independent_unavailable(
    *,
    policy: HierarchyMaskPolicy,
    index: _HierarchyIndex,
    target_count: int,
    code: str,
) -> HierarchyMaskUnavailableReason:
    return HierarchyMaskUnavailableReason.create(
        policy=policy,
        code=code,
        candidate_count=index.note_count,
        pitched_note_count=index.note_count,
        requested_hidden_note_count=target_count,
    )


def _resolve_from_index(
    *,
    index: _HierarchyIndex,
    identity: SampleIdentity,
    independent_plan: MaskPlan,
    config: HierarchyMaskPolicyConfig,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    requested_mask_rate: float,
) -> HierarchyMaskResolution:
    target_count = _target_count(
        index.note_count,
        requested_mask_rate,
    )
    candidate_plans: dict[
        HierarchyMaskPolicy,
        MaskPlan | HierarchicalMaskPlan,
    ] = {INDEPENDENT_NOTE_PITCH: independent_plan}
    eligibility = []
    for policy in HIERARCHY_MASK_POLICIES:
        weight = config.weight(policy)
        reason: HierarchyMaskUnavailableReason | None = None
        if weight == 0.0:
            reason = HierarchyMaskUnavailableReason.create(
                policy=policy,
                code="policy_disabled",
                candidate_count=0,
                pitched_note_count=index.note_count,
                requested_hidden_note_count=target_count,
            )
        elif policy == INDEPENDENT_NOTE_PITCH:
            if requested_mask_rate == 0.0:
                reason = _independent_unavailable(
                    policy=policy,
                    index=index,
                    target_count=target_count,
                    code="zero_requested_mask_rate",
                )
            elif index.note_count < 2:
                reason = _independent_unavailable(
                    policy=policy,
                    index=index,
                    target_count=target_count,
                    code="fewer_than_two_pitched_notes",
                )
            elif (
                independent_plan.selected_count
                >= index.note_count
            ):
                reason = _independent_unavailable(
                    policy=policy,
                    index=index,
                    target_count=target_count,
                    code="would_mask_all_pitched_notes",
                )
        else:
            candidate = _hierarchical_plan_from_index(
                policy=policy,
                index=index,
                identity=identity,
                config=config,
                stage=stage,
                epoch=epoch,
                encoder_view_index=encoder_view_index,
                global_seed=global_seed,
                requested_mask_rate=requested_mask_rate,
            )
            candidate_plans[policy] = candidate
            if not candidate.available:
                reason = candidate.unavailable_reason
        eligibility.append(
            HierarchyPolicyEligibility(
                policy=policy,
                configured_weight=weight,
                eligible=reason is None,
                unavailable_reason=reason,
            )
        )
    eligible = tuple(
        item.policy
        for item in eligibility
        if item.eligible and item.configured_weight > 0.0
    )
    resolution_seed = derive_stable_seed(
        namespace="music_critic.ssl.hierarchy_mask.mixture.v1",
        global_seed=global_seed,
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        epoch=epoch,
        view_index=encoder_view_index,
        extra={
            "stage": stage,
            "policy_configuration_fingerprint": config.fingerprint,
            "eligible_policies": list(eligible),
            "relevant_structure_fingerprint": (
                index.structure_fingerprint
            ),
        },
    )
    if not eligible:
        return HierarchyMaskResolution.create(
            dataset_id=identity.dataset_id,
            piece_id=identity.piece_id,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            requested_mask_rate=requested_mask_rate,
            relevant_structure_fingerprint=(
                index.structure_fingerprint
            ),
            config=config,
            eligibility=tuple(eligibility),
            eligible_normalized_weights=(),
            resolved_policy=None,
            plan=None,
            stable_seed=resolution_seed,
        )
    weights = {
        policy: Fraction(config.weight(policy))
        for policy in eligible
    }
    total = sum(weights.values(), start=Fraction(0))
    normalized = tuple(
        (policy, float(weights[policy] / total))
        for policy in eligible
    )
    point = Fraction(resolution_seed.value, 1 << 64) * total
    cumulative = Fraction(0)
    resolved = eligible[-1]
    for policy in eligible:
        cumulative += weights[policy]
        if point < cumulative:
            resolved = policy
            break
    return HierarchyMaskResolution.create(
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        requested_mask_rate=requested_mask_rate,
        relevant_structure_fingerprint=(
            index.structure_fingerprint
        ),
        config=config,
        eligibility=tuple(eligibility),
        eligible_normalized_weights=normalized,
        resolved_policy=resolved,
        plan=candidate_plans[resolved],
        stable_seed=resolution_seed,
    )


def build_batched_hierarchy_mask_resolutions(
    graph_batch: Batch,
    *,
    dataset_ids: Sequence[str],
    piece_ids: Sequence[str],
    global_seed: int,
    epoch: int,
    encoder_view_index: int = 0,
    requested_mask_rate: float = 0.30,
    stage: MaskStage = "train",
    policy_config: HierarchyMaskPolicyConfig = (
        HierarchyMaskPolicyConfig()
    ),
) -> tuple[HierarchyMaskResolution, ...]:
    """Resolve one batch-order-independent configured policy per sample."""

    config = validate_hierarchy_policy_config(policy_config)
    if not isinstance(graph_batch, Batch):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.batch_planner_requires_batch"
        )
    if isinstance(dataset_ids, (str, bytes)) or isinstance(
        piece_ids, (str, bytes)
    ):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.batch_identities_invalid"
        )
    datasets = tuple(dataset_ids)
    pieces = tuple(piece_ids)
    sample_count = int(graph_batch.num_graphs)
    if (
        sample_count <= 0
        or len(datasets) != sample_count
        or len(pieces) != sample_count
    ):
        raise HierarchyMaskContractError(
            "phase8a.hierarchy.batch_identity_count_invalid"
        )
    identities = tuple(
        SampleIdentity(dataset_id, piece_id)
        for dataset_id, piece_id in zip(
            datasets, pieces, strict=True
        )
    )
    validate_global_seed(global_seed)
    canonical_epoch = _canonical_epoch(stage, epoch)
    validate_non_negative_integer(
        encoder_view_index,
        name="encoder_view_index",
    )
    rate = validate_mask_rate(requested_mask_rate)
    indices = _build_hierarchy_indices(graph_batch)
    independent = build_batched_mask_plans(
        graph_batch,
        dataset_ids=datasets,
        piece_ids=pieces,
        global_seed=global_seed,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=rate,
        stage=stage,
    )
    return tuple(
        _resolve_from_index(
            index=index,
            identity=identity,
            independent_plan=control,
            config=config,
            stage=stage,
            epoch=canonical_epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            requested_mask_rate=rate,
        )
        for index, identity, control in zip(
            indices, identities, independent, strict=True
        )
    )


def build_batched_hierarchy_mask_plans(
    graph_batch: Batch,
    **kwargs: object,
) -> tuple[MaskPlan | HierarchicalMaskPlan, ...]:
    """Return resolved available plans or one structured batch failure."""

    resolutions = build_batched_hierarchy_mask_resolutions(
        graph_batch,
        **kwargs,
    )
    if any(resolution.plan is None for resolution in resolutions):
        raise HierarchyMaskUnavailableError(resolutions)
    return tuple(
        resolution.plan
        for resolution in resolutions
        if resolution.plan is not None
    )


# Public spelling aliases retained to make the Phase 8A surface discoverable.
build_hierarchical_mask_plan = build_hierarchy_mask_plan
build_batched_hierarchical_mask_plans = (
    build_batched_hierarchy_mask_plans
)
build_batched_hierarchical_mask_resolutions = (
    build_batched_hierarchy_mask_resolutions
)


__all__ = [
    "BEAT_PITCH_DESCENDANTS",
    "CONTIGUOUS_BAR_PITCH_SPAN",
    "DEFAULT_SPAN_BUDGET_ERROR_SLACK",
    "DEFAULT_SPAN_SELECTION_POOL_SIZE",
    "HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION",
    "HIERARCHY_MASK_POLICIES",
    "HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT",
    "HIERARCHY_MASK_POLICY_VERSION",
    "HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION",
    "HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION",
    "HIERARCHY_PREPARED_BINDING_PROFILE_VERSION",
    "HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION",
    "HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION",
    "INDEPENDENT_NOTE_PITCH",
    "MAX_SPAN_BARS",
    "MAX_SPAN_BUDGET_ERROR_SLACK",
    "MAX_SPAN_SELECTION_POOL_SIZE",
    "ONSET_PITCH_DESCENDANTS",
    "SPAN_FINAL_CHOICE_RANK_METHOD",
    "SPAN_POOL_MEMBERSHIP_RANK_METHOD",
    "SPAN_SELECTION_METHOD",
    "TRACK_BAR_PITCH_SPAN",
    "HierarchicalMaskPlan",
    "HierarchyMaskContractError",
    "HierarchyMaskPolicy",
    "HierarchyMaskPolicyConfig",
    "HierarchyMaskResolution",
    "HierarchyMaskUnavailableError",
    "HierarchyMaskUnavailableReason",
    "HierarchyPolicyEligibility",
    "SelectedHierarchyUnits",
    "build_batched_hierarchical_mask_plans",
    "build_batched_hierarchical_mask_resolutions",
    "build_batched_hierarchy_mask_plans",
    "build_batched_hierarchy_mask_resolutions",
    "build_hierarchical_mask_plan",
    "build_hierarchy_mask_plan",
    "hierarchical_mask_plan_fingerprint",
    "hierarchy_mask_resolution_fingerprint",
    "hierarchy_policy_config_fingerprint",
    "validate_hierarchical_mask_plans_against_graph",
    "validate_hierarchy_policy_config",
]
