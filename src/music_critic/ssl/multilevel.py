"""Phase 8B.1 independently ablatable hierarchy-level SSL objectives.

This module is additive to the accepted Phase 7A/8A surfaces.  It aligns
masked-online and detached full-view representations only through canonical
raw hierarchy identities already bound by ``PreparedHierarchyMaskBinding``.
It defines representation recovery mechanics, not likelihood, supervision,
preference, critic, or musical-quality semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.contracts import MaskPlan, is_sha256
from music_critic.ssl.data import SSLBatch
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchicalMaskPlan,
)
from music_critic.ssl.masking import (
    PreparedHierarchyMaskBinding,
    PreparedMaskBinding,
)
from music_critic.ssl.model import (
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
    Phase8AHierarchySSLForwardOutput,
    SSLForwardOutput,
)
from music_critic.ssl.objective import (
    AntiCollapseDiagnostics,
    LatentProjectorPredictor,
    RepresentationLoss,
    anti_collapse_diagnostics,
    representation_cosine_loss,
)


PHASE8B_OBJECTIVE_REGISTRY_CONTRACT_VERSION = "1.1.0"
PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION = "1.1.0"
PHASE8B_ELIGIBLE_ENTITY_CONTRACT_VERSION = "1.0.0"
PHASE8B_PREPARED_OBJECTIVE_BINDING_CONTRACT_VERSION = "1.0.0"
PHASE8B_FAMILY_LOSS_CONTRACT_VERSION = "1.1.0"
PHASE8B_OBJECTIVE_LOSS_CONTRACT_VERSION = "1.1.0"
PHASE8B_BATCH_OBJECTIVE_AGGREGATE_CONTRACT_VERSION = "1.0.0"
PHASE8B_LATENT_PREDICTION_CONTRACT_VERSION = "1.0.0"
PHASE8B_MODEL_CONTRACT_VERSION = "1.1.0"
PHASE8B_MODEL_OUTPUT_CONTRACT_VERSION = "1.1.0"
PHASE8B_METRIC_AGGREGATE_CONTRACT_VERSION = "1.1.0"

PHASE7A_NOTE_RECONSTRUCTION = "phase7a_note_reconstruction"
PHASE7A_BAR_LATENT = "phase7a_bar_latent"
PHASE7A_SONG_LATENT = "phase7a_song_latent"
ONSET_LATENT = "onset_latent"
BEAT_LATENT = "beat_latent"
HIERARCHY_BAR_LATENT = "hierarchy_bar_latent"
TRACK_LATENT = "track_latent"

PHASE8B_OBJECTIVE_FAMILIES = (
    PHASE7A_NOTE_RECONSTRUCTION,
    PHASE7A_BAR_LATENT,
    PHASE7A_SONG_LATENT,
    ONSET_LATENT,
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    TRACK_LATENT,
)
PHASE8B_NEW_OBJECTIVE_FAMILIES = (
    ONSET_LATENT,
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    TRACK_LATENT,
)
PHASE8B_CANONICAL_POLICY_ORDER = (
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    TRACK_BAR_PITCH_SPAN,
)

Phase8BObjectiveFamily = Literal[
    "phase7a_note_reconstruction",
    "phase7a_bar_latent",
    "phase7a_song_latent",
    "onset_latent",
    "beat_latent",
    "hierarchy_bar_latent",
    "track_latent",
]
Phase8BObjectiveMode = Literal[
    "phase7a_control",
    "onset_only",
    "beat_only",
    "bar_only",
    "track_only",
    "multilevel_equal_weight",
]

PHASE8B_OBJECTIVE_MODES = (
    "phase7a_control",
    "onset_only",
    "beat_only",
    "bar_only",
    "track_only",
    "multilevel_equal_weight",
)

_NO_ELIGIBLE_ENTITIES = "no_eligible_entities_for_resolved_policies"
_INACTIVE_ZERO_WEIGHT = "inactive_zero_weight"
_NO_AVAILABLE_ACTIVE_FAMILY = "no_available_active_objective_family"
PHASE8B_SCHEDULED_VIEW_AGGREGATION = (
    "sum_family_numerators_across_scheduled_views_divided_by_"
    "sum_family_denominators_then_apply_each_family_weight_once"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _weight(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class Phase8BObjectiveSpec:
    """One immutable registry row separating level and representation path."""

    family: Phase8BObjectiveFamily
    level: str
    encoder_output: str
    eligible_mask_policies: tuple[str, ...]
    new_phase8b_head: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "level": self.level,
            "encoder_output": self.encoder_output,
            "eligible_mask_policies": list(self.eligible_mask_policies),
            "new_phase8b_head": self.new_phase8b_head,
        }


PHASE8B_OBJECTIVE_REGISTRY = (
    Phase8BObjectiveSpec(
        family=PHASE7A_NOTE_RECONSTRUCTION,
        level="note",
        encoder_output="fused_contextual_note_rows_selected_by_mask_plan",
        eligible_mask_policies=("all_phase7a_phase8a_policies",),
        new_phase8b_head=False,
    ),
    Phase8BObjectiveSpec(
        family=PHASE7A_BAR_LATENT,
        level="bar",
        encoder_output="fused_contextual_bar_rows_all",
        eligible_mask_policies=("all_phase7a_phase8a_policies",),
        new_phase8b_head=False,
    ),
    Phase8BObjectiveSpec(
        family=PHASE7A_SONG_LATENT,
        level="song",
        encoder_output="fused_contextual_song_rows_all",
        eligible_mask_policies=("all_phase7a_phase8a_policies",),
        new_phase8b_head=False,
    ),
    Phase8BObjectiveSpec(
        family=ONSET_LATENT,
        level="onset",
        encoder_output="fused_contextual_local_onset_rows",
        eligible_mask_policies=(ONSET_PITCH_DESCENDANTS,),
        new_phase8b_head=True,
    ),
    Phase8BObjectiveSpec(
        family=BEAT_LATENT,
        level="beat",
        encoder_output="fused_contextual_local_beat_rows",
        eligible_mask_policies=(BEAT_PITCH_DESCENDANTS,),
        new_phase8b_head=True,
    ),
    Phase8BObjectiveSpec(
        family=HIERARCHY_BAR_LATENT,
        level="bar",
        encoder_output="coarse_contextual_bar_rows",
        eligible_mask_policies=(
            CONTIGUOUS_BAR_PITCH_SPAN,
            TRACK_BAR_PITCH_SPAN,
        ),
        new_phase8b_head=True,
    ),
    Phase8BObjectiveSpec(
        family=TRACK_LATENT,
        level="track",
        encoder_output="coarse_contextual_track_rows",
        eligible_mask_policies=(TRACK_BAR_PITCH_SPAN,),
        new_phase8b_head=True,
    ),
)

PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT = _canonical_sha256(
    {
        "contract_version": PHASE8B_OBJECTIVE_REGISTRY_CONTRACT_VERSION,
        "formula": "mean(1-cosine(P_l(z_masked),stopgrad(T_l(z_full))))",
        "cosine_epsilon": 1e-8,
        "target_mode": "shared_stop_gradient_full_view_no_ema",
        "alignment": "exact_raw_graph_identity_and_canonical_ordering",
        "scheduled_view_aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
        "families": [spec.to_dict() for spec in PHASE8B_OBJECTIVE_REGISTRY],
    }
)


@dataclass(frozen=True, slots=True)
class Phase8BObjectiveConfig:
    """Fixed coefficients; unavailable families never renormalize the rest."""

    contract_version: str
    mode: Phase8BObjectiveMode
    family_weights: tuple[tuple[Phase8BObjectiveFamily, float], ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        mode: Phase8BObjectiveMode,
        *,
        weights: Mapping[str, object],
    ) -> Phase8BObjectiveConfig:
        """Create a fingerprinted config, including explicit weight overrides."""

        if mode not in PHASE8B_OBJECTIVE_MODES:
            raise ValueError("unknown Phase 8B.1 objective mode")
        if set(weights) != set(PHASE8B_OBJECTIVE_FAMILIES):
            raise ValueError("Phase 8B.1 weights must cover the exact registry")
        ordered = tuple(
            (
                family,
                _weight(weights[family], name=f"{family}_weight"),
            )
            for family in PHASE8B_OBJECTIVE_FAMILIES
        )
        payload = {
            "contract_version": PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION,
            "mode": mode,
            "objective_registry_fingerprint": (
                PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
            ),
            "aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
            "family_weights": [list(row) for row in ordered],
        }
        return cls(
            contract_version=PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION,
            mode=mode,
            family_weights=ordered,
            fingerprint=_canonical_sha256(payload),
        )

    @classmethod
    def from_hydra(cls, config: object) -> Phase8BObjectiveConfig:
        """Materialize a preset plus any explicit Hydra weight overrides."""

        if getattr(config, "contract_version", None) != (
            PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION
        ):
            raise ValueError("Phase 8B.1 Hydra config version is incompatible")
        mode = getattr(config, "mode", None)
        if mode not in PHASE8B_OBJECTIVE_MODES:
            raise ValueError("Phase 8B.1 Hydra mode is incompatible")
        return cls.create(
            mode=mode,
            weights={
                family: getattr(config, family, None)
                for family in PHASE8B_OBJECTIVE_FAMILIES
            },
        )

    @classmethod
    def for_mode(
        cls,
        mode: Phase8BObjectiveMode,
        *,
        phase7a_note_weight: float = 1.0,
        phase7a_bar_weight: float = 1.0,
        phase7a_song_weight: float = 1.0,
    ) -> Phase8BObjectiveConfig:
        if mode not in PHASE8B_OBJECTIVE_MODES:
            raise ValueError("unknown Phase 8B.1 objective mode")
        weights = {family: 0.0 for family in PHASE8B_OBJECTIVE_FAMILIES}
        if mode == "phase7a_control":
            weights.update(
                {
                    PHASE7A_NOTE_RECONSTRUCTION: _weight(
                        phase7a_note_weight,
                        name="phase7a_note_weight",
                    ),
                    PHASE7A_BAR_LATENT: _weight(
                        phase7a_bar_weight,
                        name="phase7a_bar_weight",
                    ),
                    PHASE7A_SONG_LATENT: _weight(
                        phase7a_song_weight,
                        name="phase7a_song_weight",
                    ),
                }
            )
        elif mode == "onset_only":
            weights[ONSET_LATENT] = 1.0
        elif mode == "beat_only":
            weights[BEAT_LATENT] = 1.0
        elif mode == "bar_only":
            weights[HIERARCHY_BAR_LATENT] = 1.0
        elif mode == "track_only":
            weights[TRACK_LATENT] = 1.0
        else:
            for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
                weights[family] = 1.0
        return cls.create(mode, weights=weights)

    def __post_init__(self) -> None:
        if self.contract_version != PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION:
            raise ValueError("Phase 8B.1 objective config version is incompatible")
        if self.mode not in PHASE8B_OBJECTIVE_MODES:
            raise ValueError("Phase 8B.1 objective mode is incompatible")
        if tuple(family for family, _ in self.family_weights) != (
            PHASE8B_OBJECTIVE_FAMILIES
        ):
            raise ValueError("Phase 8B.1 family weights are not registry ordered")
        canonical_weights = tuple(
            (family, _weight(value, name=f"{family}_weight"))
            for family, value in self.family_weights
        )
        if canonical_weights != self.family_weights or not any(
            value > 0.0 for _, value in canonical_weights
        ):
            raise ValueError("Phase 8B.1 requires at least one active family")
        expected_fingerprint = _canonical_sha256(
            {
                "contract_version": PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION,
                "mode": self.mode,
                "objective_registry_fingerprint": (
                    PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
                ),
                "aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
                "family_weights": [list(row) for row in self.family_weights],
            }
        )
        if (
            not is_sha256(self.fingerprint)
            or self.fingerprint != expected_fingerprint
        ):
            raise ValueError("Phase 8B.1 objective config fingerprint is invalid")

    def weight(self, family: Phase8BObjectiveFamily) -> float:
        if family not in PHASE8B_OBJECTIVE_FAMILIES:
            raise ValueError("unknown Phase 8B.1 objective family")
        return dict(self.family_weights)[family]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "mode": self.mode,
            "objective_registry_fingerprint": (
                PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
            ),
            "aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
            "family_weights": [list(row) for row in self.family_weights],
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class Phase8BEligibleEntities:
    """Canonical, deduplicated exact entities for one new latent family."""

    contract_version: str
    family: Phase8BObjectiveFamily
    node_type: str
    sample_indices: tuple[int, ...]
    local_entity_indices: tuple[int, ...]
    global_entity_indices: tuple[int, ...]
    available: bool
    unavailable_reason: str | None
    fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != PHASE8B_ELIGIBLE_ENTITY_CONTRACT_VERSION:
            raise ValueError("Phase 8B.1 eligibility version is incompatible")
        expected_node_type = {
            ONSET_LATENT: "onset",
            BEAT_LATENT: "beat",
            HIERARCHY_BAR_LATENT: "bar",
            TRACK_LATENT: "track",
        }.get(self.family)
        if self.node_type != expected_node_type:
            raise ValueError("Phase 8B.1 eligibility family/node type is invalid")
        lengths = {
            len(self.sample_indices),
            len(self.local_entity_indices),
            len(self.global_entity_indices),
        }
        if len(lengths) != 1:
            raise ValueError("Phase 8B.1 eligibility columns differ in length")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for values in (
                self.sample_indices,
                self.local_entity_indices,
                self.global_entity_indices,
            )
            for value in values
        ):
            raise ValueError("Phase 8B.1 eligibility indices are invalid")
        rows = tuple(
            zip(
                self.sample_indices,
                self.local_entity_indices,
                self.global_entity_indices,
                strict=True,
            )
        )
        if rows != tuple(sorted(set(rows))):
            raise ValueError("Phase 8B.1 eligibility rows are not canonical")
        if self.available != bool(rows):
            raise ValueError("Phase 8B.1 eligibility availability is inconsistent")
        if self.available != (self.unavailable_reason is None):
            raise ValueError("Phase 8B.1 eligibility reason is inconsistent")
        if not is_sha256(self.fingerprint) or self.fingerprint != _canonical_sha256(
            self._payload()
        ):
            raise ValueError("Phase 8B.1 eligibility fingerprint is invalid")

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "family": self.family,
            "node_type": self.node_type,
            "sample_indices": list(self.sample_indices),
            "local_entity_indices": list(self.local_entity_indices),
            "global_entity_indices": list(self.global_entity_indices),
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def _eligible_entities(
    binding: PreparedMaskBinding,
    family: Phase8BObjectiveFamily,
) -> Phase8BEligibleEntities:
    node_type = {
        ONSET_LATENT: "onset",
        BEAT_LATENT: "beat",
        HIERARCHY_BAR_LATENT: "bar",
        TRACK_LATENT: "track",
    }[family]
    ptr = dict(binding.node_ptrs)[node_type]
    rows: set[tuple[int, int, int]] = set()
    for sample_index, plan in enumerate(binding.mask_plans):
        if type(plan) is MaskPlan:
            local_indices: tuple[int, ...] = ()
        elif type(plan) is HierarchicalMaskPlan:
            policy = plan.resolved_policy
            if family == ONSET_LATENT and policy == ONSET_PITCH_DESCENDANTS:
                local_indices = plan.selected_local_unit_indices
            elif family == BEAT_LATENT and policy == BEAT_PITCH_DESCENDANTS:
                local_indices = plan.selected_local_unit_indices
            elif family == HIERARCHY_BAR_LATENT and policy in {
                CONTIGUOUS_BAR_PITCH_SPAN,
                TRACK_BAR_PITCH_SPAN,
            }:
                local_indices = plan.selected_local_unit_indices
            elif family == TRACK_LATENT and policy == TRACK_BAR_PITCH_SPAN:
                track_index = plan.selected_local_track_index
                local_indices = () if track_index is None else (track_index,)
            else:
                local_indices = ()
        else:
            raise TypeError("Phase 8B.1 requires bound Phase 7A/8A plans")
        start, end = ptr[sample_index], ptr[sample_index + 1]
        for local_index in sorted(set(local_indices)):
            if local_index < 0 or local_index >= end - start:
                raise ValueError("Phase 8B.1 eligible entity is outside its sample")
            rows.add((sample_index, local_index, start + local_index))
    ordered = tuple(sorted(rows))
    payload = {
        "contract_version": PHASE8B_ELIGIBLE_ENTITY_CONTRACT_VERSION,
        "family": family,
        "node_type": node_type,
        "sample_indices": [row[0] for row in ordered],
        "local_entity_indices": [row[1] for row in ordered],
        "global_entity_indices": [row[2] for row in ordered],
        "available": bool(ordered),
        "unavailable_reason": None if ordered else _NO_ELIGIBLE_ENTITIES,
    }
    return Phase8BEligibleEntities(
        contract_version=PHASE8B_ELIGIBLE_ENTITY_CONTRACT_VERSION,
        family=family,
        node_type=node_type,
        sample_indices=tuple(row[0] for row in ordered),
        local_entity_indices=tuple(row[1] for row in ordered),
        global_entity_indices=tuple(row[2] for row in ordered),
        available=bool(ordered),
        unavailable_reason=None if ordered else _NO_ELIGIBLE_ENTITIES,
        fingerprint=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class PreparedPhase8BObjectiveBinding:
    """Portable objective sidecar over one exact Phase 8A prepared binding."""

    contract_version: str
    prepared_hierarchy_binding_fingerprint: str
    objective_registry_fingerprint: str
    objective_config_fingerprint: str
    eligible_entities: tuple[Phase8BEligibleEntities, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != PHASE8B_PREPARED_OBJECTIVE_BINDING_CONTRACT_VERSION
            or not is_sha256(self.prepared_hierarchy_binding_fingerprint)
            or self.objective_registry_fingerprint
            != PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
            or not is_sha256(self.objective_config_fingerprint)
            or tuple(row.family for row in self.eligible_entities)
            != PHASE8B_NEW_OBJECTIVE_FAMILIES
        ):
            raise ValueError("Phase 8B.1 prepared objective binding is invalid")
        if not is_sha256(self.fingerprint) or self.fingerprint != _canonical_sha256(
            self._payload()
        ):
            raise ValueError("Phase 8B.1 prepared objective fingerprint is invalid")

    def _payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "prepared_hierarchy_binding_fingerprint": (
                self.prepared_hierarchy_binding_fingerprint
            ),
            "objective_registry_fingerprint": self.objective_registry_fingerprint,
            "objective_config_fingerprint": self.objective_config_fingerprint,
            "eligible_entity_fingerprints": [
                row.fingerprint for row in self.eligible_entities
            ],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["eligible_entities"] = [
            row.to_dict() for row in self.eligible_entities
        ]
        payload["fingerprint"] = self.fingerprint
        return payload

    def selection(self, family: Phase8BObjectiveFamily) -> Phase8BEligibleEntities:
        for selection in self.eligible_entities:
            if selection.family == family:
                return selection
        raise ValueError("unknown Phase 8B.1 eligible family")


def prepare_phase8b_objective_binding(
    prepared_hierarchy_binding: PreparedMaskBinding,
    objective_config: Phase8BObjectiveConfig,
) -> PreparedPhase8BObjectiveBinding:
    """Bind canonical objective rows without reading graph values or targets."""

    if type(prepared_hierarchy_binding) not in {
        PreparedMaskBinding,
        PreparedHierarchyMaskBinding,
    }:
        raise TypeError("Phase 8B.1 objectives require a prepared SSL binding")
    if type(objective_config) is not Phase8BObjectiveConfig:
        raise TypeError("Phase 8B.1 objective config is invalid")
    selections = tuple(
        _eligible_entities(prepared_hierarchy_binding, family)
        for family in PHASE8B_NEW_OBJECTIVE_FAMILIES
    )
    payload = {
        "contract_version": PHASE8B_PREPARED_OBJECTIVE_BINDING_CONTRACT_VERSION,
        "prepared_hierarchy_binding_fingerprint": (
            prepared_hierarchy_binding.fingerprint
        ),
        "objective_registry_fingerprint": PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
        "objective_config_fingerprint": objective_config.fingerprint,
        "eligible_entity_fingerprints": [
            selection.fingerprint for selection in selections
        ],
    }
    return PreparedPhase8BObjectiveBinding(
        contract_version=PHASE8B_PREPARED_OBJECTIVE_BINDING_CONTRACT_VERSION,
        prepared_hierarchy_binding_fingerprint=(
            prepared_hierarchy_binding.fingerprint
        ),
        objective_registry_fingerprint=PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
        objective_config_fingerprint=objective_config.fingerprint,
        eligible_entities=selections,
        fingerprint=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class Phase8BFamilyLoss:
    """One family with explicit numerator, denominator, state, and weight."""

    contract_version: str
    family: Phase8BObjectiveFamily
    numerator: Tensor | None
    eligible_denominator: int
    mean_loss: Tensor | None
    available: bool
    unavailable_reason: str | None
    configured_weight: float
    active: bool
    zero_norm_count: int

    def __post_init__(self) -> None:
        if self.contract_version != PHASE8B_FAMILY_LOSS_CONTRACT_VERSION:
            raise ValueError("Phase 8B.1 family loss version is incompatible")
        if self.family not in PHASE8B_OBJECTIVE_FAMILIES:
            raise ValueError("Phase 8B.1 family is unknown")
        _weight(self.configured_weight, name="configured_weight")
        if self.active != (self.configured_weight > 0.0):
            raise ValueError("Phase 8B.1 family active state is inconsistent")
        if (
            isinstance(self.eligible_denominator, bool)
            or not isinstance(self.eligible_denominator, int)
            or self.eligible_denominator < 0
            or isinstance(self.zero_norm_count, bool)
            or not isinstance(self.zero_norm_count, int)
            or not 0 <= self.zero_norm_count <= self.eligible_denominator
        ):
            raise ValueError("Phase 8B.1 family numerical evidence is invalid")
        if self.available != (self.eligible_denominator > 0):
            raise ValueError("Phase 8B.1 family availability is inconsistent")
        if self.active and self.available:
            if (
                not isinstance(self.numerator, Tensor)
                or self.numerator.ndim != 0
                or not self.numerator.is_floating_point()
                or self.mean_loss is None
                or self.unavailable_reason is not None
            ):
                raise ValueError("active available Phase 8B.1 family needs a mean")
        elif self.active:
            if (
                self.numerator is not None
                or self.mean_loss is not None
                or self.unavailable_reason != _NO_ELIGIBLE_ENTITIES
            ):
                raise ValueError("empty active Phase 8B.1 family must be unavailable")
        elif (
            self.numerator is not None
            or self.eligible_denominator != 0
            or self.available
            or self.mean_loss is not None
            or self.unavailable_reason != _INACTIVE_ZERO_WEIGHT
        ):
            raise ValueError("inactive Phase 8B.1 family must be explicit")


def _family_loss_from_representation(
    family: Phase8BObjectiveFamily,
    loss: object,
    *,
    configured_weight: float,
) -> Phase8BFamilyLoss:
    numerator = getattr(loss, "numerator")
    denominator = getattr(loss, "denominator")
    mean = getattr(loss, "mean")
    zero_norm_count = getattr(loss, "zero_norm_count")
    active = configured_weight > 0.0
    available = active and denominator > 0
    return Phase8BFamilyLoss(
        contract_version=PHASE8B_FAMILY_LOSS_CONTRACT_VERSION,
        family=family,
        numerator=numerator if available else None,
        eligible_denominator=denominator if active else 0,
        mean_loss=mean if available else None,
        available=available,
        unavailable_reason=(
            None
            if active and denominator > 0
            else _NO_ELIGIBLE_ENTITIES
            if active
            else _INACTIVE_ZERO_WEIGHT
        ),
        configured_weight=configured_weight,
        active=active,
        zero_norm_count=zero_norm_count if active else 0,
    )


def _inactive_new_family_loss(
    family: Phase8BObjectiveFamily,
    selection: Phase8BEligibleEntities,
    reference: Tensor,
) -> Phase8BFamilyLoss:
    return Phase8BFamilyLoss(
        contract_version=PHASE8B_FAMILY_LOSS_CONTRACT_VERSION,
        family=family,
        numerator=None,
        eligible_denominator=0,
        mean_loss=None,
        available=False,
        unavailable_reason=_INACTIVE_ZERO_WEIGHT,
        configured_weight=0.0,
        active=False,
        zero_norm_count=0,
    )


@dataclass(frozen=True, slots=True)
class Phase8BObjectiveLoss:
    """Fixed weighted sum; unavailable active families never renormalize it."""

    contract_version: str
    objective_config_fingerprint: str
    family_losses: tuple[Phase8BFamilyLoss, ...]
    active_families: tuple[Phase8BObjectiveFamily, ...]
    unavailable_active_families: tuple[tuple[Phase8BObjectiveFamily, str], ...]
    total_loss: Tensor | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.contract_version != PHASE8B_OBJECTIVE_LOSS_CONTRACT_VERSION
            or not is_sha256(self.objective_config_fingerprint)
            or tuple(row.family for row in self.family_losses)
            != PHASE8B_OBJECTIVE_FAMILIES
        ):
            raise ValueError("Phase 8B.1 objective loss contract is invalid")
        expected_active = tuple(
            row.family for row in self.family_losses if row.active
        )
        if self.active_families != expected_active:
            raise ValueError("Phase 8B.1 active family evidence is invalid")
        available_active = tuple(
            row for row in self.family_losses if row.active and row.available
        )
        if available_active:
            if self.total_loss is None or self.unavailable_reason is not None:
                raise ValueError("Phase 8B.1 objective requires a scalar total")
        elif self.total_loss is not None or self.unavailable_reason != (
            _NO_AVAILABLE_ACTIVE_FAMILY
        ):
            raise ValueError("Phase 8B.1 all-unavailable objective is invalid")

    @property
    def available(self) -> bool:
        return self.total_loss is not None


def combine_phase8b_family_losses(
    family_losses: tuple[Phase8BFamilyLoss, ...],
    *,
    objective_config: Phase8BObjectiveConfig,
) -> Phase8BObjectiveLoss:
    """Use ``sum_f weight_f * mean_f`` with no normalization by active weight."""

    if tuple(row.family for row in family_losses) != PHASE8B_OBJECTIVE_FAMILIES:
        raise ValueError("Phase 8B.1 losses must use registry order")
    for row in family_losses:
        if row.configured_weight != objective_config.weight(row.family):
            raise ValueError("Phase 8B.1 report weight differs from its config")
    available = tuple(row for row in family_losses if row.active and row.available)
    unavailable = tuple(
        (row.family, row.unavailable_reason or _NO_ELIGIBLE_ENTITIES)
        for row in family_losses
        if row.active and not row.available
    )
    if available:
        means = tuple(row.mean_loss for row in available)
        assert all(mean is not None for mean in means)
        first = means[0]
        assert first is not None
        with torch.autocast(device_type=first.device.type, enabled=False):
            total = torch.stack(
                [
                    mean * row.configured_weight
                    for row, mean in zip(available, means, strict=True)
                    if mean is not None
                ]
            ).sum()
    else:
        total = None
    return Phase8BObjectiveLoss(
        contract_version=PHASE8B_OBJECTIVE_LOSS_CONTRACT_VERSION,
        objective_config_fingerprint=objective_config.fingerprint,
        family_losses=family_losses,
        active_families=tuple(row.family for row in family_losses if row.active),
        unavailable_active_families=unavailable,
        total_loss=total,
        unavailable_reason=None if total is not None else _NO_AVAILABLE_ACTIVE_FAMILY,
    )


@dataclass(frozen=True, slots=True)
class Phase8BLatentPrediction:
    """Inspectable active prediction rows; targets are always detached."""

    contract_version: str
    family: Phase8BObjectiveFamily
    node_type: str
    global_entity_indices: Tensor
    prediction: Tensor
    target: Tensor
    loss: RepresentationLoss
    diagnostics: AntiCollapseDiagnostics

    def __post_init__(self) -> None:
        if self.contract_version != PHASE8B_LATENT_PREDICTION_CONTRACT_VERSION:
            raise ValueError("Phase 8B.1 latent prediction version is incompatible")
        if self.family not in PHASE8B_NEW_OBJECTIVE_FAMILIES:
            raise ValueError("Phase 8B.1 latent prediction family is invalid")
        if self.target.requires_grad:
            raise ValueError("Phase 8B.1 full-view target must be stop-gradient")
        if (
            self.global_entity_indices.dtype != torch.long
            or self.global_entity_indices.ndim != 1
            or self.prediction.shape != self.target.shape
            or self.prediction.ndim != 2
            or self.prediction.shape[0] != self.global_entity_indices.shape[0]
        ):
            raise ValueError("Phase 8B.1 latent rows are incompatible")


@dataclass(frozen=True, slots=True)
class Phase8BMultilevelSSLForwardOutput:
    """Additive output envelope; the nested Phase 8A output is unchanged."""

    contract_version: str
    base_output: Phase8AHierarchySSLForwardOutput
    prepared_objective_binding_fingerprint: str
    eligible_entities: tuple[Phase8BEligibleEntities, ...]
    latent_predictions: tuple[Phase8BLatentPrediction, ...]
    objective: Phase8BObjectiveLoss

    def __post_init__(self) -> None:
        if (
            self.contract_version != PHASE8B_MODEL_OUTPUT_CONTRACT_VERSION
            or type(self.base_output) is not Phase8AHierarchySSLForwardOutput
            or not is_sha256(self.prepared_objective_binding_fingerprint)
            or tuple(row.family for row in self.eligible_entities)
            != PHASE8B_NEW_OBJECTIVE_FAMILIES
        ):
            raise ValueError("Phase 8B.1 model output is incompatible")


class Phase8BMultilevelSSLModel(MaskedGraphSSLModel):
    """Phase 7A model plus four small, independently weighted latent heads."""

    def __init__(
        self,
        encoder_config: HierarchicalBaselineConfig = HierarchicalBaselineConfig(),
        ssl_config: MaskedGraphSSLConfig = MaskedGraphSSLConfig(),
        objective_config: Phase8BObjectiveConfig = Phase8BObjectiveConfig.for_mode(
            "multilevel_equal_weight"
        ),
    ) -> None:
        if objective_config.mode == "phase7a_control":
            raise ValueError(
                "phase7a_control must construct the unchanged Phase 7A model"
            )
        super().__init__(encoder_config, ssl_config)
        self.phase8b_objective_config = objective_config
        hidden_dim = encoder_config.hidden_dim
        projector_hidden_dim = ssl_config.projector_hidden_dim
        self.phase8b_latent_heads = nn.ModuleDict(
            {
                family: LatentProjectorPredictor(hidden_dim, projector_hidden_dim)
                for family in PHASE8B_NEW_OBJECTIVE_FAMILIES
            }
        )

    def ssl_contract_metadata(self) -> dict[str, object]:
        metadata = super().ssl_contract_metadata()
        metadata["phase8b_multilevel"] = {
            "model_contract_version": PHASE8B_MODEL_CONTRACT_VERSION,
            "model_output_contract_version": PHASE8B_MODEL_OUTPUT_CONTRACT_VERSION,
            "objective_registry_contract_version": (
                PHASE8B_OBJECTIVE_REGISTRY_CONTRACT_VERSION
            ),
            "objective_registry_fingerprint": (
                PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
            ),
            "objective_config": self.phase8b_objective_config.to_dict(),
            "target_mode": "shared_stop_gradient_full_view_no_ema",
            "aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
            "new_head_parameter_count": self.new_head_parameter_count(),
        }
        return metadata

    def new_head_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.phase8b_latent_heads.parameters()
        )

    def _detached_full_view_encoder(
        self,
        batch: SSLBatch,
        prepared_mask_binding: PreparedHierarchyMaskBinding,
    ):
        from music_critic.ssl.masking import validate_prepared_mask_binding

        token = validate_prepared_mask_binding(
            batch,
            prepared_mask_binding,
            expected_mask_rate=self.ssl_config.mask_rate,
        )
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                return self.encoder._encode_prepared(
                    batch.raw_graph_batch,
                    prepared_input_token=token,
                )
        finally:
            self.encoder.train(was_training)

    @staticmethod
    def _level_rows(encoded: object, family: Phase8BObjectiveFamily) -> Tensor:
        if family == ONSET_LATENT:
            return encoded.fused.embeddings["onset"]
        if family == BEAT_LATENT:
            return encoded.fused.embeddings["beat"]
        if family == HIERARCHY_BAR_LATENT:
            return encoded.coarse.bar_embeddings
        if family == TRACK_LATENT:
            return encoded.coarse.track_embeddings
        raise ValueError("unknown Phase 8B.1 latent family")

    def forward_multilevel(
        self,
        batch: SSLBatch,
        *,
        prepared_mask_binding: PreparedHierarchyMaskBinding,
        prepared_objective_binding: PreparedPhase8BObjectiveBinding,
    ) -> Phase8BMultilevelSSLForwardOutput:
        """Run exact Phase 8A masking and the configured Phase 8B.1 families."""

        if type(prepared_mask_binding) is not PreparedHierarchyMaskBinding:
            raise TypeError("Phase 8B.1 requires a prepared hierarchy binding")
        if type(prepared_objective_binding) is not PreparedPhase8BObjectiveBinding:
            raise TypeError("Phase 8B.1 requires a prepared objective binding")
        if (
            prepared_objective_binding.prepared_hierarchy_binding_fingerprint
            != prepared_mask_binding.fingerprint
            or prepared_objective_binding.objective_registry_fingerprint
            != PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT
            or prepared_objective_binding.objective_config_fingerprint
            != self.phase8b_objective_config.fingerprint
        ):
            raise ValueError("Phase 8B.1 prepared objective binding mismatch")
        base = super().forward_hierarchy(
            batch,
            prepared_mask_binding=prepared_mask_binding,
        )
        config = self.phase8b_objective_config
        family_reports: list[Phase8BFamilyLoss] = [
            _family_loss_from_representation(
                PHASE7A_NOTE_RECONSTRUCTION,
                base.note_loss,
                configured_weight=config.weight(PHASE7A_NOTE_RECONSTRUCTION),
            ),
            _family_loss_from_representation(
                PHASE7A_BAR_LATENT,
                base.bar_latent.loss,
                configured_weight=config.weight(PHASE7A_BAR_LATENT),
            ),
            _family_loss_from_representation(
                PHASE7A_SONG_LATENT,
                base.song_latent.loss,
                configured_weight=config.weight(PHASE7A_SONG_LATENT),
            ),
        ]
        active_new = tuple(
            family
            for family in PHASE8B_NEW_OBJECTIVE_FAMILIES
            if config.weight(family) > 0.0
        )
        full_view = (
            self._detached_full_view_encoder(batch, prepared_mask_binding)
            if active_new
            else None
        )
        predictions: list[Phase8BLatentPrediction] = []
        for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
            selection = prepared_objective_binding.selection(family)
            online_rows = self._level_rows(base.online_encoder, family)
            weight = config.weight(family)
            if weight == 0.0:
                family_reports.append(
                    _inactive_new_family_loss(family, selection, online_rows)
                )
                continue
            assert full_view is not None
            indices = torch.tensor(
                selection.global_entity_indices,
                dtype=torch.long,
                device=online_rows.device,
            )
            selected_online = online_rows.index_select(0, indices)
            selected_full = self._level_rows(full_view, family).index_select(
                0, indices
            )
            prediction, target = self.phase8b_latent_heads[family](
                selected_online,
                selected_full,
            )
            loss = representation_cosine_loss(
                prediction,
                target,
                component=family,
            )
            predictions.append(
                Phase8BLatentPrediction(
                    contract_version=PHASE8B_LATENT_PREDICTION_CONTRACT_VERSION,
                    family=family,
                    node_type=selection.node_type,
                    global_entity_indices=indices,
                    prediction=prediction,
                    target=target,
                    loss=loss,
                    diagnostics=anti_collapse_diagnostics(
                        target, prediction
                    ),
                )
            )
            family_reports.append(
                _family_loss_from_representation(
                    family,
                    loss,
                    configured_weight=weight,
                )
            )
        reports = tuple(family_reports)
        objective = combine_phase8b_family_losses(
            reports,
            objective_config=config,
        )
        return Phase8BMultilevelSSLForwardOutput(
            contract_version=PHASE8B_MODEL_OUTPUT_CONTRACT_VERSION,
            base_output=base,
            prepared_objective_binding_fingerprint=(
                prepared_objective_binding.fingerprint
            ),
            eligible_entities=prepared_objective_binding.eligible_entities,
            latent_predictions=tuple(predictions),
            objective=objective,
        )


def build_phase8b_model(
    encoder_config: HierarchicalBaselineConfig,
    ssl_config: MaskedGraphSSLConfig,
    objective_config: Phase8BObjectiveConfig,
) -> MaskedGraphSSLModel:
    """Return the literal old model for ``phase7a_control``."""

    if objective_config.mode == "phase7a_control":
        expected = {
            PHASE7A_NOTE_RECONSTRUCTION: ssl_config.note_weight,
            PHASE7A_BAR_LATENT: ssl_config.bar_weight,
            PHASE7A_SONG_LATENT: ssl_config.song_weight,
        }
        if any(
            objective_config.weight(family) != value
            for family, value in expected.items()
        ) or any(
            objective_config.weight(family) != 0.0
            for family in PHASE8B_NEW_OBJECTIVE_FAMILIES
        ):
            raise ValueError("phase7a_control weights must match Phase 7A config")
        return MaskedGraphSSLModel(encoder_config, ssl_config)
    return Phase8BMultilevelSSLModel(
        encoder_config,
        ssl_config,
        objective_config,
    )


def build_phase8b_model_from_config(
    model_config: object,
    ssl_config: object,
    objective_mode_config: object,
) -> MaskedGraphSSLModel:
    """Build the old control or additive model from resolved Hydra objects."""

    if getattr(model_config, "name", None) != "hierarchical":
        raise ValueError("Phase 8B.1 supports only model=hierarchical")
    encoder = HierarchicalBaselineConfig(
        hidden_dim=int(model_config.hidden_dim),
        local_gnn_layers=int(model_config.local_gnn_layers),
        transformer_layers=int(model_config.transformer_layers),
        attention_heads=int(model_config.attention_heads),
        ffn_multiplier=int(model_config.ffn_multiplier),
        dropout=float(model_config.dropout),
        local_residual=bool(model_config.residual),
    )
    objective = MaskedGraphSSLConfig(
        mask_rate=float(ssl_config.mask_rate),
        decoder_views=int(ssl_config.decoder_views),
        decoder_remask_probability=float(ssl_config.decoder_remask_prob),
        decoder_hidden_dim=int(ssl_config.decoder_hidden_dim),
        projector_hidden_dim=int(ssl_config.projector_hidden_dim),
        note_weight=float(ssl_config.note_weight),
        bar_weight=float(ssl_config.bar_weight),
        song_weight=float(ssl_config.song_weight),
        cosine_epsilon=float(ssl_config.epsilon),
    )
    resolved = Phase8BObjectiveConfig.from_hydra(objective_mode_config)
    return build_phase8b_model(encoder, objective, resolved)


@dataclass(frozen=True, slots=True)
class Phase8BDifferentiableFamilyAggregate:
    """One active family aggregated canonically across scheduled views."""

    family: Phase8BObjectiveFamily
    numerator: Tensor | None
    eligible_denominator: int
    mean_loss: Tensor | None
    family_view_pass_count: int
    configured_weight: float
    active: bool
    available: bool
    unavailable_reason: str | None
    applied_family_weight_count: int

    def __post_init__(self) -> None:
        if self.family not in PHASE8B_OBJECTIVE_FAMILIES:
            raise ValueError("Phase 8B.1 batch aggregate family is invalid")
        if (
            isinstance(self.eligible_denominator, bool)
            or not isinstance(self.eligible_denominator, int)
            or self.eligible_denominator < 0
            or isinstance(self.family_view_pass_count, bool)
            or not isinstance(self.family_view_pass_count, int)
            or self.family_view_pass_count < 0
        ):
            raise ValueError("Phase 8B.1 batch aggregate counts are invalid")
        if self.active != (self.configured_weight > 0.0):
            raise ValueError("Phase 8B.1 batch aggregate active state is invalid")
        if self.available != (self.eligible_denominator > 0):
            raise ValueError("Phase 8B.1 batch aggregate availability is invalid")
        if self.available:
            if (
                not isinstance(self.numerator, Tensor)
                or self.numerator.ndim != 0
                or not self.numerator.is_floating_point()
                or not isinstance(self.mean_loss, Tensor)
                or self.mean_loss.ndim != 0
                or not self.mean_loss.is_floating_point()
                or self.mean_loss.device != self.numerator.device
                or self.mean_loss.dtype != self.numerator.dtype
                or self.family_view_pass_count <= 0
                or self.unavailable_reason is not None
                or self.applied_family_weight_count != 1
            ):
                raise ValueError("available Phase 8B.1 batch family is invalid")
        else:
            expected_reason = (
                _NO_ELIGIBLE_ENTITIES if self.active else _INACTIVE_ZERO_WEIGHT
            )
            if (
                self.numerator is not None
                or self.mean_loss is not None
                or self.family_view_pass_count != 0
                or self.unavailable_reason != expected_reason
                or self.applied_family_weight_count != 0
            ):
                raise ValueError("unavailable Phase 8B.1 batch family is invalid")


@dataclass(frozen=True, slots=True)
class Phase8BBatchObjectiveAggregate:
    """Differentiable family-global objective for exactly one CPU batch."""

    contract_version: str
    objective_config_fingerprint: str
    scheduled_policy_pass_count: int
    families: tuple[Phase8BDifferentiableFamilyAggregate, ...]
    total_loss: Tensor | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.contract_version
            != PHASE8B_BATCH_OBJECTIVE_AGGREGATE_CONTRACT_VERSION
            or not is_sha256(self.objective_config_fingerprint)
            or isinstance(self.scheduled_policy_pass_count, bool)
            or not isinstance(self.scheduled_policy_pass_count, int)
            or self.scheduled_policy_pass_count <= 0
            or tuple(row.family for row in self.families)
            != PHASE8B_OBJECTIVE_FAMILIES
        ):
            raise ValueError("Phase 8B.1 batch aggregate contract is invalid")
        available = tuple(row for row in self.families if row.active and row.available)
        if available:
            first = available[0].mean_loss
            if (
                not isinstance(self.total_loss, Tensor)
                or self.total_loss.ndim != 0
                or not self.total_loss.is_floating_point()
                or not isinstance(first, Tensor)
                or self.total_loss.device != first.device
                or self.total_loss.dtype != first.dtype
                or self.unavailable_reason is not None
            ):
                raise ValueError("available Phase 8B.1 batch aggregate needs a total")
        elif self.total_loss is not None or self.unavailable_reason != (
            _NO_AVAILABLE_ACTIVE_FAMILY
        ):
            raise ValueError("unavailable Phase 8B.1 batch aggregate is invalid")

    @property
    def family_view_pass_count(self) -> int:
        return sum(row.family_view_pass_count for row in self.families)

    @property
    def eligible_prediction_row_count(self) -> int:
        return sum(
            row.eligible_denominator for row in self.families if row.active
        )


def _active_family_rows(
    output: (
        Phase8BMultilevelSSLForwardOutput
        | Phase8AHierarchySSLForwardOutput
        | SSLForwardOutput
    ),
    objective_config: Phase8BObjectiveConfig,
) -> tuple[Phase8BFamilyLoss, ...]:
    if type(output) is Phase8BMultilevelSSLForwardOutput:
        if output.objective.objective_config_fingerprint != (
            objective_config.fingerprint
        ):
            raise ValueError("Phase 8B.1 policy output/config mismatch")
        rows = output.objective.family_losses
    elif type(output) in {Phase8AHierarchySSLForwardOutput, SSLForwardOutput}:
        if objective_config.mode != "phase7a_control":
            raise ValueError("old-objective output requires phase7a_control config")
        rows = (
            _family_loss_from_representation(
                PHASE7A_NOTE_RECONSTRUCTION,
                output.note_loss,
                configured_weight=objective_config.weight(
                    PHASE7A_NOTE_RECONSTRUCTION
                ),
            ),
            _family_loss_from_representation(
                PHASE7A_BAR_LATENT,
                output.bar_latent.loss,
                configured_weight=objective_config.weight(PHASE7A_BAR_LATENT),
            ),
            _family_loss_from_representation(
                PHASE7A_SONG_LATENT,
                output.song_latent.loss,
                configured_weight=objective_config.weight(PHASE7A_SONG_LATENT),
            ),
        )
    else:
        raise TypeError("Phase 8B.1 policy output type is invalid")
    return tuple(row for row in rows if row.active)


def aggregate_phase8b_family_loss_views(
    policy_family_losses: tuple[tuple[str, tuple[Phase8BFamilyLoss, ...]], ...],
    *,
    objective_config: Phase8BObjectiveConfig,
) -> Phase8BBatchObjectiveAggregate:
    """Aggregate each family across views before applying its weight once."""

    if not policy_family_losses:
        raise ValueError("Phase 8B.1 requires at least one scheduled view")
    policies = tuple(policy for policy, _rows in policy_family_losses)
    if len(set(policies)) != len(policies):
        raise ValueError("Phase 8B.1 scheduled policy views must be unique")
    if any(policy not in PHASE8B_CANONICAL_POLICY_ORDER for policy in policies):
        raise ValueError("Phase 8B.1 scheduled policy view is unknown")
    order = {
        policy: index
        for index, policy in enumerate(PHASE8B_CANONICAL_POLICY_ORDER)
    }
    canonical = tuple(
        sorted(
            policy_family_losses,
            key=lambda row: (order.get(row[0], len(order)), row[0]),
        )
    )
    by_family: dict[Phase8BObjectiveFamily, list[Tensor]] = {
        family: [] for family in PHASE8B_OBJECTIVE_FAMILIES
    }
    denominators = {family: 0 for family in PHASE8B_OBJECTIVE_FAMILIES}
    view_counts = {family: 0 for family in PHASE8B_OBJECTIVE_FAMILIES}
    for _policy, rows in canonical:
        seen: set[str] = set()
        for row in rows:
            if row.family in seen:
                raise ValueError("Phase 8B.1 view repeats one objective family")
            seen.add(row.family)
            if not row.active:
                continue
            if row.configured_weight != objective_config.weight(row.family):
                raise ValueError("Phase 8B.1 view family weight/config mismatch")
            if row.eligible_denominator == 0:
                if row.numerator is not None or row.mean_loss is not None:
                    raise ValueError("unavailable Phase 8B.1 view fabricated zero")
                continue
            if row.numerator is None:
                raise ValueError("available Phase 8B.1 view lacks a numerator")
            by_family[row.family].append(row.numerator)
            denominators[row.family] += row.eligible_denominator
            view_counts[row.family] += 1
    families = []
    for family in PHASE8B_OBJECTIVE_FAMILIES:
        weight = objective_config.weight(family)
        active = weight > 0.0
        numerators = by_family[family] if active else []
        denominator = denominators[family] if active else 0
        if numerators:
            first = numerators[0]
            if any(
                value.device != first.device or value.dtype != first.dtype
                for value in numerators[1:]
            ):
                raise ValueError("Phase 8B.1 view numerators differ in device/dtype")
            with torch.autocast(device_type=first.device.type, enabled=False):
                numerator = torch.stack(numerators).sum()
                mean = numerator / denominator
        else:
            numerator = None
            mean = None
        families.append(
            Phase8BDifferentiableFamilyAggregate(
                family=family,
                numerator=numerator,
                eligible_denominator=denominator,
                mean_loss=mean,
                family_view_pass_count=view_counts[family] if active else 0,
                configured_weight=weight,
                active=active,
                available=denominator > 0,
                unavailable_reason=(
                    None
                    if denominator > 0
                    else (
                        _NO_ELIGIBLE_ENTITIES
                        if active
                        else _INACTIVE_ZERO_WEIGHT
                    )
                ),
                applied_family_weight_count=1 if denominator > 0 else 0,
            )
        )
    available = tuple(row for row in families if row.active and row.available)
    if available:
        means = tuple(row.mean_loss for row in available)
        assert all(mean is not None for mean in means)
        first_mean = means[0]
        assert first_mean is not None
        with torch.autocast(device_type=first_mean.device.type, enabled=False):
            total = torch.stack(
                [
                    mean * row.configured_weight
                    for row, mean in zip(available, means, strict=True)
                    if mean is not None
                ]
            ).sum()
    else:
        total = None
    return Phase8BBatchObjectiveAggregate(
        contract_version=PHASE8B_BATCH_OBJECTIVE_AGGREGATE_CONTRACT_VERSION,
        objective_config_fingerprint=objective_config.fingerprint,
        scheduled_policy_pass_count=len(canonical),
        families=tuple(families),
        total_loss=total,
        unavailable_reason=None if total is not None else _NO_AVAILABLE_ACTIVE_FAMILY,
    )


def aggregate_phase8b_policy_pass_losses(
    policy_outputs: tuple[
        tuple[
            str,
            Phase8BMultilevelSSLForwardOutput
            | Phase8AHierarchySSLForwardOutput
            | SSLForwardOutput,
        ],
        ...,
    ],
    *,
    objective_config: Phase8BObjectiveConfig,
) -> Phase8BBatchObjectiveAggregate:
    """Convert official policy outputs into the canonical batch objective."""

    return aggregate_phase8b_family_loss_views(
        tuple(
            (policy, _active_family_rows(output, objective_config))
            for policy, output in policy_outputs
        ),
        objective_config=objective_config,
    )


@dataclass(frozen=True, slots=True)
class Phase8BAggregatedFamilyLoss:
    family: Phase8BObjectiveFamily
    numerator: float | None
    eligible_denominator: int
    mean_loss: float | None
    available: bool
    unavailable_reason: str | None
    configured_weight: float
    active: bool
    family_view_pass_count: int
    applied_family_weight_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Phase8BObjectiveAccumulator:
    """Fixed-size CPU scalar accumulator; it never retains prediction tensors."""

    contract_version = PHASE8B_METRIC_AGGREGATE_CONTRACT_VERSION

    def __init__(self, objective_config: Phase8BObjectiveConfig) -> None:
        self.objective_config = objective_config
        self._numerators = {family: 0.0 for family in PHASE8B_OBJECTIVE_FAMILIES}
        self._denominators = {family: 0 for family in PHASE8B_OBJECTIVE_FAMILIES}
        self._family_view_pass_counts = {
            family: 0 for family in PHASE8B_OBJECTIVE_FAMILIES
        }
        self.update_count = 0
        self.packed_host_materialization_count = 0
        self.packed_device_to_host_transfer_count = 0
        self.packed_metric_scalar_count = 0
        self._batch_consistency_count = 0
        self._batch_consistency_all = True
        self._batch_consistency_max_absolute_difference = 0.0
        self._batch_optimizer_total_loss_sum = 0.0
        self._batch_reported_total_loss_sum = 0.0
        self.retained_cuda_tensor_count = 0
        self.retained_prediction_tensor_count = 0

    def update(
        self,
        output: (
            Phase8BMultilevelSSLForwardOutput
            | Phase8AHierarchySSLForwardOutput
            | SSLForwardOutput
        ),
    ) -> None:
        """Backward-compatible single-view update using one packed transfer."""

        active = _active_family_rows(output, self.objective_config)
        policy = next(
            (
                spec.eligible_mask_policies[0]
                for spec in PHASE8B_OBJECTIVE_REGISTRY
                if spec.family == active[0].family
                and spec.eligible_mask_policies[0] in PHASE8B_CANONICAL_POLICY_ORDER
            ),
            INDEPENDENT_NOTE_PITCH,
        )
        self.update_batch(
            aggregate_phase8b_family_loss_views(
                ((policy, active),),
                objective_config=self.objective_config,
            )
        )

    def update_batch(self, batch: Phase8BBatchObjectiveAggregate) -> None:
        """Consume one CPU-batch aggregate with at most one packed D2H."""

        if batch.objective_config_fingerprint != self.objective_config.fingerprint:
            raise ValueError("Phase 8B.1 accumulator/config mismatch")
        available = tuple(
            row for row in batch.families if row.active and row.available
        )
        if available:
            tensors = []
            for row in available:
                if row.numerator is None:
                    raise ValueError("available Phase 8B.1 batch row lacks numerator")
                tensors.append(row.numerator.detach())
            if batch.total_loss is None:
                raise ValueError("available Phase 8B.1 batch lacks optimizer total")
            tensors.append(batch.total_loss.detach())
            first = tensors[0]
            if any(
                value.device != first.device or value.dtype != first.dtype
                for value in tensors[1:]
            ):
                raise ValueError("Phase 8B.1 packed metric tensors mismatch")
            packed_device = torch.stack(tensors)
            packed = packed_device.to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(packed).all()):
                raise ValueError("Phase 8B.1 batch metrics are non-finite")
            self.packed_host_materialization_count += 1
            self.packed_device_to_host_transfer_count += int(
                packed_device.device.type != "cpu"
            )
            self.packed_metric_scalar_count += packed.numel()
            for index, row in enumerate(available):
                numerator = float(packed[index])
                self._numerators[row.family] += numerator
                self._denominators[row.family] += row.eligible_denominator
                self._family_view_pass_counts[row.family] += (
                    row.family_view_pass_count
                )
            optimizer_total = float(packed[-1])
            reported_total = sum(
                row.configured_weight
                * (float(packed[index]) / row.eligible_denominator)
                for index, row in enumerate(available)
            )
            difference = abs(optimizer_total - reported_total)
            tolerance = max(
                1e-12,
                16.0
                * torch.finfo(first.dtype).eps
                * max(1.0, abs(optimizer_total), abs(reported_total)),
            )
            self._batch_consistency_count += 1
            self._batch_consistency_all = (
                self._batch_consistency_all and difference <= tolerance
            )
            self._batch_consistency_max_absolute_difference = max(
                self._batch_consistency_max_absolute_difference,
                difference,
            )
            self._batch_optimizer_total_loss_sum += optimizer_total
            self._batch_reported_total_loss_sum += reported_total
        self.update_count += 1

    def finalize(self) -> dict[str, object]:
        families = []
        for family in PHASE8B_OBJECTIVE_FAMILIES:
            denominator = self._denominators[family]
            weight = self.objective_config.weight(family)
            families.append(
                Phase8BAggregatedFamilyLoss(
                    family=family,
                    numerator=(
                        self._numerators[family] if denominator > 0 else None
                    ),
                    eligible_denominator=denominator,
                    mean_loss=(
                        self._numerators[family] / denominator
                        if denominator > 0
                        else None
                    ),
                    available=denominator > 0,
                    unavailable_reason=(
                        None if denominator > 0 else _NO_ELIGIBLE_ENTITIES
                    ),
                    configured_weight=weight,
                    active=weight > 0.0,
                    family_view_pass_count=self._family_view_pass_counts[family],
                    applied_family_weight_count=(
                        1 if weight > 0.0 and denominator > 0 else 0
                    ),
                )
            )
        total = sum(
            row.configured_weight * row.mean_loss
            for row in families
            if row.active and row.mean_loss is not None
        )
        any_available = any(
            row.active and row.mean_loss is not None for row in families
        )
        return {
            "contract_version": self.contract_version,
            "objective_config_fingerprint": self.objective_config.fingerprint,
            "aggregation": PHASE8B_SCHEDULED_VIEW_AGGREGATION,
            "families": [row.to_dict() for row in families],
            "total_loss": total if any_available else None,
            "differentiable_family_numerators": {
                row.family: row.numerator for row in families
            },
            "family_denominators": {
                row.family: row.eligible_denominator for row in families
            },
            "family_means": {row.family: row.mean_loss for row in families},
            "family_view_pass_counts": {
                row.family: row.family_view_pass_count for row in families
            },
            "applied_family_weight_count": {
                row.family: row.applied_family_weight_count for row in families
            },
            "optimizer_total_loss": (
                self._batch_optimizer_total_loss_sum
                if self._batch_consistency_count
                else None
            ),
            "reported_total_loss": (
                self._batch_reported_total_loss_sum
                if self._batch_consistency_count
                else None
            ),
            "stage_family_global_total_loss": (
                total if any_available else None
            ),
            "optimizer_reported_total_consistency": {
                "consistent": self._batch_consistency_all,
                "checked_batch_count": self._batch_consistency_count,
                "max_absolute_difference": (
                    self._batch_consistency_max_absolute_difference
                ),
                "stage_formula_values_equal": self._batch_consistency_all,
            },
            "unavailable_reason": (
                None if any_available else _NO_AVAILABLE_ACTIVE_FAMILY
            ),
            "update_count": self.update_count,
            "family_view_pass_count": sum(self._family_view_pass_counts.values()),
            "eligible_prediction_row_count": sum(self._denominators.values()),
            "batch_optimizer_total_loss_sum": self._batch_optimizer_total_loss_sum,
            "batch_reported_total_loss_sum": self._batch_reported_total_loss_sum,
            "packed_host_materialization_count": self.packed_host_materialization_count,
            "packed_device_to_host_transfer_count": (
                self.packed_device_to_host_transfer_count
            ),
            "maximum_packed_d2h_transfers_per_cpu_batch": (
                1 if self.packed_device_to_host_transfer_count else 0
            ),
            "packed_metric_scalar_count": self.packed_metric_scalar_count,
            "retained_cuda_tensor_count": self.retained_cuda_tensor_count,
            "retained_prediction_tensor_count": self.retained_prediction_tensor_count,
        }


__all__ = [
    "BEAT_LATENT",
    "HIERARCHY_BAR_LATENT",
    "ONSET_LATENT",
    "PHASE7A_BAR_LATENT",
    "PHASE7A_NOTE_RECONSTRUCTION",
    "PHASE7A_SONG_LATENT",
    "PHASE8B_ELIGIBLE_ENTITY_CONTRACT_VERSION",
    "PHASE8B_BATCH_OBJECTIVE_AGGREGATE_CONTRACT_VERSION",
    "PHASE8B_CANONICAL_POLICY_ORDER",
    "PHASE8B_FAMILY_LOSS_CONTRACT_VERSION",
    "PHASE8B_LATENT_PREDICTION_CONTRACT_VERSION",
    "PHASE8B_METRIC_AGGREGATE_CONTRACT_VERSION",
    "PHASE8B_MODEL_CONTRACT_VERSION",
    "PHASE8B_MODEL_OUTPUT_CONTRACT_VERSION",
    "PHASE8B_NEW_OBJECTIVE_FAMILIES",
    "PHASE8B_OBJECTIVE_CONFIG_CONTRACT_VERSION",
    "PHASE8B_OBJECTIVE_FAMILIES",
    "PHASE8B_OBJECTIVE_LOSS_CONTRACT_VERSION",
    "PHASE8B_OBJECTIVE_MODES",
    "PHASE8B_OBJECTIVE_REGISTRY",
    "PHASE8B_OBJECTIVE_REGISTRY_CONTRACT_VERSION",
    "PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT",
    "PHASE8B_PREPARED_OBJECTIVE_BINDING_CONTRACT_VERSION",
    "PHASE8B_SCHEDULED_VIEW_AGGREGATION",
    "TRACK_LATENT",
    "Phase8BAggregatedFamilyLoss",
    "Phase8BBatchObjectiveAggregate",
    "Phase8BDifferentiableFamilyAggregate",
    "Phase8BEligibleEntities",
    "Phase8BFamilyLoss",
    "Phase8BLatentPrediction",
    "Phase8BMultilevelSSLForwardOutput",
    "Phase8BMultilevelSSLModel",
    "Phase8BObjectiveAccumulator",
    "Phase8BObjectiveConfig",
    "Phase8BObjectiveLoss",
    "Phase8BObjectiveSpec",
    "PreparedPhase8BObjectiveBinding",
    "aggregate_phase8b_family_loss_views",
    "aggregate_phase8b_policy_pass_losses",
    "build_phase8b_model",
    "build_phase8b_model_from_config",
    "combine_phase8b_family_losses",
    "prepare_phase8b_objective_binding",
]
