"""Immutable model-side feature masks for raw graph encoder views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from music_critic.graph import validate_raw_graph, validate_raw_graph_batch
from music_critic.ssl.contracts import (
    MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
    CollateralFeatureMask,
    FeatureKind,
    MaskPlan,
    MaskedFeature,
    SSLContractError,
    canonical_sha256,
    is_sha256,
    mask_plan_fingerprint,
)
from music_critic.ssl.field_registry import (
    NOTE_PITCH_GROUP,
    resolve_feature_column,
)


_TRACK_CONTAINS_NOTE_EDGE = ("track", "contains_note", "note")


@dataclass(frozen=True, slots=True)
class FeatureSlotMask:
    """Global row indices masked for one named raw-registry feature."""

    role: str
    field: MaskedFeature
    global_node_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role not in {
            "primary",
            "collateral",
            "primary_with_peer_collateral",
        }:
            raise SSLContractError("feature slot mask role is invalid")
        indices = self.global_node_indices
        if not isinstance(indices, tuple):
            raise SSLContractError("global feature-mask indices must be a tuple")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise SSLContractError(
                "global feature-mask indices must be non-negative integers"
            )
        if any(
            left >= right
            for left, right in zip(indices, indices[1:])
        ):
            raise SSLContractError(
                "global feature-mask indices must be uniquely sorted"
            )

    @property
    def node_type(self) -> str:
        return self.field.node_type

    @property
    def kind(self) -> FeatureKind:
        return self.field.kind

    @property
    def feature_name(self) -> str:
        return self.field.feature_name

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "field": self.field.to_dict(),
            "global_node_indices": list(self.global_node_indices),
        }


def _overlay_payload(
    *,
    graph_count: int,
    node_counts: tuple[tuple[str, int], ...],
    mask_plan_fingerprints: tuple[str, ...],
    slot_masks: tuple[FeatureSlotMask, ...],
) -> dict[str, object]:
    return {
        "contract_version": MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
        "graph_count": graph_count,
        "node_counts": [
            {"node_type": node_type, "count": count}
            for node_type, count in node_counts
        ],
        "mask_plan_fingerprints": list(mask_plan_fingerprints),
        "slot_masks": [slot.to_dict() for slot in slot_masks],
    }


@dataclass(frozen=True, slots=True)
class FeatureMaskOverlay:
    """Graph-independent immutable row/field masks for one encoder call."""

    contract_version: str
    graph_count: int
    node_counts: tuple[tuple[str, int], ...]
    mask_plan_fingerprints: tuple[str, ...]
    slot_masks: tuple[FeatureSlotMask, ...]
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        graph_count: int,
        node_counts: tuple[tuple[str, int], ...],
        mask_plan_fingerprints: tuple[str, ...],
        slot_masks: tuple[FeatureSlotMask, ...],
    ) -> FeatureMaskOverlay:
        payload = _overlay_payload(
            graph_count=graph_count,
            node_counts=node_counts,
            mask_plan_fingerprints=mask_plan_fingerprints,
            slot_masks=slot_masks,
        )
        return cls(
            contract_version=MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
            graph_count=graph_count,
            node_counts=node_counts,
            mask_plan_fingerprints=mask_plan_fingerprints,
            slot_masks=slot_masks,
            fingerprint=canonical_sha256(payload),
        )

    def __post_init__(self) -> None:
        if self.contract_version != MASKED_FEATURE_OVERLAY_CONTRACT_VERSION:
            raise SSLContractError("feature overlay contract version is incompatible")
        if (
            isinstance(self.graph_count, bool)
            or not isinstance(self.graph_count, int)
            or self.graph_count <= 0
        ):
            raise SSLContractError("feature overlay graph count must be positive")
        node_types = tuple(node_type for node_type, _ in self.node_counts)
        if (
            any(
                left >= right
                for left, right in zip(
                    node_types,
                    node_types[1:],
                )
            )
            or any(
                not isinstance(node_type, str)
                or not node_type
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for node_type, count in self.node_counts
            )
        ):
            raise SSLContractError(
                "feature overlay node counts must be uniquely sorted and non-negative"
            )
        if (
            not isinstance(self.mask_plan_fingerprints, tuple)
            or len(self.mask_plan_fingerprints) != self.graph_count
            or not all(is_sha256(value) for value in self.mask_plan_fingerprints)
        ):
            raise SSLContractError(
                "feature overlay requires one plan fingerprint per graph"
            )
        if not isinstance(self.slot_masks, tuple) or not all(
            isinstance(slot, FeatureSlotMask) for slot in self.slot_masks
        ):
            raise SSLContractError(
                "feature overlay slot masks must be feature-slot contracts"
            )
        keys = tuple(
            (slot.node_type, slot.kind, slot.feature_name)
            for slot in self.slot_masks
        )
        if len(keys) != len(set(keys)):
            raise SSLContractError(
                "feature overlay semantic slot masks must be unique"
            )
        count_by_type = dict(self.node_counts)
        for slot in self.slot_masks:
            if slot.node_type not in count_by_type:
                raise SSLContractError("feature overlay slot has an unknown node type")
            if any(
                index >= count_by_type[slot.node_type]
                for index in slot.global_node_indices
            ):
                raise SSLContractError(
                    "feature overlay contains an out-of-range global node index"
                )
            resolve_feature_column(slot.field)
        if not is_sha256(self.fingerprint):
            raise SSLContractError("feature overlay fingerprint must be SHA-256")
        expected = canonical_sha256(
            _overlay_payload(
                graph_count=self.graph_count,
                node_counts=self.node_counts,
                mask_plan_fingerprints=self.mask_plan_fingerprints,
                slot_masks=self.slot_masks,
            )
        )
        if self.fingerprint != expected:
            raise SSLContractError(
                "feature overlay fingerprint differs from its contents"
            )

    def to_dict(self) -> dict[str, object]:
        payload = _overlay_payload(
            graph_count=self.graph_count,
            node_counts=self.node_counts,
            mask_plan_fingerprints=self.mask_plan_fingerprints,
            slot_masks=self.slot_masks,
        )
        payload["fingerprint"] = self.fingerprint
        return payload

    def _slot(
        self,
        *,
        node_type: str,
        kind: str,
        feature_name: str,
    ) -> FeatureSlotMask | None:
        matches = tuple(
            slot
            for slot in self.slot_masks
            if slot.node_type == node_type
            and slot.kind == kind
            and slot.feature_name == feature_name
        )
        if len(matches) > 1:
            # Primary and collateral groups cannot overlap in Phase 7A.
            raise SSLContractError("feature overlay contains overlapping slot masks")
        return matches[0] if matches else None

    def feature_row_mask(
        self,
        *,
        node_type: str,
        kind: str,
        feature_name: str,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Materialize a boolean row mask without consulting feature values."""

        count_by_type = dict(self.node_counts)
        if node_type not in count_by_type:
            raise SSLContractError(f"feature overlay has no {node_type!r} store")
        if kind not in {"categorical", "continuous"}:
            raise SSLContractError("feature overlay kind is invalid")
        row_mask = torch.zeros(
            count_by_type[node_type],
            dtype=torch.bool,
            device=device,
        )
        slot = self._slot(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
        )
        if slot is not None and slot.global_node_indices:
            indices = torch.tensor(
                slot.global_node_indices,
                dtype=torch.long,
                device=row_mask.device,
            )
            row_mask.index_fill_(0, indices, True)
        return row_mask

    def bind(self, mask_token: Tensor) -> BoundFeatureMaskOverlay:
        """Bind an externally owned learnable token for encoder application."""

        return BoundFeatureMaskOverlay(overlay=self, mask_token=mask_token)

    def replace_combined_contribution(
        self,
        *,
        node_type: str,
        kind: str,
        feature_name: str,
        combined_contribution: Tensor,
        mask_token: Tensor,
    ) -> Tensor:
        """Replace a combined value+availability term with an external token."""

        return self.bind(mask_token).replace_combined_contribution(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
            combined_contribution=combined_contribution,
        )


@dataclass(frozen=True, slots=True)
class BoundFeatureMaskOverlay:
    """An immutable mask sidecar bound to an SSL-model-owned token tensor."""

    overlay: FeatureMaskOverlay
    mask_token: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mask_token, Tensor):
            raise SSLContractError("feature overlay mask token must be a tensor")
        if self.mask_token.ndim not in {1, 2}:
            raise SSLContractError("feature overlay mask token must have rank one or two")
        if self.mask_token.ndim == 2 and int(self.mask_token.shape[0]) != 1:
            raise SSLContractError(
                "rank-two feature overlay mask token must contain one row"
            )
        if not self.mask_token.is_floating_point():
            raise SSLContractError("feature overlay mask token must be floating point")

    def _replacement(
        self,
        *,
        contribution: Tensor,
        row_mask: Tensor,
    ) -> Tensor:
        if contribution.ndim != 2:
            raise SSLContractError("feature contribution must have shape [N, D]")
        if int(contribution.shape[0]) != int(row_mask.shape[0]):
            raise SSLContractError(
                "feature contribution row count differs from overlay"
            )
        token = self.mask_token
        if token.ndim == 2:
            token = token.squeeze(0)
        if int(token.shape[0]) != int(contribution.shape[1]):
            raise SSLContractError(
                "feature overlay mask token hidden dimension is incompatible"
            )
        token = token.to(
            device=contribution.device,
            dtype=contribution.dtype,
        )
        return torch.where(
            row_mask.unsqueeze(-1),
            token.unsqueeze(0),
            contribution,
        )

    def replace_combined_contribution(
        self,
        *,
        node_type: str,
        kind: str,
        feature_name: str,
        combined_contribution: Tensor,
    ) -> Tensor:
        """Replace masked rows of a combined contribution with the token."""

        slot = self.overlay._slot(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
        )
        if slot is None or not slot.global_node_indices:
            return combined_contribution
        row_mask = self.overlay.feature_row_mask(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
            device=combined_contribution.device,
        )
        return self._replacement(
            contribution=combined_contribution,
            row_mask=row_mask,
        )

    def replace_feature_contributions(
        self,
        *,
        node_type: str,
        kind: str,
        feature_name: str,
        value_contribution: Tensor,
        availability_contribution: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encoder hook replacing the complete value+availability evidence."""

        if (
            value_contribution.shape != availability_contribution.shape
            or value_contribution.device != availability_contribution.device
        ):
            raise SSLContractError(
                "value and availability contributions must have equal shape/device"
            )
        slot = self.overlay._slot(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
        )
        if slot is None or not slot.global_node_indices:
            return value_contribution, availability_contribution
        row_mask = self.overlay.feature_row_mask(
            node_type=node_type,
            kind=kind,
            feature_name=feature_name,
            device=value_contribution.device,
        )
        replacement = self._replacement(
            contribution=value_contribution,
            row_mask=row_mask,
        )
        availability = torch.where(
            row_mask.unsqueeze(-1),
            torch.zeros_like(availability_contribution),
            availability_contribution,
        )
        return replacement, availability


def _ptr(
    graph: HeteroData,
    *,
    node_type: str,
    graph_count: int,
) -> tuple[int, ...]:
    if isinstance(graph, Batch):
        values = tuple(
            int(value)
            for value in graph[node_type].ptr.detach().cpu().tolist()
        )
    else:
        values = (0, int(graph[node_type].num_nodes))
    if (
        len(values) != graph_count + 1
        or values[0] != 0
        or values[-1] != int(graph[node_type].num_nodes)
        or any(left > right for left, right in zip(values, values[1:]))
    ):
        raise SSLContractError(f"{node_type}.ptr is incompatible with feature overlay")
    return values


def _validate_collateral(
    plan: object,
) -> tuple[CollateralFeatureMask, CollateralFeatureMask]:
    expected = NOTE_PITCH_GROUP
    if plan.primary_feature_group != expected.name:
        raise SSLContractError("feature overlay received an unsupported primary group")
    by_key = {
        (mask.node_type, mask.reason): mask
        for mask in plan.collateral_feature_masks
    }
    expected_keys = {
        ("note", expected.peer_note_collateral_reason),
        ("track", expected.collateral_reason),
    }
    if set(by_key) != expected_keys:
        raise SSLContractError(
            "note_pitch_group collateral mask families are incomplete"
        )
    peer = by_key[("note", expected.peer_note_collateral_reason)]
    track = by_key[("track", expected.collateral_reason)]
    if (
        peer.features != expected.peer_note_collateral_fields
        or track.features != expected.collateral_fields
    ):
        raise SSLContractError(
            "mask plan collateral fields differ from the Phase 7A registry"
        )
    return peer, track


def _validated_plan_fingerprint(plan: object) -> str:
    """Dispatch only exact known portable plan contracts."""

    if type(plan) is MaskPlan:
        return mask_plan_fingerprint(plan)
    from music_critic.ssl.hierarchical_masking import (
        HierarchicalMaskPlan,
        hierarchical_mask_plan_fingerprint,
    )

    if type(plan) is not HierarchicalMaskPlan:
        raise SSLContractError(
            "feature overlay received an unsupported plan type"
        )
    if not plan.available:
        raise SSLContractError(
            "feature overlay cannot bind an unavailable hierarchy plan"
        )
    return hierarchical_mask_plan_fingerprint(plan)


def _is_supported_plan(plan: object) -> bool:
    if type(plan) is MaskPlan:
        return True
    from music_critic.ssl.hierarchical_masking import (
        HierarchicalMaskPlan,
    )

    return type(plan) is HierarchicalMaskPlan


def _owner_track_by_note(graph: HeteroData) -> tuple[int, ...]:
    note_count = int(graph["note"].num_nodes)
    track_count = int(graph["track"].num_nodes)
    owners = [-1] * note_count
    source, target = (
        graph[_TRACK_CONTAINS_NOTE_EDGE]
        .edge_index.detach()
        .cpu()
        .tolist()
    )
    for track_index, note_index in zip(source, target, strict=True):
        if (
            not 0 <= track_index < track_count
            or not 0 <= note_index < note_count
            or owners[note_index] != -1
        ):
            raise SSLContractError(
                "feature overlay note ownership is invalid"
            )
        owners[note_index] = track_index
    if any(owner < 0 for owner in owners):
        raise SSLContractError(
            "feature overlay requires exactly one owner track per note"
        )
    return tuple(owners)


def _merge_sorted_unique_indices(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    """Return a sorted union in linear time."""

    merged: list[int] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value = left[left_index]
        right_value = right[right_index]
        if left_value < right_value:
            merged.append(left_value)
            left_index += 1
        elif right_value < left_value:
            merged.append(right_value)
            right_index += 1
        else:
            merged.append(left_value)
            left_index += 1
            right_index += 1
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return tuple(merged)


def build_feature_mask_overlay(
    graph: HeteroData,
    plans: object | Sequence[object],
) -> FeatureMaskOverlay:
    """Build global row masks without copying or mutating the raw graph."""

    if _is_supported_plan(plans):
        prepared = (plans,)
    elif isinstance(plans, (str, bytes)) or not isinstance(plans, Sequence):
        raise SSLContractError("feature overlay plans must be a sequence")
    else:
        prepared = tuple(plans)
    graph_count = int(graph.num_graphs) if isinstance(graph, Batch) else 1
    if not prepared or len(prepared) != graph_count:
        raise SSLContractError(
            "feature overlay requires exactly one plan per source graph"
        )
    if not all(_is_supported_plan(plan) for plan in prepared):
        raise SSLContractError(
            "feature overlay received an unsupported plan value"
        )
    if isinstance(graph, Batch):
        validate_raw_graph_batch(graph, sample_count=graph_count)
    else:
        validate_raw_graph(graph)
    if any(type(plan) is not MaskPlan for plan in prepared):
        from music_critic.ssl.hierarchical_masking import (
            validate_hierarchical_mask_plans_against_graph,
        )

        validate_hierarchical_mask_plans_against_graph(
            graph,
            prepared,
        )
    collateral_by_plan = []
    for plan in prepared:
        if plan.fingerprint != _validated_plan_fingerprint(plan):
            raise SSLContractError("feature overlay received a mutated mask plan")
        collateral_by_plan.append(_validate_collateral(plan))

    note_ptr = _ptr(graph, node_type="note", graph_count=graph_count)
    track_ptr = _ptr(graph, node_type="track", graph_count=graph_count)
    owner_track_by_note = _owner_track_by_note(graph)
    primary_global_by_sample: list[tuple[int, ...]] = []
    peer_global_by_sample: list[tuple[int, ...]] = []
    collateral_global_by_sample: list[tuple[int, ...]] = []
    for sample_index, (plan, collateral_masks) in enumerate(
        zip(prepared, collateral_by_plan, strict=True)
    ):
        note_count = note_ptr[sample_index + 1] - note_ptr[sample_index]
        track_count = track_ptr[sample_index + 1] - track_ptr[sample_index]
        if plan.maskable_node_count != note_count:
            raise SSLContractError(
                "mask plan note cardinality differs from its source graph"
            )
        if any(index >= note_count for index in plan.selected_local_node_indices):
            raise SSLContractError("mask plan selected note index is out of range")
        peer_collateral, track_collateral = collateral_masks
        if any(
            index >= note_count
            for index in peer_collateral.local_node_indices
        ):
            raise SSLContractError(
                "collateral peer-note index is out of range"
            )
        if any(
            index >= track_count
            for index in track_collateral.local_node_indices
        ):
            raise SSLContractError("collateral track index is out of range")
        selected_owner_tracks = {
            owner_track_by_note[
                note_ptr[sample_index] + local_index
            ]
            - track_ptr[sample_index]
            for local_index in plan.selected_local_node_indices
        }
        expected_track_collateral = tuple(
            local_track_index
            for local_track_index in range(track_count)
            if local_track_index in selected_owner_tracks
        )
        if (
            track_collateral.local_node_indices
            != expected_track_collateral
        ):
            raise SSLContractError(
                "collateral track mask differs from selected-note ownership"
            )
        selected_set = set(plan.selected_local_node_indices)
        expected_owner_globals = {
            track_ptr[sample_index] + local_index
            for local_index in expected_track_collateral
        }
        expected_peer_collateral = tuple(
            local_index
            for local_index in range(note_count)
            if local_index not in selected_set
            and owner_track_by_note[
                note_ptr[sample_index] + local_index
            ]
            in expected_owner_globals
        )
        if (
            peer_collateral.local_node_indices
            != expected_peer_collateral
        ):
            raise SSLContractError(
                "peer-relative mask differs from selected-note ownership"
            )
        primary_global_by_sample.append(
            tuple(
                note_ptr[sample_index] + index
                for index in plan.selected_local_node_indices
            )
        )
        peer_global_by_sample.append(
            tuple(
                note_ptr[sample_index] + index
                for index in peer_collateral.local_node_indices
            )
        )
        collateral_global_by_sample.append(
            tuple(
                track_ptr[sample_index] + index
                for index in track_collateral.local_node_indices
            )
        )

    primary_indices = tuple(
        index for sample in primary_global_by_sample for index in sample
    )
    collateral_indices = tuple(
        index for sample in collateral_global_by_sample for index in sample
    )
    peer_indices = tuple(
        index for sample in peer_global_by_sample for index in sample
    )
    slot_masks = tuple(
        FeatureSlotMask(
            role=(
                "primary_with_peer_collateral"
                if field
                in NOTE_PITCH_GROUP.peer_note_collateral_fields
                else "primary"
            ),
            field=field,
            global_node_indices=(
                _merge_sorted_unique_indices(
                    primary_indices,
                    peer_indices,
                )
                if field
                in NOTE_PITCH_GROUP.peer_note_collateral_fields
                else primary_indices
            ),
        )
        for field in NOTE_PITCH_GROUP.primary_fields
    ) + tuple(
        FeatureSlotMask(
            role="collateral",
            field=field,
            global_node_indices=collateral_indices,
        )
        for field in NOTE_PITCH_GROUP.collateral_fields
    )
    node_counts = tuple(
        sorted(
            (
                node_type,
                int(graph[node_type].num_nodes),
            )
            for node_type in graph.node_types
        )
    )
    return FeatureMaskOverlay.create(
        graph_count=graph_count,
        node_counts=node_counts,
        mask_plan_fingerprints=tuple(plan.fingerprint for plan in prepared),
        slot_masks=slot_masks,
    )


def build_masked_feature_overlay(
    graph: HeteroData,
    plans: object | Sequence[object],
    *,
    mask_token: Tensor | None = None,
) -> FeatureMaskOverlay | BoundFeatureMaskOverlay:
    """Convenience builder, optionally binding an externally owned token."""

    overlay = build_feature_mask_overlay(graph, plans)
    return overlay if mask_token is None else overlay.bind(mask_token)


def bind_feature_mask_overlay(
    overlay: FeatureMaskOverlay,
    mask_token: Tensor,
) -> BoundFeatureMaskOverlay:
    """Bind an externally owned token without changing the immutable masks."""

    return overlay.bind(mask_token)


# Concise aliases for callers that prefer the Phase 7A terminology.
MaskedFeatureOverlay = FeatureMaskOverlay
BoundMaskedFeatureOverlay = BoundFeatureMaskOverlay


__all__ = [
    "BoundFeatureMaskOverlay",
    "BoundMaskedFeatureOverlay",
    "FeatureMaskOverlay",
    "FeatureSlotMask",
    "MaskedFeatureOverlay",
    "bind_feature_mask_overlay",
    "build_feature_mask_overlay",
    "build_masked_feature_overlay",
]
