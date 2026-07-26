"""Deterministic raw-edge hierarchy, coarse context, and top-down fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.encoder import (
    ENCODER_OUTPUT_VERSION,
    EncoderOutput,
    MultiScaleEncoderOutput,
)
from music_critic.models.hierarchy_contracts import (
    COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION,
    HIERARCHICAL_ENCODER_OUTPUT_VERSION,
    HIERARCHY_POOLING_CONTRACT_VERSION,
    TOP_DOWN_FUSION_CONTRACT_VERSION,
    HierarchicalBaselineConfig,
)


class HierarchyContractError(ValueError):
    """Raised when raw containment cannot define the Phase 6B hierarchy."""


_OWNERSHIP_RELATIONS = (
    (
        "beat_to_bar",
        ("beat", "belongs_to_bar", "bar"),
        ("bar", "contains_beat", "beat"),
    ),
    (
        "onset_to_bar",
        ("onset", "belongs_to_bar", "bar"),
        ("bar", "contains_onset", "onset"),
    ),
    (
        "note_to_bar",
        ("note", "belongs_to_bar", "bar"),
        ("bar", "contains_note", "note"),
    ),
    (
        "note_to_track",
        ("note", "belongs_to_track", "track"),
        ("track", "contains_note", "note"),
    ),
    (
        "bar_to_song",
        ("bar", "belongs_to_song", "song"),
        ("song", "contains_bar", "bar"),
    ),
    (
        "track_to_song",
        ("track", "belongs_to_song", "song"),
        ("song", "contains_track", "track"),
    ),
)


@dataclass(frozen=True, slots=True)
class _HierarchyRows:
    embeddings: Mapping[str, Tensor]
    batch_membership: Mapping[str, Tensor]


def _hierarchy_relation_stores(
    graph: object,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Read existing hierarchy stores without PyG's store-creation indexing."""

    if not isinstance(graph, HeteroData):
        raise HierarchyContractError("hierarchy.input_type_invalid")
    node_stores = dict(graph.node_items())
    node_types = tuple(node_stores)
    for node_type in MANDATORY_NODE_TYPES:
        if node_type not in node_types:
            raise HierarchyContractError(
                f"hierarchy.node_store_missing:{node_type}"
            )
    edge_stores = dict(graph.edge_items())
    edge_types = tuple(edge_stores)
    result = {}
    for name, ownership_type, containment_type in _OWNERSHIP_RELATIONS:
        tensors = []
        for direction, edge_type in (
            ("ownership", ownership_type),
            ("containment", containment_type),
        ):
            if edge_type not in edge_types:
                raise HierarchyContractError(
                    f"hierarchy.edge_store_missing:{name}:{direction}"
                )
            store = edge_stores[edge_type]
            if "edge_index" not in store:
                raise HierarchyContractError(
                    f"hierarchy.edge_index_missing:{name}:{direction}"
                )
            edge_index = store["edge_index"]
            if not isinstance(edge_index, Tensor):
                raise HierarchyContractError(
                    f"hierarchy.edge_index_type_invalid:{name}:{direction}"
                )
            if edge_index.dtype != torch.long:
                raise HierarchyContractError(
                    f"hierarchy.edge_index_dtype_invalid:{name}:{direction}"
                )
            if edge_index.ndim != 2:
                raise HierarchyContractError(
                    f"hierarchy.edge_index_rank_invalid:{name}:{direction}"
                )
            if edge_index.shape[0] != 2:
                raise HierarchyContractError(
                    f"hierarchy.edge_index_shape_invalid:{name}:{direction}"
                )
            tensors.append(edge_index)
        result[name] = (tensors[0], tensors[1])
    return result


def validate_hierarchy_graph_structure(graph: object) -> None:
    """Validate mandatory existing stores without scanning ownership values."""

    _hierarchy_relation_stores(graph)


def _validate_membership(
    membership: object,
    *,
    row_count: int,
    node_type: str,
    sample_count: int,
) -> Tensor:
    if (
        not isinstance(membership, Tensor)
        or membership.dtype != torch.long
        or membership.ndim != 1
        or membership.shape[0] != row_count
    ):
        raise HierarchyContractError(
            f"hierarchy.membership_invalid:{node_type}"
        )
    if membership.numel():
        if bool((membership < 0).any()) or bool(
            (membership >= sample_count).any()
        ):
            raise HierarchyContractError(
                f"hierarchy.membership_sample_out_of_range:{node_type}"
            )
        if membership.numel() > 1 and bool(
            (membership[1:] < membership[:-1]).any()
        ):
            raise HierarchyContractError(
                f"hierarchy.membership_not_monotonic:{node_type}"
            )
    return membership


def _owner_index(
    local: EncoderOutput | _HierarchyRows,
    *,
    name: str,
    reverse_edge_type: tuple[str, str, str],
    ownership_edge_index: Tensor,
    containment_edge_index: Tensor,
) -> Tensor:
    child_type, _, parent_type = reverse_edge_type
    reverse = ownership_edge_index
    forward = containment_edge_index
    child_count = local.embeddings[child_type].shape[0]
    parent_count = local.embeddings[parent_type].shape[0]
    expected_device = local.embeddings[child_type].device
    if (
        reverse.device != expected_device
        or forward.device != expected_device
        or local.embeddings[parent_type].device != expected_device
    ):
        raise HierarchyContractError(
            f"hierarchy.edge_index_device_mismatch:{name}"
        )
    if reverse.shape[1] < child_count:
        raise HierarchyContractError(f"hierarchy.owner_missing:{name}")
    if reverse.shape[1] > child_count:
        raise HierarchyContractError(f"hierarchy.owner_duplicate:{name}")
    child_rows = reverse[0]
    if child_rows.numel() and (
        bool((child_rows < 0).any())
        or bool((child_rows >= child_count).any())
    ):
        raise HierarchyContractError(
            f"hierarchy.child_out_of_range:{name}"
        )
    child_counts = torch.bincount(
        child_rows, minlength=child_count
    )
    if bool((child_counts > 1).any()):
        raise HierarchyContractError(f"hierarchy.owner_duplicate:{name}")
    if bool((child_counts == 0).any()):
        raise HierarchyContractError(f"hierarchy.owner_missing:{name}")
    expected_children = torch.arange(
        child_count, dtype=torch.long, device=reverse.device
    )
    if not torch.equal(child_rows, expected_children):
        raise HierarchyContractError(f"hierarchy.child_reordered:{name}")
    if not torch.equal(forward, reverse.flip(0)):
        raise HierarchyContractError(
            f"hierarchy.reverse_containment_mismatch:{name}"
        )
    owners = reverse[1]
    if owners.numel() and (
        bool((owners < 0).any()) or bool((owners >= parent_count).any())
    ):
        raise HierarchyContractError(
            f"hierarchy.owner_out_of_range:{name}"
        )
    child_membership = local.batch_membership[child_type]
    parent_membership = local.batch_membership[parent_type]
    if owners.numel() and not torch.equal(
        child_membership,
        parent_membership.index_select(0, owners),
    ):
        raise HierarchyContractError(
            f"hierarchy.cross_sample_ownership:{name}"
        )
    return owners


@dataclass(frozen=True, slots=True)
class HierarchyOwnership:
    """Global parent row for each child row, derived only from raw edges."""

    contract_version: str
    sample_count: int
    owners: Mapping[str, Tensor]
    batch_membership: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.contract_version != HIERARCHY_POOLING_CONTRACT_VERSION:
            raise HierarchyContractError(
                "hierarchy.ownership_version_incompatible"
            )
        if tuple(self.owners) != tuple(
            item[0] for item in _OWNERSHIP_RELATIONS
        ):
            raise HierarchyContractError(
                "hierarchy.ownership_keys_incomplete"
            )
        if tuple(self.batch_membership) != MANDATORY_NODE_TYPES:
            raise HierarchyContractError(
                "hierarchy.ownership_membership_keys_incomplete"
            )


def extract_hierarchy_ownership(
    graph: object,
    local: EncoderOutput | None = None,
) -> HierarchyOwnership:
    """Validate deterministic one-owner containment without regrouping rows."""

    relation_tensors = _hierarchy_relation_stores(graph)
    if local is None:
        stores = dict(graph.node_items())
        for node_type, store in stores.items():
            if "x_cont" not in store:
                raise HierarchyContractError(
                    f"hierarchy.node_rows_missing:{node_type}"
                )
            if (
                not isinstance(store["x_cont"], Tensor)
                or store["x_cont"].ndim != 2
            ):
                raise HierarchyContractError(
                    f"hierarchy.node_rows_invalid:{node_type}"
                )
        embeddings = {
            node_type: stores[node_type]["x_cont"]
            for node_type in MANDATORY_NODE_TYPES
        }
        batch_membership = {
            node_type: (
                stores[node_type]["batch"]
                if "batch" in stores[node_type]
                else torch.zeros(
                    embeddings[node_type].shape[0],
                    dtype=torch.long,
                    device=embeddings[node_type].device,
                )
            )
            for node_type in MANDATORY_NODE_TYPES
        }
    else:
        if not isinstance(local, EncoderOutput):
            raise HierarchyContractError(
                "hierarchy.local_output_type_invalid"
            )
        embeddings = local.embeddings
        batch_membership = local.batch_membership
    if tuple(embeddings) != MANDATORY_NODE_TYPES:
        raise HierarchyContractError("hierarchy.local_node_types_incomplete")
    song_count = embeddings["song"].shape[0]
    song_membership = batch_membership["song"]
    if (
        song_membership.dtype != torch.long
        or song_membership.ndim != 1
        or song_membership.shape[0] != song_count
        or not torch.equal(
            song_membership,
            torch.arange(
                song_count,
                dtype=torch.long,
                device=song_membership.device,
            ),
        )
    ):
        raise HierarchyContractError(
            "hierarchy.song_rows_must_match_samples"
        )
    membership = {}
    for node_type in MANDATORY_NODE_TYPES:
        values = embeddings[node_type]
        membership[node_type] = _validate_membership(
            batch_membership[node_type],
            row_count=values.shape[0],
            node_type=node_type,
            sample_count=song_count,
        )
    row_view = _HierarchyRows(
        embeddings=embeddings,
        batch_membership=membership,
    )
    owners = {
        name: _owner_index(
            row_view,
            name=name,
            reverse_edge_type=reverse,
            ownership_edge_index=relation_tensors[name][0],
            containment_edge_index=relation_tensors[name][1],
        )
        for name, reverse, _forward in _OWNERSHIP_RELATIONS
    }
    return HierarchyOwnership(
        contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
        sample_count=song_count,
        owners=owners,
        batch_membership=membership,
    )


def _validate_ownership_local_contract(
    local: EncoderOutput,
    ownership: HierarchyOwnership,
) -> HierarchyOwnership:
    """Validate a precomputed object completely against retained local rows."""

    if not isinstance(ownership, HierarchyOwnership):
        raise HierarchyContractError(
            "hierarchy.precomputed_ownership_type_invalid"
        )
    if ownership.contract_version != HIERARCHY_POOLING_CONTRACT_VERSION:
        raise HierarchyContractError(
            "hierarchy.ownership_version_incompatible"
        )
    expected_names = tuple(item[0] for item in _OWNERSHIP_RELATIONS)
    if tuple(ownership.owners) != expected_names:
        raise HierarchyContractError(
            "hierarchy.ownership_keys_incomplete"
        )
    if tuple(ownership.batch_membership) != MANDATORY_NODE_TYPES:
        raise HierarchyContractError(
            "hierarchy.ownership_membership_keys_incomplete"
        )
    expected_sample_count = local.embeddings["song"].shape[0]
    if (
        isinstance(ownership.sample_count, bool)
        or not isinstance(ownership.sample_count, int)
        or ownership.sample_count != expected_sample_count
    ):
        raise HierarchyContractError(
            "hierarchy.ownership_sample_count_mismatch"
        )
    relation_by_name = {
        name: reverse
        for name, reverse, _forward in _OWNERSHIP_RELATIONS
    }
    for name in expected_names:
        child_type, _, parent_type = relation_by_name[name]
        owners = ownership.owners[name]
        child_count = local.embeddings[child_type].shape[0]
        parent_count = local.embeddings[parent_type].shape[0]
        if not isinstance(owners, Tensor):
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_type_invalid:{name}"
            )
        if owners.dtype != torch.long:
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_dtype_invalid:{name}"
            )
        if owners.ndim != 1:
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_rank_invalid:{name}"
            )
        if owners.shape != (child_count,):
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_shape_invalid:{name}"
            )
        if owners.device != local.embeddings[child_type].device:
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_device_mismatch:{name}"
            )
        if owners.numel() and (
            bool((owners < 0).any())
            or bool((owners >= parent_count).any())
        ):
            raise HierarchyContractError(
                f"hierarchy.ownership_owner_out_of_range:{name}"
            )
        if owners.numel() and not torch.equal(
            local.batch_membership[child_type],
            local.batch_membership[parent_type].index_select(0, owners),
        ):
            raise HierarchyContractError(
                f"hierarchy.ownership_cross_sample:{name}"
            )
    for node_type in MANDATORY_NODE_TYPES:
        membership = ownership.batch_membership[node_type]
        expected_membership = local.batch_membership[node_type]
        if not isinstance(membership, Tensor):
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_type_invalid:{node_type}"
            )
        if membership.dtype != torch.long:
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_dtype_invalid:{node_type}"
            )
        if membership.ndim != 1:
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_rank_invalid:{node_type}"
            )
        if membership.shape != expected_membership.shape:
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_shape_invalid:{node_type}"
            )
        if membership.device != expected_membership.device:
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_device_mismatch:{node_type}"
            )
        if not torch.equal(membership, expected_membership):
            raise HierarchyContractError(
                f"hierarchy.ownership_membership_mismatch:{node_type}"
            )
    return ownership


def validate_hierarchy_ownership(
    graph: object,
    local: EncoderOutput,
    ownership: HierarchyOwnership,
) -> HierarchyOwnership:
    """Validate externally supplied ownership against raw graph and local rows."""

    _validate_ownership_local_contract(local, ownership)
    expected_names = tuple(item[0] for item in _OWNERSHIP_RELATIONS)
    expected = extract_hierarchy_ownership(graph, local)
    for name in expected_names:
        if not torch.equal(
            ownership.owners[name], expected.owners[name]
        ):
            raise HierarchyContractError(
                f"hierarchy.ownership_graph_mismatch:{name}"
            )
    return ownership


def _scatter_family_statistics(
    children: Tensor,
    owners: Tensor,
    parent_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hidden_dim = children.shape[1]
    sums = children.new_zeros((parent_count, hidden_dim))
    counts = torch.zeros(
        parent_count, dtype=torch.long, device=children.device
    )
    if children.shape[0]:
        sums.index_add_(0, owners, children)
        counts.index_add_(
            0,
            owners,
            torch.ones_like(owners, dtype=torch.long),
        )
    available = counts > 0
    means = sums / counts.clamp_min(1).to(children.dtype).unsqueeze(-1)
    maxima = children.new_full((parent_count, hidden_dim), -torch.inf)
    if children.shape[0]:
        maxima.scatter_reduce_(
            0,
            owners[:, None].expand(-1, hidden_dim),
            children,
            reduce="amax",
            include_self=True,
        )
    maxima = torch.where(available[:, None], maxima, torch.zeros_like(maxima))
    return means, maxima, counts, available


class _FamilyPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.availability_embedding = nn.Embedding(2, hidden_dim)
        self.projection = nn.Linear(hidden_dim * 3 + 1, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        children: Tensor,
        owners: Tensor,
        parent_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        means, maxima, counts, available = _scatter_family_statistics(
            children, owners, parent_count
        )
        evidence = torch.cat(
            (
                means,
                maxima,
                torch.log1p(counts.to(children.dtype)).unsqueeze(-1),
                self.availability_embedding(available.long()),
            ),
            dim=-1,
        )
        return (
            self.dropout(
                self.activation(
                    self.normalization(self.projection(evidence))
                )
            ),
            counts,
            available,
        )


class _ParentTokenBuilder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        child_family_count: int,
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(
            hidden_dim * (child_family_count + 1), hidden_dim
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, parent: Tensor, family_contexts: tuple[Tensor, ...]
    ) -> Tensor:
        context = self.context_projection(
            torch.cat((parent, *family_contexts), dim=-1)
        )
        return self.dropout(
            self.activation(self.normalization(parent + context))
        )


@dataclass(frozen=True, slots=True)
class HierarchyPoolingOutput:
    """Bar/track tokens plus explicit child-count and availability evidence."""

    contract_version: str
    bar_tokens: Tensor
    track_tokens: Tensor
    child_counts: Mapping[str, Tensor]
    child_available: Mapping[str, Tensor]
    ownership: HierarchyOwnership

    def __post_init__(self) -> None:
        if self.contract_version != HIERARCHY_POOLING_CONTRACT_VERSION:
            raise HierarchyContractError(
                "hierarchy.pooling_version_incompatible"
            )
        expected = ("bar_beats", "bar_onsets", "bar_notes", "track_notes")
        if tuple(self.child_counts) != expected or tuple(
            self.child_available
        ) != expected:
            raise HierarchyContractError(
                "hierarchy.pooling_evidence_incomplete"
            )
        for name in expected:
            count = self.child_counts[name]
            available = self.child_available[name]
            if (
                count.dtype != torch.long
                or count.ndim != 1
                or available.dtype != torch.bool
                or available.shape != count.shape
                or not torch.equal(available, count > 0)
            ):
                raise HierarchyContractError(
                    f"hierarchy.pooling_evidence_invalid:{name}"
                )


class DeterministicHierarchyPool(nn.Module):
    """Vectorized family-aware bar/track pooling over raw containment."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.bar_beat = _FamilyPool(hidden_dim, dropout)
        self.bar_onset = _FamilyPool(hidden_dim, dropout)
        self.bar_note = _FamilyPool(hidden_dim, dropout)
        self.track_note = _FamilyPool(hidden_dim, dropout)
        self.bar_builder = _ParentTokenBuilder(hidden_dim, dropout, 3)
        self.track_builder = _ParentTokenBuilder(hidden_dim, dropout, 1)

    def forward(
        self,
        local: EncoderOutput,
        ownership: HierarchyOwnership,
    ) -> HierarchyPoolingOutput:
        bar_count = local.embeddings["bar"].shape[0]
        track_count = local.embeddings["track"].shape[0]
        beat_context, beat_count, beat_available = self.bar_beat(
            local.embeddings["beat"],
            ownership.owners["beat_to_bar"],
            bar_count,
        )
        onset_context, onset_count, onset_available = self.bar_onset(
            local.embeddings["onset"],
            ownership.owners["onset_to_bar"],
            bar_count,
        )
        bar_note_context, bar_note_count, bar_note_available = self.bar_note(
            local.embeddings["note"],
            ownership.owners["note_to_bar"],
            bar_count,
        )
        track_note_context, track_note_count, track_note_available = (
            self.track_note(
                local.embeddings["note"],
                ownership.owners["note_to_track"],
                track_count,
            )
        )
        return HierarchyPoolingOutput(
            contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
            bar_tokens=self.bar_builder(
                local.embeddings["bar"],
                (beat_context, onset_context, bar_note_context),
            ),
            track_tokens=self.track_builder(
                local.embeddings["track"], (track_note_context,)
            ),
            child_counts={
                "bar_beats": beat_count,
                "bar_onsets": onset_count,
                "bar_notes": bar_note_count,
                "track_notes": track_note_count,
            },
            child_available={
                "bar_beats": beat_available,
                "bar_onsets": onset_available,
                "bar_notes": bar_note_available,
                "track_notes": track_note_available,
            },
            ownership=ownership,
        )


def _coarse_family_layout(
    membership: Tensor,
    sample_count: int,
    *,
    position_offset_by_sample: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return counts, within-family ordinals, and padded positions."""

    counts = torch.bincount(membership, minlength=sample_count)
    starts = torch.cumsum(counts, dim=0) - counts
    ordinals = (
        torch.arange(
            membership.shape[0],
            dtype=torch.long,
            device=membership.device,
        )
        - starts.index_select(0, membership)
    )
    positions = (
        position_offset_by_sample.index_select(0, membership) + ordinals
    )
    return counts, ordinals, positions


def _maximum_padded_length(lengths: Tensor) -> int:
    """Make the sole device-to-host synchronization needed for allocation."""

    return int(lengths.max().item()) if lengths.numel() else 0


def _sinusoidal_position(
    ordinals: Tensor, hidden_dim: int, dtype: torch.dtype
) -> Tensor:
    if ordinals.numel() == 0:
        return torch.empty(
            (0, hidden_dim), dtype=dtype, device=ordinals.device
        )
    positions = ordinals.to(dtype).unsqueeze(-1)
    even_width = (hidden_dim + 1) // 2
    exponent = torch.arange(
        even_width, dtype=dtype, device=ordinals.device
    )
    denominator = torch.pow(
        torch.tensor(10_000.0, dtype=dtype, device=ordinals.device),
        (2 * exponent) / hidden_dim,
    )
    angles = positions / denominator
    encoded = torch.zeros(
        (ordinals.shape[0], hidden_dim),
        dtype=dtype,
        device=ordinals.device,
    )
    encoded[:, 0::2] = torch.sin(angles[:, : encoded[:, 0::2].shape[1]])
    encoded[:, 1::2] = torch.cos(angles[:, : encoded[:, 1::2].shape[1]])
    return encoded


@dataclass(frozen=True, slots=True)
class CoarseTokenSequence:
    """Padded per-sample [SONG]+bars+tracks token contract."""

    contract_version: str
    tokens: Tensor
    padding_mask: Tensor
    type_codes: Tensor
    ordinals: Tensor
    sequence_lengths: Tensor
    bar_positions: Tensor
    track_positions: Tensor

    def __post_init__(self) -> None:
        if self.contract_version != COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION:
            raise HierarchyContractError(
                "hierarchy.sequence_version_incompatible"
            )
        if (
            self.tokens.ndim != 3
            or self.padding_mask.dtype != torch.bool
            or self.padding_mask.shape != self.tokens.shape[:2]
            or self.type_codes.dtype != torch.long
            or self.type_codes.shape != self.padding_mask.shape
            or self.ordinals.dtype != torch.long
            or self.ordinals.shape != self.padding_mask.shape
            or self.sequence_lengths.dtype != torch.long
            or self.sequence_lengths.shape != (self.tokens.shape[0],)
        ):
            raise HierarchyContractError(
                "hierarchy.sequence_tensors_inconsistent"
            )


@dataclass(frozen=True, slots=True)
class ContextualCoarseOutput:
    """Contextual SONG, bar, and track rows in original deterministic order."""

    contract_version: str
    song_embeddings: Tensor
    bar_embeddings: Tensor
    track_embeddings: Tensor
    sequence: CoarseTokenSequence

    def __post_init__(self) -> None:
        if self.contract_version != COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION:
            raise HierarchyContractError(
                "hierarchy.context_version_incompatible"
            )


class CoarseMusicTransformer(nn.Module):
    """Batch-first pre-norm Transformer over isolated coarse sequences."""

    def __init__(self, config: HierarchicalBaselineConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.type_embedding = nn.Embedding(3, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * config.ffn_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def build_sequence(
        self,
        local: EncoderOutput,
        pooling: HierarchyPoolingOutput,
    ) -> CoarseTokenSequence:
        ownership = pooling.ownership
        sample_count = ownership.sample_count
        bar_membership = ownership.batch_membership["bar"]
        track_membership = ownership.batch_membership["track"]
        unit_offsets = torch.ones(
            sample_count,
            dtype=torch.long,
            device=pooling.bar_tokens.device,
        )
        bar_counts, bar_ordinals, bar_positions = _coarse_family_layout(
            bar_membership,
            sample_count,
            position_offset_by_sample=unit_offsets,
        )
        track_counts, track_ordinals, track_positions = (
            _coarse_family_layout(
                track_membership,
                sample_count,
                position_offset_by_sample=unit_offsets + bar_counts,
            )
        )
        lengths = unit_offsets + bar_counts + track_counts
        max_length = _maximum_padded_length(lengths)
        tokens = pooling.bar_tokens.new_zeros(
            (sample_count, max_length, self.hidden_dim)
        )
        column_indices = torch.arange(
            max_length, dtype=torch.long, device=tokens.device
        )
        padding = column_indices.unsqueeze(0) >= lengths.unsqueeze(1)
        type_codes = torch.full(
            (sample_count, max_length),
            -1,
            dtype=torch.long,
            device=tokens.device,
        )
        ordinals = torch.full_like(type_codes, -1)
        sample_indices = torch.arange(
            sample_count, dtype=torch.long, device=tokens.device
        )
        tokens[sample_indices, 0] = local.embeddings["song"]
        tokens[bar_membership, bar_positions] = pooling.bar_tokens
        tokens[track_membership, track_positions] = pooling.track_tokens
        type_codes[sample_indices, 0] = 0
        type_codes[bar_membership, bar_positions] = 1
        type_codes[track_membership, track_positions] = 2
        ordinals[sample_indices, 0] = 0
        ordinals[bar_membership, bar_positions] = bar_ordinals
        ordinals[track_membership, track_positions] = track_ordinals
        tokens = tokens + self.type_embedding(type_codes.clamp_min(0))
        positional_rows = ~padding & (type_codes != 0)
        tokens = tokens + torch.where(
            positional_rows.unsqueeze(-1),
            _sinusoidal_position(
                ordinals.clamp_min(0).reshape(-1),
                self.hidden_dim,
                tokens.dtype,
            ).reshape_as(tokens),
            torch.zeros_like(tokens),
        )
        tokens = torch.where(
            padding.unsqueeze(-1), torch.zeros_like(tokens), tokens
        )
        return CoarseTokenSequence(
            contract_version=COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION,
            tokens=tokens,
            padding_mask=padding,
            type_codes=type_codes,
            ordinals=ordinals,
            sequence_lengths=lengths,
            bar_positions=bar_positions,
            track_positions=track_positions,
        )

    def forward(
        self,
        local: EncoderOutput,
        pooling: HierarchyPoolingOutput,
    ) -> ContextualCoarseOutput:
        sequence = self.build_sequence(local, pooling)
        contextual = self.final_norm(
            self.encoder(
                sequence.tokens,
                src_key_padding_mask=sequence.padding_mask,
            )
        )
        song = contextual[:, 0]
        bar_sample = pooling.ownership.batch_membership["bar"]
        track_sample = pooling.ownership.batch_membership["track"]
        bars = contextual[
            bar_sample,
            sequence.bar_positions,
        ]
        tracks = contextual[
            track_sample,
            sequence.track_positions,
        ]
        return ContextualCoarseOutput(
            contract_version=COARSE_TOKEN_SEQUENCE_CONTRACT_VERSION,
            song_embeddings=song,
            bar_embeddings=bars,
            track_embeddings=tracks,
            sequence=sequence,
        )


class _ResidualContextFusion(nn.Module):
    def __init__(
        self, hidden_dim: int, context_count: int, dropout: float
    ) -> None:
        super().__init__()
        self.context_projection = nn.Linear(
            hidden_dim * context_count, hidden_dim
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, local: Tensor, contexts: tuple[Tensor, ...]) -> Tensor:
        context = self.context_projection(torch.cat(contexts, dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat((local, context), dim=-1)))
        return self.normalization(local + self.dropout(gate * context))


class TopDownFusion(nn.Module):
    """Return contextual bar/track/song evidence to every original row."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fusions = nn.ModuleDict(
            {
                "song": _ResidualContextFusion(hidden_dim, 1, dropout),
                "track": _ResidualContextFusion(hidden_dim, 2, dropout),
                "bar": _ResidualContextFusion(hidden_dim, 2, dropout),
                "beat": _ResidualContextFusion(hidden_dim, 2, dropout),
                "onset": _ResidualContextFusion(hidden_dim, 2, dropout),
                "note": _ResidualContextFusion(hidden_dim, 3, dropout),
            }
        )

    def forward(
        self,
        local: EncoderOutput,
        coarse: ContextualCoarseOutput,
        ownership: HierarchyOwnership,
    ) -> EncoderOutput:
        owner = ownership.owners
        song_by_node = {
            node_type: coarse.song_embeddings.index_select(
                0, ownership.batch_membership[node_type]
            )
            for node_type in MANDATORY_NODE_TYPES
        }
        embeddings = {
            "song": self.fusions["song"](
                local.embeddings["song"], (coarse.song_embeddings,)
            ),
            "track": self.fusions["track"](
                local.embeddings["track"],
                (coarse.track_embeddings, song_by_node["track"]),
            ),
            "bar": self.fusions["bar"](
                local.embeddings["bar"],
                (coarse.bar_embeddings, song_by_node["bar"]),
            ),
            "beat": self.fusions["beat"](
                local.embeddings["beat"],
                (
                    coarse.bar_embeddings.index_select(
                        0, owner["beat_to_bar"]
                    ),
                    song_by_node["beat"],
                ),
            ),
            "onset": self.fusions["onset"](
                local.embeddings["onset"],
                (
                    coarse.bar_embeddings.index_select(
                        0, owner["onset_to_bar"]
                    ),
                    song_by_node["onset"],
                ),
            ),
            "note": self.fusions["note"](
                local.embeddings["note"],
                (
                    coarse.bar_embeddings.index_select(
                        0, owner["note_to_bar"]
                    ),
                    coarse.track_embeddings.index_select(
                        0, owner["note_to_track"]
                    ),
                    song_by_node["note"],
                ),
            ),
        }
        return EncoderOutput(
            contract_version=ENCODER_OUTPUT_VERSION,
            embeddings=embeddings,
            batch_membership=local.batch_membership,
        )


@dataclass(frozen=True, slots=True)
class ContextualEncoderOutput:
    """Retained Phase 6A local rows plus coarse and fused Phase 6B rows."""

    contract_version: str
    fusion_contract_version: str
    local_encoder: MultiScaleEncoderOutput
    pooling: HierarchyPoolingOutput
    coarse: ContextualCoarseOutput
    fused: EncoderOutput

    def __post_init__(self) -> None:
        if (
            self.contract_version != HIERARCHICAL_ENCODER_OUTPUT_VERSION
            or self.fusion_contract_version
            != TOP_DOWN_FUSION_CONTRACT_VERSION
        ):
            raise HierarchyContractError(
                "hierarchy.encoder_output_version_incompatible"
            )
        local = self.local_encoder.final_output
        for node_type in MANDATORY_NODE_TYPES:
            if (
                self.fused.embeddings[node_type].shape
                != local.embeddings[node_type].shape
                or not torch.equal(
                    self.fused.batch_membership[node_type],
                    local.batch_membership[node_type],
                )
            ):
                raise HierarchyContractError(
                    f"hierarchy.fusion_changed_rows:{node_type}"
                )


class HierarchicalContextEncoder(nn.Module):
    """Compose pooling, Transformer, and top-down fusion over Phase 6A rows."""

    def __init__(self, config: HierarchicalBaselineConfig) -> None:
        super().__init__()
        self.pooling = DeterministicHierarchyPool(
            config.hidden_dim, config.dropout
        )
        self.transformer = CoarseMusicTransformer(config)
        self.fusion = TopDownFusion(config.hidden_dim, config.dropout)

    def forward(
        self,
        local_encoder: MultiScaleEncoderOutput,
        graph: object,
        *,
        ownership: HierarchyOwnership | None = None,
    ) -> ContextualEncoderOutput:
        local = local_encoder.final_output
        ownership = (
            extract_hierarchy_ownership(graph, local)
            if ownership is None
            else validate_hierarchy_ownership(graph, local, ownership)
        )
        return self._encode_with_ownership(local_encoder, ownership)

    def _forward_with_extracted_ownership(
        self,
        local_encoder: MultiScaleEncoderOutput,
        ownership: HierarchyOwnership,
    ) -> ContextualEncoderOutput:
        """Internal single-scan path for ownership extracted before Phase 6A."""

        _validate_ownership_local_contract(
            local_encoder.final_output, ownership
        )
        return self._encode_with_ownership(local_encoder, ownership)

    def _encode_with_ownership(
        self,
        local_encoder: MultiScaleEncoderOutput,
        ownership: HierarchyOwnership,
    ) -> ContextualEncoderOutput:
        local = local_encoder.final_output
        pooling = self.pooling(local, ownership)
        coarse = self.transformer(local, pooling)
        fused = self.fusion(local, coarse, ownership)
        return ContextualEncoderOutput(
            contract_version=HIERARCHICAL_ENCODER_OUTPUT_VERSION,
            fusion_contract_version=TOP_DOWN_FUSION_CONTRACT_VERSION,
            local_encoder=local_encoder,
            pooling=pooling,
            coarse=coarse,
            fused=fused,
        )


__all__ = [
    "CoarseMusicTransformer",
    "CoarseTokenSequence",
    "ContextualCoarseOutput",
    "ContextualEncoderOutput",
    "DeterministicHierarchyPool",
    "HierarchicalContextEncoder",
    "HierarchyContractError",
    "HierarchyOwnership",
    "HierarchyPoolingOutput",
    "TopDownFusion",
    "extract_hierarchy_ownership",
    "validate_hierarchy_graph_structure",
    "validate_hierarchy_ownership",
]
