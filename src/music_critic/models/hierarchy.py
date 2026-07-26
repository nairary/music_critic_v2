"""Deterministic raw-edge hierarchy, coarse context, and top-down fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

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
    graph: object,
    local: EncoderOutput | _HierarchyRows,
    *,
    name: str,
    reverse_edge_type: tuple[str, str, str],
    forward_edge_type: tuple[str, str, str],
) -> Tensor:
    child_type, _, parent_type = reverse_edge_type
    reverse = graph[reverse_edge_type].edge_index
    forward = graph[forward_edge_type].edge_index
    child_count = local.embeddings[child_type].shape[0]
    parent_count = local.embeddings[parent_type].shape[0]
    if (
        not isinstance(reverse, Tensor)
        or reverse.dtype != torch.long
        or reverse.ndim != 2
        or reverse.shape[0] != 2
        or not isinstance(forward, Tensor)
        or forward.dtype != torch.long
        or forward.ndim != 2
        or forward.shape[0] != 2
    ):
        raise HierarchyContractError(
            f"hierarchy.edge_index_invalid:{name}"
        )
    expected_children = torch.arange(
        child_count, dtype=torch.long, device=reverse.device
    )
    if (
        reverse.shape[1] != child_count
        or not torch.equal(reverse[0], expected_children)
    ):
        raise HierarchyContractError(
            f"hierarchy.owner_missing_duplicate_or_unordered:{name}"
        )
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


def extract_hierarchy_ownership(
    graph: object,
    local: EncoderOutput | None = None,
) -> HierarchyOwnership:
    """Validate deterministic one-owner containment without regrouping rows."""

    if local is None:
        embeddings = {
            node_type: graph[node_type].x_cont
            for node_type in MANDATORY_NODE_TYPES
        }
        batch_membership = {
            node_type: (
                graph[node_type].batch
                if hasattr(graph[node_type], "batch")
                else torch.zeros(
                    int(graph[node_type].num_nodes),
                    dtype=torch.long,
                    device=graph[node_type].x_cont.device,
                )
            )
            for node_type in MANDATORY_NODE_TYPES
        }
    else:
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
            graph,
            row_view,
            name=name,
            reverse_edge_type=reverse,
            forward_edge_type=forward,
        )
        for name, reverse, forward in _OWNERSHIP_RELATIONS
    }
    return HierarchyOwnership(
        contract_version=HIERARCHY_POOLING_CONTRACT_VERSION,
        sample_count=song_count,
        owners=owners,
        batch_membership=membership,
    )


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


def _sample_boundaries(
    membership: Tensor, sample_count: int
) -> tuple[int, ...]:
    boundaries = [0]
    cursor = 0
    for sample_index in range(sample_count):
        while (
            cursor < membership.shape[0]
            and int(membership[cursor].item()) == sample_index
        ):
            cursor += 1
        boundaries.append(cursor)
    if cursor != membership.shape[0]:
        raise HierarchyContractError(
            "hierarchy.membership_boundaries_inconsistent"
        )
    return tuple(boundaries)


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
        bar_boundaries = _sample_boundaries(
            ownership.batch_membership["bar"], sample_count
        )
        track_boundaries = _sample_boundaries(
            ownership.batch_membership["track"], sample_count
        )
        lengths = torch.tensor(
            [
                1
                + bar_boundaries[index + 1]
                - bar_boundaries[index]
                + track_boundaries[index + 1]
                - track_boundaries[index]
                for index in range(sample_count)
            ],
            dtype=torch.long,
            device=pooling.bar_tokens.device,
        )
        max_length = int(lengths.max().item()) if sample_count else 0
        tokens = pooling.bar_tokens.new_zeros(
            (sample_count, max_length, self.hidden_dim)
        )
        padding = torch.ones(
            (sample_count, max_length),
            dtype=torch.bool,
            device=tokens.device,
        )
        type_codes = torch.full(
            (sample_count, max_length),
            -1,
            dtype=torch.long,
            device=tokens.device,
        )
        ordinals = torch.full_like(type_codes, -1)
        bar_positions = torch.empty(
            pooling.bar_tokens.shape[0],
            dtype=torch.long,
            device=tokens.device,
        )
        track_positions = torch.empty(
            pooling.track_tokens.shape[0],
            dtype=torch.long,
            device=tokens.device,
        )
        for sample_index in range(sample_count):
            bar_start, bar_end = (
                bar_boundaries[sample_index],
                bar_boundaries[sample_index + 1],
            )
            track_start, track_end = (
                track_boundaries[sample_index],
                track_boundaries[sample_index + 1],
            )
            bar_count = bar_end - bar_start
            track_count = track_end - track_start
            length = 1 + bar_count + track_count
            tokens[sample_index, 0] = local.embeddings["song"][sample_index]
            if bar_count:
                tokens[sample_index, 1 : 1 + bar_count] = (
                    pooling.bar_tokens[bar_start:bar_end]
                )
                bar_positions[bar_start:bar_end] = torch.arange(
                    1, 1 + bar_count, device=tokens.device
                )
                ordinals[sample_index, 1 : 1 + bar_count] = torch.arange(
                    bar_count, device=tokens.device
                )
            if track_count:
                track_offset = 1 + bar_count
                tokens[
                    sample_index,
                    track_offset : track_offset + track_count,
                ] = pooling.track_tokens[track_start:track_end]
                track_positions[track_start:track_end] = torch.arange(
                    track_offset,
                    track_offset + track_count,
                    device=tokens.device,
                )
                ordinals[
                    sample_index,
                    track_offset : track_offset + track_count,
                ] = torch.arange(track_count, device=tokens.device)
            type_codes[sample_index, 0] = 0
            type_codes[sample_index, 1 : 1 + bar_count] = 1
            type_codes[
                sample_index,
                1 + bar_count : length,
            ] = 2
            ordinals[sample_index, 0] = 0
            padding[sample_index, :length] = False
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
        ownership = ownership or extract_hierarchy_ownership(graph, local)
        for node_type in MANDATORY_NODE_TYPES:
            if not torch.equal(
                ownership.batch_membership[node_type],
                local.batch_membership[node_type],
            ):
                raise HierarchyContractError(
                    f"hierarchy.local_membership_mismatch:{node_type}"
                )
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
]
