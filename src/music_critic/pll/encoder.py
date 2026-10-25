"""Differentiable, tiled C3 export for masked conditional prediction.

The old frozen critic adapter is deliberately untouched. No target or theory
head is used here; callers supply an already sanitized ObservedView.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch
from torch import Tensor, nn

from music_critic.critic.c3_backbone import (
    C3_ADAPTER_VERSION,
    C3ThreePassRelationGateModel,
    load_c3_relationgate_adapter,
)
from music_critic.critic.features import expected_stages
from music_critic.experiments.analysisgnn.temporal_slices.graph import NODES, TemporalGraphBatch

if TYPE_CHECKING:
    from .contracts import ContextProfile
    from .observed_view import ObservedTile, ObservedMusicView


@dataclass(frozen=True)
class EncodedTile:
    tile: ObservedTile
    graph: TemporalGraphBatch
    embeddings: Mapping[str, Tensor]
    stages: Mapping[str, str]


class PLLContextEncoder(nn.Module):
    """Shared C3 over independent tiles, with explicit freezing and provenance.

    ``frozen`` freezes all inherited C3 weights; ``upper`` trains its hierarchy,
    onset decoder and temporal passes; ``full`` additionally trains the local
    graph encoder. Historical theory heads remain frozen and unused in all
    modes. New boundary projections are trainable in every mode.
    """

    def __init__(self, profile: ContextProfile | None = None, *,
                 backbone: C3ThreePassRelationGateModel | None = None,
                 freeze_policy: str = "frozen"):
        super().__init__()
        if profile is None:
            from .contracts import ContextProfile
            profile = ContextProfile()
        self.profile = profile
        if backbone is not None and type(backbone) is not C3ThreePassRelationGateModel:
            raise TypeError("PLL requires the selected C3 backbone architecture")
        self.backbone = C3ThreePassRelationGateModel() if backbone is None else backbone
        self.weights_origin = "random_initialization" if backbone is None else "caller_supplied_unverified"
        # left/right censoring and exact duration beyond the tile are separate
        # raw conditions, never synthetic note-start/end events.
        self.boundary_projection = nn.Linear(2, 128, bias=False)
        nn.init.zeros_(self.boundary_projection.weight)
        self.set_freeze_policy(freeze_policy)

    @classmethod
    def from_pretrained(cls, checkpoint_path, *, expected_sha256: str,
                        expected_size_bytes: int, profile: ContextProfile | None = None,
                        freeze_policy: str = "frozen", expected_update: int = 80000):
        """Use the existing byte-bound, weight-only loader; never resume training."""
        adapter = load_c3_relationgate_adapter(
            checkpoint_path, expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes, expected_update=expected_update)
        result = cls(profile, backbone=adapter.model, freeze_policy=freeze_policy)
        result.weights_origin = "sha256:" + expected_sha256
        return result

    def set_freeze_policy(self, policy: str) -> None:
        if policy not in {"frozen", "upper", "full"}:
            raise ValueError("PLL freeze policy must be frozen, upper or full")
        self.freeze_policy = policy
        self.backbone.requires_grad_(policy == "full")
        if policy == "upper":
            self.backbone.base.context_encoder.requires_grad_(True)
            self.backbone.base.sequence_decoder.requires_grad_(True)
            self.backbone.slice_projection.requires_grad_(True)
            self.backbone.temporal_layers.requires_grad_(True)
        self.backbone.base.task_heads.requires_grad_(False)
        # Stale gradients must not survive a transition to a frozen policy.
        for parameter in self.backbone.parameters():
            if not parameter.requires_grad:
                parameter.grad = None
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "freeze_policy"):
            if self.freeze_policy == "frozen":
                self.backbone.eval()
            elif self.freeze_policy == "upper":
                self.backbone.base.local_baseline.eval()
            self.backbone.base.task_heads.eval()
        return self

    def _encode_graph(self, graph: TemporalGraphBatch,
                      boundary_features: Tensor | None = None) -> Mapping[str, Tensor]:
        """Private graph seam, kept separate for synthetic gradient checks."""
        graph.validate()
        base = self.backbone.base.encode(graph.base)
        encoded = base.fused.embeddings
        embeddings = {name: encoded[name] for name in NODES if name != "slice"}
        if boundary_features is not None:
            if boundary_features.shape != (embeddings["note"].shape[0], 2):
                raise ValueError("PLL boundary features must align with note rows")
            embeddings["note"] = embeddings["note"] + self.boundary_projection(boundary_features)
        x = graph.slice_features
        embeddings["slice"] = self.backbone.slice_projection(x / (1 + x.abs()))
        for layer in self.backbone.temporal_layers:
            embeddings = layer(embeddings, graph.edges)
        return {
            "note_local": base.local_encoder.final_output.embeddings["note"],
            "note": embeddings["note"], "onset": embeddings["onset"],
            "beat": embeddings["beat"], "bar": embeddings["bar"],
            "track": encoded["track"], "song": encoded["song"],
            "slice": embeddings["slice"],
        }

    def forward(self, view: ObservedMusicView) -> tuple[EncodedTile, ...]:
        from .graph import build_tile_graph
        from .masking import build_observed_tiles
        device = next(self.parameters()).device
        result = []
        for tile in build_observed_tiles(view, self.profile):
            graph = build_tile_graph(tile, self.profile).to(device)
            flags = torch.tensor(
                [[row.left_censored, row.right_censored] for row in tile.note_geometry],
                dtype=torch.float32, device=device).reshape(-1, 2)
            values = self._encode_graph(graph, flags)
            result.append(EncodedTile(tile, graph, values, expected_stages(C3_ADAPTER_VERSION)))
        return tuple(result)
