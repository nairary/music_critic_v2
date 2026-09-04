"""Raw-only onset-sequence decoder for the Phase 9C-B diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.encoder import ENCODER_OUTPUT_VERSION, EncoderOutput
from music_critic.models.hierarchy import extract_hierarchy_ownership


ONSET_BIGRU_DECODER_CONTRACT_VERSION = "1.0.0"


class DilemmadataDecoderConfigError(ValueError):
    """Stable structured configuration or raw-ownership failure."""


@dataclass(frozen=True, slots=True)
class DilemmadataDecoderConfig:
    kind: Literal["mlp", "onset_bigru"] = "mlp"

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "onset_bigru"}:
            raise DilemmadataDecoderConfigError(
                "dilemmadata.decoder.kind_invalid"
            )


def _onset_to_beat_owners(graph: object, encoded: EncoderOutput) -> Tensor:
    """Read one raw beat owner per onset without consulting sidecars."""

    try:
        stores = dict(graph.edge_items())  # type: ignore[attr-defined]
    except Exception as exc:
        raise DilemmadataDecoderConfigError(
            "dilemmadata.decoder.raw_graph_invalid"
        ) from exc
    ownership_type = ("onset", "belongs_to_beat", "beat")
    containment_type = ("beat", "contains_onset", "onset")
    if ownership_type not in stores or containment_type not in stores:
        raise DilemmadataDecoderConfigError(
            "dilemmadata.decoder.onset_beat_relation_missing"
        )
    reverse = stores[ownership_type].get("edge_index")
    forward = stores[containment_type].get("edge_index")
    onset = encoded.embeddings["onset"]
    beat = encoded.embeddings["beat"]
    count = onset.shape[0]
    if (
        not isinstance(reverse, Tensor)
        or not isinstance(forward, Tensor)
        or reverse.dtype != torch.long
        or forward.dtype != torch.long
        or reverse.device != onset.device
        or forward.device != onset.device
        or reverse.shape != (2, count)
        or forward.shape != (2, count)
        or not torch.equal(forward, reverse.flip(0))
        or not torch.equal(
            reverse[0],
            torch.arange(count, dtype=torch.long, device=onset.device),
        )
    ):
        raise DilemmadataDecoderConfigError(
            "dilemmadata.decoder.onset_beat_relation_invalid"
        )
    owners = reverse[1]
    if owners.numel() and (
        bool((owners < 0).any()) or bool((owners >= beat.shape[0]).any())
    ):
        raise DilemmadataDecoderConfigError(
            "dilemmadata.decoder.onset_beat_owner_out_of_range"
        )
    if owners.numel() and not torch.equal(
        encoded.batch_membership["onset"],
        encoded.batch_membership["beat"].index_select(0, owners),
    ):
        raise DilemmadataDecoderConfigError(
            "dilemmadata.decoder.onset_beat_cross_sample"
        )
    return owners


class OnsetBiGRUDecoder(nn.Module):
    """Contextualize raw onset rows and propagate their context to owners."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
            or hidden_dim % 2
        ):
            raise DilemmadataDecoderConfigError(
                "dilemmadata.decoder.hidden_dim_must_be_positive_even"
            )
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise DilemmadataDecoderConfigError(
                "dilemmadata.decoder.dropout_invalid"
            )
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.onset_projection = nn.Linear(hidden_dim, hidden_dim)
        self.onset_gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.onset_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(float(dropout))
        self.beat_availability = nn.Embedding(2, hidden_dim)
        self.beat_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.beat_gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.beat_norm = nn.LayerNorm(hidden_dim)
        self.bar_availability = nn.Embedding(2, hidden_dim)
        self.bar_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bar_gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bar_norm = nn.LayerNorm(hidden_dim)

    def _sequence_context(self, onset: Tensor, membership: Tensor, samples: int) -> Tensor:
        """Run isolated non-empty sequences and restore original onset row order.

        Separate GRU calls make one song numerically independent of the other
        sequence lengths present in a batch.  Raw rows are already grouped by
        sample and exactly ordered by the graph contract.
        """

        if onset.shape[0] == 0:
            return onset
        lengths = torch.bincount(membership, minlength=samples)
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=onset.device),
                lengths.cumsum(0),
            )
        )
        restored = torch.empty_like(onset)
        for sample_index in torch.nonzero(
            lengths > 0, as_tuple=False
        ).flatten().tolist():
            start = int(offsets[sample_index])
            end = int(offsets[sample_index + 1])
            sequence, _ = self.gru(onset[start:end].unsqueeze(0))
            restored[start:end] = sequence.squeeze(0)
        return restored

    @staticmethod
    def _mean_pool(context: Tensor, owners: Tensor, parent_count: int) -> tuple[Tensor, Tensor]:
        pooled = torch.zeros(
            (parent_count, context.shape[1]),
            dtype=context.dtype,
            device=context.device,
        ).index_add(0, owners, context)
        counts = torch.bincount(owners, minlength=parent_count)
        available = counts > 0
        pooled = pooled / counts.clamp_min(1).to(context.dtype).unsqueeze(1)
        return pooled, available

    def _owner_fusion(
        self,
        local: Tensor,
        pooled: Tensor,
        available: Tensor,
        *,
        availability: nn.Embedding,
        projection: nn.Linear,
        gate_projection: nn.Linear,
        norm: nn.LayerNorm,
    ) -> Tensor:
        state = availability(available.to(dtype=torch.long))
        context = projection(pooled + state)
        gate = torch.sigmoid(gate_projection(torch.cat((local, context), dim=-1)))
        return norm(local + self.output_dropout(gate * context))

    def forward(self, encoded: EncoderOutput, raw_graph_batch: object) -> EncoderOutput:
        ownership = extract_hierarchy_ownership(raw_graph_batch, encoded)
        onset_to_beat = _onset_to_beat_owners(raw_graph_batch, encoded)
        onset = encoded.embeddings["onset"]
        membership = encoded.batch_membership["onset"]
        context = self._sequence_context(onset, membership, ownership.sample_count)
        projected = self.onset_projection(context)
        gate = torch.sigmoid(
            self.onset_gate_projection(torch.cat((onset, projected), dim=-1))
        )
        contextual_onset = self.onset_norm(
            onset + self.output_dropout(gate * projected)
        )
        beat_pool, beat_available = self._mean_pool(
            contextual_onset, onset_to_beat, encoded.embeddings["beat"].shape[0]
        )
        bar_pool, bar_available = self._mean_pool(
            contextual_onset,
            ownership.owners["onset_to_bar"],
            encoded.embeddings["bar"].shape[0],
        )
        embeddings: Mapping[str, Tensor] = {
            **encoded.embeddings,
            "onset": contextual_onset,
            "beat": self._owner_fusion(
                encoded.embeddings["beat"],
                beat_pool,
                beat_available,
                availability=self.beat_availability,
                projection=self.beat_context_projection,
                gate_projection=self.beat_gate_projection,
                norm=self.beat_norm,
            ),
            "bar": self._owner_fusion(
                encoded.embeddings["bar"],
                bar_pool,
                bar_available,
                availability=self.bar_availability,
                projection=self.bar_context_projection,
                gate_projection=self.bar_gate_projection,
                norm=self.bar_norm,
            ),
        }
        if tuple(embeddings) != MANDATORY_NODE_TYPES or any(
            not bool(torch.isfinite(value).all()) for value in embeddings.values()
        ):
            raise DilemmadataDecoderConfigError(
                "dilemmadata.decoder.output_non_finite_or_incomplete"
            )
        return EncoderOutput(
            contract_version=ENCODER_OUTPUT_VERSION,
            embeddings=embeddings,
            batch_membership=encoded.batch_membership,
        )


__all__ = [
    "DilemmadataDecoderConfig",
    "DilemmadataDecoderConfigError",
    "ONSET_BIGRU_DECODER_CONTRACT_VERSION",
    "OnsetBiGRUDecoder",
]
