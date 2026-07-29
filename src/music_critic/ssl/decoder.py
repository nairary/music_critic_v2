"""Deterministic decoder re-masking and row-wise representation decoding.

Decoder re-masking is deliberately defined over *latent* rows.  This module
never receives a raw graph or raw feature values.  A ``MaskPlan`` contributes
only its deterministic identity and the number of selected note rows; the
positions stored in a :class:`DecoderRemaskPlan` are relative to that compact
selected-row tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn


DECODER_REMASK_CONTRACT_VERSION = "1.0.0"
REPRESENTATION_DECODER_CONTRACT_VERSION = "1.0.0"

_UINT64_MODULUS = 1 << 64


@runtime_checkable
class _MaskPlanLike(Protocol):
    """The narrow mask-plan surface used by decoder re-masking."""

    fingerprint: str
    stable_seed: int
    selected_node_type: str
    selected_local_node_indices: tuple[int, ...]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _validate_mask_plan_surface(mask_plan: object) -> _MaskPlanLike:
    """Fail closed without coupling this module to the MaskPlan definition."""

    required = (
        "fingerprint",
        "stable_seed",
        "selected_node_type",
        "selected_local_node_indices",
    )
    missing = tuple(name for name in required if not hasattr(mask_plan, name))
    if missing:
        raise TypeError(
            "mask_plan is missing decoder-required fields: "
            + ", ".join(missing)
        )
    fingerprint = _validate_sha256(
        getattr(mask_plan, "fingerprint"),
        name="mask_plan.fingerprint",
    )
    del fingerprint
    stable_seed = getattr(mask_plan, "stable_seed")
    if (
        isinstance(stable_seed, bool)
        or not isinstance(stable_seed, int)
        or not 0 <= stable_seed < _UINT64_MODULUS
    ):
        raise ValueError("mask_plan.stable_seed must be a uint64 integer")
    selected_node_type = getattr(mask_plan, "selected_node_type")
    if not isinstance(selected_node_type, str) or not selected_node_type:
        raise ValueError("mask_plan.selected_node_type must be non-empty")
    selected = getattr(mask_plan, "selected_local_node_indices")
    if not isinstance(selected, tuple):
        raise TypeError("selected_local_node_indices must be a tuple")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in selected
    ):
        raise ValueError(
            "selected_local_node_indices must contain non-negative integers"
        )
    if tuple(sorted(set(selected))) != selected:
        raise ValueError(
            "selected_local_node_indices must be sorted and unique"
        )
    return mask_plan  # type: ignore[return-value]


def _validate_view_index(decoder_view_index: object) -> int:
    if (
        isinstance(decoder_view_index, bool)
        or not isinstance(decoder_view_index, int)
        or decoder_view_index < 0
    ):
        raise ValueError("decoder_view_index must be a non-negative integer")
    return decoder_view_index


def _validate_probability(remask_probability: object) -> float:
    if (
        isinstance(remask_probability, bool)
        or not isinstance(remask_probability, (int, float))
    ):
        raise TypeError("remask_probability must be a real number")
    probability = float(remask_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("remask_probability must lie in [0, 1]")
    return probability


def _view_seed_sha256(
    mask_plan: _MaskPlanLike,
    decoder_view_index: int,
) -> str:
    """Bind each decoder view to one encoder mask without Python ``hash``."""

    return _canonical_sha256(
        {
            "contract_version": DECODER_REMASK_CONTRACT_VERSION,
            "decoder_view_index": decoder_view_index,
            "mask_plan_fingerprint": mask_plan.fingerprint,
            "mask_plan_stable_seed": mask_plan.stable_seed,
            "purpose": "decoder_remask_view_seed",
        }
    )


def _remasked_positions(
    *,
    row_count: int,
    probability: float,
    view_seed_sha256: str,
) -> tuple[int, ...]:
    """Perform reproducible Bernoulli selection with SHA-256 scores.

    Comparing uint64 scores against one exact integer threshold avoids global
    RNG state and makes selection independent of torch, worker, and batch
    ordering.  The two boundary probabilities are handled exactly.
    """

    if row_count == 0 or probability == 0.0:
        return ()
    if probability == 1.0:
        return tuple(range(row_count))
    threshold = int(probability * _UINT64_MODULUS)
    selected = []
    for position in range(row_count):
        digest = hashlib.sha256(
            (
                f"{view_seed_sha256}:latent-position:{position}"
            ).encode("ascii")
        ).digest()
        score = int.from_bytes(digest[:8], byteorder="big", signed=False)
        if score < threshold:
            selected.append(position)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class DecoderRemaskPlan:
    """One deterministic latent-row re-mask view for one encoder MaskPlan."""

    contract_version: str
    mask_plan_fingerprint: str
    selected_node_type: str
    decoder_view_index: int
    remask_probability: float
    selected_row_count: int
    remasked_positions: tuple[int, ...]
    stable_seed: int
    stable_seed_sha256: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.contract_version != DECODER_REMASK_CONTRACT_VERSION:
            raise ValueError("decoder re-mask contract version is incompatible")
        _validate_sha256(
            self.mask_plan_fingerprint,
            name="mask_plan_fingerprint",
        )
        if not isinstance(self.selected_node_type, str) or not self.selected_node_type:
            raise ValueError("selected_node_type must be non-empty")
        _validate_view_index(self.decoder_view_index)
        probability = _validate_probability(self.remask_probability)
        if probability != self.remask_probability:
            raise ValueError("remask_probability must use its canonical float value")
        if (
            isinstance(self.selected_row_count, bool)
            or not isinstance(self.selected_row_count, int)
            or self.selected_row_count < 0
        ):
            raise ValueError("selected_row_count must be a non-negative integer")
        if not isinstance(self.remasked_positions, tuple):
            raise TypeError("remasked_positions must be a tuple")
        if (
            any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position < self.selected_row_count
                for position in self.remasked_positions
            )
            or tuple(sorted(set(self.remasked_positions)))
            != self.remasked_positions
        ):
            raise ValueError(
                "remasked_positions must be sorted, unique selected-row positions"
            )
        if self.remask_probability == 0.0 and self.remasked_positions:
            raise ValueError("zero remask probability cannot select latent rows")
        if self.remask_probability == 1.0 and self.remasked_positions != tuple(
            range(self.selected_row_count)
        ):
            raise ValueError("unit remask probability must select every latent row")
        if (
            isinstance(self.stable_seed, bool)
            or not isinstance(self.stable_seed, int)
            or not 0 <= self.stable_seed < _UINT64_MODULUS
        ):
            raise ValueError("stable_seed must be a uint64 integer")
        seed_digest = _validate_sha256(
            self.stable_seed_sha256,
            name="stable_seed_sha256",
        )
        if self.stable_seed != int(seed_digest[:16], 16):
            raise ValueError("stable_seed does not match stable_seed_sha256")
        _validate_sha256(self.fingerprint, name="fingerprint")
        expected_fingerprint = _canonical_sha256(
            {
                "contract_version": self.contract_version,
                "decoder_view_index": self.decoder_view_index,
                "mask_plan_fingerprint": self.mask_plan_fingerprint,
                "remask_probability_hex": self.remask_probability.hex(),
                "remasked_positions": list(self.remasked_positions),
                "selected_node_type": self.selected_node_type,
                "selected_row_count": self.selected_row_count,
                "stable_seed": self.stable_seed,
                "stable_seed_sha256": self.stable_seed_sha256,
            }
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("decoder re-mask fingerprint is inconsistent")

    @property
    def remasked_latent_row_indices(self) -> tuple[int, ...]:
        """Explicit alias documenting that positions address compact latents."""

        return self.remasked_positions

    @property
    def realized_remask_rate(self) -> float:
        if self.selected_row_count == 0:
            return 0.0
        return len(self.remasked_positions) / self.selected_row_count


def build_decoder_remask_plan(
    mask_plan: object,
    *,
    decoder_view_index: int,
    remask_probability: float,
) -> DecoderRemaskPlan:
    """Build one batch-order-independent decoder view from one MaskPlan."""

    checked = _validate_mask_plan_surface(mask_plan)
    view_index = _validate_view_index(decoder_view_index)
    probability = _validate_probability(remask_probability)
    view_seed_sha256 = _view_seed_sha256(checked, view_index)
    stable_seed = int(view_seed_sha256[:16], 16)
    row_count = len(checked.selected_local_node_indices)
    positions = _remasked_positions(
        row_count=row_count,
        probability=probability,
        view_seed_sha256=view_seed_sha256,
    )
    payload = {
        "contract_version": DECODER_REMASK_CONTRACT_VERSION,
        "decoder_view_index": view_index,
        "mask_plan_fingerprint": checked.fingerprint,
        "remask_probability_hex": probability.hex(),
        "remasked_positions": list(positions),
        "selected_node_type": checked.selected_node_type,
        "selected_row_count": row_count,
        "stable_seed": stable_seed,
        "stable_seed_sha256": view_seed_sha256,
    }
    return DecoderRemaskPlan(
        contract_version=DECODER_REMASK_CONTRACT_VERSION,
        mask_plan_fingerprint=checked.fingerprint,
        selected_node_type=checked.selected_node_type,
        decoder_view_index=view_index,
        remask_probability=probability,
        selected_row_count=row_count,
        remasked_positions=positions,
        stable_seed=stable_seed,
        stable_seed_sha256=view_seed_sha256,
        fingerprint=_canonical_sha256(payload),
    )


def build_decoder_remask_plans(
    mask_plan: object,
    *,
    decoder_views: int,
    remask_probability: float,
) -> tuple[DecoderRemaskPlan, ...]:
    """Build all deterministic views, including the one-view/no-remask case."""

    if (
        isinstance(decoder_views, bool)
        or not isinstance(decoder_views, int)
        or decoder_views < 1
    ):
        raise ValueError("decoder_views must be a positive integer")
    return tuple(
        build_decoder_remask_plan(
            mask_plan,
            decoder_view_index=view_index,
            remask_probability=remask_probability,
        )
        for view_index in range(decoder_views)
    )


def selected_global_node_indices(
    mask_plans: Sequence[object],
    batch_offsets: Sequence[int] | Tensor,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Resolve per-sample selected local rows through a PyG-style ``ptr``.

    ``batch_offsets`` must contain ``sample_count + 1`` cumulative offsets.
    Only integer indices are read; raw feature values are neither accepted nor
    accessed.
    """

    checked_plans = tuple(
        _validate_mask_plan_surface(mask_plan) for mask_plan in mask_plans
    )
    if isinstance(batch_offsets, Tensor):
        if (
            batch_offsets.dtype != torch.long
            or batch_offsets.ndim != 1
        ):
            raise ValueError("tensor batch_offsets must be rank-one torch.long")
        offsets = tuple(
            int(value)
            for value in batch_offsets.detach().cpu().tolist()
        )
    else:
        offsets = tuple(batch_offsets)
    if len(offsets) != len(checked_plans) + 1:
        raise ValueError("batch_offsets must contain sample_count + 1 entries")
    if any(
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
        for offset in offsets
    ):
        raise ValueError("batch_offsets must contain non-negative integers")
    if not offsets or offsets[0] != 0 or any(
        right < left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("batch_offsets must be monotonic and start at zero")
    global_indices: list[int] = []
    for sample_index, mask_plan in enumerate(checked_plans):
        sample_size = offsets[sample_index + 1] - offsets[sample_index]
        if any(
            local_index >= sample_size
            for local_index in mask_plan.selected_local_node_indices
        ):
            raise ValueError(
                "selected local node index is outside its batch-offset range"
            )
        global_indices.extend(
            offsets[sample_index] + local_index
            for local_index in mask_plan.selected_local_node_indices
        )
    return torch.tensor(global_indices, dtype=torch.long, device=device)


def gather_selected_latent_rows(
    latents: Tensor,
    mask_plans: Sequence[object],
    batch_offsets: Sequence[int] | Tensor,
) -> Tensor:
    """Gather compact selected rows without reading or accepting raw values."""

    if not isinstance(latents, Tensor) or latents.ndim != 2:
        raise ValueError("latents must be a rank-two tensor")
    indices = selected_global_node_indices(
        mask_plans,
        batch_offsets,
        device=latents.device,
    )
    if int(latents.shape[0]) != (
        int(batch_offsets[-1])
        if not isinstance(batch_offsets, Tensor)
        else int(batch_offsets[-1].item())
    ):
        raise ValueError("latent row count must equal the final batch offset")
    return latents.index_select(0, indices)


def apply_decoder_remask(
    latents: Tensor,
    plan: DecoderRemaskPlan,
    mask_token: Tensor,
) -> Tensor:
    """Return a non-mutating latent view with exactly the planned rows replaced."""

    if not isinstance(latents, Tensor) or latents.ndim != 2:
        raise ValueError("latents must be a rank-two tensor")
    if not latents.is_floating_point():
        raise TypeError("latents must use a floating-point dtype")
    if not isinstance(plan, DecoderRemaskPlan):
        raise TypeError("plan must be a DecoderRemaskPlan")
    if int(latents.shape[0]) != plan.selected_row_count:
        raise ValueError(
            "latent rows must be the compact rows selected by the encoder MaskPlan"
        )
    if (
        not isinstance(mask_token, Tensor)
        or mask_token.ndim not in {1, 2}
        or (
            mask_token.ndim == 1
            and mask_token.shape != (latents.shape[1],)
        )
        or (
            mask_token.ndim == 2
            and mask_token.shape != (1, latents.shape[1])
        )
    ):
        raise ValueError("mask_token must have shape [D] or [1, D]")
    if mask_token.device != latents.device or mask_token.dtype != latents.dtype:
        raise ValueError("mask_token must match latent device and dtype")
    result = latents.clone()
    if not plan.remasked_positions:
        return result
    positions = torch.tensor(
        plan.remasked_positions,
        dtype=torch.long,
        device=latents.device,
    )
    replacement = mask_token.reshape(1, -1).expand(positions.shape[0], -1)
    result.index_copy_(0, positions, replacement)
    return result


class RepresentationDecoder(nn.Module):
    """Deterministic contextual MLP over compact selected latent rows."""

    contract_version = REPRESENTATION_DECODER_CONTRACT_VERSION

    def __init__(
        self,
        hidden_dim: int,
        decoder_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if decoder_hidden_dim is None:
            decoder_hidden_dim = hidden_dim
        if (
            isinstance(decoder_hidden_dim, bool)
            or not isinstance(decoder_hidden_dim, int)
            or decoder_hidden_dim <= 0
        ):
            raise ValueError("decoder_hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.context_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.input_normalization = nn.LayerNorm(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_hidden_dim),
            nn.Linear(decoder_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        latents: Tensor,
        plan: DecoderRemaskPlan | None = None,
        *,
        context: Tensor | None = None,
    ) -> Tensor:
        if not isinstance(latents, Tensor) or latents.ndim != 2:
            raise ValueError("latents must be a rank-two tensor")
        if int(latents.shape[1]) != self.hidden_dim:
            raise ValueError("latent hidden dimension is incompatible with decoder")
        decoder_input = (
            latents
            if plan is None
            else apply_decoder_remask(
                latents,
                plan,
                self.mask_token.to(
                    device=latents.device,
                    dtype=latents.dtype,
                ),
            )
        )
        if context is not None:
            if (
                not isinstance(context, Tensor)
                or context.shape != latents.shape
                or context.device != latents.device
                or context.dtype != latents.dtype
            ):
                raise ValueError(
                    "decoder context must match latent shape/device/dtype"
                )
            decoder_input = (
                decoder_input + self.context_projection(context)
            )
        decoder_input = self.input_normalization(decoder_input)
        return self.network(decoder_input)

    def forward_views(
        self,
        latents: Tensor,
        plans: Sequence[DecoderRemaskPlan],
        *,
        context: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if not plans:
            raise ValueError("at least one decoder re-mask plan is required")
        return tuple(
            self(latents, plan, context=context)
            for plan in plans
        )


__all__ = [
    "DECODER_REMASK_CONTRACT_VERSION",
    "REPRESENTATION_DECODER_CONTRACT_VERSION",
    "DecoderRemaskPlan",
    "RepresentationDecoder",
    "apply_decoder_remask",
    "build_decoder_remask_plan",
    "build_decoder_remask_plans",
    "gather_selected_latent_rows",
    "selected_global_node_indices",
]
