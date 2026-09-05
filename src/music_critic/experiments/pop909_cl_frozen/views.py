"""Preregistered raw-input views for frozen POP909-CL evaluation.

The input must already be the production adapter's channel-0 score projection.
This module never reconstructs notes from targets or removes arbitrary graph
columns. Both views rebuild through the public raw graph builder.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Literal

from torch_geometric.data import HeteroData

from music_critic.data import CanonicalPiece
from music_critic.graph import (
    build_raw_graph,
    graph_fingerprint,
    model_input_fingerprint,
)


VIEW_SCHEMA = "Pop909ClFrozenInputView@1.0.0"
ViewId = Literal["native", "metadata_removed"]
VIEW_IDS: tuple[ViewId, ...] = ("native", "metadata_removed")
REMOVED_FIELDS = (
    "note.velocity",
    "note.program",
    "note.channel",
    "track.program",
    "track.channel",
)


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def view_contract() -> dict[str, object]:
    """Return a fresh deterministic policy, independent of target content."""

    return {
        "schema": VIEW_SCHEMA,
        "primary": "native",
        "secondary": "metadata_removed",
        "ordered_views": list(VIEW_IDS),
        "metadata_removed_fields": list(REMOVED_FIELDS),
        "replacement": None,
        "other_canonical_fields": "preserve",
        "construction": "deepcopy_then_public_build_raw_graph",
        "derived_feature_closure": "rebuild_values_and_availability_including_track.mean_velocity",
        "input_source": "pop909_cl_channel0_combined_score",
        "channel1": "target_only_excluded_by_production_adapter",
        "fingerprint_inputs": ["schema", "policy", "view_id", "graph_fingerprint", "model_input_fingerprint"],
    }


@dataclass(frozen=True, slots=True)
class FrozenInputView:
    """An independent canonical view and its strictly bound raw graph.

    ``piece`` retains source annotations as a sidecar; only ``graph`` is model
    input. Fingerprints deliberately exclude source targets, split, and
    provenance. ``view_fingerprint`` distinguishes the two policies even if a
    source has all five optional fields absent already.
    """

    piece: CanonicalPiece
    graph: HeteroData
    graph_fingerprint: str
    model_input_fingerprint: str
    view_fingerprint: str
    view_id: ViewId


def build_view(piece: CanonicalPiece, view_id: ViewId = "native") -> FrozenInputView:
    """Clone a canonical score, apply exactly the selected view, rebuild it."""

    if not isinstance(piece, CanonicalPiece):
        raise TypeError("piece must be a CanonicalPiece score projection")
    if view_id not in VIEW_IDS:
        raise ValueError(f"unsupported frozen input view: {view_id!r}")
    viewed_piece = deepcopy(piece)
    if view_id == "metadata_removed":
        viewed_piece = replace(
            viewed_piece,
            notes=tuple(
                replace(note, velocity=None, program=None, channel=None)
                for note in viewed_piece.notes
            ),
            tracks=tuple(
                replace(track, program=None, channel=None)
                for track in viewed_piece.tracks
            ),
        )
    graph = build_raw_graph(viewed_piece)
    graph_digest = graph_fingerprint(graph)
    input_digest = model_input_fingerprint(graph)
    view_digest = _fingerprint({
        "schema": VIEW_SCHEMA,
        "policy": view_contract(),
        "view_id": view_id,
        "graph_fingerprint": graph_digest,
        "model_input_fingerprint": input_digest,
    })
    return FrozenInputView(
        piece=viewed_piece,
        graph=graph,
        graph_fingerprint=graph_digest,
        model_input_fingerprint=input_digest,
        view_fingerprint=view_digest,
        view_id=view_id,
    )
