"""Opt-in complete raw-input transposition, separate from frozen C1/C2 APIs.

This module changes no trainer routing or historical profile identity. Future
consumers must bind ``complete_transposition_contract()`` in their protocol.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from math import sqrt
from statistics import fmean

import torch
from torch_geometric.data import HeteroData

from music_critic.data.serialization import canonical_json_sha256
from music_critic.experiments.analysisgnn.transposition import (
    DIRECTED_TRANSPOSITION_SCHEMA,
    AnalysisGNNTranspositionError,
    DirectedTransposition,
)
from music_critic.graph.feature_registry import RAW_FEATURE_REGISTRY
from music_critic.graph.relations import GRAPH_BUILDER_VERSION, GRAPH_SCHEMA_VERSION
from music_critic.graph.validation import GraphContractError, validate_raw_graph


COMPLETE_TRANSPOSITION_SCHEMA = "CompleteRawGraphTransposition@1.0.0"
COMPLETE_PITCH_FEATURES = (
    "note.pitch", "note.pitch_class", "note.octave",
    "note.track_relative_pitch", "track.mean_pitch", "track.pitch_std",
    "track.min_pitch", "track.max_pitch",
)
_OWNERSHIP = ("track", "contains_note", "note")


def complete_transposition_contract() -> dict[str, object]:
    """Return the new protocol identity; frozen B5A/B5G/B5H stay unchanged."""

    body: dict[str, object] = {
        "schema": COMPLETE_TRANSPOSITION_SCHEMA,
        "directed_token_schema": DIRECTED_TRANSPOSITION_SCHEMA,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "feature_registry_version": RAW_FEATURE_REGISTRY.version,
        "feature_registry": [asdict(spec) for spec in RAW_FEATURE_REGISTRY.specs],
        "pitch_features": list(COMPLETE_PITCH_FEATURES),
        "pitch_shift_policy": "signed_integer_addition_non_percussion_notes_only",
        "range_policy": "reject_whole_view_outside_0_127_no_wrap_clip_or_fold",
        "statistics_policy": "all_owned_note_pitches_fmean_population_std",
        "availability_policy": "builder_nonempty_track_and_positive_std_for_relative",
        "ownership_policy": "exactly_one_track_per_note_including_percussion",
        "input_policy": "single_validated_raw_graph_default_feature_registry",
        "output_policy": "detached_copy_same_ids_topology_timing_and_non_pitch_fields",
        "historical_apis_changed": False,
        "trainer_opt_in": False,
    }
    return {**body, "fingerprint": canonical_json_sha256(body)}


def _error(reason: str, detail: str) -> AnalysisGNNTranspositionError:
    return AnalysisGNNTranspositionError(
        f"analysisgnn.transposition.{reason}: {detail}"
    )


def transpose_complete_raw_graph_view(
    graph: HeteroData, *, transform: DirectedTransposition
) -> HeteroData:
    """Recompute every pitch-derived field from actual note ownership.

    Accept a single target-free graph from the default raw builder. All note
    pitches (including percussion) participate in their owner's statistics,
    as in that builder; only non-percussion pitches move. Empty tracks have
    unavailable zero statistics; constant tracks have available zero std but
    unavailable zero relative pitch. Signed inverses include the existing
    ``DirectedTransposition(6, -6)`` tritone token.

    The protocol fingerprint belongs in the caller's view/experiment identity,
    not in raw graph attributes, whose strict serialization stays unchanged.
    """

    if not isinstance(transform, DirectedTransposition):
        raise _error("directed_transform_required", type(transform).__name__)
    try:
        validate_raw_graph(graph)
    except GraphContractError as exc:
        raise _error("complete_raw_graph_invalid", str(exc)) from exc

    note = graph["note"]
    track = graph["track"]
    note_cat = {name: i for i, name in enumerate(note.cat_feature_names)}
    track_cat = {name: i for i, name in enumerate(track.cat_feature_names)}
    owned: list[list[int]] = [[] for _ in range(track.num_nodes)]
    counts = [0] * note.num_nodes
    for owner, child in graph[_OWNERSHIP].edge_index.t().tolist():
        owned[owner].append(child)
        counts[child] += 1
        if note.x_cat[child, note_cat["is_percussion"]].item() != track.x_cat[
            owner, track_cat["is_percussion"]
        ].item():
            raise _error("complete_ownership_invalid", "percussion flags disagree")
    if any(count != 1 for count in counts):
        raise _error("complete_ownership_invalid", "each note needs exactly one track")

    pitches = note.x_cat[:, note_cat["pitch"]].tolist()
    percussion = note.x_cat[:, note_cat["is_percussion"]].tolist()
    shifted = [
        pitch if drum else pitch + transform.signed_semitones
        for pitch, drum in zip(pitches, percussion, strict=True)
    ]
    if any(pitch < 0 or pitch > 127 for pitch in shifted):
        raise _error("midi_range_violation", "directed shift leaves MIDI 0..127")

    # Seed deepcopy with detached tensors: preserve tuple metadata and support
    # tensor views/non-leaf inputs without sharing storage or autograd history.
    memo = {
        id(value): value.detach().clone()
        for store in graph.stores for value in store.values()
        if isinstance(value, torch.Tensor)
    }
    view = copy.deepcopy(graph, memo)
    note = view["note"]
    track = view["track"]
    values = torch.tensor(shifted, dtype=torch.long, device=note.x_cat.device)
    note.x_cat[:, note_cat["pitch"]] = values
    note.x_cat[:, note_cat["pitch_class"]] = values.remainder(12)
    note.x_cat[:, note_cat["octave"]] = values.div(12, rounding_mode="floor")

    relative_i = note.cont_feature_names.index("track_relative_pitch")
    track_cont = {name: i for i, name in enumerate(track.cont_feature_names)}
    note.x_cont[:, relative_i] = 0.0
    note.x_cont_available[:, relative_i] = False
    for owner, indices in enumerate(owned):
        # Canonical note order plus the builder's float64 fmean/sqrt operations
        # make float32 packing identical to an independent canonical rebuild.
        indices.sort()
        pitches_in_track = [shifted[index] for index in indices]
        mean = fmean(pitches_in_track) if indices else 0.0
        std = sqrt(fmean((p - mean) ** 2 for p in pitches_in_track)) if indices else 0.0
        statistics = {
            "mean_pitch": mean,
            "pitch_std": std,
            "min_pitch": min(pitches_in_track) if indices else 0.0,
            "max_pitch": max(pitches_in_track) if indices else 0.0,
        }
        for name, value in statistics.items():
            column = track_cont[name]
            track.x_cont[owner, column] = value
            track.x_cont_available[owner, column] = bool(indices)
        if std > 0:
            note.x_cont[indices, relative_i] = torch.tensor(
                [(p - mean) / std for p in pitches_in_track],
                dtype=note.x_cont.dtype, device=note.x_cont.device,
            )
            note.x_cont_available[indices, relative_i] = True
    return view


__all__ = [
    "COMPLETE_PITCH_FEATURES",
    "COMPLETE_TRANSPOSITION_SCHEMA",
    "complete_transposition_contract",
    "transpose_complete_raw_graph_view",
]
