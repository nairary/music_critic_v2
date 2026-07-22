from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from music_critic.data import CanonicalPiece, CanonicalTrack
from music_critic.graph import (
    RAW_FEATURE_REGISTRY,
    GraphBuildError,
    GraphContractError,
    build_raw_graph,
    dumps_graph,
    graph_fingerprint,
    validate_raw_graph,
)


@pytest.mark.parametrize(
    "injection",
    (
        "global_gold_chord",
        "global_split",
        "global_provenance",
        "node_gold_chord",
        "node_theory",
        "edge_label",
    ),
)
def test_strict_raw_validator_rejects_every_extra_attribute(
    canonical_piece: CanonicalPiece,
    injection: str,
) -> None:
    graph = build_raw_graph(canonical_piece)
    if injection == "global_gold_chord":
        graph.gold_chord = torch.tensor([1])
    elif injection == "global_split":
        graph.split = "test"
    elif injection == "global_provenance":
        graph.provenance = ("prov:gold",)
    elif injection == "node_gold_chord":
        graph["beat"].gold_chord = torch.ones(graph["beat"].num_nodes)
    elif injection == "node_theory":
        graph["note"].theory_label = torch.zeros(graph["note"].num_nodes)
    else:
        graph[("note", "active_at", "beat")].edge_label = torch.ones(
            graph[("note", "active_at", "beat")].edge_index.shape[1]
        )

    with pytest.raises(GraphContractError, match="attributes differ"):
        validate_raw_graph(graph)
    with pytest.raises(GraphContractError, match="attributes differ"):
        dumps_graph(graph)
    with pytest.raises(GraphContractError, match="attributes differ"):
        graph_fingerprint(graph)


def _feature_column(graph, node_type: str, kind: str, name: str) -> int:
    names = getattr(graph[node_type], f"{kind}_feature_names")
    return tuple(names).index(name)


def test_program_and_channel_zero_do_not_collide_with_unavailable(
    canonical_piece: CanonicalPiece,
) -> None:
    empty = CanonicalTrack(
        track_id="track:unavailable",
        source_track_index=2,
        name=None,
        instrument_name=None,
        program=None,
        channel=None,
        is_percussion=False,
        provenance_id=None,
    )
    piece = replace(canonical_piece, tracks=(*canonical_piece.tracks, empty))
    graph = build_raw_graph(piece)
    program = _feature_column(graph, "track", "cat", "program")
    channel = _feature_column(graph, "track", "cat", "channel")

    assert graph["track"].x_cat[0, program].item() == 0
    assert graph["track"].x_cat_available[0, program].item() is True
    assert graph["track"].x_cat[0, channel].item() == 0
    assert graph["track"].x_cat_available[0, channel].item() is True
    assert graph["track"].x_cat[2, program].item() == 128
    assert graph["track"].x_cat_available[2, program].item() is False
    assert graph["track"].x_cat[2, channel].item() == 16
    assert graph["track"].x_cat_available[2, channel].item() is False

    note = replace(canonical_piece.notes[0], program=None, channel=None)
    note_piece = replace(
        canonical_piece,
        notes=(note, *canonical_piece.notes[1:]),
    )
    note_graph = build_raw_graph(note_piece)
    note_program = _feature_column(note_graph, "note", "cat", "program")
    note_channel = _feature_column(note_graph, "note", "cat", "channel")
    assert note_graph["note"].x_cat[0, note_program].item() == 128
    assert note_graph["note"].x_cat[0, note_channel].item() == 16


def test_validator_enforces_canonical_unavailable_placeholders(
    canonical_piece: CanonicalPiece,
) -> None:
    empty = CanonicalTrack(
        track_id="track:unavailable",
        source_track_index=2,
        name=None,
        instrument_name=None,
        program=None,
        channel=None,
        is_percussion=False,
        provenance_id=None,
    )
    piece = replace(canonical_piece, tracks=(*canonical_piece.tracks, empty))
    graph = build_raw_graph(piece)
    program = _feature_column(graph, "track", "cat", "program")
    graph["track"].x_cat[2, program] = 0
    with pytest.raises(GraphContractError, match="unknown ID 128"):
        validate_raw_graph(graph)

    graph = build_raw_graph(piece)
    mean_pitch = _feature_column(graph, "track", "cont", "mean_pitch")
    assert not graph["track"].x_cont_available[2, mean_pitch]
    graph["track"].x_cont[2, mean_pitch] = 1.0
    with pytest.raises(GraphContractError, match="placeholder 0.0"):
        validate_raw_graph(graph)


def test_known_oov_meter_categories_are_rejected_without_sentinels(
    canonical_piece: CanonicalPiece,
) -> None:
    numerator = next(
        spec
        for spec in RAW_FEATURE_REGISTRY.specs
        if spec.node_type == "bar" and spec.name == "meter_numerator"
    )
    denominator = next(
        spec
        for spec in RAW_FEATURE_REGISTRY.specs
        if spec.node_type == "beat" and spec.name == "meter_denominator_log2"
    )
    assert numerator.unknown_id is None
    assert denominator.unknown_id is None
    assert numerator.encode_category(255, available=True) == 255
    assert denominator.encode_category(127, available=True) == 127
    with pytest.raises(ValueError, match="known bar.meter_numerator=256"):
        numerator.encode_category(256, available=True)
    with pytest.raises(ValueError, match="known beat.meter_denominator_log2=128"):
        denominator.encode_category(128, available=True)

    bad_meter = replace(canonical_piece.meter_events[0], numerator=256)
    with pytest.raises(GraphBuildError, match="known bar.meter_numerator=256"):
        build_raw_graph(
            replace(canonical_piece, meter_events=(bad_meter,)),
            assume_valid=True,
        )


def test_build_rejects_unsorted_canonical_piece(
    canonical_piece: CanonicalPiece,
) -> None:
    unsorted = replace(
        canonical_piece,
        notes=(canonical_piece.notes[1], canonical_piece.notes[0], *canonical_piece.notes[2:]),
    )
    with pytest.raises(GraphBuildError, match="COLLECTION_ORDER_INVALID"):
        build_raw_graph(unsorted)
