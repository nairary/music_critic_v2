from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import CanonicalPiece, RationalTime, load_piece
from music_critic.experiments.analysisgnn.transposition import (
    AnalysisGNNTranspositionError,
    DirectedTransposition,
    canonical_directed_transposition,
    directed_transposition_contract,
    graph_changed_fields,
    transpose_raw_graph_view_directed,
)
from music_critic.experiments.analysisgnn.transposition_inputs import (
    COMPLETE_PITCH_FEATURES,
    COMPLETE_TRANSPOSITION_SCHEMA,
    complete_transposition_contract,
    transpose_complete_raw_graph_view,
)
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.graph.feature_registry import RAW_FEATURE_REGISTRY


FIXTURE = Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"


def _piece() -> CanonicalPiece:
    piece = load_piece(FIXTURE)
    constant_track = replace(
        piece.tracks[0], track_id="track:constant", source_track_index=2
    )
    empty_track = replace(
        piece.tracks[0], track_id="track:empty", source_track_index=3
    )
    constant_notes = tuple(
        replace(
            piece.notes[0],
            note_id=f"note:constant-{i}",
            track_id=constant_track.track_id,
            onset_qn=RationalTime(i),
            duration_qn=RationalTime(1),
            pitch=61,
        )
        for i in range(2)
    )
    tracks = (*piece.tracks, constant_track, empty_track)
    track_order = {track.track_id: i for i, track in enumerate(tracks)}
    notes = tuple(sorted(
        (*piece.notes, *constant_notes),
        key=lambda note: (
            note.onset_qn, track_order[note.track_id], note.pitch,
            note.duration_qn, note.note_id,
        ),
    ))
    return replace(piece, tracks=tracks, notes=notes, targets=(), annotations=())


def _oracle(piece: CanonicalPiece, signed: int):
    # Only canonical raw observations are changed. The ordinary builder is the
    # independent oracle for every feature, mask, ownership relation and ID.
    return build_raw_graph(replace(piece, notes=tuple(
        replace(note, pitch=note.pitch + signed)
        if not note.is_percussion else note
        for note in piece.notes
    )))


def test_frozen_forward_reproduces_stale_track_statistics() -> None:
    piece = _piece()
    source = build_raw_graph(piece)
    old = transpose_raw_graph_view_directed(
        source, transform=canonical_directed_transposition(2)
    )
    rebuilt = _oracle(piece, 2)
    assert torch.equal(old["track"].x_cont, source["track"].x_cont)
    assert set(graph_changed_fields(old, rebuilt)) == {
        "track.mean_pitch", "track.min_pitch", "track.max_pitch"
    }


def _assert_exact_graph(left, right) -> None:
    # Serialization validates the full raw-only schema, fields and masks too.
    assert graph_fingerprint(left) == graph_fingerprint(right)


def test_new_identity_covers_the_entire_current_pitch_feature_registry() -> None:
    contract = complete_transposition_contract()
    assert contract["schema"] == COMPLETE_TRANSPOSITION_SCHEMA
    assert contract["fingerprint"] != directed_transposition_contract()["fingerprint"]
    assert contract == complete_transposition_contract()
    pitch_features = {
        f"{spec.node_type}.{spec.name}" for spec in RAW_FEATURE_REGISTRY.specs
        if "pitch" in spec.name or spec.name == "octave"
    }
    assert pitch_features == set(COMPLETE_PITCH_FEATURES)
    assert contract["historical_apis_changed"] is False
    assert contract["trainer_opt_in"] is False


@pytest.mark.parametrize("shift", range(12))
def test_all_shifts_and_inverse_equal_full_canonical_rebuild_exactly(shift: int) -> None:
    piece = _piece()
    source = build_raw_graph(piece)
    before = graph_fingerprint(source)
    transform = canonical_directed_transposition(shift)
    view = transpose_complete_raw_graph_view(source, transform=transform)
    _assert_exact_graph(view, _oracle(piece, transform.signed_semitones))
    restored = transpose_complete_raw_graph_view(view, transform=transform.inverse())
    _assert_exact_graph(source, restored)
    assert graph_fingerprint(source) == before
    assert set(graph_changed_fields(source, view)) <= set(COMPLETE_PITCH_FEATURES)
    for node_type in source.node_types:
        for key, value in source[node_type].items():
            if isinstance(value, torch.Tensor) and value.numel():
                assert value.data_ptr() != view[node_type][key].data_ptr()
    for edge_type in source.edge_types:
        assert torch.equal(source[edge_type].edge_index, view[edge_type].edge_index)
        if source[edge_type].edge_index.numel():
            assert source[edge_type].edge_index.data_ptr() != view[edge_type].edge_index.data_ptr()


def test_empty_constant_and_percussion_statistics_and_availability() -> None:
    source = build_raw_graph(_piece())
    view = transpose_complete_raw_graph_view(
        source, transform=canonical_directed_transposition(6)
    )
    track = view["track"]
    note = view["note"]
    columns = [track.cont_feature_names.index(name) for name in (
        "mean_pitch", "pitch_std", "min_pitch", "max_pitch"
    )]
    empty = track.entity_id.index("track:empty")
    constant = track.entity_id.index("track:constant")
    drum = track.entity_id.index("track:drums")
    assert track.x_cont[empty, columns].tolist() == [0.0] * 4
    assert not track.x_cont_available[empty, columns].any()
    assert track.x_cont[constant, columns].tolist() == [67.0, 0.0, 67.0, 67.0]
    assert track.x_cont_available[constant, columns].all()
    assert torch.equal(track.x_cont[drum], source["track"].x_cont[drum])
    relative = note.cont_feature_names.index("track_relative_pitch")
    constant_notes = [i for i, entity in enumerate(note.entity_id) if "constant" in entity]
    assert not note.x_cont_available[constant_notes, relative].any()
    assert note.x_cont[constant_notes, relative].eq(0).all()
    drum_notes = note.x_cat[:, note.cat_feature_names.index("is_percussion")].eq(1)
    for name in ("x_cat", "x_cont", "x_cat_available", "x_cont_available"):
        assert torch.equal(note[name][drum_notes], source["note"][name][drum_notes])


@pytest.mark.parametrize(("pitch", "signed", "valid"), [
    (0, 0, True), (127, 0, True), (0, 1, True), (127, -1, True),
    (0, -1, False), (127, 1, False), (121, 6, True), (122, 6, False),
    (6, -6, True), (5, -6, False), (60, 12, True), (60, -(10**30), False),
])
def test_range_and_tritone_boundary_fail_closed(pitch: int, signed: int, valid: bool) -> None:
    piece = _piece()
    piece = replace(piece, notes=tuple(
        replace(note, pitch=pitch) if not note.is_percussion else note
        for note in piece.notes
    ))
    source = build_raw_graph(piece)
    before = graph_fingerprint(source)
    transform = DirectedTransposition(signed % 12, signed)
    if valid:
        view = transpose_complete_raw_graph_view(source, transform=transform)
        _assert_exact_graph(view, _oracle(piece, signed))
        _assert_exact_graph(source, transpose_complete_raw_graph_view(view, transform=transform.inverse()))
    else:
        with pytest.raises(AnalysisGNNTranspositionError, match="midi_range_violation"):
            transpose_complete_raw_graph_view(source, transform=transform)
    assert graph_fingerprint(source) == before


def test_drum_midi_boundaries_do_not_make_pitched_shift_ineligible() -> None:
    piece = _piece()
    piece = replace(piece, notes=tuple(
        replace(note, pitch=127) if note.is_percussion else note
        for note in piece.notes
    ))
    graph = build_raw_graph(piece)
    view = transpose_complete_raw_graph_view(graph, transform=DirectedTransposition(6, 6))
    _assert_exact_graph(view, _oracle(piece, 6))


def test_graph_with_no_notes_stays_exact_and_unavailable() -> None:
    piece = replace(_piece(), notes=())
    graph = build_raw_graph(piece)
    _assert_exact_graph(graph, transpose_complete_raw_graph_view(
        graph, transform=DirectedTransposition(6, -6)
    ))


def test_track_ownership_not_entity_name_or_order_drives_statistics() -> None:
    piece = _piece()
    renamed = {track.track_id: f"track:opaque-{i}" for i, track in enumerate(piece.tracks)}
    # Keeping canonical positions but obscuring source naming catches name-based
    # ownership shortcuts; change edge order with its reverse independently.
    piece = replace(piece, tracks=tuple(
        replace(track, track_id=renamed[track.track_id]) for track in piece.tracks
    ), notes=tuple(replace(note, track_id=renamed[note.track_id]) for note in piece.notes))
    graph = build_raw_graph(piece)
    expected = _oracle(piece, 2)
    for item in (graph, expected):
        edge = item[("track", "contains_note", "note")].edge_index.flip(1)
        item[("track", "contains_note", "note")].edge_index = edge
        item[("note", "belongs_to_track", "track")].edge_index = edge.flip(0)
    _assert_exact_graph(expected, transpose_complete_raw_graph_view(
        graph, transform=DirectedTransposition(2, 2)
    ))


@pytest.mark.parametrize("fault", ["missing_owner", "duplicate_owner", "target", "version", "mask", "percussion"])
def test_incompatible_or_target_bearing_graphs_fail_closed(fault: str) -> None:
    graph = build_raw_graph(_piece())
    edge_type = ("track", "contains_note", "note")
    if fault in {"missing_owner", "duplicate_owner"}:
        edge = graph[edge_type].edge_index
        edge = edge[:, 1:] if fault == "missing_owner" else torch.cat((edge, edge[:, :1]), dim=1)
        graph[edge_type].edge_index = edge
        graph[("note", "belongs_to_track", "track")].edge_index = edge.flip(0)
    elif fault == "target":
        graph["note"].target = torch.ones(graph["note"].num_nodes)
    elif fault == "version":
        graph.feature_registry_version = "unsupported"
    elif fault == "mask":
        graph["note"].x_cat_available[0, 0] = False
    else:
        graph["track"].x_cat[0, graph["track"].cat_feature_names.index("is_percussion")] = 1
    with pytest.raises(AnalysisGNNTranspositionError, match="complete_"):
        transpose_complete_raw_graph_view(graph, transform=DirectedTransposition(0, 0))


def test_views_are_detached_even_when_continuous_inputs_have_autograd_history() -> None:
    graph = build_raw_graph(_piece())
    leaf = graph["note"].x_cont.clone().requires_grad_()
    graph["note"].x_cont = leaf * 1.0
    view = transpose_complete_raw_graph_view(graph, transform=DirectedTransposition(2, 2))
    assert not view["note"].x_cont.requires_grad
    assert view["note"].x_cont.grad_fn is None
    assert graph["note"].x_cont.requires_grad
    _assert_exact_graph(view, _oracle(_piece(), 2))
