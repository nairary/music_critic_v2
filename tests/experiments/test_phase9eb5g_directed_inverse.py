from __future__ import annotations

import pytest

from music_critic.experiments.analysisgnn.corrected_training import (
    build_source_free_fixture,
)
from music_critic.experiments.analysisgnn.directed_transposition_diagnostics import (
    source_free_directed_regression,
)
from music_critic.experiments.analysisgnn.transposition import (
    AnalysisGNNTranspositionError,
    DirectedTransposition,
    SHIFT_PCS,
    SIGNED_BY_SHIFT_PC,
    canonical_directed_transposition,
    transpose_raw_graph_view,
    transpose_raw_graph_view_directed,
    valid_directed_transposition_for_midi,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    _graph_differences,
)


def test_tritone_forward_is_plus_six_and_directed_inverse_is_minus_six() -> None:
    forward = canonical_directed_transposition(6)
    inverse = forward.inverse()
    assert (forward.shift_pc, forward.signed_semitones) == (6, 6)
    assert (inverse.shift_pc, inverse.signed_semitones) == (6, -6)
    assert SIGNED_BY_SHIFT_PC[6] == 6


def test_all_twelve_directed_pairs_round_trip_raw_graph() -> None:
    batch, _sidecar = build_source_free_fixture()
    graph = batch.raw_graph_batch.to_data_list()[0]
    for shift in SHIFT_PCS:
        forward = canonical_directed_transposition(shift)
        shifted = transpose_raw_graph_view_directed(graph, transform=forward)
        restored = transpose_raw_graph_view_directed(
            shifted, transform=forward.inverse()
        )
        assert _graph_differences(graph, restored) == ()


def test_directed_midi_boundaries_are_checked_without_wrap_clip_or_fold() -> None:
    plus_six = canonical_directed_transposition(6)
    minus_six = plus_six.inverse()
    assert valid_directed_transposition_for_midi([0], plus_six)
    assert not valid_directed_transposition_for_midi([127], plus_six)
    assert valid_directed_transposition_for_midi([127], minus_six)
    assert not valid_directed_transposition_for_midi([0], minus_six)


@pytest.mark.parametrize(
    ("shift_pc", "signed", "category"),
    [
        (6, -5, "directed_identity_mismatch"),
        (12, 12, "directed_shift_pc_out_of_range"),
        (0, True, "directed_signed_semitones_invalid"),
    ],
)
def test_incompatible_directed_identity_fails_closed(
    shift_pc: int, signed: int, category: str
) -> None:
    with pytest.raises(AnalysisGNNTranspositionError, match=category) as caught:
        DirectedTransposition(shift_pc, signed)
    assert caught.value.category == f"analysisgnn.transposition.{category}"


def test_old_public_forward_and_directed_forward_are_identical() -> None:
    batch, _sidecar = build_source_free_fixture()
    graph = batch.raw_graph_batch.to_data_list()[0]
    for shift in SHIFT_PCS:
        old = transpose_raw_graph_view(graph, shift_pc=shift)
        new = transpose_raw_graph_view_directed(
            graph, transform=canonical_directed_transposition(shift)
        )
        assert _graph_differences(old, new) == ()
    assert _graph_differences(graph, transpose_raw_graph_view(graph, shift_pc=0)) == ()


def test_source_free_directed_regression_passes_every_shift() -> None:
    row = source_free_directed_regression()
    assert row["shift_count"] == 12
    assert row["canonical_forward_identical"] is True
    assert row["raw_round_trip_failure_count"] == 0
    assert row["target_round_trip_failure_count"] == 0
    assert row["identity_exact"] is True
